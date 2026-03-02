"""
CAPI 推論核心模組

提供 CAPI 面板異常檢測的完整推論流程：
1. 圖片載入與 Otsu 去背景
2. 排除區域識別（MARK、右下角機構）
3. 512x512 切塊與座標追蹤
4. PatchCore 模型推論
5. 異常結果匯總與座標轉換

使用方式:
    from capi_inference import CAPIInferencer
    from capi_config import CAPIConfig
    
    config = CAPIConfig.from_yaml("configs/capi_3f.yaml")
    inferencer = CAPIInferencer(config, model_path="path/to/model")
    results = inferencer.process_panel("path/to/panel_folder")
"""

import os
# 設置環境變數以允許載入模型 (必須在 import anomalib 之前)
os.environ["TRUST_REMOTE_CODE"] = "1"

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from capi_config import CAPIConfig, ExclusionZone, BombDefect


@dataclass
class TileInfo:
    """切塊資訊"""
    tile_id: int
    x: int  # 切塊在原圖的 x 座標
    y: int  # 切塊在原圖的 y 座標
    width: int
    height: int
    image: np.ndarray = field(repr=False)
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # 遮罩: 255=有效, 0=排除
    has_exclusion: bool = False  # 是否包含排除區域
    is_bottom_edge: bool = False # 是否為底部邊緣切塊
    is_suspected_dust_or_scratch: bool = False  # 是否疑似灰塵或刮痕 (透過 OMIT0000 檢查)
    omit_crop_image: Optional[np.ndarray] = field(default=None, repr=False)  # OMIT 圖片的對應裁切 (用於灰塵檢查)
    dust_mask: Optional[np.ndarray] = field(default=None, repr=False)
    dust_heatmap_iou: float = 0.0
    dust_bright_ratio: float = 0.0
    dust_detail_text: str = ""  # 灰塵判定詳細資訊
    dust_iou_debug_image: Optional[np.ndarray] = field(default=None, repr=False)  # IOU debug 可視化圖
    is_bomb: bool = False       # 是否為炸彈系統模擬缺陷
    bomb_defect_code: str = ""  # 匹配到的炸彈 Defect Code
    anomaly_peak_x: int = -1    # 熱力圖峰值 x (圖片座標, -1=未計算)
    anomaly_peak_y: int = -1    # 熱力圖峰值 y (圖片座標, -1=未計算)
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def valid_ratio(self) -> float:
        """有效區域比例 (0.0~1.0)"""
        if self.mask is None:
            return 1.0
        return np.sum(self.mask > 0) / self.mask.size


@dataclass
class ExclusionRegion:
    """實際排除區域（計算後的座標）"""
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    def contains_point(self, x: int, y: int) -> bool:
        """檢查點是否在排除區域內"""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def overlaps_rect(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """檢查矩形是否與排除區域重疊"""
        return not (x2 < self.x1 or x1 > self.x2 or y2 < self.y1 or y1 > self.y2)
    
    def overlap_ratio(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """計算矩形與排除區域的重疊比例"""
        # 計算交集
        ix1 = max(self.x1, x1)
        iy1 = max(self.y1, y1)
        ix2 = min(self.x2, x2)
        iy2 = min(self.y2, y2)
        
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        tile_area = (x2 - x1) * (y2 - y1)
        
        return intersection / tile_area if tile_area > 0 else 0.0


@dataclass
class AOIDefect:
    """AOI 缺陷資訊"""
    defect_code: str
    product_x: int
    product_y: int
    image_x: int
    image_y: int
    bounds: Tuple[int, int, int, int]  # (x1, y1, x2, y2) 標記框

@dataclass
class ImageResult:
    """單張圖片推論結果"""
    image_path: Path
    image_size: Tuple[int, int]  # (width, height)
    otsu_bounds: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    exclusion_regions: List[ExclusionRegion]
    tiles: List[TileInfo]
    excluded_tile_count: int
    processed_tile_count: int
    processing_time: float
    
    # 推論結果（由 PatchCore 填入）
    # (TileInfo, score, anomaly_map)
    anomaly_tiles: List[Tuple[TileInfo, float, Optional[np.ndarray]]] = field(default_factory=list)
    
    # AOI 缺陷結果
    aoi_defects: List[AOIDefect] = field(default_factory=list)
    
    # 裁切區域 (x1, y1, x2, y2) - 用於視覺化
    cropped_region: Optional[Tuple[int, int, int, int]] = None
    
    # 原始物件邊界 (用於 AOI 座標映射，避免重複讀取圖片)
    raw_bounds: Optional[Tuple[int, int, int, int]] = None
    
    # 推論耗時 (秒)
    inference_time: float = 0.0
    
    @property
    def total_tiles(self) -> int:
        return len(self.tiles)


class CAPIInferencer:
    """CAPI 推論器"""
    
    def __init__(
        self, 
        config: CAPIConfig, 
        model_path: Optional[str] = None,
        device: str = "auto",
        threshold: float = 0.5,
        base_dir: Optional[Path] = None,
    ):
        """
        初始化推論器
        
        Args:
            config: CAPI 配置
            model_path: PatchCore 模型路徑 (.xml 或 .pt)
            device: 運算裝置 ("auto", "cpu", "cuda")
            threshold: 異常判斷閾值
            base_dir: 基礎目錄（用於解析相對路徑）
        """
        self.config = config
        self.base_dir = base_dir or Path(__file__).parent
        self.mark_template = None
        self.model_path = Path(model_path) if model_path else None
        self.threshold = threshold
        self.inferencer = None
        
        # 決定運算裝置
        self.device = self._get_device(device)
        
        # 載入 MARK 模板
        self._load_mark_template()
        
        # 載入模型（如果有指定）
        if self.model_path:
            self._load_model()
    
    def _get_device(self, device: str) -> str:
        """取得運算裝置"""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def _load_model(self) -> None:
        """載入 PatchCore 模型 (支援 OpenVINO 和 PyTorch)"""
        if self.model_path is None or not self.model_path.exists():
            print(f"⚠️ 模型路徑無效: {self.model_path}")
            return
        
        print(f"載入模型: {self.model_path}")
        print(f"使用裝置: {self.device}")
        
        model_ext = self.model_path.suffix.lower()
        
        if model_ext == ".xml":
            # OpenVINO 格式
            from anomalib.deploy import OpenVINOInferencer
            print("📦 偵測到 OpenVINO 格式模型")
            self.inferencer = OpenVINOInferencer(
                path=str(self.model_path),
                device="CPU",  # OpenVINO 使用 CPU
            )
        elif model_ext in (".pt", ".pth", ".ckpt"):
            # PyTorch 格式
            from anomalib.deploy import TorchInferencer
            import pathlib
            import platform
            
            print("📦 偵測到 PyTorch 格式模型")
            
            # 解決 WindowsPath 權重檔在 Linux 載入報錯的 workaround
            original_windows_path = pathlib.WindowsPath
            if platform.system() != 'Windows':
                pathlib.WindowsPath = pathlib.PosixPath
                
            try:
                self.inferencer = TorchInferencer(
                    path=str(self.model_path),
                    device=self.device,
                )
            finally:
                if platform.system() != 'Windows':
                    pathlib.WindowsPath = original_windows_path
        else:
            print(f"⚠️ 未知模型格式: {model_ext}")
            return
        
        print("✅ 模型載入完成")
        
        # GPU Warm-up: 預先編譯 CUDA kernels，避免首次推論延遲
        if self.device != "cpu" and self.inferencer is not None:
            try:
                print("🔥 GPU Warm-up 中...")
                dummy = np.zeros((self.config.tile_size, self.config.tile_size, 3), dtype=np.uint8)
                self.inferencer.predict(dummy)
                print("✅ GPU Warm-up 完成")
            except Exception as e:
                print(f"⚠️ GPU Warm-up 失敗 (不影響推論): {e}")
    
    def _load_mark_template(self) -> None:
        """載入 MARK 模板"""
        template_path = self.config.get_mark_template_full_path(self.base_dir)
        if template_path.exists():
            self.mark_template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if self.mark_template is not None:
                print(f"✅ MARK 模板載入: {template_path.name} ({self.mark_template.shape})")
        else:
            print(f"⚠️ MARK 模板不存在: {template_path}")
    
    def _find_raw_object_bounds(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """找尋物件的原始邊界 (不含 Offset)"""
        img_height, img_width = image.shape[:2]
        
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((15, 15), np.uint8)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf
        
        MIN_AREA = 1000
        for contour in contours:
            if cv2.contourArea(contour) > MIN_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x + w, x_max)
                y_max = max(y + h, y_max)
        
        if x_min == np.inf:
            return 0, 0, img_width, img_height
            
        return int(x_min), int(y_min), int(x_max), int(y_max)

    def calculate_otsu_bounds(self, image: np.ndarray) -> Tuple[Tuple[int, int, int, int], Optional[int]]:
        """
        計算 Otsu 前景邊界
        Returns:
            (final_bounds, original_y2) - original_y2 是裁切前的底部 y 座標
        """
        img_height, img_width = image.shape[:2]
        
        # 取得原始物件邊界
        x_min, y_min, x_max, y_max = self._find_raw_object_bounds(image)
        
        offset = self.config.otsu_offset
        x_start = max(0, int(x_min) + offset)
        y_start = max(0, int(y_min) + offset)
        x_end = min(img_width, int(x_max) - offset)
        y_end = min(img_height, int(y_max) - offset)
        
        if x_start >= x_end or y_start >= y_end:
            x_start, y_start = 0, 0
            x_end, y_end = img_width, img_height
            
        # 應用底部裁切 (otsu_bottom_crop)
        original_y2 = None
        if self.config.otsu_bottom_crop > 0:
            h = y_end - y_start
            desired_height = max(self.config.tile_size, h - self.config.otsu_bottom_crop)
            final_height = min(h, desired_height)
            
            if final_height < h:
                original_y2 = y_end
                y_end = y_start + final_height
        
        return (x_start, y_start, x_end, y_end), original_y2
    
    def find_mark_region(self, image: np.ndarray) -> Optional[ExclusionRegion]:
        """使用模板匹配找到 MARK 區域"""
        if self.mark_template is None:
            return None
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        img_h, img_w = gray.shape[:2]
        min_y_position = int(img_h * self.config.mark_min_y_ratio)
        
        best_match = None
        best_val = 0
        
        template_h, template_w = self.mark_template.shape[:2]
        scales = [0.75, 1.0, 1.5, 2.0, 3.0]
        
        for scale in scales:
            scaled_w = int(template_w * scale)
            scaled_h = int(template_h * scale)
            
            if scaled_w <= 0 or scaled_h <= 0:
                continue
            if scaled_w >= gray.shape[1] or scaled_h >= gray.shape[0]:
                continue
            
            scaled_template = cv2.resize(self.mark_template, (scaled_w, scaled_h))
            
            try:
                result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_loc[1] < min_y_position:
                    continue
                
                if max_val > best_val:
                    best_val = max_val
                    best_match = (max_loc[0], max_loc[1], scaled_w, scaled_h)
            except:
                continue
        
        if best_match is None or best_val < self.config.mark_match_threshold:
            return None
        
        mx, my, mw, mh = best_match
        return ExclusionRegion(
            name="mark_area",
            x1=mx,
            y1=my,
            x2=mx + mw,
            y2=my + mh,
        )
    
    def calculate_exclusion_regions(
        self, 
        image: np.ndarray,
        otsu_bounds: Tuple[int, int, int, int],
        cached_mark: Optional[ExclusionRegion] = None,
    ) -> List[ExclusionRegion]:
        """計算所有排除區域
        
        Args:
            image: 原始圖片
            otsu_bounds: Otsu 邊界
            cached_mark: 快取的 MARK 區域（Panel 級共用），若提供則跳過模板匹配
        """
        # 如果啟用了底部裁切，且裁切量足夠大（例如 > 0），則假設機構和 MARK 都已被切除
        # 因此不再計算排外區域，避免 relative_bottom_right 誤標記到新的底部
        if self.config.otsu_bottom_crop > 0:
            return []
            
        regions = []
        otsu_x1, otsu_y1, otsu_x2, otsu_y2 = otsu_bounds
        
        for zone in self.config.get_enabled_exclusion_zones():
            if zone.type == "template_match" and zone.name == "mark_area":
                if cached_mark is not None:
                    # 使用快取的 MARK 位置（Panel 級共用）
                    regions.append(cached_mark)
                else:
                    # 未提供快取，進行模板匹配
                    mark_region = self.find_mark_region(image)
                    if mark_region:
                        regions.append(mark_region)
            
            elif zone.type == "relative_bottom_right":
                # 相對於 Otsu 邊界的右下角
                br_x1 = max(otsu_x1, otsu_x2 - zone.width)
                br_y1 = max(otsu_y1, otsu_y2 - zone.height)
                regions.append(ExclusionRegion(
                    name=zone.name,
                    x1=br_x1,
                    y1=br_y1,
                    x2=otsu_x2,
                    y2=otsu_y2,
                ))
        
        return regions
    
    def tile_image(
        self,
        image: np.ndarray,
        otsu_bounds: Tuple[int, int, int, int],
        exclusion_regions: List[ExclusionRegion],
        exclusion_threshold: float = 0.0,  # 重疊比例超過此值則跳過 (0.0 = 任何重疊都跳過)
    ) -> Tuple[List[TileInfo], int]:
        """
        將圖片切成 tile，完全跳過與排除區域重疊的 tile
        邊緣不足 512px 的區域會向前回推補齊
        
        Args:
            image: 原始圖片
            otsu_bounds: Otsu 邊界
            exclusion_regions: 排除區域列表
            exclusion_threshold: 重疊比例閾值，超過此值則跳過該 tile (預設 0.0 = 任何重疊都跳過)
            
        Returns:
            (有效 tiles, 被跳過的 tile 數量)
        """
        otsu_x1, otsu_y1, otsu_x2, otsu_y2 = otsu_bounds
        tile_size = self.config.tile_size
        stride = self.config.tile_stride
        
        # 計算 X 和 Y 軸的 tile 起始座標（包含邊緣補齊）
        def generate_tile_positions(start: int, end: int, size: int, step: int) -> List[int]:
            """生成 tile 座標，邊緣不足時回推補齊"""
            positions = []
            pos = start
            while pos + size <= end:
                positions.append(pos)
                pos += step
            
            # 如果最後一個 tile 沒有覆蓋到邊緣，增加一個邊緣 tile
            if positions:
                last_end = positions[-1] + size
                if last_end < end:
                    # 向前回推，讓最後一個 tile 剛好貼齊邊緣
                    edge_pos = end - size
                    if edge_pos > positions[-1]:  # 避免重複
                        positions.append(edge_pos)
            elif end - start >= size:
                # 如果區域剛好等於 tile 大小
                positions.append(start)
            
            return positions
        
        x_positions = generate_tile_positions(otsu_x1, otsu_x2, tile_size, stride)
        y_positions = generate_tile_positions(otsu_y1, otsu_y2, tile_size, stride)
        
        # 判斷底排 y 座標 (tile 底部邊緣接近 otsu_y2)
        bottom_y_threshold = otsu_y2 - tile_size  # 底排 tile 的起始 y 門檻
        
        tiles = []
        excluded_count = 0
        tile_id = 0
        
        for y in y_positions:
            for x in x_positions:
                tile_x2 = x + tile_size
                tile_y2 = y + tile_size
                
                # 檢查是否與任何排除區域重疊
                should_skip = False
                for region in exclusion_regions:
                    overlap = region.overlap_ratio(x, y, tile_x2, tile_y2)
                    if overlap > exclusion_threshold:
                        should_skip = True
                        break
                
                if should_skip:
                    excluded_count += 1
                    continue  # 完全跳過此 tile
                
                # 判斷是否為底排 tile
                is_bottom = (y >= bottom_y_threshold)
                
                # 擷取 tile 圖片
                tile_img = image[y:tile_y2, x:tile_x2].copy()
                
                tiles.append(TileInfo(
                    tile_id=tile_id,
                    x=x,
                    y=y,
                    width=tile_size,
                    height=tile_size,
                    image=tile_img,
                    mask=None,  # 不再使用遮罩
                    has_exclusion=False,  # 保留此欄位以免影響其他程式碼
                    is_bottom_edge=is_bottom,
                ))
                tile_id += 1
        
        return tiles, excluded_count
    
    def preprocess_image(self, image_path: Path, cached_mark: Optional[ExclusionRegion] = None) -> Optional[ImageResult]:
        """
        預處理圖片：Otsu + 排除區域 + 切塊
        
        Args:
            image_path: 圖片路徑
            cached_mark: 快取的 MARK 區域（Panel 級共用）
            
        Returns:
            ImageResult 或 None（如果載入失敗）
        """
        start_time = time.time()
        
        # 載入圖片 (保持原始深度，例如 8-bit 灰階)
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"⚠️ 無法載入: {image_path}")
            return None
        
        img_h, img_w = image.shape[:2]
        
        # 計算原始物件邊界 (用於 AOI 座標映射，只算一次)
        raw_bounds = self._find_raw_object_bounds(image)
        
        # Otsu 裁切
        otsu_bounds, original_y2 = self.calculate_otsu_bounds(image)
        
        # 記錄裁切區域
        cropped_region = None
        if original_y2 is not None:
            # (x1, y2_new, x2, y2_old)
            cropped_region = (otsu_bounds[0], otsu_bounds[3], otsu_bounds[2], original_y2)
        
        # 計算排除區域（使用快取的 MARK 位置）
        exclusion_regions = self.calculate_exclusion_regions(image, otsu_bounds, cached_mark=cached_mark)
        
        # 切塊
        tiles, excluded_count = self.tile_image(image, otsu_bounds, exclusion_regions)
        
        elapsed = time.time() - start_time
        
        return ImageResult(
            image_path=image_path,
            image_size=(img_w, img_h),
            otsu_bounds=otsu_bounds,
            cropped_region=cropped_region,
            exclusion_regions=exclusion_regions,
            tiles=tiles,
            excluded_tile_count=excluded_count,
            processed_tile_count=len(tiles),
            processing_time=elapsed,
            raw_bounds=raw_bounds,
        )
    
    def _apply_edge_margin(self, anomaly_map: np.ndarray, margin_px: int) -> np.ndarray:
        """
        對 anomaly_map 底部 margin_px 像素做線性漸層衰減 (1→0)
        用於過濾底部邊緣光影造成的假陽性
        
        Args:
            anomaly_map: 異常熱圖 (H, W)
            margin_px: 衰減區域高度 (像素)
            
        Returns:
            衰減後的 anomaly_map
        """
        h = anomaly_map.shape[0]
        if margin_px <= 0 or margin_px >= h:
            return anomaly_map
        
        result = anomaly_map.copy()
        # 線性漸層：從 1.0 (margin 頂端) 衰減到 0.0 (底部邊緣)
        # 改用平方衰減 (Quadric Decay)，讓抑制效果更強 (數值下降更快)
        linear = np.linspace(1.0, 0.0, margin_px).astype(np.float32)
        gradient = np.power(linear, 2)
        result[-margin_px:, :] *= gradient[:, None]
        return result
    
    def predict_tile(self, tile: TileInfo) -> Tuple[float, Optional[np.ndarray]]:
        """
        對單一 tile 進行推論
        
        Args:
            tile: TileInfo 物件
            
        Returns:
            (異常分數, 異常熱圖) - 如果有遮罩，會過濾排除區域的異常
        """
        if self.inferencer is None:
            raise RuntimeError("模型尚未載入")
        
        # 使用 numpy array 進行推論
        # 如果是灰階 (2D 或 1 channel)，轉為 BGR
        input_image = tile.image
        if len(input_image.shape) == 2 or (len(input_image.shape) == 3 and input_image.shape[2] == 1):
             if len(input_image.shape) == 2:
                 input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
             else:
                 input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

        predictions = self.inferencer.predict(input_image)
        
        # 取得分數
        pred_score = float(predictions.pred_score.item()) if hasattr(predictions.pred_score, 'item') else float(predictions.pred_score)
        
        # 取得熱圖（如果有的話）
        anomaly_map = None
        if hasattr(predictions, 'anomaly_map') and predictions.anomaly_map is not None:
            anomaly_map = predictions.anomaly_map.squeeze().cpu().numpy() if hasattr(predictions.anomaly_map, 'cpu') else predictions.anomaly_map.squeeze()
            
            # 如果有遮罩，將排除區域的熱圖值設為 0
            if tile.mask is not None:
                # 確保遮罩尺寸匹配
                if anomaly_map.shape != tile.mask.shape:
                    mask_resized = cv2.resize(tile.mask, (anomaly_map.shape[1], anomaly_map.shape[0]))
                else:
                    mask_resized = tile.mask
                # 將排除區域設為 0
                anomaly_map = anomaly_map * (mask_resized / 255.0)
            
            # 底部邊緣衰減：過濾光影假陽性
            edge_margin = self.config.edge_margin_px
            if edge_margin > 0:
                should_apply = tile.is_bottom_edge or (not self.config.edge_margin_bottom_only)
                if should_apply:
                    # 將 margin_px 按 anomaly_map 實際尺寸縮放
                    scale = anomaly_map.shape[0] / tile.height
                    scaled_margin = int(edge_margin * scale)
                    anomaly_map = self._apply_edge_margin(anomaly_map, scaled_margin)
        
        # 如果有遮罩或底部衰減，重新計算分數
        need_recalc = (tile.mask is not None) or (tile.is_bottom_edge and self.config.edge_margin_px > 0)
        if need_recalc and anomaly_map is not None:
            if tile.mask is not None:
                valid_mask = tile.mask > 0
                if np.any(valid_mask):
                    pred_score = float(np.max(anomaly_map))
            else:
                pred_score = float(np.max(anomaly_map))
        
        return pred_score, anomaly_map
    
    def run_inference(self, result: ImageResult, progress_callback=None) -> ImageResult:
        """
        對預處理結果執行推論
        
        Args:
            result: preprocess_image 的結果
            progress_callback: 進度回呼函數 (current, total)
            
        Returns:
            更新後的 ImageResult（包含異常 tile 資訊）
        """
        if self.inferencer is None:
            raise RuntimeError("模型尚未載入，請在初始化時指定 model_path")
        
        inference_start = time.time()
        anomaly_tiles = []
        total = len(result.tiles)
        
        for i, tile in enumerate(result.tiles):
            if progress_callback:
                progress_callback(i + 1, total)
            
            score, anomaly_map = self.predict_tile(tile)
            
            if score >= self.threshold:
                anomaly_tiles.append((tile, score, anomaly_map))
        
        # 更新結果
        result.anomaly_tiles = anomaly_tiles
        result.inference_time = time.time() - inference_start
        
        return result
    
    def get_anomaly_summary(self, result: ImageResult) -> Dict[str, Any]:
        """取得異常摘要"""
        if not result.anomaly_tiles:
            return {
                "is_anomaly": False,
                "anomaly_count": 0,
                "max_score": 0.0,
                "anomaly_positions": [],
            }
        
        scores = [score for _, score, _ in result.anomaly_tiles]
        positions = [(tile.x, tile.y, tile.width, tile.height) for tile, _, _ in result.anomaly_tiles]
        
        return {
            "is_anomaly": True,
            "anomaly_count": len(result.anomaly_tiles),
            "max_score": max(scores),
            "avg_score": sum(scores) / len(scores),
            "anomaly_positions": positions,
        }
    
    def visualize_preprocessing(
        self, 
        image_path: Path, 
        result: ImageResult,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """視覺化預處理結果"""
        image = cv2.imread(str(image_path))
        vis = image.copy()
        
        # Otsu 邊界（藍色）
        x1, y1, x2, y2 = result.otsu_bounds
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 10)
        cv2.putText(vis, "Otsu Bounds", (x1 + 10, y1 + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 4)

        # 顯示裁切區域（灰色斜線或半透明）
        if result.cropped_region:
            cx1, cy1, cx2, cy2 = result.cropped_region
            # 畫半透明紅色區域
            overlay = vis.copy()
            cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
            # 畫邊框
            cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (100, 100, 100), 5)
            # 文字
            text_y = cy1 + (cy2 - cy1) // 2
            cv2.putText(vis, "BOTTOM CROP", (cx1 + 20, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 4)
        
        # 排除區域視覺化已移除 - 根據使用者要求，排除區域不需要在 Overview 上標出
        # 被跳過的 tiles 也不會顯示在這裡，因為它們根本沒有被生成
        
        # Tile 網格：所有 tiles 都用綠色（排除區域的 tiles 已被完全跳過）
        for tile in result.tiles:
            cv2.rectangle(vis, (tile.x, tile.y), 
                         (tile.x + tile.width, tile.y + tile.height), 
                         (0, 255, 0), 3)  # 綠色
        
        # 圖片資訊
        info_text = f"Total Tiles: {result.processed_tile_count} | Excluded: {result.excluded_tile_count}"
        cv2.putText(vis, info_text, (x1 + 10, y2 - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
        
        if output_path:
            # 縮小後儲存
            scale = 0.3
            vis_small = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))
            cv2.imwrite(str(output_path), vis_small)
        
        return vis

    
    def check_dust_or_scratch_feature(self, image: np.ndarray) -> tuple:
        """
        進階灰塵/刮痕偵測 — 使用 CLAHE 增強 + Otsu 自適應閾值 + 形態學 + 面積篩選
        
        流程:
          1. CLAHE 局部對比增強（偵測微弱灰塵）
          2. Otsu 自適應二值化（自動判定最佳閾值）
          3. 形態學開運算去雜訊 + 膨脹延伸
          4. Connected Components 面積篩選
          5. 分析顆粒 vs 刮傷（寬高比判定）
        
        Args:
            image: OMIT 圖片裁切區域 (BGR 或灰階)
            
        Returns:
            (is_dust, dust_mask, bright_ratio, detail_text)
            - is_dust: 是否偵測到灰塵/刮痕
            - dust_mask: 灰塵區域遮罩 (uint8, 255=灰塵)
            - bright_ratio: 灰塵面積佔比
            - detail_text: 判定詳細說明
        """
        if image is None or image.size == 0:
            return False, None, 0.0, "No image"
            
        # 讀取配置參數
        fallback_threshold = self.config.dust_brightness_threshold
        area_min = self.config.dust_area_min
        area_max = self.config.dust_area_max
        extension = self.config.dust_extension
        
        # Step 1: 轉灰階
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 2: CLAHE 局部對比增強 — 強化微弱灰塵的可見度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Step 3: Otsu 自適應二值化
        # 先嘗試 Otsu，若背景過暗 (均值 < 10) 則用固定閾值
        mean_val = np.mean(enhanced)
        if mean_val < 10:
            # 幾乎全黑，Otsu 可能不穩定，使用固定閾值
            _, binary = cv2.threshold(enhanced, fallback_threshold, 255, cv2.THRESH_BINARY)
            used_threshold = fallback_threshold
        else:
            otsu_thresh, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            used_threshold = otsu_thresh
        
        # Step 3.5: 合理性檢查 — CLAHE+Otsu 有時會在暗背景邊緣產生大面積假陽性
        # 若前景佔比超過 5%，判定為不合理結果，改用自適應回退策略
        MAX_REASONABLE_RATIO = 0.10
        initial_ratio = float(np.sum(binary > 0)) / binary.size if binary.size > 0 else 0.0
        if initial_ratio > MAX_REASONABLE_RATIO:
            # 回退策略：取原始灰階圖的 99th percentile 作為自適應閾值
            # 確保只有真正的亮點被偵測到，而非背景雜訊
            p99 = float(np.percentile(gray, 99))
            adaptive_thr = max(fallback_threshold, p99)
            print(f"    ⚠️ CLAHE+Otsu 結果異常 (前景佔比 {initial_ratio:.2%} > {MAX_REASONABLE_RATIO:.0%})，回退至自適應閾值 {adaptive_thr:.0f} (p99={p99:.0f}, cfg={fallback_threshold})")
            _, binary = cv2.threshold(gray, int(adaptive_thr), 255, cv2.THRESH_BINARY)
            used_threshold = adaptive_thr
        
        # Step 4: 形態學處理
        # 開運算：去除小噪點
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
        
        # 膨脹：延伸灰塵區域（對應廠商「延伸」概念）
        if extension > 0:
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                      (extension * 2 + 1, extension * 2 + 1))
            binary = cv2.dilate(binary, dilate_kernel, iterations=1)
        
        # Step 5: Connected Components 面積篩選
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        dust_mask = np.zeros_like(binary)
        particle_count = 0
        scratch_count = 0
        total_dust_area = 0
        
        for i in range(1, num_labels):  # 跳過背景 (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 面積篩選
            if area < area_min or area > area_max:
                continue
            
            # 寫入灰塵遮罩
            dust_mask[labels == i] = 255
            total_dust_area += area
            
            # 形態分類：寬高比 > 5 → 刮傷，否則 → 顆粒
            aspect = max(w, h) / (min(w, h) + 1e-5)
            if aspect > 5:
                scratch_count += 1
            else:
                particle_count += 1
        
        # 計算亮點佔比
        bright_ratio = float(np.sum(dust_mask > 0)) / dust_mask.size if dust_mask.size > 0 else 0.0
        is_dust = (particle_count + scratch_count) > 0
        
        detail_text = (f"Thr:{used_threshold:.0f} P:{particle_count} S:{scratch_count} "
                       f"Area:{total_dust_area} Ratio:{bright_ratio:.4f}")
        
        return is_dust, dust_mask, bright_ratio, detail_text
    
    def check_omit_overexposure(self, omit_image: np.ndarray) -> tuple:
        """
        檢查 OMIT 圖片是否曝光過度
        
        過曝的 OMIT 圖片(整張很白很亮)無法用於灰塵檢測，
        需要記錄並標記，供工程機台追蹤改善。
        
        Args:
            omit_image: OMIT 圖片 (BGR 或灰階)
            
        Returns:
            (is_overexposed, mean_brightness, bright_ratio, detail_text)
        """
        if omit_image is None or omit_image.size == 0:
            return False, 0.0, 0.0, "No OMIT image"
        
        # 轉灰階
        if len(omit_image.shape) == 3:
            gray = cv2.cvtColor(omit_image, cv2.COLOR_BGR2GRAY)
        elif len(omit_image.shape) == 2:
            gray = omit_image
        else:
            gray = omit_image.reshape(omit_image.shape[0], omit_image.shape[1])
        
        mean_brightness = float(np.mean(gray))
        bright_ratio = float(np.sum(gray > 230)) / gray.size if gray.size > 0 else 0.0
        
        mean_thr = self.config.omit_overexposure_mean_threshold
        ratio_thr = self.config.omit_overexposure_ratio_threshold
        
        is_overexposed = (mean_brightness > mean_thr) and (bright_ratio > ratio_thr)
        
        detail_text = (f"Mean:{mean_brightness:.1f}(thr={mean_thr}) "
                       f"BrightRatio:{bright_ratio:.3f}(thr={ratio_thr})")
        
        return is_overexposed, mean_brightness, bright_ratio, detail_text
    
    def compute_dust_heatmap_iou(self, dust_mask: np.ndarray, 
                                  anomaly_map: np.ndarray,
                                  top_percent: float = 5.0) -> tuple:
        """
        計算灰塵遮罩與 Heatmap「最紅區域」的 IOU (Intersection over Union)
        
        使用 Percentile 方式：取 anomaly_map 中數值最高的前 X% 像素作為熱區，
        比舊的 max*ratio 更穩定、不受單一極端值影響。
        
        Args:
            dust_mask: 灰塵遮罩 (uint8, 255=灰塵)
            anomaly_map: Heatmap 異常圖 (float, 可含負值)
            top_percent: 取最高的前百分之幾作為「最紅區域」(建議 3~8)
            
        Returns:
            (iou, heatmap_binary) - IOU 值 (0.0~1.0), 二值化後的熱區遮罩
        """
        if dust_mask is None or anomaly_map is None:
            return 0.0, None
        
        # 預處理
        anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
        anomaly_map = np.maximum(anomaly_map, 0.0)  # 去除負值
        dust_mask = np.asarray(dust_mask, dtype=np.uint8)
        
        if np.max(anomaly_map) <= 0:
            return 0.0, None
        
        # === 核心：取「最紅的前 top_percent%」像素 ===
        positive_values = anomaly_map[anomaly_map > 0]
        if len(positive_values) == 0:
            return 0.0, None
        
        threshold = np.percentile(positive_values, 100 - top_percent)
        heat_bool = anomaly_map >= threshold
        
        # 產生二值化遮罩 (供可視化用)
        heatmap_binary = (heat_bool.astype(np.uint8)) * 255
        
        # 灰塵遮罩轉單通道
        if len(dust_mask.shape) == 3:
            dust_mask = cv2.cvtColor(dust_mask, cv2.COLOR_BGR2GRAY)
        
        # 尺寸匹配
        if dust_mask.shape != heat_bool.shape:
            dust_resized = cv2.resize(dust_mask,
                                      (heat_bool.shape[1], heat_bool.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
        else:
            dust_resized = dust_mask
        
        dust_bool = dust_resized > 0
        
        # IOU 計算
        intersection = np.count_nonzero(dust_bool & heat_bool)
        union = np.count_nonzero(dust_bool | heat_bool)
        
        iou = float(intersection / union) if union > 0 else 0.0
        return iou, heatmap_binary
    
    def generate_dust_iou_debug_image(
        self,
        tile_image: np.ndarray,
        anomaly_map: np.ndarray,
        dust_mask: np.ndarray,
        heatmap_binary: np.ndarray,
        iou: float,
        top_percent: float,
        is_dust: bool,
    ) -> np.ndarray:
        """
        產生灰塵 IOU 交叉驗證的 Debug 可視化圖
        
        顯示：
          左上: Heatmap 疊加原圖 (紅色=異常熱區)
          右上: 灰塵遮罩 (黃色=灰塵區域) 
          左下: 熱區二值化 (白色=top X% 最紅像素)
          右下: 重疊分析 (綠色=交集, 紅色=僅熱區, 藍色=僅灰塵)
        
        Args:
            tile_image: 原始 tile 圖片
            anomaly_map: 異常熱圖 (float)
            dust_mask: 灰塵遮罩 (uint8, 255=灰塵)
            heatmap_binary: 二值化熱區遮罩
            iou: 計算出的 IOU 值
            top_percent: 使用的百分位數
            is_dust: 最終判定是否為灰塵
            
        Returns:
            Debug 可視化圖 (BGR)
        """
        sz = 256  # 每個子圖大小
        
        # --- 準備基底圖 ---
        if len(tile_image.shape) == 2:
            base = cv2.cvtColor(tile_image, cv2.COLOR_GRAY2BGR)
        elif tile_image.shape[2] == 1:
            base = cv2.cvtColor(tile_image, cv2.COLOR_GRAY2BGR)
        else:
            base = tile_image.copy()
        base = cv2.resize(base, (sz, sz))
        
        # --- 左上: Heatmap Overlay ---
        if anomaly_map is not None:
            norm = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            norm = cv2.resize(norm, (sz, sz))
            heatmap_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            panel_tl = cv2.addWeighted(base, 0.5, heatmap_color, 0.5, 0)
        else:
            panel_tl = base.copy()
        cv2.putText(panel_tl, "Heatmap", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # --- 右上: 灰塵遮罩 ---
        panel_tr = base.copy()
        if dust_mask is not None:
            dm = dust_mask
            if len(dm.shape) == 3:
                dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
            dm = cv2.resize(dm, (sz, sz), interpolation=cv2.INTER_NEAREST)
            dust_overlay = np.zeros_like(panel_tr)
            dust_overlay[dm > 0] = (0, 255, 255)  # 黃色
            panel_tr = cv2.addWeighted(panel_tr, 0.6, dust_overlay, 0.4, 0)
        cv2.putText(panel_tr, "Dust Mask", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # --- 左下: 熱區二值化 ---
        panel_bl = np.zeros((sz, sz, 3), dtype=np.uint8)
        if heatmap_binary is not None:
            hb = cv2.resize(heatmap_binary, (sz, sz), interpolation=cv2.INTER_NEAREST)
            panel_bl[hb > 0] = (255, 255, 255)  # 白色 = 熱區
        cv2.putText(panel_bl, f"Top {top_percent:.0f}%", (5, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        
        # --- 右下: 重疊分析 ---
        panel_br = np.zeros((sz, sz, 3), dtype=np.uint8)
        if heatmap_binary is not None and dust_mask is not None:
            hb = cv2.resize(heatmap_binary, (sz, sz), interpolation=cv2.INTER_NEAREST)
            dm = dust_mask
            if len(dm.shape) == 3:
                dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
            dm = cv2.resize(dm, (sz, sz), interpolation=cv2.INTER_NEAREST)
            
            heat_only = (hb > 0) & (dm == 0)   # 僅熱區
            dust_only = (dm > 0) & (hb == 0)    # 僅灰塵
            overlap = (hb > 0) & (dm > 0)       # 交集
            
            panel_br[heat_only] = (0, 0, 255)     # 紅色 = 僅熱區
            panel_br[dust_only] = (255, 100, 0)   # 藍色 = 僅灰塵
            panel_br[overlap]   = (0, 255, 0)     # 綠色 = 交集
        
        verdict_color = (0, 200, 255) if is_dust else (0, 0, 255)
        verdict_text = "DUST" if is_dust else "REAL_NG"
        cv2.putText(panel_br, f"IOU:{iou:.3f} {verdict_text}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, verdict_color, 1)
        # 圖例
        cv2.putText(panel_br, "Green=Overlap", (5, sz - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(panel_br, "Red=Heat Blue=Dust", (5, sz - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # --- 組合 2x2 ---
        top_row = np.hstack([panel_tl, panel_tr])
        bottom_row = np.hstack([panel_bl, panel_br])
        debug_img = np.vstack([top_row, bottom_row])
        
        return debug_img

    @staticmethod
    def _select_latest_panel_images(image_files: List[Path]) -> List[Path]:
        """
        當面版資料夾存在重複投片（圖片數量超過上限）時，
        依每個圖片前綴（去除時間戳尾碼後）只保留「最新」的一張。

        命名規則假設: {前綴}_{HHMMSS}.{副檔名}
        例如: G0F00000_104441.tif → 前綴 G0F00000，時間戳 104441。

        排序優先順序：
          1. 【優先】檔名中的 HHMMSS 時間戳（跨 Windows/Linux 完全一致，
                     不受 rsync/cp 複製時 mtime 被保留的影響）
          2. 【Fallback】st_mtime（當檔名格式不符時使用）
        """
        from collections import defaultdict

        def _sort_key(f: Path):
            """回傳排序鍵：優先取檔名時間戳，否則用 st_mtime"""
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 6:
                # 檔名時間戳 (HHMMSS) 轉整數，數值越大越新
                return (1, int(parts[1]), 0.0)
            # fallback: mtime (跨平台一致，但可能受複製影響)
            return (0, 0, f.stat().st_mtime)

        prefix_map: Dict[str, List[Path]] = defaultdict(list)
        for f in image_files:
            stem = f.stem  # e.g. "G0F00000_104441" 或 "PINIGBI _104432"
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 6:
                prefix = parts[0]
            else:
                prefix = stem
            prefix_map[prefix].append(f)

        selected = []
        for prefix, files in prefix_map.items():
            latest = max(files, key=_sort_key)
            selected.append(latest)

        return sorted(selected)


    def _parse_defect_txt(self, defect_file: Path) -> Dict[str, List[Dict]]:
        """解析 Defect.txt"""

        defects_map = {}
        if not defect_file.exists():
            return defects_map
            
        try:
            with open(defect_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            if not content:
                return defects_map
                
            records = content.split(';')
            for record in records:
                record = record.strip()
                if not record:
                    continue
                parts = record.split(',')
                if len(parts) >= 4:
                    filename = parts[0].strip()
                    if filename not in defects_map:
                        defects_map[filename] = []
                    
                    defects_map[filename].append({
                        'defect_code': parts[1].strip(),
                        'x': int(parts[2].strip()),
                        'y': int(parts[3].strip())
                    })
        except Exception as e:
            print(f"解析 Defect.txt 失敗: {e}")
            
        return defects_map

    def _map_aoi_coords(self, px: int, py: int, raw_bounds: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """將產品座標 (1920x1080) 映射到圖片座標"""
        PRODUCT_WIDTH = 1920
        PRODUCT_HEIGHT = 1080
        
        x_start, y_start, x_end, y_end = raw_bounds
        
        # 計算產品在圖片中的實際尺寸
        product_img_width = x_end - x_start
        product_img_height = y_end - y_start
        
        # 計算縮放比例
        scale_x = product_img_width / PRODUCT_WIDTH
        scale_y = product_img_height / PRODUCT_HEIGHT
        
        # 轉換座標
        img_x = int(px * scale_x + x_start)
        img_y = int(py * scale_y + y_start)
        
        return img_x, img_y

    def _check_heatmap_line_shape(
        self,
        anomaly_map: np.ndarray,
        min_aspect_ratio: float = 3.0,
        top_percent: float = 10.0,
    ) -> Tuple[bool, float]:
        """
        檢查 anomaly_map 的熱區形態是否為線狀 (高長寬比)
        
        Args:
            anomaly_map: Heatmap 異常圖 (float)
            min_aspect_ratio: 最小長寬比閾值 (>= 此值判定為線狀)
            top_percent: 取前 X% 最熱像素進行形態分析
            
        Returns:
            (is_line, aspect_ratio) - 是否為線狀, 實際長寬比
        """
        if anomaly_map is None or anomaly_map.size == 0:
            return False, 0.0
        
        amap = np.asarray(anomaly_map, dtype=np.float32)
        amap = np.maximum(amap, 0.0)
        
        positive_values = amap[amap > 0]
        if len(positive_values) == 0:
            return False, 0.0
        
        # Percentile 二值化
        threshold = np.percentile(positive_values, 100 - top_percent)
        binary = (amap >= threshold).astype(np.uint8) * 255
        
        # 找輪廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, 0.0
        
        # 取最大輪廓
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 10:
            return False, 0.0
        
        # 最小外接矩形 → 計算長寬比
        rect = cv2.minAreaRect(largest)
        w, h = rect[1]
        if min(w, h) <= 0:
            return False, 0.0
        
        aspect_ratio = max(w, h) / min(w, h)
        is_line = aspect_ratio >= min_aspect_ratio
        
        return is_line, aspect_ratio

    def check_bomb_match(
        self,
        image_prefix: str,
        tile_center_x: int,
        tile_center_y: int,
        raw_bounds: Tuple[int, int, int, int],
        anomaly_map: Optional[np.ndarray] = None,
    ) -> Tuple[bool, str]:
        """
        檢查異常 tile 是否匹配炸彈系統的已知座標
        
        Args:
            image_prefix: 圖片檔名前綴 (e.g. "G0F00000")
            tile_center_x: 異常 tile 中心 x (圖片座標)
            tile_center_y: 異常 tile 中心 y (圖片座標)
            raw_bounds: 原始 Otsu 邊界 (用於座標轉換)
            anomaly_map: 該 tile 的 anomaly map (用於 line 型形態驗證)
            
        Returns:
            (is_bomb, defect_code) - 是否為炸彈, 對應的 Defect Code
        """
        tolerance = self.config.bomb_match_tolerance
        
        for bomb in self.config.bomb_defects:
            # 比對前綴 (支援帶時間戳的檔名, e.g. "G0F00000" 匹配 "G0F00000_031447")
            if not (image_prefix == bomb.image_prefix or 
                    image_prefix.startswith(bomb.image_prefix + "_")):
                continue
            
            if bomb.defect_type == "line" and len(bomb.coordinates) >= 2:
                # 豎線型: 將兩端座標轉換，判斷 tile 是否在緩衝帶內
                pt1 = bomb.coordinates[0]
                pt2 = bomb.coordinates[1]
                img_x1, img_y1 = self._map_aoi_coords(pt1[0], pt1[1], raw_bounds)
                img_x2, img_y2 = self._map_aoi_coords(pt2[0], pt2[1], raw_bounds)
                
                # 線段 x 範圍 ± tolerance (轉換到圖片座標的 tolerance)
                PRODUCT_WIDTH = 1920
                x_start, _, x_end, _ = raw_bounds
                scale_x = (x_end - x_start) / PRODUCT_WIDTH
                img_tolerance_x = int(tolerance * scale_x)
                
                min_x = min(img_x1, img_x2) - img_tolerance_x
                max_x = max(img_x1, img_x2) + img_tolerance_x
                min_y = min(img_y1, img_y2)
                max_y = max(img_y1, img_y2)
                
                if min_x <= tile_center_x <= max_x and min_y <= tile_center_y <= max_y:
                    # 額外驗證：heatmap 是否呈現線狀形態
                    if anomaly_map is not None:
                        is_line, aspect_ratio = self._check_heatmap_line_shape(
                            anomaly_map,
                            min_aspect_ratio=self.config.bomb_line_min_aspect_ratio,
                        )
                        if not is_line:
                            print(f"⚠️ BOMB line 位置匹配但熱力圖非線狀 (aspect_ratio={aspect_ratio:.2f} < {self.config.bomb_line_min_aspect_ratio})，跳過 {bomb.defect_code}")
                            continue
                        print(f"✅ BOMB line 形態驗證通過 (aspect_ratio={aspect_ratio:.2f})")
                    return True, bomb.defect_code
                    
            elif bomb.defect_type == "point":
                # 點型: 判斷 tile 中心是否在任一炸彈點座標 ± tolerance 範圍內
                PRODUCT_WIDTH = 1920
                PRODUCT_HEIGHT = 1080
                x_start, y_start, x_end, y_end = raw_bounds
                scale_x = (x_end - x_start) / PRODUCT_WIDTH
                scale_y = (y_end - y_start) / PRODUCT_HEIGHT
                img_tolerance_x = int(tolerance * scale_x)
                img_tolerance_y = int(tolerance * scale_y)
                
                for coord in bomb.coordinates:
                    img_bx, img_by = self._map_aoi_coords(coord[0], coord[1], raw_bounds)
                    if (abs(tile_center_x - img_bx) <= img_tolerance_x and
                        abs(tile_center_y - img_by) <= img_tolerance_y):
                        return True, bomb.defect_code
        
        return False, ""

    def process_panel(
        self, 
        panel_dir: Path, 
        progress_callback=None,
        cpu_workers: int = 4
    ) -> List[ImageResult]:
        """
        處理整個面板的圖片 (包含 PINIGBI 灰塵檢查 和 AOI Defect 整合)
        
        使用三階段平行處理：
          Phase 1: 多執行緒平行預處理 (imread + Otsu + tiling) — CPU bound, OpenCV 釋放 GIL
          Phase 2: 序列 GPU 推論 (predict_tile) — GPU bound
          Phase 3: 多執行緒平行後處理 (灰塵 IOU 計算) — CPU bound
        
        Args:
            panel_dir: 面板資料夾路徑
            progress_callback: 進度回呼
            cpu_workers: CPU 平行化的執行緒數量 (預設 4)
            
        Returns:
            (該面板所有圖片的推論結果, OMIT 視覺化圖片(可選))
        """
        image_files = sorted(
            list(panel_dir.glob("*.png")) + 
            list(panel_dir.glob("*.jpg")) + 
            list(panel_dir.glob("*.tif")) + 
            list(panel_dir.glob("*.tiff"))
        )
        results = []
        
        # === 重複投片檢查 ===
        max_imgs = self.config.max_images_per_panel
        is_duplicate = False
        if len(image_files) > max_imgs:
            is_duplicate = True
            original_count = len(image_files)
            image_files = self._select_latest_panel_images(image_files)
            print(
                f"⚠️ 重複投片偵測: {panel_dir.name} 共 {original_count} 張圖片 (上限 {max_imgs})，"
                f"依建立時間選出最新 {len(image_files)} 張繼續推論"
            )
            for f in image_files:
                print(f"   ✅ 選用: {f.name}")
        
        # 如果有的話，解析 Defect.txt
        defect_map = self._parse_defect_txt(panel_dir / "Defect.txt")
        
        # 1. 分離一般圖片和灰塵檢查用圖片 (支援 OMIT0000 和 PINIGBI 兩種格式)
        def is_dust_check_image(f):
            return f.stem.startswith("PINIGBI") or "OMIT0000" in f.name
        omit_files = [f for f in image_files if is_dust_check_image(f)]
        normal_files = [f for f in image_files if not is_dust_check_image(f)]
        
        # 載入 OMIT 圖片 (如果有)
        omit_image = None
        omit_overexposed = False
        omit_overexposure_info = ""
        if omit_files:
            omit_path = omit_files[0]
            # 載入 OMIT 圖片 (保持原始深度)
            omit_image = cv2.imread(str(omit_path), cv2.IMREAD_UNCHANGED)
            if omit_image is None:
                print(f"⚠️ 無法載入 OMIT 圖片: {omit_path}")
        
        # 過曝檢查 (在灰塵檢測之前)
        if omit_image is not None:
            omit_overexposed, oe_mean, oe_ratio, oe_detail = self.check_omit_overexposure(omit_image)
            omit_overexposure_info = oe_detail
            if omit_overexposed:
                print(f"⚠️ OMIT OVEREXPOSED [{omit_path.name}]: {oe_detail}")
                print(f"   → Dust detection DISABLED for this panel (unreliable due to overexposure)")
            else:
                print(f"✅ OMIT exposure OK [{omit_path.name}]: {oe_detail}")
        
        # 準備 OMIT 視覺化圖 (BGR)
        omit_vis = None
        if omit_image is not None:
             omit_vis = omit_image.copy()
             if len(omit_vis.shape) == 2:
                 omit_vis = cv2.cvtColor(omit_vis, cv2.COLOR_GRAY2BGR)
             elif len(omit_vis.shape) == 3 and omit_vis.shape[2] == 1:
                 omit_vis = cv2.cvtColor(omit_vis, cv2.COLOR_GRAY2BGR)
             
             # 在過曝的 OMIT 圖上標記警告
             if omit_overexposed:
                 h, w = omit_vis.shape[:2]
                 # 半透明紅色覆蓋
                 overlay = omit_vis.copy()
                 cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 200), -1)
                 cv2.addWeighted(overlay, 0.5, omit_vis, 0.5, 0, omit_vis)
                 # 警告文字
                 cv2.putText(omit_vis, "!! OVEREXPOSED !!", (20, 50),
                             cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
                 cv2.putText(omit_vis, oe_detail, (20, 100),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # === MARK 位置快取 (Panel 級共用) ===
        # 掃描所有圖片，找到第一個成功匹配的 MARK 位置
        cached_mark = None
        has_mark_zone = any(
            z.type == "template_match" and z.name == "mark_area" 
            for z in self.config.get_enabled_exclusion_zones()
        )
        
        if has_mark_zone and self.config.otsu_bottom_crop <= 0:
            # 只有在需要 MARK 排除且沒有底部裁切時才掃描
            files_to_scan = [f for f in normal_files if not self.config.should_skip_file(f.name)]
            for scan_path in files_to_scan:
                try:
                    scan_img = cv2.imread(str(scan_path), cv2.IMREAD_UNCHANGED)
                    if scan_img is not None:
                        mark_region = self.find_mark_region(scan_img)
                        if mark_region:
                            cached_mark = mark_region
                            print(f"✅ MARK 位置已找到 (來源: {scan_path.name}) → ({mark_region.x1}, {mark_region.y1})-({mark_region.x2}, {mark_region.y2})")
                            break
                except Exception as e:
                    print(f"⚠️ MARK 掃描失敗 ({scan_path.name}): {e}")
                    continue
            
            # 若全部失敗，使用 Fallback 預設位置
            if cached_mark is None and self.config.mark_fallback_position:
                pos = self.config.mark_fallback_position
                cached_mark = ExclusionRegion(
                    name="mark_area",
                    x1=pos['x'],
                    y1=pos['y'],
                    x2=pos['x'] + pos['width'],
                    y2=pos['y'] + pos['height'],
                )
                print(f"⚠️ MARK 模板匹配全部失敗，使用 Fallback 預設位置 → ({cached_mark.x1}, {cached_mark.y1})-({cached_mark.x2}, {cached_mark.y2})")
            elif cached_mark is None:
                print(f"❌ MARK 模板匹配全部失敗，且未設定 Fallback 位置")
        
        # 過濾出需要處理的檔案
        files_to_process = [f for f in normal_files if not self.config.should_skip_file(f.name)]
        for f in normal_files:
            if self.config.should_skip_file(f.name):
                print(f"⏭️ 跳過檔案 (設定): {f.name}")
        
        total_files = len(files_to_process)
        if total_files == 0:
            return results, omit_vis, omit_overexposed, omit_overexposure_info, False
        
        # 決定實際 worker 數量 (不超過檔案數量)
        actual_workers = min(cpu_workers, total_files)
        print(f"🔀 平行處理: {total_files} 張圖片, {actual_workers} 個 CPU 執行緒")
        
        # ================================================================
        # Phase 1: 多執行緒平行預處理 (imread + Otsu + tiling)
        # OpenCV 在 C 層釋放 GIL，多執行緒可獲得真正的平行加速
        # ================================================================
        def _preprocess_one(img_path: Path) -> Optional[ImageResult]:
            """單張圖片的預處理 (可安全在多執行緒中呼叫)"""
            result = self.preprocess_image(img_path, cached_mark=cached_mark)
            if result is None:
                return None
            
            # 整合 AOI Defect Data
            stem = img_path.stem
            if stem in defect_map and result.raw_bounds is not None:
                raw_bounds = result.raw_bounds
                img_w, img_h = result.image_size
                
                for d in defect_map[stem]:
                    img_x, img_y = self._map_aoi_coords(d['x'], d['y'], raw_bounds)
                    
                    # 定義 AOI 標記框 (約 50x50)
                    size = 50
                    x1 = max(0, img_x - size // 2)
                    y1 = max(0, img_y - size // 2)
                    x2 = min(img_w, img_x + size // 2)
                    y2 = min(img_h, img_y + size // 2)
                    
                    result.aoi_defects.append(AOIDefect(
                        defect_code=d['defect_code'],
                        product_x=d['x'], 
                        product_y=d['y'],
                        image_x=img_x,
                        image_y=img_y,
                        bounds=(x1, y1, x2, y2)
                    ))
            
            return result
        
        preprocess_start = time.time()
        preprocessed_results = []
        
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # 提交所有預處理任務，保持原始順序
            future_to_path = {}
            for img_path in files_to_process:
                future = executor.submit(_preprocess_one, img_path)
                future_to_path[future] = img_path
            
            # 按提交順序收集結果 (使用 dict 保持對應)
            results_map = {}
            for future in as_completed(future_to_path):
                img_path = future_to_path[future]
                try:
                    result = future.result()
                    if result is not None:
                        results_map[img_path] = result
                except Exception as e:
                    print(f"⚠️ 預處理失敗 ({img_path.name}): {e}")
            
            # 按原始檔案順序排列結果
            for img_path in files_to_process:
                if img_path in results_map:
                    preprocessed_results.append(results_map[img_path])
        
        preprocess_time = time.time() - preprocess_start
        print(f"⚡ Phase 1 完成: {len(preprocessed_results)} 張圖片預處理耗時 {preprocess_time:.2f}s (平行 {actual_workers} 執行緒)")
        
        # ================================================================
        # Phase 2: 序列 GPU 推論 (predict_tile)
        # PyTorch GPU 推論不適合跨執行緒平行化，保持序列執行
        # ================================================================
        inference_start = time.time()
        for i, result in enumerate(preprocessed_results):
            result = self.run_inference(result)
            preprocessed_results[i] = result
            
            if progress_callback:
                progress_callback(i + 1, len(preprocessed_results))
        
        inference_time = time.time() - inference_start
        print(f"🔥 Phase 2 完成: GPU 推論耗時 {inference_time:.2f}s")
        
        # ================================================================
        # Phase 3: 多執行緒平行後處理 (灰塵 IOU 交叉驗證)
        # ================================================================
        def _dust_check_one(result: ImageResult) -> ImageResult:
            """單張圖片的灰塵交叉驗證 (可安全在多執行緒中呼叫)"""
            img_path = result.image_path
            
            if result.anomaly_tiles and omit_image is not None and omit_overexposed:
                # OMIT 過曝：無法進行灰塵檢測，記錄但不判定
                for tile, score, anomaly_map in result.anomaly_tiles:
                    tile.dust_detail_text = f"OMIT_OVEREXPOSED ({omit_overexposure_info}) -> Cannot verify dust, treated as REAL_NG"
                    print(f"⚠️ {img_path.name} Tile@({tile.x},{tile.y}) Score:{score:.3f} → OMIT OVEREXPOSED, skip dust check")
            elif result.anomaly_tiles and omit_image is not None and not omit_overexposed:
                for tile, score, anomaly_map in result.anomaly_tiles:
                    # 在 OMIT 圖片上裁切相同區域
                    tx, ty, tw, th = tile.x, tile.y, tile.width, tile.height
                    
                    # 邊界檢查
                    oh, ow = omit_image.shape[:2]
                    if tx < ow and ty < oh:
                        x2 = min(tx + tw, ow)
                        y2 = min(ty + th, oh)
                        
                        omit_crop = omit_image[ty:y2, tx:x2]
                        tile.omit_crop_image = omit_crop.copy()
                        
                        # Step A: 進階灰塵偵測 (CLAHE + Otsu + 面積篩選)
                        is_dust, dust_mask, bright_ratio, detail_text = self.check_dust_or_scratch_feature(omit_crop)
                        tile.dust_mask = dust_mask
                        tile.dust_bright_ratio = bright_ratio
                        
                        # Step B: Heatmap IOU 交叉驗證 (Percentile-based)
                        iou = 0.0
                        heatmap_binary = None
                        top_pct = self.config.dust_heatmap_top_percent
                        if is_dust and anomaly_map is not None:
                            iou, heatmap_binary = self.compute_dust_heatmap_iou(
                                dust_mask, anomaly_map, top_percent=top_pct
                            )
                            tile.dust_heatmap_iou = iou
                            
                            # 判定：灰塵偵測到 且 IOU > 閾值 → 灰塵造成的偽陽性
                            if iou >= self.config.dust_heatmap_iou_threshold:
                                tile.is_suspected_dust_or_scratch = True
                                detail_text += f" IOU:{iou:.3f}>=IOU_THR -> DUST"
                            else:
                                detail_text += f" IOU:{iou:.3f}<IOU_THR -> REAL_NG"
                            
                            # 產生 Debug 可視化圖
                            try:
                                tile.dust_iou_debug_image = self.generate_dust_iou_debug_image(
                                    tile.image, anomaly_map, dust_mask,
                                    heatmap_binary, iou, top_pct,
                                    tile.is_suspected_dust_or_scratch,
                                )
                            except Exception as dbg_err:
                                print(f"⚠️ Debug image generation failed: {dbg_err}")
                        elif is_dust:
                            # 有灰塵但沒有 heatmap → 保守標記為灰塵
                            tile.is_suspected_dust_or_scratch = True
                            detail_text += " (no heatmap, marked as dust)"
                        else:
                            detail_text += " NO_DUST -> REAL_NG"
                        
                        tile.dust_detail_text = detail_text
                        
                        log_icon = "🧹" if tile.is_suspected_dust_or_scratch else "🔴"
                        print(f"{log_icon} {img_path.name} Tile@({tx},{ty}) → {detail_text}")
            
            return result
        
        postprocess_start = time.time()
        
        # 只對有異常且有 OMIT 的結果進行平行灰塵檢測
        needs_dust_check = [r for r in preprocessed_results if r.anomaly_tiles and omit_image is not None]
        
        if needs_dust_check:
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                dust_futures = {executor.submit(_dust_check_one, r): r for r in needs_dust_check}
                for future in as_completed(dust_futures):
                    try:
                        future.result()
                    except Exception as e:
                        r = dust_futures[future]
                        print(f"⚠️ 灰塵檢測失敗 ({r.image_path.name}): {e}")
        
        # 在 OMIT 總圖上畫框 (序列化執行以避免競爭)
        if omit_vis is not None:
            for result in preprocessed_results:
                if result.anomaly_tiles:
                    for tile, score, anomaly_map in result.anomaly_tiles:
                        tx, ty, tw, th = tile.x, tile.y, tile.width, tile.height
                        oh, ow = omit_vis.shape[:2]
                        if tx < ow and ty < oh:
                            x2 = min(tx + tw, ow)
                            y2 = min(ty + th, oh)
                            iou = tile.dust_heatmap_iou
                            if tile.is_suspected_dust_or_scratch:
                                box_color = (0, 165, 255)
                                label = f"DUST IOU:{iou:.2f}"
                            else:
                                box_color = (0, 0, 255)
                                label = f"REAL_NG IOU:{iou:.2f}"
                            cv2.rectangle(omit_vis, (tx, ty), (x2, y2), box_color, 5)
                            cv2.putText(omit_vis, f"{result.image_path.name}", (tx, ty - 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, box_color, 4)
                            cv2.putText(omit_vis, label, (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, box_color, 4)
        
        postprocess_time = time.time() - postprocess_start
        print(f"🧹 Phase 3 完成: 灰塵檢測後處理耗時 {postprocess_time:.2f}s")
        
        results = preprocessed_results
                
        # === 炸彈系統比對 ===
        if self.config.bomb_defects:
            for result in results:
                if result.anomaly_tiles and result.raw_bounds is not None:
                    img_prefix = result.image_path.stem
                    for tile, score, anomaly_map in result.anomaly_tiles:
                        # 計算熱力圖峰值位置 (更精確的缺陷位置)
                        if anomaly_map is not None and anomaly_map.size > 0:
                            try:
                                peak_local = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
                                tile.anomaly_peak_y = tile.y + int(peak_local[0])
                                tile.anomaly_peak_x = tile.x + int(peak_local[1])
                            except Exception:
                                tile.anomaly_peak_x = tile.x + tile.width // 2
                                tile.anomaly_peak_y = tile.y + tile.height // 2
                        else:
                            tile.anomaly_peak_x = tile.x + tile.width // 2
                            tile.anomaly_peak_y = tile.y + tile.height // 2
                        
                        # 用熱力圖峰值座標來比對炸彈
                        is_bomb, bomb_code = self.check_bomb_match(
                            img_prefix, tile.anomaly_peak_x, tile.anomaly_peak_y, result.raw_bounds,
                            anomaly_map=anomaly_map
                        )
                        if is_bomb:
                            tile.is_bomb = True
                            tile.bomb_defect_code = bomb_code
                            print(f"💣 {result.image_path.name} Tile@({tile.x},{tile.y}) Peak@({tile.anomaly_peak_x},{tile.anomaly_peak_y}) → BOMB match ({bomb_code})")

        total_panel_time = preprocess_time + inference_time + postprocess_time
        print(f"📊 Panel {panel_dir.name} 總計: 預處理 {preprocess_time:.2f}s + 推論 {inference_time:.2f}s + 後處理 {postprocess_time:.2f}s = {total_panel_time:.2f}s")
        
        return results, omit_vis, omit_overexposed, omit_overexposure_info, is_duplicate

    def visualize_inference_result(self, image_path: Path, result: ImageResult) -> np.ndarray:
        """視覺化推論結果（含異常標記 與 AOI 標記）"""
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        
        # 如果是灰階，轉為 BGR 以便畫上彩色標記
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        elif len(vis.shape) == 3 and vis.shape[2] == 1:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        # Otsu 邊界（藍色）
        x1, y1, x2, y2 = result.otsu_bounds
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 6)
        
        # 顯示裁切區域
        if result.cropped_region:
            cx1, cy1, cx2, cy2 = result.cropped_region
            # 畫半透明黑色
            overlay = vis.copy()
            cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
            # 虛線框 (OpenCV 不直接支援虛線，用實線代替)
            cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (100, 100, 100), 3)
            cv2.putText(vis, "CROPPED", (cx1 + 10, cy1 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        
        # AOI 缺陷標記已移除 (無意義)
        
        # 排除區域
        for region in result.exclusion_regions:
            cv2.rectangle(vis, (region.x1, region.y1), (region.x2, region.y2), (128, 128, 128), 4)
        
        # 正常 tiles（整齊的綠色網格線）
        # 收集所有唯一的水平和垂直座標，避免邊緣 tile 回推造成的雙線
        h_lines = set()  # 水平線 y 座標
        v_lines = set()  # 垂直線 x 座標
        otsu_x1, otsu_y1, otsu_x2, otsu_y2 = result.otsu_bounds
        for tile in result.tiles:
            # 對齊到最近的 tile step (通常是 512 的倍數)
            # 只取 Otsu 邊界內的線條
            if otsu_y1 <= tile.y <= otsu_y2:
                h_lines.add(tile.y)
            if otsu_y1 <= tile.y + tile.height <= otsu_y2:
                h_lines.add(tile.y + tile.height)
            if otsu_x1 <= tile.x <= otsu_x2:
                v_lines.add(tile.x)
            if otsu_x1 <= tile.x + tile.width <= otsu_x2:
                v_lines.add(tile.x + tile.width)
        
        # 畫水平線 (在 Otsu x 範圍內)
        for y in sorted(h_lines):
            cv2.line(vis, (otsu_x1, y), (otsu_x2, y), (0, 255, 0), 1, cv2.LINE_AA)
        # 畫垂直線 (在 Otsu y 範圍內)
        for x in sorted(v_lines):
            cv2.line(vis, (x, otsu_y1), (x, otsu_y2), (0, 255, 0), 1, cv2.LINE_AA)
        
        # 異常 tiles（紅色粗框 + 分數）
        if result.anomaly_tiles:
            print(f"🔍 發現異常 tiles，共 {len(result.anomaly_tiles)} 個")
        for tile, score, _ in result.anomaly_tiles:
            color = (0, 0, 255)  # 紅色 (預設異常)
            label = f"{score:.2f}"
            thickness = 6
            
            # 炸彈 tile：洋紅色
            if tile.is_bomb:
                color = (255, 0, 255)  # 洋紅色 (BGR)
                label = f"{score:.2f} BOMB({tile.bomb_defect_code})"
            # 如果是疑似灰塵/刮痕，改為橘色
            elif tile.is_suspected_dust_or_scratch:
                color = (0, 165, 255)  # 橘色 (BGR: 0, 165, 255)
                label = f"{score:.2f} DUST(IOU:{tile.dust_heatmap_iou:.2f})"
            elif tile.dust_heatmap_iou > 0:
                # 有 OMIT 分析結果但非灰塵，顯示 IOU
                label = f"{score:.2f} NG(IOU:{tile.dust_heatmap_iou:.2f})"
            
            # 座標邊界檢查
            h, w = vis.shape[:2]
            x1_clip = max(0, min(tile.x, w-1))
            y1_clip = max(0, min(tile.y, h-1))
            x2_clip = max(0, min(tile.x + tile.width, w))
            y2_clip = max(0, min(tile.y + tile.height, h))
            
            try:
                cv2.rectangle(vis, (x1_clip, y1_clip), (x2_clip, y2_clip), color, thickness)
                
                # 在 tile 中心畫一個大圓作為額外標記
                center_x = (x1_clip + x2_clip) // 2
                center_y = (y1_clip + y2_clip) // 2
                cv2.circle(vis, (center_x, center_y), 80, color, 10)
                
                # 文字位置也要在邊界內
                text_x = max(10, min(x1_clip + 10, w - 200))
                text_y = max(50, min(y1_clip + 50, h - 10))
                cv2.putText(vis, label, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            except Exception as e:
                print(f"❌ 繪製 Tile {tile.tile_id} 失敗: {e}, 座標: ({x1_clip},{y1_clip})->({x2_clip},{y2_clip})")
        
        # 結果標籤
        status = "NG" if result.anomaly_tiles else "OK"
        # 檢查是否所有異常都是疑似灰塵
        all_dust = False
        all_bomb = False
        if result.anomaly_tiles:
            non_dust = [t for t in result.anomaly_tiles if not t[0].is_suspected_dust_or_scratch or t[0].is_bomb]
            all_dust = all(t[0].is_suspected_dust_or_scratch and not t[0].is_bomb for t in result.anomaly_tiles)
            all_bomb = non_dust and all(t[0].is_bomb for t in non_dust)
            if all_dust:
                status = "NG (Dust?)"
            elif all_bomb:
                status = "BOMB"
                
        if not result.anomaly_tiles:
            color = (0, 255, 0)
        elif all_bomb:
            color = (255, 0, 255)  # 洋紅色
        elif all_dust:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)
        
        cv2.putText(vis, status, (x1 + 20, y1 + 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, color, 10)
        
        return vis
    
    def generate_bomb_diagram(self, image_path: Path, result: ImageResult) -> np.ndarray:
        """
        生成炸彈座標位置示意圖
        在原始圖片上疊加：
        1. 已設定的炸彈座標位置 (洋紅色十字)
        2. AD 偵測到的異常 tile 位置 (青色方框)
        3. 匹配連線 + 距離標示
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        elif len(vis.shape) == 3 and vis.shape[2] == 1:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        # 不暗化，直接使用原始圖片
        
        raw_bounds = result.raw_bounds
        if raw_bounds is None:
            return vis
        
        img_prefix = image_path.stem
        BOMB_COLOR = (255, 0, 255)    # 洋紅色 = 炸彈設定座標
        AD_COLOR = (255, 255, 0)      # 青色 = AD 偵測到的 tile
        MATCH_LINE_COLOR = (0, 255, 0) # 綠色 = 匹配連線
        
        # 計算 tolerance (與 check_bomb_match 一致)
        tolerance = self.config.bomb_match_tolerance
        PRODUCT_WIDTH = 1920
        PRODUCT_HEIGHT = 1080
        x_start, y_start, x_end, y_end = raw_bounds
        scale_x = (x_end - x_start) / PRODUCT_WIDTH
        scale_y = (y_end - y_start) / PRODUCT_HEIGHT
        img_tolerance_x = int(tolerance * scale_x)
        img_tolerance_y = int(tolerance * scale_y)
        
        # === 1. 繪製所有 AD 偵測到的異常 tile (青色框 + 峰值點) ===
        ad_tiles_info = []  # [(peak_x, peak_y, tile, score)]
        if result.anomaly_tiles:
            for tile, score, _ in result.anomaly_tiles:
                # 使用熱力圖峰值位置 (更精確)
                if tile.anomaly_peak_x >= 0 and tile.anomaly_peak_y >= 0:
                    px, py = tile.anomaly_peak_x, tile.anomaly_peak_y
                else:
                    px, py = tile.x + tile.width // 2, tile.y + tile.height // 2
                ad_tiles_info.append((px, py, tile, score))
                
                # 畫 tile 框 (半透明)
                cv2.rectangle(vis, (tile.x, tile.y), 
                              (tile.x + tile.width, tile.y + tile.height), 
                              AD_COLOR, 3)
                # 畫峰值點 (實心圓 = 精確缺陷位置)
                cv2.circle(vis, (px, py), 20, AD_COLOR, -1)
                # 從峰值到 tile 框畫十字準星
                cv2.line(vis, (px - 40, py), (px + 40, py), (0, 0, 255), 2)
                cv2.line(vis, (px, py - 40), (px, py + 40), (0, 0, 255), 2)
                
                # 標籤
                label = f"AD: {score:.2f}"
                if tile.is_bomb:
                    label += f" [BOMB:{tile.bomb_defect_code}]"
                elif tile.is_suspected_dust_or_scratch:
                    label += " [DUST]"
                cv2.putText(vis, label, (tile.x + 5, tile.y + tile.height - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, AD_COLOR, 2)
        
        # === 2. 繪製炸彈設定座標 (洋紅色十字) + 匹配連線 ===
        bomb_count = 0
        matched_count = 0
        
        for bomb in self.config.bomb_defects:
            if not (img_prefix == bomb.image_prefix or
                    img_prefix.startswith(bomb.image_prefix + "_")):
                continue
            
            if bomb.defect_type == "line" and len(bomb.coordinates) >= 2:
                pt1 = bomb.coordinates[0]
                pt2 = bomb.coordinates[1]
                img_x1, img_y1 = self._map_aoi_coords(pt1[0], pt1[1], raw_bounds)
                img_x2, img_y2 = self._map_aoi_coords(pt2[0], pt2[1], raw_bounds)
                
                cv2.line(vis, (img_x1, img_y1), (img_x2, img_y2), BOMB_COLOR, 6)
                mid_x = (img_x1 + img_x2) // 2
                mid_y = (img_y1 + img_y2) // 2
                label = f"BOMB LINE ({bomb.defect_code})"
                cv2.putText(vis, label, (mid_x + 20, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, BOMB_COLOR, 3)
                bomb_count += 1
                
            elif bomb.defect_type == "point":
                for idx, coord in enumerate(bomb.coordinates):
                    img_bx, img_by = self._map_aoi_coords(coord[0], coord[1], raw_bounds)
                    bomb_count += 1
                    
                    # 找最近的 AD tile 並計算距離
                    nearest_dist = float('inf')
                    nearest_ad = None
                    is_matched = False
                    for acx, acy, atile, ascore in ad_tiles_info:
                        dx = abs(img_bx - acx)
                        dy = abs(img_by - acy)
                        dist = (dx**2 + dy**2) ** 0.5
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_ad = (acx, acy)
                        # 使用與 check_bomb_match 相同的容忍度
                        if dx <= img_tolerance_x and dy <= img_tolerance_y:
                            is_matched = True
                    
                    if is_matched:
                        matched_count += 1
                    
                    pt_color = MATCH_LINE_COLOR if is_matched else BOMB_COLOR
                    
                    # 畫十字 + 圓
                    size = 60
                    cv2.circle(vis, (img_bx, img_by), size, pt_color, 5)
                    cv2.line(vis, (img_bx - size, img_by), (img_bx + size, img_by), pt_color, 3)
                    cv2.line(vis, (img_bx, img_by - size), (img_bx, img_by + size), pt_color, 3)
                    
                    # 座標標籤 (產品座標)
                    label = f"#{idx+1} ({coord[0]},{coord[1]})"
                    status_txt = "MATCHED" if is_matched else "NOT DETECTED"
                    cv2.putText(vis, label, (img_bx + size + 10, img_by - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, pt_color, 3)
                    cv2.putText(vis, status_txt, (img_bx + size + 10, img_by + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, pt_color, 2)
                    
                    # 畫到最近 AD tile 的連線和距離
                    if nearest_ad is not None:
                        line_color = MATCH_LINE_COLOR if is_matched else (128, 128, 128)
                        cv2.line(vis, (img_bx, img_by), nearest_ad, line_color, 2, cv2.LINE_AA)
                        mid_lx = (img_bx + nearest_ad[0]) // 2
                        mid_ly = (img_by + nearest_ad[1]) // 2
                        dist_label = f"d={nearest_dist:.0f}px"
                        cv2.putText(vis, dist_label, (mid_lx + 10, mid_ly - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)
        
        # === 3. 標題列 ===
        header_h = 90
        cv2.rectangle(vis, (0, 0), (vis.shape[1], header_h), (0, 0, 0), -1)
        title = f"BOMB Diagram: {image_path.stem} | Bombs: {bomb_count} | AD Tiles: {len(ad_tiles_info)} | Matched: {matched_count}"
        cv2.putText(vis, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, BOMB_COLOR, 3)
        tol_text = f"Tolerance: {tolerance}px (product) -> {img_tolerance_x}x{img_tolerance_y}px (image)"
        cv2.putText(vis, tol_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        
        # === 4. 圖例列 ===
        legend_h = 50
        legend_y0 = vis.shape[0] - legend_h
        cv2.rectangle(vis, (0, legend_y0), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
        ly = legend_y0 + 30
        # 炸彈座標
        cv2.circle(vis, (30, ly), 12, BOMB_COLOR, -1)
        cv2.putText(vis, "Bomb Config", (55, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BOMB_COLOR, 2)
        # AD tile
        cv2.rectangle(vis, (340, ly - 12), (364, ly + 12), AD_COLOR, 3)
        cv2.putText(vis, "AD Detected", (375, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, AD_COLOR, 2)
        # 匹配
        cv2.circle(vis, (660, ly), 12, MATCH_LINE_COLOR, -1)
        cv2.putText(vis, "Matched", (685, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, MATCH_LINE_COLOR, 2)
        
        return vis


def test_inferencer():
    """測試推論器"""
    from capi_config import CAPIConfig
    
    print("=" * 60)
    print("CAPI 推論器測試")
    print("=" * 60)
    
    # 載入配置
    config = CAPIConfig.from_yaml("configs/capi_3f.yaml")
    print(f"\n配置: {config}")
    
    # 建立推論器
    inferencer = CAPIInferencer(config)
    
    # 測試圖片
    test_folder = Path(r"D:\CAPI_3F\ok")
    panel_folders = sorted([f for f in test_folder.iterdir() if f.is_dir()])[:1]
    
    output_dir = Path("capi_inference_test")
    output_dir.mkdir(exist_ok=True)
    
    for panel_folder in panel_folders:
        print(f"\n=== 面板: {panel_folder.name} ===")
        
        image_files = list(panel_folder.glob("*.png"))[:2]
        
        for img_file in image_files:
            if img_file.stem.startswith("PINIGBI") or "OMIT0000" in img_file.name:
                continue
            
            result = inferencer.preprocess_image(img_file)
            if result:
                print(f"  {img_file.name}:")
                print(f"    - 尺寸: {result.image_size}")
                print(f"    - Otsu: {result.otsu_bounds}")
                print(f"    - 排除區域: {len(result.exclusion_regions)}")
                print(f"    - Tiles: {result.processed_tile_count} (排除: {result.excluded_tile_count})")
                print(f"    - 時間: {result.processing_time:.2f}s")
                
                # 視覺化
                output_path = output_dir / f"{panel_folder.name}_{img_file.stem}_tiles.jpg"
                inferencer.visualize_preprocessing(img_file, result, output_path)
    
    print(f"\n輸出目錄: {output_dir}")


if __name__ == "__main__":
    test_inferencer()
