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
    from capi_config import CAPIConfig, BombDefect
    
    config = CAPIConfig.from_yaml("configs/capi_3f.yaml")
    inferencer = CAPIInferencer(config, model_path="path/to/model")
    results = inferencer.process_panel("path/to/panel_folder")
"""

import os
# 設置環境變數以允許載入模型 (必須在 import anomalib 之前)
os.environ["TRUST_REMOTE_CODE"] = "1"
# 抑制 anomalib 棄用警告 (TorchInferencer legacy / TRUST_REMOTE_CODE)
import logging as _logging
_logging.getLogger("anomalib.deploy.inferencers.torch_inferencer").setLevel(_logging.ERROR)
# 抑制 dinov2 載入時 xFormers 不可用的 UserWarning
import warnings as _warnings
_warnings.filterwarnings("ignore", message=r".*xFormers is not available.*")

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import logging

# ── 舊版 anomalib 相容性修補 ─────────────────────────────
# 修補 1: PrecisionType stub
# 舊版 anomalib checkpoint 裡 pickle 序列化了 anomalib.PrecisionType enum，
# 新版已移除 → torch.load() AttributeError。注入 stub 讓反序列化成功。
import anomalib as _anomalib
if not hasattr(_anomalib, "PrecisionType"):
    import enum as _enum
    class _PrecisionType(str, _enum.Enum):
        FLOAT16 = "float16"
        FLOAT32 = "float32"
        BFLOAT16 = "bfloat16"
    _anomalib.PrecisionType = _PrecisionType
    logging.getLogger("capi.inference").info(
        "Injected PrecisionType stub into anomalib for legacy checkpoint compat"
    )

# 修補 2: TorchInferencer.predict → 強制 float32 輸入
# 舊版 checkpoint 序列化了 PrecisionType.FLOAT16，新版 anomalib 的
# TorchInferencer.predict() 會將輸入轉 fp16，但 backbone 權重仍為 float32
# → RuntimeError: Input type (HalfTensor) and weight type (FloatTensor) mismatch
# 解法: 包裝 model.forward，在 forward 前強制輸入轉 float32
try:
    from anomalib.deploy import TorchInferencer as _TI
    _orig_predict = _TI.predict

    def _fp32_predict(self, *args, **kwargs):
        _orig_fwd = self.model.forward
        def _force_fp32_fwd(batch, *a, **kw):
            # anomalib 使用 InferenceBatch dataclass，圖片在 .image 屬性
            if hasattr(batch, 'image') and isinstance(batch.image, torch.Tensor):
                batch.image = batch.image.float()
            elif isinstance(batch, torch.Tensor):
                batch = batch.float()
            return _orig_fwd(batch, *a, **kw)
        self.model.forward = _force_fp32_fwd
        try:
            return _orig_predict(self, *args, **kwargs)
        finally:
            self.model.forward = _orig_fwd

    _TI.predict = _fp32_predict
    logging.getLogger("capi.inference").info(
        "Patched TorchInferencer.predict → force float32 input"
    )
except Exception as _e:
    logging.getLogger("capi.inference").warning(
        "Failed to patch TorchInferencer.predict: %s", _e
    )

from capi_config import CAPIConfig, ExclusionZone, BombDefect
from capi_edge_cv import CVEdgeInspector, EdgeInspectionConfig, EdgeDefect, clamp_median_kernel, compute_fg_aware_diff
from scratch_classifier import ScratchClassifier, ScratchClassifierLoadError
from scratch_filter import ScratchFilter

logger = logging.getLogger("capi.inference")


# ── 產品解析度映射 ─────────────────────────────────────
# 機種名稱第六碼 → 產品解析度 (寬, 高)
# 此為程式碼級預設值，正式使用時由 capi_config.yaml 中的 model_resolution_map 覆蓋
MODEL_RESOLUTION_MAP = {
    'B': (1366, 768),
    'H': (1920, 1080),
    'J': (1920, 1200),
    'K': (2560, 1440),
    'G': (2560, 1600),
}

DEFAULT_PRODUCT_RESOLUTION = (1920, 1080)


def _anomaly_max_cc_area(anomaly_map: Optional[np.ndarray], peak_value: Optional[float] = None) -> int:
    """以 peak×0.5 二值化後取最大連通面積。peak_value 省略則用 anomaly_map 自身的 max。"""
    if anomaly_map is None or anomaly_map.size == 0:
        return 0
    peak = float(peak_value) if peak_value is not None else float(np.max(anomaly_map))
    if peak <= 0:
        return 0
    binary = (anomaly_map > (peak * 0.5)).astype(np.uint8) * 255
    num_labels, _, cc_stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels <= 1:
        return 0
    return int(cc_stats[1:, cv2.CC_STAT_AREA].max())


def resolve_product_resolution(model_id: str, resolution_map: Optional[Dict] = None) -> Tuple[int, int]:
    """
    依機種名稱第六碼推導產品解析度

    例如 "GN140JCAL010S" 第六碼 'J' → (1920, 1200)

    Args:
        model_id: 機種 ID (例如 "GN140JCAL010S")
        resolution_map: 可選的映射表 (來自 config.model_resolution_map)，
                        若為 None 則使用模組級預設 MODEL_RESOLUTION_MAP
    """
    if resolution_map is None:
        resolution_map = MODEL_RESOLUTION_MAP
    if model_id and len(model_id) >= 6:
        code = model_id[5].upper()
        res = resolution_map.get(code, None)
        if res is not None:
            return (int(res[0]), int(res[1])) if isinstance(res, (list, tuple)) else DEFAULT_PRODUCT_RESOLUTION
    return DEFAULT_PRODUCT_RESOLUTION


@dataclass
class TileInfo:
    """切塊資訊"""
    tile_id: int
    x: int  # 切塊在原圖的 x 座標
    y: int  # 切塊在原圖的 y 座標
    width: int
    height: int
    image: np.ndarray = field(repr=False)
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # 遮罩: 255=panel 內, 0=panel 外 (tile 完全在 polygon 內時為 None)
    has_exclusion: bool = False  # 是否包含排除區域
    is_bottom_edge: bool = False # 是否為底部邊緣切塊
    is_top_edge: bool = False    # 是否為頂部邊緣切塊
    is_left_edge: bool = False   # 是否為左側邊緣切塊
    is_right_edge: bool = False  # 是否為右側邊緣切塊
    is_suspected_dust_or_scratch: bool = False  # 是否疑似灰塵或刮痕 (透過 OMIT0000 檢查)
    omit_crop_image: Optional[np.ndarray] = field(default=None, repr=False)  # OMIT 圖片的對應裁切 (用於灰塵檢查)
    dust_mask: Optional[np.ndarray] = field(default=None, repr=False)
    dust_heatmap_iou: float = 0.0  # overall coverage (intersection / total dust area)
    dust_region_max_cov: float = 0.0  # per-region max coverage (用於實際判定)
    dust_region_details: Optional[list] = field(default=None, repr=False)  # per-region 判定詳情
    dust_heatmap_binary: Optional[np.ndarray] = field(default=None, repr=False)  # 二值化 heatmap
    dust_bright_ratio: float = 0.0
    dust_detail_text: str = ""  # 灰塵判定詳細資訊
    dust_iou_debug_image: Optional[np.ndarray] = field(default=None, repr=False)  # IOU debug 可視化圖
    dust_two_stage_features: Optional[list] = field(default=None, repr=False)  # 兩階段特徵點列表
    is_bomb: bool = False       # 是否為炸彈系統模擬缺陷
    bomb_defect_code: str = ""  # 匹配到的炸彈 Defect Code
    is_in_exclude_zone: bool = False  # 是否位於不檢測排除區域內
    anomaly_peak_x: int = -1    # 熱力圖峰值 x (圖片座標, -1=未計算)
    anomaly_peak_y: int = -1    # 熱力圖峰值 y (圖片座標, -1=未計算)
    is_aoi_coord_tile: bool = False  # 是否來自 AOI 機檢座標
    aoi_defect_code: str = ""        # AOI 異常代碼 (PCDK2, C1111, PTMD6)
    aoi_product_x: int = -1         # AOI 產品座標 X (-1=非 AOI 座標 tile)
    aoi_product_y: int = -1         # AOI 產品座標 Y (-1=非 AOI 座標 tile)
    is_bright_spot_detection: bool = False  # 是否為二值化亮點偵測（非 PatchCore）
    bright_spot_max_diff: int = 0           # B0F 偵測：最大局部差異值
    bright_spot_diff_threshold: int = 0     # B0F 偵測：使用的差異閾值
    bright_spot_area: int = 0               # B0F 偵測：偵測到的亮點面積 (px)
    bright_spot_min_area: int = 0           # B0F 偵測：使用的最小面積
    # Scratch classifier post-filter (over-review reduction)
    scratch_score: float = 0.0              # 0 = 未跑 classifier
    scratch_filtered: bool = False          # True = 被翻回 OK

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
class AOIReportDefect:
    """AOI 機台 NG 報告缺陷 (解析自 Report TXT)"""
    defect_code: str      # 異常代碼 (PCDK2, C1111, PTMD6)
    product_x: int        # 產品座標 X
    product_y: int        # 產品座標 Y
    image_prefix: str     # 圖片前綴 (W0F00000, B0F00000)

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

    # 面板 4 角 polygon (shape (4,2) float32，順序 TL/TR/BR/BL)
    # None 代表 polygon 偵測失敗或未啟用，下游應 fallback 回 axis-aligned bbox
    panel_polygon: Optional[np.ndarray] = field(default=None, repr=False)

    # 推論耗時 (秒)
    inference_time: float = 0.0

    # Scratch classifier post-filter stats
    scratch_filter_count: int = 0           # 此 image 中被翻 OK 的 tile 數

    # 客戶端傳送的炸彈資訊 (供繪圖使用)
    client_bomb_info: Optional[Dict[str, Any]] = None
    
    # CV 邊緣檢查結果
    edge_defects: List[EdgeDefect] = field(default_factory=list)
    
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
            model_path: PatchCore 模型路徑 (.xml 或 .pt)，作為 fallback
            device: 運算裝置 ("auto", "cpu", "cuda")
            threshold: 異常判斷閾值 (fallback，當 threshold_mapping 無對應時使用)
            base_dir: 基礎目錄（用於解析相對路徑）
        """
        self.config = config
        self.base_dir = base_dir or Path(__file__).parent
        self.mark_template = None
        self.model_path = Path(model_path) if model_path else None
        self.threshold = threshold
        self.inferencer = None  # 保留向後相容 (fallback 單一模型)
        
        # 多模型快取: {model_path_str: inferencer_object}
        self._inferencers: Dict[str, Any] = {}
        
        # 前綴 → 模型路徑映射 (從 config 讀取)
        self._model_mapping: Dict[str, Path] = {}
        for prefix, mpath in config.model_mapping.items():
            p = Path(mpath)
            if not p.is_absolute():
                p = self.base_dir / p
            self._model_mapping[prefix] = p
        
        # 前綴 → 閾值映射
        self._threshold_mapping: Dict[str, float] = dict(config.threshold_mapping)
        
        # 決定運算裝置
        self.device = self._get_device(device)
        
        # 載入 MARK 模板
        self._load_mark_template()
        
        # 初始化傳統 CV 邊緣檢測器 (之後會從 DB 中讀取設定後覆蓋)
        self.edge_inspector = CVEdgeInspector()

        # 載入模型
        if self._model_mapping:
            # 多模型模式：預載所有映射的模型
            print(f"🔀 多模型模式: 偵測到 {len(self._model_mapping)} 個前綴映射")
            for prefix, mpath in self._model_mapping.items():
                print(f"   {prefix} → {mpath}")
                try:
                    inf = self._load_model_from_path(mpath)
                    if inf:
                        self._inferencers[str(mpath)] = inf
                except Exception as e:
                    print(f"   ⚠️ 載入失敗: {e}")
            print(f"✅ 已載入 {len(self._inferencers)}/{len(self._model_mapping)} 個模型")
            # 設定 self.inferencer 為第一個載入成功的模型 (向後相容)
            if self._inferencers:
                self.inferencer = next(iter(self._inferencers.values()))
        elif self.model_path:
            # 單一模型模式 (向後相容)
            self._load_model()

        # Scratch classifier post-filter (lazy-loaded on first NG tile)
        self.scratch_filter: ScratchFilter | None = None
        self._scratch_load_failed = False

    def _get_scratch_filter(self):
        """Lazy-load ScratchFilter (first call only). Thread-safe via _gpu_lock
        (caller responsibility — called inside process_panel)."""
        if not getattr(self.config, "scratch_classifier_enabled", False):
            return None
        if self._scratch_load_failed:
            return None

        current_safety = float(getattr(self.config, "scratch_safety_multiplier", 1.1))
        if self.scratch_filter is not None and \
                abs(self.scratch_filter._safety - current_safety) > 1e-9:
            self.scratch_filter = ScratchFilter(self.scratch_filter._classifier,
                                                safety_multiplier=current_safety)
        if self.scratch_filter is not None:
            return self.scratch_filter

        bundle = getattr(self.config, "scratch_bundle_path", "")
        weights = getattr(self.config, "scratch_dinov2_weights_path", "")
        repo_path = getattr(self.config, "scratch_dinov2_repo_path", "")
        try:
            clf = ScratchClassifier(
                bundle_path=bundle,
                dinov2_weights_path=weights or None,
                dinov2_repo_path=repo_path or None,
                device=self.device,
            )
        except Exception as e:
            logger.error("ScratchClassifier load failed: %s", e, exc_info=True)
            self._scratch_load_failed = True
            return None
        self.scratch_filter = ScratchFilter(clf, safety_multiplier=current_safety)
        logger.info("ScratchClassifier filter ready (safety=%.2f, threshold=%.6f)",
                    current_safety, self.scratch_filter.effective_threshold)
        return self.scratch_filter

    def _get_device(self, device: str) -> str:
        """取得運算裝置"""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def _load_model_from_path(self, model_path: Path) -> Optional[Any]:
        """載入 PatchCore 模型並回傳 inferencer 物件 (支援 OpenVINO 和 PyTorch)
        
        Args:
            model_path: 模型檔案路徑
            
        Returns:
            inferencer 物件，載入失敗回傳 None
        """
        if model_path is None or not model_path.exists():
            print(f"⚠️ 模型路徑無效: {model_path}")
            return None
        
        print(f"載入模型: {model_path}")
        print(f"使用裝置: {self.device}")
        
        model_ext = model_path.suffix.lower()
        inferencer_obj = None
        
        if model_ext == ".xml":
            # OpenVINO 格式
            from anomalib.deploy import OpenVINOInferencer
            print("📦 偵測到 OpenVINO 格式模型")
            inferencer_obj = OpenVINOInferencer(
                path=str(model_path),
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
                inferencer_obj = TorchInferencer(
                    path=str(model_path),
                    device=self.device,
                )
            finally:
                if platform.system() != 'Windows':
                    pathlib.WindowsPath = original_windows_path
        else:
            print(f"⚠️ 未知模型格式: {model_ext}")
            return None
        
        print(f"✅ 模型載入完成: {model_path.name}")

        # ── 舊版 checkpoint fp16 precision 修補 ──────────────────
        # 舊版 anomalib checkpoint 可能序列化了 PrecisionType.FLOAT16，
        # 新版 TorchInferencer.predict() 會依此將輸入轉 fp16，
        # 但 backbone 權重仍為 float32 → RuntimeError: Input type mismatch
        # 修補: 強制將 precision 設回 float32，並確保模型全為 float32
        self._fix_legacy_precision(inferencer_obj)

        # fp16 KNN 優化: 將 memory bank 轉為 fp16，並 patch euclidean_dist 使用 tensor core
        self._optimize_model_fp16(inferencer_obj)

        # GPU Warm-up: 預先編譯 CUDA kernels，避免首次推論延遲
        if self.device != "cpu" and inferencer_obj is not None:
            try:
                print("🔥 GPU Warm-up 中...")
                dummy = np.zeros((self.config.tile_size, self.config.tile_size, 3), dtype=np.uint8)
                inferencer_obj.predict(dummy)
                print("✅ GPU Warm-up 完成")
            except Exception as e:
                print(f"⚠️ GPU Warm-up 失敗 (不影響推論): {e}")
        
        return inferencer_obj
    
    def _load_model(self) -> None:
        """載入 fallback 單一模型 (向後相容)"""
        result = self._load_model_from_path(self.model_path)
        if result is not None:
            self.inferencer = result
            self._inferencers[str(self.model_path)] = result
    
    def _get_image_prefix(self, filename: str) -> str:
        """從檔名中取得圖片前綴 (去除時間戳尾碼)
        
        例如: 'G0F00000_114438.tif' → 'G0F00000'
              'STANDARD.png' → 'STANDARD'
        """
        stem = Path(filename).stem
        # 如果含有底線，嘗試去掉時間戳部分
        if '_' in stem:
            prefix = stem.rsplit('_', 1)[0]
        else:
            prefix = stem
        return prefix
    
    def _get_inferencer_for_prefix(self, prefix: str) -> Optional[Any]:
        """根據圖片前綴取得對應的 inferencer (含 lazy loading)
        
        查找順序:
        1. model_mapping 中的對應模型
        2. fallback 到 self.inferencer (單一模型)
        """
        # 查找映射
        if prefix in self._model_mapping:
            model_path = self._model_mapping[prefix]
            path_key = str(model_path)
            
            # 快取命中
            if path_key in self._inferencers:
                return self._inferencers[path_key]
            
            # Lazy loading
            print(f"🔄 Lazy loading 模型: {prefix} → {model_path}")
            inf = self._load_model_from_path(model_path)
            if inf is not None:
                self._inferencers[path_key] = inf
                return inf
            else:
                print(f"⚠️ {prefix} 模型載入失敗，fallback 到預設模型")
        
        # Fallback: 使用預設模型
        return self.inferencer
    
    def _get_threshold_for_prefix(self, prefix: str) -> float:
        """根據圖片前綴取得對應的閾值"""
        return self._threshold_mapping.get(prefix, self.threshold)
    
    def _load_mark_template(self) -> None:
        """載入 MARK 模板"""
        template_path = self.config.get_mark_template_full_path(self.base_dir)
        if template_path.exists():
            self.mark_template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if self.mark_template is not None:
                print(f"✅ MARK 模板載入: {template_path.name} ({self.mark_template.shape})")
        else:
            print(f"⚠️ MARK 模板不存在: {template_path}")
    
    def _find_raw_object_bounds(
        self, image: np.ndarray
    ) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
        """找尋物件的原始邊界 (不含 Offset)

        Returns:
            ((x_min, y_min, x_max, y_max), binary_mask) — binary_mask 是 Otsu +
            morphology close 後的 uint8 前景圖（255=前景），供後續 polygon 偵測重用。
        """
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
            return (0, 0, img_width, img_height), closing

        return (int(x_min), int(y_min), int(x_max), int(y_max)), closing

    def _find_panel_polygon(
        self,
        binary_mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        在既有 Otsu binary mask 上用逐邊 polyfit 找 4 角 panel polygon。

        演算法:
          1. 逐邊 (top/bottom/left/right) 取樣前景邊緣點
          2. 每邊用 np.polyfit 擬合直線 + 3σ robust filter
          3. 4 線兩兩相交求 4 角 (TL/TR/BR/BL)
          4. 品質檢查 (面積、邊長、線近乎平行) — 任一失敗回傳 None

        Args:
            binary_mask: Otsu + morphology close 後的 uint8 前景圖 (255=前景)
            bbox: 粗略 bbox (x_min, y_min, x_max, y_max)，作為邊緣掃描範圍

        Returns:
            np.ndarray shape (4,2) float32，順序 [TL, TR, BR, BL]；
            偵測失敗或品質不足回傳 None。
        """
        # 常數 (hardcode，不入 config)
        EDGE_MARGIN = 20
        SAMPLE_STEP = 50
        OUTLIER_SIGMA = 3.0
        MIN_EDGE_LEN_RATIO = 1.0          # 相對 tile_size
        MIN_POLYGON_AREA_RATIO = 0.9      # 相對 bbox 面積
        MIN_SAMPLES_PER_EDGE = 5

        if binary_mask is None or binary_mask.size == 0:
            return None

        H, W = binary_mask.shape[:2]
        xmin, ymin, xmax, ymax = bbox
        if xmax - xmin < 2 * EDGE_MARGIN or ymax - ymin < 2 * EDGE_MARGIN:
            return None

        # --- Step 1: 逐邊掃描 ---
        tops, bots = [], []
        for x in range(xmin + EDGE_MARGIN, xmax - EDGE_MARGIN, SAMPLE_STEP):
            if x < 0 or x >= W:
                continue
            col = binary_mask[:, x]
            ys = np.where(col > 0)[0]
            if len(ys) > 0:
                tops.append((x, int(ys[0])))
                bots.append((x, int(ys[-1])))

        lefts, rights = [], []
        for y in range(ymin + EDGE_MARGIN, ymax - EDGE_MARGIN, SAMPLE_STEP):
            if y < 0 or y >= H:
                continue
            row = binary_mask[y, :]
            xs = np.where(row > 0)[0]
            if len(xs) > 0:
                lefts.append((int(xs[0]), y))
                rights.append((int(xs[-1]), y))

        if (len(tops) < MIN_SAMPLES_PER_EDGE or len(bots) < MIN_SAMPLES_PER_EDGE
                or len(lefts) < MIN_SAMPLES_PER_EDGE or len(rights) < MIN_SAMPLES_PER_EDGE):
            return None

        # --- Step 2: 每邊 polyfit + 3σ robust filter ---
        def fit_line_robust(pts, horizontal: bool) -> Optional[Tuple[float, float]]:
            """horizontal=True: 回傳 (a, b) 代表 y = a*x + b；否則代表 x = a*y + b"""
            arr = np.array(pts, dtype=float)
            if horizontal:
                ind = arr[:, 0]; dep = arr[:, 1]
            else:
                ind = arr[:, 1]; dep = arr[:, 0]
            try:
                a, b = np.polyfit(ind, dep, 1)
            except (np.linalg.LinAlgError, ValueError):
                return None
            residuals = dep - (a * ind + b)
            sigma = float(residuals.std())
            if sigma > 0:
                keep = np.abs(residuals) < OUTLIER_SIGMA * sigma
                if keep.sum() >= 3:
                    try:
                        a, b = np.polyfit(ind[keep], dep[keep], 1)
                    except (np.linalg.LinAlgError, ValueError):
                        pass
            return (float(a), float(b))

        top_line = fit_line_robust(tops, horizontal=True)
        bot_line = fit_line_robust(bots, horizontal=True)
        left_line = fit_line_robust(lefts, horizontal=False)
        right_line = fit_line_robust(rights, horizontal=False)
        if None in (top_line, bot_line, left_line, right_line):
            return None

        # --- Step 3: 4 線相交 ---
        def intersect_hv(h_line, v_line):
            # h: y = a_h*x + b_h ; v: x = a_v*y + b_v
            a_h, b_h = h_line
            a_v, b_v = v_line
            denom = 1.0 - a_h * a_v
            if abs(denom) < 1e-9:
                return None
            y = (a_h * b_v + b_h) / denom
            x = a_v * y + b_v
            return (x, y)

        TL = intersect_hv(top_line, left_line)
        TR = intersect_hv(top_line, right_line)
        BR = intersect_hv(bot_line, right_line)
        BL = intersect_hv(bot_line, left_line)
        if None in (TL, TR, BR, BL):
            return None

        polygon = np.array([TL, TR, BR, BL], dtype=np.float32)

        # --- Step 4: 品質檢查 ---
        # 4a. 所有角必須大致在 image 範圍內 (容忍 50 px 溢出)
        tol = 50
        if (polygon[:, 0].min() < -tol or polygon[:, 0].max() > W + tol or
                polygon[:, 1].min() < -tol or polygon[:, 1].max() > H + tol):
            return None

        # 4b. 邊長必須 > tile_size
        tile_size = self.config.tile_size
        min_edge_len = tile_size * MIN_EDGE_LEN_RATIO
        for i in range(4):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % 4]
            edge_len = float(np.linalg.norm(p2 - p1))
            if edge_len < min_edge_len:
                return None

        # 4c. polygon 面積必須 >= bbox 面積 * 0.9
        bbox_area = float((xmax - xmin) * (ymax - ymin))
        poly_area = float(cv2.contourArea(polygon))
        if bbox_area <= 0 or poly_area < bbox_area * MIN_POLYGON_AREA_RATIO:
            return None

        return polygon

    def calculate_otsu_bounds(
        self,
        image: np.ndarray,
        otsu_offset_override: Optional[int] = None,
        reference_raw_bounds: Optional[Tuple[int, int, int, int]] = None,
        reference_polygon: Optional[np.ndarray] = None,
    ) -> Tuple[Tuple[int, int, int, int], Optional[int], Optional[np.ndarray]]:
        """
        計算 Otsu 前景邊界與 panel polygon。

        Args:
            reference_raw_bounds: 參考用的原始邊界 (來自同資料夾的白圖)。
                                  黑圖 (B0F) OTSU 無法正確偵測邊界時使用。
            reference_polygon: 參考用的 panel polygon (來自同資料夾的白圖)，
                               與 reference_raw_bounds 同時使用於 B0F fallback。
        Returns:
            (final_bounds, original_y2, panel_polygon)
        """
        img_height, img_width = image.shape[:2]

        # 取得原始物件邊界與 binary mask
        if reference_raw_bounds is not None:
            x_min, y_min, x_max, y_max = reference_raw_bounds
            binary_mask = None
        else:
            (x_min, y_min, x_max, y_max), binary_mask = self._find_raw_object_bounds(image)

        offset = otsu_offset_override if otsu_offset_override is not None else self.config.otsu_offset
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

        bounds = (x_start, y_start, x_end, y_end)

        # 計算 panel polygon
        panel_polygon: Optional[np.ndarray] = None
        if self.config.enable_panel_polygon:
            if reference_polygon is not None:
                panel_polygon = reference_polygon.copy()
            elif binary_mask is not None:
                # 使用原始 (未內縮) bbox 做邊緣掃描
                raw_bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
                panel_polygon = self._find_panel_polygon(binary_mask, raw_bbox)

            # 對「新鮮偵測出的 polygon」做 offset 內縮 (reference_polygon 已由來源 caller
            # 處理過 offset，不可在這裡再縮一次，否則 B0F 路徑會雙重內縮)
            # 數學: 朝 centroid 沿對角線方向縮 offset px。對 axis-aligned 矩形而言
            # 結果會比 bbox 的 4-edge inset 略大一點 (長寬比越懸殊差異越明顯)，但因
            # Task 4 的 tile grid 是以 bbox 為界，差距的那幾個 px 不會真的出現在 mask 上。
            if (panel_polygon is not None
                    and offset != 0
                    and reference_polygon is None):
                cx = (panel_polygon[:, 0].mean())
                cy = (panel_polygon[:, 1].mean())
                for i in range(4):
                    dx = panel_polygon[i, 0] - cx
                    dy = panel_polygon[i, 1] - cy
                    length = float(np.hypot(dx, dy))
                    if length > 1e-6:
                        shrink = offset / length
                        panel_polygon[i, 0] -= dx * shrink
                        panel_polygon[i, 1] -= dy * shrink

            # 若 polygon 存在且啟用 otsu_bottom_crop，截掉下半部 —
            # 用 left/right 邊與新底線 y=new_bottom 的交點當新的 BL/BR，
            # 保留 panel 的底部傾斜度，而不是把兩角硬壓成同一 y。
            if panel_polygon is not None and original_y2 is not None:
                new_bottom = float(y_end)
                TL = panel_polygon[0]
                TR = panel_polygon[1]
                BR = panel_polygon[2]
                BL = panel_polygon[3]

                def _intersect_edge_with_horizontal(p_top, p_bot, y_line):
                    """線段 (p_top→p_bot) 與水平線 y=y_line 的交點 x 座標"""
                    dy = p_bot[1] - p_top[1]
                    if abs(dy) < 1e-9:
                        return float(p_top[0])
                    t = (y_line - p_top[1]) / dy
                    return float(p_top[0] + t * (p_bot[0] - p_top[0]))

                # 只有當現有 BR/BL 已經低於 new_bottom 時才做裁切
                if BR[1] > new_bottom or BL[1] > new_bottom:
                    new_BR_x = _intersect_edge_with_horizontal(TR, BR, new_bottom)
                    new_BL_x = _intersect_edge_with_horizontal(TL, BL, new_bottom)
                    panel_polygon[2, 0] = new_BR_x
                    panel_polygon[2, 1] = new_bottom
                    panel_polygon[3, 0] = new_BL_x
                    panel_polygon[3, 1] = new_bottom

        return bounds, original_y2, panel_polygon

    def find_panel_boundaries(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        向後相容別名，對應舊版呼叫。
        注意：此版本僅回傳四元組邊界，不回傳 original_y2 / polygon。
        """
        bounds, _, _ = self.calculate_otsu_bounds(image)
        return bounds

    def update_edge_config(self, config: Any):
        """
        更新 CV 邊緣檢測設定
        """
        self.edge_inspector = CVEdgeInspector(config)
    
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
        panel_polygon: Optional[np.ndarray] = None,
    ) -> List[ExclusionRegion]:
        """計算所有排除區域

        Args:
            image: 原始圖片
            otsu_bounds: Otsu 邊界
            cached_mark: 快取的 MARK 區域（Panel 級共用），若提供則跳過模板匹配
            panel_polygon: 面板 4 角 polygon，若提供則 relative_bottom_right 以
                           polygon BR 為錨點
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
                # 錨點: polygon BR 優先，否則回退到 bbox 右下
                if panel_polygon is not None:
                    anchor_x = int(round(float(panel_polygon[2][0])))
                    anchor_y = int(round(float(panel_polygon[2][1])))
                else:
                    anchor_x = otsu_x2
                    anchor_y = otsu_y2

                br_x1 = max(otsu_x1, anchor_x - zone.width)
                br_y1 = max(otsu_y1, anchor_y - zone.height)
                regions.append(ExclusionRegion(
                    name=zone.name,
                    x1=br_x1,
                    y1=br_y1,
                    x2=anchor_x,
                    y2=anchor_y,
                ))
        
        return regions
    
    def tile_image(
        self,
        image: np.ndarray,
        otsu_bounds: Tuple[int, int, int, int],
        exclusion_regions: List[ExclusionRegion],
        panel_polygon: Optional[np.ndarray] = None,
        exclusion_threshold: float = 0.0,  # 重疊比例超過此值則跳過 (0.0 = 任何重疊都跳過)
    ) -> Tuple[List[TileInfo], int]:
        """
        將圖片切成 tile，完全跳過與排除區域重疊的 tile
        邊緣不足 512px 的區域會向前回推補齊

        Args:
            image: 原始圖片
            otsu_bounds: Otsu 邊界
            exclusion_regions: 排除區域列表
            panel_polygon: 面板 4 角 polygon (shape (4,2))。若提供，會與每個 tile
                           求交集產生 tile.mask，完全在 polygon 外的 tile 會被跳過，
                           完全在 polygon 內的 tile mask 設為 None 以節省記憶體。
            exclusion_threshold: 重疊比例閾值，超過此值則跳過該 tile (預設 0.0 = 任何重疊都跳過)

        Returns:
            (有效 tiles, 被跳過的 tile 數量)
        """
        otsu_x1, otsu_y1, otsu_x2, otsu_y2 = otsu_bounds
        tile_size = self.config.tile_size
        stride = self.config.tile_stride

        # 若提供 polygon，先在整張圖尺寸上畫好 panel mask，
        # 之後每個 tile 只要從裡面 slice 出對應區塊即可。
        # 這樣可以避免「shifted polygon 超出 tile canvas 時 cv2.fillPoly
        # 邊緣 rasterization 與 full-canvas 路徑不一致」的問題，
        # 同時保證與外部 ground truth (fillPoly(full_image_size)) 完全相符。
        # 記憶體成本: H*W uint8 (~28 MB for 6576x4384 panels)，
        # tile_image 返回時就釋放。
        full_panel_mask: Optional[np.ndarray] = None
        if panel_polygon is not None:
            H, W = image.shape[:2]
            full_panel_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(full_panel_mask, [panel_polygon.astype(np.int32)], 255)
        
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
        
        # 判斷邊緣 tile 座標門檻
        bottom_y_threshold = otsu_y2 - tile_size  # 底排 tile 的起始 y 門檻
        right_x_threshold = otsu_x2 - tile_size   # 右排 tile 的起始 x 門檻
        
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
                
                # 判斷是否為邊緣 tile
                is_bottom = (y >= bottom_y_threshold)
                is_top = (y <= otsu_y1)
                is_left = (x <= otsu_x1)
                is_right = (x >= right_x_threshold)
                
                # 擷取 tile 圖片
                tile_img = image[y:tile_y2, x:tile_x2].copy()

                # 計算 tile 的 panel mask (polygon 交集)
                # 注意: .copy() 是必要的 — 不 copy 的話 tile.mask 會是
                # full_panel_mask 的 view，讓整張 28 MB buffer 無法在
                # tile_image 返回時釋放。不要刪除這個 copy。
                tile_mask: Optional[np.ndarray] = None
                if full_panel_mask is not None:
                    mask = full_panel_mask[y:tile_y2, x:tile_x2].copy()
                    if mask.max() == 0:
                        # Tile 完全在 polygon 外 → 跳過
                        excluded_count += 1
                        continue
                    if mask.min() == 255:
                        # Tile 完全在 polygon 內 → 省記憶體
                        tile_mask = None
                    else:
                        tile_mask = mask

                tiles.append(TileInfo(
                    tile_id=tile_id,
                    x=x,
                    y=y,
                    width=tile_size,
                    height=tile_size,
                    image=tile_img,
                    mask=tile_mask,
                    has_exclusion=False,  # 保留此欄位以免影響其他程式碼
                    is_bottom_edge=is_bottom,
                    is_top_edge=is_top,
                    is_left_edge=is_left,
                    is_right_edge=is_right,
                ))
                tile_id += 1
        
        return tiles, excluded_count
    
    def preprocess_image(
        self,
        image_path: Path,
        cached_mark: Optional[ExclusionRegion] = None,
        otsu_offset_override: Optional[int] = None,
        reference_raw_bounds: Optional[Tuple[int, int, int, int]] = None,
        reference_polygon: Optional[np.ndarray] = None,
    ) -> Optional[ImageResult]:
        """
        預處理圖片：Otsu + 排除區域 + 切塊

        Args:
            image_path: 圖片路徑
            cached_mark: 快取的 MARK 區域（Panel 級共用）
            otsu_offset_override: Debug 用 Otsu 內縮覆寫值 (px)
            reference_raw_bounds: 參考用的原始邊界 (來自同資料夾的白圖)。
                                  黑圖 (B0F) OTSU 無法正確偵測邊界時使用。
            reference_polygon: 參考用的 panel polygon (來自同資料夾的白圖)，
                               與 reference_raw_bounds 同時使用於 B0F fallback。

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
        # 黑圖 (B0F) 使用參考邊界，因為 OTSU 無法正確偵測全黑畫面的邊界
        if reference_raw_bounds is not None:
            raw_bounds = reference_raw_bounds
            print(f"📐 {image_path.name}: 使用參考邊界 (來自白圖) → {raw_bounds}")
        else:
            raw_bounds, _raw_binary = self._find_raw_object_bounds(image)

        # Otsu 裁切 (同樣使用參考邊界)
        otsu_bounds, original_y2, panel_polygon = self.calculate_otsu_bounds(
            image,
            otsu_offset_override=otsu_offset_override,
            reference_raw_bounds=reference_raw_bounds,
            reference_polygon=reference_polygon,
        )
        
        # 記錄裁切區域
        cropped_region = None
        if original_y2 is not None:
            # (x1, y2_new, x2, y2_old)
            cropped_region = (otsu_bounds[0], otsu_bounds[3], otsu_bounds[2], original_y2)
        
        # 計算排除區域（使用快取的 MARK 位置）
        exclusion_regions = self.calculate_exclusion_regions(
            image, otsu_bounds,
            cached_mark=cached_mark,
            panel_polygon=panel_polygon,
        )
        
        # 切塊
        tiles, excluded_count = self.tile_image(
            image, otsu_bounds, exclusion_regions,
            panel_polygon=panel_polygon,
        )
        
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
            panel_polygon=panel_polygon,
        )
    
    def _apply_edge_margin(self, anomaly_map: np.ndarray, margin_px: int,
                           sides: list = None) -> np.ndarray:
        """
        對 anomaly_map 指定邊做線性漸層衰減 (1→0)
        用於過濾邊緣光影造成的假陽性
        
        Args:
            anomaly_map: 異常熱圖 (H, W)
            margin_px: 衰減區域寬度 (像素)
            sides: 要衰減的方向列表, 如 ['top', 'bottom', 'right']
            
        Returns:
            衰減後的 anomaly_map
        """
        h, w = anomaly_map.shape[:2]
        if margin_px <= 0:
            return anomaly_map
        if sides is None:
            sides = ['bottom']
        
        result = anomaly_map.copy()
        # 平方衰減 (Quadric Decay)，讓抑制效果更強
        linear = np.linspace(1.0, 0.0, margin_px).astype(np.float32)
        gradient = np.power(linear, 2)
        
        if 'bottom' in sides and margin_px < h:
            result[-margin_px:, :] *= gradient[:, None]
        
        if 'top' in sides and margin_px < h:
            result[:margin_px, :] *= gradient[::-1, None]  # 反向：頂部邊緣 0 → 1
        
        if 'right' in sides and margin_px < w:
            result[:, -margin_px:] *= gradient[None, :]
        
        if 'left' in sides and margin_px < w:
            result[:, :margin_px] *= gradient[None, ::-1]  # 反向：左側邊緣 0 → 1
        
        return result

    def _fix_legacy_precision(self, inferencer) -> None:
        """
        修補舊版/跨版本 anomalib checkpoint 的 fp16 precision 問題。

        問題: 模型 checkpoint 若以 fp16 precision 訓練（或跨版本載入時
        precision 屬性被錯誤還原），anomalib 的 forward 路徑可能使用
        torch.autocast(dtype=float16) 將輸入轉為 fp16，但 backbone
        (feature_extractor) 權重仍為 float32 → RuntimeError。

        策略:
        1. 修補所有能找到的 precision 屬性為 float32
        2. 在 feature_extractor (timm backbone) 上註冊 forward pre-hook，
           在每次 forward 前強制將輸入轉為 float32 — 此 hook 在最底層攔截，
           不受上層 autocast 影響，是最可靠的修補方式
        3. 確保模型權重為 float32
        """
        import enum

        model = getattr(inferencer, 'model', None)
        if model is None:
            return

        # ── Step 1: 修補所有 precision 屬性 ──
        fp32_values = {"float32", "FLOAT32", "32"}
        targets = [inferencer, model]

        # 也檢查 model.model (inner PatchcoreModel)
        inner_model = getattr(model, 'model', None)
        if inner_model is not None:
            targets.append(inner_model)

        for obj in targets:
            obj_name = type(obj).__name__
            for attr_name in ("precision", "_precision"):
                val = getattr(obj, attr_name, None)
                if val is None:
                    continue
                val_str = val.value if isinstance(val, enum.Enum) else str(val)
                if val_str not in fp32_values:
                    # 嘗試用同類 enum 的 FLOAT32 成員替換
                    if isinstance(val, enum.Enum):
                        try:
                            setattr(obj, attr_name, type(val)("float32"))
                            print(f"  🔧 修補 {obj_name}.{attr_name}: {val} → float32")
                            continue
                        except (ValueError, KeyError):
                            pass
                    setattr(obj, attr_name, "float32")
                    print(f"  🔧 修補 {obj_name}.{attr_name}: {val} → 'float32'")

        # ── Step 2: 在 feature_extractor 上註冊 forward pre-hook ──
        # 找到 feature_extractor (可能在 model 或 model.model 上)
        fe = None
        for candidate in [inner_model, model]:
            if candidate is not None and hasattr(candidate, 'feature_extractor'):
                fe = candidate.feature_extractor
                break

        if fe is not None:
            def _force_fp32_hook(module, args):
                """Forward pre-hook: 強制所有 tensor 輸入轉為 float32"""
                new_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor) and arg.dtype == torch.float16:
                        new_args.append(arg.float())
                    else:
                        new_args.append(arg)
                return tuple(new_args)

            fe.register_forward_pre_hook(_force_fp32_hook)
            print(f"  🔧 已在 feature_extractor ({type(fe).__name__}) 註冊 float32 pre-hook")
        else:
            print("  ⚠️ 未找到 feature_extractor，無法註冊 pre-hook")

        # ── Step 3: 確保模型權重為 float32 ──
        if isinstance(model, torch.nn.Module):
            has_half = any(p.dtype == torch.float16 for p in model.parameters())
            if has_half:
                model.float()
                print("  🔧 模型權重已全部轉回 float32")

    def _optimize_model_fp16(self, inferencer) -> None:
        """
        PatchCore KNN 加速: memory bank → fp16 + nearest_neighbors matmul → fp16 tensor core

        PatchCore 的推論瓶頸在 euclidean_dist 中的 torch.matmul(embedding, memory_bank.T)
        將此 matmul 轉為 fp16 可利用 GPU tensor core 大幅加速 (Ampere+ ~8x matmul throughput)
        norms 保持 fp32 避免 catastrophic cancellation
        """
        model = getattr(inferencer, 'model', None)
        if model is None or not isinstance(model, torch.nn.Module):
            return

        # 導航到 PatchCore torch model (可能被 Lightning module 包裝)
        # inferencer.model → Patchcore (Lightning) → .model → PatchcoreModel (torch)
        torch_model = model
        if hasattr(model, 'model') and isinstance(model.model, torch.nn.Module):
            torch_model = model.model

        # 1. Memory bank → fp16 (省 VRAM + 讓 matmul 自動使用 fp16)
        if hasattr(torch_model, 'memory_bank') and torch_model.memory_bank.numel() > 0:
            mb_shape = torch_model.memory_bank.shape
            vram_save_mb = mb_shape[0] * mb_shape[1] * 2 / 1024 / 1024  # fp32→fp16 省一半
            torch_model.memory_bank = torch_model.memory_bank.half()
            # 預算 y_norm (fp32) 並快取，memory bank 不變所以只需算一次
            torch_model._y_norm_cache = torch_model.memory_bank.float().pow(2).sum(dim=-1, keepdim=True)
            print(f"  ⚡ Memory bank → fp16 ({mb_shape[0]} vectors, 省 {vram_save_mb:.0f}MB VRAM)")
        else:
            print("  ⚠️ 未找到 memory_bank，跳過 fp16 優化")
            return

        # 2. Patch nearest_neighbors — 注入快取的 y_norm，matmul 用 fp16 tensor core
        torch_model_class = type(torch_model)
        if not getattr(torch_model_class, '_fp16_patched', False):
            # Patch euclidean_dist — 被 nearest_neighbors 和 compute_anomaly_score 共用
            # memory_bank 已轉 fp16，所有經過 euclidean_dist 的路徑都需要處理 dtype
            if not hasattr(torch_model_class, 'euclidean_dist'):
                print(f"  ⚠️ {torch_model_class.__name__} 無 euclidean_dist 方法，跳過 patch")
                return

            @staticmethod
            def _fp16_euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # norms 保持 fp32 (避免 catastrophic cancellation: a²-2ab+b²)
                x_norm = x.float().pow(2).sum(dim=-1, keepdim=True)
                y_norm = y.float().pow(2).sum(dim=-1, keepdim=True)
                # 核心 matmul 用 fp16 → tensor core 加速
                xy = torch.matmul(x.half(), y.half().transpose(-2, -1)).float()
                res = x_norm - 2 * xy + y_norm.transpose(-2, -1)
                return res.clamp_min_(0).sqrt_()

            torch_model_class.euclidean_dist = _fp16_euclidean_dist

            # Patch nearest_neighbors — 注入快取的 y_norm (memory bank 不變，只需算一次)
            if hasattr(torch_model_class, 'nearest_neighbors'):
                def _fp16_nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int):
                    x_norm = embedding.pow(2).sum(dim=-1, keepdim=True)
                    y_norm = getattr(self, '_y_norm_cache', None)
                    if y_norm is None:
                        y_norm = self.memory_bank.float().pow(2).sum(dim=-1, keepdim=True)
                    xy = torch.matmul(embedding.half(), self.memory_bank.half().transpose(-2, -1)).float()
                    distances = (x_norm - 2 * xy + y_norm.transpose(-2, -1)).clamp_min_(0).sqrt_()
                    if n_neighbors == 1:
                        return distances.min(1)
                    return distances.topk(k=n_neighbors, largest=False, dim=1)

                torch_model_class.nearest_neighbors = _fp16_nearest_neighbors

            torch_model_class._fp16_patched = True
            print("  ⚡ KNN → fp16 matmul + cached y_norm (tensor core acceleration)")

    def _prepare_tile_tensor(self, tile_image: np.ndarray) -> torch.Tensor:
        """將單一 tile 的 numpy 圖片轉為 tensor (CHW float32 [0,1])，匹配 anomalib 的預處理"""
        img = tile_image
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0

    def _batch_forward(self, tiles: List[TileInfo], inferencer, batch_size: int = 4) -> Optional[List[Tuple[float, Optional[np.ndarray]]]]:
        """
        批次模型推論，回傳每個 tile 的 (pred_score, anomaly_map_numpy)

        僅支援 PyTorch 模型，OpenVINO 模型回傳 None (fallback 到逐 tile 推論)
        batch_size 預設 4，512x512 tiles 在 16GB VRAM 下安全運行
        """
        model = getattr(inferencer, 'model', None)
        if model is None or not isinstance(model, torch.nn.Module):
            return None

        device = getattr(inferencer, 'device', self.device)
        model.eval()

        # 預先轉換所有 tile 為 tensor (CPU 上)
        tensors = [self._prepare_tile_tensor(t.image) for t in tiles]
        results = []

        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start:start + batch_size]).to(device)
            with torch.no_grad():
                preds = model(batch)

            scores = preds.pred_score
            amaps = preds.anomaly_map

            for i in range(batch.shape[0]):
                score = float(scores[i].item()) if scores.ndim > 0 else float(scores.item())
                amap = None
                if amaps is not None:
                    amap = amaps[i].squeeze().cpu().numpy()
                results.append((score, amap))

            # 釋放 GPU 記憶體
            del batch, preds, scores, amaps

        return results

    def predict_tile(self, tile: TileInfo, inferencer=None, edge_margin_override: Optional[int] = None, patchcore_overrides: Optional[Dict[str, Any]] = None, threshold: Optional[float] = None, raw_prediction: Optional[Tuple[float, Optional[np.ndarray]]] = None) -> Tuple[float, Optional[np.ndarray]]:
        """
        對單一 tile 進行推論

        Args:
            tile: TileInfo 物件
            inferencer: 指定的 inferencer 物件，若為 None 使用 self.inferencer
            threshold: 異常判斷閾值（用於面積過濾），若為 None 使用 self.threshold
            raw_prediction: 預先計算的 (pred_score, anomaly_map)，若提供則跳過模型推論

        Returns:
            (異常分數, 異常熱圖) - 如果有遮罩，會過濾排除區域的異常
        """
        active_threshold = threshold if threshold is not None else self.threshold

        if raw_prediction is not None:
            # 使用預先批次計算的結果，跳過模型推論
            pred_score, anomaly_map = raw_prediction
        else:
            # 逐 tile 推論 (fallback)
            active_inferencer = inferencer or self.inferencer
            if active_inferencer is None:
                raise RuntimeError("模型尚未載入")

            # 使用 numpy array 進行推論
            # 如果是灰階 (2D 或 1 channel)，轉為 BGR
            input_image = tile.image
            if len(input_image.shape) == 2 or (len(input_image.shape) == 3 and input_image.shape[2] == 1):
                 if len(input_image.shape) == 2:
                     input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
                 else:
                     input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

            predictions = active_inferencer.predict(input_image)

            # 取得分數
            pred_score = float(predictions.pred_score.item()) if hasattr(predictions.pred_score, 'item') else float(predictions.pred_score)

            # 取得熱圖（如果有的話）
            anomaly_map = None
            if hasattr(predictions, 'anomaly_map') and predictions.anomaly_map is not None:
                anomaly_map = predictions.anomaly_map.squeeze().cpu().numpy() if hasattr(predictions.anomaly_map, 'cpu') else predictions.anomaly_map.squeeze()
            
        # === 以下為 anomaly_map 後處理 (batch 和 fallback 共用) ===
        if anomaly_map is not None:
            # 記錄衰減/遮罩處理前的 anomaly_map max (用於後續比率計算)
            pre_process_max = float(np.max(anomaly_map))

            # --- PatchCore 後處理過濾 ---
            patchcore_enabled = getattr(self.config, 'patchcore_filter_enabled', False)
            if patchcore_overrides is not None and 'patchcore_filter_enabled' in patchcore_overrides:
                patchcore_enabled = patchcore_overrides['patchcore_filter_enabled']

            if patchcore_enabled:
                # 1. 高斯平滑
                sigma = getattr(self.config, 'patchcore_blur_sigma', 1.5)
                if patchcore_overrides is not None and 'patchcore_blur_sigma' in patchcore_overrides:
                    sigma = patchcore_overrides['patchcore_blur_sigma']

                if sigma > 0:
                    ksize = int(2 * round(3 * sigma) + 1)
                    anomaly_map = cv2.GaussianBlur(anomaly_map, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

                # 2. 特徵值計算模式
                metric = getattr(self.config, 'patchcore_score_metric', 'max')
                if patchcore_overrides is not None and 'patchcore_score_metric' in patchcore_overrides:
                    metric = patchcore_overrides['patchcore_score_metric']

                if metric == 'top_k_avg':
                    # 取前 10 個最高值的平均 (k=10)
                    k = 10
                    flat = anomaly_map.flatten()
                    if len(flat) >= k:
                        idx = np.argpartition(flat, -k)[-k:]
                        top_k_val = np.mean(flat[idx])
                    else:
                        top_k_val = np.max(anomaly_map)
                    pre_process_max = float(top_k_val)
                elif metric == 'percentile_99':
                    pre_process_max = float(np.percentile(anomaly_map, 99))
                else: # 'max' 或其他
                    pre_process_max = float(np.max(anomaly_map))

                # 更新基礎預測分數 (覆寫原本直接從 anomalib 拿的 score)
                # 這裡假設 anomaly_map 的尺度與 pred_score 一致，直接取代
                # 如果尺度不一致，這裡會成為新的基準
                pred_score = pre_process_max

                # 3. 面積過濾 (只在分數超過閾值時檢查，節省效能)
                min_area = getattr(self.config, 'patchcore_min_area', 10)
                if patchcore_overrides is not None and 'patchcore_min_area' in patchcore_overrides:
                    min_area = patchcore_overrides['patchcore_min_area']

                if min_area > 0 and pred_score >= active_threshold:
                    # 以 peak×0.5 二值化找最大 cluster，判斷是否為雜訊點
                    max_cluster_area = _anomaly_max_cc_area(anomaly_map, pre_process_max)

                    if max_cluster_area < min_area:
                        # 面積不足，大幅降權
                        pred_score = pred_score * 0.5
                        print(f"    ℹ️ Tile 異常面積過小 ({max_cluster_area} < {min_area})，降權懲罰")

            # --- 集中度檢查 (Concentration Check) ---
            # 瀰漫性假陽性: heatmap 均勻偏暖但無局部峰值 → Peak/Mean ratio 低
            # 真實缺陷: heatmap 有明顯局部峰值 → Peak/Mean ratio 高
            concentration_enabled = getattr(self.config, 'patchcore_concentration_enabled', True)
            if patchcore_overrides is not None and 'patchcore_concentration_enabled' in patchcore_overrides:
                concentration_enabled = patchcore_overrides['patchcore_concentration_enabled']

            if concentration_enabled:
                positive_vals = anomaly_map[anomaly_map > 0]
                if len(positive_vals) > 0:
                    peak_val = float(np.max(positive_vals))
                    mean_val = float(np.mean(positive_vals))
                    concentration_ratio = peak_val / mean_val if mean_val > 0 else float('inf')

                    min_ratio = getattr(self.config, 'patchcore_concentration_min_ratio', 2.0)
                    if patchcore_overrides is not None and 'patchcore_concentration_min_ratio' in patchcore_overrides:
                        min_ratio = patchcore_overrides['patchcore_concentration_min_ratio']

                    if concentration_ratio < min_ratio and min_ratio > 1.0:
                        penalty = getattr(self.config, 'patchcore_concentration_penalty', 0.5)
                        if patchcore_overrides is not None and 'patchcore_concentration_penalty' in patchcore_overrides:
                            penalty = patchcore_overrides['patchcore_concentration_penalty']

                        # 線性插值: ratio=1.0 → penalty, ratio=min_ratio → 1.0 (無懲罰)
                        factor = (concentration_ratio - 1.0) / (min_ratio - 1.0)
                        factor = max(0.0, min(1.0, factor))
                        penalty_mult = penalty + (1.0 - penalty) * factor
                        pred_score *= penalty_mult
                        logger.debug(f"瀰漫性檢查: Peak/Mean={concentration_ratio:.2f} < {min_ratio:.1f}，降權 x{penalty_mult:.3f}")

            # --- 擴散面積檢查 (Diffuse Area Check) ---
            # 梯度型假陽性: heatmap 有大面積偏暖 (左熱右冷等梯度) → 熱區佔比高
            # 真實缺陷: heatmap 熱區集中在小區域 → 熱區佔比低
            diffuse_enabled = getattr(self.config, 'patchcore_diffuse_area_enabled', True)
            if patchcore_overrides is not None and 'patchcore_diffuse_area_enabled' in patchcore_overrides:
                diffuse_enabled = patchcore_overrides['patchcore_diffuse_area_enabled']

            if diffuse_enabled:
                map_max = float(np.max(anomaly_map))
                if map_max > 0:
                    half_peak = map_max * 0.5
                    hot_pixels = int(np.count_nonzero(anomaly_map >= half_peak))
                    total_pixels = anomaly_map.size
                    hot_ratio = hot_pixels / total_pixels if total_pixels > 0 else 0.0

                    diffuse_threshold = getattr(self.config, 'patchcore_diffuse_area_threshold', 0.3)
                    if patchcore_overrides is not None and 'patchcore_diffuse_area_threshold' in patchcore_overrides:
                        diffuse_threshold = patchcore_overrides['patchcore_diffuse_area_threshold']

                    if hot_ratio > diffuse_threshold:
                        diffuse_penalty = getattr(self.config, 'patchcore_diffuse_area_penalty', 0.5)
                        if patchcore_overrides is not None and 'patchcore_diffuse_area_penalty' in patchcore_overrides:
                            diffuse_penalty = patchcore_overrides['patchcore_diffuse_area_penalty']

                        # 線性插值: hot_ratio=threshold → 1.0 (無懲罰), hot_ratio=1.0 → penalty (最大懲罰)
                        factor = (hot_ratio - diffuse_threshold) / (1.0 - diffuse_threshold) if diffuse_threshold < 1.0 else 0.0
                        factor = max(0.0, min(1.0, factor))
                        penalty_mult = 1.0 - (1.0 - diffuse_penalty) * factor
                        pred_score *= penalty_mult
                        logger.debug(f"擴散面積檢查: HotRatio={hot_ratio:.2%} > {diffuse_threshold:.0%}，降權 x{penalty_mult:.3f}")

            # 記錄 mask/邊緣衰減前的 max (用於 decay ratio，統一使用 max 避免 metric 不一致)
            pre_decay_max = float(np.max(anomaly_map))

            # 如果有遮罩，將排除區域的熱圖值設為 0
            if tile.mask is not None:
                # 確保遮罩尺寸匹配
                if anomaly_map.shape != tile.mask.shape:
                    mask_resized = cv2.resize(tile.mask, (anomaly_map.shape[1], anomaly_map.shape[0]))
                else:
                    mask_resized = tile.mask
                # 將排除區域設為 0
                anomaly_map = anomaly_map * (mask_resized / 255.0)

            # 邊緣衰減：過濾光影假陽性 (debug 模式可覆寫數值)
            edge_margin = self.config.edge_margin_px if edge_margin_override is None else edge_margin_override
            if edge_margin > 0:
                # 收集此 tile 需要衰減的方向
                cfg_sides = self.config.edge_margin_sides
                sides = []
                if tile.is_top_edge and cfg_sides.get('top', False):
                    sides.append('top')
                if tile.is_bottom_edge and cfg_sides.get('bottom', False):
                    sides.append('bottom')
                if tile.is_left_edge and cfg_sides.get('left', False):
                    sides.append('left')
                if tile.is_right_edge and cfg_sides.get('right', False):
                    sides.append('right')

                if sides:
                    # 將 margin_px 按 anomaly_map 實際尺寸縮放
                    scale = anomaly_map.shape[0] / tile.height
                    scaled_margin = int(edge_margin * scale)
                    anomaly_map = self._apply_edge_margin(anomaly_map, scaled_margin, sides=sides)
        
        # 如果有遮罩或邊緣衰減，使用衰減比率調整分數（保持與 anomalib pred_score 相同尺度）
        actual_edge_margin = self.config.edge_margin_px if edge_margin_override is None else edge_margin_override
        has_edge_margin = actual_edge_margin > 0 and any([
            tile.is_top_edge and self.config.edge_margin_sides.get('top', False),
            tile.is_bottom_edge and self.config.edge_margin_sides.get('bottom', False),
            tile.is_left_edge and self.config.edge_margin_sides.get('left', False),
            tile.is_right_edge and self.config.edge_margin_sides.get('right', False),
        ])
        need_recalc = (tile.mask is not None) or has_edge_margin
        if need_recalc and anomaly_map is not None:
            post_decay_max = float(np.max(anomaly_map))

            if pre_decay_max > 0:
                # 用 max 比率調整 pred_score (統一使用 max 作為 decay 基準，避免 metric 不一致)
                decay_ratio = post_decay_max / pre_decay_max
                pred_score = pred_score * decay_ratio
            else:
                pred_score = 0.0
        
        return pred_score, anomaly_map

    def _detect_bright_spots(self, tile: 'TileInfo') -> Tuple[float, Optional[np.ndarray]]:
        """
        B0F00000 專用：偵測黑色背景上的異常亮點。

        取代 PatchCore 推論，用於無訓練模型的圖片。
        使用局部對比偵測（median filter 背景估計 → 差異計算 → 閾值判定），
        同時保留絕對亮度上限保護。

        偵測邏輯：
          1. median filter 估計局部背景
          2. 原圖 - 背景 = 局部差異（比背景亮的部分）
          3. 差異 > diff_threshold → 候選亮點
          4. 連通分量面積篩選 ≥ min_area
          5. 絕對亮度 > bright_spot_threshold 的也直接納入（上限保護）

        Args:
            tile: TileInfo 物件

        Returns:
            (score, binary_map) - score: 0.0 (無亮點) 或 1.0 (有亮點)
                                  binary_map: 二值化結果 (uint8, 0/255)
        """
        abs_threshold = self.config.bright_spot_threshold
        min_area = self.config.bright_spot_min_area
        median_kernel = self.config.bright_spot_median_kernel
        diff_threshold = self.config.bright_spot_diff_threshold

        img = tile.image
        if img is None:
            return 0.0, None

        # 灰階化
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        mk = clamp_median_kernel(median_kernel, min(gray.shape[:2]) - 1)

        # 背景估計 → 局部差異
        bg = cv2.medianBlur(gray, mk)
        diff = cv2.subtract(gray, bg)  # 只取比背景亮的部分 (saturate at 0)

        # 局部對比閾值：差異超過 diff_threshold 的為候選亮點
        _, binary_diff = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

        # 絕對亮度上限保護：超過 abs_threshold 的直接納入
        _, binary_abs = cv2.threshold(gray, abs_threshold, 255, cv2.THRESH_BINARY)

        # 合併兩種偵測結果
        binary = cv2.bitwise_or(binary_diff, binary_abs)

        # 如果 tile 有 mask（排除區域），套用 mask
        if tile.mask is not None:
            mask_resized = cv2.resize(tile.mask, (binary.shape[1], binary.shape[0]))
            binary = cv2.bitwise_and(binary, mask_resized)

        # 連通分量分析，過濾小面積雜訊
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        filtered_binary = np.zeros_like(binary)
        has_bright_spot = False
        max_component_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            max_component_area = max(max_component_area, area)
            if area >= min_area:
                filtered_binary[labels == i] = 255
                has_bright_spot = True

        score = 1.0 if has_bright_spot else 0.0

        # 轉為 float anomaly_map 格式（與 PatchCore 輸出格式相容）
        anomaly_map = filtered_binary.astype(np.float32) / 255.0

        # 標記此 tile 為二值化偵測模式，並儲存偵測統計供 heatmap header 顯示
        tile.is_bright_spot_detection = True
        tile.bright_spot_max_diff = int(diff.max())
        tile.bright_spot_diff_threshold = diff_threshold
        tile.bright_spot_area = int(np.sum(filtered_binary > 0))
        tile.bright_spot_min_area = min_area

        # 偵測結果 log
        max_pixel_val = int(gray.max())
        max_diff_val = int(diff.max())
        bright_pixel_count = int(np.sum(filtered_binary > 0))
        raw_bright_count = int(np.sum(binary > 0))
        if has_bright_spot:
            print(f"  💡 B0F 偵測 Tile@({tile.x},{tile.y}): 發現亮點 ({bright_pixel_count} px), "
                  f"max_diff={max_diff_val}, diff_thr={diff_threshold}, abs_thr={abs_threshold}, "
                  f"max_pixel={max_pixel_val}, median_k={mk}")
        else:
            print(f"  💡 B0F 偵測 Tile@({tile.x},{tile.y}): 未發現亮點, "
                  f"max_diff={max_diff_val}, diff_thr={diff_threshold}, abs_thr={abs_threshold}, "
                  f"max_pixel={max_pixel_val}, raw_bright={raw_bright_count} px, "
                  f"max_component={max_component_area} px (min_area={min_area})")

        return score, anomaly_map

    def run_inference(self, result: ImageResult, progress_callback=None,
                      inferencer=None, threshold: Optional[float] = None,
                      edge_margin_override: Optional[int] = None,
                      patchcore_overrides: Optional[Dict[str, Any]] = None,
                      model_id: Optional[str] = None) -> ImageResult:
        """
        對預處理結果執行推論
        
        Args:
            result: preprocess_image 的結果
            progress_callback: 進度回呼函數 (current, total)
            inferencer: 指定的 inferencer 物件，若為 None 使用 self.inferencer
            threshold: 指定的閾值，若為 None 使用 self.threshold
            model_id: 機種名稱，用於推導產品解析度 (例如 'H', 'J')
            
        Returns:
            更新後的 ImageResult（包含異常 tile 資訊）
        """
        active_inferencer = inferencer or self.inferencer
        if active_inferencer is None:
            raise RuntimeError("模型尚未載入，請在初始化時指定 model_path")
        
        active_threshold = threshold if threshold is not None else self.threshold
        
        inference_start = time.time()
        anomaly_tiles = []
        total = len(result.tiles)

        # 逐 tile 推論 (fp16 KNN 加速已在 model 載入時 patch)
        # 注意: batch 推論在 PatchCore 反而更慢 (KNN 距離矩陣隨 batch 線性膨脹)
        for i, tile in enumerate(result.tiles):
            if progress_callback:
                progress_callback(i + 1, total)

            score, anomaly_map = self.predict_tile(tile, inferencer=active_inferencer, edge_margin_override=edge_margin_override, patchcore_overrides=patchcore_overrides, threshold=active_threshold)

            if score >= active_threshold:
                anomaly_tiles.append((tile, score, anomaly_map))
            elif tile.is_aoi_coord_tile:
                # AOI 座標 tile 即使低於閾值也保留，供追蹤查看
                tile.is_aoi_coord_below_threshold = True
                anomaly_tiles.append((tile, score, anomaly_map))

        # 執行傳統 CV 邊緣檢查
        # 如果 edge_inspector 啟用，並且我們有 raw_bounds
        if getattr(self, "edge_inspector", None) and self.edge_inspector.config.enabled and result.raw_bounds:
            try:
                # 取得產品解析度代碼 (e.g. "H", "J")
                resolution_code = "UNKNOWN"
                if model_id and len(model_id) >= 6:
                    resolution_code = model_id[5].upper()
                
                # 切換 active zones 為當前產品
                self.edge_inspector.config.set_active_zones_for_product(resolution_code)
                
                # 重新讀取原圖 (全尺寸) 給 CV 處理，因為它需要高解析度才能看清楚
                # 如果 cv2 記憶體太大，可以在 preprocess 前把 raw cv_image 傳過來，但此處再次讀取較安全
                full_image = cv2.imread(str(result.image_path), cv2.IMREAD_UNCHANGED)
                if full_image is not None:
                    # CV 內部處理需要時間，記錄一下
                    cv_start = time.time()
                    edge_defects = self.edge_inspector.inspect(full_image, result.raw_bounds)
                    result.edge_defects = edge_defects
                    logger.debug(f"CV Edge Inspection: 找到 {len(edge_defects)} 個邊緣異常，耗時 {time.time() - cv_start:.3f}s")
            except Exception as e:
                logger.error(f"CV 邊緣檢查失敗 {result.image_path.name}: {e}", exc_info=True)
        
        # 更新結果
        result.anomaly_tiles = anomaly_tiles
        result.inference_time = time.time() - inference_start

        return result
    
    def get_anomaly_summary(self, result: ImageResult) -> Dict[str, Any]:
        """取得異常摘要"""
        # 計算真實的 CV 邊緣異常數 (排除疑似灰塵)
        real_edge_defects = [ed for ed in getattr(result, 'edge_defects', []) if not getattr(ed, 'is_suspected_dust_or_scratch', False)]
        
        if not result.anomaly_tiles and not real_edge_defects:
            return {
                "is_anomaly": False,
                "anomaly_count": 0,
                "max_score": 0.0,
                "anomaly_positions": [],
                "cv_edge_anomaly_count": 0,
            }
        
        scores = [score for _, score, _ in result.anomaly_tiles]
        positions = [(tile.x, tile.y, tile.width, tile.height) for tile, _, _ in result.anomaly_tiles]
        
        return {
            "is_anomaly": True,
            "anomaly_count": len(result.anomaly_tiles),
            "max_score": max(scores) if scores else 0.0,
            "avg_score": (sum(scores) / len(scores)) if scores else 0.0,
            "anomaly_positions": positions,
            "cv_edge_anomaly_count": len(real_edge_defects),
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

        # Panel polygon（紅色）
        if result.panel_polygon is not None:
            poly_int = result.panel_polygon.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [poly_int], True, (0, 0, 255), 10)
            cv2.putText(vis, "Panel Polygon", (x1 + 10, y1 + 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

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

    
    def check_dust_or_scratch_feature(self, image: np.ndarray, extension_override: Optional[int] = None) -> tuple:
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
            extension_override: 覆寫 Config 中的 dust_extension 設定
            
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
        extension = self.config.dust_extension if extension_override is None else extension_override
        
        # Step 1: 轉灰階
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 2: CLAHE 局部對比增強 — 強化微弱灰塵的可見度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Step 2.5: Top-Hat 變換 (去除大面積高光背景)
        # 邊緣區域常包含大片高光背景(如載台/膠帶)，會嚴重干擾 Otsu 閾值，導致玻璃上的微弱灰塵被忽略
        # 使用 45x45 核做開運算估計背景 (足以覆蓋多數灰塵，area_max 一般<=1000 => radius~18)
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        bg_est = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel_bg)
        # 相減保留局部亮點 (Top-Hat)
        tophat = cv2.subtract(enhanced, bg_est)
        
        # Step 3: 二值化
        # Top-Hat 後背景趨近於 0，使用 Otsu 可能因為單峰分佈而失真，
        # 故以 config 中的 fallback_threshold 為基準，並取 otsu 為輔
        otsu_thresh, _ = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 若 Otsu 閾值異常高或低，限制在合理範圍內
        threshold_floor = self.config.dust_threshold_floor
        adaptive_thr = min(max(otsu_thresh, threshold_floor), fallback_threshold)
        
        _, binary = cv2.threshold(tophat, adaptive_thr, 255, cv2.THRESH_BINARY)
        used_threshold = adaptive_thr
        
        # 合理性檢查：若前景佔比仍過高，可能全是雜訊，用更嚴格的閾值
        # 但需注意：表面嚴重刮傷的面板確實可能有 15-25% 的灰塵/刮痕前景
        MAX_REASONABLE_RATIO = 0.25
        initial_ratio = float(np.sum(binary > 0)) / binary.size if binary.size > 0 else 0.0
        if initial_ratio > MAX_REASONABLE_RATIO:
            # 嚴格閾值前，先保護寬鬆閾值下的有效灰塵/刮痕特徵
            feature_preserved = np.zeros_like(binary)
            _n, _labels, _stats, _ = cv2.connectedComponentsWithStats(binary)
            for _i in range(1, _n):
                _area = _stats[_i, cv2.CC_STAT_AREA]
                _w = _stats[_i, cv2.CC_STAT_WIDTH]
                _h = _stats[_i, cv2.CC_STAT_HEIGHT]
                if _area < area_min or _area > area_max:
                    continue
                _aspect = max(_w, _h) / (min(_w, _h) + 1e-5)
                # 保護線性刮痕(aspect>5) 以及有一定面積的灰塵顆粒
                if _aspect > 5 or _area >= area_min * 5:
                    feature_preserved[_labels == _i] = 255

            # 使用 p95 作為嚴格閾值，但限制最高不超過 adaptive_thr 的 2 倍
            # 避免極亮刮痕導致閾值飆到 170+ 而漏掉中等亮度灰塵
            p95 = float(np.percentile(tophat, 95))
            strict_thr_cap = adaptive_thr * 2.0
            strict_thr = min(max(adaptive_thr, p95), strict_thr_cap)
            _, binary = cv2.threshold(tophat, strict_thr, 255, cv2.THRESH_BINARY)
            used_threshold = strict_thr

            # 合併保護的特徵回 binary
            binary = cv2.bitwise_or(binary, feature_preserved)
        
        # Step 3.5: 明顯亮區救回 — Top-Hat 會吃掉寬度>kernel 的大面積污染/刮痕
        # 對 CLAHE 增強後的原圖做高閾值直接檢測，把肉眼明顯的亮區補回來
        bright_rescue_thr = self.config.dust_bright_rescue_threshold
        if bright_rescue_thr > 0:
            _, bright_binary = cv2.threshold(enhanced, bright_rescue_thr, 255, cv2.THRESH_BINARY)
            binary = cv2.bitwise_or(binary, bright_binary)

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
        
        # Step 6: 暗色顆粒偵測 — 偵測暗色 MARK 等暗色圖案
        # 某些機種 MARK 樣式偏黑，在 OMIT 圖上呈現暗色顆粒
        # 使用 THRESH_BINARY_INV 偵測低於背景的暗色區域
        dark_particle_count = 0
        dark_scratch_count = 0
        dark_total_area = 0
        
        if getattr(self.config, 'dust_detect_dark_particles', True):
            # 計算背景統計（排除全黑邊界像素）
            non_zero_pixels = gray[gray > 0]
            if len(non_zero_pixels) > 100:  # 確保有足夠像素做統計
                bg_median = float(np.median(non_zero_pixels))
                # 暗色閾值：取低 1st percentile 或 背景中位數的一半，取較大者
                p1 = float(np.percentile(non_zero_pixels, 1))
                dark_threshold = max(p1, bg_median * 0.5)
                
                # 只在背景中位數夠亮時才偵測暗色顆粒（避免全暗圖誤判）
                if bg_median > 20:
                    _, dark_binary = cv2.threshold(gray, int(dark_threshold), 255, cv2.THRESH_BINARY_INV)
                    
                    # 排除全黑像素（圖的邊界/padding 區域）
                    dark_binary[gray == 0] = 0
                    
                    # 合理性檢查
                    dark_ratio = float(np.sum(dark_binary > 0)) / dark_binary.size if dark_binary.size > 0 else 0.0
                    if dark_ratio <= MAX_REASONABLE_RATIO:
                        # 形態學處理
                        dark_binary = cv2.morphologyEx(dark_binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
                        if extension > 0:
                            dark_binary = cv2.dilate(dark_binary, dilate_kernel, iterations=1)
                        
                        # Connected Components 面積篩選
                        d_num_labels, d_labels, d_stats, _ = cv2.connectedComponentsWithStats(dark_binary)
                        
                        for i in range(1, d_num_labels):
                            d_area = d_stats[i, cv2.CC_STAT_AREA]
                            d_w = d_stats[i, cv2.CC_STAT_WIDTH]
                            d_h = d_stats[i, cv2.CC_STAT_HEIGHT]
                            
                            if d_area < area_min or d_area > area_max:
                                continue
                            
                            # 合併至灰塵遮罩
                            dust_mask[d_labels == i] = 255
                            dark_total_area += d_area
                            
                            d_aspect = max(d_w, d_h) / (min(d_w, d_h) + 1e-5)
                            if d_aspect > 5:
                                dark_scratch_count += 1
                            else:
                                dark_particle_count += 1
                        
                        if dark_particle_count + dark_scratch_count > 0:
                            logging.debug(f"    暗色顆粒偵測: P:{dark_particle_count} S:{dark_scratch_count} Area:{dark_total_area} (Thr:{dark_threshold:.0f}, Median:{bg_median:.0f})")
        
        # 合併計數
        total_particle = particle_count + dark_particle_count
        total_scratch = scratch_count + dark_scratch_count
        total_area = total_dust_area + dark_total_area
        
        # 計算灰塵面積佔比
        bright_ratio = float(np.sum(dust_mask > 0)) / dust_mask.size if dust_mask.size > 0 else 0.0
        is_dust = (total_particle + total_scratch) > 0
        
        dark_info = f" DkP:{dark_particle_count} DkS:{dark_scratch_count}" if (dark_particle_count + dark_scratch_count) > 0 else ""
        detail_text = (f"Thr:{used_threshold:.0f} P:{particle_count} S:{scratch_count} "
                       f"Area:{total_area} Ratio:{bright_ratio:.4f}{dark_info}")
        
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
                                  top_percent: float = 5.0,
                                  metric: str = "coverage") -> tuple:
        """
        計算灰塵遮罩與 Heatmap「最紅區域」的重疊指標 (Coverage 或 IOU)
        
        Coverage = 交集 / 灰塵面積 (適合熱區遠大於灰塵的場景)
        IOU = 交集 / (灰塵面積 + 熱區面積 - 交集)
        
        使用 Percentile 方式：取 anomaly_map 中數值最高的前 X% 像素作為熱區，
        比舊的 max*ratio 更穩定、不受單一極端值影響。
        
        Args:
            dust_mask: 灰塵遮罩 (uint8, 255=灰塵)
            anomaly_map: Heatmap 異常圖 (float, 可含負值)
            top_percent: 取最高的前百分之幾作為「最紅區域」(建議 3~8)
            metric: "coverage" 或 "iou"
            
        Returns:
            (metric_value, heatmap_binary) - 指標值 (0.0~1.0), 二值化後的熱區遮罩
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
        
        intersection = np.count_nonzero(dust_bool & heat_bool)
        
        if metric == "coverage":
            dust_area = np.count_nonzero(dust_bool)
            metric_val = float(intersection / dust_area) if dust_area > 0 else 0.0
        else:
            union = np.count_nonzero(dust_bool | heat_bool)
            metric_val = float(intersection / union) if union > 0 else 0.0
            
        return metric_val, heatmap_binary

    def check_dust_per_region(
        self,
        dust_mask: np.ndarray,
        anomaly_map: np.ndarray,
        top_percent: float = 5.0,
        metric: str = "coverage",
        iou_threshold: float = 0.01,
    ) -> tuple:
        """
        逐區域灰塵判定 — 將 anomaly_map 的熱區拆成獨立連通區域，
        分別與 dust_mask 做交叉驗證，只抑制與灰塵重疊的區域，保留真實缺陷。

        Returns:
            (has_real_defect, real_peak_yx, overall_iou, region_details, heatmap_binary, labels)
            - has_real_defect: 是否存在非灰塵的真實異常區域
            - real_peak_yx: 非灰塵區域中 anomaly_map 最大值的 (row, col)，None 表示全為灰塵
            - overall_iou: 整體 coverage/iou (向後相容)
            - region_details: list of dict，每個區域的判定詳情
            - heatmap_binary: 二值化後的熱區遮罩 (向後相容)
        """
        if dust_mask is None or anomaly_map is None:
            return True, None, 0.0, [], None, None

        anomaly_map_f = np.asarray(anomaly_map, dtype=np.float32)
        anomaly_map_f = np.maximum(anomaly_map_f, 0.0)
        dust_mask_u8 = np.asarray(dust_mask, dtype=np.uint8)

        if np.max(anomaly_map_f) <= 0:
            return True, None, 0.0, [], None, None

        # === Step 1.5: 灰塵遮罩前處理 (提前到 Step 1 之前，供 mask 模式使用) ===
        if len(dust_mask_u8.shape) == 3:
            dust_mask_u8 = cv2.cvtColor(dust_mask_u8, cv2.COLOR_BGR2GRAY)
        if dust_mask_u8.shape != anomaly_map_f.shape:
            dust_mask_u8 = cv2.resize(
                dust_mask_u8,
                (anomaly_map_f.shape[1], anomaly_map_f.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        dust_bool = dust_mask_u8 > 0

        # === Step 1: 取前 top_percent% 像素作為熱區 ===
        use_mask_mode = getattr(self.config, 'dust_mask_before_binarize', False)

        if use_mask_mode:
            # 方法 4: 先將灰塵區域歸零，再對剩餘區域做 top% 二值化
            masked_anomaly = anomaly_map_f.copy()
            masked_anomaly[dust_bool] = 0
            positive_values = masked_anomaly[masked_anomaly > 0]
            if len(positive_values) == 0:
                # 全部被 mask 掉 → 整張都是灰塵
                heatmap_binary = np.zeros_like(anomaly_map_f, dtype=np.uint8)
                return False, None, 1.0, [], heatmap_binary, None
            threshold = np.percentile(positive_values, 100 - top_percent)
            heat_bool = masked_anomaly >= threshold
        else:
            # 現有流程: 直接對原始 anomaly_map 做 top% 二值化
            positive_values = anomaly_map_f[anomaly_map_f > 0]
            if len(positive_values) == 0:
                return True, None, 0.0, [], None, None
            threshold = np.percentile(positive_values, 100 - top_percent)
            heat_bool = anomaly_map_f >= threshold

        heatmap_binary = (heat_bool.astype(np.uint8)) * 255

        # === Step 3: 整體 IOU (向後相容，用於 DB 記錄) ===
        intersection_all = np.count_nonzero(dust_bool & heat_bool)
        if metric == "coverage":
            dust_area_all = np.count_nonzero(dust_bool)
            overall_iou = float(intersection_all / dust_area_all) if dust_area_all > 0 else 0.0
        else:
            union_all = np.count_nonzero(dust_bool | heat_bool)
            overall_iou = float(intersection_all / union_all) if union_all > 0 else 0.0

        # === Step 4: 連通區域分析 ===
        heat_u8 = heatmap_binary.copy()
        num_labels, labels = cv2.connectedComponents(heat_u8, connectivity=8)

        region_details = []
        has_real_defect = False
        real_peak_yx = None
        real_peak_score = -1.0

        for label_id in range(1, num_labels):
            region_mask = labels == label_id
            region_area = np.count_nonzero(region_mask)

            # 計算此區域與灰塵的重疊
            region_dust_overlap = np.count_nonzero(region_mask & dust_bool)

            if metric == "coverage":
                # 此異常區域被灰塵覆蓋的比例
                region_coverage = float(region_dust_overlap / region_area) if region_area > 0 else 0.0
            else:
                # IOU = 交集 / 聯集 (此異常區域 ∪ 全部灰塵)
                total_dust = np.count_nonzero(dust_bool)
                region_union = region_area + total_dust - region_dust_overlap
                region_coverage = float(region_dust_overlap / region_union) if region_union > 0 else 0.0

            # 此區域內 anomaly_map 的最大值與位置 (無須複製整個陣列)
            region_vals = anomaly_map_f[region_mask]
            region_max_score = float(np.max(region_vals))
            region_indices = np.where(region_mask)
            local_argmax = np.argmax(region_vals)
            peak_pos = (region_indices[0][local_argmax], region_indices[1][local_argmax])

            # 判定是否為灰塵：覆蓋率須達閾值 且 峰值（最熱點）必須落在灰塵 mask 內
            # 若峰值不在灰塵上，代表缺陷核心與灰塵無關，僅邊緣碰到，不應判為灰塵
            # 但 heatmap peak 有膨脹偏移問題，高覆蓋率時直接判 dust 不依賴 peak 位置
            peak_in_dust = bool(dust_bool[peak_pos[0], peak_pos[1]])
            high_cov_thr = getattr(self.config, 'dust_high_cov_threshold', 0.5)
            is_dust_region = region_coverage >= iou_threshold and (peak_in_dust or region_coverage >= high_cov_thr)

            # 殘餘異常檢查：即使 peak 在灰塵上，若非灰塵區域仍有強異常信號則 rescue
            # 解決「灰塵信號遮蔽同區域內細微真實缺陷」的漏檢問題
            residual_ratio = 0.0
            if is_dust_region:
                non_dust_in_region = region_mask & (~dust_bool)
                if np.any(non_dust_in_region):
                    sub_peak_score = float(np.max(anomaly_map_f[non_dust_in_region]))
                    residual_ratio = sub_peak_score / region_max_score if region_max_score > 0 else 0.0
                    residual_thr = getattr(self.config, 'dust_residual_ratio', 0.7)
                    if residual_ratio >= residual_thr:
                        is_dust_region = False
                        logging.info(
                            f"    Region {label_id}: DUST->REAL_NG rescue "
                            f"(residual sub-peak {sub_peak_score:.4f}/{region_max_score:.4f}"
                            f"={residual_ratio:.2f} >= {residual_thr})"
                        )

            region_details.append({
                "label_id": label_id,
                "area": region_area,
                "dust_overlap": region_dust_overlap,
                "coverage": region_coverage,
                "is_dust": is_dust_region,
                "peak_in_dust": peak_in_dust,
                "residual_ratio": residual_ratio,
                "max_score": region_max_score,
                "peak_yx": peak_pos,
            })

            if not is_dust_region:
                has_real_defect = True
                if region_max_score > real_peak_score:
                    real_peak_score = region_max_score
                    real_peak_yx = peak_pos

        return has_real_defect, real_peak_yx, overall_iou, region_details, heatmap_binary, labels

    def check_dust_two_stage(
        self,
        tile_image: np.ndarray,
        anomaly_map: np.ndarray,
        dust_mask: np.ndarray,
        score: float,
    ) -> tuple:
        """
        兩階段灰塵判定：
          Stage 1: 用 heatmap 找出 hot zone（大概位置）
          Stage 2: 回到原圖找 feature 點 → 精準比對 dust_mask（無擴散問題）

        Returns:
            (has_real_defect, real_peak_yx, feature_details, detail_text)
        """
        cfg = self.config
        dust_ratio_thr = cfg.dust_two_stage_dust_ratio
        bg_blur_k = cfg.dust_two_stage_bg_blur
        diff_pct = cfg.dust_two_stage_diff_percentile
        min_area = cfg.dust_two_stage_min_area
        fallback_score = cfg.dust_two_stage_fallback_score

        if tile_image is None or anomaly_map is None or dust_mask is None:
            return True, None, [], "TWO_STAGE: missing data -> REAL_NG"

        # --- Prepare images ---
        if len(tile_image.shape) == 3:
            tile_gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        else:
            tile_gray = tile_image.copy()
        tile_h, tile_w = tile_gray.shape

        anomaly_f = np.asarray(anomaly_map, dtype=np.float32)
        anomaly_f = np.maximum(anomaly_f, 0.0)
        h_am, w_am = anomaly_f.shape

        dm = np.asarray(dust_mask, dtype=np.uint8)
        if len(dm.shape) == 3:
            dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
        if dm.shape != (tile_h, tile_w):
            dm = cv2.resize(dm, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)

        # --- Stage 1: Heatmap -> hot zones ---
        pos_vals = anomaly_f[anomaly_f > 0]
        if len(pos_vals) == 0:
            return True, None, [], "TWO_STAGE: no positive heatmap -> REAL_NG"

        hot_thr = np.percentile(pos_vals, 95)
        hot_mask = (anomaly_f >= hot_thr).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        hot_mask = cv2.dilate(hot_mask, kernel, iterations=2)
        n_labels, labels = cv2.connectedComponents(hot_mask, connectivity=8)

        scale = tile_w / w_am
        pad = 20

        # --- Stage 2: Find features on original ---
        all_features = []

        for lid in range(1, n_labels):
            rm = labels == lid
            ys, xs = np.where(rm)
            y1, y2 = int(np.min(ys)), int(np.max(ys))
            x1, x2 = int(np.min(xs)), int(np.max(xs))

            # convert to tile space with padding
            ty1 = max(0, int(y1 * scale) - pad)
            ty2 = min(tile_h, int((y2 + 1) * scale) + pad)
            tx1 = max(0, int(x1 * scale) - pad)
            tx2 = min(tile_w, int((x2 + 1) * scale) + pad)

            crop_gray = tile_gray[ty1:ty2, tx1:tx2]
            crop_dust = dm[ty1:ty2, tx1:tx2]

            if crop_gray.size == 0:
                continue

            # ensure blur kernel is odd and <= crop size
            bk = bg_blur_k
            bk = min(bk, min(crop_gray.shape) - 1)
            if bk % 2 == 0:
                bk += 1
            if bk < 3:
                bk = 3

            blur = cv2.GaussianBlur(crop_gray, (bk, bk), 0)

            # detect dark + bright spots
            for diff, spot_type in [
                (blur.astype(np.float32) - crop_gray.astype(np.float32), "dark"),
                (crop_gray.astype(np.float32) - blur.astype(np.float32), "bright"),
            ]:
                diff_pos = diff[diff > 0]
                if len(diff_pos) < 10:
                    continue
                thr = max(float(np.percentile(diff_pos, diff_pct)), 3.0)
                binary = (diff >= thr).astype(np.uint8) * 255
                morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_k)

                n_feat, feat_labels = cv2.connectedComponents(binary, connectivity=8)
                for fid in range(1, n_feat):
                    fm = feat_labels == fid
                    farea = int(np.count_nonzero(fm))
                    if farea < min_area:
                        continue

                    fys, fxs = np.where(fm)
                    fcy, fcx = int(np.mean(fys)), int(np.mean(fxs))

                    # dust check: use ALL feature pixels
                    feat_dust = crop_dust[fm]
                    dust_overlap = int(np.count_nonzero(feat_dust > 0))
                    feat_dust_ratio = dust_overlap / farea

                    abs_x = tx1 + fcx
                    abs_y = ty1 + fcy

                    all_features.append({
                        "abs_pos": (abs_x, abs_y),
                        "area": farea,
                        "type": spot_type,
                        "dust_ratio": feat_dust_ratio,
                        "is_dust": feat_dust_ratio >= dust_ratio_thr,
                    })

        # --- Verdict ---
        real_features = [f for f in all_features if not f["is_dust"]]
        dust_features = [f for f in all_features if f["is_dust"]]

        if real_features:
            # find peak position of largest real feature
            best = max(real_features, key=lambda f: f["area"])
            bx, by = best["abs_pos"]
            # convert to anomaly_map space for peak_yx
            real_peak_yx = (int(by / scale), int(bx / scale))
            detail = (f"TWO_STAGE: {len(real_features)}real+{len(dust_features)}dust "
                      f"-> REAL_NG (best@({bx},{by}) area={best['area']})")
            return True, real_peak_yx, all_features, detail

        elif not all_features and score >= fallback_score:
            detail = (f"TWO_STAGE: 0features but score={score:.3f}>={fallback_score} "
                      f"-> REAL_NG (fallback)")
            return True, None, all_features, detail

        else:
            detail = (f"TWO_STAGE: 0real+{len(dust_features)}dust "
                      f"-> DUST")
            return False, None, all_features, detail

    def generate_dust_iou_debug_image(
        self,
        tile_image: np.ndarray,
        anomaly_map: np.ndarray,
        dust_mask: np.ndarray,
        heatmap_binary: np.ndarray,
        iou: float,
        top_percent: float,
        is_dust: bool,
        region_details: Optional[list] = None,
        region_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        產生灰塵 IOU 交叉驗證的 Debug 可視化圖

        顯示：
          左上: Heatmap 疊加原圖 (紅色=異常熱區)
          右上: 灰塵遮罩 (黃色=灰塵區域)
          左下: 熱區二值化 (白色=top X% 最紅像素)
          右下: 逐區域判定結果 (紅色=真實缺陷, 綠色=灰塵區域, 藍色=僅灰塵遮罩)

        Args:
            tile_image: 原始 tile 圖片
            anomaly_map: 異常熱圖 (float)
            dust_mask: 灰塵遮罩 (uint8, 255=灰塵)
            heatmap_binary: 二值化熱區遮罩
            iou: 計算出的 IOU 值
            top_percent: 使用的百分位數
            is_dust: 最終判定是否為灰塵
            region_details: 逐區域判定詳情 (from check_dust_per_region)

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
        cv2.putText(panel_bl, f"Top {top_percent:g}%", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

        # --- 右下: 逐區域判定結果 ---
        panel_br = np.zeros((sz, sz, 3), dtype=np.uint8)
        if region_details is not None and heatmap_binary is not None and dust_mask is not None:
            # 使用逐區域判定結果上色 (labels 由 check_dust_per_region 傳入，避免重複計算)
            if region_labels is not None:
                labels = region_labels
            else:
                _, labels = cv2.connectedComponents(heatmap_binary.copy(), connectivity=8)
            num_labels = int(labels.max()) + 1
            region_dust_map = {r["label_id"]: r["is_dust"] for r in region_details}

            # 縮放 labels 到 sz x sz
            labels_resized = cv2.resize(labels.astype(np.float32), (sz, sz),
                                        interpolation=cv2.INTER_NEAREST).astype(np.int32)

            dm = dust_mask
            if len(dm.shape) == 3:
                dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
            dm = cv2.resize(dm, (sz, sz), interpolation=cv2.INTER_NEAREST)

            # 先畫僅灰塵遮罩區域 (藍色)
            hb_resized = cv2.resize(heatmap_binary, (sz, sz), interpolation=cv2.INTER_NEAREST)
            dust_only = (dm > 0) & (hb_resized == 0)
            panel_br[dust_only] = (255, 100, 0)  # 藍色 = 僅灰塵遮罩

            # 逐區域上色
            region_peak_dust_map = {r["label_id"]: r.get("peak_in_dust", True) for r in region_details}
            orig_h, orig_w = heatmap_binary.shape[:2]
            scale_x = sz / orig_w
            scale_y = sz / orig_h
            for label_id in range(1, num_labels):
                region_mask = labels_resized == label_id
                if region_dust_map.get(label_id, False):
                    panel_br[region_mask] = (0, 200, 0)     # 綠色 = 灰塵(已抑制)
                else:
                    panel_br[region_mask] = (0, 0, 255)      # 紅色 = 真實缺陷(保留)

            # 在每個 region 的峰值位置畫標記
            for r in region_details:
                py, px = r["peak_yx"]
                sx = int(px * scale_x)
                sy = int(py * scale_y)
                peak_in = r.get("peak_in_dust", True)
                if peak_in:
                    # 峰值在灰塵上 → 黃色圓點
                    cv2.circle(panel_br, (sx, sy), 4, (0, 255, 255), -1)
                else:
                    # 峰值不在灰塵上 → 白色十字 (關鍵：這是被救回的真實缺陷)
                    cv2.drawMarker(panel_br, (sx, sy), (255, 255, 255),
                                   cv2.MARKER_CROSS, 10, 2)

            real_count = sum(1 for r in region_details if not r["is_dust"])
            dust_count = sum(1 for r in region_details if r["is_dust"])
            verdict_text = f"R:{real_count} D:{dust_count}"
        elif heatmap_binary is not None and dust_mask is not None:
            # Fallback: 舊版整塊分析
            hb = cv2.resize(heatmap_binary, (sz, sz), interpolation=cv2.INTER_NEAREST)
            dm = dust_mask
            if len(dm.shape) == 3:
                dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
            dm = cv2.resize(dm, (sz, sz), interpolation=cv2.INTER_NEAREST)

            heat_only = (hb > 0) & (dm == 0)
            dust_only = (dm > 0) & (hb == 0)
            overlap = (hb > 0) & (dm > 0)

            panel_br[heat_only] = (0, 0, 255)
            panel_br[dust_only] = (255, 100, 0)
            panel_br[overlap]   = (0, 255, 0)
            verdict_text = "DUST" if is_dust else "REAL_NG"
        else:
            verdict_text = "DUST" if is_dust else "REAL_NG"

        metric_name = "COV" if self.config.dust_heatmap_metric == "coverage" else "IOU"
        verdict_color = (0, 200, 255) if is_dust else (0, 0, 255)
        # 顯示 per-region max coverage（實際判定用的值）而非 overall
        if region_details:
            region_max_cov = max(r["coverage"] for r in region_details)
            peak_out_count = sum(1 for r in region_details
                                 if r["coverage"] >= self.config.dust_heatmap_iou_threshold
                                 and not r.get("peak_in_dust", True))
            peak_hint = f" PeakOut:{peak_out_count}" if peak_out_count > 0 else ""
            cv2.putText(panel_br, f"Region{metric_name}:{region_max_cov:.3f} {verdict_text}{peak_hint}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, verdict_color, 1)
        else:
            cv2.putText(panel_br, f"{metric_name}:{iou:.3f} {verdict_text}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, verdict_color, 1)

        # --- 組合 2x2 ---
        top_row = np.hstack([panel_tl, panel_tr])
        bottom_row = np.hstack([panel_bl, panel_br])
        debug_img = np.vstack([top_row, bottom_row])

        return debug_img

    def generate_two_stage_debug_image(
        self,
        tile_image: np.ndarray,
        anomaly_map: np.ndarray,
        dust_mask_no_ext: np.ndarray,
        features: list,
        is_dust: bool,
    ) -> np.ndarray:
        """
        產生兩階段灰塵判定的 Debug 可視化圖

        顯示：
          左上: Heatmap + Hot Zone 框
          右上: 原圖 + Dust Mask (黃, ext=0) + 特徵點標記
          左下: 原圖放大 (hot zone 區域)
          右下: 特徵判定結果 (紅=REAL, 綠=DUST)
        """
        sz = 256

        # --- base image ---
        if len(tile_image.shape) == 2:
            base = cv2.cvtColor(tile_image, cv2.COLOR_GRAY2BGR)
        elif tile_image.shape[2] == 1:
            base = cv2.cvtColor(tile_image, cv2.COLOR_GRAY2BGR)
        else:
            base = tile_image.copy()
        tile_h, tile_w = base.shape[:2]
        base_sm = cv2.resize(base, (sz, sz))

        anomaly_f = np.asarray(anomaly_map, dtype=np.float32)
        anomaly_f = np.maximum(anomaly_f, 0.0)
        h_am, w_am = anomaly_f.shape
        scale_tile = tile_w / w_am

        # --- dust mask prep ---
        dm = np.asarray(dust_mask_no_ext, dtype=np.uint8)
        if len(dm.shape) == 3:
            dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
        if dm.shape != (tile_h, tile_w):
            dm = cv2.resize(dm, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)

        # --- hot zone detection (same as check_dust_two_stage) ---
        pos_vals = anomaly_f[anomaly_f > 0]
        hot_thr = np.percentile(pos_vals, 95) if len(pos_vals) > 0 else 0
        hot_mask = (anomaly_f >= hot_thr).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        hot_mask = cv2.dilate(hot_mask, kernel, iterations=2)

        # --- 左上: Heatmap + Hot Zone ---
        norm = cv2.normalize(anomaly_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm_rsz = cv2.resize(norm, (sz, sz))
        hm_color = cv2.applyColorMap(norm_rsz, cv2.COLORMAP_JET)
        panel_tl = cv2.addWeighted(base_sm, 0.5, hm_color, 0.5, 0)
        # draw hot zone contour
        hot_rsz = cv2.resize(hot_mask, (sz, sz), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(hot_rsz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel_tl, contours, -1, (0, 255, 0), 1)
        cv2.putText(panel_tl, "Heatmap+HotZone", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- 右上: 原圖 + Dust Mask + 特徵標記 ---
        panel_tr = base_sm.copy()
        dm_sm = cv2.resize(dm, (sz, sz), interpolation=cv2.INTER_NEAREST)
        dust_ol = np.zeros_like(panel_tr)
        dust_ol[dm_sm > 0] = (0, 255, 255)
        panel_tr = cv2.addWeighted(panel_tr, 0.6, dust_ol, 0.4, 0)
        sx, sy = sz / tile_w, sz / tile_h
        for feat in features:
            fx, fy = feat["abs_pos"]
            dx, dy = int(fx * sx), int(fy * sy)
            color = (0, 200, 0) if feat["is_dust"] else (0, 0, 255)
            cv2.circle(panel_tr, (dx, dy), 5, color, 2)
            label = "D" if feat["is_dust"] else "R"
            cv2.putText(panel_tr, label, (dx + 7, dy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        cv2.putText(panel_tr, "Features (R=Real G=Dust)", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        # --- 左下: Hot Zone 放大 ---
        # find bounding box of hot zone in tile space
        n_labels, labels = cv2.connectedComponents(hot_mask, connectivity=8)
        if n_labels > 1:
            ys, xs = np.where(labels > 0)
            pad = 20
            zy1 = max(0, int(np.min(ys) * scale_tile) - pad)
            zy2 = min(tile_h, int((np.max(ys) + 1) * scale_tile) + pad)
            zx1 = max(0, int(np.min(xs) * scale_tile) - pad)
            zx2 = min(tile_w, int((np.max(xs) + 1) * scale_tile) + pad)
        else:
            zx1, zy1, zx2, zy2 = 0, 0, tile_w, tile_h

        crop = base[zy1:zy2, zx1:zx2].copy()
        # overlay dust mask on crop
        crop_dm = dm[zy1:zy2, zx1:zx2]
        dust_crop_ol = np.zeros_like(crop)
        dust_crop_ol[crop_dm > 0] = (0, 255, 255)
        crop = cv2.addWeighted(crop, 0.7, dust_crop_ol, 0.3, 0)
        # mark features
        for feat in features:
            fx, fy = feat["abs_pos"]
            lx, ly = fx - zx1, fy - zy1
            if 0 <= lx < crop.shape[1] and 0 <= ly < crop.shape[0]:
                color = (0, 200, 0) if feat["is_dust"] else (0, 0, 255)
                cv2.circle(crop, (lx, ly), 6, color, 2)
                label = f"D{feat['dust_ratio']:.0%}" if feat["is_dust"] else f"R{feat['dust_ratio']:.0%}"
                cv2.putText(crop, label, (lx + 8, ly + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        panel_bl = cv2.resize(crop, (sz, sz))
        cv2.putText(panel_bl, "Zone Zoom (R=Real G=Dust)", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # --- 右下: 判定結果 ---
        panel_br = np.zeros((sz, sz, 3), dtype=np.uint8)
        # dust mask as blue background
        panel_br[dm_sm > 0] = (180, 80, 0)
        # feature areas
        real_count = sum(1 for f in features if not f["is_dust"])
        dust_count = sum(1 for f in features if f["is_dust"])
        for feat in features:
            fx, fy = feat["abs_pos"]
            dx, dy = int(fx * sx), int(fy * sy)
            r = max(3, int(feat["area"] ** 0.5))
            color = (0, 0, 255) if not feat["is_dust"] else (0, 200, 0)
            cv2.circle(panel_br, (dx, dy), r, color, -1)
            cv2.circle(panel_br, (dx, dy), r + 1, (255, 255, 255), 1)

        verdict_color = (0, 0, 255) if not is_dust else (0, 200, 255)
        verdict_text = f"R:{real_count} D:{dust_count}"
        cv2.putText(panel_br, f"TwoStage {verdict_text}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, verdict_color, 1)
        cv2.putText(panel_br, "B=DustMask R=Real G=Dust", (5, sz - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (128, 128, 128), 1)

        # --- 組合 2x2 ---
        top_row = np.hstack([panel_tl, panel_tr])
        bottom_row = np.hstack([panel_bl, panel_br])
        return np.vstack([top_row, bottom_row])

    @staticmethod
    def _select_latest_panel_images(image_files: List[Path]) -> List[Path]:
        """
        當面版資料夾存在重複投片（圖片數量超過上限）時，
        依每個圖片前綴（去除時間戳尾碼後）只保留「最新」的一張。

        命名規則假設: {前綴}_{HHMMSS}.{副檔名}
        例如: G0F00000_104441.tif → 前綴 G0F00000，時間戳 104441。

        使用 st_mtime 排序，避免跨日時 HHMMSS 時間戳倒序 (如 235959 → 000001)。
        """
        from collections import defaultdict

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
            latest = max(files, key=lambda f: f.stat().st_mtime)
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

    def _get_known_image_prefixes(self) -> List[str]:
        """取得所有已知的圖片前綴 (來自 model_mapping + skip_files)"""
        prefixes = set()
        if self._model_mapping:
            prefixes.update(self._model_mapping.keys())
        for sf in self.config.skip_files:
            prefixes.add(sf)
        # 常見固定前綴
        for p in ['G0F00000', 'R0F00000', 'W0F00000', 'WGF00000', 'B0F00000', 'STANDARD']:
            prefixes.add(p)
        return sorted(prefixes, key=len, reverse=True)

    def _parse_aoi_report_txt(self, panel_dir: Path) -> Dict[str, List['AOIReportDefect']]:
        """
        解析 AOI 機台 NG 報告 TXT。

        1. panel_dir 路徑替換 (預設 yuantu→Report) 取得報告目錄
        2. 找到最新 .TXT 檔
        3. 解析第二行的 NG 缺陷字串

        格式: NG{異常代碼}{10位座標}{8字元前綴}...
        例: NGPCDK20028800554W0F00000PCDK20171100894B0F00000

        Returns:
            {image_prefix: [AOIReportDefect, ...]}
        """
        result_map: Dict[str, List[AOIReportDefect]] = {}

        # 路徑替換取得報告目錄
        replace_from = self.config.aoi_report_path_replace_from
        replace_to = self.config.aoi_report_path_replace_to
        panel_str = str(panel_dir)

        if replace_from not in panel_str:
            logger.warning(f"AOI Report: 路徑中找不到 '{replace_from}': {panel_str}")
            return result_map

        report_dir = Path(panel_str.replace(replace_from, replace_to, 1))

        if not report_dir.exists():
            logger.info(f"AOI Report: 報告目錄不存在: {report_dir}")
            return result_map

        # 找最新的 .TXT 檔案 (依 st_mtime 排序，避免跨日時 HHMMSS 時間戳倒序)
        # 用 set 去重，避免 Windows 大小寫不敏感時 *.TXT 與 *.txt 回傳相同檔案
        txt_files = list({f.resolve() for f in
            list(report_dir.glob("*.TXT")) + list(report_dir.glob("*.txt"))})
        if not txt_files:
            logger.info(f"AOI Report: 報告目錄中無 TXT 檔案: {report_dir}")
            return result_map

        report_file = max(txt_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"AOI Report: 讀取報告 {report_file.name}")

        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if len(lines) < 2:
                logger.warning(f"AOI Report: 報告內容不足 2 行: {report_file}")
                return result_map

            # 第二行以 @ 開頭，; 分隔
            line2 = lines[1].strip().rstrip(',')
            if not line2.startswith('@'):
                logger.warning(f"AOI Report: 第二行格式異常 (不以@開頭): {line2[:50]}")
                return result_map

            # 以 ; 分隔，找到以 NG 開頭的欄位
            fields = line2.split(';')
            ng_string = None
            for field in fields:
                field = field.strip()
                if field.startswith('NG') and len(field) > 2:
                    ng_string = field[2:]  # 去掉 NG 前綴
                    break

            if not ng_string:
                logger.info(f"AOI Report: 報告中未發現 NG 記錄")
                return result_map

            # 用已知前綴建構 regex 解析缺陷記錄
            # 格式: {異常代碼}{10位座標}{8字元前綴}
            known_prefixes = self._get_known_image_prefixes()
            if not known_prefixes:
                logger.warning("AOI Report: 無已知圖片前綴，無法解析")
                return result_map

            prefix_pattern = '|'.join(re.escape(p) for p in known_prefixes)
            pattern = re.compile(r'([A-Za-z0-9]+?)(\d{10})(' + prefix_pattern + r')')
            matches = pattern.findall(ng_string)

            if not matches:
                logger.warning(f"AOI Report: 無法解析缺陷記錄: {ng_string[:80]}")
                return result_map

            for defect_code, coord_str, image_prefix in matches:
                product_x = int(coord_str[:5])
                product_y = int(coord_str[5:])

                defect = AOIReportDefect(
                    defect_code=defect_code,
                    product_x=product_x,
                    product_y=product_y,
                    image_prefix=image_prefix,
                )

                if image_prefix not in result_map:
                    result_map[image_prefix] = []
                result_map[image_prefix].append(defect)

            total = sum(len(v) for v in result_map.values())
            per_prefix = ", ".join(f"{p}×{len(v)}" for p, v in result_map.items())
            logger.info(
                f"AOI Report: 解析到 {total} 筆缺陷 (涉及 {len(result_map)} 種圖片前綴) [{per_prefix}]"
            )
            for prefix, defects in result_map.items():
                for d in defects:
                    logger.debug(f"  🎯 {d.defect_code} @ ({d.product_x}, {d.product_y}) → {prefix}")

        except Exception as e:
            logger.error(f"AOI Report: 解析失敗: {e}")

        return result_map

    def _create_aoi_coord_tiles(
        self,
        image: np.ndarray,
        result: 'ImageResult',
        aoi_defects: List['AOIReportDefect'],
        product_resolution: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List['TileInfo'], List['AOIReportDefect']]:
        """
        以 AOI 機檢座標為中心切取 512x512 tile。

        Args:
            image: 原始圖片
            result: 已預處理的 ImageResult (含 otsu_bounds, raw_bounds)
            aoi_defects: 該圖片對應的 AOI 報告缺陷列表
            product_resolution: 產品解析度

        Returns:
            (patchcore_tiles, edge_defects_for_cv)
            - patchcore_tiles: 可做 PatchCore 推論的 tiles
            - edge_defects_for_cv: 碰到邊緣無法完整切塊的 defects (需 CV 處理)
        """
        tile_size = self.config.tile_size
        half = tile_size // 2
        patchcore_tiles = []
        edge_defects = []
        is_skip_file = self.config.should_skip_file(result.image_path.name)

        if result.raw_bounds is None:
            logger.warning("AOI Coord: raw_bounds 為 None，無法建立切塊")
            return patchcore_tiles, edge_defects

        otsu_x1, otsu_y1, otsu_x2, otsu_y2 = result.otsu_bounds
        img_h, img_w = image.shape[:2]

        # 需要 tile_id 從現有 tiles 之後遞增
        next_tile_id = max((t.tile_id for t in result.tiles), default=-1) + 1

        for defect in aoi_defects:
            # 產品座標 → 圖片座標
            img_x, img_y = self._map_aoi_coords(
                defect.product_x, defect.product_y,
                result.raw_bounds, product_resolution
            )

            # 計算 tile 起點 (以座標為中心)
            tx = img_x - half
            ty = img_y - half

            # 檢查是否能完整放入 Otsu bounds 內
            otsu_width = otsu_x2 - otsu_x1
            otsu_height = otsu_y2 - otsu_y1

            if otsu_width < tile_size or otsu_height < tile_size:
                # 產品區域太小，無法放入 512x512 tile
                edge_defects.append(defect)
                print(f"  ⚠️ AOI Coord ({defect.defect_code}) @ ({img_x},{img_y}): 產品區域太小，轉 CV 處理")
                continue

            # 判定是否碰到邊緣
            at_edge = (
                img_x - otsu_x1 < half or
                otsu_x2 - img_x < half or
                img_y - otsu_y1 < half or
                otsu_y2 - img_y < half
            )

            if at_edge:
                if is_skip_file:
                    # skip_files (如 B0F00000) 使用二值化偵測，不依賴 PatchCore 模型
                    # 對邊緣 tile 用 clamping 建立即可，不需轉 CV 處理
                    print(f"  🎯 AOI Coord ({defect.defect_code}) @ ({img_x},{img_y}): 碰到邊緣，skip_file 模式仍建立 tile (clamping)")
                else:
                    edge_defects.append(defect)
                    print(f"  📐 AOI Coord ({defect.defect_code}) @ ({img_x},{img_y}): 碰到邊緣，轉 CV 處理")
                    continue

            # 邊界 clamp (確保不超出圖片範圍)
            tx = max(0, min(tx, img_w - tile_size))
            ty = max(0, min(ty, img_h - tile_size))

            # 切取 tile
            tile_img = image[ty:ty + tile_size, tx:tx + tile_size].copy()

            if tile_img.shape[0] != tile_size or tile_img.shape[1] != tile_size:
                edge_defects.append(defect)
                print(f"  ⚠️ AOI Coord ({defect.defect_code}) @ ({img_x},{img_y}): 切塊尺寸異常 {tile_img.shape}, 轉 CV 處理")
                continue

            # 判定邊緣旗標
            is_top = (ty <= otsu_y1 + tile_size)
            is_bottom = (ty + tile_size >= otsu_y2 - tile_size)
            is_left = (tx <= otsu_x1 + tile_size)
            is_right = (tx + tile_size >= otsu_x2 - tile_size)

            tile = TileInfo(
                tile_id=next_tile_id,
                x=tx,
                y=ty,
                width=tile_size,
                height=tile_size,
                image=tile_img,
                is_bottom_edge=is_bottom,
                is_top_edge=is_top,
                is_left_edge=is_left,
                is_right_edge=is_right,
                is_aoi_coord_tile=True,
                aoi_defect_code=defect.defect_code,
                aoi_product_x=defect.product_x,
                aoi_product_y=defect.product_y,
            )

            patchcore_tiles.append(tile)
            next_tile_id += 1
            logger.debug(f"  🎯 AOI Coord ({defect.defect_code}) @ ({img_x},{img_y}) → Tile ({tx},{ty}) {tile_size}x{tile_size}")

        return patchcore_tiles, edge_defects

    def _map_aoi_coords(self, px: int, py: int, raw_bounds: Tuple[int, int, int, int],
                         product_resolution: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """將產品座標映射到圖片座標"""
        if product_resolution is None:
            product_resolution = DEFAULT_PRODUCT_RESOLUTION
        PRODUCT_WIDTH, PRODUCT_HEIGHT = product_resolution
        
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

    def _inspect_roi_patchcore(
        self,
        image: np.ndarray,
        img_x: int,
        img_y: int,
        img_prefix: str,
        panel_polygon: Optional[np.ndarray] = None,
    ) -> Tuple[List[EdgeDefect], Dict[str, Any]]:
        """用 PatchCore 做 AOI 座標邊緣 ROI 推論。

        ROI 中心對齊 AOI 座標 + 黑 pad (panel 外本來就是黑色) + fg_mask 遮罩
        anomaly_map (panel 外歸零)。TileInfo 不標 is_*_edge，避免 edge_margin
        decay 在近 ROI 邊緣誤抑 AOI 中心的 defect。

        Returns:
            (defects, stats)
            - defects: 若 NG 則 1 個 EdgeDefect，否則空 list
            - stats: {"score", "threshold", "area", "min_area", "ok_reason"}
        """
        tile_size = self.config.tile_size
        half = tile_size // 2
        img_h, img_w = image.shape[:2]

        # 中心對齊 ROI (image 外的像素留 0 — panel 外本來就是黑色)
        rx1 = img_x - half
        ry1 = img_y - half
        rx2 = rx1 + tile_size
        ry2 = ry1 + tile_size

        # image 內可擷取的子區 (src) 與貼到 ROI 的位置 (dst)
        sx1 = max(0, rx1)
        sy1 = max(0, ry1)
        sx2 = min(img_w, rx2)
        sy2 = min(img_h, ry2)

        # 若 ROI 完全在 image 外 → 無法推論
        if sx2 <= sx1 or sy2 <= sy1:
            stats = {"score": 0.0, "threshold": 0.0, "area": 0, "min_area": 0,
                     "ok_reason": "ROI out of image",
                     "roi": None, "fg_mask": None, "anomaly_map": None}
            return [], stats

        dx1 = sx1 - rx1
        dy1 = sy1 - ry1
        dx2 = dx1 + (sx2 - sx1)
        dy2 = dy1 + (sy2 - sy1)

        channels = image.shape[2] if image.ndim == 3 else 1
        if channels == 1:
            roi = np.zeros((tile_size, tile_size), dtype=image.dtype)
            roi[dy1:dy2, dx1:dx2] = image[sy1:sy2, sx1:sx2]
        else:
            roi = np.zeros((tile_size, tile_size, channels), dtype=image.dtype)
            roi[dy1:dy2, dx1:dx2] = image[sy1:sy2, sx1:sx2]

        # fg_mask: ROI 內落在 panel polygon 內的像素 = 255，panel 外 = 0
        fg_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
        if panel_polygon is not None:
            local_poly = panel_polygon.copy().astype(np.float32)
            local_poly[:, 0] -= rx1
            local_poly[:, 1] -= ry1
            cv2.fillPoly(fg_mask, [local_poly.astype(np.int32)], 255)
        else:
            # 無 polygon 時 fallback: ROI 在 image 內的區塊視為前景
            fg_mask[dy1:dy2, dx1:dx2] = 255

        # 取 inferencer / threshold (沿用中央 tile pipeline 的 prefix 路由)
        inferencer = self._get_inferencer_for_prefix(img_prefix)
        threshold = self._get_threshold_for_prefix(img_prefix)

        if inferencer is None:
            stats = {"score": 0.0, "threshold": threshold, "area": 0, "min_area": 0,
                     "ok_reason": f"No model for prefix {img_prefix}",
                     "roi": roi, "fg_mask": fg_mask, "anomaly_map": None}
            return [], stats

        tile = TileInfo(
            tile_id=-1,
            x=rx1, y=ry1,
            width=tile_size, height=tile_size,
            image=roi,
            mask=fg_mask,
            is_top_edge=False, is_bottom_edge=False,
            is_left_edge=False, is_right_edge=False,
        )

        score, anomaly_map = self.predict_tile(
            tile, inferencer=inferencer, threshold=threshold,
        )

        max_area = _anomaly_max_cc_area(anomaly_map)

        min_area = int(getattr(self.config, "patchcore_min_area", 10))
        is_ng = score >= threshold

        stats = {
            "score": float(score),
            "threshold": float(threshold),
            "area": int(max_area),
            "min_area": min_area,
            "roi": roi,
            "fg_mask": fg_mask,
            "anomaly_map": anomaly_map,
        }

        if is_ng:
            defect = EdgeDefect(
                side="aoi_edge",
                area=int(max_area),
                bbox=(rx1, ry1, tile_size, tile_size),
                center=(img_x, img_y),
                max_diff=0,  # PatchCore path 不用灰階差
                solidity=1.0,
                inspector_mode="patchcore",
                patchcore_score=float(score),
                patchcore_threshold=float(threshold),
                patchcore_ok_reason="",
            )
            defect.pc_roi = roi
            defect.pc_fg_mask = fg_mask
            defect.pc_anomaly_map = anomaly_map
            stats["ok_reason"] = ""
            return [defect], stats

        # OK — 推算原因
        if score < threshold:
            reason = "Score<Thr"
        elif max_area < min_area:
            reason = "Area<Min"
        else:
            reason = ""
        stats["ok_reason"] = reason
        return [], stats

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

    def _match_bomb_defect_code(self, bomb_info: Dict[str, Any]) -> str:
        """
        從 config 的 bomb_defects 中查找匹配的 defect_code
        
        匹配策略:
            1. 比對 image_prefix
            2. 比對 defect_type
            3. 找到 → 返回 defect_code；找不到 → 返回 "UNKNOWN"
        """
        target_prefix = bomb_info["image_prefix"]
        target_type = bomb_info["defect_type"]
        
        for bomb in self.config.bomb_defects:
            if (bomb.image_prefix == target_prefix and 
                bomb.defect_type == target_type):
                return bomb.defect_code
        
        # 若只有 prefix 匹配 (不分 type)，也可以 fallback
        for bomb in self.config.bomb_defects:
            if bomb.image_prefix == target_prefix:
                return bomb.defect_code
        
        return "UNKNOWN"

    def check_bomb_match(
        self,
        image_prefix: str,
        tile_center_x: int,
        tile_center_y: int,
        raw_bounds: Tuple[int, int, int, int],
        anomaly_map: Optional[np.ndarray] = None,
        product_resolution: Optional[Tuple[int, int]] = None,
        bomb_list: Optional[List] = None,
        skip_shape_check: bool = False,
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
        if product_resolution is None:
            product_resolution = DEFAULT_PRODUCT_RESOLUTION
        tolerance = self.config.bomb_match_tolerance
    
        # 使用傳入的 bomb_list 或 config 預設值
        bombs = bomb_list if bomb_list is not None else self.config.bomb_defects
    
        for bomb in bombs:
            # 比對前綴 (支援帶時間戳的檔名, e.g. "G0F00000" 匹配 "G0F00000_031447")
            if not (image_prefix == bomb.image_prefix or 
                    image_prefix.startswith(bomb.image_prefix + "_")):
                continue
            
            if bomb.defect_type == "line" and len(bomb.coordinates) >= 2:
                # 豎線型: 將兩端座標轉換，判斷 tile 是否在緩衝帶內
                pt1 = bomb.coordinates[0]
                pt2 = bomb.coordinates[1]
                img_x1, img_y1 = self._map_aoi_coords(pt1[0], pt1[1], raw_bounds, product_resolution)
                img_x2, img_y2 = self._map_aoi_coords(pt2[0], pt2[1], raw_bounds, product_resolution)
                
                # 線段 x 範圍 ± tolerance (轉換到圖片座標的 tolerance)
                product_width = product_resolution[0]
                x_start, _, x_end, _ = raw_bounds
                scale_x = (x_end - x_start) / product_width
                img_tolerance_x = int(tolerance * scale_x)
                
                min_x = min(img_x1, img_x2) - img_tolerance_x
                max_x = max(img_x1, img_x2) + img_tolerance_x
                min_y = min(img_y1, img_y2)
                max_y = max(img_y1, img_y2)
                
                if min_x <= tile_center_x <= max_x and min_y <= tile_center_y <= max_y:
                    # 額外驗證：heatmap 是否呈現線狀形態
                    if anomaly_map is not None and not skip_shape_check:
                        is_line, aspect_ratio = self._check_heatmap_line_shape(
                            anomaly_map,
                            min_aspect_ratio=self.config.bomb_line_min_aspect_ratio,
                        )
                        if not is_line:
                            print(f"⚠️ BOMB line 位置匹配但熱力圖非線狀 (aspect_ratio={aspect_ratio:.2f} < {self.config.bomb_line_min_aspect_ratio})，跳過 {bomb.defect_code}")
                            continue
                    return True, bomb.defect_code
                    
            elif bomb.defect_type == "point":
                # 點型: 判斷 tile 中心是否在任一炸彈點座標 ± tolerance 範圍內
                product_width, product_height = product_resolution
                x_start, y_start, x_end, y_end = raw_bounds
                scale_x = (x_end - x_start) / product_width
                scale_y = (y_end - y_start) / product_height
                img_tolerance_x = int(tolerance * scale_x)
                img_tolerance_y = int(tolerance * scale_y)
                
                for coord in bomb.coordinates:
                    img_bx, img_by = self._map_aoi_coords(coord[0], coord[1], raw_bounds, product_resolution)
                    if (abs(tile_center_x - img_bx) <= img_tolerance_x and
                        abs(tile_center_y - img_by) <= img_tolerance_y):
                        return True, bomb.defect_code
        
        return False, ""

    def process_panel(
        self, 
        panel_dir: Path, 
        progress_callback=None,
        cpu_workers: int = 4,
        product_resolution: Optional[Tuple[int, int]] = None,
        bomb_info: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
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
            print(f"   ✅ 選用: {', '.join(f.name for f in image_files)}")
        
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
        
        # === 統一參考邊界 (Panel 級)：依優先序挑一張圖計算，套用到所有圖片 ===
        # 同一 panel 的所有圖片皆為同一位置拍攝，理論邊界相同；各圖獨立 OTSU
        # 在光源較弱的機種（如 G0F / R0F / STANDARD）會抓歪 polygon，連帶影響
        # tiling / mask / exclusion。這裡以 W0F (白光) 優先作為參考，其他圖套用。
        # B0F (暗色) 本來就無法獨立偵測邊界，同樣走這條路徑。
        _DARK_PREFIXES = ("B0F",)  # 不得作為 reference 的暗色圖片前綴
        _REFERENCE_PRIORITY = ("W0F", "WGF", "G0F", "R0F", "STANDARD")

        def _prefix_rank(filename: str) -> int:
            up = filename.upper()
            for i, p in enumerate(_REFERENCE_PRIORITY):
                if up.startswith(p):
                    return i
            return len(_REFERENCE_PRIORITY)  # 兜底：其他非暗色非 OMIT

        panel_reference_raw_bounds: Optional[Tuple[int, int, int, int]] = None
        panel_reference_polygon: Optional[np.ndarray] = None

        ref_candidates = sorted(
            [f for f in normal_files
             if not f.name.upper().startswith(_DARK_PREFIXES)
             and not is_dust_check_image(f)],
            key=lambda f: _prefix_rank(f.name),
        )
        for ref_path in ref_candidates:
            try:
                ref_img = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED)
                if ref_img is None:
                    continue
                ref_bounds, ref_binary = self._find_raw_object_bounds(ref_img)
                panel_reference_raw_bounds = ref_bounds
                if self.config.enable_panel_polygon:
                    panel_reference_polygon = self._find_panel_polygon(
                        ref_binary, ref_bounds
                    )
                    poly_str = "有" if panel_reference_polygon is not None else "品質不足"
                else:
                    poly_str = "關閉"
                print(f"📐 統一參考邊界 (Panel 級) 已從 {ref_path.name} 計算 → "
                      f"{panel_reference_raw_bounds} (polygon: {poly_str})")
                break
            except Exception as e:
                print(f"⚠️ 計算參考邊界失敗 ({ref_path.name}): {e}")
                continue
        if panel_reference_raw_bounds is None:
            print("⚠️ 無法計算統一參考邊界，所有圖片將各自計算 OTSU (可能不一致)")

        # 過濾出需要處理的檔案
        files_to_process = [f for f in normal_files if not self.config.should_skip_file(f.name)]
        skipped = [f.name for f in normal_files if self.config.should_skip_file(f.name)]
        if skipped:
            print(f"⏭️ 跳過檔案 (設定) ×{len(skipped)}: {', '.join(skipped)}")
        
        total_files = len(files_to_process)
        if total_files == 0:
            return results, omit_vis, omit_overexposed, omit_overexposure_info, False, omit_image, {}
        
        # 決定實際 worker 數量 (不超過檔案數量)
        actual_workers = min(cpu_workers, total_files)
        print(f"🔀 平行處理: {total_files} 張圖片, {actual_workers} 個 CPU 執行緒")
        
        # ================================================================
        # Phase 1: 多執行緒平行預處理 (imread + Otsu + tiling)
        # OpenCV 在 C 層釋放 GIL，多執行緒可獲得真正的平行加速
        # ================================================================
        def _preprocess_one(img_path: Path) -> Optional[ImageResult]:
            """單張圖片的預處理 (可安全在多執行緒中呼叫)"""
            # 所有圖片套用統一參考邊界 (若計算成功)
            result = self.preprocess_image(
                img_path,
                cached_mark=cached_mark,
                reference_raw_bounds=panel_reference_raw_bounds,
                reference_polygon=panel_reference_polygon,
            )
            if result is None:
                return None
            
            # 整合 AOI Defect Data
            stem = img_path.stem
            if stem in defect_map and result.raw_bounds is not None:
                raw_bounds = result.raw_bounds
                img_w, img_h = result.image_size
                
                for d in defect_map[stem]:
                    img_x, img_y = self._map_aoi_coords(d['x'], d['y'], raw_bounds, product_resolution)
                    
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
        # Phase 1.5: AOI 機檢座標目標切塊
        # 解析 AOI 機台 NG 報告，以缺陷座標為中心建立額外的 512x512 tiles
        # ================================================================
        aoi_report = {}
        if self.config.aoi_coord_inspection_enabled:
            aoi_report = self._parse_aoi_report_txt(panel_dir)
            if aoi_report:
                aoi_tile_count = 0
                aoi_edge_count = 0

                # 收集已有的圖片前綴
                existing_prefixes = set()
                for result in preprocessed_results:
                    existing_prefixes.add(self._get_image_prefix(result.image_path.name))

                # 對 skip_files 中有 AOI 報告的圖片，預處理後加入（僅保留 AOI coord tiles）
                for report_prefix in aoi_report:
                    if report_prefix not in existing_prefixes:
                        # 在所有圖片檔（含被 skip 的）中找對應圖片
                        matched_file = None
                        skipped_files = [f for f in image_files if self.config.should_skip_file(f.name) and not is_dust_check_image(f)]
                        for f in skipped_files:
                            if self._get_image_prefix(f.name) == report_prefix:
                                matched_file = f
                                break
                        if matched_file is not None:
                            print(f"🎯 AOI Coord: 為跳過的圖片 {matched_file.name} 建立預處理 (有 {len(aoi_report[report_prefix])} 筆 AOI 座標)")
                            # 套用統一參考邊界 (若計算成功)
                            skip_result = self.preprocess_image(
                                matched_file,
                                cached_mark=cached_mark,
                                reference_raw_bounds=panel_reference_raw_bounds,
                                reference_polygon=panel_reference_polygon,
                            )
                            if skip_result is not None:
                                # 清除 grid tiles，只保留結構供 AOI coord 使用
                                skip_result.tiles = []
                                skip_result.excluded_tile_count = 0
                                skip_result.processed_tile_count = 0
                                preprocessed_results.append(skip_result)
                                existing_prefixes.add(report_prefix)

                for result in preprocessed_results:
                    img_prefix = self._get_image_prefix(result.image_path.name)
                    if img_prefix in aoi_report:
                        aoi_image = cv2.imread(str(result.image_path), cv2.IMREAD_UNCHANGED)
                        if aoi_image is not None:
                            new_tiles, edge_defs = self._create_aoi_coord_tiles(
                                aoi_image, result, aoi_report[img_prefix], product_resolution
                            )
                            result.tiles.extend(new_tiles)
                            aoi_tile_count += len(new_tiles)
                            aoi_edge_count += len(edge_defs)
                            # 邊緣 defects: AOI 座標落在產品邊緣，無法切 512x512 tile
                            # 無論 CV edge inspection 是否啟用，都要建立 EdgeDefect 以追蹤判定
                            inspector_mode = getattr(
                                self.edge_inspector.config, "aoi_edge_inspector", "cv"
                            ) if getattr(self, 'edge_inspector', None) else "cv"
                            img_prefix = self._get_image_prefix(result.image_path.name)

                            for edef in edge_defs:
                                img_x, img_y = self._map_aoi_coords(
                                    edef.product_x, edef.product_y,
                                    result.raw_bounds, product_resolution
                                )
                                roi_size = self.config.tile_size
                                roi_half = roi_size // 2
                                img_h, img_w = aoi_image.shape[:2]
                                rx1 = max(0, img_x - roi_half)
                                ry1 = max(0, img_y - roi_half)
                                rx2 = min(img_w, img_x + roi_half)
                                ry2 = min(img_h, img_y + roi_half)

                                detected = False

                                if inspector_mode == "patchcore":
                                    try:
                                        pc_defects, pc_stats = self._inspect_roi_patchcore(
                                            aoi_image, img_x, img_y, img_prefix,
                                            panel_polygon=result.panel_polygon,
                                        )
                                        if pc_defects:
                                            merged = pc_defects[0]
                                            # 強制 center 為 AOI 座標以確保 BOMB 比對一致
                                            merged.center = (img_x, img_y)
                                            result.edge_defects.append(merged)
                                            detected = True
                                            print(f"  🔍 AOI Coord PatchCore edge ({edef.defect_code}) "
                                                  f"@ ({img_x},{img_y}): score={pc_stats.get('score', 0):.3f} "
                                                  f">= thr={pc_stats.get('threshold', 0):.3f}, "
                                                  f"area={pc_stats.get('area', 0)}")
                                        if not detected:
                                            ok_defect = EdgeDefect(
                                                side="aoi_coord_ok",
                                                area=int(pc_stats.get("area", 0)),
                                                bbox=(rx1, ry1, rx2 - rx1, ry2 - ry1),
                                                center=(img_x, img_y),
                                                max_diff=0,
                                                is_cv_ok=True,
                                                inspector_mode="patchcore",
                                                patchcore_score=float(pc_stats.get("score", 0.0)),
                                                patchcore_threshold=float(pc_stats.get("threshold", 0.0)),
                                                patchcore_ok_reason=str(pc_stats.get("ok_reason", "")),
                                            )
                                            ok_defect.pc_roi = pc_stats.get("roi")
                                            ok_defect.pc_fg_mask = pc_stats.get("fg_mask")
                                            ok_defect.pc_anomaly_map = pc_stats.get("anomaly_map")
                                            result.edge_defects.append(ok_defect)
                                            print(f"  ✅ AOI Coord PatchCore edge ({edef.defect_code}) "
                                                  f"@ ({img_x},{img_y}): OK "
                                                  f"(score={pc_stats.get('score', 0):.3f}, "
                                                  f"thr={pc_stats.get('threshold', 0):.3f}, "
                                                  f"reason={pc_stats.get('ok_reason', '')})")
                                    except Exception as e:
                                        logger.warning(f"AOI Coord PatchCore edge 失敗 ({edef.defect_code}): {e}")
                                    continue

                                # CV 路徑 (現行)
                                roi = aoi_image[ry1:ry2, rx1:rx2]
                                roi_stats = {"max_diff": 0, "max_area": 0, "threshold": 0, "min_area": 0}
                                if roi.size > 0 and getattr(self, 'edge_inspector', None):
                                    try:
                                        edge_results, roi_stats = self.edge_inspector.inspect_roi(
                                            roi, offset_x=rx1, offset_y=ry1,
                                            otsu_bounds=result.otsu_bounds,
                                            panel_polygon=result.panel_polygon,
                                        )
                                        if edge_results:
                                            # 合併為單一 EdgeDefect (以 AOI 座標為中心)
                                            # 避免拆成多筆小 defect 導致 BOMB 比對時部分 center 偏離
                                            unified_bbox = (rx1, ry1, rx2 - rx1, ry2 - ry1)
                                            total_area = sum(ed.area for ed in edge_results)
                                            worst_diff = max(ed.max_diff for ed in edge_results)
                                            merged = EdgeDefect(
                                                side="aoi_edge",
                                                area=total_area,
                                                bbox=unified_bbox,
                                                center=(img_x, img_y),  # 使用 AOI 座標中心，確保 BOMB 比對一致
                                                max_diff=worst_diff,
                                                threshold_used=roi_stats.get("threshold", 0),
                                                min_area_used=roi_stats.get("min_area", 0),
                                                inspector_mode="cv",
                                            )
                                            result.edge_defects.append(merged)
                                            detected = True
                                            print(f"  🔍 AOI Coord CV edge ({edef.defect_code}) @ ({img_x},{img_y}): "
                                                  f"偵測到 {len(edge_results)} 個缺陷 → 合併為 1 筆 (area={total_area}, diff={worst_diff})")
                                    except Exception as e:
                                        logger.warning(f"AOI Coord CV edge 檢測失敗 ({edef.defect_code}): {e}")

                                if not detected:
                                    # 建立 OK 記錄，帶入實際計算的 max_diff / max_area
                                    ok_defect = EdgeDefect(
                                        side="aoi_coord_ok",
                                        area=roi_stats.get("max_area", 0),
                                        bbox=(rx1, ry1, rx2 - rx1, ry2 - ry1),
                                        center=(img_x, img_y),
                                        max_diff=roi_stats.get("max_diff", 0),
                                        is_cv_ok=True,
                                        threshold_used=roi_stats.get("threshold", 0),
                                        min_area_used=roi_stats.get("min_area", 0),
                                        inspector_mode="cv",
                                    )
                                    result.edge_defects.append(ok_defect)
                                    print(f"  ✅ AOI Coord edge ({edef.defect_code}) @ ({img_x},{img_y}): CV 未偵測到缺陷，判定 OK"
                                          f"（max_diff={roi_stats.get('max_diff', 0)}, max_area={roi_stats.get('max_area', 0)}, "
                                          f"thr={roi_stats.get('threshold', 0)}, min_area={roi_stats.get('min_area', 0)}）")
                print(f"🎯 Phase 1.5 完成: AOI 座標新增 {aoi_tile_count} 個 tiles, {aoi_edge_count} 個邊緣 defects")

        # === Grid Tiling 開關控制 ===
        # 如果 grid_tiling_enabled=False，移除非 AOI coord 的 tiles (只推論 AOI 座標 tiles)
        if not self.config.grid_tiling_enabled:
            for result in preprocessed_results:
                original_count = len(result.tiles)
                result.tiles = [t for t in result.tiles if t.is_aoi_coord_tile]
                removed = original_count - len(result.tiles)
                if removed > 0:
                    print(f"⏭️ Grid Tiling 關閉: {result.image_path.name} 移除 {removed} 個 grid tiles，保留 {len(result.tiles)} 個 AOI tiles")

        # ================================================================
        # Phase 2: 序列 GPU 推論 (predict_tile)
        # PyTorch GPU 推論不適合跨執行緒平行化，保持序列執行
        # ================================================================
        inference_start = time.time()
        for i, result in enumerate(preprocessed_results):
            # 多模型路由：依圖片前綴選擇對應的 inferencer 和 threshold
            img_prefix = self._get_image_prefix(result.image_path.name)

            # === skip_files 圖片（如 B0F00000）：使用二值化偵測取代 PatchCore ===
            if self.config.should_skip_file(result.image_path.name):
                print(f"💡 {result.image_path.name} (skip_file) → 使用二值化偵測亮點")
                anomaly_tiles = []
                for tile in result.tiles:
                    score, anomaly_map = self._detect_bright_spots(tile)
                    if score <= 0:
                        # 未偵測到亮點，標記為 below_threshold 不影響判定，但仍保留以便查看原圖
                        tile.is_aoi_coord_below_threshold = True
                    anomaly_tiles.append((tile, score, anomaly_map))
                result.anomaly_tiles = anomaly_tiles
                result.inference_time = 0.0
                preprocessed_results[i] = result
                if progress_callback:
                    progress_callback(i + 1, len(preprocessed_results))
                continue

            target_inferencer = self._get_inferencer_for_prefix(img_prefix)
            target_threshold = self._get_threshold_for_prefix(img_prefix)

            if target_inferencer is None:
                print(f"⚠️ {result.image_path.name} 無可用模型，跳過推論")
                continue

            # 模型路由 log (僅在多模型模式下顯示)
            if self._model_mapping:
                model_name = "fallback"
                if img_prefix in self._model_mapping:
                    model_name = self._model_mapping[img_prefix].name
                print(f"🎯 {result.image_path.name} → 模型: {model_name}, 閾值: {target_threshold}")

            result = self.run_inference(
                result,
                inferencer=target_inferencer,
                threshold=target_threshold,
                model_id=model_id,
            )
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

            # skip_files 圖片（如 B0F00000）不做 OMIT 灰塵比對
            if self.config.should_skip_file(img_path.name):
                return result

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
                        
                        # Step B: 逐區域灰塵交叉驗證 (Per-Region Dust Filtering)
                        iou = 0.0
                        heatmap_binary = None
                        top_pct = self.config.dust_heatmap_top_percent
                        metric_mode = self.config.dust_heatmap_metric
                        metric_name = "COV" if metric_mode == "coverage" else "IOU"

                        if is_dust and anomaly_map is not None:
                            # 逐區域判定：拆開異常連通區域，各自與灰塵比對
                            has_real_defect, real_peak_yx, overall_iou, region_details, heatmap_binary, region_labels = \
                                self.check_dust_per_region(
                                    dust_mask, anomaly_map,
                                    top_percent=top_pct,
                                    metric=metric_mode,
                                    iou_threshold=self.config.dust_heatmap_iou_threshold,
                                )
                            iou = overall_iou
                            tile.dust_heatmap_iou = iou
                            # 記錄 per-region 最大 coverage（實際判定用的值）
                            tile.dust_region_details = region_details
                            tile.dust_heatmap_binary = heatmap_binary
                            if region_details:
                                tile.dust_region_max_cov = max(r["coverage"] for r in region_details)

                            dust_regions = [r for r in region_details if r["is_dust"]]
                            real_regions = [r for r in region_details if not r["is_dust"]]

                            _two_stage_ran = False
                            _ts_features = []
                            _ts_dust_mask_no_ext = None

                            if has_real_defect:
                                # 有非灰塵的真實異常區域 → 保留為 NG
                                tile.is_suspected_dust_or_scratch = False
                                detail_text += (
                                    f" PER_REGION: {len(real_regions)}real+"
                                    f"{len(dust_regions)}dust -> REAL_NG"
                                )

                                # 更新 peak 座標到非灰塵區域的最大值位置
                                if real_peak_yx is not None:
                                    amap_h, amap_w = anomaly_map.shape[:2]
                                    tile.anomaly_peak_y = tile.y + int(real_peak_yx[0] * tile.height / amap_h)
                                    tile.anomaly_peak_x = tile.x + int(real_peak_yx[1] * tile.width / amap_w)
                            else:
                                # 所有異常區域都與灰塵重疊 → 初步標記為灰塵
                                # 如果啟用兩階段判定，進行二次確認
                                if self.config.dust_two_stage_enabled:
                                    # 兩階段: 用原圖精準定位 feature 點，比對 dust_mask (ext=0)
                                    dust_mask_no_ext = None
                                    if omit_crop is not None:
                                        _, dust_mask_no_ext, _, _ = self.check_dust_or_scratch_feature(
                                            omit_crop, extension_override=0)
                                    ts_has_real, ts_peak_yx, ts_features, ts_detail = \
                                        self.check_dust_two_stage(
                                            tile.image, anomaly_map,
                                            dust_mask_no_ext if dust_mask_no_ext is not None else dust_mask,
                                            score,
                                        )
                                    _two_stage_ran = True
                                    _ts_features = ts_features
                                    _ts_dust_mask_no_ext = dust_mask_no_ext
                                    tile.dust_two_stage_features = ts_features
                                    if ts_has_real:
                                        tile.is_suspected_dust_or_scratch = False
                                        detail_text += (
                                            f" PER_REGION: 0real+{len(dust_regions)}dust"
                                            f" -> {ts_detail}"
                                        )
                                        if ts_peak_yx is not None:
                                            amap_h, amap_w = anomaly_map.shape[:2]
                                            tile.anomaly_peak_y = tile.y + int(ts_peak_yx[0] * tile.height / amap_h)
                                            tile.anomaly_peak_x = tile.x + int(ts_peak_yx[1] * tile.width / amap_w)
                                    else:
                                        tile.is_suspected_dust_or_scratch = True
                                        detail_text += (
                                            f" PER_REGION: 0real+{len(dust_regions)}dust"
                                            f" -> {ts_detail}"
                                        )
                                else:
                                    tile.is_suspected_dust_or_scratch = True
                                    detail_text += (
                                        f" PER_REGION: 0real+"
                                        f"{len(dust_regions)}dust -> DUST"
                                    )

                            # 產生 Debug 可視化圖
                            try:
                                if _two_stage_ran:
                                    dm_for_debug = _ts_dust_mask_no_ext if _ts_dust_mask_no_ext is not None else dust_mask
                                    tile.dust_iou_debug_image = self.generate_two_stage_debug_image(
                                        tile.image, anomaly_map, dm_for_debug,
                                        _ts_features,
                                        tile.is_suspected_dust_or_scratch,
                                    )
                                else:
                                    tile.dust_iou_debug_image = self.generate_dust_iou_debug_image(
                                        tile.image, anomaly_map, dust_mask,
                                        heatmap_binary, iou, top_pct,
                                        tile.is_suspected_dust_or_scratch,
                                        region_details=region_details,
                                        region_labels=region_labels,
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

            # === 加入 CV Edge 灰塵檢測 (與 OMIT 擷取) ===
            if getattr(result, 'edge_defects', []) and omit_image is not None:
                if omit_overexposed:
                    for ed in result.edge_defects:
                        ed.dust_detail_text = f"OMIT_OVEREXPOSED ({omit_overexposure_info}) -> Cannot verify dust, treated as REAL_NG"
                        print(f"⚠️ {img_path.name} Edge@{ed.side} Score:{ed.max_diff:.3f} → OMIT OVEREXPOSED, skip dust check")
                else:
                    # 讀取原始圖片一次，供所有 edge defect 的 defect mask 計算
                    orig_for_edge = None
                    if getattr(self, 'edge_inspector', None) and self.edge_inspector.config.dust_filter_enabled:
                        orig_for_edge = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

                    for ed in result.edge_defects:
                        ex, ey, ew, eh = ed.bbox
                        oh, ow = omit_image.shape[:2]
                        # 使用擴展 ROI (bbox ± 100px)，與 save_edge_defect_image 一致
                        # 原始 bbox 可能極小 (如 4x15px)，無法可靠偵測灰塵
                        edge_dust_padding = 100
                        tx = max(0, ex - edge_dust_padding)
                        ty = max(0, ey - edge_dust_padding)
                        x2 = min(ex + ew + edge_dust_padding, ow)
                        y2 = min(ey + eh + edge_dust_padding, oh)
                        if tx < ow and ty < oh:

                            omit_crop = omit_image[ty:y2, tx:x2]
                            ed.omit_crop_image = omit_crop.copy()

                            if getattr(self, 'edge_inspector', None) and self.edge_inspector.config.dust_filter_enabled:
                                is_dust, dust_mask, bright_ratio, detail_text = self.check_dust_or_scratch_feature(omit_crop)
                                ed.dust_mask = dust_mask
                                ed.dust_bright_ratio = bright_ratio

                                metric_mode = self.config.dust_heatmap_metric
                                metric_name = "COV" if metric_mode == "coverage" else "IOU"

                                if is_dust and dust_mask is not None and orig_for_edge is not None:
                                    # Step B: 空間重疊驗證 — 使用實際 CV defect mask (與 heatmap 一致)
                                    # 從原始圖片裁切相同 ROI，重建 CV 缺陷二值 mask
                                    crop_h, crop_w = omit_crop.shape[:2]
                                    orig_crop = orig_for_edge[ty:y2, tx:x2]
                                    if len(orig_crop.shape) == 3:
                                        orig_gray = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2GRAY)
                                    else:
                                        orig_gray = orig_crop

                                    # 與 save_edge_defect_image / _inspect_side 相同的 CV 檢測邏輯
                                    ecfg = self.edge_inspector.config
                                    ek = ecfg.blur_kernel
                                    emk = clamp_median_kernel(ecfg.median_kernel, min(orig_gray.shape[:2]) - 1)
                                    eblurred = cv2.GaussianBlur(orig_gray, (ek, ek), 0)

                                    if ed.side == "aoi_edge":
                                        _, ediff = compute_fg_aware_diff(eblurred, orig_gray, emk)
                                    else:
                                        ebg = cv2.medianBlur(eblurred, emk)
                                        ediff = cv2.absdiff(eblurred, ebg)

                                    edge_threshold = ecfg.get_threshold_for_side(ed.side)
                                    _, defect_mask_cv = cv2.threshold(ediff, edge_threshold, 255, cv2.THRESH_BINARY)

                                    # 只保留缺陷 BBox 範圍內的像素（與 heatmap 一致）
                                    rel_x = ex - tx
                                    rel_y = ey - ty
                                    bbox_only = np.zeros_like(defect_mask_cv)
                                    ry1 = max(0, rel_y)
                                    rx1 = max(0, rel_x)
                                    ry2 = min(defect_mask_cv.shape[0], rel_y + eh)
                                    rx2 = min(defect_mask_cv.shape[1], rel_x + ew)
                                    bbox_only[ry1:ry2, rx1:rx2] = 255
                                    defect_mask_cv = cv2.bitwise_and(defect_mask_cv, bbox_only)

                                    # dust_mask 轉單通道並對齊尺寸
                                    dm = dust_mask
                                    if len(dm.shape) == 3:
                                        dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
                                    if dm.shape[:2] != defect_mask_cv.shape[:2]:
                                        dm = cv2.resize(dm, (defect_mask_cv.shape[1], defect_mask_cv.shape[0]),
                                                        interpolation=cv2.INTER_NEAREST)

                                    defect_bool = defect_mask_cv > 0
                                    dust_bool = dm > 0
                                    intersection = np.count_nonzero(defect_bool & dust_bool)

                                    if metric_mode == "coverage":
                                        defect_area = np.count_nonzero(defect_bool)
                                        cov = float(intersection / defect_area) if defect_area > 0 else 0.0
                                    else:
                                        union = np.count_nonzero(defect_bool | dust_bool)
                                        cov = float(intersection / union) if union > 0 else 0.0

                                    if cov >= self.config.dust_heatmap_iou_threshold:
                                        ed.is_suspected_dust_or_scratch = True
                                        detail_text += f" {metric_name}:{cov:.3f}>={metric_name}_THR -> DUST (edge defect)"
                                    else:
                                        detail_text += f" {metric_name}:{cov:.3f}<{metric_name}_THR -> REAL_NG"
                                elif is_dust:
                                    # 有灰塵特徵但無法做空間驗證 → 保守視為真缺陷
                                    detail_text += " (dust detected, no spatial mask) -> REAL_NG"
                                else:
                                    detail_text += " NO_DUST -> REAL_NG"

                                ed.dust_detail_text = detail_text
                                log_icon = "🧹" if ed.is_suspected_dust_or_scratch else "🔴"
                                print(f"{log_icon} {img_path.name} Edge@{ed.side} → {detail_text}")
                            else:
                                ed.dust_detail_text = "Dust filter disabled -> REAL_NG"
            
            return result
        
        postprocess_start = time.time()
        
        # 只對有異常或有 CV 邊緣缺陷，且有 OMIT 的結果進行平行灰塵檢測
        needs_dust_check = [r for r in preprocessed_results if (r.anomaly_tiles or getattr(r, 'edge_defects', [])) and omit_image is not None]
        
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
                            rcov = getattr(tile, 'dust_region_max_cov', 0.0)
                            metric_name = "COV" if self.config.dust_heatmap_metric == "coverage" else "IOU"
                            if tile.is_suspected_dust_or_scratch:
                                box_color = (0, 165, 255)
                                label = f"DUST R.{metric_name}:{rcov:.3f}"
                            else:
                                box_color = (0, 0, 255)
                                label = f"REAL_NG R.{metric_name}:{rcov:.3f}"
                            cv2.rectangle(omit_vis, (tx, ty), (x2, y2), box_color, 5)
                            cv2.putText(omit_vis, f"{result.image_path.name}", (tx, ty - 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, box_color, 4)
                            cv2.putText(omit_vis, label, (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, box_color, 4)
                
                # 在 OMIT 總圖上畫 Edge 框
                if getattr(result, 'edge_defects', []):
                    for ed in result.edge_defects:
                        tx, ty, tw, th = ed.bbox
                        oh, ow = omit_vis.shape[:2]
                        if tx < ow and ty < oh:
                            x2 = min(tx + tw, ow)
                            y2 = min(ty + th, oh)
                            if getattr(ed, 'is_suspected_dust_or_scratch', False):
                                box_color = (0, 165, 255)
                                label = f"Edge DUST ({ed.side})"
                            else:
                                box_color = (0, 0, 255)
                                label = f"Edge NG ({ed.side})"
                            cv2.rectangle(omit_vis, (tx, ty), (x2, y2), box_color, 5)
                            cv2.putText(omit_vis, f"{result.image_path.name}", (tx, ty - 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, box_color, 4)
                            cv2.putText(omit_vis, label, (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, box_color, 4)
        
        postprocess_time = time.time() - postprocess_start
        print(f"🧹 Phase 3 完成: 灰塵檢測後處理耗時 {postprocess_time:.2f}s")
        
        results = preprocessed_results
                
        # === 炸彈系統比對 ===
        # 決定炸彈來源：優先使用 Client 端傳入的 runtime 資料
        active_bombs = []
        if bomb_info is not None:
            # 從協議取得炸彈座標，defect_code 從 config 匹配
            defect_code = self._match_bomb_defect_code(bomb_info)
            active_bombs = [BombDefect(
                image_prefix=bomb_info["image_prefix"],
                defect_code=defect_code,
                defect_type=bomb_info["defect_type"],
                coordinates=bomb_info["coordinates"],
            )]
            print(f"💣 使用協議炸彈資料: prefix={bomb_info['image_prefix']} "
                  f"type={bomb_info['defect_type']} defect_code={defect_code} "
                  f"coords={bomb_info['coordinates']}")
        elif self.config.bomb_defects:
            active_bombs = self.config.bomb_defects
            print(f"💣 使用 Config 炸彈資料: {len(active_bombs)} 組設定")

        if active_bombs:
            for result in results:
                if result.anomaly_tiles and result.raw_bounds is not None:
                    img_prefix = result.image_path.stem
                    for tile, score, anomaly_map in result.anomaly_tiles:
                        # 計算熱力圖峰值位置 (更精確的缺陷位置)
                        if anomaly_map is not None and anomaly_map.size > 0:
                            try:
                                amap_h, amap_w = anomaly_map.shape[:2]
                                # 二值化偵測 (B0F00000 等): 使用亮點重心而非 argmax
                                # argmax 在 binary map 上回傳最左上方像素，容易偏離實際缺陷位置
                                if tile.is_bright_spot_detection:
                                    ys, xs = np.where(anomaly_map > 0.5)
                                    if len(xs) > 0:
                                        centroid_x = int(np.mean(xs))
                                        centroid_y = int(np.mean(ys))
                                    else:
                                        logger.warning(f"Bright spot tile has no pixels > 0.5: {result.image_path.name} Tile@({tile.x},{tile.y})")
                                        centroid_x = amap_w // 2
                                        centroid_y = amap_h // 2
                                    tile.anomaly_peak_x = tile.x + int(centroid_x * tile.width / amap_w)
                                    tile.anomaly_peak_y = tile.y + int(centroid_y * tile.height / amap_h)
                                else:
                                    peak_local = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
                                    # anomaly_map 尺寸可能和 tile 不同，需要縮放
                                    tile.anomaly_peak_y = tile.y + int(peak_local[0] * tile.height / amap_h)
                                    tile.anomaly_peak_x = tile.x + int(peak_local[1] * tile.width / amap_w)
                            except Exception as e:
                                logger.warning(f"Anomaly peak calculation failed: {e}")
                                tile.anomaly_peak_x, tile.anomaly_peak_y = tile.center
                        else:
                            tile.anomaly_peak_x, tile.anomaly_peak_y = tile.center

                        # 用熱力圖峰值座標來比對炸彈
                        is_bomb, bomb_code = self.check_bomb_match(
                            img_prefix, tile.anomaly_peak_x, tile.anomaly_peak_y, result.raw_bounds,
                            anomaly_map=anomaly_map, product_resolution=product_resolution,
                            bomb_list=active_bombs,
                        )
                        # AOI coord tile fallback: 若峰值未匹配，改用 tile 中心再試一次
                        # AOI coord tile 本身就是以 AOI 座標為中心切塊，中心位置更可靠
                        if not is_bomb and tile.is_aoi_coord_tile:
                            tile_cx, tile_cy = tile.center
                            is_bomb, bomb_code = self.check_bomb_match(
                                img_prefix, tile_cx, tile_cy, result.raw_bounds,
                                anomaly_map=anomaly_map, product_resolution=product_resolution,
                                bomb_list=active_bombs,
                            )
                        # AOI coord tile 保護: peak 可能被鄰近炸彈亮點吸引，
                        # 需驗證原始 AOI 產品座標本身也在炸彈容忍範圍內
                        if is_bomb and tile.is_aoi_coord_tile and tile.aoi_product_x >= 0:
                            aoi_matches_bomb = False
                            tolerance = self.config.bomb_match_tolerance
                            for bomb in active_bombs:
                                if not (img_prefix == bomb.image_prefix or
                                        img_prefix.startswith(bomb.image_prefix + "_")):
                                    continue
                                if bomb.defect_type == "point":
                                    for coord in bomb.coordinates:
                                        if (abs(tile.aoi_product_x - coord[0]) <= tolerance and
                                            abs(tile.aoi_product_y - coord[1]) <= tolerance):
                                            aoi_matches_bomb = True
                                            break
                                elif bomb.defect_type == "line" and len(bomb.coordinates) >= 2:
                                    # line 型: 檢查 AOI 座標是否在線段緩衝帶內
                                    pt1, pt2 = bomb.coordinates[0], bomb.coordinates[1]
                                    min_x = min(pt1[0], pt2[0]) - tolerance
                                    max_x = max(pt1[0], pt2[0]) + tolerance
                                    min_y = min(pt1[1], pt2[1])
                                    max_y = max(pt1[1], pt2[1])
                                    if (min_x <= tile.aoi_product_x <= max_x and
                                        min_y <= tile.aoi_product_y <= max_y):
                                        aoi_matches_bomb = True
                                if aoi_matches_bomb:
                                    break
                            if not aoi_matches_bomb:
                                is_bomb = False
                                print(f"🛡️ {result.image_path.name} Tile@({tile.x},{tile.y}) Peak 匹配炸彈但 AOI 座標 ({tile.aoi_product_x},{tile.aoi_product_y}) 距離炸彈超過容忍度，保留為真實缺陷")
                        if is_bomb:
                            tile.is_bomb = True
                            tile.bomb_defect_code = bomb_code

            # === Bomb line 共識機制 ===
            # 若同一條 bomb line 已有 ≥3 個 tile 通過形態驗證，
            # 則位置匹配但形態不通過的 tile 也視為 bomb（線已被確認存在）
            for bomb in active_bombs:
                if bomb.defect_type != "line":
                    continue
                for result in results:
                    if not result.anomaly_tiles or result.raw_bounds is None:
                        continue
                    img_prefix = result.image_path.stem
                    if not (img_prefix == bomb.image_prefix or
                            img_prefix.startswith(bomb.image_prefix + "_")):
                        continue
                    # 計算此 bomb line 已確認的 tile 數
                    confirmed = sum(
                        1 for t, _, _ in result.anomaly_tiles
                        if t.is_bomb and t.bomb_defect_code == bomb.defect_code
                    )
                    if confirmed < 3:
                        continue
                    # 對未匹配的 tile 做位置-only 第二輪檢查
                    for tile, score, anomaly_map in result.anomaly_tiles:
                        if tile.is_bomb:
                            continue
                        peak_x = getattr(tile, 'anomaly_peak_x', tile.center[0])
                        peak_y = getattr(tile, 'anomaly_peak_y', tile.center[1])
                        is_bomb, bomb_code = self.check_bomb_match(
                            img_prefix, peak_x, peak_y, result.raw_bounds,
                            anomaly_map=anomaly_map, product_resolution=product_resolution,
                            bomb_list=[bomb], skip_shape_check=True,
                        )
                        if not is_bomb and tile.is_aoi_coord_tile:
                            tile_cx, tile_cy = tile.center
                            is_bomb, bomb_code = self.check_bomb_match(
                                img_prefix, tile_cx, tile_cy, result.raw_bounds,
                                anomaly_map=anomaly_map, product_resolution=product_resolution,
                                bomb_list=[bomb], skip_shape_check=True,
                            )
                        if is_bomb:
                            tile.is_bomb = True
                            tile.bomb_defect_code = bomb_code

            # === 邊緣缺陷炸彈比對 ===
            for result in results:
                if not hasattr(result, 'edge_defects') or not result.edge_defects or result.raw_bounds is None:
                    continue
                img_prefix = result.image_path.stem
                for ed in result.edge_defects:
                    if getattr(ed, 'is_cv_ok', False):
                        continue
                    cx, cy = ed.center
                    is_bomb, bomb_code = self.check_bomb_match(
                        img_prefix, cx, cy, result.raw_bounds,
                        product_resolution=product_resolution,
                        bomb_list=active_bombs,
                    )
                    if is_bomb:
                        ed.is_bomb = True
                        ed.bomb_defect_code = bomb_code

            from collections import Counter
            for result in results:
                edge_bombs = [ed for ed in result.edge_defects if ed.is_bomb]
                summary = Counter(t.bomb_defect_code for t, _, _ in result.anomaly_tiles if t.is_bomb)
                summary.update(ed.bomb_defect_code for ed in edge_bombs)
                if summary:
                    parts = [f"{c}×{n}" for c, n in summary.items()]
                    suffix = f" (含 edge {len(edge_bombs)})" if edge_bombs else ""
                    print(f"💣 {result.image_path.name} BOMB match: {', '.join(parts)}{suffix}")

        # === 不檢測排除區域判定 (基於 peak 位置) ===
        # 排除區域來自 cv_edge_exclude_zones，原僅用於邊緣檢測，現擴展至 PatchCore 推論
        # 以熱力圖峰值 (defect 精確位置) 判斷是否落在排除區域，而非整塊 512x512 tile
        if getattr(self, "edge_inspector", None):
            try:
                resolution_code = ""
                if model_id and len(model_id) >= 6:
                    resolution_code = model_id[5].upper()
                self.edge_inspector.config.set_active_zones_for_product(resolution_code)

                active_zones = [z for z in self.edge_inspector.config.exclude_zones if z.enabled]
                if active_zones:
                    for result in results:
                        if not result.anomaly_tiles:
                            continue
                        for tile, score, anomaly_map in result.anomaly_tiles:
                            # 亮點偵測 tile (B0F 等黑圖) 不受排除區域影響
                            if tile.is_bright_spot_detection:
                                continue
                            # 使用熱力圖峰值座標 (更精確的缺陷位置)
                            if tile.anomaly_peak_x >= 0 and tile.anomaly_peak_y >= 0:
                                px, py = tile.anomaly_peak_x, tile.anomaly_peak_y
                            else:
                                px, py = tile.x + tile.width // 2, tile.y + tile.height // 2

                            for zone in active_zones:
                                # 判斷峰值點是否在排除區域內
                                if (zone.x <= px <= zone.x + zone.w and
                                    zone.y <= py <= zone.y + zone.h):
                                    tile.is_in_exclude_zone = True
                                    logger.info(f"Tile #{tile.tile_id} peak@({px},{py}) 位於排除區域 ({zone.x},{zone.y},{zone.w},{zone.h})，標記為不檢測區域")
                                    break
            except Exception as e:
                logger.error(f"排除區域檢查失敗: {e}", exc_info=True)

        sf = self._get_scratch_filter()
        if sf is not None:
            panel_tiles = panel_filtered = 0
            panel_ms = 0.0
            for result in results:
                if result.anomaly_tiles:
                    sf.apply_to_image_result(result)
                    panel_tiles += len(result.anomaly_tiles)
                    panel_filtered += getattr(result, "scratch_filter_count", 0)
                    panel_ms += getattr(result, "scratch_elapsed_ms", 0.0)
            if panel_tiles:
                logger.info("[scratch] Panel 總計: 檢查 %d tiles, filtered=%d, TT=%.1fms",
                            panel_tiles, panel_filtered, panel_ms)

        total_panel_time = preprocess_time + inference_time + postprocess_time
        print(f"📊 Panel {panel_dir.name} 總計: 預處理 {preprocess_time:.2f}s + 推論 {inference_time:.2f}s + 後處理 {postprocess_time:.2f}s = {total_panel_time:.2f}s")

        if bomb_info is not None:
            for result in results:
                result.client_bomb_info = bomb_info

        return results, omit_vis, omit_overexposed, omit_overexposure_info, is_duplicate, omit_image, aoi_report

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

        # Panel polygon（紅色）
        if result.panel_polygon is not None:
            poly_int = result.panel_polygon.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [poly_int], True, (0, 0, 255), 6)

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
        
        # 排除區域 (MARK / 機構)
        for region in result.exclusion_regions:
            cv2.rectangle(vis, (region.x1, region.y1), (region.x2, region.y2), (128, 128, 128), 4)

        # CV 邊緣不檢測排除區域 (settings 設定，按機種劃分)
        EXCLUDE_ZONE_COLOR = (0, 200, 200)  # 青黃色 (BGR)
        if getattr(self, "edge_inspector", None):
            active_zones = [z for z in self.edge_inspector.config.exclude_zones if z.enabled]
            for zone in active_zones:
                zx1, zy1 = zone.x, zone.y
                zx2, zy2 = zone.x + zone.w, zone.y + zone.h
                # 半透明填充
                overlay = vis.copy()
                cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), EXCLUDE_ZONE_COLOR, -1)
                cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
                # 邊框
                cv2.rectangle(vis, (zx1, zy1), (zx2, zy2), EXCLUDE_ZONE_COLOR, 3)
                # 標籤
                cv2.putText(vis, "EXCLUDE ZONE", (zx1 + 5, zy1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, EXCLUDE_ZONE_COLOR, 2)

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
            # AOI 座標 tile 但 AI 判定未達閾值：綠色框標 OK（不算 NG）
            if getattr(tile, 'is_aoi_coord_below_threshold', False):
                color = (0, 255, 0)  # 綠色 (BGR)
                label = f"{score:.2f} OK"
                thickness = 3
            else:
                color = (0, 0, 255)  # 紅色 (預設異常)
                label = f"{score:.2f}"
                thickness = 6

            # 不檢測排除區域：灰色虛線風格
            if getattr(tile, 'is_in_exclude_zone', False):
                color = (180, 180, 180)  # 灰色 (BGR)
                label = f"{score:.2f} EXCLUDED"
                thickness = 3
            # 炸彈 tile：洋紅色 (紫色)
            elif getattr(tile, 'is_bomb', False):
                color = (255, 0, 255)  # 洋紅色 (BGR)
                code = getattr(tile, 'bomb_defect_code', '')
                label = f"{score:.2f} BOMB({code})"
            # 如果是疑似灰塵/刮痕，改為黃色
            elif getattr(tile, 'is_suspected_dust_or_scratch', False):
                color = (0, 255, 255)  # 黃色 (BGR: 0, 255, 255)
                metric_name = "COV" if self.config.dust_heatmap_metric == "coverage" else "IOU"
                rcov = getattr(tile, 'dust_region_max_cov', 0.0)
                label = f"{score:.2f} DUST(R.{metric_name}:{rcov:.3f})"
            elif getattr(tile, 'dust_heatmap_iou', 0.0) > 0:
                # 有 OMIT 分析結果但非灰塵，顯示 per-region max COV
                metric_name = "COV" if self.config.dust_heatmap_metric == "coverage" else "IOU"
                rcov = getattr(tile, 'dust_region_max_cov', 0.0)
                label = f"{score:.2f} NG(R.{metric_name}:{rcov:.3f})"
            
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
                
                # 右下角標上區域編號 (e.g. #64)
                id_label = f"#{tile.tile_id}"
                (tw_text, th_text), _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
                id_x = max(0, min(x2_clip - tw_text - 10, w - tw_text))
                id_y = max(th_text, min(y2_clip - 10, h - 5))
                cv2.putText(vis, id_label, (id_x, id_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            except Exception as e:
                print(f"❌ 繪製 Tile {tile.tile_id} 失敗: {e}, 座標: ({x1_clip},{y1_clip})->({x2_clip},{y2_clip})")
        
        # 結果標籤 — 過濾掉 AOI 座標未達閾值的 tile (不影響 NG 判定)
        effective_anomaly_tiles = [t for t in result.anomaly_tiles
                                   if not getattr(t[0], 'is_aoi_coord_below_threshold', False)]
        status = "NG" if effective_anomaly_tiles else "OK"
        # 檢查是否所有異常都是疑似灰塵
        all_dust = False
        all_bomb = False
        all_excluded = False
        if effective_anomaly_tiles:
            non_dust = [t for t in effective_anomaly_tiles if not t[0].is_suspected_dust_or_scratch or t[0].is_bomb]
            all_dust = all(t[0].is_suspected_dust_or_scratch and not t[0].is_bomb for t in effective_anomaly_tiles)
            all_bomb = non_dust and all(t[0].is_bomb for t in non_dust)
            all_excluded = all(
                t[0].is_in_exclude_zone or t[0].is_suspected_dust_or_scratch or t[0].is_bomb
                for t in effective_anomaly_tiles
            ) and any(t[0].is_in_exclude_zone for t in effective_anomaly_tiles)
            if all_excluded and not any(
                not t[0].is_in_exclude_zone and not t[0].is_suspected_dust_or_scratch and not t[0].is_bomb
                for t in effective_anomaly_tiles
            ):
                status = "OK (Excluded)"
            elif all_dust:
                status = "NG (Dust?)"
            elif all_bomb:
                status = "BOMB"

        if not effective_anomaly_tiles:
            color = (0, 255, 0)
        elif all_excluded and status == "OK (Excluded)":
            color = (180, 180, 180)  # 灰色
        elif all_bomb:
            color = (255, 0, 255)  # 洋紅色
        elif all_dust:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        
        cv2.putText(vis, status, (x1 + 20, y1 + 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, color, 10)
        
        # 標記 CV 邊緣檢測異常
        if hasattr(result, 'edge_defects') and result.edge_defects:
            for ed in result.edge_defects:
                bx, by, bw, bh = ed.bbox
                
                is_bomb_ed = getattr(ed, 'is_bomb', False)
                is_dust = getattr(ed, 'is_suspected_dust_or_scratch', False)
                
                if is_bomb_ed:
                    box_color = (255, 0, 255)  # 紫色 (洋紅色)
                elif is_dust:
                    box_color = (0, 165, 255)
                else:
                    box_color = (0, 0, 255)    # 紅色
                
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), box_color, 4)
                
                # 在框的旁邊加上文字（處理文字是否超出邊界的邏輯）
                if is_bomb_ed:
                    status_label = f"BOMB({getattr(ed, 'bomb_defect_code', '')})"
                elif is_dust:
                    status_label = "DUST"
                else:
                    status_label = "NG"
                text = f"Edge {status_label}: {ed.side} ({ed.max_diff:.0f})"
                text_x = max(10, bx)
                text_y = max(30, by - 10)
                if ed.side == 'top':
                    text_y = by + bh + 30
                elif ed.side == 'left':
                    text_x = bx + bw + 10
                    
                cv2.putText(vis, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 3)

        return vis
    
    def generate_bomb_diagram(self, image_path: Path, result: ImageResult,
                              product_resolution: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        生成炸彈座標位置示意圖
        在原始圖片上疊加：
        1. 已設定的炸彈座標位置 (洋紅色十字)
        2. AD 偵測到的異常 tile 位置 (青色方框)
        3. 匹配連線 + 距離標示
        """
        if product_resolution is None:
            product_resolution = DEFAULT_PRODUCT_RESOLUTION
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            image = np.zeros((product_resolution[1], product_resolution[0], 3), dtype=np.uint8)
        
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
        PRODUCT_WIDTH, PRODUCT_HEIGHT = product_resolution
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
        
        active_bombs = []
        if result.client_bomb_info:
            bomb_info = result.client_bomb_info
            defect_code = self._match_bomb_defect_code(bomb_info)
            active_bombs = [BombDefect(
                image_prefix=bomb_info["image_prefix"],
                defect_code=defect_code,
                defect_type=bomb_info["defect_type"],
                coordinates=bomb_info["coordinates"],
            )]
        elif self.config.bomb_defects:
            active_bombs = self.config.bomb_defects
            
        for bomb in active_bombs:
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
