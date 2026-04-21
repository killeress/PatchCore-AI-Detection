"""
CAPI AI — CV 邊緣缺陷檢測模組

使用傳統電腦視覺 (OpenCV) 對產品四邊邊緣進行獨立的缺陷檢測，
與 AI (PatchCore) 互補，解決 AI 在產品邊界附近敏感度不足的問題。

檢測流程:
  1. 從 raw_bounds 切出四邊 ROI（上/下/左/右）
  2. 灰階 → 高斯模糊 3x3
  3. 中值濾波估計背景 → 計算差異
  4. 二值化 → 連通組件分析 → 面積過濾

參數參考: 天目 AOI 機台「邊框各別判定」設定
"""

import cv2
import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


def clamp_median_kernel(kernel_size: int, max_dim: int) -> int:
    """將 median filter 核大小限制為有效奇數值 (≥3, ≤max_dim)"""
    mk = min(kernel_size, max_dim)
    if mk % 2 == 0:
        mk -= 1
    return max(3, mk)


def inpaint_non_fg_region(blurred: np.ndarray, fg_mask: np.ndarray, inpaint_radius: int = 3) -> np.ndarray:
    """用 Navier-Stokes inpaint 填非前景區，取代 fg_median 常數填充，避免 medianBlur bg 估計在邊界內側被拉偏。"""
    if fg_mask is None:
        return blurred
    non_fg = (fg_mask == 0).astype(np.uint8)
    if not np.any(non_fg) or not np.any(fg_mask > 0):
        return blurred
    blurred_c = np.ascontiguousarray(blurred)
    return cv2.inpaint(blurred_c, non_fg, inpaint_radius, cv2.INPAINT_NS)


# 前景亮度閾值：低於此值的像素視為產品外背景（黑色邊界）
FG_BRIGHTNESS_THRESHOLD = 15


def compute_fg_aware_diff(
    blurred: np.ndarray,
    gray: np.ndarray,
    median_kernel: int,
    brightness_threshold: int = FG_BRIGHTNESS_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    計算前景感知的差異圖，排除非產品區域對背景估計的干擾。

    將非前景像素填充為前景中值後做 median filter 背景估計，
    並在差異圖中將非前景區域歸零。

    Args:
        blurred: 高斯模糊後的灰階影像
        gray: 原始灰階影像（用於建立前景遮罩）
        median_kernel: 中值濾波核大小（需為已 clamp 的奇數）
        brightness_threshold: 前景亮度閾值

    Returns:
        (fg_mask, diff) — fg_mask (255=前景, 0=背景), diff (前景感知差異圖)
    """
    fg_mask = np.zeros_like(gray)
    fg_mask[gray >= brightness_threshold] = 255

    blurred_for_bg = inpaint_non_fg_region(blurred, fg_mask)

    bg = cv2.medianBlur(blurred_for_bg, median_kernel)
    diff = cv2.absdiff(blurred, bg)
    diff[fg_mask == 0] = 0

    return fg_mask, diff


@dataclass
class EdgeDefect:
    """單一邊緣缺陷"""
    side: str            # "left", "right", "top", "bottom"
    area: int            # 缺陷面積 (px)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) 在原圖絕對座標
    center: Tuple[int, int]          # (cx, cy) 中心點原圖絕對座標
    max_diff: int = 0    # 該區域最大灰階差值
    solidity: float = 1.0  # convex-hull solidity；aoi_edge 用以過濾 L 形邊界偽影
    is_suspected_dust_or_scratch: bool = False
    omit_crop_image: Optional[np.ndarray] = field(default=None, repr=False)
    dust_mask: Optional[np.ndarray] = field(default=None, repr=False)
    dust_bright_ratio: float = 0.0
    dust_detail_text: str = ""
    is_bomb: bool = False
    bomb_defect_code: str = ""
    is_cv_ok: bool = False  # CV 檢查後未偵測到缺陷，僅作記錄用
    threshold_used: int = 0       # 使用的閾值 (用於 heatmap header 顯示)
    min_area_used: int = 0        # 使用的最小面積 (用於 heatmap header 顯示)
    inspector_mode: str = "cv"    # "cv" | "patchcore"
    patchcore_score: float = 0.0
    patchcore_threshold: float = 0.0
    patchcore_ok_reason: str = ""
    # 推論當下保留的 render artifacts (async heatmap 用，不入 DB)
    pc_roi: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    pc_fg_mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    pc_anomaly_map: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


@dataclass
class EdgeSideConfig:
    """單邊檢測參數"""
    width: int = 450          # 從邊界往內掃描寬度 (px)
    threshold: int = 5        # 灰階差值閾值 (明暗度)
    min_area: int = 70        # 最小缺陷面積 (px)
    exclude_top: int = 80     # 不判定區域 - 上
    exclude_bottom: int = 80  # 不判定區域 - 下
    exclude_left: int = 10    # 不判定區域 - 左
    exclude_right: int = 10   # 不判定區域 - 右


@dataclass
class EdgeExclusionZoneConfig:
    """不檢測排除區域：PatchCore 推論及邊緣檢測結果若與此區域重疊則標記為排除區域並視為 OK"""
    enabled: bool = False
    x: int = 0
    y: int = 0
    w: int = 100
    h: int = 100


@dataclass
class EdgeInspectionConfig:
    """四邊檢測參數組"""
    enabled: bool = True
    dust_filter_enabled: bool = False
    blur_kernel: int = 3          # 高斯模糊核大小
    median_kernel: int = 65       # 中值濾波核大小（用於估計背景）
    left: EdgeSideConfig = field(default_factory=lambda: EdgeSideConfig(
        width=450, threshold=5, min_area=70,
        exclude_top=80, exclude_bottom=80, exclude_left=10, exclude_right=10
    ))
    right: EdgeSideConfig = field(default_factory=lambda: EdgeSideConfig(
        width=650, threshold=5, min_area=60,
        exclude_top=110, exclude_bottom=110, exclude_left=100, exclude_right=10
    ))
    top: EdgeSideConfig = field(default_factory=lambda: EdgeSideConfig(
        width=550, threshold=5, min_area=60,
        exclude_top=10, exclude_bottom=10, exclude_left=80, exclude_right=80
    ))
    bottom: EdgeSideConfig = field(default_factory=lambda: EdgeSideConfig(
        width=360, threshold=4, min_area=65,
        exclude_top=10, exclude_bottom=10, exclude_left=80, exclude_right=80
    ))
    # AOI 座標邊緣檢測獨立參數 (不共用四邊設定)
    aoi_threshold: int = 4        # AOI 座標邊緣明暗差閾值
    aoi_min_area: int = 40        # AOI 座標邊緣最小缺陷面積 (px) — 對齊 debug 頁預設
    aoi_solidity_min: float = 0.2 # Solidity 下限: 低於此值視為 L 形邊界偽影 (0=停用)
    aoi_polygon_erode_px: int = 3 # polygon fg_mask 內縮 px 數，避開面板邊緣亮帶轉換區 (0=停用，僅 polygon 模式有效)
    aoi_morph_open_kernel: int = 3 # 二值化後 morphological opening kernel 大小，去除 1-px 條紋與細雜訊橋 (0=停用)
    aoi_min_max_diff: int = 20 # component 內最大 diff 下限：低於此值視為低對比度紋理雜訊 (0=停用)
    aoi_line_min_length: int = 30 # 投影法偵測薄線最小長度 px (垂直/水平連續活化像素投影, 0=停用)
    aoi_line_max_width: int = 3 # 薄線最大寬度 px (超過視為一般 component, 由 CC path 處理)
    aoi_edge_inspector: str = "cv"  # "cv" | "patchcore"
    exclude_zones: List[EdgeExclusionZoneConfig] = field(default_factory=list)
    # 儲存完整的按產品分組排除區域 (key=resolution_code, value=list of zones)
    all_exclude_zones_by_product: Dict[str, List[dict]] = field(default_factory=dict)

    def get_threshold_for_side(self, side: str) -> int:
        """取得指定邊的閾值，aoi_edge 使用獨立閾值"""
        if side == "aoi_edge":
            return self.aoi_threshold
        side_cfg = getattr(self, side, None)
        if side_cfg is not None and hasattr(side_cfg, 'threshold'):
            return side_cfg.threshold
        return min(s.threshold for s in [self.left, self.right, self.top, self.bottom])

    def set_active_zones_for_product(self, resolution_code: str):
        """根據產品解析度碼切換當前作用中的排除區域"""
        if not self.all_exclude_zones_by_product:
            return  # 沒有按產品分組的設定，維持現有 zones
        
        product_zones = self.all_exclude_zones_by_product.get(resolution_code, [])
        if not product_zones and self.all_exclude_zones_by_product:
            # Fallback: 取第一組
            first_key = next(iter(self.all_exclude_zones_by_product))
            product_zones = self.all_exclude_zones_by_product[first_key]
        
        self.exclude_zones = [
            EdgeExclusionZoneConfig(
                enabled=z.get("enabled", True),
                x=int(z.get("x", 0)),
                y=int(z.get("y", 0)),
                w=int(z.get("w", 100)),
                h=int(z.get("h", 100))
            ) for z in product_zones if isinstance(z, dict)
        ]

    @classmethod
    def from_db_params(cls, params: Dict[str, Any], resolution_code: str = "") -> "EdgeInspectionConfig":
        """從 DB config_params 載入
        
        Args:
            params: DB 參數字典
            resolution_code: 產品解析度碼 (例如 "H", "J")，用於選取對應的排除區域
        """
        def get(key, default):
            v = params.get(key)
            if v is None:
                return default
            if isinstance(v, dict):
                return v.get("decoded_value", default)
            return v

        cfg = cls(
            enabled=bool(get("cv_edge_enabled", True)),
            dust_filter_enabled=bool(get("cv_edge_dust_filter_enabled", False)),
        )
        # 左
        cfg.left.width = int(get("cv_edge_left_width", 450))
        cfg.left.threshold = int(get("cv_edge_left_threshold", 5))
        cfg.left.min_area = int(get("cv_edge_left_min_area", 70))
        cfg.left.exclude_top = int(get("cv_edge_left_exclude_top", 80))
        cfg.left.exclude_bottom = int(get("cv_edge_left_exclude_bottom", 80))
        cfg.left.exclude_left = int(get("cv_edge_left_exclude_left", 10))
        cfg.left.exclude_right = int(get("cv_edge_left_exclude_right", 10))
        # 右
        cfg.right.width = int(get("cv_edge_right_width", 650))
        cfg.right.threshold = int(get("cv_edge_right_threshold", 5))
        cfg.right.min_area = int(get("cv_edge_right_min_area", 60))
        cfg.right.exclude_top = int(get("cv_edge_right_exclude_top", 110))
        cfg.right.exclude_bottom = int(get("cv_edge_right_exclude_bottom", 110))
        cfg.right.exclude_left = int(get("cv_edge_right_exclude_left", 100))
        cfg.right.exclude_right = int(get("cv_edge_right_exclude_right", 10))
        # 上
        cfg.top.width = int(get("cv_edge_top_width", 550))
        cfg.top.threshold = int(get("cv_edge_top_threshold", 5))
        cfg.top.min_area = int(get("cv_edge_top_min_area", 60))
        cfg.top.exclude_top = int(get("cv_edge_top_exclude_top", 10))
        cfg.top.exclude_bottom = int(get("cv_edge_top_exclude_bottom", 10))
        cfg.top.exclude_left = int(get("cv_edge_top_exclude_left", 80))
        cfg.top.exclude_right = int(get("cv_edge_top_exclude_right", 80))
        # 下
        cfg.bottom.width = int(get("cv_edge_bottom_width", 360))
        cfg.bottom.threshold = int(get("cv_edge_bottom_threshold", 4))
        cfg.bottom.min_area = int(get("cv_edge_bottom_min_area", 65))
        cfg.bottom.exclude_top = int(get("cv_edge_bottom_exclude_top", 10))
        cfg.bottom.exclude_bottom = int(get("cv_edge_bottom_exclude_bottom", 10))
        cfg.bottom.exclude_left = int(get("cv_edge_bottom_exclude_left", 80))
        cfg.bottom.exclude_right = int(get("cv_edge_bottom_exclude_right", 80))
        # AOI 座標邊緣獨立參數
        cfg.aoi_threshold = int(get("cv_edge_aoi_threshold", 4))
        cfg.aoi_min_area = int(get("cv_edge_aoi_min_area", 40))
        cfg.aoi_solidity_min = float(get("cv_edge_aoi_solidity_min", 0.2))
        cfg.aoi_polygon_erode_px = int(get("cv_edge_aoi_polygon_erode_px", 3))
        cfg.aoi_morph_open_kernel = int(get("cv_edge_aoi_morph_open_kernel", 3))
        cfg.aoi_min_max_diff = int(get("cv_edge_aoi_min_max_diff", 20))
        cfg.aoi_line_min_length = int(get("cv_edge_aoi_line_min_length", 30))
        cfg.aoi_line_max_width = int(get("cv_edge_aoi_line_max_width", 3))
        inspector_val = str(get("aoi_edge_inspector", "cv")).lower().strip()
        cfg.aoi_edge_inspector = inspector_val if inspector_val in ("cv", "patchcore") else "cv"

        # 排除區域 (支援按產品解析度碼分組的 dict 格式，或向後相容的 list 格式)
        zones_raw = get("cv_edge_exclude_zones", None)
        zones_data = None
        if zones_raw:
            # 可能已被 _decode_config_value 解析為 dict/list，或仍是 JSON 字串
            if isinstance(zones_raw, (dict, list)):
                zones_data = zones_raw
            elif isinstance(zones_raw, str):
                try:
                    zones_data = json.loads(zones_raw)
                except:
                    pass
        
        if zones_data is not None:
            # 新格式: dict keyed by resolution_code，例如 {"H": [...], "J": [...]}
            if isinstance(zones_data, dict):
                # 儲存完整的 per-product zones dict (用於 runtime 切換)
                cfg.all_exclude_zones_by_product = zones_data
                # 選取對應產品的排除區域
                product_zones = zones_data.get(resolution_code, []) if resolution_code else []
                # 若找不到對應的 code，嘗試取第一組作為 fallback
                if not product_zones and zones_data:
                    first_key = next(iter(zones_data))
                    product_zones = zones_data[first_key]
                    if resolution_code:
                        print(f"⚠️ 排除區域未找到產品 '{resolution_code}'，使用 fallback '{first_key}'")
                if isinstance(product_zones, list):
                    cfg.exclude_zones = [
                        EdgeExclusionZoneConfig(
                            enabled=z.get("enabled", True),
                            x=int(z.get("x", 0)),
                            y=int(z.get("y", 0)),
                            w=int(z.get("w", 100)),
                            h=int(z.get("h", 100))
                        ) for z in product_zones
                    ]
            # 舊格式: 直接是 list
            elif isinstance(zones_data, list):
                cfg.exclude_zones = [
                    EdgeExclusionZoneConfig(
                        enabled=z.get("enabled", True),
                        x=int(z.get("x", 0)),
                        y=int(z.get("y", 0)),
                        w=int(z.get("w", 100)),
                        h=int(z.get("h", 100))
                    ) for z in zones_data
                ]
        
        # 向後相容：若無多組區域列表，嘗試讀取舊版單一區域設定
        if not cfg.exclude_zones:
            old_enabled = bool(get("cv_edge_exclude_enabled", False))
            if old_enabled or get("cv_edge_exclude_x", None) is not None:
                cfg.exclude_zones = [
                    EdgeExclusionZoneConfig(
                        enabled=old_enabled,
                        x=int(get("cv_edge_exclude_x", 0)),
                        y=int(get("cv_edge_exclude_y", 0)),
                        w=int(get("cv_edge_exclude_w", 100)),
                        h=int(get("cv_edge_exclude_h", 100))
                    )
                ]
        
        return cfg


class CVEdgeInspector:
    """
    傳統 CV 邊緣缺陷檢測器

    用法:
        inspector = CVEdgeInspector(config)
        defects = inspector.inspect(image, raw_bounds)
    """

    def __init__(self, config: Optional[EdgeInspectionConfig] = None):
        self.config = config or EdgeInspectionConfig()

    def inspect(
        self,
        image: np.ndarray,
        raw_bounds: Tuple[int, int, int, int],
    ) -> List[EdgeDefect]:
        """
        對圖片四邊執行 CV 邊緣檢測

        Args:
            image: 原始高解析度影像 (灰階或彩色)
            raw_bounds: 產品實際邊界 (x1, y1, x2, y2)

        Returns:
            偵測到的邊緣缺陷列表
        """
        if not self.config.enabled:
            return []

        rx1, ry1, rx2, ry2 = raw_bounds
        img_h, img_w = image.shape[:2]
        all_defects: List[EdgeDefect] = []

        # 轉灰階
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        sides = [
            ("left", self.config.left, self._get_left_roi),
            ("right", self.config.right, self._get_right_roi),
            ("top", self.config.top, self._get_top_roi),
            ("bottom", self.config.bottom, self._get_bottom_roi),
        ]

        for side_name, side_cfg, roi_fn in sides:
            roi, offset_x, offset_y = roi_fn(gray, raw_bounds, side_cfg)
            if roi is None or roi.size == 0:
                continue

            defects = self._inspect_side(roi, side_name, side_cfg, offset_x, offset_y)
            
            # 排除區域過濾 (支援多組)
            all_zones = self.config.exclude_zones
            active_zones = [z for z in all_zones if z.enabled]
            if active_zones:
                filtered_defects = []
                for d in defects:
                    dx, dy, dw, dh = d.bbox
                    is_excluded = False
                    for zone in active_zones:
                        # 檢查是否與排除區域重疊 (Intersection check)
                        overlap = not (
                            dx + dw <= zone.x or
                            dx >= zone.x + zone.w or
                            dy + dh <= zone.y or
                            dy >= zone.y + zone.h
                        )
                        if overlap:
                            is_excluded = True
                            break
                    
                    if not is_excluded:
                        filtered_defects.append(d)
                defects = filtered_defects

            all_defects.extend(defects)

        return all_defects

    def inspect_single_side(
        self,
        image: np.ndarray,
        raw_bounds: Tuple[int, int, int, int],
        side: str,
        config_override: Optional[EdgeSideConfig] = None,
    ) -> Tuple[List[EdgeDefect], Dict[str, np.ndarray]]:
        """
        對單邊執行檢測，並回傳 debug 影像（用於 debug 頁面）

        Returns:
            (defects, debug_images) 其中 debug_images 包含:
              - "roi": 原始 ROI
              - "background": 估計背景
              - "diff": 差異圖
              - "mask": 二值化遮罩
              - "result": 標記缺陷的結果圖
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        side_map = {
            "left": (self.config.left, self._get_left_roi),
            "right": (self.config.right, self._get_right_roi),
            "top": (self.config.top, self._get_top_roi),
            "bottom": (self.config.bottom, self._get_bottom_roi),
        }

        if side not in side_map:
            return [], {}

        default_cfg, roi_fn = side_map[side]
        side_cfg = config_override or default_cfg
        roi, offset_x, offset_y = roi_fn(gray, raw_bounds, side_cfg)

        if roi is None or roi.size == 0:
            return [], {}

        defects, debug_imgs = self._inspect_side_debug(
            roi, side, side_cfg, offset_x, offset_y
        )
        return defects, debug_imgs

    def inspect_roi(
        self,
        roi: np.ndarray,
        offset_x: int = 0,
        offset_y: int = 0,
        otsu_bounds: Optional[Tuple[int, int, int, int]] = None,
        boundary_padding: int = 15,
        boundary_min_brightness: int = 15,
        panel_polygon: Optional[np.ndarray] = None,
    ) -> Tuple[List[EdgeDefect], Dict[str, Any]]:
        """
        對預先擷取的 ROI 執行邊緣缺陷檢測（用於 AOI 座標邊緣 defect）

        當 AOI 座標 defect 位於產品邊緣無法切出完整 512x512 tile 時，
        由 _create_aoi_coord_tiles 轉交此方法進行 CV 檢測。

        使用獨立的 aoi_threshold / aoi_min_area 參數，不共用四邊設定。

        Args:
            roi: 預先擷取的 ROI 影像（灰階或彩色）
            offset_x: ROI 左上角在原圖的 x 偏移
            offset_y: ROI 左上角在原圖的 y 偏移
            otsu_bounds: 產品前景 Otsu 邊界 (x1, y1, x2, y2)，用於建立前景遮罩
                         排除非產品區域對背景估計的干擾
            boundary_padding: 前景遮罩向外擴展的像素數，用於偵測 Otsu 邊界附近的缺陷
            boundary_min_brightness: 擴展區域中像素最低亮度閾值，低於此值視為真正背景
            panel_polygon: 面板 4 角 polygon (shape (4,2) 原圖絕對座標, 順序 TL/TR/BR/BL)。
                          若提供，fg_mask 會用 polygon rasterize（精確貼合面板邊緣，
                          且 polygon 已內縮 → 排除邊緣亮帶轉換區），優先於 otsu_bounds。

        Returns:
            (defects, stats)
            - defects: 偵測到的邊緣缺陷列表（座標已轉換為原圖絕對座標）
            - stats: 實際計算的統計資訊 {"max_diff", "max_area", "threshold", "min_area"}
        """
        empty_stats = {"max_diff": 0, "max_area": 0,
                       "threshold": self.config.aoi_threshold,
                       "min_area": self.config.aoi_min_area}

        # 注意：不檢查 self.config.enabled
        # AOI 座標邊緣 CV 偵測由呼叫端決定是否執行，不受全域 cv_edge_enabled 開關影響
        if roi is None or roi.size == 0:
            return [], empty_stats

        # 轉灰階
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # 建立前景遮罩：只在產品區域內做檢測，排除邊緣外黑色背景干擾
        fg_mask = None
        used_polygon = False  # 標記本次是否走 polygon 分支 (決定後續要 erode 還是 dilate)
        if panel_polygon is not None:
            used_polygon = True
            roi_h, roi_w = gray.shape[:2]
            fg_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            # 將 polygon 由原圖絕對座標轉成 ROI 局部座標再 rasterize
            local_poly = panel_polygon.copy().astype(np.float32)
            local_poly[:, 0] -= offset_x
            local_poly[:, 1] -= offset_y
            cv2.fillPoly(fg_mask, [local_poly.astype(np.int32)], 255)
            print(f"  🔲 inspect_roi: fg_mask 由 panel_polygon rasterize (4 角 ROI 局部座標 = "
                  f"{[(int(p[0]), int(p[1])) for p in local_poly]})")
        elif otsu_bounds is not None:
            ox1, oy1, ox2, oy2 = otsu_bounds
            roi_h, roi_w = gray.shape[:2]
            fg_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            # 將 Otsu 邊界轉換為 ROI 局部座標
            local_x1 = max(0, ox1 - offset_x)
            local_y1 = max(0, oy1 - offset_y)
            local_x2 = min(roi_w, ox2 - offset_x)
            local_y2 = min(roi_h, oy2 - offset_y)
            if local_x2 > local_x1 and local_y2 > local_y1:
                fg_mask[local_y1:local_y2, local_x1:local_x2] = 255

        if fg_mask is not None:
            if used_polygon:
                # Polygon 已精確貼合面板邊緣 → 內縮 N px 排除邊緣亮帶轉換區
                # 不做外擴 boundary_padding（polygon 模式下外擴只會把亮帶塞回前景）
                erode_n = int(getattr(self.config, 'aoi_polygon_erode_px', 0))
                if erode_n > 0 and np.any(fg_mask > 0):
                    erode_kernel = cv2.getStructuringElement(
                        cv2.MORPH_RECT,
                        (erode_n * 2 + 1, erode_n * 2 + 1)
                    )
                    before_px = int(np.count_nonzero(fg_mask))
                    fg_mask = cv2.erode(fg_mask, erode_kernel)
                    after_px = int(np.count_nonzero(fg_mask))
                    print(f"  🔲 inspect_roi: fg_mask polygon 內縮 -{erode_n} px ({before_px} → {after_px} px, 跳過 boundary_padding)")
            else:
                # 矩形 bbox 模式：保持原行為，向外擴展抓邊界附近亮像素
                if boundary_padding > 0 and np.any(fg_mask > 0):
                    dilate_kernel = cv2.getStructuringElement(
                        cv2.MORPH_RECT,
                        (boundary_padding * 2 + 1, boundary_padding * 2 + 1)
                    )
                    fg_mask_expanded = cv2.dilate(fg_mask, dilate_kernel, iterations=1)

                    expansion_zone = (fg_mask_expanded > 0) & (fg_mask == 0)
                    expansion_valid = expansion_zone & (gray >= boundary_min_brightness)
                    fg_mask[expansion_valid] = 255

                    expanded_px = int(np.count_nonzero(expansion_valid))
                    if expanded_px > 0:
                        print(f"  🔲 inspect_roi: fg_mask 邊界擴展 +{expanded_px} px (padding={boundary_padding}, min_bright={boundary_min_brightness})")

            fg_coverage = np.count_nonzero(fg_mask) / fg_mask.size * 100
            print(f"  🔲 inspect_roi: fg_mask coverage={fg_coverage:.1f}%, ROI=({offset_x},{offset_y}) {roi_w}x{roi_h}")

        # 使用 AOI 座標獨立參數 (不共用四邊設定)
        aoi_threshold = self.config.aoi_threshold
        aoi_min_area = self.config.aoi_min_area

        roi_cfg = EdgeSideConfig(
            width=0,
            threshold=aoi_threshold,
            min_area=aoi_min_area,
            exclude_top=0,
            exclude_bottom=0,
            exclude_left=0,
            exclude_right=0,
        )

        defects = self._inspect_side(gray, "aoi_edge", roi_cfg, offset_x, offset_y, fg_mask=fg_mask)

        # 為每個偵測到的缺陷記錄使用的判定參數
        for d in defects:
            d.threshold_used = aoi_threshold
            d.min_area_used = aoi_min_area

        # 計算統計供 header 顯示
        actual_max_diff = 0
        actual_max_area = 0
        if defects:
            # 有缺陷時直接從結果取統計，避免重新計算
            actual_max_diff = max(d.max_diff for d in defects)
            actual_max_area = max(d.area for d in defects)
        elif fg_mask is not None and np.any(fg_mask > 0):
            # 無缺陷時才做完整計算，供 debug 了解為何未偵測到
            k = self.config.blur_kernel
            blurred = cv2.GaussianBlur(gray, (k, k), 0)
            mk = clamp_median_kernel(self.config.median_kernel, min(gray.shape[:2]) - 1)
            blurred_for_bg = inpaint_non_fg_region(blurred, fg_mask)
            bg = cv2.medianBlur(blurred_for_bg, mk)
            diff = cv2.absdiff(blurred, bg)
            diff[fg_mask == 0] = 0
            actual_max_diff = int(np.max(diff)) if diff.size > 0 else 0

            _, thresh_mask = cv2.threshold(diff, aoi_threshold, 255, cv2.THRESH_BINARY)
            if self.config.aoi_morph_open_kernel > 0:
                mk_open = self.config.aoi_morph_open_kernel
                kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (mk_open, mk_open))
                thresh_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel_open)
            n_labels, _, stats_cc, _ = cv2.connectedComponentsWithStats(thresh_mask, connectivity=8)
            for i in range(1, n_labels):
                a = stats_cc[i, cv2.CC_STAT_AREA]
                if a > actual_max_area:
                    actual_max_area = a

            print(f"  🔲 inspect_roi: 未偵測到缺陷, max_diff={actual_max_diff}, max_area={actual_max_area}, "
                  f"threshold={aoi_threshold}, min_area={aoi_min_area}, median_k={mk}")

        roi_stats = {
            "max_diff": actual_max_diff,
            "max_area": actual_max_area,
            "threshold": aoi_threshold,
            "min_area": aoi_min_area,
        }

        return defects, roi_stats

    # ── ROI 擷取 ──────────────────────────────

    def _get_left_roi(
        self, gray: np.ndarray, raw_bounds: Tuple, cfg: EdgeSideConfig
    ) -> Tuple[Optional[np.ndarray], int, int]:
        rx1, ry1, rx2, ry2 = raw_bounds
        x1 = rx1
        x2 = min(rx1 + cfg.width, rx2)
        y1 = ry1 + cfg.exclude_top
        y2 = ry2 - cfg.exclude_bottom
        if x2 <= x1 or y2 <= y1:
            return None, 0, 0
        return gray[y1:y2, x1:x2], x1, y1

    def _get_right_roi(
        self, gray: np.ndarray, raw_bounds: Tuple, cfg: EdgeSideConfig
    ) -> Tuple[Optional[np.ndarray], int, int]:
        rx1, ry1, rx2, ry2 = raw_bounds
        x1 = max(rx2 - cfg.width, rx1)
        x2 = rx2
        y1 = ry1 + cfg.exclude_top
        y2 = ry2 - cfg.exclude_bottom
        if x2 <= x1 or y2 <= y1:
            return None, 0, 0
        return gray[y1:y2, x1:x2], x1, y1

    def _get_top_roi(
        self, gray: np.ndarray, raw_bounds: Tuple, cfg: EdgeSideConfig
    ) -> Tuple[Optional[np.ndarray], int, int]:
        rx1, ry1, rx2, ry2 = raw_bounds
        x1 = rx1 + cfg.exclude_left
        x2 = rx2 - cfg.exclude_right
        y1 = ry1
        y2 = min(ry1 + cfg.width, ry2)
        if x2 <= x1 or y2 <= y1:
            return None, 0, 0
        return gray[y1:y2, x1:x2], x1, y1

    def _get_bottom_roi(
        self, gray: np.ndarray, raw_bounds: Tuple, cfg: EdgeSideConfig
    ) -> Tuple[Optional[np.ndarray], int, int]:
        rx1, ry1, rx2, ry2 = raw_bounds
        x1 = rx1 + cfg.exclude_left
        x2 = rx2 - cfg.exclude_right
        y1 = max(ry2 - cfg.width, ry1)
        y2 = ry2
        if x2 <= x1 or y2 <= y1:
            return None, 0, 0
        return gray[y1:y2, x1:x2], x1, y1

    # ── 檢測邏輯 ──────────────────────────────

    def _inspect_side(
        self,
        roi: np.ndarray,
        side: str,
        cfg: EdgeSideConfig,
        offset_x: int,
        offset_y: int,
        fg_mask: Optional[np.ndarray] = None,
    ) -> List[EdgeDefect]:
        """對單邊 ROI 執行檢測

        Args:
            fg_mask: 前景遮罩 (255=前景, 0=背景)。若提供，非前景區域在背景估計前
                     會被替換為前景中值，避免邊緣外黑色背景干擾 median filter。
        """
        k = self.config.blur_kernel
        blurred = cv2.GaussianBlur(roi, (k, k), 0)

        blurred_for_bg = inpaint_non_fg_region(blurred, fg_mask) if fg_mask is not None else blurred

        mk = clamp_median_kernel(self.config.median_kernel, min(roi.shape[:2]) - 1)
        bg = cv2.medianBlur(blurred_for_bg, mk)

        diff = cv2.absdiff(blurred, bg)

        if fg_mask is not None:
            diff[fg_mask == 0] = 0

        _, mask = cv2.threshold(diff, cfg.threshold, 255, cv2.THRESH_BINARY)

        # 薄線偵測 (在 morph_open 之前，否則 1-px 寬線被殺掉)；直接產生 EdgeDefect 旁路 CC filter
        line_defects: List[EdgeDefect] = []
        if side == "aoi_edge" and self.config.aoi_line_min_length > 0:
            line_defects = self._detect_thin_lines(mask, diff, offset_x, offset_y)

        if side == "aoi_edge" and self.config.aoi_morph_open_kernel > 0:
            mk_open = self.config.aoi_morph_open_kernel
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (mk_open, mk_open))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        defects = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= cfg.min_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[i]

                # 該區域的最大差異值
                component_mask = (labels == i).astype(np.uint8)
                max_diff = int(np.max(diff[component_mask > 0])) if np.any(component_mask) else 0

                # max_diff 過濾：低對比度紋理雜訊 (max_diff 剛壓過 threshold) 視為非缺陷
                if side == "aoi_edge" and self.config.aoi_min_max_diff > 0:
                    if max_diff < self.config.aoi_min_max_diff:
                        print(f"  🔲 _inspect_side: 排除低對比 component "
                              f"(side={side}, area={area}, max_diff={max_diff} < {self.config.aoi_min_max_diff})")
                        continue

                # Solidity 過濾：L 形邊界偽影 solidity 極低 (~0.15)，真缺陷 >0.5
                solidity = 1.0
                if side == "aoi_edge":
                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        hull = cv2.convexHull(contours[0])
                        hull_area = cv2.contourArea(hull)
                        solidity = area / hull_area if hull_area > 0 else 1.0
                        if self.config.aoi_solidity_min > 0 and solidity < self.config.aoi_solidity_min:
                            print(f"  🔲 _inspect_side: 排除低 solidity component "
                                  f"(side={side}, area={area}, solidity={solidity:.3f} < {self.config.aoi_solidity_min})")
                            continue

                defects.append(EdgeDefect(
                    side=side,
                    area=area,
                    bbox=(offset_x + x, offset_y + y, w, h),
                    center=(int(offset_x + cx), int(offset_y + cy)),
                    max_diff=max_diff,
                    solidity=solidity,
                ))

        defects.extend(line_defects)
        return defects

    @staticmethod
    def _group_true_runs(bool_arr: np.ndarray, max_gap: int = 0) -> List[Tuple[int, int]]:
        """找 bool 陣列中連續 True 的區段 (允許中間最多 max_gap 個 False 跨接)。"""
        runs = []
        n = len(bool_arr)
        in_run = False
        start = 0
        last_true = -1
        gap = 0
        for i in range(n):
            if bool_arr[i]:
                if not in_run:
                    start = i
                    in_run = True
                last_true = i
                gap = 0
            else:
                if in_run:
                    gap += 1
                    if gap > max_gap:
                        runs.append((start, last_true))
                        in_run = False
        if in_run:
            runs.append((start, last_true))
        return runs

    def _detect_thin_lines(
        self,
        mask: np.ndarray,
        diff: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> List[EdgeDefect]:
        """投影法偵測薄線缺陷：垂直/水平連續活化像素數達門檻即視為線狀缺陷。

        在 morph_open 之前執行，避免 1-px 寬真實線條被 3x3 opening erode 歸零。
        產出的 EdgeDefect 旁路 CC 路徑後續過濾（min_area/solidity/min_max_diff）。
        """
        min_len = self.config.aoi_line_min_length
        max_w = self.config.aoi_line_max_width
        if min_len <= 0:
            return []

        # 先做 1D close 橋接點狀缺失 (線被 threshold 切成虛線時有機會還原)
        k_close_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        k_close_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        mask_v = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close_v)
        mask_h = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close_h)

        defects: List[EdgeDefect] = []

        # 垂直線：各 column 活化 pixel 數 ≥ min_len
        col_counts = (mask_v > 0).sum(axis=0)
        for x_start, x_end in self._group_true_runs(col_counts >= min_len, max_gap=1):
            line_w = x_end - x_start + 1
            if line_w > max_w:
                continue
            band = mask_v[:, x_start:x_end + 1]
            rows_any = band.any(axis=1)
            ys = np.where(rows_any)[0]
            if len(ys) < min_len:
                continue
            y_min, y_max = int(ys[0]), int(ys[-1])
            area = int((mask[:, x_start:x_end + 1] > 0).sum())  # 原 mask 活化數
            max_diff = int(diff[y_min:y_max + 1, x_start:x_end + 1].max())
            defects.append(EdgeDefect(
                side="aoi_edge",
                area=area,
                bbox=(offset_x + x_start, offset_y + y_min, line_w, y_max - y_min + 1),
                center=(int(offset_x + (x_start + x_end) / 2),
                        int(offset_y + (y_min + y_max) / 2)),
                max_diff=max_diff,
                solidity=1.0,
            ))
            print(f"  🔲 _detect_thin_lines: 垂直線 ({offset_x + x_start},{offset_y + y_min}) "
                  f"{line_w}x{y_max - y_min + 1}, area={area}, max_diff={max_diff}")

        # 水平線：各 row 活化 pixel 數 ≥ min_len
        row_counts = (mask_h > 0).sum(axis=1)
        for y_start, y_end in self._group_true_runs(row_counts >= min_len, max_gap=1):
            line_h = y_end - y_start + 1
            if line_h > max_w:
                continue
            band = mask_h[y_start:y_end + 1, :]
            cols_any = band.any(axis=0)
            xs = np.where(cols_any)[0]
            if len(xs) < min_len:
                continue
            x_min, x_max = int(xs[0]), int(xs[-1])
            area = int((mask[y_start:y_end + 1, :] > 0).sum())
            max_diff = int(diff[y_start:y_end + 1, x_min:x_max + 1].max())
            defects.append(EdgeDefect(
                side="aoi_edge",
                area=area,
                bbox=(offset_x + x_min, offset_y + y_start, x_max - x_min + 1, line_h),
                center=(int(offset_x + (x_min + x_max) / 2),
                        int(offset_y + (y_start + y_end) / 2)),
                max_diff=max_diff,
                solidity=1.0,
            ))
            print(f"  🔲 _detect_thin_lines: 水平線 ({offset_x + x_min},{offset_y + y_start}) "
                  f"{x_max - x_min + 1}x{line_h}, area={area}, max_diff={max_diff}")

        return defects

    def _inspect_side_debug(
        self,
        roi: np.ndarray,
        side: str,
        cfg: EdgeSideConfig,
        offset_x: int,
        offset_y: int,
    ) -> Tuple[List[EdgeDefect], Dict[str, np.ndarray]]:
        """對單邊 ROI 執行檢測，同時回傳 debug 影像"""
        import time
        t0 = time.time()
        print(f"[CV-DEBUG] start blur on shape={roi.shape}", flush=True)

        k = self.config.blur_kernel
        blurred = cv2.GaussianBlur(roi, (k, k), 0)

        mk = clamp_median_kernel(self.config.median_kernel, min(roi.shape[:2]) - 1)
        print(f"[CV-DEBUG] start medianBlur, k={mk}", flush=True)
        bg = cv2.medianBlur(blurred, mk)

        diff = cv2.absdiff(blurred, bg)
        _, mask = cv2.threshold(diff, cfg.threshold, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)  # 確保 uint8

        print(f"[CV-DEBUG] start connectedComponents", flush=True)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        print(f"[CV-DEBUG] connectedComponents done, num_labels={num_labels}, took {time.time()-t0:.2f}s", flush=True)

        # 結果圖 (BGR)
        result_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        defects = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= cfg.min_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[i]

                component_mask = (labels == i).astype(np.uint8)
                max_diff = int(np.max(diff[component_mask > 0])) if np.any(component_mask) else 0

                defects.append(EdgeDefect(
                    side=side,
                    area=area,
                    bbox=(offset_x + x, offset_y + y, w, h),
                    center=(int(offset_x + cx), int(offset_y + cy)),
                    max_diff=max_diff,
                ))

                # 畫框和標籤
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                label_text = f"A:{area}"
                cv2.putText(result_img, label_text, (x, max(y - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 差異圖加強對比 (適合人眼觀察)
        diff_colored = cv2.applyColorMap(
            np.clip(diff * 10, 0, 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )

        debug_imgs = {
            "roi": cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR),
            "background": cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR),
            "diff": diff_colored,
            "mask": cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            "result": result_img,
        }

        return defects, debug_imgs
