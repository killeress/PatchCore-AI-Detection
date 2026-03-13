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


@dataclass
class EdgeDefect:
    """單一邊緣缺陷"""
    side: str            # "left", "right", "top", "bottom"
    area: int            # 缺陷面積 (px)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) 在原圖絕對座標
    center: Tuple[int, int]          # (cx, cy) 中心點原圖絕對座標
    max_diff: int = 0    # 該區域最大灰階差值


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
    """邊緣檢測排除區域：檢測結果若與此區域重疊則被忽略"""
    enabled: bool = False
    x: int = 0
    y: int = 0
    w: int = 100
    h: int = 100


@dataclass
class EdgeInspectionConfig:
    """四邊檢測參數組"""
    enabled: bool = True
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
    exclude_zones: List[EdgeExclusionZoneConfig] = field(default_factory=list)

    @classmethod
    def from_db_params(cls, params: Dict[str, Any]) -> "EdgeInspectionConfig":
        """從 DB config_params 載入"""
        def get(key, default):
            v = params.get(key)
            if v is None:
                return default
            if isinstance(v, dict):
                return v.get("decoded_value", default)
            return v

        cfg = cls(
            enabled=get("cv_edge_enabled", True),
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
        
        
        # 排除區域 (支援多組區域列表)
        zones_json = get("cv_edge_exclude_zones", None)
        if zones_json and isinstance(zones_json, str):
            try:
                zones_data = json.loads(zones_json)
                if isinstance(zones_data, list):
                    cfg.exclude_zones = [
                        EdgeExclusionZoneConfig(
                            enabled=z.get("enabled", True),
                            x=int(z.get("x", 0)),
                            y=int(z.get("y", 0)),
                            w=int(z.get("w", 100)),
                            h=int(z.get("h", 100))
                        ) for z in zones_data
                    ]
            except:
                pass
        
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
            active_zones = [z for z in self.config.exclude_zones if z.enabled]
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
                            print(f"⚠️ 邊緣缺陷 (Side:{d.side}, Area:{d.area}) 落入排除區域 (x:{zone.x}, y:{zone.y}), 已被忽略")
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
    ) -> List[EdgeDefect]:
        """對單邊 ROI 執行檢測"""
        k = self.config.blur_kernel
        blurred = cv2.GaussianBlur(roi, (k, k), 0)

        mk = self.config.median_kernel
        # 中值濾波核必須為奇數且不超過 ROI 尺寸
        mk = min(mk, min(roi.shape[:2]) - 1)
        if mk % 2 == 0:
            mk -= 1
        if mk < 3:
            mk = 3
        bg = cv2.medianBlur(blurred, mk)

        diff = cv2.absdiff(blurred, bg)
        _, mask = cv2.threshold(diff, cfg.threshold, 255, cv2.THRESH_BINARY)

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

                defects.append(EdgeDefect(
                    side=side,
                    area=area,
                    bbox=(offset_x + x, offset_y + y, w, h),
                    center=(int(offset_x + cx), int(offset_y + cy)),
                    max_diff=max_diff,
                ))

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

        mk = self.config.median_kernel
        mk = min(mk, min(roi.shape[:2]) - 1)
        if mk % 2 == 0:
            mk -= 1
        if mk < 3:
            mk = 3
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
