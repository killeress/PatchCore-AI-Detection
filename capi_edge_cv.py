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


def compute_boundary_band_mask(
    roi_shape: Tuple[int, int],
    roi_origin: Tuple[int, int],
    panel_polygon: Optional[list],
    band_px: int,
    fg_mask: np.ndarray,
) -> np.ndarray:
    """Phase 6 fusion — 計算 ROI 內 boundary band mask。

    Band = 「ROI 內所有距 polygon 邊 ≤ band_px 的像素」且「在 fg_mask 內」。
    語意：fusion 模式下 CV 管轄區（PC backbone 感受野受 panel 邊界污染處）。

    Args:
        roi_shape: (h, w) ROI 尺寸
        roi_origin: (ox, oy) ROI 左上角在 panel 原圖的絕對座標
        panel_polygon: panel 邊界多邊形頂點 list[(x, y)] (panel 座標系)；None → 回空 band
        band_px: band 寬度 (px)；≤0 → 回空 band
        fg_mask: ROI 大小的前景遮罩 (uint8, 0/255)

    Returns:
        ROI 大小 uint8 mask，0/255。
        - 邊界情境（polygon=None / band_px≤0 / ROI 內無 polygon edge）→ 全 0
    """
    h, w = roi_shape[:2]
    if panel_polygon is None or band_px <= 0:
        return np.zeros((h, w), dtype=np.uint8)

    roi_ox, roi_oy = roi_origin
    poly_roi = np.array(
        [(int(x - roi_ox), int(y - roi_oy)) for x, y in panel_polygon],
        dtype=np.int32,
    )

    edge_img = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(edge_img, [poly_roi], isClosed=True, color=255, thickness=1)

    if not edge_img.any():
        # ROI 完全在 panel 內或外、polygon 邊未進入 ROI → band 空
        return np.zeros((h, w), dtype=np.uint8)

    kernel_size = 2 * band_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    band_near_edge = cv2.dilate(edge_img, kernel)
    return cv2.bitwise_and(band_near_edge, fg_mask)


def _point_to_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> Tuple[float, float, float]:
    """Returns (distance, closest_x, closest_y) — 點 P 到線段 AB 的最短距離與最近點。"""
    dx = bx - ax
    dy = by - ay
    len_sq = dx * dx + dy * dy
    if len_sq <= 1e-9:
        closest_x, closest_y = ax, ay
    else:
        t = ((px - ax) * dx + (py - ay) * dy) / len_sq
        t = max(0.0, min(1.0, t))
        closest_x = ax + t * dx
        closest_y = ay + t * dy
    ddx = px - closest_x
    ddy = py - closest_y
    return (ddx * ddx + ddy * ddy) ** 0.5, closest_x, closest_y


def _nearest_polygon_edge_info(
    aoi_x: float, aoi_y: float, polygon: np.ndarray
) -> Tuple[float, float, float]:
    """返回 AOI 到 polygon 所有邊的「最近點距離、最近點 x、最近點 y」。

    只回無號距離；caller 用 cv2.pointPolygonTest 再決定內/外號位。
    """
    best_d = float("inf")
    best_cx, best_cy = 0.0, 0.0
    n = len(polygon)
    for i in range(n):
        ax, ay = float(polygon[i][0]), float(polygon[i][1])
        bx, by = float(polygon[(i + 1) % n][0]), float(polygon[(i + 1) % n][1])
        d, cx, cy = _point_to_segment_distance(aoi_x, aoi_y, ax, ay, bx, by)
        if d < best_d:
            best_d = d
            best_cx, best_cy = cx, cy
    return best_d, best_cx, best_cy


def compute_pc_roi_offset(
    aoi_xy: Tuple[int, int],
    polygon: Optional[np.ndarray],
    band_px: int = 40,
    aoi_margin_px: int = 64,
    roi_size: int = 512,
) -> Tuple[Tuple[int, int], Tuple[int, int], float]:
    """Phase 7 — 計算 PC ROI 應有的內移偏移量。

    目標：把 PC ROI 從 centered 位置往 panel 內側偏移，使偏移後 PC ROI 距
    polygon 邊 ≥ `band_px`，讓 backbone feature map 完全脫離 panel 邊
    discontinuity；同時 AOI 座標必須仍在 PC ROI 內，距邊 ≥ `aoi_margin_px`。

    偏移方向 = AOI 座標到最近 polygon 邊的 inward normal（指向 panel 內）。
    偏移量大小 = max(0, band_px - (d_edge - roi_size/2))
               然後 clamp 到 [0, roi_size/2 - aoi_margin_px]。

    Args:
        aoi_xy: AOI 座標 (x, y) panel 絕對座標
        polygon: panel 邊界 Nx2 np.ndarray，None / 頂點<3 → 不偏移
        band_px: 希望 PC ROI 距 polygon 的最小距離 (px)
        aoi_margin_px: AOI 座標距 PC ROI 邊的最小 margin (px)
        roi_size: ROI 邊長 (px)

    Returns:
        (pc_roi_origin, shift_vec, d_edge)
        - pc_roi_origin: (ox, oy) 偏移後 PC ROI 左上角 panel 絕對座標
        - shift_vec: (dx, dy) 相對 centered ROI 的偏移量；(0,0) 代表無偏移
        - d_edge: AOI 座標到 polygon 的有號距離（內側 >0，外側 <0）；polygon 無效回 0

    Rules:
        polygon=None / 頂點<3 → shift=(0,0), origin=centered, d_edge=0
        AOI 在 polygon 外（d_edge<0） → shift=(0,0), origin=centered
        Deep interior (d_edge >= roi_size/2 + band_px) → shift=(0,0)
    """
    aoi_x, aoi_y = int(aoi_xy[0]), int(aoi_xy[1])
    half = roi_size // 2
    centered_origin = (aoi_x - half, aoi_y - half)

    if polygon is None or len(polygon) < 3:
        return centered_origin, (0, 0), 0.0

    polygon_int = np.asarray(polygon, dtype=np.int32)

    # 內外判定 — 在 polygon 外則不偏移
    signed_dist = float(cv2.pointPolygonTest(
        polygon_int, (float(aoi_x), float(aoi_y)), True
    ))
    if signed_dist <= 0:
        return centered_origin, (0, 0), signed_dist

    # 需要的偏移量（往內側）
    needed = band_px - (signed_dist - half)
    if needed <= 0:
        return centered_origin, (0, 0), signed_dist

    max_shift = max(0, half - aoi_margin_px)
    actual = int(min(needed, max_shift))
    if actual <= 0:
        return centered_origin, (0, 0), signed_dist

    # 找最近 polygon 邊上的點 → inward normal = AOI - closest_pt（指向 panel 內）
    _, cx, cy = _nearest_polygon_edge_info(float(aoi_x), float(aoi_y), polygon_int)
    nx = float(aoi_x) - cx
    ny = float(aoi_y) - cy
    nlen = (nx * nx + ny * ny) ** 0.5
    if nlen <= 1e-6:
        # AOI 剛好在 polygon 邊上，無法決定方向 → 不偏移
        return centered_origin, (0, 0), signed_dist

    # 取主軸：axis-aligned polygon 下 normal 會是純 x 或純 y
    # 若 |nx| >> |ny| 取水平偏移，反之垂直；相近則仍取較大者
    if abs(nx) >= abs(ny):
        dx = int(round((nx / abs(nx)) * actual))
        dy = 0
    else:
        dx = 0
        dy = int(round((ny / abs(ny)) * actual))

    pc_origin = (aoi_x + dx - half, aoi_y + dy - half)
    return pc_origin, (dx, dy), signed_dist


def verify_polygon_clear_of_pc_roi(
    pc_roi_origin: Tuple[int, int],
    roi_size: int,
    polygon: Optional[np.ndarray],
    band_px: int,
) -> bool:
    """Phase 7 — 驗證偏移後 PC ROI 內部所有像素距 polygon 邊 ≥ band_px。

    做法：
      1. 逐條 polygon 邊，取線段端點在 PC ROI 局部座標系的位置
      2. 若該線段與 PC ROI 矩形有任一點在 ROI 內側 < band_px → False
      3. 另也檢查 polygon 頂點是否在 PC ROI 邊界 band_px 範圍內

    凹角 / 複雜形狀情境：polygon 角點從另一側逼近 PC ROI → 此函式回 False
    讓 caller 退回 centered + band_mask 排除的 Phase 6 行為。

    Args:
        pc_roi_origin: (ox, oy) PC ROI 左上角
        roi_size: ROI 邊長
        polygon: Nx2 np.ndarray；None → True（無 polygon 不限制）
        band_px: 最小允許距離

    Returns:
        True：polygon 所有邊距 PC ROI ≥ band_px（安全偏移）
        False：有邊侵入或太近（需 fallback）
    """
    if polygon is None or len(polygon) < 2:
        return True

    ox, oy = pc_roi_origin
    # PC ROI bounds (panel 絕對座標)
    x1, y1 = ox, oy
    x2, y2 = ox + roi_size, oy + roi_size

    n = len(polygon)
    for i in range(n):
        ax = float(polygon[i][0])
        ay = float(polygon[i][1])
        bx = float(polygon[(i + 1) % n][0])
        by = float(polygon[(i + 1) % n][1])

        # 線段 AB 任一點距 PC ROI 矩形 < band_px?
        # 做法：對 PC ROI 四角測距線段，再對線段頂點測距 PC ROI 矩形
        # 簡化：取 PC ROI 矩形內最近點到線段 AB 的距離
        dist = _segment_to_rect_distance(ax, ay, bx, by, x1, y1, x2, y2)
        if dist < band_px:
            return False
    return True


def classify_pc_roi_verify_failure(
    aoi_xy: Tuple[int, int],
    pc_roi_origin: Tuple[int, int],
    roi_size: int,
    polygon: np.ndarray,
    band_px: int,
) -> str:
    """Phase 7.1 — fallback 原因分類（verify 失敗時呼叫）。

    區分兩種常見情境以產生更精確的 UI / log tag：

    - `shift_insufficient`：唯一侵入 PC ROI 的 polygon 線段就是「AOI 最近的那條邊」
      —— 代表 shift 方向正確，只是 max_shift (= roi_size/2 - aoi_margin_px) 不足以
      把這條邊清到 band_px 之外。常見於 axis-aligned 四邊形 + AOI 距邊極近的情境。

    - `concave_polygon`：有多條邊同時侵入，或唯一侵入的邊不是最近邊 →
      polygon 形狀導致 shift 沿單一 inward normal 無法清場；典型為凹角 / L 形。

    Args:
        aoi_xy: AOI 座標 panel 絕對座標
        pc_roi_origin: shifted PC ROI 左上角
        roi_size: ROI 邊長
        polygon: panel polygon Nx2
        band_px: 判定門檻（同 fusion band 寬度）

    Returns:
        "shift_insufficient" 或 "concave_polygon"
    """
    if polygon is None or len(polygon) < 2:
        return "concave_polygon"  # 防呆：理論上 caller 不該在此情況下來

    # 找 AOI 最近邊的 index（與 compute_pc_roi_offset 同演算法，確保一致）
    n = len(polygon)
    best_d = float("inf")
    nearest_edge_idx = 0
    for i in range(n):
        ax, ay = float(polygon[i][0]), float(polygon[i][1])
        bx, by = float(polygon[(i + 1) % n][0]), float(polygon[(i + 1) % n][1])
        d, _, _ = _point_to_segment_distance(
            float(aoi_xy[0]), float(aoi_xy[1]), ax, ay, bx, by
        )
        if d < best_d:
            best_d = d
            nearest_edge_idx = i

    # 找所有違反 band_px 的邊 index
    ox, oy = pc_roi_origin
    x1, y1 = float(ox), float(oy)
    x2, y2 = float(ox + roi_size), float(oy + roi_size)
    violating_indices = []
    for i in range(n):
        ax, ay = float(polygon[i][0]), float(polygon[i][1])
        bx, by = float(polygon[(i + 1) % n][0]), float(polygon[(i + 1) % n][1])
        dist = _segment_to_rect_distance(ax, ay, bx, by, x1, y1, x2, y2)
        if dist < band_px:
            violating_indices.append(i)

    # 僅 1 條邊違反 且 正好是最近邊 → shift_insufficient
    # 其他情境（多邊違反 或 違反的不是最近邊）→ concave_polygon
    if len(violating_indices) == 1 and violating_indices[0] == nearest_edge_idx:
        return "shift_insufficient"
    return "concave_polygon"


def _segment_to_rect_distance(
    ax: float, ay: float, bx: float, by: float,
    rx1: float, ry1: float, rx2: float, ry2: float,
) -> float:
    """Returns minimum distance from line segment AB to axis-aligned rectangle.

    若線段穿過矩形 → 0
    否則取「矩形四角到線段距離」與「線段兩端點到矩形距離」的最小值
    """
    # 線段是否穿過矩形（cohen-sutherland 式粗略判定 + 精確 intersection check）
    if _segment_intersects_rect(ax, ay, bx, by, rx1, ry1, rx2, ry2):
        return 0.0

    # 矩形四角到線段
    best = float("inf")
    for (cx, cy) in [(rx1, ry1), (rx2, ry1), (rx1, ry2), (rx2, ry2)]:
        d, _, _ = _point_to_segment_distance(cx, cy, ax, ay, bx, by)
        if d < best:
            best = d
    # 線段兩端點到矩形
    for (px, py) in [(ax, ay), (bx, by)]:
        cx = min(max(px, rx1), rx2)
        cy = min(max(py, ry1), ry2)
        d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        if d < best:
            best = d
    return best


def _segment_intersects_rect(
    ax: float, ay: float, bx: float, by: float,
    rx1: float, ry1: float, rx2: float, ry2: float,
) -> bool:
    """檢查線段 AB 是否與 axis-aligned 矩形相交（含端點在矩形內）"""
    # 端點任一在矩形內
    if (rx1 <= ax <= rx2 and ry1 <= ay <= ry2) or (rx1 <= bx <= rx2 and ry1 <= by <= ry2):
        return True
    # 線段與矩形四邊相交？對每條矩形邊做線段交測
    edges = [
        (rx1, ry1, rx2, ry1),  # top
        (rx2, ry1, rx2, ry2),  # right
        (rx2, ry2, rx1, ry2),  # bottom
        (rx1, ry2, rx1, ry1),  # left
    ]
    for (ex1, ey1, ex2, ey2) in edges:
        if _segments_intersect(ax, ay, bx, by, ex1, ey1, ex2, ey2):
            return True
    return False


def _segments_intersect(
    p1x: float, p1y: float, p2x: float, p2y: float,
    p3x: float, p3y: float, p4x: float, p4y: float,
) -> bool:
    """標準線段交測"""
    def ccw(ax, ay, bx, by, cx, cy):
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)
    return (ccw(p1x, p1y, p3x, p3y, p4x, p4y) != ccw(p2x, p2y, p3x, p3y, p4x, p4y)
            and ccw(p1x, p1y, p2x, p2y, p3x, p3y) != ccw(p1x, p1y, p2x, p2y, p4x, p4y))


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
    min_max_diff_used: int = 0    # 使用的 min_max_diff 下限 (用於 CV OK 原因推斷，0=未啟用)
    inspector_mode: str = "cv"    # "cv" | "patchcore" | "fusion" (Phase 6 新增 fusion 為 ROI 層模式)
    patchcore_score: float = 0.0
    patchcore_threshold: float = 0.0
    patchcore_ok_reason: str = ""
    # Phase 6 fusion 欄位 — 個別 defect 層 (與 inspector_mode 並存)
    source_inspector: str = ""    # fusion 模式下："cv"=來自 band / "patchcore"=來自 interior；非 fusion 為 ""
    d_edge_px: float = 0.0        # defect center 到 panel polygon 邊距離 (px)；fusion 模式才有值，其餘 0
    fusion_fallback_reason: str = ""  # fusion 模式下退回 fallback 原因 (例 "polygon_unavailable")；正常 fusion 為 ""
    # Phase 7: PC ROI 內移欄位 (fusion 專用，非 fusion 全為預設)
    pc_roi_origin_x: int = 0      # PC 實際跑的 ROI 左上角 x (panel 絕對座標)
    pc_roi_origin_y: int = 0      # PC 實際跑的 ROI 左上角 y (panel 絕對座標)
    pc_roi_shift_dx: int = 0      # 相對 centered ROI 的偏移 x (往 panel 內為正/負依 inward normal)
    pc_roi_shift_dy: int = 0      # 相對 centered ROI 的偏移 y
    pc_roi_fallback_reason: str = ""  # PC ROI 內移 fallback 原因: "" / "concave_polygon" / "shift_disabled" / "polygon_unavailable"
    # 推論當下保留的 render artifacts (async heatmap 用，不入 DB)
    pc_roi: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    pc_fg_mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    pc_anomaly_map: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    # CV 路徑真實過濾後 mask（_inspect_side 輸出），heatmap 直接疊色不再重算 threshold
    cv_filtered_mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    cv_mask_offset: Tuple[int, int] = (0, 0)  # mask 左上角原圖絕對座標 (對位用)


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
    aoi_edge_inspector: str = "cv"  # "cv" | "patchcore" | "fusion"
    aoi_edge_boundary_band_px: int = 40  # Phase 6 fusion 模式 CV 管轄帶寬度 (polygon 邊往 panel 內延伸 px); 0=等同 patchcore
    aoi_edge_pc_roi_inward_shift_enabled: bool = True  # Phase 7: fusion 模式 PC ROI 自動內移到距 polygon ≥ band_px
    aoi_edge_aoi_margin_px: int = 64  # Phase 7: PC ROI 內移時 AOI 座標距 PC ROI 邊最小 margin (px)
    aoi_edge_pc_shift_band_px: int = 0  # Phase 7.1c: PC ROI shift target / verify 寬帶 (0=最大程度利用 shift, >0=留 buffer 避開 polygon 邊 discontinuity，清不到時走 fallback)
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
        cfg.aoi_edge_inspector = inspector_val if inspector_val in ("cv", "patchcore", "fusion") else "cv"
        cfg.aoi_edge_boundary_band_px = int(get("aoi_edge_boundary_band_px", 40))
        cfg.aoi_edge_pc_roi_inward_shift_enabled = bool(get("aoi_edge_pc_roi_inward_shift_enabled", True))
        cfg.aoi_edge_aoi_margin_px = int(get("aoi_edge_aoi_margin_px", 64))
        cfg.aoi_edge_pc_shift_band_px = int(get("aoi_edge_pc_shift_band_px", 0))

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

            defects, _ = self._inspect_side(roi, side_name, side_cfg, offset_x, offset_y)
            
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
                       "min_area": self.config.aoi_min_area,
                       "min_max_diff": self.config.aoi_min_max_diff}

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

        defects, filtered_mask = self._inspect_side(gray, "aoi_edge", roi_cfg, offset_x, offset_y, fg_mask=fg_mask)

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
            "min_max_diff": self.config.aoi_min_max_diff,
            "filtered_mask": filtered_mask,          # ROI 局部座標過濾後 mask，供 heatmap pixel 還原
            "roi_offset": (int(offset_x), int(offset_y)),  # ROI 左上角原圖絕對座標
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
    ) -> Tuple[List[EdgeDefect], np.ndarray]:
        """對單邊 ROI 執行檢測

        Args:
            fg_mask: 前景遮罩 (255=前景, 0=背景)。若提供，非前景區域在背景估計前
                     會被替換為前景中值，避免邊緣外黑色背景干擾 median filter。

        Returns:
            (defects, filtered_mask) — filtered_mask 為 ROI 局部尺寸 uint8 (255=通過
            所有過濾並納入判定的 pixel / 0=被過濾或非前景)，供上游生 heatmap 時直接
            還原實際判定用 mask，不用再靠 threshold 重算。
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

        filtered_mask = np.zeros_like(mask, dtype=np.uint8)

        # 薄線偵測 (在 morph_open 之前，否則 1-px 寬線被殺掉)；直接產生 EdgeDefect 旁路 CC filter
        line_defects: List[EdgeDefect] = []
        if side == "aoi_edge" and self.config.aoi_line_min_length > 0:
            line_defects, line_mask = self._detect_thin_lines(mask, diff, offset_x, offset_y)
            if line_mask is not None and line_mask.shape == filtered_mask.shape:
                filtered_mask = cv2.bitwise_or(filtered_mask, line_mask)

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

                filtered_mask[component_mask > 0] = 255
                defects.append(EdgeDefect(
                    side=side,
                    area=area,
                    bbox=(offset_x + x, offset_y + y, w, h),
                    center=(int(offset_x + cx), int(offset_y + cy)),
                    max_diff=max_diff,
                    solidity=solidity,
                ))

        defects.extend(line_defects)
        return defects, filtered_mask

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
    ) -> Tuple[List[EdgeDefect], np.ndarray]:
        """投影法偵測薄線缺陷：垂直/水平連續活化像素數達門檻即視為線狀缺陷。

        在 morph_open 之前執行，避免 1-px 寬真實線條被 3x3 opening erode 歸零。
        產出的 EdgeDefect 旁路 CC 路徑後續過濾（min_area/solidity/min_max_diff）。

        Returns:
            (defects, line_mask) — line_mask 為 ROI 局部尺寸 uint8，255=通過線偵測
            的 pixel，供上游累積到 filtered_mask 還原真實判定區域。
        """
        min_len = self.config.aoi_line_min_length
        max_w = self.config.aoi_line_max_width
        line_mask = np.zeros_like(mask, dtype=np.uint8)
        if min_len <= 0:
            return [], line_mask

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
            line_mask[y_min:y_max + 1, x_start:x_end + 1] = cv2.bitwise_or(
                line_mask[y_min:y_max + 1, x_start:x_end + 1],
                mask_v[y_min:y_max + 1, x_start:x_end + 1],
            )
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
            line_mask[y_start:y_end + 1, x_min:x_max + 1] = cv2.bitwise_or(
                line_mask[y_start:y_end + 1, x_min:x_max + 1],
                mask_h[y_start:y_end + 1, x_min:x_max + 1],
            )
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

        return defects, line_mask

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
