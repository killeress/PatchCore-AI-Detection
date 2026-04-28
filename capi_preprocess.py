"""共用前處理：訓練 / 推論皆使用此模組。

從 capi_inference.py 抽出 Otsu / panel polygon / tile 切分 / zone 分類，
讓訓練端與推論端走同一套邏輯。
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import cv2


@dataclass
class PreprocessConfig:
    tile_size: int = 512
    tile_stride: int = 512
    otsu_offset: int = 5
    enable_panel_polygon: bool = True
    edge_threshold_px: int = 768
    coverage_min: float = 0.3


@dataclass
class TileResult:
    tile_id: int
    x1: int; y1: int; x2: int; y2: int
    image: np.ndarray
    mask: Optional[np.ndarray]
    coverage: float
    zone: str  # "inner" | "edge" | "outside"
    center_dist_to_edge: float


@dataclass
class PanelPreprocessResult:
    image_path: Path
    lighting: str
    foreground_bbox: Tuple[int, int, int, int]
    panel_polygon: Optional[np.ndarray]
    tiles: List[TileResult] = field(default_factory=list)
    polygon_detection_failed: bool = False


LIGHTING_PREFIXES = ("G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD")
SKIP_EXACT = ("Optics.log",)

EDGE_MARGIN = 20
SAMPLE_STEP = 50
OUTLIER_SIGMA = 3.0
MIN_EDGE_LEN_RATIO = 1.0
MIN_POLYGON_AREA_RATIO = 0.9
MIN_SAMPLES_PER_EDGE = 5


def filter_panel_lighting_files(folder: Path) -> Dict[str, Path]:
    """從 panel folder 過濾出 5 個有效 lighting 圖。

    只保留檔名以 5 個 lighting prefix 開頭的圖；其他（S* 側拍 / B0F 黑屏 /
    PINIGBI / OMIT / Optics.log）自然被忽略。

    Returns: {"G0F00000": Path, ...}，缺哪個 lighting 就少哪個 key。
    """
    result: Dict[str, Path] = {}
    for entry in folder.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name in SKIP_EXACT:
            continue
        # 優先比對 lighting prefix（STANDARD 開頭含 S，須先於 skip 判斷）
        matched = False
        for lighting in LIGHTING_PREFIXES:
            if name.startswith(lighting):
                if lighting not in result:
                    result[lighting] = entry
                matched = True
                break
        if matched:
            continue
    return result


def detect_panel_polygon(
    image: np.ndarray,
    config: PreprocessConfig,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """Otsu binarize → 最大連通輪廓 bbox → polyfit 4 角 polygon。

    Returns:
        (bbox, polygon)
        bbox = (x1, y1, x2, y2)，若 binarize 失敗回 (None, None)
        polygon = (4,2) float32 [TL, TR, BR, BL]，偵測失敗或 enable_panel_polygon=False 回 None
    """
    if image is None or image.size == 0:
        return None, None

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    offset = config.otsu_offset
    bbox = (x + offset, y + offset, x + w - offset, y + h - offset)

    if not config.enable_panel_polygon:
        return bbox, None

    polygon = _polyfit_polygon(binary, bbox, config.tile_size)
    return bbox, polygon


def _polyfit_polygon(
    binary_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    tile_size: int,
) -> Optional[np.ndarray]:
    """從 capi_inference._find_panel_polygon 抽出，邏輯不變。"""
    H, W = binary_mask.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    if xmax - xmin < 2 * EDGE_MARGIN or ymax - ymin < 2 * EDGE_MARGIN:
        return None

    tops, bots, lefts, rights = [], [], [], []
    for x in range(xmin + EDGE_MARGIN, xmax - EDGE_MARGIN, SAMPLE_STEP):
        if 0 <= x < W:
            ys = np.where(binary_mask[:, x] > 0)[0]
            if len(ys):
                tops.append((x, int(ys[0])))
                bots.append((x, int(ys[-1])))
    for y in range(ymin + EDGE_MARGIN, ymax - EDGE_MARGIN, SAMPLE_STEP):
        if 0 <= y < H:
            xs = np.where(binary_mask[y, :] > 0)[0]
            if len(xs):
                lefts.append((int(xs[0]), y))
                rights.append((int(xs[-1]), y))

    if min(len(tops), len(bots), len(lefts), len(rights)) < MIN_SAMPLES_PER_EDGE:
        return None

    def fit(pts, horizontal):
        arr = np.array(pts, dtype=float)
        ind, dep = (arr[:, 0], arr[:, 1]) if horizontal else (arr[:, 1], arr[:, 0])
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
        return float(a), float(b)

    top_l, bot_l, left_l, right_l = fit(tops, True), fit(bots, True), fit(lefts, False), fit(rights, False)
    if None in (top_l, bot_l, left_l, right_l):
        return None

    def intersect(h, v):
        a_h, b_h = h; a_v, b_v = v
        denom = 1.0 - a_h * a_v
        if abs(denom) < 1e-9:
            return None
        y = (a_h * b_v + b_h) / denom
        x = a_v * y + b_v
        return (x, y)

    TL, TR, BR, BL = intersect(top_l, left_l), intersect(top_l, right_l), intersect(bot_l, right_l), intersect(bot_l, left_l)
    if None in (TL, TR, BR, BL):
        return None

    polygon = np.array([TL, TR, BR, BL], dtype=np.float32)

    tol = 50
    if (polygon[:, 0].min() < -tol or polygon[:, 0].max() > W + tol or
            polygon[:, 1].min() < -tol or polygon[:, 1].max() > H + tol):
        return None

    min_edge_len = tile_size * MIN_EDGE_LEN_RATIO
    for i in range(4):
        if float(np.linalg.norm(polygon[(i + 1) % 4] - polygon[i])) < min_edge_len:
            return None

    bbox_area = float((xmax - xmin) * (ymax - ymin))
    poly_area = float(cv2.contourArea(polygon))
    if bbox_area <= 0 or poly_area < bbox_area * MIN_POLYGON_AREA_RATIO:
        return None

    return polygon


def classify_tile_zone(
    tile_rect: Tuple[int, int, int, int],
    polygon: Optional[np.ndarray],
    config: PreprocessConfig,
) -> Tuple[str, float, float, Optional[np.ndarray]]:
    """根據 polygon 與 tile 幾何決定 zone + 計算 coverage / center_dist。

    Returns: (zone, coverage, center_dist_to_edge, mask)
        - zone: "inner" | "edge" | "outside"
        - mask: tile 內 polygon 的 binary mask（uint8 0/255），fully inside 時 None
        - polygon=None → fallback ("inner", 1.0, inf, None)
    """
    x1, y1, x2, y2 = tile_rect
    tile_w = x2 - x1
    tile_h = y2 - y1

    if polygon is None:
        return "inner", 1.0, float("inf"), None

    # tile 內生成 polygon mask
    mask = np.zeros((tile_h, tile_w), np.uint8)
    shifted = polygon.copy()
    shifted[:, 0] -= x1
    shifted[:, 1] -= y1
    cv2.fillPoly(mask, [shifted.astype(np.int32)], 255)
    coverage = float((mask > 0).sum()) / (tile_w * tile_h)

    if coverage < config.coverage_min:
        return "outside", coverage, 0.0, mask

    # 計算 tile 中心到 polygon 邊的最短距離
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dists = []
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        d = _point_segment_dist((cx, cy), tuple(p1), tuple(p2))
        dists.append(d)
    center_dist = float(min(dists))

    # 決定 zone：完全在內 + 距邊 >= threshold → inner，其他 → edge
    if coverage >= 1.0 - 1e-6 and center_dist >= config.edge_threshold_px:
        return "inner", 1.0, center_dist, None
    return "edge", coverage, center_dist, mask if coverage < 1.0 - 1e-6 else None


def _point_segment_dist(p, a, b):
    px, py = p; ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    seg_sq = dx * dx + dy * dy
    if seg_sq < 1e-9:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_sq))
    qx, qy = ax + t * dx, ay + t * dy
    return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5


def preprocess_panel_image(
    image_path: Path,
    lighting: str,
    config: PreprocessConfig,
    reference_polygon: Optional[np.ndarray] = None,
) -> PanelPreprocessResult:
    """單張 lighting 圖完整前處理。

    1. 讀圖
    2. 偵測 panel polygon（或沿用 reference_polygon）
    3. 走 bbox grid 切 tile，分類 zone
    4. 回傳 PanelPreprocessResult
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"無法讀取圖片: {image_path}")

    if reference_polygon is not None:
        # 沿用 reference polygon；仍要跑 detect_panel_polygon 取 bbox
        bbox, _ = detect_panel_polygon(img, config)
        polygon = reference_polygon
        polygon_failed = False
    else:
        bbox, polygon = detect_panel_polygon(img, config)
        polygon_failed = config.enable_panel_polygon and polygon is None

    if bbox is None:
        return PanelPreprocessResult(
            image_path=image_path,
            lighting=lighting,
            foreground_bbox=(0, 0, 0, 0),
            panel_polygon=None,
            tiles=[],
            polygon_detection_failed=True,
        )

    tiles = _generate_tiles(img, bbox, polygon, config)
    return PanelPreprocessResult(
        image_path=image_path,
        lighting=lighting,
        foreground_bbox=bbox,
        panel_polygon=polygon,
        tiles=tiles,
        polygon_detection_failed=polygon_failed,
    )


def preprocess_panel_folder(
    folder: Path,
    config: PreprocessConfig,
) -> Dict[str, "PanelPreprocessResult"]:
    """處理整個 panel folder 的 5 lighting 圖。

    流程：filter 出 5 lighting → STANDARD 先處理取 reference polygon →
          其他 4 lighting 套 reference。STANDARD 失敗 fallback G0F00000。
    """
    files = filter_panel_lighting_files(folder)
    if not files:
        return {}

    # 決定 reference image：STANDARD > G0F00000 > W0F00000 > R0F00000 > WGF50500
    ref_lighting = None
    for cand in ("STANDARD", "G0F00000", "W0F00000", "R0F00000", "WGF50500"):
        if cand in files:
            ref_lighting = cand
            break
    if ref_lighting is None:
        return {}

    ref_result = preprocess_panel_image(files[ref_lighting], ref_lighting, config)
    if ref_result.polygon_detection_failed and ref_lighting != "G0F00000" and "G0F00000" in files:
        ref_lighting = "G0F00000"
        ref_result = preprocess_panel_image(files[ref_lighting], ref_lighting, config)

    results: Dict[str, PanelPreprocessResult] = {ref_lighting: ref_result}
    ref_poly = ref_result.panel_polygon
    for lighting, path in files.items():
        if lighting == ref_lighting:
            continue
        results[lighting] = preprocess_panel_image(path, lighting, config, reference_polygon=ref_poly)
    return results


def _generate_tiles(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    polygon: Optional[np.ndarray],
    config: PreprocessConfig,
) -> List[TileResult]:
    """在 bbox 範圍內走格子，切 tile 並分類 zone。outside tile 直接略過。"""
    x1, y1, x2, y2 = bbox
    ts = config.tile_size

    def positions(lo: int, hi: int) -> List[int]:
        if hi - lo < ts:
            return []
        out = list(range(lo, hi - ts + 1, config.tile_stride))
        if out and out[-1] != hi - ts:
            out.append(hi - ts)
        return out

    xs = positions(x1, x2)
    ys = positions(y1, y2)
    tiles: List[TileResult] = []
    tid = 0
    for ty in ys:
        for tx in xs:
            tile_rect = (tx, ty, tx + ts, ty + ts)
            zone, cov, dist, mask = classify_tile_zone(tile_rect, polygon, config)
            if zone == "outside":
                continue
            tile_img = img[ty:ty + ts, tx:tx + ts].copy()
            tiles.append(TileResult(
                tile_id=tid,
                x1=tx, y1=ty, x2=tx + ts, y2=ty + ts,
                image=tile_img,
                mask=mask,
                coverage=cov,
                zone=zone,
                center_dist_to_edge=dist,
            ))
            tid += 1
    return tiles
