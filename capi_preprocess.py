"""共用前處理：訓練 / 推論皆使用此模組。

從 capi_inference.py 抽出 Otsu / panel polygon / tile 切分 / zone 分類，
讓訓練端與推論端走同一套邏輯。
"""
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Iterable
import numpy as np
import cv2


@dataclass
class PreprocessConfig:
    tile_size: int = 512
    tile_stride: int = 512
    otsu_offset: int = 5
    enable_panel_polygon: bool = True
    edge_threshold_px: int = 768  # retained for config compatibility; zone split is coverage-based
    coverage_min: float = 0.3
    # Edge 取樣 anchor 仍可往 panel 外推半個 tile，但實際 ROI 會再依 polygon 往產品
    # 內側推回來，避免 edge.pt 學到黑色背景邊界。
    # None → tile_size // 2；0 → 關閉額外 edge anchor。
    outer_edge_extend: Optional[int] = None

    def __post_init__(self):
        if self.outer_edge_extend is None:
            self.outer_edge_extend = self.tile_size // 2


@dataclass
class TileResult:
    tile_id: int
    x1: int; y1: int; x2: int; y2: int
    image: np.ndarray
    mask: Optional[np.ndarray]
    coverage: float
    zone: str  # "inner" | "edge" | "outside"
    center_dist_to_edge: float
    # 4 個 outer-extension 角 + 4 個 inner-edge 角；訓練 wizard 的「corners-only」
    # panel 模式只收 is_corner=True 的 tile（只給 edge 模型補強用）。
    is_corner: bool = False


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


def filter_panel_lighting_files(
    folder: Path,
    image_files: Optional[Iterable[Path]] = None,
) -> Dict[str, Path]:
    """從 panel folder 過濾出 5 個有效 lighting 圖。

    只保留檔名以 5 個 lighting prefix 開頭的圖；其他（S* 側拍 / B0F 黑屏 /
    PINIGBI / OMIT / Optics.log）自然被忽略。

    Returns: {"G0F00000": Path, ...}，缺哪個 lighting 就少哪個 key。
    """
    result: Dict[str, Path] = {}
    entries = image_files if image_files is not None else folder.iterdir()
    for entry in entries:
        entry = Path(entry)
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

    # 決定 zone：完整在 panel 內且外框未貼到 panel 邊界才是 inner。
    if coverage >= 1.0 - 1e-6:
        if _tile_touches_polygon_boundary(tile_rect, polygon):
            return "edge", 1.0, center_dist, None
        return "inner", 1.0, center_dist, None
    return "edge", coverage, center_dist, mask if coverage < 1.0 - 1e-6 else None


def _clamp_tile_origin(tx: int, ty: int, img_w: int, img_h: int,
                       tile_size: int) -> Tuple[int, int]:
    """Clamp a tile origin so the tile stays inside the image when possible."""
    max_x = max(0, img_w - tile_size)
    max_y = max(0, img_h - tile_size)
    return max(0, min(int(tx), max_x)), max(0, min(int(ty), max_y))


def _clamp_tile_origin_keep_anchor(
    tx: int,
    ty: int,
    anchor_xy: Tuple[int, int],
    img_w: int,
    img_h: int,
    tile_size: int,
) -> Tuple[int, int]:
    """Clamp tile origin to image bounds while keeping anchor inside when possible."""
    tx, ty = _clamp_tile_origin(tx, ty, img_w, img_h, tile_size)
    ax, ay = int(anchor_xy[0]), int(anchor_xy[1])
    max_x = max(0, img_w - tile_size)
    max_y = max(0, img_h - tile_size)

    lo_x = max(0, ax - tile_size + 1)
    hi_x = min(max_x, ax)
    if lo_x <= hi_x:
        tx = max(lo_x, min(tx, hi_x))

    lo_y = max(0, ay - tile_size + 1)
    hi_y = min(max_y, ay)
    if lo_y <= hi_y:
        ty = max(lo_y, min(ty, hi_y))

    return tx, ty


def _tile_polygon_coverage(
    tile_rect: Tuple[int, int, int, int],
    polygon: Optional[np.ndarray],
) -> float:
    """Return polygon coverage ratio inside a tile rectangle."""
    if polygon is None:
        return 1.0
    x1, y1, x2, y2 = tile_rect
    tile_w = max(0, x2 - x1)
    tile_h = max(0, y2 - y1)
    if tile_w <= 0 or tile_h <= 0:
        return 0.0
    mask = np.zeros((tile_h, tile_w), np.uint8)
    shifted = np.asarray(polygon, dtype=np.float32).copy()
    shifted[:, 0] -= x1
    shifted[:, 1] -= y1
    cv2.fillPoly(mask, [shifted.astype(np.int32)], 255)
    return float(np.count_nonzero(mask)) / float(tile_w * tile_h)


def _tile_corner_signed_distances(
    tx: int,
    ty: int,
    tile_size: int,
    polygon: np.ndarray,
) -> List[float]:
    poly = np.asarray(polygon, dtype=np.float32)
    # Use inclusive pixel corners. A tile ending on the polygon boundary is valid.
    x2 = tx + tile_size - 1
    y2 = ty + tile_size - 1
    corners = ((tx, ty), (x2, ty), (x2, y2), (tx, y2))
    return [
        float(cv2.pointPolygonTest(poly, (float(x), float(y)), True))
        for x, y in corners
    ]


def resolve_inward_polygon_tile(
    anchor_xy: Tuple[int, int],
    polygon: Optional[np.ndarray],
    image_shape: Tuple[int, int],
    tile_size: int,
    initial_origin: Optional[Tuple[int, int]] = None,
    target_coverage: float = 0.999,
    keep_anchor_inside: bool = False,
) -> Tuple[int, int, float, bool]:
    """Resolve a tile origin that stays inside the product polygon when possible.

    ``anchor_xy`` is the AOI point or the synthetic edge-sampling center. The
    returned tile is first clamped to the image, then iteratively pushed toward
    the polygon centroid until its corners are inside the polygon. If the panel
    geometry cannot fit a full tile, the best-coverage origin found so far is
    returned instead of padding or masking black background into the sample.
    When ``keep_anchor_inside`` is True, movement is constrained so the AOI
    anchor remains inside the tile whenever image bounds make that possible.

    Returns: ``(tx, ty, coverage, shifted)``.
    """
    img_h, img_w = image_shape[:2]
    half = tile_size // 2
    if initial_origin is None:
        tx = int(anchor_xy[0]) - half
        ty = int(anchor_xy[1]) - half
    else:
        tx, ty = int(initial_origin[0]), int(initial_origin[1])
    if keep_anchor_inside:
        tx, ty = _clamp_tile_origin_keep_anchor(tx, ty, anchor_xy, img_w, img_h, tile_size)
    else:
        tx, ty = _clamp_tile_origin(tx, ty, img_w, img_h, tile_size)
    original = (tx, ty)

    if polygon is None or len(polygon) < 3:
        return tx, ty, 1.0, False

    poly = np.asarray(polygon, dtype=np.float32)
    centroid = poly.mean(axis=0)

    def _score(origin: Tuple[int, int]) -> Tuple[float, float]:
        ox, oy = origin
        cov = _tile_polygon_coverage((ox, oy, ox + tile_size, oy + tile_size), poly)
        min_dist = min(_tile_corner_signed_distances(ox, oy, tile_size, poly))
        return cov, min_dist

    best = (tx, ty)
    best_cov, best_dist = _score(best)
    if best_cov >= target_coverage and best_dist >= -0.5:
        return tx, ty, best_cov, best != original

    for _ in range(64):
        distances = _tile_corner_signed_distances(tx, ty, tile_size, poly)
        min_dist = min(distances)
        if min_dist >= -0.5:
            cov = _tile_polygon_coverage((tx, ty, tx + tile_size, ty + tile_size), poly)
            if cov >= target_coverage:
                return tx, ty, cov, (tx, ty) != original

        x2 = tx + tile_size - 1
        y2 = ty + tile_size - 1
        corners = np.array(((tx, ty), (x2, ty), (x2, y2), (tx, y2)), dtype=np.float32)
        bad = [corners[i] for i, dist in enumerate(distances) if dist < 0.5]
        if not bad:
            bad = [corners[int(np.argmin(distances))]]
        direction = np.zeros(2, dtype=np.float32)
        for corner in bad:
            direction += centroid - corner
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            break

        step = max(1.0, min(float(tile_size), 1.0 - min_dist))
        delta = direction / norm * step
        dx = int(round(float(delta[0])))
        dy = int(round(float(delta[1])))
        if dx == 0 and abs(delta[0]) > 1e-6:
            dx = 1 if delta[0] > 0 else -1
        if dy == 0 and abs(delta[1]) > 1e-6:
            dy = 1 if delta[1] > 0 else -1

        if keep_anchor_inside:
            ntx, nty = _clamp_tile_origin_keep_anchor(
                tx + dx, ty + dy, anchor_xy, img_w, img_h, tile_size
            )
        else:
            ntx, nty = _clamp_tile_origin(tx + dx, ty + dy, img_w, img_h, tile_size)
        if (ntx, nty) == (tx, ty):
            break
        tx, ty = ntx, nty

        cov, dist = _score((tx, ty))
        if (cov, dist) > (best_cov, best_dist):
            best = (tx, ty)
            best_cov, best_dist = cov, dist

    return best[0], best[1], best_cov, best != original


def _tile_touches_polygon_boundary(
    tile_rect: Tuple[int, int, int, int],
    polygon: np.ndarray,
    tolerance_px: float = 1.0,
) -> bool:
    """True when the tile rectangle touches the panel polygon boundary."""
    x1, y1, x2, y2 = tile_rect
    corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    poly = polygon.astype(np.float32)
    for pt in corners:
        dist = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), True)
        if abs(float(dist)) <= tolerance_px:
            return True
    return False


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
        # 沿用 reference polygon；只需 bbox，跳過 polyfit 節省運算
        bbox_only_cfg = replace(config, enable_panel_polygon=False)
        bbox, _ = detect_panel_polygon(img, bbox_only_cfg)
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
    image_files: Optional[Iterable[Path]] = None,
) -> Dict[str, "PanelPreprocessResult"]:
    """處理整個 panel folder 的 5 lighting 圖。

    流程：filter 出 5 lighting → STANDARD 先處理取 reference polygon →
          其他 4 lighting 套 reference。STANDARD 失敗 fallback G0F00000。
    """
    files = filter_panel_lighting_files(folder, image_files=image_files)
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
    """在 bbox 範圍內走格子 + edge anchors，切 tile 並分類 zone。

    內圈 tile 仍以 bbox grid 為基礎；任何 edge tile 若有 polygon，會先往產品
    內側推到完整落在 polygon 內。``outer_edge_extend`` 現在只提供額外 edge
    anchor，不再讓訓練樣本看見 panel 外黑色背景。
    """
    x1, y1, x2, y2 = bbox
    ts = config.tile_size
    half = ts // 2
    img_h, img_w = img.shape[:2]

    def positions(lo: int, hi: int) -> List[int]:
        if hi - lo < ts:
            return []
        out = list(range(lo, hi - ts + 1, config.tile_stride))
        if out and out[-1] != hi - ts:
            out.append(hi - ts)
        return out

    xs = positions(x1, x2)
    ys = positions(y1, y2)

    extend = max(0, int(config.outer_edge_extend))
    push_top = min(extend, max(0, y1))
    push_bottom = min(extend, max(0, img_h - y2))
    push_left = min(extend, max(0, x1))
    push_right = min(extend, max(0, img_w - x2))

    top_ty = (y1 - push_top) if push_top > 0 else None
    bottom_ty = (y2 - ts + push_bottom) if push_bottom > 0 else None
    left_tx = (x1 - push_left) if push_left > 0 else None
    right_tx = (x2 - ts + push_right) if push_right > 0 else None

    extra_xs = []
    if left_tx is not None:
        extra_xs.append(left_tx)
    extra_xs.extend(xs)
    if right_tx is not None:
        extra_xs.append(right_tx)

    tiles: List[TileResult] = []
    emitted_positions = set()
    tid = 0

    inner_corner_xs = {xs[0], xs[-1]} if xs else set()
    inner_corner_ys = {ys[0], ys[-1]} if ys else set()
    outer_corner_xs = {x for x in (left_tx, right_tx) if x is not None}
    outer_corner_ys = {y for y in (top_ty, bottom_ty) if y is not None}

    def _is_corner(tx: int, ty: int) -> bool:
        return ((tx in inner_corner_xs and ty in inner_corner_ys)
                or (tx in outer_corner_xs and ty in outer_corner_ys))

    def _resolve_edge_origin(tx: int, ty: int) -> Tuple[int, int]:
        if polygon is None:
            return tx, ty
        anchor = (tx + half, ty + half)
        rx, ry, _cov, _shifted = resolve_inward_polygon_tile(
            anchor_xy=anchor,
            polygon=polygon,
            image_shape=(img_h, img_w),
            tile_size=ts,
            initial_origin=(tx, ty),
        )
        return rx, ry

    def _emit(tx: int, ty: int, zone: str, cov: float, dist: float, mask,
              is_corner_override: Optional[bool] = None) -> None:
        nonlocal tid
        if (tx, ty) in emitted_positions:
            return
        emitted_positions.add((tx, ty))
        tiles.append(TileResult(
            tile_id=tid,
            x1=tx, y1=ty, x2=tx + ts, y2=ty + ts,
            image=img[ty:ty + ts, tx:tx + ts].copy(),
            mask=mask,
            coverage=cov,
            zone=zone,
            center_dist_to_edge=dist,
            is_corner=_is_corner(tx, ty) if is_corner_override is None else is_corner_override,
        ))
        tid += 1

    for ty0 in ys:
        for tx0 in xs:
            zone, cov, dist, mask = classify_tile_zone((tx0, ty0, tx0 + ts, ty0 + ts), polygon, config)
            if zone == "outside":
                continue
            force_edge = zone == "edge" or (
                zone == "inner" and (tx0 == xs[0] or tx0 == xs[-1] or ty0 == ys[0] or ty0 == ys[-1])
            )
            tx, ty = tx0, ty0
            is_corner = _is_corner(tx0, ty0)
            if force_edge:
                tx, ty = _resolve_edge_origin(tx0, ty0)
                zone, cov, dist, mask = classify_tile_zone((tx, ty, tx + ts, ty + ts), polygon, config)
                zone = "edge"
                if cov >= 1.0 - 1e-6:
                    mask = None
            _emit(tx, ty, zone, cov, dist, mask, is_corner_override=is_corner)

    extension_positions: List[Tuple[int, int, bool]] = []
    if top_ty is not None:
        extension_positions.extend(
            (tx, top_ty, tx in outer_corner_xs and top_ty in outer_corner_ys)
            for tx in extra_xs
        )
    if bottom_ty is not None:
        extension_positions.extend(
            (tx, bottom_ty, tx in outer_corner_xs and bottom_ty in outer_corner_ys)
            for tx in extra_xs
        )
    if left_tx is not None:
        extension_positions.extend(
            (left_tx, ty, left_tx in outer_corner_xs and ty in outer_corner_ys)
            for ty in ys
        )
    if right_tx is not None:
        extension_positions.extend(
            (right_tx, ty, right_tx in outer_corner_xs and ty in outer_corner_ys)
            for ty in ys
        )

    for tx0, ty0, is_corner in extension_positions:
        tx, ty = _resolve_edge_origin(tx0, ty0)
        _zone, cov, dist, mask = classify_tile_zone((tx, ty, tx + ts, ty + ts), polygon, config)
        if cov >= 1.0 - 1e-6:
            mask = None
        _emit(tx, ty, "edge", cov, dist, mask, is_corner_override=is_corner)

    return tiles
