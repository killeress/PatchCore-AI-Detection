"""共用前處理：訓練 / 推論皆使用此模組。

從 capi_inference.py 抽出 Otsu / panel polygon / tile 切分 / zone 分類，
讓訓練端與推論端走同一套邏輯。
"""
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Iterable, Any, Callable
import logging
import numpy as np
import cv2
from capi_image_orientation import read_detection_image
from capi_image_preprocess_lab import apply_preprocess_method, normalize_preprocess_pipeline
from capi_image_naming import canonical_image_prefix


logger = logging.getLogger("capi.preprocess")


# AOI product-coordinate mapping uses this legacy fallback when the model or
# inference record does not carry an explicit product resolution.
DEFAULT_PRODUCT_RESOLUTION = (1920, 1080)


def rect_polygon_from_bounds(
    bounds: Optional[Tuple[int, int, int, int]],
) -> Optional[np.ndarray]:
    """Build the same inclusive rectangular panel polygon used by inference."""
    if bounds is None:
        return None
    x1, y1, x2, y2 = (int(value) for value in bounds)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array(
        [[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]],
        dtype=np.float32,
    )


def resolve_aoi_inward_shift_axes(
    img_x: int,
    img_y: int,
    bounds: Tuple[int, int, int, int],
    tile_size: int,
) -> str:
    """Return the panel axis that an AOI tile is allowed to shift inward on."""
    half = int(tile_size) // 2
    x1, y1, x2, y2 = (int(value) for value in bounds)
    near_top_or_bottom = (img_y - y1 < half) or (y2 - img_y < half)
    near_left_or_right = (img_x - x1 < half) or (x2 - img_x < half)
    if near_top_or_bottom and not near_left_or_right:
        return "y"
    if near_left_or_right and not near_top_or_bottom:
        return "x"
    return "xy"


def map_product_coord_to_image(
    px: int,
    py: int,
    raw_bounds: Tuple[int, int, int, int],
    product_resolution: Optional[Tuple[int, int]] = None,
    panel_polygon: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    """Map a product AOI coordinate using the formal inference mapping.

    The linear mapping is retained as the first choice.  When a valid panel
    polygon shows that the linear point is outside a contaminated/raw bound,
    the same four-corner perspective correction as production inference is
    used instead.  Keeping this helper shared prevents NG training crops from
    being sampled from a different coordinate system than validation.
    """
    product_width, product_height = product_resolution or DEFAULT_PRODUCT_RESOLUTION
    x_start, y_start, x_end, y_end = (int(value) for value in raw_bounds)
    product_img_width = x_end - x_start
    product_img_height = y_end - y_start
    if product_width <= 0 or product_height <= 0:
        product_width, product_height = DEFAULT_PRODUCT_RESOLUTION

    img_x = int(px * product_img_width / product_width + x_start)
    img_y = int(py * product_img_height / product_height + y_start)

    if panel_polygon is None:
        return img_x, img_y

    try:
        polygon = np.asarray(panel_polygon, dtype=np.float32).reshape(-1, 2)
        if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
            return img_x, img_y

        raw_distance = float(cv2.pointPolygonTest(
            polygon, (float(img_x), float(img_y)), True,
        ))
        raw_area = float(max(1, product_img_width * product_img_height))
        polygon_coverage = abs(float(cv2.contourArea(polygon))) / raw_area
        raw_bounds_contaminated = polygon_coverage < 0.90
        if raw_distance >= -1.0 and not raw_bounds_contaminated:
            return img_x, img_y

        source = np.array([
            [0.0, 0.0],
            [float(product_width), 0.0],
            [float(product_width), float(product_height)],
            [0.0, float(product_height)],
        ], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(source, polygon)
        mapped = cv2.perspectiveTransform(
            np.array([[[float(px), float(py)]]], dtype=np.float32),
            transform,
        )[0, 0]
        if not np.isfinite(mapped).all():
            return img_x, img_y

        polygon_x = int(round(float(mapped[0])))
        polygon_y = int(round(float(mapped[1])))
        polygon_distance = float(cv2.pointPolygonTest(
            polygon, (float(polygon_x), float(polygon_y)), True,
        ))
        if polygon_distance < -1.0:
            return img_x, img_y

        logger.warning(
            "AOI mapping corrected by panel polygon: product=(%d,%d) "
            "raw=(%d,%d) raw_dist=%.1fpx coverage=%.3f polygon=(%d,%d)",
            px, py, img_x, img_y, raw_distance, polygon_coverage,
            polygon_x, polygon_y,
        )
        return polygon_x, polygon_y
    except (cv2.error, TypeError, ValueError):
        return img_x, img_y


@dataclass
class PreprocessConfig:
    tile_size: int = 512
    tile_stride: int = 512
    otsu_offset: int = 5
    enable_panel_polygon: bool = True
    edge_threshold_px: int = 768  # retained for config compatibility; zone split uses anchor distance <= tile_size / 2
    coverage_min: float = 0.3
    # Edge 取樣 anchor 仍可往 panel 外推半個 tile，但實際 ROI 會再依 polygon 往產品
    # 內側推回來，避免 edge.pt 學到黑色背景邊界。
    # None → tile_size // 2；0 → 關閉額外 edge anchor。
    outer_edge_extend: Optional[int] = None
    image_preprocess_pipeline: List[Dict[str, Any]] = field(default_factory=list)
    cache_processed_image: bool = False
    generate_grid_tiles: bool = True
    preprocess_after_tiling: bool = False
    product_resolution: Optional[Tuple[int, int]] = None
    rotate_180: bool = False
    image_preprocess_pipelines: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self):
        if self.outer_edge_extend is None:
            self.outer_edge_extend = self.tile_size // 2


def image_preprocess_pipeline_for_zone(
    config: PreprocessConfig,
    zone: Optional[str],
) -> List[Dict[str, Any]]:
    """Return a zone-specific tile pipeline, with legacy shared fallback."""
    pipelines = getattr(config, "image_preprocess_pipelines", None) or {}
    if zone in ("inner", "edge") and zone in pipelines:
        return list(pipelines[zone] or [])
    return list(getattr(config, "image_preprocess_pipeline", None) or [])


@dataclass
class TileResult:
    tile_id: int
    x1: int; y1: int; x2: int; y2: int
    image: np.ndarray
    mask: Optional[np.ndarray]
    coverage: float
    zone: str  # "inner" | "edge" | "outside"
    center_dist_to_edge: float
    original_image: Optional[np.ndarray] = field(default=None, repr=False)
    preprocess_pipeline: List[Dict[str, Any]] = field(default_factory=list)
    # 4 個 outer-extension 角 + 4 個 inner-edge 角；訓練 wizard 的「corners-only」
    # panel 模式只收 is_corner=True 的 tile（只給 edge 模型補強用）。
    is_corner: bool = False
    preprocess_steps: List[Dict[str, Any]] = field(default_factory=list)
    preprocess_total_ms: float = 0.0


@dataclass
class PanelPreprocessResult:
    image_path: Path
    lighting: str
    foreground_bbox: Tuple[int, int, int, int]
    panel_polygon: Optional[np.ndarray]
    tiles: List[TileResult] = field(default_factory=list)
    polygon_detection_failed: bool = False
    preprocess_steps: List[Dict[str, Any]] = field(default_factory=list)
    preprocess_total_ms: float = 0.0
    processed_image: Optional[np.ndarray] = field(default=None, repr=False)


LIGHTING_PREFIXES = ("G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD")
BOUNDARY_REFERENCE_PRIORITY = ("W0F00000", "STANDARD", "G0F00000", "R0F00000", "WGF50500")
BOUNDARY_GRAY_BAND_SHIFT_PARAMS = {
    "low_threshold": 105,
    "high_threshold": 110,
    "dark_shift": 10,
    "bright_shift": 10,
    "band_mode": "keep",
}
SKIP_EXACT = ("Optics.log",)

EDGE_MARGIN = 20
SAMPLE_STEP = 50
OUTLIER_SIGMA = 3.0
MIN_EDGE_LEN_RATIO = 1.0
MIN_POLYGON_AREA_RATIO = 0.9
MIN_SAMPLES_PER_EDGE = 5
EDGE_ENDPOINT_TRIM_RATIO = 0.05
MAX_EDGE_RESIDUAL_P95_RATIO = 0.03
MIN_EDGE_RESIDUAL_P95_PX = 8.0
SMALL_PRODUCT_MAX_RESOLUTION = (1366, 768)
# Half-tile edge extension has IoU ~= 1/3 with the original edge row. If
# polygon inward-shift makes it overlap more than this, it is not a new sample.
EDGE_EXTENSION_DUPLICATE_IOU = 0.36
LOW_CONTRAST_SIDE_MAX_EXTEND_PX = 128
SIDE_EDGE_STABILIZE_SPAN_PX = 32


def is_small_product_resolution(product_resolution: Optional[Tuple[int, int]]) -> bool:
    """True for the small-panel family that needs robust boundary detection."""
    if not product_resolution:
        return False
    try:
        width, height = product_resolution
        return int(width) <= SMALL_PRODUCT_MAX_RESOLUTION[0] and int(height) <= SMALL_PRODUCT_MAX_RESOLUTION[1]
    except Exception:
        return False


def _use_robust_panel_boundary(config: PreprocessConfig) -> bool:
    return is_small_product_resolution(getattr(config, "product_resolution", None))


def _boundary_detection_gray(
    image: np.ndarray,
    config: PreprocessConfig,
) -> np.ndarray:
    """Boundary detection uses a gray-band-shift contrast split only for edge finding."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pipeline = getattr(config, "image_preprocess_pipeline", None) or []
    params = BOUNDARY_GRAY_BAND_SHIFT_PARAMS

    try:
        normalized = normalize_preprocess_pipeline(pipeline)
    except Exception:
        normalized = []

    for step in normalized:
        if step.get("method") != "gray_band_shift":
            continue
        params = step.get("params") or params
        break

    try:
        result = apply_preprocess_method(gray, "gray_band_shift", params)
        return result["image"]
    except Exception:
        return gray


def _find_low_contrast_upper_boundary(
    gray: np.ndarray,
    bbox: Tuple[int, int, int, int],
    tile_size: int,
) -> int:
    """Find a dim top band missed by Otsu but attached to the bright panel area."""
    x1, y1, x2, _y2 = bbox
    if y1 <= 0 or x2 - x1 < 32:
        return y1

    search_px = min(max(80, int(tile_size) // 2), y1)
    y0 = y1 - search_px
    width = x2 - x1
    pad = max(20, width // 20)
    xl = min(x2 - 1, x1 + pad)
    xr = max(xl + 1, x2 - pad)
    roi = gray[y0:y1, xl:xr]
    if roi.shape[0] < 16 or roi.shape[1] < 32:
        return y1

    profile = np.median(roi, axis=1).astype(np.float32)
    bg_count = max(5, min(30, len(profile) // 4))
    bg_rows = profile[:bg_count]
    bg = float(np.median(bg_rows))
    mad = float(np.median(np.abs(bg_rows - bg)))
    delta = max(3.0, min(16.0, 3.0 * mad + 2.0))

    above = (profile > bg + delta).astype(np.uint8)
    if not np.any(above):
        return y1
    kernel = np.ones((1, 5), np.uint8)
    closed = cv2.morphologyEx(above.reshape(1, -1), cv2.MORPH_CLOSE, kernel).ravel() > 0

    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, val in enumerate(closed):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(closed)))

    min_run = max(8, min(24, len(profile) // 12))
    max_gap = max(12, min(24, len(profile) // 10))
    for run_start, run_end in runs:
        if run_end - run_start < min_run:
            continue
        if len(profile) - run_end <= max_gap:
            return y0 + run_start
    return y1


def _find_low_contrast_side_boundary(
    gray: np.ndarray,
    bbox: Tuple[int, int, int, int],
    tile_size: int,
    side: str,
) -> int:
    """Find a dim left/right band missed by Otsu but attached to the panel."""
    x1, y1, x2, y2 = bbox
    img_w = gray.shape[1]
    if y2 - y1 < 32 or x2 <= x1:
        return x1 if side == "left" else x2

    if side == "left":
        search_px = min(max(80, int(tile_size) // 2), x1)
        if search_px <= 0:
            return x1
        x0, x_end = x1 - search_px, x1
    elif side == "right":
        search_px = min(max(80, int(tile_size) // 2), img_w - x2)
        if search_px <= 0:
            return x2
        x0, x_end = x2, x2 + search_px
    else:
        raise ValueError("side must be 'left' or 'right'")

    height = y2 - y1
    pad = max(20, height // 20)
    yt = min(y2 - 1, y1 + pad)
    yb = max(yt + 1, y2 - pad)
    roi = gray[yt:yb, x0:x_end]
    if roi.shape[0] < 32 or roi.shape[1] < 16:
        return x1 if side == "left" else x2

    profile = np.median(roi, axis=0).astype(np.float32)
    bg_count = max(5, min(30, len(profile) // 4))
    bg_cols = profile[:bg_count] if side == "left" else profile[-bg_count:]
    bg = float(np.median(bg_cols))
    mad = float(np.median(np.abs(bg_cols - bg)))
    delta = max(3.0, min(16.0, 3.0 * mad + 2.0))

    above = (profile > bg + delta).astype(np.uint8)
    if not np.any(above):
        return x1 if side == "left" else x2
    kernel = np.ones((1, 5), np.uint8)
    closed = cv2.morphologyEx(above.reshape(1, -1), cv2.MORPH_CLOSE, kernel).ravel() > 0

    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, val in enumerate(closed):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(closed)))

    min_run = max(8, min(24, len(profile) // 12))
    max_gap = max(12, min(24, len(profile) // 10))
    for run_start, run_end in runs:
        if run_end - run_start < min_run:
            continue
        if side == "left" and len(profile) - run_end <= max_gap:
            candidate = x0 + run_start
            if x1 - candidate > LOW_CONTRAST_SIDE_MAX_EXTEND_PX:
                return x1
            return candidate
        if side == "right" and run_start <= max_gap:
            candidate = x0 + run_end
            if candidate - x2 > LOW_CONTRAST_SIDE_MAX_EXTEND_PX:
                return x2
            return candidate
    return x1 if side == "left" else x2


def filter_panel_lighting_files(
    folder: Path,
    image_files: Optional[Iterable[Path]] = None,
    prefix_resolver: Optional[Callable[[str], str]] = None,
    allowed_prefixes: Optional[Iterable[str]] = None,
) -> Dict[str, Path]:
    """從 panel folder 過濾出 5 個有效 lighting 圖。

    只保留檔名以 5 個 lighting prefix 開頭的圖；其他（S* 側拍 / B0F 黑屏 /
    PINIGBI / OMIT / Optics.log）自然被忽略。

    Returns: {"G0F00000": Path, ...}，缺哪個 lighting 就少哪個 key。
    """
    def _is_newer(candidate: Path, current: Path) -> bool:
        try:
            cand_key = (candidate.stat().st_mtime_ns, candidate.name)
            curr_key = (current.stat().st_mtime_ns, current.name)
        except OSError:
            cand_key = (0, candidate.name)
            curr_key = (0, current.name)
        return cand_key > curr_key

    resolve_prefix = prefix_resolver or canonical_image_prefix
    allowed = set(allowed_prefixes or LIGHTING_PREFIXES)
    result: Dict[str, Path] = {}
    entries = image_files if image_files is not None else folder.iterdir()
    for entry in entries:
        entry = Path(entry)
        if not entry.is_file():
            continue
        name = entry.name
        if name in SKIP_EXACT:
            continue
        lighting = resolve_prefix(name)
        if lighting in allowed:
            if lighting not in result or _is_newer(entry, result[lighting]):
                result[lighting] = entry
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

    gray = _boundary_detection_gray(image, config)
    if _use_robust_panel_boundary(config):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((15, 15), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        min_area = 1000
        xs = []
        ys = []
        for c in contours:
            if cv2.contourArea(c) <= min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            xs.extend((x, x + w))
            ys.extend((y, y + h))
        if not xs or not ys:
            return None, None

        img_h, img_w = binary.shape[:2]
        raw_x1 = max(0, min(xs))
        raw_y1 = max(0, min(ys))
        raw_x2 = min(img_w, max(xs))
        raw_y2 = min(img_h, max(ys))
        top_y = _find_low_contrast_upper_boundary(
            gray,
            (raw_x1, raw_y1, raw_x2, raw_y2),
            config.tile_size,
        )
        left_x = _find_low_contrast_side_boundary(
            gray,
            (raw_x1, raw_y1, raw_x2, raw_y2),
            config.tile_size,
            "left",
        )
        right_x = _find_low_contrast_side_boundary(
            gray,
            (raw_x1, raw_y1, raw_x2, raw_y2),
            config.tile_size,
            "right",
        )
        if top_y < raw_y1:
            binary[top_y:raw_y1, left_x:right_x] = 255
        if left_x < raw_x1:
            binary[top_y:raw_y2, left_x:raw_x1] = 255
        if right_x > raw_x2:
            binary[top_y:raw_y2, raw_x2:right_x] = 255
        raw_x1 = left_x
        raw_y1 = top_y
        raw_x2 = right_x

        offset = max(0, int(config.otsu_offset))
        x1 = max(0, raw_x1 + offset)
        y1 = max(0, raw_y1 + offset)
        x2 = min(img_w, raw_x2 - offset)
        y2 = min(img_h, raw_y2 - offset)
        if x1 >= x2 or y1 >= y2:
            x1, y1, x2, y2 = 0, 0, img_w, img_h
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        offset = config.otsu_offset
        x1, y1, x2, y2 = x + offset, y + offset, x + w - offset, y + h - offset
    bbox = (x1, y1, x2, y2)

    if not config.enable_panel_polygon:
        return bbox, None

    polygon = _polyfit_polygon(
        binary,
        bbox,
        config.tile_size,
        stabilize_near_vertical_edges=_use_robust_panel_boundary(config),
    )
    return bbox, polygon


def _polyfit_polygon(
    binary_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    tile_size: int,
    stabilize_near_vertical_edges: bool = False,
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

    def _trim_edge_endpoints(pts):
        # bbox 角落會混入相鄰斜邊；只用每邊中央 90% 評估直線品質。
        trim = int(round(len(pts) * EDGE_ENDPOINT_TRIM_RATIO))
        if trim <= 0 or len(pts) - 2 * trim < MIN_SAMPLES_PER_EDGE:
            return pts
        return pts[trim:-trim]

    def _trim_side_points(pts):
        if not stabilize_near_vertical_edges:
            return pts
        trim = min(max(64, int(tile_size) // 4), max(0, (ymax - ymin) // 4))
        trimmed = [p for p in pts if ymin + trim <= p[1] <= ymax - trim]
        return trimmed if len(trimmed) >= MIN_SAMPLES_PER_EDGE else pts

    def fit(pts, horizontal):
        arr = np.array(pts, dtype=float)
        ind, dep = (arr[:, 0], arr[:, 1]) if horizontal else (arr[:, 1], arr[:, 0])
        if stabilize_near_vertical_edges and not horizontal:
            q10, q90 = np.percentile(dep, [10, 90])
            if q90 - q10 <= SIDE_EDGE_STABILIZE_SPAN_PX:
                return 0.0, float(np.median(dep))
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
        residuals = np.abs(dep - (a * ind + b))
        residual_p95 = float(np.percentile(residuals, 95))
        residual_limit = max(
            MIN_EDGE_RESIDUAL_P95_PX,
            float(tile_size) * MAX_EDGE_RESIDUAL_P95_RATIO,
        )
        if residual_p95 > residual_limit:
            logger.warning(
                "[boundary] reject non-linear %s edge: residual_p95=%.1fpx > %.1fpx (%d samples)",
                "horizontal" if horizontal else "vertical",
                residual_p95,
                residual_limit,
                len(arr),
            )
            return None
        return float(a), float(b)

    top_points = _trim_edge_endpoints(tops)
    bottom_points = _trim_edge_endpoints(bots)
    side_lefts = _trim_side_points(_trim_edge_endpoints(lefts))
    side_rights = _trim_side_points(_trim_edge_endpoints(rights))
    top_l, bot_l = fit(top_points, True), fit(bottom_points, True)
    left_l, right_l = fit(side_lefts, False), fit(side_rights, False)
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


def classify_anchor_zone(
    anchor_xy: Tuple[float, float],
    polygon: Optional[np.ndarray],
    edge_distance_px: float,
) -> Tuple[str, float]:
    """依 anchor 到 panel polygon 的 signed distance 分 INNER / EDGE。"""
    if polygon is None:
        return "inner", float("inf")

    signed_distance = float(cv2.pointPolygonTest(
        np.asarray(polygon, dtype=np.float32),
        (float(anchor_xy[0]), float(anchor_xy[1])),
        True,
    ))
    zone = "edge" if signed_distance <= float(edge_distance_px) else "inner"
    return zone, signed_distance


def classify_tile_zone(
    tile_rect: Tuple[int, int, int, int],
    polygon: Optional[np.ndarray],
    config: PreprocessConfig,
) -> Tuple[str, float, float, Optional[np.ndarray]]:
    """以 tile 中心到 polygon 的距離決定 zone，並計算 coverage / mask。

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

    # 訓練圖沒有缺陷點，以 tile 中心作為與推論缺陷中心等價的 anchor。
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    # bbox grid 會受 otsu_offset 向產品內縮；補回這段偵測偏移後，實際分界仍是
    # anchor 距真實 panel 邊界半個 tile。AOI 缺陷中心不經 bbox grid，無此補償。
    grid_edge_distance = config.tile_size / 2.0 + max(
        0.0, float(getattr(config, "otsu_offset", 0)),
    )
    zone, signed_center_dist = classify_anchor_zone(
        (cx, cy), polygon, grid_edge_distance,
    )
    center_dist = abs(signed_center_dist)
    output_mask = mask if coverage < 1.0 - 1e-6 else None
    return zone, coverage, center_dist, output_mask


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
    shift_axes: str = "xy",
) -> Tuple[int, int, float, bool]:
    """Resolve a tile origin that stays inside the product polygon when possible.

    ``anchor_xy`` is the AOI point or the synthetic edge-sampling center. The
    returned tile is first clamped to the image, then iteratively pushed toward
    the polygon centroid until its corners are inside the polygon. If the panel
    geometry cannot fit a full tile, the best-coverage origin found so far is
    returned instead of padding or masking black background into the sample.
    When ``keep_anchor_inside`` is True, movement is constrained so the AOI
    anchor remains inside the tile whenever image bounds make that possible.

    ``shift_axes`` can be "xy", "x", or "y". AOI single-edge samples use this
    to avoid correcting the unrelated axis when only one panel edge is close.

    Returns: ``(tx, ty, coverage, shifted)``.
    """
    allowed_axes = set(shift_axes or "xy")
    if not allowed_axes or allowed_axes - {"x", "y"}:
        raise ValueError("shift_axes must be one of 'xy', 'x', or 'y'")
    allow_x = "x" in allowed_axes
    allow_y = "y" in allowed_axes

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
        if not allow_x:
            dx = 0
        if not allow_y:
            dy = 0
        if dx == 0 and dy == 0:
            break

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
    img = read_detection_image(image_path, cv2.IMREAD_GRAYSCALE, config.rotate_180)
    if img is None:
        raise FileNotFoundError(f"無法讀取圖片: {image_path}")
    original_img = img.copy()
    normalized_pipeline: List[Dict[str, Any]] = []
    preprocess_steps: List[Dict[str, Any]] = []
    preprocess_total_ms = 0.0
    if config.image_preprocess_pipeline and not getattr(config, "preprocess_after_tiling", False):
        from capi_image_preprocess_lab import apply_preprocess_pipeline, describe_preprocess_pipeline
        logger.info("[preprocess] pipeline: %s", describe_preprocess_pipeline(config.image_preprocess_pipeline))
        pipeline_result = apply_preprocess_pipeline(img, config.image_preprocess_pipeline)
        img = pipeline_result["image"]
        normalized_pipeline = pipeline_result["pipeline"]
        preprocess_steps = pipeline_result["steps"]
        preprocess_total_ms = float(pipeline_result.get("total_elapsed_ms") or 0.0)
        for step in pipeline_result["steps"]:
            logger.info(
                "[preprocess] step %d %s params=%s elapsed=%.3fms stats=%s",
                step["index"], step["method_label"], step["applied_params"],
                float(step.get("elapsed_ms") or 0.0), step["stats"],
            )

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
            preprocess_steps=preprocess_steps,
            preprocess_total_ms=preprocess_total_ms,
            processed_image=img if config.cache_processed_image else None,
        )

    tiles = []
    if config.generate_grid_tiles:
        tiles = _generate_tiles(
            img,
            bbox,
            polygon,
            config,
            original_img=original_img,
            preprocess_pipeline=normalized_pipeline,
        )
        if getattr(config, "preprocess_after_tiling", False):
            preprocess_steps = [
                step
                for tile in tiles
                for step in tile.preprocess_steps
            ]
            preprocess_total_ms = sum(tile.preprocess_total_ms for tile in tiles)
    return PanelPreprocessResult(
        image_path=image_path,
        lighting=lighting,
        foreground_bbox=bbox,
        panel_polygon=polygon,
        tiles=tiles,
        polygon_detection_failed=polygon_failed,
        preprocess_steps=preprocess_steps,
        preprocess_total_ms=preprocess_total_ms,
        processed_image=img if config.cache_processed_image else None,
    )


def preprocess_panel_folder(
    folder: Path,
    config: PreprocessConfig,
    image_files: Optional[Iterable[Path]] = None,
    boundary_reference_files: Optional[Iterable[Path]] = None,
    prefix_resolver: Optional[Callable[[str], str]] = None,
    allowed_prefixes: Optional[Iterable[str]] = None,
    boundary_reference_priority: Optional[Iterable[str]] = None,
) -> Dict[str, "PanelPreprocessResult"]:
    """處理整個 panel folder 的 5 lighting 圖。

    流程：filter 出目標 lighting → 依 W0F/STD/G0F/R0F/WGF 優先序選
          reference polygon → 所有目標 lighting 套同一個 reference。
          boundary_reference_files 可提供只抓邊、不加入回傳結果的候選圖。
    """
    files = filter_panel_lighting_files(
        folder,
        image_files=image_files,
        prefix_resolver=prefix_resolver,
        allowed_prefixes=allowed_prefixes,
    )
    if not files:
        return {}

    reference_files = (
        filter_panel_lighting_files(
            folder,
            image_files=boundary_reference_files,
            prefix_resolver=prefix_resolver,
            allowed_prefixes=allowed_prefixes,
        )
        if boundary_reference_files is not None
        else files
    )
    if not reference_files:
        reference_files = files

    # 決定 reference image：W0F00000 對低對比前後景較穩，找邊優先用它。
    ref_lighting = None
    for cand in tuple(boundary_reference_priority or BOUNDARY_REFERENCE_PRIORITY):
        if cand in reference_files:
            ref_lighting = cand
            break
    if ref_lighting is None:
        return {}

    ref_result = preprocess_panel_image(reference_files[ref_lighting], ref_lighting, config)
    if ref_result.polygon_detection_failed:
        for cand in tuple(boundary_reference_priority or BOUNDARY_REFERENCE_PRIORITY):
            if cand == ref_lighting or cand not in reference_files:
                continue
            fallback_result = preprocess_panel_image(reference_files[cand], cand, config)
            if not fallback_result.polygon_detection_failed:
                ref_lighting = cand
                ref_result = fallback_result
                break

    logger.info(
        "[boundary] reference=%s file=%s polygon=%s",
        ref_lighting,
        ref_result.image_path.name,
        None if ref_result.panel_polygon is None else np.round(ref_result.panel_polygon, 1).tolist(),
    )

    results: Dict[str, PanelPreprocessResult] = {}
    if ref_lighting in files:
        results[ref_lighting] = ref_result
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
    original_img: Optional[np.ndarray] = None,
    preprocess_pipeline: Optional[List[Dict[str, Any]]] = None,
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
        span = hi - lo - ts
        if span < 0:
            return []
        if span == 0:
            return [lo]
        stride = max(1, int(config.tile_stride))
        count = int(np.ceil(span / stride)) + 1
        return [
            int(round(lo + span * i / (count - 1)))
            for i in range(count)
        ]

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
    emitted_edge_rects: List[Tuple[int, int, int, int]] = []
    tid = 0

    inner_corner_xs = {xs[0], xs[-1]} if xs else set()
    inner_corner_ys = {ys[0], ys[-1]} if ys else set()
    outer_corner_xs = {x for x in (left_tx, right_tx) if x is not None}
    outer_corner_ys = {y for y in (top_ty, bottom_ty) if y is not None}

    def _is_corner(tx: int, ty: int) -> bool:
        return ((tx in inner_corner_xs and ty in inner_corner_ys)
                or (tx in outer_corner_xs and ty in outer_corner_ys))

    def _resolve_edge_origin(tx: int, ty: int) -> Tuple[int, int, bool]:
        if polygon is None:
            return tx, ty, False
        anchor = (tx + half, ty + half)
        rx, ry, _cov, shifted = resolve_inward_polygon_tile(
            anchor_xy=anchor,
            polygon=polygon,
            image_shape=(img_h, img_w),
            tile_size=ts,
            initial_origin=(tx, ty),
        )
        return rx, ry, shifted

    def _emit(tx: int, ty: int, zone: str, cov: float, dist: float, mask,
              is_corner_override: Optional[bool] = None,
              dedupe_edge_overlap: bool = False) -> None:
        nonlocal tid
        if (tx, ty) in emitted_positions:
            return
        rect = (tx, ty, tx + ts, ty + ts)
        if zone == "edge" and dedupe_edge_overlap:
            if any(_rect_iou(rect, prev) >= EDGE_EXTENSION_DUPLICATE_IOU
                   for prev in emitted_edge_rects):
                return
        emitted_positions.add((tx, ty))
        tile_img = img[ty:ty + ts, tx:tx + ts].copy()
        tile_pipeline = image_preprocess_pipeline_for_zone(config, zone)
        tile_preprocess_steps: List[Dict[str, Any]] = []
        tile_preprocess_total_ms = 0.0
        if getattr(config, "preprocess_after_tiling", False) and tile_pipeline:
            from capi_image_preprocess_lab import apply_preprocess_pipeline
            pipeline_result = apply_preprocess_pipeline(tile_img, tile_pipeline)
            tile_img = pipeline_result["image"]
            tile_preprocess_steps = pipeline_result["steps"]
            tile_preprocess_total_ms = float(
                pipeline_result.get("total_elapsed_ms") or 0.0
            )

        tiles.append(TileResult(
            tile_id=tid,
            x1=tx, y1=ty, x2=tx + ts, y2=ty + ts,
            image=tile_img,
            mask=mask,
            coverage=cov,
            zone=zone,
            center_dist_to_edge=dist,
            original_image=(
                original_img[ty:ty + ts, tx:tx + ts].copy()
                if original_img is not None else None
            ),
            preprocess_pipeline=(
                list(tile_pipeline)
                if getattr(config, "preprocess_after_tiling", False)
                else list(preprocess_pipeline or [])
            ),
            is_corner=_is_corner(tx, ty) if is_corner_override is None else is_corner_override,
            preprocess_steps=tile_preprocess_steps,
            preprocess_total_ms=tile_preprocess_total_ms,
        ))
        if zone == "edge":
            emitted_edge_rects.append(rect)
        tid += 1

    def _rect_iou(
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
    ) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        denom = area_a + area_b - inter
        return float(inter) / float(denom) if denom > 0 else 0.0

    for ty0 in ys:
        for tx0 in xs:
            zone, cov, dist, mask = classify_tile_zone((tx0, ty0, tx0 + ts, ty0 + ts), polygon, config)
            if zone == "outside":
                continue
            anchor_zone = zone
            # Polygon 偵測失敗時以 bbox 首末排作為中心恰在半個 tile 的 fallback。
            if polygon is None and (
                tx0 == xs[0] or tx0 == xs[-1] or ty0 == ys[0] or ty0 == ys[-1]
            ):
                anchor_zone = "edge"
            anchor_dist = dist
            tx, ty = tx0, ty0
            is_corner = _is_corner(tx0, ty0)
            needs_inward_shift = anchor_zone == "edge" or (
                polygon is not None and cov < 1.0 - 1e-6
            )
            if needs_inward_shift:
                tx, ty, _shifted = _resolve_edge_origin(tx0, ty0)
                _shifted_zone, cov, _shifted_dist, mask = classify_tile_zone(
                    (tx, ty, tx + ts, ty + ts), polygon, config,
                )
                if cov >= 1.0 - 1e-6:
                    mask = None
            _emit(
                tx, ty, anchor_zone, cov, anchor_dist, mask,
                is_corner_override=is_corner,
            )

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
        tx, ty, shifted = _resolve_edge_origin(tx0, ty0)
        _zone, cov, dist, mask = classify_tile_zone((tx, ty, tx + ts, ty + ts), polygon, config)
        if cov >= 1.0 - 1e-6:
            mask = None
        _emit(
            tx, ty, "edge", cov, dist, mask,
            is_corner_override=is_corner,
            dedupe_edge_overlap=shifted,
        )

    return tiles
