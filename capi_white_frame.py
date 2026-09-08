"""Rotation-tolerant WHITEFRA bright-frame inspection.

The detector is intentionally independent from PatchCore.  It aligns the
visible long bright segments to a canonical rectangle, then checks each side
as a one-dimensional continuity signal.  An NG result participates in the
formal panel judgment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from capi_image_naming import is_white_frame_image_name
from capi_image_orientation import read_detection_image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SIDE_ORDER = ("top", "right", "bottom", "left")
SIDE_LABELS = {
    "top": "上邊",
    "right": "右邊",
    "bottom": "下邊",
    "left": "左邊",
}


@dataclass(frozen=True)
class WhiteFrameConfig:
    max_locator_dimension: int = 1800
    brightness_floor: int = 32
    brightness_ceiling: int = 160
    brightness_percentile: float = 99.5
    min_component_span_ratio: float = 0.12
    min_component_area: int = 80
    max_component_fill_ratio: float = 0.20
    min_frame_width_ratio: float = 0.30
    min_frame_height_ratio: float = 0.30
    min_frame_area_ratio: float = 0.12
    max_frame_area_ratio: float = 0.90
    edge_band_ratio: float = 0.008
    min_edge_band_px: int = 16
    max_edge_band_px: int = 48
    corner_margin_ratio: float = 0.01
    min_corner_margin_px: int = 50
    min_break_gap_px: int = 15
    close_gap_px: int = 9
    open_signal_px: int = 3


@dataclass
class WhiteFrameInspection:
    image_path: Path
    image_size: Tuple[int, int]
    bounds: Tuple[int, int, int, int]
    payload: Dict[str, Any]


def find_white_frame_image(panel_dir: Path) -> Optional[Path]:
    """Return the newest WHITEFRA image, or ``None`` when not supplied."""
    candidates = [
        path
        for path in Path(panel_dir).iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and is_white_frame_image_name(path.name)
    ]
    if not candidates:
        return None

    def latest_key(path: Path) -> Tuple[int, str]:
        try:
            modified = path.stat().st_mtime_ns
        except OSError:
            modified = 0
        return modified, path.name

    return max(candidates, key=latest_key)


def inspect_white_frame_panel(
    panel_dir: Path,
    *,
    rotate_180: bool = False,
    config: WhiteFrameConfig = WhiteFrameConfig(),
) -> Optional[WhiteFrameInspection]:
    image_path = find_white_frame_image(panel_dir)
    if image_path is None:
        return None
    return inspect_white_frame_image(
        image_path,
        rotate_180=rotate_180,
        config=config,
    )


def inspect_white_frame_image(
    image_path: Path,
    *,
    rotate_180: bool = False,
    config: WhiteFrameConfig = WhiteFrameConfig(),
    aoi_points: Optional[list] = None,
    product_resolution: Optional[Tuple[int, int]] = None,
    tile_size: int = 512,
) -> WhiteFrameInspection:
    started = perf_counter()
    path = Path(image_path)
    try:
        image = read_detection_image(path, cv2.IMREAD_UNCHANGED, rotate_180)
    except Exception as exc:
        return _unreadable(path, (0, 0), f"image_read_failed:{type(exc).__name__}", started)
    if image is None or image.size == 0:
        return _unreadable(path, (0, 0), "image_read_failed", started)

    try:
        gray = _to_gray_u8(image)
    except (ValueError, TypeError, cv2.error) as exc:
        return _unreadable(path, (0, 0), f"image_format_invalid:{type(exc).__name__}", started)
    height, width = gray.shape[:2]
    try:
        quad, threshold = _locate_frame(gray, config)
        side_results, output_size = _inspect_sides(gray, quad, threshold, config)
    except (ValueError, cv2.error) as exc:
        return _unreadable(
            path,
            (width, height),
            str(exc) or type(exc).__name__,
            started,
        )

    ng_sides = [side for side in SIDE_ORDER if side_results[side]["status"] == "NG"]
    angle = float(np.degrees(np.arctan2(
        float(quad[1][1] - quad[0][1]),
        float(quad[1][0] - quad[0][0]),
    )))
    x, y, box_w, box_h = cv2.boundingRect(np.rint(quad).astype(np.int32))
    bounds = (
        max(0, x),
        max(0, y),
        min(width, x + box_w),
        min(height, y + box_h),
    )
    payload = {
        "algorithm": "white-frame-cv-v1",
        "affects_judgment": True,
        "status": "NG" if ng_sides else "OK",
        "image_name": path.name,
        "angle_deg": round(angle, 3),
        "brightness_threshold": int(threshold),
        "aligned_width": int(output_size[0]),
        "aligned_height": int(output_size[1]),
        "ng_sides": ng_sides,
        "sides": side_results,
        "processing_ms": round((perf_counter() - started) * 1000.0, 1),
    }
    if aoi_points:
        payload["full_frame_status"] = payload["status"]
        payload["aoi_algorithm"] = "white-frame-aoi-cv-v1"
        payload["aoi_tiles"] = _inspect_aoi_tiles(
            gray, quad, bounds, aoi_points, product_resolution, tile_size, config,
        )
        if any(tile["status"] == "NG" for tile in payload["aoi_tiles"]):
            payload["status"] = "NG"
        payload["processing_ms"] = round((perf_counter() - started) * 1000.0, 1)
    return WhiteFrameInspection(path, (width, height), bounds, payload)


def _inspect_aoi_tiles(gray, quad, bounds, points, product_resolution, tile_size, config):
    """Inspect native-resolution line strength without closing small notches."""
    from capi_preprocess import map_product_coord_to_image

    height, width = gray.shape
    half = tile_size // 2
    results = []
    for index, point in enumerate(points):
        image_x, image_y = map_product_coord_to_image(
            point["product_x"], point["product_y"], bounds, product_resolution, quad,
        )
        tx = max(0, min(image_x - half, width - tile_size))
        ty = max(0, min(image_y - half, height - tile_size))
        crop_w, crop_h = min(tile_size, width - tx), min(tile_size, height - ty)
        item = {
            "tile_id": index, "aoi_defect_code": point["defect_code"],
            "aoi_product_x": point["product_x"], "aoi_product_y": point["product_y"],
            "aoi_image_x": image_x, "aoi_image_y": image_y,
            "x": tx, "y": ty, "width": crop_w, "height": crop_h,
            "aoi_tile_shift_dx": tx - (image_x - half),
            "aoi_tile_shift_dy": ty - (image_y - half),
            "status": "UNREADABLE", "gaps": [],
        }
        results.append(item)
        anchor = np.array([image_x, image_y], dtype=np.float32)
        candidates = []
        for side_index, side in enumerate(SIDE_ORDER):
            start = quad[side_index]
            vector = quad[(side_index + 1) % 4] - start
            length = float(np.linalg.norm(vector))
            direction = vector / max(length, 1.0)
            along = float(np.clip(np.dot(anchor - start, direction), 0, length))
            distance = float(np.linalg.norm(anchor - start - along * direction))
            candidates.append((distance, side, start, direction, length))
        _, side, start, direction, length = min(candidates, key=lambda entry: entry[0])
        item["side"] = side
        item["side_label"] = SIDE_LABELS[side]
        # Only sample the portion of this frame side inside the actual TILE.
        positions = np.arange(int(length) + 1, dtype=np.float32)
        centers = start + positions[:, None] * direction
        inside = (
            (centers[:, 0] >= tx + 2) & (centers[:, 0] < tx + crop_w - 2)
            & (centers[:, 1] >= ty + 2) & (centers[:, 1] < ty + crop_h - 2)
            & (positions >= 2) & (positions <= length - 2)
        )
        centers = centers[inside]
        if len(centers) < 16:
            item["reason"] = "TILE 內沒有足夠的白框線段"
            continue
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        offsets = np.arange(-config.max_edge_band_px, config.max_edge_band_px + 1, dtype=np.float32)
        samples = centers[None, :, :] + offsets[:, None, None] * normal
        strip = cv2.remap(
            gray, samples[:, :, 0], samples[:, :, 1], cv2.INTER_LINEAR,
        ).astype(np.float32)
        background = np.median(strip, axis=0)
        contrast = np.maximum(strip - background, 0)
        peak = float(np.median(contrast.max(axis=0)))
        if peak < config.brightness_floor:
            item["reason"] = "白框線亮度不足，無法建立局部基準"
            continue
        # Integrating the bright cross-section catches a narrowed line even
        # when its brightest pixel remains white. No gap-closing morphology.
        strength = np.maximum(contrast - peak * 0.25, 0).sum(axis=0)
        baseline = float(np.median(strength))
        ratios = strength / max(baseline, 1.0)
        item["min_line_ratio"] = round(float(ratios.min()), 3)
        for begin, end in _missing_runs(ratios >= 0.60, 2, len(centers) - 2):
            if end - begin < 2:
                continue
            center = centers[(begin + end - 1) // 2]
            item["gaps"].append({
                "center_x": int(round(float(center[0]))),
                "center_y": int(round(float(center[1]))),
                "length_px": end - begin,
                "line_ratio": round(float(ratios[begin:end].min()), 3),
                "source": "aoi_tile",
            })
        item["status"] = "NG" if item["gaps"] else "OK"
    return results


def render_white_frame_aoi_tile(image: np.ndarray, tile: dict) -> np.ndarray:
    """Render stored CV findings on the original TILE, without re-inspection."""
    x, y, width, height = (int(tile[key]) for key in ("x", "y", "width", "height"))
    crop = _to_gray_u8(image[y:y + height, x:x + width])
    canvas = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    for gap in tile.get("gaps", []):
        cx, cy = int(gap["center_x"]) - x, int(gap["center_y"]) - y
        radius = max(10, min(40, int(gap["length_px"]) // 2 + 6))
        cv2.rectangle(canvas, (cx - radius, cy - radius), (cx + radius, cy + radius), (0, 0, 255), 1)
    return canvas


def _unreadable(
    path: Path,
    image_size: Tuple[int, int],
    reason: str,
    started: float,
) -> WhiteFrameInspection:
    sides = {
        side: {"label": SIDE_LABELS[side], "status": "UNKNOWN", "gap_count": 0}
        for side in SIDE_ORDER
    }
    payload = {
        "algorithm": "white-frame-cv-v1",
        "affects_judgment": True,
        "status": "UNREADABLE",
        "reason": reason,
        "image_name": path.name,
        "ng_sides": [],
        "sides": sides,
        "processing_ms": round((perf_counter() - started) * 1000.0, 1),
    }
    return WhiteFrameInspection(
        path,
        image_size,
        (0, 0, image_size[0], image_size[1]),
        payload,
    )


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype == np.uint8:
        return gray
    max_value = float(np.iinfo(gray.dtype).max) if np.issubdtype(gray.dtype, np.integer) else float(np.max(gray))
    if max_value <= 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    return np.clip(gray.astype(np.float32) * (255.0 / max_value), 0, 255).astype(np.uint8)


def _threshold_value(gray: np.ndarray, config: WhiteFrameConfig) -> int:
    percentile = float(np.percentile(gray, config.brightness_percentile))
    return int(round(np.clip(
        max(float(config.brightness_floor), percentile),
        config.brightness_floor,
        config.brightness_ceiling,
    )))


def _locate_frame(
    gray: np.ndarray,
    config: WhiteFrameConfig,
) -> Tuple[np.ndarray, int]:
    height, width = gray.shape[:2]
    scale = min(1.0, float(config.max_locator_dimension) / max(height, width))
    if scale < 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray

    threshold = _threshold_value(small, config)
    mask = (small >= threshold).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    small_h, small_w = small.shape[:2]
    keep_ids = []
    for component_id in range(1, count):
        _x, _y, comp_w, comp_h, area = (int(value) for value in stats[component_id])
        spans_frame = (
            comp_w >= small_w * config.min_component_span_ratio
            or comp_h >= small_h * config.min_component_span_ratio
        )
        fill_ratio = float(area) / max(1, comp_w * comp_h)
        short_span = max(1, min(comp_w, comp_h))
        is_thin_segment = (
            short_span <= max(8, int(round(min(small_w, small_h) * 0.03)))
            or max(comp_w, comp_h) / short_span >= 4.0
        )
        if (
            spans_frame
            and area >= config.min_component_area
            and (is_thin_segment or fill_ratio <= config.max_component_fill_ratio)
        ):
            keep_ids.append(component_id)
    if not keep_ids:
        raise ValueError("frame_segments_not_found")

    ys, xs = np.where(np.isin(labels, keep_ids))
    if len(xs) < 4:
        raise ValueError("insufficient_frame_points")
    rect = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))
    quad = _order_quad(cv2.boxPoints(rect))
    quad[:, 0] *= float(width) / small_w
    quad[:, 1] *= float(height) / small_h

    top, right, bottom, left = _quad_lengths(quad)
    frame_w = max(top, bottom)
    frame_h = max(left, right)
    frame_area = abs(float(cv2.contourArea(quad)))
    image_area = float(width * height)
    if frame_w < width * config.min_frame_width_ratio:
        raise ValueError("frame_width_too_small")
    if frame_h < height * config.min_frame_height_ratio:
        raise ValueError("frame_height_too_small")
    if not (image_area * config.min_frame_area_ratio <= frame_area <= image_area * config.max_frame_area_ratio):
        raise ValueError("frame_area_out_of_range")
    return quad, _threshold_value(gray, config)


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    return np.array([
        points[int(np.argmin(sums))],
        points[int(np.argmin(diffs))],
        points[int(np.argmax(sums))],
        points[int(np.argmax(diffs))],
    ], dtype=np.float32)


def _quad_lengths(quad: np.ndarray) -> Tuple[float, float, float, float]:
    tl, tr, br, bl = quad
    return (
        float(np.linalg.norm(tr - tl)),
        float(np.linalg.norm(br - tr)),
        float(np.linalg.norm(br - bl)),
        float(np.linalg.norm(bl - tl)),
    )


def _inspect_sides(
    gray: np.ndarray,
    quad: np.ndarray,
    threshold: int,
    config: WhiteFrameConfig,
) -> Tuple[Dict[str, Dict[str, Any]], Tuple[int, int]]:
    top, right, bottom, left = _quad_lengths(quad)
    output_w = max(2, int(round(max(top, bottom))))
    output_h = max(2, int(round(max(left, right))))
    target = np.array([
        [0, 0],
        [output_w - 1, 0],
        [output_w - 1, output_h - 1],
        [0, output_h - 1],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(quad, target)
    inverse = cv2.getPerspectiveTransform(target, quad)
    aligned = cv2.warpPerspective(gray, transform, (output_w, output_h))
    band = int(round(min(output_w, output_h) * config.edge_band_ratio))
    band = max(config.min_edge_band_px, min(config.max_edge_band_px, band))
    band = min(band, max(1, min(output_w, output_h) // 4))

    profiles = {
        "top": aligned[:band, :].max(axis=0),
        "right": aligned[:, -band:].max(axis=1),
        "bottom": aligned[-band:, :].max(axis=0),
        "left": aligned[:, :band].max(axis=1),
    }
    results: Dict[str, Dict[str, Any]] = {}
    for side in SIDE_ORDER:
        present = _clean_presence(profiles[side] >= threshold, config)
        length = int(present.size)
        margin = max(
            config.min_corner_margin_px,
            int(round(length * config.corner_margin_ratio)),
        )
        margin = min(margin, max(0, (length - config.min_break_gap_px) // 2))
        gaps = []
        for start, end in _missing_runs(present, margin, length - margin):
            gap_length = end - start
            if gap_length < config.min_break_gap_px:
                continue
            center_x, center_y = _gap_center(side, (start + end - 1) / 2.0, output_w, output_h, inverse)
            gaps.append({
                "start_px": int(start),
                "end_px": int(end),
                "length_px": int(gap_length),
                "start_ratio": round(start / max(1, length - 1), 4),
                "end_ratio": round((end - 1) / max(1, length - 1), 4),
                "center_x": center_x,
                "center_y": center_y,
            })
        valid = present[margin:length - margin] if length - margin > margin else present
        results[side] = {
            "label": SIDE_LABELS[side],
            "status": "NG" if gaps else "OK",
            "gap_count": len(gaps),
            "largest_gap_px": max((gap["length_px"] for gap in gaps), default=0),
            "coverage_ratio": round(float(valid.mean()) if valid.size else 0.0, 4),
            "corner_margin_px": int(margin),
            "gaps": gaps,
        }
    return results, (output_w, output_h)


def _clean_presence(present: np.ndarray, config: WhiteFrameConfig) -> np.ndarray:
    signal = present.astype(np.uint8).reshape(1, -1)
    if config.open_signal_px > 1:
        kernel = np.ones((1, config.open_signal_px), dtype=np.uint8)
        signal = cv2.morphologyEx(signal, cv2.MORPH_OPEN, kernel)
    if config.close_gap_px > 1:
        kernel = np.ones((1, config.close_gap_px), dtype=np.uint8)
        signal = cv2.morphologyEx(signal, cv2.MORPH_CLOSE, kernel)
    return signal.reshape(-1) > 0


def _missing_runs(present: np.ndarray, start: int, end: int):
    missing = ~present[start:end]
    padded = np.concatenate(([False], missing, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return [(start + int(run_start), start + int(run_end)) for run_start, run_end in zip(starts, ends)]


def _gap_center(
    side: str,
    position: float,
    output_w: int,
    output_h: int,
    inverse: np.ndarray,
) -> Tuple[int, int]:
    if side == "top":
        point = (position, 0.0)
    elif side == "right":
        point = (float(output_w - 1), position)
    elif side == "bottom":
        point = (position, float(output_h - 1))
    else:
        point = (0.0, position)
    mapped = cv2.perspectiveTransform(
        np.array([[[point[0], point[1]]]], dtype=np.float32),
        inverse,
    )[0, 0]
    return int(round(float(mapped[0]))), int(round(float(mapped[1])))
