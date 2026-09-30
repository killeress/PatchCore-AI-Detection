"""Conservative raw-image refinement of an existing panel quadrilateral.

Otsu can cut into dim panel edges. Search only near the existing four sides,
require unambiguous bright-to-dark transitions and distributed line support,
and retain the entire original polygon if any side or geometry check fails.
This changes geometry only; detection/model pixels are never modified.
"""
import logging
from typing import Optional

import cv2
import numpy as np


logger = logging.getLogger("capi.preprocess")
SEARCH_RADIUS_PX = 48
MAX_CORNER_SHIFT_PX = 24.0
MIN_CORRECTION_PX = 1.0
SAMPLE_STEP_PX = 25
ENDPOINT_TRIM_RATIO = 0.15
PROFILE_HALF_WIDTH_PX = 4
MIN_GRADIENT = 2.0
MIN_CONTRAST = 12.0
MAX_COMPETING_GRADIENT_RATIO = 0.65
MIN_SAMPLES = 8
MIN_SUPPORT_RATIO = 0.70
MAX_RESIDUAL_P95_PX = 1.5
MAX_AREA_CHANGE_RATIO = 0.03
_SIDES = (("top", 0, 1, True, 1), ("right", 1, 2, False, -1),
          ("bottom", 3, 2, True, -1), ("left", 0, 3, False, 1))


def _sample_edge(gray, polygon, side):
    _, first, last, horizontal, polarity = side
    independent, dependent = (0, 1) if horizontal else (1, 0)
    p1, p2 = polygon[[first, last]]
    start, stop = float(p1[independent]), float(p2[independent])
    span = stop - start
    if span <= 0:
        return None, "invalid_side_order"
    seed_slope = float((p2[dependent] - p1[dependent]) / span)
    seed_intercept = float(p1[dependent] - seed_slope * start)
    tangents = np.arange(int(start + ENDPOINT_TRIM_RATIO * span),
                         int(stop - ENDPOINT_TRIM_RATIO * span), SAMPLE_STEP_PX)
    if len(tangents) < MIN_SAMPLES:
        return None, "too_few_samples"
    normal_size, tangent_size = gray.shape if horizontal else gray.shape[::-1]
    samples = []
    for tangent in tangents:
        estimate = seed_slope * tangent + seed_intercept
        lo = int(np.floor(estimate)) - SEARCH_RADIUS_PX
        hi = int(np.floor(estimate)) + SEARCH_RADIUS_PX + 1
        if (lo < 0 or hi > normal_size or tangent < PROFILE_HALF_WIDTH_PX
                or tangent + PROFILE_HALF_WIDTH_PX >= tangent_size):
            continue
        half = PROFILE_HALF_WIDTH_PX
        strip = (gray[lo:hi, tangent-half:tangent+half+1] if horizontal
                 else gray[tangent-half:tangent+half+1, lo:hi])
        profile = np.median(strip, axis=1 if horizontal else 0).astype(np.float32)
        profile = cv2.GaussianBlur(profile[None, :], (5, 1), 1.0)[0]
        gradient = np.diff(profile) * polarity
        peak = int(np.argmax(gradient))
        strength = float(gradient[peak])
        # Leave enough real pixels on both sides to verify the transition.
        if strength < MIN_GRADIENT or peak < 13 or peak + 14 >= len(profile):
            continue
        before = float(np.median(profile[peak-12:peak-5]))
        after = float(np.median(profile[peak+6:peak+13]))
        if polarity * (after - before) < MIN_CONTRAST:
            continue
        competitors = np.abs(gradient).copy()
        competitors[max(0, peak-6):peak+7] = 0
        if float(competitors.max()) > strength * MAX_COMPETING_GRADIENT_RATIO:
            continue
        # Parabolic interpolation estimates a subpixel transition location.
        denominator = float(gradient[peak-1] - 2*gradient[peak] + gradient[peak+1])
        offset = (0.5 * float(gradient[peak-1] - gradient[peak+1]) / denominator
                  if abs(denominator) > 1e-8 else 0.0)
        location = lo + peak + 0.5 + float(np.clip(offset, -0.5, 0.5))
        samples.append((float(tangent), location))
    if len(samples) < max(MIN_SAMPLES, int(np.ceil(len(tangents) * MIN_SUPPORT_RATIO))):
        return None, "weak_or_ambiguous_edge"

    points = np.asarray(samples, np.float64)
    slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
    residual = points[:, 1] - (slope * points[:, 0] + intercept)
    center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - center)))
    keep = np.abs(residual - center) <= max(1.0, 3.0 * 1.4826 * mad)
    if int(keep.sum()) < max(MIN_SAMPLES, int(np.ceil(len(tangents) * MIN_SUPPORT_RATIO))):
        return None, "insufficient_line_support"
    # A small cluster cannot stand in for the whole side, even if perfectly straight.
    bins = np.linspace(float(tangents[0]), float(tangents[-1]) + 1, 4)
    if np.any(np.histogram(points[keep, 0], bins=bins)[0] == 0):
        return None, "poor_sample_coverage"
    slope, intercept = np.polyfit(points[keep, 0], points[keep, 1], 1)
    residual = np.abs(points[keep, 1] - (slope * points[keep, 0] + intercept))
    p95 = float(np.percentile(residual, 95))
    if p95 > MAX_RESIDUAL_P95_PX:
        return None, "non_linear_edge"
    movement = max(abs((slope-seed_slope)*value + intercept-seed_intercept)
                   for value in (start, stop))
    if movement > MAX_CORNER_SHIFT_PX:
        return None, "edge_shift_too_large"
    if movement <= MIN_CORRECTION_PX:
        slope, intercept = seed_slope, seed_intercept
    return (float(slope), float(intercept), int(keep.sum()), len(tangents), p95), "ok"


def _intersection(horizontal, vertical):
    ah, bh = horizontal[:2]
    av, bv = vertical[:2]
    denominator = 1.0 - ah * av
    if abs(denominator) < 1e-6:
        return None
    y = (ah * bv + bh) / denominator
    return (av * y + bv, y)


def refine_panel_polygon_from_raw(
    image: np.ndarray,
    polygon: Optional[np.ndarray],
    *,
    source_name: str = "",
) -> Optional[np.ndarray]:
    """Return a validated refinement, or the untouched original on rejection.

    Call once on the raw reference before masks/grid canonicalization, never
    on each lighting after a shared reference has already been selected.
    """
    if polygon is None:
        return None

    def reject(reason, side="-"):
        logger.info("[boundary-gradient] source=%s status=kept reason=%s side=%s",
                    source_name or "-", reason, side)
        return polygon

    seed = np.asarray(polygon, dtype=np.float32)
    if (image is None or image.size == 0 or image.dtype != np.uint8
            or seed.shape != (4, 2) or not np.isfinite(seed).all()
            or not cv2.isContourConvex(seed)):
        return reject("unsupported_input")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    else:
        return reject("unsupported_channels")
    lines = {}
    for side in _SIDES:
        line, reason = _sample_edge(gray, seed, side)
        if line is None:
            return reject(reason, side[0])
        lines[side[0]] = line
    corners = [_intersection(lines[h], lines[v]) for h, v in
               (("top", "left"), ("top", "right"), ("bottom", "right"), ("bottom", "left"))]
    if any(corner is None for corner in corners):
        return reject("parallel_edges")
    candidate = np.asarray(corners, np.float32)
    height, width = gray.shape
    if (not np.isfinite(candidate).all() or not cv2.isContourConvex(candidate)
            or candidate[:, 0].min() < 0 or candidate[:, 0].max() >= width
            or candidate[:, 1].min() < 0 or candidate[:, 1].max() >= height):
        return reject("invalid_geometry")
    original_area = cv2.contourArea(seed, oriented=True)
    area = cv2.contourArea(candidate, oriented=True)
    if original_area * area <= 0:
        return reject("invalid_winding")
    area_ratio = area / original_area
    if abs(area_ratio - 1.0) > MAX_AREA_CHANGE_RATIO:
        return reject("area_change_too_large")
    max_shift = float(np.linalg.norm(candidate - seed, axis=1).max())
    if max_shift > MAX_CORNER_SHIFT_PX:
        return reject("corner_shift_too_large")
    if max_shift <= MIN_CORRECTION_PX:
        return reject("already_aligned")
    logger.info(
        "[boundary-gradient] source=%s status=applied max_shift=%.2fpx area_ratio=%.5f "
        "support=%s residual_p95=%s",
        source_name or "-", max_shift, area_ratio,
        ",".join(f"{side}:{line[2]}/{line[3]}" for side, line in lines.items()),
        ",".join(f"{side}:{line[4]:.2f}px" for side, line in lines.items()),
    )
    return candidate
