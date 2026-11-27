"""Shared white-mura DoG math for previews, training and inference."""
from __future__ import annotations

import math
from pathlib import Path
import cv2
import numpy as np


DEFAULT_PARAMS = {
    "signal_sigma_x": 20.0, "signal_sigma_y": 6.0,
    "background_sigma_x": 96.0, "background_sigma_y": 32.0,
    "gain": 40.0, "offset": 128.0,
}
BLACK_MURA_PARAMS = {
    **DEFAULT_PARAMS,
    "signal_sigma_x": 8.0, "signal_sigma_y": 8.0,
    "background_sigma_x": 64.0, "background_sigma_y": 64.0,
}
PARAM_LIMITS = {
    "signal_sigma_x": (.5, 64), "signal_sigma_y": (.5, 64),
    "background_sigma_x": (1, 192), "background_sigma_y": (1, 192),
    "gain": (1, 100), "offset": (0, 255),
}
_MASK_WEIGHT_EPS = 1e-12
_BOUNDARY_GUARD_PX = 2


def crop_mask_sidecar_path(crop_path: str | Path) -> Path:
    """Return the optional 0/255 panel-mask sidecar path for a saved tile crop."""
    path = Path(crop_path)
    return path.with_name(f"{path.stem}.mask.png")


def load_crop_mask_sidecar(
    crop_path: str | Path,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Load an optional panel-mask sidecar if present and geometry-compatible."""
    mask_path = crop_mask_sidecar_path(crop_path)
    if not mask_path.is_file():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if expected_shape is not None and mask.shape[:2] != tuple(expected_shape[:2]):
        return None
    return mask


def _prepare_panel_mask(mask, image_shape: tuple[int, ...]) -> np.ndarray | None:
    """Normalize an optional 2D panel mask; return None when all pixels are valid.

    When outside-panel pixels are present, step inward by ``_BOUNDARY_GUARD_PX``
    only along the exterior boundary (never along interior tile crop edges) so
    1-2 px polygon rasterization / lens anti-aliasing dark fringes cannot leak
    into the high-gain DoG residual.
    """
    if mask is None:
        return None
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2 or mask_arr.shape != image_shape[:2]:
        raise ValueError("DoG mask shape must match image height and width")
    valid = mask_arr > 0
    if bool(np.all(valid)):
        return None
    if not bool(np.any(valid)):
        return valid
    if _BOUNDARY_GUARD_PX > 0:
        ksize = 2 * _BOUNDARY_GUARD_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        outside = (~valid).astype(np.uint8)
        guarded_outside = cv2.dilate(
            outside, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0,
        ) > 0
        guarded_valid = valid & ~guarded_outside
        if bool(np.any(guarded_valid)):
            return guarded_valid
    return valid


def normalize_params(params=None):
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("params 必須是物件")
    if set(params) - set(DEFAULT_PARAMS):
        raise ValueError("包含不支援的 DoG 參數")
    result = {}
    for key, default in DEFAULT_PARAMS.items():
        value = params.get(key, default)
        if isinstance(value, bool):
            raise ValueError(f"{key} 必須是數值")
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"{key} 必須是數值") from None
        low, high = PARAM_LIMITS[key]
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} 必須介於 {low}～{high}")
        result[key] = value
    for axis in ("x", "y"):
        if result[f"background_sigma_{axis}"] <= result[f"signal_sigma_{axis}"]:
            raise ValueError(f"背景 sigma {axis.upper()} 必須大於平滑 sigma {axis.upper()}")
    return result


def _masked_gaussian_blur(
    work: np.ndarray,
    weights: np.ndarray,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    weighted = work * (weights[:, :, None] if work.ndim == 3 else weights)
    numerator = cv2.GaussianBlur(
        weighted, (0, 0), sigmaX=sigma_x,
        sigmaY=sigma_y, borderType=cv2.BORDER_REFLECT_101,
    )
    denominator = cv2.GaussianBlur(
        weights, (0, 0), sigmaX=sigma_x,
        sigmaY=sigma_y, borderType=cv2.BORDER_REFLECT_101,
    )
    safe_den = np.maximum(denominator, _MASK_WEIGHT_EPS)
    if work.ndim == 3:
        safe_den = safe_den[:, :, None]
    return numerator / safe_den


def dog_residual(image, params=None, *, mask=None):
    params = normalize_params(params)
    work = np.asarray(image, dtype=np.float32)
    valid = _prepare_panel_mask(mask, work.shape)
    if valid is None:
        signal = cv2.GaussianBlur(
            work, (0, 0), sigmaX=params["signal_sigma_x"],
            sigmaY=params["signal_sigma_y"], borderType=cv2.BORDER_REFLECT_101,
        )
        background = cv2.GaussianBlur(
            work, (0, 0), sigmaX=params["background_sigma_x"],
            sigmaY=params["background_sigma_y"], borderType=cv2.BORDER_REFLECT_101,
        )
        return signal - background

    if not bool(np.any(valid)):
        return np.zeros_like(work, dtype=np.float32)

    weights = valid.astype(np.float32)
    signal = _masked_gaussian_blur(
        work, weights, params["signal_sigma_x"], params["signal_sigma_y"],
    )
    background = _masked_gaussian_blur(
        work, weights, params["background_sigma_x"], params["background_sigma_y"],
    )
    residual = signal - background
    residual[~valid] = 0.0
    return residual


def dog_residual_crop(image, bounds, params=None, *, mask=None):
    """Filter real surrounding pixels before cropping; keep sub-gray precision."""
    params = normalize_params(params)
    x1, y1, x2, y2 = bounds
    h, w = image.shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("DoG crop bounds must be inside the source image")
    if mask is not None and np.asarray(mask).shape != (h, w):
        raise ValueError("DoG mask shape must match image height and width")
    pad_x = math.ceil(4 * params["background_sigma_x"])
    pad_y = math.ceil(4 * params["background_sigma_y"])
    bx1, by1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    bx2, by2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    work = image[by1:by2, bx1:bx2]
    if work.ndim == 3:
        work = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    crop_mask = None if mask is None else np.asarray(mask)[by1:by2, bx1:bx2]
    residual = dog_residual(work, params, mask=crop_mask)[y1-by1:y2-by1, x1-bx1:x2-bx1].copy()
    return residual, (bx1, by1, bx2, by2)


def filter_components(binary, min_area):
    """Keep connected support without joining nearby, independent candidates."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (np.asarray(binary) > 0).astype(np.uint8), connectivity=8,
    )
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(1, int(min_area))
    return keep[labels]


def detect_black_mura(image, bounds, *, panel_polygon=None, threshold=0.5, min_area=100):
    """Detect faint broad bright pressure marks using the debug 8/64 DoG.

    Thresholds are in source gray levels, before display gain/offset. Exclude
    four background sigmas along panel/image boundaries to avoid normal edge
    halos. The caller combines this evidence with the existing point detector.
    """
    threshold = float(threshold)
    min_area = int(min_area)
    if not math.isfinite(threshold) or threshold <= 0 or min_area < 1:
        raise ValueError("Black-mura threshold and min_area must be positive")
    residual, context_bounds = dog_residual_crop(image, bounds, BLACK_MURA_PARAMS)
    x1, y1, x2, y2 = bounds
    bx1, by1, bx2, by2 = context_bounds
    valid = np.full((by2-by1, bx2-bx1), 255, dtype=np.uint8)
    if panel_polygon is not None:
        valid[:] = 0
        polygon = np.rint(np.asarray(panel_polygon) - [bx1, by1]).astype(np.int32)
        cv2.fillPoly(valid, [polygon], 255)
    padded = cv2.copyMakeBorder(valid, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    clearance = 4 * BLACK_MURA_PARAMS["background_sigma_x"]
    eligible = distance[y1-by1:y2-by1, x1-bx1:x2-bx1] > clearance
    candidates = ((residual > threshold) & eligible).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    kept = filter_components(candidates, min_area)
    return kept, {
        "enabled": True, "method": "dog_8_64", "threshold": threshold,
        "min_area": min_area, "signal_sigma": 8.0, "background_sigma": 64.0,
        "boundary_clearance_px": int(clearance), "context_bounds": list(context_bounds),
        "eligible_pixels": int(np.count_nonzero(eligible)),
        "max_residual": max(0.0, float(residual[eligible].max())) if np.any(eligible) else 0.0,
        "max_component_area": int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats) > 1 else 0,
        "candidate_pixels": int(np.count_nonzero(kept)),
    }


def render_dog(residual, params=None):
    params = normalize_params(params)
    return np.rint(np.clip(params["offset"] + params["gain"] * residual, 0, 255)).astype(np.uint8)


def validate_execution_mode(pipeline, zone_pipelines=None, *, after_tiling=False):
    """Keep DoG out of foreground detection, which expects raw panel brightness."""
    pipelines = [pipeline or [], *(zone_pipelines or {}).values()]
    if not after_tiling and any(
        step.get("method") == "dog" for steps in pipelines for step in steps
    ):
        raise ValueError("白丸 DoG 必須啟用「先切分後處理」，讓產品定位使用原圖；訓練與推論會保存並套用相同流程")
