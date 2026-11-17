"""Shared white-mura DoG math for previews, training and inference."""
from __future__ import annotations

import math
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


def dog_residual(image, params=None):
    params = normalize_params(params)
    work = np.asarray(image, dtype=np.float32)
    signal = cv2.GaussianBlur(
        work, (0, 0), sigmaX=params["signal_sigma_x"],
        sigmaY=params["signal_sigma_y"], borderType=cv2.BORDER_REFLECT_101,
    )
    background = cv2.GaussianBlur(
        work, (0, 0), sigmaX=params["background_sigma_x"],
        sigmaY=params["background_sigma_y"], borderType=cv2.BORDER_REFLECT_101,
    )
    return signal - background


def dog_residual_crop(image, bounds, params=None):
    """Filter real surrounding pixels before cropping; keep sub-gray precision."""
    params = normalize_params(params)
    x1, y1, x2, y2 = bounds
    h, w = image.shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("DoG crop bounds must be inside the source image")
    pad_x = math.ceil(4 * params["background_sigma_x"])
    pad_y = math.ceil(4 * params["background_sigma_y"])
    bx1, by1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    bx2, by2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    work = image[by1:by2, bx1:bx2]
    if work.ndim == 3:
        work = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    residual = dog_residual(work, params)[y1-by1:y2-by1, x1-bx1:x2-bx1].copy()
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
