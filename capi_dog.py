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
