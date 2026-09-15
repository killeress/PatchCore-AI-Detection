"""Lightweight, shared validation for the SoftPatch+ experimental recipe."""

from __future__ import annotations

import math
from typing import Any


SOFTPATCH_MODE = "softpatch_plus_v1"
SOFTPATCH_DEFAULTS = {
    "discriminator": "lof_gaussian",
    "soft_weight": True,
    "weight_strength": 1.0,
    "context_overlap": True,
    "projection_dim": 32,
    "reference_size": 2048,
}


def normalize_softpatch_config(raw: Any) -> dict:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("softpatch_plus_config must be an object")
    unknown = set(raw) - set(SOFTPATCH_DEFAULTS)
    if unknown:
        raise ValueError(f"softpatch_plus_config has unknown keys: {sorted(unknown)}")
    result = {**SOFTPATCH_DEFAULTS, **raw}
    if result["discriminator"] not in ("lof", "lof_gaussian"):
        raise ValueError("softpatch_plus_config.discriminator must be lof or lof_gaussian")
    for key in ("soft_weight", "context_overlap"):
        if type(result[key]) is not bool:
            raise ValueError(f"softpatch_plus_config.{key} must be a boolean")
    for key, low, high, integer in (
        ("weight_strength", 0, 4, False),
        ("projection_dim", 8, 128, True),
        ("reference_size", 256, 8192, True),
    ):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"softpatch_plus_config.{key} must be numeric")
        if not math.isfinite(value) or not low <= value <= high or (integer and int(value) != value):
            raise ValueError(f"softpatch_plus_config.{key} must be {'an integer ' if integer else ''}between {low} and {high}")
        result[key] = int(value) if integer else float(value)
    return result


def cleaning_keep_ratio_min(mode: str) -> float:
    return 0.50 if mode == SOFTPATCH_MODE else 0.90
