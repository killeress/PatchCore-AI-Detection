"""Scratch binary classifier for CAPI over-review filtering.

Exposes ScratchClassifier that wraps LoRA-fine-tuned DINOv2 ViT-B/14 +
LogisticRegression head. Bundles are pickled dicts containing LoRA weights,
LogReg, metadata, and calibration scores.
"""
from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TRUST_REMOTE_CODE", "1")

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@dataclass
class ScratchClassifierMetadata:
    """Frozen at training time; single source of truth for preprocessing + model id."""
    preprocessing_id: str
    clahe_clip: float
    clahe_tile: int
    input_size: int
    dinov2_repo: str
    dinov2_model: str
    lora_rank: int
    lora_alpha: int
    lora_n_blocks: int
    conformal_threshold: float
    safety_multiplier: float
    trained_at: str
    dataset_sha256: str
    git_commit: str


class ScratchClassifierLoadError(RuntimeError):
    """Raised when bundle / DINOv2 weights cannot be loaded."""


def save_bundle(
    path: str | Path,
    lora_state_dict: dict[str, Any],
    logreg: LogisticRegression,
    metadata: ScratchClassifierMetadata,
    calibration_scores: np.ndarray,
) -> None:
    """Pickle the deployment artifact to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lora_state_dict": lora_state_dict,
        "logreg": logreg,
        "metadata": asdict(metadata),
        "calibration_scores": np.asarray(calibration_scores),
        "format_version": 1,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved scratch classifier bundle: %s", path)


def load_bundle(
    path: str | Path,
) -> tuple[dict[str, Any], LogisticRegression, ScratchClassifierMetadata, np.ndarray]:
    """Load pickled artifact; returns (lora_sd, logreg, metadata, calib_scores)."""
    path = Path(path)
    if not path.exists():
        raise ScratchClassifierLoadError(f"Bundle not found: {path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("format_version", 0) != 1:
        raise ScratchClassifierLoadError(
            f"Unsupported bundle format_version: {payload.get('format_version')}"
        )
    meta = ScratchClassifierMetadata(**payload["metadata"])
    return (
        payload["lora_state_dict"],
        payload["logreg"],
        meta,
        payload["calibration_scores"],
    )
