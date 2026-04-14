"""Scratch binary classifier for CAPI over-review filtering.

Exposes ScratchClassifier that wraps LoRA-fine-tuned DINOv2 ViT-B/14 +
LogisticRegression head. Bundles are pickled dicts containing LoRA weights,
LogReg, metadata, and calibration scores.
"""
from __future__ import annotations

import logging
import math
import os
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TRUST_REMOTE_CODE", "1")

import numpy as np
import torch
import torch.nn as nn
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
        "calibration_scores": np.asarray(calibration_scores, dtype=np.float64),
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
    required_keys = {"lora_state_dict", "logreg", "metadata", "calibration_scores", "format_version"}
    missing_keys = required_keys - set(payload.keys())
    if missing_keys:
        raise ScratchClassifierLoadError(f"Bundle missing keys: {sorted(missing_keys)}")
    try:
        meta = ScratchClassifierMetadata(**payload["metadata"])
    except TypeError as e:
        raise ScratchClassifierLoadError(f"Metadata schema mismatch: {e}") from e
    return (
        payload["lora_state_dict"],
        payload["logreg"],
        meta,
        payload["calibration_scores"],
    )


class LoRALinear(nn.Module):
    """Linear with LoRA adapter: W + scale * B @ A. Only A, B trainable.

    Mirrors scripts.over_review_poc.finetune_lora.LoRALinear — duplicated here
    so deployment doesn't depend on POC script package. Keep the two in sync
    when LoRA adapter implementation changes.
    """
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        nn.init.zeros_(self.lora_B.weight)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


def _apply_lora(model: nn.Module, n_blocks: int, rank: int, alpha: int) -> None:
    """Replace qkv + mlp.fc1 + mlp.fc2 of last n_blocks with LoRA variants (in-place)."""
    targets = model.blocks[-n_blocks:]
    for block in targets:
        block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha)
        block.mlp.fc1 = LoRALinear(block.mlp.fc1, rank, alpha)
        block.mlp.fc2 = LoRALinear(block.mlp.fc2, rank, alpha)


def _load_dinov2(repo: str, name: str,
                 weights_path: str | Path | None) -> nn.Module:
    """Load DINOv2 from torch.hub; if weights_path given, load local checkpoint."""
    if weights_path is not None:
        model = torch.hub.load(repo, name, pretrained=False, source="github")
        state = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state)
    else:
        model = torch.hub.load(repo, name, source="github")
    return model


class ScratchClassifier:
    def __init__(
        self,
        bundle_path: str | Path,
        dinov2_weights_path: str | Path | None = None,
        device: str = "cuda",
    ):
        lora_sd, logreg, meta, calib_scores = load_bundle(bundle_path)
        self.metadata = meta
        self._logreg = logreg
        self._calibration_scores = calib_scores

        try:
            model = _load_dinov2(meta.dinov2_repo, meta.dinov2_model,
                                  dinov2_weights_path)
        except Exception as e:  # pragma: no cover - depends on network/filesystem
            raise ScratchClassifierLoadError(
                f"Failed to load DINOv2 ({meta.dinov2_repo}/{meta.dinov2_model}): {e}"
            ) from e

        _apply_lora(model, meta.lora_n_blocks, meta.lora_rank, meta.lora_alpha)
        missing, unexpected = model.load_state_dict(lora_sd, strict=False)
        if unexpected:
            logger.warning("Unexpected keys in LoRA state_dict: %s", unexpected[:5])

        model.eval()
        self._device = torch.device(device)
        self._model = model.to(self._device)
        logger.info("ScratchClassifier loaded: rank=%d blocks=%d threshold=%.4f",
                    meta.lora_rank, meta.lora_n_blocks, meta.conformal_threshold)

    @property
    def conformal_threshold(self) -> float:
        """Raw conformal threshold (calib max true_ng score)."""
        return self.metadata.conformal_threshold

    @property
    def device(self) -> torch.device:
        return self._device
