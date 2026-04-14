"""Unit tests for scratch_classifier module (bundle I/O, predict)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from scratch_classifier import (
    ScratchClassifierMetadata,
    save_bundle,
    load_bundle,
)


def _fake_metadata() -> ScratchClassifierMetadata:
    return ScratchClassifierMetadata(
        preprocessing_id="v3_clahe_cl4.0_tg8",
        clahe_clip=4.0,
        clahe_tile=8,
        input_size=224,
        dinov2_repo="facebookresearch/dinov2",
        dinov2_model="dinov2_vitb14",
        lora_rank=16,
        lora_alpha=16,
        lora_n_blocks=2,
        conformal_threshold=0.85,
        safety_multiplier=1.1,
        trained_at="2026-04-14T10:00:00",
        dataset_sha256="abc123",
        git_commit="deadbeef",
    )


def test_bundle_roundtrip(tmp_path: Path):
    lora_sd = {"lora_A.weight": torch.randn(16, 768)}
    logreg = LogisticRegression().fit(np.random.randn(10, 768), [0]*5 + [1]*5)
    calib_scores = np.array([0.1, 0.2, 0.85])
    meta = _fake_metadata()
    bundle_path = tmp_path / "bundle.pkl"

    save_bundle(bundle_path, lora_sd, logreg, meta, calib_scores)

    loaded_sd, loaded_logreg, loaded_meta, loaded_calib = load_bundle(bundle_path)
    assert list(loaded_sd.keys()) == ["lora_A.weight"]
    assert torch.allclose(loaded_sd["lora_A.weight"], lora_sd["lora_A.weight"])
    assert loaded_logreg.coef_.shape == logreg.coef_.shape
    assert np.allclose(loaded_logreg.coef_, logreg.coef_)
    assert loaded_meta.conformal_threshold == 0.85
    assert loaded_meta.preprocessing_id == "v3_clahe_cl4.0_tg8"
    assert np.allclose(loaded_calib, calib_scores)
