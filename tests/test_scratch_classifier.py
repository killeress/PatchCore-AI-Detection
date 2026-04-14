"""Unit tests for scratch_classifier module (bundle I/O, predict)."""
from __future__ import annotations

import pickle
from dataclasses import asdict
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
    assert asdict(loaded_meta) == asdict(meta)


def _train_tiny_stub(tmp_path: Path, device: str = "cpu") -> Path:
    """Create a minimal valid bundle + fake DINOv2 weights for testing."""
    # Build LoRA state_dict keys that match what ScratchClassifier expects
    # (last 2 blocks, qkv/mlp.fc1/mlp.fc2; rank=8 for speed)
    rank = 8
    lora_sd = {}
    for blk in range(-2, 0):  # last 2 blocks (logical indexing in implementation)
        for proj in ("attn.qkv", "mlp.fc1", "mlp.fc2"):
            for side, out_dim in (("lora_A.weight", 768), ("lora_B.weight", 768)):
                # Shapes based on DINOv2 ViT-B/14 (D=768, qkv→3*768, mlp→3072)
                in_dim = 768
                actual_out = 3 * 768 if "qkv" in proj else (3072 if "fc1" in proj else 768)
                in_dim_actual = 768 if "fc1" in proj or "qkv" in proj else 3072
                if "lora_A" in side:
                    lora_sd[f"blocks.{blk}.{proj}.lora_A.weight"] = torch.randn(rank, in_dim_actual)
                else:
                    lora_sd[f"blocks.{blk}.{proj}.lora_B.weight"] = torch.zeros(actual_out, rank)
    # Random linear head produces 768-d embedding; LogReg trained on noise.
    X = np.random.randn(20, 768)
    y = np.array([0]*10 + [1]*10)
    logreg = LogisticRegression(max_iter=200).fit(X, y)

    meta = ScratchClassifierMetadata(
        preprocessing_id="v3_clahe_cl4.0_tg8",
        clahe_clip=4.0, clahe_tile=8, input_size=224,
        dinov2_repo="facebookresearch/dinov2", dinov2_model="dinov2_vitb14",
        lora_rank=rank, lora_alpha=16, lora_n_blocks=2,
        conformal_threshold=0.8, safety_multiplier=1.1,
        trained_at="t", dataset_sha256="x", git_commit="y",
    )
    bundle_path = tmp_path / "bundle.pkl"
    save_bundle(bundle_path, lora_sd, logreg, meta, np.array([0.5, 0.8]))
    return bundle_path


@pytest.mark.slow
def test_classifier_loads_bundle_and_weights(tmp_path: Path):
    """Requires network on first run (downloads DINOv2); marked slow."""
    from scratch_classifier import ScratchClassifier
    bundle_path = _train_tiny_stub(tmp_path)
    clf = ScratchClassifier(bundle_path=bundle_path, device="cpu")
    assert clf.conformal_threshold == 0.8
    assert clf.metadata.lora_rank == 8
    assert clf.metadata.lora_n_blocks == 2


def test_classifier_missing_bundle_raises(tmp_path: Path):
    from scratch_classifier import ScratchClassifier, ScratchClassifierLoadError
    with pytest.raises(ScratchClassifierLoadError):
        ScratchClassifier(bundle_path=tmp_path / "missing.pkl", device="cpu")
