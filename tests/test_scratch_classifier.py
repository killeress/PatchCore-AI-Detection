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
    # DINOv2 ViT-B/14 has 12 transformer blocks; LoRA is applied to last 2 → indices 10, 11
    for blk in (10, 11):
        for proj in ("attn.qkv", "mlp.fc1", "mlp.fc2"):
            for side, out_dim in (("lora_A.weight", 768), ("lora_B.weight", 768)):
                # Shapes based on DINOv2 ViT-B/14 (D=768, qkv→3*768, mlp→3072)
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

    # Verify LoRA weights actually loaded — pick one key from the bundle and
    # compare to the value in the loaded model. Regression test for the bug
    # where range(-2, 0) produced keys that silently failed to load.
    import pickle
    with open(bundle_path, "rb") as f:
        payload = pickle.load(f)
    bundle_lora_sd = payload["lora_state_dict"]
    sample_key = next(iter(bundle_lora_sd.keys()))    # e.g. "blocks.10.attn.qkv.lora_A.weight"
    loaded_model_sd = clf._model.state_dict()
    assert sample_key in loaded_model_sd, f"{sample_key} missing from loaded model"
    assert torch.allclose(
        loaded_model_sd[sample_key].cpu(),
        bundle_lora_sd[sample_key].cpu(),
    ), f"{sample_key} value mismatch — weights didn't transfer"


def test_classifier_missing_bundle_raises(tmp_path: Path):
    from scratch_classifier import ScratchClassifier, ScratchClassifierLoadError
    with pytest.raises(ScratchClassifierLoadError):
        ScratchClassifier(bundle_path=tmp_path / "missing.pkl", device="cpu")


@pytest.mark.slow
def test_classifier_fails_when_no_lora_keys_match(tmp_path: Path):
    """I1 regression: bundle with zero matching LoRA keys should raise, not silently skip."""
    from scratch_classifier import ScratchClassifier, ScratchClassifierLoadError
    # Create bundle with intentionally wrong key names
    lora_sd = {"wrong.key.name.lora_A.weight": torch.randn(8, 768)}
    logreg = LogisticRegression(max_iter=200).fit(
        np.random.randn(20, 768), np.array([0]*10 + [1]*10))
    meta = ScratchClassifierMetadata(
        preprocessing_id="v3_clahe_cl4.0_tg8",
        clahe_clip=4.0, clahe_tile=8, input_size=224,
        dinov2_repo="facebookresearch/dinov2", dinov2_model="dinov2_vitb14",
        lora_rank=8, lora_alpha=16, lora_n_blocks=2,
        conformal_threshold=0.8, safety_multiplier=1.1,
        trained_at="t", dataset_sha256="x", git_commit="y",
    )
    bundle_path = tmp_path / "bad.pkl"
    save_bundle(bundle_path, lora_sd, logreg, meta, np.array([0.5]))

    with pytest.raises(ScratchClassifierLoadError,
                       match="No LoRA keys from bundle matched"):
        ScratchClassifier(bundle_path=bundle_path, device="cpu")


from PIL import Image


def _make_fake_image(size: int = 256) -> Image.Image:
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_preprocessing_matches_poc(tmp_path: Path):
    """ScratchClassifier preprocessing must produce identical tensor to POC helper."""
    from scratch_classifier import _build_transform  # module-private helper
    from scripts.over_review_poc.features import build_transform_clahe

    img = _make_fake_image()
    poc_tensor = build_transform_clahe(clip_limit=4.0, tile_grid=8)(img)
    our_tensor = _build_transform(clahe_clip=4.0, clahe_tile=8, input_size=224)(img)
    assert torch.allclose(poc_tensor, our_tensor, atol=1e-6)


@pytest.mark.slow
def test_predict_score_range(tmp_path: Path):
    from scratch_classifier import ScratchClassifier
    bundle_path = _train_tiny_stub(tmp_path)
    clf = ScratchClassifier(bundle_path=bundle_path, device="cpu")
    img = _make_fake_image()
    score = clf.predict(img)
    assert 0.0 <= score <= 1.0


@pytest.mark.slow
def test_predict_batch_matches_loop(tmp_path: Path):
    from scratch_classifier import ScratchClassifier
    bundle_path = _train_tiny_stub(tmp_path)
    clf = ScratchClassifier(bundle_path=bundle_path, device="cpu")
    imgs = [_make_fake_image(size=s) for s in (100, 150, 200, 256)]
    batch_scores = clf.predict_batch(imgs)
    loop_scores = np.array([clf.predict(i) for i in imgs])
    assert np.allclose(batch_scores, loop_scores, atol=1e-5)


def test_to_pil_rejects_float_ndarray():
    """I1 regression: float ndarrays must be rejected explicitly, not silently corrupted."""
    from scratch_classifier import _to_pil
    float_img = np.random.rand(64, 64, 3).astype(np.float32)  # [0,1] range
    with pytest.raises(TypeError, match="float ndarray"):
        _to_pil(float_img)


def test_to_pil_rejects_wrong_channel_count():
    """I2 regression: RGBA or other non-3-channel shapes must raise ValueError."""
    from scratch_classifier import _to_pil
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="HxW or HxWx3"):
        _to_pil(rgba)


def test_to_pil_accepts_grayscale():
    """Sanity: 2D uint8 grayscale should still work (converted to 3-channel)."""
    from scratch_classifier import _to_pil
    from PIL import Image
    gray = np.zeros((32, 32), dtype=np.uint8)
    result = _to_pil(gray)
    assert isinstance(result, Image.Image)
    assert result.size == (32, 32)


def test_load_dinov2_uses_local_source_when_repo_path_given(tmp_path, monkeypatch):
    """With repo_local_path set, _load_dinov2 must call torch.hub.load with source='local'
    (no network). Offline-deployment regression test."""
    from scratch_classifier import _load_dinov2
    import torch.nn as nn

    repo_dir = tmp_path / "dinov2_repo"
    repo_dir.mkdir()
    (repo_dir / "hubconf.py").write_text("")   # minimal valid local hub repo

    captured = {}

    def fake_hub_load(path_or_repo, name, **kwargs):
        captured["path"] = str(path_or_repo)
        captured["name"] = name
        captured.update(kwargs)
        return nn.Linear(2, 2)

    monkeypatch.setattr("torch.hub.load", fake_hub_load)
    monkeypatch.setattr("torch.load", lambda *a, **kw: {"fake": torch.zeros(1)})

    _load_dinov2("facebookresearch/dinov2", "dinov2_vitb14",
                 weights_path=None, repo_local_path=str(repo_dir))

    assert captured["source"] == "local"
    assert captured["path"] == str(repo_dir)
    assert captured["name"] == "dinov2_vitb14"


def test_load_dinov2_local_source_with_weights_path(tmp_path, monkeypatch):
    """Both repo_local_path AND weights_path provided → local source + state_dict load."""
    from scratch_classifier import _load_dinov2
    import torch.nn as nn

    repo_dir = tmp_path / "dinov2_repo"
    repo_dir.mkdir()
    (repo_dir / "hubconf.py").write_text("")
    weights = tmp_path / "dinov2.pth"
    weights.write_bytes(b"fake")

    captured = {"torch_load_called": False}

    def fake_hub_load(*args, **kwargs):
        captured["source"] = kwargs.get("source")
        m = nn.Linear(2, 2)
        return m

    def fake_torch_load(path, **kw):
        captured["torch_load_called"] = True
        captured["weights_path"] = str(path)
        # Return a state_dict whose only key matches the nn.Linear we return above.
        return {"weight": torch.zeros(2, 2), "bias": torch.zeros(2)}

    monkeypatch.setattr("torch.hub.load", fake_hub_load)
    monkeypatch.setattr("torch.load", fake_torch_load)

    _load_dinov2("facebookresearch/dinov2", "dinov2_vitb14",
                 weights_path=weights, repo_local_path=repo_dir)

    assert captured["source"] == "local"
    assert captured["torch_load_called"] is True
    assert captured["weights_path"] == str(weights)


def test_load_dinov2_missing_local_repo_raises(tmp_path, monkeypatch):
    """If repo_local_path is given but the directory doesn't exist, raise explicit error."""
    from scratch_classifier import _load_dinov2

    missing = tmp_path / "does_not_exist"
    # Do not create the directory.

    with pytest.raises((FileNotFoundError, ValueError)):
        _load_dinov2("facebookresearch/dinov2", "dinov2_vitb14",
                     weights_path=None, repo_local_path=missing)


def test_load_dinov2_falls_back_to_github_when_no_local_path(monkeypatch):
    """Preserve legacy behavior: no repo_local_path → source='github' (needs network/cache)."""
    from scratch_classifier import _load_dinov2
    import torch.nn as nn

    captured = {}

    def fake_hub_load(repo_or_path, name, **kwargs):
        captured["source"] = kwargs.get("source")
        return nn.Linear(2, 2)

    monkeypatch.setattr("torch.hub.load", fake_hub_load)

    _load_dinov2("facebookresearch/dinov2", "dinov2_vitb14", weights_path=None)
    assert captured["source"] == "github"
