"""Smoke test for scripts.over_review_poc.train_final_model CLI."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_final_calibration_includes_true_defect_subtypes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import torch
    from scripts.over_review_poc import train_final_model as training

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("fixture", encoding="utf-8")
    labels = ["over_surface_scratch", "true_ng", "true_black_spot", "true_white_spot",
              "misrescue_negative", "over_other"]
    samples = [SimpleNamespace(original_label=label, label="scratch" if i == 0 else "not_scratch")
               for i, label in enumerate(labels)]
    monkeypatch.setattr(training, "load_samples", lambda path: samples)
    monkeypatch.setattr(training, "_group_aware_split", lambda *args: (np.array([0, 1]), np.array([2, 3, 4, 5])))
    monkeypatch.setattr(training, "_train_fold", lambda *args: torch.nn.Linear(1, 1))
    monkeypatch.setattr(training, "_extract_cls", lambda model, rows, *args: np.zeros((len(rows), 1)))
    class LogReg:
        def __init__(self, **kwargs):
            pass
        def fit(self, *args):
            return self
        def predict_proba(self, features):
            scores = np.array([0.2, 0.4, 0.8, 0.99])
            return np.stack([1 - scores, scores], axis=1)
    monkeypatch.setattr(training, "LogisticRegression", LogReg)
    monkeypatch.setattr(training, "save_bundle", lambda *args: None)
    summary = training.main(["--manifest", str(manifest), "--output", str(tmp_path / "candidate.pkl")])
    assert summary["calib_ng_count"] == 3
    assert summary["conformal_threshold"] == pytest.approx(0.8)


def test_scratch_only_training_fails_before_loading_model(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from scripts.over_review_poc import train_final_model as training
    samples = [SimpleNamespace(original_label="over_surface_scratch", label="scratch") for _ in range(4)]
    monkeypatch.setattr(training, "load_samples", lambda path: samples)
    monkeypatch.setattr(training, "_group_aware_split", lambda *args: (np.array([0, 1]), np.array([2, 3])))
    def must_not_train(*args):
        pytest.fail("Invalid data must be rejected before starting LoRA training")
    monkeypatch.setattr(training, "_train_fold", must_not_train)
    with pytest.raises(ValueError, match="兩類資料"):
        training.main(["--manifest", str(tmp_path / "manifest.csv"), "--output", str(tmp_path / "candidate.pkl")])


@pytest.mark.slow
def test_train_final_model_smoke(tmp_path):
    manifest = REPO_ROOT / "datasets" / "over_review" / "manifest.csv"
    if not manifest.exists():
        pytest.skip("Real manifest not available in CI")
    out = tmp_path / "bundle.pkl"
    # Run with minimal epochs to keep test fast
    res = subprocess.run(
        [sys.executable, "-m", "scripts.over_review_poc.train_final_model",
         "--manifest", str(manifest),
         "--transform", "clahe", "--clahe-clip", "4.0",
         "--rank", "4", "--n-lora-blocks", "1", "--epochs", "1",
         "--calib-frac", "0.2",
         "--output", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert res.returncode == 0, f"Script failed: {res.stderr}"
    assert out.exists()
    # Verify bundle is loadable
    from scratch_classifier import load_bundle
    lora_sd, logreg, meta, calib = load_bundle(out)
    assert meta.lora_rank == 4
    assert meta.conformal_threshold > 0.0
    assert len(calib) > 0
