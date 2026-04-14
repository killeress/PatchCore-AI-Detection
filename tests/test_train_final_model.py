"""Smoke test for scripts.over_review_poc.train_final_model CLI."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


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
