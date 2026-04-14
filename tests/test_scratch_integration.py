"""Integration test: CAPIInferencer with scratch filter enabled (mocked classifier)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer, TileInfo, ImageResult


def _make_image_result_with_ng_tiles(n: int) -> ImageResult:
    tiles = [
        TileInfo(tile_id=i, x=0, y=0, width=512, height=512,
                 image=np.zeros((512, 512, 3), dtype=np.uint8))
        for i in range(n)
    ]
    ir = ImageResult(
        image_path=Path("/fake"), image_size=(1024, 1024),
        otsu_bounds=(0, 0, 1024, 1024),
        exclusion_regions=[], tiles=tiles,
        excluded_tile_count=0, processed_tile_count=n, processing_time=0.0,
    )
    ir.anomaly_tiles = [(t, 0.9, None) for t in tiles]
    return ir


class _FakeClassifier:
    def __init__(self, score=0.95):
        self.conformal_threshold = 0.7
        self._score = score
    def predict(self, image):
        return self._score


def test_get_scratch_filter_disabled_returns_none(tmp_path):
    cfg = CAPIConfig()
    cfg.scratch_classifier_enabled = False
    inferencer = CAPIInferencer(config=cfg, model_path="")
    assert inferencer._get_scratch_filter() is None


def test_get_scratch_filter_caches_on_success(tmp_path):
    cfg = CAPIConfig()
    cfg.scratch_classifier_enabled = True
    inferencer = CAPIInferencer(config=cfg, model_path="")
    with patch("capi_inference.ScratchClassifier", return_value=_FakeClassifier()) as mock_cls:
        sf1 = inferencer._get_scratch_filter()
        sf2 = inferencer._get_scratch_filter()
        assert sf1 is sf2
        assert mock_cls.call_count == 1    # cached


def test_get_scratch_filter_load_failure_sentinel(tmp_path):
    from scratch_classifier import ScratchClassifierLoadError
    cfg = CAPIConfig()
    cfg.scratch_classifier_enabled = True
    inferencer = CAPIInferencer(config=cfg, model_path="")
    with patch("capi_inference.ScratchClassifier",
               side_effect=ScratchClassifierLoadError("no bundle")) as mock_cls:
        sf1 = inferencer._get_scratch_filter()
        sf2 = inferencer._get_scratch_filter()
        assert sf1 is None
        assert sf2 is None
        assert mock_cls.call_count == 1   # no retry


def test_process_panel_applies_scratch_filter_inline(tmp_path, monkeypatch):
    """Simulate the process_panel hook: build ImageResult with NG tiles,
    fake the filter, verify apply_to_image_result was called."""
    cfg = CAPIConfig()
    cfg.scratch_classifier_enabled = True
    inferencer = CAPIInferencer(config=cfg, model_path="")

    from scratch_filter import ScratchFilter
    fake_sf = ScratchFilter(_FakeClassifier(score=0.99), safety_multiplier=1.0)
    monkeypatch.setattr(inferencer, "_get_scratch_filter", lambda: fake_sf)

    ir = _make_image_result_with_ng_tiles(2)
    sf = inferencer._get_scratch_filter()
    sf.apply_to_image_result(ir)
    # C1 fix: tiles remain in anomaly_tiles (for DB audit), just flagged
    assert len(ir.anomaly_tiles) == 2
    assert ir.scratch_filter_count == 2
    for t, _, _ in ir.anomaly_tiles:
        assert t.scratch_filtered is True
