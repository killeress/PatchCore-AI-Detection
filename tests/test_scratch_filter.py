"""Unit tests for scratch_filter module + related TileInfo/ImageResult fields."""
from __future__ import annotations

import numpy as np
import pytest

from capi_inference import TileInfo, ImageResult


def _fake_tile(tid: int) -> TileInfo:
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    return TileInfo(tile_id=tid, x=0, y=0, width=512, height=512, image=img)


def test_tileinfo_has_scratch_fields():
    t = _fake_tile(1)
    assert t.scratch_score == 0.0
    assert t.scratch_filtered is False


def test_image_result_has_scratch_count():
    from pathlib import Path
    ir = ImageResult(
        image_path=Path("/fake"),
        image_size=(512, 512),
        otsu_bounds=(0, 0, 512, 512),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
    )
    assert ir.scratch_filter_count == 0


# --- ScratchFilter tests ---

class _MockClassifier:
    """Returns fixed score; exposes conformal_threshold for effective calc."""
    def __init__(self, fixed_score: float, conformal_threshold: float = 0.7,
                 raise_on_call: bool = False):
        self._score = fixed_score
        self.conformal_threshold = conformal_threshold
        self._raise = raise_on_call

    def predict(self, image):
        if self._raise:
            raise RuntimeError("simulated failure")
        return self._score


def _fake_image_result_with_tiles(n_tiles: int) -> ImageResult:
    from pathlib import Path
    tiles = [_fake_tile(i) for i in range(n_tiles)]
    ir = ImageResult(
        image_path=Path("/fake"),
        image_size=(1024, 1024),
        otsu_bounds=(0, 0, 1024, 1024),
        exclusion_regions=[],
        tiles=tiles,
        excluded_tile_count=0,
        processed_tile_count=n_tiles,
        processing_time=0.01,
    )
    # Populate anomaly_tiles: (TileInfo, score, anomaly_map)
    ir.anomaly_tiles = [(t, 0.9, None) for t in tiles]
    return ir


def test_filter_flips_high_score():
    from scratch_filter import ScratchFilter
    clf = _MockClassifier(fixed_score=0.95, conformal_threshold=0.7)
    sf = ScratchFilter(clf, safety_multiplier=1.0)   # threshold = 0.7
    ir = _fake_image_result_with_tiles(3)

    sf.apply_to_image_result(ir)

    assert len(ir.anomaly_tiles) == 0
    assert ir.scratch_filter_count == 3
    for t in ir.tiles:
        assert t.scratch_filtered is True
        assert t.scratch_score == pytest.approx(0.95)


def test_filter_keeps_low_score():
    from scratch_filter import ScratchFilter
    clf = _MockClassifier(fixed_score=0.2, conformal_threshold=0.7)
    sf = ScratchFilter(clf, safety_multiplier=1.0)   # threshold = 0.7
    ir = _fake_image_result_with_tiles(3)

    sf.apply_to_image_result(ir)

    assert len(ir.anomaly_tiles) == 3
    assert ir.scratch_filter_count == 0
    for t in ir.tiles:
        assert t.scratch_filtered is False
        assert t.scratch_score == pytest.approx(0.2)


def test_filter_no_anomaly_is_noop():
    from scratch_filter import ScratchFilter
    clf = _MockClassifier(fixed_score=0.99, conformal_threshold=0.7)
    sf = ScratchFilter(clf, safety_multiplier=1.0)
    ir = _fake_image_result_with_tiles(0)   # anomaly_tiles = []

    sf.apply_to_image_result(ir)   # should not raise
    assert ir.scratch_filter_count == 0


def test_filter_exception_safety():
    """If classifier raises, tile stays NG (safety default) and other tiles continue."""
    from scratch_filter import ScratchFilter
    # First tile raises; build a classifier that raises once then returns low score
    class _Sometimes(_MockClassifier):
        def __init__(self):
            super().__init__(fixed_score=0.1, conformal_threshold=0.7)
            self._calls = 0
        def predict(self, image):
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("fail once")
            return self._score
    clf = _Sometimes()
    sf = ScratchFilter(clf, safety_multiplier=1.0)
    ir = _fake_image_result_with_tiles(3)

    sf.apply_to_image_result(ir)

    # All 3 kept (first exception, other two score<threshold)
    assert len(ir.anomaly_tiles) == 3
    assert ir.tiles[0].scratch_score == 0.0     # exception path default
    assert ir.tiles[0].scratch_filtered is False
    assert ir.tiles[1].scratch_score == pytest.approx(0.1)
    assert ir.tiles[2].scratch_score == pytest.approx(0.1)


def test_effective_threshold():
    from scratch_filter import ScratchFilter
    clf = _MockClassifier(fixed_score=0.5, conformal_threshold=0.8)
    sf = ScratchFilter(clf, safety_multiplier=1.25)
    # effective = 0.8 * 1.25 = 1.0 → clamped to 0.9999 per spec
    assert sf.effective_threshold == pytest.approx(0.9999)

    sf2 = ScratchFilter(clf, safety_multiplier=1.0)
    assert sf2.effective_threshold == pytest.approx(0.8)
