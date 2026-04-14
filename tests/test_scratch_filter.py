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
