"""Verify aggregate_judgment respects scratch_filtered flag (C1 fix)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from capi_inference import TileInfo, ImageResult
from capi_server import aggregate_judgment


def _ir_with_mixed_tiles() -> ImageResult:
    """Build ImageResult with 2 tiles: one scratch-flipped, one real NG."""
    t1 = TileInfo(tile_id=1, x=0, y=0, width=512, height=512,
                  image=np.zeros((512, 512, 3), dtype=np.uint8))
    t1.scratch_filtered = True      # Should be skipped
    t1.scratch_score = 0.92

    t2 = TileInfo(tile_id=2, x=512, y=0, width=512, height=512,
                  image=np.zeros((512, 512, 3), dtype=np.uint8))
    # t2 is a real NG (no scratch_filtered)

    ir = ImageResult(
        image_path=Path("/fake/img.jpg"),
        image_size=(1024, 512),
        otsu_bounds=(0, 0, 1024, 512),
        exclusion_regions=[],
        tiles=[t1, t2],
        excluded_tile_count=0, processed_tile_count=2, processing_time=0.0,
    )
    ir.anomaly_tiles = [(t1, 0.9, None), (t2, 0.85, None)]
    return ir


def test_aggregate_judgment_skips_scratch_filtered_tiles():
    """Tile with scratch_filtered=True must not contribute to NG judgment."""
    ir = _ir_with_mixed_tiles()
    judgment, details_json = aggregate_judgment([ir])

    # NG from tile 2 (real NG); tile 1 (scratch_filtered) skipped
    assert judgment == "NG"
    import json
    details = json.loads(details_json)
    assert len(details) == 1
    assert details[0]["tile_id"] == 2


def test_aggregate_judgment_all_scratch_becomes_ok():
    """If ALL anomaly_tiles are scratch_filtered, judgment should be OK."""
    t1 = TileInfo(tile_id=1, x=0, y=0, width=512, height=512,
                  image=np.zeros((512, 512, 3), dtype=np.uint8))
    t1.scratch_filtered = True
    ir = ImageResult(
        image_path=Path("/fake/img.jpg"),
        image_size=(1024, 512),
        otsu_bounds=(0, 0, 1024, 512),
        exclusion_regions=[],
        tiles=[t1],
        excluded_tile_count=0, processed_tile_count=1, processing_time=0.0,
    )
    ir.anomaly_tiles = [(t1, 0.9, None)]

    judgment, details_json = aggregate_judgment([ir])
    assert judgment == "OK"
    assert details_json == "[]"
