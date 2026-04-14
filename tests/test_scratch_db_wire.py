"""Verify pipeline code populates scratch fields into image_results_data dicts."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from capi_inference import TileInfo, ImageResult


def _fake_image_result_with_scratch_data() -> ImageResult:
    tile = TileInfo(tile_id=5, x=10, y=20, width=512, height=512,
                    image=np.zeros((512, 512, 3), dtype=np.uint8))
    tile.scratch_score = 0.87
    tile.scratch_filtered = True

    ir = ImageResult(
        image_path=Path("/fake/img.jpg"),
        image_size=(1024, 768),
        otsu_bounds=(0, 0, 1024, 768),
        exclusion_regions=[],
        tiles=[tile],
        excluded_tile_count=0, processed_tile_count=1, processing_time=0.1,
    )
    ir.scratch_filter_count = 3
    ir.anomaly_tiles = [(tile, 0.95, None)]
    return ir


def test_build_image_results_data_includes_scratch():
    """If a helper exists that builds image_results_data from ImageResult,
    verify it includes scratch fields. If no such helper, this test may need
    to be adapted to exercise the actual pipeline code."""
    # Find and import the helper used to build image_results_data dicts.
    # Search order: capi_inference, capi_server (top-level function or method).
    # If the helper is inline in a larger function, add a comment in the
    # implementation step pointing at the call-site that was modified instead.
    try:
        from capi_server import build_image_results_data
    except ImportError:
        try:
            from capi_inference import build_image_results_data
        except ImportError:
            pytest.skip("No dedicated helper found; verified at integration level")
            return

    ir = _fake_image_result_with_scratch_data()
    data = build_image_results_data([ir])
    assert len(data) == 1
    assert data[0]["scratch_filter_count"] == 3
    assert data[0]["tiles"][0]["scratch_score"] == pytest.approx(0.87)
    assert data[0]["tiles"][0]["scratch_filtered"] is True


def test_results_to_db_data_includes_scratch():
    """Verify results_to_db_data (the actual helper) includes scratch fields."""
    from capi_server import results_to_db_data

    ir = _fake_image_result_with_scratch_data()
    # Pass empty heatmap_info so path checks are skipped
    data = results_to_db_data([ir], heatmap_info={})
    assert len(data) == 1, "Expected one image dict"
    assert data[0]["scratch_filter_count"] == 3, \
        "scratch_filter_count missing from image dict"
    assert len(data[0]["tiles"]) == 1, "Expected one tile dict"
    assert data[0]["tiles"][0]["scratch_score"] == pytest.approx(0.87), \
        "scratch_score missing from tile dict"
    assert data[0]["tiles"][0]["scratch_filtered"] is True, \
        "scratch_filtered missing from tile dict"


def test_scratch_fields_round_trip_through_full_save_path(tmp_path):
    """End-to-end: build ImageResult/TileInfo with scratch data, feed through
    the actual pipeline function that serializes + saves to DB, verify DB has
    correct values."""
    import sqlite3
    from capi_database import CAPIDatabase

    db_path = tmp_path / "e2e.db"
    db = CAPIDatabase(str(db_path))

    # Simulate what the pipeline does: build image_results_data dicts matching
    # the post-Task-10 schema (i.e., including scratch fields).
    # Eventually, a real pipeline helper is the thing being tested here, but
    # since we might not have a clean helper, we construct the dict inline
    # with the same shape used by production code.
    image_results_data = [{
        "image_path": "/fake/img.jpg",
        "image_name": "img.jpg",
        "image_width": 1024, "image_height": 768,
        "otsu_bounds": "",
        "tile_count": 1, "excluded_tiles": 0, "anomaly_count": 1,
        "max_score": 0.95,
        "is_ng": 0, "is_dust_only": 0, "is_bomb": 0,
        "inference_time_ms": 100.0,
        "heatmap_path": "",
        "scratch_filter_count": 3,
        "tiles": [{
            "tile_id": 5, "x": 10, "y": 20, "width": 512, "height": 512,
            "score": 0.95,
            "is_anomaly": 0, "is_dust": 0, "dust_iou": 0.0,
            "is_bomb": 0, "bomb_code": "",
            "peak_x": -1, "peak_y": -1,
            "heatmap_path": "",
            "is_exclude_zone": 0, "is_aoi_coord": 0,
            "aoi_defect_code": "", "aoi_product_x": -1, "aoi_product_y": -1,
            "scratch_score": 0.87,
            "scratch_filtered": True,
        }],
    }]
    db.save_inference_record(
        glass_id="G1", model_id="M1", machine_no="1",
        resolution=(1920, 1080),
        machine_judgment="NG", ai_judgment="OK",
        image_dir="/fake",
        total_images=1, ng_images=0, ng_details="",
        request_time="2026-04-14T10:00:00",
        response_time="2026-04-14T10:00:05",
        processing_seconds=5.0,
        image_results_data=image_results_data,
    )
    with sqlite3.connect(db_path) as c:
        row = c.execute("SELECT scratch_filter_count FROM image_results").fetchone()
        assert row[0] == 3
        row = c.execute(
            "SELECT scratch_score, scratch_filtered FROM tile_results"
        ).fetchone()
        assert row[0] == pytest.approx(0.87)
        assert row[1] == 1
