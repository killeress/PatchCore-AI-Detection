"""Verify schema migration adds scratch columns to existing DB."""
import sqlite3
from pathlib import Path

import pytest

from capi_database import CAPIDatabase


def _get_columns(db_path: Path, table: str) -> set[str]:
    with sqlite3.connect(db_path) as c:
        return {row[1] for row in c.execute(f"PRAGMA table_info({table})").fetchall()}


def test_fresh_db_has_scratch_columns(tmp_path):
    db_path = tmp_path / "fresh.db"
    CAPIDatabase(str(db_path))  # triggers init

    tile_cols = _get_columns(db_path, "tile_results")
    assert "scratch_score" in tile_cols
    assert "scratch_filtered" in tile_cols

    image_cols = _get_columns(db_path, "image_results")
    assert "scratch_filter_count" in image_cols


def test_existing_db_migrated(tmp_path):
    """Verify migration adds scratch columns to database with pre-existing schema."""
    db_path = tmp_path / "old.db"

    # Create a database with a partial schema (simulating an older version)
    # by directly inserting minimal rows into fresh db, then deleting the scratch columns
    db = CAPIDatabase(str(db_path))

    # Connect directly and drop the new scratch columns to simulate old schema
    with sqlite3.connect(db_path) as c:
        c.execute("ALTER TABLE tile_results RENAME TO tile_results_backup")
        c.execute("""
            CREATE TABLE tile_results AS
            SELECT id, image_result_id, tile_id, x, y, width, height, score, is_anomaly,
                   is_dust, dust_iou, is_bomb, bomb_code, peak_x, peak_y, is_exclude_zone,
                   is_aoi_coord, aoi_defect_code, aoi_product_x, aoi_product_y
            FROM tile_results_backup
        """)
        c.execute("DROP TABLE tile_results_backup")

        c.execute("ALTER TABLE image_results RENAME TO image_results_backup")
        c.execute("""
            CREATE TABLE image_results AS
            SELECT id, record_id, image_path, image_name, image_width, image_height,
                   otsu_bounds, tile_count, excluded_tiles, anomaly_count, max_score,
                   is_ng, is_dust_only, is_bomb, inference_time_ms, heatmap_path
            FROM image_results_backup
        """)
        c.execute("DROP TABLE image_results_backup")
        c.commit()

    # Verify columns are gone before migration
    with sqlite3.connect(db_path) as c:
        tile_cols_before = {row[1] for row in c.execute("PRAGMA table_info(tile_results)").fetchall()}
        assert "scratch_score" not in tile_cols_before
        assert "scratch_filtered" not in tile_cols_before

    # Now re-open via CAPIDatabase — should detect and run migration
    db2 = CAPIDatabase(str(db_path))

    tile_cols = _get_columns(db_path, "tile_results")
    assert "scratch_score" in tile_cols
    assert "scratch_filtered" in tile_cols
    image_cols = _get_columns(db_path, "image_results")
    assert "scratch_filter_count" in image_cols


def test_scratch_fields_persist(tmp_path):
    """Verify scratch fields are persisted via save_inference_record's image_results_data dict API."""
    db = CAPIDatabase(str(tmp_path / "persist.db"))
    image_results_data = [{
        "image_path": "/fake/G0F0000001.jpg",
        "image_name": "G0F0000001.jpg",
        "image_width": 512, "image_height": 512,
        "otsu_bounds": "",
        "tile_count": 1, "excluded_tiles": 0, "anomaly_count": 0,
        "max_score": 0.5,
        "is_ng": 0, "is_dust_only": 0, "is_bomb": 0,
        "inference_time_ms": 100.0,
        "heatmap_path": "",
        "scratch_filter_count": 2,           # NEW
        "tiles": [{
            "tile_id": 1, "x": 0, "y": 0, "width": 512, "height": 512,
            "score": 0.95, "is_anomaly": 0, "is_dust": 0, "dust_iou": 0.0,
            "is_bomb": 0, "bomb_code": "",
            "peak_x": -1, "peak_y": -1,
            "heatmap_path": "",
            "is_exclude_zone": 0, "is_aoi_coord": 0,
            "aoi_defect_code": "", "aoi_product_x": -1, "aoi_product_y": -1,
            "scratch_score": 0.88,           # NEW
            "scratch_filtered": True,        # NEW
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
    with sqlite3.connect(tmp_path / "persist.db") as c:
        row = c.execute("SELECT scratch_filter_count FROM image_results").fetchone()
        assert row[0] == 2
        row = c.execute(
            "SELECT scratch_score, scratch_filtered FROM tile_results"
        ).fetchone()
        assert row[0] == pytest.approx(0.88)
        assert row[1] == 1


def test_scratch_fields_persist_on_rerun(tmp_path):
    """update_record_for_rerun must also write scratch fields."""
    db = CAPIDatabase(str(tmp_path / "rerun.db"))
    # Create initial record (no image_results_data)
    rec_id = db.save_inference_record(
        glass_id="G1", model_id="M1", machine_no="1",
        resolution=(1920, 1080),
        machine_judgment="NG", ai_judgment="OK",
        image_dir="/fake",
        total_images=1, ng_images=0, ng_details="",
        request_time="2026-04-14T10:00:00",
        response_time="2026-04-14T10:00:05",
        processing_seconds=5.0,
    )
    # Now rerun with scratch fields
    db.update_record_for_rerun(
        record_id=rec_id,
        ai_judgment="OK",
        total_images=1, ng_images=0, ng_details="",
        processing_seconds=3.0,
        image_results_data=[{
            "image_path": "/fake/x.jpg", "image_name": "x.jpg",
            "image_width": 512, "image_height": 512,
            "otsu_bounds": "",
            "tile_count": 1, "excluded_tiles": 0, "anomaly_count": 0,
            "max_score": 0.9, "is_ng": 0, "is_dust_only": 0, "is_bomb": 0,
            "inference_time_ms": 50.0, "heatmap_path": "",
            "scratch_filter_count": 5,
            "tiles": [{
                "tile_id": 1, "x": 0, "y": 0, "width": 512, "height": 512,
                "score": 0.9, "is_anomaly": 0, "is_dust": 0, "dust_iou": 0.0,
                "is_bomb": 0, "bomb_code": "",
                "peak_x": -1, "peak_y": -1, "heatmap_path": "",
                "is_exclude_zone": 0, "is_aoi_coord": 0,
                "aoi_defect_code": "", "aoi_product_x": -1, "aoi_product_y": -1,
                "scratch_score": 0.77, "scratch_filtered": True,
            }],
        }],
    )
    with sqlite3.connect(tmp_path / "rerun.db") as c:
        row = c.execute("SELECT scratch_filter_count FROM image_results").fetchone()
        assert row[0] == 5
        row = c.execute(
            "SELECT scratch_score, scratch_filtered FROM tile_results"
        ).fetchone()
        assert row[0] == pytest.approx(0.77)
        assert row[1] == 1
