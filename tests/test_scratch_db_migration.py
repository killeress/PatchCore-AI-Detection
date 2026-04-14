"""Verify schema migration adds scratch columns to existing DB."""
import sqlite3
from pathlib import Path

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
    import pytest
    db = CAPIDatabase(str(tmp_path / "persist.db"))
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
    img_id = db.save_image_result(
        record_id=rec_id,
        image_path="/fake/G0F0000001.jpg",
        image_name="G0F0000001.jpg",
        image_width=512,
        image_height=512,
        is_ng=False,
        tile_count=1,
        anomaly_count=0,
        max_score=0.5,
        inference_time_ms=0.1,
        scratch_filter_count=2,
    )
    db.save_tile_result(
        image_result_id=img_id,
        tile_id=1, x=0, y=0, width=512, height=512,
        score=0.95, is_anomaly=False, is_dust=False,
        heatmap_path="",
        scratch_score=0.88, scratch_filtered=True,
    )

    # Reload and verify
    import sqlite3
    with sqlite3.connect(tmp_path / "persist.db") as c:
        row = c.execute("SELECT scratch_filter_count FROM image_results").fetchone()
        assert row[0] == 2
        row = c.execute(
            "SELECT scratch_score, scratch_filtered FROM tile_results"
        ).fetchone()
        assert row[0] == pytest.approx(0.88)
        assert row[1] == 1
