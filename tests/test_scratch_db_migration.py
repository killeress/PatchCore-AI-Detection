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
