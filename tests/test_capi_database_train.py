"""
Tests for training-related schema additions to CAPIDatabase:
  - training_jobs
  - model_registry
  - training_tile_pool
"""
import sqlite3
import tempfile
from pathlib import Path

import pytest

from capi_database import CAPIDatabase


def _make_db(tmp_path) -> CAPIDatabase:
    return CAPIDatabase(Path(tmp_path) / "test.db")


def _conn(db: CAPIDatabase) -> sqlite3.Connection:
    """回傳一個直連 db 檔案的 connection，供測試查詢用。"""
    conn = sqlite3.connect(str(db.db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _table_names(db: CAPIDatabase) -> set:
    with _conn(db) as conn:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {row[0] for row in cur.fetchall()}


def _col_names(db: CAPIDatabase, table: str) -> set:
    with _conn(db) as conn:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return {row[1] for row in cur.fetchall()}


def _index_names(db: CAPIDatabase) -> set:
    with _conn(db) as conn:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        return {row[0] for row in cur.fetchall()}


class TestTrainingSchema:
    def test_three_tables_exist(self, tmp_path):
        db = _make_db(tmp_path)
        tables = _table_names(db)
        assert "training_jobs" in tables
        assert "model_registry" in tables
        assert "training_tile_pool" in tables

    def test_training_jobs_columns(self, tmp_path):
        db = _make_db(tmp_path)
        cols = _col_names(db, "training_jobs")
        required = {"id", "job_id", "machine_id", "state", "started_at",
                    "completed_at", "panel_paths", "output_bundle", "error_message"}
        assert required.issubset(cols)

    def test_model_registry_columns(self, tmp_path):
        db = _make_db(tmp_path)
        cols = _col_names(db, "model_registry")
        required = {"id", "machine_id", "bundle_path", "trained_at",
                    "panel_count", "inner_tile_count", "edge_tile_count",
                    "ng_tile_count", "bundle_size_bytes", "is_active", "job_id", "notes"}
        assert required.issubset(cols)

    def test_training_tile_pool_columns(self, tmp_path):
        db = _make_db(tmp_path)
        cols = _col_names(db, "training_tile_pool")
        required = {"id", "job_id", "lighting", "zone", "source",
                    "source_path", "thumb_path", "decision"}
        assert required.issubset(cols)

    def test_tile_pool_index_exists(self, tmp_path):
        db = _make_db(tmp_path)
        assert "idx_tile_pool_job" in _index_names(db)

    def test_tile_pool_decision_default_accept(self, tmp_path):
        db = _make_db(tmp_path)
        with _conn(db) as conn:
            conn.execute(
                "INSERT INTO training_tile_pool (job_id, lighting, source, source_path) "
                "VALUES ('j1', 'G0F', 'ok', '/some/path.png')"
            )
            conn.commit()
            cur = conn.execute(
                "SELECT decision, zone FROM training_tile_pool WHERE job_id='j1'"
            )
            row = cur.fetchone()
        assert row["decision"] == "accept", "decision default should be 'accept'"
        assert row["zone"] is None, "zone should allow NULL (for NG tiles)"

    def test_existing_tables_untouched(self, tmp_path):
        db = _make_db(tmp_path)
        tables = _table_names(db)
        existing = {
            "inference_records", "image_results", "tile_results",
            "edge_defect_results", "ric_import_batches", "ric_records",
            "config_params", "config_change_history",
        }
        assert existing.issubset(tables)
