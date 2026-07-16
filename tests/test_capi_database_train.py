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

from capi_config import CAPIConfig
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
        assert "auto_model_switch_rules" in tables
        assert "auto_model_switch_history" in tables

    def test_training_jobs_columns(self, tmp_path):
        db = _make_db(tmp_path)
        cols = _col_names(db, "training_jobs")
        required = {"id", "job_id", "machine_id", "state", "started_at",
                    "completed_at", "panel_paths", "output_bundle", "error_message",
                    "training_params", "tile_stride"}
        assert required.issubset(cols)

    def test_model_registry_columns(self, tmp_path):
        db = _make_db(tmp_path)
        cols = _col_names(db, "model_registry")
        required = {"id", "machine_id", "bundle_path", "trained_at",
                    "panel_count", "inner_tile_count", "edge_tile_count",
                    "ng_tile_count", "bundle_size_bytes", "is_active", "job_id", "notes"}
        assert required.issubset(cols)

    def test_auto_model_switch_columns(self, tmp_path):
        db = _make_db(tmp_path)
        rule_cols = _col_names(db, "auto_model_switch_rules")
        assert {"id", "series_prefix", "bundle_id", "notes",
                "created_at", "updated_at"}.issubset(rule_cols)

        history_cols = _col_names(db, "auto_model_switch_history")
        assert {"id", "checked_at", "requested_model_id", "series_prefix",
                "previous_bundle_id", "previous_bundle_label",
                "target_bundle_id", "target_bundle_label",
                "action", "status", "message"}.issubset(history_cols)

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


class TestSettingsUsers:
    def test_default_admin_account_exists_and_verifies(self, tmp_path):
        db = _make_db(tmp_path)

        user = db.verify_settings_user("admin", "INXCAPI")

        assert user is not None
        assert user["username"] == "admin"
        assert user["is_admin"] is True
        assert user["can_manage_accounts"] is True
        assert db.verify_settings_user("admin", "wrong") is None

    def test_config_history_records_changed_by(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.update_config_param(
            "sample_threshold",
            0.7,
            "測試修改",
            changed_by="operator01",
        )

        history = db.get_config_change_history("sample_threshold", limit=1)

        assert history[0]["changed_by"] == "operator01"

    def test_config_update_promotes_int_param_to_float(self, tmp_path):
        db = _make_db(tmp_path)
        with _conn(db) as conn:
            conn.execute(
                """INSERT INTO config_params
                   (param_name, param_value, param_type, description)
                   VALUES (?, ?, ?, ?)""",
                ("scratch_safety_multiplier", "1", "int", "刮痕判定安全倍率"),
            )
            conn.commit()

        assert db.update_config_param(
            "scratch_safety_multiplier",
            0.95,
            "調整刮痕安全倍率",
            changed_by="operator01",
        )

        row = db.get_config_param("scratch_safety_multiplier")
        assert row["param_type"] == "float"
        assert row["decoded_value"] == pytest.approx(0.95)

    def test_non_admin_account_crud(self, tmp_path):
        db = _make_db(tmp_path)

        created = db.create_settings_user("operator01", "pw1")
        assert created["is_admin"] is False
        assert db.verify_settings_user("operator01", "pw1") is not None

        updated = db.update_settings_user(created["id"], username="operator02", password="pw2")
        assert updated["username"] == "operator02"
        assert db.verify_settings_user("operator01", "pw1") is None
        assert db.verify_settings_user("operator02", "pw2") is not None

        assert db.delete_settings_user(created["id"]) is True
        assert db.verify_settings_user("operator02", "pw2") is None


class TestConfigParamDefaults:
    def test_image_abnormal_threshold_migration_preserves_manual_values(self, tmp_path):
        db = _make_db(tmp_path)
        with _conn(db) as conn:
            conn.execute(
                """INSERT INTO config_params
                   (param_name, param_value, param_type, description)
                   VALUES (?, ?, ?, ?)""",
                ("image_abnormal_w0f00000_mean_lower", "49", "int", "old desc"),
            )
            conn.execute(
                """INSERT INTO config_params
                   (param_name, param_value, param_type, description)
                   VALUES (?, ?, ?, ?)""",
                ("image_abnormal_w0f00000_mean_upper", "85", "int", "old desc"),
            )
            conn.commit()

        db.init_config_from_yaml(CAPIConfig())

        with _conn(db) as conn:
            lower = conn.execute(
                "SELECT param_value, description FROM config_params WHERE param_name = ?",
                ("image_abnormal_w0f00000_mean_lower",),
            ).fetchone()
            upper = conn.execute(
                "SELECT param_value, description FROM config_params WHERE param_name = ?",
                ("image_abnormal_w0f00000_mean_upper",),
            ).fetchone()
            history_count = conn.execute(
                """SELECT COUNT(*) FROM config_change_history
                   WHERE param_name = ? AND change_reason = ?""",
                (
                    "image_abnormal_w0f00000_mean_lower",
                    "自動更新畫異 polygon mean 預設門檻",
                ),
            ).fetchone()[0]

        assert lower["param_value"] == "70"
        assert "產品區平均亮度" in lower["description"]
        assert upper["param_value"] == "85"
        assert "產品區平均亮度" in upper["description"]
        assert history_count == 1


class TestTrainingJobsCRUD:
    def test_training_jobs_crud(self, tmp_path):
        db = _make_db(tmp_path)
        # create
        job_id = "train_GN160_20260428_153045"
        db.create_training_job(
            job_id=job_id, machine_id="GN160JCEL250S",
            panel_paths=["/p/a", "/p/b"],
        )
        # read
        job = db.get_training_job(job_id)
        assert job["machine_id"] == "GN160JCEL250S"
        assert job["state"] == "preprocess"
        assert job["panel_paths"] == ["/p/a", "/p/b"]
        # update state
        db.update_training_job_state(job_id, "review")
        assert db.get_training_job(job_id)["state"] == "review"
        # update with error
        db.update_training_job_state(job_id, "failed", error_message="OOM")
        job = db.get_training_job(job_id)
        assert job["state"] == "failed"
        assert job["error_message"] == "OOM"

    def test_get_training_job_not_found(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.get_training_job("nonexistent") is None

    def test_update_output_bundle(self, tmp_path):
        db = _make_db(tmp_path)
        job_id = "train_test_bundle"
        db.create_training_job(job_id=job_id, machine_id="M1", panel_paths=[])
        db.update_training_job_state(job_id, "completed", output_bundle="/models/bundle.zip")
        job = db.get_training_job(job_id)
        assert job["state"] == "completed"
        assert job["output_bundle"] == "/models/bundle.zip"
        assert job["completed_at"] is not None

    def test_get_active_training_job(self, tmp_path):
        db = _make_db(tmp_path)
        # No active job initially
        assert db.get_active_training_job() is None
        # Create + check
        db.create_training_job(job_id="j1", machine_id="M", panel_paths=[])
        active = db.get_active_training_job()
        assert active["job_id"] == "j1"
        # After completion → no active
        db.update_training_job_state("j1", "completed")
        assert db.get_active_training_job() is None

    def test_active_job_all_active_states(self, tmp_path):
        """preprocess / review / train 三種 state 都算 active。"""
        db = _make_db(tmp_path)
        for state in ("preprocess", "review", "train"):
            db.create_training_job(job_id=f"j_{state}", machine_id="M", panel_paths=[])
            db.update_training_job_state(f"j_{state}", state)
        # 取最新的 active job（started_at DESC）
        active = db.get_active_training_job()
        assert active is not None
        assert active["state"] in ("preprocess", "review", "train")

    def test_panel_paths_empty_list(self, tmp_path):
        db = _make_db(tmp_path)
        db.create_training_job(job_id="j_empty", machine_id="M", panel_paths=[])
        job = db.get_training_job("j_empty")
        assert job["panel_paths"] == []

    def test_create_returns_rowid(self, tmp_path):
        db = _make_db(tmp_path)
        rowid = db.create_training_job(job_id="j_rowid", machine_id="M", panel_paths=[])
        assert isinstance(rowid, int)
        assert rowid > 0

    def test_training_params_default_none(self, tmp_path):
        """無傳 training_params 時，get_training_job 回傳 None。"""
        db = _make_db(tmp_path)
        db.create_training_job(job_id="j_no_params", machine_id="M", panel_paths=[])
        job = db.get_training_job("j_no_params")
        assert job["training_params"] is None

    def test_training_params_round_trip(self, tmp_path):
        """create 帶 dict → get 反序列化回 dict。"""
        db = _make_db(tmp_path)
        params = {
            "batch_size": 16,
            "coreset_ratio": 0.05,
            "max_epochs": 2,
        }
        db.create_training_job(
            job_id="j_with_params", machine_id="M", panel_paths=[],
            training_params=params,
        )
        job = db.get_training_job("j_with_params")
        assert job["training_params"] == params

    def test_training_data_source_round_trip(self, tmp_path):
        db = _make_db(tmp_path)
        source = {
            "type": "manual_folder",
            "batch_root": "/training/M/batch_01",
            "confirmed_normal": True,
        }
        db.create_training_job(
            job_id="j_manual_source", machine_id="M", panel_paths=["/training/M/batch_01/p1"],
            training_data_source=source,
        )

        assert db.get_training_job("j_manual_source")["training_data_source"] == source

    def test_image_preprocess_pipelines_round_trip(self, tmp_path):
        db = _make_db(tmp_path)
        pipelines = {
            "inner": [{"method": "gaussian", "params": {"kernel_size": 3, "sigma": 1.0}}],
            "edge": [{"method": "bilateral", "params": {"diameter": 5, "sigma_color": 20.0, "sigma_space": 20.0}}],
        }
        db.create_training_job(
            job_id="j_zone_preprocess",
            machine_id="M",
            panel_paths=[],
            preprocess_after_tiling=True,
            image_preprocess_pipelines=pipelines,
        )

        job = db.get_training_job("j_zone_preprocess")
        assert job["preprocess_after_tiling"] is True
        assert job["image_preprocess_pipelines"] == pipelines

    def test_tile_stride_default_and_round_trip(self, tmp_path):
        db = _make_db(tmp_path)
        db.create_training_job(job_id="j_default_stride", machine_id="M", panel_paths=[])
        assert db.get_training_job("j_default_stride")["tile_stride"] == 256

        db.create_training_job(
            job_id="j_custom_stride", machine_id="M", panel_paths=[],
            tile_stride=128,
        )
        assert db.get_training_job("j_custom_stride")["tile_stride"] == 128

    def test_active_job_includes_training_params(self, tmp_path):
        """get_active_training_job 也應該反序列化 training_params。"""
        db = _make_db(tmp_path)
        params = {"batch_size": 4}
        db.create_training_job(
            job_id="j_active", machine_id="M", panel_paths=[],
            training_params=params,
        )
        active = db.get_active_training_job()
        assert active is not None
        assert active["training_params"] == params
        assert active["tile_stride"] == 256


class TestTilePoolCRUD:
    def test_tile_pool_crud(self, tmp_path):
        db = _make_db(tmp_path)
        db.create_training_job(job_id="j1", machine_id="M", panel_paths=[])
        # bulk insert
        tiles = [
            {"lighting": "G0F00000", "zone": "inner", "source": "ok",
             "source_path": "/t/1.png", "thumb_path": "/t/thumb_1.png"},
            {"lighting": "G0F00000", "zone": "edge", "source": "ok",
             "source_path": "/t/2.png", "thumb_path": "/t/thumb_2.png"},
            {"lighting": "G0F00000", "zone": None, "source": "ng",
             "source_path": "/t/n1.png", "thumb_path": "/t/thumb_n1.png"},
        ]
        ids = db.insert_tile_pool("j1", tiles)
        assert len(ids) == 3
        # query all
        all_g0f = db.list_tile_pool("j1", lighting="G0F00000")
        assert len(all_g0f) == 3
        # query by zone
        inner = db.list_tile_pool("j1", lighting="G0F00000", zone="inner")
        assert len(inner) == 1
        # update decision
        db.update_tile_decisions("j1", [ids[0]], "reject")
        rejected = db.list_tile_pool("j1", decision="reject")
        assert len(rejected) == 1
        assert rejected[0]["id"] == ids[0]

    def test_cleanup_tile_pool(self, tmp_path):
        db = _make_db(tmp_path)
        db.create_training_job(job_id="j1", machine_id="M", panel_paths=[])
        db.insert_tile_pool("j1", [{"lighting": "G0F00000", "zone": "inner",
                                    "source": "ok", "source_path": "/t/1.png"}])
        assert len(db.list_tile_pool("j1")) == 1
        db.cleanup_tile_pool("j1")
        assert len(db.list_tile_pool("j1")) == 0


class TestModelRegistryCRUD:
    def test_model_registry_crud(self, tmp_path):
        db = _make_db(tmp_path)
        # register
        bid = db.register_model_bundle({
            "machine_id": "GN160", "bundle_path": "model/GN160-20260428",
            "trained_at": "2026-04-28T15:30:45",
            "panel_count": 5, "inner_tile_count": 2400,
            "edge_tile_count": 900, "ng_tile_count": 150,
            "bundle_size_bytes": 478_000_000, "job_id": "j1",
        })
        # list
        bundles = db.list_model_bundles(machine_id="GN160")
        assert len(bundles) == 1
        assert bundles[0]["bundle_path"] == "model/GN160-20260428"
        assert bundles[0]["is_active"] == 0
        # activate
        db.set_bundle_active(bid, True)
        assert db.list_model_bundles()[0]["is_active"] == 1
        # deactivate others when activating new
        bid2 = db.register_model_bundle({
            "machine_id": "GN160", "bundle_path": "model/GN160-20260501",
            "trained_at": "2026-05-01T10:00:00", "panel_count": 5,
            "inner_tile_count": 2500, "edge_tile_count": 950,
            "ng_tile_count": 150, "bundle_size_bytes": 480_000_000, "job_id": "j2",
        })
        db.deactivate_other_bundles_for_machine("GN160", except_id=bid2)
        db.set_bundle_active(bid2, True)
        bundles = db.list_model_bundles(machine_id="GN160")
        actives = [b for b in bundles if b["is_active"] == 1]
        assert len(actives) == 1
        assert actives[0]["id"] == bid2
        # delete
        db.delete_model_bundle(bid)
        assert len(db.list_model_bundles(machine_id="GN160")) == 1

    def test_update_model_bundle_notes(self, tmp_path):
        db = _make_db(tmp_path)
        bid = db.register_model_bundle({
            "machine_id": "GN160", "bundle_path": "model/GN160-20260428",
            "trained_at": "2026-04-28T15:30:45",
            "panel_count": 5, "inner_tile_count": 2400,
            "edge_tile_count": 900, "ng_tile_count": 150,
            "bundle_size_bytes": 478_000_000, "job_id": "j1",
        })
        notes = "量產用\n低誤報 threshold"

        assert db.update_model_bundle_notes(bid, notes) is True
        assert db.get_model_bundle(bid)["notes"] == notes
        assert db.update_model_bundle_notes(9999, "missing") is False


def test_list_active_training_jobs_returns_all_open(tmp_path):
    """preprocess + review 共存時都應該被列出，completed/failed 不算。"""
    db = _make_db(tmp_path)
    db.create_training_job(job_id="j_pre", machine_id="M", panel_paths=["/a", "/b", "/c"])
    db.create_training_job(job_id="j_rev", machine_id="M", panel_paths=["/a", "/b", "/c"])
    db.update_training_job_state("j_rev", "review")
    db.create_training_job(job_id="j_done", machine_id="M", panel_paths=["/a", "/b", "/c"])
    db.update_training_job_state("j_done", "completed")
    db.create_training_job(job_id="j_failed", machine_id="M", panel_paths=["/a", "/b", "/c"])
    db.update_training_job_state("j_failed", "failed")

    rows = db.list_active_training_jobs()
    ids = sorted(r["job_id"] for r in rows)
    assert ids == ["j_pre", "j_rev"]


def test_list_active_training_jobs_empty(tmp_path):
    db = _make_db(tmp_path)
    assert db.list_active_training_jobs() == []
