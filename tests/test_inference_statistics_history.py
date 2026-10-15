"""Historical report counts survive detail retention, retries, and upgrades."""

import sqlite3
from datetime import datetime, timedelta

import pytest

from capi_database import CAPIDatabase


def _save_record(
    db,
    glass_id,
    *,
    ai_judgment="OK",
    machine_judgment="NG",
    request_time="2026-07-02 10:00:00",
    machine_no="M1",
    model_id="MODEL_A",
    **overrides,
):
    values = {
        "glass_id": glass_id,
        "model_id": model_id,
        "machine_no": machine_no,
        "resolution": (100, 100),
        "machine_judgment": machine_judgment,
        "ai_judgment": ai_judgment,
        "image_dir": "",
        "total_images": 1,
        "ng_images": int(ai_judgment.startswith("NG")),
        "ng_details": "[]",
        "request_time": request_time,
        "response_time": request_time,
        "processing_seconds": 0.1,
        "image_results_data": [{"image_name": "image.png", "tiles": [{"tile_id": 1}]}],
    }
    values.update(overrides)
    return db.save_inference_record(**values)


def _set_age(db, record_id, days):
    created_at = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(str(db.db_path)) as conn:
        conn.execute(
            "UPDATE inference_records SET created_at = ? WHERE id = ?",
            (created_at, record_id),
        )


def _rerun(db, record_id, *, ai_judgment="NG", **overrides):
    values = {
        "record_id": record_id,
        "ai_judgment": ai_judgment,
        "total_images": 1,
        "ng_images": int(ai_judgment.startswith("NG")),
        "ng_details": "[]",
        "processing_seconds": 0.2,
    }
    values.update(overrides)
    db.update_record_for_rerun(**values)


def _cleanup(db):
    return db.cleanup_old_records(
        ok_retain_days=30,
        ng_retain_days=90,
        tile_retain_days=15,
        vacuum=False,
    )


def _remove_history_feature(path):
    with sqlite3.connect(str(path)) as conn:
        conn.execute("DROP TRIGGER inference_statistics_insert")
        conn.execute("DROP TRIGGER inference_statistics_update")
        conn.execute("DROP TABLE inference_statistics_history")
        conn.execute("DROP TABLE inference_statistics_metadata")


def test_differential_retention_preserves_every_inference_report_section(tmp_path):
    db = CAPIDatabase(tmp_path / "history.db")
    cases = [
        ("OK", "OK", 45),
        ("NG", "OK-i", 45),
        ("NG", "NG", 45),
        ("OK", "NG:SCRATCH", 120),
        ("NG", "ERR:HY:W0F00000", 120),
        ("NG", "ERR:DIR_NOT_FOUND", 120),
        ("OK", "ERR:INFERENCE_FAILED (RuntimeError: CUDA out of memory)", 120),
        ("NG", "OK", 0),
    ]
    record_ids = []
    for index, (aoi, ai, age) in enumerate(cases):
        first_day = index < 3
        record_id = _save_record(
            db,
            f"PANEL_{index}",
            machine_judgment=aoi,
            ai_judgment=ai,
            request_time="2026-07-02 10:00:00" if first_day else "2026-07-03 10:00:00",
            machine_no="M1" if first_day else "M2",
            model_id="MODEL_A" if first_day else "MODEL_B",
        )
        _set_age(db, record_id, age)
        record_ids.append(record_id)

    before = db.get_inference_stats("2026-07-02", "2026-07-03")
    assert before["summary"] == {
        "total": 8, "aoi_ng": 5, "ai_ng": 2, "ai_revival": 2,
        "err_count": 2, "hy_count": 1,
    }
    assert before["cross_matrix"] == {"ok_ok": 1, "ng_ok": 2, "ok_ng": 1, "ng_ng": 1}
    assert before["by_machine"] == [
        {"machine": "M2", "total": 5, "aoi_ng_rate": 60.0, "ai_ng_rate": 20.0},
        {"machine": "M1", "total": 3, "aoi_ng_rate": 66.67, "ai_ng_rate": 33.33},
    ]
    assert before["by_model"] == [
        {"model": "MODEL_B", "total": 5}, {"model": "MODEL_A", "total": 3},
    ]
    assert before["err_types"] == [
        {"type": "DIR_NOT_FOUND", "count": 1},
        {"type": "INFERENCE_FAILED", "count": 1},
    ]
    assert [(row["date"], row["total"]) for row in before["daily_trend"]] == [
        ("2026-07-02", 3), ("2026-07-03", 5),
    ]

    cleaned = _cleanup(db)

    assert cleaned["inference_records_deleted"] == 6
    with sqlite3.connect(str(db.db_path)) as conn:
        retained_ids = {row[0] for row in conn.execute("SELECT id FROM inference_records")}
        assert retained_ids == {record_ids[2], record_ids[7]}
        assert conn.execute("SELECT COUNT(*) FROM image_results").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM inference_statistics_history").fetchone()[0] == 8
    assert db.get_inference_stats("2026-07-02", "2026-07-03") == before
    assert _cleanup(db)["inference_records_deleted"] == 0
    assert db.get_inference_stats("2026-07-02", "2026-07-03") == before


def test_rerun_reclassifies_history_without_counting_another_inference(tmp_path):
    db = CAPIDatabase(tmp_path / "rerun.db")
    record_id = _save_record(db, "RERUN", ai_judgment="ERR:MODEL_NOT_LOADED")
    _set_age(db, record_id, 120)
    assert db.get_inference_stats()["summary"]["err_count"] == 1

    for _ in range(2):
        _rerun(db, record_id, ai_judgment="OK-i")
        stats = db.get_inference_stats("2026-07-02", "2026-07-02")
        assert stats["summary"] == {
            "total": 1, "aoi_ng": 1, "ai_ng": 0, "ai_revival": 1,
            "err_count": 0, "hy_count": 0,
        }
        assert stats["cross_matrix"] == {"ok_ok": 0, "ng_ok": 1, "ok_ng": 0, "ng_ng": 0}
        assert stats["err_types"] == []

    _rerun(db, record_id, ai_judgment="NG:SCRATCH", machine_judgment="OK")
    before_cleanup = db.get_inference_stats("2026-07-02", "2026-07-02")
    assert before_cleanup["summary"] == {
        "total": 1, "aoi_ng": 0, "ai_ng": 1, "ai_revival": 0,
        "err_count": 0, "hy_count": 0,
    }
    assert before_cleanup["cross_matrix"]["ok_ng"] == 1
    assert before_cleanup["by_machine"][0]["ai_ng_rate"] == 100.0
    assert _cleanup(db)["inference_records_deleted"] == 1
    assert db.get_inference_stats("2026-07-02", "2026-07-02") == before_cleanup


def test_reopen_after_all_details_deleted_keeps_history_and_new_records_distinct(tmp_path):
    path = tmp_path / "reopen.db"
    db = CAPIDatabase(path)
    first_id = _save_record(db, "OLD", ai_judgment="OK")
    _set_age(db, first_id, 120)
    initial = db.get_inference_stats()
    assert initial["history"]["initialized_at"]
    assert _cleanup(db)["inference_records_deleted"] == 1

    for _ in range(2):
        db = CAPIDatabase(path)
        assert db.get_inference_stats() == initial

    second_instance = CAPIDatabase(path)
    second_id = _save_record(second_instance, "NEW", ai_judgment="NG")
    assert second_id > first_id
    stats = db.get_inference_stats()
    assert stats["history"] == initial["history"]
    assert stats["summary"]["total"] == 2
    assert stats["summary"]["ai_ng"] == 1
    assert stats["summary"]["ai_revival"] == 1
    assert CAPIDatabase(path).get_inference_stats() == stats


def test_legacy_upgrade_backfills_retained_records_once_then_keeps_deleted_history(tmp_path):
    path = tmp_path / "legacy.db"
    db = CAPIDatabase(path)
    ok_id = _save_record(db, "LEGACY_OK", ai_judgment="OK-i")
    ng_id = _save_record(db, "LEGACY_NG", ai_judgment="NG")
    expected = db.get_inference_stats()
    _set_age(db, ok_id, 120)
    _set_age(db, ng_id, 120)

    # Emulate an existing installation before the history feature was deployed.
    _remove_history_feature(path)

    upgraded = CAPIDatabase(path)
    migrated = upgraded.get_inference_stats()
    assert migrated["history"]["initialized_at"]
    for key in ("summary", "daily_trend", "by_machine", "by_model", "err_types", "cross_matrix"):
        assert migrated[key] == expected[key]
    assert CAPIDatabase(path).get_inference_stats() == migrated
    assert _cleanup(upgraded)["inference_records_deleted"] == 2
    assert CAPIDatabase(path).get_inference_stats() == migrated


def test_failed_legacy_migration_rolls_back_and_can_retry(tmp_path, monkeypatch):
    path = tmp_path / "migration_retry.db"
    db = CAPIDatabase(path)
    _save_record(db, "LEGACY", ai_judgment="OK-i")
    _remove_history_feature(path)
    initialize_history = CAPIDatabase._init_inference_statistics_history

    def fail_after_backfill(conn):
        initialize_history(conn)
        raise RuntimeError("simulated startup failure before commit")

    with monkeypatch.context() as patch:
        patch.setattr(
            CAPIDatabase, "_init_inference_statistics_history", staticmethod(fail_after_backfill)
        )
        with pytest.raises(RuntimeError, match="simulated startup failure"):
            CAPIDatabase(path)

    with sqlite3.connect(str(path)) as conn:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'inference_statistics_%'"
        ).fetchall() == []
        assert conn.execute("SELECT COUNT(*) FROM inference_records").fetchone()[0] == 1

    recovered = CAPIDatabase(path).get_inference_stats()
    assert recovered["summary"]["total"] == 1
    assert recovered["summary"]["ai_revival"] == 1
    assert recovered["history"]["initialized_at"]
    assert CAPIDatabase(path).get_inference_stats() == recovered


def test_factory_day_boundaries_and_judgment_categories_survive_detail_deletion(tmp_path):
    db = CAPIDatabase(tmp_path / "factory_day.db")
    cases = [
        ("2026-07-02 07:29:59", "NG", "NG"),
        ("2026-07-02T07:30:00", "OK-i", "NG"),
        ("2026-07-02 23:59:59", "ERR:HY:G0F00000", "OK"),
        ("2026-07-03T00:00:00", "ERR:DIR_NOT_FOUND", "NG"),
        ("2026-07-03 07:29:59", "NG:SCRATCH", "NG"),
        ("2026-07-03T07:30:00", "OK", "OK"),
    ]
    for index, (request_time, ai, aoi) in enumerate(cases):
        record_id = _save_record(
            db, f"BOUNDARY_{index}", request_time=request_time,
            ai_judgment=ai, machine_judgment=aoi,
        )
        _set_age(db, record_id, 120)

    before = db.get_inference_stats("2026-07-02", "2026-07-02")
    assert before["summary"] == {
        "total": 4, "aoi_ng": 3, "ai_ng": 1, "ai_revival": 1,
        "err_count": 1, "hy_count": 1,
    }
    assert len(before["daily_trend"]) == 1
    assert before["daily_trend"][0]["date"] == "2026-07-02"
    assert before["daily_trend"][0]["total"] == 4
    assert db.get_inference_stats("2026-07-01", "2026-07-01")["summary"]["total"] == 1
    assert db.get_inference_stats("2026-07-03", "2026-07-03")["summary"]["total"] == 1
    assert _cleanup(db)["inference_records_deleted"] == 6
    assert db.get_inference_stats("2026-07-02", "2026-07-02") == before
    assert db.get_inference_stats()["summary"]["total"] == 6


def test_direct_database_updates_move_history_dimensions_and_error_type(tmp_path):
    db = CAPIDatabase(tmp_path / "direct_update.db")
    with sqlite3.connect(str(db.db_path)) as conn:
        cursor = conn.execute(
            """INSERT INTO inference_records
               (glass_id, model_id, machine_no, request_time, machine_judgment, ai_judgment)
               VALUES ('SQL', 'OLD_MODEL', 'OLD_MACHINE', '2026-07-02 10:00:00', 'NG', 'NG')"""
        )
        record_id = cursor.lastrowid
    assert db.get_inference_stats("2026-07-02", "2026-07-02")["summary"]["total"] == 1

    with sqlite3.connect(str(db.db_path)) as conn:
        conn.execute(
            """UPDATE inference_records SET
               request_time = '2026-07-03T10:00:00', model_id = 'NEW_MODEL',
               machine_no = 'NEW_MACHINE', machine_judgment = 'OK', ai_judgment = 'ERR',
               error_message = 'INFERENCE_FAILED (RuntimeError: CUDA out of memory)'
               WHERE id = ?""",
            (record_id,),
        )
    assert db.get_inference_stats("2026-07-02", "2026-07-02")["summary"]["total"] == 0
    updated = db.get_inference_stats("2026-07-03", "2026-07-03")
    assert updated["summary"] == {
        "total": 1, "aoi_ng": 0, "ai_ng": 0, "ai_revival": 0,
        "err_count": 1, "hy_count": 0,
    }
    assert updated["by_machine"] == [
        {"machine": "NEW_MACHINE", "total": 1, "aoi_ng_rate": 0.0, "ai_ng_rate": 0.0},
    ]
    assert updated["by_model"] == [{"model": "NEW_MODEL", "total": 1}]
    assert updated["err_types"] == [{"type": "INFERENCE_FAILED", "count": 1}]

    with sqlite3.connect(str(db.db_path)) as conn:
        conn.execute(
            "UPDATE inference_records SET error_message = 'DIR_NOT_FOUND: image missing' WHERE id = ?",
            (record_id,),
        )
    assert db.get_inference_stats()["err_types"] == [{"type": "DIR_NOT_FOUND", "count": 1}]
    assert db.get_inference_stats()["summary"]["total"] == 1


@pytest.mark.parametrize("operation", ["save", "rerun"])
def test_failed_child_save_rolls_back_history_in_same_transaction(tmp_path, operation):
    db = CAPIDatabase(tmp_path / "rollback.db")
    record_id = _save_record(db, "EXISTING", ai_judgment="OK")
    before = db.get_inference_stats()
    invalid_images = [{"image_name": "bad.png", "max_score": {"invalid": "sqlite value"}}]

    with pytest.raises(sqlite3.Error):
        if operation == "save":
            _save_record(db, "FAILED", ai_judgment="NG", image_results_data=invalid_images)
        else:
            _rerun(db, record_id, ai_judgment="NG", image_results_data=invalid_images)

    assert db.get_inference_stats() == before
    with sqlite3.connect(str(db.db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM inference_records").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM inference_statistics_history").fetchone()[0] == 1
        assert conn.execute("SELECT ai_judgment FROM inference_records").fetchone()[0] == "OK"
        assert conn.execute("SELECT COUNT(*) FROM image_results").fetchone()[0] == 1
