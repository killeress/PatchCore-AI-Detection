import os
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from capi_database import CAPIDatabase


def _insert_old_scratch_review(db: CAPIDatabase) -> int:
    conn = sqlite3.connect(str(db.db_path))
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            """
            INSERT INTO inference_records
                (glass_id, model_id, machine_no, request_time, ai_judgment, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("glass-old", "model-a", "machine-a", "2020-01-01 00:00:00", "NG", "2020-01-01 00:00:00"),
        )
        record_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO image_results (record_id, image_path, image_name)
            VALUES (?, ?, ?)
            """,
            (record_id, "/tmp/old.png", "old.png"),
        )
        image_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO tile_results (image_result_id) VALUES (?)",
            (image_id,),
        )
        tile_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO scratch_rescue_review (tile_result_id, is_misrescue, note)
            VALUES (?, 1, ?)
            """,
            (tile_id, "old review"),
        )
        conn.commit()
        return tile_id
    finally:
        conn.close()


def _insert_old_heatmap_record(db: CAPIDatabase, heatmap_dir: Path) -> int:
    created_at = (datetime.now() - timedelta(days=91)).strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(str(db.db_path))
    try:
        conn.execute(
            """
            INSERT INTO inference_records
                (glass_id, model_id, machine_no, request_time, ai_judgment,
                 heatmap_dir, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "glass-old-heatmap",
                "model-a",
                "machine-a",
                created_at,
                "NG",
                str(heatmap_dir),
                created_at,
            ),
        )
        record_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
        return record_id
    finally:
        conn.close()


def test_cleanup_removes_scratch_review_before_expired_tile(tmp_path):
    db = CAPIDatabase(tmp_path / "cleanup.db")
    tile_id = _insert_old_scratch_review(db)

    stats = db.cleanup_old_records(
        ok_retain_days=30,
        ng_retain_days=90,
        tile_retain_days=15,
        vacuum=False,
    )

    assert stats["scratch_rescue_review_deleted"] == 1
    assert stats["tile_results_deleted"] == 1

    with sqlite3.connect(str(db.db_path)) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM scratch_rescue_review WHERE tile_result_id = ?",
            (tile_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM tile_results WHERE id = ?",
            (tile_id,),
        ).fetchone()[0] == 0


def test_cleanup_removes_expired_within_spec_inference_dirs(tmp_path):
    db = CAPIDatabase(tmp_path / "cleanup.db")
    heatmap_root = tmp_path / "heatmaps"
    within_spec_root = heatmap_root / "within_spec_inference"
    old_dir = within_spec_root / "PANEL_OLD_20260101"
    recent_dir = within_spec_root / "PANEL_RECENT_20260714"
    old_dir.mkdir(parents=True)
    recent_dir.mkdir(parents=True)
    (old_dir / "overlay.png").write_bytes(b"old")
    (recent_dir / "overlay.png").write_bytes(b"recent")

    old_timestamp = (datetime.now() - timedelta(days=91)).timestamp()
    os.utime(old_dir, (old_timestamp, old_timestamp))

    stats = db.cleanup_old_records(
        vacuum=False,
        heatmap_retain_days=90,
        heatmap_base_dir=str(heatmap_root),
    )

    assert stats["within_spec_dirs_deleted"] == 1
    assert not old_dir.exists()
    assert recent_dir.exists()


def test_cleanup_retries_failed_heatmap_delete_before_clearing_db_path(
    tmp_path, monkeypatch, caplog
):
    db = CAPIDatabase(tmp_path / "cleanup.db")
    heatmap_dir = tmp_path / "heatmaps" / "20260101" / "glass-old-heatmap"
    heatmap_dir.mkdir(parents=True)
    (heatmap_dir / "overview.png").write_bytes(b"heatmap")
    record_id = _insert_old_heatmap_record(db, heatmap_dir)

    real_rmtree = shutil.rmtree
    failed_once = False

    def fail_first_delete(path):
        nonlocal failed_once
        if Path(path) == heatmap_dir and not failed_once:
            failed_once = True
            raise PermissionError("heatmap is locked")
        return real_rmtree(path)

    monkeypatch.setattr(shutil, "rmtree", fail_first_delete)

    first_stats = db.cleanup_old_records(
        ok_retain_days=365,
        ng_retain_days=365,
        vacuum=False,
        heatmap_retain_days=90,
    )

    assert first_stats["heatmap_dirs_failed"] == 1
    assert heatmap_dir.exists()
    assert "heatmap is locked" in caplog.text
    with sqlite3.connect(str(db.db_path)) as conn:
        assert conn.execute(
            "SELECT heatmap_dir FROM inference_records WHERE id = ?", (record_id,)
        ).fetchone()[0] == str(heatmap_dir)

    second_stats = db.cleanup_old_records(
        ok_retain_days=365,
        ng_retain_days=365,
        vacuum=False,
        heatmap_retain_days=90,
    )

    assert second_stats["heatmap_dirs_deleted"] == 1
    assert second_stats["heatmap_dirs_failed"] == 0
    assert not heatmap_dir.exists()
    with sqlite3.connect(str(db.db_path)) as conn:
        assert conn.execute(
            "SELECT heatmap_dir FROM inference_records WHERE id = ?", (record_id,)
        ).fetchone()[0] == ""


def test_cleanup_keeps_expired_record_as_failed_heatmap_retry_anchor(
    tmp_path, monkeypatch
):
    db = CAPIDatabase(tmp_path / "cleanup.db")
    heatmap_dir = tmp_path / "heatmaps" / "20260101" / "glass-old-heatmap"
    heatmap_dir.mkdir(parents=True)
    (heatmap_dir / "overview.png").write_bytes(b"heatmap")
    record_id = _insert_old_heatmap_record(db, heatmap_dir)

    real_rmtree = shutil.rmtree

    def fail_delete(path):
        if Path(path) == heatmap_dir:
            raise PermissionError("heatmap is locked")
        return real_rmtree(path)

    monkeypatch.setattr(shutil, "rmtree", fail_delete)

    first_stats = db.cleanup_old_records(
        ng_retain_days=90,
        vacuum=False,
        heatmap_retain_days=90,
    )

    assert first_stats["heatmap_dirs_failed"] == 1
    assert first_stats["inference_records_deleted"] == 0
    with sqlite3.connect(str(db.db_path)) as conn:
        assert conn.execute(
            "SELECT heatmap_dir FROM inference_records WHERE id = ?", (record_id,)
        ).fetchone()[0] == str(heatmap_dir)

    monkeypatch.setattr(shutil, "rmtree", real_rmtree)
    second_stats = db.cleanup_old_records(
        ng_retain_days=90,
        vacuum=False,
        heatmap_retain_days=90,
    )

    assert second_stats["heatmap_dirs_deleted"] == 1
    assert second_stats["inference_records_deleted"] == 1
    with sqlite3.connect(str(db.db_path)) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM inference_records WHERE id = ?", (record_id,)
        ).fetchone()[0] == 0


def test_cleanup_removes_only_expired_valid_heatmap_date_roots(tmp_path):
    db = CAPIDatabase(tmp_path / "cleanup.db")
    heatmap_root = tmp_path / "heatmaps"
    expired_date_dir = heatmap_root / (
        datetime.now() - timedelta(days=91)
    ).strftime("%Y%m%d")
    recent_date_dir = heatmap_root / datetime.now().strftime("%Y%m%d")
    invalid_date_dir = heatmap_root / "20240230"
    other_dir = heatmap_root / "operator_notes"
    within_spec_dir = heatmap_root / "within_spec_inference" / "recent-panel"

    for directory in (
        expired_date_dir,
        recent_date_dir,
        invalid_date_dir,
        other_dir,
        within_spec_dir,
    ):
        directory.mkdir(parents=True)
        (directory / "keep.txt").write_text("data", encoding="utf-8")

    stats = db.cleanup_old_records(
        vacuum=False,
        heatmap_retain_days=90,
        heatmap_base_dir=str(heatmap_root),
    )

    assert stats["heatmap_date_dirs_deleted"] == 1
    assert not expired_date_dir.exists()
    assert recent_date_dir.exists()
    assert invalid_date_dir.exists()
    assert other_dir.exists()
    assert within_spec_dir.exists()


def test_cleanup_retries_failed_within_spec_delete(tmp_path, monkeypatch, caplog):
    db = CAPIDatabase(tmp_path / "cleanup.db")
    heatmap_root = tmp_path / "heatmaps"
    old_dir = heatmap_root / "within_spec_inference" / "PANEL_OLD"
    old_dir.mkdir(parents=True)
    (old_dir / "overlay.png").write_bytes(b"old")
    old_timestamp = (datetime.now() - timedelta(days=91)).timestamp()
    os.utime(old_dir, (old_timestamp, old_timestamp))

    real_rmtree = shutil.rmtree
    failed_once = False

    def fail_first_delete(path):
        nonlocal failed_once
        if Path(path) == old_dir and not failed_once:
            failed_once = True
            raise PermissionError("within-spec heatmap is locked")
        return real_rmtree(path)

    monkeypatch.setattr(shutil, "rmtree", fail_first_delete)

    first_stats = db.cleanup_old_records(
        vacuum=False,
        heatmap_retain_days=90,
        heatmap_base_dir=str(heatmap_root),
    )

    assert first_stats["within_spec_dirs_failed"] == 1
    assert old_dir.exists()
    assert "within-spec heatmap is locked" in caplog.text

    second_stats = db.cleanup_old_records(
        vacuum=False,
        heatmap_retain_days=90,
        heatmap_base_dir=str(heatmap_root),
    )

    assert second_stats["within_spec_dirs_deleted"] == 1
    assert second_stats["within_spec_dirs_failed"] == 0
    assert not old_dir.exists()


def test_cleanup_retries_partially_deleted_within_spec_dir_next_round(
    tmp_path, monkeypatch
):
    db = CAPIDatabase(tmp_path / "cleanup.db")
    heatmap_root = tmp_path / "heatmaps"
    old_dir = heatmap_root / "within_spec_inference" / "PANEL_PARTIAL"
    old_dir.mkdir(parents=True)
    first_file = old_dir / "first.png"
    second_file = old_dir / "second.png"
    first_file.write_bytes(b"first")
    second_file.write_bytes(b"second")
    old_timestamp = (datetime.now() - timedelta(days=91)).timestamp()
    os.utime(old_dir, (old_timestamp, old_timestamp))

    real_rmtree = shutil.rmtree
    failed_once = False

    def partially_delete_then_fail(path):
        nonlocal failed_once
        if Path(path) == old_dir and not failed_once:
            failed_once = True
            first_file.unlink()
            raise PermissionError("within-spec delete failed after removing one file")
        return real_rmtree(path)

    monkeypatch.setattr(shutil, "rmtree", partially_delete_then_fail)

    first_stats = db.cleanup_old_records(
        vacuum=False,
        heatmap_retain_days=90,
        heatmap_base_dir=str(heatmap_root),
    )

    assert first_stats["within_spec_dirs_failed"] == 1
    assert old_dir.exists()
    assert not first_file.exists()
    assert second_file.exists()

    second_stats = db.cleanup_old_records(
        vacuum=False,
        heatmap_retain_days=90,
        heatmap_base_dir=str(heatmap_root),
    )

    assert second_stats["within_spec_dirs_deleted"] == 1
    assert not old_dir.exists()
