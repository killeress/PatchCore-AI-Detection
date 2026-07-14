import os
import sqlite3
from datetime import datetime, timedelta

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
