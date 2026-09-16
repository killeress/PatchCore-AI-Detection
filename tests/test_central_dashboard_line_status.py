"""中控看板線體狀態：停線判斷與機種切換提醒的 DB 層測試。"""
import sqlite3

import pytest

from capi_database import CAPIDatabase


def _insert_record(db_path, glass_id, model_id, machine_no, ai_judgment, created_at):
    """以明確 created_at 直接寫入一筆 inference_records（繞過 save_inference_record 的預設時間）。"""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """INSERT INTO inference_records
           (glass_id, model_id, machine_no, resolution_x, resolution_y,
            machine_judgment, ai_judgment, image_dir, total_images, ng_images,
            ng_details, request_time, response_time, processing_seconds, created_at)
           VALUES (?, ?, ?, 0, 0, 'OK', ?, '', 0, 0, '[]', ?, ?, 0.0, ?)""",
        (glass_id, model_id, machine_no, ai_judgment, created_at, created_at, created_at),
    )
    conn.commit()
    conn.close()


def _insert_records(db_path, count, minutes_ago=5):
    conn = sqlite3.connect(str(db_path))
    for i in range(count):
        conn.execute(
            """INSERT INTO inference_records
               (glass_id, model_id, machine_no, resolution_x, resolution_y,
                machine_judgment, ai_judgment, image_dir, total_images, ng_images,
                ng_details, request_time, response_time, processing_seconds, created_at)
               VALUES (?, 'MODEL_A', 'M1', 0, 0, 'OK', 'OK', '', 0, 0, '[]',
                       datetime('now','localtime'), datetime('now','localtime'), 0.0,
                       datetime('now','localtime', ?))""",
            (f"GLASS-{minutes_ago}-{i}", f"-{minutes_ago} minutes"),
        )
    conn.commit()
    conn.close()


# ── 停線：get_recent_request_count ──────────────────────────

def test_recent_request_count_zero_when_empty(tmp_path):
    db = CAPIDatabase(tmp_path / "line.db")
    assert db.get_recent_request_count(120) == 0


def test_recent_request_count_boundary_20_and_21(tmp_path):
    db_path = tmp_path / "line.db"
    db = CAPIDatabase(db_path)
    _insert_records(db_path, 20)
    assert db.get_recent_request_count(120) == 20
    _insert_records(db_path, 1)
    assert db.get_recent_request_count(120) == 21


def test_recent_request_count_includes_err_and_duplicate_glass(tmp_path):
    """口徑：含 ERR、同 glass_id 重送不去重。"""
    db_path = tmp_path / "line.db"
    db = CAPIDatabase(db_path)
    now = sqlite3.connect(str(db_path)).execute(
        "SELECT datetime('now','localtime')"
    ).fetchone()[0]
    _insert_record(db_path, "G1", "MODEL_A", "M1", "ERR:HY", now)
    _insert_record(db_path, "G1", "MODEL_A", "M1", "OK", now)  # 同片重送
    _insert_record(db_path, "G1", "MODEL_A", "M1", "NG@x", now)
    assert db.get_recent_request_count(120) == 3


def test_recent_request_count_excludes_outside_window(tmp_path):
    """滾動窗口與班別無關：119 分鐘前算入、121 分鐘前不算。"""
    db_path = tmp_path / "line.db"
    db = CAPIDatabase(db_path)
    _insert_records(db_path, 2, minutes_ago=119)
    _insert_records(db_path, 5, minutes_ago=121)
    assert db.get_recent_request_count(120) == 2


# ── 機種切換：record / get_active / latest baseline ─────────

def test_record_and_query_active_switch(tmp_path):
    db = CAPIDatabase(tmp_path / "line.db")
    db.record_model_switch_event("M1", "MODEL_A", "MODEL_B")
    events = db.get_active_model_switches(120)
    assert len(events) == 1
    ev = events[0]
    assert ev["machine_no"] == "M1"
    assert ev["previous_model"] == "MODEL_A"
    assert ev["new_model"] == "MODEL_B"
    assert ev["switched_at"] and ev["expires_at"]


def test_active_switches_only_latest_per_machine(tmp_path):
    """提醒期內再次切換：以最新一次為準（重置語意）。"""
    db_path = tmp_path / "line.db"
    db = CAPIDatabase(db_path)
    first_id = db.record_model_switch_event("M1", "MODEL_A", "MODEL_B")
    db.record_model_switch_event("M1", "MODEL_B", "MODEL_C")
    db.record_model_switch_event("M2", "MODEL_X", "MODEL_Y")
    # 把 M1 較早那筆推到窗口外，確認 M1 回傳的是最新一筆（重置語意）
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "UPDATE model_switch_events SET switched_at = datetime('now','localtime','-130 minutes') WHERE id = ?",
        (first_id,),
    )
    conn.commit()
    conn.close()
    events = db.get_active_model_switches(120)
    by_machine = {e["machine_no"]: e for e in events}
    assert by_machine["M1"]["new_model"] == "MODEL_C"
    assert by_machine["M2"]["new_model"] == "MODEL_Y"


def test_active_switches_expire_after_window(tmp_path):
    db_path = tmp_path / "line.db"
    db = CAPIDatabase(db_path)
    event_id = db.record_model_switch_event("M1", "MODEL_A", "MODEL_B")
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "UPDATE model_switch_events SET switched_at = datetime('now','localtime','-121 minutes') WHERE id = ?",
        (event_id,),
    )
    conn.commit()
    conn.close()
    assert db.get_active_model_switches(120) == []


def test_latest_models_by_machine_for_startup_backfill(tmp_path):
    db_path = tmp_path / "line.db"
    db = CAPIDatabase(db_path)
    now = sqlite3.connect(str(db_path)).execute(
        "SELECT datetime('now','localtime')"
    ).fetchone()[0]
    _insert_record(db_path, "G1", "MODEL_A", "M1", "OK", now)
    _insert_record(db_path, "G2", "MODEL_B", "M1", "OK", now)  # M1 最新
    _insert_record(db_path, "G3", "MODEL_C", "M2", "OK", now)
    _insert_record(db_path, "G4", "", "M3", "OK", now)          # 空機種略過
    assert db.get_latest_models_by_machine() == {"M1": "MODEL_B", "M2": "MODEL_C"}
