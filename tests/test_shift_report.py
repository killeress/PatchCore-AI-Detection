"""歷史班報：指定日期＋班別的線體端統計（DB 層與 API 層）。"""
import io
import json
import sqlite3

import pytest

from capi_database import CAPIDatabase
from capi_web import CAPIWebHandler


def _insert_record(
    db_path,
    glass_id,
    ai_judgment,
    created_at,
    machine_judgment="OK",
):
    """以明確 created_at / machine_judgment 直接寫入一筆 inference_records。"""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """INSERT INTO inference_records
           (glass_id, model_id, machine_no, resolution_x, resolution_y,
            machine_judgment, ai_judgment, image_dir, total_images, ng_images,
            ng_details, request_time, response_time, processing_seconds, created_at)
           VALUES (?, 'MODEL_A', 'M1', 0, 0, ?, ?, '', 0, 0, '[]', ?, ?, 0.0, ?)""",
        (glass_id, machine_judgment, ai_judgment, created_at, created_at, created_at),
    )
    conn.commit()
    conn.close()


# ── get_shift_statistics_for：白班窗口 07:30 ~ 19:30 ─────────

def test_shift_report_day_shift_window_boundaries(tmp_path):
    """白班：含 07:30 與 19:29，排除 07:29 與 19:30。"""
    db_path = tmp_path / "line.db"
    db = CAPIDatabase(db_path)
    day = "2026-01-20"
    _insert_record(db_path, "BEFORE", "OK", f"{day} 07:29:59")
    _insert_record(db_path, "START", "OK", f"{day} 07:30:00")
    _insert_record(db_path, "END", "NG", f"{day} 19:29:59")
    _insert_record(db_path, "AFTER", "OK", f"{day} 19:30:00")

    stats = db.get_shift_statistics_for(day, "day")

    assert stats["total"] == 2
    assert stats["ok_count"] == 1
    assert stats["ng_count"] == 1


def test_shift_report_night_shift_crosses_midnight(tmp_path):
    """晚班（夜班）：當日 19:30 ~ 隔日 07:30，跨日歸屬起班日。"""
    db_path = tmp_path / "line.db"
    db = CAPIDatabase(db_path)
    day = "2026-01-20"
    _insert_record(db_path, "BEFORE", "OK", f"{day} 19:29:59")
    _insert_record(db_path, "START", "OK", f"{day} 19:30:00")
    _insert_record(db_path, "MID", "NG", "2026-01-21 00:10:00")
    _insert_record(db_path, "END", "OK", "2026-01-21 07:29:59")
    _insert_record(db_path, "AFTER", "OK", "2026-01-21 07:30:00")

    stats = db.get_shift_statistics_for(day, "night")

    assert stats["total"] == 3
    assert stats["ok_count"] == 2
    assert stats["ng_count"] == 1
    assert stats["shift_name"] == "夜班"
    assert stats["time_range"] == "01/20 19:30 ~ 01/21 07:30"


def test_shift_report_counts_aoi_ng_and_err(tmp_path):
    """AOI 排片率分子 = machine_judgment 非空且非 OK；ERR 計入投入。"""
    db_path = tmp_path / "line.db"
    db = CAPIDatabase(db_path)
    day = "2026-01-20"
    _insert_record(db_path, "G1", "OK", f"{day} 10:00:00", machine_judgment="OK")
    _insert_record(db_path, "G2", "OK", f"{day} 10:01:00", machine_judgment="NG")
    _insert_record(db_path, "G3", "NG", f"{day} 10:02:00", machine_judgment="NG")
    _insert_record(db_path, "G4", "ERR:timeout", f"{day} 10:03:00", machine_judgment="")

    stats = db.get_shift_statistics_for(day, "day")

    assert stats["total"] == 4
    assert stats["aoi_ng_count"] == 2
    assert stats["ng_count"] == 1
    assert stats["err_count"] == 1


def test_shift_report_empty_shift_returns_zeros(tmp_path):
    db = CAPIDatabase(tmp_path / "line.db")
    stats = db.get_shift_statistics_for("2026-01-20", "day")
    assert stats["total"] == 0
    assert (stats["ok_count"] or 0) == 0
    assert (stats["ng_count"] or 0) == 0
    assert stats["shift_name"] == "白班"


def test_shift_report_rejects_invalid_shift(tmp_path):
    db = CAPIDatabase(tmp_path / "line.db")
    with pytest.raises(ValueError):
        db.get_shift_statistics_for("2026-01-20", "evening")


def test_shift_report_rejects_invalid_date(tmp_path):
    db = CAPIDatabase(tmp_path / "line.db")
    with pytest.raises(ValueError):
        db.get_shift_statistics_for("2026/01/20", "day")


# ── /api/shift_report ────────────────────────────────────────

def _make_handler(db, responses):
    handler = object.__new__(CAPIWebHandler)
    handler.db = db
    handler.headers = {}
    handler._send_json = lambda data, status=200, headers=None: responses.append(
        (status, data, headers or {})
    )
    return handler


def test_shift_report_api_returns_counts_with_cors(tmp_path):
    db_path = tmp_path / "line.db"
    db = CAPIDatabase(db_path)
    _insert_record(db_path, "G1", "OK", "2026-01-20 10:00:00")
    _insert_record(db_path, "G2", "NG", "2026-01-20 11:00:00", machine_judgment="NG")

    responses = []
    handler = _make_handler(db, responses)
    handler._handle_api_shift_report({"date": ["2026-01-20"], "shift": ["day"]})

    status, payload, headers = responses[-1]
    assert status == 200
    assert headers.get("Access-Control-Allow-Origin") == "*"
    assert payload["date"] == "2026-01-20"
    assert payload["shift"] == "day"
    assert payload["shift_name"] == "白班"
    assert payload["total"] == 2
    assert payload["ok_count"] == 1
    assert payload["ng_count"] == 1
    assert payload["aoi_ng_count"] == 1
    assert payload["err_count"] == 0
    assert payload["time_range"] == "01/20 07:30 ~ 01/20 19:30"
    assert payload["start"] == "2026-01-20 07:30:00"
    assert payload["end"] == "2026-01-20 19:30:00"


def test_shift_report_api_rejects_missing_or_invalid_params(tmp_path):
    db = CAPIDatabase(tmp_path / "line.db")
    responses = []
    handler = _make_handler(db, responses)

    handler._handle_api_shift_report({})
    assert responses[-1][0] == 400

    handler._handle_api_shift_report({"date": ["2026-01-20"]})
    assert responses[-1][0] == 400

    handler._handle_api_shift_report({"date": ["bad"], "shift": ["day"]})
    assert responses[-1][0] == 400

    handler._handle_api_shift_report({"date": ["2026-01-20"], "shift": ["bad"]})
    assert responses[-1][0] == 400
    # 錯誤回覆同樣帶 CORS，前端才讀得到錯誤內容
    assert responses[-1][2].get("Access-Control-Allow-Origin") == "*"


def test_shift_report_route_is_registered():
    """do_GET 路由必須包含 /api/shift_report（內容級檢查）。"""
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "capi_web.py").read_text(
        encoding="utf-8"
    )
    assert '"/api/shift_report"' in source
