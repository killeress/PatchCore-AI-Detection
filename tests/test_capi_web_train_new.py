"""API 測試：/api/train/new/panels + /api/train/new/start

使用 mock server + 直接呼叫 handler 方法，不需啟動真實 HTTP server。
"""
import io
import json
import sqlite3
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_handler(db_inst, path="/api/train/new/panels?machine_id=GN160&days=7"):
    """建一個 minimal handler 物件，用於直接呼叫 _handle_* 方法。"""
    from capi_web import CAPIWebHandler

    h = CAPIWebHandler.__new__(CAPIWebHandler)
    # 直接掛 db 類別屬性（handler 用 self.db）
    h.db = db_inst

    h.path = path
    h.headers = MagicMock()
    h.headers.get = MagicMock(return_value="0")
    h.wfile = io.BytesIO()

    # 攔截 _send_json 輸出
    h._sent_response = []

    def capture_json(payload, status=200):
        h._sent_response.append({"status": status, "body": json.dumps(payload)})

    def capture_send(status, body, content_type="text/html; charset=utf-8"):
        h._sent_response.append({"status": status, "body": body})

    h._send_json = capture_json
    h._send_response = capture_send
    return h


def _make_handler_with_server(server, path="/api/train/new/start"):
    """建一個 handler，_capi_server_instance 指向 mock server。"""
    from capi_web import CAPIWebHandler

    h = CAPIWebHandler.__new__(CAPIWebHandler)
    h._capi_server_instance = server
    h.db = server.database

    h.path = path
    h.headers = MagicMock()
    h.headers.get = MagicMock(return_value="0")
    h.rfile = io.BytesIO(b"")
    h.wfile = io.BytesIO()

    h._sent_response = []

    def capture_json(payload, status=200):
        h._sent_response.append({"status": status, "body": json.dumps(payload)})

    def capture_send(status, body, content_type="text/html; charset=utf-8"):
        h._sent_response.append({"status": status, "body": body})

    h._send_json = capture_json
    h._send_response = capture_send
    return h


def _make_real_db():
    """建一個 in-memory SQLite，填入測試資料，包裝成 minimal DB 物件。"""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE inference_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            glass_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            machine_no TEXT NOT NULL,
            machine_judgment TEXT DEFAULT '',
            ai_judgment TEXT DEFAULT '',
            image_dir TEXT DEFAULT '',
            request_time TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.execute("""
        INSERT INTO inference_records
            (glass_id, model_id, machine_no, machine_judgment, ai_judgment,
             image_dir, request_time, created_at)
        VALUES
            ('G001', 'GN160', 'CAPI07', 'OK', 'OK',
             '/data/G001', datetime('now'), datetime('now'))
    """)
    conn.execute("""
        INSERT INTO inference_records
            (glass_id, model_id, machine_no, machine_judgment, ai_judgment,
             image_dir, request_time, created_at)
        VALUES
            ('G002', 'GN160', 'CAPI07', 'OK', 'NG@img(10,20)',
             '/data/G002', datetime('now'), datetime('now'))
    """)
    conn.execute("""
        INSERT INTO inference_records
            (glass_id, model_id, machine_no, machine_judgment, ai_judgment,
             image_dir, request_time, created_at)
        VALUES
            ('G003', 'OTHER_MODEL', 'CAPI07', 'OK', 'OK',
             '/data/G003', datetime('now'), datetime('now'))
    """)
    conn.commit()

    db_mock = MagicMock()
    db_mock._get_conn.return_value = conn
    return db_mock


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_handle_train_new_panels_returns_db_result():
    """正常情境：回傳指定 machine_id 的 machine_judgment='OK' panel 清單（ai_judgment 不限）。"""
    db = _make_real_db()
    h = _make_handler(db, "/api/train/new/panels?machine_id=GN160&days=7")
    h._handle_train_new_panels()

    assert len(h._sent_response) == 1
    resp = h._sent_response[0]
    assert resp["status"] == 200
    body = json.loads(resp["body"])
    assert "panels" in body
    # G001 是 OK；G002 是 NG (ai_judgment)；G003 是其他 model_id — 應回傳 G001 和 G002
    assert len(body["panels"]) == 2
    glass_ids = {p["glass_id"] for p in body["panels"]}
    assert glass_ids == {"G001", "G002"}


def test_handle_train_new_panels_requires_machine_id():
    """缺少 machine_id 應回 400。"""
    db = MagicMock()
    h = _make_handler(db, "/api/train/new/panels?days=7")
    h._handle_train_new_panels()

    assert len(h._sent_response) == 1
    assert h._sent_response[0]["status"] == 400
    body = json.loads(h._sent_response[0]["body"])
    assert "error" in body


def test_handle_train_new_panels_db_not_set():
    """db 為 None 時應回 503。"""
    h = _make_handler(None, "/api/train/new/panels?machine_id=GN160&days=7")
    h._handle_train_new_panels()

    assert len(h._sent_response) == 1
    assert h._sent_response[0]["status"] == 503


# ── /api/train/new/start tests ────────────────────────────────────────────────

def test_handle_train_new_start_validates_params():
    """空 body（無 machine_id）→ 400。"""
    from capi_web import CAPIWebHandler

    server = MagicMock()
    server.database.get_active_training_job.return_value = None

    h = _make_handler_with_server(server, "/api/train/new/start")
    h.headers.get = MagicMock(return_value="0")
    h.rfile = io.BytesIO(b"")

    h._handle_train_new_start()

    assert len(h._sent_response) == 1
    assert h._sent_response[0]["status"] == 400
    body = json.loads(h._sent_response[0]["body"])
    assert "error" in body


def test_handle_train_new_start_rejects_concurrent():
    """現有 active job → 409。"""
    from capi_web import CAPIWebHandler

    server = MagicMock()
    server.database.get_active_training_job.return_value = {
        "job_id": "j_old",
        "state": "preprocess",
    }

    h = _make_handler_with_server(server, "/api/train/new/start")
    body = json.dumps({"machine_id": "M", "panel_paths": ["/p1"]}).encode()
    h.headers.get = MagicMock(return_value=str(len(body)))
    h.rfile = io.BytesIO(body)

    h._handle_train_new_start()

    assert len(h._sent_response) == 1
    assert h._sent_response[0]["status"] == 409
    resp_body = json.loads(h._sent_response[0]["body"])
    assert resp_body.get("error") == "job_already_running"
    assert resp_body.get("active_job_id") == "j_old"


# ── /api/train/new/status tests ───────────────────────────────────────────────

def test_handle_train_new_status_idle():
    """無 active job，未指定 job_id → state: idle。"""
    server = MagicMock()
    server.database.get_active_training_job.return_value = None
    h = _make_handler_with_server(server, "/api/train/new/status")
    h._handle_train_new_status()
    body = json.loads(h._sent_response[0]["body"])
    assert body["state"] == "idle"


def test_handle_train_new_status_with_job_id():
    """指定 job_id，存在 → 回傳該 job 狀態。"""
    server = MagicMock()
    server.database.get_training_job.return_value = {
        "job_id": "j1", "machine_id": "M", "state": "review",
        "started_at": "2026-04-28 10:00:00", "completed_at": None,
        "output_bundle": None, "error_message": None,
    }
    h = _make_handler_with_server(server, "/api/train/new/status?job_id=j1")
    h._handle_train_new_status()
    body = json.loads(h._sent_response[0]["body"])
    assert body["state"] == "review"
    assert body["job_id"] == "j1"


def test_handle_train_new_status_not_found():
    """指定 job_id，不存在 → 404。"""
    server = MagicMock()
    server.database.get_training_job.return_value = None
    h = _make_handler_with_server(server, "/api/train/new/status?job_id=missing")
    h._handle_train_new_status()
    assert h._sent_response[0]["status"] == 404


# ── /api/train/new/tiles tests ────────────────────────────────────────────────

def test_handle_train_new_tiles_returns_pool():
    """正常情境：回傳 tile pool 清單。"""
    server = MagicMock()
    server.database.list_tile_pool.return_value = [
        {"id": 1, "lighting": "G0F00000", "zone": "inner", "source": "ok",
         "decision": "accept", "thumb_path": "/t/1.png"},
    ]
    h = _make_handler_with_server(server, "/api/train/new/tiles?job_id=j1&lighting=G0F00000")
    h._handle_train_new_tiles()
    body = json.loads(h._sent_response[0]["body"])
    assert len(body["tiles"]) == 1
    server.database.list_tile_pool.assert_called_with("j1", lighting="G0F00000")


def test_handle_train_new_tiles_requires_params():
    """缺少 lighting → 400。"""
    server = MagicMock()
    h = _make_handler_with_server(server, "/api/train/new/tiles?job_id=j1")  # missing lighting
    h._handle_train_new_tiles()
    assert h._sent_response[0]["status"] == 400


def test_handle_train_new_tiles_decision_updates():
    """正常情境：更新 tile decisions，回傳 ok + updated count。"""
    server = MagicMock()
    h = _make_handler_with_server(server, "/api/train/new/tiles/decision")
    body_bytes = json.dumps({"job_id": "j1", "tile_ids": [1, 2, 3], "decision": "reject"}).encode()
    h.headers.get = MagicMock(return_value=str(len(body_bytes)))
    h.rfile = io.BytesIO(body_bytes)
    h._handle_train_new_tiles_decision()
    server.database.update_tile_decisions.assert_called_with("j1", [1, 2, 3], "reject")
    resp_body = json.loads(h._sent_response[0]["body"])
    assert resp_body["ok"] is True
    assert resp_body["updated"] == 3


def test_handle_train_new_tiles_decision_validates_decision():
    """decision 不在 accept|reject → 400。"""
    server = MagicMock()
    h = _make_handler_with_server(server, "/api/train/new/tiles/decision")
    body_bytes = json.dumps({"job_id": "j1", "tile_ids": [1], "decision": "maybe"}).encode()
    h.headers.get = MagicMock(return_value=str(len(body_bytes)))
    h.rfile = io.BytesIO(body_bytes)
    h._handle_train_new_tiles_decision()
    assert h._sent_response[0]["status"] == 400
