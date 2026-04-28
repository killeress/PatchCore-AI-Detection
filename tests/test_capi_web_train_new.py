"""API 測試：/api/train/new/panels

使用 mock server + 直接呼叫 handler 方法，不需啟動真實 HTTP server。
"""
import io
import json
import sqlite3
import tempfile
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
    """正常情境：回傳指定 machine_id 的 OK 判定 panel 清單。"""
    db = _make_real_db()
    h = _make_handler(db, "/api/train/new/panels?machine_id=GN160&days=7")
    h._handle_train_new_panels()

    assert len(h._sent_response) == 1
    resp = h._sent_response[0]
    assert resp["status"] == 200
    body = json.loads(resp["body"])
    assert "panels" in body
    # G001 是 OK；G002 是 NG；G003 是其他 model_id — 只應回傳 G001
    assert len(body["panels"]) == 1
    assert body["panels"][0]["glass_id"] == "G001"


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
