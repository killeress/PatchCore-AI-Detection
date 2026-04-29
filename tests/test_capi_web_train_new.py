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


def _make_handler(db_inst, path="/api/train/new/panels?days=3"):
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

    def _list_ok_panels(machine_id="", days=3, limit=100):
        cur = conn.cursor()
        where = ["machine_judgment = 'OK'", "created_at >= datetime('now', ? || ' days')"]
        params = [f"-{days}"]
        if machine_id:
            where.insert(0, "model_id = ?")
            params.insert(0, machine_id)
        params.append(limit)
        cur.execute(
            f"""SELECT id, glass_id, model_id, machine_no,
                       machine_judgment, ai_judgment, image_dir, request_time, created_at
                FROM inference_records
                WHERE {' AND '.join(where)}
                ORDER BY created_at DESC LIMIT ?""",
            params,
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    db_mock = MagicMock()
    db_mock._get_conn.return_value = conn
    db_mock.list_ok_panels_for_machine.side_effect = _list_ok_panels
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
    assert all(p["image_path"] == p["image_dir"] for p in body["panels"])


def test_handle_train_new_panels_all_recent_without_machine_id():
    """未指定 machine_id 時，回傳最近 AOI OK 推論紀錄供前端直接挑選。"""
    db = _make_real_db()
    h = _make_handler(db, "/api/train/new/panels?days=3")
    h._handle_train_new_panels()

    body = json.loads(h._sent_response[0]["body"])
    assert h._sent_response[0]["status"] == 200
    assert {p["glass_id"] for p in body["panels"]} == {"G001", "G002", "G003"}
    assert body["days"] == 3


def test_handle_train_new_panels_clamps_days_to_three():
    """機台只保留 3 天，API 即使收到更大 days 也只查 3 天。"""
    db = MagicMock()
    db.list_ok_panels_for_machine.return_value = []
    h = _make_handler(db, "/api/train/new/panels?days=30")
    h._handle_train_new_panels()

    db.list_ok_panels_for_machine.assert_called_once_with("", days=3)
    body = json.loads(h._sent_response[0]["body"])
    assert body["days"] == 3


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

    class AliveThread:
        def is_alive(self):
            return True

    server = MagicMock()
    server.database.get_active_training_job.return_value = {
        "job_id": "j_old",
        "state": "preprocess",
    }
    CAPIWebHandler._train_new_state = {
        "lock": threading.Lock(),
        "log_lock": threading.Lock(),
        "active_job_id": "j_old",
        "thread": AliveThread(),
        "cancel_event": threading.Event(),
        "log_lines": [],
    }

    h = _make_handler_with_server(server, "/api/train/new/start")
    body = json.dumps({"machine_id": "M", "panel_paths": [f"/p{i}" for i in range(5)]}).encode()
    h.headers.get = MagicMock(return_value=str(len(body)))
    h.rfile = io.BytesIO(body)

    h._handle_train_new_start()

    assert len(h._sent_response) == 1
    assert h._sent_response[0]["status"] == 409
    resp_body = json.loads(h._sent_response[0]["body"])
    assert resp_body.get("error") == "job_already_running"
    assert resp_body.get("active_job_id") == "j_old"


def test_handle_train_new_start_rejects_invalid_panel_path():
    server = MagicMock()
    server.database.get_active_training_job.return_value = None

    h = _make_handler_with_server(server, "/api/train/new/start")
    body = json.dumps({"machine_id": "M", "panel_paths": ["undefined"] * 5}).encode()
    h.headers.get = MagicMock(return_value=str(len(body)))
    h.rfile = io.BytesIO(body)

    h._handle_train_new_start()

    assert h._sent_response[0]["status"] == 400
    assert "invalid path" in json.loads(h._sent_response[0]["body"])["error"]


# ── training_params validation ────────────────────────────────────────────────

class TestValidateTrainingParams:
    def test_none_returns_none(self):
        from capi_web import CAPIWebHandler
        params, err = CAPIWebHandler._validate_training_params(None)
        assert params is None and err is None

    def test_empty_dict_returns_none(self):
        from capi_web import CAPIWebHandler
        params, err = CAPIWebHandler._validate_training_params({})
        assert params is None and err is None

    def test_full_valid_dict(self):
        from capi_web import CAPIWebHandler
        raw = {"batch_size": 16, "coreset_ratio": 0.05,
               "max_epochs": 2, "inner_panels": 4}
        params, err = CAPIWebHandler._validate_training_params(raw)
        assert err is None
        assert params == raw

    def test_partial_dict_keeps_only_supplied_keys(self):
        from capi_web import CAPIWebHandler
        params, err = CAPIWebHandler._validate_training_params({"batch_size": 4})
        assert err is None
        assert params == {"batch_size": 4}

    def test_unknown_key_rejected(self):
        from capi_web import CAPIWebHandler
        params, err = CAPIWebHandler._validate_training_params(
            {"learning_rate": 0.01}
        )
        assert params is None
        assert "unknown" in err

    def test_out_of_range_rejected(self):
        from capi_web import CAPIWebHandler
        for raw in [
            {"batch_size": 0},
            {"batch_size": 64},
            {"coreset_ratio": 0.0},
            {"coreset_ratio": 0.6},
            {"max_epochs": 0},
            {"max_epochs": 100},
            {"inner_panels": 0},
            {"inner_panels": 6},
        ]:
            _, err = CAPIWebHandler._validate_training_params(raw)
            assert err and "out of range" in err, f"expected error for {raw}"

    def test_wrong_type_rejected(self):
        from capi_web import CAPIWebHandler
        _, err = CAPIWebHandler._validate_training_params({"batch_size": "abc"})
        assert err and "must be int" in err

    def test_bool_not_treated_as_int(self):
        """bool 是 int 子類，但 batch_size=True 顯然不合理。"""
        from capi_web import CAPIWebHandler
        _, err = CAPIWebHandler._validate_training_params({"batch_size": True})
        assert err and "must be int" in err

    def test_non_dict_rejected(self):
        from capi_web import CAPIWebHandler
        _, err = CAPIWebHandler._validate_training_params([1, 2, 3])
        assert err and "must be an object" in err


def test_handle_train_new_start_rejects_bad_training_params():
    """training_params 含越界值 → 400。"""
    server = MagicMock()
    server.database.get_active_training_job.return_value = None

    h = _make_handler_with_server(server, "/api/train/new/start")
    body = json.dumps({
        "machine_id": "M",
        "panel_paths": [f"/p{i}" for i in range(5)],
        "training_params": {"batch_size": 999},
    }).encode()
    h.headers.get = MagicMock(return_value=str(len(body)))
    h.rfile = io.BytesIO(body)

    h._handle_train_new_start()

    assert h._sent_response[0]["status"] == 400
    err = json.loads(h._sent_response[0]["body"])["error"]
    assert "training_params.batch_size" in err
    assert "out of range" in err


def test_handle_train_new_start_persists_training_params(monkeypatch):
    """有效 training_params → 寫進 create_training_job 呼叫。"""
    from capi_web import CAPIWebHandler

    server = MagicMock()
    server.database.get_active_training_job.return_value = None
    server.database.create_training_job.return_value = 1

    CAPIWebHandler._train_new_state = {
        "lock": threading.Lock(),
        "log_lock": threading.Lock(),
        "active_job_id": None,
        "thread": None,
        "cancel_event": threading.Event(),
        "log_lines": [],
    }

    started = []

    class FakeThread:
        def __init__(self, *args, **kwargs):
            started.append(kwargs.get("name", ""))

        def start(self):
            pass

    monkeypatch.setattr("capi_web.threading.Thread", FakeThread)

    h = _make_handler_with_server(server, "/api/train/new/start")
    payload = {
        "machine_id": "M",
        "panel_paths": [f"/p{i}" for i in range(5)],
        "training_params": {
            "batch_size": 16, "coreset_ratio": 0.05,
            "max_epochs": 2, "inner_panels": 4,
        },
    }
    body = json.dumps(payload).encode()
    h.headers.get = MagicMock(return_value=str(len(body)))
    h.rfile = io.BytesIO(body)

    h._handle_train_new_start()

    assert h._sent_response[0]["status"] == 200
    server.database.create_training_job.assert_called_once()
    call_kwargs = server.database.create_training_job.call_args.kwargs
    assert call_kwargs["training_params"] == payload["training_params"]


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


def test_handle_train_new_tiles_adds_confined_thumb_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    thumb = tmp_path / ".tmp" / "train_new_thumbs" / "j1" / "thumb" / "a.png"
    thumb.parent.mkdir(parents=True)
    thumb.write_bytes(b"x")

    server = MagicMock()
    server.database.list_tile_pool.return_value = [
        {"id": 1, "lighting": "G0F00000", "zone": "inner", "source": "ok",
         "decision": "accept", "thumb_path": str(thumb)},
    ]
    h = _make_handler_with_server(server, "/api/train/new/tiles?job_id=j1&lighting=G0F00000")
    h._handle_train_new_tiles()

    body = json.loads(h._sent_response[0]["body"])
    assert body["tiles"][0]["thumb_url"] == "/api/train/new/thumb/j1/thumb/a.png"


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


# ── /api/train/new/start_training/<job_id> tests ──────────────────────────────

def test_handle_train_new_start_training_404_no_job():
    server = MagicMock()
    server.database.get_training_job.return_value = None
    h = _make_handler_with_server(server, "/api/train/new/start_training/missing")
    h._handle_train_new_start_training()
    assert h._sent_response[0]["status"] == 404


def test_handle_train_new_start_training_409_wrong_state():
    server = MagicMock()
    server.database.get_training_job.return_value = {
        "job_id": "j1", "machine_id": "M", "state": "preprocess", "panel_paths": []
    }
    h = _make_handler_with_server(server, "/api/train/new/start_training/j1")
    h._handle_train_new_start_training()
    assert h._sent_response[0]["status"] == 409


def test_handle_train_new_start_training_starts_thread(monkeypatch):
    """驗證在 review state 時，handler 會 update state + spawn thread。"""
    import threading
    server = MagicMock()
    server.database.get_training_job.return_value = {
        "job_id": "j1", "machine_id": "M", "state": "review", "panel_paths": ["/p"]
    }

    started_threads = []
    real_thread = threading.Thread
    def fake_thread(**kw):
        t = real_thread(target=lambda: None, daemon=True)
        started_threads.append(kw)
        return t
    monkeypatch.setattr("capi_web.threading.Thread", fake_thread)

    h = _make_handler_with_server(server, "/api/train/new/start_training/j1")
    h._handle_train_new_start_training()

    server.database.update_training_job_state.assert_called_with("j1", "train")
    assert len(started_threads) == 1
    body = json.loads(h._sent_response[0]["body"])
    assert body["state"] == "train"


def test_handle_train_new_cancel_marks_review_job_failed():
    from capi_web import CAPIWebHandler

    server = MagicMock()
    server.database.get_training_job.return_value = {
        "job_id": "j1", "machine_id": "M", "state": "review", "panel_paths": []
    }
    CAPIWebHandler._train_new_state = {
        "lock": threading.Lock(),
        "log_lock": threading.Lock(),
        "active_job_id": "j1",
        "log_lines": [],
    }

    h = _make_handler_with_server(server, "/api/train/new/cancel/j1")
    h._handle_train_new_cancel()

    server.database.update_training_job_state.assert_called_once_with(
        "j1", "failed", error_message="cancelled by user"
    )
    body = json.loads(h._sent_response[0]["body"])
    assert body["ok"] is True
    assert CAPIWebHandler._train_new_state["active_job_id"] is None


def test_handle_train_new_cancel_marks_stale_running_job_failed():
    server = MagicMock()
    server.database.get_training_job.return_value = {
        "job_id": "j1", "machine_id": "M", "state": "train", "panel_paths": []
    }
    from capi_web import CAPIWebHandler
    CAPIWebHandler._train_new_state = {
        "lock": threading.Lock(),
        "log_lock": threading.Lock(),
        "active_job_id": None,
        "thread": None,
        "log_lines": [],
    }
    h = _make_handler_with_server(server, "/api/train/new/cancel/j1")

    h._handle_train_new_cancel()

    server.database.update_training_job_state.assert_called_with(
        "j1",
        "failed",
        error_message="interrupted: server restarted or training worker is not running",
    )
    body = json.loads(h._sent_response[0]["body"])
    assert body["ok"] is True
    assert body["state"] == "failed"


def test_handle_train_new_cancel_requests_live_training_stop():
    from capi_web import CAPIWebHandler

    class AliveThread:
        def is_alive(self):
            return True

    server = MagicMock()
    server.database.get_training_job.return_value = {
        "job_id": "j1", "machine_id": "M", "state": "train", "panel_paths": []
    }
    cancel_event = threading.Event()
    CAPIWebHandler._train_new_state = {
        "lock": threading.Lock(),
        "log_lock": threading.Lock(),
        "active_job_id": "j1",
        "thread": AliveThread(),
        "cancel_event": cancel_event,
        "log_lines": [],
    }
    h = _make_handler_with_server(server, "/api/train/new/cancel/j1")

    h._handle_train_new_cancel()

    server.database.update_training_job_state.assert_not_called()
    assert cancel_event.is_set()
    body = json.loads(h._sent_response[0]["body"])
    assert body["cancel_requested"] is True


def test_handle_train_new_thumb_rejects_sibling_prefix_escape(tmp_path, monkeypatch):
    """Sibling paths such as train_new_thumbs_evil must not pass containment checks."""
    monkeypatch.chdir(tmp_path)
    leak_dir = tmp_path / ".tmp" / "train_new_thumbs_evil"
    leak_dir.mkdir(parents=True)
    (leak_dir / "leak.png").write_bytes(b"x")

    server = MagicMock()
    h = _make_handler_with_server(
        server,
        "/api/train/new/thumb/../train_new_thumbs_evil/leak.png",
    )
    h._send_binary = lambda path: h._sent_response.append({"status": 200, "body": path})

    h._handle_train_new_thumb()

    assert h._sent_response[0]["status"] == 403


def test_handle_train_new_progress_page_uses_step4_template_for_train_state():
    server = MagicMock()
    server.database.get_training_job.return_value = {"job_id": "j1", "state": "train"}
    h = _make_handler_with_server(server, "/train/new/progress?job_id=j1")
    template = MagicMock()
    template.render.return_value = "<html>step4</html>"
    h.jinja_env = MagicMock()
    h.jinja_env.get_template.return_value = template

    h._handle_train_new_progress_page()

    h.jinja_env.get_template.assert_called_with("train_new/step4_progress.html")
    assert h._sent_response[0]["body"] == "<html>step4</html>"


def test_handle_train_new_page_renders_step1_when_active_job_is_review():
    """review 狀態可以回 Step 1；只有 preprocess/train 需要強制回進度頁。"""
    server = MagicMock()
    server.database.get_active_training_job.return_value = {
        "job_id": "j1", "machine_id": "M", "state": "review"
    }
    h = _make_handler_with_server(server, "/train/new")
    template = MagicMock()
    template.render.return_value = "<html>step1</html>"
    h.jinja_env = MagicMock()
    h.jinja_env.get_template.return_value = template

    h._handle_train_new_page()

    h.jinja_env.get_template.assert_called_with("train_new/step1_select.html")
    template.render.assert_called_once()
    assert template.render.call_args.kwargs["active_review_job"]["job_id"] == "j1"
    assert h._sent_response[0]["body"] == "<html>step1</html>"


def test_handle_models_list_filters_by_machine_id():
    server = MagicMock()
    server.database.list_model_bundles.return_value = [{"id": 1, "machine_id": "M"}]
    h = _make_handler_with_server(server, "/api/models?machine_id=M")

    h._handle_models_list()

    server.database.list_model_bundles.assert_called_with(machine_id="M")
    body = json.loads(h._sent_response[0]["body"])
    assert body["bundles"] == [{"id": 1, "machine_id": "M"}]
