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
import numpy as np
import cv2


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
    conn.execute("""
        INSERT INTO inference_records
            (glass_id, model_id, machine_no, machine_judgment, ai_judgment,
             image_dir, request_time, created_at)
        VALUES
            ('G004', 'GN160A', 'CAPI07', 'OK', 'OK',
             '/data/G004', datetime('now'), datetime('now'))
    """)
    conn.commit()

    def _list_ok_panels(machine_id="", days=3, limit=100, machine_id_prefix=""):
        cur = conn.cursor()
        where = []
        params = []
        if machine_id_prefix:
            where.append("substr(model_id, 1, ?) = ?")
            params.extend([len(machine_id_prefix), machine_id_prefix])
        elif machine_id:
            where.append("model_id = ?")
            params.append(machine_id)
        where.extend(["machine_judgment = 'OK'", "created_at >= datetime('now', ? || ' days')"])
        params.append(f"-{days}")
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
    # G001/G002 是精準 GN160；G003/G004 非精準 GN160，應排除。
    assert len(body["panels"]) == 2
    glass_ids = {p["glass_id"] for p in body["panels"]}
    assert glass_ids == {"G001", "G002"}
    assert all(p["image_path"] == p["image_dir"] for p in body["panels"])


def test_handle_train_new_panels_adds_w0_preview_image_path(tmp_path):
    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    w0_path = panel_dir / "W0F00000_084027.tif"
    g0_path = panel_dir / "G0F00000_084027.tif"
    w0_path.write_bytes(b"w0")
    g0_path.write_bytes(b"g0")

    db = MagicMock()
    db.list_ok_panels_for_machine.return_value = [{
        "id": 1,
        "glass_id": "G001",
        "model_id": "GN160",
        "machine_no": "CAPI07",
        "machine_judgment": "OK",
        "ai_judgment": "OK",
        "image_dir": str(panel_dir),
        "request_time": "2026-05-28 08:40:00",
        "created_at": "2026-05-28 08:40:00",
    }]
    h = _make_handler(db, "/api/train/new/panels?machine_id=GN160&days=3")

    h._handle_train_new_panels()

    body = json.loads(h._sent_response[0]["body"])
    assert body["panels"][0]["image_path"] == str(panel_dir)
    assert body["panels"][0]["preview_image_path"] == str(w0_path)


def test_handle_train_new_panels_supports_machine_id_prefix():
    """局部重訓可用料號前綴找同 family panel。"""
    db = _make_real_db()
    h = _make_handler(db, "/api/train/new/panels?machine_id_prefix=GN160&days=3")
    h._handle_train_new_panels()

    body = json.loads(h._sent_response[0]["body"])
    assert h._sent_response[0]["status"] == 200
    assert {p["glass_id"] for p in body["panels"]} == {"G001", "G002", "G004"}


def test_handle_train_new_panels_all_recent_without_machine_id():
    """未指定 machine_id 時，回傳最近 AOI OK 推論紀錄供前端直接挑選。"""
    db = _make_real_db()
    h = _make_handler(db, "/api/train/new/panels?days=3")
    h._handle_train_new_panels()

    body = json.loads(h._sent_response[0]["body"])
    assert h._sent_response[0]["status"] == 200
    assert {p["glass_id"] for p in body["panels"]} == {"G001", "G002", "G003", "G004"}
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


def test_handle_train_new_start_allows_concurrent_with_review_job():
    """有別人在 review state 不再阻擋新 start（多 job 共存）。"""
    from capi_web import CAPIWebHandler

    server = MagicMock()
    server.database.list_active_training_jobs.return_value = [
        {"job_id": "j_old", "state": "review"},
    ]
    server.database.create_training_job = MagicMock()
    CAPIWebHandler._train_new_jobs = {}
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()

    h = _make_handler_with_server(server, "/api/train/new/start")
    body = json.dumps({"machine_id": "M", "panel_paths": [f"/p{i}" for i in range(8)]}).encode()
    h.headers.get = MagicMock(return_value=str(len(body)))
    h.rfile = io.BytesIO(body)

    with patch("capi_web.threading.Thread") as MockThread:
        MockThread.return_value.start = MagicMock()
        h._handle_train_new_start()

    assert h._sent_response[0]["status"] == 200
    body_json = json.loads(h._sent_response[0]["body"])
    assert body_json["state"] == "preprocess"
    assert body_json["job_id"]
    server.database.create_training_job.assert_called_once()


def test_handle_train_new_start_parallel_creates_two_jobs():
    """TOCTOU race regression：兩個 thread 同時 start 都成功。"""
    from concurrent.futures import ThreadPoolExecutor
    from capi_web import CAPIWebHandler

    created = []
    db_lock = threading.Lock()

    server = MagicMock()
    server.database.list_active_training_jobs.return_value = []

    def record_create(job_id, machine_id, panel_paths, training_params=None, **_kwargs):
        with db_lock:
            created.append(job_id)

    server.database.create_training_job = MagicMock(side_effect=record_create)

    CAPIWebHandler._train_new_jobs = {}
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()

    def run_start():
        h = _make_handler_with_server(server, "/api/train/new/start")
        body = json.dumps({"machine_id": "M", "panel_paths": [f"/p{i}" for i in range(8)]}).encode()
        h.headers.get = MagicMock(return_value=str(len(body)))
        h.rfile = io.BytesIO(body)
        with patch("capi_web.threading.Thread") as MockThread:
            MockThread.return_value.start = MagicMock()
            h._handle_train_new_start()
        return h._sent_response[0]

    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(run_start) for _ in range(2)]
        results = [f.result() for f in futs]

    statuses = sorted(r["status"] for r in results)
    assert statuses == [200, 200]
    assert len(set(created)) == 2  # 兩個不同 job_id


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
               "max_epochs": 2}
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

    def test_precision_choice_accepted(self):
        from capi_web import CAPIWebHandler
        for val in ("float16", "float32"):
            params, err = CAPIWebHandler._validate_training_params(
                {"precision": val}
            )
            assert err is None
            assert params == {"precision": val}

    def test_precision_invalid_choice_rejected(self):
        from capi_web import CAPIWebHandler
        for raw in [{"precision": "float8"}, {"precision": 16},
                    {"precision": "fp16"}]:
            _, err = CAPIWebHandler._validate_training_params(raw)
            assert err and "must be one of" in err, f"expected error for {raw}"

    def test_precision_mixed_with_numeric_params(self):
        from capi_web import CAPIWebHandler
        raw = {"batch_size": 16, "coreset_ratio": 0.05,
               "precision": "float32"}
        params, err = CAPIWebHandler._validate_training_params(raw)
        assert err is None
        assert params == raw


def test_handle_train_new_start_rejects_bad_training_params():
    """training_params 含越界值 → 400。"""
    server = MagicMock()
    server.database.get_active_training_job.return_value = None

    h = _make_handler_with_server(server, "/api/train/new/start")
    body = json.dumps({
        "machine_id": "M",
        "panel_paths": [f"/p{i}" for i in range(8)],
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
        "panel_paths": [f"/p{i}" for i in range(8)],
        "training_params": {
            "batch_size": 16, "coreset_ratio": 0.05,
            "max_epochs": 2,
        },
        "image_preprocess_pipeline": [
            {"method": "bilateral", "params": {"diameter": 9, "sigma_color": 35.0, "sigma_space": 35.0}},
            {"method": "gaussian", "params": {"kernel_size": 5, "sigma": 1.0}},
        ],
    }
    body = json.dumps(payload).encode()
    h.headers.get = MagicMock(return_value=str(len(body)))
    h.rfile = io.BytesIO(body)

    h._handle_train_new_start()

    assert h._sent_response[0]["status"] == 200
    server.database.create_training_job.assert_called_once()
    call_kwargs = server.database.create_training_job.call_args.kwargs
    assert call_kwargs["training_params"] == payload["training_params"]
    assert call_kwargs["image_preprocess_pipeline"][0]["method"] == "bilateral"


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
    from capi_web import CAPIWebHandler

    server = MagicMock()
    server.database.get_training_job.return_value = {
        "job_id": "j1", "machine_id": "M", "state": "review", "panel_paths": ["/p"]
    }
    CAPIWebHandler._train_new_jobs = {}
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()
    CAPIWebHandler._train_slot = {"lock": threading.Lock(), "active_job_id": None}

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
    assert CAPIWebHandler._train_slot["active_job_id"] == "j1"


def test_handle_train_new_start_training_rejects_when_slot_held():
    """另一個 job 已在 train → 第二個 start_training 收 409。"""
    from capi_web import CAPIWebHandler

    server = MagicMock()
    server.database.get_training_job.return_value = {
        "job_id": "j2", "machine_id": "M", "state": "review", "panel_paths": []
    }
    CAPIWebHandler._train_new_jobs = {}
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()
    CAPIWebHandler._train_slot = {
        "lock": threading.Lock(),
        "active_job_id": "j1",
    }

    h = _make_handler_with_server(server, "/api/train/new/start_training/j2")
    h._handle_train_new_start_training()

    assert h._sent_response[0]["status"] == 409
    body = json.loads(h._sent_response[0]["body"])
    assert body["error"] == "another_job_training"
    assert body["training_job_id"] == "j1"
    # slot 不應被改寫
    assert CAPIWebHandler._train_slot["active_job_id"] == "j1"


def test_handle_train_new_cancel_marks_review_job_failed():
    from capi_web import CAPIWebHandler

    server = MagicMock()
    server.database.get_training_job.return_value = {
        "job_id": "j1", "machine_id": "M", "state": "review", "panel_paths": []
    }
    CAPIWebHandler._train_new_jobs = {
        "j1": {
            "thread": None, "proc": None, "cancel_flag": None,
            "log_file": None, "cancel_event": threading.Event(),
            "log_lines": [], "log_lock": threading.Lock(),
            "unit_status": {}, "phase": "review",
        }
    }
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()
    CAPIWebHandler._train_slot = {"lock": threading.Lock(), "active_job_id": None}

    h = _make_handler_with_server(server, "/api/train/new/cancel/j1")
    h._handle_train_new_cancel()

    server.database.update_training_job_state.assert_called_once_with(
        "j1", "failed", error_message="cancelled by user"
    )
    body = json.loads(h._sent_response[0]["body"])
    assert body["ok"] is True
    assert "j1" not in CAPIWebHandler._train_new_jobs


def test_handle_train_new_cancel_marks_stale_running_job_failed():
    server = MagicMock()
    server.database.get_training_job.return_value = {
        "job_id": "j1", "machine_id": "M", "state": "train", "panel_paths": []
    }
    from capi_web import CAPIWebHandler
    CAPIWebHandler._train_new_jobs = {}
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()
    CAPIWebHandler._train_slot = {"lock": threading.Lock(), "active_job_id": None}
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
    CAPIWebHandler._train_new_jobs = {
        "j1": {
            "thread": AliveThread(), "proc": None, "cancel_flag": None,
            "log_file": None, "cancel_event": cancel_event,
            "log_lines": [], "log_lock": threading.Lock(),
            "unit_status": {}, "phase": "train",
        }
    }
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()
    CAPIWebHandler._train_slot = {"lock": threading.Lock(), "active_job_id": "j1"}
    h = _make_handler_with_server(server, "/api/train/new/cancel/j1")

    h._handle_train_new_cancel()

    server.database.update_training_job_state.assert_not_called()
    assert cancel_event.is_set()
    body = json.loads(h._sent_response[0]["body"])
    assert body["cancel_requested"] is True


def test_handle_train_new_status_returns_per_job_log():
    """兩個 job 的 log 不應互串。"""
    from capi_web import CAPIWebHandler

    CAPIWebHandler._train_new_jobs = {
        "jA": {
            "thread": None, "proc": None, "cancel_flag": None,
            "log_file": None, "cancel_event": threading.Event(),
            "log_lines": ["[hh:mm:ss] A line"], "log_lock": threading.Lock(),
            "unit_status": {}, "phase": "review",
        },
        "jB": {
            "thread": None, "proc": None, "cancel_flag": None,
            "log_file": None, "cancel_event": threading.Event(),
            "log_lines": ["[hh:mm:ss] B line"], "log_lock": threading.Lock(),
            "unit_status": {}, "phase": "review",
        },
    }
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()

    server = MagicMock()
    server.database.get_training_job.side_effect = lambda jid: {
        "job_id": jid, "machine_id": "M", "state": "review",
        "started_at": None, "completed_at": None,
        "output_bundle": None, "error_message": None,
        "panel_paths": [],
    }

    h = _make_handler_with_server(server, "/api/train/new/status?job_id=jA")
    h._handle_train_new_status()
    body = json.loads(h._sent_response[0]["body"])
    assert body["log_lines"] == ["[hh:mm:ss] A line"]

    h2 = _make_handler_with_server(server, "/api/train/new/status?job_id=jB")
    h2._handle_train_new_status()
    body2 = json.loads(h2._sent_response[0]["body"])
    assert body2["log_lines"] == ["[hh:mm:ss] B line"]


def test_handle_train_new_cancel_isolates_flags(tmp_path):
    """取消 j1 不應 touch j2 的 cancel flag 檔（事故 root cause regression）。"""
    from capi_web import CAPIWebHandler

    flag_a = tmp_path / "a.cancel"
    flag_b = tmp_path / "b.cancel"

    class AliveProc:
        def poll(self):
            return None

    class AliveThread:
        def is_alive(self):
            return True

    CAPIWebHandler._train_new_jobs = {
        "j1": {
            "thread": AliveThread(), "proc": AliveProc(),
            "cancel_flag": str(flag_a), "log_file": None,
            "cancel_event": threading.Event(),
            "log_lines": [], "log_lock": threading.Lock(),
            "unit_status": {}, "phase": "train",
        },
        "j2": {
            "thread": AliveThread(), "proc": AliveProc(),
            "cancel_flag": str(flag_b), "log_file": None,
            "cancel_event": threading.Event(),
            "log_lines": [], "log_lock": threading.Lock(),
            "unit_status": {}, "phase": "train",
        },
    }
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()
    CAPIWebHandler._train_slot = {"lock": threading.Lock(), "active_job_id": "j1"}

    server = MagicMock()
    server.database.get_training_job.return_value = {
        "job_id": "j1", "machine_id": "M", "state": "train", "panel_paths": []
    }
    h = _make_handler_with_server(server, "/api/train/new/cancel/j1")
    h._handle_train_new_cancel()

    assert flag_a.exists()
    assert not flag_b.exists()
    assert CAPIWebHandler._train_new_jobs["j1"]["cancel_event"].is_set()
    assert not CAPIWebHandler._train_new_jobs["j2"]["cancel_event"].is_set()


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


def test_train_new_step4_progress_template_exists():
    template_path = Path(__file__).resolve().parent.parent / "templates" / "train_new" / "step4_progress.html"
    assert template_path.exists()


def test_train_new_done_template_uses_chinese_summary_labels():
    template_path = Path(__file__).resolve().parent.parent / "templates" / "train_new" / "step5_done.html"
    text = template_path.read_text(encoding="utf-8")
    assert "<th>光源</th>" in text
    assert "<th>區域</th>" in text
    assert "訓練 tile" in text
    assert "模型包" in text
    assert "<th>Lighting</th>" not in text
    assert "<th>Zone</th>" not in text
    assert "BUNDLE</span>" not in text


def test_train_new_select_panel_renderer_uses_text_content():
    template_path = Path(__file__).resolve().parent.parent / "templates" / "train_new" / "step1_select.html"
    text = template_path.read_text(encoding="utf-8")
    assert "function appendTextCell" in text
    assert "td.textContent = String(value);" in text
    assert "${escapeHtml(p.glass_id" not in text
    assert "${escapeHtml(p.machine_no" not in text


def test_train_new_select_has_pipeline_preview_controls():
    template_path = Path(__file__).resolve().parent.parent / "templates" / "train_new" / "step1_select.html"
    text = template_path.read_text(encoding="utf-8")
    assert "用一張圖片跑一次當前流程" in text
    assert "preprocess_pipeline_preview" in text
    assert "runPreprocessPreview" in text
    assert "showDiffMapHelp" in text
    assert "差異圖怎麼看" in text
    assert "_previewPathByPanel" in text
    assert "p.preview_image_path" in text
    assert "responseText ? JSON.parse(responseText)" in text
    assert "preprocess_after_tiling: afterTiling" in text


def test_record_detail_templates_show_preprocess_pipeline():
    base = Path(__file__).resolve().parent.parent / "templates"
    for name in ("record_detail.html", "record_detail_v3.html"):
        text = (base / name).read_text(encoding="utf-8")
        assert "影像前處理" in text
        assert "image_preprocess_pipeline_steps" in text
        assert "前處理總耗時" in text
        assert "step.timing_text" in text
        assert "舊紀錄未記錄" in text


def test_record_preprocess_info_decoration_formats_steps():
    from capi_web import CAPIWebHandler

    detail = {
        "image_preprocess_pipeline": json.dumps([
            {"method": "bilateral", "params": {"diameter": 9, "sigma_color": 35.0, "sigma_space": 35.0}},
            {"method": "gaussian", "params": {"kernel_size": 5, "sigma": 1.0}},
        ]),
        "image_preprocess_timing": json.dumps({
            "total_elapsed_ms": 30.0,
            "steps": [
                {
                    "index": 1,
                    "method": "bilateral",
                    "method_label": "雙邊濾波",
                    "applied_params": {"diameter": 9, "sigma_color": 35.0, "sigma_space": 35.0},
                    "calls": 3,
                    "elapsed_ms_total": 21.0,
                    "elapsed_ms_avg": 7.0,
                },
                {
                    "index": 2,
                    "method": "gaussian",
                    "method_label": "高斯平滑",
                    "applied_params": {"kernel_size": 5, "sigma": 1.0},
                    "calls": 3,
                    "elapsed_ms_total": 9.0,
                    "elapsed_ms_avg": 3.0,
                },
            ],
        }),
    }

    CAPIWebHandler._decorate_record_preprocess_info(detail)

    assert detail["image_preprocess_pipeline_recorded"] is True
    assert detail["image_preprocess_pipeline_steps"][0]["method_label"] == "雙邊濾波"
    assert "Diameter=9" in detail["image_preprocess_pipeline_steps"][0]["params_text"]
    assert detail["image_preprocess_pipeline_steps"][0]["timing_text"] == "耗時 0.021s / 3 次 / 平均 7.00ms"
    assert detail["image_preprocess_total_seconds_text"] == "0.030s"
    assert "高斯平滑" in detail["image_preprocess_pipeline_summary"]


def test_do_post_routes_train_new_preprocess_pipeline_preview(monkeypatch):
    from capi_web import CAPIWebHandler

    server = MagicMock()
    h = _make_handler_with_server(server, "/api/train/new/preprocess_pipeline_preview")
    called = []

    def fake_preview(self):
        called.append(self.path)
        self._send_json({"ok": True})

    monkeypatch.setattr(
        CAPIWebHandler,
        "_handle_train_new_preprocess_pipeline_preview",
        fake_preview,
    )

    h.do_POST()

    assert called == ["/api/train/new/preprocess_pipeline_preview"]
    assert json.loads(h._sent_response[0]["body"]) == {"ok": True}


def test_handle_train_new_preprocess_pipeline_preview_uses_panel_folder(tmp_path, monkeypatch):
    from capi_web import CAPIWebHandler

    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    img = np.full((64, 64), 128, dtype=np.uint8)
    img[16:48, 16:48] = 200
    image_path = panel_dir / "W0F00000_084027.tif"
    other_path = panel_dir / "G0F00000_084027.tif"
    assert cv2.imwrite(str(image_path), img)
    assert cv2.imwrite(str(other_path), img)

    debug_dir = tmp_path / "debug"
    monkeypatch.setattr(CAPIWebHandler, "_debug_heatmap_dir", debug_dir)

    server = MagicMock()
    h = _make_handler_with_server(server, "/api/train/new/preprocess_pipeline_preview")
    payload = {
        "image_path": str(panel_dir),
        "image_preprocess_pipeline": [
            {"method": "gaussian", "params": {"kernel_size": 3, "sigma": 1.0}},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    h.headers.get = MagicMock(return_value=str(len(body)))
    h.rfile = io.BytesIO(body)

    h._handle_train_new_preprocess_pipeline_preview()

    assert h._sent_response[0]["status"] == 200
    resp = json.loads(h._sent_response[0]["body"])
    assert resp["success"] is True
    assert resp["image_path"] == str(image_path)
    assert resp["pipeline"][0]["method"] == "gaussian"
    assert resp["processed_url"].startswith("/debug/heatmaps/train_preprocess_preview_")
    assert Path(resp["output_path"]).exists()
    assert Path(resp["diff_path"]).exists()


def test_handle_train_new_preprocess_pipeline_preview_after_tiling_uses_tile(tmp_path, monkeypatch):
    from capi_web import CAPIWebHandler

    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    fixture = Path(__file__).resolve().parent / "fixtures" / "preprocess" / "synthetic_panel.png"
    img = cv2.imread(str(fixture), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    image_path = panel_dir / "W0F00000_084027.png"
    assert cv2.imwrite(str(image_path), img)

    debug_dir = tmp_path / "debug"
    monkeypatch.setattr(CAPIWebHandler, "_debug_heatmap_dir", debug_dir)

    server = MagicMock()
    h = _make_handler_with_server(server, "/api/train/new/preprocess_pipeline_preview")
    payload = {
        "image_path": str(panel_dir),
        "preprocess_after_tiling": True,
        "image_preprocess_pipeline": [
            {"method": "gaussian", "params": {"kernel_size": 5, "sigma": 1.0}},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    h.headers.get = MagicMock(return_value=str(len(body)))
    h.rfile = io.BytesIO(body)

    h._handle_train_new_preprocess_pipeline_preview()

    assert h._sent_response[0]["status"] == 200
    resp = json.loads(h._sent_response[0]["body"])
    assert resp["success"] is True
    assert resp["preprocess_after_tiling"] is True
    assert resp["preview_mode"] == "tile"
    assert resp["preview_size"] == [512, 512]
    assert resp["pipeline"][0]["method"] == "gaussian"
    assert resp["steps"][0]["method"] == "gaussian"
    assert len(resp["tile_rect"]) == 4
    assert resp["original_url"].startswith("/debug/heatmaps/train_preprocess_preview_")
    assert Path(resp["original_path"]).exists()
    assert Path(resp["output_path"]).exists()
    assert Path(resp["diff_path"]).exists()


def test_handle_train_new_page_lists_all_active_jobs():
    """Step 1 banner 應列出所有 open job（preprocess / review / train），不再只有最新一筆。"""
    server = MagicMock()
    server.config = None
    server.database.list_model_bundles.return_value = []
    server.database.list_active_training_jobs.return_value = [
        {"job_id": "j_pre", "machine_id": "M", "state": "preprocess", "panel_paths": []},
        {"job_id": "j_rev", "machine_id": "M", "state": "review", "panel_paths": []},
    ]
    h = _make_handler_with_server(server, "/train/new")
    template = MagicMock()
    template.render.return_value = "<html>step1</html>"
    h.jinja_env = MagicMock()
    h.jinja_env.get_template.return_value = template

    h._handle_train_new_page()

    h.jinja_env.get_template.assert_called_with("train_new/step1_scope.html")
    template.render.assert_called_once()
    active_jobs = template.render.call_args.kwargs["active_jobs"]
    ids = [j["job_id"] for j in active_jobs]
    assert "j_rev" in ids
    # j_pre 在 preprocess 狀態：worker 不存在 → _mark_train_new_stale_if_needed 會把它標 failed → 從清單剔除
    assert h._sent_response[0]["body"] == "<html>step1</html>"


def test_handle_models_list_filters_by_machine_id():
    server = MagicMock()
    server.database.list_model_bundles.return_value = [{"id": 1, "machine_id": "M"}]
    h = _make_handler_with_server(server, "/api/models?machine_id=M")

    h._handle_models_list()

    server.database.list_model_bundles.assert_called_with(machine_id="M")
    body = json.loads(h._sent_response[0]["body"])
    assert body["bundles"] == [{"id": 1, "machine_id": "M"}]


def test_handle_models_update_notes_persists_free_text(tmp_path):
    from capi_database import CAPIDatabase

    db = CAPIDatabase(tmp_path / "test.db")
    bid = db.register_model_bundle({
        "machine_id": "GN160", "bundle_path": "model/GN160-20260428",
        "trained_at": "2026-04-28T15:30:45",
        "panel_count": 5, "inner_tile_count": 2400,
        "edge_tile_count": 900, "ng_tile_count": 150,
        "bundle_size_bytes": 478_000_000, "job_id": "j1",
    })
    server = MagicMock()
    server.database = db
    h = _make_handler_with_server(server, f"/api/models/{bid}/notes")
    notes = "量產 A 線\n<script>text only</script>"
    payload = json.dumps({"notes": notes}, ensure_ascii=False).encode("utf-8")
    h.rfile = io.BytesIO(payload)
    h.headers.get = MagicMock(return_value=str(len(payload)))

    h._handle_models_update_notes()

    body = json.loads(h._sent_response[0]["body"])
    assert h._sent_response[0]["status"] == 200
    assert body["notes"] == notes
    assert db.get_model_bundle(bid)["notes"] == notes


def test_handle_train_new_scope_page_exposes_submodel_retrain_choices():
    server = MagicMock()
    server.config = None
    server.database.list_active_training_jobs.return_value = []
    server.database.list_model_bundles.return_value = [
        {
            "id": 2,
            "machine_id": "M2",
            "bundle_path": "model/M2-old",
            "trained_at": "2026-05-01T10:00:00",
            "panel_count": 3,
            "inner_tile_count": 30,
            "edge_tile_count": 12,
            "ng_tile_count": 4,
            "is_active": 0,
            "job_id": "job2",
        },
        {
            "id": 1,
            "machine_id": "M1",
            "bundle_path": "model/M1-latest",
            "trained_at": "2026-05-02T10:00:00",
            "panel_count": 5,
            "inner_tile_count": 50,
            "edge_tile_count": 20,
            "ng_tile_count": 8,
            "is_active": 1,
            "job_id": "job1",
        },
        {
            "id": 3,
            "machine_id": "M3",
            "bundle_path": "model/M3-no-data",
            "trained_at": "2026-05-03T10:00:00",
            "is_active": 0,
            "job_id": "",
        },
    ]
    h = _make_handler_with_server(server, "/train/new")
    template = MagicMock()
    template.render.return_value = "<html>scope</html>"
    h.jinja_env = MagicMock()
    h.jinja_env.get_template.return_value = template

    h._handle_train_new_scope_page()

    h.jinja_env.get_template.assert_called_with("train_new/step1_scope.html")
    kwargs = template.render.call_args.kwargs
    assert [b["id"] for b in kwargs["bundles"]] == [1, 3, 2]
    assert kwargs["bundles"][0]["bundle_name"] == "M1-latest"
    assert kwargs["bundles"][0]["is_active"] is True
    assert "G0F00000-inner" in [u["label"] for u in kwargs["units"]]
    assert h._sent_response[0]["body"] == "<html>scope</html>"


def test_handle_models_retrain_submodel_with_panels_creates_job():
    from capi_web import CAPIWebHandler

    CAPIWebHandler._submodel_retrain_state = {"lock": threading.Lock(), "job": None}

    server = MagicMock()
    server.database.get_model_bundle.return_value = {
        "id": 1,
        "machine_id": "M1",
        "bundle_path": "model/M1-bundle",
    }
    server.database.create_training_job = MagicMock()

    h = _make_handler_with_server(server, "/api/models/1/retrain_submodel_with_panels")
    payload = {
        "lighting": "G0F00000",
        "zone": "inner",
        "panel_paths": [f"/panel/{i}" for i in range(4)],
        "training_params": {"batch_size": 8},
    }
    body = json.dumps(payload).encode()
    h.headers.get = MagicMock(return_value=str(len(body)))
    h.rfile = io.BytesIO(body)

    with patch("capi_train_new.generate_job_id", return_value="train_M1_20260518_abcd"):
        with patch("capi_web.threading.Thread") as MockThread:
            MockThread.return_value.start = MagicMock()
            h._handle_models_retrain_submodel_with_panels()

    assert h._sent_response[0]["status"] == 200
    resp = json.loads(h._sent_response[0]["body"])
    assert resp["job_id"] == "subtrain_M1_20260518_abcd"
    server.database.create_training_job.assert_called_once()
    kwargs = server.database.create_training_job.call_args.kwargs
    assert kwargs["job_id"] == "subtrain_M1_20260518_abcd"
    assert kwargs["machine_id"] == "M1"
    assert kwargs["panel_paths"] == payload["panel_paths"]
    assert kwargs["panel_modes"] == ["full", "full", "full", "corners_only"]
    assert kwargs["training_params"] == {"batch_size": 8}
    assert CAPIWebHandler._submodel_retrain_state["job"]["step"] == "preprocess"
    MockThread.return_value.start.assert_called_once()


def test_handle_train_new_start_persists_partial_training_scope():
    from capi_web import CAPIWebHandler

    server = MagicMock()
    server.database.get_model_bundle.return_value = {
        "id": 7,
        "machine_id": "GN156HRAA280S",
        "bundle_path": "model/GN156HRAA280S-bundle",
    }
    server.database.create_training_job = MagicMock()
    CAPIWebHandler._train_new_jobs = {}
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()

    h = _make_handler_with_server(server, "/api/train/new/start")
    payload = {
        "machine_id": "GN156HRAA2L0S",
        "panel_paths": [f"/p{i}" for i in range(8)],
        "training_scope": {
            "mode": "partial",
            "target_bundle_id": 7,
            "selected_units": ["G0F00000-inner", "R0F00000-edge"],
        },
    }
    body = json.dumps(payload).encode()
    h.headers.get = MagicMock(return_value=str(len(body)))
    h.rfile = io.BytesIO(body)

    with patch("capi_train_new.generate_job_id", return_value="train_GN156HRAA280S_20260518_abcd"):
        with patch("capi_web.threading.Thread") as MockThread:
            MockThread.return_value.start = MagicMock()
            h._handle_train_new_start()

    assert h._sent_response[0]["status"] == 200
    kwargs = server.database.create_training_job.call_args.kwargs
    assert kwargs["machine_id"] == "GN156HRAA280S"
    assert kwargs["training_scope"] == {
        "mode": "partial",
        "selected_units": ["G0F00000-inner", "R0F00000-edge"],
        "target_bundle_id": 7,
    }
    assert kwargs["panel_modes"] == ["full", "full", "full", "corners_only", "corners_only", "corners_only", "corners_only", "corners_only"]


def test_handle_train_new_start_rejects_wrong_panel_count():
    """非 8 片 panel 一律拒絕。"""
    server = MagicMock()
    server.database.get_active_training_job.return_value = None

    for n in (0, 1, 2, 3, 4, 5, 6, 7, 9):
        h = _make_handler_with_server(server, "/api/train/new/start")
        body = json.dumps({"machine_id": "M", "panel_paths": [f"/p{i}" for i in range(n)]}).encode()
        h.headers.get = MagicMock(return_value=str(len(body)))
        h.rfile = io.BytesIO(body)

        h._handle_train_new_start()
        assert h._sent_response[0]["status"] == 400, f"n={n} 應該被拒"
        if n > 0:
            err_body = json.loads(h._sent_response[0]["body"])
            assert "exactly 8" in err_body.get("error", ""), f"n={n} error: {err_body}"
