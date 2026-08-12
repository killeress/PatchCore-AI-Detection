import http.client
import io
import json
import threading
import urllib.parse

import pytest


@pytest.fixture(autouse=True)
def isolate_training_state(monkeypatch):
    from capi_web import CAPIWebHandler

    monkeypatch.setattr(CAPIWebHandler, "_train_new_jobs", {})
    monkeypatch.setattr(CAPIWebHandler, "_train_new_jobs_lock", threading.Lock())
    monkeypatch.setattr(
        CAPIWebHandler,
        "_train_slot",
        {"lock": threading.Lock(), "active_job_id": None},
    )
    monkeypatch.setattr(
        CAPIWebHandler,
        "_retrain_state",
        {"lock": threading.Lock(), "job": None},
        raising=False,
    )
    monkeypatch.setattr(
        CAPIWebHandler,
        "_submodel_retrain_state",
        {"lock": threading.Lock(), "job": None},
    )


def test_update_status_exposes_pending_version_without_package_path(tmp_path, monkeypatch):
    import capi_web
    from capi_web import CAPIWebHandler

    state_file = tmp_path / "auto_update_state.json"
    state_file.write_text(json.dumps({
        "status": "pending",
        "pending_update": {
            "version": "2099.01.02.2",
            "package": "/secret/path/update.zip",
            "sha256": "a" * 64,
            "detected_at": "2099-01-02T03:04:05+08:00",
        },
    }), encoding="utf-8")

    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.status_tracker = None
    captured = {}
    handler._send_json = lambda payload, status=200: captured.update({
        "payload": payload,
        "status": status,
    })
    monkeypatch.setattr(CAPIWebHandler, "_update_state_file", state_file)
    monkeypatch.setattr(capi_web, "get_version_info", lambda: {"version": "2099.01.02.1"})

    handler._handle_api_update_status()

    assert captured["status"] == 200
    assert captured["payload"]["status"] == "pending"
    assert captured["payload"]["pending_version"] == "2099.01.02.2"
    assert captured["payload"]["can_apply"] is True
    assert captured["payload"]["central_apply_supported"] is True
    assert "/secret/path" not in json.dumps(captured["payload"])


def test_dashboard_status_includes_sanitized_update_status(tmp_path, monkeypatch):
    import capi_web
    from capi_web import CAPIWebHandler

    state_file = tmp_path / "auto_update_state.json"
    state_file.write_text(json.dumps({
        "status": "pending",
        "pending_update": {
            "version": "2099.01.02.2",
            "package": "/secret/path/update.zip",
            "detected_at": "2099-01-02T03:04:05+08:00",
        },
    }), encoding="utf-8")

    class Tracker:
        @staticmethod
        def get_status():
            return {"server": {"running": True}, "stats": {}, "traffic": {}}

    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.status_tracker = Tracker()
    handler.db = None
    handler.heatmap_base_dir = tmp_path
    captured = {}
    handler._send_json = lambda payload, status=200, headers=None: captured.update({
        "payload": payload,
        "status": status,
        "headers": headers,
    })
    monkeypatch.setattr(CAPIWebHandler, "_update_state_file", state_file)
    monkeypatch.setattr(capi_web, "get_version_info", lambda: {"version": "2099.01.02.1"})
    monkeypatch.setattr(capi_web, "_get_host_identity", lambda: "CAPI01")
    monkeypatch.setattr(capi_web, "_get_cached_hardware_status", lambda path: {})

    handler._handle_api_status()

    assert captured["status"] == 200
    assert captured["headers"] == {"Access-Control-Allow-Origin": "*"}
    assert captured["payload"]["update"] == {
        "status": "pending",
        "current_version": "2099.01.02.1",
        "pending_version": "2099.01.02.2",
        "detected_at": "2099-01-02T03:04:05+08:00",
        "can_apply": True,
        "failure_reason": "",
        "central_apply_supported": True,
    }
    assert "/secret/path" not in json.dumps(captured["payload"])


def test_manual_apply_endpoint_launches_detached_updater(tmp_path, monkeypatch):
    import capi_web
    from capi_web import CAPIWebHandler

    state_file = tmp_path / "update" / "auto_update_state.json"
    state_file.parent.mkdir()
    state_file.write_text(json.dumps({
        "status": "pending",
        "pending_update": {
            "version": "2099.01.02.2",
            "package": str(tmp_path / "update" / "incoming" / "update.zip"),
            "sha256": "a" * 64,
        },
    }), encoding="utf-8")

    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return object()

    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.status_tracker = None
    captured = {}
    handler._send_json = lambda payload, status=200: captured.update({
        "payload": payload,
        "status": status,
    })
    monkeypatch.setattr(CAPIWebHandler, "_update_state_file", state_file)
    monkeypatch.setattr(CAPIWebHandler, "_update_apply_lock", threading.Lock())
    monkeypatch.setattr(capi_web.subprocess, "Popen", fake_popen)

    handler._handle_api_update_apply()

    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert captured["status"] == 202
    assert state["status"] == "apply_requested"
    assert calls[0][0][2] == "apply"
    assert "--delay" in calls[0][0]
    assert calls[0][1]["stderr"] == capi_web.subprocess.STDOUT


def test_manual_apply_is_rejected_while_inference_is_active(tmp_path, monkeypatch):
    import capi_web
    from capi_web import CAPIWebHandler

    state_file = tmp_path / "auto_update_state.json"
    state_file.write_text(json.dumps({
        "status": "pending",
        "pending_update": {"version": "2099.01.02.2"},
    }), encoding="utf-8")

    class BusyTracker:
        @staticmethod
        def get_status():
            return {"traffic": {"active_inferences": 1}}

    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.status_tracker = BusyTracker()
    captured = {}
    handler._send_json = lambda payload, status=200: captured.update({
        "payload": payload,
        "status": status,
    })
    monkeypatch.setattr(CAPIWebHandler, "_update_state_file", state_file)
    monkeypatch.setattr(capi_web.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("busy server must not launch updater")
    ))

    handler._handle_api_update_apply()

    assert captured["status"] == 409
    assert "檢測進行中" in captured["payload"]["error"]


@pytest.mark.parametrize(
    "training_kind",
    ["patchcore_preprocess", "patchcore_train", "scratch", "submodel"],
)
def test_manual_apply_is_rejected_while_training_is_active(
    tmp_path,
    monkeypatch,
    training_kind,
):
    import capi_web
    from capi_web import CAPIWebHandler

    class AliveThread:
        @staticmethod
        def is_alive():
            return True

    state_file = tmp_path / "auto_update_state.json"
    state_file.write_text(json.dumps({
        "status": "pending",
        "pending_update": {"version": "2099.01.02.2"},
    }), encoding="utf-8")

    monkeypatch.setattr(CAPIWebHandler, "_train_new_jobs", {})
    monkeypatch.setattr(CAPIWebHandler, "_train_new_jobs_lock", threading.Lock())
    monkeypatch.setattr(
        CAPIWebHandler,
        "_train_slot",
        {"lock": threading.Lock(), "active_job_id": None},
    )
    monkeypatch.setattr(
        CAPIWebHandler,
        "_retrain_state",
        {"lock": threading.Lock(), "job": None},
        raising=False,
    )
    monkeypatch.setattr(
        CAPIWebHandler,
        "_submodel_retrain_state",
        {"lock": threading.Lock(), "job": None},
    )

    if training_kind.startswith("patchcore_"):
        phase = training_kind.removeprefix("patchcore_")
        CAPIWebHandler._train_new_jobs["train-active"] = {
            "phase": phase,
            "thread": AliveThread(),
            "proc": None,
        }
    elif training_kind == "scratch":
        CAPIWebHandler._retrain_state["job"] = {
            "job_id": "scratch-active",
            "state": "running",
        }
    else:
        CAPIWebHandler._submodel_retrain_state["job"] = {
            "job_id": "submodel-active",
            "state": "running",
        }

    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.status_tracker = None
    captured = {}
    handler._send_json = lambda payload, status=200: captured.update({
        "payload": payload,
        "status": status,
    })
    monkeypatch.setattr(CAPIWebHandler, "_update_state_file", state_file)
    monkeypatch.setattr(capi_web.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("training server must not launch updater")
    ))

    handler._handle_api_update_apply()

    assert captured["status"] == 409
    assert "訓練工作進行中" in captured["payload"]["error"]


def test_manual_apply_is_not_blocked_by_training_review_waiting_for_user(
    tmp_path,
    monkeypatch,
):
    import capi_web
    from capi_web import CAPIWebHandler

    class FinishedThread:
        @staticmethod
        def is_alive():
            return False

    state_file = tmp_path / "update" / "auto_update_state.json"
    state_file.parent.mkdir()
    state_file.write_text(json.dumps({
        "status": "pending",
        "pending_update": {"version": "2099.01.02.2"},
    }), encoding="utf-8")

    monkeypatch.setattr(CAPIWebHandler, "_train_new_jobs", {
        "train-review": {
            "phase": "review",
            "thread": FinishedThread(),
            "proc": None,
        },
    })
    monkeypatch.setattr(CAPIWebHandler, "_train_new_jobs_lock", threading.Lock())
    monkeypatch.setattr(
        CAPIWebHandler,
        "_train_slot",
        {"lock": threading.Lock(), "active_job_id": None},
    )
    monkeypatch.setattr(
        CAPIWebHandler,
        "_retrain_state",
        {"lock": threading.Lock(), "job": None},
        raising=False,
    )
    monkeypatch.setattr(
        CAPIWebHandler,
        "_submodel_retrain_state",
        {"lock": threading.Lock(), "job": None},
    )

    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.status_tracker = None
    captured = {}
    handler._send_json = lambda payload, status=200: captured.update({
        "payload": payload,
        "status": status,
    })
    monkeypatch.setattr(CAPIWebHandler, "_update_state_file", state_file)
    monkeypatch.setattr(CAPIWebHandler, "_update_apply_lock", threading.Lock())
    monkeypatch.setattr(capi_web.subprocess, "Popen", lambda *args, **kwargs: object())

    handler._handle_api_update_apply()

    assert captured["status"] == 202


def test_manual_apply_rejects_stale_expected_version(tmp_path, monkeypatch):
    import capi_web
    from capi_web import CAPIWebHandler

    state_file = tmp_path / "auto_update_state.json"
    state_file.write_text(json.dumps({
        "status": "pending",
        "pending_update": {"version": "2099.01.02.3"},
    }), encoding="utf-8")

    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.status_tracker = None
    captured = {}
    handler._send_json = lambda payload, status=200: captured.update({
        "payload": payload,
        "status": status,
    })
    monkeypatch.setattr(CAPIWebHandler, "_update_state_file", state_file)
    monkeypatch.setattr(capi_web.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("stale request must not launch updater")
    ))

    handler._handle_api_update_apply(expected_version="2099.01.02.2")

    assert captured["status"] == 409
    assert "版本已變更" in captured["payload"]["error"]


def test_manual_apply_route_requires_admin_login():
    from capi_web import CAPIWebHandler

    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.path = "/api/update/apply"
    captured = {}

    def require_user(**kwargs):
        captured.update(kwargs)
        return None

    handler._require_settings_user = require_user
    handler._handle_api_update_apply = lambda: (_ for _ in ()).throw(
        AssertionError("unauthorized request must not apply an update")
    )

    handler.do_POST()

    assert captured == {"api": True, "admin": True}


def test_central_dashboard_update_route_requires_admin_login():
    from capi_web import CAPIWebHandler

    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.path = "/api/central-dashboard/update/apply"
    captured = {}

    def require_user(**kwargs):
        captured.update(kwargs)
        return None

    handler._require_settings_user = require_user
    handler._handle_api_central_dashboard_update_apply = lambda user: (_ for _ in ()).throw(
        AssertionError("unauthorized request must not proxy an update")
    )

    handler.do_POST()

    assert captured == {"api": True, "admin": True}


def test_central_dashboard_update_proxy_calls_trusted_device_endpoint(monkeypatch):
    import capi_web
    from capi_web import CAPIWebHandler

    class DB:
        @staticmethod
        def get_central_dashboard_config(_initial=None):
            return {
                "requestTimeoutSeconds": 8,
                "lines": [{
                    "id": "mod2-capi01",
                    "line": "CAPI01",
                    "apiUrl": "http://10.174.37.137/api/status",
                    "enabled": True,
                }],
            }

    calls = []

    class Response:
        status = 202

        @staticmethod
        def read():
            return json.dumps({
                "status": "apply_requested",
                "version": "2099.01.02.2",
            }).encode("utf-8")

    class Connection:
        def __init__(self, host, port=80, timeout=None):
            calls.append({"host": host, "port": port, "timeout": timeout})

        def request(self, method, path, body=None, headers=None):
            calls[-1].update({
                "method": method,
                "path": path,
                "body": body,
                "headers": dict(headers or {}),
            })

        @staticmethod
        def getresponse():
            return Response()

        @staticmethod
        def close():
            pass

    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.db = DB()
    handler._read_json_body = lambda: {
        "lineId": "mod2-capi01",
        "expectedVersion": "2099.01.02.2",
    }
    handler._detect_central_dashboard_webserver_ip = lambda: "10.174.37.81"
    responses = []
    handler._send_json = lambda payload, status=200, headers=None: responses.append(
        (status, payload)
    )
    monkeypatch.setattr(http.client, "HTTPConnection", Connection)

    handler._handle_api_central_dashboard_update_apply({"username": "Ray"})

    assert responses[-1] == (202, {
        "success": True,
        "lineId": "mod2-capi01",
        "status": "apply_requested",
        "version": "2099.01.02.2",
        "message": "更新程序已啟動，設備即將重新啟動",
    })
    assert calls[0]["host"] == "10.174.37.137"
    assert calls[0]["port"] == 80
    assert calls[0]["path"] == "/api/update/apply-central"
    assert calls[0]["method"] == "POST"
    assert json.loads(calls[0]["body"]) == {
        "expectedVersion": "2099.01.02.2",
    }
    assert calls[0]["headers"][capi_web.CENTRAL_UPDATE_AUTH_HEADER] == "1"
    assert urllib.parse.unquote(
        calls[0]["headers"][capi_web.CENTRAL_UPDATE_USER_HEADER]
    ) == "Ray"


def test_device_accepts_central_update_only_from_configured_center():
    import capi_web
    from capi_web import CAPIWebHandler

    body = json.dumps({"expectedVersion": "2099.01.02.2"}).encode("utf-8")
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.client_address = ("10.174.37.81", 52100)
    handler.headers = {
        "Content-Length": str(len(body)),
        capi_web.CENTRAL_UPDATE_AUTH_HEADER: "1",
        capi_web.CENTRAL_UPDATE_USER_HEADER: urllib.parse.quote("Ray"),
    }
    handler.rfile = io.BytesIO(body)
    handler._load_central_account_location = lambda: {
        "facility": "MOD2",
        "ip": "10.174.37.81",
    }
    captured = {}
    handler._handle_api_update_apply = lambda **kwargs: captured.update(kwargs)

    handler._handle_api_central_update_apply()

    assert captured == {
        "expected_version": "2099.01.02.2",
        "requested_by": "Ray",
        "request_source": "central_dashboard",
    }


def test_device_rejects_central_update_from_other_host():
    import capi_web
    from capi_web import CAPIWebHandler

    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.client_address = ("10.174.37.208", 52100)
    handler.headers = {capi_web.CENTRAL_UPDATE_AUTH_HEADER: "1"}
    handler._load_central_account_location = lambda: {
        "facility": "MOD2",
        "ip": "10.174.37.81",
    }
    captured = {}
    handler._send_json = lambda payload, status=200: captured.update({
        "payload": payload,
        "status": status,
    })
    handler._handle_api_update_apply = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("untrusted host must not apply update")
    )

    handler._handle_api_central_update_apply()

    assert captured["status"] == 403
    assert "中央更新來源" in captured["payload"]["error"]


def test_update_notice_is_rendered_in_both_dashboard_shells():
    from capi_web import CAPIWebHandler

    original_env = CAPIWebHandler.jinja_env
    CAPIWebHandler.jinja_env = None
    try:
        CAPIWebHandler.init_jinja()
        base = CAPIWebHandler.jinja_env.get_template("base.html").render(request_path="/")
        v3 = CAPIWebHandler.jinja_env.get_template("dashboard_v3.html").render(request_path="/v3/dashboard")
    finally:
        CAPIWebHandler.jinja_env = original_env

    for rendered in (base, v3):
        assert 'id="capi-update-notice"' in rendered
        assert "套用更新並重啟" in rendered
        assert "/api/update/status" in rendered
        assert "/api/update/apply" in rendered
        assert "訓練工作進行中" in rendered
