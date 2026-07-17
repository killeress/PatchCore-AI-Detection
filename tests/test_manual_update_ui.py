import json
import threading


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
