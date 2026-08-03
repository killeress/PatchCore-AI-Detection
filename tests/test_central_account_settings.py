import io
import http.client
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from capi_web import (
    CENTRAL_ACCOUNT_LOCATION_PARAM,
    CAPIWebHandler,
    _default_central_account_location,
    _normalize_central_account_location,
)


class _SettingsDB:
    def __init__(self, params=None):
        self.params = list(params or [])
        self.updated = []

    def get_all_config_params(self):
        return list(self.params)

    def update_config_param(self, param_name, new_value, reason, changed_by=""):
        self.updated.append((param_name, new_value, reason, changed_by))
        return True


class _LoginDB:
    def __init__(self, *, users=None, credentials=None, location=None):
        self.users = dict(users or {})
        self.credentials = dict(credentials or {})
        self.location = location or {
            "facility": "MOD2",
            "ip": "10.174.37.81",
        }
        self.lookup_count = 0

    def verify_settings_user(self, username, password):
        return self.credentials.get((username, password))

    def get_settings_user_by_username(self, username):
        self.lookup_count += 1
        return self.users.get(username)

    def get_config_param(self, param_name):
        if param_name != CENTRAL_ACCOUNT_LOCATION_PARAM:
            return None
        return {
            "param_name": param_name,
            "decoded_value": dict(self.location),
        }


class _CentralResponse:
    def __init__(self, status, payload):
        self.status = status
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body


def _mock_central_http(monkeypatch, *responses):
    queued = list(responses)
    calls = []

    class _Connection:
        def __init__(self, host, port=80, timeout=None):
            self.host = host
            self.port = port
            self.timeout = timeout
            self.response = queued.pop(0)

        def request(self, method, path, body=None, headers=None):
            calls.append({
                "host": self.host,
                "port": self.port,
                "timeout": self.timeout,
                "method": method,
                "path": path,
                "body": body,
                "headers": dict(headers or {}),
            })
            if isinstance(self.response, Exception):
                raise self.response

        def getresponse(self):
            return self.response

        def close(self):
            pass

    monkeypatch.setattr(http.client, "HTTPConnection", _Connection)
    return calls


def _login_handler(db, *, username="Ray", password="pw", headers=None):
    handler = object.__new__(CAPIWebHandler)
    handler.db = db
    handler.headers = dict(headers or {})
    handler._capi_server_instance = SimpleNamespace(
        server_config={"mes_report": {"facility": "MOD2"}}
    )
    handler._settings_sessions = {}
    handler._settings_session_lock = threading.Lock()
    handler._read_json_body = lambda: {
        "username": username,
        "password": password,
        "next": "/settings",
    }
    responses = []
    handler._send_json = lambda payload, status=200, headers=None: responses.append(
        (status, payload, dict(headers or {}))
    )
    return handler, responses


def _settings_handler(*, facility="MOD2", params=None, can_manage_accounts=True):
    handler = object.__new__(CAPIWebHandler)
    handler.db = _SettingsDB(params)
    handler.inferencer = None
    handler._capi_server_instance = SimpleNamespace(
        server_config={"mes_report": {"facility": facility}}
    )
    handler._current_settings_user = lambda: {
        "username": "tester",
        "can_manage_accounts": can_manage_accounts,
    }
    responses = []
    handler._send_json = lambda payload, status=200, headers=None: responses.append(
        (status, payload)
    )
    return handler, responses


@pytest.mark.parametrize(
    ("facility", "expected"),
    [
        ("MOD1", {"facility": "MOD1", "ip": "10.172.25.105"}),
        ("MOD2", {"facility": "MOD2", "ip": "10.174.37.81"}),
        ("unknown", {"facility": "MOD2", "ip": "10.174.37.81"}),
    ],
)
def test_default_central_account_location_uses_server_facility(facility, expected):
    assert _default_central_account_location(
        {"mes_report": {"facility": facility}}
    ) == expected


def test_normalize_central_account_location_accepts_edited_ipv4():
    assert _normalize_central_account_location(
        {"facility": "mod1", "ip": "10.172.25.200"}
    ) == {"facility": "MOD1", "ip": "10.172.25.200"}


@pytest.mark.parametrize(
    "payload",
    [
        {"facility": "MOD3", "ip": "10.172.25.105"},
        {"facility": "MOD1", "ip": "999.172.25.105"},
        {"facility": "MOD1", "ip": ""},
        "MOD1",
    ],
)
def test_normalize_central_account_location_rejects_invalid_values(payload):
    with pytest.raises(ValueError):
        _normalize_central_account_location(payload)


def test_settings_api_exposes_virtual_location_with_mod1_default():
    handler, responses = _settings_handler(facility="MOD1")

    handler._handle_api_settings()

    assert responses[0][0] == 200
    params = responses[0][1]["params"]
    location = next(
        item for item in params
        if item["param_name"] == CENTRAL_ACCOUNT_LOCATION_PARAM
    )
    assert json.loads(location["param_value"]) == {
        "facility": "MOD1",
        "ip": "10.172.25.105",
    }
    assert location["param_type"] == "dict"


def test_settings_api_preserves_stored_edited_location():
    edited = {"facility": "MOD2", "ip": "10.174.37.88"}
    handler, responses = _settings_handler(params=[{
        "param_name": CENTRAL_ACCOUNT_LOCATION_PARAM,
        "param_value": json.dumps(edited),
        "param_type": "dict",
        "description": "",
        "decoded_value": edited,
    }])

    handler._handle_api_settings()

    location = next(
        item for item in responses[0][1]["params"]
        if item["param_name"] == CENTRAL_ACCOUNT_LOCATION_PARAM
    )
    assert json.loads(location["param_value"]) == edited


def test_settings_update_normalizes_and_records_central_account_location():
    handler, responses = _settings_handler()
    payload = json.dumps({
        "param_name": CENTRAL_ACCOUNT_LOCATION_PARAM,
        "new_value": {"facility": "mod2", "ip": "10.174.37.99"},
        "reason": "測試修改中心",
    }).encode("utf-8")
    handler.headers = {"Content-Length": str(len(payload))}
    handler.rfile = io.BytesIO(payload)

    handler._handle_api_settings_update()

    assert responses[-1][0] == 200
    assert handler.db.updated == [(
        CENTRAL_ACCOUNT_LOCATION_PARAM,
        {"facility": "MOD2", "ip": "10.174.37.99"},
        "測試修改中心",
        "tester",
    )]


def test_settings_update_rejects_invalid_central_account_ip():
    handler, responses = _settings_handler()
    payload = json.dumps({
        "param_name": CENTRAL_ACCOUNT_LOCATION_PARAM,
        "new_value": {"facility": "MOD2", "ip": "10.174.37.999"},
        "reason": "測試錯誤 IP",
    }).encode("utf-8")
    handler.headers = {"Content-Length": str(len(payload))}
    handler.rfile = io.BytesIO(payload)

    handler._handle_api_settings_update()

    assert responses[-1] == (400, {"error": "請輸入有效的 IPv4 位址"})
    assert handler.db.updated == []


def test_settings_update_rejects_central_account_location_for_non_admin():
    handler, responses = _settings_handler(can_manage_accounts=False)
    payload = json.dumps({
        "param_name": CENTRAL_ACCOUNT_LOCATION_PARAM,
        "new_value": {"facility": "MOD2", "ip": "10.174.37.99"},
        "reason": "測試一般帳號修改",
    }).encode("utf-8")
    handler.headers = {"Content-Length": str(len(payload))}
    handler.rfile = io.BytesIO(payload)

    handler._handle_api_settings_update()

    assert responses[-1] == (
        403,
        {"error": "只有 admin 可以修改中心位置"},
    )
    assert handler.db.updated == []


def test_settings_login_uses_configured_central_account_when_local_user_missing(
    monkeypatch,
):
    central_user = {
        "id": 7,
        "username": "Ray",
        "is_admin": False,
        "can_manage_accounts": False,
        "created_at": "2026-07-31 10:00:00",
        "updated_at": "2026-07-31 10:00:00",
    }
    calls = _mock_central_http(
        monkeypatch,
        _CentralResponse(200, {"success": True, "user": central_user}),
    )
    db = _LoginDB()
    handler, responses = _login_handler(db)

    handler._handle_api_settings_login()

    assert responses[-1][0] == 200
    assert responses[-1][1]["user"] == {
        **central_user,
        "auth_source": "central",
        "central_facility": "MOD2",
    }
    assert calls[0]["host"] == "10.174.37.81"
    assert calls[0]["path"] == "/api/settings/central-auth"
    assert calls[0]["headers"]["X-CAPI-Central-Auth"] == "1"
    assert json.loads(calls[0]["body"]) == {
        "username": "Ray",
        "password": "pw",
    }

    cookie = responses[-1][2]["Set-Cookie"]
    token = cookie.split("=", 1)[1].split(";", 1)[0]
    assert "password" not in handler._settings_sessions[token]
    assert "password" not in handler._settings_sessions[token]["user"]
    handler.headers = {"Cookie": f"capi_settings_session={token}"}
    assert handler._current_settings_user()["username"] == "Ray"
    assert db.lookup_count == 1


def test_settings_login_keeps_existing_local_username_authoritative(monkeypatch):
    local_user = {
        "id": 2,
        "username": "Ray",
        "is_admin": False,
        "can_manage_accounts": False,
        "created_at": "",
        "updated_at": "",
    }
    db = _LoginDB(users={"Ray": local_user})
    handler, responses = _login_handler(db, password="wrong")

    def _unexpected_connection(*args, **kwargs):
        raise AssertionError("本機已有同名帳號時不應查詢中央")

    monkeypatch.setattr(http.client, "HTTPConnection", _unexpected_connection)

    handler._handle_api_settings_login()

    assert responses[-1][:2] == (401, {"error": "帳號或密碼錯誤"})


def test_settings_login_falls_back_to_legacy_central_login_endpoint(monkeypatch):
    central_user = {
        "id": 8,
        "username": "Ray",
        "is_admin": False,
        "can_manage_accounts": False,
        "created_at": "",
        "updated_at": "",
    }
    calls = _mock_central_http(
        monkeypatch,
        _CentralResponse(404, {"error": "Page Not Found"}),
        _CentralResponse(200, {"success": True, "user": central_user}),
    )
    handler, responses = _login_handler(_LoginDB())

    handler._handle_api_settings_login()

    assert responses[-1][0] == 200
    assert [call["path"] for call in calls] == [
        "/api/settings/central-auth",
        "/api/settings/login",
    ]


def test_settings_login_reports_central_service_unavailable(monkeypatch):
    _mock_central_http(monkeypatch, OSError("connection timed out"))
    handler, responses = _login_handler(_LoginDB())

    handler._handle_api_settings_login()

    assert responses[-1][:2] == (
        503,
        {"error": "中心帳號服務無法連線，請改用本機帳號或聯絡管理員"},
    )


def test_forwarded_settings_login_does_not_query_central_again(monkeypatch):
    handler, responses = _login_handler(
        _LoginDB(),
        headers={"X-CAPI-Central-Auth": "1"},
    )

    def _unexpected_connection(*args, **kwargs):
        raise AssertionError("中央轉送登入不可再次轉送")

    monkeypatch.setattr(http.client, "HTTPConnection", _unexpected_connection)

    handler._handle_api_settings_login()

    assert responses[-1][:2] == (401, {"error": "帳號或密碼錯誤"})


def test_central_auth_endpoint_verifies_only_local_account():
    local_user = {
        "id": 7,
        "username": "Ray",
        "is_admin": False,
        "can_manage_accounts": False,
        "created_at": "",
        "updated_at": "",
    }
    db = _LoginDB(
        users={"Ray": local_user},
        credentials={("Ray", "pw"): local_user},
    )
    handler, responses = _login_handler(
        db,
        headers={"X-CAPI-Central-Auth": "1"},
    )

    handler._handle_api_settings_central_auth()

    assert responses[-1][:2] == (
        200,
        {"success": True, "user": local_user},
    )
    assert handler._settings_sessions == {}


def test_central_auth_endpoint_rejects_normal_browser_request():
    handler, responses = _login_handler(_LoginDB())

    handler._handle_api_settings_central_auth()

    assert responses[-1][:2] == (
        403,
        {"error": "不允許的驗證來源"},
    )


def test_settings_template_integrates_central_location_into_accounts():
    template = (
        Path(__file__).resolve().parent.parent / "templates" / "settings.html"
    ).read_text(encoding="utf-8")
    accounts_renderer = template.split(
        "function renderAccountsPane(centralAccountLocation)", 1
    )[1].split("async function loadMarkCalibration()", 1)[0]

    assert 'data-target="central-account"' not in template
    assert 'id="pane-central-account"' not in template
    assert 'id="pane-accounts"' in accounts_renderer
    assert 'id="central-account-facility"' in accounts_renderer
    assert 'id="central-account-ip"' in accounts_renderer
    assert "10.172.25.105" in accounts_renderer
    assert "10.174.37.81" in accounts_renderer
    assert "saveCentralAccountLocation()" in accounts_renderer
    assert "renderWithinSpecPane(withinSpecParam, centralAccountLocation)" in template
    assert "function renderWithinSpecPane(param, centralAccountLocation)" in template
    assert "${renderAccountsPane(centralAccountLocation)}" in template
