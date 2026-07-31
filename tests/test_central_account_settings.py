import io
import json
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


def _settings_handler(*, facility="MOD2", params=None):
    handler = object.__new__(CAPIWebHandler)
    handler.db = _SettingsDB(params)
    handler.inferencer = None
    handler._capi_server_instance = SimpleNamespace(
        server_config={"mes_report": {"facility": facility}}
    )
    handler._current_settings_user = lambda: {"username": "tester"}
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


def test_settings_template_has_central_account_controls():
    template = (
        Path(__file__).resolve().parent.parent / "templates" / "settings.html"
    ).read_text(encoding="utf-8")

    assert "🏢 中心位置" in template
    assert 'id="central-account-facility"' in template
    assert 'id="central-account-ip"' in template
    assert "10.172.25.105" in template
    assert "10.174.37.81" in template
    assert "saveCentralAccountLocation()" in template
