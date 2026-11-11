"""Confirmation clears one RAM notice while retaining its audit history."""
import json
import sqlite3
from datetime import datetime
from types import SimpleNamespace
from urllib.parse import unquote

import pytest

import capi_web
from capi_database import CAPIDatabase


HOST = "CAPI02"
CENTER = "10.174.37.81"


@pytest.fixture
def monitored(tmp_path, monkeypatch):
    db = CAPIDatabase(tmp_path / "ram.db")
    db.record_ram_capacity_observation(HOST, 62.3)
    db.record_ram_capacity_observation(HOST, 30.3)
    memory = {"total_gb": 30.3}
    monkeypatch.setattr(capi_web, "_get_host_identity", lambda: HOST)
    monkeypatch.setattr(capi_web, "_read_memory_status", lambda: dict(memory))
    monitor = capi_web._RAMCapacityMonitor(db)
    monitor.poll()
    return db, memory, monitor


def capture_handler(db=None):
    handler = object.__new__(capi_web.CAPIWebHandler)
    handler.db = db
    captured = {}
    handler._send_json = lambda payload, status=200, headers=None: captured.update(
        payload=payload, status=status, headers=headers,
    )
    return handler, captured


def test_old_ram_schema_migrates_without_losing_history(tmp_path):
    path = tmp_path / "existing.db"
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE ram_capacity_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, hostname TEXT NOT NULL,
            previous_total_gb REAL NOT NULL, new_total_gb REAL NOT NULL,
            last_seen_at TEXT NOT NULL, detected_at TEXT NOT NULL)""")
        conn.execute("INSERT INTO ram_capacity_events VALUES (1, ?, 62.3, 30.3, ?, ?)",
                     (HOST, "2026-11-10T10:00:00+08:00", "2026-11-11T10:00:00+08:00"))
    db = CAPIDatabase(path)
    event = db.get_ram_capacity_history(HOST)[0]
    assert event["id"] == 1
    assert event["previous_total_gb"] == 62.3
    assert event["acknowledged_by"] == event["acknowledged_at"] == ""
    confirmed = db.acknowledge_ram_capacity_event(HOST, 1, "現場人員")["latest_change"]
    assert confirmed["acknowledged_by"] == "現場人員"
    assert datetime.fromisoformat(confirmed["acknowledged_at"]).tzinfo is not None


def test_confirmation_is_immediate_persistent_and_new_change_alerts_again(monitored):
    db, memory, monitor = monitored
    original = monitor.poll()["latest_change"]
    result = monitor.acknowledge(original["id"], "Ray")
    assert result["active"] is False
    assert monitor.poll()["active"] is False  # Includes the cached API response.
    assert result["latest_change"]["detected_at"] == original["detected_at"]
    assert result["latest_change"]["acknowledged_by"] == "Ray"
    assert result["latest_change"]["acknowledged_at"]
    assert len(db.get_ram_capacity_history(HOST)) == 1
    reopened = CAPIDatabase(db.db_path)
    restarted = capi_web._RAMCapacityMonitor(reopened)
    assert restarted.poll()["active"] is False
    memory["total_gb"] = 62.3
    restarted._next_check = 0
    changed = restarted.poll()
    assert changed["active"] is True
    assert changed["latest_change"]["id"] > original["id"]
    assert changed["latest_change"]["acknowledged_at"] == ""
    history = reopened.get_ram_capacity_history(HOST)
    assert len(history) == 2
    assert history[1]["acknowledged_by"] == "Ray"


def test_stale_confirmation_only_acknowledges_displayed_event_and_is_idempotent(monitored):
    db, _, monitor = monitored
    old_id = monitor.poll()["latest_change"]["id"]
    db.record_ram_capacity_observation(HOST, 94.3)
    result = monitor.acknowledge(old_id, "First user")
    assert result["active"] is True
    assert result["latest_change"]["id"] != old_id
    confirmed = db.get_ram_capacity_history(HOST)[1]
    monitor.acknowledge(old_id, "Second user")
    assert db.get_ram_capacity_history(HOST)[1] == confirmed
    assert confirmed["acknowledged_by"] == "First user"
    with pytest.raises(KeyError):
        db.acknowledge_ram_capacity_event("OTHER-HOST", old_id, "Ray")
    for invalid in [True, None, 0, -1, 1.5, "1"]:
        with pytest.raises(ValueError):
            db.acknowledge_ram_capacity_event(HOST, invalid, "Ray")
    with pytest.raises(ValueError):
        db.acknowledge_ram_capacity_event(HOST, old_id, " ")


def test_history_pagination_is_stable_with_new_changes_and_other_hosts(monitored, monkeypatch):
    db, _, _ = monitored
    for total in [62.3, 30.3, 94.3]:
        db.record_ram_capacity_observation(HOST, total)
    handler, captured = capture_handler(db)
    handler.path = "/api/ram-capacity/history?limit=2"
    handler.do_GET()
    first = captured["payload"]
    assert captured["status"] == 200
    assert captured["headers"] == {"Access-Control-Allow-Origin": "*"}
    assert len(first["events"]) == 2
    db.record_ram_capacity_observation(HOST, 126.3)
    db.record_ram_capacity_observation("OTHER-HOST", 8)
    db.record_ram_capacity_observation("OTHER-HOST", 16)
    handler._handle_api_ram_capacity_history({"limit": ["2"], "beforeId": [str(first["nextBeforeId"])]})
    second = captured["payload"]
    assert second["nextBeforeId"] is None
    assert len(second["events"]) == 2
    assert {e["id"] for e in first["events"]}.isdisjoint(e["id"] for e in second["events"])
    assert all(event["hostname"] == HOST for event in second["events"])


@pytest.mark.parametrize("query", [{"limit": ["bad"]}, {"limit": ["0"]}, {"beforeId": ["-1"]}])
def test_history_errors_keep_cors_headers(query):
    handler, captured = capture_handler()
    handler._handle_api_ram_capacity_history(query)
    assert captured["status"] == 400
    assert captured["headers"]["Access-Control-Allow-Origin"] == "*"


def test_history_without_database_is_explicitly_unavailable():
    handler, captured = capture_handler()
    handler._handle_api_ram_capacity_history({})
    assert captured["status"] == 503
    assert captured["headers"]["Access-Control-Allow-Origin"] == "*"


def test_central_confirmation_route_requires_login():
    handler, _ = capture_handler()
    handler.path = "/api/central-dashboard/ram-capacity/acknowledge"
    requests = []
    handler._require_settings_user = lambda **kwargs: requests.append(kwargs)
    handler._handle_api_central_dashboard_ram_acknowledge = lambda user: pytest.fail("Unauthorized proxy")
    handler.do_POST()
    assert requests == [{"api": True}]


def test_device_confirms_requested_event_and_retains_history(monitored):
    db, _, monitor = monitored
    handler, captured = capture_handler(db)
    handler.path = capi_web.CENTRAL_RAM_ACK_PATH
    handler.client_address = (CENTER, 52000)
    handler._load_central_account_location = lambda: {"ip": CENTER}
    handler.headers = {capi_web.CENTRAL_RAM_ACK_HEADER: "1", capi_web.CENTRAL_UPDATE_USER_HEADER: "Ray"}
    event_id = monitor.poll()["latest_change"]["id"]
    handler._read_json_body = lambda: {"eventId": event_id}
    handler.server = SimpleNamespace(ram_capacity_monitor=monitor)
    handler.do_POST()
    assert captured["status"] == 200
    assert captured["payload"]["ram_capacity_alert"]["active"] is False
    assert db.get_ram_capacity_history(HOST)[0]["acknowledged_by"] == "Ray"
    handler._read_json_body = lambda: {"eventId": 99999}
    handler.do_POST()
    assert captured["status"] == 404
    assert len(db.get_ram_capacity_history(HOST)) == 1


@pytest.mark.parametrize("source,header,actor", [("10.174.37.99", "1", "Ray"), (CENTER, "", "Ray"), (CENTER, "1", "")])
def test_device_rejects_untrusted_confirmation(source, header, actor):
    handler, captured = capture_handler()
    handler.client_address = (source, 52000)
    handler._load_central_account_location = lambda: {"ip": CENTER}
    handler.headers = {capi_web.CENTRAL_RAM_ACK_HEADER: header, capi_web.CENTRAL_UPDATE_USER_HEADER: actor}
    handler._read_json_body = lambda: pytest.fail("Must reject before reading the event")
    handler._handle_api_central_ram_acknowledge()
    assert captured["status"] == 403


def proxy_handler(url="http://10.174.37.137/api/status", enabled=True):
    db = SimpleNamespace(get_central_dashboard_config=lambda initial: {
        "lines": [{"id": "line", "apiUrl": url, "enabled": enabled}],
    })
    handler, captured = capture_handler(db)
    handler._read_json_body = lambda: {"lineId": "line", "eventId": 7}
    handler._load_central_dashboard_file_config = lambda: None
    handler._detect_central_dashboard_webserver_ip = lambda: CENTER
    return handler, captured


def test_proxy_uses_configured_device_and_authenticated_actor(monkeypatch):
    handler, captured = proxy_handler()
    requests = []
    payload = {"success": True, "ram_capacity_alert": {"active": False, "latest_change": {"id": 7}}}
    class Connection:
        def __init__(self, host, port, timeout):
            requests.append({"host": host, "port": port})
        def request(self, method, path, **kwargs):
            requests[-1].update(method=method, path=path, **kwargs)
        def getresponse(self):
            return SimpleNamespace(status=200, read=lambda: json.dumps(payload).encode("utf-8"))
        def close(self):
            requests[-1]["closed"] = True
    monkeypatch.setattr(capi_web.http.client, "HTTPConnection", Connection)
    handler._handle_api_central_dashboard_ram_acknowledge({"username": "現場人員"})
    assert captured["status"] == 200
    assert captured["payload"] == payload
    assert requests[0]["host"] == "10.174.37.137"
    assert requests[0]["path"] == capi_web.CENTRAL_RAM_ACK_PATH
    assert json.loads(requests[0]["body"]) == {"eventId": 7}
    assert unquote(requests[0]["headers"][capi_web.CENTRAL_UPDATE_USER_HEADER]) == "現場人員"
    assert requests[0]["headers"][capi_web.CENTRAL_RAM_ACK_HEADER] == "1"
    assert requests[0]["closed"] is True


@pytest.mark.parametrize("url,enabled,status", [
    ("http://8.8.8.8/api/status", True, 400),
    ("file:///tmp/status", True, 400),
    ("http://10.172.25.105/api/status", True, 403),
    ("http://10.174.37.137/api/status", False, 404),
])
def test_proxy_rejects_invalid_or_disabled_targets(url, enabled, status):
    handler, captured = proxy_handler(url, enabled)
    handler._handle_api_central_dashboard_ram_acknowledge({"username": "Ray"})
    assert captured["status"] == status
