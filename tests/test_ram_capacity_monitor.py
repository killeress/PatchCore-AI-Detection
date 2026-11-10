"""RAM capacity auditing survives restarts and runs without dashboard traffic."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
import sqlite3
import threading
from types import SimpleNamespace

import pytest

import capi_web
from capi_database import CAPIDatabase


HOST = "CAPI02"
BEFORE = "2026-11-10T10:00:00+08:00"
LAST_SEEN = "2026-11-10T10:00:30+08:00"
DETECTED = "2026-11-10T10:01:00+08:00"


def test_first_observation_is_baseline_and_existing_database_migrates(tmp_path):
    path = tmp_path / "existing.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE existing_data (value TEXT)")
        conn.execute("INSERT INTO existing_data VALUES ('preserved')")
    db = CAPIDatabase(path)
    assert db.get_ram_capacity_status(HOST)["latest_change"] is None
    state = db.record_ram_capacity_observation(HOST, 62.3, BEFORE)
    assert state["baseline_total_gb"] == state["current_total_gb"] == 62.3
    assert state["first_observed_at"] == state["last_observed_at"] == BEFORE
    assert state["latest_change"] is None
    assert db.get_ram_capacity_history(HOST) == []
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT value FROM existing_data").fetchone()[0] == "preserved"


def test_decrease_is_logged_once_with_detection_interval_and_survives_restart(tmp_path):
    path = tmp_path / "ram.db"
    db = CAPIDatabase(path)
    db.record_ram_capacity_observation(HOST, 62.3, BEFORE)
    db.record_ram_capacity_observation(HOST, 62.3, LAST_SEEN)
    # A missing DIMM will usually be discovered after a machine/service restart.
    db = CAPIDatabase(path)
    state = db.record_ram_capacity_observation(HOST, 30.3, DETECTED)
    event = state["latest_change"]
    assert event == {
        "id": 1, "hostname": HOST, "previous_total_gb": 62.3, "new_total_gb": 30.3,
        "last_seen_at": LAST_SEEN, "detected_at": DETECTED,
    }
    repeated = db.record_ram_capacity_observation(HOST, 30.3, "2026-11-10T10:01:30+08:00")
    assert repeated["latest_change"] == event
    assert repeated["baseline_total_gb"] == 62.3
    assert len(CAPIDatabase(path).get_ram_capacity_history(HOST)) == 1


def test_increase_and_restoration_keep_all_history_and_hosts_are_independent(tmp_path):
    db = CAPIDatabase(tmp_path / "ram.db")
    db.record_ram_capacity_observation(HOST, 62.3, BEFORE)
    db.record_ram_capacity_observation("CAPI03", 16, BEFORE)
    db.record_ram_capacity_observation(HOST, 30.3, LAST_SEEN)
    state = db.record_ram_capacity_observation(HOST, 62.3, DETECTED)
    assert state["latest_change"]["previous_total_gb"] == 30.3
    assert state["latest_change"]["new_total_gb"] == 62.3
    assert len(db.get_ram_capacity_history(HOST)) == 2
    assert len(db.get_ram_capacity_history(HOST, limit=1)) == 1
    assert db.get_ram_capacity_history("CAPI03") == []
    db.cleanup_old_records(vacuum=False)
    assert len(db.get_ram_capacity_history(HOST)) == 2


@pytest.mark.parametrize("total", [None, "", "bad", 0, -1, float("nan"), float("inf"), True])
def test_invalid_readings_do_not_replace_valid_capacity(tmp_path, total):
    db = CAPIDatabase(tmp_path / "ram.db")
    previous = db.record_ram_capacity_observation(HOST, 62.3, BEFORE)
    with pytest.raises(ValueError):
        db.record_ram_capacity_observation(HOST, total, DETECTED)
    assert db.get_ram_capacity_status(HOST) == previous


def test_concurrent_database_instances_do_not_duplicate_change(tmp_path):
    path = tmp_path / "ram.db"
    databases = [CAPIDatabase(path), CAPIDatabase(path)]
    databases[0].record_ram_capacity_observation(HOST, 62.3, BEFORE)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda index: databases[index % 2].record_ram_capacity_observation(HOST, 30.3, DETECTED),
            range(8),
        ))
    assert all(result["latest_change"]["id"] == 1 for result in results)
    assert len(databases[0].get_ram_capacity_history(HOST)) == 1


def test_monitor_throttles_and_failed_samples_keep_latest_change(monkeypatch, tmp_path):
    db = CAPIDatabase(tmp_path / "ram.db")
    now = [100.0]
    readings = iter([{"total_gb": 62.3}, {"total_gb": 30.3}, {}, {"total_gb": 30.3}])
    monkeypatch.setattr(capi_web, "_get_host_identity", lambda: HOST)
    monkeypatch.setattr(capi_web, "_read_memory_status", lambda: next(readings))
    monkeypatch.setattr(capi_web.time, "monotonic", lambda: now[0])
    monitor = capi_web._RAMCapacityMonitor(db)
    assert monitor.poll()["active"] is False
    now[0] = 129.0
    assert monitor.poll()["current_total_gb"] == 62.3
    now[0] = 130.0
    changed = monitor.poll()
    assert changed["active"] is True
    now[0] = 160.0
    failed = monitor.poll()
    assert failed["available"] is False
    assert failed["active"] is True
    assert failed["latest_change"] == changed["latest_change"]
    now[0] = 190.0
    assert monitor.poll()["available"] is True
    assert len(db.get_ram_capacity_history(HOST)) == 1


def test_database_failure_is_logged_and_does_not_break_monitor(monkeypatch, tmp_path, caplog):
    db = CAPIDatabase(tmp_path / "ram.db")
    db.record_ram_capacity_observation(HOST, 62.3, BEFORE)
    db.record_ram_capacity_observation(HOST, 30.3, DETECTED)
    monkeypatch.setattr(capi_web, "_get_host_identity", lambda: HOST)
    monkeypatch.setattr(capi_web, "_read_memory_status", lambda: {"total_gb": 30.3})
    monitor = capi_web._RAMCapacityMonitor(db)
    previous = monitor.poll()
    def fail(*args):
        raise sqlite3.OperationalError("database locked")
    monkeypatch.setattr(db, "record_ram_capacity_observation", fail)
    monitor._next_check = 0
    result = monitor.poll()
    assert result["active"] is True
    assert result["latest_change"] == previous["latest_change"]
    assert result["available"] is False
    assert "Failed to monitor RAM capacity" in caplog.text


def test_server_records_change_without_any_http_requests(monkeypatch, tmp_path):
    db = CAPIDatabase(tmp_path / "ram.db")
    memory = {"total_gb": 62.3}
    monkeypatch.setattr(capi_web, "_get_host_identity", lambda: HOST)
    monkeypatch.setattr(capi_web, "_read_memory_status", lambda: dict(memory))
    server = capi_web._RAMMonitoredHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler, db)
    assert db.get_ram_capacity_status(HOST)["current_total_gb"] == 62.3
    changed = threading.Event()
    original_record = db.record_ram_capacity_observation
    def record(*args):
        result = original_record(*args)
        changed.set()
        return result
    monkeypatch.setattr(db, "record_ram_capacity_observation", record)
    memory["total_gb"] = 30.3
    server.ram_capacity_monitor._next_check = 0
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        assert changed.wait(timeout=3), "Background sampling did not run"
        assert db.get_ram_capacity_history(HOST)[0]["new_total_gb"] == 30.3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_api_status_returns_persisted_capacity_alert(monkeypatch, tmp_path):
    db = CAPIDatabase(tmp_path / "ram.db")
    db.record_ram_capacity_observation(HOST, 62.3, BEFORE)
    db.record_ram_capacity_observation(HOST, 30.3, DETECTED)
    monkeypatch.setattr(capi_web, "_get_host_identity", lambda: HOST)
    monkeypatch.setattr(capi_web, "_read_memory_status", lambda: {"total_gb": 30.3})
    monkeypatch.setattr(capi_web, "_get_cached_hardware_status", lambda path: {"memory": {"total_gb": 30.3}})
    handler = object.__new__(capi_web.CAPIWebHandler)
    handler.db = db
    handler.server = SimpleNamespace(ram_capacity_monitor=capi_web._RAMCapacityMonitor(db))
    handler.status_tracker = SimpleNamespace(get_status=lambda: {"server": {"running": True}})
    captured = {}
    handler._send_json = lambda payload, **kwargs: captured.update(payload=payload, **kwargs)
    handler._handle_api_status()
    assert "error" not in captured["payload"]
    alert = captured["payload"]["ram_capacity_alert"]
    assert alert["available"] is True
    assert alert["active"] is True
    assert alert["latest_change"]["detected_at"] == DETECTED
    assert captured["headers"] == {"Access-Control-Allow-Origin": "*"}
