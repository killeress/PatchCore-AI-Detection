import sqlite3

import capi_web
from capi_web import CAPIWebHandler


class _StatusTracker:
    def get_status(self):
        return {
            "server": {"running": True, "threshold_mapping": {}},
            "traffic": {},
            "stats": {},
        }


class _Database:
    db_path = "missing/status.db"

    def get_shift_statistics(self):
        return {
            "total": 280,
            "ok_count": 189,
            "ng_count": 22,
            "aoi_ng_count": 64,
            "err_count": 0,
            "avg_time": 1.625,
            "overexposed_count": 7,
            "shift_name": "白班",
            "time_range": "07/24 07:30 ~ 07/24 19:30",
        }

    def get_active_model_bundle(self):
        return None


def test_gpu_status_parses_nvidia_smi_output(monkeypatch):
    class _Completed:
        returncode = 0
        stdout = "NVIDIA RTX A4000, 42, 7373, 16384, 58\n"

    monkeypatch.setattr(capi_web.subprocess, "run", lambda *args, **kwargs: _Completed())

    result = capi_web._read_gpu_status()

    assert result == {
        "available": True,
        "name": "NVIDIA RTX A4000",
        "utilization_percent": 42.0,
        "vram_used_gb": 7.2,
        "vram_total_gb": 16.0,
        "temperature_c": 58.0,
    }


def test_hardware_status_is_cached_for_30_seconds(monkeypatch, tmp_path):
    calls = []
    times = iter([100.0, 110.0, 131.0])

    def collect(path):
        calls.append(path)
        return {"sample": len(calls)}

    capi_web._hardware_status_cache.clear()
    monkeypatch.setattr(capi_web, "_collect_hardware_status", collect)
    monkeypatch.setattr(capi_web.time, "monotonic", lambda: next(times))

    first = capi_web._get_cached_hardware_status(tmp_path)
    second = capi_web._get_cached_hardware_status(tmp_path)
    third = capi_web._get_cached_hardware_status(tmp_path)
    capi_web._hardware_status_cache.clear()

    assert first == second == {"sample": 1}
    assert third == {"sample": 2}
    assert len(calls) == 2


def test_api_status_includes_shift_and_hardware_metrics(monkeypatch):
    hardware = {
        "gpu": {
            "available": True,
            "name": "NVIDIA RTX A4000",
            "utilization_percent": 42.0,
            "vram_used_gb": 7.2,
            "vram_total_gb": 16.0,
            "temperature_c": 58.0,
        },
        "memory": {"used_gb": 19.5, "total_gb": 32.0, "used_percent": 60.9},
        "disk": {"free_gb": 182.4, "total_gb": 500.0},
    }
    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("missing")),
    )
    monkeypatch.setattr(capi_web, "_get_cached_hardware_status", lambda path: hardware)
    monkeypatch.setattr(capi_web, "_get_host_identity", lambda: "CAPI34")

    handler = object.__new__(CAPIWebHandler)
    handler.status_tracker = _StatusTracker()
    handler.db = _Database()
    handler.heatmap_base_dir = None
    captured = {}

    def capture(payload, status=200, headers=None):
        captured.update({"payload": payload, "status": status, "headers": headers})

    handler._send_json = capture
    handler._handle_api_status()

    stats = captured["payload"]["stats"]
    assert stats["avg_time"] == 1.625
    assert stats["overexposed_count"] == 7
    assert stats["aoi_ng_count"] == 64
    assert stats["ai_ng_count"] == 22
    assert captured["payload"]["server"]["hostname"] == "CAPI34"
    assert captured["payload"]["hardware"] == hardware
    assert captured["headers"] == {"Access-Control-Allow-Origin": "*"}
