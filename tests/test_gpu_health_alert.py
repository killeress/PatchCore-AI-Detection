from types import SimpleNamespace
import subprocess
import threading
from unittest.mock import MagicMock

import pytest

import capi_web
from capi_server import ServerStatusTracker


@pytest.mark.parametrize("failure,code", [
    (FileNotFoundError("nvidia-smi"), "tool_missing"),
    (subprocess.TimeoutExpired("nvidia-smi", 3), "query_timeout"),
    (PermissionError("denied"), "query_error"),
])
def test_gpu_probe_preserves_reason_without_raising(monkeypatch, failure, code):
    def run(*args, **kwargs):
        assert kwargs["timeout"] == 3
        raise failure

    monkeypatch.setattr(capi_web.subprocess, "run", run)
    result = capi_web._read_gpu_status()
    assert result["available"] is False
    assert result["error_code"] == code
    assert result["error"]


def test_gpu_probe_recognizes_capi36_device_handle_failure(monkeypatch):
    monkeypatch.setattr(capi_web.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=255,
        stdout="Unable to determine the device handle for GPU0: 0000:01:00.0: Unknown Error\nNo devices were found\n",
        stderr="",
    ))
    result = capi_web._read_gpu_status()
    health = capi_web._build_gpu_health(result, "auto", ["cuda"])
    assert health["active"] is True
    assert health["severity"] == "critical"
    assert health["state"] == "unavailable"
    assert "Unknown Error" in health["detail"]


@pytest.mark.parametrize("stdout", ["", "invalid result\n", ",1,2,3,4\n"])
def test_gpu_probe_does_not_report_malformed_output_as_healthy(monkeypatch, stdout):
    monkeypatch.setattr(capi_web.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout=stdout, stderr="",
    ))
    assert capi_web._read_gpu_status()["available"] is False


@pytest.mark.parametrize("gpu", [
    {"available": True},
    {"available": False, "error_code": "driver_unavailable"},
    {"available": False, "error_code": "tool_missing"},
])
def test_cpu_fallback_remains_visible_even_after_driver_recovers(gpu):
    health = capi_web._build_gpu_health(gpu, "auto", ["cpu"])
    assert health["active"] is True
    assert "CPU" in health["message"]
    assert "重新啟動 AI" in health["message"]
    assert health["severity"] == ("critical" if gpu.get("error_code") == "driver_unavailable" else "warning")


def test_explicit_cpu_mode_does_not_report_missing_gpu_as_fault():
    health = capi_web._build_gpu_health(
        {"available": False, "error_code": "tool_missing"}, "cpu", ["cpu"]
    )
    assert health["active"] is False
    assert health["state"] == "cpu"


def test_driver_probe_timeout_is_unknown_not_confirmed_gpu_loss():
    health = capi_web._build_gpu_health(
        {"available": False, "error_code": "query_timeout"}, "cuda", ["cuda"]
    )
    assert health["active"] is True
    assert health["severity"] == "warning"
    assert health["state"] == "unknown"


def test_gpu_probe_success_clears_driver_alert():
    failed = capi_web._build_gpu_health(
        {"available": False, "error_code": "driver_unavailable"}, "auto", ["cuda"]
    )
    recovered = capi_web._build_gpu_health({"available": True}, "auto", ["cuda"])
    assert failed["active"] is True
    assert recovered["active"] is False
    assert recovered["state"] == "healthy"


def test_first_fatal_cuda_error_survives_successful_hardware_probe():
    tracker = ServerStatusTracker()
    tracker.record_gpu_error(RuntimeError("[v2] tile: CUDA error: unspecified launch failure"))
    tracker.record_gpu_error(RuntimeError("CUDA error: an illegal memory access was encountered"))
    fault = tracker.get_status()["server"]["gpu_error"]
    assert "unspecified launch failure" in fault["message"]
    assert fault["detected_at"]
    health = capi_web._build_gpu_health({"available": True}, "auto", ["cuda"], fault)
    assert health["active"] is True
    assert health["state"] == "cuda_error"
    fault["message"] = "changed"
    assert tracker.get_status()["server"]["gpu_error"]["message"] != "changed"
    assert ServerStatusTracker().gpu_error is None  # A new process/session starts clean.


@pytest.mark.parametrize("message", ["image not found", "CUDA out of memory", "invalid model"])
def test_ordinary_errors_do_not_latch_fatal_gpu_fault(message):
    tracker = ServerStatusTracker()
    tracker.record_gpu_error(RuntimeError(message))
    assert tracker.gpu_error is None


def test_inference_failure_publishes_gpu_alert_without_changing_client_protocol(tmp_path, monkeypatch):
    import capi_server
    from capi_config import CAPIConfig

    tracker = ServerStatusTracker()
    monkeypatch.setattr(capi_server, "server_status", tracker)
    inferencer = SimpleNamespace(
        config=CAPIConfig(), device="cuda",
        process_panel=MagicMock(side_effect=RuntimeError("CUDA error: unspecified launch failure")),
    )
    server = capi_server.CAPIServer.__new__(capi_server.CAPIServer)
    server.path_mapping = {}
    server.cpu_workers = 1
    server._gpu_lock = threading.Lock()
    server._get_or_create_inferencer = lambda model_id: inferencer
    result = server._process_request({
        "model_id": "GN160JCEL270S", "glass_id": "G1", "image_dir": str(tmp_path),
    })
    assert result[0].startswith("ERR:INFERENCE_FAILED")
    assert result[2] == []
    assert "unspecified launch failure" in tracker.get_status()["server"]["gpu_error"]["message"]


def test_device_snapshot_uses_current_server_not_stale_web_inferencer():
    live = SimpleNamespace(device="cuda:0")
    server = SimpleNamespace(
        inference_config={"device": "auto"}, inferencer=live, inferencers={"M": live}
    )
    requested, devices = capi_web._inference_devices(server, SimpleNamespace(device="cpu"))
    assert requested == "auto"
    assert devices == ["cuda:0"]


def test_mixed_model_devices_still_report_cpu_fallback():
    server = SimpleNamespace(
        inference_config={"device": "auto"}, inferencer=None,
        inferencers={"A": SimpleNamespace(device="cuda"), "B": SimpleNamespace(device="cpu")},
    )
    requested, devices = capi_web._inference_devices(server)
    assert capi_web._build_gpu_health({"available": True}, requested, devices)["active"] is True


@pytest.mark.parametrize("device,expected", [("cpu", "CPU"), ("cuda", "GPU (RTX 5070 Ti)")])
def test_status_api_reports_actual_device_and_gpu_health(monkeypatch, device, expected):
    hardware = {"gpu": {"available": True, "name": "RTX 5070 Ti"}, "checked_at": "2026-09-18T15:00:00+08:00"}
    monkeypatch.setattr(capi_web, "_get_cached_hardware_status", lambda path: hardware)
    handler = object.__new__(capi_web.CAPIWebHandler)
    handler.status_tracker = ServerStatusTracker()
    handler.db = None
    handler._capi_server_instance = SimpleNamespace(
        inference_config={"device": "auto"}, inferencer=SimpleNamespace(device=device), inferencers={},
    )
    handler._get_update_status_payload = lambda: {}
    captured = {}
    handler._send_json = lambda payload, **kwargs: captured.update(payload)
    handler._handle_api_status()
    assert captured["server"]["device"] == expected
    assert captured["server"]["requested_device"] == "auto"
    assert captured["gpu_health"]["active"] is (device == "cpu")
    assert captured["gpu_health"]["checked_at"] == hardware["checked_at"]
    assert "gpu_health" not in hardware  # Do not mutate the shared hardware cache.
