"""Post-panel maintenance must preserve completed inference output and locking."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import capi_server
from capi_config import CAPIConfig
from capi_inference import CAPIInferencer, ImageResult, TileInfo
from capi_server import CAPIServer


@pytest.fixture
def cuda(monkeypatch):
    mib = 1024 * 1024
    state = {"allocated": 3072, "reserved": 14336, "free": 1024,
             "clears": 0, "syncs": 0, "now": 100.0}
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: state["allocated"] * mib)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: state["reserved"] * mib)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (state["free"] * mib, 16384 * mib))
    monkeypatch.setattr(capi_server.time, "monotonic", lambda: state["now"])

    def synchronize():
        state["syncs"] += 1

    def clear():
        state["clears"] += 1
        # Leave pressure unchanged to exercise cooldown even on no-op cleanup.

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "empty_cache", clear)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats",
                        lambda: pytest.fail("maintenance must not reset diagnostics"))
    return state


def make_server(policy=None):
    server = CAPIServer.__new__(CAPIServer)
    server.server_config = {} if policy is None else {"inference": {"cuda_cache_cleanup": policy}}
    server._gpu_lock = threading.Lock()
    return server


def cleanup(server, glass="G1"):
    with server._gpu_lock:
        server._maybe_clear_cuda_cache_after_panel(glass_id=glass)


@pytest.mark.parametrize("unused,free,expected", [
    (2047, 1024, 0), (2048, 2048, 1), (4096, 2049, 0), (11264, 1024, 1),
])
def test_cleanup_requires_both_thresholds(cuda, unused, free, expected):
    cuda.update(reserved=cuda["allocated"] + unused, free=free)
    cleanup(make_server())
    assert cuda["clears"] == expected
    assert cuda["syncs"] == 2 * expected


def test_configurable_thresholds_and_zero_cooldown(cuda):
    server = make_server({"min_unused_mib": 512, "max_device_free_mib": 4096,
                          "cooldown_seconds": 0})
    cuda.update(reserved=4096, free=3000)
    cleanup(server)
    cleanup(server)
    assert cuda["clears"] == 2


def test_cooldown_is_shared_across_panels(cuda):
    server = make_server()
    cleanup(server, "G1")
    cuda["now"] += 59
    cleanup(server, "G2")
    assert cuda["clears"] == 1
    cuda["now"] += 1
    cleanup(server, "G3")
    assert cuda["clears"] == 2


def test_disabled_policy_does_not_touch_cuda(cuda, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: pytest.fail("disabled"))
    cleanup(make_server({"enabled": False}))
    assert cuda["clears"] == 0


def test_unused_cuda_is_not_initialized(cuda, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU service"))
    cleanup(make_server())
    assert cuda["clears"] == cuda["syncs"] == 0


@pytest.mark.parametrize("policy", [
    None, "invalid", {"enabled": "false"}, {"min_unused_mib": 0},
    {"max_device_free_mib": -1}, {"cooldown_seconds": float("nan")},
    {"min_unused_mib": float("inf")}, {"cooldown_seconds": "invalid"},
])
def test_invalid_policy_skips_cleanup_and_warns_once(cuda, caplog, policy):
    server = make_server()
    server.server_config = {"inference": {"cuda_cache_cleanup": policy}}
    cleanup(server)
    cleanup(server)
    assert cuda["clears"] == 0
    assert caplog.text.count("disabled by invalid config") == 1


def test_query_failure_is_nonfatal(cuda, monkeypatch, caplog):
    def fail():
        raise RuntimeError("query failed")

    monkeypatch.setattr(torch.cuda, "mem_get_info", fail)
    cleanup(make_server())
    assert cuda["clears"] == 0
    assert "query failed" in caplog.text


def test_failed_cleanup_is_throttled(cuda, monkeypatch, caplog):
    attempts = []

    def fail():
        attempts.append(True)
        raise RuntimeError("cache failed")

    monkeypatch.setattr(torch.cuda, "empty_cache", fail)
    server = make_server()
    cleanup(server)
    cleanup(server)
    assert attempts == [True]
    assert "cache failed" in caplog.text


@pytest.mark.parametrize("mode", ["disabled", "enabled", "cleanup_failure", "query_failure"])
@pytest.mark.parametrize("below_threshold", [False, True])
def test_request_preserves_scores_maps_and_judgment_under_gpu_lock(
        cuda, monkeypatch, tmp_path, mode, below_threshold):
    server = make_server({"enabled": mode != "disabled"})
    server.path_mapping = {}
    server.cpu_workers = 1
    events = []
    score = 0.499999 if below_threshold else 0.500001
    anomaly_map = np.array([[score, 0.125]], dtype=np.float32)
    expected_map = anomaly_map.copy()
    tile = TileInfo(tile_id=0, x=0, y=0, width=2, height=2,
                    image=np.zeros((2, 2, 3), dtype=np.uint8))
    tile.is_aoi_coord_below_threshold = below_threshold
    result = ImageResult(image_path=tmp_path / "W0F00000.tif", image_size=(2, 2),
                         otsu_bounds=(0, 0, 2, 2), exclusion_regions=[], tiles=[tile],
                         excluded_tile_count=0, processed_tile_count=1, processing_time=0.0)
    result.anomaly_tiles = [(tile, score, anomaly_map)]
    results = [result]
    expected_judgment = capi_server.aggregate_judgment(results)

    def process_panel(*args, **kwargs):
        assert server._gpu_lock.locked()
        events.append("inference")
        return results, None, False, "", False, None, {}

    aggregate = capi_server.aggregate_judgment

    def judgment(items):
        events.append("judgment")
        return aggregate(items)

    def clear():
        assert server._gpu_lock.locked()
        assert events == ["inference", "judgment"]
        events.append("cleanup")
        cuda["clears"] += 1
        if mode == "cleanup_failure":
            raise RuntimeError("cache failed")
        cuda.update(reserved=cuda["allocated"], free=12000)

    if mode == "query_failure":
        def fail_query():
            raise RuntimeError("query failed")
        monkeypatch.setattr(torch.cuda, "mem_get_info", fail_query)
    monkeypatch.setattr(torch.cuda, "empty_cache", clear)
    monkeypatch.setattr(capi_server, "aggregate_judgment", judgment)
    inferencer = SimpleNamespace(config=CAPIConfig(), process_panel=process_panel)
    server._get_or_create_inferencer = lambda model_id: inferencer
    server._evaluate_within_spec_for_inference = lambda *args: None
    response = server._process_request({"model_id": "GN160JCEL270S", "glass_id": "G1",
                                        "image_dir": str(tmp_path)})
    assert response[:2] == expected_judgment
    assert response[2] is results
    assert result.anomaly_tiles[0][0] is tile
    assert result.anomaly_tiles[0][1] == score
    assert result.anomaly_tiles[0][2] is anomaly_map
    np.testing.assert_array_equal(anomaly_map, expected_map)
    assert cuda["clears"] == int(mode in ("enabled", "cleanup_failure"))
    assert not server._gpu_lock.locked()


def test_failed_inference_does_not_run_cache_maintenance(cuda, monkeypatch, tmp_path):
    server = make_server()
    server.path_mapping = {}
    server.cpu_workers = 1

    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")

    server._get_or_create_inferencer = lambda model_id: SimpleNamespace(
        config=CAPIConfig(), process_panel=fail)
    monkeypatch.setattr(CAPIInferencer, "_clear_cuda_cache",
                        lambda stage: pytest.fail("failed inference must bypass maintenance"))
    response = server._process_request({"model_id": "GN160JCEL270S", "glass_id": "G1",
                                        "image_dir": str(tmp_path)})
    assert response[0].startswith("ERR:INFERENCE_FAILED")
    assert cuda["clears"] == 0
