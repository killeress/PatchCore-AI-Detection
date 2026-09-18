import threading
from types import SimpleNamespace

import pytest

from capi_inference import CAPIInferencer
from capi_server import CAPIServer


def test_memory_status_deduplicates_model_aliases(monkeypatch):
    model = object()
    inf = SimpleNamespace(
        _model_cache_v2={("machine", "light", "inner"): model},
        _inferencers={"same.pt": model, "failed.pt": None},
        inferencer=model,
        scratch_filter=object(),
    )
    server = SimpleNamespace(
        inferencers={"machine": inf, "alias": inf}, inferencer=inf,
        fallback_config=SimpleNamespace(machine_id="machine"),
    )
    samples = []
    monkeypatch.setattr(CAPIInferencer, "_log_cuda_memory",
                        lambda stage, **kwargs: samples.append((stage, kwargs)))
    CAPIServer._log_gpu_memory_status(server, "periodic")
    assert samples == [(
        "periodic machine=machine inferencers=1 patchcore_models=1 scratch_models=1",
        {"synchronize": False},
    )]


@pytest.mark.parametrize("config, expected_interval", [
    ({}, 300),
    ({"inference": {"cuda_memory_log_interval_seconds": 600}}, 600),
    ({"inference": {"cuda_memory_log_interval_seconds": "invalid"}}, 300),
    ({"inference": {"cuda_memory_log_interval_seconds": -1}}, 300),
])
def test_monitor_waits_between_samples_without_busy_polling(config, expected_interval):
    waits = []
    stages = []

    class FakeEvent:
        def clear(self):
            pass

        def wait(self, interval):
            waits.append(interval)
            return len(waits) > 2

    server = SimpleNamespace(
        server_config=config, _running=True,
        _cuda_memory_thread=None, _cuda_memory_stop_event=FakeEvent(),
        _log_gpu_memory_status=stages.append,
    )
    thread = CAPIServer._start_cuda_memory_monitor(server)
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert waits == [expected_interval] * 3
    assert stages == ["periodic", "periodic"]


def test_monitor_can_be_disabled():
    server = SimpleNamespace(
        server_config={"inference": {"cuda_memory_log_interval_seconds": 0}},
        _cuda_memory_thread=None, _cuda_memory_stop_event=threading.Event(),
    )
    assert CAPIServer._start_cuda_memory_monitor(server) is None
    assert server._cuda_memory_thread is None
