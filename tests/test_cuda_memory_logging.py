import logging
from types import SimpleNamespace

import anomalib.deploy

import capi_inference
from capi_inference import CAPIInferencer


def test_log_cuda_memory_reports_allocator_and_device_values(monkeypatch, caplog):
    mib = 1024 * 1024
    monkeypatch.setattr(capi_inference.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(capi_inference.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(capi_inference.torch.cuda, "mem_get_info", lambda: (4 * mib, 16 * mib))
    monkeypatch.setattr(capi_inference.torch.cuda, "memory_allocated", lambda: 2 * mib)
    monkeypatch.setattr(capi_inference.torch.cuda, "memory_reserved", lambda: 3 * mib)
    monkeypatch.setattr(capi_inference.torch.cuda, "max_memory_allocated", lambda: 5 * mib)
    monkeypatch.setattr(capi_inference.torch.cuda, "max_memory_reserved", lambda: 7 * mib)

    with caplog.at_level(logging.INFO, logger="capi.inference"):
        CAPIInferencer._log_cuda_memory("after-warmup model=test.pt")

    message = caplog.records[-1].getMessage()
    assert "[CUDA-MEM] after-warmup model=test.pt" in message
    assert "allocated=2.0 MiB" in message
    assert "reserved=3.0 MiB" in message
    assert "peak_allocated=5.0 MiB" in message
    assert "peak_reserved=7.0 MiB" in message
    assert "device_used=12.0 MiB" in message
    assert "device_free=4.0 MiB" in message


def test_clear_cuda_cache_reports_released_reserved_memory(monkeypatch, caplog):
    mib = 1024 * 1024
    reserved_values = iter((10 * mib, 4 * mib))
    empty_cache_calls = []
    monkeypatch.setattr(capi_inference.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(capi_inference.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(capi_inference.torch.cuda, "memory_reserved", lambda: next(reserved_values))
    monkeypatch.setattr(capi_inference.torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))

    with caplog.at_level(logging.INFO, logger="capi.inference"):
        CAPIInferencer._clear_cuda_cache("model=test.pt")

    assert empty_cache_calls == [True]
    message = caplog.records[-1].getMessage()
    assert "[CUDA-MEM] cache-clear model=test.pt" in message
    assert "reserved_before=10.0 MiB" in message
    assert "reserved_after=4.0 MiB" in message
    assert "released=6.0 MiB" in message


def test_clear_cuda_cache_failure_is_logged_without_raising(monkeypatch, caplog):
    monkeypatch.setattr(capi_inference.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(capi_inference.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(capi_inference.torch.cuda, "memory_reserved", lambda: 10)

    def fail_empty_cache():
        raise RuntimeError("cache failure")

    monkeypatch.setattr(capi_inference.torch.cuda, "empty_cache", fail_empty_cache)

    with caplog.at_level(logging.WARNING, logger="capi.inference"):
        CAPIInferencer._clear_cuda_cache("model=test.pt")

    assert "cache-clear model=test.pt failed: cache failure" in caplog.records[-1].getMessage()


def test_torch_model_load_logs_before_after_and_warmup_memory(monkeypatch, tmp_path):
    class FakeTorchInferencer:
        def __init__(self, path, device):
            self.path = path
            self.device = device

        def predict(self, image):
            return object()

    model_path = tmp_path / "sample.pt"
    model_path.write_bytes(b"test")

    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.device = "cuda"
    inferencer.config = SimpleNamespace(tile_size=8)
    stages = []
    cache_clears = []
    inferencer._log_cuda_memory = stages.append
    inferencer._clear_cuda_cache = cache_clears.append
    inferencer._fix_legacy_precision = lambda _model: None
    inferencer._optimize_model_fp16 = lambda _model: None

    monkeypatch.setattr(anomalib.deploy, "TorchInferencer", FakeTorchInferencer)
    monkeypatch.setattr(capi_inference.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(capi_inference.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(capi_inference.torch.cuda, "reset_peak_memory_stats", lambda: None)

    result = inferencer._load_model_from_path(model_path)

    assert isinstance(result, FakeTorchInferencer)
    assert stages == [
        "before-load model=sample.pt",
        "after-load model=sample.pt",
        "after-warmup model=sample.pt",
        "after-cache-clear model=sample.pt",
    ]
    assert cache_clears == ["model=sample.pt"]
