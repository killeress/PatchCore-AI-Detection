import json
import logging
import sys
from types import SimpleNamespace

import pytest

import capi_cuda_diagnostics as diagnostics


@pytest.fixture(autouse=True)
def fresh_recorder(monkeypatch):
    monkeypatch.setattr(diagnostics, "_recorder", diagnostics.FlightRecorder(growth_mib=5))


@pytest.fixture
def cuda(monkeypatch):
    monkeypatch.setenv("CAPI_CUDA_MEMORY_TRACE", "1")
    state = {"reserved": 4, "split": 1, "retries": 0, "oom": 0}
    fake = SimpleNamespace(
        is_initialized=lambda: True,
        memory_stats=lambda: {
            "inactive_split_bytes.all.current": state["split"] * 2**20,
            "num_alloc_retries": state["retries"], "num_ooms": state["oom"],
        },
        mem_get_info=lambda: (8 * 2**20, 16 * 2**20),
        memory_allocated=lambda: 2 * 2**20,
        memory_reserved=lambda: state["reserved"] * 2**20,
    )
    # Deliberately no synchronize / empty_cache / reset_peak_memory_stats APIs.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake))
    return state, fake


def records(caplog):
    return [json.loads(r.message.split("[CUDA-TRACE] ", 1)[1])
            for r in caplog.records if r.message.startswith("[CUDA-TRACE] {")]


def test_nested_trace_correlates_panel_and_deltas(cuda, caplog):
    state, _ = cuda

    @diagnostics.trace_call("panel")
    def process(panel_dir, glass_id=None, model_id=None):
        with diagnostics.cuda_stage("patchcore", image="W0F.tif", batch=1):
            state.update(reserved=10, split=3, retries=1)
        return "result"

    with caplog.at_level(logging.INFO, logger="capi.inference"):
        assert process("panel", glass_id="glass-1", model_id="model-A") == "result"
    rows = records(caplog)
    assert [r["phase"] for r in rows] == ["before", "before", "after", "after"]
    assert len({r["request"] for r in rows}) == 1
    assert all(r["glass"] == "glass-1" for r in rows)
    assert rows[2]["delta_reserved_mib"] == 6
    assert rows[2]["delta_inactive_split_mib"] == 2
    assert rows[2]["delta_allocation_retries_total"] == 1
    assert rows[1]["span"] == rows[2]["span"]
    assert "image" not in rows[3]
    assert diagnostics._context.get() == {}


def test_error_records_after_and_restores_context(cuda, caplog):
    with caplog.at_level(logging.INFO, logger="capi.inference"):
        with pytest.raises(RuntimeError, match="inference failed"):
            with diagnostics.cuda_stage("scratch", glass="g"):
                raise RuntimeError("inference failed")
    assert records(caplog)[-1]["outcome"] == "error"
    assert diagnostics._context.get() == {}


def test_snapshot_failure_does_not_break_work(cuda, monkeypatch, caplog):
    def fail():
        raise RuntimeError("CUDA unavailable")
    monkeypatch.setattr(cuda[1], "memory_stats", fail)
    with diagnostics.cuda_stage("panel"):
        pass
    assert "unavailable" in caplog.text
    assert diagnostics._context.get() == {}


@pytest.mark.parametrize("disabled", [True, False])
def test_disabled_or_uninitialized_does_not_query_cuda(cuda, monkeypatch, disabled):
    if disabled:
        monkeypatch.setenv("CAPI_CUDA_MEMORY_TRACE", "0")
    else:
        monkeypatch.setattr(cuda[1], "is_initialized", lambda: False)
    monkeypatch.setattr(cuda[1], "memory_stats", lambda: pytest.fail("must not query CUDA"))
    with diagnostics.cuda_stage("panel"):
        pass


def test_cached_prediction_does_not_claim_gpu_work(cuda, caplog):
    @diagnostics.trace_call("patchcore-tile")
    def predict(raw_prediction=None):
        return raw_prediction
    assert predict(raw_prediction=(1, None)) == (1, None)
    assert not records(caplog)


def test_real_batch_entry_reports_tail_batch_without_changing_results(monkeypatch, caplog):
    import numpy as np
    import torch
    from capi_inference import CAPIInferencer

    monkeypatch.setenv("CAPI_CUDA_MEMORY_TRACE", "1")
    diagnostics._recorder.remaining = 64
    diagnostics._recorder.deadline = float("inf")
    monkeypatch.setattr(diagnostics, "_snapshot", lambda: {
        "allocated_mib": 0, "reserved_mib": 0, "inactive_split_mib": 0,
        "allocation_retries_total": 0, "oom_total": 0,
    })

    class Model(torch.nn.Module):
        def forward(self, batch):
            return SimpleNamespace(pred_score=batch.mean(dim=(1, 2, 3)), anomaly_map=None)

    worker = CAPIInferencer.__new__(CAPIInferencer)
    worker.device = "cpu"
    worker._prepare_tile_tensor = lambda image: torch.from_numpy(image)
    tiles = [SimpleNamespace(image=np.full((3, 2, 2), i, dtype=np.float32)) for i in range(5)]
    model = SimpleNamespace(model=Model(), device="cpu", _cuda_trace_model_path="W0-inner.pt")
    with caplog.at_level(logging.INFO, logger="capi.inference"):
        result = worker._batch_forward(tiles, model, batch_size=4)
    assert result == [(float(i), None) for i in range(5)]
    before = [row for row in records(caplog) if row["phase"] == "before"]
    assert [row["batch"] for row in before] == [4, 1]
    assert [row["tensor_shape"] for row in before] == [[4, 3, 2, 2], [1, 3, 2, 2]]


def test_scratch_actual_forward_records_transformed_shape(monkeypatch, caplog):
    import numpy as np
    import torch
    from scratch_classifier import ScratchClassifier

    monkeypatch.setenv("CAPI_CUDA_MEMORY_TRACE", "1")
    diagnostics._recorder.remaining = 64
    diagnostics._recorder.deadline = float("inf")
    monkeypatch.setattr(diagnostics, "_snapshot", lambda: {
        "allocated_mib": 0, "reserved_mib": 0, "inactive_split_mib": 0,
        "allocation_retries_total": 0, "oom_total": 0,
    })
    worker = ScratchClassifier.__new__(ScratchClassifier)
    worker.metadata = SimpleNamespace(dinov2_model="test-dino")
    worker._device = "cpu"
    worker._transform = lambda image: torch.ones(3, 8, 8)
    worker._model = lambda batch: batch.mean(dim=(2, 3))
    worker._logreg = SimpleNamespace(predict_proba=lambda feats: np.tile([0.2, 0.8], (len(feats), 1)))
    with caplog.at_level(logging.INFO, logger="capi.inference"):
        result = worker.predict_batch([np.zeros((16, 16, 3), dtype=np.uint8)] * 2)
    assert np.allclose(result, [0.8, 0.8])
    forward = next(row for row in records(caplog) if row["stage"] == "scratch-forward")
    assert forward["batch"] == 2
    assert forward["tensor_shape"] == [2, 3, 8, 8]


def sample(reserved=3000, **extra):
    return dict(stage="tile", phase="after", reserved_mib=reserved,
                device_used_mib=reserved, device_free_mib=16000-reserved,
                **extra)


def test_normal_work_only_writes_panel_summary(caplog):
    recorder = diagnostics.FlightRecorder()
    with caplog.at_level(logging.INFO, logger="capi.inference"):
        recorder.accept(sample())
        recorder.accept(dict(sample(), stage="panel"))
    assert len(caplog.records) == 1
    assert caplog.records[0].message.startswith("[CUDA-SUMMARY]")
    assert len(recorder.history) == 2


def test_gradual_growth_flushes_prior_samples_and_bounded_after(caplog):
    recorder = diagnostics.FlightRecorder(post_records=2)
    with caplog.at_level(logging.INFO, logger="capi.inference"):
        for value in (3000, 3200, 3400, 3600, 3600, 3600, 3600):
            recorder.accept(sample(value))
    rows = records(caplog)
    assert [r["reserved_mib"] for r in rows] == [3000, 3200, 3400, 3600, 3600, 3600]
    assert [r["sequence"] for r in rows] == list(range(1, 7))
    assert len(recorder.history) == 1
    assert "reserved_growth" in caplog.text


def test_history_is_bounded_by_count_and_bytes(caplog):
    recorder = diagnostics.FlightRecorder(max_records=3, max_bytes=700)
    with caplog.at_level(logging.INFO, logger="capi.inference"):
        for index in range(100):
            recorder.accept(sample(glass=str(index)))
    assert len(recorder.history) <= 3
    assert recorder.history_bytes <= 700
    assert recorder.history_bytes == sum(size for _, size in recorder.history)
    assert not caplog.records


def test_high_usage_is_edge_triggered_with_hysteresis(caplog):
    recorder = diagnostics.FlightRecorder(growth_mib=100000, post_records=0)
    with caplog.at_level(logging.INFO, logger="capi.inference"):
        for value in (14000, 14000, 13500, 14000, 12000, 14000):
            recorder.accept(sample(value))
    triggers = [r for r in caplog.records if r.message.startswith("[CUDA-TRIGGER]")]
    assert len(triggers) == 2


@pytest.mark.parametrize("field", ["allocation_retries_total", "oom_total"])
def test_allocator_error_counter_triggers_once(field, caplog):
    recorder = diagnostics.FlightRecorder(post_records=0)
    with caplog.at_level(logging.INFO, logger="capi.inference"):
        recorder.accept(sample(**{field: 0}))
        recorder.accept(sample(**{field: 1}))
        recorder.accept(sample(**{field: 1}))
    assert len(records(caplog)) == 2
    assert field in caplog.text


def test_after_window_expires_without_extending_on_every_sample(monkeypatch, caplog):
    recorder = diagnostics.FlightRecorder(post_seconds=30)
    clock = [0]
    monkeypatch.setattr(diagnostics.time, "monotonic", lambda: clock[0])
    with caplog.at_level(logging.INFO, logger="capi.inference"):
        recorder.accept(sample())
        recorder.accept(sample(4000))
        clock[0] = 31
        recorder.accept(sample(4000))
    assert len(records(caplog)) == 2
    assert len(recorder.history) == 1


def test_snapshot_failure_flushes_prior_history(cuda, monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="capi.inference"):
        with diagnostics.cuda_stage("tile", glass="prior"):
            pass
        monkeypatch.setattr(cuda[1], "memory_stats", lambda: (_ for _ in ()).throw(RuntimeError("lost GPU")))
        with diagnostics.cuda_stage("tile", glass="failure"):
            pass
    rows = records(caplog)
    assert any(row.get("glass") == "prior" for row in rows)
    assert any("snapshot_error" in row for row in rows)
