"""Central dashboard training activity follows live workers, not saved history."""
import json
import threading
from types import SimpleNamespace

import pytest

import capi_web
from capi_database import CAPIDatabase
from capi_web import CAPIWebHandler


@pytest.fixture
def handler(monkeypatch):
    for name in ("_retrain_state", "_submodel_retrain_state"):
        monkeypatch.setattr(CAPIWebHandler, name, {"lock": threading.Lock(), "job": None}, raising=False)
    monkeypatch.setattr(CAPIWebHandler, "_train_new_jobs", {})
    monkeypatch.setattr(CAPIWebHandler, "_train_new_jobs_lock", threading.Lock())
    monkeypatch.setattr(CAPIWebHandler, "_train_slot", {"lock": threading.Lock(), "active_job_id": None})
    result = object.__new__(CAPIWebHandler)
    result.db = None
    return result


def _runtime(phase="train", alive=True, proc_alive=False):
    return {
        "phase": phase,
        "thread": SimpleNamespace(is_alive=lambda: alive),
        "proc": SimpleNamespace(poll=lambda: None if proc_alive else 0),
    }


def test_idle_and_saved_jobs_do_not_imply_active_training(handler, tmp_path):
    db = CAPIDatabase(tmp_path / "training.db")
    db.create_training_job("stale", "MODEL-A", [])
    db.update_training_job_state("stale", "train")
    handler.db = db

    assert handler._get_training_status_payload() == {"active": False, "jobs": []}
    assert db.get_training_job("stale")["state"] == "train"  # Summary is read-only.


@pytest.mark.parametrize("phase,alive,proc_alive,expected", [
    ("train", True, False, True),
    ("train", False, True, True),
    ("preprocess", True, False, True),
    ("review", True, False, False),
    ("train", False, False, False),
    ("preprocess", False, False, False),
])
def test_only_live_training_or_preprocessing_is_reported(handler, phase, alive, proc_alive, expected):
    CAPIWebHandler._train_new_jobs["job"] = _runtime(phase, alive, proc_alive)
    payload = handler._get_training_status_payload()
    assert payload["active"] is expected
    assert len(payload["jobs"]) == int(expected)


def test_active_jobs_include_labels_without_private_training_data(handler, tmp_path):
    db = CAPIDatabase(tmp_path / "training.db")
    db.create_training_job("full", "MODEL-A", ["/private/panel"], training_params={"private": "value"})
    db.update_training_job_state("full", "train")
    handler.db = db
    CAPIWebHandler._train_new_jobs["full"] = _runtime()
    CAPIWebHandler._train_slot["active_job_id"] = "full"

    payload = handler._get_training_status_payload()
    assert payload == {"active": True, "jobs": [{
        "kind": "patchcore", "job_id": "full", "machine_id": "MODEL-A",
        "phase": "train", "unit": "",
    }]}
    assert "private" not in json.dumps(payload)

    db.update_training_job_state("full", "completed")
    assert handler._get_training_status_payload() == {"active": False, "jobs": []}


def test_partial_retraining_and_concurrent_preparation(handler):
    handler.db = SimpleNamespace(get_training_job=lambda job_id: {
        "machine_id": job_id, "training_scope": {"mode": "partial" if job_id == "partial" else "full"},
    })
    CAPIWebHandler._train_new_jobs.update({
        "partial": _runtime(), "prepare": _runtime("preprocess"), "review": _runtime("review"),
    })
    payload = handler._get_training_status_payload()
    assert [(job["kind"], job["phase"]) for job in payload["jobs"]] == [
        ("patchcore_partial", "train"), ("patchcore", "preprocess"),
    ]


def test_reserved_slot_is_visible_before_worker_start(handler):
    CAPIWebHandler._train_slot["active_job_id"] = "starting"
    assert handler._get_training_status_payload()["jobs"][0]["job_id"] == "starting"


@pytest.mark.parametrize("state_name,kind", [
    ("_submodel_retrain_state", "patchcore_submodel"), ("_retrain_state", "scratch"),
])
def test_retraining_lifecycle_and_no_duplicate_slot(handler, state_name, kind):
    job = {"job_id": "retrain", "state": "running", "step": "train", "log_lines": ["private"]}
    getattr(CAPIWebHandler, state_name)["job"] = job
    CAPIWebHandler._train_slot["active_job_id"] = "retrain"
    CAPIWebHandler._train_new_jobs["retrain"] = _runtime()
    payload = handler._get_training_status_payload()
    assert len(payload["jobs"]) == 1
    assert payload["jobs"][0]["kind"] == kind
    assert "log_lines" not in payload["jobs"][0]

    CAPIWebHandler._train_slot["active_job_id"] = None
    CAPIWebHandler._train_new_jobs.clear()
    for terminal in ("completed", "failed"):
        job["state"] = terminal
        assert handler._get_training_status_payload() == {"active": False, "jobs": []}


def test_existing_submodel_retrain_reads_machine_from_bundle(handler):
    handler.db = SimpleNamespace(get_model_bundle=lambda bundle_id: {"machine_id": "MODEL-B"})
    CAPIWebHandler._submodel_retrain_state["job"] = {
        "state": "running", "bundle_id": 7, "lighting": "W", "zone": "inner", "step": "stage",
    }
    job = handler._get_training_status_payload()["jobs"][0]
    assert (job["machine_id"], job["unit"], job["phase"]) == ("MODEL-B", "W-inner", "stage")


def test_metadata_failure_keeps_live_activity_visible(handler):
    def unavailable(job_id):
        raise RuntimeError("database temporarily unavailable")
    handler.db = SimpleNamespace(get_training_job=unavailable)
    CAPIWebHandler._train_new_jobs["job"] = _runtime()
    payload = handler._get_training_status_payload()
    assert payload["active"] is True
    assert payload["jobs"][0]["machine_id"] == ""


def test_status_api_exposes_training_with_cors_even_when_inference_is_stopped(handler, monkeypatch):
    CAPIWebHandler._train_new_jobs["job"] = _runtime()
    handler.status_tracker = SimpleNamespace(get_status=lambda: {"server": {"running": False}})
    handler._get_update_status_payload = lambda: {}
    monkeypatch.setattr(capi_web, "_get_cached_hardware_status", lambda _: {})
    monkeypatch.setattr(capi_web, "_inference_devices", lambda *_: ("cpu", []))
    captured = {}
    handler._send_json = lambda payload, **kwargs: captured.update(payload=payload, **kwargs)

    handler._handle_api_status()

    assert captured.get("status", 200) == 200
    assert captured["headers"] == {"Access-Control-Allow-Origin": "*"}
    assert captured["payload"]["training"]["active"] is True
    assert captured["payload"]["server"]["running"] is False
