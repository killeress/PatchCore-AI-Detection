import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from capi_database import CAPIDatabase
from capi_model_validation import (
    build_model_validation_summary,
    classify_model_validation_result,
    score_bundle_sample,
)
from capi_web import CAPIWebHandler


def _result(sample_id, candidate, baseline, lighting="G0F00000", zone="inner"):
    return {
        "sample_id": sample_id,
        "review_id": sample_id + 100,
        "lighting": lighting,
        "zone": zone,
        "candidate_caught": candidate,
        "baseline_caught": baseline,
    }


def test_model_validation_summary_reports_paired_gain_and_regression():
    results = [
        _result(1, True, True),
        _result(2, True, False),
        _result(3, False, True, zone="edge"),
        _result(4, False, False, zone="edge"),
        _result(5, None, True, lighting="W0F00000"),
    ]

    summary = build_model_validation_summary(results, has_baseline=True)

    assert summary["sample_count"] == 5
    assert summary["review_count"] == 5
    assert summary["candidate_evaluable"] == 4
    assert summary["candidate_caught"] == 2
    assert summary["candidate_missed"] == 2
    assert summary["candidate_errors"] == 1
    assert summary["candidate_caught_rate"] == 0.5
    assert summary["baseline_evaluable"] == 5
    assert summary["candidate_gain"] == 1
    assert summary["candidate_regression"] == 1
    assert summary["both_caught"] == 1
    assert summary["both_missed"] == 1
    assert summary["units"]["G0F00000-inner"]["candidate_gain"] == 1
    assert summary["units"]["G0F00000-edge"]["candidate_regression"] == 1
    assert classify_model_validation_result(results[1], has_baseline=True) == "gain"
    assert classify_model_validation_result(results[2], has_baseline=True) == "regression"


def test_score_bundle_sample_uses_matching_unit_threshold_and_pipeline(tmp_path):
    validation_root = tmp_path / "ng-validation"
    validation_root.mkdir()
    crop_path = validation_root / "sample.png"
    crop_path.write_bytes(b"fixture")
    calls = []

    class FakeScorer:
        def score_image_path(self, **kwargs):
            calls.append(kwargs)
            return 0.61

    bundle = {
        "bundle_path": str(tmp_path / "bundle"),
        "thresholds": {"G0F00000": {"inner": 0.55, "edge": 0.65}},
        "manifest": {
            "image_preprocess_pipeline": [{"method": "clahe", "params": {}}],
            "image_preprocess_pipelines": {
                "inner": [{"method": "median_blur", "params": {"kernel_size": 3}}],
            },
        },
    }
    sample = {
        "lighting": "G0F00000",
        "zone": "inner",
        "crop_path": str(crop_path),
    }

    result = score_bundle_sample(
        bundle,
        sample,
        FakeScorer(),
        validation_base_dir=validation_root,
    )

    assert result == {"score": 0.61, "threshold": 0.55, "caught": True}
    assert calls[0]["lighting"] == "G0F00000"
    assert calls[0]["zone"] == "inner"
    assert calls[0]["preprocess_pipeline"][0]["method"] == "median_blur"


def test_score_bundle_sample_rejects_crop_outside_validation_root(tmp_path):
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"fixture")
    bundle = {
        "bundle_path": str(tmp_path / "bundle"),
        "thresholds": {"G0F00000": {"inner": 0.5}},
        "manifest": {},
    }

    with pytest.raises(ValueError):
        score_bundle_sample(
            bundle,
            {
                "lighting": "G0F00000",
                "zone": "inner",
                "crop_path": str(outside),
            },
            SimpleNamespace(),
            validation_base_dir=tmp_path / "ng-validation",
        )


def test_model_validation_database_persists_run_and_results(tmp_path):
    db = CAPIDatabase(tmp_path / "validation.db")
    run_id = db.create_model_validation_run(
        candidate_bundle_id=10,
        baseline_bundle_id=9,
        machine_id="MODEL-1",
        sample_count=1,
        candidate_snapshot={"id": 10, "label": "candidate"},
        baseline_snapshot={"id": 9, "label": "baseline"},
    )
    db.start_model_validation_run(run_id)
    db.save_model_validation_result(
        run_id,
        {
            "sample_id": 101,
            "review_id": 201,
            "glass_id": "PANEL-1",
            "model_id": "MODEL-1",
            "lighting": "G0F00000",
            "zone": "inner",
            "image_name": "G0F00000.png",
            "aoi_defect_code": "PCDK2",
            "category": "score_below_threshold",
            "candidate_score": 0.7,
            "candidate_threshold": 0.5,
            "candidate_caught": True,
            "candidate_error": "",
            "baseline_score": 0.4,
            "baseline_threshold": 0.5,
            "baseline_caught": False,
            "baseline_error": "",
        },
        progress=1,
    )
    summary = build_model_validation_summary(
        db.list_model_validation_results(run_id),
        has_baseline=True,
    )
    db.finish_model_validation_run(run_id, state="completed", summary=summary)

    run = db.get_model_validation_run(run_id)

    assert run["state"] == "completed"
    assert run["progress"] == 1
    assert run["candidate"]["label"] == "candidate"
    assert run["baseline"]["label"] == "baseline"
    assert run["summary"]["candidate_gain"] == 1
    assert run["results"][0]["candidate_caught"] == 1
    assert db.list_model_validation_runs(10)[0]["id"] == run_id


def test_model_validation_worker_scores_candidate_and_active_same_questions(
    tmp_path,
    monkeypatch,
):
    db = CAPIDatabase(tmp_path / "validation.db")
    validation_root = tmp_path / "ng-validation"
    validation_root.mkdir()
    sample_paths = []
    for name in ("sample-1.png", "sample-2.png"):
        path = validation_root / name
        path.write_bytes(b"fixture")
        sample_paths.append(path)

    candidate = {
        "id": 10,
        "machine_id": "MODEL-1",
        "bundle_path": str(tmp_path / "candidate"),
        "trained_at": "2026-07-28",
        "is_active": 0,
        "thresholds": {"G0F00000": {"inner": 0.5}},
        "thresholds_source": "machine_config.yaml",
        "manifest": {},
    }
    baseline = {
        "id": 9,
        "machine_id": "MODEL-1",
        "bundle_path": str(tmp_path / "baseline"),
        "trained_at": "2026-07-20",
        "is_active": 1,
        "thresholds": {"G0F00000": {"inner": 0.5}},
        "thresholds_source": "machine_config.yaml",
        "manifest": {},
    }
    samples = [
        {
            "id": index,
            "review_id": 100 + index,
            "glass_id": f"PANEL-{index}",
            "model_id": "MODEL-1",
            "lighting": "G0F00000",
            "zone": "inner",
            "image_name": path.name,
            "crop_path": str(path),
            "aoi_defect_code": "PCDK2",
            "category": "score_below_threshold",
        }
        for index, path in enumerate(sample_paths, 1)
    ]
    run_id = db.create_model_validation_run(
        candidate_bundle_id=10,
        baseline_bundle_id=9,
        machine_id="MODEL-1",
        sample_count=2,
        candidate_snapshot={"id": 10},
        baseline_snapshot={"id": 9},
    )

    class FakeScorer:
        def __init__(self, **_kwargs):
            self._inferencer_cache = {}

        def score_image_path(self, *, bundle_dir, image_path, **_kwargs):
            scores = {
                ("candidate", "sample-1.png"): 0.8,
                ("candidate", "sample-2.png"): 0.2,
                ("baseline", "sample-1.png"): 0.3,
                ("baseline", "sample-2.png"): 0.7,
            }
            return scores[(Path(bundle_dir).name, Path(image_path).name)]

    monkeypatch.setattr("capi_inference.SubmodelScorer", FakeScorer)
    monkeypatch.setattr(
        CAPIWebHandler,
        "_free_server_gpu_cache",
        classmethod(lambda cls: None),
    )
    monkeypatch.setattr(
        CAPIWebHandler,
        "_scan_state",
        {
            "lock": threading.Lock(),
            "job": {
                "run_id": run_id,
                "kind": "model_validation",
                "state": "running",
                "done": 0,
            },
        },
    )
    server = SimpleNamespace(database=db, _gpu_lock=threading.Lock())

    CAPIWebHandler._model_validation_worker(
        run_id,
        candidate,
        baseline,
        samples,
        validation_root,
        threading.Event(),
        server,
    )

    run = db.get_model_validation_run(run_id)
    assert run["state"] == "completed"
    assert run["progress"] == 2
    assert run["summary"]["candidate_gain"] == 1
    assert run["summary"]["candidate_regression"] == 1
    assert CAPIWebHandler._scan_state["job"]["state"] == "completed"


def test_model_validation_start_does_not_gate_a_single_ng_sample(tmp_path, monkeypatch):
    candidate = {
        "id": 10,
        "machine_id": "MODEL-1",
        "bundle_path": str(tmp_path / "candidate"),
        "manifest": {},
        "thresholds": {},
    }
    baseline = {
        "id": 9,
        "machine_id": "MODEL-1",
        "bundle_path": str(tmp_path / "baseline"),
        "manifest": {},
        "thresholds": {},
    }
    sample = {
        "id": 1,
        "model_id": "MODEL-1",
        "lighting": "G0F00000",
        "zone": "inner",
        "crop_path": str(tmp_path / "sample.png"),
    }

    class FakeDb:
        def get_active_model_bundle(self):
            return baseline

        def list_ng_validation_samples(self, **_kwargs):
            return [sample], 1

    monkeypatch.setattr(
        "capi_model_registry.get_bundle_detail",
        lambda _db, bundle_id: candidate if bundle_id == 10 else baseline,
    )
    started = []

    def fake_start(cls, **kwargs):
        started.append(kwargs)
        return True, {"run_id": 55, "total": len(kwargs["samples"])}

    monkeypatch.setattr(
        CAPIWebHandler,
        "_start_model_validation_job",
        classmethod(fake_start),
    )
    handler = object.__new__(CAPIWebHandler)
    handler.path = "/api/models/10/validation/start"
    handler._capi_server_instance = SimpleNamespace(database=FakeDb())
    handler._ng_validation_base_dir = lambda: tmp_path
    responses = []
    handler._send_json = lambda data, status=200: responses.append((status, data))

    handler._handle_models_validation_start()

    assert len(started) == 1
    assert len(started[0]["samples"]) == 1
    assert started[0]["baseline"]["id"] == 9
    assert responses == [(202, {"run_id": 55, "total": 1})]


def test_models_template_exposes_ng_model_exam_workflow():
    template = Path("templates/models.html").read_text(encoding="utf-8")

    assert "🧪 NG 能力考試" in template
    assert "openModelValidation" in template
    assert "/validation/start" in template
    assert "Lighting × Zone" in template
    assert "候選模型退步" in template
    assert "不設最低題數" in template
