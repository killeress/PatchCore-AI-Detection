import json
from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np
import pytest
import yaml

from capi_training_validation import (
    adopt_threshold, build_report, evaluate_model, load_report,
    normalize_validation_config, rates, seal_reports, sha256_file,
    suggest_threshold, training_tiles, write_json,
)


def sample(role, label, score, group=None):
    return {"role": role, "label": label, "score": score, "group": group or role}


def samples():
    return [sample("calibration", "ok", 0.4), sample("calibration", "ng", 0.6),
            sample("acceptance", "ok", 0.3), sample("acceptance", "ng", 0.7)]


def config(tmp_path):
    return {"panels": {str(tmp_path / role): {"role": role, "group": role} for role in ("train", "calibration", "acceptance")}}


def test_partition_requires_disjoint_batches_and_complete_panel_list(tmp_path):
    raw = config(tmp_path)
    normalized = normalize_validation_config(raw, list(raw["panels"]))
    assert normalize_validation_config(normalized, list(raw["panels"])) == normalized
    with pytest.raises(ValueError, match="完全一致"):
        normalize_validation_config(raw, [tmp_path / "train"])
    raw["panels"][str(tmp_path / "acceptance")]["group"] = "calibration"
    with pytest.raises(ValueError, match="同一批次"):
        normalize_validation_config(raw)


def test_same_physical_panel_cannot_cross_roles_even_in_different_folders(tmp_path):
    raw = config(tmp_path)
    raw["panels"][str(tmp_path / "other" / "train")] = raw["panels"].pop(str(tmp_path / "acceptance"))
    with pytest.raises(ValueError, match="同一 panel"):
        normalize_validation_config(raw)


@pytest.mark.parametrize("value", [True, -1, 1.1, float("nan"), float("inf"), "0.1"])
def test_invalid_targets_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        normalize_validation_config({**config(tmp_path), "max_miss_rate": value})


def test_changing_acceptance_cannot_change_suggested_threshold():
    data = samples()
    original = build_report(data, {})
    changed = build_report(data[:2] + [sample("acceptance", "ok", 0.99), sample("acceptance", "ng", 0.0)], {})
    assert original["suggested_threshold"] == changed["suggested_threshold"]
    assert original["suggested"]["misses"] == 0
    assert changed["suggested"]["misses"] == 1
    assert changed["suggested"]["false_positives"] == 1
    assert original["status"] == "unassessed"


def test_threshold_boundary_and_rounding_match_production():
    calibration = [sample("calibration", "ok", 0.35000001), sample("calibration", "ng", 0.3501)]
    threshold = suggest_threshold(calibration, {"max_miss_rate": 0})
    assert threshold == 0.3501
    assert round(threshold, 4) == threshold
    assert rates(calibration, threshold)["misses"] == 0
    assert rates(calibration, threshold)["false_positives"] == 0


def test_status_requires_both_targets_and_both_classes():
    limits = {"max_false_positive_rate": 0, "max_miss_rate": 0}
    assert build_report(samples(), limits)["status"] == "pass"
    assert build_report(samples(), {"max_miss_rate": 0})["status"] == "unassessed"
    assert build_report(samples()[:-1], limits)["status"] == "insufficient"
    assert build_report(samples(), limits, complete=False)["suggested_threshold"] is None
    failed = samples()[:2] + [sample("acceptance", "ok", 0.9), sample("acceptance", "ng", 0.1)]
    assert build_report(failed, limits)["status"] == "fail"
    inseparable = [sample(role, label, 0.5) for role in ("calibration", "acceptance") for label in ("ok", "ng")]
    assert build_report(inseparable, limits)["status"] == "fail"


def test_default_legacy_rows_are_training_only():
    rows = [{"id": 1}, {"id": 2, "dataset_role": "calibration"}, {"id": 3, "dataset_role": "acceptance"}]
    assert training_tiles(rows) == [{"id": 1}]


def make_tiles(tmp_path):
    rows = []
    for index, (role, label) in enumerate([("train", ""), ("calibration", "ok"), ("calibration", "ng"), ("acceptance", "ok"), ("acceptance", "ng")], 1):
        path = tmp_path / f"{index}.png"
        cv2.imwrite(str(path), np.full((8, 8, 3), index * 30, np.uint8))
        rows.append({"id": index, "source": "ok", "source_path": str(path), "dataset_role": role,
                     "validation_label": label, "validation_group": role, "decision": "accept",
                     "lighting": "W0F00000", "zone": "inner", "panel_path": str(tmp_path / role)})
    return rows


def fake_inferencer(monkeypatch):
    seen = []
    class Inferencer:
        def __init__(self, **kwargs):
            pass
        def predict(self, image):
            seen.append(int(image[0, 0, 0]))
            return SimpleNamespace(pred_score=SimpleNamespace(item=lambda: float(image[0, 0, 0]) / 255))
    monkeypatch.setitem(sys.modules, "anomalib.deploy", SimpleNamespace(TorchInferencer=Inferencer))
    return seen


def test_evaluation_streams_processed_tiles_and_preserves_inputs(tmp_path, monkeypatch):
    rows = make_tiles(tmp_path)
    model = tmp_path / "W0F00000_inner.pt"
    model.write_bytes(b"model")
    seen = fake_inferencer(monkeypatch)
    summary = evaluate_model(model, rows[1:], rows[:1], {}, tmp_path, "W0F00000-inner", lambda _: None)
    assert seen == [60, 90, 120, 150]
    report_path = tmp_path / summary["report_path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(report["samples"]) == 4
    assert report["model_file"] == model.name
    for item in report["samples"]:
        assert sha256_file(tmp_path / item["asset_path"]) == item["sha256"]
    assert (report_path.parent / "inputs.json").exists()


def test_unreviewed_or_duplicate_samples_never_produce_adoptable_report(tmp_path, monkeypatch):
    rows = make_tiles(tmp_path)
    rows[1]["validation_label"] = ""
    rows[3]["source_path"] = rows[0]["source_path"]
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    seen = fake_inferencer(monkeypatch)
    summary = evaluate_model(model, rows[1:], rows[:1], {}, tmp_path, "W0F00000-inner", lambda _: None)
    assert seen == [90, 150]
    assert summary["status"] == "insufficient"
    assert summary["suggested_threshold"] is None


def test_validation_labels_and_decisions_freeze_when_training_starts(tmp_path):
    from capi_database import CAPIDatabase
    db = CAPIDatabase(tmp_path / "test.db")
    db.create_training_job("j", "machine", [])
    db.update_training_job_state("j", "review")
    rows = make_tiles(tmp_path)
    ids = db.insert_tile_pool("j", rows)
    assert db.update_validation_review("j", [ids[0]], label="ng") == 0
    assert db.update_validation_review("j", [ids[1]], label="ng") == 1
    db.update_training_job_state("j", "train")
    assert db.update_validation_review("j", [ids[1]], label="ok") == 0
    assert db.update_validation_review("j", [ids[1]], decision="reject") == 0
    assert db.list_tile_pool("j")[1]["validation_label"] == "ng"


def make_bundle(tmp_path):
    unit = "W0F00000-inner"
    model = tmp_path / "W0F00000_inner.pt"
    model.write_bytes(b"model")
    recipe = {"machine_id": "machine", "threshold_mapping": {"W0F00000": {"inner": 0.35}}, "image_preprocess_pipeline": []}
    (tmp_path / "machine_config.yaml").write_text(yaml.safe_dump(recipe), encoding="utf-8")
    write_json(tmp_path / "thresholds.json", recipe["threshold_mapping"])
    report = build_report(samples(), {})
    report.update(unit_label=unit, model_file=model.name, model_sha256=sha256_file(model))
    relative = f"validation_reports/{unit}/report.json"
    write_json(tmp_path / relative, report)
    seal_reports(tmp_path, {unit: {"validation": {"report_path": relative}}})
    bundle = {"bundle_path": str(tmp_path), "machine_id": "machine", "is_active": False}
    return unit, bundle


def test_adoption_updates_yaml_json_audit_without_changing_report_metrics(tmp_path):
    unit, bundle = make_bundle(tmp_path)
    db = SimpleNamespace(get_model_bundle=lambda _: bundle)
    before = load_report(tmp_path, unit)
    result = adopt_threshold(db, 1, unit)
    threshold = before["suggested_threshold"]
    assert result["value"] == threshold
    assert yaml.safe_load((tmp_path / "machine_config.yaml").read_text())["threshold_mapping"]["W0F00000"]["inner"] == threshold
    assert json.loads((tmp_path / "thresholds.json").read_text())["W0F00000"]["inner"] == threshold
    audit = json.loads((tmp_path / "validation_reports" / unit / "adoptions.json").read_text(encoding="utf-8"))
    assert audit[0]["previous_threshold"] == 0.35
    assert audit[0]["threshold"] == threshold
    assert load_report(tmp_path, unit) == before


@pytest.mark.parametrize("change", ["model", "recipe", "active"])
def test_changed_or_active_bundle_cannot_adopt_old_report(tmp_path, change):
    unit, bundle = make_bundle(tmp_path)
    if change == "model":
        (tmp_path / "W0F00000_inner.pt").write_bytes(b"new model")
    elif change == "recipe":
        with (tmp_path / "machine_config.yaml").open("a") as f:
            f.write("preprocess_after_tiling: true\n")
    else:
        bundle["is_active"] = True
    before = (tmp_path / "thresholds.json").read_bytes()
    with pytest.raises(ValueError):
        adopt_threshold(SimpleNamespace(get_model_bundle=lambda _: bundle), 1, unit)
    assert (tmp_path / "thresholds.json").read_bytes() == before


def test_adoption_rolls_back_thresholds_if_audit_cannot_be_saved(tmp_path, monkeypatch):
    import capi_training_validation as validation
    unit, bundle = make_bundle(tmp_path)
    before = {name: (tmp_path / name).read_bytes() for name in ("machine_config.yaml", "thresholds.json")}
    def fail_write(*args):
        raise OSError("audit disk failure")
    monkeypatch.setattr(validation, "write_json", fail_write)
    with pytest.raises(OSError):
        adopt_threshold(SimpleNamespace(get_model_bundle=lambda _: bundle), 1, unit)
    assert {name: (tmp_path / name).read_bytes() for name in before} == before


def test_validation_report_template_exposes_no_pass_without_targets(tmp_path):
    from jinja2 import Environment, FileSystemLoader
    unit, bundle = make_bundle(tmp_path)
    report = load_report(tmp_path, unit)
    report.update(current_threshold=0.35, errors=[])
    env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
    html = env.get_template("train_new/_validation_report.html").render(validation_reports=[report], job_id="j")
    assert "未設定完整目標" in html
    assert "採用建議門檻" in html
    assert "本次樣本符合目標" not in html


@pytest.mark.parametrize("complete_calibration", [True, False])
@pytest.mark.parametrize("auto_review", [True, False])
def test_training_stages_only_training_and_calibration_never_acceptance(tmp_path, monkeypatch, complete_calibration, auto_review):
    import capi_train_new as training
    import capi_training_validation as validation
    rows = make_tiles(tmp_path)
    if auto_review:
        rows[0]["validation_label"] = "ok"
    if not complete_calibration:
        rows[2]["validation_label"] = "ok"
    rows += [{**rows[0], "id": index} for index in range(6, 35)]
    rows.append({**rows[2], "id": 100, "source": "ng", "dataset_role": "train"})
    rows.append({**rows[3], "id": 101, "zone": "edge"})
    if auto_review:
        rows.extend([{**rows[0], "id": 102, "validation_label": "ng"},
                     {**rows[0], "id": 103, "validation_label": ""}])
    db = SimpleNamespace(list_tile_pool=lambda job, **filters: [t for t in rows if all(t.get(k) == v for k, v in filters.items())])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, "_setup_offline_env", lambda *a: None)
    monkeypatch.setattr(training, "_calibrate_from_model", lambda *a: (0.1, [0.1], [0.6]))
    staged = []
    def fit(staging, run_root, unit_label, cfg, **kwargs):
        assert (tmp_path / "bundle" / "validation_reports" / unit_label / "inputs.json").exists()
        train_files = list((staging / "train").iterdir())
        assert len(train_files) == 30
        assert all(p.read_bytes() == Path(rows[0]["source_path"]).read_bytes() for p in train_files)
        if complete_calibration:
            assert [p.read_bytes() for p in (staging / "test" / "normal").iterdir()] == [Path(rows[1]["source_path"]).read_bytes()]
            assert [p.read_bytes() for p in (staging / "test" / "anormal").iterdir()] == [Path(rows[2]["source_path"]).read_bytes()]
        else:
            assert not (staging / "test" / "normal").exists()
            assert not list((staging / "test" / "anormal").iterdir())
        staged.append(True)
        run_root.mkdir(parents=True)
        output = run_root / "model.pt"
        output.write_bytes(b"model")
        return output
    evaluated = []
    def evaluate(model, held_out, train, *args):
        assert {t["id"] for t in held_out} == {2, 3, 4, 5}
        assert len(train) == 30
        assert all(t["dataset_role"] == "train" for t in train)
        evaluated.append(True)
        return {"status": "unassessed"}
    monkeypatch.setattr(training, "train_one_patchcore", fit)
    monkeypatch.setattr(validation, "evaluate_model", evaluate)
    cfg = training.TrainingConfig(machine_id="M", panel_paths=[], over_review_root=tmp_path, validation_config=config(tmp_path))
    if auto_review:
        cfg.validation_config["split_mode"] = "auto_batch"
    result = training.train_single_submodel(db, "j", "W0F00000", "inner", cfg, tmp_path / "bundle" / "W0F00000_inner.pt", log=lambda _: None)
    assert staged and evaluated
    assert result["tile_count"] == 30
    assert result["ng_count"] == (1 if complete_calibration else 0)
    assert result["threshold"] == 0.35  # Adoption is a separate user action.


def test_manual_scan_can_select_separate_batches_without_flattening(tmp_path):
    from capi_web import CAPIWebHandler
    from capi_station_adapter import create_station_adapter
    for batch in ("batch-a", "batch-b", "batch-c"):
        panel = tmp_path / batch / ("panel-" + batch)
        panel.mkdir(parents=True)
        cv2.imwrite(str(panel / "W0F00000.jpg"), np.zeros((8, 8, 3), np.uint8))
    adapter = create_station_adapter("capi")
    assert CAPIWebHandler._scan_train_new_manual_batch(tmp_path, "M", adapter) == []
    panels = CAPIWebHandler._scan_train_new_manual_batch(tmp_path, "M", adapter, include_batches=True)
    assert {p["batch_id"] for p in panels} == {"batch-a", "batch-b", "batch-c"}


def test_mixed_review_request_is_atomic(tmp_path):
    from capi_database import CAPIDatabase
    db = CAPIDatabase(tmp_path / "test.db")
    db.create_training_job("j", "machine", [])
    db.update_training_job_state("j", "review")
    ids = db.insert_tile_pool("j", make_tiles(tmp_path))
    assert db.update_validation_review("j", ids[:2], label="ng") == 0
    assert db.list_tile_pool("j")[1]["validation_label"] == "ok"


def handler_for(db, path, payload):
    import io
    from capi_web import CAPIWebHandler
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler._capi_server_instance = SimpleNamespace(database=db)
    handler.path = path
    data = json.dumps(payload).encode()
    handler.headers = {"Content-Length": str(len(data))}
    handler.rfile = io.BytesIO(data)
    responses = []
    handler._send_json = lambda body, status=200, **kwargs: responses.append((status, body))
    return handler, responses


def test_review_api_stores_labels_and_rejects_edits_after_start(tmp_path):
    from capi_database import CAPIDatabase
    db = CAPIDatabase(tmp_path / "test.db")
    db.create_training_job("j", "machine", [])
    db.update_training_job_state("j", "review")
    ids = db.insert_tile_pool("j", make_tiles(tmp_path))
    payload = {"job_id": "j", "tile_ids": [ids[1]], "validation_label": "ng"}
    h, responses = handler_for(db, "/api/train/new/tiles/decision", payload)
    h._handle_train_new_tiles_decision()
    assert responses[0][0] == 200
    db.update_training_job_state("j", "train")
    h, responses = handler_for(db, "/api/train/new/tiles/decision", payload)
    h._handle_train_new_tiles_decision()
    assert responses[0][0] == 409


def test_start_rejects_unreviewed_validation_before_reserving_gpu(tmp_path):
    from capi_database import CAPIDatabase
    from capi_web import CAPIWebHandler
    db = CAPIDatabase(tmp_path / "test.db")
    db.create_training_job("j", "machine", [], training_params={"validation_config": config(tmp_path)})
    db.update_training_job_state("j", "review")
    rows = make_tiles(tmp_path)
    rows[3]["validation_label"] = ""
    db.insert_tile_pool("j", rows)
    h, responses = handler_for(db, "/api/train/new/start_training/j", {})
    h._mark_train_new_stale_if_needed = lambda database, job: job
    before = CAPIWebHandler._train_slot.get("active_job_id")
    h._handle_train_new_start_training()
    assert responses[0][0] == 400
    assert "1 張" in responses[0][1]["error"]
    assert CAPIWebHandler._train_slot.get("active_job_id") == before


def test_apply_api_resolves_exact_completed_bundle(tmp_path):
    unit, bundle = make_bundle(tmp_path)
    write_json(tmp_path / "manifest.json", {"unit_metrics": {unit: {"validation": {"status": "unassessed"}}}})
    db = SimpleNamespace(
        get_training_job=lambda job: {"state": "completed", "output_bundle": str(tmp_path), "machine_id": "machine"},
        get_model_bundle=lambda bid: bundle,
        list_model_bundles=lambda machine: [{"id": 1, **bundle}],
    )
    h, responses = handler_for(db, "/api/train/new/apply-validation/j", {"unit_label": unit})
    h._handle_train_new_apply_validation()
    assert responses[0][0] == 200
    assert responses[0][1]["value"] == load_report(tmp_path, unit)["suggested_threshold"]


def test_failed_panel_preprocessing_cannot_silently_shrink_validation_set(tmp_path, monkeypatch):
    import capi_train_new as training
    raw = config(tmp_path)
    cfg = training.TrainingConfig(machine_id="M", panel_paths=[Path(p) for p in raw["panels"]], over_review_root=tmp_path, validation_config=raw)
    pre_cfg = SimpleNamespace(image_preprocess_pipeline=[], image_preprocess_pipelines={}, preprocess_after_tiling=False)
    monkeypatch.setattr(training, "preprocess_panel_folder", lambda *a, **kw: {})
    with pytest.raises(RuntimeError, match="3 片前處理失敗"):
        training.preprocess_panels_to_pool("j", cfg, pre_cfg, None, tmp_path / "thumbs", lambda _: None)


def auto_config(tmp_path, groups=5):
    return {"split_mode": "auto_batch", "panels": {
        str(tmp_path / f"batch-{group}" / f"panel-{group}-{panel}"): {"group": f"batch-{group}"}
        for group in range(groups) for panel in range(2)
    }}


def test_auto_split_is_stable_and_never_splits_a_batch(tmp_path):
    raw = auto_config(tmp_path)
    normalized = normalize_validation_config(raw)
    reordered = {**raw, "panels": dict(reversed(list(raw["panels"].items())))}
    assert normalize_validation_config(reordered) == normalized
    assert normalize_validation_config(normalized) == normalized
    role_groups = {}
    for panel in normalized["panels"].values():
        role_groups.setdefault(panel["group"], set()).add(panel["role"])
    assert all(len(roles) == 1 for roles in role_groups.values())
    assert sorted(p["role"] for p in normalized["panels"].values()).count("train") == 6
    assert {p["role"] for p in normalized["panels"].values()} == {"train", "calibration", "acceptance"}


@pytest.mark.parametrize("groups", [1, 2])
def test_insufficient_batches_keep_training_without_claiming_acceptance(tmp_path, groups):
    normalized = normalize_validation_config(auto_config(tmp_path, groups))
    assert {p["role"] for p in normalized["panels"].values()} == {"train"}
    report = build_report([], normalized)
    assert report["status"] == "insufficient"
    assert report["suggested_threshold"] is None
    assert any("不足三組" in reason for reason in report["reasons"])


def test_auto_split_merges_repeated_physical_panels_across_batches(tmp_path):
    raw = {"split_mode": "auto_batch", "panels": {
        str(tmp_path / "a" / "same"): {"group": "a"},
        str(tmp_path / "b" / "same"): {"group": "b"},
        str(tmp_path / "c" / "different"): {"group": "c"},
    }}
    normalized = normalize_validation_config(raw)
    assert {p["role"] for p in normalized["panels"].values()} == {"train"}


def test_auto_training_uses_only_confirmed_ok():
    rows = [{"id": i, "dataset_role": role, "validation_label": label} for i, (role, label) in enumerate(
        [("train", "ok"), ("train", "ng"), ("train", ""), ("calibration", "ok"), ("acceptance", "ok")])]
    assert [t["id"] for t in training_tiles(rows, require_confirmed=True)] == [0]
    assert [t["id"] for t in training_tiles(rows)] == [0, 2]


def test_auto_review_api_can_label_training_ng_without_training_it(tmp_path):
    from capi_database import CAPIDatabase
    db = CAPIDatabase(tmp_path / "test.db")
    normalized = normalize_validation_config(auto_config(tmp_path))
    db.create_training_job("j", "machine", [], training_params={"validation_config": normalized})
    db.update_training_job_state("j", "review")
    ids = db.insert_tile_pool("j", make_tiles(tmp_path))
    h, responses = handler_for(db, "/api/train/new/tiles/decision", {"job_id": "j", "tile_ids": [ids[0]], "validation_label": "ng"})
    h._handle_train_new_tiles_decision()
    assert responses[0][0] == 200
    assert db.list_tile_pool("j")[0]["validation_label"] == "ng"
    assert training_tiles(db.list_tile_pool("j"), require_confirmed=True) == []
    db.update_training_job_state("j", "train")
    assert db.update_validation_review("j", [ids[0]], label="ok", allow_training_labels=True) == 0


def test_auto_config_passes_web_validation_and_runner_without_changing_split(tmp_path):
    from capi_web import CAPIWebHandler
    from capi_train_new import TrainingConfig, apply_user_training_params
    raw = auto_config(tmp_path)
    params, error = CAPIWebHandler._validate_training_params({"validation_config": raw})
    assert error is None
    cfg = TrainingConfig(machine_id="M", panel_paths=[Path(p) for p in raw["panels"]], over_review_root=tmp_path)
    apply_user_training_params(cfg, params)
    assert cfg.validation_config == params["validation_config"]
