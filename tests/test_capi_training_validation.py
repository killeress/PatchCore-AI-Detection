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
    suggest_threshold, training_tiles, write_json, review_decision_labels, validation_tiles,
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
@pytest.mark.parametrize("auto_review", ["auto_batch", "auto_panel", None])
def test_training_stages_only_training_and_calibration_never_acceptance(tmp_path, monkeypatch, complete_calibration, auto_review):
    import capi_train_new as training
    import capi_training_validation as validation
    rows = make_tiles(tmp_path)
    if auto_review:
        rows[0]["validation_label"] = "ok"
    if not complete_calibration:
        rows[2]["validation_label"] = "ok"
    if auto_review == "auto_panel":
        for row in rows:
            row["decision"] = "reject" if row["validation_label"] == "ng" else "accept"
        rows[0]["validation_label"] = "ng"  # Old labels cannot override the include decision.
    rows += [{**rows[0], "id": index} for index in range(6, 35)]
    rows.append({**rows[2], "id": 100, "source": "ng", "dataset_role": "train"})
    rows.append({**rows[3], "id": 101, "zone": "edge"})
    if auto_review:
        decision = "reject" if auto_review == "auto_panel" else "accept"
        rows.extend([{**rows[0], "id": 102, "validation_label": "ng", "decision": decision},
                     {**rows[0], "id": 103, "validation_label": "", "decision": decision}])
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
        cfg.validation_config["split_mode"] = auto_review
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


def test_auto_review_api_uses_include_exclude_instead_of_extra_labels(tmp_path):
    from capi_database import CAPIDatabase
    db = CAPIDatabase(tmp_path / "test.db")
    normalized = normalize_validation_config({**config(tmp_path), "split_mode": "auto_panel"})
    db.create_training_job("j", "machine", list(normalized["panels"]), training_params={"validation_config": normalized})
    db.update_training_job_state("j", "review")
    ids = db.insert_tile_pool("j", make_tiles(tmp_path))
    h, responses = handler_for(db, "/api/train/new/tiles/decision", {"job_id": "j", "tile_ids": [ids[0]], "validation_label": "ng"})
    h._handle_train_new_tiles_decision()
    assert responses[0][0] == 409
    for decision, label in [("reject", "ng"), ("accept", "ok")]:
        h, responses = handler_for(db, "/api/train/new/tiles/decision", {"job_id": "j", "tile_ids": [ids[0]], "decision": decision})
        h._handle_train_new_tiles_decision()
        assert responses[0][0] == 200
        h, responses = handler_for(db, "/api/train/new/tiles?job_id=j&lighting=W0F00000", {})
        h._handle_train_new_tiles()
        tile = responses[0][1]["tiles"][0]
        assert tile["validation_label"] == label
        assert tile["decision_review"] is True
    db.update_training_job_state("j", "train")
    assert db.update_validation_review("j", [ids[0]], decision="reject") == 0


def test_auto_config_passes_web_validation_and_runner_without_changing_split(tmp_path):
    from capi_web import CAPIWebHandler
    from capi_train_new import TrainingConfig, apply_user_training_params
    raw = auto_config(tmp_path)
    params, error = CAPIWebHandler._validate_training_params({"validation_config": raw})
    assert error is None
    cfg = TrainingConfig(machine_id="M", panel_paths=[Path(p) for p in raw["panels"]], over_review_root=tmp_path)
    apply_user_training_params(cfg, params)
    assert cfg.validation_config == params["validation_config"]


def test_same_day_six_panel_ids_split_four_one_one_stably(tmp_path):
    raw = {"split_mode": "auto_panel", "panels": {
        str(tmp_path / "2026-09-08" / f"panel-{i}"): {} for i in range(6)}}
    normalized = normalize_validation_config(raw)
    assert normalize_validation_config(normalized) == normalized
    assert normalize_validation_config({**raw, "panels": dict(reversed(list(raw["panels"].items())))}) == normalized
    assert [p["role"] for p in normalized["panels"].values()].count("train") == 4
    assert [p["role"] for p in normalized["panels"].values()].count("calibration") == 1
    assert [p["role"] for p in normalized["panels"].values()].count("acceptance") == 1


def test_panel_identity_uses_glass_id_even_when_folders_differ(tmp_path):
    raw = {"split_mode": "auto_panel", "panels": {
        str(tmp_path / f"capture-{i}"): {"group": f"GLASS-{i // 2}"} for i in range(6)}}
    normalized = normalize_validation_config(raw)
    assert {p["role"] for p in normalized["panels"].values()} == {"train", "calibration", "acceptance"}
    for panel_id in ("GLASS-0", "GLASS-1", "GLASS-2"):
        assert len({p["role"] for p in normalized["panels"].values() if p["group"] == panel_id}) == 1


def test_zone_split_covers_inner_and_edge_for_every_small_selection_mix(tmp_path):
    from itertools import product
    for inner, edge, both in product(range(5), repeat=3):
        if inner + edge + both == 0:
            continue
        modes = ["inner_only"] * inner + ["edge_only"] * edge + ["full"] * both
        raw = {"split_mode": "auto_panel", "panels": {str(tmp_path / f"P{i}"): {} for i in range(len(modes))}}
        paths = list(raw["panels"])
        cfg = normalize_validation_config(raw, paths, modes)
        assert normalize_validation_config(cfg) == cfg
        assert normalize_validation_config(raw, paths[::-1], modes[::-1]) == cfg
        for zone, total in (("inner", inner + both), ("edge", edge + both)):
            roles = {p["role"] for p in cfg["panels"].values() if zone in p["zones"]}
            if total:
                assert "train" in roles, (inner, edge, both, zone)
            if total >= 3:
                assert roles == {"train", "calibration", "acceptance"}, (inner, edge, both, zone)


def test_zone_split_uses_selected_modes_and_does_not_force_global_four_one_one(tmp_path):
    raw = {"split_mode": "auto_panel", "panels": {
        str(tmp_path / f"P{i}"): {"zones": ["inner", "edge"]} for i in range(6)}}
    cfg = normalize_validation_config(raw, list(raw["panels"]), ["inner_only"] * 3 + ["edge_only"] * 3)
    for zone in ("inner", "edge"):
        rows = [p for p in cfg["panels"].values() if zone in p["zones"]]
        assert len(rows) == 3
        assert sorted(p["role"] for p in rows) == ["acceptance", "calibration", "train"]
    full = normalize_validation_config(raw, list(raw["panels"]), ["full"] * 6)
    assert [p["role"] for p in full["panels"].values()].count("train") == 4


def test_zone_split_merges_same_panel_id_across_inner_and_edge_captures(tmp_path):
    raw = {"split_mode": "auto_panel", "panels": {
        str(tmp_path / f"capture{i}"): {"group": f"P{i // 2}"} for i in range(6)}}
    cfg = normalize_validation_config(raw, list(raw["panels"]), ["inner_only", "corners_only"] * 3)
    for group in ("P0", "P1", "P2"):
        panels = [p for p in cfg["panels"].values() if p["group"] == group]
        assert len({p["role"] for p in panels}) == 1
        assert {z for p in panels for z in p["zones"]} == {"inner", "edge"}
    assert {p["role"] for p in cfg["panels"].values()} == {"train", "calibration", "acceptance"}


def test_preprocessing_writes_zone_aware_panel_roles(tmp_path, monkeypatch):
    import capi_train_new as training
    from capi_preprocess import PreprocessConfig
    from capi_training_validation import path_key
    paths = [tmp_path / f"P{i}" for i in range(6)]
    modes = ["inner_only"] * 3 + ["edge_only"] * 3
    raw = {"split_mode": "auto_panel", "panels": {str(p): {} for p in paths}}
    cfg = training.TrainingConfig(machine_id="M", panel_paths=paths, over_review_root=tmp_path, validation_config=raw)
    tiles = [SimpleNamespace(tile_id=i, zone=zone, is_corner=False, image=np.full((16, 16), i, np.uint8))
             for i, zone in enumerate(("inner", "edge"))]
    monkeypatch.setattr(training, "preprocess_panel_folder", lambda *a: {
        "W0F00000": SimpleNamespace(polygon_detection_failed=False, tiles=tiles)})
    written = []
    db = SimpleNamespace(insert_tile_pool=lambda job_id, rows: written.extend(rows))
    training.preprocess_panels_to_pool("j", cfg, PreprocessConfig(), db, tmp_path / "thumbs", lambda _: None, panel_modes=modes)
    expected = normalize_validation_config(raw, paths, modes)
    assert len(written) == 6
    for tile in written:
        panel = expected["panels"][path_key(tile["panel_path"])]
        assert [tile["zone"]] == panel["zones"]
        assert tile["dataset_role"] == panel["role"]


@pytest.mark.parametrize("count", [1, 2])
def test_insufficient_panel_ids_keep_normal_training(count, tmp_path):
    cfg = normalize_validation_config({"split_mode": "auto_panel", "panels": {
        str(tmp_path / str(i)): {} for i in range(count)}})
    assert {p["role"] for p in cfg["panels"].values()} == {"train"}
    report = build_report([], cfg)
    assert report["status"] == "insufficient"
    assert any("PANEL ID" in reason for reason in report["reasons"])


@pytest.mark.parametrize("state", ["review", "train", "completed"])
@pytest.mark.parametrize("split_mode", ["auto_batch", "auto_panel"])
def test_existing_review_migration_preserves_decisions_and_finished_history(tmp_path, state, split_mode):
    from capi_database import CAPIDatabase
    from capi_training_validation import path_key
    db = CAPIDatabase(tmp_path / "test.db")
    raw = {"split_mode": split_mode, "panels": {
        str(tmp_path / f"panel-{i}"): {"group": "same-day" if split_mode == "auto_batch" else f"GLASS-{i}"} for i in range(6)}}
    cfg = normalize_validation_config(raw)
    db.create_training_job("j", "M", list(raw["panels"]), panel_modes=["inner_only"] * 3 + ["edge_only"] * 3,
                           training_params={"validation_config": cfg, "precision": "float32"})
    db.update_training_job_state("j", state)
    rows = make_tiles(tmp_path)
    rows += [{**rows[0], "id": 6}]
    for i, row in enumerate(rows):
        row.update(panel_path=list(raw["panels"])[i], dataset_role="train", validation_group="same-day",
                   decision="reject" if i % 2 else "accept")
    db.insert_tile_pool("j", rows)
    before = db.list_tile_pool("j")
    db.migrate_panel_validation_review("j")
    db.migrate_panel_validation_review("j")  # Refreshing again is harmless.
    after = db.list_tile_pool("j")
    params = db.get_training_job("j")["training_params"]
    assert params["precision"] == "float32"
    assert [t["decision"] for t in after] == [t["decision"] for t in before]
    if state != "review":
        assert after == before
        assert params["validation_config"] == cfg
    else:
        cfg = params["validation_config"]
        assert cfg["split_mode"] == "auto_panel"
        assert {t["dataset_role"] for t in after} == {"train", "calibration", "acceptance"}
        for zone in ("inner", "edge"):
            assert {p["role"] for p in cfg["panels"].values() if zone in p["zones"]} == {"train", "calibration", "acceptance"}
        for tile in after:
            panel = cfg["panels"][path_key(tile["panel_path"])]
            assert (tile["dataset_role"], tile["validation_group"]) == (panel["role"], panel["group"])
        if split_mode == "auto_panel":
            assert {p["group"] for p in cfg["panels"].values()} == {f"GLASS-{i}" for i in range(6)}


def test_excluded_tiles_are_scored_as_ng_and_recorded_with_provenance(tmp_path, monkeypatch):
    rows = make_tiles(tmp_path)
    for row in rows:
        row["decision"] = "reject" if row["validation_label"] == "ng" else "accept"
        row["validation_label"] = ""  # No second annotation step.
    cfg = {"split_mode": "auto_panel", "panels": {"p": {"role": "acceptance"}}}
    db = SimpleNamespace(list_tile_pool=lambda job, **kw: rows)
    held_out = validation_tiles(db, "j", "W0F00000", "inner", cfg)
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    seen = fake_inferencer(monkeypatch)
    summary = evaluate_model(model, held_out, rows[:1], cfg, tmp_path, "W0F00000-inner", lambda _: None)
    assert seen == [60, 90, 120, 150]
    report = json.loads((tmp_path / summary["report_path"]).read_text(encoding="utf-8"))
    assert report["label_source"] == "review_decision"
    assert report["grouping"] == "panel_id"
    assert report["baseline"]["ng_count"] == 1
    assert report["calibration"]["ng_count"] == 1
    assert all(s["decision"] == "reject" for s in report["samples"] if s["label"] == "ng")
    assert [t["id"] for t in training_tiles(review_decision_labels(rows, cfg))] == [1]


def test_feature_memory_estimate_tracks_precision_layers_size_and_warning_boundary():
    from capi_train_new import estimate_patchcore_feature_memory
    baseline = estimate_patchcore_feature_memory(600)
    assert baseline["feature_bytes"] / 2**30 == 7.03125
    assert baseline["stack_bytes"] / 2**30 == 14.0625
    assert not baseline["warning"]
    assert estimate_patchcore_feature_memory(601)["warning"]
    assert estimate_patchcore_feature_memory(600, precision="float32")["feature_bytes"] == 2 * baseline["feature_bytes"]
    assert estimate_patchcore_feature_memory(600, image_size=(256, 256))["feature_bytes"] == baseline["feature_bytes"] // 4
    assert estimate_patchcore_feature_memory(600, feature_layers="layer3")["feature_bytes"] / 2**30 == 1.171875


def test_simplified_start_allows_unlabeled_tiles_and_freezes_decisions(tmp_path, monkeypatch):
    import threading
    from capi_database import CAPIDatabase
    from capi_web import CAPIWebHandler
    db = CAPIDatabase(tmp_path / "test.db")
    cfg = normalize_validation_config({**config(tmp_path), "split_mode": "auto_panel"})
    db.create_training_job("j", "M", list(cfg["panels"]), training_params={"validation_config": cfg})
    db.update_training_job_state("j", "review")
    rows = make_tiles(tmp_path)
    for row in rows:
        row["validation_label"] = ""
    ids = db.insert_tile_pool("j", rows)
    monkeypatch.setattr(CAPIWebHandler, "_train_slot", {"lock": threading.Lock(), "active_job_id": None})
    monkeypatch.setattr(CAPIWebHandler, "_train_new_jobs", {})
    started = []
    monkeypatch.setattr("capi_web.threading.Thread", lambda **kw: SimpleNamespace(start=lambda: started.append(kw)))
    h, responses = handler_for(db, "/api/train/new/start_training/j", {})
    h._mark_train_new_stale_if_needed = lambda database, job: job
    h._migrate_legacy_aapi_training_job = lambda database, job, server: (job, {})
    h._handle_train_new_start_training()
    assert responses[0][0] == 200 and started
    assert db.get_training_job("j")["state"] == "train"
    assert db.update_validation_review("j", [ids[0]], decision="reject") == 0


def test_review_memory_counts_each_pt_without_held_out_or_rejected_tiles():
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for review UI behavior test")
    template = (Path(__file__).resolve().parents[1] / "templates/train_new/step3_review.html").read_text(encoding="utf-8")
    functions = "function trainingMemoryRows()" + template.split("function trainingMemoryRows()", 1)[1].split("function visibleTiles(", 1)[0]
    functions += "function groupTiles(tiles)" + template.split("function groupTiles(tiles)", 1)[1].split("async function labelValidation", 1)[0]
    functions += "function visibleTiles(list)" + template.split("function visibleTiles(list)", 1)[1].split("// ── 鍵盤導航", 1)[0]
    functions += "function zoneSplitRows()" + template.split("function zoneSplitRows()", 1)[1].split("function groupTiles(", 1)[0]
    script = r"""
const assert = require('node:assert/strict');
const trainingScope = {selected_units: ['W0F00000-inner', 'W0F00000-edge']};
const _memoryByLighting = {W0F00000: {bytes_per_tile: 12582912, warning_tiles: 600}};
const _splitByLighting = {W0F00000: {decision_review: true}};
const base = {source: 'ok', zone: 'inner', dataset_role: 'train', validation_group: 'P1', decision: 'accept', auto_review: true, decision_review: true, validation_label: 'ng'};
let _currentLighting = 'W0F00000';
const _filter = 'all', _focusedTileId = null;
const elements = {};
const document = {getElementById: id => elements[id] ||= {innerHTML: '', textContent: ''}};
function updateBadge() {}
const _tilesByLighting = {W0F00000: [
  ...Array.from({length: 601}, () => ({...base})),
  ...Array.from({length: 20}, () => ({...base, zone: 'edge'})),
  ...Array.from({length: 200}, () => ({...base, dataset_role: 'calibration'})),
  ...Array.from({length: 200}, () => ({...base, dataset_role: 'acceptance'})),
  ...Array.from({length: 200}, () => ({...base, decision: 'reject'})),
  {...base, source: 'ng'},
]};
""" + functions + r"""
assert.deepEqual(trainingMemoryRows().map(r => r.count), [601, 20]);
assert.match(oomWarningText(), /W0F00000-inner：601/);
assert.ok(!oomWarningText().includes('W0F00000-edge'));
_tilesByLighting.W0F00000[0].decision = 'reject';
assert.deepEqual(trainingMemoryRows().map(r => r.count), [600, 20]);
assert.equal(oomWarningText(), '');
assert.equal(trainingMemoryRows()[0].featureGiB, 7.03125);
render();
assert.ok(!elements['content-area'].innerHTML.includes('待標記'));
assert.ok(!elements['content-area'].innerHTML.includes('labelValidationGroup'));
assert.ok(elements['content-area'].innerHTML.includes('保留＝OK，排除＝NG'));
assert.ok(elements['content-area'].innerHTML.includes('>NG</span>'));
assert.ok(elements['auto-split-summary'].innerHTML.includes('缺驗收 PANEL'));
trainingScope.selected_units.push('R0F00000-inner', 'R0F00000-edge');
_splitByLighting.R0F00000 = {decision_review: true};
_tilesByLighting.R0F00000 = [
  {...base, validation_group: 'TRAIN'},
  {...base, validation_group: 'CAL', dataset_role: 'calibration'},
  {...base, validation_group: 'cal', dataset_role: 'calibration', decision: 'reject'},
  {...base, validation_group: 'ACCEPT', dataset_role: 'acceptance'},
  {...base, validation_group: 'ACCEPT', dataset_role: 'acceptance', decision: 'reject'},
  {...base, validation_group: 'EDGE', zone: 'edge'},
];
_currentLighting = 'R0F00000';
assert.equal(zoneSplitRows()[0].calibration.panels, 1);
assert.deepEqual(zoneSplitRows()[0].missing, []);
assert.deepEqual(zoneSplitRows()[1].missing, ['缺校正 PANEL', '缺驗收 PANEL']);
_tilesByLighting.R0F00000[1].decision = 'reject';
assert.deepEqual(zoneSplitRows()[0].missing, ['校正缺 OK']);
showAutoSplit();
assert.ok(elements['auto-split-summary'].innerHTML.includes('校正缺 OK'));
assert.ok(elements['auto-split-summary'].innerHTML.includes('OK 0／NG 2'));
_tilesByLighting.R0F00000[1].decision = 'accept';
assert.deepEqual(zoneSplitRows()[0].missing, []);
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
