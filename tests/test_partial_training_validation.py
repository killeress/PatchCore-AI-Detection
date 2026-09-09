"""Partial wizard calibration, report provenance and replacement failure tests (no GPU)."""
import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from capi_training_validation import (
    build_report, evaluate_model, load_report, normalize_validation_config,
    run_manifest_path, seal_reports, sha256_file, write_json,
)
from capi_web import CAPIWebHandler


def validation_config(tmp_path, selected_zones=("inner", "edge"), modes=None):
    paths = [str(tmp_path / f"panel-{i}") for i in range(6)]
    raw = {"split_mode": "auto_panel", "selected_zones": list(selected_zones),
           "panels": {p: {"group": Path(p).name} for p in paths}}
    return normalize_validation_config(raw, paths, modes or ["full"] * 6)


@pytest.mark.parametrize("zone", ["inner", "edge"])
def test_partial_split_ignores_unselected_zone_and_survives_preprocess(tmp_path, zone):
    cfg = validation_config(tmp_path, [zone], ["inner_only"] * 3 + ["edge_only"] * 3)
    selected = [p for p in cfg["panels"].values() if zone in p["zones"]]
    assert sorted(p["role"] for p in selected) == ["acceptance", "calibration", "train"]
    assert all(p["role"] == "train" for p in cfg["panels"].values() if not p["zones"])
    assert normalize_validation_config(cfg) == cfg
    assert normalize_validation_config(cfg, list(cfg["panels"]), ["inner_only"] * 3 + ["edge_only"] * 3) == cfg


def sample_scores():
    return [{"role": role, "label": label, "score": score, "group": role,
             "tile_id": index, "asset_path": "validation_reports/test.png"}
            for index, (role, label, score) in enumerate([
                ("calibration", "ok", 0.4), ("calibration", "ng", 0.6),
                ("acceptance", "ok", 0.45), ("acceptance", "ng", 0.7),
            ])]


def test_existing_threshold_is_frozen_as_baseline_and_tie_breaker():
    report = build_report(sample_scores(), {}, baseline_threshold=0.55)
    assert report["baseline_threshold"] == report["suggested_threshold"] == 0.55
    assert report["baseline"]["false_positives"] == 0
    assert report["acceptance_by_batch"]["acceptance"]["baseline"] == report["baseline"]
    changed = sample_scores()
    changed[-1]["score"] = 0.1
    assert build_report(changed, {}, baseline_threshold=0.55)["suggested_threshold"] == 0.55


def file_contents(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def partial_run(tmp_path, monkeypatch):
    import capi_train_new as training
    import capi_model_registry as registry

    root = tmp_path / "bundle"
    root.mkdir()
    units = ["W0F00000-inner", "W0F00000-edge", "G0F00000-inner"]
    thresholds = {"W0F00000": {"inner": 0.55, "edge": 0.75}, "G0F00000": {"inner": 0.35}}
    recipe = {"machine_id": "M", "threshold_mapping": thresholds, "image_preprocess_pipeline": []}
    (root / "machine_config.yaml").write_text(yaml.safe_dump(recipe), encoding="utf-8")
    write_json(root / "thresholds.json", thresholds)
    manifest = {"machine_id": "M", "trained_with_job_id": "original", "unit_metrics": {},
                "tiles_per_unit": {}, "model_files": {}, "patchcore_params": {
                    "precision": "float32", "feature_layers": "layer3", "image_size": [256, 256],
                }, "image_preprocess_pipeline": [], "tile_stride": 128}
    for unit in units:
        pt = root / f"{unit}.pt"
        pt.write_bytes(("old:" + unit).encode())
        report = build_report(sample_scores(), {})
        report.update(unit_label=unit, model_file=pt.name, model_sha256=sha256_file(pt))
        relative = f"validation_reports/{unit}/report.json"
        write_json(root / relative, report)
        manifest["unit_metrics"][unit] = {"auroc": 0.8, "train_count": 40,
                                          "validation": {"report_path": relative}}
        manifest["tiles_per_unit"][unit] = {"train": 40, "ng": 1}
        manifest["model_files"][unit] = {"path": pt.name, "size_bytes": pt.stat().st_size}
    write_json(root / "manifest.json", manifest)
    seal_reports(root, manifest["unit_metrics"])
    bundle = {"id": 7, "machine_id": "M", "bundle_path": str(root), "job_id": "original", "is_active": False}
    jobs = {"original": {"job_id": "original", "machine_id": "M", "state": "completed", "output_bundle": str(root)}}
    db = SimpleNamespace(
        get_training_job=lambda job_id: jobs.get(job_id),
        get_model_bundle=lambda _: bundle,
        list_model_bundles=lambda _: [bundle],
        list_tile_pool=lambda *a, **k: [],
        update_training_job_state=lambda job_id, state, **fields: jobs[job_id].update(state=state, **fields),
    )
    server = SimpleNamespace(database=db, _gpu_lock=threading.Lock(), inferencers={"M": MagicMock()})
    monkeypatch.setattr(CAPIWebHandler, "_train_new_jobs", {})
    monkeypatch.setattr(CAPIWebHandler, "_train_new_jobs_lock", threading.Lock())
    monkeypatch.setattr(CAPIWebHandler, "_train_slot", {"lock": threading.Lock(), "active_job_id": None})
    monkeypatch.setattr(CAPIWebHandler, "_cancel_and_wait_scan_idle", lambda **kw: None)
    monkeypatch.setattr(CAPIWebHandler, "_free_server_gpu_cache", lambda: None)
    monkeypatch.setattr(CAPIWebHandler, "_cleanup_train_new_job_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(CAPIWebHandler, "_load_train_new_config", lambda _: {
        "over_review_root": tmp_path, "output_root": tmp_path, "backbone_cache_dir": tmp_path, "required_backbones": [],
    })
    invalidation = MagicMock(return_value=0)
    monkeypatch.setattr(registry, "invalidate_score_cache", invalidation)
    env = SimpleNamespace(root=root, db=db, server=server, jobs=jobs, bundle=bundle, units=units,
                          failure=None, incomplete=False, invalidation=invalidation, calls=[])

    def train(**kw):
        unit = f"{kw['lighting']}-{kw['zone']}"
        env.calls.append(kw)
        assert file_contents(root) == env.before  # No earlier unit has been installed yet.
        assert kw["output_pt_path"].parent != root
        assert kw["cfg"].validation_config["split_mode"] == "auto_panel"
        assert kw["cfg"].precision == "float32"
        assert kw["cfg"].image_size == (256, 256)
        assert kw["cfg"].tile_stride == 128
        assert kw["baseline_threshold"] == thresholds[kw["lighting"]][kw["zone"]]
        assert kw["fail_on_validation_error"]
        model = kw["output_pt_path"]
        model.write_bytes((kw["job_id"] + ":" + unit).encode())
        if len(env.calls) == 2:
            if env.failure in ("training", "evaluation"):
                raise RuntimeError(env.failure + " failed")
            if env.failure == "cancel":
                kw["cancel_event"].set()
        scores = sample_scores() if not env.incomplete else []
        report = build_report(scores, kw["cfg"].validation_config, baseline_threshold=kw["baseline_threshold"])
        report.update(unit_label=unit, model_file=model.name, model_sha256=sha256_file(model), job_id=kw["job_id"])
        relative = f"validation_reports/{kw['job_id']}/{unit}/report.json"
        write_json(model.parent / relative, report)
        return {"metrics": {"auroc": 0.9, "train_count": 30, "validation": {
                    "report_path": relative, "status": report["status"], "job_id": kw["job_id"],
                }}, "tile_count": 30, "ng_count": 1, "ng_used": "independent_calibration",
                "used_tile_ids": [1, 2], "size_bytes": model.stat().st_size, "threshold": 0.35, "elapsed_seconds": 1}

    monkeypatch.setattr(training, "train_single_submodel", train)

    def run(job_id="partial-1"):
        cfg = validation_config(tmp_path)
        jobs[job_id] = {"job_id": job_id, "machine_id": "M", "state": "train", "panel_paths": list(cfg["panels"]),
                       "panel_modes": ["full"] * 6, "tile_stride": 128,
                       "training_params": {"validation_config": cfg},
                       "training_scope": {"mode": "partial", "target_bundle_id": 7, "selected_units": units[:2]}}
        env.before = file_contents(root)
        env.calls = []
        CAPIWebHandler._train_slot["active_job_id"] = job_id
        CAPIWebHandler._train_new_partial_training_worker(job_id, server)
        assert CAPIWebHandler._train_slot["active_job_id"] is None
        return jobs[job_id]

    env.run = run
    return env


def test_partial_replaces_only_selected_pts_and_preserves_thresholds_and_history(partial_run):
    env = partial_run
    assert env.run()["state"] == "completed"
    root = env.root
    current = file_contents(root)
    for relative, content in env.before.items():
        if relative not in {"manifest.json", *(f"{u}.pt" for u in env.units[:2])}:
            assert current[relative] == content
    for unit in env.units[:2]:
        report = load_report(root, unit)
        assert not report["stale"]
        assert report["job_id"] == "partial-1"
        assert report["baseline_threshold"] == (0.55 if unit.endswith("inner") else 0.75)
    snapshot = json.loads(run_manifest_path(root, "partial-1").read_text(encoding="utf-8"))
    assert set(snapshot["unit_metrics"]) == set(env.units[:2])
    assert snapshot["patchcore_params"]["precision"] == "float32"
    original = json.loads(run_manifest_path(root, "original").read_text(encoding="utf-8"))
    assert all(m["train_count"] == 40 for m in original["unit_metrics"].values())
    assert env.invalidation.call_count == 2
    assert env.server.inferencers["M"].reload_submodel.call_count == 2


@pytest.mark.parametrize("failure", ["training", "evaluation", "cancel", "seal", "install"])
def test_partial_failure_keeps_all_original_pts_and_reports(partial_run, monkeypatch, failure):
    import capi_training_validation as validation
    import os
    env = partial_run
    env.failure = failure
    if failure == "seal":
        def fail_seal(*a):
            raise OSError("report disk failure")
        monkeypatch.setattr(validation, "seal_reports", fail_seal)
    if failure == "install":
        replace = os.replace
        failed = False
        def fail_last_file(source, target):
            nonlocal failed
            if Path(target) == env.root / "manifest.json" and not failed:
                failed = True
                raise OSError("manifest disk failure")
            return replace(source, target)
        monkeypatch.setattr(os, "replace", fail_last_file)
    assert env.run()["state"] == "failed"
    assert file_contents(env.root) == env.before
    env.invalidation.assert_not_called()
    env.server.inferencers["M"].reload_submodel.assert_not_called()


@pytest.mark.parametrize("failure", ["cache", "completion_record"])
def test_post_install_failure_restores_original_files(partial_run, failure):
    env = partial_run
    if failure == "cache":
        env.invalidation.side_effect = RuntimeError("cache database failure")
    else:
        update = env.db.update_training_job_state
        def fail_completion(job_id, state, **fields):
            if state == "completed":
                raise RuntimeError("completion database failure")
            update(job_id, state, **fields)
        env.db.update_training_job_state = fail_completion
    assert env.run()["state"] == "failed"
    assert file_contents(env.root) == env.before
    env.server.inferencers["M"].reload_submodel.assert_not_called()


@pytest.mark.parametrize("changed_file", ["manifest.json", "machine_config.yaml"])
def test_partial_preserves_changes_made_to_target_during_training(partial_run, monkeypatch, changed_file):
    import capi_training_validation as validation
    env = partial_run
    seal = validation.seal_reports
    changed = None
    def change_target_after_evaluation(*args):
        nonlocal changed
        seal(*args)
        path = env.root / changed_file
        if changed_file == "manifest.json":
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["unit_metrics"][env.units[-1]]["auroc"] = 0.99
            write_json(path, manifest)
        else:
            recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
            recipe["preprocess_after_tiling"] = True
            path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
        changed = path.read_bytes()
    monkeypatch.setattr(validation, "seal_reports", change_target_after_evaluation)
    assert env.run()["state"] == "failed"
    assert file_contents(env.root) == {**env.before, changed_file: changed}
    env.invalidation.assert_not_called()


@pytest.mark.parametrize("message", ["CUDA out of memory", "incorrect inference shape"])
def test_partial_evaluation_errors_propagate_instead_of_publishing_model(tmp_path, monkeypatch, message):
    import sys
    import cv2
    import numpy as np
    source = tmp_path / "sample.png"
    cv2.imwrite(str(source), np.full((8, 8, 3), 100, dtype=np.uint8))
    def fail_predict(*args):
        raise ValueError(message)
    monkeypatch.setitem(sys.modules, "anomalib.deploy", SimpleNamespace(
        TorchInferencer=lambda **kw: SimpleNamespace(predict=fail_predict)))
    frozen = ([{"tile_id": 1, "asset_path": "sample.png", "role": "acceptance", "label": "ok"}], [])
    with pytest.raises(ValueError, match=message):
        evaluate_model(tmp_path / "candidate.pt", [], [], {}, tmp_path, "W0F00000-inner", lambda _: None,
                       frozen_inputs=frozen, job_id="partial", fail_on_error=True)
    assert not (tmp_path / "validation_reports/partial/W0F00000-inner/report.json").exists()


def handler(server, path, payload=None):
    h = CAPIWebHandler.__new__(CAPIWebHandler)
    h._capi_server_instance = server
    h.path = path
    raw = json.dumps(payload or {}).encode()
    h.headers = {"Content-Length": str(len(raw))}
    h.rfile = io.BytesIO(raw)
    h.responses = []
    h._send_json = lambda body, status=200, **kw: h.responses.append((status, body))
    h._send_response = lambda status, body, **kw: h.responses.append((status, body))
    return h


def test_partial_adoption_changes_only_requested_unit(partial_run):
    env = partial_run
    env.run()
    h = handler(env.server, "/api/train/new/apply-validation/partial-1", {"unit_label": "W0F00000-edge"})
    h._handle_train_new_apply_validation()
    assert h.responses[0][0] == 200
    values = yaml.safe_load((env.root / "machine_config.yaml").read_text())["threshold_mapping"]
    assert values["W0F00000"]["inner"] == 0.55
    assert values["W0F00000"]["edge"] == load_report(env.root, "W0F00000-edge")["suggested_threshold"]
    assert values["G0F00000"]["inner"] == 0.35
    assert not env.bundle["is_active"]
    assert (env.root / "validation_reports/partial-1/W0F00000-edge/adoptions.json").exists()


def test_partial_cannot_adopt_report_for_unselected_pt(partial_run):
    env = partial_run
    env.run()
    h = handler(env.server, "/api/train/new/apply-validation/partial-1", {"unit_label": "G0F00000-inner"})
    h._handle_train_new_apply_validation()
    assert h.responses[0][0] == 400
    assert "重訓範圍" in h.responses[0][1]["error"]


def test_insufficient_data_still_trains_but_cannot_adopt_threshold(partial_run):
    env = partial_run
    env.incomplete = True
    assert env.run()["state"] == "completed"
    assert load_report(env.root, env.units[0])["status"] == "insufficient"
    h = handler(env.server, "/api/train/new/apply-validation/partial-1", {"unit_label": env.units[0]})
    h._handle_train_new_apply_validation()
    assert h.responses[0][0] == 400
    assert "資料不足" in h.responses[0][1]["error"]


def test_repeated_partial_runs_keep_old_report_and_reject_old_adoption(partial_run, monkeypatch):
    env = partial_run
    env.run()
    report_path = f"validation_reports/partial-1/{env.units[0]}/report.json"
    original = (env.root / report_path).read_bytes()
    assert env.run("partial-2")["state"] == "completed"
    assert (env.root / report_path).read_bytes() == original
    assert load_report(env.root, env.units[0], report_path)["stale"]
    h = handler(env.server, "/api/train/new/apply-validation/partial-1", {"unit_label": env.units[0]})
    h._handle_train_new_apply_validation()
    assert h.responses[0][0] == 400
    h = handler(env.server, "/train/new/done/partial-1")
    h.jinja_env = MagicMock()
    monkeypatch.setattr(CAPIWebHandler, "_train_new_lighting_labels", lambda *a: {})
    monkeypatch.setattr(CAPIWebHandler, "_train_new_station_adapter", lambda *a: None)
    h._handle_train_new_done_page()
    assert h.responses[0][0] == 200
    rendered = h.jinja_env.get_template.return_value.render.call_args.kwargs
    assert {unit for unit, _ in rendered["units"]} == set(env.units[:2])
    assert all(r["job_id"] == "partial-1" and r["stale"] for r in rendered["validation_reports"])
    edge = next(r for r in rendered["validation_reports"] if r["unit_label"].endswith("edge"))
    assert edge["errors"][0]["label"] == "ng" and edge["errors"][0]["baseline_wrong"]


def test_partial_pages_render_enabled_validation_and_frozen_threshold(partial_run, monkeypatch):
    import re
    import shutil
    import subprocess
    from jinja2 import Environment, FileSystemLoader
    env = partial_run
    env.run()
    template_env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
    template_env.globals["app_version"] = {"version": "test"}
    monkeypatch.setattr(CAPIWebHandler, "_list_open_train_new_jobs", lambda *a: [])
    monkeypatch.setattr(CAPIWebHandler, "_all_train_unit_labels", lambda *a: env.units)
    h = handler(env.server, "/train/new/select?mode=partial&target_bundle_id=7&units=W0F00000-inner")
    h.jinja_env = template_env
    h._handle_train_new_select_page()
    assert h.responses[0][0] == 200
    html = h.responses[0][1]
    checkbox = re.search(r'<input id="validation-auto"[^>]*>', html).group()
    assert "checked" in checkbox and "disabled" not in checkbox
    node = shutil.which("node")
    if node:
        for attributes, script in re.findall(r"<script([^>]*)>(.*?)</script>", html, re.S):
            args = [node, "--check"] + (["--input-type=module"] if 'type="module"' in attributes else [])
            result = subprocess.run(args, input=script, text=True, capture_output=True, encoding="utf-8")
            assert result.returncode == 0, result.stderr
    h = handler(env.server, "/train/new/done/partial-1")
    h.jinja_env = template_env
    monkeypatch.setattr(CAPIWebHandler, "_train_new_lighting_labels", lambda *a: {})
    monkeypatch.setattr(CAPIWebHandler, "_train_new_station_adapter", lambda *a: None)
    env.bundle["is_active"] = True
    h._handle_train_new_done_page()
    assert h.responses[0][0] == 200
    html = h.responses[0][1]
    assert "局部重訓完成" in html and "本次重訓 PT 大小" in html
    assert "驗收／原門檻 0.75" in html
    assert "停用後可採用建議門檻" in html
    assert 'data-unit="W0F00000-edge"' not in html


@pytest.mark.parametrize("override", [{}, {"precision": "float16", "feature_layers": "layer2_layer3"}])
def test_partial_memory_estimate_uses_inherited_settings(partial_run, override):
    from capi_train_new import estimate_patchcore_feature_memory
    env = partial_run
    env.run()
    env.jobs["partial-1"]["training_params"].update(override)
    h = handler(env.server, "/api/train/new/tiles?job_id=partial-1&lighting=W0F00000")
    h._handle_train_new_tiles()
    assert h.responses[0][0] == 200
    expected = estimate_patchcore_feature_memory(1, image_size=(256, 256),
        precision=override.get("precision", "float32"), feature_layers=override.get("feature_layers", "layer3"))
    assert h.responses[0][1]["training_memory"] == expected


def test_heldout_ok_tiles_do_not_appear_as_pending_training_changes(tmp_path):
    from capi_model_registry import get_pending_change_count
    write_json(tmp_path / "manifest.json", {"submodel_history": {"W0F00000-inner": [{"job_id": "partial", "used_tile_ids": [1]}]}})
    db = SimpleNamespace(list_tile_pool=lambda job, **kw: [{"id": 1, "dataset_role": "train"},
        {"id": 2, "dataset_role": "calibration"}, {"id": 3, "dataset_role": "acceptance"}])
    assert get_pending_change_count(db, {"bundle_path": str(tmp_path), "job_id": "original"}, "W0F00000", "inner") == 0
