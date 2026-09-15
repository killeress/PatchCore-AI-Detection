import json
from types import SimpleNamespace

import pytest

from capi_model_provenance import snapshot_model_training, log_model_training


class Weights:
    def numel(self): return 2
    def min(self): return 1.0
    def max(self): return 1.8


def softpatch():
    cls = type("SoftPatchPlusModel", (), {"__module__": "capi_patchcore_softpatch"})
    return cls()


def write_manifest(tmp_path, params):
    (tmp_path / "manifest.json").write_text(json.dumps({"patchcore_params": params}))


def test_loaded_metadata_takes_priority_over_changed_manifest(tmp_path):
    model = softpatch()
    model.softpatch_weights = Weights()
    model.training_provenance = {"mode": "softpatch_plus_v1", "softpatch_plus_config": {
        "discriminator": "lof", "soft_weight": True, "weight_strength": .5}}
    write_manifest(tmp_path, {"feature_cleaning_mode": "off"})
    snapshot = snapshot_model_training(SimpleNamespace(model=SimpleNamespace(model=model)), tmp_path / "G0F00000-inner.pt")
    assert snapshot["source"] == "model"
    assert snapshot["discriminator"] == "lof"
    assert snapshot["weight_strength"] == .5
    assert snapshot["weight_min"] == 1 and snapshot["weight_max"] == 1.8


def test_previously_exported_softpatch_uses_manifest_options(tmp_path):
    write_manifest(tmp_path, {"feature_cleaning_mode": "softpatch_plus_v1", "softpatch_plus_config": {
        "discriminator": "lof_gaussian", "soft_weight": False, "weight_strength": 1}})
    snapshot = snapshot_model_training(SimpleNamespace(model=softpatch()), tmp_path / "G0F00000-inner.pt")
    assert snapshot["mode"] == "softpatch_plus_v1"
    assert snapshot["source"] == "manifest"
    assert snapshot["soft_weight"] is False


@pytest.mark.parametrize("zone,expected", [("inner", "context_overlap_adaptive_v1"), ("edge", "off")])
def test_manifest_respects_inner_only_scope(tmp_path, zone, expected):
    write_manifest(tmp_path, {"feature_cleaning_mode": "context_overlap_adaptive_v1", "feature_cleaning_scope": "inner_only"})
    snapshot = snapshot_model_training(SimpleNamespace(model=object()), tmp_path / f"G0F00000-{zone}.pt")
    assert snapshot["mode"] == expected


def test_manifest_respects_separate_zone_settings(tmp_path):
    write_manifest(tmp_path, {"feature_cleaning_mode": "off", "feature_cleaning_scope": "inner_only",
                             "feature_cleaning_by_zone": {"edge": {"mode": "knn_cosine_q99_v1"}}})
    assert snapshot_model_training(SimpleNamespace(model=object()), tmp_path / "G0F00000-edge.pt")["mode"] == "knn_cosine_q99_v1"


@pytest.mark.parametrize("content", [None, "{", "[]", '{"patchcore_params": {"feature_cleaning_mode": {}}}'])
def test_missing_or_bad_metadata_does_not_invent_cleaning_mode(tmp_path, content):
    if content is not None:
        (tmp_path / "manifest.json").write_text(content)
    snapshot = snapshot_model_training(SimpleNamespace(model=object()), tmp_path / "G0F00000-inner.pt")
    assert snapshot["mode"] == "unknown"


def test_recipe_cannot_label_plain_loaded_model_as_softpatch(tmp_path):
    write_manifest(tmp_path, {"feature_cleaning_mode": "softpatch_plus_v1"})
    assert snapshot_model_training(SimpleNamespace(model=object()), tmp_path / "G0F00000-inner.pt")["mode"] == "unknown"


def test_runtime_can_identify_softpatch_without_inventing_options():
    snapshot = snapshot_model_training(SimpleNamespace(model=softpatch()))
    assert snapshot["mode"] == "softpatch_plus_v1"
    assert "discriminator" not in snapshot and "soft_weight" not in snapshot


def test_cached_model_logs_again_on_next_pass_and_snapshots_survive_file_changes(tmp_path, capsys):
    write_manifest(tmp_path, {"feature_cleaning_mode": "off"})
    model = SimpleNamespace(model=object())
    model._capi_training_provenance = snapshot_model_training(model, tmp_path / "G0F00000-inner.pt")
    write_manifest(tmp_path, {"feature_cleaning_mode": "knn_cosine_q99_v1"})
    seen = set()
    log_model_training(model, "G0F00000", "inner", seen)
    log_model_training(model, "G0F00000", "inner", seen)
    log_model_training(model, "G0F00000", "inner", set())
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert all(json.loads(line.split("] ", 1)[1])["mode"] == "off" for line in lines)
