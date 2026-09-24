"""A newly scaled model must be installed with a compatible decision threshold."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from capi_model_registry import install_trained_submodel
from capi_normalization_config import OK_MAX_NORMALIZATION


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    mapping = {"G0F00000": {"inner": .18, "edge": .35}}
    config = {"machine_id": "M", "threshold_mapping": mapping, "image_size": [512, 512]}
    (root / "machine_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (root / "thresholds.json").write_text(json.dumps(mapping), encoding="utf-8")
    (root / "manifest.json").write_text('{"unit_metrics": {}}', encoding="utf-8")
    target = root / "G0F00000-inner.pt"
    config["model_mapping"] = {"G0F00000": {"inner": str(target)}}
    (root / "machine_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    target.write_bytes(b"old-model")
    source = tmp_path / "trained.pt"
    source.write_bytes(b"new-model")
    return root, source, target


def contents(root):
    return {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}


@pytest.mark.parametrize("has_threshold_json", [True, False])
def test_legacy_model_retrain_resets_only_its_threshold(bundle, has_threshold_json):
    root, source, target = bundle
    if not has_threshold_json:
        (root / "thresholds.json").unlink()
    reset = install_trained_submodel(source, target, {"normalization_mode": OK_MAX_NORMALIZATION})
    assert reset == {"previous": .18, "current": .35}
    recipe = yaml.safe_load((root / "machine_config.yaml").read_text(encoding="utf-8"))
    assert recipe["threshold_mapping"] == {"G0F00000": {"inner": .35, "edge": .35}}
    assert json.loads((root / "thresholds.json").read_text(encoding="utf-8")) == recipe["threshold_mapping"]
    assert recipe["image_size"] == [512, 512]
    assert target.read_bytes() == b"new-model"


def test_retrain_on_same_scale_preserves_operator_threshold(bundle):
    root, source, target = bundle
    manifest = {"unit_metrics": {target.stem: {"normalization_mode": OK_MAX_NORMALIZATION}}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    before = contents(root)
    assert install_trained_submodel(source, target, {"normalization_mode": OK_MAX_NORMALIZATION}) is None
    assert contents(root) == {**before, target.name: b"new-model"}


@pytest.mark.parametrize("mapping", [None, {}, {"STANDARD": {"inner": .82, "edge": .9}},
                                      {"U0F00000": {"edge": .7}, "STANDARD": {"inner": .82}}])
def test_new_u0f_threshold_does_not_reuse_standard(bundle, mapping):
    root, source, _ = bundle
    recipe = {"machine_id": "M"}
    if mapping is not None:
        recipe["threshold_mapping"] = mapping
    (root / "machine_config.yaml").write_text(yaml.safe_dump(recipe), encoding="utf-8")
    (root / "thresholds.json").write_text(json.dumps(mapping or {}), encoding="utf-8")
    target = root / "U0F00000-inner.pt"

    reset = install_trained_submodel(source, target, {"normalization_mode": OK_MAX_NORMALIZATION})

    assert reset == {"previous": None, "current": .35}
    expected = {**(mapping or {}), "U0F00000": {**((mapping or {}).get("U0F00000") or {}), "inner": .35}}
    actual = yaml.safe_load((root / "machine_config.yaml").read_text(encoding="utf-8"))
    assert actual["threshold_mapping"] == expected
    assert json.loads((root / "thresholds.json").read_text(encoding="utf-8")) == expected
    assert target.read_bytes() == b"new-model"
    assert actual["model_mapping"]["U0F00000"]["inner"] == str(target)


@pytest.mark.parametrize("yaml_value,json_value", [(.18, .19), (None, None)])
def test_retrain_accepts_legacy_scalar_or_empty_thresholds(bundle, yaml_value, json_value):
    root, source, target = bundle
    recipe = yaml.safe_load((root / "machine_config.yaml").read_text(encoding="utf-8"))
    recipe["threshold_mapping"]["G0F00000"] = yaml_value
    (root / "machine_config.yaml").write_text(yaml.safe_dump(recipe), encoding="utf-8")
    (root / "thresholds.json").write_text(json.dumps({"G0F00000": json_value}), encoding="utf-8")
    reset = install_trained_submodel(source, target, {"normalization_mode": OK_MAX_NORMALIZATION})
    assert reset == {"previous": yaml_value, "current": .35}
    recipe = yaml.safe_load((root / "machine_config.yaml").read_text(encoding="utf-8"))
    expected = {"inner": .35, **({"edge": yaml_value} if yaml_value is not None else {})}
    assert recipe["threshold_mapping"]["G0F00000"] == expected
    expected_json = {"inner": .35, **({"edge": json_value} if json_value is not None else {})}
    assert json.loads((root / "thresholds.json").read_text(encoding="utf-8"))["G0F00000"] == expected_json


def test_same_scale_retrain_repairs_missing_threshold_and_model_mapping(bundle):
    root, source, _ = bundle
    target = root / "U0F00000-inner.pt"
    (root / "manifest.json").write_text(json.dumps({"unit_metrics": {
        target.stem: {"normalization_mode": OK_MAX_NORMALIZATION},
    }}), encoding="utf-8")
    assert install_trained_submodel(source, target, {"normalization_mode": OK_MAX_NORMALIZATION}) == {
        "previous": None, "current": .35,
    }
    recipe = yaml.safe_load((root / "machine_config.yaml").read_text(encoding="utf-8"))
    assert recipe["model_mapping"]["U0F00000"]["inner"] == str(target)
    assert recipe["threshold_mapping"]["U0F00000"]["inner"] == .35


@pytest.mark.parametrize("text", [
    "# recipe\nthreshold_mapping:\n  STANDARD:\n    inner: 0.82 # keep\n# size\nimage_size: [512, 512]\n",
    "threshold_mapping: {STANDARD: {inner: 0.82}}\nimage_size: [512, 512]\n",
    "threshold_mapping:\n  U0F00000:\n    edge: 0.7\n  STANDARD:\n    inner: 0.82",
    "threshold_mapping:\nimage_size: [512, 512]\n",
    "threshold_mapping:\n  U0F00000:\n    inner:\n",
    "# recipe",
])
def test_missing_unit_yaml_insert_preserves_other_values_and_comments(text):
    from capi_model_registry import _upsert_unit_mapping_text
    updated = _upsert_unit_mapping_text(text, "threshold_mapping", "U0F00000", "inner", .5)
    expected = yaml.safe_load(text) or {}
    expected["threshold_mapping"] = expected.get("threshold_mapping") or {}
    expected["threshold_mapping"].setdefault("U0F00000", {})["inner"] = .5
    assert yaml.safe_load(updated) == expected
    for comment in ("# recipe", "# keep", "# size"):
        if comment in text:
            assert comment in updated


@pytest.mark.parametrize("failing_file", ["machine_config.yaml", "thresholds.json"])
def test_threshold_install_failure_restores_original_model_and_settings(bundle, monkeypatch, failing_file):
    import os
    root, source, target = bundle
    before = contents(root)
    replace = os.replace
    failed = False

    def fail_once(src, dst):
        nonlocal failed
        if Path(dst) == root / failing_file and not failed:
            failed = True
            raise OSError("disk failure")
        return replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_once)
    with pytest.raises(OSError, match="disk failure"):
        install_trained_submodel(source, target, {"normalization_mode": OK_MAX_NORMALIZATION})
    assert contents(root) == before


@pytest.mark.parametrize("relative_model_path", [False, True])
def test_runtime_reload_refreshes_threshold_for_retrained_unit(bundle, relative_model_path):
    from capi_inference import CAPIInferencer
    root, source, target = bundle
    engine = CAPIInferencer.__new__(CAPIInferencer)
    engine.base_dir = root.parent
    path = target.relative_to(root.parent) if relative_model_path else target
    engine.config = SimpleNamespace(
        model_mapping={"G0F00000": {"inner": str(path)}},
        threshold_mapping={"G0F00000": {"inner": .18, "edge": .35}},
    )
    key = ("M", "G0F00000", "inner")
    engine._model_cache_v2 = {key: object()}
    install_trained_submodel(source, target, {"normalization_mode": OK_MAX_NORMALIZATION})
    assert engine.reload_submodel(*key)
    assert key not in engine._model_cache_v2
    assert engine.config.threshold_mapping == {"G0F00000": {"inner": .35, "edge": .35}}


def test_new_screen_reload_does_not_take_model_from_another_bundle(bundle, tmp_path):
    from capi_inference import CAPIInferencer
    root, _, target = bundle
    other = tmp_path / "inactive"
    other.mkdir()
    (other / "machine_config.yaml").write_text(yaml.safe_dump({
        "model_mapping": {"U0F00000": {"inner": str(other / "U0F00000-inner.pt")}},
        "threshold_mapping": {"U0F00000": {"inner": .5}},
    }), encoding="utf-8")
    engine = CAPIInferencer.__new__(CAPIInferencer)
    engine.base_dir = tmp_path
    engine.config = SimpleNamespace(model_mapping={"STANDARD": {"inner": str(target)}},
                                    threshold_mapping={"STANDARD": {"inner": .82}})
    engine._model_cache_v2 = {}
    assert not engine.reload_submodel("M", "U0F00000", "inner")
    assert "U0F00000" not in engine.config.model_mapping
    assert engine.config.threshold_mapping == {"STANDARD": {"inner": .82}}


def test_runtime_reload_converts_legacy_scalar_without_changing_other_zone(bundle):
    from capi_inference import CAPIInferencer
    root, source, target = bundle
    recipe = yaml.safe_load((root / "machine_config.yaml").read_text(encoding="utf-8"))
    recipe["threshold_mapping"]["G0F00000"] = .18
    (root / "machine_config.yaml").write_text(yaml.safe_dump(recipe), encoding="utf-8")
    engine = CAPIInferencer.__new__(CAPIInferencer)
    engine.base_dir = root.parent
    engine.config = SimpleNamespace(model_mapping={"G0F00000": {"inner": str(target)}},
                                    threshold_mapping={"G0F00000": .18})
    engine._model_cache_v2 = {("M", "G0F00000", "inner"): object()}
    install_trained_submodel(source, target, {"normalization_mode": OK_MAX_NORMALIZATION})
    assert engine.reload_submodel("M", "G0F00000", "inner")
    assert engine.config.threshold_mapping == {"G0F00000": {"inner": .35, "edge": .18}}


def test_threshold_edit_syncs_legacy_scalar_json(bundle):
    from capi_model_registry import update_threshold
    root, _, _ = bundle
    (root / "thresholds.json").write_text('{"G0F00000": 0.18}', encoding="utf-8")
    db = SimpleNamespace(get_model_bundle=lambda _: {"bundle_path": str(root), "machine_id": "M"})
    update_threshold(db, 1, "G0F00000", "inner", .6)
    assert json.loads((root / "thresholds.json").read_text(encoding="utf-8")) == {
        "G0F00000": {"inner": .6, "edge": .18},
    }


def test_same_scale_missing_model_mapping_preserves_operator_threshold(bundle):
    root, source, target = bundle
    recipe_path = root / "machine_config.yaml"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    recipe.pop("model_mapping")
    recipe_path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"unit_metrics": {
        target.stem: {"normalization_mode": OK_MAX_NORMALIZATION},
    }}), encoding="utf-8")
    before_json = (root / "thresholds.json").read_bytes()
    assert install_trained_submodel(source, target, {"normalization_mode": OK_MAX_NORMALIZATION}) is None
    actual = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    assert actual["threshold_mapping"] == recipe["threshold_mapping"]
    assert actual["model_mapping"]["G0F00000"]["inner"] == str(target)
    assert (root / "thresholds.json").read_bytes() == before_json
