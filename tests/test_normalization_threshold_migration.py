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
    assert reset == {"previous": .18, "current": .5}
    recipe = yaml.safe_load((root / "machine_config.yaml").read_text(encoding="utf-8"))
    assert recipe["threshold_mapping"] == {"G0F00000": {"inner": .5, "edge": .35}}
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
    assert engine.config.threshold_mapping == {"G0F00000": {"inner": .5, "edge": .35}}
