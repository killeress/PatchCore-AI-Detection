from capi_config import CAPIConfig


def test_default_scratch_config_enabled():
    cfg = CAPIConfig()
    assert cfg.scratch_classifier_enabled is True
    assert cfg.scratch_safety_multiplier == 1.1
    assert cfg.scratch_bundle_path == "deployment/scratch_classifier_v1.pkl"
    assert cfg.scratch_dinov2_weights_path == "deployment/dinov2_vitb14.pth"
    # Offline-deployment path: repo code directory (source='local')
    assert cfg.scratch_dinov2_repo_path == ""   # empty = fall back to torch.hub


def test_dinov2_repo_path_roundtrip(tmp_path):
    """scratch_dinov2_repo_path flows through yaml + from_dict + to_dict + apply_db_overrides."""
    import yaml
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "scratch_dinov2_repo_path": "/opt/capi/deployment/dinov2_repo",
    }))
    cfg = CAPIConfig.from_yaml(str(cfg_path))
    assert cfg.scratch_dinov2_repo_path == "/opt/capi/deployment/dinov2_repo"
    assert "scratch_dinov2_repo_path" in cfg.to_dict()

    cfg2 = CAPIConfig()
    cfg2.apply_db_overrides([
        {"param_name": "scratch_dinov2_repo_path",
         "decoded_value": "/override/repo"},
    ])
    assert cfg2.scratch_dinov2_repo_path == "/override/repo"


def test_yaml_roundtrip_preserves_scratch(tmp_path):
    import yaml
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "scratch_classifier_enabled": False,
        "scratch_safety_multiplier": 1.25,
        "scratch_bundle_path": "/custom/path.pkl",
    }))
    cfg = CAPIConfig.from_yaml(str(cfg_path))
    assert cfg.scratch_classifier_enabled is False
    assert cfg.scratch_safety_multiplier == 1.25
    assert cfg.scratch_bundle_path == "/custom/path.pkl"


def test_apply_db_overrides_scratch_fields():
    """Spec §F: scratch_* fields must be overridable via DB config_params."""
    cfg = CAPIConfig()

    # Simulate what apply_db_overrides would see from DB (list of dicts with param_name/decoded_value)
    db_overrides = [
        {"param_name": "scratch_classifier_enabled", "decoded_value": False},
        {"param_name": "scratch_safety_multiplier", "decoded_value": 1.3},
        {"param_name": "scratch_bundle_path", "decoded_value": "/opt/custom/bundle.pkl"},
        {"param_name": "scratch_dinov2_weights_path", "decoded_value": "/opt/custom/dinov2.pth"},
    ]
    cfg.apply_db_overrides(db_overrides)
    assert cfg.scratch_classifier_enabled is False
    assert cfg.scratch_safety_multiplier == 1.3
    assert cfg.scratch_bundle_path == "/opt/custom/bundle.pkl"
    assert cfg.scratch_dinov2_weights_path == "/opt/custom/dinov2.pth"


def test_to_dict_includes_scratch_fields():
    """to_dict must include all scratch_* fields."""
    cfg = CAPIConfig()
    d = cfg.to_dict()
    assert d["scratch_classifier_enabled"] is True
    assert d["scratch_safety_multiplier"] == 1.1
    assert d["scratch_bundle_path"] == "deployment/scratch_classifier_v1.pkl"
    assert d["scratch_dinov2_weights_path"] == "deployment/dinov2_vitb14.pth"


def test_to_yaml_roundtrip_full(tmp_path):
    """Modify → to_yaml → from_yaml preserves scratch values."""
    cfg = CAPIConfig()
    cfg.scratch_safety_multiplier = 1.33
    cfg.scratch_classifier_enabled = False
    out_path = tmp_path / "roundtrip.yaml"
    cfg.to_yaml(str(out_path))
    reloaded = CAPIConfig.from_yaml(str(out_path))
    assert reloaded.scratch_safety_multiplier == 1.33
    assert reloaded.scratch_classifier_enabled is False
