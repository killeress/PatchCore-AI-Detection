from capi_config import CAPIConfig


def test_default_scratch_config_enabled():
    cfg = CAPIConfig()
    assert cfg.scratch_classifier_enabled is True
    assert cfg.scratch_safety_multiplier == 1.1
    assert cfg.scratch_bundle_path == "deployment/scratch_classifier_v1.pkl"
    assert cfg.scratch_dinov2_weights_path == "deployment/dinov2_vitb14.pth"


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
