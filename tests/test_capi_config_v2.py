"""
tests/test_capi_config_v2.py
測試 CAPIConfig 新架構欄位支援（machine_id, is_new_architecture, edge_threshold_px）
"""

import tempfile
from pathlib import Path
import yaml
from capi_config import CAPIConfig


def test_capi_config_legacy_yaml_default_machine_id():
    cfg_data = {
        "model_path": "model.pt",
        "model_mapping": {"G0F00000": "g.pt"},
        "threshold_mapping": {"G0F00000": 0.75},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg_data, f)
        path = f.name
    cfg = CAPIConfig.from_yaml(path)
    assert cfg.machine_id == "CAPI_3F"
    assert cfg.is_new_architecture is False
    assert cfg.model_mapping == {"G0F00000": "g.pt"}
    Path(path).unlink()


def test_capi_config_new_arch_yaml():
    cfg_data = {
        "machine_id": "GN160JCEL250S",
        "edge_threshold_px": 768,
        "model_mapping": {
            "G0F00000": {"inner": "g_inner.pt", "edge": "g_edge.pt"},
            "STANDARD": {"inner": "s_inner.pt", "edge": "s_edge.pt"},
        },
        "threshold_mapping": {
            "G0F00000": {"inner": 0.62, "edge": 0.71},
        },
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg_data, f)
        path = f.name
    cfg = CAPIConfig.from_yaml(path)
    assert cfg.machine_id == "GN160JCEL250S"
    assert cfg.is_new_architecture is True
    assert cfg.edge_threshold_px == 768
    assert cfg.model_mapping["G0F00000"]["inner"] == "g_inner.pt"
    assert cfg.threshold_mapping["G0F00000"]["inner"] == 0.62
    Path(path).unlink()


def test_apply_db_overrides_new_arch_threshold_mapping_keeps_nested_values():
    cfg = CAPIConfig(
        is_new_architecture=True,
        threshold_mapping={"G0F00000": {"inner": 0.5, "edge": 0.5}},
    )
    cfg.apply_db_overrides([
        {
            "param_name": "threshold_mapping",
            "decoded_value": {
                "G0F00000": {"inner": "0.42", "edge": "0.73"},
                "STANDARD": {"inner": 0.61, "edge": 0.82},
            },
        }
    ])

    assert cfg.threshold_mapping == {
        "G0F00000": {"inner": 0.5, "edge": 0.5},
    }


def test_capi_config_machine_id_alone_is_not_new_arch():
    """machine_id 存在但 model_mapping 為 flat dict → 仍視為 legacy。"""
    cfg_data = {
        "machine_id": "SOME_MACHINE",
        "model_mapping": {"G0F00000": "g.pt"},
        "threshold_mapping": {"G0F00000": 0.75},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg_data, f)
        path = f.name
    cfg = CAPIConfig.from_yaml(path)
    assert cfg.machine_id == "SOME_MACHINE"
    assert cfg.is_new_architecture is False  # model_mapping 是 flat
    Path(path).unlink()


def test_capi_config_preprocess_after_tiling_serialization():
    # 測試預設值
    cfg_default = CAPIConfig()
    assert cfg_default.preprocess_after_tiling is False

    # 測試從 yaml 讀取
    cfg_data = {
        "preprocess_after_tiling": True,
        "model_mapping": {"G0F00000": "g.pt"},
        "threshold_mapping": {"G0F00000": 0.75},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg_data, f)
        path = f.name
    try:
        cfg = CAPIConfig.from_yaml(path)
        assert cfg.preprocess_after_tiling is True
        
        # 測試 to_dict
        d = cfg.to_dict()
        assert d["preprocess_after_tiling"] is True

        # 測試 to_yaml
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "output.yaml"
            cfg.to_yaml(str(out_path))
            
            with open(out_path, "r", encoding="utf-8") as rf:
                loaded = yaml.safe_load(rf)
            assert loaded["preprocess_after_tiling"] is True
    finally:
        Path(path).unlink()


def test_aoi_heatmap_center_seed_enabled_serialization():
    cfg = CAPIConfig.from_dict({
        "aoi_heatmap_center_seed_enabled": False,
    })

    assert cfg.aoi_heatmap_center_seed_enabled is False
    assert cfg.to_dict()["aoi_heatmap_center_seed_enabled"] is False

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "output.yaml"
        cfg.to_yaml(str(out_path))

        with open(out_path, "r", encoding="utf-8") as rf:
            loaded = yaml.safe_load(rf)
        assert loaded["aoi_heatmap_center_seed_enabled"] is False


def test_report_result_defect_codes_defaults_and_db_overrides():
    cfg = CAPIConfig()
    assert cfg.report_black_dot_defect_code == "PCDK2"
    assert cfg.report_white_dot_defect_code == "PTMD6"
    assert cfg.report_unknown_dot_defect_code == "PCDK2"
    assert cfg.report_bomb_defect_code == "PCDK3"
    assert cfg.report_image_abnormal_defect_code == "PCO05"

    cfg.apply_db_overrides([
        {"param_name": "report_black_dot_defect_code", "decoded_value": "B1234"},
        {"param_name": "report_white_dot_defect_code", "decoded_value": "W1234"},
        {"param_name": "report_unknown_dot_defect_code", "decoded_value": "U1234"},
        {"param_name": "report_bomb_defect_code", "decoded_value": "X1234"},
        {"param_name": "report_image_abnormal_defect_code", "decoded_value": "H1234"},
    ])

    assert cfg.report_black_dot_defect_code == "B1234"
    assert cfg.report_white_dot_defect_code == "W1234"
    assert cfg.report_unknown_dot_defect_code == "U1234"
    assert cfg.report_bomb_defect_code == "X1234"
    assert cfg.report_image_abnormal_defect_code == "H1234"
    assert cfg.to_dict()["report_white_dot_defect_code"] == "W1234"
    assert cfg.to_dict()["report_bomb_defect_code"] == "X1234"


def test_image_abnormal_settings_defaults_fallback_and_db_overrides():
    cfg = CAPIConfig.from_dict({"omit_overexposure_mean_threshold": 82})

    assert cfg.image_abnormal_detection_enabled is False
    assert cfg.image_abnormal_standard_mean_lower == 47
    assert cfg.image_abnormal_standard_mean_upper == 67
    assert cfg.image_abnormal_wgf50500_mean_lower == 50
    assert cfg.image_abnormal_wgf50500_mean_upper == 70
    assert cfg.image_abnormal_g0f00000_mean_lower == 46
    assert cfg.image_abnormal_g0f00000_mean_upper == 66
    assert cfg.image_abnormal_r0f00000_mean_lower == 50
    assert cfg.image_abnormal_r0f00000_mean_upper == 70
    assert cfg.image_abnormal_w0f00000_mean_lower == 49
    assert cfg.image_abnormal_w0f00000_mean_upper == 69
    assert cfg.image_abnormal_b0f00000_mean_lower == 0
    assert cfg.image_abnormal_b0f00000_mean_upper == 12

    legacy = CAPIConfig.from_dict({"image_abnormal_w0f00000_mean_threshold": 91})
    assert legacy.image_abnormal_w0f00000_mean_lower == 49
    assert legacy.image_abnormal_w0f00000_mean_upper == 91

    cfg.apply_db_overrides([
        {"param_name": "image_abnormal_detection_enabled", "decoded_value": True},
        {"param_name": "image_abnormal_w0f00000_mean_lower", "decoded_value": 51},
        {"param_name": "image_abnormal_w0f00000_mean_upper", "decoded_value": 91},
        {"param_name": "image_abnormal_b0f00000_mean_lower", "decoded_value": 1},
        {"param_name": "image_abnormal_b0f00000_mean_upper", "decoded_value": 77},
    ])

    assert cfg.image_abnormal_detection_enabled is True
    assert cfg.image_abnormal_w0f00000_mean_lower == 51
    assert cfg.image_abnormal_w0f00000_mean_upper == 91
    assert cfg.image_abnormal_b0f00000_mean_lower == 1
    assert cfg.image_abnormal_b0f00000_mean_upper == 77
    assert cfg.to_dict()["image_abnormal_w0f00000_mean_upper"] == 91


def test_within_spec_judgment_rules_defaults_and_overrides():
    cfg = CAPIConfig()
    default_rules = cfg.within_spec_judgment_rules["default"]

    assert default_rules["screens"]["STANDARD"]["black_dot"]["area_threshold_mm"] == 0.3
    assert default_rules["screens"]["STANDARD"]["black_dot"]["defect_code"] == "C1111"
    assert default_rules["screens"]["STANDARD"]["white_dot"]["defect_code"] == "C1111"
    assert default_rules["screens"]["STANDARD"]["white_dot"]["screen_count_limit"] == 1
    assert default_rules["screens"]["STANDARD"]["black_dot"]["tile_count_threshold"] == 2
    assert default_rules["screens"]["B0F00000"]["black_dot"]["enabled"] is False
    assert default_rules["screens"]["B0F00000"]["white_dot"]["area_threshold_mm"] == 0.2
    assert default_rules["screens"]["B0F00000"]["white_dot"]["screen_count_limit"] == 1
    assert default_rules["dot_detection"]["diff_threshold"] == 4
    assert default_rules["dot_detection"]["segmentation_method"] == "background_diff"
    assert default_rules["dot_detection"]["hysteresis_low_threshold"] == 2
    assert default_rules["dot_detection"]["hysteresis_high_threshold"] == 4
    assert default_rules["dot_detection"]["hysteresis_edge_width_percent"] == 3.0
    assert default_rules["dot_detection"]["hysteresis_edge_extra_threshold"] == 2
    assert default_rules["dot_detection"]["hysteresis_second_low_threshold"] == 3
    assert default_rules["dot_detection"]["hysteresis_second_high_threshold"] == 4
    assert default_rules["dot_detection"]["hysteresis_second_edge_width_percent"] == 9.5
    assert default_rules["dot_detection"]["hysteresis_second_edge_extra_threshold"] == 2
    assert default_rules["dot_detection"]["hysteresis_switch_count_threshold"] == 5
    assert default_rules["dot_detection"]["hysteresis_second_max_count"] == 5
    assert default_rules["dot_detection"]["hysteresis_edge_suppress_percent"] == 0.0
    assert default_rules["dot_detection"]["background_kernel"] == 33
    assert default_rules["dot_detection"]["morph_open"] == 0
    assert default_rules["dot_detection"]["min_aspect_ratio"] == 0.45
    assert default_rules["dot_detection"]["edge_margin_px"] == 4

    custom_rules = {
        "CAPI_3F": {},
        "MACHINE_A": {
            "screens": {
                "STANDARD": {
                    "black_dot": {
                        "enabled": True,
                        "area_threshold_mm": 0.45,
                        "screen_count_limit": 4,
                        "tile_count_threshold": 3,
                    },
                },
            },
        },
    }
    cfg2 = CAPIConfig.from_dict({"within_spec_judgment_rules": custom_rules})
    assert "CAPI_3F" not in cfg2.within_spec_judgment_rules
    assert "default" in cfg2.within_spec_judgment_rules
    assert cfg2.within_spec_judgment_rules["MACHINE_A"]["screens"]["STANDARD"]["black_dot"]["area_threshold_mm"] == 0.45
    assert cfg2.within_spec_judgment_rules["MACHINE_A"]["screens"]["STANDARD"]["black_dot"]["defect_code"] == "C1111"
    assert cfg2.to_dict()["within_spec_judgment_rules"]["MACHINE_A"]["screens"]["STANDARD"]["black_dot"]["defect_code"] == "C1111"

    cfg2.apply_db_overrides([
        {"param_name": "within_spec_judgment_rules", "decoded_value": cfg.within_spec_judgment_rules}
    ])
    assert cfg2.within_spec_judgment_rules["default"]["screens"]["W0F00000"]["white_dot"]["area_threshold_mm"] == 0.3
