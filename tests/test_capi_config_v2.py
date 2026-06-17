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
