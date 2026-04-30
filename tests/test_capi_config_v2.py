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
        "G0F00000": {"inner": 0.42, "edge": 0.73},
        "STANDARD": {"inner": 0.61, "edge": 0.82},
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
