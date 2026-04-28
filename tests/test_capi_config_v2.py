"""
tests/test_capi_config_v2.py
測試 CAPIConfig 新架構欄位支援（machine_id, is_new_architecture, edge_threshold_px）
"""

import tempfile
from pathlib import Path
import yaml
from capi_config import CAPIConfig


def test_capi_config_legacy_yaml_no_machine_id():
    cfg_data = {
        "model_path": "model.pt",
        "model_mapping": {"G0F00000": "g.pt"},
        "threshold_mapping": {"G0F00000": 0.75},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg_data, f)
        path = f.name
    cfg = CAPIConfig.from_yaml(path)
    assert cfg.machine_id is None
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
    Path(path).unlink()
