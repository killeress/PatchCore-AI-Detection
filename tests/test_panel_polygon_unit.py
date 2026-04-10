"""面板 4 角 polygon 功能的單元測試"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows cp950 console 無法顯示 Unicode 檢查記號，強制 utf-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import yaml
import numpy as np
from capi_config import CAPIConfig


def test_config_enable_panel_polygon_default_true():
    """enable_panel_polygon 預設必須為 True"""
    cfg = CAPIConfig()
    assert cfg.enable_panel_polygon is True, \
        f"expected default True, got {cfg.enable_panel_polygon}"
    print("✅ test_config_enable_panel_polygon_default_true")


def test_config_roundtrip_enable_panel_polygon():
    """from_dict / to_dict 必須保留 enable_panel_polygon 欄位"""
    cfg1 = CAPIConfig()
    cfg1.enable_panel_polygon = False
    d = cfg1.to_dict()
    assert "enable_panel_polygon" in d
    assert d["enable_panel_polygon"] is False

    cfg2 = CAPIConfig.from_dict(d)
    assert cfg2.enable_panel_polygon is False
    print("✅ test_config_roundtrip_enable_panel_polygon")


def test_config_yaml_loads_enable_panel_polygon():
    """從實際 capi_3f.yaml 讀取應該能抓到 enable_panel_polygon"""
    yaml_path = Path(__file__).resolve().parent.parent / "configs" / "capi_3f.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "enable_panel_polygon" in data, \
        f"capi_3f.yaml 缺少 enable_panel_polygon 欄位"
    assert data["enable_panel_polygon"] is True
    print("✅ test_config_yaml_loads_enable_panel_polygon")


if __name__ == "__main__":
    test_config_enable_panel_polygon_default_true()
    test_config_roundtrip_enable_panel_polygon()
    test_config_yaml_loads_enable_panel_polygon()
    print("\n✅ 所有 config 測試通過")
