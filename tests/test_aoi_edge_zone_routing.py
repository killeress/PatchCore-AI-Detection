"""新架構 AOI 座標邊緣 zone-aware routing 單元測試。"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer


@pytest.fixture
def legacy_inferencer():
    """舊架構 (5-model)：is_new_architecture=False，flat model_mapping。"""
    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.is_new_architecture = False
    cfg.model_mapping = {"G0F00000": "/fake/legacy.pt"}
    cfg.threshold_mapping = {"G0F00000": 0.55}
    inf = CAPIInferencer.__new__(CAPIInferencer)
    inf.config = cfg
    inf.threshold = 0.5
    inf._model_mapping = {"G0F00000": Path("/fake/legacy.pt")}
    inf._threshold_mapping = {"G0F00000": 0.55}
    inf._inferencers = {}
    inf.inferencer = None
    return inf


@pytest.fixture
def new_arch_inferencer():
    """新架構 (C-10)：nested model_mapping with inner/edge slots。"""
    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.is_new_architecture = True
    cfg.machine_id = "M1"
    cfg.model_mapping = {
        "G0F00000": {"inner": "/fake/G0F-inner.pt", "edge": "/fake/G0F-edge.pt"},
    }
    cfg.threshold_mapping = {
        "G0F00000": {"inner": 0.40, "edge": 0.65},
    }
    inf = CAPIInferencer.__new__(CAPIInferencer)
    inf.config = cfg
    inf.threshold = 0.5
    inf.base_dir = Path(".")
    inf._model_mapping = {}
    inf._threshold_mapping = {}
    inf._model_cache_v2 = {}
    inf._inferencers = {}
    inf.inferencer = None
    return inf


def test_zone_inferencer_legacy_ignores_zone(legacy_inferencer):
    """舊架構：zone 參數應該被忽略，走 prefix-only lookup。"""
    legacy_inferencer._inferencers[str(Path("/fake/legacy.pt"))] = "LEGACY_MODEL"
    assert legacy_inferencer._get_inferencer_for_zone("G0F00000", "edge") == "LEGACY_MODEL"
    assert legacy_inferencer._get_inferencer_for_zone("G0F00000", "inner") == "LEGACY_MODEL"


def test_zone_inferencer_new_arch_routes_to_edge(new_arch_inferencer):
    """新架構：zone='edge' 應路由到 edge.pt。"""
    with patch.object(new_arch_inferencer, "_load_model_from_path", return_value="EDGE_MODEL") as mock_load:
        result = new_arch_inferencer._get_inferencer_for_zone("G0F00000", "edge")
        assert result == "EDGE_MODEL"
        loaded_path = mock_load.call_args[0][0]
        assert loaded_path.name == "G0F-edge.pt"


def test_zone_inferencer_new_arch_routes_to_inner(new_arch_inferencer):
    """新架構：zone='inner' 應路由到 inner.pt。"""
    with patch.object(new_arch_inferencer, "_load_model_from_path", return_value="INNER_MODEL") as mock_load:
        result = new_arch_inferencer._get_inferencer_for_zone("G0F00000", "inner")
        assert result == "INNER_MODEL"
        loaded_path = mock_load.call_args[0][0]
        assert loaded_path.name == "G0F-inner.pt"


def test_zone_threshold_legacy_returns_flat(legacy_inferencer):
    assert legacy_inferencer._get_threshold_for_zone("G0F00000", "edge") == 0.55
    assert legacy_inferencer._get_threshold_for_zone("G0F00000", "inner") == 0.55


def test_zone_threshold_new_arch_picks_zone_value(new_arch_inferencer):
    assert new_arch_inferencer._get_threshold_for_zone("G0F00000", "edge") == 0.65
    assert new_arch_inferencer._get_threshold_for_zone("G0F00000", "inner") == 0.40


def test_zone_threshold_new_arch_unknown_prefix_falls_back(new_arch_inferencer):
    """新架構：prefix 不在 mapping → fallback 到 self.threshold。"""
    assert new_arch_inferencer._get_threshold_for_zone("UNKNOWN", "edge") == 0.5
