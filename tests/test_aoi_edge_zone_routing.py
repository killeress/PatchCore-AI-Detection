"""新架構 AOI 座標邊緣 zone-aware routing 單元測試。"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
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


def test_preload_v2_models_loads_all_units_once(new_arch_inferencer):
    """新架構 startup prewarm 應載入所有 inner/edge，並重用 v2 cache。"""
    with patch.object(
        new_arch_inferencer,
        "_load_model_from_path",
        side_effect=lambda path: f"MODEL:{path.name}",
    ) as mock_load:
        loaded, total = new_arch_inferencer.preload_v2_models()
        loaded_again, total_again = new_arch_inferencer.preload_v2_models()

    assert (loaded, total) == (2, 2)
    assert (loaded_again, total_again) == (2, 2)
    assert mock_load.call_count == 2
    assert new_arch_inferencer._model_cache_v2[("M1", "G0F00000", "inner")] == "MODEL:G0F-inner.pt"
    assert new_arch_inferencer._model_cache_v2[("M1", "G0F00000", "edge")] == "MODEL:G0F-edge.pt"


def test_zone_threshold_legacy_returns_flat(legacy_inferencer):
    assert legacy_inferencer._get_threshold_for_zone("G0F00000", "edge") == 0.55
    assert legacy_inferencer._get_threshold_for_zone("G0F00000", "inner") == 0.55


def test_zone_threshold_new_arch_picks_zone_value(new_arch_inferencer):
    assert new_arch_inferencer._get_threshold_for_zone("G0F00000", "edge") == 0.65
    assert new_arch_inferencer._get_threshold_for_zone("G0F00000", "inner") == 0.40


def test_zone_threshold_new_arch_unknown_prefix_falls_back(new_arch_inferencer):
    """新架構：prefix 不在 mapping → fallback 到 self.threshold。"""
    assert new_arch_inferencer._get_threshold_for_zone("UNKNOWN", "edge") == 0.5


def test_inspector_mode_legacy_reads_config(legacy_inferencer):
    """舊架構：照舊讀 edge_inspector.config.aoi_edge_inspector。"""
    legacy_inferencer.edge_inspector = MagicMock()
    legacy_inferencer.edge_inspector.config.aoi_edge_inspector = "fusion"
    assert legacy_inferencer._resolve_aoi_edge_inspector_mode() == "fusion"


def test_inspector_mode_new_arch_forces_patchcore(new_arch_inferencer):
    """新架構：無視 config，強制回 'patchcore'（edge.pt 已專為 edge 訓練，
    fusion / cv 不再有理論基礎）。"""
    new_arch_inferencer.edge_inspector = MagicMock()
    new_arch_inferencer.edge_inspector.config.aoi_edge_inspector = "fusion"
    assert new_arch_inferencer._resolve_aoi_edge_inspector_mode() == "patchcore"

    new_arch_inferencer.edge_inspector.config.aoi_edge_inspector = "cv"
    assert new_arch_inferencer._resolve_aoi_edge_inspector_mode() == "patchcore"


def test_inspector_mode_no_edge_inspector_default_cv(legacy_inferencer):
    """舊架構 + edge_inspector 不存在 → 'cv' (與既有 fallback 一致)。"""
    legacy_inferencer.edge_inspector = None
    assert legacy_inferencer._resolve_aoi_edge_inspector_mode() == "cv"


def test_inspect_roi_patchcore_new_arch_uses_edge_model(new_arch_inferencer):
    """_inspect_roi_patchcore(zone='edge') 在新架構下應呼叫 edge.pt 對應的 inferencer。"""
    edge_model = MagicMock()
    edge_model.predict.return_value = MagicMock(pred_score=0.1, anomaly_map=np.zeros((512, 512), dtype=np.float32))

    new_arch_inferencer.edge_inspector = MagicMock()
    new_arch_inferencer.config.patchcore_min_area = 10
    new_arch_inferencer.config.patchcore_filter_enabled = False

    with patch.object(new_arch_inferencer, "_get_model_for", return_value=edge_model) as mock_for:
        with patch.object(new_arch_inferencer, "predict_tile", return_value=(0.1, np.zeros((512, 512), dtype=np.float32))):
            image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            defects, stats = new_arch_inferencer._inspect_roi_patchcore(
                image, img_x=200, img_y=200, img_prefix="G0F00000",
                panel_polygon=None, zone="edge",
            )

    mock_for.assert_called_with("M1", "G0F00000", "edge")
    assert stats["threshold"] == 0.65, "新架構應使用 edge_threshold=0.65"


def test_inspect_roi_patchcore_legacy_default_zone_unchanged(legacy_inferencer):
    """舊架構 + 預設 zone='edge'：行為與既有 prefix-only lookup 完全一致。"""
    legacy_model = MagicMock()
    legacy_inferencer.edge_inspector = MagicMock()
    legacy_inferencer.config.patchcore_min_area = 10
    legacy_inferencer.config.patchcore_filter_enabled = False

    with patch.object(legacy_inferencer, "_get_inferencer_for_prefix", return_value=legacy_model) as mock_p:
        with patch.object(legacy_inferencer, "predict_tile", return_value=(0.1, np.zeros((512, 512), dtype=np.float32))):
            image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            legacy_inferencer._inspect_roi_patchcore(
                image, img_x=200, img_y=200, img_prefix="G0F00000",
                panel_polygon=None,  # zone 預設 "edge"
            )

    mock_p.assert_called_with("G0F00000")


def test_inspect_roi_fusion_new_arch_pc_half_uses_edge_model(new_arch_inferencer):
    """新架構下 fusion (理論上不會走到，但行為要正確) PC half 應走 edge.pt。"""
    edge_model = MagicMock()
    new_arch_inferencer.edge_inspector = MagicMock()
    new_arch_inferencer.edge_inspector.config.aoi_edge_boundary_band_px = 40
    new_arch_inferencer.edge_inspector.config.aoi_edge_pc_roi_inward_shift_enabled = False
    new_arch_inferencer.edge_inspector.inspect_roi.return_value = ([], {})

    with patch.object(new_arch_inferencer, "_inspect_roi_patchcore") as mock_pc:
        mock_pc.return_value = ([], {"score": 0.1, "threshold": 0.65,
                                      "anomaly_map": np.zeros((512, 512), np.float32),
                                      "fg_mask": np.zeros((512, 512), np.uint8),
                                      "roi": np.zeros((512, 512, 3), np.uint8),
                                      "area": 0, "min_area": 10})
        new_arch_inferencer._inspect_roi_fusion(
            image=np.zeros((1080, 1920, 3), np.uint8),
            img_x=200, img_y=200, img_prefix="G0F00000",
            panel_polygon=np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], dtype=np.float32),
            zone="edge",
        )

    # 確認 fusion 內部呼叫 _inspect_roi_patchcore 時帶上 zone="edge"
    pc_kwargs = mock_pc.call_args.kwargs
    assert pc_kwargs.get("zone") == "edge", \
        f"fusion 內部 _inspect_roi_patchcore 未帶 zone='edge'，實際 kwargs={pc_kwargs}"
