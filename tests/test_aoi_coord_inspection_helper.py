"""_apply_aoi_coord_inspection helper 在 v1 / v2 都能直接呼叫的測試。"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer, ImageResult


@pytest.fixture
def new_arch_inferencer(tmp_path):
    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.is_new_architecture = True
    cfg.machine_id = "M1"
    cfg.aoi_coord_inspection_enabled = True
    cfg.aoi_report_path_replace_from = ""
    cfg.aoi_report_path_replace_to = ""
    cfg.model_mapping = {
        "G0F00000": {"inner": "/fake/inner.pt", "edge": "/fake/edge.pt"},
    }
    cfg.threshold_mapping = {"G0F00000": {"inner": 0.4, "edge": 0.65}}
    inf = CAPIInferencer.__new__(CAPIInferencer)
    inf.config = cfg
    inf.threshold = 0.5
    inf.base_dir = tmp_path
    inf.edge_inspector = MagicMock()
    inf.edge_inspector.config.aoi_edge_inspector = "fusion"  # 應被 new_arch override
    inf._model_mapping = {}
    inf._threshold_mapping = {}
    inf._model_cache_v2 = {}
    inf._inferencers = {}
    inf.inferencer = None
    return inf


def test_helper_returns_zero_when_disabled(new_arch_inferencer, tmp_path):
    new_arch_inferencer.config.aoi_coord_inspection_enabled = False
    stats = new_arch_inferencer._apply_aoi_coord_inspection(
        panel_dir=tmp_path,
        preprocessed_results=[],
        omit_image=None, omit_overexposed=False,
        product_resolution=(1920, 1080),
    )
    assert stats == {"aoi_tile_count": 0, "aoi_edge_count": 0}


def test_helper_returns_zero_when_no_aoi_report(new_arch_inferencer, tmp_path):
    """panel_dir 內沒有 aoi report → helper 回傳 0/0，不丟例外。"""
    stats = new_arch_inferencer._apply_aoi_coord_inspection(
        panel_dir=tmp_path,
        preprocessed_results=[],
        omit_image=None, omit_overexposed=False,
        product_resolution=(1920, 1080),
    )
    assert stats == {"aoi_tile_count": 0, "aoi_edge_count": 0}


def test_helper_calls_patchcore_with_edge_zone(new_arch_inferencer, tmp_path):
    """新架構下 helper 對 AOI 邊緣 defect 呼叫 _inspect_roi_patchcore(zone='edge')。"""
    img_path = tmp_path / "G0F00000_001.png"
    np.ones((1080, 1920, 3), dtype=np.uint8).tofile(img_path)  # placeholder file

    fake_result = ImageResult(
        image_path=img_path,
        image_size=(1920, 1080),
        otsu_bounds=(0, 0, 1920, 1080),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        anomaly_tiles=[],
        raw_bounds=(0, 0, 1920, 1080),
        panel_polygon=np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], dtype=np.float32),
    )

    fake_edge_def = MagicMock(product_x=1900, product_y=540, defect_code="L01")

    with patch.object(new_arch_inferencer, "_parse_aoi_report_txt",
                      return_value={"G0F00000": [fake_edge_def]}), \
         patch.object(new_arch_inferencer, "_create_aoi_coord_tiles",
                      return_value=([], [fake_edge_def])), \
         patch("cv2.imread", return_value=np.ones((1080, 1920, 3), dtype=np.uint8)), \
         patch.object(new_arch_inferencer, "_inspect_roi_patchcore",
                      return_value=([], {"score": 0.1, "threshold": 0.65, "area": 0,
                                          "ok_reason": "", "roi": None, "fg_mask": None,
                                          "anomaly_map": None})) as mock_pc:

        stats = new_arch_inferencer._apply_aoi_coord_inspection(
            panel_dir=tmp_path,
            preprocessed_results=[fake_result],
            omit_image=None, omit_overexposed=False,
            product_resolution=(1920, 1080),
        )

    assert stats["aoi_edge_count"] == 1
    pc_kwargs = mock_pc.call_args.kwargs
    assert pc_kwargs.get("zone") == "edge", \
        f"new arch 下 helper 應呼叫 _inspect_roi_patchcore(zone='edge')；實際 kwargs={pc_kwargs}"


def test_helper_skips_lighting_without_aoi_report(new_arch_inferencer, tmp_path):
    img_path = tmp_path / "R0F00000_001.png"
    img_path.touch()

    other_result = ImageResult(
        image_path=img_path,
        image_size=(1920, 1080),
        otsu_bounds=(0, 0, 1920, 1080),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        anomaly_tiles=[],
        raw_bounds=(0, 0, 1920, 1080),
        panel_polygon=None,
    )

    with patch.object(new_arch_inferencer, "_parse_aoi_report_txt",
                      return_value={"G0F00000": [MagicMock()]}), \
         patch.object(new_arch_inferencer, "_create_aoi_coord_tiles") as mock_create:
        stats = new_arch_inferencer._apply_aoi_coord_inspection(
            panel_dir=tmp_path,
            preprocessed_results=[other_result],
            omit_image=None, omit_overexposed=False,
            product_resolution=(1920, 1080),
        )

    mock_create.assert_not_called()
    assert stats == {"aoi_tile_count": 0, "aoi_edge_count": 0}


def test_v2_process_panel_invokes_aoi_coord_helper(new_arch_inferencer, tmp_path):
    """新架構 _process_panel_v2 應呼叫 _apply_aoi_coord_inspection."""
    import cv2
    from capi_preprocess import PanelPreprocessResult

    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    # 建立最小可運行的 panel：一張 G0F00000 圖
    img_file = panel_dir / "G0F00000_001.png"
    cv2.imwrite(str(img_file), np.ones((1080, 1920, 3), dtype=np.uint8) * 128)

    new_arch_inferencer.config.tile_size = 512
    new_arch_inferencer.config.otsu_offset = 5
    new_arch_inferencer.config.enable_panel_polygon = False
    new_arch_inferencer.config.edge_threshold_px = 768

    # 回傳有一個 lighting、但 0 tiles 的結果，讓 for 迴圈結束後繼續執行到 helper
    fake_pre_result = PanelPreprocessResult(
        image_path=img_file,
        lighting="G0F00000",
        foreground_bbox=(0, 0, 1920, 1080),
        panel_polygon=None,
        tiles=[],
    )
    fake_panel_results = {"G0F00000": fake_pre_result}

        # _process_panel_v2 內部用 local import (`from capi_preprocess import ...`)，
        # 故 patch 目標是 capi_preprocess 模組而非 capi_inference。若 import 提升到
        # module-level，patch 路徑需改為 "capi_inference.preprocess_panel_folder"。
    with patch("capi_preprocess.preprocess_panel_folder", return_value=fake_panel_results), \
         patch.object(new_arch_inferencer, "_load_omit_context",
                      return_value=(None, False, "", None)), \
         patch.object(new_arch_inferencer, "_apply_aoi_coord_inspection",
                      return_value={"aoi_tile_count": 0, "aoi_edge_count": 0}) as mock_helper, \
         patch.object(new_arch_inferencer, "_apply_cv_edge_inspection"), \
         patch.object(new_arch_inferencer, "_apply_omit_dust_postprocess"), \
         patch.object(new_arch_inferencer, "_apply_bomb_postprocess"), \
         patch.object(new_arch_inferencer, "_apply_exclude_zone_postprocess"), \
         patch.object(new_arch_inferencer, "_apply_scratch_postprocess"), \
         patch("cv2.imread", return_value=np.ones((1080, 1920, 3), dtype=np.uint8)):

        new_arch_inferencer._process_panel_v2(
            panel_dir=panel_dir,
            product_resolution=(1920, 1080),
        )

    assert mock_helper.call_count == 1
    kwargs = mock_helper.call_args.kwargs
    assert kwargs["panel_dir"] == panel_dir


def test_v2_e2e_aoi_edge_routes_through_edge_pt(new_arch_inferencer, tmp_path):
    """E2E：新架構下，AOI 座標邊緣 defect → _apply_aoi_coord_inspection
    → _inspect_aoi_edge_defect → _inspect_roi_patchcore(zone='edge')
    → _get_inferencer_for_zone → _get_model_for(_, _, 'edge')。"""
    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()

    fake_result = ImageResult(
        image_path=panel_dir / "G0F00000_001.png",
        image_size=(1920, 1080),
        otsu_bounds=(0, 0, 1920, 1080),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        anomaly_tiles=[],
        raw_bounds=(0, 0, 1920, 1080),
        panel_polygon=np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], dtype=np.float32),
    )

    fake_edge_def = MagicMock(product_x=1900, product_y=540, defect_code="L01")
    edge_model = MagicMock()

    with patch.object(new_arch_inferencer, "_parse_aoi_report_txt",
                      return_value={"G0F00000": [fake_edge_def]}), \
         patch.object(new_arch_inferencer, "_create_aoi_coord_tiles",
                      return_value=([], [fake_edge_def])), \
         patch("cv2.imread", return_value=np.ones((1080, 1920, 3), dtype=np.uint8)), \
         patch.object(new_arch_inferencer, "_get_model_for", return_value=edge_model) as mock_for, \
         patch.object(new_arch_inferencer, "predict_tile",
                      return_value=(0.1, np.zeros((512, 512), dtype=np.float32))):

        new_arch_inferencer._apply_aoi_coord_inspection(
            panel_dir=panel_dir,
            preprocessed_results=[fake_result],
            omit_image=None, omit_overexposed=False,
            product_resolution=(1920, 1080),
        )

    # 串通驗證：_get_model_for 被叫，且 zone='edge'
    assert mock_for.called, "_get_model_for 應被呼叫（新架構走 edge.pt 路徑）"
    args = mock_for.call_args.args
    assert args[0] == "M1"
    assert args[1] == "G0F00000"
    assert args[2] == "edge", f"zone 應為 'edge'，實際 {args[2]!r}"
