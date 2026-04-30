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
