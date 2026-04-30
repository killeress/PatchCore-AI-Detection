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


def test_parse_aoi_report_uses_nested_new_arch_model_mapping_prefixes(new_arch_inferencer, tmp_path):
    """新架構 nested model_mapping 的 prefix 也必須參與 AOI report 解析。

    Regression: WGF50500 不在舊 hard-coded WGF00000 清單內，且新架構
    _model_mapping 為空，曾導致 PCDK2...WGF50500 連續記錄整串解析失敗。
    """
    panel_dir = tmp_path / "yuantu" / "GN160JCEL250S" / "panel"
    report_dir = tmp_path / "Report" / "GN160JCEL250S" / "panel"
    panel_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    new_arch_inferencer.config.aoi_report_path_replace_from = "yuantu"
    new_arch_inferencer.config.aoi_report_path_replace_to = "Report"
    new_arch_inferencer.config.model_mapping = {
        "G0F00000": {"inner": "/fake/g-inner.pt", "edge": "/fake/g-edge.pt"},
        "WGF50500": {"inner": "/fake/wgf-inner.pt", "edge": "/fake/wgf-edge.pt"},
    }
    new_arch_inferencer._model_mapping = {}

    (report_dir / "151525.TXT").write_text(
        "header\n"
        "@;OK;NG"
        "PCDK20035200136WGF50500"
        "PCDK20156500325WGF50500"
        "PCDK20028900553WGF50500;\n",
        encoding="utf-8",
    )

    parsed = new_arch_inferencer._parse_aoi_report_txt(panel_dir)

    assert set(parsed) == {"WGF50500"}
    defects = parsed["WGF50500"]
    assert [d.defect_code for d in defects] == ["PCDK2", "PCDK2", "PCDK2"]
    assert [(d.product_x, d.product_y) for d in defects] == [
        (352, 136),
        (1565, 325),
        (289, 553),
    ]


def test_helper_new_arch_creates_centered_tile(new_arch_inferencer, tmp_path):
    """新架構：helper 以 AOI 座標為中心建 512x512 centered tile（黑邊 zero-pad），
    既存 grid tile 不被動到。AOI 座標永遠在 tile 中心 (256, 256)。"""
    import cv2
    from capi_inference import TileInfo

    img_path = tmp_path / "G0F00000_001.png"
    cv2.imwrite(str(img_path), np.full((1080, 1920), 128, dtype=np.uint8))

    # 既存 grid tile (768, 256, 1280, 768) — 同樣覆蓋 AOI 座標 (960, 540)
    grid_tile = TileInfo(
        tile_id=0, x=768, y=256, width=512, height=512,
        image=np.zeros((512, 512), dtype=np.uint8),
        zone="inner",
    )
    fake_result = ImageResult(
        image_path=img_path,
        image_size=(1920, 1080),
        otsu_bounds=(0, 0, 1920, 1080),
        exclusion_regions=[],
        tiles=[grid_tile],
        excluded_tile_count=0,
        processed_tile_count=1,
        processing_time=0.0,
        anomaly_tiles=[],
        raw_bounds=(0, 0, 1920, 1080),
        panel_polygon=None,
    )

    fake_defect = MagicMock(product_x=960, product_y=540, defect_code="L01")

    with patch.object(new_arch_inferencer, "_parse_aoi_report_txt",
                      return_value={"G0F00000": [fake_defect]}), \
         patch.object(new_arch_inferencer, "_create_aoi_coord_tiles") as mock_v1_create, \
         patch.object(new_arch_inferencer, "_inspect_roi_patchcore") as mock_pc, \
         patch.object(new_arch_inferencer, "_inspect_aoi_edge_defect") as mock_inspect:

        stats = new_arch_inferencer._apply_aoi_coord_inspection(
            panel_dir=tmp_path,
            preprocessed_results=[fake_result],
            omit_image=None, omit_overexposed=False,
            product_resolution=(1920, 1080),
        )

    # v2 不再走 v1 helper / fusion / patchcore inspector
    mock_v1_create.assert_not_called()
    mock_pc.assert_not_called()
    mock_inspect.assert_not_called()

    assert stats["aoi_tile_count"] == 1
    assert stats["aoi_edge_count"] == 0

    # 既存 grid tile 不被動：tile_id=0, is_aoi_coord_tile 仍為 False
    assert grid_tile.is_aoi_coord_tile is False
    # 新加的 centered tile：以 (960, 540) 為中心，左上角 (704, 284)
    assert len(fake_result.tiles) == 2
    centered = fake_result.tiles[-1]
    assert centered.is_aoi_coord_tile is True
    assert centered.aoi_defect_code == "L01"
    assert centered.aoi_product_x == 960
    assert centered.aoi_product_y == 540
    assert centered.x == 960 - 256
    assert centered.y == 540 - 256
    assert centered.image.shape == (512, 512)


def test_helper_new_arch_creates_tile_even_at_image_corner(new_arch_inferencer, tmp_path):
    """AOI 座標靠近圖片邊緣時，centered tile 用黑邊 zero-pad 填補 OOB，
    AOI 座標仍在 tile (256, 256)（不往內推）。"""
    import cv2
    from capi_inference import TileInfo

    img_path = tmp_path / "G0F00000_001.png"
    cv2.imwrite(str(img_path), np.full((1080, 1920), 128, dtype=np.uint8))

    grid_tile = TileInfo(
        tile_id=0, x=0, y=0, width=512, height=512,
        image=np.zeros((512, 512), dtype=np.uint8),
        zone="inner",
    )
    fake_result = ImageResult(
        image_path=img_path,
        image_size=(1920, 1080),
        otsu_bounds=(0, 0, 1920, 1080),
        exclusion_regions=[],
        tiles=[grid_tile],
        excluded_tile_count=0,
        processed_tile_count=1,
        processing_time=0.0,
        anomaly_tiles=[],
        raw_bounds=(0, 0, 1920, 1080),
        panel_polygon=None,
    )
    # AOI 座標貼右下角 (1900, 1000) → centered tile 從 (1644, 744) 起，
    # 右側 236px / 下側 176px 超出圖片，必須 zero-pad
    fake_defect = MagicMock(product_x=1900, product_y=1000, defect_code="L01")

    with patch.object(new_arch_inferencer, "_parse_aoi_report_txt",
                      return_value={"G0F00000": [fake_defect]}):
        stats = new_arch_inferencer._apply_aoi_coord_inspection(
            panel_dir=tmp_path,
            preprocessed_results=[fake_result],
            omit_image=None, omit_overexposed=False,
            product_resolution=(1920, 1080),
        )

    assert stats["aoi_tile_count"] == 1
    assert stats["aoi_edge_count"] == 0
    assert len(fake_result.tiles) == 2

    centered = fake_result.tiles[-1]
    assert centered.is_aoi_coord_tile is True
    assert centered.x == 1900 - 256
    assert centered.y == 1000 - 256
    # 圖片內中央灰階 128，OOB 區域應為 0（黑邊）
    # tile 內座標 (256, 256) 對應圖片 (1900, 1000)，仍在圖片內
    assert centered.image[256, 256] == 128
    # tile 內座標 (500, 500) 對應圖片 (1888, 1244)，超出圖片下緣 → 應為 0
    assert centered.image[500, 500] == 0


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
    assert stats["aoi_tile_count"] == 0
    assert stats["aoi_edge_count"] == 0


def test_v1_process_panel_reuses_parsed_aoi_report(tmp_path):
    """v1 Phase 1.5 解析 AOI report 後，helper 應直接重用同一份資料。"""
    import cv2

    cfg = CAPIConfig()
    cfg.is_new_architecture = False
    cfg.aoi_coord_inspection_enabled = True
    cfg.enable_panel_polygon = False
    cfg.scratch_classifier_enabled = False

    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.config = cfg
    inferencer.threshold = 0.5
    inferencer.base_dir = tmp_path
    inferencer.mark_template = None
    inferencer.inferencer = MagicMock()
    inferencer.edge_inspector = None
    inferencer._model_mapping = {}
    inferencer._threshold_mapping = {}
    inferencer._inferencers = {}

    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    img_path = panel_dir / "G0F00000_001.png"
    cv2.imwrite(str(img_path), np.ones((64, 64, 3), dtype=np.uint8) * 128)

    fake_result = ImageResult(
        image_path=img_path,
        image_size=(64, 64),
        otsu_bounds=(0, 0, 64, 64),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        anomaly_tiles=[],
        raw_bounds=(0, 0, 64, 64),
        panel_polygon=None,
    )
    fake_defect = MagicMock()
    fake_defect.defect_code = "L01"
    fake_defect.product_x = 32
    fake_defect.product_y = 32
    fake_defect.image_prefix = "G0F00000"
    parsed_report = {"G0F00000": [fake_defect]}

    def _run_inference_passthrough(result, **_kwargs):
        return result

    with patch.object(inferencer, "_parse_defect_txt", return_value={}), \
         patch.object(inferencer, "_find_raw_object_bounds",
                      return_value=((0, 0, 64, 64), np.ones((64, 64), dtype=np.uint8))), \
         patch.object(inferencer, "preprocess_image", return_value=fake_result), \
         patch.object(inferencer, "_parse_aoi_report_txt",
                      return_value=parsed_report) as mock_parse, \
         patch.object(inferencer, "_create_aoi_coord_tiles",
                      return_value=([], [])) as mock_create_tiles, \
         patch.object(inferencer, "run_inference",
                      side_effect=_run_inference_passthrough):
        *_, returned_report = inferencer._process_panel_v1(
            panel_dir,
            cpu_workers=1,
            product_resolution=(64, 64),
        )

    assert returned_report is parsed_report
    assert mock_parse.call_count == 1
    mock_create_tiles.assert_called_once()


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


def test_v2_helper_does_not_invoke_edge_pt_for_aoi_coords(new_arch_inferencer, tmp_path):
    """新架構 helper 不應再對 AOI 座標 ROI 跑 edge.pt（避免重複 PC 推論 +
    避免寫入 edge_defects 讓記錄頁出現「CV 邊緣檢測」區塊）。
    edge.pt 已由 grid tiling 在 _process_panel_v2 內為 edge zone 推論過。"""
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

    with patch.object(new_arch_inferencer, "_parse_aoi_report_txt",
                      return_value={"G0F00000": [fake_edge_def]}), \
         patch.object(new_arch_inferencer, "_get_model_for") as mock_for, \
         patch.object(new_arch_inferencer, "predict_tile") as mock_pred:

        new_arch_inferencer._apply_aoi_coord_inspection(
            panel_dir=panel_dir,
            preprocessed_results=[fake_result],
            omit_image=None, omit_overexposed=False,
            product_resolution=(1920, 1080),
        )

    mock_for.assert_not_called()
    mock_pred.assert_not_called()
    assert fake_result.edge_defects == [], "新架構 helper 不應寫入 edge_defects"
