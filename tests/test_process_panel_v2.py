"""
tests/test_process_panel_v2.py
Smoke tests for _process_panel_v2 — uses MagicMock so no real GPU / model files needed.
"""
import tempfile
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from capi_config import BombDefect, CAPIConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_dir: Path) -> CAPIConfig:
    """Build a minimal new-architecture CAPIConfig with nested model_mapping."""
    inner_pt = tmp_dir / "g_inner.pt"
    edge_pt = tmp_dir / "g_edge.pt"
    # Files don't have to exist — _get_model_for is mocked in tests
    return CAPIConfig(
        machine_id="TEST_MACHINE",
        is_new_architecture=True,
        edge_threshold_px=768,
        tile_size=512,
        otsu_offset=5,
        enable_panel_polygon=True,
        model_mapping={
            "G0F00000": {
                "inner": str(inner_pt),
                "edge": str(edge_pt),
            }
        },
        threshold_mapping={
            "G0F00000": {"inner": 0.5, "edge": 0.5},
        },
        scratch_classifier_enabled=False,
    )


def _write_grey_panel_image_at(path: Path) -> Path:
    """Write a small grayscale PNG that Otsu can binarize sensibly."""
    import cv2
    h, w = 1024, 1024
    img = np.zeros((h, w), dtype=np.uint8)
    # bright panel region in the centre
    img[100:900, 100:900] = 200
    cv2.imwrite(str(path), img)
    return path


def _write_grey_panel_image(folder: Path, prefix: str = "G0F00000") -> Path:
    return _write_grey_panel_image_at(folder / f"{prefix}_test.png")


def _make_fake_predict_result(score: float = 0.3) -> Any:
    result = MagicMock()
    result.pred_score = score
    amap = np.zeros((512, 512), dtype=np.float32)
    # Realistic PatchCore hot-spot: 3x3 peak region + noise floor
    # Single pixel would trigger concentration check (peak/mean=1.0 → 50% penalty)
    # This creates: peak=score in center, surrounding pixels at 70%, noise floor at 5%
    # Result: peak/mean ≈ 20 → passes concentration check (>2.0 threshold)
    amap[255:258, 255:258] = score * 0.7
    amap[256, 256] = score  # center peak
    # Add slight noise floor so concentration ratio isn't degenerate
    amap[200:300, 200:300] = np.maximum(amap[200:300, 200:300], score * 0.05)
    result.anomaly_map = amap
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_process_panel_v2_returns_compatible_tuple(tmp_path):
    """v2 must return a 7-tuple compatible with the v1 return signature."""
    _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)

    from capi_inference import CAPIInferencer

    fake_model = MagicMock()
    fake_model.predict.return_value = _make_fake_predict_result(0.3)

    with patch.object(CAPIInferencer, "_get_model_for", return_value=fake_model):
        inferencer = CAPIInferencer(cfg)
        ret = inferencer.process_panel(tmp_path)

    assert isinstance(ret, tuple), "process_panel should return a tuple"
    assert len(ret) == 7, "should return 7-tuple (results, omit_vis, omit_oe, omit_info, is_dup, omit_img, aoi_report)"

    results, omit_vis, omit_oe, omit_info, is_dup, omit_img, aoi_report = ret
    assert isinstance(results, list), "results should be List[ImageResult]"


def test_process_panel_v2_duplicate_panel_uses_latest_lighting_file(tmp_path):
    old_img = _write_grey_panel_image_at(tmp_path / "G0F00000_010000.png")
    latest_img = _write_grey_panel_image_at(tmp_path / "G0F00000_020000.png")
    os.utime(old_img, (1000, 1000))
    os.utime(latest_img, (2000, 2000))

    cfg = _make_config(tmp_path)
    cfg.max_images_per_panel = 1

    from capi_inference import CAPIInferencer

    fake_model = MagicMock()
    fake_model.predict.return_value = _make_fake_predict_result(0.1)

    with patch.object(CAPIInferencer, "_get_model_for", return_value=fake_model):
        inferencer = CAPIInferencer(cfg)
        results, _omit_vis, _omit_oe, _omit_info, is_dup, _omit_img, _report = \
            inferencer.process_panel(tmp_path)

    assert is_dup is True
    assert [r.image_path.name for r in results] == ["G0F00000_020000.png"]


def test_process_panel_v2_duplicate_panel_uses_latest_hm_lighting_file(tmp_path):
    old_img = _write_grey_panel_image_at(tmp_path / "G0F00000010000.png")
    latest_img = _write_grey_panel_image_at(tmp_path / "G0F00000020000.png")
    os.utime(old_img, (1000, 1000))
    os.utime(latest_img, (2000, 2000))

    cfg = _make_config(tmp_path)
    cfg.max_images_per_panel = 20

    from capi_inference import CAPIInferencer

    fake_model = MagicMock()
    fake_model.predict.return_value = _make_fake_predict_result(0.1)

    with patch.object(CAPIInferencer, "_get_model_for", return_value=fake_model):
        inferencer = CAPIInferencer(cfg)
        results, _omit_vis, _omit_oe, _omit_info, is_dup, _omit_img, _report = \
            inferencer.process_panel(tmp_path)

    assert is_dup is True
    assert [r.image_path.name for r in results] == ["G0F00000020000.png"]


def test_process_panel_v2_passes_requested_product_resolution_to_preprocess(tmp_path, monkeypatch):
    _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)
    cfg.machine_id = "UNKNOWN_SIZE_CODE"
    captured = {}

    def fake_preprocess_panel_folder(
        _panel_path,
        pre_cfg,
        image_files=None,
        boundary_reference_files=None,
    ):
        captured["product_resolution"] = pre_cfg.product_resolution
        return {}

    monkeypatch.setattr("capi_preprocess.preprocess_panel_folder", fake_preprocess_panel_folder)

    from capi_inference import CAPIInferencer

    inferencer = CAPIInferencer(cfg)
    inferencer.process_panel(tmp_path, product_resolution=(1366, 768))

    assert captured["product_resolution"] == (1366, 768)


def test_process_panel_v2_no_anomaly_when_score_below_threshold(tmp_path):
    """All tiles below threshold → no anomaly_tiles in results."""
    _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)

    from capi_inference import CAPIInferencer

    fake_model = MagicMock()
    # score = 0.1 < threshold 0.5
    fake_model.predict.return_value = _make_fake_predict_result(0.1)

    with patch.object(CAPIInferencer, "_get_model_for", return_value=fake_model):
        inferencer = CAPIInferencer(cfg)
        results, *_ = inferencer.process_panel(tmp_path)

    for r in results:
        assert r.anomaly_tiles == [], f"Expected no anomaly tiles, got {len(r.anomaly_tiles)}"


def test_process_panel_v2_detects_anomaly_when_score_above_threshold(tmp_path):
    """Tiles above threshold appear in anomaly_tiles."""
    _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)

    from capi_inference import CAPIInferencer

    fake_model = MagicMock()
    # score = 0.9 > threshold 0.5
    fake_model.predict.return_value = _make_fake_predict_result(0.9)

    with patch.object(CAPIInferencer, "_get_model_for", return_value=fake_model):
        inferencer = CAPIInferencer(cfg)
        results, *_ = inferencer.process_panel(tmp_path)

    assert results, "Expected at least one ImageResult"
    ng_results = [r for r in results if r.anomaly_tiles]
    assert ng_results, "Expected at least one NG image result when all scores are 0.9"


def test_process_panel_v2_empty_folder(tmp_path):
    """Empty folder → empty results (no crash)."""
    cfg = _make_config(tmp_path)
    from capi_inference import CAPIInferencer

    with patch.object(CAPIInferencer, "_get_model_for", return_value=MagicMock()):
        inferencer = CAPIInferencer(cfg)
        results, omit_vis, omit_oe, omit_info, is_dup, omit_img, aoi_report = inferencer.process_panel(tmp_path)

    assert results == []


def test_process_panel_v2_image_result_fields(tmp_path):
    """ImageResult returned by v2 has expected attributes."""
    _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)

    from capi_inference import CAPIInferencer, ImageResult

    fake_model = MagicMock()
    fake_model.predict.return_value = _make_fake_predict_result(0.3)

    with patch.object(CAPIInferencer, "_get_model_for", return_value=fake_model):
        inferencer = CAPIInferencer(cfg)
        results, *_ = inferencer.process_panel(tmp_path)

    for r in results:
        assert isinstance(r, ImageResult)
        assert r.image_path.exists()
        assert r.image_size[0] > 0 and r.image_size[1] > 0
        assert r.otsu_bounds is not None
        assert r.tiles is not None


def test_process_panel_v2_preserves_original_tile_for_heatmap(tmp_path):
    """v2 should infer on preprocessed tile.image while preserving raw original_image for debug panels."""
    img_path = _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)

    from capi_inference import CAPIInferencer
    from capi_preprocess import PanelPreprocessResult, TileResult

    processed_tile = np.full((512, 512), 23, dtype=np.uint8)
    original_tile = np.full((512, 512), 17, dtype=np.uint8)
    fake_pre_result = PanelPreprocessResult(
        image_path=img_path,
        lighting="G0F00000",
        foreground_bbox=(0, 0, 1024, 1024),
        panel_polygon=None,
        tiles=[
            TileResult(
                tile_id=0,
                x1=0, y1=0, x2=512, y2=512,
                image=processed_tile,
                original_image=original_tile,
                mask=None,
                coverage=1.0,
                zone="inner",
                center_dist_to_edge=999.0,
            )
        ],
    )
    amap = np.zeros((512, 512), dtype=np.float32)

    with patch("capi_preprocess.preprocess_panel_folder",
               return_value={"G0F00000": fake_pre_result}), \
         patch.object(CAPIInferencer, "_get_model_for",
                      return_value=MagicMock()), \
         patch.object(CAPIInferencer, "predict_tile",
                      return_value=(0.1, amap)) as pred:
        inferencer = CAPIInferencer(cfg)
        results, *_ = inferencer.process_panel(tmp_path)

    assert len(results) == 1
    assert len(results[0].tiles) == 1
    tile = results[0].tiles[0]
    np.testing.assert_array_equal(tile.image, processed_tile)
    np.testing.assert_array_equal(tile.original_image, original_tile)
    np.testing.assert_array_equal(pred.call_args.args[0].image, processed_tile)


def test_process_panel_v2_aoi_centered_tiles_use_preprocess_pipeline(tmp_path):
    """AOI-centered v2 tiles must infer on the same preprocessed image pipeline as grid tiles."""
    img_path = _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)
    cfg.aoi_coord_inspection_enabled = True
    cfg.grid_tiling_enabled = False
    cfg.image_preprocess_pipeline = [
        {"method": "mean", "params": {"kernel_size": 3}},
    ]

    from capi_inference import AOIReportDefect, CAPIInferencer
    from capi_preprocess import PanelPreprocessResult

    fake_pre_result = PanelPreprocessResult(
        image_path=img_path,
        lighting="G0F00000",
        foreground_bbox=(0, 0, 1024, 1024),
        panel_polygon=None,
        tiles=[],
    )
    parsed_report = {
        "G0F00000": [
            AOIReportDefect(
                defect_code="L01",
                product_x=512,
                product_y=512,
                image_prefix="G0F00000",
            )
        ]
    }

    def _fake_apply_preprocess(image, pipeline):
        return {
            "image": np.full_like(image, 23),
            "pipeline": pipeline,
            "steps": [],
            "total_elapsed_ms": 0.0,
        }

    amap = np.zeros((512, 512), dtype=np.float32)
    with patch("capi_preprocess.preprocess_panel_folder",
               return_value={"G0F00000": fake_pre_result}), \
         patch("capi_image_preprocess_lab.apply_preprocess_pipeline",
               side_effect=_fake_apply_preprocess) as apply_preprocess, \
         patch.object(CAPIInferencer, "_parse_aoi_report_txt",
                      return_value=parsed_report), \
         patch.object(CAPIInferencer, "_get_model_for",
                      return_value=MagicMock()), \
         patch.object(CAPIInferencer, "predict_tile",
                      return_value=(0.1, amap)) as pred:
        inferencer = CAPIInferencer(cfg)
        results, *_ = inferencer.process_panel(
            tmp_path,
            product_resolution=(1024, 1024),
        )

    apply_preprocess.assert_called_once()
    tile = results[0].tiles[0]
    np.testing.assert_array_equal(pred.call_args.args[0].image, np.full((512, 512), 23, dtype=np.uint8))
    np.testing.assert_array_equal(tile.image, np.full((512, 512), 23, dtype=np.uint8))
    assert tile.original_image is not None
    assert np.max(tile.original_image) == 200


def test_process_panel_v2_missing_lighting_config_fails(tmp_path):
    """A configured lighting image without inner/edge models must fail the request."""
    # Write an image with a prefix NOT in model_mapping
    _write_grey_panel_image(tmp_path, "R0F00000")
    cfg = _make_config(tmp_path)  # only G0F00000 in model_mapping

    from capi_inference import CAPIInferencer

    with patch.object(CAPIInferencer, "_get_model_for", return_value=MagicMock()):
        inferencer = CAPIInferencer(cfg)
        with pytest.raises(RuntimeError, match="model_mapping"):
            inferencer.process_panel(tmp_path)


def test_process_panel_v2_model_failure_fails_request(tmp_path):
    """Model load/predict failures must not be converted into clean OK results."""
    _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)

    from capi_inference import CAPIInferencer

    with patch.object(CAPIInferencer, "_get_model_for", side_effect=FileNotFoundError("missing model")):
        inferencer = CAPIInferencer(cfg)
        with pytest.raises(RuntimeError, match="推論失敗"):
            inferencer.process_panel(tmp_path)


def test_process_panel_v2_uses_shared_predict_tile_postprocess(tmp_path):
    """v2 must call predict_tile so edge margin/mask/PatchCore postprocess stays shared with v1."""
    _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)

    from capi_inference import CAPIInferencer

    amap = np.zeros((512, 512), dtype=np.float32)
    amap[256, 256] = 0.9
    fake_model = MagicMock()

    with patch.object(CAPIInferencer, "_get_model_for", return_value=fake_model), \
         patch.object(CAPIInferencer, "predict_tile", return_value=(0.9, amap)) as pred:
        inferencer = CAPIInferencer(cfg)
        inferencer.edge_inspector.config.enabled = False
        results, *_ = inferencer.process_panel(tmp_path)

    assert results
    assert pred.called


def test_process_panel_v2_grid_disabled_infers_only_aoi_tiles(tmp_path):
    """grid_tiling_enabled=False must not run v2 full-grid inference before AOI attribution."""
    _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)
    cfg.aoi_coord_inspection_enabled = True
    cfg.grid_tiling_enabled = False

    from capi_inference import CAPIInferencer, AOIReportDefect
    from capi_preprocess import PanelPreprocessResult, TileResult

    img_path = tmp_path / "G0F00000_test.png"
    fake_tiles = [
        TileResult(
            tile_id=0, x1=0, y1=0, x2=512, y2=512,
            image=np.zeros((512, 512), dtype=np.uint8),
            mask=None, coverage=1.0, zone="inner", center_dist_to_edge=999.0,
        ),
        TileResult(
            tile_id=1, x1=512, y1=0, x2=1024, y2=512,
            image=np.zeros((512, 512), dtype=np.uint8),
            mask=None, coverage=1.0, zone="edge", center_dist_to_edge=0.0,
        ),
    ]
    fake_pre_result = PanelPreprocessResult(
        image_path=img_path,
        lighting="G0F00000",
        foreground_bbox=(0, 0, 1024, 1024),
        panel_polygon=None,
        tiles=fake_tiles,
    )
    parsed_report = {
        "G0F00000": [
            AOIReportDefect(
                defect_code="L01",
                product_x=100,
                product_y=100,
                image_prefix="G0F00000",
            )
        ]
    }

    amap = np.zeros((512, 512), dtype=np.float32)
    with patch("capi_preprocess.preprocess_panel_folder",
               return_value={"G0F00000": fake_pre_result}), \
         patch.object(CAPIInferencer, "_parse_aoi_report_txt",
                      return_value=parsed_report), \
         patch.object(CAPIInferencer, "_get_model_for",
                      return_value=MagicMock()) as get_model, \
         patch.object(CAPIInferencer, "predict_tile",
                      return_value=(0.1, amap)) as pred:
        inferencer = CAPIInferencer(cfg)
        results, *_, returned_report = inferencer.process_panel(
            tmp_path,
            product_resolution=(1024, 1024),
        )

    assert returned_report is parsed_report
    assert pred.call_count == 1
    assert get_model.call_count == 1
    assert len(results) == 1
    assert len(results[0].tiles) == 1
    assert results[0].processed_tile_count == 1
    kept_tile = results[0].tiles[0]
    # v2 以 AOI 座標 (100,100) 為 anchor 建 tile；沒有偵測到 polygon 時
    # 使用 raw_bounds 的矩形 fallback，靠邊 tile 仍會往 panel 內側推。
    assert kept_tile.is_aoi_coord_tile is True
    assert abs(kept_tile.x - results[0].raw_bounds[0]) <= 1
    assert abs(kept_tile.y - results[0].raw_bounds[1]) <= 1
    assert kept_tile.aoi_product_x == 100
    assert kept_tile.aoi_product_y == 100
    assert kept_tile.image.shape == (512, 512)
    assert kept_tile.is_aoi_coord_below_threshold is True
    assert len(results[0].anomaly_tiles) == 1


def test_process_panel_v2_aoi_only_uses_full_folder_for_boundary_reference(tmp_path):
    """AOI-only keeps report targets but may use W0F as a boundary-only reference."""
    for prefix in ["STANDARD", "G0F00000", "R0F00000", "W0F00000", "WGF50500"]:
        _write_grey_panel_image(tmp_path, prefix)
    cfg = _make_config(tmp_path)
    cfg.aoi_coord_inspection_enabled = True
    cfg.grid_tiling_enabled = False

    from capi_inference import CAPIInferencer, AOIReportDefect
    from capi_preprocess import PanelPreprocessResult

    img_path = tmp_path / "G0F00000_test.png"
    fake_pre_result = PanelPreprocessResult(
        image_path=img_path,
        lighting="G0F00000",
        foreground_bbox=(0, 0, 1024, 1024),
        panel_polygon=None,
        tiles=[],
    )
    parsed_report = {
        "G0F00000": [
            AOIReportDefect(
                defect_code="L01",
                product_x=512,
                product_y=512,
                image_prefix="G0F00000",
            )
        ]
    }

    amap = np.zeros((512, 512), dtype=np.float32)
    with patch("capi_preprocess.preprocess_panel_folder",
               return_value={"G0F00000": fake_pre_result}) as pre_folder, \
         patch.object(CAPIInferencer, "_parse_aoi_report_txt",
                      return_value=parsed_report), \
         patch.object(CAPIInferencer, "_get_model_for",
                      return_value=MagicMock()), \
         patch.object(CAPIInferencer, "predict_tile",
                      return_value=(0.1, amap)):
        inferencer = CAPIInferencer(cfg)
        results, *_ = inferencer.process_panel(
            tmp_path,
            product_resolution=(1024, 1024),
        )

    assert len(results) == 1
    pre_cfg = pre_folder.call_args.args[1]
    selected_names = [p.name for p in pre_folder.call_args.kwargs["image_files"]]
    reference_names = [
        p.name for p in pre_folder.call_args.kwargs["boundary_reference_files"]
    ]
    assert pre_cfg.cache_processed_image is True
    assert pre_cfg.generate_grid_tiles is False
    assert selected_names == ["G0F00000_test.png"]
    assert any(name.startswith("W0F00000") for name in reference_names)


def test_process_panel_v2_aoi_tiles_reuse_cached_processed_image(tmp_path):
    """AOI-centered tiles should reuse Phase 1 processed image when it is cached."""
    _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)
    cfg.aoi_coord_inspection_enabled = True
    cfg.grid_tiling_enabled = False
    cfg.image_preprocess_pipeline = [
        {"method": "mean", "params": {"kernel_size": 3}},
    ]

    from capi_inference import AOIReportDefect, CAPIInferencer
    from capi_preprocess import PanelPreprocessResult

    img_path = tmp_path / "G0F00000_test.png"
    cached_processed = np.full((1024, 1024), 23, dtype=np.uint8)
    fake_pre_result = PanelPreprocessResult(
        image_path=img_path,
        lighting="G0F00000",
        foreground_bbox=(0, 0, 1024, 1024),
        panel_polygon=None,
        tiles=[],
        processed_image=cached_processed,
    )
    parsed_report = {
        "G0F00000": [
            AOIReportDefect(
                defect_code="L01",
                product_x=512,
                product_y=512,
                image_prefix="G0F00000",
            )
        ]
    }

    amap = np.zeros((512, 512), dtype=np.float32)
    with patch("capi_preprocess.preprocess_panel_folder",
               return_value={"G0F00000": fake_pre_result}), \
         patch("capi_image_preprocess_lab.apply_preprocess_pipeline") as apply_preprocess, \
         patch.object(CAPIInferencer, "_parse_aoi_report_txt",
                      return_value=parsed_report), \
         patch.object(CAPIInferencer, "_get_model_for",
                      return_value=MagicMock()), \
         patch.object(CAPIInferencer, "predict_tile",
                      return_value=(0.1, amap)) as pred:
        inferencer = CAPIInferencer(cfg)
        results, *_ = inferencer.process_panel(
            tmp_path,
            product_resolution=(1024, 1024),
        )

    apply_preprocess.assert_not_called()
    tile = results[0].tiles[0]
    np.testing.assert_array_equal(tile.image, np.full((512, 512), 23, dtype=np.uint8))
    np.testing.assert_array_equal(pred.call_args.args[0].image, tile.image)


@pytest.mark.parametrize("preprocess_after_tiling", [False, True])
def test_process_panel_v2_runs_b0f_skip_file_with_bright_spot_logic(tmp_path, preprocess_after_tiling):
    """B0F skip_files stay on the legacy bright-spot path instead of model routing."""
    import cv2

    _write_grey_panel_image(tmp_path, "G0F00000")
    b0f_path = tmp_path / "B0F00000_test.png"
    cv2.imwrite(str(b0f_path), np.zeros((1024, 1024), dtype=np.uint8))

    cfg = _make_config(tmp_path)
    cfg.aoi_coord_inspection_enabled = True
    cfg.grid_tiling_enabled = False
    cfg.skip_files = ["B0F00000"]
    cfg.image_preprocess_pipeline = [
        {"method": "mean", "params": {"kernel_size": 3}},
    ]
    cfg.preprocess_after_tiling = preprocess_after_tiling

    from capi_inference import CAPIInferencer, AOIReportDefect
    from capi_preprocess import PanelPreprocessResult

    g_path = tmp_path / "G0F00000_test.png"
    fake_pre_result = PanelPreprocessResult(
        image_path=g_path,
        lighting="G0F00000",
        foreground_bbox=(0, 0, 1024, 1024),
        panel_polygon=None,
        tiles=[],
    )
    parsed_report = {
        "B0F00000": [
            AOIReportDefect(
                defect_code="B01",
                product_x=100,
                product_y=100,
                image_prefix="B0F00000",
            )
        ]
    }

    amap = np.ones((512, 512), dtype=np.float32)

    def _fake_bright(tile):
        tile.is_bright_spot_detection = True
        return 1.0, amap

    with patch("capi_preprocess.preprocess_panel_folder",
               return_value={"G0F00000": fake_pre_result}), \
         patch("capi_image_preprocess_lab.apply_preprocess_pipeline") as apply_preprocess, \
         patch.object(CAPIInferencer, "_parse_aoi_report_txt",
                      return_value=parsed_report), \
         patch.object(CAPIInferencer, "_get_model_for") as get_model, \
         patch.object(CAPIInferencer, "_detect_bright_spots",
                      side_effect=_fake_bright) as bright:
        inferencer = CAPIInferencer(cfg)
        results, *_, returned_report = inferencer.process_panel(
            tmp_path,
            product_resolution=(1024, 1024),
        )

    assert returned_report is parsed_report
    b0f_results = [r for r in results if r.image_path.name.startswith("B0F00000")]
    assert len(b0f_results) == 1
    b0f = b0f_results[0]
    assert len(b0f.tiles) == 1
    tile = b0f.tiles[0]
    assert tile.is_aoi_coord_tile is True
    assert tile.mask is not None
    assert tile.mask.shape == tile.image.shape[:2]
    assert tile.mask.dtype == np.uint8
    np.testing.assert_array_equal(tile.image, np.zeros((512, 512), dtype=np.uint8))
    assert tile.is_bright_spot_detection is True
    assert len(b0f.anomaly_tiles) == 1
    apply_preprocess.assert_not_called()
    bright.assert_called_once()
    get_model.assert_not_called()


def test_bomb_postprocess_skips_b0f_aoi_track_only_tile(tmp_path):
    """B0F AOI track-only tiles are AI OK, so bomb matching must not mark them."""
    from capi_inference import CAPIInferencer, ImageResult, TileInfo

    cfg = _make_config(tmp_path)
    cfg.bomb_defects = [
        BombDefect(
            image_prefix="B0F00000",
            defect_code="B01",
            defect_type="point",
            coordinates=[(0, 0)],
        )
    ]
    inferencer = CAPIInferencer(cfg)

    track_tile = TileInfo(
        tile_id=1,
        x=0,
        y=0,
        width=512,
        height=512,
        image=np.zeros((512, 512), dtype=np.uint8),
        zone="bright_spot",
    )
    track_tile.is_aoi_coord_tile = True
    track_tile.is_aoi_coord_below_threshold = True
    track_tile.is_bright_spot_detection = True
    track_tile.aoi_product_x = 0
    track_tile.aoi_product_y = 0

    ng_tile = TileInfo(
        tile_id=2,
        x=0,
        y=0,
        width=512,
        height=512,
        image=np.zeros((512, 512), dtype=np.uint8),
        zone="bright_spot",
    )
    ng_tile.is_bright_spot_detection = True

    track_map = np.zeros((512, 512), dtype=np.float32)
    ng_map = np.zeros((512, 512), dtype=np.float32)
    ng_map[0, 0] = 1.0
    result = ImageResult(
        image_path=Path("B0F00000_test.png"),
        image_size=(1024, 1024),
        otsu_bounds=(0, 0, 1024, 1024),
        exclusion_regions=[],
        tiles=[track_tile, ng_tile],
        excluded_tile_count=0,
        processed_tile_count=2,
        processing_time=0.0,
        anomaly_tiles=[(track_tile, 0.0, track_map), (ng_tile, 1.0, ng_map)],
        raw_bounds=(0, 0, 1024, 1024),
    )

    inferencer._apply_bomb_postprocess([result], None, (1024, 1024))

    assert track_tile.is_bomb is False
    assert track_tile.bomb_defect_code == ""
    assert ng_tile.is_bomb is True
    assert ng_tile.bomb_defect_code == "B01"


def test_client_point_bomb_can_match_multiple_aoi_tiles_within_tolerance(tmp_path):
    from capi_inference import CAPIInferencer, ImageResult, TileInfo

    cfg = _make_config(tmp_path)
    cfg.bomb_match_tolerance = 50
    inferencer = CAPIInferencer(cfg)

    far_tile = TileInfo(
        tile_id=1,
        x=400,
        y=100,
        width=512,
        height=512,
        image=np.zeros((512, 512), dtype=np.uint8),
        zone="bright_spot",
    )
    far_tile.is_aoi_coord_tile = True
    far_tile.is_bright_spot_detection = True
    far_tile.aoi_product_x = 627
    far_tile.aoi_product_y = 336

    near_tile = TileInfo(
        tile_id=2,
        x=400,
        y=100,
        width=512,
        height=512,
        image=np.zeros((512, 512), dtype=np.uint8),
        zone="bright_spot",
    )
    near_tile.is_aoi_coord_tile = True
    near_tile.is_bright_spot_detection = True
    near_tile.aoi_product_x = 660
    near_tile.aoi_product_y = 310

    anomaly_map = np.zeros((512, 512), dtype=np.float32)
    anomaly_map[209, 256] = 1.0
    result = ImageResult(
        image_path=Path("B0F00000_test.png"),
        image_size=(2000, 1000),
        otsu_bounds=(0, 0, 2000, 1000),
        exclusion_regions=[],
        tiles=[far_tile, near_tile],
        excluded_tile_count=0,
        processed_tile_count=2,
        processing_time=0.0,
        anomaly_tiles=[(far_tile, 1.0, anomaly_map), (near_tile, 1.0, anomaly_map)],
        raw_bounds=(0, 0, 2000, 1000),
    )
    bomb_info = {
        "image_prefix": "B0F00000",
        "defect_type": "point",
        "coordinates": [(656, 309)],
    }

    inferencer._apply_bomb_postprocess([result], bomb_info, (2000, 1000))

    assert far_tile.is_bomb is True
    assert far_tile.bomb_defect_code == "UNKNOWN"
    assert near_tile.is_bomb is True
    assert near_tile.bomb_defect_code == "UNKNOWN"


def test_forced_bomb_detection_skips_client_coord_already_covered_by_aoi(capsys):
    from capi_inference import AOIReportDefect, CAPIInferencer

    cfg = _make_config(Path("."))
    cfg.bomb_area_force_detection_enabled = True
    cfg.bomb_match_tolerance = 50
    inferencer = CAPIInferencer(cfg)

    original_defect = AOIReportDefect(
        defect_code="PCDK2",
        product_x=234,
        product_y=128,
        image_prefix="WGF50500",
    )
    aoi_report = {"WGF50500": [original_defect]}
    bomb_info = {
        "image_prefix": "WGF50500",
        "defect_type": "point",
        "coordinates": [(228, 128), (683, 138)],
    }

    forced_report, forced_count = inferencer._aoi_report_with_forced_client_bomb_coords(
        aoi_report,
        bomb_info,
    )

    assert forced_count == 1
    assert aoi_report["WGF50500"] == [original_defect]
    assert [(d.defect_code, d.product_x, d.product_y) for d in forced_report["WGF50500"]] == [
        ("PCDK2", 234, 128),
        ("BOMB_FORCE", 683, 138),
    ]
    log = capsys.readouterr().out
    assert "AOI已涵蓋=1" in log
    assert "補切=1" in log
    assert "(683, 138)" in log


def test_forced_bomb_detection_does_not_add_tile_when_aoi_covers_all_client_coords(capsys):
    from capi_inference import AOIReportDefect, CAPIInferencer

    cfg = _make_config(Path("."))
    cfg.bomb_area_force_detection_enabled = True
    cfg.bomb_match_tolerance = 50
    inferencer = CAPIInferencer(cfg)

    aoi_report = {
        "WGF50500": [
            AOIReportDefect("PCDK2", 234, 128, "WGF50500"),
            AOIReportDefect("PCDK2", 689, 140, "WGF50500"),
        ]
    }
    bomb_info = {
        "image_prefix": "WGF50500",
        "defect_type": "point",
        "coordinates": [(228, 128), (683, 138)],
    }

    forced_report, forced_count = inferencer._aoi_report_with_forced_client_bomb_coords(
        aoi_report,
        bomb_info,
    )

    assert forced_count == 0
    assert forced_report == aoi_report
    log = capsys.readouterr().out
    assert "AOI已涵蓋=2" in log
    assert "補切=0" in log
    assert "不額外補切 tile" in log


def test_process_panel_v2_duplicate_panel_uses_latest_b0f_skip_file(tmp_path):
    import cv2

    _write_grey_panel_image(tmp_path, "G0F00000")
    old_b0f = tmp_path / "B0F00000_010000.png"
    latest_b0f = tmp_path / "B0F00000_020000.png"
    cv2.imwrite(str(old_b0f), np.zeros((1024, 1024), dtype=np.uint8))
    cv2.imwrite(str(latest_b0f), np.zeros((1024, 1024), dtype=np.uint8))
    os.utime(old_b0f, (1000, 1000))
    os.utime(latest_b0f, (2000, 2000))

    cfg = _make_config(tmp_path)
    cfg.max_images_per_panel = 1
    cfg.aoi_coord_inspection_enabled = True
    cfg.grid_tiling_enabled = False
    cfg.skip_files = ["B0F00000"]

    from capi_inference import CAPIInferencer, AOIReportDefect
    from capi_preprocess import PanelPreprocessResult

    fake_pre_result = PanelPreprocessResult(
        image_path=tmp_path / "G0F00000_test.png",
        lighting="G0F00000",
        foreground_bbox=(0, 0, 1024, 1024),
        panel_polygon=None,
        tiles=[],
    )
    parsed_report = {
        "B0F00000": [
            AOIReportDefect(
                defect_code="B01",
                product_x=100,
                product_y=100,
                image_prefix="B0F00000",
            )
        ]
    }
    amap = np.ones((512, 512), dtype=np.float32)

    def _fake_bright(tile):
        tile.is_bright_spot_detection = True
        return 1.0, amap

    with patch("capi_preprocess.preprocess_panel_folder",
               return_value={"G0F00000": fake_pre_result}), \
         patch.object(CAPIInferencer, "_parse_aoi_report_txt",
                      return_value=parsed_report), \
         patch.object(CAPIInferencer, "_get_model_for") as get_model, \
         patch.object(CAPIInferencer, "_detect_bright_spots",
                      side_effect=_fake_bright) as bright:
        inferencer = CAPIInferencer(cfg)
        results, _omit_vis, _omit_oe, _omit_info, is_dup, _omit_img, returned_report = \
            inferencer.process_panel(tmp_path, product_resolution=(1024, 1024))

    assert is_dup is True
    assert returned_report is parsed_report
    b0f_results = [r for r in results if r.image_path.name.startswith("B0F00000")]
    assert len(b0f_results) == 1
    assert b0f_results[0].image_path.name == "B0F00000_020000.png"
    bright.assert_called_once()
    get_model.assert_not_called()


def test_process_panel_v2_skips_cv_edge_inspector(tmp_path):
    """新架構不應再跑傳統 CV 邊緣檢測：edge.pt 已專責 edge zone。
    若仍呼叫 edge_inspector.inspect，視為 regression。"""
    _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)

    from capi_edge_cv import EdgeDefect
    from capi_inference import CAPIInferencer

    fake_model = MagicMock()
    fake_model.predict.return_value = _make_fake_predict_result(0.1)

    class FakeEdgeInspector:
        def __init__(self):
            self.config = MagicMock()
            self.config.enabled = True
            self.config.exclude_zones = []
            self.config.set_active_zones_for_product = MagicMock()
            self.inspect_calls = 0

        def inspect(self, image, raw_bounds):
            self.inspect_calls += 1
            return [EdgeDefect(side="left", area=10, bbox=(1, 2, 3, 4), center=(2, 3))]

    with patch.object(CAPIInferencer, "_get_model_for", return_value=fake_model):
        inferencer = CAPIInferencer(cfg)
        fake_inspector = FakeEdgeInspector()
        inferencer.edge_inspector = fake_inspector
        results, *_ = inferencer.process_panel(tmp_path)

    assert fake_inspector.inspect_calls == 0, \
        f"新架構 v2 不應呼叫 edge_inspector.inspect，實際 {fake_inspector.inspect_calls} 次"
    assert all(not r.edge_defects for r in results), \
        "新架構不應產生 edge_defects（不跑 CV、不跑 AOI ROI PC）"


def test_process_panel_v2_runs_scratch_filter(tmp_path):
    _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)
    cfg.scratch_classifier_enabled = True

    from capi_inference import CAPIInferencer

    fake_model = MagicMock()
    fake_model.predict.return_value = _make_fake_predict_result(0.9)
    fake_filter = MagicMock()

    with patch.object(CAPIInferencer, "_get_model_for", return_value=fake_model), \
         patch.object(CAPIInferencer, "_get_scratch_filter", return_value=fake_filter):
        inferencer = CAPIInferencer(cfg)
        inferencer.edge_inspector.config.enabled = False
        inferencer.process_panel(tmp_path)

    assert fake_filter.apply_to_image_result.called


def test_predict_tile_applies_mask(tmp_path):
    """_predict_tile should zero out score where mask=0."""
    cfg = _make_config(tmp_path)
    from capi_inference import CAPIInferencer

    inferencer = CAPIInferencer(cfg)

    # Fake model returns anomaly_map with value 1.0 everywhere
    fake_result = MagicMock()
    fake_result.pred_score = 1.0
    amap = np.ones((512, 512), dtype=np.float32)
    fake_result.anomaly_map = amap
    fake_model = MagicMock()
    fake_model.predict.return_value = fake_result

    # mask with only top-left quadrant as panel interior
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[:256, :256] = 255

    tile_img = np.zeros((512, 512, 3), dtype=np.uint8)
    score, out_map = inferencer._predict_tile(fake_model, tile_img, mask)

    assert out_map is not None
    # Pixels outside mask should be 0
    assert out_map[300, 300] == 0.0
    # Pixels inside mask should be 1.0
    assert out_map[100, 100] == 1.0
    # score should equal max of masked map = 1.0
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# run_inference_v2_single_image — debug 單圖路徑
# ---------------------------------------------------------------------------

def test_run_inference_v2_single_image_returns_image_result(tmp_path):
    """v2 debug 單圖推論：missing prefix→None；正常 prefix→ImageResult。"""
    img = _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)

    from capi_inference import CAPIInferencer, ImageResult

    fake_model = MagicMock()
    fake_model.predict.return_value = _make_fake_predict_result(0.3)

    with patch.object(CAPIInferencer, "_get_model_for", return_value=fake_model):
        inferencer = CAPIInferencer(cfg)
        result = inferencer.run_inference_v2_single_image(img, threshold=0.5)

    assert isinstance(result, ImageResult)
    assert result.image_path == img
    assert result.tiles, "should produce at least one tile"
    # 0.3 < 0.5 threshold → 無 anomaly_tiles
    assert result.anomaly_tiles == []


def test_run_inference_v2_single_image_returns_none_when_prefix_missing(tmp_path):
    """Prefix 不在 model_mapping → 回傳 None (不 crash)。"""
    img = _write_grey_panel_image(tmp_path, "R0F00000")  # only G0F00000 in mapping
    cfg = _make_config(tmp_path)

    from capi_inference import CAPIInferencer

    with patch.object(CAPIInferencer, "_get_model_for", return_value=MagicMock()):
        inferencer = CAPIInferencer(cfg)
        result = inferencer.run_inference_v2_single_image(img)

    assert result is None


def test_run_inference_v2_single_image_threshold_override_applies_to_all_zones(tmp_path):
    """傳入的 threshold 同時覆寫 inner/edge (debug UI 拖拉行為)。"""
    img = _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)

    from capi_inference import CAPIInferencer

    fake_model = MagicMock()
    # score=0.6 介於 0.5 (傳入) 與 0.75 (config 預設) 之間
    fake_model.predict.return_value = _make_fake_predict_result(0.6)

    with patch.object(CAPIInferencer, "_get_model_for", return_value=fake_model):
        inferencer = CAPIInferencer(cfg)
        # threshold=0.5 → score 0.6 ≥ 0.5 → 全部 tile 進入 anomaly_tiles
        result = inferencer.run_inference_v2_single_image(img, threshold=0.5)

    assert result is not None
    assert len(result.anomaly_tiles) == len(result.tiles)


def test_run_inference_v2_single_image_dispatches_per_zone(tmp_path):
    """確認 per-tile 依 zone 走 _get_model_for(machine, lighting, zone)。"""
    img = _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)

    from capi_inference import CAPIInferencer

    fake_model = MagicMock()
    fake_model.predict.return_value = _make_fake_predict_result(0.1)

    with patch.object(CAPIInferencer, "_get_model_for", return_value=fake_model) as get_model:
        inferencer = CAPIInferencer(cfg)
        result = inferencer.run_inference_v2_single_image(img, threshold=0.5)

    assert result is not None
    assert get_model.called
    # 每次呼叫都帶 (machine_id, lighting, zone)
    for call in get_model.call_args_list:
        args = call.args
        assert args[0] == "TEST_MACHINE"
        assert args[1] == "G0F00000"
        assert args[2] in ("inner", "edge")


def test_process_panel_v2_aoi_only_mode_preprocesses_reference_for_black_images(tmp_path):
    """When aoi_only_mode is True and report only contains black images, ensure W0F00000 (preferred reference) is preprocessed."""
    _write_grey_panel_image_at(tmp_path / "W0F00000_123.png")
    _write_grey_panel_image_at(tmp_path / "B0F00000_123.png")

    cfg = _make_config(tmp_path)
    cfg.aoi_coord_inspection_enabled = True
    cfg.grid_tiling_enabled = False

    from capi_inference import CAPIInferencer
    import capi_preprocess

    captured_files = []
    original_preprocess_folder = capi_preprocess.preprocess_panel_folder

    def fake_preprocess_panel_folder(
        folder,
        config,
        image_files=None,
        boundary_reference_files=None,
    ):
        if image_files:
            captured_files.extend([f.name for f in image_files])
        return original_preprocess_folder(
            folder,
            config,
            image_files=image_files,
            boundary_reference_files=boundary_reference_files,
        )

    with patch("capi_preprocess.preprocess_panel_folder", side_effect=fake_preprocess_panel_folder), \
         patch.object(CAPIInferencer, "_parse_aoi_report_txt") as mock_parse:
        
        from capi_inference import AOIReportDefect
        mock_parse.return_value = {
            "B0F00000": [
                AOIReportDefect(defect_code="L01", product_x=512, product_y=512, image_prefix="B0F00000")
            ]
        }
        
        inferencer = CAPIInferencer(cfg)
        results, *_ = inferencer.process_panel(tmp_path)

    # W0F00000 should be in captured_files since it's the preferred reference image
    assert any("W0F00000" in f for f in captured_files), f"Expected W0F00000 in preprocessed files, got: {captured_files}"
