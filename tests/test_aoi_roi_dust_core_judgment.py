"""Formal ROI dust coverage uses exactly the Top% core shown in diagnostics."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

import capi_heatmap
from capi_config import CAPIConfig
from capi_heatmap import HeatmapManager
from capi_inference import CAPIInferencer, ImageResult, TileInfo
from capi_server import _iter_qjpg_defect_records, results_to_db_data


def _scene(monkeypatch, *, two_stage=False):
    worker = CAPIInferencer.__new__(CAPIInferencer)
    worker.config = CAPIConfig(
        aoi_judgment_roi_enabled=True, dust_heatmap_top_percent=0.3,
        dust_heatmap_iou_threshold=0.15, dust_two_stage_enabled=two_stage,
        scratch_classifier_enabled=False, enable_panel_polygon=False,
    )
    image = np.full((512, 512), 90, np.uint8)
    roi = np.zeros_like(image)
    roi[180:292, 190:311] = 1  # 13552 ROI samples; Top 0.3% retains 41 pixels.
    full_map = np.full(image.shape, 2, np.float32)  # Strong outside heat is irrelevant.
    full_map[roi > 0] = 0.55
    roi_y, roi_x = np.where(roi > 0)
    full_map[roi_y[:263], roi_x[:263]] = 0.2
    core = np.zeros_like(image)
    core[240:245, 250:258] = 255
    core[245, 254] = 255
    full_map[core > 0] = np.linspace(0.9, 0.9347, 41)
    amap = np.where((roi > 0) & (full_map >= 0.34), full_map, 0).astype(np.float32)
    dust = np.zeros_like(image)
    dust[242:245, 253:256] = 255  # 9 core pixels, with the peak just outside dust.
    other_y, other_x = np.where((amap > 0) & (core == 0))
    dust[other_y[:223], other_x[:223]] = 255
    tile = TileInfo(
        tile_id=0, x=0, y=0, width=512, height=512, image=image,
        original_image=image.copy(), is_aoi_coord_tile=True,
        aoi_image_x=256, aoi_image_y=256, aoi_product_x=1025, aoi_product_y=135,
        score_threshold=0.34, aoi_roi_mask=roi, aoi_roi_full_map=full_map,
        decision_context={"aoi_judgment_roi": dict(
            enabled=True, radius_product_px=25, aoi_product_xy=[1025, 135],
            full_tile_score=0.9347, roi_score=0.9347, display_top_percent=0.3)},
    )
    result = ImageResult(
        image_path=Path("W0F00000_core.tif"), image_size=(512, 512),
        otsu_bounds=(0, 0, 512, 512), raw_bounds=(0, 0, 512, 512),
        exclusion_regions=[], tiles=[tile], excluded_tile_count=0,
        processed_tile_count=1, processing_time=0,
        anomaly_tiles=[(tile, 0.9347, amap)],
    )
    monkeypatch.setattr(worker, "_check_dust_or_scratch_feature_with_context",
                        lambda *args, **kwargs: (True, dust.copy(), 0.1, "TEST_OMIT"))
    return worker, tile, result, amap, dust, core


def _run_v1(worker, result, monkeypatch, tmp_path):
    worker.threshold = 0.34
    worker.base_dir = tmp_path
    worker.mark_template = None
    worker.inferencer = MagicMock()
    worker.edge_inspector = None
    worker._model_mapping = {}
    worker._threshold_mapping = {}
    worker._inferencers = {}
    monkeypatch.setattr(worker, "_prepare_panel_image_files",
                        lambda _: ([result.image_path, Path("PINIGBI _core.tif")], False))
    monkeypatch.setattr(worker, "_detect_panel_mark_binary_region", lambda *a, **kw: (None, []))
    monkeypatch.setattr(worker, "_parse_defect_txt", lambda _: {})
    monkeypatch.setattr(worker, "_find_raw_object_bounds",
                        lambda *a, **kw: (result.raw_bounds, np.ones((64, 64), np.uint8)))
    monkeypatch.setattr(worker, "_read_detection_image", lambda *a, **kw: result.tiles[0].image.copy())
    monkeypatch.setattr(worker, "check_omit_overexposure", lambda *a, **kw: (False, 20, 0, "TEST"))
    monkeypatch.setattr(worker, "preprocess_image", lambda *a, **kw: result)
    monkeypatch.setattr(worker, "run_inference", lambda image_result, **kw: image_result)
    # This fixture supplies the ROI geometry captured before the dust stage.
    monkeypatch.setattr(worker, "_apply_aoi_judgment_roi", lambda *a, **kw: None)
    worker._process_panel_v1(tmp_path, cpu_workers=1, product_resolution=(1920, 1200),
                             aoi_report_override={})


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("two_stage", [False, True])
def test_formal_coverage_is_9_over_41_and_reaches_aoi_ok(monkeypatch, tmp_path, version, two_stage):
    worker, tile, result, amap, dust, core = _scene(monkeypatch, two_stage=two_stage)
    legacy = worker.check_dust_per_region(dust, amap, top_percent=100,
                                         iou_threshold=0.15)
    assert legacy[0] is True
    assert legacy[3][0]["area"] == 13289
    assert legacy[3][0]["dust_overlap"] == 232
    assert legacy[3][0]["coverage"] == pytest.approx(232 / 13289)
    two_stage_cores = []
    actual_two_stage = worker.check_dust_two_stage

    def capture_two_stage(image, values, *args, **kwargs):
        selected = kwargs["heatmap_core_mask"]
        two_stage_cores.append(selected.copy())
        np.testing.assert_array_equal(selected, core)
        assert not np.any(values[selected == 0])
        return actual_two_stage(image, values, *args, **kwargs)

    monkeypatch.setattr(worker, "check_dust_two_stage", capture_two_stage)
    if version == "v1":
        _run_v1(worker, result, monkeypatch, tmp_path)
    else:
        worker._apply_omit_dust_postprocess([result], tile.image, False, "", cpu_workers=1)
    np.testing.assert_array_equal(tile.dust_heatmap_binary, core)
    region = tile.dust_region_details[0]
    assert region["area"] == region["metric_denominator"] == 41
    assert region["dust_overlap"] == 9
    assert region["coverage"] == pytest.approx(9 / 41)
    assert tile.dust_heatmap_iou == pytest.approx(9 / 41)
    assert region["peak_in_dust"] is False
    assert region["dust_sub_peak_rescue"] is True
    assert tile.is_suspected_dust_or_scratch
    assert bool(two_stage_cores) is two_stage
    stored = results_to_db_data([result], {})[0]
    assert stored["is_ng"] == 0
    scope = json.loads(stored["tiles"][0]["decision_context"])["aoi_judgment_roi"]
    assert scope["dust_top_percent"] == 0.3
    assert scope["dust_core_pixel_count"] == 41
    assert not any(row.startswith("PCDK2") for row in
                   _iter_qjpg_defect_records([result], (1920, 1200), worker.config))


def test_render_reuses_formal_core_without_reranking(monkeypatch, tmp_path):
    worker, tile, result, amap, _, core = _scene(monkeypatch)
    worker._apply_omit_dust_postprocess([result], tile.image, False, "", cpu_workers=1)
    worker.config.dust_heatmap_top_percent = 10  # Later settings cannot change this evidence.
    monkeypatch.setattr(capi_heatmap, "build_aoi_roi_core_mask",
                        lambda *a, **kw: pytest.fail("The formal core must not be selected twice"))
    masks, texts = [], []
    regular_debug = capi_heatmap.build_aoi_roi_debug_panel
    regular_text = cv2.putText

    def capture_debug(image, values, binary, *args, **kwargs):
        masks.append(binary.copy())
        return regular_debug(image, values, binary, *args, **kwargs)

    def capture_text(image, text, *args, **kwargs):
        texts.append(text)
        return regular_text(image, text, *args, **kwargs)

    monkeypatch.setattr(capi_heatmap, "build_aoi_roi_debug_panel", capture_debug)
    monkeypatch.setattr(cv2, "putText", capture_text)
    output = HeatmapManager(tmp_path).save_tile_heatmap(
        tmp_path, "unified", 0, tile.image, amap, 0.9347, tile_info=tile,
        dust_heatmap_top_percent=10,
    )
    np.testing.assert_array_equal(masks[0], core)
    assert "COV: 9/41 = 0.2195" in texts
    assert "ROI Top 0.3% (decision / display)" in texts
    assert "Dust Mask (ROI CoreCOV:0.220)" in texts
    assert not any(text.startswith("Decision COV") or "(display)" in text for text in texts)
    assert cv2.imread(output) is not None


def test_empty_formal_core_cannot_be_filtered_as_dust(monkeypatch):
    worker, _, _, amap, dust, _ = _scene(monkeypatch)
    outcome = worker.check_dust_per_region(dust, amap, core_mask=np.zeros_like(dust))
    assert outcome[0] is True
    assert outcome[3] == []
    assert not np.any(outcome[4])


def test_selected_roi_core_is_not_changed_by_legacy_mask_before_binarize(monkeypatch):
    worker, tile, _, amap, dust, core = _scene(monkeypatch)
    worker.config.dust_mask_before_binarize = True
    outcome = worker._check_tile_dust_regions(tile, amap, dust)
    np.testing.assert_array_equal(outcome[4], core)
    assert outcome[3][0]["area"] == 41
    assert outcome[3][0]["coverage"] == pytest.approx(9 / 41)
    assert outcome[0] is False
