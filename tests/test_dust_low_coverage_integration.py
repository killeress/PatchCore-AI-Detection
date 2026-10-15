"""Exercise the production OMIT postprocess, including strict-review routing."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer, ImageResult, TileInfo


def _case(monkeypatch, *, independent=False, original=True, two_stage=False):
    worker = object.__new__(CAPIInferencer)
    worker.config = CAPIConfig()
    worker.config.dust_heatmap_iou_threshold = 0.1
    worker.config.dust_two_stage_enabled = two_stage
    raw = np.full((128, 128), 100, np.uint8)
    cv2.circle(raw, (40, 50), 4, 75, -1)
    if independent:
        cv2.circle(raw, (88, 82), 4, 90, -1)
    precise = np.zeros_like(raw)
    cv2.circle(precise, (40, 50), 7, 255, -1)
    roi = np.zeros_like(raw)
    roi[16:112, 16:112] = 1
    amap = np.where(roi, 0.55, 0.0).astype(np.float32)
    amap[50, 40] = 1.0
    tile = TileInfo(tile_id=0, x=0, y=0, width=128, height=128,
                    image=np.full_like(raw, 100),  # Review must use raw, not this flat image.
                    original_image=raw if original else None,
                    is_aoi_coord_tile=True, aoi_image_x=40, aoi_image_y=50)
    tile.aoi_roi_mask = roi
    tile.score_threshold = 0.34
    result = ImageResult(image_path=Path("W0F00000_test.tif"), image_size=(128, 128),
                         otsu_bounds=(0, 0, 128, 128), exclusion_regions=[], tiles=[tile],
                         excluded_tile_count=0, processed_tile_count=1, processing_time=0,
                         anomaly_tiles=[(tile, 0.83, amap)])
    calls = []

    def mask(_image, extension_override=None, **_kwargs):
        calls.append(extension_override)
        return True, precise.copy(), 0.01, "TEST_OMIT"

    monkeypatch.setattr(worker, "check_dust_or_scratch_feature", mask)
    return worker, tile, result, precise, calls


@pytest.mark.parametrize("two_stage", [False, True])
@pytest.mark.parametrize("independent", [False, True])
def test_postprocess_reviews_raw_and_keeps_unmatched_defect(monkeypatch, two_stage, independent):
    worker, tile, result, omit, calls = _case(
        monkeypatch, two_stage=two_stage, independent=independent)
    worker._apply_omit_dust_postprocess([result], omit, False, "", cpu_workers=1)

    assert tile.is_suspected_dust_or_scratch is (not independent)
    assert 0 in calls  # A fresh unextended core, not the existing candidate mask.
    assert "LOW_COV_FEATURE" in tile.dust_detail_text
    assert tile.dust_region_max_cov < 0.1
    assert tile.dust_two_stage_features is None  # Keep bomb/legacy two-stage evidence separate.
    assert any(f["is_dust"] for f in tile.dust_low_coverage_features)
    if independent:
        assert any(not f["is_dust"] for f in tile.dust_low_coverage_features)
        assert abs(tile.anomaly_peak_x - 88) <= 2
        assert abs(tile.anomaly_peak_y - 82) <= 2


def test_zero_heatmap_and_empty_regions_keep_ng(monkeypatch):
    worker, tile, result, omit, _ = _case(monkeypatch, two_stage=True)
    result.anomaly_tiles[0][2][:] = 0
    worker._apply_omit_dust_postprocess([result], omit, False, "", cpu_workers=1)
    assert tile.is_suspected_dust_or_scratch is False
    assert tile.dust_region_details == []
    assert "REAL_NG" in tile.dust_detail_text


def test_missing_original_does_not_fall_back_to_preprocessed_dust(monkeypatch):
    worker, tile, result, omit, calls = _case(monkeypatch, original=False)
    worker._apply_omit_dust_postprocess([result], omit, False, "", cpu_workers=1)
    assert tile.is_suspected_dust_or_scratch is False
    assert 0 not in calls
    assert "missing raw/OMIT" in tile.dust_detail_text


def test_extended_mask_cannot_substitute_for_precise_core(monkeypatch):
    worker, tile, result, omit, _ = _case(monkeypatch)

    def mask(_image, extension_override=None, **_kwargs):
        selected = np.zeros_like(omit) if extension_override == 0 else omit.copy()
        return True, selected, 0.01, "TEST_OMIT"

    monkeypatch.setattr(worker, "check_dust_or_scratch_feature", mask)
    worker._apply_omit_dust_postprocess([result], omit, False, "", cpu_workers=1)
    assert tile.is_suspected_dust_or_scratch is False
    assert not tile.dust_low_coverage_features


def test_overexposed_omit_skips_low_coverage_review(monkeypatch):
    worker, tile, result, omit, calls = _case(monkeypatch)
    worker._apply_omit_dust_postprocess([result], omit, True, "overexposed", cpu_workers=1)
    assert tile.is_suspected_dust_or_scratch is False
    assert not calls
    assert tile.dust_low_coverage_features is None


def test_non_roi_tile_retains_existing_coverage_decision(monkeypatch):
    worker, tile, result, omit, calls = _case(monkeypatch)
    tile.aoi_roi_mask = None
    worker.config.dust_heatmap_top_percent = 100
    worker._apply_omit_dust_postprocess([result], omit, False, "", cpu_workers=1)
    assert tile.is_suspected_dust_or_scratch is False
    assert 0 not in calls
    assert tile.dust_low_coverage_features is None
