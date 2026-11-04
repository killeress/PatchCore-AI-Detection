"""A dust candidate may pass only after both full-ROI residual checks clear."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from capi_config import BombDefect, CAPIConfig
from capi_inference import CAPIInferencer, ImageResult, TileInfo
from capi_server import _iter_qjpg_defect_records, results_to_db_data
from capi_dust_review import confirmed_roi_surface_mask, surface_removed_images


def _case(monkeypatch, *, defect=None, model_results=None):
    worker = CAPIInferencer.__new__(CAPIInferencer)
    worker.config = CAPIConfig(dust_heatmap_top_percent=0.4, scratch_classifier_enabled=False)
    raw = np.full((128, 128), 100, np.uint8)
    cv2.circle(raw, (40, 50), 6, 75, -1)
    if defect is not None:
        cv2.circle(raw, (88, 82), 4, 100 - defect, -1)
    dust = np.zeros_like(raw)
    cv2.circle(dust, (40, 50), 10, 255, -1)
    roi = np.zeros_like(raw)
    roi[16:112, 16:112] = 1
    support = np.where(roi, 0.42, 0).astype(np.float32)
    support[32, 40] = 1
    core = np.zeros_like(raw)
    core[31:34, 39:42] = 255
    tile = TileInfo(
        tile_id=0, x=0, y=0, width=128, height=128, image=raw.copy(),
        original_image=raw.copy(), is_aoi_coord_tile=True, zone="inner",
        aoi_roi_mask=roi, aoi_roi_full_map=support.copy(), score_threshold=0.34,
        is_suspected_dust_or_scratch=True, dust_heatmap_binary=core,
        dust_region_details=[dict(label_id=1, area=9, dust_overlap=0,
                                 coverage=0., is_dust=True, max_score=1., peak_yx=(32, 40))],
    )
    result = ImageResult(
        image_path=Path("W0F00000_test.tif"), image_size=(128, 128),
        otsu_bounds=(0, 0, 128, 128), exclusion_regions=[], tiles=[tile],
        excluded_tile_count=0, processed_tile_count=1, processing_time=0,
        anomaly_tiles=[(tile, 1., support)],
    )
    model = object()
    monkeypatch.setattr(worker, "_get_model_for", lambda *a: model)
    monkeypatch.setattr(worker, "_get_inferencer_for_prefix", lambda *a: model)
    monkeypatch.setattr(worker, "_check_dust_or_scratch_feature_with_context",
                        lambda *a, **kw: (True, dust.copy(), .1, "OMIT ext0"))
    probes = []

    def predict(probe, *, inferencer, threshold, **kwargs):
        assert inferencer is model and threshold == .34
        assert probe is not tile
        np.testing.assert_array_equal(probe.image[dust == 0], tile.image[dust == 0])
        probes.append(probe.image.copy())
        if model_results is not None:
            value = model_results[len(probes) - 1]
            if value is None:
                return .1, None
            if isinstance(value, Exception):
                raise value
        else:
            # A weak defect can be invisible to CV while remaining visible to
            # the model. The surface edit must preserve this pixel exactly.
            value = .42 if probe.image[82, 88] < 100 else .1
        amap = np.full((64, 64), .05, np.float32)
        amap[41, 44] = value
        return value, amap

    monkeypatch.setattr(worker, "predict_tile", predict)
    return worker, tile, result, raw, dust, support, probes


@pytest.mark.parametrize("defect", [None, 2, 10])
def test_whole_roi_residual_preserves_weak_and_visible_nearby_defects(monkeypatch, defect):
    worker, tile, result, raw, dust, support, probes = _case(monkeypatch, defect=defect)
    core_before = tile.dust_heatmap_binary.copy()
    note = worker._confirm_aoi_dust_residual(result, tile, support, raw, raw)

    assert tile.is_suspected_dust_or_scratch is (defect is None)
    snapshot = tile.decision_context["aoi_dust_residual"]
    assert snapshot["scope"] == "whole_roi"
    assert snapshot["status"] == ("clear" if defect is None else "remaining")
    if defect is None or defect == 2:
        assert len(probes) == 2
    assert "ROI_RESIDUAL" in note
    assert worker.config.dust_heatmap_top_percent == .4
    np.testing.assert_array_equal(tile.image, raw)
    np.testing.assert_array_equal(tile.original_image, raw)
    np.testing.assert_array_equal(tile.dust_heatmap_binary, core_before)
    np.testing.assert_array_equal(tile.aoi_roi_full_map, support)
    if defect == 2:
        assert tile.anomaly_peak_x == 88 and tile.anomaly_peak_y == 82


@pytest.mark.parametrize("model_results", [[.1, .42], [.42, .1], [.1, None],
                                         [.1, float("nan")], [.1, RuntimeError("failed")]])
def test_uncertain_or_disagreeing_residual_checks_never_release(monkeypatch, model_results):
    worker, tile, result, raw, _, support, _ = _case(monkeypatch, model_results=model_results)
    worker._confirm_aoi_dust_residual(result, tile, support, raw, raw)
    assert not tile.is_suspected_dust_or_scratch
    assert tile.decision_context["aoi_dust_residual"]["status"] != "clear"


@pytest.mark.parametrize("missing", ["raw", "model", "precise", "support", "flat_raw", "clipped_mask"])
def test_missing_or_unconfirmed_surface_evidence_keeps_ng(monkeypatch, missing):
    worker, tile, result, raw, dust, support, probes = _case(monkeypatch)
    if missing == "raw":
        tile.original_image = None
    elif missing == "model":
        monkeypatch.setattr(worker, "_get_model_for", lambda *a: None)
        monkeypatch.setattr(worker, "_get_inferencer_for_prefix", lambda *a: None)
    elif missing == "precise":
        monkeypatch.setattr(worker, "_check_dust_or_scratch_feature_with_context",
                            lambda *a, **kw: (False, np.zeros_like(dust), 0., "empty"))
    elif missing == "support":
        support = None
    elif missing == "flat_raw":
        tile.original_image = np.full_like(raw, 100)
    else:
        dust[:51, 35:46] = 255
        monkeypatch.setattr(worker, "_check_dust_or_scratch_feature_with_context",
                            lambda *a, **kw: (True, dust, .1, "clipped"))
    worker._confirm_aoi_dust_residual(result, tile, support, raw, raw)
    assert not tile.is_suspected_dust_or_scratch
    assert not probes
    assert tile.decision_context["aoi_dust_residual"]["status"] == "unconfirmed"


def test_residual_model_ignores_heat_outside_roi(monkeypatch):
    worker, tile, result, raw, _, support, probes = _case(monkeypatch)

    def outside_only(probe, **kwargs):
        probes.append(probe.image.copy())
        values = np.full((64, 64), .1, np.float32)
        values[2, 2] = 1
        return 1., values

    monkeypatch.setattr(worker, "predict_tile", outside_only)
    worker._confirm_aoi_dust_residual(result, tile, support, raw, raw)
    assert tile.is_suspected_dust_or_scratch
    assert len(probes) == 2


def test_guard_skips_non_roi_and_binary_workflows(monkeypatch):
    worker, tile, result, raw, _, support, probes = _case(monkeypatch)
    tile.is_bright_spot_detection = True
    assert worker._confirm_aoi_dust_residual(result, tile, support, raw, raw) == ""
    tile.is_bright_spot_detection = False
    tile.aoi_roi_mask = None
    assert worker._confirm_aoi_dust_residual(result, tile, support, raw, raw) == ""
    assert not probes


def test_surface_edit_preserves_unverified_pixels_inside_omit_component(monkeypatch):
    _, tile, _, raw, dust, support, _ = _case(monkeypatch)
    mask, _, reason = confirmed_roi_surface_mask(raw, dust, tile.aoi_roi_mask, support)
    assert reason == "confirmed_surface"
    assert np.count_nonzero(mask) < np.count_nonzero(dust)
    assert dust[50, 49] > 0 and mask[50, 49] == 0
    variants = list(surface_removed_images(tile.image, mask))
    assert len(variants) == 2
    for _, edited in variants:
        np.testing.assert_array_equal(edited[mask == 0], tile.image[mask == 0])


@pytest.mark.parametrize("status", ["remaining", "unconfirmed"])
def test_later_scratch_filter_cannot_override_residual_ng(monkeypatch, status):
    from types import SimpleNamespace
    from scratch_filter import ScratchFilter
    worker, tile, result, raw, _, support, _ = _case(monkeypatch, defect=2)
    if status == "unconfirmed":
        tile.original_image = None
    worker._confirm_aoi_dust_residual(result, tile, support, raw, raw)
    calls = []
    classifier = SimpleNamespace(conformal_threshold=.7, predict=lambda im: calls.append(im) or .999)
    ScratchFilter(classifier).apply_to_image_result(result)
    assert tile.decision_context["aoi_dust_residual"]["status"] == status
    assert not tile.scratch_filtered and not calls
    assert results_to_db_data([result], {})[0]["is_ng"] == 1


def test_neighbor_bomb_cannot_use_pre_removal_core_to_clear_new_residual(monkeypatch):
    worker, tile, result, raw, _, support, _ = _case(monkeypatch, defect=2)
    worker._confirm_aoi_dust_residual(result, tile, support, raw, raw)
    result.raw_bounds = (0, 0, 128, 128)
    tile.aoi_product_x = tile.aoi_image_x = 80
    tile.aoi_product_y = tile.aoi_image_y = 80
    worker.config.bomb_match_tolerance = 5
    bomb = BombDefect("W0F00000", "B01", "point", [(40, 32)])
    matched = worker._match_aoi_point_bomb_regions(result, tile, support, [bomb], (128, 128), {})
    assert matched == (False, "")
    assert (tile.anomaly_peak_x, tile.anomaly_peak_y) == (88, 82)


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("defect", [None, 2, 10])
def test_production_paths_gate_dust_on_whole_roi_residual(monkeypatch, tmp_path, version, defect):
    worker, tile, result, raw, _, _, probes = _case(monkeypatch, defect=defect)
    worker.config.dust_heatmap_iou_threshold = .1
    yy, xx = np.mgrid[:128, :128]
    full_map = np.exp(-((xx - 40) ** 2 + (yy - 32) ** 2) / (2 * 16. ** 2)).astype(np.float32)
    if defect is not None:
        weaker = .42 * np.exp(-((xx - 88) ** 2 + (yy - 82) ** 2) / (2 * 7. ** 2))
        full_map = np.maximum(full_map, weaker).astype(np.float32)
    full_map[tile.aoi_roi_mask == 0] = 0
    source = np.where(full_map >= .34, full_map, 0)
    tile.aoi_roi_full_map = full_map
    tile.is_suspected_dust_or_scratch = False
    result.raw_bounds = (0, 0, 128, 128)
    result.anomaly_tiles = [(tile, 1., source)]
    if version == "v1":
        from test_aoi_roi_dust_core_judgment import _run_v1
        _run_v1(worker, result, monkeypatch, tmp_path)
    else:
        worker._apply_omit_dust_postprocess([result], raw, False, "", cpu_workers=1)
    assert tile.is_suspected_dust_or_scratch is (defect is None)
    assert "ROI_RESIDUAL" in tile.dust_detail_text
    assert np.count_nonzero(tile.dust_heatmap_binary) < np.count_nonzero(tile.aoi_roi_mask) * .005
    assert tile.dust_heatmap_iou == 0
    snapshot = tile.decision_context["aoi_dust_residual"]
    assert snapshot["status"] == ("clear" if defect is None else "remaining")
    assert results_to_db_data([result], {})[0]["is_ng"] == int(defect is not None)
    reported = _iter_qjpg_defect_records([result], (128, 128), worker.config)
    assert any(row.startswith("PCDK2") for row in reported) is (defect is not None)
    if defect == 2:
        assert len(probes) == 2
