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


@pytest.mark.parametrize("flat_raw", [False, True])
def test_high_coverage_faint_diffuse_dust_falls_back_to_whole_roi_residual(monkeypatch, flat_raw):
    worker, tile, result, raw, dust, support, probes = _case(monkeypatch)
    rng = np.random.default_rng(23)
    noise_bg = np.clip(100 + rng.normal(0, 0.3, size=raw.shape), 0, 255).astype(np.uint8)
    raw = noise_bg.copy()
    if not flat_raw:
        # 漫射弱殘差 (< 2.0，不足以形成 >= 3px 獨立元件，但在精準灰塵區明顯高於背景雜訊)
        raw[60:64, 30:90] = np.clip(raw[60:64, 30:90].astype(np.int16) - 2, 0, 255).astype(np.uint8)
    dust = np.zeros_like(raw)
    dust[59:65, 28:92] = 255
    omit = rng.integers(10, 14, size=raw.shape, dtype=np.uint8)
    omit[59:65, 28:92] = 60
    tile.image = raw.copy()
    tile.original_image = raw.copy()
    tile.dust_region_details = [dict(
        label_id=1, area=300, dust_overlap=276, coverage=0.92, is_dust=True,
        max_score=0.45, peak_yx=(61, 60),
    )]
    tile.is_suspected_dust_or_scratch = True

    monkeypatch.setattr(worker, "_check_dust_or_scratch_feature_with_context",
                        lambda *a, **kw: (True, dust.copy(), 0.05, "STRIPE_OMIT"))
    monkeypatch.setattr(worker, "predict_tile", lambda probe, **kw: (
        probes.append(probe.image.copy()) or 0.15,
        np.full(raw.shape, 0.15, np.float32),
    ))

    note = worker._confirm_aoi_dust_residual(result, tile, support, omit, omit)
    if flat_raw:
        assert tile.is_suspected_dust_or_scratch is False
        assert tile.decision_context["aoi_dust_residual"]["status"] == "unconfirmed"
        assert not probes
    else:
        assert tile.is_suspected_dust_or_scratch is True
        assert "ROI_RESIDUAL: clear both_fills_below_threshold checks=2/2 -> DUST" in note
        assert len(probes) == 2


def _faint_stripe(seed=23):
    rng = np.random.default_rng(seed)
    raw = np.clip(100 + rng.normal(0, 0.3, size=(128, 128)), 0, 255).astype(np.uint8)
    raw[60:64, 30:90] -= 2
    dust = np.zeros_like(raw)
    dust[59:65, 28:92] = 255
    omit = rng.integers(10, 14, size=raw.shape, dtype=np.uint8)
    omit[59:65, 28:92] = 60
    roi = np.zeros_like(raw)
    roi[16:112, 16:112] = 1
    support = np.where(roi, 0.42, 0).astype(np.float32)
    return raw, dust, omit, roi, support


def test_high_coverage_fallback_edits_only_components_with_own_signal():
    raw, dust, omit, roi, support = _faint_stripe()
    dust[90:96, 40:50] = 255  # Separate OMIT component without any raw signal
    omit[90:96, 40:50] = 60

    edit, _, reason = confirmed_roi_surface_mask(
        raw, dust, roi, support, omit_crop=omit, high_coverage_fallback=True,
    )

    assert reason == "confirmed_surface"
    assert np.any(edit[60:64, 30:90] > 0)
    assert not np.any(edit[90:96, 40:50] > 0)


def test_high_coverage_fallback_rejects_stripe_clipped_by_roi_or_tile_edge():
    raw, dust, omit, roi, support = _faint_stripe()
    raw[60:64, :] = raw[60:64, :].min(axis=1, keepdims=True)
    dust[59:65, :] = 255  # Original OMIT stripe runs past ROI to both tile edges
    omit[59:65, :] = 60

    edit, _, reason = confirmed_roi_surface_mask(
        raw, dust, roi, support, omit_crop=omit, high_coverage_fallback=True,
    )

    assert edit is None
    assert reason == "surface_mask_clipped"


def test_high_coverage_fallback_allows_tiny_roi_fringe_and_isolated_spike_noise():
    rng = np.random.default_rng(23)
    raw = np.clip(100 + rng.normal(0, 0.3, size=(128, 128)), 0, 255).astype(np.uint8)
    raw[60:64, 24:108] -= 2
    # Isolated 1-2px fluctuations inside the diffuse stripe (not >=3px strong component)
    raw[61, 50] = 96
    raw[62, 80:82] = 96
    dust = np.zeros_like(raw)
    dust[59:65, 16:110] = 255
    dust[60:63, 15:16] = 255  # Extends 1 px left of ROI (3 px outside / 567 px = 0.53%)
    omit = rng.integers(10, 14, size=raw.shape, dtype=np.uint8)
    omit[dust > 0] = 60
    roi = np.zeros_like(raw)
    roi[16:112, 16:112] = 1
    support = np.where((roi > 0) & (np.arange(128)[None, :] >= 20) & (np.arange(128)[None, :] <= 105), 0.42, 0).astype(np.float32)

    edit, _, reason = confirmed_roi_surface_mask(
        raw, dust, roi, support, omit_crop=omit, high_coverage_fallback=True,
    )

    assert reason == "confirmed_surface"
    assert int(np.count_nonzero(edit)) == int(np.count_nonzero(dust))


def test_high_coverage_fallback_rejects_connected_3px_strong_feature(monkeypatch):
    import capi_dust_review

    raw, dust, omit, roi, support = _faint_stripe()
    # Create a >=3px connected strong anomaly inside the stripe
    raw[60:63, 55:58] = 88

    # An inconclusive primary review is required to enter the fallback. Normally
    # this strong feature matches OMIT and is handled by the primary review.
    monkeypatch.setattr(
        capi_dust_review, "review_low_coverage_regions",
        lambda *args, **kwargs: ([dict(
            is_dust=False, feature_review_reason="no_matched_features",
        )], []),
    )

    edit, _, reason = confirmed_roi_surface_mask(
        raw, dust, roi, support, omit_crop=omit, high_coverage_fallback=True,
    )

    assert edit is None
    assert reason == "fallback_strong_feature"
