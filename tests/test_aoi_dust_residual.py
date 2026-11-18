"""Exercise whole-ROI dust decisions on original Grid tiles and legacy residual probes."""
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


@pytest.mark.parametrize("scene,checks,clear", [
    ("perimeter", 2, True), ("model_detects_edge", 2, False),
    ("missing_polygon", 0, False), ("offset_polygon", 0, False),
    ("inward_branch", 0, False), ("compact_edge_spot", 0, False),
    ("independent_spot", 0, False), ("no_dust_signal", 0, False),
])
def test_panel_perimeter_in_corner_tile_is_preserved_for_both_probes(monkeypatch, scene, checks, clear):
    worker, tile, result, _, _, _, probes = _case(monkeypatch)
    # Reproduce the reported corner geometry, including the inward tile shift.
    tile.x, tile.y = 5438, 3684
    tile.width = tile.height = 512
    result.panel_polygon = np.float32([[263, 656.3], [5942.2, 647.3],
                                       [5950.7, 4195.8], [267.9, 4207.7]])
    if scene == "missing_polygon":
        result.panel_polygon = None
    if scene == "offset_polygon":
        result.panel_polygon[:, 0] += 35
    raw = np.full((512, 512), 100, np.uint8)
    if scene == "compact_edge_spot":
        raw[240:248, -3:] = 50
    else:
        raw[:, -8:] -= np.arange(8, dtype=np.uint8) * 4
    if scene == "inward_branch":
        raw[200:204, -32:] = 60
    if scene == "independent_spot":
        cv2.circle(raw, (486, 200), 3, 65, -1)
    if scene != "no_dust_signal":
        cv2.circle(raw, (434, 291), 6, 75, -1)
    dust = np.zeros_like(raw)
    cv2.circle(dust, (434, 291), 10, 255, -1)
    roi = np.zeros_like(raw)
    roi[99:483, 242:] = 1
    support = np.where(roi, .42, 0).astype(np.float32)
    tile.image = raw.copy()
    tile.original_image = raw.copy()
    tile.aoi_roi_mask = roi
    tile.aoi_roi_full_map = support.copy()
    tile.dust_region_details[0]["coverage"] = 1.
    monkeypatch.setattr(worker, "_check_dust_or_scratch_feature_with_context",
                        lambda *a, **kw: (True, dust.copy(), .1, "OMIT"))

    def predict(probe, **kwargs):
        probes.append(probe.image.copy())
        values = np.full(raw.shape, .1, np.float32)
        if scene == "model_detects_edge" and len(probes) == 2:
            values[220, 509] = .5
        return float(values.max()), values

    monkeypatch.setattr(worker, "predict_tile", predict)
    worker._confirm_aoi_dust_residual(result, tile, support, raw, raw)
    assert len(probes) == checks
    assert tile.is_suspected_dust_or_scratch is clear
    for probe in probes:
        np.testing.assert_array_equal(probe[dust == 0], raw[dust == 0])
        assert probe[291, 434] > raw[291, 434]
    if scene == "model_detects_edge":
        assert (tile.anomaly_peak_x, tile.anomaly_peak_y) == (tile.x + 509, tile.y + 220)


@pytest.mark.parametrize("pipeline", ["grid_only", "before_grid", "edge_dog"])
def test_grid_dust_uses_configured_coverage_in_production(monkeypatch, tmp_path, pipeline):
    from capi_inference import AOIReportDefect
    from test_process_panel_v2 import _make_config

    config = _make_config(tmp_path)
    config.tile_size = 128
    config.grid_tiling_enabled = False
    config.grid_canonicalization_enabled = True
    config.grid_product_resolution = (184, 124)
    config.grid_samples_per_cell = 1 if pipeline == "before_grid" else 3
    config.inference_rotate_180_enabled = True
    config.aoi_coord_inspection_enabled = config.aoi_judgment_roi_enabled = True
    config.dust_heatmap_top_percent = 5.0
    if pipeline == "before_grid":
        config.image_preprocess_pipeline = [{"method": "clahe"}]
    if pipeline == "edge_dog":
        config.preprocess_after_tiling = True
        config.image_preprocess_pipelines = {"edge": [{"method": "dog"}], "inner": []}

    raw = np.zeros((310, 432), np.uint8)
    polygon = np.array([[32, 34], [400, 29], [395, 278], [37, 280]], np.int32)
    cv2.fillPoly(raw, [polygon], 100)
    point = (355, 177)
    cv2.circle(raw, point, 6, 75, -1)
    omit = (18 + np.indices(raw.shape)[0] % 3).astype(np.uint8)
    cv2.circle(omit, point, 10, 175, -1)
    path = tmp_path / "G0F00000_test.tif"
    assert cv2.imwrite(str(path), cv2.flip(raw, -1))
    assert cv2.imwrite(str(tmp_path / "PINIGBI _test.tif"), cv2.flip(omit, -1))
    report = AOIReportDefect("C1111", raw.shape[1] - 1 - point[0], raw.shape[0] - 1 - point[1],
                             "G0F00000", coordinate_space="image")

    model = object()
    worker = CAPIInferencer(config)
    monkeypatch.setattr(worker, "_get_model_for", lambda machine, prefix, zone: model if zone == "edge" else None)
    inputs = []

    def predict(tile, *, inferencer, threshold, **kwargs):
        assert inferencer is model and threshold == .5
        x, y = point[0] - tile.x, point[1] - tile.y
        values = np.zeros((tile.height, tile.width), np.float32)
        cv2.circle(values, (x, y), 24, .6, -1)
        cv2.circle(values, (x, y), 5, 1., -1)
        inputs.append(tile.image.copy())
        return 1., values

    monkeypatch.setattr(worker, "predict_tile", predict)
    monkeypatch.setattr("capi_dust_review.surface_removed_images",
                        lambda *args: pytest.fail("Grid dust review must use the original tile"))
    results, *_ = worker.process_panel(tmp_path, cpu_workers=1, client_product_resolution=(184, 124),
                                       aoi_report_override={"G0F00000": [report]})
    result, = results
    tile, = result.tiles
    assert tile.is_suspected_dust_or_scratch
    assert "aoi_dust_residual" not in tile.decision_context
    assert "PER_REGION" in tile.dust_detail_text
    assert tile.decision_context["aoi_judgment_roi"]["dust_top_percent"] == 5.0
    assert len(inputs) == 1
    assert tile.x > 0 and tile.y > 0 and tile.aoi_tile_shift_dx != 0
    np.testing.assert_array_equal(tile.original_image, raw[tile.y:tile.y+128, tile.x:tile.x+128])


@pytest.mark.parametrize("low_cov_overlap", [False, True])
def test_grid_real_defect_with_unrelated_dust_records_real_regions_not_unconfirmed(
    monkeypatch, tmp_path, low_cov_overlap,
):
    from capi_inference import AOIReportDefect
    from test_process_panel_v2 import _make_config

    config = _make_config(tmp_path)
    config.model_mapping["W0F00000"] = config.model_mapping["G0F00000"]
    config.threshold_mapping["W0F00000"] = {"inner": 0.5, "edge": 0.5}
    config.tile_size = 128
    config.grid_tiling_enabled = False
    config.grid_canonicalization_enabled = True
    config.grid_product_resolution = (184, 124)
    config.grid_samples_per_cell = 3
    config.inference_rotate_180_enabled = False
    config.aoi_coord_inspection_enabled = config.aoi_judgment_roi_enabled = True
    config.dust_heatmap_top_percent = 5.0
    config.dust_heatmap_iou_threshold = 0.5

    raw = np.zeros((310, 432), np.uint8)
    polygon = np.array([[32, 34], [400, 29], [395, 278], [37, 280]], np.int32)
    cv2.fillPoly(raw, [polygon], 100)
    defect_xy = (200, 150)
    cv2.circle(raw, defect_xy, 5, 60, -1)
    omit = (18 + np.indices(raw.shape)[0] % 3).astype(np.uint8)
    dust_xy = (193, 150) if low_cov_overlap else (160, 120)
    cv2.circle(omit, dust_xy, 5, 180, -1)
    if low_cov_overlap:
        cv2.circle(raw, dust_xy, 5, 75, -1)
    path = tmp_path / "W0F00000_test.tif"
    assert cv2.imwrite(str(path), raw)
    assert cv2.imwrite(str(tmp_path / "PINIGBI _test.tif"), omit)
    report = AOIReportDefect("C1111", defect_xy[0], defect_xy[1], "W0F00000", coordinate_space="image")

    model = object()
    worker = CAPIInferencer(config)
    monkeypatch.setattr(worker, "_get_model_for", lambda machine, prefix, zone: model if zone == "inner" else None)

    def predict(tile, *, inferencer, threshold, **kwargs):
        assert inferencer is model and threshold == .5
        x, y = defect_xy[0] - tile.x, defect_xy[1] - tile.y
        values = np.zeros((tile.height, tile.width), np.float32)
        cv2.circle(values, (x, y), 18, .6, -1)
        cv2.circle(values, (x, y), 4, 1., -1)
        return 1., values

    monkeypatch.setattr(worker, "predict_tile", predict)
    results, *_ = worker.process_panel(tmp_path, cpu_workers=1, client_product_resolution=(184, 124),
                                       aoi_report_override={"W0F00000": [report]})
    result, = results
    tile, = result.tiles
    assert tile.is_suspected_dust_or_scratch is False
    assert "aoi_dust_residual" not in (tile.decision_context or {})
    assert "GRID_ORIGINAL_ROI" not in tile.dust_detail_text
    assert "LOW_COV_FEATURE" not in tile.dust_detail_text
    assert "REAL_NG" in tile.dust_detail_text
    cov = tile.dust_region_details[0]["coverage"]
    assert (0 < cov < 0.5) if low_cov_overlap else (cov == 0)
    db_data = results_to_db_data([result], {})
    import json
    saved_ctx = json.loads(db_data[0]["tiles"][0]["decision_context"])
    assert "aoi_dust_residual" not in saved_ctx
    assert saved_ctx["aoi_real_regions"]["count"] == 1
    assert len(saved_ctx["aoi_real_regions"]["peaks_image_xy"]) == 1
    saved_peak = saved_ctx["aoi_real_regions"]["peaks_image_xy"][0]
    assert np.linalg.norm(np.subtract(saved_peak, defect_xy)) <= 5


@pytest.mark.parametrize("scene,expected", [
    ("clean", "clear"), ("outside_top", "remaining"), ("partial_core", "unconfirmed"),
    ("missing_raw", "unconfirmed"), ("empty_core", "unconfirmed"),
    ("no_background", "unconfirmed"), ("outside_roi", "clear"),
    ("excluded_spot", "clear"), ("compact_edge_spot", "remaining"),
])
def test_grid_checks_precise_core_and_original_even_outside_model_support(monkeypatch, scene, expected):
    worker, tile, result, raw, dust, support, probes = _case(monkeypatch)
    worker.config.grid_canonicalization_enabled = True
    # The visible dust feature and the model core share the same native mask.
    tile.dust_heatmap_binary[:] = 0
    tile.dust_heatmap_binary[49:52, 39:42] = 255
    support[:] = 0
    support[49:52, 39:42] = 1
    omit = (16 + np.indices(raw.shape)[0] % 3).astype(np.uint8)
    omit[dust > 0] = 180
    if scene == "outside_top":
        cv2.circle(tile.original_image, (88, 82), 4, 70, -1)
    elif scene == "partial_core":
        tile.dust_heatmap_binary[49, 65] = 255
        tile.dust_mask = np.full_like(dust, 255)  # Dilated mask must never grant coverage.
    elif scene == "missing_raw":
        tile.original_image = None
    elif scene == "empty_core":
        tile.dust_heatmap_binary[:] = 0
    elif scene == "no_background":
        dust[:] = 255
    elif scene == "outside_roi":
        cv2.circle(tile.original_image, (5, 5), 4, 70, -1)
    elif scene in ("excluded_spot", "compact_edge_spot"):
        cv2.circle(tile.original_image, (102, 82), 4, 70, -1)
        if scene == "excluded_spot":
            tile.mask = np.full_like(raw, 255)
            tile.mask[72:93, 92:113] = 0
        else:
            result.panel_polygon = np.float32([[0, 0], [110, 0], [110, 127], [0, 127]])
    note = worker._confirm_aoi_dust_residual(result, tile, support, omit, omit)
    review = tile.decision_context["aoi_dust_residual"]
    assert review["status"] == expected
    assert tile.is_suspected_dust_or_scratch == (expected == "clear")
    assert not probes and "GRID_ORIGINAL_ROI" in note
    if scene == "outside_top":
        assert np.linalg.norm(np.subtract(review["remaining_image_xy"], [88, 82])) <= 5


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
def test_uncovered_core_keeps_ng_without_rerunning_the_model(monkeypatch, tmp_path, version, defect):
    worker, tile, result, raw, _, _, probes = _case(monkeypatch, defect=defect)
    worker.config.dust_heatmap_iou_threshold = .1
    worker.config.aoi_heatmap_center_seed_enabled = False
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
    assert not tile.is_suspected_dust_or_scratch
    assert "REAL_NG" in tile.dust_detail_text
    assert np.count_nonzero(tile.dust_heatmap_binary) < np.count_nonzero(tile.aoi_roi_mask) * .005
    assert tile.dust_heatmap_iou == 0
    assert "aoi_dust_residual" not in tile.decision_context
    assert results_to_db_data([result], {})[0]["is_ng"] == 1
    reported = _iter_qjpg_defect_records([result], (128, 128), worker.config)
    assert any(row.startswith("PCDK2") for row in reported)
    assert not probes


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


def test_diffuse_dust_edit_leaves_omit_response_at_polygon_edge_untouched():
    raw, dust, omit, roi, _ = _faint_stripe()
    roi[16:112, 16:] = 1
    raw[:, -8:] -= np.arange(8, dtype=np.uint8) * 4
    dust[:, -5:] = 255
    omit[:, -5:] = 70
    polygon = np.float32([[-600, -600], [127, -600], [128, 140], [-600, 140]])
    edit, _, reason = confirmed_roi_surface_mask(
        raw, dust, roi, roi.astype(np.float32), omit_crop=omit, high_coverage_fallback=True,
        panel_polygon=polygon,
    )
    assert reason == "confirmed_surface"
    np.testing.assert_array_equal(edit[:, :-17], dust[:, :-17])
    assert not np.any(edit[:, -17:])


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
