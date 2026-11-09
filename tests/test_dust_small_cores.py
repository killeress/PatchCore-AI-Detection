"""Small surface particles must survive precise review without hiding neighbors."""
import cv2
import numpy as np
import pytest

from capi_config import CAPIConfig
from capi_dust_review import (
    confirmed_roi_surface_mask, recover_small_particle_cores,
    small_particle_core_mask, surface_removed_images,
)
from capi_inference import CAPIInferencer
from capi_server import _iter_qjpg_defect_records, results_to_db_data


def _detector():
    worker = CAPIInferencer.__new__(CAPIInferencer)
    worker.config = CAPIConfig(
        dust_brightness_threshold=20, dust_threshold_floor=20,
        dust_area_min=50, dust_area_max=100000, dust_extension=3,
        dust_detect_dark_particles=False, dust_detect_bubbles_enabled=False,
        dust_bright_rescue_threshold=0,
    )
    return worker


def test_precise_detector_keeps_only_the_native_core_of_an_accepted_small_particle():
    worker = _detector()
    omit = np.full((128, 128), 17, np.uint8)
    cv2.circle(omit, (40, 50), 2, 200, -1)
    before = omit.copy()
    config_minimum = worker.config.dust_area_min

    found, candidate, _, _ = worker.check_dust_or_scratch_feature(omit)
    precise_found, precise, _, detail = worker.check_dust_or_scratch_feature(
        omit, extension_override=0, preserve_small_cores=True)

    assert found and precise_found
    assert 0 < np.count_nonzero(precise) < config_minimum <= np.count_nonzero(candidate)
    assert not np.any((precise > 0) & (candidate == 0))
    assert "SmallCore:1" in detail
    assert worker.config.dust_area_min == config_minimum
    np.testing.assert_array_equal(omit, before)
    assert precise[50, 45] == 0 and candidate[50, 45] > 0


@pytest.mark.parametrize("extension,minimum,maximum", [(0, 50, 1000), (3, 500, 1000), (3, 50, 60)])
def test_unaccepted_small_components_and_single_pixel_noise_are_not_rescued(extension, minimum, maximum):
    binary = np.zeros((96, 96), np.uint8)
    cv2.circle(binary, (40, 40), 2, 255, -1)
    binary[70, 70] = 255
    rescued = recover_small_particle_cores(binary, minimum, maximum, extension)
    assert not np.any(rescued)


def test_native_recovery_never_fills_the_candidate_halo_or_unrelated_noise():
    binary = np.zeros((96, 96), np.uint8)
    cv2.circle(binary, (40, 40), 2, 255, -1)
    binary[70, 70] = 255
    rescued = recover_small_particle_cores(binary, 50, 1000, 3)
    assert np.count_nonzero(rescued) == 13
    assert not np.any((rescued > 0) & (binary == 0))
    assert rescued[70, 70] == 0


def test_precise_core_recovery_requires_acceptance_by_the_configured_extension():
    worker = _detector()
    omit = np.full((128, 128), 17, np.uint8)
    cv2.circle(omit, (40, 50), 2, 200, -1)
    worker.config.dust_extension = 0
    assert not worker.check_dust_or_scratch_feature(omit, extension_override=0, preserve_small_cores=True)[0]
    worker.config.dust_extension = 3
    worker.config.dust_area_min = 500
    assert not worker.check_dust_or_scratch_feature(omit, extension_override=0, preserve_small_cores=True)[0]


def test_core_recovery_does_not_change_unguarded_binary_or_non_roi_workflows():
    worker = _detector()
    omit = np.full((128, 128), 17, np.uint8)
    cv2.circle(omit, (40, 50), 2, 200, -1)
    assert not worker.check_dust_or_scratch_feature(omit, extension_override=0)[0]


def test_small_raw_feature_boundary_can_match_but_is_never_added_to_the_edit():
    raw = np.full((128, 128), 100, np.uint8)
    cv2.circle(raw, (40, 50), 2, 80, -1)
    native = np.zeros_like(raw)
    cv2.circle(native, (40, 50), 2, 255, -1)
    roi = np.zeros_like(raw)
    roi[16:112, 16:112] = 1
    support = np.where(roi, .42, 0).astype(np.float32)

    assert confirmed_roi_surface_mask(raw, native, roi, support)[0] is None
    edit, features, reason = confirmed_roi_surface_mask(
        raw, native, roi, support, small_core_mask=small_particle_core_mask(native, 50),
    )

    assert reason == "confirmed_surface"
    assert any(feature["dust_reason"] == "small_core_boundary" for feature in features)
    assert np.count_nonzero(edit) == 13
    assert not np.any((edit > 0) & (native == 0))
    for _, probe in surface_removed_images(raw, edit):
        np.testing.assert_array_equal(probe[native == 0], raw[native == 0])


def _small_case(monkeypatch, defect=None):
    from test_aoi_dust_residual import _case
    worker, tile, result, raw, dust, support, probes = _case(monkeypatch)
    worker.config.dust_area_min = 50
    worker.config.dust_extension = 3
    worker.config.dust_two_stage_enabled = True
    raw.fill(100)
    cv2.circle(raw, (40, 50), 2, 80, -1)
    dust.fill(0)
    cv2.circle(dust, (40, 50), 2, 255, -1)
    target = None
    if defect is not None:
        sign, contrast, distance = defect
        target = (40 + distance, 50)
        cv2.circle(raw, target, 2, 100 + sign * contrast, -1)
    tile.original_image = raw.copy()
    tile.image = raw.copy()
    # Preprocessing may produce a bright ring even though the raw image has
    # only one surface particle. It is not itself sufficient to release NG.
    cv2.circle(tile.image, (40, 50), 9, 115, 2)
    tile.is_suspected_dust_or_scratch = False
    tile.dust_two_stage_dust_mask = dust.copy()
    tile.dust_two_stage_features = [dict(
        abs_pos=(40, 50), type="bright", area=100, is_dust=False,
        dust_ratio=0., dust_reason="clean",
    )]

    def predict(probe, *, inferencer, threshold, **kwargs):
        np.testing.assert_array_equal(probe.image[dust == 0], tile.image[dust == 0])
        probes.append(probe.image.copy())
        value = .42 if target is not None and probe.image[target[1], target[0]] != 100 else .1
        amap = np.full(raw.shape, .05, np.float32)
        px, py = target or (40, 50)
        amap[py, px] = value
        return value, amap

    monkeypatch.setattr(worker, "predict_tile", predict)
    return worker, tile, result, raw, dust, support, probes


@pytest.mark.parametrize("defect", [None, (-1, 2, 9), (1, 2, 9), (-1, 10, 9), (1, 10, 9),
                                  (-1, 2, 48), (1, 2, 48), (-1, 10, 48), (1, 10, 48)])
def test_small_core_review_releases_dust_but_preserves_nearby_and_weak_defects(monkeypatch, defect):
    worker, tile, result, raw, dust, support, probes = _small_case(monkeypatch, defect)
    image_before = tile.image.copy()
    note = worker._confirm_aoi_dust_residual(result, tile, support, raw, raw)

    assert tile.is_suspected_dust_or_scratch is (defect is None)
    assert "ROI_RESIDUAL" in note
    snapshot = tile.decision_context["aoi_dust_residual"]
    assert snapshot["small_core_review"] is True
    assert snapshot["status"] == ("clear" if defect is None else "remaining")
    assert results_to_db_data([result], {})[0]["is_ng"] == int(defect is not None)
    reported = _iter_qjpg_defect_records([result], (128, 128), worker.config)
    assert any(row.startswith("PCDK2") for row in reported) is (defect is not None)
    np.testing.assert_array_equal(tile.image, image_before)
    if defect is None:
        assert len(probes) == 2
        feature = tile.dust_two_stage_features[0]
        assert feature["is_dust"] and feature["residual_confirmed"]
        assert feature["pre_residual_is_dust"] is False
        assert feature["dust_reason"] == "small_core_residual"


@pytest.mark.parametrize("failure", ["missing_core", "large_only", "real_first_stage", "model_disagrees"])
def test_small_core_exception_cannot_clear_missing_evidence_or_model_disagreement(monkeypatch, failure):
    worker, tile, result, raw, dust, support, probes = _small_case(monkeypatch)
    if failure == "missing_core":
        tile.dust_two_stage_dust_mask = np.zeros_like(dust)
    elif failure == "large_only":
        tile.dust_two_stage_dust_mask = np.full_like(dust, 255)
    elif failure == "real_first_stage":
        tile.dust_region_details[0]["is_dust"] = False
    else:
        values = iter([.1, .42])
        monkeypatch.setattr(worker, "predict_tile", lambda *a, **kw: (
            (value := next(values)), np.full((64, 64), value, np.float32)))
    worker._confirm_aoi_dust_residual(result, tile, support, raw, raw)
    assert not tile.is_suspected_dust_or_scratch
    assert tile.dust_two_stage_features[0]["is_dust"] is False


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("defect", [None, (-1, 2, 9), (1, 10, 48)])
def test_production_paths_reconcile_small_dust_only_after_full_roi_model_checks(monkeypatch, tmp_path, version, defect):
    worker, tile, result, raw, dust, _, probes = _small_case(monkeypatch, defect)
    omit = np.full_like(raw, 17)
    cv2.circle(omit, (40, 50), 2, 200, -1)
    worker.config.dust_brightness_threshold = worker.config.dust_threshold_floor = 20
    worker.config.dust_detect_bubbles_enabled = worker.config.dust_detect_dark_particles = False
    monkeypatch.setattr(worker, "_check_dust_or_scratch_feature_with_context", lambda *a, **kw:
        worker.check_dust_or_scratch_feature(omit, extension_override=kw.get("extension_override"),
                                           preserve_small_cores=kw.get("preserve_small_cores", False)))
    yy, xx = np.mgrid[:128, :128]
    full_map = np.exp(-((xx - 40) ** 2 + (yy - 50) ** 2) / (2 * 6. ** 2)).astype(np.float32)
    source = np.where((tile.aoi_roi_mask > 0) & (full_map >= .34), full_map, 0)
    tile.aoi_roi_full_map = full_map
    tile.dust_two_stage_dust_mask = tile.dust_two_stage_features = None
    result.raw_bounds = (0, 0, 128, 128)
    result.anomaly_tiles = [(tile, 1., source)]
    outcomes = []
    actual_two_stage = worker.check_dust_two_stage

    def capture_two_stage(*args, **kwargs):
        outcome = actual_two_stage(*args, **kwargs)
        outcomes.append(outcome[0])
        return outcome

    monkeypatch.setattr(worker, "check_dust_two_stage", capture_two_stage)
    if version == "v1":
        from test_aoi_roi_dust_core_judgment import _run_v1
        _run_v1(worker, result, monkeypatch, tmp_path)
    else:
        worker._apply_omit_dust_postprocess([result], omit, False, "", cpu_workers=1)

    assert "PER_REGION: 0real+1dust" in tile.dust_detail_text
    assert outcomes == [True]  # The actual legacy halo check initially keeps NG.
    assert "ROI_RESIDUAL" in tile.dust_detail_text
    assert tile.is_suspected_dust_or_scratch is (defect is None)
    assert results_to_db_data([result], {})[0]["is_ng"] == int(defect is not None)
    if defect is None:
        assert len(probes) == 2
        assert all(feature["dust_reason"] == "small_core_residual" for feature in tile.dust_two_stage_features)


def test_feature_diagnostic_explains_residual_verification(monkeypatch):
    from capi_heatmap import build_feature_zoom_panels
    worker, tile, result, raw, dust, support, _ = _small_case(monkeypatch)
    worker._confirm_aoi_dust_residual(result, tile, support, raw, raw)
    texts = []
    original_put_text = cv2.putText

    def capture_text(image, text, *args, **kwargs):
        texts.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_text)
    panels = build_feature_zoom_panels(
        tile.dust_heatmap_binary, dust, tile.dust_two_stage_features,
        128, 128, final_label="DUST_OK", roi_mask=tile.aoi_roi_mask,
    )
    assert panels
    assert "Feature: DUST-VERIFIED" in texts
    assert "Verified: raw core + whole ROI" in texts
