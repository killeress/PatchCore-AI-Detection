"""Original-image evidence must explain every reviewed low-coverage region."""
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_dust_review import build_aoi_roi_core_mask, review_low_coverage_regions


def _scene():
    image = np.full((256, 256), 100, dtype=np.uint8)
    precise = np.zeros_like(image)
    labels = np.zeros_like(image, dtype=np.int32)
    labels[32:224, 32:224] = 1
    anomaly = np.where(labels > 0, 0.55, 0.0).astype(np.float32)
    return image, precise, labels, anomaly


def _spot(image, center, *, contrast=-25, radius=4):
    cv2.circle(image, center, radius, 100 + contrast, thickness=-1)


def _dust(precise, center, radius=7):
    # The measured OMIT core can be larger than the dark spot in the lighting.
    cv2.circle(precise, center, radius, 255, thickness=-1)


def _regions(labels, precise, anomaly):
    regions = []
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        selected = labels == label_id
        area = int(np.count_nonzero(selected))
        overlap = int(np.count_nonzero(selected & (precise > 0)))
        ys, xs = np.where(selected)
        peak_index = int(np.argmax(anomaly[selected]))
        peak_yx = (int(ys[peak_index]), int(xs[peak_index]))
        regions.append({
            "label_id": int(label_id),
            "area": area,
            "dust_overlap": overlap,
            "metric_denominator": area,
            "coverage": overlap / area,
            "is_dust": False,
            "peak_in_dust": bool(precise[peak_yx] > 0),
            "max_score": float(np.max(anomaly[selected])),
            "peak_yx": peak_yx,
        })
    return regions


def _review(image, precise, labels, anomaly, *, evaluation_mask=None):
    regions = _regions(labels, precise, anomaly)
    updated, features = review_low_coverage_regions(
        image, precise, anomaly, labels, regions,
        coverage_threshold=0.1, evaluation_mask=evaluation_mask,
    )
    return regions, updated, features


@pytest.mark.parametrize("contrast,spot_type", [(-25, "dark"), (25, "bright")])
def test_low_coverage_matched_feature_is_filtered_without_rewriting_coverage(contrast, spot_type):
    image, precise, labels, anomaly = _scene()
    _spot(image, (90, 90), contrast=contrast)
    _dust(precise, (90, 90))
    anomaly[90, 90] = 1.0

    original, updated, features = _review(image, precise, labels, anomaly)

    assert original[0]["coverage"] < 0.1
    assert updated[0]["is_dust"] is True
    assert updated[0]["feature_reviewed"] is True
    assert updated[0]["feature_review_reason"]
    for key in ("label_id", "area", "dust_overlap", "metric_denominator", "coverage"):
        assert updated[0][key] == original[0][key]
    assert any(f["type"] == spot_type and f["is_dust"] for f in features)
    assert all(f["zone_id"] == 1 for f in features)
    assert all(f["feature_bbox"][2] > 0 and f["feature_contour"] for f in features)


@pytest.mark.parametrize("contrast", [-8, 8])
def test_weaker_independent_feature_in_same_hot_region_remains_real(contrast):
    image, precise, labels, anomaly = _scene()
    _spot(image, (75, 75))
    _dust(precise, (75, 75))
    anomaly[70:81, 70:81] = 1.0
    # This feature is far from the original peak but inside the same hot region.
    _spot(image, (185, 185), contrast=contrast, radius=5)

    _, updated, features = _review(image, precise, labels, anomaly)

    assert updated[0]["is_dust"] is False
    assert any(f["is_dust"] for f in features)
    assert any(
        not f["is_dust"] and abs(f["abs_pos"][0] - 185) <= 3
        and abs(f["abs_pos"][1] - 185) <= 3
        for f in features
    )


def test_unmatched_feature_near_precise_dust_is_not_accepted_as_halo():
    image, precise, labels, anomaly = _scene()
    _spot(image, (100, 110))
    _dust(precise, (100, 110))
    _spot(image, (114, 110), contrast=-12, radius=3)
    anomaly[105:116, 95:120] = 1.0

    _, updated, features = _review(image, precise, labels, anomaly)

    assert updated[0]["is_dust"] is False
    assert any(not f["is_dust"] for f in features)


def test_connected_small_unmatched_tail_cannot_hide_in_majority_dust_overlap():
    image, precise, labels, anomaly = _scene()
    _spot(image, (100, 110), radius=8)
    _dust(precise, (100, 110), radius=11)
    # The original spots have a one-pixel gap; smoothing connects their dark
    # responses into one component with about 88% overall dust overlap.
    _spot(image, (112, 110), radius=3)
    assert precise[110, 112] == 0

    _, updated, features = _review(image, precise, labels, anomaly)

    assert updated[0]["is_dust"] is False
    assert any(not feature["is_dust"] for feature in features)
    peak_y, peak_x = updated[0]["real_feature_peak_yx"]
    assert precise[peak_y, peak_x] == 0
    assert labels[peak_y, peak_x] == updated[0]["label_id"]
    assert peak_x > 110  # The independent tail, rather than the main dust center.
    for feature in features:
        if not feature["is_dust"]:
            real_x, real_y = feature["real_pos"]
            assert precise[real_y, real_x] == 0
            assert labels[real_y, real_x] == feature["zone_id"]


def test_partial_precise_dust_overlap_does_not_explain_whole_feature():
    image, precise, labels, anomaly = _scene()
    _spot(image, (100, 100), radius=5)
    _dust(precise, (100, 100))
    precise[:, 100:] = 0

    _, updated, features = _review(image, precise, labels, anomaly)

    assert updated[0]["is_dust"] is False
    assert any(not f["is_dust"] for f in features)


def test_dust_nearby_without_precise_overlap_does_not_filter_spot():
    image, precise, labels, anomaly = _scene()
    _spot(image, (100, 100))
    _dust(precise, (114, 100), radius=4)

    _, updated, features = _review(image, precise, labels, anomaly)

    assert updated[0]["is_dust"] is False
    assert any(not f["is_dust"] for f in features)


def test_no_original_feature_does_not_inherit_a_dust_verdict():
    image, precise, labels, anomaly = _scene()
    _dust(precise, (90, 90))
    anomaly[90, 90] = 1.0

    _, updated, features = _review(image, precise, labels, anomaly)

    assert updated[0]["is_dust"] is False
    assert updated[0]["feature_review_reason"]
    assert not features


@pytest.mark.parametrize("second_has_feature", [False, True])
def test_each_disconnected_hot_region_requires_its_own_evidence(second_has_feature):
    image, precise, labels, anomaly = _scene()
    labels[:] = 0
    labels[32:224, 32:120] = 1
    labels[32:224, 136:224] = 3
    anomaly[:] = np.where(labels > 0, 0.55, 0.0)
    _spot(image, (75, 100))
    _dust(precise, (75, 100))
    _dust(precise, (158, 70), radius=4)
    if second_has_feature:
        _spot(image, (180, 170), contrast=-8, radius=5)

    _, updated, features = _review(image, precise, labels, anomaly)

    by_label = {region["label_id"]: region for region in updated}
    assert by_label[1]["is_dust"] is True
    assert by_label[3]["is_dust"] is False
    if second_has_feature:
        assert any(f["zone_id"] == 3 and not f["is_dust"] for f in features)


def test_aoi_excludes_outside_feature_but_retains_inside_feature():
    image, precise, labels, anomaly = _scene()
    _spot(image, (75, 90))
    _dust(precise, (75, 90))
    _spot(image, (190, 180), contrast=-8, radius=5)
    allowed = np.zeros_like(image)
    allowed[32:224, 32:140] = 255

    _, outside_result, outside_features = _review(
        image, precise, labels, anomaly, evaluation_mask=allowed,
    )
    assert outside_result[0]["is_dust"] is True
    assert all(f["abs_pos"][0] < 140 for f in outside_features)

    allowed[32:224, 32:224] = 255
    _, inside_result, inside_features = _review(
        image, precise, labels, anomaly, evaluation_mask=allowed,
    )
    assert inside_result[0]["is_dust"] is False
    assert any(not f["is_dust"] for f in inside_features)


def test_clipped_aoi_feature_prevents_matching_dust_elsewhere_from_clearing_region():
    image, precise, labels, anomaly = _scene()
    _spot(image, (75, 90))
    _dust(precise, (75, 90))
    _spot(image, (140, 180), contrast=-12, radius=5)
    allowed = np.zeros_like(image)
    allowed[32:224, 32:140] = 255

    _, updated, features = _review(image, precise, labels, anomaly, evaluation_mask=allowed)

    assert updated[0]["is_dust"] is False
    peak_y, peak_x = updated[0]["real_feature_peak_yx"]
    assert allowed[peak_y, peak_x] > 0
    assert labels[peak_y, peak_x] == updated[0]["label_id"]
    assert any(not feature["is_dust"] and feature["dust_reason"] == "clipped"
               for feature in features)
    for feature in features:
        if not feature["is_dust"]:
            real_x, real_y = feature["real_pos"]
            assert allowed[real_y, real_x] > 0


def test_tile_border_feature_is_inconclusive_even_with_matching_dust():
    image, precise, labels, anomaly = _scene()
    labels[:] = 1
    anomaly[:] = 0.55
    _spot(image, (2, 120), radius=5)
    _dust(precise, (2, 120), radius=8)

    _, updated, _ = _review(image, precise, labels, anomaly)

    assert updated[0]["is_dust"] is False


def test_b03_sized_region_keeps_measured_heat_coverage_after_feature_review():
    image = np.full((512, 512), 100, dtype=np.uint8)
    precise = np.zeros_like(image)
    labels = np.zeros_like(image, dtype=np.int32)
    allowed = np.zeros_like(image)
    allowed[166:314, 162:309] = 255  # 148 x 147 = 21,756 AOI pixels.
    labels[allowed > 0] = 1
    labels[166:170, 162:309] = 0
    labels[170, 162:190] = 0  # Remove 616 pixels, leaving 21,140 connected pixels.
    anomaly = np.where(labels > 0, 0.6, 0.0).astype(np.float32)
    _spot(image, (240, 240), radius=5)
    _dust(precise, (240, 240), radius=12)

    original, updated, features = _review(
        image, precise, labels, anomaly, evaluation_mask=allowed,
    )

    assert original[0]["area"] == 21140
    assert np.count_nonzero(allowed) == 21756
    assert original[0]["coverage"] < 0.1
    assert updated[0]["coverage"] == original[0]["coverage"]
    assert updated[0]["is_dust"] is True
    assert any(f["type"] == "dark" and f["is_dust"] for f in features)


@pytest.mark.parametrize("invalid_input", ["missing_raw", "missing_dust", "dust_shape", "map_dimensions", "label_shape"])
def test_missing_or_misaligned_evidence_keeps_original_ng(invalid_input):
    image, precise, labels, anomaly = _scene()
    _spot(image, (90, 90))
    _dust(precise, (90, 90))
    regions = _regions(labels, precise, anomaly)
    if invalid_input == "missing_raw":
        image = None
    elif invalid_input == "missing_dust":
        precise = None
    elif invalid_input == "dust_shape":
        precise = precise[:128, :128]
    elif invalid_input == "map_dimensions":
        anomaly = anomaly[:, :, None]
    elif invalid_input == "label_shape":
        labels = labels[:128, :128]

    updated, features = review_low_coverage_regions(
        image, precise, anomaly, labels, regions, coverage_threshold=0.1,
    )

    assert updated[0]["is_dust"] is False
    assert updated[0]["feature_reviewed"] is False
    assert updated[0]["feature_review_reason"] == "missing_evidence"
    assert updated[0]["coverage"] == regions[0]["coverage"]
    assert features == []


def _displaced_core_scene():
    image = np.full((256, 256), 100, np.uint8)
    precise = np.zeros_like(image)
    roi = np.zeros_like(image)
    roi[32:224, 32:224] = 1
    _spot(image, (124, 132), contrast=-18, radius=18)
    _dust(precise, (124, 132), radius=23)
    yy, xx = np.mgrid[:256, :256]
    support = np.exp(-((xx - 124) ** 2 + (yy - 96) ** 2) / (2 * 24.0 ** 2)).astype(np.float32)
    support[(roi == 0) | (support < 0.34)] = 0
    core = build_aoi_roi_core_mask(support, support > 0, roi, 0.4)
    _, labels = cv2.connectedComponents(core, connectivity=8)
    anomaly = np.where(core > 0, support, 0)
    return image, precise, roi, support, labels, anomaly


@pytest.mark.parametrize("map_scale", [1, 2])
def test_displaced_top_point_four_core_matches_bubble_without_changing_coverage(map_scale):
    image, precise, roi, support, labels, anomaly = _displaced_core_scene()
    if map_scale == 2:
        support = cv2.resize(support, (128, 128), interpolation=cv2.INTER_AREA)
        core = build_aoi_roi_core_mask(support, support > 0, roi, 0.4)
        _, labels = cv2.connectedComponents(core, connectivity=8)
        anomaly = np.where(core > 0, support, 0)
    dust_in_map = cv2.resize(precise, (anomaly.shape[1], anomaly.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    regions = _regions(labels, dust_in_map, anomaly)
    before = [array.copy() for array in (image, precise, roi, support, labels, anomaly)]

    updated, features = review_low_coverage_regions(
        image, precise, anomaly, labels, regions, coverage_threshold=0.1,
        evaluation_mask=roi, support_anomaly_map=support,
    )

    assert len(regions) == 1
    assert regions[0]["dust_overlap"] == 0
    assert updated[0]["is_dust"] is True
    assert updated[0]["feature_review_scope"] == "half_peak_component"
    assert updated[0]["feature_review_scope_pixels"] > regions[0]["area"]
    assert features and all(feature["is_dust"] for feature in features)
    for key in ("area", "dust_overlap", "metric_denominator", "coverage", "peak_yx"):
        assert updated[0][key] == regions[0][key]
    assert regions[0]["is_dust"] is False
    for array, original in zip((image, precise, roi, support, labels, anomaly), before):
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("reason", [
    "missing_support", "invalid_support", "no_feature", "disconnected_dust",
    "independent_feature", "outside_roi", "clipped_feature",
])
def test_displaced_core_requires_connected_complete_matched_image_evidence(reason):
    image, precise, roi, support, labels, anomaly = _displaced_core_scene()
    regions = _regions(labels, precise, anomaly)
    if reason == "missing_support":
        support = None
    elif reason == "invalid_support":
        support = support[:128, :128]
    elif reason == "no_feature":
        image[:] = 100
    elif reason == "disconnected_dust":
        support[108:114] = 0  # There is no continuous response joining core and bubble.
    elif reason == "independent_feature":
        _spot(image, (145, 100), contrast=-10, radius=4)
    elif reason == "outside_roi":
        roi[108:] = 0
    elif reason == "clipped_feature":
        roi[135:] = 0

    updated, _ = review_low_coverage_regions(
        image, precise, anomaly, labels, regions, coverage_threshold=0.1,
        evaluation_mask=roi, support_anomaly_map=support,
    )

    assert updated[0]["is_dust"] is False
    assert updated[0]["coverage"] == 0


def test_shared_response_cannot_use_one_bubble_to_clear_another_hot_core():
    image, precise, roi, support, labels, anomaly = _displaced_core_scene()
    # Two separate strongest cores share a lower-temperature bridge. Only the
    # first has an actual bubble, so neither can borrow the other's evidence.
    support[92:101, 166:175] = 1
    support[92:101, 132:166] = 0.6
    labels[92:101, 166:175] = 2
    anomaly = np.where(labels > 0, support, 0)
    regions = _regions(labels, precise, anomaly)

    updated, _ = review_low_coverage_regions(
        image, precise, anomaly, labels, regions, coverage_threshold=0.1,
        evaluation_mask=roi, support_anomaly_map=support,
    )

    assert all(not region["is_dust"] for region in updated)
