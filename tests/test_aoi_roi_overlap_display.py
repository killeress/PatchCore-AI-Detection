"""AOI diagnostics show the evaluated masks without changing decision evidence."""
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import capi_heatmap
from capi_heatmap import (
    HeatmapManager, build_aoi_roi_core_mask, build_aoi_roi_debug_panel,
    build_dust_iou_debug_image, build_two_stage_debug_image,
)


def _has_color(image, color):
    return np.count_nonzero(np.all(image == color, axis=-1)) > 50


def _assert_regular_background_with_roi_frame(debug, regular, roi):
    allowed = cv2.resize((roi > 0).astype(np.uint8), (256, 256),
                         interpolation=cv2.INTER_NEAREST)
    border = (cv2.dilate(allowed, np.ones((3, 3), np.uint8)) > 0) & (allowed == 0)
    for y, x in ((0, 0), (0, 256), (256, 0), (256, 256)):
        actual = debug[y:y + 256, x:x + 256]
        expected = regular[y:y + 256, x:x + 256]
        unchanged = ~border
        if y == 256 and x == 0:
            unchanged = unchanged.copy()
            unchanged[:25] = False  # The Top% caption gains the ROI suffix.
        np.testing.assert_array_equal(actual[unchanged], expected[unchanged])
        assert np.all(actual[border] == (255, 255, 0))


@pytest.mark.parametrize("corner", [False, True])
def test_roi_overlap_shows_exact_masks_and_ignores_outside_pixels(corner):
    # Heatmap, dust and ROI can have different resolutions in production.
    roi = np.zeros((128, 128), np.uint8)
    heat = np.zeros((32, 32), np.uint8)
    dust = np.zeros((512, 512, 3), np.uint8)
    offset = 0 if corner else 40
    roi[offset:offset + 48, offset:offset + 48] = 1
    heat[offset // 4 + 2:offset // 4 + 8, offset // 4 + 2:offset // 4 + 8] = 255
    dust[(offset + 20) * 4:(offset + 40) * 4,
         (offset + 20) * 4:(offset + 40) * 4] = 255
    image = np.full((256, 256), 90, np.uint8)
    amap = np.ones((64, 64), np.float32)
    copies = [values.copy() for values in (roi, heat, dust, image, amap)]

    panel = build_aoi_roi_debug_panel(image, amap, heat, dust, roi)

    assert panel.shape == (512, 512, 3)
    scoped_heat = cv2.resize(heat, (128, 128), interpolation=cv2.INTER_NEAREST)
    scoped_heat[roi == 0] = 0
    scoped_dust = cv2.resize(dust[:, :, 0], (128, 128), interpolation=cv2.INTER_NEAREST)
    scoped_dust[roi == 0] = 0
    regular = build_dust_iou_debug_image(
        image, amap, scoped_dust, scoped_heat, 0, 100, False,
    )
    _assert_regular_background_with_roi_frame(panel, regular, roi)
    for values, before in zip((roi, heat, dust, image, amap), copies):
        np.testing.assert_array_equal(values, before)

    # Out-of-scope candidates must not reappear in the enlarged comparison.
    outside_heat = heat.copy()
    outside_dust = dust.copy()
    outside_heat[25:, 25:] = 255
    outside_dust[400:, 400:] = 255
    changed = build_aoi_roi_debug_panel(image, amap, outside_heat, outside_dust, roi)
    np.testing.assert_array_equal(changed, panel)


@pytest.mark.parametrize("missing", ["heat", "dust", "roi", "empty_roi"])
def test_incomplete_roi_evidence_keeps_existing_debug_available(missing):
    heat = np.ones((32, 32), np.uint8)
    dust = np.ones((32, 32), np.uint8)
    roi = np.ones((32, 32), np.uint8)
    if missing == "heat":
        heat = None
    elif missing == "dust":
        dust = None
    elif missing == "roi":
        roi = None
    else:
        roi[:] = 0
    assert build_aoi_roi_debug_panel(np.zeros((32, 32), np.uint8), None,
                                     heat, dust, roi) is None


@pytest.mark.parametrize("enabled", [False, True])
def test_saved_roi_diagnostic_keeps_scores_and_coverage_and_full_tile(tmp_path, monkeypatch, enabled):
    texts = []
    put_text = cv2.putText

    def capture(image, text, *args, **kwargs):
        texts.append(text)
        return put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture)
    image = np.full((512, 512), 90, np.uint8)
    roi = np.zeros_like(image)
    roi[200:312, 200:312] = 1
    heat = np.zeros_like(image)
    heat[220:280, 220:280] = 255
    dust = np.zeros_like(image)
    dust[250:290, 250:290] = 255
    amap = heat.astype(np.float32) / 255
    region = dict(label_id=1, area=3600, dust_overlap=900, metric_denominator=3600,
                  coverage=0.25, is_dust=False, max_score=0.8, peak_yx=(240, 240),
                  peak_in_dust=False)
    cached_debug = np.full((512, 512, 3), 123, np.uint8)
    tile = SimpleNamespace(
        original_image=image, omit_crop_image=image, dust_mask=dust,
        dust_heatmap_binary=heat, dust_iou_debug_image=cached_debug,
        dust_region_details=[region], dust_region_max_cov=0.25, dust_heatmap_iou=0.25,
        dust_detail_text="PER_REGION: 1real+0dust -> REAL_NG",
        is_suspected_dust_or_scratch=False, score_threshold=0.34,
        aoi_roi_mask=roi, aoi_roi_full_map=amap,
        decision_context={"aoi_judgment_roi": dict(
            enabled=enabled, radius_product_px=25,
            display_top_percent=100,
            aoi_product_xy=[1025, 135], full_tile_score=0.9)},
    )

    path = HeatmapManager(tmp_path).save_tile_heatmap(
        tmp_path, "roi", 0, image, amap, 0.8, tile_info=tile,
    )
    saved = cv2.imread(path)
    top = 148 if enabled else 60
    debug = saved[top:top + 512, 5 * 512:6 * 512]
    if enabled:
        assert "COV Debug | ROI Top 100%" in texts
        regular = build_dust_iou_debug_image(
            image, amap, dust, heat, 0.25, 100, False,
            region_details=[region], original_image=image,
        )
        _assert_regular_background_with_roi_frame(debug, regular, roi)
        zoom = saved[top + 120:top + 480, 6 * 512:7 * 512]
        for color in ((147, 147, 147), (58, 147, 147), (58, 147, 58)):
            assert _has_color(zoom, color)
        np.testing.assert_array_equal(saved[top + 400, 6 * 512 + 450], [90, 90, 90])
        np.testing.assert_array_equal(saved[top + 400, 6 * 512 + 300], [255, 255, 0])
    else:
        np.testing.assert_array_equal(debug, cached_debug)
    assert ("Decision COV: 900/3600 = 0.2500" if enabled else "COV: 900/3600 = 0.2500") in texts
    if enabled:
        assert "Core COV: 900/3600 = 0.2500 (display)" in texts
    assert "Score: 0.8000" in texts
    assert tile.dust_region_details == [region]
    assert not tile.is_suspected_dust_or_scratch
    np.testing.assert_array_equal(cached_debug, np.full_like(cached_debug, 123))
    # The original, preprocessed, heatmap, OMIT and dust panels keep full-tile context.
    for index in (0, 1, 3):
        np.testing.assert_array_equal(saved[top + 400, index * 512 + 400], [90, 90, 90])


@pytest.mark.parametrize("percent, count", [(0.5, 18), (5, 180), (20, 720), (100, 3600)])
def test_top_percent_uses_only_roi_values_and_preserves_inputs(percent, count):
    values = np.full((100, 100), 1e6, np.float32)
    values[20:80, 20:80] = np.arange(1, 3601).reshape(60, 60)
    roi = np.zeros((100, 100), np.uint8)
    roi[20:80, 20:80] = 1
    candidates = np.full((100, 100), 255, np.uint8)
    originals = [array.copy() for array in (values, roi, candidates)]

    core = build_aoi_roi_core_mask(values, candidates, roi, percent)

    assert np.count_nonzero(core) == count
    assert not np.any(core[roi == 0])
    assert values[core > 0].min() == 3601 - count
    # Still show the same in-ROI core when out-of-ROI heat changes completely.
    changed = values.copy()
    changed[roi == 0] = 0
    np.testing.assert_array_equal(build_aoi_roi_core_mask(changed, candidates, roi, percent), core)
    for array, original in zip((values, roi, candidates), originals):
        np.testing.assert_array_equal(array, original)


def test_top_core_does_not_restore_excluded_candidates_or_nonpositive_heat():
    values = np.arange(100, dtype=np.float32).reshape(10, 10)
    roi = np.ones((10, 10), np.uint8)
    candidates = np.ones((10, 10), np.uint8)
    candidates[9, 9] = 0
    core = build_aoi_roi_core_mask(values, candidates, roi, 10)
    assert not core[9, 9]
    assert np.count_nonzero(core) == 9
    values[:] = 0
    values[0, 0] = -1
    assert not np.any(build_aoi_roi_core_mask(values, candidates, roi, 10))


@pytest.mark.parametrize("percent,snapshot", [(5, None), (10, None), (10, 0.5)])
def test_saved_core_uses_config_or_captured_percent_without_changing_judgment(
    tmp_path, monkeypatch, percent, snapshot,
):
    texts = []
    core_masks = []
    real_debug = capi_heatmap.build_aoi_roi_debug_panel
    real_text = cv2.putText

    def capture_debug(image, amap, core, *args, **kwargs):
        core_masks.append(core.copy())
        return real_debug(image, amap, core, *args, **kwargs)

    def capture_text(image, text, *args, **kwargs):
        texts.append(text)
        return real_text(image, text, *args, **kwargs)

    monkeypatch.setattr(capi_heatmap, "build_aoi_roi_debug_panel", capture_debug)
    monkeypatch.setattr(cv2, "putText", capture_text)
    image = np.full((100, 100), 90, np.uint8)
    values = np.full((100, 100), 2000, np.float32)
    values[40:60, 40:60] = 0.5 + np.arange(400).reshape(20, 20) / 1000
    roi = np.zeros_like(image)
    roi[40:60, 40:60] = 1
    candidates = roi * 255
    dust = np.zeros_like(image)
    dust[57:60, 50:60] = 255
    region = dict(label_id=1, area=400, dust_overlap=30, metric_denominator=400,
                  coverage=0.075, is_dust=False, peak_in_dust=False,
                  max_score=0.899, peak_yx=(59, 59))
    metadata = dict(enabled=True, radius_product_px=25, aoi_product_xy=[100, 100],
                    full_tile_score=0.9)
    if snapshot is not None:
        metadata["display_top_percent"] = snapshot
    tile = SimpleNamespace(
        original_image=image, omit_crop_image=image, dust_mask=dust,
        dust_heatmap_binary=candidates, dust_region_details=[region],
        dust_detail_text="PER_REGION: 1real+0dust -> REAL_NG",
        dust_heatmap_iou=1.0, is_suspected_dust_or_scratch=False, score_threshold=0.34,
        aoi_roi_mask=roi, aoi_roi_full_map=values,
        decision_context={"aoi_judgment_roi": metadata},
    )
    path = HeatmapManager(tmp_path).save_tile_heatmap(
        tmp_path, "core", 0, image, values, 0.899, tile_info=tile,
        dust_heatmap_top_percent=percent,
    )

    selected_percent = percent if snapshot is None else snapshot
    expected_count = int(400 * selected_percent / 100)
    assert np.count_nonzero(core_masks[0]) == expected_count
    assert f"Top {selected_percent:g}% (ROI)" in texts
    assert "Decision COV: 30/400 = 0.0750" in texts
    expected_overlap = min(10, expected_count) if selected_percent <= 5 else 20
    assert (f"Core COV: {expected_overlap}/{expected_count} = "
            f"{expected_overlap / expected_count:.4f} (display)") in texts
    assert not tile.is_suspected_dust_or_scratch
    assert tile.dust_region_details == [region]
    np.testing.assert_array_equal(tile.dust_heatmap_binary, candidates)
    assert not np.any(core_masks[0][roi == 0])
    assert cv2.imread(path) is not None


def test_roi_two_stage_uses_the_regular_panel_backgrounds(tmp_path):
    image = np.full((128, 128), 90, np.uint8)
    yy, xx = np.mgrid[:128, :128]
    amap = (0.1 + np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / 200)).astype(np.float32)
    roi = np.ones_like(image)
    candidates = np.ones_like(image) * 255
    core = build_aoi_roi_core_mask(amap, candidates, roi, 0.5)
    dust = np.zeros_like(image)
    dust[62:66, 62:66] = 255
    features = [{"abs_pos": (64, 64), "area": 4, "is_dust": True}]
    regular = build_two_stage_debug_image(image, amap, dust, features, True, top_percent=0.5)
    tile = SimpleNamespace(
        original_image=image, omit_crop_image=image, dust_mask=dust,
        dust_heatmap_binary=candidates, dust_two_stage_features=features,
        dust_two_stage_dust_mask=dust, dust_detail_text="TWO_STAGE: 0real+1dust -> DUST",
        is_suspected_dust_or_scratch=True, aoi_roi_mask=roi, aoi_roi_full_map=amap,
        width=128, height=128,
        decision_context={"aoi_judgment_roi": dict(enabled=True, radius_product_px=25,
            aoi_product_xy=[100, 100], full_tile_score=1.0, display_top_percent=0.5)},
    )
    path = HeatmapManager(tmp_path).save_tile_heatmap(
        tmp_path, "two_stage", 0, image, amap, 1.0, tile_info=tile,
    )
    saved = cv2.imread(path)
    debug = saved[148:660, 5 * 512:6 * 512]
    np.testing.assert_array_equal(debug[:256], regular[:256])
    np.testing.assert_array_equal(debug[281:], regular[281:])
    assert np.any(core)


def test_two_stage_roi_frame_preserves_core_pixels_at_the_boundary():
    image = np.full((128, 128), 90, np.uint8)
    amap = np.zeros_like(image, dtype=np.float32)
    amap[63:66, 63:66] = 0.9
    roi = np.zeros_like(image)
    roi[16:112, 16:112] = 1
    core = np.zeros_like(image)
    core[16, 64] = 255  # Candidate lies directly on the upper evaluated edge.
    dust = np.zeros_like(image)
    regular = build_two_stage_debug_image(
        image, amap, dust, [], False, top_percent=0.5, heatmap_binary=core,
    )
    debug = build_two_stage_debug_image(
        image, amap, dust, [], False, top_percent=0.5,
        heatmap_binary=core, evaluation_mask=roi,
    )
    _assert_regular_background_with_roi_frame(debug, regular, roi)
    # In the lower-left subpanel, the white candidate remains intact and the
    # cyan outline lies immediately outside its evaluated edge.
    np.testing.assert_array_equal(debug[256 + 32, 128], [255, 255, 255])
    np.testing.assert_array_equal(debug[256 + 31, 128], [255, 255, 0])
