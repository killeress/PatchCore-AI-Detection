from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import capi_heatmap
from capi_heatmap import HeatmapManager, build_region_zoom_panels


def _region(**changes):
    region = {
        "label_id": 1, "area": 21140, "dust_overlap": 958,
        "metric_denominator": 21140, "coverage": 958 / 21140,
        "is_dust": True, "peak_in_dust": False,
        "dust_sub_peak_rescue": False, "max_score": 0.8293,
        "peak_yx": (10, 10), "feature_reviewed": True,
        "feature_review_reason": "all_features_match_precise_dust",
        "feature_review_matched": 2, "feature_review_unmatched": 0,
    }
    return dict(region, **changes)


def _capture_text(monkeypatch):
    captured = []
    original = cv2.putText

    def capture(image, text, *args, **kwargs):
        captured.append(str(text))
        return original(image, text, *args, **kwargs)

    monkeypatch.setattr(capi_heatmap.cv2, "putText", capture)
    return captured


def test_reviewed_dust_displays_feature_reason_and_preserves_low_coverage(monkeypatch):
    texts = _capture_text(monkeypatch)
    region = _region()
    original = dict(region)
    panels = build_region_zoom_panels(
        np.ones((128, 128), np.uint8), np.zeros((128, 128), np.uint8),
        [region], iou_threshold=0.1,
    )

    assert len(panels) == 1
    assert "COV: 958/21140 = 0.0453" in texts
    assert "FEATURE_MATCH Dust:2 Real:0 -> DUST" in texts
    assert not any("HighCOV" in text or "PeakInDust" in text or "-> REAL_NG" in text
                   for text in texts)
    assert region == original


def test_reviewed_real_feature_controls_zoom_instead_of_dust_peak(monkeypatch):
    texts = _capture_text(monkeypatch)
    yy, xx = np.mgrid[:256, :256]
    original = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).astype(np.uint8)
    region = _region(
        is_dust=False, feature_review_reason="unmatched_features",
        feature_review_matched=1, feature_review_unmatched=1,
        real_feature_peak_yx=(90, 100),
    )
    panels = build_region_zoom_panels(
        np.zeros((128, 128), np.uint8), None, [region],
        tile_size=256, tile_image=original, iou_threshold=0.1,
    )

    # Map-space (90,100) becomes (180,200); the right edge clamps the crop to x=128.
    np.testing.assert_array_equal(panels[0][0][160, 128], [192, 196, 0])
    assert "FEATURE_MATCH Dust:1 Real:1 -> REAL_NG" in texts
    assert "COV: 958/21140 = 0.0453" in texts


@pytest.mark.parametrize("reviewed, expected", [
    (False, "COV:0.045<Thr:0.100 -> REAL_NG"),
    (True, "FEATURE_MATCH NoMatch Dust:0 Real:0 -> REAL_NG"),
])
def test_no_match_review_is_distinct_from_ordinary_coverage_failure(monkeypatch, reviewed, expected):
    texts = _capture_text(monkeypatch)
    region = _region(
        is_dust=False, feature_reviewed=reviewed,
        feature_review_reason="no_matched_features", feature_review_matched=0,
    )
    build_region_zoom_panels(np.ones((128, 128), np.uint8), None, [region], iou_threshold=0.1)
    assert expected in texts


def test_bomb_reason_keeps_priority_over_feature_review(monkeypatch):
    texts = _capture_text(monkeypatch)
    region = _region(is_dust=False, bomb_status="BOMB", bomb_excluded_px=100)
    panels = build_region_zoom_panels(
        np.ones((128, 128), np.uint8), None, [region], iou_threshold=0.1,
    )
    assert "BOMB excluded: 100 px" in texts
    assert not any("FEATURE_MATCH" in text for text in texts)
    assert " BOMB " in panels[0][1]


def test_tile_review_header_and_contours_do_not_use_two_stage_zoom(tmp_path, monkeypatch):
    texts = _capture_text(monkeypatch)
    drawn = []
    original_polylines = cv2.polylines

    def capture_contour(image, points, closed, color, *args, **kwargs):
        drawn.append((np.asarray(points[0]).copy(), color))
        return original_polylines(image, points, closed, color, *args, **kwargs)

    monkeypatch.setattr(capi_heatmap.cv2, "polylines", capture_contour)
    monkeypatch.setattr(capi_heatmap, "build_feature_zoom_panels",
                        lambda *args, **kwargs: pytest.fail("Feature review must keep region zooms"))
    image = np.full((128, 128), 80, np.uint8)
    region = _region()
    tile = SimpleNamespace(
        original_image=image, width=128, height=128, x=0, y=0,
        omit_crop_image=image, dust_mask=np.zeros_like(image),
        dust_heatmap_binary=np.ones_like(image), dust_heatmap_iou=958 / 21140,
        dust_region_max_cov=958 / 21140, dust_region_details=[region],
        dust_detail_text="FEATURE_MATCH 2dust+0real -> DUST",
        is_suspected_dust_or_scratch=True, dust_two_stage_features=None,
        dust_low_coverage_features=[{
            "abs_pos": (30, 40), "is_dust": True, "area": 20,
            "feature_contour": [[28, 38], [32, 38], [32, 42], [28, 42]],
        }],
    )
    path = HeatmapManager(tmp_path).save_tile_heatmap(
        tmp_path, "review", 1, image, np.full((128, 128), 0.8, np.float32),
        0.8293, tile_info=tile, iou_threshold=0.1,
    )

    assert Path(path).is_file()
    assert any("DUST (Filtered as OK) | FEATURE_MATCH Dust:2 Real:0" in text for text in texts)
    assert not any("RegionCOV:" in text for text in texts)
    assert "COV: 958/21140 = 0.0453" in texts
    assert "FM-D" in texts
    assert len(drawn) == 1
    np.testing.assert_array_equal(drawn[0][0], [[112, 152], [128, 152], [128, 168], [112, 168]])
    assert drawn[0][1] == (0, 200, 0)
    assert tile.dust_two_stage_features is None
