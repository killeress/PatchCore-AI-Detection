"""Regression coverage for slanted MARK glyphs and detached underlines."""
import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

from capi_mark_detector import _PATTERNS, _separate_mark_underlines, detect_panel_mark
from capi_mark_shadow import build_mark_shadow_payload


def _set_min_area_rect_convention(monkeypatch, convention):
    """Exercise the equivalent rectangle angles used before/after OpenCV 4.13."""
    original = cv2.minAreaRect

    def min_area_rect(points):
        center, (width, height), angle = original(points)
        if convention == "negative" and angle >= 0:
            return center, (height, width), angle - 90
        if convention == "positive" and angle < 0:
            return center, (height, width), angle + 90
        return center, (width, height), angle

    monkeypatch.setattr(cv2, "minAreaRect", min_area_rect)


@pytest.fixture(params=["positive", "negative"])
def min_area_rect_convention(monkeypatch, request):
    _set_min_area_rect_convention(monkeypatch, request.param)


def _synthetic_mark(angle=0, underlined=True):
    local = np.full((240, 240), 160, np.uint8)
    for index, char in enumerate("AY"):
        for row, pattern in enumerate(_PATTERNS[char][0]):
            for column, on in enumerate(pattern):
                if on == "1":
                    cv2.circle(local, (65 + index * 48 + column * 8, 60 + row * 8), 3, 45, -1)
    centers = []
    if underlined:
        for x in (65, 113):
            cv2.line(local, (x, 140), (x + 32, 140), 45, 4)
            centers.append([x + 16, 140])
    transform = cv2.getRotationMatrix2D((120, 120), angle, 1)
    local = cv2.warpAffine(local, transform, (240, 240), borderValue=160)
    image = np.full((768, 1024), 160, np.uint8)
    image[70:310, 750:990] = local
    expected = (cv2.transform(np.float32([centers]), transform)[0] + [750, 70]
                if centers else [])
    return image, expected


def _reported_ay_path():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "W0F00000_142426.tif", root / "marks" / "W0F00000_142426.tif"):
        if path.is_file():
            return path
    pytest.skip("local AY sample is unavailable")


@pytest.mark.parametrize("angle", [0, -42, 42, 138])
def test_underlined_mark_rectifies_and_masks_both_lines(angle, min_area_rect_convention):
    image, centers = _synthetic_mark(angle)
    result = detect_panel_mark(image)
    assert result["found"]
    assert result["text"] == "AY"
    assert len(result["underline_bboxes"]) == 2
    for x, y in centers:
        assert any(box["x"] <= x <= box["x"] + box["width"]
                   and box["y"] <= y <= box["y"] + box["height"]
                   for box in result["underline_bboxes"])
    payload = build_mark_shadow_payload(image, result, "synthetic.png")
    assert payload["crop_rectified"]
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(payload["image_png_base64"]), np.uint8), 0)
    assert crop.shape[1] > crop.shape[0]


def test_plain_upright_mark_keeps_legacy_crop_and_has_no_underline(min_area_rect_convention):
    image, _ = _synthetic_mark(underlined=False)
    result = detect_panel_mark(image)
    assert result["found"] and result["text"] == "AY"
    assert result["deskew_angle_deg"] == 0
    assert result["ocr_quad"] is None
    assert result["underline_bboxes"] == []


@pytest.mark.parametrize("angle", [-42, 42, 138])
def test_slanted_mark_without_underlines_still_rectifies(angle, min_area_rect_convention):
    image, _ = _synthetic_mark(angle, underlined=False)
    result = detect_panel_mark(image)
    assert result["found"] and result["text"] == "AY"
    assert 35 <= abs(result["deskew_angle_deg"]) <= 50
    assert result["underline_bboxes"] == []


def test_single_continuous_underline_is_excluded(min_area_rect_convention):
    image, _ = _synthetic_mark(underlined=False)
    cv2.line(image, (815, 210), (895, 210), 45, 4)
    result = detect_panel_mark(image)
    assert result["found"] and result["text"] == "AY"
    assert len(result["underline_bboxes"]) == 1
    line = result["underline_bboxes"][0]
    assert line["x"] <= 815 and line["x"] + line["width"] >= 895
    assert line["y"] <= 210 <= line["y"] + line["height"]


@pytest.mark.parametrize("kind", ["far_horizontal", "side_horizontal", "vertical"])
def test_unrelated_stroke_is_not_a_mark_underline(kind):
    mask = np.zeros((140, 160), np.uint8)
    mask[20:60, 20:45] = 255
    mask[20:60, 65:90] = 255
    if kind == "far_horizontal":
        mask[95:98, 25:85] = 255
    elif kind == "side_horizontal":
        mask[70:73, 100:150] = 255
    else:
        mask[70:110, 50:53] = 255
    clean, lines = _separate_mark_underlines(mask)
    assert lines == []
    np.testing.assert_array_equal(clean, mask)


@pytest.mark.parametrize("rotate180", [False, True])
def test_reported_ay_sample_has_upright_glyph_quad_and_two_exclusion_lines(
    rotate180, min_area_rect_convention,
):
    path = _reported_ay_path()
    image = cv2.imread(str(path), 0)
    if rotate180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    result = detect_panel_mark(image)
    assert result["found"] and result["text"] == "AY"
    assert 38 <= abs(result["deskew_angle_deg"]) <= 48
    assert len(result["underline_bboxes"]) == 2
    centers = np.float32([[4872, 916], [4919, 963]])
    if rotate180:
        centers = [image.shape[1] - 1, image.shape[0] - 1] - centers
    for x, y in centers:
        assert any(box["x"] <= x < box["x"] + box["width"]
                   and box["y"] <= y < box["y"] + box["height"]
                   for box in result["underline_bboxes"])
    payload = build_mark_shadow_payload(image, result, path)
    assert payload["crop_rectified"] is True
    assert payload["crop_width"] > payload["crop_height"]
    assert result["orientation"] == ("normal" if rotate180 else "rot180")
