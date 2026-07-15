import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import capi_mark_detector as mark_detector
from capi_mark_detector import _remove_tiny_components, detect_panel_mark


_ROOT = Path(__file__).resolve().parent.parent

_PATTERNS = {
    "B": ("01110", "11111", "11010", "11110", "11011", "11111", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "J": ("01111", "01111", "00110", "00110", "00110", "11110", "11100"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "r": ("11111", "10000", "10000", "11110", "10100", "10010", "10001"),
}


def _draw_mark(image, text, x, y, cell=10, gap=8, radius=3, row_shear=0.0):
    cursor_x = x
    for char in text:
        for row_idx, row in enumerate(_PATTERNS[char]):
            for col_idx, value in enumerate(row):
                if value == "1":
                    cv2.circle(
                        image,
                        (
                            cursor_x + col_idx * cell,
                            y + row_idx * cell + int(round(row_shear * col_idx * cell)),
                        ),
                        radius,
                        45,
                        -1,
                    )
        cursor_x += 5 * cell + gap


def test_detect_panel_mark_top_right():
    image = np.full((768, 1024), 160, dtype=np.uint8)
    _draw_mark(image, "EJ", 790, 150)

    result = detect_panel_mark(image)

    assert result["found"] is True
    assert result["text"] == "EJ"
    assert result["roi"] == "top_right"
    assert result["bbox"]["x"] >= 780
    assert result["bbox"]["y"] >= 140


def test_detect_panel_mark_top_right_bo_low_contrast():
    image = np.full((768, 1024), 120, dtype=np.uint8)
    _draw_mark(image, "BO", 790, 150, cell=10, gap=8, radius=3)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    result = detect_panel_mark(image)

    assert result["found"] is True
    assert result["text"] == "BO"
    assert result["roi"] == "top_right"
    assert result["orientation"] == "normal"


def test_detect_panel_mark_bottom_left():
    image = np.full((768, 1024), 160, dtype=np.uint8)
    _draw_mark(image, "EJ", 80, 520)

    result = detect_panel_mark(image)

    assert result["found"] is True
    assert result["text"] == "EJ"
    assert result["roi"] == "bottom_left"
    assert result["bbox"]["x"] >= 70
    assert result["bbox"]["y"] >= 510


def test_detect_panel_mark_bottom_left_rotated_180_reads_canonical_text():
    image = np.full((768, 1024), 160, dtype=np.uint8)
    _draw_mark(image, "EJ", 790, 150)
    rotated = cv2.rotate(image, cv2.ROTATE_180)

    result = detect_panel_mark(rotated)

    assert result["found"] is True
    assert result["text"] == "EJ"
    assert result["roi"] == "bottom_left"
    assert result["orientation"] == "rot180"


@pytest.mark.parametrize("text", ["F5", "P5", "R5", "T5"])
def test_detect_panel_mark_keeps_similar_first_chars_distinct(text):
    image = np.full((768, 1024), 160, dtype=np.uint8)
    _draw_mark(image, text, 790, 150)

    result = detect_panel_mark(image)

    assert result["found"] is True
    assert result["text"] == text


def test_detect_panel_mark_uses_diagonal_leg_to_distinguish_r_from_f():
    image = np.full((768, 1024), 160, dtype=np.uint8)
    _draw_mark(image, "r5", 790, 150)

    result = detect_panel_mark(image)

    assert result["found"] is True
    assert result["text"] == "R5"


def test_detect_panel_mark_deskews_slanted_dot_rows():
    image = np.full((768, 1024), 160, dtype=np.uint8)
    _draw_mark(image, "R5", 790, 170, row_shear=-0.25)

    result = detect_panel_mark(image)

    assert result["found"] is True
    assert result["text"] == "R5"


def test_recognize_char_uses_five_structure_when_j_and_five_are_tied(monkeypatch):
    densities = np.array(
        [
            [0.000, 0.315, 0.315, 0.259, 0.622],
            [0.032, 0.416, 0.279, 0.273, 0.113],
            [0.070, 0.448, 0.490, 0.252, 0.167],
            [0.110, 0.513, 0.331, 0.506, 0.292],
            [0.000, 0.000, 0.000, 0.413, 0.269],
            [0.000, 0.253, 0.318, 0.435, 0.161],
            [0.448, 0.325, 0.123, 0.195, 0.000],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(mark_detector, "_grid_densities", lambda _mask: densities)

    result = mark_detector._recognize_char(
        np.zeros((1, 1), dtype=np.uint8),
        mark_detector._SECOND_CHARS,
        char_index=1,
    )

    assert result["char"] == "5"


def test_remove_tiny_components_drops_isolated_noise():
    mask = np.zeros((40, 80), dtype=np.uint8)
    cv2.rectangle(mask, (5, 5), (14, 14), 255, -1)
    cv2.rectangle(mask, (30, 5), (38, 13), 255, -1)
    cv2.rectangle(mask, (60, 25), (62, 27), 255, -1)

    cleaned = _remove_tiny_components(mask)

    assert cleaned[8, 8] == 255
    assert cleaned[8, 34] == 255
    assert cleaned[26, 61] == 0


@pytest.mark.parametrize(
    ("filename", "expected_text", "expected_roi", "expected_orientation"),
    [
        ("W0F00000_080306.tif", "K5", "top_right", "normal"),
        ("W0F00000_140513.tif", "AS", "bottom_left", "rot180"),
        ("W0F00000_152329.tif", "0T", "top_right", "normal"),
        ("W0F00000_174608.tif", "EJ", "bottom_left", "rot180"),
        ("W0F00000_182349.tif", "K1", "top_right", "normal"),
    ],
)
def test_detect_panel_mark_real_regressions(filename, expected_text, expected_roi, expected_orientation):
    image_path = _ROOT / "mark" / filename
    if not image_path.exists():
        pytest.skip(f"real mark fixture not available: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    assert image is not None
    result = detect_panel_mark(image)

    assert result["found"] is True
    assert result["text"] == expected_text
    assert result["roi"] == expected_roi
    assert result["orientation"] == expected_orientation
