import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_mark_detector import detect_panel_mark


_PATTERNS = {
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "J": ("01111", "01111", "00110", "00110", "00110", "11110", "11100"),
}


def _draw_mark(image, text, x, y, cell=10, gap=8, radius=3):
    cursor_x = x
    for char in text:
        for row_idx, row in enumerate(_PATTERNS[char]):
            for col_idx, value in enumerate(row):
                if value == "1":
                    cv2.circle(
                        image,
                        (cursor_x + col_idx * cell, y + row_idx * cell),
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
