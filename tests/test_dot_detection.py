import cv2
import numpy as np
from pathlib import Path

from capi_web import _detect_dot_components


def test_detects_black_dot_without_unit_calibration():
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    cv2.circle(image, (48, 48), 4, (70, 70, 70), -1)

    result = _detect_dot_components(
        image,
        polarity="black",
        diff_threshold=8,
        background_kernel=31,
        min_area=2,
        max_area=1000,
        morph_open=1,
        size_metric="bbox_max",
        unit_per_px=0.0,
        defect_threshold=0.3,
    )

    assert result["calibrated"] is False
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["size_px"] >= 8
    assert result["candidates"][0]["size_units"] is None
    assert result["candidates"][0]["is_defect"] is False


def test_detects_white_dot_and_applies_physical_threshold():
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    cv2.circle(image, (48, 48), 4, (210, 210, 210), -1)

    result = _detect_dot_components(
        image,
        polarity="white",
        diff_threshold=8,
        background_kernel=31,
        min_area=2,
        max_area=1000,
        morph_open=1,
        size_metric="bbox_max",
        unit_per_px=0.05,
        defect_threshold=0.3,
    )

    assert result["calibrated"] is True
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["size_units"] >= 0.3
    assert result["candidates"][0]["is_defect"] is True


def test_bundled_black_sample_default_opening_filters_background_texture():
    image_path = Path("templates/imgs/black/3.png")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None

    result = _detect_dot_components(
        image,
        polarity="black",
        diff_threshold=8,
        background_kernel=31,
        min_area=2,
        max_area=5000,
        morph_open=3,
        size_metric="bbox_max",
        unit_per_px=0.0,
        defect_threshold=0.3,
    )

    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["size_px"] <= 10
