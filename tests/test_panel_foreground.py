"""Exterior lamps must not enlarge panel coordinates or contaminate fitted edges."""
import cv2
import numpy as np
import pytest

from capi_inference import CAPIInferencer
from capi_preprocess import PreprocessConfig, detect_panel_polygon


def panel_image():
    image = np.zeros((1600, 2000), dtype=np.uint8)
    image[300:1300, 300:1700] = 180
    image[700:760, 900:960] = 0  # An internal dark defect must remain in the mask.
    return image


@pytest.mark.parametrize("center", [(1000, 150), (1000, 1450), (150, 800), (1850, 800)])
@pytest.mark.parametrize("radius", [24, 80])
def test_raw_bounds_ignore_exterior_lamps_on_every_side(center, radius):
    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    clean = panel_image()
    expected_bounds, expected_mask = inferencer._find_raw_object_bounds(clean)
    image = clean.copy()
    cv2.circle(image, center, radius, 255, -1)
    original = image.copy()

    bounds, mask = inferencer._find_raw_object_bounds(image)

    assert bounds == expected_bounds == (300, 300, 1700, 1300)
    np.testing.assert_array_equal(mask, expected_mask)
    assert mask[center[1], center[0]] == 0
    assert mask[730, 930] == 0
    assert mask[600, 600] == 255
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("resolution", [(1366, 768), (1920, 1200)])
@pytest.mark.parametrize("enable_polygon", [False, True])
def test_shared_detector_uses_only_panel_for_bounds_and_polygon(resolution, enable_polygon):
    clean = panel_image()
    config = PreprocessConfig(
        tile_size=256, product_resolution=resolution, enable_panel_polygon=enable_polygon,
    )
    expected_bounds, expected_polygon = detect_panel_polygon(clean, config)
    image = clean.copy()
    # These objects pass the old 1000-pixel threshold and overlap edge scan lines.
    cv2.rectangle(image, (800, 1400), (1000, 1510), 255, -1)
    cv2.rectangle(image, (100, 600), (220, 900), 255, -1)

    bounds, polygon = detect_panel_polygon(image, config)

    assert bounds == expected_bounds
    if enable_polygon:
        assert polygon is not None and expected_polygon is not None
        np.testing.assert_allclose(polygon, expected_polygon, atol=0.1)
    else:
        assert polygon is None


def test_empty_foreground_keeps_existing_no_detection_fallbacks():
    image = np.zeros((1600, 2000), dtype=np.uint8)
    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    bounds, mask = inferencer._find_raw_object_bounds(image)
    assert bounds == (0, 0, 2000, 1600)
    assert not mask.any()
    assert detect_panel_polygon(image, PreprocessConfig()) == (None, None)
