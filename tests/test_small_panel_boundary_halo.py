"""Small-product dim-band recovery must not promote external light halos."""
from pathlib import Path

import cv2
import numpy as np

from capi_preprocess import (
    PreprocessConfig,
    detect_panel_boundary,
    detect_panel_geometry,
    detect_panel_polygon,
    panel_boundary_config_for_station,
    preprocess_panel_folder,
)


def _slanted_panel_with_halo():
    corners = np.array([[250, 200], [1250, 220], [1240, 850], [240, 830]], np.float32)
    mask = np.zeros((1100, 1500), np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
    distance = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
    image = (8 + 14 * np.clip(1 - distance / 220, 0, 1)).astype(np.uint8)
    image[mask > 0] = 100
    return cv2.GaussianBlur(image, (5, 5), 1), corners


def _config():
    return PreprocessConfig(
        product_resolution=(1366, 768), generate_grid_tiles=False,
        **panel_boundary_config_for_station("capi"),
    )


def test_gradual_exterior_halo_does_not_expand_small_panel_and_retains_tilt(tmp_path):
    image, expected = _slanted_panel_with_halo()
    original = image.copy()
    config = _config()
    for detect in (detect_panel_polygon, detect_panel_boundary, detect_panel_geometry):
        bbox, polygon = detect(image, config)
        assert bbox is not None
        np.testing.assert_allclose(polygon, expected, atol=3)
        # Both side slopes remain real geometry instead of median-x lines.
        assert polygon[0, 0] - polygon[3, 0] > 7
        assert polygon[1, 0] - polygon[2, 0] > 7
    np.testing.assert_array_equal(image, original)

    assert cv2.imwrite(str(tmp_path / "W0F00000_001.png"), image)
    shared = preprocess_panel_folder(tmp_path, config)["W0F00000"]
    np.testing.assert_allclose(shared.panel_polygon, expected, atol=3)


def test_rejecting_halo_still_recovers_distinct_dim_top_and_side_bands():
    image = np.full((1000, 1500), 18, np.uint8)
    image[180:840, 210:1300] = 28
    image[230:840, 260:1250] = 210
    image = cv2.GaussianBlur(image, (5, 5), 1)
    _, polygon = detect_panel_polygon(image, _config())
    expected = np.array([[210, 180], [1299, 180], [1299, 839], [210, 839]])
    np.testing.assert_allclose(polygon, expected, atol=5)


def test_clean_near_vertical_panel_does_not_acquire_false_tilt():
    image = np.full((1000, 1500), 18, np.uint8)
    image[180:840, 210:1300] = 180
    # Local missing edge pixels should retain the established stabilization.
    image[200:250, 210:300] = 18
    image[200:250, 1210:1300] = 18
    _, polygon = detect_panel_polygon(image, _config())
    assert abs(polygon[0, 0] - polygon[3, 0]) < 1
    assert abs(polygon[1, 0] - polygon[2, 0]) < 1


def test_reported_ay_sample_excludes_halo_when_available():
    """Keep the optional production repro out of the portable fixture payload."""
    import pytest

    sample = Path(__file__).resolve().parents[1] / "marks" / "W0F00000_142426.tif"
    if not sample.exists():
        pytest.skip("local AY sample is unavailable")
    image = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
    _, polygon = detect_panel_geometry(image, _config(), source_name=sample.name)
    assert polygon is not None
    # Measured visible edges in the supplied raw image, before inward offset.
    expected = np.array([[1176, 731], [5158, 770], [5143, 3005], [1154, 2976]])
    np.testing.assert_allclose(polygon, expected, atol=8)
    assert cv2.pointPolygonTest(polygon, (1100., 1500.), False) < 0
    assert cv2.pointPolygonTest(polygon, (3000., 650.), False) < 0
    assert cv2.pointPolygonTest(polygon, (4900., 950.), False) > 0
