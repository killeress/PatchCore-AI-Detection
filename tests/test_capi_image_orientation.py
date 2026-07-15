import cv2
import numpy as np

from capi_image_orientation import apply_detection_orientation, requires_detection_rotation
from capi_preprocess import PreprocessConfig, preprocess_panel_image


def test_requires_detection_rotation_only_for_exact_capi13():
    assert requires_detection_rotation("capi13") is True
    assert requires_detection_rotation("CAPI13") is True
    assert requires_detection_rotation("capi13.local") is False
    assert requires_detection_rotation("capi12") is False


def test_apply_detection_orientation_rotates_180_degrees():
    image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)

    rotated = apply_detection_orientation(image, rotate_180=True)

    np.testing.assert_array_equal(rotated, np.array([[6, 5, 4], [3, 2, 1]], dtype=np.uint8))
    np.testing.assert_array_equal(apply_detection_orientation(image, rotate_180=False), image)


def test_preprocess_panel_image_rotates_before_detection(tmp_path):
    image_path = tmp_path / "G0F00000_test.png"
    image = np.zeros((64, 96), dtype=np.uint8)
    image[4:20, 8:32] = 200
    assert cv2.imwrite(str(image_path), image)

    result = preprocess_panel_image(
        image_path,
        "G0F00000",
        PreprocessConfig(
            enable_panel_polygon=False,
            generate_grid_tiles=False,
            cache_processed_image=True,
            rotate_180=True,
        ),
    )

    np.testing.assert_array_equal(result.processed_image, cv2.rotate(image, cv2.ROTATE_180))
