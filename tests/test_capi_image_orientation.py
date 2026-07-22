import cv2
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from capi_config import CAPIConfig
from capi_image_orientation import apply_detection_orientation
from capi_preprocess import PreprocessConfig, preprocess_panel_image
from capi_web import CAPIWebHandler


def test_inference_rotation_defaults_off_and_is_configurable():
    config = CAPIConfig()
    assert config.inference_rotate_180_enabled is False

    config.apply_db_overrides([{
        "param_name": "inference_rotate_180_enabled",
        "decoded_value": True,
    }])

    assert config.inference_rotate_180_enabled is True
    assert config.to_dict()["inference_rotate_180_enabled"] is True


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


def test_debug_image_reader_uses_runtime_rotation_setting(tmp_path):
    image_path = tmp_path / "W0F00000.png"
    image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    handler = object.__new__(CAPIWebHandler)
    handler.inferencer = SimpleNamespace(
        config=CAPIConfig(inference_rotate_180_enabled=True),
    )

    actual = handler._read_inference_image(image_path, cv2.IMREAD_GRAYSCALE)

    np.testing.assert_array_equal(actual, cv2.rotate(image, cv2.ROTATE_180))


def test_settings_page_exposes_rotation_under_inference_settings():
    template = (Path(__file__).resolve().parent.parent / "templates" / "settings.html").read_text(
        encoding="utf-8"
    )

    assert "🔍 推論設定" in template
    assert "inference_rotate_180_enabled" in template
    assert "輸入影像處理" in template
