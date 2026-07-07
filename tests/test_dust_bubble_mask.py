from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer


def _make_inferencer(*, detect_bubbles: bool) -> CAPIInferencer:
    config = CAPIConfig()
    config.dust_area_min = 5
    config.dust_area_max = 100000
    config.dust_extension = 0
    config.dust_detect_dark_particles = False
    config.dust_detect_bubbles_enabled = detect_bubbles

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config
    return inferencer


def _low_contrast_bubble_image() -> np.ndarray:
    image = np.full((128, 128), 80, dtype=np.uint8)
    cv2.ellipse(image, (64, 64), (18, 24), 0, 0, 360, 72, -1)
    return cv2.GaussianBlur(image, (9, 9), 0)


def _low_contrast_bright_bubble_image() -> np.ndarray:
    image = np.full((128, 128), 60, dtype=np.uint8)
    cv2.ellipse(image, (64, 64), (18, 24), 0, 0, 360, 68, -1)
    return cv2.GaussianBlur(image, (9, 9), 0)


def _low_contrast_ring_bubble_image() -> np.ndarray:
    image = np.full((128, 128), 80, dtype=np.uint8)
    cv2.ellipse(image, (64, 64), (22, 28), 0, 0, 360, 90, 5)
    return cv2.GaussianBlur(image, (9, 9), 0)


def test_low_contrast_bubble_mask_is_opt_in():
    image = _low_contrast_bubble_image()

    inferencer = _make_inferencer(detect_bubbles=False)
    is_dust, mask, ratio, detail = inferencer.check_dust_or_scratch_feature(image)

    assert is_dust is False
    assert np.count_nonzero(mask) == 0
    assert ratio == 0.0
    assert "Bub:" not in detail


def test_low_contrast_bubble_is_added_to_dust_mask_when_enabled():
    image = _low_contrast_bubble_image()
    inferencer = _make_inferencer(detect_bubbles=True)

    is_dust, mask, ratio, detail = inferencer.check_dust_or_scratch_feature(image)

    yy, xx = np.ogrid[:128, :128]
    expected_bubble = ((xx - 64) ** 2 / (18 ** 2) + (yy - 64) ** 2 / (24 ** 2)) <= 1
    covered = np.count_nonzero(mask[expected_bubble] > 0) / np.count_nonzero(expected_bubble)

    assert is_dust is True
    assert covered >= 0.60
    assert ratio > 0.0
    assert "Bub:1" in detail


def test_low_contrast_bright_bubble_is_added_to_dust_mask_when_enabled():
    image = _low_contrast_bright_bubble_image()
    inferencer = _make_inferencer(detect_bubbles=True)

    is_dust, mask, ratio, detail = inferencer.check_dust_or_scratch_feature(image)

    yy, xx = np.ogrid[:128, :128]
    expected_bubble = ((xx - 64) ** 2 / (18 ** 2) + (yy - 64) ** 2 / (24 ** 2)) <= 1
    covered = np.count_nonzero(mask[expected_bubble] > 0) / np.count_nonzero(expected_bubble)

    assert is_dust is True
    assert covered >= 0.60
    assert ratio > 0.0
    assert "Bub:1" in detail


def test_low_contrast_ring_bubble_fills_interior_when_enabled():
    image = _low_contrast_ring_bubble_image()
    inferencer = _make_inferencer(detect_bubbles=True)

    is_dust, mask, ratio, detail = inferencer.check_dust_or_scratch_feature(image)

    yy, xx = np.ogrid[:128, :128]
    expected_bubble = ((xx - 64) ** 2 / (22 ** 2) + (yy - 64) ** 2 / (28 ** 2)) <= 1
    covered = np.count_nonzero(mask[expected_bubble] > 0) / np.count_nonzero(expected_bubble)

    assert is_dust is True
    assert covered >= 0.75
    assert mask[64, 64] == 255
    assert ratio > 0.0
    assert "Bub:1" in detail


def test_bubble_detector_ignores_small_dark_specks():
    image = np.full((128, 128), 80, dtype=np.uint8)
    image[60:64, 60:64] = 72
    inferencer = _make_inferencer(detect_bubbles=True)

    is_dust, mask, ratio, detail = inferencer.check_dust_or_scratch_feature(image)

    assert is_dust is False
    assert np.count_nonzero(mask) == 0
    assert ratio == 0.0
    assert "Bub:0" in detail


def test_bubble_detector_setting_can_hot_reload_from_db():
    config = CAPIConfig()

    config.apply_db_overrides([
        {"param_name": "dust_detect_bubbles_enabled", "decoded_value": "true"},
    ])

    assert config.dust_detect_bubbles_enabled is True
    assert config.to_dict()["dust_detect_bubbles_enabled"] is True


def test_settings_api_marks_dataclass_bool_params_as_bool():
    from capi_web import CAPIWebHandler

    captured = {}
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.db = SimpleNamespace(get_all_config_params=lambda: [])
    handler.inferencer = SimpleNamespace(config=CAPIConfig())
    handler._current_settings_user = lambda: None
    handler._send_json = lambda payload: captured.update(payload)

    handler._handle_api_settings()

    params = {p["param_name"]: p for p in captured["params"]}
    assert params["dust_detect_bubbles_enabled"]["param_type"] == "bool"


def test_settings_api_normalizes_existing_db_bool_param_type():
    from capi_web import CAPIWebHandler

    captured = {}
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.db = SimpleNamespace(get_all_config_params=lambda: [{
        "param_name": "dust_detect_bubbles_enabled",
        "param_value": '"True"',
        "param_type": "str",
        "decoded_value": "True",
    }])
    handler.inferencer = SimpleNamespace(config=CAPIConfig())
    handler._current_settings_user = lambda: None
    handler._send_json = lambda payload: captured.update(payload)

    handler._handle_api_settings()

    params = {p["param_name"]: p for p in captured["params"]}
    assert params["dust_detect_bubbles_enabled"]["param_type"] == "bool"


def test_bool_update_repairs_legacy_string_param_type():
    from capi_database import CAPIDatabase

    db_path = Path(f"_test_bool_param_repair_{uuid.uuid4().hex}.db")
    db = CAPIDatabase(str(db_path))
    try:
        conn = db._get_conn()
        try:
            conn.execute(
                """INSERT INTO config_params
                   (param_name, param_value, param_type, description)
                   VALUES (?, ?, ?, ?)""",
                ("dust_detect_bubbles_enabled", '"True"', "str", ""),
            )
            conn.commit()
        finally:
            conn.close()

        db.update_config_param(
            "dust_detect_bubbles_enabled",
            False,
            reason="unit test",
        )
        updated = db.get_config_param("dust_detect_bubbles_enabled")

        assert updated["param_type"] == "bool"
        assert updated["decoded_value"] is False
    finally:
        db_path.unlink(missing_ok=True)
