from pathlib import Path
import sys

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


def test_bubble_detector_ignores_small_dark_specks():
    image = np.full((128, 128), 80, dtype=np.uint8)
    image[60:64, 60:64] = 72
    inferencer = _make_inferencer(detect_bubbles=True)

    is_dust, mask, ratio, detail = inferencer.check_dust_or_scratch_feature(image)

    assert is_dust is False
    assert np.count_nonzero(mask) == 0
    assert ratio == 0.0
    assert "Bub:" not in detail


def test_bubble_detector_setting_can_hot_reload_from_db():
    config = CAPIConfig()

    config.apply_db_overrides([
        {"param_name": "dust_detect_bubbles_enabled", "decoded_value": "true"},
    ])

    assert config.dust_detect_bubbles_enabled is True
    assert config.to_dict()["dust_detect_bubbles_enabled"] is True
