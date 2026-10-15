"""Broad, faint OMIT bubbles under uneven illumination must retain their shape."""

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer


def _inferencer(*, enabled=True):
    config = CAPIConfig()
    config.dust_detect_bubbles_enabled = enabled
    config.dust_detect_dark_particles = False
    config.dust_extension = 0
    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config
    return inferencer


def _uneven_background():
    yy, xx = np.mgrid[:512, :512]
    return 32 + yy * 0.006 + yy ** 2 * 0.00012 - xx * (0.003 + yy * 0.00003)


def _two_broad_bubbles(contrast=4):
    image = _uneven_background()
    bubbles = []
    for center, axes, angle in [
        ((215, 280), (56, 46), 10),
        ((360, 374), (75, 62), 30),
    ]:
        bubble = np.zeros((512, 512), dtype=np.uint8)
        cv2.ellipse(bubble, center, axes, angle, 0, 360, 255, -1)
        image[bubble > 0] += contrast
        bubbles.append(bubble)
    image = cv2.GaussianBlur(np.rint(image).astype(np.uint8), (9, 9), 0)
    return image, bubbles


def _broad_mask(image):
    inferencer = _inferencer()
    return inferencer._detect_broad_bubble_mask(image, 15, 100000, 0)


@pytest.mark.parametrize("contrast", [4, 5])
def test_broad_low_contrast_bubbles_survive_uneven_illumination(contrast):
    image, bubbles = _two_broad_bubbles(contrast)
    mask = _inferencer()._detect_bubble_mask(image, 15, 100000, 0)

    for bubble in bubbles:
        coverage = np.count_nonzero(mask[bubble > 0]) / np.count_nonzero(bubble)
        assert coverage >= 0.90
    assert mask[200, 310] == 0  # Background between the two isolated bubbles.


def test_broad_bubble_detection_remains_opt_in(monkeypatch):
    image, _ = _two_broad_bubbles()
    inferencer = _inferencer(enabled=False)

    def unexpected_detection(*args, **kwargs):
        pytest.fail("Disabled bubble detection must not run background fitting")

    monkeypatch.setattr(inferencer, "_detect_broad_bubble_mask", unexpected_detection)
    _, _, _, detail = inferencer.check_dust_or_scratch_feature(image)
    assert "Bub:" not in detail


@pytest.mark.parametrize("width,height", [(100, 100), (100, 150)])
def test_broad_detector_rejects_medium_rectangular_brightness(width, height):
    image = _uneven_background()
    image[190:190 + height, 180:180 + width] += 5
    image = cv2.GaussianBlur(np.rint(image).astype(np.uint8), (9, 9), 0)

    assert np.count_nonzero(_broad_mask(image)) == 0


def test_broad_detector_rejects_oversized_brightness_and_sensor_noise():
    image = _uneven_background()
    oversized = np.zeros((512, 512), dtype=np.uint8)
    cv2.ellipse(oversized, (256, 256), (170, 130), 0, 0, 360, 255, -1)
    image[oversized > 0] += 5
    image = cv2.GaussianBlur(np.rint(image).astype(np.uint8), (9, 9), 0)
    assert np.count_nonzero(_broad_mask(image)) == 0

    noise = np.random.default_rng(42).normal(0, 1.0, (512, 512))
    noisy_background = np.rint(_uneven_background() + noise).astype(np.uint8)
    assert np.count_nonzero(_broad_mask(noisy_background)) == 0


def test_broad_detector_does_not_fill_irregular_bright_texture():
    image = _uneven_background()
    irregular = np.zeros((512, 512), dtype=np.uint8)
    points = np.array([
        [190, 180], [300, 180], [300, 270], [230, 270],
        [230, 235], [270, 235], [270, 215], [190, 215],
    ], dtype=np.int32)
    cv2.fillPoly(irregular, [points], 255)
    image[irregular > 0] += 5
    image = cv2.GaussianBlur(np.rint(image).astype(np.uint8), (9, 9), 0)

    assert np.count_nonzero(_broad_mask(image)) == 0


def test_broad_bubble_mask_preserves_a_separate_small_feature():
    image, bubbles = _two_broad_bubbles()
    image[347:354, 107:114] += 20
    mask = _inferencer()._detect_bubble_mask(image, 15, 100000, 0)

    for bubble in bubbles:
        coverage = np.count_nonzero(mask[bubble > 0]) / np.count_nonzero(bubble)
        assert coverage >= 0.90
    assert np.count_nonzero(mask[340:360, 100:120]) == 0


def test_broad_bubbles_are_filtered_while_a_separate_hotspot_remains_ng():
    image, _ = _two_broad_bubbles()
    inferencer = _inferencer()
    mask = inferencer._detect_bubble_mask(image, 15, 100000, 0)
    heatmap = np.zeros((512, 512), dtype=np.float32)
    cv2.circle(heatmap, (215, 280), 8, 0.8, -1)
    cv2.circle(heatmap, (360, 374), 8, 0.9, -1)
    cv2.circle(heatmap, (110, 350), 5, 0.7, -1)
    heatmap[350, 110] = 1.0

    has_real, peak, _, regions, _, _ = inferencer.check_dust_per_region(
        mask, heatmap, top_percent=100, metric="coverage", iou_threshold=0.5,
    )

    assert has_real is True
    assert peak == (350, 110)
    assert sum(region["is_dust"] for region in regions) == 2
    assert sum(not region["is_dust"] for region in regions) == 1
