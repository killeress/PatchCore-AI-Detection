from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer


def _inferencer(*, extension=0, rescue_threshold=180):
    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = CAPIConfig()
    inferencer.config.dust_area_min = 15
    inferencer.config.dust_area_max = 1000
    inferencer.config.dust_extension = extension
    inferencer.config.dust_bright_rescue_threshold = rescue_threshold
    inferencer.config.dust_detect_dark_particles = False
    return inferencer


@pytest.mark.parametrize("extension", [0, 5])
def test_large_white_region_is_masked_despite_particle_area_limit(extension):
    image = np.full((512, 512), 60, dtype=np.uint8)
    image[380:, 80:] = 255
    inferencer = _inferencer(extension=extension)

    is_dust, mask, ratio, detail = inferencer.check_dust_or_scratch_feature(image)

    assert is_dust is True
    assert np.all(mask[385:, 85:] == 255)
    assert mask[200, 200] == 0
    assert mask[450, 40] == 0
    assert ratio == np.count_nonzero(mask) / mask.size
    assert "BrightRescue:1" in detail
    if extension:
        assert mask[450, 77] == 255


def test_large_white_region_respects_rescue_disabled():
    image = np.full((512, 512), 60, dtype=np.uint8)
    image[380:, 80:] = 255

    _, mask, _, detail = _inferencer(rescue_threshold=0).check_dust_or_scratch_feature(image)

    assert mask[450, 300] == 0
    assert "BrightRescue:" not in detail


def test_large_non_white_region_keeps_particle_area_limit():
    image = np.full((512, 512), 60, dtype=np.uint8)
    image[380:, 80:] = 110

    _, mask, _, detail = _inferencer().check_dust_or_scratch_feature(image)

    assert mask[450, 300] == 0
    assert "BrightRescue:" not in detail


def test_small_white_particle_is_not_counted_twice():
    image = np.full((512, 512), 60, dtype=np.uint8)
    cv2.circle(image, (200, 200), 8, 255, -1)

    is_dust, mask, _, detail = _inferencer().check_dust_or_scratch_feature(image)

    assert is_dust is True
    assert mask[200, 200] == 255
    assert "P:1 S:0" in detail
    assert "BrightRescue:" not in detail


def test_full_white_tile_is_kept_as_mask():
    image = np.full((512, 512), 255, dtype=np.uint8)

    is_dust, mask, ratio, detail = _inferencer().check_dust_or_scratch_feature(image)

    assert is_dust is True
    assert np.all(mask == 255)
    assert ratio == 1.0
    assert "BrightRescue:1" in detail


def test_white_region_filters_its_heat_but_keeps_unmasked_defect():
    image = np.full((512, 512), 60, dtype=np.uint8)
    image[380:, 80:] = 255
    inferencer = _inferencer()
    _, mask, _, _ = inferencer.check_dust_or_scratch_feature(image)
    heatmap = np.zeros((512, 512), dtype=np.float32)
    heatmap[440:450, 290:300] = 1.0

    has_real_defect, peak, _, regions, _, _ = inferencer.check_dust_per_region(
        mask, heatmap, iou_threshold=0.1
    )
    assert has_real_defect is False
    assert peak is None
    assert len(regions) == 1
    assert regions[0]["is_dust"] is True

    heatmap[190:200, 190:200] = 1.0
    has_real_defect, peak, _, regions, _, _ = inferencer.check_dust_per_region(
        mask, heatmap, iou_threshold=0.1
    )
    assert has_real_defect is True
    assert 190 <= peak[0] < 200 and 190 <= peak[1] < 200
    assert sum(region["is_dust"] for region in regions) == 1
