import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer


def test_high_coverage_region_is_dust_even_when_peak_is_outside_dust_mask():
    """Heatmap smoothing can move the max pixel slightly outside the OMIT dust mask."""
    config = CAPIConfig()
    config.dust_heatmap_iou_threshold = 0.2
    config.dust_high_cov_threshold = 0.5
    config.dust_peak_fraction_threshold = 0.8

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config

    anomaly_map = np.ones((100, 100), dtype=np.float32)
    anomaly_map[50, 99] = 2.0  # peak outside the dust mask

    dust_mask = np.zeros((100, 100), dtype=np.uint8)
    dust_mask[:, :97] = 255  # 97% of the heat region overlaps dust

    has_real, real_peak, _overall_iou, details, _heat_binary, _labels = (
        inferencer.check_dust_per_region(
            dust_mask,
            anomaly_map,
            top_percent=100.0,
            metric="coverage",
            iou_threshold=config.dust_heatmap_iou_threshold,
        )
    )

    assert has_real is False
    assert real_peak is None
    assert len(details) == 1
    assert details[0]["coverage"] == 0.97
    assert details[0]["peak_in_dust"] is False
    assert details[0]["dust_sub_peak_rescue"] is False
    assert details[0]["is_dust"] is True


def test_force_include_seed_keeps_aoi_center_from_being_dropped_by_top_percent():
    """AOI center can be a weaker hot spot than dust/tape, but must still be evaluated."""
    config = CAPIConfig()
    config.dust_heatmap_iou_threshold = 0.2
    config.dust_high_cov_threshold = 0.5
    config.dust_peak_fraction_threshold = 0.8

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config

    anomaly_map = np.zeros((100, 100), dtype=np.float32)
    anomaly_map[88:94, 88:94] = 100.0  # strongest hot region on dust
    anomaly_map[49:52, 49:52] = 20.0   # AOI center hot region, below top-percent cutoff

    dust_mask = np.zeros((100, 100), dtype=np.uint8)
    dust_mask[85:97, 85:97] = 255

    has_real_without_seed, _peak_without_seed, _iou, details_without_seed, _hm, _labels = (
        inferencer.check_dust_per_region(
            dust_mask,
            anomaly_map,
            top_percent=0.1,
            metric="coverage",
            iou_threshold=config.dust_heatmap_iou_threshold,
        )
    )

    has_real_with_seed, real_peak, _iou, details_with_seed, heat_binary, _labels = (
        inferencer.check_dust_per_region(
            dust_mask,
            anomaly_map,
            top_percent=0.1,
            metric="coverage",
            iou_threshold=config.dust_heatmap_iou_threshold,
            force_include_yx=(50, 50),
            force_include_radius=2,
            force_include_min_score=10.0,
        )
    )

    assert has_real_without_seed is False
    assert all(r["is_dust"] for r in details_without_seed)

    assert has_real_with_seed is True
    assert real_peak == (49, 49)
    assert any(not r["is_dust"] for r in details_with_seed)
    assert heat_binary[50, 50] == 255
