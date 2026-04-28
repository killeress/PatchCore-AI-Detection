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
