import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer


def test_two_stage_does_not_promote_clean_feature_inside_dust_dominated_hot_zone():
    config = CAPIConfig()
    config.dust_two_stage_dust_ratio = 0.3
    config.dust_two_stage_diff_percentile = 50.0
    config.dust_two_stage_min_area = 3
    config.dust_high_cov_threshold = 0.5

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config

    tile = np.full((64, 64), 100, dtype=np.uint8)
    tile[30:35, 30:35] = 30

    anomaly_map = np.ones((16, 16), dtype=np.float32)

    dust_mask = np.full((64, 64), 255, dtype=np.uint8)
    dust_mask[28:37, 28:37] = 0

    has_real, real_peak, features, detail = inferencer.check_dust_two_stage(
        tile,
        anomaly_map,
        dust_mask,
        score=1.0,
    )

    assert has_real is False
    assert real_peak is None
    assert "-> DUST" in detail
    target = next(
        f for f in features
        if f["dust_ratio"] == 0.0
        and f["zone_dust_dominated"] is True
        and f["dust_reason"] == "zone_dominated"
        and f["is_dust"] is True
    )
    assert target["dust_overlap"] == 0
    assert target["feature_bbox"][2] > 0
    assert target["feature_bbox"][3] > 0
    assert target["feature_contour"]


def test_two_stage_ignores_tile_border_feature_as_real_evidence():
    config = CAPIConfig()
    config.dust_two_stage_dust_ratio = 0.3
    config.dust_two_stage_diff_percentile = 50.0
    config.dust_two_stage_min_area = 3
    config.dust_high_cov_threshold = 1.1

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config

    tile = np.full((64, 64), 100, dtype=np.uint8)
    tile[0:6, 28:38] = 30

    anomaly_map = np.ones((16, 16), dtype=np.float32)
    dust_mask = np.zeros((64, 64), dtype=np.uint8)

    has_real, real_peak, features, detail = inferencer.check_dust_two_stage(
        tile,
        anomaly_map,
        dust_mask,
        score=1.0,
    )

    assert has_real is False
    assert real_peak is None
    assert features == []
    assert "ignored_border=" in detail
