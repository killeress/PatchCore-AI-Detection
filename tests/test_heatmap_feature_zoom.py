import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_heatmap import build_feature_zoom_panels, build_region_zoom_panels


def test_two_stage_feature_zoom_prioritizes_real_features_over_larger_dust():
    heatmap_binary = np.zeros((16, 16), dtype=np.uint8)
    heatmap_binary[8, 8] = 255
    dust_mask = np.zeros((16, 16), dtype=np.uint8)

    features = [
        {
            "abs_pos": (12, 12),
            "area": 500,
            "type": "dark",
            "dust_ratio": 1.0,
            "is_dust": True,
        },
        {
            "abs_pos": (8, 8),
            "area": 5,
            "type": "bright",
            "dust_ratio": 0.0,
            "is_dust": False,
        },
        {
            "abs_pos": (4, 4),
            "area": 300,
            "type": "dark",
            "dust_ratio": 1.0,
            "is_dust": True,
        },
    ]

    panels = build_feature_zoom_panels(
        heatmap_binary,
        dust_mask,
        features,
        tile_w_orig=16,
        tile_h_orig=16,
        tile_size=64,
        max_panels=3,
        final_label="REAL_NG",
    )

    labels = [label for _panel, label in panels]

    assert labels[0].startswith("Feat#1 Feature:REAL")
    assert "Final:REAL_NG" in labels[0]
    assert labels[1].startswith("Feat#2 Feature:DUST")
    assert labels[2].startswith("Feat#3 Feature:DUST")


def test_region_zoom_prioritizes_real_ng_regions_over_higher_score_dust():
    heatmap_binary = np.zeros((16, 16), dtype=np.uint8)
    heatmap_binary[2, 2] = 255
    heatmap_binary[8, 8] = 255
    heatmap_binary[12, 12] = 255
    dust_mask = np.zeros((16, 16), dtype=np.uint8)
    dust_mask[2, 2] = 255
    dust_mask[12, 12] = 255

    regions = [
        {
            "label_id": 1,
            "area": 1,
            "dust_overlap": 1,
            "metric_denominator": 1,
            "coverage": 1.0,
            "is_dust": True,
            "peak_in_dust": True,
            "dust_sub_peak_rescue": False,
            "max_score": 0.9,
            "peak_yx": (2, 2),
        },
        {
            "label_id": 2,
            "area": 1,
            "dust_overlap": 0,
            "metric_denominator": 1,
            "coverage": 0.0,
            "is_dust": False,
            "peak_in_dust": False,
            "dust_sub_peak_rescue": False,
            "max_score": 0.2,
            "peak_yx": (8, 8),
        },
        {
            "label_id": 3,
            "area": 1,
            "dust_overlap": 1,
            "metric_denominator": 1,
            "coverage": 1.0,
            "is_dust": True,
            "peak_in_dust": True,
            "dust_sub_peak_rescue": False,
            "max_score": 0.8,
            "peak_yx": (12, 12),
        },
    ]

    panels = build_region_zoom_panels(
        heatmap_binary,
        dust_mask,
        regions,
        tile_size=64,
        metric_name="COV",
        iou_threshold=0.02,
        max_panels=3,
    )

    labels = [label for _panel, label in panels]

    assert labels[0].startswith("Region#2 REAL_NG Score:0.2000")
    assert labels[1].startswith("Region#1 DUST Score:0.9000")
    assert labels[2].startswith("Region#3 DUST Score:0.8000")
