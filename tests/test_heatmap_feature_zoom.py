import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_heatmap import build_feature_zoom_panels


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
