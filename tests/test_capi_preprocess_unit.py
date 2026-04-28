"""capi_preprocess 模組的單元測試"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_preprocess import PreprocessConfig, TileResult, PanelPreprocessResult


def test_preprocess_config_defaults():
    cfg = PreprocessConfig()
    assert cfg.tile_size == 512
    assert cfg.tile_stride == 512
    assert cfg.otsu_offset == 5
    assert cfg.enable_panel_polygon is True
    assert cfg.edge_threshold_px == 768
    assert cfg.coverage_min == 0.3


def test_tile_result_zone_values():
    import numpy as np
    t = TileResult(tile_id=0, x1=0, y1=0, x2=512, y2=512,
                   image=np.zeros((512,512), np.uint8),
                   mask=None, coverage=1.0, zone="inner",
                   center_dist_to_edge=999.0)
    assert t.zone == "inner"
