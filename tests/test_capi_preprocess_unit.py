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


import tempfile


def test_filter_panel_lighting_files_keeps_5():
    from capi_preprocess import filter_panel_lighting_files
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for name in [
            "G0F00000_164039.tif", "R0F00000_164041.tif",
            "W0F00000_164033.tif", "WGF50500_164035.tif",
            "STANDARD_164038.tif",
            # filter out:
            "B0F00000_164043.tif", "PINIGBI _164030.tif",
            "SG0F00000_164039.tif", "SR0F00000_164041.tif",
            "SSTANDARD_164038.tif",
            "Optics.log",
        ]:
            (base / name).write_bytes(b"x")
        result = filter_panel_lighting_files(base)
        assert set(result.keys()) == {"G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD"}
        assert result["G0F00000"].name == "G0F00000_164039.tif"

def test_filter_panel_lighting_files_partial_panel():
    from capi_preprocess import filter_panel_lighting_files
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "G0F00000_x.tif").write_bytes(b"x")
        (base / "STANDARD_x.tif").write_bytes(b"x")
        result = filter_panel_lighting_files(base)
        assert set(result.keys()) == {"G0F00000", "STANDARD"}
