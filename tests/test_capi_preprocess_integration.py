"""capi_preprocess 模組的整合測試：preprocess_panel_image 完整 pipeline"""
import sys
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from capi_preprocess import preprocess_panel_image, preprocess_panel_folder, PreprocessConfig

FIXTURE = Path(__file__).parent / "fixtures" / "preprocess" / "synthetic_panel.png"


def test_preprocess_panel_image_basic():
    cfg = PreprocessConfig(tile_size=256, tile_stride=256, edge_threshold_px=384)
    result = preprocess_panel_image(FIXTURE, "STANDARD", cfg)
    assert result.lighting == "STANDARD"
    assert not result.polygon_detection_failed
    assert result.panel_polygon is not None
    assert len(result.tiles) > 0
    zones = {t.zone for t in result.tiles}
    assert "inner" in zones
    assert "edge" in zones
    # outside tile 不應出現
    assert "outside" not in zones


def test_preprocess_panel_image_with_reference_polygon():
    cfg = PreprocessConfig(tile_size=256)
    ref = np.array([[200, 150], [1300, 150], [1300, 850], [200, 850]], np.float32)
    result = preprocess_panel_image(FIXTURE, "G0F00000", cfg, reference_polygon=ref)
    # 應該直接套 reference 不重新偵測
    np.testing.assert_array_almost_equal(result.panel_polygon, ref)


def test_preprocess_panel_folder_uses_reference_polygon(tmp_path):
    # 複製 fixture 5 份模擬不同 lighting
    for lighting in ["STANDARD", "G0F00000", "R0F00000", "W0F00000", "WGF50500"]:
        shutil.copy(FIXTURE, tmp_path / f"{lighting}_x.png")
    cfg = PreprocessConfig(tile_size=256)
    results = preprocess_panel_folder(tmp_path, cfg)
    assert set(results.keys()) == {"STANDARD", "G0F00000", "R0F00000", "W0F00000", "WGF50500"}
    # 所有 lighting 應共用同一 polygon
    ref_poly = results["STANDARD"].panel_polygon
    for lighting, r in results.items():
        np.testing.assert_array_almost_equal(r.panel_polygon, ref_poly)
