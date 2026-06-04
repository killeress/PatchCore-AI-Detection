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


def test_preprocess_panel_image_keeps_original_tiles_when_pipeline_enabled():
    cfg = PreprocessConfig(
        tile_size=256,
        image_preprocess_pipeline=[
            {"method": "bilateral", "params": {"diameter": 9, "sigma_color": 35.0, "sigma_space": 35.0}},
            {"method": "gaussian", "params": {"kernel_size": 5, "sigma": 1.0}},
        ],
    )
    result = preprocess_panel_image(FIXTURE, "STANDARD", cfg)
    assert result.tiles
    tile = result.tiles[0]
    assert tile.original_image is not None
    assert tile.preprocess_pipeline
    assert tile.original_image.shape == tile.image.shape
    assert len(result.preprocess_steps) == 2
    assert result.preprocess_total_ms >= 0.0
    assert all("elapsed_ms" in step for step in result.preprocess_steps)


def test_preprocess_panel_image_can_skip_grid_tiles_and_cache_image():
    cfg = PreprocessConfig(
        tile_size=256,
        cache_processed_image=True,
        generate_grid_tiles=False,
        image_preprocess_pipeline=[
            {"method": "mean", "params": {"kernel_size": 3}},
        ],
    )
    result = preprocess_panel_image(FIXTURE, "STANDARD", cfg)
    assert result.foreground_bbox != (0, 0, 0, 0)
    assert result.panel_polygon is not None
    assert result.tiles == []
    assert result.processed_image is not None
    assert len(result.preprocess_steps) == 1


def test_preprocess_timing_summary_aggregates_steps():
    from capi_image_preprocess_lab import summarize_preprocess_timings

    summary = summarize_preprocess_timings([
        [
            {"index": 1, "method": "gaussian", "method_label": "高斯平滑", "applied_params": {"kernel_size": 5, "sigma": 1.0}, "elapsed_ms": 2.0},
            {"index": 2, "method": "laplace_sharpen", "method_label": "Laplace 銳化", "applied_params": {"kernel_size": 3, "strength": 0.5}, "elapsed_ms": 3.0},
        ],
        [
            {"index": 1, "method": "gaussian", "method_label": "高斯平滑", "applied_params": {"kernel_size": 5, "sigma": 1.0}, "elapsed_ms": 4.0},
        ],
    ])

    assert summary["total_elapsed_ms"] == pytest.approx(9.0)
    assert summary["steps"][0]["calls"] == 2
    assert summary["steps"][0]["elapsed_ms_total"] == pytest.approx(6.0)
    assert summary["steps"][0]["elapsed_ms_avg"] == pytest.approx(3.0)


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


def test_preprocess_panel_image_with_preprocess_after_tiling():
    cfg = PreprocessConfig(
        tile_size=256,
        preprocess_after_tiling=True,
        image_preprocess_pipeline=[
            {"method": "gaussian", "params": {"kernel_size": 5, "sigma": 2.0}},
        ],
    )
    result = preprocess_panel_image(FIXTURE, "STANDARD", cfg)
    assert result.tiles
    
    # 驗證大圖的前處理步驟被跳過
    assert len(result.preprocess_steps) == 0
    
    # 驗證每個 tile 的 image 確實套用了前處理，而 original_image 是原來的
    tile = result.tiles[0]
    assert tile.original_image is not None
    # 由於套用了模糊，處理後的 image 應與 original_image 不同
    assert not np.array_equal(tile.image, tile.original_image)
