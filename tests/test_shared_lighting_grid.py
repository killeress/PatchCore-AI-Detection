"""Different lightings of one glass must sample the same training geometry."""
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

import capi_preprocess as pre
from capi_preprocess import PreprocessConfig, preprocess_panel_folder


def _write_lightings(folder):
    bright = np.zeros((1000, 1500), np.uint8)
    bright[100:900, 100:1400] = 180
    dim = bright.copy()
    dim[100:900, 100:1400] = 40
    dim[150:850, 160:1340] = 180
    paths = {}
    for lighting, image in (("W0F00000", bright), ("G0F00000", dim)):
        paths[lighting] = folder / f"{lighting}_test.png"
        assert cv2.imwrite(str(paths[lighting]), image)
    return paths, {"W0F00000": bright, "G0F00000": dim}


def _geometry(result):
    return [
        (t.x1, t.y1, t.x2, t.y2, t.zone, t.coverage, t.is_corner)
        for t in result.tiles
    ]


@pytest.mark.parametrize("boundary_mode", ["legacy", "raw", "raw_fallback"])
@pytest.mark.parametrize("stride", [64, 128])
def test_lightings_share_grid_but_keep_own_pixels_and_aoi_bounds(
    tmp_path, boundary_mode, stride,
):
    paths, images = _write_lightings(tmp_path)
    cfg = PreprocessConfig(
        tile_size=128, tile_stride=stride,
        large_panel_raw_boundary_enabled=boundary_mode != "legacy",
        large_panel_min_width_ratio=0.5 if boundary_mode == "raw" else 0.95,
        large_panel_min_height_ratio=0.5 if boundary_mode == "raw" else 0.95,
    )
    aoi = preprocess_panel_folder(tmp_path, replace(cfg, generate_grid_tiles=False))
    results = preprocess_panel_folder(tmp_path, cfg)
    reference = results["W0F00000"]
    target = results["G0F00000"]

    assert reference.tiles
    assert reference.foreground_bbox != target.foreground_bbox
    np.testing.assert_array_equal(reference.panel_polygon, target.panel_polygon)
    assert _geometry(reference) == _geometry(target)
    assert {t.zone for t in target.tiles} == {"inner", "edge"}
    for lighting, result in results.items():
        assert result.foreground_bbox == aoi[lighting].foreground_bbox
        for tile in result.tiles:
            assert tile.image.shape == (128, 128)
            np.testing.assert_array_equal(
                tile.image, images[lighting][tile.y1:tile.y2, tile.x1:tile.x2],
            )

    # The reference may be used only for boundary detection, without being a
    # selected output lighting (as in partial training and targeted previews).
    selected = preprocess_panel_folder(
        tmp_path, cfg, image_files=[paths["G0F00000"]],
        boundary_reference_files=paths.values(),
    )
    assert set(selected) == {"G0F00000"}
    assert _geometry(selected["G0F00000"]) == _geometry(reference)


def test_recovered_polygon_also_shares_grid_across_lightings(tmp_path, monkeypatch):
    _write_lightings(tmp_path)
    polygon = np.array([[100, 100], [1399, 100], [1399, 899], [100, 899]], np.float32)
    real_detect = pre.detect_panel_polygon

    def failed_legacy_polygon(image, config):
        bbox, _ = real_detect(image, config)
        return bbox, None

    monkeypatch.setattr(pre, "detect_panel_polygon", failed_legacy_polygon)
    monkeypatch.setattr(
        pre, "_detect_large_panel_boundary_file",
        lambda *args: ((105, 105, 1395, 895), polygon, False),
    )
    cfg = PreprocessConfig(
        tile_size=128, tile_stride=128, large_panel_raw_boundary_enabled=True,
        recover_failed_raw_boundary=True, product_resolution=(1920, 1200),
    )
    results = preprocess_panel_folder(tmp_path, cfg)

    assert results["W0F00000"].tiles
    assert all(not r.polygon_detection_failed for r in results.values())
    assert _geometry(results["W0F00000"]) == _geometry(results["G0F00000"])


def test_reference_grid_outside_image_falls_back_without_truncated_tiles(tmp_path):
    paths, _ = _write_lightings(tmp_path)
    cfg = PreprocessConfig(tile_size=128, tile_stride=128)
    local = pre.preprocess_panel_image(paths["G0F00000"], "G0F00000", cfg)
    result = pre.preprocess_panel_image(
        paths["G0F00000"], "G0F00000", cfg,
        reference_polygon=local.panel_polygon,
        reference_grid_bbox=(105, 105, 2400, 1800),
    )
    assert _geometry(result) == _geometry(local)
    assert all(t.image.shape == (128, 128) for t in result.tiles)
