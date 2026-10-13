"""Training grids must not feed slanted panel boundaries to the INNER model."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from capi_preprocess import PreprocessConfig, _generate_tiles, classify_tile_zone


def _slanted_fixture(top, turns):
    polygon = np.array(
        [[100, 100], [2100, 200], [2100, 1700], [100, 1600]], np.float32,
    )
    corners = np.array(
        [[600, top], [1111, top], [1111, top + 511], [600, top + 511]],
        np.float32,
    )
    for _ in range(turns):
        polygon = np.column_stack((polygon[:, 1], 2399 - polygon[:, 0]))
        corners = np.column_stack((corners[:, 1], 2399 - corners[:, 0]))
    lo, hi = corners.min(axis=0).astype(int), corners.max(axis=0).astype(int)
    return polygon, (int(lo[0]), int(lo[1]), int(hi[0]) + 1, int(hi[1]) + 1)


@pytest.mark.parametrize("turns", range(4))
@pytest.mark.parametrize("top", [145, 160])
def test_slanted_crossing_or_near_touching_corner_is_edge(top, turns):
    # At y=145 only a corner crosses the edge; at y=160 the whole tile is
    # inside but the corner is within the boundary fit's existing tolerance.
    polygon, rect = _slanted_fixture(top, turns)
    cfg = PreprocessConfig(tile_size=512)
    zone, coverage, center_distance, mask = classify_tile_zone(rect, polygon, cfg)

    assert center_distance > cfg.tile_size / 2 + cfg.otsu_offset
    assert coverage > 0.99
    assert zone == "edge"
    assert (mask is None) == (coverage == 1.0)


@pytest.mark.parametrize("turns", range(4))
def test_slanted_tile_with_clear_margin_stays_inner(turns):
    polygon, rect = _slanted_fixture(200, turns)
    zone, coverage, _distance, mask = classify_tile_zone(rect, polygon, PreprocessConfig())
    assert zone == "inner"
    assert coverage == 1.0
    assert mask is None


@pytest.mark.parametrize("raw_boundary,clearance", [(False, 18), (True, 24)])
def test_grid_edge_tolerance_includes_otsu_inset_and_boundary_fit(raw_boundary, clearance):
    polygon = np.array([[100, 100], [2100, 100], [2100, 1700], [100, 1700]], np.float32)
    cfg = PreprocessConfig(
        large_panel_raw_boundary_enabled=raw_boundary,
        raw_boundary_max_edge_residual_p95_ratio=0.04,
    )
    zone, coverage, _distance, _mask = classify_tile_zone(
        (600, 100 + clearance, 1112, 612 + clearance), polygon, cfg,
    )
    assert coverage == 1.0
    assert zone == "edge"


@pytest.mark.parametrize("stride", [256, 512])
def test_generated_inner_tiles_exclude_boundary_and_use_correct_zone_pipeline(stride):
    polygon, _ = _slanted_fixture(145, 0)
    image = np.zeros((1800, 2200), np.uint8)
    cv2.fillConvexPoly(image, polygon.astype(np.int32), 180)
    cfg = PreprocessConfig(
        tile_size=512, tile_stride=stride, outer_edge_extend=0,
        preprocess_after_tiling=True,
        image_preprocess_pipelines={
            "inner": [{"method": "mean", "params": {"kernel_size": 3}}],
            "edge": [{"method": "median", "params": {"kernel_size": 3}}],
        },
    )
    tiles = _generate_tiles(image, (105, 145, 2095, 1650), polygon, cfg)

    assert {t.zone for t in tiles} == {"inner", "edge"}
    assert len({(t.x1, t.y1) for t in tiles}) == len(tiles)
    for tile in tiles:
        corners = ((tile.x1, tile.y1), (tile.x2 - 1, tile.y1),
                   (tile.x2 - 1, tile.y2 - 1), (tile.x1, tile.y2 - 1))
        clearance = min(cv2.pointPolygonTest(polygon, corner, True) for corner in corners)
        if tile.zone == "inner":
            assert clearance > 5 + 512 * 0.03
            assert tile.coverage == 1.0
        assert tile.preprocess_steps[0]["method"] == (
            "median" if tile.zone == "edge" else "mean"
        )
    # The inward-shifted top tile must retain EDGE even though its centre is
    # already beyond the old centre-only cutoff.
    assert any(t.zone == "edge" and t.center_dist_to_edge > 261 for t in tiles)
