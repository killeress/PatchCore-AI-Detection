"""AOI bomb matching must survive polygon correction of raw image bounds."""
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from capi_config import BombDefect, CAPIConfig
from capi_inference import CAPIInferencer, ImageResult, TileInfo


RESOLUTION = (1920, 1200)
# Reconstructed to reproduce all five logged linear anchors exactly.
RAW_BOUNDS = (691, 715, 5612, 4267)
POLYGON = np.array([
    [706.7, 710.8], [5614.6, 749.8], [5592.6, 3809.4], [688.2, 3782.8],
], dtype=np.float32)
# Product coordinates and corrected image anchors from T563R583BF18's log.
AOI_POINTS = [(359, 234), (1484, 346), (360, 534), (1077, 569), (1801, 702)]
IMAGE_POINTS = [(1624, 1317), (4497, 1624), (1621, 2085), (3455, 2187), (5299, 2538)]
BOMB_POINTS = [(350, 231), (1475, 343), (356, 530), (1071, 565), (1798, 700)]


def make_inferencer():
    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.config = CAPIConfig(
        bomb_match_tolerance=20,
        bomb_defects=[BombDefect("STANDARD", "B01", "point", BOMB_POINTS)],
    )
    return inferencer


def make_tile(index, product_point, image_point):
    px, py = product_point
    ix, iy = image_point
    return TileInfo(
        tile_id=index, x=ix - 256, y=iy - 256, width=512, height=512,
        image=np.zeros((512, 512), dtype=np.uint8), zone="inner",
        is_aoi_coord_tile=True, aoi_defect_code="PCDK2",
        aoi_product_x=px, aoi_product_y=py, aoi_image_x=ix, aoi_image_y=iy,
        anomaly_peak_source="aoi_real_region",
        anomaly_peak_x=ix, anomaly_peak_y=iy,
    )


def make_result(tiles):
    return ImageResult(
        image_path=Path("STANDARD_080924.tif"), image_size=(6576, 4384),
        otsu_bounds=RAW_BOUNDS, raw_bounds=RAW_BOUNDS, panel_polygon=POLYGON.copy(),
        exclusion_regions=[], tiles=tiles, excluded_tile_count=0,
        processed_tile_count=len(tiles), processing_time=0.0,
        anomaly_tiles=[(tile, 0.85, None) for tile in tiles],
    )


def apply_bomb_postprocess(inferencer, result, bomb_info, version, monkeypatch):
    if version == "v2":
        inferencer._apply_bomb_postprocess([result], bomb_info, RESOLUTION)
        return

    inferencer.config.enable_panel_polygon = False
    inferencer.config.scratch_classifier_enabled = False
    inferencer.threshold = 0.35
    inferencer.base_dir = Path(".")
    inferencer.mark_template = None
    inferencer.inferencer = MagicMock()
    inferencer.edge_inspector = None
    inferencer._model_mapping = {}
    inferencer._threshold_mapping = {}
    inferencer._inferencers = {}
    monkeypatch.setattr(inferencer, "_prepare_panel_image_files",
                        lambda _path: ([result.image_path], False))
    monkeypatch.setattr(inferencer, "_detect_panel_mark_binary_region",
                        lambda *_args, **_kwargs: (None, []))
    monkeypatch.setattr(inferencer, "_parse_defect_txt", lambda _path: {})
    monkeypatch.setattr(inferencer, "_find_raw_object_bounds",
                        lambda *_args, **_kwargs: (RAW_BOUNDS, np.ones((64, 64), dtype=np.uint8)))
    monkeypatch.setattr(inferencer, "preprocess_image", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(inferencer, "run_inference", lambda result, **_kwargs: result)
    inferencer._process_panel_v1(
        Path("."), cpu_workers=1, product_resolution=RESOLUTION,
        bomb_info=bomb_info, aoi_report_override={},
    )


@pytest.mark.parametrize("source", ["client", "config"])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_five_standard_bombs_match_polygon_corrected_aoi_anchors(source, version, capsys, monkeypatch):
    inferencer = make_inferencer()
    tiles = [make_tile(i, product, image)
             for i, (product, image) in enumerate(zip(AOI_POINTS, IMAGE_POINTS))]
    # A neighboring bomb peak cannot turn an unrelated AOI point into a bomb.
    unrelated = make_tile(5, (1000, 500), IMAGE_POINTS[0])
    below_threshold = make_tile(6, AOI_POINTS[0], IMAGE_POINTS[0])
    below_threshold.is_aoi_coord_below_threshold = True
    tiles.extend([unrelated, below_threshold])
    result = make_result(tiles)
    bomb_info = dict(image_prefix="STANDARD", defect_type="point", coordinates=BOMB_POINTS)

    apply_bomb_postprocess(
        inferencer, result, bomb_info if source == "client" else None, version, monkeypatch,
    )

    assert [tile.is_bomb for tile in tiles] == [True] * 5 + [False, False]
    assert all(tile.bomb_defect_code == "B01" for tile in tiles[:5])
    assert unrelated.bomb_defect_code == below_threshold.bomb_defect_code == ""
    log = capsys.readouterr().out
    assert "B01×5" in log
    if version == "v2":
        assert log.count("matched=True") == 5


@pytest.mark.parametrize("offset,expected", [
    ((20, 0), True), ((0, 20), True), ((20, 20), True),
    ((21, 0), False), ((0, 21), False),
])
def test_corrected_aoi_point_keeps_product_pixel_tolerance(offset, expected):
    inferencer = make_inferencer()
    bx, by = BOMB_POINTS[0]
    tile = make_tile(0, (bx + offset[0], by + offset[1]), IMAGE_POINTS[0])
    result = make_result([tile])
    bomb_info = dict(image_prefix="STANDARD", defect_type="point", coordinates=[(bx, by)])

    inferencer._apply_bomb_postprocess([result], bomb_info, RESOLUTION)

    assert tile.is_bomb is expected


def test_corrected_aoi_point_does_not_match_another_screen():
    inferencer = make_inferencer()
    tile = make_tile(0, AOI_POINTS[0], IMAGE_POINTS[0])
    bomb_info = dict(image_prefix="U0F00000", defect_type="point", coordinates=BOMB_POINTS)

    inferencer._apply_bomb_postprocess([make_result([tile])], bomb_info, RESOLUTION)

    assert tile.is_bomb is False
    assert tile.bomb_defect_code == ""


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("line_shape", [True, False])
def test_corrected_aoi_line_bomb_preserves_heatmap_shape_check(version, line_shape, monkeypatch):
    inferencer = make_inferencer()
    tile = make_tile(0, AOI_POINTS[3], IMAGE_POINTS[3])
    result = make_result([tile])
    anomaly_map = np.zeros((512, 512), dtype=np.float32)
    if line_shape:
        anomaly_map[250:260, 100:412] = 1.0
    else:
        anomaly_map[250:260, 250:260] = 1.0
    result.anomaly_tiles = [(tile, 0.85, anomaly_map)]
    bomb_info = dict(image_prefix="STANDARD", defect_type="line",
                     coordinates=[(1000, 565), (1200, 565)])

    apply_bomb_postprocess(inferencer, result, bomb_info, version, monkeypatch)

    assert tile.is_bomb is line_shape


def test_missing_aoi_product_coordinates_keeps_image_anchor_fallback():
    inferencer = make_inferencer()
    anchor = inferencer._map_aoi_coords(*BOMB_POINTS[0], RAW_BOUNDS, RESOLUTION)
    tile = make_tile(0, AOI_POINTS[0], anchor)
    tile.aoi_product_x = tile.aoi_product_y = -1
    bomb_info = dict(image_prefix="STANDARD", defect_type="point", coordinates=BOMB_POINTS)

    inferencer._apply_bomb_postprocess([make_result([tile])], bomb_info, RESOLUTION)

    assert tile.is_bomb is True


@pytest.mark.parametrize("prefix,bounds,points,anchors,bombs,expected", [
    pytest.param(
        "W0F00000", (259, 273, 6369, 3719),
        [(1892, 962), (98, 93), (123, 117), (97, 142), (148, 142), (140, 90)],
        [(6279, 3342), (570, 569), (650, 646), (567, 726), (729, 726), (704, 560)],
        [(90, 90), (115, 115), (90, 140), (140, 140), (140, 90)],
        [False] + [True] * 5, id="CAPI07",
    ),
    pytest.param(
        "WGF50500", (373, 442, 6002, 3625),
        [(360, 137), (1573, 328), (291, 555), (1017, 570), (1681, 725)],
        [(1428, 845), (4984, 1408), (1226, 2077), (3354, 2121), (5301, 2578)],
        [(353, 134), (1565, 323), (286, 550), (1011, 565), (1677, 720)],
        [True] * 5, id="CAPI01",
    ),
])
def test_linear_mapping_matches_bombs_before_and_after_aoi_fallback(
    prefix, bounds, points, anchors, bombs, expected,
):
    inferencer = make_inferencer()
    resolution = (1920, 1080)
    bomb_info = dict(image_prefix=prefix, defect_type="point", coordinates=bombs)
    inferencer.config.bomb_defects = [BombDefect(prefix, "B01", "point", bombs)]
    tiles = [make_tile(i, product, image)
             for i, (product, image) in enumerate(zip(points, anchors))]

    # These successful logs use the uncorrected linear mapping for every anchor.
    assert [inferencer._map_aoi_coords(*point, bounds, resolution) for point in points] == anchors
    # Without product_coords, the original image-coordinate matching still works.
    assert [inferencer.check_bomb_match(
        prefix, *anchor, bounds, product_resolution=resolution,
    )[0] for anchor in anchors] == expected

    result = make_result(tiles)
    result.image_path = Path(f"{prefix}_test.tif")
    result.raw_bounds = result.otsu_bounds = bounds
    result.panel_polygon = None
    inferencer._apply_bomb_postprocess([result], bomb_info, resolution)

    assert [tile.is_bomb for tile in tiles] == expected
