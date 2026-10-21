"""Polygon-inward ROI tests for edge PatchCore samples."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from capi_config import CAPIConfig
from capi_inference import AOIReportDefect, CAPIInferencer, ImageResult
from capi_preprocess import (
    PreprocessConfig,
    resolve_aoi_inward_shift_axes,
    resolve_inward_polygon_tile,
)


@pytest.mark.parametrize("transpose,expected_axes", [(False, "x"), (True, "y")])
def test_inward_roi_retries_both_axes_when_raw_bounds_miss_polygon_corner(
    transpose, expected_axes,
):
    # Logged WGF50500 polygon/anchor; the synthetic raw box extends below
    # the real panel, so its axis hint misses the bottom edge.
    polygon = np.array(
        [[851.4, 887.6], [5912.0, 893.8], [5920.0, 4066.2], [832.3, 4055.2]],
        dtype=np.float32,
    )
    anchor = (5868, 3990)
    bounds = (840, 898, 5919, 4366)
    image_shape = (4384, 6576)
    if transpose:
        polygon = polygon[:, ::-1].copy()
        anchor = anchor[::-1]
        bounds = (bounds[1], bounds[0], bounds[3], bounds[2])
        image_shape = image_shape[::-1]

    axes = resolve_aoi_inward_shift_axes(*anchor, bounds, 512)
    assert axes == expected_axes
    tx, ty, coverage, shifted = resolve_inward_polygon_tile(
        anchor_xy=anchor,
        polygon=polygon,
        image_shape=image_shape,
        tile_size=512,
        keep_anchor_inside=True,
        shift_axes=axes,
    )

    assert shifted
    assert tx < anchor[0] - 256 and ty < anchor[1] - 256
    assert coverage >= 0.999
    assert tx <= anchor[0] <= tx + 511
    assert ty <= anchor[1] <= ty + 511
    for corner in ((tx, ty), (tx + 511, ty), (tx + 511, ty + 511), (tx, ty + 511)):
        assert cv2.pointPolygonTest(polygon, corner, True) >= -0.5


@pytest.mark.parametrize("transpose,axes", [(False, "x"), (True, "y")])
def test_inward_roi_preserves_single_axis_when_it_fits(transpose, axes):
    polygon = np.array(
        [[500, 500], [1900, 500], [1900, 1500], [500, 1500]],
        dtype=np.float32,
    )
    anchor = (550, 1000)
    image_shape = (2000, 2400)
    if transpose:
        polygon = polygon[:, ::-1].copy()
        anchor = anchor[::-1]
        image_shape = image_shape[::-1]

    tx, ty, coverage, shifted = resolve_inward_polygon_tile(
        anchor_xy=anchor,
        polygon=polygon,
        image_shape=image_shape,
        tile_size=512,
        keep_anchor_inside=True,
        shift_axes=axes,
    )

    assert shifted
    assert coverage >= 0.999
    if axes == "x":
        assert tx >= 500 and ty == anchor[1] - 256
    else:
        assert ty >= 500 and tx == anchor[0] - 256


@pytest.mark.parametrize("flip_x,flip_y", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("reverse_winding", [False, True])
def test_logged_aoi_corner_uses_minimum_shift_and_retains_defect_context(
    flip_x, flip_y, reverse_winding,
):
    # YQ512H230C41 / CM00 (1850, 1184): the old centroid walk returned
    # (5283, 3015), leaving the anchor at x=511 and cropping its right half.
    polygon = np.array(
        [[816.1, 289.7], [5973.3, 297.7], [5967.4, 3527.7], [821.1, 3514.9]],
        dtype=np.float32,
    )
    anchor = [5794, 3485]
    expected = [5456, 3015]
    for axis, flip, span in ((0, flip_x, 6576), (1, flip_y, 4384)):
        if flip:
            polygon[:, axis] = span - 1 - polygon[:, axis]
            anchor[axis] = span - 1 - anchor[axis]
            expected[axis] = span - 512 - expected[axis]
    if reverse_winding:
        polygon = polygon[::-1].copy()

    x, y, coverage, shifted = resolve_inward_polygon_tile(
        tuple(anchor), polygon, (4384, 6576), 512, keep_anchor_inside=True,
    )

    assert (x, y) == tuple(expected)
    assert coverage >= 0.999 and shifted
    assert 32 <= anchor[0] - x <= 511 - 32
    assert 32 <= anchor[1] - y <= 511 - 32


@pytest.mark.parametrize("anchor", [(12, 12), (42, 28), (25, 25)])
def test_aoi_origin_is_nearest_valid_integer_crop(anchor):
    polygon = np.array([[8, 8], [47, 11], [43, 39], [11, 37]], dtype=np.float32)
    size, margin = 16, 2
    original = (anchor[0] - size // 2, anchor[1] - size // 2)
    candidates = []
    # Independent exhaustive oracle, including raster coverage and AOI margin.
    for y in range(48 - size + 1):
        for x in range(60 - size + 1):
            if not (margin <= anchor[0] - x < size - margin
                    and margin <= anchor[1] - y < size - margin):
                continue
            corners = ((x, y), (x + size - 1, y),
                       (x + size - 1, y + size - 1), (x, y + size - 1))
            if min(cv2.pointPolygonTest(polygon, p, True) for p in corners) < -0.5:
                continue
            mask = np.zeros((size, size), dtype=np.uint8)
            cv2.fillPoly(mask, [(polygon - [x, y]).astype(np.int32)], 1)
            if mask.mean() >= 0.999:
                candidates.append(((x - original[0]) ** 2 + (y - original[1]) ** 2, x, y))
    assert candidates

    x, y, coverage, _ = resolve_inward_polygon_tile(
        anchor, polygon, (48, 60), size, keep_anchor_inside=True, anchor_margin_px=margin,
    )
    assert (x - original[0]) ** 2 + (y - original[1]) ** 2 == min(c[0] for c in candidates)
    assert coverage >= 0.999


def test_aoi_near_panel_corner_keeps_margin_when_full_inward_crop_is_impossible():
    polygon = np.array([[500, 500], [1900, 500], [1900, 1500], [500, 1500]], dtype=np.float32)
    anchor = (1895, 1495)
    x, y, coverage, shifted = resolve_inward_polygon_tile(
        anchor, polygon, (2000, 2400), 512, keep_anchor_inside=True,
    )
    assert (x, y) == (1416, 1016)
    assert (x + 511 - anchor[0], y + 511 - anchor[1]) == (32, 32)
    assert coverage < 0.999 and shifted


@pytest.mark.parametrize("image_shape,tile_size,anchor", [((80, 100), 64, (1, 78)), ((20, 30), 64, (29, 19))])
def test_aoi_margin_clips_at_image_bounds_without_padding(image_shape, tile_size, anchor):
    height, width = image_shape
    polygon = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    x, y, coverage, _ = resolve_inward_polygon_tile(
        anchor, polygon, image_shape, tile_size, keep_anchor_inside=True,
    )
    assert 0 <= x <= max(0, width - tile_size)
    assert 0 <= y <= max(0, height - tile_size)
    assert x <= max(0, anchor[0] - 31) and x + tile_size > min(width - 1, anchor[0] + 31)
    assert y <= max(0, anchor[1] - 31) and y + tile_size > min(height - 1, anchor[1] + 31)
    assert coverage == 1.0


def test_v2_logged_aoi_crop_preserves_defect_pixels_and_original_coordinates():
    cfg = CAPIConfig()
    cfg.tile_size = 512
    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.config = cfg
    inferencer._map_aoi_coords = lambda *_args, **_kwargs: (5794, 3485)
    image = np.full((4384, 6576), 180, dtype=np.uint8)
    image[3485 - 32:3485 + 33, 5794 - 32:5794 + 33] = 17
    polygon = np.array(
        [[816.1, 289.7], [5973.3, 297.7], [5967.4, 3527.7], [821.1, 3514.9]], dtype=np.float32,
    )
    result = ImageResult(
        image_path=Path("YQ512H230C41W0F00000133258.tif"), image_size=(6576, 4384),
        otsu_bounds=(818, 296, 5972, 3524), raw_bounds=(818, 296, 5972, 3524),
        exclusion_regions=[], tiles=[], excluded_tile_count=0, processed_tile_count=0,
        processing_time=0.0, panel_polygon=polygon,
    )
    count = inferencer._create_aoi_centered_tiles_v2(
        image, result,
        [AOIReportDefect(defect_code="CM00", product_x=1850, product_y=1184, image_prefix="W0F00000")],
        (1920, 1200), PreprocessConfig(tile_size=512),
    )
    assert count == 1
    tile = result.tiles[0]
    assert (tile.x, tile.y) == (5456, 3015)
    assert (tile.aoi_tile_shift_dx, tile.aoi_tile_shift_dy) == (-82, -214)
    assert (tile.aoi_product_x, tile.aoi_product_y, tile.aoi_defect_code) == (1850, 1184, "CM00")
    assert (tile.aoi_image_x, tile.aoi_image_y) == (5794, 3485)
    assert tile.zone == "edge"
    assert np.count_nonzero(tile.image == 17) == 65 * 65
    assert np.count_nonzero(tile.original_image == 17) == 65 * 65


def test_v2_aoi_logged_bottom_right_tile_fits_polygon_with_extended_raw_bounds():
    cfg = CAPIConfig()
    cfg.is_new_architecture = True
    cfg.tile_size = 512
    cfg.enable_panel_polygon = True
    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.config = cfg

    polygon = np.array(
        [[851.4, 887.6], [5912.0, 893.8], [5920.0, 4066.2], [832.3, 4055.2]],
        dtype=np.float32,
    )
    image = np.zeros((4384, 6576), dtype=np.uint8)
    cv2.fillConvexPoly(image, polygon.astype(np.int32), 180)
    result = ImageResult(
        image_path=Path("WGF50500_152418.tif"),
        image_size=(6576, 4384),
        otsu_bounds=(838, 898, 5914, 4062),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        raw_bounds=(840, 898, 5919, 4366),
        panel_polygon=polygon,
    )
    # This regression isolates ROI correction using the original logged anchor.
    # Product-coordinate remapping has its own tests and may correct this raw box.
    inferencer._map_aoi_coords = lambda *_args, **_kwargs: (5868, 3990)

    created = inferencer._create_aoi_centered_tiles_v2(
        image=image,
        result=result,
        defects=[AOIReportDefect(
            defect_code="PCDCL", product_x=1901, product_y=1070,
            image_prefix="WGF50500",
        )],
        product_resolution=(1920, 1200),
        pre_cfg=PreprocessConfig(tile_size=512),
    )

    assert created == 1
    tile = result.tiles[0]
    assert (tile.aoi_product_x, tile.aoi_product_y) == (1901, 1070)
    assert (tile.aoi_image_x, tile.aoi_image_y) == (5868, 3990)
    assert tile.aoi_tile_shift_dx < 0 and tile.aoi_tile_shift_dy < 0
    assert tile.x <= tile.aoi_image_x <= tile.x + 511
    assert tile.y <= tile.aoi_image_y <= tile.y + 511
    assert tile.zone == "edge"
    assert tile.image.shape == (512, 512)
    assert np.mean(tile.image == 0) < 0.001


def test_resolve_inward_polygon_tile_pushes_edge_roi_inside_product():
    polygon = np.array(
        [[500, 500], [1900, 500], [1900, 1500], [500, 1500]],
        dtype=np.float32,
    )

    tx, ty, coverage, shifted = resolve_inward_polygon_tile(
        anchor_xy=(550, 1000),
        polygon=polygon,
        image_shape=(2000, 2400),
        tile_size=512,
    )

    assert shifted is True
    assert tx >= 500
    assert coverage >= 0.999
    assert ty <= 1000 <= ty + 512


def test_resolve_inward_polygon_tile_can_keep_anchor_inside():
    polygon = np.array(
        [[700, 500], [1900, 500], [1700, 1500], [500, 1500]],
        dtype=np.float32,
    )

    tx, ty, _coverage, _shifted = resolve_inward_polygon_tile(
        anchor_xy=(500, 1499),
        polygon=polygon,
        image_shape=(2000, 2400),
        tile_size=512,
        keep_anchor_inside=True,
    )

    assert tx <= 500 <= tx + 511
    assert ty <= 1499 <= ty + 511


def test_v2_aoi_coord_tile_uses_inward_polygon_roi():
    cfg = CAPIConfig()
    cfg.is_new_architecture = True
    cfg.tile_size = 512
    cfg.enable_panel_polygon = True

    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.config = cfg

    image = np.zeros((2000, 2400), dtype=np.uint8)
    image[500:1500, 500:1900] = 180
    polygon = np.array(
        [[500, 500], [1900, 500], [1900, 1500], [500, 1500]],
        dtype=np.float32,
    )
    result = ImageResult(
        image_path=Path("G0F00000_test.png"),
        image_size=(2400, 2000),
        otsu_bounds=(500, 500, 1900, 1500),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        raw_bounds=(500, 500, 1900, 1500),
        panel_polygon=polygon,
    )

    created = inferencer._create_aoi_centered_tiles_v2(
        image=image,
        result=result,
        defects=[
            AOIReportDefect(
                defect_code="D01",
                product_x=50,
                product_y=500,
                image_prefix="G0F00000",
            )
        ],
        product_resolution=(1400, 1000),
        pre_cfg=PreprocessConfig(tile_size=512),
    )

    assert created == 1
    tile = result.tiles[0]
    assert tile.zone == "edge"
    assert tile.x >= 500
    assert tile.image.shape == (512, 512)
    assert int(tile.image.min()) > 0
    assert tile.aoi_image_x == 550
    assert tile.aoi_image_y == 1000
    assert tile.aoi_tile_shift_dx == tile.x - (550 - 256)
    assert tile.aoi_tile_shift_dy == tile.y - (1000 - 256)
    assert tile.aoi_tile_shift_dx > 0


def test_v2_aoi_coord_uses_polygon_when_raw_bounds_map_outside_panel():
    """Panel 外輪廓拉長 raw bounds 時，G/W 與 B0F 都應回到同一個 panel 座標。"""
    cfg = CAPIConfig()
    cfg.is_new_architecture = True
    cfg.tile_size = 512
    cfg.enable_panel_polygon = True

    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.config = cfg

    image = np.full((4400, 5500), 180, dtype=np.uint8)
    polygon = np.array(
        [
            [1003.3, 548.3],
            [5214.0, 491.7],
            [5245.0, 2861.1],
            [1033.7, 2913.9],
        ],
        dtype=np.float32,
    )
    raw_bounds = (1003, 492, 5245, 4384)
    product_resolution = (1920, 1080)
    defect = AOIReportDefect(
        defect_code="PCDK2",
        product_x=1572,
        product_y=933,
        image_prefix="W0F00000",
    )

    assert inferencer._map_aoi_coords(
        defect.product_x,
        defect.product_y,
        raw_bounds,
        product_resolution,
    ) == (4476, 3854)

    for is_skip_file, expected_zone in ((False, "inner"), (True, "bright_spot")):
        result = ImageResult(
            image_path=Path("B0F00000_test.png" if is_skip_file else "W0F00000_test.png"),
            image_size=(5500, 4400),
            otsu_bounds=(1003, 491, 5245, 2914),
            exclusion_regions=[],
            tiles=[],
            excluded_tile_count=0,
            processed_tile_count=0,
            processing_time=0.0,
            raw_bounds=raw_bounds,
            panel_polygon=polygon,
        )

        created = inferencer._create_aoi_centered_tiles_v2(
            image=image,
            result=result,
            defects=[defect],
            product_resolution=product_resolution,
            pre_cfg=PreprocessConfig(tile_size=512),
            is_skip_file=is_skip_file,
        )

        assert created == 1
        tile = result.tiles[0]
        assert (tile.aoi_image_x, tile.aoi_image_y) == (4476, 2548)
        assert (tile.x, tile.y) == (4220, 2292)
        assert (tile.aoi_tile_shift_dx, tile.aoi_tile_shift_dy) == (0, 0)
        assert tile.valid_ratio == 1.0
        assert tile.zone == expected_zone


def test_v2_aoi_coord_uses_polygon_when_wrong_raw_mapping_is_still_inside_panel():
    """Raw bounds 被下方字樣拉長時，不能因錯誤座標仍在 panel 內就沿用。"""
    cfg = CAPIConfig()
    cfg.is_new_architecture = True
    cfg.tile_size = 512
    cfg.enable_panel_polygon = True

    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.config = cfg

    image = np.full((4400, 5500), 180, dtype=np.uint8)
    polygon = np.array(
        [
            [1003.3, 548.3],
            [5214.0, 491.7],
            [5245.0, 2861.1],
            [1033.7, 2913.9],
        ],
        dtype=np.float32,
    )
    raw_bounds = (1003, 492, 5245, 4384)
    product_resolution = (1920, 1080)
    defect = AOIReportDefect(
        defect_code="C1111",
        product_x=1049,
        product_y=645,
        image_prefix="WGF50500",
    )

    raw_mapped = inferencer._map_aoi_coords(
        defect.product_x,
        defect.product_y,
        raw_bounds,
        product_resolution,
    )
    assert raw_mapped == (3320, 2816)
    assert cv2.pointPolygonTest(polygon, raw_mapped, True) > 0

    result = ImageResult(
        image_path=Path("WGF50500_test.png"),
        image_size=(5500, 4400),
        otsu_bounds=(1003, 491, 5245, 2914),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        raw_bounds=raw_bounds,
        panel_polygon=polygon,
    )

    created = inferencer._create_aoi_centered_tiles_v2(
        image=image,
        result=result,
        defects=[defect],
        product_resolution=product_resolution,
        pre_cfg=PreprocessConfig(tile_size=512),
    )

    assert created == 1
    tile = result.tiles[0]
    assert (tile.aoi_image_x, tile.aoi_image_y) == (3321, 1931)
    assert (tile.x, tile.y) == (3065, 1675)
    assert (tile.aoi_tile_shift_dx, tile.aoi_tile_shift_dy) == (0, 0)
    assert tile.zone == "inner"


def test_v2_aoi_top_edge_locks_inward_shift_to_y_axis():
    cfg = CAPIConfig()
    cfg.is_new_architecture = True
    cfg.tile_size = 512
    cfg.enable_panel_polygon = True

    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.config = cfg

    image = np.full((2600, 5000), 180, dtype=np.uint8)
    polygon = np.array(
        [[500, 450], [4500, 350], [4500, 2300], [500, 2300]],
        dtype=np.float32,
    )
    result = ImageResult(
        image_path=Path("W0F00000_test.png"),
        image_size=(5000, 2600),
        otsu_bounds=(500, 300, 4500, 2300),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        raw_bounds=(500, 300, 4500, 2300),
        panel_polygon=polygon,
    )

    created = inferencer._create_aoi_centered_tiles_v2(
        image=image,
        result=result,
        defects=[
            AOIReportDefect(
                defect_code="PCDK2",
                product_x=3000,
                product_y=100,
                image_prefix="W0F00000",
            )
        ],
        product_resolution=(4000, 2000),
        pre_cfg=PreprocessConfig(tile_size=512),
    )

    assert created == 1
    tile = result.tiles[0]
    assert tile.zone == "edge"
    assert tile.aoi_image_x == 3500
    assert tile.aoi_image_y == 400
    assert tile.aoi_tile_shift_dx == 0
    assert tile.aoi_tile_shift_dy > 0
    assert tile.x == tile.aoi_image_x - cfg.tile_size // 2
    assert tile.y <= tile.aoi_image_y <= tile.y + tile.height - 1


def test_v2_aoi_anchor_just_outside_half_tile_routes_to_inner():
    cfg = CAPIConfig()
    cfg.is_new_architecture = True
    cfg.tile_size = 512
    cfg.otsu_offset = 5
    cfg.enable_panel_polygon = True

    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.config = cfg

    image = np.full((4384, 6576), 180, dtype=np.uint8)
    raw_bounds = (344, 631, 5837, 3706)
    result = ImageResult(
        image_path=Path("R0F00000_test.png"),
        image_size=(6576, 4384),
        otsu_bounds=raw_bounds,
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        raw_bounds=raw_bounds,
        panel_polygon=None,
    )

    created = inferencer._create_aoi_centered_tiles_v2(
        image=image,
        result=result,
        defects=[
            AOIReportDefect(
                defect_code="PCE07",
                product_x=565,
                product_y=259,
                image_prefix="R0F00000",
            )
        ],
        product_resolution=(raw_bounds[2] - raw_bounds[0], raw_bounds[3] - raw_bounds[1]),
        pre_cfg=PreprocessConfig(tile_size=512),
    )

    assert created == 1
    tile = result.tiles[0]
    assert tile.aoi_image_x == 909
    assert tile.aoi_image_y == 890
    assert tile.x == 653
    assert tile.y == 634
    assert tile.zone == "inner"


def test_v2_aoi_bottom_anchor_just_past_half_tile_routes_to_inner():
    cfg = CAPIConfig()
    cfg.is_new_architecture = True
    cfg.tile_size = 512
    cfg.otsu_offset = 5
    cfg.enable_panel_polygon = True

    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.config = cfg

    image = np.full((4384, 6576), 180, dtype=np.uint8)
    raw_bounds = (0, 203, 6576, 3713)
    result = ImageResult(
        image_path=Path("W0F00000_test.png"),
        image_size=(6576, 4384),
        otsu_bounds=raw_bounds,
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        raw_bounds=raw_bounds,
        panel_polygon=None,
    )

    created = inferencer._create_aoi_centered_tiles_v2(
        image=image,
        result=result,
        defects=[
            AOIReportDefect(
                defect_code="PCDK2",
                product_x=1309,
                product_y=995,
                image_prefix="W0F00000",
            )
        ],
        product_resolution=(1920, 1080),
        pre_cfg=PreprocessConfig(tile_size=512),
    )

    assert created == 1
    tile = result.tiles[0]
    assert tile.aoi_image_y == 3436
    assert tile.y == 3180
    assert raw_bounds[3] - tile.aoi_image_y > cfg.tile_size // 2
    assert tile.zone == "inner"


def test_v2_aoi_zone_uses_defect_center_for_logged_w0f_case():
    cfg = CAPIConfig()
    cfg.is_new_architecture = True
    cfg.tile_size = 512
    cfg.enable_panel_polygon = True

    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.config = cfg

    image = np.full((4384, 6576), 180, dtype=np.uint8)
    polygon = np.array(
        [
            [549.9, 746.6],
            [6152.4, 725.6],
            [6107.9, 3823.6],
            [608.4, 3863.1],
        ],
        dtype=np.float32,
    )
    result = ImageResult(
        image_path=Path("W0F00000_144202.tif"),
        image_size=(6576, 4384),
        otsu_bounds=(550, 726, 6152, 3863),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        raw_bounds=(550, 726, 6152, 3863),
        panel_polygon=polygon,
    )
    mapped_points = iter([(5725, 2823), (1579, 1075)])
    inferencer._map_aoi_coords = lambda *_args, **_kwargs: next(mapped_points)

    created = inferencer._create_aoi_centered_tiles_v2(
        image=image,
        result=result,
        defects=[
            AOIReportDefect(
                defect_code="PCDB5",
                product_x=1774,
                product_y=722,
                image_prefix="W0F00000",
            ),
            AOIReportDefect(
                defect_code="PCDB5",
                product_x=353,
                product_y=120,
                image_prefix="W0F00000",
            ),
        ],
        product_resolution=(1920, 1080),
        pre_cfg=PreprocessConfig(tile_size=512),
    )

    assert created == 2
    assert [(tile.aoi_image_x, tile.aoi_image_y) for tile in result.tiles] == [
        (5725, 2823),
        (1579, 1075),
    ]
    assert [tile.zone for tile in result.tiles] == ["inner", "inner"]


def test_v2_aoi_near_polygon_fit_boundary_routes_to_edge():
    cfg = CAPIConfig()
    cfg.is_new_architecture = True
    cfg.tile_size = 512
    cfg.enable_panel_polygon = True

    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.config = cfg

    image = np.full((4384, 6576), 180, dtype=np.uint8)
    polygon = np.array(
        [
            [1206.9, 1377.4],
            [5331.2, 1397.0],
            [5317.3, 3965.8],
            [1200.1, 3957.2],
        ],
        dtype=np.float32,
    )
    result = ImageResult(
        image_path=Path("W0F00000_092442.tif"),
        image_size=(6576, 4384),
        otsu_bounds=(1200, 1377, 5331, 3966),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        raw_bounds=(1200, 1377, 5331, 3966),
        panel_polygon=polygon,
    )
    inferencer._map_aoi_coords = lambda *_args, **_kwargs: (1980, 1645)

    created = inferencer._create_aoi_centered_tiles_v2(
        image=image,
        result=result,
        defects=[
            AOIReportDefect(
                defect_code="BOMB_FORCE",
                product_x=363,
                product_y=124,
                image_prefix="W0F00000",
            )
        ],
        product_resolution=(1920, 1200),
        pre_cfg=PreprocessConfig(tile_size=512),
    )

    assert created == 1
    tile = result.tiles[0]
    assert (tile.x, tile.y) == (1724, 1389)
    assert tile.zone == "edge"


def test_v2_aoi_coord_tile_keeps_anchor_inside_when_inward_conflicts():
    cfg = CAPIConfig()
    cfg.is_new_architecture = True
    cfg.tile_size = 512
    cfg.enable_panel_polygon = True

    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.config = cfg

    image = np.full((2000, 2400), 180, dtype=np.uint8)
    polygon = np.array(
        [[700, 500], [1900, 500], [1700, 1500], [500, 1500]],
        dtype=np.float32,
    )
    result = ImageResult(
        image_path=Path("R0F00000_test.png"),
        image_size=(2400, 2000),
        otsu_bounds=(500, 500, 1900, 1500),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        raw_bounds=(500, 500, 1900, 1500),
        panel_polygon=polygon,
    )

    created = inferencer._create_aoi_centered_tiles_v2(
        image=image,
        result=result,
        defects=[
            AOIReportDefect(
                defect_code="C1111",
                product_x=0,
                product_y=999,
                image_prefix="R0F00000",
            )
        ],
        product_resolution=(1400, 1000),
        pre_cfg=PreprocessConfig(tile_size=512),
    )

    assert created == 1
    tile = result.tiles[0]
    assert tile.zone == "edge"
    assert tile.x <= tile.aoi_image_x <= tile.x + tile.width - 1
    assert tile.y <= tile.aoi_image_y <= tile.y + tile.height - 1
    assert tile.mask is not None
