"""Polygon-inward ROI tests for edge PatchCore samples."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from capi_config import CAPIConfig
from capi_inference import AOIReportDefect, CAPIInferencer, ImageResult
from capi_preprocess import PreprocessConfig, resolve_inward_polygon_tile


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
