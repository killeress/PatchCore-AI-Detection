"""Polygon-inward ROI tests for edge PatchCore samples."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def test_v2_aoi_outer_ring_tile_routes_to_edge_even_when_anchor_just_outside_half_tile():
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
