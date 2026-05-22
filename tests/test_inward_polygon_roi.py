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
