"""AOI crops and outgoing peaks must share correction of protruding raw bounds."""
import numpy as np
import pytest

from capi_preprocess import map_image_coord_to_product, map_product_coord_to_image


FIELD_BOUNDS = (1031, 1167, 5528, 4269)
FIELD_POLYGON = np.array([
    [1029.1, 1166.0], [5528.3, 1168.2], [5528.6, 3973.8], [1031.2, 3982.3],
], dtype=np.float32)
RESOLUTION = (1920, 1200)


def test_capi14_first_peak_maps_back_to_client_bomb():
    assert map_image_coord_to_product(1854, 1710, FIELD_BOUNDS, RESOLUTION) == (351, 210)
    assert map_image_coord_to_product(
        1854, 1710, FIELD_BOUNDS, RESOLUTION, FIELD_POLYGON,
    ) == (351, 232)
    anchor = map_product_coord_to_image(354, 235, FIELD_BOUNDS, RESOLUTION, FIELD_POLYGON)
    assert anchor[1] < 1774 - 40  # Old crop was displaced downward by the raw protrusion.


@pytest.mark.parametrize("point", [(354, 235), (1481, 346), (360, 534), (1077, 569), (1803, 702)])
def test_capi14_corrected_coordinates_round_trip(point):
    image = map_product_coord_to_image(*point, FIELD_BOUNDS, RESOLUTION, FIELD_POLYGON)
    assert map_image_coord_to_product(*image, FIELD_BOUNDS, RESOLUTION, FIELD_POLYGON) == point


@pytest.mark.parametrize("bounds", [(0, 0, 2100, 1200), (0, 0, 1920, 1300)])
def test_single_axis_protrusion_is_corrected_even_above_ninety_percent_area(bounds):
    polygon = np.array([[0, 0], [1920, 0], [1920, 1200], [0, 1200]], dtype=np.float32)
    assert map_product_coord_to_image(500, 300, bounds, RESOLUTION, polygon) == (500, 300)
    assert map_image_coord_to_product(500, 300, bounds, RESOLUTION, polygon) == (500, 300)


@pytest.mark.parametrize("polygon", [None, [], [[0, 0]] * 4, [[float('nan'), 0]] * 4])
def test_unusable_polygon_retains_linear_mapping(polygon):
    assert map_image_coord_to_product(1854, 1710, FIELD_BOUNDS, RESOLUTION, polygon) == (351, 210)
    assert map_product_coord_to_image(354, 235, FIELD_BOUNDS, RESOLUTION, polygon) == (1860, 1774)


def test_small_boundary_refinement_keeps_linear_coordinates():
    polygon = np.array([[1, 1], [1919, 1], [1919, 1199], [1, 1199]], dtype=np.float32)
    bounds = (0, 0, 1920, 1220)
    assert map_product_coord_to_image(500, 300, bounds, RESOLUTION, polygon) == (500, 305)
    assert map_image_coord_to_product(500, 305, bounds, RESOLUTION, polygon) == (500, 300)


def test_five_aoi_crops_move_up_while_edge_inward_shift_stays_zero():
    from pathlib import Path
    from capi_config import CAPIConfig
    from capi_inference import AOIReportDefect, CAPIInferencer, ImageResult
    from capi_preprocess import PreprocessConfig

    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.config = CAPIConfig()
    points = [(354, 235), (1481, 346), (360, 534), (1077, 569), (1803, 702)]
    expected_centers = [(1862, 1718), (4503, 1978), (1876, 2419), (3558, 2500), (5255, 2810)]
    result = ImageResult(
        image_path=Path("STANDARD_125214.tif"), image_size=(6576, 4384),
        raw_bounds=FIELD_BOUNDS, otsu_bounds=FIELD_BOUNDS, panel_polygon=FIELD_POLYGON,
        exclusion_regions=[], tiles=[], excluded_tile_count=0,
        processed_tile_count=0, processing_time=0,
    )
    count = inferencer._create_aoi_centered_tiles_v2(
        np.zeros((4384, 6576), dtype=np.uint8), result,
        [AOIReportDefect("PCDK2", x, y, "STANDARD") for x, y in points],
        RESOLUTION, PreprocessConfig(),
    )
    assert count == 5
    assert [tile.center for tile in result.tiles] == expected_centers
    assert [(t.aoi_image_x, t.aoi_image_y) for t in result.tiles] == expected_centers
    assert all((t.aoi_tile_shift_dx, t.aoi_tile_shift_dy) == (0, 0) for t in result.tiles)
