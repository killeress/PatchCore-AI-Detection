"""AOI report row numbers stay stable when inference tiles are reordered."""
import json

import pytest

from capi_web import CAPIWebHandler
from capi_station_adapter import create_station_adapter


def _tile(tile_id, x, y, code="CM00"):
    return dict(tile_id=tile_id, id=tile_id + 100, is_aoi_coord=True,
                aoi_product_x=x, aoi_product_y=y, aoi_defect_code=code,
                x=0, y=0, width=512, height=512, score=0.8,
                is_anomaly=True, heatmap_path=f"/heatmaps/tile{tile_id}.png")


def _detail(coords, tiles, name="W0F00000.tif", prefix="W0F00000"):
    return dict(id=1, ai_judgment="NG", aoi_machine_coords=json.dumps({prefix: coords}),
                processing_seconds=1.0, image_prefix_labels={},
                images=[dict(image_name=name, image_path=name, tiles=tiles,
                             is_ng=True, max_score=0.8, inference_time_ms=1.0)])


@pytest.mark.parametrize("template", ["record_detail.html", "record_detail_v3.html"])
def test_report_numbers_match_screenshot_in_table_and_heatmaps(template):
    points = [(368, 131), (1551, 337), (288, 555), (1001, 558),
              (1772, 722), (1032, 1091)]
    coords = [dict(product_x=x, product_y=y) for x, y in points]
    tiles = [_tile(i, *points[i]) for i in [5, 1, 4, 0, 2, 3]]
    detail = _detail(coords, tiles)
    CAPIWebHandler._decorate_record_aoi_point_numbers(detail)
    assert [t["aoi_point_number"] for t in tiles] == [6, 2, 5, 1, 3, 4]
    assert [t["tile_id"] for t in tiles] == [5, 1, 4, 0, 2, 3]
    CAPIWebHandler.init_jinja()
    html = CAPIWebHandler.jinja_env.get_template(template).render(
        detail=detail, heatmap_base_dir="/heatmaps")
    for number in range(1, 7):
        assert html.count(f"AOI 點位 #{number}") == 3  # table, heading, image alt
    assert 'data-tile-id="0"' in html
    assert "AOI 點位 #0" not in html


def test_point_numbers_use_coordinates_not_tile_ids_and_preserve_missing_rows():
    coords = [dict(product_x=10, product_y=20),
              dict(product_x=30, product_y=40),
              dict(product_x=50, product_y=60)]
    tiles = [_tile(24, 50, 60), _tile(20, 10, 20), _tile(25, 99, 99),
             dict(tile_id=0, is_aoi_coord=False)]
    detail = _detail(coords, tiles, name="U0F00000083755.tif", prefix="U0F00000")
    CAPIWebHandler._decorate_record_aoi_point_numbers(detail)
    assert [t["aoi_point_number"] for t in tiles] == [3, 1, None, None]


def test_duplicate_coordinates_match_codes_then_creation_order():
    coords = [dict(product_x=10, product_y=20, defect_code=code)
              for code in ["CDB6", "CM00", "CM00"]]
    tiles = [_tile(7, 10, 20), _tile(5, 10, 20), _tile(6, 10, 20, "CDB6")]
    CAPIWebHandler._decorate_record_aoi_point_numbers(_detail(coords, tiles))
    assert [t["aoi_point_number"] for t in tiles] == [3, 2, 1]


def test_image_coordinates_and_white_frame_followup():
    coords = [dict(coordinate_space="image", product_x=-1, source_x=100, source_y=200)]
    tile = _tile(20, -1, -1)
    tile.update(aoi_image_x=100, aoi_image_y=200, is_white_frame_followup=True)
    detail = _detail(coords, [tile], prefix="WHITEFRA")
    CAPIWebHandler._decorate_record_aoi_point_numbers(detail, create_station_adapter("aapi"))
    assert tile["aoi_point_number"] == 1


@pytest.mark.parametrize("raw", [None, "", "invalid", "[]", "{}"])
def test_missing_report_does_not_invent_a_point_number(raw):
    tile = _tile(0, 10, 20)
    detail = _detail([], [tile])
    detail["aoi_machine_coords"] = raw
    CAPIWebHandler._decorate_record_aoi_point_numbers(detail)
    assert tile["aoi_point_number"] is None
