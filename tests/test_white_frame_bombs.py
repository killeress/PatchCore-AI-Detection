"""Multi-segment WHITEFRA bombs apply to frame CV without suppressing other screens."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from capi_config import BombDefect, CAPIConfig
from capi_inference import CAPIInferencer, ImageResult
from capi_server import (
    _iter_qjpg_defect_records, _normalize_machine_judgment_for_bomb_only_panel,
    _white_frame_aoi_kwargs, _white_frame_image_result, aggregate_judgment,
    parse_request, results_to_db_data,
)
from capi_station_adapter import create_station_adapter
from capi_white_frame import inspect_white_frame_image


REQUEST = (
    "AOI@YQ61H5214D14;GN160JCEL250S;AAPI2612;1920,1200;NG;White_Frame;"
    "(1/500/1/550;1/800/1/850;1920/500/1920/550;1920/300/1920/350;1920/850/1920/900);"
    "/image/YQ61H5214D14;White_Frame,CM00(05714,00353)White_Frame,CM00(00020,00499)"
    "White_Frame,CM00(05714,00501)White_Frame,CM00(00020,00550)"
    "White_Frame,CM00(05714,00551)White_Frame,CM00(00022,00796)"
    "White_Frame,CM00(00022,00846)White_Frame,CM00(05714,00897)"
)
SEGMENTS = [[(1, 500), (1, 550)], [(1, 800), (1, 850)], [(1920, 500), (1920, 550)],
            [(1920, 300), (1920, 350)], [(1920, 850), (1920, 900)]]
POINTS = [(1904, 353), (6, 499), (1904, 501), (6, 550),
          (1904, 551), (7, 796), (7, 846), (1904, 897)]


def _inferencer():
    inf = CAPIInferencer.__new__(CAPIInferencer)
    inf.config = CAPIConfig(bomb_match_tolerance=20)
    inf.station_adapter = create_station_adapter("aapi")
    return inf


def test_client_multiline_bombs_preserve_independent_segments_and_report():
    parsed = parse_request(REQUEST)
    bombs = BombDefect.from_client_info(parsed["bomb_info"])
    assert parsed["bomb_info"]["defect_type"] == "line"
    assert [bomb.coordinates for bomb in bombs] == SEGMENTS
    assert all(bomb.image_prefix == "White_Frame" for bomb in bombs)
    inf = _inferencer()
    report = inf._parse_aoi_report_txt(
        Path(parsed["image_dir"]), glass_id=parsed["glass_id"], machine_judgment="NG",
        report_payload=parsed["aoi_report_payload"],
    )
    assert [(point.product_x, point.product_y) for point in report["WHITEFRA"]] == POINTS
    for point in POINTS:
        assert inf.check_bomb_match("WHITEFRA", 0, 0, (0, 0, 1920, 1200),
            product_resolution=(1920, 1200), bomb_list=bombs, product_coords=point)[0]
    for point in [(1, 650), (1920, 700), (960, 525)]:
        assert not inf.check_bomb_match("WHITEFRA", 0, 0, (0, 0, 1920, 1200),
            product_resolution=(1920, 1200), bomb_list=bombs, product_coords=point)[0]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_white_frame_bombs_do_not_exclude_regular_white_screen_points(version, monkeypatch):
    from test_bomb_aoi_mapping import apply_bomb_postprocess, make_tile

    inf = _inferencer()
    tiles = [make_tile(i, point, point) for i, point in enumerate(POINTS + [(6, 499), (1, 650)])]
    for tile in tiles:
        tile.aoi_source_prefix = "W0F00000"
    for index in (0, 2, 4, 7):
        tiles[index].is_aoi_coord_below_threshold = True
    amap = np.zeros((512, 512), np.float32)
    amap[250:262, 250:262] = .8
    result = ImageResult(
        image_path=Path("YQ61H5214D14W0F00000153923.tif"), image_size=(1920, 1200),
        otsu_bounds=(0, 0, 1920, 1200), raw_bounds=(0, 0, 1920, 1200),
        exclusion_regions=[], tiles=tiles, excluded_tile_count=0,
        processed_tile_count=len(tiles), processing_time=0,
        anomaly_tiles=[(tile, .8, amap) for tile in tiles], report_image_prefix="W0F00000",
    )
    apply_bomb_postprocess(inf, result, parse_request(REQUEST)["bomb_info"], version, monkeypatch)
    assert not any(tile.is_bomb for tile in tiles)
    assert aggregate_judgment([result])[0] == "NG"
    assert _iter_qjpg_defect_records([result], (1920, 1200), inf.config, bomb_only=True) == []


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_multiline_shape_consensus_does_not_borrow_from_another_segment(version, monkeypatch):
    from test_bomb_aoi_mapping import apply_bomb_postprocess, make_tile, make_result

    inf = _inferencer()
    tiles = [make_tile(i, point, point) for i, point in enumerate(
        [(200, 500), (400, 500), (600, 500), (1000, 1000)])]
    line = np.zeros((512, 512), np.float32)
    line[250:262, 64:448] = .8
    dot = np.zeros_like(line)
    dot[250:262, 250:262] = .8
    result = make_result(tiles)
    result.image_path = Path("GSTANDARD100000.tif")
    result.raw_bounds = result.otsu_bounds = (0, 0, 1920, 1200)
    result.panel_polygon = None
    result.anomaly_tiles = [(tile, .8, line if index < 3 else dot) for index, tile in enumerate(tiles)]
    bomb_info = {"image_prefix": "STANDARD", "defect_type": "line",
                 "coordinates": [(1, 500), (1920, 500), (1, 1000), (1920, 1000)]}
    monkeypatch.setattr(inf, "_apply_aoi_peak_postprocess", lambda _results: None)
    apply_bomb_postprocess(inf, result, bomb_info, version, monkeypatch)
    assert [tile.is_bomb for tile in tiles] == [True, True, True, False]


@pytest.mark.parametrize("extra", ["none", "separate", "extends"])
def test_cv_excludes_full_bomb_gaps_but_keeps_other_or_extended_gaps(tmp_path, extra):
    parsed = parse_request(REQUEST)
    image = np.zeros((800, 1200), np.uint8)
    cv2.rectangle(image, (180, 140), (1020, 660), 255, 7)
    for index, ((x, start), (_, end)) in enumerate(SEGMENTS):
        if extra == "extends" and index == 0:
            start, end = 450, 600  # Center matches, but the full defect does not.
        ix = 180 if x == 1 else 1020
        y1, y2 = round(140 + start * 520 / 1200), round(140 + end * 520 / 1200)
        cv2.rectangle(image, (ix - 12, y1), (ix + 12, y2), 0, -1)
    if extra == "separate":
        cv2.rectangle(image, (168, 422), (192, 443), 0, -1)
    path = tmp_path / "YQ61H5214D14White_Frame153927.png"
    assert cv2.imwrite(str(path), image)
    inf = _inferencer()
    # Scale the local crop with this smaller synthetic frame; production
    # 512 px crops do not reach corners from these side coordinates.
    inf.config.tile_size = 128
    report = inf._parse_aoi_report_txt(tmp_path, glass_id=parsed["glass_id"],
        machine_judgment="NG", report_payload=parsed["aoi_report_payload"])
    kwargs = _white_frame_aoi_kwargs(inf.station_adapter, report, inf.config, parsed["model_id"],
        bomb_info=parsed["bomb_info"], product_resolution=parsed["resolution"])
    inspection = inspect_white_frame_image(path, **kwargs)
    payload = inspection.payload
    assert payload["status_before_bomb"] == "NG"
    assert payload["bomb_excluded_count"] >= 4
    expected = "OK" if extra == "none" else "NG"
    assert payload["status"] == expected
    result = _white_frame_image_result(inspection)
    assert aggregate_judgment([result])[0] == expected
    assert _normalize_machine_judgment_for_bomb_only_panel("NG", [result]) == expected
    records = _iter_qjpg_defect_records([result], parsed["resolution"], inf.config)
    assert any(record.startswith("PCDK3") and record.endswith("WHITEFRA") for record in records)
    assert any(not record.startswith("PCDK3") for record in records) is (extra != "none")
    data = results_to_db_data([result], {})[0]
    assert data["is_ng"] == int(extra != "none")
    assert data["is_bomb"] == int(extra == "none")
    from capi_web import CAPIWebHandler
    CAPIWebHandler.init_jinja()
    html = CAPIWebHandler.jinja_env.get_template("_white_frame_result.html").render(
        wf=payload, img={"image_name": path.name}, detail={"id": 1})
    assert "炸彈已排除" in html
