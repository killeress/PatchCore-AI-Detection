"""Bomb matching must use station lighting names consistently."""
from pathlib import Path

import numpy as np
import pytest

from capi_config import BombDefect, CAPIConfig
from capi_inference import AOIReportDefect, CAPIInferencer, ImageResult, TileInfo
from capi_station_adapter import create_station_adapter


AOI_POINTS = [(355, 237), (1484, 349), (358, 536), (1078, 571), (1801, 707)]
BOMB_POINTS = [(350, 231), (1475, 343), (356, 530), (1071, 565), (1798, 700)]


def make_inferencer(profile="capi"):
    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.config = CAPIConfig(
        bomb_match_tolerance=20,
        bomb_area_force_detection_enabled=True,
        bomb_defects=[BombDefect("U0F00000", "B01", "point", BOMB_POINTS)],
    )
    inferencer.station_adapter = create_station_adapter(profile)
    return inferencer


@pytest.mark.parametrize("client_prefix,report_prefix", [
    ("U0F00000", "U0F00000"), ("STANDARD", "STANDARD"),
])
def test_same_screen_report_covers_all_five_client_bombs(client_prefix, report_prefix, capsys):
    inferencer = make_inferencer()
    report = {report_prefix: [
        AOIReportDefect("PCDK2", x, y, report_prefix) for x, y in AOI_POINTS
    ]}
    bomb_info = dict(image_prefix=client_prefix, defect_type="point", coordinates=BOMB_POINTS)
    updated, added = inferencer._aoi_report_with_forced_client_bomb_coords(report, bomb_info)
    assert added == 0
    assert updated == report
    assert "AOI已涵蓋=5" in capsys.readouterr().out


@pytest.mark.parametrize("source", ["client", "config"])
def test_five_bombs_match_same_screen_without_absorbing_unrelated_aoi_point(source, capsys):
    inferencer = make_inferencer()
    # Identity image/product mapping isolates prefix matching from calibration.
    tiles = []
    for index, (x, y) in enumerate(AOI_POINTS + [(1000, 500)]):
        tile = TileInfo(index, x - 256, y - 256, 512, 512,
                        np.zeros((512, 512), dtype=np.uint8), zone="inner")
        tile.is_aoi_coord_tile = True
        tile.aoi_product_x, tile.aoi_product_y = x, y
        tile.aoi_image_x, tile.aoi_image_y = x, y
        tile.anomaly_peak_source = "aoi_real_region"
        # The unrelated tile has a peak on a bomb: the AOI guard must reject it.
        tile.anomaly_peak_x, tile.anomaly_peak_y = (x, y) if index < 5 else BOMB_POINTS[3]
        tiles.append(tile)
    result = ImageResult(
        image_path=Path("U0F00000_085943.tif"), image_size=(1920, 1080),
        otsu_bounds=(0, 0, 1920, 1080), raw_bounds=(0, 0, 1920, 1080),
        exclusion_regions=[], tiles=tiles, excluded_tile_count=0,
        processed_tile_count=len(tiles), processing_time=0.0,
        anomaly_tiles=[(tile, 0.8, None) for tile in tiles],
    )
    bomb_info = dict(image_prefix="U0F00000", defect_type="point", coordinates=BOMB_POINTS)
    if source == "config":
        inferencer.config.bomb_defects[0].image_prefix = "U0F00000"
    inferencer._apply_bomb_postprocess(
        [result], bomb_info if source == "client" else None, (1920, 1080),
    )
    assert [tile.is_bomb for tile in tiles] == [True] * 5 + [False]
    assert all(tile.bomb_defect_code == "B01" for tile in tiles[:5])
    log = capsys.readouterr().out
    assert log.count("BOMB distance") == 6
    assert "B01×5" in log
    assert bomb_info["image_prefix"] == "U0F00000"


@pytest.mark.parametrize("defect_type,coords", [
    ("point", [(350, 231)]), ("line", [(300, 231), (400, 231)]),
])
@pytest.mark.parametrize("prefix,expected", [
    ("STANDARD", False), ("U0F00000_085943", True), ("G0F00000", False),
])
def test_bomb_point_and_line_matching_preserves_screen(defect_type, coords, prefix, expected):
    inferencer = make_inferencer()
    bomb = BombDefect("U0F00000", "B01", defect_type, coords)
    matched, _ = inferencer.check_bomb_match(
        prefix, 355, 237, (0, 0, 1920, 1080),
        product_resolution=(1920, 1080), bomb_list=[bomb],
    )
    assert matched is expected


@pytest.mark.parametrize("profile", ["capi", "aapi"])
def test_stations_keep_u0f_separate_from_standard(profile):
    inferencer = make_inferencer(profile)
    assert not inferencer._aoi_prefix_matches("STANDARD", "U0F00000")
    matched, _ = inferencer.check_bomb_match(
        "STANDARD", 350, 231, (0, 0, 1920, 1080),
        product_resolution=(1920, 1080),
        bomb_list=[BombDefect("U0F00000", "B01", "point", BOMB_POINTS)],
    )
    assert not matched


@pytest.mark.parametrize("client_prefix", ["STANDARD", "Windows_BG", "WINDOWS_BG"])
@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_aapi_standard_line_bomb_matches_raw_testing_request(client_prefix, orientation, capsys):
    from capi_server import parse_request

    # Actual horizontal/vertical incidents: bomb endpoints use product units,
    # while AOI report X values use three-channel units and must be divided by 3.
    if orientation == "horizontal":
        glass_id, machine, timestamp = "YQ713C226G17", "AAPI2612", "144114"
        bomb_coords = [(1, 540), (1920, 540)]
        report_payload = (
            "W0F00000,CM00(03142,00963)STANDARD,CLH2(00592,00534)"
            "STANDARD,CLH2(01022,00546)STANDARD,CLH2(01275,00546)"
            "STANDARD,CDB6(01596,00540)STANDARD,CLH2(01998,00547)"
            "STANDARD,CDB6(02275,00540)STANDARD,CLH2(02820,00535)"
            "STANDARD,CLH2(03198,00548)STANDARD,CLH2(03501,00534)"
            "STANDARD,CLH2(03747,00546)STANDARD,CLH2(04293,00547)"
            "STANDARD,CLH2(04571,00547)STANDARD,CLH2(04893,00541)"
            "STANDARD,CM00(05431,00542)"
        )
        expected_points = [
            (197, 534), (340, 546), (425, 546), (532, 540), (666, 547),
            (758, 540), (940, 535), (1066, 548), (1167, 534), (1249, 546),
            (1431, 547), (1523, 547), (1631, 541), (1810, 542),
        ]
        outside_point = (960, 561)
    else:
        glass_id, machine, timestamp = "YQ71QN205F46", "AAPI2622", "154655"
        bomb_coords = [(960, 1), (960, 1200)]
        report_payload = (
            "W0F00000,CO05(00345,00289)W0F00000,CO05(00345,00289)"
            "STANDARD,CLV2(02873,00298)STANDARD,CLV2(02886,00297)"
            "STANDARD,CLV2(02872,00697)STANDARD,CLV2(02884,00697)"
            "STANDARD,CLV2(02870,00998)STANDARD,CLV2(02882,00996)"
            "White_Frame,CDB6(00601,01186)B8F00000,CLH2(00533,01186)"
        )
        expected_points = [
            (957, 298), (962, 297), (957, 697), (961, 697), (956, 998), (960, 996),
        ]
        outside_point = (981, 600)
    wire_coords = "/".join(str(value) for point in bomb_coords for value in point)
    request = parse_request(
        f"AOI@{glass_id};GN160JCEL250S;{machine};1920,1200;NG;"
        f"{client_prefix};({wire_coords});/image/{glass_id};{report_payload}"
    )
    inferencer = make_inferencer("aapi")
    inferencer.config.bomb_defects = [
        BombDefect("STANDARD", "B01", "line", bomb_coords),
    ]
    report = inferencer.station_adapter.parse_aoi_report(
        Path(request["image_dir"]), glass_id=request["glass_id"],
        machine_judgment=request["machine_judgment"],
        report_payload=request["aoi_report_payload"],
    )
    points = [(defect.x, defect.y) for defect in report["WINDOWS_BG"]]
    assert points == expected_points
    assert request["bomb_info"]["coordinates"] == bomb_coords

    # Synthetic maps and identity bounds isolate matching from GPU inference.
    # Include one neighboring real defect just outside the 20 px tolerance.
    tiles = []
    for index, (x, y) in enumerate(points + [outside_point]):
        tiles.append(TileInfo(
            index, x - 256, y - 256, 512, 512,
            np.zeros((512, 512), dtype=np.uint8), zone="inner",
            is_aoi_coord_tile=True, aoi_product_x=x, aoi_product_y=y,
            aoi_image_x=x, aoi_image_y=y,
            anomaly_peak_source="aoi_real_region", anomaly_peak_x=x,
            anomaly_peak_y=y, aoi_roi_mask=np.ones((512, 512), dtype=np.uint8),
        ))
    anomaly_map = np.zeros((512, 512), dtype=np.float32)
    anomaly_map[253:259, 20:492] = 0.8
    if orientation == "vertical":
        anomaly_map = anomaly_map.T.copy()
    result = ImageResult(
        image_path=Path(f"{glass_id}STANDARD{timestamp}.tif"),
        image_size=(1920, 1200), otsu_bounds=(0, 0, 1920, 1200),
        raw_bounds=(0, 0, 1920, 1200), exclusion_regions=[], tiles=tiles,
        excluded_tile_count=0, processed_tile_count=len(tiles), processing_time=0.0,
        anomaly_tiles=[(tile, 0.8, anomaly_map) for tile in tiles],
    )

    inferencer._apply_bomb_postprocess(
        [result], request["bomb_info"], request["resolution"],
    )

    assert [tile.is_bomb for tile in tiles] == [True] * len(points) + [False]
    assert all(tile.bomb_defect_code == "B01" for tile in tiles[:-1])
    assert f"B01×{len(points)}" in capsys.readouterr().out
    assert request["bomb_info"]["image_prefix"] == client_prefix
