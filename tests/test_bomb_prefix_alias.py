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
