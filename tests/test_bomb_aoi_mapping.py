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


@pytest.mark.parametrize("source", ["client", "config"])
@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("product_point,line_shape,expected", [
    pytest.param((1053, 1122), True, False, id="tile13-line-peak"),
    pytest.param((1053, 1122), False, False, id="tile13-point-peak"),
    pytest.param((980, 1122), False, True, id="at-line-tolerance"),
    pytest.param((981, 1122), False, False, id="outside-line-tolerance"),
    pytest.param((960, 1020), False, True, id="at-endpoint-tolerance"),
    pytest.param((960, 1021), False, False, id="outside-endpoint-tolerance"),
])
def test_line_consensus_requires_aoi_position_even_when_peak_matches(
    source, version, product_point, line_shape, expected, capsys, monkeypatch,
):
    from capi_server import results_to_db_data

    inferencer = make_inferencer()
    # Synthetic endpoints: the incident log contains tile #13's coordinates,
    # but does not record the client bomb's endpoints.
    line_end_y = 1000 if product_point[0] == 960 else 1200
    bomb_points = [(960, 1), (960, line_end_y)]
    inferencer.config.bomb_defects = (
        [BombDefect("WGF50500", "B01", "line", bomb_points)]
        if source == "config" else []
    )
    bounds = (828, 902, 5906, 4068)
    line_map = np.zeros((512, 512), dtype=np.float32)
    line_map[64:448, 250:262] = 1.0
    point_map = np.zeros_like(line_map)
    point_map[250:262, 250:262] = 1.0

    tiles = [
        make_tile(i, (960, y), inferencer._map_aoi_coords(960, y, bounds, RESOLUTION))
        for i, y in enumerate((200, 400, 600))
    ]
    candidate = make_tile(
        12, product_point, inferencer._map_aoi_coords(*product_point, bounds, RESOLUTION),
    )
    if product_point == (1053, 1122):
        candidate.x, candidate.y = 3358, 3555
        candidate.aoi_image_x, candidate.aoi_image_y = 3614, 3867
        candidate.aoi_tile_shift_dy = -56
    # All candidates have a nearby peak on the bomb line. Only the original
    # AOI product position can distinguish #13 from the confirmed line.
    candidate.anomaly_peak_x, candidate.anomaly_peak_y = inferencer._map_aoi_coords(
        960, min(product_point[1], line_end_y), bounds, RESOLUTION,
    )
    candidate_map = line_map if line_shape else point_map
    result = make_result(tiles + [candidate])
    result.image_path = Path("WGF50500_121835.tif")
    result.raw_bounds = result.otsu_bounds = bounds
    result.panel_polygon = None
    result.anomaly_tiles = [(tile, 0.6, line_map) for tile in tiles]
    result.anomaly_tiles.append((candidate, 0.4594, candidate_map))
    # Isolate bomb postprocessing from the earlier AOI peak-selection stage.
    monkeypatch.setattr(inferencer, "_apply_aoi_peak_postprocess", lambda _results: None)
    bomb_info = dict(image_prefix="WGF50500", defect_type="line", coordinates=bomb_points)

    apply_bomb_postprocess(
        inferencer, result, bomb_info if source == "client" else None, version, monkeypatch,
    )

    code = "UNKNOWN" if source == "client" else "B01"
    assert all(tile.is_bomb and tile.bomb_defect_code == code for tile in tiles)
    assert candidate.is_bomb is expected
    assert candidate.bomb_defect_code == (code if expected else "")
    assert results_to_db_data([result], {})[0]["is_ng"] == int(not expected)
    assert f"{code}×{3 + int(expected)}" in capsys.readouterr().out


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


def make_overlapping_point_bomb_case(source, map_size=512):
    """YQ6280211D47's two inward crops share the same corner bomb."""
    inferencer = make_inferencer()
    bomb_points = [(11, 11), (960, 11), (11, 540), (1909, 540), (960, 1069)]
    inferencer.config.bomb_defects = (
        [BombDefect("WGF50500", "B01", "point", bomb_points)]
        if source == "config" else []
    )
    tiles, entries = [], []
    for tid, origin, product, anchor, score in [
        (0, (851, 884), (104, 49), (1107, 1017), 0.8156),
        (10, (861, 884), (11, 11), (861, 917), 0.744),
    ]:
        tile = make_tile(tid, product, anchor)
        tile.x, tile.y = origin
        tile.zone = "edge"
        tile.aoi_defect_code = "PCDK2" if tid == 0 else "BOMB_FORCE"
        tile.aoi_tile_shift_dx = tile.x - (anchor[0] - 256)
        tile.aoi_tile_shift_dy = tile.y - (anchor[1] - 256)
        tile.anomaly_peak_source = ""
        amap = np.zeros((map_size, map_size), dtype=np.float32)
        px = int((861 - tile.x) * map_size / 512)
        py = int((917 - tile.y) * map_size / 512)
        amap[max(0, py - 2):py + 3, max(0, px - 2):px + 3] = score * 0.98
        amap[py, px] = score
        tiles.append(tile)
        entries.append((tile, score, amap))
    result = make_result(tiles)
    result.image_path = Path("WGF50500_140955.tif")
    result.raw_bounds = result.otsu_bounds = (838, 894, 5916, 4058)
    result.panel_polygon = np.array([
        [850.9, 883.4], [5911.1, 890.0], [5921.3, 4062.4], [830.1, 4052.4],
    ], dtype=np.float32)
    result.anomaly_tiles = entries
    bomb_info = dict(image_prefix="WGF50500", defect_type="point", coordinates=bomb_points)
    return inferencer, result, bomb_info if source == "client" else None


@pytest.mark.parametrize("source", ["client", "config"])
@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("map_size", [128, 512])
def test_overlapping_aoi_crops_with_only_point_bomb_hotspot_are_bombs(
    source, version, map_size, monkeypatch, capsys, caplog,
):
    from capi_server import _normalize_machine_judgment_for_bomb_only_panel, results_to_db_data
    from capi_web import _target_tiles_for_within_spec

    inferencer, result, bomb_info = make_overlapping_point_bomb_case(source, map_size)
    caplog.set_level("INFO", logger="capi.inference")
    # Production peak selection falls back to (104,49), away from the bomb.
    inferencer._apply_aoi_peak_postprocess([result])
    assert result.tiles[0].anomaly_peak_source == "aoi_report_fallback"

    apply_bomb_postprocess(inferencer, result, bomb_info, version, monkeypatch)

    assert all(tile.is_bomb for tile in result.tiles)
    code = "UNKNOWN" if source == "client" else "B01"
    assert all(tile.bomb_defect_code == code for tile in result.tiles)
    stored = results_to_db_data([result], {})[0]
    assert stored["is_ng"] == 0
    assert stored["is_bomb"] == 1
    assert _target_tiles_for_within_spec(stored) == []
    assert _normalize_machine_judgment_for_bomb_only_panel("NG", [result]) == "OK"
    log = capsys.readouterr().out
    assert "BOMB_REGIONS" in caplog.text
    assert f"{code}×2" in log


@pytest.mark.parametrize("source", ["client", "config"])
@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("cached_regions", [False, True])
def test_point_bomb_cannot_hide_weaker_aoi_hotspot(
    source, version, cached_regions, monkeypatch,
):
    from capi_server import _iter_qjpg_defect_records, _normalize_machine_judgment_for_bomb_only_panel, results_to_db_data

    inferencer, result, bomb_info = make_overlapping_point_bomb_case(source)
    tile, _score, amap = result.anomaly_tiles[0]
    # This weaker independent defect is below the global top-percent cutoff.
    amap[130:137, 253:260] = 0.20
    amap[133, 256] = 0.25
    if cached_regions:
        seed, radius, min_score = inferencer._aoi_center_seed_for_tile(tile, amap)
        _real, _peak, _iou, details, binary, _labels = inferencer.check_dust_per_region(
            np.zeros_like(amap, dtype=np.uint8), amap,
            top_percent=inferencer.config.dust_heatmap_top_percent,
            force_include_yx=seed, force_include_radius=radius,
            force_include_min_score=min_score,
        )
        tile.dust_region_details, tile.dust_heatmap_binary = details, binary
    inferencer._apply_aoi_peak_postprocess([result])

    apply_bomb_postprocess(inferencer, result, bomb_info, version, monkeypatch)

    assert tile.is_bomb is False
    assert tile.bomb_defect_code == ""
    assert (tile.anomaly_peak_x, tile.anomaly_peak_y) == (1107, 1017)
    assert result.tiles[1].is_bomb is True
    assert results_to_db_data([result], {})[0]["is_ng"] == 1
    assert _normalize_machine_judgment_for_bomb_only_panel("NG", [result]) == "NG"
    records = _iter_qjpg_defect_records([result], RESOLUTION, inferencer.config)
    real_records = [record for record in records if record.startswith("PCDK2")]
    assert real_records == ["PCDK20010200047WGF50500"]


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("on_dust", [False, True])
def test_point_bomb_aoi_anchor_does_not_hide_other_non_dust_region(
    version, on_dust, monkeypatch,
):
    from capi_server import results_to_db_data

    inferencer, result, bomb_info = make_overlapping_point_bomb_case("client")
    tile, score, amap = result.anomaly_tiles[1]
    amap[300:307, 300:307] = score
    tile.dust_mask = np.zeros_like(amap, dtype=np.uint8)
    if on_dust:
        tile.dust_mask[300:307, 300:307] = 255
    result.tiles = [tile]
    result.anomaly_tiles = [(tile, score, amap)]

    apply_bomb_postprocess(inferencer, result, bomb_info, version, monkeypatch)

    assert tile.is_bomb is on_dust
    assert results_to_db_data([result], {})[0]["is_ng"] == int(not on_dust)
    assert {r["bomb_status"] for r in tile.bomb_region_diagnostics["regions"]} == (
        {"BOMB", "DUST"} if on_dust else {"BOMB", "REAL_NG"}
    )
    if not on_dust:
        assert (tile.anomaly_peak_x, tile.anomaly_peak_y) == (1161, 1184)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_connected_hot_region_extending_past_point_bomb_tolerance_keeps_ng(version, monkeypatch):
    inferencer, result, bomb_info = make_overlapping_point_bomb_case("client")
    tile, score, amap = result.anomaly_tiles[0]
    # One connected component has both bomb and non-bomb hot pixels.
    amap[33, 10:257] = score * 0.98
    amap[33:134, 256] = score * 0.98
    amap[133, 256] = score * 0.98

    apply_bomb_postprocess(inferencer, result, bomb_info, version, monkeypatch)

    assert tile.is_bomb is False
    assert tile.bomb_defect_code == ""


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_point_bomb_regions_follow_polygon_corrected_coordinates(version, monkeypatch):
    inferencer = make_inferencer()
    anchor = inferencer._map_aoi_coords(445, 269, RAW_BOUNDS, RESOLUTION, POLYGON)
    bomb_anchor = inferencer._map_aoi_coords(350, 231, RAW_BOUNDS, RESOLUTION, POLYGON)
    tile = make_tile(0, (445, 269), anchor)
    amap = np.zeros((512, 512), dtype=np.float32)
    px, py = bomb_anchor[0] - tile.x, bomb_anchor[1] - tile.y
    amap[py - 2:py + 3, px - 2:px + 3] = 0.8
    result = make_result([tile])
    result.anomaly_tiles = [(tile, 0.8, amap)]
    bomb_info = dict(image_prefix="STANDARD", defect_type="point", coordinates=[(350, 231)])

    apply_bomb_postprocess(inferencer, result, bomb_info, version, monkeypatch)

    assert tile.is_bomb is True
    assert tile.anomaly_peak_source == "bomb_region"


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_point_bomb_regions_do_not_suppress_two_stage_rescued_defect(version, monkeypatch):
    inferencer, result, bomb_info = make_overlapping_point_bomb_case("client")
    tile = result.tiles[0]
    tile.dust_two_stage_features = [{"abs_pos": (256, 133), "is_dust": False, "area": 25}]
    tile.anomaly_peak_source = "aoi_real_region"
    tile.anomaly_peak_x, tile.anomaly_peak_y = tile.aoi_image_x, tile.aoi_image_y

    apply_bomb_postprocess(inferencer, result, bomb_info, version, monkeypatch)

    assert tile.is_bomb is False
    assert tile.bomb_remaining_points is None


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("offset,expected", [
    ((20, 20), True), ((21, 0), False), ((0, 21), False),
])
def test_point_bomb_region_coverage_keeps_product_pixel_tolerance(version, offset, expected, monkeypatch):
    inferencer = make_inferencer()
    tile = make_tile(0, (104, 49), (104, 49))
    tile.x = tile.y = 0
    amap = np.zeros((512, 512), dtype=np.float32)
    amap[11, 11] = 0.8
    amap[11 + offset[1], 11 + offset[0]] = 0.8
    result = make_result([tile])
    result.raw_bounds = result.otsu_bounds = (0, 0, *RESOLUTION)
    result.panel_polygon = None
    result.anomaly_tiles = [(tile, 0.8, amap)]
    bomb_info = dict(image_prefix="STANDARD", defect_type="point", coordinates=[(11, 11)])

    apply_bomb_postprocess(inferencer, result, bomb_info, version, monkeypatch)

    assert tile.is_bomb is expected


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("cached_regions", [False, True])
@pytest.mark.parametrize("case", ["bomb_only", "mixed", "partial"])
def test_point_bomb_heatmap_displays_final_region_decisions(
    version, cached_regions, case, monkeypatch, tmp_path,
):
    import cv2
    from copy import deepcopy
    from capi_heatmap import HeatmapManager

    inferencer, result, bomb_info = make_overlapping_point_bomb_case("client")
    tile, score, amap = result.anomaly_tiles[0]
    if case == "mixed":
        amap[300:307, 300:307] = score
    elif case == "partial":
        # A connected bomb core extends outside the tolerance box.
        amap[33, 10:257] = score
    tile.omit_crop_image = tile.image.copy()
    tile.dust_mask = np.zeros_like(amap, dtype=np.uint8)
    tile.dust_detail_text = "PER_REGION: stale dust-stage result -> REAL_NG"
    if cached_regions:
        seed, radius, min_score = inferencer._aoi_center_seed_for_tile(tile, amap)
        _, _, _, details, binary, _ = inferencer.check_dust_per_region(
            tile.dust_mask, amap,
            top_percent=inferencer.config.dust_heatmap_top_percent,
            force_include_yx=seed, force_include_radius=radius,
            force_include_min_score=min_score,
        )
        tile.dust_region_details, tile.dust_heatmap_binary = details, binary
    dust_before = deepcopy(tile.dust_region_details)

    apply_bomb_postprocess(inferencer, result, bomb_info, version, monkeypatch)

    assert tile.is_bomb is (case == "bomb_only")
    assert tile.dust_region_details == dust_before
    diagnostics = tile.bomb_region_diagnostics
    statuses = [r["bomb_status"] for r in diagnostics["regions"]]
    assert sorted(statuses) == {
        "bomb_only": ["BOMB"], "mixed": ["BOMB", "REAL_NG"],
        "partial": ["PARTIAL_BOMB"],
    }[case]
    assert bool(tile.bomb_remaining_points) is (case != "bomb_only")

    captured = []
    real_put_text = cv2.putText

    def capture(img, text, *args, **kwargs):
        captured.append(str(text))
        return real_put_text(img, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture)
    output = HeatmapManager(tmp_path, save_format="png").save_tile_heatmap(
        tmp_path, case, tile.tile_id, tile.image, amap, score, tile_info=tile,
    )
    assert cv2.imread(output) is not None
    text = "\n".join(captured)
    assert "stale dust-stage" not in text
    assert "Final: M=BOMB R=NG G=DUST" in text
    if case == "partial":
        assert "NG (partial BOMB)" in captured
        assert "BOMB:0 Partial:1 Remaining NG:1 Dust:0" in text
    else:
        assert "BOMB" in captured
        assert "BOMB excluded:" in text
        assert ("REAL_NG" in captured) is (case == "mixed")
    # Reusing a tile without point-bomb evidence must not show old diagnostics.
    inferencer._match_aoi_point_bomb_regions(result, tile, None, [], RESOLUTION, {})
    assert tile.bomb_region_diagnostics is None


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("source", ["client", "config"])
@pytest.mark.parametrize("map_size", [128, 512])
def test_aoi_bomb_priority_ignores_other_region_in_shifted_crop(
    version, enabled, source, map_size, monkeypatch,
):
    import json
    from capi_server import (results_to_db_data, _iter_qjpg_defect_records,
                             _normalize_machine_judgment_for_bomb_only_panel)
    from capi_tile_diagnostics import decision_evidence
    from capi_web import _target_tiles_for_within_spec

    inferencer, result, bomb_info = make_overlapping_point_bomb_case(source, map_size)
    inferencer.config.aoi_bomb_priority_enabled = enabled
    tile, score, amap = result.anomaly_tiles[1]
    lo, hi = int(map_size * .58), int(map_size * .60)
    # Stronger, unrelated hot spot must not replace the original AOI anchor.
    amap[lo:hi, lo:hi] = score + .1
    result.tiles, result.anomaly_tiles = [tile], [(tile, score + .1, amap)]
    inferencer._apply_aoi_peak_postprocess([result])
    apply_bomb_postprocess(inferencer, result, bomb_info, version, monkeypatch)

    assert tile.is_bomb is enabled
    diagnostics = tile.bomb_region_diagnostics
    assert diagnostics["priority_applied"] is enabled
    assert {r["bomb_status"] for r in diagnostics["regions"]} == {
        "BOMB", "IGNORED_BY_AOI_BOMB" if enabled else "REAL_NG",
    }
    assert bool(tile.bomb_remaining_points) is (not enabled)
    stored = results_to_db_data([result], {})[0]
    assert stored["is_ng"] == int(not enabled)
    assert _normalize_machine_judgment_for_bomb_only_panel("NG", [result]) == ("OK" if enabled else "NG")
    records = _iter_qjpg_defect_records([result], RESOLUTION, inferencer.config)
    assert any(r.startswith("PCDK2") for r in records) is (not enabled)
    if enabled:
        assert abs(tile.anomaly_peak_x - tile.aoi_image_x) <= 4
        assert abs(tile.anomaly_peak_y - tile.aoi_image_y) <= 4
        assert _target_tiles_for_within_spec(stored) == []
        ctx = json.loads(stored["tiles"][0]["decision_context"])
        assert ctx["aoi_bomb_priority_applied"] is True
        assert ctx["aoi_bomb_priority_ignored_regions"] == 1
        assert ctx["aoi_bomb_priority_tolerance_product_px"] == 20
        assert decision_evidence(stored["tiles"][0])["title"] == "AOI 炸彈優先 → BOMB 排除"


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_aoi_bomb_priority_keeps_ng_on_other_tile(version, monkeypatch):
    from capi_server import results_to_db_data, _iter_qjpg_defect_records

    inferencer, result, bomb_info = make_overlapping_point_bomb_case("client")
    inferencer.config.aoi_bomb_priority_enabled = True
    # The first tile's AOI point is NOT the bomb; the second one's is.
    for tile, score, amap in result.anomaly_tiles:
        amap[130:137, 253:260] = score
    apply_bomb_postprocess(inferencer, result, bomb_info, version, monkeypatch)

    assert [t.is_bomb for t in result.tiles] == [False, True]
    assert results_to_db_data([result], {})[0]["is_ng"] == 1
    records = _iter_qjpg_defect_records([result], RESOLUTION, inferencer.config)
    assert sum(r.startswith("PCDK2") for r in records) == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_aoi_bomb_priority_accepts_region_tail_beyond_tolerance(enabled, monkeypatch):
    inferencer, result, bomb_info = make_overlapping_point_bomb_case("client")
    inferencer.config.aoi_bomb_priority_enabled = enabled
    tile, score, amap = result.anomaly_tiles[1]
    amap[33, :220] = score * .98
    amap[33, 0] = score
    apply_bomb_postprocess(inferencer, result, bomb_info, "v2", monkeypatch)
    assert tile.is_bomb is enabled
    assert tile.bomb_region_diagnostics["regions"][0]["bomb_status"] == (
        "BOMB" if enabled else "PARTIAL_BOMB"
    )


@pytest.mark.parametrize("anchor,peak,expected", [
    ((100, 100), (120, 120), True),  # Inclusive per-axis tolerance, not radius.
    ((100, 100), (121, 100), False),
    ((79, 100), (100, 100), False),
    ((100, 100), (150, 100), False),  # Peak at a DIFFERENT known bomb.
    ((-1, -1), (100, 100), False),
])
def test_aoi_bomb_priority_requires_anchor_and_peak_at_same_bomb(anchor, peak, expected):
    inferencer = make_inferencer()
    inferencer.config.aoi_bomb_priority_enabled = True
    tile = make_tile(0, anchor, anchor)
    tile.x = tile.y = 0
    result = make_result([tile])
    result.raw_bounds, result.panel_polygon = (0, 0, *RESOLUTION), None
    amap = np.zeros((512, 512), dtype=np.float32)
    amap[peak[1], peak[0]] = .9
    details = [{"label_id": 1, "peak_yx": (peak[1], peak[0]), "is_dust": False}]
    bombs = [BombDefect("STANDARD", "B01", "point", [(100, 100), (150, 100)])]
    match = inferencer._aoi_bomb_priority_match(tile, amap, details, bombs, result, RESOLUTION)
    assert (match is not None) is expected


def test_aoi_bomb_priority_heatmap_shows_ignored_regions(monkeypatch, tmp_path):
    import cv2
    from capi_heatmap import HeatmapManager, build_bomb_region_debug_panel

    inferencer, result, bomb_info = make_overlapping_point_bomb_case("client")
    inferencer.config.aoi_bomb_priority_enabled = True
    tile, score, amap = result.anomaly_tiles[1]
    amap[300:307, 300:307] = score
    tile.omit_crop_image = tile.image.copy()
    tile.dust_mask = np.zeros_like(amap, dtype=np.uint8)
    apply_bomb_postprocess(inferencer, result, bomb_info, "v2", monkeypatch)
    panel = build_bomb_region_debug_panel(tile.bomb_region_diagnostics, 512)
    assert tuple(panel[303, 303]) == (128, 128, 128)
    assert tuple(panel[33, 0]) == (255, 0, 255)
    captured = []
    original = cv2.putText

    def capture(img, text, *args, **kwargs):
        captured.append(str(text))
        return original(img, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture)
    output = HeatmapManager(tmp_path, save_format="png").save_tile_heatmap(
        tmp_path, "priority", tile.tile_id, tile.image, amap, score, tile_info=tile,
    )
    assert cv2.imread(output) is not None
    text = "\n".join(captured)
    assert "BOMB: AOI priority (Filtered as OK)" in text
    assert "IGNORED (AOI BOMB)" in text
    assert "Not evaluated: AOI bomb priority" in text
    assert "Remaining NG:0 Dust:0 Ignored:1" in text
    assert "REAL_NG" not in text
