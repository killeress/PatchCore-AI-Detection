"""Black-screen binary candidates must be filtered using exact OMIT pixels."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from capi_config import BombDefect, CAPIConfig
from capi_inference import AOIReportDefect, CAPIInferencer, ImageResult, TileInfo
from capi_station_adapter import create_station_adapter


def make_case(anomaly_map, dust_mask, *, profile="capi", name="B0F00000_015846.tif",
              x=0, y=0, min_area=4):
    worker = object.__new__(CAPIInferencer)
    worker.station_adapter = create_station_adapter(profile)
    worker.config = CAPIConfig(skip_files=["B0F00000"])
    worker.config.bright_spot_min_area = 4
    worker.config.dust_two_stage_enabled = True
    height, width = dust_mask.shape
    tile = TileInfo(
        tile_id=1, x=x, y=y, width=width, height=height,
        image=np.zeros((height, width), np.uint8),
        is_bright_spot_detection=True, bright_spot_min_area=min_area,
        bright_spot_area=0 if anomaly_map is None else int(np.count_nonzero(anomaly_map)),
        score_threshold=0.5, zone="bright_spot",
    )
    result = ImageResult(
        image_path=Path(name), image_size=(32, 32), otsu_bounds=(0, 0, 32, 32),
        exclusion_regions=[], tiles=[tile], excluded_tile_count=0,
        processed_tile_count=1, processing_time=0,
        anomaly_tiles=[(tile, 1.0, anomaly_map)],
    )
    worker._check_dust_or_scratch_feature_with_context = Mock(
        return_value=(True, dust_mask.copy(), 0.1, "OMIT exact mask"),
    )
    worker.check_dust_per_region = Mock(side_effect=AssertionError("binary map entered PatchCore dust logic"))
    worker.check_dust_two_stage = Mock(side_effect=AssertionError("binary map entered PatchCore two-stage logic"))
    return worker, result, tile


def apply_filter(worker, result, *, omit_image=None, overexposed=False):
    if omit_image is None:
        omit_image = np.full((32, 32), 64, np.uint8)
    worker._apply_omit_dust_postprocess(
        [result], omit_image, overexposed, "fixture", cpu_workers=1,
    )
    worker.check_dust_per_region.assert_not_called()
    worker.check_dust_two_stage.assert_not_called()


def assert_precise_mask_requested(worker):
    worker._check_dust_or_scratch_feature_with_context.assert_called_once()
    assert worker._check_dust_or_scratch_feature_with_context.call_args.kwargs["extension_override"] == 0


@pytest.mark.parametrize("profile,name", [
    ("capi", "B0F00000_015846.tif"),
    ("aapi", "T865HG77BB28B0F00000015846.tif"),
    ("aapi", "T865HG77BB28B8F00000015846.tif"),
])
def test_black_dust_filters_label_and_keeps_nearby_real_bright_spot(profile, name):
    amap = np.zeros((16, 16), np.float32)
    amap[4:8, 3:7] = 1  # sticker
    amap[4:6, 8:10] = 1  # real point only one pixel away
    dust = np.zeros((16, 16), np.uint8)
    dust[4:8, 3:7] = 255
    worker, result, tile = make_case(amap, dust, profile=profile, name=name, x=5, y=7)

    apply_filter(worker, result)

    expected = np.zeros_like(amap)
    expected[4:6, 8:10] = 1
    np.testing.assert_array_equal(result.anomaly_tiles[0][2], expected)
    assert result.anomaly_tiles[0][1] == 1.0
    assert tile.is_suspected_dust_or_scratch is False
    assert expected[tile.anomaly_peak_y - tile.y, tile.anomaly_peak_x - tile.x] == 1
    assert_precise_mask_requested(worker)


def test_black_dust_preserves_real_pixels_connected_to_label():
    amap = np.zeros((16, 16), np.float32)
    amap[4:8, 3:7] = 1
    amap[4:6, 7:9] = 1  # shares an edge with the sticker component
    dust = np.zeros((16, 16), np.uint8)
    dust[4:8, 3:7] = 255
    worker, result, tile = make_case(amap, dust)

    apply_filter(worker, result)

    expected = np.zeros_like(amap)
    expected[4:6, 7:9] = 1
    np.testing.assert_array_equal(result.anomaly_tiles[0][2], expected)
    assert tile.is_suspected_dust_or_scratch is False


def test_black_dust_debug_separates_removed_label_from_connected_real_point():
    amap = np.zeros((16, 16), np.float32)
    amap[4:8, 3:7] = 1
    amap[4:6, 7:9] = 1
    dust = np.zeros((16, 16), np.uint8)
    dust[4:8, 3:7] = 255
    worker, result, tile = make_case(amap, dust)
    worker.generate_dust_iou_debug_image = Mock(return_value=np.zeros((16, 16, 3), np.uint8))

    apply_filter(worker, result)

    worker.generate_dust_iou_debug_image.assert_called_once()
    debug_args = worker.generate_dust_iou_debug_image.call_args.kwargs
    details = debug_args["region_details"]
    labels = debug_args["region_labels"]
    assert details == tile.dust_region_details
    assert len(details) == 2
    real = next(region for region in details if not region["is_dust"])
    removed = next(region for region in details if region["is_dust"])
    assert (removed["area"], real["area"]) == (16, 4)
    assert np.all(labels[4:8, 3:7] == removed["label_id"])
    assert np.all(labels[4:6, 7:9] == real["label_id"])
    assert removed["label_id"] != real["label_id"]


def test_black_dust_reports_both_real_points_after_removing_sticker_connector():
    from capi_server import _tile_real_defect_points

    amap = np.zeros((16, 16), np.float32)
    amap[4:6, 1:3] = 1
    amap[4:8, 3:7] = 1  # sticker connects the two real candidates
    amap[4:6, 7:9] = 1
    dust = np.zeros((16, 16), np.uint8)
    dust[4:8, 3:7] = 255
    worker, result, tile = make_case(amap, dust, x=5, y=7)

    apply_filter(worker, result)

    real_regions = [region for region in tile.dust_region_details if not region["is_dust"]]
    assert len(real_regions) == 2
    assert [region["area"] for region in real_regions] == [4, 4]
    points = _tile_real_defect_points(tile)
    assert len(points) == 2
    local_points = [(x - tile.x, y - tile.y) for x, y in points]
    assert sum(1 <= x < 3 and 4 <= y < 6 for x, y in local_points) == 1
    assert sum(7 <= x < 9 and 4 <= y < 6 for x, y in local_points) == 1
    assert all(result.anomaly_tiles[0][2][y, x] == 1 for x, y in local_points)


def test_black_dust_iou_region_uses_union_with_entire_dust_mask():
    amap = np.zeros((16, 16), np.float32)
    amap[4:8, 4:8] = 1
    dust = np.zeros((16, 16), np.uint8)
    dust[4:8, 4:12] = 255
    worker, result, tile = make_case(amap, dust)
    worker.config.dust_heatmap_metric = "iou"

    apply_filter(worker, result)

    assert tile.is_suspected_dust_or_scratch is True
    assert len(tile.dust_region_details) == 1
    region = tile.dust_region_details[0]
    assert region["area"] == region["dust_overlap"] == 16
    assert region["metric_denominator"] == 32
    assert region["coverage"] == pytest.approx(0.5)
    assert tile.dust_region_max_cov == pytest.approx(0.5)
    assert tile.dust_heatmap_iou == pytest.approx(0.5)


def test_black_dust_partial_overlap_does_not_remove_whole_component():
    amap = np.zeros((16, 16), np.float32)
    amap[4:8, 4:8] = 1
    dust = np.zeros((16, 16), np.uint8)
    dust[4, 4] = 255
    worker, result, tile = make_case(amap, dust)

    apply_filter(worker, result)

    expected = amap.copy()
    expected[4, 4] = 0
    np.testing.assert_array_equal(result.anomaly_tiles[0][2], expected)
    assert tile.is_suspected_dust_or_scratch is False


def test_black_dust_flags_fully_covered_label():
    amap = np.zeros((16, 16), np.float32)
    amap[4:8, 4:8] = 1
    worker, result, tile = make_case(amap, (amap * 255).astype(np.uint8))

    apply_filter(worker, result)

    assert tile.is_suspected_dust_or_scratch is True
    assert_precise_mask_requested(worker)


@pytest.mark.parametrize("tile_min_area,config_min_area", [(4, 1), (0, 4)])
def test_black_dust_rechecks_remaining_component_area(tile_min_area, config_min_area):
    amap = np.zeros((16, 16), np.float32)
    amap[4:6, 4:8] = 1
    dust = np.zeros((16, 16), np.uint8)
    dust[4:6, 4:7] = 255  # only two pixels survive the subtraction
    worker, result, tile = make_case(amap, dust, min_area=tile_min_area)
    worker.config.bright_spot_min_area = config_min_area

    apply_filter(worker, result)

    assert tile.is_suspected_dust_or_scratch is True


def test_black_dust_drops_subthreshold_fragment_but_keeps_valid_candidate():
    amap = np.zeros((16, 16), np.float32)
    amap[4:6, 3:7] = 1
    amap[9:11, 9:11] = 1
    dust = np.zeros((16, 16), np.uint8)
    dust[4:6, 3:6] = 255
    worker, result, tile = make_case(amap, dust)

    apply_filter(worker, result)

    expected = np.zeros_like(amap)
    expected[9:11, 9:11] = 1
    np.testing.assert_array_equal(result.anomaly_tiles[0][2], expected)
    assert tile.is_suspected_dust_or_scratch is False
    assert expected[tile.anomaly_peak_y, tile.anomaly_peak_x] == 1


def test_black_dust_ignores_real_candidate_outside_aoi_roi():
    amap = np.zeros((16, 16), np.float32)
    amap[4:6, 3:5] = 1
    amap[9:11, 10:12] = 1
    dust = np.zeros((16, 16), np.uint8)
    dust[4:6, 3:5] = 255
    worker, result, tile = make_case(amap, dust)
    tile.aoi_roi_mask = np.zeros((16, 16), np.uint8)
    tile.aoi_roi_mask[:, :8] = 255

    apply_filter(worker, result)

    assert tile.is_suspected_dust_or_scratch is True


@pytest.mark.parametrize("condition", ["missing_omit", "overexposed", "missing_heatmap"])
def test_black_dust_keeps_ng_when_dust_cannot_be_verified(condition):
    amap = np.ones((16, 16), np.float32)
    worker, result, tile = make_case(
        None if condition == "missing_heatmap" else amap,
        np.full((16, 16), 255, np.uint8),
    )
    omit = None if condition == "missing_omit" else np.full((32, 32), 64, np.uint8)

    worker._apply_omit_dust_postprocess(
        [result], omit, condition == "overexposed", "fixture", cpu_workers=1,
    )

    assert tile.is_suspected_dust_or_scratch is False
    assert result.anomaly_tiles[0][1] == 1.0
    if condition != "missing_heatmap":
        np.testing.assert_array_equal(result.anomaly_tiles[0][2], amap)
        worker._check_dust_or_scratch_feature_with_context.assert_not_called()
    worker.check_dust_per_region.assert_not_called()
    worker.check_dust_two_stage.assert_not_called()


def test_black_dust_does_not_promote_track_only_tile_to_ng():
    amap = np.zeros((16, 16), np.float32)
    worker, result, tile = make_case(amap, np.full((16, 16), 255, np.uint8))
    tile.is_aoi_coord_below_threshold = True
    result.anomaly_tiles = [(tile, 0.0, amap)]

    apply_filter(worker, result)

    assert tile.is_suspected_dust_or_scratch is False
    assert tile.is_aoi_coord_below_threshold is True
    assert result.anomaly_tiles[0][1] == 0.0
    np.testing.assert_array_equal(result.anomaly_tiles[0][2], amap)
    assert "REAL_NG" not in tile.dust_detail_text


def test_black_dust_zero_pads_negative_tile_coordinates():
    amap = np.zeros((16, 16), np.float32)
    amap[7:9, 6:8] = 1
    dust = np.zeros((16, 16), np.uint8)
    dust[7:9, 6:8] = 255
    worker, result, tile = make_case(amap, dust, x=-3, y=-5)
    omit = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)

    apply_filter(worker, result, omit_image=omit)

    expected = np.zeros((16, 16), np.uint16)
    expected[5:, 3:] = omit[:11, :13]
    np.testing.assert_array_equal(tile.omit_crop_image, expected)
    np.testing.assert_array_equal(
        worker._check_dust_or_scratch_feature_with_context.call_args.args[5], expected,
    )
    assert tile.is_suspected_dust_or_scratch is True


@pytest.mark.parametrize("skip_files", [["B0F00000"], []])
def test_black_without_binary_detection_marker_never_uses_patchcore_dust_filter(skip_files):
    amap = np.ones((16, 16), np.float32)
    worker, result, tile = make_case(amap, np.full((16, 16), 255, np.uint8))
    worker.config.skip_files = skip_files
    tile.is_bright_spot_detection = False

    apply_filter(worker, result)

    worker._check_dust_or_scratch_feature_with_context.assert_not_called()
    assert tile.is_suspected_dust_or_scratch is False
    np.testing.assert_array_equal(result.anomaly_tiles[0][2], amap)


@pytest.mark.parametrize("keep_real_point", [False, True])
def test_v1_black_dust_uses_shared_filter_and_final_panel_judgment(tmp_path, keep_real_point):
    from capi_server import aggregate_judgment

    amap = np.zeros((16, 16), np.float32)
    amap[4:8, 3:7] = 1
    if keep_real_point:
        amap[4:6, 8:10] = 1
    dust = np.zeros((16, 16), np.uint8)
    dust[4:8, 3:7] = 255
    worker, black_result, tile = make_case(amap, dust)
    worker.config.is_new_architecture = False
    worker.config.aoi_coord_inspection_enabled = True
    worker.config.aoi_judgment_roi_enabled = False
    worker.config.grid_tiling_enabled = False
    worker.config.enable_panel_polygon = False
    worker.config.scratch_classifier_enabled = False
    worker.config.bright_spot_median_kernel = 11
    worker.config.bright_spot_threshold = 100
    worker.config.bright_spot_diff_threshold = 20
    worker.edge_inspector = None
    worker._dispatch_process_panel = worker._process_panel_v1
    black_path = tmp_path / black_result.image_path.name
    white_path = tmp_path / "W0F00000_015836.tif"
    omit_path = tmp_path / "PINIGBI0_015833.tif"
    black_result.image_path = black_path
    black_result.raw_bounds = (0, 0, 32, 32)
    white_result = ImageResult(
        image_path=white_path, image_size=(32, 32), otsu_bounds=(0, 0, 32, 32),
        raw_bounds=(0, 0, 32, 32), exclusion_regions=[], tiles=[],
        excluded_tile_count=0, processed_tile_count=0, processing_time=0,
    )
    tile.is_bright_spot_detection = False  # Phase 2 must run the real detector.
    tile.is_aoi_coord_tile = True
    tile.aoi_image_x = tile.aoi_image_y = 6
    tile.image = (amap * 255).astype(np.uint8)
    report = {"B0F00000": [AOIReportDefect("PCDK2", 6, 6, "B0F00000")]}
    worker._prepare_panel_image_files = Mock(return_value=([white_path, black_path, omit_path], False))
    worker._parse_defect_txt = Mock(return_value={})
    worker._detect_panel_mark_binary_region = Mock(return_value=(None, []))
    worker._read_detection_image = Mock(return_value=np.full((32, 32), 64, np.uint8))
    worker.check_omit_overexposure = Mock(return_value=(False, 64.0, 0.0, "OK"))
    worker._find_raw_object_bounds = Mock(return_value=((0, 0, 32, 32), np.ones((32, 32), np.uint8)))
    worker.preprocess_image = Mock(side_effect=lambda path, **_kw: black_result if path == black_path else white_result)
    worker._get_inferencer_for_prefix = Mock(return_value=None)
    worker._get_threshold_for_prefix = Mock(return_value=0.5)

    def add_aoi_tile(**_kwargs):
        black_result.tiles = [tile]
        return {"aoi_tile_count": 1, "aoi_edge_count": 0}

    worker._apply_aoi_coord_inspection = Mock(side_effect=add_aoi_tile)
    worker._apply_omit_dust_postprocess = Mock(wraps=worker._apply_omit_dust_postprocess)

    results, *_ = worker.process_panel(tmp_path, cpu_workers=1, aoi_report_override=report)

    worker._apply_omit_dust_postprocess.assert_called_once()
    assert worker._apply_omit_dust_postprocess.call_args.args[0] == [black_result]
    assert tile.is_bright_spot_detection is True
    assert tile.is_suspected_dust_or_scratch is (not keep_real_point)
    assert aggregate_judgment(results)[0] == ("NG" if keep_real_point else "OK")
    assert_precise_mask_requested(worker)
    worker.check_dust_per_region.assert_not_called()
    worker.check_dust_two_stage.assert_not_called()


def test_black_bomb_matching_uses_surviving_pixels_without_reintroducing_label():
    amap = np.zeros((16, 16), np.float32)
    amap[4:8, 3:7] = 1
    amap[4:6, 7:9] = 1  # one connected component before dust filtering
    dust = np.zeros((16, 16), np.uint8)
    dust[4:8, 3:7] = 255
    worker, result, tile = make_case(amap, dust)
    tile.is_aoi_coord_tile = True
    tile.aoi_roi_mask = np.full((16, 16), 255, np.uint8)
    tile.aoi_image_x = tile.aoi_product_x = 3
    tile.aoi_image_y = tile.aoi_product_y = 4
    result.raw_bounds = (0, 0, 16, 16)
    worker.config.bomb_match_tolerance = 1

    apply_filter(worker, result)
    filtered_map = result.anomaly_tiles[0][2]
    assert np.count_nonzero(filtered_map) == 4
    assert np.count_nonzero(tile.dust_heatmap_binary) == 20
    # Bomb-region extraction is a separate downstream consumer of the final map.
    worker.check_dust_per_region = CAPIInferencer.check_dust_per_region.__get__(worker)
    matched = worker._match_aoi_point_bomb_regions(
        result, tile, filtered_map,
        [BombDefect("B0F00000", "B01", "point", [(8, 5)])],
        (16, 16), {},
    )

    assert matched == (True, "B01")
    assert tile.bomb_remaining_points == []
    np.testing.assert_array_equal(
        tile.bomb_region_diagnostics["heatmap_binary"] > 0, filtered_map > 0,
    )


def _fragmented_faint_defect(shape=(64, 64), origin=(10, 10), rows=4, cols=6, step=6):
    """A faint defect near diff_thr breaks into evenly spaced 2x2 fragments."""
    amap = np.zeros(shape, np.float32)
    top, left = origin
    for row in range(rows):
        for col in range(cols):
            y, x = top + row * step, left + col * step
            amap[y:y + 2, x:x + 2] = 1
    return amap


def test_black_dust_merges_fragments_of_one_faint_defect_into_one_report_point():
    from capi_server import _tile_real_defect_points

    amap = _fragmented_faint_defect()
    dust = np.zeros((64, 64), np.uint8)
    dust[10, 10] = 255  # OMIT evidence overlaps one fragment pixel only.
    worker, result, tile = make_case(amap, dust, x=100, y=200, min_area=2)

    apply_filter(worker, result)

    real_regions = [region for region in tile.dust_region_details if not region["is_dust"]]
    assert len(real_regions) == 1
    assert real_regions[0]["area"] == int(np.count_nonzero(amap)) - 1
    assert "merged_from=24" in tile.dust_detail_text
    # A single real region keeps the existing one-point report path.
    assert _tile_real_defect_points(tile) == []
    local_x, local_y = tile.anomaly_peak_x - tile.x, tile.anomaly_peak_y - tile.y
    assert result.anomaly_tiles[0][2][local_y, local_x] == 1


def test_black_dust_keeps_distant_bright_defects_as_separate_points():
    from capi_server import _tile_real_defect_points

    amap = np.zeros((64, 64), np.float32)
    amap[4:6, 4:6] = 1
    amap[56:58, 56:58] = 1
    dust = np.zeros((64, 64), np.uint8)
    dust[30, 30] = 255
    amap[30, 30] = 1
    worker, result, tile = make_case(amap, dust, min_area=2)

    apply_filter(worker, result)

    assert len(_tile_real_defect_points(tile)) == 2
