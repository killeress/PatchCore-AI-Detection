"""MOD2 B8F00000 aliases use the existing black-screen inspection path."""
import os
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from capi_config import BombDefect, CAPIConfig
from capi_inference import CAPIInferencer
from capi_preprocess import PanelPreprocessResult, filter_panel_lighting_files
from capi_server import aggregate_judgment, build_qjpg_response, check_image_abnormal_precheck
from capi_station_adapter import create_station_adapter


BLACK_NAME = "T863MF77AD50B8F00000000628.tif"
WHITE_NAME = "T863MF77AD50W0F00000000623.tif"


@pytest.mark.parametrize("name", [
    BLACK_NAME, "T863MF77AD50b8f00000000628.TIF", "B8F00000",
    "T863MF77AD50B0F00000000628.tif", "B0F00000",
])
def test_black_alias_uses_b0f_configuration(name):
    adapter = create_station_adapter("aapi")
    assert adapter.image_prefix(name) == "B0F00000"
    assert adapter.training_image_prefix(name) == "B0F00000"
    assert adapter.report_prefix(name) == "B0F00000"
    assert adapter.image_group_key(name) == "B0F00000"
    assert adapter.model_prefix("B8F00000") == "B0F00000"
    assert CAPIConfig(skip_files=["B0F00000"]).should_skip_file(name, adapter)
    assert "B0F00000" not in adapter.training_prefixes


def test_retake_selection_groups_b0f_and_b8f_as_one_black_screen(tmp_path):
    old = tmp_path / "T863MF77AD50B0F00000000620.tif"
    latest = tmp_path / BLACK_NAME
    white = tmp_path / WHITE_NAME
    for path, modified in ((old, 10), (latest, 20), (white, 10)):
        path.write_bytes(b"image")
        os.utime(path, (modified, modified))
    worker = CAPIInferencer.__new__(CAPIInferencer)
    worker.station_adapter = create_station_adapter("aapi")
    worker.config = SimpleNamespace(max_images_per_panel=7)
    selected, duplicate = worker._prepare_panel_image_files(tmp_path)
    assert duplicate is True
    assert set(selected) == {latest, white}
    for prefix in ("B0F00000", "B8F00000"):
        assert worker.station_adapter.find_lighting_image(tmp_path, prefix) == latest
    assert filter_panel_lighting_files(
        tmp_path, prefix_resolver=worker.station_adapter.image_prefix,
        allowed_prefixes=worker.station_adapter.inference_prefixes,
    ) == {"W0F00000": white}
    assert create_station_adapter("capi").image_prefix(BLACK_NAME) != "B0F00000"


def test_aoi_and_client_bomb_coordinates_match_both_black_aliases(tmp_path):
    worker = CAPIInferencer.__new__(CAPIInferencer)
    worker.station_adapter = create_station_adapter("aapi")
    worker.config = CAPIConfig(bomb_area_force_detection_enabled=True)
    report = worker._parse_aoi_report_txt(
        tmp_path, glass_id="T863MF77AD50", machine_judgment="NG",
        report_payload="B8F00000,CTMD6(00900,00400);B0F00000,CTMD6(01800,00700)",
    )
    assert set(report) == {"B0F00000"}
    assert [(point.product_x, point.product_y) for point in report["B0F00000"]] == [(300, 400), (600, 700)]
    assert all(point.image_prefix == "B0F00000" for point in report["B0F00000"])
    assert worker._aoi_prefix_matches("B8F00000", "B0F00000")
    bomb = dict(image_prefix="B8F00000", defect_type="point", coordinates=[(300, 400)])
    unchanged, added = worker._aoi_report_with_forced_client_bomb_coords(report, bomb)
    assert added == 0
    assert unchanged == report
    updated, added = worker._aoi_report_with_forced_client_bomb_coords({}, bomb)
    assert added == 1
    assert set(updated) == {"B0F00000"}
    matched, _ = worker.check_bomb_match(
        "B0F00000", 300, 400, (0, 0, 1024, 1024),
        product_resolution=(1024, 1024),
        bomb_list=[BombDefect("B8F00000", "B01", "point", [(300, 400)])],
    )
    assert matched is True
    assert bomb["image_prefix"] == "B8F00000"


@pytest.mark.parametrize("trigger", ["aoi", "forced_bomb"])
def test_b8f_tile_runs_actual_bright_spot_detection_without_patchcore(tmp_path, trigger):
    black = np.zeros((1024, 1024), np.uint8)
    black[507:517, 507:517] = 255
    assert cv2.imwrite(str(tmp_path / BLACK_NAME), black)
    assert cv2.imwrite(str(tmp_path / WHITE_NAME), np.full_like(black, 80))
    cfg = CAPIConfig(
        is_new_architecture=True, grid_tiling_enabled=False,
        aoi_coord_inspection_enabled=True, enable_panel_polygon=False,
        scratch_classifier_enabled=False, model_mapping={}, threshold_mapping={},
        bomb_area_force_detection_enabled=True,
    )
    adapter = create_station_adapter("aapi")
    worker = CAPIInferencer(cfg, station_adapter=adapter)
    report = worker._parse_aoi_report_txt(
        tmp_path, glass_id="T863MF77AD50", machine_judgment="NG",
        report_payload="B8F00000,CTMD6(01536,00512)",
    )
    bomb = None
    if trigger == "forced_bomb":
        report = {}
        bomb = dict(image_prefix="B8F00000", defect_type="point", coordinates=[(512, 512)])
    reference = PanelPreprocessResult(
        image_path=tmp_path / WHITE_NAME, lighting="W0F00000",
        foreground_bbox=(0, 0, 1024, 1024), panel_polygon=None, tiles=[],
    )
    with patch("capi_preprocess.preprocess_panel_folder", return_value={"W0F00000": reference}), \
         patch.object(worker, "_detect_panel_mark_binary_region", return_value=(None, [])), \
         patch.object(worker, "_get_model_for", side_effect=AssertionError("black screen requested PatchCore")) as get_model, \
         patch.object(worker, "_detect_bright_spots", wraps=worker._detect_bright_spots) as bright:
        results, *_rest = worker.process_panel(
            tmp_path, product_resolution=(1024, 1024), aoi_report_override=report,
            bomb_info=bomb,
        )
    black_results = [result for result in results if result.image_path.name == BLACK_NAME]
    assert len(black_results) == 1
    result = black_results[0]
    assert result.report_image_prefix == "B0F00000"
    assert len(result.tiles) == 1
    assert result.tiles[0].is_bright_spot_detection is True
    assert result.tiles[0].bright_spot_area == 100
    assert result.tiles[0].zone == "bright_spot"
    bright.assert_called_once()
    get_model.assert_not_called()
    judgment, _ = aggregate_judgment(results)
    assert judgment == ("NG" if trigger == "aoi" else "OK")
    assert result.tiles[0].is_bomb == (trigger == "forced_bomb")
    response = build_qjpg_response(
        {"glass_id": "T863MF77AD50", "resolution": (1024, 1024)},
        judgment, results, cfg,
    )
    assert response.endswith("B0F00000,")


def test_b8f_uses_existing_black_brightness_limits(tmp_path, monkeypatch):
    assert cv2.imwrite(str(tmp_path / BLACK_NAME), np.full((64, 64), 50, np.uint8))
    monkeypatch.setattr("capi_server._detect_image_abnormal_product_polygon", lambda *a: (None, "full_image"))
    adapter = create_station_adapter("aapi")
    result = check_image_abnormal_precheck(
        tmp_path, CAPIConfig(image_abnormal_detection_enabled=True),
        report_prefixes=["B0F00000"], image_prefix_resolver=adapter.image_prefix,
        screen_alias_resolver=adapter.model_prefix, station_profile="aapi",
    )
    assert result["screen"] == "B0F00000"
    assert result["upper"] == 13
    assert result["mean_brightness"] == 50


def test_hy_response_recognizes_b8f_as_black():
    response = build_qjpg_response(
        {"glass_id": "T863MF77AD50"}, "ERR:HY:B8F00000", [], CAPIConfig(),
    )
    assert response == "@QJPG-T863MF77AD50;NG;00;NGPCO050000000000B0F00000,"
