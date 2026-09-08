import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from capi_config import CAPIConfig
from capi_inference import AOIReportDefect, CAPIInferencer, ImageResult
from capi_server import _white_frame_image_result, aggregate_judgment, results_to_db_data
from capi_station_adapter import create_station_adapter
from capi_white_frame import inspect_white_frame_image


def _frame(path, *, notch=False, angle=0.0, vertical=False):
    image = np.zeros((800, 1200), np.uint8)
    cv2.rectangle(image, (180, 140), (1020, 660), 255, 7)
    if notch:
        if vertical:
            image[398:402, 1014:1027] = 0
            image[398:402, 1020] = 255  # narrowed line, still connected
        else:
            image[134:147, 598:602] = 0
            image[140, 598:602] = 255
    if angle:
        image = cv2.warpAffine(image, cv2.getRotationMatrix2D((600, 400), angle, 1), (1200, 800))
    assert cv2.imwrite(str(path), image)
    return path


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("angle", [0.0, 2.4, -1.7])
@pytest.mark.parametrize("notch", [False, True])
def test_native_tile_finds_thin_notch_that_whole_frame_accepts(tmp_path, vertical, angle, notch):
    path = _frame(tmp_path / "GWhite_Frame100000.png", notch=notch, angle=angle, vertical=vertical)
    inspection = inspect_white_frame_image(
        path, product_resolution=(1920, 1200),
        aoi_points=[{"defect_code": "CDK2", "product_x": 1910 if vertical else 960,
                     "product_y": 600 if vertical else 11}],
    )
    assert inspection.payload["full_frame_status"] == "OK"
    assert inspection.payload["status"] == ("NG" if notch else "OK")
    tile = inspection.payload["aoi_tiles"][0]
    assert tile["status"] == ("NG" if notch else "OK")
    assert tile["width"] == tile["height"] == 512
    if notch:
        assert tile["gaps"][0]["length_px"] < 15
        result = _white_frame_image_result(inspection)
        judgment, details = aggregate_judgment([result])
        assert judgment == "NG"
        assert json.loads(details)[0]["type"] == "white_frame_aoi_gap"
        assert results_to_db_data([result], {})[0]["is_ng"] == 1
        from capi_database import CAPIDatabase
        summary = CAPIDatabase._parse_white_frame_row({"white_frame_result": json.dumps(inspection.payload)})
        assert summary["white_frame_status"] == "NG"
        assert tile["side"] in summary["white_frame_ng_sides"]


def test_aapi_followup_expansion_is_idempotent_and_keeps_original_coordinates():
    inf = CAPIInferencer.__new__(CAPIInferencer)
    inf.station_adapter = create_station_adapter("aapi")
    point = AOIReportDefect("CDK2", 461, 11, "WHITEFRA")
    regular = AOIReportDefect("CM00", 1000, 555, "W0F00000")
    original = {"WHITEFRA": [point], "W0F00000": [regular]}
    expanded = inf._filter_aoi_report_for_inference(original)
    assert set(expanded) == {"W0F00000"}
    assert len(expanded["W0F00000"]) == 2
    copied = expanded["W0F00000"][1]
    assert copied == point and copied is not point
    copied.resolved_image_x = 300
    assert point.resolved_image_x == -1
    assert original["W0F00000"] == [regular]
    assert inf._filter_aoi_report_for_inference(expanded) == expanded
    inf.station_adapter = create_station_adapter("capi")
    assert inf._filter_aoi_report_for_inference(original) == {"W0F00000": [regular]}


def test_local_aoi_check_does_not_apply_whole_frame_corner_exclusion(tmp_path):
    path = _frame(tmp_path / "GWhite_Frame100000.png")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    image[134:147, 209:213] = 0
    image[140, 209:213] = 255
    assert cv2.imwrite(str(path), image)
    inspection = inspect_white_frame_image(path, product_resolution=(1920, 1200), aoi_points=[
        {"defect_code": "CDK2", "product_x": 69, "product_y": 11},
    ])
    assert inspection.payload["full_frame_status"] == "OK"
    assert inspection.payload["aoi_tiles"][0]["status"] == "NG"
    assert any(abs(gap["center_x"] - 211) < 4 for gap in inspection.payload["aoi_tiles"][0]["gaps"])


@pytest.mark.parametrize("new_arch", [True, False])
def test_white_screen_followup_uses_inward_patchcore_tile_at_edge(tmp_path, new_arch):
    inf = CAPIInferencer.__new__(CAPIInferencer)
    inf.station_adapter = create_station_adapter("aapi")
    inf.config = CAPIConfig(is_new_architecture=new_arch, tile_size=512, aoi_coord_inspection_enabled=True)
    image = np.full((800, 1200), 200, np.uint8)
    result = ImageResult(
        image_path=tmp_path / "GW0F00000100000.png", image_size=(1200, 800),
        otsu_bounds=(100, 100, 1100, 700), raw_bounds=(100, 100, 1100, 700),
        exclusion_regions=[], tiles=[], excluded_tile_count=0, processed_tile_count=0, processing_time=0,
    )
    inf._read_detection_image = lambda _path: image
    inf._resolve_aoi_edge_inspector_mode = lambda: "cv"
    inf._inspect_aoi_edge_defect = MagicMock(side_effect=AssertionError("follow-up must use PatchCore"))
    stats = inf._apply_aoi_coord_inspection(
        tmp_path, [result], None, False, (1920, 1200),
        {"WHITEFRA": [AOIReportDefect("CDK2", 960, 11, "WHITEFRA")]},
    )
    assert stats == {"aoi_tile_count": 1, "aoi_edge_count": 0}
    tile = result.tiles[0]
    assert tile.aoi_source_prefix == "WHITEFRA"
    assert tile.zone == "edge"
    assert tile.aoi_tile_shift_dy > 0 and tile.aoi_tile_shift_dx == 0
    assert tile.y >= 100 and tile.y + tile.height <= 700
    assert tile.y <= tile.aoi_image_y < tile.y + tile.height
    assert (tile.aoi_product_x, tile.aoi_product_y) == (960, 11)


def test_white_frame_tile_image_endpoint_uses_saved_crop_and_findings(tmp_path):
    from capi_web import CAPIWebHandler

    path = _frame(tmp_path / "GWhite_Frame100000.png", notch=True)
    inspection = inspect_white_frame_image(path, product_resolution=(1920, 1200), aoi_points=[
        {"defect_code": "CDK2", "product_x": 960, "product_y": 11},
    ])
    detail = {"image_dir": str(tmp_path), "images": [
        {"image_name": path.name, "white_frame_result": inspection.payload},
    ]}
    handler = object.__new__(CAPIWebHandler)
    handler.db = SimpleNamespace(get_record_detail=lambda _id: detail)
    handler._read_inference_image = lambda *_args: cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    images = []
    handler._send_image_array_png = images.append
    handler._send_404 = MagicMock()
    handler._handle_source_image(f"/images/7/{path.name}", {"white_frame_tile": ["0"]})
    assert images[0].shape == (512, 512, 3)
    assert np.any((images[0][:, :, 2] == 255) & (images[0][:, :, 1] == 0))
    handler._handle_source_image(f"/images/7/{path.name}", {"white_frame_tile": ["-1"]})
    handler._send_404.assert_called_once()


@pytest.mark.parametrize("score,white_available", [(0.2, True), (0.8, True), (0.2, False)])
def test_aapi_white_frame_only_report_runs_white_screen_model_and_persists_details(tmp_path, monkeypatch, score, white_available):
    from capi_database import CAPIDatabase
    from capi_heatmap import HeatmapManager
    from capi_server import CAPIServer, build_dual_protocol_response
    from capi_web import CAPIWebHandler
    import threading

    frame_path = _frame(tmp_path / "GWhite_Frame100000.png")
    white_path = tmp_path / "GW0F00000100000.png"
    image = np.zeros((800, 1200), np.uint8)
    image[100:700, 100:1100] = 200
    if white_available:
        assert cv2.imwrite(str(white_path), image)
    config = CAPIConfig(
        machine_id="GN160JCEL250S", is_new_architecture=True, grid_tiling_enabled=False,
        aoi_coord_inspection_enabled=True, enable_panel_polygon=False,
        scratch_classifier_enabled=False, image_abnormal_detection_enabled=False,
        model_mapping={"W0F00000": {"inner": "white-inner.pt", "edge": "white-edge.pt"}},
        threshold_mapping={"W0F00000": {"inner": 0.5, "edge": 0.5}},
    )
    adapter = create_station_adapter("aapi")
    inf = CAPIInferencer(config, station_adapter=adapter)
    predicted = []
    model = object()

    def predict(tile, **kwargs):
        assert kwargs["inferencer"] is model
        predicted.append(tile)
        amap = np.zeros((512, 512), np.float32)
        amap[8:20, 250:262] = score
        return score, amap

    server = CAPIServer.__new__(CAPIServer)
    server.path_mapping = {}
    server.station_adapter = adapter
    server._get_or_create_inferencer = lambda _model: inf
    server._gpu_lock = threading.Lock()
    server.cpu_workers = 1
    server._evaluate_within_spec_for_inference = MagicMock(
        return_value={"converted": True, "reason": "within-spec rule matched"},
    )
    parsed = {"glass_id": "G", "model_id": config.machine_id, "machine_no": "AAPI09",
              "image_dir": str(tmp_path), "machine_judgment": "NG", "resolution": (1920, 1200),
              "bomb_info": None, "aoi_report_payload": "White_Frame,CDK2(02880,00011)"}
    with patch.object(inf, "_get_model_for", return_value=model) as get_model, \
         patch.object(inf, "predict_tile", side_effect=predict), \
         patch.object(inf, "_detect_panel_mark_binary_region", return_value=(None, [])):
        judgment, ng_details, results, *_rest = server._process_request(parsed)
    assert judgment == ("NG" if score > 0.5 else "OK"), judgment
    server._evaluate_within_spec_for_inference.assert_not_called()
    if not white_available:
        assert not predicted
        assert len(results) == 1
        wf = results[0].white_frame_result
        assert wf["aoi_tiles"][0]["status"] == "OK"
        assert wf["white_screen"]["status"] == "UNAVAILABLE"
        CAPIWebHandler.init_jinja()
        html = CAPIWebHandler.jinja_env.get_template("_white_frame_result.html").render(
            wf=wf, img={"image_name": frame_path.name}, detail={"id": 1},
        )
        assert "未取得 W0F00000 補檢結果" in html
        return
    assert len(predicted) == 1
    get_model.assert_called_once_with(config.machine_id, "W0F00000", "edge")
    tile = predicted[0]
    assert tile.aoi_tile_shift_dy > 0
    assert tile.aoi_source_prefix == "WHITEFRA"
    assert [result.image_path for result in results] == [white_path, frame_path]
    wf = results[-1].white_frame_result
    assert wf["white_screen"]["tile_ids"] == [tile.tile_id]
    assert wf["aoi_tiles"][0]["aoi_product_x"] == 960  # raw X / 3 exactly once
    response = build_dual_protocol_response(parsed, judgment, results, config)
    if score > 0.5:
        assert "W0F00000" in response

    heatmaps = tmp_path / "heatmaps"
    manager = HeatmapManager(str(heatmaps))
    manager.save_tile_heatmap(heatmaps, white_path.stem, tile.tile_id, tile.image,
                              results[0].anomaly_tiles[0][2], score, tile_info=tile)
    db = CAPIDatabase(tmp_path / "record.db")
    data = results_to_db_data(results, {"dir": str(heatmaps)})
    record_id = db.save_inference_record(
        glass_id="G", model_id=config.machine_id, machine_no="AAPI09", resolution=(1920, 1200),
        machine_judgment="NG", ai_judgment=judgment, image_dir=str(tmp_path), total_images=2,
        ng_images=int(score > 0.5), ng_details=ng_details, request_time="2026-09-08 12:00:00",
        response_time="2026-09-08 12:00:01", processing_seconds=1, image_results_data=data,
        client_request_text=f"AOI@G;{config.machine_id};AAPI09;1920,1200;NG;{tmp_path};White_Frame,CDK2(02880,00011)",
    )
    detail = db.get_record_detail(record_id)
    stored_wf = detail["images"][-1]["white_frame_result"]
    stored_tile = stored_wf["white_screen_result"]["tiles"][0]
    assert stored_tile["id"] == detail["images"][0]["tiles"][0]["id"]
    assert stored_tile["is_anomaly"] == int(score > 0.5)
    assert Path(stored_tile["heatmap_path"]).is_file()
    assert stored_tile["aoi_tile_shift_dy"] > 0
    CAPIWebHandler.init_jinja()
    for template in ("record_detail.html", "record_detail_v3.html"):
        html = CAPIWebHandler.jinja_env.get_template(template).render(
            detail=detail, heatmap_base_dir=str(heatmaps),
        )
        assert "AOI 座標放大檢查 · OpenCV" in html
        assert "W0F00000 白畫面 PatchCore 補檢" in html
        assert f"?white_frame_tile=0" in html
        assert html.count(f'heatmap_{white_path.stem}_tile{tile.tile_id}.png') == 1
        assert "AI 分數" in html and "修正量" in html
        if template == "record_detail.html":
            assert f'data-tile-result-id="{stored_tile["id"]}"' in html
        else:
            assert "openSampleClassification(this)" not in html

    # Rerunning an existing record must restore the same request points and
    # rebuild references to the replacement tile rows, using the same model.
    monkeypatch.setattr(CAPIWebHandler, "inferencer", inf)
    monkeypatch.setattr(CAPIWebHandler, "_capi_server_instance", server)
    monkeypatch.setattr(CAPIWebHandler, "db", db)
    monkeypatch.setattr(CAPIWebHandler, "heatmap_manager", None)
    monkeypatch.setattr(CAPIWebHandler, "_gpu_lock", None, raising=False)
    monkeypatch.setattr(CAPIWebHandler, "_rerun_lock", threading.Lock(), raising=False)
    monkeypatch.setattr(CAPIWebHandler, "_rerun_tasks", {record_id: {"status": "running"}}, raising=False)
    with patch.object(inf, "_get_model_for", return_value=model), \
         patch.object(inf, "predict_tile", side_effect=predict), \
         patch.object(inf, "_detect_panel_mark_binary_region", return_value=(None, [])):
        CAPIWebHandler._rerun_worker(record_id, detail)
    assert CAPIWebHandler._rerun_tasks[record_id]["status"] == "done"
    server._evaluate_within_spec_for_inference.assert_not_called()
    rerun_detail = db.get_record_detail(record_id)
    rerun_wf = rerun_detail["images"][-1]["white_frame_result"]
    assert rerun_wf["aoi_tiles"][0]["aoi_product_x"] == 960
    assert rerun_wf["white_screen"]["status"] == "INSPECTED"
    assert len(rerun_wf["white_screen_result"]["tiles"]) == 1
    assert rerun_wf["white_screen_result"]["tiles"][0]["is_anomaly"] == int(score > 0.5)


def test_local_white_frame_ng_is_reported_on_its_own_screen(tmp_path):
    from capi_server import build_dual_protocol_response
    path = _frame(tmp_path / "GWhite_Frame100000.png", notch=True)
    inspection = inspect_white_frame_image(path, product_resolution=(1920, 1200), aoi_points=[
        {"defect_code": "CDK2", "product_x": 960, "product_y": 11},
    ])
    result = _white_frame_image_result(inspection)
    judgment, _ = aggregate_judgment([result])
    response = build_dual_protocol_response(
        {"glass_id": "G", "model_id": "GN160JCEL250S", "machine_no": "AAPI09",
         "machine_judgment": "NG", "resolution": (1920, 1200)},
        judgment, [result], CAPIConfig(),
    )
    assert "WHITEFRA" in response
    assert "W0F00000" not in response


@pytest.mark.parametrize("frame_ng,white_ng", [(False, False), (True, False), (False, True), (True, True)])
def test_either_white_frame_check_ng_preserves_panel_ng(frame_ng, white_ng):
    from capi_server import _has_white_frame_ng

    tile = SimpleNamespace(aoi_source_prefix="WHITEFRA", is_aoi_coord_below_threshold=not white_ng)
    results = [
        SimpleNamespace(white_frame_result={"status": "NG" if frame_ng else "OK"}, anomaly_tiles=[]),
        SimpleNamespace(white_frame_result=None, anomaly_tiles=[(tile, 0.8 if white_ng else 0.2, None)]),
    ]
    assert _has_white_frame_ng(results) == (frame_ng or white_ng)


@pytest.mark.parametrize("filtered_by", [
    "is_bomb", "is_suspected_dust_or_scratch", "is_in_exclude_zone",
    "scratch_filtered", "is_aoi_coord_below_threshold", "ordinary_screen",
])
def test_white_frame_ng_override_preserves_patchcore_filters_and_other_screens(filtered_by):
    from capi_server import _has_white_frame_ng

    tile = SimpleNamespace(aoi_source_prefix="WHITEFRA")
    if filtered_by == "ordinary_screen":
        tile.aoi_source_prefix = "W0F00000"
    else:
        setattr(tile, filtered_by, True)
    result = SimpleNamespace(white_frame_result=None, anomaly_tiles=[(tile, 0.8, None)])
    assert _has_white_frame_ng([result]) is False
