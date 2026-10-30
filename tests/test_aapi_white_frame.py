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


@pytest.mark.parametrize("profile", ["aapi", "capi"])
def test_white_frame_points_stay_out_of_patchcore(profile):
    inf = CAPIInferencer.__new__(CAPIInferencer)
    inf.station_adapter = create_station_adapter(profile)
    point = AOIReportDefect("CDK2", 461, 11, "WHITEFRA")
    regular = AOIReportDefect("CM00", 1000, 555, "W0F00000")
    original = {"WHITEFRA": [point], "W0F00000": [regular]}
    filtered = inf._filter_aoi_report_for_inference(original)
    assert filtered == {"W0F00000": [regular]}
    assert original == {"WHITEFRA": [point], "W0F00000": [regular]}
    assert inf._filter_aoi_report_for_inference({"WHITEFRA": [point]}) == {}
    # Also reject a previously expanded report passed by an internal caller.
    assert inf._filter_aoi_report_for_inference({"W0F00000": [regular, point]}) == filtered


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
def test_white_frame_report_does_not_create_white_screen_tiles(tmp_path, new_arch):
    inf = CAPIInferencer.__new__(CAPIInferencer)
    inf.station_adapter = create_station_adapter("aapi")
    inf.config = CAPIConfig(is_new_architecture=new_arch, tile_size=512, aoi_coord_inspection_enabled=True)
    result = ImageResult(
        image_path=tmp_path / "GW0F00000100000.png", image_size=(1200, 800),
        otsu_bounds=(100, 100, 1100, 700), raw_bounds=(100, 100, 1100, 700),
        exclusion_regions=[], tiles=[], excluded_tile_count=0, processed_tile_count=0, processing_time=0,
    )
    inf._read_detection_image = MagicMock(side_effect=AssertionError("no white-screen follow-up"))
    stats = inf._apply_aoi_coord_inspection(
        tmp_path, [result], None, False, (1920, 1200),
        {"WHITEFRA": [AOIReportDefect("CDK2", 960, 11, "WHITEFRA")]},
    )
    assert stats == {"aoi_tile_count": 0, "aoi_edge_count": 0}
    assert result.tiles == [] and result.edge_defects == []
    inf._read_detection_image.assert_not_called()


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


@pytest.mark.parametrize("white_available,regular_white,frame_notch,with_bomb", [
    (True, False, False, False), (False, False, True, False),
    (True, False, True, True), (True, True, True, True),
])
def test_white_frame_cv_only_and_regular_white_points_survive_rerun(
    tmp_path, monkeypatch, white_available, regular_white, frame_notch, with_bomb,
):
    import threading
    from capi_database import CAPIDatabase
    from capi_server import CAPIServer, build_dual_protocol_response, parse_request
    from capi_web import CAPIWebHandler

    frame_path = _frame(tmp_path / "GWhite_Frame100000.png", notch=frame_notch)
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
        assert regular_white and tile.aoi_source_prefix == "W0F00000"
        assert kwargs["inferencer"] is model
        predicted.append(tile)
        amap = np.zeros((512, 512), np.float32)
        amap[250:262, 250:262] = .8
        return .8, amap

    server = CAPIServer.__new__(CAPIServer)
    server.path_mapping = {}
    server.station_adapter = adapter
    server._get_or_create_inferencer = lambda _model: inf
    server._gpu_lock = threading.Lock()
    server.cpu_workers = 1
    server._evaluate_within_spec_for_inference = MagicMock(return_value={"converted": False})
    bomb_fields = "White_Frame;(940/0/980/0;500/1200/600/1200);" if with_bomb else ""
    request_text = (f"AOI@G;{config.machine_id};AAPI09;1920,1200;NG;{bomb_fields}"
                    f"{tmp_path};White_Frame,CDK2(02880,00011)")
    if regular_white:
        request_text += "W0F00000,CM00(02880,00600)"
    parsed = parse_request(request_text)
    with patch.object(inf, "_get_model_for", return_value=model) as get_model, \
         patch.object(inf, "predict_tile", side_effect=predict), \
         patch.object(inf, "_detect_panel_mark_binary_region", return_value=(None, [])):
        judgment, ng_details, results, *_rest = server._process_request(parsed)
    expected = "NG" if regular_white or (frame_notch and not with_bomb) else "OK"
    assert judgment == expected, judgment
    assert len(predicted) == int(regular_white)
    if regular_white:
        assert not predicted[0].is_bomb
        get_model.assert_called_once_with(config.machine_id, "W0F00000", "inner")
    else:
        get_model.assert_not_called()
        server._evaluate_within_spec_for_inference.assert_not_called()
    wf = next(result.white_frame_result for result in results if result.white_frame_result)
    assert "white_screen" not in wf
    assert wf["aoi_tiles"][0]["aoi_product_x"] == 960
    assert wf["status"] == ("NG" if frame_notch and not with_bomb else "OK")
    if with_bomb:
        assert wf["bomb_excluded_count"] > 0
    response = build_dual_protocol_response(parsed, judgment, results, config)
    assert ("W0F00000" in response) is regular_white

    db = CAPIDatabase(tmp_path / "record.db")
    data = results_to_db_data(results, {})
    record_id = db.save_inference_record(
        glass_id="G", model_id=config.machine_id, machine_no="AAPI09", resolution=(1920, 1200),
        machine_judgment="OK" if with_bomb and not regular_white else "NG",
        ai_judgment=judgment, image_dir=str(tmp_path), total_images=len(results),
        ng_images=sum(item["is_ng"] for item in data), ng_details=ng_details,
        request_time="2026-10-30 12:00:00", response_time="2026-10-30 12:00:01",
        processing_seconds=1, image_results_data=data,
        # Old records may have lost bomb JSON or normalized machine judgment.
        client_request_text=request_text,
    )
    detail = db.get_record_detail(record_id)
    stored_wf = next(item["white_frame_result"] for item in detail["images"] if item["white_frame_result"])
    assert "white_screen_result" not in stored_wf
    CAPIWebHandler.init_jinja()
    for template in ("record_detail.html", "record_detail_v3.html"):
        html = CAPIWebHandler.jinja_env.get_template(template).render(detail=detail, heatmap_base_dir=str(tmp_path))
        assert "AOI 座標放大檢查 · OpenCV" in html
        assert "白畫面 PatchCore 補檢" not in html
        assert "未取得 W0F00000 補檢結果" not in html

    monkeypatch.setattr(CAPIWebHandler, "inferencer", inf)
    monkeypatch.setattr(CAPIWebHandler, "_capi_server_instance", server)
    monkeypatch.setattr(CAPIWebHandler, "db", db)
    monkeypatch.setattr(CAPIWebHandler, "heatmap_manager", None)
    monkeypatch.setattr(CAPIWebHandler, "_gpu_lock", None, raising=False)
    monkeypatch.setattr(CAPIWebHandler, "_rerun_lock", threading.Lock(), raising=False)
    monkeypatch.setattr(CAPIWebHandler, "_rerun_tasks", {record_id: {"status": "running"}}, raising=False)
    predicted.clear()
    with patch.object(inf, "_get_model_for", return_value=model) as get_model, \
         patch.object(inf, "predict_tile", side_effect=predict), \
         patch.object(inf, "_detect_panel_mark_binary_region", return_value=(None, [])):
        CAPIWebHandler._rerun_worker(record_id, detail)
    assert CAPIWebHandler._rerun_tasks[record_id]["status"] == "done"
    assert len(predicted) == int(regular_white)
    if not regular_white:
        get_model.assert_not_called()
    rerun_detail = db.get_record_detail(record_id)
    assert rerun_detail["ai_judgment"] == expected
    rerun_wf = next(item["white_frame_result"] for item in rerun_detail["images"] if item["white_frame_result"])
    assert "white_screen" not in rerun_wf and "white_screen_result" not in rerun_wf
    assert rerun_wf["aoi_tiles"][0]["aoi_product_x"] == 960
    if with_bomb:
        assert rerun_wf["bomb_excluded_count"] > 0


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
def test_only_frame_cv_ng_blocks_within_spec(frame_ng, white_ng):
    from capi_server import _has_white_frame_ng
    tile = SimpleNamespace(aoi_source_prefix="WHITEFRA", is_aoi_coord_below_threshold=not white_ng)
    results = [
        SimpleNamespace(white_frame_result={"status": "NG" if frame_ng else "OK"}, anomaly_tiles=[]),
        SimpleNamespace(white_frame_result=None, anomaly_tiles=[(tile, .8 if white_ng else .2, None)]),
    ]
    assert _has_white_frame_ng(results) is frame_ng
