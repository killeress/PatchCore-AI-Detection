"""Side release must reach Testing and persist the exact pre-response evidence."""
import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from capi_config import CAPIConfig, normalize_side_white_params
from capi_database import CAPIDatabase
from capi_inference import AOIReportDefect, ImageResult
from capi_server import CAPIServer, build_dual_protocol_response
from capi_side_report import compare_side_record, record_conversion, side_inference_decision
from capi_station_adapter import create_station_adapter
from capi_web import CAPIWebHandler, ai_badge, ai_simple
from test_side_conversion_statistics import sample, save, query


def clear_row():
    row = sample(("unknown",))
    row["status"] = row["payload"]["status"] = "NO_CANDIDATES"
    row["payload"]["evidence_context"]["inference_marks"] = []
    row["payload"]["parameters"]["csjbl_match_tolerance_product_px"] = 5
    return row


@pytest.mark.parametrize("distance,blocked", [(0, True), (4, True), (5, True), (5.1, False), (6, False)])
@pytest.mark.parametrize("preview_radius", [0, 50])
def test_distance_to_contour_is_inclusive_and_independent_of_reference_radius(distance, blocked, preview_radius):
    row = clear_row()
    # Product and camera spaces are identical in this fixture.
    x = row["ng_details"][0]["product_x"] + distance
    contour = [[x, 90], [x + 10, 90], [x + 10, 110], [x, 110]]
    row["status"] = row["payload"]["status"] = "CANDIDATES"
    row["payload"]["candidates"] = [{"id": 1, "side_contour": contour, "disposition": "retained"}]
    before = copy.deepcopy(row)
    comparison = compare_side_record(row, preview_radius)
    point = comparison["points"][0]
    assert point["status"] == ("matched" if blocked else "no_anomaly")
    assert record_conversion(row, comparison)["eligible"] == (not blocked)
    assert point["retained_candidate_ids"] == ([1] if blocked else [])
    assert row == before


def test_tolerance_uses_product_distance_with_scaled_camera_geometry():
    row = clear_row()
    row["payload"]["front_size"] = row["payload"]["side_size"] = [1000, 600]
    row["payload"]["mapping"]["front_polygon"] = [[0, 0], [1000, 0], [1000, 600], [0, 600]]
    row["status"] = row["payload"]["status"] = "CANDIDATES"
    # AOI is (900, 200) in camera pixels. Eight camera pixels = four product px.
    row["payload"]["candidates"] = [{"id": 1, "disposition": "retained",
        "side_contour": [[908, 190], [918, 190], [918, 210], [908, 210]]}]
    comparison = compare_side_record(row)
    assert comparison["points"][0]["status"] == "matched"
    assert comparison["points"][0]["nearest_retained_distance_product_px"] == pytest.approx(4)


@pytest.mark.parametrize("problem", ["no_image", "error", "not_run", "truncated", "mapping", "outside",
                                   "malformed_candidate", "other_ng", "missing_ng", "bomb"])
def test_missing_or_uncovered_evidence_cannot_release_whole_panel(problem):
    row = clear_row()
    payload = row["payload"]
    if problem in {"no_image", "error", "not_run"}:
        row["status"] = payload["status"] = {"no_image": "NO_IMAGE", "error": "ERROR", "not_run": "NOT_RUN"}[problem]
    elif problem == "truncated":
        payload["truncated"] = True
    elif problem == "mapping":
        payload["mapping"]["status"] = "unavailable"
    elif problem == "outside":
        payload["mapping"]["matrix"][0][2] = 1000
    elif problem == "malformed_candidate":
        payload["candidates"] = [{"id": 1, "side_contour": []}]
    elif problem == "other_ng":
        row["ng_details"].append({"image": "R0F00000", "type": "ai_defect"})
    elif problem == "missing_ng":
        row["ng_details"] = []
    else:
        payload["bombs"] = [{"id": 1, "side_tolerance_polygon": [[454, 90], [464, 90], [464, 110], [454, 110]]}]
    decision = side_inference_decision(row)
    assert not decision["converted"] and decision["final_judgment"] == "NG"


def test_mixed_mark_and_clear_points_are_one_panel_and_ok_s_is_counted_as_ok(tmp_path):
    db = CAPIDatabase(str(tmp_path / "mixed.db"))
    row = sample(("mark", "unknown"))
    decision = side_inference_decision(row)
    assert decision["converted"] and decision["conversion"]["category"] == "mixed"
    assert [p["status"] for p in decision["points"]] == ["mark_suppressed", "no_anomaly"]
    row["payload"]["inference_decision"] = decision
    record_id, _ = save(db, row, judgment="OK-s")
    report = query(db)
    assert report["conversion_summary"]["converted_records"] == 1
    assert report["conversion_summary"]["eligible_records"] == 0
    assert report["conversion_summary"]["scope"]["ok"] == 1
    assert report["conversion_summary"]["scope"]["other"] == 0
    assert db.get_shift_statistics_for("2026-10-22", "day")["ok_count"] == 1
    assert query(db, conversion="clear")["total"] == 1
    assert db.search_records(ai_judgment="OK")[1] == 1
    detail = db.get_record_detail(record_id)
    assert "最終判定 OK-s" in detail["inference_log"]
    assert "側拍轉 OK" in detail["inference_log"]
    assert ai_simple("OK-s") == "OK-s" and ai_badge("OK-s") == "badge-ok-s"
    CAPIWebHandler.init_jinja()
    html = CAPIWebHandler.jinja_env.get_template("_side_white_result.html").render(detail=detail)
    assert "OK-s · 側拍轉 OK" in html and "AOI 位置無異常 → OK" in html
    assert "判定距離容錯" in html


@pytest.mark.parametrize("rotated", [False, True])
def test_real_side_detection_releases_clear_location_and_keeps_generated_evidence(tmp_path, rotated):
    import cv2
    import numpy as np
    from capi_server import apply_side_white_inference
    from capi_side_white import snapshot_context

    raw = np.zeros((500, 700), np.uint8)
    cv2.rectangle(raw, (50, 50), (650, 450), 90, -1)
    front = tmp_path / "W0F00000_120000.tif"
    side = tmp_path / "SW0F00000_120000.tif"
    cv2.imwrite(str(front), raw)
    cv2.imwrite(str(side), raw)
    config = CAPIConfig(side_white_detection_enabled=True, inference_rotate_180_enabled=rotated)
    parsed = {"image_dir": str(tmp_path), "glass_id": "REAL-SIDE", "model_id": "GN140JCAL010S",
              "machine_no": "CAPI1", "machine_judgment": "NG",
              "aoi_passthrough_records": ["CSJBL0096000600W0F00000"]}
    report = {"W0F00000": [AOIReportDefect("CSJBL", 960, 600, "W0F00000")]}
    details = json.dumps([{"image": "W0F00000", "type": "aoi_passthrough", "defect_code": "CSJBL",
                           "product_x": 960, "product_y": 600}])
    judgment = apply_side_white_inference(parsed, [], "NG", details, report, config,
                                          tmp_path / "evidence", snapshot_context(config, None, parsed))
    assert judgment == "OK-s"
    payload = parsed["_side_white_payload"]
    assert payload["status"] == "NO_CANDIDATES"
    assert payload["inference_decision"]["points"][0]["status"] == "no_anomaly"
    assert payload["inference_decision"]["match_tolerance_product_px"] == 5
    assert payload["rotation_applied"] == rotated
    assert payload["artifacts"] and all(Path(p).is_file() for p in payload["artifacts"].values())
    assert build_dual_protocol_response(parsed, judgment, [], config).endswith(";OK,")


@pytest.mark.parametrize("other_ng", [False, True])
def test_live_process_converts_before_response_and_background_reuses_payload(tmp_path, monkeypatch, other_ng):
    import capi_server
    row = clear_row()
    row["payload"]["evidence_context"]["product_resolution"] = [500, 300]
    payload = copy.deepcopy(row["payload"])
    inspect = MagicMock(return_value=payload)
    monkeypatch.setattr("capi_side_white.inspect_side_white_folder", inspect)
    monkeypatch.setattr(capi_server, "results_to_db_data", lambda *_: [])
    monkeypatch.setattr(capi_server, "aggregate_judgment", lambda _: (
        "NG" if other_ng else "OK", json.dumps([{"type": "ai_defect", "image": "R0F00000"}] if other_ng else [])))
    config = CAPIConfig(side_white_detection_enabled=True)
    result = ImageResult(image_path=tmp_path / "W0F00000.tif", image_size=(500, 300),
                         otsu_bounds=(0, 0, 500, 300), exclusion_regions=[], tiles=[],
                         excluded_tile_count=0, processed_tile_count=0, processing_time=0., mark_text="MA")
    report = {"W0F00000": [AOIReportDefect("CSJBL", 450, 100, "W0F00000")]}
    inferencer = SimpleNamespace(config=config, process_panel=MagicMock(
        return_value=([result], None, False, "", False, None, report)))
    server = CAPIServer.__new__(CAPIServer)
    server.path_mapping = {}
    server.cpu_workers = 1
    server._gpu_lock = threading.Lock()
    server.station_adapter = create_station_adapter("capi")
    server._get_or_create_inferencer = lambda _: inferencer
    server._maybe_clear_cuda_cache_after_panel = MagicMock()
    server._evaluate_within_spec_for_inference = MagicMock(return_value={"converted": True})
    server.heatmap_manager = SimpleNamespace(base_dir=tmp_path / "heatmaps", save_panel_heatmaps=lambda **kw: {})
    server.db = CAPIDatabase(str(tmp_path / "live.db"))
    server.save_overview = server.save_tile_detail = False
    parsed = {"glass_id": "SIDE1", "model_id": "MODEL", "machine_no": "CAPI1",
              "machine_judgment": "NG", "resolution": (500, 300), "image_dir": str(tmp_path)}
    judgment, ng_details, results, *_ = server._process_request(parsed)
    assert judgment == ("NG" if other_ng else "OK-s")
    assert json.loads(ng_details)[-1]["defect_code"] == "CSJBL"  # Original NG evidence remains available.
    server._evaluate_within_spec_for_inference.assert_not_called()
    response = build_dual_protocol_response(parsed, judgment, results, config)
    if other_ng:
        assert "CSJBL0045000100W0F00000" in response
    else:
        assert response == "AOI@SIDE1;MODEL;CAPI1;NG;OK\r\n@QJPG-SIDE1;OK;MA;OK,"
    server._save_results_async(("test", 1), parsed, results, judgment, ng_details,
                               "2026-10-30 12:00:00", "2026-10-30 12:00:01", .1,
                               aoi_report=report, client_response_text=response, side_white_enabled=True)
    inspect.assert_called_once()
    saved = server.db.get_record_detail(1)
    assert saved["ai_judgment"] == judgment and saved["client_response_text"] == response
    assert saved["side_white_result"]["payload"]["inference_decision"] == payload["inference_decision"]
    assert "最終判定 " + judgment in saved["inference_log"]


def test_rerun_applies_same_release_and_preserves_original_sent_response(tmp_path, monkeypatch):
    import capi_server
    row = clear_row()
    payload = copy.deepcopy(row["payload"])
    payload["evidence_context"]["product_resolution"] = [500, 300]
    inspect = MagicMock(return_value=payload)
    monkeypatch.setattr("capi_side_white.inspect_side_white_folder", inspect)
    monkeypatch.setattr(capi_server, "aggregate_judgment", lambda _: ("OK", "[]"))
    monkeypatch.setattr(capi_server, "results_to_db_data", lambda *_: [])
    monkeypatch.setattr(capi_server.InferenceLogCapture, "start_capture", lambda: None)
    monkeypatch.setattr(capi_server.InferenceLogCapture, "stop_capture", lambda: "")
    config = CAPIConfig(side_white_detection_enabled=True)
    result = ImageResult(image_path=tmp_path / "W0F00000.tif", image_size=(500, 300),
                         otsu_bounds=(0, 0, 500, 300), exclusion_regions=[], tiles=[],
                         excluded_tile_count=0, processed_tile_count=0, processing_time=0., mark_text="MA")
    report = {"W0F00000": [AOIReportDefect("CSJBL", 450, 100, "W0F00000")]}
    inferencer = SimpleNamespace(config=config, station_adapter=create_station_adapter("capi"),
        process_panel=MagicMock(return_value=([result], None, False, "", False, None, report)))
    db = CAPIDatabase(str(tmp_path / "rerun.db"))
    record_id, _ = save(db, row)
    with db._get_conn() as conn:
        conn.execute("UPDATE inference_records SET image_dir=?, client_response_text='original NG' WHERE id=?",
                     (str(tmp_path), record_id))
    cls = CAPIWebHandler
    for name, value in {"inferencer": inferencer, "_capi_server_instance": None, "db": db,
                        "_gpu_lock": None, "_rerun_lock": threading.Lock(),
                        "_rerun_tasks": {record_id: {"status": "running"}}, "heatmap_manager": None,
                        "heatmap_base_dir": str(tmp_path / "heatmaps")}.items():
        monkeypatch.setattr(cls, name, value, raising=False)
    cls._rerun_worker(record_id, db.get_record_detail(record_id))
    assert cls._rerun_tasks[record_id]["status"] == "done", cls._rerun_tasks[record_id]
    saved = db.get_record_detail(record_id)
    assert saved["ai_judgment"] == "OK-s"
    assert saved["client_response_text"] == "original NG"
    assert "@QJPG-REPEATED;OK;MA;OK," in saved["inference_log"]
    assert "側拍轉 OK" in saved["inference_log"]
    assert saved["side_white_result"]["payload"]["inference_decision"]["converted"]
    inspect.assert_called_once()


def test_side_inspection_exception_preserves_ng_and_passthrough_reply(tmp_path, monkeypatch):
    from capi_server import apply_side_white_inference
    monkeypatch.setattr("capi_side_white.inspect_side_white_folder", MagicMock(side_effect=OSError("unreadable")))
    row = clear_row()
    parsed = {"image_dir": str(tmp_path), "glass_id": "FAIL", "aoi_passthrough_records": ["CSJBL0045000100W0F00000"]}
    config = CAPIConfig(side_white_detection_enabled=True)
    judgment = apply_side_white_inference(parsed, [], "NG", json.dumps(row["ng_details"]), {}, config,
                                          tmp_path / "evidence", coords=row["aoi_machine_coords"])
    assert judgment == "NG" and parsed["_side_white_payload"]["status"] == "ERROR"
    assert build_dual_protocol_response(parsed, judgment, [], config).endswith(";NGCSJBL0045000100W0F00000,")


@pytest.mark.parametrize("value", [-1, 501, 1.5, True, float("nan")])
def test_invalid_match_tolerances_are_rejected(value):
    with pytest.raises(ValueError, match="距離容錯"):
        normalize_side_white_params({"csjbl_match_tolerance_product_px": value})
