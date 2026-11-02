"""Mixed front and CSJBL defects keep independent decisions and protocol records."""
import copy
import json
import io
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import capi_server
from capi_config import CAPIConfig
from capi_database import CAPIDatabase
from capi_inference import AOIReportDefect
from capi_server import CAPIServer, build_dual_protocol_response
from capi_side_report import remaining_csjbl_records, side_white_view
from capi_station_adapter import create_station_adapter
from capi_web import CAPIWebHandler, _format_within_spec_inference_note
from test_qjpg_report_response import _image_result, _tile
from test_side_conversion_statistics import sample


@pytest.fixture
def mixed_panel(tmp_path, monkeypatch):
    row = sample(("mark", "omit", "unknown"))
    payload = copy.deepcopy(row["payload"])
    payload.update(status="NO_CANDIDATES", candidates=[])
    payload["evidence_context"]["inference_marks"] = []
    payload["evidence_context"]["product_resolution"] = [500, 300]
    inspect = MagicMock(return_value=payload)
    monkeypatch.setattr("capi_side_white.inspect_side_white_folder", inspect)

    results = []
    for prefix in ("W0F00000", "WGF50500"):
        result = _image_result(f"{prefix}_120000.tif", mark_text="MA")
        tile = _tile(1, 600, 450)
        result.tiles = [tile]
        result.anomaly_tiles = [(tile, .91, None)]
        results.append(result)
    bomb = _tile(2, 700, 450)
    bomb.is_bomb = True
    results[0].tiles.append(bomb)
    results[0].anomaly_tiles.append((bomb, .95, None))
    report = {"W0F00000": [AOIReportDefect("CSJBL", point["product_x"], point["product_y"], "W0F00000")
                            for point in row["ng_details"]]}
    config = CAPIConfig(side_white_detection_enabled=True)
    inferencer = SimpleNamespace(config=config, station_adapter=create_station_adapter("capi"),
        process_panel=MagicMock(return_value=(results, None, False, "", False, None, report)))
    evaluate = MagicMock(return_value={"converted": True, "status": "within_spec", "reason": "front passed",
                                     "suggestion": {"suggested": True}, "detail": {}})
    db = CAPIDatabase(str(tmp_path / "mixed.db"))
    server = CAPIServer.__new__(CAPIServer)
    server.path_mapping = {}
    server.cpu_workers = 1
    server._gpu_lock = threading.Lock()
    server.station_adapter = inferencer.station_adapter
    server._get_or_create_inferencer = lambda _: inferencer
    server._maybe_clear_cuda_cache_after_panel = MagicMock()
    server._evaluate_within_spec_for_inference = evaluate
    server.heatmap_manager = None
    server.db = db
    server.save_overview = server.save_tile_detail = False
    parsed = {"glass_id": "MIXED", "model_id": "MODEL", "machine_no": "CAPI1", "machine_judgment": "NG",
              "resolution": (500, 300), "image_dir": str(tmp_path)}
    return SimpleNamespace(server=server, inferencer=inferencer, parsed=parsed, db=db, results=results,
                           report=report, payload=payload, inspect=inspect, evaluate=evaluate)


def set_side_mode(panel, mode):
    if mode == "retained":
        panel.payload.update(status="CANDIDATES", candidates=[{
            "id": 1, "disposition": "retained", "side_contour": [[450, 90], [460, 90], [460, 110], [450, 110]],
        }])
    elif mode in {"NO_IMAGE", "ERROR"}:
        panel.payload["status"] = mode
    elif mode == "exception":
        panel.inspect.side_effect = OSError("missing side image")
    elif mode == "truncated":
        panel.payload["truncated"] = True
    elif mode == "disabled":
        panel.inferencer.config.side_white_detection_enabled = False


@pytest.mark.parametrize("front_within", [False, True])
@pytest.mark.parametrize("side_mode", ["clear", "retained", "NO_IMAGE", "ERROR", "exception", "truncated", "disabled"])
def test_live_mixed_panel_decisions_reply_and_saved_evidence(mixed_panel, front_within, side_mode):
    p = mixed_panel
    set_side_mode(p, side_mode)
    p.evaluate.return_value.update(converted=front_within, status="within_spec" if front_within else "not_within_spec")
    judgment, details, results, *_rest, within = p.server._process_request(p.parsed)
    assert judgment == ("OK-s" if front_within and side_mode == "clear" else "NG")
    p.evaluate.assert_called_once()
    original = json.loads(details)
    assert len(original) == 5  # Two real front NG tiles and three original CSJBL records.
    assert len(within["detail"]["front_decision"]["resolved_ng_details"]) == (2 if front_within else 0)
    assert not within["detail"]["inference_auto_decision"]["converted_to_ok_i"]
    note = _format_within_spec_inference_note(within, "/within-spec")
    assert "整片判定合併側拍結果" in note and "最終判定 OK-i" not in note

    response = build_dual_protocol_response(p.parsed, judgment, results, p.inferencer.config)
    # Ordinary front points are omitted even when an independent side defect remains.
    assert ("PCDK2" in response) == (not front_within)
    assert "PCDK3" in response  # Bomb reporting follows the existing protocol for OK and NG.
    assert response.count("CSJBL") == (0 if side_mode == "clear" else 1 if side_mode == "retained" else 3)

    p.server._save_results_async(("test", 1), p.parsed, results, judgment, details,
        "2026-10-22 12:00:00", "2026-10-22 12:00:01", .1,
        aoi_report=p.report, client_response_text=response, within_spec_info=within,
        side_white_enabled=p.inferencer.config.side_white_detection_enabled)
    saved = p.db.get_record_detail(1)
    assert saved["ai_judgment"] == judgment and saved["client_response_text"] == response
    assert json.loads(saved["ng_details"]) == original
    assert saved["within_spec_log_id"]
    if side_mode != "disabled":
        p.inspect.assert_called_once()
        side = saved["side_white_result"]
        assert side["payload"]["front_decision"] == p.parsed["front_decision"]
        view = side_white_view(side, saved)
        assert len(view["panel"]["remaining_ng"]) == (0 if judgment == "OK-s" else
            (0 if front_within else 2) + (0 if side_mode == "clear" else 1 if side_mode == "retained" else 3))
    else:
        p.inspect.assert_not_called()


@pytest.mark.parametrize("front_within", [False, True])
@pytest.mark.parametrize("side_mode", ["clear", "retained", "NO_IMAGE", "disabled"])
@pytest.mark.parametrize("server_evaluator", [False, True])
def test_rerun_uses_same_mixed_decisions_and_keeps_original_response(
    mixed_panel, monkeypatch, front_within, side_mode, server_evaluator,
):
    p = mixed_panel
    set_side_mode(p, side_mode)
    p.evaluate.return_value.update(converted=front_within, status="within_spec" if front_within else "not_within_spec")
    # Exercise both the running server evaluator and the standalone rerun evaluator.
    monkeypatch.setattr("capi_web._evaluate_within_spec_suggestion_detail", lambda *a, **kw: {
        "suggestion": {"suggested": True} if front_within else None,
        "panel_totals": [{"within": front_within}],
    })
    record_id = p.db.save_inference_record(**{
        key: p.parsed[key] for key in ("glass_id", "model_id", "machine_no", "machine_judgment", "resolution", "image_dir")
    }, ai_judgment="NG", total_images=2, ng_images=2, ng_details="[]",
        request_time="2026-10-22 12:00:00", response_time="2026-10-22 12:00:01", processing_seconds=.1,
        client_response_text="original NG response")
    cls = CAPIWebHandler
    for name, value in {"inferencer": p.inferencer, "_capi_server_instance": p.server if server_evaluator else None,
                        "_gpu_lock": threading.Lock(), "_rerun_lock": threading.Lock(),
                        "_rerun_tasks": {record_id: {"status": "running"}}, "db": p.db,
                        "heatmap_manager": None, "heatmap_base_dir": str(p.db.db_path.parent / "heatmaps")}.items():
        monkeypatch.setattr(cls, name, value, raising=False)
    cls._rerun_worker(record_id, p.db.get_record_detail(record_id))
    assert cls._rerun_tasks[record_id]["status"] == "done", cls._rerun_tasks[record_id]
    saved = p.db.get_record_detail(record_id)
    assert saved["ai_judgment"] == ("OK-s" if front_within and side_mode == "clear" else "NG")
    assert saved["client_response_text"] == "original NG response"
    assert len(json.loads(saved["ng_details"])) == 5
    preview = saved["inference_log"].split("[RERUN_RESPONSE_PREVIEW]", 1)[1].splitlines()[0]
    assert ("PCDK2" in preview) == (not front_within)
    assert "PCDK3" in preview
    assert preview.count("CSJBL") == (0 if side_mode == "clear" else 1 if side_mode == "retained" else 3)
    assert saved["within_spec_log_id"]
    assert p.evaluate.call_count == int(server_evaluator)


def test_white_frame_ng_cannot_be_released_by_side_or_within_spec(mixed_panel, monkeypatch):
    p = mixed_panel
    monkeypatch.setattr(capi_server, "_has_white_frame_ng", lambda _: True)
    judgment, details, *_ = p.server._process_request(p.parsed)
    assert judgment == "NG"
    p.evaluate.assert_not_called()
    assert len(json.loads(details)) == 5
    assert not p.parsed["aoi_passthrough_records"]


def test_missing_front_ng_details_still_block_side_release(mixed_panel, monkeypatch):
    p = mixed_panel
    monkeypatch.setattr(capi_server, "aggregate_judgment", lambda _: ("NG", "[]"))
    p.evaluate.return_value.update(converted=False)
    assert p.server._process_request(p.parsed)[0] == "NG"


def test_csjbl_filter_requires_exact_screen_and_coordinates():
    records = ["CSJBL0010000100W0F00000", "CSJBL0010000100WGF50500", "CSJBL0010100100W0F00000", "bad"]
    assert remaining_csjbl_records(records, [{"status": "no_anomaly", "image_prefix": "W0F00000",
                                            "product_xy": [100, 100]}]) == records[1:]


@pytest.mark.parametrize("outcome", [None, {"converted": False, "status": "not_evaluable"},
                                     {"converted": False, "status": "check_failed"}])
def test_unevaluable_front_does_not_release_the_panel(mixed_panel, outcome):
    p = mixed_panel
    p.evaluate.return_value = outcome
    judgment, details, results, *_ = p.server._process_request(p.parsed)
    assert judgment == "NG"
    response = build_dual_protocol_response(p.parsed, judgment, results, p.inferencer.config)
    assert "PCDK2" in response and "CSJBL" not in response


def test_regenerated_visuals_keep_the_front_only_decision_scope(monkeypatch):
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    body = json.dumps({"log_id": 1}).encode()
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.heatmap_base_dir = ""
    handler.db = MagicMock()
    handler.db.get_within_spec_review_log.return_value = {
        "inference_record_id": 1, "source": "inference",
        "detail": {"inference_auto_decision": {"scope": "front"}},
    }
    handler.db.get_record_detail.return_value = {"glass_id": "MIXED", "model_id": "MODEL"}
    handler.db.save_within_spec_review_log.return_value = {"id": 2}
    handler._send_json = MagicMock()
    handler._load_within_spec_rules_for_review = lambda: {}
    handler._inference_rotate_180_enabled = lambda: False
    monkeypatch.setattr("capi_web._within_spec_auto_visual_output", lambda *a: (None, ""))
    monkeypatch.setattr("capi_web._evaluate_within_spec_suggestion_detail", lambda *a, **kw: {
        "suggestion": {"suggested": True}, "panel_totals": [{"within": True}],
    })
    handler._handle_within_spec_log_regenerate()
    assert handler._send_json.call_args.args[0]["success"]
    decision = handler.db.save_within_spec_review_log.call_args.kwargs["detail"]["inference_auto_decision"]
    assert decision["scope"] == "front" and decision["front_within_spec_passed"]
    assert not decision["converted_to_ok_i"]
