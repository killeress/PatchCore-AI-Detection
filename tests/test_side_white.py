import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from capi_config import CAPIConfig, normalize_side_white_params
from capi_database import CAPIDatabase
from capi_server import CAPIServer
from capi_side_white import _candidates, detect_panel_quad, find_side_white_pair, inspect_side_white_image
from capi_web import CAPIWebHandler


def _pair(tmp_path, defect=True):
    front = np.zeros((800, 1200), np.uint8)
    cv2.rectangle(front, (80, 70), (1120, 730), 90, -1)
    if defect:
        cv2.ellipse(front, (570, 400), (18, 8), 0, 0, 360, 65, -1)
        cv2.ellipse(front, (570, 424), (18, 8), 0, 0, 360, 118, -1)
        cv2.ellipse(front, (950, 670), (16, 7), 0, 0, 360, 115, -1)
    fq = np.array([[80, 70], [1120, 70], [1120, 730], [80, 730]], np.float32)
    sq = np.array([[120, 160], [930, 170], [1010, 570], [70, 555]], np.float32)
    h = cv2.getPerspectiveTransform(fq, sq)
    side = cv2.warpPerspective(front, h, (1100, 700))
    sp, fp = tmp_path / "SW0F00000_153501.tif", tmp_path / "W0F00000_153501.tif"
    assert cv2.imwrite(str(sp), side)
    assert cv2.imwrite(str(fp), front)
    return sp, fp, h


def _record(db, judgment="OK"):
    return db.save_inference_record(
        glass_id="SIDE-TEST", model_id="MODEL", machine_no="CAPI1", resolution=(1920, 1080),
        machine_judgment="OK", ai_judgment=judgment, image_dir="samples", total_images=1,
        ng_images=0, ng_details="[]", request_time="2026-09-07 12:00:00",
        response_time="2026-09-07 12:00:01", processing_seconds=.1,
        client_response_text="AOI@SIDE-TEST;OK\r\n@QJPG-SIDE-TEST;NG;00;OK,",
    )


def test_only_white_side_and_exact_acquisition_partner_are_selected(tmp_path):
    for name in ["SG0F00000_153501.tif", "SWGF50500_153501.tif", "defect.tif", "W0F00000_153501.tif"]:
        (tmp_path / name).touch()
    assert find_side_white_pair(tmp_path) == (None, None)
    side = tmp_path / "SW0F00000_153501.tif"
    side.touch()
    assert find_side_white_pair(tmp_path) == (side, tmp_path / "W0F00000_153501.tif")
    newer = tmp_path / "SW0F00000_153502.tif"
    newer.touch()
    os.utime(side, (1, 1))
    assert find_side_white_pair(tmp_path) == (newer, None)


def test_glass_prefixed_and_compact_names_match(tmp_path):
    side = tmp_path / "GLASS123SW0F00000153501.TIF"
    front = tmp_path / "GLASS123W0F00000153501.tif"
    side.touch(); front.touch()
    assert find_side_white_pair(tmp_path) == (side, front)


def test_plane_mapping_uses_independent_points(tmp_path):
    side, front, forward = _pair(tmp_path)
    sq = detect_panel_quad(cv2.imread(str(side), 0))
    fq = detect_panel_quad(cv2.imread(str(front), 0))
    backward = cv2.getPerspectiveTransform(sq, fq)
    points = np.array([[[400, 350]], [[900, 630]], [[250, 200]]], np.float32)
    observed = cv2.perspectiveTransform(points, forward)
    mapped = cv2.perspectiveTransform(observed, backward)
    assert np.linalg.norm(mapped - points, axis=2).max() < 3


def test_candidates_have_mapped_contours_and_artifacts(tmp_path):
    side, front, _ = _pair(tmp_path)
    p = inspect_side_white_image(side, front, tmp_path / "out")
    assert p["shadow_only"] is True
    assert p["status"] == "CANDIDATES"
    assert p["mapping"]["status"] == "estimated"
    assert p["candidates"]
    assert any(np.linalg.norm(np.array(c["front_xy"]) - [570, 412]) < 35 for c in p["candidates"])
    assert any(np.linalg.norm(np.array(c["front_xy"]) - [950, 670]) < 25 for c in p["candidates"])
    for candidate in p["candidates"]:
        assert candidate["front_contour"]
        assert candidate["front_raw_xy"] == candidate["front_xy"]
    assert all(Path(path).is_file() for path in p["artifacts"].values())


def test_uniform_panel_has_no_candidates(tmp_path):
    side, front, _ = _pair(tmp_path, defect=False)
    p = inspect_side_white_image(side, front, tmp_path / "out")
    assert p["status"] == "NO_CANDIDATES"
    assert p["candidates"] == []


def test_missing_front_keeps_candidates_and_no_fake_coordinates(tmp_path):
    side, _, _ = _pair(tmp_path)
    p = inspect_side_white_image(side, None, tmp_path / "out")
    assert p["status"] == "CANDIDATES"
    assert p["mapping"]["status"] == "unavailable"
    assert all(c["front_xy"] is None and c["front_raw_xy"] is None for c in p["candidates"])
    assert "front" not in p["artifacts"]


def test_rotation_preserves_raw_image_coordinate_contract(tmp_path):
    side, front, _ = _pair(tmp_path)
    p = inspect_side_white_image(side, front, tmp_path / "out", rotate_180=True)
    assert p["status"] == "CANDIDATES"
    for c in p["candidates"]:
        assert np.allclose(np.array(c["side_xy"]) + c["side_raw_xy"], [1099, 699])
        assert np.allclose(np.array(c["front_xy"]) + c["front_raw_xy"], [1199, 799])
    assert any(np.linalg.norm(np.array(c["front_raw_xy"]) - [570, 412]) < 35 for c in p["candidates"])


def test_unreadable_side_is_error_not_clean(tmp_path):
    side = tmp_path / "bad.tif"
    side.write_bytes(b"not an image")
    p = inspect_side_white_image(side, None, tmp_path / "out")
    assert p["status"] == "ERROR"
    assert p["reason"]


def test_config_default_persistence_and_hot_reload(tmp_path):
    config = CAPIConfig()
    assert config.side_white_detection_enabled is False
    assert config.side_white_detection_params == normalize_side_white_params()
    custom = {"min_contrast_gray": 3.1, "noise_sigma_factor": 6.0, "min_area_px": 30, "edge_margin_px": 25}
    config.apply_db_overrides([{"param_name":"side_white_detection_params", "decoded_value":custom}])
    assert config.side_white_detection_params == custom
    custom["min_area_px"] = 999
    assert config.side_white_detection_params["min_area_px"] == 30
    assert CAPIConfig().side_white_detection_params["min_area_px"] == 20
    snapshot = config.to_dict()["side_white_detection_params"]
    snapshot["min_area_px"] = 500
    assert config.side_white_detection_params["min_area_px"] == 30
    config.apply_db_overrides([{"param_name":"side_white_detection_enabled", "decoded_value":True}])
    assert config.side_white_detection_enabled is True
    assert config.to_dict()["side_white_detection_enabled"] is True
    path = tmp_path / "config.yaml"
    config.to_yaml(str(path))
    assert CAPIConfig.from_yaml(str(path)).side_white_detection_enabled is True
    assert CAPIConfig.from_yaml(str(path)).side_white_detection_params == config.side_white_detection_params
    config.apply_db_overrides([{"param_name":"side_white_detection_enabled", "decoded_value":"false"}])
    assert config.side_white_detection_enabled is False


def test_review_persists_without_changing_formal_record_and_cascades(tmp_path):
    db = CAPIDatabase(str(tmp_path / "results.db"))
    record_id = _record(db)
    before = db.get_record_detail(record_id)
    result_id = db.save_side_white_result(record_id, {"status":"CANDIDATES","candidates":[{"id":1}]})
    assert db.review_side_white_result(result_id, "confirmed", "<script>alert(1)</script>", "reviewer")
    after = db.get_record_detail(record_id)
    for key in ("ai_judgment", "machine_judgment", "ng_images", "total_images", "ng_details", "client_response_text"):
        assert after[key] == before[key]
    assert after["side_white_result"]["review_decision"] == "confirmed"
    assert db.list_side_white_results(review="unreviewed")["total"] == 0
    assert db.list_side_white_results(review="confirmed", machine_no="CAPI1")["total"] == 1
    assert db.list_side_white_results(glass_id="' OR 1=1 --")["total"] == 0
    with pytest.raises(ValueError):
        db.review_side_white_result(result_id, "NG", "", "reviewer")
    with db._get_conn() as conn:
        conn.execute("DELETE FROM inference_records WHERE id=?", (record_id,))
    assert db.get_side_white_result(result_id) is None


@pytest.mark.parametrize("enabled,contrast", [(False, 2.2), (True, 2.2), (True, 60)])
def test_background_branch_is_optional_and_preserves_formal_output(tmp_path, enabled, contrast):
    _pair(tmp_path)
    server = CAPIServer.__new__(CAPIServer)
    server.db = CAPIDatabase(str(tmp_path / "results.db"))
    server.path_mapping = {}
    server.heatmap_manager = SimpleNamespace(base_dir=tmp_path / "heatmaps")
    server._save_results_async(
        client_addr=("127.0.0.1",1), parsed={"glass_id":"SIDE-TEST","model_id":"MODEL",
        "machine_no":"CAPI1","resolution":(1920,1080),"machine_judgment":"OK","image_dir":str(tmp_path)},
        results=[], ai_judgment="OK", ng_details="[]", request_time="2026-09-07 12:00:00",
        response_time="2026-09-07 12:00:01", processing_seconds=.1,
        client_response_text="formal-response", side_white_enabled=enabled,
        side_white_params={"min_contrast_gray": contrast},
    )
    record = server.db.get_record_detail(1)
    assert record["ai_judgment"] == "OK"
    assert record["ng_details"] == "[]"
    assert record["ng_images"] == 0
    assert record["client_response_text"] == "formal-response"
    assert bool(record["side_white_result"]) is enabled
    if enabled:
        assert (record["side_white_result"]["candidate_count"] > 0) is (contrast < 60)
        assert record["side_white_result"]["payload"]["parameters"]["min_contrast_gray"] == contrast
        assert record["heatmap_dir"]


def test_inspection_failure_is_saved_and_cannot_change_verdict(tmp_path, monkeypatch):
    import capi_side_white
    db = CAPIDatabase(str(tmp_path / "results.db")); record_id = _record(db, "NG")
    server = CAPIServer.__new__(CAPIServer); server.db = db; server.path_mapping = {}
    monkeypatch.setattr(capi_side_white, "find_side_white_pair", MagicMock(side_effect=RuntimeError("camera read failed")))
    server._save_side_white_review(record_id, {"image_dir":str(tmp_path)}, {}, False)
    record = db.get_record_detail(record_id)
    assert record["ai_judgment"] == "NG"
    assert record["side_white_result"]["status"] == "ERROR"


def test_image_endpoint_rejects_paths_outside_heatmap_root(tmp_path):
    db = CAPIDatabase(str(tmp_path / "results.db")); record_id = _record(db)
    external = tmp_path / "external.jpg"; external.write_bytes(b"private")
    result_id = db.save_side_white_result(record_id, {"status":"ERROR","artifacts":{"side":str(external)}})
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.db = db; handler.heatmap_base_dir = str(tmp_path / "heatmaps")
    handler._send_404 = MagicMock(); handler.wfile = io.BytesIO()
    handler._handle_api_side_white_image({"id":[str(result_id)],"kind":["side"]})
    handler._send_404.assert_called_once()
    assert handler.wfile.getvalue() == b""


def test_record_review_notes_are_escaped_even_with_legacy_jinja_defaults(tmp_path):
    CAPIWebHandler.init_jinja()
    template = CAPIWebHandler.jinja_env.get_template("_side_white_result.html")
    html = template.render(detail={"glass_id":"G", "side_white_result":{
        "id":1,"status":"ERROR","candidate_count":0,"review_decision":"uncertain",
        "review_note":"<script>alert(1)</script>","payload":{"algorithm":"test","candidates":[]}}})
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_review_api_rejects_non_object_json():
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler._read_json_body = lambda: []
    handler._send_json = MagicMock()
    handler._handle_api_side_white_review()
    assert handler._send_json.call_args.kwargs["status"] == 400


def test_review_routes_require_same_admin_role_as_mark(tmp_path):
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import Request, build_opener, ProxyHandler
    from urllib.error import HTTPError

    db = CAPIDatabase(str(tmp_path / "results.db"))
    record_id = _record(db)
    result_id = db.save_side_white_result(record_id, {"status":"NO_CANDIDATES","candidates":[]})

    class Handler(CAPIWebHandler):
        def _current_settings_user(self):
            role = self.headers.get("X-Test-Role")
            return {"username":"tester","can_manage_accounts":role == "admin"} if role else None

    Handler.db = db
    Handler.inferencer = SimpleNamespace(config=CAPIConfig())
    other_inferencer = SimpleNamespace(config=CAPIConfig())
    Handler._capi_server_instance = SimpleNamespace(inferencers={"other":other_inferencer})
    db.init_config_from_yaml(Handler.inferencer.config)
    params = normalize_side_white_params({"min_contrast_gray": 7.5, "min_area_px": 40})
    update_body = json.dumps({"param_name":"side_white_detection_params", "new_value":params,
                              "reason":"UI test"}).encode()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    opener = build_opener(ProxyHandler({}))
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        for role, status in [(None, 401), ("operator", 403)]:
            headers = {"X-Test-Role":role} if role else {}
            for path, body in [("/api/settings/side-white", None),
                               ("/api/settings/side-white/review", b'{}'),
                               ("/api/settings/update", update_body)]:
                with pytest.raises(HTTPError) as error:
                    opener.open(Request(base + path, data=body, headers=headers), timeout=5)
                assert error.value.code == status
        body = json.dumps({"id":result_id,"decision":"missed","note":"test"}).encode()
        with opener.open(Request(base + "/api/settings/side-white/review", data=body,
                             headers={"X-Test-Role":"admin","Content-Type":"application/json"}), timeout=5) as response:
            assert json.load(response)["success"] is True
        assert db.get_record_detail(record_id)["ai_judgment"] == "OK"
        assert db.get_side_white_result(result_id)["review_decision"] == "missed"
        with opener.open(Request(base + "/api/settings/update", data=update_body,
                             headers={"X-Test-Role":"admin","Content-Type":"application/json"}), timeout=5) as response:
            assert json.load(response)["success"] is True
        assert db.get_config_param("side_white_detection_params")["decoded_value"] == params
        assert Handler.inferencer.config.side_white_detection_params == params
        assert other_inferencer.config.side_white_detection_params == params
        assert db.get_record_detail(record_id)["ai_judgment"] == "OK"
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)


def test_controls_change_detection_of_weak_small_and_near_edge_candidates(tmp_path):
    side, _, transform = _pair(tmp_path)
    gray = cv2.imread(str(side), 0)
    quad = detect_panel_quad(gray)
    baseline, _, _, _, _ = _candidates(gray, quad, normalize_side_white_params())
    assert len(baseline) >= 2
    high_contrast, _, _, _, _ = _candidates(gray, quad, normalize_side_white_params({"min_contrast_gray":60}))
    large_area, _, _, _, _ = _candidates(gray, quad, normalize_side_white_params({"min_area_px":5000}))
    assert high_contrast == [] and large_area == []
    wide_margin, _, _, _, _ = _candidates(gray, quad, normalize_side_white_params({"edge_margin_px":100}))
    center, edge = cv2.perspectiveTransform(np.array([[[570,412]],[[950,670]]], np.float32), transform)[:,0]
    assert any(np.linalg.norm(np.array(c["side_xy"]) - center) < 30 for c in wide_margin)
    assert any(np.linalg.norm(np.array(c["side_xy"]) - edge) < 30 for c in baseline)
    assert not any(np.linalg.norm(np.array(c["side_xy"]) - edge) < 30 for c in wide_margin)
    noisy = np.clip(gray.astype(float) + np.random.default_rng(7).normal(0,4,gray.shape),0,255).astype(np.uint8)
    normal, _, _, normal_threshold, _ = _candidates(noisy, quad, normalize_side_white_params())
    strict, _, _, strict_threshold, _ = _candidates(noisy, quad, normalize_side_white_params({"noise_sigma_factor":100}))
    assert normal and strict == []
    assert strict_threshold > normal_threshold


def test_queued_parameters_do_not_change_when_live_settings_change():
    import threading
    config = CAPIConfig()
    server = CAPIServer.__new__(CAPIServer)
    server._async_executor_lock = threading.Lock()
    server._async_executor_shutdown = False
    server._async_executor = MagicMock()
    server._queue_save_results_async(side_white_params=config.side_white_detection_params)
    queued = server._async_executor.submit.call_args.kwargs["side_white_params"]
    config.side_white_detection_params["min_contrast_gray"] = 99
    assert queued["min_contrast_gray"] == 2.2


def test_missing_image_keeps_parameter_snapshot_after_later_updates(tmp_path):
    db = CAPIDatabase(str(tmp_path / "results.db")); record_id = _record(db)
    server = CAPIServer.__new__(CAPIServer); server.db = db; server.path_mapping = {}
    params = normalize_side_white_params({"min_area_px":35})
    server._save_side_white_review(record_id, {"image_dir":str(tmp_path)}, {}, False, params)
    params["min_area_px"] = 800
    result = db.get_record_detail(record_id)["side_white_result"]
    assert result["status"] == "NO_IMAGE"
    assert result["payload"]["parameters"]["min_area_px"] == 35


@pytest.mark.parametrize("invalid", [
    [], {"unknown":1}, {"min_contrast_gray":0}, {"min_contrast_gray":float('nan')},
    {"noise_sigma_factor":float('inf')}, {"min_area_px":1.5}, {"min_area_px":True},
    {"edge_margin_px":-1}, {"edge_margin_px":513}, {"edge_margin_px":"16"},
])
def test_parameter_api_rejects_invalid_values_before_saving(invalid):
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    body = json.dumps({"param_name":"side_white_detection_params", "new_value":invalid,
                       "reason":"test"}).encode()
    handler.headers = {"Content-Length":str(len(body))}; handler.rfile = io.BytesIO(body)
    handler._current_settings_user = lambda: {"username":"tester","can_manage_accounts":True}
    handler.db = MagicMock(); handler._send_json = MagicMock()
    handler._handle_api_settings_update()
    assert handler._send_json.call_args.kwargs["status"] == 400
    handler.db.update_config_param.assert_not_called()


def test_record_shows_snapshot_without_substituting_current_defaults():
    CAPIWebHandler.init_jinja()
    template = CAPIWebHandler.jinja_env.get_template("_side_white_result.html")
    result = {"id":1,"status":"NO_IMAGE","candidate_count":0,"review_decision":"unreviewed",
              "payload":{"algorithm":"test","candidates":[],"parameters":normalize_side_white_params({"min_contrast_gray":7.5})}}
    html = template.render(detail={"glass_id":"G","side_white_result":result})
    assert "最低局部反差 7.5" in html
    del result["payload"]["parameters"]
    html = template.render(detail={"glass_id":"G","side_white_result":result})
    assert "此筆未保存參數快照" in html
    assert "最低局部反差 2.2" not in html
