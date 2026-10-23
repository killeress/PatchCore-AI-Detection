"""AOI-to-side geometry, report denominators, and observation-only review storage."""
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import cv2
import numpy as np
import pytest
from jinja2 import Environment, FileSystemLoader

from capi_database import CAPIDatabase
from capi_side_report import compare_side_record, csjbl_points, has_csjbl, render_comparison_image


def coordinate(x=400, y=200, **extra):
    return {"defect_code": "CSJBL", "product_x": x, "product_y": y, **extra}


def mapped_row(rotated=False):
    product = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], np.float32)
    front = np.array([[100, 100], [1100, 100], [1100, 600], [100, 600]], np.float32)
    side = np.array([[200, 200], [1000, 180], [1100, 700], [100, 680]], np.float32)
    to_side = cv2.getPerspectiveTransform(product, side)
    contour = cv2.perspectiveTransform(np.array([[[390, 190]], [[410, 190]], [[410, 210]], [[390, 210]]], np.float32), to_side).reshape(-1, 2).tolist()
    return {"status": "CANDIDATES", "resolution_x": 1000, "resolution_y": 500,
            "aoi_machine_coords": json.dumps({"W0F00000": [coordinate()]}), "ng_details": "[]",
            "payload": {"algorithm": "test", "rotation_applied": rotated,
                        "side_size": [1200, 800], "front_size": [1200, 800],
                        "mapping": {"status": "estimated", "front_polygon": front.tolist(),
                                    "matrix": cv2.getPerspectiveTransform(side, front).tolist()},
                        "candidates": [{"id": 1, "side_contour": contour, "remaining_contours": [contour],
                                        "disposition": "retained", "bomb_ids": []}],
                        "artifacts": {}, "preview_bounds": {"side": [100, 150, 1000, 600]}}}


def record(db, *, coords="", ng_details="[]", status=None, when="2026-10-22 12:00:00", machine="CAPI1", glass="SAMPLE", payload=None):
    record_id = db.save_inference_record(
        glass_id=glass, model_id="MODEL", machine_no=machine, resolution=(1000, 500),
        machine_judgment="NG", ai_judgment="NG", image_dir="samples", total_images=1, ng_images=1,
        ng_details=ng_details, request_time=when, response_time=when, processing_seconds=.1,
        client_response_text="NG;CSJBL0040000200W0F00000,", aoi_machine_coords=coords)
    result_id = None
    if status:
        result_id = db.save_side_white_result(record_id, payload or {"status": status, "algorithm": "test", "candidates": []})
    return record_id, result_id


@pytest.fixture
def db(tmp_path):
    return CAPIDatabase(str(tmp_path / "results.db"))


def query(db, **kwargs):
    return db.get_side_statistics(start_date="2026-10-22", end_date="2026-10-22", **kwargs)


def test_only_exact_aoi_code_or_legacy_passthrough_is_classified():
    assert has_csjbl(json.dumps({"W0F00000": [coordinate()]}), "") == 1
    assert has_csjbl(json.dumps({"W0F00000": [coordinate(defect_code="csjbl")]}), "") == 1
    assert has_csjbl(json.dumps({"CSJBL": [{"defect_code": "OTHER", "note": "CSJBL"}]}), "") == 0
    legacy = json.dumps([{**coordinate(), "image": "W0F00000", "type": "aoi_passthrough"}])
    assert csjbl_points("", legacy)[1] == "legacy"
    assert has_csjbl("broken JSON", legacy) == 1
    assert has_csjbl(json.dumps({"W0F00000": []}), legacy) == 0  # authoritative snapshot
    assert has_csjbl("", json.dumps([coordinate()])) == 0  # AI findings are not AOI tags
    assert has_csjbl("null", "invalid") == 0


@pytest.mark.parametrize("rotated", [False, True])
def test_product_coordinates_map_through_front_plane_and_keep_raw_orientation(rotated):
    row = mapped_row(rotated)
    result = compare_side_record(row, tolerance=0)
    point = result["points"][0]
    assert point["status"] == "matched" and result["matched"] == result["mapped"] == 1
    assert point["retained_candidate_ids"] == [1]
    assert np.allclose(point["front_xy"], [500, 300])
    assert not np.allclose(point["side_xy"], point["product_xy"])
    for kind in ("side", "front"):
        expected = np.array([1199, 799]) - point[kind + "_xy"] if rotated else point[kind + "_xy"]
        assert np.allclose(point[kind + "_raw_xy"], expected)


def test_distance_uses_contour_boundary_and_product_pixels():
    row = mapped_row()
    row["aoi_machine_coords"] = json.dumps({"W0F00000": [coordinate(440, 200), coordinate(800, 400)]})
    assert compare_side_record(row, 31)["points"][0]["status"] == "matched"
    assert compare_side_record(row, 29)["points"][0]["status"] == "unmatched"
    result = compare_side_record(row, 31)
    assert result["points"][1]["status"] == "unmatched" and result["mapped"] == 2
    assert result["points"][1]["side_xy"]  # a missed point still has a location and crop


@pytest.mark.parametrize("disposition,bombs", [("excluded", []), ("dust_suppressed", []), ("retained", [1])])
def test_excluded_dust_and_bombs_are_not_counted_as_retained_matches(disposition, bombs):
    row = mapped_row()
    row["payload"]["candidates"][0].update(disposition=disposition, bomb_ids=bombs)
    point = compare_side_record(row)["points"][0]
    assert point["status"] == "filtered" and point["candidate_ids"] == [1]
    assert point["retained_candidate_ids"] == []


def test_removed_part_of_candidate_is_not_a_retained_match():
    row = mapped_row()
    row["payload"]["candidates"][0]["remaining_contours"] = []
    assert compare_side_record(row)["points"][0]["status"] == "filtered"


def test_missing_mapping_invalid_coordinates_and_failed_detection_stay_unverified():
    row = mapped_row()
    row["payload"]["mapping"] = {"status": "unavailable", "reason": "no paired front"}
    assert compare_side_record(row)["points"][0]["status"] == "unavailable"
    row = mapped_row()
    row["aoi_machine_coords"] = json.dumps({"W0F00000": [coordinate(-1, 5), coordinate(1001, 200)]})
    assert compare_side_record(row)["mapped"] == 0
    row = mapped_row()
    row["status"] = "ERROR"
    assert compare_side_record(row)["points"][0]["status"] == "unverified"
    assert compare_side_record(row)["matched"] == 0
    row["status"] = "CANDIDATES"
    row["payload"]["truncated"] = True
    assert compare_side_record(row)["points"][0]["status"] == "unverified"


def test_image_coordinates_rotate_before_mapping_and_require_same_front_exposure():
    row = mapped_row(rotated=True)
    row["aoi_machine_coords"] = json.dumps({"W0F00000": [coordinate(699, 499, coordinate_space="image", source_x=699, source_y=499)]})
    point = compare_side_record(row)["points"][0]
    assert np.allclose(point["product_xy"], [400, 200]) and point["status"] == "matched"
    row["aoi_machine_coords"] = row["aoi_machine_coords"].replace("W0F00000", "R0F00000")
    assert compare_side_record(row)["mapped"] == 0


@pytest.mark.parametrize("tolerance", [-1, 501, float("nan"), float("inf")])
def test_bad_tolerance_rejected_even_without_points(tolerance):
    with pytest.raises(ValueError):
        compare_side_record({}, tolerance)


def test_previews_apply_saved_crop_and_resize_and_reject_external_paths(tmp_path):
    root = tmp_path / "heatmaps"
    root.mkdir()
    image_path = root / "side.jpg"
    assert cv2.imwrite(str(image_path), np.full((300, 500, 3), 90, np.uint8))
    row = mapped_row()
    row["payload"]["artifacts"]["side"] = str(image_path)
    comparison = compare_side_record(row)
    image = render_comparison_image(row, comparison, root)
    xy = np.rint((np.array(comparison["points"][0]["side_xy"]) - [100, 150]) * .5).astype(int)
    assert np.array_equal(image[xy[1], xy[0]], [255, 255, 0])
    crop = render_comparison_image(row, comparison, root, point_id=1)
    assert 90 <= crop.shape[0] < 300 and 90 <= crop.shape[1] < 500
    assert np.all(crop == 90)  # local evidence is not covered by the AOI marker
    with pytest.raises(FileNotFoundError):
        render_comparison_image(row, comparison, root / "different")
    with pytest.raises(FileNotFoundError):
        render_comparison_image(row, comparison, root, point_id=2)
    image_path.unlink()
    with pytest.raises(FileNotFoundError):
        render_comparison_image(row, comparison, root)


def test_groups_and_summary_are_independent_of_pagination_and_include_not_run(db):
    coords = json.dumps({"W0F00000": [coordinate()]})
    record(db, coords=coords, status="CANDIDATES", glass="A_1")
    _, reviewed = record(db, coords=coords, status="NO_CANDIDATES", when="2026-10-22 23:59:59")
    pending_id, _ = record(db, coords=coords)  # inspection disabled or queued
    record(db, status="CANDIDATES")
    record(db, status="NO_IMAGE")
    record(db, status="ERROR")
    record(db, coords=coords, status="CANDIDATES", when="2026-10-23 00:00:00")
    record(db, coords=coords, status="CANDIDATES", machine="CAPI2")
    db.save_side_statistics_review(reviewed, "missed", "AOI point visible", "QA", "")
    result = query(db, machine_no="CAPI1", limit=1)
    assert result["total"] == 3 and len(result["rows"]) == 1
    assert result["rows"][0]["status"] == "NOT_RUN"
    assert result["rows"][0]["record_id"] == pending_id
    assert result["summary"]["with_csjbl"] == {
        "total": 3, "candidates": 1, "clear": 1, "unavailable": 1,
        "unreviewed": 2, "confirmed": 0, "correct_clear": 0, "false_positive": 0, "missed": 1,
        "mapping_error": 0, "uncertain": 0, "verified": 1, "accuracy": 0.0}
    assert result["summary"]["without_csjbl"]["total"] == 3
    assert result["summary"]["without_csjbl"]["unavailable"] == 2
    assert result["summary"]["without_csjbl"]["accuracy"] is None
    assert query(db, machine_no="CAPI1", offset=1, limit=1)["summary"] == result["summary"]
    assert query(db, machine_no="CAPI1", group="without_csjbl")["total"] == 3
    assert query(db, glass_id="A_1")["total"] == 1
    assert query(db, glass_id="%")['total'] == 0  # literal search, not wildcard
    assert query(db, glass_id="' OR 1=1 --")["total"] == 0
    assert query(db, status="NOT_RUN")["total"] == 1
    assert query(db, review="missed")["total"] == 1
    assert query(db, model_id="OTHER")["total"] == 0
    assert query(db, include_rows=False)["rows"] == []


def test_review_persists_and_cannot_change_formal_verdict_or_overwrite_newer_review(db):
    record_id, result_id = record(db, status="CANDIDATES")
    before = db.get_record_detail(record_id)
    saved = db.save_side_statistics_review(result_id, "confirmed", " defect verified ", " QA ", "")
    assert saved["review_note"] == "defect verified" and saved["reviewed_by"] == "QA"
    after = db.get_record_detail(record_id)
    for field in ("ai_judgment", "ng_details", "client_response_text", "machine_judgment", "ng_images"):
        assert after[field] == before[field]
    assert after["side_white_result"]["payload"] == before["side_white_result"]["payload"]
    with pytest.raises(RuntimeError, match="其他人"):
        db.save_side_statistics_review(result_id, "missed", "stale", "QA2", "")
    db.save_side_statistics_review(result_id, "uncertain", "need evidence", "QA", saved["reviewed_at"])
    summary = query(db, group="without_csjbl")["summary"]["without_csjbl"]
    assert summary["uncertain"] == 1 and summary["verified"] == 0 and summary["accuracy"] is None
    # A rerun deletes the old observation; an old form cannot annotate its replacement.
    with db._get_conn() as conn:
        conn.execute("DELETE FROM side_white_results WHERE id=?", (result_id,))
    new_id = db.save_side_white_result(record_id, {"status": "NO_CANDIDATES", "candidates": []})
    with pytest.raises(LookupError):
        db.save_side_statistics_review(result_id, "confirmed", "old", "QA", saved["reviewed_at"])
    assert db.get_side_white_result(new_id)["review_decision"] == "unreviewed"


@pytest.mark.parametrize("field,value", [("decision", "bogus"), ("note", "x" * 2001), ("note", []), ("reviewer", " "), ("expected_reviewed_at", None)])
def test_review_rejects_bad_input_without_writing(db, field, value):
    _, result_id = record(db, status="NO_IMAGE")
    values = dict(result_id=result_id, decision="uncertain", note="", reviewer="QA", expected_reviewed_at="")
    values[field] = value
    with pytest.raises(ValueError):
        db.save_side_statistics_review(**values)
    assert db.get_side_white_result(result_id)["reviewed_at"] == ""


def test_report_row_escapes_annotations_and_shows_no_match_crops():
    row = mapped_row()
    row.update(id=2, record_id=1, candidate_count=0, glass_id='<script>bad()</script>',
               review_decision="unreviewed", review_note='<img src=x onerror=bad()>',
               reviewed_at="", reviewed_by='" onfocus="bad()')
    row["payload"]["candidates"] = []
    row["payload"]["artifacts"] = {"side": "side.jpg"}
    from capi_side_report import REVIEW_DECISIONS
    env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parents[1] / "templates"))
    html = env.get_template("_side_statistics_row.html").render(row=row, comparison=compare_side_record(row), review_decisions=REVIEW_DECISIONS)
    assert '<script>bad()</script>' not in html and '&lt;script&gt;bad()&lt;/script&gt;' in html
    assert '<img src=x onerror=bad()>' not in html
    assert '附近未找到候選' in html and '&amp;point=1' in html
    assert 'data-result-id="2"' in html and '儲存驗證結果' in html


def test_side_report_dependencies_are_in_codeonly_release_inputs():
    from scripts.build_deploy_zip import CODE_FILES, _is_codeonly_excluded_file
    for path in ("capi_side_report.py", "templates/_side_statistics.html",
                 "templates/_side_statistics_row.html", "static/js/side_statistics.js"):
        assert path in CODE_FILES
        assert not _is_codeonly_excluded_file(path)


def test_http_report_images_and_review_roundtrip(db, tmp_path):
    from capi_web import CAPIWebHandler

    payload = mapped_row()["payload"]
    payload["status"] = "NO_CANDIDATES"
    payload["candidates"] = []
    image_path = tmp_path / "side.jpg"
    cv2.imwrite(str(image_path), np.full((300, 500, 3), 90, np.uint8))
    payload["artifacts"]["side"] = str(image_path)
    _, result_id = record(db, coords=json.dumps({"W0F00000": [coordinate()]}), status="NO_CANDIDATES", payload=payload)

    class Handler(CAPIWebHandler):
        def _current_settings_user(self):
            return None

        def log_message(self, *args):
            pass

    Handler.db = db
    Handler.heatmap_base_dir = str(tmp_path)
    Handler.init_jinja()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = build_opener(ProxyHandler({}))
    base = f"http://127.0.0.1:{server.server_port}"

    def get(path):
        with opener.open(base + path, timeout=10) as response:
            return response.read()

    def post(data):
        request = Request(base + "/api/ric/side-stats/review", data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=10) as response:
            return json.load(response)

    try:
        query_url = "/api/ric/side-stats?start_date=2026-10-22&end_date=2026-10-22"
        data = json.loads(get(query_url))
        assert data["total"] == 1 and "附近未找到候選" in data["rows_html"]
        assert get(f"/api/ric/side-stats/image?id={result_id}&point=1").startswith(b"\x89PNG")
        with pytest.raises(HTTPError) as error:
            get(query_url + "&tolerance=nan")
        assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            post([])
        assert error.value.code == 400
        saved = post({"id": result_id, "decision": "missed", "reviewer": "QA", "note": "AOI confirmed", "expected_reviewed_at": ""})
        assert saved["success"]
        data = json.loads(get(query_url + "&summary_only=1"))
        assert data["rows_html"] == "" and data["summary"]["with_csjbl"]["missed"] == 1
        assert "AOI confirmed" in json.loads(get(query_url))["rows_html"]
        with pytest.raises(HTTPError) as error:
            post({"id": result_id, "decision": "confirmed", "reviewer": "QA", "expected_reviewed_at": ""})
        assert error.value.code == 409
        with pytest.raises(HTTPError) as error:
            get(f"/api/ric/side-stats/image?id={result_id}&kind=../../private")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
