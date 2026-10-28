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


def marked_row(rotated=False):
    row = mapped_row(rotated)
    row["payload"]["front_image"] = "W0F00000_120000.tif"
    row["payload"]["evidence_context"] = {
        "mark_rotation_applied": rotated,
        "inference_marks": [{"source_image": "W0F00000_120000.tif", "name": "mark_binary",
                             "front_bbox": [480, 280, 40, 40]}],
    }
    return row


@pytest.mark.parametrize("rotated", [False, True])
def test_mark_point_can_clear_without_clearing_other_aoi_points(rotated):
    row = marked_row(rotated)
    row["aoi_machine_coords"] = json.dumps({"W0F00000": [coordinate(), coordinate(800, 400)]})
    result = compare_side_record(row)
    assert result["mark_suppressed"] == 1 and result["matched"] == 0
    assert result["points"][0]["status"] == "mark_suppressed"
    assert result["points"][0]["retained_candidate_ids"] == []
    assert result["points"][1]["status"] == "unmatched"
    # Report suggestions never modify the saved candidate or formal verdict.
    assert row["payload"]["candidates"][0]["disposition"] == "retained"


def test_mark_does_not_expand_with_report_tolerance_or_hide_nearby_defect():
    row = marked_row()
    row["aoi_machine_coords"] = json.dumps({"W0F00000": [coordinate(420.1, 200)]})
    assert compare_side_record(row, 500)["mark_suppressed"] == 0
    row = marked_row()
    contour = np.array(row["payload"]["candidates"][0]["side_contour"]) + [25, 0]
    row["payload"]["candidates"].append({"id": 2, "side_contour": contour.tolist(), "disposition": "retained"})
    result = compare_side_record(row, 50)
    assert result["mark_suppressed"] == 0
    assert result["points"][0]["retained_candidate_ids"] == [2]
    assert "範圍外" in result["points"][0]["reason"]


@pytest.mark.parametrize("invalid", ["source", "rotation", "mapping", "incomplete", "failed", "bbox"])
def test_mark_without_valid_evidence_never_recommends_ok(invalid):
    row = marked_row()
    if invalid == "source":
        row["payload"]["evidence_context"]["inference_marks"][0]["source_image"] = "W0F00000_OLD.tif"
    elif invalid == "rotation":
        row["payload"]["evidence_context"]["mark_rotation_applied"] = True
    elif invalid == "mapping":
        row["payload"]["mapping"]["matrix"] = [[0] * 3] * 3
    elif invalid == "incomplete":
        row["payload"]["truncated"] = True
    elif invalid == "failed":
        row["status"] = "ERROR"
    else:
        row["payload"]["evidence_context"]["inference_marks"][0]["front_bbox"][0] = float("nan")
    assert compare_side_record(row)["mark_suppressed"] == 0


def test_mark_main_box_and_underline_are_copied_without_mutable_state():
    from capi_side_report import inference_mark_evidence
    image = {"mark_source_image": "W0F00000.tif", "mark_exclusion_regions": [
        {"name": "mark_binary", "x1": 10, "y1": 20, "x2": 40, "y2": 50},
        {"name": "mark_underline", "x1": 10, "y1": 55, "x2": 40, "y2": 60}]}
    evidence = inference_mark_evidence([image, image])
    assert len(evidence) == 2
    image["mark_exclusion_regions"][0]["x1"] = 999
    assert evidence[0]["front_bbox"] == [10, 20, 30, 30]
    assert evidence[1]["name"] == "mark_underline"


def test_old_report_uses_saved_inference_mark_without_mutating_observation(db):
    row = marked_row()
    row["payload"].pop("evidence_context")
    row["payload"]["status"] = "CANDIDATES"
    record_id, result_id = record(db, coords=row["aoi_machine_coords"], status="CANDIDATES", payload=row["payload"])
    with db._get_conn() as conn:
        conn.execute("INSERT INTO image_results(record_id, image_path, image_name, mark_bbox, mark_source_image) "
                     "VALUES (?, ?, ?, ?, ?)", (record_id, "samples/W0F00000_120000.tif", "W0F00000_120000.tif",
                                               "480,280,40,40", "W0F00000_120000.tif"))
    for saved in (query(db)["rows"][0], db.get_side_statistics_record(result_id)):
        assert compare_side_record(saved)["mark_suppressed"] == 1
    assert db.get_side_white_result(result_id)["payload"] == row["payload"]
    # An explicit empty current snapshot must not resurrect historical bboxes.
    row["payload"]["evidence_context"] = {"inference_marks": []}
    with db._get_conn() as conn:
        conn.execute("UPDATE side_white_results SET payload=? WHERE id=?", (json.dumps(row["payload"]), result_id))
    assert compare_side_record(db.get_side_statistics_record(result_id))["mark_suppressed"] == 0


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
    # When an unannotated preview exists, point crops use it instead of the marked preview.
    clean_path = root / "side_clean.jpg"
    assert cv2.imwrite(str(clean_path), np.full((300, 500), 40, np.uint8))
    row["payload"]["artifacts"]["side_clean"] = str(clean_path)
    row["payload"]["preview_bounds"]["side_clean"] = row["payload"]["preview_bounds"]["side"]
    assert np.all(render_comparison_image(row, comparison, root, point_id=1) == 40)
    assert np.array_equal(render_comparison_image(row, comparison, root)[xy[1], xy[0]], [255, 255, 0])
    del row["payload"]["artifacts"]["side_clean"]
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
    assert set(result["summary"]) == {"with_csjbl"}
    assert query(db, machine_no="CAPI1", offset=1, limit=1)["summary"] == result["summary"]
    with pytest.raises(ValueError, match="僅提供"):
        query(db, machine_no="CAPI1", group="without_csjbl")
    assert query(db, glass_id="A_1")["total"] == 1
    assert query(db, glass_id="%")['total'] == 0  # literal search, not wildcard
    assert query(db, glass_id="' OR 1=1 --")["total"] == 0
    assert query(db, status="NOT_RUN")["total"] == 1
    assert query(db, review="missed")["total"] == 1
    # Merged summary categories filter with comma-separated values.
    assert query(db, status="NOT_RUN,NO_IMAGE,ERROR")["total"] == 1
    assert query(db, review="false_positive,missed")["total"] == 1
    assert query(db, review="confirmed,correct_clear")["total"] == 0
    with pytest.raises(ValueError):
        query(db, status="CANDIDATES,BOGUS")
    with pytest.raises(ValueError):
        query(db, review="missed,bogus")
    assert query(db, model_id="OTHER")["total"] == 0
    assert query(db, include_rows=False)["rows"] == []


def test_review_persists_and_cannot_change_formal_verdict_or_overwrite_newer_review(db):
    record_id, result_id = record(db, coords=json.dumps({"W0F00000": [coordinate()]}), status="CANDIDATES")
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
    summary = query(db)["summary"]["with_csjbl"]
    assert summary["uncertain"] == 1 and summary["verified"] == 0 and summary["accuracy"] is None
    # A rerun deletes the old observation; an old form cannot annotate its replacement.
    with db._get_conn() as conn:
        conn.execute("DELETE FROM side_white_results WHERE id=?", (result_id,))
    new_id = db.save_side_white_result(record_id, {"status": "NO_CANDIDATES", "candidates": []})
    with pytest.raises(LookupError):
        db.save_side_statistics_review(result_id, "confirmed", "old", "QA", saved["reviewed_at"])
    assert db.get_side_white_result(new_id)["review_decision"] == "unreviewed"


def test_detail_filters_keep_overview_scope_and_refresh_after_review(db):
    coords = json.dumps({"W0F00000": [coordinate()]})
    ids = []
    for status, decision in (("CANDIDATES", "confirmed"), ("NO_CANDIDATES", "correct_clear"),
                             ("CANDIDATES", "false_positive"), ("FILTERED", "unreviewed")):
        _, result_id = record(db, coords=coords, status=status, glass="SCOPE")
        ids.append(result_id)
        if decision != "unreviewed":
            db.save_side_statistics_review(result_id, decision, "", "QA", "")
    record(db, glass="SCOPE")  # Other AOI group, not yet inspected.
    record(db, coords=coords, status="ERROR", machine="OTHER", glass="SCOPE")
    record(db, coords=coords, status="ERROR", when="2026-10-23 00:00:00", glass="SCOPE")
    record(db, coords=coords, status="ERROR", glass="OTHER")
    scope = dict(machine_no="CAPI1", model_id="MODEL", glass_id="SCOPE")
    baseline = query(db, **scope)["summary"]
    assert baseline["with_csjbl"]["accuracy"] == 66.7
    assert baseline["with_csjbl"]["total"] == 4
    assert "without_csjbl" not in baseline
    for filters, expected in (
        ({"review": "confirmed,correct_clear"}, 2),
        ({"review": "false_positive,missed"}, 1),
        ({"status": "CANDIDATES", "review": "confirmed,correct_clear"}, 1),
        ({"review": "mapping_error,uncertain"}, 0),
        ({"status": "NOT_RUN,NO_IMAGE,ERROR"}, 0),
    ):
        for include_rows in (True, False):
            filtered = query(db, **scope, **filters, limit=1, offset=1, include_rows=include_rows)
            assert filtered["summary"] == baseline
            assert filtered["total"] == expected
            assert len(filtered["rows"]) == (min(1, max(0, expected - 1)) if include_rows else 0)
    matching = query(db, **scope, status="CANDIDATES", review="confirmed,correct_clear")
    assert [row["id"] for row in matching["rows"]] == [ids[0]]
    db.save_side_statistics_review(ids[-1], "correct_clear", "", "QA", "")
    refreshed = query(db, **scope, review="unreviewed", include_rows=False)
    assert refreshed["total"] == 0
    assert refreshed["summary"]["with_csjbl"]["accuracy"] == 75.0
    assert refreshed["summary"]["with_csjbl"]["total"] == 4
    assert query(db, **dict(scope, model_id="OTHER"))["summary"]["with_csjbl"]["total"] == 0


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


def test_mark_suggestion_is_shown_per_point_and_keeps_formal_ng_label():
    from capi_side_report import REVIEW_DECISIONS
    row = marked_row()
    row.update(id=2, record_id=1, ai_judgment="NG", review_decision="unreviewed")
    row["payload"]["candidates"] = []
    row["aoi_machine_coords"] = json.dumps({"W0F00000": [coordinate(), coordinate(800, 400)]})
    env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parents[1] / "templates"))
    html = env.get_template("_side_statistics_row.html").render(
        row=row, comparison=compare_side_record(row), review_decisions=REVIEW_DECISIONS)
    assert "MARK 屏蔽／可轉 OK：1 點" in html
    assert "正式結果：NG" in html and "附近未找到候選" in html


def test_side_report_dependencies_are_in_codeonly_release_inputs():
    from scripts.build_deploy_zip import CODE_FILES, _is_codeonly_excluded_file
    for path in ("capi_side_report.py", "templates/_side_statistics.html",
                 "templates/_side_statistics_row.html", "static/js/side_statistics.js"):
        assert path in CODE_FILES
        assert not _is_codeonly_excluded_file(path)


@pytest.mark.parametrize("rotated", [False, True])
def test_original_crop_keeps_file_pixels_depth_and_orientation(tmp_path, rotated):
    from capi_side_report import crop_original_image
    # 16-bit, 3-channel position code (row, col, constant): any rescale, rotation
    # or colour conversion changes the values, and crop[0, 0] reveals its origin.
    yy, xx = np.mgrid[0:800, 0:1200]
    raw = np.dstack([yy, xx, np.full_like(yy, 40000)]).astype(np.uint16)
    assert cv2.imwrite(str(tmp_path / "SW0F00000.tif"), raw)
    row = mapped_row(rotated)
    row["payload"]["side_image"] = "SW0F00000.tif"
    comparison = compare_side_record(row)
    crop = crop_original_image(row, comparison, tmp_path, "side", 1)
    assert crop.dtype == np.uint16 and crop.shape[2] == 3 and 0 < crop.size < raw.size
    xy = np.array(comparison["points"][0]["side_xy"])
    if rotated:
        xy = np.array([1200, 800]) - 1 - xy
    # The crop is an exact, unrotated slice of the file containing the mapped point.
    top, left = int(crop[0, 0, 0]), int(crop[0, 0, 1])
    assert np.array_equal(crop, raw[top:top + crop.shape[0], left:left + crop.shape[1]])
    assert left <= xy[0] < left + crop.shape[1] and top <= xy[1] < top + crop.shape[0]
    with pytest.raises(FileNotFoundError):
        crop_original_image(row, comparison, tmp_path / "missing", "side", 1)
    row["payload"]["side_image"] = "../SW0F00000.tif"
    with pytest.raises(FileNotFoundError):
        crop_original_image(row, comparison, tmp_path, "side", 1)


def test_http_report_images_and_review_roundtrip(db, tmp_path):
    from capi_web import CAPIWebHandler

    payload = mapped_row()["payload"]
    payload["status"] = "NO_CANDIDATES"
    payload["candidates"] = []
    image_path = tmp_path / "side.jpg"
    cv2.imwrite(str(image_path), np.full((300, 500, 3), 90, np.uint8))
    payload["artifacts"]["side"] = str(image_path)
    omit_path = tmp_path / "omit.jpg"
    cv2.imwrite(str(omit_path), np.full((300, 500), 25, np.uint8))
    payload["artifacts"]["omit"] = str(omit_path)
    payload["preview_bounds"]["omit"] = payload["preview_bounds"]["side"]
    payload["omit"] = {"image": "SPINIGBI _120000.tif", "status": "available",
                       "coordinate_space": "side_detection_pixels", "size": [1200, 800]}
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
        assert 'id="ss-summary"' in get("/ric?tab=side-statistics").decode("utf-8")
        query_url = "/api/ric/side-stats?start_date=2026-10-22&end_date=2026-10-22"
        data = json.loads(get(query_url))
        assert data["total"] == 1 and "附近未找到候選" in data["rows_html"]
        assert data["conversion_summary"]["scope"]["denominator"] == 1
        assert data["conversion_summary"]["scope"]["before_pct"] == 100
        eligible = json.loads(get(query_url + "&conversion=eligible"))
        assert eligible["total"] == 0 and eligible["conversion_summary"] == data["conversion_summary"]
        with pytest.raises(HTTPError) as error:
            get(query_url + "&conversion=unknown")
        assert error.value.code == 400
        assert get(f"/api/ric/side-stats/image?id={result_id}&point=1").startswith(b"\x89PNG")
        assert "OMIT 表面證據" in data["rows_html"] and "kind=omit" in data["rows_html"]
        assert get(f"/api/side-white/image?id={result_id}&kind=omit").startswith(b"\xff\xd8")
        assert get(f"/api/ric/side-stats/image?id={result_id}&kind=omit").startswith(b"\x89PNG")
        # With the AOI original still on disk, the point crop is cut from it untouched.
        original = np.random.default_rng(0).integers(0, 256, (800, 1200), dtype=np.uint8)
        assert cv2.imwrite(str(tmp_path / "SW0F00000.png"), original)
        with db._get_conn() as conn:
            conn.execute("UPDATE inference_records SET image_dir=?", (str(tmp_path),))
            row_payload = dict(payload, side_image="SW0F00000.png")
            conn.execute("UPDATE side_white_results SET payload=? WHERE id=?", (json.dumps(row_payload), result_id))
        with opener.open(base + f"/api/ric/side-stats/image?id={result_id}&point=1", timeout=10) as response:
            assert response.headers["X-Side-Image-Source"] == "original"
            served = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        assert served.ndim == 2 and served.size < original.size
        top_left = np.argwhere(cv2.matchTemplate(original, served, cv2.TM_SQDIFF) == 0)
        assert len(top_left) == 1
        yy, xx = np.mgrid[0:800, 0:1200]
        omit_original = np.dstack([yy, xx, np.full_like(yy, 40000)]).astype(np.uint16)
        omit_file = tmp_path / "SPINIGBI _120000.tif"
        cv2.imwrite(str(omit_file), omit_original)
        with opener.open(base + f"/api/ric/side-stats/image?id={result_id}&kind=omit&point=1", timeout=10) as response:
            assert response.headers["X-Side-Image-Source"] == "original"
            served = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        top, left = map(int, served[0, 0, :2])
        assert served.dtype == np.uint16
        assert np.array_equal(served, omit_original[top:top + served.shape[0], left:left + served.shape[1]])
        omit_file.unlink()
        fallback = cv2.imdecode(np.frombuffer(get(f"/api/ric/side-stats/image?id={result_id}&kind=omit&point=1"), np.uint8), cv2.IMREAD_UNCHANGED)
        assert np.all(fallback == 25)
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
        filtered = json.loads(get(query_url + "&review=confirmed,correct_clear&summary_only=1"))
        assert filtered["total"] == 0 and filtered["summary"] == data["summary"]
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
