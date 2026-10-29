"""Peel-tab evidence is a separate, bounded exclusion, not a blanket corner OK."""
import copy
import json

import cv2
import numpy as np
import pytest

from capi_database import CAPIDatabase
from capi_side_report import compare_side_record, peel_tape_regions, record_conversion, side_white_view
from test_side_sticker_results import sticker_row


def tape_row():
    row = sticker_row()
    points = [{"defect_code": "CSJBL", "product_x": x, "product_y": y} for x, y in [(375, 228), (385, 231)]]
    row.update(status="NO_CANDIDATES", aoi_machine_coords={"W0F00000": points},
               ng_details=[{**p, "image": "W0F00000", "type": "aoi_passthrough"} for p in points])
    payload = row["payload"]
    payload.update(status="NO_CANDIDATES", side_polygon=[[0, 0], [400, 0], [400, 240], [0, 240]],
                   exclusions=[{"type": "corner_tape", "bbox": [365, 222, 28, 12]}], candidates=[])
    payload["evidence_context"]["aoi_csjbl_points"] = [{**p, "image_prefix": "W0F00000"} for p in points]
    return row


@pytest.mark.parametrize("mode", ["off", "observe", "suppress"])
def test_saved_corner_tape_converts_without_candidates_or_omit(mode):
    row = tape_row()
    row["payload"]["parameters"]["dust_mode"] = mode
    row["payload"]["omit"] = {"status": "unavailable"}
    before = copy.deepcopy(row)
    comparison = compare_side_record(row)
    assert comparison["tape_suppressed"] == 2
    assert comparison["mark_suppressed"] == comparison["omit_suppressed"] == 0
    assert all(p["status"] == "tape_suppressed" and p["tape_regions"] == [1] for p in comparison["points"])
    conversion = record_conversion(row, comparison)
    assert conversion["category"] == "tape" and conversion["eligible"]
    view = side_white_view(row)
    assert view["headline"] == "AOI 2 點可轉 OK"
    assert view["tape"]["text"] == "已排除 · 2 點可轉 OK"
    assert row == before


@pytest.mark.parametrize("issue", ["no_detection", "wrong_type", "no_panel", "wrong_corner", "oversize",
                                   "nan", "outside_image", "bad_mapping", "truncated", "error"])
def test_position_or_incomplete_evidence_never_creates_a_tape_ok(issue):
    row = tape_row()
    p = row["payload"]
    # Isolate tape evidence: a valid MARK elsewhere in this large test ROI can
    # independently clear the point under the MARK ROI rule.
    p["evidence_context"]["inference_marks"] = []
    if issue == "no_detection":
        p["exclusions"] = []
    elif issue == "wrong_type":
        p["exclusions"][0]["type"] = "configured_zone"
    elif issue == "no_panel":
        p.pop("side_polygon")
    elif issue == "wrong_corner":
        p["exclusions"][0]["bbox"] = [6, 6, 28, 12]
        for point in row["aoi_machine_coords"]["W0F00000"]:
            point.update(product_x=20, product_y=12)
    elif issue == "oversize":
        p["exclusions"][0]["bbox"] = [300, 200, 100, 40]
    elif issue == "nan":
        p["exclusions"][0]["bbox"][0] = float("nan")
    elif issue == "outside_image":
        p["exclusions"][0]["bbox"] = [375, 222, 28, 12]
    elif issue == "bad_mapping":
        p["mapping"]["matrix"] = [[0, 0, 0]] * 3
    elif issue == "truncated":
        p["truncated"] = True
    else:
        row["status"] = "ERROR"
    comparison = compare_side_record(row, tolerance=500)
    assert comparison["tape_suppressed"] == 0
    assert not record_conversion(row, comparison)["eligible"]


@pytest.mark.parametrize("xy", [(364.9, 228), (393, 228), (375, 221.9), (375, 234)])
def test_aoi_tolerance_never_expands_half_open_tape_bounds(xy):
    row = tape_row()
    for point in row["aoi_machine_coords"]["W0F00000"]:
        point.update(product_x=xy[0], product_y=xy[1])
    assert compare_side_record(row, tolerance=500)["tape_suppressed"] == 0


@pytest.mark.parametrize("rotated", [False, True])
def test_tape_uses_same_detection_orientation_and_perspective_mapping(rotated):
    row = tape_row()
    p = row["payload"]
    p["rotation_applied"] = rotated
    if rotated:
        p["exclusions"][0]["bbox"] = [7, 6, 28, 12]
    side_xy = [21, 12] if rotated else [375, 228]
    p["front_size"] = [1200, 800]
    matrix = np.array([[2, .1, 35], [.1, 2, 50], [.0002, .0001, 1]], np.float64)
    front_quad = cv2.perspectiveTransform(np.array([p["side_polygon"]], np.float32), matrix)[0]
    front_xy = cv2.perspectiveTransform(np.array([[side_xy]], np.float32), matrix)[0, 0]
    p["mapping"].update(matrix=matrix.tolist(), front_polygon=front_quad.tolist())
    raw_xy = np.array(p["front_size"]) - 1 - front_xy if rotated else front_xy
    row["aoi_machine_coords"] = {"W0F00000": [{"defect_code": "CSJBL", "coordinate_space": "image",
                                               "source_x": float(raw_xy[0]), "source_y": float(raw_xy[1])}]}
    comparison = compare_side_record(row, tolerance=0)
    assert comparison["tape_suppressed"] == 1
    assert comparison["points"][0]["side_xy"] == pytest.approx(side_xy, abs=.001)


def test_tape_ignores_neighbours_but_blocks_an_aoi_hit_outside_the_tape():
    row = tape_row()
    p = row["payload"]
    p["candidates"] = [{"id": 2, "disposition": "retained", "side_contour":
                         [[370, 225], [388, 225], [388, 232], [370, 232]]}]
    comparison = compare_side_record(row)
    assert comparison["tape_suppressed"] == 2
    assert side_white_view(row)["candidates"][0]["label"] == "易撕貼已排除"
    p["candidates"].append({"id": 3, "disposition": "retained", "side_contour":
                             [[350, 225], [360, 225], [360, 230], [350, 230]]})
    comparison = compare_side_record(row, tolerance=50)
    assert comparison["tape_suppressed"] == 2
    assert all(pt["retained_candidate_ids"] == [] for pt in comparison["points"])
    assert record_conversion(row, comparison)["eligible"]
    assert side_white_view(row)["tape"]["tone"] == "ok"
    assert compare_side_record(row, tolerance=0)["tape_suppressed"] == 2
    p["candidates"] = [{"id": 4, "disposition": "retained", "side_contour":
                         [[350, 225], [388, 225], [388, 232], [350, 232]]}]
    assert compare_side_record(row, tolerance=0)["tape_suppressed"] == 0


@pytest.mark.parametrize("kind", ["candidate", "polygon"])
def test_bomb_evidence_inside_tape_is_not_released(kind):
    row = tape_row()
    polygon = [[365, 222], [393, 222], [393, 234], [365, 234]]
    if kind == "candidate":
        row["payload"]["candidates"] = [{"id": 1, "disposition": "retained", "side_contour": polygon, "bomb_ids": [1]}]
    else:
        row["payload"]["bombs"] = [{"id": 1, "status": "missed", "side_tolerance_polygon": polygon}]
    assert compare_side_record(row)["tape_suppressed"] == 0
    assert not record_conversion(row, compare_side_record(row))["eligible"]


def test_three_classes_count_points_separately_and_records_once(tmp_path):
    tape = tape_row()
    mixed = sticker_row()
    mixed["payload"].update(side_polygon=tape["payload"]["side_polygon"], exclusions=tape["payload"]["exclusions"])
    mixed["aoi_machine_coords"]["W0F00000"] += [
        {"defect_code": "CSJBL", "product_x": 25, "product_y": 185},
        *tape["aoi_machine_coords"]["W0F00000"]]
    mixed["ng_details"] = [{**p, "image": "W0F00000", "type": "aoi_passthrough"}
                           for p in mixed["aoi_machine_coords"]["W0F00000"]]
    comparison = compare_side_record(mixed)
    assert (comparison["mark_suppressed"], comparison["omit_suppressed"], comparison["tape_suppressed"]) == (1, 3, 2)
    assert side_white_view(mixed)["headline"] == "AOI 6 點可轉 OK"
    assert record_conversion(mixed, comparison)["eligible"]
    blocked = copy.deepcopy(tape)
    blocked["ng_details"].append({"type": "ai_defect", "image": "R0F00000"})
    db = CAPIDatabase(str(tmp_path / "tape.db"))
    record_ids = []
    for row in (tape, mixed, blocked):
        record_id = db.save_inference_record(glass_id="TAPE", model_id="MODEL", machine_no="CAPI1",
            resolution=(400, 240), machine_judgment="NG", ai_judgment="NG", image_dir="samples", total_images=1,
            ng_images=1, ng_details=json.dumps(row["ng_details"]), request_time="2026-10-28 12:00:00",
            response_time="2026-10-28 12:00:01", processing_seconds=1, client_response_text="original",
            aoi_machine_coords=json.dumps(row["aoi_machine_coords"]))
        db.save_side_white_result(record_id, row["payload"])
        record_ids.append(record_id)
    before = [db.get_record_detail(i) for i in record_ids]
    query = lambda **kw: db.get_side_statistics(start_date="2026-10-28", end_date="2026-10-28", **kw)
    result = query()
    s = result["conversion_summary"]
    assert s["tape"] == {"points": 6, "records": 3}
    assert s["mark"] == {"points": 1, "records": 1} and s["omit"] == {"points": 3, "records": 1}
    assert s["mixed_records"] == s["both_records"] == 1
    assert s["eligible"] == {"mark": 0, "omit": 0, "tape": 1, "mixed": 1}
    assert s["eligible_records"] == 2 and s["blocked_records"] == 1
    assert s["scope"]["scenarios"]["tape"]["converted"] == 1
    assert s["scope"]["scenarios"]["combined"]["remaining_ng"] == 1
    for value, count in (("tape", 3), ("eligible_tape", 1), ("mixed", 1), ("eligible", 2), ("blocked", 1)):
        filtered = query(conversion=value, limit=1)
        assert filtered["total"] == count and filtered["conversion_summary"] == s
    assert [db.get_record_detail(i) for i in record_ids] == before
    assert side_white_view(db.list_side_white_results()["rows"][-1])["tape"]["text"] == "已排除 · 2 點可轉 OK"


def test_duplicate_region_does_not_inflate_evidence_and_unknown_mapping_stays_unavailable():
    row = tape_row()
    row["payload"]["exclusions"] *= 2
    assert len(peel_tape_regions(row["payload"])) == 1
    row["payload"]["mapping"]["status"] = "unavailable"
    assert compare_side_record(row)["tape_suppressed"] == 0
