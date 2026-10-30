"""An AOI ROI can contain a MARK while its centre lies outside the MARK mask."""
import copy

import pytest

from capi_database import CAPIDatabase
from capi_side_report import compare_side_record, record_conversion, side_white_view
from test_side_conversion_statistics import sample, save, query


def roi_row(source="formal", rotated=False, candidates=True):
    row = sample()
    point = row["aoi_machine_coords"]["W0F00000"][0]
    point["product_x"] = 140
    row["ng_details"][0]["product_x"] = 140
    payload = row["payload"]
    row["status"] = payload["status"] = "CANDIDATES" if candidates else "NO_CANDIDATES"
    payload["rotation_applied"] = rotated
    payload["evidence_context"]["mark_rotation_applied"] = rotated
    contour = [[95, 95], [115, 95], [115, 115], [95, 115]]
    payload["candidates"] = [{"id": 1, "disposition": "retained", "bomb_ids": [],
                              "side_contour": contour, "remaining_contours": [contour],
                              "dust": {"status": "clear"}}] if candidates else []
    if source != "formal":
        payload["evidence_context"]["inference_marks"] = []
        payload["exclusions"] = [{"type": source, "bbox": [90, 90, 30, 30]}]
    return row


@pytest.mark.parametrize("source", ["formal", "mark", "corner_print"])
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("candidates", [False, True])
def test_roi_mark_is_excluded_even_when_aoi_centre_is_outside(source, rotated, candidates):
    row = roi_row(source, rotated, candidates)
    before = copy.deepcopy(row)
    comparison = compare_side_record(row, tolerance=50)
    point = comparison["points"][0]
    assert point["front_xy"][0] > 120  # The actual MARK mask ends at x=120.
    assert point["status"] == "mark_suppressed" and point["mark_match_basis"] == "roi"
    assert point["mark_candidate_ids"] == ([1] if candidates else [])
    assert point["retained_candidate_ids"] == [] and comparison["mark_suppressed"] == 1
    assert record_conversion(row, comparison)["eligible"]
    view = side_white_view(dict(row, comparison=comparison))
    assert view["mark"]["tone"] == "ok" and "ROI" in view["mark"]["reason"]
    assert "ROI 內的 MARK 已屏蔽" in point["reason"]
    if candidates:
        assert view["candidates"][0]["label"] == "MARK 已屏蔽"
        assert view["sticker"]["text"] == "無相關候選"
    assert row == before


@pytest.mark.parametrize("issue", ["other_candidate", "partial_mark", "bomb_candidate", "bomb_region",
                                   "truncated", "error", "wrong_source", "bad_mapping", "outside_roi"])
def test_roi_mark_never_hides_other_defects_or_incomplete_evidence(issue):
    row = roi_row()
    payload = row["payload"]
    if issue in {"other_candidate", "partial_mark"}:
        contour = ([[135, 95], [145, 95], [145, 105], [135, 105]] if issue == "other_candidate" else
                   [[110, 95], [150, 95], [150, 105], [110, 105]])
        payload["candidates"].append({"id": 2, "disposition": "retained", "bomb_ids": [],
                                      "side_contour": contour, "remaining_contours": [contour]})
    elif issue == "bomb_candidate":
        payload["candidates"].append({"id": 2, "disposition": "retained", "bomb_ids": [1],
                                      "side_contour": [[135, 95], [145, 95], [145, 105], [135, 105]]})
    elif issue == "bomb_region":
        payload["bombs"] = [{"id": 1, "side_tolerance_polygon": [[130, 90], [150, 90], [150, 110], [130, 110]]}]
    elif issue == "truncated":
        payload["truncated"] = True
    elif issue == "error":
        row["status"] = "ERROR"
    elif issue == "wrong_source":
        payload["evidence_context"]["inference_marks"][0]["source_image"] = "W0F00000_OTHER.tif"
    elif issue == "bad_mapping":
        payload["mapping"]["matrix"] = [[0, 0, 0]] * 3
    else:
        row["aoi_machine_coords"]["W0F00000"][0]["product_x"] = 240
    before = copy.deepcopy(row)
    comparison = compare_side_record(row, tolerance=50)
    assert comparison["mark_suppressed"] == 0
    assert record_conversion(row, comparison)["eligible"] == (issue != "outside_roi" and bool(comparison["no_anomaly"]))
    view = side_white_view(dict(row, comparison=comparison))
    assert view["mark"]["tone"] != "ok"
    if issue in {"other_candidate", "partial_mark"}:
        assert comparison["points"][0]["mark_candidate_ids"] == [1]
        assert comparison["points"][0]["retained_candidate_ids"] == [2]
        assert view["mark"]["tone"] == "warn" and "ROI 有 MARK 但仍需保留" in view["mark"]["text"]
    assert row == before


@pytest.mark.parametrize("tolerance", [0, 10])
def test_mark_outside_the_selected_roi_stays_gray(tolerance):
    row = roi_row()
    comparison = compare_side_record(row, tolerance=tolerance)
    assert comparison["mark_suppressed"] == 0
    view = side_white_view(dict(row, comparison=comparison))
    assert view["mark"]["tone"] == "muted"
    assert view["mark"]["text"] == "MARK 屏蔽範圍 1 區，AOI 未命中"


@pytest.mark.parametrize("issue", ["corner_tape", "configured_zone", "nan", "outside", "empty"])
def test_unrelated_or_invalid_local_masks_cannot_create_mark_evidence(issue):
    row = roi_row("mark")
    exclusion = row["payload"]["exclusions"][0]
    if issue in {"corner_tape", "configured_zone"}:
        exclusion["type"] = issue
    elif issue == "nan":
        exclusion["bbox"][0] = float("nan")
    elif issue == "outside":
        exclusion["bbox"] = [490, 90, 30, 30]
    else:
        exclusion["bbox"][2] = 0
    comparison = compare_side_record(row, tolerance=50)
    assert comparison["mark_suppressed"] == 0
    assert record_conversion(row, comparison)["eligible"] == bool(comparison["no_anomaly"])


@pytest.mark.parametrize("other_ng", [False, True])
def test_roi_mark_updates_projection_only_when_all_formal_ng_is_covered(tmp_path, other_ng):
    db = CAPIDatabase(str(tmp_path / "roi.db"))
    row = roi_row()
    row["payload"]["candidates"].extend([
        {"id": 7, "disposition": "retained", "side_contour": [[146, 95], [156, 95], [156, 105], [146, 105]]},
        {"id": 10, "disposition": "retained", "side_contour": [[125, 70], [135, 70], [135, 80], [125, 80]]},
    ])
    if other_ng:
        row["ng_details"].append({"type": "ai_defect", "image": "G0F00000"})
    record_id, _ = save(db, row)
    before = db.get_record_detail(record_id)
    summary = query(db)["conversion_summary"]
    assert summary["mark"] == {"points": 1, "records": 1}
    assert summary["eligible_records"] == int(not other_ng)
    assert summary["scope"]["scenarios"]["mark"]["after_pct"] == (100 if other_ng else 0)
    assert db.get_record_detail(record_id) == before


def test_an_omit_point_keeps_its_category_when_a_mark_is_only_nearby():
    row = sample(("omit",))
    row["payload"]["evidence_context"]["inference_marks"][0]["front_bbox"] = [260, 90, 20, 30]
    comparison = compare_side_record(row, tolerance=50)
    assert comparison["omit_suppressed"] == 1 and comparison["mark_suppressed"] == 0
    assert record_conversion(row, comparison)["category"] == "omit"


@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("with_bombs", [False, True])
def test_candidates_7_and_10_nearby_do_not_block_mark_when_neither_hits_aoi(rotated, with_bombs):
    row = roi_row(rotated=rotated)
    payload = row["payload"]
    for candidate_id, contour in [(7, [[146, 95], [156, 95], [156, 105], [146, 105]]),
                                  (10, [[125, 70], [135, 70], [135, 80], [125, 80]])]:
        payload["candidates"].append({"id": candidate_id, "disposition": "retained",
            "bomb_ids": [1] if with_bombs else [], "side_contour": contour})
    if with_bombs:
        payload["bombs"] = [{"id": 1, "side_tolerance_polygon": [[160, 90], [180, 90], [180, 110], [160, 110]]}]
    before = copy.deepcopy(row)
    comparison = compare_side_record(row, 50)
    point = comparison["points"][0]
    assert {7, 10} <= set(point["candidate_ids"])
    assert point["aoi_candidate_ids"] == point["retained_candidate_ids"] == []
    assert comparison["mark_suppressed"] == 1 and record_conversion(row, comparison)["eligible"]
    view = side_white_view(dict(row, comparison=comparison))
    assert view["yellow_candidates"] == {"nearby": 0, "total": 0}
    if not with_bombs:
        assert all(c["label"] == "僅供參考 · AOI 未命中" for c in view["candidates"] if c["id"] in {7, 10})
    assert row == before


@pytest.mark.parametrize("tolerance", [0, 10, 50, 500])
def test_reference_radius_cannot_widen_blocking_scope_for_a_mark_point(tolerance):
    row = roi_row()
    row["aoi_machine_coords"]["W0F00000"][0]["product_x"] = 100
    row["ng_details"][0]["product_x"] = 100
    row["payload"]["candidates"].append({"id": 7, "disposition": "retained",
        "side_contour": [[130, 95], [140, 95], [140, 105], [130, 105]]})
    comparison = compare_side_record(row, tolerance)
    assert comparison["points"][0]["retained_candidate_ids"] == []
    assert comparison["mark_suppressed"] == 1 and record_conversion(row, comparison)["eligible"]
