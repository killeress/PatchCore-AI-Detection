"""Record-level counterfactual rejection rates never add point counts as panels."""
import copy
import json

import pytest

from capi_database import CAPIDatabase
from capi_side_report import compare_side_record, record_conversion


@pytest.fixture
def db(tmp_path):
    return CAPIDatabase(str(tmp_path / "conversion.db"))


def sample(kinds=("mark",), *, mode="suppress"):
    points = [{"defect_code": "CSJBL", "product_x": {"mark": 100, "omit": 300, "unknown": 450}[kind] + i,
               "product_y": 100} for i, kind in enumerate(kinds)]
    contour = [[290, 90], [310, 90], [310, 110], [290, 110]]
    candidates = [{"id": 1, "side_contour": contour, "remaining_contours": [] if mode == "suppress" else [contour],
                   "disposition": "dust_suppressed" if mode == "suppress" else "retained", "bomb_ids": [],
                   "dust": {"status": "suppressed" if mode == "suppress" else "suspected", "overlap_ratio": 1,
                            "suppressed_regions": [{"contour": contour, "holes": []}]}}] if "omit" in kinds else []
    payload = {"status": "FILTERED", "side_size": [500, 300], "front_size": [500, 300],
               "front_image": "W0F00000_120000.tif", "candidates": candidates,
               "parameters": {"dust_mode": mode}, "rotation_applied": False,
               "mapping": {"status": "estimated", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                           "front_polygon": [[0, 0], [500, 0], [500, 300], [0, 300]]},
               "evidence_context": {"mark_rotation_applied": False,
                   "inference_marks": [{"source_image": "W0F00000_120000.tif", "name": "mark_binary",
                                        "front_bbox": [90, 90, 30, 30]}]},
               "omit": {"status": "available", "coordinate_space": "side_detection_pixels", "size": [500, 300]}}
    return {"status": "FILTERED", "ai_judgment": "NG", "resolution_x": 500, "resolution_y": 300,
            "payload": payload, "aoi_machine_coords": {"W0F00000": points},
            "ng_details": [{**point, "image": "W0F00000", "type": "aoi_passthrough"} for point in points]}


def save(db, row=None, *, judgment="NG", when="2026-10-22 12:00:00", machine="CAPI1", model="MODEL"):
    record_id = db.save_inference_record(
        glass_id="REPEATED", model_id=model, machine_no=machine, resolution=(500, 300),
        machine_judgment="NG", ai_judgment=judgment, image_dir="samples", total_images=1,
        ng_images=int(judgment.startswith("NG")), ng_details=json.dumps(row["ng_details"] if row else []),
        request_time=when, response_time=when, processing_seconds=.1, client_response_text="unchanged",
        aoi_machine_coords=json.dumps(row["aoi_machine_coords"] if row else {}))
    if row:
        return record_id, db.save_side_white_result(record_id, row["payload"])
    return record_id, None


def query(db, **kwargs):
    return db.get_side_statistics(start_date="2026-10-22", end_date="2026-10-22", **kwargs)


def test_rates_cover_whole_scope_and_mixed_records_are_counted_once(db):
    first, _ = save(db, sample(("mark", "mark")))
    save(db, sample(("omit",)))
    save(db, sample(("mark", "omit")))
    save(db, sample(("mark", "unknown")))
    other = sample()
    other["ng_details"].append({"image": "R0F00000", "tile_id": 2, "peak_x": 50, "peak_y": 50})
    save(db, other)
    save(db, sample(), judgment="OK")
    unknown = sample()
    unknown["ng_details"] = []
    save(db, unknown)
    save(db, sample(("omit",), mode="observe"))
    for judgment in ("NG", "OK", "OK-i", "ERR:NO_IMAGE", ""):
        save(db, judgment=judgment)
    before = db.get_record_detail(first)
    result = query(db, limit=1)
    s = result["conversion_summary"]
    assert result["total"] == 8 and len(result["rows"]) == 1
    assert s["mark"] == {"points": 7, "records": 6}
    assert s["omit"] == {"points": 2, "records": 2}
    assert s["both_records"] == 1 and s["any_records"] == 7
    assert s["eligible"] == {"mark": 1, "omit": 1, "mixed": 1}
    assert s["eligible_records"] == 3 and s["blocked_records"] == 3 and s["already_ok_records"] == 1
    assert s["scope"]["denominator"] == 11 and s["scope"]["total"] == 13 and s["scope"]["other"] == 2
    assert s["scope"]["before_pct"] == pytest.approx(800 / 11)
    combined = s["scope"]["scenarios"]["combined"]
    assert combined["converted"] == 3 and combined["remaining_ng"] == 5
    assert combined["after_pct"] == pytest.approx(500 / 11)
    assert combined["drop_pp"] == pytest.approx(300 / 11) and combined["relative_drop_pct"] == 37.5
    assert s["scope"]["scenarios"]["mark"]["converted"] == s["scope"]["scenarios"]["omit"]["converted"] == 1
    assert s["csjbl"]["denominator"] == 8 and s["csjbl"]["scenarios"]["combined"]["after_pct"] == 50
    for filters, expected in (({"conversion": "mark"}, 6), ({"conversion": "omit"}, 2),
                              ({"conversion": "both"}, 1), ({"conversion": "eligible"}, 3),
                              ({"conversion": "eligible_mark"}, 1), ({"conversion": "eligible_omit"}, 1),
                              ({"conversion": "blocked"}, 3), ({"review": "confirmed"}, 0),
                              ({"status": "NOT_RUN"}, 0)):
        filtered = query(db, **filters, offset=1, limit=1, include_rows=False)
        assert filtered["total"] == expected and filtered["rows"] == []
        assert filtered["conversion_summary"] == s
    after = db.get_record_detail(first)
    for field in ("ai_judgment", "client_response_text", "ng_details", "aoi_machine_coords"):
        assert after[field] == before[field]


@pytest.mark.parametrize("issue", ["other_ng", "wrong_point", "wrong_image", "malformed", "missing",
                                   "incomplete", "bad_mapping", "confirmed", "missed", "mapping_error", "uncertain"])
def test_uncertain_or_other_ng_is_not_counted_as_whole_record_conversion(issue):
    row = sample()
    if issue == "other_ng":
        row["ng_details"].append({"type": "aoi_passthrough", "defect_code": "OTHER"})
    elif issue == "wrong_point":
        row["ng_details"][0]["product_x"] = 200
    elif issue == "wrong_image":
        row["ng_details"][0]["image"] = "R0F00000"
    elif issue == "malformed":
        row["ng_details"] = "broken"
    elif issue == "missing":
        row["ng_details"] = []
    elif issue == "incomplete":
        row["payload"]["truncated"] = True
    elif issue == "bad_mapping":
        row["payload"]["mapping"]["status"] = "unavailable"
    else:
        row["review_decision"] = issue
    before = copy.deepcopy(row)
    assert not record_conversion(row, compare_side_record(row))["eligible"]
    assert row == before


def test_scope_dates_machine_model_and_zero_denominator(db):
    save(db, sample(), when="2026-10-21 23:59:59")
    save(db, sample(), when="2026-10-23 00:00:00")
    save(db, sample(), machine="OTHER")
    save(db, sample(), model="OTHER")
    save(db, judgment="ERR:FAIL")
    s = query(db, machine_no="CAPI1", model_id="MODEL")["conversion_summary"]
    assert s["scope"]["total"] == 1 and s["scope"]["denominator"] == 0
    assert s["scope"]["before_pct"] is None
    assert s["scope"]["scenarios"]["combined"]["drop_pp"] is None
    assert query(db, glass_id="NO_MATCH")["conversion_summary"]["scope"]["total"] == 0
    with pytest.raises(ValueError, match="轉 OK 分類"):
        query(db, conversion="unknown")


def test_invalid_formal_judgment_cannot_remove_ng_from_an_empty_denominator(db):
    save(db, sample(), judgment=" NG")
    s = query(db)["conversion_summary"]
    assert s["mark"]["records"] == 1 and s["eligible_records"] == 0
    assert s["scope"]["denominator"] == 0
    assert s["scope"]["scenarios"]["combined"]["remaining_ng"] == 0


def test_saved_review_and_report_tolerance_change_eligibility_consistently(db):
    row = sample()
    contour = [[130, 90], [140, 90], [140, 110], [130, 110]]
    row["payload"]["candidates"].append({"id": 2, "side_contour": contour, "disposition": "retained"})
    _, result_id = save(db, row)
    assert query(db, tolerance=50)["conversion_summary"]["eligible_records"] == 0
    result = query(db, tolerance=0)
    assert result["conversion_summary"]["eligible_records"] == 1
    assert result["rows"][0]["conversion"]["eligible"]
    db.save_side_statistics_review(result_id, "mapping_error", "wrong position", "QA", "")
    assert query(db, tolerance=0, include_rows=False)["conversion_summary"]["eligible_records"] == 0


def test_streamed_summary_includes_records_after_first_batch_and_legacy_marks(db):
    row = sample()
    row["payload"].pop("evidence_context")
    record_id, _ = save(db, row)
    with db._get_conn() as conn:
        conn.execute("INSERT INTO image_results(record_id,image_path,image_name,mark_bbox,mark_source_image) "
                     "VALUES (?, '', 'W0F00000_120000.tif', '90,90,30,30', 'W0F00000_120000.tif')", (record_id,))
    for _ in range(65):
        save(db, sample())
    result = query(db, offset=65, limit=1)
    assert result["total"] == 66 and result["rows"][0]["record_id"] == record_id
    assert result["conversion_summary"]["eligible_records"] == 66
    assert result["conversion_summary"]["scope"]["scenarios"]["combined"]["after_pct"] == 0
