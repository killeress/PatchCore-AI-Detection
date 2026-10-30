"""Record-level counterfactual rejection rates never add point counts as panels."""
import copy
import json
from datetime import datetime

import pytest

from capi_database import CAPIDatabase
from capi_side_report import compare_side_record, conversion_rate_projection, record_conversion


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


def save(db, row=None, *, judgment="NG", when="2026-10-22 12:00:00", machine="CAPI1", model="MODEL", created_at=None):
    record_id = db.save_inference_record(
        glass_id="REPEATED", model_id=model, machine_no=machine, resolution=(500, 300),
        machine_judgment="NG", ai_judgment=judgment, image_dir="samples", total_images=1,
        ng_images=int(judgment.startswith("NG")), ng_details=json.dumps(row["ng_details"] if row else []),
        request_time=when, response_time=when, processing_seconds=.1, client_response_text="unchanged",
        aoi_machine_coords=json.dumps(row["aoi_machine_coords"] if row else {}))
    conn = db._get_conn()
    conn.execute("UPDATE inference_records SET created_at=? WHERE id=?", (created_at or when, record_id))
    conn.commit()
    conn.close()
    if row:
        return record_id, db.save_side_white_result(record_id, row["payload"])
    return record_id, None


def query(db, **kwargs):
    kwargs.setdefault("shift", "day")
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
    assert s["clear"] == {"points": 1, "records": 1}
    assert s["eligible"] == {"mark": 1, "omit": 1, "tape": 0, "clear": 0, "mixed": 2}
    assert s["eligible_records"] == 4 and s["blocked_records"] == 2 and s["already_ok_records"] == 1
    assert s["scope"]["denominator"] == s["scope"]["total"] == 13 and s["scope"]["other"] == 2
    assert s["scope"]["err"] == 1
    assert s["scope"]["before_pct"] == pytest.approx(800 / 13)
    combined = s["scope"]["scenarios"]["combined"]
    assert combined["converted"] == 4 and combined["remaining_ng"] == 4
    assert combined["after_pct"] == pytest.approx(400 / 13)
    assert combined["drop_pp"] == pytest.approx(400 / 13) and combined["relative_drop_pct"] == 50
    assert s["scope"]["scenarios"]["mark"]["converted"] == s["scope"]["scenarios"]["omit"]["converted"] == 1
    assert s["csjbl"]["denominator"] == 8 and s["csjbl"]["scenarios"]["combined"]["after_pct"] == 37.5
    for filters, expected in (({"conversion": "mark"}, 6), ({"conversion": "omit"}, 2),
                              ({"conversion": "both"}, 1), ({"conversion": "eligible"}, 4),
                              ({"conversion": "eligible_mark"}, 1), ({"conversion": "eligible_omit"}, 1),
                              ({"conversion": "blocked"}, 2), ({"review": "confirmed"}, 0),
                              ({"status": "NOT_RUN"}, 0)):
        filtered = query(db, **filters, offset=1, limit=1, include_rows=False)
        assert filtered["total"] == expected and filtered["rows"] == []
        assert filtered["conversion_summary"] == s
    after = db.get_record_detail(first)
    for field in ("ai_judgment", "client_response_text", "ng_details", "aoi_machine_coords"):
        assert after[field] == before[field]


def test_formal_ok_s_impact_is_separate_from_remaining_historical_ng(db):
    record_id, _ = save(db, sample(("mark", "omit")), judgment="OK-s")
    save(db, sample(), judgment="OK-s")
    save(db, sample())  # One historical NG still eligible for an extra reduction.
    for judgment in ("NG", "OK", "OK-i", "ERR:timeout", ""):
        save(db, judgment=judgment)
    before = db.get_record_detail(record_id)
    s = query(db)["conversion_summary"]
    assert s["scope"]["ok"] == 4 and s["scope"]["ok_s"] == s["converted_records"] == 2
    assert s["scope"]["formal_impact"] == {
        "converted": 2, "before_ng": 4, "after_ng": 2,
        "before_pct": 50, "after_pct": 25, "drop_pp": 25, "relative_drop_pct": 50,
    }
    assert s["scope"]["scenarios"]["combined"]["converted"] == s["eligible_records"] == 1
    assert s["scope"]["scenarios"]["combined"]["remaining_ng"] == 1
    assert s["scope"]["scenarios"]["combined"]["after_pct"] == 12.5
    subset = s["csjbl"]["formal_impact"]
    assert s["csjbl"]["denominator"] == 3 and subset["converted"] == 2
    assert subset["before_pct"] == 100 and subset["after_pct"] == pytest.approx(100 / 3)
    assert subset["drop_pp"] == pytest.approx(200 / 3)
    for filters in ({"conversion": "eligible"}, {"conversion": "mixed"}, {"status": "NOT_RUN"}):
        assert query(db, **filters, offset=1, limit=1)["conversion_summary"] == s
    assert db.get_record_detail(record_id) == before


def test_formal_ok_s_impact_does_not_require_side_payload_or_csjbl_details(db):
    record_id, _ = save(db, judgment="OK-s")
    row = sample()
    with db._get_conn() as conn:
        conn.execute("UPDATE inference_records SET aoi_machine_coords=?, ng_details=? WHERE id=?",
                     (json.dumps(row["aoi_machine_coords"]), json.dumps(row["ng_details"]), record_id))
    save(db, judgment="OK-s")  # Formal result remains authoritative without AOI evidence.
    s = query(db)["conversion_summary"]
    assert s["scope"]["formal_impact"]["converted"] == 2
    assert s["csjbl"]["formal_impact"]["converted"] == s["converted_records"] == 1
    assert s["eligible_records"] == 0
    for scope in (s["scope"], s["csjbl"]):
        assert scope["formal_impact"]["before_pct"] == 100
        assert scope["formal_impact"]["after_pct"] == 0
        assert scope["formal_impact"]["relative_drop_pct"] == 100
        assert scope["scenarios"]["combined"]["converted"] == 0


@pytest.mark.parametrize("total,ng,ok_s,expected", [
    (747, 140, 5, (19.410977, 18.741633, .669344, 3.448276)),
    (10, 4, 0, (40, 40, 0, 0)),
    (10, 0, 0, (0, 0, 0, None)),
    (0, 0, 0, (None, None, None, None)),
])
def test_formal_impact_rates_and_empty_baselines(total, ng, ok_s, expected):
    projection = conversion_rate_projection(
        {"total": total, "ng": ng, "ok_s": ok_s}, {"mark": 0, "omit": 0})
    formal = projection["formal_impact"]
    assert tuple(formal[key] for key in ("before_pct", "after_pct", "drop_pp", "relative_drop_pct")) == expected


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


def test_scope_dates_machine_model_and_err_denominator(db):
    save(db, sample(), when="2026-10-21 23:59:59", judgment="OK-s")
    save(db, sample(), when="2026-10-23 00:00:00", judgment="OK-s")
    save(db, sample(), machine="OTHER", judgment="OK-s")
    save(db, sample(), model="OTHER", judgment="OK-s")
    save(db, judgment="ERR:FAIL")
    s = query(db, machine_no="CAPI1", model_id="MODEL")["conversion_summary"]
    assert s["scope"]["total"] == s["scope"]["denominator"] == s["scope"]["err"] == 1
    assert s["scope"]["before_pct"] == 0
    assert s["scope"]["formal_impact"]["converted"] == 0
    assert s["scope"]["scenarios"]["combined"]["drop_pp"] == 0
    assert query(db, glass_id="NO_MATCH")["conversion_summary"]["scope"]["total"] == 0
    assert query(db, glass_id="NO_MATCH")["conversion_summary"]["scope"]["before_pct"] is None
    assert query(db, glass_id="NO_MATCH")["conversion_summary"]["scope"]["formal_impact"]["before_pct"] is None
    with pytest.raises(ValueError, match="轉 OK 分類"):
        query(db, conversion="unknown")
    with pytest.raises(ValueError, match="班別"):
        query(db, shift="invalid")


@pytest.mark.parametrize("clock", [
    "2026-10-22 07:29:59", "2026-10-22 07:30:00", "2026-10-22 19:29:59",
    "2026-10-22 19:30:00", "2026-10-22 23:50:00", "2026-10-23 06:00:00",
    "2026-10-23 07:30:00",
])
def test_current_shift_matches_dashboard_with_err_and_different_request_times(db, clock):
    records = [
        ("2026-10-21 19:30:00", "OK-s"), ("2026-10-22 07:29:59", "NG"),
        ("2026-10-22 07:30:00", "NG"), ("2026-10-22 08:00:00", "OK-i"),
        ("2026-10-22 12:00:00", "OK-s"), ("2026-10-22 19:29:59", "ERR:timeout"),
        ("2026-10-22 19:30:00", "NG"), ("2026-10-22 23:00:00", "OK-s"),
        ("2026-10-23 00:10:00", "ERR:NO_IMAGE"), ("2026-10-23 07:29:59", "NG"),
        ("2026-10-23 07:30:00", "OK"), ("2026-10-23 19:30:00", "NG"),
    ]
    ids = [save(db, sample(), judgment=judgment, when="2025-01-01 00:00:00", created_at=created)[0]
           for created, judgment in records]
    # A matching request_time must not pull an out-of-shift input into the report.
    save(db, sample(), when=clock, created_at="2025-01-01 00:00:00")
    now = datetime.fromisoformat(clock)
    dashboard = db.get_shift_statistics(now)
    # No shift parameter: default follows the server's current shift, even when
    # an old tab still sends yesterday's dates. request_time is outside the scope.
    result = db.get_side_statistics(start_date="2020-01-02", end_date="2020-01-01", now=now, limit=20)
    scope = result["conversion_summary"]["scope"]
    assert scope["denominator"] == scope["total"] == dashboard["total"]
    assert scope["ng"] == dashboard["ng_count"] and scope["ok"] == dashboard["ok_count"]
    assert scope["err"] == dashboard["err_count"]
    assert scope["formal_impact"]["converted"] == dashboard["ok_s_count"]
    assert scope["before_pct"] == pytest.approx(100 * dashboard["ng_count"] / dashboard["total"])
    time = result["time_scope"]
    assert time["mode"] == "current" and time["shift_name"] == dashboard["shift_name"]
    assert time["start_date"] == time["end_date"] == time["start"][:10]
    expected = {record_id for record_id, (created, _) in zip(ids, records) if time["start"] <= created < time["end"]}
    assert {row["record_id"] for row in result["rows"]} == expected
    assert all(time["start"] <= row["inference_created_at"] < time["end"] for row in result["rows"])
    assert result["summary"]["with_csjbl"]["total"] == len(expected)
    combined = scope["scenarios"]["combined"]
    assert combined["converted"] == scope["ng"] and combined["after_pct"] == 0


@pytest.mark.parametrize("shift", ["day", "night"])
def test_historical_date_range_includes_only_selected_shifts(db, shift):
    for day in ("2026-10-22", "2026-10-23", "2026-10-24"):
        for time, judgment in (("00:00:00", "ERR:timeout"), ("07:29:59", "NG"),
                               ("07:30:00", "OK-s"), ("19:29:59", "NG"), ("19:30:00", "OK-s")):
            save(db, sample(), judgment=judgment, when=f"{day} {time}")
    result = db.get_side_statistics(start_date="2026-10-22", end_date="2026-10-23", shift=shift, limit=20)
    scope = result["conversion_summary"]["scope"]
    shifts = [db.get_shift_statistics_for(day, shift) for day in ("2026-10-22", "2026-10-23")]
    assert scope["total"] == scope["denominator"] == sum(row["total"] for row in shifts)
    assert scope["ng"] == sum(row["ng_count"] for row in shifts)
    assert scope["err"] == sum(row["err_count"] for row in shifts)
    assert scope["formal_impact"]["converted"] == sum(row["ok_s_count"] for row in shifts)
    assert result["total"] == scope["total"]
    assert scope["scenarios"]["combined"]["converted"] == scope["ng"]
    assert result["time_scope"]["end_date"] == "2026-10-23"
    assert result["time_scope"]["end"] == shifts[-1]["end"]


def test_invalid_formal_judgment_counts_as_input_but_cannot_remove_ng(db):
    save(db, sample(), judgment=" NG")
    s = query(db)["conversion_summary"]
    assert s["mark"]["records"] == 1 and s["eligible_records"] == 0
    assert s["scope"]["denominator"] == 1
    assert s["scope"]["scenarios"]["combined"]["remaining_ng"] == 0


def test_saved_review_blocks_but_nearby_reference_does_not_block_conversion(db):
    row = sample()
    contour = [[130, 90], [140, 90], [140, 110], [130, 110]]
    row["payload"]["candidates"].append({"id": 2, "side_contour": contour, "disposition": "retained"})
    _, result_id = save(db, row)
    assert query(db, tolerance=50)["conversion_summary"]["eligible_records"] == 1
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
