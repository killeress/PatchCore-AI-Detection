import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_database import CAPIDatabase
from capi_web import CAPIWebHandler


def _save_record(
    db: CAPIDatabase,
    glass_id: str,
    *,
    tile_is_dust: int,
    request_time: str = "2026-05-26T10:00:00",
    ai_judgment: str = "OK",
) -> int:
    image_results_data = [{
        "image_path": f"/fake/{glass_id}.jpg",
        "image_name": f"{glass_id}.jpg",
        "image_width": 1024,
        "image_height": 768,
        "otsu_bounds": "",
        "tile_count": 1,
        "excluded_tiles": 0,
        "anomaly_count": 1,
        "max_score": 0.95,
        "is_ng": 0,
        "is_dust_only": 0,
        "is_bomb": 0,
        "inference_time_ms": 100.0,
        "heatmap_path": "",
        "tiles": [{
            "tile_id": 1,
            "x": 10,
            "y": 20,
            "width": 512,
            "height": 512,
            "score": 0.95,
            "is_anomaly": 1,
            "is_dust": tile_is_dust,
            "dust_iou": 0.8 if tile_is_dust else 0.0,
            "is_bomb": 0,
            "bomb_code": "",
            "peak_x": 100,
            "peak_y": 120,
            "heatmap_path": "",
            "is_exclude_zone": 0,
            "is_aoi_coord": 0,
            "aoi_defect_code": "",
            "aoi_product_x": -1,
            "aoi_product_y": -1,
        }],
    }]
    return db.save_inference_record(
        glass_id=glass_id,
        model_id="M1",
        machine_no="1",
        resolution=(1920, 1080),
        machine_judgment="NG",
        ai_judgment=ai_judgment,
        image_dir="/fake",
        total_images=1,
        ng_images=0,
        ng_details="[]",
        request_time=request_time,
        response_time="2026-05-26T10:00:01",
        processing_seconds=1.0,
        image_results_data=image_results_data,
    )


def test_dust_affected_records_include_tile_dust_when_image_is_not_dust_only(tmp_path):
    db = CAPIDatabase(str(tmp_path / "ric_omit.db"))
    dust_record_id = _save_record(db, "DUST_TILE", tile_is_dust=1)
    clean_record_id = _save_record(db, "CLEAN_TILE", tile_is_dust=0)

    assert db.get_dust_affected_record_ids([dust_record_id, clean_record_id]) == {dust_record_id}


def test_client_accuracy_date_filter_is_end_date_inclusive(tmp_path):
    db = CAPIDatabase(str(tmp_path / "ric_date_filter.db"))
    db.save_client_accuracy_records([
        {
            "time_stamp": "2026-05-20T07:29:59",
            "pnl_id": "P0",
            "mach_id": "M1",
            "result_eqp": "OK",
            "result_ai": "OK",
            "result_ric": "OK",
            "datastr": "DEFECT,OK;1;",
        },
        {
            "time_stamp": "2026-05-20T07:30:00",
            "pnl_id": "P1",
            "mach_id": "M1",
            "result_eqp": "OK",
            "result_ai": "OK",
            "result_ric": "OK",
            "datastr": "DEFECT,OK;1;",
        },
        {
            "time_stamp": "2026-05-21 00:00:00",
            "pnl_id": "P2",
            "mach_id": "M1",
            "result_eqp": "OK",
            "result_ai": "OK",
            "result_ric": "OK",
            "datastr": "DEFECT,OK;1;",
        },
        {
            "time_stamp": "2026-05-22T07:29:59",
            "pnl_id": "P3",
            "mach_id": "M1",
            "result_eqp": "OK",
            "result_ai": "OK",
            "result_ric": "OK",
            "datastr": "DEFECT,OK;1;",
        },
        {
            "time_stamp": "2026-05-22T07:30:00",
            "pnl_id": "P4",
            "mach_id": "M1",
            "result_eqp": "OK",
            "result_ai": "OK",
            "result_ric": "OK",
            "datastr": "DEFECT,OK;1;",
        },
    ])

    rows = db.get_client_accuracy_records("2026-05-20", "2026-05-21")

    assert {r["pnl_id"] for r in rows} == {"P1", "P2", "P3"}


def test_inference_stats_date_filter_is_end_date_inclusive(tmp_path):
    db = CAPIDatabase(str(tmp_path / "ric_inference_stats_filter.db"))
    _save_record(db, "INF0", tile_is_dust=0, request_time="2026-05-20T07:29:59")
    _save_record(db, "INF1", tile_is_dust=0, request_time="2026-05-20T07:30:00")
    _save_record(db, "INF2", tile_is_dust=0, request_time="2026-05-21 00:00:00")
    _save_record(db, "INF3", tile_is_dust=0, request_time="2026-05-22T07:29:59")
    _save_record(db, "INF4", tile_is_dust=0, request_time="2026-05-22T07:30:00")

    stats = db.get_inference_stats("2026-05-20", "2026-05-21")

    assert stats["summary"]["total"] == 3
    assert [row["date"] for row in stats["daily_trend"]] == ["2026-05-20", "2026-05-21"]
    assert [row["total"] for row in stats["daily_trend"]] == [2, 1]


def test_shift_window_uses_0730_1930_boundaries():
    cases = [
        ("2026-07-06 07:29:59", "夜班", "2026-07-05 19:30:00", "2026-07-06 07:30:00"),
        ("2026-07-06 07:30:00", "白班", "2026-07-06 07:30:00", "2026-07-06 19:30:00"),
        ("2026-07-06 19:29:59", "白班", "2026-07-06 07:30:00", "2026-07-06 19:30:00"),
        ("2026-07-06 19:30:00", "夜班", "2026-07-06 19:30:00", "2026-07-07 07:30:00"),
    ]

    for now_text, expected_name, expected_start, expected_end in cases:
        name, start, end = CAPIDatabase._get_shift_window(datetime.fromisoformat(now_text))
        assert name == expected_name
        assert start.strftime("%Y-%m-%d %H:%M:%S") == expected_start
        assert end.strftime("%Y-%m-%d %H:%M:%S") == expected_end


def test_shift_statistics_counts_current_shift_half_open_window(tmp_path):
    db = CAPIDatabase(str(tmp_path / "shift_stats.db"))
    ids = [
        _save_record(db, "SHIFT_BEFORE", tile_is_dust=0, request_time="2026-07-06T07:29:59"),
        _save_record(db, "SHIFT_START", tile_is_dust=0, request_time="2026-07-06T07:30:00", ai_judgment="OK"),
        _save_record(db, "SHIFT_INSIDE", tile_is_dust=0, request_time="2026-07-06T19:29:59", ai_judgment="NG"),
        _save_record(db, "SHIFT_END", tile_is_dust=0, request_time="2026-07-06T19:30:00"),
    ]
    conn = db._get_conn()
    try:
        for record_id, created_at in zip(ids, [
            "2026-07-06 07:29:59",
            "2026-07-06 07:30:00",
            "2026-07-06 19:29:59",
            "2026-07-06 19:30:00",
        ]):
            conn.execute("UPDATE inference_records SET created_at = ? WHERE id = ?", (created_at, record_id))
        conn.commit()
    finally:
        conn.close()

    stats = db.get_shift_statistics(datetime.fromisoformat("2026-07-06 12:00:00"))

    assert stats["shift_name"] == "白班"
    assert stats["time_range"] == "07/06 07:30 ~ 07/06 19:30"
    assert stats["total"] == 2
    assert stats["ok_count"] == 1
    assert stats["ng_count"] == 1


def test_inference_stats_daily_trend_includes_review_adjusted_ai_miss_rate(tmp_path):
    db = CAPIDatabase(str(tmp_path / "ric_inference_stats_ai_miss.db"))
    _save_record(db, "INF_MISS", tile_is_dust=0, request_time="2026-05-20T10:00:00")
    _save_record(db, "INF_OUTSIDE", tile_is_dust=0, request_time="2026-05-21T10:00:00")

    db.save_client_accuracy_records([
        {
            "time_stamp": "2026-05-20T08:00:00",
            "pnl_id": "MISS_THRESHOLD",
            "mach_id": "M1",
            "result_eqp": "NG",
            "result_ai": "OK",
            "result_ric": "NG",
            "datastr": "DEFECT,NG;1;",
        },
        {
            "time_stamp": "2026-05-20T08:01:00",
            "pnl_id": "MISS_DUST",
            "mach_id": "M1",
            "result_eqp": "NG",
            "result_ai": "OK-i",
            "result_ric": "NG",
            "datastr": "DEFECT,NG;1;",
        },
        {
            "time_stamp": "2026-05-20T08:02:00",
            "pnl_id": "MISS_WITHIN_SPEC",
            "mach_id": "M1",
            "result_eqp": "NG",
            "result_ai": "OK",
            "result_ric": "NG",
            "datastr": "DEFECT,NG;1;",
        },
        {
            "time_stamp": "2026-05-20T08:03:00",
            "pnl_id": "MISS_UNREVIEWED",
            "mach_id": "M1",
            "result_eqp": "NG",
            "result_ai": "OK",
            "result_ric": "NG",
            "datastr": "DEFECT,NG;1;",
        },
        {
            "time_stamp": "2026-05-20T08:04:00",
            "pnl_id": "MISS_EQP_OK",
            "mach_id": "M1",
            "result_eqp": "OK",
            "result_ai": "OK",
            "result_ric": "NG",
            "datastr": "DEFECT,NG;1;",
        },
        {
            "time_stamp": "2026-05-21T08:00:00",
            "pnl_id": "MISS_OUTSIDE_DATE",
            "mach_id": "M1",
            "result_eqp": "NG",
            "result_ai": "OK",
            "result_ric": "NG",
            "datastr": "DEFECT,NG;1;",
        },
    ])
    records = {r["pnl_id"]: r["id"] for r in db.get_client_accuracy_records("2026-05-20", "2026-05-21")}
    db.save_miss_review(records["MISS_THRESHOLD"], "threshold_high")
    db.save_miss_review(records["MISS_DUST"], "dust_misfilter")
    db.save_miss_review(records["MISS_WITHIN_SPEC"], "ai_miss_within_spec")
    db.save_miss_review(records["MISS_EQP_OK"], "threshold_high")
    db.save_miss_review(records["MISS_OUTSIDE_DATE"], "threshold_high")

    stats = db.get_inference_stats("2026-05-20", "2026-05-20")

    assert len(stats["daily_trend"]) == 1
    day = stats["daily_trend"][0]
    assert day["date"] == "2026-05-20"
    assert day["ai_miss"] == 2
    assert day["ai_miss_total"] == 4
    assert day["ai_miss_rate"] == 50.0


def test_client_accuracy_records_link_inference_by_same_day_range(tmp_path):
    db = CAPIDatabase(str(tmp_path / "ric_inference_link.db"))
    inference_record_id = _save_record(db, "LINK_DATE", tile_is_dust=0, ai_judgment="OK-i")
    db.save_client_accuracy_records([{
        "time_stamp": "2026-05-26 18:30:00",
        "pnl_id": "LINK_DATE",
        "mach_id": "M1",
        "result_eqp": "OK",
        "result_ai": "OK",
        "result_ric": "OK",
        "datastr": "DEFECT,OK;1;",
    }])

    rows = db.get_client_accuracy_records("2026-05-26", "2026-05-26")

    assert rows[0]["inference_record_id"] == inference_record_id
    assert rows[0]["inference_ai_judgment"] == "OK-i"


def _client_record(
    record_id: int,
    *,
    eqp: str,
    ai: str,
    datastr: str,
    review_category=None,
    inference_ai_judgment=None,
) -> dict:
    return {
        "id": record_id,
        "time_stamp": "2026-05-26T10:00:00",
        "pnl_id": f"P{record_id}",
        "mach_id": "M1",
        "result_eqp": eqp,
        "result_ai": ai,
        "result_ric": "NG" if "NG" in datastr else "OK",
        "datastr": datastr,
        "review_id": record_id if review_category else None,
        "review_category": review_category,
        "review_note": "",
        "review_updated_at": "2026-05-26T10:05:00" if review_category else None,
        "inference_ai_judgment": inference_ai_judgment,
    }


def test_client_summary_marks_within_spec_share_of_revival_cases():
    records = [
        _client_record(1, eqp="NG", ai="OK", datastr="DEFECT,OK;1;", inference_ai_judgment="OK-i"),
        _client_record(2, eqp="NG", ai="OK-i", datastr="DEFECT,OK;1;"),
        _client_record(3, eqp="NG", ai="OK", datastr="DEFECT,OK;1;"),
    ]

    summary, out_records = CAPIWebHandler._compute_client_summary(None, records)

    assert summary["revival"] == 3
    assert summary["revivalRate"] == 100.0
    assert summary["revivalWithinSpec"] == 2
    assert summary["revivalWithinSpecRate"] == 66.67
    assert out_records[0]["within_spec_converted_to_ok"] is True
    assert out_records[1]["result_ai"] == "OK-i"
    assert out_records[1]["within_spec_converted_to_ok"] is True


def test_client_summary_applies_manual_actual_ok_reviews_to_analysis_stats():
    records = [
        _client_record(1, eqp="OK", ai="OK", datastr="DEFECT,NG;1;", review_category="ric_misjudge"),
        _client_record(2, eqp="NG", ai="OK", datastr="DEFECT,NG;1;", review_category="data_error_actually_ok"),
        _client_record(5, eqp="OK", ai="OK", datastr="DEFECT,NG;1;", review_category="within_spec_misjudge"),
        _client_record(3, eqp="OK", ai="OK", datastr="DEFECT,NG;1;"),
        _client_record(4, eqp="NG", ai="NG", datastr="DEFECT,OK;1;"),
    ]

    summary, out_records = CAPIWebHandler._compute_client_summary(None, records)

    assert summary["ricNG"] == 1
    assert summary["aoiOK"] == 3
    assert summary["aiCorrect"] == 3
    assert summary["aiMiss"] == 0
    assert summary["aoiOver"] == 2
    assert summary["aiOver"] == 1
    assert summary["revival"] == 1
    assert summary["revivalRate"] == 50.0
    assert summary["missReviewStats"]["total"] == 4
    assert summary["missReviewStats"]["reviewed"] == 3
    assert summary["daily"]["2026-05-26"]["ricMisjudge"] == 3
    assert summary["daily"]["2026-05-26"]["withinSpecMisjudge"] == 1
    assert summary["manualTruthAdjustments"] == {
        "total": 3,
        "byCategory": {
            "ric_misjudge": 1,
            "within_spec_misjudge": 1,
            "data_error_actually_ok": 1,
        },
    }
    assert out_records[0]["actual_judgment"] == "OK"
    assert out_records[0]["truth_adjusted_by_review"] is True
    assert out_records[2]["actual_judgment"] == "OK"
    assert out_records[2]["truth_adjusted_by_review"] is True


def test_client_summary_counts_only_selected_review_categories_as_ai_miss():
    records = [
        _client_record(1, eqp="OK", ai="OK", datastr="DEFECT,NG;1;", review_category="threshold_high"),
        _client_record(2, eqp="OK", ai="OK", datastr="DEFECT,NG;1;", review_category="dust_misfilter"),
        _client_record(3, eqp="OK", ai="OK", datastr="DEFECT,NG;1;", review_category="ai_miss_within_spec"),
        _client_record(4, eqp="OK", ai="OK", datastr="DEFECT,NG;1;", review_category="outside_aoi_area"),
        _client_record(5, eqp="OK", ai="OK", datastr="DEFECT,NG;1;", review_category="other"),
        _client_record(6, eqp="OK", ai="OK", datastr="DEFECT,NG;1;", review_category="ric_misjudge"),
        _client_record(7, eqp="OK", ai="OK", datastr="DEFECT,NG;1;", review_category="data_error_actually_ok"),
        _client_record(8, eqp="OK", ai="OK", datastr="DEFECT,NG;1;"),
    ]

    summary, _ = CAPIWebHandler._compute_client_summary(None, records)

    assert summary["aiMiss"] == 2
    assert summary["aiMissRate"] == 25.0
    assert summary["daily"]["2026-05-26"]["aiMiss"] == 2
    assert summary["missReviewStats"]["total"] == 8
    assert summary["missReviewStats"]["reviewed"] == 7
    assert summary["missReviewStats"]["byCategory"]["ai_miss_within_spec"] == 1
