import sys
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
        ai_judgment="OK",
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
            "time_stamp": "2026-05-20T23:59:59",
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
            "time_stamp": "2026-05-22T00:00:00",
            "pnl_id": "P3",
            "mach_id": "M1",
            "result_eqp": "OK",
            "result_ai": "OK",
            "result_ric": "OK",
            "datastr": "DEFECT,OK;1;",
        },
    ])

    rows = db.get_client_accuracy_records("2026-05-20", "2026-05-21")

    assert {r["pnl_id"] for r in rows} == {"P1", "P2"}


def test_inference_stats_date_filter_is_end_date_inclusive(tmp_path):
    db = CAPIDatabase(str(tmp_path / "ric_inference_stats_filter.db"))
    _save_record(db, "INF1", tile_is_dust=0, request_time="2026-05-20T23:59:59")
    _save_record(db, "INF2", tile_is_dust=0, request_time="2026-05-21 00:00:00")
    _save_record(db, "INF3", tile_is_dust=0, request_time="2026-05-22T00:00:00")

    stats = db.get_inference_stats("2026-05-20", "2026-05-21")

    assert stats["summary"]["total"] == 2


def test_client_accuracy_records_link_inference_by_same_day_range(tmp_path):
    db = CAPIDatabase(str(tmp_path / "ric_inference_link.db"))
    inference_record_id = _save_record(db, "LINK_DATE", tile_is_dust=0)
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


def _client_record(
    record_id: int,
    *,
    eqp: str,
    ai: str,
    datastr: str,
    review_category=None,
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
    }


def test_client_summary_applies_manual_actual_ok_reviews_to_analysis_stats():
    records = [
        _client_record(1, eqp="OK", ai="OK", datastr="DEFECT,NG;1;", review_category="ric_misjudge"),
        _client_record(2, eqp="NG", ai="OK", datastr="DEFECT,NG;1;", review_category="data_error_actually_ok"),
        _client_record(3, eqp="OK", ai="OK", datastr="DEFECT,NG;1;"),
        _client_record(4, eqp="NG", ai="NG", datastr="DEFECT,OK;1;"),
    ]

    summary, out_records = CAPIWebHandler._compute_client_summary(None, records)

    assert summary["ricNG"] == 1
    assert summary["aiCorrect"] == 2
    assert summary["aiMiss"] == 1
    assert summary["aoiOver"] == 2
    assert summary["aiOver"] == 1
    assert summary["revival"] == 1
    assert summary["missReviewStats"]["total"] == 3
    assert summary["missReviewStats"]["reviewed"] == 2
    assert summary["daily"]["2026-05-26"]["ricMisjudge"] == 2
    assert summary["manualTruthAdjustments"] == {
        "total": 2,
        "byCategory": {
            "ric_misjudge": 1,
            "data_error_actually_ok": 1,
        },
    }
    assert out_records[0]["actual_judgment"] == "OK"
    assert out_records[0]["truth_adjusted_by_review"] is True
