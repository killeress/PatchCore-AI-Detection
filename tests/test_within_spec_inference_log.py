from capi_database import CAPIDatabase


def test_within_spec_inference_log_allows_missing_client_record(tmp_path):
    db = CAPIDatabase(str(tmp_path / "test.db"))
    inference_id = db.save_inference_record(
        glass_id="PANEL-OKI",
        model_id="MODEL-A",
        machine_no="CAPI07",
        resolution=(100, 100),
        machine_judgment="NG",
        ai_judgment="OK-i",
        image_dir="",
        total_images=1,
        ng_images=1,
        ng_details="[]",
        request_time="2026-06-18 15:00:00",
        response_time="2026-06-18 15:00:01",
        processing_seconds=1.0,
    )

    saved = db.save_within_spec_review_log(
        client_record_id=None,
        inference_record_id=inference_id,
        suggestion={
            "suggested": True,
            "category": "within_spec",
            "reason": "W0F00000 黑點 0.2mm <= 0.3mm",
        },
        detail={"source": "inference", "matches": []},
        processing_seconds=0.2,
        source="inference",
    )

    assert saved["client_record_id"] is None
    assert saved["inference_record_id"] == inference_id
    assert saved["source"] == "inference"
    assert saved["suggested"] is True


def test_within_spec_report_uses_inference_glass_id_without_client_record(tmp_path):
    db = CAPIDatabase(str(tmp_path / "test.db"))
    inference_id = db.save_inference_record(
        glass_id="PANEL-AUTO-001",
        model_id="MODEL-A",
        machine_no="CAPI07",
        resolution=(100, 100),
        machine_judgment="NG",
        ai_judgment="NG",
        image_dir="",
        total_images=1,
        ng_images=1,
        ng_details="[]",
        request_time="2026-06-18 15:00:00",
        response_time="2026-06-18 15:00:01",
        processing_seconds=1.0,
    )
    saved = db.save_within_spec_review_log(
        client_record_id=None,
        inference_record_id=inference_id,
        suggestion=None,
        detail={
            "rule_selection": {"matched_machine_key": "MODEL-A", "fallback_used": False},
            "panel_summary": {"total_dot_count": 5, "target_tile_count": 5, "evaluated_tile_count": 5},
            "matches": [],
        },
        processing_seconds=0.2,
        source="inference",
    )

    rows = db.list_within_spec_review_log_report(keyword="PANEL-AUTO-001")

    assert rows[0]["id"] == saved["id"]
    assert rows[0]["client_record_id"] is None
    assert rows[0]["pnl_id"] == "PANEL-AUTO-001"
    assert rows[0]["model_id"] == "MODEL-A"
