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
