from capi_database import CAPIDatabase
from jinja2 import Environment, FileSystemLoader


def test_within_spec_review_log_persists_and_joins_latest_summary(tmp_path):
    db = CAPIDatabase(str(tmp_path / "capi.db"))
    db.save_client_accuracy_records([
        {
            "time_stamp": "2026-06-18 07:00:00",
            "pnl_id": "PANEL001",
            "mach_id": "CAPI0703",
            "result_eqp": "NG",
            "result_ai": "NG",
            "result_ric": "OK",
            "datastr": "W0F00000;OK;1;",
        }
    ])
    inference_id = db.save_inference_record(
        glass_id="PANEL001",
        model_id="MODEL",
        machine_no="CAPI0703",
        resolution=(100, 100),
        machine_judgment="NG",
        ai_judgment="NG",
        image_dir="",
        total_images=0,
        ng_images=0,
        ng_details="",
        request_time="2026-06-18 07:00:01",
        response_time="2026-06-18 07:00:02",
        processing_seconds=0.1,
    )
    client = db.get_client_accuracy_records("2026-06-18", "2026-06-18")[0]

    saved = db.save_within_spec_review_log(
        client_record_id=client["id"],
        inference_record_id=inference_id,
        suggestion={
            "suggested": True,
            "category": "within_spec",
            "reason": "W0F00000 黑點 0.2mm <= 0.3mm",
        },
        detail={
            "rule_selection": {"matched_machine_key": "MODEL", "fallback_used": False},
            "panel_summary": {"total_dot_count": 1, "target_tile_count": 1, "evaluated_tile_count": 1},
            "steps": [{"message": "測試 log"}],
            "matches": [{"screen": "W0F00000", "dot_label": "黑點"}],
        },
        processing_seconds=0.12,
    )

    assert saved["suggested"] is True
    assert saved["detail"]["steps"][0]["message"] == "測試 log"

    latest = db.get_latest_within_spec_review_log(client["id"])
    assert latest["id"] == saved["id"]

    rows = db.get_client_accuracy_records("2026-06-18", "2026-06-18")
    assert rows[0]["within_spec_log_id"] == saved["id"]
    assert rows[0]["within_spec_suggested"] == 1
    assert rows[0]["within_spec_reason"] == "W0F00000 黑點 0.2mm <= 0.3mm"

    report_rows = db.list_within_spec_review_log_report(
        start_date="2026-06-18",
        end_date="2026-06-18",
        keyword="PANEL001",
        suggested=True,
    )
    assert report_rows[0]["id"] == saved["id"]
    assert report_rows[0]["pnl_id"] == "PANEL001"
    assert report_rows[0]["model_id"] == "MODEL"
    assert report_rows[0]["matched_machine_key"] == "MODEL"
    assert report_rows[0]["total_dot_count"] == 1


def test_within_spec_detail_template_renders():
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("within_spec_detail.html")

    html = template.render(
        request_path="/ric/within-spec-log/1",
        log={
            "id": 1,
            "client_record_id": 2,
            "inference_record_id": 3,
            "suggested": True,
            "reason": "W0F00000 黑點 0.2mm <= 0.3mm",
            "error_message": "",
            "processing_seconds": 0.1,
            "created_at": "2026-06-18 08:00:00",
            "detail": {
                "rule_selection": {
                    "matched_machine_key": "MODEL_A",
                    "fallback_used": False,
                },
                "panel_summary": {
                    "total_dot_count": 1,
                    "target_tile_count": 1,
                    "evaluated_tile_count": 1,
                    "skipped_tiles": {},
                },
                "panel_totals": [{
                    "screen": "W0F00000",
                    "dot_type": "black_dot",
                    "dot_label": "黑點",
                    "total_count": 1,
                    "max_size_mm": 0.13,
                    "threshold_mm": 0.3,
                    "screen_count_limit": 3,
                    "tile_count_threshold": 2,
                    "evaluated_tiles": 1,
                }],
                "visuals": [{
                    "image_name": "W0F00000.png",
                    "tile_id": 1,
                    "dot_type": "black_dot",
                    "dot_label": "黑點",
                    "count": 1,
                    "max_size_mm": 0.13,
                    "urls": {
                        "overlay_url": "/heatmaps/ws/overlay.png",
                        "crop_url": "/heatmaps/ws/crop.png",
                        "diff_url": "/heatmaps/ws/diff.png",
                        "mask_url": "/heatmaps/ws/mask.png",
                    },
                    "candidates": [{"id": 1, "size_mm": 0.13, "size_px": 6.0, "x": 1, "y": 2, "w": 3, "h": 4, "max_diff": 10}],
                }],
                "steps": [{"message": "開始判定圖片", "screen": "W0F00000"}],
            },
        },
    )

    assert "規格內計算明細 #1" in html
    assert "MODEL_A" in html
    assert "點偵測結果圖片" in html
    assert "/heatmaps/ws/overlay.png" in html


def test_within_spec_report_template_renders():
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("within_spec_report.html")

    html = template.render(
        request_path="/ric/within-spec-logs",
        filters={
            "start_date": "2026-06-18",
            "end_date": "2026-06-18",
            "keyword": "PANEL001",
            "status": "suggested",
            "limit": 200,
        },
        summary={"total": 1, "suggested": 1, "not_suggested": 0, "fallback": 0},
        rows=[{
            "id": 8,
            "created_at": "2026-06-18 08:00:00",
            "client_time_stamp": "2026-06-18 07:00:00",
            "pnl_id": "PANEL001",
            "mach_id": "CAPI0703",
            "machine_no": "CAPI0703",
            "model_id": "MODEL",
            "matched_machine_key": "MODEL",
            "fallback_used": False,
            "suggested": True,
            "total_dot_count": 1,
            "target_tile_count": 1,
            "evaluated_tile_count": 1,
            "reason": "W0F00000 黑點 0.2mm <= 0.3mm",
            "error_message": "",
            "processing_seconds": 0.12,
            "inference_record_id": 3,
        }],
    )

    assert "規格內計算清單" in html
    assert "PANEL001" in html
    assert "/ric/within-spec-log/8" in html
