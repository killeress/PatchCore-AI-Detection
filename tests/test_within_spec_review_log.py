from capi_database import CAPIDatabase
from jinja2 import Environment, FileSystemLoader


def test_within_spec_review_log_persists_and_joins_latest_summary(tmp_path):
    db = CAPIDatabase(str(tmp_path / "capi.db"))
    db.save_client_accuracy_records([
        {
            "time_stamp": "2026-06-18 08:00:00",
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
        request_time="2026-06-18 08:00:01",
        response_time="2026-06-18 08:00:02",
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

    log_date = saved["created_at"][:10]
    report_rows = db.list_within_spec_review_log_report(
        start_date=log_date,
        end_date=log_date,
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
    env.globals.update(app_version={"version": "test"}, host_identity="")
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
                    "dust_mask_filtered_count": 1,
                    "urls": {
                        "overlay_url": "/heatmaps/ws/overlay.png",
                        "crop_url": "/heatmaps/ws/crop.png",
                        "diff_url": "/heatmaps/ws/diff.png",
                        "mask_url": "/heatmaps/ws/mask.png",
                        "dust_mask_url": "/heatmaps/ws/dust_mask.png",
                        "dust_overlay_url": "/heatmaps/ws/dust_overlay.png",
                    },
                    "candidates": [{"id": 1, "size_mm": 0.13, "size_px": 6.0, "x": 1, "y": 2, "w": 3, "h": 4, "max_diff": 10}],
                    "thresholds": {
                        "hysteresis_selected_group": 1,
                        "hysteresis_group1_count": 6,
                        "hysteresis_group2_count": 4,
                        "hysteresis_group2_attempted": True,
                        "hysteresis_switch_reason": "group1_count_above_switch",
                        "hysteresis_group2_reject_reason": "group2_count_above_max2",
                    },
                }],
                "non_dot_residues": [{
                    "screen": "W0F00000",
                    "image": "W0F00000.png",
                    "tile_id": 1,
                    "reason": "aspect_ratio_below_min",
                    "x": 0,
                    "y": 0,
                    "w": 128,
                    "h": 5,
                    "area_px": 640,
                    "aspect_ratio": 0.039,
                    "long_side_px": 128,
                    "long_side_ratio": 1.0,
                    "max_diff": 20,
                }],
                "steps": [{"message": "開始判定圖片", "screen": "W0F00000"}],
                "parameter_snapshot": {"matched_machine_key": "MODEL_A"},
                "inference_auto_decision": {"reason": "非點狀殘留"},
            },
        },
    )

    assert "規格內計算明細 #1" in html
    assert "/record/3" in html
    assert "MODEL_A" in html
    assert "灰塵遮罩" in html
    assert "灰塵疊圖" in html
    assert "點偵測結果圖片" in html
    assert "非點狀殘留" in html
    assert "aspect_ratio_below_min" in html
    assert "/heatmaps/ws/overlay.png" in html
    assert "複製排查 Log" in html
    assert "parameter_snapshot" in html
    assert "non_dot_residues" in html
    assert "Hysteresis group 1" in html
    assert "group2_count_above_max2" in html
    assert "重新產生偵測圖片" in html


def test_within_spec_detail_explains_dust_only_ok_i():
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("within_spec_detail.html")

    html = template.render(
        request_path="/ric/within-spec-log/18707",
        app_version={"version": ""},
        host_identity="",
        log={
            "id": 18707,
            "client_record_id": None,
            "inference_record_id": 339088,
            "suggested": True,
            "reason": "",
            "error_message": "",
            "processing_seconds": 0.24,
            "created_at": "2026-07-13 15:32:24",
            "detail": {
                "panel_summary": {
                    "total_dot_count": 0,
                    "target_tile_count": 3,
                    "evaluated_tile_count": 3,
                    "candidate_summary": {
                        "raw_candidate_count": 3,
                        "final_candidate_count": 0,
                        "dust_mask_filtered_count": 3,
                        "no_detect_mask_filtered_count": 0,
                    },
                },
                "panel_totals": [{
                    "screen": "W0F00000",
                    "dot_type": "black_dot",
                    "dot_label": "黑點",
                    "total_count": 0,
                    "max_size_mm": 0,
                    "threshold_mm": 0.35,
                    "screen_count_limit": 2,
                    "max_tile_count": 0,
                    "tile_count_threshold": 1,
                    "evaluated_tiles": 1,
                    "within": True,
                }],
                "inference_auto_decision": {
                    "converted_to_ok_i": True,
                    "status": "within_spec",
                },
            },
        },
    )

    text = " ".join(html.split())
    assert (
        "OpenCV 初步抓到 3 個點候選，3 個皆由灰塵遮罩排除，因此以 0 點納入 "
        "Panel 規格比較；整片 Panel 均符合設定門檻，判定 OK-I。"
    ) in text
    assert "仍判定為點偵測未命中" not in text


def test_saved_single_polarity_visual_shows_both_dot_types_from_original_trace():
    from capi_web import _within_spec_log_view

    source = {"image": "panelW0F00000.tif", "tile_id": 0, "crop_box": [345, 2166, 857, 2678]}
    black = {"x": 223, "y": 267, "w": 7, "h": 7, "size_mm": 0.2425, "reason": "dust_mask_overlap"}
    white = {"x": 45, "y": 242, "w": 20, "h": 17, "size_mm": 0.6431, "reason": "dust_mask_overlap"}
    small = {"x": 506, "y": 185, "w": 2, "h": 2, "reason": "dust_mask_overlap", "source_reason": "area_too_small"}
    counts = {kind: {"count": 0, "dust_mask_filtered_count": 1} for kind in ("black_dot", "white_dot")}
    log = {
        "id": 3696, "inference_record_id": 10902, "client_record_id": None, "suggested": True,
        "reason": "W0F00000 黑點 0.0000 mm", "created_at": "", "detail": {
            "steps": [
                {"image": source["image"], "screen": "W0F00000"},
                {**source, "rejected": {"black_dot": [black], "white_dot": [small, white]}},
                {**source, "detection": counts},
                {**source, "detection": counts, "classified_as": "black_dot"},
            ],
            "panel_totals": [{"screen": "W0F00000", "dot_type": "black_dot", "total_count": 0}],
            "visuals": [{"image_name": source["image"], "tile_id": 0, "crop_box": source["crop_box"],
                         "dot_type": "black_dot", "dot_label": "黑點", "candidates": [],
                         "urls": {"crop_url": "/heatmaps/crop.png", "dust_mask_url": "/heatmaps/dust-mask.png",
                                  "dust_overlay_url": "/heatmaps/dust.png"}}],
        },
    }
    env = Environment(loader=FileSystemLoader("templates"))
    page = env.get_template("within_spec_detail.html").render(
        log=log, app_version={"version": ""}, host_identity="", **_within_spec_log_view(log))
    summary = page.split("判定原因與點偵測摘要", 1)[1].split("</section>", 1)[0]
    for label, size in (("黑點", "0.2425"), ("白點", "0.6431")):
        assert f"{label}偵測 1 點（最大 {size} mm），灰塵排除 1，不檢測區排除 0，計入 Panel 0 點" in summary
    overlay = page.split('<svg class="ws-main-img ws-dot-map"', 1)[1].split("</svg>", 1)[0]
    assert 'viewBox="0 0 512 512"' in overlay and 'href="/heatmaps/crop.png"' in overlay
    assert 'class="ws-map-mask" href="/heatmaps/dust-mask.png"' in overlay
    assert 'x="223" y="267" width="7" height="7"' in overlay
    assert 'x="45" y="242" width="20" height="17"' in overlay
    assert overlay.count("ws-excluded") == 2 and 'x="506"' not in overlay
    assert "/heatmaps/dust.png" in page
    assert "白點 / white_dot" in page.split("Panel 點數加總</div>")[-1].split("</table>")[0]

    # Some older logs kept counts after truncating the rejected-coordinate list.
    log["detail"]["steps"][1]["rejected"]["white_dot"] = [small]
    log["detail"]["visuals"][0]["aoi_judgment_roi"] = {"enabled": True}
    limited = env.get_template("within_spec_detail.html").render(
        log=log, app_version={"version": ""}, host_identity="", **_within_spec_log_view(log))
    assert "白點有 1 點未保存座標" in limited and "白點偵測 1 點，灰塵排除 1" in limited
    overlay = limited.split('<svg class="ws-main-img ws-dot-map"', 1)[1].split("</svg>", 1)[0]
    assert 'viewBox="0 0 512 536"' in overlay and 'translate(0 24)' in overlay
    assert overlay.count("<rect ") == 1


def test_within_spec_report_template_renders():
    env = Environment(loader=FileSystemLoader("templates"))
    env.globals.update(app_version={"version": "test"}, host_identity="")
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
