import json

import pytest

from capi_database import CAPIDatabase
from capi_station_adapter import create_station_adapter
from capi_web import CAPIWebHandler, _decorate_record_within_spec, _format_within_spec_inference_note


@pytest.mark.parametrize("profile,new_arch,grid,deferred", [
    ("capi", True, False, True),
    ("aapi", True, False, False),
    ("capi", False, False, False),
    ("capi", True, True, False),
])
def test_server_defers_visuals_only_for_capi_aoi_mode(monkeypatch, tmp_path, profile, new_arch, grid, deferred):
    from types import SimpleNamespace
    from capi_server import CAPIServer

    server = CAPIServer.__new__(CAPIServer)
    server.station_adapter = SimpleNamespace(profile=profile)
    server.heatmap_manager = SimpleNamespace(base_dir=str(tmp_path))
    server._load_within_spec_rules_for_inference = lambda _inferencer: {}
    inferencer = SimpleNamespace(
        config=SimpleNamespace(is_new_architecture=new_arch, grid_tiling_enabled=grid, aoi_coord_inspection_enabled=True),
        station_adapter=SimpleNamespace(profile="capi"),
    )
    monkeypatch.setattr("capi_server._attach_no_detect_regions_to_within_spec_detail", lambda *args: None)
    captured = {}

    def evaluate(*args, **kwargs):
        captured.update(kwargs)
        return {"suggestion": None, "panel_totals": []}

    monkeypatch.setattr("capi_server._evaluate_within_spec_suggestion_detail", evaluate)
    result = server._evaluate_within_spec_for_inference({"glass_id": "G1"}, [], inferencer)
    assert result is not None
    assert captured["station_adapter"] is server.station_adapter
    assert (captured["deferred_visual_jobs"] is not None) is deferred


def test_server_flushes_deferred_visuals_before_persisting_detail(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    from capi_server import CAPIServer

    server = CAPIServer.__new__(CAPIServer)
    server.db = MagicMock()
    server.db.save_inference_record.return_value = 7
    pending = {"pending": True}
    jobs = [{"kwargs": {}, "result": pending}]
    info = {
        "status": "not_within_spec", "reason": "test", "suggestion": None,
        "detail": {"visuals": [pending]}, "_visual_jobs": jobs,
    }

    def save_visual(**kwargs):
        server.db.save_inference_record.assert_not_called()
        return {"urls": {"crop_url": "/heatmaps/crop.png"}, "count": 1}

    monkeypatch.setattr("capi_web._save_within_spec_dot_visuals", save_visual)
    parsed = {
        "glass_id": "G1", "model_id": "M1", "machine_no": "CAPI39",
        "machine_judgment": "NG", "resolution": (1920, 1200),
        "image_dir": str(tmp_path),
    }
    server._save_results_async(
        ("test", 1), parsed, [], "NG", "[]", "2026-09-14 08:00:00", "2026-09-14 08:00:01", 1.0,
        within_spec_info=info,
    )
    assert jobs == []
    assert "_visual_jobs" not in info
    server.db.save_within_spec_review_log.assert_called_once()
    saved = server.db.save_within_spec_review_log.call_args.kwargs["detail"]
    assert saved["visuals"] == [{"urls": {"crop_url": "/heatmaps/crop.png"}, "count": 1}]


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
    assert db.get_record_detail(inference_id)["within_spec_log_id"] == saved["id"]


@pytest.mark.parametrize("handler_name", ["_handle_record_detail", "_handle_record_detail_v3"])
def test_record_shows_original_ng_and_front_release_at_the_matched_image_and_point(tmp_path, handler_name):
    db = CAPIDatabase(str(tmp_path / "record.db"))
    black, red = "B0F00000_024703", "R0F00000_024656"
    released = {"image": black, "tile_id": 0, "x": 1971, "y": 2058}
    side_point = {"image": "W0F00000", "type": "aoi_passthrough", "defect_code": "CSJBL",
                  "product_x": 21, "product_y": 566}
    note = _format_within_spec_inference_note({
        "converted": True, "scope": "front", "detail": {"panel_totals": [{
            "screen": "B0F00000", "dot_label": "白點", "max_size_mm": .295, "threshold_mm": .3,
            "total_count": 1, "screen_count_limit": 1, "max_tile_count": 1,
            "tile_count_threshold": 1, "within": True,
        }]},
    }, "/ric/within-spec-logs")
    tile = dict(tile_id=0, x=1971, y=2058, score=1., is_anomaly=1, is_aoi_coord=1,
                aoi_product_x=346, aoi_product_y=238, aoi_defect_code="PCV88")
    images = [
        dict(image_name=name + ".tif", is_ng=int(name == black), is_bomb=int(name == red),
             heatmap_path=f"/heatmaps/{name}.png", tiles=[
                 dict(tile, is_bomb=int(name == red), zone="bright_spot" if name == black else "inner",
                      heatmap_path=f"/heatmaps/{name}_tile0.png")])
        for name in (black, red)
    ]
    record_id = db.save_inference_record(
        glass_id="YQ90XL017G46", model_id="MODEL", machine_no="CAPI39", resolution=(1366, 768),
        machine_judgment="NG", ai_judgment="OK-s", image_dir="/images", total_images=2, ng_images=1,
        ng_details=json.dumps([released, side_point]), image_results_data=images, inference_log=note,
        request_time="2026-11-21 02:36:59", response_time="2026-11-21 02:37:27", processing_seconds=27.66,
    )
    db.save_side_white_result(record_id, {
        "status": "NO_CANDIDATES", "candidates": [],
        "front_decision": {"judgment": "OK-i", "within_spec_passed": True, "resolved_ng_details": [released]},
    })
    # A later review uses different rules. Its assessment is not the one that released the point.
    db.save_within_spec_review_log(None, record_id, None, {"panel_totals": []}, .2, source="review")
    CAPIWebHandler.init_jinja()
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.db, handler.heatmap_base_dir = db, "/heatmaps"
    handler._station_adapter = lambda: create_station_adapter("capi")
    rendered = []
    handler._send_response = lambda status, body: rendered.append((status, body))
    getattr(handler, handler_name)(str(record_id), f"/record/{record_id}")
    status, html = rendered[0]
    assert status == 200 and "OK-s" in html
    assert html.count('class="within-spec-badge">1 處規格內轉 OK') == 1
    assert html.count('class="within-spec-badge">原始 NG → 規格內轉 OK') == 2  # table and tile heatmap
    assert 'class="within-spec-measurement">白點：最大尺寸 0.2950mm ≤ 0.3000mm' in html
    assert html.count("原始 NG 標記") == 1 and "亮點偵測 → 原始 NG → 規格內轉 OK" in html
    assert db.get_record_detail(record_id)["images"][0]["tiles"][0]["is_anomaly"] == 1


@pytest.mark.parametrize("judgment,front,normal_pass_note,expected", [
    ("OK-i", None, False, True),   # Historical record with only ng_details.
    ("NG", None, True, True),      # Front released, side disabled.
    ("NG", True, False, True),     # Front released, side still NG.
    ("OK-s", None, False, False),  # Side-only conversion.
    ("NG", False, False, False),
    ("NG", None, False, False),    # A saved review alone is insufficient.
])
def test_within_spec_badge_requires_current_run_and_exact_tile(judgment, front, normal_pass_note, expected):
    released = {"image": "B0F00000_024703", "tile_id": 0}
    images = [
        {"image_name": stem + ".tif", "tiles": [
            {"tile_id": tile_id, "decision_evidence": {"status": "NG", "rule_label": "亮點偵測"}}
            for tile_id in (0, 1)
        ]} for stem in ("B0F00000_024703", "B0F00000_024704")
    ]
    detail = dict(images=images, ai_judgment=judgment, ng_details=json.dumps([released]), within_spec_log_id=99)
    if front is not None:
        detail["side_white_result"] = {"payload": {"front_decision": {
            "within_spec_passed": front, "judgment": "OK-i" if front else "NG",
            "resolved_ng_details": [released] if front else [],
        }}}
    if normal_pass_note:
        detail["inference_log"] = _format_within_spec_inference_note({"scope": "front", "converted": True}, "/logs")
    _decorate_record_within_spec(detail, create_station_adapter("capi"))
    assert [t.get("within_spec_converted", False) for im in images for t in im["tiles"]] == [
        expected, False, False, False,
    ]
    assert images[1].get("within_spec_tile_count", 0) == 0


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
