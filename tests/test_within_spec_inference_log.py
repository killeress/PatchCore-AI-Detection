from capi_database import CAPIDatabase
import pytest


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
