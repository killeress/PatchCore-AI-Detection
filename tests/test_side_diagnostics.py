"""Copy exports preserve the selected side snapshot and its actual log provenance."""
import copy
import json
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from jinja2 import Environment, FileSystemLoader

from capi_database import CAPIDatabase
from capi_side_report import side_white_diagnostics, side_white_view


def payload():
    return {"algorithm": "side-white-cv-v7", "status": "CANDIDATES", "processing_ms": 1372.3,
            "side_image": "SW0F00000_144217.tif", "front_image": "W0F00000_144217.tif",
            "candidates": [{"id": 1, "side_contour": [[1, 2], [3, 4], [5, 6]],
                            "dust": {"status": "unavailable", "holes": [[[1, 1], [2, 1], [1, 2]]]}}],
            "parameters": {"csjbl_match_tolerance_product_px": 5, "csjbl_release_guard_radius_product_px": 10},
            "future_field": {"text": '</script><img src=x onerror="bad()">', "large_array": list(range(2000))},
            "inference_decision": {"original_judgment": "NG", "final_judgment": "NG", "converted": False,
                "reason": "A1 最近候選 #1 距離 9.52 產品 px，位置待確認，保留 NG", "points": []}}


def test_database_export_is_complete_and_does_not_mix_rerun_logs(tmp_path):
    db = CAPIDatabase(str(tmp_path / "diagnostics.db"))
    raw = ("[2026-11-03 08:00:00] INFO [capi.side_white] [SIDE_WHITE_PAIR] paired-original\n"
           "[2026-11-03 08:00:01] INFO [capi.server] front-only-data\n"
           "[2026-11-03 08:00:02] ERROR [capi.side_white] side-details\nTraceback (most recent call last):\n  side failure detail")
    record_id = db.save_inference_record(
        glass_id="GLASS-1", model_id="MODEL", machine_no="CAPI07", resolution=(1920, 1080),
        machine_judgment="NG", ai_judgment="NG", image_dir="/images/GLASS-1", total_images=1, ng_images=1,
        ng_details='[{"defect_code":"CSJBL"}]', aoi_machine_coords='{"W0F00000":[]}',
        request_time="2026-11-03 08:00:00", response_time="2026-11-03 08:00:05", processing_seconds=5,
        inference_log=raw)
    original = payload()
    result_id = db.save_side_white_result(record_id, original)
    row = db.get_side_white_diagnostics(result_id)
    before = copy.deepcopy(row)
    exported = side_white_diagnostics(row)
    assert exported["payload"] == original and row == before
    assert exported["record"]["record_id"] == record_id
    assert exported["record"]["side_result_id"] == result_id
    assert exported["record"]["glass_id"] == "GLASS-1"
    assert exported["record"]["machine_no"] == "CAPI07"
    assert exported["record"]["app_version"] == row["app_version"]
    assert exported["ng_details"] == [{"defect_code": "CSJBL"}]
    assert exported["side_log_source"] == "inference_log"
    assert "paired-original" in exported["side_log"] and "front-only-data" not in exported["side_log"]
    assert "Traceback" in exported["side_log"] and "side failure detail" in exported["side_log"]
    assert "[SIDE_WHITE_RESULT]" in exported["side_log"] and "[SIDE_WHITE_INFERENCE]" in exported["side_log"]
    newer = {**original, "processing_ms": 2000, "side_image": "SW0F00000_150000.tif", "front_image": "W0F00000_150000.tif"}
    new_log = "[SIDE_WHITE_PAIR] paired-new\n[SIDE_WHITE_RESULT] {}"
    old_export = side_white_diagnostics({**row, "inference_log": new_log})
    assert old_export["payload"] == original
    assert old_export["side_log_source"] == "payload_snapshot"
    assert "非原始執行 LOG" in old_export["side_log"]
    assert "150000" not in old_export["side_log"]
    db.update_record_for_rerun(record_id, ai_judgment="NG", total_images=1, ng_images=1,
                              ng_details='[{"defect_code":"CSJBL"}]', processing_seconds=5,
                              inference_log="[SIDE_WHITE_PAIR] paired-new")
    assert db.get_side_white_diagnostics(result_id) is None
    newer_id = db.save_side_white_result(record_id, newer)
    latest = side_white_diagnostics(db.get_side_white_diagnostics(newer_id))
    assert latest["side_log_source"] == "inference_log"
    assert "paired-original" not in latest["side_log"]
    assert "paired-new" in latest["side_log"]
    assert "150000" in latest["side_log"] and "144217" not in latest["side_log"]
    assert db.get_side_white_diagnostics(999999) is None


def test_missing_or_malformed_log_is_explicitly_labelled_as_reconstructed():
    for raw in (None, "", "[SIDE_WHITE_RESULT] {broken", "[SIDE_WHITE_RESULT] []"):
        row = {"id": 3, "record_id": 5, "payload": payload(), "inference_log": raw}
        exported = side_white_diagnostics(row)
        assert exported["side_log_source"] == "payload_snapshot"
        assert "非原始執行 LOG" in exported["side_log"]
        assert exported["payload"] == payload()


@pytest.mark.parametrize("debug", [False, True])
def test_copy_button_supports_saved_and_debug_cards_without_unsafe_html(debug):
    env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parents[1] / "templates"))
    env.globals["side_white_view"] = side_white_view
    original = payload()
    detail = {"id": 51, "glass_id": "G-1", "machine_no": "CAPI07", "model_id": "MODEL",
              "side_white_result": {"id": "debug" if debug else 3, "status": "CANDIDATES", "payload": original}}
    html = env.get_template("_side_white_result.html").render(
        detail=detail, debug_result=debug, artifact_urls={})
    assert "data-sw-copy-diagnostics" in html and "複製排查資料" in html
    assert html.index("data-sw-copy-diagnostics") < html.index("data-sw-conclusion")
    assert 'role="status" aria-live="polite"' in html
    assert '</script><img src=x onerror="bad()">' not in html
    if debug:
        class Scripts(HTMLParser):
            def __init__(self):
                super().__init__()
                self.active = False
                self.data = ""
            def handle_starttag(self, tag, attrs):
                self.active = tag == "script" and "data-sw-debug-diagnostics" in dict(attrs)
            def handle_data(self, data):
                if self.active:
                    self.data += data
            def handle_endtag(self, tag):
                if tag == "script":
                    self.active = False
        parsed = Scripts()
        parsed.feed(html)
        exported = json.loads(parsed.data)
        assert exported["payload"] == original
        assert exported["source"] == "debug_side_white_result"
        assert exported["record"]["record_id"] == 51
    else:
        assert 'data-sw-result-id="3"' in html
        assert "data-sw-debug-diagnostics" not in html
        assert "large_array" not in html  # Saved payloads are loaded only on click.


def test_diagnostics_endpoint_validates_id_and_keeps_read_only_access():
    from capi_web import CAPIWebHandler

    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.db = MagicMock()
    row = {"id": 3, "record_id": 5, "payload": payload()}
    handler.db.get_side_white_diagnostics.return_value = row
    responses = []
    handler._send_json = lambda data, status=200, **kw: responses.append((status, data, kw))
    handler._handle_api_side_white_diagnostics({"id": ["3"]})
    assert responses[-1][0] == 200 and responses[-1][1]["payload"] == row["payload"]
    assert responses[-1][2]["headers"]["Cache-Control"] == "no-store"
    assert responses[-1][2]["compress"]
    handler.db.get_side_white_diagnostics.assert_called_once_with(3)
    handler.db.get_side_white_diagnostics.reset_mock()
    for query in ({}, {"id": []}, {"id": ["bad"]}, {"id": ["0"]}, {"id": ["-1"]}):
        handler._handle_api_side_white_diagnostics(query)
        assert responses[-1][0] == 400
    handler.db.get_side_white_diagnostics.assert_not_called()
    handler.db.get_side_white_diagnostics.return_value = None
    handler._handle_api_side_white_diagnostics({"id": ["99"]})
    assert responses[-1][0] == 404
