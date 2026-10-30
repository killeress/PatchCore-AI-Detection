"""Whole-panel previews use all formal NG reasons from the same inference."""
import copy
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from capi_side_report import compare_side_record, record_conversion, side_white_view
from test_side_conversion_statistics import sample


def report_row(kinds=("mark",)):
    return dict(sample(kinds), id=10, record_id=100, glass_id="SAME-PANEL", machine_no="CAPI1",
                model_id="MODEL", request_time="2026-10-29 10:00:00")


def render_row(row):
    env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parents[1] / "templates"))
    env.globals["side_white_view"] = side_white_view
    return env.get_template("_side_statistics_row.html").render(row=row, comparison=compare_side_record(row))


@pytest.mark.parametrize("kind", ["mark", "omit"])
def test_panel_can_convert_when_all_formal_ng_is_covered_by_side_results(kind):
    row = report_row((kind,))
    before = copy.deepcopy(row)
    view = side_white_view(row)
    assert view["panel"] == {
        "formal": "NG", "projected": "OK", "state": "eligible", "tone": "ok",
        "text": "整片可轉 OK", "reason": "各畫面的全部正式 NG 原因均已由側拍處理涵蓋",
        "remaining_sources": [],
    }
    html = render_row(row)
    summary = html.split('</summary>', 1)[0]
    assert 'data-panel-state="eligible"' in summary and "整片可轉 OK" in summary
    assert summary.index("整片試算") < summary.index("貼紙／表面異物")
    assert "正式 NG → 試算 OK" in html and "正式結果：NG" in html
    assert row == before


def test_side_ok_does_not_clear_formal_ng_from_other_screens():
    row = report_row(("omit",))
    row["ng_details"].extend([
        {"type": "ai_defect", "image": "G0F00000_100001.tif"},
        {"type": "ai_defect", "image": "B0F00000_100002.tif"},
        {"type": "ai_defect", "image": "G0F00000_100003.tif"},
    ])
    before = copy.deepcopy(row)
    view = side_white_view(row)
    assert view["sticker"]["tone"] == "ok"
    assert view["panel"]["state"] == "retained" and view["panel"]["projected"] == "NG"
    assert view["panel"]["remaining_sources"] == ["G0F00000", "B0F00000"]
    assert view["panel"]["reason"] == "仍有正式 NG：G0F00000、B0F00000"
    html = render_row(row)
    summary = html.split('</summary>', 1)[0]
    assert "整片仍為 NG" in summary and "G0F00000" in summary and "B0F00000" in summary
    assert "已排除 · 1 點可轉 OK" in summary
    assert "正式 NG → 試算 NG" in html
    assert row == before


@pytest.mark.parametrize("details", [[], "broken JSON", {}, None])
def test_missing_formal_ng_details_cannot_present_a_panel_as_ok(details):
    row = report_row()
    row["ng_details"] = details
    panel = side_white_view(row)["panel"]
    assert panel["projected"] == "NG" and panel["state"] == "retained"
    assert "明細不足" in panel["reason"]
    assert "整片仍為 NG" in render_row(row).split('</summary>', 1)[0]


@pytest.mark.parametrize("formal", ["OK", "OK-i", "ERR:NO_IMAGE", ""])
def test_formal_ok_and_invalid_results_are_distinguished(formal):
    row = report_row()
    row["ai_judgment"] = formal
    panel = side_white_view(row)["panel"]
    if formal in {"OK", "OK-i"}:
        assert panel["state"] == "already_ok" and panel["projected"] == formal
        assert panel["text"] == f"原本已 {formal}"
    else:
        assert panel["state"] == "unavailable" and panel["projected"] is None
        assert panel["text"] == "無法試算"


def test_aoi_points_without_side_anomalies_can_complete_whole_panel_release():
    row = report_row(("mark", "unknown"))
    panel = side_white_view(row)["panel"]
    assert panel["projected"] == "OK" and panel["state"] == "eligible"


def test_repeated_glass_ids_keep_separate_whole_panel_results():
    first = report_row()
    second = copy.deepcopy(first)
    second.update(id=11, record_id=101)
    second["ng_details"].append({"type": "ai_defect", "image": "R0F00000"})
    first_html, second_html = render_row(first), render_row(second)
    assert 'data-panel-state="eligible"' in first_html.split('</summary>', 1)[0]
    assert 'data-panel-state="retained"' in second_html.split('</summary>', 1)[0]
    assert 'href="/record/100#' in first_html and 'href="/record/101#' in second_html


def test_panel_projection_keeps_existing_eligibility_and_escapes_source_labels():
    row = report_row()
    row["ng_details"].append({"type": "ai_defect", "image": '<img src=x onerror=bad()>'})
    comparison = compare_side_record(row)
    conversion = record_conversion(row, comparison)
    view = side_white_view(dict(row, comparison=comparison, conversion=conversion))
    assert view["panel"]["projected"] == ("OK" if conversion["eligible"] else "NG")
    html = render_row(row)
    assert '<img src=x' not in html.lower() and '&lt;img src=x' in html.lower()
