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
        "remaining_sources": [], "remaining_ng": [],
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
    assert view["panel"]["reason"] == "仍有 3 筆正式 NG：G0F00000、B0F00000"
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


@pytest.mark.parametrize("location,label,xy", [
    ({"peak_x": 4344, "peak_y": 1108}, "推論影像座標", "(4344, 1108)"),
    ({"product_x": 1359, "product_y": 232, "peak_x": 4344, "peak_y": 1108,
      "defect_code": "PCDK2"}, "產品座標", "(1359, 232)"),
    ({"x": 0, "y": 852}, "影像框左上", "(0, 852)"),
])
def test_clear_side_point_still_shows_other_ng_in_summary_and_details(location, label, xy):
    row = report_row(("unknown",))
    row["ng_details"].append({"image": "W0F00000_130558", "tile_id": 1, "score": 1.116, **location})
    row["payload"]["inference_decision"] = {
        "converted": False, "original_judgment": "NG", "final_judgment": "NG",
        "reason": "另有正式 NG 原因，或 NG 明細不足以確認全部已涵蓋",
    }
    before = copy.deepcopy(row)
    view = side_white_view(row)
    assert view["comparison"]["no_anomaly"] == 1
    assert view["headline"] == "側拍 1 點可轉 OK；整片仍為 NG" and view["tone"] == "warn"
    assert not view["conversion"]["eligible"]
    assert len(view["panel"]["remaining_ng"]) == 1  # The covered CSJBL is not another blocker.
    html = render_row(row)
    summary = html.split('</summary>', 1)[0]
    assert "仍有 1 筆正式 NG：W0F00000" in summary
    assert "側拍 1／1 點可轉 OK（無異常 1 點）" in summary
    clear_card = html.split('data-sw-treatment="clear">', 1)[1].split('</div>', 1)[0]
    assert "AOI 位置無異常" in clear_card and "1 點可轉 OK" in clear_card
    main = html.split('<details class="sw-technical">', 1)[0]
    assert "另有正式 NG 原因，或 NG 明細不足" not in main
    blockers = html.split('data-sw-ng-blockers>', 1)[1].split('</table>', 1)[0]
    assert "W0F00000_130558" in blockers and "AI 檢出異常" in blockers
    assert label in blockers and xy in blockers and "1.116" in blockers
    assert ("PCDK2" in blockers) == ("defect_code" in location)
    assert "判定時保存原因：另有正式 NG 原因" in html
    assert row == before


def test_blocker_details_keep_missing_evidence_explicit_and_zero_values_visible():
    row = report_row(("unknown",))
    row["ng_details"].extend([None, {"image": "G0F00000", "type": "cv_edge_left",
                                    "peak_x": 0, "peak_y": 0, "score": 0},
                             {"image": "B0F00000", "peak_x": "nan", "peak_y": 20, "score": "inf"}])
    view = side_white_view(row)
    assert view["panel"]["reason"] == "仍有 3 筆正式 NG：未保存來源、G0F00000、B0F00000"
    missing, edge, invalid = view["panel"]["remaining_ng"]
    assert missing["label"] == "正式 NG 明細不足" and missing["coordinates"] == "未保存座標"
    assert edge["label"] == "CV 邊緣異常" and edge["score_label"] == "最大灰階差"
    assert edge["coordinates"] == "(0, 0)" and edge["score"] == "0"
    assert invalid["coordinates"] == "未保存座標" and invalid["score"] == "未保存"
    assert not view["conversion"]["eligible"]
    html = render_row(row)
    assert "最大灰階差 0" in html and "未保存座標" in html


def test_same_record_blockers_are_available_in_statistics_and_record_detail(tmp_path):
    from capi_database import CAPIDatabase
    from test_side_conversion_statistics import save, query

    db = CAPIDatabase(str(tmp_path / "blockers.db"))
    row = report_row(("unknown",))
    row["ng_details"].append({"image": "W0F00000_130558", "tile_id": 1,
                              "peak_x": 4344, "peak_y": 1108, "score": 1.116})
    record_id, _ = save(db, row)
    before = db.get_record_detail(record_id)
    report = query(db)
    assert report["conversion_summary"]["blocked_records"] == 1
    assert report["conversion_summary"]["eligible_records"] == 0
    statistics = side_white_view(report["rows"][0])
    detail = side_white_view(before["side_white_result"], before)
    assert statistics["panel"]["remaining_ng"] == detail["panel"]["remaining_ng"]
    assert "W0F00000_130558" in render_row(report["rows"][0])
    assert db.get_record_detail(record_id) == before
