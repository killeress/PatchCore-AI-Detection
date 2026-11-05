import re

import pytest

from capi_web import CAPIWebHandler, machine_judgment_note


@pytest.mark.parametrize("original,stored,expected_note", [
    ("NG", "OK", "原始 NG → OK（炸彈排除）"),
    ("OK", "OK", ""),
    ("NG", "NG", ""),
    ("HY", "HY", ""),
    (None, "OK", ""),
])
@pytest.mark.parametrize("template", ["record_detail.html", "record_detail_v3.html"])
def test_machine_conversion_note_is_in_machine_judgment_cell(template, original, stored, expected_note):
    detail = dict(id=1, ai_judgment="OK", machine_judgment=stored,
                  processing_seconds=1.0, images=[])
    if original:
        detail["client_request_text"] = f"AOI@G1;M1;AAPI2612;1920,1200;{original};/images"
    assert machine_judgment_note(detail) == expected_note
    CAPIWebHandler.init_jinja()
    html = CAPIWebHandler.jinja_env.get_template(template).render(
        detail=detail, heatmap_base_dir="/heatmaps")
    cell = re.search(r"機檢判定</(?:th|td)>\s*<td[^>]*>(.*?)</td>", html, re.S)
    assert cell is not None
    assert ("data-machine-judgment-note" in cell.group(1)) is bool(expected_note)
    if expected_note:
        assert expected_note in cell.group(1)


def test_legacy_converted_record_is_annotated_from_original_response():
    assert machine_judgment_note({
        "machine_judgment": "OK",
        "client_response_text": "AOI@G1;M1;AAPI2612;NG;OK\r\n@QJPG-G1;OK;KA;NG,",
    }) == "原始 NG → OK（炸彈排除）"


@pytest.mark.parametrize("template", ["index.html", "search.html"])
def test_record_lists_distinguish_converted_ok_from_original_ok(template):
    records = [
        dict(id=index, glass_id=f"G{index}", model_id="M1", machine_no="AAPI2612",
             ai_judgment="OK", machine_judgment=stored, processing_seconds=1.0,
             client_request_text=f"AOI@G{index};M1;AAPI2612;1920,1200;{original};/images")
        for index, (original, stored) in enumerate([("NG", "OK"), ("OK", "OK"), ("NG", "NG")], 1)
    ]
    CAPIWebHandler.init_jinja()
    html = CAPIWebHandler.jinja_env.get_template(template).render(
        records=records, stats={}, ok_rate=0, ng_rate=0, total_count=3,
        page=1, total_pages=1, limit=50, cross_filter="ng_ok")

    rows = re.findall(r"<tr>\s*<td><a href=\"/record/\d+\".*?</tr>", html, re.S)
    assert len(rows) == 3
    assert "原始 NG → OK（炸彈排除）" in rows[0]
    assert all("data-machine-judgment-note" not in row for row in rows[1:])
