"""Covered stickers are AOI over-detections; their CV halo is not a new defect."""
import copy
import json
from html.parser import HTMLParser
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from capi_config import normalize_side_white_params
from capi_database import CAPIDatabase
from capi_side_report import compare_side_record, record_conversion, side_white_view


def sticker_row():
    contour = [[90, 155], [210, 155], [210, 220], [90, 220]]
    core = [[110, 178], [190, 178], [190, 204], [110, 204]]
    points = [{"defect_code": "CSJBL", "product_x": x, "product_y": y}
              for x, y in [(150, 170), (120, 190), (185, 190)]]
    return {"id": 1, "record_id": 7, "glass_id": "STICKER", "status": "CANDIDATES", "ai_judgment": "NG",
            "resolution_x": 400, "resolution_y": 240, "review_decision": "unreviewed",
            "aoi_machine_coords": {"W0F00000": points},
            "ng_details": [{**p, "image": "W0F00000", "type": "aoi_passthrough"} for p in points],
            "payload": {"algorithm": "side-white-cv-v5", "status": "CANDIDATES",
                "side_size": [400, 240], "front_size": [400, 240], "front_image": "W0F00000.tif",
                "parameters": normalize_side_white_params({"dust_mode": "suppress"}),
                "mapping": {"status": "estimated", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            "front_polygon": [[0, 0], [400, 0], [400, 240], [0, 240]]},
                "evidence_context": {"product_resolution": [400, 240],
                    "aoi_csjbl_points": [{**p, "image_prefix": "W0F00000"} for p in points],
                    "inference_marks": [{"source_image": "W0F00000.tif", "name": "mark_binary",
                                         "front_bbox": [20, 180, 15, 10]}]},
                "omit": {"status": "available", "coordinate_space": "side_detection_pixels", "size": [400, 240]},
                "artifacts": {"side": "side.jpg", "front": "front.jpg", "omit": "omit.jpg"},
                "candidates": [{"id": 1, "disposition": "retained", "bomb_ids": [],
                    "side_contour": contour, "remaining_contours": [contour], "area_px": 5000,
                    "remaining_area_px": 2500, "dust": {"status": "suppressed", "core_area_px": 1800,
                        "core_overlap_ratio": 1.0, "overlap_ratio": .5, "halo_removed_area_px": 0,
                        "suppressed_regions": [{"contour": core, "holes": []}]}}]}}


def test_existing_fully_covered_sticker_and_its_halo_convert_all_three_points_without_rerun():
    row = sticker_row()
    before = copy.deepcopy(row)
    comparison = compare_side_record(row, tolerance=0)
    assert comparison["omit_suppressed"] == 3 and comparison["matched"] == 0
    assert all(p["sticker_candidate_ids"] == [1] for p in comparison["points"])
    assert record_conversion(row, comparison)["eligible"]
    assert row == before  # Classification never rewrites old masks or formal results.
    view = side_white_view(row)
    assert view["headline"] == "AOI 3 點可轉 OK"
    assert view["sticker"]["text"] == "已排除 · 3 點可轉 OK"
    assert view["mark"]["text"] == "MARK 屏蔽範圍 1 區，AOI 未命中"
    assert view["mark"]["tone"] == "muted"
    assert view["candidates"][0]["label"] == "貼紙已排除"


@pytest.mark.parametrize("invalid", ["partial_core", "observe", "off", "unavailable", "size", "no_mask",
                                     "bomb", "error", "truncated", "outside", "mask_hole"])
def test_sticker_rule_requires_full_evidence_and_an_aoi_point_on_the_object(invalid):
    row = sticker_row()
    payload = row["payload"]
    dust = payload["candidates"][0]["dust"]
    if invalid == "partial_core":
        dust["core_overlap_ratio"] = .99
    elif invalid in {"observe", "off"}:
        payload["parameters"]["dust_mode"] = invalid
        dust["status"] = "suspected" if invalid == "observe" else "disabled"
    elif invalid == "unavailable":
        payload["omit"]["status"] = "unavailable"
    elif invalid == "size":
        payload["omit"]["size"] = [200, 120]
    elif invalid == "no_mask":
        dust.pop("suppressed_regions")
    elif invalid == "bomb":
        payload["candidates"][0]["bomb_ids"] = [1]
    elif invalid == "error":
        row["status"] = "ERROR"
    elif invalid == "truncated":
        payload["truncated"] = True
    else:
        row["aoi_machine_coords"]["W0F00000"] = [{"defect_code": "CSJBL", "product_x": 225 if invalid == "outside" else 150,
                                                   "product_y": 190}]
        if invalid == "mask_hole":
            dust["suppressed_regions"][0]["holes"] = [[[145, 185], [155, 185], [155, 195], [145, 195]]]
    comparison = compare_side_record(row, tolerance=500)
    assert comparison["omit_suppressed"] == 0
    assert not record_conversion(row, comparison)["eligible"]


def test_other_retained_candidate_or_formal_ng_is_not_released_with_sticker():
    row = sticker_row()
    row["payload"]["candidates"].append({"id": 2, "disposition": "retained", "bomb_ids": [],
        "side_contour": [[145, 163], [155, 163], [155, 175], [145, 175]]})
    comparison = compare_side_record(row, tolerance=0)
    assert comparison["points"][0]["status"] == "matched"
    assert comparison["points"][0]["retained_candidate_ids"] == [2]
    assert not record_conversion(row, comparison)["eligible"]
    row = sticker_row()
    row["ng_details"].append({"type": "ai_defect", "image": "G0F00000"})
    assert compare_side_record(row)["omit_suppressed"] == 3
    assert not record_conversion(row, compare_side_record(row))["eligible"]


def test_statistics_count_the_sticker_record_once_and_preserve_formal_record(tmp_path):
    row = sticker_row()
    db = CAPIDatabase(str(tmp_path / "sticker.db"))
    record_id = db.save_inference_record(glass_id="STICKER", model_id="MODEL", machine_no="CAPI1",
        resolution=(400, 240), machine_judgment="NG", ai_judgment="NG", image_dir="samples", total_images=1,
        ng_images=1, ng_details=json.dumps(row["ng_details"]), request_time="2026-10-28 12:00:00",
        response_time="2026-10-28 12:00:01", processing_seconds=1, client_response_text="original",
        aoi_machine_coords=json.dumps(row["aoi_machine_coords"]))
    conn = db._get_conn()
    conn.execute("UPDATE inference_records SET created_at='2026-10-28 12:00:00' WHERE id=?", (record_id,))
    conn.commit()
    conn.close()
    db.save_side_white_result(record_id, row["payload"])
    before = db.get_record_detail(record_id)
    result = db.get_side_statistics(start_date="2026-10-28", end_date="2026-10-28", shift="day")
    assert result["conversion_summary"]["omit"] == {"points": 3, "records": 1}
    assert result["conversion_summary"]["eligible_records"] == 1
    assert db.get_record_detail(record_id) == before
    assert side_white_view(db.list_side_white_results()["rows"][0])["conversion"]["eligible"]


class DisclosureParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = self.maximum = self.details = 0
        self.treatments = []

    def handle_starttag(self, tag, attrs):
        if tag == "details":
            self.depth += 1
            self.details += 1
            self.maximum = max(self.maximum, self.depth)
        if "data-sw-treatment" in dict(attrs):
            self.treatments.append((dict(attrs)["data-sw-treatment"], self.depth))

    def handle_endtag(self, tag):
        if tag == "details":
            self.depth -= 1


@pytest.mark.parametrize("statistics", [False, True])
def test_treatment_conclusions_are_visible_without_nested_evidence_disclosures(statistics):
    row = sticker_row()
    env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parents[1] / "templates"))
    env.globals["side_white_view"] = side_white_view
    if statistics:
        html = env.get_template("_side_statistics_row.html").render(
            row=row, comparison=compare_side_record(row))
        summary = html.split('</summary>', 1)[0]
        assert "已排除 · 3 點可轉 OK" in summary and "MARK 屏蔽範圍 1 區，AOI 未命中" in summary
        assert '<span class="sw-tone-muted">MARK 屏蔽範圍 1 區，AOI 未命中</span>' in summary
    else:
        html = env.get_template("_side_white_result.html").render(detail={**row, "side_white_result": row})
    parser = DisclosureParser()
    parser.feed(html)
    assert '<div class="sw-conclusion sw-tone-muted" data-sw-treatment="mark">' in html
    assert parser.treatments == [(key, int(statistics)) for key in ("sticker", "mark", "tape", "clear")]
    assert parser.details == 2 + int(statistics)  # Legend help and detection details.
    assert parser.maximum == 1 + int(statistics)
    assert html.count('<figure>') == 3 + 3 * len(row["aoi_machine_coords"]["W0F00000"])
    assert '正式結果：NG' in html and 'AOI 3 點可轉 OK' in html
