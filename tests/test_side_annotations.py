"""AOI source coordinates remain visible independently of candidate detection."""
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from jinja2 import Environment, FileSystemLoader

from capi_side_report import (SIDE_ANNOTATION_LEGEND, annotation_color, compare_side_record,
                              draw_side_annotations, render_comparison_image, side_white_view)


def side_row(rotated=False):
    return {"id": 1, "status": "NO_CANDIDATES",
            "aoi_machine_coords": {"W0F00000": [{"defect_code": "CSJBL", "coordinate_space": "product",
                "source_x": 1830, "source_y": 980, "product_x": 1830, "product_y": 980}]},
            "payload": {"status": "NO_CANDIDATES", "candidates": [], "rotation_applied": rotated,
                "side_size": [1000, 600], "front_size": [1000, 600],
                "mapping": {"status": "estimated", "matrix": np.eye(3).tolist(),
                    "front_polygon": [[0, 0], [1000, 0], [1000, 600], [0, 600]]},
                "evidence_context": {"product_resolution": [1920, 1080]}, "artifacts": {},
                "preview_bounds": {"side": [0, 0, 1000, 600], "side_clean": [0, 0, 1000, 600]}}}


@pytest.mark.parametrize("rotated", [False, True])
def test_unmatched_aoi_is_drawn_at_scaled_position_without_changing_source_or_rotation(rotated, monkeypatch):
    row = side_row(rotated)
    before = copy.deepcopy(row)
    comparison = compare_side_record(row)
    point = comparison["points"][0]
    assert point["status"] == "unmatched"
    image = np.full((300, 500, 3), 80, np.uint8)
    labels = []
    put_text = cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda image, text, *args: (labels.append(text), put_text(image, text, *args))[1])
    draw_side_annotations(image, row["payload"], comparison, "side", [0, 0, 1000, 600])
    x, y = np.rint(np.array(point["side_xy"]) * .5).astype(int)
    assert np.array_equal(image[y, x], [80, 80, 80])
    assert np.array_equal(image[y, x + 3], annotation_color("aoi"))
    assert set(labels) == {"A1"}
    assert point["source_xy"] == [1830, 980]
    expected_raw = np.array([999, 599]) - point["side_xy"] if rotated else point["side_xy"]
    assert np.allclose(point["side_raw_xy"], expected_raw)
    assert row == before


def test_all_aoi_points_keep_source_coordinates_and_invalid_points_do_not_get_invented_markers(monkeypatch):
    row = side_row()
    points = row["aoi_machine_coords"]["W0F00000"]
    points[0].update(source_x=1900, source_y=1000)  # source and resolved coordinates can differ
    points.append({**points[0], "product_x": 600, "product_y": 400, "source_x": 600, "source_y": 400})
    points.append({**points[0], "product_x": -1, "source_x": -1})
    comparison = compare_side_record(row, tolerance=0)
    assert comparison["mapped"] == 2
    labels = []
    monkeypatch.setattr(cv2, "putText", lambda image, text, *args: labels.append(text))
    image = np.full((300, 500, 3), 80, np.uint8)
    draw_side_annotations(image, row["payload"], comparison, "side", [0, 0, 1000, 600])
    assert set(labels) == {"A1", "A2"}
    assert comparison["points"][0]["source_xy"] == [1900, 1000]
    assert comparison["points"][2]["status"] == "unavailable"


def test_region_sources_use_distinct_colors_with_names_outside_the_picture(monkeypatch):
    row = side_row()
    row["payload"].update(
        exclusions=[{"type": "mark", "bbox": [30, 420, 40, 30]},
                    {"type": "corner_tape", "bbox": [900, 550, 60, 20]}],
        configured_zones=[{"front_bbox": [800, 500, 80, 30],
                           "side_polygon": [[800, 500], [880, 500], [880, 530], [800, 530]]}],
        bombs=[{"id": 1, "side_tolerance_polygon": [[300, 300], [340, 300], [340, 340], [300, 340]]}],
        candidates=[{"id": 1, "side_contour": [[500, 200], [530, 200], [530, 230], [500, 230]]}])
    labels = {}
    put_text = cv2.putText
    def capture(image, text, origin, font, size, color, *args):
        labels[text] = color
        return put_text(image, text, origin, font, size, color, *args)
    monkeypatch.setattr(cv2, "putText", capture)
    image = np.full((600, 1000, 3), 80, np.uint8)
    draw_side_annotations(image, row["payload"], compare_side_record(row), "side", [0, 0, 1000, 600],
                          all_candidates=True)
    assert set(labels) == {"A1", "B1", "#1"}
    assert labels["B1"] == annotation_color("bomb")
    assert labels["#1"] == annotation_color("candidate")
    assert len({annotation_color(item["key"]) for item in SIDE_ANNOTATION_LEGEND}) == 6
    for item in SIDE_ANNOTATION_LEGEND:
        difference = np.abs(image.astype(np.int16) - annotation_color(item["key"]))
        assert np.any(difference.max(axis=2) < 25), item["key"]


def test_aoi_annotation_keeps_defect_pixels_and_covers_under_three_percent_of_its_neighbourhood():
    row = side_row()
    comparison = compare_side_record(row)
    x, y = np.rint(comparison["points"][0]["side_xy"]).astype(int)
    image = np.full((600, 1000, 3), 160, np.uint8)
    image[y - 1:y + 2, x - 1:x + 2] = [40, 45, 50]  # a weak, small defect at the reported point
    before = image.copy()
    draw_side_annotations(image, row["payload"], comparison, "side", [0, 0, 1000, 600])
    assert np.array_equal(image[y - 1:y + 2, x - 1:x + 2], before[y - 1:y + 2, x - 1:x + 2])
    changed = np.any(image != before, axis=2)[y - 50:y + 50, x - 150:x + 50]
    assert changed.mean() < .03
    assert not np.any(np.all(image == [15, 23, 42], axis=2))  # no opaque label backing


def test_saved_clean_preview_rebuilds_old_colors_and_keeps_point_evidence_clean(tmp_path):
    row = side_row()
    clean = np.full((300, 500, 3), 80, np.uint8)
    marked = clean.copy()
    cv2.rectangle(marked, (5, 100), (15, 110), (230, 150, 60), 2)
    for name, pixels in (("side", marked), ("side_clean", clean)):
        path = tmp_path / (name + ".jpg")
        assert cv2.imwrite(str(path), pixels)
        row["payload"]["artifacts"][name] = str(path)
    before = copy.deepcopy(row)
    comparison = compare_side_record(row)
    overview = render_comparison_image(row, comparison, tmp_path)
    assert np.array_equal(overview[100, 10], [225, 225, 225])
    point = comparison["points"][0]
    x, y = np.rint(np.array(point["side_xy"]) * .5).astype(int)
    assert np.array_equal(overview[y, x], [225, 225, 225])
    assert np.array_equal(overview[y, x + 3], annotation_color("aoi"))
    assert np.all(render_comparison_image(row, comparison, tmp_path, point_id=1) == 80)
    assert row == before
    assert Path(row["payload"]["artifacts"]["side"]).read_bytes() != Path(row["payload"]["artifacts"]["side_clean"]).read_bytes()


def test_raw_txt_table_is_visible_without_mapping_and_escapes_report_text():
    row = side_row()
    row["aoi_machine_coords"]["W0F00000"][0]["defect_code"] = "CSJBL"
    row["aoi_machine_coords"] = json.dumps({"<img src=x onerror=alert(1)>": row["aoi_machine_coords"]["W0F00000"]})
    row["payload"]["mapping"] = {"status": "unavailable", "reason": "缺少正拍白畫面"}
    env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parents[1] / "templates"))
    env.globals["side_white_view"] = side_white_view
    html = env.get_template("_side_white_result.html").render(detail={"side_white_result": row})
    assert html.index('data-sw-aoi-point="1"') < html.index('<details class="sw-technical">')
    assert "(1830, 980)" in html and "CSJBL" in html
    assert "缺少正拍白畫面" in html and "無法定位" in html
    assert "<img src=x onerror=alert(1)>" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert all(item["color"] in html and item["hint"] in html for item in SIDE_ANNOTATION_LEGEND)


def test_clean_image_links_are_available_for_record_and_debug_views():
    row = side_row()
    row["payload"]["artifacts"] = {"side": "side.jpg", "side_clean": "clean.jpg"}
    env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parents[1] / "templates"))
    env.globals["side_white_view"] = side_white_view
    template = env.get_template("_side_white_result.html")
    html = template.render(detail={"side_white_result": row})
    assert "kind=side_clean" in html and "無標記圖 ↗" in html
    html = template.render(detail={"side_white_result": row}, debug_result=True,
                           artifact_urls={"side": "/debug/side.jpg", "side_clean": "/debug/clean.jpg"})
    assert 'href="/debug/clean.jpg"' in html
