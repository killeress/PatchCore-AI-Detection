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
from test_side_mark_roi import roi_row


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
    assert point["status"] == "no_anomaly"
    image = np.full((300, 500, 3), 80, np.uint8)
    labels = []
    put_text = cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda image, text, *args: (labels.append(text), put_text(image, text, *args))[1])
    draw_side_annotations(image, row["payload"], comparison, "side", [0, 0, 1000, 600])
    x, y = np.rint(np.array(point["side_xy"]) * .5).astype(int)
    assert np.array_equal(image[y, x], [80, 80, 80])
    assert np.array_equal(image[y, x + 16], annotation_color("aoi"))
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
        candidates=[{"id": 1, "side_contour": [[945, 535], [965, 535], [965, 555], [945, 555]]},
                    {"id": 2, "side_contour": [[500, 200], [530, 200], [530, 230], [500, 230]]}])
    labels = {}
    put_text = cv2.putText
    def capture(image, text, origin, font, size, color, *args):
        labels[text] = color
        return put_text(image, text, origin, font, size, color, *args)
    monkeypatch.setattr(cv2, "putText", capture)
    image = np.full((600, 1000, 3), 80, np.uint8)
    draw_side_annotations(image, row["payload"], compare_side_record(row), "side", [0, 0, 1000, 600],
                          all_candidates=True)
    assert set(labels) == {"A1", "B1", "#1", "#2"}
    assert labels["B1"] == annotation_color("bomb")
    assert labels["#1"] == annotation_color("candidate")
    assert labels["#2"] == annotation_color("reference")
    assert len({annotation_color(item["key"]) for item in SIDE_ANNOTATION_LEGEND}) == 7
    for item in SIDE_ANNOTATION_LEGEND:
        difference = np.abs(image.astype(np.int16) - annotation_color(item["key"]))
        assert np.any(difference.max(axis=2) < 25), item["key"]


@pytest.mark.parametrize("kind", ["side", "front"])
@pytest.mark.parametrize("all_candidates", [False, True])
def test_yellow_only_marks_retained_candidates_in_nearby_and_all_views(kind, all_candidates, monkeypatch):
    row = roi_row()
    payload = row["payload"]
    for candidate_id, x, disposition, bombs in [
            (2, 137, "retained", []), (3, 300, "retained", []),
            (4, 160, "excluded", []), (5, 165, "dust_suppressed", []),
            (6, 170, "mark_suppressed", []), (7, 175, "retained", [1])]:
        contour = [[x, 95], [x + 5, 95], [x + 5, 105], [x, 105]]
        payload["candidates"].append({"id": candidate_id, "side_contour": contour,
                                      "disposition": disposition, "bomb_ids": bombs})
    before = copy.deepcopy(row)
    comparison = compare_side_record(row)
    labels = []
    put_text = cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda image, text, *args: (labels.append(text), put_text(image, text, *args))[1])
    image = np.full((300, 500, 3), 160, np.uint8)
    draw_side_annotations(image, payload, comparison, kind, [0, 0, 500, 300], all_candidates=all_candidates)
    assert {label for label in labels if label.startswith('#')} == ({"#2", "#3"} if all_candidates else {"#2"})
    roi = image[90:121, 90:121]
    assert not np.any((roi[:, :, 0] < 100) & (roi[:, :, 1] > 130) & (roi[:, :, 2] > 170))
    assert side_white_view(dict(row, comparison=comparison))["yellow_candidates"] == {"nearby": 1, "total": 1}
    assert row == before


@pytest.mark.parametrize("kind", ["side", "front"])
@pytest.mark.parametrize("remaining_x", [170, 300])
def test_yellow_draws_only_nearby_remaining_contours_of_a_partly_masked_candidate(kind, remaining_x):
    row = roi_row()
    candidate = row["payload"]["candidates"][0]
    candidate["id"] = 8
    outside = [[remaining_x, 95], [remaining_x + 10, 95], [remaining_x + 10, 105], [remaining_x, 105]]
    candidate["remaining_contours"].append(outside)
    before = copy.deepcopy(row)
    image = np.full((300, 500, 3), 160, np.uint8)
    draw_side_annotations(image, row["payload"], compare_side_record(row), kind, [0, 0, 500, 300])
    yellow = (image[:, :, 0] < 100) & (image[:, :, 1] > 130) & (image[:, :, 2] > 170)
    assert not yellow[90:121, 90:121].any()
    assert not yellow[94:107, remaining_x - 1:remaining_x + 12].any()
    assert side_white_view(row)["reference_candidates"]["total"] == 1
    assert row == before


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


def test_aoi_frame_is_64_native_pixels_and_leaves_the_entire_interior_untouched():
    row = side_row()
    comparison = compare_side_record(row)
    xy = comparison["points"][0]["side_xy"]
    # Offset preview origin and reduce resolution: the frame follows original pixels.
    bounds = [100, 100, 1000, 600]
    image = np.full((300, 500, 3), 80, np.uint8)
    draw_side_annotations(image, row["payload"], comparison, "side", bounds)
    low = np.rint((np.asarray(xy) - 32 - bounds[:2]) * .5).astype(int)
    high = np.rint((np.asarray(xy) + 32 - bounds[:2]) * .5).astype(int)
    assert np.array_equal(high - low, [32, 32])
    assert np.all(image[low[1] + 1:high[1], low[0] + 1:high[0]] == 80)
    assert np.array_equal(image[low[1], low[0]], annotation_color("aoi"))
    assert np.array_equal(image[high[1], high[0]], annotation_color("aoi"))


def test_point_crops_render_inline_with_detachable_frame_and_missing_camera_placeholder():
    from html.parser import HTMLParser

    class Images(HTMLParser):
        def __init__(self):
            super().__init__()
            self.crops = []
            self.current_crop = False

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'a':
                self.current_crop = attrs.get('class') == 'sw-crop-open'
                if self.current_crop:
                    assert 'target' not in attrs
                    assert 'data-sw-image' in attrs
                    assert float(attrs['data-sw-frame-ratio']) > 0
            if tag == 'img' and self.current_crop:
                self.crops.append(attrs)

        def handle_endtag(self, tag):
            if tag == 'a':
                self.current_crop = False

    row = side_row()
    row['payload']['artifacts'] = {'front': 'front.jpg', 'side': 'side.jpg'}
    env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parents[1] / 'templates'))
    env.globals['side_white_view'] = side_white_view
    html = env.get_template('_side_white_result.html').render(detail={'side_white_result': row})
    parser = Images()
    parser.feed(html)
    assert len(parser.crops) == 2
    assert all('point=1' in item['src'] and item['loading'] == 'lazy' for item in parser.crops)
    assert '未保存OMIT影像' in html
    assert '64×64 原圖 px' in html


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
    assert np.array_equal(overview[y, x + 16], annotation_color("aoi"))
    crop = render_comparison_image(row, comparison, tmp_path, point_id=1)
    centre = crop.shape[0] // 2
    assert np.all(crop[centre - 16:centre + 16, centre - 16:centre + 16] == 80)
    assert crop.shape[0] == crop.shape[1] == point['crop_size']
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
    assert "kind=side_clean" in html and "無標記圖 ⤢" in html
    html = template.render(detail={"side_white_result": row}, debug_result=True,
                           artifact_urls={"side": "/debug/side.jpg", "side_clean": "/debug/clean.jpg"})
    assert 'href="/debug/clean.jpg"' in html
