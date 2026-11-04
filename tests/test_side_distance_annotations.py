"""Distance rulers identify retained regions in native and fallback point crops."""
import copy
import io
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from jinja2 import Environment, FileSystemLoader

from capi_side_report import (annotation_color, compare_side_record, crop_original_image,
                              render_comparison_image, side_white_view)
from test_side_release_guard import guard_row


def distance_row(rotated=False):
    row = guard_row(5.33, scale=2)
    row["payload"].update(rotation_applied=rotated, side_image="side.tif", front_image="front.tif", artifacts={},
                          side_size=[1200, 800], front_size=[1200, 800],
                          omit={"image": "omit.tif", "coordinate_space": "side_detection_pixels",
                                "size": [1200, 800]})
    return row


def test_distance_endpoint_is_on_the_nearest_remaining_edge_in_product_space():
    row = distance_row()
    payload = row["payload"]
    payload["front_size"] = [1500, 600]
    payload["side_size"] = [1000, 900]
    payload["mapping"].update(matrix=np.diag([1.5, 2 / 3, 1]).tolist(),
        front_polygon=[[0, 0], [1500, 0], [1500, 600], [0, 600]])
    product = np.array([[461, 90], [451, 110], [471, 120], [481, 100]], np.float32)
    candidate = payload["candidates"][0]
    candidate["side_contour"] = (product * [2, 3]).tolist()
    candidate["remaining_contours"] = [candidate["side_contour"]]
    before = copy.deepcopy(row)
    point = compare_side_record(row)["points"][0]
    distance = point["retained_candidate_distances"][0]
    assert point["status"] == "guarded"
    assert distance["distance_product_px"] == pytest.approx(np.sqrt(28.8), abs=.0001)
    assert distance["nearest_product_xy"] == pytest.approx([454.8, 102.4], abs=.0001)
    assert distance["nearest_side_xy"] == pytest.approx([909.6, 307.2], abs=.0001)
    assert distance["nearest_front_xy"] == pytest.approx([1364.4, 204.8], abs=.0001)
    assert row == before


def test_ruler_uses_only_retained_contours_and_reports_zero_inside():
    row = distance_row()
    candidate = row["payload"]["candidates"][0]
    # The old support covers the AOI, but only its right-hand remainder survives.
    candidate["side_contour"] = [[880, 180], [940, 180], [940, 220], [880, 220]]
    point = compare_side_record(row)["points"][0]
    distance = point["retained_candidate_distances"][0]
    assert distance["distance_product_px"] == pytest.approx(5.33, abs=.0001)
    assert distance["nearest_side_xy"] == pytest.approx([910.66, 200], abs=.0001)
    row["aoi_machine_coords"]["W0F00000"][0]["product_x"] = 460
    point = compare_side_record(row)["points"][0]
    distance = point["retained_candidate_distances"][0]
    assert distance["distance_product_px"] == 0
    assert distance["nearest_side_xy"] == point["side_xy"]
    candidate.update(disposition="dust_suppressed", remaining_contours=[])
    assert compare_side_record(row)["points"][0]["retained_candidate_distances"] == []


@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("kind", ["side", "front", "omit"])
@pytest.mark.parametrize("depth", [np.uint8, np.uint16])
def test_native_crop_draws_candidates_and_distance_at_file_coordinates(tmp_path, monkeypatch, rotated, kind, depth):
    row = distance_row(rotated)
    original = np.full((800, 1200), 90 if depth == np.uint8 else 30000, depth)
    file = tmp_path / (kind + ".tif")
    assert cv2.imwrite(str(file), original)
    saved = file.read_bytes()
    comparison = compare_side_record(row)
    point = comparison["points"][0]
    crop = crop_original_image(row, comparison, tmp_path, kind, 1)
    lines = []
    draw_line = cv2.line
    factor = 1 if depth == np.uint8 else 257
    blue = tuple(v * factor for v in annotation_color("distance"))

    def capture(image, start, end, color, *args):
        if color == blue:
            lines.append((start, end))
        return draw_line(image, start, end, color, *args)

    monkeypatch.setattr(cv2, "line", capture)
    marked = crop_original_image(row, comparison, tmp_path, kind, 1, candidates=True)
    assert marked.shape[:2] == crop.shape and marked.dtype == depth
    assert np.any(marked[:, :, 0] != marked[:, :, 1])
    centre = point["crop_size"] // 2
    # Ruler ends 10.66 camera pixels to the right; a rotated file reverses it.
    end = (centre + (-11 if rotated else 11), centre)
    assert lines[0][1] == end
    assert np.array_equal(marked[centre, centre], [original[0, 0]] * 3)
    assert file.read_bytes() == saved
    assert np.array_equal(crop, crop_original_image(row, comparison, tmp_path, kind, 1))


@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("kind,scale", [("front", .25), ("side", .5), ("omit", .75)])
def test_preview_fallback_marks_the_same_region_in_saved_orientation(tmp_path, monkeypatch, rotated, kind, scale):
    row = distance_row(rotated)
    payload = row["payload"]
    artifact = kind if kind == "omit" else kind + "_clean"
    path = tmp_path / (artifact + ".jpg")
    assert cv2.imwrite(str(path), np.full((round(800 * scale), round(1200 * scale), 3), 90, np.uint8))
    payload["artifacts"][artifact] = str(path)
    payload.setdefault("preview_bounds", {})[artifact] = [0, 0, 1200, 800]
    comparison = compare_side_record(row)
    clean = render_comparison_image(row, comparison, tmp_path, kind, 1)
    lines = []
    draw_line = cv2.line

    def capture(image, start, end, color, *args):
        if color == annotation_color("distance"):
            lines.append((start, end))
        return draw_line(image, start, end, color, *args)

    monkeypatch.setattr(cv2, "line", capture)
    marked = render_comparison_image(row, comparison, tmp_path, kind, 1, candidates=True)
    assert marked.shape == clean.shape
    assert lines[0][1][0] > lines[0][0][0]  # Saved previews use detection orientation.
    assert np.any(marked[:, :, 0] != marked[:, :, 1])
    assert np.all(clean == 90)


def test_point_crop_links_supply_marked_and_clean_images_with_candidate_ids():
    row = distance_row()
    row["payload"]["artifacts"] = {kind: kind + ".jpg" for kind in ("side", "front", "omit")}
    env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parents[1] / "templates"))
    env.globals["side_white_view"] = side_white_view
    html = env.get_template("_side_white_result.html").render(detail={"side_white_result": row})
    assert html.count('data-sw-clean-url=') == html.count('data-sw-marked-url=') == 3
    assert 'data-sw-candidates-toggle aria-pressed="true"' in html
    assert "異常區域候選 · #1" in html and "5.33 產品 px" in html
    assert "藍色直線" in html and "圓圈" not in html


@pytest.mark.parametrize("original_available", [False, True])
def test_image_route_switches_between_candidates_and_clean_pixels(tmp_path, original_available):
    from capi_web import CAPIWebHandler

    row = distance_row(rotated=True)
    row["image_dir"] = str(tmp_path)
    original = np.full((800, 1200), 90, np.uint8)
    if original_available:
        assert cv2.imwrite(str(tmp_path / "side.tif"), original)
    preview = tmp_path / "side_clean.jpg"
    assert cv2.imwrite(str(preview), original)
    payload = row["payload"]
    payload["artifacts"]["side_clean"] = str(preview)
    payload["preview_bounds"] = {"side_clean": [0, 0, 1200, 800]}
    before = copy.deepcopy(row)
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.db = SimpleNamespace(get_side_statistics_record=lambda result_id: row)
    handler.heatmap_base_dir = str(tmp_path)
    handler.send_response = lambda status: None
    headers = {}
    handler.send_header = lambda name, value: headers.update({name: value})
    handler.end_headers = lambda: None
    handler._send_404 = lambda: pytest.fail("Valid candidate crop returned 404")
    images = []
    handler._send_image_array_png = images.append
    results = []
    for candidates in ("0", "1"):
        handler.wfile = io.BytesIO()
        handler._handle_side_statistics_image({"id": ["1"], "kind": ["side"], "point": ["1"],
                                               "candidates": [candidates]})
        results.append(cv2.imdecode(np.frombuffer(handler.wfile.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
                       if original_available else images[-1])
    assert np.all(results[0] == 90)
    assert np.any(results[1][:, :, 0] != results[1][:, :, 1])
    if original_available:
        assert headers["X-Side-Image-Source"] == "original"
    assert row == before
