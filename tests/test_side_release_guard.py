"""An unmatched AOI must not release an unexcluded nearby candidate."""
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from jinja2 import Environment, FileSystemLoader

from capi_config import normalize_side_white_params
from capi_side_report import (annotation_color, compare_side_record, draw_side_annotations,
                              remaining_csjbl_records, side_inference_decision, side_white_view)
from capi_side_white import _candidate_comparison, _candidate_preview_scope
from test_side_conversion_statistics import sample


def guard_row(distance=9.521440455604399, *, scale=1):
    row = sample(("unknown",))
    payload = row["payload"]
    row["status"] = payload["status"] = "CANDIDATES"
    payload["parameters"] = normalize_side_white_params()
    payload["evidence_context"]["product_resolution"] = [500, 300]
    payload["evidence_context"]["inference_marks"] = []
    payload["side_size"] = payload["front_size"] = [500 * scale, 300 * scale]
    payload["mapping"]["front_polygon"] = [[0, 0], [500 * scale, 0], [500 * scale, 300 * scale], [0, 300 * scale]]
    x = 450 + distance
    contour = (np.array([[x, 90], [x + 10, 90], [x + 10, 110], [x, 110]]) * scale).tolist()
    payload["candidates"] = [{"id": 1, "side_contour": contour, "front_contour": contour,
                               "remaining_contours": [contour], "disposition": "retained",
                               "crop_bounds": {"side": [420 * scale, 60 * scale, 500 * scale, 140 * scale],
                                               "front": [420 * scale, 60 * scale, 500 * scale, 140 * scale]}}]
    return row


@pytest.mark.parametrize("preview", [0, 50, 100])
@pytest.mark.parametrize("distance,status,judgment", [
    (0, "matched", "NG"), (5, "matched", "NG"), (5.1, "guarded", "NG"),
    (9.521440455604399, "guarded", "NG"), (10, "guarded", "NG"),
    (10.1, "no_anomaly", "OK-s"), (40, "no_anomaly", "OK-s"),
])
def test_guard_boundaries_and_release_do_not_depend_on_preview_radius(preview, distance, status, judgment):
    row = guard_row(distance)
    row["payload"]["parameters"]["csjbl_preview_radius_product_px"] = preview
    before = copy.deepcopy(row)
    decision = side_inference_decision(row)
    point = decision["points"][0]
    assert point["status"] == status
    assert decision["final_judgment"] == judgment
    assert decision["match_tolerance_product_px"] == 5
    assert decision["release_guard_radius_product_px"] == 10
    assert point["guard_candidate_ids"] == ([1] if status == "guarded" else [])
    records = ["CSJBL0045000100W0F00000"]
    assert remaining_csjbl_records(records, decision["points"]) == ([] if judgment == "OK-s" else records)
    if status == "guarded":
        assert "位置待確認" in decision["reason"]
        assert point["nearest_retained_candidate_id"] == 1
        assert point["nearest_retained_distance_product_px"] == pytest.approx(distance, abs=.0001)
    assert row == before


def test_guard_uses_product_distance_and_snapshot_survives_json():
    row = guard_row(scale=2)
    decision = json.loads(json.dumps(side_inference_decision(row)))
    point = decision["points"][0]
    assert point["status"] == "guarded"
    assert point["nearest_retained_distance_product_px"] == pytest.approx(9.52144, abs=.0001)
    assert point["retained_candidate_distances"][0]["candidate_id"] == 1


@pytest.mark.parametrize("disposition", ["excluded", "mark_suppressed", "dust_suppressed"])
def test_excluded_candidate_does_not_block_release(disposition):
    row = guard_row()
    row["payload"]["candidates"][0].update(disposition=disposition, remaining_contours=[])
    assert side_inference_decision(row)["final_judgment"] == "OK-s"


def test_only_remaining_contours_can_block_release():
    row = guard_row()
    candidate = row["payload"]["candidates"][0]
    candidate["side_contour"] = [[440, 80], [480, 80], [480, 120], [440, 120]]
    point = compare_side_record(row)["points"][0]
    assert point["aoi_candidate_ids"] == [1]  # The original support intersects the point.
    assert point["retained_candidate_ids"] == []
    assert point["guard_candidate_ids"] == [1]
    assert point["status"] == "guarded"


def test_mark_at_aoi_cannot_release_a_separate_guard_candidate():
    row = guard_row()
    row["payload"]["evidence_context"]["inference_marks"] = [
        {"source_image": "W0F00000_120000.tif", "name": "mark_binary", "front_bbox": [445, 95, 10, 10]}]
    decision = side_inference_decision(row)
    assert decision["points"][0]["status"] == "guarded"
    assert decision["final_judgment"] == "NG"


def test_old_snapshot_is_not_silently_reclassified():
    row = guard_row()
    row["payload"]["parameters"].pop("csjbl_release_guard_radius_product_px")
    before = copy.deepcopy(row)
    assert side_inference_decision(row)["final_judgment"] == "OK-s"
    assert not compare_side_record(row)["release_guard_enabled"]
    assert row == before


@pytest.mark.parametrize("value", [-1, 501, 1.5, True, float("nan"), 4])
def test_invalid_guard_radius_is_rejected(value):
    with pytest.raises(ValueError, match="放行保護範圍"):
        normalize_side_white_params({"csjbl_release_guard_radius_product_px": value})


def test_new_default_is_compatible_with_existing_larger_match_tolerance():
    assert normalize_side_white_params()["csjbl_release_guard_radius_product_px"] == 10
    assert normalize_side_white_params({"csjbl_match_tolerance_product_px": 20})["csjbl_release_guard_radius_product_px"] == 20


def test_guard_candidate_has_consistent_labels_counts_and_distance():
    row = guard_row()
    scope = _candidate_preview_scope({**row["payload"], "evidence_context": {
        **row["payload"]["evidence_context"], "aoi_csjbl_points": [
            {**row["aoi_machine_coords"]["W0F00000"][0], "image_prefix": "W0F00000"}]}})
    assert scope["candidate_ids"] == []
    assert scope["guard_candidate_ids"] == [1] and scope["guard_total"] == 1
    assert scope["reference_total"] == 0
    view = side_white_view(row)
    candidate = view["candidates"][0]
    assert candidate["guarded"] and not candidate["aoi_hit"]
    assert "保留 NG" in candidate["label"]
    assert candidate["aoi_distances"][0]["distance"] == pytest.approx(9.52144, abs=.0001)
    assert view["guard_candidates"] == {"nearby": 1, "total": 1}
    assert view["reference_candidates"] == {"nearby": 0, "total": 0}


def test_main_and_candidate_comparison_draw_guard_aoi_and_distance(monkeypatch):
    row = guard_row()
    payload = row["payload"]
    comparison = compare_side_record(row)
    texts = []
    put_text = cv2.putText
    def capture(image, text, *args):
        texts.append(text)
        return put_text(image, text, *args)
    monkeypatch.setattr(cv2, "putText", capture)
    source = np.full((300, 500), 140, np.uint8)
    before = source.copy()
    composite = _candidate_comparison(payload["candidates"][0], source, source, None,
                                       payload=payload, comparison=comparison)
    assert composite.shape == (382, 1600, 3)
    assert "A1" in texts
    assert any("A1 to #1: 9.52 product px" in t and "POSITION CHECK - NG" in t for t in texts)
    assert np.array_equal(source, before)
    for index in range(2):
        panel = composite[:350, index * 400:(index + 1) * 400]
        for key in ("aoi", "distance", "guard"):
            assert np.any(np.max(np.abs(panel.astype(int) - annotation_color(key)), axis=2) < 20), key
    image = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    draw_side_annotations(image, payload, comparison, "side", [0, 0, 500, 300])
    assert np.any(np.max(np.abs(image.astype(int) - annotation_color("guard")), axis=2) < 20)
    assert "9.52 px" in texts
    assert np.array_equal(image[100, 440], [140, 140, 140])  # Former outer circle stays clear.
    assert not np.any(np.max(np.abs(image.astype(int) - annotation_color("candidate")), axis=2) < 20)
    env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parents[1] / "templates"))
    env.globals["side_white_view"] = side_white_view
    html = env.get_template("_side_white_result.html").render(detail={"side_white_result": row})
    assert "放行保護範圍：10.0 產品 px" in html and "位置待確認" in html
    assert "最近保留候選 #1：9.52 產品 px" in html
    assert "與 AOI 距離（產品 px）" in html


def test_inspector_saves_guard_evidence_before_formal_decision(tmp_path, monkeypatch):
    import capi_side_white as sw

    source = np.zeros((500, 700), np.uint8)
    cv2.rectangle(source, (50, 50), (650, 450), 90, -1)
    front, side = tmp_path / "W0F00000_144217.tif", tmp_path / "SW0F00000_144217.tif"
    cv2.imwrite(str(front), source)
    cv2.imwrite(str(side), source)
    quad = np.float32([[50, 50], [650, 50], [650, 450], [50, 450]])
    def locate(image, **kwargs):
        if "localization" in kwargs:
            kwargs["localization"]["boundary"] = quad.tolist()
        return quad
    def detect(image, panel, params, **kwargs):
        contour = [[345, 245], [354, 245], [354, 254], [345, 254]]
        return [{"id": 1, "side_xy": [349.5, 249.5], "side_bbox": [345, 245, 10, 10],
                 "side_contour": contour, "area_px": 100, "_mask": np.ones((10, 10), np.uint8)}], np.full(image.shape, -10, np.float32), [], 2.2, False
    monkeypatch.setattr(sw, "detect_panel_quad", locate)
    monkeypatch.setattr(sw, "_candidates", detect)
    point = {"defect_code": "CSJBL", "image_prefix": "W0F00000", "product_x": 285.47856, "product_y": 200}
    payload = sw.inspect_side_white_image(side, front, tmp_path / "out",
        context={"product_resolution": [600, 400], "aoi_csjbl_points": [point]})
    assert payload["status"] == "CANDIDATES", payload.get("reason")
    assert payload["parameters"]["csjbl_release_guard_radius_product_px"] == 10
    assert payload["candidate_display"]["guard_candidate_ids"] == [1]
    assert payload["candidate_comparison_version"] == 3
    assert payload["preview_annotation_version"] == 4
    assert Path(payload["artifacts"]["candidate_1"]).is_file()
    decision = side_inference_decision({"payload": payload, "status": payload["status"], "ai_judgment": "NG",
        "ng_details": [{**point, "image": "W0F00000", "type": "aoi_passthrough"}]})
    assert decision["final_judgment"] == "NG"
    assert decision["points"][0]["status"] == "guarded"
    assert decision["points"][0]["nearest_retained_distance_product_px"] == pytest.approx(9.52144, abs=.0001)
    json.dumps(payload)  # Deferred rendering must not leak masks/arrays into saved evidence.
