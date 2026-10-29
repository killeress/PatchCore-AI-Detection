"""OMIT evidence, point-level suggestions, and preservation of uncovered defects."""
import json

import cv2
import numpy as np
import pytest

from capi_config import CAPIConfig, normalize_side_white_params
from capi_side_report import compare_side_record, crop_original_image, render_comparison_image
from capi_side_white import _candidate_evidence, inspect_side_white_folder, load_omit_evidence


def observed_row(omit, *, mode="suppress", threshold=.8, info=None, detector=None):
    side = np.full((100, 100), 100, np.uint8)
    side[20:40, 20:60] = 10
    contour = [[20, 20], [59, 20], [59, 39], [20, 39]]
    candidate = {"id": 1, "side_bbox": [20, 20, 40, 20], "side_contour": contour,
                 "front_contour": contour, "_mask": np.ones((20, 40), np.uint8), "bomb_ids": []}
    params = normalize_side_white_params({"dust_mode": mode, "dust_overlap_ratio": threshold, "mapping_margin_px": 0})
    info = info or {"status": "available", "coordinate_space": "side_detection_pixels", "size": [100, 100]}
    _candidate_evidence(candidate, side, side, omit, detector or (lambda crop: (True, crop, 1, "surface")),
                        info, np.eye(3), np.zeros_like(side), np.full(side.shape, 10, np.float32), params)
    return {"status": "FILTERED" if candidate["disposition"] != "retained" else "CANDIDATES",
            "resolution_x": 100, "resolution_y": 100,
            "aoi_machine_coords": {"W0F00000": [{"defect_code": "CSJBL", "product_x": 40, "product_y": 30}]},
            "payload": {"parameters": params, "omit": info, "side_size": [100, 100], "front_size": [100, 100],
                        "mapping": {"status": "estimated", "matrix": np.eye(3).tolist(),
                                    "front_polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]},
                        "candidates": [candidate]}}


def full_omit():
    omit = np.zeros((100, 100), np.uint8)
    omit[20:40, 20:60] = 255
    return omit


@pytest.mark.parametrize("mode", ["observe", "suppress", "off"])
def test_surface_filtering_stays_enabled_with_legacy_mode_values(mode):
    row = observed_row(full_omit(), mode=mode)
    result = compare_side_record(row)
    point = result["points"][0]
    assert row["payload"]["parameters"]["dust_mode"] == "suppress"
    assert point["status"] == "omit_suppressed" and point["omit"]["status"] == "suppressed"
    assert result["omit_suppressed"] == 1
    evidence = point["omit"]["candidates"][0]
    assert evidence["overlap_ratio"] == evidence["mask_overlap_ratio"] == 1
    assert evidence["compared_area_px"] == evidence["overlap_area_px"] == 800
    assert evidence["uncovered_area_px"] == 0
    json.dumps(row)


@pytest.mark.parametrize("x,expected", [(40, "omit_suppressed"), (50, "matched")])
def test_partial_omit_only_blocks_the_aoi_point_on_uncovered_pixels(x, expected):
    omit = full_omit()
    omit[:, 45:] = 0
    row = observed_row(omit, threshold=.5)
    row["aoi_machine_coords"]["W0F00000"][0]["product_x"] = x
    point = compare_side_record(row)["points"][0]
    assert point["status"] == expected and point["omit"]["status"] == "partial"
    evidence = point["omit"]["candidates"][0]
    assert evidence["overlap_ratio"] == .625
    assert evidence["remaining_area_px"] == evidence["uncovered_area_px"] == 300


def test_omit_mask_hole_and_report_tolerance_do_not_create_ok_suggestions():
    omit = full_omit()
    omit[29:32, 39:42] = 0  # Smaller than min_area; still no evidence at the AOI centre.
    row = observed_row(omit)
    assert row["payload"]["candidates"][0]["disposition"] == "dust_suppressed"
    assert compare_side_record(row)["omit_suppressed"] == 0
    row["aoi_machine_coords"]["W0F00000"][0].update(product_x=25, product_y=25)
    assert compare_side_record(row)["omit_suppressed"] == 1
    row["aoi_machine_coords"]["W0F00000"][0].update(product_x=65, product_y=30)
    assert compare_side_record(row, 50)["omit_suppressed"] == 0


@pytest.mark.parametrize("invalid", ["missing", "size", "exposure", "bad_mask", "failed", "truncated", "legacy", "bomb"])
def test_unreliable_omit_evidence_never_suggests_ok(invalid):
    if invalid == "missing":
        row = observed_row(None)
    elif invalid == "size":
        row = observed_row(np.zeros((50, 50), np.uint8))
    elif invalid == "exposure":
        row = observed_row(full_omit(), info={"status": "unavailable", "reason": "OMIT 過曝"})
        assert row["payload"]["candidates"][0]["disposition"] == "retained"
    elif invalid == "bad_mask":
        row = observed_row(full_omit(), detector=lambda crop: (True, np.full(crop.shape, np.nan), 1, "invalid"))
    else:
        row = observed_row(full_omit())
        if invalid == "failed":
            row["status"] = "ERROR"
        elif invalid == "truncated":
            row["payload"]["truncated"] = True
        elif invalid == "legacy":
            row["payload"]["candidates"][0]["dust"].pop("suppressed_regions")
        else:
            row["payload"]["candidates"][0]["bomb_ids"] = [1]
    assert compare_side_record(row)["omit_suppressed"] == 0


@pytest.mark.parametrize("rotated", [False, True])
def test_real_omit_detector_clears_supported_black_patch_but_keeps_other_defect(tmp_path, rotated):
    front = np.zeros((1000, 1400), np.uint8)
    cv2.rectangle(front, (50, 50), (1350, 950), 100, -1)
    cv2.rectangle(front, (585, 392), (615, 408), 8, -1)
    cv2.circle(front, (740, 400), 7, 40, -1)
    for name in ("W0F00000_120000.tif", "SW0F00000_120000.tif"):
        cv2.imwrite(str(tmp_path / name), front)
    omit = np.zeros_like(front)
    cv2.rectangle(omit, (585, 392), (615, 408), 235, -1)
    cv2.imwrite(str(tmp_path / "SPINIGBI _120000.tif"), omit)
    points = [{"defect_code": "CSJBL", "image_prefix": "W0F00000", "coordinate_space": "image",
               "source_x": x, "source_y": 400} for x in (600, 740)]
    context = {"config": CAPIConfig(), "product_resolution": [1300, 900], "aoi_csjbl_points": points}
    payload = inspect_side_white_folder(tmp_path, tmp_path / "output", context=context, rotate_180=rotated,
                                        params={"dust_mode": "suppress", "mapping_margin_px": 0})
    assert payload["status"] == "CANDIDATES", payload.get("reason")
    assert payload["omit"]["status"] == "available"
    row = {"status": payload["status"], "payload": payload, "aoi_machine_coords": {"W0F00000": points}}
    comparison = compare_side_record(row, tolerance=20)
    assert comparison["points"][0]["status"] == "omit_suppressed", comparison
    evidence = comparison["points"][0]["omit"]["candidates"][0]
    assert evidence["match_basis"] == "dark_core" and evidence["halo_removed_area_px"] > 0
    assert comparison["points"][1]["status"] == "matched", comparison
    assert comparison["omit_suppressed"] == comparison["matched"] == 1
    assert "omit" in payload["artifacts"]
    assert payload["preview_bounds"]["omit"] == payload["preview_bounds"]["side"]
    crop = crop_original_image(row, comparison, tmp_path, "omit", 1)
    assert crop.dtype == omit.dtype and crop.max() == 235
    preview = render_comparison_image(row, comparison, tmp_path, "omit", 1)
    assert preview.size and np.max(np.abs(preview[:, :, 0].astype(int) - preview[:, :, 1])) == 0
    json.dumps(payload)


def test_single_stale_omit_is_not_reused_for_multiple_side_acquisitions(tmp_path):
    side = tmp_path / "SW0F00000_120000.tif"
    for path in (side, tmp_path / "SW0F00000_120010.tif", tmp_path / "SPINIGBI _120001.tif"):
        cv2.imwrite(str(path), np.zeros((100, 100), np.uint8))
    raw, detector, info = load_omit_evidence(tmp_path, side, {"config": CAPIConfig()}, False)
    assert raw is None and detector is None and info["status"] == "unavailable"
    assert "多次" in info["reason"]


def halo_candidate(*, mode="suppress", defect=None, edge=False):
    side = np.full((1000, 1200), 100, np.uint8)
    cv2.rectangle(side, (500, 450), (700, 520), 8, -1)
    if defect is not None:
        cv2.circle(side, (680, 425), 5, 220 if defect == "bright" else 10, -1)
    source = side.astype(np.float32)
    residual = cv2.GaussianBlur(source, (0, 0), 2) - cv2.GaussianBlur(source, (0, 0), 24)
    params = normalize_side_white_params({"dust_mode": mode, "mapping_margin_px": 3})
    threshold = params["min_contrast_gray"]
    support = cv2.morphologyEx((np.abs(residual) > threshold).astype(np.uint8),
                               cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(support, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1  # The real defect and sticker halo form one candidate.
    x, y, w, h = cv2.boundingRect(contours[0])
    candidate = {"id": 1, "side_bbox": [x, y, w, h], "side_contour": contours[0].reshape(-1, 2).tolist(),
                 "_mask": support[y:y+h, x:x+w], "bomb_ids": []}
    omit = np.zeros_like(side)
    cv2.rectangle(omit, (500, 450), (700, 520), 255, -1)
    quad = np.array([[0, 350 if edge else 0], [1199, 350 if edge else 0], [1199, 999], [0, 999]], np.float32)
    _candidate_evidence(candidate, side, None, omit, lambda crop: (True, crop, 1, "surface"),
                        {"status": "available"}, None, np.zeros_like(side), residual, params,
                        detection_threshold=threshold, panel_quad=quad)
    return candidate


@pytest.mark.parametrize("mode", ["observe", "suppress", "off"])
def test_supported_dark_core_can_clear_its_halo_without_lowering_overlap_threshold(mode):
    candidate = halo_candidate(mode=mode)
    dust = candidate["dust"]
    assert dust["overlap_ratio"] < .8 and dust["core_overlap_ratio"] == 1
    assert dust["match_basis"] == "dark_core"
    assert dust["halo_removed_area_px"] > 1000
    assert dust["status"] == "suppressed"
    assert candidate["disposition"] == "dust_suppressed"
    from capi_side_report import _inside_omit_regions
    assert _inside_omit_regions([600, 480], dust["suppressed_regions"])
    assert not _inside_omit_regions([600, 430], dust["suppressed_regions"])


@pytest.mark.parametrize("defect", ["bright", "dark"])
def test_omit_halo_correction_preserves_real_defect_connected_to_same_candidate(defect):
    candidate = halo_candidate(defect=defect)
    assert candidate["dust"]["core_overlap_ratio"] >= .8
    assert candidate["dust"]["halo_removed_area_px"] > 1000
    assert candidate["disposition"] == "retained"
    assert any(cv2.pointPolygonTest(np.asarray(contour, np.float32), (680., 425.), False) >= 0
               for contour in candidate["remaining_contours"])


def test_omit_halo_correction_is_not_applied_near_panel_edge():
    candidate = halo_candidate(edge=True)
    assert candidate["dust"]["core_overlap_ratio"] == 1
    assert candidate["dust"]["halo_removed_area_px"] == 0
    assert candidate["disposition"] == "retained"
