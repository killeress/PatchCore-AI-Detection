import cv2
import numpy as np
from pathlib import Path

from capi_web import (
    _candidate_dust_overlap,
    _detect_dot_components,
    _detect_dot_components_auto,
    _detect_dot_components_debug_polarity,
    _detect_white_halo_components,
    _evaluate_within_spec_suggestion,
    _evaluate_within_spec_suggestion_detail,
    _format_within_spec_inference_note,
    _format_within_spec_panel_summary,
    _within_spec_auto_visual_output,
)


def test_candidate_dust_overlap_uses_component_pixels_instead_of_whole_bbox():
    candidate = {
        "x": 0,
        "y": 0,
        "w": 10,
        "h": 10,
        "center_x": 5,
        "center_y": 5,
    }
    candidate_mask = np.zeros((10, 10), dtype=np.uint8)
    candidate_mask[:, :2] = 255
    dust_mask = candidate_mask > 0

    rejected = _candidate_dust_overlap(candidate, dust_mask, candidate_mask)

    assert rejected is not None
    assert rejected["dust_overlap_ratio"] == 1.0
    assert rejected["dust_overlap_basis"] == "component"


def _rules(*, screen_limit=1, tile_limit=1, threshold_mm=0.3, white_enabled=False, segmentation_method="background_diff"):
    return {
        "default": {
            "dot_detection": {
                "preprocess_method": "gaussian",
                "preprocess_params": {"kernel_size": 7, "sigma": 1.0},
                "segmentation_method": segmentation_method,
                "diff_threshold": 8,
                "hysteresis_low_threshold": 2,
                "hysteresis_high_threshold": 4,
                "hysteresis_edge_width_percent": 3.0,
                "hysteresis_edge_extra_threshold": 2,
                "hysteresis_second_low_threshold": 3,
                "hysteresis_second_high_threshold": 4,
                "hysteresis_second_edge_width_percent": 9.5,
                "hysteresis_second_edge_extra_threshold": 2,
                "hysteresis_switch_count_threshold": 5,
                "hysteresis_second_max_count": 5,
                "hysteresis_edge_suppress_percent": 0.0,
                "background_kernel": 31,
                "min_area_px": 2,
                "max_area_px": 1000,
                "morph_open": 1,
                "size_metric": "bbox_max",
                "unit_per_px": 0.02,
            },
            "screens": {
                "W0F00000": {
                    "black_dot": {
                        "enabled": True,
                        "defect_code": "C1111",
                        "area_threshold_mm": threshold_mm,
                        "screen_count_limit": screen_limit,
                        "tile_count_threshold": tile_limit,
                    },
                    "white_dot": {
                        "enabled": white_enabled,
                        "defect_code": "W1111",
                        "area_threshold_mm": threshold_mm,
                        "screen_count_limit": screen_limit,
                        "tile_count_threshold": tile_limit,
                    },
                }
            },
        }
    }


def _detail(image_path, *, defect_code="OTHER", tile_size=96, is_bomb=0, model_id=""):
    return {
        "model_id": model_id,
        "machine_no": "UNKNOWN",
        "images": [
            {
                "image_name": "W0F00000.png",
                "image_path": str(image_path),
                "tiles": [
                    {
                        "x": 0,
                        "y": 0,
                        "width": tile_size,
                        "height": tile_size,
                        "is_anomaly": 1,
                        "is_dust": 0,
                        "is_bomb": is_bomb,
                        "is_exclude_zone": 0,
                        "scratch_filtered": 0,
                        "is_aoi_coord": 1,
                        "aoi_defect_code": defect_code,
                    }
                ],
            }
        ],
    }


def _write_black_dot_image(path, centers):
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    for center in centers:
        cv2.circle(image, center, 3, (60, 60, 60), -1)
    cv2.imwrite(str(path), image)


def _write_white_dot_image(path, centers):
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    for center in centers:
        cv2.circle(image, center, 3, (210, 210, 210), -1)
    cv2.imwrite(str(path), image)


def test_within_spec_suggestion_does_not_filter_by_defect_code(tmp_path):
    image_path = tmp_path / "W0F00000.png"
    _write_black_dot_image(image_path, [(48, 48)])

    suggestion = _evaluate_within_spec_suggestion(
        _detail(image_path, defect_code="P9999"),
        _rules(),
    )

    assert suggestion is not None
    assert suggestion["suggested"] is True
    assert suggestion["category"] == "within_spec"
    assert suggestion["matches"][0]["aoi_defect_codes"] == ["P9999"]


def test_within_spec_suggestion_accepts_counts_equal_to_limits(tmp_path):
    image_path = tmp_path / "W0F00000.png"
    _write_black_dot_image(image_path, [(32, 48), (64, 48)])

    suggestion = _evaluate_within_spec_suggestion(
        _detail(image_path),
        _rules(screen_limit=2, tile_limit=2),
    )

    assert suggestion is not None
    match = suggestion["matches"][0]
    assert match["screen_count"] == 2
    assert match["screen_count_limit"] == 2
    assert match["max_tile_count"] == 2
    assert match["tile_count_threshold"] == 2


def test_within_spec_suggestion_rejects_counts_over_limits(tmp_path):
    image_path = tmp_path / "W0F00000.png"
    _write_black_dot_image(image_path, [(32, 48), (64, 48)])

    suggestion = _evaluate_within_spec_suggestion(
        _detail(image_path),
        _rules(screen_limit=1, tile_limit=2),
    )

    assert suggestion is None


def test_within_spec_suggestion_rotates_source_before_using_inference_tile_coordinates(tmp_path):
    image_path = tmp_path / "W0F00000.png"
    image = np.full((96, 192, 3), 128, dtype=np.uint8)
    cv2.circle(image, (144, 48), 3, (60, 60, 60), -1)
    assert cv2.imwrite(str(image_path), image)

    unrotated = _evaluate_within_spec_suggestion_detail(
        _detail(image_path),
        _rules(),
        rotate_180=False,
    )
    rotated = _evaluate_within_spec_suggestion_detail(
        _detail(image_path),
        _rules(),
        rotate_180=True,
    )

    assert unrotated["suggestion"] is None
    assert rotated["suggestion"] is not None
    assert rotated["panel_summary"]["total_dot_count"] == 1
    assert rotated["parameter_snapshot"]["input_rotation_degrees"] == 180


def test_within_spec_suggestion_uses_model_rule_before_default(tmp_path):
    image_path = tmp_path / "W0F00000.png"
    _write_black_dot_image(image_path, [(48, 48)])
    rules = _rules(threshold_mm=0.001)
    rules["MODEL_A"] = _rules(threshold_mm=0.3)["default"]

    detail = _evaluate_within_spec_suggestion_detail(
        _detail(image_path, model_id="MODEL_A"),
        rules,
    )

    assert detail["suggestion"] is not None
    assert detail["rule_selection"]["matched_machine_key"] == "MODEL_A"
    assert detail["rule_selection"]["fallback_used"] is False


def test_within_spec_suggestion_uses_requested_machine_when_model_missing(tmp_path):
    image_path = tmp_path / "W0F00000.png"
    _write_black_dot_image(image_path, [(48, 48)])
    rules = _rules(threshold_mm=0.001)
    rules["CAPI0703"] = _rules(threshold_mm=0.3)["default"]

    detail = _evaluate_within_spec_suggestion_detail(
        _detail(image_path, model_id="UNKNOWN_MODEL"),
        rules,
        machine_id="CAPI0703",
    )

    assert detail["suggestion"] is not None
    assert detail["rule_selection"]["candidate_keys"][:2] == ["UNKNOWN_MODEL", "CAPI0703"]
    assert detail["rule_selection"]["matched_machine_key"] == "CAPI0703"


def test_within_spec_suggestion_falls_back_to_default_when_machine_missing(tmp_path):
    image_path = tmp_path / "W0F00000.png"
    _write_black_dot_image(image_path, [(48, 48)])

    detail = _evaluate_within_spec_suggestion_detail(
        _detail(image_path, model_id="UNKNOWN_MODEL"),
        _rules(),
    )

    assert detail["suggestion"] is not None
    assert detail["rule_selection"]["matched_machine_key"] == "default"
    assert detail["rule_selection"]["fallback_used"] is True


def test_within_spec_suggestion_skips_bomb_tiles(tmp_path):
    image_path = tmp_path / "W0F00000.png"
    _write_black_dot_image(image_path, [(48, 48)])

    detail = _evaluate_within_spec_suggestion_detail(
        _detail(image_path, is_bomb=1),
        _rules(),
    )

    assert detail["suggestion"] is None
    assert detail["skipped_tiles"]["bomb"] == 1


def test_within_spec_suggestion_off_mode_disables_judgment(tmp_path):
    image_path = tmp_path / "W0F00000.png"
    _write_black_dot_image(image_path, [(48, 48)])

    detail = _evaluate_within_spec_suggestion_detail(
        _detail(image_path),
        _rules(segmentation_method="off"),
    )

    assert detail["suggestion"] is None
    assert detail["target_tile_count"] == 0
    assert detail["evaluated_tile_count"] == 0
    assert detail["parameter_snapshot"]["dot_detection"]["effective"]["segmentation_method"] == "off"
    assert any(step["message"] == "規格內判定已關閉，停止判定" for step in detail["steps"])


def test_within_spec_suggestion_classifies_white_dot_tile(tmp_path):
    image_path = tmp_path / "W0F00000.png"
    _write_white_dot_image(image_path, [(48, 48)])

    suggestion = _evaluate_within_spec_suggestion(
        _detail(image_path),
        _rules(white_enabled=True),
    )

    assert suggestion is not None
    assert suggestion["matches"][0]["dot_type"] == "white_dot"


def test_within_spec_zero_white_dot_limit_means_not_allowed(tmp_path):
    image_path = tmp_path / "W0F00000.png"
    _write_white_dot_image(image_path, [(48, 48)])

    rules = _rules(white_enabled=True)
    white_rule = rules["default"]["screens"]["W0F00000"]["white_dot"]
    white_rule["screen_count_limit"] = 0
    white_rule["tile_count_threshold"] = 0

    detail = _evaluate_within_spec_suggestion_detail(
        _detail(image_path),
        rules,
    )

    assert detail["suggestion"] is None
    assert detail["panel_summary"]["total_dot_count"] == 1
    assert detail["panel_totals"][0]["dot_type"] == "white_dot"
    assert detail["panel_totals"][0]["total_count"] == 1
    assert detail["panel_totals"][0]["screen_count_limit"] == 0
    assert detail["panel_totals"][0]["tile_count_threshold"] == 0
    assert detail["panel_totals"][0]["within"] is False
    assert not any(
        step.get("dot_type") == "white_dot"
        and step["message"] == "略過點類規則：門檻或數量設定無效"
        for step in detail["steps"]
    )


def test_within_spec_detail_saves_visuals_and_panel_totals(tmp_path):
    image_path = tmp_path / "W0F00000.png"
    visual_dir = tmp_path / "visuals"
    _write_black_dot_image(image_path, [(48, 48)])

    detail = _evaluate_within_spec_suggestion_detail(
        _detail(image_path),
        _rules(),
        visual_output_dir=visual_dir,
        visual_url_prefix="/heatmaps/test/within_spec",
    )

    assert detail["suggestion"] is not None
    assert detail["panel_summary"]["total_dot_count"] == 1
    assert detail["panel_totals"][0]["dot_type"] == "black_dot"
    assert detail["parameter_snapshot"]["matched_machine_key"] == "default"
    assert detail["parameter_snapshot"]["dot_detection"]["effective"]["diff_threshold"] == 8
    assert detail["parameter_snapshot"]["preprocess"]["params"]["kernel_size"] == 7
    assert detail["parameter_snapshot"]["screen_rules_used"]["W0F00000"]["black_dot"]["area_threshold_mm"] == 0.3
    assert detail["parameter_snapshot"]["full_rules"]["default"]["dot_detection"]["size_metric"] == "bbox_max"
    assert detail["visuals"][0]["urls"]["overlay_url"].startswith("/heatmaps/test/within_spec/")
    assert any(p.name.endswith("_overlay.png") for p in visual_dir.iterdir())


def test_within_spec_rejects_aoi_ng_tile_when_dot_detector_misses(tmp_path):
    image_path = tmp_path / "WGF50500.png"
    visual_dir = tmp_path / "visuals"
    image = np.full((64, 128, 3), 128, dtype=np.uint8)
    cv2.circle(image, (96, 32), 3, (60, 60, 60), -1)
    cv2.imwrite(str(image_path), image)

    rules = _rules(screen_limit=2, tile_limit=1, white_enabled=True)
    rules["default"]["screens"]["WGF50500"] = rules["default"]["screens"].pop("W0F00000")
    detail = {
        "model_id": "GN156HRAAPF0S",
        "images": [
            {
                "image_name": "WGF50500.png",
                "image_path": str(image_path),
                "tiles": [
                    {
                        "tile_id": 0,
                        "x": 0,
                        "y": 0,
                        "width": 64,
                        "height": 64,
                        "is_anomaly": 1,
                        "is_dust": 0,
                        "is_bomb": 0,
                        "is_exclude_zone": 0,
                        "scratch_filtered": 0,
                        "is_aoi_coord": 1,
                        "aoi_defect_code": "C1111",
                        "aoi_product_x": 1444,
                        "aoi_product_y": 724,
                        "aoi_image_x": 32,
                        "aoi_image_y": 32,
                    },
                    {
                        "tile_id": 1,
                        "x": 64,
                        "y": 0,
                        "width": 64,
                        "height": 64,
                        "is_anomaly": 1,
                        "is_dust": 0,
                        "is_bomb": 0,
                        "is_exclude_zone": 0,
                        "scratch_filtered": 0,
                        "is_aoi_coord": 1,
                        "aoi_defect_code": "C1111",
                        "aoi_product_x": 1545,
                        "aoi_product_y": 933,
                        "aoi_image_x": 96,
                        "aoi_image_y": 32,
                    },
                ],
            }
        ],
    }

    result = _evaluate_within_spec_suggestion_detail(
        detail,
        rules,
        visual_output_dir=visual_dir,
        visual_url_prefix="/heatmaps/test/within_spec",
    )

    assert result["suggestion"] is None
    assert result["panel_summary"]["target_tile_count"] == 2
    assert result["panel_summary"]["evaluated_tile_count"] == 1
    assert result["panel_summary"]["total_dot_count"] == 1
    assert result["panel_summary"]["fallback_count"] == 0
    assert result["panel_summary"]["missed_dot_tile_count"] == 1
    assert result["missed_dot_tiles"][0]["screen"] == "WGF50500"
    assert result["missed_dot_tiles"][0]["tile_id"] == 0
    assert result["missed_dot_tiles"][0]["is_aoi_coord"] is True
    assert result["panel_totals"][0]["screen"] == "WGF50500"
    assert result["panel_totals"][0]["total_count"] == 1
    assert result["matches"][0]["screen_count"] == 1
    assert any(
        step["message"] == "tile 點偵測未命中：不符合規格內"
        for step in result["steps"]
    )
    assert any(
        step["message"] == "點偵測未命中：不建議規格內"
        for step in result["steps"]
    )


def test_within_spec_counts_dust_mask_candidate_as_zero_point(tmp_path):
    image_path = tmp_path / "WGF50500.png"
    visual_dir = tmp_path / "visuals"
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    cv2.circle(image, (32, 32), 3, (60, 60, 60), -1)
    cv2.imwrite(str(image_path), image)

    dust_mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(dust_mask, (32, 32), 7, 255, -1)

    rules = _rules(screen_limit=2, tile_limit=1, white_enabled=True)
    rules["default"]["screens"]["WGF50500"] = rules["default"]["screens"].pop("W0F00000")
    detail = {
        "model_id": "GN156HRAAPF0S",
        "images": [
            {
                "image_name": "WGF50500.png",
                "image_path": str(image_path),
                "tiles": [
                    {
                        "tile_id": 0,
                        "x": 0,
                        "y": 0,
                        "width": 64,
                        "height": 64,
                        "is_anomaly": 1,
                        "is_dust": 0,
                        "is_bomb": 0,
                        "is_exclude_zone": 0,
                        "scratch_filtered": 0,
                        "is_aoi_coord": 1,
                        "aoi_defect_code": "C1111",
                        "aoi_product_x": 704,
                        "aoi_product_y": 120,
                        "aoi_image_x": 32,
                        "aoi_image_y": 32,
                        "_runtime_dust_mask": dust_mask,
                    },
                ],
            }
        ],
    }

    result = _evaluate_within_spec_suggestion_detail(
        detail,
        rules,
        visual_output_dir=visual_dir,
        visual_url_prefix="/visuals",
    )

    assert result["suggestion"] is not None
    assert result["panel_summary"]["target_tile_count"] == 1
    assert result["panel_summary"]["evaluated_tile_count"] == 1
    assert result["panel_summary"]["total_dot_count"] == 0
    assert result["panel_summary"]["missed_dot_tile_count"] == 0
    assert result["panel_summary"]["candidate_summary"]["raw_candidate_count"] == 1
    assert result["panel_summary"]["candidate_summary"]["final_candidate_count"] == 0
    assert result["panel_summary"]["candidate_summary"]["dust_mask_filtered_count"] == 1
    assert result["panel_totals"][0]["total_count"] == 0
    assert result["panel_totals"][0]["within"] is True
    assert any(
        step["message"] == "tile 點候選皆落在灰塵遮罩：以 0 點納入 Panel 判定"
        for step in result["steps"]
    )
    assert not any(
        step["message"] == "tile 點偵測未命中：不符合規格內"
        for step in result["steps"]
    )
    rejected_step = next(
        step for step in result["steps"]
        if step["message"] == "點候選被過濾" and step["tile_id"] == 0
    )
    assert rejected_step["rejected"]["black_dot"][0]["reason"] == "dust_mask_overlap"
    assert rejected_step["rejected"]["black_dot"][0]["center_in_dust"] is True
    dust_visual = next(visual for visual in result["visuals"] if visual["tile_id"] == 0)
    assert dust_visual["dust_mask_filtered_count"] == 1
    urls = dust_visual["urls"]
    assert urls["dust_mask_url"].endswith("_dust_mask.png")
    assert urls["dust_overlay_url"].endswith("_dust_overlay.png")
    assert (visual_dir / Path(urls["dust_mask_url"]).name).is_file()
    assert (visual_dir / Path(urls["dust_overlay_url"]).name).is_file()


def test_within_spec_ignores_dot_candidate_inside_mark_bbox(tmp_path):
    image_path = tmp_path / "W0F00000.png"
    visual_dir = tmp_path / "visuals"
    _write_black_dot_image(image_path, [(48, 48)])

    detail = _detail(image_path)
    detail["images"][0]["mark_bbox"] = "40,40,20,20"
    detail["images"][0]["within_spec_no_detect"] = {"mark_padding_px": 0}

    result = _evaluate_within_spec_suggestion_detail(
        detail,
        _rules(),
        visual_output_dir=visual_dir,
        visual_url_prefix="/visuals",
    )

    assert result["suggestion"] is None
    assert result["panel_summary"]["target_tile_count"] == 1
    assert result["panel_summary"]["evaluated_tile_count"] == 0
    assert result["panel_summary"]["missed_dot_tile_count"] == 0
    assert result["panel_summary"]["candidate_summary"]["raw_candidate_count"] == 1
    assert result["panel_summary"]["candidate_summary"]["final_candidate_count"] == 0
    assert result["panel_summary"]["candidate_summary"]["no_detect_mask_filtered_count"] == 1
    assert result["skipped_tiles"]["no_detect_mask"] == 1
    assert any(
        step["message"] == "tile 點候選落在 MARK/不檢測區：略過規格內判定"
        for step in result["steps"]
    )
    visual = result["visuals"][0]
    assert visual["no_detect_mask_filtered_count"] == 1
    assert "no_detect_mask_url" in visual["urls"]
    assert "no_detect_overlay_url" in visual["urls"]
    assert (visual_dir / Path(visual["urls"]["no_detect_mask_url"]).name).is_file()
    assert (visual_dir / Path(visual["urls"]["no_detect_overlay_url"]).name).is_file()


def test_within_spec_all_zero_runtime_dust_mask_does_not_break_visuals(tmp_path):
    image_path = tmp_path / "WGF50500.png"
    visual_dir = tmp_path / "visuals"
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    cv2.circle(image, (32, 32), 3, (60, 60, 60), -1)
    cv2.imwrite(str(image_path), image)

    rules = _rules(screen_limit=1, tile_limit=1, white_enabled=True)
    rules["default"]["screens"]["WGF50500"] = rules["default"]["screens"].pop("W0F00000")
    detail = {
        "model_id": "GN156HRAAPF0S",
        "images": [
            {
                "image_name": "WGF50500.png",
                "image_path": str(image_path),
                "tiles": [
                    {
                        "tile_id": 0,
                        "x": 0,
                        "y": 0,
                        "width": 64,
                        "height": 64,
                        "is_anomaly": 1,
                        "is_dust": 0,
                        "is_bomb": 0,
                        "is_exclude_zone": 0,
                        "scratch_filtered": 0,
                        "is_aoi_coord": 1,
                        "aoi_defect_code": "C1111",
                        "aoi_product_x": 704,
                        "aoi_product_y": 120,
                        "aoi_image_x": 32,
                        "aoi_image_y": 32,
                        "_runtime_dust_mask": np.zeros((64, 64), dtype=np.uint8),
                    },
                ],
            }
        ],
    }

    result = _evaluate_within_spec_suggestion_detail(
        detail,
        rules,
        visual_output_dir=visual_dir,
        visual_url_prefix="/visuals",
    )

    assert result["panel_summary"]["evaluated_tile_count"] == 1
    assert result["panel_summary"]["total_dot_count"] == 1
    visual = result["visuals"][0]
    assert visual["dust_mask_filtered_count"] == 0
    urls = visual["urls"]
    assert urls["dust_mask_url"].endswith("_dust_mask.png")
    assert urls["dust_overlay_url"].endswith("_dust_overlay.png")
    assert (visual_dir / Path(urls["dust_mask_url"]).name).is_file()
    assert (visual_dir / Path(urls["dust_overlay_url"]).name).is_file()


def test_within_spec_no_detect_zone_filters_non_dot_residue_rejections(tmp_path):
    image_path = tmp_path / "W0F00000.png"
    image = np.full((128, 128, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (127, 5), (60, 60, 60), -1)
    cv2.circle(image, (64, 64), 3, (60, 60, 60), -1)
    cv2.imwrite(str(image_path), image)

    detail = _detail(image_path, tile_size=128)
    detail["images"][0]["tiles"][0]["tile_id"] = 0
    detail["images"][0]["within_spec_no_detect"] = {
        "cv_edge_exclude_zones": [
            {"enabled": True, "x": 0, "y": 0, "w": 128, "h": 8},
        ],
    }

    rules = _rules(screen_limit=1, tile_limit=1, threshold_mm=0.5, segmentation_method="hysteresis")
    rules["default"]["dot_detection"]["max_area_px"] = 50000

    result = _evaluate_within_spec_suggestion_detail(
        detail,
        rules,
        visual_output_dir=tmp_path / "visuals",
        visual_url_prefix="/visuals",
    )

    assert result["suggestion"] is not None
    assert result["panel_summary"]["non_dot_residue_count"] == 0
    rejected_step = next(
        step for step in result["steps"]
        if step["message"] == "點候選被過濾" and step["tile_id"] == 0
    )
    no_detect_rejections = [
        item for item in rejected_step["rejected"]["black_dot"]
        if item["reason"] == "no_detect_mask_overlap"
    ]
    assert any(item.get("source_reason") == "aspect_ratio_below_min" for item in no_detect_rejections)
    assert result["visuals"][0]["no_detect_mask_filtered_rejected_count"] >= 1
    assert "no_detect_mask_url" in result["visuals"][0]["urls"]


def test_within_spec_rejects_large_non_dot_residue(tmp_path):
    image_path = tmp_path / "W0F00000.png"
    image = np.full((128, 128, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (127, 5), (60, 60, 60), -1)
    cv2.circle(image, (64, 64), 3, (60, 60, 60), -1)
    cv2.imwrite(str(image_path), image)

    rules = _rules(screen_limit=2, tile_limit=1, threshold_mm=0.5, segmentation_method="hysteresis")
    rules["default"]["dot_detection"]["max_area_px"] = 50000

    detail = _evaluate_within_spec_suggestion_detail(
        _detail(image_path, tile_size=128),
        rules,
        visual_output_dir=tmp_path / "visuals",
        visual_url_prefix="/visuals",
    )

    assert detail["suggestion"] is None
    assert detail["panel_totals"][0]["within"] is True
    assert detail["panel_summary"]["non_dot_residue_count"] == 1
    residue = detail["non_dot_residues"][0]
    assert residue["screen"] == "W0F00000"
    assert residue["reason"] == "aspect_ratio_below_min"
    assert residue["area_px"] >= 500
    assert residue["long_side_px"] >= 80
    assert detail["visuals"][0]["non_dot_residues"][0]["reason"] == "aspect_ratio_below_min"
    assert detail["visuals"][0]["thresholds"]["hysteresis_selected_group"] in (1, 2)
    classified_step = next(
        step for step in detail["steps"]
        if step["message"] == "tile 點類分類完成"
    )
    assert "hysteresis_group2_attempted" in classified_step["detection"]["black_dot"]["thresholds"]
    overlay_name = Path(detail["visuals"][0]["urls"]["overlay_url"]).name
    overlay = cv2.imread(str(tmp_path / "visuals" / overlay_name))
    assert overlay is not None
    red_pixels = (overlay[:, :, 2] > 180) & (overlay[:, :, 1] < 80) & (overlay[:, :, 0] < 80)
    assert int(red_pixels.sum()) > 0
    summary = _format_within_spec_panel_summary(detail)
    assert "非點狀殘留" in summary
    assert "結果=NG" in summary
    assert any(
        step["message"] == "非點狀殘留命中：不建議規格內"
        for step in detail["steps"]
    )


def test_within_spec_dust_mask_filters_non_dot_residue_rejections(tmp_path):
    image_path = tmp_path / "W0F00000.png"
    image = np.full((128, 128, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (127, 5), (60, 60, 60), -1)
    cv2.circle(image, (64, 64), 3, (60, 60, 60), -1)
    cv2.imwrite(str(image_path), image)

    dust_mask = np.zeros((128, 128), dtype=np.uint8)
    dust_mask[0:8, :] = 255

    detail = _detail(image_path, tile_size=128)
    detail["images"][0]["tiles"][0]["tile_id"] = 0
    detail["images"][0]["tiles"][0]["_runtime_dust_mask"] = dust_mask

    rules = _rules(screen_limit=1, tile_limit=1, threshold_mm=0.5, segmentation_method="hysteresis")
    rules["default"]["dot_detection"]["max_area_px"] = 50000

    result = _evaluate_within_spec_suggestion_detail(
        detail,
        rules,
        visual_output_dir=tmp_path / "visuals",
        visual_url_prefix="/visuals",
    )

    assert result["suggestion"] is not None
    assert result["panel_summary"]["non_dot_residue_count"] == 0
    rejected_step = next(
        step for step in result["steps"]
        if step["message"] == "點候選被過濾" and step["tile_id"] == 0
    )
    dust_rejections = [
        item for item in rejected_step["rejected"]["black_dot"]
        if item["reason"] == "dust_mask_overlap"
    ]
    assert any(item.get("source_reason") == "aspect_ratio_below_min" for item in dust_rejections)
    assert result["visuals"][0]["dust_mask_filtered_rejected_count"] >= 1
    assert "dust_mask_url" in result["visuals"][0]["urls"]


def test_dot_detection_hysteresis_expands_low_contrast_boundary():
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    cv2.circle(image, (48, 48), 9, (123, 123, 123), -1)
    cv2.circle(image, (48, 48), 3, (80, 80, 80), -1)

    strict = _detect_dot_components(
        image,
        polarity="black",
        diff_threshold=20,
        background_kernel=31,
        min_area=1,
        max_area=5000,
        morph_open=0,
        size_metric="bbox_max",
        unit_per_px=1.0,
        defect_threshold=0.0,
        include_visuals=False,
    )
    hysteresis = _detect_dot_components(
        image,
        polarity="black",
        diff_threshold=20,
        background_kernel=31,
        min_area=1,
        max_area=5000,
        morph_open=0,
        size_metric="bbox_max",
        unit_per_px=1.0,
        defect_threshold=0.0,
        segmentation_method="hysteresis",
        hysteresis_low_threshold=4,
        hysteresis_high_threshold=20,
        include_visuals=False,
    )

    assert strict["candidates"]
    assert hysteresis["candidates"]
    assert hysteresis["candidates"][0]["size_px"] > strict["candidates"][0]["size_px"]
    assert hysteresis["thresholds"]["hysteresis_low_threshold"] == 4
    assert hysteresis["thresholds"]["hysteresis_high_threshold"] == 20


def test_dot_detection_hysteresis_v2_switches_to_second_group_when_group1_empty():
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    cv2.circle(image, (48, 48), 6, (120, 120, 120), -1)

    detected = _detect_dot_components(
        image,
        polarity="black",
        diff_threshold=30,
        background_kernel=31,
        min_area=5,
        max_area=5000,
        morph_open=0,
        size_metric="bbox_max",
        unit_per_px=1.0,
        defect_threshold=0.0,
        segmentation_method="hysteresis",
        hysteresis_low_threshold=20,
        hysteresis_high_threshold=30,
        hysteresis_edge_width_percent=3.0,
        hysteresis_edge_extra_threshold=2,
        hysteresis_second_low_threshold=3,
        hysteresis_second_high_threshold=4,
        hysteresis_second_edge_width_percent=9.5,
        hysteresis_second_edge_extra_threshold=2,
        hysteresis_switch_count_threshold=5,
        hysteresis_second_max_count=5,
        hysteresis_edge_suppress_percent=0.0,
        include_visuals=False,
    )

    assert detected["candidates"]
    assert detected["candidates"][0]["polarity"] == "black"
    assert detected["thresholds"]["hysteresis_group1_count"] == 0
    assert detected["thresholds"]["hysteresis_group2_count"] >= 1
    assert detected["thresholds"]["hysteresis_selected_group"] == 2
    assert detected["thresholds"]["hysteresis_group2_attempted"] is True
    assert detected["thresholds"]["hysteresis_group2_adopted"] is True
    assert detected["thresholds"]["hysteresis_switch_reason"] == "group1_empty"
    assert detected["thresholds"]["hysteresis_group2_reject_reason"] == ""


def test_dot_detection_hysteresis_v2_records_second_group_reject_reason():
    image = np.full((128, 128, 3), 128, dtype=np.uint8)
    for center in ((24, 24), (64, 24), (104, 24), (24, 64), (64, 64), (104, 64)):
        cv2.circle(image, center, 5, (118, 118, 118), -1)

    detected = _detect_dot_components(
        image,
        polarity="black",
        diff_threshold=4,
        background_kernel=31,
        min_area=5,
        max_area=5000,
        morph_open=0,
        size_metric="bbox_max",
        unit_per_px=1.0,
        defect_threshold=0.0,
        segmentation_method="hysteresis",
        hysteresis_low_threshold=2,
        hysteresis_high_threshold=3,
        hysteresis_edge_width_percent=3.0,
        hysteresis_edge_extra_threshold=2,
        hysteresis_second_low_threshold=3,
        hysteresis_second_high_threshold=4,
        hysteresis_second_edge_width_percent=3.0,
        hysteresis_second_edge_extra_threshold=2,
        hysteresis_switch_count_threshold=2,
        hysteresis_second_max_count=1,
        hysteresis_edge_suppress_percent=0.0,
        include_visuals=False,
    )

    assert detected["thresholds"]["hysteresis_group1_count"] > 2
    assert detected["thresholds"]["hysteresis_group2_count"] > 1
    assert detected["thresholds"]["hysteresis_selected_group"] == 1
    assert detected["thresholds"]["hysteresis_group2_attempted"] is True
    assert detected["thresholds"]["hysteresis_group2_adopted"] is False
    assert detected["thresholds"]["hysteresis_switch_reason"] == "group1_count_above_switch"
    assert detected["thresholds"]["hysteresis_group2_reject_reason"] == "group2_count_above_max2"


def test_dot_detection_hysteresis_v2_switch_count_uses_filtered_candidates():
    image = np.full((128, 128, 3), 128, dtype=np.uint8)
    cv2.circle(image, (64, 64), 5, (105, 105, 105), -1)
    for y in (5, 30, 55, 80, 105):
        cv2.rectangle(image, (124, y), (127, min(127, y + 12)), (118, 118, 118), -1)

    detected = _detect_dot_components(
        image,
        polarity="black",
        diff_threshold=4,
        background_kernel=31,
        min_area=5,
        max_area=5000,
        morph_open=0,
        size_metric="bbox_max",
        unit_per_px=1.0,
        defect_threshold=0.0,
        min_aspect_ratio=0.45,
        edge_margin=8,
        segmentation_method="hysteresis",
        hysteresis_low_threshold=2,
        hysteresis_high_threshold=3,
        hysteresis_edge_width_percent=0.0,
        hysteresis_edge_extra_threshold=0,
        hysteresis_second_low_threshold=3,
        hysteresis_second_high_threshold=4,
        hysteresis_second_edge_width_percent=0.0,
        hysteresis_second_edge_extra_threshold=0,
        hysteresis_switch_count_threshold=0,
        hysteresis_second_max_count=1,
        hysteresis_edge_suppress_percent=0.0,
        include_visuals=False,
    )

    thresholds = detected["thresholds"]
    assert thresholds["hysteresis_group1_count"] == 1
    assert thresholds["hysteresis_group2_count"] == 1
    assert thresholds["hysteresis_selected_group"] == 2
    assert thresholds["hysteresis_group2_adopted"] is True
    assert len(detected["candidates"]) == 1
    assert detected["candidates"][0]["aspect_ratio"] == 1.0
    assert any(r["reason"] == "aspect_ratio_below_min" for r in detected["rejected_candidates"])


def test_dot_detection_morph_hat_expands_low_contrast_dark_dot():
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    cv2.circle(image, (48, 48), 9, (123, 123, 123), -1)
    cv2.circle(image, (48, 48), 3, (80, 80, 80), -1)

    strict = _detect_dot_components(
        image,
        polarity="black",
        diff_threshold=20,
        background_kernel=31,
        min_area=1,
        max_area=5000,
        morph_open=0,
        size_metric="bbox_max",
        unit_per_px=1.0,
        defect_threshold=0.0,
        include_visuals=False,
    )
    morph_hat = _detect_dot_components(
        image,
        polarity="black",
        diff_threshold=20,
        background_kernel=31,
        min_area=1,
        max_area=5000,
        morph_open=0,
        size_metric="bbox_max",
        unit_per_px=1.0,
        defect_threshold=0.0,
        segmentation_method="morph_hat",
        hysteresis_low_threshold=4,
        hysteresis_high_threshold=20,
        include_visuals=False,
    )

    assert strict["candidates"]
    assert morph_hat["segmentation_method"] == "morph_hat"
    assert morph_hat["candidates"]
    assert morph_hat["candidates"][0]["size_px"] > strict["candidates"][0]["size_px"]


def test_dot_detection_adaptive_mean_detects_low_contrast_black_and_white_dots():
    black_image = np.full((96, 96, 3), 128, dtype=np.uint8)
    white_image = np.full((96, 96, 3), 128, dtype=np.uint8)
    cv2.circle(black_image, (48, 48), 6, (120, 120, 120), -1)
    cv2.circle(white_image, (48, 48), 6, (136, 136, 136), -1)

    common = {
        "diff_threshold": 4,
        "background_kernel": 31,
        "min_area": 5,
        "max_area": 5000,
        "morph_open": 0,
        "size_metric": "bbox_max",
        "unit_per_px": 1.0,
        "defect_threshold": 0.0,
        "segmentation_method": "adaptive_mean",
        "include_visuals": False,
    }
    black = _detect_dot_components(black_image, polarity="black", **common)
    white = _detect_dot_components(white_image, polarity="white", **common)

    assert black["segmentation_method"] == "adaptive_mean"
    assert white["segmentation_method"] == "adaptive_mean"
    assert black["candidates"]
    assert white["candidates"]
    assert black["candidates"][0]["size_px"] >= 10
    assert white["candidates"][0]["size_px"] >= 10


def test_dot_detection_filters_line_and_edge_components():
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (0, 10), (2, 80), (60, 60, 60), -1)
    cv2.circle(image, (48, 48), 4, (60, 60, 60), -1)

    detected = _detect_dot_components(
        image,
        polarity="black",
        diff_threshold=20,
        background_kernel=31,
        min_area=1,
        max_area=5000,
        morph_open=0,
        size_metric="bbox_max",
        unit_per_px=1.0,
        defect_threshold=0.0,
        min_aspect_ratio=0.45,
        edge_margin=4,
        include_visuals=False,
    )

    assert len(detected["candidates"]) == 1
    candidate = detected["candidates"][0]
    assert 43 <= candidate["x"] <= 49
    assert candidate["aspect_ratio"] >= 0.45
    assert any(
        rejected["reason"] == "aspect_ratio_below_min"
        and rejected["x"] == 0
        for rejected in detected["rejected_candidates"]
    )


def test_white_halo_detection_measures_area_around_dark_seed():
    image = np.full((128, 128, 3), 80, dtype=np.uint8)
    cv2.circle(image, (64, 64), 28, (90, 90, 90), -1)
    cv2.circle(image, (64, 64), 3, (55, 55, 55), -1)

    halo = _detect_white_halo_components(
        image,
        diff_threshold=2,
        background_kernel=31,
        min_area=50,
        max_area=10000,
        morph_open=0,
        size_metric="bbox_max",
        unit_per_px=1.0,
        defect_threshold=0.0,
        min_aspect_ratio=0.25,
        edge_margin=4,
        include_visuals=False,
    )

    assert halo["segmentation_method"] == "halo"
    assert halo["candidates"]
    assert halo["candidates"][0]["size_px"] > 30


def test_dot_detection_auto_selects_white_halo_around_dark_seed():
    image = np.full((128, 128, 3), 80, dtype=np.uint8)
    cv2.circle(image, (64, 64), 28, (83, 83, 83), -1)
    cv2.circle(image, (64, 64), 3, (55, 55, 55), -1)

    auto = _detect_dot_components_auto(
        image,
        polarity="white",
        segmentation_method="auto",
        diff_threshold=4,
        background_kernel=31,
        min_area=20,
        max_area=10000,
        morph_open=0,
        size_metric="bbox_max",
        unit_per_px=1.0,
        defect_threshold=0.0,
        min_aspect_ratio=0.45,
        edge_margin=4,
        hysteresis_low_threshold=2,
        hysteresis_high_threshold=4,
        include_visuals=False,
    )

    assert auto["segmentation_method"] == "auto:halo"
    assert auto["candidates"]
    assert auto["candidates"][0]["size_px"] > 30
    assert any(item["segmentation_method"] == "halo" for item in auto["auto_candidates"])


def test_dot_detection_debug_auto_polarity_selects_white_halo():
    image = np.full((128, 128, 3), 80, dtype=np.uint8)
    cv2.circle(image, (64, 64), 28, (83, 83, 83), -1)
    cv2.circle(image, (64, 64), 3, (55, 55, 55), -1)

    detected = _detect_dot_components_debug_polarity(
        image,
        polarity="auto",
        segmentation_method="auto",
        diff_threshold=4,
        background_kernel=31,
        min_area=20,
        max_area=10000,
        morph_open=0,
        size_metric="bbox_max",
        unit_per_px=1.0,
        defect_threshold=0.0,
        min_aspect_ratio=0.45,
        edge_margin=4,
        hysteresis_low_threshold=2,
        hysteresis_high_threshold=4,
        include_visuals=False,
    )

    assert detected["detected_polarity"] == "white"
    assert detected["segmentation_method"] == "auto:white:halo"
    assert detected["candidates"]
    assert detected["candidates"][0]["polarity"] == "white"
    assert any(item["segmentation_method"] == "white:halo" for item in detected["auto_candidates"])


def test_dot_detection_debug_auto_keeps_rejected_candidates_from_nonwinning_polarity():
    image = np.full((128, 128, 3), 128, dtype=np.uint8)
    cv2.circle(image, (64, 64), 3, (60, 60, 60), -1)
    cv2.rectangle(image, (0, 0), (14, 59), (190, 190, 190), -1)

    detected = _detect_dot_components_debug_polarity(
        image,
        polarity="auto",
        segmentation_method="background_diff",
        diff_threshold=8,
        background_kernel=31,
        min_area=10,
        max_area=50000,
        morph_open=1,
        size_metric="bbox_diagonal",
        unit_per_px=0.02,
        defect_threshold=0.3,
        min_aspect_ratio=0.0,
        edge_margin=8,
        include_visuals=False,
    )

    assert detected["detected_polarity"] == "black"
    assert detected["candidates"][0]["polarity"] == "black"
    assert set(detected["polarity_results"]) == {"black", "white"}
    assert detected["polarity_results"]["black"]["detected_polarity"] == "black"
    assert detected["polarity_results"]["white"]["detected_polarity"] == "white"
    assert any(
        rejected["reason"] == "edge_margin"
        and rejected.get("source_polarity") == "white"
        and rejected.get("expected_polarity") == "white"
        for rejected in detected["rejected_candidates"]
    )


def test_within_spec_auto_visual_output_uses_heatmap_url(tmp_path):
    visual_dir, visual_prefix = _within_spec_auto_visual_output(str(tmp_path), "PANEL:001")

    assert visual_dir.parent.name == "within_spec_inference"
    assert visual_dir.name.startswith("PANEL_001_")
    assert visual_prefix.startswith("/heatmaps/within_spec_inference/PANEL_001_")


def test_within_spec_panel_summary_lists_all_screens():
    detail = {
        "panel_totals": [
            {
                "screen": "W0F00000",
                "dot_label": "黑點",
                "max_size_mm": 0.1952,
                "threshold_mm": 0.3,
                "total_count": 1,
                "screen_count_limit": 3,
                "max_tile_count": 1,
                "tile_count_threshold": 2,
                "within": True,
            },
            {
                "screen": "WGF50500",
                "dot_label": "黑點",
                "max_size_mm": 0.18,
                "threshold_mm": 0.3,
                "total_count": 1,
                "screen_count_limit": 3,
                "max_tile_count": 1,
                "tile_count_threshold": 2,
                "within": True,
            },
        ],
    }

    summary = _format_within_spec_panel_summary(detail)

    assert "W0F00000 黑點" in summary
    assert "WGF50500 黑點" in summary
    assert summary.count("結果=OK") == 2


def test_within_spec_panel_summary_shows_failed_comparison_operator():
    detail = {
        "panel_totals": [
            {
                "screen": "R0F00000",
                "dot_label": "黑點",
                "max_size_mm": 0.2817,
                "threshold_mm": 0.3,
                "total_count": 7,
                "screen_count_limit": 2,
                "max_tile_count": 7,
                "tile_count_threshold": 1,
                "within": False,
            },
        ],
    }

    summary = _format_within_spec_panel_summary(detail)

    assert "最大尺寸 0.2817mm <= 0.3000mm [OK]" in summary
    assert "畫面總點數 7 > 2 [NG]" in summary
    assert "單Tile最大點數 7 > 1 [NG]" in summary
    assert "結果=NG" in summary


def test_within_spec_inference_note_formats_panel_results_on_separate_lines():
    within_spec_info = {
        "converted": False,
        "status": "not_within_spec",
        "reason": "",
        "detail": {
            "rule_selection": {"matched_machine_key": "GN156HRAAPF0S"},
            "panel_summary": {"target_tile_count": 2, "evaluated_tile_count": 2},
            "panel_totals": [
                {
                    "screen": "W0F00000",
                    "dot_label": "黑點",
                    "max_size_mm": 0.2113,
                    "threshold_mm": 0.35,
                    "total_count": 1,
                    "screen_count_limit": 2,
                    "max_tile_count": 1,
                    "tile_count_threshold": 1,
                    "within": True,
                },
                {
                    "screen": "R0F00000",
                    "dot_label": "黑點",
                    "max_size_mm": 0.2817,
                    "threshold_mm": 0.35,
                    "total_count": 7,
                    "screen_count_limit": 2,
                    "max_tile_count": 7,
                    "tile_count_threshold": 1,
                    "within": False,
                },
            ],
        },
    }

    note = _format_within_spec_inference_note(within_spec_info, "http://example/within-spec-logs")
    lines = note.splitlines()

    assert lines[0] == "[WITHIN_SPEC_INFERENCE] 原始 AI=NG，已執行規格內檢查，結果=not_within_spec"
    assert lines[1] == "  matched_machine=GN156HRAAPF0S；target_tiles=2；evaluated_tiles=2"
    assert lines[2].startswith("  - W0F00000 黑點：")
    assert lines[3].startswith("  - R0F00000 黑點：")
    assert "畫面總點數 7 > 2 [NG]" in lines[3]
    assert lines[4] == "  明細：http://example/within-spec-logs"


def test_within_spec_inference_note_includes_missed_aoi_dot_tiles():
    within_spec_info = {
        "converted": False,
        "status": "not_within_spec",
        "reason": "",
        "detail": {
            "rule_selection": {"matched_machine_key": "GN156HRAAPF0S"},
            "panel_summary": {"target_tile_count": 3, "evaluated_tile_count": 2},
            "panel_totals": [
                {
                    "screen": "WGF50500",
                    "dot_label": "黑點",
                    "max_size_mm": 0.2347,
                    "threshold_mm": 0.3,
                    "total_count": 1,
                    "screen_count_limit": 2,
                    "max_tile_count": 1,
                    "tile_count_threshold": 1,
                    "within": True,
                },
            ],
            "missed_dot_tiles": [
                {
                    "screen": "WGF50500",
                    "tile_id": 0,
                    "is_aoi_coord": True,
                    "aoi_product_x": 704,
                    "aoi_product_y": 120,
                },
            ],
        },
    }

    note = _format_within_spec_inference_note(within_spec_info, "http://example/within-spec-logs")

    assert "[WITHIN_SPEC_INFERENCE]" in note
    assert "WGF50500 AOI點未檢出：數量 1 > 0 [NG]；tile 0 AOI(704,120)；結果=NG" in note
