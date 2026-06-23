import cv2
import numpy as np

from capi_web import (
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
