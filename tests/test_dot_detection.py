import cv2
import numpy as np
from pathlib import Path

from capi_web import (
    CAPIWebHandler,
    DOT_RULER_MM_PER_PX,
    _detect_dot_components,
    _preprocess_dot_image_for_detection,
)


def test_detects_black_dot_without_unit_calibration():
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    cv2.circle(image, (48, 48), 4, (70, 70, 70), -1)

    result = _detect_dot_components(
        image,
        polarity="black",
        diff_threshold=8,
        background_kernel=31,
        min_area=2,
        max_area=1000,
        morph_open=1,
        size_metric="bbox_max",
        unit_per_px=0.0,
        defect_threshold=0.3,
    )

    assert result["calibrated"] is False
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["size_px"] >= 8
    assert result["candidates"][0]["size_units"] is None
    assert result["candidates"][0]["is_defect"] is False


def test_detects_white_dot_and_applies_physical_threshold():
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    cv2.circle(image, (48, 48), 4, (210, 210, 210), -1)

    result = _detect_dot_components(
        image,
        polarity="white",
        diff_threshold=8,
        background_kernel=31,
        min_area=2,
        max_area=1000,
        morph_open=1,
        size_metric="bbox_max",
        unit_per_px=0.05,
        defect_threshold=0.3,
    )

    assert result["calibrated"] is True
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["size_units"] >= 0.3
    assert result["candidates"][0]["is_defect"] is True
    assert result["candidates"][0]["size_mm"] >= 0.3


def test_applies_dot_ruler_mm_scale():
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (44, 44), (50, 51), (60, 60, 60), -1)

    result = _detect_dot_components(
        image,
        polarity="black",
        diff_threshold=8,
        background_kernel=31,
        min_area=2,
        max_area=1000,
        morph_open=1,
        size_metric="bbox_diagonal",
        unit_per_px=DOT_RULER_MM_PER_PX,
        defect_threshold=0.3,
    )

    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["size_px"] == result["candidates"][0]["bbox_diagonal_px"]
    assert result["candidates"][0]["size_mm"] == result["candidates"][0]["size_units"]
    assert result["candidates"][0]["size_mm"] == 0.2604
    assert result["candidates"][0]["is_defect"] is False


def test_bbox_diagonal_size_uses_component_bbox_diagonal():
    image = np.full((120, 120, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (55, 69), (60, 60, 60), -1)

    result = _detect_dot_components(
        image,
        polarity="black",
        diff_threshold=8,
        background_kernel=31,
        min_area=2,
        max_area=1000,
        morph_open=1,
        size_metric="bbox_diagonal",
        unit_per_px=DOT_RULER_MM_PER_PX,
        defect_threshold=0.3,
    )

    candidate = result["candidates"][0]
    assert candidate["size_mode"] == "bbox_diagonal"
    assert candidate["bbox_max_diameter_px"] == 20
    assert candidate["size_px"] == candidate["bbox_diagonal_px"]
    assert candidate["size_px"] > candidate["bbox_max_diameter_px"]


def test_custom_unit_per_px_controls_size_mm():
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (44, 44), (46, 47), (60, 60, 60), -1)

    result = _detect_dot_components(
        image,
        polarity="black",
        diff_threshold=8,
        background_kernel=31,
        min_area=2,
        max_area=1000,
        morph_open=1,
        size_metric="bbox_diagonal",
        unit_per_px=0.1,
        defect_threshold=0.3,
    )

    candidate = result["candidates"][0]
    assert candidate["size_px"] == 5.0
    assert candidate["size_mm"] == 0.5


def test_dot_detection_preprocess_uses_gaussian_lab_method(monkeypatch):
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    calls = []

    def fake_apply_preprocess_method(input_image, method, params):
        calls.append((input_image, method, params))
        return {
            "image": input_image.copy(),
            "method": method,
            "method_label": "高斯平滑",
            "applied_params": params,
            "notes": [],
        }

    monkeypatch.setattr(
        "capi_image_preprocess_lab.apply_preprocess_method",
        fake_apply_preprocess_method,
    )

    processed, info = _preprocess_dot_image_for_detection(image)

    assert len(calls) == 1
    assert calls[0][0] is image
    assert calls[0][1] == "gaussian"
    assert calls[0][2] == {"kernel_size": 7, "sigma": 1.0}
    assert np.array_equal(processed, image)
    assert info["method"] == "gaussian"
    assert info["applied_params"] == {"kernel_size": 7, "sigma": 1.0}


def test_bundled_black_sample_default_opening_filters_background_texture():
    image_path = Path("templates/imgs/black/3.png")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None

    result = _detect_dot_components(
        image,
        polarity="black",
        diff_threshold=8,
        background_kernel=31,
        min_area=2,
        max_area=5000,
        morph_open=3,
        size_metric="bbox_max",
        unit_per_px=0.0,
        defect_threshold=0.3,
    )

    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["size_px"] <= 10


def test_debug_dot_api_returns_separate_black_and_white_results(tmp_path, monkeypatch):
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    cv2.circle(image, (32, 48), 4, (60, 60, 60), -1)
    cv2.circle(image, (64, 48), 4, (210, 210, 210), -1)
    image_path = tmp_path / "black_white.png"
    assert cv2.imwrite(str(image_path), image)

    payload = {
        "image_path": str(image_path),
        "polarity": "auto",
        "segmentation_method": "background_diff",
        "diff_threshold": 8,
        "background_kernel": 31,
        "min_area": 2,
        "max_area": 1000,
        "morph_open": 0,
        "min_aspect_ratio": 0.0,
        "edge_margin": 4,
        "size_metric": "bbox_diagonal",
        "unit_per_px": 0.02,
        "defect_threshold": 0.3,
    }
    responses = []
    handler = object.__new__(CAPIWebHandler)
    handler._read_json_body = lambda: payload
    handler._send_json = lambda data, status=200: responses.append((status, data))
    monkeypatch.setattr(CAPIWebHandler, "_debug_heatmap_dir", tmp_path / "debug")

    handler._handle_debug_dot_detection()

    assert responses[0][0] == 200
    polarity_results = responses[0][1]["polarity_results"]
    assert set(polarity_results) == {"black", "white"}
    assert polarity_results["black"]["count"] == 1
    assert polarity_results["white"]["count"] == 1
    for result in polarity_results.values():
        for key in ("overlay_url", "mask_url", "diff_url"):
            assert (CAPIWebHandler._debug_heatmap_dir / Path(result[key]).name).is_file()
