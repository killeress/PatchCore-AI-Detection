import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_image_preprocess_lab import apply_preprocess_method, get_method_specs, make_diff_image


def test_median_filter_clamps_to_odd_kernel_and_reduces_impulse():
    image = np.full((21, 21), 100, dtype=np.uint8)
    image[10, 10] = 255

    result = apply_preprocess_method(image, "median", {"kernel_size": 4})

    assert result["applied_params"]["kernel_size"] == 5
    assert result["image"][10, 10] == 100
    assert result["image"].shape == image.shape


def test_gaussian_filter_returns_diff_image_with_color_map():
    image = np.zeros((24, 24), dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (15, 15), 200, -1)

    result = apply_preprocess_method(image, "gaussian", {"kernel_size": 5, "sigma": 1.0})
    diff = make_diff_image(image, result["image"])

    assert result["image"].shape == image.shape
    assert diff.shape == (24, 24, 3)
    assert diff.dtype == np.uint8


def test_clahe_enhances_local_contrast_and_clamps_params():
    image = np.full((64, 64), 96, dtype=np.uint8)
    image[:, 32:] = 104
    cv2.circle(image, (32, 32), 10, 112, -1)

    result = apply_preprocess_method(
        image,
        "clahe",
        {"clip_limit": 99.0, "tile_grid_size": 1},
    )

    assert result["method_label"] == "CLAHE 局部對比增強"
    assert result["applied_params"] == {"clip_limit": 20.0, "tile_grid_size": 2}
    assert result["image"].shape == image.shape
    assert result["image"].dtype == np.uint8
    assert result["image"].std() > image.std()


def test_gray_band_shift_pushes_values_outside_band_and_can_fill_band():
    image = np.array([[0, 104, 105, 108, 110, 111, 250]], dtype=np.uint8)

    keep = apply_preprocess_method(
        image,
        "gray_band_shift",
        {
            "low_threshold": 105,
            "high_threshold": 110,
            "dark_shift": 10,
            "bright_shift": 10,
            "band_mode": "keep",
        },
    )
    fill = apply_preprocess_method(
        image,
        "gray_band_shift",
        {
            "low_threshold": 105,
            "high_threshold": 110,
            "dark_shift": 10,
            "bright_shift": 10,
            "band_mode": "to_high",
        },
    )

    np.testing.assert_array_equal(keep["image"], np.array([[0, 94, 105, 108, 110, 121, 255]], dtype=np.uint8))
    np.testing.assert_array_equal(fill["image"], np.array([[0, 94, 110, 110, 110, 121, 255]], dtype=np.uint8))
    assert keep["method_label"] == "分段灰階映射"


def test_stripe_profile_correction_reduces_vertical_profile_variation():
    base = np.full((64, 96), 128, dtype=np.float32)
    stripe = (np.sin(np.arange(96) / 2.0) * 18).astype(np.float32)
    image = np.clip(base + stripe.reshape(1, -1), 0, 255).astype(np.uint8)

    result = apply_preprocess_method(
        image,
        "stripe_profile",
        {"orientation": "vertical", "smooth_kernel": 31, "strength": 1.0},
    )

    before = np.std(np.median(image, axis=0))
    after = np.std(np.median(result["image"], axis=0))
    assert after < before * 0.5


def test_laplace_sharpen_preserves_shape_and_enhances_edges():
    image = np.full((32, 32), 100, dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (21, 21), 140, -1)

    result = apply_preprocess_method(
        image,
        "laplace_sharpen",
        {"kernel_size": 4, "strength": 0.5},
    )

    assert result["applied_params"]["kernel_size"] == 5
    assert result["image"].shape == image.shape
    assert result["image"].dtype == np.uint8
    assert np.max(cv2.absdiff(image, result["image"])) > 0


def test_each_method_has_frontend_explanation_fields():
    required = {"id", "label", "purpose", "noise_types", "risk", "suggested", "mix", "params"}

    for spec in get_method_specs():
        assert required <= set(spec)
        assert spec["noise_types"]
        assert spec["params"]


def test_debug_preprocess_lab_exposes_clahe_params():
    template = (Path(__file__).resolve().parent.parent / "templates" / "debug_inference.html").read_text(
        encoding="utf-8"
    )

    assert 'data-lab-panel="clahe"' in template
    assert 'id="lab-clahe-clip-limit"' in template
    assert 'id="lab-clahe-tile-grid-size"' in template
    assert "clip_limit: parseFloat" in template
    assert "tile_grid_size: parseInt" in template


def test_debug_dot_results_render_black_and_white_separately():
    template = (Path(__file__).resolve().parent.parent / "templates" / "debug_inference.html").read_text(
        encoding="utf-8"
    )

    assert 'id="dot-polarity-results"' in template
    assert "renderDotPolarityResults(data, polarityResults)" in template
    assert "const sections = ['black', 'white']" in template
