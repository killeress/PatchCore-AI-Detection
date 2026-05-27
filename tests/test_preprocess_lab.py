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
