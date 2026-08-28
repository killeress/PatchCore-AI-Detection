from pathlib import Path

import cv2
import numpy as np
import pytest

from capi_grid_canonicalization import (
    canonicalize_panel_grid,
    normalize_grid_canonicalization,
    validate_samples_per_cell,
)
from capi_preprocess import PreprocessConfig, preprocess_panel_image


@pytest.mark.parametrize("value", [1, 3, "1", "3"])
def test_samples_per_cell_accepts_only_product_options(value):
    assert validate_samples_per_cell(value) in (1, 3)


@pytest.mark.parametrize("value", [0, 1.5, 2, 4, True, None, "bad"])
def test_samples_per_cell_rejects_other_values(value):
    with pytest.raises(ValueError, match="1 or 3"):
        validate_samples_per_cell(value)


def test_enabled_contract_requires_product_resolution_when_persisted():
    with pytest.raises(ValueError, match="product_resolution"):
        normalize_grid_canonicalization(
            {"enabled": True, "samples_per_cell": 3},
            require_resolution=True,
        )


def test_grid_canonicalization_keeps_camera_shape_and_outside_polygon():
    image = np.full((100, 160), 20, dtype=np.uint8)
    image[10:90, 20:140] = np.tile(
        np.array([40, 220, 40, 220], dtype=np.uint8),
        (80, 30),
    )
    polygon = np.array(
        [[20, 10], [139, 10], [139, 89], [20, 89]], dtype=np.float32
    )

    result = canonicalize_panel_grid(
        image,
        polygon,
        product_resolution=(30, 20),
        samples_per_cell=1,
    )

    assert result.image.shape == image.shape
    assert result.image.dtype == image.dtype
    assert result.canonical_size == (30, 20)
    assert result.rectified_size == (120, 80)
    assert np.array_equal(result.image[:8, :], image[:8, :])
    assert np.array_equal(result.image[:, :18], image[:, :18])


def test_one_sample_per_cell_suppresses_phase_shift_but_keeps_cell_scale_defect():
    height, width = 80, 120
    base_a = np.indices((height, width)).sum(axis=0) % 2 * 180 + 30
    base_b = (np.indices((height, width)).sum(axis=0) + 1) % 2 * 180 + 30
    base_a = base_a.astype(np.uint8)
    base_b = base_b.astype(np.uint8)
    # A multi-cell bright defect should survive the area average.
    base_a[28:44, 48:64] = 255
    base_b[28:44, 48:64] = 255
    polygon = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    out_a = canonicalize_panel_grid(base_a, polygon, (30, 20), 1).image
    out_b = canonicalize_panel_grid(base_b, polygon, (30, 20), 1).image

    raw_phase_difference = np.mean(np.abs(base_a.astype(float) - base_b.astype(float)))
    normalized_difference = np.mean(np.abs(out_a.astype(float) - out_b.astype(float)))
    assert normalized_difference < raw_phase_difference * 0.1
    assert float(out_a[32:40, 52:60].mean()) > float(out_a[:16, :16].mean()) + 50


def test_preprocess_panel_image_records_grid_step_and_preserves_tile_coordinates(
    tmp_path: Path,
    monkeypatch,
):
    path = tmp_path / "G0F00000_grid.png"
    image = np.zeros((192, 256), dtype=np.uint8)
    image[16:176, 16:240] = 180
    cv2.imwrite(str(path), image)
    polygon = np.array(
        [[16, 16], [239, 16], [239, 175], [16, 175]], dtype=np.float32
    )
    monkeypatch.setattr(
        "capi_preprocess.detect_panel_polygon",
        lambda _image, _config: ((16, 16, 240, 176), polygon.copy()),
    )

    result = preprocess_panel_image(
        path,
        "G0F00000",
        PreprocessConfig(
            tile_size=64,
            tile_stride=64,
            product_resolution=(56, 40),
            grid_canonicalization_enabled=True,
            grid_samples_per_cell=3,
            cache_processed_image=True,
        ),
    )

    assert result.processed_image.shape == image.shape
    assert result.tiles
    assert all(tile.x2 - tile.x1 == 64 for tile in result.tiles)
    assert all(tile.y2 - tile.y1 == 64 for tile in result.tiles)
    grid_steps = [
        step for step in result.preprocess_steps
        if step["method"] == "grid_canonicalization"
    ]
    assert len(grid_steps) == 1
    assert grid_steps[0]["applied_params"]["samples_per_cell"] == 3
    assert grid_steps[0]["stats"]["canonical_size"] == [168, 120]
