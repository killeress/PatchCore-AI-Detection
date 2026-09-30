"""Raw gradient correction must recover dim borders without following clutter."""
from dataclasses import replace
import logging

import cv2
import numpy as np
import pytest

from capi_boundary_refinement import refine_panel_polygon_from_raw
from capi_preprocess import (
    PreprocessConfig, detect_panel_boundary, detect_panel_geometry,
    panel_boundary_config_for_station, preprocess_panel_folder, preprocess_panel_image,
)


def faded_panel():
    image = np.full((1100, 1400), 8, np.uint8)
    image[170:930, 200:1200] = 140
    image[170:930, 1160:1200] = np.linspace(140, 44, 40).astype(np.uint8)
    return cv2.GaussianBlur(image, (5, 5), .8)


def seed_polygon():
    return np.array([[201, 170], [1187, 170], [1187, 929], [201, 929]], np.float32)


def right_x(polygon, y=550):
    top, bottom = polygon[1:3]
    return top[0] + (bottom[0]-top[0]) * (y-top[1]) / (bottom[1]-top[1])


def config(**kwargs):
    return PreprocessConfig(
        **panel_boundary_config_for_station("capi"),
        product_resolution=(1920, 1200), generate_grid_tiles=False, **kwargs,
    )


def test_recovers_dim_border_without_changing_pixels_and_is_idempotent(caplog):
    image, seed = faded_panel(), seed_polygon()
    original_image, original_seed = image.copy(), seed.copy()
    with caplog.at_level(logging.INFO, logger="capi.preprocess"):
        result = refine_panel_polygon_from_raw(image, seed, source_name="W0F_test")
    assert 1198.5 < right_x(result) < 1200
    assert cv2.pointPolygonTest(result, (1194., 550.), True) > 0
    assert cv2.pointPolygonTest(result, (1205., 550.), True) < 0
    assert "status=applied" in caplog.text and "source=W0F_test" in caplog.text
    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(seed, original_seed)
    np.testing.assert_array_equal(refine_panel_polygon_from_raw(image, result), result)


@pytest.mark.parametrize("case", ["flat", "weak", "inverted", "reflection", "far", "curved"])
def test_rejects_unsupported_edges_and_preserves_entire_seed(case, caplog):
    image, seed = faded_panel(), seed_polygon()
    if case == "flat":
        image[:] = 100
    elif case == "weak":
        image[:] = 8
        image[170:930, 200:1200] = 16
    elif case == "inverted":
        image = 255 - image
    elif case == "reflection":
        image[170:930, 1210:1220] = 240
    elif case == "far":
        seed[1:3, 0] = 1167
    elif case == "curved":
        image[:] = 8
        for y in range(170, 930):
            x = 1200 + int(8 * np.sin((y-170) * np.pi / 760))
            image[y, 200:x] = 140
    with caplog.at_level(logging.INFO, logger="capi.preprocess"):
        result = refine_panel_polygon_from_raw(image, seed)
    assert result is seed
    assert "status=kept" in caplog.text and "reason=" in caplog.text


def test_distributed_support_ignores_local_reflection():
    image = faded_panel()
    image[440:515, 1210:1220] = 240
    refined = refine_panel_polygon_from_raw(image, seed_polygon())
    assert 1198.5 < right_x(refined) < 1200


def test_rejects_area_change_even_when_individual_shifts_are_small(caplog):
    image = np.full((960, 960), 8, np.uint8)
    image[180:780, 200:680] = 140
    seed = np.array([[210, 180], [660, 180], [660, 779], [210, 779]], np.float32)
    with caplog.at_level(logging.INFO, logger="capi.preprocess"):
        result = refine_panel_polygon_from_raw(image, seed)
    assert result is seed
    assert "area_change_too_large" in caplog.text


def test_already_aligned_polygon_is_preserved_exactly():
    image = np.full((1100, 1400), 8, np.uint8)
    image[170:930, 200:1200] = 140
    seed = np.array([[200, 170], [1199, 170], [1199, 929], [200, 929]], np.float32)
    assert refine_panel_polygon_from_raw(image, seed) is seed


@pytest.mark.parametrize("case", ["none", "nonfinite", "concave", "uint16", "clipped"])
def test_invalid_or_unavailable_seed_is_not_promoted(case):
    image, seed = faded_panel(), seed_polygon()
    if case == "none":
        seed = None
    elif case == "nonfinite":
        seed[0, 0] = np.nan
    elif case == "concave":
        seed[2] = (500, 500)
    elif case == "uint16":
        image = image.astype(np.uint16)
    elif case == "clipped":
        seed[:, 0] -= 180
    assert refine_panel_polygon_from_raw(image, seed) is seed


def test_capi_station_enables_refinement_for_training_and_inference():
    for training in (False, True):
        assert panel_boundary_config_for_station("capi", for_training=training)["raw_gradient_refinement_enabled"]
    assert not panel_boundary_config_for_station("aapi")["raw_gradient_refinement_enabled"]
    assert not PreprocessConfig().raw_gradient_refinement_enabled


def test_raw_and_legacy_entrypoints_use_original_pixels(tmp_path):
    raw = faded_panel()
    # A model-preprocessed image with a narrower border must not be used for refinement.
    processed = raw.copy()
    processed[:, 1187:] = 8
    cfg = config()
    legacy_cfg = replace(cfg, large_panel_raw_boundary_enabled=False)
    bbox, refined = detect_panel_geometry(raw, legacy_cfg, processed_image=processed)
    _, coarse = detect_panel_geometry(raw, replace(legacy_cfg, raw_gradient_refinement_enabled=False),
                                     processed_image=processed)
    assert right_x(coarse) < 1190
    assert 1198.5 < right_x(refined) < 1200
    raw_bbox, raw_refined = detect_panel_boundary(raw, cfg)
    assert 1198.5 < right_x(raw_refined) < 1200
    assert bbox is not None and raw_bbox is not None

    path = tmp_path / "W0F00000_001.png"
    assert cv2.imwrite(str(path), raw)
    folder = preprocess_panel_folder(tmp_path, cfg)["W0F00000"]
    single = preprocess_panel_image(path, "W0F00000", cfg)
    assert 1198.5 < right_x(folder.panel_polygon) < 1200
    np.testing.assert_array_equal(folder.panel_polygon, single.panel_polygon)
    disabled = preprocess_panel_image(path, "W0F00000", replace(cfg, raw_gradient_refinement_enabled=False))
    assert right_x(disabled.panel_polygon) < right_x(folder.panel_polygon) - 5
    assert folder.foreground_bbox == disabled.foreground_bbox


def test_shared_reference_is_not_refined_again_for_other_lighting(tmp_path):
    white = faded_panel()
    other = np.roll(white, 15, axis=1)
    assert cv2.imwrite(str(tmp_path / "W0F00000_001.png"), white)
    assert cv2.imwrite(str(tmp_path / "R0F00000_001.png"), other)
    results = preprocess_panel_folder(tmp_path, config())
    np.testing.assert_array_equal(results["W0F00000"].panel_polygon, results["R0F00000"].panel_polygon)
    assert 1198.5 < right_x(results["R0F00000"].panel_polygon) < 1200


def test_polygon_toggle_disables_refinement():
    _, polygon = detect_panel_geometry(faded_panel(), config(enable_panel_polygon=False))
    assert polygon is None


def test_small_product_keeps_existing_dim_band_boundary(monkeypatch):
    seed = seed_polygon()
    monkeypatch.setattr("capi_preprocess.detect_panel_polygon", lambda *a: ((200, 170, 1200, 930), seed))
    cfg = replace(config(), product_resolution=(1366, 768), large_panel_raw_boundary_enabled=False)
    _, polygon = detect_panel_geometry(faded_panel(), cfg)
    assert polygon is seed


def test_b0f_edge_spot_is_retained_while_outside_spot_remains_excluded():
    from capi_config import CAPIConfig
    from capi_inference import CAPIInferencer, TileInfo
    from capi_preprocess import classify_tile_zone

    polygon = refine_panel_polygon_from_raw(faded_panel(), seed_polygon())
    image = np.zeros((512, 512), np.uint8)
    origin = (944, 294)
    image[252:257, 248:253] = 24  # x=1192..1196, inside the dim product border.
    image[290:295, 270:275] = 40  # x=1214..1218, outside even after refinement.
    worker = object.__new__(CAPIInferencer)
    worker.config = CAPIConfig(bright_spot_diff_threshold=2, bright_spot_min_area=2,
                               bright_spot_threshold=30, bright_spot_median_kernel=37)
    outputs = []
    for boundary in (seed_polygon(), polygon):
        _, _, _, mask = classify_tile_zone((*origin, origin[0]+512, origin[1]+512), boundary, config())
        tile = TileInfo(tile_id=1, x=origin[0], y=origin[1], width=512, height=512, image=image, mask=mask)
        score, anomaly_map = worker._detect_bright_spots(tile)
        outputs.append(score)
        assert not np.any(anomaly_map[290:295, 270:275])
    assert outputs == [0., 1.]
