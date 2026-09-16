"""Recover a validated raw reference without replacing working boundaries."""
from dataclasses import replace

import cv2
import numpy as np
import pytest

import capi_preprocess as pre


def padded_panel(*, noisy_edge):
    image = np.zeros((1536, 2048), np.uint8)
    image[330:1290, 270:1770] = 160
    if noisy_edge:
        # Thin bright edge texture affects legacy first-foreground samples;
        # the existing raw downscale/blur path removes it before edge fitting.
        for y in range(335, 1285, 5):
            shift = abs(int(20 * np.sin(y * 0.4)))
            image[y, 270 - shift:271] = 160
    return image


@pytest.mark.parametrize("working_reference", [False, True])
@pytest.mark.parametrize("profile", ["capi", "aapi"])
def test_padded_panel_recovers_only_after_all_legacy_references_fail(tmp_path, working_reference, profile):
    prefixes = ["W0F00000", "STANDARD", "G0F00000"]
    for prefix in prefixes:
        image = padded_panel(noisy_edge=not (working_reference and prefix == "STANDARD"))
        assert cv2.imwrite(str(tmp_path / f"{prefix}_001.png"), image)
    cfg = pre.PreprocessConfig(
        tile_size=256, generate_grid_tiles=False, cache_processed_image=True,
        product_resolution=(1920, 1200), aoi_only_fast_path_enabled=profile == "capi",
        **pre.panel_boundary_config_for_station(profile),
    )
    _bbox, raw_polygon, eligible = pre._detect_large_panel_raw_boundary(padded_panel(noisy_edge=True), cfg)
    assert raw_polygon is not None
    assert eligible is False
    before = pre.preprocess_panel_folder(tmp_path, cfg)
    after = pre.preprocess_panel_folder(tmp_path, replace(cfg, recover_failed_raw_boundary=True))
    for prefix in prefixes:
        old, new = before[prefix], after[prefix]
        assert old.polygon_detection_failed is (not working_reference)
        assert new.polygon_detection_failed is False
        np.testing.assert_array_equal(new.panel_polygon, old.panel_polygon if working_reference else raw_polygon)
        assert new.foreground_bbox == old.foreground_bbox
        np.testing.assert_array_equal(new.processed_image, old.processed_image)
        assert old.tiles == new.tiles == []
    if not working_reference:
        assert not np.shares_memory(after["W0F00000"].panel_polygon, after["STANDARD"].panel_polygon)


@pytest.mark.parametrize("guard", ["disabled", "polygon_disabled", "small_product", "no_polygon", "nan", "concave", "empty_bbox"])
@pytest.mark.parametrize("mode", ["aoi", "tiles", "canonical"])
def test_recovery_respects_mode_and_candidate_guards(tmp_path, monkeypatch, guard, mode):
    path = tmp_path / "W0F00000_001.png"
    path.touch()
    polygon = np.array([[100, 100], [800, 100], [800, 800], [100, 800]], np.float32)
    if guard == "no_polygon":
        polygon = None
    elif guard == "nan":
        polygon[0, 0] = np.nan
    elif guard == "concave":
        polygon[2] = [300, 300]
    result = pre.PanelPreprocessResult(
        path, "W0F00000", (0, 0, 0, 0) if guard == "empty_bbox" else (100, 100, 800, 800),
        None, polygon_detection_failed=True,
    )
    monkeypatch.setattr(pre, "_detect_large_panel_boundary_file", lambda *a: ((100, 100, 800, 800), polygon, False))
    monkeypatch.setattr(pre, "_preprocess_panel_folder_legacy", lambda *a: {"W0F00000": result})
    cfg = pre.PreprocessConfig(
        large_panel_raw_boundary_enabled=True,
        recover_failed_raw_boundary=guard != "disabled",
        generate_grid_tiles=mode != "aoi",
        grid_canonicalization_enabled=mode == "canonical",
        enable_panel_polygon=guard != "polygon_disabled",
        product_resolution=(1366, 768) if guard == "small_product" else (1920, 1200),
    )
    actual = pre.preprocess_panel_folder(tmp_path, cfg)["W0F00000"]
    assert actual.panel_polygon is None
    assert actual.polygon_detection_failed is True


def assert_same_tiles(actual, expected):
    assert len(actual) == len(expected) > 0
    for a, e in zip(actual, expected):
        for attr in ("x1", "y1", "x2", "y2", "zone", "coverage", "center_dist_to_edge", "is_corner"):
            assert getattr(a, attr) == getattr(e, attr), attr
        for attr in ("image", "original_image", "mask"):
            np.testing.assert_array_equal(getattr(a, attr), getattr(e, attr))


@pytest.mark.parametrize("profile", ["capi", "aapi"])
@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("after_tiling", [False, True])
def test_training_and_single_image_preview_rebuild_tiles_with_recovered_boundary(
    tmp_path, profile, canonical, after_tiling,
):
    path = tmp_path / "W0F00000_001.png"
    raw = padded_panel(noisy_edge=True)
    assert cv2.imwrite(str(path), raw)
    cfg = pre.PreprocessConfig(
        **pre.panel_boundary_config_for_station(profile, for_training=True),
        tile_size=256, tile_stride=256, cache_processed_image=True,
        product_resolution=(1920, 1200), grid_samples_per_cell=1,
        grid_canonicalization_enabled=canonical,
        preprocess_after_tiling=after_tiling,
        image_preprocess_pipeline=[{"method": "gray_band_shift", "params": {
            "low_threshold": 105, "high_threshold": 110,
            "dark_shift": 0, "bright_shift": 20, "band_mode": "keep",
        }}],
    )
    _, raw_polygon, eligible = pre._detect_large_panel_raw_boundary(raw, cfg)
    assert raw_polygon is not None and eligible is False
    old = pre.preprocess_panel_folder(tmp_path, replace(cfg, recover_failed_raw_boundary=False))["W0F00000"]
    assert old.polygon_detection_failed
    if canonical:
        assert not old.tiles
    expected = pre.preprocess_panel_image(
        path, "W0F00000", replace(cfg, large_panel_raw_boundary_enabled=False),
        reference_polygon=raw_polygon,
    )
    assert {tile.zone for tile in expected.tiles} == {"inner", "edge"}
    folder = pre.preprocess_panel_folder(tmp_path, cfg)["W0F00000"]
    preview = pre.preprocess_panel_image(path, "W0F00000", cfg)
    for result in (folder, preview):
        assert not result.polygon_detection_failed
        np.testing.assert_array_equal(result.panel_polygon, raw_polygon)
        assert result.foreground_bbox == old.foreground_bbox
        np.testing.assert_array_equal(result.processed_image, expected.processed_image)
        assert_same_tiles(result.tiles, expected.tiles)
    bbox, polygon = pre.detect_panel_geometry(raw, cfg)
    assert bbox == old.foreground_bbox
    np.testing.assert_array_equal(polygon, raw_polygon)


@pytest.mark.parametrize("canonical", [False, True])
def test_training_preserves_successful_secondary_reference(tmp_path, canonical):
    for prefix, noisy in (("W0F00000", True), ("STANDARD", False)):
        assert cv2.imwrite(str(tmp_path / f"{prefix}_001.png"), padded_panel(noisy_edge=noisy))
    cfg = pre.PreprocessConfig(
        **pre.panel_boundary_config_for_station("aapi", for_training=True),
        tile_size=256, tile_stride=256, product_resolution=(1920, 1200),
        grid_canonicalization_enabled=canonical, grid_samples_per_cell=1,
        cache_processed_image=True,
    )
    before = pre.preprocess_panel_folder(tmp_path, replace(cfg, recover_failed_raw_boundary=False))
    after = pre.preprocess_panel_folder(tmp_path, cfg)
    for prefix, result in after.items():
        assert not result.polygon_detection_failed
        np.testing.assert_array_equal(result.panel_polygon, before[prefix].panel_polygon)
        np.testing.assert_array_equal(result.processed_image, before[prefix].processed_image)
        assert_same_tiles(result.tiles, before[prefix].tiles)
