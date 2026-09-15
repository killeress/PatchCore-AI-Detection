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


@pytest.mark.parametrize("guard", ["disabled", "grid", "canonical", "small_product", "no_polygon", "nan", "concave", "empty_bbox"])
def test_recovery_respects_mode_and_candidate_guards(tmp_path, monkeypatch, guard):
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
        generate_grid_tiles=guard == "grid",
        grid_canonicalization_enabled=guard == "canonical",
        product_resolution=(1366, 768) if guard == "small_product" else (1920, 1200),
    )
    actual = pre.preprocess_panel_folder(tmp_path, cfg)["W0F00000"]
    assert actual.panel_polygon is None
    assert actual.polygon_detection_failed is True
