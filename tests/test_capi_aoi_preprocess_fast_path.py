"""CAPI AOI optimizations must preserve pixels and failed-boundary semantics."""
from dataclasses import replace

import cv2
import numpy as np
import pytest

import capi_preprocess as pre


@pytest.mark.parametrize("mode", ["keep", "to_high"])
@pytest.mark.parametrize("dtype,channels", [(np.uint8, 1), (np.uint8, 3), (np.uint16, 1)])
def test_boundary_lut_matches_original_for_every_gray_value(mode, dtype, channels, monkeypatch):
    image = np.tile(np.arange(256, dtype=dtype), (32, 1))
    if dtype == np.uint16:
        image *= 257
    if channels == 3:
        image = np.dstack([image, np.flip(image, axis=1), image // 2])
    image = image[::2]  # Noncontiguous views occur after cropping.
    cfg = pre.PreprocessConfig(image_preprocess_pipeline=[{
        "method": "gray_band_shift",
        "params": {"low_threshold": 100, "high_threshold": 160,
                   "dark_shift": 80, "bright_shift": 90, "band_mode": mode},
    }])
    expected = pre._boundary_detection_gray(image, cfg)
    stats_shapes = []
    import capi_image_preprocess_lab as lab
    original_stats = lab.image_stats

    def stats(original, processed):
        stats_shapes.append(original.shape)
        return original_stats(original, processed)

    monkeypatch.setattr(lab, "image_stats", stats)
    actual = pre._boundary_detection_gray(image, replace(cfg, aoi_only_fast_path_enabled=True))
    np.testing.assert_array_equal(actual, expected)
    if dtype == np.uint8:
        assert stats_shapes == [(256, 1)]


@pytest.mark.parametrize("success_at,generate_grid,canonical", [
    (None, False, False), (2, False, False), (None, True, False),
    (2, True, False), (None, False, True),
])
def test_failed_candidates_reused_only_without_a_new_reference(
    tmp_path, monkeypatch, success_at, generate_grid, canonical,
):
    prefixes = list(pre.BOUNDARY_REFERENCE_PRIORITY)
    for idx, prefix in enumerate(prefixes):
        assert cv2.imwrite(str(tmp_path / f"{prefix}_001.png"), np.full((768, 1024), idx + 1, np.uint8))
    calls = []
    polygon = np.array([[20, 20], [1000, 20], [1000, 740], [20, 740]], np.float32)

    def detect(image, config):
        marker = int(image[0, 0]) - 1
        calls.append(marker)
        return (20, 20, 1000, 740), polygon if marker == success_at and config.enable_panel_polygon else None

    monkeypatch.setattr(pre, "detect_panel_polygon", detect)
    cfg = pre.PreprocessConfig(
        generate_grid_tiles=generate_grid, grid_canonicalization_enabled=canonical,
        cache_processed_image=True,
    )
    before = pre.preprocess_panel_folder(tmp_path, cfg)
    before_calls = len(calls)
    calls.clear()
    after = pre.preprocess_panel_folder(tmp_path, replace(cfg, aoi_only_fast_path_enabled=True))
    if success_at is None and not generate_grid and not canonical:
        assert before_calls == 9
        assert len(calls) == 5
    else:
        assert len(calls) == before_calls
    assert list(after) == list(before)
    for lighting, old in before.items():
        new = after[lighting]
        assert old.foreground_bbox == new.foreground_bbox
        assert old.polygon_detection_failed == new.polygon_detection_failed
        np.testing.assert_array_equal(old.panel_polygon, new.panel_polygon)
        np.testing.assert_array_equal(old.processed_image, new.processed_image)
        assert len(old.tiles) == len(new.tiles)
        for old_tile, new_tile in zip(old.tiles, new.tiles):
            assert old_tile.zone == new_tile.zone
            np.testing.assert_array_equal(old_tile.image, new_tile.image)
            np.testing.assert_array_equal(old_tile.mask, new_tile.mask)


def test_failed_candidate_is_reread_when_source_changes(tmp_path, monkeypatch):
    import os

    prefixes = ["W0F00000", "STANDARD", "G0F00000"]
    paths = {p: tmp_path / f"{p}_001.png" for p in prefixes}
    for path in paths.values():
        assert cv2.imwrite(str(path), np.ones((32, 32), np.uint8))
    original_process = pre.preprocess_panel_image
    calls = []

    def process(path, lighting, config, **kwargs):
        calls.append(lighting)
        if lighting == "G0F00000":
            stamp = paths["STANDARD"].stat().st_mtime_ns
            cv2.imwrite(str(paths["STANDARD"]), np.full((32, 32), 99, np.uint8))
            os.utime(paths["STANDARD"], ns=(stamp + 1000000000, stamp + 1000000000))
        return original_process(path, lighting, config, **kwargs)

    monkeypatch.setattr(pre, "preprocess_panel_image", process)
    monkeypatch.setattr(pre, "detect_panel_polygon", lambda *a: ((0, 0, 32, 32), None))
    results = pre.preprocess_panel_folder(tmp_path, pre.PreprocessConfig(
        aoi_only_fast_path_enabled=True, generate_grid_tiles=False, cache_processed_image=True,
    ))
    assert calls.count("STANDARD") == 2
    assert calls.count("G0F00000") == 1
    assert results["STANDARD"].processed_image[0, 0] == 99
