"""OMIT exposure must describe the product, not its bright fixture/background."""
import os
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer
from capi_station_adapter import create_station_adapter


@pytest.fixture
def worker():
    inf = object.__new__(CAPIInferencer)
    inf.config = CAPIConfig(
        omit_overexposure_mean_threshold=82,
        omit_overexposure_ratio_threshold=0.05,
        tile_size=64,
    )
    inf.station_adapter = create_station_adapter("capi")
    return inf


def _diamond():
    polygon = np.array([[50, 20], [80, 50], [50, 80], [20, 50]], np.float32)
    mask = np.zeros((100, 100), np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return polygon, mask > 0


def test_bright_background_excluded_using_polygon_not_bbox(worker):
    polygon, inside = _diamond()
    image = np.full(inside.shape, 255, np.uint8)
    image[inside] = 45
    original = image.copy()
    assert worker.check_omit_overexposure(image)[0] is True
    overexposed, mean, ratio, detail = worker.check_omit_overexposure(image, polygon)
    assert (overexposed, mean, ratio) == (False, 45.0, 0.0)
    assert "Scope:product_polygon" in detail
    assert f"Pixels:{np.count_nonzero(inside)}" in detail
    np.testing.assert_array_equal(image, original)


def test_dark_background_cannot_dilute_real_product_overexposure(worker):
    polygon, inside = _diamond()
    image = np.zeros(inside.shape, np.uint8)
    image[inside] = 240
    assert worker.check_omit_overexposure(image)[0] is False
    overexposed, mean, ratio, _ = worker.check_omit_overexposure(image, polygon)
    assert (overexposed, mean, ratio) == (True, 240.0, 1.0)


@pytest.mark.parametrize("hot_pixels,base,expected", [(5, 100, False), (6, 100, True), (6, 0, False)])
def test_strict_ratio_limit_and_both_conditions_required(worker, hot_pixels, base, expected):
    image = np.full((10, 10), base, np.uint8)
    image.flat[:hot_pixels] = 231
    polygon = np.array([[0, 0], [9, 0], [9, 9], [0, 9]], np.float32)
    overexposed, _, ratio, _ = worker.check_omit_overexposure(image, polygon)
    assert overexposed is expected
    assert ratio == hot_pixels / 100


def test_bright_cutoff_is_strictly_above_230_for_color_images(worker):
    image = np.full((10, 10, 3), 230, np.uint8)
    polygon = np.array([[0, 0], [9, 0], [9, 9], [0, 9]], np.float32)
    assert worker.check_omit_overexposure(image, polygon)[:3] == (False, 230.0, 0.0)


@pytest.mark.parametrize("polygon", [
    None, [], [[1, 1], [2, 2]], [[1, 1], [2, 2], [3, 3]],
    [[float('nan'), 0], [10, 0], [10, 10]],
    [[200, 200], [220, 200], [220, 220], [200, 220]],
])
def test_missing_invalid_or_empty_polygon_preserves_full_image_fallback(worker, polygon):
    overexposed, mean, ratio, detail = worker.check_omit_overexposure(
        np.full((100, 100), 240, np.uint8), polygon,
    )
    assert (overexposed, mean, ratio) == (True, 240.0, 1.0)
    assert "Scope:full_image" in detail and "Fallback:" in detail


def test_missing_omit(worker):
    assert worker.check_omit_overexposure(None) == (False, 0.0, 0.0, "No OMIT image")


def _write_panel(folder):
    white = np.zeros((1024, 1280), np.uint8)
    white[200:850, 250:1050] = 100
    omit = np.full(white.shape, 255, np.uint8)
    omit[200:850, 250:1050] = 45
    paths = [folder / "W0F00000_134458.tif", folder / "PINIGBI _134451.tif"]
    for path, image in zip(paths, [white, omit]):
        assert cv2.imwrite(str(path), image)
    return paths, white, omit


@pytest.mark.parametrize("rotate", [False, True])
def test_context_load_uses_white_polygon_and_same_orientation(worker, tmp_path, rotate):
    paths, _, omit = _write_panel(tmp_path)
    worker.config.inference_rotate_180_enabled = rotate
    _, overexposed, detail, loaded = worker._load_omit_context(tmp_path, image_files=paths)
    assert not overexposed
    assert "Scope:product_polygon" in detail
    assert "Reference:W0F00000_134458.tif" in detail
    np.testing.assert_array_equal(loaded, cv2.rotate(omit, cv2.ROTATE_180) if rotate else omit)


def test_white_reference_selected_before_other_lighting_and_latest_retake(worker, tmp_path):
    paths, white, omit = _write_panel(tmp_path)
    old = tmp_path / "W0F00000_130000.tif"
    green = tmp_path / "G0F00000_140000.tif"
    for path in [old, green]:
        assert cv2.imwrite(str(path), white)
    sidecar = paths[0].with_suffix(".json")
    sidecar.write_text("{}", encoding="utf-8")
    os.utime(old, (1000, 1000))
    os.utime(paths[0], (2000, 2000))
    os.utime(green, (3000, 3000))
    with patch.object(worker, "_read_detection_image", wraps=worker._read_detection_image) as read:
        outcome = worker.check_omit_overexposure(omit, image_files=[green, old, sidecar, *paths])
    assert not outcome[0]
    assert "Reference:W0F00000_134458.tif" in outcome[3]
    assert [call.args[0] for call in read.call_args_list] == [paths[0]]


@pytest.mark.parametrize("failure", ["missing", "mismatch", "no_polygon"])
def test_unusable_reference_fallback_is_visible(worker, tmp_path, failure):
    paths, white, omit = _write_panel(tmp_path)
    if failure == "missing":
        paths = paths[1:]
    elif failure == "mismatch":
        assert cv2.imwrite(str(paths[0]), white[:100, :100])
    else:
        assert cv2.imwrite(str(paths[0]), np.zeros_like(white))
    outcome = worker.check_omit_overexposure(omit, image_files=paths)
    assert outcome[0]
    assert "Scope:full_image" in outcome[3] and "Fallback:" in outcome[3]
    if failure == "mismatch":
        assert "reference_size_mismatch" in outcome[3]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_production_paths_pass_reference_files_to_exposure_check(worker, tmp_path, monkeypatch, version):
    _write_panel(tmp_path)
    worker.config.aoi_coord_inspection_enabled = False
    worker.config.exclusion_zones = []
    worker.config.should_skip_file = lambda *args: True
    monkeypatch.setattr(worker, "_detect_panel_mark_binary_region", lambda *a, **k: (None, []))
    with patch("capi_preprocess.preprocess_panel_folder", return_value={}):
        result = getattr(worker, f"_process_panel_{version}")(tmp_path)
    assert not result[2]
    assert "Scope:product_polygon" in result[3]
    assert "Reference:W0F00000_134458.tif" in result[3]
