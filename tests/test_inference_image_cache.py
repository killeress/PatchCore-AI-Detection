from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace
import os

import cv2
import numpy as np
import pytest

import capi_image_orientation as image_io


@pytest.mark.parametrize("dtype,channels", [(np.uint8, 1), (np.uint16, 1), (np.uint8, 3)])
@pytest.mark.parametrize("rotate", [False, True])
def test_cache_preserves_decode_modes_rotation_and_mutable_callers(tmp_path, monkeypatch, dtype, channels, rotate):
    path = tmp_path / "panel.tif"
    shape = (32, 48) if channels == 1 else (32, 48, channels)
    source = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    assert cv2.imwrite(str(path), source)
    flags = [cv2.IMREAD_UNCHANGED, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR]
    expected = {flag: image_io.read_detection_image(path, flag, rotate) for flag in flags}
    original_read = cv2.imread
    calls = []

    def read(*args):
        calls.append(args)
        return original_read(*args)

    monkeypatch.setattr(image_io.cv2, "imread", read)
    with image_io.panel_image_cache(enabled=True):
        for flag in flags:
            first = image_io.read_detection_image(path, flag, rotate)
            np.testing.assert_array_equal(first, expected[flag])
            first[:] = 0
        for flag in flags:
            second = image_io.read_detection_image(path, flag, rotate)
            np.testing.assert_array_equal(second, expected[flag])
            second[:] = 1
            np.testing.assert_array_equal(image_io.read_detection_image(path, flag, rotate), expected[flag])
    assert len(calls) == 3
    image_io.read_detection_image(path, flags[0], rotate)
    assert len(calls) == 4


def test_cache_expires_on_file_change_and_exception(tmp_path):
    path = tmp_path / "panel.png"
    assert cv2.imwrite(str(path), np.full((8, 8), 10, np.uint8))
    with pytest.raises(RuntimeError), image_io.panel_image_cache(enabled=True):
        first = image_io.read_detection_image(path, cv2.IMREAD_UNCHANGED, False)
        stamp = path.stat().st_mtime_ns
        assert cv2.imwrite(str(path), np.full((8, 8), 20, np.uint8))
        os.utime(path, ns=(stamp + 1000000000, stamp + 1000000000))
        second = image_io.read_detection_image(path, cv2.IMREAD_UNCHANGED, False)
        assert first[0, 0] == 10
        assert second[0, 0] == 20
        raise RuntimeError("failed request")
    assert image_io._panel_image_cache.get() is None


def test_cache_is_bounded_and_nested_scopes_share_it(tmp_path, monkeypatch):
    monkeypatch.setattr(image_io, "_IMAGE_CACHE_MAX_BYTES", 64)
    paths = [tmp_path / f"{i}.png" for i in range(3)]
    for i, path in enumerate(paths):
        assert cv2.imwrite(str(path), np.full((8, 8), i, np.uint8))
    with image_io.panel_image_cache(enabled=True):
        cache = image_io._panel_image_cache.get()
        for path in paths:
            image_io.read_detection_image(path, cv2.IMREAD_UNCHANGED, False)
            assert cache["bytes"] <= 64
        with image_io.panel_image_cache(enabled=True):
            image_io.read_detection_image(paths[-1], cv2.IMREAD_UNCHANGED, False)
        assert cache["hits"] == 1
        assert cache["misses"] == 3
        assert image_io._panel_image_cache.get() is cache
    assert not cache["images"]


def test_cache_is_not_shared_with_background_threads(tmp_path):
    path = tmp_path / "panel.png"
    assert cv2.imwrite(str(path), np.zeros((8, 8), np.uint8))
    with image_io.panel_image_cache(enabled=True), ThreadPoolExecutor(max_workers=1) as pool:
        image_io.read_detection_image(path, cv2.IMREAD_UNCHANGED, False)
        assert pool.submit(image_io._panel_image_cache.get).result() is None
        cache = image_io._panel_image_cache.get()
        pool.submit(copy_context().run, image_io.read_detection_image, path, cv2.IMREAD_UNCHANGED, False).result()
        assert cache["hits"] == 0
        assert cache["misses"] == 1


@pytest.mark.parametrize("profile,new_arch,aoi,grid,expected", [
    ("capi", True, True, False, True),
    ("capi", False, True, False, False),
    ("capi", True, True, True, False),
    ("capi", True, False, False, False),
    ("aapi", True, True, False, False),
    ("dapi", True, True, False, False),
])
def test_fast_path_leaves_other_station_modes_unchanged(profile, new_arch, aoi, grid, expected):
    config = SimpleNamespace(
        is_new_architecture=new_arch, aoi_coord_inspection_enabled=aoi,
        grid_tiling_enabled=grid,
    )
    assert image_io.use_capi_aoi_fast_path(config, profile) is expected
