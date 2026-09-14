from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace
import os

import cv2
import numpy as np
import pytest

import capi_image_orientation as image_io


@pytest.mark.parametrize("dtype,channels", [(np.uint8, 1), (np.uint16, 1), (np.uint8, 3), (np.uint16, 3), (np.uint8, 4)])
@pytest.mark.parametrize("rotate", [False, True])
def test_cache_preserves_decode_modes_rotation_and_mutable_callers(tmp_path, monkeypatch, dtype, channels, rotate):
    path = tmp_path / "panel.tif"
    shape = (32, 48) if channels == 1 else (32, 48, channels)
    source = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    assert cv2.imwrite(str(path), source, [259, 1, 278, 1])
    flags = [cv2.IMREAD_UNCHANGED, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR]
    expected = {flag: image_io.read_detection_image(path, flag, rotate) for flag in flags}
    original_read = cv2.imread
    original_decode = cv2.imdecode
    calls = []
    decodes = []

    def read(*args):
        calls.append(args)
        return original_read(*args)

    monkeypatch.setattr(image_io.cv2, "imread", read)
    def decode(*args):
        decodes.append(args[1])
        return original_decode(*args)

    monkeypatch.setattr(image_io.cv2, "imdecode", decode)
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
    buffered = dtype == np.uint8 and channels == 1
    assert len(calls) == (0 if buffered else 3)
    assert decodes == ([cv2.IMREAD_UNCHANGED] if buffered else [])
    image_io.read_detection_image(path, flags[0], rotate)
    assert len(calls) == (1 if buffered else 4)


@pytest.mark.parametrize("failure", ["decode_none", "decode_error", "read_error", "oversize"])
def test_buffered_tiff_falls_back_to_original_reader(tmp_path, monkeypatch, failure):
    from pathlib import Path

    path = tmp_path / "panel.tif"
    source = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert cv2.imwrite(str(path), source, [259, 1, 278, 1])
    if failure == "decode_none":
        monkeypatch.setattr(cv2, "imdecode", lambda *a: None)
    elif failure == "decode_error":
        def fail_decode(*a):
            raise cv2.error("memory decode unavailable")
        monkeypatch.setattr(cv2, "imdecode", fail_decode)
    elif failure == "read_error":
        def fail_open(*a, **k):
            raise OSError("buffer read unavailable")
        monkeypatch.setattr(Path, "open", fail_open)
    else:
        monkeypatch.setattr(image_io, "_TIFF_BUFFER_MAX_BYTES", 1)
        monkeypatch.setattr(cv2, "imdecode", lambda *a: pytest.fail("oversized TIFF was buffered"))
    with image_io.panel_image_cache(enabled=True):
        actual = image_io.read_detection_image(path, cv2.IMREAD_UNCHANGED, False)
    np.testing.assert_array_equal(actual, source)


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


@pytest.mark.parametrize("first_flag", [cv2.IMREAD_UNCHANGED, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR])
@pytest.mark.parametrize("rotate", [False, True])
def test_single_row_tiff_shares_one_gray_copy_across_inference_and_within_spec(
    tmp_path, monkeypatch, first_flag, rotate,
):
    path = tmp_path / "panel.tif"
    source = np.arange(32 * 48, dtype=np.uint8).reshape(32, 48)
    assert cv2.imwrite(str(path), source, [259, 1, 278, 1])
    flags = [first_flag, cv2.IMREAD_UNCHANGED, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR]
    expected = {flag: image_io.read_detection_image(path, flag, rotate) for flag in flags}
    with image_io.panel_image_cache(enabled=True):
        for flag in flags:
            actual = image_io.read_detection_image(path, flag, rotate)
            np.testing.assert_array_equal(actual, expected[flag])
            actual[:] = 0
        cache = image_io._panel_image_cache.get()
        assert cache["misses"] == 1
        assert cache["hits"] == 3
        assert cache["bytes"] == source.nbytes
        assert len(cache["images"]) == 1


def test_gray_mode_alias_is_invalidated_when_file_changes_format(tmp_path, monkeypatch):
    path = tmp_path / "panel.tif"
    source = np.arange(32 * 48, dtype=np.uint8).reshape(32, 48)
    assert cv2.imwrite(str(path), source, [259, 1, 278, 1])
    with image_io.panel_image_cache(enabled=True):
        image_io.read_detection_image(path, cv2.IMREAD_UNCHANGED, False)
        stamp = path.stat().st_mtime_ns
        color = np.dstack([source, np.flip(source, axis=1), source // 2])
        assert cv2.imwrite(str(path), color, [259, 1, 278, 1])
        os.utime(path, ns=(stamp + 1000000000, stamp + 1000000000))
        for flag in (cv2.IMREAD_UNCHANGED, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR):
            actual = image_io.read_detection_image(path, flag, False)
            np.testing.assert_array_equal(actual, cv2.imread(str(path), flag))
        cache = image_io._panel_image_cache.get()
        assert cache["bytes"] == sum(entry[1].nbytes for entry in cache["images"].values())


@pytest.mark.parametrize("orientation", range(1, 9))
def test_buffered_tiff_preserves_file_orientation_and_reduced_decode(tmp_path, orientation):
    import struct

    path = tmp_path / "oriented.tif"
    source = np.arange(32 * 48, dtype=np.uint8).reshape(32, 48)
    assert cv2.imwrite(str(path), source, [259, 1, 278, 1])
    # Append a replacement directory, retaining the existing strip offsets.
    data = bytearray(path.read_bytes())
    order = "<" if data[:2] == b"II" else ">"
    offset = struct.unpack_from(order + "I", data, 4)[0]
    count = struct.unpack_from(order + "H", data, offset)[0]
    entries = [bytes(data[offset + 2 + i * 12:offset + 14 + i * 12]) for i in range(count)]
    entries = [entry for entry in entries if struct.unpack_from(order + "H", entry)[0] != 274]
    entries.append(struct.pack(order + "HHIH", 274, 3, 1, orientation) + b"\x00\x00")
    entries.sort(key=lambda entry: struct.unpack_from(order + "H", entry)[0])
    struct.pack_into(order + "I", data, 4, len(data))
    data.extend(struct.pack(order + "H", len(entries)) + b"".join(entries) + b"\x00" * 4)
    path.write_bytes(data)
    with path.open("rb") as stream:
        assert image_io._is_single_row_gray_tiff(stream) is (orientation == 1)
    for flag in (cv2.IMREAD_UNCHANGED, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR,
                 cv2.IMREAD_REDUCED_GRAYSCALE_2, cv2.IMREAD_IGNORE_ORIENTATION):
        expected = cv2.imread(str(path), flag)
        with image_io.panel_image_cache(enabled=True):
            actual = image_io.read_detection_image(path, flag, False)
        np.testing.assert_array_equal(actual, expected)


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
