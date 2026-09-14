"""Orientation correction for images entering detection."""

from pathlib import Path
from typing import Optional, Union
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import time
import threading

import cv2
import numpy as np

logger = logging.getLogger("capi.image_io")
_IMAGE_CACHE_MAX_BYTES = 256 * 1024 * 1024
_TIFF_BUFFER_MAX_BYTES = 64 * 1024 * 1024
_NATIVE_GRAY_MODE = "single-row-gray-tiff"
_panel_image_cache = ContextVar("panel_image_cache", default=None)


def use_capi_aoi_fast_path(config, station_profile: str) -> bool:
    return bool(
        station_profile == "capi"
        and getattr(config, "is_new_architecture", False)
        and getattr(config, "aoi_coord_inspection_enabled", False)
        and not getattr(config, "grid_tiling_enabled", True)
    )


@contextmanager
def panel_image_cache(*, enabled: bool, panel: str = ""):
    """Bound image reuse to one request; nested inference shares its server cache."""
    if not enabled or _panel_image_cache.get() is not None:
        yield
        return
    cache = {"images": OrderedDict(), "bytes": 0, "hits": 0, "misses": 0, "owner": threading.get_ident()}
    token = _panel_image_cache.set(cache)
    try:
        yield
    finally:
        _panel_image_cache.reset(token)
        logger.info(
            "[image-io] panel=%s hits=%d misses=%d retained_mb=%.1f",
            panel, cache["hits"], cache["misses"], cache["bytes"] / (1024 * 1024),
        )
        cache["images"].clear()


@contextmanager
def timed_inference_stage(stage: str):
    """Detailed timing is enabled only inside the optimized request scope."""
    if _panel_image_cache.get() is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        logger.info("[stage] %s elapsed_ms=%.1f", stage, (time.perf_counter() - started) * 1000)


def apply_detection_orientation(
    image: Optional[np.ndarray],
    rotate_180: bool,
) -> Optional[np.ndarray]:
    if image is None or not rotate_180:
        return image
    return cv2.rotate(image, cv2.ROTATE_180)


def _read_cached_source(path: Path, flags: int, signature):
    """Avoid per-strip filesystem seeks on TIFFs from network storage.

    Used only for uncompressed, single-row, 8-bit gray TIFFs in the CAPI AOI
    request cache. Other encodings/orientations retain the original reader.
    """
    if (
        path.suffix.lower() in (".tif", ".tiff")
        and signature is not None
        and 0 < signature[0] <= _TIFF_BUFFER_MAX_BYTES
        and flags in (cv2.IMREAD_UNCHANGED, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR)
    ):
        try:
            with path.open("rb") as stream:
                if _is_single_row_gray_tiff(stream):
                    stream.seek(0)
                    encoded = stream.read(_TIFF_BUFFER_MAX_BYTES + 1)
                else:
                    encoded = b""
            if len(encoded) == signature[0]:
                decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if decoded is not None and decoded.dtype == np.uint8 and decoded.ndim == 2:
                    return decoded, "buffered-tiff"
        except (OSError, cv2.error):
            pass
    return cv2.imread(str(path), flags), "opencv"


def _is_single_row_gray_tiff(stream) -> bool:
    """Read only the classic TIFF directory, with no new runtime dependency.

    Narrow eligibility intentionally excludes orientation transforms: OpenCV
    imread and imdecode can differ for rotated TIFFs on some versions.
    """
    header = stream.read(8)
    if len(header) != 8 or header[:4] not in (b"II\x2a\x00", b"MM\x00\x2a"):
        return False
    order = "little" if header[:2] == b"II" else "big"
    offset = int.from_bytes(header[4:8], order)
    if offset < 8:
        return False
    stream.seek(offset)
    count_bytes = stream.read(2)
    count = int.from_bytes(count_bytes, order)
    if len(count_bytes) != 2 or not 1 <= count <= 256:
        return False
    entries = stream.read(count * 12)
    if len(entries) != count * 12:
        return False
    # BitsPerSample, Compression, Photometric, Orientation, SamplesPerPixel,
    # RowsPerStrip and SampleFormat. Missing optional tags use TIFF defaults.
    required = {258: 8, 259: 1, 262: 1, 274: 1, 277: 1, 278: 1, 339: 1}
    values = {259: 1, 274: 1, 277: 1, 339: 1}
    for start in range(0, len(entries), 12):
        entry = entries[start:start + 12]
        tag = int.from_bytes(entry[:2], order)
        if tag not in required:
            continue
        kind = int.from_bytes(entry[2:4], order)
        items = int.from_bytes(entry[4:8], order)
        if items != 1 or kind not in (3, 4):
            return False
        values[tag] = int.from_bytes(entry[8:10 if kind == 3 else 12], order)
    return all(values.get(tag) == value for tag, value in required.items())


def read_detection_image(
    image_path: Union[str, Path],
    flags: int,
    rotate_180: bool,
) -> Optional[np.ndarray]:
    cache = _panel_image_cache.get()
    if cache is None or cache["owner"] != threading.get_ident():
        image = cv2.imread(str(image_path), flags)
        return apply_detection_orientation(image, rotate_180)

    started = time.perf_counter()
    path = Path(image_path)
    try:
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        signature = None
    # Only the validated 8-bit gray TIFF format can share decode modes. Keep
    # all other encodings separate, including 16-bit and EXIF behavior.
    key = (str(path), flags, bool(rotate_180))
    native_gray_key = (str(path), _NATIVE_GRAY_MODE, bool(rotate_180))
    cached = cache["images"].pop(key, None)
    if cached is None and flags in (cv2.IMREAD_UNCHANGED, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR):
        cached = cache["images"].pop(native_gray_key, None)
        if cached is not None:
            key = native_gray_key
    if cached is not None:
        cached_signature, cached_image = cached
        if signature is not None and signature == cached_signature:
            cache["images"][key] = cached
            cache["hits"] += 1
            # Callers may modify their image in place; never expose the cache.
            image = (
                cv2.cvtColor(cached_image, cv2.COLOR_GRAY2BGR)
                if key == native_gray_key and flags == cv2.IMREAD_COLOR
                else cached_image.copy()
            )
            logger.info(
                "[image-io] source=%s flags=%s rotate_180=%s cache=hit total_ms=%.1f",
                path.name, flags, rotate_180, (time.perf_counter() - started) * 1000,
            )
            return image
        cache["bytes"] -= cached_image.nbytes

    cache["misses"] += 1
    read_started = time.perf_counter()
    image, reader = _read_cached_source(path, flags, signature)
    read_ms = (time.perf_counter() - read_started) * 1000
    image = apply_detection_orientation(image, rotate_180)
    key = native_gray_key if reader == "buffered-tiff" else (str(path), flags, bool(rotate_180))
    if image is not None and signature is not None and image.nbytes <= _IMAGE_CACHE_MAX_BYTES:
        replaced = cache["images"].pop(key, None)
        if replaced is not None:
            cache["bytes"] -= replaced[1].nbytes
        while cache["images"] and cache["bytes"] + image.nbytes > _IMAGE_CACHE_MAX_BYTES:
            _, (_, evicted) = cache["images"].popitem(last=False)
            cache["bytes"] -= evicted.nbytes
        cache["images"][key] = (signature, image.copy())
        cache["bytes"] += image.nbytes
    if reader == "buffered-tiff" and flags == cv2.IMREAD_COLOR:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    logger.info(
        "[image-io] source=%s flags=%s rotate_180=%s cache=miss reader=%s read_decode_ms=%.1f total_ms=%.1f",
        path.name, flags, rotate_180, reader, read_ms, (time.perf_counter() - started) * 1000,
    )
    return image
