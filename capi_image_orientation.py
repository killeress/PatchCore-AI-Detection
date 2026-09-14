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
    # Keep each OpenCV decode mode separate, including 16-bit and EXIF behavior.
    key = (str(path), flags, bool(rotate_180))
    cached = cache["images"].pop(key, None)
    if cached is not None:
        cached_signature, cached_image = cached
        if signature is not None and signature == cached_signature:
            cache["images"][key] = cached
            cache["hits"] += 1
            # Callers may modify their image in place; never expose the cache.
            image = cached_image.copy()
            logger.info(
                "[image-io] source=%s flags=%s rotate_180=%s cache=hit total_ms=%.1f",
                path.name, flags, rotate_180, (time.perf_counter() - started) * 1000,
            )
            return image
        cache["bytes"] -= cached_image.nbytes

    cache["misses"] += 1
    read_started = time.perf_counter()
    image = cv2.imread(str(image_path), flags)
    read_ms = (time.perf_counter() - read_started) * 1000
    image = apply_detection_orientation(image, rotate_180)
    if image is not None and signature is not None and image.nbytes <= _IMAGE_CACHE_MAX_BYTES:
        while cache["images"] and cache["bytes"] + image.nbytes > _IMAGE_CACHE_MAX_BYTES:
            _, (_, evicted) = cache["images"].popitem(last=False)
            cache["bytes"] -= evicted.nbytes
        cache["images"][key] = (signature, image.copy())
        cache["bytes"] += image.nbytes
    logger.info(
        "[image-io] source=%s flags=%s rotate_180=%s cache=miss read_decode_ms=%.1f total_ms=%.1f",
        path.name, flags, rotate_180, read_ms, (time.perf_counter() - started) * 1000,
    )
    return image
