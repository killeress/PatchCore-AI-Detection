"""Product-pixel-grid canonicalization for training and inference images.

The transform uses the detected panel quadrilateral as a stable phase anchor,
resamples the rectified panel to a fixed number of camera samples per logical
product pixel, then projects it back to the original camera canvas.  Returning
the original image shape keeps the existing AOI/dust/result coordinate system
unchanged while making the pixels inside the panel deterministic.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


GRID_CANONICALIZATION_VERSION = 1
VALID_SAMPLES_PER_CELL = (1, 3)
DEFAULT_SAMPLES_PER_CELL = 3
MAX_CANONICAL_PIXELS = 80_000_000


@dataclass(frozen=True)
class GridCanonicalizationResult:
    image: np.ndarray
    product_resolution: Tuple[int, int]
    samples_per_cell: int
    canonical_size: Tuple[int, int]
    rectified_size: Tuple[int, int]


def normalize_product_resolution(value: Any) -> Optional[Tuple[int, int]]:
    """Normalize ``[width, height]``; return ``None`` for an absent value."""
    if value is None or value == "":
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("product_resolution must be [width, height]")
    if isinstance(value[0], bool) or isinstance(value[1], bool):
        raise ValueError("product_resolution must contain integers")
    try:
        width_value = float(value[0])
        height_value = float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("product_resolution must contain integers") from exc
    if (
        not math.isfinite(width_value)
        or not math.isfinite(height_value)
        or not width_value.is_integer()
        or not height_value.is_integer()
    ):
        raise ValueError("product_resolution must contain integers")
    width = int(width_value)
    height = int(height_value)
    if width <= 0 or height <= 0:
        raise ValueError("product_resolution must contain positive integers")
    return width, height


def validate_samples_per_cell(value: Any) -> int:
    """Accept only the two product-approved sampling densities: 1 or 3."""
    if isinstance(value, bool):
        raise ValueError("samples_per_cell must be 1 or 3")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("samples_per_cell must be 1 or 3") from exc
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError("samples_per_cell must be 1 or 3")
    samples = int(parsed)
    if samples not in VALID_SAMPLES_PER_CELL:
        raise ValueError("samples_per_cell must be 1 or 3")
    return samples


def normalize_grid_canonicalization(
    raw: Any,
    *,
    require_resolution: bool = False,
) -> Dict[str, Any]:
    """Return the persisted v1 grid-canonicalization contract."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("grid_canonicalization must be an object")

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("grid_canonicalization.enabled must be true or false")

    raw_version = raw.get("version", GRID_CANONICALIZATION_VERSION)
    if isinstance(raw_version, bool):
        raise ValueError("grid_canonicalization.version must be 1")
    try:
        parsed_version = float(raw_version)
    except (TypeError, ValueError) as exc:
        raise ValueError("grid_canonicalization.version must be 1") from exc
    if not math.isfinite(parsed_version) or not parsed_version.is_integer():
        raise ValueError("grid_canonicalization.version must be 1")
    version = int(parsed_version)
    if version != GRID_CANONICALIZATION_VERSION:
        raise ValueError(
            f"unsupported grid_canonicalization.version: {version}"
        )

    samples = validate_samples_per_cell(
        raw.get("samples_per_cell", DEFAULT_SAMPLES_PER_CELL)
    )
    resolution = normalize_product_resolution(raw.get("product_resolution"))
    if enabled and require_resolution and resolution is None:
        raise ValueError(
            "grid_canonicalization.product_resolution is required when enabled"
        )

    coordinate_preserving = raw.get("coordinate_preserving", True)
    if coordinate_preserving is not True:
        raise ValueError(
            "grid_canonicalization.coordinate_preserving=false is not supported"
        )

    return {
        "enabled": enabled,
        "version": GRID_CANONICALIZATION_VERSION,
        "samples_per_cell": samples,
        "product_resolution": list(resolution) if resolution else None,
        "coordinate_preserving": True,
    }


def _panel_rectified_size(polygon: np.ndarray) -> Tuple[int, int]:
    tl, tr, br, bl = polygon
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    # Polygon corners point at pixel centres, so an edge spanning 0..119
    # contains 120 source samples rather than 119.
    rectified_width = max(2, int(round(float(width))) + 1)
    rectified_height = max(2, int(round(float(height))) + 1)
    return rectified_width, rectified_height


def canonicalize_panel_grid(
    image: np.ndarray,
    panel_polygon: Sequence[Sequence[float]],
    product_resolution: Sequence[int],
    samples_per_cell: int = DEFAULT_SAMPLES_PER_CELL,
) -> GridCanonicalizationResult:
    """Canonicalize panel sampling while retaining the camera-image geometry.

    ``samples_per_cell`` is per axis.  A 1920x1080 product therefore uses a
    1920x1080 canonical plane at 1, or 5760x3240 at 3.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty numpy array")
    if image.ndim not in (2, 3):
        raise ValueError("image must be grayscale or color")

    polygon = np.asarray(panel_polygon, dtype=np.float32)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError("panel_polygon must be finite TL/TR/BR/BL coordinates")
    if abs(float(cv2.contourArea(polygon))) < 4.0:
        raise ValueError("panel_polygon area is too small")

    resolution = normalize_product_resolution(product_resolution)
    if resolution is None:  # pragma: no cover - guarded by normalizer
        raise ValueError("product_resolution is required")
    samples = validate_samples_per_cell(samples_per_cell)
    product_width, product_height = resolution
    canonical_width = product_width * samples
    canonical_height = product_height * samples
    if canonical_width * canonical_height > MAX_CANONICAL_PIXELS:
        raise ValueError(
            "canonical image is too large: "
            f"{canonical_width}x{canonical_height} exceeds "
            f"{MAX_CANONICAL_PIXELS} pixels"
        )

    rectified_width, rectified_height = _panel_rectified_size(polygon)
    rectified_corners = np.array(
        [
            [0, 0],
            [rectified_width - 1, 0],
            [rectified_width - 1, rectified_height - 1],
            [0, rectified_height - 1],
        ],
        dtype=np.float32,
    )
    to_rectified = cv2.getPerspectiveTransform(polygon, rectified_corners)
    rectified = cv2.warpPerspective(
        image,
        to_rectified,
        (rectified_width, rectified_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    interpolation = (
        cv2.INTER_AREA
        if canonical_width <= rectified_width
        and canonical_height <= rectified_height
        else cv2.INTER_LINEAR
    )
    canonical = cv2.resize(
        rectified,
        (canonical_width, canonical_height),
        interpolation=interpolation,
    )

    canonical_corners = np.array(
        [
            [0, 0],
            [canonical_width - 1, 0],
            [canonical_width - 1, canonical_height - 1],
            [0, canonical_height - 1],
        ],
        dtype=np.float32,
    )
    to_camera = cv2.getPerspectiveTransform(canonical_corners, polygon)
    height, width = image.shape[:2]
    restored = cv2.warpPerspective(
        canonical,
        to_camera,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
    output = image.copy()
    if image.ndim == 2:
        output[mask != 0] = restored[mask != 0]
    else:
        output[mask != 0, :] = restored[mask != 0, :]

    return GridCanonicalizationResult(
        image=output,
        product_resolution=resolution,
        samples_per_cell=samples,
        canonical_size=(canonical_width, canonical_height),
        rectified_size=(rectified_width, rectified_height),
    )
