"""Host-specific orientation correction for images entering detection."""

import socket
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


ROTATE_180_HOSTNAME = "capi13"


def requires_detection_rotation(hostname: Optional[str] = None) -> bool:
    """Return whether detection images must be rotated for this exact host."""
    active_hostname = socket.gethostname() if hostname is None else hostname
    return str(active_hostname).strip().casefold() == ROTATE_180_HOSTNAME


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
    image = cv2.imread(str(image_path), flags)
    return apply_detection_orientation(image, rotate_180)
