"""Orientation correction for images entering detection."""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

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
