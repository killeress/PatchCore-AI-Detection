"""AAPI dust images with and without the legacy PINIGBI0 screen suffix."""

from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from capi_inference import CAPIInferencer
from capi_station_adapter import AAPIStationAdapter


@pytest.mark.parametrize("name", [
    "TL6681DAU131PINIGBI141747.tif",
    "T863MF77AD50PINIGBI000623.tif",
    "TL6681DAU131PINIGBI235959.tiff",
    "TL6681DAU131pinigbi141747.TIF",
    "YQ607S210B12PINIGBI0164814.tif",
    "SAMPLEPINIGBI0000623.tif",
])
def test_aapi_omit_filename_formats_share_dust_identity(name):
    adapter = AAPIStationAdapter()

    assert adapter.image_prefix(name) == "PINIGBI"
    assert adapter.image_group_key(name) == "PINIGBI"
    assert adapter.is_omit_image(name)


@pytest.mark.parametrize("name", [
    "TL6681DAU131PINIGBI14174.tif",
    "TL6681DAU131PINIGBI1417478.tif",
    "TL6681DAU131PINIGBI01417478.tif",
    "TL6681DAU131PINIGBI14174X.tif",
    "TL6681DAU131W0F00000141747.tif",
])
def test_aapi_omit_rejects_incomplete_names_and_other_screens(name):
    assert not AAPIStationAdapter().is_omit_image(name)


@pytest.mark.parametrize("rotate_180", [False, True])
def test_aapi_actual_panel_loads_dust_image_before_postprocessing(tmp_path, rotate_180):
    names = (
        "TL6681DAU131B8F00000141755.tif",
        "TL6681DAU131PINIGBI141747.tif",
        "TL6681DAU131PWM00000141752.tif",
        "TL6681DAU131STANDARD141750.tif",
        "TL6681DAU131W0F00000141747.tif",
        "TL6681DAU131White_Frame141753.tif",
    )
    dust_path = tmp_path / names[1]
    dust = np.arange(16 * 24, dtype=np.uint8).reshape(16, 24)
    for name in names:
        image = dust if name == dust_path.name else np.full_like(dust, 80)
        assert cv2.imwrite(str(tmp_path / name), image)

    worker = CAPIInferencer.__new__(CAPIInferencer)
    worker.station_adapter = AAPIStationAdapter()
    worker.config = SimpleNamespace(
        max_images_per_panel=7,
        inference_rotate_180_enabled=rotate_180,
    )
    worker.check_omit_overexposure = Mock(return_value=(False, 80.0, 0.0, "exposure OK"))
    image_files, duplicate = worker._prepare_panel_image_files(tmp_path)

    assert not duplicate
    assert len(image_files) == len(names)
    assert worker.station_adapter.find_omit_image(tmp_path) == dust_path
    omit_vis, overexposed, info, omit_image = worker._load_omit_context(
        tmp_path, image_files=image_files, product_resolution=(1920, 1200),
    )

    expected = cv2.rotate(dust, cv2.ROTATE_180) if rotate_180 else dust
    assert np.array_equal(omit_image, expected)
    assert np.array_equal(omit_vis, cv2.cvtColor(expected, cv2.COLOR_GRAY2BGR))
    assert overexposed is False
    assert info == "exposure OK"
    worker.check_omit_overexposure.assert_called_once()
    assert np.array_equal(worker.check_omit_overexposure.call_args.args[0], expected)
    assert worker.check_omit_overexposure.call_args.kwargs == {
        "image_files": image_files,
        "product_resolution": (1920, 1200),
    }
