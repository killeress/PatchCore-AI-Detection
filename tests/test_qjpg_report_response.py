from pathlib import Path

import numpy as np

from capi_config import CAPIConfig
from capi_inference import ImageResult, TileInfo
from capi_server import (
    build_dual_protocol_response,
    build_qjpg_response,
    check_image_abnormal_precheck,
)


def _image_result(image_name: str, *, mark_text: str = "EJ") -> ImageResult:
    return ImageResult(
        image_path=Path(image_name),
        image_size=(1200, 800),
        otsu_bounds=(100, 200, 1100, 700),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        raw_bounds=(100, 200, 1100, 700),
        mark_text=mark_text,
    )


def _tile(tile_id: int, peak_x: int, peak_y: int) -> TileInfo:
    tile = TileInfo(
        tile_id=tile_id,
        x=0,
        y=0,
        width=512,
        height=512,
        image=np.zeros((512, 512, 3), dtype=np.uint8),
    )
    tile.anomaly_peak_x = peak_x
    tile.anomaly_peak_y = peak_y
    return tile


def test_qjpg_response_uses_final_ng_points_and_product_coordinates():
    result = _image_result("W0F00000_114438.tif")
    tile = _tile(1, 600, 450)
    result.tiles = [tile]
    result.anomaly_tiles = [(tile, 0.91, None)]

    response = build_qjpg_response(
        {"glass_id": "T863BF29AH44", "resolution": (2000, 1000)},
        "NG",
        [result],
        CAPIConfig(),
    )

    assert response == "@QJPG-T863BF29AH44;OK;EJ;NGPCDK20100000500W0F00000,"


def test_dual_protocol_response_sends_legacy_aoi_then_qjpg():
    result = _image_result("W0F00000_114438.tif")
    tile = _tile(1, 600, 450)
    result.tiles = [tile]
    result.anomaly_tiles = [(tile, 0.91, None)]

    response = build_dual_protocol_response(
        {
            "glass_id": "T863BF29AH44",
            "model_id": "GN156HCAB6G0S",
            "machine_no": "CAPI1403",
            "machine_judgment": "OK",
            "resolution": (2000, 1000),
        },
        "NG",
        [result],
        CAPIConfig(),
    )

    assert response == (
        "AOI@T863BF29AH44;GN156HCAB6G0S;CAPI1403;OK;NG"
        "\r\n"
        "@QJPG-T863BF29AH44;OK;EJ;NGPCDK20100000500W0F00000,"
    )


def test_dual_protocol_legacy_response_maps_ok_i_to_ok():
    response = build_dual_protocol_response(
        {
            "glass_id": "G1",
            "model_id": "M1",
            "machine_no": "CAPI1403",
            "machine_judgment": "OK",
            "resolution": (2000, 1000),
        },
        "OK-i",
        [],
        CAPIConfig(),
    )

    assert response == "AOI@G1;M1;CAPI1403;OK;OK\r\n@QJPG-G1;NG;00;OK,"


def test_dual_protocol_response_without_parsed_data_keeps_legacy_error_shape():
    response = build_dual_protocol_response(None, "ERR:PROTOCOL_ERROR (bad)", [], None)

    assert response == (
        "AOI@;;;;ERR:PROTOCOL_ERROR (bad)"
        "\r\n"
        "@QJPG-;NG;00;ERR:PROTOCOL_ERROR (bad),"
    )


def test_qjpg_response_uses_white_dot_code_for_b0f_defect():
    result = _image_result("B0F00000_114438.tif")
    tile = _tile(1, 600, 450)
    result.tiles = [tile]
    result.anomaly_tiles = [(tile, 0.91, None)]

    response = build_qjpg_response(
        {"glass_id": "G1", "resolution": (2000, 1000)},
        "NG",
        [result],
        CAPIConfig(report_white_dot_defect_code="WHT01"),
    )

    assert response == "@QJPG-G1;OK;EJ;NGWHT010100000500B0F00000,"


def test_qjpg_response_uses_bomb_code_for_bomb_defect_even_when_internal_ok():
    result = _image_result("W0F00000_114438.tif")
    tile = _tile(1, 600, 450)
    tile.is_bomb = True
    result.tiles = [tile]
    result.anomaly_tiles = [(tile, 0.91, None)]

    response = build_qjpg_response(
        {"glass_id": "G1", "resolution": (2000, 1000)},
        "OK",
        [result],
        CAPIConfig(report_bomb_defect_code="BMB99"),
    )

    assert response == "@QJPG-G1;OK;EJ;NGBMB990100000500W0F00000,"


def test_qjpg_response_uses_image_abnormal_code_for_hy():
    response = build_qjpg_response(
        {"glass_id": "G1", "resolution": (2000, 1000), "image_dir": "D:/panels/W0F00000_114438.tif"},
        "ERR:HY:W0F00000",
        [],
        CAPIConfig(report_image_abnormal_defect_code="HY999"),
    )

    assert response == "@QJPG-G1;NG;00;NGHY9990000000000W0F00000,"


def test_image_abnormal_precheck_detects_mean_brightness(monkeypatch):
    def fake_imread(path, flags):
        return np.full((4, 4), 90, dtype=np.uint8)

    monkeypatch.setattr("capi_server.cv2.imread", fake_imread)
    cfg = CAPIConfig(
        image_abnormal_detection_enabled=True,
        image_abnormal_w0f00000_mean_lower=40,
        image_abnormal_w0f00000_mean_upper=82,
    )

    result = check_image_abnormal_precheck(
        Path("unused"),
        cfg,
        [Path("W0F00000_113600.tif")],
        report_prefixes=["W0F00000"],
    )

    assert result["screen"] == "W0F00000"
    assert result["mean_brightness"] == 90.0
    assert result["mean_source"] == "full_image"
    assert result["mean_pixels"] == 16
    assert result["lower"] == 40
    assert result["upper"] == 82
    assert result["detail"] == "Mean:90.000(range=40-82, source=full_image, pixels=16)"

    cfg_low = CAPIConfig(
        image_abnormal_detection_enabled=True,
        image_abnormal_w0f00000_mean_lower=95,
        image_abnormal_w0f00000_mean_upper=110,
    )
    low_result = check_image_abnormal_precheck(
        Path("unused"),
        cfg_low,
        [Path("W0F00000_113600.tif")],
        report_prefixes=["W0F00000"],
    )
    assert low_result["screen"] == "W0F00000"
    assert low_result["lower"] == 95
    assert low_result["upper"] == 110


def test_image_abnormal_precheck_uses_polygon_mean_for_judgment(monkeypatch):
    image = np.full((6, 6), 250, dtype=np.uint8)
    image[1:5, 1:5] = 60
    polygon = np.array([[1, 1], [4, 1], [4, 4], [1, 4]], dtype=np.float32)

    def fake_imread(path, flags):
        return image.copy()

    def fake_detect_panel_polygon(img, cfg):
        return (0, 0, 6, 6), polygon

    monkeypatch.setattr("capi_server.cv2.imread", fake_imread)
    monkeypatch.setattr("capi_server.detect_panel_polygon", fake_detect_panel_polygon)
    cfg = CAPIConfig(
        image_abnormal_detection_enabled=True,
        image_abnormal_w0f00000_mean_lower=50,
        image_abnormal_w0f00000_mean_upper=70,
    )

    result = check_image_abnormal_precheck(
        Path("unused"),
        cfg,
        [Path("W0F00000_113600.tif")],
        report_prefixes=["W0F00000"],
    )

    assert result is None


def test_image_abnormal_precheck_only_checks_aoi_report_prefixes(monkeypatch):
    read_paths = []

    def fake_imread(path, flags):
        read_paths.append(Path(path).name)
        if "B0F00000" in str(path):
            return np.full((4, 4), 90, dtype=np.uint8)
        return np.full((4, 4), 10, dtype=np.uint8)

    monkeypatch.setattr("capi_server.cv2.imread", fake_imread)
    cfg = CAPIConfig(
        image_abnormal_detection_enabled=True,
        image_abnormal_w0f00000_mean_lower=0,
        image_abnormal_w0f00000_mean_upper=82,
        image_abnormal_b0f00000_mean_lower=0,
        image_abnormal_b0f00000_mean_upper=82,
    )

    result = check_image_abnormal_precheck(
        Path("unused"),
        cfg,
        [Path("B0F00000_113600.tif"), Path("W0F00000_113600.tif")],
        report_prefixes=["W0F00000"],
    )

    assert result is None
    assert read_paths == ["W0F00000_113600.tif"]

    read_paths.clear()
    result = check_image_abnormal_precheck(
        Path("unused"),
        cfg,
        [Path("B0F00000_113600.tif"), Path("W0F00000_113600.tif")],
        report_prefixes=["B0F00000"],
    )

    assert result["screen"] == "B0F00000"
    assert read_paths == ["W0F00000_113600.tif", "B0F00000_113600.tif"]


def test_qjpg_response_ok_i_reports_detected_points_and_missing_mark_is_00():
    result = _image_result("STANDARD_114438.tif", mark_text="")
    tile = _tile(1, 600, 450)
    result.tiles = [tile]
    result.anomaly_tiles = [(tile, 0.91, None)]

    response = build_qjpg_response(
        {"glass_id": "G1", "resolution": (2000, 1000)},
        "OK-i",
        [result],
        CAPIConfig(),
    )

    assert response == "@QJPG-G1;NG;00;NGPCDK20100000500STANDARD,"
