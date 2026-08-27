from pathlib import Path

import numpy as np

from capi_config import CAPIConfig
from capi_inference import ImageResult, TileInfo
from capi_station_adapter import AAPIStationAdapter
from capi_server import (
    build_dual_protocol_response,
    build_qjpg_response,
    check_image_abnormal_precheck,
    parse_request,
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


def test_parse_request_keeps_standard_no_bomb_image_dir():
    parsed = parse_request(
        "AOI@TL62U17BC17B;GZ0790KA0017S;CAPI07HM-P2;1080,1920;OK;"
        "//192.168.20.12/yuantu/GZ0790KA0017S/260616/TL62U17BC17B"
    )

    assert parsed["image_dir"] == (
        "//192.168.20.12/yuantu/GZ0790KA0017S/260616/TL62U17BC17B"
    )
    assert parsed["bomb_info"] is None


def test_parse_request_skips_empty_no_bomb_reserved_fields():
    parsed = parse_request(
        "AOI@TL62U17BC17B;GZ0790KA0017S;CAPI07HM-P2;1080,1920;OK;;;"
        "//192.168.20.12/yuantu/GZ0790KA0017S/260616/TL62U17BC17B"
    )

    assert parsed["image_dir"] == (
        "//192.168.20.12/yuantu/GZ0790KA0017S/260616/TL62U17BC17B"
    )
    assert parsed["bomb_info"] is None


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


def test_qjpg_aoi_report_fallback_keeps_exact_source_product_coordinate():
    result = _image_result("W0F00000_114438.tif")
    tile = _tile(1, 600, 450)
    tile.is_aoi_coord_tile = True
    tile.aoi_product_x = 1136
    tile.aoi_product_y = 872
    tile.aoi_image_x = 600
    tile.aoi_image_y = 450
    tile.anomaly_peak_source = "aoi_report_fallback"
    result.tiles = [tile]
    result.anomaly_tiles = [(tile, 0.91, None)]

    response = build_qjpg_response(
        {"glass_id": "YQ52TV232E45", "resolution": (1920, 1200)},
        "NG",
        [result],
        CAPIConfig(),
    )

    assert response == (
        "@QJPG-YQ52TV232E45;OK;EJ;"
        "NGPCDK20113600872W0F00000,"
    )


def test_qjpg_aoi_center_real_region_matches_field_product_coordinate():
    result = _image_result("YQ52TV232E45W0F00000085923.tif", mark_text="N0")
    result.raw_bounds = (77, 202, 6320, 4114)
    result.otsu_bounds = result.raw_bounds
    result.report_image_prefix = "W0F00000"
    tile = _tile(1, 3770, 3045)
    tile.is_aoi_coord_tile = True
    tile.aoi_product_x = 1136
    tile.aoi_product_y = 872
    tile.aoi_image_x = 3770
    tile.aoi_image_y = 3045
    tile.anomaly_peak_source = "aoi_real_region"
    result.tiles = [tile]
    result.anomaly_tiles = [(tile, 0.4283, None)]

    response = build_qjpg_response(
        {"glass_id": "YQ52TV232E45", "resolution": (1920, 1200)},
        "NG",
        [result],
        CAPIConfig(),
    )

    assert response == (
        "@QJPG-YQ52TV232E45;OK;N0;"
        "NGPCDK20113600872W0F00000,"
    )


def test_qjpg_response_keeps_source_prefix_for_hm_standard_image():
    result = _image_result("U0F00000092908.tif")
    tile = _tile(1, 600, 450)
    result.tiles = [tile]
    result.anomaly_tiles = [(tile, 0.91, None)]

    response = build_qjpg_response(
        {"glass_id": "TL6380GAL102", "resolution": (2000, 1000)},
        "NG",
        [result],
        CAPIConfig(),
    )

    assert response == "@QJPG-TL6380GAL102;OK;EJ;NGPCDK20100000500U0F00000,"


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
    non_bomb_tile = _tile(1, 600, 450)
    tile = _tile(2, 600, 450)
    tile.is_bomb = True
    result.tiles = [non_bomb_tile, tile]
    result.anomaly_tiles = [
        (non_bomb_tile, 0.90, None),
        (tile, 0.91, None),
    ]

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


def test_image_abnormal_precheck_uses_latest_retake_image(monkeypatch):
    old_image = Path("W0F00000_145854.tif")
    latest_image = Path("W0F00000_150610.tif")
    read_paths = []

    class FakeStat:
        def __init__(self, mtime):
            self.st_mtime = mtime

    def fake_stat(path):
        return FakeStat(2 if path.name == latest_image.name else 1)

    def fake_imread(path, flags):
        image_name = Path(path).name
        read_paths.append(image_name)
        if image_name == old_image.name:
            return np.full((4, 4), 0, dtype=np.uint8)
        return np.full((4, 4), 60, dtype=np.uint8)

    monkeypatch.setattr(Path, "stat", fake_stat)
    monkeypatch.setattr("capi_server.cv2.imread", fake_imread)
    monkeypatch.setattr("capi_server.detect_panel_polygon", lambda image, cfg: (None, None))
    cfg = CAPIConfig(
        image_abnormal_detection_enabled=True,
        image_abnormal_w0f00000_mean_lower=40,
        image_abnormal_w0f00000_mean_upper=82,
    )

    result = check_image_abnormal_precheck(
        Path("unused"),
        cfg,
        [old_image, latest_image],
        report_prefixes=["W0F00000"],
    )

    assert result is None
    assert read_paths == [latest_image.name]


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


def test_aapi_image_abnormal_alias_keeps_w0f00010_source_distinct(monkeypatch):
    adapter = AAPIStationAdapter()
    read_paths = []

    def fake_imread(path, flags):
        image_name = Path(path).name
        read_paths.append(image_name)
        value = 90 if "W0F00010" in image_name else 60
        return np.full((4, 4), value, dtype=np.uint8)

    monkeypatch.setattr("capi_server.cv2.imread", fake_imread)
    monkeypatch.setattr("capi_server.detect_panel_polygon", lambda image, cfg: (None, None))
    cfg = CAPIConfig(
        image_abnormal_detection_enabled=True,
        image_abnormal_wgf50500_mean_lower=40,
        image_abnormal_wgf50500_mean_upper=82,
    )
    w0f00010 = Path("YQ607S210B12W0F00010164822.tif")
    wgf50500 = Path("YQ607S210B12WGF50500164821.tif")

    result = check_image_abnormal_precheck(
        Path("unused"),
        cfg,
        [w0f00010, wgf50500],
        report_prefixes=["W0F00010"],
        image_prefix_resolver=adapter.image_prefix,
        screen_alias_resolver=adapter.model_prefix,
        boundary_reference_priority=adapter.boundary_reference_priority,
    )

    assert result["screen"] == "WGF50500"
    assert result["image_name"] == w0f00010.name
    assert read_paths.count(w0f00010.name) == 1


def test_qjpg_response_ok_i_omits_within_spec_points_and_missing_mark_is_00():
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

    assert response == "@QJPG-G1;NG;00;OK,"


def test_qjpg_response_ok_i_reports_only_bomb_points():
    result = _image_result("W0F00000_114438.tif")
    within_spec_tile = _tile(1, 600, 450)
    bomb_tile = _tile(2, 700, 500)
    bomb_tile.is_bomb = True
    result.tiles = [within_spec_tile, bomb_tile]
    result.anomaly_tiles = [
        (within_spec_tile, 0.91, None),
        (bomb_tile, 0.92, None),
    ]

    response = build_qjpg_response(
        {"glass_id": "G1", "resolution": (2000, 1000)},
        "OK-i",
        [result],
        CAPIConfig(report_bomb_defect_code="BMB99"),
    )

    assert response == "@QJPG-G1;OK;EJ;NGBMB990120000600W0F00000,"
