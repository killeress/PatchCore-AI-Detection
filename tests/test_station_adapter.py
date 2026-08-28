import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from capi_config import CAPIConfig
from capi_inference import AOIReportDefect, CAPIInferencer, ImageResult
from capi_preprocess import filter_panel_lighting_files
from capi_station_adapter import (
    AAPIStationAdapter,
    create_station_adapter,
    resolve_station_profile_from_hostname,
)


def test_station_adapter_factory_rejects_unknown_profile():
    assert create_station_adapter("capi").profile == "capi"
    assert create_station_adapter("AAPI").profile == "aapi"
    with pytest.raises(ValueError, match="Unsupported station_profile"):
        create_station_adapter("auto")


@pytest.mark.parametrize(
    ("hostname", "expected"),
    [
        ("mod2-aapi09", "aapi"),
        ("MOD2-AAPI-09", "aapi"),
        ("capi13", "capi"),
        ("MOD2-CAPI07", "capi"),
        ("capihm", "capi"),
    ],
)
def test_station_profile_is_selected_from_hostname(hostname, expected):
    assert resolve_station_profile_from_hostname(hostname) == expected


@pytest.mark.parametrize("hostname", ["mod2-ai09", "", "aapi-capi-test"])
def test_unknown_or_ambiguous_station_hostname_is_rejected(hostname):
    with pytest.raises(RuntimeError, match="hostname"):
        resolve_station_profile_from_hostname(hostname)


def test_unknown_hostname_can_use_explicit_windows_development_fallback():
    assert resolve_station_profile_from_hostname(
        "developer-pc",
        default_if_unknown="capi",
    ) == "capi"

    with pytest.raises(RuntimeError, match="hostname"):
        resolve_station_profile_from_hostname(
            "aapi-capi-test",
            default_if_unknown="capi",
        )


@pytest.mark.parametrize("config_path", ["server_config.yaml", "server_config_local.yaml"])
def test_server_config_does_not_select_station_profile(config_path):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    assert "station_profile" not in config
    assert "aapi" not in config


def test_aapi_adapter_uses_hostname_selected_default_report_root():
    profile = resolve_station_profile_from_hostname("mod2-aapi09")
    adapter = create_station_adapter(profile)

    assert str(adapter.report_root).replace("\\", "/") == "/192.168.2.190/d/LOG"


@pytest.mark.parametrize("host", ["192.168.2.190", "192.168.2.191"])
def test_aapi_report_follows_panel_source_host(host):
    adapter = AAPIStationAdapter()

    report_file = adapter._report_file_for_panel(
        Path(f"/{host}/d/image/20260823/YQ536J221B12")
    )

    assert str(report_file).replace("\\", "/").endswith(
        f"/{host}/d/LOG/Report260823.log"
    )


def test_aapi_mod1_report_uses_tianmu_daily_log_layout():
    adapter = AAPIStationAdapter()

    report_file = adapter._report_file_for_panel(
        Path("/192.168.1.11/d/tianmu/yuantu/20260825/T362H8E0NN04")
    )

    assert str(report_file).replace("\\", "/").endswith(
        "/192.168.1.11/d/tianmu/report/Report260825.log"
    )


@pytest.mark.parametrize(
    "panel_dir",
    [
        Path("/192.168.1.11/d/image/20260825/T362H8E0NN04"),
        Path("/192.168.2.190/d/tianmu/yuantu/20260825/T362H8E0NN04"),
        Path("/192.168.9.9/d/image/20260825/T362H8E0NN04"),
    ],
)
def test_aapi_report_rejects_unknown_source_or_layout(panel_dir):
    adapter = AAPIStationAdapter()

    with pytest.raises(RuntimeError, match="Unsupported AAPI image/report layout"):
        adapter._report_file_for_panel(panel_dir)


def test_aapi_explicit_report_root_override_wins_over_panel_source():
    adapter = AAPIStationAdapter(report_root=Path("/test/LOG"))

    report_file = adapter._report_file_for_panel(
        Path("/192.168.2.191/d/image/20260823/YQ536J221B12")
    )

    assert str(report_file).replace("\\", "/").endswith(
        "/test/LOG/Report260823.log"
    )


def test_aapi_filename_mapping_keeps_source_images_distinct():
    adapter = AAPIStationAdapter()
    samples = {
        "YQ607S210B12G0F00000164818.tif": "G0F00000",
        "YQ607S210B12R0F00000164819.tif": "R0F00000",
        "YQ607S210B12W0F00000164814.tif": "W0F00000",
        "YQ607S210B12W0F00010164822.tif": "W0F00010",
        "YQ607S210B12WGF25250164820.tif": "WGF25250",
        "YQ607S210B12WGF50500164821.tif": "WGF50500",
        "YQ607S210B12U0F00000164817.tif": "U0F00000",
        "YQ607S210B12Windows_BG164820.tif": "WINDOWS_BG",
        "YQ607S210B12White_Frame164823.tif": "WHITEFRA",
        "YQ607S210B12PINIGBI0164814.tif": "PINIGBI",
    }

    for image_name, expected in samples.items():
        assert adapter.image_prefix(image_name) == expected

    assert adapter.model_prefix("WINDOWS_BG") == "STANDARD"
    assert adapter.model_prefix("W0F00010") == "WGF50500"
    assert adapter.report_prefix("YQ607S210B12Windows_BG164820.tif") == "STANDARD"
    assert adapter.report_prefix("YQ607S210B12W0F00010164822.tif") == "WGF50500"
    assert adapter.report_prefix("YQ607S210B12WGF25250164820.tif") == "WGF25250"
    assert adapter.report_prefix("YQ607S210B12U0F00000164817.tif") == "U0F00000"
    assert adapter.image_group_key("YQ607S210B12W0F00010164822.tif") != \
        adapter.image_group_key("YQ607S210B12WGF50500164821.tif")
    assert adapter.is_white_frame_image("YQ607S210B12White_Frame164823.tif")
    assert adapter.is_omit_image("YQ607S210B12PINIGBI0164814.tif")
    assert adapter.training_image_prefix("YQ607S210B12Windows_BG164820.tif") == "STANDARD"
    assert adapter.training_image_prefix("YQ607S210B12W0F00010164822.tif") == "WGF50500"
    assert adapter.training_prefixes == (
        "G0F00000", "R0F00000", "W0F00000", "WGF25250",
        "WGF50500", "U0F00000", "STANDARD",
    )


def test_aapi_mod1_filename_aliases_use_existing_models_and_white_frame_route(tmp_path):
    adapter = AAPIStationAdapter()
    standard_name = "T362H8E0NN04STANDARD143141.tif"
    white_frame_name = "T362H8E0NN04BWFRAME0143143.tif"

    assert adapter.image_prefix(standard_name) == "WINDOWS_BG"
    assert adapter.model_prefix(adapter.image_prefix(standard_name)) == "STANDARD"
    assert adapter.report_prefix(standard_name) == "STANDARD"
    assert adapter.training_image_prefix(standard_name) == "STANDARD"
    assert adapter.image_prefix(white_frame_name) == "WHITEFRA"
    assert adapter.is_white_frame_image(white_frame_name)

    (tmp_path / standard_name).write_bytes(b"")
    (tmp_path / white_frame_name).write_bytes(b"")
    files = filter_panel_lighting_files(
        tmp_path,
        prefix_resolver=adapter.image_prefix,
        allowed_prefixes=adapter.inference_prefixes,
    )

    assert set(files) == {"WINDOWS_BG"}
    assert files["WINDOWS_BG"].name == standard_name


def test_aapi_preprocess_keeps_w0f00010_and_wgf50500_as_two_images(tmp_path):
    adapter = AAPIStationAdapter()
    for image_name in (
        "YQ607S210B12W0F00010164822.tif",
        "YQ607S210B12WGF50500164821.tif",
        "YQ607S210B12Windows_BG164820.tif",
    ):
        (tmp_path / image_name).write_bytes(b"")

    files = filter_panel_lighting_files(
        tmp_path,
        prefix_resolver=adapter.image_prefix,
        allowed_prefixes=adapter.inference_prefixes,
    )

    assert set(files) == {"W0F00010", "WGF50500", "WINDOWS_BG"}
    assert adapter.model_prefix("W0F00010") == adapter.model_prefix("WGF50500")


def test_aapi_reserved_lightings_are_independent_models(tmp_path):
    adapter = AAPIStationAdapter()
    image_names = (
        "YQ607S210B12WGF25250164820.tif",
        "YQ607S210B12G0F00000164818.tif",
        "YQ607S210B12U0F00000164817.tif",
    )
    for image_name in image_names:
        (tmp_path / image_name).write_bytes(b"")

    files = filter_panel_lighting_files(
        tmp_path,
        prefix_resolver=adapter.image_prefix,
        allowed_prefixes=adapter.inference_prefixes,
    )

    assert set(files) == {"WGF25250", "G0F00000", "U0F00000"}
    assert {
        adapter.training_image_prefix(image_name)
        for image_name in image_names
    } == {"WGF25250", "G0F00000", "U0F00000"}


def test_aapi_model_routing_uses_alias_without_merging_source_images():
    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.station_adapter = AAPIStationAdapter()
    inferencer.config = SimpleNamespace(
        is_new_architecture=True,
        machine_id="MODEL-A",
        threshold_mapping={
            "STANDARD": {"inner": 0.41, "edge": 0.51},
            "WGF50500": {"inner": 0.42, "edge": 0.52},
        },
    )
    inferencer.threshold = 0.75
    inferencer._get_model_for = MagicMock(return_value="model")

    assert inferencer._get_inferencer_for_zone("WINDOWS_BG", "inner") == "model"
    inferencer._get_model_for.assert_called_once_with("MODEL-A", "STANDARD", "inner")
    assert inferencer._get_threshold_for_zone("W0F00010", "edge") == 0.52


def test_aapi_model_routing_keeps_reserved_models_independent():
    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.station_adapter = AAPIStationAdapter()
    inferencer.config = SimpleNamespace(
        is_new_architecture=True,
        machine_id="MODEL-A",
        threshold_mapping={
            "WGF25250": {"inner": 0.31, "edge": 0.41},
            "WGF50500": {"inner": 0.32, "edge": 0.42},
            "U0F00000": {"inner": 0.33, "edge": 0.43},
            "STANDARD": {"inner": 0.34, "edge": 0.44},
        },
    )
    inferencer.threshold = 0.75
    inferencer._get_model_for = MagicMock(return_value="model")

    assert inferencer._get_inferencer_for_zone("WGF25250", "inner") == "model"
    inferencer._get_model_for.assert_called_once_with(
        "MODEL-A", "WGF25250", "inner"
    )
    assert inferencer._get_threshold_for_zone("WGF25250", "edge") == 0.41
    assert inferencer._get_threshold_for_zone("WGF50500", "edge") == 0.42
    assert inferencer._get_threshold_for_zone("U0F00000", "inner") == 0.33
    assert inferencer._get_threshold_for_zone("WINDOWS_BG", "inner") == 0.34


def test_aapi_report_uses_last_complete_exact_glass_record(tmp_path):
    report_root = tmp_path / "LOG"
    report_root.mkdir()
    report = report_root / "Report260820.log"
    report.write_text(
        "2026/8/20 16:40:00,YQ-OTHER,NG,W0F00000,CDK2(00001,00002)\n"
        "2026/8/20 16:41:00,YQ607S210B12,NG,W0F00000,CDK2(01094,00129)\n"
        "2026/8/20 16:48:24,YQ607S210B12,NG,"
        "W0F00010,CM00(04649,00235)Windows_BG,CO05(02996,00555)"
        "White_Frame,CLV2(00277,00881)\n",
        encoding="utf-8",
    )
    adapter = AAPIStationAdapter(report_root=report_root, report_retry_count=1)

    parsed = adapter.parse_aoi_report(
        tmp_path / "image" / "20260820" / "YQ607S210B12",
        glass_id="YQ607S210B12",
        machine_judgment="NG",
    )

    assert set(parsed) == {"W0F00010", "WINDOWS_BG", "WHITEFRA"}
    assert parsed["W0F00010"][0].coordinate_space == "product"
    assert (parsed["W0F00010"][0].x, parsed["W0F00010"][0].y) == (1549, 235)
    assert parsed["WHITEFRA"][0].defect_code == "CLV2"


def test_aapi_report_parses_reserved_lightings_independently(tmp_path):
    report_root = tmp_path / "LOG"
    report_root.mkdir()
    (report_root / "Report260820.log").write_text(
        "2026/8/20 16:48:24,YQ607S210B12,NG,"
        "WGF25250,C250(00300,00400)"
        "G0F00000,CG00(00600,00700)"
        "U0F00000,CU00(00900,01000)\n",
        encoding="utf-8",
    )
    adapter = AAPIStationAdapter(report_root=report_root, report_retry_count=1)

    parsed = adapter.parse_aoi_report(
        tmp_path / "image" / "20260820" / "YQ607S210B12",
        glass_id="YQ607S210B12",
        machine_judgment="NG",
    )

    assert set(parsed) == {"WGF25250", "G0F00000", "U0F00000"}
    assert parsed["WGF25250"][0].image_prefix == "WGF25250"
    assert parsed["G0F00000"][0].image_prefix == "G0F00000"
    assert parsed["U0F00000"][0].image_prefix == "U0F00000"


def test_aapi_mod1_cp950_report_maps_standard_and_bwframe0(tmp_path):
    report_root = tmp_path / "report"
    report_root.mkdir()
    report = report_root / "Report260825.log"
    report.write_bytes(
        (
            "2026/8/25 下午 01:33:40,T362L7J7NQ03,NG,"
            "W0F00000,CO05(00201,00661)"
            "STANDARD,CLH2(00111,00250)"
            "BWFRAME0,CLV2(00220,00395)\n"
        ).encode("cp950")
    )
    adapter = AAPIStationAdapter(report_root=report_root, report_retry_count=1)

    parsed = adapter.parse_aoi_report(
        tmp_path / "yuantu" / "20260825" / "T362L7J7NQ03",
        glass_id="T362L7J7NQ03",
        machine_judgment="NG",
    )

    assert set(parsed) == {"W0F00000", "WINDOWS_BG", "WHITEFRA"}
    assert (parsed["W0F00000"][0].x, parsed["W0F00000"][0].y) == (67, 661)
    assert (parsed["WINDOWS_BG"][0].x, parsed["WINDOWS_BG"][0].y) == (37, 250)
    assert (parsed["WHITEFRA"][0].x, parsed["WHITEFRA"][0].y) == (73, 395)


def test_aapi_report_coordinate_is_mapped_from_protocol_product_resolution(tmp_path):
    report_root = tmp_path / "LOG"
    report_root.mkdir()
    (report_root / "Report260823.log").write_text(
        "2026/8/23 09:09:37,YQ62PY211B12,NG,W0F00000,CDK2(03078,00497)\n",
        encoding="utf-8",
    )
    adapter = AAPIStationAdapter(report_root=report_root, report_retry_count=1)
    parsed = adapter.parse_aoi_report(
        tmp_path / "image" / "20260823" / "YQ62PY211B12",
        glass_id="YQ62PY211B12",
        machine_judgment="NG",
    )
    defect = parsed["W0F00000"][0]

    assert defect.coordinate_space == "product"
    assert (defect.x, defect.y) == (1026, 497)

    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.station_adapter = adapter
    inferencer.config = CAPIConfig(inference_rotate_180_enabled=False)
    report_defect = AOIReportDefect(
        defect_code=defect.defect_code,
        product_x=defect.x,
        product_y=defect.y,
        image_prefix=defect.image_prefix,
        coordinate_space=defect.coordinate_space,
    )
    result = ImageResult(
        image_path=Path("YQ62PY211B12W0F00000090937.tif"),
        image_size=(6576, 4384),
        otsu_bounds=(68, 219, 6320, 4093),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0,
        anomaly_tiles=[],
        raw_bounds=(68, 219, 6320, 4093),
    )

    assert inferencer._resolve_aoi_report_defect(
        report_defect,
        result,
        (1920, 1200),
    ) == (3408, 1823, 1026, 497)


def test_aapi_report_does_not_fall_back_to_stale_row_when_latest_is_partial(tmp_path):
    report_root = tmp_path / "LOG"
    report_root.mkdir()
    (report_root / "Report260820.log").write_text(
        "2026/8/20 16:41:00,YQ607S210B12,NG,W0F00000,CDK2(01094,00129)\n"
        "2026/8/20 16:48:24,YQ607S210B12,NG,W0F00000,CDK2(0531",
        encoding="utf-8",
    )
    adapter = AAPIStationAdapter(report_root=report_root, report_retry_count=1)

    with pytest.raises(RuntimeError, match="latest_record_incomplete"):
        adapter.parse_aoi_report(
            tmp_path / "image" / "20260820" / "YQ607S210B12",
            glass_id="YQ607S210B12",
            machine_judgment="NG",
        )


def test_aapi_ok_without_log_row_has_no_aoi_candidates(tmp_path):
    report_root = tmp_path / "LOG"
    adapter = AAPIStationAdapter(report_root=report_root, report_retry_count=1)

    assert adapter.parse_aoi_report(
        Path("/image/20260820/YQ607S210B12"),
        glass_id="YQ607S210B12",
        machine_judgment="OK",
    ) == {}


def test_aapi_ng_rejects_latest_ok_record(tmp_path):
    report_root = tmp_path / "LOG"
    report_root.mkdir()
    (report_root / "Report260820.log").write_text(
        "2026/8/20 16:48:24,YQ607S210B12,OK\n",
        encoding="utf-8",
    )
    adapter = AAPIStationAdapter(report_root=report_root, report_retry_count=1)

    with pytest.raises(RuntimeError, match="latest_record_status_ok"):
        adapter.parse_aoi_report(
            tmp_path / "image" / "20260820" / "YQ607S210B12",
            glass_id="YQ607S210B12",
            machine_judgment="NG",
        )


def test_aapi_product_coordinate_is_resolved_to_image_coordinate():
    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.station_adapter = AAPIStationAdapter()
    inferencer.config = CAPIConfig(inference_rotate_180_enabled=False)
    defect = AOIReportDefect(
        defect_code="CDK2",
        product_x=600,
        product_y=400,
        image_prefix="W0F00000",
        coordinate_space="product",
    )
    result = ImageResult(
        image_path=Path("YQ607S210B12W0F00000164814.tif"),
        image_size=(1200, 800),
        otsu_bounds=(100, 100, 1100, 700),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0,
        anomaly_tiles=[],
        raw_bounds=(100, 100, 1100, 700),
    )

    assert inferencer._resolve_aoi_report_defect(defect, result, (2000, 1200)) == (
        400,
        300,
        600,
        400,
    )
    assert (defect.resolved_image_x, defect.resolved_image_y) == (400, 300)
    assert (defect.resolved_product_x, defect.resolved_product_y) == (600, 400)

    from capi_server import _serialize_aoi_machine_coords
    stored = json.loads(_serialize_aoi_machine_coords({"W0F00000": [defect]}))
    assert stored["W0F00000"][0]["coordinate_space"] == "product"
    assert (stored["W0F00000"][0]["image_x"], stored["W0F00000"][0]["image_y"]) == (400, 300)
    assert (stored["W0F00000"][0]["product_x"], stored["W0F00000"][0]["product_y"]) == (600, 400)


def test_absolute_image_coordinate_respects_formal_180_degree_input_rotation():
    inferencer = CAPIInferencer.__new__(CAPIInferencer)
    inferencer.config = CAPIConfig(inference_rotate_180_enabled=True)
    defect = AOIReportDefect(
        defect_code="CDK2",
        product_x=99,
        product_y=49,
        image_prefix="W0F00000",
        coordinate_space="image",
    )
    result = ImageResult(
        image_path=Path("YQ607S210B12W0F00000164814.tif"),
        image_size=(1000, 500),
        otsu_bounds=(0, 0, 1000, 500),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0,
        anomaly_tiles=[],
        raw_bounds=(0, 0, 1000, 500),
    )

    assert inferencer._resolve_aoi_report_defect(defect, result, (2000, 1000)) == (
        900,
        450,
        1800,
        900,
    )


def test_unresolved_aapi_report_coordinate_is_stored_as_product_space():
    from capi_server import _serialize_aoi_machine_coords

    defect = AOIReportDefect(
        defect_code="CLV2",
        product_x=277,
        product_y=881,
        image_prefix="WHITEFRA",
        coordinate_space="product",
    )

    stored = json.loads(_serialize_aoi_machine_coords({"WHITEFRA": [defect]}))
    row = stored["WHITEFRA"][0]
    assert (row["product_x"], row["product_y"]) == (277, 881)
    assert (row["image_x"], row["image_y"]) == (-1, -1)
