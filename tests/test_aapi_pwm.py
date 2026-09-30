"""MOD2 PWM screens retain their own images, coordinates, and model settings."""
import os
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
from jinja2 import Environment, FileSystemLoader
import numpy as np
import pytest

from capi_config import CAPIConfig
from capi_database import CAPIDatabase
from capi_inference import CAPIInferencer
from capi_preprocess import filter_panel_lighting_files
from capi_server import check_image_abnormal_precheck, parse_request
from capi_station_adapter import create_station_adapter
from capi_web import CAPIWebHandler, _within_spec_screen_code


PANEL_NAMES = (
    "T863MF77AD50B8F00000000628.tif",
    "T863MF77AD50PINIGBI000623.tif",
    "T863MF77AD50PWM00000000626.tif",
    "T863MF77AD50STANDARD000626.tif",
    "T863MF77AD50W0F00000000623.tif",
    "T863MF77AD50White_Frame000627.tif",
)


@pytest.mark.parametrize("name", [
    PANEL_NAMES[2], "T863MF77AD50PWM00000235959.tiff",
    "T863MF77AD50pwm00000000626.TIF",
])
def test_pwm_filename_keeps_its_model_and_report_identity(name):
    adapter = create_station_adapter("aapi")
    assert adapter.image_prefix(name) == "PWM00000"
    assert adapter.training_image_prefix(name) == "PWM00000"
    assert adapter.report_prefix(name) == "PWM00000"
    assert adapter.image_group_key(name) == "PWM00000"


def test_mod2_selection_keeps_pwm_retake_separate_from_standard_and_white(tmp_path):
    adapter = create_station_adapter("aapi")
    for name in PANEL_NAMES:
        path = tmp_path / name
        path.write_bytes(b"image")
        os.utime(path, (10, 10))
    retake = tmp_path / "T863MF77AD50PWM00000000630.tif"
    retake.write_bytes(b"retake")
    os.utime(retake, (20, 20))
    expected = {
        "PWM00000": retake,
        "WINDOWS_BG": tmp_path / PANEL_NAMES[3],
        "W0F00000": tmp_path / PANEL_NAMES[4],
    }
    assert filter_panel_lighting_files(
        tmp_path, prefix_resolver=adapter.image_prefix,
        allowed_prefixes=adapter.inference_prefixes,
    ) == expected
    worker = CAPIInferencer.__new__(CAPIInferencer)
    worker.station_adapter = adapter
    worker.config = SimpleNamespace(max_images_per_panel=7)
    selected, duplicate = worker._prepare_panel_image_files(tmp_path)
    assert duplicate is True
    assert all(path in selected for path in expected.values())
    assert tmp_path / PANEL_NAMES[2] not in selected
    assert adapter.find_lighting_image(tmp_path, "PWM00000") == retake
    assert CAPIWebHandler._detect_train_new_lightings([tmp_path], adapter) == [
        "W0F00000", "PWM00000", "STANDARD",
    ]
    assert "PWM00000" not in create_station_adapter("capi").inference_prefixes


def test_pwm_testing_payload_and_bomb_keep_independent_coordinates(tmp_path):
    parsed = parse_request(
        f"AOI@T863MF77AD50;TEST-MODEL;AAPI12;1920,1200;NG;"
        f"PWM00000;(350/230);{tmp_path};"
        "PWM00000,CDK2(00300,00200);STANDARD,CDK2(01050,00230)"
    )
    worker = CAPIInferencer.__new__(CAPIInferencer)
    worker.station_adapter = create_station_adapter("aapi")
    worker.config = CAPIConfig(bomb_area_force_detection_enabled=True)
    report = worker._parse_aoi_report_txt(
        tmp_path, glass_id=parsed["glass_id"], machine_judgment="NG",
        report_payload=parsed["aoi_report_payload"],
    )
    pwm = report["PWM00000"][0]
    assert (pwm.product_x, pwm.product_y, pwm.image_prefix) == (100, 200, "PWM00000")
    assert report["WINDOWS_BG"][0].product_x == 350
    updated, added = worker._aoi_report_with_forced_client_bomb_coords(
        report, parsed["bomb_info"],
    )
    assert added == 1
    assert len(updated["PWM00000"]) == 2
    assert len(updated["WINDOWS_BG"]) == 1


def test_pwm_uses_own_inner_and_edge_models_and_thresholds():
    worker = CAPIInferencer.__new__(CAPIInferencer)
    worker.station_adapter = create_station_adapter("aapi")
    worker.config = CAPIConfig(
        machine_id="TEST-MODEL", is_new_architecture=True,
        threshold_mapping={"PWM00000": {"inner": 0.31, "edge": 0.41},
                           "STANDARD": {"inner": 0.81, "edge": 0.91}},
    )
    worker.threshold = 0.75
    worker._get_model_for = Mock(side_effect=lambda _machine, lighting, zone: (lighting, zone))
    for zone, threshold in worker.config.threshold_mapping["PWM00000"].items():
        assert worker._get_inferencer_for_zone("PWM00000", zone) == ("PWM00000", zone)
        assert worker._get_threshold_for_zone("PWM00000", zone) == threshold
    assert worker._get_inferencer_for_zone("WINDOWS_BG", "inner") == ("STANDARD", "inner")


@pytest.mark.parametrize("configured", [False, True])
def test_pwm_legacy_model_missing_or_failed_never_uses_default(configured):
    worker = CAPIInferencer.__new__(CAPIInferencer)
    worker.station_adapter = create_station_adapter("aapi")
    worker._model_mapping = {"STANDARD": "standard.pt"}
    if configured:
        worker._model_mapping["PWM00000"] = "pwm.pt"
    worker._inferencers = {}
    worker.inferencer = object()
    worker._load_model_from_path = Mock(return_value=None)
    with pytest.raises(RuntimeError, match="PWM00000.*model_mapping"):
        worker._get_inferencer_for_prefix("PWM00000")


def test_pwm_brightness_and_within_spec_settings_roundtrip_independently(tmp_path):
    cfg = CAPIConfig.from_dict({
        "image_abnormal_standard_mean_lower": 70,
        "image_abnormal_standard_mean_upper": 90,
    })
    assert (cfg.image_abnormal_pwm00000_mean_lower, cfg.image_abnormal_pwm00000_mean_upper) == (0, 255)
    db = CAPIDatabase(tmp_path / "config.db")
    db.init_config_from_yaml(cfg)
    assert db.get_config_param("image_abnormal_pwm00000_mean_upper")["decoded_value"] == 255
    assert db.update_config_param("image_abnormal_pwm00000_mean_lower", 30)
    assert db.update_config_param("image_abnormal_pwm00000_mean_upper", 60)
    cfg.apply_db_overrides(db.get_all_config_params())
    cfg.to_yaml(str(tmp_path / "config.yaml"))
    for restored in (CAPIConfig.from_dict(cfg.to_dict()), CAPIConfig.from_yaml(str(tmp_path / "config.yaml"))):
        assert (restored.image_abnormal_pwm00000_mean_lower, restored.image_abnormal_pwm00000_mean_upper) == (30, 60)
        assert (restored.image_abnormal_standard_mean_lower, restored.image_abnormal_standard_mean_upper) == (70, 90)
        screens = restored.within_spec_judgment_rules["default"]["screens"]
        assert _within_spec_screen_code(PANEL_NAMES[2], screens, create_station_adapter("aapi")) == "PWM00000"
        assert screens["PWM00000"] is not screens["STANDARD"]


def test_pwm_brightness_precheck_uses_only_pwm_limits(tmp_path, monkeypatch):
    adapter = create_station_adapter("aapi")
    pwm = tmp_path / PANEL_NAMES[2]
    standard = tmp_path / PANEL_NAMES[3]
    for path, gray in ((pwm, 100), (standard, 80)):
        assert cv2.imwrite(str(path), np.full((64, 64), gray, np.uint8))
    monkeypatch.setattr("capi_server._detect_image_abnormal_product_polygon", lambda *a: (None, "full_image"))
    cfg = CAPIConfig(image_abnormal_detection_enabled=True)
    kwargs = dict(
        image_files=[pwm, standard], image_prefix_resolver=adapter.image_prefix,
        screen_alias_resolver=adapter.model_prefix,
        boundary_reference_priority=adapter.boundary_reference_priority,
        station_profile="aapi",
    )
    assert check_image_abnormal_precheck(tmp_path, cfg, report_prefixes=["PWM00000"], **kwargs) is None
    cfg.image_abnormal_pwm00000_mean_upper = 60
    assert check_image_abnormal_precheck(tmp_path, cfg, report_prefixes=["WINDOWS_BG"], **kwargs) is None
    result = check_image_abnormal_precheck(tmp_path, cfg, report_prefixes=["PWM00000"], **kwargs)
    assert result["screen"] == "PWM00000"
    assert result["mean_brightness"] == 100
    assert result["upper"] == 60


@pytest.mark.parametrize("station", ["AAPI", "CAPI"])
def test_pwm_settings_controls_follow_station(station):
    templates = Path(__file__).resolve().parents[1] / "templates"
    html = Environment(loader=FileSystemLoader(templates)).get_template("settings.html").render(
        station_name=station, settings_user={}, app_version={"version": "test"},
    )
    assert ("['PWM00000', 'PWM00000']" in html) == (station == "AAPI")
    assert ("['PWM00000', 'PWM00000 畫面'" in html) == (station == "AAPI")


@pytest.mark.parametrize("after_tiling", [True, False])
def test_pwm_training_preview_preserves_lighting(tmp_path, monkeypatch, after_tiling):
    import capi_preprocess
    from test_capi_web_train_new import _make_handler_with_server

    fixture = Path(__file__).parent / "fixtures" / "preprocess" / "synthetic_panel.png"
    image_path = tmp_path / PANEL_NAMES[2]
    image_path.write_bytes(fixture.read_bytes())
    original_preprocess = capi_preprocess.preprocess_panel_image
    lightings = []

    def capture_preprocess(image_path, lighting, config, **kwargs):
        lightings.append(lighting)
        return original_preprocess(image_path, lighting, config, **kwargs)

    monkeypatch.setattr(capi_preprocess, "preprocess_panel_image", capture_preprocess)
    monkeypatch.setattr(CAPIWebHandler, "_debug_heatmap_dir", tmp_path / "debug")
    server = SimpleNamespace(
        station_adapter=create_station_adapter("aapi"), database=Mock(),
        path_mapping={}, inferencers={},
    )
    handler = _make_handler_with_server(server, "/api/train/new/preprocess_pipeline_preview")
    payload = {
        "image_path": str(image_path), "preprocess_after_tiling": after_tiling,
        "zone": "edge",
        "image_preprocess_pipeline": [{"method": "gaussian", "params": {"kernel_size": 5, "sigma": 1.0}}],
    }
    if not after_tiling:
        payload["grid_canonicalization"] = {
            "enabled": True, "samples_per_cell": 3, "product_resolution": [1920, 1200],
        }
    body = json.dumps(payload).encode()
    handler.headers.get = Mock(return_value=str(len(body)))
    handler.rfile = io.BytesIO(body)
    handler._handle_train_new_preprocess_pipeline_preview()
    response = handler._sent_response[0]
    assert response["status"] == 200, response
    assert json.loads(response["body"])["success"] is True
    assert lightings == ["PWM00000"]
