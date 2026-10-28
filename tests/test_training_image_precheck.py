"""Training source screening and API gates, using real small images and no GPU."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from capi_config import CAPIConfig
from capi_station_adapter import create_station_adapter
from capi_training_precheck import (
    TrainingPanelPrecheckError, check_training_panels, verify_training_panel_sources,
)
from capi_web import CAPIWebHandler


def write_image(panel, name, value):
    panel.mkdir(parents=True, exist_ok=True)
    path = panel / name
    image = np.full((20, 24), value, np.uint8) if np.isscalar(value) else value
    assert cv2.imwrite(str(path), image)
    return path


@pytest.fixture(autouse=True)
def fixed_product_geometry(monkeypatch):
    # Polygon fitting is covered by its own tests. Use the actual mask/mean code.
    monkeypatch.setattr("capi_server.detect_panel_geometry", lambda *_: (None, None))


def screen(paths, config=None, profile="capi"):
    return check_training_panels(paths, config or CAPIConfig(), create_station_adapter(profile))


@pytest.mark.parametrize("screen_name,lower,upper", [
    ("STANDARD", 68, 88), ("U0F00000", 68, 88), ("WGF50500", 72, 92),
    ("G0F00000", 67, 87), ("R0F00000", 71, 91), ("W0F00000", 70, 90),
    ("W0F00010", 72, 92), ("B0F00000", 0, 13),
])
def test_training_checks_each_screen_with_inclusive_saved_limits(tmp_path, screen_name, lower, upper):
    write_image(tmp_path, f"{screen_name}_lower.png", lower)
    write_image(tmp_path, f"{screen_name}_upper.png", upper)
    result = screen([tmp_path])  # Inference switch is off by default.
    assert result["panels"][0]["passed"]
    assert [image["mean_brightness"] for image in result["panels"][0]["images"]] == [lower, upper]


@pytest.mark.parametrize("value", [0, 91])
def test_one_bad_image_rejects_the_whole_panel_and_preserves_other_panels(tmp_path, value):
    good, bad = tmp_path / "GOOD", tmp_path / "BAD"
    write_image(good, "W0F00000.png", 80)
    write_image(bad, "W0F00000.png", 80)
    write_image(bad, "G0F00000.png", value)
    with pytest.raises(TrainingPanelPrecheckError) as caught:
        screen([good, bad])
    assert [p["panel_id"] for p in caught.value.rejected_panels] == ["BAD"]
    issue = caught.value.rejected_panels[0]["issues"][0]
    assert issue["screen"] == "G0F00000"
    assert issue["mean_brightness"] == value
    assert "重選新的 Panel ID" in str(caught.value)
    assert (good / "W0F00000.png").exists()


def test_all_retakes_are_checked_even_if_latest_image_is_healthy(tmp_path):
    old = write_image(tmp_path, "W0F00000_old.png", 0)
    write_image(tmp_path, "W0F00000_new.png", 80)
    with pytest.raises(TrainingPanelPrecheckError) as caught:
        screen([tmp_path])
    assert caught.value.rejected_panels[0]["issues"][0]["image_name"] == old.name


def test_polygon_mean_ignores_dark_background_and_measures_raw_pixels(tmp_path, monkeypatch):
    image = np.zeros((20, 24), np.uint8)
    image[5:15, 6:18] = 80
    polygon = np.array([[6, 5], [17, 5], [17, 14], [6, 14]], np.float32)
    monkeypatch.setattr("capi_server.detect_panel_geometry", lambda *_: ((6, 5, 18, 15), polygon))
    write_image(tmp_path, "W0F00000.png", image)
    write_image(tmp_path, "G0F00000.png", image)
    result = screen([tmp_path])
    measurements = result["panels"][0]["images"]
    assert all(m["mean_brightness"] == 80 for m in measurements)
    assert all(m["mean_pixels"] == 120 for m in measurements)
    assert all(m["mean_source"] == "reference_W0F00000_polygon" for m in measurements)


def test_aapi_aliases_black_screen_and_u0f_are_checked(tmp_path):
    write_image(tmp_path, "P1Windows_BG123456.png", 80)
    write_image(tmp_path, "P1B8F00000123456.png", 0)
    write_image(tmp_path, "P1U0F00000123456.png", 0)
    with pytest.raises(TrainingPanelPrecheckError) as caught:
        screen([tmp_path], profile="aapi")
    panel = caught.value.rejected_panels[0]
    assert {m["screen"] for m in panel["images"]} == {"STANDARD", "B0F00000", "U0F00000"}
    assert [issue["screen"] for issue in panel["issues"]] == ["U0F00000"]


@pytest.mark.parametrize("kind", ["missing", "empty", "unreadable"])
def test_uncheckable_panels_do_not_silently_pass(tmp_path, kind):
    panel = tmp_path / "P1"
    if kind != "missing":
        panel.mkdir()
    if kind == "unreadable":
        (panel / "W0F00000.png").write_bytes(b"corrupt")
    with pytest.raises(TrainingPanelPrecheckError):
        screen([panel])


@pytest.mark.parametrize("change", ["replace", "add", "delete"])
def test_sources_are_pinned_and_changes_block_all_slicing(tmp_path, monkeypatch, change):
    import capi_train_new
    panel = tmp_path / "P1"
    path = write_image(panel, "W0F00000.png", 80)
    result = screen([panel])
    assert verify_training_panel_sources(result, [panel])[str(panel.resolve())] == [path]
    if change == "replace":
        path.write_bytes(b"changed and invalid")
    elif change == "add":
        write_image(panel, "G0F00000.png", 0)
    else:
        path.unlink()
    preprocess = MagicMock()
    monkeypatch.setattr(capi_train_new, "preprocess_panel_folder", preprocess)
    output = tmp_path / "output"
    cfg = capi_train_new.TrainingConfig(machine_id="M", panel_paths=[panel], over_review_root=tmp_path)
    with pytest.raises(TrainingPanelPrecheckError):
        capi_train_new.preprocess_panels_to_pool(
            "check", cfg, capi_train_new.PreprocessConfig(), MagicMock(), output,
            lambda _: None, image_precheck=result,
        )
    preprocess.assert_not_called()
    assert not output.exists()


def make_handler(tmp_path, paths, source="inference_records"):
    server = SimpleNamespace(
        database=MagicMock(), configs_by_machine={}, fallback_config=CAPIConfig(),
        station_adapter=create_station_adapter("capi"), path_mapping={},
    )
    server.database.get_all_config_params.return_value = []
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler._capi_server_instance = server
    handler.db = server.database
    handler.path = "/api/train/new/start"
    payload = {"machine_id": "M", "panel_paths": [str(p) for p in paths]}
    if source == "manual_folder":
        payload["training_data_source"] = {
            "type": source, "confirmed_normal": True, "batch_root": str(tmp_path),
        }
    body = json.dumps(payload).encode()
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.responses = []
    handler._send_json = lambda data, status=200: handler.responses.append((status, data))
    return handler, server


@pytest.mark.parametrize("source", ["inference_records", "manual_folder"])
def test_api_rejects_whole_selection_before_job_thread_or_tiles(tmp_path, monkeypatch, source):
    good, bad = tmp_path / "GOOD", tmp_path / "BAD"
    write_image(good, "W0F00000.png", 80)
    write_image(bad, "W0F00000.png", 80)
    write_image(bad, "B0F00000.png", 40)  # Not a training lighting or AOI report screen.
    h, server = make_handler(tmp_path, [good, bad], source)
    thread = MagicMock()
    monkeypatch.setattr("capi_web.threading.Thread", thread)
    h._handle_train_new_start()
    status, body = h.responses[0]
    assert status == 422
    assert body["code"] == "training_panel_image_abnormal"
    assert [p["panel_id"] for p in body["rejected_panels"]] == ["BAD"]
    assert body["rejected_panels"][0]["issues"][0]["screen"] == "B0F00000"
    server.database.create_training_job.assert_not_called()
    server.database.insert_tile_pool.assert_not_called()
    thread.assert_not_called()


def test_new_panel_id_can_start_and_saved_settings_apply_without_inference_reload(tmp_path, monkeypatch):
    panel = tmp_path / "REPLACEMENT"
    write_image(panel, "W0F00000.png", 50)  # Passes only after saved DB override.
    h, server = make_handler(tmp_path, [panel])
    server.database.get_all_config_params.return_value = [
        {"param_name": "image_abnormal_w0f00000_mean_lower", "decoded_value": 40},
        {"param_name": "image_abnormal_w0f00000_mean_upper", "decoded_value": 60},
        {"param_name": "image_abnormal_detection_enabled", "decoded_value": False},
    ]
    thread = MagicMock()
    monkeypatch.setattr("capi_web.threading.Thread", thread)
    h._handle_train_new_start()
    assert h.responses[0][0] == 200
    data_source = server.database.create_training_job.call_args.kwargs["training_data_source"]
    assert data_source["image_precheck"]["panels"][0]["passed"]
    assert data_source["image_precheck"]["limits"]["W0F00000"] == [40, 60]
    assert server.fallback_config.image_abnormal_w0f00000_mean_lower == 70
    thread.return_value.start.assert_called_once()


def test_settings_read_failure_never_starts_slicing(tmp_path, monkeypatch):
    panel = tmp_path / "P1"
    write_image(panel, "W0F00000.png", 80)
    h, server = make_handler(tmp_path, [panel])
    server.database.get_all_config_params.side_effect = RuntimeError("settings unavailable")
    thread = MagicMock()
    monkeypatch.setattr("capi_web.threading.Thread", thread)
    h._handle_train_new_start()
    assert h.responses[0][0] == 503
    server.database.create_training_job.assert_not_called()
    thread.assert_not_called()


def test_submodel_panel_entry_cannot_bypass_precheck(tmp_path, monkeypatch):
    panel = tmp_path / "BAD"
    write_image(panel, "W0F00000.png", 0)
    h, server = make_handler(tmp_path, [panel])
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text('{}')
    server.database.get_model_bundle.return_value = {"machine_id": "M", "bundle_path": str(bundle)}
    h.path = "/api/models/1/retrain-submodel-with-panels"
    body = json.dumps({"panel_paths": [str(panel)], "lighting": "W0F00000", "zone": "inner"}).encode()
    h.headers = {"Content-Length": str(len(body))}
    h.rfile = io.BytesIO(body)
    thread = MagicMock()
    monkeypatch.setattr("capi_web.threading.Thread", thread)
    h._handle_models_retrain_submodel_with_panels()
    assert h.responses[0][0] == 422
    server.database.create_training_job.assert_not_called()
    thread.assert_not_called()
