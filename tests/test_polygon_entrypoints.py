"""Boundary settings must reach detection, not just be present in a config."""
import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from capi_config import CAPIConfig
from capi_preprocess import PreprocessConfig, panel_boundary_config_for_station
from capi_station_adapter import create_station_adapter


@pytest.fixture(params=["capi", "aapi"])
def raw_boundary(request, monkeypatch):
    profile = request.param
    calls = []
    polygon = np.array([[100, 100], [899, 100], [899, 799], [100, 799]], np.float32)

    def detect(image, cfg, **kwargs):
        assert cfg.large_panel_raw_boundary_enabled is True
        assert cfg.large_panel_min_width_ratio == (0.75 if profile == "capi" else 0.85)
        assert cfg.large_panel_min_height_ratio == (0.60 if profile == "capi" else 0.80)
        assert cfg.raw_boundary_max_edge_residual_p95_ratio == (0.04 if profile == "capi" else 0.03)
        calls.append(image.copy())
        return (100, 100, 900, 800), polygon.copy(), True

    monkeypatch.setattr("capi_preprocess._detect_large_panel_raw_boundary", detect)
    return profile, calls, polygon


def test_image_abnormal_precheck_uses_station_raw_boundary(tmp_path, raw_boundary):
    from capi_server import check_image_abnormal_precheck

    profile, calls, _ = raw_boundary
    image = np.zeros((900, 1000), np.uint8)
    image[100:800, 100:900] = 80
    path = tmp_path / "W0F00000_test.png"
    assert cv2.imwrite(str(path), image)
    cfg = CAPIConfig(
        image_abnormal_detection_enabled=True,
        image_abnormal_w0f00000_mean_lower=70,
        image_abnormal_w0f00000_mean_upper=90,
    )
    assert check_image_abnormal_precheck(
        tmp_path, cfg, [path], report_prefixes=["W0F00000"],
        station_profile=profile,
    ) is None
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], image)


def test_ng_crop_detects_raw_boundary_before_model_filter(tmp_path, raw_boundary):
    from capi_train_new import sample_ng_tiles

    profile, calls, _ = raw_boundary
    image = np.zeros((900, 1000), np.uint8)
    image[100:800, 100:900] = 200
    image[400, 400] = 0
    path = tmp_path / "R0F00000_test.png"
    assert cv2.imwrite(str(path), image)
    db = MagicMock()
    db.list_training_bomb_candidates.return_value = [{
        "inference_record_id": 1, "source_result_id": 2,
        "client_bomb_info": json.dumps({
            "image_prefix": "R0F00000", "defect_type": "point",
            "coordinates": [[1800, 540]],
        }),
        "image_path": str(path), "image_dir": str(tmp_path), "image_name": path.name,
        "resolution_x": 1920, "resolution_y": 1080, "otsu_bounds": "0,0,1000,900",
    }]
    cfg = PreprocessConfig(
        **panel_boundary_config_for_station(profile),
        product_resolution=(1920, 1080),
        image_preprocess_pipeline=[{"method": "gaussian", "params": {"kernel_size": 3, "sigma": 1.0}}],
    )
    stats = sample_ng_tiles(
        "j", tmp_path, db, tmp_path / "thumbs", per_lighting=1,
        lightings=("R0F00000",), machine_id="M", preprocess_cfg=cfg,
        log=lambda _: None,
    )
    assert stats["sampled"] == 1
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], image)
    tile = db.insert_tile_pool.call_args.args[1][0]
    assert tile["zone"] == "edge"
    assert tile["tile_x"] + tile["tile_width"] <= 900


@pytest.mark.parametrize("grid", [False, True])
def test_coord_debug_uses_raw_boundary(tmp_path, monkeypatch, raw_boundary, grid):
    from capi_web import CAPIWebHandler

    profile, calls, _ = raw_boundary
    image = np.zeros((900, 1000), np.uint8)
    image[100:800, 100:900] = 200
    path = tmp_path / "G0F00000_test.png"
    assert cv2.imwrite(str(path), image)
    cfg = CAPIConfig(
        machine_id="GN140HCAAD70S", enable_panel_polygon=True,
        model_mapping={"G0F00000": {"inner": "inner.pt", "edge": "edge.pt"}},
        image_preprocess_pipeline=[{"method": "gaussian", "params": {"kernel_size": 3, "sigma": 1.0}}],
    )
    cfg.grid_canonicalization_enabled = grid
    cfg.grid_product_resolution = (1920, 1080)
    cfg.is_new_architecture = True
    inferencer = SimpleNamespace(
        config=cfg, station_adapter=create_station_adapter(profile),
        _get_image_prefix=lambda _: "G0F00000",
        _find_raw_object_bounds=lambda _: ((0, 0, 1000, 900), None),
        calculate_otsu_bounds=lambda _: ((0, 0, 1000, 900), None, None),
    )
    # Stop once geometry has been resolved; this test does not load GPU models.
    def stop_after_boundary(*args, **kwargs):
        raise RuntimeError("geometry resolved")

    inferencer._map_aoi_coords = stop_after_boundary
    body = json.dumps({"image_path": str(path), "product_x": 960, "product_y": 540}).encode()
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.inferencer = inferencer
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Length": str(len(body))}
    responses = []
    handler._send_json = lambda payload, **kw: responses.append(payload)
    handler._read_inference_image = lambda *a: image.copy()
    handler._debug_product_resolution = lambda *a: {"product_resolution": (1920, 1080)}
    handler._handle_debug_coord_inference()
    assert len(calls) == 1, responses
    np.testing.assert_array_equal(calls[0], image)
    assert "geometry resolved" in responses[0]["error"]


@pytest.mark.parametrize("mode", ["cv", "patchcore", "fusion"])
def test_corner_debug_modes_use_station_raw_boundary(tmp_path, raw_boundary, mode):
    from capi_web import CAPIWebHandler
    from capi_edge_cv import EdgeInspectionConfig

    profile, calls, polygon = raw_boundary
    image = np.zeros((900, 1000), np.uint8)
    path = tmp_path / "G0F00000_test.png"
    assert cv2.imwrite(str(path), image)
    inferencer = SimpleNamespace(
        config=CAPIConfig(), station_adapter=create_station_adapter(profile),
        edge_inspector=SimpleNamespace(config=EdgeInspectionConfig()),
        _get_image_prefix=lambda _: "G0F00000",
        _inspect_roi_patchcore=MagicMock(return_value=([], {})),
        _inspect_roi_fusion=MagicMock(return_value=([], {})),
    )
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.inferencer = inferencer
    handler._read_inference_image = lambda *a: image.copy()
    handler._debug_product_resolution = lambda *a: {"product_resolution": (1920, 1080)}
    responses = []
    handler._send_json = lambda payload, **kw: responses.append(payload)
    # CV stops after geometry with an out-of-image ROI, before rendering.
    body = json.dumps({"image_path": str(path), "inspector": mode, "roi_x": 2000}).encode()
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Length": str(len(body))}
    handler._handle_api_debug_edge_inspect_corner()
    assert len(calls) == 1, responses
    if mode == "cv":
        assert "ROI 超出影像範圍" in responses[0]["error"]
    else:
        inspect = getattr(inferencer, "_inspect_roi_" + mode)
        np.testing.assert_array_equal(inspect.call_args.kwargs["panel_polygon"], polygon)


@pytest.mark.parametrize("eligible,valid", [(False, True), (True, False)])
def test_geometry_preserves_legacy_fallback(monkeypatch, eligible, valid):
    from capi_preprocess import detect_panel_geometry

    raw = np.zeros((20, 20), np.uint8)
    processed = np.ones_like(raw)
    polygon = np.array([[1, 1], [18, 1], [18, 18], [1, 18]], np.float32)
    monkeypatch.setattr("capi_preprocess._detect_large_panel_raw_boundary", lambda *a, **kw: (
        (1, 1, 19, 19), polygon if valid else None, eligible,
    ))
    def legacy(image, cfg):
        assert image is processed
        return (2, 2, 18, 18), polygon
    monkeypatch.setattr("capi_preprocess.detect_panel_polygon", legacy)
    bbox, _ = detect_panel_geometry(
        raw, PreprocessConfig(**panel_boundary_config_for_station("capi")),
        processed_image=processed,
    )
    assert bbox == (2, 2, 18, 18)
