import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer, ImageResult, TileInfo
from capi_station_adapter import create_station_adapter
from capi_web import CAPIWebHandler
from test_debug_coord_heatmap_diagnostics import _DiagnosticInferencer


@pytest.fixture(params=["capi", "aapi"])
def station(request):
    return create_station_adapter(request.param)


def image_name(adapter, lighting):
    if adapter.profile == "aapi":
        return f"SAMPLE{lighting}082933.tif"
    return f"{lighting}_082933.tif"


@pytest.fixture
def handler_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(CAPIWebHandler, "_debug_heatmap_dir", tmp_path / "debug")

    def make(adapter, lighting="W0F00000"):
        image = tmp_path / image_name(adapter, lighting)
        assert cv2.imwrite(str(image), np.full((128, 128), 90, np.uint8))
        handler = object.__new__(CAPIWebHandler)
        handler.inferencer = _DiagnosticInferencer()
        handler.inferencer.station_adapter = adapter
        handler.inferencer._get_image_prefix = adapter.image_prefix
        handler.heatmap_manager = None
        handler._capi_server_instance = SimpleNamespace(station_adapter=adapter, path_mapping={})
        body = json.dumps({"image_path": str(image), "product_x": 64, "product_y": 64,
                           "threshold": 0.35}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}
        handler.responses = []
        handler._send_json = lambda data, status=200: handler.responses.append((status, data))
        return handler, image

    return make


def result_for(image):
    tile = TileInfo(tile_id=0, x=32, y=32, width=64, height=64,
                    image=np.full((64, 64), 90, np.uint8))
    return ImageResult(
        image_path=image, image_size=(128, 128), otsu_bounds=(0, 0, 128, 128),
        exclusion_regions=[], tiles=[tile], excluded_tile_count=0,
        processed_tile_count=1, processing_time=0,
        anomaly_tiles=[(tile, 1.0, np.ones((64, 64), np.float32))],
    )


def test_full_debug_and_fusion_receive_omit(station, handler_factory):
    handler, image = handler_factory(station)
    omit = image.parent / image_name(station, "PINIGBI0")
    expected = np.full((128, 128), 80, np.uint8)
    assert cv2.imwrite(str(omit), expected)
    result = result_for(image)
    worker = handler.inferencer
    worker.preprocess_image = lambda *a, **kw: result
    worker.run_inference = lambda *a, **kw: result
    worker.visualize_inference_result = lambda *a: np.zeros((128, 128, 3), np.uint8)
    worker.check_dust_or_scratch_feature = Mock(return_value=(True, expected[:64, :64], 1, "dust"))
    worker.compute_dust_heatmap_iou = lambda *a, **kw: (1, expected[:64, :64])
    worker.config.dust_iou_threshold = 0.3
    handler._handle_debug_inference_run()
    assert handler.responses[-1][1]["success"] is True
    assert handler.responses[-1][1]["judgment"] == "OK (DUST Filtered)"
    assert np.array_equal(result.tiles[0].omit_crop_image, expected[32:96, 32:96])

    worker.edge_inspector = SimpleNamespace(config=SimpleNamespace(aoi_edge_boundary_band_px=40))
    worker._inspect_roi_fusion = Mock(return_value=([], {}))
    handler._debug_panel_boundary = lambda *a: (None, None)
    handler._handle_api_debug_edge_corner_fusion({}, image)
    assert handler.responses[-1][1]["success"] is True
    assert np.array_equal(worker._inspect_roi_fusion.call_args.kwargs["omit_image"], expected)


@pytest.mark.parametrize("lighting", ["Windows_BG", "STANDARD"])
@pytest.mark.parametrize("method", ["_handle_debug_inference_run", "_handle_debug_coord_inference"])
def test_aapi_debug_routes_alias_to_standard_model(handler_factory, monkeypatch, lighting, method):
    handler, image = handler_factory(create_station_adapter("aapi"), lighting)
    worker = handler.inferencer
    worker.config.is_new_architecture = True
    worker.config.model_mapping = {"STANDARD": {"inner": "inner.pt", "edge": "edge.pt"}}
    worker._get_model_for = Mock(return_value=object())
    worker.run_inference_v2_single_image = Mock(return_value=result_for(image))
    worker.visualize_inference_result = lambda *a: np.zeros((128, 128, 3), np.uint8)
    monkeypatch.setattr("capi_preprocess.detect_panel_geometry", lambda *a, **kw: ((0, 0, 128, 128), None))
    monkeypatch.setattr("capi_preprocess.classify_tile_zone", lambda *a: ("inner", 1, 1, None))
    getattr(handler, method)()
    payload = handler.responses[-1][1]
    assert payload.get("success"), payload
    assert payload["image_prefix"] == "WINDOWS_BG"
    assert payload["model_name"].startswith("STANDARD")
    if method == "_handle_debug_coord_inference":
        worker._get_model_for.assert_called_once_with("TEST_MODEL", "STANDARD", "inner")
    else:
        worker.run_inference_v2_single_image.assert_called_once()


def test_black_debug_uses_white_reference(station, handler_factory):
    handler, black = handler_factory(station, "B0F00000")
    white = black.parent / image_name(station, "W0F00000")
    assert cv2.imwrite(str(white), np.full((128, 128), 180, np.uint8))
    worker = handler.inferencer
    for name in ("bright_spot_diff_threshold", "bright_spot_threshold", "bright_spot_min_area"):
        setattr(worker.config, name, 10)
    worker.config.bright_spot_median_kernel = 3
    worker.calculate_otsu_bounds = Mock(return_value=((0, 0, 128, 128), None, None))
    worker._detect_bright_spots = lambda *a: (0, None)
    handler._handle_debug_bright_spot_inference()
    payload = handler.responses[-1][1]
    assert payload.get("success"), payload
    assert payload["ref_image"] == white.name
    assert worker.calculate_otsu_bounds.call_args.kwargs["reference_raw_bounds"] == (0, 0, 128, 128)


@pytest.mark.parametrize("source,expected", [
    ("W0F00000", "W0F00000"), ("W0F00010", "W0F00010"),
    ("WGF25250", "WGF25250"), ("U0F00000", "U0F00000"),
    ("Windows_BG", "STANDARD"), ("STANDARD", "STANDARD"),
])
def test_aapi_mes_candidates_and_saved_crop_keep_model_lighting(handler_factory, source, expected, tmp_path):
    handler, image = handler_factory(create_station_adapter("aapi"), source)
    candidate = {"tile_result_id": 1, "image_result_id": 2, "image_name": image.name,
                 "image_path": str(image), "tile_x": 0, "tile_y": 0,
                 "tile_w": 64, "tile_h": 64, "zone": "inner"}
    record = {"id": 3, "glass_id": "SAMPLE", "model_id": "TEST_MODEL"}
    handler.db = SimpleNamespace(
        get_mes_comparison_record=lambda *a: record,
        get_mes_review_aoi_candidates=lambda *a: [candidate],
        get_mes_comparison_review=lambda *a: None,
        get_mes_review_aoi_candidate=lambda *a: candidate,
    )
    handler._ng_validation_base_dir = lambda: tmp_path / "validation"
    handler._handle_mes_review_candidates_api({"record_id": ["3"]})
    assert handler.responses[-1][1]["candidates"][0]["lighting"] == expected
    crops = []
    handler._send_image_array_png = crops.append
    handler._handle_mes_review_crop_api({"tile_result_id": ["1"]})
    assert crops and crops[0].shape == (512, 512)
    saved = handler._prepare_ng_validation_samples(record, [candidate])
    assert saved[0]["lighting"] == expected
    assert Path(saved[0]["crop_path"]).is_file()


@pytest.mark.parametrize("source,report,expected", [
    ("W0F00000", "W0F00000", "W0F00000"),
    ("Windows_BG", "STANDARD", "STANDARD"),
    ("U0F00000", "U0F00000", "U0F00000"),
])
def test_aapi_over_retrain_pool_matches_ric_lighting(handler_factory, tmp_path, source, report, expected):
    handler, image = handler_factory(create_station_adapter("aapi"), source)
    tile = {"id": 1, "tile_id": 0, "is_anomaly": 1, "x": 0, "y": 0, "width": 64, "height": 64}
    detail = {"id": 1, "glass_id": "SAMPLE", "images": [
        {"id": 2, "image_name": image.name, "image_path": str(image), "tiles": [tile]},
    ]}
    insert = Mock(return_value={"inserted_ids": [1], "existing_ids": []})
    handler._capi_server_instance.database = SimpleNamespace(
        get_client_accuracy_record=lambda *a: {"result_ai": "NG", "datastr": f"{report},OK;", "inference_record_id": 1},
        get_record_detail=lambda *a: detail,
        insert_over_retrain_pool_rows=insert,
        list_over_retrain_pool=lambda **kw: ([], 1),
    )
    handler._read_json_body = lambda: {"client_record_id": 1}
    handler._retrain_pool_base_dir = lambda: tmp_path / "pool"
    handler._handle_over_retrain_pool_add()
    assert handler.responses[-1][0] == 200, handler.responses
    assert insert.call_args.args[0][0]["lighting"] == expected


def test_station_skip_files_and_cli_image_lookup(station, tmp_path):
    from run_single_inference import find_omit_image
    from tools.diag_aoi_coord_inference import find_image_for_prefix
    from validate_peak_detection import _find_image, _find_omit

    cfg = CAPIConfig(skip_files=["B0F00000"])
    assert cfg.should_skip_file(image_name(station, "B0F00000"), station)
    assert not cfg.should_skip_file(image_name(station, "W0F00000"), station)
    white = tmp_path / image_name(station, "W0F00000")
    omit = tmp_path / image_name(station, "PINIGBI0")
    white.write_bytes(b"fixture")
    omit.write_bytes(b"fixture")
    assert find_omit_image(white, station) == omit
    assert _find_omit(tmp_path, station) == omit
    assert _find_image(tmp_path, "W0F00000", station) == white
    assert find_image_for_prefix(tmp_path, "W0F00000", station) == white
    assert CAPIWebHandler._resolve_debug_image_path(tmp_path, station) == white


def test_skipped_black_image_bypasses_production_dust_filter(station, tmp_path):
    worker = object.__new__(CAPIInferencer)
    worker.station_adapter = station
    worker.config = CAPIConfig(skip_files=["B0F00000"])
    worker._check_dust_or_scratch_feature_with_context = Mock(side_effect=AssertionError("black image entered dust filter"))
    result = result_for(tmp_path / image_name(station, "B0F00000"))
    worker._apply_omit_dust_postprocess([result], np.full((128, 128), 90, np.uint8), False, "")
    worker._check_dust_or_scratch_feature_with_context.assert_not_called()
    assert not result.tiles[0].is_suspected_dust_or_scratch


@pytest.mark.parametrize("source", ["Windows_BG", "STANDARD", "U0F00000", "W0F00010"])
def test_aapi_cli_resolves_alias_and_keeps_distinct_sources(tmp_path, source):
    from tools.diag_aoi_coord_inference import find_image_for_prefix
    from validate_peak_detection import _find_image

    adapter = create_station_adapter("aapi")
    images = {}
    for lighting in ("Windows_BG", "U0F00000", "W0F00010", "WGF50500"):
        path = tmp_path / image_name(adapter, lighting)
        path.write_bytes(b"fixture")
        images[lighting] = path
    expected = images["Windows_BG" if source == "STANDARD" else source]
    assert find_image_for_prefix(tmp_path, source, adapter) == expected
    assert _find_image(tmp_path, source, adapter) == expected


def test_record_labels_preserve_station_specific_u0f_mapping(station):
    detail = {"images": [
        {"image_name": image_name(station, "U0F00000")},
        {"image_name": image_name(station, "W0F00000")},
    ]}
    if station.profile == "aapi":
        detail["images"].append({"image_name": image_name(station, "Windows_BG")})
    CAPIWebHandler._decorate_record_image_prefix_labels(detail, station)
    assert detail["image_prefix_labels"] == (
        {"U0F00000": "U0F00000", "STANDARD": "WINDOWS_BG", "W0F00000": "W0F00000"}
        if station.profile == "aapi" else
        {"STANDARD": "U0F00000", "W0F00000": "W0F00000"}
    )
