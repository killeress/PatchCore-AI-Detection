"""Low-contrast B0F pressure marks, detector provenance and formal reporting."""
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from capi_config import CAPIConfig
from capi_dog import BLACK_MURA_PARAMS, detect_black_mura, dog_residual_crop
from capi_inference import AOIReportDefect, CAPIInferencer, ImageResult, TileInfo
from capi_preprocess import PreprocessConfig
from capi_server import _iter_qjpg_defect_records, aggregate_judgment, results_to_db_data
from capi_tile_diagnostics import decision_evidence
from capi_white_spot_debug import prepare_white_spot
from test_station_feature_parity import handler_factory


SAMPLES = Path(__file__).resolve().parents[1] / "HM_黑畫面特例"
CASES = [
    ("B0F00000120741.tif", (1633, 2231), (1196, 670, 5420, 3060), .9802, 1815),
    ("B0F00000161238.tif", (1460, 2399), (1015, 880, 5224, 3254), 1.0849, 2082),
]


def worker(**overrides):
    instance = CAPIInferencer.__new__(CAPIInferencer)
    overrides.setdefault("black_mura_enabled", True)  # Detection cases explicitly opt in.
    instance.config = CAPIConfig(
        skip_files=["B0F00000"], bright_spot_diff_threshold=2,
        bright_spot_min_area=2, aoi_judgment_roi_enabled=True,
        aoi_judgment_radius_px=50, inference_rotate_180_enabled=False,
        **overrides,
    )
    instance.threshold = .5
    return instance


def make_result(image, path, bounds):
    return ImageResult(
        image_path=Path(path), image_size=(image.shape[1], image.shape[0]),
        otsu_bounds=bounds, raw_bounds=bounds, exclusion_regions=[], tiles=[],
        excluded_tile_count=0, processed_tile_count=0, processing_time=0,
    )


def create_tile(instance, image, result, center, architecture):
    defect = AOIReportDefect("C1111", *center, "B0F00000", coordinate_space="image")
    if architecture == "v1":
        result.tiles, edges = instance._create_aoi_coord_tiles(image, result, [defect], (1920, 1080))
        assert not edges
    else:
        instance._create_aoi_centered_tiles_v2(
            image, result, [defect], (1920, 1080), PreprocessConfig(), is_skip_file=True,
        )
    tile, = result.tiles
    return tile


@pytest.mark.parametrize("architecture", ["v1", "v2"])
@pytest.mark.parametrize("name,center,bounds,peak,area", CASES)
def test_supplied_pressure_marks_reach_formal_ng(architecture, name, center, bounds, peak, area):
    path = SAMPLES / name
    if not path.exists():
        pytest.skip("Local customer TIFF is not included in the repository")
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    instance = worker(black_mura_enabled=False)
    result = make_result(image, path, bounds)
    old_tile = create_tile(instance, image, result, center, architecture)
    old_score, _ = instance._detect_bright_spots(old_tile)
    assert old_score == 0

    instance.config.black_mura_enabled = True
    result.tiles = []
    tile = create_tile(instance, image, result, center, architecture)
    score, amap = instance._detect_bright_spots(tile)
    evidence = tile.decision_context["black_mura"]
    assert evidence["max_residual"] == pytest.approx(peak, abs=.001)
    assert evidence["candidate_pixels"] == area
    assert not np.any(tile.bright_spot_point_mask)
    tile.score_threshold = .5
    result.anomaly_tiles = [(tile, score, amap)]
    instance._apply_aoi_judgment_roi([result], (1920, 1080))

    assert result.anomaly_tiles[0][1] == 1
    assert aggregate_judgment([result])[0] == "NG"
    records = _iter_qjpg_defect_records([result], (1920, 1080), instance.config)
    assert records and all("B0F00000" in record for record in records)
    stored = results_to_db_data([result], {})[0]
    assert stored["is_ng"] == 1
    context = json.loads(stored["tiles"][0]["decision_context"])
    assert context["black_mura"] == evidence
    explanation = decision_evidence(stored["tiles"][0])
    assert "壓傷" in explanation["rule_label"]
    assert any("DoG" in metric["label"] for metric in explanation["metrics"])


@pytest.mark.parametrize("name,center,bounds,peak,area", CASES)
def test_supplied_image_background_controls(name, center, bounds, peak, area):
    path = SAMPLES / name
    if not path.exists():
        pytest.skip("Local customer TIFF is not included in the repository")
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    polygon = CAPIInferencer._rect_polygon_from_bounds(bounds)
    for cx, cy in [(2600, 1500), (3800, 1800), (4500, 2500), (2000, 2700), (3000, 2400), (2000, 1200)]:
        kept, evidence = detect_black_mura(
            image, (cx-256, cy-256, cx+256, cy+256), panel_polygon=polygon,
        )
        assert not kept.any(), (name, cx, cy, evidence)


def synthetic_pressure():
    y, x = np.mgrid[:1200, :1400]
    u, v = (x-700)*.8 + (y-600)*.6, -(x-700)*.6 + (y-600)*.8
    return np.rint(7 + .0005*x + .0002*y + 1.8*np.exp(-.5*((u/15)**2 + (v/35)**2))).astype(np.uint8)


@pytest.mark.parametrize("architecture", ["v1", "v2"])
def test_faint_synthetic_pressure_survives_roi_and_reports_ng(architecture):
    image = synthetic_pressure()
    instance = worker()
    result = make_result(image, "B0F00000_test.tif", (100, 100, 1300, 1100))
    tile = create_tile(instance, image, result, (700, 600), architecture)
    before = tile.image.copy()
    score, amap = instance._detect_bright_spots(tile)
    np.testing.assert_array_equal(tile.image, before)
    assert not tile.bright_spot_point_mask.any()
    assert score == 1
    tile.score_threshold = .5
    result.anomaly_tiles = [(tile, score, amap)]
    instance._apply_aoi_judgment_roi([result], (1920, 1080))
    assert aggregate_judgment([result])[0] == "NG"


@pytest.mark.parametrize("architecture", ["v1", "v2"])
def test_process_panel_entry_reaches_pressure_detector(tmp_path, monkeypatch, architecture):
    image = synthetic_pressure()
    white = np.zeros_like(image)
    white[100:1100, 100:1300] = 30
    assert cv2.imwrite(str(tmp_path / "W0F00000_test.tif"), white)
    assert cv2.imwrite(str(tmp_path / "B0F00000_test.tif"), image)
    config = worker().config
    config.is_new_architecture = architecture == "v2"
    config.model_mapping = {"W0F00000": {"inner": "unused_inner.pt", "edge": "unused_edge.pt"}}
    config.grid_tiling_enabled = False
    config.otsu_bottom_crop = 0
    config.aoi_coord_inspection_enabled = True
    config.scratch_classifier_enabled = False
    instance = CAPIInferencer(config)
    monkeypatch.setattr(instance, "_get_model_for", lambda *a, **kw: None)
    monkeypatch.setattr(instance, "_get_inferencer_for_prefix", lambda *a: None)
    report = {"B0F00000": [AOIReportDefect("C1111", 960, 540, "B0F00000")]}
    results, *_ = instance.process_panel(tmp_path, cpu_workers=1, aoi_report_override=report)
    black, = [result for result in results if result.image_path.name.startswith("B0F")]
    tile, score, amap = black.anomaly_tiles[0]
    assert tile.decision_context["black_mura"]["candidate_pixels"] >= 100
    assert score == 1
    assert np.count_nonzero(amap) >= 100
    assert aggregate_judgment(results)[0] == "NG"


def test_black_debug_executes_the_same_pressure_detector(handler_factory):
    from capi_station_adapter import create_station_adapter

    handler, path = handler_factory(create_station_adapter("capi"), "B0F00000")
    image = synthetic_pressure()
    assert cv2.imwrite(str(path), image)
    assert cv2.imwrite(str(path.with_name("W0F00000_test.tif")), np.full_like(image, 30))
    instance = worker()
    instance._find_raw_object_bounds = lambda *a: ((100, 100, 1300, 1100), None)
    instance.calculate_otsu_bounds = lambda *a, **kw: ((100, 100, 1300, 1100), None, None)
    handler.inferencer = instance
    body = json.dumps({"image_path": str(path), "product_x": 960, "product_y": 540}).encode()
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Length": str(len(body))}
    handler._handle_debug_bright_spot_inference()
    status, response = handler.responses[-1]
    assert response.get("success"), response
    assert status == 200
    assert response["judgment"] == "NG"
    assert response["black_mura"]["candidate_pixels"] >= 100


@pytest.mark.parametrize("scene", ["flat", "gradient", "noise", "panel_edge", "isolated_hot_pixel"])
def test_normal_background_and_panel_edge_do_not_trigger_mura(scene):
    y, x = np.mgrid[:1100, :1200]
    image = np.full((1100, 1200), 8, np.uint8)
    polygon = None
    bounds = (344, 294, 856, 806)
    if scene == "gradient":
        image = (8 + .003*x + .002*y).astype(np.float32)
    elif scene == "noise":
        image = np.rint(8 + np.random.default_rng(12).normal(0, .5, image.shape)).astype(np.uint8)
    elif scene == "panel_edge":
        image[:] = 0
        polygon = np.array([[250, 200], [1000, 210], [990, 900], [260, 910]])
        cv2.fillPoly(image, [polygon], 20)
        bounds = (200, 250, 712, 762)
    elif scene == "isolated_hot_pixel":
        image[550, 600] = 255
    kept, evidence = detect_black_mura(image, bounds, panel_polygon=polygon)
    assert not kept.any()
    assert evidence["candidate_pixels"] == 0


def test_production_crop_matches_debug_isotropic_filter():
    image = synthetic_pressure()
    expected = prepare_white_spot(image, (700, 600), params=BLACK_MURA_PARAMS)
    residual, context = dog_residual_crop(image, tuple(expected["crop_bounds"]), BLACK_MURA_PARAMS)
    np.testing.assert_array_equal(residual, expected["residual"])
    assert list(context) == expected["context_bounds"]
    assert residual.dtype == np.float32


@pytest.mark.parametrize("prefix,enabled,aoi", [
    ("B0F00000", False, True), ("W0F00000", True, True),
    ("BWFHALF0", True, True), ("B0F00000", True, False),
])
def test_mura_scope_and_switch(prefix, enabled, aoi):
    instance = worker(black_mura_enabled=enabled)
    image = synthetic_pressure()
    tile = TileInfo(0, 444, 344, 512, 512, image[344:856, 444:956], is_aoi_coord_tile=aoi)
    instance._prepare_black_mura_tile(tile, image, image_prefix=prefix)
    assert tile.black_mura_mask is None


def split_evidence_case():
    instance = worker()
    image = np.zeros((64, 64), np.uint8)
    tile = TileInfo(0, 0, 0, 64, 64, image, is_bright_spot_detection=True,
                    bright_spot_min_area=2, black_mura_mask=np.zeros_like(image, dtype=bool),
                    bright_spot_point_mask=np.zeros_like(image, dtype=bool),
                    decision_context={"black_mura": {"min_area": 100}})
    tile.black_mura_mask[10:26, 10:26] = True
    tile.bright_spot_point_mask[40, 40:42] = True
    binary = tile.black_mura_mask | tile.bright_spot_point_mask
    return instance, tile, binary


def test_dust_rechecks_mura_area_without_raising_the_point_minimum(monkeypatch):
    instance, tile, binary = split_evidence_case()
    dust = np.zeros(binary.shape, np.uint8)
    dust[10:21, 10:26] = 255  # 80 DoG pixels remain: below 100, above point minimum.
    monkeypatch.setattr(instance, "generate_dust_iou_debug_image", lambda *a, **kw: None)
    remaining, _ = instance._filter_bright_spot_dust(tile, binary.astype(np.float32), dust)
    np.testing.assert_array_equal(remaining > 0, tile.bright_spot_point_mask)
    assert tile.is_suspected_dust_or_scratch is False
    tile.bright_spot_point_mask[:] = False
    remaining, _ = instance._filter_bright_spot_dust(tile, binary.astype(np.float32), dust)
    assert not remaining.any()
    assert tile.is_suspected_dust_or_scratch is True


def test_roi_rechecks_mura_area_after_clipping():
    instance, tile, binary = split_evidence_case()
    instance.config.aoi_judgment_radius_px = 4
    tile.is_aoi_coord_tile = True
    tile.aoi_product_x = tile.aoi_product_y = 17
    tile.aoi_image_x = tile.aoi_image_y = 17
    tile.score_threshold = .5
    result = make_result(tile.image, "B0F00000_test.tif", (0, 0, 64, 64))
    result.tiles = [tile]
    result.anomaly_tiles = [(tile, 1., binary.astype(np.float32))]
    instance._apply_aoi_judgment_roi([result], (64, 64))
    assert result.anomaly_tiles[0][1] == 0
    assert not result.anomaly_tiles[0][2].any()


def test_mura_parameters_roundtrip_and_database_override():
    config = CAPIConfig()
    assert (config.black_mura_enabled, config.black_mura_threshold, config.black_mura_min_area) == (False, .5, 100)
    assert CAPIConfig.from_dict({}).black_mura_enabled is False
    config.apply_db_overrides([
        {"param_name": "black_mura_enabled", "decoded_value": "true"},
    ])
    assert CAPIConfig.from_dict(config.to_dict()).black_mura_enabled is True
    config.apply_db_overrides([
        {"param_name": "black_mura_enabled", "decoded_value": "false"},
        {"param_name": "black_mura_threshold", "decoded_value": .75},
        {"param_name": "black_mura_min_area", "decoded_value": 200},
    ])
    restored = CAPIConfig.from_dict(config.to_dict())
    assert (restored.black_mura_enabled, restored.black_mura_threshold, restored.black_mura_min_area) == (False, .75, 200)
