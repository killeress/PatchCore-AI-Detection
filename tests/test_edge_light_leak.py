from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from capi_edge_cv import EdgeInspectionConfig, inspect_aoi_edge_light_leak
from capi_config import CAPIConfig
from capi_database import CAPIDatabase
from capi_inference import CAPIInferencer, ImageResult, TileInfo


def _light_leak_config() -> EdgeInspectionConfig:
    config = EdgeInspectionConfig()
    config.light_leak_enabled = True
    config.light_leak_edge_distance = 20
    config.light_leak_aoi_radius = 35
    config.light_leak_threshold = 4.0
    config.light_leak_dark_threshold = 4.0
    config.light_leak_min_length = 24
    config.light_leak_boundary_offset = 5
    config.light_leak_band_width = 10
    config.light_leak_reference_gap = 10
    config.light_leak_max_dust_overlap = 0.2
    return config


def _synthetic_panel(side: str, *, leak: bool = True, dark: bool = False):
    height, width = 120, 160
    image = np.full((height, width), 100, dtype=np.uint8)
    dust = np.zeros_like(image)
    if side == "top":
        aoi = (80, 2)
        leak_slice = (slice(5, 15), slice(55, 106))
    elif side == "bottom":
        aoi = (80, 117)
        leak_slice = (slice(105, 115), slice(55, 106))
    elif side == "left":
        aoi = (2, 60)
        leak_slice = (slice(35, 86), slice(5, 15))
    elif side == "right":
        aoi = (157, 60)
        leak_slice = (slice(35, 86), slice(145, 155))
    else:
        raise AssertionError(side)
    if leak:
        image[leak_slice] = 90 if dark else 110
    polygon = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    return image, dust, polygon, aoi, leak_slice


@pytest.mark.parametrize("side", ["top", "bottom", "left", "right"])
def test_edge_light_leak_detects_continuous_bright_band_on_all_four_sides(side):
    image, dust, polygon, aoi, _ = _synthetic_panel(side)

    result = inspect_aoi_edge_light_leak(
        tile_image=image,
        tile_origin=(0, 0),
        panel_polygon=polygon,
        raw_bounds=(0, 0, image.shape[1], image.shape[0]),
        aoi_product_xy=aoi,
        product_resolution=(image.shape[1], image.shape[0]),
        dust_mask=dust,
        config=_light_leak_config(),
    )

    assert result["applicable"] is True
    assert result["detected"] is True
    assert result["side"] == side
    assert result["continuous_length"] >= 24
    assert result["max_delta"] >= 9.0
    assert result["dust_overlap"] == 0.0


@pytest.mark.parametrize("side", ["top", "bottom", "left", "right"])
def test_edge_light_leak_detects_continuous_dark_drop_on_all_four_sides(side):
    image, dust, polygon, aoi, _ = _synthetic_panel(side, dark=True)

    result = inspect_aoi_edge_light_leak(
        tile_image=image,
        tile_origin=(0, 0),
        panel_polygon=polygon,
        raw_bounds=(0, 0, image.shape[1], image.shape[0]),
        aoi_product_xy=aoi,
        product_resolution=(image.shape[1], image.shape[0]),
        dust_mask=dust,
        config=_light_leak_config(),
    )

    assert result["applicable"] is True
    assert result["detected"] is True
    assert result["side"] == side
    assert result["anomaly_type"] == "DARK_DROP"
    assert result["continuous_length"] >= 24
    assert result["max_delta"] >= 9.0
    assert result["dust_overlap"] == 0.0


def test_edge_light_leak_keeps_normal_edge_ok():
    image, dust, polygon, aoi, _ = _synthetic_panel("bottom", leak=False)

    result = inspect_aoi_edge_light_leak(
        tile_image=image,
        tile_origin=(0, 0),
        panel_polygon=polygon,
        raw_bounds=(0, 0, image.shape[1], image.shape[0]),
        aoi_product_xy=aoi,
        product_resolution=(image.shape[1], image.shape[0]),
        dust_mask=dust,
        config=_light_leak_config(),
    )

    assert result["applicable"] is True
    assert result["detected"] is False
    assert result["continuous_length"] == 0


def test_edge_light_leak_does_not_rescue_band_covered_by_dust():
    image, dust, polygon, aoi, leak_slice = _synthetic_panel("bottom")
    dust[leak_slice] = 255

    result = inspect_aoi_edge_light_leak(
        tile_image=image,
        tile_origin=(0, 0),
        panel_polygon=polygon,
        raw_bounds=(0, 0, image.shape[1], image.shape[0]),
        aoi_product_xy=aoi,
        product_resolution=(image.shape[1], image.shape[0]),
        dust_mask=dust,
        config=_light_leak_config(),
    )

    assert result["continuous_length"] >= 24
    assert result["dust_overlap"] > 0.9
    assert result["detected"] is False
    assert result["reason"] == "dust_overlap"


@pytest.mark.parametrize(
    ("dark", "anomaly_type"),
    [(False, "BRIGHT_LEAK"), (True, "DARK_DROP")],
)
def test_formal_dust_postprocess_rescues_aoi_edge_light_leak(dark, anomaly_type):
    image, dust, polygon, aoi, _ = _synthetic_panel("bottom", dark=dark)
    config = CAPIConfig()
    config.dust_two_stage_enabled = False
    edge_config = _light_leak_config()

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config
    inferencer.edge_inspector = SimpleNamespace(config=edge_config)

    tile = TileInfo(
        tile_id=1,
        x=0,
        y=0,
        width=image.shape[1],
        height=image.shape[0],
        image=image.copy(),
        original_image=image.copy(),
        is_aoi_coord_tile=True,
        aoi_product_x=aoi[0],
        aoi_product_y=aoi[1],
        aoi_image_x=aoi[0],
        aoi_image_y=aoi[1],
        score_threshold=0.28,
    )
    anomaly_map = np.ones((12, 16), dtype=np.float32)
    result = ImageResult(
        image_path=Path("W0F00000_edge_leak.tif"),
        image_size=(image.shape[1], image.shape[0]),
        otsu_bounds=(0, 0, image.shape[1], image.shape[0]),
        exclusion_regions=[],
        tiles=[tile],
        excluded_tile_count=0,
        processed_tile_count=1,
        processing_time=0.0,
        anomaly_tiles=[(tile, 0.66, anomaly_map)],
        raw_bounds=(0, 0, image.shape[1], image.shape[0]),
        panel_polygon=polygon,
    )

    inferencer._check_dust_or_scratch_feature_with_context = (
        lambda *args, **kwargs: (True, dust.copy(), 0.01, "OMIT dust")
    )
    inferencer.check_dust_per_region = lambda *args, **kwargs: (
        False,
        None,
        1.0,
        [{"is_dust": True, "coverage": 1.0}],
        np.ones_like(anomaly_map, dtype=np.uint8) * 255,
        np.ones_like(anomaly_map, dtype=np.int32),
    )
    inferencer.generate_dust_iou_debug_image = (
        lambda *args, **kwargs: np.zeros((8, 8, 3), dtype=np.uint8)
    )

    inferencer._apply_omit_dust_postprocess(
        [result],
        omit_image=np.zeros_like(image),
        omit_overexposed=False,
        omit_overexposure_info="",
        cpu_workers=1,
        product_resolution=(image.shape[1], image.shape[0]),
    )

    assert tile.is_suspected_dust_or_scratch is False
    assert tile.edge_light_leak_result["detected"] is True
    assert tile.edge_light_leak_result["side"] == "bottom"
    assert tile.edge_light_leak_result["anomaly_type"] == anomaly_type
    assert tile.anomaly_peak_source == "aoi_edge_light_leak"
    assert f"EDGE_LIGHT_LEAK_RESCUE: {anomaly_type} BOTTOM" in tile.dust_detail_text


@pytest.mark.parametrize(
    ("dark", "anomaly_type", "anomaly_type_zh"),
    [
        (False, "BRIGHT_LEAK", "邊緣亮帶"),
        (True, "DARK_DROP", "邊緣暗段"),
    ],
)
def test_debug_dust_pipeline_reports_edge_light_leak_rescue(
    dark, anomaly_type, anomaly_type_zh
):
    from capi_web import CAPIWebHandler

    image, dust, polygon, aoi, _ = _synthetic_panel("bottom", dark=dark)
    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = CAPIConfig()
    inferencer.config.dust_two_stage_enabled = False
    inferencer.edge_inspector = SimpleNamespace(config=_light_leak_config())
    inferencer.check_omit_overexposure = (
        lambda _image: (False, 80.0, 0.0, "正常")
    )
    inferencer._check_dust_or_scratch_feature_with_context = (
        lambda *args, **kwargs: (True, dust.copy(), 0.01, "OMIT dust")
    )
    inferencer.check_dust_per_region = lambda *args, **kwargs: (
        False,
        None,
        1.0,
        [{"is_dust": True, "coverage": 1.0}],
        np.ones((12, 16), dtype=np.uint8) * 255,
        np.ones((12, 16), dtype=np.int32),
    )
    inferencer.generate_dust_iou_debug_image = (
        lambda *args, **kwargs: np.zeros((8, 8, 3), dtype=np.uint8)
    )

    tile = TileInfo(
        tile_id=1,
        x=0,
        y=0,
        width=image.shape[1],
        height=image.shape[0],
        image=image.copy(),
        original_image=image.copy(),
        is_aoi_coord_tile=True,
        aoi_product_x=aoi[0],
        aoi_product_y=aoi[1],
        aoi_image_x=aoi[0],
        aoi_image_y=aoi[1],
        score_threshold=0.28,
    )
    handler = object.__new__(CAPIWebHandler)
    handler.inferencer = inferencer

    payload, _anomaly, _dust = handler._run_debug_coord_dust_pipeline(
        tile_info=tile,
        tile_image=image,
        anomaly_map=np.ones((12, 16), dtype=np.float32),
        score=0.66,
        score_threshold=0.28,
        omit_image=np.zeros_like(image),
        omit_crop=np.zeros_like(image),
        product_resolution=(image.shape[1], image.shape[0]),
        model_id="TEST",
        panel_polygon=polygon,
        raw_bounds=(0, 0, image.shape[1], image.shape[0]),
    )

    assert payload["final_judgment"] == "NG"
    assert payload["dust_filter_result"] == "EDGE_LIGHT_LEAK_RESCUE"
    assert payload["edge_light_leak"]["detected"] is True
    assert payload["edge_light_leak"]["side_zh"] == "下邊"
    assert payload["edge_light_leak"]["anomaly_type"] == anomaly_type
    assert payload["edge_light_leak"]["anomaly_type_zh"] == anomaly_type_zh
    assert tile.edge_light_leak_debug_image is not None


def test_edge_light_leak_settings_and_debug_ui_are_exposed():
    root = Path(__file__).resolve().parents[1]
    settings = (root / "templates" / "settings.html").read_text(encoding="utf-8")
    debug = (root / "templates" / "debug_inference.html").read_text(encoding="utf-8")
    database = (root / "capi_database.py").read_text(encoding="utf-8")

    assert "邊緣漏光檢測" in settings
    assert settings.index("邊緣漏光檢測") < settings.index("CV 邊緣檢測設定")
    assert "cv_edge_light_leak_enabled" in settings
    assert "cv_edge_light_leak_dark_threshold" in settings
    assert "cv_edge_light_leak_max_dust_overlap" in settings
    assert database.index("cv_edge_light_leak_enabled") < database.index("cv_edge_enabled")
    assert "邊緣漏光救援判定" in debug
    assert "edge_light_leak_url" in debug


def test_edge_light_leak_config_loads_independently_from_cv_main_switch():
    config = EdgeInspectionConfig.from_db_params({
        "cv_edge_enabled": {"decoded_value": False},
        "cv_edge_light_leak_enabled": {"decoded_value": True},
        "cv_edge_light_leak_threshold": {"decoded_value": 5.5},
        "cv_edge_light_leak_dark_threshold": {"decoded_value": 6.5},
        "cv_edge_light_leak_min_length": {"decoded_value": 42},
        "cv_edge_light_leak_max_dust_overlap": {"decoded_value": 0.15},
    })

    assert config.enabled is False
    assert config.light_leak_enabled is True
    assert config.light_leak_threshold == pytest.approx(5.5)
    assert config.light_leak_dark_threshold == pytest.approx(6.5)
    assert config.light_leak_min_length == 42
    assert config.light_leak_max_dust_overlap == pytest.approx(0.15)


def test_edge_light_leak_database_defaults_are_created(tmp_path):
    database = CAPIDatabase(str(tmp_path / "edge_light_leak.db"))
    database.init_config_from_yaml(CAPIConfig())

    enabled = database.get_config_param("cv_edge_light_leak_enabled")
    threshold = database.get_config_param("cv_edge_light_leak_threshold")
    dark_threshold = database.get_config_param(
        "cv_edge_light_leak_dark_threshold"
    )
    dust_overlap = database.get_config_param(
        "cv_edge_light_leak_max_dust_overlap"
    )

    assert enabled["decoded_value"] is False
    assert threshold["decoded_value"] == pytest.approx(4.0)
    assert dark_threshold["decoded_value"] == pytest.approx(4.0)
    assert dust_overlap["decoded_value"] == pytest.approx(0.2)
