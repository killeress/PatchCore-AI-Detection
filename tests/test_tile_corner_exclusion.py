from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_config import CAPIConfig
from capi_database import CAPIDatabase
from capi_inference import CAPIInferencer, TileInfo


def _inferencer(config: CAPIConfig) -> CAPIInferencer:
    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config
    inferencer.threshold = 0.5
    return inferencer


def _tile(size: int = 512) -> TileInfo:
    return TileInfo(
        tile_id=1,
        x=0,
        y=0,
        width=size,
        height=size,
        image=np.zeros((size, size), dtype=np.uint8),
    )


def _base_config() -> CAPIConfig:
    config = CAPIConfig()
    config.patchcore_filter_enabled = False
    config.patchcore_concentration_enabled = False
    config.patchcore_diffuse_area_enabled = False
    config.edge_margin_px = 0
    return config


def test_tile_corner_exclusion_defaults_off_and_preserves_prediction():
    config = _base_config()
    anomaly_map = np.zeros((64, 64), dtype=np.float32)
    anomaly_map[0, 0] = 1.0

    score, result_map = _inferencer(config).predict_tile(
        _tile(),
        raw_prediction=(1.0, anomaly_map.copy()),
    )

    assert config.tile_corner_exclusion_enabled is False
    assert config.tile_corner_exclusion_size_px == 32
    assert score == pytest.approx(1.0)
    assert np.array_equal(result_map, anomaly_map)


def test_tile_corner_exclusion_masks_four_scaled_squares_and_recalculates_score():
    config = _base_config()
    config.tile_corner_exclusion_enabled = True
    config.tile_corner_exclusion_size_px = 32
    anomaly_map = np.zeros((64, 64), dtype=np.float32)
    anomaly_map[:4, :4] = 1.0
    anomaly_map[:4, -4:] = 1.0
    anomaly_map[-4:, :4] = 1.0
    anomaly_map[-4:, -4:] = 1.0
    anomaly_map[30, 30] = 0.4

    score, result_map = _inferencer(config).predict_tile(
        _tile(),
        raw_prediction=(1.0, anomaly_map),
    )

    assert np.count_nonzero(result_map[:4, :4]) == 0
    assert np.count_nonzero(result_map[:4, -4:]) == 0
    assert np.count_nonzero(result_map[-4:, :4]) == 0
    assert np.count_nonzero(result_map[-4:, -4:]) == 0
    assert result_map[30, 30] == pytest.approx(0.4)
    assert score == pytest.approx(0.4)


def test_tile_corner_exclusion_size_is_adjustable():
    config = _base_config()
    config.tile_corner_exclusion_enabled = True
    config.tile_corner_exclusion_size_px = 64
    anomaly_map = np.ones((64, 64), dtype=np.float32)

    _, result_map = _inferencer(config).predict_tile(
        _tile(),
        raw_prediction=(1.0, anomaly_map),
    )

    assert np.count_nonzero(result_map[:8, :8]) == 0
    assert result_map[8, 8] == pytest.approx(1.0)


def test_tile_corner_exclusion_applies_to_bright_spot_tiles():
    config = _base_config()
    config.tile_corner_exclusion_enabled = True
    config.tile_corner_exclusion_size_px = 3
    config.bright_spot_threshold = 200
    config.bright_spot_diff_threshold = 200
    config.bright_spot_min_area = 1
    config.bright_spot_median_kernel = 3
    tile = _tile(10)
    tile.image[0:2, 0:2] = 255

    score, result_map = _inferencer(config)._detect_bright_spots(tile)

    assert score == pytest.approx(0.0)
    assert np.count_nonzero(result_map) == 0


def test_tile_corner_exclusion_config_roundtrip_and_hot_reload():
    config = CAPIConfig.from_dict({
        "tile_corner_exclusion_enabled": True,
        "tile_corner_exclusion_size_px": 48,
    })

    assert config.to_dict()["tile_corner_exclusion_enabled"] is True
    assert config.to_dict()["tile_corner_exclusion_size_px"] == 48

    config.apply_db_overrides([
        {"param_name": "tile_corner_exclusion_enabled", "decoded_value": "false"},
        {"param_name": "tile_corner_exclusion_size_px", "decoded_value": 24},
    ])

    assert config.tile_corner_exclusion_enabled is False
    assert config.tile_corner_exclusion_size_px == 24


def test_tile_corner_exclusion_database_defaults_and_settings_ui(tmp_path):
    db = CAPIDatabase(str(tmp_path / "corner-settings.db"))
    db.init_config_from_yaml(CAPIConfig())

    enabled = db.get_config_param("tile_corner_exclusion_enabled")
    size = db.get_config_param("tile_corner_exclusion_size_px")
    assert enabled["param_type"] == "bool"
    assert enabled["decoded_value"] is False
    assert size["param_type"] == "int"
    assert size["decoded_value"] == 32

    settings_html = (
        Path(__file__).resolve().parent.parent / "templates" / "settings.html"
    ).read_text(encoding="utf-8")
    patchcore_start = settings_html.index("const PATCHCORE_PARAMS")
    patchcore_end = settings_html.index("];", patchcore_start)
    patchcore_params = settings_html[patchcore_start:patchcore_end]
    assert "'tile_corner_exclusion_enabled'" in patchcore_params
    assert "'tile_corner_exclusion_size_px'" in patchcore_params
    assert "paramLockMap['tile_corner_exclusion_size_px']" in settings_html
    assert "'tile_corner_exclusion_enabled'," in settings_html[
        settings_html.index("const LOCK_AFFECTING_BOOL_PARAMS"):
    ]
