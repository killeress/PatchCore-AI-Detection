import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer, ImageResult, TileInfo


def test_high_coverage_region_is_dust_even_when_peak_is_outside_dust_mask():
    """Heatmap smoothing can move the max pixel slightly outside the OMIT dust mask."""
    config = CAPIConfig()
    config.dust_heatmap_iou_threshold = 0.2
    config.dust_high_cov_threshold = 0.5
    config.dust_peak_fraction_threshold = 0.8

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config

    anomaly_map = np.ones((100, 100), dtype=np.float32)
    anomaly_map[50, 99] = 2.0  # peak outside the dust mask

    dust_mask = np.zeros((100, 100), dtype=np.uint8)
    dust_mask[:, :97] = 255  # 97% of the heat region overlaps dust

    has_real, real_peak, _overall_iou, details, _heat_binary, _labels = (
        inferencer.check_dust_per_region(
            dust_mask,
            anomaly_map,
            top_percent=100.0,
            metric="coverage",
            iou_threshold=config.dust_heatmap_iou_threshold,
        )
    )

    assert has_real is False
    assert real_peak is None
    assert len(details) == 1
    assert details[0]["coverage"] == 0.97
    assert details[0]["peak_in_dust"] is False
    assert details[0]["dust_sub_peak_rescue"] is False
    assert details[0]["is_dust"] is True


def test_force_include_seed_keeps_aoi_center_from_being_dropped_by_top_percent():
    """AOI center can be a weaker hot spot than dust/tape, but must still be evaluated."""
    config = CAPIConfig()
    config.dust_heatmap_iou_threshold = 0.2
    config.dust_high_cov_threshold = 0.5
    config.dust_peak_fraction_threshold = 0.8

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config

    anomaly_map = np.zeros((100, 100), dtype=np.float32)
    anomaly_map[88:94, 88:94] = 100.0  # strongest hot region on dust
    anomaly_map[49:52, 49:52] = 20.0   # AOI center hot region, below top-percent cutoff

    dust_mask = np.zeros((100, 100), dtype=np.uint8)
    dust_mask[85:97, 85:97] = 255

    has_real_without_seed, _peak_without_seed, _iou, details_without_seed, _hm, _labels = (
        inferencer.check_dust_per_region(
            dust_mask,
            anomaly_map,
            top_percent=0.1,
            metric="coverage",
            iou_threshold=config.dust_heatmap_iou_threshold,
        )
    )

    has_real_with_seed, real_peak, _iou, details_with_seed, heat_binary, _labels = (
        inferencer.check_dust_per_region(
            dust_mask,
            anomaly_map,
            top_percent=0.1,
            metric="coverage",
            iou_threshold=config.dust_heatmap_iou_threshold,
            force_include_yx=(50, 50),
            force_include_radius=2,
            force_include_min_score=10.0,
        )
    )

    assert has_real_without_seed is False
    assert all(r["is_dust"] for r in details_without_seed)

    assert has_real_with_seed is True
    assert real_peak == (49, 49)
    assert any(not r["is_dust"] for r in details_with_seed)
    assert heat_binary[50, 50] == 255


def test_aoi_center_seed_can_be_disabled_by_config():
    config = CAPIConfig()
    config.aoi_heatmap_center_seed_enabled = False

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config

    tile = TileInfo(
        tile_id=1,
        x=0,
        y=0,
        width=100,
        height=100,
        image=np.zeros((100, 100), dtype=np.uint8),
        is_aoi_coord_tile=True,
        aoi_image_x=50,
        aoi_image_y=50,
    )
    anomaly_map = np.ones((100, 100), dtype=np.float32)

    assert inferencer._aoi_center_seed_for_tile(tile, anomaly_map) == (None, 0, None)


def test_aoi_peak_prefers_center_real_region_and_bomb_keeps_it():
    config = CAPIConfig()
    config.dust_heatmap_top_percent = 5.0

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config

    tile = TileInfo(
        tile_id=1,
        x=3514,
        y=2789,
        width=512,
        height=512,
        image=np.zeros((512, 512), dtype=np.uint8),
        is_aoi_coord_tile=True,
        aoi_product_x=1136,
        aoi_product_y=872,
        aoi_image_x=3770,
        aoi_image_y=3045,
    )
    anomaly_map = np.zeros((512, 512), dtype=np.float32)
    anomaly_map[4, 0] = 1.0       # stronger tile-border artifact
    anomaly_map[254:259, 254:259] = 0.8
    anomaly_map[256, 256] = 0.9   # weaker but AOI-centered real hot spot
    tile.anomaly_peak_x = tile.x
    tile.anomaly_peak_y = tile.y + 4

    result = ImageResult(
        image_path=Path("W0F00000_test.tif"),
        image_size=(6576, 4384),
        otsu_bounds=(78, 219, 6320, 4114),
        exclusion_regions=[],
        tiles=[tile],
        excluded_tile_count=0,
        processed_tile_count=1,
        processing_time=0.0,
        anomaly_tiles=[(tile, 0.4283, anomaly_map)],
        raw_bounds=(78, 219, 6320, 4114),
    )

    inferencer._apply_aoi_peak_postprocess([result])

    assert (tile.anomaly_peak_x, tile.anomaly_peak_y) == (3770, 3045)
    assert tile.anomaly_peak_source == "aoi_real_region"

    inferencer._apply_bomb_postprocess(
        [result],
        {
            "image_prefix": "W0F00000",
            "defect_type": "point",
            "coordinates": [(363, 124), (1555, 233)],
        },
        (1920, 1200),
    )

    assert (tile.anomaly_peak_x, tile.anomaly_peak_y) == (3770, 3045)
    assert tile.anomaly_peak_source == "aoi_real_region"


def test_aoi_peak_falls_back_to_report_coordinate_without_center_hot_region():
    config = CAPIConfig()
    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config

    tile = TileInfo(
        tile_id=1,
        x=3514,
        y=2789,
        width=512,
        height=512,
        image=np.zeros((512, 512), dtype=np.uint8),
        is_aoi_coord_tile=True,
        aoi_product_x=1136,
        aoi_product_y=872,
        aoi_image_x=3770,
        aoi_image_y=3045,
    )
    anomaly_map = np.zeros((512, 512), dtype=np.float32)
    anomaly_map[4, 0] = 1.0
    result = ImageResult(
        image_path=Path("W0F00000_test.tif"),
        image_size=(6576, 4384),
        otsu_bounds=(78, 219, 6320, 4114),
        exclusion_regions=[],
        tiles=[tile],
        excluded_tile_count=0,
        processed_tile_count=1,
        processing_time=0.0,
        anomaly_tiles=[(tile, 0.4283, anomaly_map)],
        raw_bounds=(78, 219, 6320, 4114),
    )

    inferencer._apply_aoi_peak_postprocess([result])

    assert (tile.anomaly_peak_x, tile.anomaly_peak_y) == (3770, 3045)
    assert tile.anomaly_peak_source == "aoi_report_fallback"
