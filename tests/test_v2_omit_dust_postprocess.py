from pathlib import Path

import numpy as np

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer, ImageResult, TileInfo


def test_v2_omit_postprocess_runs_two_stage_before_suppressing_dust():
    config = CAPIConfig()
    config.dust_two_stage_enabled = True
    config.dust_heatmap_top_percent = 0.3
    config.aoi_heatmap_center_seed_radius_px = 2.0
    config.aoi_heatmap_center_seed_min_peak_ratio = 0.2

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config

    tile = TileInfo(
        tile_id=1,
        x=0,
        y=0,
        width=16,
        height=16,
        image=np.full((16, 16), 64, dtype=np.uint8),
        is_aoi_coord_tile=True,
        aoi_image_x=8,
        aoi_image_y=8,
    )
    anomaly_map = np.ones((8, 8), dtype=np.float32)
    result = ImageResult(
        image_path=Path("W0F00000_000000.tif"),
        image_size=(16, 16),
        otsu_bounds=(0, 0, 16, 16),
        exclusion_regions=[],
        tiles=[tile],
        excluded_tile_count=0,
        processed_tile_count=1,
        processing_time=0.0,
        anomaly_tiles=[(tile, 1.0, anomaly_map)],
    )

    dust_mask = np.full((16, 16), 255, dtype=np.uint8)
    dust_mask_no_ext = np.zeros((16, 16), dtype=np.uint8)
    calls = {"no_ext": False, "two_stage": False, "debug": False}

    def fake_check_dust_or_scratch_feature(image, extension_override=None):
        if extension_override == 0:
            calls["no_ext"] = True
            return True, dust_mask_no_ext, 0.0, "OMIT ext0"
        return True, dust_mask, 0.0, "OMIT"

    def fake_check_dust_per_region(*args, **kwargs):
        assert kwargs["force_include_yx"] == (4, 4)
        assert kwargs["force_include_radius"] == 1
        assert kwargs["force_include_min_score"] == 0.2
        return (
            False,
            None,
            1.0,
            [{"is_dust": True, "coverage": 1.0}],
            np.full((8, 8), 255, dtype=np.uint8),
            np.ones((8, 8), dtype=np.int32),
        )

    def fake_check_dust_two_stage(tile_image, amap, dm, score):
        calls["two_stage"] = True
        assert dm is dust_mask_no_ext
        return (
            True,
            (4, 5),
            [{"abs_pos": (10, 8), "area": 4, "is_dust": False}],
            "TWO_STAGE: 1real+0dust -> REAL_NG",
        )

    def fake_generate_two_stage_debug_image(tile_image, amap, dm, features, is_dust):
        calls["debug"] = True
        assert dm is dust_mask_no_ext
        assert is_dust is False
        return np.zeros((4, 4, 3), dtype=np.uint8)

    inferencer.check_dust_or_scratch_feature = fake_check_dust_or_scratch_feature
    inferencer.check_dust_per_region = fake_check_dust_per_region
    inferencer.check_dust_two_stage = fake_check_dust_two_stage
    inferencer.generate_two_stage_debug_image = fake_generate_two_stage_debug_image

    inferencer._apply_omit_dust_postprocess(
        [result],
        omit_image=np.zeros((16, 16), dtype=np.uint8),
        omit_overexposed=False,
        omit_overexposure_info="",
        cpu_workers=1,
    )

    assert calls == {"no_ext": True, "two_stage": True, "debug": True}
    assert tile.is_suspected_dust_or_scratch is False
    assert "PER_REGION: 0real+1dust -> TWO_STAGE: 1real+0dust -> REAL_NG" in tile.dust_detail_text
    assert tile.anomaly_peak_x == 10
    assert tile.anomaly_peak_y == 8
    assert tile.dust_two_stage_features == [{"abs_pos": (10, 8), "area": 4, "is_dust": False}]
    assert tile.dust_iou_debug_image is not None
