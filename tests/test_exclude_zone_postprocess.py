from pathlib import Path
from types import SimpleNamespace

import numpy as np

from capi_edge_cv import EdgeExclusionZoneConfig, EdgeInspectionConfig
from capi_inference import CAPIInferencer, ImageResult, TileInfo


def _make_inferencer_with_zone():
    inferencer = object.__new__(CAPIInferencer)
    edge_config = EdgeInspectionConfig(
        exclude_zones=[
            EdgeExclusionZoneConfig(enabled=True, x=100, y=100, w=50, h=50)
        ]
    )
    inferencer.edge_inspector = SimpleNamespace(config=edge_config)
    return inferencer


def _make_result(tile):
    return ImageResult(
        image_path=Path("W0F00000_000000.tif"),
        image_size=(256, 256),
        otsu_bounds=(0, 0, 256, 256),
        exclusion_regions=[],
        tiles=[tile],
        excluded_tile_count=0,
        processed_tile_count=1,
        processing_time=0.0,
        anomaly_tiles=[(tile, 1.0, None)],
    )


def test_exclude_zone_keeps_aoi_tile_when_only_peak_is_inside_zone(capsys):
    inferencer = _make_inferencer_with_zone()
    tile = TileInfo(
        tile_id=1,
        x=80,
        y=80,
        width=128,
        height=128,
        image=np.zeros((128, 128), dtype=np.uint8),
        is_aoi_coord_tile=True,
        aoi_image_x=120,
        aoi_image_y=90,
        zone="edge",
    )
    tile.anomaly_peak_x = 120
    tile.anomaly_peak_y = 120

    inferencer._apply_exclude_zone_postprocess([_make_result(tile)], model_id="GN156H")

    assert tile.is_in_exclude_zone is False
    assert "保留 NG" in capsys.readouterr().out


def test_exclude_zone_suppresses_aoi_tile_when_aoi_point_is_inside_zone():
    inferencer = _make_inferencer_with_zone()
    tile = TileInfo(
        tile_id=1,
        x=80,
        y=80,
        width=128,
        height=128,
        image=np.zeros((128, 128), dtype=np.uint8),
        is_aoi_coord_tile=True,
        aoi_image_x=120,
        aoi_image_y=120,
        zone="edge",
    )
    tile.anomaly_peak_x = 120
    tile.anomaly_peak_y = 90

    inferencer._apply_exclude_zone_postprocess([_make_result(tile)], model_id="GN156H")

    assert tile.is_in_exclude_zone is True


def test_exclude_zone_still_suppresses_non_aoi_tile_by_peak():
    inferencer = _make_inferencer_with_zone()
    tile = TileInfo(
        tile_id=1,
        x=80,
        y=80,
        width=128,
        height=128,
        image=np.zeros((128, 128), dtype=np.uint8),
    )
    tile.anomaly_peak_x = 120
    tile.anomaly_peak_y = 120

    inferencer._apply_exclude_zone_postprocess([_make_result(tile)], model_id="GN156H")

    assert tile.is_in_exclude_zone is True
