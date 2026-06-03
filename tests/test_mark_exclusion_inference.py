from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_config import CAPIConfig
from capi_database import CAPIDatabase
from capi_inference import CAPIInferencer, ExclusionRegion, ImageResult, TileInfo
from capi_server import results_to_db_data


def test_mark_exclusion_masks_tile_heatmap_and_recalculates_score():
    config = CAPIConfig()
    config.patchcore_filter_enabled = False
    config.patchcore_concentration_enabled = False
    config.patchcore_diffuse_area_enabled = False
    config.edge_margin_px = 0

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config
    inferencer.threshold = 0.5

    tile = TileInfo(
        tile_id=1,
        x=0,
        y=0,
        width=16,
        height=16,
        image=np.zeros((16, 16), dtype=np.uint8),
        mark_exclusion_regions=[ExclusionRegion("mark_binary", 8, 0, 16, 16)],
    )
    anomaly_map = np.zeros((8, 8), dtype=np.float32)
    anomaly_map[:, 4:] = 1.0
    anomaly_map[4, 2] = 0.25

    score, masked_map = inferencer.predict_tile(
        tile,
        threshold=0.5,
        raw_prediction=(1.0, anomaly_map),
    )

    assert tile.mark_exclusion_masked is True
    assert np.all(masked_map[:, 4:] == 0)
    assert masked_map[4, 2] == pytest.approx(0.25)
    assert score == pytest.approx(0.25)


def test_mark_detection_metadata_is_serialized_to_db_data():
    result = ImageResult(
        image_path=Path("W0F00000_000000.tif"),
        image_size=(1024, 768),
        otsu_bounds=(0, 0, 1024, 768),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        mark_text="EJ",
        mark_confidence=0.87,
        mark_bbox=(80, 520, 120, 90),
        mark_roi="bottom_left",
        mark_orientation="rot180",
        mark_source_image="W0F00000_000000.tif",
    )

    data = results_to_db_data([result], heatmap_info={})

    assert data[0]["mark_text"] == "EJ"
    assert data[0]["mark_confidence"] == pytest.approx(0.87)
    assert data[0]["mark_bbox"] == "80,520,120,90"
    assert data[0]["mark_roi"] == "bottom_left"
    assert data[0]["mark_orientation"] == "rot180"
    assert data[0]["mark_source_image"] == "W0F00000_000000.tif"


def test_mark_detection_metadata_persists_in_image_results(tmp_path):
    db = CAPIDatabase(str(tmp_path / "mark.db"))
    db.save_inference_record(
        glass_id="G1",
        model_id="M1",
        machine_no="1",
        resolution=(1920, 1080),
        machine_judgment="OK",
        ai_judgment="OK",
        image_dir="/fake",
        total_images=1,
        ng_images=0,
        ng_details="[]",
        request_time="2026-06-03T10:00:00",
        response_time="2026-06-03T10:00:01",
        processing_seconds=1.0,
        image_results_data=[{
            "image_path": "/fake/W0F00000_000000.tif",
            "image_name": "W0F00000_000000.tif",
            "image_width": 1024,
            "image_height": 768,
            "otsu_bounds": "0,0,1024,768",
            "tile_count": 0,
            "excluded_tiles": 0,
            "anomaly_count": 0,
            "max_score": 0.0,
            "is_ng": 0,
            "is_dust_only": 0,
            "is_bomb": 0,
            "inference_time_ms": 0.0,
            "heatmap_path": "",
            "mark_text": "EJ",
            "mark_confidence": 0.87,
            "mark_bbox": "80,520,120,90",
            "mark_roi": "bottom_left",
            "mark_orientation": "rot180",
            "mark_source_image": "W0F00000_000000.tif",
            "tiles": [],
        }],
    )

    detail = db.get_record_detail(1)

    assert detail["images"][0]["mark_text"] == "EJ"
    assert detail["images"][0]["mark_bbox"] == "80,520,120,90"
    assert detail["images"][0]["mark_orientation"] == "rot180"
