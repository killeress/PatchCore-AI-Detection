from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_config import CAPIConfig
from capi_database import CAPIDatabase
from capi_edge_cv import EdgeExclusionZoneConfig, EdgeInspectionConfig
from capi_inference import CAPIInferencer, ExclusionRegion, ImageResult, TileInfo
from capi_server import results_to_db_data


class _FakePatchcoreModel:
    def compute_anomaly_score(self, patch_scores, _locations, embedding):
        batch_size, patch_count = patch_scores.shape
        selected = torch.argmax(patch_scores, dim=1)
        features = embedding.reshape(batch_size, patch_count, -1)
        rows = torch.arange(batch_size, device=patch_scores.device)
        return patch_scores[rows, selected] + features[rows, selected, 0]


class _FakePostProcessor:
    enable_normalization = True
    image_min = torch.tensor(0.0)
    image_max = torch.tensor(2.0)
    image_threshold = torch.tensor(1.0)

    @staticmethod
    def _normalize(preds, norm_min, norm_max, threshold):
        return (((preds - threshold) / (norm_max - norm_min)) + 0.5).clamp(0, 1)


class _FakePatchcoreInferencer:
    def __init__(self):
        self.patchcore = _FakePatchcoreModel()
        self.post_processor = _FakePostProcessor()
        self.model = SimpleNamespace(
            model=self.patchcore,
            post_processor=self.post_processor,
        )

    def predict(self, _image):
        # Bottom-right is MARK (0.90); top-left is the real point defect (0.60).
        patch_scores = torch.tensor([[0.60, 0.40, 0.30, 0.90]])
        locations = torch.zeros((1, 4), dtype=torch.long)
        embedding = torch.tensor([[0.10], [0.05], [0.02], [0.50]])
        raw_score = self.patchcore.compute_anomaly_score(
            patch_scores, locations, embedding
        )
        pred_score = self.post_processor._normalize(
            raw_score,
            self.post_processor.image_min,
            self.post_processor.image_max,
            self.post_processor.image_threshold,
        )
        anomaly_map = torch.zeros((1, 64, 64), dtype=torch.float32)
        anomaly_map[:, :32, :32] = 0.60
        anomaly_map[:, 32:, 32:] = 0.90
        return SimpleNamespace(pred_score=pred_score, anomaly_map=anomaly_map)


def test_mark_exclusion_recomputes_formal_score_from_strongest_valid_patch():
    config = CAPIConfig()
    config.patchcore_filter_enabled = False
    config.patchcore_concentration_enabled = False
    config.patchcore_diffuse_area_enabled = False
    config.edge_margin_px = 0
    config.mark_exclusion_padding_px = 0
    config.mark_exclusion_soft_decay_px = 0
    config.no_detect_soft_decay_min_weight = 0.10

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config
    inferencer.threshold = 0.5
    fake_model = _FakePatchcoreInferencer()
    tile = TileInfo(
        tile_id=1,
        x=0,
        y=0,
        width=64,
        height=64,
        image=np.zeros((64, 64), dtype=np.uint8),
        mark_exclusion_regions=[ExclusionRegion("mark_binary", 32, 32, 64, 64)],
    )

    score, masked_map = inferencer.predict_tile(
        tile, inferencer=fake_model, threshold=0.5
    )

    # Original: (0.90 + 0.50) normalized = 0.70.
    # MARK-free: (0.60 + 0.10) normalized = 0.35.
    assert tile.raw_pred_score == pytest.approx(0.70)
    assert score == pytest.approx(0.35)
    assert tile.mark_patch_score_applied is True
    assert tile.mark_patchcore_score == pytest.approx(0.35)
    assert tile.mark_patch_valid_count == 3
    assert tile.mark_patch_total_count == 4
    assert (tile.mark_patch_peak_x, tile.mark_patch_peak_y) == (16, 16)
    assert tile.mark_patch_score_reason == "applied"
    assert masked_map[48, 48] == pytest.approx(0.09)
    assert masked_map[16, 16] == pytest.approx(0.60)
    assert "compute_anomaly_score" not in fake_model.patchcore.__dict__


def test_mark_exclusion_downweights_tile_heatmap_and_recalculates_score():
    config = CAPIConfig()
    config.patchcore_filter_enabled = False
    config.patchcore_concentration_enabled = False
    config.patchcore_diffuse_area_enabled = False
    config.edge_margin_px = 0
    config.mark_exclusion_padding_px = 0
    config.mark_exclusion_soft_decay_px = 0
    config.no_detect_soft_decay_min_weight = 0.10

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
    assert np.allclose(masked_map[:, 4:], 0.10)
    assert masked_map[4, 2] == pytest.approx(0.25)
    assert score == pytest.approx(0.25)


def test_mark_exclusion_padding_masks_heatmap_bleed_outside_bbox():
    config = CAPIConfig()
    config.patchcore_filter_enabled = False
    config.patchcore_concentration_enabled = False
    config.patchcore_diffuse_area_enabled = False
    config.edge_margin_px = 0
    config.mark_exclusion_padding_px = 10
    config.mark_exclusion_soft_decay_px = 0
    config.no_detect_soft_decay_min_weight = 0.10

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config
    inferencer.threshold = 0.5

    tile = TileInfo(
        tile_id=1,
        x=0,
        y=0,
        width=100,
        height=100,
        image=np.zeros((100, 100), dtype=np.uint8),
        mark_exclusion_regions=[ExclusionRegion("mark_binary", 60, 20, 80, 80)],
    )
    anomaly_map = np.zeros((10, 10), dtype=np.float32)
    anomaly_map[5, 6] = 1.0   # inside MARK bbox
    anomaly_map[5, 5] = 0.9   # heatmap bleed just outside bbox, inside padding
    anomaly_map[5, 3] = 0.33  # outside padded MARK area

    score, masked_map = inferencer.predict_tile(
        tile,
        threshold=0.5,
        raw_prediction=(1.0, anomaly_map),
    )

    assert tile.mark_exclusion_masked is True
    assert masked_map[5, 6] == pytest.approx(0.10)
    assert masked_map[5, 5] == pytest.approx(0.09)
    assert masked_map[5, 3] == pytest.approx(0.33)
    assert score == pytest.approx(0.33)


def test_mark_exclusion_soft_decay_reduces_heatmap_bleed_outside_padding():
    config = CAPIConfig()
    config.patchcore_filter_enabled = False
    config.patchcore_concentration_enabled = False
    config.patchcore_diffuse_area_enabled = False
    config.edge_margin_px = 0
    config.mark_exclusion_padding_px = 0
    config.mark_exclusion_soft_decay_px = 20
    config.no_detect_soft_decay_min_weight = 0.10

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config
    inferencer.threshold = 0.5

    tile = TileInfo(
        tile_id=1,
        x=0,
        y=0,
        width=100,
        height=100,
        image=np.zeros((100, 100), dtype=np.uint8),
        mark_exclusion_regions=[ExclusionRegion("mark_binary", 60, 20, 80, 80)],
    )
    anomaly_map = np.zeros((10, 10), dtype=np.float32)
    anomaly_map[5, 6] = 1.0   # inside MARK bbox
    anomaly_map[5, 5] = 1.0   # 5 px outside bbox, inside soft decay band
    anomaly_map[5, 3] = 0.2   # outside soft decay band

    score, masked_map = inferencer.predict_tile(
        tile,
        threshold=0.5,
        raw_prediction=(1.0, anomaly_map),
    )

    assert tile.mark_exclusion_masked is True
    assert masked_map[5, 6] == pytest.approx(0.10)
    assert masked_map[5, 5] == pytest.approx(0.325, abs=0.001)
    assert masked_map[5, 3] == pytest.approx(0.2)
    assert score == pytest.approx(0.325, abs=0.001)


def test_configured_exclude_zone_soft_decay_recalculates_main_score():
    config = CAPIConfig()
    config.patchcore_filter_enabled = False
    config.patchcore_concentration_enabled = False
    config.patchcore_diffuse_area_enabled = False
    config.edge_margin_px = 0
    config.cv_edge_exclude_soft_decay_px = 20
    config.no_detect_soft_decay_min_weight = 0.10

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config
    inferencer.threshold = 0.5
    inferencer.edge_inspector = SimpleNamespace(
        config=EdgeInspectionConfig(
            exclude_zones=[
                EdgeExclusionZoneConfig(enabled=True, x=60, y=20, w=20, h=60)
            ]
        )
    )

    tile = TileInfo(
        tile_id=1,
        x=0,
        y=0,
        width=100,
        height=100,
        image=np.zeros((100, 100), dtype=np.uint8),
    )
    anomaly_map = np.zeros((10, 10), dtype=np.float32)
    anomaly_map[5, 6] = 1.0
    anomaly_map[5, 5] = 1.0
    anomaly_map[5, 3] = 0.2

    score, masked_map = inferencer.predict_tile(
        tile,
        threshold=0.5,
        raw_prediction=(1.0, anomaly_map),
    )

    assert masked_map[5, 6] == 0
    assert masked_map[5, 5] == pytest.approx(0.325, abs=0.001)
    assert masked_map[5, 3] == pytest.approx(0.2)
    assert score == pytest.approx(0.325, abs=0.001)


def test_configured_exclude_zone_soft_decay_preserves_aoi_seed_outside_zone():
    config = CAPIConfig()
    config.patchcore_filter_enabled = False
    config.patchcore_concentration_enabled = False
    config.patchcore_diffuse_area_enabled = False
    config.edge_margin_px = 0
    config.cv_edge_exclude_soft_decay_px = 20
    config.no_detect_soft_decay_min_weight = 0.10

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config
    inferencer.threshold = 0.5
    inferencer.edge_inspector = SimpleNamespace(
        config=EdgeInspectionConfig(
            exclude_zones=[
                EdgeExclusionZoneConfig(enabled=True, x=60, y=20, w=20, h=60)
            ]
        )
    )

    tile = TileInfo(
        tile_id=1,
        x=0,
        y=0,
        width=100,
        height=100,
        image=np.zeros((100, 100), dtype=np.uint8),
        is_aoi_coord_tile=True,
        aoi_image_x=55,
        aoi_image_y=55,
    )
    anomaly_map = np.zeros((10, 10), dtype=np.float32)
    anomaly_map[5, 6] = 1.0   # inside exclude zone
    anomaly_map[5, 5] = 1.0   # AOI seed just outside zone

    score, masked_map = inferencer.predict_tile(
        tile,
        threshold=0.5,
        raw_prediction=(1.0, anomaly_map),
    )

    assert masked_map[5, 6] == 0
    assert masked_map[5, 5] == pytest.approx(1.0)
    assert score == pytest.approx(1.0)


def test_mark_exclusion_padding_is_not_restored_by_aoi_seed():
    config = CAPIConfig()
    config.patchcore_filter_enabled = False
    config.patchcore_concentration_enabled = False
    config.patchcore_diffuse_area_enabled = False
    config.edge_margin_px = 0
    config.mark_exclusion_padding_px = 10
    config.mark_exclusion_soft_decay_px = 20
    config.no_detect_soft_decay_min_weight = 0.10

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = config
    inferencer.threshold = 0.5

    tile = TileInfo(
        tile_id=1,
        x=0,
        y=0,
        width=100,
        height=100,
        image=np.zeros((100, 100), dtype=np.uint8),
        mark_exclusion_regions=[ExclusionRegion("mark_binary", 60, 20, 80, 80)],
        is_aoi_coord_tile=True,
        aoi_image_x=55,
        aoi_image_y=55,
    )
    anomaly_map = np.zeros((10, 10), dtype=np.float32)
    anomaly_map[5, 5] = 1.0   # inside MARK hard padding
    anomaly_map[5, 2] = 0.2   # outside padding and soft decay band

    score, masked_map = inferencer.predict_tile(
        tile,
        threshold=0.5,
        raw_prediction=(1.0, anomaly_map),
    )

    assert masked_map[5, 5] == pytest.approx(0.10)
    assert masked_map[5, 2] == pytest.approx(0.2)
    assert score == pytest.approx(0.2)


def test_mark_exclusion_padding_roundtrips_config():
    cfg = CAPIConfig()
    cfg.mark_exclusion_padding_px = 48
    cfg.mark_exclusion_soft_decay_px = 40
    cfg.cv_edge_exclude_soft_decay_px = 72
    cfg.no_detect_soft_decay_min_weight = 0.2

    reloaded = CAPIConfig.from_dict(cfg.to_dict())

    assert reloaded.mark_exclusion_padding_px == 48
    assert reloaded.mark_exclusion_soft_decay_px == 40
    assert reloaded.cv_edge_exclude_soft_decay_px == 72
    assert reloaded.no_detect_soft_decay_min_weight == pytest.approx(0.2)


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
