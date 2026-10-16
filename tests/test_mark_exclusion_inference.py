from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_config import CAPIConfig
from capi_database import CAPIDatabase
from capi_edge_cv import EdgeExclusionZoneConfig, EdgeInspectionConfig
from capi_inference import CAPIInferencer, ExclusionRegion, ImageResult, TileInfo
from capi_server import results_to_db_data
from test_mark_geometry import _reported_ay_path, _set_min_area_rect_convention, _synthetic_mark


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


class _InterpolatedMap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_scores = None
        self.fail = False

    def forward(self, scores, size):
        self.input_scores = scores.detach().clone()
        if self.fail:
            raise RuntimeError("heatmap generation failed")
        upsampled = torch.nn.functional.interpolate(scores, size=size).numpy()[0, 0]
        blurred = cv2.GaussianBlur(upsampled, (21, 21), 3)
        return torch.from_numpy(blurred)[None, None]


class _InterpolatedPatchcoreInferencer(_FakePatchcoreInferencer):
    def __init__(self):
        super().__init__()
        self.patchcore.anomaly_map_generator = _InterpolatedMap()

    def predict(self, _image):
        self.patch_scores = torch.tensor([[0.60, 0.40, 0.30, 0.90]])
        locations = torch.zeros((1, 4), dtype=torch.long)
        embedding = torch.tensor([[0.10], [0.05], [0.02], [0.50]])
        raw_score = self.patchcore.compute_anomaly_score(self.patch_scores, locations, embedding)
        score = self.post_processor._normalize(raw_score, 0, 2, 1)
        amap = self.patchcore.anomaly_map_generator(self.patch_scores.reshape(1, 1, 2, 2), (64, 64))
        return SimpleNamespace(pred_score=score, anomaly_map=amap)


@pytest.mark.parametrize("recognition_mode", ["hm", "deferred", "online"])
def test_mark_underlines_are_masked_and_attached_in_every_recognition_mode(
    monkeypatch, capsys, recognition_mode
):
    image = np.zeros((100, 100), dtype=np.uint8)
    detection = {
        "found": True,
        "text": "AY",
        "roi": "top_right",
        "orientation": "normal",
        "deskew_angle_deg": 45.0,
        "bbox": {"x": 60, "y": 20, "width": 20, "height": 60},
        "underline_bboxes": [
            {"x": 60, "y": 85, "width": 20, "height": 5},
            {"x": -2, "y": 98, "width": 5, "height": 8},
            {"x": 100, "y": 20, "width": 5, "height": 2},
            {"x": 20, "y": 20, "width": -5, "height": 2},
            {"x": float("nan"), "y": 20, "width": 5, "height": 2},
            {"x": 20},
            None,
        ],
    }
    if recognition_mode == "hm":
        detection["recognition_skipped"] = True
    monkeypatch.setattr(
        "capi_mark_detector.detect_panel_mark",
        lambda image, include_debug=False: detection,
    )
    monkeypatch.setattr(
        "capi_mark_detector.detect_hm_panel_mark", lambda image, bounds: detection
    )
    inferencer = object.__new__(CAPIInferencer)
    config = CAPIConfig()
    config.patchcore_filter_enabled = False
    config.patchcore_concentration_enabled = False
    config.patchcore_diffuse_area_enabled = False
    config.edge_margin_px = 0
    config.mark_exclusion_padding_px = 0
    config.mark_exclusion_soft_decay_px = 0
    config.no_detect_soft_decay_min_weight = 0.10
    inferencer.config = config
    inferencer.threshold = 0.5
    monkeypatch.setattr(inferencer, "_is_mark_binary_source", lambda filename: True)
    monkeypatch.setattr(inferencer, "_read_detection_image", lambda path: image)
    monkeypatch.setattr(
        inferencer, "_is_hm_mark_layout", lambda filename, machine: recognition_mode == "hm"
    )
    monkeypatch.setattr(
        inferencer, "_find_raw_object_bounds", lambda image: ((0, 0, 100, 100), None)
    )
    recognition_calls = []
    monkeypatch.setattr(
        inferencer,
        "_apply_online_paddle_mark_recognition",
        lambda *args: recognition_calls.append(args),
    )

    found, regions = inferencer._detect_panel_mark_binary_region(
        [Path("W0F00000_000000.tif")],
        apply_recognition=recognition_mode != "deferred",
    )

    assert found["mark_bbox_tuple"] == (60, 20, 20, 60)
    assert [(r.name, r.x1, r.y1, r.x2, r.y2) for r in regions] == [
        ("mark_binary", 60, 20, 80, 80),
        ("mark_underline", 60, 85, 80, 90),
        ("mark_underline", 0, 98, 3, 100),
    ]
    assert len(recognition_calls) == (1 if recognition_mode == "online" else 0)
    log = capsys.readouterr().out
    assert "deskew_angle_deg=45.0" in log
    assert "underline_count=2" in log

    tile = TileInfo(tile_id=1, x=0, y=0, width=100, height=100, image=image)
    result = ImageResult(
        image_path=Path("W0F00000_000000.tif"),
        image_size=(100, 100),
        otsu_bounds=(0, 0, 100, 100),
        exclusion_regions=[],
        tiles=[tile],
        excluded_tile_count=0,
        processed_tile_count=1,
        processing_time=0.0,
    )
    inferencer._attach_panel_mark_binary_to_results([result], found, regions)
    assert result.mark_exclusion_regions == regions
    assert result.mark_bbox == (60, 20, 20, 60)
    assert tile.mark_exclusion_region_count == 3

    anomaly_map = np.zeros((100, 100), dtype=np.float32)
    anomaly_map[86, 70] = 1.0  # MARK underline outside the glyph bounding box.
    anomaly_map[50, 70] = 0.9  # MARK glyph.
    anomaly_map[86, 40] = 0.3  # Real defect beside the MARK.
    score, masked_map = inferencer.predict_tile(
        tile, threshold=0.5, raw_prediction=(1.0, anomaly_map)
    )
    assert masked_map[86, 70] == pytest.approx(0.10)
    assert masked_map[50, 70] == pytest.approx(0.09)
    assert masked_map[86, 40] == pytest.approx(0.30)
    assert score == pytest.approx(0.30)


@pytest.mark.parametrize("convention", ["positive", "negative"])
@pytest.mark.parametrize("rotate180", [False, True])
@pytest.mark.parametrize("recognition_mode", ["deferred", "paddle_ok", "paddle_invalid"])
def test_formal_mark_rectification_excludes_both_underlines_in_aoi_lighting(
    monkeypatch, tmp_path, capsys, convention, rotate180, recognition_mode,
):
    import capi_mark_shadow

    _set_min_area_rect_convention(monkeypatch, convention)
    image, centers = _synthetic_mark(138)
    source = tmp_path / "W0F00000_142426.tif"
    assert cv2.imwrite(str(source), image)
    if rotate180:
        centers = np.asarray([image.shape[1] - 1, image.shape[0] - 1]) - centers

    config = CAPIConfig(
        inference_rotate_180_enabled=rotate180,
        patchcore_filter_enabled=False,
        patchcore_concentration_enabled=False,
        patchcore_diffuse_area_enabled=False,
        edge_margin_px=0,
        mark_exclusion_padding_px=0,
        mark_exclusion_soft_decay_px=0,
        aoi_judgment_roi_enabled=True,
        aoi_judgment_radius_px=25,
    )
    inf = object.__new__(CAPIInferencer)
    inf.config = config
    inf.threshold = 0.34
    payloads = []
    client = object.__new__(capi_mark_shadow.MarkShadowClient)
    client.padding_ratio = 0.15

    def recognize(payload):
        payloads.append(payload)
        return {
            "success": True,
            "paddle_text": "AY" if recognition_mode == "paddle_ok" else "公",
            "paddle_confidence": 0.96,
            "worker_version": "6",
        }

    monkeypatch.setattr(client, "_send", recognize)
    monkeypatch.setattr(capi_mark_shadow, "_CLIENT", client)
    found, regions = inf._detect_panel_mark_binary_region(
        [source], machine_no="CAPI45", model_id="GN116BGEAB40S",
        apply_recognition=recognition_mode != "deferred",
    )
    assert found["text"] == "AY"
    assert 35 < abs(found["deskew_angle_deg"]) < 50
    assert found["orientation"] == ("normal" if rotate180 else "rot180")
    assert len(regions) == 3
    assert [region.name for region in regions].count("mark_underline") == 2
    for x, y in centers:
        assert any(r.x1 <= x < r.x2 and r.y1 <= y < r.y2
                   for r in regions if r.name == "mark_underline")
    assert len(payloads) == (recognition_mode != "deferred")
    for payload in payloads:
        assert payload["crop_rectified"] is True
        assert payload["crop_width"] > payload["crop_height"]
    if recognition_mode != "deferred":
        assert found["recognition_fallback"] is (recognition_mode == "paddle_invalid")
    locator_log = capsys.readouterr().out
    assert "underline_count=2" in locator_log
    assert "crop_rectified=True" in locator_log
    assert f"opencv={cv2.__version__}" in locator_log

    tx, ty = (0, 256) if rotate180 else (512, 0)
    # Check each AOI seed independently in both reported lighting images. A real
    # defect within the same ROI must survive even when both printed lines do not.
    for prefix in ("W0F00000", "WGF50500"):
        for ax, ay in np.rint(centers).astype(int):
            for add_real_defect in (False, True):
                tile = TileInfo(
                    tile_id=0, x=tx, y=ty, width=512, height=512,
                    image=np.zeros((512, 512), dtype=np.uint8),
                    is_aoi_coord_tile=True,
                    aoi_image_x=int(ax), aoi_image_y=int(ay),
                    aoi_product_x=int(ax), aoi_product_y=int(ay),
                    score_threshold=0.34,
                )
                result = ImageResult(
                    image_path=Path(f"{prefix}_142426.tif"),
                    image_size=(1024, 768), otsu_bounds=(0, 0, 1024, 768),
                    raw_bounds=(0, 0, 1024, 768), exclusion_regions=[],
                    tiles=[tile], excluded_tile_count=0, processed_tile_count=1,
                    processing_time=0.0,
                )
                inf._attach_panel_mark_binary_to_results([result], found, regions)
                assert tile.mark_exclusion_region_count == 3
                amap = np.zeros((512, 512), dtype=np.float32)
                for x, y in np.rint(centers).astype(int):
                    amap[y - ty, x - tx] = 0.95
                yy, xx = np.mgrid[:512, :512]
                free = (np.abs(xx + tx - ax) <= 24) & (np.abs(yy + ty - ay) <= 24)
                for r in regions:
                    free &= ~((xx + tx >= r.x1) & (xx + tx < r.x2)
                              & (yy + ty >= r.y1) & (yy + ty < r.y2))
                real_y, real_x = np.argwhere(free)[0]
                if add_real_defect:
                    amap[real_y, real_x] = 0.70
                score, masked = inf.predict_tile(
                    tile, threshold=0.34, raw_prediction=(0.95, amap),
                )
                for x, y in np.rint(centers).astype(int):
                    assert masked[y - ty, x - tx] == pytest.approx(0.095)
                assert masked[real_y, real_x] == pytest.approx(0.70 if add_real_defect else 0)
                result.anomaly_tiles = [(tile, score, masked)]
                inf._apply_aoi_judgment_roi([result], (1024, 768))
                assert bool(tile.is_aoi_coord_below_threshold) is (not add_real_defect)
                assert (result.anomaly_tiles[0][1] >= 0.34) is add_real_defect


@pytest.mark.parametrize("rotate180", [False, True])
def test_reported_ay_formal_exclusions_suppress_both_lines_with_default_padding(rotate180):
    source = _reported_ay_path()
    inf = object.__new__(CAPIInferencer)
    inf.config = CAPIConfig(
        inference_rotate_180_enabled=rotate180,
        patchcore_filter_enabled=False,
        patchcore_concentration_enabled=False,
        patchcore_diffuse_area_enabled=False,
        edge_margin_px=0,
    )
    inf.threshold = 0.34
    found, regions = inf._detect_panel_mark_binary_region([source], apply_recognition=False)
    assert found["text"] == "AY" and found["ocr_quad"] is not None
    assert len(regions) == 3
    width, height = found["image_size"]
    centers = np.float32([[4872, 916], [4919, 963]])
    tx, ty = 4631, 770  # Exact tile origin from the reported inference log.
    if rotate180:
        centers = [width - 1, height - 1] - centers
        tx, ty = width - tx - 512, height - ty - 512
    for prefix in ("W0F00000", "WGF50500"):
        for ax, ay in np.rint(centers).astype(int):
            tile = TileInfo(
                tile_id=0, x=tx, y=ty, width=512, height=512,
                image=np.zeros((512, 512), dtype=np.uint8),
                is_aoi_coord_tile=True, aoi_image_x=int(ax), aoi_image_y=int(ay),
            )
            result = ImageResult(
                image_path=Path(f"{prefix}_142426.tif"), image_size=(width, height),
                otsu_bounds=(0, 0, width, height), exclusion_regions=[], tiles=[tile],
                excluded_tile_count=0, processed_tile_count=1, processing_time=0.0,
            )
            inf._attach_panel_mark_binary_to_results([result], found, regions)
            for real_score in (0.0, 0.7):
                amap = np.zeros((512, 512), dtype=np.float32)
                for x, y in np.rint(centers).astype(int):
                    amap[y - ty, x - tx] = 0.95
                amap[0, 0] = real_score  # Outside all MARK padding/decay bands.
                score, masked = inf.predict_tile(tile, raw_prediction=(0.95, amap))
                for x, y in np.rint(centers).astype(int):
                    assert masked[y - ty, x - tx] == pytest.approx(0.095)
                assert score == pytest.approx(max(0.095, real_score))
                assert masked[0, 0] == pytest.approx(real_score)


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


def _mark_heatmap_case():
    inf = object.__new__(CAPIInferencer)
    inf.config = CAPIConfig(
        patchcore_filter_enabled=False, patchcore_concentration_enabled=False,
        patchcore_diffuse_area_enabled=False, edge_margin_px=0,
        mark_exclusion_padding_px=0, mark_exclusion_soft_decay_px=0,
    )
    inf.threshold = .5
    tile = TileInfo(tile_id=0, x=0, y=0, width=64, height=64,
                    image=np.zeros((64, 64), np.uint8),
                    mark_exclusion_regions=[ExclusionRegion("mark_binary", 32, 32, 64, 64)])
    return inf, tile, _InterpolatedPatchcoreInferencer()


def test_mark_patches_are_excluded_before_heatmap_interpolation_and_blur():
    inf, tile, model = _mark_heatmap_case()
    unfiltered = model.predict(tile.image).anomaly_map.numpy()[0, 0]
    score, filtered = inf.predict_tile(tile, inferencer=model)
    generator = model.patchcore.anomaly_map_generator

    assert tile.mark_patch_heatmap_applied is True
    assert score == pytest.approx(.35)  # Keep valid-patch image-score normalization.
    torch.testing.assert_close(generator.input_scores, torch.tensor([[[[.6, .4], [.3, 0]]]]))
    torch.testing.assert_close(model.patch_scores, torch.tensor([[.6, .4, .3, .9]]))
    assert unfiltered[48, 30] > .4  # MARK leaks into an unexcluded background pixel.
    assert filtered[48, 30] < .3
    assert filtered[16, 16] == pytest.approx(unfiltered[16, 16])  # Real feature remains.
    assert not generator._forward_pre_hooks
    assert "compute_anomaly_score" not in model.patchcore.__dict__
    model.predict(tile.image)
    assert generator.input_scores[0, 0, 1, 1] == pytest.approx(.9)  # No leaked override.


def test_mark_heatmap_override_is_removed_after_prediction_error():
    inf, tile, model = _mark_heatmap_case()
    generator = model.patchcore.anomaly_map_generator
    generator.fail = True
    with pytest.raises(RuntimeError, match="heatmap generation failed"):
        inf.predict_tile(tile, inferencer=model)
    assert not generator._forward_pre_hooks
    assert "compute_anomaly_score" not in model.patchcore.__dict__


def test_mark_soft_decay_band_retains_nearby_feature_heatmap_sources():
    inf, tile, model = _mark_heatmap_case()
    inf.config.mark_exclusion_soft_decay_px = 16
    tile.mark_exclusion_regions = [ExclusionRegion("mark_binary", 28, 32, 64, 64)]
    inf.predict_tile(tile, inferencer=model)
    source = model.patchcore.anomaly_map_generator.input_scores
    assert source[0, 0, 1, 1] == 0  # MARK core.
    assert source[0, 0, 1, 0] == pytest.approx(.3)  # A nearby feature in the soft band.
    assert tile.mark_patch_heatmap_applied


@pytest.mark.parametrize("real_feature", [False, True])
def test_mark_heatmap_spillover_cannot_trigger_aoi_ng_but_nearby_real_patch_can(real_feature):
    class Model(_InterpolatedPatchcoreInferencer):
        def predict(self, _image):
            scores = torch.full((8, 8), .1)
            scores[1:3, 1:3] = .7  # Valid defect outside this AOI keeps full score high.
            scores[4:7, 4:6] = .9  # Known MARK; smoothing spills into the adjacent AOI.
            if real_feature:
                scores[5, 3] = .7  # Independent defect immediately beside the MARK.
            scores = scores.reshape(1, -1)
            embedding = torch.full((64, 1), .5)
            raw_score = self.patchcore.compute_anomaly_score(scores, torch.zeros_like(scores), embedding)
            score = self.post_processor._normalize(raw_score, 0, 2, 1)
            amap = self.patchcore.anomaly_map_generator(scores.reshape(1, 1, 8, 8), (64, 64))
            return SimpleNamespace(pred_score=score, anomaly_map=amap)

    inf, tile, _ = _mark_heatmap_case()
    inf.config.aoi_judgment_roi_enabled = True
    inf.config.aoi_judgment_radius_px = 25
    tile.mark_exclusion_regions = [ExclusionRegion("mark_binary", 32, 32, 48, 56)]
    tile.is_aoi_coord_tile = True
    tile.aoi_image_x, tile.aoi_image_y = 30, 44
    tile.aoi_product_x, tile.aoi_product_y = 240, 352
    tile.score_threshold = .34
    result = ImageResult(image_path=Path("W0F00000_sample.tif"), image_size=(64, 64),
                         otsu_bounds=(0, 0, 64, 64), raw_bounds=(0, 0, 64, 64),
                         exclusion_regions=[], tiles=[tile], excluded_tile_count=0,
                         processed_tile_count=1, processing_time=0.)
    score, amap = inf.predict_tile(tile, inferencer=Model(), threshold=.34)
    result.anomaly_tiles = [(tile, score, amap)]
    inf._apply_aoi_judgment_roi([result], (512, 512))
    assert score == pytest.approx(.6)  # Do not lower the whole-tile defect score.
    assert bool(tile.is_aoi_coord_below_threshold) is (not real_feature)
    assert results_to_db_data([result], {})[0]["is_ng"] == int(real_feature)


def test_no_valid_mark_patch_keeps_original_heatmap_source():
    inf, tile, model = _mark_heatmap_case()
    tile.mark_exclusion_regions = [ExclusionRegion("mark_binary", 0, 0, 64, 64)]
    inf.predict_tile(tile, inferencer=model)
    assert not tile.mark_patch_heatmap_applied
    assert tile.mark_patch_score_reason == "all_patches_excluded"
    assert model.patchcore.anomaly_map_generator.input_scores[0, 0, 1, 1] == pytest.approx(.9)


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
