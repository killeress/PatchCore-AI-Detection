import io
import json
from types import SimpleNamespace

import cv2
import numpy as np

from capi_inference import CAPIInferencer


class _DiagnosticInferencer:
    check_dust_per_region = CAPIInferencer.check_dust_per_region
    _aoi_center_seed_for_tile = CAPIInferencer._aoi_center_seed_for_tile

    def __init__(self):
        self.config = SimpleNamespace(
            machine_id="TEST_MODEL",
            tile_size=64,
            otsu_offset=0,
            enable_panel_polygon=False,
            edge_threshold_px=8,
            image_preprocess_pipeline=[],
            image_preprocess_pipelines={},
            preprocess_after_tiling=False,
            edge_margin_px=0,
            should_skip_file=lambda _name: False,
            dust_heatmap_top_percent=0.5,
            dust_heatmap_metric="coverage",
            dust_heatmap_iou_threshold=0.15,
            dust_high_cov_threshold=0.5,
            dust_peak_fraction_threshold=0.8,
            dust_mask_before_binarize=False,
            dust_two_stage_enabled=True,
            aoi_heatmap_center_seed_enabled=False,
        )
        self._model_mapping = {}

    def _find_raw_object_bounds(self, image):
        height, width = image.shape[:2]
        return (0, 0, width, height), None

    def calculate_otsu_bounds(self, image):
        height, width = image.shape[:2]
        return (0, 0, width, height), None, None

    def _get_image_prefix(self, _name):
        return "W0F00000"

    def _get_inferencer_for_prefix(self, _prefix):
        return object()

    def predict_tile(
        self,
        tile,
        inferencer=None,
        edge_margin_override=None,
        threshold=None,
        model_id=None,
    ):
        anomaly = np.full((64, 64), 0.01, dtype=np.float32)
        anomaly[50:58, 50:58] = 1.0  # 強氣泡：吃掉 Top 0.5% 配額
        anomaly[30:35, 30:35] = 0.6  # AOI 中心真缺陷候選
        tile.raw_pred_score = 1.0
        tile.pre_decay_map_max = 1.0
        tile.post_decay_map_max = 1.0
        tile.score_decay_ratio = 1.0
        tile.score_mask_valid_ratio = 1.0
        tile.score_edge_margin_sides = ""
        return 1.0, anomaly

    def check_omit_overexposure(self, _omit_image):
        return False, 80.0, 0.01, "正常"

    def _check_dust_or_scratch_feature_with_context(
        self,
        omit_image,
        tile_x,
        tile_y,
        tile_width,
        tile_height,
        omit_crop,
        extension_override=None,
        context_shift=96,
        focus_x=None,
        product_resolution=None,
    ):
        dust_mask = np.zeros((tile_height, tile_width), dtype=np.uint8)
        dust_mask[47:61, 47:61] = 255
        return True, dust_mask, 0.01, "Bub:1"

    def _mask_aoi_exclude_zones_for_dust(
        self, tile_info, anomaly_map, model_id=None
    ):
        return anomaly_map, False

    def check_dust_two_stage(
        self,
        tile_image,
        anomaly_map,
        dust_mask,
        score,
        score_threshold=None,
        candidate_dust_mask=None,
    ):
        assert score_threshold == 0.35
        return (
            False,
            None,
            [],
            "TWO_STAGE: 0real+0dust ignored_outside_hot_core=1 -> DUST",
        )

    def generate_two_stage_debug_image(
        self, tile_image, anomaly_map, dust_mask, features, is_dust
    ):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def generate_dust_iou_debug_image(self, *args, **kwargs):
        return np.zeros((64, 64, 3), dtype=np.uint8)


def test_coord_debug_exposes_bubble_score_competition_in_chinese(tmp_path):
    from capi_web import CAPIWebHandler

    image = np.full((128, 128), 90, dtype=np.uint8)
    image_path = tmp_path / "W0F00000_sample.tif"
    omit_path = tmp_path / "PINIGBI_sample.tif"
    assert cv2.imwrite(str(image_path), image)
    assert cv2.imwrite(str(omit_path), image)

    body = json.dumps({
        "image_path": str(image_path),
        "product_x": 64,
        "product_y": 64,
        "product_w": 128,
        "product_h": 128,
        "threshold": 0.35,
        "edge_margin_px": 0,
        "peak_window_px": 16,
    }).encode("utf-8")

    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.inferencer = _DiagnosticInferencer()
    handler.heatmap_manager = None
    handler.rfile = io.BytesIO(body)
    handler.headers = SimpleNamespace(
        get=lambda key, default=0: (
            str(len(body)) if key == "Content-Length" else default
        )
    )
    sent = []
    handler._send_json = lambda payload, status=200: sent.append((payload, status))
    CAPIWebHandler._debug_heatmap_dir = tmp_path / "debug"

    handler._handle_debug_coord_inference()

    assert sent and sent[0][1] == 200
    response = sent[0][0]
    assert response["judgment"] == "NG"
    assert response["final_judgment"] == "OK"
    assert response["dust_analysis"]["center_seed"]["enabled"] is False
    assert response["dust_analysis"]["regions"]
    assert response["dust_analysis"]["regions"][0]["is_dust"] is True
    assert (
        response["dust_analysis"]["two_stage"]["counters"]
        ["ignored_outside_hot_core"]["count"]
        == 1
    )

    analysis = response["heatmap_analysis"]
    assert analysis["available"] is True
    assert analysis["score_competition_detected"] is True
    assert analysis["dominant_peak"]["in_dust"] is True
    assert analysis["aoi_best_peak"]["in_top_percent"] is False
    assert analysis["aoi_best_peak"]["estimated_score"] == 0.6
    assert "搶走高分" in analysis["conclusion_zh"]
    assert response["peak_diagnostic_url"]


def test_coord_debug_template_has_operator_facing_heatmap_tables():
    template = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "templates"
        / "debug_inference.html"
    ).read_text(encoding="utf-8")

    assert 'id="coord-diagnostic-section"' in template
    assert "正式 Top % 熱區（全部列出）" in template
    assert "局部峰值掃描（不受 Top % 配額限制，全部列出）" in template
    assert "被 Top % 排除" in template
    assert "僅供排查，不修改正式設定或判定" in template
