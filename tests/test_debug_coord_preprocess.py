import io
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


class _FakeInferencer:
    def __init__(self, pipeline, after_tiling):
        self.config = SimpleNamespace(
            tile_size=8,
            otsu_offset=0,
            enable_panel_polygon=False,
            edge_threshold_px=8,
            image_preprocess_pipeline=pipeline,
            preprocess_after_tiling=after_tiling,
            edge_margin_px=0,
            should_skip_file=lambda _name: False,
        )
        self._model_mapping = {}
        self.last_tile = None

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

    def predict_tile(self, tile, inferencer=None, edge_margin_override=None):
        self.last_tile = tile
        return 0.1, None


@pytest.mark.parametrize("after_tiling", [False, True])
def test_coord_inference_applies_configured_preprocess(tmp_path, after_tiling):
    from capi_web import CAPIWebHandler
    from capi_image_preprocess_lab import apply_preprocess_pipeline

    pipeline = [{"method": "gaussian", "params": {"kernel_size": 3, "sigma": 1.0}}]
    fake_inferencer = _FakeInferencer(pipeline, after_tiling)
    image = np.tile(np.arange(16, dtype=np.uint8), (16, 1))
    image_path = tmp_path / "W0F00000_sample.tif"
    assert cv2.imwrite(str(image_path), image)

    body = json.dumps({
        "image_path": str(image_path),
        "product_x": 8,
        "product_y": 8,
        "product_w": 16,
        "product_h": 16,
        "threshold": 0.5,
        "edge_margin_px": 0,
    }).encode("utf-8")

    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.inferencer = fake_inferencer
    handler.heatmap_manager = None
    handler.rfile = io.BytesIO(body)
    handler.headers = SimpleNamespace(
        get=lambda key, default=0: str(len(body)) if key == "Content-Length" else default
    )
    sent = []
    handler._send_json = lambda payload, status=200: sent.append((payload, status))
    CAPIWebHandler._debug_heatmap_dir = tmp_path / "debug"

    handler._handle_debug_coord_inference()

    assert sent and sent[0][1] == 200
    response = sent[0][0]
    assert response["success"] is True
    assert response["preprocess"]["applied"] is True
    assert response["preprocess"]["after_tiling"] is after_tiling
    assert response["preprocess"]["steps"][0]["method"] == "gaussian"
    assert response["preprocess"]["total_time_ms"] > 0

    raw_crop = image[4:12, 4:12]
    expected = (
        apply_preprocess_pipeline(raw_crop, pipeline)["image"]
        if after_tiling
        else apply_preprocess_pipeline(image, pipeline)["image"][4:12, 4:12]
    )
    assert np.array_equal(fake_inferencer.last_tile.image, expected)
