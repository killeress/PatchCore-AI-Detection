"""
tests/test_process_panel_v2.py
Smoke tests for _process_panel_v2 — uses MagicMock so no real GPU / model files needed.
"""
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from capi_config import CAPIConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_dir: Path) -> CAPIConfig:
    """Build a minimal new-architecture CAPIConfig with nested model_mapping."""
    inner_pt = tmp_dir / "g_inner.pt"
    edge_pt = tmp_dir / "g_edge.pt"
    # Files don't have to exist — _get_model_for is mocked in tests
    return CAPIConfig(
        machine_id="TEST_MACHINE",
        is_new_architecture=True,
        edge_threshold_px=768,
        tile_size=512,
        otsu_offset=5,
        enable_panel_polygon=True,
        model_mapping={
            "G0F00000": {
                "inner": str(inner_pt),
                "edge": str(edge_pt),
            }
        },
        threshold_mapping={
            "G0F00000": {"inner": 0.5, "edge": 0.5},
        },
    )


def _write_grey_panel_image(folder: Path, prefix: str = "G0F00000") -> Path:
    """Write a small grayscale PNG that Otsu can binarize sensibly."""
    import cv2
    h, w = 1024, 1024
    img = np.zeros((h, w), dtype=np.uint8)
    # bright panel region in the centre
    img[100:900, 100:900] = 200
    path = folder / f"{prefix}_test.png"
    cv2.imwrite(str(path), img)
    return path


def _make_fake_predict_result(score: float = 0.3) -> Any:
    result = MagicMock()
    result.pred_score = score
    amap = np.zeros((512, 512), dtype=np.float32)
    amap[256, 256] = score
    result.anomaly_map = amap
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_process_panel_v2_returns_compatible_tuple(tmp_path):
    """v2 must return a 7-tuple compatible with the v1 return signature."""
    _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)

    from capi_inference import CAPIInferencer

    fake_model = MagicMock()
    fake_model.predict.return_value = _make_fake_predict_result(0.3)

    with patch.object(CAPIInferencer, "_get_model_for", return_value=fake_model):
        inferencer = CAPIInferencer(cfg)
        ret = inferencer.process_panel(tmp_path)

    assert isinstance(ret, tuple), "process_panel should return a tuple"
    assert len(ret) == 7, "should return 7-tuple (results, omit_vis, omit_oe, omit_info, is_dup, omit_img, aoi_report)"

    results, omit_vis, omit_oe, omit_info, is_dup, omit_img, aoi_report = ret
    assert isinstance(results, list), "results should be List[ImageResult]"


def test_process_panel_v2_no_anomaly_when_score_below_threshold(tmp_path):
    """All tiles below threshold → no anomaly_tiles in results."""
    _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)

    from capi_inference import CAPIInferencer

    fake_model = MagicMock()
    # score = 0.1 < threshold 0.5
    fake_model.predict.return_value = _make_fake_predict_result(0.1)

    with patch.object(CAPIInferencer, "_get_model_for", return_value=fake_model):
        inferencer = CAPIInferencer(cfg)
        results, *_ = inferencer.process_panel(tmp_path)

    for r in results:
        assert r.anomaly_tiles == [], f"Expected no anomaly tiles, got {len(r.anomaly_tiles)}"


def test_process_panel_v2_detects_anomaly_when_score_above_threshold(tmp_path):
    """Tiles above threshold appear in anomaly_tiles."""
    _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)

    from capi_inference import CAPIInferencer

    fake_model = MagicMock()
    # score = 0.9 > threshold 0.5
    fake_model.predict.return_value = _make_fake_predict_result(0.9)

    with patch.object(CAPIInferencer, "_get_model_for", return_value=fake_model):
        inferencer = CAPIInferencer(cfg)
        results, *_ = inferencer.process_panel(tmp_path)

    assert results, "Expected at least one ImageResult"
    ng_results = [r for r in results if r.anomaly_tiles]
    assert ng_results, "Expected at least one NG image result when all scores are 0.9"


def test_process_panel_v2_empty_folder(tmp_path):
    """Empty folder → empty results (no crash)."""
    cfg = _make_config(tmp_path)
    from capi_inference import CAPIInferencer

    with patch.object(CAPIInferencer, "_get_model_for", return_value=MagicMock()):
        inferencer = CAPIInferencer(cfg)
        results, omit_vis, omit_oe, omit_info, is_dup, omit_img, aoi_report = inferencer.process_panel(tmp_path)

    assert results == []


def test_process_panel_v2_image_result_fields(tmp_path):
    """ImageResult returned by v2 has expected attributes."""
    _write_grey_panel_image(tmp_path, "G0F00000")
    cfg = _make_config(tmp_path)

    from capi_inference import CAPIInferencer, ImageResult

    fake_model = MagicMock()
    fake_model.predict.return_value = _make_fake_predict_result(0.3)

    with patch.object(CAPIInferencer, "_get_model_for", return_value=fake_model):
        inferencer = CAPIInferencer(cfg)
        results, *_ = inferencer.process_panel(tmp_path)

    for r in results:
        assert isinstance(r, ImageResult)
        assert r.image_path.exists()
        assert r.image_size[0] > 0 and r.image_size[1] > 0
        assert r.otsu_bounds is not None
        assert r.tiles is not None


def test_process_panel_v2_missing_lighting_skipped(tmp_path):
    """If lighting image has no matching model_mapping entry, it is skipped gracefully."""
    # Write an image with a prefix NOT in model_mapping
    _write_grey_panel_image(tmp_path, "R0F00000")
    cfg = _make_config(tmp_path)  # only G0F00000 in model_mapping

    from capi_inference import CAPIInferencer

    with patch.object(CAPIInferencer, "_get_model_for", return_value=MagicMock()):
        inferencer = CAPIInferencer(cfg)
        results, *_ = inferencer.process_panel(tmp_path)

    # R0F00000 image should be skipped → no results
    assert results == [], "R0F00000 not in model_mapping → should be skipped"


def test_predict_tile_applies_mask(tmp_path):
    """_predict_tile should zero out score where mask=0."""
    cfg = _make_config(tmp_path)
    from capi_inference import CAPIInferencer

    inferencer = CAPIInferencer(cfg)

    # Fake model returns anomaly_map with value 1.0 everywhere
    fake_result = MagicMock()
    fake_result.pred_score = 1.0
    amap = np.ones((512, 512), dtype=np.float32)
    fake_result.anomaly_map = amap
    fake_model = MagicMock()
    fake_model.predict.return_value = fake_result

    # mask with only top-left quadrant as panel interior
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[:256, :256] = 255

    tile_img = np.zeros((512, 512, 3), dtype=np.uint8)
    score, out_map = inferencer._predict_tile(fake_model, tile_img, mask)

    assert out_map is not None
    # Pixels outside mask should be 0
    assert out_map[300, 300] == 0.0
    # Pixels inside mask should be 1.0
    assert out_map[100, 100] == 1.0
    # score should equal max of masked map = 1.0
    assert score == pytest.approx(1.0)
