import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_heatmap import HeatmapManager


def _capture_text(monkeypatch):
    captured = []
    real_put_text = cv2.putText

    def capture(image, text, *args, **kwargs):
        captured.append(str(text))
        return real_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr("capi_heatmap.cv2.putText", capture)
    return captured


@pytest.mark.parametrize(
    "is_bomb,is_dust,score,expected",
    [
        (True, True, 1.0, "BOMB: X1 (Filtered as OK)"),
        (True, False, 0.0, "BOMB: X1 (Filtered as OK)"),
        (False, True, 1.0, "DUST (Filtered as OK)"),
        (False, True, 0.0, "DUST (Filtered as OK)"),
        (False, False, 0.1, "NG"),
        (False, False, 0.0, "OK"),
    ],
)
def test_black_header_uses_final_filter_flags(tmp_path, monkeypatch, is_bomb, is_dust, score, expected):
    captured = _capture_text(monkeypatch)
    tile = SimpleNamespace(
        is_bright_spot_detection=True,
        is_bomb=is_bomb,
        bomb_defect_code="X1",
        is_suspected_dust_or_scratch=is_dust,
    )
    output = HeatmapManager(tmp_path).save_tile_heatmap(
        tmp_path, "B0F00000", 1,
        np.zeros((16, 16), dtype=np.uint8),
        np.zeros((16, 16), dtype=np.float32),
        score, tile_info=tile, score_threshold=0.9,
    )

    verdicts = [text for text in captured if text in ("NG", "OK") or "Filtered as OK" in text]
    assert verdicts == [expected]
    assert cv2.imread(output).shape == (612, 1536, 3)


def test_black_dust_evidence_shows_filtered_candidates_and_remaining_ng(tmp_path, monkeypatch):
    captured = _capture_text(monkeypatch)
    original = np.full((16, 16), 20, dtype=np.uint8)
    omit = np.full((16, 16), 100, dtype=np.uint8)
    dust_mask = np.zeros((16, 16), dtype=np.uint8)
    dust_mask[4:6, 4:6] = 255
    source_binary = dust_mask.copy()
    source_binary[10:12, 10:12] = 255
    remaining = np.zeros((16, 16), dtype=np.float32)
    remaining[10:12, 10:12] = 1.0
    debug = np.full((16, 16, 3), (100, 70, 50), dtype=np.uint8)
    tile = SimpleNamespace(
        is_bright_spot_detection=True,
        is_suspected_dust_or_scratch=False,
        omit_crop_image=omit,
        dust_mask=dust_mask,
        dust_heatmap_binary=source_binary,
        dust_iou_debug_image=debug,
        dust_detail_text="BLACK_DUST: 1 real + 1 dust",
    )

    output = HeatmapManager(tmp_path).save_tile_heatmap(
        tmp_path, "B0F00000", 2, original, remaining, 1.0, tile_info=tile,
    )
    rendered = cv2.imread(output)

    assert rendered.shape == (612, 3072, 3)
    assert "NG" in captured
    assert "OMIT Crop" in captured
    assert "Dust Mask" in captured
    assert "Dust Debug (G=Dust R=RealNG B=DustOnly)" in captured
    assert any("BLACK_DUST: 1 real + 1 dust" in text for text in captured)
    np.testing.assert_array_equal(rendered[60 + 150, 1024 + 150], (10, 110, 10))
    np.testing.assert_array_equal(rendered[60 + 340, 1024 + 340], (10, 10, 138))
    np.testing.assert_array_equal(rendered[60 + 150, 1536 + 150], (100, 100, 100))
    np.testing.assert_array_equal(rendered[60 + 150, 2048 + 150], (60, 162, 162))
    np.testing.assert_array_equal(rendered[60 + 20, 2048 + 20], (100, 100, 100))
    np.testing.assert_array_equal(rendered[60 + 150, 2560 + 150], (100, 70, 50))
    assert source_binary[4, 4] == 255
    assert remaining[4, 4] == 0.0
    assert np.all(omit == 100)


def test_black_dust_evidence_allows_missing_omit_and_debug(tmp_path, monkeypatch):
    captured = _capture_text(monkeypatch)
    tile = SimpleNamespace(
        is_bright_spot_detection=True,
        is_suspected_dust_or_scratch=True,
        dust_mask=np.zeros((16, 16), dtype=np.uint8),
    )
    output = HeatmapManager(tmp_path).save_tile_heatmap(
        tmp_path, "B0F00000", 3,
        np.zeros((16, 16), dtype=np.uint8), None, 0.0, tile_info=tile,
    )

    assert "DUST (Filtered as OK)" in captured
    assert "No OMIT" in captured
    assert "No Dust Debug" in captured
    assert "No Binary" in captured
    assert cv2.imread(output).shape == (612, 3072, 3)
