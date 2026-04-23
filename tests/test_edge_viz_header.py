"""Phase 7.2 Phase C — Edge header helper 單元測試"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from capi_heatmap import HeatmapManager


class TestRenderEdgeHeader:
    """helper：產生 header image，含 info 白字 + verdict 彩字 + 可選 extra"""

    def test_returns_image_of_correct_size(self):
        saver = HeatmapManager(base_dir=".", save_format="png")
        img = saver._render_edge_header(
            width=1200, info="PC Edge | Score:0.5 | ", verdict="NG",
            verdict_color=(0, 0, 255),
        )
        assert img.shape[0] == 50 and img.shape[1] == 1200
        assert img.dtype == np.uint8

    def test_verdict_uses_correct_color(self):
        """NG 區塊應有紅色像素、OK 區塊應有綠色像素"""
        saver = HeatmapManager(base_dir=".", save_format="png")
        img_ng = saver._render_edge_header(
            width=800, info="X | ", verdict="NG", verdict_color=(0, 0, 255),
        )
        img_ok = saver._render_edge_header(
            width=800, info="X | ", verdict="OK", verdict_color=(0, 255, 0),
        )
        # NG header 右側區塊掃 red pixel 數量
        right_ng = img_ng[:, 400:]
        red = (right_ng[:, :, 2] > 150) & (right_ng[:, :, 0] < 80)
        assert np.sum(red) > 5, "NG verdict 應畫紅字"
        right_ok = img_ok[:, 400:]
        green = (right_ok[:, :, 1] > 150) & (right_ok[:, :, 0] < 80) & (right_ok[:, :, 2] < 80)
        assert np.sum(green) > 5, "OK verdict 應畫綠字"

    def test_extra_right_side_rendered(self, monkeypatch):
        """extra_right 文字應渲染（透過 putText 攔截驗證內容）"""
        captured = []
        _real = cv2.putText
        def cap(img, text, *a, **kw):
            captured.append(text)
            return _real(img, text, *a, **kw)
        monkeypatch.setattr("capi_heatmap.cv2.putText", cap)

        saver = HeatmapManager(base_dir=".", save_format="png")
        saver._render_edge_header(
            width=1200, info="PC Edge | ", verdict="NG", verdict_color=(0, 0, 255),
            extra_right="PC dx=+0 dy=-192",
        )
        all_text = " ".join(captured)
        assert "PC dx=+0 dy=-192" in all_text
