"""Phase 7.2 Phase A — PC 組 3 板視覺化單元測試"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from capi_edge_cv import EdgeDefect
from capi_heatmap import HeatmapManager


def _make_edge_defect(
    center=(1000, 500),
    pc_roi=None,
    pc_fg_mask=None,
    pc_anomaly_map=None,
    shift=(0, 0),
    pc_roi_origin=None,
    fallback_reason="",
    source="patchcore",
    inspector_mode="fusion",
):
    """建立測試用 EdgeDefect（PC fusion 情境）"""
    roi_size = 512
    if pc_roi is None:
        pc_roi = np.full((roi_size, roi_size, 3), 128, dtype=np.uint8)
    if pc_fg_mask is None:
        pc_fg_mask = np.full((roi_size, roi_size), 255, dtype=np.uint8)
    if pc_anomaly_map is None:
        pc_anomaly_map = np.zeros((roi_size, roi_size), dtype=np.float32)
    if pc_roi_origin is None:
        pc_roi_origin = (center[0] - roi_size // 2 + shift[0],
                         center[1] - roi_size // 2 + shift[1])

    ed = EdgeDefect(
        side="aoi_edge", area=100,
        bbox=(pc_roi_origin[0], pc_roi_origin[1], roi_size, roi_size),
        center=center, max_diff=0,
    )
    ed.source_inspector = source
    ed.inspector_mode = inspector_mode
    ed.patchcore_score = 0.8
    ed.patchcore_threshold = 0.5
    ed.pc_roi = pc_roi
    ed.pc_fg_mask = pc_fg_mask
    ed.pc_anomaly_map = pc_anomaly_map
    ed.pc_roi_origin_x = pc_roi_origin[0]
    ed.pc_roi_origin_y = pc_roi_origin[1]
    ed.pc_roi_shift_dx = shift[0]
    ed.pc_roi_shift_dy = shift[1]
    ed.pc_roi_fallback_reason = fallback_reason
    return ed


class TestPCGroupPanel1:
    """Panel 1: AOI centered raw ROI，不加 mask 不加 marker"""

    def test_panel1_is_raw_aoi_centered_crop(self, tmp_path):
        """Panel 1 像素 = full_image AOI 中心 512×512 區塊（不被 fg_mask 蓋紅）"""
        full = np.zeros((2000, 2000, 3), dtype=np.uint8)
        full[400:600, 900:1100] = 200  # AOI 區塊一塊亮色方塊

        # 提供 omit_image 讓 layout 為 3-panel，方便以 w//3 定位 Panel 1
        omit = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        ed = _make_edge_defect(center=(1000, 500),
                                shift=(0, -192))  # shifted up
        saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
        path = saver._save_patchcore_edge_image(
            save_dir=tmp_path, image_name="test", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
        )
        out = cv2.imread(path)
        assert out is not None, "output image should exist"

        h, w = out.shape[:2]
        panel_w_approx = w // 3
        panel1_center_y = 50 + 400 // 2
        panel1_center_x = panel_w_approx // 2
        px = out[panel1_center_y, panel1_center_x]
        # AOI 中心應為 raw 亮方塊 (fixture 設定 full[400:600, 900:1100] = 200)
        # 不是暗紅 mask (0,0,60/120)，也不是黑 pad
        assert px[0] > 150 and px[1] > 150 and px[2] > 150, \
            f"Panel 1 中心應為 raw pixel (~200)，實測 {px.tolist()}"

    def test_panel1_has_no_yellow_marker(self, tmp_path):
        """Panel 1 中央不應有黃色圓圈 (0, 255, 255)"""
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        # 提供 omit_image 讓 layout 為 3-panel
        omit = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        ed = _make_edge_defect(center=(1000, 500))
        saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
        path = saver._save_patchcore_edge_image(
            save_dir=tmp_path, image_name="test", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
        )
        out = cv2.imread(path)
        h, w = out.shape[:2]
        panel_w_approx = w // 3
        cy, cx = 50 + 200, panel_w_approx // 2
        region = out[cy - 20:cy + 20, cx - 20:cx + 20]
        yellow_mask = (region[:, :, 0] <= 50) & (region[:, :, 1] >= 180) & (region[:, :, 2] >= 180)
        assert not np.any(yellow_mask), "Panel 1 中心 40×40 區域不應出現黃色 marker 像素"
