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


class TestPCGroupPanel3OMITAlign:
    """Panel 3: OMIT 擷取位置 = shifted ROI origin，不是 AOI center"""

    def test_omit_crop_uses_shifted_origin(self, tmp_path):
        """shift=(0,-192) 時 OMIT 擷取應從 pc_roi_origin_y 開始（比 AOI center 上移 192 px）"""
        omit = np.zeros((2000, 2000), dtype=np.uint8)
        omit[300:400, :] = 255

        # AOI center y=500，shift dy=-192 → pc_roi_origin_y = 500 - 256 + (-192) = 52
        # shifted crop y=[52, 564]，亮帶 y=[300,400] 在 crop 中段
        ed = _make_edge_defect(center=(1000, 500), shift=(0, -192))
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
        path = saver._save_patchcore_edge_image(
            save_dir=tmp_path, image_name="test_omit", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
        )
        out = cv2.imread(path)
        h, w = out.shape[:2]
        panel_w_approx = w // 3
        panel3_left = panel_w_approx * 2 + 20
        panel3_right = w - 5

        header_h = 50
        panel_h = 400
        col_mid = (panel3_left + panel3_right) // 2
        vertical_profile = cv2.cvtColor(
            out[header_h:header_h + panel_h, col_mid:col_mid + 5], cv2.COLOR_BGR2GRAY
        ).mean(axis=1)

        peak_y_ratio = int(np.argmax(vertical_profile)) / panel_h
        # shifted crop: 亮帶 y=300-400 在 crop y=[52,564] 映射到 ratio ≈ 0.484~0.680
        assert 0.35 <= peak_y_ratio <= 0.75, \
            f"shifted crop 亮帶應該在 Panel 3 中段 (0.35~0.75)，實 ratio={peak_y_ratio}"

    def test_omit_crop_no_shift_matches_centered(self, tmp_path):
        """shift=(0,0) 時 OMIT crop 仍以 centered 位置擷取"""
        omit = np.zeros((2000, 2000), dtype=np.uint8)
        omit[244:500, :] = 255  # AOI center crop 上半部亮

        ed = _make_edge_defect(center=(1000, 500), shift=(0, 0))
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
        path = saver._save_patchcore_edge_image(
            save_dir=tmp_path, image_name="test_omit_nc", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
        )
        out = cv2.imread(path)
        h, w = out.shape[:2]
        panel_w_approx = w // 3
        panel3_left = panel_w_approx * 2 + 20
        col_mid = (panel3_left + (w - 5)) // 2
        header_h = 50; panel_h = 400
        vp = cv2.cvtColor(
            out[header_h:header_h + panel_h, col_mid:col_mid + 5], cv2.COLOR_BGR2GRAY
        ).mean(axis=1)
        peak_y_ratio = int(np.argmax(vp)) / panel_h
        # 亮帶 y=[244, 500] 在 crop y=[244, 756] 映射到 ratio [0, 0.5]
        assert peak_y_ratio <= 0.5, \
            f"無 shift 時亮帶應該在 Panel 3 上半 (<=0.5)，實 ratio={peak_y_ratio}"


class TestPCGroupPanel3DustOverlay:
    """Panel 3: 套 dust_check_fn 回傳的藍色 overlay (BGR 255,100,0)"""

    def test_dust_overlay_applied_when_fn_returns_mask(self, tmp_path):
        """dust_check_fn 回傳 dust_mask 時，Panel 3 中 dust 區塊偏藍色 (B 高 R 低)"""
        omit = np.full((2000, 2000), 128, dtype=np.uint8)

        def stub_dust_check(img):
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[h // 2 - 50:h // 2 + 50, w // 2 - 50:w // 2 + 50] = 255
            return True, mask, 0.1, "stub dust"

        ed = _make_edge_defect(center=(1000, 500), shift=(0, 0))
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
        path = saver._save_patchcore_edge_image(
            save_dir=tmp_path, image_name="test_dust", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
            dust_check_fn=stub_dust_check,
        )
        out = cv2.imread(path)
        h_out, w_out = out.shape[:2]
        panel_w_approx = w_out // 3
        panel3_left = panel_w_approx * 2 + 20
        header_h = 50; panel_h = 400
        center_x = (panel3_left + w_out - 5) // 2
        center_y = header_h + panel_h // 2
        px = out[center_y, center_x]
        # 藍色 overlay 後 B > R 明顯
        assert px[0] > px[2] + 30, \
            f"dust 中心像素應偏藍 (B>R+30)，實為 B={px[0]} G={px[1]} R={px[2]}"

    def test_no_dust_overlay_when_fn_is_none(self, tmp_path):
        """dust_check_fn 為 None 時 Panel 3 保持純 OMIT (灰)，無藍色覆蓋"""
        omit = np.full((2000, 2000), 128, dtype=np.uint8)
        ed = _make_edge_defect(center=(1000, 500))
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
        path = saver._save_patchcore_edge_image(
            save_dir=tmp_path, image_name="test_nodust", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
            dust_check_fn=None,
        )
        out = cv2.imread(path)
        h_out, w_out = out.shape[:2]
        panel3_left = (w_out // 3) * 2 + 20
        center_x = (panel3_left + w_out - 5) // 2
        center_y = 50 + 200
        px = out[center_y, center_x]
        # 純灰，B/G/R 差不多
        assert abs(int(px[0]) - int(px[2])) < 20, \
            f"無 dust_check_fn 時 Panel 3 應為灰 (B≈R)，實為 B={px[0]} R={px[2]}"

    def test_dust_check_exception_does_not_crash(self, tmp_path):
        """dust_check_fn 丟 exception 時 Panel 3 仍能產圖（不爆，無 overlay）"""
        omit = np.full((2000, 2000), 128, dtype=np.uint8)
        def failing_fn(img):
            raise RuntimeError("dust check failed")
        ed = _make_edge_defect(center=(1000, 500))
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
        path = saver._save_patchcore_edge_image(
            save_dir=tmp_path, image_name="test_except", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
            dust_check_fn=failing_fn,
        )
        assert Path(path).exists(), "即使 dust_check_fn 丟 exception 也要能產出圖"
