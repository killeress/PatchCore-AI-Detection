"""Phase 7.2 Phase B — CV fusion 組 3 板視覺化單元測試"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from capi_edge_cv import EdgeDefect
from capi_heatmap import HeatmapManager


def _make_cv_fusion_defect(
    bbox=(950, 450, 40, 40),
    center=(970, 470),
    cv_filtered_mask=None,
    cv_mask_offset=(850, 350),
    is_dust=False,
    is_bomb=False,
):
    ed = EdgeDefect(
        side="aoi_edge", area=80,
        bbox=bbox, center=center, max_diff=15,
    )
    ed.source_inspector = "cv"
    ed.inspector_mode = "fusion"
    ed.threshold_used = 4
    ed.min_area_used = 40
    ed.min_max_diff_used = 20
    ed.is_suspected_dust_or_scratch = is_dust
    ed.is_bomb = is_bomb
    if cv_filtered_mask is None:
        cv_filtered_mask = np.zeros((200, 200), dtype=np.uint8)
        cv_filtered_mask[95:105, 95:105] = 255
    ed.cv_filtered_mask = cv_filtered_mask
    ed.cv_mask_offset = cv_mask_offset
    return ed


class TestCVFusionDispatch:
    """dispatch: fusion+cv 走新 renderer，其他 CV 走舊 renderer"""

    def test_fusion_cv_routes_to_new_renderer(self, tmp_path):
        ed = _make_cv_fusion_defect()
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
        with patch.object(saver, "_save_cv_fusion_edge_image",
                          return_value=str(tmp_path / "stub.png")) as mock_new, \
             patch.object(saver, "_save_patchcore_edge_image") as mock_pc:
            saver.save_edge_defect_image(
                tmp_path, "img", 0, ed, full, omit_image=None,
            )
            assert mock_new.called, "fusion+cv 應呼叫 _save_cv_fusion_edge_image"
            assert not mock_pc.called, "fusion+cv 不該走 PC renderer"

    def test_four_side_cv_still_uses_old_renderer(self, tmp_path):
        ed = _make_cv_fusion_defect(bbox=(100, 100, 40, 40), center=(120, 120))
        ed.side = "left"
        ed.inspector_mode = "cv"
        ed.source_inspector = ""
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
        with patch.object(saver, "_save_cv_fusion_edge_image") as mock_new:
            saver.save_edge_defect_image(
                tmp_path, "img", 0, ed, full, omit_image=None,
            )
            assert not mock_new.called, "四邊 CV 不該走 fusion renderer"

    def test_non_fusion_cv_aoi_edge_uses_old_renderer(self, tmp_path):
        """inspector_mode='cv' + side='aoi_edge' 仍走舊 4 板路徑"""
        ed = _make_cv_fusion_defect()
        ed.inspector_mode = "cv"
        ed.source_inspector = ""
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
        with patch.object(saver, "_save_cv_fusion_edge_image") as mock_new:
            saver.save_edge_defect_image(
                tmp_path, "img", 0, ed, full, omit_image=None,
            )
            assert not mock_new.called, "非 fusion 的 aoi_edge CV 應繼續走舊 renderer"


class TestCVFusionPanel1Detection:
    """Panel 1: 原圖 + 藍色 band 虛線輪廓 + 紅色 defect 像素"""

    def test_panel1_has_red_defect_overlay(self, tmp_path):
        """cv_filtered_mask=255 的像素在 Panel 1 應呈紅色 (R 高 G/B 低)"""
        ed = _make_cv_fusion_defect()
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
        path = saver._save_cv_fusion_edge_image(
            save_dir=tmp_path, image_name="b2", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=None,
            dust_check_fn=None,
        )
        out = cv2.imread(path)
        h, w = out.shape[:2]
        header_h = 50
        panel_h = 400
        # Panel 1 在左側（0..panel_w），掃整個 Panel 1 區域找紅色像素（容忍位置偏移）
        panel_w_approx = w // 3
        panel1 = out[header_h:header_h + panel_h, 0:panel_w_approx]
        red_pixels = (panel1[:, :, 2] > 150) & (panel1[:, :, 0] < 100) & (panel1[:, :, 1] < 100)
        assert np.sum(red_pixels) >= 10, \
            f"Panel 1 應有紅色 defect overlay 像素 (>=10 px)，實 {np.sum(red_pixels)}"

    def test_panel1_has_blue_band_contour(self, tmp_path):
        """有 panel_polygon 與 band_px 時，Panel 1 沿 polygon 邊畫藍虛線"""
        polygon = np.array([[900, 400], [1100, 400], [1100, 600], [900, 600]], dtype=np.int32)
        ed = _make_cv_fusion_defect(bbox=(920, 420, 40, 40), center=(940, 440))
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
        class _Cfg:
            aoi_edge_boundary_band_px = 40
        path = saver._save_cv_fusion_edge_image(
            save_dir=tmp_path, image_name="b2band", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=None,
            dust_check_fn=None, edge_config=_Cfg(), panel_polygon=polygon,
        )
        out = cv2.imread(path)
        h, w = out.shape[:2]
        panel_w_approx = w // 3
        header_h = 50; panel_h = 400
        panel1 = out[header_h:header_h + panel_h, 0:panel_w_approx]
        blue_pixels = (panel1[:, :, 0] > 150) & (panel1[:, :, 2] < 100)
        assert np.sum(blue_pixels) > 10, \
            f"Panel 1 應該有藍色 band 輪廓像素（>10 px），實 {np.sum(blue_pixels)}"


class TestCVFusionPanel2OMITDust:
    """Panel 2: OMIT 同 ROI + 藍色 dust overlay"""

    def test_panel2_shows_omit_gray(self, tmp_path):
        """Panel 2 中心像素應反映 OMIT 原圖灰度（180），不是黑 placeholder (40)"""
        omit = np.full((2000, 2000), 180, dtype=np.uint8)
        ed = _make_cv_fusion_defect()
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
        path = saver._save_cv_fusion_edge_image(
            save_dir=tmp_path, image_name="b3omit", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
            dust_check_fn=None,
        )
        out = cv2.imread(path)
        h, w = out.shape[:2]
        panel_w_approx = w // 3
        header_h = 50; panel_h = 400
        p2_cx = panel_w_approx + panel_w_approx // 2 + 5
        p2_cy = header_h + panel_h // 2
        px = out[p2_cy, p2_cx]
        assert 120 < px[0] < 200 and 120 < px[1] < 200 and 120 < px[2] < 200, \
            f"Panel 2 應為灰 OMIT (~180)，實 {px.tolist()}"

    def test_panel2_has_blue_dust_overlay(self, tmp_path):
        """dust_check_fn 回傳 mask + is_dust=True 時 Panel 2 對應區塊偏藍"""
        omit = np.full((2000, 2000), 180, dtype=np.uint8)

        def stub_dust(img):
            h, w = img.shape[:2]
            m = np.zeros((h, w), dtype=np.uint8)
            m[h // 2 - 30:h // 2 + 30, w // 2 - 30:w // 2 + 30] = 255
            return True, m, 0.1, ""

        ed = _make_cv_fusion_defect()
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
        path = saver._save_cv_fusion_edge_image(
            save_dir=tmp_path, image_name="b3dust", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
            dust_check_fn=stub_dust,
        )
        out = cv2.imread(path)
        h, w = out.shape[:2]
        panel_w_approx = w // 3
        header_h = 50; panel_h = 400
        p2_cx = panel_w_approx + panel_w_approx // 2 + 5
        p2_cy = header_h + panel_h // 2
        px = out[p2_cy, p2_cx]
        assert px[0] > px[2] + 30, \
            f"Panel 2 dust 區應偏藍 B>R+30，實 B={px[0]} R={px[2]}"
