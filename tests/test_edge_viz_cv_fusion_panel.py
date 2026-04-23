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
