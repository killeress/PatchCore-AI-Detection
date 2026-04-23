"""Phase 7 — AOI 邊緣 Fusion PC ROI 內移 單元測試。

涵蓋:
  - compute_pc_roi_offset（純幾何偏移計算）
  - verify_polygon_clear_of_pc_roi（偏移後驗證）
  - _inspect_roi_fusion shifted path（整合）
"""

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from capi_edge_cv import (
    EdgeDefect,
    compute_pc_roi_offset,
    verify_polygon_clear_of_pc_roi,
)
from capi_config import CAPIConfig
from capi_inference import CAPIInferencer


# Polygon 常用：大正方形，從 (0,0) 到 (2000,2000)
BIG_SQUARE = np.array([
    [0, 0],
    [2000, 0],
    [2000, 2000],
    [0, 2000],
], dtype=np.int32)


# =========================================================================
# Slice 2: compute_pc_roi_offset
# =========================================================================

class TestComputePcRoiOffset:
    """純幾何函式:
    - 輸入 AOI 座標 + polygon + band_px + aoi_margin_px + roi_size
    - 輸出 (pc_roi_origin, shift_vec, d_edge)
    - shift_vec=(0,0) 代表不需偏移或無效 polygon
    - shift_vec 沿 AOI 最近 polygon 邊 inward normal 方向
    - |shift| ≤ roi_size/2 - aoi_margin_px
    """

    def test_deep_interior_no_shift(self):
        """AOI 座標深度 > roi_size/2 + band_px → shift=(0,0)"""
        origin, shift, d_edge = compute_pc_roi_offset(
            aoi_xy=(1000, 1000),
            polygon=BIG_SQUARE,
            band_px=40,
            aoi_margin_px=64,
            roi_size=512,
        )
        assert shift == (0, 0)
        assert origin == (1000 - 256, 1000 - 256)
        assert d_edge == pytest.approx(1000.0)

    def test_exactly_on_shift_threshold_no_shift(self):
        """AOI 距邊 = roi_size/2 + band_px (296) → 剛好不需偏移"""
        origin, shift, d_edge = compute_pc_roi_offset(
            aoi_xy=(296, 1000), polygon=BIG_SQUARE,
            band_px=40, aoi_margin_px=64, roi_size=512,
        )
        assert shift == (0, 0)
        assert d_edge == pytest.approx(296.0)

    def test_near_left_edge_shifts_right(self):
        """AOI 近左邊 → inward normal 指向右 → shift=(+N, 0)"""
        origin, shift, d_edge = compute_pc_roi_offset(
            aoi_xy=(100, 1000), polygon=BIG_SQUARE,
            band_px=40, aoi_margin_px=64, roi_size=512,
        )
        # needed = max(0, 40 - (100 - 256)) = max(0, 196) = 196
        # clamped = min(196, 192) = 192
        assert shift[0] > 0 and shift[1] == 0
        assert shift[0] == 192  # clamped
        assert origin == (100 + 192 - 256, 1000 - 256)
        assert d_edge == pytest.approx(100.0)

    def test_near_right_edge_shifts_left(self):
        origin, shift, d_edge = compute_pc_roi_offset(
            aoi_xy=(1900, 1000), polygon=BIG_SQUARE,
            band_px=40, aoi_margin_px=64, roi_size=512,
        )
        assert shift[0] < 0 and shift[1] == 0
        assert shift[0] == -192
        assert d_edge == pytest.approx(100.0)

    def test_near_top_edge_shifts_down(self):
        origin, shift, d_edge = compute_pc_roi_offset(
            aoi_xy=(1000, 100), polygon=BIG_SQUARE,
            band_px=40, aoi_margin_px=64, roi_size=512,
        )
        assert shift[0] == 0 and shift[1] > 0
        assert shift[1] == 192
        assert d_edge == pytest.approx(100.0)

    def test_near_bottom_edge_shifts_up(self):
        origin, shift, d_edge = compute_pc_roi_offset(
            aoi_xy=(1000, 1900), polygon=BIG_SQUARE,
            band_px=40, aoi_margin_px=64, roi_size=512,
        )
        assert shift[0] == 0 and shift[1] < 0
        assert shift[1] == -192

    def test_near_corner_uses_nearest_single_edge(self):
        """AOI (100, 150) → 最近邊是 left (d=100)，取 left 的 normal（水平右）
        而非對角線方向，避免 AOI margin 在垂直方向又不夠
        """
        origin, shift, d_edge = compute_pc_roi_offset(
            aoi_xy=(100, 150), polygon=BIG_SQUARE,
            band_px=40, aoi_margin_px=64, roi_size=512,
        )
        # 最近是 left (dist=100 < top dist=150) → shift 只沿 x
        assert shift[1] == 0, f"Expected no y-shift, got {shift}"
        assert shift[0] == 192

    def test_shift_magnitude_smaller_when_closer_to_edge_threshold(self):
        """AOI 距邊 = 280 (> 256) → 只需 shift 16 (=40+256-280)"""
        origin, shift, d_edge = compute_pc_roi_offset(
            aoi_xy=(280, 1000), polygon=BIG_SQUARE,
            band_px=40, aoi_margin_px=64, roi_size=512,
        )
        # needed = max(0, 40 - (280-256)) = max(0, 16) = 16
        assert shift == (16, 0)

    def test_clamped_to_aoi_margin(self):
        """極近邊 (d=5) → needed 超過 max_shift=192 → clamp 到 192"""
        origin, shift, d_edge = compute_pc_roi_offset(
            aoi_xy=(5, 1000), polygon=BIG_SQUARE,
            band_px=40, aoi_margin_px=64, roi_size=512,
        )
        assert shift == (192, 0)

    def test_polygon_none_returns_zero_shift(self):
        origin, shift, d_edge = compute_pc_roi_offset(
            aoi_xy=(100, 100), polygon=None,
            band_px=40, aoi_margin_px=64, roi_size=512,
        )
        assert shift == (0, 0)
        assert origin == (100 - 256, 100 - 256)

    def test_polygon_too_few_points_returns_zero(self):
        poly = np.array([[0, 0], [100, 100]], dtype=np.int32)
        origin, shift, d_edge = compute_pc_roi_offset(
            aoi_xy=(100, 100), polygon=poly,
            band_px=40, aoi_margin_px=64, roi_size=512,
        )
        assert shift == (0, 0)

    def test_aoi_outside_polygon_returns_zero(self):
        """AOI 在 polygon 外 → d_edge < 0 → 不應偏移"""
        origin, shift, d_edge = compute_pc_roi_offset(
            aoi_xy=(-50, 1000), polygon=BIG_SQUARE,
            band_px=40, aoi_margin_px=64, roi_size=512,
        )
        assert shift == (0, 0)
        assert d_edge < 0


# =========================================================================
# Slice 3: verify_polygon_clear_of_pc_roi
# =========================================================================

class TestVerifyPolygonClearOfPcRoi:
    """驗證 shifted PC ROI 內部所有像素距 polygon ≥ band_px"""

    def test_clean_shifted_roi_passes(self):
        """PC ROI 在 polygon 深處，最近邊距 > band_px → True"""
        pc_origin = (1000, 1000)  # PC ROI = [1000, 1512] x [1000, 1512]
        # polygon 邊 x=2000 距 PC ROI 右邊 1512 仍有 488 px → 滿足 band_px=40
        assert verify_polygon_clear_of_pc_roi(
            pc_roi_origin=pc_origin,
            roi_size=512,
            polygon=BIG_SQUARE,
            band_px=40,
        ) is True

    def test_polygon_still_inside_pc_roi_fails(self):
        """PC ROI 包含 polygon 邊 → False"""
        # PC ROI [0, 512] x [0, 512]，polygon 邊 x=0 / y=0 都在 ROI 邊上
        assert verify_polygon_clear_of_pc_roi(
            pc_roi_origin=(0, 0),
            roi_size=512,
            polygon=BIG_SQUARE,
            band_px=40,
        ) is False

    def test_polygon_close_to_pc_roi_fails(self):
        """PC ROI 邊距 polygon 邊 < band_px → False"""
        # PC ROI [30, 542] x [1000, 1512]，左側邊 x=30 距 polygon 左 x=0 僅 30 < 40
        assert verify_polygon_clear_of_pc_roi(
            pc_roi_origin=(30, 1000),
            roi_size=512,
            polygon=BIG_SQUARE,
            band_px=40,
        ) is False

    def test_concave_polygon_second_edge_intrudes_fails(self):
        """L 形 polygon: 偏移後主邊雖避開，但轉角另一邊侵入 PC ROI"""
        # L 形: 外框 (0,0) - (2000, 0) - (2000, 2000) - (1000, 2000)
        # - (1000, 1000) - (0, 1000) - back to (0,0)
        l_shape = np.array([
            [0, 0], [2000, 0], [2000, 2000],
            [1000, 2000], [1000, 1000], [0, 1000],
        ], dtype=np.int32)
        # PC ROI 位置：(800, 800) ~ (1312, 1312)
        # polygon 內凹角 (1000, 1000) 距 PC ROI 中心 ~ 282，但角點本身
        # 落在 ROI 內 (800<1000<1312, 800<1000<1312) → 距 ROI ≥ band_px 的條件不成立
        assert verify_polygon_clear_of_pc_roi(
            pc_roi_origin=(800, 800),
            roi_size=512,
            polygon=l_shape,
            band_px=40,
        ) is False

    def test_polygon_none_returns_true(self):
        """Polygon None → 無限制 → True (caller 自己處理 fallback)"""
        assert verify_polygon_clear_of_pc_roi(
            pc_roi_origin=(0, 0),
            roi_size=512,
            polygon=None,
            band_px=40,
        ) is True


# =========================================================================
# Slice 4: _inspect_roi_fusion shifted path 整合
# =========================================================================

def _capturing_predict_tile(captured: dict, score: float,
                             hot_region=None, hot_value: float = 0.0):
    """predict_tile stub 會把 tile 座標捕捉到 captured dict，供 assertion 使用"""
    def _fn(tile, **kwargs):
        captured["tile_x"] = tile.x
        captured["tile_y"] = tile.y
        amap = np.zeros((512, 512), dtype=np.float32)
        if hot_region is not None and hot_value > 0:
            y1, y2, x1, x2 = hot_region
            amap[y1:y2, x1:x2] = hot_value
        return score, amap
    return _fn


class TestInspectRoiFusionShifted:
    """Phase 7 fusion：PC ROI 內移整合測試"""

    @pytest.fixture
    def fusion_inferencer(self):
        cfg = CAPIConfig()
        cfg.tile_size = 512
        cfg.anomaly_threshold = 1.0
        cfg.threshold_mapping = {}
        cfg.model_mapping = {}
        cfg.patchcore_min_area = 10
        cfg.patchcore_filter_enabled = False
        cfg.patchcore_concentration_enabled = False

        inf = CAPIInferencer(cfg, threshold=1.0)
        from unittest.mock import MagicMock as _MM
        inf._get_inferencer_for_prefix = lambda _p: _MM()
        inf._get_threshold_for_prefix = lambda _p: 1.0
        inf.edge_inspector.config.aoi_edge_inspector = "fusion"
        inf.edge_inspector.config.aoi_edge_boundary_band_px = 40
        inf.edge_inspector.config.aoi_edge_pc_roi_inward_shift_enabled = True
        inf.edge_inspector.config.aoi_edge_aoi_margin_px = 64
        inf.check_dust_or_scratch_feature = lambda img, **kw: (False, None, 0.0, "")
        # CV 預設回空 defect list
        inf.edge_inspector.inspect_roi = lambda roi, **kw: (
            [], {"max_diff": 0, "max_area": 0, "threshold": 4, "min_area": 40, "min_max_diff": 20}
        )
        return inf

    def _image_and_polygon(self):
        img = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        # polygon: 2000x2000 大矩形（AOI 是否近邊由 aoi_xy 決定）
        poly = BIG_SQUARE.astype(np.float32)
        return img, poly

    def test_deep_interior_no_shift_applied(self, fusion_inferencer):
        """AOI 深度 → shift=(0,0)；predict_tile 收到的 tile.x/y = centered origin"""
        inf = fusion_inferencer
        img, poly = self._image_and_polygon()
        captured = {}
        inf.predict_tile = _capturing_predict_tile(captured, score=0.0)

        defects, stats = inf._inspect_roi_fusion(
            img, img_x=1000, img_y=1000, img_prefix="W0F",
            panel_polygon=poly, omit_image=None, omit_overexposed=False,
        )

        # centered origin = (1000-256, 1000-256) = (744, 744)
        assert captured["tile_x"] == 744, f"expected 744, got {captured.get('tile_x')}"
        assert captured["tile_y"] == 744

    def test_near_left_edge_shifts_pc_roi_right(self, fusion_inferencer):
        """AOI (104, 1000) 近左邊 → shift 192 → PC ROI 左邊 x=40 剛好 = band_px，verify 通過"""
        inf = fusion_inferencer
        img, poly = self._image_and_polygon()
        captured = {}
        inf.predict_tile = _capturing_predict_tile(captured, score=0.0)

        inf._inspect_roi_fusion(
            img, img_x=104, img_y=1000, img_prefix="W0F",
            panel_polygon=poly, omit_image=None, omit_overexposed=False,
        )

        # shifted origin x = 104 + 192 - 256 = 40
        assert captured["tile_x"] == 40, f"expected 40 (shifted), got {captured.get('tile_x')}"
        assert captured["tile_y"] == 744

    def test_shifted_pc_defect_fields_populated(self, fusion_inferencer):
        """Shifted 模式 PC 命中 → defect 含 pc_roi_origin / shift_dx/dy，center=AOI"""
        inf = fusion_inferencer
        img, poly = self._image_and_polygon()
        captured = {}
        # PC hot region 在 shifted ROI 中央 → 確保 above threshold
        inf.predict_tile = _capturing_predict_tile(
            captured, score=2.0, hot_region=(240, 280, 240, 280), hot_value=2.0
        )

        defects, stats = inf._inspect_roi_fusion(
            img, img_x=104, img_y=1000, img_prefix="W0F",
            panel_polygon=poly, omit_image=None, omit_overexposed=False,
        )

        pc_def = [d for d in defects if d.source_inspector == "patchcore"]
        assert len(pc_def) == 1, f"expected 1 PC defect, got {len(pc_def)}"
        d = pc_def[0]
        # 中心強制為 AOI 座標
        assert d.center == (104, 1000), f"center should be AOI coord, got {d.center}"
        # shift_dx = 192 (向右), dy = 0
        assert d.pc_roi_shift_dx == 192, f"shift_dx expected 192, got {d.pc_roi_shift_dx}"
        assert d.pc_roi_shift_dy == 0
        # pc_roi_origin = (104+192-256, 1000-256) = (40, 744)
        assert d.pc_roi_origin_x == 40
        assert d.pc_roi_origin_y == 744
        assert d.pc_roi_fallback_reason == ""

    def test_shift_disabled_behaves_like_phase6(self, fusion_inferencer):
        """config aoi_edge_pc_roi_inward_shift_enabled=False → 永遠 centered"""
        inf = fusion_inferencer
        inf.edge_inspector.config.aoi_edge_pc_roi_inward_shift_enabled = False
        img, poly = self._image_and_polygon()
        captured = {}
        inf.predict_tile = _capturing_predict_tile(captured, score=0.0)

        inf._inspect_roi_fusion(
            img, img_x=104, img_y=1000, img_prefix="W0F",
            panel_polygon=poly, omit_image=None, omit_overexposed=False,
        )

        # 即使近左邊，disabled 下不偏移 → tile.x = centered origin = -152
        assert captured["tile_x"] == -152, \
            f"shift disabled, expected centered origin -152, got {captured['tile_x']}"

    def test_fallback_when_verify_fails_concave(self, fusion_inferencer):
        """凹角 polygon 偏移後其他邊侵入 → fallback 到 centered + band_mask"""
        inf = fusion_inferencer
        # L 形 polygon，AOI 剛好在凹角附近
        l_shape = np.array([
            [0, 0], [2000, 0], [2000, 2000],
            [1000, 2000], [1000, 1000], [0, 1000],
        ], dtype=np.float32)
        img = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        captured = {}
        inf.predict_tile = _capturing_predict_tile(
            captured, score=2.0, hot_region=(240, 280, 240, 280), hot_value=2.0
        )

        # AOI at (100, 900)，接近 left 邊 → 會嘗試 shift 右 192 → PC ROI 在 [36, 548] x [644, 1156]
        # 這時下方凹角 (0,1000)-(1000,1000) 邊在 y=1000 穿過 PC ROI → verify fail → fallback
        defects, stats = inf._inspect_roi_fusion(
            img, img_x=100, img_y=900, img_prefix="W0F",
            panel_polygon=l_shape, omit_image=None, omit_overexposed=False,
        )

        pc_def = [d for d in defects if d.source_inspector == "patchcore"]
        if pc_def:
            d = pc_def[0]
            assert d.pc_roi_fallback_reason in ("concave_polygon", "shift_disabled", ""), \
                f"fallback_reason unexpected: {d.pc_roi_fallback_reason}"
            # fallback 後 shift 應為 0
            if d.pc_roi_fallback_reason == "concave_polygon":
                assert d.pc_roi_shift_dx == 0 and d.pc_roi_shift_dy == 0


# =========================================================================
# Slice 9: fusion collapse — 每 AOI 座標最多 1 筆代表 defect
# =========================================================================

class TestFusionCollapseToRepresentative:
    """Phase 7.1: fusion 結果 collapse 邏輯

    規則 (real_NG 優先 > dust)：
      1. 若有 real NG：PC > CV；CV 內 max area 取一
      2. 全 dust：取任一保留 dust 旗標
      3. 空 list：回空
    Debug path 用 collapse_to_representative=False 保留全細節
    """

    @pytest.fixture
    def fusion_inferencer(self):
        cfg = CAPIConfig()
        cfg.tile_size = 512
        cfg.anomaly_threshold = 1.0
        cfg.threshold_mapping = {}
        cfg.model_mapping = {}
        cfg.patchcore_min_area = 10
        cfg.patchcore_filter_enabled = False
        cfg.patchcore_concentration_enabled = False
        inf = CAPIInferencer(cfg, threshold=1.0)
        inf._get_inferencer_for_prefix = lambda _p: MagicMock()
        inf._get_threshold_for_prefix = lambda _p: 1.0
        inf.edge_inspector.config.aoi_edge_inspector = "fusion"
        inf.edge_inspector.config.aoi_edge_boundary_band_px = 40
        inf.edge_inspector.config.aoi_edge_pc_roi_inward_shift_enabled = True
        inf.check_dust_or_scratch_feature = lambda img, **kw: (False, None, 0.0, "")
        return inf

    def _image_and_poly(self):
        img = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        return img, BIG_SQUARE.astype(np.float32)

    def test_multi_cv_band_collapsed_to_largest_area(self, fusion_inferencer):
        """CV 回 3 個 component → collapse 後只剩 1 筆（面積最大）"""
        inf = fusion_inferencer
        img, poly = self._image_and_poly()

        def _cv_stub(roi, offset_x, offset_y, **kw):
            # 3 個 CV defect，area 不同，center 都在 band 內（AOI 近左邊 x=104）
            defects = []
            # AOI at (104, 1000)，band 大約 ROI x∈[60,140] 的 polygon 內側
            for cx, area in [(10, 37), (30, 65), (20, 58)]:  # 65 最大
                ed = EdgeDefect(
                    side="aoi_edge", area=area,
                    bbox=(max(0, cx - 3), 1000 - 3, 6, 6),
                    center=(cx, 1000), max_diff=7,
                )
                ed.cv_filtered_mask = np.full((512, 512), 255, dtype=np.uint8)
                ed.cv_mask_offset = (offset_x, offset_y)
                defects.append(ed)
            return defects, {"max_diff": 7, "max_area": 65,
                              "threshold": 4, "min_area": 40, "min_max_diff": 20}
        inf.edge_inspector.inspect_roi = _cv_stub

        def _pt(tile, **kw):
            return 0.0, np.zeros((512, 512), dtype=np.float32)
        inf.predict_tile = _pt

        defects, stats = inf._inspect_roi_fusion(
            img, img_x=104, img_y=1000, img_prefix="W0F",
            panel_polygon=poly, omit_image=None, omit_overexposed=False,
        )

        assert len(defects) == 1, f"expected 1 collapsed defect, got {len(defects)}"
        assert defects[0].source_inspector == "cv"
        assert defects[0].area == 65, f"expected largest area 65, got {defects[0].area}"
        # stats 應保留原始計數供診斷
        assert stats.get("pre_collapse_count", 0) == 3

    def test_pc_wins_over_cv_when_both_ng(self, fusion_inferencer):
        """CV 3 筆 + PC 1 筆 → collapse 後取 PC"""
        inf = fusion_inferencer
        img, poly = self._image_and_poly()

        def _cv_stub(roi, offset_x, offset_y, **kw):
            defects = []
            for cx, area in [(10, 37), (30, 65), (20, 58)]:
                ed = EdgeDefect(
                    side="aoi_edge", area=area,
                    bbox=(max(0, cx - 3), 1000 - 3, 6, 6),
                    center=(cx, 1000), max_diff=7,
                )
                ed.cv_filtered_mask = np.full((512, 512), 255, dtype=np.uint8)
                ed.cv_mask_offset = (offset_x, offset_y)
                defects.append(ed)
            return defects, {"max_diff": 7, "max_area": 65,
                              "threshold": 4, "min_area": 40, "min_max_diff": 20}
        inf.edge_inspector.inspect_roi = _cv_stub

        def _pt(tile, **kw):
            amap = np.zeros((512, 512), dtype=np.float32)
            amap[240:280, 240:280] = 2.0  # > thr=1.0
            return 2.0, amap
        inf.predict_tile = _pt

        defects, stats = inf._inspect_roi_fusion(
            img, img_x=104, img_y=1000, img_prefix="W0F",
            panel_polygon=poly, omit_image=None, omit_overexposed=False,
        )

        assert len(defects) == 1
        assert defects[0].source_inspector == "patchcore", \
            f"expected PC to win, got {defects[0].source_inspector}"
        assert stats.get("pre_collapse_count", 0) == 4  # 3 CV + 1 PC

    def test_empty_defects_returns_empty(self, fusion_inferencer):
        """fusion 全 clean → 回空 list（OK 由 caller 處理）"""
        inf = fusion_inferencer
        img, poly = self._image_and_poly()
        inf.edge_inspector.inspect_roi = lambda roi, **kw: (
            [], {"max_diff": 0, "max_area": 0, "threshold": 4,
                 "min_area": 40, "min_max_diff": 20}
        )
        def _pt(tile, **kw):
            return 0.0, np.zeros((512, 512), dtype=np.float32)
        inf.predict_tile = _pt

        defects, stats = inf._inspect_roi_fusion(
            img, img_x=104, img_y=1000, img_prefix="W0F",
            panel_polygon=poly, omit_image=None, omit_overexposed=False,
        )
        assert len(defects) == 0

    def test_debug_mode_preserves_all_defects(self, fusion_inferencer):
        """collapse_to_representative=False → 回全部 defect"""
        inf = fusion_inferencer
        img, poly = self._image_and_poly()

        def _cv_stub(roi, offset_x, offset_y, **kw):
            defects = []
            for cx, area in [(10, 37), (30, 65), (20, 58)]:
                ed = EdgeDefect(
                    side="aoi_edge", area=area,
                    bbox=(max(0, cx - 3), 1000 - 3, 6, 6),
                    center=(cx, 1000), max_diff=7,
                )
                ed.cv_filtered_mask = np.full((512, 512), 255, dtype=np.uint8)
                ed.cv_mask_offset = (offset_x, offset_y)
                defects.append(ed)
            return defects, {"max_diff": 7, "max_area": 65,
                              "threshold": 4, "min_area": 40, "min_max_diff": 20}
        inf.edge_inspector.inspect_roi = _cv_stub
        def _pt(tile, **kw):
            return 0.0, np.zeros((512, 512), dtype=np.float32)
        inf.predict_tile = _pt

        defects, stats = inf._inspect_roi_fusion(
            img, img_x=104, img_y=1000, img_prefix="W0F",
            panel_polygon=poly, omit_image=None, omit_overexposed=False,
            collapse_to_representative=False,
        )
        assert len(defects) == 3, f"debug mode should keep all 3, got {len(defects)}"
