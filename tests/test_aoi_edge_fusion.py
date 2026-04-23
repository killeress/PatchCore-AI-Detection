"""Phase 6 — AOI 邊緣 CV+PatchCore 空間分權 Fusion Inspector 單元測試。

涵蓋:
  - compute_boundary_band_mask 幾何 (Slice 2)
  - _inspect_roi_patchcore return_raw (Slice 3)
  - apply_omit_dust_filter (Slice 4)
  - _inspect_roi_fusion 主流程 (Slice 5)
"""

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

# Slice 2 — boundary band mask
from capi_edge_cv import compute_boundary_band_mask
from capi_config import CAPIConfig
from capi_inference import CAPIInferencer


# =========================================================================
# Slice 2: compute_boundary_band_mask 幾何
# =========================================================================

class TestBoundaryBandMask:
    """幾何性質：band = ROI 內離 polygon 邊 ≤ band_px 的 pixel 且在 fg_mask 內"""

    def test_deep_interior_returns_empty_band(self):
        """ROI 在 panel 中央 (polygon 邊不進入 ROI) → band 全空"""
        fg_mask = np.full((100, 100), 255, dtype=np.uint8)
        polygon = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
        # ROI 從 (450, 450) 開始，size 100x100，遠離所有 polygon 邊
        band = compute_boundary_band_mask(
            roi_shape=(100, 100),
            roi_origin=(450, 450),
            panel_polygon=polygon,
            band_px=40,
            fg_mask=fg_mask,
        )
        assert band.shape == (100, 100)
        assert band.dtype == np.uint8
        assert not band.any(), f"Expected empty band, got {int(band.sum())} pixels"

    def test_near_left_edge_band_along_polygon(self):
        """ROI 跨越 polygon 左邊 → band 為 polygon 內側 band_px 寬垂直帶"""
        # ROI origin (-100, 200) → ROI 內 polygon 左邊在 x=100 (ROI 座標系)
        fg_mask = np.zeros((200, 200), dtype=np.uint8)
        fg_mask[:, 100:] = 255  # 右半 (panel 內)
        polygon = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
        band = compute_boundary_band_mask(
            roi_shape=(200, 200),
            roi_origin=(-100, 200),
            panel_polygon=polygon,
            band_px=40,
            fg_mask=fg_mask,
        )
        assert band.shape == (200, 200)
        # 在 fg 內、距 polygon 邊 ≤ 40 → 為 band
        assert band[100, 105] > 0, "(100,105) 距 polygon 邊 5px、在 fg 內 → 應為 band"
        assert band[100, 139] > 0, "(100,139) 距 polygon 邊 39px、在 fg 內 → 應為 band 邊界"
        # 距 polygon 邊 > 40 → 不在 band
        assert band[100, 145] == 0, "(100,145) 距 polygon 邊 45px → 不應在 band"
        # 在 fg 外 → 不在 band (band 限定 fg_mask 內)
        assert band[100, 50] == 0, "(100,50) 在 fg 外 (panel 外) → 不應在 band"

    def test_on_edge_band_covers_most_of_fg(self):
        """AOI 貼 polygon 邊 → band 覆蓋 ROI 內多數 fg pixel"""
        fg_mask = np.zeros((100, 100), dtype=np.uint8)
        fg_mask[:, 50:] = 255  # 右半 fg, 50x100=5000 px
        polygon = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
        # ROI origin (-50, 0) → polygon 左邊在 ROI x=50
        band = compute_boundary_band_mask(
            roi_shape=(100, 100),
            roi_origin=(-50, 0),
            panel_polygon=polygon,
            band_px=40,
            fg_mask=fg_mask,
        )
        # band 應為 x=50 到 x=89 的垂直帶 (40 px 寬，限定在 fg 內)
        # 面積約 100 * 40 = 4000 px
        assert band.sum() > 100 * 40 * 255 * 0.7, f"Band 面積太小: {int(band.sum() // 255)} px"
        # 內側 (x=95) 不在 band
        assert band[50, 95] == 0, "(50,95) 距 polygon 邊 45px > band_px=40 → 不應在 band"
        # band 邊緣
        assert band[50, 55] > 0, "(50,55) 在 band 內 → 應有值"

    def test_band_px_zero_returns_empty(self):
        """band_px=0 → band 全空 (語意：等同 patchcore，CV 管轄區歸零)"""
        fg_mask = np.full((100, 100), 255, dtype=np.uint8)
        polygon = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
        band = compute_boundary_band_mask(
            roi_shape=(100, 100),
            roi_origin=(-10, 0),
            panel_polygon=polygon,
            band_px=0,
            fg_mask=fg_mask,
        )
        assert not band.any(), f"band_px=0 應回空 band，實得 {int(band.sum())} px"

    def test_polygon_none_returns_empty(self):
        """panel_polygon=None → 回空 band (caller 自行 fallback CV only)"""
        fg_mask = np.full((100, 100), 255, dtype=np.uint8)
        band = compute_boundary_band_mask(
            roi_shape=(100, 100),
            roi_origin=(0, 0),
            panel_polygon=None,
            band_px=40,
            fg_mask=fg_mask,
        )
        assert not band.any()


# =========================================================================
# Slice 3: _inspect_roi_patchcore return_raw 參數
# =========================================================================

@pytest.fixture
def pc_inferencer():
    """最小可用 CAPIInferencer，PatchCore 推論打樁，回固定 anomaly_map"""
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

    def _fake_predict(tile, **kwargs):
        # score=0.5 (< thr=1.0 → OK), anomaly_map 中央有熱區
        amap = np.zeros((512, 512), dtype=np.float32)
        amap[200:220, 200:220] = 0.8
        return 0.5, amap

    inf.predict_tile = _fake_predict
    return inf


class TestInspectRoiPatchcoreReturnRaw:
    """Phase 6 — return_raw 參數新增；不影響既有 Phase 5 行為"""

    def test_return_raw_true_skips_post_processing(self, pc_inferencer):
        """return_raw=True → 不抽 defect、stats 內含 anomaly_map 供 fusion 後處理"""
        image = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        polygon = np.array([[100, 100], [1800, 100], [1800, 1000], [100, 1000]], dtype=np.float32)

        defects, stats = pc_inferencer._inspect_roi_patchcore(
            image, img_x=960, img_y=540, img_prefix="W0F",
            panel_polygon=polygon, return_raw=True,
        )
        # return_raw=True → 不產 defect (即使 NG 也不抽)
        assert defects == []
        # stats 必有 anomaly_map / threshold / fg_mask / roi
        assert "anomaly_map" in stats and stats["anomaly_map"] is not None
        assert "threshold" in stats and stats["threshold"] == 1.0
        assert "fg_mask" in stats and stats["fg_mask"] is not None
        assert "roi" in stats and stats["roi"] is not None
        # anomaly_map 應該是 predict_tile 的輸出 (中央有熱區)
        assert stats["anomaly_map"].shape == (512, 512)
        assert float(stats["anomaly_map"].max()) > 0.7

    def test_return_raw_false_unchanged_behavior(self, pc_inferencer):
        """return_raw=False (預設) → 既有行為不變 (Phase 5 8 項 regression 由其他測試 cover)"""
        image = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        polygon = np.array([[100, 100], [1800, 100], [1800, 1000], [100, 1000]], dtype=np.float32)

        # 預設 return_raw=False
        defects, stats = pc_inferencer._inspect_roi_patchcore(
            image, img_x=960, img_y=540, img_prefix="W0F",
            panel_polygon=polygon,
        )
        # score=0.5 < thr=1.0 → OK，不出 defect
        assert defects == []
        # OK reason 應有
        assert "ok_reason" in stats
        assert stats["ok_reason"] == "Score<Thr"


# =========================================================================
# Slice 4: apply_omit_dust_filter — fusion 後 OMIT 屏蔽
# =========================================================================

from capi_edge_cv import EdgeDefect


def _make_pc_defect(bbox=(100, 100, 50, 50), center=(125, 125), score=0.8):
    d = EdgeDefect(
        side="aoi_edge", area=200, bbox=bbox, center=center,
        max_diff=0, inspector_mode="fusion",
        patchcore_score=score, patchcore_threshold=1.0,
    )
    d.source_inspector = "patchcore"
    # 模擬 PC anomaly_map (50x50 內中央有熱點)
    amap = np.zeros((50, 50), dtype=np.float32)
    amap[20:30, 20:30] = 0.9
    d.pc_anomaly_map = amap
    return d


def _make_cv_defect(bbox=(200, 200, 30, 30), center=(215, 215), max_diff=25):
    d = EdgeDefect(
        side="aoi_edge", area=100, bbox=bbox, center=center,
        max_diff=max_diff, inspector_mode="fusion",
    )
    d.source_inspector = "cv"
    # 模擬 CV filtered mask (30x30 全活躍)
    d.cv_filtered_mask = np.full((30, 30), 255, dtype=np.uint8)
    d.cv_mask_offset = (200, 200)
    return d


class TestApplyOmitDustFilter:
    """fusion 後統一 OMIT 屏蔽 — 不分 source 一套邏輯"""

    def test_omit_none_keeps_all_defects_unchanged(self, pc_inferencer):
        """omit_image=None (OMIT 缺失) → defect 全保留、不做 dust check"""
        defects = [_make_pc_defect(), _make_cv_defect()]

        result = pc_inferencer._apply_omit_dust_filter_to_edge_defects(
            defects, omit_image=None, omit_overexposed=False,
        )

        assert len(result) == 2
        for d in result:
            assert d.is_suspected_dust_or_scratch is False
            assert d.dust_detail_text == ""

    def test_omit_overexposed_keeps_all_with_detail_text(self, pc_inferencer):
        """OMIT 過曝 → defect 全保留 + dust_detail_text 提示，不判 dust"""
        omit = np.full((1080, 1920), 200, dtype=np.uint8)  # arbitrary, not used when overexposed
        defects = [_make_pc_defect(), _make_cv_defect()]

        result = pc_inferencer._apply_omit_dust_filter_to_edge_defects(
            defects, omit_image=omit, omit_overexposed=True,
        )

        assert len(result) == 2
        for d in result:
            assert d.is_suspected_dust_or_scratch is False
            assert "OMIT_OVEREXPOSED" in d.dust_detail_text

    def test_dust_hit_marks_is_suspected_dust(self, pc_inferencer):
        """OMIT 命中 dust + 與 defect 熱區重疊高 → is_suspected_dust_or_scratch=True"""
        omit = np.full((1080, 1920), 50, dtype=np.uint8)  # 暗背景
        # 在 PC defect bbox (100,100,50,50) 內製造亮點 (對應 PC 熱區位置)
        omit[120:130, 120:130] = 230  # ~對應 anomaly_map[20:30, 20:30] 區
        defects = [_make_pc_defect()]

        # stub check_dust_or_scratch_feature 強制回 dust 命中
        dust_mask = np.zeros((50, 50), dtype=np.uint8)
        dust_mask[20:30, 20:30] = 255  # 與 anomaly_map 熱區重疊
        pc_inferencer.check_dust_or_scratch_feature = lambda img, **kwargs: (
            True, dust_mask, 0.04, "Dust detected (synthetic)"
        )

        result = pc_inferencer._apply_omit_dust_filter_to_edge_defects(
            defects, omit_image=omit, omit_overexposed=False,
        )

        assert len(result) == 1
        assert result[0].is_suspected_dust_or_scratch is True, \
            f"應判 dust 屏蔽; detail={result[0].dust_detail_text}, ratio={result[0].dust_bright_ratio}"
        assert "Dust detected" in result[0].dust_detail_text

    def test_no_dust_keeps_defects_clean(self, pc_inferencer):
        """OMIT 影像下無 dust 特徵 → defect 不標 is_suspected"""
        omit = np.full((1080, 1920), 50, dtype=np.uint8)
        defects = [_make_pc_defect()]

        # stub 回 False
        pc_inferencer.check_dust_or_scratch_feature = lambda img, **kwargs: (
            False, None, 0.0, "Clean"
        )

        result = pc_inferencer._apply_omit_dust_filter_to_edge_defects(
            defects, omit_image=omit, omit_overexposed=False,
        )

        assert len(result) == 1
        assert result[0].is_suspected_dust_or_scratch is False
        assert result[0].dust_detail_text == "Clean"


# =========================================================================
# Slice 5: _inspect_roi_fusion 主方法
# =========================================================================

def _stub_predict_tile(score: float, hot_region: Optional[tuple] = None, hot_value: float = 0.0):
    """生成 predict_tile 替身。hot_region=(y1,y2,x1,x2) 在 anomaly_map 中設熱區。"""
    def _fn(tile, **kwargs):
        amap = np.zeros((512, 512), dtype=np.float32)
        if hot_region is not None and hot_value > 0:
            y1, y2, x1, x2 = hot_region
            amap[y1:y2, x1:x2] = hot_value
        return score, amap
    return _fn


class TestInspectRoiFusion:
    """Phase 6 — fusion 空間分權主流程"""

    @pytest.fixture
    def fusion_inferencer(self):
        """fusion 模式 inferencer，CV inspector 與 PC 推論可分別 stub"""
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
        # Phase 7.1c: 此 class 專測 Phase 6 band_mask，鎖 shift band=40 重現 pre-7.1c
        # 行為（近邊 shift verify fail → fallback centered + band_mask 生效）。
        inf.edge_inspector.config.aoi_edge_pc_shift_band_px = 40

        # default: OMIT 不做 dust check
        inf.check_dust_or_scratch_feature = lambda img, **kwargs: (False, None, 0.0, "")

        return inf

    def _build_test_image_and_polygon(self):
        """建立測試影像 (1080x1920) 與 polygon (內縮 100 px 的矩形)，AOI 座標 (160, 540) 貼左邊"""
        image = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        polygon = np.array(
            [[100, 100], [1820, 100], [1820, 980], [100, 980]],
            dtype=np.float32,
        )
        return image, polygon

    def test_polygon_none_falls_back_to_cv_only(self, fusion_inferencer):
        """polygon=None → fusion fallback CV only，defects 標 fusion_fallback_reason"""
        inf = fusion_inferencer
        image = np.full((1080, 1920, 3), 128, dtype=np.uint8)

        # CV inspector stub：回 1 個 defect (在 ROI 內任意位置)
        def _cv_stub(roi, offset_x, offset_y, **kwargs):
            ed = EdgeDefect(
                side="aoi_edge", area=50,
                bbox=(offset_x + 100, offset_y + 100, 30, 30),
                center=(offset_x + 115, offset_y + 115),
                max_diff=30,
            )
            ed.cv_filtered_mask = np.full((512, 512), 255, dtype=np.uint8)
            ed.cv_mask_offset = (offset_x, offset_y)
            return [ed], {"max_diff": 30, "max_area": 50, "threshold": 4,
                          "min_area": 40, "min_max_diff": 20}
        inf.edge_inspector.inspect_roi = _cv_stub
        inf.predict_tile = _stub_predict_tile(score=0.0)  # PC clean

        defects, stats = inf._inspect_roi_fusion(
            image, img_x=160, img_y=540, img_prefix="W0F",
            panel_polygon=None,  # ← fallback trigger
            omit_image=None, omit_overexposed=False,
        )

        assert len(defects) == 1
        assert defects[0].source_inspector == "cv"
        assert defects[0].fusion_fallback_reason == "polygon_unavailable"
        assert stats.get("fusion_fallback_reason") == "polygon_unavailable"

    def test_cv_defect_in_band_kept(self, fusion_inferencer):
        """CV 在 band 內找到 defect → 保留，source_inspector='cv'"""
        inf = fusion_inferencer
        image, polygon = self._build_test_image_and_polygon()

        # AOI(160, 540) → ROI [160-256, 540-256] = [-96, 284] to [416, 796]
        # polygon 左邊在 panel x=100，ROI 內 x=100-(-96)=196
        # band_px=40 → CV 管 ROI x∈[156, 236]
        # CV defect center at panel (220, 540) → ROI (316, 256) → 在 band 內 (256>236? no, 256 在 band 外)
        # Actually 220 in panel = ROI x = 220 - (-96) = 316; band is around polygon edge ROI x=196 ± 40 = [156, 236]
        # 316 > 236 → outside band
        # 我要 CV defect 在 band 內：center panel x=180 → ROI x=180-(-96)=276... still outside
        # 試 panel x=140 → ROI x=140-(-96)=236; just on band edge
        # 試 panel x=130 → ROI x=130-(-96)=226 → 在 band 內 ✓ (band x∈[156, 236])
        cv_def_panel_center = (130, 540)

        def _cv_stub(roi, offset_x, offset_y, **kwargs):
            ed = EdgeDefect(
                side="aoi_edge", area=50,
                bbox=(cv_def_panel_center[0] - 5, cv_def_panel_center[1] - 5, 10, 10),
                center=cv_def_panel_center,
                max_diff=30,
            )
            ed.cv_filtered_mask = np.full((512, 512), 255, dtype=np.uint8)
            ed.cv_mask_offset = (offset_x, offset_y)
            return [ed], {"max_diff": 30, "max_area": 50, "threshold": 4,
                          "min_area": 40, "min_max_diff": 20}
        inf.edge_inspector.inspect_roi = _cv_stub
        inf.predict_tile = _stub_predict_tile(score=0.0)

        defects, stats = inf._inspect_roi_fusion(
            image, img_x=160, img_y=540, img_prefix="W0F",
            panel_polygon=polygon,
            omit_image=None, omit_overexposed=False,
        )

        cv_kept = [d for d in defects if d.source_inspector == "cv"]
        assert len(cv_kept) == 1, \
            f"CV defect 在 band 內應保留，實得 {len(cv_kept)} 個 cv defect (total {len(defects)})"

    def test_cv_defect_outside_band_dropped(self, fusion_inferencer):
        """CV 在 interior 找到 defect → 空間過濾丟棄"""
        inf = fusion_inferencer
        image, polygon = self._build_test_image_and_polygon()

        # CV defect center at panel (300, 540) → ROI (396, 256) → 遠在 band 外
        cv_def_panel_center = (300, 540)

        def _cv_stub(roi, offset_x, offset_y, **kwargs):
            ed = EdgeDefect(
                side="aoi_edge", area=50,
                bbox=(cv_def_panel_center[0] - 5, cv_def_panel_center[1] - 5, 10, 10),
                center=cv_def_panel_center,
                max_diff=30,
            )
            ed.cv_filtered_mask = np.full((512, 512), 255, dtype=np.uint8)
            ed.cv_mask_offset = (offset_x, offset_y)
            return [ed], {"max_diff": 30, "max_area": 50, "threshold": 4,
                          "min_area": 40, "min_max_diff": 20}
        inf.edge_inspector.inspect_roi = _cv_stub
        inf.predict_tile = _stub_predict_tile(score=0.0)

        defects, stats = inf._inspect_roi_fusion(
            image, img_x=160, img_y=540, img_prefix="W0F",
            panel_polygon=polygon,
            omit_image=None, omit_overexposed=False,
        )

        cv_kept = [d for d in defects if d.source_inspector == "cv"]
        assert len(cv_kept) == 0, \
            f"CV defect 在 band 外應丟棄，實得 {len(cv_kept)} 個 cv defect"

    def test_pc_anomaly_in_interior_creates_pc_defect(self, fusion_inferencer):
        """PC interior anomaly 高 → 產 PC defect，source_inspector='patchcore'"""
        inf = fusion_inferencer
        image, polygon = self._build_test_image_and_polygon()

        # PC anomaly hot region 在 interior（ROI 中心區），不在 band 內
        # ROI x=256 (中心) 在 band ROI x∈[156,236] 之外 → interior
        inf.edge_inspector.inspect_roi = lambda roi, **kw: ([], {"max_diff": 0, "max_area": 0,
                                                                   "threshold": 4, "min_area": 40,
                                                                   "min_max_diff": 20})
        inf.predict_tile = _stub_predict_tile(score=2.0, hot_region=(250, 270, 250, 270),
                                                hot_value=2.0)  # > thr=1.0

        defects, stats = inf._inspect_roi_fusion(
            image, img_x=160, img_y=540, img_prefix="W0F",
            panel_polygon=polygon,
            omit_image=None, omit_overexposed=False,
        )

        pc_kept = [d for d in defects if d.source_inspector == "patchcore"]
        assert len(pc_kept) == 1, \
            f"PC interior anomaly 應產 PC defect，實得 {len(pc_kept)} 個 pc defect"
        assert pc_kept[0].patchcore_score >= 1.0

    def test_pc_anomaly_in_band_dropped(self, fusion_inferencer):
        """PC anomaly 全部在 band 內 (受感受野污染假陽) → masked 掉、不產 PC defect"""
        inf = fusion_inferencer
        image, polygon = self._build_test_image_and_polygon()

        # PC hot region 落在 band 內 (ROI x=200 周邊)
        # ROI x∈[156, 236] 是 band → 設 hot region (250, 270, 190, 220) 部分在 band 內
        # 為了確保 hot region 完全在 band 內：x in [180, 220] 都應在 band [156, 236] 內
        inf.edge_inspector.inspect_roi = lambda roi, **kw: ([], {"max_diff": 0, "max_area": 0,
                                                                   "threshold": 4, "min_area": 40,
                                                                   "min_max_diff": 20})
        inf.predict_tile = _stub_predict_tile(score=2.0, hot_region=(250, 270, 180, 220),
                                                hot_value=2.0)

        defects, stats = inf._inspect_roi_fusion(
            image, img_x=160, img_y=540, img_prefix="W0F",
            panel_polygon=polygon,
            omit_image=None, omit_overexposed=False,
        )

        pc_kept = [d for d in defects if d.source_inspector == "patchcore"]
        assert len(pc_kept) == 0, \
            f"PC anomaly 在 band 內應被 mask 掉、不產 defect，實得 {len(pc_kept)} 個 pc defect"

    def test_deep_interior_runs_fusion_no_skip(self, fusion_inferencer):
        """深處 AOI 座標 (ROI 內無 polygon edge) → fusion 仍跑完整流程，band 空、PC only 行為"""
        inf = fusion_inferencer
        image, polygon = self._build_test_image_and_polygon()

        # AOI (960, 540) → ROI 完全在 panel 內 [704, 284] to [1216, 796]
        # polygon 邊不進入 ROI → band 空 → CV defect 全丟、PC 管全部
        cv_called = {"flag": False}

        def _cv_stub(roi, **kw):
            cv_called["flag"] = True
            ed = EdgeDefect(side="aoi_edge", area=50, bbox=(800, 540, 10, 10),
                             center=(810, 545), max_diff=30)
            ed.cv_filtered_mask = np.full((512, 512), 255, dtype=np.uint8)
            return [ed], {"max_diff": 30, "max_area": 50, "threshold": 4,
                          "min_area": 40, "min_max_diff": 20}
        inf.edge_inspector.inspect_roi = _cv_stub

        inf.predict_tile = _stub_predict_tile(score=2.0, hot_region=(250, 270, 250, 270),
                                                hot_value=2.0)

        defects, stats = inf._inspect_roi_fusion(
            image, img_x=960, img_y=540, img_prefix="W0F",
            panel_polygon=polygon,
            omit_image=None, omit_overexposed=False,
        )

        # 驗證 CV 仍被呼叫（流程一致原則）
        assert cv_called["flag"] is True, "deep interior 仍應呼叫 CV inspector (統一 pipeline)"
        # CV defect 應被 band mask 過濾掉 (band 空)
        cv_kept = [d for d in defects if d.source_inspector == "cv"]
        assert len(cv_kept) == 0
        # PC 管 interior 全部 → PC defect 應產
        pc_kept = [d for d in defects if d.source_inspector == "patchcore"]
        assert len(pc_kept) == 1
