"""AOI 座標邊緣 PatchCore inspector 單元測試。

測試重點:
  - ROI 建構 (中心對齊 + 黑 pad)
  - fg_mask 由 panel_polygon 正確 rasterize
  - EdgeDefect 欄位 mapping (inspector_mode / patchcore_score / ...)
  - Stats dict 帶出 roi / fg_mask / anomaly_map 供 heatmap 渲染

這些測試 stub 掉 PatchCore 推論，不需要 GPU / 真實模型。
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer
from capi_edge_cv import EdgeDefect, EdgeInspectionConfig


@pytest.fixture
def inferencer():
    """建立最小可用的 CAPIInferencer，PatchCore 相關方法打樁"""
    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.anomaly_threshold = 1.0
    cfg.threshold_mapping = {}
    cfg.model_mapping = {}
    cfg.patchcore_min_area = 10
    cfg.patchcore_filter_enabled = False
    cfg.patchcore_concentration_enabled = False

    inf = CAPIInferencer(cfg, threshold=1.0)
    inf.edge_inspector.config.aoi_edge_inspector = "patchcore"
    return inf


def _stub_predict_tile(score: float, anomaly_peak: float = 0.0):
    """回傳一個 predict_tile 的替身：固定輸出 (score, anomaly_map)"""
    def _fn(tile, inferencer=None, threshold=None, **kwargs):
        if anomaly_peak > 0:
            amap = np.zeros((64, 64), dtype=np.float32)
            amap[20:30, 20:30] = anomaly_peak
        else:
            amap = np.zeros((64, 64), dtype=np.float32)
        return score, amap
    return _fn


def test_roi_center_aligned_and_black_padded(inferencer):
    """AOI 座標貼右邊界: ROI 右半部應被黑色 pad 到 512×512"""
    inferencer._get_inferencer_for_prefix = lambda _p: MagicMock()
    inferencer._get_threshold_for_prefix = lambda _p: 1.0
    inferencer.predict_tile = _stub_predict_tile(score=0.5)

    image = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    img_x, img_y = 1900, 540  # 貼右邊

    defects, stats = inferencer._inspect_roi_patchcore(
        image, img_x, img_y, "W0F00000", panel_polygon=None,
    )

    assert stats["roi"].shape == (512, 512, 3), "ROI 必須 pad 到 512×512"
    # 左半應有 image 內容 (128)，右半貼到 image 外的部分應為 0
    left_half = stats["roi"][:, :256]
    assert np.mean(left_half) > 100, "left 半部應保留原圖灰階"
    # 最右邊 100 px 應全為 0 (image 外 pad)
    pad_region = stats["roi"][:, -100:]
    assert np.mean(pad_region) == 0, f"pad 區應為黑色，實際 mean={np.mean(pad_region)}"


def test_fg_mask_from_polygon(inferencer):
    """panel_polygon 應正確 rasterize 到 ROI 局部座標"""
    inferencer._get_inferencer_for_prefix = lambda _p: MagicMock()
    inferencer._get_threshold_for_prefix = lambda _p: 1.0
    inferencer.predict_tile = _stub_predict_tile(score=0.5)

    image = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    img_x, img_y = 960, 540

    # 矩形 polygon 完全包住 ROI 中心
    polygon = np.array([[500, 300], [1400, 300], [1400, 800], [500, 800]], dtype=np.float32)

    _, stats = inferencer._inspect_roi_patchcore(
        image, img_x, img_y, "W0F00000", panel_polygon=polygon,
    )

    fg_mask = stats["fg_mask"]
    assert fg_mask.shape == (512, 512)
    # ROI 中心點必在 polygon 內
    assert fg_mask[256, 256] == 255
    # ROI 邊角 (超出 polygon 或 ROI 外) 應為 0
    assert fg_mask[0, 0] == 0 or fg_mask[-1, -1] == 0  # 至少一個角應在 polygon 外


def test_ng_defect_mapping(inferencer):
    """score >= threshold → 產生 EdgeDefect, 欄位正確填入"""
    inferencer._get_inferencer_for_prefix = lambda _p: MagicMock()
    inferencer._get_threshold_for_prefix = lambda _p: 1.0
    inferencer.predict_tile = _stub_predict_tile(score=1.5, anomaly_peak=2.0)

    image = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    img_x, img_y = 960, 540

    defects, stats = inferencer._inspect_roi_patchcore(
        image, img_x, img_y, "W0F00000", panel_polygon=None,
    )

    assert len(defects) == 1
    d = defects[0]
    assert d.inspector_mode == "patchcore"
    assert d.side == "aoi_edge"
    assert d.center == (img_x, img_y)
    assert abs(d.patchcore_score - 1.5) < 1e-6
    assert abs(d.patchcore_threshold - 1.0) < 1e-6
    assert d.patchcore_ok_reason == ""
    assert d.max_diff == 0  # PatchCore path 不用灰階差
    assert d.pc_roi is not None
    assert d.pc_fg_mask is not None
    assert d.pc_anomaly_map is not None


def test_ok_reason_score_below_threshold(inferencer):
    """score < threshold → OK with reason='Score<Thr'"""
    inferencer._get_inferencer_for_prefix = lambda _p: MagicMock()
    inferencer._get_threshold_for_prefix = lambda _p: 1.0
    inferencer.predict_tile = _stub_predict_tile(score=0.3, anomaly_peak=0.5)

    image = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    defects, stats = inferencer._inspect_roi_patchcore(
        image, 960, 540, "W0F00000", panel_polygon=None,
    )
    assert defects == []
    assert stats["ok_reason"] == "Score<Thr"


def test_roi_completely_out_of_image(inferencer):
    """AOI 座標完全超出影像 → 回傳 empty + reason"""
    inferencer._get_inferencer_for_prefix = lambda _p: MagicMock()
    inferencer._get_threshold_for_prefix = lambda _p: 1.0

    image = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    # 座標 (-1000, -1000) 加 tile/2=256 仍在 image 外
    defects, stats = inferencer._inspect_roi_patchcore(
        image, -1000, -1000, "W0F00000", panel_polygon=None,
    )
    assert defects == []
    assert "out of image" in stats["ok_reason"]


def test_no_inferencer_for_prefix(inferencer):
    """前綴找不到模型 → empty + reason"""
    inferencer._get_inferencer_for_prefix = lambda _p: None
    inferencer._get_threshold_for_prefix = lambda _p: 1.0

    image = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    defects, stats = inferencer._inspect_roi_patchcore(
        image, 960, 540, "UNKNOWN", panel_polygon=None,
    )
    assert defects == []
    assert "No model" in stats["ok_reason"]


def test_edge_defect_new_fields_default():
    """既有 EdgeDefect 建構不需要新欄位即可運作 (向下相容)"""
    d = EdgeDefect(side="left", area=100, bbox=(0, 0, 10, 10), center=(5, 5))
    assert d.inspector_mode == "cv"
    assert d.patchcore_score == 0.0
    assert d.patchcore_threshold == 0.0
    assert d.patchcore_ok_reason == ""


def test_inspection_config_reads_inspector_mode():
    """EdgeInspectionConfig.from_db_params 讀取 aoi_edge_inspector"""
    params = {
        "aoi_edge_inspector": "patchcore",
    }
    cfg = EdgeInspectionConfig.from_db_params(params)
    assert cfg.aoi_edge_inspector == "patchcore"

    # 預設 fallback
    cfg2 = EdgeInspectionConfig.from_db_params({})
    assert cfg2.aoi_edge_inspector == "cv"

    # 非法值 fallback
    cfg3 = EdgeInspectionConfig.from_db_params({"aoi_edge_inspector": "invalid"})
    assert cfg3.aoi_edge_inspector == "cv"
