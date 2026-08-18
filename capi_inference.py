"""
CAPI 推論核心模組

提供 CAPI 面板異常檢測的完整推論流程：
1. 圖片載入與 Otsu 去背景
2. 排除區域識別（MARK、右下角機構）
3. 512x512 切塊與座標追蹤
4. PatchCore 模型推論
5. 異常結果匯總與座標轉換

使用方式:
    from capi_inference import CAPIInferencer
    from capi_config import CAPIConfig, BombDefect
    
    config = CAPIConfig.from_yaml("configs/capi_3f.yaml")
    inferencer = CAPIInferencer(config, model_path="path/to/model")
    results = inferencer.process_panel("path/to/panel_folder")
"""

import os
# 設置環境變數以允許載入模型 (必須在 import anomalib 之前)
os.environ["TRUST_REMOTE_CODE"] = "1"
# 抑制 anomalib 棄用警告 (TorchInferencer legacy / TRUST_REMOTE_CODE)
import logging as _logging
_logging.getLogger("anomalib.deploy.inferencers.torch_inferencer").setLevel(_logging.ERROR)
# 抑制 dinov2 載入時 xFormers 不可用的 UserWarning
import warnings as _warnings
_warnings.filterwarnings("ignore", message=r".*xFormers is not available.*")

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import re
import time
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import logging
import re
from capi_image_naming import AOI_REPORT_PREFIXES, canonical_image_prefix, panel_image_group_key
from capi_image_orientation import read_detection_image

# ── 舊版 anomalib 相容性修補 ─────────────────────────────
# 修補 1: PrecisionType stub
# 舊版 anomalib checkpoint 裡 pickle 序列化了 anomalib.PrecisionType enum，
# 新版已移除 → torch.load() AttributeError。注入 stub 讓反序列化成功。
import anomalib as _anomalib
if not hasattr(_anomalib, "PrecisionType"):
    import enum as _enum
    class _PrecisionType(str, _enum.Enum):
        FLOAT16 = "float16"
        FLOAT32 = "float32"
        BFLOAT16 = "bfloat16"
    _anomalib.PrecisionType = _PrecisionType
    logging.getLogger("capi.inference").info(
        "Injected PrecisionType stub into anomalib for legacy checkpoint compat"
    )

# 修補 2: TorchInferencer.predict → 強制 float32 輸入
# 舊版 checkpoint 序列化了 PrecisionType.FLOAT16，新版 anomalib 的
# TorchInferencer.predict() 會將輸入轉 fp16，但 backbone 權重仍為 float32
# → RuntimeError: Input type (HalfTensor) and weight type (FloatTensor) mismatch
# 解法: 包裝 model.forward，在 forward 前強制輸入轉 float32
try:
    from anomalib.deploy import TorchInferencer as _TI
    _orig_predict = _TI.predict

    def _fp32_predict(self, *args, **kwargs):
        _orig_fwd = self.model.forward
        def _force_fp32_fwd(batch, *a, **kw):
            # anomalib 使用 InferenceBatch dataclass，圖片在 .image 屬性
            if hasattr(batch, 'image') and isinstance(batch.image, torch.Tensor):
                batch.image = batch.image.float()
            elif isinstance(batch, torch.Tensor):
                batch = batch.float()
            return _orig_fwd(batch, *a, **kw)
        self.model.forward = _force_fp32_fwd
        try:
            return _orig_predict(self, *args, **kwargs)
        finally:
            self.model.forward = _orig_fwd

    _TI.predict = _fp32_predict
    logging.getLogger("capi.inference").info(
        "Patched TorchInferencer.predict → force float32 input"
    )
except Exception as _e:
    logging.getLogger("capi.inference").warning(
        "Failed to patch TorchInferencer.predict: %s", _e
    )

from capi_config import CAPIConfig, ExclusionZone, BombDefect
from capi_edge_cv import (
    CVEdgeInspector, EdgeInspectionConfig, EdgeDefect, clamp_median_kernel,
    compute_fg_aware_diff, compute_boundary_band_mask, compute_pc_roi_offset,
    verify_polygon_clear_of_pc_roi, classify_pc_roi_verify_failure,
)
from capi_preprocess import (
    PreprocessConfig,
    detect_panel_polygon,
    _polyfit_polygon as _pf_polygon,
    is_small_product_resolution,
    rect_polygon_from_bounds,
    resolve_aoi_inward_shift_axes,
    map_product_coord_to_image,
)
from scratch_classifier import ScratchClassifier, ScratchClassifierLoadError
from scratch_filter import ScratchFilter

logger = logging.getLogger("capi.inference")
_MARK_PATCH_SCORE_LOCK = threading.RLock()


# ── 產品解析度映射 ─────────────────────────────────────
# 機種名稱第六碼 → 產品解析度 (寬, 高)
# 此為程式碼級預設值，正式使用時由 capi_config.yaml 中的 model_resolution_map 覆蓋
MODEL_RESOLUTION_MAP = {
    'B': (1366, 768),
    'H': (1920, 1080),
    'J': (1920, 1200),
    'K': (2560, 1440),
    'G': (2560, 1600),
}

DEFAULT_PRODUCT_RESOLUTION = (1920, 1080)


def score_normalization_diagnostic(
    raw_score: Optional[float],
    image_min: Optional[float],
    image_max: Optional[float],
    image_threshold: Optional[float],
    normalization_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    """Explain Anomalib's clamp-to-zero image-score normalization boundary."""
    if normalization_enabled is False:
        return {
            "normalization_available": False,
            "normalization_zero_boundary": None,
            "normalization_zero_clamped": False,
        }
    values = (raw_score, image_min, image_max)
    if any(value is None or not np.isfinite(float(value)) for value in values):
        return {
            "normalization_available": False,
            "normalization_zero_boundary": None,
            "normalization_zero_clamped": None,
        }
    minimum = float(image_min)
    maximum = float(image_max)
    if maximum <= minimum:
        return {
            "normalization_available": False,
            "normalization_zero_boundary": None,
            "normalization_zero_clamped": None,
        }
    threshold = (
        float(image_threshold)
        if image_threshold is not None and np.isfinite(float(image_threshold))
        else (minimum + maximum) / 2.0
    )
    zero_boundary = threshold - (maximum - minimum) * 0.5
    raw = float(raw_score)
    return {
        "normalization_available": True,
        "normalization_zero_boundary": zero_boundary,
        "normalization_zero_clamped": bool(raw <= zero_boundary + 1e-9),
    }


def _anomaly_max_cc_area(anomaly_map: Optional[np.ndarray], peak_value: Optional[float] = None) -> int:
    """以 peak×0.5 二值化後取最大連通面積。peak_value 省略則用 anomaly_map 自身的 max。"""
    if anomaly_map is None or anomaly_map.size == 0:
        return 0
    peak = float(peak_value) if peak_value is not None else float(np.max(anomaly_map))
    if peak <= 0:
        return 0
    binary = (anomaly_map > (peak * 0.5)).astype(np.uint8) * 255
    num_labels, _, cc_stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels <= 1:
        return 0
    return int(cc_stats[1:, cv2.CC_STAT_AREA].max())


def resolve_product_resolution(model_id: str, resolution_map: Optional[Dict] = None) -> Tuple[int, int]:
    """
    依機種名稱第六碼推導產品解析度

    例如 "GN140JCAL010S" 第六碼 'J' → (1920, 1200)

    Args:
        model_id: 機種 ID (例如 "GN140JCAL010S")
        resolution_map: 可選的映射表 (來自 config.model_resolution_map)，
                        若為 None 則使用模組級預設 MODEL_RESOLUTION_MAP
    """
    if resolution_map is None:
        resolution_map = MODEL_RESOLUTION_MAP
    if model_id and len(model_id) >= 6:
        code = model_id[5].upper()
        res = resolution_map.get(code, None)
        if res is not None:
            return (int(res[0]), int(res[1])) if isinstance(res, (list, tuple)) else DEFAULT_PRODUCT_RESOLUTION
    return DEFAULT_PRODUCT_RESOLUTION


@dataclass
class TileInfo:
    """切塊資訊"""
    tile_id: int
    x: int  # 切塊在原圖的 x 座標
    y: int  # 切塊在原圖的 y 座標
    width: int
    height: int
    image: np.ndarray = field(repr=False)
    original_image: Optional[np.ndarray] = field(default=None, repr=False)
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # 遮罩: 255=panel 內, 0=panel 外 (tile 完全在 polygon 內時為 None)
    has_exclusion: bool = False  # 是否包含排除區域
    is_bottom_edge: bool = False # 是否為底部邊緣切塊
    is_top_edge: bool = False    # 是否為頂部邊緣切塊
    is_left_edge: bool = False   # 是否為左側邊緣切塊
    is_right_edge: bool = False  # 是否為右側邊緣切塊
    is_suspected_dust_or_scratch: bool = False  # 是否疑似灰塵或刮痕 (透過 OMIT0000 檢查)
    omit_crop_image: Optional[np.ndarray] = field(default=None, repr=False)  # OMIT 圖片的對應裁切 (用於灰塵檢查)
    dust_mask: Optional[np.ndarray] = field(default=None, repr=False)
    dust_heatmap_iou: float = 0.0  # overall coverage (intersection / total dust area)
    dust_region_max_cov: float = 0.0  # per-region max coverage (用於實際判定)
    dust_region_details: Optional[list] = field(default=None, repr=False)  # per-region 判定詳情
    dust_heatmap_binary: Optional[np.ndarray] = field(default=None, repr=False)  # 二值化 heatmap
    dust_bright_ratio: float = 0.0
    dust_detail_text: str = ""  # 灰塵判定詳細資訊
    dust_iou_debug_image: Optional[np.ndarray] = field(default=None, repr=False)  # IOU debug 可視化圖
    dust_two_stage_features: Optional[list] = field(default=None, repr=False)  # 兩階段特徵點列表
    dust_two_stage_dust_mask: Optional[np.ndarray] = field(default=None, repr=False)  # two-stage 實際使用的 dust mask
    is_bomb: bool = False       # 是否為炸彈系統模擬缺陷
    bomb_defect_code: str = ""  # 匹配到的炸彈 Defect Code
    is_in_exclude_zone: bool = False  # 是否位於不檢測排除區域內
    anomaly_peak_x: int = -1    # 熱力圖峰值 x (圖片座標, -1=未計算)
    anomaly_peak_y: int = -1    # 熱力圖峰值 y (圖片座標, -1=未計算)
    is_aoi_coord_tile: bool = False  # 是否來自 AOI 機檢座標
    aoi_defect_code: str = ""        # AOI 異常代碼 (PCDK2, C1111, PTMD6)
    aoi_product_x: int = -1         # AOI 產品座標 X (-1=非 AOI 座標 tile)
    aoi_product_y: int = -1         # AOI 產品座標 Y (-1=非 AOI 座標 tile)
    aoi_image_x: int = -1           # AOI 映射後圖片座標 X (-1=非 AOI 座標 tile)
    aoi_image_y: int = -1           # AOI 映射後圖片座標 Y (-1=非 AOI 座標 tile)
    aoi_tile_shift_dx: int = 0      # AOI tile 最終左上角相對「AOI 置中」左上角的修正量
    aoi_tile_shift_dy: int = 0
    zone: str = ""                  # 新架構推論 zone："inner" / "edge" / "bright_spot"；v1 為 ""
    is_bright_spot_detection: bool = False  # 是否為二值化亮點偵測（非 PatchCore）
    bright_spot_max_diff: int = 0           # B0F 偵測：最大局部差異值
    bright_spot_diff_threshold: int = 0     # B0F 偵測：使用的差異閾值
    bright_spot_area: int = 0               # B0F 偵測：偵測到的亮點面積 (px)
    bright_spot_min_area: int = 0           # B0F 偵測：使用的最小面積
    score_threshold: Optional[float] = None # 此 tile 推論時實際使用的門檻（v2 依 zone 不同）
    raw_pred_score: float = 0.0             # 模型 normalized pred_score，未經 mask/edge margin 比率調整
    raw_model_score: Optional[float] = None # 未經 Anomalib image-score normalization 的模型距離
    model_image_min: Optional[float] = None # Anomalib image-score normalization 下界
    model_image_max: Optional[float] = None # Anomalib image-score normalization 上界
    model_image_threshold: Optional[float] = None # Anomalib image-score normalization threshold
    model_normalization_enabled: Optional[bool] = None # 本次模型是否啟用 Anomalib normalization
    raw_anomaly_map_max: Optional[float] = None # 未正規化 anomaly map 最高值
    normalized_anomaly_map_max: Optional[float] = None # 正規化 anomaly map 最高值
    pre_decay_map_max: float = 0.0          # mask/edge margin 前 anomaly_map max
    post_decay_map_max: float = 0.0         # mask/edge margin 後 anomaly_map max
    score_decay_ratio: float = 1.0          # post_decay_map_max / pre_decay_map_max
    score_edge_margin_sides: str = ""       # 此 tile 實際套用的 edge margin sides
    score_mask_valid_ratio: float = 1.0     # tile.mask 中有效區域比例
    mark_exclusion_regions: List[Any] = field(default_factory=list, repr=False)  # MARK binary 不檢測區域
    mark_exclusion_masked: bool = False      # 此 tile 的 heatmap 是否被 MARK binary 區域遮罩
    mark_exclusion_region_count: int = 0
    mark_patch_score_applied: bool = False   # 是否排除 MARK Patch 後重算正式分數
    mark_patchcore_score: float = 0.0        # 排除 MARK 後、其他遮罩前的正式分數
    mark_patch_valid_count: int = 0
    mark_patch_total_count: int = 0
    mark_patch_peak_x: int = -1
    mark_patch_peak_y: int = -1
    mark_patch_score_reason: str = ""
    # Scratch classifier post-filter (over-review reduction)
    scratch_score: float = 0.0              # 0 = 未跑 classifier
    scratch_filtered: bool = False          # True = 被翻回 OK

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def valid_ratio(self) -> float:
        """有效區域比例 (0.0~1.0)"""
        if self.mask is None:
            return 1.0
        return np.sum(self.mask > 0) / self.mask.size


@dataclass
class ExclusionRegion:
    """實際排除區域（計算後的座標）"""
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    def contains_point(self, x: int, y: int) -> bool:
        """檢查點是否在排除區域內"""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def overlaps_rect(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """檢查矩形是否與排除區域重疊"""
        return not (x2 < self.x1 or x1 > self.x2 or y2 < self.y1 or y1 > self.y2)
    
    def overlap_ratio(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """計算矩形與排除區域的重疊比例"""
        # 計算交集
        ix1 = max(self.x1, x1)
        iy1 = max(self.y1, y1)
        ix2 = min(self.x2, x2)
        iy2 = min(self.y2, y2)
        
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        tile_area = (x2 - x1) * (y2 - y1)
        
        return intersection / tile_area if tile_area > 0 else 0.0


@dataclass
class AOIDefect:
    """AOI 缺陷資訊"""
    defect_code: str
    product_x: int
    product_y: int
    image_x: int
    image_y: int
    bounds: Tuple[int, int, int, int]  # (x1, y1, x2, y2) 標記框

@dataclass
class AOIReportDefect:
    """AOI 機台 NG 報告缺陷 (解析自 Report TXT)"""
    defect_code: str      # 異常代碼 (PCDK2, C1111, PTMD6)
    product_x: int        # 產品座標 X
    product_y: int        # 產品座標 Y
    image_prefix: str     # 圖片前綴 (W0F00000, B0F00000)

@dataclass
class ImageResult:
    """單張圖片推論結果"""
    image_path: Path
    image_size: Tuple[int, int]  # (width, height)
    otsu_bounds: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    exclusion_regions: List[ExclusionRegion]
    tiles: List[TileInfo]
    excluded_tile_count: int
    processed_tile_count: int
    processing_time: float
    
    # 推論結果（由 PatchCore 填入）
    # (TileInfo, score, anomaly_map)
    anomaly_tiles: List[Tuple[TileInfo, float, Optional[np.ndarray]]] = field(default_factory=list)
    
    # AOI 缺陷結果
    aoi_defects: List[AOIDefect] = field(default_factory=list)
    
    # 裁切區域 (x1, y1, x2, y2) - 用於視覺化
    cropped_region: Optional[Tuple[int, int, int, int]] = None
    
    # 原始物件邊界 (用於 AOI 座標映射，避免重複讀取圖片)
    raw_bounds: Optional[Tuple[int, int, int, int]] = None

    # 面板 4 角 polygon (shape (4,2) float32，順序 TL/TR/BR/BL)
    # None 代表 polygon 偵測失敗或未啟用，下游應 fallback 回 axis-aligned bbox
    panel_polygon: Optional[np.ndarray] = field(default=None, repr=False)

    # Optional full preprocessed image cache for AOI-only v2 tile creation.
    processed_image: Optional[np.ndarray] = field(default=None, repr=False)

    # 推論耗時 (秒)
    inference_time: float = 0.0

    # Scratch classifier post-filter stats
    scratch_filter_count: int = 0           # 此 image 中被翻 OK 的 tile 數

    # 客戶端傳送的炸彈資訊 (供繪圖使用)
    client_bomb_info: Optional[Dict[str, Any]] = None
    
    # CV 邊緣檢查結果
    edge_defects: List[EdgeDefect] = field(default_factory=list)

    # WHITEFRA bright-frame observation.  Shadow-only: never affects formal judgment.
    white_frame_result: Optional[Dict[str, Any]] = None

    # Dot-matrix MARK binary detection metadata (detected from W0F0000 image)
    mark_text: str = ""
    mark_raw_text: str = ""
    mark_final_text: str = ""
    mark_adoption_reason: str = ""
    mark_temporal_history_count: int = 0
    mark_temporal_support_count: int = 0
    mark_confidence: float = 0.0
    mark_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    mark_roi: str = ""
    mark_orientation: str = ""
    mark_source_image: str = ""
    mark_shadow_result_id: int = 0
    mark_exclusion_regions: List[ExclusionRegion] = field(default_factory=list)

    # Image preprocessing timing metadata for record traceability.
    preprocess_steps: List[Dict[str, Any]] = field(default_factory=list)
    preprocess_total_ms: float = 0.0
    
    @property
    def total_tiles(self) -> int:
        return len(self.tiles)


class CAPIInferencer:
    """CAPI 推論器"""
    
    def __init__(
        self, 
        config: CAPIConfig, 
        model_path: Optional[str] = None,
        device: str = "auto",
        threshold: float = 0.5,
        base_dir: Optional[Path] = None,
    ):
        """
        初始化推論器
        
        Args:
            config: CAPI 配置
            model_path: PatchCore 模型路徑 (.xml 或 .pt)，作為 fallback
            device: 運算裝置 ("auto", "cpu", "cuda")
            threshold: 異常判斷閾值 (fallback，當 threshold_mapping 無對應時使用)
            base_dir: 基礎目錄（用於解析相對路徑）
        """
        self.config = config
        self.base_dir = base_dir or Path(__file__).parent
        if self._rotate_detection_images_180:
            logger.info("Detection input rotation enabled: config=inference_rotate_180_enabled angle=180")
        self.mark_template = None
        self.model_path = Path(model_path) if model_path else None
        self.threshold = threshold
        self.inferencer = None  # 保留向後相容 (fallback 單一模型)
        
        # 多模型快取: {model_path_str: inferencer_object}
        self._inferencers: Dict[str, Any] = {}
        
        # 前綴 → 模型路徑映射 (從 config 讀取)
        # 新架構 (is_new_architecture=True) 的 model_mapping value 為 nested dict
        # {prefix: {"inner": path, "edge": path}}，不在此處轉換；v2 直接讀 config.model_mapping
        self._model_mapping: Dict[str, Path] = {}
        for prefix, mpath in config.model_mapping.items():
            if isinstance(mpath, dict):
                # 新架構：nested dict，跳過舊式 flat mapping；v2 自行處理
                continue
            p = Path(mpath)
            if not p.is_absolute():
                p = self.base_dir / p
            self._model_mapping[prefix] = p

        # 前綴 → 閾值映射
        self._threshold_mapping: Dict[str, float] = {
            k: v for k, v in config.threshold_mapping.items()
            if not isinstance(v, dict)
        }
        
        # 決定運算裝置
        self.device = self._get_device(device)
        
        # 載入 MARK 模板
        self._load_mark_template()
        
        # 初始化傳統 CV 邊緣檢測器 (之後會從 DB 中讀取設定後覆蓋)
        self.edge_inspector = CVEdgeInspector()

        # 載入模型
        if self._model_mapping:
            # 多模型模式：預載所有映射的模型
            print(f"🔀 多模型模式: 偵測到 {len(self._model_mapping)} 個前綴映射")
            for prefix, mpath in self._model_mapping.items():
                print(f"   {prefix} → {mpath}")
                try:
                    inf = self._load_model_from_path(mpath)
                    if inf:
                        self._inferencers[str(mpath)] = inf
                except Exception as e:
                    print(f"   ⚠️ 載入失敗: {e}")
            print(f"✅ 已載入 {len(self._inferencers)}/{len(self._model_mapping)} 個模型")
            # 設定 self.inferencer 為第一個載入成功的模型 (向後相容)
            if self._inferencers:
                self.inferencer = next(iter(self._inferencers.values()))
        elif self.model_path:
            # 單一模型模式 (向後相容)
            self._load_model()

        # Scratch classifier post-filter (lazy-loaded on first NG tile)
        self.scratch_filter: ScratchFilter | None = None
        self._scratch_load_failed = False
        self._scratch_filter_signature = None

        # 分發器：依架構選擇 v1（舊 5 模型）或 v2（新 C-10）
        if getattr(config, "is_new_architecture", False):
            self._dispatch_process_panel = self._process_panel_v2
            self._model_cache_v2: Dict[tuple, Any] = {}
        else:
            self._dispatch_process_panel = self._process_panel_v1

    def _get_scratch_filter(self):
        """Lazy-load ScratchFilter (first call only). Thread-safe via _gpu_lock
        (caller responsibility — called inside process_panel)."""
        if not getattr(self.config, "scratch_classifier_enabled", False):
            return None

        current_safety = float(getattr(self.config, "scratch_safety_multiplier", 1.1))
        bundle = getattr(self.config, "scratch_bundle_path", "")
        weights = getattr(self.config, "scratch_dinov2_weights_path", "")
        repo_path = getattr(self.config, "scratch_dinov2_repo_path", "")
        signature = (str(bundle or ""), str(weights or ""), str(repo_path or ""), str(self.device))
        if self._scratch_filter_signature is not None and self._scratch_filter_signature != signature:
            self.scratch_filter = None
            self._scratch_load_failed = False
            self._scratch_filter_signature = None

        if self._scratch_load_failed:
            return None

        if self.scratch_filter is not None and \
                abs(self.scratch_filter._safety - current_safety) > 1e-9:
            self.scratch_filter = ScratchFilter(self.scratch_filter._classifier,
                                                safety_multiplier=current_safety)
        if self.scratch_filter is not None:
            return self.scratch_filter

        try:
            clf = ScratchClassifier(
                bundle_path=bundle,
                dinov2_weights_path=weights or None,
                dinov2_repo_path=repo_path or None,
                device=self.device,
            )
        except Exception as e:
            logger.error("ScratchClassifier load failed: %s", e, exc_info=True)
            self._scratch_load_failed = True
            self._scratch_filter_signature = signature
            return None
        self.scratch_filter = ScratchFilter(clf, safety_multiplier=current_safety)
        self._scratch_filter_signature = signature
        logger.info("ScratchClassifier filter ready (safety=%.2f, threshold=%.6f)",
                    current_safety, self.scratch_filter.effective_threshold)
        return self.scratch_filter

    def _get_device(self, device: str) -> str:
        """取得運算裝置"""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    @staticmethod
    def _log_cuda_memory(stage: str) -> None:
        """記錄 PyTorch allocator 與整張 GPU 的顯存快照。"""
        if not torch.cuda.is_available():
            return

        try:
            torch.cuda.synchronize()
            mib = 1024 * 1024
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            logger.info(
                "[CUDA-MEM] %s | "
                "allocated=%.1f MiB reserved=%.1f MiB "
                "peak_allocated=%.1f MiB peak_reserved=%.1f MiB "
                "device_used=%.1f MiB device_free=%.1f MiB",
                stage,
                torch.cuda.memory_allocated() / mib,
                torch.cuda.memory_reserved() / mib,
                torch.cuda.max_memory_allocated() / mib,
                torch.cuda.max_memory_reserved() / mib,
                (total_bytes - free_bytes) / mib,
                free_bytes / mib,
            )
        except Exception as exc:
            logger.warning("[CUDA-MEM] %s unavailable: %s", stage, exc)

    @staticmethod
    def _clear_cuda_cache(stage: str) -> None:
        """釋放未使用的 PyTorch CUDA cache，並記錄實際釋放量。"""
        if not torch.cuda.is_available():
            return

        try:
            torch.cuda.synchronize()
            mib = 1024 * 1024
            reserved_before = torch.cuda.memory_reserved()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            reserved_after = torch.cuda.memory_reserved()
            logger.info(
                "[CUDA-MEM] cache-clear %s | "
                "reserved_before=%.1f MiB reserved_after=%.1f MiB released=%.1f MiB",
                stage,
                reserved_before / mib,
                reserved_after / mib,
                max(0, reserved_before - reserved_after) / mib,
            )
        except Exception as exc:
            logger.warning("[CUDA-MEM] cache-clear %s failed: %s", stage, exc)
    
    def _load_model_from_path(self, model_path: Path) -> Optional[Any]:
        """載入 PatchCore 模型並回傳 inferencer 物件 (支援 OpenVINO 和 PyTorch)
        
        Args:
            model_path: 模型檔案路徑
            
        Returns:
            inferencer 物件，載入失敗回傳 None
        """
        if model_path is None or not model_path.exists():
            print(f"⚠️ 模型路徑無效: {model_path}")
            return None
        
        print(f"載入模型: {model_path}")
        print(f"使用裝置: {self.device}")
        
        model_ext = model_path.suffix.lower()
        inferencer_obj = None
        
        if model_ext == ".xml":
            # OpenVINO 格式
            from anomalib.deploy import OpenVINOInferencer
            print("📦 偵測到 OpenVINO 格式模型")
            inferencer_obj = OpenVINOInferencer(
                path=str(model_path),
                device="CPU",  # OpenVINO 使用 CPU
            )
        elif model_ext in (".pt", ".pth", ".ckpt"):
            # PyTorch 格式
            from anomalib.deploy import TorchInferencer
            import pathlib
            import platform
            
            print("📦 偵測到 PyTorch 格式模型")
            
            # 解決 WindowsPath 權重檔在 Linux 載入報錯的 workaround
            original_windows_path = pathlib.WindowsPath
            if platform.system() != 'Windows':
                pathlib.WindowsPath = pathlib.PosixPath

            try:
                self._log_cuda_memory(f"before-load model={model_path.name}")
                inferencer_obj = TorchInferencer(
                    path=str(model_path),
                    device=self.device,
                )
            finally:
                if platform.system() != 'Windows':
                    pathlib.WindowsPath = original_windows_path
        else:
            print(f"⚠️ 未知模型格式: {model_ext}")
            return None
        
        print(f"✅ 模型載入完成: {model_path.name}")

        # ── 舊版 checkpoint fp16 precision 修補 ──────────────────
        # 舊版 anomalib checkpoint 可能序列化了 PrecisionType.FLOAT16，
        # 新版 TorchInferencer.predict() 會依此將輸入轉 fp16，
        # 但 backbone 權重仍為 float32 → RuntimeError: Input type mismatch
        # 修補: 強制將 precision 設回 float32，並確保模型全為 float32
        self._fix_legacy_precision(inferencer_obj)

        # fp16 KNN 優化: 將 memory bank 轉為 fp16，並 patch euclidean_dist 使用 tensor core
        self._optimize_model_fp16(inferencer_obj)

        self._log_cuda_memory(f"after-load model={model_path.name}")

        # GPU Warm-up: 預先編譯 CUDA kernels，避免首次推論延遲
        if self.device != "cpu" and inferencer_obj is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
            except Exception as e:
                logger.warning("[CUDA-MEM] reset peak failed for %s: %s", model_path.name, e)
            try:
                print("🔥 GPU Warm-up 中...")
                dummy = np.zeros((self.config.tile_size, self.config.tile_size, 3), dtype=np.uint8)
                warmup_result = inferencer_obj.predict(dummy)
                del warmup_result
                print("✅ GPU Warm-up 完成")
            except Exception as e:
                print(f"⚠️ GPU Warm-up 失敗 (不影響推論): {e}")
            finally:
                self._log_cuda_memory(f"after-warmup model={model_path.name}")
                self._clear_cuda_cache(f"model={model_path.name}")
                self._log_cuda_memory(f"after-cache-clear model={model_path.name}")
        
        return inferencer_obj
    
    def _load_model(self) -> None:
        """載入 fallback 單一模型 (向後相容)"""
        result = self._load_model_from_path(self.model_path)
        if result is not None:
            self.inferencer = result
            self._inferencers[str(self.model_path)] = result
    
    def _get_image_prefix(self, filename: str) -> str:
        """從檔名中取得圖片前綴 (去除時間戳尾碼)
        
        例如: 'G0F00000_114438.tif' → 'G0F00000'
              'STANDARD.png' → 'STANDARD'
        """
        return canonical_image_prefix(filename)

    def _read_detection_image(
        self,
        image_path: Path,
        flags: int = cv2.IMREAD_UNCHANGED,
    ) -> Optional[np.ndarray]:
        return read_detection_image(
            image_path,
            flags,
            rotate_180=getattr(self, "_rotate_detection_images_180", False),
        )

    @property
    def _rotate_detection_images_180(self) -> bool:
        """Return the live runtime setting so settings hot-reload takes effect."""
        return bool(getattr(self.config, "inference_rotate_180_enabled", False))

    @staticmethod
    def _is_mark_binary_source(filename: str) -> bool:
        """正式推論只從 W0F0000 畫面讀 dot-matrix MARK。"""
        return Path(filename).name.upper().startswith("W0F0000")

    @staticmethod
    def _apply_online_paddle_mark_recognition(
        image: np.ndarray,
        detection: Dict[str, Any],
        source_path: Path,
    ) -> Dict[str, Any]:
        """Use PaddleOCR for the formal text while retaining the CV locator metadata."""
        legacy_text = str(detection.get("text") or "")
        legacy_confidence = float(detection.get("confidence") or 0.0)
        legacy_profile = int(detection.get("profile_version") or 0)
        detection["legacy_text"] = legacy_text
        detection["legacy_confidence"] = legacy_confidence

        try:
            from capi_mark_shadow import recognize_mark_online

            paddle_result = recognize_mark_online(image, detection, source_path)
        except Exception as exc:
            paddle_result = {
                "success": False,
                "error": str(exc),
                "round_trip_ms": 0.0,
            }

        paddle_text = str(paddle_result.get("paddle_text") or "").strip().upper()
        final_text = str(
            paddle_result.get("final_text") or paddle_text
        ).strip().upper()
        paddle_valid = bool(
            paddle_result.get("success")
            and re.fullmatch(r"[A-Z0-9]{2}", paddle_text)
            and re.fullmatch(r"[A-Z0-9]{2}", final_text)
        )
        detection["paddle_text"] = paddle_text
        detection["final_text"] = final_text if paddle_valid else ""
        detection["mark_adoption_reason"] = str(
            paddle_result.get("adoption_reason") or ""
        )
        detection["mark_temporal_stable_text"] = str(
            paddle_result.get("temporal_stable_text") or ""
        )
        detection["mark_temporal_history_count"] = int(
            paddle_result.get("temporal_history_count") or 0
        )
        detection["mark_temporal_support_count"] = int(
            paddle_result.get("temporal_stable_support_count") or 0
        )
        detection["paddle_confidence"] = float(
            paddle_result.get("paddle_confidence") or 0.0
        )
        detection["paddle_model"] = str(
            paddle_result.get("model_name") or "unknown"
        )
        detection["paddle_engine_version"] = str(
            paddle_result.get("engine_version") or "unknown"
        )
        detection["paddle_worker_version"] = str(
            paddle_result.get("worker_version") or "unknown"
        )
        detection["paddle_latency_ms"] = float(
            paddle_result.get("latency_ms") or 0.0
        )
        detection["paddle_round_trip_ms"] = float(
            paddle_result.get("round_trip_ms") or 0.0
        )
        detection["recognition_error"] = str(
            paddle_result.get("error") or ""
        ).replace("\r", " ").replace("\n", " ")[:240]
        try:
            detection["mark_shadow_result_id"] = max(
                0,
                int(paddle_result.get("id") or 0),
            )
        except (TypeError, ValueError):
            detection["mark_shadow_result_id"] = 0

        if paddle_valid:
            detection["text"] = final_text
            detection["confidence"] = detection["paddle_confidence"]
            detection["recognition_technique"] = "PaddleOCR"
            detection["recognition_version"] = detection["paddle_engine_version"]
            detection["recognition_fallback"] = False
            detection["recognition_reason"] = str(
                paddle_result.get("adoption_reason") or "paddle_primary"
            )
        else:
            detection["recognition_technique"] = "DotMatrixCV"
            detection["recognition_version"] = f"profile-v{legacy_profile}"
            detection["recognition_fallback"] = True
            detection["recognition_reason"] = detection.get("recognition_error") or "no_valid_two_chars"
        return paddle_result

    @staticmethod
    def _build_mark_stream_key(
        machine_no: Optional[str],
        model_id: Optional[str],
        detection: Dict[str, Any],
    ) -> str:
        """Partition temporal MARK history by the fixed PPOCR crop direction."""
        parts = [
            str(machine_no or "unknown").strip(),
            str(model_id or "unknown").strip(),
            str(detection.get("roi") or "unknown").strip(),
            "rot180",
        ]
        return "|".join(part.replace("|", "/") or "unknown" for part in parts)

    def _detect_panel_mark_binary_region(
        self,
        image_files: List[Path],
        *,
        machine_no: Optional[str] = None,
        model_id: Optional[str] = None,
        apply_recognition: bool = True,
    ) -> Tuple[Optional[Dict[str, Any]], List[ExclusionRegion]]:
        source_path = next(
            (f for f in image_files if self._is_mark_binary_source(f.name)),
            None,
        )
        if source_path is None:
            return None, []

        try:
            from capi_mark_detector import detect_panel_mark

            image = self._read_detection_image(source_path)
            if image is None:
                raise FileNotFoundError(f"cannot read image: {source_path}")
            detection = detect_panel_mark(image, include_debug=False)
            image_h, image_w = image.shape[:2]
            detection["source_image"] = source_path.name
            detection["image_size"] = (int(image_w), int(image_h))
        except Exception as exc:
            print(f"MARK Binary 偵測失敗 ({source_path.name}): {exc}")
            return None, []

        if not detection.get("found"):
            message = detection.get("message") or detection.get("error") or "not found"
            print(f"MARK Binary 未偵測到 ({source_path.name}): {message}")
            return detection, []

        bbox = detection.get("bbox") or {}
        try:
            x = int(bbox["x"])
            y = int(bbox["y"])
            width = int(bbox["width"])
            height = int(bbox["height"])
        except Exception:
            print(f"MARK Binary bbox 格式錯誤 ({source_path.name}): {bbox}")
            return detection, []

        region = ExclusionRegion(
            name="mark_binary",
            x1=x,
            y1=y,
            x2=x + width,
            y2=y + height,
        )
        detection["mark_bbox_tuple"] = (x, y, width, height)
        detection["mark_stream_key"] = self._build_mark_stream_key(
            machine_no,
            model_id,
            detection,
        )
        detection["mark_machine_no"] = str(machine_no or "")
        detection["mark_model_id"] = str(model_id or "")
        if not apply_recognition:
            return detection, [region]
        self._apply_online_paddle_mark_recognition(
            image,
            detection,
            source_path,
        )
        print(
            f"MARK Locator {source_path.name}: technique=DotMatrixCV "
            f"profile=v{int(detection.get('profile_version') or 0)} "
            f"roi={detection.get('roi', '')} "
            f"search={detection.get('search_pass', 'primary')} "
            f"orientation={detection.get('orientation', '')} "
            f"paddle_crop_rotation=rot180_fixed "
            f"bbox=({x},{y},{width},{height})"
        )
        if detection.get("recognition_fallback"):
            print(
                f"MARK Recognition {source_path.name}: "
                f"technique=DotMatrixCV "
                f"version={detection.get('recognition_version', '')} "
                f"decision=fallback text={detection.get('text', '')} "
                f"conf={float(detection.get('confidence') or 0.0):.3f} "
                f"paddle_model={detection.get('paddle_model', 'unknown')} "
                f"paddle_engine_version="
                f"{detection.get('paddle_engine_version', 'unknown')} "
                f"paddle_worker_api=v"
                f"{detection.get('paddle_worker_version', 'unknown')} "
                f"paddle_round_trip_ms="
                f"{float(detection.get('paddle_round_trip_ms') or 0.0):.1f} "
                f"reason={detection.get('recognition_error') or 'no_valid_two_chars'}"
            )
        else:
            print(
                f"MARK Recognition {source_path.name}: "
                f"technique=PaddleOCR "
                f"engine_version="
                f"{detection.get('paddle_engine_version', 'unknown')} "
                f"model={detection.get('paddle_model', 'unknown')} "
                f"worker_api=v"
                f"{detection.get('paddle_worker_version', 'unknown')} "
                f"decision=primary text={detection.get('text', '')} "
                f"raw={detection.get('paddle_text', '')} "
                f"adoption={detection.get('recognition_reason', '')} "
                f"history={int(detection.get('mark_temporal_support_count') or 0)}/"
                f"{int(detection.get('mark_temporal_history_count') or 0)} "
                f"conf={float(detection.get('confidence') or 0.0):.3f} "
                f"model_latency_ms="
                f"{float(detection.get('paddle_latency_ms') or 0.0):.1f} "
                f"round_trip_ms="
                f"{float(detection.get('paddle_round_trip_ms') or 0.0):.1f} "
                f"legacy_text={detection.get('legacy_text', '')}"
            )
        return detection, [region]

    def _attach_panel_mark_binary_to_results(
        self,
        results: List[ImageResult],
        mark_detection: Optional[Dict[str, Any]],
        mark_regions: List[ExclusionRegion],
    ) -> None:
        if not results or not mark_detection:
            return

        bbox_tuple = mark_detection.get("mark_bbox_tuple")
        for result in results:
            result.mark_exclusion_regions = list(mark_regions)
            result.mark_source_image = str(mark_detection.get("source_image", ""))
            result.mark_shadow_result_id = int(
                mark_detection.get("mark_shadow_result_id") or 0
            )
            if bbox_tuple is not None:
                result.mark_bbox = tuple(int(v) for v in bbox_tuple)
            if mark_detection.get("found"):
                result.mark_text = str(mark_detection.get("text", ""))
                result.mark_raw_text = str(mark_detection.get("paddle_text", ""))
                result.mark_final_text = str(
                    mark_detection.get("final_text")
                    or mark_detection.get("text", "")
                )
                result.mark_adoption_reason = str(
                    mark_detection.get("mark_adoption_reason")
                    or mark_detection.get("recognition_reason")
                    or ""
                )
                result.mark_temporal_history_count = int(
                    mark_detection.get("mark_temporal_history_count") or 0
                )
                result.mark_temporal_support_count = int(
                    mark_detection.get("mark_temporal_support_count") or 0
                )
                result.mark_confidence = float(mark_detection.get("confidence") or 0.0)
                result.mark_roi = str(mark_detection.get("roi", ""))
                result.mark_orientation = str(mark_detection.get("orientation", ""))

            for tile in result.tiles:
                tile.mark_exclusion_regions = list(mark_regions)
                tile.mark_exclusion_region_count = len(mark_regions)
    
    def _get_inferencer_for_prefix(self, prefix: str) -> Optional[Any]:
        """根據圖片前綴取得對應的 inferencer (含 lazy loading)
        
        查找順序:
        1. model_mapping 中的對應模型
        2. fallback 到 self.inferencer (單一模型)
        """
        # 查找映射
        if prefix in self._model_mapping:
            model_path = self._model_mapping[prefix]
            path_key = str(model_path)
            
            # 快取命中
            if path_key in self._inferencers:
                return self._inferencers[path_key]
            
            # Lazy loading
            print(f"🔄 Lazy loading 模型: {prefix} → {model_path}")
            inf = self._load_model_from_path(model_path)
            if inf is not None:
                self._inferencers[path_key] = inf
                return inf
            else:
                print(f"⚠️ {prefix} 模型載入失敗，fallback 到預設模型")
        
        # Fallback: 使用預設模型
        return self.inferencer
    
    def _get_threshold_for_prefix(self, prefix: str) -> float:
        """根據圖片前綴取得對應的閾值"""
        return self._threshold_mapping.get(prefix, self.threshold)

    def _get_inferencer_for_zone(self, prefix: str, zone: str) -> Optional[Any]:
        """新架構：依 (prefix, zone) 走 nested model_mapping；舊架構：fallback 到 prefix。

        新架構 (is_new_architecture=True) 的 model_mapping 是
        ``{prefix: {"inner": path, "edge": path}}``，需要 zone 才能解析。
        舊架構 ``{prefix: path}``，zone 參數忽略。
        """
        if getattr(self.config, "is_new_architecture", False):
            return self._get_model_for(self.config.machine_id, prefix, zone)
        return self._get_inferencer_for_prefix(prefix)

    def preload_v2_models(self) -> Tuple[int, int]:
        """新架構啟動預熱：載入所有 lighting × zone 模型到 v2 cache。"""
        if not getattr(self.config, "is_new_architecture", False):
            return 0, 0
        if not hasattr(self, "_model_cache_v2"):
            self._model_cache_v2 = {}

        total = 0
        loaded = 0
        for lighting, mapping in self.config.model_mapping.items():
            if not isinstance(mapping, dict):
                continue
            for zone in ("inner", "edge"):
                if not mapping.get(zone):
                    continue
                total += 1
                model = self._get_model_for(self.config.machine_id, lighting, zone)
                if model is None:
                    raise RuntimeError(
                        f"[v2] 模型預熱失敗: {self.config.machine_id}/{lighting}/{zone}"
                    )
                loaded += 1

        logger.info(
            "[v2] Preloaded %s/%s model units for machine=%s",
            loaded,
            total,
            self.config.machine_id,
        )
        self._log_cuda_memory(
            f"preload-complete machine={self.config.machine_id} loaded={loaded}/{total}"
        )
        return loaded, total

    def _get_threshold_for_zone(self, prefix: str, zone: str) -> float:
        """同上，threshold 版本。新架構 threshold_mapping 是 ``{prefix: {inner, edge}}``。"""
        if getattr(self.config, "is_new_architecture", False):
            thr_map = self.config.threshold_mapping.get(prefix)
            if isinstance(thr_map, dict):
                return float(thr_map.get(zone, self.threshold))
            if thr_map is not None:
                return float(thr_map)
            return self.threshold
        return self._get_threshold_for_prefix(prefix)

    def _resolve_aoi_edge_inspector_mode(self) -> str:
        """回傳實際使用的 inspector mode。

        新架構 (is_new_architecture=True) 強制 'patchcore'：edge.pt 已專為 edge zone
        訓練，CV+PC 空間分權的 fusion 失去理論基礎；同步把 cv 路徑停用以統一行為。
        舊架構讀 ``edge_inspector.config.aoi_edge_inspector`` (cv / patchcore / fusion)。
        """
        if getattr(self.config, "is_new_architecture", False):
            return "patchcore"
        if not getattr(self, "edge_inspector", None):
            return "cv"
        return getattr(self.edge_inspector.config, "aoi_edge_inspector", "cv")

    @staticmethod
    def _rect_polygon_from_bounds(bounds: Optional[Tuple[int, int, int, int]]) -> Optional[np.ndarray]:
        """Build a rectangular panel polygon from raw bounds for AOI inward clamping."""
        return rect_polygon_from_bounds(bounds)

    @staticmethod
    def _resolve_aoi_inward_shift_axes(
        img_x: int,
        img_y: int,
        bounds: Tuple[int, int, int, int],
        tile_size: int,
    ) -> str:
        """Limit AOI inward ROI correction to the axis implied by the nearest edge."""
        return resolve_aoi_inward_shift_axes(img_x, img_y, bounds, tile_size)

    @staticmethod
    def _format_aoi_tile_log_suffix(tile: 'TileInfo') -> str:
        if not getattr(tile, "is_aoi_coord_tile", False):
            return ""
        parts = []
        code = getattr(tile, "aoi_defect_code", "") or ""
        px = int(getattr(tile, "aoi_product_x", -1))
        py = int(getattr(tile, "aoi_product_y", -1))
        if px >= 0 and py >= 0:
            label = f"AOI({px},{py})"
            if code:
                label = f"AOI({code}:{px},{py})"
            parts.append(label)
        ix = int(getattr(tile, "aoi_image_x", -1))
        iy = int(getattr(tile, "aoi_image_y", -1))
        if ix >= 0 and iy >= 0:
            parts.append(f"img@({ix},{iy})")
        dx = int(getattr(tile, "aoi_tile_shift_dx", 0))
        dy = int(getattr(tile, "aoi_tile_shift_dy", 0))
        parts.append(f"shift=({dx:+d},{dy:+d})")
        return " " + " ".join(parts) if parts else ""

    def _load_mark_template(self) -> None:
        """載入 MARK 模板"""
        template_path = self.config.get_mark_template_full_path(self.base_dir)
        if template_path.exists():
            self.mark_template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if self.mark_template is not None:
                print(f"✅ MARK 模板載入: {template_path.name} ({self.mark_template.shape})")
        else:
            print(f"⚠️ MARK 模板不存在: {template_path}")
    
    def _find_raw_object_bounds(
        self, image: np.ndarray
    ) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
        """找尋物件的原始邊界 (不含 Offset)

        Returns:
            ((x_min, y_min, x_max, y_max), binary_mask) — binary_mask 是 Otsu +
            morphology close 後的 uint8 前景圖（255=前景），供後續 polygon 偵測重用。
        """
        img_height, img_width = image.shape[:2]

        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((15, 15), np.uint8)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf

        MIN_AREA = 1000
        for contour in contours:
            if cv2.contourArea(contour) > MIN_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x + w, x_max)
                y_max = max(y + h, y_max)

        if x_min == np.inf:
            return (0, 0, img_width, img_height), closing

        return (int(x_min), int(y_min), int(x_max), int(y_max)), closing

    def _find_panel_polygon(
        self,
        binary_mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """thin wrapper → capi_preprocess._polyfit_polygon（邏輯完全不變）。"""
        if binary_mask is None or binary_mask.size == 0:
            return None
        return _pf_polygon(binary_mask, bbox, self.config.tile_size)

    def _find_robust_object_bounds(
        self,
        image: np.ndarray,
    ) -> Tuple[Tuple[int, int, int, int], Optional[np.ndarray]]:
        """Use training preprocess boundary logic for small-product inference."""
        pre_cfg = PreprocessConfig(
            tile_size=self.config.tile_size,
            tile_stride=getattr(self.config, "tile_stride", self.config.tile_size),
            otsu_offset=0,
            enable_panel_polygon=self.config.enable_panel_polygon,
            product_resolution=self._product_resolution(),
        )
        bbox, polygon = detect_panel_polygon(image, pre_cfg)
        if bbox is None:
            h, w = image.shape[:2]
            return (0, 0, w, h), None
        return bbox, polygon

    def _product_resolution(self) -> Tuple[int, int]:
        return resolve_product_resolution(
            self.config.machine_id,
            getattr(self.config, "model_resolution_map", None),
        )

    def _use_robust_panel_boundary(self) -> bool:
        product_resolution = self._product_resolution()
        return is_small_product_resolution(product_resolution)

    def calculate_otsu_bounds(
        self,
        image: np.ndarray,
        otsu_offset_override: Optional[int] = None,
        reference_raw_bounds: Optional[Tuple[int, int, int, int]] = None,
        reference_polygon: Optional[np.ndarray] = None,
    ) -> Tuple[Tuple[int, int, int, int], Optional[int], Optional[np.ndarray]]:
        """
        計算 Otsu 前景邊界與 panel polygon。

        Args:
            reference_raw_bounds: 參考用的原始邊界 (來自同資料夾的白圖)。
                                  黑圖 (B0F) OTSU 無法正確偵測邊界時使用。
            reference_polygon: 參考用的 panel polygon (來自同資料夾的白圖)，
                               與 reference_raw_bounds 同時使用於 B0F fallback。
        Returns:
            (final_bounds, original_y2, panel_polygon)
        """
        img_height, img_width = image.shape[:2]

        # 取得原始物件邊界與 binary mask
        precomputed_polygon: Optional[np.ndarray] = None
        if reference_raw_bounds is not None:
            x_min, y_min, x_max, y_max = reference_raw_bounds
            binary_mask = None
        elif self._use_robust_panel_boundary():
            (x_min, y_min, x_max, y_max), precomputed_polygon = self._find_robust_object_bounds(image)
            binary_mask = None
        else:
            (x_min, y_min, x_max, y_max), binary_mask = self._find_raw_object_bounds(image)

        offset = otsu_offset_override if otsu_offset_override is not None else self.config.otsu_offset
        x_start = max(0, int(x_min) + offset)
        y_start = max(0, int(y_min) + offset)
        x_end = min(img_width, int(x_max) - offset)
        y_end = min(img_height, int(y_max) - offset)

        if x_start >= x_end or y_start >= y_end:
            x_start, y_start = 0, 0
            x_end, y_end = img_width, img_height

        # 應用底部裁切 (otsu_bottom_crop)
        original_y2 = None
        if self.config.otsu_bottom_crop > 0:
            h = y_end - y_start
            desired_height = max(self.config.tile_size, h - self.config.otsu_bottom_crop)
            final_height = min(h, desired_height)

            if final_height < h:
                original_y2 = y_end
                y_end = y_start + final_height

        bounds = (x_start, y_start, x_end, y_end)

        # 計算 panel polygon
        panel_polygon: Optional[np.ndarray] = None
        if self.config.enable_panel_polygon:
            if reference_polygon is not None:
                panel_polygon = reference_polygon.copy()
            elif precomputed_polygon is not None:
                panel_polygon = precomputed_polygon.copy()
            elif binary_mask is not None:
                # 使用原始 (未內縮) bbox 做邊緣掃描
                raw_bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
                panel_polygon = self._find_panel_polygon(binary_mask, raw_bbox)

            # 舊機種維持原本 polygon offset 內縮；小尺寸面板用 raw polygon
            # 畫推論紀錄外框，避免色差造成的邊界已偏內又再次內縮。
            if (panel_polygon is not None
                    and offset != 0
                    and reference_polygon is None
                    and not self._use_robust_panel_boundary()):
                cx = (panel_polygon[:, 0].mean())
                cy = (panel_polygon[:, 1].mean())
                for i in range(4):
                    dx = panel_polygon[i, 0] - cx
                    dy = panel_polygon[i, 1] - cy
                    length = float(np.hypot(dx, dy))
                    if length > 1e-6:
                        shrink = offset / length
                        panel_polygon[i, 0] -= dx * shrink
                        panel_polygon[i, 1] -= dy * shrink

            # 若 polygon 存在且啟用 otsu_bottom_crop，截掉下半部 —
            # 用 left/right 邊與新底線 y=new_bottom 的交點當新的 BL/BR，
            # 保留 panel 的底部傾斜度，而不是把兩角硬壓成同一 y。
            if panel_polygon is not None and original_y2 is not None:
                new_bottom = float(y_end)
                TL = panel_polygon[0]
                TR = panel_polygon[1]
                BR = panel_polygon[2]
                BL = panel_polygon[3]

                def _intersect_edge_with_horizontal(p_top, p_bot, y_line):
                    """線段 (p_top→p_bot) 與水平線 y=y_line 的交點 x 座標"""
                    dy = p_bot[1] - p_top[1]
                    if abs(dy) < 1e-9:
                        return float(p_top[0])
                    t = (y_line - p_top[1]) / dy
                    return float(p_top[0] + t * (p_bot[0] - p_top[0]))

                # 只有當現有 BR/BL 已經低於 new_bottom 時才做裁切
                if BR[1] > new_bottom or BL[1] > new_bottom:
                    new_BR_x = _intersect_edge_with_horizontal(TR, BR, new_bottom)
                    new_BL_x = _intersect_edge_with_horizontal(TL, BL, new_bottom)
                    panel_polygon[2, 0] = new_BR_x
                    panel_polygon[2, 1] = new_bottom
                    panel_polygon[3, 0] = new_BL_x
                    panel_polygon[3, 1] = new_bottom

        return bounds, original_y2, panel_polygon

    def find_panel_boundaries(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        向後相容別名，對應舊版呼叫。
        注意：此版本僅回傳四元組邊界，不回傳 original_y2 / polygon。
        """
        bounds, _, _ = self.calculate_otsu_bounds(image)
        return bounds

    def update_edge_config(self, config: Any):
        """
        更新 CV 邊緣檢測設定
        """
        self.edge_inspector = CVEdgeInspector(config)
    
    def find_mark_region(self, image: np.ndarray) -> Optional[ExclusionRegion]:
        """使用模板匹配找到 MARK 區域"""
        if self.mark_template is None:
            return None
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        img_h, img_w = gray.shape[:2]
        min_y_position = int(img_h * self.config.mark_min_y_ratio)
        
        best_match = None
        best_val = 0
        
        template_h, template_w = self.mark_template.shape[:2]
        scales = [0.75, 1.0, 1.5, 2.0, 3.0]
        
        for scale in scales:
            scaled_w = int(template_w * scale)
            scaled_h = int(template_h * scale)
            
            if scaled_w <= 0 or scaled_h <= 0:
                continue
            if scaled_w >= gray.shape[1] or scaled_h >= gray.shape[0]:
                continue
            
            scaled_template = cv2.resize(self.mark_template, (scaled_w, scaled_h))
            
            try:
                result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_loc[1] < min_y_position:
                    continue
                
                if max_val > best_val:
                    best_val = max_val
                    best_match = (max_loc[0], max_loc[1], scaled_w, scaled_h)
            except:
                continue
        
        if best_match is None or best_val < self.config.mark_match_threshold:
            return None
        
        mx, my, mw, mh = best_match
        return ExclusionRegion(
            name="mark_area",
            x1=mx,
            y1=my,
            x2=mx + mw,
            y2=my + mh,
        )
    
    def calculate_exclusion_regions(
        self,
        image: np.ndarray,
        otsu_bounds: Tuple[int, int, int, int],
        cached_mark: Optional[ExclusionRegion] = None,
        panel_polygon: Optional[np.ndarray] = None,
    ) -> List[ExclusionRegion]:
        """計算所有排除區域

        Args:
            image: 原始圖片
            otsu_bounds: Otsu 邊界
            cached_mark: 快取的 MARK 區域（Panel 級共用），若提供則跳過模板匹配
            panel_polygon: 面板 4 角 polygon，若提供則 relative_bottom_right 以
                           polygon BR 為錨點
        """
        # 如果啟用了底部裁切，且裁切量足夠大（例如 > 0），則假設機構和 MARK 都已被切除
        # 因此不再計算排外區域，避免 relative_bottom_right 誤標記到新的底部
        if self.config.otsu_bottom_crop > 0:
            return []
            
        regions = []
        otsu_x1, otsu_y1, otsu_x2, otsu_y2 = otsu_bounds
        
        for zone in self.config.get_enabled_exclusion_zones():
            if zone.type == "template_match" and zone.name == "mark_area":
                if cached_mark is not None:
                    # 使用快取的 MARK 位置（Panel 級共用）
                    regions.append(cached_mark)
                else:
                    # 未提供快取，進行模板匹配
                    mark_region = self.find_mark_region(image)
                    if mark_region:
                        regions.append(mark_region)
            
            elif zone.type == "relative_bottom_right":
                # 錨點: polygon BR 優先，否則回退到 bbox 右下
                if panel_polygon is not None:
                    anchor_x = int(round(float(panel_polygon[2][0])))
                    anchor_y = int(round(float(panel_polygon[2][1])))
                else:
                    anchor_x = otsu_x2
                    anchor_y = otsu_y2

                br_x1 = max(otsu_x1, anchor_x - zone.width)
                br_y1 = max(otsu_y1, anchor_y - zone.height)
                regions.append(ExclusionRegion(
                    name=zone.name,
                    x1=br_x1,
                    y1=br_y1,
                    x2=anchor_x,
                    y2=anchor_y,
                ))
        
        return regions
    
    def tile_image(
        self,
        image: np.ndarray,
        otsu_bounds: Tuple[int, int, int, int],
        exclusion_regions: List[ExclusionRegion],
        panel_polygon: Optional[np.ndarray] = None,
        exclusion_threshold: float = 0.0,  # 重疊比例超過此值則跳過 (0.0 = 任何重疊都跳過)
        original_image: Optional[np.ndarray] = None,
        skip_preprocess: bool = False,
    ) -> Tuple[List[TileInfo], int]:
        """
        將圖片切成 tile，完全跳過與排除區域重疊的 tile
        邊緣不足 512px 的區域會向前回推補齊

        Args:
            image: 原始圖片
            otsu_bounds: Otsu 邊界
            exclusion_regions: 排除區域列表
            panel_polygon: 面板 4 角 polygon (shape (4,2))。若提供，會與每個 tile
                           求交集產生 tile.mask，完全在 polygon 外的 tile 會被跳過，
                           完全在 polygon 內的 tile mask 設為 None 以節省記憶體。
            exclusion_threshold: 重疊比例閾值，超過此值則跳過該 tile (預設 0.0 = 任何重疊都跳過)

        Returns:
            (有效 tiles, 被跳過的 tile 數量)
        """
        otsu_x1, otsu_y1, otsu_x2, otsu_y2 = otsu_bounds
        tile_size = self.config.tile_size
        stride = self.config.tile_stride

        # 若提供 polygon，先在整張圖尺寸上畫好 panel mask，
        # 之後每個 tile 只要從裡面 slice 出對應區塊即可。
        # 這樣可以避免「shifted polygon 超出 tile canvas 時 cv2.fillPoly
        # 邊緣 rasterization 與 full-canvas 路徑不一致」的問題，
        # 同時保證與外部 ground truth (fillPoly(full_image_size)) 完全相符。
        # 記憶體成本: H*W uint8 (~28 MB for 6576x4384 panels)，
        # tile_image 返回時就釋放。
        full_panel_mask: Optional[np.ndarray] = None
        if panel_polygon is not None:
            H, W = image.shape[:2]
            full_panel_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(full_panel_mask, [panel_polygon.astype(np.int32)], 255)
        
        # 計算 X 和 Y 軸的 tile 起始座標（包含邊緣補齊）
        def generate_tile_positions(start: int, end: int, size: int, step: int) -> List[int]:
            """生成 tile 座標，邊緣不足時回推補齊"""
            positions = []
            pos = start
            while pos + size <= end:
                positions.append(pos)
                pos += step
            
            # 如果最後一個 tile 沒有覆蓋到邊緣，增加一個邊緣 tile
            if positions:
                last_end = positions[-1] + size
                if last_end < end:
                    # 向前回推，讓最後一個 tile 剛好貼齊邊緣
                    edge_pos = end - size
                    if edge_pos > positions[-1]:  # 避免重複
                        positions.append(edge_pos)
            elif end - start >= size:
                # 如果區域剛好等於 tile 大小
                positions.append(start)
            
            return positions
        
        x_positions = generate_tile_positions(otsu_x1, otsu_x2, tile_size, stride)
        y_positions = generate_tile_positions(otsu_y1, otsu_y2, tile_size, stride)
        
        # 判斷邊緣 tile 座標門檻
        bottom_y_threshold = otsu_y2 - tile_size  # 底排 tile 的起始 y 門檻
        right_x_threshold = otsu_x2 - tile_size   # 右排 tile 的起始 x 門檻
        
        tiles = []
        excluded_count = 0
        tile_id = 0
        
        for y in y_positions:
            for x in x_positions:
                tile_x2 = x + tile_size
                tile_y2 = y + tile_size
                
                # 檢查是否與任何排除區域重疊
                should_skip = False
                for region in exclusion_regions:
                    overlap = region.overlap_ratio(x, y, tile_x2, tile_y2)
                    if overlap > exclusion_threshold:
                        should_skip = True
                        break
                
                if should_skip:
                    excluded_count += 1
                    continue  # 完全跳過此 tile
                
                # 判斷是否為邊緣 tile
                is_bottom = (y >= bottom_y_threshold)
                is_top = (y <= otsu_y1)
                is_left = (x <= otsu_x1)
                is_right = (x >= right_x_threshold)
                
                # 擷取 tile 圖片
                tile_img = image[y:tile_y2, x:tile_x2].copy()
                original_tile = None
                if original_image is not None:
                    original_tile = original_image[y:tile_y2, x:tile_x2].copy()

                # 計算 tile 的 panel mask (polygon 交集)
                # 注意: .copy() 是必要的 — 不 copy 的話 tile.mask 會是
                # full_panel_mask 的 view，讓整張 28 MB buffer 無法在
                # tile_image 返回時釋放。不要刪除這個 copy。
                tile_mask: Optional[np.ndarray] = None
                if full_panel_mask is not None:
                    mask = full_panel_mask[y:tile_y2, x:tile_x2].copy()
                    if mask.max() == 0:
                        # Tile 完全在 polygon 外 → 跳過
                        excluded_count += 1
                        continue
                    if mask.min() == 255:
                        # Tile 完全在 polygon 內 → 省記憶體
                        tile_mask = None
                    else:
                        tile_mask = mask

                if (
                    not skip_preprocess
                    and getattr(self.config, "preprocess_after_tiling", False)
                    and getattr(self.config, "image_preprocess_pipeline", None)
                ):
                    from capi_image_preprocess_lab import apply_preprocess_pipeline
                    pipeline_result = apply_preprocess_pipeline(tile_img, self.config.image_preprocess_pipeline)
                    tile_img = pipeline_result["image"]

                tiles.append(TileInfo(
                    tile_id=tile_id,
                    x=x,
                    y=y,
                    width=tile_size,
                    height=tile_size,
                    image=tile_img,
                    original_image=original_tile,
                    mask=tile_mask,
                    has_exclusion=False,  # 保留此欄位以免影響其他程式碼
                    is_bottom_edge=is_bottom,
                    is_top_edge=is_top,
                    is_left_edge=is_left,
                    is_right_edge=is_right,
                ))
                tile_id += 1
        
        return tiles, excluded_count
    
    def preprocess_image(
        self,
        image_path: Path,
        cached_mark: Optional[ExclusionRegion] = None,
        otsu_offset_override: Optional[int] = None,
        reference_raw_bounds: Optional[Tuple[int, int, int, int]] = None,
        reference_polygon: Optional[np.ndarray] = None,
    ) -> Optional[ImageResult]:
        """
        預處理圖片：Otsu + 排除區域 + 切塊

        Args:
            image_path: 圖片路徑
            cached_mark: 快取的 MARK 區域（Panel 級共用）
            otsu_offset_override: Debug 用 Otsu 內縮覆寫值 (px)
            reference_raw_bounds: 參考用的原始邊界 (來自同資料夾的白圖)。
                                  黑圖 (B0F) OTSU 無法正確偵測邊界時使用。
            reference_polygon: 參考用的 panel polygon (來自同資料夾的白圖)，
                               與 reference_raw_bounds 同時使用於 B0F fallback。

        Returns:
            ImageResult 或 None（如果載入失敗）
        """
        start_time = time.time()

        # 載入圖片 (保持原始深度，例如 8-bit 灰階)
        raw_image = self._read_detection_image(image_path)
        if raw_image is None:
            print(f"⚠️ 無法載入: {image_path}")
            return None

        img_h, img_w = raw_image.shape[:2]

        is_skip_file = self.config.should_skip_file(image_path.name)
        processed_image = raw_image
        preprocess_steps: List[Dict[str, Any]] = []
        preprocess_total_ms = 0.0
        if (
            not is_skip_file
            and getattr(self.config, "image_preprocess_pipeline", None)
            and not getattr(self.config, "preprocess_after_tiling", False)
        ):
            from capi_image_preprocess_lab import apply_preprocess_pipeline, describe_preprocess_pipeline
            pipeline = self.config.image_preprocess_pipeline
            logger.info(f"[preprocess] pipeline: {describe_preprocess_pipeline(pipeline)}")
            pipeline_result = apply_preprocess_pipeline(raw_image, pipeline)
            processed_image = pipeline_result["image"]
            preprocess_steps = pipeline_result["steps"]
            preprocess_total_ms = float(pipeline_result.get("total_elapsed_ms") or 0.0)
            for step in pipeline_result["steps"]:
                logger.info(
                    "[preprocess] step %d %s params=%s elapsed=%.3fms stats=%s",
                    step["index"], step["method_label"], step["applied_params"],
                    float(step.get("elapsed_ms") or 0.0), step["stats"],
                )

        # 計算原始物件邊界 (用於 AOI 座標映射，只算一次)
        # 黑圖 (B0F) 使用參考邊界，因為 OTSU 無法正確偵測全黑畫面的邊界
        if reference_raw_bounds is not None:
            raw_bounds = reference_raw_bounds
            print(f"📐 {image_path.name}: 使用參考邊界 (來自白圖) → {raw_bounds}")
        else:
            raw_bounds, _raw_binary = self._find_raw_object_bounds(raw_image)

        # Otsu 裁切 (同樣使用參考邊界)
        otsu_bounds, original_y2, panel_polygon = self.calculate_otsu_bounds(
            processed_image,
            otsu_offset_override=otsu_offset_override,
            reference_raw_bounds=reference_raw_bounds,
            reference_polygon=reference_polygon,
        )
        
        # 記錄裁切區域
        cropped_region = None
        if original_y2 is not None:
            # (x1, y2_new, x2, y2_old)
            cropped_region = (otsu_bounds[0], otsu_bounds[3], otsu_bounds[2], original_y2)
        
        # 計算排除區域（使用快取的 MARK 位置）
        exclusion_regions = self.calculate_exclusion_regions(
            processed_image, otsu_bounds,
            cached_mark=cached_mark,
            panel_polygon=panel_polygon,
        )

        # 切塊
        tiles, excluded_count = self.tile_image(
            processed_image, otsu_bounds, exclusion_regions,
            panel_polygon=panel_polygon,
            original_image=raw_image,
            skip_preprocess=is_skip_file,
        )
        
        elapsed = time.time() - start_time
        
        return ImageResult(
            image_path=image_path,
            image_size=(img_w, img_h),
            otsu_bounds=otsu_bounds,
            cropped_region=cropped_region,
            exclusion_regions=exclusion_regions,
            tiles=tiles,
            excluded_tile_count=excluded_count,
            processed_tile_count=len(tiles),
            processing_time=elapsed,
            raw_bounds=raw_bounds,
            panel_polygon=panel_polygon,
            preprocess_steps=preprocess_steps,
            preprocess_total_ms=preprocess_total_ms,
        )
    
    def _apply_edge_margin(self, anomaly_map: np.ndarray, margin_px: int,
                           sides: list = None) -> np.ndarray:
        """
        對 anomaly_map 指定邊做線性漸層衰減 (1→0)
        用於過濾邊緣光影造成的假陽性
        
        Args:
            anomaly_map: 異常熱圖 (H, W)
            margin_px: 衰減區域寬度 (像素)
            sides: 要衰減的方向列表, 如 ['top', 'bottom', 'right']
            
        Returns:
            衰減後的 anomaly_map
        """
        h, w = anomaly_map.shape[:2]
        if margin_px <= 0:
            return anomaly_map
        if sides is None:
            sides = ['bottom']
        
        result = anomaly_map.copy()
        # 平方衰減 (Quadric Decay)，讓抑制效果更強
        linear = np.linspace(1.0, 0.0, margin_px).astype(np.float32)
        gradient = np.power(linear, 2)
        
        if 'bottom' in sides and margin_px < h:
            result[-margin_px:, :] *= gradient[:, None]
        
        if 'top' in sides and margin_px < h:
            result[:margin_px, :] *= gradient[::-1, None]  # 反向：頂部邊緣 0 → 1
        
        if 'right' in sides and margin_px < w:
            result[:, -margin_px:] *= gradient[None, :]
        
        if 'left' in sides and margin_px < w:
            result[:, :margin_px] *= gradient[None, ::-1]  # 反向：左側邊緣 0 → 1

        return result

    def _mask_tile_corner_exclusion(
        self,
        tile: TileInfo,
        anomaly_map: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], bool]:
        """將每個 Tile 四角的正方形區域設為不檢測。"""
        if anomaly_map is None or not getattr(self.config, "tile_corner_exclusion_enabled", False):
            return anomaly_map, False

        size_px = max(0, int(getattr(self.config, "tile_corner_exclusion_size_px", 32)))
        if size_px == 0:
            return anomaly_map, False

        map_h, map_w = anomaly_map.shape[:2]
        tile_h = max(1, int(tile.height))
        tile_w = max(1, int(tile.width))
        mask_h = min(map_h, max(1, int(np.ceil(size_px * map_h / tile_h))))
        mask_w = min(map_w, max(1, int(np.ceil(size_px * map_w / tile_w))))

        masked_map = anomaly_map.copy()
        masked_map[:mask_h, :mask_w] = 0
        masked_map[:mask_h, -mask_w:] = 0
        masked_map[-mask_h:, :mask_w] = 0
        masked_map[-mask_h:, -mask_w:] = 0
        return masked_map, True

    def _mask_tile_mark_exclusion_regions(
        self,
        tile: TileInfo,
        anomaly_map: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], bool]:
        regions = getattr(tile, "mark_exclusion_regions", None) or []
        masked, changed = self._apply_no_detect_region_weighting(
            tile,
            anomaly_map,
            regions,
            hard_padding_px=max(0, int(getattr(self.config, "mark_exclusion_padding_px", 32))),
            soft_decay_px=max(0, int(getattr(self.config, "mark_exclusion_soft_decay_px", 48))),
            core_weight=float(getattr(self.config, "no_detect_soft_decay_min_weight", 0.10)),
        )
        if changed:
            tile.mark_exclusion_masked = True
        return masked, changed

    def _configured_exclude_regions_for_model(
        self,
        model_id: Optional[str] = None,
    ) -> List[ExclusionRegion]:
        edge_inspector = getattr(self, "edge_inspector", None)
        edge_config = getattr(edge_inspector, "config", None) if edge_inspector else None
        if edge_config is None:
            return []
        try:
            if model_id and hasattr(edge_config, "set_active_zones_for_product"):
                resolution_code = model_id[5].upper() if len(model_id) >= 6 else ""
                edge_config.set_active_zones_for_product(resolution_code)
        except Exception as e:
            logger.error(f"取得不檢測區失敗: {e}", exc_info=True)
            return []

        regions = []
        for zone in getattr(edge_config, "exclude_zones", []) or []:
            if not getattr(zone, "enabled", False):
                continue
            x = int(getattr(zone, "x", 0))
            y = int(getattr(zone, "y", 0))
            w = int(getattr(zone, "w", 0))
            h = int(getattr(zone, "h", 0))
            if w <= 0 or h <= 0:
                continue
            regions.append(ExclusionRegion(
                name="cv_edge_exclude",
                x1=x,
                y1=y,
                x2=x + w,
                y2=y + h,
            ))
        return regions

    def _mask_tile_configured_exclude_regions(
        self,
        tile: TileInfo,
        anomaly_map: Optional[np.ndarray],
        model_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], bool]:
        regions = self._configured_exclude_regions_for_model(model_id)
        return self._apply_no_detect_region_weighting(
            tile,
            anomaly_map,
            regions,
            hard_padding_px=0,
            soft_decay_px=max(0, int(getattr(self.config, "cv_edge_exclude_soft_decay_px", 64))),
        )

    def _apply_no_detect_region_weighting(
        self,
        tile: TileInfo,
        anomaly_map: Optional[np.ndarray],
        regions: List[Any],
        hard_padding_px: int = 0,
        soft_decay_px: int = 0,
        core_weight: float = 0.0,
    ) -> Tuple[Optional[np.ndarray], bool]:
        if anomaly_map is None or not regions:
            return anomaly_map, False

        amap = np.asarray(anomaly_map)
        if amap.ndim < 2 or amap.size == 0:
            return anomaly_map, False

        tile_x = int(getattr(tile, "x", 0))
        tile_y = int(getattr(tile, "y", 0))
        tile_w = max(1, int(getattr(tile, "width", 0) or 1))
        tile_h = max(1, int(getattr(tile, "height", 0) or 1))
        amap_h, amap_w = amap.shape[:2]
        hard_padding_px = max(0, int(hard_padding_px))
        soft_decay_px = max(0, int(soft_decay_px))
        min_weight = float(getattr(self.config, "no_detect_soft_decay_min_weight", 0.10))
        min_weight = max(0.0, min(1.0, min_weight))
        core_weight = max(0.0, min(1.0, float(core_weight)))

        weight = np.ones((amap_h, amap_w), dtype=np.float32)
        changed = False

        for region in regions:
            rx1 = int(getattr(region, "x1", getattr(region, "x", 0)))
            ry1 = int(getattr(region, "y1", getattr(region, "y", 0)))
            rx2 = int(getattr(region, "x2", rx1 + int(getattr(region, "w", 0))))
            ry2 = int(getattr(region, "y2", ry1 + int(getattr(region, "h", 0))))
            if rx2 <= rx1 or ry2 <= ry1:
                continue

            core_x1 = rx1 - hard_padding_px
            core_y1 = ry1 - hard_padding_px
            core_x2 = rx2 + hard_padding_px
            core_y2 = ry2 + hard_padding_px
            influence_x1 = core_x1 - soft_decay_px
            influence_y1 = core_y1 - soft_decay_px
            influence_x2 = core_x2 + soft_decay_px
            influence_y2 = core_y2 + soft_decay_px

            ox1 = max(tile_x, influence_x1)
            oy1 = max(tile_y, influence_y1)
            ox2 = min(tile_x + tile_w, influence_x2)
            oy2 = min(tile_y + tile_h, influence_y2)
            if ox2 <= ox1 or oy2 <= oy1:
                continue

            mx1 = max(0, min(amap_w, int(np.floor((ox1 - tile_x) * amap_w / tile_w))))
            my1 = max(0, min(amap_h, int(np.floor((oy1 - tile_y) * amap_h / tile_h))))
            mx2 = max(0, min(amap_w, int(np.ceil((ox2 - tile_x) * amap_w / tile_w))))
            my2 = max(0, min(amap_h, int(np.ceil((oy2 - tile_y) * amap_h / tile_h))))
            if mx2 <= mx1 or my2 <= my1:
                continue

            yy, xx = np.mgrid[my1:my2, mx1:mx2]
            abs_x = tile_x + (xx.astype(np.float32) + 0.5) * tile_w / amap_w
            abs_y = tile_y + (yy.astype(np.float32) + 0.5) * tile_h / amap_h

            inside_core = (
                (abs_x >= core_x1) & (abs_x < core_x2) &
                (abs_y >= core_y1) & (abs_y < core_y2)
            )
            region_weight = np.ones_like(abs_x, dtype=np.float32)
            region_weight[inside_core] = core_weight

            if soft_decay_px > 0:
                dx = np.maximum(np.maximum(core_x1 - abs_x, 0), abs_x - core_x2)
                dy = np.maximum(np.maximum(core_y1 - abs_y, 0), abs_y - core_y2)
                dist = np.sqrt(dx * dx + dy * dy)
                ring = (~inside_core) & (dist <= soft_decay_px)
                region_weight[ring] = min_weight + (1.0 - min_weight) * (
                    dist[ring] / float(soft_decay_px)
                )

            weight[my1:my2, mx1:mx2] = np.minimum(weight[my1:my2, mx1:mx2], region_weight)
            changed = True

        if not changed:
            return anomaly_map, False

        self._restore_aoi_seed_weight(tile, weight, regions, hard_padding_px=hard_padding_px)
        return (amap * weight).astype(amap.dtype, copy=False), True

    def _restore_aoi_seed_weight(
        self,
        tile: TileInfo,
        weight: np.ndarray,
        regions: List[Any],
        hard_padding_px: int = 0,
    ) -> None:
        if not getattr(tile, "is_aoi_coord_tile", False):
            return
        if not getattr(self.config, "aoi_heatmap_center_seed_enabled", True):
            return

        ax = int(getattr(tile, "aoi_image_x", -1))
        ay = int(getattr(tile, "aoi_image_y", -1))
        if ax < 0 or ay < 0:
            return
        hard_padding_px = max(0, int(hard_padding_px))
        for region in regions:
            rx1 = int(getattr(region, "x1", getattr(region, "x", 0)))
            ry1 = int(getattr(region, "y1", getattr(region, "y", 0)))
            rx2 = int(getattr(region, "x2", rx1 + int(getattr(region, "w", 0))))
            ry2 = int(getattr(region, "y2", ry1 + int(getattr(region, "h", 0))))
            core_x1 = rx1 - hard_padding_px
            core_y1 = ry1 - hard_padding_px
            core_x2 = rx2 + hard_padding_px
            core_y2 = ry2 + hard_padding_px
            if core_x1 <= ax < core_x2 and core_y1 <= ay < core_y2:
                return

        tile_w = max(1, int(getattr(tile, "width", 0) or 1))
        tile_h = max(1, int(getattr(tile, "height", 0) or 1))
        local_x = ax - int(getattr(tile, "x", 0))
        local_y = ay - int(getattr(tile, "y", 0))
        if local_x < 0 or local_y < 0 or local_x >= tile_w or local_y >= tile_h:
            return

        h, w = weight.shape[:2]
        seed_x = int(round(local_x * (w - 1) / max(tile_w - 1, 1)))
        seed_y = int(round(local_y * (h - 1) / max(tile_h - 1, 1)))
        radius_tile_px = float(getattr(self.config, "aoi_heatmap_center_seed_radius_px", 12.0))
        radius = max(1, int(round(radius_tile_px * min(w / tile_w, h / tile_h))))
        yy, xx = np.ogrid[:h, :w]
        seed_mask = (yy - seed_y) ** 2 + (xx - seed_x) ** 2 <= radius ** 2
        grid_y, grid_x = np.mgrid[:h, :w]
        abs_x = int(getattr(tile, "x", 0)) + (grid_x.astype(np.float32) + 0.5) * tile_w / w
        abs_y = int(getattr(tile, "y", 0)) + (grid_y.astype(np.float32) + 0.5) * tile_h / h
        for region in regions:
            rx1 = int(getattr(region, "x1", getattr(region, "x", 0)))
            ry1 = int(getattr(region, "y1", getattr(region, "y", 0)))
            rx2 = int(getattr(region, "x2", rx1 + int(getattr(region, "w", 0))))
            ry2 = int(getattr(region, "y2", ry1 + int(getattr(region, "h", 0))))
            core_x1 = rx1 - hard_padding_px
            core_y1 = ry1 - hard_padding_px
            core_x2 = rx2 + hard_padding_px
            core_y2 = ry2 + hard_padding_px
            seed_mask &= ~(
                (abs_x >= core_x1) & (abs_x < core_x2) &
                (abs_y >= core_y1) & (abs_y < core_y2)
            )
        weight[seed_mask] = 1.0

    @staticmethod
    def _fix_legacy_precision(inferencer) -> None:
        """
        修補舊版/跨版本 anomalib checkpoint 的 fp16 precision 問題。

        問題: 模型 checkpoint 若以 fp16 precision 訓練（或跨版本載入時
        precision 屬性被錯誤還原），anomalib 的 forward 路徑可能使用
        torch.autocast(dtype=float16) 將輸入轉為 fp16，但 backbone
        (feature_extractor) 權重仍為 float32 → RuntimeError。

        策略:
        1. 修補所有能找到的 precision 屬性為 float32
        2. 在 feature_extractor (timm backbone) 上註冊 forward pre-hook，
           在每次 forward 前強制將輸入轉為 float32 — 此 hook 在最底層攔截，
           不受上層 autocast 影響，是最可靠的修補方式
        3. 確保模型權重為 float32
        """
        import enum

        model = getattr(inferencer, 'model', None)
        if model is None:
            return

        # ── Step 1: 修補所有 precision 屬性 ──
        fp32_values = {"float32", "FLOAT32", "32"}
        targets = [inferencer, model]

        # 也檢查 model.model (inner PatchcoreModel)
        inner_model = getattr(model, 'model', None)
        if inner_model is not None:
            targets.append(inner_model)

        for obj in targets:
            obj_name = type(obj).__name__
            for attr_name in ("precision", "_precision"):
                val = getattr(obj, attr_name, None)
                if val is None:
                    continue
                val_str = val.value if isinstance(val, enum.Enum) else str(val)
                if val_str not in fp32_values:
                    # 嘗試用同類 enum 的 FLOAT32 成員替換
                    if isinstance(val, enum.Enum):
                        try:
                            setattr(obj, attr_name, type(val)("float32"))
                            print(f"  🔧 修補 {obj_name}.{attr_name}: {val} → float32")
                            continue
                        except (ValueError, KeyError):
                            pass
                    setattr(obj, attr_name, "float32")
                    print(f"  🔧 修補 {obj_name}.{attr_name}: {val} → 'float32'")

        # ── Step 2: 在 feature_extractor 上註冊 forward pre-hook ──
        # 找到 feature_extractor (可能在 model 或 model.model 上)
        fe = None
        for candidate in [inner_model, model]:
            if candidate is not None and hasattr(candidate, 'feature_extractor'):
                fe = candidate.feature_extractor
                break

        if fe is not None:
            def _force_fp32_hook(module, args):
                """Forward pre-hook: 強制所有 tensor 輸入轉為 float32"""
                new_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor) and arg.dtype == torch.float16:
                        new_args.append(arg.float())
                    else:
                        new_args.append(arg)
                return tuple(new_args)

            fe.register_forward_pre_hook(_force_fp32_hook)
            print(f"  🔧 已在 feature_extractor ({type(fe).__name__}) 註冊 float32 pre-hook")
        else:
            print("  ⚠️ 未找到 feature_extractor，無法註冊 pre-hook")

        # ── Step 3: 確保模型權重為 float32 ──
        if isinstance(model, torch.nn.Module):
            has_half = any(p.dtype == torch.float16 for p in model.parameters())
            if has_half:
                model.float()
                print("  🔧 模型權重已全部轉回 float32")

    def _optimize_model_fp16(self, inferencer) -> None:
        """
        PatchCore KNN 加速: memory bank → fp16 + nearest_neighbors matmul → fp16 tensor core

        PatchCore 的推論瓶頸在 euclidean_dist 中的 torch.matmul(embedding, memory_bank.T)
        將此 matmul 轉為 fp16 可利用 GPU tensor core 大幅加速 (Ampere+ ~8x matmul throughput)
        norms 保持 fp32 避免 catastrophic cancellation
        """
        model = getattr(inferencer, 'model', None)
        if model is None or not isinstance(model, torch.nn.Module):
            return

        # 導航到 PatchCore torch model (可能被 Lightning module 包裝)
        # inferencer.model → Patchcore (Lightning) → .model → PatchcoreModel (torch)
        torch_model = model
        if hasattr(model, 'model') and isinstance(model.model, torch.nn.Module):
            torch_model = model.model

        # 1. Memory bank → fp16 (省 VRAM + 讓 matmul 自動使用 fp16)
        if hasattr(torch_model, 'memory_bank') and torch_model.memory_bank.numel() > 0:
            mb_shape = torch_model.memory_bank.shape
            vram_save_mb = mb_shape[0] * mb_shape[1] * 2 / 1024 / 1024  # fp32→fp16 省一半
            torch_model.memory_bank = torch_model.memory_bank.half()
            # 預算 y_norm (fp32) 並快取，memory bank 不變所以只需算一次
            torch_model._y_norm_cache = torch_model.memory_bank.float().pow(2).sum(dim=-1, keepdim=True)
            print(f"  ⚡ Memory bank → fp16 ({mb_shape[0]} vectors, 省 {vram_save_mb:.0f}MB VRAM)")
        else:
            print("  ⚠️ 未找到 memory_bank，跳過 fp16 優化")
            return

        # 2. Patch nearest_neighbors — 注入快取的 y_norm，matmul 用 fp16 tensor core
        torch_model_class = type(torch_model)
        if not getattr(torch_model_class, '_fp16_patched', False):
            # Patch euclidean_dist — 被 nearest_neighbors 和 compute_anomaly_score 共用
            # memory_bank 已轉 fp16，所有經過 euclidean_dist 的路徑都需要處理 dtype
            if not hasattr(torch_model_class, 'euclidean_dist'):
                print(f"  ⚠️ {torch_model_class.__name__} 無 euclidean_dist 方法，跳過 patch")
                return

            @staticmethod
            def _fp16_euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # norms 保持 fp32 (避免 catastrophic cancellation: a²-2ab+b²)
                x_norm = x.float().pow(2).sum(dim=-1, keepdim=True)
                y_norm = y.float().pow(2).sum(dim=-1, keepdim=True)
                # 核心 matmul 用 fp16 → tensor core 加速
                xy = torch.matmul(x.half(), y.half().transpose(-2, -1)).float()
                res = x_norm - 2 * xy + y_norm.transpose(-2, -1)
                return res.clamp_min_(0).sqrt_()

            torch_model_class.euclidean_dist = _fp16_euclidean_dist

            # Patch nearest_neighbors — 注入快取的 y_norm (memory bank 不變，只需算一次)
            if hasattr(torch_model_class, 'nearest_neighbors'):
                def _fp16_nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int):
                    x_norm = embedding.pow(2).sum(dim=-1, keepdim=True)
                    y_norm = getattr(self, '_y_norm_cache', None)
                    if y_norm is None:
                        y_norm = self.memory_bank.float().pow(2).sum(dim=-1, keepdim=True)
                    xy = torch.matmul(embedding.half(), self.memory_bank.half().transpose(-2, -1)).float()
                    distances = (x_norm - 2 * xy + y_norm.transpose(-2, -1)).clamp_min_(0).sqrt_()
                    if n_neighbors == 1:
                        return distances.min(1)
                    return distances.topk(k=n_neighbors, largest=False, dim=1)

                torch_model_class.nearest_neighbors = _fp16_nearest_neighbors

            torch_model_class._fp16_patched = True
            print("  ⚡ KNN → fp16 matmul + cached y_norm (tensor core acceleration)")

    def _prepare_tile_tensor(self, tile_image: np.ndarray) -> torch.Tensor:
        """將單一 tile 的 numpy 圖片轉為 tensor (CHW float32 [0,1])，匹配 anomalib 的預處理"""
        img = tile_image
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0

    def _tile_intersects_mark_influence(self, tile: TileInfo) -> bool:
        """Return whether the tile overlaps MARK padding or soft-decay range."""
        regions = getattr(tile, "mark_exclusion_regions", None) or []
        influence = max(0, int(getattr(self.config, "mark_exclusion_padding_px", 32))) + \
            max(0, int(getattr(self.config, "mark_exclusion_soft_decay_px", 48)))
        tx1, ty1 = int(tile.x), int(tile.y)
        tx2 = tx1 + max(1, int(tile.width))
        ty2 = ty1 + max(1, int(tile.height))
        for region in regions:
            base_x1 = int(getattr(region, "x1", getattr(region, "x", 0)))
            base_y1 = int(getattr(region, "y1", getattr(region, "y", 0)))
            base_x2 = int(
                getattr(region, "x2", base_x1 + int(getattr(region, "w", 0)))
            )
            base_y2 = int(
                getattr(region, "y2", base_y1 + int(getattr(region, "h", 0)))
            )
            rx1, ry1 = base_x1 - influence, base_y1 - influence
            rx2, ry2 = base_x2 + influence, base_y2 + influence
            if max(tx1, rx1) < min(tx2, rx2) and max(ty1, ry1) < min(ty2, ry2):
                return True
        return False

    @staticmethod
    def _normalize_patchcore_image_score(outer_model: Any, raw_score: torch.Tensor) -> float:
        """Apply the same image-score normalization used by anomalib post-processing."""
        normalized = raw_score
        post_processor = getattr(outer_model, "post_processor", None)
        if post_processor is not None and bool(
            getattr(post_processor, "enable_normalization", False)
        ):
            normalized = post_processor._normalize(
                raw_score,
                post_processor.image_min,
                post_processor.image_max,
                post_processor.image_threshold,
            )
        return float(normalized.reshape(-1)[0].item())

    @staticmethod
    def _prediction_value(value: Any) -> Optional[float]:
        """Convert a tensor/scalar prediction to a finite float for diagnostics."""
        if value is None:
            return None
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "reshape"):
                value = value.reshape(-1)[0]
            if hasattr(value, "item"):
                value = value.item()
            result = float(value)
            return result if np.isfinite(result) else None
        except (TypeError, ValueError, RuntimeError):
            return None

    def _capture_model_score_diagnostics(
        self,
        inferencer: Any,
        input_image: np.ndarray,
        normalized_predictions: Any,
    ) -> Dict[str, Any]:
        """Capture raw distance and normalization bounds for coordinate diagnostics.

        Formal inference must not pay for a second model call.  This helper is
        called only by the opt-in coordinate debug path; it temporarily disables
        Anomalib post-processing normalization, then restores the original flag.
        """
        diagnostics: Dict[str, Any] = {
            "raw_model_score": None,
            "model_image_min": None,
            "model_image_max": None,
            "model_image_threshold": None,
            "model_normalization_enabled": None,
            "raw_anomaly_map_max": None,
            "normalized_anomaly_map_max": None,
        }
        outer_model = getattr(inferencer, "model", None)
        post_processor = getattr(outer_model, "post_processor", None)
        if post_processor is None:
            return diagnostics

        for field_name in (
            "image_min", "image_max", "image_threshold",
        ):
            diagnostics[f"model_{field_name}"] = self._prediction_value(
                getattr(post_processor, field_name, None)
            )
        normalized_map = getattr(normalized_predictions, "anomaly_map", None)
        if normalized_map is not None:
            try:
                diagnostics["normalized_anomaly_map_max"] = self._prediction_value(
                    torch.amax(normalized_map)
                    if hasattr(normalized_map, "detach")
                    else np.max(np.asarray(normalized_map))
                )
            except (TypeError, ValueError, RuntimeError):
                pass

        if not hasattr(post_processor, "enable_normalization"):
            return diagnostics
        previous_normalization = bool(post_processor.enable_normalization)
        diagnostics["model_normalization_enabled"] = previous_normalization
        try:
            post_processor.enable_normalization = False
            raw_predictions = inferencer.predict(input_image)
            diagnostics["raw_model_score"] = self._prediction_value(
                getattr(raw_predictions, "pred_score", None)
            )
            raw_map = getattr(raw_predictions, "anomaly_map", None)
            if raw_map is not None:
                try:
                    diagnostics["raw_anomaly_map_max"] = self._prediction_value(
                        torch.amax(raw_map)
                        if hasattr(raw_map, "detach")
                        else np.max(np.asarray(raw_map))
                    )
                except (TypeError, ValueError, RuntimeError):
                    pass
        except Exception as exc:
            logger.warning("Coordinate raw-score diagnostics failed: %s", exc)
        finally:
            post_processor.enable_normalization = previous_normalization
        return diagnostics

    def _predict_with_mark_patch_score(
        self,
        tile: TileInfo,
        inferencer: Any,
        input_image: np.ndarray,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Exclude MARK-affected patches before PatchCore selects its image score."""
        detail: Dict[str, Any] = {
            "applied": False,
            "reason": "unsupported_model_api",
            "valid_count": 0,
            "total_count": 0,
            "peak_x": -1,
            "peak_y": -1,
            "elapsed_ms": 0.0,
        }
        outer_model = getattr(inferencer, "model", None)
        patchcore_model = getattr(outer_model, "model", None)
        if patchcore_model is None and callable(
            getattr(outer_model, "compute_anomaly_score", None)
        ):
            patchcore_model = outer_model
        original_compute = getattr(patchcore_model, "compute_anomaly_score", None)
        if not callable(original_compute):
            return inferencer.predict(input_image), detail

        instance_dict = getattr(patchcore_model, "__dict__", {})
        had_override = "compute_anomaly_score" in instance_dict
        previous_override = instance_dict.get("compute_anomaly_score")

        def _mark_aware_score(patch_scores, locations, embedding):
            original_score = original_compute(patch_scores, locations, embedding)
            detail["original_raw_score"] = original_score.detach()
            started = time.perf_counter()
            try:
                scores = patch_scores if patch_scores.ndim == 2 else patch_scores.unsqueeze(0)
                if scores.shape[0] != 1:
                    detail["reason"] = "unsupported_patch_batch"
                    return original_score

                total_count = int(scores.shape[1])
                grid_size = int(round(total_count ** 0.5))
                detail["total_count"] = total_count
                if grid_size <= 0 or grid_size * grid_size != total_count:
                    detail["reason"] = "non_square_patch_grid"
                    return original_score

                probe = np.ones((grid_size, grid_size), dtype=np.float32)
                weighted, changed = self._apply_no_detect_region_weighting(
                    tile,
                    probe,
                    getattr(tile, "mark_exclusion_regions", None) or [],
                    hard_padding_px=max(
                        0, int(getattr(self.config, "mark_exclusion_padding_px", 32))
                    ),
                    soft_decay_px=max(
                        0, int(getattr(self.config, "mark_exclusion_soft_decay_px", 48))
                    ),
                    core_weight=float(
                        getattr(self.config, "no_detect_soft_decay_min_weight", 0.10)
                    ),
                )
                if not changed:
                    detail["reason"] = "mark_does_not_reach_patch_grid"
                    return original_score

                valid = np.asarray(weighted).reshape(-1) >= (1.0 - 1e-6)
                valid_count = int(np.count_nonzero(valid))
                detail["valid_count"] = valid_count
                if valid_count <= 0:
                    detail["reason"] = "all_patches_excluded"
                    return original_score
                if valid_count == total_count:
                    detail["reason"] = "no_patch_excluded"
                    return original_score

                valid_tensor = torch.as_tensor(
                    valid, dtype=torch.bool, device=scores.device
                )
                masked_scores = scores.clone()
                masked_scores[:, ~valid_tensor] = float("-inf")
                peak_index = int(torch.argmax(masked_scores[0]).item())
                mark_free_score = original_compute(
                    masked_scores, locations, embedding
                )
                peak_row, peak_col = divmod(peak_index, grid_size)
                detail.update({
                    "applied": True,
                    "reason": "applied",
                    "raw_score": mark_free_score.detach(),
                    "peak_x": int(round(
                        tile.x + (peak_col + 0.5) * tile.width / grid_size
                    )),
                    "peak_y": int(round(
                        tile.y + (peak_row + 0.5) * tile.height / grid_size
                    )),
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                })
                return mark_free_score
            except Exception as exc:
                detail["reason"] = f"score_recompute_failed:{type(exc).__name__}"
                logger.warning("MARK PatchCore score recompute failed: %s", exc)
                return original_score

        patchcore_model.compute_anomaly_score = _mark_aware_score
        try:
            predictions = inferencer.predict(input_image)
        finally:
            if had_override:
                patchcore_model.compute_anomaly_score = previous_override
            else:
                delattr(patchcore_model, "compute_anomaly_score")

        if detail["applied"]:
            pred_score = predictions.pred_score
            detail["score"] = float(
                pred_score.item() if hasattr(pred_score, "item") else pred_score
            )
            detail["original_score"] = self._normalize_patchcore_image_score(
                outer_model, detail["original_raw_score"]
            )
            detail["model"] = type(patchcore_model).__name__
        elif "original_raw_score" not in detail:
            detail["reason"] = "patch_score_not_emitted"
        return predictions, detail

    def _batch_forward(self, tiles: List[TileInfo], inferencer, batch_size: int = 4) -> Optional[List[Tuple[float, Optional[np.ndarray]]]]:
        """
        批次模型推論，回傳每個 tile 的 (pred_score, anomaly_map_numpy)

        僅支援 PyTorch 模型，OpenVINO 模型回傳 None (fallback 到逐 tile 推論)
        batch_size 預設 4，512x512 tiles 在 16GB VRAM 下安全運行
        """
        model = getattr(inferencer, 'model', None)
        if model is None or not isinstance(model, torch.nn.Module):
            return None

        device = getattr(inferencer, 'device', self.device)
        model.eval()

        # 預先轉換所有 tile 為 tensor (CPU 上)
        tensors = [self._prepare_tile_tensor(t.image) for t in tiles]
        results = []

        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start:start + batch_size]).to(device)
            with torch.no_grad():
                preds = model(batch)

            scores = preds.pred_score
            amaps = preds.anomaly_map

            for i in range(batch.shape[0]):
                score = float(scores[i].item()) if scores.ndim > 0 else float(scores.item())
                amap = None
                if amaps is not None:
                    amap = amaps[i].squeeze().cpu().numpy()
                results.append((score, amap))

            # 釋放 GPU 記憶體
            del batch, preds, scores, amaps

        return results

    def predict_tile(self, tile: TileInfo, inferencer=None, edge_margin_override: Optional[int] = None, patchcore_overrides: Optional[Dict[str, Any]] = None, threshold: Optional[float] = None, raw_prediction: Optional[Tuple[float, Optional[np.ndarray]]] = None, model_id: Optional[str] = None, capture_raw_diagnostics: bool = False) -> Tuple[float, Optional[np.ndarray]]:
        """
        對單一 tile 進行推論

        Args:
            tile: TileInfo 物件
            inferencer: 指定的 inferencer 物件，若為 None 使用 self.inferencer
            threshold: 異常判斷閾值（用於面積過濾），若為 None 使用 self.threshold
            raw_prediction: 預先計算的 (pred_score, anomaly_map)，若提供則跳過模型推論
            capture_raw_diagnostics: 僅供座標診斷頁，額外記錄未正規化模型距離

        Returns:
            (異常分數, 異常熱圖) - 如果有遮罩，會過濾排除區域的異常
        """
        active_threshold = threshold if threshold is not None else self.threshold
        patchcore_enabled = getattr(self.config, 'patchcore_filter_enabled', False)
        if patchcore_overrides is not None and 'patchcore_filter_enabled' in patchcore_overrides:
            patchcore_enabled = patchcore_overrides['patchcore_filter_enabled']
        mark_patch_detail: Dict[str, Any] = {
            "applied": False,
            "reason": "raw_prediction" if raw_prediction is not None else "not_applicable",
        }

        tile.mark_patch_score_applied = False
        tile.mark_patchcore_score = 0.0
        tile.mark_patch_valid_count = 0
        tile.mark_patch_total_count = 0
        tile.mark_patch_peak_x = -1
        tile.mark_patch_peak_y = -1
        tile.mark_patch_score_reason = ""
        tile.raw_model_score = None
        tile.model_image_min = None
        tile.model_image_max = None
        tile.model_image_threshold = None
        tile.model_normalization_enabled = None
        tile.raw_anomaly_map_max = None
        tile.normalized_anomaly_map_max = None

        if raw_prediction is not None:
            # 使用預先批次計算的結果，跳過模型推論
            pred_score, anomaly_map = raw_prediction
        else:
            # 逐 tile 推論 (fallback)
            active_inferencer = inferencer or self.inferencer
            if active_inferencer is None:
                raise RuntimeError("模型尚未載入")

            # 使用 numpy array 進行推論
            # 如果是灰階 (2D 或 1 channel)，轉為 BGR
            input_image = tile.image
            if len(input_image.shape) == 2 or (len(input_image.shape) == 3 and input_image.shape[2] == 1):
                 if len(input_image.shape) == 2:
                     input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
                 else:
                     input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

            with _MARK_PATCH_SCORE_LOCK:
                if (
                    not patchcore_enabled
                    and self._tile_intersects_mark_influence(tile)
                ):
                    predictions, mark_patch_detail = \
                        self._predict_with_mark_patch_score(
                            tile, active_inferencer, input_image
                        )
                else:
                    predictions = active_inferencer.predict(input_image)
                    if patchcore_enabled:
                        mark_patch_detail["reason"] = "patchcore_filter_enabled"
                    else:
                        mark_patch_detail["reason"] = "mark_not_in_tile"

            # 取得分數
            pred_score = float(predictions.pred_score.item()) if hasattr(predictions.pred_score, 'item') else float(predictions.pred_score)

            # 取得熱圖（如果有的話）
            anomaly_map = None
            if hasattr(predictions, 'anomaly_map') and predictions.anomaly_map is not None:
                anomaly_map = predictions.anomaly_map.squeeze().cpu().numpy() if hasattr(predictions.anomaly_map, 'cpu') else predictions.anomaly_map.squeeze()

            if capture_raw_diagnostics and not mark_patch_detail.get("applied"):
                diagnostics = self._capture_model_score_diagnostics(
                    active_inferencer, input_image, predictions,
                )
                tile.raw_model_score = diagnostics["raw_model_score"]
                tile.model_image_min = diagnostics["model_image_min"]
                tile.model_image_max = diagnostics["model_image_max"]
                tile.model_image_threshold = diagnostics["model_image_threshold"]
                tile.model_normalization_enabled = diagnostics[
                    "model_normalization_enabled"
                ]
                tile.raw_anomaly_map_max = diagnostics["raw_anomaly_map_max"]
                tile.normalized_anomaly_map_max = diagnostics["normalized_anomaly_map_max"]

        raw_pred_score = float(
            mark_patch_detail.get("original_score", pred_score)
            if mark_patch_detail.get("applied")
            else pred_score
        )
        tile.raw_pred_score = raw_pred_score
        tile.pre_decay_map_max = 0.0
        tile.post_decay_map_max = 0.0
        tile.score_decay_ratio = 1.0
        tile.score_edge_margin_sides = ""
        tile.score_mask_valid_ratio = tile.valid_ratio
        tile.mark_exclusion_masked = False
        if mark_patch_detail.get("applied"):
            tile.mark_patch_score_applied = True
            tile.mark_patchcore_score = float(mark_patch_detail["score"])
            tile.mark_patch_valid_count = int(mark_patch_detail.get("valid_count", 0))
            tile.mark_patch_total_count = int(mark_patch_detail.get("total_count", 0))
            tile.mark_patch_peak_x = int(mark_patch_detail.get("peak_x", -1))
            tile.mark_patch_peak_y = int(mark_patch_detail.get("peak_y", -1))
            tile.mark_patch_score_reason = "applied"
        mark_masked = False
        corner_masked = False
        configured_exclude_weighted = False
        pre_mark_map_max = 0.0
        post_mark_map_max = 0.0

        # === 以下為 anomaly_map 後處理 (batch 和 fallback 共用) ===
        if anomaly_map is not None:
            # 記錄衰減/遮罩處理前的 anomaly_map max (用於後續比率計算)
            pre_process_max = float(np.max(anomaly_map))

            # --- PatchCore 後處理過濾 ---
            if patchcore_enabled:
                # 1. 高斯平滑
                sigma = getattr(self.config, 'patchcore_blur_sigma', 1.5)
                if patchcore_overrides is not None and 'patchcore_blur_sigma' in patchcore_overrides:
                    sigma = patchcore_overrides['patchcore_blur_sigma']

                if sigma > 0:
                    ksize = int(2 * round(3 * sigma) + 1)
                    anomaly_map = cv2.GaussianBlur(anomaly_map, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

                # 2. 特徵值計算模式
                metric = getattr(self.config, 'patchcore_score_metric', 'max')
                if patchcore_overrides is not None and 'patchcore_score_metric' in patchcore_overrides:
                    metric = patchcore_overrides['patchcore_score_metric']

                if metric == 'top_k_avg':
                    # 取前 10 個最高值的平均 (k=10)
                    k = 10
                    flat = anomaly_map.flatten()
                    if len(flat) >= k:
                        idx = np.argpartition(flat, -k)[-k:]
                        top_k_val = np.mean(flat[idx])
                    else:
                        top_k_val = np.max(anomaly_map)
                    pre_process_max = float(top_k_val)
                elif metric == 'percentile_99':
                    pre_process_max = float(np.percentile(anomaly_map, 99))
                else: # 'max' 或其他
                    pre_process_max = float(np.max(anomaly_map))

                # 更新基礎預測分數 (覆寫原本直接從 anomalib 拿的 score)
                # 這裡假設 anomaly_map 的尺度與 pred_score 一致，直接取代
                # 如果尺度不一致，這裡會成為新的基準
                pred_score = pre_process_max

                # 3. 面積過濾 (只在分數超過閾值時檢查，節省效能)
                min_area = getattr(self.config, 'patchcore_min_area', 10)
                if patchcore_overrides is not None and 'patchcore_min_area' in patchcore_overrides:
                    min_area = patchcore_overrides['patchcore_min_area']

                if min_area > 0 and pred_score >= active_threshold:
                    # 以 peak×0.5 二值化找最大 cluster，判斷是否為雜訊點
                    max_cluster_area = _anomaly_max_cc_area(anomaly_map, pre_process_max)

                    if max_cluster_area < min_area:
                        # 面積不足，大幅降權
                        pred_score = pred_score * 0.5
                        print(f"    ℹ️ Tile 異常面積過小 ({max_cluster_area} < {min_area})，降權懲罰")

            # --- 集中度檢查 (Concentration Check) ---
            # 瀰漫性假陽性: heatmap 均勻偏暖但無局部峰值 → Peak/Mean ratio 低
            # 真實缺陷: heatmap 有明顯局部峰值 → Peak/Mean ratio 高
            concentration_enabled = getattr(self.config, 'patchcore_concentration_enabled', True)
            if patchcore_overrides is not None and 'patchcore_concentration_enabled' in patchcore_overrides:
                concentration_enabled = patchcore_overrides['patchcore_concentration_enabled']

            if concentration_enabled:
                positive_vals = anomaly_map[anomaly_map > 0]
                if len(positive_vals) > 0:
                    peak_val = float(np.max(positive_vals))
                    mean_val = float(np.mean(positive_vals))
                    concentration_ratio = peak_val / mean_val if mean_val > 0 else float('inf')

                    min_ratio = getattr(self.config, 'patchcore_concentration_min_ratio', 2.0)
                    if patchcore_overrides is not None and 'patchcore_concentration_min_ratio' in patchcore_overrides:
                        min_ratio = patchcore_overrides['patchcore_concentration_min_ratio']

                    if concentration_ratio < min_ratio and min_ratio > 1.0:
                        penalty = getattr(self.config, 'patchcore_concentration_penalty', 0.5)
                        if patchcore_overrides is not None and 'patchcore_concentration_penalty' in patchcore_overrides:
                            penalty = patchcore_overrides['patchcore_concentration_penalty']

                        # 線性插值: ratio=1.0 → penalty, ratio=min_ratio → 1.0 (無懲罰)
                        factor = (concentration_ratio - 1.0) / (min_ratio - 1.0)
                        factor = max(0.0, min(1.0, factor))
                        penalty_mult = penalty + (1.0 - penalty) * factor
                        pred_score *= penalty_mult
                        logger.debug(f"瀰漫性檢查: Peak/Mean={concentration_ratio:.2f} < {min_ratio:.1f}，降權 x{penalty_mult:.3f}")

            # --- 擴散面積檢查 (Diffuse Area Check) ---
            # 梯度型假陽性: heatmap 有大面積偏暖 (左熱右冷等梯度) → 熱區佔比高
            # 真實缺陷: heatmap 熱區集中在小區域 → 熱區佔比低
            diffuse_enabled = getattr(self.config, 'patchcore_diffuse_area_enabled', True)
            if patchcore_overrides is not None and 'patchcore_diffuse_area_enabled' in patchcore_overrides:
                diffuse_enabled = patchcore_overrides['patchcore_diffuse_area_enabled']

            if diffuse_enabled:
                map_max = float(np.max(anomaly_map))
                if map_max > 0:
                    half_peak = map_max * 0.5
                    hot_pixels = int(np.count_nonzero(anomaly_map >= half_peak))
                    total_pixels = anomaly_map.size
                    hot_ratio = hot_pixels / total_pixels if total_pixels > 0 else 0.0

                    diffuse_threshold = getattr(self.config, 'patchcore_diffuse_area_threshold', 0.3)
                    if patchcore_overrides is not None and 'patchcore_diffuse_area_threshold' in patchcore_overrides:
                        diffuse_threshold = patchcore_overrides['patchcore_diffuse_area_threshold']

                    if hot_ratio > diffuse_threshold:
                        diffuse_penalty = getattr(self.config, 'patchcore_diffuse_area_penalty', 0.5)
                        if patchcore_overrides is not None and 'patchcore_diffuse_area_penalty' in patchcore_overrides:
                            diffuse_penalty = patchcore_overrides['patchcore_diffuse_area_penalty']

                        # 線性插值: hot_ratio=threshold → 1.0 (無懲罰), hot_ratio=1.0 → penalty (最大懲罰)
                        factor = (hot_ratio - diffuse_threshold) / (1.0 - diffuse_threshold) if diffuse_threshold < 1.0 else 0.0
                        factor = max(0.0, min(1.0, factor))
                        penalty_mult = 1.0 - (1.0 - diffuse_penalty) * factor
                        pred_score *= penalty_mult
                        logger.debug(f"擴散面積檢查: HotRatio={hot_ratio:.2%} > {diffuse_threshold:.0%}，降權 x{penalty_mult:.3f}")

            # 記錄 mask/邊緣衰減前的 max (用於 decay ratio，統一使用 max 避免 metric 不一致)
            pre_decay_max = float(np.max(anomaly_map))
            edge_margin_sides = []

            # 如果有遮罩，將排除區域的熱圖值設為 0
            if tile.mask is not None:
                # 確保遮罩尺寸匹配
                if anomaly_map.shape != tile.mask.shape:
                    mask_resized = cv2.resize(tile.mask, (anomaly_map.shape[1], anomaly_map.shape[0]))
                else:
                    mask_resized = tile.mask
                # 將排除區域設為 0
                anomaly_map = anomaly_map * (mask_resized / 255.0)

            # 每個 Tile 的四角可設定為正方形不檢測區。
            anomaly_map, corner_masked = self._mask_tile_corner_exclusion(tile, anomaly_map)
            pre_mark_map_max = float(np.max(anomaly_map))

            # MARK binary 區域屬於不檢測區域，只遮掉 tile 內重疊的 heatmap 像素。
            anomaly_map, mark_masked = self._mask_tile_mark_exclusion_regions(tile, anomaly_map)
            post_mark_map_max = float(np.max(anomaly_map))

            # 手動不檢測區也要在主 score 階段套用，否則固定結構的 heatmap
            # 只會在 dust 後處理被歸零，原始 Score 仍可能保留過檢。
            anomaly_map, configured_exclude_weighted = self._mask_tile_configured_exclude_regions(
                tile, anomaly_map, model_id=model_id
            )

            # 邊緣衰減：過濾光影假陽性 (debug 模式可覆寫數值)
            edge_margin = self.config.edge_margin_px if edge_margin_override is None else edge_margin_override
            if edge_margin > 0:
                # 收集此 tile 需要衰減的方向
                cfg_sides = self.config.edge_margin_sides
                sides = []
                if tile.is_top_edge and cfg_sides.get('top', False):
                    sides.append('top')
                if tile.is_bottom_edge and cfg_sides.get('bottom', False):
                    sides.append('bottom')
                if tile.is_left_edge and cfg_sides.get('left', False):
                    sides.append('left')
                if tile.is_right_edge and cfg_sides.get('right', False):
                    sides.append('right')

                if sides:
                    edge_margin_sides = sides
                    # 將 margin_px 按 anomaly_map 實際尺寸縮放
                    scale = anomaly_map.shape[0] / tile.height
                    scaled_margin = int(edge_margin * scale)
                    anomaly_map = self._apply_edge_margin(anomaly_map, scaled_margin, sides=sides)
            tile.pre_decay_map_max = pre_decay_max
            tile.score_edge_margin_sides = ",".join(edge_margin_sides)
        
        # 如果有遮罩或邊緣衰減，使用衰減比率調整分數（保持與 anomalib pred_score 相同尺度）
        actual_edge_margin = self.config.edge_margin_px if edge_margin_override is None else edge_margin_override
        has_edge_margin = actual_edge_margin > 0 and any([
            tile.is_top_edge and self.config.edge_margin_sides.get('top', False),
            tile.is_bottom_edge and self.config.edge_margin_sides.get('bottom', False),
            tile.is_left_edge and self.config.edge_margin_sides.get('left', False),
            tile.is_right_edge and self.config.edge_margin_sides.get('right', False),
        ])
        need_recalc = (
            (tile.mask is not None)
            or has_edge_margin
            or mark_masked
            or corner_masked
            or configured_exclude_weighted
        )
        if need_recalc and anomaly_map is not None:
            post_decay_max = float(np.max(anomaly_map))

            if tile.mark_patch_score_applied and mark_masked:
                # MARK 已在 PatchCore 選分階段排除；只保留其他遮罩與邊緣的衰減，
                # 不可再用 MARK 前後 max 比率降低整張 Tile。
                before_mark_ratio = (
                    pre_mark_map_max / pre_decay_max if pre_decay_max > 0 else 1.0
                )
                after_mark_ratio = (
                    post_decay_max / post_mark_map_max if post_mark_map_max > 0 else 1.0
                )
                decay_ratio = before_mark_ratio * after_mark_ratio
                pred_score = pred_score * decay_ratio
                logger.info(
                    "[MARK_PATCH_SCORE] method=patchcore_valid_patch_v1 "
                    "model=%s raw=%.4f mark_free=%.4f final=%.4f threshold=%.4f "
                    "patches=%d/%d peak=(%d,%d) other_decay=%.4f elapsed_ms=%.2f",
                    mark_patch_detail.get("model", "unknown"),
                    raw_pred_score,
                    tile.mark_patchcore_score,
                    pred_score,
                    active_threshold,
                    tile.mark_patch_valid_count,
                    tile.mark_patch_total_count,
                    tile.mark_patch_peak_x,
                    tile.mark_patch_peak_y,
                    decay_ratio,
                    float(mark_patch_detail.get("elapsed_ms", 0.0)),
                )
            elif pre_decay_max > 0:
                # 無 Patch evidence 時保留既有比例計分，避免不支援模型中斷推論。
                decay_ratio = post_decay_max / pre_decay_max
                pred_score = pred_score * decay_ratio
                if mark_masked:
                    tile.mark_patch_score_reason = str(
                        mark_patch_detail.get("reason") or "unknown_fallback"
                    )
                    logger.info(
                        "[MARK_PATCH_SCORE] method=max_ratio fallback=%s raw=%.4f "
                        "final=%.4f threshold=%.4f",
                        tile.mark_patch_score_reason,
                        raw_pred_score,
                        pred_score,
                        active_threshold,
                    )
            else:
                decay_ratio = 0.0
                pred_score = 0.0
            tile.post_decay_map_max = post_decay_max
            tile.score_decay_ratio = decay_ratio
        elif anomaly_map is not None:
            tile.post_decay_map_max = float(np.max(anomaly_map))
            tile.score_decay_ratio = 1.0
        
        return pred_score, anomaly_map

    def _detect_bright_spots(self, tile: 'TileInfo') -> Tuple[float, Optional[np.ndarray]]:
        """
        B0F00000 專用：偵測黑色背景上的異常亮點。

        取代 PatchCore 推論，用於無訓練模型的圖片。
        使用局部對比偵測（median filter 背景估計 → 差異計算 → 閾值判定），
        同時保留絕對亮度上限保護。

        偵測邏輯：
          1. median filter 估計局部背景
          2. 原圖 - 背景 = 局部差異（比背景亮的部分）
          3. 差異 > diff_threshold → 候選亮點
          4. 連通分量面積篩選 ≥ min_area
          5. 絕對亮度 > bright_spot_threshold 的也直接納入（上限保護）

        Args:
            tile: TileInfo 物件

        Returns:
            (score, binary_map) - score: 0.0 (無亮點) 或 1.0 (有亮點)
                                  binary_map: 二值化結果 (uint8, 0/255)
        """
        abs_threshold = self.config.bright_spot_threshold
        min_area = self.config.bright_spot_min_area
        median_kernel = self.config.bright_spot_median_kernel
        diff_threshold = self.config.bright_spot_diff_threshold

        img = tile.image
        if img is None:
            return 0.0, None

        # 灰階化
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        mk = clamp_median_kernel(median_kernel, min(gray.shape[:2]) - 1)

        # 背景估計 → 局部差異
        bg = cv2.medianBlur(gray, mk)
        diff = cv2.subtract(gray, bg)  # 只取比背景亮的部分 (saturate at 0)

        # 局部對比閾值：差異超過 diff_threshold 的為候選亮點
        _, binary_diff = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

        # 絕對亮度上限保護：超過 abs_threshold 的直接納入
        _, binary_abs = cv2.threshold(gray, abs_threshold, 255, cv2.THRESH_BINARY)

        # 合併兩種偵測結果
        binary = cv2.bitwise_or(binary_diff, binary_abs)

        # 與 PatchCore Tile 共用四角不檢測設定。
        binary, _ = self._mask_tile_corner_exclusion(tile, binary)

        # 如果 tile 有 mask（排除區域），套用 mask
        if tile.mask is not None:
            mask_resized = cv2.resize(tile.mask, (binary.shape[1], binary.shape[0]))
            binary = cv2.bitwise_and(binary, mask_resized)

        # 連通分量分析，過濾小面積雜訊
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        filtered_binary = np.zeros_like(binary)
        has_bright_spot = False
        max_component_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            max_component_area = max(max_component_area, area)
            if area >= min_area:
                filtered_binary[labels == i] = 255
                has_bright_spot = True

        score = 1.0 if has_bright_spot else 0.0

        # 轉為 float anomaly_map 格式（與 PatchCore 輸出格式相容）
        anomaly_map = filtered_binary.astype(np.float32) / 255.0

        # 標記此 tile 為二值化偵測模式，並儲存偵測統計供 heatmap header 顯示
        tile.is_bright_spot_detection = True
        tile.bright_spot_max_diff = int(diff.max())
        tile.bright_spot_diff_threshold = diff_threshold
        tile.bright_spot_area = int(np.sum(filtered_binary > 0))
        tile.bright_spot_min_area = min_area

        # 偵測結果 log
        max_pixel_val = int(gray.max())
        max_diff_val = int(diff.max())
        bright_pixel_count = int(np.sum(filtered_binary > 0))
        raw_bright_count = int(np.sum(binary > 0))
        aoi_suffix = self._format_aoi_tile_log_suffix(tile)
        if has_bright_spot:
            print(f"  💡 B0F 偵測 Tile@({tile.x},{tile.y}){aoi_suffix}: 發現亮點 ({bright_pixel_count} px), "
                  f"max_diff={max_diff_val}, diff_thr={diff_threshold}, abs_thr={abs_threshold}, "
                  f"max_pixel={max_pixel_val}, median_k={mk}")
        else:
            print(f"  💡 B0F 偵測 Tile@({tile.x},{tile.y}){aoi_suffix}: 未發現亮點, "
                  f"max_diff={max_diff_val}, diff_thr={diff_threshold}, abs_thr={abs_threshold}, "
                  f"max_pixel={max_pixel_val}, raw_bright={raw_bright_count} px, "
                  f"max_component={max_component_area} px (min_area={min_area})")

        return score, anomaly_map

    def run_inference(self, result: ImageResult, progress_callback=None,
                      inferencer=None, threshold: Optional[float] = None,
                      edge_margin_override: Optional[int] = None,
                      patchcore_overrides: Optional[Dict[str, Any]] = None,
                      model_id: Optional[str] = None) -> ImageResult:
        """
        對預處理結果執行推論
        
        Args:
            result: preprocess_image 的結果
            progress_callback: 進度回呼函數 (current, total)
            inferencer: 指定的 inferencer 物件，若為 None 使用 self.inferencer
            threshold: 指定的閾值，若為 None 使用 self.threshold
            model_id: 機種名稱，用於推導產品解析度 (例如 'H', 'J')
            
        Returns:
            更新後的 ImageResult（包含異常 tile 資訊）
        """
        active_inferencer = inferencer or self.inferencer
        if active_inferencer is None:
            raise RuntimeError("模型尚未載入，請在初始化時指定 model_path")
        
        active_threshold = threshold if threshold is not None else self.threshold
        
        inference_start = time.time()
        anomaly_tiles = []
        total = len(result.tiles)

        # 逐 tile 推論 (fp16 KNN 加速已在 model 載入時 patch)
        # 注意: batch 推論在 PatchCore 反而更慢 (KNN 距離矩陣隨 batch 線性膨脹)
        for i, tile in enumerate(result.tiles):
            if progress_callback:
                progress_callback(i + 1, total)

            score, anomaly_map = self.predict_tile(tile, inferencer=active_inferencer, edge_margin_override=edge_margin_override, patchcore_overrides=patchcore_overrides, threshold=active_threshold, model_id=model_id)
            tile.score_threshold = active_threshold
            if tile.is_aoi_coord_tile and score <= 1e-9:
                print(
                    f"    [score_diag] AOI Tile@({tile.x},{tile.y}) "
                    f"raw={tile.raw_pred_score:.4f} preMax={tile.pre_decay_map_max:.4f} "
                    f"postMax={tile.post_decay_map_max:.4f} decay={tile.score_decay_ratio:.4f} "
                    f"mask={tile.score_mask_valid_ratio:.3f} edge={tile.score_edge_margin_sides or '-'}"
                )

            if score >= active_threshold:
                anomaly_tiles.append((tile, score, anomaly_map))
            elif tile.is_aoi_coord_tile:
                # AOI 座標 tile 即使低於閾值也保留，供追蹤查看
                tile.is_aoi_coord_below_threshold = True
                anomaly_tiles.append((tile, score, anomaly_map))

        # 執行傳統 CV 邊緣檢查
        # 如果 edge_inspector 啟用，並且我們有 raw_bounds
        if getattr(self, "edge_inspector", None) and self.edge_inspector.config.enabled and result.raw_bounds:
            try:
                # 取得產品解析度代碼 (e.g. "H", "J")
                resolution_code = "UNKNOWN"
                if model_id and len(model_id) >= 6:
                    resolution_code = model_id[5].upper()
                
                # 切換 active zones 為當前產品
                self.edge_inspector.config.set_active_zones_for_product(resolution_code)
                
                # 重新讀取原圖 (全尺寸) 給 CV 處理，因為它需要高解析度才能看清楚
                # 如果 cv2 記憶體太大，可以在 preprocess 前把 raw cv_image 傳過來，但此處再次讀取較安全
                full_image = self._read_detection_image(result.image_path)
                if full_image is not None:
                    # CV 內部處理需要時間，記錄一下
                    cv_start = time.time()
                    edge_defects = self.edge_inspector.inspect(full_image, result.raw_bounds)
                    result.edge_defects = edge_defects
                    logger.debug(f"CV Edge Inspection: 找到 {len(edge_defects)} 個邊緣異常，耗時 {time.time() - cv_start:.3f}s")
            except Exception as e:
                logger.error(f"CV 邊緣檢查失敗 {result.image_path.name}: {e}", exc_info=True)
        
        # 更新結果
        result.anomaly_tiles = anomaly_tiles
        result.inference_time = time.time() - inference_start

        return result

    def run_inference_v2_single_image(
        self,
        image_path: Path,
        threshold: Optional[float] = None,
        edge_margin_override: Optional[int] = None,
        patchcore_overrides: Optional[Dict[str, Any]] = None,
        otsu_offset_override: Optional[int] = None,
    ) -> Optional[ImageResult]:
        """新架構單圖預處理 + per-tile zone-aware 推論 (debug 用)。

        對應 v1 的 preprocess_image + run_inference 雙呼叫，新架構單圖路徑用此。
        Tile zone 來自 capi_preprocess.preprocess_panel_image 的 polygon 分類，
        每個 tile 依 zone (inner/edge) 走 _get_model_for(machine_id, lighting, zone)。

        Args:
            image_path: 圖片絕對路徑
            threshold: 若 None，inner/edge 各自走 config.threshold_mapping；
                       若指定則 inner/edge 同用此值 (debug UI 拖拉)
            edge_margin_override / patchcore_overrides / otsu_offset_override:
                與 v1 run_inference 同義

        Returns:
            ImageResult (與 v1 同格式)；若 model_mapping 缺對應 lighting 之 inner/edge
            或圖片載入/前處理失敗，回傳 None。
        """
        from capi_preprocess import preprocess_panel_image, PreprocessConfig

        lighting = self._get_image_prefix(image_path.name)

        lighting_map = self.config.model_mapping.get(lighting, {})
        if not isinstance(lighting_map, dict) or "inner" not in lighting_map or "edge" not in lighting_map:
            return None

        pre_cfg = PreprocessConfig(
            tile_size=self.config.tile_size,
            tile_stride=getattr(self.config, "tile_stride", self.config.tile_size),
            otsu_offset=otsu_offset_override if otsu_offset_override is not None else self.config.otsu_offset,
            enable_panel_polygon=self.config.enable_panel_polygon,
            edge_threshold_px=self.config.edge_threshold_px,
            image_preprocess_pipeline=getattr(self.config, "image_preprocess_pipeline", []),
            image_preprocess_pipelines=getattr(self.config, "image_preprocess_pipelines", {}),
            preprocess_after_tiling=getattr(self.config, "preprocess_after_tiling", False),
            product_resolution=self._product_resolution(),
            rotate_180=getattr(self, "_rotate_detection_images_180", False),
        )
        pre_result = preprocess_panel_image(image_path, lighting, pre_cfg)

        if pre_result.foreground_bbox == (0, 0, 0, 0):
            return None

        raw_img = self._read_detection_image(image_path)
        if raw_img is None:
            return None
        img_h, img_w = raw_img.shape[:2]

        bbox = pre_result.foreground_bbox
        polygon = pre_result.panel_polygon

        lighting_thr = self.config.threshold_mapping.get(lighting, {})
        if isinstance(lighting_thr, dict):
            cfg_inner_thr = float(lighting_thr.get("inner", 0.75))
            cfg_edge_thr = float(lighting_thr.get("edge", 0.75))
        else:
            cfg_inner_thr = cfg_edge_thr = float(lighting_thr) if lighting_thr else 0.75

        if threshold is not None:
            inner_thr = float(threshold)
            edge_thr = float(threshold)
        else:
            inner_thr = cfg_inner_thr
            edge_thr = cfg_edge_thr

        ts = self.config.tile_size
        bottom_y_threshold = bbox[3] - ts
        right_x_threshold = bbox[2] - ts
        tile_infos: List[TileInfo] = []
        for tr in pre_result.tiles:
            ti = TileInfo(
                tile_id=tr.tile_id,
                x=tr.x1,
                y=tr.y1,
                width=ts,
                height=ts,
                image=tr.image,
                original_image=tr.original_image,
                mask=tr.mask,
                is_bottom_edge=tr.y1 >= bottom_y_threshold,
                is_top_edge=tr.y1 <= bbox[1],
                is_left_edge=tr.x1 <= bbox[0],
                is_right_edge=tr.x1 >= right_x_threshold,
                zone=tr.zone,
            )
            tile_infos.append(ti)

        image_result = ImageResult(
            image_path=image_path,
            image_size=(img_w, img_h),
            otsu_bounds=bbox,
            exclusion_regions=[],
            tiles=tile_infos,
            excluded_tile_count=0,
            processed_tile_count=len(tile_infos),
            processing_time=0.0,
            anomaly_tiles=[],
            raw_bounds=bbox,
            panel_polygon=polygon,
            inference_time=0.0,
            preprocess_steps=list(getattr(pre_result, "preprocess_steps", []) or []),
            preprocess_total_ms=float(getattr(pre_result, "preprocess_total_ms", 0.0) or 0.0),
        )

        t_infer_start = time.time()
        anomaly_tiles: List[Tuple[TileInfo, float, Optional[np.ndarray]]] = []
        for ti in tile_infos:
            zone = ti.zone if ti.zone in ("inner", "edge") else "inner"
            active_thr = inner_thr if zone == "inner" else edge_thr
            ti.score_threshold = active_thr
            try:
                model = self._get_model_for(self.config.machine_id, lighting, zone)
            except Exception as exc:
                raise RuntimeError(
                    f"[v2-debug] {lighting}/{zone} 模型載入失敗: {exc}"
                ) from exc

            score, anomaly_map = self.predict_tile(
                ti,
                inferencer=model,
                edge_margin_override=edge_margin_override,
                patchcore_overrides=patchcore_overrides,
                threshold=active_thr,
                model_id=self.config.machine_id,
            )
            if score >= active_thr:
                if anomaly_map is not None:
                    peak_idx = int(np.argmax(anomaly_map))
                    ah, aw = anomaly_map.shape[:2]
                    ti.anomaly_peak_x = ti.x + peak_idx % aw
                    ti.anomaly_peak_y = ti.y + peak_idx // aw
                anomaly_tiles.append((ti, score, anomaly_map))

        image_result.anomaly_tiles = anomaly_tiles
        image_result.inference_time = time.time() - t_infer_start
        return image_result

    def get_anomaly_summary(self, result: ImageResult) -> Dict[str, Any]:
        """取得異常摘要"""
        # 計算真實的 CV 邊緣異常數 (排除疑似灰塵)
        real_edge_defects = [ed for ed in getattr(result, 'edge_defects', []) if not getattr(ed, 'is_suspected_dust_or_scratch', False)]
        
        if not result.anomaly_tiles and not real_edge_defects:
            return {
                "is_anomaly": False,
                "anomaly_count": 0,
                "max_score": 0.0,
                "anomaly_positions": [],
                "cv_edge_anomaly_count": 0,
            }
        
        scores = [score for _, score, _ in result.anomaly_tiles]
        positions = [(tile.x, tile.y, tile.width, tile.height) for tile, _, _ in result.anomaly_tiles]
        
        return {
            "is_anomaly": True,
            "anomaly_count": len(result.anomaly_tiles),
            "max_score": max(scores) if scores else 0.0,
            "avg_score": (sum(scores) / len(scores)) if scores else 0.0,
            "anomaly_positions": positions,
            "cv_edge_anomaly_count": len(real_edge_defects),
        }
    
    def visualize_preprocessing(
        self, 
        image_path: Path, 
        result: ImageResult,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """視覺化預處理結果"""
        image = self._read_detection_image(image_path, cv2.IMREAD_COLOR)
        vis = image.copy()
        
        # Otsu 邊界（藍色）
        x1, y1, x2, y2 = result.otsu_bounds
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 10)
        cv2.putText(vis, "Otsu Bounds", (x1 + 10, y1 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 4)

        # Panel polygon（紅色）
        if result.panel_polygon is not None:
            poly_int = result.panel_polygon.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [poly_int], True, (0, 0, 255), 10)
            cv2.putText(vis, "Panel Polygon", (x1 + 10, y1 + 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

        # 顯示裁切區域（灰色斜線或半透明）
        if result.cropped_region:
            cx1, cy1, cx2, cy2 = result.cropped_region
            # 畫半透明紅色區域
            overlay = vis.copy()
            cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
            # 畫邊框
            cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (100, 100, 100), 5)
            # 文字
            text_y = cy1 + (cy2 - cy1) // 2
            cv2.putText(vis, "BOTTOM CROP", (cx1 + 20, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 4)
        
        # 排除區域視覺化已移除 - 根據使用者要求，排除區域不需要在 Overview 上標出
        # 被跳過的 tiles 也不會顯示在這裡，因為它們根本沒有被生成
        
        # Tile 網格：所有 tiles 都用綠色（排除區域的 tiles 已被完全跳過）
        for tile in result.tiles:
            cv2.rectangle(vis, (tile.x, tile.y), 
                         (tile.x + tile.width, tile.y + tile.height), 
                         (0, 255, 0), 3)  # 綠色
        
        # 圖片資訊
        info_text = f"Total Tiles: {result.processed_tile_count} | Excluded: {result.excluded_tile_count}"
        cv2.putText(vis, info_text, (x1 + 10, y2 - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
        
        if output_path:
            # 縮小後儲存
            scale = 0.3
            vis_small = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))
            cv2.imwrite(str(output_path), vis_small)
        
        return vis

    
    def _use_omit_pixel_grid_filter(self) -> bool:
        """Return whether opt-in OMIT pixel-grid smoothing is enabled."""
        return bool(getattr(self.config, "dust_pixel_grid_filter_enabled", False))

    def check_dust_or_scratch_feature(
        self,
        image: np.ndarray,
        extension_override: Optional[int] = None,
        product_resolution: Optional[Tuple[int, int]] = None,
    ) -> tuple:
        """
        進階灰塵/刮痕偵測 — 使用 CLAHE 增強 + Otsu 自適應閾值 + 形態學 + 面積篩選
        
        流程:
          1. CLAHE 局部對比增強（偵測微弱灰塵）
          2. Otsu 自適應二值化（自動判定最佳閾值）
          3. 形態學開運算去雜訊 + 膨脹延伸
          4. Connected Components 面積篩選
          5. 分析顆粒 vs 刮傷（寬高比判定）
        
        Args:
            image: OMIT 圖片裁切區域 (BGR 或灰階)
            extension_override: 覆寫 Config 中的 dust_extension 設定
            product_resolution: 保留呼叫相容性；像素紋理平滑不再依產品解析度限制
            
        Returns:
            (is_dust, dust_mask, bright_ratio, detail_text)
            - is_dust: 是否偵測到灰塵/刮痕
            - dust_mask: 灰塵區域遮罩 (uint8, 255=灰塵)
            - bright_ratio: 灰塵面積佔比
            - detail_text: 判定詳細說明
        """
        if image is None or image.size == 0:
            return False, None, 0.0, "No image"
            
        # 讀取配置參數
        fallback_threshold = self.config.dust_brightness_threshold
        area_min = self.config.dust_area_min
        area_max = self.config.dust_area_max
        extension = self.config.dust_extension if extension_override is None else extension_override
        
        # Step 1: 轉灰階
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 開關啟用時，先在 CLAHE 放大局部對比前做週期紋理平滑，
        # 避免產品像素格被誤當成灰塵/刮痕；不限制產品解析度。
        pixel_grid_filter_active = self._use_omit_pixel_grid_filter()
        pixel_grid_blur_kernel = 0
        if pixel_grid_filter_active:
            requested_kernel = max(
                3,
                int(getattr(self.config, "dust_pixel_grid_blur_kernel", 7)),
            )
            if requested_kernel % 2 == 0:
                requested_kernel += 1
            max_kernel = min(gray.shape[:2])
            if max_kernel % 2 == 0:
                max_kernel -= 1
            pixel_grid_blur_kernel = min(requested_kernel, max_kernel)
            if pixel_grid_blur_kernel >= 3:
                valid_mask = (gray > 0).astype(np.float32)
                blurred_values = cv2.GaussianBlur(
                    gray.astype(np.float32),
                    (pixel_grid_blur_kernel, pixel_grid_blur_kernel),
                    0,
                )
                blurred_weights = cv2.GaussianBlur(
                    valid_mask,
                    (pixel_grid_blur_kernel, pixel_grid_blur_kernel),
                    0,
                )
                gray = np.where(
                    valid_mask > 0,
                    blurred_values / np.maximum(blurred_weights, 1e-6),
                    0,
                )
                gray = np.clip(np.rint(gray), 0, 255).astype(np.uint8)
        
        # Step 2: CLAHE 局部對比增強 — 強化微弱灰塵的可見度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Step 2.5: Top-Hat 變換 (去除大面積高光背景)
        # 邊緣區域常包含大片高光背景(如載台/膠帶)，會嚴重干擾 Otsu 閾值，導致玻璃上的微弱灰塵被忽略
        # 使用 45x45 核做開運算估計背景 (足以覆蓋多數灰塵，area_max 一般<=1000 => radius~18)
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        bg_est = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel_bg)
        # 相減保留局部亮點 (Top-Hat)
        tophat = cv2.subtract(enhanced, bg_est)
        
        # Step 3: 二值化
        # Top-Hat 後背景趨近於 0，使用 Otsu 可能因為單峰分佈而失真，
        # 故以 config 中的 fallback_threshold 為基準，並取 otsu 為輔
        otsu_thresh, _ = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 若 Otsu 閾值異常高或低，限制在合理範圍內
        threshold_floor = self.config.dust_threshold_floor
        adaptive_thr = min(max(otsu_thresh, threshold_floor), fallback_threshold)
        
        _, binary = cv2.threshold(tophat, adaptive_thr, 255, cv2.THRESH_BINARY)
        used_threshold = adaptive_thr
        
        # 合理性檢查：若前景佔比仍過高，可能全是雜訊，用更嚴格的閾值
        # 但需注意：表面嚴重刮傷的面板確實可能有 15-25% 的灰塵/刮痕前景
        MAX_REASONABLE_RATIO = 0.25
        initial_ratio = float(np.sum(binary > 0)) / binary.size if binary.size > 0 else 0.0
        if initial_ratio > MAX_REASONABLE_RATIO:
            # 嚴格閾值前，先保護寬鬆閾值下的有效灰塵/刮痕特徵
            feature_preserved = np.zeros_like(binary)
            _n, _labels, _stats, _ = cv2.connectedComponentsWithStats(binary)
            for _i in range(1, _n):
                _area = _stats[_i, cv2.CC_STAT_AREA]
                _w = _stats[_i, cv2.CC_STAT_WIDTH]
                _h = _stats[_i, cv2.CC_STAT_HEIGHT]
                if _area < area_min or _area > area_max:
                    continue
                _aspect = max(_w, _h) / (min(_w, _h) + 1e-5)
                # 保護線性刮痕(aspect>5) 以及有一定面積的灰塵顆粒
                if _aspect > 5 or _area >= area_min * 5:
                    feature_preserved[_labels == _i] = 255

            # 使用 p95 作為嚴格閾值，但限制最高不超過 adaptive_thr 的 2 倍
            # 避免極亮刮痕導致閾值飆到 170+ 而漏掉中等亮度灰塵
            p95 = float(np.percentile(tophat, 95))
            strict_thr_cap = adaptive_thr * 2.0
            strict_thr = min(max(adaptive_thr, p95), strict_thr_cap)
            _, binary = cv2.threshold(tophat, strict_thr, 255, cv2.THRESH_BINARY)
            used_threshold = strict_thr

            # 合併保護的特徵回 binary
            binary = cv2.bitwise_or(binary, feature_preserved)
        
        # Step 3.5: 明顯亮區救回 — Top-Hat 會吃掉寬度>kernel 的大面積污染/刮痕
        # 對 CLAHE 增強後的原圖做高閾值直接檢測，把肉眼明顯的亮區補回來
        bright_rescue_thr = self.config.dust_bright_rescue_threshold
        if bright_rescue_thr > 0:
            _, bright_binary = cv2.threshold(enhanced, bright_rescue_thr, 255, cv2.THRESH_BINARY)
            binary = cv2.bitwise_or(binary, bright_binary)

        # Step 4: 形態學處理
        # 開運算：去除小噪點
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
        
        # 膨脹：延伸灰塵區域（對應廠商「延伸」概念）
        if extension > 0:
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                      (extension * 2 + 1, extension * 2 + 1))
            binary = cv2.dilate(binary, dilate_kernel, iterations=1)
        
        # Step 5: Connected Components 面積篩選
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        dust_mask = np.zeros_like(binary)
        particle_count = 0
        scratch_count = 0
        total_dust_area = 0
        
        for i in range(1, num_labels):  # 跳過背景 (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 面積篩選
            if area < area_min or area > area_max:
                continue
            
            # 寫入灰塵遮罩
            dust_mask[labels == i] = 255
            total_dust_area += area
            
            # 形態分類：寬高比 > 5 → 刮傷，否則 → 顆粒
            aspect = max(w, h) / (min(w, h) + 1e-5)
            if aspect > 5:
                scratch_count += 1
            else:
                particle_count += 1
        
        # Step 6: 暗色顆粒偵測 — 偵測暗色 MARK 等暗色圖案
        # 某些機種 MARK 樣式偏黑，在 OMIT 圖上呈現暗色顆粒
        # 使用 THRESH_BINARY_INV 偵測低於背景的暗色區域
        dark_particle_count = 0
        dark_scratch_count = 0
        dark_total_area = 0
        
        if getattr(self.config, 'dust_detect_dark_particles', True):
            # 計算背景統計（排除全黑邊界像素）
            non_zero_pixels = gray[gray > 0]
            if len(non_zero_pixels) > 100:  # 確保有足夠像素做統計
                bg_median = float(np.median(non_zero_pixels))
                # 暗色閾值：取低 1st percentile 或 背景中位數的一半，取較大者
                p1 = float(np.percentile(non_zero_pixels, 1))
                dark_threshold = max(p1, bg_median * 0.5)
                
                # 只在背景中位數夠亮時才偵測暗色顆粒（避免全暗圖誤判）
                if bg_median > 20:
                    _, dark_binary = cv2.threshold(gray, int(dark_threshold), 255, cv2.THRESH_BINARY_INV)
                    
                    # 排除全黑像素（圖的邊界/padding 區域）
                    dark_binary[gray == 0] = 0
                    
                    # 合理性檢查
                    dark_ratio = float(np.sum(dark_binary > 0)) / dark_binary.size if dark_binary.size > 0 else 0.0
                    if dark_ratio <= MAX_REASONABLE_RATIO:
                        # 形態學處理
                        dark_binary = cv2.morphologyEx(dark_binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
                        if extension > 0:
                            dark_binary = cv2.dilate(dark_binary, dilate_kernel, iterations=1)
                        
                        # Connected Components 面積篩選
                        d_num_labels, d_labels, d_stats, _ = cv2.connectedComponentsWithStats(dark_binary)
                        
                        for i in range(1, d_num_labels):
                            d_area = d_stats[i, cv2.CC_STAT_AREA]
                            d_w = d_stats[i, cv2.CC_STAT_WIDTH]
                            d_h = d_stats[i, cv2.CC_STAT_HEIGHT]
                            
                            if d_area < area_min or d_area > area_max:
                                continue
                            
                            # 合併至灰塵遮罩
                            dust_mask[d_labels == i] = 255
                            dark_total_area += d_area
                            
                            d_aspect = max(d_w, d_h) / (min(d_w, d_h) + 1e-5)
                            if d_aspect > 5:
                                dark_scratch_count += 1
                            else:
                                dark_particle_count += 1
                        
                        if dark_particle_count + dark_scratch_count > 0:
                            logging.debug(f"    暗色顆粒偵測: P:{dark_particle_count} S:{dark_scratch_count} Area:{dark_total_area} (Thr:{dark_threshold:.0f}, Median:{bg_median:.0f})")

        # Step 6.5: 低對比氣泡偵測 — 抓大面積柔邊亮/暗斑，避免只標到氣泡內的小點
        bubble_count = 0
        bubble_total_area = 0
        bubble_detection_enabled = getattr(self.config, 'dust_detect_bubbles_enabled', False)
        if bubble_detection_enabled:
            accepted_bubble_mask = self._detect_bubble_mask(
                gray,
                area_min,
                area_max,
                extension,
            )
            if np.any(accepted_bubble_mask > 0):
                b_count, _ = cv2.connectedComponents(accepted_bubble_mask, connectivity=8)
                bubble_count = max(0, b_count - 1)
                bubble_total_area = int(np.count_nonzero(accepted_bubble_mask > 0))
                dust_mask[accepted_bubble_mask > 0] = 255
        
        # 合併計數
        total_particle = particle_count + dark_particle_count
        total_scratch = scratch_count + dark_scratch_count
        total_area = total_dust_area + dark_total_area + bubble_total_area
        
        # 計算灰塵面積佔比
        bright_ratio = float(np.sum(dust_mask > 0)) / dust_mask.size if dust_mask.size > 0 else 0.0
        is_dust = (total_particle + total_scratch + bubble_count) > 0
        
        dark_info = f" DkP:{dark_particle_count} DkS:{dark_scratch_count}" if (dark_particle_count + dark_scratch_count) > 0 else ""
        bubble_info = f" Bub:{bubble_count}" if bubble_detection_enabled else ""
        detail_text = (f"Thr:{used_threshold:.0f} P:{particle_count} S:{scratch_count} "
                       f"Area:{total_area} Ratio:{bright_ratio:.4f}{dark_info}{bubble_info}")
        if pixel_grid_filter_active:
            detail_text += f" PxGridBlur:{pixel_grid_blur_kernel}"
        
        return is_dust, dust_mask, bright_ratio, detail_text

    def _detect_bubble_mask(
        self,
        gray: np.ndarray,
        area_min: int,
        area_max: int,
        extension: int,
        reference_area: Optional[int] = None,
    ) -> np.ndarray:
        """Detect regular and large low-contrast bubbles in one tile."""
        accepted_bubble_mask = np.zeros_like(gray, dtype=np.uint8)
        if gray is None or gray.size == 0:
            return accepted_bubble_mask

        non_zero_pixels = gray[gray > 0]
        if len(non_zero_pixels) <= 100 or float(np.median(non_zero_pixels)) <= 20:
            return accepted_bubble_mask

        min_dim = min(gray.shape[:2])
        blur_k = min(101, min_dim - 1 if min_dim % 2 == 0 else min_dim)
        if blur_k % 2 == 0:
            blur_k -= 1
        if blur_k < 15:
            return accepted_bubble_mask

        ref_area = gray.size if reference_area is None else max(1, int(reference_area))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        open_kernel_bubble = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Smaller bright regions belong to the particle detector; promoting them to
        # bubbles would convex-hull fill the background around the actual white dust.
        bubble_min_area = max(area_min * 20, int(ref_area * 0.003))
        bubble_max_area = min(area_max, int(ref_area * 0.08))
        bubble_fill_max_area = min(
            area_max,
            max(int(ref_area * 0.08), bubble_min_area * 30),
        )
        border_margin = 6
        surface_edge_y = None
        if min_dim >= 64:
            row_mean = gray.astype(np.float32).mean(axis=1)
            row_mean = cv2.GaussianBlur(row_mean.reshape(-1, 1), (1, 9), 0).ravel()
            row_grad = np.gradient(row_mean)
            search_start = int(gray.shape[0] * 0.45)
            search_end = max(search_start + 1, gray.shape[0] - 10)
            edge_y = search_start + int(np.argmax(row_grad[search_start:search_end]))
            if float(row_grad[edge_y]) >= 10.0:
                surface_edge_y = edge_y

        bubble_sources = [(gray, 4.0, 4.0)]
        if min_dim >= 64:
            smooth_gray = cv2.GaussianBlur(cv2.medianBlur(gray, 5), (9, 9), 0)
            bubble_sources.append((smooth_gray, 2.0, 2.0))

        for bubble_source, delta_threshold, mean_threshold in bubble_sources:
            bubble_bg = cv2.GaussianBlur(bubble_source, (blur_k, blur_k), 0)
            for delta in (
                bubble_bg.astype(np.int16) - bubble_source.astype(np.int16),
                bubble_source.astype(np.int16) - bubble_bg.astype(np.int16),
            ):
                delta[gray == 0] = 0
                bubble_binary = (delta >= delta_threshold).astype(np.uint8) * 255
                bubble_binary = cv2.morphologyEx(
                    bubble_binary,
                    cv2.MORPH_CLOSE,
                    close_kernel,
                    iterations=1,
                )
                bubble_binary = cv2.morphologyEx(
                    bubble_binary,
                    cv2.MORPH_OPEN,
                    open_kernel_bubble,
                    iterations=1,
                )

                b_num_labels, b_labels, b_stats, _ = \
                    cv2.connectedComponentsWithStats(bubble_binary)
                for i in range(1, b_num_labels):
                    b_area = int(b_stats[i, cv2.CC_STAT_AREA])
                    b_x = int(b_stats[i, cv2.CC_STAT_LEFT])
                    b_y = int(b_stats[i, cv2.CC_STAT_TOP])
                    b_w = int(b_stats[i, cv2.CC_STAT_WIDTH])
                    b_h = int(b_stats[i, cv2.CC_STAT_HEIGHT])

                    if b_area < bubble_min_area or b_area > bubble_max_area:
                        continue
                    if (
                        b_x <= border_margin
                        or b_y <= border_margin
                        or b_x + b_w >= gray.shape[1] - border_margin
                        or b_y + b_h >= gray.shape[0] - border_margin
                    ):
                        continue
                    if surface_edge_y is not None and b_y >= surface_edge_y - 20:
                        continue

                    b_aspect = max(b_w, b_h) / (min(b_w, b_h) + 1e-5)
                    b_fill_ratio = b_area / max(1, b_w * b_h)
                    if b_aspect > 3.5 or b_fill_ratio < 0.25:
                        continue

                    component_mask = b_labels == i
                    component_mean_delta = float(np.mean(delta[component_mask]))
                    if component_mean_delta < mean_threshold:
                        continue

                    component_u8 = component_mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(
                        component_u8,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    bubble_mask = np.zeros_like(component_u8)
                    if contours:
                        outline = max(contours, key=cv2.contourArea)
                        outline_perimeter = float(cv2.arcLength(outline, True))
                        outline_circularity = (
                            4.0 * np.pi * float(cv2.contourArea(outline))
                            / max(1.0, outline_perimeter * outline_perimeter)
                        )
                        # The relaxed striped-surface branch can connect weak background
                        # texture into an irregular blob that its convex hull then overfills.
                        # Keep strong broken-ring bubbles and compact elongated surface
                        # bubbles, but reject weak non-round background blobs.
                        weak_irregular_blob = (
                            component_mean_delta < 3.0
                            and outline_circularity < 0.40
                        )
                        compact_elongated_bubble = b_aspect >= 2.0 and b_fill_ratio >= 0.40
                        if weak_irregular_blob and not compact_elongated_bubble:
                            continue

                        contour_points = np.vstack(contours)
                        if len(contour_points) >= 3:
                            hull = cv2.convexHull(contour_points)
                            cv2.drawContours(
                                bubble_mask,
                                [hull],
                                -1,
                                255,
                                thickness=-1,
                            )
                        else:
                            cv2.drawContours(
                                bubble_mask,
                                contours,
                                -1,
                                255,
                                thickness=-1,
                            )
                    else:
                        bubble_mask = component_u8

                    surface_gap = None
                    if surface_edge_y is not None:
                        surface_gap = surface_edge_y - (b_y + b_h)
                    if (
                        surface_gap is not None
                        and b_w >= b_h * 1.4
                        and b_h * 0.5 <= surface_gap <= b_h
                    ):
                        ellipse_bottom = surface_edge_y - 1
                        ellipse_center = (
                            b_x + b_w // 2,
                            (b_y + ellipse_bottom) // 2,
                        )
                        ellipse_axes = (
                            max(1, b_w // 2),
                            max(1, (ellipse_bottom - b_y) // 2),
                        )
                        cv2.ellipse(
                            bubble_mask,
                            ellipse_center,
                            ellipse_axes,
                            0,
                            0,
                            360,
                            255,
                            thickness=-1,
                        )

                    filled_area = int(np.count_nonzero(bubble_mask > 0))
                    if filled_area < bubble_min_area or filled_area > bubble_fill_max_area:
                        continue

                    if extension > 0:
                        bubble_dilate = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (extension * 2 + 1, extension * 2 + 1),
                        )
                        bubble_mask = cv2.dilate(
                            bubble_mask,
                            bubble_dilate,
                            iterations=1,
                        )

                    accepted_bubble_mask[bubble_mask > 0] = 255

        large_bubble_mask = self._detect_large_surface_bubble_mask(
            gray,
            area_min,
            area_max,
            extension,
            reference_area=ref_area,
        )
        accepted_bubble_mask[large_bubble_mask > 0] = 255
        return accepted_bubble_mask

    def _detect_large_surface_bubble_mask(
        self,
        gray: np.ndarray,
        area_min: int,
        area_max: int,
        extension: int,
        reference_area: Optional[int] = None,
    ) -> np.ndarray:
        """Detect large elongated bubbles that sit directly above the surface edge."""
        bubble_mask = np.zeros_like(gray, dtype=np.uint8)
        if gray is None or gray.size == 0 or min(gray.shape[:2]) < 256:
            return bubble_mask

        row_mean = gray.astype(np.float32).mean(axis=1)
        row_mean = cv2.GaussianBlur(row_mean.reshape(-1, 1), (1, 9), 0).ravel()
        row_grad = np.gradient(row_mean)
        search_start = int(gray.shape[0] * 0.45)
        search_end = max(search_start + 1, gray.shape[0] - 10)
        surface_edge_y = search_start + int(np.argmax(row_grad[search_start:search_end]))
        if float(row_grad[surface_edge_y]) < 10.0:
            return bubble_mask

        ref_area = gray.size if reference_area is None else max(1, int(reference_area))
        base_bubble_min_area = max(area_min * 20, int(ref_area * 0.002))
        large_min_area = max(base_bubble_min_area * 10, int(ref_area * 0.03))
        large_max_area = min(area_max, int(ref_area * 0.25))
        if large_max_area < large_min_area:
            return bubble_mask

        border_margin = 6
        large_source = cv2.GaussianBlur(gray, (15, 15), 0)
        background_bottom = max(border_margin + 1, surface_edge_y - 8)
        background_roi = large_source[
            border_margin:background_bottom,
            border_margin:gray.shape[1] - border_margin,
        ]
        background_pixels = background_roi[background_roi > 0]
        if background_pixels.size < 100:
            return bubble_mask

        large_threshold = float(np.median(background_pixels)) + 9.0
        large_binary = (large_source >= large_threshold).astype(np.uint8) * 255
        large_binary[gray == 0] = 0
        large_binary[surface_edge_y:, :] = 0
        large_binary[:border_margin, :] = 0
        large_binary[:, :border_margin] = 0
        large_binary[:, gray.shape[1] - border_margin:] = 0

        large_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        large_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        large_binary = cv2.morphologyEx(
            large_binary, cv2.MORPH_OPEN, large_open, iterations=1
        )
        large_binary = cv2.morphologyEx(
            large_binary, cv2.MORPH_CLOSE, large_close, iterations=1
        )

        large_num, large_labels, large_stats, _ = \
            cv2.connectedComponentsWithStats(large_binary, connectivity=8)
        for i in range(1, large_num):
            b_x = int(large_stats[i, cv2.CC_STAT_LEFT])
            b_y = int(large_stats[i, cv2.CC_STAT_TOP])
            b_w = int(large_stats[i, cv2.CC_STAT_WIDTH])
            b_h = int(large_stats[i, cv2.CC_STAT_HEIGHT])
            b_area = int(large_stats[i, cv2.CC_STAT_AREA])

            if b_area < large_min_area or b_area > large_max_area:
                continue
            if (
                b_x <= border_margin
                or b_y <= border_margin
                or b_x + b_w >= gray.shape[1] - border_margin
            ):
                continue

            surface_gap = surface_edge_y - (b_y + b_h)
            if surface_gap < 0 or surface_gap > max(30, int(b_h * 0.25)):
                continue

            b_aspect = max(b_w, b_h) / (min(b_w, b_h) + 1e-5)
            b_fill_ratio = b_area / max(1, b_w * b_h)
            if b_aspect < 1.2 or b_aspect > 3.5:
                continue
            if b_fill_ratio < 0.4 or b_fill_ratio > 0.8:
                continue

            component_u8 = (large_labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                component_u8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if len(contour) < 5:
                continue
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = float(cv2.contourArea(contour)) / max(1.0, hull_area)
            ellipse_axes = cv2.fitEllipse(contour)[1]
            ellipse_aspect = max(ellipse_axes) / max(1.0, min(ellipse_axes))
            if solidity < 0.85 or ellipse_aspect < 1.5 or ellipse_aspect > 4.0:
                continue

            if extension > 0:
                large_dilate = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (extension * 2 + 1, extension * 2 + 1),
                )
                component_u8 = cv2.dilate(component_u8, large_dilate, iterations=1)
            bubble_mask[component_u8 > 0] = 255

        return bubble_mask

    def _check_dust_or_scratch_feature_with_context(
        self,
        omit_image: np.ndarray,
        tile_x: int,
        tile_y: int,
        tile_width: int,
        tile_height: int,
        omit_crop: np.ndarray,
        extension_override: Optional[int] = None,
        context_shift: int = 96,
        focus_x: Optional[int] = None,
        product_resolution: Optional[Tuple[int, int]] = None,
    ) -> tuple:
        """Supplement boundary-clipped bubbles from horizontally shifted tiles."""
        if product_resolution is None:
            is_dust, dust_mask, bright_ratio, detail_text = \
                self.check_dust_or_scratch_feature(omit_crop, extension_override)
        else:
            is_dust, dust_mask, bright_ratio, detail_text = \
                self.check_dust_or_scratch_feature(
                    omit_crop,
                    extension_override,
                    product_resolution=product_resolution,
                )
        if not getattr(self.config, 'dust_detect_bubbles_enabled', False):
            return is_dust, dust_mask, bright_ratio, detail_text
        if omit_image is None or omit_image.size == 0:
            return is_dust, dust_mask, bright_ratio, detail_text

        bubble_match = re.search(r"\bBub:(\d+)", detail_text)
        base_bubble_count = int(bubble_match.group(1)) if bubble_match else 0

        shift = max(0, int(context_shift))
        if shift == 0 or min(tile_width, tile_height) < 256:
            return is_dust, dust_mask, bright_ratio, detail_text

        context_offsets = (-shift, shift)
        if focus_x is not None:
            if focus_x < shift:
                context_offsets = (-shift,)
            elif focus_x >= tile_width - shift:
                context_offsets = (shift,)
            else:
                return is_dust, dust_mask, bright_ratio, detail_text

        context_bubble_mask = np.zeros((tile_height, tile_width), dtype=np.uint8)
        oh, ow = omit_image.shape[:2]
        extension = self.config.dust_extension \
            if extension_override is None else extension_override
        for offset_x in context_offsets:
            shifted_x = tile_x + offset_x
            if omit_image.ndim == 3:
                shifted_crop = np.zeros(
                    (tile_height, tile_width, omit_image.shape[2]),
                    dtype=omit_image.dtype,
                )
            else:
                shifted_crop = np.zeros((tile_height, tile_width), dtype=omit_image.dtype)

            src_x1 = max(0, shifted_x)
            src_y1 = max(0, tile_y)
            src_x2 = min(ow, shifted_x + tile_width)
            src_y2 = min(oh, tile_y + tile_height)
            if src_x2 <= src_x1 or src_y2 <= src_y1:
                continue

            dst_x1 = src_x1 - shifted_x
            dst_y1 = src_y1 - tile_y
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)
            shifted_crop[dst_y1:dst_y2, dst_x1:dst_x2] = \
                omit_image[src_y1:src_y2, src_x1:src_x2]
            if shifted_crop.ndim == 3:
                shifted_gray = cv2.cvtColor(shifted_crop, cv2.COLOR_BGR2GRAY)
            else:
                shifted_gray = shifted_crop

            shifted_bubble_mask = self._detect_bubble_mask(
                shifted_gray,
                self.config.dust_area_min,
                self.config.dust_area_max,
                extension,
                reference_area=tile_width * tile_height,
            )
            # 只補回穿越原 tile 左右邊界的元件，避免帶入平移視窗內的其他氣泡。
            seam_x = -offset_x if offset_x < 0 else tile_width - offset_x
            if seam_x <= 0 or seam_x >= tile_width:
                continue
            seam_start = max(0, seam_x - 6)
            seam_end = min(tile_width, seam_x + 7)
            seam_num, seam_labels = cv2.connectedComponents(
                (shifted_bubble_mask > 0).astype(np.uint8),
                connectivity=8,
            )
            if seam_num <= 1:
                continue
            crossing_labels = np.unique(seam_labels[:, seam_start:seam_end])
            crossing_labels = crossing_labels[crossing_labels != 0]
            if crossing_labels.size == 0:
                continue
            shifted_bubble_mask = (
                np.isin(seam_labels, crossing_labels).astype(np.uint8) * 255
            )
            overlap_x1 = max(tile_x, shifted_x)
            overlap_x2 = min(tile_x + tile_width, shifted_x + tile_width)
            if overlap_x2 <= overlap_x1:
                continue
            source_x1 = overlap_x1 - shifted_x
            source_x2 = overlap_x2 - shifted_x
            target_x1 = overlap_x1 - tile_x
            target_x2 = overlap_x2 - tile_x
            context_bubble_mask[:, target_x1:target_x2] = cv2.bitwise_or(
                context_bubble_mask[:, target_x1:target_x2],
                shifted_bubble_mask[:, source_x1:source_x2],
            )

        if not np.any(context_bubble_mask > 0):
            return is_dust, dust_mask, bright_ratio, detail_text

        merged_mask = np.zeros((tile_height, tile_width), dtype=np.uint8)
        if dust_mask is not None:
            copy_h = min(tile_height, dust_mask.shape[0])
            copy_w = min(tile_width, dust_mask.shape[1])
            merged_mask[:copy_h, :copy_w] = dust_mask[:copy_h, :copy_w]
        merged_mask[context_bubble_mask > 0] = 255

        mask_area = int(np.count_nonzero(merged_mask > 0))
        bright_ratio = float(mask_area / merged_mask.size) if merged_mask.size else 0.0
        context_count, _ = cv2.connectedComponents(context_bubble_mask, connectivity=8)
        context_bubble_count = max(0, context_count - 1)
        bubble_count = base_bubble_count + context_bubble_count
        detail_text = re.sub(r"\bArea:\d+", f"Area:{mask_area}", detail_text, count=1)
        detail_text = re.sub(
            r"\bRatio:\d+(?:\.\d+)?",
            f"Ratio:{bright_ratio:.4f}",
            detail_text,
            count=1,
        )
        if bubble_match:
            detail_text = re.sub(r"\bBub:\d+", f"Bub:{bubble_count}", detail_text, count=1)
        else:
            detail_text += f" Bub:{bubble_count}"
        detail_text += f" CtxShift:{shift}"
        return True, merged_mask, bright_ratio, detail_text
    
    def check_omit_overexposure(self, omit_image: np.ndarray) -> tuple:
        """
        檢查 OMIT 圖片是否曝光過度
        
        過曝的 OMIT 圖片(整張很白很亮)無法用於灰塵檢測，
        需要記錄並標記，供工程機台追蹤改善。
        
        Args:
            omit_image: OMIT 圖片 (BGR 或灰階)
            
        Returns:
            (is_overexposed, mean_brightness, bright_ratio, detail_text)
        """
        if omit_image is None or omit_image.size == 0:
            return False, 0.0, 0.0, "No OMIT image"
        
        # 轉灰階
        if len(omit_image.shape) == 3:
            gray = cv2.cvtColor(omit_image, cv2.COLOR_BGR2GRAY)
        elif len(omit_image.shape) == 2:
            gray = omit_image
        else:
            gray = omit_image.reshape(omit_image.shape[0], omit_image.shape[1])
        
        mean_brightness = float(np.mean(gray))
        bright_ratio = float(np.sum(gray > 230)) / gray.size if gray.size > 0 else 0.0
        
        mean_thr = self.config.omit_overexposure_mean_threshold
        ratio_thr = self.config.omit_overexposure_ratio_threshold
        
        is_overexposed = (mean_brightness > mean_thr) and (bright_ratio > ratio_thr)
        
        detail_text = (f"Mean:{mean_brightness:.1f}(thr={mean_thr}) "
                       f"BrightRatio:{bright_ratio:.3f}(thr={ratio_thr})")
        
        return is_overexposed, mean_brightness, bright_ratio, detail_text
    
    def compute_dust_heatmap_iou(self, dust_mask: np.ndarray, 
                                  anomaly_map: np.ndarray,
                                  top_percent: float = 5.0,
                                  metric: str = "coverage") -> tuple:
        """
        計算灰塵遮罩與 Heatmap「最紅區域」的重疊指標 (Coverage 或 IOU)
        
        Coverage = 交集 / 灰塵面積 (適合熱區遠大於灰塵的場景)
        IOU = 交集 / (灰塵面積 + 熱區面積 - 交集)
        
        使用 Percentile 方式：取 anomaly_map 中數值最高的前 X% 像素作為熱區，
        比舊的 max*ratio 更穩定、不受單一極端值影響。
        
        Args:
            dust_mask: 灰塵遮罩 (uint8, 255=灰塵)
            anomaly_map: Heatmap 異常圖 (float, 可含負值)
            top_percent: 取最高的前百分之幾作為「最紅區域」(建議 3~8)
            metric: "coverage" 或 "iou"
            
        Returns:
            (metric_value, heatmap_binary) - 指標值 (0.0~1.0), 二值化後的熱區遮罩
        """
        if dust_mask is None or anomaly_map is None:
            return 0.0, None
        
        # 預處理
        anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
        anomaly_map = np.maximum(anomaly_map, 0.0)  # 去除負值
        dust_mask = np.asarray(dust_mask, dtype=np.uint8)
        
        if np.max(anomaly_map) <= 0:
            return 0.0, None
        
        # === 核心：取「最紅的前 top_percent%」像素 ===
        positive_values = anomaly_map[anomaly_map > 0]
        if len(positive_values) == 0:
            return 0.0, None
        
        threshold = np.percentile(positive_values, 100 - top_percent)
        heat_bool = anomaly_map >= threshold
        
        # 產生二值化遮罩 (供可視化用)
        heatmap_binary = (heat_bool.astype(np.uint8)) * 255
        
        # 灰塵遮罩轉單通道
        if len(dust_mask.shape) == 3:
            dust_mask = cv2.cvtColor(dust_mask, cv2.COLOR_BGR2GRAY)
        
        # 尺寸匹配
        if dust_mask.shape != heat_bool.shape:
            dust_resized = cv2.resize(dust_mask,
                                      (heat_bool.shape[1], heat_bool.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
        else:
            dust_resized = dust_mask
        
        dust_bool = dust_resized > 0
        
        intersection = np.count_nonzero(dust_bool & heat_bool)
        
        if metric == "coverage":
            dust_area = np.count_nonzero(dust_bool)
            metric_val = float(intersection / dust_area) if dust_area > 0 else 0.0
        else:
            union = np.count_nonzero(dust_bool | heat_bool)
            metric_val = float(intersection / union) if union > 0 else 0.0
            
        return metric_val, heatmap_binary

    def _aoi_center_seed_for_tile(
        self,
        tile_info: Optional[TileInfo],
        anomaly_map: Optional[np.ndarray],
    ) -> tuple:
        """
        AOI 座標 tile 的中心點是機檢指定的缺陷 seed。回傳 anomaly_map 座標系
        的小型 seed，供 top% heatmap 二值化後額外納入，避免中心弱熱區被
        極少量 top% 排除。
        """
        if (
            tile_info is None
            or anomaly_map is None
            or not getattr(tile_info, "is_aoi_coord_tile", False)
        ):
            return None, 0, None
        if not getattr(self.config, "aoi_heatmap_center_seed_enabled", True):
            return None, 0, None

        amap = np.asarray(anomaly_map, dtype=np.float32)
        if amap.ndim < 2 or amap.size == 0 or float(np.max(amap)) <= 0:
            return None, 0, None

        amap_h, amap_w = amap.shape[:2]
        tile_w = max(1, int(getattr(tile_info, "width", 0) or 1))
        tile_h = max(1, int(getattr(tile_info, "height", 0) or 1))

        center_img_x = int(getattr(tile_info, "aoi_image_x", -1))
        center_img_y = int(getattr(tile_info, "aoi_image_y", -1))
        if center_img_x < 0 or center_img_y < 0:
            center_img_x = int(getattr(tile_info, "x", 0)) + tile_w // 2
            center_img_y = int(getattr(tile_info, "y", 0)) + tile_h // 2

        local_x = min(max(center_img_x - int(getattr(tile_info, "x", 0)), 0), tile_w - 1)
        local_y = min(max(center_img_y - int(getattr(tile_info, "y", 0)), 0), tile_h - 1)
        seed_x = int(round(local_x * (amap_w - 1) / max(tile_w - 1, 1)))
        seed_y = int(round(local_y * (amap_h - 1) / max(tile_h - 1, 1)))

        radius_tile_px = float(getattr(self.config, "aoi_heatmap_center_seed_radius_px", 12.0))
        scale = min(amap_w / tile_w, amap_h / tile_h)
        seed_radius = max(1, int(round(radius_tile_px * scale)))

        min_peak_ratio = float(getattr(self.config, "aoi_heatmap_center_seed_min_peak_ratio", 0.10))
        min_score = float(np.max(amap)) * max(0.0, min_peak_ratio)

        return (seed_y, seed_x), seed_radius, min_score

    def _mask_aoi_exclude_zones_for_dust(
        self,
        tile_info: Optional[TileInfo],
        anomaly_map: Optional[np.ndarray],
        model_id: Optional[str] = None,
    ) -> tuple:
        """
        AOI tile 的缺陷 seed 若不在不檢測區，dust 判定不應被不檢測區內
        的強 heatmap 主導。回傳供 dust/COV/two-stage 使用的 anomaly_map copy。
        """
        if (
            tile_info is None
            or anomaly_map is None
            or not getattr(tile_info, "is_aoi_coord_tile", False)
        ):
            return anomaly_map, False

        regions = self._configured_exclude_regions_for_model(model_id)
        if not regions:
            return anomaly_map, False

        aoi_x = int(getattr(tile_info, "aoi_image_x", -1))
        aoi_y = int(getattr(tile_info, "aoi_image_y", -1))
        if aoi_x >= 0 and aoi_y >= 0:
            if any(
                int(r.x1) <= aoi_x <= int(r.x2) and int(r.y1) <= aoi_y <= int(r.y2)
                for r in regions
            ):
                return anomaly_map, False

        return self._apply_no_detect_region_weighting(
            tile_info,
            anomaly_map,
            regions,
            hard_padding_px=0,
            soft_decay_px=max(0, int(getattr(self.config, "cv_edge_exclude_soft_decay_px", 64))),
        )

    def check_dust_per_region(
        self,
        dust_mask: np.ndarray,
        anomaly_map: np.ndarray,
        top_percent: float = 5.0,
        metric: str = "coverage",
        iou_threshold: float = 0.01,
        force_include_yx: Optional[Tuple[int, int]] = None,
        force_include_radius: int = 0,
        force_include_min_score: Optional[float] = None,
    ) -> tuple:
        """
        逐區域灰塵判定 — 將 anomaly_map 的熱區拆成獨立連通區域，
        分別與 dust_mask 做交叉驗證，只抑制與灰塵重疊的區域，保留真實缺陷。

        Returns:
            (has_real_defect, real_peak_yx, overall_iou, region_details, heatmap_binary, labels)
            - has_real_defect: 是否存在非灰塵的真實異常區域
            - real_peak_yx: 非灰塵區域中 anomaly_map 最大值的 (row, col)，None 表示全為灰塵
            - overall_iou: 整體 coverage/iou (向後相容)
            - region_details: list of dict，每個區域的判定詳情
            - heatmap_binary: 二值化後的熱區遮罩 (向後相容)
            - force_include_yx: 額外納入的 seed 中心 (row, col)，用於 AOI center rescue
        """
        if dust_mask is None or anomaly_map is None:
            return True, None, 0.0, [], None, None

        anomaly_map_f = np.asarray(anomaly_map, dtype=np.float32)
        anomaly_map_f = np.maximum(anomaly_map_f, 0.0)
        dust_mask_u8 = np.asarray(dust_mask, dtype=np.uint8)

        if np.max(anomaly_map_f) <= 0:
            return True, None, 0.0, [], None, None

        # === Step 1.5: 灰塵遮罩前處理 (提前到 Step 1 之前，供 mask 模式使用) ===
        if len(dust_mask_u8.shape) == 3:
            dust_mask_u8 = cv2.cvtColor(dust_mask_u8, cv2.COLOR_BGR2GRAY)
        if dust_mask_u8.shape != anomaly_map_f.shape:
            dust_mask_u8 = cv2.resize(
                dust_mask_u8,
                (anomaly_map_f.shape[1], anomaly_map_f.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        dust_bool = dust_mask_u8 > 0

        # === Step 1: 取前 top_percent% 像素作為熱區 ===
        use_mask_mode = getattr(self.config, 'dust_mask_before_binarize', False)

        if use_mask_mode:
            # 方法 4: 先將灰塵區域歸零，再對剩餘區域做 top% 二值化
            masked_anomaly = anomaly_map_f.copy()
            masked_anomaly[dust_bool] = 0
            positive_values = masked_anomaly[masked_anomaly > 0]
            if len(positive_values) == 0:
                # 全部被 mask 掉 → 整張都是灰塵
                heatmap_binary = np.zeros_like(anomaly_map_f, dtype=np.uint8)
                return False, None, 1.0, [], heatmap_binary, None
            threshold = np.percentile(positive_values, 100 - top_percent)
            heat_bool = masked_anomaly >= threshold
        else:
            # 現有流程: 直接對原始 anomaly_map 做 top% 二值化
            positive_values = anomaly_map_f[anomaly_map_f > 0]
            if len(positive_values) == 0:
                return True, None, 0.0, [], None, None
            threshold = np.percentile(positive_values, 100 - top_percent)
            heat_bool = anomaly_map_f >= threshold

        if force_include_yx is not None:
            seed_y, seed_x = int(force_include_yx[0]), int(force_include_yx[1])
            if 0 <= seed_y < heat_bool.shape[0] and 0 <= seed_x < heat_bool.shape[1]:
                seed_radius = max(0, int(force_include_radius))
                yy, xx = np.ogrid[:heat_bool.shape[0], :heat_bool.shape[1]]
                if seed_radius > 0:
                    seed_mask = (yy - seed_y) ** 2 + (xx - seed_x) ** 2 <= seed_radius ** 2
                else:
                    seed_mask = np.zeros_like(heat_bool, dtype=bool)
                    seed_mask[seed_y, seed_x] = True

                if force_include_min_score is None:
                    seed_score_mask = anomaly_map_f > 0
                else:
                    seed_score_mask = anomaly_map_f >= float(force_include_min_score)
                heat_bool = heat_bool | (seed_mask & seed_score_mask)

        heatmap_binary = (heat_bool.astype(np.uint8)) * 255

        # === Step 3: 整體 IOU (向後相容，用於 DB 記錄) ===
        intersection_all = np.count_nonzero(dust_bool & heat_bool)
        if metric == "coverage":
            dust_area_all = np.count_nonzero(dust_bool)
            overall_iou = float(intersection_all / dust_area_all) if dust_area_all > 0 else 0.0
        else:
            union_all = np.count_nonzero(dust_bool | heat_bool)
            overall_iou = float(intersection_all / union_all) if union_all > 0 else 0.0

        # === Step 4: 連通區域分析 ===
        heat_u8 = heatmap_binary.copy()
        num_labels, labels = cv2.connectedComponents(heat_u8, connectivity=8)

        region_details = []
        has_real_defect = False
        real_peak_yx = None
        real_peak_score = -1.0

        for label_id in range(1, num_labels):
            region_mask = labels == label_id
            region_area = np.count_nonzero(region_mask)

            # 計算此區域與灰塵的重疊
            region_dust_overlap = np.count_nonzero(region_mask & dust_bool)

            if metric == "coverage":
                # 此異常區域被灰塵覆蓋的比例
                region_coverage = float(region_dust_overlap / region_area) if region_area > 0 else 0.0
                metric_denominator = region_area
            else:
                # IOU = 交集 / 聯集 (此異常區域 ∪ 全部灰塵)
                total_dust = np.count_nonzero(dust_bool)
                region_union = region_area + total_dust - region_dust_overlap
                region_coverage = float(region_dust_overlap / region_union) if region_union > 0 else 0.0
                metric_denominator = region_union

            # 此區域內 anomaly_map 的最大值與位置 (無須複製整個陣列)
            region_vals = anomaly_map_f[region_mask]
            region_max_score = float(np.max(region_vals))
            region_indices = np.where(region_mask)
            local_argmax = np.argmax(region_vals)
            peak_pos = (region_indices[0][local_argmax], region_indices[1][local_argmax])

            # 判定是否為灰塵：覆蓋率須達閾值 且 峰值（最熱點）必須落在灰塵 mask 內
            # 若峰值不在灰塵上，代表缺陷核心與灰塵無關，僅邊緣碰到，不應判為灰塵
            # 但 heatmap peak 有膨脹偏移問題，高覆蓋率時直接判 dust 不依賴 peak 位置
            peak_in_dust = bool(dust_bool[peak_pos[0], peak_pos[1]])
            high_cov_thr = getattr(self.config, 'dust_high_cov_threshold', 0.5)

            # 灰塵次峰救援：peak 不在灰塵上，但灰塵區內最強分數 >= 區域 peak 的 X%
            # 代表 heatmap 熱點僅輕微偏移（上採樣/平滑效果），灰塵仍是主要異常來源
            dust_sub_peak_rescue = False
            if not peak_in_dust and region_coverage >= iou_threshold:
                dust_in_region = region_mask & dust_bool
                if np.any(dust_in_region):
                    dust_sub_peak = float(np.max(anomaly_map_f[dust_in_region]))
                    frac_thr = getattr(self.config, 'dust_peak_fraction_threshold', 0.80)
                    if region_max_score > 0 and dust_sub_peak / region_max_score >= frac_thr:
                        dust_sub_peak_rescue = True

            is_dust_region = region_coverage >= iou_threshold and (peak_in_dust or dust_sub_peak_rescue or region_coverage >= high_cov_thr)

            region_details.append({
                "label_id": label_id,
                "area": region_area,
                "dust_overlap": region_dust_overlap,
                "metric_denominator": metric_denominator,
                "coverage": region_coverage,
                "is_dust": is_dust_region,
                "peak_in_dust": peak_in_dust,
                "dust_sub_peak_rescue": dust_sub_peak_rescue,
                "max_score": region_max_score,
                "peak_yx": peak_pos,
            })

            if not is_dust_region:
                has_real_defect = True
                if region_max_score > real_peak_score:
                    real_peak_score = region_max_score
                    real_peak_yx = peak_pos

        return has_real_defect, real_peak_yx, overall_iou, region_details, heatmap_binary, labels

    def check_dust_two_stage(
        self,
        tile_image: np.ndarray,
        anomaly_map: np.ndarray,
        dust_mask: np.ndarray,
        score: float,
        score_threshold: Optional[float] = None,
        candidate_dust_mask: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        兩階段灰塵判定：
          Stage 1: 用原 heatmap 的 Top 5% 找候選區；若有擴展 dust mask，
                   另做一份灰塵抑制後的 heatmap 重新排名（原 score 不變）
          Stage 2: 回到原圖找 feature 點，依原 Top 核心、重排核心或
                   feature 局部等效分數確認，再精準比對無擴展 dust mask

        Returns:
            (has_real_defect, real_peak_yx, feature_details, detail_text)
        """
        cfg = self.config
        dust_ratio_thr = cfg.dust_two_stage_dust_ratio
        bg_blur_k = cfg.dust_two_stage_bg_blur
        diff_pct = cfg.dust_two_stage_diff_percentile
        min_area = cfg.dust_two_stage_min_area
        hot_zone_dust_thr = getattr(
            cfg,
            "dust_two_stage_hot_zone_dust_cov_threshold",
            getattr(cfg, "dust_high_cov_threshold", 0.5),
        )

        if tile_image is None or anomaly_map is None or dust_mask is None:
            return True, None, [], "TWO_STAGE: missing data -> REAL_NG"

        # --- Prepare images ---
        if len(tile_image.shape) == 3:
            tile_gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        else:
            tile_gray = tile_image.copy()
        tile_h, tile_w = tile_gray.shape

        anomaly_f = np.asarray(anomaly_map, dtype=np.float32)
        anomaly_f = np.maximum(anomaly_f, 0.0)
        h_am, w_am = anomaly_f.shape

        dm = np.asarray(dust_mask, dtype=np.uint8)
        if len(dm.shape) == 3:
            dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
        if dm.shape != (tile_h, tile_w):
            dm = cv2.resize(dm, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
        dust_bool_tile = dm > 0

        # --- Stage 1: Heatmap -> broad candidate zones ---
        pos_vals = anomaly_f[anomaly_f > 0]
        if len(pos_vals) == 0:
            return True, None, [], "TWO_STAGE: no positive heatmap -> REAL_NG"

        hot_thr = np.percentile(pos_vals, 95)
        hot_mask = (anomaly_f >= hot_thr).astype(np.uint8) * 255
        core_top_percent = min(
            100.0,
            max(0.01, float(getattr(cfg, "dust_heatmap_top_percent", 5.0))),
        )
        core_thr = np.percentile(pos_vals, 100.0 - core_top_percent)
        hot_core_mask = (anomaly_f >= core_thr).astype(np.uint8)

        # 僅在候選排名的副本抑制已知灰塵/氣泡；保留原 anomaly map 與 score。
        reranked_core_mask = np.zeros_like(hot_core_mask)
        if candidate_dust_mask is not None:
            candidate_dm = np.asarray(candidate_dust_mask, dtype=np.uint8)
            if len(candidate_dm.shape) == 3:
                candidate_dm = cv2.cvtColor(candidate_dm, cv2.COLOR_BGR2GRAY)
            if candidate_dm.shape != (tile_h, tile_w):
                candidate_dm = cv2.resize(
                    candidate_dm,
                    (tile_w, tile_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            candidate_dm_map = cv2.resize(
                candidate_dm,
                (w_am, h_am),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
            if np.any(candidate_dm_map):
                reranked_anomaly = anomaly_f.copy()
                reranked_anomaly[candidate_dm_map] = 0.0
                reranked_pos_vals = reranked_anomaly[reranked_anomaly > 0]
                if len(reranked_pos_vals) > 0:
                    reranked_hot_thr = np.percentile(reranked_pos_vals, 95)
                    reranked_hot_mask = (
                        reranked_anomaly >= reranked_hot_thr
                    ).astype(np.uint8) * 255
                    hot_mask = cv2.bitwise_or(hot_mask, reranked_hot_mask)
                    reranked_core_thr = np.percentile(
                        reranked_pos_vals,
                        100.0 - core_top_percent,
                    )
                    reranked_core_mask = (
                        reranked_anomaly >= reranked_core_thr
                    ).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        hot_mask = cv2.dilate(hot_mask, kernel, iterations=2)
        n_labels, labels = cv2.connectedComponents(hot_mask, connectivity=8)

        scale = tile_w / w_am
        pad = 20
        feature_core_margin_px = min(8, max(0, (min(tile_w, tile_h) - 1) // 2))

        def _core_support_on_tile(core_mask: np.ndarray) -> np.ndarray:
            core_tile = cv2.resize(
                core_mask,
                (tile_w, tile_h),
                interpolation=cv2.INTER_NEAREST,
            )
            if not feature_core_margin_px:
                return core_tile > 0
            core_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (feature_core_margin_px * 2 + 1, feature_core_margin_px * 2 + 1),
            )
            return cv2.dilate(core_tile, core_kernel) > 0

        hot_core_support = _core_support_on_tile(hot_core_mask)
        reranked_core_support = _core_support_on_tile(reranked_core_mask)
        anomaly_tile = cv2.resize(
            anomaly_f,
            (tile_w, tile_h),
            interpolation=cv2.INTER_LINEAR,
        )
        global_anomaly_peak = float(np.max(anomaly_f))
        local_support_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (feature_core_margin_px * 2 + 1, feature_core_margin_px * 2 + 1),
        )

        # --- Stage 2: Find features on original ---
        all_features = []
        ignored_border_features = 0
        ignored_outside_hot_core_features = 0
        reranked_after_dust_features = 0
        local_score_rescued_features = 0
        feature_border_margin_px = 8

        for lid in range(1, n_labels):
            rm = labels == lid
            ys, xs = np.where(rm)
            y1, y2 = int(np.min(ys)), int(np.max(ys))
            x1, x2 = int(np.min(xs)), int(np.max(xs))

            zone_mask_tile = cv2.resize(
                rm.astype(np.uint8),
                (tile_w, tile_h),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
            zone_area = int(np.count_nonzero(zone_mask_tile))
            zone_dust_overlap = int(np.count_nonzero(zone_mask_tile & dust_bool_tile))
            zone_dust_cov = float(zone_dust_overlap / zone_area) if zone_area > 0 else 0.0
            zone_dust_dominated = zone_dust_cov >= hot_zone_dust_thr

            # convert to tile space with padding
            ty1 = max(0, int(y1 * scale) - pad)
            ty2 = min(tile_h, int((y2 + 1) * scale) + pad)
            tx1 = max(0, int(x1 * scale) - pad)
            tx2 = min(tile_w, int((x2 + 1) * scale) + pad)

            crop_gray = tile_gray[ty1:ty2, tx1:tx2]
            crop_dust = dm[ty1:ty2, tx1:tx2]

            if crop_gray.size == 0:
                continue

            # ensure blur kernel is odd and <= crop size
            bk = bg_blur_k
            bk = min(bk, min(crop_gray.shape) - 1)
            if bk % 2 == 0:
                bk += 1
            if bk < 3:
                bk = 3

            blur = cv2.GaussianBlur(crop_gray, (bk, bk), 0)

            # detect dark + bright spots
            for diff, spot_type in [
                (blur.astype(np.float32) - crop_gray.astype(np.float32), "dark"),
                (crop_gray.astype(np.float32) - blur.astype(np.float32), "bright"),
            ]:
                diff_pos = diff[diff > 0]
                if len(diff_pos) < 10:
                    continue
                thr = max(float(np.percentile(diff_pos, diff_pct)), 3.0)
                binary = (diff >= thr).astype(np.uint8) * 255
                morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_k)

                n_feat, feat_labels = cv2.connectedComponents(binary, connectivity=8)
                for fid in range(1, n_feat):
                    fm = feat_labels == fid
                    farea = int(np.count_nonzero(fm))
                    if farea < min_area:
                        continue

                    fys, fxs = np.where(fm)
                    fcy, fcx = int(np.mean(fys)), int(np.mean(fxs))
                    fmin_x = int(np.min(fxs))
                    fmax_x = int(np.max(fxs))
                    fmin_y = int(np.min(fys))
                    fmax_y = int(np.max(fys))
                    feature_bbox = (
                        int(tx1 + fmin_x),
                        int(ty1 + fmin_y),
                        int(fmax_x - fmin_x + 1),
                        int(fmax_y - fmin_y + 1),
                    )
                    fb_x, fb_y, fb_w, fb_h = feature_bbox
                    if (
                        fb_x <= feature_border_margin_px
                        or fb_y <= feature_border_margin_px
                        or fb_x + fb_w >= tile_w - feature_border_margin_px
                        or fb_y + fb_h >= tile_h - feature_border_margin_px
                    ):
                        ignored_border_features += 1
                        continue

                    feature_in_broad_zone = bool(
                        np.any(zone_mask_tile[ty1:ty2, tx1:tx2][fm])
                    )
                    hot_core_supported = bool(
                        np.any(hot_core_support[ty1:ty2, tx1:tx2][fm])
                    )
                    dust_rerank_supported = bool(
                        np.any(reranked_core_support[ty1:ty2, tx1:tx2][fm])
                    )

                    local_support = cv2.dilate(
                        fm.astype(np.uint8),
                        local_support_kernel,
                    ) > 0
                    local_values = anomaly_tile[ty1:ty2, tx1:tx2][local_support]
                    local_peak = (
                        float(np.max(local_values))
                        if local_values.size > 0
                        else 0.0
                    )
                    local_equiv_score = (
                        float(score) * local_peak / global_anomaly_peak
                        if global_anomaly_peak > 0
                        else 0.0
                    )
                    local_score_supported = bool(
                        score_threshold is not None
                        and score_threshold > 0
                        and local_equiv_score >= float(score_threshold)
                    )

                    # Top 5% 僅圈候選範圍；feature 可由原 Top 核心、
                    # 灰塵抑制後重排核心，或自身局部分數取得判定資格。
                    if not feature_in_broad_zone or not (
                        hot_core_supported
                        or dust_rerank_supported
                        or local_score_supported
                    ):
                        ignored_outside_hot_core_features += 1
                        continue

                    if not hot_core_supported and dust_rerank_supported:
                        reranked_after_dust_features += 1
                    elif (
                        not hot_core_supported
                        and not dust_rerank_supported
                        and local_score_supported
                    ):
                        local_score_rescued_features += 1

                    if hot_core_supported:
                        support_source = "original_core"
                    elif dust_rerank_supported:
                        support_source = "dust_rerank"
                    else:
                        support_source = "local_score"

                    contours, _ = cv2.findContours(
                        (fm.astype(np.uint8)) * 255,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    feature_contour = []
                    if contours:
                        contour = max(contours, key=cv2.contourArea)
                        feature_contour = [
                            [int(tx1 + pt[0][0]), int(ty1 + pt[0][1])]
                            for pt in contour
                        ]

                    # dust check: use ALL feature pixels
                    feat_dust = crop_dust[fm]
                    dust_overlap = int(np.count_nonzero(feat_dust > 0))
                    feat_dust_ratio = dust_overlap / farea
                    feature_on_dust = feat_dust_ratio >= dust_ratio_thr
                    # 原 Top 核心保留既有整區 dust 保護；重排/局部分數救回的
                    # feature 則看自身與無擴展 dust mask 的精準重疊。
                    feature_is_dust = feature_on_dust or (
                        hot_core_supported and zone_dust_dominated
                    )
                    dust_reason = (
                        "feature_overlap" if feature_on_dust
                        else "zone_dominated"
                        if hot_core_supported and zone_dust_dominated
                        else "clean"
                    )

                    abs_x = tx1 + fcx
                    abs_y = ty1 + fcy

                    all_features.append({
                        "abs_pos": (abs_x, abs_y),
                        "area": farea,
                        "type": spot_type,
                        "feature_bbox": feature_bbox,
                        "feature_contour": feature_contour,
                        "dust_overlap": dust_overlap,
                        "dust_ratio": feat_dust_ratio,
                        "is_dust": feature_is_dust,
                        "dust_reason": dust_reason,
                        "zone_id": lid,
                        "zone_dust_cov": zone_dust_cov,
                        "zone_dust_dominated": zone_dust_dominated,
                        "broad_candidate_supported": feature_in_broad_zone,
                        "hot_core_supported": hot_core_supported,
                        "dust_rerank_supported": dust_rerank_supported,
                        "local_score_supported": local_score_supported,
                        "local_peak": local_peak,
                        "local_equiv_score": local_equiv_score,
                        "support_source": support_source,
                    })

        # --- Verdict ---
        real_features = [f for f in all_features if not f["is_dust"]]
        dust_features = [f for f in all_features if f["is_dust"]]
        ignored_parts = []
        if ignored_border_features:
            ignored_parts.append(f"ignored_border={ignored_border_features}")
        if ignored_outside_hot_core_features:
            ignored_parts.append(
                f"ignored_outside_hot_core={ignored_outside_hot_core_features}"
            )
        if reranked_after_dust_features:
            ignored_parts.append(
                f"reranked_after_dust={reranked_after_dust_features}"
            )
        if local_score_rescued_features:
            ignored_parts.append(
                f"local_score_rescued={local_score_rescued_features}"
            )
        ignored_hint = f" {' '.join(ignored_parts)}" if ignored_parts else ""

        if real_features:
            # find peak position of largest real feature
            best = max(real_features, key=lambda f: f["area"])
            bx, by = best["abs_pos"]
            # convert to anomaly_map space for peak_yx
            real_peak_yx = (int(by / scale), int(bx / scale))
            threshold_hint = (
                f"/{float(score_threshold):.3f}"
                if score_threshold is not None
                else ""
            )
            detail = (f"TWO_STAGE: {len(real_features)}real+{len(dust_features)}dust"
                      f"{ignored_hint} -> REAL_NG (best@({bx},{by}) area={best['area']}"
                      f" support={best['support_source']}"
                      f" local_eq={best['local_equiv_score']:.3f}{threshold_hint})")
            return True, real_peak_yx, all_features, detail

        else:
            # 找不到 real feature -> 信任 PER_REGION 的 dust 判定
            # （MARK 等規則紋理在 OMIT 已被 dust mask 涵蓋；二階段抓不到 feature 屬正常）
            detail = (f"TWO_STAGE: 0real+{len(dust_features)}dust"
                      f"{ignored_hint} -> DUST")
            return False, None, all_features, detail

    def generate_dust_iou_debug_image(
        self,
        tile_image: np.ndarray,
        anomaly_map: np.ndarray,
        dust_mask: np.ndarray,
        heatmap_binary: np.ndarray,
        iou: float,
        top_percent: float,
        is_dust: bool,
        region_details: Optional[list] = None,
        region_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        產生灰塵 IOU 交叉驗證的 Debug 可視化圖

        顯示：
          左上: Heatmap 疊加原圖 (紅色=異常熱區)
          右上: 灰塵遮罩 (黃色=灰塵區域)
          左下: 熱區二值化 (白色=top X% 最紅像素)
          右下: 逐區域判定結果 (紅色=真實缺陷, 綠色=灰塵區域, 藍色=僅灰塵遮罩)

        Args:
            tile_image: 原始 tile 圖片
            anomaly_map: 異常熱圖 (float)
            dust_mask: 灰塵遮罩 (uint8, 255=灰塵)
            heatmap_binary: 二值化熱區遮罩
            iou: 計算出的 IOU 值
            top_percent: 使用的百分位數
            is_dust: 最終判定是否為灰塵
            region_details: 逐區域判定詳情 (from check_dust_per_region)

        Returns:
            Debug 可視化圖 (BGR)
        """
        sz = 256  # 每個子圖大小

        # --- 準備基底圖 ---
        if len(tile_image.shape) == 2:
            base = cv2.cvtColor(tile_image, cv2.COLOR_GRAY2BGR)
        elif tile_image.shape[2] == 1:
            base = cv2.cvtColor(tile_image, cv2.COLOR_GRAY2BGR)
        else:
            base = tile_image.copy()
        base = cv2.resize(base, (sz, sz))

        # --- 左上: Heatmap Overlay ---
        if anomaly_map is not None:
            norm = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            norm = cv2.resize(norm, (sz, sz))
            heatmap_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            panel_tl = cv2.addWeighted(base, 0.5, heatmap_color, 0.5, 0)
        else:
            panel_tl = base.copy()
        cv2.putText(panel_tl, "Heatmap", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # --- 右上: 灰塵遮罩 ---
        panel_tr = base.copy()
        if dust_mask is not None:
            dm = dust_mask
            if len(dm.shape) == 3:
                dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
            dm = cv2.resize(dm, (sz, sz), interpolation=cv2.INTER_NEAREST)
            dust_overlay = np.zeros_like(panel_tr)
            dust_overlay[dm > 0] = (0, 255, 255)  # 黃色
            panel_tr = cv2.addWeighted(panel_tr, 0.6, dust_overlay, 0.4, 0)
        cv2.putText(panel_tr, "Dust Mask", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # --- 左下: 熱區二值化 ---
        panel_bl = np.zeros((sz, sz, 3), dtype=np.uint8)
        if heatmap_binary is not None:
            hb = cv2.resize(heatmap_binary, (sz, sz), interpolation=cv2.INTER_NEAREST)
            panel_bl[hb > 0] = (255, 255, 255)  # 白色 = 熱區
        cv2.putText(panel_bl, f"Top {top_percent:g}%", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

        # --- 右下: 逐區域判定結果 ---
        panel_br = np.zeros((sz, sz, 3), dtype=np.uint8)
        if region_details is not None and heatmap_binary is not None and dust_mask is not None:
            # 使用逐區域判定結果上色 (labels 由 check_dust_per_region 傳入，避免重複計算)
            if region_labels is not None:
                labels = region_labels
            else:
                _, labels = cv2.connectedComponents(heatmap_binary.copy(), connectivity=8)
            num_labels = int(labels.max()) + 1
            region_dust_map = {r["label_id"]: r["is_dust"] for r in region_details}

            # 縮放 labels 到 sz x sz
            labels_resized = cv2.resize(labels.astype(np.float32), (sz, sz),
                                        interpolation=cv2.INTER_NEAREST).astype(np.int32)

            dm = dust_mask
            if len(dm.shape) == 3:
                dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
            dm = cv2.resize(dm, (sz, sz), interpolation=cv2.INTER_NEAREST)

            # 先畫僅灰塵遮罩區域 (藍色)
            hb_resized = cv2.resize(heatmap_binary, (sz, sz), interpolation=cv2.INTER_NEAREST)
            dust_only = (dm > 0) & (hb_resized == 0)
            panel_br[dust_only] = (255, 100, 0)  # 藍色 = 僅灰塵遮罩

            # 逐區域上色
            region_peak_dust_map = {r["label_id"]: r.get("peak_in_dust", True) for r in region_details}
            orig_h, orig_w = heatmap_binary.shape[:2]
            scale_x = sz / orig_w
            scale_y = sz / orig_h
            for label_id in range(1, num_labels):
                region_mask = labels_resized == label_id
                if region_dust_map.get(label_id, False):
                    panel_br[region_mask] = (0, 200, 0)     # 綠色 = 灰塵(已抑制)
                else:
                    panel_br[region_mask] = (0, 0, 255)      # 紅色 = 真實缺陷(保留)

            # 在每個 region 的峰值位置畫標記
            for r in region_details:
                py, px = r["peak_yx"]
                sx = int(px * scale_x)
                sy = int(py * scale_y)
                peak_in = r.get("peak_in_dust", True)
                if peak_in:
                    # 峰值在灰塵上 → 黃色圓點
                    cv2.circle(panel_br, (sx, sy), 4, (0, 255, 255), -1)
                else:
                    # 峰值不在灰塵上 → 白色十字 (關鍵：這是被救回的真實缺陷)
                    cv2.drawMarker(panel_br, (sx, sy), (255, 255, 255),
                                   cv2.MARKER_CROSS, 10, 2)

            real_count = sum(1 for r in region_details if not r["is_dust"])
            dust_count = sum(1 for r in region_details if r["is_dust"])
            verdict_text = f"R:{real_count} D:{dust_count}"
        elif heatmap_binary is not None and dust_mask is not None:
            # Fallback: 舊版整塊分析
            hb = cv2.resize(heatmap_binary, (sz, sz), interpolation=cv2.INTER_NEAREST)
            dm = dust_mask
            if len(dm.shape) == 3:
                dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
            dm = cv2.resize(dm, (sz, sz), interpolation=cv2.INTER_NEAREST)

            heat_only = (hb > 0) & (dm == 0)
            dust_only = (dm > 0) & (hb == 0)
            overlap = (hb > 0) & (dm > 0)

            panel_br[heat_only] = (0, 0, 255)
            panel_br[dust_only] = (255, 100, 0)
            panel_br[overlap]   = (0, 255, 0)
            verdict_text = "DUST" if is_dust else "REAL_NG"
        else:
            verdict_text = "DUST" if is_dust else "REAL_NG"

        metric_name = "COV" if self.config.dust_heatmap_metric == "coverage" else "IOU"
        verdict_color = (0, 200, 255) if is_dust else (0, 0, 255)
        # 顯示 per-region max coverage（實際判定用的值）而非 overall
        if region_details:
            region_max_cov = max(r["coverage"] for r in region_details)
            peak_out_count = sum(1 for r in region_details
                                 if r["coverage"] >= self.config.dust_heatmap_iou_threshold
                                 and not r.get("peak_in_dust", True))
            peak_hint = f" PeakOut:{peak_out_count}" if peak_out_count > 0 else ""
            cv2.putText(panel_br, f"Region{metric_name}:{region_max_cov:.3f} {verdict_text}{peak_hint}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, verdict_color, 1)
        else:
            cv2.putText(panel_br, f"{metric_name}:{iou:.3f} {verdict_text}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, verdict_color, 1)

        # --- 組合 2x2 ---
        top_row = np.hstack([panel_tl, panel_tr])
        bottom_row = np.hstack([panel_bl, panel_br])
        debug_img = np.vstack([top_row, bottom_row])

        return debug_img

    def generate_two_stage_debug_image(
        self,
        tile_image: np.ndarray,
        anomaly_map: np.ndarray,
        dust_mask_no_ext: np.ndarray,
        features: list,
        is_dust: bool,
    ) -> np.ndarray:
        """
        產生兩階段灰塵判定的 Debug 可視化圖

        顯示：
          左上: Heatmap + Hot Zone 框
          右上: 原圖 + Dust Mask (黃, ext=0) + 特徵點標記
          左下: 原圖放大 (hot zone 區域)
          右下: 特徵判定結果 (紅=REAL, 綠=DUST)
        """
        sz = 256

        # --- base image ---
        if len(tile_image.shape) == 2:
            base = cv2.cvtColor(tile_image, cv2.COLOR_GRAY2BGR)
        elif tile_image.shape[2] == 1:
            base = cv2.cvtColor(tile_image, cv2.COLOR_GRAY2BGR)
        else:
            base = tile_image.copy()
        tile_h, tile_w = base.shape[:2]
        base_sm = cv2.resize(base, (sz, sz))

        anomaly_f = np.asarray(anomaly_map, dtype=np.float32)
        anomaly_f = np.maximum(anomaly_f, 0.0)
        h_am, w_am = anomaly_f.shape
        scale_tile = tile_w / w_am

        # --- dust mask prep ---
        dm = np.asarray(dust_mask_no_ext, dtype=np.uint8)
        if len(dm.shape) == 3:
            dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
        if dm.shape != (tile_h, tile_w):
            dm = cv2.resize(dm, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)

        # --- hot zone detection (same as check_dust_two_stage) ---
        pos_vals = anomaly_f[anomaly_f > 0]
        hot_thr = np.percentile(pos_vals, 95) if len(pos_vals) > 0 else 0
        hot_mask = (anomaly_f >= hot_thr).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        hot_mask = cv2.dilate(hot_mask, kernel, iterations=2)

        # --- 左上: Heatmap + Hot Zone ---
        norm = cv2.normalize(anomaly_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm_rsz = cv2.resize(norm, (sz, sz))
        hm_color = cv2.applyColorMap(norm_rsz, cv2.COLORMAP_JET)
        panel_tl = cv2.addWeighted(base_sm, 0.5, hm_color, 0.5, 0)
        # draw hot zone contour
        hot_rsz = cv2.resize(hot_mask, (sz, sz), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(hot_rsz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel_tl, contours, -1, (0, 255, 0), 1)
        cv2.putText(panel_tl, "Heatmap+HotZone", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- 右上: 原圖 + Dust Mask + 特徵標記 ---
        panel_tr = base_sm.copy()
        dm_sm = cv2.resize(dm, (sz, sz), interpolation=cv2.INTER_NEAREST)
        dust_ol = np.zeros_like(panel_tr)
        dust_ol[dm_sm > 0] = (0, 255, 255)
        panel_tr = cv2.addWeighted(panel_tr, 0.6, dust_ol, 0.4, 0)
        sx, sy = sz / tile_w, sz / tile_h

        def _feature_color(feat):
            return (255, 255, 0) if feat["is_dust"] else (0, 0, 255)

        def _feature_contour_small(feat):
            contour_points = feat.get("feature_contour") or []
            if not contour_points:
                return None
            pts = [
                [int(round(float(px) * sx)), int(round(float(py) * sy))]
                for px, py in contour_points
            ]
            if len(pts) < 2:
                return None
            return np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))

        def _draw_feature(panel, feat, fill=False):
            fx, fy = feat["abs_pos"]
            dx, dy = int(round(float(fx) * sx)), int(round(float(fy) * sy))
            color = _feature_color(feat)
            contour = _feature_contour_small(feat)
            if contour is not None:
                if fill:
                    overlay = panel.copy()
                    cv2.drawContours(overlay, [contour], -1, color, -1)
                    cv2.addWeighted(overlay, 0.65, panel, 0.35, 0, dst=panel)
                    cv2.drawContours(panel, [contour], -1, (255, 255, 255), 1)
                else:
                    cv2.drawContours(panel, [contour], -1, (0, 0, 0), 3)
                    cv2.drawContours(panel, [contour], -1, color, 2)
            else:
                cv2.circle(panel, (dx, dy), 5, color, -1 if fill else 2)
                if fill:
                    cv2.circle(panel, (dx, dy), 6, (255, 255, 255), 1)

            cv2.drawMarker(panel, (dx, dy), (0, 0, 0), cv2.MARKER_CROSS, 10, 3)
            cv2.drawMarker(panel, (dx, dy), color, cv2.MARKER_CROSS, 10, 1)
            return dx, dy, color

        for feat in features:
            dx, dy, color = _draw_feature(panel_tr, feat, fill=False)
            label = "D" if feat["is_dust"] else "R"
            cv2.putText(panel_tr, label, (dx + 7, dy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        cv2.putText(panel_tr, "Feature contour (R=Real C=Dust)", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        # --- 左下: 熱區二值化 (Top X%) ---
        panel_bl = np.zeros((sz, sz, 3), dtype=np.uint8)
        top_percent = self.config.dust_heatmap_top_percent
        pos_vals = anomaly_f[anomaly_f > 0]
        if len(pos_vals) > 0:
            thr = np.percentile(pos_vals, 100 - top_percent)
            hb = (anomaly_f >= thr).astype(np.uint8) * 255
            hb_rsz = cv2.resize(hb, (sz, sz), interpolation=cv2.INTER_NEAREST)
            panel_bl[hb_rsz > 0] = (255, 255, 255)
        cv2.putText(panel_bl, f"Top {top_percent:g}%", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

        # --- 右下: 判定結果 ---
        panel_br = np.zeros((sz, sz, 3), dtype=np.uint8)
        # dust mask as blue background
        panel_br[dm_sm > 0] = (180, 80, 0)
        # feature areas
        real_count = sum(1 for f in features if not f["is_dust"])
        dust_count = sum(1 for f in features if f["is_dust"])
        for feat in features:
            _draw_feature(panel_br, feat, fill=True)

        verdict_color = (0, 0, 255) if not is_dust else (0, 200, 255)
        verdict_text = f"R:{real_count} D:{dust_count}"
        cv2.putText(panel_br, f"TwoStage {verdict_text}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, verdict_color, 1)
        cv2.putText(panel_br, "B=DustMask R=Real C=Dust", (5, sz - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (128, 128, 128), 1)

        # --- 組合 2x2 ---
        top_row = np.hstack([panel_tl, panel_tr])
        bottom_row = np.hstack([panel_bl, panel_br])
        return np.vstack([top_row, bottom_row])

    _PANEL_IMAGE_EXTENSIONS = (".png", ".jpg", ".tif", ".tiff")

    @classmethod
    def _list_panel_image_files(cls, panel_dir: Path) -> List[Path]:
        return sorted(
            f for f in Path(panel_dir).iterdir()
            if f.is_file() and f.suffix.lower() in cls._PANEL_IMAGE_EXTENSIONS
        )

    @staticmethod
    def _select_latest_panel_images(image_files: List[Path]) -> List[Path]:
        """
        當面版資料夾存在重複投片時，依每個圖片前綴只保留「最新」的一張。

        支援舊命名 {前綴}_{HHMMSS} 與 HM 新命名 {前綴}{HHMMSS}。

        使用 st_mtime 排序，避免跨日時 HHMMSS 時間戳倒序 (如 235959 → 000001)。
        """
        from collections import defaultdict

        prefix_map: Dict[str, List[Path]] = defaultdict(list)
        for f in image_files:
            prefix = panel_image_group_key(f.name)
            prefix_map[prefix].append(f)

        selected = []
        for prefix, files in prefix_map.items():
            latest = max(files, key=lambda f: f.stat().st_mtime)
            selected.append(latest)

        return sorted(selected)

    def _prepare_panel_image_files(self, panel_dir: Path) -> Tuple[List[Path], bool]:
        """Return image files for inference, applying duplicate-panel selection."""
        image_files = self._list_panel_image_files(panel_dir)
        max_imgs = self.config.max_images_per_panel
        group_keys = [panel_image_group_key(f.name) for f in image_files]
        has_duplicate_groups = len(set(group_keys)) < len(group_keys)
        if len(image_files) <= max_imgs and not has_duplicate_groups:
            return image_files, False

        selected = self._select_latest_panel_images(image_files)
        reason = "圖片數量超過上限" if len(image_files) > max_imgs else "同一畫面前綴重複"
        print(
            f"⚠️ 重複投片偵測({reason}): {Path(panel_dir).name} 共 {len(image_files)} 張圖片 "
            f"(上限 {max_imgs})，依建立時間選出最新 {len(selected)} 張繼續推論"
        )
        print(f"   ✅ 選用: {', '.join(f.name for f in selected)}")
        return selected, True


    def _parse_defect_txt(self, defect_file: Path) -> Dict[str, List[Dict]]:
        """解析 Defect.txt"""

        defects_map = {}
        if not defect_file.exists():
            return defects_map
            
        try:
            with open(defect_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            if not content:
                return defects_map
                
            records = content.split(';')
            for record in records:
                record = record.strip()
                if not record:
                    continue
                parts = record.split(',')
                if len(parts) >= 4:
                    filename = parts[0].strip()
                    if filename not in defects_map:
                        defects_map[filename] = []
                    
                    defects_map[filename].append({
                        'defect_code': parts[1].strip(),
                        'x': int(parts[2].strip()),
                        'y': int(parts[3].strip())
                    })
        except Exception as e:
            print(f"解析 Defect.txt 失敗: {e}")
            
        return defects_map

    def _get_known_image_prefixes(self) -> List[str]:
        """取得所有已知的圖片前綴 (來自 model_mapping + skip_files)"""
        prefixes = set()
        if getattr(self.config, "model_mapping", None):
            prefixes.update(self.config.model_mapping.keys())
        if self._model_mapping:
            prefixes.update(self._model_mapping.keys())
        for sf in self.config.skip_files:
            prefixes.add(sf)
        # 常見固定前綴
        for p in AOI_REPORT_PREFIXES:
            prefixes.add(p)
        return sorted(prefixes, key=len, reverse=True)

    def _parse_aoi_report_txt(self, panel_dir: Path) -> Dict[str, List['AOIReportDefect']]:
        """
        解析 AOI 機台 NG 報告 TXT。

        1. panel_dir 路徑替換 (預設 yuantu→Report) 取得報告目錄
        2. 找到最新 .TXT 檔
        3. 解析 NG 缺陷字串

        格式1: @;OK;NG{異常代碼}{10位座標}{8字元前綴}...
        格式2: 獨立一行 NG，後續每行 {異常代碼}{10位座標}{8字元前綴}
        例: NGPCDK20028800554W0F00000PCDK20171100894B0F00000

        Returns:
            {image_prefix: [AOIReportDefect, ...]}
        """
        result_map: Dict[str, List[AOIReportDefect]] = {}

        # 路徑替換取得報告目錄
        replace_from = self.config.aoi_report_path_replace_from
        replace_to = self.config.aoi_report_path_replace_to
        panel_str = str(panel_dir)

        if replace_from not in panel_str:
            logger.warning(f"AOI Report: 路徑中找不到 '{replace_from}': {panel_str}")
            return result_map

        report_dir = Path(panel_str.replace(replace_from, replace_to, 1))

        if not report_dir.exists():
            logger.info(f"AOI Report: 報告目錄不存在: {report_dir}")
            return result_map

        # 找最新的 .TXT 檔案 (依 st_mtime 排序，避免跨日時 HHMMSS 時間戳倒序)
        # 用 set 去重，避免 Windows 大小寫不敏感時 *.TXT 與 *.txt 回傳相同檔案
        txt_files = list({f.resolve() for f in
            list(report_dir.glob("*.TXT")) + list(report_dir.glob("*.txt"))})
        if not txt_files:
            logger.info(f"AOI Report: 報告目錄中無 TXT 檔案: {report_dir}")
            return result_map

        report_file = max(txt_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"AOI Report: 讀取報告 {report_file.name}")

        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if len(lines) < 2:
                logger.warning(f"AOI Report: 報告內容不足 2 行: {report_file}")
                return result_map

            ng_string = None
            stripped_lines = [line.strip().rstrip(',') for line in lines]

            # 舊格式：第二行以 @ 開頭，; 分隔，找到以 NG 開頭的欄位。
            line2 = stripped_lines[1]
            if line2.startswith('@'):
                fields = line2.split(';')
                for field in fields:
                    field = field.strip()
                    if field.startswith('NG') and len(field) > 2:
                        ng_string = field[2:]  # 去掉 NG 前綴
                        break
            else:
                # 新格式：前幾行是機種/PCB/機檢資訊，獨立一行 NG 後面才是座標。
                for idx, line in enumerate(stripped_lines):
                    if line.strip().upper().rstrip(';') != "NG":
                        continue
                    ng_string = "".join(
                        part.strip().rstrip(';,')
                        for part in stripped_lines[idx + 1:]
                        if part.strip()
                    )
                    break

            if not ng_string:
                logger.info(f"AOI Report: 報告中未發現 NG 記錄")
                return result_map

            # 用已知前綴建構 regex 解析缺陷記錄
            # 格式: {異常代碼}{10位座標}{8字元前綴}
            known_prefixes = self._get_known_image_prefixes()
            if not known_prefixes:
                logger.warning("AOI Report: 無已知圖片前綴，無法解析")
                return result_map

            prefix_pattern = '|'.join(re.escape(p) for p in known_prefixes)
            pattern = re.compile(r'([A-Za-z0-9]+?)(\d{10})(' + prefix_pattern + r')')
            matches = pattern.findall(ng_string)

            if not matches:
                logger.warning(f"AOI Report: 無法解析缺陷記錄: {ng_string[:80]}")
                return result_map

            for defect_code, coord_str, image_prefix in matches:
                product_x = int(coord_str[:5])
                product_y = int(coord_str[5:])

                canonical_prefix = canonical_image_prefix(image_prefix)
                defect = AOIReportDefect(
                    defect_code=defect_code,
                    product_x=product_x,
                    product_y=product_y,
                    image_prefix=canonical_prefix,
                )

                if canonical_prefix not in result_map:
                    result_map[canonical_prefix] = []
                result_map[canonical_prefix].append(defect)

            total = sum(len(v) for v in result_map.values())
            per_prefix = ", ".join(f"{p}×{len(v)}" for p, v in result_map.items())
            logger.info(
                f"AOI Report: 解析到 {total} 筆缺陷 (涉及 {len(result_map)} 種圖片前綴) [{per_prefix}]"
            )
            for prefix, defects in result_map.items():
                for d in defects:
                    logger.debug(f"  🎯 {d.defect_code} @ ({d.product_x}, {d.product_y}) → {prefix}")

        except Exception as e:
            logger.error(f"AOI Report: 解析失敗: {e}")

        return result_map

    def _create_aoi_coord_tiles(
        self,
        image: np.ndarray,
        result: 'ImageResult',
        aoi_defects: List['AOIReportDefect'],
        product_resolution: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List['TileInfo'], List['AOIReportDefect']]:
        """
        以 AOI 機檢座標為中心切取 512x512 tile。

        Args:
            image: 原始圖片
            result: 已預處理的 ImageResult (含 otsu_bounds, raw_bounds)
            aoi_defects: 該圖片對應的 AOI 報告缺陷列表
            product_resolution: 產品解析度

        Returns:
            (patchcore_tiles, edge_defects_for_cv)
            - patchcore_tiles: 可做 PatchCore 推論的 tiles
            - edge_defects_for_cv: 碰到邊緣無法完整切塊的 defects (需 CV 處理)
        """
        tile_size = self.config.tile_size
        half = tile_size // 2
        patchcore_tiles = []
        edge_defects = []
        is_skip_file = self.config.should_skip_file(result.image_path.name)

        if result.raw_bounds is None:
            logger.warning("AOI Coord: raw_bounds 為 None，無法建立切塊")
            return patchcore_tiles, edge_defects

        otsu_x1, otsu_y1, otsu_x2, otsu_y2 = result.otsu_bounds
        img_h, img_w = image.shape[:2]

        # 需要 tile_id 從現有 tiles 之後遞增
        next_tile_id = max((t.tile_id for t in result.tiles), default=-1) + 1

        for defect in aoi_defects:
            # 產品座標 → 圖片座標
            img_x, img_y = self._map_aoi_coords(
                defect.product_x, defect.product_y,
                result.raw_bounds, product_resolution
            )

            # 計算 tile 起點 (以座標為中心)
            centered_tx = img_x - half
            centered_ty = img_y - half
            tx = centered_tx
            ty = centered_ty

            # 檢查是否能完整放入 Otsu bounds 內
            otsu_width = otsu_x2 - otsu_x1
            otsu_height = otsu_y2 - otsu_y1

            if otsu_width < tile_size or otsu_height < tile_size:
                # 產品區域太小，無法放入 512x512 tile
                edge_defects.append(defect)
                print(f"  ⚠️ AOI Coord ({defect.defect_code}) @ ({img_x},{img_y}): 產品區域太小，轉 CV 處理")
                continue

            # 判定是否碰到邊緣
            at_edge = (
                img_x - otsu_x1 < half or
                otsu_x2 - img_x < half or
                img_y - otsu_y1 < half or
                otsu_y2 - img_y < half
            )

            if at_edge:
                if is_skip_file:
                    # skip_files (如 B0F00000) 使用二值化偵測，不依賴 PatchCore 模型
                    # 對邊緣 tile 用 clamping 建立即可，不需轉 CV 處理
                    print(f"  🎯 AOI Coord ({defect.defect_code}) @ ({img_x},{img_y}): 碰到邊緣，skip_file 模式仍建立 tile (clamping)")
                else:
                    edge_defects.append(defect)
                    print(f"  📐 AOI Coord ({defect.defect_code}) @ ({img_x},{img_y}): 碰到邊緣，轉 CV 處理")
                    continue

            # 邊界 clamp (確保不超出圖片範圍)
            tx = max(0, min(tx, img_w - tile_size))
            ty = max(0, min(ty, img_h - tile_size))

            # 切取 tile
            tile_img = image[ty:ty + tile_size, tx:tx + tile_size].copy()

            if tile_img.shape[0] != tile_size or tile_img.shape[1] != tile_size:
                edge_defects.append(defect)
                print(f"  ⚠️ AOI Coord ({defect.defect_code}) @ ({img_x},{img_y}): 切塊尺寸異常 {tile_img.shape}, 轉 CV 處理")
                continue

            # 判定邊緣旗標
            is_top = (ty <= otsu_y1 + tile_size)
            is_bottom = (ty + tile_size >= otsu_y2 - tile_size)
            is_left = (tx <= otsu_x1 + tile_size)
            is_right = (tx + tile_size >= otsu_x2 - tile_size)

            tile = TileInfo(
                tile_id=next_tile_id,
                x=tx,
                y=ty,
                width=tile_size,
                height=tile_size,
                image=tile_img,
                is_bottom_edge=is_bottom,
                is_top_edge=is_top,
                is_left_edge=is_left,
                is_right_edge=is_right,
                is_aoi_coord_tile=True,
                aoi_defect_code=defect.defect_code,
                aoi_product_x=defect.product_x,
                aoi_product_y=defect.product_y,
                aoi_image_x=img_x,
                aoi_image_y=img_y,
                aoi_tile_shift_dx=tx - centered_tx,
                aoi_tile_shift_dy=ty - centered_ty,
                zone="bright_spot" if is_skip_file else "",
            )

            patchcore_tiles.append(tile)
            next_tile_id += 1
            logger.debug(
                f"  🎯 AOI Coord ({defect.defect_code}) @ ({img_x},{img_y}) "
                f"→ Tile ({tx},{ty}) {tile_size}x{tile_size} "
                f"shift=({tile.aoi_tile_shift_dx},{tile.aoi_tile_shift_dy})"
            )

        return patchcore_tiles, edge_defects

    def _map_aoi_coords(
        self,
        px: int,
        py: int,
        raw_bounds: Tuple[int, int, int, int],
        product_resolution: Optional[Tuple[int, int]] = None,
        panel_polygon: Optional[np.ndarray] = None,
    ) -> Tuple[int, int]:
        """將產品座標映射到圖片座標。

        平常保留舊的 ``raw_bounds`` 線性映射。若映射點已落在 panel
        polygon 外，或 polygon 只佔 raw bounds 明顯較小的面積（代表
        raw bounds 可能被產品外字樣拉大），改用產品四角到 panel
        四角的透視映射。
        """
        return map_product_coord_to_image(
            px, py, raw_bounds, product_resolution, panel_polygon,
        )

    def _inspect_roi_fusion(
        self,
        image: np.ndarray,
        img_x: int,
        img_y: int,
        img_prefix: str,
        panel_polygon: Optional[np.ndarray] = None,
        omit_image: Optional[np.ndarray] = None,
        omit_overexposed: bool = False,
        otsu_bounds: Optional[Tuple[int, int, int, int]] = None,
        collapse_to_representative: bool = True,
        group_cv_band: bool = False,
        zone: str = "edge",
    ) -> Tuple[List[EdgeDefect], Dict[str, Any]]:
        """Phase 6 — AOI 邊緣 CV+PatchCore 空間分權 Fusion。

        對單顆 AOI 座標 ROI 同時跑 CV + PC，依 boundary_band_mask 把:
          - CV 的 defect 限定在 band 內 (CV 管 polygon 邊內側 boundary_band_px 寬帶)
          - PC 的 anomaly_map 把 band 部分歸零後再 threshold (PC 管 interior)
        最後對合併 defect list 統一套 OMIT 灰塵屏蔽。

        Args:
            panel_polygon: panel 邊界 polygon (np.ndarray Nx2)；None → fallback CV only
            omit_image / omit_overexposed: OMIT 影像與過曝旗標 (沿用 tile 路徑語意)
            otsu_bounds: 傳給 CV inspect_roi (僅四邊掃用得到，AOI 邊緣可不傳)

        Returns:
            (defects, stats)
            - defects: fusion 後保留的 EdgeDefect list (含 source_inspector 標籤)
            - stats: {"band_mask", "interior_mask", "pc_anomaly_map",
                     "pc_anomaly_map_interior", "cv_stats", "pc_stats",
                     "fusion_fallback_reason"}
        """
        tile_size = self.config.tile_size
        half = tile_size // 2
        rx1 = img_x - half
        ry1 = img_y - half
        rx2 = rx1 + tile_size
        ry2 = ry1 + tile_size
        img_h, img_w = image.shape[:2]

        band_px = int(getattr(self.edge_inspector.config, "aoi_edge_boundary_band_px", 40))

        # === Fallback: polygon 偵測失敗 → CV only ===
        if panel_polygon is None:
            sx1 = max(0, rx1)
            sy1 = max(0, ry1)
            sx2 = min(img_w, rx2)
            sy2 = min(img_h, ry2)
            cv_defects: List[EdgeDefect] = []
            cv_stats: Dict[str, Any] = {}
            if sx2 > sx1 and sy2 > sy1:
                roi_for_cv = image[sy1:sy2, sx1:sx2]
                cv_defects, cv_stats = self.edge_inspector.inspect_roi(
                    roi_for_cv, offset_x=sx1, offset_y=sy1,
                    otsu_bounds=otsu_bounds, panel_polygon=None,
                )
            for d in cv_defects:
                d.source_inspector = "cv"
                d.inspector_mode = "fusion"
                d.fusion_fallback_reason = "polygon_unavailable"
                d.d_edge_px = 0.0
                # Phase 7.2 fix: 補填 cv_filtered_mask / cv_mask_offset 給新 CV fusion renderer
                d.cv_filtered_mask = cv_stats.get("filtered_mask")
                d.cv_mask_offset = cv_stats.get("roi_offset", (sx1, sy1))
            cv_defects = self._apply_omit_dust_filter_to_edge_defects(
                cv_defects, omit_image, omit_overexposed,
            )
            return cv_defects, {
                "band_mask": None,
                "interior_mask": None,
                "pc_anomaly_map": None,
                "pc_anomaly_map_interior": None,
                "cv_stats": cv_stats,
                "pc_stats": {},
                "fusion_fallback_reason": "polygon_unavailable",
            }

        # === Build ROI 與 fg_mask（與 _inspect_roi_patchcore 一致）===
        sx1 = max(0, rx1)
        sy1 = max(0, ry1)
        sx2 = min(img_w, rx2)
        sy2 = min(img_h, ry2)
        if sx2 <= sx1 or sy2 <= sy1:
            return [], {
                "band_mask": None, "interior_mask": None,
                "pc_anomaly_map": None, "pc_anomaly_map_interior": None,
                "cv_stats": {}, "pc_stats": {},
                "fusion_fallback_reason": "roi_out_of_image",
            }

        dx1 = sx1 - rx1
        dy1 = sy1 - ry1
        dx2 = dx1 + (sx2 - sx1)
        dy2 = dy1 + (sy2 - sy1)
        channels = image.shape[2] if image.ndim == 3 else 1
        if channels == 1:
            roi = np.zeros((tile_size, tile_size), dtype=image.dtype)
        else:
            roi = np.zeros((tile_size, tile_size, channels), dtype=image.dtype)
        roi[dy1:dy2, dx1:dx2] = image[sy1:sy2, sx1:sx2]

        fg_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
        local_poly = panel_polygon.copy().astype(np.float32)
        local_poly[:, 0] -= rx1
        local_poly[:, 1] -= ry1
        cv2.fillPoly(fg_mask, [local_poly.astype(np.int32)], 255)

        # === Boundary band mask ===
        band_mask = compute_boundary_band_mask(
            roi_shape=(tile_size, tile_size),
            roi_origin=(rx1, ry1),
            panel_polygon=[(int(p[0]), int(p[1])) for p in panel_polygon],
            band_px=band_px,
            fg_mask=fg_mask,
        )

        # === CV 路徑 ===
        cv_defects_all, cv_stats = self.edge_inspector.inspect_roi(
            roi, offset_x=rx1, offset_y=ry1,
            otsu_bounds=otsu_bounds, panel_polygon=panel_polygon,
        )
        cv_defects_kept: List[EdgeDefect] = []
        polygon_int = panel_polygon.astype(np.int32)
        cv_mask_offset = cv_stats.get("roi_offset", (rx1, ry1)) if cv_stats else (rx1, ry1)
        cv_mask_all = cv_stats.get("filtered_mask") if cv_stats else None
        cv_band_mask = None
        if cv_mask_all is not None:
            cv_mask_all = np.asarray(cv_mask_all)
            if cv_mask_all.ndim == 3:
                cv_mask_all = cv2.cvtColor(cv_mask_all, cv2.COLOR_BGR2GRAY)
            cv_mask_all = cv_mask_all.astype(np.uint8)
            if cv_mask_all.shape[:2] == band_mask.shape[:2]:
                cv_band_mask = cv2.bitwise_and(cv_mask_all, band_mask)
            else:
                band_resized = cv2.resize(
                    band_mask,
                    (cv_mask_all.shape[1], cv_mask_all.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                cv_band_mask = cv2.bitwise_and(cv_mask_all, band_resized)

        def _defect_intersects_cv_band(defect: EdgeDefect) -> bool:
            cx, cy = defect.center
            roi_cx = int(cx - rx1)
            roi_cy = int(cy - ry1)
            center_in_band = bool(0 <= roi_cx < tile_size and 0 <= roi_cy < tile_size
                                  and band_mask[roi_cy, roi_cx] > 0)
            if cv_band_mask is None:
                return center_in_band

            bx, by, bw, bh = defect.bbox
            mo_x, mo_y = cv_mask_offset
            lx1 = max(0, int(bx - mo_x))
            ly1 = max(0, int(by - mo_y))
            lx2 = min(cv_band_mask.shape[1], int(bx + bw - mo_x))
            ly2 = min(cv_band_mask.shape[0], int(by + bh - mo_y))
            if lx2 <= lx1 or ly2 <= ly1:
                return center_in_band
            return bool(center_in_band or np.any(cv_band_mask[ly1:ly2, lx1:lx2] > 0))

        def _make_grouped_cv_band_defect(src_defects: List[EdgeDefect]) -> Optional[EdgeDefect]:
            if not src_defects:
                return None

            if cv_band_mask is not None and np.any(cv_band_mask > 0):
                ys, xs = np.where(cv_band_mask > 0)
                x1 = int(xs.min())
                y1 = int(ys.min())
                x2 = int(xs.max()) + 1
                y2 = int(ys.max()) + 1
                mo_x, mo_y = cv_mask_offset
                bbox = (int(mo_x + x1), int(mo_y + y1), int(x2 - x1), int(y2 - y1))
                area = int(np.count_nonzero(cv_band_mask))
                center = (int(mo_x + round(float(xs.mean()))),
                          int(mo_y + round(float(ys.mean()))))
                band_mask_for_defect = cv_band_mask
            else:
                x1 = min(int(d.bbox[0]) for d in src_defects)
                y1 = min(int(d.bbox[1]) for d in src_defects)
                x2 = max(int(d.bbox[0] + d.bbox[2]) for d in src_defects)
                y2 = max(int(d.bbox[1] + d.bbox[3]) for d in src_defects)
                bbox = (x1, y1, x2 - x1, y2 - y1)
                area = int(sum(int(d.area) for d in src_defects))
                center = (int(round(sum(d.center[0] for d in src_defects) / len(src_defects))),
                          int(round(sum(d.center[1] for d in src_defects) / len(src_defects))))
                band_mask_for_defect = None

            max_diff = max((int(d.max_diff) for d in src_defects), default=0)
            grouped = EdgeDefect(
                side="aoi_edge",
                area=area,
                bbox=bbox,
                center=center,
                max_diff=max_diff,
                inspector_mode="fusion",
                threshold_used=int(cv_stats.get("threshold", 0)) if cv_stats else 0,
                min_area_used=int(cv_stats.get("min_area", 0)) if cv_stats else 0,
                min_max_diff_used=int(cv_stats.get("min_max_diff", 0)) if cv_stats else 0,
            )
            grouped.source_inspector = "cv"
            grouped.d_edge_px = float(max(0.0, cv2.pointPolygonTest(
                polygon_int, (float(center[0]), float(center[1])), True)))
            grouped.cv_filtered_mask = band_mask_for_defect
            grouped.cv_mask_offset = cv_mask_offset
            grouped.panel_polygon = panel_polygon
            return grouped

        # 診斷：band 過濾前的完整 CV defect 列表（供 debug UI 顯示用）
        cv_defects_all_debug = []
        for d in cv_defects_all:
            cx, cy = d.center
            roi_cx = int(cx - rx1)
            roi_cy = int(cy - ry1)
            in_roi = 0 <= roi_cx < tile_size and 0 <= roi_cy < tile_size
            in_band = bool(in_roi and band_mask[roi_cy, roi_cx] > 0)
            cv_defects_all_debug.append({
                "center": [int(cx), int(cy)],
                "area": int(d.area),
                "max_diff": int(d.max_diff),
                "in_roi": in_roi,
                "in_band": in_band,
                "reject_reason": "" if in_band else (
                    "roi_out" if not in_roi else "not_in_band"
                ),
            })
            if not in_roi:
                continue
            if _defect_intersects_cv_band(d):
                d.source_inspector = "cv"
                d.inspector_mode = "fusion"
                d.d_edge_px = float(max(0.0, cv2.pointPolygonTest(
                    polygon_int, (float(cx), float(cy)), True)))
                # Phase 7.2 fix: 補填 cv_filtered_mask / cv_mask_offset 給新 CV fusion renderer
                if cv_band_mask is not None:
                    bx, by, bw, bh = d.bbox
                    mo_x, mo_y = cv_mask_offset
                    defect_mask = np.zeros_like(cv_band_mask, dtype=np.uint8)
                    lx1 = max(0, int(bx - mo_x))
                    ly1 = max(0, int(by - mo_y))
                    lx2 = min(cv_band_mask.shape[1], int(bx + bw - mo_x))
                    ly2 = min(cv_band_mask.shape[0], int(by + bh - mo_y))
                    if lx2 > lx1 and ly2 > ly1:
                        defect_mask[ly1:ly2, lx1:lx2] = cv_band_mask[ly1:ly2, lx1:lx2]
                    d.cv_filtered_mask = defect_mask if np.any(defect_mask > 0) else cv_band_mask
                else:
                    d.cv_filtered_mask = cv_stats.get("filtered_mask") if cv_stats else None
                d.cv_mask_offset = cv_mask_offset
                d.panel_polygon = panel_polygon
                cv_defects_kept.append(d)

        if group_cv_band and cv_defects_kept:
            grouped_cv = _make_grouped_cv_band_defect(cv_defects_kept)
            cv_defects_kept = [grouped_cv] if grouped_cv is not None else []

        # === PC 路徑（Phase 7.3: 內移 PC ROI 完全避開 CV 管轄 band 區）===
        # shift 目標：polygon 邊距 PC ROI ≥ band_px（= CV band 寬度），讓 PC ROI
        # 物理上完全在 band 之外，不需 mask 排除。
        # 若 needed > roi_size/2 → AOI 會脫離 ROI → 不 shift → fallback="aoi_exit_roi"
        shift_enabled = bool(getattr(
            self.edge_inspector.config, "aoi_edge_pc_roi_inward_shift_enabled", True))

        pc_roi_origin_candidate, shift_vec, d_edge_signed = compute_pc_roi_offset(
            aoi_xy=(img_x, img_y),
            polygon=polygon_int,
            band_px=band_px,
            roi_size=tile_size,
        )

        use_shifted = False
        pc_fallback_reason = ""
        half = tile_size // 2
        needed_shift = band_px + half - d_edge_signed if d_edge_signed > 0 else 0

        if not shift_enabled:
            # shift 功能被 config 關閉
            if needed_shift > 0:
                pc_fallback_reason = "shift_disabled"
        elif needed_shift > half:
            # AOI 距邊太近，shift 量超過 ROI 半徑 → AOI 會脫離 ROI → 不 shift
            pc_fallback_reason = "aoi_exit_roi"
        elif shift_vec != (0, 0):
            if verify_polygon_clear_of_pc_roi(
                pc_roi_origin=pc_roi_origin_candidate,
                roi_size=tile_size,
                polygon=polygon_int,
                band_px=band_px,
            ):
                use_shifted = True
            else:
                pc_fallback_reason = classify_pc_roi_verify_failure(
                    aoi_xy=(img_x, img_y),
                    pc_roi_origin=pc_roi_origin_candidate,
                    roi_size=tile_size,
                    polygon=polygon_int,
                    band_px=band_px,
                )
                shift_vec = (0, 0)

        if use_shifted:
            pc_center_x = img_x + shift_vec[0]
            pc_center_y = img_y + shift_vec[1]
            pc_roi_x1 = pc_roi_origin_candidate[0]
            pc_roi_y1 = pc_roi_origin_candidate[1]
        else:
            pc_center_x = img_x
            pc_center_y = img_y
            pc_roi_x1 = rx1
            pc_roi_y1 = ry1

        _, pc_stats = self._inspect_roi_patchcore(
            image, pc_center_x, pc_center_y, img_prefix,
            panel_polygon=panel_polygon, return_raw=True,
            zone=zone,
        )
        pc_anomaly_map = pc_stats.get("anomaly_map")
        pc_threshold = float(pc_stats.get("threshold", 0.0))
        pc_fg_mask_returned = pc_stats.get("fg_mask")
        anomaly_map_interior = None
        pc_defects: List[EdgeDefect] = []

        if pc_anomaly_map is not None:
            anomaly_map_interior = pc_anomaly_map.copy()
            if use_shifted:
                # Shifted: PC ROI 已物理上在 CV band 外，只排除 polygon 外像素
                # （band_mask 以 centered ROI 座標計算，套在 shifted map 上座標不對應）
                if pc_fg_mask_returned is not None:
                    anomaly_map_interior[pc_fg_mask_returned == 0] = 0.0
            else:
                # Fallback/centered: Phase 6 band_mask 排除 CV 管轄帶 + polygon 外
                anomaly_map_interior[band_mask > 0] = 0.0
                anomaly_map_interior[fg_mask == 0] = 0.0

            interior_score = float(np.max(anomaly_map_interior))
            if interior_score >= pc_threshold:
                interior_max_area = _anomaly_max_cc_area(anomaly_map_interior)
                pc_def = EdgeDefect(
                    side="aoi_edge",
                    area=int(interior_max_area),
                    bbox=(pc_roi_x1, pc_roi_y1, tile_size, tile_size),
                    center=(img_x, img_y),
                    max_diff=0,
                    inspector_mode="fusion",
                    patchcore_score=interior_score,
                    patchcore_threshold=pc_threshold,
                )
                pc_def.source_inspector = "patchcore"
                pc_def.d_edge_px = float(max(0.0, cv2.pointPolygonTest(
                    polygon_int, (float(img_x), float(img_y)), True)))
                pc_def.pc_roi = pc_stats.get("roi")
                pc_def.pc_fg_mask = pc_fg_mask_returned if use_shifted else fg_mask
                pc_def.pc_anomaly_map = anomaly_map_interior
                # Phase 7 shift 標記
                pc_def.pc_roi_origin_x = int(pc_roi_x1)
                pc_def.pc_roi_origin_y = int(pc_roi_y1)
                pc_def.pc_roi_shift_dx = int(shift_vec[0])
                pc_def.pc_roi_shift_dy = int(shift_vec[1])
                pc_def.pc_roi_fallback_reason = pc_fallback_reason
                pc_def.panel_polygon = panel_polygon  # 供 heatmap renderer 畫 CV band 輪廓
                pc_defects.append(pc_def)

        # === Merge + OMIT ===
        fusion_defects = cv_defects_kept + pc_defects
        fusion_defects = self._apply_omit_dust_filter_to_edge_defects(
            fusion_defects, omit_image, omit_overexposed,
        )

        # Phase 7.1 — collapse 到 1 筆代表 defect：
        #   real_NG 優先 > dust；real_NG 內 PC > CV；CV 內 max area
        pre_collapse_count = len(fusion_defects)
        cv_band_count = len(cv_defects_kept)
        pc_interior_count = len(pc_defects)
        if collapse_to_representative and fusion_defects:
            real_ng = [d for d in fusion_defects if not d.is_suspected_dust_or_scratch]
            if real_ng:
                pc_real = [d for d in real_ng if d.source_inspector == "patchcore"]
                if pc_real:
                    rep = pc_real[0]
                else:
                    rep = max(real_ng, key=lambda d: (d.area, d.max_diff))
            else:
                # 全 dust — 以相同優先序挑代表（保留 dust 旗標）
                pc_dust = [d for d in fusion_defects if d.source_inspector == "patchcore"]
                if pc_dust:
                    rep = pc_dust[0]
                else:
                    rep = max(fusion_defects, key=lambda d: (d.area, d.max_diff))
            fusion_defects = [rep]

        stats = {
            "band_mask": band_mask,
            "interior_mask": cv2.bitwise_and(fg_mask, cv2.bitwise_not(band_mask)),
            "pc_anomaly_map": pc_anomaly_map,
            "pc_anomaly_map_interior": anomaly_map_interior,
            "cv_stats": cv_stats,
            "pc_stats": pc_stats,
            "fusion_fallback_reason": "",
            # Phase 7 PC ROI 內移資訊（供 UI / OK defect 紀錄）
            "pc_roi_origin": (int(pc_roi_x1), int(pc_roi_y1)),
            "pc_roi_shift": (int(shift_vec[0]), int(shift_vec[1])),
            "pc_roi_fallback_reason": pc_fallback_reason,
            "pc_roi_fg_mask": pc_fg_mask_returned if use_shifted else fg_mask,
            # Phase 7.1 collapse 診斷（pre_collapse_count = collapse 前的 defect 總數）
            "pre_collapse_count": pre_collapse_count,
            "cv_band_count": cv_band_count,
            "pc_interior_count": pc_interior_count,
            "collapsed": collapse_to_representative and pre_collapse_count > 1,
            # 診斷：CV band 過濾前完整列表
            "cv_defects_all_debug": cv_defects_all_debug,
            "band_mask_pixels": int(np.count_nonzero(band_mask)) if band_mask is not None else 0,
            "fg_mask_pixels": int(np.count_nonzero(fg_mask)),
            "cv_fg_mask_pixels": int(cv_stats.get("fg_mask_pixels", 0)) if cv_stats else 0,
        }
        return fusion_defects, stats

    def _apply_omit_dust_filter_to_edge_defects(
        self,
        defects: List[EdgeDefect],
        omit_image: Optional[np.ndarray],
        omit_overexposed: bool,
    ) -> List[EdgeDefect]:
        """Phase 6 fusion — fusion 後對 edge defect list 統一套 OMIT 灰塵屏蔽。

        依 source_inspector 使用與主路徑一致的空間判定：
          - patchcore: check_dust_per_region，避免整張 OMIT dust 面積稀釋局部重疊。
          - cv: 直接比對 CV 實際 defect mask 被 OMIT dust 覆蓋的比例。

        Args:
            defects: fusion 產出的 EdgeDefect list (bbox in panel 座標系)
            omit_image: full panel OMIT 灰階影像；None 視為 OMIT 缺失
            omit_overexposed: True 時跳過 dust 判定、defect 全保留並標 detail_text

        Returns:
            原 defect list (in-place mutation 寫回 is_suspected_dust_or_scratch /
            dust_mask / dust_bright_ratio / dust_detail_text / omit_crop_image)。
        """
        if omit_image is None:
            return defects

        if omit_overexposed:
            for d in defects:
                d.dust_detail_text = "OMIT_OVEREXPOSED → REAL_NG"
            return defects

        oh, ow = omit_image.shape[:2]
        top_pct = self.config.dust_heatmap_top_percent
        metric = self.config.dust_heatmap_metric
        iou_thr = self.config.dust_heatmap_iou_threshold

        def _crop_omit_canvas(origin_x: int, origin_y: int, width: int, height: int) -> np.ndarray:
            """Crop OMIT with zero padding so mask/anomaly coordinates stay aligned."""
            origin_x = int(origin_x)
            origin_y = int(origin_y)
            width = int(max(0, width))
            height = int(max(0, height))
            if omit_image.ndim == 3:
                channels = omit_image.shape[2]
                canvas = np.zeros((height, width, channels), dtype=omit_image.dtype)
            else:
                canvas = np.zeros((height, width), dtype=omit_image.dtype)

            sx1 = max(0, origin_x)
            sy1 = max(0, origin_y)
            sx2 = min(ow, origin_x + width)
            sy2 = min(oh, origin_y + height)
            if sx2 > sx1 and sy2 > sy1:
                dx1 = sx1 - origin_x
                dy1 = sy1 - origin_y
                dx2 = dx1 + (sx2 - sx1)
                dy2 = dy1 + (sy2 - sy1)
                canvas[dy1:dy2, dx1:dx2] = omit_image[sy1:sy2, sx1:sx2]
            return canvas

        def _as_gray_mask(mask: np.ndarray) -> np.ndarray:
            if mask is not None and mask.ndim == 3:
                return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            return mask

        for d in defects:
            bx, by, bw, bh = d.bbox
            source = getattr(d, "source_inspector", "")

            defect_anomaly = None
            crop_origin = (int(bx), int(by))
            crop_w = int(bw)
            crop_h = int(bh)
            cv_binary_mask = None

            if source == "patchcore" and d.pc_anomaly_map is not None:
                defect_anomaly = d.pc_anomaly_map.astype(np.float32)
                crop_h, crop_w = defect_anomaly.shape[:2]
                crop_origin = (int(bx), int(by))
            elif source == "cv" and d.cv_filtered_mask is not None:
                cv_mask_full = _as_gray_mask(d.cv_filtered_mask).astype(np.uint8)
                mo_x, mo_y = getattr(d, "cv_mask_offset", (int(bx), int(by)))
                crop_origin = (int(mo_x), int(mo_y))
                crop_h, crop_w = cv_mask_full.shape[:2]

                # Limit the shared ROI mask to this EdgeDefect's bbox, matching
                # the non-fusion edge dust path and avoiding unrelated CV pixels.
                cv_binary_mask = np.zeros_like(cv_mask_full, dtype=np.uint8)
                rel_x1 = max(0, int(bx) - int(mo_x))
                rel_y1 = max(0, int(by) - int(mo_y))
                rel_x2 = min(crop_w, int(bx + bw) - int(mo_x))
                rel_y2 = min(crop_h, int(by + bh) - int(mo_y))
                if rel_x2 > rel_x1 and rel_y2 > rel_y1:
                    cv_binary_mask[rel_y1:rel_y2, rel_x1:rel_x2] = \
                        cv_mask_full[rel_y1:rel_y2, rel_x1:rel_x2]
                defect_anomaly = cv_binary_mask.astype(np.float32)
            else:
                if bx >= ow or by >= oh or bw <= 0 or bh <= 0:
                    continue

            if crop_w <= 0 or crop_h <= 0:
                continue

            omit_crop = _crop_omit_canvas(crop_origin[0], crop_origin[1], crop_w, crop_h)

            is_dust, dust_mask, bright_ratio, detail = self.check_dust_or_scratch_feature(omit_crop)
            d.dust_bright_ratio = float(bright_ratio)
            d.dust_detail_text = detail
            d.omit_crop_image = omit_crop.copy()

            if not is_dust or dust_mask is None:
                continue

            if defect_anomaly is None:
                continue

            dust_mask = _as_gray_mask(dust_mask)

            if source == "patchcore":
                has_real_defect, _peak, overall_iou, region_details, heatmap_binary, _labels = \
                    self.check_dust_per_region(
                        dust_mask, defect_anomaly,
                        top_percent=top_pct,
                        metric=metric,
                        iou_threshold=iou_thr,
                    )
                d.dust_mask = dust_mask
                d.dust_heatmap_iou = overall_iou
                d.dust_heatmap_binary = heatmap_binary
                d.dust_region_details = region_details
                if region_details:
                    d.dust_region_max_cov = max(r["coverage"] for r in region_details)
                dust_regions = [r for r in region_details if r["is_dust"]]
                real_regions = [r for r in region_details if not r["is_dust"]]

                if has_real_defect:
                    detail += (
                        f" PER_REGION: {len(real_regions)}real+"
                        f"{len(dust_regions)}dust -> REAL_NG"
                    )
                else:
                    d.is_suspected_dust_or_scratch = True
                    detail += (
                        f" PER_REGION: 0real+"
                        f"{len(dust_regions)}dust -> DUST"
                    )
                d.dust_detail_text = detail
            elif source == "cv" and cv_binary_mask is not None:
                if dust_mask.shape != cv_binary_mask.shape:
                    dust_cmp = cv2.resize(
                        dust_mask,
                        (cv_binary_mask.shape[1], cv_binary_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    dust_cmp = dust_mask

                defect_bool = cv_binary_mask > 0
                dust_bool = dust_cmp > 0
                intersection = int(np.count_nonzero(defect_bool & dust_bool))
                metric_name = "COV" if metric == "coverage" else "IOU"
                if metric == "coverage":
                    defect_area = int(np.count_nonzero(defect_bool))
                    metric_val = float(intersection / defect_area) if defect_area > 0 else 0.0
                else:
                    union = int(np.count_nonzero(defect_bool | dust_bool))
                    metric_val = float(intersection / union) if union > 0 else 0.0

                d.dust_mask = dust_cmp
                d.dust_iou = metric_val
                if metric_val >= iou_thr:
                    d.is_suspected_dust_or_scratch = True
                    detail += f" CV_{metric_name}:{metric_val:.3f}>={iou_thr:.3f} -> DUST"
                else:
                    detail += f" CV_{metric_name}:{metric_val:.3f}<{iou_thr:.3f} -> REAL_NG"
                d.dust_detail_text = detail

        return defects

    @staticmethod
    def _aoi_prefix_matches(report_prefix: str, target_prefix: str) -> bool:
        if not report_prefix or not target_prefix:
            return False
        return (
            report_prefix == target_prefix
            or report_prefix.startswith(target_prefix + "_")
            or target_prefix.startswith(report_prefix + "_")
        )

    def _aoi_report_has_bomb_coord(
        self,
        aoi_report: Dict[str, List['AOIReportDefect']],
        image_prefix: str,
        product_x: int,
        product_y: int,
    ) -> bool:
        tolerance = int(getattr(self.config, "bomb_match_tolerance", 50))
        for report_prefix, defects in (aoi_report or {}).items():
            if not self._aoi_prefix_matches(str(report_prefix), image_prefix):
                continue
            for defect in defects:
                if (
                    abs(int(defect.product_x) - product_x) <= tolerance
                    and abs(int(defect.product_y) - product_y) <= tolerance
                ):
                    return True
        return False

    def _aoi_report_with_forced_client_bomb_coords(
        self,
        aoi_report: Optional[Dict[str, List['AOIReportDefect']]],
        bomb_info: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, List['AOIReportDefect']], int]:
        """Return AOI candidates for inference, adding Client bomb coords only when AOI missed them."""
        base_report = aoi_report or {}
        if not bomb_info:
            return base_report, 0

        if not getattr(self.config, "bomb_area_force_detection_enabled", False):
            print("💣 [BOMB_FORCE] Client 有炸彈座標，但炸彈區域強制偵測未啟用，沿用 AOI Report 座標")
            return base_report, 0

        image_prefix = str(bomb_info.get("image_prefix") or "").strip()
        defect_type = str(bomb_info.get("defect_type") or "").strip().lower()
        raw_coords = bomb_info.get("coordinates") or []
        if not image_prefix or defect_type != "point":
            print(
                f"💣 [BOMB_FORCE] Client bomb prefix={image_prefix or '-'} "
                f"type={defect_type or '-'}，未補切 tile（僅支援 point 座標）"
            )
            return base_report, 0

        coords: List[Tuple[int, int]] = []
        for coord in raw_coords:
            if len(coord) < 2:
                continue
            try:
                coords.append((int(coord[0]), int(coord[1])))
            except (TypeError, ValueError):
                continue

        if not coords:
            print(f"💣 [BOMB_FORCE] Client bomb prefix={image_prefix} 無有效 point 座標，未補切 tile")
            return base_report, 0

        forced_report: Dict[str, List[AOIReportDefect]] = {
            prefix: list(defects)
            for prefix, defects in base_report.items()
        }
        covered = 0
        forced_coords: List[Tuple[int, int]] = []
        for product_x, product_y in coords:
            if self._aoi_report_has_bomb_coord(base_report, image_prefix, product_x, product_y):
                covered += 1
                continue
            forced_report.setdefault(image_prefix, []).append(
                AOIReportDefect(
                    defect_code="BOMB_FORCE",
                    product_x=product_x,
                    product_y=product_y,
                    image_prefix=image_prefix,
                )
            )
            forced_coords.append((product_x, product_y))

        tolerance = int(getattr(self.config, "bomb_match_tolerance", 50))
        print(
            f"💣 [BOMB_FORCE] prefix={image_prefix} client_points={len(coords)} "
            f"AOI已涵蓋={covered} 補切={len(forced_coords)} tol={tolerance}"
        )
        if forced_coords:
            print(f"💣 [BOMB_FORCE] AOI 未給座標，改用 Client 炸彈座標補切 tile: {forced_coords}")
        else:
            print("💣 [BOMB_FORCE] AOI 已給出 Client 炸彈座標附近的點，不額外補切 tile")
        return forced_report, len(forced_coords)

    def _apply_aoi_coord_inspection(
        self,
        panel_dir: Path,
        preprocessed_results: List[Any],  # ImageResult
        omit_image: Optional[np.ndarray],
        omit_overexposed: bool,
        product_resolution: Optional[Tuple[int, int]],
        aoi_report: Optional[Dict[str, List['AOIReportDefect']]] = None,
    ) -> Dict[str, int]:
        """執行 AOI 機檢座標切塊 + 邊緣 ROI inspection；mutates ``preprocessed_results`` in place。

        v1 / v2 共用入口。Inspector mode 由 ``_resolve_aoi_edge_inspector_mode``
        決定（新架構強制 'patchcore'）。新架構下 PC ROI 走 zone='edge' → edge.pt。

        **Note**：v1 在呼叫此 helper 之前還會做「skip-file 重新 preprocess」的分支
        （把 should_skip_file 的圖片補進 preprocessed_results 以便對它跑 AOI coord
        inspection），那是 v1 專屬流程，不在此 helper 內。

        Args:
            panel_dir: 面板目錄（用來找 aoi_report.txt）
            preprocessed_results: 已預處理的 ImageResult list
            omit_image, omit_overexposed: OMIT 灰塵屏蔽參數
            product_resolution: 產品解析度 (w, h)，用於 AOI 座標映射
            aoi_report: 已解析的 AOI report；None 時由 helper 自行解析

        Returns:
            ``{"aoi_tile_count": int, "aoi_edge_count": int}``
        """
        is_new_arch = bool(getattr(self.config, "is_new_architecture", False))

        if not self.config.aoi_coord_inspection_enabled:
            if is_new_arch:
                print("[v2] AOI 座標 attribution: 已停用（aoi_coord_inspection_enabled=False）")
            return {"aoi_tile_count": 0, "aoi_edge_count": 0}

        if aoi_report is None:
            aoi_report = self._parse_aoi_report_txt(panel_dir)
        if not aoi_report:
            if is_new_arch:
                print("[v2] AOI 座標 attribution: AOI report 解析為空（report 不存在或無 NG 條目）")
            return {"aoi_tile_count": 0, "aoi_edge_count": 0}

        # 新架構：以 AOI 機檢座標為 anchor 建 512x512 tile；靠 polygon 邊時
        # 往產品內側推，讓 edge.pt 的訓練與推論都只看產品本體，不看黑背景。
        if getattr(self.config, "is_new_architecture", False):
            from capi_preprocess import PreprocessConfig
            pre_cfg = PreprocessConfig(
                tile_size=self.config.tile_size,
                tile_stride=getattr(self.config, "tile_stride", self.config.tile_size),
                otsu_offset=self.config.otsu_offset,
                enable_panel_polygon=self.config.enable_panel_polygon,
                edge_threshold_px=self.config.edge_threshold_px,
                image_preprocess_pipeline=getattr(self.config, "image_preprocess_pipeline", []),
                image_preprocess_pipelines=getattr(self.config, "image_preprocess_pipelines", {}),
                preprocess_after_tiling=getattr(self.config, "preprocess_after_tiling", False),
                product_resolution=product_resolution or self._product_resolution(),
                rotate_180=getattr(self, "_rotate_detection_images_180", False),
            )
            aoi_tile_count = 0
            for result in preprocessed_results:
                img_prefix = self._get_image_prefix(result.image_path.name)
                if img_prefix not in aoi_report:
                    continue
                image = self._read_detection_image(result.image_path)
                if image is None:
                    logger.warning(f"[v2] AOI Coord: 無法讀取圖片 {result.image_path}")
                    continue
                is_skip_file = self.config.should_skip_file(result.image_path.name)
                aoi_tile_count += self._create_aoi_centered_tiles_v2(
                    image=image,
                    result=result,
                    defects=aoi_report[img_prefix],
                    product_resolution=product_resolution,
                    pre_cfg=pre_cfg,
                    is_skip_file=is_skip_file,
                )
            print(f"[v2] AOI 座標中心切塊: 建立 {aoi_tile_count} 個 centered tiles")
            return {"aoi_tile_count": aoi_tile_count, "aoi_edge_count": 0}

        inspector_mode = self._resolve_aoi_edge_inspector_mode()
        aoi_tile_count = 0
        aoi_edge_count = 0

        for result in preprocessed_results:
            img_prefix = self._get_image_prefix(result.image_path.name)
            if img_prefix not in aoi_report:
                continue

            aoi_image = self._read_detection_image(result.image_path)
            if aoi_image is None:
                continue

            new_tiles, edge_defs = self._create_aoi_coord_tiles(
                aoi_image, result, aoi_report[img_prefix], product_resolution,
            )
            result.tiles.extend(new_tiles)
            aoi_tile_count += len(new_tiles)
            aoi_edge_count += len(edge_defs)

            for edef in edge_defs:
                self._inspect_aoi_edge_defect(
                    edef=edef,
                    aoi_image=aoi_image,
                    result=result,
                    product_resolution=product_resolution,
                    inspector_mode=inspector_mode,
                    img_prefix=img_prefix,
                    omit_image=omit_image,
                    omit_overexposed=omit_overexposed,
                )

        return {"aoi_tile_count": aoi_tile_count, "aoi_edge_count": aoi_edge_count}

    def _create_aoi_centered_tiles_v2(
        self,
        image: np.ndarray,
        result: 'ImageResult',
        defects: List['AOIReportDefect'],
        product_resolution: Optional[Tuple[int, int]],
        pre_cfg: 'PreprocessConfig',
        is_skip_file: bool = False,
    ) -> int:
        """v2 新架構：以 AOI 機檢座標為 anchor 建 512x512 tile。

        若 AOI 靠近 panel polygon 邊，ROI 會往產品內側推到完整落在 polygon
        內，避免 edge.pt 看到黑色背景邊界；AOI 不一定在 tile 中心。

        差異重點：
          - 邊緣不 bail 到 CV，每筆 AOI 座標都建立 PatchCore tile
          - OOB / polygon edge 用 inward clamp，不再 ``cv2.copyMakeBorder`` zero-pad
          - zone 由 AOI 缺陷中心到 panel polygon 的距離判定；距離不超過
            半個 tile 時為 edge，否則為 inner；skip_file 強制 bright_spot

        Args:
            image: 原始圖片（cv2.imread 結果）
            result: 已預處理的 ImageResult;新 tile 直接 append 進 ``result.tiles``
            defects: 該圖片對應的 AOI 報告缺陷列表
            product_resolution: 產品解析度
            pre_cfg: PreprocessConfig（傳給 classify_tile_zone）
            is_skip_file: 是否為 skip_file（B0F00000 等黑圖）→ 強制 bright_spot zone

        Returns: 新建立的 tile 數
        """
        from capi_preprocess import (
            classify_anchor_zone,
            classify_tile_zone,
            resolve_inward_polygon_tile,
        )

        if result.raw_bounds is None:
            logger.warning("[v2] AOI Coord: raw_bounds 為 None，無法建立切塊")
            return 0

        tile_size = self.config.tile_size
        half = tile_size // 2
        raw_image = image
        cached_processed = getattr(result, "processed_image", None)
        if (
            cached_processed is not None
            and getattr(cached_processed, "shape", None) is not None
            and cached_processed.shape[:2] == raw_image.shape[:2]
        ):
            processed_image = cached_processed
        else:
            processed_image = raw_image
        preprocess_steps: List[Dict[str, Any]] = []
        preprocess_total_ms = 0.0
        if (
            not is_skip_file
            and processed_image is raw_image
            and getattr(pre_cfg, "image_preprocess_pipeline", None)
            and not getattr(pre_cfg, "preprocess_after_tiling", False)
        ):
            from capi_image_preprocess_lab import apply_preprocess_pipeline, describe_preprocess_pipeline
            logger.info(
                "[v2][AOI] pipeline: %s",
                describe_preprocess_pipeline(pre_cfg.image_preprocess_pipeline),
            )
            pipeline_result = apply_preprocess_pipeline(raw_image, pre_cfg.image_preprocess_pipeline)
            processed_image = pipeline_result["image"]
            preprocess_steps = pipeline_result["steps"]
            preprocess_total_ms = float(pipeline_result.get("total_elapsed_ms") or 0.0)
            for step in pipeline_result["steps"]:
                logger.info(
                    "[v2][AOI] step %d %s params=%s elapsed=%.3fms stats=%s",
                    step["index"], step["method_label"], step["applied_params"],
                    float(step.get("elapsed_ms") or 0.0), step["stats"],
                )
        if preprocess_steps:
            result.preprocess_steps.extend(preprocess_steps)
            result.preprocess_total_ms += preprocess_total_ms
        img_h, img_w = processed_image.shape[:2]
        polygon = result.panel_polygon
        if polygon is None:
            polygon = self._rect_polygon_from_bounds(result.raw_bounds)
        next_tile_id = max((t.tile_id for t in result.tiles), default=-1) + 1
        otsu_x1, otsu_y1, otsu_x2, otsu_y2 = result.raw_bounds
        created = 0

        for defect in defects:
            img_x, img_y = self._map_aoi_coords(
                defect.product_x, defect.product_y,
                result.raw_bounds, product_resolution,
                panel_polygon=polygon,
            )

            # B0F 只用 polygon 粗略定位 AOI 中心；亮點判定必須保留完整
            # 512x512，不用 polygon 當有效範圍，也不因 polygon 抓歪而拒絕座標。
            if not is_skip_file and polygon is not None:
                polygon_distance = float(cv2.pointPolygonTest(
                    np.asarray(polygon, dtype=np.float32),
                    (float(img_x), float(img_y)),
                    True,
                ))
                if polygon_distance < -1.0:
                    raise RuntimeError(
                        f"[v2] AOI 座標映射到 panel 外: "
                        f"{defect.defect_code} product=({defect.product_x},{defect.product_y}) "
                        f"image=({img_x},{img_y}) distance={polygon_distance:.1f}px"
                    )

            # Inference/training aligned inward ROI：先用 AOI-centered origin，
            # 若跨出 polygon 則往產品內側推；不 zero-pad、不帶黑背景進 edge.pt。
            centered_tx = img_x - half
            centered_ty = img_y - half
            tx = max(0, centered_tx)
            ty = max(0, centered_ty)
            tx2 = tx + tile_size
            ty2 = ty + tile_size
            if tx2 > img_w:
                tx2 = img_w
                tx = max(0, tx2 - tile_size)
            if ty2 > img_h:
                ty2 = img_h
                ty = max(0, ty2 - tile_size)

            if not is_skip_file and polygon is not None:
                shift_axes = self._resolve_aoi_inward_shift_axes(
                    img_x, img_y, result.raw_bounds, tile_size,
                )
                tx, ty, _cov, _shifted = resolve_inward_polygon_tile(
                    anchor_xy=(img_x, img_y),
                    polygon=polygon,
                    image_shape=(img_h, img_w),
                    tile_size=tile_size,
                    initial_origin=(tx, ty),
                    keep_anchor_inside=True,
                    shift_axes=shift_axes,
                )

            tx2 = min(img_w, tx + tile_size)
            ty2 = min(img_h, ty + tile_size)
            crop_w = tx2 - tx
            crop_h = ty2 - ty
            tile_img = processed_image[ty:ty2, tx:tx2].copy()
            original_tile = raw_image[ty:ty2, tx:tx2].copy()
            tile_mask = None
            if not is_skip_file:
                _tile_zone, _cov, _dist, tile_mask = classify_tile_zone(
                    (tx, ty, tx2, ty2), polygon, pre_cfg,
                )
                if tile_mask is not None and not np.any(tile_mask):
                    raise RuntimeError(
                        f"[v2] AOI tile 與 panel 無有效重疊: "
                        f"{defect.defect_code} image=({img_x},{img_y}) "
                        f"tile=({tx},{ty},{tx2},{ty2})"
                    )

            if is_skip_file:
                zone = "bright_spot"
            else:
                zone, _anchor_distance = classify_anchor_zone(
                    (img_x, img_y), polygon, half,
                )

            if not is_skip_file and getattr(pre_cfg, "preprocess_after_tiling", False):
                from capi_preprocess import image_preprocess_pipeline_for_zone
                tile_pipeline = image_preprocess_pipeline_for_zone(pre_cfg, zone)
                if tile_pipeline:
                    from capi_image_preprocess_lab import apply_preprocess_pipeline
                    pipeline_result = apply_preprocess_pipeline(tile_img, tile_pipeline)
                    tile_img = pipeline_result["image"]
                    result.preprocess_steps.extend(pipeline_result["steps"])
                    result.preprocess_total_ms += float(
                        pipeline_result.get("total_elapsed_ms") or 0.0
                    )

            # 邊緣旗標以 AOI 中心相對 otsu_bounds 判定（與 tile 位置無關）
            is_top = img_y - otsu_y1 < half
            is_bottom = otsu_y2 - img_y < half
            is_left = img_x - otsu_x1 < half
            is_right = otsu_x2 - img_x < half

            tile = TileInfo(
                tile_id=next_tile_id,
                x=tx, y=ty,
                width=crop_w, height=crop_h,
                image=tile_img,
                original_image=original_tile,
                mask=tile_mask,
                is_aoi_coord_tile=True,
                aoi_defect_code=defect.defect_code,
                aoi_product_x=defect.product_x,
                aoi_product_y=defect.product_y,
                aoi_image_x=img_x,
                aoi_image_y=img_y,
                aoi_tile_shift_dx=tx - centered_tx,
                aoi_tile_shift_dy=ty - centered_ty,
                zone=zone,
                is_top_edge=is_top,
                is_bottom_edge=is_bottom,
                is_left_edge=is_left,
                is_right_edge=is_right,
            )
            result.tiles.append(tile)
            next_tile_id += 1
            created += 1

            logger.debug(
                f"  🎯 [v2] AOI Coord ({defect.defect_code}) @ ({img_x},{img_y}) "
                f"→ Tile ({tx},{ty}) zone={zone} "
                f"shift=({tile.aoi_tile_shift_dx},{tile.aoi_tile_shift_dy})"
            )

        return created

    def _inspect_aoi_edge_defect(
        self,
        edef,
        aoi_image: np.ndarray,
        result,  # ImageResult
        product_resolution: Optional[Tuple[int, int]],
        inspector_mode: str,
        img_prefix: str,
        omit_image: Optional[np.ndarray],
        omit_overexposed: bool,
    ) -> None:
        """對單一 AOI 座標 edge defect 跑 fusion / patchcore / cv 任一 inspector。

        把 result.edge_defects mutate（append NG 或 OK record）。新架構走 zone='edge'。
        """
        img_x, img_y = self._map_aoi_coords(
            edef.product_x, edef.product_y,
            result.raw_bounds, product_resolution
        )
        roi_size = self.config.tile_size
        roi_half = roi_size // 2
        img_h, img_w = aoi_image.shape[:2]
        rx1 = max(0, img_x - roi_half)
        ry1 = max(0, img_y - roi_half)
        rx2 = min(img_w, img_x + roi_half)
        ry2 = min(img_h, img_y + roi_half)

        detected = False

        if inspector_mode == "fusion":
            try:
                fusion_defects, fusion_stats = self._inspect_roi_fusion(
                    aoi_image, img_x, img_y, img_prefix,
                    panel_polygon=result.panel_polygon,
                    omit_image=omit_image,
                    omit_overexposed=omit_overexposed,
                    otsu_bounds=result.otsu_bounds,
                    collapse_to_representative=False,
                    group_cv_band=True,
                    zone="edge",
                )
                pc_shift = fusion_stats.get("pc_roi_shift", (0, 0))
                pc_origin = fusion_stats.get("pc_roi_origin", (rx1, ry1))
                pc_fb = str(fusion_stats.get("pc_roi_fallback_reason", ""))
                # Force defect center to AOI 座標確保 BOMB 比對一致
                for d in fusion_defects:
                    d.center = (img_x, img_y)
                    result.edge_defects.append(d)
                    detected = True
                    shift_tag = f" shift=({d.pc_roi_shift_dx:+d},{d.pc_roi_shift_dy:+d})" \
                                if (d.pc_roi_shift_dx or d.pc_roi_shift_dy) else ""
                    # Phase 7.3 — PC fallback 原因標示
                    if d.pc_roi_fallback_reason == "aoi_exit_roi":
                        fb_tag = " PC-FB=aoi_exit_roi(AOI將離開ROI)"
                    elif d.pc_roi_fallback_reason == "concave_polygon":
                        fb_tag = " PC-FB=concave_polygon(凹角)"
                    elif d.pc_roi_fallback_reason == "shift_disabled":
                        fb_tag = " PC-FB=shift_disabled(內移停用)"
                    elif d.pc_roi_fallback_reason:
                        fb_tag = f" PC-FB={d.pc_roi_fallback_reason}"
                    else:
                        fb_tag = ""
                    print(f"  🔍 AOI Coord Fusion edge ({edef.defect_code}) "
                          f"@ ({img_x},{img_y}): src={d.source_inspector} "
                          f"score={d.patchcore_score:.3f} max_diff={d.max_diff} "
                          f"d_edge={d.d_edge_px:.1f}px"
                          f"{shift_tag}{fb_tag} "
                          f"is_dust={d.is_suspected_dust_or_scratch}")
                if not detected:
                    ok_defect = EdgeDefect(
                        side="aoi_coord_ok",
                        area=0,
                        bbox=(rx1, ry1, rx2 - rx1, ry2 - ry1),
                        center=(img_x, img_y),
                        max_diff=0,
                        is_cv_ok=True,
                        inspector_mode="fusion",
                        fusion_fallback_reason=str(fusion_stats.get("fusion_fallback_reason", "")),
                    )
                    ok_defect.source_inspector = ""  # OK row 不指定 source
                    ok_defect.pc_anomaly_map = fusion_stats.get("pc_anomaly_map_interior")
                    ok_defect.pc_fg_mask = fusion_stats.get("interior_mask")
                    ok_defect.pc_roi_origin_x = int(pc_origin[0])
                    ok_defect.pc_roi_origin_y = int(pc_origin[1])
                    ok_defect.pc_roi_shift_dx = int(pc_shift[0])
                    ok_defect.pc_roi_shift_dy = int(pc_shift[1])
                    ok_defect.pc_roi_fallback_reason = pc_fb
                    result.edge_defects.append(ok_defect)
                    fb = fusion_stats.get("fusion_fallback_reason", "")
                    shift_tag = f" shift=({pc_shift[0]:+d},{pc_shift[1]:+d})" \
                                if (pc_shift[0] or pc_shift[1]) else ""
                    # Phase 7.3 — PC fallback 原因標示（OK record）
                    if pc_fb == "aoi_exit_roi":
                        pcfb_tag = " PC-FB=aoi_exit_roi(AOI將離開ROI)"
                    elif pc_fb == "concave_polygon":
                        pcfb_tag = " PC-FB=concave_polygon(凹角)"
                    elif pc_fb == "shift_disabled":
                        pcfb_tag = " PC-FB=shift_disabled(內移停用)"
                    elif pc_fb:
                        pcfb_tag = f" PC-FB={pc_fb}"
                    else:
                        pcfb_tag = ""
                    print(f"  ✅ AOI Coord Fusion edge ({edef.defect_code}) @ ({img_x},{img_y}): OK"
                          f"{' (fallback=' + fb + ')' if fb else ''}"
                          f"{shift_tag}{pcfb_tag}")
            except Exception as e:
                logger.warning(f"AOI Coord Fusion edge 失敗 ({edef.defect_code}): {e}")
            return

        if inspector_mode == "patchcore":
            try:
                pc_defects, pc_stats = self._inspect_roi_patchcore(
                    aoi_image, img_x, img_y, img_prefix,
                    panel_polygon=result.panel_polygon,
                    zone="edge",
                )
                if pc_defects:
                    merged = pc_defects[0]
                    # 強制 center 為 AOI 座標以確保 BOMB 比對一致
                    merged.center = (img_x, img_y)
                    result.edge_defects.append(merged)
                    detected = True
                    print(f"  🔍 AOI Coord PatchCore edge ({edef.defect_code}) "
                          f"@ ({img_x},{img_y}): score={pc_stats.get('score', 0):.3f} "
                          f">= thr={pc_stats.get('threshold', 0):.3f}, "
                          f"area={pc_stats.get('area', 0)}")
                if not detected:
                    ok_defect = EdgeDefect(
                        side="aoi_coord_ok",
                        area=int(pc_stats.get("area", 0)),
                        bbox=(rx1, ry1, rx2 - rx1, ry2 - ry1),
                        center=(img_x, img_y),
                        max_diff=0,
                        is_cv_ok=True,
                        inspector_mode="patchcore",
                        patchcore_score=float(pc_stats.get("score", 0.0)),
                        patchcore_threshold=float(pc_stats.get("threshold", 0.0)),
                        patchcore_ok_reason=str(pc_stats.get("ok_reason", "")),
                    )
                    ok_defect.pc_roi = pc_stats.get("roi")
                    ok_defect.pc_fg_mask = pc_stats.get("fg_mask")
                    ok_defect.pc_anomaly_map = pc_stats.get("anomaly_map")
                    result.edge_defects.append(ok_defect)
                    print(f"  ✅ AOI Coord PatchCore edge ({edef.defect_code}) "
                          f"@ ({img_x},{img_y}): OK "
                          f"(score={pc_stats.get('score', 0):.3f}, "
                          f"thr={pc_stats.get('threshold', 0):.3f}, "
                          f"reason={pc_stats.get('ok_reason', '')})")
            except Exception as e:
                logger.warning(f"AOI Coord PatchCore edge 失敗 ({edef.defect_code}): {e}")
            return

        # CV 路徑 (現行)
        roi = aoi_image[ry1:ry2, rx1:rx2]
        roi_stats = {"max_diff": 0, "max_area": 0, "threshold": 0, "min_area": 0, "min_max_diff": 0}
        if roi.size > 0 and getattr(self, 'edge_inspector', None):
            try:
                edge_results, roi_stats = self.edge_inspector.inspect_roi(
                    roi, offset_x=rx1, offset_y=ry1,
                    otsu_bounds=result.otsu_bounds,
                    panel_polygon=result.panel_polygon,
                )
                if edge_results:
                    # 合併為單一 EdgeDefect (以 AOI 座標為中心)
                    # 避免拆成多筆小 defect 導致 BOMB 比對時部分 center 偏離
                    unified_bbox = (rx1, ry1, rx2 - rx1, ry2 - ry1)
                    total_area = sum(ed.area for ed in edge_results)
                    worst_diff = max(ed.max_diff for ed in edge_results)
                    merged = EdgeDefect(
                        side="aoi_edge",
                        area=total_area,
                        bbox=unified_bbox,
                        center=(img_x, img_y),  # 使用 AOI 座標中心，確保 BOMB 比對一致
                        max_diff=worst_diff,
                        threshold_used=roi_stats.get("threshold", 0),
                        min_area_used=roi_stats.get("min_area", 0),
                        min_max_diff_used=roi_stats.get("min_max_diff", 0),
                        inspector_mode="cv",
                        cv_mask_offset=roi_stats.get("roi_offset", (rx1, ry1)),
                    )
                    merged.cv_filtered_mask = roi_stats.get("filtered_mask")
                    result.edge_defects.append(merged)
                    detected = True
                    print(f"  🔍 AOI Coord CV edge ({edef.defect_code}) @ ({img_x},{img_y}): "
                          f"偵測到 {len(edge_results)} 個缺陷 → 合併為 1 筆 (area={total_area}, diff={worst_diff})")
            except Exception as e:
                logger.warning(f"AOI Coord CV edge 檢測失敗 ({edef.defect_code}): {e}")

        if not detected:
            # 建立 OK 記錄，帶入實際計算的 max_diff / max_area
            ok_defect = EdgeDefect(
                side="aoi_coord_ok",
                area=roi_stats.get("max_area", 0),
                bbox=(rx1, ry1, rx2 - rx1, ry2 - ry1),
                center=(img_x, img_y),
                max_diff=roi_stats.get("max_diff", 0),
                is_cv_ok=True,
                threshold_used=roi_stats.get("threshold", 0),
                min_area_used=roi_stats.get("min_area", 0),
                min_max_diff_used=roi_stats.get("min_max_diff", 0),
                inspector_mode="cv",
                cv_mask_offset=roi_stats.get("roi_offset", (rx1, ry1)),
            )
            ok_defect.cv_filtered_mask = roi_stats.get("filtered_mask")
            result.edge_defects.append(ok_defect)
            print(f"  ✅ AOI Coord edge ({edef.defect_code}) @ ({img_x},{img_y}): CV 未偵測到缺陷，判定 OK"
                  f"（max_diff={roi_stats.get('max_diff', 0)}, max_area={roi_stats.get('max_area', 0)}, "
                  f"thr={roi_stats.get('threshold', 0)}, min_area={roi_stats.get('min_area', 0)}, "
                  f"min_max_diff={roi_stats.get('min_max_diff', 0)}）")

    def _inspect_roi_patchcore(
        self,
        image: np.ndarray,
        img_x: int,
        img_y: int,
        img_prefix: str,
        panel_polygon: Optional[np.ndarray] = None,
        return_raw: bool = False,
        zone: str = "edge",
    ) -> Tuple[List[EdgeDefect], Dict[str, Any]]:
        """用 PatchCore 做 AOI 座標邊緣 ROI 推論。

        ROI 中心對齊 AOI 座標 + 黑 pad (panel 外本來就是黑色) + fg_mask 遮罩
        anomaly_map (panel 外歸零)。TileInfo 不標 is_*_edge，避免 edge_margin
        decay 在近 ROI 邊緣誤抑 AOI 中心的 defect。

        Args:
            return_raw: Phase 6 fusion 用——True 時跳過 defect 抽取/OK 原因判定，
                只回 (空 list, stats with anomaly_map+roi+fg_mask)，供 fusion 後
                外掛 boundary band mask + 自行 thresholding。預設 False (Phase 5 行為)。
            zone: "inner" | "edge"，新架構 (C-10) 用來路由到對應的 inner.pt / edge.pt。
                AOI 座標邊緣路徑預設 "edge"。舊架構忽略此值。

        Returns:
            (defects, stats)
            - defects: 若 NG 則 1 個 EdgeDefect，否則空 list；return_raw=True 永遠空
            - stats: {"score", "threshold", "area", "min_area", "ok_reason",
                     "roi", "fg_mask", "anomaly_map"}
        """
        tile_size = self.config.tile_size
        half = tile_size // 2
        img_h, img_w = image.shape[:2]

        # 中心對齊 ROI (image 外的像素留 0 — panel 外本來就是黑色)
        rx1 = img_x - half
        ry1 = img_y - half
        rx2 = rx1 + tile_size
        ry2 = ry1 + tile_size

        # image 內可擷取的子區 (src) 與貼到 ROI 的位置 (dst)
        sx1 = max(0, rx1)
        sy1 = max(0, ry1)
        sx2 = min(img_w, rx2)
        sy2 = min(img_h, ry2)

        # 若 ROI 完全在 image 外 → 無法推論
        if sx2 <= sx1 or sy2 <= sy1:
            stats = {"score": 0.0, "threshold": 0.0, "area": 0, "min_area": 0,
                     "ok_reason": "ROI out of image",
                     "roi": None, "fg_mask": None, "anomaly_map": None}
            return [], stats

        dx1 = sx1 - rx1
        dy1 = sy1 - ry1
        dx2 = dx1 + (sx2 - sx1)
        dy2 = dy1 + (sy2 - sy1)

        channels = image.shape[2] if image.ndim == 3 else 1
        if channels == 1:
            roi = np.zeros((tile_size, tile_size), dtype=image.dtype)
            roi[dy1:dy2, dx1:dx2] = image[sy1:sy2, sx1:sx2]
        else:
            roi = np.zeros((tile_size, tile_size, channels), dtype=image.dtype)
            roi[dy1:dy2, dx1:dx2] = image[sy1:sy2, sx1:sx2]

        # fg_mask: ROI 內落在 panel polygon 內的像素 = 255，panel 外 = 0
        fg_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
        if panel_polygon is not None:
            local_poly = panel_polygon.copy().astype(np.float32)
            local_poly[:, 0] -= rx1
            local_poly[:, 1] -= ry1
            cv2.fillPoly(fg_mask, [local_poly.astype(np.int32)], 255)
        else:
            # 無 polygon 時 fallback: ROI 在 image 內的區塊視為前景
            fg_mask[dy1:dy2, dx1:dx2] = 255

        # 取 inferencer / threshold：新架構走 zone-aware (走 edge.pt)，舊架構走 prefix-only
        inferencer = self._get_inferencer_for_zone(img_prefix, zone)
        threshold = self._get_threshold_for_zone(img_prefix, zone)

        if inferencer is None:
            stats = {"score": 0.0, "threshold": threshold, "area": 0, "min_area": 0,
                     "ok_reason": f"No model for prefix {img_prefix}",
                     "roi": roi, "fg_mask": fg_mask, "anomaly_map": None}
            return [], stats

        tile = TileInfo(
            tile_id=-1,
            x=rx1, y=ry1,
            width=tile_size, height=tile_size,
            image=roi,
            mask=fg_mask,
            is_top_edge=False, is_bottom_edge=False,
            is_left_edge=False, is_right_edge=False,
        )

        score, anomaly_map = self.predict_tile(
            tile, inferencer=inferencer, threshold=threshold, model_id=self.config.machine_id,
        )

        max_area = _anomaly_max_cc_area(anomaly_map)

        min_area = int(getattr(self.config, "patchcore_min_area", 10))
        is_ng = score >= threshold

        stats = {
            "score": float(score),
            "threshold": float(threshold),
            "area": int(max_area),
            "min_area": min_area,
            "roi": roi,
            "fg_mask": fg_mask,
            "anomaly_map": anomaly_map,
        }

        if return_raw:
            # Phase 6 fusion 路徑：跳過 defect 抽取與 OK 原因，把原始 anomaly_map
            # 交給 fusion 自行 mask + threshold + CC
            return [], stats

        if is_ng:
            defect = EdgeDefect(
                side="aoi_edge",
                area=int(max_area),
                bbox=(rx1, ry1, tile_size, tile_size),
                center=(img_x, img_y),
                max_diff=0,  # PatchCore path 不用灰階差
                solidity=1.0,
                inspector_mode="patchcore",
                patchcore_score=float(score),
                patchcore_threshold=float(threshold),
                patchcore_ok_reason="",
            )
            defect.pc_roi = roi
            defect.pc_fg_mask = fg_mask
            defect.pc_anomaly_map = anomaly_map
            stats["ok_reason"] = ""
            return [defect], stats

        # OK — 推算原因
        if score < threshold:
            reason = "Score<Thr"
        elif max_area < min_area:
            reason = "Area<Min"
        else:
            reason = ""
        stats["ok_reason"] = reason
        return [], stats

    @staticmethod
    def _point_within_line_segment_tolerance(
        point_x: float,
        point_y: float,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        tolerance: float,
    ) -> bool:
        """判斷產品座標點是否位於有限線段的容忍帶內。"""
        x1, y1 = float(pt1[0]), float(pt1[1])
        x2, y2 = float(pt2[0]), float(pt2[1])
        dx = x2 - x1
        dy = y2 - y1
        length_squared = dx * dx + dy * dy

        if length_squared == 0:
            nearest_x, nearest_y = x1, y1
        else:
            projection = ((point_x - x1) * dx + (point_y - y1) * dy) / length_squared
            projection = max(0.0, min(1.0, projection))
            nearest_x = x1 + projection * dx
            nearest_y = y1 + projection * dy

        offset_x = point_x - nearest_x
        offset_y = point_y - nearest_y
        return offset_x * offset_x + offset_y * offset_y <= float(tolerance) ** 2

    def _check_heatmap_line_shape(
        self,
        anomaly_map: np.ndarray,
        min_aspect_ratio: float = 3.0,
        top_percent: float = 10.0,
    ) -> Tuple[bool, float]:
        """
        檢查 anomaly_map 的熱區形態是否為線狀 (高長寬比)
        
        Args:
            anomaly_map: Heatmap 異常圖 (float)
            min_aspect_ratio: 最小長寬比閾值 (>= 此值判定為線狀)
            top_percent: 取前 X% 最熱像素進行形態分析
            
        Returns:
            (is_line, aspect_ratio) - 是否為線狀, 實際長寬比
        """
        if anomaly_map is None or anomaly_map.size == 0:
            return False, 0.0
        
        amap = np.asarray(anomaly_map, dtype=np.float32)
        amap = np.maximum(amap, 0.0)
        
        positive_values = amap[amap > 0]
        if len(positive_values) == 0:
            return False, 0.0
        
        # Percentile 二值化
        threshold = np.percentile(positive_values, 100 - top_percent)
        binary = (amap >= threshold).astype(np.uint8) * 255
        
        # 找輪廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, 0.0
        
        # 取最大輪廓
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 10:
            return False, 0.0
        
        # 最小外接矩形 → 計算長寬比
        rect = cv2.minAreaRect(largest)
        w, h = rect[1]
        if min(w, h) <= 0:
            return False, 0.0
        
        aspect_ratio = max(w, h) / min(w, h)
        is_line = aspect_ratio >= min_aspect_ratio
        
        return is_line, aspect_ratio

    def _match_bomb_defect_code(self, bomb_info: Dict[str, Any]) -> str:
        """
        從 config 的 bomb_defects 中查找匹配的 defect_code
        
        匹配策略:
            1. 比對 image_prefix
            2. 比對 defect_type
            3. 找到 → 返回 defect_code；找不到 → 返回 "UNKNOWN"
        """
        target_prefix = bomb_info["image_prefix"]
        target_type = bomb_info["defect_type"]
        
        for bomb in self.config.bomb_defects:
            if (bomb.image_prefix == target_prefix and 
                bomb.defect_type == target_type):
                return bomb.defect_code
        
        # 若只有 prefix 匹配 (不分 type)，也可以 fallback
        for bomb in self.config.bomb_defects:
            if bomb.image_prefix == target_prefix:
                return bomb.defect_code
        
        return "UNKNOWN"

    def check_bomb_match(
        self,
        image_prefix: str,
        tile_center_x: int,
        tile_center_y: int,
        raw_bounds: Tuple[int, int, int, int],
        anomaly_map: Optional[np.ndarray] = None,
        product_resolution: Optional[Tuple[int, int]] = None,
        bomb_list: Optional[List] = None,
        skip_shape_check: bool = False,
    ) -> Tuple[bool, str]:
        """
        檢查異常 tile 是否匹配炸彈系統的已知座標
        
        Args:
            image_prefix: 圖片檔名前綴 (e.g. "G0F00000")
            tile_center_x: 異常 tile 中心 x (圖片座標)
            tile_center_y: 異常 tile 中心 y (圖片座標)
            raw_bounds: 原始 Otsu 邊界 (用於座標轉換)
            anomaly_map: 該 tile 的 anomaly map (用於 line 型形態驗證)
            
        Returns:
            (is_bomb, defect_code) - 是否為炸彈, 對應的 Defect Code
        """
        if product_resolution is None:
            product_resolution = DEFAULT_PRODUCT_RESOLUTION
        tolerance = self.config.bomb_match_tolerance
    
        # 使用傳入的 bomb_list 或 config 預設值
        bombs = bomb_list if bomb_list is not None else self.config.bomb_defects
    
        for bomb in bombs:
            # 比對前綴 (支援帶時間戳的檔名, e.g. "G0F00000" 匹配 "G0F00000_031447")
            if not (image_prefix == bomb.image_prefix or 
                    image_prefix.startswith(bomb.image_prefix + "_")):
                continue
            
            if bomb.defect_type == "line" and len(bomb.coordinates) >= 2:
                # 線型: 將 tile 位置轉回產品座標，判斷是否在線段容忍帶內
                pt1 = bomb.coordinates[0]
                pt2 = bomb.coordinates[1]
                product_width, product_height = product_resolution
                x_start, y_start, x_end, y_end = raw_bounds
                product_x = (tile_center_x - x_start) * product_width / (x_end - x_start)
                product_y = (tile_center_y - y_start) * product_height / (y_end - y_start)

                if self._point_within_line_segment_tolerance(
                    product_x, product_y, pt1, pt2, tolerance
                ):
                    # 額外驗證：heatmap 是否呈現線狀形態
                    if anomaly_map is not None and not skip_shape_check:
                        is_line, aspect_ratio = self._check_heatmap_line_shape(
                            anomaly_map,
                            min_aspect_ratio=self.config.bomb_line_min_aspect_ratio,
                        )
                        if not is_line:
                            print(f"⚠️ BOMB line 位置匹配但熱力圖非線狀 (aspect_ratio={aspect_ratio:.2f} < {self.config.bomb_line_min_aspect_ratio})，跳過 {bomb.defect_code}")
                            continue
                    return True, bomb.defect_code
                    
            elif bomb.defect_type == "point":
                # 點型: 判斷 tile 中心是否在任一炸彈點座標 ± tolerance 範圍內
                product_width, product_height = product_resolution
                x_start, y_start, x_end, y_end = raw_bounds
                scale_x = (x_end - x_start) / product_width
                scale_y = (y_end - y_start) / product_height
                img_tolerance_x = int(tolerance * scale_x)
                img_tolerance_y = int(tolerance * scale_y)
                
                for coord in bomb.coordinates:
                    img_bx, img_by = self._map_aoi_coords(coord[0], coord[1], raw_bounds, product_resolution)
                    if (abs(tile_center_x - img_bx) <= img_tolerance_x and
                        abs(tile_center_y - img_by) <= img_tolerance_y):
                        return True, bomb.defect_code
        
        return False, ""

    def process_panel(
        self,
        panel_dir: Path,
        progress_callback=None,
        cpu_workers: int = 4,
        product_resolution: Optional[Tuple[int, int]] = None,
        bomb_info: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
        machine_no: Optional[str] = None,
        aoi_report_override: Optional[Dict[str, List['AOIReportDefect']]] = None,
        machine_judgment: Optional[str] = None,
    ):
        """分發器：依 config.is_new_architecture 路由至 v1 或 v2 實作。"""
        return self._dispatch_process_panel(
            panel_dir,
            progress_callback=progress_callback,
            cpu_workers=cpu_workers,
            product_resolution=product_resolution,
            bomb_info=bomb_info,
            model_id=model_id,
            machine_no=machine_no,
            aoi_report_override=aoi_report_override,
            machine_judgment=machine_judgment,
        )

    def _process_panel_v1(
        self,
        panel_dir: Path,
        progress_callback=None,
        cpu_workers: int = 4,
        product_resolution: Optional[Tuple[int, int]] = None,
        bomb_info: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
        machine_no: Optional[str] = None,
        aoi_report_override: Optional[Dict[str, List['AOIReportDefect']]] = None,
        machine_judgment: Optional[str] = None,
    ) -> List[ImageResult]:
        """
        處理整個面板的圖片 (包含 PINIGBI 灰塵檢查 和 AOI Defect 整合)
        
        使用三階段平行處理：
          Phase 1: 多執行緒平行預處理 (imread + Otsu + tiling) — CPU bound, OpenCV 釋放 GIL
          Phase 2: 序列 GPU 推論 (predict_tile) — GPU bound
          Phase 3: 多執行緒平行後處理 (灰塵 IOU 計算) — CPU bound
        
        Args:
            panel_dir: 面板資料夾路徑
            progress_callback: 進度回呼
            cpu_workers: CPU 平行化的執行緒數量 (預設 4)
            
        Returns:
            (該面板所有圖片的推論結果, OMIT 視覺化圖片(可選))
        """
        image_files, is_duplicate = self._prepare_panel_image_files(panel_dir)
        results = []

        # 如果有的話，解析 Defect.txt
        defect_map = self._parse_defect_txt(panel_dir / "Defect.txt")
        
        # 1. 分離一般圖片和灰塵檢查用圖片 (支援 OMIT0000 和 PINIGBI 兩種格式)
        def is_dust_check_image(f):
            return f.stem.startswith("PINIGBI") or "OMIT0000" in f.name
        omit_files = [f for f in image_files if is_dust_check_image(f)]
        normal_files = [f for f in image_files if not is_dust_check_image(f)]
        panel_mark_detection, panel_mark_regions = self._detect_panel_mark_binary_region(
            normal_files,
            machine_no=machine_no,
            model_id=model_id,
        )

        # 載入 OMIT 圖片 (如果有)
        omit_image = None
        omit_overexposed = False
        omit_overexposure_info = ""
        if omit_files:
            omit_path = omit_files[0]
            # 載入 OMIT 圖片 (保持原始深度)
            omit_image = self._read_detection_image(omit_path)
            if omit_image is None:
                print(f"⚠️ 無法載入 OMIT 圖片: {omit_path}")
        
        # 過曝檢查 (在灰塵檢測之前)
        if omit_image is not None:
            omit_overexposed, oe_mean, oe_ratio, oe_detail = self.check_omit_overexposure(omit_image)
            omit_overexposure_info = oe_detail
            if omit_overexposed:
                print(f"⚠️ OMIT OVEREXPOSED [{omit_path.name}]: {oe_detail}")
                print(f"   → Dust detection DISABLED for this panel (unreliable due to overexposure)")
            else:
                print(f"✅ OMIT exposure OK [{omit_path.name}]: {oe_detail}")
        
        # 準備 OMIT 視覺化圖 (BGR)
        omit_vis = None
        if omit_image is not None:
             omit_vis = omit_image.copy()
             if len(omit_vis.shape) == 2:
                 omit_vis = cv2.cvtColor(omit_vis, cv2.COLOR_GRAY2BGR)
             elif len(omit_vis.shape) == 3 and omit_vis.shape[2] == 1:
                 omit_vis = cv2.cvtColor(omit_vis, cv2.COLOR_GRAY2BGR)
             
             # 在過曝的 OMIT 圖上標記警告
             if omit_overexposed:
                 h, w = omit_vis.shape[:2]
                 # 半透明紅色覆蓋
                 overlay = omit_vis.copy()
                 cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 200), -1)
                 cv2.addWeighted(overlay, 0.5, omit_vis, 0.5, 0, omit_vis)
                 # 警告文字
                 cv2.putText(omit_vis, "!! OVEREXPOSED !!", (20, 50),
                             cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
                 cv2.putText(omit_vis, oe_detail, (20, 100),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # === MARK 位置快取 (Panel 級共用) ===
        # 掃描所有圖片，找到第一個成功匹配的 MARK 位置
        cached_mark = None
        has_mark_zone = any(
            z.type == "template_match" and z.name == "mark_area" 
            for z in self.config.get_enabled_exclusion_zones()
        )
        
        if has_mark_zone and self.config.otsu_bottom_crop <= 0:
            # 只有在需要 MARK 排除且沒有底部裁切時才掃描
            files_to_scan = [f for f in normal_files if not self.config.should_skip_file(f.name)]
            for scan_path in files_to_scan:
                try:
                    scan_img = self._read_detection_image(scan_path)
                    if scan_img is not None:
                        mark_region = self.find_mark_region(scan_img)
                        if mark_region:
                            cached_mark = mark_region
                            print(f"✅ MARK 位置已找到 (來源: {scan_path.name}) → ({mark_region.x1}, {mark_region.y1})-({mark_region.x2}, {mark_region.y2})")
                            break
                except Exception as e:
                    print(f"⚠️ MARK 掃描失敗 ({scan_path.name}): {e}")
                    continue
            
            # 若全部失敗，使用 Fallback 預設位置
            if cached_mark is None and self.config.mark_fallback_position:
                pos = self.config.mark_fallback_position
                cached_mark = ExclusionRegion(
                    name="mark_area",
                    x1=pos['x'],
                    y1=pos['y'],
                    x2=pos['x'] + pos['width'],
                    y2=pos['y'] + pos['height'],
                )
                print(f"⚠️ MARK 模板匹配全部失敗，使用 Fallback 預設位置 → ({cached_mark.x1}, {cached_mark.y1})-({cached_mark.x2}, {cached_mark.y2})")
            elif cached_mark is None:
                print(f"❌ MARK 模板匹配全部失敗，且未設定 Fallback 位置")
        
        # === 統一參考邊界 (Panel 級)：依優先序挑一張圖計算，套用到所有圖片 ===
        # 同一 panel 的所有圖片皆為同一位置拍攝，理論邊界相同；各圖獨立 OTSU
        # 在光源較弱的機種（如 G0F / R0F / STANDARD）會抓歪 polygon，連帶影響
        # tiling / mask / exclusion。這裡以 W0F (白光) 優先作為參考，其他圖套用。
        # B0F (暗色) 本來就無法獨立偵測邊界，同樣走這條路徑。
        _DARK_PREFIXES = ("B0F",)  # 不得作為 reference 的暗色圖片前綴
        _REFERENCE_PRIORITY = ("W0F", "WGF", "G0F", "R0F", "STANDARD")

        def _prefix_rank(filename: str) -> int:
            up = filename.upper()
            for i, p in enumerate(_REFERENCE_PRIORITY):
                if up.startswith(p):
                    return i
            return len(_REFERENCE_PRIORITY)  # 兜底：其他非暗色非 OMIT

        panel_reference_raw_bounds: Optional[Tuple[int, int, int, int]] = None
        panel_reference_polygon: Optional[np.ndarray] = None

        ref_candidates = sorted(
            [f for f in normal_files
             if not f.name.upper().startswith(_DARK_PREFIXES)
             and not is_dust_check_image(f)],
            key=lambda f: _prefix_rank(f.name),
        )
        for ref_path in ref_candidates:
            try:
                ref_img = self._read_detection_image(ref_path)
                if ref_img is None:
                    continue
                if self._use_robust_panel_boundary():
                    ref_bounds, ref_polygon = self._find_robust_object_bounds(ref_img)
                    ref_binary = None
                else:
                    ref_bounds, ref_binary = self._find_raw_object_bounds(ref_img)
                    ref_polygon = None
                panel_reference_raw_bounds = ref_bounds
                if self.config.enable_panel_polygon:
                    panel_reference_polygon = (
                        ref_polygon.copy()
                        if ref_polygon is not None
                        else self._find_panel_polygon(ref_binary, ref_bounds)
                    )
                    poly_str = "有" if panel_reference_polygon is not None else "品質不足"
                else:
                    poly_str = "關閉"
                print(f"📐 統一參考邊界 (Panel 級) 已從 {ref_path.name} 計算 → "
                      f"{panel_reference_raw_bounds} (polygon: {poly_str})")
                break
            except Exception as e:
                print(f"⚠️ 計算參考邊界失敗 ({ref_path.name}): {e}")
                continue
        if panel_reference_raw_bounds is None:
            print("⚠️ 無法計算統一參考邊界，所有圖片將各自計算 OTSU (可能不一致)")

        # 過濾出需要處理的檔案
        files_to_process = [f for f in normal_files if not self.config.should_skip_file(f.name)]
        skipped = [f.name for f in normal_files if self.config.should_skip_file(f.name)]
        if skipped:
            print(f"⏭️ 跳過檔案 (設定) ×{len(skipped)}: {', '.join(skipped)}")
        
        total_files = len(files_to_process)
        if total_files == 0:
            return results, omit_vis, omit_overexposed, omit_overexposure_info, False, omit_image, {}
        
        # 決定實際 worker 數量 (不超過檔案數量)
        actual_workers = min(cpu_workers, total_files)
        print(f"🔀 平行處理: {total_files} 張圖片, {actual_workers} 個 CPU 執行緒")
        
        # ================================================================
        # Phase 1: 多執行緒平行預處理 (imread + Otsu + tiling)
        # OpenCV 在 C 層釋放 GIL，多執行緒可獲得真正的平行加速
        # ================================================================
        def _preprocess_one(img_path: Path) -> Optional[ImageResult]:
            """單張圖片的預處理 (可安全在多執行緒中呼叫)"""
            # 所有圖片套用統一參考邊界 (若計算成功)
            result = self.preprocess_image(
                img_path,
                cached_mark=cached_mark,
                reference_raw_bounds=panel_reference_raw_bounds,
                reference_polygon=panel_reference_polygon,
            )
            if result is None:
                return None
            
            # 整合 AOI Defect Data
            stem = img_path.stem
            if stem in defect_map and result.raw_bounds is not None:
                raw_bounds = result.raw_bounds
                img_w, img_h = result.image_size
                
                for d in defect_map[stem]:
                    img_x, img_y = self._map_aoi_coords(d['x'], d['y'], raw_bounds, product_resolution)
                    
                    # 定義 AOI 標記框 (約 50x50)
                    size = 50
                    x1 = max(0, img_x - size // 2)
                    y1 = max(0, img_y - size // 2)
                    x2 = min(img_w, img_x + size // 2)
                    y2 = min(img_h, img_y + size // 2)
                    
                    result.aoi_defects.append(AOIDefect(
                        defect_code=d['defect_code'],
                        product_x=d['x'], 
                        product_y=d['y'],
                        image_x=img_x,
                        image_y=img_y,
                        bounds=(x1, y1, x2, y2)
                    ))
            
            return result
        
        preprocess_start = time.time()
        preprocessed_results = []
        
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # 提交所有預處理任務，保持原始順序
            # 每次 submit 都 copy_context()，因為同一 Context 物件不能被多個 worker 同時 enter；
            # copy_context 會 snapshot 當前 thread 的 ContextVar values 含 InferenceLogCapture buffer
            future_to_path = {}
            for img_path in files_to_process:
                ctx = contextvars.copy_context()
                future = executor.submit(ctx.run, _preprocess_one, img_path)
                future_to_path[future] = img_path
            
            # 按提交順序收集結果 (使用 dict 保持對應)
            results_map = {}
            for future in as_completed(future_to_path):
                img_path = future_to_path[future]
                try:
                    result = future.result()
                    if result is not None:
                        results_map[img_path] = result
                except Exception as e:
                    print(f"⚠️ 預處理失敗 ({img_path.name}): {e}")
            
            # 按原始檔案順序排列結果
            for img_path in files_to_process:
                if img_path in results_map:
                    preprocessed_results.append(results_map[img_path])
        
        preprocess_time = time.time() - preprocess_start
        print(f"⚡ Phase 1 完成: {len(preprocessed_results)} 張圖片預處理耗時 {preprocess_time:.2f}s (平行 {actual_workers} 執行緒)")

        # ================================================================
        # Phase 1.5: AOI 機檢座標目標切塊
        # 解析 AOI 機台 NG 報告，以缺陷座標為中心建立額外的 512x512 tiles
        # ================================================================
        aoi_report = {}
        aoi_report_for_inference = {}
        if self.config.aoi_coord_inspection_enabled:
            aoi_report = aoi_report_override if aoi_report_override is not None else self._parse_aoi_report_txt(panel_dir)
            aoi_report_for_inference, _forced_bomb_count = self._aoi_report_with_forced_client_bomb_coords(
                aoi_report,
                bomb_info,
            )
            if aoi_report_for_inference:
                # 收集已有的圖片前綴
                existing_prefixes = set()
                for result in preprocessed_results:
                    existing_prefixes.add(self._get_image_prefix(result.image_path.name))

                # v1-only: 對 skip_files 中有 AOI 報告的圖片，預處理後加入
                for report_prefix in aoi_report_for_inference:
                    if report_prefix not in existing_prefixes:
                        matched_file = None
                        skipped_files = [f for f in image_files
                                         if self.config.should_skip_file(f.name)
                                         and not is_dust_check_image(f)]
                        for f in skipped_files:
                            if self._get_image_prefix(f.name) == report_prefix:
                                matched_file = f
                                break
                        if matched_file is not None:
                            print(f"🎯 AOI Coord: 為跳過的圖片 {matched_file.name} 建立預處理 "
                                  f"(有 {len(aoi_report_for_inference[report_prefix])} 筆 AOI/forced 座標)")
                            skip_result = self.preprocess_image(
                                matched_file,
                                cached_mark=cached_mark,
                                reference_raw_bounds=panel_reference_raw_bounds,
                                reference_polygon=panel_reference_polygon,
                            )
                            if skip_result is not None:
                                skip_result.tiles = []
                                skip_result.excluded_tile_count = 0
                                skip_result.processed_tile_count = 0
                                preprocessed_results.append(skip_result)
                                existing_prefixes.add(report_prefix)

                stats = self._apply_aoi_coord_inspection(
                    panel_dir=panel_dir,
                    preprocessed_results=preprocessed_results,
                    omit_image=omit_image,
                    omit_overexposed=omit_overexposed,
                    product_resolution=product_resolution,
                    aoi_report=aoi_report_for_inference,
                )
                print(f"🎯 Phase 1.5 完成: AOI 座標新增 {stats['aoi_tile_count']} 個 tiles, "
                      f"{stats['aoi_edge_count']} 個邊緣 defects")
        elif bomb_info is not None and getattr(self.config, "bomb_area_force_detection_enabled", False):
            print("💣 [BOMB_FORCE] 已啟用但 aoi_coord_inspection_enabled=False，無法補切 Client 炸彈座標")

        self._attach_panel_mark_binary_to_results(
            preprocessed_results,
            panel_mark_detection,
            panel_mark_regions,
        )

        # === Grid Tiling 開關控制 ===
        # 如果 grid_tiling_enabled=False，移除非 AOI coord 的 tiles (只推論 AOI 座標 tiles)
        if not self.config.grid_tiling_enabled:
            for result in preprocessed_results:
                original_count = len(result.tiles)
                result.tiles = [t for t in result.tiles if t.is_aoi_coord_tile]
                removed = original_count - len(result.tiles)
                if removed > 0:
                    print(f"⏭️ Grid Tiling 關閉: {result.image_path.name} 移除 {removed} 個 grid tiles，保留 {len(result.tiles)} 個 AOI tiles")

        # ================================================================
        # Phase 2: 序列 GPU 推論 (predict_tile)
        # PyTorch GPU 推論不適合跨執行緒平行化，保持序列執行
        # ================================================================
        inference_start = time.time()
        for i, result in enumerate(preprocessed_results):
            # 多模型路由：依圖片前綴選擇對應的 inferencer 和 threshold
            img_prefix = self._get_image_prefix(result.image_path.name)

            # === skip_files 圖片（如 B0F00000）：使用二值化偵測取代 PatchCore ===
            if self.config.should_skip_file(result.image_path.name):
                print(f"💡 {result.image_path.name} (skip_file) → 使用二值化偵測亮點")
                anomaly_tiles = []
                for tile in result.tiles:
                    score, anomaly_map = self._detect_bright_spots(tile)
                    tile.score_threshold = 0.5
                    if score <= 0:
                        # 未偵測到亮點，標記為 below_threshold 不影響判定，但仍保留以便查看原圖
                        tile.is_aoi_coord_below_threshold = True
                    anomaly_tiles.append((tile, score, anomaly_map))
                result.anomaly_tiles = anomaly_tiles
                result.inference_time = 0.0
                preprocessed_results[i] = result
                if progress_callback:
                    progress_callback(i + 1, len(preprocessed_results))
                continue

            target_inferencer = self._get_inferencer_for_prefix(img_prefix)
            target_threshold = self._get_threshold_for_prefix(img_prefix)

            if target_inferencer is None:
                print(f"⚠️ {result.image_path.name} 無可用模型，跳過推論")
                continue

            # 模型路由 log (僅在多模型模式下顯示)
            if self._model_mapping:
                model_name = "fallback"
                if img_prefix in self._model_mapping:
                    model_name = self._model_mapping[img_prefix].name
                print(f"🎯 {result.image_path.name} → 模型: {model_name}, 閾值: {target_threshold}")

            result = self.run_inference(
                result,
                inferencer=target_inferencer,
                threshold=target_threshold,
                model_id=model_id,
            )
            preprocessed_results[i] = result

            if progress_callback:
                progress_callback(i + 1, len(preprocessed_results))
        
        inference_time = time.time() - inference_start
        print(f"🔥 Phase 2 完成: GPU 推論耗時 {inference_time:.2f}s")
        
        # ================================================================
        # Phase 3: 多執行緒平行後處理 (灰塵 IOU 交叉驗證)
        # ================================================================
        def _dust_check_one(result: ImageResult) -> ImageResult:
            """單張圖片的灰塵交叉驗證 (可安全在多執行緒中呼叫)"""
            img_path = result.image_path

            # skip_files 圖片（如 B0F00000）不做 OMIT 灰塵比對
            if self.config.should_skip_file(img_path.name):
                return result

            if result.anomaly_tiles and omit_image is not None and omit_overexposed:
                # OMIT 過曝：無法進行灰塵檢測，記錄但不判定
                for tile, score, anomaly_map in result.anomaly_tiles:
                    if getattr(tile, 'is_aoi_coord_below_threshold', False):
                        tile.dust_detail_text = (
                            f"OMIT_OVEREXPOSED ({omit_overexposure_info}) -> "
                            "TRACK_ONLY Score<THR, dust check skipped"
                        )
                    else:
                        tile.dust_detail_text = f"OMIT_OVEREXPOSED ({omit_overexposure_info}) -> Cannot verify dust, treated as REAL_NG"
                    aoi_suffix = self._format_aoi_tile_log_suffix(tile)
                    print(f"⚠️ {img_path.name} Tile@({tile.x},{tile.y}){aoi_suffix} Score:{score:.3f} → OMIT OVEREXPOSED, skip dust check")
            elif result.anomaly_tiles and omit_image is not None and not omit_overexposed:
                for tile, score, anomaly_map in result.anomaly_tiles:
                    # 在 OMIT 圖片上裁切相同區域
                    tx, ty, tw, th = tile.x, tile.y, tile.width, tile.height
                    
                    # 邊界檢查
                    oh, ow = omit_image.shape[:2]
                    if tx < ow and ty < oh:
                        x2 = min(tx + tw, ow)
                        y2 = min(ty + th, oh)
                        
                        omit_crop = omit_image[ty:y2, tx:x2]
                        tile.omit_crop_image = omit_crop.copy()
                        focus_image_x = int(getattr(tile, 'aoi_image_x', -1))
                        if focus_image_x < 0:
                            focus_image_x = int(getattr(tile, 'anomaly_peak_x', -1))
                        context_focus_x = focus_image_x - tx if focus_image_x >= 0 else None
                        
                        # Step A: 進階灰塵偵測 (CLAHE + Otsu + 面積篩選)
                        is_dust, dust_mask, bright_ratio, detail_text = \
                            self._check_dust_or_scratch_feature_with_context(
                                omit_image,
                                tx,
                                ty,
                                tw,
                                th,
                                omit_crop,
                                focus_x=context_focus_x,
                                product_resolution=product_resolution,
                            )
                        tile.dust_mask = dust_mask
                        tile.dust_bright_ratio = bright_ratio

                        if getattr(tile, 'is_aoi_coord_below_threshold', False):
                            tile.is_suspected_dust_or_scratch = False
                            detail_text += " TRACK_ONLY Score<THR -> AI_OK"
                            tile.dust_detail_text = detail_text
                            aoi_suffix = self._format_aoi_tile_log_suffix(tile)
                            print(
                                f"🟡 {img_path.name} Tile@({tx},{ty}){aoi_suffix} "
                                f"Score:{score:.3f} → {detail_text}"
                            )
                            continue
                        
                        # Step B: 逐區域灰塵交叉驗證 (Per-Region Dust Filtering)
                        iou = 0.0
                        heatmap_binary = None
                        top_pct = self.config.dust_heatmap_top_percent
                        metric_mode = self.config.dust_heatmap_metric
                        metric_name = "COV" if metric_mode == "coverage" else "IOU"

                        if is_dust and anomaly_map is not None:
                            # 逐區域判定：拆開異常連通區域，各自與灰塵比對
                            aoi_seed_yx, aoi_seed_radius, aoi_seed_min_score = \
                                self._aoi_center_seed_for_tile(tile, anomaly_map)
                            has_real_defect, real_peak_yx, overall_iou, region_details, heatmap_binary, region_labels = \
                                self.check_dust_per_region(
                                    dust_mask, anomaly_map,
                                    top_percent=top_pct,
                                    metric=metric_mode,
                                    iou_threshold=self.config.dust_heatmap_iou_threshold,
                                    force_include_yx=aoi_seed_yx,
                                    force_include_radius=aoi_seed_radius,
                                    force_include_min_score=aoi_seed_min_score,
                                )
                            iou = overall_iou
                            tile.dust_heatmap_iou = iou
                            # 記錄 per-region 最大 coverage（實際判定用的值）
                            tile.dust_region_details = region_details
                            tile.dust_heatmap_binary = heatmap_binary
                            if region_details:
                                tile.dust_region_max_cov = max(r["coverage"] for r in region_details)

                            dust_regions = [r for r in region_details if r["is_dust"]]
                            real_regions = [r for r in region_details if not r["is_dust"]]

                            _two_stage_ran = False
                            _ts_features = []
                            _ts_dust_mask_no_ext = None

                            if has_real_defect:
                                # 有非灰塵的真實異常區域 → 保留為 NG
                                tile.is_suspected_dust_or_scratch = False
                                detail_text += (
                                    f" PER_REGION: {len(real_regions)}real+"
                                    f"{len(dust_regions)}dust -> REAL_NG"
                                )

                                # 更新 peak 座標到非灰塵區域的最大值位置
                                if real_peak_yx is not None:
                                    amap_h, amap_w = anomaly_map.shape[:2]
                                    tile.anomaly_peak_y = tile.y + int(real_peak_yx[0] * tile.height / amap_h)
                                    tile.anomaly_peak_x = tile.x + int(real_peak_yx[1] * tile.width / amap_w)
                            else:
                                # 所有異常區域都與灰塵重疊 → 初步標記為灰塵
                                # 如果啟用兩階段判定，進行二次確認
                                if self.config.dust_two_stage_enabled:
                                    # 兩階段: 用原圖精準定位 feature 點，比對 dust_mask (ext=0)
                                    dust_mask_no_ext = None
                                    if omit_crop is not None:
                                        _, dust_mask_no_ext, _, _ = \
                                            self._check_dust_or_scratch_feature_with_context(
                                                omit_image,
                                                tx,
                                                ty,
                                                tw,
                                                th,
                                                omit_crop,
                                                extension_override=0,
                                                focus_x=context_focus_x,
                                                product_resolution=product_resolution,
                                            )
                                    ts_has_real, ts_peak_yx, ts_features, ts_detail = \
                                        self.check_dust_two_stage(
                                            tile.image, anomaly_map,
                                            dust_mask_no_ext if dust_mask_no_ext is not None else dust_mask,
                                            score,
                                            score_threshold=tile.score_threshold,
                                            candidate_dust_mask=dust_mask,
                                        )
                                    _two_stage_ran = True
                                    _ts_features = ts_features
                                    _ts_dust_mask_no_ext = dust_mask_no_ext
                                    tile.dust_two_stage_features = ts_features
                                    tile.dust_two_stage_dust_mask = dust_mask_no_ext if dust_mask_no_ext is not None else dust_mask
                                    if ts_has_real:
                                        tile.is_suspected_dust_or_scratch = False
                                        detail_text += (
                                            f" PER_REGION: 0real+{len(dust_regions)}dust"
                                            f" -> {ts_detail}"
                                        )
                                        if ts_peak_yx is not None:
                                            amap_h, amap_w = anomaly_map.shape[:2]
                                            tile.anomaly_peak_y = tile.y + int(ts_peak_yx[0] * tile.height / amap_h)
                                            tile.anomaly_peak_x = tile.x + int(ts_peak_yx[1] * tile.width / amap_w)
                                    else:
                                        tile.is_suspected_dust_or_scratch = True
                                        detail_text += (
                                            f" PER_REGION: 0real+{len(dust_regions)}dust"
                                            f" -> {ts_detail}"
                                        )
                                else:
                                    tile.is_suspected_dust_or_scratch = True
                                    detail_text += (
                                        f" PER_REGION: 0real+"
                                        f"{len(dust_regions)}dust -> DUST"
                                    )

                            # 產生 Debug 可視化圖
                            try:
                                if _two_stage_ran:
                                    dm_for_debug = _ts_dust_mask_no_ext if _ts_dust_mask_no_ext is not None else dust_mask
                                    tile.dust_iou_debug_image = self.generate_two_stage_debug_image(
                                        tile.image, anomaly_map, dm_for_debug,
                                        _ts_features,
                                        tile.is_suspected_dust_or_scratch,
                                    )
                                else:
                                    tile.dust_iou_debug_image = self.generate_dust_iou_debug_image(
                                        tile.image, anomaly_map, dust_mask,
                                        heatmap_binary, iou, top_pct,
                                        tile.is_suspected_dust_or_scratch,
                                        region_details=region_details,
                                        region_labels=region_labels,
                                    )
                            except Exception as dbg_err:
                                print(f"⚠️ Debug image generation failed: {dbg_err}")
                        elif is_dust:
                            # 有灰塵但沒有 heatmap → 保守標記為灰塵
                            tile.is_suspected_dust_or_scratch = True
                            detail_text += " (no heatmap, marked as dust)"
                        else:
                            detail_text += " NO_DUST -> REAL_NG"
                        
                        tile.dust_detail_text = detail_text
                        
                        log_icon = "🧹" if tile.is_suspected_dust_or_scratch else "🔴"
                        aoi_suffix = self._format_aoi_tile_log_suffix(tile)
                        print(f"{log_icon} {img_path.name} Tile@({tx},{ty}){aoi_suffix} → {detail_text}")

            # === 加入 CV Edge 灰塵檢測 (與 OMIT 擷取) ===
            # 注意：inspector_mode == "fusion" 的 defect 已在 _inspect_roi_fusion
            # 內透過 _apply_omit_dust_filter_to_edge_defects 完成灰塵判定，
            # 此處需跳過以免重覆計算並覆寫 fusion 已寫入的欄位 (spec:
            # docs/superpowers/specs/2026-04-22-aoi-edge-fusion-inspector-design.md)
            if getattr(result, 'edge_defects', []) and omit_image is not None:
                if omit_overexposed:
                    for ed in result.edge_defects:
                        if getattr(ed, 'inspector_mode', 'cv') == 'fusion':
                            continue
                        ed.dust_detail_text = f"OMIT_OVEREXPOSED ({omit_overexposure_info}) -> Cannot verify dust, treated as REAL_NG"
                        print(f"⚠️ {img_path.name} Edge@{ed.side} Score:{ed.max_diff:.3f} → OMIT OVEREXPOSED, skip dust check")
                else:
                    # 讀取原始圖片一次，供所有 edge defect 的 defect mask 計算
                    orig_for_edge = None
                    if getattr(self, 'edge_inspector', None) and self.edge_inspector.config.dust_filter_enabled:
                        orig_for_edge = self._read_detection_image(img_path)

                    for ed in result.edge_defects:
                        if getattr(ed, 'inspector_mode', 'cv') == 'fusion':
                            continue
                        ex, ey, ew, eh = ed.bbox
                        oh, ow = omit_image.shape[:2]
                        # 使用擴展 ROI (bbox ± 100px)，與 save_edge_defect_image 一致
                        # 原始 bbox 可能極小 (如 4x15px)，無法可靠偵測灰塵
                        edge_dust_padding = 100
                        tx = max(0, ex - edge_dust_padding)
                        ty = max(0, ey - edge_dust_padding)
                        x2 = min(ex + ew + edge_dust_padding, ow)
                        y2 = min(ey + eh + edge_dust_padding, oh)
                        if tx < ow and ty < oh:

                            omit_crop = omit_image[ty:y2, tx:x2]
                            ed.omit_crop_image = omit_crop.copy()

                            if getattr(self, 'edge_inspector', None) and self.edge_inspector.config.dust_filter_enabled:
                                is_dust, dust_mask, bright_ratio, detail_text = \
                                    self.check_dust_or_scratch_feature(
                                        omit_crop,
                                        product_resolution=product_resolution,
                                    )
                                ed.dust_mask = dust_mask
                                ed.dust_bright_ratio = bright_ratio

                                metric_mode = self.config.dust_heatmap_metric
                                metric_name = "COV" if metric_mode == "coverage" else "IOU"

                                if is_dust and dust_mask is not None and orig_for_edge is not None:
                                    # Step B: 空間重疊驗證 — 使用實際 CV defect mask (與 heatmap 一致)
                                    # 從原始圖片裁切相同 ROI，重建 CV 缺陷二值 mask
                                    crop_h, crop_w = omit_crop.shape[:2]
                                    orig_crop = orig_for_edge[ty:y2, tx:x2]
                                    if len(orig_crop.shape) == 3:
                                        orig_gray = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2GRAY)
                                    else:
                                        orig_gray = orig_crop

                                    # 與 save_edge_defect_image / _inspect_side 相同的 CV 檢測邏輯
                                    ecfg = self.edge_inspector.config
                                    ek = ecfg.blur_kernel
                                    emk = clamp_median_kernel(ecfg.median_kernel, min(orig_gray.shape[:2]) - 1)
                                    eblurred = cv2.GaussianBlur(orig_gray, (ek, ek), 0)

                                    if ed.side == "aoi_edge":
                                        _, ediff = compute_fg_aware_diff(eblurred, orig_gray, emk)
                                    else:
                                        ebg = cv2.medianBlur(eblurred, emk)
                                        ediff = cv2.absdiff(eblurred, ebg)

                                    edge_threshold = ecfg.get_threshold_for_side(ed.side)
                                    _, defect_mask_cv = cv2.threshold(ediff, edge_threshold, 255, cv2.THRESH_BINARY)

                                    # 只保留缺陷 BBox 範圍內的像素（與 heatmap 一致）
                                    rel_x = ex - tx
                                    rel_y = ey - ty
                                    bbox_only = np.zeros_like(defect_mask_cv)
                                    ry1 = max(0, rel_y)
                                    rx1 = max(0, rel_x)
                                    ry2 = min(defect_mask_cv.shape[0], rel_y + eh)
                                    rx2 = min(defect_mask_cv.shape[1], rel_x + ew)
                                    bbox_only[ry1:ry2, rx1:rx2] = 255
                                    defect_mask_cv = cv2.bitwise_and(defect_mask_cv, bbox_only)

                                    # dust_mask 轉單通道並對齊尺寸
                                    dm = dust_mask
                                    if len(dm.shape) == 3:
                                        dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
                                    if dm.shape[:2] != defect_mask_cv.shape[:2]:
                                        dm = cv2.resize(dm, (defect_mask_cv.shape[1], defect_mask_cv.shape[0]),
                                                        interpolation=cv2.INTER_NEAREST)

                                    defect_bool = defect_mask_cv > 0
                                    dust_bool = dm > 0
                                    intersection = np.count_nonzero(defect_bool & dust_bool)

                                    if metric_mode == "coverage":
                                        defect_area = np.count_nonzero(defect_bool)
                                        cov = float(intersection / defect_area) if defect_area > 0 else 0.0
                                    else:
                                        union = np.count_nonzero(defect_bool | dust_bool)
                                        cov = float(intersection / union) if union > 0 else 0.0

                                    if cov >= self.config.dust_heatmap_iou_threshold:
                                        ed.is_suspected_dust_or_scratch = True
                                        detail_text += f" {metric_name}:{cov:.3f}>={metric_name}_THR -> DUST (edge defect)"
                                    else:
                                        detail_text += f" {metric_name}:{cov:.3f}<{metric_name}_THR -> REAL_NG"
                                elif is_dust:
                                    # 有灰塵特徵但無法做空間驗證 → 保守視為真缺陷
                                    detail_text += " (dust detected, no spatial mask) -> REAL_NG"
                                else:
                                    detail_text += " NO_DUST -> REAL_NG"

                                ed.dust_detail_text = detail_text
                                log_icon = "🧹" if ed.is_suspected_dust_or_scratch else "🔴"
                                print(f"{log_icon} {img_path.name} Edge@{ed.side} → {detail_text}")
                            else:
                                ed.dust_detail_text = "Dust filter disabled -> REAL_NG"
            
            return result
        
        postprocess_start = time.time()
        
        # 只對有異常或有 CV 邊緣缺陷，且有 OMIT 的結果進行平行灰塵檢測
        needs_dust_check = [r for r in preprocessed_results if (r.anomaly_tiles or getattr(r, 'edge_defects', [])) and omit_image is not None]
        
        if needs_dust_check:
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                # 每次 submit 都 copy_context()，避免「Context already entered」錯誤
                dust_futures = {
                    executor.submit(contextvars.copy_context().run, _dust_check_one, r): r
                    for r in needs_dust_check
                }
                for future in as_completed(dust_futures):
                    try:
                        future.result()
                    except Exception as e:
                        r = dust_futures[future]
                        print(f"⚠️ 灰塵檢測失敗 ({r.image_path.name}): {e}")
        
        # 在 OMIT 總圖上畫框 (序列化執行以避免競爭)
        if omit_vis is not None:
            for result in preprocessed_results:
                if result.anomaly_tiles:
                    for tile, score, anomaly_map in result.anomaly_tiles:
                        tx, ty, tw, th = tile.x, tile.y, tile.width, tile.height
                        oh, ow = omit_vis.shape[:2]
                        if tx < ow and ty < oh:
                            x2 = min(tx + tw, ow)
                            y2 = min(ty + th, oh)
                            rcov = getattr(tile, 'dust_region_max_cov', 0.0)
                            metric_name = "COV" if self.config.dust_heatmap_metric == "coverage" else "IOU"
                            if tile.is_suspected_dust_or_scratch:
                                box_color = (0, 165, 255)
                                label = f"DUST R.{metric_name}:{rcov:.3f}"
                            else:
                                box_color = (0, 0, 255)
                                label = f"REAL_NG R.{metric_name}:{rcov:.3f}"
                            cv2.rectangle(omit_vis, (tx, ty), (x2, y2), box_color, 5)
                            cv2.putText(omit_vis, f"{result.image_path.name}", (tx, ty - 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, box_color, 4)
                            cv2.putText(omit_vis, label, (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, box_color, 4)
                
                # 在 OMIT 總圖上畫 Edge 框
                if getattr(result, 'edge_defects', []):
                    for ed in result.edge_defects:
                        tx, ty, tw, th = ed.bbox
                        oh, ow = omit_vis.shape[:2]
                        if tx < ow and ty < oh:
                            x2 = min(tx + tw, ow)
                            y2 = min(ty + th, oh)
                            if getattr(ed, 'is_suspected_dust_or_scratch', False):
                                box_color = (0, 165, 255)
                                label = f"Edge DUST ({ed.side})"
                            else:
                                box_color = (0, 0, 255)
                                label = f"Edge NG ({ed.side})"
                            cv2.rectangle(omit_vis, (tx, ty), (x2, y2), box_color, 5)
                            cv2.putText(omit_vis, f"{result.image_path.name}", (tx, ty - 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, box_color, 4)
                            cv2.putText(omit_vis, label, (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, box_color, 4)
        
        postprocess_time = time.time() - postprocess_start
        print(f"🧹 Phase 3 完成: 灰塵檢測後處理耗時 {postprocess_time:.2f}s")
        
        results = preprocessed_results
                
        # === 炸彈系統比對 ===
        # 決定炸彈來源：優先使用 Client 端傳入的 runtime 資料
        active_bombs = []
        if bomb_info is not None:
            # 從協議取得炸彈座標，defect_code 從 config 匹配
            defect_code = self._match_bomb_defect_code(bomb_info)
            active_bombs = [BombDefect(
                image_prefix=bomb_info["image_prefix"],
                defect_code=defect_code,
                defect_type=bomb_info["defect_type"],
                coordinates=bomb_info["coordinates"],
            )]
            print(f"💣 使用協議炸彈資料: prefix={bomb_info['image_prefix']} "
                  f"type={bomb_info['defect_type']} defect_code={defect_code} "
                  f"coords={bomb_info['coordinates']}")
        elif self.config.bomb_defects:
            active_bombs = self.config.bomb_defects
            print(f"💣 使用 Config 炸彈資料: {len(active_bombs)} 組設定")

        if active_bombs:
            point_coord_count = sum(
                len(bomb.coordinates) for bomb in active_bombs if bomb.defect_type == "point"
            )
            line_def_count = sum(1 for bomb in active_bombs if bomb.defect_type == "line")
            print(
                f"💣 BOMB matching: tolerance={self.config.bomb_match_tolerance} product_px, "
                f"point_coords={point_coord_count}, line_defs={line_def_count}"
            )
            for result in results:
                if result.anomaly_tiles and result.raw_bounds is not None:
                    img_prefix = result.image_path.stem
                    for tile, score, anomaly_map in result.anomaly_tiles:
                        if getattr(tile, "is_aoi_coord_below_threshold", False):
                            continue
                        # 計算熱力圖峰值位置 (更精確的缺陷位置)
                        if anomaly_map is not None and anomaly_map.size > 0:
                            try:
                                amap_h, amap_w = anomaly_map.shape[:2]
                                # 二值化偵測 (B0F00000 等): 使用亮點重心而非 argmax
                                # argmax 在 binary map 上回傳最左上方像素，容易偏離實際缺陷位置
                                if tile.is_bright_spot_detection:
                                    ys, xs = np.where(anomaly_map > 0.5)
                                    if len(xs) > 0:
                                        centroid_x = int(np.mean(xs))
                                        centroid_y = int(np.mean(ys))
                                    else:
                                        logger.warning(f"Bright spot tile has no pixels > 0.5: {result.image_path.name} Tile@({tile.x},{tile.y})")
                                        centroid_x = amap_w // 2
                                        centroid_y = amap_h // 2
                                    tile.anomaly_peak_x = tile.x + int(centroid_x * tile.width / amap_w)
                                    tile.anomaly_peak_y = tile.y + int(centroid_y * tile.height / amap_h)
                                else:
                                    peak_local = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
                                    # anomaly_map 尺寸可能和 tile 不同，需要縮放
                                    tile.anomaly_peak_y = tile.y + int(peak_local[0] * tile.height / amap_h)
                                    tile.anomaly_peak_x = tile.x + int(peak_local[1] * tile.width / amap_w)
                            except Exception as e:
                                logger.warning(f"Anomaly peak calculation failed: {e}")
                                tile.anomaly_peak_x, tile.anomaly_peak_y = tile.center
                        else:
                            tile.anomaly_peak_x, tile.anomaly_peak_y = tile.center

                        # 用熱力圖峰值座標來比對炸彈
                        is_bomb, bomb_code = self.check_bomb_match(
                            img_prefix, tile.anomaly_peak_x, tile.anomaly_peak_y, result.raw_bounds,
                            anomaly_map=anomaly_map, product_resolution=product_resolution,
                            bomb_list=active_bombs,
                        )
                        # AOI coord tile fallback: 若峰值未匹配，改用 tile 中心再試一次
                        # AOI coord tile 本身就是以 AOI 座標為中心切塊，中心位置更可靠
                        if not is_bomb and tile.is_aoi_coord_tile:
                            tile_cx, tile_cy = tile.center
                            is_bomb, bomb_code = self.check_bomb_match(
                                img_prefix, tile_cx, tile_cy, result.raw_bounds,
                                anomaly_map=anomaly_map, product_resolution=product_resolution,
                                bomb_list=active_bombs,
                            )
                        # AOI coord tile 保護: peak 可能被鄰近炸彈亮點吸引，
                        # 需驗證原始 AOI 產品座標本身也在炸彈容忍範圍內
                        if is_bomb and tile.is_aoi_coord_tile and tile.aoi_product_x >= 0:
                            aoi_matches_bomb = False
                            tolerance = self.config.bomb_match_tolerance
                            for bomb in active_bombs:
                                if not (img_prefix == bomb.image_prefix or
                                        img_prefix.startswith(bomb.image_prefix + "_")):
                                    continue
                                if bomb.defect_type == "point":
                                    for coord in bomb.coordinates:
                                        if (abs(tile.aoi_product_x - coord[0]) <= tolerance and
                                            abs(tile.aoi_product_y - coord[1]) <= tolerance):
                                            aoi_matches_bomb = True
                                            break
                                elif bomb.defect_type == "line" and len(bomb.coordinates) >= 2:
                                    # line 型: 檢查 AOI 座標是否在線段緩衝帶內
                                    pt1, pt2 = bomb.coordinates[0], bomb.coordinates[1]
                                    if self._point_within_line_segment_tolerance(
                                        tile.aoi_product_x,
                                        tile.aoi_product_y,
                                        pt1,
                                        pt2,
                                        tolerance,
                                    ):
                                        aoi_matches_bomb = True
                                if aoi_matches_bomb:
                                    break
                            if not aoi_matches_bomb:
                                is_bomb = False
                                aoi_suffix = self._format_aoi_tile_log_suffix(tile)
                                print(f"🛡️ {result.image_path.name} Tile@({tile.x},{tile.y}){aoi_suffix} Peak 匹配炸彈但 AOI 座標 ({tile.aoi_product_x},{tile.aoi_product_y}) 距離炸彈超過容忍度，保留為真實缺陷")
                        if is_bomb:
                            tile.is_bomb = True
                            tile.bomb_defect_code = bomb_code

            # === Bomb line 共識機制 ===
            # 若同一條 bomb line 已有 ≥3 個 tile 通過形態驗證，
            # 則位置匹配但形態不通過的 tile 也視為 bomb（線已被確認存在）
            for bomb in active_bombs:
                if bomb.defect_type != "line":
                    continue
                for result in results:
                    if not result.anomaly_tiles or result.raw_bounds is None:
                        continue
                    img_prefix = result.image_path.stem
                    if not (img_prefix == bomb.image_prefix or
                            img_prefix.startswith(bomb.image_prefix + "_")):
                        continue
                    # 計算此 bomb line 已確認的 tile 數
                    confirmed = sum(
                        1 for t, _, _ in result.anomaly_tiles
                        if t.is_bomb and t.bomb_defect_code == bomb.defect_code
                    )
                    if confirmed < 3:
                        continue
                    # 對未匹配的 tile 做位置-only 第二輪檢查
                    for tile, score, anomaly_map in result.anomaly_tiles:
                        if getattr(tile, "is_aoi_coord_below_threshold", False):
                            continue
                        if tile.is_bomb:
                            continue
                        peak_x = getattr(tile, 'anomaly_peak_x', tile.center[0])
                        peak_y = getattr(tile, 'anomaly_peak_y', tile.center[1])
                        is_bomb, bomb_code = self.check_bomb_match(
                            img_prefix, peak_x, peak_y, result.raw_bounds,
                            anomaly_map=anomaly_map, product_resolution=product_resolution,
                            bomb_list=[bomb], skip_shape_check=True,
                        )
                        if not is_bomb and tile.is_aoi_coord_tile:
                            tile_cx, tile_cy = tile.center
                            is_bomb, bomb_code = self.check_bomb_match(
                                img_prefix, tile_cx, tile_cy, result.raw_bounds,
                                anomaly_map=anomaly_map, product_resolution=product_resolution,
                                bomb_list=[bomb], skip_shape_check=True,
                            )
                        if is_bomb:
                            tile.is_bomb = True
                            tile.bomb_defect_code = bomb_code

            # === 邊緣缺陷炸彈比對 ===
            for result in results:
                if not hasattr(result, 'edge_defects') or not result.edge_defects or result.raw_bounds is None:
                    continue
                img_prefix = result.image_path.stem
                for ed in result.edge_defects:
                    if getattr(ed, 'is_cv_ok', False):
                        continue
                    cx, cy = ed.center
                    is_bomb, bomb_code = self.check_bomb_match(
                        img_prefix, cx, cy, result.raw_bounds,
                        product_resolution=product_resolution,
                        bomb_list=active_bombs,
                    )
                    if is_bomb:
                        ed.is_bomb = True
                        ed.bomb_defect_code = bomb_code

            from collections import Counter
            for result in results:
                edge_bombs = [ed for ed in result.edge_defects if ed.is_bomb]
                summary = Counter(t.bomb_defect_code for t, _, _ in result.anomaly_tiles if t.is_bomb)
                summary.update(ed.bomb_defect_code for ed in edge_bombs)
                if summary:
                    parts = [f"{c}×{n}" for c, n in summary.items()]
                    suffix = f" (含 edge {len(edge_bombs)})" if edge_bombs else ""
                    print(f"💣 {result.image_path.name} BOMB match: {', '.join(parts)}{suffix}")

        # === 不檢測排除區域判定 (基於 peak 位置) ===
        # 排除區域來自 cv_edge_exclude_zones，原僅用於邊緣檢測，現擴展至 PatchCore 推論
        # 以熱力圖峰值 (defect 精確位置) 判斷是否落在排除區域，而非整塊 512x512 tile
        if getattr(self, "edge_inspector", None):
            try:
                resolution_code = ""
                if model_id and len(model_id) >= 6:
                    resolution_code = model_id[5].upper()
                self.edge_inspector.config.set_active_zones_for_product(resolution_code)

                active_zones = [z for z in self.edge_inspector.config.exclude_zones if z.enabled]
                if active_zones:
                    for result in results:
                        if not result.anomaly_tiles:
                            continue
                        for tile, score, anomaly_map in result.anomaly_tiles:
                            # 亮點偵測 tile (B0F 等黑圖) 不受排除區域影響
                            if tile.is_bright_spot_detection:
                                continue
                            # 使用熱力圖峰值座標 (更精確的缺陷位置)
                            if tile.anomaly_peak_x >= 0 and tile.anomaly_peak_y >= 0:
                                px, py = tile.anomaly_peak_x, tile.anomaly_peak_y
                            else:
                                px, py = tile.x + tile.width // 2, tile.y + tile.height // 2

                            for zone in active_zones:
                                # 判斷峰值點是否在排除區域內
                                if (zone.x <= px <= zone.x + zone.w and
                                    zone.y <= py <= zone.y + zone.h):
                                    tile.is_in_exclude_zone = True
                                    logger.info(f"Tile #{tile.tile_id} peak@({px},{py}) 位於排除區域 ({zone.x},{zone.y},{zone.w},{zone.h})，標記為不檢測區域")
                                    break
            except Exception as e:
                logger.error(f"排除區域檢查失敗: {e}", exc_info=True)

        sf = self._get_scratch_filter()
        if sf is not None:
            panel_tiles = panel_filtered = 0
            panel_ms = 0.0
            for result in results:
                if result.anomaly_tiles:
                    sf.apply_to_image_result(result)
                    panel_tiles += len(result.anomaly_tiles)
                    panel_filtered += getattr(result, "scratch_filter_count", 0)
                    panel_ms += getattr(result, "scratch_elapsed_ms", 0.0)
            if panel_tiles:
                logger.info("[scratch] Panel 總計: 檢查 %d tiles, filtered=%d, TT=%.1fms",
                            panel_tiles, panel_filtered, panel_ms)

        total_panel_time = preprocess_time + inference_time + postprocess_time
        print(f"📊 Panel {panel_dir.name} 總計: 預處理 {preprocess_time:.2f}s + 推論 {inference_time:.2f}s + 後處理 {postprocess_time:.2f}s = {total_panel_time:.2f}s")

        if bomb_info is not None:
            for result in results:
                result.client_bomb_info = bomb_info

        return results, omit_vis, omit_overexposed, omit_overexposure_info, is_duplicate, omit_image, aoi_report

    def _load_omit_context(self, panel_dir: Path, image_files: Optional[List[Path]] = None):
        """Load the panel-level OMIT/PINIGBI image used by dust filtering."""
        if image_files is None:
            image_files = self._list_panel_image_files(panel_dir)
        omit_files = [
            f for f in image_files
            if f.stem.startswith("PINIGBI") or "OMIT0000" in f.name
        ]
        omit_image = None
        omit_overexposed = False
        omit_overexposure_info = ""
        if omit_files:
            omit_image = self._read_detection_image(omit_files[0])
            if omit_image is not None:
                omit_overexposed, _mean, _ratio, omit_overexposure_info = \
                    self.check_omit_overexposure(omit_image)

        omit_vis = None
        if omit_image is not None:
            omit_vis = omit_image.copy()
            if len(omit_vis.shape) == 2:
                omit_vis = cv2.cvtColor(omit_vis, cv2.COLOR_GRAY2BGR)
            elif len(omit_vis.shape) == 3 and omit_vis.shape[2] == 1:
                omit_vis = cv2.cvtColor(omit_vis, cv2.COLOR_GRAY2BGR)
        return omit_vis, omit_overexposed, omit_overexposure_info, omit_image

    def _apply_cv_edge_inspection(self, result: ImageResult, model_id: Optional[str] = None) -> None:
        """Run the shared CV edge inspector for an ImageResult."""
        if not (getattr(self, "edge_inspector", None) and self.edge_inspector.config.enabled and result.raw_bounds):
            return
        try:
            resolution_code = "UNKNOWN"
            if model_id and len(model_id) >= 6:
                resolution_code = model_id[5].upper()
            self.edge_inspector.config.set_active_zones_for_product(resolution_code)
            full_image = self._read_detection_image(result.image_path)
            if full_image is not None:
                result.edge_defects = self.edge_inspector.inspect(full_image, result.raw_bounds)
        except Exception as e:
            logger.error(f"CV 邊緣檢查失敗 {result.image_path.name}: {e}", exc_info=True)

    def _apply_omit_dust_postprocess(
        self,
        results: List[ImageResult],
        omit_image: Optional[np.ndarray],
        omit_overexposed: bool,
        omit_overexposure_info: str,
        cpu_workers: int = 4,
        model_id: Optional[str] = None,
        product_resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Apply panel-level OMIT dust filtering to PatchCore anomaly tiles."""
        if omit_image is None:
            return

        def _dust_check_one(result: ImageResult) -> None:
            if self.config.should_skip_file(result.image_path.name):
                return
            if not result.anomaly_tiles:
                return
            if omit_overexposed:
                for tile, score, _anomaly_map in result.anomaly_tiles:
                    if getattr(tile, 'is_aoi_coord_below_threshold', False):
                        tile.dust_detail_text = (
                            f"OMIT_OVEREXPOSED ({omit_overexposure_info}) -> "
                            "TRACK_ONLY Score<THR, dust check skipped"
                        )
                    else:
                        tile.dust_detail_text = (
                            f"OMIT_OVEREXPOSED ({omit_overexposure_info}) -> "
                            "Cannot verify dust, treated as REAL_NG"
                        )
                    zone_tag = f" [{tile.zone}]" if tile.zone else ""
                    aoi_suffix = self._format_aoi_tile_log_suffix(tile)
                    print(
                        f"⚠️ {result.image_path.name} Tile@({tile.x},{tile.y}){zone_tag}{aoi_suffix} "
                        f"Score:{score:.3f} → OMIT OVEREXPOSED, skip dust check"
                    )
                return

            oh, ow = omit_image.shape[:2]
            for tile, score, anomaly_map in result.anomaly_tiles:
                tx, ty, tw, th = tile.x, tile.y, tile.width, tile.height
                # AOI centered tile 可能在影像外 (tx<0 或 ty<0)，必須用零填補
                # 對齊 cv2.copyMakeBorder 切 tile.image 的行為，否則 omit_crop 會
                # 變成空陣列導致下游 cv2.resize 在 save_tile_heatmap 拋例外，heatmap 不會被寫出。
                sx1, sy1 = max(0, tx), max(0, ty)
                sx2 = min(ow, tx + tw)
                sy2 = min(oh, ty + th)
                if omit_image.ndim == 3:
                    omit_crop = np.zeros((th, tw, omit_image.shape[2]), dtype=omit_image.dtype)
                else:
                    omit_crop = np.zeros((th, tw), dtype=omit_image.dtype)
                if sx2 > sx1 and sy2 > sy1:
                    omit_crop[sy1 - ty:sy2 - ty, sx1 - tx:sx2 - tx] = omit_image[sy1:sy2, sx1:sx2]
                tile.omit_crop_image = omit_crop.copy()
                focus_image_x = int(getattr(tile, 'aoi_image_x', -1))
                if focus_image_x < 0:
                    focus_image_x = int(getattr(tile, 'anomaly_peak_x', -1))
                context_focus_x = focus_image_x - tx if focus_image_x >= 0 else None

                is_dust, dust_mask, bright_ratio, detail_text = \
                    self._check_dust_or_scratch_feature_with_context(
                        omit_image,
                        tx,
                        ty,
                        tw,
                        th,
                        omit_crop,
                        focus_x=context_focus_x,
                        product_resolution=product_resolution,
                    )
                tile.dust_mask = dust_mask
                tile.dust_bright_ratio = bright_ratio

                if getattr(tile, 'is_aoi_coord_below_threshold', False):
                    tile.is_suspected_dust_or_scratch = False
                    detail_text += " TRACK_ONLY Score<THR -> AI_OK"
                    tile.dust_detail_text = detail_text

                    log_icon = "🟡"
                    zone_tag = f" [{tile.zone}]" if tile.zone else ""
                    aoi_suffix = self._format_aoi_tile_log_suffix(tile)
                    print(
                        f"{log_icon} {result.image_path.name} Tile@({tx},{ty}){zone_tag}{aoi_suffix} "
                        f"Score:{score:.3f} → {detail_text}"
                    )
                    continue

                if is_dust and anomaly_map is not None and dust_mask is not None:
                    top_pct = self.config.dust_heatmap_top_percent
                    metric_mode = self.config.dust_heatmap_metric
                    anomaly_map_for_dust, exclude_zone_masked = \
                        self._mask_aoi_exclude_zones_for_dust(tile, anomaly_map, model_id)
                    if exclude_zone_masked:
                        detail_text += " EXCLUDE_ZONE_HEATMAP_ZEROED"
                    aoi_seed_yx, aoi_seed_radius, aoi_seed_min_score = \
                        self._aoi_center_seed_for_tile(tile, anomaly_map_for_dust)
                    has_real, real_peak_yx, overall_iou, region_details, heatmap_binary, region_labels = \
                        self.check_dust_per_region(
                            dust_mask,
                            anomaly_map_for_dust,
                            top_percent=top_pct,
                            metric=metric_mode,
                            iou_threshold=self.config.dust_heatmap_iou_threshold,
                            force_include_yx=aoi_seed_yx,
                            force_include_radius=aoi_seed_radius,
                            force_include_min_score=aoi_seed_min_score,
                        )
                    tile.dust_heatmap_iou = overall_iou
                    tile.dust_region_details = region_details
                    tile.dust_heatmap_binary = heatmap_binary
                    if region_details:
                        tile.dust_region_max_cov = max(r["coverage"] for r in region_details)
                    dust_regions = [r for r in region_details if r["is_dust"]]
                    real_regions = [r for r in region_details if not r["is_dust"]]

                    _two_stage_ran = False
                    _ts_features = []
                    _ts_dust_mask_no_ext = None

                    if has_real:
                        tile.is_suspected_dust_or_scratch = False
                        detail_text += (
                            f" PER_REGION: {len(real_regions)}real+"
                            f"{len(dust_regions)}dust -> REAL_NG"
                        )
                        if real_peak_yx is not None:
                            amap_h, amap_w = anomaly_map_for_dust.shape[:2]
                            tile.anomaly_peak_y = tile.y + int(real_peak_yx[0] * tile.height / amap_h)
                            tile.anomaly_peak_x = tile.x + int(real_peak_yx[1] * tile.width / amap_w)
                    else:
                        if self.config.dust_two_stage_enabled:
                            dust_mask_no_ext = None
                            if omit_crop is not None:
                                _, dust_mask_no_ext, _, _ = \
                                    self._check_dust_or_scratch_feature_with_context(
                                        omit_image,
                                        tx,
                                        ty,
                                        tw,
                                        th,
                                        omit_crop,
                                        extension_override=0,
                                        focus_x=context_focus_x,
                                        product_resolution=product_resolution,
                                    )
                            ts_has_real, ts_peak_yx, ts_features, ts_detail = \
                                self.check_dust_two_stage(
                                    tile.image,
                                    anomaly_map_for_dust,
                                    dust_mask_no_ext if dust_mask_no_ext is not None else dust_mask,
                                    score,
                                    score_threshold=tile.score_threshold,
                                    candidate_dust_mask=dust_mask,
                                )
                            _two_stage_ran = True
                            _ts_features = ts_features
                            _ts_dust_mask_no_ext = dust_mask_no_ext
                            tile.dust_two_stage_features = ts_features
                            tile.dust_two_stage_dust_mask = dust_mask_no_ext if dust_mask_no_ext is not None else dust_mask

                            if ts_has_real:
                                tile.is_suspected_dust_or_scratch = False
                                detail_text += (
                                    f" PER_REGION: 0real+{len(dust_regions)}dust"
                                    f" -> {ts_detail}"
                                )
                                if ts_peak_yx is not None:
                                    amap_h, amap_w = anomaly_map_for_dust.shape[:2]
                                    tile.anomaly_peak_y = tile.y + int(ts_peak_yx[0] * tile.height / amap_h)
                                    tile.anomaly_peak_x = tile.x + int(ts_peak_yx[1] * tile.width / amap_w)
                            else:
                                tile.is_suspected_dust_or_scratch = True
                                detail_text += (
                                    f" PER_REGION: 0real+{len(dust_regions)}dust"
                                    f" -> {ts_detail}"
                                )
                        else:
                            tile.is_suspected_dust_or_scratch = True
                            detail_text += (
                                f" PER_REGION: 0real+{len(dust_regions)}dust -> DUST"
                            )

                    try:
                        if _two_stage_ran:
                            dm_for_debug = _ts_dust_mask_no_ext if _ts_dust_mask_no_ext is not None else dust_mask
                            tile.dust_iou_debug_image = self.generate_two_stage_debug_image(
                                tile.image,
                                anomaly_map_for_dust,
                                dm_for_debug,
                                _ts_features,
                                tile.is_suspected_dust_or_scratch,
                            )
                        else:
                            tile.dust_iou_debug_image = self.generate_dust_iou_debug_image(
                                tile.image, anomaly_map_for_dust, dust_mask,
                                heatmap_binary, overall_iou,
                                top_pct,
                                tile.is_suspected_dust_or_scratch,
                                region_details=region_details,
                                region_labels=region_labels,
                            )
                    except Exception as dbg_err:
                        print(f"⚠️ [v2] Debug image generation failed: {dbg_err}")
                elif is_dust:
                    tile.is_suspected_dust_or_scratch = True
                    detail_text += " (no heatmap, marked as dust)"
                else:
                    detail_text += " NO_DUST -> REAL_NG"
                tile.dust_detail_text = detail_text

                log_icon = "🧹" if tile.is_suspected_dust_or_scratch else "🔴"
                zone_tag = f" [{tile.zone}]" if tile.zone else ""
                aoi_suffix = self._format_aoi_tile_log_suffix(tile)
                print(
                    f"{log_icon} {result.image_path.name} Tile@({tx},{ty}){zone_tag}{aoi_suffix} "
                    f"Score:{score:.3f} → {detail_text}"
                )

        needs_dust = [r for r in results if r.anomaly_tiles]
        if not needs_dust:
            return
        max_workers = max(1, min(cpu_workers, len(needs_dust)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(contextvars.copy_context().run, _dust_check_one, r): r
                for r in needs_dust
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.warning("灰塵檢測失敗 (%s): %s", futures[future].image_path.name, e)

    def _apply_bomb_postprocess(
        self,
        results: List[ImageResult],
        bomb_info: Optional[Dict[str, Any]],
        product_resolution: Optional[Tuple[int, int]],
    ) -> None:
        """Mark anomaly tiles and edge defects that match configured/client bomb defects."""
        active_bombs = []
        if bomb_info is not None:
            defect_code = self._match_bomb_defect_code(bomb_info)
            active_bombs = [BombDefect(
                image_prefix=bomb_info["image_prefix"],
                defect_code=defect_code,
                defect_type=bomb_info["defect_type"],
                coordinates=bomb_info["coordinates"],
            )]
        elif self.config.bomb_defects:
            active_bombs = self.config.bomb_defects
        if not active_bombs:
            return

        tolerance = self.config.bomb_match_tolerance
        point_coord_count = sum(
            len(bomb.coordinates) for bomb in active_bombs if bomb.defect_type == "point"
        )
        line_def_count = sum(1 for bomb in active_bombs if bomb.defect_type == "line")
        source = "client" if bomb_info is not None else "config"
        print(
            f"💣 [v2] BOMB matching: source={source}, "
            f"tolerance={tolerance} product_px, "
            f"point_coords={point_coord_count}, line_defs={line_def_count}"
        )

        def nearest_point_distance(img_prefix: str, product_x: int, product_y: int):
            nearest = None
            for bomb in active_bombs:
                if bomb.defect_type != "point":
                    continue
                if not (img_prefix == bomb.image_prefix or
                        img_prefix.startswith(bomb.image_prefix + "_")):
                    continue
                for coord in bomb.coordinates:
                    dx = abs(product_x - coord[0])
                    dy = abs(product_y - coord[1])
                    dist2 = dx * dx + dy * dy
                    if nearest is None or dist2 < nearest[0]:
                        nearest = (dist2, coord, dx, dy)
            return nearest

        def image_to_product_coords(img_x: int, img_y: int, raw_bounds: Tuple[int, int, int, int]) -> Tuple[int, int]:
            product_w, product_h = product_resolution or DEFAULT_PRODUCT_RESOLUTION
            x_start, y_start, x_end, y_end = raw_bounds
            panel_w = max(1, x_end - x_start)
            panel_h = max(1, y_end - y_start)
            return (
                int(round((img_x - x_start) * product_w / panel_w)),
                int(round((img_y - y_start) * product_h / panel_h)),
            )

        for result in results:
            if result.anomaly_tiles and result.raw_bounds is not None:
                img_prefix = result.image_path.stem
                for tile, _score, anomaly_map in result.anomaly_tiles:
                    if getattr(tile, "is_aoi_coord_below_threshold", False):
                        continue
                    if anomaly_map is not None and anomaly_map.size > 0:
                        try:
                            amap_h, amap_w = anomaly_map.shape[:2]
                            peak_local = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
                            tile.anomaly_peak_y = tile.y + int(peak_local[0] * tile.height / amap_h)
                            tile.anomaly_peak_x = tile.x + int(peak_local[1] * tile.width / amap_w)
                        except Exception:
                            tile.anomaly_peak_x, tile.anomaly_peak_y = tile.center
                    else:
                        tile.anomaly_peak_x, tile.anomaly_peak_y = tile.center

                    is_bomb, bomb_code = self.check_bomb_match(
                        img_prefix,
                        tile.anomaly_peak_x,
                        tile.anomaly_peak_y,
                        result.raw_bounds,
                        anomaly_map=anomaly_map,
                        product_resolution=product_resolution,
                        bomb_list=active_bombs,
                    )
                    # AOI coord tile fallback: 若 heatmap peak 偏到 tile 邊緣超出 tolerance，
                    # 改用 tile.center 重試 — AOI tile 的中心就是機檢座標，可信度更高
                    if not is_bomb and tile.is_aoi_coord_tile:
                        tile_cx, tile_cy = tile.center
                        is_bomb, bomb_code = self.check_bomb_match(
                            img_prefix, tile_cx, tile_cy, result.raw_bounds,
                            anomaly_map=anomaly_map,
                            product_resolution=product_resolution,
                            bomb_list=active_bombs,
                        )
                    # AOI coord tile 保護: peak 可能被鄰近炸彈亮點吸引而誤判，
                    # 需驗證原始 AOI 產品座標本身也在炸彈容忍範圍內
                    if is_bomb and tile.is_aoi_coord_tile and tile.aoi_product_x >= 0:
                        aoi_matches_bomb = False
                        tolerance = self.config.bomb_match_tolerance
                        for bomb in active_bombs:
                            if not (img_prefix == bomb.image_prefix or
                                    img_prefix.startswith(bomb.image_prefix + "_")):
                                continue
                            if bomb.defect_type == "point":
                                for coord in bomb.coordinates:
                                    if (abs(tile.aoi_product_x - coord[0]) <= tolerance and
                                        abs(tile.aoi_product_y - coord[1]) <= tolerance):
                                        aoi_matches_bomb = True
                                        break
                            elif bomb.defect_type == "line" and len(bomb.coordinates) >= 2:
                                pt1, pt2 = bomb.coordinates[0], bomb.coordinates[1]
                                if self._point_within_line_segment_tolerance(
                                    tile.aoi_product_x,
                                    tile.aoi_product_y,
                                    pt1,
                                    pt2,
                                    tolerance,
                                ):
                                    aoi_matches_bomb = True
                            if aoi_matches_bomb:
                                break
                        if not aoi_matches_bomb:
                            is_bomb = False
                            aoi_suffix = self._format_aoi_tile_log_suffix(tile)
                            print(
                                f"🛡️ [v2] {result.image_path.name} Tile@({tile.x},{tile.y}){aoi_suffix} "
                                f"Peak 匹配炸彈但 AOI 座標 ({tile.aoi_product_x},{tile.aoi_product_y}) "
                                f"距離炸彈超過容忍度，保留為真實缺陷"
                            )
                    if is_bomb:
                        tile.is_bomb = True
                        tile.bomb_defect_code = bomb_code
                    if point_coord_count:
                        if tile.is_aoi_coord_tile and tile.aoi_product_x >= 0:
                            product_x = tile.aoi_product_x
                            product_y = tile.aoi_product_y
                            coord_source = "aoi"
                        else:
                            product_x, product_y = image_to_product_coords(
                                tile.anomaly_peak_x, tile.anomaly_peak_y, result.raw_bounds
                            )
                            coord_source = "peak"
                        nearest = nearest_point_distance(img_prefix, product_x, product_y)
                        if nearest is not None:
                            dist2, coord, dx, dy = nearest
                            aoi_suffix = self._format_aoi_tile_log_suffix(tile)
                            print(
                                f"💣 [v2] BOMB distance {result.image_path.name} "
                                f"Tile@({tile.x},{tile.y}){aoi_suffix} "
                                f"{coord_source}=({product_x},{product_y}) nearest={coord} "
                                f"dx={dx} dy={dy} dist={dist2 ** 0.5:.1f} "
                                f"tol={tolerance} matched={tile.is_bomb}"
                            )

            if getattr(result, "edge_defects", None) and result.raw_bounds is not None:
                img_prefix = result.image_path.stem
                for ed in result.edge_defects:
                    if getattr(ed, "is_cv_ok", False):
                        continue
                    cx, cy = ed.center
                    is_bomb, bomb_code = self.check_bomb_match(
                        img_prefix, cx, cy, result.raw_bounds,
                        product_resolution=product_resolution,
                        bomb_list=active_bombs,
                    )
                    if is_bomb:
                        ed.is_bomb = True
                        ed.bomb_defect_code = bomb_code
                    if point_coord_count:
                        product_x, product_y = image_to_product_coords(int(cx), int(cy), result.raw_bounds)
                        nearest = nearest_point_distance(img_prefix, product_x, product_y)
                        if nearest is not None:
                            dist2, coord, dx, dy = nearest
                            print(
                                f"💣 [v2] BOMB distance {result.image_path.name} "
                                f"Edge@({int(cx)},{int(cy)}) product=({product_x},{product_y}) "
                                f"nearest={coord} dx={dx} dy={dy} dist={dist2 ** 0.5:.1f} "
                                f"tol={tolerance} matched={ed.is_bomb}"
                            )

        # === Bomb line 共識機制 ===
        # 若同一條 bomb line 已有 ≥3 個 tile 通過形態驗證，
        # 則位置匹配但形態不通過的 tile 也視為 bomb（線已被確認存在）
        for bomb in active_bombs:
            if bomb.defect_type != "line":
                continue
            for result in results:
                if not result.anomaly_tiles or result.raw_bounds is None:
                    continue
                img_prefix = result.image_path.stem
                if not (img_prefix == bomb.image_prefix or
                        img_prefix.startswith(bomb.image_prefix + "_")):
                    continue
                confirmed = sum(
                    1 for t, _, _ in result.anomaly_tiles
                    if t.is_bomb and t.bomb_defect_code == bomb.defect_code
                )
                if confirmed < 3:
                    continue
                for tile, _score, anomaly_map in result.anomaly_tiles:
                    if getattr(tile, "is_aoi_coord_below_threshold", False):
                        continue
                    if tile.is_bomb:
                        continue
                    peak_x = getattr(tile, "anomaly_peak_x", tile.center[0])
                    peak_y = getattr(tile, "anomaly_peak_y", tile.center[1])
                    is_bomb, bomb_code = self.check_bomb_match(
                        img_prefix, peak_x, peak_y, result.raw_bounds,
                        anomaly_map=anomaly_map, product_resolution=product_resolution,
                        bomb_list=[bomb], skip_shape_check=True,
                    )
                    if not is_bomb and tile.is_aoi_coord_tile:
                        tile_cx, tile_cy = tile.center
                        is_bomb, bomb_code = self.check_bomb_match(
                            img_prefix, tile_cx, tile_cy, result.raw_bounds,
                            anomaly_map=anomaly_map, product_resolution=product_resolution,
                            bomb_list=[bomb], skip_shape_check=True,
                        )
                    if is_bomb:
                        tile.is_bomb = True
                        tile.bomb_defect_code = bomb_code

        # Per-image bomb 匹配摘要 log（對齊 v1 line 5236）
        from collections import Counter
        for result in results:
            edge_bombs = [
                ed for ed in getattr(result, "edge_defects", []) or []
                if getattr(ed, "is_bomb", False)
            ]
            summary = Counter(
                t.bomb_defect_code for t, _, _ in result.anomaly_tiles if t.is_bomb
            )
            summary.update(ed.bomb_defect_code for ed in edge_bombs)
            if summary:
                parts = [f"{c}×{n}" for c, n in summary.items()]
                suffix = f" (含 edge {len(edge_bombs)})" if edge_bombs else ""
                print(f"💣 {result.image_path.name} BOMB match: {', '.join(parts)}{suffix}")

    def _apply_exclude_zone_postprocess(
        self,
        results: List[ImageResult],
        model_id: Optional[str],
    ) -> None:
        """Mark PatchCore anomaly tiles that fall inside enabled CV exclude zones."""
        if not getattr(self, "edge_inspector", None):
            return
        try:
            resolution_code = ""
            if model_id and len(model_id) >= 6:
                resolution_code = model_id[5].upper()
            self.edge_inspector.config.set_active_zones_for_product(resolution_code)
            active_zones = [z for z in self.edge_inspector.config.exclude_zones if z.enabled]
            if not active_zones:
                return

            def _zone_contains(zone, x: int, y: int) -> bool:
                return zone.x <= x <= zone.x + zone.w and zone.y <= y <= zone.y + zone.h

            for result in results:
                for tile, _score, _anomaly_map in result.anomaly_tiles:
                    if tile.is_bright_spot_detection:
                        continue
                    px = tile.anomaly_peak_x if tile.anomaly_peak_x >= 0 else tile.center[0]
                    py = tile.anomaly_peak_y if tile.anomaly_peak_y >= 0 else tile.center[1]

                    if (
                        getattr(tile, "is_aoi_coord_tile", False)
                        and getattr(tile, "aoi_image_x", -1) >= 0
                        and getattr(tile, "aoi_image_y", -1) >= 0
                    ):
                        ax = int(tile.aoi_image_x)
                        ay = int(tile.aoi_image_y)
                        aoi_zone = next((z for z in active_zones if _zone_contains(z, ax, ay)), None)
                        peak_zone = next((z for z in active_zones if _zone_contains(z, px, py)), None)

                        if aoi_zone is None:
                            if peak_zone is not None:
                                zone_tag = f" [{tile.zone}]" if tile.zone else ""
                                aoi_suffix = self._format_aoi_tile_log_suffix(tile)
                                print(
                                    f"🔴 {result.image_path.name} Tile@({tile.x},{tile.y}){zone_tag}{aoi_suffix} "
                                    f"AOI@({ax},{ay}) 不在不檢測區，peak@({px},{py}) 落在不檢測區 "
                                    f"({peak_zone.x},{peak_zone.y},{peak_zone.w}x{peak_zone.h})，保留 NG"
                                )
                            continue

                        tile.is_in_exclude_zone = True
                        zone_tag = f" [{tile.zone}]" if tile.zone else ""
                        aoi_suffix = self._format_aoi_tile_log_suffix(tile)
                        print(
                            f"🚫 {result.image_path.name} Tile@({tile.x},{tile.y}){zone_tag}{aoi_suffix} "
                            f"AOI@({ax},{ay}) 落在不檢測區 "
                            f"({aoi_zone.x},{aoi_zone.y},{aoi_zone.w}x{aoi_zone.h})"
                        )
                        continue

                    for zone in active_zones:
                        if _zone_contains(zone, px, py):
                            tile.is_in_exclude_zone = True
                            zone_tag = f" [{tile.zone}]" if tile.zone else ""
                            aoi_suffix = self._format_aoi_tile_log_suffix(tile)
                            print(
                                f"🚫 {result.image_path.name} Tile@({tile.x},{tile.y}){zone_tag}{aoi_suffix} "
                                f"peak@({px},{py}) 落在不檢測區 ({zone.x},{zone.y},{zone.w}x{zone.h})"
                            )
                            break
        except Exception as e:
            logger.error(f"排除區域檢查失敗: {e}", exc_info=True)

    def _apply_scratch_postprocess(self, results: List[ImageResult]) -> None:
        """Apply the optional DINOv2 scratch classifier post-filter."""
        sf = self._get_scratch_filter()
        if sf is None:
            return
        panel_tiles = panel_filtered = 0
        panel_ms = 0.0
        for result in results:
            if result.anomaly_tiles:
                sf.apply_to_image_result(result)
                panel_tiles += len(result.anomaly_tiles)
                panel_filtered += getattr(result, "scratch_filter_count", 0)
                panel_ms += getattr(result, "scratch_elapsed_ms", 0.0)
        if panel_tiles:
            logger.info(
                "[scratch] Panel 總計: 檢查 %d tiles, filtered=%d, TT=%.1fms",
                panel_tiles, panel_filtered, panel_ms,
            )

    def _process_panel_v2(
        self,
        panel_dir: Path,
        progress_callback=None,
        cpu_workers: int = 4,
        product_resolution: Optional[Tuple[int, int]] = None,
        bomb_info: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
        machine_no: Optional[str] = None,
        aoi_report_override: Optional[Dict[str, List['AOIReportDefect']]] = None,
        machine_judgment: Optional[str] = None,
    ):
        """新架構：依 tile zone routing inner/edge model。

        最小化實作 (Phase 9.3)：
          1. capi_preprocess.preprocess_panel_folder 切 tile + inner/edge 分流
          2. 對每張 tile 呼叫對應 model（inner.pt / edge.pt）
          3. 回傳與 v1 相同的 tuple：
               (results, omit_vis, omit_overexposed, omit_overexposure_info,
                is_duplicate, omit_image, aoi_report)
             其中 results 為 List[ImageResult]

        透過既有 predict_tile/postprocess helper 保持 v1 相容：
        edge margin、dust filter、bomb check、scratch classifier。
        新架構 edge.pt 已專責 edge zone，故不再呼叫傳統 CV 邊緣檢測。
        """
        import time
        from capi_preprocess import preprocess_panel_folder, PreprocessConfig

        panel_path = Path(panel_dir)
        t0 = time.time()
        image_files, is_duplicate = self._prepare_panel_image_files(panel_path)
        panel_mark_detection, panel_mark_regions = self._detect_panel_mark_binary_region(
            image_files,
            machine_no=machine_no,
            model_id=model_id,
        )

        aoi_report: Optional[Dict[str, List['AOIReportDefect']]] = None
        aoi_report_for_inference: Dict[str, List['AOIReportDefect']] = {}
        if self.config.aoi_coord_inspection_enabled:
            aoi_report = (
                aoi_report_override
                if aoi_report_override is not None
                else self._parse_aoi_report_txt(Path(panel_dir))
            )
            aoi_report_for_inference, _forced_bomb_count = self._aoi_report_with_forced_client_bomb_coords(
                aoi_report,
                bomb_info,
            )
        elif bomb_info is not None and getattr(self.config, "bomb_area_force_detection_enabled", False):
            print("💣 [BOMB_FORCE] 已啟用但 aoi_coord_inspection_enabled=False，無法補切 Client 炸彈座標")

        aoi_ok_skip = bool(
            str(machine_judgment or "").strip().upper() == "OK"
            and not self.config.grid_tiling_enabled
            and self.config.aoi_coord_inspection_enabled
            and not aoi_report_for_inference
        )
        if aoi_ok_skip:
            mark_source_name = str(
                (panel_mark_detection or {}).get("source_image") or ""
            )
            mark_source_path = next(
                (path for path in image_files if path.name == mark_source_name),
                None,
            )
            if mark_source_path is None:
                mark_source_path = next(
                    (path for path in image_files if self._is_mark_binary_source(path.name)),
                    image_files[0] if image_files else None,
                )

            mark_results: List[ImageResult] = []
            if mark_source_path is not None:
                image_size = (panel_mark_detection or {}).get("image_size")
                try:
                    image_width = int(image_size[0])
                    image_height = int(image_size[1])
                except (TypeError, ValueError, IndexError):
                    mark_image = self._read_detection_image(mark_source_path)
                    if mark_image is None:
                        image_width = image_height = 0
                    else:
                        image_height, image_width = mark_image.shape[:2]

                full_bounds = (0, 0, image_width, image_height)
                mark_result = ImageResult(
                    image_path=mark_source_path,
                    image_size=(image_width, image_height),
                    otsu_bounds=full_bounds,
                    exclusion_regions=[],
                    tiles=[],
                    excluded_tile_count=0,
                    processed_tile_count=0,
                    processing_time=time.time() - t0,
                    anomaly_tiles=[],
                    raw_bounds=full_bounds,
                    inference_time=0.0,
                )
                if bomb_info is not None:
                    mark_result.client_bomb_info = bomb_info
                mark_results.append(mark_result)

            self._attach_panel_mark_binary_to_results(
                mark_results,
                panel_mark_detection,
                panel_mark_regions,
            )
            total_elapsed = time.time() - t0
            print(
                "[v2] AOI_OK_SKIP: machine_judgment=OK、Grid Tiling 關閉且無 "
                "AOI/forced 座標；保留 MARK，跳過 OMIT、影像預處理、模型與後處理"
            )
            print(
                f"📊 Panel {panel_path.name} 總耗時 {total_elapsed:.2f}s | "
                f"前置={total_elapsed:.2f}s, 預處理=0.00s, Tile準備=0.00s, "
                f"GPU推論=0.00s, 後處理/收尾=0.00s | "
                f"{len(mark_results)} MARK result(s), 0 NG, AOI_OK_SKIP"
            )
            return (
                mark_results,
                None,
                False,
                "",
                is_duplicate,
                None,
                aoi_report or {},
            )

        omit_vis, omit_overexposed, omit_overexposure_info, omit_image = \
            self._load_omit_context(panel_path, image_files=image_files)
        if omit_image is not None:
            tag = "OVEREXPOSED" if omit_overexposed else "OK"
            print(f"[v2] OMIT {tag}: {omit_overexposure_info}")

        aoi_only_mode = bool(
            not self.config.grid_tiling_enabled
            and self.config.aoi_coord_inspection_enabled
            and aoi_report_for_inference
        )
        if (
            not self.config.grid_tiling_enabled
            and self.config.aoi_coord_inspection_enabled
            and not aoi_report_for_inference
        ):
            print(
                "[v2] Grid Tiling 關閉，AOI report 無 NG/forced 座標："
                "本次不建立任何 inference tile"
            )
        preprocess_image_files = image_files
        if aoi_only_mode:
            report_prefixes = set(aoi_report_for_inference.keys())
            preprocess_prefixes = {
                prefix for prefix in report_prefixes
                if not prefix.upper().startswith("B0")
            }
            has_black_image = any(p.upper().startswith("B0") for p in report_prefixes)
            if has_black_image:
                from capi_preprocess import BOUNDARY_REFERENCE_PRIORITY
                folder_prefixes = {self._get_image_prefix(f.name) for f in image_files}
                for cand in BOUNDARY_REFERENCE_PRIORITY:
                    if cand in folder_prefixes:
                        preprocess_prefixes.add(cand)
                        break

            preprocess_image_files = [
                f for f in image_files
                if self._get_image_prefix(f.name) in preprocess_prefixes
            ]
            skipped = len(image_files) - len(preprocess_image_files)
            print(
                f"[v2] AOI-only: report prefixes="
                f"{','.join(sorted(report_prefixes)) or '-'}; "
                f"Phase 1 僅處理 {len(preprocess_image_files)} 張 AOI 相關 lighting"
                + (f"，跳過 {skipped} 張非 AOI lighting" if skipped > 0 else "")
            )

        pre_cfg = PreprocessConfig(
            tile_size=self.config.tile_size,
            tile_stride=getattr(self.config, "tile_stride", self.config.tile_size),
            otsu_offset=self.config.otsu_offset,
            enable_panel_polygon=self.config.enable_panel_polygon,
            edge_threshold_px=self.config.edge_threshold_px,
            image_preprocess_pipeline=getattr(self.config, "image_preprocess_pipeline", []),
            image_preprocess_pipelines=getattr(self.config, "image_preprocess_pipelines", {}),
            cache_processed_image=aoi_only_mode,
            generate_grid_tiles=bool(self.config.grid_tiling_enabled),
            preprocess_after_tiling=getattr(self.config, "preprocess_after_tiling", False),
            product_resolution=product_resolution or self._product_resolution(),
            rotate_180=getattr(self, "_rotate_detection_images_180", False),
        )

        preprocess_start = time.time()
        setup_elapsed = preprocess_start - t0
        panel_results = preprocess_panel_folder(
            panel_path,
            pre_cfg,
            image_files=preprocess_image_files,
            boundary_reference_files=image_files if aoi_only_mode else None,
        )
        if not panel_results and not aoi_only_mode:
            logger.warning(f"[v2] {panel_path}: preprocess_panel_folder 回傳空結果")
            # 回傳與 v1 格式相容的空結果
            return [], omit_vis, omit_overexposed, omit_overexposure_info, is_duplicate, omit_image, {}
        preprocess_end = time.time()
        preprocess_elapsed = preprocess_end - preprocess_start
        grid_tile_count = sum(len(result.tiles) for result in panel_results.values())
        print(
            f"⚡ Phase 1 完成: {len(panel_results)} 個 lighting 預處理耗時 "
            f"{preprocess_elapsed:.2f}s (grid tiles={grid_tile_count})"
        )

        results: List[ImageResult] = []
        v2_entries: List[Dict[str, Any]] = []

        for lighting, pre_result in panel_results.items():
            img_path = pre_result.image_path
            bbox = pre_result.foreground_bbox  # (x1, y1, x2, y2)
            polygon = pre_result.panel_polygon

            # 取得圖片尺寸
            raw_img = self._read_detection_image(img_path)
            if raw_img is None:
                logger.warning(f"[v2] 無法讀取圖片: {img_path}")
                continue
            img_h, img_w = raw_img.shape[:2]

            # raw_bounds 必須是「物件原始邊界」（不含 otsu_offset 內推），用於
            # AOI 機檢座標 ↔ 圖片座標映射；對齊 v1 / DEBUG 路徑的 _find_raw_object_bounds
            # 語意。pre_result.foreground_bbox 已套 +/-offset，作為 otsu_bounds 用。
            raw_bounds_unoffset, _ = self._find_raw_object_bounds(raw_img)
            if raw_bounds_unoffset is None:
                raw_bounds_unoffset = bbox

            # 取得此 lighting 對應的 inner / edge model 路徑
            lighting_map = self.config.model_mapping.get(lighting, {})
            if not isinstance(lighting_map, dict):
                lighting_map = {}

            lighting_thr = self.config.threshold_mapping.get(lighting, {})
            if isinstance(lighting_thr, dict):
                inner_thr = float(lighting_thr.get("inner", 0.75))
                edge_thr = float(lighting_thr.get("edge", 0.75))
            else:
                # fallback：舊式 flat float
                inner_thr = edge_thr = float(lighting_thr) if lighting_thr else 0.75

            # 把 capi_preprocess.TileResult 轉換成 capi_inference.TileInfo
            tile_infos: List[TileInfo] = []
            ts = self.config.tile_size
            bottom_y_threshold = bbox[3] - ts
            right_x_threshold = bbox[2] - ts
            zone_by_tile_id: Dict[int, str] = {}
            for tr in pre_result.tiles:
                ti = TileInfo(
                    tile_id=tr.tile_id,
                    x=tr.x1,
                    y=tr.y1,
                    width=ts,
                    height=ts,
                    image=tr.image,
                    original_image=tr.original_image,
                    mask=tr.mask,
                    is_bottom_edge=tr.y1 >= bottom_y_threshold,
                    is_top_edge=tr.y1 <= bbox[1],
                    is_left_edge=tr.x1 <= bbox[0],
                    is_right_edge=tr.x1 >= right_x_threshold,
                    zone=tr.zone,
                )
                tile_infos.append(ti)
                zone_by_tile_id[ti.tile_id] = tr.zone

            # 如果此 lighting 有任何需要推論的 tiles，才驗證 model_mapping 必須存在
            if len(tile_infos) > 0:
                if "inner" not in lighting_map or "edge" not in lighting_map:
                    raise RuntimeError(f"[v2] {lighting}: model_mapping 必須同時包含 inner/edge")

            image_result = ImageResult(
                image_path=img_path,
                image_size=(img_w, img_h),
                otsu_bounds=bbox,
                exclusion_regions=[],
                tiles=tile_infos,
                excluded_tile_count=0,
                processed_tile_count=len(tile_infos),
                processing_time=time.time() - t0,
                anomaly_tiles=[],
                raw_bounds=raw_bounds_unoffset,
                panel_polygon=polygon,
                processed_image=getattr(pre_result, "processed_image", None),
                inference_time=0.0,
                preprocess_steps=list(getattr(pre_result, "preprocess_steps", []) or []),
                preprocess_total_ms=float(getattr(pre_result, "preprocess_total_ms", 0.0) or 0.0),
            )

            if bomb_info is not None:
                image_result.client_bomb_info = bomb_info
            # 新架構不再跑傳統 CV 邊緣檢測（edge.pt 已專責 edge zone）

            results.append(image_result)
            v2_entries.append({
                "lighting": lighting,
                "result": image_result,
                "zone_by_tile_id": zone_by_tile_id,
                "inner_thr": inner_thr,
                "edge_thr": edge_thr,
                "inner_path": str(lighting_map.get("inner", "")),
                "edge_path": str(lighting_map.get("edge", "")),
            })

        # AOI 機檢座標 attribution（新架構：找包含座標的既存 grid tile 並標屬性，
        # 不再切新 tile，也不再對 edge defect 跑 PC ROI 推論）
        # log 由 helper 內部負責印（disabled / empty / attribution 三種情境統一處理）
        self._apply_aoi_coord_inspection(
            panel_dir=Path(panel_dir),
            preprocessed_results=results,
            omit_image=omit_image,
            omit_overexposed=omit_overexposed,
            product_resolution=product_resolution,
            aoi_report=aoi_report_for_inference,
        )

        # v1 相容：B0F00000 等 skip_files 沒有 PatchCore 模型，不屬於 5-lighting
        # grid；若 AOI report 指到這些黑圖，仍要建立 AOI-centered tiles，後續走
        # _detect_bright_spots() 二值化亮點偵測。
        if aoi_report_for_inference:
            existing_prefixes = {
                self._get_image_prefix(result.image_path.name)
                for result in results
            }
            ref_result = next((r for r in results if r.raw_bounds), None)
            ref_bounds = ref_result.raw_bounds if ref_result else None
            ref_polygon = ref_result.panel_polygon if ref_result else None

            for report_prefix, defects in aoi_report_for_inference.items():
                if report_prefix in existing_prefixes:
                    continue

                # 只對 B 開頭（B = Black）的黑畫面走 bright_spot 補丁邏輯。
                # 新架構 machine_config.yaml 預設沒有 skip_files 欄位，因此這裡
                # 不依賴 should_skip_file，直接以 prefix 判斷；OMIT/PINIGBI/側拍
                # 開頭都不是 B，不會誤匹配。
                if not report_prefix.upper().startswith("B0"):
                    continue

                matched_file = next(
                    (
                        f for f in image_files
                        if self._get_image_prefix(f.name) == report_prefix
                    ),
                    None,
                )
                if matched_file is None:
                    continue

                black_image = self._read_detection_image(matched_file)
                if black_image is None:
                    logger.warning(f"[v2] 無法讀取 skip_file 圖片: {matched_file}")
                    continue

                img_h, img_w = black_image.shape[:2]
                bounds = ref_bounds or (0, 0, img_w, img_h)
                skip_result = ImageResult(
                    image_path=matched_file,
                    image_size=(img_w, img_h),
                    otsu_bounds=bounds,
                    exclusion_regions=[],
                    tiles=[],
                    excluded_tile_count=0,
                    processed_tile_count=0,
                    processing_time=time.time() - t0,
                    anomaly_tiles=[],
                    raw_bounds=bounds,
                    panel_polygon=ref_polygon.copy() if ref_polygon is not None else None,
                    inference_time=0.0,
                )

                created = self._create_aoi_centered_tiles_v2(
                    image=black_image,
                    result=skip_result,
                    defects=defects,
                    product_resolution=product_resolution,
                    pre_cfg=pre_cfg,
                    is_skip_file=True,
                )
                skip_result.processed_tile_count = len(skip_result.tiles)
                if bomb_info is not None:
                    skip_result.client_bomb_info = bomb_info

                results.append(skip_result)
                v2_entries.append({
                    "lighting": report_prefix,
                    "result": skip_result,
                    "zone_by_tile_id": {t.tile_id: "bright_spot" for t in skip_result.tiles},
                    "inner_thr": 1.0,
                    "edge_thr": 1.0,
                    "skip_file": True,
                })
                existing_prefixes.add(report_prefix)
                print(
                    f"[v2] skip_file AOI: {matched_file.name} "
                    f"建立 {created} 個 bright-spot tiles"
                )

        self._attach_panel_mark_binary_to_results(
            results,
            panel_mark_detection,
            panel_mark_regions,
        )

        # 同步 zone_by_tile_id 與 result.tiles —— 把 _apply_aoi_coord_inspection
        # 跟 skip_file 路徑新增的 AOI centered tiles 的 zone 補進來，
        # 避免 inference loop 落到 fallback "inner"。
        for entry in v2_entries:
            entry["zone_by_tile_id"] = {
                ti.tile_id: (ti.zone or "inner")
                for ti in entry["result"].tiles
            }

        # grid_tiling_enabled=False 代表 AOI 座標目標模式：只推論已標記的 AOI tiles。
        # v2 的 AOI attribution 必須在模型推論前完成，否則會退化成整片 grid 推論。
        if not self.config.grid_tiling_enabled:
            for entry in v2_entries:
                result = entry["result"]
                original_count = len(result.tiles)
                result.tiles = [t for t in result.tiles if t.is_aoi_coord_tile]
                result.processed_tile_count = len(result.tiles)
                removed = original_count - len(result.tiles)
                if removed > 0:
                    print(
                        f"[v2] Grid Tiling 關閉: {result.image_path.name} "
                        f"移除 {removed} 個 grid tiles，保留 {len(result.tiles)} 個 AOI tiles"
                    )

        # 推論：只對目前 result.tiles 內保留的 tiles 跑模型。
        inference_start = time.time()
        tile_prepare_elapsed = inference_start - preprocess_end
        for entry in v2_entries:
            lighting = entry["lighting"]
            result = entry["result"]
            zone_by_tile_id = entry["zone_by_tile_id"]
            inner_thr = entry["inner_thr"]
            edge_thr = entry["edge_thr"]

            t_infer_start = time.time()
            anomaly_tiles: List[Tuple[TileInfo, float, Optional[np.ndarray]]] = []

            if entry.get("skip_file") or self.config.should_skip_file(result.image_path.name):
                print(f"💡 {result.image_path.name} (skip_file) → 使用二值化偵測亮點")
                for ti in result.tiles:
                    score, anomaly_map = self._detect_bright_spots(ti)
                    ti.score_threshold = 0.5
                    if score <= 0:
                        ti.is_aoi_coord_below_threshold = True
                    anomaly_tiles.append((ti, score, anomaly_map))

                infer_elapsed = time.time() - t_infer_start
                result.anomaly_tiles = anomaly_tiles
                result.inference_time = infer_elapsed
                result.processing_time = time.time() - t0
                bright_ng_count = sum(1 for _t, score, _m in anomaly_tiles if score > 0)
                print(
                    f"[v2] {lighting}: bright_spot tiles={len(result.tiles)}, "
                    f"NG={bright_ng_count}, infer {infer_elapsed:.2f}s"
                )
                continue

            inner_path = entry.get("inner_path", "")
            edge_path = entry.get("edge_path", "")
            print(
                f"🎯 {result.image_path.name} → "
                f"inner: {inner_path or '?'} (thr={inner_thr:.3f}), "
                f"edge: {edge_path or '?'} (thr={edge_thr:.3f})"
            )

            for ti in result.tiles:
                zone = zone_by_tile_id.get(ti.tile_id, "inner")
                threshold = inner_thr if zone == "inner" else edge_thr
                ti.score_threshold = threshold
                try:
                    model = self._get_model_for(self.config.machine_id, lighting, zone)
                    score, anomaly_map = self.predict_tile(
                        ti,
                        inferencer=model,
                        threshold=threshold,
                        model_id=model_id or self.config.machine_id,
                    )
                    if ti.is_aoi_coord_tile and score <= 1e-9:
                        print(
                            f"    [v2 score_diag] {lighting}/{zone} Tile@({ti.x},{ti.y}) "
                            f"raw={ti.raw_pred_score:.4f} preMax={ti.pre_decay_map_max:.4f} "
                            f"postMax={ti.post_decay_map_max:.4f} decay={ti.score_decay_ratio:.4f} "
                            f"mask={ti.score_mask_valid_ratio:.3f} edge={ti.score_edge_margin_sides or '-'}"
                        )
                except Exception as exc:
                    raise RuntimeError(
                        f"[v2] {lighting}/{zone} tile({ti.x},{ti.y}) 推論失敗: {exc}"
                    ) from exc

                if score >= threshold:
                    # 計算熱力圖峰值座標（圖片絕對座標）
                    if anomaly_map is not None:
                        peak_idx = int(np.argmax(anomaly_map))
                        ah, aw = anomaly_map.shape[:2]
                        peak_y_rel = peak_idx // aw
                        peak_x_rel = peak_idx % aw
                        ti.anomaly_peak_x = ti.x + peak_x_rel
                        ti.anomaly_peak_y = ti.y + peak_y_rel
                    anomaly_tiles.append((ti, score, anomaly_map))
                elif ti.is_aoi_coord_tile:
                    # AOI 座標 tile 即使低於閾值也保留，供紀錄頁追蹤查看。
                    ti.is_aoi_coord_below_threshold = True
                    anomaly_tiles.append((ti, score, anomaly_map))

            infer_elapsed = time.time() - t_infer_start
            result.anomaly_tiles = anomaly_tiles
            result.inference_time = infer_elapsed
            result.processing_time = time.time() - t0

            active_zones = [zone_by_tile_id.get(t.tile_id, "inner") for t in result.tiles]
            inner_count = sum(1 for z in active_zones if z == "inner")
            edge_count = sum(1 for z in active_zones if z == "edge")
            model_ng_count = sum(
                1 for t, _score, _map in anomaly_tiles
                if not getattr(t, "is_aoi_coord_below_threshold", False)
            )
            track_count = len(anomaly_tiles) - model_ng_count
            track_suffix = f", track={track_count}" if track_count else ""
            print(
                f"[v2] {lighting}: tiles={len(result.tiles)} "
                f"(inner={inner_count}, edge={edge_count}), "
                f"NG={model_ng_count}{track_suffix}, infer {infer_elapsed:.2f}s"
            )

        inference_elapsed = time.time() - inference_start
        print(f"🔥 Phase 2 完成: GPU 推論耗時 {inference_elapsed:.2f}s")

        post_start = time.time()
        self._apply_omit_dust_postprocess(
            results,
            omit_image=omit_image,
            omit_overexposed=omit_overexposed,
            omit_overexposure_info=omit_overexposure_info,
            cpu_workers=cpu_workers,
            model_id=model_id,
            product_resolution=product_resolution,
        )
        self._apply_bomb_postprocess(results, bomb_info, product_resolution)
        self._apply_exclude_zone_postprocess(results, model_id)
        self._apply_scratch_postprocess(results)
        post_elapsed = time.time() - post_start
        print(f"🧹 Phase 3 完成: 後處理耗時 {post_elapsed:.2f}s")

        if bomb_info is not None:
            for result in results:
                result.client_bomb_info = bomb_info

        def _has_effective_ng(result: ImageResult) -> bool:
            real_tiles = [
                t for t, _s, _m in result.anomaly_tiles
                if not getattr(t, "is_aoi_coord_below_threshold", False)
                and not t.is_suspected_dust_or_scratch
                and not t.is_bomb
                and not t.is_in_exclude_zone
                and not t.scratch_filtered
            ]
            real_edges = [
                ed for ed in getattr(result, "edge_defects", []) or []
                if not getattr(ed, "is_suspected_dust_or_scratch", False)
                and not getattr(ed, "is_bomb", False)
                and not getattr(ed, "is_cv_ok", False)
            ]
            return bool(real_tiles or real_edges)

        ng_count = sum(1 for r in results if _has_effective_ng(r))
        total_elapsed = time.time() - t0
        post_finalize_elapsed = max(
            0.0,
            total_elapsed
            - setup_elapsed
            - preprocess_elapsed
            - tile_prepare_elapsed
            - inference_elapsed,
        )
        print(
            f"📊 Panel {Path(panel_dir).name} 總耗時 {total_elapsed:.2f}s | "
            f"前置={setup_elapsed:.2f}s, "
            f"預處理={preprocess_elapsed:.2f}s, "
            f"Tile準備={tile_prepare_elapsed:.2f}s, "
            f"GPU推論={inference_elapsed:.2f}s, "
            f"後處理/收尾={post_finalize_elapsed:.2f}s | "
            f"{len(results)} lighting(s), {ng_count} NG"
        )

        # 回傳與 v1 格式相容的 7-tuple
        return results, omit_vis, omit_overexposed, omit_overexposure_info, is_duplicate, omit_image, aoi_report or {}

    def _get_model_for(self, machine_id: str, lighting: str, zone: str):
        """新架構 lazy loading，cache key 含 zone。

        zone: "inner" | "edge"
        """
        key = (machine_id, lighting, zone)
        if key not in self._model_cache_v2:
            path_raw = self.config.model_mapping[lighting][zone]
            model_path = Path(path_raw)
            if not model_path.is_absolute():
                model_path = self.base_dir / model_path
            logger.info(f"[v2] 載入 model: {machine_id}/{lighting}/{zone} → {model_path}")
            self._model_cache_v2[key] = self._load_model_from_path(model_path)
        return self._model_cache_v2[key]

    def reload_submodel(self, machine_id: str, lighting: str, zone: str) -> bool:
        """重訓完成後丟掉 cache 中的舊 model，下次 inference 自動 lazy reload。

        回 True 表示有被踢掉舊 cache；False 代表本來就沒載入過（也不需要 reload）。
        """
        key = (machine_id, lighting, zone)
        if key in self._model_cache_v2:
            del self._model_cache_v2[key]
            logger.info("[v2] 已將 cache key %s pop，下次推論會 lazy reload", key)
            return True
        logger.info("[v2] cache 中無 key %s，不需要 reload", key)
        return False

    def _predict_tile(self, model, tile_img: np.ndarray, mask: Optional[np.ndarray] = None):
        """跑 PatchCore 推論 + 應用 polygon mask（如有）。

        Returns: (score: float, anomaly_map: np.ndarray | None)
        """
        # 確保圖片為 BGR 3-channel（anomalib TorchInferencer 預期 BGR uint8）
        if tile_img.ndim == 2:
            input_img = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2BGR)
        elif tile_img.shape[2] == 1:
            input_img = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2BGR)
        else:
            input_img = tile_img

        result = model.predict(input_img)
        score = float(getattr(result, "pred_score", 0.0))
        anomaly_map = getattr(result, "anomaly_map", None)

        if mask is not None and anomaly_map is not None:
            # mask 255=panel 內, 0=panel 外 → 外部分數歸 0
            mask_f = mask.astype(np.float32) / 255.0
            if anomaly_map.ndim == 3:
                mask_f = mask_f[:, :, np.newaxis]
            anomaly_map = anomaly_map * mask_f
            score = float(anomaly_map.max())

        return score, anomaly_map

    def visualize_inference_result(self, image_path: Path, result: ImageResult) -> np.ndarray:
        """視覺化推論結果（含異常標記 與 AOI 標記）"""
        image = self._read_detection_image(image_path)
        
        # 如果是灰階，轉為 BGR 以便畫上彩色標記
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        elif len(vis.shape) == 3 and vis.shape[2] == 1:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        x1, y1, x2, y2 = result.otsu_bounds  # 供後續 status 文字定位用

        # Panel polygon（紅色）— tile 範圍由綠色 grid 呈現，不再重複畫 Otsu 矩形
        if result.panel_polygon is not None:
            poly_int = result.panel_polygon.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [poly_int], True, (0, 0, 255), 6)

        # 顯示裁切區域
        if result.cropped_region:
            cx1, cy1, cx2, cy2 = result.cropped_region
            # 畫半透明黑色
            overlay = vis.copy()
            cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
            # 虛線框 (OpenCV 不直接支援虛線，用實線代替)
            cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (100, 100, 100), 3)
            cv2.putText(vis, "CROPPED", (cx1 + 10, cy1 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        
        # AOI 缺陷標記已移除 (無意義)
        
        # 排除區域 (MARK / 機構)
        for region in result.exclusion_regions:
            cv2.rectangle(vis, (region.x1, region.y1), (region.x2, region.y2), (128, 128, 128), 4)

        # CV 邊緣不檢測排除區域 (settings 設定，按機種劃分)
        EXCLUDE_ZONE_COLOR = (0, 200, 200)  # 青黃色 (BGR)
        if getattr(self, "edge_inspector", None):
            active_zones = [z for z in self.edge_inspector.config.exclude_zones if z.enabled]
            for zone in active_zones:
                zx1, zy1 = zone.x, zone.y
                zx2, zy2 = zone.x + zone.w, zone.y + zone.h
                # 半透明填充
                overlay = vis.copy()
                cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), EXCLUDE_ZONE_COLOR, -1)
                cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
                # 邊框
                cv2.rectangle(vis, (zx1, zy1), (zx2, zy2), EXCLUDE_ZONE_COLOR, 3)
                # 標籤
                cv2.putText(vis, "EXCLUDE ZONE", (zx1 + 5, zy1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, EXCLUDE_ZONE_COLOR, 2)

        # 正常 tiles（整齊的綠色網格線）
        # 收集所有唯一的水平和垂直座標，避免邊緣 tile 回推造成的雙線
        h_lines = set()  # 水平線 y 座標
        v_lines = set()  # 垂直線 x 座標
        otsu_x1, otsu_y1, otsu_x2, otsu_y2 = result.otsu_bounds
        for tile in result.tiles:
            # 對齊到最近的 tile step (通常是 512 的倍數)
            # 只取 Otsu 邊界內的線條
            if otsu_y1 <= tile.y <= otsu_y2:
                h_lines.add(tile.y)
            if otsu_y1 <= tile.y + tile.height <= otsu_y2:
                h_lines.add(tile.y + tile.height)
            if otsu_x1 <= tile.x <= otsu_x2:
                v_lines.add(tile.x)
            if otsu_x1 <= tile.x + tile.width <= otsu_x2:
                v_lines.add(tile.x + tile.width)
        
        # 畫水平線 (在 Otsu x 範圍內)
        for y in sorted(h_lines):
            cv2.line(vis, (otsu_x1, y), (otsu_x2, y), (0, 255, 0), 1, cv2.LINE_AA)
        # 畫垂直線 (在 Otsu y 範圍內)
        for x in sorted(v_lines):
            cv2.line(vis, (x, otsu_y1), (x, otsu_y2), (0, 255, 0), 1, cv2.LINE_AA)
        
        # 異常 tiles（紅色粗框 + 分數）
        effective_anomaly_tiles = [
            t for t in result.anomaly_tiles
            if not getattr(t[0], 'is_aoi_coord_below_threshold', False)
        ]
        track_only_count = len(result.anomaly_tiles) - len(effective_anomaly_tiles)
        if effective_anomaly_tiles:
            suffix = f"，另 {track_only_count} 個 track-only" if track_only_count else ""
            print(f"🔍 發現異常 tiles，共 {len(effective_anomaly_tiles)} 個{suffix}")
        elif track_only_count:
            print(f"🟡 AOI track-only tiles，共 {track_only_count} 個 (Score<THR，不列 NG)")
        for tile, score, _ in result.anomaly_tiles:
            # AOI 座標 tile 但 AI 判定未達閾值：綠色框標 OK（不算 NG）
            if getattr(tile, 'is_aoi_coord_below_threshold', False):
                color = (0, 255, 0)  # 綠色 (BGR)
                label = f"{score:.2f} OK"
                thickness = 3
            else:
                color = (0, 0, 255)  # 紅色 (預設異常)
                label = f"{score:.2f}"
                thickness = 6

            # 不檢測排除區域：灰色虛線風格
            if getattr(tile, 'is_in_exclude_zone', False):
                color = (180, 180, 180)  # 灰色 (BGR)
                label = f"{score:.2f} EXCLUDED"
                thickness = 3
            # 炸彈 tile：洋紅色 (紫色)
            elif getattr(tile, 'is_bomb', False):
                color = (255, 0, 255)  # 洋紅色 (BGR)
                code = getattr(tile, 'bomb_defect_code', '')
                label = f"{score:.2f} BOMB({code})"
            # 如果是疑似灰塵/刮痕，改為黃色
            elif getattr(tile, 'is_suspected_dust_or_scratch', False):
                color = (0, 255, 255)  # 黃色 (BGR: 0, 255, 255)
                metric_name = "COV" if self.config.dust_heatmap_metric == "coverage" else "IOU"
                rcov = getattr(tile, 'dust_region_max_cov', 0.0)
                label = f"{score:.2f} DUST(R.{metric_name}:{rcov:.3f})"
            elif getattr(tile, 'dust_heatmap_iou', 0.0) > 0:
                # 有 OMIT 分析結果但非灰塵，顯示 per-region max COV
                metric_name = "COV" if self.config.dust_heatmap_metric == "coverage" else "IOU"
                rcov = getattr(tile, 'dust_region_max_cov', 0.0)
                label = f"{score:.2f} NG(R.{metric_name}:{rcov:.3f})"
            
            # 座標邊界檢查
            h, w = vis.shape[:2]
            x1_clip = max(0, min(tile.x, w-1))
            y1_clip = max(0, min(tile.y, h-1))
            x2_clip = max(0, min(tile.x + tile.width, w))
            y2_clip = max(0, min(tile.y + tile.height, h))
            
            try:
                cv2.rectangle(vis, (x1_clip, y1_clip), (x2_clip, y2_clip), color, thickness)
                
                # 在 tile 中心畫一個大圓作為額外標記
                center_x = (x1_clip + x2_clip) // 2
                center_y = (y1_clip + y2_clip) // 2
                cv2.circle(vis, (center_x, center_y), 80, color, 10)
                
                # 文字位置也要在邊界內
                text_x = max(10, min(x1_clip + 10, w - 200))
                text_y = max(50, min(y1_clip + 50, h - 10))
                cv2.putText(vis, label, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                
                # 右下角標上區域編號 (e.g. #64)
                id_label = f"#{tile.tile_id}"
                (tw_text, th_text), _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
                id_x = max(0, min(x2_clip - tw_text - 10, w - tw_text))
                id_y = max(th_text, min(y2_clip - 10, h - 5))
                cv2.putText(vis, id_label, (id_x, id_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            except Exception as e:
                print(f"❌ 繪製 Tile {tile.tile_id} 失敗: {e}, 座標: ({x1_clip},{y1_clip})->({x2_clip},{y2_clip})")
        
        # 結果標籤 — 過濾掉 AOI 座標未達閾值的 tile (不影響 NG 判定)
        real_edge_defects = [
            ed for ed in getattr(result, 'edge_defects', [])
            if not getattr(ed, 'is_suspected_dust_or_scratch', False)
            and not getattr(ed, 'is_bomb', False)
            and not getattr(ed, 'is_cv_ok', False)
        ]
        status = "NG" if effective_anomaly_tiles else "OK"
        # 檢查是否所有異常都是疑似灰塵
        all_dust = False
        all_bomb = False
        all_excluded = False
        if effective_anomaly_tiles:
            non_dust = [t for t in effective_anomaly_tiles if not t[0].is_suspected_dust_or_scratch or t[0].is_bomb]
            all_dust = all(t[0].is_suspected_dust_or_scratch and not t[0].is_bomb for t in effective_anomaly_tiles)
            all_bomb = non_dust and all(t[0].is_bomb for t in non_dust)
            all_excluded = all(
                t[0].is_in_exclude_zone or t[0].is_suspected_dust_or_scratch or t[0].is_bomb
                for t in effective_anomaly_tiles
            ) and any(t[0].is_in_exclude_zone for t in effective_anomaly_tiles)
            if all_excluded and not any(
                not t[0].is_in_exclude_zone and not t[0].is_suspected_dust_or_scratch and not t[0].is_bomb
                for t in effective_anomaly_tiles
            ):
                status = "OK (Excluded)"
            elif all_dust:
                status = "NG (Dust?)"
            elif all_bomb:
                status = "BOMB"

        if real_edge_defects:
            status = "NG"
            all_dust = False
            all_bomb = False
            all_excluded = False

        if not effective_anomaly_tiles and not real_edge_defects:
            color = (0, 255, 0)
        elif all_excluded and status == "OK (Excluded)":
            color = (180, 180, 180)  # 灰色
        elif all_bomb:
            color = (255, 0, 255)  # 洋紅色
        elif all_dust:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        
        cv2.putText(vis, status, (x1 + 20, y1 + 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, color, 10)
        
        # 標記 CV 邊緣檢測異常
        if hasattr(result, 'edge_defects') and result.edge_defects:
            for ed in result.edge_defects:
                bx, by, bw, bh = ed.bbox
                
                is_bomb_ed = getattr(ed, 'is_bomb', False)
                is_dust = getattr(ed, 'is_suspected_dust_or_scratch', False)
                
                if is_bomb_ed:
                    box_color = (255, 0, 255)  # 紫色 (洋紅色)
                elif is_dust:
                    box_color = (0, 165, 255)
                else:
                    box_color = (0, 0, 255)    # 紅色
                
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), box_color, 4)
                
                # 在框的旁邊加上文字（處理文字是否超出邊界的邏輯）
                if is_bomb_ed:
                    status_label = f"BOMB({getattr(ed, 'bomb_defect_code', '')})"
                elif is_dust:
                    status_label = "DUST"
                else:
                    status_label = "NG"
                text = f"Edge {status_label}: {ed.side} ({ed.max_diff:.0f})"
                text_x = max(10, bx)
                text_y = max(30, by - 10)
                if ed.side == 'top':
                    text_y = by + bh + 30
                elif ed.side == 'left':
                    text_x = bx + bw + 10
                    
                cv2.putText(vis, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 3)

        return vis
    
    def generate_bomb_diagram(self, image_path: Path, result: ImageResult,
                              product_resolution: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        生成炸彈座標位置示意圖
        在原始圖片上疊加：
        1. 已設定的炸彈座標位置 (洋紅色十字)
        2. AD 偵測到的異常 tile 位置 (青色方框)
        3. 匹配連線 + 距離標示
        """
        if product_resolution is None:
            product_resolution = DEFAULT_PRODUCT_RESOLUTION
        image = self._read_detection_image(image_path)
        if image is None:
            image = np.zeros((product_resolution[1], product_resolution[0], 3), dtype=np.uint8)
        
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        elif len(vis.shape) == 3 and vis.shape[2] == 1:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        # 不暗化，直接使用原始圖片
        
        raw_bounds = result.raw_bounds
        if raw_bounds is None:
            return vis
        
        img_prefix = image_path.stem
        BOMB_COLOR = (255, 0, 255)    # 洋紅色 = 炸彈設定座標
        AD_COLOR = (255, 255, 0)      # 青色 = AD 偵測到的 tile
        MATCH_LINE_COLOR = (0, 255, 0) # 綠色 = 匹配連線
        
        # 計算 tolerance (與 check_bomb_match 一致)
        tolerance = self.config.bomb_match_tolerance
        PRODUCT_WIDTH, PRODUCT_HEIGHT = product_resolution
        x_start, y_start, x_end, y_end = raw_bounds
        scale_x = (x_end - x_start) / PRODUCT_WIDTH
        scale_y = (y_end - y_start) / PRODUCT_HEIGHT
        img_tolerance_x = int(tolerance * scale_x)
        img_tolerance_y = int(tolerance * scale_y)
        
        # === 1. 繪製所有 AD 偵測到的異常 tile (青色框 + 峰值點) ===
        ad_tiles_info = []  # [(peak_x, peak_y, tile, score)]
        if result.anomaly_tiles:
            for tile, score, _ in result.anomaly_tiles:
                # 使用熱力圖峰值位置 (更精確)
                if tile.anomaly_peak_x >= 0 and tile.anomaly_peak_y >= 0:
                    px, py = tile.anomaly_peak_x, tile.anomaly_peak_y
                else:
                    px, py = tile.x + tile.width // 2, tile.y + tile.height // 2
                ad_tiles_info.append((px, py, tile, score))
                
                # 畫 tile 框 (半透明)
                cv2.rectangle(vis, (tile.x, tile.y), 
                              (tile.x + tile.width, tile.y + tile.height), 
                              AD_COLOR, 3)
                # 畫峰值點 (實心圓 = 精確缺陷位置)
                cv2.circle(vis, (px, py), 20, AD_COLOR, -1)
                # 從峰值到 tile 框畫十字準星
                cv2.line(vis, (px - 40, py), (px + 40, py), (0, 0, 255), 2)
                cv2.line(vis, (px, py - 40), (px, py + 40), (0, 0, 255), 2)
                
                # 標籤
                label = f"AD: {score:.2f}"
                if tile.is_bomb:
                    label += f" [BOMB:{tile.bomb_defect_code}]"
                elif tile.is_suspected_dust_or_scratch:
                    label += " [DUST]"
                cv2.putText(vis, label, (tile.x + 5, tile.y + tile.height - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, AD_COLOR, 2)
        
        # === 2. 繪製炸彈設定座標 (洋紅色十字) + 匹配連線 ===
        bomb_count = 0
        matched_count = 0
        
        active_bombs = []
        if result.client_bomb_info:
            bomb_info = result.client_bomb_info
            defect_code = self._match_bomb_defect_code(bomb_info)
            active_bombs = [BombDefect(
                image_prefix=bomb_info["image_prefix"],
                defect_code=defect_code,
                defect_type=bomb_info["defect_type"],
                coordinates=bomb_info["coordinates"],
            )]
        elif self.config.bomb_defects:
            active_bombs = self.config.bomb_defects
            
        for bomb in active_bombs:
            if not (img_prefix == bomb.image_prefix or
                    img_prefix.startswith(bomb.image_prefix + "_")):
                continue
            
            if bomb.defect_type == "line" and len(bomb.coordinates) >= 2:
                pt1 = bomb.coordinates[0]
                pt2 = bomb.coordinates[1]
                img_x1, img_y1 = self._map_aoi_coords(pt1[0], pt1[1], raw_bounds)
                img_x2, img_y2 = self._map_aoi_coords(pt2[0], pt2[1], raw_bounds)
                
                cv2.line(vis, (img_x1, img_y1), (img_x2, img_y2), BOMB_COLOR, 6)
                mid_x = (img_x1 + img_x2) // 2
                mid_y = (img_y1 + img_y2) // 2
                label = f"BOMB LINE ({bomb.defect_code})"
                cv2.putText(vis, label, (mid_x + 20, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, BOMB_COLOR, 3)
                bomb_count += 1
                
            elif bomb.defect_type == "point":
                for idx, coord in enumerate(bomb.coordinates):
                    img_bx, img_by = self._map_aoi_coords(coord[0], coord[1], raw_bounds)
                    bomb_count += 1
                    
                    # 找最近的 AD tile 並計算距離
                    nearest_dist = float('inf')
                    nearest_ad = None
                    is_matched = False
                    for acx, acy, atile, ascore in ad_tiles_info:
                        dx = abs(img_bx - acx)
                        dy = abs(img_by - acy)
                        dist = (dx**2 + dy**2) ** 0.5
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_ad = (acx, acy)
                        # 使用與 check_bomb_match 相同的容忍度
                        if dx <= img_tolerance_x and dy <= img_tolerance_y:
                            is_matched = True
                    
                    if is_matched:
                        matched_count += 1
                    
                    pt_color = MATCH_LINE_COLOR if is_matched else BOMB_COLOR
                    
                    # 畫十字 + 圓
                    size = 60
                    cv2.circle(vis, (img_bx, img_by), size, pt_color, 5)
                    cv2.line(vis, (img_bx - size, img_by), (img_bx + size, img_by), pt_color, 3)
                    cv2.line(vis, (img_bx, img_by - size), (img_bx, img_by + size), pt_color, 3)
                    
                    # 座標標籤 (產品座標)
                    label = f"#{idx+1} ({coord[0]},{coord[1]})"
                    status_txt = "MATCHED" if is_matched else "NOT DETECTED"
                    cv2.putText(vis, label, (img_bx + size + 10, img_by - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, pt_color, 3)
                    cv2.putText(vis, status_txt, (img_bx + size + 10, img_by + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, pt_color, 2)
                    
                    # 畫到最近 AD tile 的連線和距離
                    if nearest_ad is not None:
                        line_color = MATCH_LINE_COLOR if is_matched else (128, 128, 128)
                        cv2.line(vis, (img_bx, img_by), nearest_ad, line_color, 2, cv2.LINE_AA)
                        mid_lx = (img_bx + nearest_ad[0]) // 2
                        mid_ly = (img_by + nearest_ad[1]) // 2
                        dist_label = f"d={nearest_dist:.0f}px"
                        cv2.putText(vis, dist_label, (mid_lx + 10, mid_ly - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)
        
        # === 3. 標題列 ===
        header_h = 90
        cv2.rectangle(vis, (0, 0), (vis.shape[1], header_h), (0, 0, 0), -1)
        title = f"BOMB Diagram: {image_path.stem} | Bombs: {bomb_count} | AD Tiles: {len(ad_tiles_info)} | Matched: {matched_count}"
        cv2.putText(vis, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, BOMB_COLOR, 3)
        tol_text = f"Tolerance: {tolerance}px (product) -> {img_tolerance_x}x{img_tolerance_y}px (image)"
        cv2.putText(vis, tol_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        
        # === 4. 圖例列 ===
        legend_h = 50
        legend_y0 = vis.shape[0] - legend_h
        cv2.rectangle(vis, (0, legend_y0), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
        ly = legend_y0 + 30
        # 炸彈座標
        cv2.circle(vis, (30, ly), 12, BOMB_COLOR, -1)
        cv2.putText(vis, "Bomb Config", (55, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BOMB_COLOR, 2)
        # AD tile
        cv2.rectangle(vis, (340, ly - 12), (364, ly + 12), AD_COLOR, 3)
        cv2.putText(vis, "AD Detected", (375, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, AD_COLOR, 2)
        # 匹配
        cv2.circle(vis, (660, ly), 12, MATCH_LINE_COLOR, -1)
        cv2.putText(vis, "Matched", (685, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, MATCH_LINE_COLOR, 2)
        
        return vis


def test_inferencer():
    """測試推論器"""
    from capi_config import CAPIConfig
    
    print("=" * 60)
    print("CAPI 推論器測試")
    print("=" * 60)
    
    # 載入配置
    config = CAPIConfig.from_yaml("configs/capi_3f.yaml")
    print(f"\n配置: {config}")
    
    # 建立推論器
    inferencer = CAPIInferencer(config)
    
    # 測試圖片
    test_folder = Path(r"D:\CAPI_3F\ok")
    panel_folders = sorted([f for f in test_folder.iterdir() if f.is_dir()])[:1]
    
    output_dir = Path("capi_inference_test")
    output_dir.mkdir(exist_ok=True)
    
    for panel_folder in panel_folders:
        print(f"\n=== 面板: {panel_folder.name} ===")
        
        image_files = list(panel_folder.glob("*.png"))[:2]
        
        for img_file in image_files:
            if img_file.stem.startswith("PINIGBI") or "OMIT0000" in img_file.name:
                continue
            
            result = inferencer.preprocess_image(img_file)
            if result:
                print(f"  {img_file.name}:")
                print(f"    - 尺寸: {result.image_size}")
                print(f"    - Otsu: {result.otsu_bounds}")
                print(f"    - 排除區域: {len(result.exclusion_regions)}")
                print(f"    - Tiles: {result.processed_tile_count} (排除: {result.excluded_tile_count})")
                print(f"    - 時間: {result.processing_time:.2f}s")
                
                # 視覺化
                output_path = output_dir / f"{panel_folder.name}_{img_file.stem}_tiles.jpg"
                inferencer.visualize_preprocessing(img_file, result, output_path)
    
    print(f"\n輸出目錄: {output_dir}")


if __name__ == "__main__":
    test_inferencer()


class SubmodelScorer:
    """對既有 PatchCore .pt + 一批 tile 跑分，結果寫進 tile_score_cache。

    純跑分 + 寫 cache，不做 reject 判斷、不動 anomaly map 後處理（用線上預設）。
    """

    def __init__(self, gpu_lock, db, log_fn):
        self.gpu_lock = gpu_lock
        self.db = db
        self.log = log_fn
        self._inferencer_cache = {}  # path_str → TorchInferencer

    def _load_inferencer_for_pt(self, pt_path: Path):
        """Lazy load TorchInferencer，路徑為 key 做 cache。"""
        key = str(pt_path)
        if key in self._inferencer_cache:
            return self._inferencer_cache[key]
        if not pt_path.exists():
            raise FileNotFoundError(f"模型檔不存在: {pt_path}")
        from anomalib.deploy import TorchInferencer
        inf = TorchInferencer(path=str(pt_path), device="auto")
        CAPIInferencer._fix_legacy_precision(inf)
        self._inferencer_cache[key] = inf
        return inf

    def _score_one_tile(self, image: np.ndarray, inferencer) -> float:
        """跑一張 tile，回 raw pred_score（anomalib normalized score, 不做 production post-processing）。

        Anomalib 的 TorchInferencer 需要 3-channel BGR；單通道輸入要先轉。
        """
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        pred = inferencer.predict(image=image)
        score = float(pred.pred_score.item()) if hasattr(pred.pred_score, "item") \
                else float(pred.pred_score)
        return score

    def score_image_path(
        self,
        *,
        bundle_dir: Path,
        lighting: str,
        zone: str,
        image_path: Path,
        preprocess_pipeline=None,
    ) -> float:
        """對一張已保存的 tile crop 套用 bundle 前處理後跑 raw score。"""
        pt_path = Path(bundle_dir) / f"{lighting}-{zone}.pt"
        inferencer = self._load_inferencer_for_pt(pt_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"圖片不存在或無法讀取: {image_path}")
        if preprocess_pipeline:
            from capi_image_preprocess_lab import apply_preprocess_pipeline
            image = apply_preprocess_pipeline(image, preprocess_pipeline)["image"]
        with self.gpu_lock:
            return self._score_one_tile(image, inferencer)

    def score_tiles(
        self,
        scoring_bundle_id: int,
        bundle_dir: Path,
        lighting: str,
        zone: str,
        tile_pool_job_id: str,
        tile_ids: list,
        cancel_event,
        progress_cb,
    ) -> dict:
        """主 entry。回 {scanned, skipped, cancelled, total}。

        Raises FileNotFoundError 如果 <lighting>-<zone>.pt 不存在。
        """
        pt_path = bundle_dir / f"{lighting}-{zone}.pt"
        inferencer = self._load_inferencer_for_pt(pt_path)

        # 一次性查 source_path
        pool = self.db.list_tile_pool(tile_pool_job_id, lighting=lighting, zone=zone)
        path_by_id = {row["id"]: row["source_path"] for row in pool}

        total = len(tile_ids)
        scanned = 0
        skipped = 0
        cancelled = False
        progress_cb(0, total)

        rows_to_write = []
        BATCH_FLUSH = 20

        for tile_id in tile_ids:
            if cancel_event.is_set():
                cancelled = True
                break
            src = path_by_id.get(tile_id)
            if not src or not Path(src).exists():
                self.log(f"[scorer] tile {tile_id} source 失效，跳過")
                skipped += 1
                progress_cb(scanned + skipped, total)
                continue
            try:
                img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    self.log(f"[scorer] tile {tile_id} 讀圖失敗，跳過")
                    skipped += 1
                    progress_cb(scanned + skipped, total)
                    continue
                with self.gpu_lock:
                    score = self._score_one_tile(img, inferencer)
                rows_to_write.append({
                    "tile_id": tile_id,
                    "scoring_bundle_id": scoring_bundle_id,
                    "score": score,
                })
                scanned += 1
                if len(rows_to_write) >= BATCH_FLUSH:
                    self.db.insert_score_cache(rows_to_write)
                    rows_to_write = []
                progress_cb(scanned + skipped, total)
            except Exception as e:
                self.log(f"[scorer] tile {tile_id} 跑分失敗：{type(e).__name__}: {e}")
                skipped += 1
                progress_cb(scanned + skipped, total)

        if rows_to_write:
            self.db.insert_score_cache(rows_to_write)

        return {
            "scanned": scanned,
            "skipped": skipped,
            "cancelled": cancelled,
            "total": total,
        }
