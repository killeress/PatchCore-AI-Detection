"""
過檢訓練資料集蒐集工具

從 RIC 回填結果蒐集 AI=NG&RIC=NG (true_ng) 與 AI=NG&RIC=OK 已回填 category 的樣本，
輸出 crop + heatmap 對到分類目錄結構 + manifest.csv，供後續分類 AI 訓練或 CV 後處理評估。

Spec: docs/superpowers/specs/2026-04-10-over-review-dataset-collector-design.md
"""
from __future__ import annotations

import csv
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---- 常數 ----

CROP_SIZE = 512

# 與 capi_database.py VALID_OVER_CATEGORIES 對應的 label 目錄名
OVER_LABEL_MAP = {
    "edge_false_positive": "over_edge_false_positive",
    "within_spec": "over_within_spec",
    "overexposure": "over_overexposure",
    "surface_scratch": "over_surface_scratch",
    "aoi_ai_false_positive": "over_aoi_ai_false_positive",
    "other": "over_other",
}

TRUE_NG_LABEL = "true_ng"

# Manifest CSV 欄位順序（固定，避免 DictWriter 亂序）
MANIFEST_FIELDS = [
    "sample_id", "collected_at", "label", "source_type", "prefix",
    "glass_id", "image_name",
    "inference_record_id", "image_result_id",
    "tile_idx", "edge_defect_id",
    "crop_path", "heatmap_path",
    "ai_score", "defect_x", "defect_y",
    "ric_judgment", "over_review_category", "over_review_note",
    "inference_timestamp", "status",
]

# Job 狀態常數
JOB_STATE_IDLE = "idle"
JOB_STATE_RUNNING = "running"
JOB_STATE_COMPLETED = "completed"
JOB_STATE_FAILED = "failed"
JOB_STATE_CANCELLED = "cancelled"


def determine_label(ric: str, over_category: Optional[str]) -> Optional[str]:
    """依 RIC 判定與 over_review category 決定輸出 label。

    Returns:
        - "true_ng" 若 RIC=NG
        - "over_<category>" 若 RIC=OK 且 category 在 OVER_LABEL_MAP
        - None 若 RIC=OK 且 category 未填（這類樣本不蒐集）

    Raises:
        ValueError: 若 RIC=OK 且 category 不在合法 enum
    """
    if ric == "NG":
        return TRUE_NG_LABEL
    if ric == "OK":
        if not over_category:
            return None
        if over_category not in OVER_LABEL_MAP:
            raise ValueError(f"Unknown over_review category: {over_category}")
        return OVER_LABEL_MAP[over_category]
    raise ValueError(f"Unknown RIC judgment: {ric}")


def extract_prefix(image_name: str) -> str:
    """從原圖檔名抽出光源 prefix（去掉 timestamp 尾綴）。

    Mirror of capi_inference.CAPIInferencer._get_image_prefix but stand-alone
    so 本工具不需要 inferencer 實例就能分類樣本。

    Examples:
        G0F00000_114438.tif → G0F00000
        STANDARD.png → STANDARD
        WGF_0001_20260410.bmp → WGF_0001
    """
    stem = Path(image_name).stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def build_sample_id(glass_id: str, image_name: str, source_type: str,
                    tile_idx: Optional[int] = None,
                    edge_defect_id: Optional[int] = None) -> str:
    """去重 key：glass_id + image_stem + sample_key"""
    stem = Path(image_name).stem
    if source_type == "patchcore_tile":
        if tile_idx is None:
            raise ValueError("tile_idx required for patchcore_tile")
        return f"{glass_id}_{stem}_tile{tile_idx}"
    if source_type == "edge_defect":
        if edge_defect_id is None:
            raise ValueError("edge_defect_id required for edge_defect")
        return f"{glass_id}_{stem}_edge{edge_defect_id}"
    raise ValueError(f"Unknown source_type: {source_type}")


def build_sample_filename(glass_id: str, image_name: str,
                          sample_key: str, inference_timestamp: str) -> str:
    """檔名：{YYYYMMDD}_{glass_id}_{image_stem}_{sample_key}.png

    inference_timestamp 支援 'YYYY-MM-DDTHH:MM:SS' 與 'YYYY-MM-DD HH:MM:SS'。
    """
    stem = Path(image_name).stem
    ts = inference_timestamp.replace("T", " ")[:10]  # 'YYYY-MM-DD'
    yyyymmdd = ts.replace("-", "")
    return f"{yyyymmdd}_{glass_id}_{stem}_{sample_key}.png"


# ---- Crop Tool Functions ----

def _pad_to_size(img: np.ndarray, target: int = CROP_SIZE,
                 pad_top: int = 0, pad_left: int = 0) -> np.ndarray:
    """將 img pad 到 target×target，黑邊 (value=0)。
    pad_top / pad_left 允許指定「原圖左上角在 target 中的偏移」，
    其餘空間用 BORDER_CONSTANT 補黑。
    """
    h, w = img.shape[:2]
    pad_bottom = target - h - pad_top
    pad_right = target - w - pad_left
    if pad_top < 0 or pad_left < 0 or pad_bottom < 0 or pad_right < 0:
        raise ValueError(
            f"Image too large for target: shape=({h},{w}) target={target} "
            f"offset=({pad_top},{pad_left})"
        )
    return cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0
    )


def crop_patchcore_tile(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """從原圖切出 PatchCore tile 區域，不足 CROP_SIZE 用黑邊在右/下側 pad。

    Args:
        img: 原圖 (H, W, C) 或 (H, W)
        x, y, w, h: tile 在原圖的絕對座標（tile_results.x/y/width/height）

    Returns:
        CROP_SIZE × CROP_SIZE 的 crop
    """
    H, W = img.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    crop = img[y1:y2, x1:x2]
    if crop.shape[:2] == (CROP_SIZE, CROP_SIZE):
        return crop
    # 右/下 pad（tile 通常在 grid 邊緣才會不足）
    return _pad_to_size(crop, CROP_SIZE, pad_top=0, pad_left=0)


def crop_edge_defect(img: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """以 (cx, cy) 為中心切出 CROP_SIZE × CROP_SIZE，clamp 後再 pad 保持中心對齐。

    Args:
        img: 原圖
        cx, cy: edge_defect_results.center_x / center_y

    Returns:
        CROP_SIZE × CROP_SIZE 的 crop
    """
    half = CROP_SIZE // 2
    H, W = img.shape[:2]
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half

    x1_c = max(0, x1)
    y1_c = max(0, y1)
    x2_c = min(W, x2)
    y2_c = min(H, y2)
    crop = img[y1_c:y2_c, x1_c:x2_c]

    # 以 clamp 掉的量當 top/left pad，確保 defect 中心在 crop 的 (half, half)
    pad_top = y1_c - y1
    pad_left = x1_c - x1
    return _pad_to_size(crop, CROP_SIZE, pad_top=pad_top, pad_left=pad_left)


# ---- Dataclasses ----

@dataclass
class SampleCandidate:
    """一個待處理樣本（尚未實際寫入 disk）"""
    sample_id: str          # 去重 key: f"{glass_id}_{image_stem}_{sample_key}"
    source_type: str        # "patchcore_tile" | "edge_defect"
    glass_id: str
    image_name: str
    image_path: str         # 原圖完整路徑（已做 UNC → Linux mapping）
    inference_record_id: int
    image_result_id: int
    tile_idx: Optional[int]
    edge_defect_id: Optional[int]
    prefix: str             # G0F / R0F / W0F / WGF / STANDARD / …
    label: str              # true_ng / over_edge_false_positive / …
    # Crop 座標資訊（依 source_type 填不同欄位，未用到的留 None）
    tile_x: Optional[int] = None
    tile_y: Optional[int] = None
    tile_w: Optional[int] = None
    tile_h: Optional[int] = None
    edge_center_x: Optional[int] = None
    edge_center_y: Optional[int] = None
    # 通用 metadata
    src_heatmap_path: str = ""      # DB 紀錄的原 heatmap 檔案路徑
    ai_score: float = 0.0
    ric_judgment: str = ""
    over_review_category: str = ""
    over_review_note: str = ""
    inference_timestamp: str = ""


@dataclass
class JobSummary:
    """Job 完成後的統計摘要"""
    job_id: str
    started_at: str
    finished_at: str
    duration_sec: float
    total: int = 0
    labels: Dict[str, int] = field(default_factory=dict)  # label → 數量
    skipped: Dict[str, int] = field(default_factory=dict)  # reason → 數量
    output_dir: str = ""


# ---- Exporter（實作在後續 Task 加入） ----

class DatasetExporter:
    """過檢訓練資料集蒐集器（pure logic，不含 HTTP）"""

    def __init__(self, db, base_dir: str, path_mapping: Dict[str, str]):
        self.db = db
        self.base_dir = Path(base_dir).resolve()
        self.path_mapping = path_mapping

    def run(self, days: int, include_true_ng: bool, skip_existing: bool,
            status_callback=None, cancel_event: Optional[threading.Event] = None
            ) -> JobSummary:
        """主執行入口（實作在 Task 6）"""
        raise NotImplementedError("Implemented in Task 6")
