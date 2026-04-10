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
