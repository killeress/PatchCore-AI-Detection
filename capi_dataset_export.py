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


def read_manifest(manifest_path: Path) -> Dict[str, Dict[str, str]]:
    """讀 manifest.csv 成 {sample_id: row_dict}。檔案不存在回空 dict。"""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("sample_id")
            if sid:
                out[sid] = row
    return out


def write_manifest(manifest_path: Path, rows: Dict[str, Dict[str, str]]) -> None:
    """整批 rewrite manifest.csv（呼叫者須持有 job lock 保證單寫入者）"""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for sid in sorted(rows.keys()):
            row = rows[sid]
            # 確保所有欄位都存在（補空值）
            full = {k: str(row.get(k, "")) for k in MANIFEST_FIELDS}
            writer.writerow(full)
    tmp_path.replace(manifest_path)  # atomic on same filesystem


def move_sample_files(
    base_dir: Path, old_crop_rel: str, old_heatmap_rel: str,
    new_label: str, prefix: str,
) -> Tuple[str, str]:
    """把 crop 與 heatmap 從舊 label 目錄 move 到新 label 目錄。

    Returns:
        (new_crop_rel, new_heatmap_rel) — 兩個新的相對路徑
    """
    base_dir = Path(base_dir)
    old_crop = base_dir / old_crop_rel
    old_hm = base_dir / old_heatmap_rel

    filename = old_crop.name
    new_crop_rel = f"{new_label}/{prefix}/crop/{filename}"
    new_hm_rel = f"{new_label}/{prefix}/heatmap/{filename}"
    new_crop = base_dir / new_crop_rel
    new_hm = base_dir / new_hm_rel

    new_crop.parent.mkdir(parents=True, exist_ok=True)
    new_hm.parent.mkdir(parents=True, exist_ok=True)

    if old_crop.exists():
        shutil.move(str(old_crop), str(new_crop))
    if old_hm.exists():
        shutil.move(str(old_hm), str(new_hm))

    return new_crop_rel, new_hm_rel


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

    def collect_candidates(self, days: int, include_true_ng: bool) -> List[SampleCandidate]:
        """從 DB 讀最近 N 日的 client_accuracy_records，展開成 candidate 清單。

        篩選規則：
          - result_ai == "NG"
          - 若 result_ric == "NG" 且 include_true_ng → 收為 true_ng
          - 若 result_ric == "OK" 且 over_review_category 已填 → 收為 over_<category>
          - 其他情況跳過

        過濾：
          - image_results.is_bomb == 1 整張跳過
          - tile_results.is_bomb == 1 該 tile 跳過
          - tile_results.is_anomaly == 0 跳過（只蒐集 NG tile）
        """
        end = datetime.now().date()
        start = end - timedelta(days=days - 1)
        rows = self.db.get_client_accuracy_records(
            start_date=start.isoformat(), end_date=end.isoformat()
        )

        candidates: List[SampleCandidate] = []
        for row in rows:
            if row.get("result_ai") != "NG":
                continue

            ric = row.get("result_ric") or ""
            over_category = row.get("over_review_category")

            if ric == "NG":
                if not include_true_ng:
                    continue
                label = TRUE_NG_LABEL
            elif ric == "OK":
                if not over_category:
                    continue  # 未回填的跳過
                try:
                    label = determine_label(ric, over_category)
                except ValueError as e:
                    logger.warning("Skip unknown over_review category: %s", e)
                    continue
                if label is None:
                    continue
            else:
                continue

            inference_record_id = row.get("inference_record_id")
            if not inference_record_id:
                logger.warning("No inference_record linked: pnl_id=%s time=%s",
                               row.get("pnl_id"), row.get("time_stamp"))
                continue

            detail = self.db.get_record_detail(inference_record_id)
            if not detail:
                continue

            candidates.extend(self._flatten_record_to_candidates(
                detail=detail, label=label, row=row,
            ))

        return candidates

    def _flatten_record_to_candidates(
        self, detail: Dict, label: str, row: Dict
    ) -> List[SampleCandidate]:
        """把一筆 inference_record 展開成多個 SampleCandidate（依 image / tile / edge）。"""
        out: List[SampleCandidate] = []
        glass_id = detail.get("glass_id") or row.get("pnl_id") or ""
        inference_timestamp = detail.get("request_time") or row.get("time_stamp") or ""
        record_id = detail.get("id")

        for img in detail.get("images") or []:
            if img.get("is_bomb"):
                continue
            image_name = img.get("image_name") or ""
            image_path = img.get("image_path") or ""
            image_result_id = img.get("id")
            prefix = extract_prefix(image_name)

            # PatchCore tile 樣本
            for tile in img.get("tiles") or []:
                if tile.get("is_bomb"):
                    continue
                if not tile.get("is_anomaly"):
                    continue
                tile_idx = tile.get("tile_id", 0)
                sample_id = build_sample_id(glass_id, image_name, "patchcore_tile", tile_idx=tile_idx)
                out.append(SampleCandidate(
                    sample_id=sample_id,
                    source_type="patchcore_tile",
                    glass_id=glass_id,
                    image_name=image_name,
                    image_path=image_path,
                    inference_record_id=record_id,
                    image_result_id=image_result_id,
                    tile_idx=tile_idx,
                    edge_defect_id=None,
                    prefix=prefix,
                    label=label,
                    tile_x=tile.get("x", 0),
                    tile_y=tile.get("y", 0),
                    tile_w=tile.get("width", CROP_SIZE),
                    tile_h=tile.get("height", CROP_SIZE),
                    src_heatmap_path=tile.get("heatmap_path", ""),
                    ai_score=float(tile.get("score", 0.0)),
                    ric_judgment=row.get("result_ric") or "",
                    over_review_category=row.get("over_review_category") or "",
                    over_review_note=row.get("over_review_note") or "",
                    inference_timestamp=inference_timestamp,
                ))

            # Edge defect 樣本
            for edge in img.get("edge_defects") or []:
                edge_id = edge.get("id")
                sample_id = build_sample_id(glass_id, image_name, "edge_defect", edge_defect_id=edge_id)
                out.append(SampleCandidate(
                    sample_id=sample_id,
                    source_type="edge_defect",
                    glass_id=glass_id,
                    image_name=image_name,
                    image_path=image_path,
                    inference_record_id=record_id,
                    image_result_id=image_result_id,
                    tile_idx=None,
                    edge_defect_id=edge_id,
                    prefix=prefix,
                    label=label,
                    edge_center_x=edge.get("center_x", 0),
                    edge_center_y=edge.get("center_y", 0),
                    src_heatmap_path=edge.get("heatmap_path", ""),
                    ai_score=float(edge.get("max_diff", 0.0)),
                    ric_judgment=row.get("result_ric") or "",
                    over_review_category=row.get("over_review_category") or "",
                    over_review_note=row.get("over_review_note") or "",
                    inference_timestamp=inference_timestamp,
                ))

        return out

    def run(self, days: int, include_true_ng: bool, skip_existing: bool,
            status_callback=None, cancel_event: Optional[threading.Event] = None
            ) -> JobSummary:
        """主執行入口（實作在 Task 6）"""
        raise NotImplementedError("Implemented in Task 6")
