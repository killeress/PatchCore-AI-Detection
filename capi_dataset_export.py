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

# 黑畫面光源 prefix：該光源條件下面板是關燈狀態，整張圖無訓練價值，直接以檔名過濾
BLACK_IMAGE_PREFIX = "B0F"


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
    # 診斷用：collect_candidates 每階段的過濾計數（幫助排查為何 total=0）
    diagnostics: Dict[str, int] = field(default_factory=dict)


# ---- Exporter（實作在後續 Task 加入） ----

class DatasetExporter:
    """過檢訓練資料集蒐集器（pure logic，不含 HTTP）"""

    def __init__(self, db, base_dir: str, path_mapping: Dict[str, str]):
        self.db = db
        self.base_dir = Path(base_dir).resolve()
        self.path_mapping = path_mapping

    def collect_candidates(self, days: int, include_true_ng: bool) -> List[SampleCandidate]:
        """便利包裝：只回傳 candidates，忽略診斷統計。"""
        candidates, _diag = self.collect_candidates_with_diagnostics(days, include_true_ng)
        return candidates

    def collect_candidates_with_diagnostics(
        self, days: int, include_true_ng: bool
    ) -> Tuple[List[SampleCandidate], Dict[str, int]]:
        """從 DB 讀最近 N 日的 client_accuracy_records，展開成 candidate 清單，
        同時記錄每個過濾階段掉了多少筆，方便排查為何 total=0。

        篩選規則：
          - result_ai == "NG"
          - 若 result_ric == "NG" 且 include_true_ng → 收為 true_ng
          - 若 result_ric == "OK" 且 over_review_category 已填 → 收為 over_<category>
          - 其他情況跳過

        過濾：
          - image_results.is_bomb == 1 整張跳過
          - tile_results.is_bomb == 1 該 tile 跳過
          - tile_results.is_anomaly == 0 跳過（只蒐集 NG tile）

        Returns:
            (candidates, diagnostics) — diagnostics 是 dict，key 包含：
              total_accuracy_rows, ai_not_ng, ric_ng_kept, ric_ng_skipped_opt_out,
              ric_ok_filled, ric_ok_empty, ric_other, missing_inference_id,
              missing_record_detail, bomb_record_skipped, final_candidates
        """
        end = datetime.now().date()
        start = end - timedelta(days=days - 1)
        rows = self.db.get_client_accuracy_records(
            start_date=start.isoformat(), end_date=end.isoformat()
        )

        diag: Dict[str, int] = {
            "date_start": start.isoformat(),
            "date_end": end.isoformat(),
            "total_accuracy_rows": len(rows),
            "ai_not_ng": 0,
            "ric_ng_kept": 0,
            "ric_ng_skipped_opt_out": 0,
            "ric_ok_filled": 0,
            "ric_ok_empty": 0,
            "ric_other": 0,
            "missing_inference_id": 0,
            "missing_record_detail": 0,
            "bomb_record_skipped": 0,
            "final_candidates": 0,
        }

        candidates: List[SampleCandidate] = []
        for row in rows:
            if row.get("result_ai") != "NG":
                diag["ai_not_ng"] += 1
                continue

            ric = row.get("result_ric") or ""
            over_category = row.get("over_review_category")

            if ric == "NG":
                if not include_true_ng:
                    diag["ric_ng_skipped_opt_out"] += 1
                    continue
                label = TRUE_NG_LABEL
                diag["ric_ng_kept"] += 1
            elif ric == "OK":
                if not over_category:
                    diag["ric_ok_empty"] += 1
                    continue  # 未回填的跳過
                try:
                    label = determine_label(ric, over_category)
                except ValueError as e:
                    logger.warning("Skip unknown over_review category: %s", e)
                    diag["ric_ok_empty"] += 1
                    continue
                if label is None:
                    diag["ric_ok_empty"] += 1
                    continue
                diag["ric_ok_filled"] += 1
            else:
                diag["ric_other"] += 1
                continue

            inference_record_id = row.get("inference_record_id")
            if not inference_record_id:
                logger.warning("No inference_record linked: pnl_id=%s time=%s",
                               row.get("pnl_id"), row.get("time_stamp"))
                diag["missing_inference_id"] += 1
                continue

            detail = self.db.get_record_detail(inference_record_id)
            if not detail:
                diag["missing_record_detail"] += 1
                continue

            # 注意：_flatten_record_to_candidates 內部會在 client_bomb_info 非空時
            # 回空 list，這裡用前後數量差判斷是否因此被刷掉
            before_len = len(candidates)
            flattened = self._flatten_record_to_candidates(
                detail=detail, label=label, row=row,
            )
            if not flattened and detail.get("client_bomb_info"):
                diag["bomb_record_skipped"] += 1
            candidates.extend(flattened)

        diag["final_candidates"] = len(candidates)
        logger.info("collect_candidates diagnostics: %s", diag)
        return candidates, diag

    def _flatten_record_to_candidates(
        self, detail: Dict, label: str, row: Dict
    ) -> List[SampleCandidate]:
        """把一筆 inference_record 展開成多個 SampleCandidate（依 image / tile / edge）。"""
        out: List[SampleCandidate] = []

        # 若 AOI 機台 request 帶有 bomb_info，代表整片面板屬於炸彈測試，整個 record 跳過。
        # 這比 image_results.is_bomb 更可靠 —— 後者只在「有炸彈 AND 沒真 NG AND 沒邊緣缺陷」
        # 時才會設為 1，實測會漏掉同時有其他異常的炸彈 panel (capi_server.py:549)。
        if detail.get("client_bomb_info"):
            return out

        glass_id = detail.get("glass_id") or row.get("pnl_id") or ""
        inference_timestamp = detail.get("request_time") or row.get("time_stamp") or ""
        record_id = detail.get("id")

        for img in detail.get("images") or []:
            if img.get("is_bomb"):
                continue
            image_name = img.get("image_name") or ""
            # 黑光源圖 (B0F) 無訓練價值，整張跳過
            if image_name.startswith(BLACK_IMAGE_PREFIX):
                continue
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
        """主執行入口。

        Args:
            days: 抓最近 N 日 (含今天)
            include_true_ng: 是否包含 AI=NG & RIC=NG 樣本
            skip_existing: True → 已存在且 label 未變的樣本 skip
            status_callback: callable(current, total, last_glass_id) 可選
            cancel_event: threading.Event，set 後會在下一筆前停止
        """
        from capi_server import resolve_unc_path  # lazy import 避免 circular

        started_at = datetime.now()
        job_id = started_at.strftime("job_%Y%m%d_%H%M%S")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.base_dir / "manifest.csv"

        # 1. 讀 manifest
        existing = read_manifest(manifest_path)

        # 2. 蒐集 candidates（含診斷資訊）
        candidates, diag = self.collect_candidates_with_diagnostics(
            days=days, include_true_ng=include_true_ng
        )
        total = len(candidates)
        logger.info("Collected %d candidates (days=%d, include_true_ng=%s)",
                    total, days, include_true_ng)

        labels_count: Dict[str, int] = {}
        skipped_count: Dict[str, int] = {}

        # 3. 處理每個 candidate
        for idx, cand in enumerate(candidates, start=1):
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Job cancelled at %d/%d", idx, total)
                break

            if status_callback:
                try:
                    status_callback(idx, total, cand.glass_id)
                except Exception:
                    logger.exception("status_callback error")

            # 解析原圖路徑（UNC → Linux）
            source_path = self._resolve_source_path(cand.image_path, resolve_unc_path)

            new_row = self._process_candidate(
                cand=cand,
                existing_row=existing.get(cand.sample_id),
                skip_existing=skip_existing,
                source_path=source_path,
            )
            if new_row is None:
                # skip_existing 命中：manifest 已有且 label 未變 → 不動檔案，但要計數
                # 否則 summary 會顯示 total=0 誤導使用者以為什麼都沒跑到
                skipped_count["already_exists"] = skipped_count.get("already_exists", 0) + 1
                continue
            existing[cand.sample_id] = new_row

            status = new_row["status"]
            if status == "ok":
                labels_count[new_row["label"]] = labels_count.get(new_row["label"], 0) + 1
            else:
                skipped_count[status] = skipped_count.get(status, 0) + 1

        # 4. 清理 stale：manifest 內存在但本次沒在 candidate list 的樣本
        # 代表該樣本在新的過濾規則下（bomb_info / B0F 等）已不再符合蒐集條件
        current_sample_ids = {c.sample_id for c in candidates}
        stale_ids = [sid for sid in list(existing.keys()) if sid not in current_sample_ids]
        cleanup_count = 0
        for sid in stale_ids:
            old_row = existing[sid]
            if old_row.get("status") == "ok":
                # 刪實體檔
                for rel_key in ("crop_path", "heatmap_path"):
                    rel = old_row.get(rel_key) or ""
                    if rel:
                        full = self.base_dir / rel
                        if full.exists():
                            try:
                                full.unlink()
                            except OSError:
                                logger.exception("Failed to unlink stale %s: %s", rel_key, full)
            del existing[sid]
            cleanup_count += 1
        if cleanup_count:
            logger.info("Cleaned up %d stale manifest entries (no longer candidates)", cleanup_count)
            skipped_count["cleaned_stale"] = cleanup_count

        # 5. 寫回 manifest
        write_manifest(manifest_path, existing)

        finished_at = datetime.now()
        return JobSummary(
            job_id=job_id,
            started_at=started_at.isoformat(timespec="seconds"),
            finished_at=finished_at.isoformat(timespec="seconds"),
            duration_sec=(finished_at - started_at).total_seconds(),
            total=sum(labels_count.values()) + sum(skipped_count.values()),
            labels=labels_count,
            skipped=skipped_count,
            output_dir=str(self.base_dir),
            diagnostics=diag,
        )

    def _resolve_source_path(self, image_path: str, resolver) -> Path:
        """套用 path_mapping 轉換，回傳 Path；若 path_mapping 為空或已是本地路徑，直接回原字串"""
        if self.path_mapping:
            try:
                mapped = resolver(image_path, self.path_mapping)
                return Path(mapped)
            except Exception:
                logger.exception("resolve_unc_path failed for %s", image_path)
        return Path(image_path)

    def _process_candidate(
        self, cand: SampleCandidate, existing_row: Optional[Dict[str, str]],
        skip_existing: bool, source_path: Path,
    ) -> Optional[Dict[str, str]]:
        """處理一個 candidate，回傳更新後的 manifest row，或 None（代表 skip 不更動 manifest）"""
        # === 去重/移動 ===
        if existing_row is not None:
            old_label = existing_row.get("label", "")
            old_status = existing_row.get("status", "")

            if old_status == "ok" and old_label == cand.label:
                if skip_existing:
                    return None  # 完全相同，skip
                # 不 skip → 重做一次（往下走）

            if old_status == "ok" and old_label != cand.label:
                # label 變了 → move 實體檔 + 更新 row
                new_crop_rel, new_hm_rel = move_sample_files(
                    base_dir=self.base_dir,
                    old_crop_rel=existing_row.get("crop_path", ""),
                    old_heatmap_rel=existing_row.get("heatmap_path", ""),
                    new_label=cand.label,
                    prefix=cand.prefix,
                )
                updated = dict(existing_row)
                updated["label"] = cand.label
                updated["crop_path"] = new_crop_rel
                updated["heatmap_path"] = new_hm_rel
                updated["collected_at"] = datetime.now().isoformat(timespec="seconds")
                updated["over_review_category"] = cand.over_review_category
                updated["over_review_note"] = cand.over_review_note
                return updated

            if old_status.startswith("skipped_") and skip_existing:
                return None  # 之前 skip 過，不 retry

        # === 新樣本（或強制重做） ===
        row = self._build_row_stub(cand)

        # 讀原圖
        if not source_path.exists():
            row["status"] = "skipped_no_source"
            logger.warning("Source not found: %s (sample_id=%s)", source_path, cand.sample_id)
            return row

        img = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            row["status"] = "skipped_no_source"
            return row
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Crop
        if cand.source_type == "patchcore_tile":
            crop = crop_patchcore_tile(img, cand.tile_x, cand.tile_y, cand.tile_w, cand.tile_h)
            defect_x = cand.tile_x + cand.tile_w // 2
            defect_y = cand.tile_y + cand.tile_h // 2
            sample_key = f"tile{cand.tile_idx}"
        else:  # edge_defect
            # 有效像素 < 25% → skip
            H, W = img.shape[:2]
            half = CROP_SIZE // 2
            valid_w = min(W, cand.edge_center_x + half) - max(0, cand.edge_center_x - half)
            valid_h = min(H, cand.edge_center_y + half) - max(0, cand.edge_center_y - half)
            if max(0, valid_w) * max(0, valid_h) < (CROP_SIZE * CROP_SIZE) * 0.25:
                row["status"] = "skipped_out_of_bounds"
                return row
            crop = crop_edge_defect(img, cand.edge_center_x, cand.edge_center_y)
            defect_x = cand.edge_center_x
            defect_y = cand.edge_center_y
            sample_key = f"edge{cand.edge_defect_id}"

        # Heatmap 必須存在
        src_hm = Path(cand.src_heatmap_path) if cand.src_heatmap_path else None
        if src_hm is None or not src_hm.exists():
            row["status"] = "skipped_no_heatmap"
            return row

        # 決定目的路徑
        filename = build_sample_filename(
            glass_id=cand.glass_id, image_name=cand.image_name,
            sample_key=sample_key, inference_timestamp=cand.inference_timestamp,
        )
        crop_rel = f"{cand.label}/{cand.prefix}/crop/{filename}"
        hm_rel = f"{cand.label}/{cand.prefix}/heatmap/{filename}"
        crop_dst = self.base_dir / crop_rel
        hm_dst = self.base_dir / hm_rel
        crop_dst.parent.mkdir(parents=True, exist_ok=True)
        hm_dst.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(crop_dst), crop)
        shutil.copy2(str(src_hm), str(hm_dst))

        row["crop_path"] = crop_rel
        row["heatmap_path"] = hm_rel
        row["defect_x"] = str(defect_x)
        row["defect_y"] = str(defect_y)
        row["status"] = "ok"
        return row

    def _build_row_stub(self, cand: SampleCandidate) -> Dict[str, str]:
        """從 candidate 組出 manifest row 的初始值（尚未 crop/copy）"""
        return {
            "sample_id": cand.sample_id,
            "collected_at": datetime.now().isoformat(timespec="seconds"),
            "label": cand.label,
            "source_type": cand.source_type,
            "prefix": cand.prefix,
            "glass_id": cand.glass_id,
            "image_name": cand.image_name,
            "inference_record_id": str(cand.inference_record_id or ""),
            "image_result_id": str(cand.image_result_id or ""),
            "tile_idx": str(cand.tile_idx) if cand.tile_idx is not None else "",
            "edge_defect_id": str(cand.edge_defect_id) if cand.edge_defect_id is not None else "",
            "crop_path": "",
            "heatmap_path": "",
            "ai_score": f"{cand.ai_score:.4f}",
            "defect_x": "",
            "defect_y": "",
            "ric_judgment": cand.ric_judgment,
            "over_review_category": cand.over_review_category,
            "over_review_note": cand.over_review_note,
            "inference_timestamp": cand.inference_timestamp,
            "status": "",
        }
