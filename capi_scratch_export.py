"""Scratch 誤救負樣本匯出工具。

從 scratch_rescue_review 標記為誤救（is_misrescue=1）的 tile 重新從原始 panel image
re-crop 出 512×512 crop，寫入 timestamped job 目錄 + manifest.csv，供後續 DINOv2 LoRA
hard-negative 再訓練使用。

與 capi_dataset_export 互為姐妹：
- capi_dataset_export 專責 RIC 過檢資料集（多 label，含 true_ng / over_*）
- capi_scratch_export  專責 scratch 分類器誤救樣本（單 label: misrescue_negative）

目錄結構：
    <base_dir>/
        <YYYYMMDD_HHMMSS>/
            misrescue_negative/<prefix>/crop/<filename>.png
            manifest.csv

去重：同一 tile_result_id 只會匯出一次（掃描所有歷史 job 的 manifest.csv 彙總 sample_id）
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from capi_dataset_export import (
    CROP_SIZE,
    build_sample_filename,
    crop_patchcore_tile,
    extract_prefix,
    load_known_sample_ids,
    resolve_source_path,
    write_manifest,
)

logger = logging.getLogger(__name__)

LABEL = "misrescue_negative"
DEFAULT_BASE_DIR = "./datasets/scratch_misrescue"

MANIFEST_FIELDS = [
    "sample_id", "collected_at", "label", "prefix",
    "glass_id", "image_name",
    "inference_record_id", "image_result_id", "tile_result_id",
    "tile_seq", "x", "y", "width", "height",
    "crop_path",
    "scratch_score", "ai_score",
    "reviewed_at", "review_note",
    "inference_timestamp", "ai_judgment", "status",
]


def build_sample_id(glass_id: str, image_name: str, tile_result_id: int) -> str:
    """tile_result_id 為全域唯一鍵，確保跨 job 去重穩定。"""
    stem = Path(image_name).stem
    return f"{glass_id}_{stem}_trid{int(tile_result_id)}"


def export_misrescue_samples(
    db,
    base_dir: str | Path,
    path_mapping: Optional[Dict[str, str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """執行一次增量匯出。回傳 summary dict。

    Args:
        db: CAPIDatabase 實例
        base_dir: 資料集根目錄（下方自動建 timestamped job 目錄）
        path_mapping: UNC → 本地掛載點映射（server config 的 path_mapping）
        start_date / end_date: 可選時段篩選（ISO YYYY-MM-DD）

    Returns:
        {
            "job_dir":        str,            # 本次 job 目錄（無新增樣本時為 ""）
            "exported":       int,            # 成功寫入的樣本數
            "skipped_dup":    int,            # 已存在於舊 job 的樣本數
            "missing_source": int,            # 原圖不存在或讀不到
            "out_of_bounds":  int,            # tile 範圍超出原圖
            "total_marked":   int,            # DB 中符合範圍的已標記總數
        }
    """
    base_dir = Path(base_dir)
    path_mapping = path_mapping or {}

    candidates = db.list_scratch_misrescue_for_export(
        start_date=start_date, end_date=end_date
    )
    total_marked = len(candidates)
    known = load_known_sample_ids(base_dir)

    stats = {
        "exported": 0, "skipped_dup": 0,
        "missing_source": 0, "out_of_bounds": 0,
    }

    pending = []
    for cand in candidates:
        sid = build_sample_id(cand["glass_id"], cand["image_name"], cand["tile_result_id"])
        if sid in known:
            stats["skipped_dup"] += 1
            continue
        pending.append((sid, cand))

    empty_summary = {"job_dir": "", **stats, "total_marked": total_marked}
    if not pending:
        return empty_summary

    job_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: Dict[str, Dict[str, str]] = {}

    image_cache: Dict[str, np.ndarray] = {}
    last_src = None
    for sid, cand in pending:
        row = _row_stub(sid, cand)

        src_raw = cand.get("image_path") or ""
        if not src_raw:
            _mark_fail(row, rows, sid, stats, "missing_source")
            continue

        src_path = resolve_source_path(src_raw, path_mapping)
        src_key = str(src_path)
        if src_key != last_src:
            image_cache.pop(last_src, None)
            last_src = src_key
        img = image_cache.get(src_key)
        if img is None:
            img = cv2.imread(src_key, cv2.IMREAD_UNCHANGED)
            if img is None:
                _mark_fail(row, rows, sid, stats, "missing_source")
                continue
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            image_cache[src_key] = img

        x = int(cand.get("x") or 0)
        y = int(cand.get("y") or 0)
        w = int(cand.get("width") or CROP_SIZE) or CROP_SIZE
        h = int(cand.get("height") or CROP_SIZE) or CROP_SIZE
        H, W = img.shape[:2]
        valid_w = min(W, x + w) - max(0, x)
        valid_h = min(H, y + h) - max(0, y)
        if max(0, valid_w) * max(0, valid_h) < (CROP_SIZE * CROP_SIZE) * 0.25:
            _mark_fail(row, rows, sid, stats, "out_of_bounds")
            continue

        crop = crop_patchcore_tile(img, x, y, w, h)
        prefix = extract_prefix(cand["image_name"])
        filename = build_sample_filename(
            glass_id=cand["glass_id"], image_name=cand["image_name"],
            sample_key=f"trid{cand['tile_result_id']}",
            inference_timestamp=cand.get("created_at") or "",
        )
        crop_rel = f"{LABEL}/{prefix}/crop/{filename}"
        crop_dst = job_dir / crop_rel
        crop_dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_dst), crop)

        row["prefix"] = prefix
        row["crop_path"] = crop_rel
        row["status"] = "ok"
        rows[sid] = row
        stats["exported"] += 1

    if rows:
        write_manifest(job_dir / "manifest.csv", rows, fieldnames=MANIFEST_FIELDS)

    return {
        "job_dir": str(job_dir) if rows else "",
        **stats,
        "total_marked": total_marked,
    }


def _mark_fail(row: dict, rows: dict, sid: str, stats: dict, status: str) -> None:
    row["status"] = status
    rows[sid] = row
    stats[status] += 1


def _row_stub(sample_id: str, cand: dict) -> Dict[str, str]:
    return {
        "sample_id": sample_id,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "label": LABEL,
        "prefix": "",
        "glass_id": cand.get("glass_id", ""),
        "image_name": cand.get("image_name", ""),
        "inference_record_id": str(cand.get("record_id") or ""),
        "image_result_id": str(cand.get("image_result_id") or ""),
        "tile_result_id": str(cand.get("tile_result_id") or ""),
        "tile_seq": str(cand["tile_seq"]) if cand.get("tile_seq") is not None else "",
        "x": str(cand.get("x") or 0),
        "y": str(cand.get("y") or 0),
        "width": str(cand.get("width") or CROP_SIZE),
        "height": str(cand.get("height") or CROP_SIZE),
        "crop_path": "",
        "scratch_score": f"{float(cand.get('scratch_score') or 0.0):.6f}",
        "ai_score": f"{float(cand.get('score') or 0.0):.4f}",
        "reviewed_at": cand.get("reviewed_at") or "",
        "review_note": cand.get("review_note") or "",
        "inference_timestamp": cand.get("created_at") or "",
        "ai_judgment": cand.get("ai_judgment") or "",
        "status": "",
    }
