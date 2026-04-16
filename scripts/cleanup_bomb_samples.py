"""清理 dataset 中殘留的炸彈樣本 + 全部 heatmap 檔 (live server 修正腳本)

背景：
  1) capi_dataset_export.py 舊版遺漏了 edge_defect_results.is_bomb 過濾，導致部分
     「炸彈 edge」被當成 over_edge_false_positive 樣本蒐集進 datasets/over_review/。
     原圖 3 天後會被刪除，無法直接重跑 export；改用此腳本對照 DB 現況剔除殘留。
  2) 新政策 heatmap 不再蒐集，需要把 datasets 下所有既存 heatmap/ 目錄裡的檔案
     清掉，並把 manifest.csv 中 heatmap_path 欄位清空。

用法 (在 live server，啟用 venv 後)：
  python scripts/cleanup_bomb_samples.py --dry-run
  python scripts/cleanup_bomb_samples.py          # 實際刪除

功能：
  1. 讀 datasets/over_review/manifest.csv
  2. 依 inference_record_id 分組，呼叫 DB.get_record_detail 取得該筆所有
     tile / edge 的最新 is_bomb 標記
  3. 對每一 manifest row：
       - source_type=patchcore_tile 且對應 tile.is_bomb=1 → 標記刪除
       - source_type=edge_defect   且對應 edge.is_bomb=1  → 標記刪除
       - image_results.is_bomb=1   → 整張影像所有樣本標記刪除
  4. 刪除 crop / heatmap 實體檔後，重寫 manifest.csv

路徑設定：
  - 預設讀 server_config.yaml 取 db_path / dataset_export.base_dir
  - 也可用 --db 與 --base-dir 指定
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

# 允許從 repo root 執行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_database import CAPIDatabase
from capi_dataset_export import read_manifest, write_manifest

logger = logging.getLogger("cleanup_bomb_samples")


def load_config(config_path: Path) -> Tuple[str, str]:
    import yaml
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    db_path = cfg.get("database", {}).get("path") or cfg.get("db_path") or "capi.db"
    base_dir = (
        cfg.get("dataset_export", {}).get("base_dir")
        or "datasets/over_review"
    )
    return db_path, base_dir


def collect_bomb_keys(
    db: CAPIDatabase, record_ids: Set[int]
) -> Tuple[Set[int], Set[Tuple[int, int]], Set[int]]:
    """對給定 record_ids 回傳：
      bomb_image_ids    — image_results.is_bomb=1 的 image_result_id 集合
      bomb_tile_keys    — (image_result_id, tile_id) 集合
      bomb_edge_ids     — edge_defect_results.id 集合
    """
    bomb_image_ids: Set[int] = set()
    bomb_tile_keys: Set[Tuple[int, int]] = set()
    bomb_edge_ids: Set[int] = set()

    for rid in record_ids:
        detail = db.get_record_detail(rid)
        if not detail:
            continue
        for img in detail.get("images") or []:
            img_id = img.get("id")
            if img.get("is_bomb"):
                bomb_image_ids.add(img_id)
            for tile in img.get("tiles") or []:
                if tile.get("is_bomb"):
                    bomb_tile_keys.add((img_id, tile.get("tile_id")))
            for edge in img.get("edge_defects") or []:
                if edge.get("is_bomb"):
                    bomb_edge_ids.add(edge.get("id"))
    return bomb_image_ids, bomb_tile_keys, bomb_edge_ids


def main() -> int:
    p = argparse.ArgumentParser(description="Remove leftover bomb samples from dataset.")
    p.add_argument("--config", default="server_config.yaml", help="server config yaml")
    p.add_argument("--db", help="override db path")
    p.add_argument("--base-dir", help="override dataset base dir (overrides config)")
    p.add_argument("--dry-run", action="store_true", help="僅列出要刪的項目，不實際動檔案")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = Path(args.config)
    if args.db and args.base_dir:
        db_path, base_dir = args.db, args.base_dir
    else:
        cfg_db, cfg_base = load_config(config_path)
        db_path = args.db or cfg_db
        base_dir = args.base_dir or cfg_base

    base = Path(base_dir).resolve()
    manifest_path = base / "manifest.csv"
    logger.info("DB: %s", db_path)
    logger.info("Dataset base: %s", base)

    if not manifest_path.exists():
        logger.error("manifest 不存在: %s", manifest_path)
        return 2

    rows = read_manifest(manifest_path)
    if not rows:
        logger.info("manifest 為空，無需清理")
        return 0
    logger.info("manifest 共 %d 筆樣本", len(rows))

    # 1. 收集所有涉及的 inference_record_id
    record_ids: Set[int] = set()
    for row in rows.values():
        rid = row.get("inference_record_id")
        if rid and str(rid).isdigit():
            record_ids.add(int(rid))
    logger.info("涵蓋 inference_records: %d 筆", len(record_ids))

    # 2. 從 DB 撈出 is_bomb=1 的 image / tile / edge
    db = CAPIDatabase(db_path)
    bomb_image_ids, bomb_tile_keys, bomb_edge_ids = collect_bomb_keys(db, record_ids)
    logger.info(
        "DB 標記為炸彈: images=%d  tiles=%d  edges=%d",
        len(bomb_image_ids), len(bomb_tile_keys), len(bomb_edge_ids),
    )

    # 3. 逐 row 判定
    to_delete: Dict[str, str] = {}  # sample_id -> reason
    for sid, row in rows.items():
        img_id_raw = row.get("image_result_id")
        if not img_id_raw or not str(img_id_raw).isdigit():
            continue
        img_id = int(img_id_raw)

        if img_id in bomb_image_ids:
            to_delete[sid] = "image_is_bomb"
            continue

        src = row.get("source_type")
        if src == "patchcore_tile":
            tile_idx_raw = row.get("tile_idx")
            if tile_idx_raw and str(tile_idx_raw).lstrip("-").isdigit():
                if (img_id, int(tile_idx_raw)) in bomb_tile_keys:
                    to_delete[sid] = "tile_is_bomb"
        elif src == "edge_defect":
            edge_id_raw = row.get("edge_defect_id")
            if edge_id_raw and str(edge_id_raw).isdigit():
                if int(edge_id_raw) in bomb_edge_ids:
                    to_delete[sid] = "edge_is_bomb"

    logger.info("共 %d 筆樣本需清理 (炸彈)", len(to_delete))

    # 4. 刪檔 + 改 manifest (炸彈部份)
    removed_files = 0
    for sid, reason in to_delete.items():
        row = rows[sid]
        for rel_key in ("crop_path", "heatmap_path"):
            rel = row.get(rel_key) or ""
            if not rel:
                continue
            full = base / rel
            if full.exists():
                if args.dry_run:
                    logger.info("[DRY] 會刪檔: %s (%s)", full, reason)
                else:
                    try:
                        full.unlink()
                        removed_files += 1
                    except OSError as e:
                        logger.warning("刪檔失敗 %s: %s", full, e)
        if args.dry_run:
            logger.info("[DRY] 會移除 manifest: %s (%s)", sid, reason)

    # 5. 清除所有殘留 heatmap 檔 + manifest 內 heatmap_path 欄位
    # (新政策：dataset 不再蒐集 heatmap)
    removed_heatmaps = 0
    cleared_heatmap_fields = 0
    for sid, row in rows.items():
        if sid in to_delete:
            continue  # 已在上面一起處理
        rel = row.get("heatmap_path") or ""
        if not rel:
            continue
        full = base / rel
        if full.exists():
            if args.dry_run:
                logger.info("[DRY] 會刪 heatmap: %s", full)
            else:
                try:
                    full.unlink()
                    removed_heatmaps += 1
                except OSError as e:
                    logger.warning("刪 heatmap 失敗 %s: %s", full, e)
        if not args.dry_run:
            row["heatmap_path"] = ""
            cleared_heatmap_fields += 1

    # 6. 清空各 label/prefix/heatmap 空目錄 (選擇性)
    removed_dirs = 0
    if not args.dry_run:
        for hm_dir in base.rglob("heatmap"):
            if hm_dir.is_dir() and not any(hm_dir.iterdir()):
                try:
                    hm_dir.rmdir()
                    removed_dirs += 1
                except OSError:
                    pass

    if args.dry_run:
        logger.info(
            "[DRY-RUN] 完成，無實際變更。炸彈樣本: %d，殘留 heatmap 檔: %d",
            len(to_delete),
            sum(1 for r in rows.values() if (r.get("heatmap_path") or "")
                and (base / r["heatmap_path"]).exists()),
        )
        return 0

    for sid in to_delete:
        rows.pop(sid, None)
    write_manifest(manifest_path, rows)
    logger.info(
        "完成: 炸彈 manifest 項 %d、炸彈檔 %d；heatmap 檔 %d、heatmap 欄位清空 %d；空目錄移除 %d。manifest 重寫於 %s",
        len(to_delete), removed_files, removed_heatmaps, cleared_heatmap_fields,
        removed_dirs, manifest_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
