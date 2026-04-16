"""清理 dataset 中 true_ng 標籤但 RIC DATASTR 對該 prefix 判 OK 的樣本

背景：
  capi_dataset_export.py 舊版只看 panel 級 result_ric 是否為 NG，
  panel 整體=NG 時就把所有 image 都當 true_ng 蒐集。實際上 RIC 的
  DATASTR 是逐 prefix 判定 (例 W0F00000,OK;WGF50500,OK;G0F00000,NG;3;)，
  W0F、WGF 雖屬同一片 NG panel，但個別其實是 OK，不該被當 NG 樣本。

  匯出端已修正 (capi_dataset_export.py:_flatten_record_to_candidates)，
  此腳本用來清理新規則上線前已蒐集的殘留樣本。

用法 (在 live server，啟用 venv 後)：
  python scripts/cleanup_ric_ok_in_true_ng.py --dry-run
  python scripts/cleanup_ric_ok_in_true_ng.py          # 實際刪除

功能：
  1. 讀 datasets/over_review/manifest.csv
  2. 從 client_accuracy_records 撈每筆 inference_record 對應的 DATASTR
  3. 對 label==true_ng 的 row：解析 DATASTR，若該 row.prefix 不在 DATASTR
     的 NG 清單中 → 標記刪除 (含 OK 與「DATASTR 沒這個 prefix」兩種情況)
  4. DATASTR 為空字串時不刪 (回退舊行為，保守處理)
  5. 刪除 crop / heatmap 實體檔後重寫 manifest.csv

路徑設定：
  - 預設讀 server_config.yaml 取 db_path / dataset_export.base_dir
  - 也可用 --db 與 --base-dir 指定
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_database import CAPIDatabase
from capi_dataset_export import (
    TRUE_NG_LABEL,
    parse_datastr_per_prefix,
    read_manifest,
    write_manifest,
)

logger = logging.getLogger("cleanup_ric_ok_in_true_ng")


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


def build_datastr_map(db: CAPIDatabase) -> Dict[int, str]:
    """回傳 {inference_record_id: datastr}。沒對應 inference_record 的 row 略過。"""
    out: Dict[int, str] = {}
    for rec in db.get_client_accuracy_records():
        rid = rec.get("inference_record_id")
        if rid is None:
            continue
        try:
            rid_int = int(rid)
        except (TypeError, ValueError):
            continue
        # 同一 inference_record_id 可能對應多筆 client record；若 datastr 不一致
        # 取較長/非空者，避免被空字串覆寫
        existing = out.get(rid_int, "")
        candidate = rec.get("datastr") or ""
        if len(candidate) > len(existing):
            out[rid_int] = candidate
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Remove true_ng samples whose RIC DATASTR judged the prefix as OK."
    )
    p.add_argument("--config", default="server_config.yaml", help="server config yaml")
    p.add_argument("--db", help="override db path")
    p.add_argument("--base-dir", help="override dataset base dir")
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

    true_ng_rows = {sid: r for sid, r in rows.items() if r.get("label") == TRUE_NG_LABEL}
    logger.info("其中 true_ng 樣本: %d 筆", len(true_ng_rows))
    if not true_ng_rows:
        logger.info("沒有 true_ng 樣本，無需清理")
        return 0

    db = CAPIDatabase(db_path)
    datastr_map = build_datastr_map(db)
    logger.info("DB DATASTR 對應表: %d 筆 inference_record", len(datastr_map))

    to_delete: Dict[str, str] = {}  # sample_id -> reason
    skipped_no_datastr = 0
    skipped_no_record = 0
    prefix_drop_counter: Counter = Counter()

    for sid, row in true_ng_rows.items():
        rid_raw = row.get("inference_record_id")
        if not rid_raw or not str(rid_raw).lstrip("-").isdigit():
            skipped_no_record += 1
            continue
        rid = int(rid_raw)
        datastr = datastr_map.get(rid, "")
        if not datastr:
            skipped_no_datastr += 1
            continue
        per_prefix = parse_datastr_per_prefix(datastr)
        if not per_prefix:
            skipped_no_datastr += 1
            continue
        prefix = row.get("prefix") or ""
        verdict = per_prefix.get(prefix)
        if verdict == "NG":
            continue  # 真的 NG，留著
        # OK 或 DATASTR 內找不到此 prefix → 不該被當 true_ng
        reason = "ric_ok" if verdict == "OK" else f"prefix_not_in_datastr({verdict or 'missing'})"
        to_delete[sid] = reason
        prefix_drop_counter[prefix] += 1

    logger.info("跳過 (true_ng 但無 inference_record_id): %d", skipped_no_record)
    logger.info("跳過 (true_ng 但 DATASTR 為空，不確定保守保留): %d", skipped_no_datastr)
    logger.info("共 %d 筆 true_ng 樣本需清理", len(to_delete))
    if prefix_drop_counter:
        logger.info("依 prefix 統計待清理數: %s", dict(prefix_drop_counter))

    if not to_delete:
        logger.info("無待清理項目，結束")
        return 0

    removed_files = 0
    for sid, reason in to_delete.items():
        row = rows[sid]
        for rel_key in ("crop_path", "heatmap_path"):
            rel = row.get(rel_key) or ""
            if not rel:
                continue
            full = base / rel
            if not full.exists():
                continue
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

    if args.dry_run:
        logger.info("[DRY-RUN] 完成，無實際變更。預計清理 %d 樣本", len(to_delete))
        return 0

    for sid in to_delete:
        rows.pop(sid, None)
    write_manifest(manifest_path, rows)
    logger.info(
        "完成: 移除 manifest 項 %d、刪檔 %d。manifest 重寫於 %s",
        len(to_delete), removed_files, manifest_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
