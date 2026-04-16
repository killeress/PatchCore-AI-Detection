"""清理 dataset 中 true_ng 標籤但 crop 實際為過曝（近乎全白）的樣本

背景：
  capi_dataset_export.py 舊版不會過濾過曝 crop，導致 true_ng 類別中混入
  大量「邊緣受強光或整體曝光過度、看不到任何缺陷特徵」的樣本，對分類
  訓練沒有幫助。匯出端已加入 is_overexposed_crop 過濾，此腳本用來清理
  過濾上線前已蒐集的殘留樣本。

  注意：這個檢查是直接看 crop 檔本身（原圖可能已不在），所以
  不需要 DB、不需要原圖路徑。

用法：
  python scripts/cleanup_overexposed_true_ng.py --dry-run
  python scripts/cleanup_overexposed_true_ng.py          # 實際刪除

功能：
  1. 讀 datasets/over_review/manifest.csv
  2. 對 label==true_ng 且有 crop_path 的 row：讀 crop 檔、套用
     is_overexposed_crop；命中 → 標記刪除
  3. 刪除 crop / heatmap 實體檔（heatmap 多半為空）後重寫 manifest.csv

路徑設定：
  - 預設讀 server_config.yaml 取 dataset_export.base_dir
  - 也可用 --base-dir 指定
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_dataset_export import (
    TRUE_NG_LABEL,
    is_overexposed_crop,
    read_manifest,
    write_manifest,
)

logger = logging.getLogger("cleanup_overexposed_true_ng")


def load_base_dir(config_path: Path) -> str:
    import yaml
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return (
        cfg.get("dataset_export", {}).get("base_dir")
        or "datasets/over_review"
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Remove true_ng samples whose crop is overexposed (near-white)."
    )
    p.add_argument("--config", default="server_config.yaml", help="server config yaml")
    p.add_argument("--base-dir", help="override dataset base dir")
    p.add_argument("--dry-run", action="store_true", help="僅列出要刪的項目，不實際動檔案")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = Path(args.config)
    if args.base_dir:
        base_dir = args.base_dir
    elif config_path.exists():
        base_dir = load_base_dir(config_path)
    else:
        base_dir = "datasets/over_review"

    base = Path(base_dir).resolve()
    manifest_path = base / "manifest.csv"
    logger.info("Dataset base: %s", base)

    if not manifest_path.exists():
        logger.error("manifest 不存在: %s", manifest_path)
        return 2

    rows = read_manifest(manifest_path)
    if not rows:
        logger.info("manifest 為空，無需清理")
        return 0
    logger.info("manifest 共 %d 筆樣本", len(rows))

    true_ng_rows = [
        (sid, r) for sid, r in rows.items()
        if r.get("label") == TRUE_NG_LABEL and r.get("crop_path")
    ]
    logger.info("其中 true_ng 且有 crop 的樣本: %d 筆", len(true_ng_rows))
    if not true_ng_rows:
        logger.info("沒有可檢查的 true_ng 樣本，結束")
        return 0

    to_delete: Dict[str, str] = {}   # sample_id -> crop_rel
    missing_count = 0
    prefix_drop_counter: Counter = Counter()

    for sid, row in true_ng_rows:
        crop_rel = row["crop_path"]
        crop_full = base / crop_rel
        if not crop_full.exists():
            missing_count += 1
            continue
        img = cv2.imread(str(crop_full), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("無法讀圖，跳過: %s", crop_full)
            continue
        if is_overexposed_crop(img):
            to_delete[sid] = crop_rel
            prefix_drop_counter[row.get("prefix") or ""] += 1

    logger.info("找不到實體 crop 檔 (跳過): %d", missing_count)
    logger.info("共 %d 筆 true_ng 樣本為過曝，將清理", len(to_delete))
    if prefix_drop_counter:
        logger.info("依 prefix 統計待清理數: %s", dict(prefix_drop_counter))

    if not to_delete:
        logger.info("無待清理項目，結束")
        return 0

    removed_files = 0
    for sid, crop_rel in to_delete.items():
        row = rows[sid]
        for rel_key in ("crop_path", "heatmap_path"):
            rel = row.get(rel_key) or ""
            if not rel:
                continue
            full = base / rel
            if not full.exists():
                continue
            if args.dry_run:
                logger.info("[DRY] 會刪檔: %s", full)
            else:
                try:
                    full.unlink()
                    removed_files += 1
                except OSError as e:
                    logger.warning("刪檔失敗 %s: %s", full, e)
        if args.dry_run:
            logger.info("[DRY] 會移除 manifest: %s (%s)", sid, crop_rel)

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
