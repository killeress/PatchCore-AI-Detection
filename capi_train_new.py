"""新機種 PatchCore 訓練 Wizard 後端 worker。

提供：
- preprocess_panels_to_pool: Step 2 切 tile + 寫 DB
- sample_ng_tiles: 從 over_review 抽 NG
- run_training_pipeline: Step 4 訓 10 模型 + 寫 bundle
"""
from __future__ import annotations
import os
import json
import logging
import platform
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable
import numpy as np
import cv2

from capi_preprocess import (
    PreprocessConfig, preprocess_panel_folder, PanelPreprocessResult,
)

logger = logging.getLogger("capi.train_new")

LIGHTINGS = ("G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD")
ZONES = ("inner", "edge")
TRAINING_UNITS = [(l, z) for l in LIGHTINGS for z in ZONES]  # 10 個

MIN_TRAIN_TILES = 30
NG_TILES_PER_LIGHTING = 30


@dataclass
class TrainingConfig:
    machine_id: str
    panel_paths: List[Path]
    over_review_root: Path
    output_root: Path = Path("model")
    backbone_cache_dir: Path = Path("deployment/torch_hub_cache")

    batch_size: int = 8
    image_size: tuple = (512, 512)
    coreset_ratio: float = 0.1
    max_epochs: int = 1


def generate_job_id(machine_id: str) -> str:
    return f"train_{machine_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def preprocess_panels_to_pool(
    job_id: str,
    cfg: "TrainingConfig",
    preprocess_cfg: "PreprocessConfig",
    db,
    thumb_dir: Path,
    log: Callable[[str], None],
) -> dict:
    """將 cfg.panel_paths 全部前處理 + 切 tile + 寫 DB。"""
    thumb_dir.mkdir(parents=True, exist_ok=True)
    panel_success = 0
    panel_fail = 0
    total_tiles = 0

    for idx, panel_dir in enumerate(cfg.panel_paths, 1):
        log(f"[{idx}/{len(cfg.panel_paths)}] panel {panel_dir.name}")
        try:
            results = preprocess_panel_folder(panel_dir, preprocess_cfg)
        except Exception as e:
            log(f"  ✗ 處理失敗: {e}")
            panel_fail += 1
            continue
        if not results:
            log(f"  ✗ 無有效 lighting 圖")
            panel_fail += 1
            continue

        polygon_failed_count = sum(1 for r in results.values() if r.polygon_detection_failed)
        if polygon_failed_count > 0:
            log(f"  ⚠ {polygon_failed_count} lighting polygon 偵測失敗")

        # 為每張 tile 存 .png + 縮圖 + 寫 DB
        tile_records = []
        for lighting, result in results.items():
            for tile in result.tiles:
                tile_filename = f"{job_id}_{panel_dir.name}_{lighting}_t{tile.tile_id:04d}.png"
                tile_path = thumb_dir / "tiles" / tile_filename
                tile_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(tile_path), tile.image)

                thumb_path = thumb_dir / "thumb" / tile_filename
                thumb_path.parent.mkdir(parents=True, exist_ok=True)
                thumb = cv2.resize(tile.image, (96, 96))
                cv2.imwrite(str(thumb_path), thumb)

                tile_records.append({
                    "lighting": lighting,
                    "zone": tile.zone,
                    "source": "ok",
                    "source_path": str(tile_path),
                    "thumb_path": str(thumb_path),
                })

        if tile_records:
            db.insert_tile_pool(job_id, tile_records)
            total_tiles += len(tile_records)
            panel_success += 1
            log(f"  ✓ 切出 {len(tile_records)} tile")

    return {
        "panel_success": panel_success,
        "panel_fail": panel_fail,
        "total_tiles": total_tiles,
    }


def sample_ng_tiles(
    job_id: str,
    over_review_root: Path,
    db,
    per_lighting: int = NG_TILES_PER_LIGHTING,
    log: Callable[[str], None] = print,
) -> dict:
    """從 over_review/{*}/true_ng/{lighting}/crop/ 隨機抽 NG tile。"""
    if not over_review_root.exists():
        log(f"⚠ over_review 不存在: {over_review_root}，跳過 NG 抽樣")
        return {"sampled": 0, "missing_lightings": list(LIGHTINGS)}

    sampled = 0
    missing = []
    snapshots = [d for d in over_review_root.iterdir() if d.is_dir() and (d / "true_ng").exists()]

    for lighting in LIGHTINGS:
        all_files = []
        for snap in snapshots:
            crop_dir = snap / "true_ng" / lighting / "crop"
            if crop_dir.exists():
                all_files.extend(crop_dir.glob("*.png"))
        if not all_files:
            missing.append(lighting)
            log(f"⚠ {lighting}: 無 NG 樣本")
            continue
        chosen = random.sample(all_files, min(per_lighting, len(all_files)))
        records = [{
            "lighting": lighting, "zone": None, "source": "ng",
            "source_path": str(p), "thumb_path": str(p),  # NG 用原圖當縮圖
        } for p in chosen]
        db.insert_tile_pool(job_id, records)
        sampled += len(records)
        log(f"  ✓ {lighting}: 抽 {len(chosen)} 個 NG")

    return {"sampled": sampled, "missing_lightings": missing}


def run_training_pipeline(*args, **kwargs):
    raise NotImplementedError("Phase 4.4-4.6")
