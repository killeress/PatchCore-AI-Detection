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


def _link_or_copy(src: Path, dst: Path) -> None:
    """建立 hardlink，跨 filesystem 或不支援時退回 copy2。"""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def stage_dataset(staging_dir: Path, train_paths: List[Path], ng_paths: List[Path]) -> None:
    """為一個 (lighting, zone) unit 準備訓練 staging。

    結構：
      staging_dir/
        train/         (個別 file 的 hardlink/copy)
        test/anormal/  (個別 file)

    為避免 anomalib Folder 對 symlink 行為不一致，用個別檔案 hardlink / copy
    （不是整目錄 mklink）。
    """
    train_dir = staging_dir / "train"
    ng_dir = staging_dir / "test" / "anormal"
    train_dir.mkdir(parents=True, exist_ok=True)
    ng_dir.mkdir(parents=True, exist_ok=True)

    for src in train_paths:
        dst = train_dir / src.name
        _link_or_copy(src, dst)
    for src in ng_paths:
        dst = ng_dir / src.name
        _link_or_copy(src, dst)


def _import_anomalib():
    """延後 import anomalib，方便 unit test monkeypatch。"""
    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    try:
        from anomalib.data.utils import ValSplitMode
        val_mode = ValSplitMode.SAME_AS_TEST
    except ImportError:
        val_mode = "same_as_test"
    return Folder, Patchcore, Engine, ExportType, val_mode


def train_one_patchcore(
    staging_dir: Path,
    run_root: Path,
    unit_label: str,
    cfg: "TrainingConfig" = None,
) -> Path:
    """訓練一個 (lighting, zone) unit。回傳 model.pt 路徑。

    mirrors tools/train_bga_all.py train_one() 的 anomalib 呼叫方式：
    - engine.fit(datamodule=..., model=...)  # 無 model_path
    - engine.export(model=..., export_type=...)  # 無 model_path
    - default_root_dir 控制輸出路徑
    """
    cfg = cfg or TrainingConfig(
        machine_id="?", panel_paths=[], over_review_root=Path("?"),
    )
    Folder, Patchcore, Engine, ExportType, val_mode = _import_anomalib()

    if run_root.exists():
        shutil.rmtree(run_root, ignore_errors=True)
    run_root.mkdir(parents=True, exist_ok=True)

    datamodule = Folder(
        name=f"unit_{unit_label}",
        root=staging_dir,
        normal_dir="train",
        abnormal_dir="test/anormal",
        train_batch_size=cfg.batch_size,
        eval_batch_size=cfg.batch_size,
        num_workers=0,
        val_split_mode=val_mode,
    )
    try:
        datamodule.image_size = cfg.image_size
    except Exception:
        pass

    model = Patchcore(coreset_sampling_ratio=cfg.coreset_ratio)
    model.pre_processor = Patchcore.configure_pre_processor(image_size=cfg.image_size)

    engine = Engine(
        max_epochs=cfg.max_epochs,
        default_root_dir=str(run_root),
        callbacks=None,
    )

    engine.fit(datamodule=datamodule, model=model)
    engine.export(model=model, export_type=ExportType.TORCH)

    candidates = list(run_root.rglob("weights/torch/model.pt"))
    if not candidates:
        candidates = list(run_root.rglob("model.pt"))
    if not candidates:
        raise RuntimeError(f"訓練後找不到 model.pt under {run_root}")
    return candidates[0]


def calibrate_threshold(ng_scores: List[float], train_max_score: float) -> float:
    """取 max(NG P10, train_max × 1.05)。"""
    if not ng_scores:
        return float(train_max_score) * 1.05
    sorted_scores = sorted(ng_scores)
    p10_idx = max(0, int(len(sorted_scores) * 0.10))
    p10 = float(sorted_scores[p10_idx])
    return float(max(p10, train_max_score * 1.05))


def write_manifest(bundle_dir: Path, info: dict) -> None:
    info_full = dict(info)
    info_full["version_schema"] = 1
    (bundle_dir / "manifest.json").write_text(
        json.dumps(info_full, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_thresholds(bundle_dir: Path, thresholds: Dict[str, Dict[str, float]]) -> None:
    (bundle_dir / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_machine_config_yaml(bundle_dir: Path, machine_id: str,
                              thresholds: Dict[str, Dict[str, float]]) -> None:
    """產出 bundle 內的 inference yaml。"""
    import yaml

    model_mapping = {}
    threshold_mapping = {}
    for lighting in LIGHTINGS:
        model_mapping[lighting] = {
            "inner": str(bundle_dir / f"{lighting}-inner.pt"),
            "edge":  str(bundle_dir / f"{lighting}-edge.pt"),
        }
        threshold_mapping[lighting] = thresholds.get(lighting, {"inner": 0.75, "edge": 0.75})

    cfg = {
        "machine_id": machine_id,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "bundle_path": str(bundle_dir),
        "edge_threshold_px": 768,
        "otsu_offset": 5,
        "enable_panel_polygon": True,
        "model_mapping": model_mapping,
        "threshold_mapping": threshold_mapping,
    }
    (bundle_dir / "machine_config.yaml").write_text(
        yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def run_training_pipeline(*args, **kwargs):
    raise NotImplementedError("Phase 4.6")
