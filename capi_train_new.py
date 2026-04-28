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
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Callable, Protocol, runtime_checkable
import cv2

from capi_preprocess import (
    PreprocessConfig, preprocess_panel_folder, PanelPreprocessResult,
)

logger = logging.getLogger("capi.train_new")


@runtime_checkable
class TrainingDB(Protocol):
    """Database interface required by train worker."""
    def insert_tile_pool(self, job_id: str, tiles: List[dict]) -> List[int]: ...
    def list_tile_pool(self, job_id: str, **filters) -> List[dict]: ...

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
    db: TrainingDB,
    thumb_dir: Path,
    log: Callable[[str], None],
) -> dict:
    """將 cfg.panel_paths 全部前處理 + 切 tile + 寫 DB。"""
    thumb_dir.mkdir(parents=True, exist_ok=True)
    (thumb_dir / "tiles").mkdir(parents=True, exist_ok=True)
    (thumb_dir / "thumb").mkdir(parents=True, exist_ok=True)
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
                cv2.imwrite(str(tile_path), tile.image)

                thumb_path = thumb_dir / "thumb" / tile_filename
                thumb = cv2.resize(tile.image, (96, 96))
                cv2.imwrite(str(thumb_path), thumb)

                tile_records.append({
                    "lighting": lighting,
                    "zone": tile.zone,
                    "source": "ok",
                    "source_path": str(tile_path.resolve()),
                    "thumb_path": str(thumb_path.resolve()),
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
    db: TrainingDB,
    thumb_dir: Optional[Path] = None,
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
        records = []
        for i, p in enumerate(chosen):
            thumb_path = p
            if thumb_dir is not None:
                candidate = thumb_dir / "thumb" / "ng" / lighting / f"{job_id}_{lighting}_ng{i:04d}_{p.name}"
                candidate.parent.mkdir(parents=True, exist_ok=True)
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    thumb = cv2.resize(img, (96, 96))
                    cv2.imwrite(str(candidate), thumb)
                    thumb_path = candidate
            records.append({
                "lighting": lighting, "zone": None, "source": "ng",
                "source_path": str(p.resolve()), "thumb_path": str(thumb_path.resolve()),
            })
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
                              thresholds: Dict[str, Dict[str, float]],
                              succeeded_units: Optional[Set[Tuple[str, str]]] = None) -> None:
    """產出 bundle 內的 inference yaml。

    若提供 succeeded_units，只寫入 inner/edge 都成功訓練的 lighting；
    None 表示寫入全部 5×2=10 組（舊行為，測試用）。
    """
    import yaml

    model_mapping = {}
    threshold_mapping = {}
    for lighting in LIGHTINGS:
        if succeeded_units is not None and not all(
            (lighting, zone) in succeeded_units for zone in ("inner", "edge")
        ):
            continue
        unit_paths = {}
        unit_thr = {}
        for zone in ("inner", "edge"):
            if succeeded_units is None or (lighting, zone) in succeeded_units:
                unit_paths[zone] = str(bundle_dir / f"{lighting}-{zone}.pt")
                unit_thr[zone] = thresholds.get(lighting, {}).get(zone, 0.75)
        if unit_paths:
            model_mapping[lighting] = unit_paths
        if unit_thr:
            threshold_mapping[lighting] = unit_thr

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


def _setup_offline_env(backbone_cache_dir: Path, log: Callable) -> None:
    """Set torch / huggingface offline env vars + verify backbone is cached.

    anomalib's PatchCore uses `timm.create_model('wide_resnet50_2', pretrained=True)`
    which downloads from HuggingFace Hub. We redirect both TORCH_HOME and HF cache
    env vars to deployment/torch_hub_cache/.
    """
    backbone_cache_dir = Path(backbone_cache_dir).resolve()

    # Redirect both torch hub and HuggingFace cache to deployment dir
    os.environ["TORCH_HOME"] = str(backbone_cache_dir)
    os.environ["HF_HOME"] = str(backbone_cache_dir)
    os.environ["HF_HUB_CACHE"] = str(backbone_cache_dir / "huggingface")
    # Force offline mode (no network calls during training)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    # Verify timm wide_resnet50_2 weights are present in HF cache
    hf_cache = backbone_cache_dir / "huggingface"
    timm_dirs = list(hf_cache.glob("models--timm--wide_resnet50_2*"))
    if not timm_dirs:
        raise RuntimeError(
            f"backbone 缺檔：未在 {hf_cache} 找到 timm wide_resnet50_2 模型。\n"
            f"請在有網路的開發機執行：\n"
            f"  HF_HOME={backbone_cache_dir} python -c \"import timm; "
            f"timm.create_model('wide_resnet50_2', pretrained=True)\"\n"
            f"然後把整個 {backbone_cache_dir} 目錄 FTP 上傳到 production。"
        )
    log(f"✓ backbone cache 已就緒: {hf_cache}")


def _calibrate_from_model(
    model_pt: Path, train_paths: List[Path], ng_paths: List[Path]
) -> Tuple[float, List[float]]:
    """單次載入模型，回傳 (train_max_score, ng_scores)。"""
    from anomalib.deploy import TorchInferencer
    inferencer = TorchInferencer(path=str(model_pt))

    sample = random.sample(train_paths, min(100, len(train_paths)))
    train_max = 0.0
    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            continue
        result = inferencer.predict(img)
        score = float(getattr(result, "pred_score", 0.0))
        if score > train_max:
            train_max = score

    ng_scores = []
    for p in ng_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        result = inferencer.predict(img)
        ng_scores.append(float(getattr(result, "pred_score", 0.0)))

    return train_max, ng_scores


def run_training_pipeline(
    job_id: str,
    cfg: TrainingConfig,
    db: TrainingDB,
    gpu_lock,
    log: Callable[[str], None] = print,
) -> Path:
    """執行 10 unit 訓練，輸出 bundle 目錄。"""
    # 1. 環境檢查
    _setup_offline_env(cfg.backbone_cache_dir, log)

    bundle_dir = cfg.output_root / (
        f"{cfg.machine_id}-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{job_id}"
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)

    thresholds: Dict[str, Dict[str, float]] = {l: {} for l in LIGHTINGS}
    tiles_per_unit: Dict[str, Dict[str, int]] = {}
    model_files: Dict[str, Dict] = {}
    success_units = 0
    succeeded_units: Set[Tuple[str, str]] = set()

    for idx, (lighting, zone) in enumerate(TRAINING_UNITS, 1):
        unit_label = f"{lighting}-{zone}"
        log(f"[{idx}/10] {unit_label}: 載 tile")

        train_tiles = db.list_tile_pool(job_id, lighting=lighting, zone=zone,
                                        source="ok", decision="accept")
        ng_tiles = db.list_tile_pool(job_id, lighting=lighting,
                                     source="ng", decision="accept")

        if len(train_tiles) < MIN_TRAIN_TILES:
            log(f"[{idx}/10] {unit_label}: 跳過：tile 不足 ({len(train_tiles)} < {MIN_TRAIN_TILES})")
            continue

        with gpu_lock:
            staging = Path(".tmp/training_staging") / job_id / unit_label
            stage_dataset(staging,
                          [Path(t["source_path"]) for t in train_tiles],
                          [Path(t["source_path"]) for t in ng_tiles])
            run_root = Path(".tmp/training_runs") / job_id / unit_label
            try:
                model_pt = train_one_patchcore(staging, run_root, unit_label, cfg)

                train_max, ng_scores = _calibrate_from_model(
                    model_pt,
                    [Path(t["source_path"]) for t in train_tiles],
                    [Path(t["source_path"]) for t in ng_tiles],
                )
                threshold = calibrate_threshold(ng_scores, train_max)

                dst_pt = bundle_dir / f"{unit_label}.pt"
                shutil.copy2(model_pt, dst_pt)
                size = dst_pt.stat().st_size

                thresholds[lighting][zone] = round(threshold, 4)
                tiles_per_unit[unit_label] = {"train": len(train_tiles), "ng": len(ng_tiles)}
                model_files[unit_label] = {"path": dst_pt.name, "size_bytes": size}
                success_units += 1
                succeeded_units.add((lighting, zone))
                log(f"[{idx}/10] {unit_label}: ✓ done, threshold={threshold:.4f}, size={size/1e6:.1f}MB")
            except Exception as e:
                log(f"[{idx}/10] {unit_label}: ✗ 訓練失敗: {e}")
                # 不增加 success_units，繼續下一個 unit
            finally:
                shutil.rmtree(run_root, ignore_errors=True)
                shutil.rmtree(staging, ignore_errors=True)

    if success_units != len(TRAINING_UNITS):
        missing = [
            f"{lighting}-{zone}"
            for lighting, zone in TRAINING_UNITS
            if (lighting, zone) not in succeeded_units
        ]
        shutil.rmtree(bundle_dir, ignore_errors=True)
        raise RuntimeError(
            f"成功 unit 數 {success_units}/{len(TRAINING_UNITS)}，缺少: {', '.join(missing)}"
        )

    # 寫 bundle metadata
    write_thresholds(bundle_dir, thresholds)
    write_machine_config_yaml(bundle_dir, cfg.machine_id, thresholds, succeeded_units=succeeded_units)
    write_manifest(bundle_dir, {
        "machine_id": cfg.machine_id,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "trained_with_job_id": job_id,
        "panel_count": len(cfg.panel_paths),
        "panel_glass_ids": [p.name for p in cfg.panel_paths],
        "edge_threshold_px": 768,
        "patchcore_params": {
            "batch_size": cfg.batch_size,
            "image_size": list(cfg.image_size),
            "coreset_ratio": cfg.coreset_ratio,
            "max_epochs": cfg.max_epochs,
        },
        "tiles_per_unit": tiles_per_unit,
        "model_files": model_files,
        "success_units": success_units,
    })
    return bundle_dir
