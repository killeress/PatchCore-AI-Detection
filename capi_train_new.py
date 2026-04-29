"""新機種 PatchCore 訓練 Wizard 後端 worker。

提供：
- preprocess_panels_to_pool: Step 2 切 tile + 寫 DB
- sample_ng_tiles: 從 over_review 抽 NG
- run_training_pipeline: Step 4 訓 10 模型 + 寫 bundle
"""
from __future__ import annotations
import gc
import os
import json
import logging
import random
import shutil
import time
import traceback
from functools import wraps
from dataclasses import dataclass, field
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
    required_backbones: List[str] = field(
        default_factory=lambda: ["wide_resnet50_2-32ee1156.pth"]
    )

    batch_size: int = 8
    image_size: tuple = (512, 512)
    coreset_ratio: float = 0.1
    max_epochs: int = 1
    # 前 N 片 panel 收 inner+edge tile；超過此索引的 panel 只收 edge tile
    # 預設 3：前 3 片提供 inner（已足夠 ~450 個樣本）；第 4-5 片補 edge
    inner_panels: int = 3


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
        inner_allowed = idx <= cfg.inner_panels
        role_label = "inner+edge" if inner_allowed else "edge only"
        log(f"[{idx}/{len(cfg.panel_paths)}] panel {panel_dir.name} ({role_label})")
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
        skipped_inner = 0
        for lighting, result in results.items():
            for tile in result.tiles:
                if tile.zone == "inner" and not inner_allowed:
                    skipped_inner += 1
                    continue
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
            extra = f"（略過 inner {skipped_inner}）" if skipped_inner else ""
            log(f"  ✓ 切出 {len(tile_records)} tile {extra}".rstrip())

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
    log: Optional[Callable[[str], None]] = None,
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

    if log:
        log(f"{unit_label}: 建立 Folder datamodule")
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

    if log:
        log(f"{unit_label}: 建立 PatchCore model")
    model = Patchcore(coreset_sampling_ratio=cfg.coreset_ratio)
    model.pre_processor = Patchcore.configure_pre_processor(image_size=cfg.image_size)

    engine = Engine(
        max_epochs=cfg.max_epochs,
        default_root_dir=str(run_root),
        callbacks=None,
    )

    if log:
        log(f"{unit_label}: engine.fit 開始")
    engine.fit(datamodule=datamodule, model=model)
    if log:
        log(f"{unit_label}: engine.fit 完成，開始 export")
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


def _setup_offline_env(
    backbone_cache_dir: Path,
    log: Callable,
    required_backbones: Optional[List[str]] = None,
) -> None:
    """Set torch / huggingface offline env vars + verify backbone is cached.

    anomalib's PatchCore uses `timm.create_model('wide_resnet50_2', pretrained=True)`
    which downloads from HuggingFace Hub. We redirect both TORCH_HOME and HF cache
    env vars to deployment/torch_hub_cache/.
    """
    backbone_cache_dir = Path(backbone_cache_dir).resolve()
    hf_cache = backbone_cache_dir / "huggingface"

    _configure_backbone_cache_runtime(backbone_cache_dir)
    _repair_hf_snapshot_symlinks(hf_cache, log)

    required_backbones = required_backbones or ["wide_resnet50_2-32ee1156.pth"]

    # Verify timm wide_resnet50_2 weights are present in HF cache. Older
    # deployments may also stage the raw torch hub checkpoint by filename.
    missing = []
    has_raw_cache = False
    has_hf_cache = False
    for backbone in required_backbones:
        cache_hits = [
            p for p in backbone_cache_dir.rglob(backbone)
            if p.is_file() and p.stat().st_size > 1024 * 1024
        ]
        if cache_hits:
            has_raw_cache = True
            continue

        if backbone.startswith("wide_resnet50_2"):
            timm_dirs = list(hf_cache.glob("models--timm--wide_resnet50_2*"))
            has_hf_cache = any(_has_valid_hf_snapshot_weights(d) for d in timm_dirs)
            if has_hf_cache:
                continue

        missing.append(backbone)

    if missing:
        raise RuntimeError(
            f"backbone 缺檔：未找到 {', '.join(missing)}。\n"
            f"已檢查 cache: {backbone_cache_dir}\n"
            f"請在有網路的開發機執行：\n"
            f"  HF_HOME={backbone_cache_dir} python -c \"import timm; "
            f"timm.create_model('wide_resnet50_2', pretrained=True)\"\n"
            f"然後把整個 {backbone_cache_dir} 目錄 FTP 上傳到 production。"
        )

    if has_raw_cache and not has_hf_cache:
        _enable_timm_old_cache()
    _patch_hf_hub_local_files_only()

    _preflight_timm_backbone(log, cache_dir=hf_cache if has_hf_cache else None)
    log(f"✓ backbone cache 已就緒: {hf_cache}")


def _configure_backbone_cache_runtime(backbone_cache_dir: Path) -> None:
    """Point torch/timm/HF runtime state at the staged offline cache."""
    backbone_cache_dir = Path(backbone_cache_dir).resolve()
    hf_cache = backbone_cache_dir / "huggingface"

    # Redirect both torch hub and HuggingFace cache to deployment dir.
    os.environ["TORCH_HOME"] = str(backbone_cache_dir)
    os.environ["HF_HOME"] = str(backbone_cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hf_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
    os.environ["HF_XET_CACHE"] = str(backbone_cache_dir / "xet")
    # Force offline mode (no network calls during training).
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["TRUST_REMOTE_CODE"] = "1"

    # capi_server imports anomalib for inference before server_config is loaded.
    # If that already imported huggingface_hub, its constants were frozen from the
    # process environment at import time. Patch them so later timm calls still use
    # this staged cache and offline mode.
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HOME = str(backbone_cache_dir)
        hf_constants.HF_HUB_CACHE = str(hf_cache)
        hf_constants.HUGGINGFACE_HUB_CACHE = str(hf_cache)
        hf_constants.HF_XET_CACHE = str(backbone_cache_dir / "xet")
        hf_constants.HF_HUB_OFFLINE = True
    except Exception:
        pass


def _enable_timm_old_cache() -> None:
    """Allow timm to use TORCH_HOME/hub/checkpoints/*.pth fallback weights."""
    os.environ["TIMM_USE_OLD_CACHE"] = "1"
    try:
        import timm.models._builder as timm_builder

        timm_builder._USE_OLD_CACHE = True
    except Exception:
        pass


def _make_local_only_hf_download(download_fn: Callable) -> Callable:
    """Wrap hf_hub_download so timm cannot open an HTTP client during training."""
    if getattr(download_fn, "_capi_forces_local_files_only", False):
        return download_fn

    @wraps(download_fn)
    def _local_only_download(*args, **kwargs):
        kwargs["local_files_only"] = True
        return download_fn(*args, **kwargs)

    _local_only_download._capi_forces_local_files_only = True
    return _local_only_download


def _patch_hf_hub_local_files_only() -> None:
    """Force HF downloads used by timm to resolve only from local cache."""
    try:
        import huggingface_hub

        huggingface_hub.hf_hub_download = _make_local_only_hf_download(
            huggingface_hub.hf_hub_download
        )
    except Exception:
        pass

    try:
        import huggingface_hub.file_download as hf_file_download

        hf_file_download.hf_hub_download = _make_local_only_hf_download(
            hf_file_download.hf_hub_download
        )
    except Exception:
        pass

    try:
        import timm.models._hub as timm_hub

        timm_hub.hf_hub_download = _make_local_only_hf_download(
            timm_hub.hf_hub_download
        )
    except Exception:
        pass


def _has_valid_hf_snapshot_weights(model_dir: Path) -> bool:
    snapshot_root = model_dir / "snapshots"
    if not snapshot_root.exists():
        return False

    for snapshot_file in snapshot_root.rglob("*"):
        if snapshot_file.suffix not in {".safetensors", ".bin", ".pth"}:
            continue
        try:
            if (
                (snapshot_file.is_file() or snapshot_file.is_symlink())
                and snapshot_file.stat().st_size > 1024 * 1024
            ):
                return True
        except OSError:
            continue
    return False


def _repair_hf_snapshot_symlinks(hf_cache: Path, log: Callable) -> None:
    """Repair HF cache snapshots copied by tools that do not preserve symlinks.

    HuggingFace snapshots usually store files as symlinks into `blobs/`. Some
    FTP/copy workflows turn those symlinks into zero-byte regular files. timm
    then falls through to HuggingFace Hub loading and can fail with opaque HTTP
    client errors during every PatchCore unit. Restore empty/broken snapshot
    weight files from the matching large blob when possible.
    """
    if not hf_cache.exists():
        return

    for model_dir in hf_cache.glob("models--timm--*"):
        blob_candidates = sorted(
            [
                p for p in (model_dir / "blobs").glob("*")
                if p.is_file() and p.stat().st_size > 1024 * 1024
            ],
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        if not blob_candidates:
            continue

        for snapshot_file in (model_dir / "snapshots").rglob("*"):
            if snapshot_file.suffix not in {".safetensors", ".bin", ".pth"}:
                continue
            if _is_valid_weight_file(snapshot_file):
                continue

            blob = _select_hf_blob_for_snapshot(snapshot_file, blob_candidates)
            if blob is None:
                log(f"  ! 無法自動修復 HF cache 權重檔: {snapshot_file}")
                continue

            if snapshot_file.is_symlink():
                snapshot_file.unlink()
            snapshot_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(blob, snapshot_file)
            log(f"  ✓ 修復 HF cache 權重檔: {snapshot_file}")


def _is_valid_weight_file(path: Path) -> bool:
    try:
        return (
            (path.is_file() or path.is_symlink())
            and path.stat().st_size > 1024 * 1024
        )
    except OSError:
        return False


def _select_hf_blob_for_snapshot(
    snapshot_file: Path,
    blob_candidates: List[Path],
) -> Optional[Path]:
    if snapshot_file.is_symlink():
        try:
            linked_blob = (snapshot_file.parent / snapshot_file.readlink()).resolve()
            for blob in blob_candidates:
                if blob.resolve() == linked_blob:
                    return blob
        except OSError:
            pass

    if len(blob_candidates) == 1:
        return blob_candidates[0]

    # timm weight repositories normally have one large weight blob. If a cache
    # contains multiple large blobs, use the largest one as a conservative
    # recovery path instead of leaving an empty snapshot pointer in place.
    return blob_candidates[0]


def _preflight_timm_backbone(log: Callable, cache_dir: Optional[Path] = None) -> None:
    """Verify timm can load the PatchCore backbone from local cache only."""
    try:
        import timm

        kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
        timm.create_model(
            "wide_resnet50_2",
            pretrained=True,
            features_only=True,
            exportable=True,
            **kwargs,
        )
    except Exception as exc:
        raise RuntimeError(
            "backbone cache 無法離線載入 wide_resnet50_2。"
            "請確認 huggingface snapshot 內的 model.safetensors 不是 0 byte，"
            "且 blobs 目錄已完整上傳。原始錯誤: "
            f"{exc}"
        ) from exc


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
    gpu_lock=None,
    log: Callable[[str], None] = print,
    cancel_event=None,
) -> Path:
    """執行 10 unit 訓練，輸出 bundle 目錄。

    gpu_lock: 同 process 多 thread 共享 GPU 時用於序列化的 lock；subprocess
        模式下傳 None 即可（VRAM 已透過 set_per_process_memory_fraction 隔離）。
    cancel_event: 任意提供 .is_set() 的物件（threading.Event 或 file-flag wrapper）。
    """
    from contextlib import nullcontext
    gpu_ctx = gpu_lock if gpu_lock is not None else nullcontext()
    # 1. 環境檢查
    _setup_offline_env(cfg.backbone_cache_dir, log, cfg.required_backbones)

    # 路徑格式：<machine_id>-<YYYYMMDD_HHMMSS>。
    # job_id 已存在 manifest.json.trained_with_job_id 與 DB model_bundles.job_id，
    # 不再放入路徑（避免 machine_id 在路徑與 job_id 中重複出現）。
    bundle_dir = cfg.output_root / f"{cfg.machine_id}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    thresholds: Dict[str, Dict[str, float]] = {l: {} for l in LIGHTINGS}
    tiles_per_unit: Dict[str, Dict[str, int]] = {}
    model_files: Dict[str, Dict] = {}
    success_units = 0
    succeeded_units: Set[Tuple[str, str]] = set()
    completed_durations: List[float] = []  # 已完成 unit 的耗時，用來算 ETA
    pipeline_start = time.monotonic()

    def _eta_text() -> str:
        if not completed_durations:
            return ""
        avg = sum(completed_durations) / len(completed_durations)
        remaining_units = len(TRAINING_UNITS) - len(completed_durations)
        if remaining_units <= 0:
            return ""
        eta_s = avg * remaining_units
        m, s = divmod(int(eta_s), 60)
        return f"預估剩 {m}m{s:02d}s（平均 {int(avg)}s/unit）"

    for idx, (lighting, zone) in enumerate(TRAINING_UNITS, 1):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("training cancelled by user")

        unit_label = f"{lighting}-{zone}"
        log(f"[{idx}/10] {unit_label}: 載 tile")
        unit_start = time.monotonic()

        train_tiles = db.list_tile_pool(job_id, lighting=lighting, zone=zone,
                                        source="ok", decision="accept")
        ng_tiles = db.list_tile_pool(job_id, lighting=lighting,
                                     source="ng", decision="accept")

        if len(train_tiles) < MIN_TRAIN_TILES:
            log(f"[{idx}/10] {unit_label}: 跳過：tile 不足 ({len(train_tiles)} < {MIN_TRAIN_TILES})")
            continue

        with gpu_ctx:
            staging = Path(".tmp/training_staging") / job_id / unit_label
            stage_dataset(staging,
                          [Path(t["source_path"]) for t in train_tiles],
                          [Path(t["source_path"]) for t in ng_tiles])
            run_root = Path(".tmp/training_runs") / job_id / unit_label
            try:
                model_pt = train_one_patchcore(staging, run_root, unit_label, cfg, log=log)

                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("training cancelled by user")

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
                unit_elapsed = time.monotonic() - unit_start
                completed_durations.append(unit_elapsed)
                eta = _eta_text()
                log(
                    f"[{idx}/10] {unit_label}: ✓ done | {int(unit_elapsed)}s, "
                    f"threshold={threshold:.4f}, size={size/1e6:.1f}MB"
                    + (f" | {eta}" if eta else "")
                )
            except Exception as e:
                completed_durations.append(time.monotonic() - unit_start)
                log(f"[{idx}/10] {unit_label}: ✗ 訓練失敗: {e}")
                for line in traceback.format_exc().rstrip().splitlines()[-8:]:
                    log(f"  {line}")
                # 不增加 success_units，繼續下一個 unit
            finally:
                shutil.rmtree(run_root, ignore_errors=True)
                shutil.rmtree(staging, ignore_errors=True)
                # 釋放 GPU 記憶體：set_per_process_memory_fraction 切的額度
                # 跨 unit 不會自動回收，前面 unit 的 model/embedding_store cache
                # 累積後會把後面 unit 推進 OOM。每 unit 結束強制 reclaim。
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

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
