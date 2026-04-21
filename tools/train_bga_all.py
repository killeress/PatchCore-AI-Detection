"""Train 5 BGA PatchCore models in sequence (one per lighting).

Input  : dataset_v2/BGA/<short>/*.png  (already tiled 512x512)
Anormal: D:/CAPI_3F/my_dataset_capi3F_train_line3_<full>/test/anormal  (borrowed for test)
Output : model/CAPI-BGA-20260417/<full>-model.pt

Uses training_gui_gradio.py defaults:
  batch_size=8, image=512x512, coreset_sampling_ratio=0.1, no tiling.

After each run: keeps only the .pt file and deletes the full anomalib results/ output.
"""

from __future__ import annotations

import os

os.environ.setdefault("TRUST_REMOTE_CODE", "1")

import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_BASE = PROJECT_ROOT / "dataset_v2" / "BGA"
STAGING_BASE = PROJECT_ROOT / "datasets" / "bga_staged"
OUTPUT_BASE = PROJECT_ROOT / "model" / "CAPI-BGA-20260417"
RESULTS_BASE = PROJECT_ROOT / "results" / "bga_training"
ANORMAL_TEMPLATE = r"D:\CAPI_3F\my_dataset_capi3F_train_line3_{full}\test\anormal"

LIGHTING_MAP = [
    ("G0F", "G0F00000"),
    ("R0F", "R0F00000"),
    ("W0F", "W0F00000"),
    ("WGF", "WGF50500"),
    ("STANDARD", "STANDARD"),
]

BATCH_SIZE = 8
IMAGE_SIZE = (512, 512)  # (height, width)
CORESET_RATIO = 0.1


def log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def remove_dir_safely(p: Path) -> None:
    """Remove a junction, symlink, or directory. Never follows junctions into target."""
    if not p.exists() and not p.is_symlink():
        return
    # rmdir removes an empty dir OR a junction (without touching target contents).
    try:
        p.rmdir()
        return
    except OSError:
        pass
    # Real non-empty directory
    shutil.rmtree(p, ignore_errors=True)


def make_junction(link: Path, target: Path) -> None:
    """Create a Windows directory junction (no admin needed)."""
    link.parent.mkdir(parents=True, exist_ok=True)
    remove_dir_safely(link)
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"mklink failed for {link} -> {target}: {result.stdout} {result.stderr}"
        )


def stage_dataset(short: str, full: str) -> Path:
    staging = STAGING_BASE / full
    staging.mkdir(parents=True, exist_ok=True)

    src_train = DATASET_BASE / short
    src_anormal = Path(ANORMAL_TEMPLATE.format(full=full))

    if not src_train.is_dir():
        raise FileNotFoundError(f"Train folder missing: {src_train}")
    if not src_anormal.is_dir():
        raise FileNotFoundError(f"Anormal folder missing: {src_anormal}")

    make_junction(staging / "train", src_train)
    make_junction(staging / "test" / "anormal", src_anormal)
    return staging


def cleanup_staging(staging: Path) -> None:
    remove_dir_safely(staging / "train")
    remove_dir_safely(staging / "test" / "anormal")
    remove_dir_safely(staging / "test")
    try:
        staging.rmdir()
    except OSError:
        pass


def train_one(short: str, full: str) -> Path:
    log(f"===== {full}  (source: BGA/{short}) =====")

    staging = stage_dataset(short, full)
    train_count = sum(1 for _ in (staging / "train").glob("*"))
    anormal_count = sum(1 for _ in (staging / "test" / "anormal").glob("*"))
    log(f"train={train_count}, test/anormal={anormal_count}")

    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    try:
        from anomalib.data.utils import ValSplitMode

        val_split_mode = ValSplitMode.SAME_AS_TEST
    except ImportError:
        val_split_mode = "same_as_test"

    run_root = RESULTS_BASE / full
    if run_root.exists():
        shutil.rmtree(run_root, ignore_errors=True)
    run_root.mkdir(parents=True, exist_ok=True)

    datamodule = Folder(
        name=f"bga_{full}",
        root=staging,
        normal_dir="train",
        abnormal_dir="test/anormal",
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        num_workers=0,
        val_split_mode=val_split_mode,
    )
    try:
        datamodule.image_size = IMAGE_SIZE
    except Exception as e:
        log(f"datamodule.image_size set failed (ignored): {e}")

    model = Patchcore(coreset_sampling_ratio=CORESET_RATIO)
    model.pre_processor = Patchcore.configure_pre_processor(image_size=IMAGE_SIZE)

    engine = Engine(
        max_epochs=1,
        default_root_dir=str(run_root),
        callbacks=None,
    )

    t0 = time.time()
    log("fit...")
    engine.fit(datamodule=datamodule, model=model)

    log("export (torch)...")
    engine.export(model=model, export_type=ExportType.TORCH)

    candidates = list(run_root.rglob("weights/torch/model.pt"))
    if not candidates:
        candidates = list(run_root.rglob("model.pt"))
    if not candidates:
        raise RuntimeError(f"Exported model.pt not found under {run_root}")
    src_pt = candidates[0]

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    dst_pt = OUTPUT_BASE / f"{full}-model.pt"
    shutil.copy2(src_pt, dst_pt)
    elapsed = time.time() - t0
    size_mb = dst_pt.stat().st_size / (1024 * 1024)
    log(f"OK  {dst_pt.name}  ({size_mb:.1f} MB, {elapsed:.1f}s)")

    shutil.rmtree(run_root, ignore_errors=True)
    cleanup_staging(staging)
    return dst_pt


def main() -> int:
    log(f"Project root : {PROJECT_ROOT}")
    log(f"Output dir   : {OUTPUT_BASE}")
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    produced: list[Path] = []
    failures: list[tuple[str, str]] = []
    total_start = time.time()

    for short, full in LIGHTING_MAP:
        try:
            pt = train_one(short, full)
            produced.append(pt)
        except Exception as e:
            log(f"FAIL {full}: {e}")
            traceback.print_exc()
            failures.append((full, str(e)))

    shutil.rmtree(RESULTS_BASE, ignore_errors=True)
    try:
        STAGING_BASE.rmdir()
    except OSError:
        pass

    log("=" * 60)
    log(f"Done in {(time.time() - total_start) / 60:.1f} min")
    log(f"Produced {len(produced)}/{len(LIGHTING_MAP)} models in {OUTPUT_BASE}:")
    for pt in produced:
        log(f"  {pt.name}  ({pt.stat().st_size / (1024 * 1024):.1f} MB)")
    if failures:
        log(f"Failures ({len(failures)}):")
        for full, msg in failures:
            log(f"  {full}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
