"""Over-Review Scratch POC CLI entry.

Usage:
    python -m scripts.over_review_poc.run_poc \
        --manifest datasets/over_review/manifest.csv \
        --output reports/over_review_scratch_poc
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from scripts.over_review_poc.dataset import load_samples, SCRATCH_BINARY
from scripts.over_review_poc.features import (
    DINOV2_MODEL,
    get_or_extract,
    build_transform,
    build_transform_clahe,
    build_transform_clahe_tile_only,
    build_transform_otsu_aspect,
)
from scripts.over_review_poc.splits import group_kfold_stratified
from scripts.over_review_poc.evaluate import run_fold
from scripts.over_review_poc.report import aggregate


def _resolve_transform(args):
    """Return (factory, preprocessing_id) for the selected transform + params.

    preprocessing_id encodes both the image transform and the pool_mode so the
    embedding cache stays distinct across every combination.
    """
    if args.transform == "naive":
        base_factory, base_id = build_transform, "v1_naive_resize"
    elif args.transform == "otsu":
        base_factory, base_id = build_transform_otsu_aspect, "v2_otsu_panel_aspect_pad"
    elif args.transform == "clahe":
        clip, tg = args.clahe_clip, args.clahe_tile
        base_factory = lambda: build_transform_clahe(clip, tg)
        base_id = f"v3_clahe_cl{clip}_tg{tg}"
    elif args.transform == "clahe_tile":
        clip, tg = args.clahe_clip, args.clahe_tile
        base_factory = lambda: build_transform_clahe_tile_only(clip, tg)
        base_id = f"v4_clahe_tile_only_cl{clip}_tg{tg}"
    else:
        raise ValueError(f"Unknown transform: {args.transform}")
    # Preserve legacy cache keys: pool_mode='cls' keeps the short base_id form
    # so previously-built caches stay valid. Non-cls modes get a pool suffix.
    if args.pool_mode == "cls":
        return base_factory, base_id
    return base_factory, f"{base_id}__pool-{args.pool_mode}"

logger = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _check_optional_deps() -> None:
    """Fail fast if umap-learn / matplotlib missing, with actionable message."""
    missing = []
    try:
        import umap  # noqa: F401
    except ImportError:
        missing.append(("umap-learn", "pip install umap-learn"))
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append(("matplotlib", "pip install matplotlib"))
    if missing:
        lines = ["Missing optional dependencies:"]
        for pkg, cmd in missing:
            lines.append(f"  - {pkg}  (install with: {cmd})")
        raise SystemExit("\n".join(lines))


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "not_installed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Over-Review Scratch POC")
    parser.add_argument("--manifest", required=True, type=Path,
                        help="datasets/over_review/manifest.csv 路徑")
    parser.add_argument("--output", required=True, type=Path,
                        help="輸出報告目錄（reports/over_review_scratch_poc/）")
    parser.add_argument("--k", type=int, default=5, help="k-fold k 值")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="選填：本地 DINOv2 .pth，預設走 torch.hub")
    parser.add_argument("--transform",
                        choices=["naive", "clahe", "clahe_tile", "otsu"],
                        default="naive",
                        help="Preprocessing variant "
                             "(naive=v1 / clahe=all / clahe_tile=tile-only / otsu)")
    parser.add_argument("--clahe-clip", type=float, default=2.0,
                        help="CLAHE clipLimit (default 2.0; tried 2/3/4)")
    parser.add_argument("--clahe-tile", type=int, default=8,
                        help="CLAHE tileGridSize (default 8 → 8x8 grid)")
    parser.add_argument("--pool-mode",
                        choices=["cls", "max", "cls_max",
                                 "cls_max_anom", "cls_top5_anom"],
                        default="cls",
                        help="Feature pooling over DINOv2 patch tokens. "
                             "cls=baseline (fast), max=channel-wise max pool, "
                             "cls_max=concat CLS+max, cls_max_anom=concat CLS + "
                             "most-anomalous patch, cls_top5_anom=concat CLS + "
                             "mean of top-5 patches by distance from image mean.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _check_optional_deps()
    _set_seeds(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    # ---- Load ----
    logger.info("Loading samples from %s", args.manifest)
    samples = load_samples(args.manifest)
    n_scratch = sum(1 for s in samples if s.label == SCRATCH_BINARY)
    n_true_ng = sum(1 for s in samples if s.original_label == "true_ng")
    logger.info("Total samples: %d (scratch=%d, true_ng=%d, other_over=%d)",
                len(samples), n_scratch, n_true_ng,
                len(samples) - n_scratch - n_true_ng)

    if n_scratch < args.k:
        raise ValueError(f"scratch 樣本 ({n_scratch}) 不足以做 {args.k}-fold")

    # ---- Features ----
    transform_factory, preprocessing_id = _resolve_transform(args)
    logger.info("Preprocessing: %s | pool=%s (id=%s)",
                args.transform, args.pool_mode, preprocessing_id)
    cache_path = args.output / "embeddings_cache.npz"
    embeddings = get_or_extract(
        samples, cache_path,
        transform_factory=transform_factory,
        preprocessing_id=preprocessing_id,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        pool_mode=args.pool_mode,
    )
    assert embeddings.shape[0] == len(samples), "embedding count mismatch"

    # ---- Splits ----
    folds_idx = group_kfold_stratified(samples, k=args.k, seed=args.seed)

    # ---- Train / Evaluate per fold ----
    fold_results = []
    for i, (train_idx, test_idx) in enumerate(folds_idx, start=1):
        fr = run_fold(i, embeddings, samples, train_idx, test_idx, seed=args.seed)
        fold_results.append(fr)

    # ---- Aggregate / Report ----
    metadata = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "manifest_path": str(args.manifest),
        "manifest_sha256": _file_sha256(args.manifest),
        "n_samples": len(samples),
        "n_scratch": n_scratch,
        "n_true_ng": n_true_ng,
        "n_folds": args.k,
        "seed": args.seed,
        "dinov2_model": DINOV2_MODEL,
        "transform": args.transform,
        "pool_mode": args.pool_mode,
        "preprocessing_id": preprocessing_id,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": platform.node(),
        "packages": {
            "torch": _pkg_version("torch"),
            "sklearn": _pkg_version("sklearn"),
            "umap": _pkg_version("umap"),
            "numpy": _pkg_version("numpy"),
        },
    }
    aggregate(fold_results, samples, embeddings, args.output, metadata)

    logger.info("POC complete. Report at %s/report.md", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
