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
from scripts.over_review_poc.features import DINOV2_MODEL, get_or_extract
from scripts.over_review_poc.splits import group_kfold_stratified
from scripts.over_review_poc.evaluate import run_fold
from scripts.over_review_poc.report import aggregate

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
    cache_path = args.output / "embeddings_cache.npz"
    embeddings = get_or_extract(samples, cache_path, checkpoint_path=args.checkpoint,
                                batch_size=args.batch_size)
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
