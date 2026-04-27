"""Train final deployment LoRA model on full dataset + pickle bundle.

Produces `deployment/scratch_classifier_v1.pkl` consumable by
`scratch_classifier.ScratchClassifier`.

Usage:
    python -m scripts.over_review_poc.train_final_model \
        --manifest datasets/over_review/manifest.csv \
        --transform clahe --clahe-clip 4.0 \
        --rank 16 --n-lora-blocks 2 --epochs 15 --alpha 16 \
        --calib-frac 0.2 \
        --output deployment/scratch_classifier_v1.pkl
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TRUST_REMOTE_CODE", "1")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from scratch_classifier import ScratchClassifierMetadata, save_bundle

from scripts.over_review_poc.dataset import load_samples, SCRATCH_BINARY
from scripts.over_review_poc.features import (
    DINOV2_MODEL, DINOV2_REPO, INPUT_SIZE,
    build_transform_clahe,
)
from scripts.over_review_poc.finetune_lora import (
    _apply_lora, _extract_cls, _train_fold,
)
from scripts.over_review_poc.zero_leak_analysis import _group_aware_split

logger = logging.getLogger(__name__)


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_lora_state_dict(model: torch.nn.Module) -> dict:
    """Return only LoRA adapter weights from the fine-tuned model."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if "lora_A" in k or "lora_B" in k}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--transform", choices=["clahe"], default="clahe")
    p.add_argument("--clahe-clip", type=float, default=4.0)
    p.add_argument("--clahe-tile", type=int, default=8)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--n-lora-blocks", type=int, default=2)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calib-frac", type=float, default=0.2)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--default-safety", type=float, default=1.1,
                   help="Default safety multiplier baked into bundle metadata "
                        "(runtime config may override).")
    p.add_argument("--dinov2-repo", type=Path, default=None,
                   help="Local path to DINOv2 repo dir (offline servers). "
                        "E.g. /root/.cache/torch/hub/facebookresearch_dinov2_main")
    p.add_argument("--dinov2-weights", type=Path, default=None,
                   help="Local .pth for DINOv2 pretrained weights (offline servers).")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_transform_clahe(args.clahe_clip, args.clahe_tile)
    transform_id = f"v3_clahe_cl{args.clahe_clip}_tg{args.clahe_tile}"

    samples = load_samples(args.manifest)
    is_true_ng = np.array([s.original_label == "true_ng" for s in samples])
    y = np.array([1 if s.label == SCRATCH_BINARY else 0 for s in samples])
    all_idx = np.arange(len(samples))
    proper_idx, calib_idx = _group_aware_split(all_idx, samples, args.calib_frac, args.seed)
    logger.info("Total=%d | proper_train=%d | calibration=%d",
                len(samples), len(proper_idx), len(calib_idx))

    # Train LoRA + head on proper_train
    logger.info("Training LoRA (rank=%d, blocks=%d, epochs=%d)...",
                args.rank, args.n_lora_blocks, args.epochs)
    model = _train_fold(samples, proper_idx, transform, device, args)
    proper_samples = [samples[i] for i in proper_idx]
    calib_samples = [samples[i] for i in calib_idx]

    logger.info("Extracting features on proper_train + calib...")
    proper_feats = _extract_cls(model, proper_samples, transform, args.batch_size, device)
    calib_feats = _extract_cls(model, calib_samples, transform, args.batch_size, device)

    logger.info("Fitting LogReg on proper_train features...")
    logreg = LogisticRegression(
        class_weight="balanced", max_iter=2000, C=1.0,
        solver="lbfgs", random_state=args.seed,
    ).fit(proper_feats, y[proper_idx])

    # Conformal threshold = max LogReg score on calib true_ng
    calib_scores = logreg.predict_proba(calib_feats)[:, 1]
    calib_ng_mask = is_true_ng[calib_idx]
    if not calib_ng_mask.any():
        raise RuntimeError("No true_ng samples in calibration split; re-seed or increase calib_frac.")
    conformal_threshold = float(calib_scores[calib_ng_mask].max())

    meta = ScratchClassifierMetadata(
        preprocessing_id=transform_id,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
        input_size=INPUT_SIZE,
        dinov2_repo=DINOV2_REPO,
        dinov2_model=DINOV2_MODEL,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        lora_n_blocks=args.n_lora_blocks,
        conformal_threshold=conformal_threshold,
        safety_multiplier=args.default_safety,
        trained_at=dt.datetime.now().isoformat(),
        dataset_sha256=_sha256_of_file(args.manifest),
        git_commit=_get_git_commit(),
    )
    save_bundle(
        args.output,
        _extract_lora_state_dict(model),
        logreg,
        meta,
        calib_scores,
    )

    print()
    print("=== train_final_model summary ===")
    print(f"Bundle:             {args.output}")
    print(f"Conformal thresh:   {conformal_threshold:.4f}")
    print(f"Default safety:     {args.default_safety}")
    print(f"Est. eff. thresh:   {min(conformal_threshold * args.default_safety, 0.9999):.4f}")
    print(f"Calib NG count:     {int(calib_ng_mask.sum())} / {len(calib_idx)}")
    print(f"Calib score range:  [{calib_scores.min():.3f}, {calib_scores.max():.3f}]")
    return {
        "output_path": str(args.output),
        "total_samples": len(samples),
        "scratch_count": int(y.sum()),
        "conformal_threshold": conformal_threshold,
        "effective_threshold": float(min(conformal_threshold * args.default_safety, 0.9999)),
        "calib_ng_count": int(calib_ng_mask.sum()),
        "calib_total": len(calib_idx),
    }


if __name__ == "__main__":
    result = main()
    sys.exit(0 if isinstance(result, dict) else result)
