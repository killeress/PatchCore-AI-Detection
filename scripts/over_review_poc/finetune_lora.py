"""LoRA fine-tune DINOv2 last-2 transformer blocks for scratch discrimination.

Goal: break the ~74% Oracle ceiling imposed by frozen ImageNet features.
LoRA (Low-Rank Adaptation) inserts r-rank trainable matrices into the qkv/mlp
projections of the last 2 blocks; all base weights stay frozen. Tiny parameter
count (~0.1M) keeps overfit risk low on 114 scratch samples.

Per fold:
  1. Apply LoRA to a fresh copy of DINOv2 ViT-B/14
  2. Train LoRA + binary head on the fold's training set (BCE + class weight)
  3. Extract CLS features on fold's train+test using fine-tuned model
  4. Fit LogReg on train features, evaluate on test (same convention as evaluate.py)

Usage:
    python -m scripts.over_review_poc.finetune_lora \
        --manifest datasets/over_review/manifest.csv \
        --transform clahe --clahe-clip 4.0 \
        --rank 8 --n-lora-blocks 2 --epochs 10
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ["TRUST_REMOTE_CODE"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

from scripts.over_review_poc.dataset import load_samples, SCRATCH_BINARY, Sample
from scripts.over_review_poc.evaluate import find_threshold_at_full_recall
from scripts.over_review_poc.features import (
    DINOV2_MODEL, DINOV2_REPO, INPUT_SIZE,
    build_transform, build_transform_clahe, build_transform_clahe_tile_only,
    build_transform_otsu_aspect,
)
from scripts.over_review_poc.splits import group_kfold_stratified
from scripts.over_review_poc.zero_leak_analysis import _group_aware_split

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapter: W + scale * B @ A. Only A, B trainable."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        nn.init.zeros_(self.lora_B.weight)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


def _apply_lora(model: nn.Module, n_blocks: int, rank: int, alpha: int) -> int:
    """Replace qkv + mlp.fc1 + mlp.fc2 of the last n_blocks with LoRA variants.

    Returns the count of trainable LoRA params for logging.
    """
    blocks = model.blocks
    targets = blocks[-n_blocks:]
    for block in targets:
        block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha)
        block.mlp.fc1 = LoRALinear(block.mlp.fc1, rank, alpha)
        block.mlp.fc2 = LoRALinear(block.mlp.fc2, rank, alpha)
    # Freeze base; unfreeze LoRA weights (they were set requires_grad=False above
    # because of the nn.Module containing the frozen base, but A/B are independent
    # Linear modules we just created and start with requires_grad=True by default)
    for p in model.parameters():
        p.requires_grad = False
    n_trainable = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            for p in [m.lora_A.weight, m.lora_B.weight]:
                p.requires_grad = True
                n_trainable += p.numel()
    return n_trainable


def _resolve_transform(args):
    if args.transform == "naive":
        return build_transform(), "v1_naive_resize"
    if args.transform == "clahe":
        return build_transform_clahe(args.clahe_clip, args.clahe_tile), \
               f"v3_clahe_cl{args.clahe_clip}_tg{args.clahe_tile}"
    if args.transform == "clahe_tile":
        return build_transform_clahe_tile_only(args.clahe_clip, args.clahe_tile), \
               f"v4_clahe_tile_only_cl{args.clahe_clip}_tg{args.clahe_tile}"
    if args.transform == "otsu":
        return build_transform_otsu_aspect(), "v2_otsu_panel_aspect_pad"
    raise ValueError(args.transform)


class _CropDataset(Dataset):
    def __init__(self, samples: list[Sample], transform, labels: np.ndarray):
        self.samples = samples
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img = Image.open(self.samples[i].crop_path).convert("RGB")
        return self.transform(img), float(self.labels[i])


def _load_dinov2() -> nn.Module:
    return torch.hub.load(DINOV2_REPO, DINOV2_MODEL, source="github")


def _extract_cls(model: nn.Module, samples: list[Sample], transform,
                 batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    ds = _CropDataset(samples, transform, np.zeros(len(samples)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_feats = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            feat = model(x)   # CLS token (B, 768)
            all_feats.append(feat.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def _train_fold(samples, train_idx, transform, device, args):
    model = _load_dinov2()
    n_lora = _apply_lora(model, args.n_lora_blocks, args.rank, args.alpha)
    # Apply LoRA first (adds new submodules), THEN move to device so the
    # freshly-created lora_A / lora_B weights land on CUDA too.
    model = model.to(device)
    head = nn.Linear(768, 1).to(device)

    n_pos = int(sum(1 for i in train_idx
                    if samples[i].label == SCRATCH_BINARY))
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    labels = np.array([1 if samples[i].label == SCRATCH_BINARY else 0
                       for i in train_idx])
    ds = _CropDataset([samples[i] for i in train_idx], transform, labels)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    trainable = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    logger.info("LoRA trainable params: %s (ViT base frozen); head: %s; pos_weight=%.2f",
                f"{n_lora:,}", sum(p.numel() for p in head.parameters()), pos_weight.item())

    for epoch in range(args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feat = model(x)              # (B, 768)
            logits = head(feat).squeeze(-1)
            loss = loss_fn(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        logger.info("  epoch %d/%d  loss=%.4f", epoch + 1, args.epochs,
                    total_loss / max(n_batches, 1))
    return model


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--transform", choices=["naive", "clahe", "clahe_tile", "otsu"],
                   default="clahe")
    p.add_argument("--clahe-clip", type=float, default=4.0)
    p.add_argument("--clahe-tile", type=int, default=8)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--n-lora-blocks", type=int, default=2,
                   help="Apply LoRA to the last N transformer blocks")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("reports/poc_lora"))
    p.add_argument("--calib-frac", type=float, default=0.2,
                   help="Fraction of training fold held out for conformal calibration "
                        "of the deployment threshold (default 0.2)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s [%(levelname)s] %(message)s")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform, transform_id = _resolve_transform(args)
    logger.info("Device: %s | transform: %s", device, transform_id)

    samples = load_samples(args.manifest)
    is_true_ng = np.array([s.original_label == "true_ng" for s in samples])
    is_scratch = np.array([s.label == SCRATCH_BINARY for s in samples])
    y = is_scratch.astype(int)
    folds = group_kfold_stratified(samples, k=args.k, seed=args.seed)

    fold_metrics = []
    for fi, (train_idx, test_idx) in enumerate(folds, start=1):
        logger.info("[Fold %d/%d] training LoRA (r=%d, %d blocks, %d epochs)...",
                    fi, args.k, args.rank, args.n_lora_blocks, args.epochs)
        model = _train_fold(samples, train_idx, transform, device, args)

        # Extract features on full training fold + test fold
        train_samples = [samples[i] for i in train_idx]
        test_samples = [samples[i] for i in test_idx]
        train_feats = _extract_cls(model, train_samples, transform, args.batch_size, device)
        test_feats = _extract_cls(model, test_samples, transform, args.batch_size, device)

        # Fit LogReg on fine-tuned features (same config as evaluate.py)
        logreg = LogisticRegression(class_weight="balanced", max_iter=2000,
                                    C=1.0, solver="lbfgs",
                                    random_state=args.seed).fit(train_feats, y[train_idx])
        train_sc = logreg.predict_proba(train_feats)[:, 1]
        test_sc = logreg.predict_proba(test_feats)[:, 1]
        thr = find_threshold_at_full_recall(
            train_sc, is_true_ng[train_idx],
            test_sc, is_true_ng[test_idx], is_scratch[test_idx],
        )
        pr_auc = float(average_precision_score(y[test_idx], test_sc)) \
            if y[test_idx].sum() else 0.0

        # Conformal calibration for deployment threshold (group-aware split)
        proper_idx, calib_idx = _group_aware_split(
            train_idx, samples, args.calib_frac, args.seed)
        proper_mask = np.isin(train_idx, proper_idx)
        calib_mask = np.isin(train_idx, calib_idx)
        logreg_conf = LogisticRegression(class_weight="balanced", max_iter=2000,
                                         C=1.0, solver="lbfgs",
                                         random_state=args.seed).fit(
            train_feats[proper_mask], y[train_idx][proper_mask])
        calib_scores = logreg_conf.predict_proba(train_feats[calib_mask])[:, 1]
        calib_ng_mask = is_true_ng[train_idx][calib_mask]
        conformal_thr = float(calib_scores[calib_ng_mask].max()) if calib_ng_mask.any() else 1.0
        test_sc_conf = logreg_conf.predict_proba(test_feats)[:, 1]
        test_scratch_sc = test_sc_conf[is_scratch[test_idx]]
        test_ng_sc = test_sc_conf[is_true_ng[test_idx]]
        conformal_scratch = float((test_scratch_sc > conformal_thr).mean()) \
            if len(test_scratch_sc) else 0.0
        conformal_leak = float((test_ng_sc > conformal_thr).mean()) \
            if len(test_ng_sc) else 0.0

        fold_metrics.append({
            "fold": fi,
            "realistic_scratch_recall": thr["realistic_scratch_recall"],
            "realistic_true_ng_recall": thr["realistic_true_ng_recall"],
            "oracle_scratch_recall": thr["oracle_scratch_recall"],
            "pr_auc": pr_auc,
            "conformal_scratch_recall": conformal_scratch,
            "conformal_leak": conformal_leak,
        })
        logger.info("[Fold %d] Realistic=%.3f (ng=%.3f) | Oracle=%.3f | "
                    "Conformal=%.3f (leak=%.3f) | PR-AUC=%.3f",
                    fi, thr["realistic_scratch_recall"], thr["realistic_true_ng_recall"],
                    thr["oracle_scratch_recall"], conformal_scratch, conformal_leak, pr_auc)
        # Free per-fold model memory before next fold
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate
    print("\n=== LoRA fine-tune summary ===")
    print(f"Transform:  {transform_id}")
    print(f"LoRA:       rank={args.rank}, alpha={args.alpha}, blocks={args.n_lora_blocks}, "
          f"epochs={args.epochs}, lr={args.lr}")
    print()
    print(f"{'fold':<5} | {'Realistic':<10} | {'true_ng':<10} | {'Oracle':<10} | "
          f"{'Conformal':<10} | {'Leak':<8} | {'PR-AUC':<8}")
    print("-" * 80)
    for m in fold_metrics:
        print(f"{m['fold']:<5} | "
              f"{m['realistic_scratch_recall']:<10.3f} | "
              f"{m['realistic_true_ng_recall']:<10.3f} | "
              f"{m['oracle_scratch_recall']:<10.3f} | "
              f"{m['conformal_scratch_recall']:<10.3f} | "
              f"{m['conformal_leak']:<8.3f} | "
              f"{m['pr_auc']:<8.3f}")
    for key in ["realistic_scratch_recall", "realistic_true_ng_recall",
                "oracle_scratch_recall",
                "conformal_scratch_recall", "conformal_leak", "pr_auc"]:
        vals = np.array([m[key] for m in fold_metrics])
        print(f"{key:<30} mean = {vals.mean():.3f} ± {vals.std():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
