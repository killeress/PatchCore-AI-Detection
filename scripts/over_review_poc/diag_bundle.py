"""Quick diagnostic for a scratch_classifier_vX.pkl bundle.

Loads the saved calibration_scores and reconstructs which samples were in the
calibration split (same seed / calib_frac as training) to show score
distributions per label group — no GPU / model reload required.

Usage:
    python -m scripts.over_review_poc.diag_bundle \
        --bundle /root/Code/CAPI_AD/deployment/scratch_classifier_v3.pkl \
        --manifest /data/capi_ai/datasets/over_review/manifest_merged.csv \
        [--calib-frac 0.2] [--seed 42]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from scripts.over_review_poc.dataset import load_samples, SCRATCH_BINARY
from scripts.over_review_poc.zero_leak_analysis import _group_aware_split

logger = logging.getLogger(__name__)


def _percentile_str(arr: np.ndarray) -> str:
    if len(arr) == 0:
        return "(empty)"
    ps = np.percentile(arr, [0, 10, 25, 50, 75, 90, 100])
    return f"min={ps[0]:.3f}  p10={ps[1]:.3f}  p25={ps[2]:.3f}  "  \
           f"median={ps[3]:.3f}  p75={ps[4]:.3f}  p90={ps[5]:.3f}  max={ps[6]:.3f}"


def _hist_str(arr: np.ndarray, bins=10, width=40) -> str:
    if len(arr) == 0:
        return "(empty)"
    counts, edges = np.histogram(arr, bins=bins, range=(0.0, 1.0))
    peak = max(counts) if counts.max() > 0 else 1
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(c / peak * width)
        lines.append(f"  [{edges[i]:.1f}-{edges[i+1]:.1f}] {bar} ({c})")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--calib-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # ── Load bundle (pickle only, no GPU) ────────────────────────────────────
    import pickle
    with open(args.bundle, "rb") as f:
        payload = pickle.load(f)

    calib_scores: np.ndarray = np.asarray(payload["calibration_scores"])
    meta = payload["metadata"]
    conformal_threshold = meta["conformal_threshold"]
    safety = meta["safety_multiplier"]
    eff_thresh = min(conformal_threshold * safety, 0.9999)

    print(f"\n=== Bundle: {args.bundle} ===")
    print(f"  conformal_threshold : {conformal_threshold:.6f}")
    print(f"  safety_multiplier   : {safety}")
    print(f"  eff. threshold      : {eff_thresh:.6f}")
    print(f"  calib_scores shape  : {calib_scores.shape}")

    # ── Reconstruct calib split ───────────────────────────────────────────────
    samples = load_samples(args.manifest)
    all_idx = np.arange(len(samples))
    proper_idx, calib_idx = _group_aware_split(all_idx, samples, args.calib_frac, args.seed)

    if len(calib_idx) != len(calib_scores):
        print(f"\n[WARN] calib_idx length ({len(calib_idx)}) != calib_scores length "
              f"({len(calib_scores)}).")
        print("       Likely --calib-frac or --seed differs from training. "
              "Score-label alignment will be wrong.")
        min_len = min(len(calib_idx), len(calib_scores))
        calib_idx = calib_idx[:min_len]
        calib_scores = calib_scores[:min_len]

    calib_samples = [samples[i] for i in calib_idx]
    labels_bin  = np.array([1 if s.label == SCRATCH_BINARY else 0 for s in calib_samples])
    is_true_ng  = np.array([s.original_label == "true_ng" for s in calib_samples])
    is_scratch  = labels_bin.astype(bool)
    is_not_scratch_non_ng = (~is_scratch) & (~is_true_ng)

    scores_scratch    = calib_scores[is_scratch]
    scores_true_ng    = calib_scores[is_true_ng]
    scores_other_ok   = calib_scores[is_not_scratch_non_ng]

    print(f"\n── Calibration split breakdown ({len(calib_idx)} samples) ──")
    print(f"  scratch (NG→OK target)  : {is_scratch.sum():5d}")
    print(f"  true_ng (must NOT flip) : {is_true_ng.sum():5d}")
    print(f"  other not_scratch       : {is_not_scratch_non_ng.sum():5d}")

    # ── Score distributions ──────────────────────────────────────────────────
    print(f"\n── Score distribution: scratch (label=1, want HIGH score) ──")
    print(f"  {_percentile_str(scores_scratch)}")
    print(_hist_str(scores_scratch))

    print(f"\n── Score distribution: true_ng (MUST stay LOW — never flipped) ──")
    print(f"  {_percentile_str(scores_true_ng)}")
    print(_hist_str(scores_true_ng))

    print(f"\n── Score distribution: other OK (also should stay LOW) ──")
    print(f"  {_percentile_str(scores_other_ok)}")
    print(_hist_str(scores_other_ok))

    # ── Threshold analysis ───────────────────────────────────────────────────
    print(f"\n── Threshold analysis (conformal_threshold = {conformal_threshold:.4f}) ──")
    if is_scratch.sum() > 0:
        scratch_recall_at_conf = float((scores_scratch > conformal_threshold).mean())
        print(f"  scratch recall @ conformal_thresh  : {scratch_recall_at_conf:.3f}")
    if is_true_ng.sum() > 0:
        true_ng_leak_at_conf = float((scores_true_ng > conformal_threshold).mean())
        print(f"  true_ng flip rate @ conformal_thresh: {true_ng_leak_at_conf:.3f}  (must be 0)")

    # ── AUC metrics ─────────────────────────────────────────────────────────
    # Binary: scratch vs all others
    if labels_bin.sum() > 0 and (1 - labels_bin).sum() > 0:
        roc = roc_auc_score(labels_bin, calib_scores)
        ap  = average_precision_score(labels_bin, calib_scores)
        print(f"\n── AUC metrics (scratch=1 vs all) ──")
        print(f"  ROC-AUC : {roc:.4f}")
        print(f"  PR-AUC  : {ap:.4f}")

    # ── Top problematic true_ng samples ──────────────────────────────────────
    if is_true_ng.sum() > 0:
        top_n = min(10, is_true_ng.sum())
        ng_scores = calib_scores[is_true_ng]
        ng_indices = np.where(is_true_ng)[0]
        top_order = np.argsort(ng_scores)[::-1][:top_n]
        print(f"\n── Top {top_n} highest-scored true_ng samples (dangerous — model thinks scratch) ──")
        print(f"  {'score':>7}  {'sample_id':<40}  {'prefix':<10}  {'source_type'}")
        for rank, local_i in enumerate(top_order):
            s = calib_samples[ng_indices[local_i]]
            sc = ng_scores[local_i]
            print(f"  {sc:7.4f}  {s.sample_id:<40}  {s.prefix:<10}  {s.source_type}")

    # ── Recommendation ───────────────────────────────────────────────────────
    print("\n── Recommendation ──")
    if conformal_threshold >= 0.999:
        print("  [!] conformal_threshold=1.0 → model cannot safely flip any sample.")
        print("      Root cause: at least one true_ng sample scored near 1.0.")
        if is_true_ng.sum() > 0:
            p90 = float(np.percentile(scores_true_ng, 90))
            p99 = float(np.percentile(scores_true_ng, 99))
            print(f"      true_ng score p90={p90:.4f}, p99={p99:.4f}")
            if p90 < 0.8:
                print("      => Only a few outlier true_ng samples have high scores.")
                print("         Consider: (1) inspect those samples for label noise,")
                print("                   (2) increase epochs/rank, or")
                print("                   (3) use p99 threshold instead of max.")
            else:
                print("      => Most true_ng samples have high scores — model is not separating classes.")
                print("         Consider: (1) more training data, (2) more LoRA blocks/rank,")
                print("                   (3) check manifest label quality.")
    elif conformal_threshold > 0.7:
        print("  [OK] threshold is high but usable. Monitor scratch recall in production.")
    else:
        print("  [OK] threshold looks healthy.")


if __name__ == "__main__":
    sys.exit(main())
