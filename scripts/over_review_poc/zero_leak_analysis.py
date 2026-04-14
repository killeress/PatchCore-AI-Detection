"""Zero-leak (100% true_ng_recall) threshold analysis using conformal calibration.

Problem with run_poc's `realistic_threshold = max(train true_ng scores)`:
  - LogReg trained with class_weight="balanced" pushes train true_ng scores
    LOW (score distribution is skewed by the training objective).
  - Test true_ng — unseen during training — can score much higher.
  - So `realistic` leaks 3-4% true_ng despite "catching all train NG".

Fix: split-conformal calibration. Hold out a calibration set (subset of the
training fold, disjoint from test) and set threshold = max of calibration true_ng
scores. Because calibration is exchangeable with test, this gives a proper
upper bound with statistical control.

Usage:
    python -m scripts.over_review_poc.zero_leak_analysis \
        --embeddings reports/poc_cl4_clsanom/embeddings_cache.npz \
        --manifest datasets/over_review/manifest.csv \
        --label cl4+max_anom
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from scripts.over_review_poc.dataset import load_samples, SCRATCH_BINARY
from scripts.over_review_poc.splits import group_kfold_stratified

logger = logging.getLogger(__name__)


def _fit_logreg(X_train, y_train, seed):
    return LogisticRegression(
        class_weight="balanced", max_iter=2000, C=1.0,
        solver="lbfgs", random_state=seed,
    ).fit(X_train, y_train)


def _group_aware_split(train_idx, samples, calib_frac, seed):
    """Split `train_idx` into (proper_train_idx, calibration_idx) keeping whole
    glass_ids in each side (no glass leaks across split)."""
    rng = np.random.RandomState(seed)
    glass_to_indices = {}
    for i in train_idx:
        glass_to_indices.setdefault(samples[i].glass_id, []).append(int(i))
    glasses = list(glass_to_indices.keys())
    rng.shuffle(glasses)
    n_calib_target = int(len(train_idx) * calib_frac)
    calib_idx, proper_idx = [], []
    for g in glasses:
        (calib_idx if len(calib_idx) < n_calib_target else proper_idx).extend(glass_to_indices[g])
    return np.array(proper_idx, dtype=int), np.array(calib_idx, dtype=int)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--label", default="")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-frac", type=float, default=0.2,
                        help="Fraction of the training fold held out for "
                             "conformal calibration (default 0.2)")
    parser.add_argument("--safety", type=float, default=1.0,
                        help="Multiplier on calib_max threshold (e.g. 1.05 for 5%% "
                             "extra headroom). >1.0 trades scratch recall for tighter leak.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    data = np.load(args.embeddings, allow_pickle=False)
    X = data["embeddings"]
    samples = load_samples(args.manifest)
    y = np.array([1 if s.label == SCRATCH_BINARY else 0 for s in samples])
    is_true_ng = np.array([s.original_label == "true_ng" for s in samples])
    is_scratch = y.astype(bool)
    folds = group_kfold_stratified(samples, k=args.k, seed=args.seed)

    print(f"\n=== Zero-leak (conformal) analysis ({args.label}) | D={X.shape[1]} ===\n")
    print(f"{'fold':<5} | {'train_max_ng':<14} | {'calib_max_ng':<14} | "
          f"{'test_max_ng':<13} | {'scratch@realistic':<17} | "
          f"{'scratch@conformal':<17} | {'leak@conformal':<14}")
    print("-" * 114)

    realistic_recalls, realistic_leaks = [], []
    conformal_recalls, conformal_leaks = [], []
    oracle_recalls = []
    for fi, (train_idx, test_idx) in enumerate(folds, start=1):
        y_train_full = y[train_idx]
        clf_full = _fit_logreg(X[train_idx], y_train_full, args.seed)
        train_scores_full = clf_full.predict_proba(X[train_idx])[:, 1]
        test_scores_full = clf_full.predict_proba(X[test_idx])[:, 1]
        train_max = float(train_scores_full[is_true_ng[train_idx]].max())
        test_max_ng = float(test_scores_full[is_true_ng[test_idx]].max())

        proper_idx, calib_idx = _group_aware_split(
            train_idx, samples, args.calib_frac, args.seed)
        clf = _fit_logreg(X[proper_idx], y[proper_idx], args.seed)
        calib_scores = clf.predict_proba(X[calib_idx])[:, 1]
        test_scores = clf.predict_proba(X[test_idx])[:, 1]
        calib_ng = is_true_ng[calib_idx]
        calib_max_ng = float(calib_scores[calib_ng].max()) if calib_ng.any() else 0.0

        test_scratch_scores = test_scores[is_scratch[test_idx]]
        test_ng_scores = test_scores[is_true_ng[test_idx]]

        # Realistic: using full-train max (run_poc style), applied to CONFORMAL
        # classifier scores so the comparison is fair
        realistic_thr = float(clf.predict_proba(X[proper_idx])[:, 1][is_true_ng[proper_idx]].max())
        # Conformal: use calibration-set max as the production threshold
        # Score-range safety: clamp below 1.0 since predict_proba is bounded
        conformal_thr = min(calib_max_ng * args.safety, 0.9999)
        # Oracle: perfect-knowledge threshold
        oracle_thr = float(test_ng_scores.max()) if len(test_ng_scores) else 0.0

        realistic_scratch = float((test_scratch_scores > realistic_thr).mean()) \
            if len(test_scratch_scores) else 0.0
        realistic_leak = float((test_ng_scores > realistic_thr).mean()) \
            if len(test_ng_scores) else 0.0
        conformal_scratch = float((test_scratch_scores > conformal_thr).mean()) \
            if len(test_scratch_scores) else 0.0
        conformal_leak = float((test_ng_scores > conformal_thr).mean()) \
            if len(test_ng_scores) else 0.0
        oracle_scratch = float((test_scratch_scores > oracle_thr).mean()) \
            if len(test_scratch_scores) else 0.0

        realistic_recalls.append(realistic_scratch)
        realistic_leaks.append(realistic_leak)
        conformal_recalls.append(conformal_scratch)
        conformal_leaks.append(conformal_leak)
        oracle_recalls.append(oracle_scratch)

        print(f"{fi:<5} | {train_max:<14.4f} | {calib_max_ng:<14.4f} | "
              f"{test_max_ng:<13.4f} | {realistic_scratch:<17.3f} | "
              f"{conformal_scratch:<17.3f} | {conformal_leak:<14.3f}")

    print("-" * 114)
    print(f"Realistic (train-max thr, leaks):")
    print(f"  scratch_recall: {np.mean(realistic_recalls):.3f} ± {np.std(realistic_recalls):.3f}")
    print(f"  true_ng leak:   {np.mean(realistic_leaks):.3f} ± {np.std(realistic_leaks):.3f}")
    print()
    print(f"Conformal (calib-max thr, production-safe):")
    print(f"  scratch_recall: {np.mean(conformal_recalls):.3f} ± {np.std(conformal_recalls):.3f}")
    print(f"  true_ng leak:   {np.mean(conformal_leaks):.3f} ± {np.std(conformal_leaks):.3f}"
          f"  ← 0 = zero production leak")
    print()
    print(f"Oracle (cheating, for ref):")
    print(f"  scratch_recall: {np.mean(oracle_recalls):.3f} ± {np.std(oracle_recalls):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
