"""Compare downstream classifiers on cached embeddings (LogReg, k-NN, RBF-SVM, GBM).

Uses identical Group-k-fold splits as run_poc.py so numbers are directly
comparable across classifiers and across preprocessing variants.

Usage:
    python -m scripts.over_review_poc.compare_classifiers \
        --embeddings reports/over_review_scratch_poc_clahe_cl3/embeddings_cache.npz \
        --manifest datasets/over_review/manifest.csv \
        --label cl3
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from scripts.over_review_poc.dataset import load_samples, SCRATCH_BINARY
from scripts.over_review_poc.evaluate import find_threshold_at_full_recall
from scripts.over_review_poc.splits import group_kfold_stratified

logger = logging.getLogger(__name__)


@dataclass
class ClassifierResult:
    name: str
    realistic_scratch_recall: list[float]
    realistic_true_ng_recall: list[float]
    oracle_scratch_recall: list[float]


def _score(clf, X: np.ndarray) -> np.ndarray:
    """Return scratch-class probability (decision-score-like, higher = more scratch)."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    # RBF-SVM without probability=True: use decision_function
    return clf.decision_function(X)


def _run_fold(clf_factory, X_train, y_train, X_test,
              is_scratch_test, is_true_ng_test, is_true_ng_train):
    clf = clf_factory()
    clf.fit(X_train, y_train)
    train_scores = _score(clf, X_train)
    test_scores = _score(clf, X_test)
    thr = find_threshold_at_full_recall(
        train_scores, is_true_ng_train, test_scores, is_true_ng_test, is_scratch_test,
    )
    return thr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classifier comparison on cached embeddings")
    parser.add_argument("--embeddings", required=True, type=Path,
                        help=".npz cache from run_poc (embeddings_cache.npz)")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--label", default="embeddings",
                        help="Short label for this embedding variant (printed in output)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s [%(levelname)s] %(message)s")

    data = np.load(args.embeddings, allow_pickle=False)
    X = data["embeddings"]
    logger.info("Loaded embeddings: %s  (shape=%s)", args.embeddings, X.shape)

    samples = load_samples(args.manifest)
    assert len(samples) == X.shape[0], "embedding count vs manifest mismatch"

    y = np.array([1 if s.label == SCRATCH_BINARY else 0 for s in samples])
    is_true_ng = np.array([s.original_label == "true_ng" for s in samples])
    is_scratch = y.astype(bool)

    folds = group_kfold_stratified(samples, k=args.k, seed=args.seed)

    classifiers = {
        "LogReg": lambda: LogisticRegression(max_iter=1000, class_weight="balanced",
                                             random_state=args.seed),
        "k-NN(k=5)": lambda: KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "RBF-SVM": lambda: SVC(kernel="rbf", probability=False,
                               class_weight="balanced", random_state=args.seed),
        "GBM": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                  random_state=args.seed),
    }

    # Normalize features for RBF-SVM (others don't need, but harmless)
    # Apply per-fold (fit on train, transform both) inside loop
    results: dict[str, ClassifierResult] = {
        name: ClassifierResult(name=name,
                               realistic_scratch_recall=[],
                               realistic_true_ng_recall=[],
                               oracle_scratch_recall=[])
        for name in classifiers
    }

    for fold_i, (train_idx, test_idx) in enumerate(folds, start=1):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        is_true_ng_train = is_true_ng[train_idx]
        is_true_ng_test = is_true_ng[test_idx]
        is_scratch_test = is_scratch[test_idx]

        for name, factory in classifiers.items():
            out = _run_fold(factory, X_train, y_train, X_test,
                            is_scratch_test, is_true_ng_test, is_true_ng_train)
            results[name].realistic_scratch_recall.append(out["realistic_scratch_recall"])
            results[name].realistic_true_ng_recall.append(out["realistic_true_ng_recall"])
            results[name].oracle_scratch_recall.append(out["oracle_scratch_recall"])

    # ---- Output ----
    print(f"\n=== Classifier comparison ({args.label}) | {X.shape[0]} samples, "
          f"{args.k}-fold CV ===")
    print(f"{'Classifier':<14} | {'Real scratch':<16} | {'Real true_ng':<16} | {'Oracle':<16}")
    print("-" * 78)
    for name, r in results.items():
        rs = np.array(r.realistic_scratch_recall)
        rn = np.array(r.realistic_true_ng_recall)
        osc = np.array(r.oracle_scratch_recall)
        print(f"{name:<14} | {rs.mean():.3f} ± {rs.std():.3f}   "
              f"| {rn.mean():.3f} ± {rn.std():.3f}   "
              f"| {osc.mean():.3f} ± {osc.std():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
