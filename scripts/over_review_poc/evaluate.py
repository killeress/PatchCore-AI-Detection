"""Fold-level training (LogReg + k-NN), threshold search, breakdown metrics."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neighbors import KNeighborsClassifier

from scripts.over_review_poc.dataset import Sample, SCRATCH_BINARY

logger = logging.getLogger(__name__)


@dataclass
class ClassifierResult:
    classifier_name: str
    test_scores: np.ndarray
    realistic_threshold: float
    realistic_scratch_recall: float
    realistic_true_ng_recall: float
    oracle_threshold: float
    oracle_scratch_recall: float
    pr_auc: float
    missed_scratch_test_idx: list[int] = field(default_factory=list)


@dataclass
class FoldResult:
    fold_idx: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    logreg: ClassifierResult
    knn: ClassifierResult
    per_prefix: dict
    per_source_type: dict


def _find_threshold(scores: np.ndarray, is_true_ng: np.ndarray) -> float:
    """Max score among true_ng samples. Any sample with score > this is 'safe to flip'."""
    tn = scores[is_true_ng]
    if len(tn) == 0:
        return float("-inf")
    return float(tn.max())


def _recall_above(scores: np.ndarray, mask: np.ndarray, threshold: float) -> float:
    sel = scores[mask]
    if len(sel) == 0:
        return 0.0
    return float((sel > threshold).mean())


def find_threshold_at_full_recall(
    train_scores: np.ndarray,
    train_is_true_ng: np.ndarray,
    test_scores: np.ndarray,
    test_is_true_ng: np.ndarray,
    test_is_scratch: np.ndarray,
) -> dict:
    """Realistic + Oracle threshold metrics.

    Realistic: threshold = max train_true_ng score → apply to test
    Oracle: threshold = max test_true_ng score → by definition test_true_ng_recall=100%
    """
    realistic_thr = _find_threshold(train_scores, train_is_true_ng)
    oracle_thr = _find_threshold(test_scores, test_is_true_ng)

    realistic_scratch_recall = _recall_above(test_scores, test_is_scratch, realistic_thr)
    # true_ng 在 realistic 下的 recall = 1 - 被誤判 flip OK 的比例
    #   flipped-to-ok 比例 = test true_ng 中 score > threshold 的比例
    realistic_true_ng_miss_rate = _recall_above(test_scores, test_is_true_ng, realistic_thr)
    realistic_true_ng_recall = 1.0 - realistic_true_ng_miss_rate

    oracle_scratch_recall = _recall_above(test_scores, test_is_scratch, oracle_thr)

    return {
        "realistic_threshold": realistic_thr,
        "realistic_scratch_recall": realistic_scratch_recall,
        "realistic_true_ng_recall": realistic_true_ng_recall,
        "oracle_threshold": oracle_thr,
        "oracle_scratch_recall": oracle_scratch_recall,
    }


def _breakdown_by_attr(
    samples: Sequence[Sample],
    test_idx: np.ndarray,
    result: ClassifierResult,
    attr: str,
) -> dict:
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"flagged": 0, "total": 0})
    for local_i, sample_i in enumerate(test_idx):
        s = samples[sample_i]
        if s.label != SCRATCH_BINARY:
            continue
        key = getattr(s, attr)
        groups[key]["total"] += 1
        if result.test_scores[local_i] > result.realistic_threshold:
            groups[key]["flagged"] += 1
    out = {}
    for key, counts in groups.items():
        total = counts["total"]
        out[key] = {
            "scratch_recall": counts["flagged"] / total if total > 0 else 0.0,
            "n_scratch": total,
        }
    return out


def run_fold(
    fold_idx: int,
    embeddings: np.ndarray,
    samples: Sequence[Sample],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
) -> FoldResult:
    """Train LogReg + k-NN on fold; return FoldResult with full metrics and breakdown."""
    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]
    y_train = np.array([1 if samples[i].label == SCRATCH_BINARY else 0 for i in train_idx])
    y_test = np.array([1 if samples[i].label == SCRATCH_BINARY else 0 for i in test_idx])

    train_is_true_ng = np.array([samples[i].original_label == "true_ng" for i in train_idx])
    test_is_true_ng = np.array([samples[i].original_label == "true_ng" for i in test_idx])
    test_is_scratch = y_test.astype(bool)

    # ---- LogReg ----
    logreg = LogisticRegression(
        class_weight="balanced", max_iter=2000, C=1.0, solver="lbfgs", random_state=seed,
    ).fit(X_train, y_train)
    logreg_train_scores = logreg.predict_proba(X_train)[:, 1]
    logreg_test_scores = logreg.predict_proba(X_test)[:, 1]

    # ---- k-NN ----
    knn = KNeighborsClassifier(n_neighbors=7, metric="cosine").fit(X_train, y_train)
    knn_train_scores = knn.predict_proba(X_train)[:, 1]
    knn_test_scores = knn.predict_proba(X_test)[:, 1]

    def _make_result(name: str, train_sc: np.ndarray, test_sc: np.ndarray) -> ClassifierResult:
        thr = find_threshold_at_full_recall(
            train_sc, train_is_true_ng, test_sc, test_is_true_ng, test_is_scratch,
        )
        missed = [
            int(i) for i, s in enumerate(test_sc)
            if test_is_scratch[i] and s <= thr["realistic_threshold"]
        ]
        pr_auc = (float(average_precision_score(y_test, test_sc))
                  if y_test.sum() > 0 else 0.0)
        return ClassifierResult(
            classifier_name=name,
            test_scores=test_sc,
            realistic_threshold=thr["realistic_threshold"],
            realistic_scratch_recall=thr["realistic_scratch_recall"],
            realistic_true_ng_recall=thr["realistic_true_ng_recall"],
            oracle_threshold=thr["oracle_threshold"],
            oracle_scratch_recall=thr["oracle_scratch_recall"],
            pr_auc=pr_auc,
            missed_scratch_test_idx=missed,
        )

    logreg_result = _make_result("logreg", logreg_train_scores, logreg_test_scores)
    knn_result = _make_result("knn", knn_train_scores, knn_test_scores)

    # ---- breakdown (基於 LogReg，因為它是主指標) ----
    per_prefix = _breakdown_by_attr(samples, test_idx, logreg_result, "prefix")
    per_source_type = _breakdown_by_attr(samples, test_idx, logreg_result, "source_type")

    logger.info(
        "Fold %d: LogReg realistic scratch_recall=%.3f (true_ng_recall=%.3f) | "
        "k-NN scratch_recall=%.3f",
        fold_idx, logreg_result.realistic_scratch_recall,
        logreg_result.realistic_true_ng_recall,
        knn_result.realistic_scratch_recall,
    )

    return FoldResult(
        fold_idx=fold_idx, train_idx=train_idx, test_idx=test_idx,
        logreg=logreg_result, knn=knn_result,
        per_prefix=per_prefix, per_source_type=per_source_type,
    )
