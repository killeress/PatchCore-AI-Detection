"""Stratified Group K-fold split for POC evaluation."""
from __future__ import annotations

import logging
from collections import Counter
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.over_review_poc.dataset import Sample

logger = logging.getLogger(__name__)


def _composite_stratify_key(samples: Sequence[Sample]) -> np.ndarray:
    """label|prefix|source_type 複合 key。"""
    return np.array([f"{s.label}|{s.prefix}|{s.source_type}" for s in samples])


def _coarse_stratify_key(samples: Sequence[Sample]) -> np.ndarray:
    """Fallback：label|source_type（捨棄 prefix）。"""
    return np.array([f"{s.label}|{s.source_type}" for s in samples])


def group_kfold_stratified(
    samples: Sequence[Sample],
    k: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield k folds; each fold is (train_idx, test_idx) ndarray pair.

    - Group by glass_id（同玻璃 tiles 必在同 fold，防洩漏）
    - Stratify by composite key (label, prefix, source_type)
    - 若 composite 某 bin < k，退到 (label, source_type) fallback
    - 若 fallback 仍不夠，raise ValueError
    """
    groups = np.array([s.glass_id for s in samples])
    y = _composite_stratify_key(samples)

    bin_sizes = Counter(y)
    min_bin = min(bin_sizes.values())
    if min_bin < k:
        logger.warning(
            "Composite stratify bin min=%d < k=%d; falling back to (label, source_type)",
            min_bin, k,
        )
        y = _coarse_stratify_key(samples)
        bin_sizes = Counter(y)
        min_bin = min(bin_sizes.values())
        if min_bin < k:
            raise ValueError(
                f"Fallback stratify still has bin smaller than k "
                f"(min_bin={min_bin}, k={k}). Reduce k or rethink stratify. "
                f"Bin sizes: {dict(bin_sizes)}"
            )

    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in sgkf.split(np.zeros(len(samples)), y, groups=groups):
        folds.append((np.asarray(train_idx), np.asarray(test_idx)))
    return folds
