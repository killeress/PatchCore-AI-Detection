"""Report aggregation: UMAP, PR curves, markdown + JSON + missed CSV."""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts.over_review_poc.dataset import Sample, SCRATCH_BINARY
from scripts.over_review_poc.evaluate import FoldResult

logger = logging.getLogger(__name__)


def plot_umap(
    fold_idx: int,
    embeddings: np.ndarray,
    samples: Sequence[Sample],
    out_path: Path,
    seed: int = 42,
) -> None:
    """2D UMAP scatter: scratch=red, true_ng=blue, other_over=gray."""
    import matplotlib.pyplot as plt
    import umap

    reducer = umap.UMAP(n_components=2, random_state=seed)
    emb2d = reducer.fit_transform(embeddings)

    colors: list[str] = []
    for s in samples:
        if s.label == SCRATCH_BINARY:
            colors.append("red")
        elif s.original_label == "true_ng":
            colors.append("blue")
        else:
            colors.append("gray")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb2d[:, 0], emb2d[:, 1], c=colors, s=8, alpha=0.6)
    ax.set_title(f"Fold {fold_idx} UMAP (red=scratch, blue=true_ng, gray=other_over)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_pr(
    fold: FoldResult,
    samples: Sequence[Sample],
    out_path: Path,
) -> None:
    """PR curve (LogReg + k-NN 疊圖) for this fold."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    y_test = np.array([1 if samples[i].label == SCRATCH_BINARY else 0 for i in fold.test_idx])

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, result in [("LogReg", fold.logreg), ("k-NN", fold.knn)]:
        p, r, _ = precision_recall_curve(y_test, result.test_scores)
        ax.plot(r, p, label=f"{name} (AUC={result.pr_auc:.3f})")
    ax.set_xlabel("Recall (scratch)")
    ax.set_ylabel("Precision (scratch)")
    ax.set_title(f"Fold {fold.fold_idx} PR Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _aggregate_classifier_metrics(
    folds: list[FoldResult],
    classifier_key: str,
) -> dict:
    """mean ± std across folds for the named classifier."""
    vals = {
        "realistic_scratch_recall": [],
        "realistic_true_ng_recall": [],
        "oracle_scratch_recall": [],
        "pr_auc": [],
    }
    for f in folds:
        cr = getattr(f, classifier_key)
        vals["realistic_scratch_recall"].append(cr.realistic_scratch_recall)
        vals["realistic_true_ng_recall"].append(cr.realistic_true_ng_recall)
        vals["oracle_scratch_recall"].append(cr.oracle_scratch_recall)
        vals["pr_auc"].append(cr.pr_auc)
    return {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in vals.items()
    }


def _aggregate_breakdown(folds: list[FoldResult], key: str) -> dict:
    """合併各 fold 的 per-prefix / per-source_type breakdown。"""
    pooled: dict[str, dict[str, int]] = defaultdict(lambda: {"flagged": 0, "total": 0})
    for f in folds:
        bd = getattr(f, key)
        for attr, counts in bd.items():
            total = counts["n_scratch"]
            flagged = int(round(counts["scratch_recall"] * total))
            pooled[attr]["total"] += total
            pooled[attr]["flagged"] += flagged
    out = {}
    for attr, c in pooled.items():
        t = c["total"]
        out[attr] = {
            "scratch_recall": c["flagged"] / t if t > 0 else 0.0,
            "n_scratch": t,
        }
    return out


def _verdict(mean_scratch_recall: float) -> str:
    if mean_scratch_recall >= 0.50:
        return "GO"
    if mean_scratch_recall >= 0.30:
        return "部分可行"
    return "NO-GO"


def _write_markdown(
    agg: dict,
    per_prefix: dict,
    per_source_type: dict,
    missed_count: int,
    metadata: dict,
    out_path: Path,
) -> None:
    lr = agg["logreg"]
    kn = agg["knn"]
    verdict = _verdict(lr["realistic_scratch_recall"]["mean"])
    warn = ""
    if lr["realistic_true_ng_recall"]["mean"] < 0.95:
        warn = (f"\n> **⚠️ 警示**: Realistic true_ng_recall 平均 "
                f"{lr['realistic_true_ng_recall']['mean']:.3f} < 0.95\n")

    lines = [
        f"# Over-Review POC: Surface Scratch Classifier",
        f"Date: {metadata.get('run_at', 'N/A')} | "
        f"Model: {metadata.get('dinov2_model', 'N/A')} | "
        f"Folds: {metadata.get('n_folds', 'N/A')}",
        "",
        "## Verdict",
        f"- Realistic scratch_recall (LogReg): "
        f"**{lr['realistic_scratch_recall']['mean']:.1%} ± "
        f"{lr['realistic_scratch_recall']['std']:.1%}** → **{verdict}**",
        f"- Realistic true_ng_recall (LogReg): "
        f"{lr['realistic_true_ng_recall']['mean']:.1%} ± "
        f"{lr['realistic_true_ng_recall']['std']:.1%}",
        warn,
        "## Main Metrics",
        "",
        "| Metric | LogReg | k-NN |",
        "|---|---|---|",
        f"| Realistic scratch_recall | {lr['realistic_scratch_recall']['mean']:.3f} ± "
        f"{lr['realistic_scratch_recall']['std']:.3f} | "
        f"{kn['realistic_scratch_recall']['mean']:.3f} ± "
        f"{kn['realistic_scratch_recall']['std']:.3f} |",
        f"| Realistic true_ng_recall | {lr['realistic_true_ng_recall']['mean']:.3f} ± "
        f"{lr['realistic_true_ng_recall']['std']:.3f} | "
        f"{kn['realistic_true_ng_recall']['mean']:.3f} ± "
        f"{kn['realistic_true_ng_recall']['std']:.3f} |",
        f"| Oracle scratch_recall | {lr['oracle_scratch_recall']['mean']:.3f} ± "
        f"{lr['oracle_scratch_recall']['std']:.3f} | "
        f"{kn['oracle_scratch_recall']['mean']:.3f} ± "
        f"{kn['oracle_scratch_recall']['std']:.3f} |",
        f"| PR-AUC | {lr['pr_auc']['mean']:.3f} ± {lr['pr_auc']['std']:.3f} | "
        f"{kn['pr_auc']['mean']:.3f} ± {kn['pr_auc']['std']:.3f} |",
        "",
        "## Per-Prefix Breakdown (LogReg, pooled over folds)",
        "",
        "| Prefix | scratch_recall | n_scratch |",
        "|---|---|---|",
    ]
    for prefix in sorted(per_prefix.keys()):
        v = per_prefix[prefix]
        lines.append(f"| {prefix} | {v['scratch_recall']:.3f} | {v['n_scratch']} |")

    lines += [
        "",
        "## Per-Source-Type Breakdown (LogReg, pooled over folds)",
        "",
        "| Source Type | scratch_recall | n_scratch |",
        "|---|---|---|",
    ]
    for src in sorted(per_source_type.keys()):
        v = per_source_type[src]
        lines.append(f"| {src} | {v['scratch_recall']:.3f} | {v['n_scratch']} |")

    lines += [
        "",
        f"## Missed Scratch Samples",
        f"總計 {missed_count} 筆 scratch 未被 LogReg realistic threshold 擋下；"
        f"詳見 `missed_scratch.csv`。",
        "",
        "## Appendix",
        "- `fold_{1..5}_umap.png`：DINOv2 embedding UMAP 2D 視覺化",
        "- `fold_{1..5}_pr.png`:各 fold 的 PR curve（LogReg + k-NN）",
        "",
        "## Metadata",
        "```json",
        json.dumps(metadata, indent=2, ensure_ascii=False),
        "```",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_missed_csv(
    folds: list[FoldResult],
    samples: Sequence[Sample],
    out_path: Path,
) -> int:
    """Export missed scratch samples (LogReg realistic threshold 沒擋下的) to CSV."""
    rows = []
    for f in folds:
        for local_i in f.logreg.missed_scratch_test_idx:
            sample_i = f.test_idx[local_i]
            s = samples[sample_i]
            rows.append({
                "fold_idx": f.fold_idx,
                "sample_id": s.sample_id,
                "glass_id": s.glass_id,
                "prefix": s.prefix,
                "source_type": s.source_type,
                "logreg_score": float(f.logreg.test_scores[local_i]),
                "realistic_threshold": float(f.logreg.realistic_threshold),
                "crop_path": str(s.crop_path),
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        if rows:
            w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        else:
            fp.write("fold_idx,sample_id,glass_id,prefix,source_type,"
                     "logreg_score,realistic_threshold,crop_path\n")
    return len(rows)


def aggregate(
    folds: list[FoldResult],
    samples: Sequence[Sample],
    embeddings: np.ndarray,
    out_dir: Path,
    metadata: dict,
) -> None:
    """Full report pipeline: UMAP + PR per fold → markdown + JSON + missed CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in folds:
        # Per-fold UMAP 用該 fold test set 的 embeddings
        test_embs = embeddings[f.test_idx]
        test_samples = [samples[i] for i in f.test_idx]
        plot_umap(f.fold_idx, test_embs, test_samples,
                  out_dir / f"fold_{f.fold_idx}_umap.png")
        plot_pr(f, samples, out_dir / f"fold_{f.fold_idx}_pr.png")

    agg = {
        "logreg": _aggregate_classifier_metrics(folds, "logreg"),
        "knn": _aggregate_classifier_metrics(folds, "knn"),
    }
    per_prefix = _aggregate_breakdown(folds, "per_prefix")
    per_source_type = _aggregate_breakdown(folds, "per_source_type")

    missed_count = _write_missed_csv(folds, samples, out_dir / "missed_scratch.csv")

    _write_markdown(agg, per_prefix, per_source_type,
                    missed_count, metadata, out_dir / "report.md")

    json_payload = {
        "metadata": metadata,
        "aggregate": agg,
        "per_prefix": per_prefix,
        "per_source_type": per_source_type,
        "missed_count": missed_count,
        "per_fold": [
            {
                "fold_idx": f.fold_idx,
                "n_train": int(len(f.train_idx)),
                "n_test": int(len(f.test_idx)),
                "logreg": {
                    "realistic_threshold": f.logreg.realistic_threshold,
                    "realistic_scratch_recall": f.logreg.realistic_scratch_recall,
                    "realistic_true_ng_recall": f.logreg.realistic_true_ng_recall,
                    "oracle_scratch_recall": f.logreg.oracle_scratch_recall,
                    "pr_auc": f.logreg.pr_auc,
                },
                "knn": {
                    "realistic_threshold": f.knn.realistic_threshold,
                    "realistic_scratch_recall": f.knn.realistic_scratch_recall,
                    "realistic_true_ng_recall": f.knn.realistic_true_ng_recall,
                    "oracle_scratch_recall": f.knn.oracle_scratch_recall,
                    "pr_auc": f.knn.pr_auc,
                },
            }
            for f in folds
        ],
    }
    (out_dir / "report.json").write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("Report written to %s", out_dir)
