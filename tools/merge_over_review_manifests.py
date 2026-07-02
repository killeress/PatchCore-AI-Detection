"""Merge over_review manifest.csv from multiple batch folders into one.

Auto-discovers all subdirs under <base> that contain a manifest.csv,
sorted by folder name (YYYYMMDD_HHMMSS order = chronological).

Output: <base>/manifest_merged.csv where crop_path is prefixed with the
batch subdir so load_samples() can resolve files relative to <base>.

Usage:
    # Local
    python -m tools.merge_over_review_manifests

    # Server
    python -m tools.merge_over_review_manifests \
        --base /data/capi_ai/datasets/over_review

    # Exclude specific batches (e.g. bad labeling / process change)
    python -m tools.merge_over_review_manifests \
        --base /data/capi_ai/datasets/over_review \
        --exclude 20260415_104812 legacy_20260414_000000
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Callable, Optional


ProgressCallback = Callable[[str], None]


def discover_batches(base: Path, exclude: set[str]) -> list[str]:
    """Return sorted list of batch dir names that have a manifest.csv."""
    batches = sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and (d / "manifest.csv").exists() and d.name not in exclude
    )
    return batches


def _batch_preview(batches: list[str]) -> str:
    if len(batches) <= 8:
        return str(batches)
    return f"{batches[:5]} ... {batches[-3:]}"


def _should_report(index: int, total: int) -> bool:
    interval = max(1, total // 20)
    return index == 1 or index == total or index % interval == 0


def _emit(progress: Optional[ProgressCallback], message: str) -> None:
    print(message)
    if progress:
        progress(message)


def run(base: Path, exclude: set[str], progress: Optional[ProgressCallback] = None) -> dict:
    """Merge all batch manifest.csv files under *base* into manifest_merged.csv.

    Args:
        base: Root folder containing all batch subdirs.
        exclude: Set of batch dir names to skip.

    Returns:
        dict with keys:
            batches       – list[str]  discovered batch names (after exclusion)
            total_rows    – int        rows written to output
            label_counts  – dict       {label: count}
            out_path      – str        absolute path of manifest_merged.csv

    Raises:
        ValueError: if no batch dirs with manifest.csv are found.
    """
    batches = discover_batches(base, exclude)
    if not batches:
        raise ValueError(f"no batch dirs with manifest.csv found under {base}")

    if exclude:
        _emit(progress, f"excluded: {sorted(exclude)}")
    _emit(progress, f"merging {len(batches)} batches: {_batch_preview(batches)}")

    out_path = base / "manifest_merged.csv"
    tmp_path = out_path.with_name(out_path.name + ".tmp")

    # --- field discovery (union of all CSV headers) ---
    all_fields: list[str] = []
    for idx, b in enumerate(batches, 1):
        if progress and _should_report(idx, len(batches)):
            progress(f"field scan {idx}/{len(batches)}: {b}")
        with open(base / b / "manifest.csv", encoding="utf-8-sig") as f:
            for col in csv.DictReader(f).fieldnames or []:
                if col not in all_fields:
                    all_fields.append(col)
    _emit(progress, f"fields: {all_fields}")

    # --- row merging ---
    seen: set[str] = set()
    labels = Counter()
    total_rows = 0
    dup = 0
    skipped_status = 0
    missing_crop = 0

    try:
        with open(tmp_path, "w", encoding="utf-8-sig", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=all_fields)
            writer.writeheader()

            for idx, b in enumerate(batches, 1):
                batch_rows = 0
                with open(base / b / "manifest.csv", encoding="utf-8-sig", newline="") as f:
                    for r in csv.DictReader(f):
                        if r.get("status", "ok") != "ok":
                            skipped_status += 1
                            continue
                        sid = r["sample_id"]
                        if sid in seen:
                            dup += 1
                            continue
                        seen.add(sid)
                        if r.get("crop_path"):
                            rel = r["crop_path"].replace("\\", "/")
                            r["crop_path"] = f"{b}/{rel}"
                        else:
                            missing_crop += 1
                            continue
                        if r.get("heatmap_path"):
                            rel = r["heatmap_path"].replace("\\", "/")
                            r["heatmap_path"] = f"{b}/{rel}"
                        p = base / r["crop_path"]
                        if not p.exists():
                            missing_crop += 1
                            continue
                        for fn in all_fields:
                            r.setdefault(fn, "")
                        writer.writerow(r)
                        labels[r["label"]] += 1
                        total_rows += 1
                        batch_rows += 1
                if progress and _should_report(idx, len(batches)):
                    progress(
                        f"merged batch {idx}/{len(batches)}: {b}, "
                        f"+{batch_rows} rows, total {total_rows}"
                    )

        tmp_path.replace(out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    _emit(progress, f"status!=ok skipped: {skipped_status}")
    _emit(progress, f"duplicate sample_id skipped: {dup}")
    _emit(progress, f"missing crop file skipped: {missing_crop}")
    _emit(progress, f"final rows: {total_rows}")
    for lab, c in labels.most_common():
        _emit(progress, f"  {lab}: {c}")
    _emit(progress, f"written: {out_path}")

    return {
        "batches": batches,
        "total_rows": total_rows,
        "label_counts": dict(labels),
        "out_path": str(out_path),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=Path("dataset_v2/over_review"),
                   help="Root folder containing all batch subdirs")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="Batch dir names to skip (e.g. batches with wrong labels or process change)")
    args = p.parse_args()

    try:
        run(args.base, set(args.exclude))
    except ValueError as e:
        print(f"[error] {e}")


if __name__ == "__main__":
    main()
