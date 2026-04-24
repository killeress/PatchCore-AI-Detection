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


def discover_batches(base: Path, exclude: set[str]) -> list[str]:
    """Return sorted list of batch dir names that have a manifest.csv."""
    batches = sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and (d / "manifest.csv").exists() and d.name not in exclude
    )
    return batches


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=Path("dataset_v2/over_review"),
                   help="Root folder containing all batch subdirs")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="Batch dir names to skip (e.g. batches with wrong labels or process change)")
    args = p.parse_args()

    base = args.base
    exclude = set(args.exclude)
    batches = discover_batches(base, exclude)
    if not batches:
        print(f"[error] no batch dirs with manifest.csv found under {base}")
        return
    if exclude:
        print(f"excluded: {sorted(exclude)}")
    print(f"merging {len(batches)} batches: {batches}")
    out_path = base / "manifest_merged.csv"

    all_fields: list[str] = []
    for b in batches:
        with open(base / b / "manifest.csv", encoding="utf-8-sig") as f:
            for col in csv.DictReader(f).fieldnames or []:
                if col not in all_fields:
                    all_fields.append(col)
    print("fields:", all_fields)

    rows: list[dict] = []
    seen: dict[str, str] = {}
    dup = 0
    skipped_status = 0
    missing_crop = 0

    for b in batches:
        with open(base / b / "manifest.csv", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("status", "ok") != "ok":
                    skipped_status += 1
                    continue
                sid = r["sample_id"]
                if sid in seen:
                    dup += 1
                    continue
                seen[sid] = b
                if r.get("crop_path"):
                    rel = r["crop_path"].replace("\\", "/")
                    r["crop_path"] = f"{b}/{rel}"
                if r.get("heatmap_path"):
                    rel = r["heatmap_path"].replace("\\", "/")
                    r["heatmap_path"] = f"{b}/{rel}"
                p = base / r["crop_path"]
                if not p.exists():
                    missing_crop += 1
                    continue
                for fn in all_fields:
                    r.setdefault(fn, "")
                rows.append(r)

    print(f"status!=ok skipped: {skipped_status}")
    print(f"duplicate sample_id skipped: {dup}")
    print(f"missing crop file skipped: {missing_crop}")
    print(f"final rows: {len(rows)}")
    labels = Counter(r["label"] for r in rows)
    for lab, c in labels.most_common():
        print(f"  {lab}: {c}")

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        w.writerows(rows)
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
