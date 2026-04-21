"""Merge over_review manifest.csv from multiple batch folders into one.

Output: dataset_v2/over_review/manifest_merged.csv where crop_path is
prefixed with the batch subdir so load_samples() can resolve files
relative to dataset_v2/over_review/.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    base = Path("dataset_v2/over_review")
    batches = [
        "legacy_20260414_000000",
        "20260415_104812",
        "20260416_132250",
    ]
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
