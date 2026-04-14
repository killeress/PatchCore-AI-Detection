"""Load manifest.csv → Sample records with binary scratch label."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SCRATCH_LABEL_KEY = "over_surface_scratch"
SCRATCH_BINARY = "scratch"
NOT_SCRATCH_BINARY = "not_scratch"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    crop_path: Path
    label: str              # "scratch" | "not_scratch"
    original_label: str     # 原始 9-way label (true_ng / over_*)
    glass_id: str
    prefix: str
    source_type: str        # "patchcore_tile" | "edge_defect"
    ai_score: float
    defect_x: int
    defect_y: int


def to_binary_label(original_label: str) -> str:
    """Map 9-way label to binary (scratch vs not_scratch)."""
    return SCRATCH_BINARY if original_label == SCRATCH_LABEL_KEY else NOT_SCRATCH_BINARY


def _safe_float(val: str) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _safe_int(val: str) -> int:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def load_samples(manifest_path: Path, datasets_root: Path | None = None) -> list[Sample]:
    """Load samples from manifest.csv.

    Args:
        manifest_path: manifest.csv 路徑
        datasets_root: crop_path 的解析根目錄；預設為 manifest 所在目錄

    Returns:
        List of Sample. Rows with status != "ok" 或 crop 檔不存在者被 skip（log warn）。
    """
    manifest_path = Path(manifest_path)
    if datasets_root is None:
        datasets_root = manifest_path.parent

    samples: list[Sample] = []
    skipped_missing = 0
    skipped_status = 0

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                skipped_status += 1
                continue

            crop_rel = row.get("crop_path", "")
            crop_path = datasets_root / crop_rel
            if not crop_path.exists():
                skipped_missing += 1
                continue

            original = row.get("label", "")
            samples.append(Sample(
                sample_id=row["sample_id"],
                crop_path=crop_path,
                label=to_binary_label(original),
                original_label=original,
                glass_id=row.get("glass_id", ""),
                prefix=row.get("prefix", ""),
                source_type=row.get("source_type", ""),
                ai_score=_safe_float(row.get("ai_score", "")),
                defect_x=_safe_int(row.get("defect_x", "")),
                defect_y=_safe_int(row.get("defect_y", "")),
            ))

    logger.info(
        "Loaded %d samples (skipped: %d missing files, %d non-ok status)",
        len(samples), skipped_missing, skipped_status,
    )
    return samples
