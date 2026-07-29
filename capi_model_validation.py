"""NG validation-set scoring and report helpers for model bundles."""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional


SUPPORTED_ZONES = {"inner", "edge"}


def bundle_validation_snapshot(detail: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the model/config identity needed to understand a historical run."""
    manifest = detail.get("manifest") or {}
    bundle_path = str(detail.get("bundle_path") or "")
    return {
        "id": int(detail["id"]),
        "machine_id": str(detail.get("machine_id") or ""),
        "bundle_path": bundle_path,
        "label": f"{detail.get('machine_id') or ''} / {Path(bundle_path).name}",
        "trained_at": str(detail.get("trained_at") or ""),
        "is_active": bool(detail.get("is_active")),
        "thresholds": detail.get("thresholds") or {},
        "thresholds_source": str(detail.get("thresholds_source") or ""),
        "preprocess_after_tiling": bool(manifest.get("preprocess_after_tiling", False)),
        "image_preprocess_pipeline": manifest.get("image_preprocess_pipeline") or [],
        "image_preprocess_pipelines": manifest.get("image_preprocess_pipelines") or {},
        "replay_mode": "saved_512_crop",
    }


def _threshold_for_sample(bundle: Dict[str, Any], lighting: str, zone: str) -> float:
    thresholds = bundle.get("thresholds") or {}
    lighting_threshold = thresholds.get(lighting)
    if isinstance(lighting_threshold, dict):
        value = lighting_threshold.get(zone)
    else:
        value = lighting_threshold
    if value is None:
        raise ValueError(f"找不到 threshold: {lighting}-{zone}")
    return float(value)


def _pipeline_for_sample(bundle: Dict[str, Any], zone: str):
    manifest = bundle.get("manifest") or {}
    zone_pipelines = manifest.get("image_preprocess_pipelines") or {}
    if isinstance(zone_pipelines, dict) and zone in zone_pipelines:
        return zone_pipelines.get(zone) or []
    return manifest.get("image_preprocess_pipeline") or []


def score_bundle_sample(
    bundle: Dict[str, Any],
    sample: Dict[str, Any],
    scorer,
    *,
    validation_base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Score one durable NG crop with the matching bundle submodel."""
    lighting = str(sample.get("lighting") or "").strip()
    zone = str(sample.get("zone") or "").strip().lower()
    if zone not in SUPPORTED_ZONES:
        raise ValueError(f"不支援的 zone: {zone or '空白'}")

    crop_path = Path(str(sample.get("crop_path") or "")).resolve()
    if validation_base_dir is not None:
        crop_path.relative_to(Path(validation_base_dir).resolve())
    if not crop_path.is_file():
        raise FileNotFoundError(f"NG crop 不存在: {crop_path}")

    threshold = _threshold_for_sample(bundle, lighting, zone)
    score = scorer.score_image_path(
        bundle_dir=Path(str(bundle.get("bundle_path") or "")),
        lighting=lighting,
        zone=zone,
        image_path=crop_path,
        preprocess_pipeline=_pipeline_for_sample(bundle, zone),
    )
    return {
        "score": round(float(score), 6),
        "threshold": round(float(threshold), 6),
        "caught": bool(float(score) >= float(threshold)),
    }


def build_model_validation_summary(
    results: Iterable[Dict[str, Any]],
    *,
    has_baseline: bool,
) -> Dict[str, Any]:
    rows = list(results)
    summary = {
        "sample_count": len(rows),
        "review_count": len({
            int(row["review_id"])
            for row in rows
            if row.get("review_id") is not None
        }),
        "candidate_evaluable": 0,
        "candidate_caught": 0,
        "candidate_missed": 0,
        "candidate_errors": 0,
        "candidate_caught_rate": None,
        "baseline_evaluable": 0,
        "baseline_caught": 0,
        "baseline_missed": 0,
        "baseline_errors": 0,
        "baseline_caught_rate": None,
        "paired_evaluable": 0,
        "both_caught": 0,
        "candidate_gain": 0,
        "candidate_regression": 0,
        "both_missed": 0,
        "units": {},
    }

    unit_rows: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        unit_label = f"{row.get('lighting') or 'unknown'}-{row.get('zone') or 'unknown'}"
        unit = unit_rows.setdefault(unit_label, {
            "lighting": str(row.get("lighting") or ""),
            "zone": str(row.get("zone") or ""),
            "sample_count": 0,
            "candidate_evaluable": 0,
            "candidate_caught": 0,
            "candidate_errors": 0,
            "baseline_evaluable": 0,
            "baseline_caught": 0,
            "baseline_errors": 0,
            "candidate_gain": 0,
            "candidate_regression": 0,
        })
        unit["sample_count"] += 1

        candidate = row.get("candidate_caught")
        if candidate is None:
            summary["candidate_errors"] += 1
            unit["candidate_errors"] += 1
        else:
            candidate = bool(candidate)
            summary["candidate_evaluable"] += 1
            unit["candidate_evaluable"] += 1
            if candidate:
                summary["candidate_caught"] += 1
                unit["candidate_caught"] += 1
            else:
                summary["candidate_missed"] += 1

        if not has_baseline:
            continue
        baseline = row.get("baseline_caught")
        if baseline is None:
            summary["baseline_errors"] += 1
            unit["baseline_errors"] += 1
        else:
            baseline = bool(baseline)
            summary["baseline_evaluable"] += 1
            unit["baseline_evaluable"] += 1
            if baseline:
                summary["baseline_caught"] += 1
                unit["baseline_caught"] += 1
            else:
                summary["baseline_missed"] += 1

        if candidate is None or baseline is None:
            continue
        summary["paired_evaluable"] += 1
        if candidate and baseline:
            summary["both_caught"] += 1
        elif candidate and not baseline:
            summary["candidate_gain"] += 1
            unit["candidate_gain"] += 1
        elif not candidate and baseline:
            summary["candidate_regression"] += 1
            unit["candidate_regression"] += 1
        else:
            summary["both_missed"] += 1

    if summary["candidate_evaluable"]:
        summary["candidate_caught_rate"] = round(
            summary["candidate_caught"] / summary["candidate_evaluable"], 4
        )
    if has_baseline and summary["baseline_evaluable"]:
        summary["baseline_caught_rate"] = round(
            summary["baseline_caught"] / summary["baseline_evaluable"], 4
        )

    for unit_label in sorted(unit_rows):
        unit = unit_rows[unit_label]
        candidate_total = unit["candidate_evaluable"]
        baseline_total = unit["baseline_evaluable"]
        unit["candidate_caught_rate"] = (
            round(unit["candidate_caught"] / candidate_total, 4)
            if candidate_total else None
        )
        unit["baseline_caught_rate"] = (
            round(unit["baseline_caught"] / baseline_total, 4)
            if has_baseline and baseline_total else None
        )
        summary["units"][unit_label] = unit
    return summary


def classify_model_validation_result(
    result: Dict[str, Any],
    *,
    has_baseline: bool,
) -> str:
    candidate = result.get("candidate_caught")
    baseline = result.get("baseline_caught")
    if candidate is None or (has_baseline and baseline is None):
        return "error"
    if not has_baseline:
        return "caught" if bool(candidate) else "missed"
    if bool(candidate) and bool(baseline):
        return "both_caught"
    if bool(candidate) and not bool(baseline):
        return "gain"
    if not bool(candidate) and bool(baseline):
        return "regression"
    return "both_missed"
