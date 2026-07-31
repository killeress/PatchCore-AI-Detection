"""Build and regression-test reviewed MARK calibration profiles."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np

from capi_image_orientation import read_detection_image
from capi_mark_detector import detect_panel_mark, normalize_mark_profile


def mark_sample_set_sha256(samples: Iterable[Dict[str, Any]]) -> str:
    """Return a stable digest for the exact reviewed sample set."""
    canonical = []
    for sample in sorted(samples, key=lambda item: int(item.get("id") or 0)):
        canonical.append(
            {
                "id": int(sample.get("id") or 0),
                "file_sha256": str(sample.get("file_sha256") or "").lower(),
                "expected_text": str(sample.get("expected_text") or ""),
                "rotation_applied": bool(sample.get("rotation_applied")),
                "prototypes": sample.get("prototypes") or [],
            }
        )
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_mark_profile(samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a deterministic multi-prototype profile from all reviewed samples."""
    prototypes: List[Dict[str, Any]] = []
    for sample in sorted(samples, key=lambda item: int(item.get("id") or 0)):
        sample_id = int(sample.get("id") or 0)
        for item in sample.get("prototypes") or []:
            prototype = {
                "char": item.get("char"),
                "position": item.get("position"),
                "densities": item.get("densities"),
                "sample_id": sample_id,
            }
            for existing in prototypes:
                if (
                    existing["position"] == prototype["position"]
                    and existing["char"] != prototype["char"]
                ):
                    left = np.asarray(existing["densities"], dtype=np.float32)
                    right = np.asarray(prototype["densities"], dtype=np.float32)
                    similarity = 1.0 - float(np.mean(np.abs(left - right)))
                    if similarity >= 0.999:
                        raise ValueError(
                            "MARK 標註衝突："
                            f"樣本 {existing['sample_id']} 與 {sample_id} "
                            "的字元特徵相同但答案不同"
                        )
            prototypes.append(prototype)

    normalized = normalize_mark_profile(
        {"schema_version": 1, "prototypes": prototypes}
    )
    return {
        "schema_version": 1,
        "prototypes": [
            {
                "char": item["char"],
                "position": item["position"],
                "densities": [list(row) for row in item["densities"]],
                "sample_id": item["sample_id"],
            }
            for item in normalized["prototypes"]
        ],
    }


def run_mark_profile_regression(
    samples: Iterable[Dict[str, Any]],
    profile: Dict[str, Any],
    *,
    profile_id: int,
) -> Dict[str, Any]:
    """Replay every labeled source image; any missing or wrong case fails the gate."""
    started = time.perf_counter()
    failures = []
    passed = 0
    sample_list = sorted(samples, key=lambda item: int(item.get("id") or 0))

    for sample in sample_list:
        sample_id = int(sample.get("id") or 0)
        image_path = Path(str(sample.get("image_path") or ""))
        expected_text = str(sample.get("expected_text") or "")
        failure: Dict[str, Any] = {
            "sample_id": sample_id,
            "filename": str(sample.get("original_filename") or image_path.name),
            "expected_text": expected_text,
        }

        if not image_path.is_file():
            failure.update({"reason": "圖片檔案不存在", "actual_text": ""})
            failures.append(failure)
            continue

        expected_sha256 = str(sample.get("file_sha256") or "").lower()
        try:
            actual_sha256 = _file_sha256(image_path)
        except OSError as exc:
            failure.update(
                {
                    "reason": f"圖片無法讀取: {exc}",
                    "actual_text": "",
                }
            )
            failures.append(failure)
            continue
        if actual_sha256 != expected_sha256:
            failure.update(
                {
                    "reason": "圖片 SHA-256 與標註紀錄不符",
                    "actual_text": "",
                }
            )
            failures.append(failure)
            continue

        image = read_detection_image(
            image_path,
            cv2.IMREAD_UNCHANGED,
            bool(sample.get("rotation_applied")),
        )
        if image is None:
            failure.update({"reason": "圖片無法讀取", "actual_text": ""})
            failures.append(failure)
            continue

        try:
            result = detect_panel_mark(
                image,
                include_debug=False,
                profile=profile,
                profile_id=profile_id,
            )
        except Exception as exc:
            failure.update(
                {
                    "reason": f"辨識例外: {exc}",
                    "actual_text": "",
                }
            )
            failures.append(failure)
            continue

        actual_text = str(result.get("text") or "") if result.get("found") else ""
        if actual_text != expected_text:
            failure.update(
                {
                    "reason": (
                        "未找到 MARK" if not result.get("found") else "兩碼不符"
                    ),
                    "actual_text": actual_text,
                    "roi": str(result.get("roi") or ""),
                    "orientation": str(result.get("orientation") or ""),
                }
            )
            failures.append(failure)
            continue

        passed += 1

    total = len(sample_list)
    return {
        "profile_id": int(profile_id),
        "total": total,
        "passed": passed,
        "failed": len(failures),
        "success": total > 0 and not failures,
        "sample_set_sha256": mark_sample_set_sha256(sample_list),
        "failures": failures,
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
