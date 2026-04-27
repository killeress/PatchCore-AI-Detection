"""Unit tests for tools.merge_over_review_manifests.run()

TDD: these tests are written first, before run() exists.
"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from tools.merge_over_review_manifests import run


def _write_manifest(batch_dir: Path, rows: list[dict]) -> None:
    fields = list(rows[0].keys())
    with open(batch_dir / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_merge_run_basic():
    """2 batches, 1 row each, different labels → total_rows == 2, both batches in result."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # batch A
        batch_a = base / "20260101_000000"
        batch_a.mkdir()
        crop_a = batch_a / "crop_a.png"
        crop_a.write_bytes(b"fake")
        _write_manifest(batch_a, [
            {"sample_id": "s001", "label": "scratch", "crop_path": "crop_a.png", "status": "ok"},
        ])

        # batch B
        batch_b = base / "20260102_000000"
        batch_b.mkdir()
        crop_b = batch_b / "crop_b.png"
        crop_b.write_bytes(b"fake")
        _write_manifest(batch_b, [
            {"sample_id": "s002", "label": "ok", "crop_path": "crop_b.png", "status": "ok"},
        ])

        stats = run(base, set())

        assert stats["total_rows"] == 2
        assert "20260101_000000" in stats["batches"]
        assert "20260102_000000" in stats["batches"]
        assert stats["label_counts"]["scratch"] == 1
        assert stats["label_counts"]["ok"] == 1


def test_merge_run_exclude():
    """2 batches, exclude one → total_rows == 1, only 1 batch in stats['batches']."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        batch_a = base / "20260101_000000"
        batch_a.mkdir()
        crop_a = batch_a / "crop_a.png"
        crop_a.write_bytes(b"fake")
        _write_manifest(batch_a, [
            {"sample_id": "s001", "label": "scratch", "crop_path": "crop_a.png", "status": "ok"},
        ])

        batch_b = base / "20260102_000000"
        batch_b.mkdir()
        crop_b = batch_b / "crop_b.png"
        crop_b.write_bytes(b"fake")
        _write_manifest(batch_b, [
            {"sample_id": "s002", "label": "ok", "crop_path": "crop_b.png", "status": "ok"},
        ])

        stats = run(base, {"20260102_000000"})

        assert stats["total_rows"] == 1
        assert len(stats["batches"]) == 1
        assert stats["batches"][0] == "20260101_000000"


def test_merge_run_skips_non_ok():
    """1 batch, 2 rows: 1 status=ok, 1 status=pending → total_rows == 1."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        batch_a = base / "20260101_000000"
        batch_a.mkdir()
        crop_ok = batch_a / "crop_ok.png"
        crop_ok.write_bytes(b"fake")
        crop_pending = batch_a / "crop_pending.png"
        crop_pending.write_bytes(b"fake")
        _write_manifest(batch_a, [
            {"sample_id": "s001", "label": "scratch", "crop_path": "crop_ok.png", "status": "ok"},
            {"sample_id": "s002", "label": "scratch", "crop_path": "crop_pending.png", "status": "pending"},
        ])

        stats = run(base, set())

        assert stats["total_rows"] == 1


def test_merge_run_no_batches_raises():
    """empty temp dir → ValueError('no batch dirs')."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        with pytest.raises(ValueError, match="no batch dirs"):
            run(base, set())
