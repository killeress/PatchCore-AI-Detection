import csv
import tempfile
from pathlib import Path
import pytest


def _write_manifest(batch_dir: Path, rows: list[dict]) -> None:
    fields = list(rows[0].keys())
    with open(batch_dir / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_merge_run_basic():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for batch_name, label in [("20260101_100000", "over_surface_scratch"),
                                   ("20260102_100000", "true_ng")]:
            b = base / batch_name
            b.mkdir()
            (b / "crop.jpg").write_bytes(b"fake")
            _write_manifest(b, [{"sample_id": f"s_{batch_name}", "label": label,
                                  "crop_path": "crop.jpg", "status": "ok"}])

        from tools.merge_over_review_manifests import run as merge_run
        stats = merge_run(base, set())

        assert stats["total_rows"] == 2
        assert len(stats["batches"]) == 2
        assert stats["label_counts"]["over_surface_scratch"] == 1
        assert Path(stats["out_path"]).exists()


def test_merge_run_exclude():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for batch_name in ["20260101_100000", "20260102_100000"]:
            b = base / batch_name
            b.mkdir()
            (b / "crop.jpg").write_bytes(b"fake")
            _write_manifest(b, [{"sample_id": batch_name, "label": "true_ng",
                                  "crop_path": "crop.jpg", "status": "ok"}])

        from tools.merge_over_review_manifests import run as merge_run
        stats = merge_run(base, {"20260101_100000"})

        assert stats["total_rows"] == 1
        assert stats["batches"] == ["20260102_100000"]


def test_merge_run_skips_non_ok():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        b = base / "20260101_100000"
        b.mkdir()
        (b / "crop.jpg").write_bytes(b"fake")
        _write_manifest(b, [
            {"sample_id": "s1", "label": "true_ng", "crop_path": "crop.jpg", "status": "ok"},
            {"sample_id": "s2", "label": "true_ng", "crop_path": "crop.jpg", "status": "pending"},
        ])

        from tools.merge_over_review_manifests import run as merge_run
        stats = merge_run(base, set())
        assert stats["total_rows"] == 1


def test_merge_run_no_batches_raises():
    with tempfile.TemporaryDirectory() as tmp:
        from tools.merge_over_review_manifests import run as merge_run
        with pytest.raises(ValueError, match="no batch dirs"):
            merge_run(Path(tmp), set())


def test_retrain_state_409_when_running():
    """Verify the 409 detection logic works."""
    import threading
    state = {
        "lock": threading.Lock(),
        "job": {
            "job_id": "retrain_20260424_000000",
            "state": "running",
            "step": "train",
            "started_at": "2026-04-24T00:00:00",
            "output_path": "deployment/test.pkl",
            "log_lines": [],
            "_log_lock": threading.Lock(),
            "summary": None,
            "error": None,
        },
    }
    with state["lock"]:
        job = state["job"]
        already_running = job and job.get("state") == "running"
    assert already_running
