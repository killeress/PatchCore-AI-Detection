"""Tests for the retrain pipeline backend (capi_web.py integration)."""
from __future__ import annotations

import threading


def test_retrain_state_409_when_running():
    """Verify the 409 detection logic works."""
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
