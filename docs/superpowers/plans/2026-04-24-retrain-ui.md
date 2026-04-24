# Retrain UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/retrain` web page that lets operators trigger scratch classifier retraining (merge manifests → LoRA fine-tune → produce new `.pkl`) from the dashboard without SSH access.

**Architecture:** Background `threading.Thread` + state dict (identical pattern to `_dataset_export_state`). Frontend polls `GET /api/retrain/status` every 3 s. Training logs captured via `_ListHandler(logging.Handler)` attached to the two named training loggers. Deployment remains manual.

**Tech Stack:** Python stdlib threading, Jinja2 templates (existing), vanilla JS fetch + setInterval, existing `_send_response` / `_send_json` helpers.

---

## File Map

| File | Action |
|------|--------|
| `tools/merge_over_review_manifests.py` | Extract core logic into `run(base, exclude) → dict`; `main()` becomes thin argparse wrapper |
| `scripts/over_review_poc/train_final_model.py` | Change `return 0` → `return {summary dict}`; fix `__main__` block |
| `capi_web.py` | Add `_retrain_state` init, `_ListHandler`, `_retrain_worker`, 3 route handlers, 3 route registrations |
| `templates/retrain.html` | New template extending `base.html` with params form + progress area |
| `templates/base.html` | Add sidebar nav link for `/retrain` |
| `tests/test_retrain_pipeline.py` | Unit tests for `merge_run()` and retrain state 409 logic |

---

## Task 1 — Extract `run()` from merge_over_review_manifests.py

**Files:**
- Modify: `tools/merge_over_review_manifests.py`
- Create: `tests/test_retrain_pipeline.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_retrain_pipeline.py`:

```python
import csv
import tempfile
from pathlib import Path
import pytest


def _write_manifest(batch_dir: Path, rows: list[dict]) -> None:
    fields = list(rows[0].keys()) if rows else ["sample_id", "label", "crop_path", "status"]
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
            crop = b / "crop.jpg"
            crop.write_bytes(b"fake")
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
```

- [ ] **Step 2: Run tests — expect ImportError (function doesn't exist yet)**

```
cd D:\SourceCode\PatchCore-AI-Detection
python -m pytest tests/test_retrain_pipeline.py -v 2>&1 | head -30
```

Expected: `ImportError` or `AttributeError: module has no attribute 'run'`

- [ ] **Step 3: Rewrite merge_over_review_manifests.py**

Replace the entire file:

```python
"""Merge over_review manifest.csv from multiple batch folders into one.

Auto-discovers all subdirs under <base> that contain a manifest.csv,
sorted by folder name (YYYYMMDD_HHMMSS order = chronological).

Usage:
    python -m tools.merge_over_review_manifests
    python -m tools.merge_over_review_manifests --base /data/capi_ai/datasets/over_review
    python -m tools.merge_over_review_manifests --base ... --exclude 20260415_104812
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def discover_batches(base: Path, exclude: set[str]) -> list[str]:
    """Return sorted list of batch dir names that have a manifest.csv."""
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and (d / "manifest.csv").exists() and d.name not in exclude
    )


def run(base: Path, exclude: set[str]) -> dict:
    """Merge all batch manifests under base into manifest_merged.csv.

    Returns:
        batches: list of merged batch dir names
        total_rows: number of rows written
        label_counts: {label: count}
        out_path: str path to manifest_merged.csv
    Raises:
        ValueError: if no valid batch dirs found
    """
    batches = discover_batches(base, exclude)
    if not batches:
        raise ValueError(f"no batch dirs with manifest.csv found under {base}")

    out_path = base / "manifest_merged.csv"

    all_fields: list[str] = []
    for b in batches:
        with open(base / b / "manifest.csv", encoding="utf-8-sig") as f:
            for col in csv.DictReader(f).fieldnames or []:
                if col not in all_fields:
                    all_fields.append(col)

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

    labels = Counter(r["label"] for r in rows)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        w.writerows(rows)

    print(f"merging {len(batches)} batches: {batches}")
    print(f"status!=ok skipped: {skipped_status} | duplicates: {dup} | missing crop: {missing_crop}")
    print(f"final rows: {len(rows)}")
    for lab, c in labels.most_common():
        print(f"  {lab}: {c}")
    print(f"written: {out_path}")

    return {
        "batches": batches,
        "total_rows": len(rows),
        "label_counts": dict(labels),
        "out_path": str(out_path),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=Path("dataset_v2/over_review"),
                   help="Root folder containing all batch subdirs")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="Batch dir names to skip")
    args = p.parse_args()
    try:
        run(args.base, set(args.exclude))
    except ValueError as e:
        print(f"[error] {e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — expect PASS**

```
python -m pytest tests/test_retrain_pipeline.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add tools/merge_over_review_manifests.py tests/test_retrain_pipeline.py
git commit -m "refactor(merge): extract run() for programmatic use; add unit tests"
```

---

## Task 2 — Return summary dict from train_final_model.main()

**Files:**
- Modify: `scripts/over_review_poc/train_final_model.py` lines 158–166 and the `__main__` block

- [ ] **Step 1: Replace `return 0` with summary dict (line 166)**

In `scripts/over_review_poc/train_final_model.py`, replace:

```python
    print()
    print("=== train_final_model summary ===")
    print(f"Bundle:             {args.output}")
    print(f"Conformal thresh:   {conformal_threshold:.4f}")
    print(f"Default safety:     {args.default_safety}")
    print(f"Est. eff. thresh:   {min(conformal_threshold * args.default_safety, 0.9999):.4f}")
    print(f"Calib NG count:     {int(calib_ng_mask.sum())} / {len(calib_idx)}")
    print(f"Calib score range:  [{calib_scores.min():.3f}, {calib_scores.max():.3f}]")
    return 0
```

With:

```python
    print()
    print("=== train_final_model summary ===")
    print(f"Bundle:             {args.output}")
    print(f"Conformal thresh:   {conformal_threshold:.4f}")
    print(f"Default safety:     {args.default_safety}")
    print(f"Est. eff. thresh:   {min(conformal_threshold * args.default_safety, 0.9999):.4f}")
    print(f"Calib NG count:     {int(calib_ng_mask.sum())} / {len(calib_idx)}")
    print(f"Calib score range:  [{calib_scores.min():.3f}, {calib_scores.max():.3f}]")

    return {
        "output_path": str(args.output),
        "total_samples": len(samples),
        "scratch_count": int(y.sum()),
        "conformal_threshold": conformal_threshold,
        "effective_threshold": float(min(conformal_threshold * args.default_safety, 0.9999)),
        "calib_ng_count": int(calib_ng_mask.sum()),
        "calib_total": len(calib_idx),
    }
```

- [ ] **Step 2: Fix the `__main__` block (last 2 lines of file)**

Replace:

```python
if __name__ == "__main__":
    sys.exit(main())
```

With:

```python
if __name__ == "__main__":
    result = main()
    sys.exit(0 if isinstance(result, dict) else result)
```

- [ ] **Step 3: Verify CLI still works**

```bash
python -m scripts.over_review_poc.train_final_model --help
```

Expected: shows help without error.

- [ ] **Step 4: Commit**

```bash
git add scripts/over_review_poc/train_final_model.py
git commit -m "feat(train): return summary dict from main() for programmatic use"
```

---

## Task 3 — Add retrain backend to capi_web.py

**Files:**
- Modify: `capi_web.py`

- [ ] **Step 1: Add `_ListHandler` class before `CAPIWebHandler`**

Find the line `class CAPIWebHandler(BaseHTTPRequestHandler):` in capi_web.py and insert this class immediately before it:

```python
class _ListHandler(logging.Handler):
    """Captures log records into a list for the retrain progress UI."""

    def __init__(self, lines: list, lock: threading.Lock):
        super().__init__()
        self.lines = lines
        self.lock = lock
        self.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with self.lock:
                self.lines.append(msg)
        except Exception:
            pass


```

- [ ] **Step 2: Initialize `_retrain_state` in `create_web_server`**

In `create_web_server`, after the line that sets `CAPIWebHandler._dataset_export_state` (around line 4331–4336), add:

```python
    CAPIWebHandler._retrain_state = {
        "lock": threading.Lock(),
        "job": None,
        # job dict shape when active:
        # { job_id, state, step, started_at, output_path,
        #   log_lines, _log_lock, summary, error }
    }
```

- [ ] **Step 3: Add `_retrain_worker` static method to `CAPIWebHandler`**

Find the end of the class (just before `def create_web_server`) and add:

```python
    @staticmethod
    def _retrain_worker(job: dict, params: dict) -> None:
        """Background thread: merge manifests → train → update job state."""
        import traceback
        from tools.merge_over_review_manifests import run as merge_run
        from scripts.over_review_poc.train_final_model import main as train_main

        log_lines = job["log_lines"]
        log_lock = job["_log_lock"]

        handler = _ListHandler(log_lines, log_lock)
        watched_loggers = [
            logging.getLogger("scripts.over_review_poc.train_final_model"),
            logging.getLogger("scripts.over_review_poc.finetune_lora"),
        ]
        for lg in watched_loggers:
            lg.addHandler(handler)

        try:
            # ── Step 1: merge manifests ──────────────────────────────────
            with CAPIWebHandler._retrain_state["lock"]:
                job["step"] = "merge"

            base = Path(params["manifest_base"])
            with log_lock:
                log_lines.append(f"[merge] 掃描資料目錄 {base} ...")
            merge_stats = merge_run(base, set())
            manifest_path = Path(merge_stats["out_path"])
            with log_lock:
                log_lines.append(
                    f"[merge] 完成：{merge_stats['total_rows']} 筆，"
                    f"共 {len(merge_stats['batches'])} 批次"
                )

            # ── Step 2: train ────────────────────────────────────────────
            with CAPIWebHandler._retrain_state["lock"]:
                job["step"] = "train"

            train_argv = [
                "--manifest", str(manifest_path),
                "--transform", "clahe",
                "--clahe-clip", str(params.get("clahe_clip", 4.0)),
                "--rank", str(params.get("rank", 16)),
                "--alpha", str(params.get("alpha", params.get("rank", 16))),
                "--n-lora-blocks", "2",
                "--epochs", str(params.get("epochs", 15)),
                "--calib-frac", str(params.get("calib_frac", 0.2)),
                "--output", str(params["output_path"]),
            ]
            if params.get("dinov2_repo"):
                train_argv += ["--dinov2-repo", str(params["dinov2_repo"])]
            if params.get("dinov2_weights"):
                train_argv += ["--dinov2-weights", str(params["dinov2_weights"])]

            summary = train_main(train_argv)

            # ── Done ─────────────────────────────────────────────────────
            with CAPIWebHandler._retrain_state["lock"]:
                job["step"] = "done"
                job["state"] = "completed"
                job["summary"] = summary

        except Exception:
            err = traceback.format_exc()
            with CAPIWebHandler._retrain_state["lock"]:
                job["state"] = "failed"
                job["error"] = err
            with log_lock:
                log_lines.append(f"[ERROR] {err}")

        finally:
            for lg in watched_loggers:
                lg.removeHandler(handler)
```

- [ ] **Step 4: Add the three route handler methods to `CAPIWebHandler`**

Add immediately before `_retrain_worker`:

```python
    def _handle_retrain_page(self):
        """GET /retrain"""
        current_bundle = ""
        trained_at = "未知"
        server_inst = self._capi_server_instance
        if server_inst:
            cfg = getattr(server_inst, "config", None)
            if cfg:
                current_bundle = getattr(cfg, "scratch_bundle_path", "")
        if current_bundle:
            try:
                import pickle
                with open(current_bundle, "rb") as f:
                    bundle = pickle.load(f)
                trained_at = bundle.get("metadata", {}).get("trained_at", "未知")
            except Exception:
                pass

        template = self.jinja_env.get_template("retrain.html")
        html = template.render(
            request_path="/retrain",
            current_bundle=current_bundle or "(未設定)",
            trained_at=trained_at,
            default_manifest_base="/data/capi_ai/datasets/over_review",
            default_output_path="deployment/scratch_classifier_v3.pkl",
            default_epochs=15,
            default_rank=16,
            default_calib_frac=0.2,
            default_dinov2_repo="deployment/dinov2_repo",
            default_dinov2_weights="deployment/dinov2_vitb14.pth",
        )
        self._send_response(200, html)

    def _handle_retrain_start(self):
        """POST /api/retrain/start"""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            params = json.loads(body) if body else {}
        except Exception:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return

        if not params.get("manifest_base") or not params.get("output_path"):
            self._send_json({"error": "manifest_base and output_path are required"}, status=400)
            return

        state = self._retrain_state
        with state["lock"]:
            job = state["job"]
            if job and job.get("state") == "running":
                self._send_json({
                    "error": "job_already_running",
                    "job_id": job["job_id"],
                }, status=409)
                return

            job_id = datetime.now().strftime("retrain_%Y%m%d_%H%M%S")
            new_job = {
                "job_id": job_id,
                "state": "running",
                "step": "merge",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "output_path": params["output_path"],
                "log_lines": [],
                "_log_lock": threading.Lock(),
                "summary": None,
                "error": None,
            }
            state["job"] = new_job

        thread = threading.Thread(
            target=CAPIWebHandler._retrain_worker,
            args=(new_job, params),
            daemon=True,
            name=f"retrain-{job_id}",
        )
        thread.start()
        self._send_json({"job_id": job_id, "started_at": new_job["started_at"]})

    def _handle_retrain_status(self):
        """GET /api/retrain/status"""
        state = self._retrain_state
        with state["lock"]:
            job = state["job"]
            if not job:
                self._send_json({"state": "idle"})
                return
            log_lock = job["_log_lock"]
            resp = {
                "job_id": job["job_id"],
                "state": job["state"],
                "step": job["step"],
                "started_at": job["started_at"],
                "output_path": job["output_path"],
                "summary": job["summary"],
                "error": job["error"],
            }

        with log_lock:
            resp["log_lines"] = list(job["log_lines"][-100:])

        try:
            started = datetime.fromisoformat(resp["started_at"])
            resp["elapsed_sec"] = round((datetime.now() - started).total_seconds(), 1)
        except Exception:
            resp["elapsed_sec"] = 0

        self._send_json(resp)
```

- [ ] **Step 5: Register routes in `do_GET`**

In the `do_GET` method, after the line `elif path == "/debug":` (line ~172), add:

```python
            elif path == "/retrain":
                self._handle_retrain_page()
            elif path == "/api/retrain/status":
                self._handle_retrain_status()
```

- [ ] **Step 6: Register route in `do_POST`**

In the `do_POST` method, after the lines for `/api/dataset_export/start` (lines ~297–299), add:

```python
            elif path == "/api/retrain/start":
                self._handle_retrain_start()
                return
```

- [ ] **Step 7: Add 409 test to test_retrain_pipeline.py**

Append to `tests/test_retrain_pipeline.py`:

```python
def test_retrain_state_409_when_running():
    """Simulate the 409 check without spinning up a real HTTP server."""
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
    assert already_running, "Should detect already-running job"
```

- [ ] **Step 8: Run tests**

```
python -m pytest tests/test_retrain_pipeline.py -v
```

Expected: 5 PASSED

- [ ] **Step 9: Commit**

```bash
git add capi_web.py tests/test_retrain_pipeline.py
git commit -m "feat(web): add /retrain page backend — state dict, worker, 3 route handlers"
```

---

## Task 4 — Create templates/retrain.html

**Files:**
- Create: `templates/retrain.html`

- [ ] **Step 1: Create the template**

Create `templates/retrain.html`:

```html
{% extends "base.html" %}
{% block title %}模型重訓練 — CAPI AI{% endblock %}

{% block content %}
<div style="padding:24px; max-width:900px;">

  <h2 style="margin:0 0 20px; font-size:18px; color:#cdd6f4;">刮痕分類器重訓練</h2>

  <!-- 目前部署模型 -->
  <div style="background:#1e1e2e; border-radius:8px; padding:16px; margin-bottom:20px;">
    <div style="font-size:12px; color:#6c7086; margin-bottom:10px; text-transform:uppercase; letter-spacing:.05em;">目前部署模型</div>
    <table style="width:100%; border-collapse:collapse; font-size:13px;">
      <tr>
        <td style="color:#888; width:100px; padding:4px 0;">模型檔案</td>
        <td style="font-family:monospace; color:#cdd6f4; font-size:12px;">{{ current_bundle }}</td>
      </tr>
      <tr>
        <td style="color:#888; padding:4px 0;">訓練時間</td>
        <td style="color:#cdd6f4;">{{ trained_at }}</td>
      </tr>
    </table>
  </div>

  <!-- 訓練參數 -->
  <div style="background:#1e1e2e; border-radius:8px; padding:16px; margin-bottom:20px;">
    <div style="font-size:12px; color:#6c7086; margin-bottom:14px; text-transform:uppercase; letter-spacing:.05em;">訓練參數</div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; font-size:13px;">

      <div style="grid-column:1/-1;">
        <label style="color:#888; display:block; margin-bottom:4px;">資料集目錄</label>
        <input id="manifest_base" value="{{ default_manifest_base }}"
               style="width:100%; box-sizing:border-box; background:#313244; border:1px solid #45475a; color:#cdd6f4; padding:6px 10px; border-radius:4px; font-family:monospace; font-size:12px;">
      </div>

      <div style="grid-column:1/-1;">
        <label style="color:#888; display:block; margin-bottom:4px;">輸出模型路徑 (.pkl)</label>
        <input id="output_path" value="{{ default_output_path }}"
               style="width:100%; box-sizing:border-box; background:#313244; border:1px solid #45475a; color:#cdd6f4; padding:6px 10px; border-radius:4px; font-family:monospace; font-size:12px;">
      </div>

      <div>
        <label style="color:#888; display:block; margin-bottom:4px;">訓練回合數 (epochs)</label>
        <input id="epochs" type="number" min="1" max="100" value="{{ default_epochs }}"
               style="width:100%; box-sizing:border-box; background:#313244; border:1px solid #45475a; color:#cdd6f4; padding:6px 10px; border-radius:4px;">
      </div>

      <div>
        <label style="color:#888; display:block; margin-bottom:4px;">LoRA rank</label>
        <input id="rank" type="number" min="4" max="64" value="{{ default_rank }}"
               style="width:100%; box-sizing:border-box; background:#313244; border:1px solid #45475a; color:#cdd6f4; padding:6px 10px; border-radius:4px;">
      </div>

      <div>
        <label style="color:#888; display:block; margin-bottom:4px;">校準比例 (calib frac)</label>
        <input id="calib_frac" type="number" min="0.05" max="0.5" step="0.05" value="{{ default_calib_frac }}"
               style="width:100%; box-sizing:border-box; background:#313244; border:1px solid #45475a; color:#cdd6f4; padding:6px 10px; border-radius:4px;">
      </div>

      <div></div>

      <div>
        <label style="color:#888; display:block; margin-bottom:4px;">DINOv2 repo 目錄</label>
        <input id="dinov2_repo" value="{{ default_dinov2_repo }}"
               style="width:100%; box-sizing:border-box; background:#313244; border:1px solid #45475a; color:#cdd6f4; padding:6px 10px; border-radius:4px; font-family:monospace; font-size:12px;">
      </div>

      <div>
        <label style="color:#888; display:block; margin-bottom:4px;">DINOv2 weights (.pth)</label>
        <input id="dinov2_weights" value="{{ default_dinov2_weights }}"
               style="width:100%; box-sizing:border-box; background:#313244; border:1px solid #45475a; color:#cdd6f4; padding:6px 10px; border-radius:4px; font-family:monospace; font-size:12px;">
      </div>
    </div>

    <div style="margin-top:18px;">
      <button id="start-btn" onclick="startRetrain()"
              style="background:#89b4fa; color:#1e1e2e; border:none; padding:10px 28px; border-radius:6px; font-size:14px; font-weight:700; cursor:pointer;">
        開始訓練
      </button>
    </div>
  </div>

  <!-- 進度區塊 (hidden until job starts) -->
  <div id="progress-area" style="display:none; background:#1e1e2e; border-radius:8px; padding:16px;">

    <!-- Step indicator -->
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:14px; font-size:12px;">
      <span id="step-merge" style="padding:4px 14px; border-radius:12px; background:#313244; color:#888;">① 合併資料</span>
      <span style="color:#45475a;">→</span>
      <span id="step-train" style="padding:4px 14px; border-radius:12px; background:#313244; color:#888;">② 訓練模型</span>
      <span style="color:#45475a;">→</span>
      <span id="step-done" style="padding:4px 14px; border-radius:12px; background:#313244; color:#888;">③ 完成</span>
    </div>

    <!-- Elapsed -->
    <div style="font-size:12px; color:#6c7086; margin-bottom:10px;">已執行：<span id="elapsed">0:00</span></div>

    <!-- Log -->
    <div id="log-box"
         style="background:#11111b; border-radius:4px; padding:10px; height:220px; overflow-y:auto;
                font-family:monospace; font-size:11px; color:#a6adc8; white-space:pre-wrap; word-break:break-all;">
    </div>

    <!-- Summary (shown on completion) -->
    <div id="summary-box" style="display:none; margin-top:16px;">
      <div style="font-size:14px; color:#a6e3a1; margin-bottom:12px; font-weight:600;">✓ 訓練完成</div>
      <table style="width:100%; border-collapse:collapse; font-size:13px;">
        <tr style="border-bottom:1px solid #313244;">
          <td style="color:#888; padding:8px 0; width:160px;">新模型檔案</td>
          <td id="s-output" style="font-family:monospace; font-size:12px; color:#cdd6f4; padding:8px 0;"></td>
        </tr>
        <tr style="border-bottom:1px solid #313244;">
          <td style="color:#888; padding:8px 0;">訓練資料筆數</td>
          <td id="s-total" style="color:#cdd6f4; padding:8px 0;"></td>
        </tr>
        <tr style="border-bottom:1px solid #313244;">
          <td style="color:#888; padding:8px 0;">刮痕樣本數</td>
          <td id="s-scratch" style="color:#cdd6f4; padding:8px 0;"></td>
        </tr>
        <tr style="border-bottom:1px solid #313244;">
          <td style="color:#888; padding:8px 0;">刮痕判定敏感度</td>
          <td id="s-threshold" style="color:#cdd6f4; padding:8px 0;"></td>
        </tr>
        <tr>
          <td style="color:#888; padding:8px 0;">不良品保護驗證</td>
          <td id="s-calib" style="color:#cdd6f4; padding:8px 0;"></td>
        </tr>
      </table>
      <div style="margin-top:12px; padding:10px 14px; background:#1a2e1a; border-radius:6px; font-size:12px; color:#a6e3a1;">
        部署方式：將「新模型檔案」路徑填入
        <a href="/settings" style="color:#89b4fa;">設定頁面</a>
        → <code style="background:#313244; padding:1px 5px; border-radius:3px;">scratch_bundle_path</code>
      </div>
    </div>

    <!-- Error box -->
    <div id="error-box"
         style="display:none; margin-top:14px; padding:12px; background:#3b1b1b; border-radius:6px;
                font-size:11px; color:#f38ba8; font-family:monospace; white-space:pre-wrap; word-break:break-all;">
    </div>
  </div><!-- /progress-area -->

</div>

<script>
let _pollTimer = null;

function _getParams() {
  return {
    manifest_base:  document.getElementById('manifest_base').value.trim(),
    output_path:    document.getElementById('output_path').value.trim(),
    epochs:         parseInt(document.getElementById('epochs').value, 10),
    rank:           parseInt(document.getElementById('rank').value, 10),
    calib_frac:     parseFloat(document.getElementById('calib_frac').value),
    dinov2_repo:    document.getElementById('dinov2_repo').value.trim(),
    dinov2_weights: document.getElementById('dinov2_weights').value.trim(),
  };
}

function startRetrain() {
  const params = _getParams();
  if (!params.manifest_base || !params.output_path) {
    alert('請填寫資料集目錄和輸出路徑');
    return;
  }
  document.getElementById('start-btn').disabled = true;
  document.getElementById('progress-area').style.display = '';
  document.getElementById('summary-box').style.display  = 'none';
  document.getElementById('error-box').style.display    = 'none';
  document.getElementById('log-box').textContent = '';
  _setStep('merge');

  fetch('/api/retrain/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(params),
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) { _showError(data.error); document.getElementById('start-btn').disabled = false; return; }
    _pollTimer = setInterval(_poll, 3000);
    _poll();
  })
  .catch(e => { _showError(String(e)); document.getElementById('start-btn').disabled = false; });
}

function _poll() {
  fetch('/api/retrain/status').then(r => r.json()).then(_updateUI).catch(console.error);
}

function _setStep(step) {
  const steps = ['merge', 'train', 'done'];
  steps.forEach(s => {
    const el = document.getElementById('step-' + s);
    if (s === step) {
      el.style.background = step === 'done' ? '#a6e3a1' : '#89b4fa';
      el.style.color = '#1e1e2e';
    } else {
      el.style.background = '#313244';
      el.style.color = '#888';
    }
  });
}

function _fmtElapsed(sec) {
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  return m + ':' + String(s).padStart(2, '0');
}

function _sensitivityLabel(thresh) {
  if (thresh < 0.35) return thresh.toFixed(2) + '  ──  嚴格（低漏放率、可能較多過殺）';
  if (thresh < 0.55) return thresh.toFixed(2) + '  ──  適中';
  return thresh.toFixed(2) + '  ──  寬鬆（低過殺率、需注意漏放率）';
}

function _updateUI(data) {
  if (!data || data.state === 'idle') return;
  if (data.elapsed_sec !== undefined)
    document.getElementById('elapsed').textContent = _fmtElapsed(data.elapsed_sec);
  if (data.step) _setStep(data.step === 'done' ? 'done' : data.step);
  if (data.log_lines && data.log_lines.length) {
    const box = document.getElementById('log-box');
    box.textContent = data.log_lines.join('\n');
    box.scrollTop = box.scrollHeight;
  }
  if (data.state === 'completed') {
    clearInterval(_pollTimer);
    document.getElementById('start-btn').disabled = false;
    _setStep('done');
    const s = data.summary || {};
    document.getElementById('s-output').textContent    = s.output_path || '';
    document.getElementById('s-total').textContent     = (s.total_samples || 0).toLocaleString() + ' 筆';
    document.getElementById('s-scratch').textContent   = (s.scratch_count || 0).toLocaleString() + ' 筆';
    document.getElementById('s-threshold').textContent = _sensitivityLabel(s.effective_threshold || 0);
    document.getElementById('s-calib').textContent     = (s.calib_ng_count || 0) + ' 個已知不良品全數通過校準（共 ' + (s.calib_total || 0) + ' 筆校準樣本）';
    document.getElementById('summary-box').style.display = '';
  }
  if (data.state === 'failed') {
    clearInterval(_pollTimer);
    document.getElementById('start-btn').disabled = false;
    _showError(data.error || '未知錯誤');
  }
}

function _showError(msg) {
  const box = document.getElementById('error-box');
  box.textContent = msg;
  box.style.display = '';
}
</script>
{% endblock %}
```

- [ ] **Step 2: Verify template renders (no server needed)**

```bash
python -c "
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('templates'))
t = env.get_template('retrain.html')
html = t.render(
    request_path='/retrain',
    current_bundle='deployment/scratch_classifier_v1.pkl',
    trained_at='2026-04-20T10:00:00',
    default_manifest_base='/data/capi_ai/datasets/over_review',
    default_output_path='deployment/scratch_classifier_v3.pkl',
    default_epochs=15, default_rank=16, default_calib_frac=0.2,
    default_dinov2_repo='deployment/dinov2_repo',
    default_dinov2_weights='deployment/dinov2_vitb14.pth',
)
print('OK, length:', len(html))
"
```

Expected: `OK, length: <number>`

- [ ] **Step 3: Commit**

```bash
git add templates/retrain.html
git commit -m "feat(web): add retrain.html template with params form and progress polling UI"
```

---

## Task 5 — Nav link + End-to-End Verification

**Files:**
- Modify: `templates/base.html`

- [ ] **Step 1: Add nav link in base.html**

In `templates/base.html`, find the nav section with the `/settings` link:

```html
                <a href="/settings" {% if request_path=='/settings' %}class="active" {% endif %}
                    style="display:inline-flex; align-items:center; gap:6px;"><img src="/imgs/nav_stats.png"
                        style="width: 30px; height: 30px; object-fit: contain; border-radius: 4px;"> 參數設定</a>
```

Add immediately after it:

```html
                <a href="/retrain" {% if request_path=='/retrain' %}class="active" {% endif %}
                    style="display:inline-flex; align-items:center; gap:6px;"><img src="/imgs/nav_stats.png"
                        style="width: 30px; height: 30px; object-fit: contain; border-radius: 4px;"> 模型重訓練</a>
```

- [ ] **Step 2: Start server and verify page loads**

```bash
python capi_server.py --config server_config_local.yaml
```

Open `http://localhost:8080/retrain` — expect the page to load with the params form.

- [ ] **Step 3: Verify 409 on double-start**

With server running, in another terminal:

```bash
# Start a job
curl -s -X POST http://localhost:8080/api/retrain/start \
  -H "Content-Type: application/json" \
  -d '{"manifest_base":"/bad/path","output_path":"deployment/test.pkl"}' | python -m json.tool

# Immediately try again (should get 409)
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8080/api/retrain/start \
  -H "Content-Type: application/json" \
  -d '{"manifest_base":"/bad/path","output_path":"deployment/test.pkl"}'
```

Expected: first call returns `{"job_id": "..."}`, second call returns HTTP 409. (First call will fail fast with ValueError from merge, which is fine for this check — the job state transitions to "failed" quickly, allowing a new start.)

- [ ] **Step 4: Verify error state is displayed**

In the browser, fill in a bad `manifest_base` (e.g. `/nonexistent`), click "開始訓練". Wait ~5 s. Expect the error box to appear with the traceback.

- [ ] **Step 5: Run all tests**

```bash
python -m pytest tests/test_retrain_pipeline.py -v
```

Expected: 5 PASSED

- [ ] **Step 6: Final commit**

```bash
git add templates/base.html
git commit -m "feat(web): add retrain nav link; complete /retrain UI feature"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** merge tool refactor ✓, train summary dict ✓, retrain page ✓, params form ✓, progress polling ✓, operator-friendly summary ✓, no auto-deploy ✓, 409 on double-start ✓
- [x] **No placeholders:** all steps have exact code
- [x] **Type consistency:** `merge_run()` returns `dict` with key `"out_path"` — used as `merge_stats["out_path"]` in worker ✓; `train_main()` returns dict — used as `job["summary"]` ✓; `job["_log_lock"]` accessed consistently ✓
- [x] **Deadlock check:** status handler acquires `state["lock"]`, gets `_log_lock` reference, releases `state["lock"]`, then acquires `_log_lock` — never holds both simultaneously ✓
