# Retrain UI — Design Spec

**Date:** 2026-04-24
**Status:** Approved

---

## Context

The scratch classifier (DINOv2 + LoRA) accumulates human-reviewed over-review data daily in `/data/capi_ai/datasets/over_review/`. Retraining currently requires SSH access and manual CLI commands. This spec defines a `/retrain` web page that lets operators trigger the full pipeline (merge manifests → train → produce new `.pkl`) from the existing dashboard, without SSH.

Deployment of the new model remains manual: the operator copies the output path and updates `scratch_bundle_path` in settings.

---

## Scope

- New page: `GET /retrain`
- New API: `POST /api/retrain/start`, `GET /api/retrain/status`
- Background job using existing `threading.Thread` + state dict pattern (same as `_dataset_export_state`)
- Log capture via `ListHandler(logging.Handler)` injected into root logger during training
- No auto-deploy, no cancel (GPU jobs cannot be safely interrupted mid-epoch)

Out of scope: hot-reload, auto-deploy, training cancellation.

---

## Architecture

### New global state in `capi_web.py`

```python
_retrain_state = {
    "lock": threading.Lock(),
    "job": None,
    # job dict when running:
    # {
    #   job_id: str,        # uuid4
    #   state: str,         # "running" | "completed" | "failed"
    #   step: str,          # "merge" | "train" | "done"
    #   started_at: str,    # ISO timestamp
    #   output_path: str,   # target .pkl path
    #   log_lines: list[str],
    #   summary: dict | None,  # populated on completion
    #   error: str | None,
    # }
}
```

### Routes

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/retrain` | `_handle_retrain_page` | Render `retrain.html` |
| POST | `/api/retrain/start` | `_handle_retrain_start` | Validate params, start thread, return `{job_id}` |
| GET | `/api/retrain/status` | `_handle_retrain_status` | Return current job state + last N log lines |

### Background thread flow

```
Thread: _retrain_worker(job, args)
│
├─ step = "merge"
│   merge_over_review_manifests.main_with_args(base, exclude=[])
│   → produces manifest_merged.csv
│
├─ step = "train"
│   Attach ListHandler to root logger
│   train_final_model.main(manifest, output, epochs, rank, ...)
│   Detach ListHandler
│   → produces scratch_classifier_vN.pkl
│
├─ step = "done"
│   job["state"] = "completed"
│   job["summary"] = {conformal_threshold, calib_ng_count, ...}
│
└─ on exception:
    job["state"] = "failed"
    job["error"] = traceback string
```

### Log capture

```python
class _ListHandler(logging.Handler):
    def __init__(self, lines: list, lock: threading.Lock):
        super().__init__()
        self.lines = lines
        self.lock = lock

    def emit(self, record):
        msg = self.format(record)
        with self.lock:
            self.lines.append(msg)
```

Attached to the specific loggers used by training scripts only (NOT root logger, to avoid capturing other request threads):
- `logging.getLogger("scripts.over_review_poc.train_final_model")`
- `logging.getLogger("scripts.over_review_poc.finetune_lora")`

Detached in `finally` block after training completes or fails.

---

## UI — `templates/retrain.html`

Extends `base.html`. Three sections:

### 1. Current model info (static, top of page)
- Active `.pkl` path from `config.scratch_bundle_path`
- `trained_at` from bundle metadata (load once at page render, or show "unknown" if unreadable)

### 2. Parameters form
Pre-filled with server-side defaults, all editable:

| Field | Default | Description |
|-------|---------|-------------|
| Manifest base dir | `/data/capi_ai/datasets/over_review` | Where batch dirs are scanned |
| Output .pkl path | `deployment/scratch_classifier_v3.pkl` | Relative to project root |
| Epochs | 15 | LoRA training epochs |
| Rank | 16 | LoRA rank |
| Calib frac | 0.2 | Calibration split fraction |
| DINOv2 repo | `deployment/dinov2_repo` | Local repo dir (offline) |
| DINOv2 weights | `deployment/dinov2_vitb14.pth` | Local weights .pth |

"開始訓練" button — disabled when job is running.

### 3. Progress area (hidden until job starts)

```
Step: [● Merge] → [● Training epoch 9/15] → [○ Done]

Elapsed: 12m 34s

Log:
  2026-04-24 16:36:19 [INFO] Loaded 7688 samples ...
  2026-04-24 16:37:34 [INFO]   epoch 1/15  loss=0.3189
  ...

[completed]
  Output: deployment/scratch_classifier_v3.pkl
  Conformal threshold: 0.4231
  Calib NG count: 287 / 1581
```

JS polls `GET /api/retrain/status` every 3 seconds. Stops polling when `state == "completed"` or `"failed"`. Shows last 100 log lines (scrolled to bottom).

---

## Files Modified

| File | Change |
|------|--------|
| `capi_web.py` | Add `_retrain_state`, `_handle_retrain_page`, `_handle_retrain_start`, `_handle_retrain_status`, `_retrain_worker`, `_ListHandler`; add `/retrain` to sidebar nav |
| `templates/retrain.html` | New template (extends base.html) |
| `tools/merge_over_review_manifests.py` | Extract core logic into `run(base: Path, exclude: set[str]) -> dict` (returns stats dict); `main()` becomes a thin argparse wrapper calling `run()` |

---

## Verification

1. Start server locally: `python capi_server.py --config server_config_local.yaml`
2. Navigate to `http://localhost:8080/retrain`
3. Fill in params pointing to local test data
4. Click "開始訓練", verify step indicator advances and log lines appear
5. Wait for completion, verify `.pkl` file is written to output path
6. Verify that starting a second job while one is running returns 409 error
7. Verify failed job (bad manifest path) shows error state in UI
