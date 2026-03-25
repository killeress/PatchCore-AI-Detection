# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CAPI AI is an industrial AOI (Automated Optical Inspection) second-layer AI validator using PatchCore anomaly detection. It receives inspection requests from AOI machines via TCP, runs inference on panel images, and returns OK/NG judgments with defect coordinates.

## Running the System

**Linux (production):**
```bash
pip install -r requirements.txt
vim server_config.yaml   # Set model paths, DB path, etc.
./start_server.sh
```

**Windows (local testing):**
```bash
# Terminal A — start server
python capi_server.py --config server_config_local.yaml

# Terminal B — send test requests
python test_client.py
python test_client.py --real "D:\path\to\panel_folder"

# Dashboard: http://localhost:8080
```

**Debug tools:**
```bash
python run_single_inference.py <image_path>   # Single image with heatmap output
python capi_missed_detection_analyzer.py      # Missed detection analysis
python diagnose_bomb.py                       # Bomb defect coordinate debug
python check_db.py                            # Inspect SQLite database
python test_cv_edge.py                        # Test edge detection
```

**Testing:**
```bash
# Start server first, then in another terminal:
python test_inference.py          # TCP protocol test (sends simulated AOI requests)
python test_cv_edge.py            # CV edge detection unit tests
```

## Architecture

```
AOI Machine (TCP, port 7891)
     │  Request: AOI@<GlassID>;<ModelID>;<MachineNo>;<ResX>,<ResY>;<Judgment>;<ImageDir>
     ▼
capi_server.py       — TCP server, per-client persistent threads, GPU lock serialization
     ├── capi_config.py      — YAML config parsing + DB override chain
     ├── capi_inference.py   — Core PatchCore pipeline (preprocessing → tiling → scoring → filtering)
     ├── capi_heatmap.py     — Anomaly heatmap generation and overlay rendering
     ├── capi_edge_cv.py     — CV-based edge defect detection (panel boundary checks)
     └── capi_database.py    — SQLite persistence (WAL mode, 3-layer schema)
          └── capi_web.py    — HTTP dashboard + REST APIs (Jinja2, port 8080)
```

**TCP protocol:**
- Request: `AOI@<GlassID>;<ModelID>;<MachineNo>;<ResX>,<ResY>;<MachineJudgment>;<ImageDir>`
- Response: `AOI@<GlassID>;<ModelID>;<MachineNo>;<MachineJudgment>;<AIJudgment>`
- AI Judgment values: `OK`, `NG@ImageName(X,Y)`, `ERR:description`
- Concatenated requests are handled: server splits multiple `AOI@` messages from a single network packet and queues them in a pending buffer.

## Threading & Concurrency Model

```
Main TCP Loop (accept connections)
  └─ Per-Client Handler Thread (persistent, long-lived, idle timeout 600s)
     ├─ Parse & buffer requests (handles concatenated AOI@ messages)
     ├─ Acquire GPU Lock (serialized — only one inference at a time)
     │  └─ CAPIInferencer.process_panel()
     ├─ Release GPU Lock → Send response immediately (non-blocking)
     └─ Submit to Background ThreadPoolExecutor (2 workers)
        ├─ Heatmap generation (CPU-bound)
        └─ DB persistence
```

**Key design decisions:**
- `_gpu_lock` prevents GPU OOM from concurrent inference — never remove this
- Response is sent to AOI machine **before** heatmap/DB save completes (fast feedback loop)
- `InferenceLogCapture` uses thread-local buffers to capture per-request stdout for web log viewer

## Inference Pipeline (capi_inference.py)

1. Load image + OMIT parallel image (light-only image for dust cross-validation)
2. Otsu binarization → foreground extraction → optional bottom crop (enables "single focus mode" when `otsu_bottom_crop > 0`, disabling exclusion zones)
3. Template-match MARK QR code (`capi_mark.png`) → compute exclusion zones for mechanical areas. Falls back to hardcoded coordinates if template match fails.
4. Tile into 512×512 non-overlapping patches, generate masks
5. PatchCore anomaly scoring (model lazy-loaded, selected by image filename prefix)
6. Dust filtering pipeline:
   - Cross-validate with OMIT image (Top-Hat morphological filter + Otsu)
   - Bright rescue pass (CLAHE-enhanced) for large contamination
   - Check concentration ratio (peak/mean < 2.0 = diffuse = likely false positive)
   - IOU/coverage threshold check
7. Bomb defect check: coordinate-based simulated defect detection (point + line types, line requires aspect ratio ≥ 3.0)
8. Edge margin decay: exponential penalty on boundary-adjacent tiles
9. Aggregate per-tile → per-image → panel judgment

**Multi-model support:** Up to 5 lighting conditions (Green/Red/White/50°Gray/Standard), each with independent thresholds configured by image prefix (`G0F`, `R0F`, `W0F`, `WGF`, `STANDARD`).

## Domain Concepts

- **OMIT image**: Parallel light-only capture (no front-light) used to identify dust vs. real defects. Filename prefix: `OMIT0000`. Cross-validated against AI anomaly heatmaps.
- **MARK**: 2D barcode template in bottom-right of panel. Template-matched at multiple scales (0.75x–3.0x) to auto-detect mechanical exclusion zones.
- **Bomb defect**: System-injected simulated defects to validate AI detection. Two types: point (multiple coordinates) and line (2 endpoints). Coordinates are in product resolution space (e.g., 1920×1080), not raw pixel space.
- **Dust filtering**: Multi-stage pipeline distinguishing dust (surface particles visible under OMIT lighting) from real defects (embedded in panel).

## Database Schema (capi_database.py)

3-layer traceability in SQLite (WAL mode):
1. `inference_records` — per-request summary (glass_id, model_id, machine_no, ai_judgment, duration)
2. `image_results` — per-image breakdown (tile counts, max_score, is_ng, is_dust_only, is_bomb)
3. `tile_results` — per-tile detail (x, y, score, is_anomaly, is_dust, bomb_code, heatmap_path)

Additional tables: `config_params` (runtime overrides), `config_change_history` (audit trail), RIC import tables for human inspection cross-validation.

## Configuration

- `server_config.yaml` / `server_config_local.yaml` — TCP port, DB path, heatmap storage, logging, path_mapping (UNC → Linux mount conversion)
- `configs/capi_3f.yaml` — Inference tuning: MARK template threshold, model-to-prefix mappings, per-model anomaly thresholds, dust detection params, OMIT settings, edge margin decay, bomb defect coordinates

**Override chain:** DB `config_params` overrides → YAML config → hard-coded defaults. Every param change is logged to `config_change_history`.

## Web Dashboard (capi_web.py, port 8080)

Key routes:
- `/` or `/v3/dashboard` — Real-time monitoring (active connections, last judgment, stats)
- `/records` — Searchable inference history with filters
- `/v3/record/<id>` — Per-record drill-down with heatmaps and tile details
- `/ric` — RIC cross-validation report (AI vs. human inspection comparison)
- `/debug` — Single-image debug inference with parameter overrides
- `/settings` — Runtime config management with change history
- `/logs` — Per-request inference log viewer
- `/api/status` — Server status JSON
- `/api/ric/report` — RIC analysis API with CSV export

## Critical Gotchas

- **`TRUST_REMOTE_CODE = "1"` must be set before importing anomalib** — done in both `capi_server.py` and `capi_inference.py`
- **Path mapping**: Windows UNC paths (e.g., `\\192.168.2.101\d`) are mapped to Linux mounts via `path_mapping` in server config
- **Models are lazy-loaded**: First request for a given prefix triggers model load; subsequent requests reuse the cached model
- **Tile coordinates → product coordinates**: Uses heatmap peak position, not tile center, for defect coordinate reporting
- **OMIT prefix must be exact**: `OMIT0000` not `OMIT_0000`

## Key Reference

`CAPI_FLOW.md` — Chinese flowchart documenting the complete inference decision logic.
