# CAPI AI — Automated Optical Inspection Intelligence System

> **An industrial-grade AI inference platform for panel defect detection, built on PatchCore anomaly detection with real-time TCP/IP communication, heatmap visualization, and human-inspection cross-validation.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PatchCore](https://img.shields.io/badge/AI-PatchCore-orange)](https://github.com/amazon-science/patchcore-inspection)
[![OpenVINO](https://img.shields.io/badge/Runtime-OpenVINO%20%7C%20PyTorch-lightblue)](https://openvino.ai)
[![SQLite](https://img.shields.io/badge/Database-SQLite-green)](https://sqlite.org)

🇹🇼 [繁體中文版說明 → README.zh-TW.md](./README.zh-TW.md)

---

## Overview

CAPI AI integrates seamlessly into existing AOI (Automated Optical Inspection) production lines, acting as a second-layer AI judge that validates machine judgment using deep learning anomaly detection. It supports real-time inference via TCP/IP, persists all results to a SQLite database, and provides a built-in web dashboard for traceability and analytics — including comparison with RIC (human re-inspection) records.

```
AOI Machine  ──TCP/IP──▶  CAPI AI Server  ──▶  SQLite DB
                                │                    │
                                ▼                    ▼
                          Heatmap Files         Web Dashboard
                                                     │
                                            RIC Comparison Report
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **PatchCore Inference** | Tile-based anomaly detection (512×512 patches) with configurable thresholds |
| 🌐 **TCP/IP Server** | Multi-client socket server for real-time AOI integration |
| 🗺️ **Heatmap Visualization** | Per-tile anomaly heatmaps with overlay rendering |
| 🧹 **Smart Filtering** | Dust/scratch suppression via OMIT image cross-validation & Heatmap IOU |
| 💣 **Bomb Defect Detection** | YAML-configurable coordinate-based bomb defect classification |
| 📊 **Web Dashboard** | Real-time monitoring, searchable record history, per-shift statistics |
| 🔎 **RIC Cross-Validation** | Import human inspection (RIC) data and compare against AI/AOI results |
| 🗃️ **3-Layer Traceability** | Record → Image → Tile level persistence with full audit trail |
| ⚡ **Dual Runtime** | Supports both OpenVINO (`.xml`) and PyTorch (`.pt`) model formats |
| 🏭 **Multi-Line Support** | Port-per-line architecture (Line N → Port 79NN) |

---

## Project Structure

```
CAPI01_AD/
│
├── configs/
│   └── capi_3f.yaml              # Inference configuration (thresholds, exclusion zones, bomb coords)
│
├── ── Core Modules ──
├── capi_config.py                # YAML config loader & validator
├── capi_inference.py             # PatchCore inference engine (tiling, scoring, filtering)
├── capi_heatmap.py               # Heatmap generation & file management
├── capi_database.py              # SQLite persistence (records / images / tiles)
│
├── ── Server ──
├── capi_server.py                # TCP Socket Server (production entry point)
├── capi_web.py                   # Web interface (HTTP server + REST API)
├── server_config.yaml            # Linux production config
├── server_config_local.yaml      # Windows local testing config
├── start_server.sh               # Linux startup script
│
├── ── Templates & Static ──
├── templates/                    # Jinja2 HTML templates
│   ├── base.html                 # Layout & navigation
│   ├── dashboard_v3.html         # Real-time monitoring dashboard
│   ├── record_detail_v3.html     # Per-record drill-down with heatmaps
│   ├── ric_report.html           # RIC human inspection comparison report
│   └── ...
├── static/                       # CSS, JS, assets
│
├── ── Tooling ──
├── capi_missed_detection_analyzer.py  # Missed detection batch analyzer
├── diagnose_bomb.py                   # Bomb defect diagnostics & visualizer
├── auto_sender.py                     # Automated test request sender
├── test_client.py                     # TCP client for manual testing
├── check_db.py                        # Database inspection utility
│
├── model.pt                      # Trained PatchCore model
├── capi_mark.png                 # MARK template for exclusion detection
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

> **GPU Note**: OpenVINO inference is recommended for production. PyTorch (`.pt`) is supported for local testing without Intel hardware.

### Linux Production Server

```bash
# 1. Edit configuration
vim server_config.yaml

# 2. Launch (background + web dashboard)
chmod +x start_server.sh
./start_server.sh
```

### Windows Local Testing

```bash
# Terminal A — Start the inference server
python capi_server.py --config server_config_local.yaml

# Terminal B — Run a test request
python test_client.py

# Terminal B — Test with a real panel folder
python test_client.py --real "D:\path\to\panel_folder"

# Browser — View the web dashboard
# http://localhost:8080
```

---

## Communication Protocol

CAPI AI uses a simple semicolon-delimited TCP text protocol:

```
[Request]   AOI@<GlassID>;<ModelID>;<MachineNo>;<ResX>,<ResY>;<MachineJudgment>;<ImageDir>
[Response]  AOI@<GlassID>;<ModelID>;<MachineNo>;<MachineJudgment>;<AIJudgment>
```

**AI Judgment values:**

| Value | Meaning |
|-------|---------|
| `OK` | No defect detected |
| `NG@ImageName(X,Y)` | Defect found at tile coordinates (X, Y) |
| `ERR:description` | Processing error |

---

## Web Dashboard

Access the dashboard at `http://<server>:<port>/` after starting the server.

| Route | Description |
|-------|-------------|
| `/` | Real-time monitoring dashboard |
| `/records` | Searchable inference history |
| `/record/<id>` | Per-record heatmap drill-down |
| `/ric` | RIC human inspection comparison report |
| `/debug` | Single-image debug inference |
| `/stats` | Historical statistics & trends |

### RIC Comparison Report

Import human re-inspection (RIC) `.xls` export files to automatically compare:
- **AOI Accuracy** — Agreement rate between AOI judgment and RIC result
- **AI Accuracy** — Agreement rate between AI judgment and RIC result
- **Over-inspection Rate** — Cases where AOI/AI judged NG but RIC judged OK
- **Miss-detection Rate** — Cases where AOI/AI judged OK but RIC judged NG

Click any stat card to instantly filter the detail table. Export filtered results as CSV.

---

## Configuration

Key parameters in `configs/capi_3f.yaml`:

```yaml
threshold: 0.65                    # Anomaly score threshold (0.0 ~ 1.0)
tile_size: 512                     # Inference tile size (px)
dust_heatmap_top_percent: 0.4      # Dust/scratch IOU top-percentile
excluded_edge_margin: 0.03         # Edge exclusion ratio
bomb_defects:                      # Coordinate-based bomb defect rules
  - image_prefix: "STANDARD"
    coordinates: [[x1,y1], ...]
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AOI Machine (Client)                  │
└──────────────────────────┬──────────────────────────────┘
                           │ TCP/IP Request
┌──────────────────────────▼──────────────────────────────┐
│                   CAPIServer (capi_server.py)            │
│  ┌──────────────┐   ┌───────────────┐   ┌────────────┐  │
│  │ Config Loader│   │ CAPIInferencer│   │HeatmapMgr  │  │
│  │(capi_config) │   │(PatchCore)    │   │(capi_heatm)│  │
│  └──────────────┘   └───────┬───────┘   └────────────┘  │
│                             │                            │
│  ┌──────────────────────────▼──────────────────────────┐ │
│  │              CAPIDatabase (SQLite)                  │ │
│  │  inference_records → image_results → tile_results   │ │
│  └──────────────────────────┬──────────────────────────┘ │
└──────────────────────────── │ ──────────────────────────-┘
                              │
┌─────────────────────────────▼───────────────────────────┐
│                 CAPIWebHandler (capi_web.py)             │
│         HTTP Dashboard + REST API + RIC Reports          │
└─────────────────────────────────────────────────────────┘
```

---

## License

Internal use only. © CAPI AI Team.
