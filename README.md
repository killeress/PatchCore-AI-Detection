# CAPI AI — AOI PatchCore Inspection Platform

> Production-oriented AI inference service for AOI panel inspection. It receives AOI requests over TCP, runs the configured PatchCore pipeline, returns the legacy AOI result together with the QJPG report, and stores traceable results for Web review.

🇹🇼 [繁體中文說明 → README.zh-TW.md](./README.zh-TW.md)

## What this repository contains

- **Inference server** — `capi_server.py` handles persistent TCP client connections, request parsing, model dispatch, inference, and protocol responses.
- **PatchCore pipeline** — `capi_inference.py` and `capi_preprocess.py` cover panel preprocessing, tile/zone routing, anomaly scoring, heatmaps, MARK and bomb handling, and post-processing rules defined by model configuration.
- **Traceability and Web UI** — `capi_database.py` stores inference, image, and tile records in SQLite; `capi_web.py` serves monitoring, search, record details, RIC review, and administration pages.
- **Training and model library** — the `/training` → `/train/new` workflow prepares training data, reviews tiles, trains model bundles, and manages activation from `/models`.
- **Deployment support** — release metadata, deploy ZIP generation, manual update, and pull-based update helpers are included in the repository.

The production data path is:

```text
AOI client
    │ TCP
    ▼
capi_server.py ──► capi_inference.py / capi_preprocess.py
    │                              │
    ├── legacy AOI + QJPG response │
    ├── SQLite inference records   └── heatmaps and diagnostics
    ▼
capi_web.py ──► dashboard, review, training, model library, settings
```

## Requirements and installation

Use Python 3.10 or newer; the current development/deployment environments use Python 3.11/3.12.

```bash
python -m pip install -r requirements.txt
```

The repository does not contain production model weights. A runnable installation also needs model bundles and image-path mappings appropriate for the target machine. Weight files, databases, local datasets, and local credentials are intentionally excluded from normal source control and deployment packaging.

## Start the server

### Windows local test

`server_config_local.yaml` is the local profile:

- TCP server: `0.0.0.0:7891`
- Web UI: `http://localhost:8080`
- SQLite database: `./test_results.db`
- Heatmaps: `./test_heatmaps`

Start it with either command:

```powershell
python capi_server.py --config server_config_local.yaml
# or
start_server_local.bat
```

If a panel dataset is available, `auto_sender.py` can send sample requests:

```powershell
python auto_sender.py --host 127.0.0.1 --port 7891 --ng-folder D:\path\to\panels --count 1
```

### Linux production

Edit `server_config.yaml` for the target machine, then use the service helper:

```bash
chmod +x start_server.sh
./start_server.sh              # stop old process, start in background, tail the log
./start_server.sh status
./start_server.sh log
./start_server.sh stop
```

The production profile currently defaults to TCP port `7907` and Web port `80`. The actual ports, database path, heatmap path, model list, path mapping, retention policy, and optional integrations are controlled by `server_config.yaml`.

For a direct foreground start:

```bash
python3 capi_server.py --config server_config.yaml
```

Do not copy production paths or credentials into the local profile. In particular, `server_config.yaml` contains machine-specific paths and MES settings that must be reviewed before deployment.

## TCP protocol

The server accepts semicolon-delimited `AOI@` requests. A request without bomb coordinates is:

```text
AOI@<glass_id>;<model_id>;<machine_no>;<resolution_x>,<resolution_y>;<machine_judgment>;<image_dir>
```

A request with bomb data adds an image prefix and coordinates before the image path:

```text
AOI@<glass_id>;<model_id>;<machine_no>;<resolution_x>,<resolution_y>;<machine_judgment>;<image_prefix>;<coordinates>;<image_dir>
```

`machine_judgment` is normally `OK`, `NG`, or `HY`. `HY` skips AI inference and is returned as an image-abnormal result.

The current response is CRLF-terminated and contains both formats, in this order:

```text
AOI@<glass_id>;<model_id>;<machine_no>;<machine_judgment>;<ai_judgment>
@QJPG-<glass_id>;<mark_status>;<mark_text>;<defect_field>,
```

Clients should identify each line by its prefix (`AOI@` or `@QJPG-`) instead of assuming that a response contains only one line. `ai_judgment` can be `OK`, `NG`, or `ERR:<description>`; the internal `OK-i` result is exposed as `OK` in the legacy response. The complete field and QJPG defect-code specification is in [docs/client_communication_protocol.zh-TW.md](./docs/client_communication_protocol.zh-TW.md).

## Main Web UI entry points

Open `http://<server>:<web_port>/` after the server starts.

| Path | Purpose |
|---|---|
| `/` | Live dashboard and current shift status |
| `/search` | Search and export inference records |
| `/record/<id>` | Record details, images, tiles, and heatmaps |
| `/ric` | RIC, over-review, miss-review, MES comparison, and related reports |
| `/ric/within-spec-logs` | Within-spec review list and details |
| `/training` | Training hub |
| `/train/new` | New-machine PatchCore training workflow |
| `/models` | Model bundle inspection and activation |
| `/debug` | Single-image and coordinate diagnostics |
| `/white-frame` | White-frame overview and records |
| `/settings` | Authenticated settings and account administration |
| `/logs` | Server log viewer |
| `/release-notes` | In-app release notes |
| `/api/status` | Runtime and hardware status JSON |
| `/api/version` | Deployed version and build metadata JSON |

## Configuration boundaries

| File or directory | Responsibility |
|---|---|
| `server_config.yaml` | Production TCP/Web settings, SQLite, heatmaps, path mapping, model list, cleanup, training, and optional integrations |
| `server_config_local.yaml` | Windows/local profile with local ports and output paths |
| `configs/capi_3f.yaml` | Legacy/fallback model configuration, image-prefix mappings, thresholds, exclusion zones, bomb rules, and post-processing |
| `model/<machine>-<timestamp>/` | Bundles produced by the training workflow; each bundle contains its own model configuration and metadata |
| `VERSION` / `CHANGELOG.md` | Release identity and operator-facing change history |

Production `model_configs` should point to the bundle `machine_config.yaml` files that match the incoming `ModelID`. `configs/capi_3f.yaml` is retained for legacy/fallback use; it is not a substitute for installing the required model weights.

## Common development checks

Run the protocol smoke test without starting a listener:

```bash
python -X utf8 capi_server.py --test-protocol
```

Run the automated test suite from the repository root:

```bash
python -m pytest tests/
```

## Related documentation

- [Client communication protocol](./docs/client_communication_protocol.zh-TW.md)
- [New-machine model training SOP](./docs/new_system_model_training_sop.zh-TW.md)
- [PatchCore training architecture](./docs/patchcore_training_architecture.zh-TW.md)
- [Experimental pull-based update workflow](./docs/experimental_auto_update.zh-TW.md)
- [Deployment ZIP builder](./scripts/build_deploy_zip.py)
- [Central dashboard](./central_dashboard/README.md)
- [Change history](./CHANGELOG.md)

Internal project; not intended for public distribution.
