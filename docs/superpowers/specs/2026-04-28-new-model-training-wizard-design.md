# 新機種 PatchCore 模型訓練 Wizard — Design Spec

**Date:** 2026-04-28
**Status:** Approved (pending user spec review)

---

## Context

CAPI AI 目前在生產線上提供 1 個機種（CAPI 3F）的 5 個 lighting condition 模型，由 `tools/build_bga_tiles.py` + `tools/train_bga_all.py` 透過 SSH 手動跑。新機種上線時需重複此流程，門檻高、無法授權給操作員執行。

本 spec 定義一個 web 端 5 步驟 wizard，讓操作員直接從瀏覽器為新機種訓練 PatchCore 模型 bundle，並引入更精細的「inner / edge」分區模型架構（每機種 10 個模型），把訓練端的前處理與推論端對齊到 polygon-aligned 邏輯。

附帶解決幾個既存痛點：
- 訓練端與推論端前處理 drift（`BBOX_INSET=60` vs `otsu_offset=5`、template MARK match vs `relative_bottom_right`）。
- MARK / 易斯貼區域被排除導致邊界 tile 過殺。
- 邊緣 tile 光影差異被內部 tile 模型稀釋 specificity。

---

## Scope

### In scope

- **新 wizard**：5 步驟 `/train/new`（選機種 → 前處理 → tile review → 訓練 → 完成）
- **共用前處理模組 `capi_preprocess.py`**：把 polygon 偵測、tile 切分、zone 分類抽出，訓練 / 推論共用，徹底消除 drift
- **新模型架構（C-10）**：5 lighting × (inner + edge) = 10 個 PatchCore 模型 / 機種
- **MARK / 易斯貼進訓練**：不再 mask，依 tile 位置自然分流到 inner / edge 模型
- **模型庫管理頁 `/models`**：列出 bundle、可啟用 / 停用 / 刪除 / 匯出 ZIP
- **DB schema 擴充**：`training_jobs`、`model_registry`、`training_tile_pool` 三表
- **Per-機種 yaml 自動產出**：訓練完寫 `configs/capi_<machine_id>.yaml`
- **離線 backbone 預載**：訓練環境無外網，PatchCore backbone 預先 stage 到 `deployment/torch_hub_cache/`
- **Inference 端改造**：支援多機種 yaml 並存，依 request 的 `model_id` 選對應 config + lazy load 10 模型

### Out of scope

- **既有 5 模型遷移**：CAPI 3F yaml 維持，舊機種繼續走 legacy 5-model 路徑，本 spec 不重訓 3F
- **熱部署**：訓練完不會自動 push 到 inference；須由模型庫「匯出 ZIP」+ FTP 上傳 + 重啟
- **訓練取消**：PatchCore GPU 訓練無法中途安全中止
- **跨機種轉移學習**：每機種獨立訓 10 模型，不做 fine-tune from 舊機種
- **MARK template 維護**：完全移除依賴
- **多人並發訓練**：一個 server instance 一次只跑一個訓練 job，第二個請求回 409
- **Edge 4 向獨立模型**（C-25）：保留為未來升級路徑，本 spec 只實作 C-10
- **`/debug` 與 `/settings` 對新架構支援**：留 follow-up spec
- **GPU 記憶體 LRU eviction**：多機種共存時 GPU 可能吃緊，本 spec 不處理

### 與既有 `/retrain` 的關係

既有 `/retrain` 是**刮痕分類器（DINOv2 LoRA 微調）**的重訓練入口，與本功能完全獨立、共存。`/training` hub 頁將從目前 1 張卡擴充為 2 張卡：

| Card | Target | 用途 |
|------|--------|------|
| 刮痕分類器 | `/retrain` | 既有，不動 |
| 新機種 PatchCore | `/train/new` + `/models` | 本 spec 新增 |

---

## Key Decisions Summary

下列澄清問答的最終決定，影響後續所有設計細節：

| # | 決定點 | 選項 | 理由 |
|---|--------|------|------|
| 1 | 模型架構 | **C-10**（5 inner + 5 edge per lighting） | C-25 在 5 片 panel 下左右邊每模型只 4 個 coreset sample，PatchCore 失靈區；C-10 specificity 已比舊 5 模型大幅提升，未來可漸進升級 |
| 2 | MARK / 易斯貼處理 | **進訓練、推論不再 mask、依 tile 位置 routing** | MARK 靠內 → 自然落入 inner 模型；易斯貼右下邊 → 落入 edge 模型；不需特殊 fixture 模型 |
| 3 | Tile review UI | **5 lighting tab，每 tab 顯示 inner/edge × OK/NG 4 sub-group** | 平衡進度感與單頁長度 |
| 4 | 「五個畫面」定義 | **per-panel 過濾規則**（去 S* / B0F / PINIGBI，留 G0F / R0F / W0F / WGF / STANDARD） | 配合 AOI 機台 panel folder 結構 |
| 5 | Panel 預設數量 | **5 片** | 內部 model 充足，邊緣 model 進入 PatchCore 健康區（coreset bank ~20-40） |
| 6 | NG 來源 | **`over_review/{*}/true_ng/{lighting}/crop/` 跨機種共用** | 既有刮痕複判已切好 tile、按 lighting 分組 |
| 7 | Step 1 panel picker | **DB 查 `inference_records` AOI 判 OK** | DB 已記錄 AOI 判定 |
| 8 | 訓練超參數 | **隱藏，全部用預設** | 操作員無需調參，預設值 = `train_bga_all.py` 既有值 |
| 9 | 模型庫動作 | **檢視 + 啟用 + 停用 + 刪除 + 匯出 ZIP** | 對應 FTP-only 部署模型 |
| 10 | 既有 5 模型遷移 | **慢慢遷移，本 spec 不處理** | 維持 production 穩定 |
| 11 | 機種設定存放 | **`model/<bundle>/machine_config.yaml`（bundle 內，wizard 自動生成）** | 自動隨 bundle 版控；`server_config.yaml.model_configs` 直接指向 bundle 內 yaml；舊 `configs/capi_3f.yaml` legacy 路徑維持 |
| 12 | edge_threshold_px | **預設 768px，yaml 可調，wizard 進階模式可改** | 約 1.5 tile 寬度 |
| 13 | Panel orientation / 易斯貼位置 | **AOI 機台保證固定** | 簡化前處理（無需 orientation 偵測） |
| 14 | Backbone | **預載至 `deployment/torch_hub_cache/`** | 訓練環境無外網 |

---

## Architecture

### 元件分布

```
┌──────────────────────────────────────────────────────────────┐
│  Web Wizard UI  (Jinja2 templates)                            │
│    templates/train_new/{step1_select,step2_progress,           │
│                         step3_review,step4_progress,           │
│                         step5_done}.html                       │
│    templates/models.html                                       │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP routes
┌────────────────────────▼─────────────────────────────────────┐
│  capi_web.py（修改）                                           │
│    /train/new/*       wizard 路由                              │
│    /models/*          模型庫路由                               │
│    /api/train/new/*   wizard API                               │
│    /api/models/*      模型庫 API                               │
└──────────┬──────────────────────────────────────────────────┘
           │
   ┌───────┴────────────────────────┬──────────────────────┐
   ▼                                ▼                      ▼
┌──────────────────┐   ┌─────────────────────┐   ┌──────────────────┐
│ capi_preprocess  │   │ capi_train_new      │   │ capi_model       │
│ (新, 純函式)      │   │ (新, threading)      │   │ _registry (新)    │
│                  │   │                     │   │                  │
│ - Otsu + polygon │   │ - 10-model 序列訓練  │   │ - list/activate/ │
│ - tile 分類      │   │ - threshold 校準     │   │   delete/export  │
│ - multi-lighting │   │ - bundle 輸出        │   │ - DB CRUD        │
│   一致性         │   │ - yaml 自動產出      │   │                  │
└──┬───────────────┘   └──┬──────────────────┘   └──────────────────┘
   │ 共用                 │
   ▼                      ▼
┌────────────────────┐  ┌─────────────────┐
│ capi_inference.py  │  │ tools/          │
│ (修改)              │  │   train_new_    │
│ - 移除 MARK / excl │  │   model.py (新)  │
│ - inner/edge route │  │   CLI 入口       │
└────────────────────┘  └─────────────────┘
```

### DB Schema 擴充（沿用既有 SQLite，WAL mode）

```sql
-- Wizard 訓練 job state
CREATE TABLE training_jobs (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id          TEXT UNIQUE,            -- "train_<machine_id>_<yyyymmdd>_<hhmmss>"
  machine_id      TEXT NOT NULL,
  state           TEXT NOT NULL,          -- pending|preprocess|review|train|completed|failed
  started_at      TEXT,
  completed_at    TEXT,
  panel_paths     TEXT,                   -- JSON: 5 個 glass folder 路徑
  output_bundle   TEXT,                   -- bundle 路徑
  error_message   TEXT
);

-- 模型庫主表
CREATE TABLE model_registry (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  machine_id        TEXT NOT NULL,
  bundle_path       TEXT UNIQUE NOT NULL,
  trained_at        TEXT NOT NULL,
  panel_count       INTEGER,
  inner_tile_count  INTEGER,
  edge_tile_count   INTEGER,
  ng_tile_count     INTEGER,
  bundle_size_bytes INTEGER,
  is_active         INTEGER DEFAULT 0,    -- 1 = 當前 inference yaml 指向此 bundle
  job_id            TEXT,
  notes             TEXT
);

-- Tile review 暫存（job 完成後可清）
CREATE TABLE training_tile_pool (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id      TEXT NOT NULL,
  lighting    TEXT NOT NULL,              -- G0F00000 / R0F00000 / W0F00000 / WGF50500 / STANDARD
  zone        TEXT NOT NULL,              -- inner / edge
  source      TEXT NOT NULL,              -- ok / ng
  source_path TEXT NOT NULL,              -- 切下來的 tile 檔絕對路徑
  thumb_path  TEXT,                       -- 縮圖
  decision    TEXT DEFAULT 'accept',      -- accept / reject (preprocess 時預設 accept，user reject 才改)
  FOREIGN KEY (job_id) REFERENCES training_jobs(job_id)
);
CREATE INDEX idx_tile_pool_job ON training_tile_pool(job_id, lighting, zone, source);
```

### Bundle 儲存

```
model/
├── GN160JCEL250S-20260428/                  # 一機種一 dated 資料夾
│   ├── manifest.json
│   ├── thresholds.json
│   ├── machine_config.yaml                  # 此 bundle 的 inference yaml（自動生成）
│   ├── G0F00000-inner.pt
│   ├── G0F00000-edge.pt
│   ├── R0F00000-inner.pt
│   ├── R0F00000-edge.pt
│   ├── W0F00000-inner.pt
│   ├── W0F00000-edge.pt
│   ├── WGF50500-inner.pt
│   ├── WGF50500-edge.pt
│   ├── STANDARD-inner.pt
│   └── STANDARD-edge.pt
└── CAPI-BGA-20260417/                       # 既有 3F bundle 不動
    └── (5 個既有 .pt)
```

「啟用版本」由 `server_config.yaml.model_configs` 列表決定（指向某 bundle 的 `machine_config.yaml`）。同機種多 bundle 共存時只有被列入的 yaml 會生效；切版 = 改 model_configs entry 從舊 bundle yaml → 新 bundle yaml。

### Bundle 內機種 yaml `model/<bundle>/machine_config.yaml`

```yaml
# 由 wizard 訓練完成後自動生成於 bundle 目錄內
machine_id: GN160JCEL250S
trained_at: 2026-04-28T15:30:45
bundle_path: model/GN160JCEL250S-20260428

# 前處理參數（沿用 capi_preprocess 預設，可調）
edge_threshold_px: 768
otsu_offset: 5
enable_panel_polygon: true

# Inner / edge 模型對應
model_mapping:
  G0F00000:
    inner: model/GN160JCEL250S-20260428/G0F00000-inner.pt
    edge:  model/GN160JCEL250S-20260428/G0F00000-edge.pt
  R0F00000:
    inner: ...
    edge:  ...
  # ... W0F00000 / WGF50500 / STANDARD

threshold_mapping:
  G0F00000:
    inner: 0.62
    edge:  0.71
  # ...

# 新架構不再使用以下（與 capi_3f.yaml 的差異）：
#   mark_template_path / mark_match_threshold / mark_min_y_ratio
#   mark_fallback_position / exclusion_zones / bottom_right_mechanism
```

### `server_config.yaml` 擴充

```yaml
model_configs:
  - configs/capi_3f.yaml                                       # legacy（路徑不變）
  - model/GN160JCEL250S-20260428/machine_config.yaml          # 新架構，指向 bundle 內 yaml
fallback_model_config: configs/capi_3f.yaml  # model_id 找不到對應 yaml 時用

training:
  backbone_cache_dir: deployment/torch_hub_cache
  required_backbones:
    - wide_resnet50_2-32ee1156.pth
```

### HTTP Routes 新增

| Method | Path | Handler | 用途 |
|--------|------|---------|------|
| GET | `/train/new` | `_handle_train_new_page` | Step 1 頁（機種選擇 + panel picker） |
| GET | `/api/train/new/panels?machine_id=X&days=7` | `_handle_train_new_panels` | DB 查 AOI OK panel 清單 |
| POST | `/api/train/new/start` | `_handle_train_new_start` | 建 job、觸發 step 2 thread |
| GET | `/api/train/new/status?job_id=X` | `_handle_train_new_status` | 取 job state + log |
| GET | `/train/new/review/<job_id>` | `_handle_train_new_review_page` | Step 3 頁（tile review） |
| GET | `/api/train/new/tiles?job_id=X&lighting=Y` | `_handle_train_new_tiles` | 取 tile pool |
| POST | `/api/train/new/tiles/decision` | `_handle_train_new_tiles_decision` | 批次更新 tile decision |
| POST | `/api/train/new/start_training/<job_id>` | `_handle_train_new_start_training` | 觸發 step 4 thread |
| GET | `/train/new/done/<job_id>` | `_handle_train_new_done_page` | Step 5 頁 |
| GET | `/models` | `_handle_models_page` | 模型庫頁 |
| GET | `/api/models?machine_id=X` | `_handle_models_list` | 列 bundle |
| GET | `/api/models/<bundle_id>/detail` | `_handle_models_detail` | manifest + thresholds + job |
| POST | `/api/models/<bundle_id>/activate` | `_handle_models_activate` | 啟用（改 server_config） |
| POST | `/api/models/<bundle_id>/deactivate` | `_handle_models_deactivate` | 停用 |
| POST | `/api/models/<bundle_id>/delete` | `_handle_models_delete` | 刪除 bundle |
| GET | `/api/models/<bundle_id>/export` | `_handle_models_export` | 串流 ZIP |

`/training` hub 頁更新：`_build_training_cards()` 新增「新機種 PatchCore」卡。

---

## Wizard 5-Step Flow

### Step 1 — 選機種 + Panel picker

**頁面**：`templates/train_new/step1_select.html`

組件：
- 機種 ID 輸入（autocomplete from `inference_records.model_id` distinct）
- 時間範圍 pill：1 天 / 7 天 / 30 天
- Panel 表格：勾 5 片 AOI OK panel，欄位包含 Glass ID / 日期時間 / 機台 / STANDARD 縮圖 / AI 判定
- 「下一步：前處理 →」disabled until 選滿 5 片

**API**

`GET /api/train/new/panels?machine_id=X&days=7`
```sql
SELECT glass_id, machine_no, image_path, ai_judgment, timestamp
FROM inference_records
WHERE model_id = ?
  AND machine_judgment = 'OK'
  AND timestamp >= datetime('now', '-7 days')
ORDER BY timestamp DESC
LIMIT 100
```

`POST /api/train/new/start` body:
```json
{
  "machine_id": "GN160JCEL250S",
  "panel_paths": [
    "/CAPI07/TIANMU/yuantu/GN160JCEL250S/20260428/YQ21KU218E45",
    "..."
  ]
}
```
回 `{job_id}`。建 `training_jobs` row（state=preprocess）+ 觸發背景 thread 跑 step 2。

### Step 2 — 前處理 + 切 tile（背景 thread）

**頁面**：`templates/train_new/step2_progress.html`，每 3 秒 poll `/api/train/new/status`。

**Backend thread**（`capi_train_new._preprocess_worker`）

```
For 每片 panel folder：
  1. capi_preprocess.filter_panel_lighting_files(folder) → {lighting: path}
  2. STANDARD 先處理 → reference_polygon
     若 STANDARD 偵測失敗 → fallback G0F00000
     若仍失敗 → 該 panel 紀錄 error 跳過
  3. 其他 4 lighting 套 reference_polygon → 切 tile
  4. 對每張 tile：classify zone (inner/edge/outside)
     outside → drop
     存縮圖 + 路徑進 training_tile_pool（source=ok）

從 over_review/{*}/true_ng/{lighting}/crop/ 隨機抽：
  per lighting 30 個 tile → training_tile_pool（source=ng, zone=null）
  注意：NG 不分 inner/edge，inner 與 edge 模型共用同一份 NG set 做 test/anormal

完成：training_jobs.state = 'review'
```

**Edge cases**
- Polygon 偵測失敗的 panel：log warning、跳過
- 至少 4 片成功才繼續，否則 job state = 'failed'
- NG 抽取若 over_review 沒對應 lighting：跳過、log warning

### Step 3 — Tile review

**頁面**：`templates/train_new/step3_review.html`

組件：
- 5 個 lighting tab（顯示 `已決定 / 總數` badge）
- 每 tab 內 4 個 sub-group：內部 OK / 內部 NG / 邊緣 OK / 邊緣 NG（每組標題顯示 `accept / reject` 計數）
- 每組是 tile 縮圖 grid（12 column），每個 tile 兩狀態：accept（綠框✓）/ reject（紅 X 半透明）
- 點 tile 切換 accept↔reject；Shift-click 多選範圍
- 過濾條件：全部 / 已加入 / 已丟棄
- 批次按鈕：本組全加入 / 本組全丟棄
- 顯示 `已加入 / 已丟棄` 計數（不是 review 進度，因為預設全 accept 沒有「未決定」狀態）
- 「上一步」「開始訓練 →」按鈕
- **預設值**：所有 tile 開始即為 `accept`（preprocess 完寫 DB 時就是 accept）
- **無強制 gate**：user 可以直接點訓練，預設行為等同接受全部 tile；責任在 user 決定要不要丟棄不適合的

**API**

```
GET /api/train/new/tiles?job_id=X&lighting=Y
  → JSON list of tiles，含 thumb_path / decision / source / zone

POST /api/train/new/tiles/decision
  body: { "tile_ids": [...], "decision": "accept" | "reject" }
  → bulk update training_tile_pool.decision

POST /api/train/new/start_training/<job_id>
  → 觸發 step 4 thread，state=train，回 {ok}
```

### Step 4 — 10-model 序列訓練

**頁面**：`templates/train_new/step4_progress.html`（仿 `/retrain` 風格）

組件：
- 10 步驟 progress chip（`G0F-inner` → `G0F-edge` → ... → `STANDARD-edge` → `完成`）
- Elapsed timer + 預計時間
- 訓練 log box（捲動）
- 完成後跳 step 5

**Backend thread**（`capi_train_new._training_worker`）

```python
# 啟動前自檢
assert backbone_cache_exists()       # deployment/torch_hub_cache/
assert torch.cuda.is_available()
os.environ["TORCH_HOME"] = backbone_cache_dir
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

TRAINING_UNITS = [
    ("G0F00000", "inner"), ("G0F00000", "edge"),
    ("R0F00000", "inner"), ("R0F00000", "edge"),
    ("W0F00000", "inner"), ("W0F00000", "edge"),
    ("WGF50500", "inner"), ("WGF50500", "edge"),
    ("STANDARD", "inner"), ("STANDARD", "edge"),
]

for idx, (lighting, zone) in enumerate(TRAINING_UNITS, 1):
    train_tiles = query_pool(job_id, lighting, zone, source="ok", decision="accept")
    test_tiles  = query_pool(job_id, lighting, source="ng", decision="accept")
    
    if len(train_tiles) < 30:  # MIN_TRAIN_TILES
        log("WARN: skip {lighting}-{zone}, only {len(train_tiles)} tiles")
        continue
    
    with self._capi_server_instance._gpu_lock:
        staging = stage_dataset(train_tiles, test_tiles)  # mklink/symlink
        model_pt = train_one_patchcore(staging, lighting, zone)
        threshold = calibrate_threshold(model_pt, test_tiles)
    
    save_model(model_pt, bundle_root / f"{lighting}-{zone}.pt")
    save_threshold(threshold, bundle_root / "thresholds.json", lighting, zone)
    cleanup_staging(staging)

# 至少 5 unit 成功才 mark completed
write_manifest(bundle_root, job_meta)
write_machine_yaml(bundle_root, machine_id)
register_in_db(bundle_root)
training_jobs.state = "completed"
```

**PatchCore 超參數（hard-coded，沿用 `train_bga_all.py`）**

```python
BATCH_SIZE = 8
IMAGE_SIZE = (512, 512)
CORESET_RATIO = 0.1
MAX_EPOCHS = 1
NUM_WORKERS = 0
VAL_SPLIT_MODE = "same_as_test"
```

**Threshold 校準**

```python
def calibrate_threshold(model, ng_tiles):
    scores = sorted(model.predict(t.image)["pred_score"] for t in ng_tiles)
    threshold = max(
        scores[int(len(scores) * 0.10)],   # P10 of NG scores
        train_max_score * 1.05,             # 高於訓練 OK 最大 score 5%
    )
    return threshold
```

存 `thresholds.json`：
```json
{
  "G0F00000": {"inner": 0.62, "edge": 0.71},
  "R0F00000": {"inner": 0.59, "edge": 0.68},
  ...
}
```

**Stage 資料（避免複製檔）**

```python
def make_dataset_link(link, target):
    if platform.system() == "Windows":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True)
    else:
        link.symlink_to(target, target_is_directory=True)
```

Staging 結構：
```
.tmp/training_staging/<job_id>/<lighting>-<zone>/
├── train/                # symlink → 加入 tile 的目錄
└── test/anormal/         # symlink → NG tile 目錄
```

訓練後 cleanup 整個 `.tmp/training_staging/<job_id>/`。

### Step 5 — 完成

**頁面**：`templates/train_new/step5_done.html`

組件：
- 機種 / Bundle / 訓練時間
- 10 個子模型摘要表（lighting / zone / train tile / NG tile / threshold / size）
- 部署提示：「到模型庫頁面點啟用 + 匯出 ZIP → FTP 上傳」
- 「前往模型庫 →」「回首頁」按鈕

### Wizard 中斷恢復

| 中斷時機 | 恢復行為 |
|---|---|
| Step 1 → 2 中關 browser | Job 已建、背景持續跑；user 重開 `/train/new` 提示「進行中：<job_id>，繼續？」 |
| Step 3 review 中 | tile decision 即時寫 DB，重開 page 不丟資料 |
| Step 4 訓練中 | 背景 thread 持續，重開看 log |
| Step 5 後 | 隨時可從 `/models` 取 bundle |

`/train/new` 入口邏輯：
- 無 active job → 直接顯示 step 1
- 有 active job → 提示「進行中：<job_id> 在 step <X>，繼續 / 拋棄並開新的」

---

## Shared Preprocess Module — `capi_preprocess.py`

從 `capi_inference.py` 抽出，訓練 / 推論共用，根除前處理 drift。

### Public API

```python
@dataclass
class PreprocessConfig:
    tile_size: int = 512
    tile_stride: int = 512
    otsu_offset: int = 5
    enable_panel_polygon: bool = True
    edge_threshold_px: int = 768       # tile center 距 polygon 邊 < 此值 → edge
    coverage_min: float = 0.3          # tile 在 polygon 內覆蓋率下限

@dataclass
class TileResult:
    tile_id: int
    x1: int; y1: int; x2: int; y2: int  # 圖片座標
    image: np.ndarray                    # 512×512 cropped tile
    mask: Optional[np.ndarray]           # polygon mask in tile, None if fully inside
    coverage: float                      # 0.0–1.0
    zone: str                            # "inner" | "edge" | "outside"
    center_dist_to_edge: float           # px

@dataclass
class PanelPreprocessResult:
    image_path: Path
    lighting: str
    foreground_bbox: Tuple[int, int, int, int]
    panel_polygon: Optional[np.ndarray]  # (4,2) float32 TL/TR/BR/BL
    tiles: List[TileResult]              # 不含 outside tile
    polygon_detection_failed: bool

def filter_panel_lighting_files(folder: Path) -> Dict[str, Path]:
    """過濾出 5 個有效 lighting 圖。
    跳過 S* (側拍) / B0F (黑屏) / PINIGBI / OMIT / Optics.log。
    """

def detect_panel_polygon(
    image: np.ndarray, config: PreprocessConfig
) -> Tuple[Tuple[int,int,int,int], Optional[np.ndarray]]:
    """Otsu binarize → 最大連通輪廓 bbox → polyfit 4 角 polygon。"""

def classify_tile_zone(
    tile_rect: Tuple[int,int,int,int],
    polygon: Optional[np.ndarray],
    config: PreprocessConfig,
) -> Tuple[str, float, float, Optional[np.ndarray]]:
    """Returns (zone, coverage, center_dist, mask).
    polygon=None → zone='inner' fallback (no edge differentiation).
    """

def preprocess_panel_image(
    image_path: Path,
    lighting: str,
    config: PreprocessConfig,
    reference_polygon: Optional[np.ndarray] = None,
) -> PanelPreprocessResult:
    """完整 pipeline。reference_polygon 給多 lighting 對齊。"""

def preprocess_panel_folder(
    folder: Path, config: PreprocessConfig
) -> Dict[str, PanelPreprocessResult]:
    """呼叫 filter_panel_lighting_files → STANDARD 先取 reference → 套用其他 4 lighting。"""
```

### Edge cases

| 情況 | 行為 |
|---|---|
| Polygon 偵測失敗 | `polygon_detection_failed=True`，所有 tile 視為 inner（無分流），呼叫端決定是否跳 panel |
| Reference polygon 套到別 lighting 對位偏差 | 不做修正（assume AOI panel 不動）；future issue 再處理 |
| Tile coverage < 0.3 | `zone='outside'`，filter 掉不返回 |
| Tile coverage 0.3–1.0 | `zone='edge'`，附 polygon mask |
| Tile fully inside 且距邊 ≥ 768px | `zone='inner'`，mask=None |
| Tile fully inside 但距邊 < 768px | `zone='edge'`，mask=None |

### 既有 polygon 測試搬遷

```
tests/test_panel_polygon_unit.py        ← import path 改為 capi_preprocess
tests/test_panel_polygon_detect.py      ← 同上
tests/test_panel_polygon_tile_mask.py   ← 同上
tests/test_capi_preprocess_unit.py      ← 新增（classify_tile_zone / filter_panel_lighting_files 單元）
tests/test_capi_preprocess_integration.py ← 新增（fixture image end-to-end）
```

---

## Inference 端修改細節（`capi_inference.py`）

### 多機種 yaml 載入

`capi_server.py` 啟動時：
```python
configs_by_machine = {}
for cfg_path in server_cfg["model_configs"]:
    cfg = CAPIConfig.from_yaml(cfg_path)
    machine_id = cfg.machine_id or "3F"  # 舊 yaml 無 machine_id 用 "3F"
    configs_by_machine[machine_id] = cfg

fallback_cfg = CAPIConfig.from_yaml(server_cfg["fallback_model_config"])
```

### Per-request 配置選擇

```python
def _process_request(req):
    cfg = configs_by_machine.get(req.model_id, fallback_cfg)
    inferencer = self._get_inferencer(req.model_id, cfg)  # 機種獨立 inferencer 實例
    return inferencer.process_panel(req.image_dir)
```

### CAPIInferencer 雙路徑

```python
class CAPIInferencer:
    def __init__(self, config: CAPIConfig, ...):
        self.config = config
        if config.is_new_architecture:
            self._process_panel_impl = self._process_panel_v2
        else:
            self._process_panel_impl = self._process_panel_v1  # 既有 5-model 路徑
    
    def _process_panel_v2(self, panel_dir):
        results = preprocess_panel_folder(panel_dir, self._build_preprocess_cfg())
        for lighting, result in results.items():
            for tile in result.tiles:
                model = self._get_model_for(self.config.machine_id, lighting, tile.zone)
                threshold = self.config.threshold_mapping[lighting][tile.zone]
                # ... PatchCore predict + dust filter + bomb check + heatmap
```

`is_new_architecture` 判斷：yaml 含 `machine_id` 且 `model_mapping[lighting]` 是 `{inner, edge}` dict。

### Lazy loading

```python
self._model_cache = {}  # key: (machine_id, lighting, zone)

def _get_model_for(self, machine_id, lighting, zone):
    key = (machine_id, lighting, zone)
    if key not in self._model_cache:
        path = self.config.model_mapping[lighting][zone]
        self._model_cache[key] = TorchInferencer(path)
    return self._model_cache[key]
```

### MARK / Exclusion 移除

新架構 yaml 不含 MARK 系列欄位。`CAPIConfig` 載入時：
- 新架構 path：`mark_template_path = None`、`exclusion_zones = []`
- 既有 path：保留所有欄位不動

`process_panel_v2` 內：
- 不呼叫 `find_mark_region` / `calculate_exclusion_regions`
- 不載入 `mark_template`

`process_panel_v1`（legacy 5-model）：完全不動。

### Tile mask 處理（edge tile）

```python
if tile.zone == "edge" and tile.mask is not None:
    # tile 部分跨 polygon 邊界
    score_map = model.predict(tile.image)["anomaly_map"]
    score_map = score_map * (tile.mask / 255.0)  # polygon 外像素強制 0
    pred_score = float(score_map.max())
```

### 後處理流程兼容

新架構不影響：
- Dust filter（OMIT cross-validation）
- Bomb defect check
- Edge margin decay（保留作 second-line defense，與 edge model 不衝突）
- Scratch classifier post-filter

這些後處理在 `process_panel` 後段，與前處理 / model routing 解耦。

### 健檢

Server 啟動時：
```python
for cfg in configs_by_machine.values():
    if cfg.is_new_architecture:
        for lighting, mapping in cfg.model_mapping.items():
            for zone in ("inner", "edge"):
                assert Path(mapping[zone]).exists(), f"模型缺檔: {mapping[zone]}"
```

啟動失敗 fail-fast，避免 inference runtime 才發現缺檔。

---

## Model Registry 細節

### 頁面 `templates/models.html`

兩區塊：
1. **新架構（10 模型 C-10）**：列出所有新架構 bundle，每筆可全套操作
2. **舊架構（5 模型 Legacy 3F）**：唯讀顯示，不能從 wizard 操作（避免誤動 production）

每筆 bundle card 顯示：
- 機種 / bundle 路徑 / 訓練時間 / job ID
- Panel 數 / Inner+Edge+NG tile 數 / 訓練時間
- Bundle 大小 / threshold 範圍
- 對應 yaml 路徑
- 啟用狀態（綠點 + 「啟用中」標籤）
- 動作：📋 細節 / ▶ 啟用 或 ⏸ 停用 / 📦 匯出 ZIP / 🗑 刪除

### 啟用 / 停用

```python
def activate_bundle(bundle_id):
    bundle = db.get_bundle(bundle_id)
    yaml_path = Path(bundle.bundle_path) / "machine_config.yaml"
    
    # 1. 同 machine 其他 bundle 從 server_config 移除（並 set is_active=0）
    for other in db.get_bundles_for_machine(bundle.machine_id, except_id=bundle_id):
        other_yaml = Path(other.bundle_path) / "machine_config.yaml"
        remove_from_server_config_model_configs(other_yaml)
        db.set_active(other.id, False)
    
    # 2. 加入此 bundle 的 yaml 到 server_config.model_configs
    add_to_server_config_model_configs(yaml_path)
    db.set_active(bundle_id, True)
    
    return {"ok": True, "message": "啟用成功，請重啟 server 才會生效"}

def deactivate_bundle(bundle_id):
    bundle = db.get_bundle(bundle_id)
    yaml_path = Path(bundle.bundle_path) / "machine_config.yaml"
    remove_from_server_config_model_configs(yaml_path)
    db.set_active(bundle_id, False)
    return {"ok": True, "message": "已停用，請重啟 server 才會生效"}
```

**不**自動重啟 server（避免影響進行中的 inference）。

### 刪除安全

```python
def delete_bundle(bundle_id):
    bundle = db.get_bundle(bundle_id)
    if bundle.is_active:
        return error("請先停用才能刪除", 409)
    # 確認 modal（前端處理）
    yaml_path = Path(bundle.bundle_path) / "machine_config.yaml"
    remove_from_server_config_model_configs(yaml_path)  # 保險：若 server_config 還掛著，移除
    shutil.rmtree(bundle.bundle_path)                   # 移整個 bundle 目錄（含 yaml）
    db.delete_bundle(bundle_id)
```

### 匯出 ZIP 內容

```
GN160JCEL250S-20260428.zip
└── model/GN160JCEL250S-20260428/   # 整個 bundle 目錄（含 yaml）
    ├── manifest.json
    ├── thresholds.json
    ├── machine_config.yaml          # 此 bundle 的 inference yaml
    ├── *.pt  (10 個)
    └── README.txt                    # FTP 部署說明
```

`README.txt` 內容範本：
```
新機種 PatchCore Bundle 部署說明
────────────────────────────────────────
機種：GN160JCEL250S
訓練時間：2026-04-28 15:30:45

部署步驟：
1. 解壓本 ZIP，保留路徑結構
2. FTP 上傳整個 bundle 目錄到 production：
     model/GN160JCEL250S-20260428/  → /capi_ai/model/GN160JCEL250S-20260428/
3. 編輯 production 的 server_config.yaml，在 model_configs 列表加入：
     - model/GN160JCEL250S-20260428/machine_config.yaml
4. （可選）若同機種有舊 bundle 想停用，從 model_configs 移除舊 bundle 的 yaml
5. 重啟 capi_server 服務

驗證：傳送該機種 panel 給 inference，confirm 走新架構（log 應顯示 "load 10 models"）。
```

---

## Files Modified / Added

### 新增

| 檔案 | 用途 |
|------|------|
| `capi_preprocess.py` | 共用前處理（polygon、tile classify、multi-lighting 一致性） |
| `capi_train_new.py` | Wizard 訓練 worker（preprocess + training threading） |
| `capi_model_registry.py` | 模型庫 CRUD、啟用 / 停用、ZIP 匯出 |
| `tools/train_new_model.py` | CLI 入口（wrap `capi_train_new` 提供 SSH 跑法） |
| `templates/train_new/step1_select.html` | Step 1 |
| `templates/train_new/step2_progress.html` | Step 2 |
| `templates/train_new/step3_review.html` | Step 3 |
| `templates/train_new/step4_progress.html` | Step 4 |
| `templates/train_new/step5_done.html` | Step 5 |
| `templates/models.html` | 模型庫 |
| `tests/test_capi_preprocess_unit.py` | preprocess 單元測試 |
| `tests/test_capi_preprocess_integration.py` | preprocess 整合測試 |
| `tests/test_capi_train_new.py` | wizard worker 單元 / 整合測試 |
| `tests/test_capi_model_registry.py` | 模型庫 CRUD 測試 |
| `docs/superpowers/specs/2026-04-28-new-model-training-wizard-design.md` | 本 spec |

### 修改

| 檔案 | 變更 |
|------|------|
| `capi_inference.py` | 移除 MARK / exclusion / polygon / tile 內部邏輯（改 import `capi_preprocess`）；新增 `_process_panel_v2` 雙路徑；inner/edge model lazy load |
| `capi_config.py` | `CAPIConfig` 新增 `machine_id` / `is_new_architecture` 欄位；支援 `model_mapping` value 為 dict 結構 |
| `capi_server.py` | 啟動時載多個 yaml 進 `configs_by_machine`；request dispatcher 依 model_id 選 config |
| `capi_database.py` | 新增 3 表（training_jobs / model_registry / training_tile_pool） + 對應 CRUD 方法 |
| `capi_web.py` | 新增 wizard / 模型庫路由（共 ~16 個 route handler）；`_build_training_cards()` 加新機種卡 |
| `tools/build_bga_tiles.py` | 廢棄（標 deprecated 但保留至 3F 重訓完成）|
| `tools/train_bga_all.py` | 廢棄（同上） |
| `server_config.yaml` / `server_config_local.yaml` | 加 `model_configs` 列表 + `training.backbone_cache_dir` |
| `tests/test_panel_polygon_*.py` | import path 從 `capi_inference` 改為 `capi_preprocess` |

### 廢棄但保留

`build_bga_tiles.py` 與 `train_bga_all.py` 不刪除，直到 CAPI 3F 完成遷移到新架構。

---

## Verification

### 開發機驗證流程

1. 啟動：`python capi_server.py --config server_config_local.yaml`
2. 開瀏覽器 `http://localhost:8080`
3. 從 sidebar 進「模型訓練」→ 看到兩張卡（刮痕 + 新機種）
4. 點「新機種 PatchCore」→ 進 `/train/new` step 1
5. 輸入測試機種 ID → 看到 panel 清單
6. 勾 5 片 → 下一步
7. Step 2 進度頁顯示 panel 處理 log
8. Step 3 tile review，預設全 accept；隨意點幾個 reject
9. 點「開始訓練」→ Step 4 看 10 個模型序列訓練
10. Step 5 完成頁，顯示 10 個模型摘要
11. 進 `/models`，看到新 bundle，「啟用」→ 提示重啟
12. 「匯出 ZIP」→ 下載 ZIP，解開驗證內容
13. 重啟 server，傳測試機種 panel，confirm 走新架構（log "load 10 models for <machine_id>"）

### Unit / Integration 測試

```bash
# 共用前處理
pytest tests/test_capi_preprocess_unit.py
pytest tests/test_capi_preprocess_integration.py

# 既有 polygon 測試（搬遷後仍 pass）
pytest tests/test_panel_polygon_unit.py
pytest tests/test_panel_polygon_detect.py
pytest tests/test_panel_polygon_tile_mask.py

# 新訓練 worker
pytest tests/test_capi_train_new.py

# 模型庫
pytest tests/test_capi_model_registry.py
```

### 邊界情況驗證

| 情境 | 預期行為 |
|---|---|
| Step 4 訓練中 server crash | Job state 卡在 train，重啟後從 `/train/new` 入口看到「進行中」提示，可拋棄 |
| Polygon 偵測失敗 panel ≥ 2 片 | Step 2 fail，提示「太多 panel 處理失敗，請換批」 |
| 訓練單元 < 30 tile | 該 unit skip 但繼續，最終至少 5 unit 才算 completed |
| 啟用後未重啟就送 inference | 仍走舊 yaml，inference 不變（不會 break） |
| 匯出 ZIP 過程中刪除 bundle | ZIP 已開始串流不受影響；後續刪除動作觸發 ROLLBACK |
| 兩個 wizard job 同時提交 | 第二個回 409 |

---

## Open Decisions（需 follow-up spec）

- **`/debug` 對新架構支援**：要支援單張對新機種跑 debug inference + heatmap
- **`/settings` 對新架構支援**：runtime config_params 套用到新架構 yaml
- **3F 機種遷移到新架構**：何時、誰主導、是否需要 A/B 比對 spec
- **Edge 4 向獨立模型升級（C-25）**：上線後若發現某邊系統性過殺，再針對性拆
- **GPU 記憶體 LRU eviction**：多機種同時上線後再評估
- **訓練取消機制**：未來考慮 epoch-level checkpoint，但 PatchCore 1 epoch 即收斂可能不適用

---

## References

- `docs/superpowers/specs/2026-04-10-panel-polygon-mask-design.md` — Polygon 偵測既有 spec
- `docs/superpowers/specs/2026-04-24-retrain-ui-design.md` — 既有 `/retrain` 設計（threading 模式參考）
- `tools/build_bga_tiles.py` / `tools/train_bga_all.py` — 既有訓練腳本（將被取代）
- `CLAUDE.md` 「Critical Gotchas」— TRUST_REMOTE_CODE / Path mapping / lazy loading 等慣例
- `CAPI_FLOW.md` — 既有推論流程
