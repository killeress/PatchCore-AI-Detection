# 過檢資料蒐集工具 設計規格

**日期**: 2026-04-10
**相關問題**: RIC 指標顯示漏檢趨近於 0，但過檢仍偏高，需蒐集實際過檢樣本以評估是否導入分類 AI 模型
**作者**: Ray (brainstorming with Claude)

---

## 1. 目標與背景

CAPI AI 目前透過 RIC 交叉驗證指標追蹤誤判情況。漏檢（AI=OK 但 RIC=NG）已趨近於 0，但**過檢**（AI=NG 但 RIC=OK）仍然偏高，影響產線效率。

為了決定是否導入**分類 AI 模型**或改用 CV 後處理來降低過檢，需要先蒐集實際過檢樣本作為分析與訓練用資料集。工廠使用者已開始對過檢樣本回填「過檢原因」，但這些資料目前散落在 DB 的 `over_review` 表中，缺少結構化的圖像資料集供後續分析。

本工具為**可重複執行的過檢資料蒐集器**，整合到現有 Web Dashboard，持續累積訓練集。

## 2. 設計決策摘要

| 決策點 | 選擇 | 理由 |
|---|---|---|
| 未回填 Review 的過檢是否蒐集 | **不蒐集**，只抓已回填 `over_review.category` 的 | 未回填的無法歸類，等填完下次再跑 |
| 一張 image 多 NG tile | **每個 NG tile 都截**，一張 image 多樣本 | 樣本最大化，同 image 不同 tile 特徵可能不同 |
| 目錄結構 | **先分 label 再分光源 prefix**（每葉下 `heatmap/` + `crop/`）| 光源特性差異大，分層便於後續單獨訓練 |
| 執行方式 | **Web Dashboard 整合**：`/ric` 過檢 Review tab 新增匯出按鈕 + 背景 async job | 使用者不需開 terminal，可重複觸發 |
| 蒐集範圍 | **最近 N 日**（預設 3，modal 可調整） | 使用者需求 |
| 炸彈過濾 | `image_results.is_bomb=1` 整張跳過；`tile_results.is_bomb=1` 該 tile 跳過 | DB 已有欄位可直接查 |
| Edge defect 樣本 | **一起蒐集**，crop 方式不同於 PatchCore tile | edge_defect 過檢也多，但需以 center 為中心裁切 |
| Label 分類來源 | `over_review.category` 的 enum（6 類）+ `true_ng`（AI=NG & RIC=NG）| 與現有 RIC 報表一致 |
| 去重策略 | Manifest CSV + `sample_id = (glass_id, image_name, sample_key)` 三元組 | CSV 最輕量，跨工具可用 |
| Label 變更處理 | 偵測到 manifest 與當前 DB category 不一致時，`shutil.move` 實體檔案到新目錄並更新 manifest | 使用者事後改 category 不會讓資料集過時 |

## 3. 系統架構

```
/ric 頁面 (過檢 Review tab)
   └─ 按「匯出訓練資料集」→ Bootstrap modal 填參數
          │ POST /api/dataset_export/start
          │   body: {days, output_dir, include_true_ng, skip_existing}
          ▼
capi_web.py
   ├─ 檢查 job lock → 取不到回 HTTP 409 + 現有 job_id
   ├─ 磁碟 free space 檢查 (< 1GB 拒絕)
   └─ 啟背景 threading.Thread → 立即回 {job_id, started_at}
                            │
                            ▼
capi_dataset_export.DatasetExporter.run()
   ├─ 1. 讀取 manifest.csv 成 dict {sample_id: row}
   │
   ├─ 2. DB 查詢（近似 SQL）:
   │        SELECT car.*, orv.category, orv.note
   │        FROM client_accuracy_records car
   │        LEFT JOIN over_review orv ON orv.client_record_id = car.id
   │        WHERE car.time_stamp >= date('now', '-N days')
   │          AND car.result_ai = 'NG'
   │          AND (
   │                car.result_ric = 'NG'                     -- 真 NG
   │             OR (car.result_ric = 'OK' AND orv.category IS NOT NULL)  -- 已回填的過檢
   │          )
   │
   ├─ 3. 對每筆 panel 關聯 inference_records / image_results / tile_results / edge_defect_results
   │        排除 image_results.is_bomb = 1 (整張跳過)
   │        排除 tile_results.is_bomb = 1 (個別 tile 跳過)
   │
   ├─ 4. 產生樣本清單（兩種 source_type）：
   │        a. patchcore_tile: 每個 tile_results where is_anomaly=1 AND is_bomb=0
   │        b. edge_defect:    每個 edge_defect_results
   │
   ├─ 5. 對每個樣本:
   │        a. 決定 label (true_ng / over_<category>)
   │        b. 決定光源 prefix (G0F/R0F/W0F/WGF/STANDARD)
   │        c. Manifest 去重/移動邏輯
   │        d. 讀原圖 (resolve_unc_path) → 讀不到 mark skipped_no_source
   │        e. Crop 512×512 → 寫入 <output>/<label>/<prefix>/crop/<filename>
   │        f. 複製 heatmap → <output>/<label>/<prefix>/heatmap/<filename>
   │        g. 更新 manifest row
   │
   ├─ 6. 每處理 1 筆更新 job_status (current / total / last_glass_id)
   └─ 7. 完成後 rewrite manifest.csv + 寫入 summary + release lock
                            ▲
                            │
前端每 2s polling: GET /api/dataset_export/status
完成後: GET /api/dataset_export/summary/<job_id> → 顯示 summary modal
```

### 3.1 DB 關聯（待實作階段確認）

`client_accuracy_records` 與 `inference_records` 的關聯 key 需在實作階段確認：
- 候選 1：`pnl_id = glass_id` + 時間視窗 (±N 分鐘)
- 候選 2：現有 code 是否已有 helper 函式可直接呼叫
- 參考 `capi_web.py` 過檢 Review tab 既有的 JOIN 實作

## 4. 輸出目錄結構

預設 `base_dir`：
- Linux (`server_config.yaml`): `/data/capi_ai/datasets/over_review`
- Windows (`server_config_local.yaml`): `./datasets/over_review`

```
<base_dir>/
├── manifest.csv                         # 全域 source-of-truth
├── export_logs/
│   └── job_20260410_153000.log          # 每次執行的 log
│
├── true_ng/                             # AI=NG & RIC=NG
│   ├── G0F/
│   │   ├── crop/
│   │   │   └── 20260408_GLS123_G0F0001_tile3.png
│   │   └── heatmap/
│   │       └── 20260408_GLS123_G0F0001_tile3.png
│   ├── R0F/  W0F/  WGF/  STANDARD/
│
├── over_edge_false_positive/
│   └── G0F/  R0F/  W0F/  WGF/  STANDARD/
├── over_within_spec/
├── over_overexposure/
├── over_surface_scratch/
├── over_aoi_ai_false_positive/
└── over_other/
```

**檔名規則**：`{YYYYMMDD}_{glass_id}_{image_name_no_ext}_{sample_key}.png`
- `YYYYMMDD`：推論日期（非 collected 日）
- `sample_key`：`tile{tile_id}`（PatchCore）或 `edge{edge_defect_id}`（edge_defect）
- crop 與對應 heatmap **檔名完全一致**，方便後續程式用同一個 key 同時讀兩個 view

**範例**：
- `20260408_GLS123_G0F0001_tile3.png` → PatchCore tile 樣本
- `20260408_GLS123_W0F0002_edge7.png` → Edge defect 樣本

## 5. Manifest CSV Schema

單一檔案 `<base_dir>/manifest.csv`，所有樣本集中記錄。

| 欄位 | 型態 | 範例 | 說明 |
|---|---|---|---|
| `sample_id` | str | `GLS123_G0F0001_tile3` | 去重 key，glass_id + image_name + sample_key |
| `collected_at` | ISO8601 | `2026-04-10T15:30:12` | 最後一次寫入時間 |
| `label` | str | `over_edge_false_positive` | 與目錄層一致 |
| `source_type` | str | `patchcore_tile` / `edge_defect` | |
| `prefix` | str | `G0F` | 光源 |
| `glass_id` | str | `GLS123` | |
| `image_name` | str | `G0F0001.bmp` | |
| `inference_record_id` | int | `45821` | FK → inference_records |
| `image_result_id` | int | `124305` | FK → image_results |
| `tile_idx` | int or blank | `3` | tile_results.tile_id（edge 樣本為空） |
| `edge_defect_id` | int or blank | `` | edge_defect_results.id（PatchCore 樣本為空） |
| `crop_path` | str | `true_ng/G0F/crop/20260408_....png` | 相對 base_dir |
| `heatmap_path` | str | `true_ng/G0F/heatmap/20260408_....png` | 相對 base_dir |
| `ai_score` | float | `0.823` | tile_results.score（edge 樣本填 edge_defect 的某欄，待確認） |
| `defect_x` | int | `512` | crop 中心在原圖的 X |
| `defect_y` | int | `768` | crop 中心在原圖的 Y |
| `ric_judgment` | str | `NG` / `OK` | |
| `over_review_category` | str or blank | `edge_false_positive` | |
| `over_review_note` | str or blank | 使用者填的備註 | |
| `inference_timestamp` | ISO8601 | `2026-04-08T14:22:03` | 原推論時間 |
| `status` | str | `ok` / `skipped_no_source` / `skipped_no_heatmap` / `skipped_out_of_bounds` | |

### 5.1 增量蒐集邏輯

1. **Job 開始** → 讀 `manifest.csv` 成 dict `{sample_id: row}`
2. **對每個候選樣本**：
   - `sample_id` 不在 dict：
     - 讀原圖、crop、拷 heatmap 成功 → append 新 row，`status='ok'`
     - 任一步失敗 → 仍 append 一列 `status='skipped_*'`（不寫實體檔），避免下一輪重覆 retry
   - 在 dict 且 `label` 相同 且 `status='ok'` → skip（`skip_existing=False` 時才強制重做）
   - 在 dict 且 `status='skipped_*'` → 不再 retry（除非 `skip_existing=False`，這時重試一次並更新 status）
   - 在 dict 但 `label` 變了（使用者事後改了 over_review category）→ `shutil.move` 舊 crop/heatmap 到新目錄 + 更新 row 的 `label` / `crop_path` / `heatmap_path` / `collected_at`（僅對 `status='ok'` 的 row 執行）
3. **Job 結束** → 整批 rewrite `manifest.csv`（job lock 保證單寫入者）

## 6. Crop 策略

### 6.1 PatchCore tile 樣本

```python
# tile_results.x, y, width, height 是 tile 在原圖的絕對座標
x, y, w, h = tile_row['x'], tile_row['y'], tile_row['width'], tile_row['height']
crop = original[y:y+h, x:x+w]

# 不足 512 以黑邊 pad 到 512×512
if crop.shape[:2] != (512, 512):
    pad_h = 512 - crop.shape[0]
    pad_w = 512 - crop.shape[1]
    crop = cv2.copyMakeBorder(crop, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

defect_x, defect_y = x + w // 2, y + h // 2
```

### 6.2 Edge defect 樣本

使用者明確要求以 AOI 座標（= `edge_defect_results.center_x/center_y`）為中心裁切：

```python
cx, cy = edge_row['center_x'], edge_row['center_y']
half = 256
x1, y1 = cx - half, cy - half
x2, y2 = cx + half, cy + half

# 邊界 clamp
H, W = original.shape[:2]
x1_clamped = max(0, x1); y1_clamped = max(0, y1)
x2_clamped = min(W, x2); y2_clamped = min(H, y2)
crop = original[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

# 以黑邊補回 clamp 掉的部分，保持 defect 中心在 crop 中央
top_pad = y1_clamped - y1
bot_pad = y2 - y2_clamped
left_pad = x1_clamped - x1
right_pad = x2 - x2_clamped
crop = cv2.copyMakeBorder(crop, top_pad, bot_pad, left_pad, right_pad,
                          cv2.BORDER_CONSTANT, value=0)

defect_x, defect_y = cx, cy
```

**邊界 clamp 後有效像素 < 25%** → skip，`status='skipped_out_of_bounds'`。

### 6.3 Heatmap 複製

不重新生成，直接 `shutil.copy2`：
- PatchCore：`tile_results.heatmap_path` → `<base_dir>/<label>/<prefix>/heatmap/<filename>`
- Edge defect：`edge_defect_results.heatmap_path` → 同上

原檔案不存在 → skip，`status='skipped_no_heatmap'`。

## 7. Web API 設計

全部新增在 `capi_web.py`。

| Method | Path | 功能 | 回應 |
|---|---|---|---|
| POST | `/api/dataset_export/start` | 啟動背景 job | `{job_id, started_at}` 或 HTTP 409 `{current_job_id, reason}` |
| GET | `/api/dataset_export/status` | 當前 job 狀態 | `{job_id, state, current, total, last_glass_id, started_at, elapsed_sec}` state ∈ `running`/`completed`/`failed`/`idle` |
| GET | `/api/dataset_export/summary/<job_id>` | 完成後摘要 | `{labels: {true_ng: 120, over_edge_false_positive: 45, ...}, skipped: {no_source: 3, no_heatmap: 2, out_of_bounds: 1}, total: 295, output_dir, duration_sec}` |
| POST | `/api/dataset_export/cancel` | 設 cancel flag | `{ok: true}` |

**Job lock 實作**：
```python
# capi_web.py 模組層級
_dataset_export_state = {
    'lock': threading.Lock(),
    'current_job': None,  # {job_id, state, current, total, ...}
    'cancel_flag': threading.Event(),
    'last_summary': None,  # 保留最近一次完成的 summary 供查詢
}
```

啟動時 `lock.acquire(blocking=False)`，釋放在 job thread 的 `finally` 裡。`cancel_flag.set()` 後工具在每筆開始前檢查。

## 8. 前端（templates/ric_report.html）

### 8.1 按鈕位置

`/ric` 過檢 Review tab 頂部加一個 `<button id="btnExportDataset">匯出訓練資料集</button>`。

### 8.2 Modal UI

```
┌──────────────────────────────────────────┐
│ 匯出訓練資料集                            │
├──────────────────────────────────────────┤
│ 天數:           [3     ]                  │
│ 輸出目錄:       [/data/capi_ai/datasets/  │
│                  over_review           ]  │
│ ☑ 包含 真 NG 樣本 (true_ng)              │
│ ☑ 跳過已存在的樣本                        │
│                                           │
│ [取消]                          [開始匯出] │
└──────────────────────────────────────────┘
```

### 8.3 進度畫面

按下「開始匯出」後，modal 切換到進度畫面：
- 進度條（`current / total`）
- 當前處理的 glass_id
- 已耗時
- 「取消」按鈕（call `/api/dataset_export/cancel`）

Polling 間隔 2 秒。

### 8.4 Summary 畫面

`state == 'completed'` 時切換到 summary：
- 各 label 樣本數
- Skip 統計（分類型）
- 輸出目錄路徑（可點擊複製）
- 總耗時
- 「關閉」按鈕

## 9. 錯誤處理與邊界條件

| 情境 | 處理 |
|---|---|
| 原圖讀不到（UNC 失效/檔案已清）| skip 該樣本，manifest `status='skipped_no_source'`，log 前 5 筆 glass_id |
| heatmap 檔案不存在 | skip，`status='skipped_no_heatmap'` |
| `image_results.is_bomb=1` | 整張 image 所有樣本都跳過 |
| `tile_results.is_bomb=1` | 該 tile 跳過，同 image 其他 tile 照做 |
| Edge defect 邊界 clamp 後有效像素 < 25% | skip，`status='skipped_out_of_bounds'` |
| Job 執行到一半 server crash | lock 在 process 內無殘留；已寫入的樣本下次跑到時被 manifest 去重自動跳過 |
| 同時兩個 job 請求 | 第二個回 HTTP 409 + 現有 job_id |
| manifest.csv 讀寫 race | 讀取時複製到 memory dict，job 結束才整批 rewrite；job lock 保證單寫入 |
| 使用者中途改 `over_review.category` | 下次 job 偵測到 label 不一致 → `shutil.move` 實體檔案並更新 manifest |
| 磁碟空間不足 | Job start 前檢查 `base_dir` 所在磁碟 free space，< 1GB 直接拒絕啟動 |

## 10. 檔案組織與變更範圍

### 新增
- `capi_dataset_export.py` — 核心 exporter，`DatasetExporter` class + `start_job/get_status/get_summary/cancel` 模組級函式
- `tests/test_dataset_export.py` — 測試：crop 邏輯（含 edge 邊界 clamp）、manifest 去重/移動、bomb 過濾、DB 查詢篩選條件

### 修改
- `capi_web.py` — 4 個 API endpoint + job state
- `templates/ric_report.html` — 按鈕 + modal + polling JS
- `server_config.yaml` / `server_config_local.yaml` — 新增 `dataset_export.base_dir` key

### 不修改
- `capi_database.py`（只讀查詢，不動 schema）
- `capi_inference.py` / `capi_heatmap.py`（完全獨立）

## 11. 非目標（Out of Scope）

明確排除以下項目，避免 scope creep：
- **分類 AI 模型本身**：本工具只負責蒐集資料集。訓練流程是下一個獨立階段
- **即時觸發**：工具只在 web 按鈕觸發，不做 cron / webhook / 推論時即時寫入
- **跨 server 同步**：單 server 執行，不處理多 server 資料合併
- **歷史資料回溯補填**：只掃「最近 N 日」+ 已回填 Review 的樣本，不處理歷史上尚未回填的過檢
- **Heatmap 重新生成**：直接沿用推論時產生的 heatmap 檔案，找不到就 skip

## 12. 後續步驟

1. 以此 spec 為基礎進入 `writing-plans` skill，產出分段實作計畫
2. 實作計畫階段需確認的技術細節：
   - `client_accuracy_records` → `inference_records` 的 JOIN key（讀既有 `/api/ric/*` code 確認）
   - edge defect 樣本的 `ai_score` 欄位要填什麼（edge_defect_results 有哪些 score 類欄位）
   - Windows 本地測試時，heatmap 路徑通常是絕對 Windows 路徑，是否需要再做 path normalize
3. 實作完成後先以 `--days 1` 執行小規模測試，驗證目錄結構與 manifest 正確後再擴到 3 日
