# Bundle 單一子模型重新訓練 — Design Spec

**Date:** 2026-05-06
**Status:** Draft

---

## Context

`/models` 模型庫中每個 bundle = 1 機台 × 10 個 PatchCore 子模型（5 燈源 × inner/edge zone）。實際使用發現**訓練資料過濾不夠乾淨**——例如 OK pool 中混入帶灰塵的 tile——導致該子模型誤判率升高。

目前唯一的修法是從 `/train/new` 5 步精靈整套重做，產生新 bundle。對「只有一個 lighting+zone 髒」這個常見情境而言成本過高（重訓 9 個本來就乾淨的子模型）。

本 spec 設計一條「單子模型在 bundle 內就地重訓」路徑：在既有 bundle detail 頁直接標記要排除的 tile，重訓被影響的子模型，**就地覆蓋同 bundle 內的 .pt 檔**。

---

## Scope

**In scope:**
- bundle detail 頁的訓練 tile 檢視介面新增 OK tile 的 accept/reject 切換能力
- 新增 API：批次更新 tile decision、觸發單子模型重訓、查重訓進度
- 重訓單一 lighting+zone 的 PatchCore，重算 AUROC 與其他 metrics
- 就地覆蓋 `bundle_dir/<lighting>_<zone>/.../model.pt`
- 通知執行中的 inferencer reload 該子模型

**Threshold 處理：刻意不動。** `calibrate_threshold` 已硬寫回 `DEFAULT_THRESHOLD = 0.5`（`capi_train_new.py:405-418`），threshold 是使用者在模型庫頁面依誤判情況**手動微調**的。重訓覆蓋使用者調過的值會破壞他們的工作成果，因此重訓只動 model.pt 與 metrics，**threshold 與 yaml 完全不動**。

**Out of scope:**
- NG 樣本標記能力（NG 在 PatchCore 訓練中不參與，只用於 threshold 校準與 AUROC，本次設計不開放編輯）
- 多子模型同時重訓（介面強制一次一個）
- 重訓完整 bundle（既有 `/train/new` 已涵蓋）
- 訓練取消（GPU job 中斷困難，沿用 retrain-ui-design 的決策）
- 舊 .pt 備份（純覆蓋，使用者明確選擇不留備份）
- 改用沒被切到的原始 panel 影像重新 preprocess（B 情境只動既有 tile pool）

---

## Architecture

### 資料層假設

`training_tile_pool` 表已存在 `decision TEXT DEFAULT 'accept'` 欄位（`capi_database.py:294`），DB 已支援 accept/reject 標記。step3 review 介面已使用此欄位做訓練前篩選。

bundle 與 tile pool 透過 `model_bundles.job_id` ↔ `training_tile_pool.job_id` 關聯。本設計成立的前提：**該 bundle 的訓練資料尚未被「刪訓練資料」清除**（`capi_model_registry.delete_training_data`）。若已清除，UI 顯示禁用提示，引導使用者改走 `/train/new`。

### 觸發路徑

```
/models  →  bundle detail
              │
              └─ 訓練 tile 區（既有 /api/models/<id>/training_tiles）
                    │
                    ├─ 新增：每張 tile 縮圖支援切換 accept/reject
                    ├─ 新增：偵測到「有未訓練的修改」時，
                    │        浮出「重訓 G0F-edge」按鈕（按單一 lighting+zone）
                    │
                    └─ 按下重訓 → 後台 job → 進度面板 → 完成 → reload
```

### 後端流程

```
POST /api/models/<bundle_id>/tiles/decision     ← 標記 accept/reject（沿用 update_tile_decisions）
POST /api/models/<bundle_id>/retrain_submodel   ← 觸發重訓
GET  /api/models/<bundle_id>/retrain_status     ← 查進度
```

重訓 worker 執行序：

```
_retrain_submodel_worker(bundle_id, lighting, zone)
│
├─ 1. 讀 bundle、找到 job_id 與 bundle_dir
├─ 2. 從 training_tile_pool 撈出 (job_id, lighting, zone, source=ok, decision=accept) 的 tile 路徑
│     若 tile 數 < 最低門檻（沿用 train_new 的下限）→ 失敗
├─ 3. stage_dataset 到 bundle_dir/_tmp/<lighting>_<zone>/staging/
├─ 4. 取得 GPU lock（與 inference 共用 _gpu_lock）
│     train_one_patchcore(...) 產出在 bundle_dir/_tmp/<lighting>_<zone>/run/
│       └─ rglob 找到 model.pt（通常在 weights/torch/model.pt）
│     用新模型對 NG 樣本打分 → 算 AUROC、ng_caught_rate 等 metrics
│     釋放 GPU lock
├─ 5. atomic 替換：
│     - 找到 bundle_dir/<lighting>_<zone>/ 內既有 model.pt 路徑
│     - 將 _tmp 內新產出的 model.pt 連同其週邊產物一起 rename 到正式位置
│       （anomalib export 會帶整個 weights/torch/ 結構，整目錄替換）
│     - 沿用「先 rename 舊目錄為 .replacing → mv 新目錄 → 刪 .replacing」做近 atomic 切換
│     - 失敗時用 .replacing 還原
│     - 成功後刪除 _tmp 目錄
├─ 6. 更新 bundle manifest（不動 yaml threshold）：
│     submodel_history 追加新 entry（trained_at、tile_count_used、auroc、used_tile_ids）
│     last_retrained_at 更新
├─ 7. 通知 inferencer：reload 該 bundle 的單一子模型
│     沿用 commit 1fc424f 的「settings 改完即時同步到 inferencer.config」機制
│     新增 `reload_submodel(bundle_id, lighting, zone)` API
│     （只 reload .pt，threshold 維持原值不需更新）
└─ 失敗時：
    - _tmp 目錄整個刪除
    - 原 model.pt 與原 yaml 完全不動
    - DB 中 tile decision 變更保留
    - job state 標 failed，UI 顯示 traceback
```

### GPU 與 inference 並存

訓練期間 inference 排隊等候（`_gpu_lock`）。預期單一子模型訓練 1–3 分鐘，AOI 端 retry 機制可吸收。**訓練不啟用獨立 process**（與 `capi_train_runner.py` 的整批訓練不同），因為單一子模型訓練短，且共用 lock 比 IPC 簡單。

若日後測得單次訓練 > 5 分鐘導致 AOI 端 retry 耗盡，再考慮拆 process。

### 失敗回滾

採用「`_tmp` 暫存 → atomic rename」設計，失敗時：
- `_tmp/` 整個刪除
- 原 bundle 內 model.pt 與 manifest.json 完全不動
- machine_config.yaml 本來就不會被動到（threshold 永不被重訓覆蓋）
- DB 中的 tile decision 變更**保留**（使用者下次修正後可再試）

不留歷史備份。**這是使用者明確決策**——理由是 bundle 列表已會膨脹，且 PatchCore weights 較大（每個子模型數十 MB），長期備份會吃掉磁碟。

---

## UI Changes

### Bundle Detail 頁（template 既有）

訓練 tile 區既有 layout（lighting 切換 + zone 切換 + tile grid）保留，新增：

1. **每張 tile 縮圖**支援點擊或鍵盤切換 accept/reject 狀態
   - 沿用 step3 review 的鍵盤約定：`Del` 排除、`A` 加入、`Enter` 放大、方向鍵切換
   - reject 狀態縮圖加暗色濾鏡 + 紅框
   - 篩選按鈕：全部 / 已加入 / 已排除（沿用 step3 樣式）

2. **變更指示條**（lighting+zone 切換列下方）
   - 偵測該 lighting+zone 是否有相對於上次訓練的待重訓修改
   - 判定：對比 `submodel_history` 最新 entry 的 `used_tile_ids` 與目前 `tile_pool.decision='accept'` 的 id 集合
   - 顯示文字：`G0F-edge：有 12 張 tile 修改未訓練` + `重訓此子模型 →` 按鈕
   - 無修改時隱藏按鈕
   - **舊 bundle 過渡相容**：若 manifest 沒有 `submodel_history`（在此功能上線前訓練的 bundle），仍允許重訓，差異判定改為「目前是否有任何 reject 標記」

3. **重訓進度面板**（按下後浮現）
   - 沿用 retrain-ui-design 的進度面板樣式：步驟指示 + log + elapsed time
   - 步驟：`stage` → `train` → `metrics` → `swap` → `reload` → `done`
   - 完成後顯示**新舊 AUROC 對照**、tile 數變化（如 1856 → 1820，排除 36 張）
   - 提示 threshold 維持原值（沿用使用者目前在模型庫調的設定）
   - 失敗顯示 traceback，原模型保持運作

4. **訓練資料已被刪除的禁用提示**
   - 若 bundle 對應 job_id 的 tile_pool 已不存在 → 整個編輯區改為唯讀提示框
   - 文字：「此 bundle 的訓練資料已清除，無法在此重訓。請至 /train/new 重新訓練。」

### Bundle 列表頁

每個 bundle 列出時，新增小徽章：「⚠ 有 N 個子模型有未訓練的修改」（若有），點擊跳轉 detail。

---

## Database / 檔案變更

### bundle 內檔案

`bundle_dir/manifest.json` 新增欄位：

```json
{
  ...既有欄位...,
  "submodel_history": {
    "G0F00000_edge": [
      {
        "trained_at": "2026-05-06T14:32:00",
        "tile_count_used": 1856,
        "auroc": 0.973,
        "used_tile_ids": [12031, 12032, ...],
        "kind": "initial"
      },
      {
        "trained_at": "2026-05-08T10:15:00",
        "tile_count_used": 1820,
        "auroc": 0.981,
        "used_tile_ids": [12031, 12033, ...],
        "kind": "retrain"
      }
    ]
  },
  "last_retrained_at": "2026-05-08T10:15:00"
}
```

`used_tile_ids` 是「該次訓練實際送進去的 tile_pool.id 集合」。**初次訓練流程也要新增這項記錄**（修改 `run_training_pipeline`），因為 UI 判斷「有未訓練的修改」要靠它對比目前 `tile_pool.decision='accept'` 的 id 集合。

`machine_config.yaml` **不動**——threshold 沿用 yaml 既有值（使用者手動調的結果保持不變）。

### DB

不新增表。沿用既有：
- `training_tile_pool.decision` — 標記 accept/reject
- `model_bundles.job_id` — 串接 tile pool

新增背景 job 狀態（runtime only，不寫 DB）：

```python
_submodel_retrain_state = {
    "lock": threading.Lock(),
    "job": None,
    # job dict when running:
    # {
    #   bundle_id: int,
    #   lighting: str,
    #   zone: str,
    #   state: str,      # "running" | "completed" | "failed"
    #   step: str,       # "stage" | "train" | "metrics" | "swap" | "reload" | "done"
    #   started_at: str,
    #   log_lines: list[str],
    #   summary: dict | None,    # {auroc_old, auroc_new, tile_count_old, tile_count_new}
    #   error: str | None,
    # }
}
```

一次只允許一個重訓 job 跑。

---

## API 變更

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/models/<id>/tiles/decision` | body: `{tile_ids: [int], decision: "accept"\|"reject"}` 批次切換 OK tile 狀態。NG tile 嘗試操作 → 400 |
| POST | `/api/models/<id>/retrain_submodel` | body: `{lighting: str, zone: "inner"\|"edge"}` 啟動重訓 worker。已有 job 跑 → 409 |
| GET | `/api/models/<id>/retrain_status` | 回 `_submodel_retrain_state["job"]` 內容 + 末 N 行 log |
| GET | `/api/models/<id>/training_tiles` | **既有**，新增回傳欄位 `decision` 給前端渲染 |

---

## Files Modified

| File | Change |
|------|--------|
| `capi_web.py` | 加 `_submodel_retrain_state`、`_handle_models_tiles_decision`、`_handle_models_retrain_submodel`、`_handle_models_retrain_status`、`_submodel_retrain_worker`；`_handle_models_training_tiles` 回傳加 `decision` 欄位 |
| `templates/models.html` | bundle detail 區加 tile accept/reject 切換、變更指示條、重訓按鈕、進度面板；bundle 列表加「N 個子模型有未訓練修改」徽章（同檔案內） |
| `capi_train_new.py` | 抽出 `train_single_submodel(job_id, lighting, zone, bundle_dir, ...)` 公用函式（沿用 `train_one_patchcore`）；`run_training_pipeline` 在初次訓練 manifest 寫入 `used_tile_ids` |
| `capi_model_registry.py` | 加 `append_submodel_history(bundle_dir, lighting, zone, entry)`、`get_used_tile_ids(bundle_dir, lighting, zone) -> set[int]`（給 UI 比對用） |
| `capi_inference.py` | 加 `reload_submodel(bundle_id, lighting, zone)`，只重讀對應 model.pt（不動 threshold） |

---

## Verification

完成後手動驗證：

1. **正常重訓路徑**
   - 選一個現有 bundle，找一個 lighting+zone（例 G0F-edge）
   - 在 tile 縮圖隨意排除幾張
   - 確認「有未訓練的修改」指示與「重訓 G0F-edge」按鈕出現
   - 按下重訓，確認進度面板更新、最終顯示新舊 AUROC 對照與 tile 數變化
   - 檢查 `bundle_dir/G0F00000_edge/...` 內 `model.pt`（rglob 找到的那個）修改時間更新
   - 檢查 `bundle_dir/machine_config.yaml` **完全沒動**（含 mtime 不變）
   - 檢查 `bundle_dir/manifest.json` 新增 `kind: "retrain"` 的 history entry，`used_tile_ids` 反映排除後的集合

2. **inference 即時生效**
   - 重訓完成後不重啟 server，直接送一個對應機台與 G0F 的測試 panel
   - 確認 inference 用的是新 model（觀察分數應有變化）；threshold 維持原值

3. **失敗回滾**
   - 故意傳一個非法 lighting/zone → API 回 400
   - 故意刪掉 _tmp 寫入權限模擬 IO 失敗 → 確認 bundle_dir 完全不動，原 inference 持續運作

4. **GPU lock 排隊**
   - 重訓進行中送 AOI 請求，確認 inference 在訓練結束後才繼續

5. **訓練資料已刪場景**
   - 對 bundle 跑「刪訓練資料」後進 detail 頁
   - 確認顯示禁用提示，按鈕灰掉

6. **NG tile 唯讀**
   - 嘗試對 NG tile 呼叫 decision API → 回 400

7. **同時兩個重訓**
   - 觸發 A 後立刻觸發 B → B 收到 409
