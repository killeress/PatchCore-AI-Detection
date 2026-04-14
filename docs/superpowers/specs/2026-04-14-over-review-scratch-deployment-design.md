# Over-Review Scratch Deployment — Design Spec

**Date:** 2026-04-14
**Status:** Design approved, ready for writing-plans
**Supersedes/extends:** `docs/superpowers/specs/2026-04-14-over-review-scratch-poc-design.md` (POC-only)
**Prev artifacts:**
- `docs/over_review_scratch_poc_result.md` — POC ablation history
- `docs/over_review_scratch_deployment_handoff.md` — handoff guide that scopes this spec (§3.1 A–G)

---

## 1. 目的與範圍

把 over-review scratch POC 的最強配置（LoRA r=16 / 2 blocks / 15 epochs + CLAHE cl=4 + Conformal threshold）整合進 CAPI AI 主線推論流程，讓 AOI 判 NG、但 AI 分類器認定為高信心 scratch 的 tile 自動翻回 OK，以降低過檢率。

**in scope：**
- `ScratchClassifier` 部署類別封裝（A）
- `CAPIInferencer.process_panel` 整合點（B, C）
- GPU lock 共用、latency budget（C, D）
- 單-threshold flip 策略 + safety multiplier（E）
- Runtime config 開關（F）
- 離線部署檔案打包（G）
- 產出 `train_final_model.py` 訓練最終部署模型（§3.2）
- DB schema 擴充 audit 欄位（§3.3）
- 單元測試 + 整合測試（§3.4）

**out of scope（明確剔除）：**
- 新 dashboard 頁面 / inline 人工覆判 UI — 現有 RIC 流程（物理覆檢）作為下游 leak 防線
- Shadow mode rollout — 決定直接 enable，runtime 可關
- Canary 分機部署 — 直接全場 enable
- 其他過檢類別（overexposure、within_spec 等）— 重用本 pipeline 但各自獨立 spec
- 動態 safety auto-tuning — 部署後手動調整即可
- Borderline tile 另外處理路徑（LOW_THR） — 單 threshold 就夠

---

## 2. 背景：POC 結論（取自 `docs/over_review_scratch_poc_result.md` 與 rerun log）

### 2.1 當前最強配置

**LoRA r=16 / 2 blocks / 15 epochs + CLAHE cl=4**（cls token）：

| Metric | Value |
|---|---|
| Realistic scratch_recall | 90.4% ± 5.8% |
| Realistic true_ng_recall | 98.3% ± 1.7% |
| **Oracle scratch_recall** | 86.0% ± 9.2% |
| **Conformal scratch_recall**（safety=1.0）| 90.4% ± 5.8% |
| **Conformal leak**（safety=1.0） | 2.6% ± 2.8% |
| PR-AUC | 0.931 ± 0.043 |
| 推論 VRAM | ~1 GB |
| 推論時間 | ~50ms/crop (2080 Ti) / ~25ms (5060 Ti 生產機) |

來源：`/tmp/lora_r16_conformal.log`（2026-04-14）。

### 2.2 Preprocessing pipeline（不可改動）

執行順序 fixed，`scripts/over_review_poc/features.py::build_transform_clahe` 已有參考實作：

```python
1. PIL.Image.open(path).convert("RGB")
2. CLAHE (clipLimit=4.0, tileGridSize=(8,8)) on grayscale → stack ×3 channel
3. Resize((224, 224))  # torchvision default bilinear
4. ToTensor()
5. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
6. model(x) → CLS token (768-d)
7. LogReg.predict_proba(x)[:, 1] → scratch score
```

任何順序/參數變更都等於新 preprocessing，需新 `preprocessing_id` 並重訓模型。

### 2.3 Conformal threshold 機制

訓練最終模型時：
- 全部 1748 筆 → 80% 正式訓練 + 20% calibration（group-aware，同 `glass_id` 不跨）
- `calib_max_ng_score` = calibration set 中 true_ng 最高分
- **Deployment threshold** = `min(calib_max_ng_score × safety_multiplier, 0.9999)`

POC rerun 顯示 safety=1.0 平均 leak 2.6%（fold 5 是 8.1% outlier）。Spec 默認 **safety=1.1** 作 runtime config 預設值，可被 DB `config_params` 覆寫。

---

## 3. 架構總覽

```
┌─────────────────────────────────────────────────────────────────┐
│ capi_server.py                                                  │
│   TCP listener → per-client thread                              │
│     └─ _gpu_lock ────────────────────────────────────────────┐  │
│                                                              │  │
│   CAPIInferencer.process_panel(panel_dir)                   │  │
│     ┌─ (1) PatchCore tiling + scoring                       │  │
│     ├─ (2) Dust / OMIT filter                               │  │
│     ├─ (3) Bomb match / Edge decay                          │  │
│     ├─ (4) ★ ScratchFilter.apply(image_result)   ← NEW      │  │
│     │     For (tile, anom_score, heatmap) in anomaly_tiles: │  │
│     │       score = classifier.predict(tile.image)          │  │
│     │       tile.scratch_score = score                      │  │
│     │       if score > effective_threshold:                 │  │
│     │         tile.scratch_filtered = True                  │  │
│     │         (entry moved out of anomaly_tiles)            │  │
│     │         image_result.scratch_filter_count += 1        │  │
│     └─ (5) Aggregate anomaly_tiles → is_ng 判定              │  │
│           ↓                                                 │  │
│   Release _gpu_lock → send response → background heatmap/DB │  │
└─────────────────────────────────────────────────────────────┘──┘
```

Scratch filter 是 PatchCore 判 NG 之後的 post-filter：只有已被標 NG 的 tile 會進分類器；若 classifier 沒載入 / config 關閉，這步整個 skip。

---

## 4. 設計細節

### A. `ScratchClassifier` 類別設計

位置：**新檔** `scratch_classifier.py`（平行於 `capi_inference.py`，import 層級同）。

```python
@dataclass
class ScratchClassifierMetadata:
    preprocessing_id: str          # e.g. "v3_clahe_cl4.0_tg8"
    clahe_clip: float              # 4.0
    clahe_tile: int                # 8
    input_size: int                # 224
    dinov2_repo: str               # "facebookresearch/dinov2"
    dinov2_model: str              # "dinov2_vitb14"
    lora_rank: int                 # 16
    lora_alpha: int                # 16
    lora_n_blocks: int             # 2
    conformal_threshold: float     # 來自訓練集 calib set 的 max true_ng score
    safety_multiplier: float       # 1.1 (default，可被 runtime config 覆寫)
    trained_at: str                # ISO datetime
    dataset_sha256: str            # manifest sha256 防模型/資料漂移
    git_commit: str                # 訓練時的 git HEAD

class ScratchClassifier:
    def __init__(self, bundle_path: str | Path,
                 dinov2_weights_path: str | Path,
                 device: str = "cuda"): ...

    @property
    def conformal_threshold(self) -> float:
        """原始 conformal threshold（bundle metadata，訓練時固化，不含 safety）"""

    def predict(self, image: PIL.Image | np.ndarray) -> float:
        """
        Returns scratch probability in [0, 1].
        Accepts PIL Image (RGB) or np.ndarray HxWxC (uint8 RGB).
        Applies full preprocessing → LoRA DINOv2 → LogReg.
        """

    def predict_batch(self, images: list[PIL.Image | np.ndarray]) -> np.ndarray:
        """Vectorised version for panel-level batching."""
```

**Bundle 檔案 (`deployment/scratch_classifier_v1.pkl`)** 內容：
- `lora_state_dict: dict[str, tensor]` — 只存 LoRA adapter 權重（~1MB）
- `logreg: sklearn.linear_model.LogisticRegression` — pickle 可序列化
- `metadata: ScratchClassifierMetadata`
- `calibration_scores: np.ndarray` — audit 用，記錄 calib set 分數分佈

DINOv2 base 權重分離：`deployment/dinov2_vitb14.pth`（330MB）— 可被多個 bundle 共用。

**錯誤處理：**
- Bundle 或 DINOv2 載入失敗 → raise `ScratchClassifierLoadError`，由 caller 決定是否降級
- `predict` 單張失敗 → propagate exception，caller 必須處理（見 §4.B）

### B. 整合點：`capi_inference.py` 變更

1. **`CAPIInferencer.__init__`** 新增：
   ```python
   self.scratch_filter: ScratchFilter | None = None  # lazy 載入
   ```

2. **新 helper 方法** `_get_scratch_filter() -> ScratchFilter | None`：
   - 首次呼叫時檢查 `self.config.scratch_classifier_enabled`
   - 若 enabled → load bundle → cache 到 `self.scratch_filter`
   - Load 失敗 → log error + 設 `self._scratch_load_failed = True`，之後呼叫直接 return None（**不嘗試重載**，避免每個 request 都 hit 磁碟）
   - 每次呼叫皆讀當前 `self.config.scratch_safety_multiplier`（可能已被 DB 動態更新）傳給 filter

3. **`process_panel` 插入點**：在 per-image PatchCore 推論結束、image-level is_ng 判定之前，對每張 `ImageResult` 的 `anomaly_tiles` 跑 scratch filter。偽代碼：
   ```python
   for image_result in panel_results:        # panel 可能有多張圖
       sf = self._get_scratch_filter()
       if sf is None or not image_result.anomaly_tiles:
           continue
       sf.apply_to_image_result(image_result)
   ```

4. **新模組** `scratch_filter.py`（與 `scratch_classifier.py` 分離，前者處理「在 CAPI pipeline 裡怎麼用」、後者處理「模型怎麼跑」）：
   ```python
   class ScratchFilter:
       def __init__(self,
                    classifier: ScratchClassifier,
                    safety_multiplier: float = 1.1):
           """
           effective_threshold = classifier.conformal_threshold × safety_multiplier
           一旦建立即固定；若要動態調整，由 CAPIInferencer._get_scratch_filter
           在 safety config 值變更時重建 filter instance。
           """

       def apply_to_image_result(self, image_result: ImageResult) -> None:
           """
           就地修改 image_result:
             - 迭代 anomaly_tiles 每筆 (tile, anom_score, heatmap)
             - 直接用 tile.image (np.ndarray, already cropped) 當 classifier 輸入
             - classifier.predict(tile.image) → tile.scratch_score
             - 若 score > self.effective_threshold:
                 tile.scratch_filtered = True
                 該 tuple 從 anomaly_tiles 移除
                 image_result.scratch_filter_count += 1
             - 單 tile predict 噴例外 → tile.scratch_score=0，留在 anomaly_tiles
               (safety default)，繼續處理下一個 tile
           """
   ```

5. **`TileInfo` dataclass 新欄位**（`capi_inference.py`）：
   ```python
   scratch_score: float = 0.0              # 0 = 未跑 classifier
   scratch_filtered: bool = False          # True = 被翻回 OK
   ```

6. **`ImageResult` dataclass 新欄位**：
   ```python
   scratch_filter_count: int = 0           # 被翻 OK 的 tile 數
   ```

7. **Tile crop 來源**：直接用 `TileInfo.image`（PatchCore tiling 階段已切好的 np.ndarray），**不需額外切圖**。CLAHE / normalize 等 preprocessing 由 `ScratchClassifier.predict` 內部處理。

### C. Threading / Concurrency

- **GPU lock：共用 `_gpu_lock`。** PatchCore 推論 → Scratch classifier 推論 串行。好處：GPU VRAM 上限可預測（PatchCore ~3GB + Scratch ~1GB = 4GB，遠低於 5060 Ti 16GB）。
- **ScratchFilter 本身無跨 request state**：predict 為 thread-safe（PyTorch `.eval()` + `torch.no_grad()`）。
- **Lazy load 時機**：第一次遇到 NG tile 時在 `_get_scratch_filter` 內載入（bundle 載入可能花 5-10s）。**不需要額外 `threading.Lock`**：所有呼叫都在 `_gpu_lock` held 狀態內（`process_panel` 已 hold），自然序列化。
- **CUDA copy 在 lazy load 時**：state_dict copy to CUDA 需要 GPU。因為 `_gpu_lock` 已 held，不會有 OOM 競爭。

### D. Latency Budget

| 階段 | 典型 tile 數 | 每 tile | 小計 |
|---|---|---|---|
| PatchCore scoring | ~500 | 既有（不變） | — |
| ★ Scratch filter | 只對 **NG tiles** | ~25ms (5060 Ti) | N_ng × 25ms |

觀察生產資料：單 panel NG tile 數常見 ~5-30 個（非全 500 張）。預估：
- Typical case (10 NG tiles): **250ms/panel** 新增
- Worst case (50 NG tiles): **1.25s/panel** 新增

相比 handoff §3.1 D 預估的 15-25 秒/panel，實際應該在 1 秒量級內，因為只跑 NG tile 而非全部 500 tile。

**不做 batching 優化**（YAGNI）：純迴圈呼叫 `predict`，單請求內 GPU 已序列化；若後續發現瓶頸再加 `predict_batch` 調用。

### E. Threshold 策略（單-threshold）

**唯一 threshold：** `effective_threshold = conformal_threshold × safety_multiplier`

- `conformal_threshold`：訓練時固化在 bundle metadata 中（`ScratchClassifier.conformal_threshold` property）
- `safety_multiplier`：來自 runtime config（預設 1.1），每次 `process_panel` 開始時由 `_get_scratch_filter()` 讀最新值
- `effective_threshold` 在 `ScratchFilter.__init__` 計算並 cache；safety 變更 → 重建 filter（見 §B.4）

**決策邏輯：**
```
score > HIGH_THR  → 翻 OK（移出 anomaly_tiles）
score ≤ HIGH_THR  → 保留 NG（不動，走 RIC 物理覆檢）
```

**不設 LOW_THR / borderline zone**：
- 簡化決策邏輯
- RIC 為天然人工覆檢閥；borderline 留 NG 對現場流程零衝擊
- 若未來發現 borderline 量大且 RIC 負擔過重，再另起 spec 加入

**Safety multiplier 選擇：**
- 1.0：~2.6% 平均 leak（POC rerun 平均）
- **1.1（default）**：預估 ~1.5% leak、scratch recall 微降至 ~88%
- 1.2：預估 ~1% leak、scratch recall ~85%

實際值上線後用 DB `config_params` 動態調整。

### F. Runtime Config

仿造現有 `config_params` 機制（`capi_database.py::config_params` table + DB override chain）。新增 keys：

| Key | Type | Default | 說明 |
|---|---|---|---|
| `scratch_classifier_enabled` | bool | `True` | 總開關，false 時整段 skip |
| `scratch_safety_multiplier` | float | `1.1` | Threshold safety，1.0~1.5 合理範圍 |
| `scratch_bundle_path` | str | `deployment/scratch_classifier_v1.pkl` | bundle 路徑，可相對可絕對 |
| `scratch_dinov2_weights_path` | str | `deployment/dinov2_vitb14.pth` | DINOv2 base 權重路徑 |

所有參數 → 若 DB 中存在則覆寫 YAML；YAML 沒指定則用 spec 默認值。變更 → 寫入 `config_change_history`（現有 audit 機制）。

**Threshold override 以 `safety_multiplier` 而非直接指定數值：** 讓 conformal_threshold（來自模型訓練）作 single source of truth，safety 做業務調整。

### G. 離線部署

**檔案打包結構（產線無外網）：**

```
deployment/
├── scratch_classifier_v1.pkl       # LoRA + LogReg + metadata, ~1-2 MB
├── dinov2_vitb14.pth               # DINOv2 base weights, 330 MB
└── README.md                       # 部署步驟、checksum、版本對應
```

**產出流程：**
1. 在有外網的機器上跑 `train_final_model.py` → 輸出 `scratch_classifier_v1.pkl`
2. 跑現有 `scripts/over_review_poc/prepare_offline_model.py --export-state-dict deployment/dinov2_vitb14.pth` 備份 DINOv2 權重
3. `deployment/` 整個目錄 rsync / SCP 到產線 Linux 主機
4. 產線 CAPI 讀取 YAML 路徑（或 DB 覆寫）載入

**版本管理：**
- pkl 檔名帶 `_v1`、`_v2`... 舊版保留可 rollback
- bundle metadata 裡的 `git_commit` 和 `dataset_sha256` 可 audit
- `README.md` 記錄每版的 conformal_threshold、訓練日期、POC benchmark

**不做：**
- 自動 OTA 更新機制 — 手動部署即可
- Online 重訓 — POC 尚未支援

---

## 5. 產出部署用模型（對應 handoff §3.2）

新 script：`scripts/over_review_poc/train_final_model.py`

**CLI：**
```bash
python -m scripts.over_review_poc.train_final_model \
    --manifest datasets/over_review/manifest.csv \
    --transform clahe --clahe-clip 4.0 \
    --rank 16 --n-lora-blocks 2 --epochs 15 --alpha 16 \
    --calib-frac 0.2 \
    --output deployment/scratch_classifier_v1.pkl
```

**流程：**
1. Load manifest → 全 1748 筆
2. Group-aware 80/20 split → `proper_train` / `calib`
3. 在 `proper_train` 上 LoRA fine-tune 15 epochs（重用 `finetune_lora.py::_train_fold`）
4. 在 `proper_train` 抽特徵 + fit LogReg（class_weight="balanced"）
5. 在 `calib` 抽特徵 → `calib_max_ng_score = max(logreg.predict_proba(X_calib)[true_ng_mask, 1])`
6. Pickle 成 bundle：lora state_dict + logreg + metadata（含 `conformal_threshold = calib_max_ng_score`、dataset_sha256、git_commit）
7. Print 驗證摘要：conformal_threshold, calib set 分數分佈, est. threshold @ safety=1.1

**重用 POC 程式碼：**
- `features.py::build_transform_clahe`、`_CropDataset`、`_extract_cls`
- `finetune_lora.py::LoRALinear`、`_apply_lora`、`_train_fold`（可能要抽出 shared helper）
- `zero_leak_analysis.py::_group_aware_split`

**重複使用而非 import 的理由：** 目前這些 helper 散在不同 POC script 內。spec 不要求先做 refactor；`train_final_model.py` 先 import 用，若日後兩邊 drift 再提 refactor task。

---

## 6. DB Schema 變更（對應 handoff §3.3）

在 `capi_database.py` 的 schema init 內加新欄位（用 `ALTER TABLE ... ADD COLUMN ... IF NOT EXISTS` 風格，或 `PRAGMA table_info` 檢查後才加）：

```sql
-- tile_results
ALTER TABLE tile_results ADD COLUMN scratch_score REAL DEFAULT 0.0;
ALTER TABLE tile_results ADD COLUMN scratch_filtered INTEGER DEFAULT 0;

-- image_results
ALTER TABLE image_results ADD COLUMN scratch_filter_count INTEGER DEFAULT 0;
```

**Migration 策略：** 採「檢查欄位存在否 → 不存在才 ADD」pattern（SQLite 不支援 `ADD COLUMN IF NOT EXISTS`）。`capi_database.py::__init__` 裡插入 inline migration。

**舊資料相容：** default 值為 0 / 0 / 0，代表「classifier 未啟用或未跑」，跟 `scratch_filtered=False` 等價。

**查詢擴充：**
- `/v3/record/<id>` tile 詳情頁顯示 `scratch_score` 和 `scratch_filtered` 標記（若非 0）
- `/records` 列表頁可選 filter「含 scratch flip 的記錄」
- RIC 比對頁面不改 — 因為 RIC 比對基於 panel-level 最終判定，scratch flip 只影響 OK/NG 分佈

---

## 7. 測試計畫（對應 handoff §3.4）

### 7.1 單元測試

**`tests/test_scratch_classifier.py`：**
1. `test_preprocessing_matches_poc` — 用同一張圖，比對 `ScratchClassifier` preprocessing 輸出與 `scripts/over_review_poc/features.py::build_transform_clahe` 輸出的 tensor 差異 < 1e-5
2. `test_predict_score_range` — 任何輸入，score ∈ [0, 1]
3. `test_batch_equivalent` — `predict_batch([a, b])` 與 `[predict(a), predict(b)]` 差異 < 1e-5
4. `test_bundle_roundtrip` — 隨機產生 LoRA weights + LogReg → save → load → predict 相等
5. `test_metadata_invariants` — threshold = conformal × safety，所有欄位非空

**`tests/test_scratch_filter.py`：**
1. `test_filter_flips_high_score` — mock classifier 回固定高分 → anomaly_tiles 被清空、scratch_filter_count 等於原 NG 數
2. `test_filter_keeps_low_score` — mock classifier 回固定低分 → anomaly_tiles 不動、count = 0
3. `test_no_anomaly_tiles_is_noop` — 空 anomaly_tiles → apply 無副作用
4. `test_exception_safety` — classifier 某 tile 噴例外 → tile.scratch_score=0、tile 保留在 anomaly_tiles（safety default）

### 7.2 整合測試

**`tests/test_integration_scratch.py`：**
1. 準備兩張已知 panel：一張有 scratch、一張有 true_ng
2. Server 啟動 with `scratch_classifier_enabled=True`，送 TCP 請求
3. 驗證：
   - scratch panel → AI 回應 `OK`（被 flip）
   - true_ng panel → AI 回應 `NG@...(x,y)`（未被 flip）
4. DB 檢查：scratch panel 的 tile_results 應有 `scratch_filtered=1` 記錄

**前置：** 測試 fixture 需實際 bundle 檔案。若 CI 沒 GPU 則 skip 此 test（`pytest.mark.gpu`）。

### 7.3 Staging 驗證（部署後人工）

- 部署到 1 台 staging AOI 機跑 **1 個完整工作天** 流量
- 檢查 latency 分佈、scratch_filter_count 統計、對照 RIC 次日結果
- 成功標準：P95 latency 增加 < 2s、scratch filter 觸發率合理（觀察 baseline）、RIC 回報的 true_ng 中從 AI OK 翻過來的數量 < 3%

---

## 8. 錯誤處理與 Edge Cases

| 情境 | 行為 |
|---|---|
| Bundle 檔案缺失 | Log ERROR，設 `_SENTINEL_FAILED`，scratch filter 整段 bypass（不影響 PatchCore 主流程）|
| DINOv2 base 權重缺失 | 同上 |
| 單 tile predict 噴 exception | Log WARNING，tile 保留 NG（safety default），其他 tile 繼續 |
| GPU OOM | 由 `_gpu_lock` + lazy load 預防；若發生，classifier 記錯、後續不再載入本次 process |
| Config `scratch_classifier_enabled=False` | 整段跳過，無 latency penalty、無 DB 欄位更新（預設值 0）|
| 訓練資料 drift（新 scratch 樣本進來但沒重訓）| Out of scope — 現有 POC workflow 重訓即可；可在 dashboard 顯示 bundle 的 `trained_at` 供人工監控 |

---

## 9. 已知限制 / 未來工作

1. **Conformal 單 threshold 的變異度**：POC rerun 顯示 fold 間 leak 最高 8.1%。Safety=1.1 收緊，但真實生產資料分佈可能再不同。監控為準。
2. **僅支援 scratch 一類過檢**：overexposure / within_spec 等 8 類需各自訓練分類器，或研究 multi-label 模型（別 spec）。
3. **沒做 cross-model generalization 驗證**：目前訓練集包含全部光源的 scratch，但各光源樣本不均（W0F/WGF/STANDARD 各 < 10 筆）。若未來新光源加入，需重訓。
4. **Edge crop 的低訊號樣本**（~21% 漏檢）：POC 已判斷 ML 無解，需幾何規則處理，not in this spec。
5. **沒有自動回滾機制**：若 bundle 壞了，改 config `scratch_classifier_enabled=False` 是手動動作。

---

## 10. 成功標準

本 spec 對應 implementation 結束時，應滿足：

1. `ScratchClassifier` 類別有完整單元測試通過（§7.1）
2. `scratch_filter` 插入 `CAPIInferencer.process_panel` 且整合測試通過（§7.2）
3. `train_final_model.py` 可產出有效 bundle（檢查 metadata 完整 + predict 合理）
4. `deployment/` 目錄有可用 bundle + DINOv2 權重
5. Runtime config 可動態開關 + 調整 safety multiplier（DB 生效 without restart）
6. DB schema 自動 migrate 舊 DB（加新欄位，舊資料 default 0）
7. Staging 1 天驗證通過（§7.3）

---

## 11. 交接給 writing-plans

寫完 plan 時請依序切出的 Task：
1. 新增 bundle 規格 + `ScratchClassifier` 類別（純 model code，可獨立測）
2. 新增 `ScratchFilter` + `TileInfo`/`ImageResult` 欄位
3. DB schema migration
4. Runtime config 串入 `capi_config.py`
5. `process_panel` hook 插入 + integration test
6. `train_final_model.py` script
7. Bundle 實際產出與 staging 驗證
8. Web dashboard 顯示新欄位（低優先，可分開）

每 task 結束跑單元 + 整合 test 做 verification checkpoint。
