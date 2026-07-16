# 新系統 PatchCore 模型訓練與啟用 SOP

> 適用對象：現場工程人員
> 核心原則：**用戶不訓練模型；由工程人員透過 Web 介面完成訓練與啟用。**

## 1. 系統介面流程

```text
模型訓練
  → 新機種 PatchCore
  → 完整訓練
  → 選擇訓練資料
  → 前處理
  → 訓練 tile 審核
  → PatchCore 模型訓練
  → 訓練完成報告
  → 模型庫
  → 啟用
  → 重啟 server
```

新系統完整訓練會產生：

```text
5 種光源 × INNER / EDGE = 10 個 PatchCore 子模型
```

## 2. 訓練模型

### 2.1 進入訓練介面

1. 開啟 CAPI Web。
2. 進入「模型訓練」頁面，或直接開啟：

   ```text
   http://<server>/training
   ```

3. 在「新機種 PatchCore」卡片按「開始訓練 →」。

| 畫面 | 用途 |
|---|---|
| 模型訓練 | 進入新機種訓練功能 |
| `/train/new` | 選擇完整訓練或局部重訓 |
| `/train/new/select` | 選擇 PANEL 與訓練資料來源 |
| `/train/new/progress` | 查看前處理進度 |
| `/train/new/review/<job_id>` | 審核訓練 tile |
| `/train/new/progress?job_id=...` | 查看 10 個子模型訓練進度 |
| `/train/new/done/<job_id>` | 查看訓練完成報告 |
| `/models` | 檢查並啟用模型 |

### 2.2 Step 1／6：選擇訓練模型

在「Step 1／6・選擇訓練模型」畫面：

1. 選擇「完整訓練」。
2. 確認畫面說明為「5 光源 × inner/edge，共 10 個 PT」。
3. 按「下一步：選擇訓練資料 →」。

「局部重訓」只用於既有模型的指定子模型修正；新系統首次建立模型不要選局部重訓。

### 2.3 Step 2／6：選擇訓練資料

系統提供兩種資料來源：

#### A. 最近推論紀錄

1. 保持「最近推論紀錄」選項。
2. 按「重新整理」。
3. 系統會列出最近 3 天、指定機種且 AOI 判定為 OK 的 PANEL。
4. 勾選正常 PANEL。

#### B. 手動資料夾

沒有足夠 AOI OK 紀錄時：

1. 選擇「手動資料夾」。
2. 輸入「機種 ID」。
3. 輸入「Batch 根目錄」。
4. 按「掃描資料夾」。
5. 確認掃描出的 PANEL 都屬於同一機種，且影像是正常樣本。
6. 勾選「我已確認這批資料是可用於 PatchCore 訓練的正常樣本」。

#### PANEL 勾選原則

- 系統最低允許選 1 片 PANEL；正式平展建議至少 3 片。
- 優先選擇不同批次、亮度與正常外觀變化的 PANEL。
- 不可選入 NG、瑕疵、髒污、嚴重過曝或嚴重模糊影像。
- 完整訓練必須涵蓋 INNER 與 EDGE；畫面若顯示「尚缺 INNER／EDGE」，要先補資料。

### 2.4 訓練設定（進階）

「訓練設定（進階）」與「影像前處理」維持系統預設值即可，現場不要自行調整。

| 介面欄位 | 標準值 |
|---|---:|
| `batch_size` | 8 |
| `coreset_ratio` | 0.1 |
| `max_epochs` | 固定 1 |
| `precision` | `float16` |
| `feature layers` | `layer2 + layer3` |
| 特徵鄰域聚合 | 3×3 |
| 特徵域清洗 | 關閉 |
| `tile step` | 256 |

若專案沒有特別核准，不要改變上述設定，也不要自行修改影像前處理順序或參數。

確認 PANEL 與設定後，按「下一步：前處理 →」。

### 2.5 Step 3／6：前處理

系統會自動執行：

- PANEL 前景與邊界偵測。
- 5 種光源篩選。
- 影像切 tile。
- tile 分類為 INNER 或 EDGE。
- 建立訓練 tile pool。

工程人員要做的事：

1. 等待狀態變成可審核。
2. 記錄畫面上的 `job_id`。
3. 若顯示錯誤，先保留錯誤訊息與 job ID，不要連續重複送出訓練。

### 2.6 Step 4／6：訓練 tile 審核

在「訓練 tile 審核」畫面，依序檢查 5 個光源及 INNER／EDGE tile：

| 操作 | 判定原則 |
|---|---|
| 保留／Accept | 清楚、正常、屬於該機種與該光源的 tile |
| 拒絕／Reject | 瑕疵、髒污、文字／MARK、空白、過曝、模糊、裁切錯誤或主體不完整 |

注意事項：

- EDGE 的正常邊界、光影與幾何變化要保留，不要全部拒絕。
- 若某個光源或區域的 tile 數量特別少，優先補充 PANEL，不要只靠 reject／accept 硬湊。
- 每個要訓練的光源／區域至少需要 30 個 accepted tile。

審核完成後按「開始訓練 →」。

### 2.7 Step 5／6：PatchCore 模型訓練

系統會依序訓練 10 個子模型：

```text
G0F00000-inner / G0F00000-edge
R0F00000-inner / R0F00000-edge
W0F00000-inner / W0F00000-edge
WGF50500-inner / WGF50500-edge
STANDARD-inner / STANDARD-edge
```

畫面會顯示：

- 每個 unit 的「等待中／訓練中／完成／失敗」。
- 訓練 log。
- 預估剩餘時間。

訓練期間不要重啟 server、刪除模型檔或移動訓練資料。若中途取消，通常必須重新從 Step 1 開始。

### 2.8 Step 6／6：訓練完成報告

在「訓練完成」畫面，確認以下內容：

- 顯示「訓練完成」。
- `success units` 為 **10 / 10**。
- 沒有 unit 顯示「失敗」或「跳過」。
- 已記錄 `job ID`、bundle path、訓練時間與來源 PANEL。
- 整體 AUROC 與各 unit 品質指標有資料；若顯示 `n/a`，先不要啟用。

確認完成後，進入「模型庫 →」或直接開啟 `/models`。

## 3. 啟用模型

### 3.1 模型庫介面說明

在「模型庫」畫面，每個 bundle 會顯示：

| 介面項目 | 說明 |
|---|---|
| 機種 ID | 模型所屬機種 |
| Bundle 名稱 | 該次訓練產生的模型版本 |
| 訓練時間／Job ID | 追查訓練紀錄用 |
| Panel／Inner／Edge／NG | 訓練資料數量摘要 |
| 路徑 | bundle 實際位置 |
| `細節／訓練資料` | 查看完整訓練設定與品質報告 |
| `啟用` | 將此 bundle 設為目前使用模型 |
| `啟用中` | 代表資料庫已標記此 bundle 為 active |

### 3.2 啟用步驟

1. 在 `/models` 找到本次最新訓練的機種與 bundle。
2. 按「細節／訓練資料」。
3. 確認：
   - machine ID 正確。
   - Job ID 與本次訓練一致。
   - `success units` 為 10 / 10。
   - bundle path 正確。
   - 訓練設定與 tile step 沒有異常。
4. 回到 bundle 操作區，按「啟用」。
5. 確認啟用訊息：

   ```text
   啟用成功，請重啟 server 才會生效
   ```

啟用只會更新模型配置；server 未重啟前，推論服務不會載入新模型。

### 3.3 重啟並確認生效

依現場服務管理方式選一種執行：

```bash
systemctl restart capi_server
```

或：

```bash
./start_server.sh restart --no-tail
```

重啟後確認：

1. 開啟 dashboard，確認 server 為正常運行。
2. 查詢：

   ```bash
   curl http://127.0.0.1:<WEB_PORT>/api/status
   ```

3. 狀態應顯示新架構與完整模型載入，內容類似：

   ```text
   5 lighting × inner/edge, 10/10 loaded
   ```

4. 使用一片已知 OK PANEL 實際送測，確認系統能正常回應。
5. 使用一片已知 NG PANEL 實際送測，確認結果與預期一致。
6. 回到 `/models`，確認該 bundle 顯示「啟用中」。

## 4. 現場完成判定

以下條件全部符合，才算模型訓練與啟用完成：

- [ ] 完整訓練成功，`10 / 10` unit 完成。
- [ ] 模型庫已確認正確 bundle。
- [ ] bundle 已顯示「啟用中」。
- [ ] server 已重啟。
- [ ] 狀態顯示 `10/10 loaded`。
- [ ] 已完成至少一片 OK 與一片 NG 的實際送測。

