# RIC Report 時間篩選 + 漏檢 Review 功能設計

## Background

RIC Report 頁面目前沒有時間篩選維度，載入後顯示全量資料。隨著資料量成長且未來可能直接對接 DB，需要加入後端時間篩選能力，UI 風格同步 AI 推論紀錄 Tab。

此外，對於 AI 漏檢案例（AI=OK, RIC=NG），目前沒有機制讓使用者 Review 並記錄漏檢原因，不利於後續改善分析。

## Feature Summary

1. **時間篩選**：將 RIC Report 頁面從「先上傳/載入 → 再看報表」改為「直接從 DB 查詢顯示報表」，頂部加入時間篩選列（快捷按鈕 + 自訂日期），上傳功能收到右上角「匯入資料」按鈕。
2. **漏檢 Review**：新增第三個 sub-tab「漏檢 Review」，列出 AI=OK & RIC=NG 的案例，可逐筆填寫分類與備註，並提供統計圖表。

## 頁面流程變更

**現狀**：進入 → 上傳 XLS 或點「載入報表」 → 顯示報表
**新流程**：進入 → 自動從 DB 查詢（預設「全部」）→ 顯示報表 → 可用時間篩選列縮小範圍

時間篩選列為全局控制，切換日期時三個 sub-tab（統計分析 / 逐筆明細 / 漏檢 Review）的資料都同步更新。

---

## Part 1: 時間篩選

### UI 變更

#### 移除項目

- 移除上傳區塊作為首頁主體（`#client_uploadCard` 整個大區塊）
- 移除「載入報表」按鈕區塊（DB 有資料時的藍色提示區）
- 移除 `client_fileInfo` 頂部檔案資訊列

#### 新增：時間篩選列

位置：報表區域頂部（sub-tabs 上方），風格同 AI 推論紀錄 Tab 的 `.inf-date-bar`。

組成：
- 快捷按鈕：`今日` / `7天` / `30天` / `全部`
- 預設 active：`全部`
- 自訂日期：起始 `<input type="date">` ~ 結束 `<input type="date">` + `查詢` 按鈕

#### 新增：右上角「匯入資料」按鈕

位置：`.ric-header` 區域右側。

行為：
1. 點擊後展開折疊式上傳區塊（不用 modal）
2. 上傳區塊包含現有的拖放 upload zone
3. 上傳成功後自動重新查詢當前日期範圍，刷新報表
4. 清除資料庫按鈕移入此折疊區塊內

#### 報表內容

保持不變：stat cards、4 種圖表、每日追蹤圖、逐筆明細表。資料來源從前端 `rawData` 全量計算改為 API 回傳。

### API 變更

#### 改造 `GET /api/ric/client-data`

**新增 query 參數：**
- `start_date`（YYYY-MM-DD，optional）— 篩選 `time_stamp >= start_date`
- `end_date`（YYYY-MM-DD，optional）— 篩選 `time_stamp <= end_date + ' 23:59:59'`
- 兩者都不帶 = 查全部（現有行為）

**回傳格式擴充：**

```json
{
  "success": true,
  "total": 1234,
  "summary": {
    "total": 1234,
    "aoiNG": 100,
    "aiNG": 80,
    "ricNG": 50,
    "aoiCorrect": 1100,
    "aiCorrect": 1150,
    "aoiOver": 60,
    "aoiOverRate": 60.0,
    "aiOver": 30,
    "aiOverRate": 37.5,
    "aiMiss": 5,
    "aiMissRate": 10.0,
    "revival": 40,
    "revivalRate": 40.0,
    "combos": {"OK/OK/OK": 900, "NG/OK/OK": 40},
    "daily": {
      "2026-03-25": {
        "total": 100,
        "aoiCorrect": 90,
        "aiCorrect": 95,
        "aiMiss": 2,
        "aiOver": 3,
        "aoiOver": 5
      }
    },
    "missReviewStats": {
      "total": 5,
      "reviewed": 3,
      "unreviewed": 2,
      "byCategory": {
        "dust_misfilter": 1,
        "threshold_high": 1,
        "ric_misjudge": 1,
        "outside_aoi_area": 0,
        "other": 0
      }
    }
  },
  "records": [
    {
      "time_stamp": "2026-03-25 15:20:03",
      "pnl_id": "PNL001",
      "mach_id": "MC01",
      "result_eqp": "NG",
      "result_ai": "OK",
      "result_ric": "OK",
      "datastr": "",
      "miss_review": null
    }
  ]
}
```

漏檢的 record（AI=OK, RIC=NG）的 `miss_review` 欄位會帶上已有的 review 資料：
```json
{
  "id": 1,
  "category": "dust_misfilter",
  "note": "OMIT 交叉比對誤判",
  "updated_at": "2026-03-26 10:30:00"
}
```
非漏檢的 record 此欄位為 `null`。

### 後端實作

#### `capi_database.py`

修改 `get_client_accuracy_records()`：
- 新增 `start_date`、`end_date` 參數
- SQL 加上 `WHERE` 日期篩選條件
- 日期格式驗證（同 `get_inference_stats` 的 `_DATE_RE`）
- LEFT JOIN `miss_review` 表，帶出已有的 review 資料

#### `capi_web.py`

修改 `_handle_client_data_api()`：
- 從 query 取得 `start_date`、`end_date` 參數
- 傳入 `get_client_accuracy_records()`
- 對回傳的 records 計算 summary 統計（將前端 `computeStats()` 邏輯搬到後端）
- 計算 `missReviewStats`（各分類計數、已 Review / 未 Review 數量）
- 回傳擴充後的 JSON 格式

#### 統計計算邏輯（從前端搬到後端）

沿用現有 `computeStats()` 的邏輯，在後端 Python 計算：
- `aoiNG`, `aiNG`, `ricNG`：各判定 NG 數量
- `aoiCorrect`, `aiCorrect`：與 DATASTR 判定一致數量
- `aoiOver`：AOI=NG 且 RIC=OK
- `aiOver`：AI=NG 且 RIC=OK
- `aiMiss`：AI=OK 且 RIC=NG
- `revival`：AOI=NG 且 AI=OK 且 RIC=OK
- `combos`：AOI/AI/RIC 三方組合計數
- `daily`：按日分組統計
- 各比率計算公式不變

注意：已移除 AOI 漏檢率，後端不計算 `aoiMiss` / `aoiMissRate`。

#### RIC 判定邏輯

前端 `deriveJudgment(datastr)` 邏輯（`datastr` 含 'NG' → NG，否則 OK）搬到後端。在統計計算時，使用 DB 中 `datastr` 欄位做判定。

### 前端 JS 變更

#### `clientComp` 模組改造

1. **移除** `loadFile()` 中的自動 `renderAll()` → 改為上傳成功後呼叫 `loadData()`
2. **移除** `loadFromDB()` → 功能由 `loadData()` 取代
3. **移除** `computeStats()` → 統計由後端計算
4. **新增** `quickFilter(range)` — 同 `inferenceTab.quickFilter`，計算日期範圍後呼叫 `loadData()`
5. **新增** `customFilter()` — 讀取自訂日期 input 後呼叫 `loadData()`
6. **新增** `loadData(startDate, endDate)` — fetch `/api/ric/client-data?start_date=...&end_date=...`，收到回傳後：
   - 用 `summary` 渲染 stat cards 和圖表
   - 用 `records` 渲染逐筆明細表
   - 用 `summary.missReviewStats` 渲染漏檢 Review tab
   - DB 無資料時顯示空狀態提示
7. **修改** `renderStatsCards()`、`renderCharts()`、`renderDetails()` — 接收後端回傳的資料格式
8. **頁面載入時** 自動呼叫 `loadData('', '')` 查詢全部

#### 上傳流程調整

上傳成功後：
1. 關閉折疊式上傳區塊
2. 呼叫 `loadData()` 以當前篩選條件重新查詢
3. 顯示上傳結果提示訊息

---

## Part 2: 漏檢 Review

### 分類定義

| key | 顯示名稱 | 說明 |
|-----|---------|------|
| `dust_misfilter` | Dust 誤濾 | AI 偵測到異常但被 dust filtering 過濾掉 |
| `threshold_high` | 閾值偏高 | 異常分數有反應但未達閾值 |
| `ric_misjudge` | RIC 誤判 | 人工複檢本身判錯，實際為 OK |
| `outside_aoi_area` | 漏檢區域不在 AOI 提供區域 | 缺陷位置不在 AOI 送檢的影像範圍內 |
| `other` | 其他 | 以上皆非，需在備註說明 |

### DB Schema

新增 `miss_review` 資料表：

```sql
CREATE TABLE IF NOT EXISTS miss_review (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_record_id INTEGER NOT NULL,
    category TEXT NOT NULL,
    note TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (client_record_id) REFERENCES client_accuracy_records(id),
    UNIQUE(client_record_id)
);
CREATE INDEX IF NOT EXISTS idx_miss_review_client ON miss_review(client_record_id);
```

- `client_record_id` 為 UNIQUE，每筆漏檢最多一筆 review
- `category` 值為上表的 key
- `note` 為選填備註文字

### API

#### `POST /api/ric/miss-review`

儲存或更新單筆漏檢 Review。使用 `INSERT OR REPLACE`（UPSERT by `client_record_id`）。

**Request body：**
```json
{
  "client_record_id": 42,
  "category": "dust_misfilter",
  "note": "OMIT 交叉比對把真實缺陷濾掉了"
}
```

**驗證：**
- `client_record_id` 必須存在於 `client_accuracy_records`
- `category` 必須是 5 個合法值之一
- `note` 為 optional，預設空字串

**Response：**
- 成功：`{"success": true, "id": 1, "message": "Review 已儲存"}`
- 參數錯誤：`{"success": false, "error": "描述"}`

#### `DELETE /api/ric/miss-review`

刪除單筆 Review（允許使用者撤回）。

**Request body：**
```json
{
  "client_record_id": 42
}
```

**Response：**
- 成功：`{"success": true, "message": "Review 已刪除"}`

### 前端 UI：漏檢 Review Tab

#### Tab 結構（由上到下）

**1. 統計卡片列**

4 張卡片，共用 `.stats-grid` 樣式：

| 卡片 | 內容 |
|------|------|
| 漏檢總數 | `missReviewStats.total` 筆 |
| 已 Review | `missReviewStats.reviewed` 筆 + 佔比 |
| 未 Review | `missReviewStats.unreviewed` 筆 + 佔比 |
| 完成率 | `reviewed / total * 100`% |

**2. 圓餅圖**

一個 Chart.js doughnut chart，顯示各分類的數量分佈。資料來源：`missReviewStats.byCategory`。未 Review 的案例算一個獨立分類顯示（灰色）。

**3. 可操作列表**

表格欄位：

| # | 檢測時間 | Panel ID | 機台 | AOI | AI | RIC | 分類 | 備註 | 操作 |
|---|---------|----------|------|-----|----|----|------|------|------|

- 資料來源：從 `records` 中篩選 AI=OK & RIC=NG 的筆數
- **分類欄**：`<select>` 下拉，選項為 5 個分類。已 Review 的顯示已選值，未 Review 的預設為空（placeholder「請選擇」）
- **備註欄**：`<input type="text">`，可直接編輯
- **操作欄**：
  - 未儲存/有修改 → 顯示「儲存」按鈕
  - 已儲存 → 顯示「已儲存 ✓」+ 小字「刪除」連結
- 儲存時 `fetch POST /api/ric/miss-review`，成功後更新本地狀態和統計卡片/圓餅圖（不需整頁 reload）

#### 時間篩選連動

漏檢 Review tab 使用同一次 `loadData()` 的回傳資料。切換日期時：
- 列表重新渲染（只顯示該時間區間內的漏檢案例）
- 統計卡片和圓餅圖重新計算
- 已 Review 的資料不會因篩選而丟失（存在 DB 中）

---

## Error Handling

- `start_date` / `end_date` 格式錯誤 → 回傳 `{"success": false, "error": "Invalid date format"}`
- DB 無資料 → 回傳 `{"success": true, "total": 0, "summary": {各欄位為0}, "records": []}`
- 前端收到 `total: 0` → 顯示「無資料，請先匯入檢測資料或調整日期範圍」提示
- `miss-review` API 的 `client_record_id` 不存在 → 回傳 `{"success": false, "error": "Record not found"}`
- `miss-review` API 的 `category` 非法 → 回傳 `{"success": false, "error": "Invalid category"}`

## 未來擴充預留

- 篩選維度（按區域、按機台、按分類）：目前不實作，但 API 回傳的 `records` 結構已包含 `mach_id` 等欄位，前端可直接加 filter 不需改後端
- 直接對接外部 DB：`get_client_accuracy_records()` 的日期篩選介面已標準化，未來換資料來源只需改 SQL query
