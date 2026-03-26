# RIC Report 時間篩選功能設計

## Background

RIC Report 頁面目前沒有時間篩選維度，載入後顯示全量資料。隨著資料量成長且未來可能直接對接 DB，需要加入後端時間篩選能力，UI 風格同步 AI 推論紀錄 Tab。

## Feature Summary

將 RIC Report 頁面從「先上傳/載入 → 再看報表」改為「直接從 DB 查詢顯示報表」，頂部加入時間篩選列（快捷按鈕 + 自訂日期），上傳功能收到右上角「匯入資料」按鈕。

## 頁面流程變更

**現狀**：進入 → 上傳 XLS 或點「載入報表」 → 顯示報表
**新流程**：進入 → 自動從 DB 查詢（預設「全部」）→ 顯示報表 → 可用時間篩選列縮小範圍

## UI 變更

### 移除項目

- 移除上傳區塊作為首頁主體（`#client_uploadCard` 整個大區塊）
- 移除「載入報表」按鈕區塊（DB 有資料時的藍色提示區）
- 移除 `client_fileInfo` 頂部檔案資訊列

### 新增：時間篩選列

位置：報表區域頂部（sub-tabs 上方），風格同 AI 推論紀錄 Tab 的 `.inf-date-bar`。

組成：
- 快捷按鈕：`今日` / `7天` / `30天` / `全部`
- 預設 active：`全部`
- 自訂日期：起始 `<input type="date">` ~ 結束 `<input type="date">` + `查詢` 按鈕

### 新增：右上角「匯入資料」按鈕

位置：`.ric-header` 區域右側。

行為：
1. 點擊後展開折疊式上傳區塊（不用 modal）
2. 上傳區塊包含現有的拖放 upload zone
3. 上傳成功後自動重新查詢當前日期範圍，刷新報表
4. 清除資料庫按鈕移入此折疊區塊內

### 報表內容

保持不變：stat cards、4 種圖表、每日追蹤圖、逐筆明細表。資料來源從前端 `rawData` 全量計算改為 API 回傳。

## API 變更

### 改造 `GET /api/ric/client-data`

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
      "datastr": ""
    }
  ]
}
```

## 後端實作

### `capi_database.py`

修改 `get_client_accuracy_records()`：
- 新增 `start_date`、`end_date` 參數
- SQL 加上 `WHERE` 日期篩選條件
- 日期格式驗證（同 `get_inference_stats` 的 `_DATE_RE`）

### `capi_web.py`

修改 `_handle_client_data_api()`：
- 從 query 取得 `start_date`、`end_date` 參數
- 傳入 `get_client_accuracy_records()`
- 對回傳的 records 計算 summary 統計（將前端 `computeStats()` 邏輯搬到後端）
- 回傳擴充後的 JSON 格式

### 統計計算邏輯（從前端搬到後端）

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

## 前端 JS 變更

### `clientComp` 模組改造

1. **移除** `loadFile()` 中的自動 `renderAll()` → 改為上傳成功後呼叫 `loadData()`
2. **移除** `loadFromDB()` → 功能由 `loadData()` 取代
3. **移除** `computeStats()` → 統計由後端計算
4. **新增** `quickFilter(range)` — 同 `inferenceTab.quickFilter`，計算日期範圍後呼叫 `loadData()`
5. **新增** `customFilter()` — 讀取自訂日期 input 後呼叫 `loadData()`
6. **新增** `loadData(startDate, endDate)` — fetch `/api/ric/client-data?start_date=...&end_date=...`，收到回傳後：
   - 用 `summary` 渲染 stat cards 和圖表
   - 用 `records` 渲染逐筆明細表
   - DB 無資料時顯示空狀態提示
7. **修改** `renderStatsCards()`、`renderCharts()`、`renderDetails()` — 接收後端回傳的資料格式
8. **頁面載入時** 自動呼叫 `loadData('', '')` 查詢全部

### 上傳流程調整

上傳成功後：
1. 關閉折疊式上傳區塊
2. 呼叫 `loadData()` 以當前篩選條件重新查詢
3. 顯示上傳結果提示訊息

## RIC 判定邏輯

前端 `deriveJudgment(datastr)` 邏輯（`datastr` 含 'NG' → NG，否則 OK）需要搬到後端。在統計計算時，使用 DB 中 `datastr` 欄位做判定。

## Error Handling

- `start_date` / `end_date` 格式錯誤 → 回傳 `{"success": false, "error": "Invalid date format"}`
- DB 無資料 → 回傳 `{"success": true, "total": 0, "summary": {各欄位為0}, "records": []}`
- 前端收到 `total: 0` → 顯示「無資料，請先匯入檢測資料或調整日期範圍」提示
