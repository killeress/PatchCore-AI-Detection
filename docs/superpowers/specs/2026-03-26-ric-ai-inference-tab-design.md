# RIC 頁面新增「AI 推論紀錄」分頁 — 設計規格

## 目標

在現有 `/ric` 頁面頂部新增分頁 Tab bar，增加「AI 推論紀錄」分頁，提供伺服器推論紀錄的統計總覽與趨勢分析，不影響原有 RIC Report 功能。

## 方案

**純前端 Tab 切換**：在 `ric_report.html` 模板頂部新增頂層 Tab bar，兩個 Tab 內容區塊在同一頁面，用 CSS `display` 切換。「AI 推論紀錄」的資料透過 AJAX 呼叫新 API endpoint 取得。

選擇理由：切換瞬間、與現有 client sub-tabs 模式一致、不影響 RIC Report 邏輯。

## 資料來源

- 資料庫表：`inference_records`（主要）、`image_results`（異常類型分類）
- 所有資料由伺服器推論產生，不需外部匯入

## 頁面結構

### 頂層 Tab Bar

| Tab | 說明 |
|-----|------|
| 📊 RIC Report | 現有 RIC 比對報表，完全不變 |
| 🤖 AI 推論紀錄 | 新增：推論統計總覽與趨勢分析 |

Tab 預設顯示「RIC Report」，維持現有行為。

### AI 推論紀錄 — 組件

#### 1. 日期篩選列

- **快捷按鈕**：今日 / 7天 / 30天 / 全部
- **自訂範圍**：起始日 ~ 結束日 + 查詢按鈕
- 切換快捷按鈕時自動查詢；自訂範圍需點「查詢」觸發
- 預設：今日

#### 2. 統計卡片（5 張）

| 卡片 | 資料來源 | 顏色 |
|------|---------|------|
| 總推論次數 | COUNT(*) from inference_records (篩選日期) | 白色 |
| AOI 判 NG | COUNT WHERE machine_judgment != 'OK'（即含 'NG' 的任何值） | 橘色 |
| AI 判 NG | COUNT WHERE ai_judgment LIKE 'NG%'（含 `NG@...` 座標格式） | 藍色 |
| AI 救回 | COUNT WHERE machine_judgment != 'OK' AND ai_judgment = 'OK' | 綠色 |
| ERR 錯誤 | COUNT WHERE ai_judgment LIKE 'ERR:%' | 琥珀色 |

每張卡片顯示數量與佔總數百分比。

#### 3. 圖表區

**第一排（2:1 比例）：**

- **每日推論趨勢（AOI vs AI）**：混合圖
  - 柱狀：每日總推論數
  - 折線（橘）：每日 AOI NG 數
  - 折線（藍）：每日 AI NG 數
  - 使用 Chart.js（與現有 RIC Report 一致）

- **AOI vs AI 判定分布**：並排兩個圓餅圖
  - AOI：OK / NG 比例
  - AI：OK / NG / ERR 比例

**第二排（1:1:1 三等分）：**

- **機台統計**：水平長條圖，每台機台顯示 AOI NG率 vs AI NG率 雙指標
- **產品型號統計**：水平長條圖，按 model_id 統計推論筆數
- **ERR 類型分布**：水平長條圖，按 `error_message` 前綴分類統計

**第三排（全寬）：**

- **AOI vs AI 判定交叉比對矩陣**：4 格卡片
  - AOI OK → AI OK（一致 OK）
  - AOI NG → AI OK（AI 救回）
  - AOI OK → AI NG（AI 額外攔截）
  - AOI NG → AI NG（一致 NG）

## 新增 API

### `GET /api/ric/inference-stats`

**查詢參數：**
- `start_date`（可選）：起始日期，格式 `YYYY-MM-DD`
- `end_date`（可選）：結束日期，格式 `YYYY-MM-DD`

**回傳 JSON：**
```json
{
  "success": true,
  "summary": {
    "total": 5678,
    "aoi_ng": 812,
    "ai_ng": 680,
    "ai_revival": 185,
    "err_count": 75
  },
  "daily_trend": [
    { "date": "2026-03-25", "total": 320, "aoi_ng": 45, "ai_ng": 38, "err": 3 }
  ],
  "by_machine": [
    { "machine": "M001", "total": 1200, "aoi_ng_rate": 18.0, "ai_ng_rate": 5.0 }
  ],
  "by_model": [
    { "model": "MODEL-A", "total": 2340 }
  ],
  "err_types": [
    { "type": "模型載入失敗", "count": 32 }
  ],
  "cross_matrix": {
    "ok_ok": 4681,
    "ng_ok": 185,
    "ok_ng": 53,
    "ng_ng": 627
  }
}
```

## 實作範圍

### 後端（capi_web.py + capi_database.py）

1. **capi_database.py**：新增 `get_inference_stats(start_date, end_date)` 方法
   - 查詢 `inference_records` 做聚合統計
   - ERR 類型：解析 `error_message` 欄位，取 `ERR:` 之後的描述做分類
   - 交叉比對：根據 `machine_judgment` 與 `ai_judgment` 組合計算

2. **capi_web.py**：新增路由 `GET /api/ric/inference-stats`
   - 解析 `start_date`、`end_date` 參數
   - 呼叫 `get_inference_stats()` 回傳 JSON

### 前端（templates/ric_report.html）

1. **頂層 Tab bar HTML**：在頁面最上方（現有內容之前）插入
2. **Tab 切換 CSS**：`.top-tab-bar`、`.top-tab`、`.top-tab.active`、`.tab-content`
3. **AI 推論紀錄內容區**：日期篩選列 + 統計卡片 + 圖表容器
4. **JS 模組 `inferenceTab`（IIFE）**：
   - `init()`：綁定 Tab 切換、日期篩選事件
   - `loadData(startDate, endDate)`：AJAX GET `/api/ric/inference-stats`
   - `renderCards(summary)`：渲染統計卡片
   - `renderCharts(data)`：使用 Chart.js 渲染 5 個圖表
   - 圖表實例管理（resize / destroy / recreate）

### 不修改的部分

- 現有 RIC Report 的所有 HTML、CSS、JS 邏輯
- 現有 `clientComp` 模組
- 現有 API endpoints
- 資料庫 schema（不需新增表或欄位）

## 技術決策

- **Chart.js**：沿用現有頁面已載入的 Chart.js，不引入新依賴
- **IIFE 封裝**：`inferenceTab` 模組與 `clientComp` 完全隔離，避免命名衝突
- **懶載入**：切換到「AI 推論紀錄」Tab 時才發 API 請求，不在頁面載入時預取
- **Tab 狀態**：不使用 URL hash，避免影響現有書籤行為
