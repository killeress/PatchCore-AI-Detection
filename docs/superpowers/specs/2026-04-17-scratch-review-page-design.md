# Scratch 救回審查頁面 Design

**Date:** 2026-04-17
**Topic:** 獨立頁面，讓 QA 逐圖檢查 DINOv2 scratch 分類器是否誤救

## 目標

CAPI AI 的 scratch filter（DINOv2 分類器）會把 PatchCore 判定的異常 tile 再做一次判斷，判為「刮痕」就翻為 OK。誤救（判成刮痕，實際是真缺陷）會造成漏檢。

需要一個獨立頁面：

1. 範圍為全系統 AI 推論（不只 RIC 過檢批次）
2. 以圖片為中心、可一眼掃過多張 tile heatmap
3. 可標記「誤救」，日後當作 DINOv2 再訓練負樣本
4. 支援時間篩選、兩種排序切換
5. 排版沿用 RIC 頁的 dark 主題與 stat-card 樣式

## 資料層

新增一張表 `scratch_rescue_review`（`capi_database.py` 建表 SQL 與 `over_review` 比鄰）：

```sql
CREATE TABLE IF NOT EXISTS scratch_rescue_review (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tile_result_id INTEGER NOT NULL,
    is_misrescue INTEGER NOT NULL DEFAULT 1,
    note TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (tile_result_id) REFERENCES tile_results(id),
    UNIQUE(tile_result_id)
);
CREATE INDEX IF NOT EXISTS idx_scratch_rescue_review_tile ON scratch_rescue_review(tile_result_id);
```

**為什麼以 tile 為單位？** Scratch 判斷本身就是 tile-level（`tile_results.scratch_filtered=1`）；未來再訓練也是 tile-level 負樣本。

新增 DB 方法：

- `list_scratch_rescued_tiles(start_date, end_date, order_by, limit, offset)`
  - 聯表 `tile_results` + `image_results` + `inference_records` + `scratch_rescue_review`
  - 篩選 `tile_results.scratch_filtered = 1`
  - 篩選 `inference_records.created_at` 在日期範圍內
  - `order_by = 'latest'` → `inference_records.created_at DESC`
  - `order_by = 'score_asc'` → `tile_results.scratch_score ASC`
  - 回傳：list of dict，含 tile_id, record_id, glass_id, machine_no, created_at, image_name, heatmap 相對路徑, tile x/y, scratch_score, is_misrescue, note
- `count_scratch_rescued_tiles(start_date, end_date)` → `{total: N, marked: M}`
- `mark_scratch_misrescue(tile_result_id, note)` → UPSERT
- `unmark_scratch_misrescue(tile_result_id)` → DELETE

## API

```
GET  /scratch-review
GET  /api/scratch-review/list?start_date=&end_date=&order=latest|score_asc&limit=24&offset=0
POST /api/scratch-review/mark    body: {tile_id, note?}
POST /api/scratch-review/unmark  body: {tile_id}
```

- heatmap URL：重用 `/heatmaps/<relative_path>`（既有 `_handle_static_file`）
- 排序白名單：僅接受 `latest` 或 `score_asc`，其他值退回 `latest`（避免 SQL 注入）

## 前端

新增 `templates/scratch_review.html`，繼承 `base.html`，沿用 RIC 的 CSS 命名（`stat-card`, `table-wrap`, `badge`, colour tokens）。

版面：

```
Header：🩹 Scratch 救回審查
篩選列：日期（start→end）、排序 toggle、每頁數
Stat cards：總救回 tile 數 / 已標記誤救 tile 數
圖片網格：每格 tile heatmap + score + pnl_id + mach_id + 時間 + 「🚩 誤救」按鈕
分頁：上一頁 / 頁碼 / 下一頁
```

圖卡互動：

- 點圖 → 開 lightbox 放大（沿用 `base.html` 既有 `openLightbox`）
- 點 pnl_id → 跳 `/record/<record_id>`
- 點「🚩 誤救」→ 立即 POST，標記成功後卡片加紅框 + 按鈕變「✓ 已標記」（再點取消）
- 排序 toggle / 日期 / 頁碼改變 → 重新 fetch list

## 導航

`templates/base.html` nav 在「統計報表」後插入：

```html
<a href="/scratch-review" ...>🩹 Scratch 審查</a>
```

## 不在此 Spec 範圍

- 負樣本匯出為訓練集（之後由 `retrain-scratch-classifier` skill 流程接手，讀 `scratch_rescue_review` 表即可）
- 批次標記 / 搜尋關鍵字 / 依機台篩選
- 導出 CSV

## 風險

- 老資料 `tile_results.heatmap_path` 若為空則該 tile 無圖可秀 → 顯示文字佔位符，不中斷
- 日期範圍查詢資料量可能很大（全系統級）→ 預設限定最近 7 天，強制 limit 上限 100

## 驗證

1. 啟動 `python capi_server.py --config server_config_local.yaml`
2. 瀏覽 `http://localhost:8080/scratch-review`
3. 頁面應顯示最新被救回的 tile 圖卡（若 DB 有 scratch_filtered=1 的 tile）
4. 點「🚩 誤救」應立即更新狀態
5. 切換排序 / 改日期應重新拉取
