# Record Rerun Inference Design

## Background

開發過程中經常發現 bug、修改推論程式後需要重新跑一次相同的 panel 來驗證修正。目前只能逐張圖片透過 debug 頁面重跑，無法整片 panel 重新推論並覆蓋原紀錄。

## Feature Summary

在 record detail 頁面（`/record/<id>` 和 `/v3/record/<id>`）新增「重新推論」按鈕，點擊後背景執行整片 panel 推論，完成後自動刷新頁面顯示新結果。舊紀錄與 heatmap 圖片直接覆蓋。

## API Design

### POST `/api/record/<id>/rerun`

觸發重新推論。

**Request:** 無 body。

**行為：**
1. 從 DB 讀取原始紀錄的 `image_dir`、`model_id`、`machine_no`、`resolution_x`、`resolution_y`、`client_bomb_info`、`aoi_machine_coords`
2. 驗證 `image_dir` 路徑存在且有圖片
3. 檢查該 record_id 是否已在重跑中（防止重複觸發）
4. 啟動 `threading.Thread` 執行背景推論
5. 立即回傳結果

**Response:**
- 成功：`{"status": "started", "record_id": 123}`
- 已在執行：`{"status": "already_running"}`
- 路徑不存在：`{"status": "error", "message": "image_dir not found: ..."}`
- 紀錄不存在：`{"status": "error", "message": "record not found"}`

### GET `/api/record/<id>/rerun/status` (SSE)

Server-Sent Events 串流，推送重跑進度。

**Event types:**
- `status`：進度更新，data 為 JSON `{"message": "推論中 3/8 張圖片..."}`
- `done`：推論完成，data 為 JSON `{"message": "完成", "record_id": 123}`
- `error`：推論失敗，data 為 JSON `{"message": "錯誤描述"}`

前端收到 `done` 後自動 `location.reload()`，收到 `error` 後顯示錯誤訊息。

若該 record_id 沒有正在執行的重跑任務，立即回傳 `status` event `{"message": "idle"}` 並關閉串流。

## Backend Implementation

### 狀態追蹤

在 `CAPIWeb` class 中新增：

```python
self._rerun_tasks: Dict[int, dict] = {}
# 每個 entry: {"status": "running"|"done"|"error", "message": str, "thread": Thread}
```

Thread-safe 存取透過一個 `threading.Lock`。

### 背景執行緒流程

1. 更新 `_rerun_tasks[record_id]` 狀態為 `running`
2. 取得 GPU lock（`self.inferencer._gpu_lock`）
3. 解析 `bomb_info`、`aoi_machine_coords` JSON
4. 呼叫 `self.inferencer.process_panel(panel_dir, product_resolution=..., bomb_info=..., model_id=...)`
5. 釋放 GPU lock
6. 刪除舊的 `image_results`、`tile_results`、`edge_defect_results`（同一 record_id）
7. 清空舊 heatmap 目錄
8. 寫入新的推論結果到 DB（複用現有的 DB 寫入邏輯）
9. 更新 `inference_records` 的 `ai_judgment`、`ng_images`、`ng_details`、`processing_seconds`、`response_time`、`inference_log` 等欄位
10. 更新 `_rerun_tasks[record_id]` 狀態為 `done`
11. 若任何步驟失敗，更新狀態為 `error` + 錯誤訊息

### GPU Lock

重新推論必須取得與正常推論相同的 GPU lock，確保不會與 TCP 請求的推論衝突。

## Frontend Implementation

### 按鈕位置

在 record detail 頁面 header 區域新增「重新推論」按鈕，兩個模板都加：
- `templates/record_detail.html`
- `templates/record_detail_v3.html`

### 互動流程

1. 使用者點擊「重新推論」按鈕
2. 彈出 `confirm()` 確認對話框（防止誤觸）
3. `fetch()` POST `/api/record/<id>/rerun`
4. 若回傳 `started`：按鈕變 disabled、顯示 spinner + 進度文字
5. 開啟 `EventSource` 監聯 `/api/record/<id>/rerun/status`
6. 收到 `status` event → 更新進度文字
7. 收到 `done` event → `location.reload()`
8. 收到 `error` event → 顯示錯誤訊息、按鈕恢復可用
9. 若回傳 `already_running`：提示「正在重跑中」並開啟 SSE 監聽

### 技術約束

- 純 JavaScript，不引入新框架
- `EventSource` 為瀏覽器原生 API，不需 polyfill

## DB Update Strategy

覆蓋式更新（同一 record_id）：

1. DELETE FROM `tile_results` WHERE `image_result_id` IN (SELECT id FROM `image_results` WHERE `record_id` = ?)
2. DELETE FROM `edge_defect_results` WHERE `image_result_id` IN (SELECT id FROM `image_results` WHERE `record_id` = ?)
3. DELETE FROM `image_results` WHERE `record_id` = ?
4. INSERT 新的 `image_results` 和 `tile_results`、`edge_defect_results`
5. UPDATE `inference_records` SET 各欄位 WHERE `id` = ?

以上步驟在同一個 DB transaction 中執行。

## Error Handling

- `image_dir` 不存在或無圖片 → 啟動前即回傳錯誤，不進入背景執行
- 推論過程中例外 → 捕獲並更新 `_rerun_tasks` 狀態為 error，不影響 server 穩定性
- GPU lock 等待中不設 timeout（與正常推論行為一致）
- SSE 連線中斷 → 前端可手動刷新頁面查看結果
