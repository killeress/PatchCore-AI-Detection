# Rerun Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "rerun inference" button to record detail pages that re-runs the full panel inference in the background, overwrites old results, and auto-refreshes the page via SSE.

**Architecture:** New POST API triggers a background thread that acquires the GPU lock, calls `process_panel()`, saves heatmaps, and overwrites DB records. A separate SSE endpoint streams progress to the frontend. Frontend uses `fetch()` + `EventSource` for interaction.

**Tech Stack:** Python threading, BaseHTTPRequestHandler SSE, vanilla JavaScript EventSource

---

### Task 1: Add DB method for rerun record update

**Files:**
- Modify: `capi_database.py`

- [ ] **Step 1: Add `update_record_for_rerun()` method to `CAPIDatabase`**

Add this method after the existing `save_inference_record()` method (around line 410):

```python
def update_record_for_rerun(
    self,
    record_id: int,
    ai_judgment: str,
    total_images: int,
    ng_images: int,
    ng_details: str,
    processing_seconds: float,
    heatmap_dir: str = "",
    error_message: str = "",
    image_results_data: Optional[List[Dict]] = None,
    inference_log: str = "",
    omit_overexposed: int = 0,
    omit_overexposure_info: str = "",
) -> None:
    """
    重新推論後覆蓋更新紀錄 (同一 record_id)

    1. 刪除舊的 tile_results, edge_defect_results, image_results
    2. 更新 inference_records 欄位
    3. 插入新的 image_results, tile_results, edge_defect_results
    """
    with self._lock:
        conn = self._get_conn()
        try:
            # --- 刪除舊的子紀錄 ---
            old_image_ids = [
                row["id"] for row in conn.execute(
                    "SELECT id FROM image_results WHERE record_id = ?",
                    (record_id,)
                ).fetchall()
            ]
            if old_image_ids:
                placeholders = ",".join("?" * len(old_image_ids))
                conn.execute(
                    f"DELETE FROM tile_results WHERE image_result_id IN ({placeholders})",
                    old_image_ids,
                )
                conn.execute(
                    f"DELETE FROM edge_defect_results WHERE image_result_id IN ({placeholders})",
                    old_image_ids,
                )
            conn.execute(
                "DELETE FROM image_results WHERE record_id = ?",
                (record_id,),
            )

            # --- 更新主紀錄 ---
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                """UPDATE inference_records SET
                       ai_judgment = ?,
                       total_images = ?,
                       ng_images = ?,
                       ng_details = ?,
                       response_time = ?,
                       processing_seconds = ?,
                       heatmap_dir = ?,
                       error_message = ?,
                       inference_log = ?,
                       omit_overexposed = ?,
                       omit_overexposure_info = ?
                   WHERE id = ?""",
                (ai_judgment, total_images, ng_images, ng_details,
                 now_str, processing_seconds, heatmap_dir, error_message,
                 inference_log, omit_overexposed, omit_overexposure_info,
                 record_id),
            )

            # --- 插入新的子紀錄 (複用 save_inference_record 的插入邏輯) ---
            if image_results_data:
                for img_data in image_results_data:
                    img_cursor = conn.execute(
                        """INSERT INTO image_results
                           (record_id, image_path, image_name, image_width, image_height,
                            otsu_bounds, tile_count, excluded_tiles, anomaly_count,
                            max_score, is_ng, is_dust_only, is_bomb, inference_time_ms,
                            heatmap_path)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (record_id,
                         img_data.get("image_path", ""),
                         img_data.get("image_name", ""),
                         img_data.get("image_width", 0),
                         img_data.get("image_height", 0),
                         img_data.get("otsu_bounds", ""),
                         img_data.get("tile_count", 0),
                         img_data.get("excluded_tiles", 0),
                         img_data.get("anomaly_count", 0),
                         img_data.get("max_score", 0.0),
                         img_data.get("is_ng", 0),
                         img_data.get("is_dust_only", 0),
                         img_data.get("is_bomb", 0),
                         img_data.get("inference_time_ms", 0.0),
                         img_data.get("heatmap_path", ""))
                    )
                    image_result_id = img_cursor.lastrowid

                    for tile_data in img_data.get("tiles", []):
                        conn.execute(
                            """INSERT INTO tile_results
                               (image_result_id, tile_id, x, y, width, height,
                                score, is_anomaly, is_dust, dust_iou, is_bomb,
                                bomb_code, peak_x, peak_y, heatmap_path,
                                is_exclude_zone, is_aoi_coord, aoi_defect_code,
                                aoi_product_x, aoi_product_y)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (image_result_id,
                             tile_data.get("tile_id", 0),
                             tile_data.get("x", 0),
                             tile_data.get("y", 0),
                             tile_data.get("width", 0),
                             tile_data.get("height", 0),
                             tile_data.get("score", 0.0),
                             tile_data.get("is_anomaly", 0),
                             tile_data.get("is_dust", 0),
                             tile_data.get("dust_iou", 0.0),
                             tile_data.get("is_bomb", 0),
                             tile_data.get("bomb_code", ""),
                             tile_data.get("peak_x", -1),
                             tile_data.get("peak_y", -1),
                             tile_data.get("heatmap_path", ""),
                             tile_data.get("is_exclude_zone", 0),
                             tile_data.get("is_aoi_coord", 0),
                             tile_data.get("aoi_defect_code", ""),
                             tile_data.get("aoi_product_x", -1),
                             tile_data.get("aoi_product_y", -1))
                        )

                    for edge_data in img_data.get("edge_defects", []):
                        conn.execute(
                            """INSERT INTO edge_defect_results
                               (image_result_id, side, area,
                                bbox_x, bbox_y, bbox_w, bbox_h,
                                max_diff, center_x, center_y, heatmap_path,
                                is_dust, is_bomb, bomb_code, is_cv_ok)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (image_result_id,
                             edge_data.get("side", ""),
                             edge_data.get("area", 0),
                             edge_data.get("bbox_x", 0),
                             edge_data.get("bbox_y", 0),
                             edge_data.get("bbox_w", 0),
                             edge_data.get("bbox_h", 0),
                             edge_data.get("max_diff", 0.0),
                             edge_data.get("center_x", 0),
                             edge_data.get("center_y", 0),
                             edge_data.get("heatmap_path", ""),
                             edge_data.get("is_dust", 0),
                             edge_data.get("is_bomb", 0),
                             edge_data.get("bomb_code", ""),
                             edge_data.get("is_cv_ok", 0))
                        )

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
```

- [ ] **Step 2: Commit**

```bash
git add capi_database.py
git commit -m "feat: add update_record_for_rerun() DB method for overwriting inference records"
```

---

### Task 2: Add rerun backend endpoints to capi_web.py

**Files:**
- Modify: `capi_web.py`

- [ ] **Step 1: Add rerun class variables and import**

At the top of `capi_web.py`, ensure `threading` is imported (likely already is). In the `create_web_server()` function (around line 2023), add after the existing class attribute assignments:

```python
CAPIWebHandler._rerun_tasks = {}      # {record_id: {"status": str, "message": str}}
CAPIWebHandler._rerun_lock = threading.Lock()
```

- [ ] **Step 2: Add route dispatch for rerun endpoints**

In `do_POST()` method (around line 199), add a new elif branch for the rerun trigger:

```python
elif path.startswith("/api/record/") and path.endswith("/rerun"):
    # Extract record_id from /api/record/<id>/rerun
    record_id_str = path.split("/api/record/")[1].split("/rerun")[0]
    self._handle_rerun_trigger(record_id_str)
```

In `do_GET()` method (around line 121), add a new elif branch for the SSE endpoint:

```python
elif path.startswith("/api/record/") and path.endswith("/rerun/status"):
    # Extract record_id from /api/record/<id>/rerun/status
    record_id_str = path.split("/api/record/")[1].split("/rerun/status")[0]
    self._handle_rerun_status_sse(record_id_str)
```

- [ ] **Step 3: Implement `_handle_rerun_trigger()` method**

Add this method to `CAPIWebHandler`:

```python
def _handle_rerun_trigger(self, record_id_str: str):
    """API: 觸發重新推論"""
    try:
        record_id = int(record_id_str)
    except ValueError:
        self._send_json({"status": "error", "message": "invalid record_id"})
        return

    if not self.inferencer:
        self._send_json({"status": "error", "message": "推論器未載入"})
        return

    # 檢查是否已在重跑中
    with self._rerun_lock:
        task = self._rerun_tasks.get(record_id)
        if task and task["status"] == "running":
            self._send_json({"status": "already_running"})
            return

    # 從 DB 讀取原始紀錄
    detail = self.db.get_record_detail(record_id) if self.db else None
    if not detail:
        self._send_json({"status": "error", "message": f"找不到紀錄 #{record_id}"})
        return

    image_dir = detail.get("image_dir", "")
    if not image_dir or not Path(image_dir).is_dir():
        self._send_json({"status": "error", "message": f"圖片目錄不存在: {image_dir}"})
        return

    # 初始化狀態
    with self._rerun_lock:
        self._rerun_tasks[record_id] = {"status": "running", "message": "正在準備推論..."}

    # 啟動背景執行緒
    thread = threading.Thread(
        target=CAPIWebHandler._rerun_worker,
        args=(record_id, detail),
        daemon=True,
    )
    thread.start()

    self._send_json({"status": "started", "record_id": record_id})
```

- [ ] **Step 4: Implement `_rerun_worker()` classmethod**

Add this classmethod to `CAPIWebHandler`. This runs in a background thread:

```python
@classmethod
def _rerun_worker(cls, record_id: int, detail: dict):
    """背景執行緒：重新推論並覆蓋紀錄"""
    import time as _time
    from capi_server import results_to_db_data, aggregate_judgment, append_cv_edge_to_judgment

    def _update_status(msg):
        with cls._rerun_lock:
            task = cls._rerun_tasks.get(record_id)
            if task:
                task["message"] = msg

    try:
        panel_dir = Path(detail["image_dir"])
        model_id = detail.get("model_id", "")
        resolution = None
        if detail.get("resolution_x") and detail.get("resolution_y"):
            resolution = (detail["resolution_x"], detail["resolution_y"])

        bomb_info = None
        if detail.get("client_bomb_info"):
            try:
                bomb_info = json.loads(detail["client_bomb_info"])
            except (json.JSONDecodeError, TypeError):
                pass

        # --- 推論 ---
        _update_status("正在等待 GPU...")
        start_time = _time.time()

        if cls._gpu_lock:
            with cls._gpu_lock:
                _update_status("正在推論中...")
                panel_result = cls.inferencer.process_panel(
                    panel_dir,
                    progress_callback=_update_status,
                    product_resolution=resolution,
                    bomb_info=bomb_info,
                    model_id=model_id,
                )
        else:
            _update_status("正在推論中...")
            panel_result = cls.inferencer.process_panel(
                panel_dir,
                progress_callback=_update_status,
                product_resolution=resolution,
                bomb_info=bomb_info,
                model_id=model_id,
            )

        processing_seconds = _time.time() - start_time

        results = panel_result[0]
        omit_overexposed = panel_result[2] if len(panel_result) > 2 else False
        omit_overexposure_info = panel_result[3] if len(panel_result) > 3 else ""
        omit_image_raw = panel_result[5] if len(panel_result) > 5 else None

        if not results:
            _update_status("推論完成但無結果")
            with cls._rerun_lock:
                cls._rerun_tasks[record_id] = {"status": "error", "message": "推論完成但無圖片結果"}
            return

        # --- 彙總判定 ---
        ai_judgment, ng_details = aggregate_judgment(results)
        for result in results:
            if hasattr(result, 'edge_defects') and result.edge_defects:
                ai_judgment, ng_details = append_cv_edge_to_judgment(
                    ai_judgment, ng_details, result.edge_defects, result.image_path.stem
                )

        # --- 儲存 heatmap ---
        _update_status("正在儲存 heatmap...")
        heatmap_info = {}
        if cls.heatmap_manager:
            # 清除舊 heatmap 目錄
            old_heatmap_dir = detail.get("heatmap_dir", "")
            if old_heatmap_dir and Path(old_heatmap_dir).is_dir():
                import shutil
                try:
                    shutil.rmtree(old_heatmap_dir)
                except Exception:
                    pass

            heatmap_info = cls.heatmap_manager.save_panel_heatmaps(
                glass_id=detail["glass_id"],
                results=results,
                inferencer=cls.inferencer,
                save_overview=True,
                save_tile_detail=True,
                omit_image=omit_image_raw,
            )

        # --- 轉換 DB 格式並更新 ---
        _update_status("正在更新資料庫...")
        image_results_data = results_to_db_data(results, heatmap_info) if results else []
        total_images = len(image_results_data)
        ng_images = sum(1 for d in image_results_data if d.get("is_ng"))
        error_message = ai_judgment if ai_judgment.startswith("ERR") else ""

        # 擷取推論日誌 (如果有 InferenceLogCapture)
        inference_log = ""
        if hasattr(cls.inferencer, '_log_capture') and cls.inferencer._log_capture:
            try:
                inference_log = cls.inferencer._log_capture.get_log()
            except Exception:
                pass

        cls.db.update_record_for_rerun(
            record_id=record_id,
            ai_judgment=ai_judgment,
            total_images=total_images,
            ng_images=ng_images,
            ng_details=ng_details,
            processing_seconds=processing_seconds,
            heatmap_dir=heatmap_info.get("dir", ""),
            error_message=error_message,
            image_results_data=image_results_data,
            inference_log=inference_log,
            omit_overexposed=int(omit_overexposed),
            omit_overexposure_info=omit_overexposure_info if omit_overexposure_info else "",
        )

        with cls._rerun_lock:
            cls._rerun_tasks[record_id] = {"status": "done", "message": "完成"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        with cls._rerun_lock:
            cls._rerun_tasks[record_id] = {"status": "error", "message": f"推論失敗: {e}"}
```

- [ ] **Step 5: Implement `_handle_rerun_status_sse()` method**

Add this method to `CAPIWebHandler`:

```python
def _handle_rerun_status_sse(self, record_id_str: str):
    """SSE: 串流重跑進度"""
    import time as _time

    try:
        record_id = int(record_id_str)
    except ValueError:
        self._send_json({"status": "error", "message": "invalid record_id"})
        return

    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream")
    self.send_header("Cache-Control", "no-cache")
    self.send_header("Connection", "keep-alive")
    self.send_header("X-Accel-Buffering", "no")
    self.end_headers()

    def sse_send(event_type, data):
        msg = f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        self.wfile.write(msg.encode("utf-8"))
        self.wfile.flush()

    last_msg = ""
    try:
        while True:
            with self._rerun_lock:
                task = self._rerun_tasks.get(record_id)

            if not task:
                sse_send("status", {"message": "idle"})
                break

            if task["message"] != last_msg:
                sse_send("status", {"message": task["message"]})
                last_msg = task["message"]

            if task["status"] == "done":
                sse_send("done", {"message": "完成", "record_id": record_id})
                # 清理已完成的任務
                with self._rerun_lock:
                    self._rerun_tasks.pop(record_id, None)
                break
            elif task["status"] == "error":
                sse_send("error", {"message": task["message"]})
                with self._rerun_lock:
                    self._rerun_tasks.pop(record_id, None)
                break

            _time.sleep(0.5)
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        pass
```

- [ ] **Step 6: Commit**

```bash
git add capi_web.py
git commit -m "feat: add rerun inference API endpoints with SSE progress streaming"
```

---

### Task 3: Add rerun button to record_detail.html

**Files:**
- Modify: `templates/record_detail.html`

- [ ] **Step 1: Add rerun button in header area**

In `templates/record_detail.html`, find the header `<h2>` line (around line 56–60):

```html
<h2 style="margin-bottom:16px">推論記錄 #{{ detail.id }}
    <span class="badge {{ detail.ai_judgment | ai_badge }}" style="font-size:1rem;margin-left:8px">{{
        detail.ai_judgment }}</span>
</h2>
```

Replace that `<h2>` block with:

```html
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
    <h2 style="margin:0">推論記錄 #{{ detail.id }}
        <span class="badge {{ detail.ai_judgment | ai_badge }}" style="font-size:1rem;margin-left:8px">{{
            detail.ai_judgment }}</span>
    </h2>
    <button id="rerunBtn" onclick="triggerRerun({{ detail.id }})"
        style="padding:8px 18px;background:#2563EB;color:#fff;border:none;border-radius:8px;font-size:0.9rem;font-weight:600;cursor:pointer;display:inline-flex;align-items:center;gap:6px;transition:all 0.2s;"
        onmouseover="this.style.background='#1D4ED8'" onmouseout="this.style.background='#2563EB'">
        <span id="rerunBtnIcon">&#x21bb;</span>
        <span id="rerunBtnText">重新推論</span>
    </button>
</div>
```

- [ ] **Step 2: Add JavaScript at the bottom of the template**

Before the closing `</body>` tag, add:

```html
<script>
function triggerRerun(recordId) {
    if (!confirm('確定要重新推論此片？舊的紀錄和圖片將被覆蓋。')) return;

    var btn = document.getElementById('rerunBtn');
    var icon = document.getElementById('rerunBtnIcon');
    var text = document.getElementById('rerunBtnText');
    btn.disabled = true;
    btn.style.background = '#64748B';
    btn.style.cursor = 'not-allowed';
    icon.innerHTML = '';
    text.textContent = '啟動中...';

    fetch('/api/record/' + recordId + '/rerun', { method: 'POST' })
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data.status === 'started' || data.status === 'already_running') {
                text.textContent = '推論中...';
                listenRerunSSE(recordId);
            } else {
                alert('啟動失敗: ' + (data.message || data.status));
                resetRerunBtn();
            }
        })
        .catch(function(err) {
            alert('請求失敗: ' + err);
            resetRerunBtn();
        });
}

function listenRerunSSE(recordId) {
    var text = document.getElementById('rerunBtnText');
    var es = new EventSource('/api/record/' + recordId + '/rerun/status');

    es.addEventListener('status', function(e) {
        var d = JSON.parse(e.data);
        if (d.message && d.message !== 'idle') {
            text.textContent = d.message;
        }
    });

    es.addEventListener('done', function(e) {
        es.close();
        text.textContent = '完成，重新載入...';
        location.reload();
    });

    es.addEventListener('error', function(e) {
        if (e.data) {
            var d = JSON.parse(e.data);
            alert('推論失敗: ' + d.message);
        }
        es.close();
        resetRerunBtn();
    });

    es.onerror = function() {
        es.close();
        text.textContent = '連線中斷，請手動刷新';
        setTimeout(resetRerunBtn, 3000);
    };
}

function resetRerunBtn() {
    var btn = document.getElementById('rerunBtn');
    var icon = document.getElementById('rerunBtnIcon');
    var text = document.getElementById('rerunBtnText');
    btn.disabled = false;
    btn.style.background = '#2563EB';
    btn.style.cursor = 'pointer';
    icon.innerHTML = '&#x21bb;';
    text.textContent = '重新推論';
}
</script>
```

- [ ] **Step 3: Commit**

```bash
git add templates/record_detail.html
git commit -m "feat: add rerun inference button to record detail page"
```

---

### Task 4: Add rerun button to record_detail_v3.html

**Files:**
- Modify: `templates/record_detail_v3.html`

- [ ] **Step 1: Add rerun button in the V3 header area**

In `templates/record_detail_v3.html`, find the page title area (around lines 387–411). Locate the outer `<div>` that has `display: flex; align-items: center; justify-content: space-between;`. Find the right-side `<div>` that currently shows `{{ detail.created_at }}`:

```html
<div style="color: #64748B; font-size: 13px; font-weight: 500;">
    {{ detail.created_at }}
</div>
```

Replace it with:

```html
<div style="display: flex; align-items: center; gap: 12px;">
    <span style="color: #64748B; font-size: 13px; font-weight: 500;">{{ detail.created_at }}</span>
    <button id="rerunBtn" onclick="triggerRerun({{ detail.id }})"
        style="padding:6px 16px;background:#2563EB;color:#fff;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;display:inline-flex;align-items:center;gap:5px;transition:all 0.2s;"
        onmouseover="this.style.background='#1D4ED8'" onmouseout="this.style.background='#2563EB'">
        <span id="rerunBtnIcon">&#x21bb;</span>
        <span id="rerunBtnText">重新推論</span>
    </button>
</div>
```

- [ ] **Step 2: Add JavaScript at the bottom of the V3 template**

Before the closing `</body>` tag, add the same JavaScript as Task 3 Step 2 (the `triggerRerun`, `listenRerunSSE`, `resetRerunBtn` functions). Copy the exact same `<script>` block.

- [ ] **Step 3: Commit**

```bash
git add templates/record_detail_v3.html
git commit -m "feat: add rerun inference button to V3 record detail page"
```

---

### Task 5: Manual integration test

- [ ] **Step 1: Start the server**

```bash
python capi_server.py --config server_config_local.yaml
```

- [ ] **Step 2: Open a record detail page**

Open `http://127.0.0.1:8080/record/<id>` (use an existing record with a valid `image_dir`).

- [ ] **Step 3: Verify the button**

1. Confirm the "重新推論" button appears in the header
2. Click the button → confirm dialog should appear
3. Click "確定" → button should show progress status
4. After inference completes → page should auto-reload with updated results

- [ ] **Step 4: Repeat for V3 page**

Open `http://127.0.0.1:8080/v3/record/<id>` and verify the same behavior.

- [ ] **Step 5: Test error case**

Test with a record whose `image_dir` no longer exists — should show an error alert immediately.
