# RIC Report 時間篩選 + 漏檢 Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add time-based filtering to RIC Report (matching AI inference tab style) and a miss-detection Review sub-tab with category/note tracking and pie chart statistics.

**Architecture:** Extend the existing `/api/ric/client-data` API to accept date filters and return computed summary statistics (moving computation from frontend to backend). Add a `miss_review` DB table and API for storing review data. Restructure the RIC Report frontend to load data from DB on page load instead of requiring file upload first.

**Tech Stack:** Python (SQLite, BaseHTTPRequestHandler), Jinja2 templates, vanilla JavaScript, Chart.js

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `capi_database.py` | Modify | Add `miss_review` table, extend `get_client_accuracy_records()` with date filter + LEFT JOIN, add `save_miss_review()` / `delete_miss_review()` |
| `capi_web.py` | Modify | Extend `_handle_client_data_api()` with date params + summary computation, add `_handle_miss_review_save()` / `_handle_miss_review_delete()`, add POST routes |
| `templates/ric_report.html` | Modify | Restructure HTML (date bar, collapsible upload, review tab), rewrite `clientComp` JS module |

---

### Task 1: DB — Add `miss_review` table and extend `get_client_accuracy_records()`

**Files:**
- Modify: `capi_database.py:201-214` (table init section)
- Modify: `capi_database.py:962-973` (`get_client_accuracy_records`)

- [ ] **Step 1: Add `miss_review` table creation in `_init_db()`**

In `capi_database.py`, after the `client_accuracy_records` table creation (after line 214), add:

```python
                -- Miss review records (漏檢原因回填)
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

- [ ] **Step 2: Extend `get_client_accuracy_records()` with date filter and LEFT JOIN**

Replace the entire `get_client_accuracy_records()` method (lines 962-973) with:

```python
    def get_client_accuracy_records(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> list:
        """取得 client accuracy records，支援日期篩選，並 LEFT JOIN miss_review"""
        if start_date and not _DATE_RE.match(start_date):
            raise ValueError(f"Invalid start_date format: {start_date}")
        if end_date and not _DATE_RE.match(end_date):
            raise ValueError(f"Invalid end_date format: {end_date}")

        conn = self._get_conn()
        try:
            where_clauses = []
            params = []
            if start_date:
                where_clauses.append("c.time_stamp >= ?")
                params.append(start_date)
            if end_date:
                where_clauses.append("c.time_stamp <= ?")
                params.append(end_date + " 23:59:59")
            where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

            rows = conn.execute(
                f"""SELECT c.id, c.time_stamp, c.pnl_id, c.mach_id,
                           c.result_eqp, c.result_ai, c.result_ric, c.datastr,
                           mr.id as review_id, mr.category as review_category,
                           mr.note as review_note, mr.updated_at as review_updated_at
                    FROM client_accuracy_records c
                    LEFT JOIN miss_review mr ON mr.client_record_id = c.id
                    {where_sql}
                    ORDER BY c.time_stamp DESC""",
                params
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
```

- [ ] **Step 3: Add `save_miss_review()` method**

Add after `get_client_accuracy_records()`:

```python
    VALID_MISS_CATEGORIES = {'dust_misfilter', 'threshold_high', 'ric_misjudge', 'outside_aoi_area', 'other'}

    def save_miss_review(self, client_record_id: int, category: str, note: str = '') -> int:
        """儲存或更新漏檢 Review (UPSERT by client_record_id)"""
        if category not in self.VALID_MISS_CATEGORIES:
            raise ValueError(f"Invalid category: {category}")

        with self._lock:
            conn = self._get_conn()
            try:
                # 驗證 client_record_id 存在
                row = conn.execute(
                    "SELECT id FROM client_accuracy_records WHERE id = ?",
                    (client_record_id,)
                ).fetchone()
                if not row:
                    raise ValueError(f"Record not found: {client_record_id}")

                conn.execute(
                    """INSERT INTO miss_review (client_record_id, category, note)
                       VALUES (?, ?, ?)
                       ON CONFLICT(client_record_id)
                       DO UPDATE SET category = excluded.category,
                                     note = excluded.note,
                                     updated_at = datetime('now', 'localtime')""",
                    (client_record_id, category, note)
                )
                conn.commit()
                review_id = conn.execute(
                    "SELECT id FROM miss_review WHERE client_record_id = ?",
                    (client_record_id,)
                ).fetchone()["id"]
                return review_id
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def delete_miss_review(self, client_record_id: int) -> bool:
        """刪除漏檢 Review"""
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    "DELETE FROM miss_review WHERE client_record_id = ?",
                    (client_record_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()
```

- [ ] **Step 4: Verify DB init works**

Run: `python -c "from capi_database import CAPIDatabase; db = CAPIDatabase('test_review.db'); print('OK')"`
Expected: `OK` (no errors, table created)

Then clean up: `rm test_review.db`

- [ ] **Step 5: Commit**

```bash
git add capi_database.py
git commit -m "feat: add miss_review table and extend client records with date filter"
```

---

### Task 2: Backend — Extend API with summary computation and review endpoints

**Files:**
- Modify: `capi_web.py:153-159` (GET routing)
- Modify: `capi_web.py:218-221` (POST routing)
- Modify: `capi_web.py:1117-1128` (`_handle_client_data_api`)

- [ ] **Step 1: Rewrite `_handle_client_data_api()` with date params and summary computation**

Replace the `_handle_client_data_api` method (lines 1117-1128) with:

```python
    def _handle_client_data_api(self, query: dict):
        """API: 取得已儲存的 client accuracy records（支援日期篩選 + summary 統計）"""
        try:
            start_date = query.get('start_date', [''])[0] or None
            end_date = query.get('end_date', [''])[0] or None
            records = self.db.get_client_accuracy_records(start_date, end_date)

            # 計算 summary 統計
            summary = self._compute_client_summary(records)

            # 整理 records 回傳格式
            out_records = []
            for r in records:
                rec = {
                    "id": r["id"],
                    "time_stamp": r["time_stamp"],
                    "pnl_id": r["pnl_id"],
                    "mach_id": r["mach_id"],
                    "result_eqp": r["result_eqp"],
                    "result_ai": r["result_ai"],
                    "result_ric": r["result_ric"],
                    "datastr": r["datastr"] or "",
                    "miss_review": None,
                }
                if r.get("review_id"):
                    rec["miss_review"] = {
                        "id": r["review_id"],
                        "category": r["review_category"],
                        "note": r["review_note"] or "",
                        "updated_at": r["review_updated_at"],
                    }
                out_records.append(rec)

            self._send_json({
                "success": True,
                "total": len(records),
                "summary": summary,
                "records": out_records,
            })
        except ValueError as ve:
            self._send_json({"success": False, "error": str(ve)})
        except Exception as e:
            logger.error(f"Client data API error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})
```

- [ ] **Step 2: Add `_compute_client_summary()` helper method**

Add right after `_handle_client_data_api`:

```python
    def _compute_client_summary(self, records: list) -> dict:
        """從 client accuracy records 計算統計摘要（搬自前端 computeStats）"""
        total = len(records)
        if total == 0:
            return {
                "total": 0, "aoiNG": 0, "aiNG": 0, "ricNG": 0,
                "aoiCorrect": 0, "aiCorrect": 0,
                "aoiOver": 0, "aoiOverRate": 0,
                "aiOver": 0, "aiOverRate": 0,
                "aiMiss": 0, "aiMissRate": 0,
                "revival": 0, "revivalRate": 0,
                "combos": {}, "daily": {},
                "missReviewStats": {
                    "total": 0, "reviewed": 0, "unreviewed": 0,
                    "byCategory": {
                        "dust_misfilter": 0, "threshold_high": 0,
                        "ric_misjudge": 0, "outside_aoi_area": 0, "other": 0,
                    },
                },
            }

        aoiNG = aiNG = ricNG = 0
        aoiCorrect = aiCorrect = 0
        aoiOver = aiOver = aiMiss = 0
        revival = 0
        combos = {}
        daily = {}
        miss_total = 0
        miss_reviewed = 0
        miss_by_cat = {
            "dust_misfilter": 0, "threshold_high": 0,
            "ric_misjudge": 0, "outside_aoi_area": 0, "other": 0,
        }

        def _derive_judgment(datastr):
            return "NG" if datastr and "NG" in datastr else "OK"

        for rec in records:
            eqp = rec["result_eqp"] or "OK"
            ai = rec["result_ai"] or "OK"
            ric = _derive_judgment(rec.get("datastr", ""))

            if eqp == "NG":
                aoiNG += 1
            if ai == "NG":
                aiNG += 1
            if ric == "NG":
                ricNG += 1

            if eqp == ric:
                aoiCorrect += 1
            if ai == ric:
                aiCorrect += 1

            if eqp == "NG" and ric == "OK":
                aoiOver += 1
            if ai == "NG" and ric == "OK":
                aiOver += 1
            if ai == "OK" and ric == "NG":
                aiMiss += 1
                miss_total += 1
                if rec.get("review_category"):
                    miss_reviewed += 1
                    cat = rec["review_category"]
                    if cat in miss_by_cat:
                        miss_by_cat[cat] += 1
            if eqp == "NG" and ai == "OK" and ric == "OK":
                revival += 1

            combo = f"{eqp}/{ai}/{ric}"
            combos[combo] = combos.get(combo, 0) + 1

            day = (rec.get("time_stamp") or "unknown")[:10]
            if day not in daily:
                daily[day] = {"total": 0, "aoiCorrect": 0, "aiCorrect": 0, "aiMiss": 0, "aiOver": 0, "aoiOver": 0}
            daily[day]["total"] += 1
            if eqp == ric:
                daily[day]["aoiCorrect"] += 1
            if ai == ric:
                daily[day]["aiCorrect"] += 1
            if ai == "OK" and ric == "NG":
                daily[day]["aiMiss"] += 1
            if ai == "NG" and ric == "OK":
                daily[day]["aiOver"] += 1
            if eqp == "NG" and ric == "OK":
                daily[day]["aoiOver"] += 1

        aoiOverRate = round(aoiOver / aoiNG * 100, 1) if aoiNG > 0 else 0
        aiOverRate = round(aiOver / aiNG * 100, 1) if aiNG > 0 else 0
        aiMissRate = round(aiMiss / ricNG * 100, 1) if ricNG > 0 else 0
        revivalRate = round(revival / aoiNG * 100, 1) if aoiNG > 0 else 0

        return {
            "total": total,
            "aoiNG": aoiNG, "aiNG": aiNG, "ricNG": ricNG,
            "aoiCorrect": aoiCorrect, "aiCorrect": aiCorrect,
            "aoiOver": aoiOver, "aoiOverRate": aoiOverRate,
            "aiOver": aiOver, "aiOverRate": aiOverRate,
            "aiMiss": aiMiss, "aiMissRate": aiMissRate,
            "revival": revival, "revivalRate": revivalRate,
            "combos": combos, "daily": daily,
            "missReviewStats": {
                "total": miss_total,
                "reviewed": miss_reviewed,
                "unreviewed": miss_total - miss_reviewed,
                "byCategory": miss_by_cat,
            },
        }
```

- [ ] **Step 3: Add miss-review POST/DELETE handlers**

Add after `_handle_client_clear()`:

```python
    def _handle_miss_review_save(self):
        """API: 儲存/更新漏檢 Review"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            client_record_id = data.get("client_record_id")
            category = data.get("category", "")
            note = data.get("note", "")

            if not client_record_id:
                self._send_json({"success": False, "error": "缺少 client_record_id"})
                return

            review_id = self.db.save_miss_review(int(client_record_id), category, note)
            self._send_json({"success": True, "id": review_id, "message": "Review 已儲存"})
        except ValueError as ve:
            self._send_json({"success": False, "error": str(ve)})
        except Exception as e:
            logger.error(f"Miss review save error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_miss_review_delete(self):
        """API: 刪除漏檢 Review"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            client_record_id = data.get("client_record_id")
            if not client_record_id:
                self._send_json({"success": False, "error": "缺少 client_record_id"})
                return

            deleted = self.db.delete_miss_review(int(client_record_id))
            if deleted:
                self._send_json({"success": True, "message": "Review 已刪除"})
            else:
                self._send_json({"success": False, "error": "Review 不存在"})
        except Exception as e:
            logger.error(f"Miss review delete error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})
```

- [ ] **Step 4: Add POST routes**

In `do_POST()` (around line 220), after the `elif path == "/api/ric/clear-client":` block, add:

```python
            elif path == "/api/ric/miss-review":
                self._handle_miss_review_save()
            elif path == "/api/ric/miss-review/delete":
                self._handle_miss_review_delete()
```

- [ ] **Step 5: Remove `client_record_count` from `_handle_ric_page()`**

The template will no longer need `client_record_count` since it auto-loads from DB. Change `_handle_ric_page` (line 886-892) to:

```python
    def _handle_ric_page(self, query: dict, path: str):
        """人工檢驗 (RIC) 比對報表頁面"""
        batches = self.db.get_ric_batches() if self.db else []
        template = self.jinja_env.get_template("ric_report.html")
        html = template.render(request_path=path, batches=batches)
        self._send_response(200, html)
```

- [ ] **Step 6: Commit**

```bash
git add capi_web.py
git commit -m "feat: extend client-data API with date filter, summary stats, and miss-review endpoints"
```

---

### Task 3: Frontend HTML — Restructure RIC Report layout

**Files:**
- Modify: `templates/ric_report.html` (HTML structure, lines 608-806)

This task restructures the HTML only. The JavaScript changes come in Task 4 and 5.

- [ ] **Step 1: Replace RIC Report tab HTML structure**

Replace the entire `<!-- Tab 1: RIC Report -->` section (from line 616 `<div class="top-tab-content active" id="tab-ric">` to line 747 `</div><!-- end #tab-ric -->`) with:

```html
<!-- Tab 1: RIC Report (existing content) -->
<div class="top-tab-content active" id="tab-ric">
<div class="ric-header">
    <div>
        <h2>📊 RIC Report</h2>
        <div class="subtitle">AI / AOI / RIC 正確率與過漏檢統計分析</div>
    </div>
    <button id="btn_toggleUpload" onclick="clientComp.toggleUpload()" style="margin-left:auto; background:linear-gradient(135deg,#6366f1,#818cf8); border:none; color:white; padding:10px 20px; border-radius:10px; font-size:0.9rem; font-weight:600; cursor:pointer; transition:all 0.2s;">📤 匯入資料</button>
</div>

<!-- Collapsible Upload Zone -->
<div id="client_uploadPanel" style="display:none; margin-bottom:20px;">
    <div class="card">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px;">
            <span style="font-size:1.2rem;">📤</span>
            <h3 style="margin:0; font-size:1rem; color:var(--text);">載入檢測資料</h3>
            <span style="color:var(--text-muted); font-size:0.8rem; margin-left:auto;">支援 .xls (HTML格式)</span>
            <button onclick="clientComp.clearDB()" style="margin-left:8px; background:rgba(239,68,68,0.15); color:var(--ng); border:1px solid rgba(239,68,68,0.3); border-radius:6px; padding:4px 12px; font-size:0.78rem; cursor:pointer;">🗑 清除資料庫</button>
        </div>
        <div class="upload-zone" id="client_uploadZone" onclick="document.getElementById('client_fileInput').click()">
            <div class="icon">📁</div>
            <div style="color:var(--text); font-weight:600; font-size:1.05rem;">拖放 XLS 檔案至此處</div>
            <div class="hint">或點擊選擇檔案上傳（新資料將自動存入資料庫，重複資料自動略過）</div>
            <input type="file" id="client_fileInput" accept=".xls,.xlsx,.html,.htm">
        </div>
        <div class="upload-msg" id="client_uploadMsg"></div>
    </div>
</div>

<!-- Date Filter Bar -->
<div class="inf-date-bar" id="client_dateBar">
    <span style="color:var(--text-muted); font-size:0.88rem; margin-right:4px;">日期篩選：</span>
    <button class="quick-btn active" data-range="all" onclick="clientComp.quickFilter('all')">全部</button>
    <button class="quick-btn" data-range="today" onclick="clientComp.quickFilter('today')">今日</button>
    <button class="quick-btn" data-range="7d" onclick="clientComp.quickFilter('7d')">7 天</button>
    <button class="quick-btn" data-range="30d" onclick="clientComp.quickFilter('30d')">30 天</button>
    <span style="margin-left:12px;"></span>
    <input type="date" id="client_startDate">
    <span style="color:var(--text-muted);">~</span>
    <input type="date" id="client_endDate">
    <button class="search-btn" onclick="clientComp.customFilter()">查詢</button>
</div>

<!-- Loading -->
<div class="inf-loading" id="client_loading">載入中...</div>

<!-- Empty State -->
<div id="client_emptyState" style="display:none; text-align:center; padding:60px 20px; color:var(--text-muted);">
    <div style="font-size:2rem; margin-bottom:12px;">📭</div>
    <div style="font-size:1rem;">無資料，請先匯入檢測資料或調整日期範圍</div>
</div>

<!-- Report Content -->
<div id="client_reportContent" style="display:none;">
    <!-- Sub-tabs -->
    <div class="client-tabs">
        <div class="client-tab active" data-client-tab="analysis" onclick="clientComp.switchTab('analysis')">📈 統計分析</div>
        <div class="client-tab" data-client-tab="detail" onclick="clientComp.switchTab('detail')">📋 逐筆明細</div>
        <div class="client-tab" data-client-tab="review" onclick="clientComp.switchTab('review')">🔍 漏檢 Review</div>
    </div>

    <!-- Sub-Tab 1: Analysis (unchanged) -->
    <div class="client-tab-panel active" id="client_panel-analysis">
        <div class="no-ref-banner">
            <span class="icon">⚠️</span>
            <div>
                <strong>RESULT_EQP=OK 無參考價值：</strong>
                RESULT_EQP=OK 表示外觀檢 (AOI) 判成 NG 後，由人員 (RIC) 目視複檢外觀覆判為 OK（非看 AI 畫面）。
                因此 <strong>AOI=OK 的資料不代表 AOI 真正判 OK</strong>，AOI 漏檢率無參考價值。
                但 <strong>AOI=NG 的 case 有參考價值</strong>，AOI 過檢率仍可做為參考指標。
            </div>
        </div>
        <div class="stats-grid" id="client_statsGrid1"></div>
        <div class="stats-grid" id="client_statsGrid2"></div>
        <div class="charts-row">
            <div class="chart-card">
                <h3>📊 判定結果分布</h3>
                <div class="chart-container"><canvas id="client_resultChart"></canvas></div>
            </div>
            <div class="chart-card">
                <h3>🧩 判定組合分佈 (AOI/AI/RIC)</h3>
                <div class="chart-container"><canvas id="client_comboChart"></canvas></div>
            </div>
        </div>
        <div class="charts-row">
            <div class="chart-card">
                <h3>📈 時間軸趨勢 (每日)</h3>
                <div class="chart-container"><canvas id="client_trendChart"></canvas></div>
            </div>
            <div class="chart-card">
                <h3>🧮 混淆矩陣 (AOI/AI vs 實際)</h3>
                <div class="chart-container"><canvas id="client_confusionChart"></canvas></div>
            </div>
        </div>
        <div class="charts-row">
            <div class="chart-card" style="grid-column: 1 / -1;">
                <h3>📅 每日過漏檢率追蹤 (分母為當日總數量)</h3>
                <div class="chart-container"><canvas id="client_dailyMetricsChart"></canvas></div>
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 8px;">
                    📌 備註：此圖表之漏檢與過檢率分母皆為「當日影像總數量」，以此呈現出整體佔比走勢。
                </div>
            </div>
        </div>
    </div>

    <!-- Sub-Tab 2: Detail (unchanged) -->
    <div class="client-tab-panel" id="client_panel-detail">
        <div class="card">
            <div class="detail-header">
                <h3>📋 逐筆明細 <span id="client_filterLabel" style="font-size:0.8rem; color:var(--accent); font-weight:400;"></span></h3>
                <button class="btn-export" onclick="clientComp.exportCSV()">📥 導出 CSV</button>
            </div>
            <div class="table-wrap" id="client_tableWrap">
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>檢測時間</th>
                            <th>Panel ID</th>
                            <th>機台</th>
                            <th>AOI (RESULT_EQP)</th>
                            <th>AI (RESULT_AI)</th>
                            <th>RIC (原始)</th>
                            <th>DATASTR 判定</th>
                            <th>DATASTR 內容</th>
                            <th>AOI vs DATASTR <span style="color:#f59e0b;font-size:0.65rem;">⚠僅參考NG</span></th>
                            <th>AI vs DATASTR</th>
                        </tr>
                    </thead>
                    <tbody id="client_detailBody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Sub-Tab 3: Miss Review -->
    <div class="client-tab-panel" id="client_panel-review">
        <div class="stats-grid" id="review_statsGrid"></div>
        <div class="charts-row">
            <div class="chart-card">
                <h3>🔍 漏檢原因分布</h3>
                <div class="chart-container"><canvas id="review_pieChart"></canvas></div>
            </div>
        </div>
        <div class="card">
            <div class="detail-header">
                <h3>🔍 漏檢案例 Review <span id="review_countLabel" style="font-size:0.8rem; color:var(--accent); font-weight:400;"></span></h3>
            </div>
            <div class="table-wrap" id="review_tableWrap">
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>檢測時間</th>
                            <th>Panel ID</th>
                            <th>機台</th>
                            <th>AOI</th>
                            <th>AI</th>
                            <th>RIC</th>
                            <th style="min-width:160px;">分類</th>
                            <th style="min-width:200px;">備註</th>
                            <th style="min-width:100px;">操作</th>
                        </tr>
                    </thead>
                    <tbody id="review_detailBody"></tbody>
                </table>
            </div>
        </div>
    </div>

</div>
</div><!-- end #tab-ric -->
```

- [ ] **Step 2: Add CSS for review tab**

In the `{% block extra_css %}` `<style>` section (before `</style>`), add:

```css
    /* ── Review Tab ── */
    .review-select {
        background: var(--surface);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 0.82rem;
        width: 100%;
    }

    .review-input {
        background: var(--surface);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 0.82rem;
        width: 100%;
    }

    .review-btn-save {
        background: linear-gradient(135deg, var(--accent), #6366f1);
        border: none;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        cursor: pointer;
    }

    .review-btn-save:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .review-saved {
        color: var(--ok);
        font-size: 0.8rem;
        font-weight: 600;
    }

    .review-btn-delete {
        color: var(--ng);
        font-size: 0.72rem;
        cursor: pointer;
        background: none;
        border: none;
        text-decoration: underline;
        margin-left: 8px;
    }
```

- [ ] **Step 3: Commit**

```bash
git add templates/ric_report.html
git commit -m "feat: restructure RIC Report HTML with date filter bar, collapsible upload, and review tab"
```

---

### Task 4: Frontend JS — Rewrite `clientComp` module with API-driven data loading

**Files:**
- Modify: `templates/ric_report.html` (JavaScript section, `clientComp` module)

- [ ] **Step 1: Replace the entire `clientComp` IIFE**

Replace from `const clientComp = (function () {` to the closing `})();` (lines 814-1629) with:

```javascript
    const clientComp = (function () {
        let rawRecords = [];     // API 回傳的 records
        let currentSummary = {}; // API 回傳的 summary
        let currentFilter = null;
        let resultChart = null, comboChart = null, trendChart = null, confusionChart = null, dailyMetricsChart = null;
        let reviewPieChart = null;

        // ── Upload Logic ──
        function initUpload() {
            const uploadZone = document.getElementById('client_uploadZone');
            const fileInput = document.getElementById('client_fileInput');
            if (!uploadZone || !fileInput) return;

            ['dragenter', 'dragover'].forEach(ev => {
                uploadZone.addEventListener(ev, e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
            });
            ['dragleave', 'drop'].forEach(ev => {
                uploadZone.addEventListener(ev, e => { e.preventDefault(); uploadZone.classList.remove('drag-over'); });
            });
            uploadZone.addEventListener('drop', e => {
                const files = e.dataTransfer.files;
                if (files.length > 0) uploadFile(files[0]);
            });
            fileInput.addEventListener('change', () => {
                if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
            });
        }

        function toggleUpload() {
            const panel = document.getElementById('client_uploadPanel');
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        }

        function uploadFile(file) {
            const msgEl = document.getElementById('client_uploadMsg');
            const reader = new FileReader();
            reader.onload = function (e) {
                try {
                    const content = e.target.result;
                    const parsed = parseHTMLTable(content);
                    if (parsed.length === 0) throw new Error('無法解析任何資料列');

                    msgEl.className = 'upload-msg success';
                    msgEl.innerHTML = `⏳ 解析 ${parsed.length} 筆資料，正在存入資料庫...`;
                    msgEl.style.display = 'block';

                    saveToDB(parsed).then(result => {
                        msgEl.innerHTML = `✅ 新增 ${result.inserted} 筆，略過 ${result.skipped} 筆重複`;
                        // 上傳成功後重新查詢
                        setTimeout(() => {
                            document.getElementById('client_uploadPanel').style.display = 'none';
                            _reloadCurrentFilter();
                        }, 1000);
                    }).catch(err => {
                        msgEl.className = 'upload-msg error';
                        msgEl.textContent = '❌ 儲存失敗: ' + err.message;
                    });
                } catch (err) {
                    msgEl.className = 'upload-msg error';
                    msgEl.textContent = '❌ 解析失敗: ' + err.message;
                    msgEl.style.display = 'block';
                }
            };
            reader.readAsText(file, 'GB2312');
        }

        function deriveJudgment(datastr) {
            return datastr && datastr.includes('NG') ? 'NG' : 'OK';
        }

        function saveToDB(data) {
            const records = data.map(row => ({
                time_stamp: row.systime,
                pnl_id: row.pnl_id,
                mach_id: row.mach_id,
                result_eqp: row.result_eqp,
                result_ai: row.result_ai,
                result_ric: row.result_ric_raw,
                datastr: row.datastr,
            }));

            return fetch('/api/ric/import-client', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ records })
            }).then(r => r.json()).then(data => {
                if (!data.success) throw new Error(data.error);
                return data;
            });
        }

        function clearDB() {
            if (!confirm('確定要清除資料庫中所有檢測紀錄嗎？\n此操作無法復原！')) return;
            fetch('/api/ric/clear-client', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        alert(`已清除 ${data.deleted} 筆資料`);
                        location.reload();
                    } else {
                        alert('❌ ' + (data.error || '清除失敗'));
                    }
                })
                .catch(err => alert('❌ 網路錯誤: ' + err.message));
        }

        function parseHTMLTable(html) {
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            const rows = doc.querySelectorAll('tr');
            const data = [];

            const headerCells = rows[0].querySelectorAll('td, th');
            let indices = {
                systime: 0, pnl_id: 1, mach_id: 2, result_eqp: 3, result_ai: 4, result_ric: 5, datastr: -1
            };

            headerCells.forEach((cell, idx) => {
                const text = cell.textContent.trim().toUpperCase();
                if (text === 'SYSTIME') indices.systime = idx;
                else if (text === 'PNL_ID') indices.pnl_id = idx;
                else if (text === 'MACH_ID') indices.mach_id = idx;
                else if (text === 'RESULT_EQP') indices.result_eqp = idx;
                else if (text === 'RESULT_AI') indices.result_ai = idx;
                else if (text === 'RESULT_RIC') indices.result_ric = idx;
                else if (text === 'DATASTR') indices.datastr = idx;
            });

            for (let i = 1; i < rows.length; i++) {
                const cells = rows[i].querySelectorAll('td');
                if (cells.length >= 6) {
                    let datastr = indices.datastr !== -1 && cells[indices.datastr] ? cells[indices.datastr].textContent.trim() : '';
                    if (datastr.toUpperCase().includes('CCD')) continue;

                    let actualResult = deriveJudgment(datastr);
                    let ricRaw = cells[indices.result_ric] ? cells[indices.result_ric].textContent.trim() : '';

                    data.push({
                        systime: cells[indices.systime] ? cells[indices.systime].textContent.trim() : '',
                        pnl_id: cells[indices.pnl_id] ? cells[indices.pnl_id].textContent.trim() : '',
                        mach_id: cells[indices.mach_id] ? cells[indices.mach_id].textContent.trim() : '',
                        result_eqp: cells[indices.result_eqp] ? cells[indices.result_eqp].textContent.trim() : '',
                        result_ai: cells[indices.result_ai] ? cells[indices.result_ai].textContent.trim() : '',
                        result_ric: actualResult,
                        result_ric_raw: ricRaw,
                        datastr: datastr,
                    });
                }
            }
            return data;
        }

        // ── Date Filter ──
        function _formatDate(d) {
            return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
        }

        function _getDateRange(range) {
            const today = new Date();
            const end = _formatDate(today);
            if (range === 'all') return { start: '', end: '' };
            if (range === 'today') return { start: end, end: end };
            const d = new Date(today);
            d.setDate(d.getDate() - (range === '7d' ? 6 : 29));
            return { start: _formatDate(d), end: end };
        }

        function quickFilter(range) {
            document.querySelectorAll('#client_dateBar .quick-btn').forEach(b => b.classList.toggle('active', b.dataset.range === range));
            const { start, end } = _getDateRange(range);
            document.getElementById('client_startDate').value = start;
            document.getElementById('client_endDate').value = end;
            loadData(start, end);
        }

        function customFilter() {
            document.querySelectorAll('#client_dateBar .quick-btn').forEach(b => b.classList.remove('active'));
            const start = document.getElementById('client_startDate').value;
            const end = document.getElementById('client_endDate').value;
            loadData(start, end);
        }

        function _reloadCurrentFilter() {
            const start = document.getElementById('client_startDate').value;
            const end = document.getElementById('client_endDate').value;
            loadData(start, end);
        }

        function loadData(startDate, endDate) {
            const loading = document.getElementById('client_loading');
            const empty = document.getElementById('client_emptyState');
            const report = document.getElementById('client_reportContent');
            loading.style.display = 'block';
            empty.style.display = 'none';
            report.style.display = 'none';

            const qp = new URLSearchParams();
            if (startDate) qp.set('start_date', startDate);
            if (endDate) qp.set('end_date', endDate);

            fetch('/api/ric/client-data?' + qp.toString())
                .then(r => r.json())
                .then(data => {
                    loading.style.display = 'none';
                    if (!data.success) {
                        empty.style.display = 'block';
                        empty.querySelector('div:last-child').textContent = '載入失敗: ' + (data.error || '');
                        return;
                    }
                    if (data.total === 0) {
                        empty.style.display = 'block';
                        return;
                    }

                    rawRecords = data.records;
                    currentSummary = data.summary;
                    report.style.display = 'block';

                    renderStatsCards(currentSummary);
                    renderCharts(currentSummary);
                    renderDetails(rawRecords);
                    renderReviewTab(currentSummary, rawRecords);
                })
                .catch(err => {
                    loading.style.display = 'none';
                    empty.style.display = 'block';
                    empty.querySelector('div:last-child').textContent = '載入失敗: ' + err.message;
                });
        }

        // ── Sub-tab switching ──
        function switchTab(tabId) {
            document.querySelectorAll('.client-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.client-tab-panel').forEach(p => p.classList.remove('active'));
            document.querySelector(`.client-tab[data-client-tab="${tabId}"]`).classList.add('active');
            document.getElementById(`client_panel-${tabId}`).classList.add('active');
        }

        // ── Render Stats Cards ──
        function renderStatsCards(s) {
            const grid1 = document.getElementById('client_statsGrid1');
            grid1.innerHTML = `
            <div class="stat-card total" data-client-filter="total" onclick="clientComp.filterByCard('total')">
                <div class="label">檢測總數</div>
                <div class="value">${s.total}</div>
                <div class="detail">${s.total} 筆資料 (已排除 CCD)</div>
                <div class="filter-hint">🔍 顯示全部</div>
            </div>
            <div class="stat-card aoi" data-client-filter="aoi_correct" onclick="clientComp.filterByCard('aoi_correct')">
                <div class="label">AOI 準確率</div>
                <div class="value">${s.total > 0 ? (s.aoiCorrect / s.total * 100).toFixed(1) : 0}%</div>
                <div class="detail">${s.aoiCorrect} / ${s.total} 筆與 DATASTR 判定一致</div>
                <div class="filter-hint">🔍 篩選 AOI 與 DATASTR 一致</div>
            </div>
            <div class="stat-card ai" data-client-filter="ai_correct" onclick="clientComp.filterByCard('ai_correct')">
                <div class="label">AI 準確率</div>
                <div class="value">${s.total > 0 ? (s.aiCorrect / s.total * 100).toFixed(1) : 0}%</div>
                <div class="detail">${s.aiCorrect} / ${s.total} 筆與 DATASTR 判定一致</div>
                <div class="filter-hint">🔍 篩選 AI 與 DATASTR 一致</div>
            </div>
            <div class="stat-card match" data-client-filter="ai_ng" onclick="clientComp.filterByCard('ai_ng')">
                <div class="label">AI 判 NG 數</div>
                <div class="value">${s.aiNG}</div>
                <div class="detail">AOI 判 NG: ${s.aoiNG} 筆</div>
                <div class="filter-hint">🔍 篩選 AI 判 NG</div>
            </div>
            `;

            const grid2 = document.getElementById('client_statsGrid2');
            grid2.innerHTML = `
            <div class="stat-card over" data-client-filter="aoi_over" onclick="clientComp.filterByCard('aoi_over')">
                <div class="label">AOI 過檢率</div>
                <div class="value">${s.aoiOverRate}%</div>
                <div class="detail">${s.aoiOver} / ${s.aoiNG} 筆 AOI判NG中為過檢</div>
                <div class="filter-hint">🔍 AOI=NG, RIC=OK</div>
            </div>
            <div class="stat-card over" data-client-filter="ai_over" onclick="clientComp.filterByCard('ai_over')">
                <div class="label">AI 過檢率</div>
                <div class="value">${s.aiOverRate}%</div>
                <div class="detail">${s.aiOver} / ${s.aiNG} 筆 AI判NG中為過檢</div>
                <div class="filter-hint">🔍 AI=NG, RIC=OK</div>
            </div>
            <div class="stat-card miss" data-client-filter="ai_miss" onclick="clientComp.filterByCard('ai_miss')">
                <div class="label">AI 漏檢率</div>
                <div class="value">${s.aiMissRate}%</div>
                <div class="detail">${s.aiMiss} / ${s.ricNG} 筆真實NG被AI漏檢</div>
                <div class="filter-hint">🔍 AI=OK, RIC=NG</div>
            </div>
            <div class="stat-card revival" data-client-filter="revival" onclick="clientComp.filterByCard('revival')">
                <div class="label">復活率</div>
                <div class="value">${s.revivalRate}%</div>
                <div class="detail">${s.revival} / ${s.aoiNG} 筆 AOI判NG被AI復活且RIC=OK</div>
                <div class="filter-hint">🔍 AOI=NG, AI=OK, RIC=OK</div>
            </div>
            `;
        }

        // ── Filter by Card ──
        function filterByCard(filterType) {
            if (currentFilter === filterType || filterType === 'total') {
                currentFilter = null;
            } else {
                currentFilter = filterType;
            }

            document.querySelectorAll('.stat-card[data-client-filter]').forEach(c => c.classList.remove('active'));
            if (currentFilter) {
                const active = document.querySelector(`.stat-card[data-client-filter="${currentFilter}"]`);
                if (active) active.classList.add('active');
            }

            const filtered = currentFilter ? rawRecords.filter(row => {
                const eqp = row.result_eqp;
                const ai = row.result_ai;
                const ric = deriveJudgment(row.datastr);
                switch (currentFilter) {
                    case 'aoi_correct': return eqp === ric;
                    case 'ai_correct': return ai === ric;
                    case 'ai_ng': return ai === 'NG';
                    case 'aoi_over': return eqp === 'NG' && ric === 'OK';
                    case 'ai_over': return ai === 'NG' && ric === 'OK';
                    case 'ai_miss': return ai === 'OK' && ric === 'NG';
                    case 'revival': return eqp === 'NG' && ai === 'OK' && ric === 'OK';
                    default: return true;
                }
            }) : rawRecords;

            const labels = {
                total: '全部', aoi_correct: 'AOI 與 DATASTR 判定一致', ai_correct: 'AI 與 DATASTR 判定一致',
                ai_ng: 'AI 判 NG', aoi_over: 'AOI 過檢 (AOI=NG 且 DATASTR=OK)',
                ai_over: 'AI 過檢', ai_miss: 'AI 漏檢', revival: '復活 (AOI=NG, AI=OK, RIC=OK)',
            };

            const labelEl = document.getElementById('client_filterLabel');
            labelEl.textContent = currentFilter ? `— 篩選：${labels[currentFilter]} (${filtered.length} 筆)` : '';

            renderDetails(filtered);
            switchTab('detail');
            document.getElementById('client_tableWrap').scrollTo({ top: 0, behavior: 'smooth' });
        }

        // ── Render Details ──
        function renderDetails(data) {
            const tbody = document.getElementById('client_detailBody');
            tbody.innerHTML = '';

            data.forEach((row, i) => {
                const ric = deriveJudgment(row.datastr);
                const aoiMatch = row.result_eqp === ric;
                const aiMatch = row.result_ai === ric;

                const eqpBadge = row.result_eqp === 'OK'
                    ? '<span class="badge badge-ok">OK</span>'
                    : '<span class="badge badge-ng">NG</span>';
                const aiBadge = row.result_ai === 'OK'
                    ? '<span class="badge badge-ok">OK</span>'
                    : '<span class="badge badge-ng">NG</span>';
                const ricBadgeRaw = row.result_ric === 'ERR'
                    ? '<span class="badge" style="background:rgba(245,158,11,0.1); color:var(--err)">ERR</span>'
                    : row.result_ric === 'OK'
                        ? '<span class="badge badge-ok">OK</span>'
                        : '<span class="badge badge-ng">NG</span>';

                const actualBadge = ric === 'OK'
                    ? '<span class="badge badge-ok">OK</span>'
                    : '<span class="badge badge-ng">NG</span>';

                const aoiComp = aoiMatch
                    ? '<span class="badge badge-match">✓ 一致</span>'
                    : '<span class="badge badge-diff">✗ 不一致</span>';
                const aiComp = aiMatch
                    ? '<span class="badge badge-match">✓ 一致</span>'
                    : '<span class="badge badge-diff">✗ 不一致</span>';

                const tr = document.createElement('tr');
                if (!aiMatch) tr.style.background = 'rgba(239,68,68,0.06)';

                tr.innerHTML = `
                <td style="color:var(--text-muted); font-size:0.8rem;">${i + 1}</td>
                <td style="font-family:'Fira Code',monospace; font-size:0.82rem; color:var(--text-muted);">${row.time_stamp}</td>
                <td style="font-family:'Fira Code',monospace; font-size:0.85rem;">${row.pnl_id}</td>
                <td>${row.mach_id}</td>
                <td>${eqpBadge}</td>
                <td>${aiBadge}</td>
                <td>${ricBadgeRaw}</td>
                <td>${actualBadge}</td>
                <td style="font-size:0.75rem; color:var(--text-muted); max-width:150px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${row.datastr}</td>
                <td>${aoiComp}</td>
                <td>${aiComp}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        // ── Render Charts (unchanged logic, uses summary from API) ──
        function renderCharts(s) {
            if (typeof Chart === 'undefined') return;

            const ctx1 = document.getElementById('client_resultChart').getContext('2d');
            if (resultChart) resultChart.destroy();
            const eqpNG = s.aoiNG, eqpOK = s.total - eqpNG;
            const aiNG = s.aiNG, aiOK = s.total - aiNG;
            const ricNG = s.ricNG, ricOK = s.total - ricNG;

            resultChart = new Chart(ctx1, {
                type: 'bar',
                plugins: [window.ChartDataLabels],
                data: {
                    labels: ['AOI (RESULT_EQP)', 'AI', 'RIC'],
                    datasets: [
                        { label: 'OK', data: [eqpOK, aiOK, ricOK], backgroundColor: 'rgba(34,197,94,0.7)', borderRadius: 4 },
                        { label: 'NG', data: [eqpNG, aiNG, ricNG], backgroundColor: 'rgba(239,68,68,0.7)', borderRadius: 4 },
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: {
                        datalabels: { color: '#e2e8f0', font: { size: 11, weight: 600 }, anchor: 'center', align: 'center', formatter: v => v === 0 ? '' : v },
                        legend: { labels: { color: '#94a3b8' } }
                    },
                    scales: {
                        x: { stacked: true, ticks: { color: '#64748b' }, grid: { color: 'rgba(51,65,85,0.3)' } },
                        y: { stacked: true, ticks: { color: '#64748b' }, grid: { color: 'rgba(51,65,85,0.5)' } }
                    }
                }
            });

            // Combo chart
            const ctx2 = document.getElementById('client_comboChart').getContext('2d');
            if (comboChart) comboChart.destroy();
            const comboLabels = Object.keys(s.combos).sort((a, b) => s.combos[b] - s.combos[a]);
            const comboValues = comboLabels.map(k => s.combos[k]);
            const comboColors = comboLabels.map(k => {
                if (k.includes('ERR')) return 'rgba(245,158,11,0.7)';
                const parts = k.split('/');
                if (parts[2] === 'OK' && parts[0] === 'NG' && parts[1] === 'NG') return 'rgba(239,68,68,0.5)';
                if (parts[0] === parts[2] && parts[1] === parts[2]) return 'rgba(34,197,94,0.7)';
                return 'rgba(129,140,248,0.6)';
            });

            comboChart = new Chart(ctx2, {
                type: 'bar',
                plugins: [window.ChartDataLabels],
                data: {
                    labels: comboLabels.map(k => `AOI/AI/RIC: ${k}`),
                    datasets: [{ label: '筆數', data: comboValues, backgroundColor: comboColors, borderRadius: 4 }]
                },
                options: {
                    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                    plugins: {
                        datalabels: { color: '#e2e8f0', font: { size: 10, weight: 600 }, anchor: 'end', align: 'left', formatter: v => v === 0 ? '' : v },
                        legend: { display: false }
                    },
                    scales: {
                        x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(51,65,85,0.3)' } },
                        y: { ticks: { color: '#94a3b8', font: { family: "'Fira Code'" } }, grid: { display: false } }
                    }
                }
            });

            // Trend chart
            const ctx3 = document.getElementById('client_trendChart').getContext('2d');
            if (trendChart) trendChart.destroy();
            const dateKeys = Object.keys(s.daily).sort();

            trendChart = new Chart(ctx3, {
                type: 'line',
                plugins: [window.ChartDataLabels],
                data: {
                    labels: dateKeys,
                    datasets: [
                        {
                            label: 'AOI 準確率 (%)',
                            data: dateKeys.map(k => { const d = s.daily[k]; return d.total > 0 ? (d.aoiCorrect / d.total * 100).toFixed(1) : null; }),
                            borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)', tension: 0.3, fill: true, pointRadius: 4, pointBackgroundColor: '#f59e0b',
                        },
                        {
                            label: 'AI 準確率 (%)',
                            data: dateKeys.map(k => { const d = s.daily[k]; return d.total > 0 ? (d.aiCorrect / d.total * 100).toFixed(1) : null; }),
                            borderColor: '#818cf8', backgroundColor: 'rgba(129,140,248,0.1)', tension: 0.3, fill: true, pointRadius: 4, pointBackgroundColor: '#818cf8',
                        },
                        {
                            label: '筆數', data: dateKeys.map(k => s.daily[k].total),
                            type: 'bar', backgroundColor: 'rgba(148,163,184,0.15)', borderRadius: 4, yAxisID: 'y1',
                        }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: {
                        datalabels: {
                            color: ctx => ['#f59e0b', '#818cf8', '#94a3b8'][ctx.datasetIndex],
                            font: { size: 10, weight: 600 },
                            align: ctx => ctx.dataset.type === 'bar' ? 'end' : 'top',
                            anchor: ctx => ctx.dataset.type === 'bar' ? 'end' : 'center',
                            formatter: (v, ctx) => { if (v === null || v === '0.0' || v === 0) return ''; return ctx.dataset.type === 'bar' ? v : v + '%'; }
                        },
                        legend: { labels: { color: '#94a3b8' } }
                    },
                    scales: {
                        x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(51,65,85,0.3)' } },
                        y: { min: 0, max: 100, ticks: { color: '#64748b', callback: v => v + '%' }, grid: { color: 'rgba(51,65,85,0.5)' } },
                        y1: { position: 'right', min: 0, ticks: { color: '#475569' }, grid: { display: false } },
                    }
                }
            });

            // Confusion matrix chart
            const ctx4 = document.getElementById('client_confusionChart').getContext('2d');
            if (confusionChart) confusionChart.destroy();
            const aoiFP = s.aoiOver;
            const aiFP = s.aiOver;
            const aiFN = s.aiMiss;
            const aiTP = s.aiNG - aiFP;
            const aiTN = s.total - s.aiNG - aiFN;
            // AOI TP/TN not fully reliable (spec says AOI=OK has no ref value) but show for comparison
            const aoiTP = s.aoiNG - aoiFP;
            const aoiFN = 0; // AOI漏檢率已移除，設為0
            const aoiTN = s.total - s.aoiNG;

            confusionChart = new Chart(ctx4, {
                type: 'bar',
                plugins: [window.ChartDataLabels],
                data: {
                    labels: ['TP (判NG 實NG)', 'FP (過檢)', 'FN (漏檢)', 'TN (判OK 實OK)'],
                    datasets: [
                        { label: 'AOI', data: [aoiTP, aoiFP, aoiFN, aoiTN], backgroundColor: 'rgba(245,158,11,0.6)', borderRadius: 4 },
                        { label: 'AI', data: [aiTP, aiFP, aiFN, aiTN], backgroundColor: 'rgba(129,140,248,0.6)', borderRadius: 4 },
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: {
                        datalabels: { color: '#e2e8f0', font: { size: 11, weight: 600 }, anchor: 'center', align: 'center', formatter: v => v === 0 ? '' : v },
                        legend: { labels: { color: '#94a3b8' } }
                    },
                    scales: {
                        x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(51,65,85,0.3)' } },
                        y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(51,65,85,0.5)' } }
                    }
                }
            });

            // Daily metrics chart
            const ctx5 = document.getElementById('client_dailyMetricsChart').getContext('2d');
            if (dailyMetricsChart) dailyMetricsChart.destroy();

            dailyMetricsChart = new Chart(ctx5, {
                type: 'line',
                plugins: [window.ChartDataLabels],
                data: {
                    labels: dateKeys,
                    datasets: [
                        {
                            label: 'AI 漏檢率 (%)',
                            data: dateKeys.map(k => { const d = s.daily[k]; return d.total > 0 ? (d.aiMiss / d.total * 100).toFixed(1) : null; }),
                            borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', borderDash: [5, 5], tension: 0.3, fill: false, pointRadius: 4, pointBackgroundColor: '#ef4444',
                        },
                        {
                            label: 'AI 過檢率 (%)',
                            data: dateKeys.map(k => { const d = s.daily[k]; return d.total > 0 ? (d.aiOver / d.total * 100).toFixed(1) : null; }),
                            borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', tension: 0.3, fill: false, pointRadius: 4, pointBackgroundColor: '#3b82f6',
                        },
                        {
                            label: 'AOI 過檢率 (%)',
                            data: dateKeys.map(k => { const d = s.daily[k]; return d.total > 0 ? (d.aoiOver / d.total * 100).toFixed(1) : null; }),
                            borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)', tension: 0.3, fill: false, pointRadius: 4, pointBackgroundColor: '#f59e0b',
                        },
                        {
                            label: '當日總數量', data: dateKeys.map(k => s.daily[k].total),
                            type: 'bar', backgroundColor: 'rgba(148,163,184,0.15)', borderRadius: 4, yAxisID: 'y1',
                        }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: {
                        datalabels: {
                            color: ctx => ['#ef4444', '#3b82f6', '#f59e0b', '#94a3b8'][ctx.datasetIndex],
                            font: { size: 10, weight: 600 },
                            align: ctx => ctx.dataset.type === 'bar' ? 'end' : 'top',
                            anchor: ctx => ctx.dataset.type === 'bar' ? 'end' : 'center',
                            formatter: (v, ctx) => { if (v === null || v === '0.0' || v === 0) return ''; return ctx.dataset.type === 'bar' ? v : v + '%'; }
                        },
                        legend: { labels: { color: '#94a3b8' } },
                        tooltip: {
                            callbacks: {
                                label: function (context) {
                                    let label = context.dataset.label || '';
                                    if (label) label += ': ';
                                    if (context.parsed.y !== null) {
                                        label += context.parsed.y;
                                        if (context.dataset.yAxisID !== 'y1') label += '%';
                                    }
                                    const day = context.chart.data.labels[context.dataIndex];
                                    const d = s.daily[day];
                                    if (context.datasetIndex === 0) label += ` (${d.aiMiss} / ${d.total})`;
                                    if (context.datasetIndex === 1) label += ` (${d.aiOver} / ${d.total})`;
                                    if (context.datasetIndex === 2) label += ` (${d.aoiOver} / ${d.total})`;
                                    return label;
                                }
                            }
                        }
                    },
                    scales: {
                        x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(51,65,85,0.3)' } },
                        y: { min: 0, title: { display: true, text: '百分比 (%)', color: '#64748b' }, ticks: { color: '#64748b', callback: v => v + '%' }, grid: { color: 'rgba(51,65,85,0.5)' } },
                        y1: { position: 'right', min: 0, title: { display: true, text: '數量', color: '#64748b' }, ticks: { color: '#475569' }, grid: { display: false } },
                    }
                }
            });
        }

        // ── Export CSV ──
        function exportCSV() {
            const tbody = document.getElementById('client_detailBody');
            const rows = tbody.querySelectorAll('tr');
            if (rows.length === 0) { alert('無資料可導出'); return; }

            const headers = ['#', '檢測時間', 'Panel ID', '機台', 'RESULT_EQP', 'RESULT_AI', 'RIC_RAW', 'DATASTR_JUDGE', 'DATASTR_RAW', 'AOI vs DATASTR', 'AI vs DATASTR'];
            let csv = '\ufeff' + headers.join(',') + '\n';

            rows.forEach(tr => {
                const cells = tr.querySelectorAll('td');
                const rowData = Array.from(cells).map(td => {
                    let text = td.textContent.trim();
                    if (text.includes(',') || text.includes('"')) text = '"' + text.replace(/"/g, '""') + '"';
                    return text;
                });
                csv += rowData.join(',') + '\n';
            });

            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            const ts = new Date().toISOString().slice(0, 16).replace(/[-T:]/g, '');
            a.href = url;
            a.download = `ai_accuracy_report_${ts}.csv`;
            a.click();
            URL.revokeObjectURL(url);
        }

        // ── Review Tab ──
        const CATEGORY_LABELS = {
            'dust_misfilter': 'Dust 誤濾',
            'threshold_high': '閾值偏高',
            'ric_misjudge': 'RIC 誤判',
            'outside_aoi_area': '漏檢區域不在 AOI 提供區域',
            'other': '其他',
        };

        const CATEGORY_COLORS = {
            'dust_misfilter': '#f59e0b',
            'threshold_high': '#3b82f6',
            'ric_misjudge': '#22c55e',
            'outside_aoi_area': '#a855f7',
            'other': '#64748b',
        };

        function renderReviewTab(summary, records) {
            const mrs = summary.missReviewStats;
            renderReviewStats(mrs);
            renderReviewPieChart(mrs);
            renderReviewTable(records);
        }

        function renderReviewStats(mrs) {
            const grid = document.getElementById('review_statsGrid');
            const completionRate = mrs.total > 0 ? (mrs.reviewed / mrs.total * 100).toFixed(1) : 0;
            grid.innerHTML = `
            <div class="stat-card miss">
                <div class="label">漏檢總數</div>
                <div class="value">${mrs.total}</div>
                <div class="detail">AI=OK, RIC=NG</div>
            </div>
            <div class="stat-card ai">
                <div class="label">已 Review</div>
                <div class="value">${mrs.reviewed}</div>
                <div class="detail">${mrs.total > 0 ? (mrs.reviewed / mrs.total * 100).toFixed(1) : 0}%</div>
            </div>
            <div class="stat-card over">
                <div class="label">未 Review</div>
                <div class="value">${mrs.unreviewed}</div>
                <div class="detail">${mrs.total > 0 ? (mrs.unreviewed / mrs.total * 100).toFixed(1) : 0}%</div>
            </div>
            <div class="stat-card match">
                <div class="label">完成率</div>
                <div class="value">${completionRate}%</div>
                <div class="detail">${mrs.reviewed} / ${mrs.total}</div>
            </div>
            `;
        }

        function renderReviewPieChart(mrs) {
            const ctx = document.getElementById('review_pieChart').getContext('2d');
            if (reviewPieChart) reviewPieChart.destroy();

            const labels = [];
            const data = [];
            const colors = [];

            Object.entries(mrs.byCategory).forEach(([key, count]) => {
                if (count > 0) {
                    labels.push(CATEGORY_LABELS[key] || key);
                    data.push(count);
                    colors.push(CATEGORY_COLORS[key] || '#64748b');
                }
            });

            if (mrs.unreviewed > 0) {
                labels.push('未 Review');
                data.push(mrs.unreviewed);
                colors.push('rgba(148,163,184,0.4)');
            }

            if (data.length === 0) {
                // No data at all
                labels.push('無漏檢');
                data.push(1);
                colors.push('rgba(148,163,184,0.2)');
            }

            reviewPieChart = new Chart(ctx, {
                type: 'doughnut',
                plugins: [window.ChartDataLabels],
                data: {
                    labels: labels,
                    datasets: [{ data: data, backgroundColor: colors, borderWidth: 0 }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { color: '#94a3b8', padding: 16 } },
                        datalabels: {
                            color: '#e2e8f0', font: { size: 12, weight: 600 },
                            formatter: (v, ctx) => {
                                const total = ctx.chart.data.datasets[0].data.reduce((a, b) => a + b, 0);
                                return total > 0 && v > 0 ? v + ' (' + (v / total * 100).toFixed(0) + '%)' : '';
                            }
                        }
                    }
                }
            });
        }

        function renderReviewTable(records) {
            const missRecords = records.filter(r => r.result_ai === 'OK' && deriveJudgment(r.datastr) === 'NG');
            document.getElementById('review_countLabel').textContent = `(${missRecords.length} 筆)`;

            const tbody = document.getElementById('review_detailBody');
            tbody.innerHTML = '';

            missRecords.forEach((row, i) => {
                const review = row.miss_review;
                const hasReview = review && review.category;

                const categoryOptions = Object.entries(CATEGORY_LABELS).map(([k, v]) =>
                    `<option value="${k}" ${hasReview && review.category === k ? 'selected' : ''}>${v}</option>`
                ).join('');

                const tr = document.createElement('tr');
                tr.id = `review_row_${row.id}`;
                tr.innerHTML = `
                <td style="color:var(--text-muted); font-size:0.8rem;">${i + 1}</td>
                <td style="font-family:'Fira Code',monospace; font-size:0.82rem; color:var(--text-muted);">${row.time_stamp}</td>
                <td style="font-family:'Fira Code',monospace; font-size:0.85rem;">${row.pnl_id}</td>
                <td>${row.mach_id}</td>
                <td><span class="badge badge-ng">NG</span></td>
                <td><span class="badge badge-ok">OK</span></td>
                <td><span class="badge badge-ng">NG</span></td>
                <td>
                    <select class="review-select" id="review_cat_${row.id}" onchange="clientComp._markDirty(${row.id})">
                        <option value="">請選擇</option>
                        ${categoryOptions}
                    </select>
                </td>
                <td>
                    <input type="text" class="review-input" id="review_note_${row.id}"
                           value="${hasReview ? (review.note || '').replace(/"/g, '&quot;') : ''}"
                           placeholder="備註 (選填)"
                           onchange="clientComp._markDirty(${row.id})">
                </td>
                <td id="review_action_${row.id}">
                    ${hasReview
                        ? `<span class="review-saved">已儲存 ✓</span><button class="review-btn-delete" onclick="clientComp._deleteReview(${row.id})">刪除</button>`
                        : `<button class="review-btn-save" id="review_save_${row.id}" onclick="clientComp._saveReview(${row.id})">儲存</button>`
                    }
                </td>
                `;
                tbody.appendChild(tr);
            });
        }

        function _markDirty(recordId) {
            const actionCell = document.getElementById(`review_action_${recordId}`);
            actionCell.innerHTML = `<button class="review-btn-save" onclick="clientComp._saveReview(${recordId})">儲存</button>`;
        }

        function _saveReview(recordId) {
            const category = document.getElementById(`review_cat_${recordId}`).value;
            if (!category) { alert('請選擇分類'); return; }
            const note = document.getElementById(`review_note_${recordId}`).value;

            const btn = document.querySelector(`#review_action_${recordId} .review-btn-save`);
            if (btn) { btn.disabled = true; btn.textContent = '儲存中...'; }

            fetch('/api/ric/miss-review', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ client_record_id: recordId, category, note })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    const actionCell = document.getElementById(`review_action_${recordId}`);
                    actionCell.innerHTML = `<span class="review-saved">已儲存 ✓</span><button class="review-btn-delete" onclick="clientComp._deleteReview(${recordId})">刪除</button>`;
                    // Update local record
                    const rec = rawRecords.find(r => r.id === recordId);
                    if (rec) rec.miss_review = { id: data.id, category, note, updated_at: new Date().toISOString() };
                    // Re-render stats (lightweight, no full reload)
                    _updateReviewStats();
                } else {
                    alert('❌ ' + (data.error || '儲存失敗'));
                    if (btn) { btn.disabled = false; btn.textContent = '儲存'; }
                }
            })
            .catch(err => {
                alert('❌ 網路錯誤: ' + err.message);
                if (btn) { btn.disabled = false; btn.textContent = '儲存'; }
            });
        }

        function _deleteReview(recordId) {
            if (!confirm('確定要刪除此 Review？')) return;

            fetch('/api/ric/miss-review/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ client_record_id: recordId })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    document.getElementById(`review_cat_${recordId}`).value = '';
                    document.getElementById(`review_note_${recordId}`).value = '';
                    const actionCell = document.getElementById(`review_action_${recordId}`);
                    actionCell.innerHTML = `<button class="review-btn-save" onclick="clientComp._saveReview(${recordId})">儲存</button>`;
                    const rec = rawRecords.find(r => r.id === recordId);
                    if (rec) rec.miss_review = null;
                    _updateReviewStats();
                } else {
                    alert('❌ ' + (data.error || '刪除失敗'));
                }
            })
            .catch(err => alert('❌ 網路錯誤: ' + err.message));
        }

        function _updateReviewStats() {
            // Recompute missReviewStats from local rawRecords
            const missRecords = rawRecords.filter(r => r.result_ai === 'OK' && deriveJudgment(r.datastr) === 'NG');
            const total = missRecords.length;
            let reviewed = 0;
            const byCategory = { dust_misfilter: 0, threshold_high: 0, ric_misjudge: 0, outside_aoi_area: 0, other: 0 };

            missRecords.forEach(r => {
                if (r.miss_review && r.miss_review.category) {
                    reviewed++;
                    const cat = r.miss_review.category;
                    if (cat in byCategory) byCategory[cat]++;
                }
            });

            const mrs = { total, reviewed, unreviewed: total - reviewed, byCategory };
            renderReviewStats(mrs);
            renderReviewPieChart(mrs);
        }

        // ── Init ──
        initUpload();
        // Auto-load data on page load (default: all)
        loadData('', '');

        return {
            switchTab, filterByCard, exportCSV, clearDB,
            quickFilter, customFilter, loadData, toggleUpload,
            _saveReview, _deleteReview, _markDirty,
        };
    })();
```

- [ ] **Step 2: Commit**

```bash
git add templates/ric_report.html
git commit -m "feat: rewrite clientComp JS with API-driven loading, date filter, and review tab"
```

---

### Task 5: Integration test — Manual verification

**Files:** None (runtime verification)

- [ ] **Step 1: Start the server**

```bash
python capi_server.py --config server_config_local.yaml
```

- [ ] **Step 2: Open RIC Report page**

Open `http://localhost:8080/ric` in the browser. Verify:
- Page loads without errors (console clean)
- If DB has data: report displays immediately with "全部" date filter active
- If DB is empty: shows empty state message
- Date filter quick buttons work (今日/7天/30天/全部)
- Custom date range works

- [ ] **Step 3: Test upload flow**

- Click "匯入資料" button → upload panel expands
- Upload a test XLS file → data saves and report refreshes
- Upload panel collapses after success

- [ ] **Step 4: Test Review tab**

- Switch to "漏檢 Review" tab
- Verify stats cards show correct counts
- Verify pie chart renders
- Select a category for a miss detection record → click save → shows "已儲存 ✓"
- Click delete → review cleared
- Stats and pie chart update in real-time

- [ ] **Step 5: Test cross-tab date filter**

- Change date filter to "7天"
- Switch between 統計分析 / 逐筆明細 / 漏檢 Review tabs
- All three tabs should show data for the same date range

- [ ] **Step 6: Commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: address integration test findings"
```
