# RIC 頁面「AI 推論紀錄」分頁 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `/ric` 頁面新增「AI 推論紀錄」分頁 Tab，顯示推論統計總覽、趨勢圖表與多維度分析。

**Architecture:** 後端新增一個 API endpoint (`GET /api/ric/inference-stats`)，在 `capi_database.py` 中查詢 `inference_records` 表做聚合統計。前端在 `ric_report.html` 頂部加入 Tab bar，用 CSS display 切換兩個 Tab 內容區塊，「AI 推論紀錄」透過 AJAX 載入資料並用 Chart.js 渲染圖表。

**Tech Stack:** Python 3 (SQLite), Jinja2 template, Chart.js, vanilla JavaScript (IIFE)

---

## File Map

| 動作 | 檔案 | 負責內容 |
|------|------|---------|
| Modify | `capi_database.py:1148` (在 `get_ric_accuracy_stats` 之後) | 新增 `get_inference_stats()` 方法 |
| Modify | `capi_web.py:157` (GET 路由區) | 新增 `/api/ric/inference-stats` 路由 |
| Modify | `capi_web.py:1126` (RIC handler 區) | 新增 `_handle_inference_stats_api()` 方法 |
| Modify | `templates/ric_report.html:4-416` (CSS 區) | 新增頂層 Tab CSS |
| Modify | `templates/ric_report.html:426-556` (HTML 區) | 包裹現有內容 + 新增 Tab bar 和 AI 推論紀錄 HTML |
| Modify | `templates/ric_report.html:562` (JS 區尾部) | 新增 `inferenceTab` IIFE 模組 |

---

### Task 1: 後端 — `get_inference_stats()` 資料庫方法

**Files:**
- Modify: `capi_database.py:1148` (在 `get_ric_accuracy_stats` 方法之後、`get_config_param` 之前)

- [ ] **Step 1: 新增 `get_inference_stats` 方法**

在 `capi_database.py` 第 1149 行（`# ── 設定參數管理方法` 之前）插入：

```python
    def get_inference_stats(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        """
        取得推論紀錄統計資料（供 AI 推論紀錄 Tab 使用）

        Args:
            start_date: 起始日期 YYYY-MM-DD（含）
            end_date: 結束日期 YYYY-MM-DD（含）
        """
        conn = self._get_conn()
        try:
            # 建立日期篩選條件
            where_clauses = []
            params = []
            if start_date:
                where_clauses.append("DATE(request_time) >= ?")
                params.append(start_date)
            if end_date:
                where_clauses.append("DATE(request_time) <= ?")
                params.append(end_date)
            where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

            # ── 1. Summary 統計 ──
            rows = conn.execute(
                f"SELECT machine_judgment, ai_judgment, error_message FROM inference_records{where_sql}",
                params
            ).fetchall()

            total = len(rows)
            aoi_ng = 0
            ai_ng = 0
            ai_revival = 0  # AOI NG → AI OK
            err_count = 0
            ok_ok = 0
            ng_ok = 0
            ok_ng = 0
            ng_ng = 0
            err_types_counter: Dict[str, int] = {}

            for row in rows:
                mj = (row["machine_judgment"] or "").strip()
                aj = (row["ai_judgment"] or "").strip()
                em = (row["error_message"] or "").strip()

                is_aoi_ng = mj != "" and mj != "OK"
                is_ai_ng = aj.startswith("NG")
                is_err = aj.startswith("ERR")

                if is_aoi_ng:
                    aoi_ng += 1
                if is_ai_ng:
                    ai_ng += 1
                if is_err:
                    err_count += 1
                    # ERR 類型分類
                    err_desc = aj[4:].strip() if len(aj) > 4 else (em if em else "Unknown")
                    err_types_counter[err_desc] = err_types_counter.get(err_desc, 0) + 1
                if is_aoi_ng and aj == "OK":
                    ai_revival += 1

                # 交叉比對矩陣（排除 ERR）
                if not is_err:
                    if not is_aoi_ng and not is_ai_ng:
                        ok_ok += 1
                    elif is_aoi_ng and not is_ai_ng:
                        ng_ok += 1
                    elif not is_aoi_ng and is_ai_ng:
                        ok_ng += 1
                    elif is_aoi_ng and is_ai_ng:
                        ng_ng += 1

            # ── 2. 每日趨勢 ──
            daily_rows = conn.execute(
                f"""SELECT DATE(request_time) as date,
                           COUNT(*) as total,
                           SUM(CASE WHEN machine_judgment != '' AND machine_judgment != 'OK' THEN 1 ELSE 0 END) as aoi_ng,
                           SUM(CASE WHEN ai_judgment LIKE 'NG%' THEN 1 ELSE 0 END) as ai_ng,
                           SUM(CASE WHEN ai_judgment LIKE 'ERR%' THEN 1 ELSE 0 END) as err
                    FROM inference_records{where_sql}
                    GROUP BY DATE(request_time)
                    ORDER BY date""",
                params
            ).fetchall()
            daily_trend = [dict(r) for r in daily_rows]

            # ── 3. 機台統計 ──
            machine_rows = conn.execute(
                f"""SELECT machine_no as machine,
                           COUNT(*) as total,
                           ROUND(SUM(CASE WHEN machine_judgment != '' AND machine_judgment != 'OK' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as aoi_ng_rate,
                           ROUND(SUM(CASE WHEN ai_judgment LIKE 'NG%' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as ai_ng_rate
                    FROM inference_records{where_sql}
                    GROUP BY machine_no
                    ORDER BY total DESC""",
                params
            ).fetchall()
            by_machine = [dict(r) for r in machine_rows]

            # ── 4. 產品型號統計 ──
            model_rows = conn.execute(
                f"""SELECT model_id as model, COUNT(*) as total
                    FROM inference_records{where_sql}
                    GROUP BY model_id
                    ORDER BY total DESC""",
                params
            ).fetchall()
            by_model = [dict(r) for r in model_rows]

            # ── 5. ERR 類型排序 ──
            err_types = sorted(
                [{"type": k, "count": v} for k, v in err_types_counter.items()],
                key=lambda x: x["count"],
                reverse=True
            )

            return {
                "success": True,
                "summary": {
                    "total": total,
                    "aoi_ng": aoi_ng,
                    "ai_ng": ai_ng,
                    "ai_revival": ai_revival,
                    "err_count": err_count,
                },
                "daily_trend": daily_trend,
                "by_machine": by_machine,
                "by_model": by_model,
                "err_types": err_types,
                "cross_matrix": {
                    "ok_ok": ok_ok,
                    "ng_ok": ng_ok,
                    "ok_ng": ok_ng,
                    "ng_ng": ng_ng,
                },
            }
        finally:
            conn.close()
```

- [ ] **Step 2: 驗證語法**

Run: `python -c "import capi_database; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add capi_database.py
git commit -m "feat: add get_inference_stats() for AI inference tab"
```

---

### Task 2: 後端 — API 路由

**Files:**
- Modify: `capi_web.py:157` (GET 路由分發)
- Modify: `capi_web.py:1126` (handler 區)

- [ ] **Step 1: 在 GET 路由分發中新增 `/api/ric/inference-stats`**

在 `capi_web.py` 第 157 行（`elif path == "/api/ric/client-data":` 之後）插入：

```python
            elif path == "/api/ric/inference-stats":
                self._handle_inference_stats_api(query)
```

- [ ] **Step 2: 新增 `_handle_inference_stats_api` handler 方法**

在 `capi_web.py` 的 `_handle_ric_report_api` 方法之後（約第 1127 行）插入：

```python
    def _handle_inference_stats_api(self, query: dict):
        """API: 取得 AI 推論紀錄統計資料"""
        try:
            start_date = query.get('start_date', [''])[0] or None
            end_date = query.get('end_date', [''])[0] or None
            stats = self.db.get_inference_stats(start_date, end_date) if self.db else {"success": False, "error": "DB not available"}
            self._send_json(stats)
        except Exception as e:
            logger.error(f"Inference stats API error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})
```

- [ ] **Step 3: 啟動伺服器測試 API**

Run: 啟動伺服器後，瀏覽器訪問 `http://localhost:8080/api/ric/inference-stats`
Expected: 回傳含 `"success": true` 的 JSON，包含 summary、daily_trend、by_machine、by_model、err_types、cross_matrix

Run: `http://localhost:8080/api/ric/inference-stats?start_date=2026-03-26&end_date=2026-03-26`
Expected: 只返回今日的統計資料

- [ ] **Step 4: Commit**

```bash
git add capi_web.py
git commit -m "feat: add /api/ric/inference-stats endpoint"
```

---

### Task 3: 前端 — 頂層 Tab Bar CSS + HTML 結構

**Files:**
- Modify: `templates/ric_report.html:4-416` (CSS 區塊)
- Modify: `templates/ric_report.html:426-556` (HTML 區塊)

- [ ] **Step 1: 新增頂層 Tab CSS**

在 `templates/ric_report.html` 的 `</style>` 結束標籤之前（約第 416 行），插入以下 CSS：

```css
    /* ── Top-level Tab Bar ── */
    .top-tab-bar {
        display: flex;
        gap: 0;
        border-bottom: 2px solid var(--border);
        margin-bottom: 24px;
        background: var(--surface);
        border-radius: 12px 12px 0 0;
        padding: 0 8px;
    }

    .top-tab {
        padding: 14px 28px;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-muted);
        cursor: pointer;
        border-bottom: 3px solid transparent;
        margin-bottom: -2px;
        transition: all 0.2s;
        user-select: none;
    }

    .top-tab:hover {
        color: var(--text);
        background: rgba(255, 255, 255, 0.03);
    }

    .top-tab.active {
        color: var(--accent);
        border-bottom-color: var(--accent);
    }

    .top-tab-content {
        display: none;
    }

    .top-tab-content.active {
        display: block;
    }

    /* ── Inference Tab specific ── */
    .inf-date-bar {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }

    .inf-date-bar .quick-btn {
        padding: 6px 16px;
        border-radius: 6px;
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text-muted);
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s;
    }

    .inf-date-bar .quick-btn:hover {
        color: var(--text);
        border-color: var(--accent);
    }

    .inf-date-bar .quick-btn.active {
        background: var(--accent);
        color: white;
        border-color: var(--accent);
    }

    .inf-date-bar input[type="date"] {
        background: var(--surface);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 0.85rem;
    }

    .inf-date-bar .search-btn {
        padding: 6px 16px;
        border-radius: 6px;
        border: none;
        background: var(--accent);
        color: white;
        font-size: 0.85rem;
        font-weight: 600;
        cursor: pointer;
    }

    .inf-stats-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin-bottom: 20px;
    }

    .inf-stat-card {
        background: var(--surface);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid var(--border);
        position: relative;
        overflow: hidden;
    }

    .inf-stat-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
    }

    .inf-stat-card .label { color: var(--text-muted); font-size: 0.8rem; }
    .inf-stat-card .value { font-size: 1.6rem; font-weight: 700; margin: 4px 0; }
    .inf-stat-card .pct { font-size: 0.78rem; }

    .inf-stat-card.aoi-ng::before { background: #f97316; }
    .inf-stat-card.aoi-ng .value, .inf-stat-card.aoi-ng .pct { color: #f97316; }
    .inf-stat-card.ai-ng::before { background: #3b82f6; }
    .inf-stat-card.ai-ng .value, .inf-stat-card.ai-ng .pct { color: #3b82f6; }
    .inf-stat-card.revival::before { background: #22c55e; }
    .inf-stat-card.revival .value, .inf-stat-card.revival .pct { color: #22c55e; }
    .inf-stat-card.err::before { background: #f59e0b; }
    .inf-stat-card.err .value, .inf-stat-card.err .pct { color: #f59e0b; }

    .inf-charts-row {
        display: grid;
        gap: 16px;
        margin-bottom: 16px;
    }

    .inf-charts-row.row-2-1 { grid-template-columns: 2fr 1fr; }
    .inf-charts-row.row-3 { grid-template-columns: 1fr 1fr 1fr; }
    .inf-charts-row.row-1 { grid-template-columns: 1fr; }

    .inf-chart-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
    }

    .inf-chart-card h3 {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text);
        margin: 0 0 12px 0;
    }

    .inf-chart-container {
        position: relative;
        height: 280px;
    }

    .inf-cross-matrix {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
    }

    .inf-cross-cell {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .inf-cross-cell .cell-label { color: var(--text-muted); font-size: 0.78rem; }
    .inf-cross-cell .cell-value { font-size: 1.5rem; font-weight: 700; margin: 6px 0; }
    .inf-cross-cell .cell-desc { font-size: 0.75rem; }

    .inf-loading {
        text-align: center;
        padding: 60px 0;
        color: var(--text-muted);
    }
```

- [ ] **Step 2: 改造 HTML — 加入 Tab bar 並包裹現有內容**

在 `templates/ric_report.html` 中，將 `{% block content %}` 之後的內容改為如下結構。注意：**現有的 HTML（從 `<div class="ric-header">` 到 `</script>` 之前的結尾 `</div>`）全部原封不動包進 `top-tab-content` 的 `#tab-ric` 裡面**。

在 `{% block content %}` 之後、`<div class="ric-header">` 之前插入：

```html
<!-- Top-level Tab Bar -->
<div class="top-tab-bar">
    <div class="top-tab active" data-top-tab="ric" onclick="switchTopTab('ric')">📊 RIC Report</div>
    <div class="top-tab" data-top-tab="inference" onclick="switchTopTab('inference')">🤖 AI 推論紀錄</div>
</div>

<!-- Tab 1: RIC Report (existing content) -->
<div class="top-tab-content active" id="tab-ric">
```

在現有 RIC Report 內容結尾（`</div>` — 即 `id="client_reportContent"` 的 `</div>` 之後、`<script>` 之前）插入：

```html
</div><!-- end #tab-ric -->

<!-- Tab 2: AI 推論紀錄 -->
<div class="top-tab-content" id="tab-inference">
    <!-- Date Filter Bar -->
    <div class="inf-date-bar">
        <span style="color:var(--text-muted); font-size:0.88rem; margin-right:4px;">日期篩選：</span>
        <button class="quick-btn active" data-range="today" onclick="inferenceTab.quickFilter('today')">今日</button>
        <button class="quick-btn" data-range="7d" onclick="inferenceTab.quickFilter('7d')">7 天</button>
        <button class="quick-btn" data-range="30d" onclick="inferenceTab.quickFilter('30d')">30 天</button>
        <button class="quick-btn" data-range="all" onclick="inferenceTab.quickFilter('all')">全部</button>
        <span style="margin-left:12px;"></span>
        <input type="date" id="inf_startDate">
        <span style="color:var(--text-muted);">~</span>
        <input type="date" id="inf_endDate">
        <button class="search-btn" onclick="inferenceTab.customFilter()">查詢</button>
    </div>

    <!-- Loading -->
    <div class="inf-loading" id="inf_loading">載入中...</div>

    <!-- Stats Cards -->
    <div class="inf-stats-grid" id="inf_statsGrid" style="display:none;"></div>

    <!-- Charts Row 1: Trend + Distribution -->
    <div class="inf-charts-row row-2-1" id="inf_chartsRow1" style="display:none;">
        <div class="inf-chart-card">
            <h3>📈 每日推論趨勢（AOI vs AI）</h3>
            <div class="inf-chart-container"><canvas id="inf_trendChart"></canvas></div>
        </div>
        <div class="inf-chart-card">
            <h3>🎯 AOI vs AI 判定分布</h3>
            <div class="inf-chart-container"><canvas id="inf_distChart"></canvas></div>
        </div>
    </div>

    <!-- Charts Row 2: Machine + Model + ERR -->
    <div class="inf-charts-row row-3" id="inf_chartsRow2" style="display:none;">
        <div class="inf-chart-card">
            <h3>🏭 機台統計（AOI vs AI NG率）</h3>
            <div class="inf-chart-container"><canvas id="inf_machineChart"></canvas></div>
        </div>
        <div class="inf-chart-card">
            <h3>📦 產品型號統計</h3>
            <div class="inf-chart-container"><canvas id="inf_modelChart"></canvas></div>
        </div>
        <div class="inf-chart-card">
            <h3>⚠️ ERR 類型分布</h3>
            <div class="inf-chart-container"><canvas id="inf_errChart"></canvas></div>
        </div>
    </div>

    <!-- Cross Matrix -->
    <div class="inf-charts-row row-1" id="inf_chartsRow3" style="display:none;">
        <div class="inf-chart-card">
            <h3>🔄 AOI vs AI 判定交叉比對</h3>
            <div class="inf-cross-matrix" id="inf_crossMatrix"></div>
        </div>
    </div>
</div><!-- end #tab-inference -->
```

- [ ] **Step 3: 驗證 HTML 結構**

在瀏覽器開啟 `http://localhost:8080/ric`，確認：
- 頂部出現兩個 Tab（RIC Report / AI 推論紀錄）
- 預設顯示 RIC Report，現有功能完全正常
- Tab 尚無法切換（JS 還沒加）

- [ ] **Step 4: Commit**

```bash
git add templates/ric_report.html
git commit -m "feat: add top-level tab bar and AI inference tab HTML/CSS structure"
```

---

### Task 4: 前端 — Tab 切換 + `inferenceTab` IIFE JS 模組

**Files:**
- Modify: `templates/ric_report.html` (JS 區，在 `</script>` 之前、`clientComp` IIFE 之後)

- [ ] **Step 1: 新增全域 Tab 切換函數和 `inferenceTab` IIFE**

在 `templates/ric_report.html` 的 `</script>` 結束標籤之前，插入以下 JavaScript：

```javascript
    // ══════════════════════════════════════════
    // Top-level Tab Switching
    // ══════════════════════════════════════════
    function switchTopTab(tabName) {
        document.querySelectorAll('.top-tab').forEach(t => t.classList.toggle('active', t.dataset.topTab === tabName));
        document.querySelectorAll('.top-tab-content').forEach(p => p.classList.toggle('active', p.id === 'tab-' + tabName));
        if (tabName === 'inference' && !inferenceTab._loaded) {
            inferenceTab.quickFilter('today');
        }
    }

    // ══════════════════════════════════════════
    // AI 推論紀錄 Tab Module
    // ══════════════════════════════════════════
    const inferenceTab = (function () {
        let _loaded = false;
        let trendChart = null, distChart = null, machineChart = null, modelChart = null, errChart = null;

        function _destroyCharts() {
            [trendChart, distChart, machineChart, modelChart, errChart].forEach(c => { if (c) c.destroy(); });
            trendChart = distChart = machineChart = modelChart = errChart = null;
        }

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
            document.querySelectorAll('.inf-date-bar .quick-btn').forEach(b => b.classList.toggle('active', b.dataset.range === range));
            const { start, end } = _getDateRange(range);
            document.getElementById('inf_startDate').value = start;
            document.getElementById('inf_endDate').value = end;
            loadData(start, end);
        }

        function customFilter() {
            document.querySelectorAll('.inf-date-bar .quick-btn').forEach(b => b.classList.remove('active'));
            const start = document.getElementById('inf_startDate').value;
            const end = document.getElementById('inf_endDate').value;
            loadData(start, end);
        }

        function loadData(startDate, endDate) {
            const loading = document.getElementById('inf_loading');
            loading.style.display = 'block';
            ['inf_statsGrid', 'inf_chartsRow1', 'inf_chartsRow2', 'inf_chartsRow3'].forEach(id => {
                document.getElementById(id).style.display = 'none';
            });

            let url = '/api/ric/inference-stats?';
            if (startDate) url += 'start_date=' + startDate + '&';
            if (endDate) url += 'end_date=' + endDate;

            fetch(url)
                .then(r => r.json())
                .then(data => {
                    if (!data.success) { loading.textContent = '載入失敗: ' + (data.error || ''); return; }
                    loading.style.display = 'none';
                    _loaded = true;
                    renderCards(data.summary);
                    renderCharts(data);
                    renderCrossMatrix(data.cross_matrix);
                    ['inf_statsGrid', 'inf_chartsRow1', 'inf_chartsRow2', 'inf_chartsRow3'].forEach(id => {
                        document.getElementById(id).style.display = '';
                    });
                })
                .catch(err => { loading.textContent = '載入失敗: ' + err.message; });
        }

        function renderCards(s) {
            const total = s.total || 1;
            const pct = v => (v / total * 100).toFixed(1) + '%';
            document.getElementById('inf_statsGrid').innerHTML = `
                <div class="inf-stat-card">
                    <div class="label">總推論次數</div>
                    <div class="value" style="color:var(--text);">${s.total.toLocaleString()}</div>
                </div>
                <div class="inf-stat-card aoi-ng">
                    <div class="label">AOI 判 NG</div>
                    <div class="value">${s.aoi_ng.toLocaleString()}</div>
                    <div class="pct">${pct(s.aoi_ng)}</div>
                </div>
                <div class="inf-stat-card ai-ng">
                    <div class="label">AI 判 NG</div>
                    <div class="value">${s.ai_ng.toLocaleString()}</div>
                    <div class="pct">${pct(s.ai_ng)}</div>
                </div>
                <div class="inf-stat-card revival">
                    <div class="label">AI 救回 (AOI NG→AI OK)</div>
                    <div class="value">${s.ai_revival.toLocaleString()}</div>
                    <div class="pct">${s.aoi_ng ? (s.ai_revival / s.aoi_ng * 100).toFixed(1) + '% of AOI NG' : '-'}</div>
                </div>
                <div class="inf-stat-card err">
                    <div class="label">ERR 錯誤</div>
                    <div class="value">${s.err_count.toLocaleString()}</div>
                    <div class="pct">${pct(s.err_count)}</div>
                </div>`;
        }

        function renderCharts(data) {
            _destroyCharts();
            _renderTrendChart(data.daily_trend);
            _renderDistChart(data.summary);
            _renderMachineChart(data.by_machine);
            _renderModelChart(data.by_model);
            _renderErrChart(data.err_types);
        }

        function _renderTrendChart(daily) {
            const ctx = document.getElementById('inf_trendChart').getContext('2d');
            trendChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: daily.map(d => d.date),
                    datasets: [
                        {
                            label: '總推論數',
                            data: daily.map(d => d.total),
                            backgroundColor: 'rgba(100, 116, 139, 0.3)',
                            borderColor: 'rgba(100, 116, 139, 0.6)',
                            borderWidth: 1,
                            order: 2,
                        },
                        {
                            label: 'AOI NG',
                            data: daily.map(d => d.aoi_ng),
                            type: 'line',
                            borderColor: '#f97316',
                            backgroundColor: 'rgba(249, 115, 22, 0.1)',
                            pointRadius: 3,
                            tension: 0.3,
                            order: 1,
                        },
                        {
                            label: 'AI NG',
                            data: daily.map(d => d.ai_ng),
                            type: 'line',
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            pointRadius: 3,
                            tension: 0.3,
                            order: 1,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#94a3b8' } }, datalabels: { display: false } },
                    scales: {
                        x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(148,163,184,0.1)' } },
                        y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(148,163,184,0.1)' }, beginAtZero: true },
                    },
                },
            });
        }

        function _renderDistChart(s) {
            const ctx = document.getElementById('inf_distChart').getContext('2d');
            const aoiOk = s.total - s.aoi_ng - s.err_count;
            const aiOk = s.total - s.ai_ng - s.err_count;
            distChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['AOI OK', 'AOI NG', 'AI OK', 'AI NG', 'ERR'],
                    datasets: [
                        {
                            label: 'AOI',
                            data: [aoiOk, s.aoi_ng, 0, 0, 0],
                            backgroundColor: ['#22c55e', '#f97316', 'transparent', 'transparent', 'transparent'],
                            borderWidth: 0,
                        },
                        {
                            label: 'AI',
                            data: [0, 0, aiOk, s.ai_ng, s.err_count],
                            backgroundColor: ['transparent', 'transparent', '#22c55e', '#ef4444', '#f59e0b'],
                            borderWidth: 0,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { color: '#94a3b8', filter: item => item.text !== '' } },
                        datalabels: { display: false },
                    },
                },
            });
        }

        function _renderMachineChart(machines) {
            const top10 = machines.slice(0, 10);
            const ctx = document.getElementById('inf_machineChart').getContext('2d');
            machineChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: top10.map(m => m.machine),
                    datasets: [
                        { label: 'AOI NG率%', data: top10.map(m => m.aoi_ng_rate), backgroundColor: 'rgba(249,115,22,0.7)' },
                        { label: 'AI NG率%', data: top10.map(m => m.ai_ng_rate), backgroundColor: 'rgba(59,130,246,0.7)' },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: { legend: { labels: { color: '#94a3b8' } }, datalabels: { display: false } },
                    scales: {
                        x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(148,163,184,0.1)' }, beginAtZero: true },
                        y: { ticks: { color: '#94a3b8' }, grid: { display: false } },
                    },
                },
            });
        }

        function _renderModelChart(models) {
            const top10 = models.slice(0, 10);
            const ctx = document.getElementById('inf_modelChart').getContext('2d');
            modelChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: top10.map(m => m.model),
                    datasets: [{ label: '推論筆數', data: top10.map(m => m.total), backgroundColor: 'rgba(124,77,255,0.7)' }],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: { legend: { display: false }, datalabels: { display: false } },
                    scales: {
                        x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(148,163,184,0.1)' }, beginAtZero: true },
                        y: { ticks: { color: '#94a3b8' }, grid: { display: false } },
                    },
                },
            });
        }

        function _renderErrChart(errTypes) {
            const top10 = errTypes.slice(0, 10);
            const ctx = document.getElementById('inf_errChart').getContext('2d');
            errChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: top10.map(e => e.type.length > 20 ? e.type.substring(0, 20) + '…' : e.type),
                    datasets: [{ label: '次數', data: top10.map(e => e.count), backgroundColor: 'rgba(245,158,11,0.7)' }],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: { legend: { display: false }, datalabels: { display: false } },
                    scales: {
                        x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(148,163,184,0.1)' }, beginAtZero: true },
                        y: { ticks: { color: '#94a3b8' }, grid: { display: false } },
                    },
                },
            });
        }

        function renderCrossMatrix(cm) {
            document.getElementById('inf_crossMatrix').innerHTML = `
                <div class="inf-cross-cell">
                    <div class="cell-label">AOI OK → AI OK</div>
                    <div class="cell-value" style="color:#22c55e;">${(cm.ok_ok || 0).toLocaleString()}</div>
                    <div class="cell-desc" style="color:var(--text-muted);">一致 OK</div>
                </div>
                <div class="inf-cross-cell">
                    <div class="cell-label">AOI NG → AI OK</div>
                    <div class="cell-value" style="color:#22c55e;">${(cm.ng_ok || 0).toLocaleString()}</div>
                    <div class="cell-desc" style="color:#22c55e;">✓ AI 救回</div>
                </div>
                <div class="inf-cross-cell">
                    <div class="cell-label">AOI OK → AI NG</div>
                    <div class="cell-value" style="color:#ef4444;">${(cm.ok_ng || 0).toLocaleString()}</div>
                    <div class="cell-desc" style="color:#ef4444;">AI 額外攔截</div>
                </div>
                <div class="inf-cross-cell">
                    <div class="cell-label">AOI NG → AI NG</div>
                    <div class="cell-value" style="color:#f97316;">${(cm.ng_ng || 0).toLocaleString()}</div>
                    <div class="cell-desc" style="color:var(--text-muted);">一致 NG</div>
                </div>`;
        }

        return { _loaded, quickFilter, customFilter, loadData, renderCards, renderCharts, renderCrossMatrix };
    })();
```

- [ ] **Step 2: 瀏覽器完整測試**

在瀏覽器開啟 `http://localhost:8080/ric`，驗證：

1. 預設顯示 RIC Report Tab，現有功能不受影響
2. 點「AI 推論紀錄」Tab → 自動載入今日資料
3. 統計卡片顯示正確數字與百分比
4. 5 個圖表正常渲染（趨勢、分布、機台、型號、ERR）
5. 交叉比對矩陣顯示 4 格數字
6. 快捷按鈕切換（7天/30天/全部）→ 圖表刷新
7. 自訂日期範圍 + 查詢按鈕 → 圖表刷新
8. 切回 RIC Report Tab → 原有功能正常

- [ ] **Step 3: Commit**

```bash
git add templates/ric_report.html
git commit -m "feat: add inferenceTab JS module with charts and date filtering"
```

---

### Task 5: 整合測試與收尾

- [ ] **Step 1: 端到端驗證清單**

1. 啟動伺服器 `python capi_server.py --config server_config_local.yaml`
2. 開啟 `http://localhost:8080/ric`
3. 確認 RIC Report Tab 功能完全正常（上傳、圖表、明細）
4. 切換到 AI 推論紀錄 Tab
5. 驗證「今日」快捷按鈕（預設選中）
6. 切換 7天 / 30天 / 全部，觀察資料變化
7. 輸入自訂日期範圍，點查詢
8. 確認無 console error
9. 確認無資料時顯示合理（數字為 0，圖表空白）

- [ ] **Step 2: Final commit（如有任何修正）**

```bash
git add -A
git commit -m "fix: polish AI inference tab edge cases"
```
