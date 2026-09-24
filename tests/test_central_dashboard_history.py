"""中控看板「歷史班報」模式：index.html / app.js / styles.css 內容級檢查。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(name):
    return (ROOT / "central_dashboard" / name).read_text(encoding="utf-8")


def test_history_mode_tab_and_section_exist():
    index_html = _read("index.html")
    assert 'id="mode-tabs"' in index_html
    assert 'data-mode="realtime"' in index_html
    assert 'data-mode="history"' in index_html
    assert "歷史班報" in index_html
    assert 'id="history-section"' in index_html


def test_history_query_bar_elements():
    index_html = _read("index.html")
    assert 'id="history-date"' in index_html
    assert 'type="date"' in index_html
    assert 'data-shift="day"' in index_html
    assert 'data-shift="night"' in index_html
    # 顯示層用字：晚班
    assert "晚班" in index_html
    # 29 天限制提示
    assert "29 天" in index_html
    assert 'id="history-range"' in index_html


def test_history_result_table_columns():
    index_html = _read("index.html")
    assert 'id="history-overview"' in index_html
    assert "該班投入" in index_html


def test_app_js_history_mode_behaviors():
    app_js = _read("app.js")
    # 29 天上限
    assert "HISTORY_MAX_LOOKBACK_DAYS = 29" in app_js
    # 線體端歷史班報 API（baseUrl 自帶結尾斜線，原始碼不含前導斜線）
    assert "api/shift_report?date=" in app_js
    # 三種異常列文案
    assert "離線無法查詢" in app_js
    assert "未更新，請更新線體程式" in app_js
    assert "該班尚未結束" in app_js
    # 歷史模式包含停用線體（不過濾 enabled）
    assert "config.lines" in app_js


def test_history_mode_hides_realtime_blocks_via_css():
    styles = _read("styles.css")
    assert '[data-mode="history"]' in styles


def test_history_table_uses_realtime_overview_default_spacing():
    """歷史班報表沿用即時總覽的預設寬度與行距，不加緊湊化或限寬覆寫。"""
    styles = _read("styles.css")
    index_html = _read("index.html")
    assert ".history-table td" not in styles
    assert ".history-table .history-state-chip" not in styles
    assert ".history-table-wrap" not in styles
    assert "history-table-wrap" not in index_html


def test_history_keeps_process_tabs_and_filters_lines_by_zone():
    """歷史模式保留製程類別頁籤，結果表跟著 CAPI/AAPI 切換過濾線體。"""
    app_js = _read("app.js")
    styles = _read("styles.css")
    assert 'body[data-mode="history"] .process-tabs' not in styles
    assert "historyLinesForActiveZone" in app_js





def test_history_table_reuses_realtime_overview_visuals():
    """歷史班報表沿用即時總覽的配色與排版：AOI 琥珀、AI 青、投入等寬粗體。"""
    app_js = _read("app.js")
    # 歷史班報渲染列時需以獨立 class 形式引用即時總覽的配色
    # （即時總覽自身是 "overview-rate overview-rate-aoi" 合寫，不含獨立引號形式）
    assert app_js.count('"overview-rate-aoi"') >= 1
    assert app_js.count('"overview-rate-ai"') >= 1
    assert "overview-total" in app_js


def test_history_error_rows_use_realtime_offline_style_and_chips():
    """無法查詢的列比照即時總覽：離線列淡紅底＋狀態小圓章。"""
    app_js = _read("app.js")
    styles = _read("styles.css")
    assert "history-state-chip" in app_js
    assert ".history-state-chip" in styles
    assert 'data-tone="error"' in styles
    assert 'data-tone="warning"' in styles


def test_history_unavailable_row_chip_beside_line_name_with_placeholders():
    """離線／未更新列：訊息小圓章放線體名稱右邊，資料欄顯示「AOI — / AI — / —」。"""
    app_js = _read("app.js")
    # 即時總覽初始化已各出現一次；歷史班報的佔位格需再各出現一次
    assert app_js.count('"AOI —"') >= 2
    assert app_js.count('"AI —"') >= 2
    # 不再使用整行 colspan 訊息格
    assert "colSpan = 3" not in app_js


def test_history_probes_status_api_when_report_request_fails():
    """舊版線體的 404 回應未帶 CORS 標頭，瀏覽器讀不到狀態碼；
    歷史查詢失敗時需改探 /api/status 區分「未更新」與「離線」。"""
    app_js = _read("app.js")
    assert "probeLineReachable" in app_js
    # 探測目標是線體既有的 /api/status（line.apiUrl）
    assert "line.apiUrl" in app_js
