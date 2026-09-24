"""中控看板即時模式：AAPI 上線且正常線體的平均排片率摘要卡（內容級檢查）。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(name):
    return (ROOT / "central_dashboard" / name).read_text(encoding="utf-8")


def test_summary_average_cards_exist():
    index_html = _read("index.html")
    assert 'id="summary-avg-aoi-rate"' in index_html
    assert 'id="summary-avg-ai-rate"' in index_html
    # 卡名固定，不加製程類別前綴
    assert "平均 AOI 排片率" in index_html
    assert "平均 AI 排片率" in index_html


def test_summary_average_logic_filters_active_zone_production_online():
    """納入目前製程類別（CAPI/AAPI 頁籤連動）、上線且狀態正常的線體，算術平均。"""
    app_js = _read("app.js")
    assert "renderSummaryAverages" in app_js
    # 既有 updateSummary 已出現一次；平均卡邏輯需再出現一次
    assert app_js.count("state.processZone === activeProcessZone") >= 2
    assert "state.line.isProduction === true" in app_js
    # 卡名固定不動態更新
    assert "summary-avg-aoi-label" not in app_js


def test_summary_averages_update_as_each_line_responds():
    """每條線回覆後立刻重算平均，不等整輪（避免被離線線體的逾時拖慢）。"""
    app_js = _read("app.js")
    # 定義 + updateSummary 呼叫 + refreshLine 每線回覆時呼叫，至少三處
    assert app_js.count("renderSummaryAverages()") >= 3


def test_process_tabs_hidden_when_single_zone():
    """製程類別頁籤維持既有行為：只有單一製程類別時隱藏。"""
    index_html = _read("index.html")
    app_js = _read("app.js")
    process_tabs_line = next(
        line for line in index_html.splitlines() if 'id="process-tabs"' in line
    )
    assert "hidden" in process_tabs_line
    assert "availableZones.size < 2" in app_js


def test_summary_average_cards_have_distinct_accent_colors():
    """比照既有摘要卡的上緣色條：AOI 琥珀、AI 青。"""
    styles = _read("styles.css")
    assert ".summary-avg-aoi" in styles
    assert ".summary-avg-ai" in styles
