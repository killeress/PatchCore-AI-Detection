"""中控看板「歷史趨勢」：index.html / app.js / styles.css / 打包清單的內容級檢查。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(name):
    return (ROOT / "central_dashboard" / name).read_text(encoding="utf-8")


def test_trend_section_and_controls_exist():
    index_html = _read("index.html")
    # 趨勢區塊位於歷史班報模式內
    history_block = index_html.split('id="history-section"', 1)[1]
    assert 'id="trend-section"' in history_block
    assert 'id="trend-from"' in index_html
    assert 'id="trend-to"' in index_html
    # 班別過濾：白班 / 晚班 / 全部（無「依日期」粒度按鈕）
    assert 'data-shift-filter="all"' in index_html
    assert 'data-shift-filter="day"' in index_html
    assert 'data-shift-filter="night"' in index_html
    assert "data-granularity" not in index_html
    # 指標切換：AOI 排片率 / AI 排片率 / 當班投入量
    assert 'data-metric="aoi"' in index_html
    assert 'data-metric="ai"' in index_html
    assert 'data-metric="total"' in index_html
    assert "當班投入量" in index_html
    # 線體勾選、畫布、無資料提示、異常線體清單
    assert 'id="trend-lines"' in index_html
    assert 'id="trend-canvas"' in index_html
    assert 'id="trend-empty"' in index_html
    assert 'id="trend-issues"' in index_html


def test_trend_uses_chartjs_bundled_locally():
    """Chart.js 與 /ric 同款，從本地路徑載入（不走 CDN）。"""
    index_html = _read("index.html")
    assert "chart.umd.min.js" in index_html
    assert "cdn" not in index_html.split("chart.umd.min.js")[0][-200:].lower()
    # 庫檔已複製進 central_dashboard 資料夾（備援模式可用）
    assert (ROOT / "central_dashboard" / "chart.umd.min.js").is_file()


def test_chartjs_file_included_in_deploy_package():
    build = (ROOT / "scripts" / "build_deploy_zip.py").read_text(encoding="utf-8")
    assert "central_dashboard/chart.umd.min.js" in build


def test_app_js_trend_behaviors():
    app_js = _read("app.js")
    # 線體端批次序列 API
    assert "api/shift_report/series?from=" in app_js
    # Chart.js 渲染
    assert "new Chart(" in app_js
    # 主要函式
    assert "initializeTrend" in app_js
    assert "refreshTrend" in app_js
    assert "renderTrendChart" in app_js
    # 班別過濾（白班/晚班/全部）；粒度固定依班別，不做每小時
    assert "shiftFilter" in app_js
    assert "granularity" not in app_js
    # 歷史資料為靜態內容：關閉動畫
    assert "animation: false" in app_js


def test_trend_renders_on_first_ok_line_and_at_completion():
    """首條成功線體回覆即出圖（不被離線線體的逾時拖住）；
    離線／未更新線體不觸發重繪；全部完成後再最終渲染一次定稿。"""
    app_js = _read("app.js")
    refresh_body = app_js.split("async function refreshTrend", 1)[1]
    refresh_body = refresh_body.split("function renderTrendIssues", 1)[0]
    # 一次在單線成功回覆時（entry.status === "ok"），一次在 Promise.all 之後
    assert refresh_body.count("renderTrendChart()") == 2
    assert 'entry.status === "ok"' in refresh_body
    assert "Promise.all" in refresh_body


def test_trend_line_colors_generated_without_fixed_palette_limit():
    """線體顏色不再受限於固定 10 色：用黃金角 HSL 依序產生，線數無上限。"""
    app_js = _read("app.js")
    assert "trendLineColor" in app_js
    # 黃金角色相遞增（約 137.5°），保證相鄰線體顏色有明顯差異
    assert "137.508" in app_js
    assert "TREND_COLORS" not in app_js
    # 異常線體沿用班報文案
    assert "未更新，請更新線體程式" in app_js
    assert "離線無法查詢" in app_js


def test_trend_styles_exist():
    styles = _read("styles.css")
    assert ".trend-lines" in styles
    assert ".trend-chart-wrap" in styles
