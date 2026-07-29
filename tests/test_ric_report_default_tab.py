from pathlib import Path
from types import SimpleNamespace

from jinja2 import Environment, FileSystemLoader


def test_ai_inference_records_is_the_default_report_tab():
    env = Environment(loader=FileSystemLoader("templates"))
    env.globals.update(
        app_version=SimpleNamespace(version="test"),
        host_identity="",
    )
    template = env.get_template("ric_report.html")

    html = template.render(request_path="/ric", batches=[])

    inference_tab = '<div class="top-tab active" data-top-tab="inference"'
    ric_tab = '<div class="top-tab" data-top-tab="ric"'
    mes_tab = '<div class="top-tab" data-top-tab="mes"'
    assert html.index(inference_tab) < html.index(ric_tab)
    assert html.index(ric_tab) < html.index(mes_tab)
    assert '<div class="top-tab-content active" id="tab-inference">' in html
    assert '<div class="top-tab-content" id="tab-ric">' in html
    assert '<div class="top-tab-content" id="tab-mes">' in html
    assert 'id="mes_exportBtn"' in html
    assert "mesReportTab.exportCSV()" in html
    assert "mesReportTab.toggleFilter('" in html
    assert "return _data.records.filter(row => row.comparison !== 'uncomparable');" in html
    assert "mes_report_comparison_${start}_${end}_${_activeFilter || 'all'}.csv" in html
    assert "正在查詢 MES Oracle 並比對 Report，已等待 ${seconds} 秒" in html
    assert "setInterval(updateWaitingText, 1000)" in html
    assert "耗時：${elapsedSeconds} 秒" in html
    assert 'id="mesReportDetailModal"' in html
    assert 'MES 完整數據' in html
    assert "/api/ric/mes-report-detail?record_id=" in html
    assert "columns.map(column =>" in html
    assert "mesReportTab.openDetail(" in html
    assert 'id="mes_ignoreAoiOk"' in html
    assert 'id="mes_panelId"' in html
    assert "params.set('panel_id', panelId)" in html
    assert "PANEL ID（完整或部分）" in html
    assert "mesReportTab.toggleIgnoreAoiOk(this.checked)" in html
    assert "let _ignoreAoiOk = false;" in html
    assert "params.set('ignore_aoi_ok', '1')" in html
    assert "已忽略 AOI=OK" in html
    assert "Array.isArray(row.qualifying_defects)" in html
    assert "function defectCatalogText(item)" in html
    assert "function defectHtml(item)" in html
    assert "mes-defect-unmapped" in html
    assert ".join('\\n')" in html
    assert "inferenceTab.quickFilter('today');" in html


def test_ng_validation_shortcut_query_opens_existing_database_modal():
    env = Environment(loader=FileSystemLoader("templates"))
    env.globals.update(
        app_version=SimpleNamespace(version="test"),
        host_identity="",
    )
    template = env.get_template("ric_report.html")

    html = template.render(request_path="/ric", batches=[])

    assert "initialParams.get('open_ng_validation') === '1'" in html
    assert "switchTopTab('mes');" in html
    assert "mesReportTab.openNgDatabase();" in html


def test_models_page_has_ng_validation_shortcut():
    html = (Path(__file__).resolve().parent.parent / "templates" / "models.html").read_text(
        encoding="utf-8"
    )

    assert 'id="modelNgValidationShortcut"' in html
    assert 'href="/ric?open_ng_validation=1"' in html
    assert "查看 NG 驗證庫" in html
