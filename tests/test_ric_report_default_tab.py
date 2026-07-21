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
    assert "inferenceTab.quickFilter('today');" in html
