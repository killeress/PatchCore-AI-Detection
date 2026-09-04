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
