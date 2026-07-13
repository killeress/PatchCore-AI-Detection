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
    assert html.index(inference_tab) < html.index(ric_tab)
    assert '<div class="top-tab-content active" id="tab-inference">' in html
    assert '<div class="top-tab-content" id="tab-ric">' in html
    assert "inferenceTab.quickFilter('today');" in html
