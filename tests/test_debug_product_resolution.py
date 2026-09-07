import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import Environment, FileSystemLoader

from capi_config import CAPIConfig
from capi_web import CAPIWebHandler


@pytest.mark.parametrize("code,resolution", [
    ("B", [1366, 768]), ("H", [1920, 1080]), ("J", [1920, 1200]),
    ("K", [2560, 1440]), ("G", [2560, 1600]), ("j", [1920, 1200]),
    ("Z", [1920, 1080]), ("0", [1920, 1080]),
])
def test_debug_resolution_uses_model_folder_not_glass_or_filename(code, resolution):
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.inferencer = SimpleNamespace(config=CAPIConfig(machine_id="GN140BCAL010S"))
    info = handler._debug_product_resolution(
        rf"D:\images\GN140{code}CAL010S\TL6380GAL102\W0F00000_image.tif"
    )
    assert info["model_id"] == f"GN140{code}CAL010S"
    assert info["product_resolution"] == resolution
    assert bool(info["resolution_warning"]) == (code in ("Z", "0"))


@pytest.mark.parametrize("model_id,warning", [("GN140JCAL010S", False), ("ABC", True), ("", True)])
def test_debug_resolution_uses_loaded_machine_when_path_has_no_model(model_id, warning):
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.inferencer = SimpleNamespace(config=CAPIConfig(machine_id=model_id))
    info = handler._debug_product_resolution("D:/images/TL6380GAL102/ABCDEK.tif")
    assert info["model_id"] == model_id
    assert info["product_resolution"] == ([1920, 1080] if warning else [1920, 1200])
    assert bool(info["resolution_warning"]) == warning


def test_debug_resolution_honors_explicit_model_and_configured_map():
    config = CAPIConfig(machine_id="GN140BCAL010S", model_resolution_map={"J": [1000, 500]})
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.inferencer = SimpleNamespace(config=config)
    info = handler._debug_product_resolution("D:/GN140KCAL010S/image.tif", "GN140JCAL010S")
    assert info["product_resolution"] == [1000, 500]
    assert info["resolution_warning"] == ""


def test_debug_resolution_ui_updates_selection_and_clears_stale_warning():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for debug UI behavior test")
    template = (Path(__file__).resolve().parents[1] / "templates/debug_inference.html").read_text(encoding="utf-8")
    functions = template.split("    function debugResolutionInfo(", 1)[1].split("    async function runCoordInference", 1)[0]
    script = """
const assert = require('node:assert/strict');
const debugResolutionMap = {B:[1366,768], H:[1920,1080], J:[1920,1200], K:[2560,1440], G:[2560,1600]};
const debugMachineId = 'GN140HCAL010S';
const elements = {};
const document = {
    getElementById: id => elements[id],
    createElement: () => ({style: {}, setAttribute() {}}),
};
function Option(text, value) { this.text = text; this.value = value; }
for (const id of ['coord-resolution', 'edge-leak-resolution', 'bs-resolution', 'cv-product-code']) {
    elements[id] = {options: [], add(o) { this.options.push(o); }, insertAdjacentElement(_, n) { elements[n.id] = n; }};
}
""" + "function debugResolutionInfo(" + functions + """
for (const id of Object.keys(elements)) {
    for (const [code, resolution] of Object.entries(debugResolutionMap)) {
        autoSelectResolution(`D:/GN140${code}CAL010S/TL6380GAL102/W0F00000.tif`, id);
        assert.equal(elements[id].value, id === 'cv-product-code' ? code : resolution.join(','));
    }
    autoSelectResolution('D:/GN140ZCAL010S/W0F00000.tif', id);
    assert.equal(elements[id].value, id === 'cv-product-code' ? '' : '1920,1080');
    assert.match(elements[id + '-resolution-note'].textContent, /1920×1080/);
    assert.match(elements[id + '-resolution-note'].textContent, /預設/);
    autoSelectResolution('D:/GN140JCAL010S/W0F00000.tif', id);
    assert.doesNotMatch(elements[id + '-resolution-note'].textContent, /預設/);
    autoSelectResolution('D:/TL6380GAL102/ABCDEK.tif', id);
    assert.equal(elements[id].value, id === 'cv-product-code' ? 'H' : '1920,1080');
}
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr


def test_debug_page_renders_configured_resolution_map_and_valid_javascript():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for debug UI syntax test")
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.inferencer = SimpleNamespace(config=CAPIConfig(machine_id="GN140JCAL010S"))
    handler.db = None
    handler.jinja_env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parents[1] / "templates"))
    handler.jinja_env.globals["app_version"] = {"version": "test"}
    rendered = []
    handler._send_response = lambda status, html: rendered.append(html)
    handler._handle_debug_page("/debug")
    assert 'const debugMachineId = "GN140JCAL010S";' in rendered[0]
    for attrs, script in re.findall(r"<script([^>]*)>(.*?)</script>", rendered[0], re.S):
        mode = "module" if 'type="module"' in attrs else "commonjs"
        result = subprocess.run([node, "--check", f"--input-type={mode}"], input=script, capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0, result.stderr
