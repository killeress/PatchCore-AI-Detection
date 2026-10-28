"""Execute the rendered selection script to verify rejection/replacement behavior."""
import re
import shutil
import subprocess
from pathlib import Path

from jinja2 import Environment
import pytest


def test_selection_waits_for_check_and_requires_new_panel_id():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for selection UI behavior tests")
    source = (Path(__file__).resolve().parents[1] / "templates/train_new/step1_select.html").read_text(encoding="utf-8")
    script = re.search(r"<script>(.*?)</script>", source, re.S).group(1)
    script = Environment().from_string(script).render(
        training_scope={"mode": "full", "selected_units": []}, target_bundle=None,
        selected_lightings=["W0F00000"], preprocess_methods=[],
        default_preprocess_pipeline=[], target_patchcore_params={},
    )
    dom = r"""
const assert = require('node:assert/strict');
class Element {
  constructor() {
    this.style = {}; this.dataset = {}; this.children = []; this.value = '';
    this.checked = false; this.disabled = false; this.hidden = true;
    this.classList = {toggle: () => {}, add: () => {}, remove: () => {}};
  }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  scrollIntoView() {}
  querySelector(name) { return name === '.chk' ? this.check : null; }
  querySelectorAll() { return this.zones || []; }
}
const elements = new Map();
const rows = ['GOOD', 'BAD', 'NEW'].map(id => {
  const row = new Element();
  row.dataset = {path: 'C:/batch/' + id, machineId: 'M'};
  row.check = new Element(); row.zones = [new Element(), new Element()];
  return row;
});
global.document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  },
  createElement: () => new Element(),
  addEventListener: () => {},
  querySelectorAll: selector => selector === '#panel-list tr[data-path]' ? rows : [],
  querySelector(selector) {
    if (selector === '.source-card') return this.getElementById('source-card');
    return rows.find(row => selector.includes('"' + row.dataset.path + '"'));
  },
};
global.CSS = {escape: value => value};
global.window = {location: {href: ''}};
global.capAlert = () => {};
"""
    behavior = r"""
collectTrainingParams = () => ({params: null, error: null});
collectImagePreprocessPipelines = () => ({pipeline: [], error: null});
collectGridCanonicalization = () => ({config: {enabled: false}, error: null});
(async () => {
  toggleSelect(rows[0]); toggleSelect(rows[1]);
  assert.equal(_selected.size, 2);
  let finish;
  let calls = 0;
  global.fetch = () => { calls++; return new Promise(resolve => { finish = resolve; }); };
  const pending = submitStart();
  assert.equal(_panelPrecheckPending, true);
  assert.equal(document.getElementById('next-btn').disabled, true);
  assert.equal(document.querySelector('.source-card').inert, true);
  assert.equal(window.location.href, '');
  await submitStart();
  toggleSelect(rows[2]);
  assert.equal(calls, 1);  // No duplicate request or selection changes while checking.
  assert.equal(_selected.size, 2);
  finish({ok: false, json: async () => ({
    code: 'training_panel_image_abnormal', error: 'reselect',
    rejected_panels: [{panel_id: 'BAD', panel_path: 'C:/batch/BAD', issues: [{
      screen: 'W0F00000', image_name: 'bad.png', image_path: 'C:/batch/BAD/bad.png',
      reason: 'Mean 0, range 70–90',
    }]}],
  })});
  await pending;
  assert.equal(_selected.has('C:/batch/BAD'), false);
  assert.equal(_selected.has('C:/batch/GOOD'), true);
  assert.equal(_requiredPanelCount, 2);
  assert.equal(document.getElementById('next-btn').disabled, true);
  assert.equal(document.getElementById('panel-precheck-errors').hidden, false);
  assert.equal(document.querySelector('.source-card').inert, false);
  assert.equal(window.location.href, '');
  toggleSelect(rows[1]);  // Rejected Panel ID stays blocked.
  assert.equal(_selected.size, 1);
  await submitStart();  // Dropping the failed panel is insufficient.
  assert.equal(calls, 1);
  toggleSelect(rows[2]);
  assert.equal(_selected.size, 2);
  assert.equal(document.getElementById('next-btn').disabled, false);
  global.fetch = async (_, options) => {
    const body = JSON.parse(options.body);
    assert.deepEqual(body.panel_paths, ['C:/batch/GOOD', 'C:/batch/NEW']);
    return {ok: true, json: async () => ({job_id: 'passed'})};
  };
  await submitStart();
  assert.equal(window.location.href, '/train/new/progress?job_id=passed');
  window.location.href = '';
  global.fetch = async () => { throw new Error('offline'); };
  await submitStart();
  assert.equal(_panelPrecheckPending, false);
  assert.equal(document.getElementById('next-btn').disabled, false);
  assert.equal(window.location.href, '');
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run([node], input=dom + script + behavior, text=True,
                            encoding="utf-8", capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
