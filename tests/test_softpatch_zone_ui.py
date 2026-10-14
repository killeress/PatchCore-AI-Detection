"""Exercise rendered zone controls and their actual layout-switching JavaScript."""

import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader


ROOT = Path(__file__).parents[1]


class ElementTree(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.root = {"tag": "root", "attrs": {}, "children": []}
        self.stack = [self.root]
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        node = {"tag": tag, "attrs": dict(attrs), "children": []}
        self.stack[-1]["children"].append(node)
        if tag not in {"input", "br", "hr"}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        if self.stack[-1]["tag"] == tag:
            self.stack.pop()


def run_node(source, data):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for form behavior checks")
    return subprocess.run([node, "-e", source], input=json.dumps(data),
                          text=True, encoding="utf-8", capture_output=True, check=True).stdout


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_zone_layout_preserves_edits_and_inherited_recipes(partial, legacy):
    template = (ROOT / "templates/train_new/step1_select.html").read_text(encoding="utf-8")
    env = Environment(loader=FileSystemLoader(ROOT / "templates"))
    env.get_template("train_new/step1_select.html")  # Also compile the complete template.
    macro = template[template.index("{% macro softpatch_fields"):template.index("{% endmacro %}") + len("{% endmacro %}")]
    fragment = template[template.index('{% set inherited_cleaning ='):template.index('<div class="tile-settings-row">')]
    params = {"feature_cleaning_mode": "softpatch_plus_v1", "feature_cleaning_scope": "inner_and_edge",
              "softpatch_plus_config": {"context_overlap": False, "weight_strength": 1.5}}
    if partial:
        params["feature_cleaning_by_zone"] = {
            "inner": {"mode": "softpatch_plus_v1"}, "edge": {"mode": "softpatch_plus_v1"},
        }
        if not legacy:
            params["feature_cleaning_by_zone"]["edge"]["softpatch_plus_config"] = {"context_overlap": True}
    html = env.from_string(macro + fragment).render(target_patchcore_params=params,
                                                   training_scope={"mode": "partial" if partial else "full"})
    source = template[template.index("function isSeparateFeatureCleaning()"):template.index("const TRAIN_PARAM_KEYS =")]
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
const ids = {};
function build(raw) {
  const children = raw.children.map(build);
  const node = {tag: raw.tag, attrs: raw.attrs, children, style: {}, hidden: false,
    value: raw.attrs.value || '', disabled: 'disabled' in raw.attrs,
    querySelectorAll(selector) {
      const matches = n => selector === '.fc-shared-field'
        ? (n.attrs.class || '').split(' ').includes('fc-shared-field')
        : selector.split(',').includes(n.tag);
      return children.flatMap(c => [...(matches(c) ? [c] : []), ...c.querySelectorAll(selector)]);
    }};
  if (node.tag === 'select') node.value = (children.find(c => 'selected' in c.attrs) || children[0]).attrs.value;
  if (raw.attrs.id) { assert.ok(!ids[raw.attrs.id], 'unique input IDs'); ids[raw.attrs.id] = node; }
  return node;
}
const root = build(data.tree);
ids['softpatch-parameter-help'] = {};
const context = {document: {getElementById: id => ids[id], querySelectorAll: s => root.querySelectorAll(s)},
                 isPartialTraining: () => data.partial};
vm.createContext(context);
vm.runInContext('let _zoneCleaningInitialized = ' + data.partial + ';' + data.source, context);
const sync = () => vm.runInContext('syncFeatureCleaningLayout()', context);
const field = id => ids[id];
if (data.partial) {
  sync();
  assert.equal(field('sp-inner-context_overlap').value, 'false');
  assert.equal(field('sp-edge-context_overlap').value, data.legacy ? 'false' : 'true');
  for (const prefix of ['sp-inner-', 'sp-edge-', 'sp-']) {
    assert.ok(field(prefix + 'context_overlap').disabled);
  }
} else {
  field('tp-feature_cleaning_mode').value = 'softpatch_plus_v1';
  field('tp-feature_cleaning_scope').value = 'inner_and_edge';
  sync();
  assert.equal(field('softpatch-settings').style.display, 'grid');
  field('tp-feature_cleaning_layout').value = 'separate'; sync();
  assert.equal(field('sp-edge-weight_strength').value, '1.5');
  assert.equal(field('sp-edge-context_overlap').value, 'false');
  field('sp-edge-context_overlap').value = 'true';
  field('tp-feature_cleaning_layout').value = 'shared'; sync();
  field('sp-weight_strength').value = '3';
  field('tp-feature_cleaning_layout').value = 'separate'; sync();
  assert.equal(field('sp-inner-context_overlap').value, 'false');
  assert.equal(field('sp-edge-context_overlap').value, 'true');
  assert.equal(field('sp-edge-weight_strength').value, '1.5');
  assert.equal(field('softpatch-settings').style.display, 'none');
  assert.equal(field('softpatch-settings-edge').style.display, 'grid');
  assert.equal(field('sp-edge-context_overlap').disabled, false);
  field('tp-feature_cleaning_edge_mode').value = 'off'; sync();
  assert.equal(field('softpatch-settings-edge').style.display, 'none');
  assert.ok(field('sp-edge-context_overlap').disabled);
  assert.equal(field('sp-inner-context_overlap').disabled, false);
}
"""
    run_node(script, {"tree": ElementTree(html).root, "source": source, "partial": partial, "legacy": legacy})


@pytest.mark.parametrize("legacy", [False, True])
def test_model_list_and_completion_show_effective_zone_options(legacy):
    params = {"softpatch_plus_config": {"context_overlap": False}, "feature_cleaning_by_zone": {
        "inner": {"mode": "softpatch_plus_v1"}, "edge": {"mode": "softpatch_plus_v1"},
    }}
    if not legacy:
        params["feature_cleaning_by_zone"]["edge"]["softpatch_plus_config"] = {"context_overlap": True}
    template = (ROOT / "templates/models.html").read_text(encoding="utf-8")
    source = template[template.index("function formatFeatureCleaning("):template.index("function formatPreprocessPipeline(")]
    output = run_node("const fs=require('fs'); const data=JSON.parse(fs.readFileSync(0,'utf8'));"
                      + source + "process.stdout.write(formatFeatureCleaning(data));", params)
    inner, edge = output.split("；EDGE:")
    assert "重疊保護關" in inner
    assert ("重疊保護關" if legacy else "重疊保護開") in edge
    template = (ROOT / "templates/train_new/step5_done.html").read_text(encoding="utf-8")
    source = template[template.index('          {% for zone in (["inner", "edge"] if cleaning_zones'):template.index('          <dt>清洗有效區域</dt>')]
    env = Environment(loader=FileSystemLoader(ROOT / "templates"))
    env.get_template("train_new/step5_done.html")
    output = env.from_string(source).render(patchcore_params=params, cleaning_zones=params["feature_cleaning_by_zone"])
    inner, edge = output.split("EDGE SoftPatch+")
    assert "重疊保護關" in inner
    assert ("重疊保護關" if legacy else "重疊保護開") in edge
