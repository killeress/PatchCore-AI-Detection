"""Exercise report filter events with Node, without browser dependencies."""
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const nodes = {};
for (const id of ['side-statistics', 'ss-filters', 'ss-message', 'ss-rows', 'ss-summary',
                  'ss-overview', 'ss-scope', 'ss-current-filter', 'ss-advanced', 'ss-more',
                  'ss-clear', 'ss-prev', 'ss-next', 'ss-page']) {
    nodes[id] = {innerHTML:'', textContent:'', hidden:true, listeners:{},
        addEventListener(type, fn){this.listeners[type] = fn;}, setAttribute(){}};
}
const form = nodes['ss-filters'];
const fields = ['start_date', 'end_date', 'machine_no', 'model_id', 'glass_id', 'status', 'review', 'tolerance'];
const template = fs.readFileSync('templates/_side_statistics.html', 'utf8');
form.elements = Object.fromEntries(fields.map(key => [key, {
    value:key === 'tolerance' ? '50' : '', defaultValue:key === 'tolerance' ? '50' : '',
    tagName:['status','review'].includes(key) ? 'SELECT' : 'INPUT',
}]));
for (const key of ['status', 'review']) {
    const select = template.match(new RegExp('<select name="' + key + '">([\\s\\S]*?)</select>'))[1];
    form.elements[key].options = [...select.matchAll(/<option value="([^"]*)">([^<]*)<\/option>/g)]
        .map(match => ({value:match[1], textContent:match[2]}));
}
form.reportValidity = () => true;
form.querySelector = () => ({disabled:false});
let dirty = false, saving = false, failNext = false, allowDiscard = false, confirmations = 0;
const requests = [];
nodes['ss-rows'].querySelector = selector =>
    ((selector.includes('data-dirty') && dirty) || (selector.includes('data-saving') && saving)) ? {} : null;
const summary = {
    with_csjbl: {total:12, candidates:10, clear:2, unreviewed:2, verified:10, confirmed:9, missed:1, accuracy:90},
    without_csjbl: {total:3, clear:3, unreviewed:1, verified:2, correct_clear:2, accuracy:100},
};
global.document = {getElementById:id => nodes[id]};
global.window = {confirm(){confirmations++; return allowDiscard;}, addEventListener(){}};
global.fetch = async url => {
    const query = new URLSearchParams(url.split('?')[1]);
    requests.push(query);
    if (failNext) { failNext = false; throw new Error('offline'); }
    return {ok:true, json:async () => ({total:query.get('review') ? 2 : 12, summary, rows_html:'loaded rows'})};
};
const flush = () => new Promise(resolve => setImmediate(resolve));
function clickFilter(field, value, group = 'with_csjbl') {
    const filter = {dataset:{filterGroup:group, filterField:field, filterValue:value}};
    nodes['side-statistics'].listeners.click({target:{closest:selector => selector === '[data-filter-field]' ? filter : null}});
}
function clearFilters() { nodes['ss-clear'].listeners.click(); }
'''


@pytest.mark.parametrize("scenario", [
    r'''
    // Cancelling a quick filter must not alter fields or turn the retry into a toggle-off.
    dirty = true;
    clickFilter('review', 'unreviewed'); await flush();
    assert.equal(requests.length, 1);
    assert.equal(confirmations, 1);
    assert.equal(form.elements.review.value, '');
    assert.equal(nodes['ss-clear'].hidden, true);
    dirty = false;
    clickFilter('review', 'unreviewed'); await flush();
    assert.equal(requests.at(-1).get('review'), 'unreviewed');
    assert.equal(form.elements.review.value, 'unreviewed');
    const appliedLabel = nodes['ss-current-filter'].textContent;
    dirty = true;
    clearFilters(); await flush();
    assert.equal(requests.length, 2);
    assert.equal(form.elements.review.value, 'unreviewed');
    assert.equal(nodes['ss-clear'].hidden, false);
    assert.equal(nodes['ss-current-filter'].textContent, appliedLabel);
    allowDiscard = true;
    clearFilters(); await flush();
    assert.equal(form.elements.review.value, '');
    assert.equal(requests.at(-1).get('review'), '');
    assert.equal(confirmations, 3); // One confirmation per attempted query.
    ''',
    r'''
    // Saving blocks navigation, and a failed request leaves the applied controls intact.
    saving = true;
    clickFilter('status', 'CANDIDATES'); await flush();
    assert.equal(requests.length, 1);
    assert.equal(form.elements.status.value, '');
    saving = false;
    clickFilter('status', 'CANDIDATES'); await flush();
    assert.equal(form.elements.status.value, 'CANDIDATES');
    saving = true;
    clearFilters(); await flush();
    assert.equal(requests.length, 2);
    assert.equal(form.elements.status.value, 'CANDIDATES');
    saving = false;
    failNext = true;
    clearFilters(); await flush();
    assert.equal(form.elements.status.value, 'CANDIDATES');
    assert.equal(nodes['ss-clear'].hidden, false);
    assert.match(nodes['ss-message'].textContent, /offline/);
    clearFilters(); await flush();
    assert.equal(form.elements.status.value, '');
    assert.equal(nodes['ss-clear'].hidden, true);
    ''',
    r'''
    // Quick filters, clearing and group switching reuse the applied scope, not draft edits.
    const overview = nodes['ss-overview'].innerHTML;
    const scope = nodes['ss-scope'].textContent;
    form.elements.machine_no.value = 'DRAFT_MACHINE';
    form.elements.start_date.value = '2020-01-01';
    clickFilter('review', 'confirmed,correct_clear'); await flush();
    assert.equal(requests.at(-1).get('machine_no'), '');
    assert.notEqual(requests.at(-1).get('start_date'), '2020-01-01');
    assert.equal(nodes['ss-overview'].innerHTML, overview);
    assert.equal(nodes['ss-scope'].textContent, scope);
    assert.match(nodes['ss-current-filter'].textContent, /人工確認正確/);
    assert.match(nodes['ss-page'].textContent, /共 2 筆/);
    clickFilter('review', 'confirmed,correct_clear', 'without_csjbl'); await flush();
    assert.equal(requests.at(-1).get('group'), 'without_csjbl');
    assert.equal(requests.at(-1).get('review'), 'confirmed,correct_clear');
    assert.equal(nodes['ss-overview'].innerHTML, overview);
    clearFilters(); await flush();
    assert.equal(requests.at(-1).get('machine_no'), '');
    assert.equal(form.elements.machine_no.value, 'DRAFT_MACHINE');
    form.listeners.submit({preventDefault(){}}); await flush();
    assert.equal(requests.at(-1).get('machine_no'), 'DRAFT_MACHINE');
    assert.equal(requests.at(-1).get('start_date'), '2020-01-01');
    assert.match(nodes['ss-scope'].textContent, /DRAFT_MACHINE/);
    ''',
], ids=["cancelled_filters", "saving_and_failed_requests", "applied_scope"])
def test_report_filter_interactions(scenario):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the report interaction check")
    source = (ROOT / "static/js/side_statistics.js").read_text(encoding="utf-8")
    checks = "(async () => { window.sideStatistics.init(); await flush();\n" + scenario + r'''
    })().catch(error => { console.error(error); process.exitCode = 1; });
    '''
    result = subprocess.run([node], input=HARNESS + source + checks, cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
