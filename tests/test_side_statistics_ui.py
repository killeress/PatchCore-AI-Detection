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
                  'ss-clear', 'ss-prev', 'ss-next', 'ss-page', 'ss-conversion']) {
    nodes[id] = {innerHTML:'', textContent:'', hidden:true, listeners:{},
        addEventListener(type, fn){this.listeners[type] = fn;}, setAttribute(){}};
}
const form = nodes['ss-filters'];
const fields = ['start_date', 'end_date', 'shift', 'machine_no', 'model_id', 'glass_id', 'status', 'conversion', 'tolerance'];
const template = fs.readFileSync('templates/_side_statistics.html', 'utf8');
form.elements = Object.fromEntries(fields.map(key => [key, {
    value:key === 'tolerance' ? '50' : '', defaultValue:key === 'tolerance' ? '50' : '',
    tagName:['status','conversion','shift'].includes(key) ? 'SELECT' : 'INPUT',
}]));
for (const key of ['status', 'conversion', 'shift']) {
    const select = template.match(new RegExp('<select name="' + key + '">([\\s\\S]*?)</select>'))[1];
    form.elements[key].options = [...select.matchAll(/<option value="([^"]*)">([^<]*)<\/option>/g)]
        .map(match => ({value:match[1], textContent:match[2]}));
}
form.reportValidity = () => true;
form.querySelector = () => ({disabled:false});
let failNext = false;
const requests = [];
const summary = {
    with_csjbl: {total:12, candidates:8, clear:2, unavailable:2, unreviewed:2, verified:10, confirmed:9, missed:1, accuracy:90},
    without_csjbl: {total:3, clear:3, unreviewed:1, verified:2, correct_clear:2, accuracy:100},
};
const projection = {
    denominator:102, total:102, ok:60, ok_s:2, ng:40, err:2, other:2, before_pct:4000/102,
    formal_impact: {converted:2, before_ng:42, after_ng:40, before_pct:41.176471,
        after_pct:39.215686, drop_pp:1.960784, relative_drop_pct:4.761905},
    scenarios: {
        mark:{converted:4, remaining_ng:36, after_pct:3600/102, drop_pp:400/102, relative_drop_pct:10},
        omit:{converted:3, remaining_ng:37, after_pct:3700/102, drop_pp:300/102, relative_drop_pct:7.5},
        tape:{converted:0, remaining_ng:40, after_pct:4000/102, drop_pp:0, relative_drop_pct:0},
        clear:{converted:0, remaining_ng:40, after_pct:4000/102, drop_pp:0, relative_drop_pct:0},
        combined:{converted:8, remaining_ng:32, after_pct:3200/102, drop_pp:800/102, relative_drop_pct:20},
    },
};
let conversionSummary = {
    mark:{points:7, records:5}, omit:{points:5, records:4}, tape:{points:2,records:1}, both_records:1, mixed_records:1, any_records:8,
    clear:{points:0,records:0}, converted_records:2,
    eligible:{mark:4,omit:3,tape:0,clear:0,mixed:1}, eligible_records:8, blocked_records:0, already_ok_records:0,
    scope:projection, csjbl:projection,
};
global.document = {getElementById:id => nodes[id]};
global.window = {};
let timeScope = {mode:'current', shift:'night', shift_name:'夜班', start_date:'2026-10-21', end_date:'2026-10-21',
    start:'2026-10-21 19:30:00', end:'2026-10-22 07:30:00', time_range:'2026-10-21 19:30 ~ 2026-10-22 07:30'};
global.fetch = async url => {
    const query = new URLSearchParams(url.split('?')[1]);
    requests.push(query);
    assert.equal(query.has('review'), false);
    if (failNext) { failNext = false; throw new Error('offline'); }
    return {ok:true, json:async () => ({total:query.get('status') ? 2 : 12, summary,
        conversion_summary:conversionSummary, time_scope:timeScope, rows_html:'loaded rows'})};
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
    // System-status filters toggle and clear without a review workflow.
    clickFilter('status', 'CANDIDATES'); await flush();
    assert.equal(requests.length, 2);
    assert.equal(form.elements.status.value, 'CANDIDATES');
    assert.equal(nodes['ss-clear'].hidden, false);
    assert.match(nodes['ss-current-filter'].textContent, /保留候選/);
    clickFilter('status', 'CANDIDATES'); await flush();
    assert.equal(form.elements.status.value, '');
    assert.equal(nodes['ss-clear'].hidden, true);
    clickFilter('status', 'NOT_RUN,NO_IMAGE,ERROR'); await flush();
    assert.equal(form.elements.status.value, 'NOT_RUN,NO_IMAGE,ERROR');
    clearFilters(); await flush();
    assert.equal(form.elements.status.value, '');
    assert.equal(requests.at(-1).get('status'), '');
    const visible = template + nodes['ss-overview'].innerHTML + nodes['ss-summary'].innerHTML
        + nodes['ss-conversion'].innerHTML;
    for (const label of ['人工驗證', '人工確認', '待驗證', '有效驗證', '正確率', '已處理', 'name="review"']) {
        assert.ok(!visible.includes(label), label);
    }
    ''',
    r'''
    // A failed request leaves the applied controls and loaded rows intact.
    clickFilter('status', 'CANDIDATES'); await flush();
    assert.equal(form.elements.status.value, 'CANDIDATES');
    const appliedLabel = nodes['ss-current-filter'].textContent;
    failNext = true;
    clearFilters(); await flush();
    assert.equal(form.elements.status.value, 'CANDIDATES');
    assert.equal(nodes['ss-clear'].hidden, false);
    assert.match(nodes['ss-message'].textContent, /offline/);
    assert.equal(nodes['ss-current-filter'].textContent, appliedLabel);
    assert.equal(nodes['ss-rows'].innerHTML, 'loaded rows');
    clearFilters(); await flush();
    assert.equal(form.elements.status.value, '');
    assert.equal(nodes['ss-clear'].hidden, true);
    ''',
    r'''
    // Only CSJBL contributes to the overview, even with an older API response.
    const overview = nodes['ss-overview'].innerHTML;
    assert.match(overview, /總筆數<b>12<\/b>/);
    assert.match(overview, /已完成檢測<b>10<\/b>/);
    assert.match(overview, /未完成檢測<b class="c-pending">2<\/b>/);
    assert.ok(!nodes['ss-summary'].innerHTML.includes('AOI 無 CSJBL'));
    const scope = nodes['ss-scope'].textContent;
    form.elements.shift.value = 'day'; form.listeners.change();
    form.elements.machine_no.value = 'DRAFT_MACHINE';
    form.elements.start_date.value = '2020-01-01';
    clickFilter('status', 'CANDIDATES'); await flush();
    assert.equal(requests.at(-1).get('machine_no'), '');
    assert.notEqual(requests.at(-1).get('start_date'), '2020-01-01');
    assert.equal(nodes['ss-overview'].innerHTML, overview);
    assert.equal(nodes['ss-scope'].textContent, scope);
    assert.match(nodes['ss-current-filter'].textContent, /保留候選/);
    assert.match(nodes['ss-page'].textContent, /共 2 筆/);
    const requestCount = requests.length;
    clickFilter('status', 'CANDIDATES', 'without_csjbl'); await flush();
    assert.equal(requests.length, requestCount);
    assert.equal(requests.at(-1).get('group'), 'with_csjbl');
    assert.equal(requests.at(-1).get('status'), 'CANDIDATES');
    assert.equal(nodes['ss-overview'].innerHTML, overview);
    clearFilters(); await flush();
    assert.equal(requests.at(-1).get('machine_no'), '');
    assert.equal(form.elements.machine_no.value, 'DRAFT_MACHINE');
    form.listeners.submit({preventDefault(){}}); await flush();
    assert.equal(requests.at(-1).get('machine_no'), 'DRAFT_MACHINE');
    assert.equal(requests.at(-1).get('start_date'), '2020-01-01');
    assert.match(nodes['ss-scope'].textContent, /DRAFT_MACHINE/);
    ''',
    r'''
    const table = nodes['ss-conversion'].innerHTML;
    assert.equal(nodes['ss-conversion'].hidden, false);
    assert.match(table, /側拍正式放行成效/);
    const headline = table.match(/<p class="ss-rate">([\s\S]*?)<\/p>/)[1];
    assert.match(headline, /未套用側拍放行<\/span><strong>41\.2%/);
    assert.match(headline, /目前正式<\/span><strong class="c-ok">39\.2%/);
    assert.match(headline, /下降 1\.96 個百分點 · 排片筆數相對減少 4\.76%/);
    assert.ok(!headline.includes('31.4%'));
    assert.match(table, /查詢範圍已正式轉 OK-s 2 筆/);
    assert.match(table, /正式已 OK（含 OK-s）/);
    assert.ok(!table.includes('原本已 OK'));
    assert.match(table, /歷史 NG 額外試算（尚未正式放行）/);
    assert.match(table, /39\.2%/);
    assert.match(table, /31\.4%/);
    assert.match(table, /7\.84 個百分點/);
    assert.match(table, /總投入 102 筆（OK 60、NG 40、ERR 2、其他 0）/);
    assert.match(table, /排片率＝NG ÷ 總投入（含 ERR）/);
    assert.match(table, /20\.00%/);
    assert.match(table, /40 − 8＝32/);
    assert.match(table, /混合類型 1 筆/);
    assert.match(table, /只轉易撕貼/);
    assert.match(table, /四類全部轉 OK/);
    clickFilter('conversion','tape'); await flush();
    assert.equal(requests.at(-1).get('conversion'),'tape');
    assert.match(nodes['ss-current-filter'].textContent,/含易撕貼/);
    clickFilter('conversion','eligible_tape'); await flush();
    assert.equal(requests.at(-1).get('conversion'),'eligible_tape');
    assert.match(nodes['ss-current-filter'].textContent,/僅靠易撕貼/);
    clickFilter('conversion','mark'); await flush();
    assert.equal(requests.at(-1).get('conversion'),'mark');
    assert.equal(form.elements.conversion.value,'mark');
    assert.match(nodes['ss-current-filter'].textContent,/含 MARK/);
    assert.equal(nodes['ss-clear'].hidden,false);
    assert.match(nodes['ss-conversion'].innerHTML,/40 − 8＝32/);
    clickFilter('conversion','omit'); await flush();
    assert.equal(form.elements.conversion.value,'omit');
    clearFilters(); await flush();
    assert.equal(requests.at(-1).get('conversion'),'');
    assert.equal(form.elements.conversion.value,'');
    conversionSummary = JSON.parse(JSON.stringify(conversionSummary));
    conversionSummary.scope.before_pct = null;
    conversionSummary.scope.denominator = 0;
    Object.assign(conversionSummary.scope.formal_impact, {
        converted:0, before_ng:0, after_ng:0, before_pct:null, after_pct:null, drop_pp:null, relative_drop_pct:null,
    });
    for (const r of Object.values(conversionSummary.scope.scenarios)) {
        r.after_pct = null; r.drop_pp = null; r.relative_drop_pct = null;
    }
    form.listeners.submit({preventDefault(){}}); await flush();
    assert.ok(!nodes['ss-conversion'].innerHTML.includes('NaN'));
    assert.ok(!nodes['ss-conversion'].innerHTML.includes('Infinity'));
    assert.match(nodes['ss-conversion'].innerHTML, /<strong>—<\/strong>/);
    ''',
    r'''
    // Live OK-s records already reduced NG; show their realized impact without a zero-only table.
    conversionSummary = JSON.parse(JSON.stringify(conversionSummary));
    conversionSummary.converted_records = 5;
    conversionSummary.eligible_records = 0;
    conversionSummary.already_ok_records = 5;
    conversionSummary.eligible = {mark:0, omit:0, tape:0, clear:0, mixed:0};
    const scope = conversionSummary.scope, subset = conversionSummary.csjbl;
    Object.assign(scope, {denominator:747, total:747, ok:495, ok_s:5, ng:140, err:112, other:112,
        formal_impact:{converted:5, before_ng:145, after_ng:140, before_pct:19.410977,
            after_pct:18.741633, drop_pp:.669344, relative_drop_pct:3.448276}});
    Object.assign(subset, {denominator:13, total:13, ok:5, ok_s:5, ng:8, err:0, other:0,
        formal_impact:{converted:5, before_ng:13, after_ng:8, before_pct:100,
            after_pct:61.538462, drop_pp:38.461538, relative_drop_pct:38.461538}});
    for (const p of [scope, subset]) {
        for (const scenario of Object.values(p.scenarios)) {
            Object.assign(scenario, {converted:0, remaining_ng:p.ng, drop_pp:0, relative_drop_pct:0});
        }
    }
    form.listeners.submit({preventDefault(){}}); await flush();
    const html = nodes['ss-conversion'].innerHTML;
    const headline = html.match(/<p class="ss-rate">([\s\S]*?)<\/p>/)[1];
    assert.match(headline, /未套用側拍放行<\/span><strong>19\.4%/);
    assert.match(headline, /目前正式<\/span><strong class="c-ok">18\.7%/);
    assert.match(headline, /下降 0\.67 個百分點 · 排片筆數相對減少 3\.45%/);
    assert.match(html, /已減少排片 5 筆/);
    assert.match(html, /100\.0% → 61\.5%，下降 38\.46 個百分點/);
    assert.match(html, /目前沒有可再整筆轉 OK 的 NG 紀錄/);
    assert.ok(!html.includes('<table'));
    clickFilter('conversion', 'mark'); await flush();
    assert.match(nodes['ss-conversion'].innerHTML, /<strong>19\.4%<\/strong>/);
    ''',
    r'''
    // Old NG-only records have no realized OK-s benefit, but retain the historical projection.
    conversionSummary = JSON.parse(JSON.stringify(conversionSummary));
    conversionSummary.converted_records = 0;
    for (const p of [conversionSummary.scope, conversionSummary.csjbl]) {
        p.ok_s = 0;
        Object.assign(p.formal_impact, {converted:0, before_ng:40, before_pct:39.215686,
            drop_pp:0, relative_drop_pct:0});
    }
    form.listeners.submit({preventDefault(){}}); await flush();
    const html = nodes['ss-conversion'].innerHTML;
    const headline = html.match(/<p class="ss-rate">([\s\S]*?)<\/p>/)[1];
    assert.equal(headline.match(/39\.2%/g).length, 2);
    assert.match(headline, /下降 0\.00 個百分點/);
    assert.match(html, /目前正式 39\.2% → 試算後 31\.4%/);
    assert.match(html, /另可減少排片 8 筆/);
    ''',
    r'''
    assert.equal(requests.at(-1).get('shift'), 'current');
    assert.equal(form.elements.start_date.disabled, true);
    assert.equal(form.elements.end_date.disabled, true);
    // Before 07:30, the server supplies the previous date's night shift.
    assert.equal(form.elements.start_date.value, '2026-10-21');
    assert.equal(form.elements.end_date.value, '2026-10-21');
    assert.match(nodes['ss-scope'].textContent, /當班 · 夜班 2026-10-21 19:30 ~ 2026-10-22 07:30/);
    form.elements.shift.value = 'night'; form.listeners.change();
    assert.equal(form.elements.start_date.disabled, false);
    form.elements.start_date.value = '2026-10-18'; form.elements.end_date.value = '2026-10-19';
    timeScope = {...timeScope, mode:'night', start_date:'2026-10-18', end_date:'2026-10-19',
        time_range:'2026-10-18 19:30 ~ 2026-10-20 07:30（僅夜班）'};
    form.listeners.submit({preventDefault(){}}); await flush();
    assert.equal(requests.at(-1).get('shift'), 'night');
    assert.equal(requests.at(-1).get('start_date'), '2026-10-18');
    assert.equal(requests.at(-1).get('end_date'), '2026-10-19');
    assert.match(nodes['ss-scope'].textContent, /僅夜班/);
    assert.ok(!nodes['ss-scope'].textContent.includes('當班'));
    clickFilter('conversion', 'eligible'); await flush();
    assert.equal(requests.at(-1).get('shift'), 'night');
    assert.equal(requests.at(-1).get('start_date'), '2026-10-18');
    ''',
    r'''
    // Dashboard locale rounding differs from toFixed at half-way values.
    conversionSummary = JSON.parse(JSON.stringify(conversionSummary));
    const s = conversionSummary.scope;
    Object.assign(s, {total:2000, denominator:2000, ng:3, ok:1997, ok_s:0, other:0, err:0,
        formal_impact:{converted:0, before_ng:3, after_ng:3, before_pct:.15, after_pct:.15,
            drop_pp:0, relative_drop_pct:0}});
    for (const scenario of Object.values(s.scenarios)) {
        Object.assign(scenario, {converted:0, remaining_ng:3, after_pct:.15, drop_pp:0, relative_drop_pct:0});
    }
    form.listeners.submit({preventDefault(){}}); await flush();
    assert.match(nodes['ss-conversion'].innerHTML, /<strong>0\.2%<\/strong>/);
    ''',
], ids=["system_status_filters", "failed_requests", "applied_scope", "conversion_projection",
        "formal_ok_s_impact", "historical_only", "shift_scope", "dashboard_rounding"])
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
