const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const okS = require('../central_dashboard/ok-s.js');
const source = fs.readFileSync(path.join(__dirname, '../central_dashboard/app.js'), 'utf8');

function loadDashboard() {
    const elements = new Map([
        ['summary-avg-aoi-rate', {}], ['summary-avg-ai-rate', {}],
    ]);
    const context = {
        window: {},
        document: {addEventListener() {}, getElementById: id => elements.get(id)},
    };
    vm.runInNewContext(source.replace(/\}\)\(\);\s*$/, `window.testApi = {
        normalizeStatus, renderSummaryAverages, lineStates,
        setProcessZone(zone) { activeProcessZone = zone; }
    };})();`), context);
    const app = context.window.testApi;
    function addLine(id, stats, overrides = {}) {
        const state = {processZone: 'capi', status: 'online',
            line: {id, line: id, isProduction: true}, data: app.normalizeStatus({stats}), ...overrides};
        app.lineStates.set(id, state);
        return state;
    }
    return {...app, addLine, aoi: elements.get('summary-avg-aoi-rate'), ai: elements.get('summary-avg-ai-rate')};
}

test('summary uses total NG / total input, matching OK-s for the same lines', () => {
    const app = loadDashboard();
    app.addLine('CAPI01', {total: 100, ng_count: 50, aoi_ng_count: 80});
    app.addLine('CAPI02', {total: 900, ng_count: 90, aoi_ng_count: 180});
    app.renderSummaryAverages();
    const sameScope = okS.summarize([
        {state: 'ok', counts: {total: 100, ng: 50, converted: 5}},
        {state: 'ok', counts: {total: 900, ng: 90, converted: 10}},
    ]);
    assert.equal(app.ai.textContent, `${sameScope.after.toFixed(2)}%`);
    assert.equal(app.ai.textContent, '14.00%'); // Per-line arithmetic mean would be 30%.
    assert.equal(app.aoi.textContent, '26.00%');
    assert.match(app.ai.title, /投入量加權/);
    assert.match(app.ai.title, /NG 合計 140 ÷ 投入合計 1,000/);
    assert.match(app.ai.title, /2 條線/);
});

test('summary retains the current process, production and online line scope', () => {
    const app = loadDashboard();
    app.addLine('CAPI01', {total: 100, ng_count: 10, aoi_ng_count: 20});
    app.addLine('AAPI01', {total: 100, ng_count: 80, aoi_ng_count: 90}, {processZone: 'aapi'});
    app.addLine('test-line', {total: 1000, ng_count: 1000}, {line: {id: 'test-line', isProduction: false}});
    for (const status of ['offline', 'warning', 'halted', 'checking']) {
        app.addLine(status, {total: 1000, ng_count: 1000}, {status});
    }
    app.addLine('no-data', {}, {data: null});
    app.renderSummaryAverages();
    assert.equal(app.ai.textContent, '10.00%');
    assert.equal(app.aoi.textContent, '20.00%');
    assert.match(app.ai.title, /1 條線/);
    app.setProcessZone('aapi');
    app.renderSummaryAverages();
    assert.equal(app.ai.textContent, '80.00%');
    assert.equal(app.aoi.textContent, '90.00%');
});

test('missing metric counts and zero input do not dilute valid rates; ERR remains in input', () => {
    const app = loadDashboard();
    app.addLine('empty', {total: 0, ng_count: 0, aoi_ng_count: 0});
    app.addLine('legacy', {total: 100, ng_count: 0});
    app.addLine('current', {total: 200, ok_count: 80, ng_count: 20, err_count: 100, aoi_ng_count: 100});
    app.renderSummaryAverages();
    assert.equal(app.ai.textContent, '6.67%');
    assert.equal(app.aoi.textContent, '50.00%');
    assert.match(app.ai.title, /2 條線/);
    assert.match(app.aoi.title, /1 條線/);
    assert.match(app.ai.title, /NG 合計 20 ÷ 投入合計 300/);
});

test('new responses update weighted rates and offline cached data is excluded', () => {
    const app = loadDashboard();
    const state = app.addLine('CAPI01', {total: 8339, ng_count: 1623});
    app.renderSummaryAverages();
    assert.equal(app.ai.textContent, '19.46%');
    assert.equal(app.aoi.textContent, '—');
    state.data = app.normalizeStatus({stats: {total: 100, ng_count: 0, aoi_ng_count: 0}});
    app.renderSummaryAverages();
    assert.equal(app.ai.textContent, '0.00%');
    assert.equal(app.aoi.textContent, '0.00%');
    state.status = 'offline';
    app.renderSummaryAverages();
    assert.equal(app.ai.textContent, '—');
    assert.match(app.ai.title, /無符合條件/);
    state.status = 'online';
    state.data = app.normalizeStatus({stats: {total: 0, ng_count: 0, aoi_ng_count: 0}});
    app.renderSummaryAverages();
    assert.equal(app.ai.textContent, '—');
    assert.equal(app.aoi.textContent, '—');
});
