const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const errors = require('../central_dashboard/inference-errors.js');
const source = fs.readFileSync(path.join(__dirname, '../central_dashboard/app.js'), 'utf8');
const failure = "ERR:INFERENCE_FAILED (RuntimeError: [v2] U0F00000/inner tile(1608,1080) 推論失敗: 'U0F00000')";

function loadDashboard() {
    function node() {
        const selectors = new Map();
        return {children: [], dataset: {}, classList: {toggle() {}}, removeAttribute() {},
            querySelector(selector) {
                if (!selectors.has(selector)) selectors.set(selector, node());
                return selectors.get(selector);
            },
            replaceChildren() { this.children = []; },
            appendChild(child) { this.children.push(child); }};
    }
    const elements = {'alert-panel': node(), 'alert-list': node()};
    const context = {
        window: {CAPIInferenceErrors: errors},
        document: {addEventListener() {}, createElement: node, getElementById: id => elements[id]},
    };
    vm.runInNewContext(source.replace(/\}\)\(\);\s*$/, 'window.testApi = {normalizeStatus, getLineAlerts, renderAlerts, renderOverviewRow, lineStates};})();'), context);
    return {...context.window.testApi, elements, node};
}

test('central alerts use the full detail even when latest judgment is only ERR', () => {
    const app = loadDashboard();
    const data = app.normalizeStatus({latest_event: {
        judgment: 'ERR', detail: failure, machine_no: 'CAPI42', model_id: 'MODEL-A', glass_id: 'GLASS1', time: '12:56:13',
    }});
    const alerts = app.getLineAlerts(data);
    assert.equal(alerts.length, 1);
    assert.equal(alerts[0].severity, 'critical');
    assert.equal(alerts[0].summary, '需重新訓練模型（U0F00000）');
    assert.match(alerts[0].message, /CAPI42.*MODEL-A.*GLASS1.*12:56:13/);
    assert.match(alerts[0].message, /重新訓練模型.*啟用更新後的模型套件/);
    app.lineStates.set('line', {processZone: 'capi', status: 'online', data, line: {line: 'CAPI42', factory: 'MOD1'}});
    app.renderAlerts();
    assert.equal(app.elements['alert-panel'].hidden, false);
    assert.match(app.elements['alert-list'].children[0].textContent, /MOD1 \/ CAPI42.*缺少畫面模型/);
});

test('latest success clears the notice and offline cached errors stay hidden', () => {
    const app = loadDashboard();
    const state = {processZone: 'capi', status: 'online', line: {line: 'CAPI42'},
        data: app.normalizeStatus({latest_event: {judgment: 'ERR', detail: failure}})};
    app.lineStates.set('line', state);
    app.renderAlerts();
    assert.equal(app.elements['alert-panel'].hidden, false);
    state.data = app.normalizeStatus({latest_event: {judgment: 'OK', detail: 'OK'}});
    app.renderAlerts();
    assert.equal(app.elements['alert-panel'].hidden, true);
    state.status = 'offline';
    state.data = app.normalizeStatus({latest_event: {detail: failure}});
    app.renderAlerts();
    assert.equal(app.elements['alert-list'].children.length, 1);
    assert.doesNotMatch(app.elements['alert-list'].children[0].textContent, /訓練|U0F00000/);
});

test('older empty payloads and unrelated failures do not invent training alerts', () => {
    const app = loadDashboard();
    for (const raw of [{}, {latest_event: {judgment: 'ERR'}}, {latest_event: {detail: 'CUDA out of memory'}}]) {
        assert.equal(app.getLineAlerts(app.normalizeStatus(raw)).length, 0);
    }
    const data = app.normalizeStatus({latest_event: {detail: failure}, hardware: {memory: {used_percent: 96}}});
    assert.equal(app.getLineAlerts(data).length, 2);
    assert.match(app.getLineAlerts(data)[1].summary, /RAM/);
});

test('overview anomaly cell renders the alert and removes it after recovery', () => {
    const app = loadDashboard();
    const row = app.node();
    const state = {status: 'online', line: {line: 'CAPI42'}, overviewRow: row,
        data: app.normalizeStatus({latest_event: {judgment: 'ERR', detail: failure}})};
    app.renderOverviewRow(state);
    const badges = row.querySelector('[data-field="overview-alerts"]');
    assert.equal(badges.children.length, 1);
    assert.equal(badges.children[0].textContent, '⚠ 需重新訓練模型（U0F00000）');
    assert.match(badges.children[0].title, /正常圖片重新訓練/);
    assert.equal(row.dataset.health, 'critical');
    state.status = 'offline';
    app.renderOverviewRow(state);
    assert.equal(badges.children.length, 0);
    state.status = 'online';
    state.data = app.normalizeStatus({latest_event: {judgment: 'OK'}});
    app.renderOverviewRow(state);
    assert.equal(badges.children.length, 0);
    assert.equal(row.dataset.health, 'normal');
});
