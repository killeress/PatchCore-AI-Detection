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

function ramChange(previous, current) {
    return {active: true, available: true, latest_change: {
        hostname: 'CAPI02', previous_total_gb: previous, new_total_gb: current,
        last_seen_at: '2026-11-10T10:00:30+08:00', detected_at: '2026-11-10T10:01:00+08:00',
    }};
}

test('RAM decrease appears in anomaly summary and includes the investigation interval', () => {
    const app = loadDashboard();
    const data = app.normalizeStatus({ram_capacity_alert: ramChange(62.3, 30.3),
        hardware: {memory: {used_gb: 6.1, total_gb: 30.3, used_percent: 20}}});
    const alerts = app.getLineAlerts(data);
    assert.equal(alerts.length, 1);
    assert.equal(alerts[0].severity, 'critical');
    assert.match(alerts[0].summary, /RAM 總容量減少 62\.30 → 30\.30 GB/);
    assert.match(alerts[0].message, /CAPI02.*上次確認 2026-11-10T10:00:30\+08:00.*首次偵測 2026-11-10T10:01:00\+08:00/);
    const state = {processZone: 'capi', status: 'halted', line: {line: 'CAPI02', factory: 'MOD2'},
        overviewRow: app.node(), data};
    app.renderOverviewRow(state);
    assert.equal(state.overviewRow.dataset.health, 'critical');
    const badge = state.overviewRow.querySelector('[data-field="overview-alerts"]').children[0];
    assert.match(badge.textContent, /RAM 總容量減少/);
    assert.match(badge.title, /調閱監視器/);
    app.lineStates.set('line', state);
    app.renderAlerts();
    assert.match(app.elements['alert-list'].children[0].textContent, /MOD2 \/ CAPI02.*首次偵測/);
});

test('RAM increases remain visible, including restoration and unavailable current readings', () => {
    const app = loadDashboard();
    const change = ramChange(30.3, 62.3);
    change.available = false;
    const alerts = app.getLineAlerts(app.normalizeStatus({ram_capacity_alert: change}));
    assert.equal(alerts.length, 1);
    assert.equal(alerts[0].severity, 'warning');
    assert.match(alerts[0].summary, /RAM 總容量增加/);
});

test('old payloads, first samples and invalid capacities do not invent RAM change alerts', () => {
    const app = loadDashboard();
    for (const raw of [{}, {ram_capacity_alert: {active: false}},
        {ram_capacity_alert: ramChange(null, 30.3)},
        {ram_capacity_alert: ramChange(62.3, 0)},
        {ram_capacity_alert: ramChange(62.3, 62.3)},
        {hardware: {memory: {used_gb: 6.1, total_gb: 62.3, used_percent: 10}}}]) {
        assert.equal(app.getLineAlerts(app.normalizeStatus(raw)).length, 0);
    }
    const state = {status: 'offline', line: {line: 'CAPI02'}, overviewRow: app.node(),
        data: app.normalizeStatus({ram_capacity_alert: ramChange(62.3, 30.3)})};
    app.renderOverviewRow(state);
    assert.equal(state.overviewRow.querySelector('[data-field="overview-alerts"]').children.length, 0);
});
