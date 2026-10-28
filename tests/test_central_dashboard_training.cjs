const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const errors = require('../central_dashboard/inference-errors.js');
const source = fs.readFileSync(path.join(__dirname, '../central_dashboard/app.js'), 'utf8');

function loadDashboard() {
    function node() {
        const selectors = new Map();
        return {children: [], dataset: {}, classList: {toggle() {}, remove() {}, add() {}},
            removeAttribute() {}, setAttribute() {},
            querySelector(selector) {
                if (!selectors.has(selector)) selectors.set(selector, node());
                return selectors.get(selector);
            },
            replaceChildren() { this.children = []; },
            appendChild(child) { this.children.push(child); }};
    }
    const elements = new Map();
    const context = {
        window: {CAPIInferenceErrors: errors},
        document: {addEventListener() {}, createElement: node, querySelectorAll: () => [],
            getElementById(id) {
                if (!elements.has(id)) elements.set(id, node());
                return elements.get(id);
            }},
    };
    vm.runInNewContext(source.replace(/\}\)\(\);\s*$/, `window.testApi = {
        normalizeStatus, renderLineCard, renderAlerts, lineStates
    };})();`), context);
    function state(training, status = 'online', processZone = 'capi') {
        return {status, processZone, line: {line: 'CAPI42'}, card: node(), overviewRow: node(),
            data: context.window.testApi.normalizeStatus({training})};
    }
    return {...context.window.testApi, elements, state};
}

function active(phase = 'train', extra = {}) {
    return {active: true, jobs: [{kind: 'patchcore', machine_id: 'MODEL-A', phase, ...extra}]};
}

function badges(state) {
    return [state.card.querySelector('[data-field="status"]'),
        state.overviewRow.querySelector('[data-field="overview-status"]')];
}

test('existing status badges show training and recover their original status after it ends', () => {
    const app = loadDashboard();
    for (const [status, label] of [['online', '正常'], ['halted', '停線'], ['warning', '服務異常']]) {
        const state = app.state(active(), status);
        app.renderLineCard(state);
        for (const badge of badges(state)) {
            assert.equal(badge.textContent, '訓練中');
            assert.equal(badge.dataset.training, 'true');
            assert.match(badge.title, /新模型訓練 · MODEL-A · 訓練中/);
        }
        state.data = app.normalizeStatus({training: {active: false, jobs: []}});
        app.renderLineCard(state);
        for (const badge of badges(state)) {
            assert.equal(badge.textContent, label);
            assert.equal(badge.dataset.training, 'false');
            assert.doesNotMatch(badge.title, /MODEL-A/);
        }
    }
});

test('offline and checking status override cached training even before the first successful poll', () => {
    const app = loadDashboard();
    for (const [status, label] of [['offline', '離線'], ['checking', '連線中']]) {
        const state = app.state(active(), status);
        for (const data of [state.data, null]) {
            state.data = data;
            app.renderLineCard(state);
            for (const badge of badges(state)) {
                assert.equal(badge.textContent, label);
                assert.equal(badge.dataset.training, 'false');
                assert.doesNotMatch(badge.title, /MODEL-A/);
            }
        }
    }
});

test('old or malformed payloads preserve the original status and explain support in the tooltip', () => {
    const app = loadDashboard();
    for (const training of [undefined, null, {}, {active: false}, {active: 'true', jobs: []}]) {
        const state = app.state(training);
        app.renderLineCard(state);
        for (const badge of badges(state)) {
            assert.equal(badge.textContent, '正常');
            assert.equal(badge.dataset.training, 'false');
            assert.match(badge.title, /未提供訓練狀態/);
        }
    }
});

test('preparation and multiple jobs keep one training label with details only in the tooltip', () => {
    const app = loadDashboard();
    const state = app.state(active('preprocess', {machine_id: '<img src=x onerror=alert(1)>'}));
    app.renderLineCard(state);
    assert.equal(badges(state)[0].textContent, '訓練中');
    assert.match(badges(state)[0].title, /<img src=x onerror=alert\(1\)> · 圖片前處理/);
    state.data = app.normalizeStatus({training: {active: true, jobs: [
        {kind: 'patchcore_partial', phase: 'train', machine_id: 'MODEL-B'},
        {kind: 'patchcore_submodel', phase: 'reload', unit: 'W-inner'},
        {kind: 'scratch', phase: 'merge'},
    ]}});
    app.renderLineCard(state);
    for (const badge of badges(state)) {
        assert.equal(badge.textContent, '訓練中');
        assert.match(badge.title, /局部重訓 · MODEL-B · 訓練中/);
        assert.match(badge.title, /子模型重訓 · W-inner · 載入模型/);
        assert.match(badge.title, /刮痕分類器重訓 · 合併資料/);
    }
});

test('training label keeps service errors and hardware alerts visible', () => {
    const app = loadDashboard();
    const state = app.state(active(), 'warning');
    state.error = 'API 可連線，但服務回報未運行。';
    state.data.hardware.ramUsedPercent = 96;
    app.lineStates.set('line', state);
    app.renderLineCard(state);
    app.renderAlerts();
    assert.equal(badges(state)[0].textContent, '訓練中');
    assert.match(badges(state)[0].title, /服務回報未運行/);
    assert.equal(state.status, 'warning');
    assert.equal(state.card.dataset.state, 'warning');
    assert.equal(state.overviewRow.dataset.health, 'critical');
    assert.equal(state.card.querySelector('[data-field="error"]').hidden, false);
    assert.ok(app.elements.get('alert-list').children.some(item => /RAM/.test(item.textContent)));
});
