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
        const listeners = new Map();
        return {children: [], dataset: {}, hidden: false, open: false,
            classList: {toggle() {}, add() {}, remove() {}}, removeAttribute() {}, setAttribute() {},
            addEventListener(name, callback) { listeners.set(name, callback); },
            trigger(name) { return listeners.get(name)?.(); },
            querySelector(selector) {
                if (!selectors.has(selector)) selectors.set(selector, node());
                return selectors.get(selector);
            },
            appendChild(child) { this.children.push(child); },
            replaceChildren() { this.children = []; },
            showModal() { this.open = true; },
            close() { this.open = false; this.trigger('close'); }};
    }
    const elements = new Map();
    const messages = [];
    const requests = [];
    let responder = () => { throw new Error('Unexpected request'); };
    const context = {
        window: {CAPIInferenceErrors: errors, location: {protocol: 'http:'},
            setTimeout: () => 1, alert: message => messages.push(message)},
        document: {addEventListener() {}, createElement: node, querySelectorAll: () => [],
            getElementById(id) {
                if (!elements.has(id)) elements.set(id, node());
                return elements.get(id);
            }},
        AbortController, URL, clearTimeout() {},
        fetch(url, options) { requests.push({url, options}); return responder(url, options); },
    };
    vm.runInNewContext(source.replace(/\}\)\(\);\s*$/, `window.testApi = {
        normalizeStatus, getLineAlerts, initializeRamHistory, openRamHistory, acknowledgeRamCapacity,
        renderRamHistoryButton, renderOverviewRow, renderLineCard, refreshLine
    };})();`), context);
    const app = context.window.testApi;
    function state(alert = ramAlert()) {
        return {status: 'online', processZone: 'capi', card: node(), overviewRow: node(),
            line: {id: 'line-02', line: 'CAPI02', pcName: 'CAPI02', factory: 'MOD2', apiUrl: 'http://10.174.37.137:8080/api/status'},
            data: app.normalizeStatus({server: {hostname: 'CAPI02'}, ram_capacity_alert: alert})};
    }
    return {...app, state, node, requests, messages, window: context.window,
        element: id => context.document.getElementById(id),
        setFetch(callback) { responder = callback; }};
}

function event(id, acknowledged = false) {
    return {id, hostname: 'CAPI02', previous_total_gb: 62.3, new_total_gb: 30.3,
        last_seen_at: '2026-11-11T10:00:00+08:00', detected_at: '2026-11-11T10:00:30+08:00',
        acknowledged_by: acknowledged ? '<維護人員>' : '',
        acknowledged_at: acknowledged ? '2026-11-11T11:00:00+08:00' : ''};
}

function ramAlert(id = 1, acknowledged = false) {
    return {history_supported: true, acknowledge_supported: true,
        active: !acknowledged, latest_change: event(id, acknowledged)};
}

function response(payload, status = 200) {
    return {ok: status >= 200 && status < 300, status, json: async () => payload};
}

test('history stays at the RAM heading and remains available after confirmation', () => {
    const html = fs.readFileSync(path.join(__dirname, '../central_dashboard/index.html'), 'utf8');
    assert.match(html, /class="ram-system-heading">\s*<span>RAM<\/span>\s*<button[^>]*data-action="ram-history"[^>]*>RAM 變動紀錄<\/button>/);
    const app = loadDashboard();
    const state = app.state(ramAlert(1, true));
    app.renderRamHistoryButton(state);
    assert.equal(state.card.querySelector('[data-action="ram-history"]').disabled, false);
    assert.equal(app.getLineAlerts(state.data).length, 0);
    for (const status of ['offline', 'checking']) {
        state.status = status;
        app.renderRamHistoryButton(state);
        assert.equal(state.card.querySelector('[data-action="ram-history"]').disabled, true);
    }
    const old = app.state({active: true});
    app.renderRamHistoryButton(old);
    assert.equal(old.card.querySelector('[data-action="ram-history"]').disabled, true);
});

test('history dialog loads pages and shows confirmation actor and timezone as text', async () => {
    const app = loadDashboard();
    app.initializeRamHistory();
    app.setFetch(async () => response({events: [event(2, true)], nextBeforeId: 2}));
    await app.openRamHistory(app.state());
    assert.equal(app.element('ram-history-dialog').open, true);
    assert.match(app.element('ram-history-device').textContent, /MOD2.*CAPI02/);
    const cells = app.element('ram-history-rows').children[0].children;
    assert.equal(cells[0].textContent, '62.30 → 30.30 GB');
    assert.equal(cells[1].textContent, '2026-11-11 10:00:00+08:00');
    assert.equal(cells[3].textContent, '已確認');
    assert.equal(cells[4].textContent, '<維護人員>');
    assert.equal(app.requests[0].options.credentials, 'omit');
    assert.equal(new URL(app.requests[0].url).host, '10.174.37.137:8080');
    app.setFetch(async () => response({events: [event(1)], nextBeforeId: null}));
    await app.element('ram-history-more').trigger('click');
    assert.equal(new URL(app.requests[1].url).searchParams.get('beforeId'), '2');
    assert.equal(app.element('ram-history-rows').children.length, 2);
    assert.equal(app.element('ram-history-more').hidden, true);
    assert.equal(app.element('ram-history-rows').children[1].children[3].textContent, '未確認');
    app.element('ram-history-close').trigger('click');
    assert.equal(app.element('ram-history-dialog').open, false);
});

test('history has explicit empty and error states, with a retry that does not duplicate rows', async () => {
    const app = loadDashboard();
    app.initializeRamHistory();
    app.setFetch(async () => response({events: [], nextBeforeId: null}));
    await app.openRamHistory(app.state());
    assert.match(app.element('ram-history-message').textContent, /沒有 RAM/);
    app.setFetch(async () => { throw new Error('connection failed'); });
    await app.openRamHistory(app.state());
    assert.match(app.element('ram-history-message').textContent, /無法讀取歷史/);
    assert.equal(app.element('ram-history-more').textContent, '重試');
    app.setFetch(async () => response({events: [event(1)], nextBeforeId: null}));
    await app.element('ram-history-more').trigger('click');
    assert.equal(app.element('ram-history-rows').children.length, 1);
});

test('late history responses do not populate a different device dialog', async () => {
    const app = loadDashboard();
    app.initializeRamHistory();
    let resolveFirst;
    app.setFetch(() => new Promise(resolve => { resolveFirst = resolve; }));
    const first = app.openRamHistory(app.state());
    const stateB = app.state();
    stateB.line.line = 'CAPI03';
    app.setFetch(async () => response({events: [event(3)], nextBeforeId: null}));
    await app.openRamHistory(stateB);
    resolveFirst(response({events: [event(1)], nextBeforeId: null}));
    await first;
    assert.equal(app.element('ram-history-rows').children.length, 1);
    assert.match(app.element('ram-history-device').textContent, /CAPI03/);
    assert.equal(app.requests[0].options.signal.aborted, true);
});

test('confirmation button calls the central route and clears only the returned notice', async () => {
    const app = loadDashboard();
    const state = app.state();
    app.renderOverviewRow(state);
    const badge = state.overviewRow.querySelector('[data-field="overview-alerts"]').children[0];
    assert.equal(badge.children[0].textContent, '已確認');
    app.setFetch(async () => response({success: true, ram_capacity_alert: ramAlert(1, true)}));
    await badge.children[0].trigger('click');
    assert.equal(app.requests[0].url, '/api/central-dashboard/ram-capacity/acknowledge');
    assert.equal(app.requests[0].options.credentials, 'same-origin');
    assert.deepEqual(JSON.parse(app.requests[0].options.body), {lineId: 'line-02', eventId: 1});
    assert.equal(state.data.ramCapacity.active, false);
    assert.equal(state.overviewRow.querySelector('[data-field="overview-alerts"]').children.length, 0);
    app.renderRamHistoryButton(state);
    assert.equal(state.card.querySelector('[data-action="ram-history"]').disabled, false);
    state.data = app.normalizeStatus({ram_capacity_alert: ramAlert(2)});
    app.renderOverviewRow(state);
    assert.equal(state.overviewRow.querySelector('[data-field="overview-alerts"]').children.length, 1);
});

test('failed or unauthorized confirmations keep the notice; file mode disables confirmation', async () => {
    const app = loadDashboard();
    const state = app.state();
    app.setFetch(async () => response({error: 'database locked'}, 503));
    await app.acknowledgeRamCapacity(state, 1);
    assert.equal(state.data.ramCapacity.active, true);
    assert.match(app.messages[0], /database locked/);
    app.setFetch(async () => response({error: 'login required'}, 401));
    await app.acknowledgeRamCapacity(state, 1);
    assert.match(app.window.location.href, /settings\/login/);
    assert.equal(state.data.ramCapacity.active, true);
    app.window.location.protocol = 'file:';
    app.renderOverviewRow(state);
    const badge = state.overviewRow.querySelector('[data-field="overview-alerts"]').children[0];
    assert.equal(badge.children[0].disabled, true);
});

test('acknowledging an older event never suppresses a newer capacity change', async () => {
    const app = loadDashboard();
    const state = app.state(ramAlert(2));
    app.setFetch(async () => response({success: true, ram_capacity_alert: ramAlert(1, true)}));
    await app.acknowledgeRamCapacity(state, 1);
    assert.equal(state.data.ramCapacity.eventId, 2);
    assert.equal(state.data.ramCapacity.active, true);
});

test('an in-flight status request cannot resurrect a successfully confirmed notice', async () => {
    const app = loadDashboard();
    const state = app.state();
    let resolveStatus;
    app.setFetch(url => url.startsWith('http:')
        ? new Promise(resolve => { resolveStatus = resolve; })
        : Promise.resolve(response({success: true, ram_capacity_alert: ramAlert(1, true)})));
    const refreshing = app.refreshLine(state);
    await app.acknowledgeRamCapacity(state, 1);
    resolveStatus(response({ram_capacity_alert: ramAlert(1)}));
    await refreshing;
    assert.equal(state.data.ramCapacity.active, false);
});
