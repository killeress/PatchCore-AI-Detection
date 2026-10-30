const {test} = require('node:test');
const assert = require('node:assert/strict');
const dashboard = require('../central_dashboard/ok-s.js');

test('shift boundaries and cross-midnight dates use UTC+8 regardless of browser timezone', () => {
    for (const [timestamp, date, shift] of [
        ['2026-10-30T07:29:59+08:00', '2026-10-29', 'night'],
        ['2026-10-30T07:30:00+08:00', '2026-10-30', 'day'],
        ['2026-10-30T19:29:59+08:00', '2026-10-30', 'day'],
        ['2026-10-30T19:30:00+08:00', '2026-10-30', 'night'],
        ['2026-11-01T02:00:00+08:00', '2026-10-31', 'night'],
    ]) assert.deepEqual(dashboard.currentShift(new Date(timestamp)), {date, shift});
    const window = dashboard.shiftWindow('2026-10-30', 'night');
    assert.equal(window.start.toISOString(), '2026-10-30T11:30:00.000Z');
    assert.equal(window.end.toISOString(), '2026-10-30T23:30:00.000Z');
});

test('invalid, future and expired selections cannot produce a misleading zero report', () => {
    const now = new Date('2026-10-30T12:00:00+08:00');
    for (const selection of [
        {date: '2026-02-30', shift: 'day'}, {date: '', shift: 'day'},
        {date: '2026-10-30', shift: 'night'}, {date: '2026-09-30', shift: 'day'},
        {date: '2026-10-30', shift: 'unknown'},
    ]) assert.notEqual(dashboard.validateSelection(selection, now), '');
    assert.equal(dashboard.validateSelection({date: '2026-10-30', shift: 'day'}, now), '');
    assert.equal(dashboard.validateSelection({date: '2026-10-01', shift: 'day'}, now), '');
});

test('all-line summary weights by total input and excludes unavailable lines', () => {
    const summary = dashboard.summarize([
        {state: 'ok', counts: {total: 100, ng: 10, converted: 20}},
        {state: 'ok', counts: {total: 900, ng: 90, converted: 0}},
        {state: 'offline', counts: {total: 10000, ng: 100, converted: 200}},
        {state: 'unsupported'}, {state: 'pending'},
    ]);
    assert.deepEqual(summary, {total: 1000, ng: 100, converted: 20, before: 12, after: 10, drop: 2, covered: 2});
    assert.notEqual(summary.drop, 10); // The arithmetic mean of per-line percentages is wrong.
});

test('zero production, zero conversions and missing support are distinct', () => {
    assert.equal(dashboard.summarize([{state: 'offline'}, {state: 'unsupported'}]), null);
    assert.deepEqual(dashboard.rates({total: 0, ng: 0, converted: 0}), {before: null, after: null, drop: null});
    assert.deepEqual(dashboard.rates({total: 100, ng: 0, converted: 0}), {before: 0, after: 0, drop: 0});
    assert.equal(dashboard.parseCounts({total: 100, ng_count: 10}), null);
    assert.deepEqual(dashboard.parseCounts({total: 0, ng_count: null, ok_s_count: 0}), {total: 0, ng: 0, converted: 0});
    for (const payload of [
        {total: 10, ng_count: 8, ok_s_count: 3}, {total: 10, ng_count: -1, ok_s_count: 0},
        {total: 10, ng_count: 2, ok_s_count: null}, {total: 10, ng_count: 2, ok_s_count: 0.5},
    ]) assert.throws(() => dashboard.parseCounts(payload));
});

test('configuration preserves disabled lines and both factory networks/processes', () => {
    const lines = [
        {id: 'capi', apiUrl: 'http://10.172.1.1/api/status', line: 'CAPI01', enabled: true},
        {id: 'aapi', apiUrl: 'http://10.174.1.1/api/status', line: 'AAPI02', enabled: false},
    ];
    const config = dashboard.normalizeConfig({lines});
    assert.deepEqual(config.lines, lines);
    assert.equal(config.refreshSeconds, 30);
    assert.equal(config.timeoutMs, 8000);
    assert.throws(() => dashboard.normalizeConfig({}));
});

const line = {apiUrl: 'http://line.test:8080/proxy/api/status'};
const selection = {date: '2026-10-30', shift: 'night'};
const response = body => ({ok: true, json: async () => body});
function mockFetch(t, fn) {
    const previous = global.fetch;
    global.fetch = fn;
    t.after(() => { global.fetch = previous; });
}

test('line request uses the selected shift, retains proxy prefix/port, and validates returned shift', async t => {
    const payload = {...selection, total: 100, ng_count: 8, ok_s_count: 2};
    mockFetch(t, async url => {
        assert.equal(url, 'http://line.test:8080/proxy/api/shift_report?date=2026-10-30&shift=night');
        return response(payload);
    });
    assert.deepEqual(await dashboard.requestLine(line, selection, 50), {state: 'ok', counts: {total: 100, ng: 8, converted: 2}});
    payload.shift = 'day';
    assert.equal((await dashboard.requestLine(line, selection, 50)).state, 'error');
});

test('legacy report without OK-s data requires update, not zero improvement', async t => {
    mockFetch(t, async () => response({...selection, total: 10, ng_count: 1}));
    assert.equal((await dashboard.requestLine(line, selection, 50)).state, 'unsupported');
});

test('CORS-hidden legacy 404 probes status; an offline line stays excluded', async t => {
    const requested = [];
    mockFetch(t, async url => {
        requested.push(url);
        if (url.includes('shift_report')) throw new TypeError('Failed to fetch');
        return response({server: {running: true}});
    });
    assert.equal((await dashboard.requestLine(line, selection, 50)).state, 'unsupported');
    assert.equal(requested[1], line.apiUrl);
    global.fetch = async () => { throw new TypeError('Failed to fetch'); };
    assert.equal((await dashboard.requestLine(line, selection, 50)).state, 'offline');
});

test('HTTP errors and malformed counts are query failures; invalid URLs are not fetched', async t => {
    mockFetch(t, async () => ({ok: false, status: 500}));
    assert.equal((await dashboard.requestLine(line, selection, 50)).state, 'error');
    global.fetch = async () => ({ok: false, status: 404});
    assert.equal((await dashboard.requestLine(line, selection, 50)).state, 'unsupported');
    global.fetch = async () => response({...selection, total: 5, ng_count: 5, ok_s_count: 5});
    assert.equal((await dashboard.requestLine(line, selection, 50)).state, 'error');
    global.fetch = async () => { assert.fail('Invalid URL must not be fetched'); };
    assert.equal((await dashboard.requestLine({apiUrl: 'javascript:alert(1)'}, selection, 50)).state, 'error');
});

test('query cancellation aborts old requests without classifying them as offline', async t => {
    const controller = new AbortController();
    mockFetch(t, async (_url, options) => new Promise((_resolve, reject) => {
        options.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), {once: true});
    }));
    const pending = dashboard.requestLine(line, selection, 100, controller.signal);
    controller.abort();
    await assert.rejects(pending, {name: 'AbortError'});
});
