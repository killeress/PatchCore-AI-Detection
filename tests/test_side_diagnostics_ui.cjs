// Run with: node --test tests/test_side_diagnostics_ui.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/js/side_image_viewer.js'), 'utf8');
const bundle = id => ({schema_version: 1, source: 'saved_side_white_result', record: {side_result_id: id},
    side_log_source: 'inference_log', side_log: '[SIDE_WHITE_INFERENCE] A1 9.52 產品 px\n保留 NG',
    payload: {id, candidates: [{side_contour: [[1, 2], [3, 4], [5, 6]]}],
        future: {text: '</script><img onerror="bad()">', values: Array.from({length: 2000}, (_, i) => i)}}});

function harness(options = {}) {
    const listeners = [], state = {writes: [], requests: [], areas: [], execCalls: 0};
    const document = {activeElement: null, addEventListener: (type, fn) => {
        if (type === 'click') listeners.push(fn);
    }};
    document.body = {append: area => { state.areas.push(area); area.parent = document.body; }};
    document.activeElement = document.body;
    document.createElement = tag => ({tagName: tag, value: '', style: {}, attrs: {},
        setAttribute(key, value) { this.attrs[key] = value; },
        focus() { if (!this.disabled) document.activeElement = this; },
        select() { state.selected = this.value; },
        remove() {
            state.areas = state.areas.filter(area => area !== this);
            if (document.activeElement === this) document.activeElement = document.body;
            if (this.parent) this.parent.manual = null;
        }});
    document.execCommand = command => {
        assert.equal(command, 'copy');
        state.execCalls++;
        return options.exec !== false;
    };
    const navigator = options.http ? {} : {clipboard: {writeText: async text => {
        if (options.rejectClipboard) throw new Error('denied');
        state.writes.push(text);
    }}};
    vm.runInNewContext(source, {document, navigator, window: {location: {href: 'http://capi07/record/51'}},
        fetch: async (url, init) => {
            state.requests.push({url, init});
            if (options.response) return options.response(url);
            const id = Number(new URL(url, 'http://capi07').searchParams.get('id'));
            return {ok: true, json: async () => bundle(id)};
        }});
    function card(id, inline) {
        const status = {hidden: true, textContent: '', after(area) { area.parent = result; result.manual = area; }};
        const result = {status, manual: null, querySelector(selector) {
            if (selector === '[data-sw-copy-status]') return status;
            if (selector === '[data-sw-debug-diagnostics]') return inline ? {textContent: JSON.stringify(inline)} : null;
            if (selector === '[data-sw-copy-manual]') return result.manual;
            throw new Error('unexpected selector: ' + selector);
        }};
        const button = {dataset: {swResultId: String(id)}, textContent: '複製排查資料', disabled: false, attrs: {},
            closest: selector => selector === 'section' ? result : null,
            setAttribute(key, value) { this.attrs[key] = value; },
            removeAttribute(key) { delete this.attrs[key]; },
            focus() { if (!this.disabled) document.activeElement = this; }};
        result.button = button;
        result.click = () => {
            document.activeElement = button;
            const event = {preventDefault() {}, target: {closest: selector =>
                selector === '[data-sw-copy-diagnostics]' ? button : null}};
            for (const listener of listeners) listener(event);
        };
        return result;
    }
    return {state, document, card};
}

async function settle() { for (let i = 0; i < 5; i++) await new Promise(resolve => setImmediate(resolve)); }
function exportedPayload(text) { return JSON.parse(text.split('=== 側拍結果 payload（完整 JSON） ===\n')[1]); }

test('saved card copies the complete payload, raw side log and identifiers on demand', async () => {
    const env = harness(), card = env.card(31);
    assert.equal(env.state.requests.length, 0);
    card.click();
    assert.equal(card.button.disabled, true);
    await settle();
    assert.equal(env.state.requests[0].url, '/api/side-white/diagnostics?id=31');
    assert.equal(env.state.requests[0].init.cache, 'no-store');
    const text = env.state.writes[0];
    assert.deepEqual(exportedPayload(text), bundle(31).payload);
    assert.ok(text.includes(bundle(31).side_log));
    assert.match(text, /"side_result_id": 31/);
    assert.match(text, /"copied_at"/);
    assert.match(text, /http:\/\/capi07\/record\/51/);
    assert.match(card.status.textContent, /已複製/);
    assert.equal(card.button.disabled, false);
    assert.equal(card.button.textContent, '複製排查資料');
});

test('HTTP fallback copies fully and removes the temporary textarea', async () => {
    const env = harness({http: true}), card = env.card(8);
    card.click();
    await settle();
    assert.equal(env.state.execCalls, 1);
    assert.deepEqual(exportedPayload(env.state.selected), bundle(8).payload);
    assert.equal(env.state.areas.length, 0);
    assert.equal(env.document.activeElement, card.button);
    assert.match(card.status.textContent, /已複製/);
});

test('clipboard denial provides selected read-only data for manual copy', async () => {
    const env = harness({rejectClipboard: true, exec: false}), card = env.card(9);
    card.click();
    await settle();
    assert.equal(env.state.writes.length, 0);
    assert.equal(card.manual.readOnly, true);
    assert.deepEqual(exportedPayload(card.manual.value), bundle(9).payload);
    assert.equal(env.state.selected, card.manual.value);
    assert.match(card.status.textContent, /Ctrl\+C/);
    assert.doesNotMatch(card.status.textContent, /已複製/);
    assert.equal(card.button.disabled, false);
});

test('a missing result reports failure and never copies incomplete data', async () => {
    const env = harness({response: async () => ({ok: false, json: async () => ({error: '找不到側拍結果'})})});
    const card = env.card(99);
    card.click();
    await settle();
    assert.match(card.status.textContent, /複製失敗.*找不到側拍結果/);
    assert.equal(env.state.writes.length, 0);
    assert.equal(env.state.execCalls, 0);
    assert.equal(card.manual, null);
    assert.equal(card.button.disabled, false);
});

test('DEBUG results copy their own unsaved payload without requesting a saved result', async () => {
    const env = harness(), data = {...bundle(10), source: 'debug_side_white_result', side_log: 'DEBUG 未保存正式 LOG'};
    const card = env.card('debug', data);
    card.click();
    await settle();
    assert.equal(env.state.requests.length, 0);
    assert.deepEqual(exportedPayload(env.state.writes[0]), data.payload);
    assert.match(env.state.writes[0], /debug_side_white_result/);
});

test('delegation keeps multiple AJAX cards separate and blocks repeated pending clicks', async () => {
    let release;
    const pending = new Promise(resolve => { release = resolve; });
    const env = harness({response: async url => {
        if (url.endsWith('=1')) await pending;
        return {ok: true, json: async () => bundle(url.endsWith('=1') ? 1 : 2)};
    }});
    const first = env.card(1), second = env.card(2);
    first.click(); first.click(); second.click();
    await settle();
    assert.equal(env.state.requests.length, 2);
    assert.deepEqual(exportedPayload(env.state.writes[0]), bundle(2).payload);
    assert.match(first.status.textContent, /正在準備/);
    release();
    await settle();
    assert.deepEqual(exportedPayload(env.state.writes[1]), bundle(1).payload);
    assert.match(first.status.textContent, /已複製/);
    assert.match(second.status.textContent, /已複製/);
});
