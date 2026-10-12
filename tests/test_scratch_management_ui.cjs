const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function loadPage(value, receive) {
    const html = fs.readFileSync(path.join(__dirname, '../templates/scratch_management.html'), 'utf8');
    const elements = {};
    for (const id of ['scratch-select-all', 'scratch-selection-count', 'distribution-result',
        'distribution-summary', 'distribute', 'scratch-bundle', 'scratch-safety-multiplier']) {
        elements[id] = {value: '', disabled: false, textContent: '', style: {}, addEventListener() {}, focus() {}};
    }
    elements['scratch-bundle'].value = 'scratch_classifier_v2.pkl';
    elements['scratch-safety-multiplier'].value = value;
    const lines = ['line1', 'line2'].map(value => ({
        value, checked: true, disabled: false, parentElement: {textContent: value}, addEventListener() {},
    }));
    const calls = [];
    const context = {
        document: {querySelectorAll: () => lines, getElementById: id => elements[id]},
        fetch: async (url, options) => {
            const payload = JSON.parse(options.body);
            calls.push(payload);
            assert.equal(elements['scratch-safety-multiplier'].disabled, true);
            return receive(payload);
        },
    };
    vm.runInNewContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
    return {elements, calls, run: () => context.distributeScratch()};
}

test('sends selected multiplier to every line and restores controls after mixed results', async () => {
    const page = loadPage('1.5', payload => ({
        ok: payload.line_id === 'line1',
        json: async () => payload.line_id === 'line1'
            ? {success: true, message: '模型與安全倍率 1.5 已套用', bundle: payload.bundle}
            : {error: '線體未確認套用安全倍率'},
    }));
    await page.run();
    assert.deepEqual(page.calls.map(payload => payload.scratch_safety_multiplier), [1.5, 1.5]);
    assert.deepEqual(page.calls.map(payload => payload.line_id), ['line1', 'line2']);
    assert.match(page.elements['distribution-summary'].textContent, /安全倍率：1.5/);
    assert.match(page.elements['distribution-summary'].textContent, /成功 1 台，失敗 1 台/);
    assert.match(page.elements['distribution-result'].textContent, /未確認套用安全倍率/);
    assert.equal(page.elements['scratch-safety-multiplier'].disabled, false);
    assert.equal(page.elements.distribute.disabled, false);
});

test('invalid multipliers do not send any requests', async () => {
    for (const value of ['', '0', '-1', 'NaN', 'Infinity']) {
        const page = loadPage(value, () => assert.fail('must not send'));
        await page.run();
        assert.equal(page.calls.length, 0);
        assert.match(page.elements['distribution-summary'].textContent, /安全倍率必須/);
        assert.equal(page.elements.distribute.disabled, false);
    }
});
