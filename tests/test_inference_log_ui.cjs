// Run with: node --test tests/test_inference_log_ui.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {parseLog} = require('../static/js/inference-log.js');

test('stage NG counts do not become final panel decisions', () => {
    const result = parseLog('[v2] W0F00000: tiles=15 (inner=15, edge=0), NG=15, infer 0.33s\nPanel X 總耗時 57.92s | 前置=10.51s | 6 lighting(s), 6 NG');
    assert.match(result.decision, /未提供/);
    assert.match(result.events[0].message, /階段 NG 15/);
    assert.equal(result.overview, '面板 X · 推論流程 57.92 秒');
});
test('final conversion takes precedence over intermediate NG, including production OK-i format', () => {
    const result = parseLog('[WITHIN_SPEC_INFERENCE] 原始 AI=NG，已執行規格內檢查，結果=not_within_spec\n[WITHIN_SPEC_INFERENCE] 原始 AI=NG，符合規格內，最終判定 OK-i');
    assert.match(result.decision, /^OK-i/);
});
test('repeated warnings aggregate but retain original timestamps and lines', () => {
    const line = '[2026-09-14 14:57:43] WARNING [capi.preprocess] [boundary] reject non-linear vertical edge: residual_p95=23.2px > 15.5px (54 samples)';
    const result = parseLog(line+'\r\n'+line.replace('14:57:43','14:58:02'));
    assert.equal(result.alerts.length, 1);
    assert.equal(result.alerts[0].count, 2);
    assert.equal(result.events[1].time, '14:58:02');
    assert.equal(result.alerts[0].lines[0],line);
});
test('specification details and unknown formats are losslessly preserved', () => {
    const line = '  - B0F00000 AOI點未檢出：數量 5 > 0 [NG]；tile 1 AOI(1419,861)；結果=NG';
    const unknown = '<img src=x onerror=alert(1)> future-format P:3';
    const result = parseLog(line+'\n'+unknown);
    assert.equal(result.specs[0][0],'B0F00000 AOI點未檢出');
    assert.match(result.specs[0][1],/AOI\(1419,861\)/);
    assert.equal(result.events[1].raw,unknown);
    assert.equal(result.events[1].message,unknown);
});
test('empty and partial logs do not invent results or timing', () => {
    const result = parseLog('');
    assert.equal(result.timings.length,0);
    assert.equal(result.specs.length,0);
    assert.match(result.decision,/未提供/);
});

test('model summary distinguishes per-zone methods and preserves inference weight settings', () => {
    const soft = '[MODEL_TRAINING] ' + JSON.stringify({lighting:'G0F00000',zone:'inner',mode:'softpatch_plus_v1',discriminator:'lof_gaussian',soft_weight:true,weight_strength:1,weight_min:1,weight_max:1.8,model_path:'/model/G0F00000-inner.pt'});
    const plain = '[MODEL_TRAINING] ' + JSON.stringify({lighting:'G0F00000',zone:'edge',mode:'off',model_path:'/model/G0F00000-edge.pt'});
    const result = parseLog([soft, plain, soft].join('\n'));
    assert.equal(result.models.length, 2);
    assert.equal(result.models[0].unit, 'G0F00000 / INNER');
    assert.match(result.models[0].mode, /SoftPatch\+/);
    assert.match(result.models[0].settings, /LOF＋Gaussian；推論權重開啟；強度 1；實際倍率 1.00～1.80/);
    assert.equal(result.models[1].mode, 'PatchCore（清洗關閉）');
    assert.match(result.events[0].message, /使用模型/);
});

test('missing metadata and malformed markers cannot be mistaken for baseline or SoftPatch+', () => {
    assert.equal(parseLog('loaded /model/SoftPatch.pt').models.length, 0);
    assert.equal(parseLog('[MODEL_TRAINING] {broken').models.length, 0);
    const result = parseLog('[MODEL_TRAINING] '+JSON.stringify({mode:'unknown',lighting:'G0F00000',zone:'inner'}));
    assert.equal(result.models[0].mode, '訓練清洗方式未記錄');
    const soft = parseLog('[MODEL_TRAINING] '+JSON.stringify({mode:'softpatch_plus_v1',lighting:'G0F00000',zone:'inner',soft_weight:false,weight_strength:0,model_path:'<img src=x onerror=alert(1)>'}));
    assert.match(soft.models[0].settings, /評分設定未記錄；推論權重關閉；強度 0/);
    assert.equal(soft.models[0].path, '<img src=x onerror=alert(1)>');
});
