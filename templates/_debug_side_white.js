    {% include "_side_white_params.js" %}
    const swDebugRecordId = new URLSearchParams(window.location.search).get('record_id');
    let swDebugBusy = false, swDebugReady = false;

    function swDebugEscape(value) {
        return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }
    function setSideWhiteDebugBusy(busy) {
        swDebugBusy = busy;
        document.getElementById('sw-debug-fields').disabled = busy || !swDebugReady;
        document.getElementById('sw-debug-run').disabled = busy || !swDebugReady;
        document.getElementById('sw-debug-reset').disabled = busy;
    }
    function updateSideWhiteDebugCandidateDisplay() {
        const showAll = document.getElementById('sw-debug-show-all').checked;
        const result = document.getElementById('sw-debug-result');
        result.querySelectorAll('[data-sw-preview-kind]').forEach(element => {
            const url = showAll ? element.dataset.swAllUrl : element.dataset.swNearbyUrl;
            if (url) element.setAttribute(element.tagName === 'IMG' ? 'src' : 'href', url);
        });
        result.querySelectorAll('[data-sw-nearby-note]').forEach(element => {
            element.textContent = showAll ? element.dataset.swAllNote : element.dataset.swNearbyNote;
        });
    }
    function renderSideWhiteDebugParams(values) {
        document.getElementById('sw-debug-params').innerHTML =
            sideWhiteParamFields.map(([key,label,,min,max,step,hint]) =>
                `<label for="sw-debug-${key}">${label}<input id="sw-debug-${key}" type="number" required min="${min}" max="${max}" step="${step}" value="${swDebugEscape(values[key])}"><span class="sw-note">${hint}</span></label>`).join('') +
            sideWhiteSwitches.map(([key,label]) =>
                `<label>${label}<input id="sw-debug-${key}" type="checkbox" ${values[key]?'checked':''}></label>`).join('');
    }
    async function loadSideWhiteDebugDefaults() {
        if (swDebugBusy) return;
        const message = document.getElementById('sw-debug-message');
        setSideWhiteDebugBusy(true);
        message.textContent = '載入中…';
        try {
            const query = swDebugRecordId ? '?record_id=' + encodeURIComponent(swDebugRecordId) : '';
            const response = await fetch('/api/debug/side-white' + query);
            const data = await response.json();
            if (!response.ok || data.error) throw new Error(data.error || '設定載入失敗');
            renderSideWhiteDebugParams(data.parameters);
            document.getElementById('sw-debug-rotate').checked = data.rotate_180;
            document.getElementById('sw-debug-show-all').checked = false;
            updateSideWhiteDebugCandidateDisplay();
            if (data.source) {
                const source = data.source;
                document.getElementById('sw-debug-path').value = source.image_path;
                document.getElementById('sw-debug-model').value = source.model_id;
                document.getElementById('sw-debug-path').readOnly = true;
                document.getElementById('sw-debug-model').readOnly = true;
                document.getElementById('sw-debug-source').textContent = `紀錄 #${source.record_id} · ${source.glass_id} · ${source.model_id} · 產品解析度 ${source.product_resolution.join(' × ')} ${source.resolution_warning || ''}`;
                const link = document.getElementById('sw-debug-record');
                link.href = '/record/' + Number(source.record_id) + '#side-white-result';
                link.hidden = false;
            }
            swDebugReady = true;
            message.textContent = '已載入設定。';
        } catch (error) {
            message.textContent = error.message;
        } finally { setSideWhiteDebugBusy(false); }
    }
    async function runSideWhiteDebug(event) {
        event.preventDefault();
        if (swDebugBusy || !swDebugReady || !event.target.reportValidity()) return;
        const parameters = Object.fromEntries(sideWhiteParamFields.map(([key]) => [key, document.getElementById('sw-debug-'+key).valueAsNumber]));
        sideWhiteSwitches.forEach(([key]) => parameters[key] = document.getElementById('sw-debug-'+key).checked);
        parameters.dust_mode = 'suppress';
        const body = {parameters, rotate_180:document.getElementById('sw-debug-rotate').checked};
        if (swDebugRecordId) body.record_id = swDebugRecordId;
        else {
            body.image_path = document.getElementById('sw-debug-path').value.trim();
            body.model_id = document.getElementById('sw-debug-model').value.trim();
        }
        const message = document.getElementById('sw-debug-message');
        setSideWhiteDebugBusy(true);
        document.getElementById('sw-debug-result').innerHTML = '';
        message.textContent = '檢測中…';
        try {
            const response = await fetch('/api/debug/side-white', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
            const data = await response.json();
            if (!response.ok || data.error) throw new Error(data.error || '側拍測試失敗');
            document.getElementById('sw-debug-result').innerHTML = data.result_html;
            updateSideWhiteDebugCandidateDisplay();
            message.textContent = data.payload.status === 'ERROR' || data.payload.status === 'NO_IMAGE'
                ? '測試未完成：' + (data.payload.reason || data.payload.status)
                : '測試完成。' + (data.resolution_warning || '');
        } catch (error) { message.textContent = error.message; }
        finally { setSideWhiteDebugBusy(false); }
    }
    document.addEventListener('DOMContentLoaded', () => {
        const query = new URLSearchParams(window.location.search);
        if (query.get('tab') === 'side-white') {
            switchTab('tab-side-white', document.querySelector('.tab-btn[onclick*="tab-side-white"]'));
        }
        document.getElementById('sw-debug-path').value = query.get('path') || '';
        document.getElementById('sw-debug-model').value = query.get('model_id') || '';
        loadSideWhiteDebugDefaults();
    });
