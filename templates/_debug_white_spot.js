const whiteSpotDebug = (() => {
    const presets = {{ white_spot_presets|default({})|tojson }};
    const presetDescriptions = {
        recommended: '建議先從這組開始：壓低細碎背景，讓比較寬、淡的白斑容易看見。',
        balanced: '左右去紋比較輕。較能保留白丸的邊緣與明暗差，背景顆粒也會多一些。',
        isotropic: '左右、上下使用相同強度。可當比較組，觀察分開調整兩個方向有沒有更好。',
        baseline: '回到前一次較弱的平滑設定，方便比較白丸是否被新的設定抹淡。',
        custom: '目前是自訂設定。建議每次只改一項，再按預覽，比較白丸和背景的變化。',
    };
    const el = id => document.getElementById(id);
    let running = false;
    let lastResult = null;
    let lastPath = '';
    const numberOrNull = input => input.value.trim() === '' ? null : Number(input.value);
    const format = value => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '—';

    function markStale() {
        if (!lastResult) return;
        el('ws-message').textContent = '參數或中心已變更；下方為上次結果，請重新預覽或執行 PC。';
        el('ws-message').className = 'ws-note ws-warning';
    }

    function applyPreset(key) {
        el('ws-preset-help').textContent = presetDescriptions[key] || presetDescriptions.custom;
        const preset = presets[key];
        if (!preset) return;
        document.querySelectorAll('[data-ws-param]').forEach(input => {
            input.value = preset.params[input.dataset.wsParam];
        });
        markStale();
    }

    function copyPath() {
        const source = el('image-path-input');
        if (source && source.value.trim()) {
            el('ws-path').value = source.value.trim();
            el('ws-center-x').removeAttribute('max');
            el('ws-center-y').removeAttribute('max');
            markStale();
        } else {
            el('ws-message').textContent = '請先填寫圖片檔案路徑。';
        }
    }

    function setImage(id, url) {
        el(id).src = url;
        const link = el(id + '-link');
        if (link) link.href = url;
    }

    function renderPC(pc) {
        const rows = el('ws-pc-rows');
        rows.replaceChildren();
        el('ws-pc-scores').hidden = true;
        el('ws-pc-images').hidden = true;
        for (const key of ['original', 'enhanced']) el(`ws-${key}-pc-card`).hidden = true;
        const message = el('ws-pc-message');
        message.className = 'ws-note';
        if (pc.status === 'not_requested') {
            message.textContent = '尚未執行 PC；可按「強化 + PC 檢測」比較模型反應。';
            return;
        }
        if (!pc.variants) {
            message.className = 'ws-note ws-warning';
            message.textContent = pc.message || 'PC 模型無法使用，強化圖仍可檢視。';
            return;
        }
        const range = pc.heatmap_display_range || [0, 0];
        message.textContent = `${pc.model.name}；共用熱圖色階 ${format(range[0])}～${format(range[1])}`;
        el('ws-pc-scores').hidden = false;
        for (const [key, title] of [['original', '原圖'], ['enhanced', 'DoG 強化圖']]) {
            const result = pc.variants[key];
            if (!result) continue;
            const row = document.createElement('tr');
            const verdict = result.status === 'error' || result.status === 'no_heatmap'
                ? result.message
                : result.roi_passes_threshold === true ? '中心達門檻' : result.roi_passes_threshold === false ? '中心未達門檻' : '無法判斷';
            for (const text of [title, format(result.tile_score), format(result.roi_score), format(pc.threshold), verdict]) {
                const td = document.createElement('td');
                td.textContent = text;
                row.appendChild(td);
            }
            rows.appendChild(row);
            if (result.heatmap_url) {
                setImage(`ws-${key}-pc`, result.heatmap_url);
                el(`ws-${key}-pc-card`).hidden = false;
                el('ws-pc-images').hidden = false;
            }
        }
    }

    function render(data) {
        lastResult = data;
        lastPath = el('ws-path').value.trim();
        el('ws-results').hidden = false;
        el('ws-center-x').value = data.center[0];
        el('ws-center-y').value = data.center[1];
        el('ws-center-x').max = data.image_size[0] - 1;
        el('ws-center-y').max = data.image_size[1] - 1;
        el('ws-radius').max = Math.floor(data.tile_size / 2);
        setImage('ws-overview', data.urls.overview);
        setImage('ws-original', data.urls.original);
        setImage('ws-enhanced', data.urls.enhanced);
        const [x1,y1,x2,y2] = data.focus_bounds;
        for (const key of ['original','enhanced']) {
            const box = el(`ws-${key}-roi`);
            Object.assign(box.style, {left:`${100*x1/data.tile_size}%`, top:`${100*y1/data.tile_size}%`,
                width:`${100*(x2-x1)/data.tile_size}%`, height:`${100*(y2-y1)/data.tile_size}%`});
        }
        const p = data.params;
        el('ws-summary').textContent = `${data.image_name} · ${data.image_size.join(' × ')} px · 中心 (${data.center.join(', ')}) · 切塊 ${data.tile_size}px · ${data.rotation_applied ? '已旋轉 180°' : '未旋轉'} · 左右／上下去紋 ${p.signal_sigma_x}／${p.signal_sigma_y} · 背景參考範圍 ${p.background_sigma_x}／${p.background_sigma_y} · 放大倍數 ${p.gain} · 基準亮度 ${p.offset} · 全黑或全白像素 ${(100*data.clipped_fraction).toFixed(2)}% · 強化耗時 ${data.preprocess_ms.toFixed(0)} ms`;
        renderPC(data.pc);
    }

    async function run(runPC, event) {
        if (event) event.preventDefault();
        if (running || !el('ws-form').reportValidity()) return;
        const cx = numberOrNull(el('ws-center-x'));
        const cy = numberOrNull(el('ws-center-y'));
        if ((cx === null) !== (cy === null)) {
            el('ws-message').textContent = '中心 X、Y 請同時填寫，或同時留空。';
            el('ws-message').className = 'ws-note ws-error';
            return;
        }
        const params = {};
        document.querySelectorAll('[data-ws-param]').forEach(input => { params[input.dataset.wsParam] = Number(input.value); });
        const payload = {image_path:el('ws-path').value.trim(), center_x:cx, center_y:cy,
            radius_px:Number(el('ws-radius').value), zone:el('ws-zone').value,
            threshold:numberOrNull(el('ws-threshold')), params, run_patchcore:runPC};
        running = true;
        el('ws-fields').disabled = true;
        el('ws-results').hidden = true;
        el('ws-message').className = 'ws-note';
        el('ws-message').textContent = runPC ? '正在強化影像並比較原圖／DoG 的 PC 反應…' : '正在計算 DoG 強化圖…';
        try {
            const response = await fetch('/api/debug/white-spot', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
            const data = await response.json();
            if (!response.ok || data.error || !data.success) throw new Error(data.error || '白丸檢測未完成');
            render(data);
            el('ws-message').textContent = `處理完成，共 ${(data.total_ms/1000).toFixed(2)} 秒。`;
        } catch (error) {
            el('ws-message').className = 'ws-note ws-error';
            el('ws-message').textContent = error.message || '白丸檢測失敗';
        } finally {
            running = false;
            el('ws-fields').disabled = false;
        }
    }

    function pickCenter(event) {
        if (running || !lastResult) return;
        if (el('ws-path').value.trim() !== lastPath) {
            el('ws-message').textContent = '路徑已變更，請先重新預覽後再點選中心。';
            return;
        }
        const rect = event.currentTarget.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        const [width,height] = lastResult.image_size;
        el('ws-center-x').value = Math.max(0,Math.min(width-1,Math.floor((event.clientX-rect.left)*width/rect.width)));
        el('ws-center-y').value = Math.max(0,Math.min(height-1,Math.floor((event.clientY-rect.top)*height/rect.height)));
        markStale();
    }

    document.querySelectorAll('#ws-fields input, #ws-zone').forEach(input => {
        input.addEventListener('input', () => {
            if (input.dataset.wsParam) {
                el('ws-preset').value = 'custom';
                el('ws-preset-help').textContent = presetDescriptions.custom;
            }
            if (input.id === 'ws-path') {
                el('ws-center-x').removeAttribute('max');
                el('ws-center-y').removeAttribute('max');
            }
            markStale();
        });
    });
    window.addEventListener('DOMContentLoaded', () => {
        el('ws-preset-help').textContent = presetDescriptions[el('ws-preset').value] || presetDescriptions.custom;
        const query = new URLSearchParams(window.location.search);
        if (query.get('path')) el('ws-path').value = query.get('path');
        for (const axis of ['x','y']) {
            const value = query.get(`image_${axis}`);
            if (value !== null) el(`ws-center-${axis}`).value = value;
        }
        if (query.get('tab') === 'white-spot') {
            const button = document.querySelector('.tab-btn[onclick*="tab-white-spot"]');
            if (button) switchTab('tab-white-spot',button);
        }
    });
    return {run, applyPreset, copyPath, pickCenter};
})();
