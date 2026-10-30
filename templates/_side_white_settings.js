    {% include "_side_white_params.js" %}
    let sideWhiteParamDraft = null, sideWhiteParamSaving = false, sideWhiteParamMessage = '';

    function savedSideWhiteParams() {
        const defaults = Object.fromEntries(sideWhiteParamFields.map(([key,,value]) => [key,value]));
        const row = allParams.find(p => p.param_name === 'side_white_detection_params');
        return {...defaults,...Object.fromEntries(sideWhiteSwitches.map(([k,,v]) => [k,v])),...parseJsonLoose(row ? row.param_value : null, defaults),dust_mode:'suppress'};
    }
    function editSideWhiteParam(key, value) {
        if (!sideWhiteParamDraft) sideWhiteParamDraft = savedSideWhiteParams();
        sideWhiteParamDraft[key] = value;
        sideWhiteParamMessage = '';
        const message = document.getElementById('sw-param-message');
        if (message) message.textContent = '尚未儲存';
    }
    function renderSideWhiteParams() {
        const values = sideWhiteParamDraft || savedSideWhiteParams();
        return `<form id="sw-params-form" class="sw-card" onsubmit="saveSideWhiteParams(event)">
            <div class="sw-head"><h3>側拍檢測參數</h3><span class="sw-badge">側拍放行 · OK-s</span></div>
            <p class="sw-note">儲存後套用新推論。側拍檢測：<strong>${getPlainParamValue('side_white_detection_enabled','false')==='true'?'啟用':'關閉'}</strong>（於「推論設定」切換）。</p>
            <fieldset class="sw-param-grid" ${sideWhiteParamSaving?'disabled':''}>${sideWhiteParamFields.map(([key,label,,min,max,step,hint]) => `<label for="sw-param-${key}">${label}<input id="sw-param-${key}" type="number" required min="${min}" max="${max}" step="${step}" value="${escapeAttr(values[key])}" oninput="editSideWhiteParam('${key}',this.value)"><span class="sw-note">${hint}</span></label>`).join('')}
            ${sideWhiteSwitches.map(([key,label]) => `<label>${label}<input id="sw-param-${key}" type="checkbox" ${values[key]?'checked':''} onchange="editSideWhiteParam('${key}',this.checked)"></label>`).join('')}
            </fieldset>
            <p class="sw-note">貼紙／灰塵過濾固定啟用，依側拍 SPINIGBI 證據屏蔽符合條件的候選。</p>
            <details><summary>參數說明</summary><p class="sw-note">檢測門檻取最低局部反差與「影像雜訊 × 倍率」較大者。不檢測區沿用 CV 邊緣檢測設定。CSJBL 顯示範圍只控制參考範圍；判定距離容錯內仍有保留候選即維持 NG。全部 CSJBL 及正式 NG 原因均可放行時，轉 OK-s 並回覆 testing OK；缺圖或映射失敗時保留原判定。</p></details>
            <button class="sw-btn" type="submit" ${sideWhiteParamSaving?'disabled':''}>${sideWhiteParamSaving?'儲存中…':'儲存側拍參數'}</button>
            <span id="sw-param-message" class="sw-note" role="status">${escapeHtml(sideWhiteParamMessage||(sideWhiteParamDraft?'尚未儲存':''))}</span>
        </form>`;
    }
    async function saveSideWhiteParams(event) {
        event.preventDefault();
        if (sideWhiteParamSaving || !event.target.reportValidity()) return;
        const values = Object.fromEntries(sideWhiteParamFields.map(([key]) => [key,document.getElementById('sw-param-'+key).valueAsNumber]));
        sideWhiteSwitches.forEach(([key]) => values[key] = document.getElementById('sw-param-'+key).checked);
        values.dust_mode = 'suppress';
        sideWhiteParamSaving = true; sideWhiteParamMessage = '';
        document.getElementById('sw-params-form').outerHTML = renderSideWhiteParams();
        try {
            const res = await fetch('/api/settings/update',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({param_name:'side_white_detection_params',new_value:values,reason:'更新側拍白畫面檢測參數'})});
            const data = await res.json();
            if (!res.ok || data.error) throw new Error(data.error || '側拍參數儲存失敗');
            const row = allParams.find(p => p.param_name === 'side_white_detection_params');
            const saved = {param_name:'side_white_detection_params',param_type:'dict',param_value:JSON.stringify(values),decoded_value:values};
            if (row) Object.assign(row,saved); else allParams.push(saved);
            sideWhiteParamDraft = null;
            sideWhiteParamMessage = '已儲存，套用新推論。';
            loadHistory();
        } catch(e) { sideWhiteParamMessage = e.message; }
        finally {
            sideWhiteParamSaving = false;
            const form = document.getElementById('sw-params-form');
            if (form) form.outerHTML = renderSideWhiteParams();
        }
    }
    function renderSideWhitePane() {
        if (!SETTINGS_USER || !SETTINGS_USER.can_manage_accounts) return '';
        const params = new URLSearchParams(location.search);
        const recordQuery = new URLSearchParams();
        ['glass_id','machine_no'].forEach(key => {
            const value = params.get(key);
            if (value) recordQuery.set(key,value);
        });
        const recordUrl = '/search' + (recordQuery.size ? `?${recordQuery}` : '');
        return `<div id="pane-side-white" class="settings-tab-pane-block ${currentTab==='side-white'?'active':''}" style="grid-column:1/-1;">
            {% include "_side_white_styles.html" %}
            ${renderSideWhiteParams()}
            <div class="sw-card">
                <div class="sw-head"><h3>側拍結果查閱</h3><span class="sw-badge">OK-s · 側拍轉 OK</span></div>
                <p class="sw-note">側拍統計提供 AOI 有 CSJBL 的結果查詢、分類統計與轉 OK 試算。</p>
                <div class="sw-controls"><a class="sw-btn" href="/ric?tab=side-statistics" target="_blank" rel="noopener">前往側拍統計 →</a></div>
                <p class="sw-note">沒有 CSJBL 的已保存結果也可由推論紀錄查閱。依玻璃 ID、機台或日期搜尋後，開啟單筆紀錄查看「側拍處理結果」。</p>
                <div class="sw-controls"><a class="sw-btn" href="${escapeAttr(recordUrl)}" target="_blank" rel="noopener">查詢推論紀錄（含無 CSJBL） →</a></div>
            </div>
        </div>`;
    }
