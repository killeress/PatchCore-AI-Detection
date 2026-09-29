    const sideWhiteLabels = {CANDIDATES:'發現候選',FILTERED:'候選已排除或屬於炸彈',NO_CANDIDATES:'未發現候選（非 OK 判定）',NO_IMAGE:'無側拍白畫面',ERROR:'檢測失敗'};
    let sideWhite = {rows:[],total:0,offset:0,limit:20,loading:false,loaded:false,error:'',status:'',machine_no:'',glass_id:new URLSearchParams(location.search).get('glass_id') || ''};
    const sideWhiteParamFields = [
        ['min_contrast_gray','最低局部反差',2.2,.1,255,.1,'灰階差；調低可找較淡異常，也會增加雜訊。'],
        ['noise_sigma_factor','雜訊門檻倍率',4.5,.1,100,.1,'倍；調高可抑制雜訊，也可能漏掉弱異常。'],
        ['min_area_px','最小候選面積',20,1,1000000,1,'側拍像素數；調高可濾掉碎點，也可能漏掉小白點。'],
        ['edge_margin_px','邊緣排除寬度',16,0,512,1,'側拍 px；調高會縮小近邊檢測範圍，0 表示不內縮。'],
        ['dust_overlap_ratio','灰塵重疊門檻',.8,.01,1,.01,'0～1；尚未以側拍樣本驗證，預設僅標示疑似灰塵。'],
        ['mapping_margin_px','灰塵比對外擴',3,0,32,1,'正拍 px；只補償小幅對位誤差，不能取代校正。'],
        ['crop_padding_px','組合圖周邊範圍',32,0,512,1,'各原圖 px；顯示同一面板位置及周邊。'],
        ['bomb_tolerance_product_px','炸彈容許誤差',50,0,500,1,'產品 px；從正拍產品座標映射至側拍。'],
    ];
    const sideWhiteSwitches = [['apply_exclusions','套用產品不檢測區域',true],['bomb_check_enabled','比對正拍炸彈座標',true],['bomb_force_detection_enabled','炸彈區域強制檢測',false]];
    let sideWhiteParamDraft = null, sideWhiteParamSaving = false, sideWhiteParamMessage = '';

    function savedSideWhiteParams() {
        const defaults = Object.fromEntries(sideWhiteParamFields.map(([key,,value]) => [key,value]));
        const row = allParams.find(p => p.param_name === 'side_white_detection_params');
        return {...defaults,...Object.fromEntries(sideWhiteSwitches.map(([k,,v]) => [k,v])),dust_mode:'observe',...parseJsonLoose(row ? row.param_value : null, defaults)};
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
            <div class="sw-head"><h3>側拍檢測參數</h3><span class="sw-badge">只影響側拍候選</span></div>
            <p class="sw-note">設定一起儲存，套用後續新排程；每筆結果保存當時的參數。總開關仍在「推論設定」，目前為 <strong>${getPlainParamValue('side_white_detection_enabled','false')==='true'?'啟用':'關閉'}</strong>。</p>
            <fieldset class="sw-param-grid" ${sideWhiteParamSaving?'disabled':''}>${sideWhiteParamFields.map(([key,label,,min,max,step,hint]) => `<label for="sw-param-${key}">${label}<input id="sw-param-${key}" type="number" required min="${min}" max="${max}" step="${step}" value="${escapeAttr(values[key])}" oninput="editSideWhiteParam('${key}',this.value)"><span class="sw-note">${hint}</span></label>`).join('')}
            ${sideWhiteSwitches.map(([key,label]) => `<label>${label}<input id="sw-param-${key}" type="checkbox" ${values[key]?'checked':''} onchange="editSideWhiteParam('${key}',this.checked)"></label>`).join('')}
            <label>灰塵處理模式<select id="sw-param-dust_mode" onchange="editSideWhiteParam('dust_mode',this.value)">${Object.entries({off:'關閉',observe:'標示疑似灰塵（預設）',suppress:'啟用灰塵屏蔽'}).map(([v,label]) => `<option value="${v}" ${values.dust_mode===v?'selected':''}>${label}</option>`).join('')}</select></label>
            </fieldset>
            <p class="sw-note">實際檢測門檻取「最低局部反差」與「影像雜訊 × 倍率」較大者。原有檢測參數以側拍原圖為準；其他欄位依標示單位。區域來源沿用「CV 邊緣檢測」的產品不檢測區域；OMIT 特徵沿用灰塵設定。</p>
            <button class="sw-btn" type="submit" ${sideWhiteParamSaving?'disabled':''}>${sideWhiteParamSaving?'儲存中…':'儲存側拍參數'}</button>
            <span id="sw-param-message" class="sw-note" role="status">${escapeHtml(sideWhiteParamMessage||(sideWhiteParamDraft?'尚未儲存':''))}</span>
        </form>`;
    }
    async function saveSideWhiteParams(event) {
        event.preventDefault();
        if (sideWhiteParamSaving || !event.target.reportValidity()) return;
        const values = Object.fromEntries(sideWhiteParamFields.map(([key]) => [key,document.getElementById('sw-param-'+key).valueAsNumber]));
        sideWhiteSwitches.forEach(([key]) => values[key] = document.getElementById('sw-param-'+key).checked);
        values.dust_mode = document.getElementById('sw-param-dust_mode').value;
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
            sideWhiteParamMessage = '已儲存，後續新排程套用；既有結果與正式判定保持不變。';
            loadHistory();
        } catch(e) { sideWhiteParamMessage = e.message; }
        finally {
            sideWhiteParamSaving = false;
            const form = document.getElementById('sw-params-form');
            if (form) form.outerHTML = renderSideWhiteParams();
        }
    }
    function updateSideWhitePane() {
        const pane = document.getElementById('pane-side-white');
        if (pane) pane.outerHTML = renderSideWhitePane();
    }
    async function loadSideWhite() {
        if (!SETTINGS_USER || !SETTINGS_USER.can_manage_accounts || sideWhite.loading) return;
        sideWhite.loading = true; sideWhite.error = ''; updateSideWhitePane();
        try {
            const query = new URLSearchParams();
            ['status','glass_id','machine_no','offset','limit'].forEach(k => query.set(k,String(sideWhite[k])));
            const res = await fetch(`/api/settings/side-white?${query}`);
            const data = await res.json();
            if (!res.ok || data.error) throw new Error(data.error || '側拍資料載入失敗');
            Object.assign(sideWhite,data,{loaded:true});
        } catch(e) { sideWhite.error = e.message; }
        finally { sideWhite.loading = false; updateSideWhitePane(); }
    }
    function searchSideWhite() {
        ['status','glass_id','machine_no'].forEach(k => sideWhite[k] = document.getElementById('sw-filter-'+k).value.trim());
        sideWhite.offset = 0; loadSideWhite();
    }
    function pageSideWhite(delta) {
        sideWhite.offset = Math.max(0,sideWhite.offset + delta * sideWhite.limit); loadSideWhite();
    }
    function renderSideWhitePane() {
        if (!SETTINGS_USER || !SETTINGS_USER.can_manage_accounts) return '';
        const options = (labels,selected) => Object.entries(labels).map(([v,t]) => `<option value="${escapeAttr(v)}" ${v===selected?'selected':''}>${escapeHtml(t)}</option>`).join('');
        const rows = sideWhite.rows.map(row => `<article>
            <div class="sw-head"><strong>${escapeHtml(row.glass_id)} · ${escapeHtml(row.machine_no)} · ${escapeHtml(row.model_id)}</strong><a href="/record/${Number(row.record_id)}#side-white-result" target="_blank" rel="noopener">記錄 #${Number(row.record_id)} →</a></div>
            <p class="sw-note">${escapeHtml(row.request_time)} · 正式結果：${escapeHtml(row.ai_judgment)}</p>
            ${row.result_html || '<p>結果內容無法顯示，請開啟記錄。</p>'}
        </article>`).join('');
        return `<div id="pane-side-white" class="settings-tab-pane-block ${currentTab==='side-white'?'active':''}" style="grid-column:1/-1;">
            {% include "_side_white_styles.html" %}
            ${renderSideWhiteParams()}
            <div class="sw-card"><div class="sw-head"><h3>側拍白畫面結果</h3><span class="sw-badge">觀察模式 · 不影響最終判定</span></div>
            <p class="sw-note">在「推論設定」啟用側拍白畫面檢測後，新紀錄會在背景產生候選。此頁只顯示自動處理結果，不需人工操作；正式判定維持不變。正拍位置由同次拍攝的面板邊界估算，仍待同設備多點校正。</p>
            <div class="sw-controls"><label>玻璃 ID <input id="sw-filter-glass_id" value="${escapeAttr(sideWhite.glass_id)}" size="18"></label><label>機台 <input id="sw-filter-machine_no" value="${escapeAttr(sideWhite.machine_no)}" size="12"></label>
            <label>檢測 <select id="sw-filter-status">${options({'':'全部',...sideWhiteLabels},sideWhite.status)}</select></label><button class="sw-btn" onclick="searchSideWhite()" ${sideWhite.loading?'disabled':''}>查詢／重新整理</button></div>
            <p class="sw-note" role="status">${escapeHtml(sideWhite.error||(sideWhite.loading?'載入中…':`共 ${sideWhite.total} 筆 · 本頁 ${sideWhite.rows.length} 筆`))}</p></div>
            ${rows||'<div class="sw-empty">目前沒有符合條件的側拍結果。</div>'}
            <div class="sw-controls"><button class="sw-btn" onclick="pageSideWhite(-1)" ${sideWhite.offset===0||sideWhite.loading?'disabled':''}>上一頁</button><span>第 ${Math.floor(sideWhite.offset/sideWhite.limit)+1} 頁</span><button class="sw-btn" onclick="pageSideWhite(1)" ${sideWhite.offset+sideWhite.limit>=sideWhite.total||sideWhite.loading?'disabled':''}>下一頁</button></div>
        </div>`;
    }
