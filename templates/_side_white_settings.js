    const sideWhiteLabels = {CANDIDATES:'發現候選',NO_CANDIDATES:'未發現候選（非 OK 判定）',NO_IMAGE:'無側拍白畫面',ERROR:'檢測失敗'};
    const sideWhiteReviews = {unreviewed:'待 Review',confirmed:'確認不良',false_positive:'過檢',missed:'漏檢',uncertain:'待確認'};
    let sideWhite = {rows:[],total:0,offset:0,limit:20,loading:false,loaded:false,error:'',status:'',review:'',machine_no:'',glass_id:new URLSearchParams(location.search).get('glass_id') || ''};
    const sideWhiteParamFields = [
        ['min_contrast_gray','最低局部反差',2.2,.1,255,.1,'灰階差；調低可找較淡異常，也會增加雜訊。'],
        ['noise_sigma_factor','雜訊門檻倍率',4.5,.1,100,.1,'倍；調高可抑制雜訊，也可能漏掉弱異常。'],
        ['min_area_px','最小候選面積',20,1,1000000,1,'側拍像素數；調高可濾掉碎點，也可能漏掉小白點。'],
        ['edge_margin_px','邊緣排除寬度',16,0,512,1,'側拍 px；調高會縮小近邊檢測範圍，0 表示不內縮。'],
    ];
    let sideWhiteParamDraft = null, sideWhiteParamSaving = false, sideWhiteParamMessage = '';

    function savedSideWhiteParams() {
        const defaults = Object.fromEntries(sideWhiteParamFields.map(([key,,value]) => [key,value]));
        const row = allParams.find(p => p.param_name === 'side_white_detection_params');
        return {...defaults,...parseJsonLoose(row ? row.param_value : null, defaults)};
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
            <p class="sw-note">四項一起儲存，套用後續新排程；每筆結果保存當時的參數。總開關仍在「推論設定」，目前為 <strong>${getPlainParamValue('side_white_detection_enabled','false')==='true'?'啟用':'關閉'}</strong>。</p>
            <fieldset class="sw-param-grid" ${sideWhiteParamSaving?'disabled':''}>${sideWhiteParamFields.map(([key,label,,min,max,step,hint]) => `<label for="sw-param-${key}">${label}<input id="sw-param-${key}" type="number" required min="${min}" max="${max}" step="${step}" value="${escapeAttr(values[key])}" oninput="editSideWhiteParam('${key}',this.value)"><span class="sw-note">${hint}</span></label>`).join('')}</fieldset>
            <p class="sw-note">實際檢測門檻取「最低局部反差」與「影像雜訊 × 倍率」較大者。像素單位以側拍原圖為準。</p>
            <button class="sw-btn" type="submit" ${sideWhiteParamSaving?'disabled':''}>${sideWhiteParamSaving?'儲存中…':'儲存側拍參數'}</button>
            <span id="sw-param-message" class="sw-note" role="status">${escapeHtml(sideWhiteParamMessage||(sideWhiteParamDraft?'尚未儲存':''))}</span>
        </form>`;
    }
    async function saveSideWhiteParams(event) {
        event.preventDefault();
        if (sideWhiteParamSaving || !event.target.reportValidity()) return;
        const values = Object.fromEntries(sideWhiteParamFields.map(([key]) => [key,document.getElementById('sw-param-'+key).valueAsNumber]));
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
    function sideWhiteSnapshot(p) {
        if (!p.parameters) return '此筆未保存參數快照。';
        return '當時參數：' + sideWhiteParamFields.map(([key,label]) => `${label} ${p.parameters[key]}`).join(' · ')
            + (p.threshold_gray == null ? '' : `；本次實際門檻 ${p.threshold_gray} 灰階。`);
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
            ['status','review','glass_id','machine_no','offset','limit'].forEach(k => query.set(k,String(sideWhite[k])));
            const res = await fetch(`/api/settings/side-white?${query}`);
            const data = await res.json();
            if (!res.ok || data.error) throw new Error(data.error || '側拍資料載入失敗');
            Object.assign(sideWhite,data,{loaded:true});
        } catch(e) { sideWhite.error = e.message; }
        finally { sideWhite.loading = false; updateSideWhitePane(); }
    }
    function searchSideWhite() {
        ['status','review','glass_id','machine_no'].forEach(k => sideWhite[k] = document.getElementById('sw-filter-'+k).value.trim());
        sideWhite.offset = 0; loadSideWhite();
    }
    function pageSideWhite(delta) {
        sideWhite.offset = Math.max(0,sideWhite.offset + delta * sideWhite.limit); loadSideWhite();
    }
    async function saveSideWhiteReview(id) {
        const button = document.getElementById('sw-save-'+id);
        const message = document.getElementById('sw-message-'+id);
        const decision = document.getElementById('sw-decision-'+id).value;
        const note = document.getElementById('sw-note-'+id).value;
        button.disabled = true; message.textContent = '儲存中…';
        try {
            const res = await fetch('/api/settings/side-white/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id,decision,note})});
            const data = await res.json();
            if (!res.ok || data.error) throw new Error(data.error || 'Review 儲存失敗');
            const row = sideWhite.rows.find(r => r.id === id);
            if (row) Object.assign(row,{review_decision:decision,review_note:note});
            message.textContent = '已儲存；最終判定維持不變。';
        } catch(e) { message.textContent = e.message; }
        finally { button.disabled = false; }
    }
    function renderSideWhitePane() {
        if (!SETTINGS_USER || !SETTINGS_USER.can_manage_accounts) return '';
        const options = (labels,selected) => Object.entries(labels).map(([v,t]) => `<option value="${escapeAttr(v)}" ${v===selected?'selected':''}>${escapeHtml(t)}</option>`).join('');
        const xy = point => Array.isArray(point) ? point.map(v => Number(v).toFixed(1)).join(', ') : '未映射';
        const kinds = {bright_dark_pair:'亮暗成對',bright_spot:'亮斑',dark_spot:'暗斑',line:'細長紋路'};
        const rows = sideWhite.rows.map(row => {
            const p = row.payload || {};
            const images = [['side','側拍候選'],['front','正拍映射（估算）']].filter(([kind]) => (p.artifacts||{})[kind]).map(([kind,label]) => `<figure><a href="/api/side-white/image?id=${row.id}&kind=${kind}" target="_blank" rel="noopener"><img loading="lazy" src="/api/side-white/image?id=${row.id}&kind=${kind}" alt="${label}" onerror="this.replaceWith(document.createTextNode('預覽已過期或遺失'))"></a><figcaption>${label}</figcaption></figure>`).join('');
            const candidates = (p.candidates||[]).map(c => `<tr><td>#${Number(c.id)}</td><td>${escapeHtml(kinds[c.kind]||c.kind)}</td><td>${xy(c.side_raw_xy)}</td><td>${xy(c.front_raw_xy)}</td><td>${Number(c.area_px)}</td><td>${Number(c.contrast_gray)}</td></tr>`).join('');
            return `<article class="sw-card">
                <div class="sw-head"><strong>${escapeHtml(row.glass_id)} · ${escapeHtml(row.machine_no)} · ${escapeHtml(row.model_id)}</strong><a href="/record/${row.record_id}#side-white-result" target="_blank" rel="noopener">記錄 #${row.record_id} →</a></div>
                <div class="sw-head"><span>${escapeHtml(sideWhiteLabels[row.status]||row.status)} · ${row.candidate_count} 個候選</span><span class="sw-badge">正式結果：${escapeHtml(row.ai_judgment)}</span></div>
                <p class="sw-note">${escapeHtml(row.request_time)} · ${escapeHtml(p.side_image||'')} · ${escapeHtml(p.algorithm||'')} · ${Number(p.processing_ms||0)} ms</p>
                <p class="sw-note">${escapeHtml(p.reason||'')} ${escapeHtml((p.mapping||{}).reason||'')}</p>
                <p class="sw-note">${escapeHtml(sideWhiteSnapshot(p))}</p>
                ${p.truncated?'<p class="sw-badge sw-warn">候選過多，僅顯示反差最高的 100 個。</p>':''}
                <p class="sw-note">面板裁切預覽：黃框為候選；藍框 MASK 為角落文字／貼紙排除區。</p><div class="sw-images">${images}</div>
                <details><summary>候選座標與局部亮暗差</summary><p class="sw-note">下表為原始 TIF 像素座標，左上角 (0,0)。${p.rotation_applied?'預覽已旋轉 180°。':''} 形態不代表已確認的缺陷成因。</p>
                    ${(p.artifacts||{}).residual?`<a href="/api/side-white/image?id=${row.id}&kind=residual" target="_blank" rel="noopener">開啟局部亮暗差圖</a>`:''}
                    <div class="sw-scroll"><table class="sw-table"><thead><tr><th>候選</th><th>形態</th><th>側拍 (X,Y)</th><th>正拍 (X,Y)，估算</th><th>面積 px</th><th>反差</th></tr></thead><tbody>${candidates||'<tr><td colspan="6">沒有候選座標</td></tr>'}</tbody></table></div>
                </details>
                <div class="sw-controls"><label>Review <select id="sw-decision-${row.id}">${options(sideWhiteReviews,row.review_decision)}</select></label><textarea id="sw-note-${row.id}" rows="2" maxlength="2000" aria-label="Review 備註" placeholder="記錄真異常、過檢原因或漏檢位置">${escapeHtml(row.review_note||'')}</textarea><button id="sw-save-${row.id}" class="sw-btn" onclick="saveSideWhiteReview(${row.id})">儲存 Review</button></div>
                <p id="sw-message-${row.id}" class="sw-note" role="status">${row.reviewed_at?escapeHtml(row.reviewed_by+' · '+row.reviewed_at):''}</p>
            </article>`;
        }).join('');
        return `<div id="pane-side-white" class="settings-tab-pane-block ${currentTab==='side-white'?'active':''}" style="grid-column:1/-1;">
            {% include "_side_white_styles.html" %}
            ${renderSideWhiteParams()}
            <div class="sw-card"><div class="sw-head"><h3>側拍白畫面 Review</h3><span class="sw-badge">觀察模式 · 不影響最終判定</span></div>
            <p class="sw-note">在「推論設定」啟用側拍白畫面檢測後，新紀錄會在背景產生候選。人工 Review 只保存標記，不改寫正式判定。正拍位置由同次拍攝的面板邊界估算，仍待同設備多點校正。</p>
            <div class="sw-controls"><label>玻璃 ID <input id="sw-filter-glass_id" value="${escapeAttr(sideWhite.glass_id)}" size="18"></label><label>機台 <input id="sw-filter-machine_no" value="${escapeAttr(sideWhite.machine_no)}" size="12"></label>
            <label>檢測 <select id="sw-filter-status">${options({'':'全部',...sideWhiteLabels},sideWhite.status)}</select></label><label>Review <select id="sw-filter-review">${options({'':'全部',...sideWhiteReviews},sideWhite.review)}</select></label><button class="sw-btn" onclick="searchSideWhite()" ${sideWhite.loading?'disabled':''}>查詢／重新整理</button></div>
            <p class="sw-note" role="status">${escapeHtml(sideWhite.error||(sideWhite.loading?'載入中…':`共 ${sideWhite.total} 筆 · 本頁 ${sideWhite.rows.length} 筆`))}</p></div>
            ${rows||'<div class="sw-empty">目前沒有符合條件的側拍結果。</div>'}
            <div class="sw-controls"><button class="sw-btn" onclick="pageSideWhite(-1)" ${sideWhite.offset===0||sideWhite.loading?'disabled':''}>上一頁</button><span>第 ${Math.floor(sideWhite.offset/sideWhite.limit)+1} 頁</span><button class="sw-btn" onclick="pageSideWhite(1)" ${sideWhite.offset+sideWhite.limit>=sideWhite.total||sideWhite.loading?'disabled':''}>下一頁</button></div>
        </div>`;
    }
