/* Read-only presentation of persisted logs. Unknown formats remain accessible verbatim. */
(function () {
    'use strict';
    const stages = {image_list: '建立影像清單', mark_primary: '主要定位標記偵測', mark_fallback: '備用定位標記偵測', mark_read_locate_recognize: '讀取及辨識定位標記', omit_read_check: '讀取灰塵檢查影像'};
    function describe(raw) {
        let m;
        if ((m = raw.match(/\[stage\] (\S+) elapsed_ms=([\d.]+)/))) return `${stages[m[1]] || (m[1].startsWith('aoi_raw_bounds:') ? '取得 AOI 原圖邊界：' + m[1].slice(15) : m[1])} · ${(Number(m[2]) / 1000).toFixed(2)} 秒`;
        if ((m = raw.match(/\[image-io\] source=(.*?) flags=.*?cache=(\w+).*?total_ms=([\d.]+)/))) return `影像讀取：${m[1]} · ${m[2] === 'hit' ? '使用快取' : '重新讀取'} · ${(Number(m[3]) / 1000).toFixed(2)} 秒`;
        if (raw.includes('no dot-matrix mark candidate found')) return '未偵測到點陣定位標記；詳見原始紀錄。';
        if (raw.includes('large-panel raw boundary skipped')) return '大面板邊界流程未採用，改用既有流程。';
        if ((m = raw.match(/reject non-linear (vertical|horizontal) edge: residual_p95=([\d.]+)px > ([\d.]+)px/))) return `${m[1] === 'vertical' ? '垂直' : '水平'}邊緣未通過直線檢查：偏差 ${m[2]} px，門檻 ${m[3]} px。`;
        if (raw.includes('BOMB_FORCE')) return '炸彈區域強制偵測未啟用，使用 AOI 報告座標（後續座標比對另行記錄）。';
        if (raw.includes('OMIT OK:')) return '灰塵檢查影像：系統回報檢查通過（OMIT OK）。';
        if ((m = raw.match(/\[v2\] (\w+): (?:bright_spot )?tiles=(\d+).*?NG=(\d+), infer ([\d.]+)s/))) return `${m[1]} · 檢測 ${m[2]} 個區塊，階段 NG ${m[3]} 個 · ${m[4]} 秒`;
        if ((m = raw.match(/BOMB match: (.+)/))) return `炸彈座標比對結果：${m[1].replace('UNKNOWN', '未定（UNKNOWN）')}`;
        if ((m = raw.match(/^(.*?)Tile@.*?AOI\(([^)]+)\).*?Score:([\d.]+).*?PER_REGION: (\d+)real\+(\d+)dust -> (\w+)/))) return `AOI ${m[2]} · 異常分數 ${m[3]} · 缺陷區域 ${m[4]}、灰塵區域 ${m[5]} · 階段判定 ${m[6]}`;
        return raw.replace(/^\[[^\]]+\]\s+(?:INFO|WARNING|ERROR|DEBUG|CRITICAL)\s+\[[^\]]+\]\s*/, '').replace(/^\[v2\]\s*/, '');
    }
    function parseLog(text) {
        const events = []; const specs = []; const timings = []; const alerts = new Map();
        let decision = '紀錄未提供可辨識的最終規格判定', overview = '', report = '';
        for (const raw of text.split(/\r?\n/).filter(line => line.trim())) {
            const message = describe(raw);
            const level = /\b(ERROR|CRITICAL)\b|Traceback/.test(raw) ? 'error' : /\bWARNING\b|no dot-matrix|large-panel raw boundary skipped|BOMB_FORCE/.test(raw) ? 'attention' : 'info';
            const time = (raw.match(/^\[\d{4}-\d\d-\d\d ([\d:]+)\]/) || [,''])[1];
            const event = {raw, message, level, time}; events.push(event);
            if (level !== 'info') {
                const key = message;
                if (!alerts.has(key)) alerts.set(key, {message, level, count: 0, lines: []});
                const alert = alerts.get(key); alert.count++; alert.lines.push(raw);
            }
            let m;
            if ((m = raw.match(/\[WITHIN_SPEC_INFERENCE\].*?結果=([^\s；]+)/))) {
                decision = ({not_within_spec: 'NG · 規格內檢查未通過', within_spec: '規格內檢查通過'})[m[1]] || `規格判定：${m[1]}`;
            }
            if (/\[WITHIN_SPEC_INFERENCE\].*最終判定 OK-i/.test(raw)) decision = 'OK-i · 符合規格內，已轉為放行';
            if (raw.includes('AOI Report: 解析到')) report = message.replace('AOI Report: ', '');
            if ((m = raw.match(/Panel (\S+) 總耗時 ([\d.]+)s/))) {
                overview = `面板 ${m[1]} · 推論流程 ${m[2]} 秒`;
                timings.length = 0;
                for (const t of raw.matchAll(/(前置|預處理|Tile準備|GPU推論|後處理\/收尾)=([\d.]+)s/g)) timings.push([({Tile準備:'檢測區塊準備',GPU推論:'推論／亮點檢測'})[t[1]] || t[1], `${t[2]} 秒`]);
            }
            if ((m = raw.match(/\[within-spec\].*decision_ms=([\d.]+)/))) timings.push(['後續規格判定', `${(Number(m[1])/1000).toFixed(2)} 秒`]);
            if (/^\s*- .*結果=/.test(raw)) {
                const parts = raw.trim().replace(/^-\s*/, '').split(/[：:]/);
                specs.push([parts.shift(), parts.join('：').replace(/；/g, '\n')]);
            }
        }
        return {events, specs, timings, alerts: [...alerts.values()], decision, overview, report};
    }
    if (typeof module !== 'undefined' && module.exports) module.exports = {parseLog, describe};
    if (typeof document === 'undefined') return;
    const root = document.querySelector('.il');
    if (!root) return;
    const find = selector => root.querySelector(selector);
    const raw = find('#inference-log').textContent;
    let parsed, initialized = false, pageSize = 100;
    function node(tag, text, className) {
        const el = document.createElement(tag);
        if (text !== undefined) el.textContent = text;
        if (className) el.className = className;
        return el;
    }
    function disclosure(parent, title, open = false) {
        const d = node('details'); d.open = open; d.append(node('summary', title)); parent.append(d); return d;
    }
    function table(parent, headings, rows) {
        const wrapper = node('div', undefined, 'il-table'), tbl = node('table'), head = node('thead'), body = node('tbody'), tr = node('tr');
        headings.forEach(h => { const th = node('th', h); th.scope = 'col'; tr.append(th); }); head.append(tr);
        rows.forEach(row => { const r = node('tr'); row.forEach(value => { const td = node('td'); String(value).split('\n').forEach((line, i) => { if(i) td.append(node('br')); const text = line.replace(/單Tile/g, '單區塊').replace(/\[NG\]/g, '【超標】').replace(/\[OK\]/g, '【合格】').replace(/結果=NG/g, '結果：未通過').replace(/結果=OK/g, '結果：通過'); td.append(node('span', text, /\[NG\]|結果=NG/.test(line) ? 'il-error' : '')); }); r.append(td); }); body.append(r); });
        tbl.append(head, body); wrapper.append(tbl); parent.append(wrapper);
    }
    function initialize() {
        if (initialized) return; initialized = true; parsed = parseLog(raw);
        const summary = find('#il-summary'); summary.append(node('strong', parsed.decision), node('p', parsed.overview || '耗時摘要未記錄'), node('p', parsed.report));
        if (parsed.specs.length) {
            const failed = parsed.specs.filter(row => /結果=NG/.test(row[1]));
            summary.append(node('p', `規格檢查 ${parsed.specs.length} 項，其中 ${failed.length} 項未通過；可展開查看尺寸、點數與 AOI 點檢出狀態。`));
        }
        const reasons = disclosure(summary, `規格判定明細（${parsed.specs.length} 項）`);
        if (parsed.specs.length) table(reasons, ['光源／檢查項目', '量測、門檻及結果'], parsed.specs);
        else reasons.append(node('p', '沒有可辨識的規格明細，請查看詳細紀錄或原始 Log。'));
        const alerts = disclosure(summary, `處理提醒（${parsed.alerts.reduce((n,a)=>n+a.count,0)} 筆）`);
        parsed.alerts.forEach(a => { const d = disclosure(alerts, `${a.message}（${a.count} 次）`); d.className = a.level === 'error' ? 'il-error' : 'il-warning'; d.append(node('pre', a.lines.join('\n'))); });
        if (!parsed.alerts.length) alerts.append(node('p', '沒有辨識到提醒或錯誤；完整內容仍可在原始 Log 查看。'));
        const times = disclosure(summary, '各階段耗時');
        if(parsed.timings.length) table(times, ['階段', '耗時'], parsed.timings); else times.append(node('p', '此紀錄未提供階段耗時摘要。'));
        summary.append(node('p', '摘要依紀錄文字整理；階段 NG 數量不等於最終缺陷數量。未辨識格式保留原文。', 'il-muted'));
        renderEvents();
    }
    function renderEvents() {
        const query = find('input').value.trim().toLowerCase(), level = find('select').value;
        const events = parsed.events.filter(e => (level === 'all' || (level === 'attention' ? e.level !== 'info' : e.level === 'error')) && (e.raw + ' ' + e.message).toLowerCase().includes(query));
        const target = find('.il-events'); target.replaceChildren();
        events.slice(0,pageSize).forEach(e => { const d = disclosure(target, `${e.time ? e.time + ' · ' : ''}${e.level === 'error' ? '錯誤 · ' : e.level === 'attention' ? '提醒 · ' : ''}${e.message}`); d.className = 'il-event-title ' + (e.level === 'error' ? 'il-error' : e.level === 'attention' ? 'il-warning' : ''); d.append(node('pre', e.raw)); });
        if (!events.length) target.append(node('p', '沒有符合條件的紀錄。'));
        find('.il-count').textContent = `${events.length} 筆，顯示 ${Math.min(pageSize,events.length)} 筆`;
        find('.il-more').hidden = events.length <= pageSize;
    }
    const tabs = [...root.querySelectorAll('[role=tab]')];
    function activate(tab) {
        tabs.forEach(t => { const active = t === tab; t.setAttribute('aria-selected', String(active)); t.tabIndex = active ? 0 : -1; find('#il-' + t.dataset.view).hidden = !active; });
    }
    tabs.forEach((tab,i) => {
        tab.addEventListener('click', () => activate(tab));
        tab.addEventListener('keydown', e => { let index; if(e.key==='ArrowRight') index=(i+1)%tabs.length; if(e.key==='ArrowLeft') index=(i+tabs.length-1)%tabs.length; if(e.key==='Home') index=0; if(e.key==='End') index=tabs.length-1; if(index!==undefined){e.preventDefault();activate(tabs[index]);tabs[index].focus();} });
    });
    function expand(on) { root.classList.toggle('il-expanded',on); const b=find('[data-action=expand]'); b.textContent=on?'還原':'放大'; b.setAttribute('aria-pressed',String(on)); }
    find('.il-toggle').addEventListener('click', () => { const panel=find('#log-panel'), open=panel.hidden; panel.hidden=!open; find('.il-toggle').setAttribute('aria-expanded',String(open)); find('.il-toggle').replaceChildren(document.createTextNode(`${open?'⌄':'›'}  推論 Log `),node('small',open?'（點擊收合）':'（點擊展開）')); if(open) initialize(); else expand(false); });
    find('input').addEventListener('input',()=>{pageSize=100;renderEvents();});
    find('select').addEventListener('change',()=>{pageSize=100;renderEvents();});
    find('.il-more').addEventListener('click',()=>{pageSize+=100;renderEvents();});
    find('[data-action=expand]').addEventListener('click',()=>expand(!root.classList.contains('il-expanded')));
    document.addEventListener('keydown',e=>{if(e.key==='Escape' && root.classList.contains('il-expanded')){expand(false);find('[data-action=expand]').focus();}});
    find('[data-action=download]').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([raw],{type:'text/plain;charset=utf-8'}));const a=node('a');a.href=url;a.download='inference-log.txt';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
    find('[data-action=copy]').addEventListener('click',async()=>{
        const previous=document.activeElement;
        try {
            try { if(!navigator.clipboard) throw new Error('fallback'); await navigator.clipboard.writeText(raw); }
            catch (_) { const area=node('textarea');area.value=raw;area.style.cssText='position:fixed;left:-9999px;top:0';root.append(area);try{area.select();if(!document.execCommand('copy'))throw new Error('copy');}finally{area.remove();previous.focus();} }
            find('.il-feedback').textContent='已複製完整原始紀錄';
        } catch (_) { find('.il-feedback').textContent='複製失敗，請使用下載或在原始 Log 選取文字。'; }
    });
})();
