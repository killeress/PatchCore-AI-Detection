/* Side-camera analysis report based on saved detection results. */
window.sideStatistics = (() => {
    const root = document.getElementById('side-statistics');
    const form = document.getElementById('ss-filters');
    const message = document.getElementById('ss-message');
    const rows = document.getElementById('ss-rows');
    const state = {loaded: false, loading: false, group: 'with_csjbl', offset: 0, limit: 10, total: 0, filters: {}, summary: {}, conversionSummary: null, timeScope: null};
    const groups = {with_csjbl: 'AOI 有 CSJBL'};
    const fields = ['start_date', 'end_date', 'shift', 'machine_no', 'model_id', 'glass_id', 'status', 'conversion', 'tolerance'];
    const number = value => Number(value) || 0;

    const advancedFields = ['model_id', 'status', 'conversion', 'tolerance'];
    const escapeHtml = value => String(value).replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
    // Merged display categories; each maps to a comma-separated API filter value.
    const candidateParts = [
        {label: '保留候選', field: 'status', value: 'CANDIDATES', count: s => s.candidates, cls: 'c-info'},
        {label: '無保留候選', field: 'status', value: 'NO_CANDIDATES,FILTERED', count: s => s.clear, cls: ''},
        {label: '未完成檢測', field: 'status', value: 'NOT_RUN,NO_IMAGE,ERROR', count: s => s.unavailable, cls: 'c-pending',
            detail: () => '未執行、無側拍圖或檢測失敗'},
    ];
    const isActive = (group, field, value) => state.group === group && (state.filters[field] || '') === value;

    function chip(group, part, s, extraClass = '') {
        const count = number(part.count(s));
        const active = isActive(group, part.field, part.value);
        const title = part.detail ? part.detail(s) : `篩選明細：${part.label}`;
        return `<button type="button" class="ss-chip ${extraClass} ${count ? '' : 'is-zero'} ${active ? 'is-active' : ''}"
            data-filter-group="${group}" data-filter-field="${part.field}" data-filter-value="${part.value}"
            title="${escapeHtml(title)}（點擊篩選明細）" ${state.loading ? 'disabled' : ''}>${part.label}<b class="${part.cls}">${count}</b></button>`;
    }

    function renderOverview() {
        const all = Object.keys(groups).map(key => state.summary[key] || {});
        const sum = key => all.reduce((total, s) => total + number(s[key]), 0);
        document.getElementById('ss-overview').innerHTML = `
            <span>總筆數<b>${sum('total')}</b></span>
            <span>已完成檢測<b>${sum('candidates') + sum('clear')}</b></span>
            <span>未完成檢測<b class="c-pending">${sum('unavailable')}</b></span>`;
        const f = state.filters;
        const time = state.timeScope;
        const range = time ? `${time.mode === 'current' ? '當班 · ' : ''}${time.shift_name} ${time.time_range}`
            : (f.start_date === f.end_date ? f.start_date : `${f.start_date}～${f.end_date}`);
        const scope = [range,
            f.machine_no || '全部機台', f.model_id || '全部機種', '僅 AOI 有 CSJBL'];
        if (f.glass_id) scope.push(`玻璃 ID 含「${f.glass_id}」`);
        document.getElementById('ss-scope').textContent = state.loaded ? `總覽範圍：${scope.join(' · ')}` : '';
    }

    function renderSummary() {
        renderOverview();
        renderConversions();
        document.getElementById('ss-summary').innerHTML = Object.entries(groups).map(([key, label]) => {
            const s = state.summary[key] || {};
            const total = number(s.total);
            const candidates = candidateParts.filter(part => number(part.count(s)) || isActive(key, part.field, part.value));
            return `<div class="ss-card" role="button" tabindex="0" data-group="${key}" aria-pressed="${key === state.group}"
                    aria-disabled="${state.loading}" title="點擊切換下方明細為「${label}」">
                <div class="ss-card-head"><strong>${label}</strong><span>${total} 筆</span></div>
                <div class="ss-lines">
                    ${candidates.length ? `<div class="ss-line"><em>狀態</em>${candidates.map(part => chip(key, part, s)).join('')}</div>` : ''}
                </div></div>`;
        }).join('');
    }

    function renderConversions() {
        const target = document.getElementById('ss-conversion');
        const s = state.conversionSummary;
        target.hidden = !s;
        if (!s) { target.innerHTML = ''; return; }
        const pct = value => value == null ? '—' : `${number(value).toFixed(2)}%`;
        const rate = (value, denominator) => number(denominator) > 0
            ? `${((number(value) / number(denominator)) * 100).toLocaleString('zh-TW', {minimumFractionDigits: 1, maximumFractionDigits: 1})}%` : '—';
        const pp = value => value == null ? '—' : `${number(value).toFixed(2)} 個百分點`;
        const filterChip = (label, value, count) => chip('with_csjbl', {
            label, field: 'conversion', value, count: () => count, cls: 'c-info',
        }, s);
        const scope = s.scope, combined = scope.scenarios.combined;
        const formal = scope.formal_impact;
        const subset = s.csjbl, subsetFormal = subset.formal_impact;
        const scenarios = [
            ['mark', '只轉 MARK', 'eligible_mark'], ['omit', '只轉貼紙／表面異物', 'eligible_omit'],
            ['tape', '只轉易撕貼', 'eligible_tape'], ['clear', '只轉 AOI 位置無異常', 'eligible_clear'],
            ['combined', '四類全部轉 OK（含混合）', 'eligible'],
        ];
        target.innerHTML = `<div class="ss-card-head"><strong>側拍正式放行成效</strong><span>查詢範圍已正式轉 OK-s ${number(formal.converted)} 筆</span></div>
            <p class="ss-rate">整體排片率 <span>未套用側拍放行</span><strong>${rate(formal.before_ng, scope.denominator)}</strong> → <span>目前正式</span><strong class="c-ok">${rate(formal.after_ng, scope.denominator)}</strong>
                <span>下降 ${pp(formal.drop_pp)} · 排片筆數相對減少 ${pct(formal.relative_drop_pct)}</span></p>
            <p class="sw-note">未套用側拍放行的 NG＝目前 NG ${number(scope.ng)} + 正式 OK-s ${number(formal.converted)}＝${number(formal.before_ng)}；已減少排片 ${number(formal.converted)} 筆。</p>
            <p class="sw-note">分母：同班別／機台／機種／玻璃 ID 範圍的總投入 ${number(scope.denominator)} 筆（OK ${number(scope.ok)}、NG ${number(scope.ng)}、ERR ${number(scope.err)}、其他 ${number(scope.other) - number(scope.err)}）。
                OK 已包含 OK-s；目前正式排片率＝NG ÷ 總投入（含 ERR），與中控總覽相同。</p>
            <p class="sw-note">僅 CSJBL 範圍（未套用側拍放行 → 目前正式）：${rate(subsetFormal.before_ng, subset.denominator)} → ${rate(subsetFormal.after_ng, subset.denominator)}，下降 ${pp(subsetFormal.drop_pp)}（正式 OK-s ${number(subsetFormal.converted)} 筆，分母 ${number(subset.denominator)} 筆，含 ERR）。</p>
            <div class="ss-lines">
                <div class="ss-line"><em>CSJBL 點位分類（含已正式放行與歷史紀錄）</em></div>
                <div class="ss-line">${filterChip('MARK 可轉 OK', 'mark', s.mark.records)}<span>筆 · ${number(s.mark.points)} 點</span>
                    ${filterChip('貼紙／表面異物可轉 OK', 'omit', s.omit.records)}<span>筆 · ${number(s.omit.points)} 點</span>
                    ${filterChip('易撕貼可轉 OK', 'tape', s.tape.records)}<span>筆 · ${number(s.tape.points)} 點</span>
                    ${filterChip('AOI 位置無異常', 'clear', s.clear.records)}<span>筆 · ${number(s.clear.points)} 點</span>
                    ${filterChip('含多類', 'mixed', s.mixed_records)} 筆</div>
                <div class="ss-line">${filterChip('歷史 NG 可再整筆轉 OK', 'eligible', s.eligible_records)}
                    筆 ${filterChip('仍需保留', 'blocked', s.blocked_records)}<span>筆 · 正式已 OK（含 OK-s）${number(s.already_ok_records)} 筆</span></div>
            </div>
            ${number(combined.converted) > 0 ? `<h4>歷史 NG 額外試算（尚未正式放行）</h4>
                <p class="sw-note">目前正式 ${rate(scope.ng, scope.denominator)} → 試算後 ${rate(combined.remaining_ng, scope.denominator)}；另可減少排片 ${number(combined.converted)} 筆，下降 ${pp(combined.drop_pp)} · 排片筆數相對減少 ${pct(combined.relative_drop_pct)}。</p>
                <div class="sw-scroll"><table class="sw-table"><thead><tr><th>假設情境</th><th>另可減少排片</th><th>試算後排片率</th><th>額外下降幅度</th><th>排片筆數相對減少</th></tr></thead><tbody>
                    ${scenarios.map(([key, label, filter]) => { const r = scope.scenarios[key]; return `<tr>
                        <td>${label}</td><td>${filterChip('查看明細', filter, r.converted)} 筆</td><td>${rate(r.remaining_ng, scope.denominator)}</td>
                        <td>${pp(r.drop_pp)}</td><td>${pct(r.relative_drop_pct)}</td></tr>`; }).join('')}
                </tbody></table></div>
                <p class="sw-note">四類全轉後 NG＝${number(scope.ng)} − ${number(combined.converted)}＝${number(combined.remaining_ng)}；僅 CSJBL 範圍試算：${rate(subset.ng, subset.denominator)} → ${rate(subset.scenarios.combined.remaining_ng, subset.denominator)}，下降 ${pp(subset.scenarios.combined.drop_pp)}。</p>`
                : '<p class="sw-note">歷史 NG 額外試算：目前沒有可再整筆轉 OK 的 NG 紀錄。</p>'}
            <details><summary>統計與試算方式</summary><p class="sw-note">正式成效依已保存的 OK-s 判定計算，未套用側拍放行的排片率＝（目前 NG + 正式 OK-s）÷ 總投入；相對減少＝正式 OK-s ÷（目前 NG + 正式 OK-s）。
                歷史試算只計尚為 NG 的紀錄，OK-s 不會再扣一次。只有整筆正式 NG 的全部 CSJBL 點及 NG 原因均已涵蓋，才計入額外減少。
                四類筆數可能重疊，合併依檢測紀錄去重；同玻璃重送視為另次檢測。混合類型 ${number(s.eligible.mixed)} 筆需同時轉 OK，僅列入合併情境。
                已是 OK、仍有其他 NG、明細不足或已有缺陷／定位疑慮紀錄者不計入額外減少；OMIT 僅標示模式不算已可轉 OK。
                分母使用現存推論紀錄，包含其他 AOI 種類；明細維持僅顯示 CSJBL。明細篩選與換頁不改變統計與試算範圍。</p></details>`;
    }

    function syncAdvanced(forceOpen = false) {
        const advanced = document.getElementById('ss-advanced');
        const more = document.getElementById('ss-more');
        const changed = advancedFields.filter(key => {
            const element = form.elements[key];
            return element.value.trim() !== (element.defaultValue ?? '') && !(element.tagName === 'SELECT' && element.value === '');
        }).length;
        if (forceOpen) advanced.hidden = false;
        more.setAttribute('aria-expanded', String(!advanced.hidden));
        more.textContent = `更多篩選${changed ? `（${changed}）` : ''} ${advanced.hidden ? '▾' : '▴'}`;
    }

    function currentFilters() {
        return Object.fromEntries(fields.map(key => [key, form.elements[key].value.trim()]));
    }

    function syncShiftControls() {
        const current = form.elements.shift.value === 'current';
        for (const key of ['start_date', 'end_date']) form.elements[key].disabled = current;
    }

    function applyQuickFilter(group, field, value) {
        const toggledOff = isActive(group, field, value);
        // Use the applied scope, leaving draft fields untouched.
        load(group, 0, {...state.filters, [field]: toggledOff ? '' : value});
    }

    function pagination() {
        document.getElementById('ss-prev').disabled = state.loading || state.offset === 0;
        document.getElementById('ss-next').disabled = state.loading || state.offset + state.limit >= state.total;
        document.getElementById('ss-page').textContent = state.total
            ? `${groups[state.group]} · 共 ${state.total} 筆 · 第 ${Math.floor(state.offset / state.limit) + 1} / ${Math.ceil(state.total / state.limit)} 頁` : '共 0 筆';
        const labels = ['status', 'conversion'].filter(key => state.filters[key]).map(key =>
            Array.from(form.elements[key].options).find(option => option.value === state.filters[key]).textContent.trim());
        document.getElementById('ss-current-filter').textContent = state.loaded
            ? `目前查看：${groups[state.group]} · ${labels.join(' · ') || '全部明細'} · ${state.total} 筆` : '';
        const clear = document.getElementById('ss-clear');
        clear.hidden = labels.length === 0;
        clear.disabled = state.loading;
    }

    async function getData(filters, group, offset) {
        const query = new URLSearchParams({...filters, group, offset, limit: state.limit});
        const response = await fetch(`/api/ric/side-stats?${query}`);
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(data.error || '側拍統計載入失敗');
        return data;
    }

    async function load(group = state.group, offset = 0, filters = state.filters) {
        if (!(group in groups)) return;
        if (state.loading) return;
        state.loading = true;
        message.textContent = '載入側拍統計中…';
        form.querySelector('button[type=submit]').disabled = true;
        rows.inert = true;
        renderSummary(); pagination();
        try {
            const data = await getData(filters, group, offset);
            const appliedFilters = {...filters};
            if (data.time_scope) {
                appliedFilters.start_date = data.time_scope.start_date;
                appliedFilters.end_date = data.time_scope.end_date;
                if (filters.shift === 'current' && form.elements.shift.value === 'current') {
                    for (const key of ['start_date', 'end_date']) form.elements[key].value = appliedFilters[key];
                }
            }
            Object.assign(state, {loaded: true, group, offset, filters: appliedFilters, total: data.total, summary: data.summary,
                conversionSummary: data.conversion_summary || null, timeScope: data.time_scope || null});
            for (const key of ['status', 'conversion']) form.elements[key].value = state.filters[key] || '';
            syncAdvanced();
            rows.innerHTML = data.rows_html || '<div class="sw-empty">目前條件下沒有 AOI 有 CSJBL 的資料，請調整日期或篩選條件。</div>';
            message.textContent = '';
        } catch (error) {
            message.textContent = `${error.message}${state.loaded ? '；下方保留上次查詢資料。' : ''}`;
        } finally {
            state.loading = false;
            rows.inert = false;
            form.querySelector('button[type=submit]').disabled = false;
            renderSummary(); pagination();
        }
    }

    form.addEventListener('submit', event => {
        event.preventDefault();
        if (form.reportValidity()) load(state.group, 0, currentFilters());
    });
    form.addEventListener('change', () => { syncShiftControls(); syncAdvanced(); });
    document.getElementById('ss-more').addEventListener('click', () => {
        const advanced = document.getElementById('ss-advanced');
        advanced.hidden = !advanced.hidden;
        syncAdvanced();
    });
    document.getElementById('ss-clear').addEventListener('click', () => {
        load(state.group, 0, {...state.filters, status: '', conversion: ''});
    });
    root.addEventListener('click', event => {
        const filter = event.target.closest('[data-filter-field]');
        if (filter) {
            applyQuickFilter(filter.dataset.filterGroup, filter.dataset.filterField, filter.dataset.filterValue);
            return;
        }
        const card = event.target.closest('.ss-card[data-group]');
        if (card && !state.loading && card.dataset.group !== state.group) load(card.dataset.group, 0);
    });
    root.addEventListener('keydown', event => {
        const card = event.target.closest('.ss-card[data-group]');
        if (!card || event.target !== card || (event.key !== 'Enter' && event.key !== ' ')) return;
        event.preventDefault();
        if (!state.loading && card.dataset.group !== state.group) load(card.dataset.group, 0);
    });
    document.getElementById('ss-prev').addEventListener('click', () => load(state.group, Math.max(0, state.offset - state.limit)));
    document.getElementById('ss-next').addEventListener('click', () => load(state.group, state.offset + state.limit));

    function init() {
        if (state.loaded || state.loading) return;
        const now = new Date();
        const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
        form.elements.start_date.value = today; form.elements.end_date.value = today;
        form.elements.shift.value = 'current';
        state.filters = currentFilters();
        syncShiftControls(); syncAdvanced();
        load();
    }
    return {init};
})();
