/* Side-camera validation report. All changes are observation annotations. */
window.sideStatistics = (() => {
    const root = document.getElementById('side-statistics');
    const form = document.getElementById('ss-filters');
    const message = document.getElementById('ss-message');
    const rows = document.getElementById('ss-rows');
    const state = {loaded: false, loading: false, group: 'with_csjbl', offset: 0, limit: 10, total: 0, filters: {}, summary: {}};
    const groups = {with_csjbl: 'AOI 有 CSJBL', without_csjbl: 'AOI 無 CSJBL'};
    const fields = ['start_date', 'end_date', 'machine_no', 'model_id', 'glass_id', 'status', 'review', 'tolerance'];
    const number = value => Number(value) || 0;

    const advancedFields = ['model_id', 'status', 'review', 'tolerance'];
    const escapeHtml = value => String(value).replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
    // Merged display categories; each maps to a comma-separated API filter value.
    const candidateParts = [
        {label: '保留候選', field: 'status', value: 'CANDIDATES', count: s => s.candidates, cls: 'c-info'},
        {label: '無保留候選', field: 'status', value: 'NO_CANDIDATES,FILTERED', count: s => s.clear, cls: ''},
        {label: '未完成檢測', field: 'status', value: 'NOT_RUN,NO_IMAGE,ERROR', count: s => s.unavailable, cls: 'c-pending',
            detail: () => '未執行、無側拍圖或檢測失敗'},
    ];
    const resultParts = [
        {label: '人工確認正確', field: 'review', value: 'confirmed,correct_clear', cls: 'c-ok', bar: 'bg-ok',
            count: s => number(s.confirmed) + number(s.correct_clear),
            detail: s => `檢出正確 ${number(s.confirmed)}／無缺陷確認 ${number(s.correct_clear)}`},
        {label: '人工確認錯誤', field: 'review', value: 'false_positive,missed', cls: 'c-bad', bar: 'bg-bad',
            count: s => number(s.false_positive) + number(s.missed),
            detail: s => `系統過檢 ${number(s.false_positive)}／系統漏檢 ${number(s.missed)}`},
        {label: '不計入正確率', field: 'review', value: 'mapping_error,uncertain', cls: 'c-muted', bar: 'bg-muted',
            count: s => number(s.mapping_error) + number(s.uncertain),
            detail: s => `座標映射不正確 ${number(s.mapping_error)}／無法確認 ${number(s.uncertain)}`},
    ];
    const accuracyText = value => value == null ? '—' : `${number(value).toFixed(1)}%`;
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
        const verified = sum('verified');
        const correct = sum('confirmed') + sum('correct_clear');
        const accuracy = verified ? Math.round(1000 * correct / verified) / 10 : null;
        document.getElementById('ss-overview').innerHTML = `
            <span>總筆數<b>${sum('total')}</b></span>
            <span>待驗證<b class="c-pending">${sum('unreviewed')}</b></span>
            <span>有效驗證<b>${verified}</b></span>
            <span title="人工確認正確 ÷ 有效驗證，兩組合計">已驗證正確率<b class="${accuracy == null ? '' : 'c-ok'}">${accuracyText(accuracy)}</b></span>`;
        const f = state.filters;
        const scope = [f.start_date === f.end_date ? f.start_date : `${f.start_date}～${f.end_date}`,
            f.machine_no || '全部機台', f.model_id || '全部機種'];
        if (f.glass_id) scope.push(`玻璃 ID 含「${f.glass_id}」`);
        document.getElementById('ss-scope').textContent = state.loaded ? `總覽範圍：${scope.join(' · ')}` : '';
    }

    function renderSummary() {
        renderOverview();
        document.getElementById('ss-summary').innerHTML = Object.entries(groups).map(([key, label]) => {
            const s = state.summary[key] || {};
            const total = number(s.total);
            const reviewed = Math.max(0, total - number(s.unreviewed));
            const pct = value => total ? (100 * value / total) : 0;
            const progress = total ? Math.round(pct(reviewed)) : 0;
            const pendingActive = isActive(key, 'review', 'unreviewed');
            const candidates = candidateParts.filter(part => number(part.count(s)) || isActive(key, part.field, part.value));
            const results = reviewed || resultParts.some(part => isActive(key, part.field, part.value))
                ? `<div class="ss-line"><em>結果</em>${resultParts.map(part => chip(key, part, s)).join('')}</div>` : '';
            return `<div class="ss-card" role="button" tabindex="0" data-group="${key}" aria-pressed="${key === state.group}"
                    aria-disabled="${state.loading}" title="點擊切換下方明細為「${label}」">
                <div class="ss-card-head"><strong>${label}</strong><span>${total} 筆</span></div>
                <div class="ss-kpis">
                    <button type="button" class="ss-kpi ${pendingActive ? 'is-active' : ''}" data-filter-group="${key}" data-filter-field="review"
                        data-filter-value="unreviewed" title="點擊篩選待驗證明細" ${state.loading ? 'disabled' : ''}>
                        <small>待驗證</small><b class="${number(s.unreviewed) ? 'c-pending' : 'c-muted'}">${number(s.unreviewed)}</b></button>
                    <div class="ss-kpi"><small>有效驗證</small><b>${number(s.verified)}</b></div>
                    <div class="ss-kpi" title="判定正確 ÷（判定正確＋判定錯誤）；待驗證與不計入不列入分母">
                        <small>正確率</small><b class="${s.accuracy == null ? 'c-muted' : 'c-ok'}">${accuracyText(s.accuracy)}</b></div>
                </div>
                <div class="ss-bar" title="處理進度：含不計入正確率的紀錄">${resultParts.map(part =>
                    `<i class="${part.bar}" style="width:${pct(part.count(s))}%"></i>`).join('')}</div>
                <div class="ss-bar-caption">已處理 ${reviewed} / ${total}（${progress}%）</div>
                <div class="ss-lines">
                    ${candidates.length ? `<div class="ss-line"><em>候選</em>${candidates.map(part => chip(key, part, s)).join('')}</div>` : ''}
                    ${results}
                </div></div>`;
        }).join('');
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

    function applyQuickFilter(group, field, value) {
        const toggledOff = isActive(group, field, value);
        // Use the applied scope, leaving draft fields and cancelled queries untouched.
        load(group, 0, {...state.filters, [field]: toggledOff ? '' : value});
    }

    function pagination() {
        document.getElementById('ss-prev').disabled = state.loading || state.offset === 0;
        document.getElementById('ss-next').disabled = state.loading || state.offset + state.limit >= state.total;
        document.getElementById('ss-page').textContent = state.total
            ? `${groups[state.group]} · 共 ${state.total} 筆 · 第 ${Math.floor(state.offset / state.limit) + 1} / ${Math.ceil(state.total / state.limit)} 頁` : '共 0 筆';
        const labels = ['status', 'review'].filter(key => state.filters[key]).map(key =>
            Array.from(form.elements[key].options).find(option => option.value === state.filters[key]).textContent.trim());
        document.getElementById('ss-current-filter').textContent = state.loaded
            ? `目前查看：${groups[state.group]} · ${labels.join(' · ') || '全部明細'} · ${state.total} 筆` : '';
        const clear = document.getElementById('ss-clear');
        clear.hidden = labels.length === 0;
        clear.disabled = state.loading;
    }

    function mayReplaceRows() {
        if (rows.querySelector('[data-saving="true"]')) {
            message.textContent = '驗證結果儲存中，請稍候。';
            return false;
        }
        return !rows.querySelector('[data-dirty="true"]') || window.confirm('有尚未儲存的驗證內容，確定放棄並重新查詢？');
    }

    async function getData(filters, group, offset, summaryOnly = false) {
        const query = new URLSearchParams({...filters, group, offset, limit: state.limit});
        if (summaryOnly) query.set('summary_only', '1');
        const response = await fetch(`/api/ric/side-stats?${query}`);
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(data.error || '側拍統計載入失敗');
        return data;
    }

    async function load(group = state.group, offset = 0, filters = state.filters) {
        if (state.loading || !mayReplaceRows()) return;
        state.loading = true;
        message.textContent = '載入側拍統計中…';
        form.querySelector('button[type=submit]').disabled = true;
        rows.inert = true;
        renderSummary(); pagination();
        try {
            const data = await getData(filters, group, offset);
            Object.assign(state, {loaded: true, group, offset, filters: {...filters}, total: data.total, summary: data.summary});
            for (const key of ['status', 'review']) form.elements[key].value = state.filters[key] || '';
            syncAdvanced();
            rows.innerHTML = data.rows_html || '<div class="sw-empty">此分組在目前條件下沒有資料，可切換另一分組或調整日期。</div>';
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

    async function saveReview(reviewForm) {
        if (reviewForm.dataset.saving === 'true' || !reviewForm.reportValidity()) return;
        const input = new FormData(reviewForm);
        const feedback = reviewForm.querySelector('[data-review-state]');
        if (rows.querySelector('[data-saving="true"]')) {
            feedback.textContent = '其他驗證結果儲存中，請稍候再儲存此筆。';
            return;
        }
        reviewForm.dataset.saving = 'true';
        Array.from(reviewForm.elements).forEach(element => element.disabled = true);
        feedback.textContent = '儲存中…';
        feedback.classList.remove('ss-saved');
        try {
            const response = await fetch('/api/ric/side-stats/review', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({id: Number(reviewForm.dataset.resultId), decision: input.get('decision'),
                    note: input.get('note'), reviewer: input.get('reviewer'), expected_reviewed_at: input.get('expected_reviewed_at')}),
            });
            const data = await response.json();
            if (!response.ok || data.error) throw new Error(data.error || '儲存失敗');
            reviewForm.elements.expected_reviewed_at.value = data.reviewed_at;
            reviewForm.elements.reviewer.value = data.reviewed_by;
            reviewForm.elements.note.value = data.review_note;
            reviewForm.dataset.dirty = 'false';
            const article = reviewForm.closest('article');
            article.querySelector('[data-review-label]').textContent = reviewForm.elements.decision.selectedOptions[0].textContent;
            article.querySelector('[data-review-dirty]').hidden = true;
            feedback.textContent = `已儲存 · ${data.reviewed_by} · ${data.reviewed_at}`;
            feedback.classList.add('ss-saved');
            try {
                const refreshed = await getData(state.filters, state.group, state.offset, true);
                state.summary = refreshed.summary; state.total = refreshed.total;
                renderSummary(); pagination();
                message.textContent = `驗證已儲存，總覽已更新。${state.filters.review ? '明細暫時保留；重新查詢可更新清單。' : ''}`;
            } catch (error) {
                message.textContent = `驗證已儲存；統計更新失敗，請重新查詢：${error.message}`;
            }
        } catch (error) {
            feedback.textContent = error.message;
            reviewForm.dataset.dirty = 'true';
        } finally {
            reviewForm.dataset.saving = 'false';
            Array.from(reviewForm.elements).forEach(element => element.disabled = false);
        }
    }

    form.addEventListener('submit', event => {
        event.preventDefault();
        if (form.reportValidity()) load(state.group, 0, currentFilters());
    });
    form.addEventListener('change', () => syncAdvanced());
    document.getElementById('ss-more').addEventListener('click', () => {
        const advanced = document.getElementById('ss-advanced');
        advanced.hidden = !advanced.hidden;
        syncAdvanced();
    });
    document.getElementById('ss-clear').addEventListener('click', () => {
        load(state.group, 0, {...state.filters, status: '', review: ''});
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
    rows.addEventListener('submit', event => {
        if (!event.target.matches('.ss-review')) return;
        event.preventDefault();
        saveReview(event.target);
    });
    rows.addEventListener('input', event => {
        const reviewForm = event.target.closest('.ss-review');
        if (!reviewForm) return;
        reviewForm.dataset.dirty = 'true';
        reviewForm.closest('article').querySelector('[data-review-dirty]').hidden = false;
        const feedback = reviewForm.querySelector('[data-review-state]');
        feedback.textContent = '尚未儲存'; feedback.classList.remove('ss-saved');
    });
    document.getElementById('ss-prev').addEventListener('click', () => load(state.group, Math.max(0, state.offset - state.limit)));
    document.getElementById('ss-next').addEventListener('click', () => load(state.group, state.offset + state.limit));
    window.addEventListener('beforeunload', event => {
        if (rows.querySelector('[data-dirty="true"], [data-saving="true"]')) { event.preventDefault(); event.returnValue = ''; }
    });

    function init() {
        if (state.loaded || state.loading) return;
        const now = new Date();
        const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
        form.elements.start_date.value = today; form.elements.end_date.value = today;
        state.filters = currentFilters();
        syncAdvanced();
        load();
    }
    return {init};
})();
