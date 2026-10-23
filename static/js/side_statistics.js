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

    function renderSummary() {
        document.getElementById('ss-summary').innerHTML = Object.entries(groups).map(([key, label]) => {
            const s = state.summary[key] || {};
            const metric = (name, value) => `<span>${name} <b>${number(value)}</b></span>`;
            return `<button type="button" data-group="${key}" aria-pressed="${key === state.group}" ${state.loading ? 'disabled' : ''}>
                <strong>${label} · ${number(s.total)} 筆</strong><div class="ss-metrics">
                ${metric('保留候選', s.candidates)}${metric('無保留候選', s.clear)}${metric('未執行／無圖／失敗', s.unavailable)}</div>
                <div class="ss-metrics">${metric('待驗證', s.unreviewed)}${metric('檢出正確', s.confirmed)}${metric('無缺陷確認', s.correct_clear)}
                ${metric('系統過檢', s.false_positive)}${metric('系統漏檢', s.missed)}${metric('映射不正確', s.mapping_error)}${metric('無法確認', s.uncertain)}</div>
                <p class="sw-note">人工確認正確率：${s.accuracy == null ? '尚無有效驗證' : number(s.accuracy).toFixed(1) + '%'}（有效驗證 ${number(s.verified)} 筆）</p></button>`;
        }).join('');
    }

    function pagination() {
        document.getElementById('ss-prev').disabled = state.loading || state.offset === 0;
        document.getElementById('ss-next').disabled = state.loading || state.offset + state.limit >= state.total;
        document.getElementById('ss-page').textContent = state.total
            ? `第 ${Math.floor(state.offset / state.limit) + 1} / ${Math.ceil(state.total / state.limit)} 頁` : '共 0 筆';
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
            rows.innerHTML = data.rows_html || '<div class="sw-empty">此分組在目前條件下沒有資料，可切換另一分組或調整日期。</div>';
            message.textContent = `${groups[group]} · 共 ${data.total} 筆 · 每頁最多 ${state.limit} 筆`;
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
            reviewForm.closest('article').querySelector('[data-review-label]').textContent = reviewForm.elements.decision.selectedOptions[0].textContent;
            feedback.textContent = `已儲存 · ${data.reviewed_by} · ${data.reviewed_at}`;
            feedback.classList.add('ss-saved');
            try {
                const refreshed = await getData(state.filters, state.group, state.offset, true);
                state.summary = refreshed.summary; state.total = refreshed.total;
                renderSummary(); pagination();
                message.textContent = `驗證已儲存，兩組統計已更新。${state.filters.review ? '目前明細保留供核對，重新查詢可套用更新後的驗證篩選。' : ''}`;
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
        if (form.reportValidity()) load(state.group, 0, Object.fromEntries(fields.map(key => [key, form.elements[key].value.trim()])));
    });
    root.addEventListener('click', event => {
        const button = event.target.closest('[data-group]');
        if (button) load(button.dataset.group, 0);
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
        state.filters = Object.fromEntries(fields.map(key => [key, form.elements[key].value.trim()]));
        load();
    }
    return {init};
})();
