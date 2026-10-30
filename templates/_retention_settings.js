const RETENTION_PARAM = 'data_retention_policy';
const RETENTION_FIELDS = [
    ['ok_retain_days', 'OK／OK-i／OK-s 判定紀錄', '包含 AOI NG → AI OK 的「AI 救回」。到期後，玻璃 ID 與判定紀錄將無法搜尋。'],
    ['ng_retain_days', 'NG／錯誤等其他判定紀錄', '包含 NG、ERR 等判定。到期後刪除主紀錄及相關影像結果。'],
    ['tile_retain_days', 'Tile 切塊明細', '切塊分數、座標、判定與相關刮痕覆核。到期後仍可查尚未到期的主紀錄。'],
    ['heatmap_retain_days', '熱圖與相關檢查圖片', '包含已產生的熱圖、規格內複判圖及 MARK 檢查圖；0 表示不自動清理這些圖片。'],
];
let retentionDraft = null;
let retentionSource = '';
let retentionDirty = false;
let retentionSaving = false;
let retentionReason = '';
let retentionError = '';

function renderRetentionPane() {
    const param = allParams.find(p => p.param_name === RETENTION_PARAM);
    if (!param) return '';
    const saved = param.decoded_value || JSON.parse(param.param_value);
    const source = JSON.stringify(saved);
    if (!retentionDraft || (!retentionDirty && source !== retentionSource)) {
        retentionDraft = {...saved};
        retentionSource = source;
    }
    const p = retentionDraft;
    return `
        <div id="pane-data-retention" class="settings-tab-pane-block ${currentTab === 'data-retention' ? 'active' : ''}">
            <div class="settings-section retention-section">
                <div class="settings-section-header">
                    <span>資料保留與自動清理</span>
                    <span class="retention-status">${saved.enabled ? `已啟用 · 每日 ${escapeHtml(saved.schedule_time)}` : '自動清理已關閉'}</span>
                </div>
                <div class="retention-intro">
                    管理這台主機所有機台的資料期限。判定紀錄預設保留 180 天，Tile 與圖片各自計算期限。
                    儲存後一分鐘內更新排程，依每日清理時間執行，無需重新載入模型或重啟服務。
                </div>
                <form id="retention-form" onsubmit="event.preventDefault(); saveRetentionSettings();">
                    <fieldset id="retention-fields" ${retentionSaving ? 'disabled' : ''}>
                        <div class="retention-layout">
                            <div>
                                ${RETENTION_FIELDS.map(([key, label, description]) => `
                                    <div class="retention-row">
                                        <div><label for="retention-${key}">${label}</label><p>${description}</p></div>
                                        <div class="retention-days">
                                            <input id="retention-${key}" type="number" min="${key === 'heatmap_retain_days' ? 0 : 1}" max="3650" step="1" required
                                                value="${escapeHtml(String(p[key]))}" oninput="updateRetentionDraft('${key}', this.value)">
                                            <span>天</span>
                                        </div>
                                    </div>`).join('')}
                            </div>
                            <div class="retention-schedule">
                                <label class="retention-check" for="retention-enabled">
                                    <input id="retention-enabled" type="checkbox" ${p.enabled ? 'checked' : ''} onchange="updateRetentionDraft('enabled', this.checked)">
                                    每日自動清理
                                </label>
                                <p>關閉後暫停本排程的資料刪除；保留天數設定仍會保存。</p>
                                <label for="retention-schedule_time">每日清理時間（伺服器本機時間）</label>
                                <input id="retention-schedule_time" type="time" step="60" required value="${escapeHtml(p.schedule_time)}" oninput="updateRetentionDraft('schedule_time', this.value)">
                                <label class="retention-check" for="retention-vacuum_after_cleanup">
                                    <input id="retention-vacuum_after_cleanup" type="checkbox" ${p.vacuum_after_cleanup ? 'checked' : ''} onchange="updateRetentionDraft('vacuum_after_cleanup', this.checked)">
                                    清理後回收資料庫空間
                                </label>
                                <p>有刪除資料時整理資料庫；資料量較大時可能增加磁碟作業時間。</p>
                            </div>
                        </div>
                        <div class="retention-notes">
                            <strong>保留範圍說明</strong>
                            <ul>
                                <li>判定紀錄依建立日期計算期限；主紀錄到期時，會連帶刪除相關影像結果、切塊明細與覆核資料。</li>
                                <li>延長判定紀錄期限不會延長 Tile 或圖片期限；Tile 最多保留至其主紀錄到期。</li>
                                <li>AI 推論歷史統計獨立保存，不受本清理排程影響；統計數字可能多於仍可搜尋的明細。</li>
                                <li>AOI 原始圖片由來源端管理，本設定不會備份原圖，也不會補產未曾保存的熱圖。</li>
                                <li>縮短期限後，超期資料會在下次清理時刪除；已刪除資料無法靠延長期限恢復。正在執行的清理會完成原有批次。</li>
                            </ul>
                        </div>
                        <div class="retention-actions">
                            <div class="retention-reason">
                                <label for="retention-reason">修改原因</label>
                                <input id="retention-reason" type="text" required maxlength="500" placeholder="例如：依用戶需求保留半年判定紀錄"
                                    value="${escapeHtml(retentionReason)}" oninput="retentionReason = this.value; retentionDirty = true;">
                            </div>
                            <button class="btn-edit" type="button" onclick="resetRetentionDraft()">還原未儲存修改</button>
                            <button class="btn-save" type="submit">${retentionSaving ? '儲存中…' : '儲存資料保留設定'}</button>
                        </div>
                    </fieldset>
                    <div class="retention-error" role="alert">${escapeHtml(retentionError)}</div>
                </form>
            </div>
        </div>`;
}

function updateRetentionDraft(key, value) {
    retentionDraft[key] = value;
    retentionDirty = true;
}

function resetRetentionDraft() {
    retentionDraft = null;
    retentionDirty = false;
    retentionReason = '';
    retentionError = '';
    renderSettings();
}

async function saveRetentionSettings() {
    if (retentionSaving) return;
    const form = document.getElementById('retention-form');
    if (!form.reportValidity()) return;
    const policy = {...retentionDraft};
    for (const [key] of RETENTION_FIELDS) {
        const minimum = key === 'heatmap_retain_days' ? 0 : 1;
        const raw = policy[key];
        const value = Number(raw);
        if (String(raw).trim() === '' || !Number.isInteger(value) || value < minimum || value > 3650) {
            showToast('請填寫有效的整數保留天數', 'error');
            return;
        }
        policy[key] = value;
    }
    if (!retentionReason.trim()) {
        showToast('請填寫修改原因', 'error');
        return;
    }
    retentionSaving = true;
    retentionError = '';
    renderSettings();
    try {
        const res = await fetch('/api/settings/update', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({param_name: RETENTION_PARAM, new_value: policy, reason: retentionReason.trim()}),
        });
        const data = await res.json();
        if (!res.ok || data.error || !data.success) throw new Error(data.error || '儲存失敗，請稍後再試');
        const saved = data.policy || policy;
        const param = allParams.find(p => p.param_name === RETENTION_PARAM);
        param.decoded_value = saved;
        param.param_value = JSON.stringify(saved);
        retentionDraft = null;
        retentionDirty = false;
        retentionReason = '';
        showToast(data.message || '資料保留設定已儲存', 'success');
        await loadHistory();
    } catch (error) {
        retentionError = error.message;
        showToast(`儲存失敗：${error.message}`, 'error');
    } finally {
        retentionSaving = false;
        renderSettings();
    }
}
