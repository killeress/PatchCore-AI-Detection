/* Shared by the portable central dashboard and persisted inference logs. */
(function () {
    'use strict';
    function missingModelNotice(text) {
        // Only recognize the legacy v2 missing screen/zone mapping error.
        const units = new Set();
        const screens = new Set();
        for (const m of String(text || '').matchAll(/\[v2\]\s+([A-Z0-9_]+)\/(inner|edge)\s+tile\([^\r\n)]*\)\s+推論失敗:\s*(?:KeyError:\s*)?['"]([A-Z0-9_]+|inner|edge)['"]/g)) {
            if (m[3] === m[1] || m[3] === m[2]) {
                units.add(`${m[1]}（${m[2] === 'inner' ? '內部' : '邊緣'}）`);
                screens.add(m[1]);
            }
        }
        if (!units.size) return null;
        return {
            title: '需要重新訓練模型',
            screens: [...screens],
            message: `目前使用的模型套件缺少畫面模型對應：${[...units].join('、')}，本次檢測未完成。`,
            action: '請使用包含上述畫面的正常圖片重新訓練模型，完成後到「模型管理」啟用更新後的模型套件，並確認此產線使用該套件，再重新推論。若已有涵蓋該畫面的模型，可直接切換至正確套件。',
        };
    }
    if (typeof module !== 'undefined' && module.exports) module.exports = {missingModelNotice};
    if (typeof window !== 'undefined') window.CAPIInferenceErrors = {missingModelNotice};
})();
