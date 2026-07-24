/*
 * CAPI AI 中控看板設定
 *
 * 新增線體時，複製 lines 內的一個物件並修改：
 * - id              : 不重複的識別碼
 * - factory         : 廠別
 * - line            : 線體名稱
 * - pcName          : 現場 PC 名稱
 * - apiUrl          : 該 PC 的 /api/status
 * - dashboardUrl    : 「開啟該 PC」超連結
 * - overexposedUrl  : Omit 過曝明細超連結
 */
window.CAPI_DASHBOARD_CONFIG = {
    title: "CAPI AI 中控看板",
    refreshIntervalSeconds: 30,
    requestTimeoutSeconds: 8,
    lines: [
        {
            id: "mod2-line-01",
            factory: "MOD2",
            line: "Line 01",
            pcName: "CAPI-PC-01",
            apiUrl: "http://10.172.25.105/api/status",
            dashboardUrl: "http://10.172.25.105/",
            overexposedUrl: "http://10.172.25.105/overexposed",
            enabled: true
        }
    ]
};
