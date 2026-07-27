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
    title: "寧波廠區 CAPI AI 中控看板",
    refreshIntervalSeconds: 30,
    requestTimeoutSeconds: 8,
    lines: [
        {
            id: "mod1-line-35",
            factory: "MOD1",
            line: "35線",
            pcName: "MOD1-35",
            apiUrl: "http://10.172.25.105/api/status",
            dashboardUrl: "http://10.172.25.105/",
            overexposedUrl: "http://10.172.25.105/overexposed",
            enabled: true
        },
        {
            id: "mod2-capi03",
            factory: "MOD2",
            line: "CAPI03",
            pcName: "CAPI03",
            apiUrl: "http://10.174.37.81/api/status",
            dashboardUrl: "http://10.174.37.81/",
            overexposedUrl: "http://10.174.37.81/overexposed",
            enabled: true
        },
        {
            id: "mod2-capi13",
            factory: "MOD2",
            line: "CAPI13",
            pcName: "CAPI13",
            apiUrl: "http://10.174.37.84/api/status",
            dashboardUrl: "http://10.174.37.84/",
            overexposedUrl: "http://10.174.37.84/overexposed",
            enabled: true
        },
        {
            id: "mod2-capi-hm",
            factory: "MOD2",
            line: "CAPI-HM",
            pcName: "CAPI-HM",
            apiUrl: "http://10.174.24.103/api/status",
            dashboardUrl: "http://10.174.24.103/",
            overexposedUrl: "http://10.174.24.103/overexposed",
            enabled: true
        },
        {
            id: "mod2-capi14",
            factory: "MOD2",
            line: "CAPI14",
            pcName: "CAPI14",
            apiUrl: "http://10.174.37.67/api/status",
            dashboardUrl: "http://10.174.37.67/",
            overexposedUrl: "http://10.174.37.67/overexposed",
            enabled: true
        }
    ]
};
