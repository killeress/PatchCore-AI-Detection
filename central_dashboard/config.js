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
            id: "mod2-hm-83",
            factory: "MOD2",
            line: "HM",
            pcName: "10.174.24.83",
            apiUrl: "http://10.174.24.83/api/status",
            dashboardUrl: "http://10.174.24.83/",
            overexposedUrl: "http://10.174.24.83/overexposed",
            enabled: true
        },
        {
            id: "mod2-hm-103",
            factory: "MOD2",
            line: "HM",
            pcName: "10.174.24.103",
            apiUrl: "http://10.174.24.103/api/status",
            dashboardUrl: "http://10.174.24.103/",
            overexposedUrl: "http://10.174.24.103/overexposed",
            enabled: true
        },
        {
            id: "mod2-capi08",
            factory: "MOD2",
            line: "CAPI08",
            pcName: "CAPI08",
            apiUrl: "http://10.174.37.160/api/status",
            dashboardUrl: "http://10.174.37.160/",
            overexposedUrl: "http://10.174.37.160/overexposed",
            enabled: true
        },
        {
            id: "mod2-capi01",
            factory: "MOD2",
            line: "CAPI01",
            pcName: "CAPI01",
            apiUrl: "http://10.174.37.137/api/status",
            dashboardUrl: "http://10.174.37.137/",
            overexposedUrl: "http://10.174.37.137/overexposed",
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
        },
        {
            id: "mod2-capi02",
            factory: "MOD2",
            line: "CAPI02",
            pcName: "CAPI02",
            apiUrl: "http://10.174.37.208/api/status",
            dashboardUrl: "http://10.174.37.208/",
            overexposedUrl: "http://10.174.37.208/overexposed",
            enabled: true
        },
        {
            id: "mod1-capi35",
            factory: "MOD1",
            line: "CAPI35",
            pcName: "CAPI35",
            apiUrl: "http://10.172.25.105/api/status",
            dashboardUrl: "http://10.172.25.105/",
            overexposedUrl: "http://10.172.25.105/overexposed",
            enabled: true
        },
        {
            id: "mod1-capi34",
            factory: "MOD1",
            line: "CAPI34",
            pcName: "CAPI34",
            apiUrl: "http://10.172.25.129/api/status",
            dashboardUrl: "http://10.172.25.129/",
            overexposedUrl: "http://10.172.25.129/overexposed",
            enabled: true
        }
    ]
};
