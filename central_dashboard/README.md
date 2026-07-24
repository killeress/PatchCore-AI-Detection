# CAPI AI 中控看板

獨立的純 HTML / CSS / JavaScript 看板。中控瀏覽器每 30 秒呼叫各 PC 的 `/api/status`，不使用中央資料庫，也不修改各 PC 的資料。

## 設定線體

編輯 `config.js` 的 `lines`。每條線至少需要：

```javascript
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
```

`refreshIntervalSeconds` 小於 30 時，前端仍會強制使用 30 秒，避免查詢過於頻繁。

## 在中控 PC 開啟（不需要 Python）

將整個 `central_dashboard` 資料夾複製到中控 PC，直接雙擊 `index.html` 即可。CSS、JavaScript 與線體設定都在同一個資料夾內，不需安裝 Python、Node.js 或其他程式。

## API 與 CORS

直接雙擊時，看板網址會是 `file:///.../index.html`，線體 API 則是：

```text
http://10.172.25.105/api/status
```

因此各 CAPI PC 必須允許這個不帶帳號驗證的唯讀 API 跨來源存取。直接開檔模式最簡單的回應標頭是：

```text
Access-Control-Allow-Origin: *
```

本版本的 `capi_web.py` 已只針對 `GET /api/status` 加入這個標頭。每台 CAPI PC 重新佈署本版程式並重啟服務後生效；尚未更新的 PC 仍會在看板上顯示離線。

若公司資安政策不允許 `*`，就不能使用直接雙擊模式；需改由 IIS 或既有內網 Web Server 提供此資料夾，讓各 PC API 只允許該中控網址。純 HTML 無法繞過瀏覽器的同源安全限制。

部分公司管理的瀏覽器也可能禁止 `file://` 讀取 `http://`。遇到這種政策時，HTML 本身不需修改，但必須改放到既有 IIS 或內網靜態網站。

## 已支援的現有 `/api/status`

看板會讀取：

- `server.running`
- `server.uptime`
- `server.model_version`
- `server.device`
- `traffic.active_connections`
- `traffic.connected_machines`
- `traffic.active_inferences`
- `stats.total_requests`
- `stats.total_ok`
- `stats.total_ng`
- `stats.total_err`
- `stats.shift_name`
- `stats.time_range`
- `latest_event`

現有 API 尚未回傳 Omit 過曝、平均耗時、VRAM、GPU 使用率、溫度、RAM 與硬碟資訊，因此這些欄位會顯示 `—`。

## 可選的硬體資料格式

日後各 PC API 若增加以下欄位，前端不需修改即可顯示：

```json
{
  "stats": {
    "avg_time": 1.6,
    "overexposed_count": 7
  },
  "hardware": {
    "gpu": {
      "vram_used_gb": 7.2,
      "vram_total_gb": 16.0,
      "utilization_percent": 42,
      "temperature_c": 58
    },
    "memory": {
      "used_percent": 61
    },
    "disk": {
      "free_gb": 182.4,
      "total_gb": 500.0
    }
  }
}
```

API 暫時離線時，看板會保留最後一次成功資料並標示離線，不會把既有數字清空。
