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
- `stats.avg_time`（當班平均處理秒數）
- `stats.overexposed_count`（當班 Omit 過曝數）
- `hardware.gpu`（型號、使用率、溫度、VRAM）
- `hardware.memory`（RAM 使用量）
- `hardware.disk`（資料庫所在磁碟空間）
- `latest_event`

硬體資訊會在各 CAPI PC 端快取 30 秒。即使既有本機頁面更頻繁呼叫 `/api/status`，也不會每次都重新執行硬體查詢。GPU 資料由 NVIDIA 驅動的 `nvidia-smi` 提供；未安裝 NVIDIA 驅動或查詢失敗時，GPU/VRAM 欄位顯示 `—`，其他狀態仍可正常顯示。

## `/api/status` 新增資料格式

```json
{
  "stats": {
    "avg_time": 1.6,
    "overexposed_count": 7
  },
  "hardware": {
    "gpu": {
      "available": true,
      "name": "NVIDIA RTX A4000",
      "vram_used_gb": 7.2,
      "vram_total_gb": 16.0,
      "utilization_percent": 42,
      "temperature_c": 58
    },
    "memory": {
      "used_gb": 19.5,
      "total_gb": 32.0,
      "used_percent": 61
    },
    "disk": {
      "path": "/aidata/capi_ai",
      "free_gb": 182.4,
      "used_gb": 317.6,
      "total_gb": 500.0,
      "used_percent": 63.5
    }
  }
}
```

API 暫時離線時，看板會保留最後一次成功資料並標示離線，不會把既有數字清空。

## 設備健康提醒

看板會在「目前告警」區塊提醒設備健康狀況；提醒只會在 API 有提供對應數值時觸發：

- 硬碟剩餘率 `<= 15%`：警告；`<= 10%`：嚴重
- RAM 使用率 `>= 85%`：警告；`>= 95%`：嚴重
- VRAM 使用率 `>= 85%`：警告；`>= 95%`：嚴重
- GPU 溫度 `>= 80°C`：警告；`>= 90°C`：嚴重

嚴重提醒會使用紅色，普通提醒使用黃色；設備卡片上方狀態色條也會同步變色。硬體提醒不會把正常運作的服務誤標成「服務異常」。
