# CAPI AI — AOI PatchCore 智慧檢測平台

> 面向產線的 AI 推論服務，透過 TCP 接收 AOI 請求，執行設定好的 PatchCore 推論流程，同時回傳既有 AOI 判定與 QJPG 報告，並將結果保存供 Web 追溯與複核。

🇺🇸 [English version → README.md](./README.md)

## 本專案包含什麼

- **推論伺服器** — `capi_server.py` 處理長連線 TCP 客戶端、請求解析、模型分派、推論與協議回覆。
- **PatchCore 推論流程** — `capi_inference.py` 與 `capi_preprocess.py` 負責面板前處理、Tile／區域路由、異常分數、熱力圖、MARK、炸彈與模型設定中的後處理規則。
- **追溯與 Web 介面** — `capi_database.py` 將推論、圖片、Tile 記錄保存到 SQLite；`capi_web.py` 提供監控、搜尋、單筆詳情、RIC 複核與管理頁面。
- **訓練與模型庫** — `/training` → `/train/new` 流程可準備訓練資料、審核 Tile、訓練模型 Bundle，再由 `/models` 管理啟用。
- **部署支援** — Repo 內含版本資訊、部署 ZIP 產生、人工更新與 pull 模式更新工具。

正式資料流如下：

```text
AOI 客戶端
    │ TCP
    ▼
capi_server.py ──► capi_inference.py / capi_preprocess.py
    │                              │
    ├── AOI 舊協議 + QJPG 回覆     │
    ├── SQLite 推論記錄             └── 熱力圖與診斷資料
    ▼
capi_web.py ──► 儀表板、複核、訓練、模型庫、設定
```

## 環境與安裝

請使用 Python 3.10 以上；目前開發／部署環境使用 Python 3.11／3.12。

```bash
python -m pip install -r requirements.txt
```

Repo 不包含正式模型權重。要實際啟動推論，還需要依目標機種準備模型 Bundle，以及可讀取 AOI 圖片的路徑映射。模型權重、資料庫、本機資料集與本機憑證預設不納入一般版本控制與部署包。

## 啟動伺服器

### Windows 本地測試

`server_config_local.yaml` 是本地測試設定：

- TCP 伺服器：`0.0.0.0:7891`
- Web 介面：`http://localhost:8080`
- SQLite 資料庫：`./test_results.db`
- 熱力圖：`./test_heatmaps`

可使用以下任一方式啟動：

```powershell
python capi_server.py --config server_config_local.yaml
# 或
start_server_local.bat
```

若已有面板資料集，可用 `auto_sender.py` 發送測試請求：

```powershell
python auto_sender.py --host 127.0.0.1 --port 7891 --ng-folder D:\path\to\panels --count 1
```

### Linux 正式環境

先依目標設備修改 `server_config.yaml`，再使用服務腳本：

```bash
chmod +x start_server.sh
./start_server.sh              # 停止舊程序、背景啟動並追蹤 Log
./start_server.sh status
./start_server.sh log
./start_server.sh stop
```

目前正式設定預設使用 TCP `7907`、Web `80`。實際 Port、資料庫路徑、熱力圖路徑、模型清單、路徑映射、保留期限與選用整合功能，都由 `server_config.yaml` 控制。

若要以前景模式直接啟動：

```bash
python3 capi_server.py --config server_config.yaml
```

不要把正式環境路徑或憑證直接複製到本地設定。尤其 `server_config.yaml` 含有設備專用路徑與 MES 設定，部署前必須逐項確認。

## TCP 通訊協議

伺服器接受以分號分隔的 `AOI@` 請求。沒有炸彈座標時：

```text
AOI@<玻璃ID>;<機種ID>;<機台編號>;<解析度X>,<解析度Y>;<機檢判定>;<圖片目錄>
```

包含炸彈資料時，圖片路徑前會增加圖片前綴與座標：

```text
AOI@<玻璃ID>;<機種ID>;<機台編號>;<解析度X>,<解析度Y>;<機檢判定>;<圖片前綴>;<座標>;<圖片目錄>
```

`機檢判定` 通常是 `OK`、`NG` 或 `HY`。`HY` 會跳過 AI 推論，改走畫異結果。

目前回覆以 CRLF 結尾，同一包包含兩種格式，順序如下：

```text
AOI@<玻璃ID>;<機種ID>;<機台編號>;<機檢判定>;<AI判定>
@QJPG-<玻璃ID>;<MARK判定>;<MARK字>;<Defect欄位>,
```

客戶端應依 prefix（`AOI@` 或 `@QJPG-`）辨識每一行，不要假設回覆只有一行。`AI判定` 可為 `OK`、`NG` 或 `ERR:<描述>`；內部的 `OK-i` 在舊 AOI 回覆中會轉成 `OK`。完整欄位與 QJPG defect code 規格請參考 [docs/client_communication_protocol.zh-TW.md](./docs/client_communication_protocol.zh-TW.md)。

## Web 介面主要入口

伺服器啟動後，開啟 `http://<伺服器>:<Web Port>/`。

| 路徑 | 用途 |
|---|---|
| `/` | 即時儀表板與當班狀態 |
| `/search` | 搜尋與匯出推論記錄 |
| `/record/<id>` | 單筆記錄、圖片、Tile 與熱力圖 |
| `/ric` | RIC、過檢、漏檢、MES 比對與相關報表 |
| `/ric/within-spec-logs` | 規格內複核清單與詳情 |
| `/training` | 模型訓練入口 |
| `/train/new` | 新機種 PatchCore 訓練流程 |
| `/models` | 模型 Bundle 檢查與啟用 |
| `/debug` | 單圖與座標診斷 |
| `/white-frame` | 白框總表與記錄 |
| `/settings` | 需登入的設定與帳號管理 |
| `/logs` | 伺服器 Log 檢視 |
| `/release-notes` | 系統內更新說明 |
| `/api/status` | 執行狀態與硬體狀態 JSON |
| `/api/version` | 部署版本與建置資訊 JSON |

## 設定檔邊界

| 檔案或目錄 | 責任 |
|---|---|
| `server_config.yaml` | 正式 TCP／Web、SQLite、熱力圖、路徑映射、模型清單、清理排程、訓練與選用整合設定 |
| `server_config_local.yaml` | Windows／本地測試用 Port 與輸出路徑 |
| `configs/capi_3f.yaml` | 舊架構／fallback 模型設定、圖片前綴映射、門檻、排除區、炸彈規則與後處理 |
| `model/<machine>-<timestamp>/` | 訓練流程產生的模型 Bundle，內含該 Bundle 的模型設定與 metadata |
| `VERSION`／`CHANGELOG.md` | 發行版本與操作員可讀的更新紀錄 |

正式環境的 `model_configs` 應指向與請求 `ModelID` 相符的 Bundle `machine_config.yaml`。`configs/capi_3f.yaml` 僅保留作舊架構／fallback 使用，不能取代必要模型權重的安裝。

## 常用開發檢查

不啟動 Listener，直接執行協議 smoke test：

```bash
python -X utf8 capi_server.py --test-protocol
```

在 repo 根目錄執行自動化測試：

```bash
python -m pytest tests/
```

## 相關文件

- [客戶端通訊協議](./docs/client_communication_protocol.zh-TW.md)
- [新機種模型訓練 SOP](./docs/new_system_model_training_sop.zh-TW.md)
- [PatchCore 訓練架構](./docs/patchcore_training_architecture.zh-TW.md)
- [實驗版 pull 模式自動更新流程](./docs/experimental_auto_update.zh-TW.md)
- [部署 ZIP 產生器](./scripts/build_deploy_zip.py)
- [中央看板](./central_dashboard/README.md)
- [更新紀錄](./CHANGELOG.md)

內部專案，非公開發行用途。
