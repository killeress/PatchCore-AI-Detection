# CAPI AI — 自動光學檢測智慧推論系統

> **工業級 AI 推論平台，搭載 PatchCore 異常檢測技術，提供即時 TCP/IP 通訊、熱力圖視覺化，並支援人工複檢 (RIC) 交叉比對，專為面板缺陷偵測設計。**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PatchCore](https://img.shields.io/badge/AI-PatchCore-orange)](https://github.com/amazon-science/patchcore-inspection)
[![OpenVINO](https://img.shields.io/badge/Runtime-OpenVINO%20%7C%20PyTorch-lightblue)](https://openvino.ai)
[![SQLite](https://img.shields.io/badge/Database-SQLite-green)](https://sqlite.org)

🇺🇸 [English Version → README.md](./README.md)

---

## 系統簡介

CAPI AI 無縫整合至現有 AOI（自動光學檢測）產線，作為第二層 AI 判定引擎，透過深度學習異常檢測技術驗證機台判定結果。系統支援即時 TCP/IP 推論、SQLite 永久記錄，並內建 Web 儀表板提供完整的追溯與分析能力，包含與 RIC（人工複檢）記錄的比對報表。

```
AOI 機台  ──TCP/IP──▶  CAPI AI 伺服器  ──▶  SQLite DB
                              │                   │
                              ▼                   ▼
                         熱力圖檔案          Web 儀表板
                                                  │
                                        RIC 人工複檢比對報表
```

---

## 核心功能

| 功能 | 說明 |
|------|------|
| 🔬 **PatchCore 推論** | 以 512×512 切塊為單位進行異常偵測，門檻值可彈性設定 |
| 🌐 **TCP/IP 伺服器** | 多客戶端 Socket 伺服器，直接整合 AOI 機台即時通訊 |
| 🗺️ **熱力圖視覺化** | 逐 Tile 異常熱力圖，支援疊加原圖顯示 |
| 🧹 **智慧過濾機制** | OMIT 圖片交叉驗證 + Heatmap IOU 灰塵/刮痕抑制 |
| 💣 **炸彈缺陷偵測** | YAML 可設定座標型炸彈缺陷分類規則 |
| 📊 **Web 儀表板** | 即時監控、可搜尋記錄歷史、按班次統計分析 |
| 🔎 **RIC 交叉比對** | 匯入人工複檢資料，與 AI/AOI 結果進行準確率、過檢率、漏檢率比對 |
| 🗃️ **三層追溯記錄** | 推論 → 圖片 → Tile 三層完整稽核記錄 |
| ⚡ **雙執行環境** | 同時支援 OpenVINO (`.xml`) 與 PyTorch (`.pt`) 模型格式 |
| 🏭 **多產線支援** | 依產線分配對應 Port（第 N 線 → Port 79NN） |

---

## 專案結構

```
CAPI01_AD/
│
├── configs/
│   └── capi_3f.yaml              # 推論設定檔（門檻值、排除區域、炸彈座標）
│
├── ── 核心模組 ──
├── capi_config.py                # YAML 配置載入與驗證
├── capi_inference.py             # PatchCore 推論引擎（切塊、評分、過濾）
├── capi_heatmap.py               # 熱力圖生成與檔案管理
├── capi_database.py              # SQLite 持久化（記錄/圖片/Tile）
│
├── ── 伺服器 ──
├── capi_server.py                # TCP Socket 伺服器（正式環境入口）
├── capi_web.py                   # Web 介面（HTTP 伺服器 + REST API）
├── server_config.yaml            # Linux 正式環境設定
├── server_config_local.yaml      # Windows 本地測試設定
├── start_server.sh               # Linux 啟動腳本
│
├── ── 範本與靜態資源 ──
├── templates/                    # Jinja2 HTML 範本
│   ├── base.html                 # 版面配置與導航欄
│   ├── dashboard_v3.html         # 即時監控儀表板
│   ├── record_detail_v3.html     # 單筆記錄熱力圖詳情
│   ├── ric_report.html           # RIC 人工複檢比對報表
│   └── ...
├── static/                       # CSS、JS、靜態資源
│
├── ── 分析工具 ──
├── capi_missed_detection_analyzer.py  # 漏檢批次分析工具
├── diagnose_bomb.py                   # 炸彈缺陷診斷與視覺化
├── auto_sender.py                     # 自動化測試請求發送器
├── test_client.py                     # TCP 客戶端手動測試工具
├── check_db.py                        # 資料庫檢查工具
│
├── model.pt                      # 已訓練的 PatchCore 模型
├── capi_mark.png                 # MARK 範本（排除區域偵測用）
├── requirements.txt
└── README.zh-TW.md
```

---

## 快速上手

### 安裝依賴

```bash
pip install -r requirements.txt
```

> **GPU 注意事項**：正式環境建議使用 OpenVINO 推論。若無 Intel 硬體，可改用 PyTorch (`.pt`) 格式於本地測試。

### Linux 正式伺服器

```bash
# 1. 編輯設定檔
vim server_config.yaml

# 2. 啟動（含後台服務與 Web 儀表板）
chmod +x start_server.sh
./start_server.sh
```

### Windows 本地測試

```bash
# 終端 A — 啟動推論伺服器
python capi_server.py --config server_config_local.yaml

# 終端 B — 執行測試請求
python test_client.py

# 終端 B — 以真實面板目錄測試
python test_client.py --real "D:\path\to\panel_folder"

# 瀏覽器 — 開啟 Web 儀表板
# http://localhost:8080
```

---

## 通訊協議

CAPI AI 採用以分號分隔的簡易 TCP 文字協議：

```
[請求]  AOI@<玻璃ID>;<機種ID>;<機台編號>;<解析度X>,<解析度Y>;<機檢判定>;<圖片目錄>
[回應]  AOI@<玻璃ID>;<機種ID>;<機台編號>;<機檢判定>;<AI判定>
```

**AI 判定值說明：**

| 值 | 說明 |
|----|------|
| `OK` | 未偵測到缺陷 |
| `NG@圖片名(X,Y)` | 於座標 (X, Y) 偵測到缺陷 |
| `ERR:描述` | 處理錯誤 |

---

## Web 儀表板

啟動伺服器後，透過瀏覽器開啟 `http://<伺服器IP>:<Port>/` 存取。

| 路由 | 說明 |
|------|------|
| `/` | 即時監控儀表板 |
| `/records` | 可搜尋的推論記錄歷史 |
| `/record/<id>` | 單筆記錄熱力圖詳情 |
| `/ric` | RIC 人工複檢比對報表 |
| `/debug` | 單圖 Debug 推論 |
| `/stats` | 歷史統計與趨勢分析 |

### RIC 人工複檢比對報表

匯入人工複檢 (RIC) `.xls` 匯出檔案，自動計算：
- **AOI 準確率** — AOI 判定與 RIC 結果一致的比率
- **AI 準確率** — AI 判定與 RIC 結果一致的比率
- **過檢率** — AOI/AI 判 NG 但 RIC 判 OK 的比率
- **漏檢率** — AOI/AI 判 OK 但 RIC 判 NG 的比率

點擊任一統計卡片即可即時篩選明細表，並可將篩選結果導出為 CSV。

---

## 設定檔說明

`configs/capi_3f.yaml` 主要參數：

```yaml
threshold: 0.65                    # 異常分數門檻值（0.0 ~ 1.0）
tile_size: 512                     # 推論切塊尺寸（像素）
dust_heatmap_top_percent: 0.4      # 灰塵/刮痕 IOU 分位數閾值
excluded_edge_margin: 0.03         # 邊緣排除比例
bomb_defects:                      # 座標型炸彈缺陷規則
  - image_prefix: "STANDARD"
    coordinates: [[x1,y1], ...]
```

---

## 系統架構

```
┌─────────────────────────────────────────────────────────┐
│                      AOI 機台（客戶端）                   │
└──────────────────────────┬──────────────────────────────┘
                           │ TCP/IP 請求
┌──────────────────────────▼──────────────────────────────┐
│                 CAPIServer (capi_server.py)              │
│  ┌──────────────┐   ┌───────────────┐   ┌────────────┐  │
│  │ Config Loader│   │ CAPIInferencer│   │HeatmapMgr  │  │
│  │(capi_config) │   │(PatchCore)    │   │(capi_heatm)│  │
│  └──────────────┘   └───────┬───────┘   └────────────┘  │
│                             │                            │
│  ┌──────────────────────────▼──────────────────────────┐ │
│  │              CAPIDatabase (SQLite)                  │ │
│  │  inference_records → image_results → tile_results   │ │
│  └──────────────────────────┬──────────────────────────┘ │
└──────────────────────────── │ ──────────────────────────-┘
                              │
┌─────────────────────────────▼───────────────────────────┐
│                CAPIWebHandler (capi_web.py)              │
│         HTTP 儀表板 + REST API + RIC 比對報表             │
└─────────────────────────────────────────────────────────┘
```

---

## 授權聲明

僅限內部使用。© CAPI AI Team。
