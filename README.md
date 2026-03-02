# CAPI AI 推論系統

CAPI 面板異常檢測推論系統，包含批量推論引擎與 TCP/IP AI 推論伺服器。

## 檔案結構

```
CAPI01_AD/
├── configs/
│   └── capi_3f.yaml                  # CAPI_3F 推論配置
│
│── 核心模組 ──
├── capi_config.py                    # 配置管理模組
├── capi_inference.py                 # 推論核心引擎 (PatchCore)
├── capi_mark.png                     # MARK 模板圖
├── model.pt                          # AI 模型 (PatchCore)
│
│── AI 推論伺服器 ──
├── capi_server.py                    # TCP Socket Server (port 可配置)
├── capi_database.py                  # SQLite 推論記錄持久化
├── capi_heatmap.py                   # 熱力圖生成與儲存
├── capi_web.py                       # Web 查閱介面 (HTTP)
├── server_config.yaml                # Linux 伺服器設定
├── server_config_local.yaml          # Windows 本地測試設定
├── start_server.sh                   # Linux 啟動腳本
├── test_client.py                    # 測試客戶端
│
│── 分析工具 ──
├── capi_missed_detection_analyzer.py # 漏檢分析工具
├── CAPI_FLOW.md                      # 檢測流程圖
├── requirements.txt                  # Python 依賴
└── README.md                         # 本文件
```

## 使用方式

### 啟動 AI 推論伺服器 (Linux)

```bash
# 安裝依賴
pip install -r requirements.txt

# 調整設定
vim server_config.yaml

# 啟動
chmod +x start_server.sh
./start_server.sh
```

### Windows 本地測試

```bash
# 終端 A：啟動 Server
python capi_server.py --config server_config_local.yaml

# 終端 B：基本測試
python test_client.py

# 終端 B：真實推論測試
python test_client.py --real "D:\path\to\panel_folder"

# 瀏覽器：查看結果
# http://localhost:8080
```

### 通訊協議

```
[Request]  AOI@玻璃ID;機種ID;機台編號;解析度X,解析度Y;機檢判定;圖片目錄路徑
[Response] AOI@玻璃ID;機種ID;機台編號;機檢判定;AI判定
```

AI 判定: `OK` / `NG@圖片名(X,Y)` / `ERR:描述`

## 功能特點

- **TCP/IP Socket 通訊**：Testing ↔ AI Server，支援多客戶端
- **PatchCore 異常檢測**：512×512 切塊推論
- **可配置排除區域**：MARK 二維碼、機構區域
- **灰塵/刮痕過濾**：OMIT 圖片交叉驗證 + Heatmap IOU
- **SQLite 追溯**：三層記錄 (推論 → 圖片 → Tile)
- **Web 查閱**：熱力圖瀏覽 + 搜尋 + 統計
- **多格式模型**：支援 OpenVINO (.xml) 和 PyTorch (.pt)
