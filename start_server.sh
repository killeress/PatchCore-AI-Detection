#!/bin/bash
# ============================================================
# CAPI AI 推論伺服器啟動腳本
# ============================================================
#
# 使用方式:
#   chmod +x start_server.sh
#   ./start_server.sh                    # 使用預設設定
#   ./start_server.sh -c my_config.yaml  # 指定設定檔
#
# 停止服務:
#   kill $(cat /tmp/capi_server.pid)
#   或 Ctrl+C
#
# ============================================================

set -e

# 切換到腳本所在目錄
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 預設設定
CONFIG_FILE="${1:-server_config.yaml}"
PID_FILE="/tmp/capi_server.pid"
LOG_DIR="/data/capi_ai/logs"

echo "============================================================"
echo "  CAPI AI Inference Server"
echo "============================================================"
echo "  Working dir : $SCRIPT_DIR"
echo "  Config      : $CONFIG_FILE"
echo "  PID file    : $PID_FILE"
echo "============================================================"

# 檢查設定檔
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# 建立必要目錄
mkdir -p "$LOG_DIR"
mkdir -p "/data/capi_ai/heatmaps"

# 檢查 Python
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found"
    exit 1
fi
echo "  Python      : $PYTHON ($($PYTHON --version 2>&1))"

# 檢查依賴
echo ""
echo "Checking dependencies..."
$PYTHON -c "import yaml" 2>/dev/null || { echo "ERROR: PyYAML not installed. Run: pip install pyyaml"; exit 1; }
$PYTHON -c "import cv2" 2>/dev/null || { echo "ERROR: OpenCV not installed. Run: pip install opencv-python"; exit 1; }
$PYTHON -c "import numpy" 2>/dev/null || { echo "ERROR: NumPy not installed. Run: pip install numpy"; exit 1; }
echo "Dependencies OK"

# 儲存 PID
echo ""
echo "Starting server..."
echo $$ > "$PID_FILE"

# 啟動伺服器
exec $PYTHON capi_server.py --config "$CONFIG_FILE"
