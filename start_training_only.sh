#!/bin/bash
# ============================================================
# CAPI 訓練專用模式啟動腳本
# ============================================================
#
# 用途：暫時停掉正常推論 server、起一個只給訓練用的最小 server，
#       把整顆 GPU 留給訓練 subprocess，避免 OOM。
#
# 使用流程：
#   1. ./start_training_only.sh   ← 停推論、起訓練專用 server
#   2. 用瀏覽器到 /train/new 跑訓練（10-20 分鐘）
#   3. 訓練完成後執行 ./start_server.sh   ← 還原正常推論 server
#
# 跟正常 server 的差別：
#   - 不載入推論模型（GPU 不被佔據）
#   - 不開 TCP 7907 給 AOI 機器（這段時間 AOI 線拿不到 AI 判定）
#   - Web UI 一樣可用，但 dashboard 顯示沒有 inferencer
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="/tmp/capi_server.pid"
CONFIG_FILE="server_config.yaml"

while [ $# -gt 0 ]; do
    case "$1" in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [-c config.yaml]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "  CAPI Training-Only Server"
echo "  （無推論模型、無 TCP，只起 web 給訓練用）"
echo "============================================================"
echo "  Working dir : $SCRIPT_DIR"
echo "  Config      : $CONFIG_FILE"
echo "  PID file    : $PID_FILE"
echo "============================================================"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# 停掉舊的 capi_server.py（不管是不是 training-only 模式）
echo "Stopping existing capi_server (if any)..."
existing_pid=""
if [ -f "$PID_FILE" ]; then
    existing_pid=$(cat "$PID_FILE" 2>/dev/null || true)
fi
if [ -z "$existing_pid" ] || ! kill -0 "$existing_pid" 2>/dev/null; then
    existing_pid=$(pgrep -f "capi_server.py" | head -1 || true)
fi
if [ -n "$existing_pid" ]; then
    echo "  Sending SIGTERM to pid $existing_pid..."
    kill "$existing_pid" 2>/dev/null || true
    i=0
    while [ $i -lt 10 ] && kill -0 "$existing_pid" 2>/dev/null; do
        sleep 1
        i=$((i + 1))
    done
    if kill -0 "$existing_pid" 2>/dev/null; then
        echo "  Still alive after 10s — sending SIGKILL"
        kill -9 "$existing_pid" 2>/dev/null || true
        sleep 1
    fi
    echo "  Stopped (pid $existing_pid)."
else
    echo "  No running server found."
fi
rm -f "$PID_FILE"

PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found"
    exit 1
fi
echo "  Python      : $PYTHON ($($PYTHON --version 2>&1))"

# 清掉舊的 .pyc
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "Starting training-only server..."
echo "  訓練完成後請執行 ./start_server.sh 還原推論 server"
echo "  Server successfully started with PID: $$"
echo "============================================================"

echo $$ > "$PID_FILE"
exec $PYTHON capi_server.py --config "$CONFIG_FILE" --training-only
