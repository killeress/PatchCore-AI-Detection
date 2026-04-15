#!/bin/bash
# ============================================================
# CAPI AI 推論伺服器啟動腳本（含 stop + restart）
# ============================================================
#
# 使用方式:
#   chmod +x start_server.sh
#   ./start_server.sh                    # 重啟（自動 stop 舊的 + start 新的）
#   ./start_server.sh start               # 啟動（若已在跑會拒絕）
#   ./start_server.sh stop                # 只停止
#   ./start_server.sh status              # 看目前狀態
#   ./start_server.sh -c my_config.yaml   # 指定設定檔（搭配上面任一動作）
#
# 停止流程：
#   先 SIGTERM 等 10 秒；還沒停就 SIGKILL。
# ============================================================

set -e

# 切換到腳本所在目錄
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 預設值
PID_FILE="/tmp/capi_server.pid"
LOG_DIR="/data/capi_ai/logs"
HEATMAP_DIR="/data/capi_ai/heatmaps"
CONFIG_FILE="server_config.yaml"
ACTION="restart"

# ----- 解析參數 -----
while [ $# -gt 0 ]; do
    case "$1" in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        start|stop|restart|status)
            ACTION="$1"
            shift
            ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [start|stop|restart|status] [-c config.yaml]"
            exit 1
            ;;
    esac
done

# ----- helper functions -----

# 檢查狀態：印出訊息並 return 0=running / 1=not running
check_status() {
    local pid=""
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE" 2>/dev/null)
    fi
    # 退而求其次 — 用 pgrep 找
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        pid=$(pgrep -f "capi_server.py" | head -1)
    fi

    if [ -z "$pid" ]; then
        echo "  Status: NOT RUNNING"
        return 1
    fi

    local cmd
    cmd=$(ps -o cmd= -p "$pid" 2>/dev/null || echo "")
    if echo "$cmd" | grep -q "capi_server.py"; then
        echo "  Status: RUNNING (pid=$pid)"
        return 0
    else
        echo "  Status: STALE pid file (pid=$pid is not capi_server)"
        return 1
    fi
}

# Graceful stop：SIGTERM → 等 10s → SIGKILL
stop_server() {
    local pid=""
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE" 2>/dev/null)
    fi
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        pid=$(pgrep -f "capi_server.py" | head -1)
    fi

    if [ -z "$pid" ]; then
        echo "  No running server found."
        rm -f "$PID_FILE"
        return 0
    fi

    echo "  Sending SIGTERM to pid $pid..."
    kill "$pid" 2>/dev/null || true

    local i=0
    while [ $i -lt 10 ] && kill -0 "$pid" 2>/dev/null; do
        sleep 1
        i=$((i + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        echo "  Still alive after 10s — sending SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi

    if kill -0 "$pid" 2>/dev/null; then
        echo "  ERROR: cannot stop pid $pid (please check manually)"
        return 1
    fi
    echo "  Stopped (pid $pid)."
    rm -f "$PID_FILE"
    return 0
}

# 啟動
start_server() {
    echo "============================================================"
    echo "  CAPI AI Inference Server"
    echo "============================================================"
    echo "  Working dir : $SCRIPT_DIR"
    echo "  Config      : $CONFIG_FILE"
    echo "  PID file    : $PID_FILE"
    echo "============================================================"

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: Config file not found: $CONFIG_FILE"
        exit 1
    fi

    mkdir -p "$LOG_DIR" "$HEATMAP_DIR"

    PYTHON=$(command -v python3 || command -v python)
    if [ -z "$PYTHON" ]; then
        echo "ERROR: Python not found"
        exit 1
    fi
    echo "  Python      : $PYTHON ($($PYTHON --version 2>&1))"

    echo ""
    echo "Checking dependencies..."
    $PYTHON -c "import yaml" 2>/dev/null || { echo "ERROR: PyYAML not installed. Run: pip install pyyaml"; exit 1; }
    $PYTHON -c "import cv2" 2>/dev/null || { echo "ERROR: OpenCV not installed. Run: pip install opencv-python"; exit 1; }
    $PYTHON -c "import numpy" 2>/dev/null || { echo "ERROR: NumPy not installed. Run: pip install numpy"; exit 1; }
    echo "Dependencies OK"

    # 清掉舊的 .pyc 確保載入新 code
    # （曾經發生 update .py 後 server 仍跑舊 module 的情況）
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

    echo ""
    echo "Starting server..."
    # exec 後 bash 變 python，$$ 會是接管後的 python pid
    echo $$ > "$PID_FILE"
    exec $PYTHON capi_server.py --config "$CONFIG_FILE"
}

# ----- main -----

case "$ACTION" in
    status)
        check_status
        ;;

    stop)
        echo "Stopping server..."
        stop_server
        ;;

    start)
        if check_status >/dev/null 2>&1; then
            echo "Server already running. Use 'restart' or 'stop' first."
            check_status
            exit 1
        fi
        start_server
        ;;

    restart)
        echo "Stopping existing server (if any)..."
        stop_server
        sleep 1
        start_server
        ;;
esac
