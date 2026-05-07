#!/bin/bash
# ============================================================
# CAPI AI 推論伺服器啟動腳本（含 stop + restart + log）
# ============================================================
#
# 使用方式:
#   chmod +x start_server.sh
#   ./start_server.sh                    # 重啟（自動 stop 舊的 + start 新的，並顯示 log）
#   ./start_server.sh start              # 啟動（若已在跑會拒絕，啟動後顯示 log）
#   ./start_server.sh stop               # 只停止
#   ./start_server.sh status             # 看目前狀態
#   ./start_server.sh log                # 實時查看伺服器 Log
#   ./start_server.sh -c my_config.yaml  # 指定設定檔（搭配上面任一動作）
#
# ============================================================

set -e

# 切換到腳本所在目錄
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 預設值
PID_FILE="/tmp/capi_server.pid"
LOG_DIR="/aidata/capi_ai/logs"
HEATMAP_DIR="/aidata/capi_ai/heatmaps"
CONFIG_FILE="server_config.yaml"
SERVER_LOG_LATEST="$LOG_DIR/server_output.log"   # symlink → 當天那份 server_output_YYYY-MM-DD.log
ACTION="restart"

# ----- 解析參數 -----
while [ $# -gt 0 ]; do
    case "$1" in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        start|stop|restart|status|log)
            ACTION="$1"
            shift
            ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [start|stop|restart|status|log] [-c config.yaml]"
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

# Graceful stop
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

    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

    local TODAY
    TODAY=$(date +%Y-%m-%d)
    local SERVER_LOG_DAILY="$LOG_DIR/server_output_${TODAY}.log"

    # 舊版 server_output.log 是 regular file，先搬開避免被下方 ln -sfn 蓋掉舊內容
    if [ -f "$SERVER_LOG_LATEST" ] && [ ! -L "$SERVER_LOG_LATEST" ]; then
        mv "$SERVER_LOG_LATEST" "$LOG_DIR/server_output_legacy.log"
        echo "  Migrated old log: server_output.log → server_output_legacy.log"
    fi

    # 用相對路徑，LOG_DIR 整體搬移時 symlink 不會壞；
    # 在 nohup 之前先建好，避免 tail_logs 看到舊 target
    ln -sfn "$(basename "$SERVER_LOG_DAILY")" "$SERVER_LOG_LATEST"

    echo ""
    echo "Starting server in background..."

    nohup "$PYTHON" capi_server.py --config "$CONFIG_FILE" >> "$SERVER_LOG_DAILY" 2>&1 &

    local NEW_PID=$!
    echo $NEW_PID > "$PID_FILE"

    echo "  Server successfully started with PID: $NEW_PID"
    echo "  Log file    : $SERVER_LOG_DAILY"
    echo "  Latest link : $SERVER_LOG_LATEST"
    echo "============================================================"
}

# 實時顯示 Log
tail_logs() {
    if [ ! -e "$SERVER_LOG_LATEST" ]; then
        echo "Log file does not exist yet: $SERVER_LOG_LATEST"
        return
    fi
    echo ">> 正在即時顯示 Log 輸出..."
    echo ">> (提示: 按下 Ctrl+C 退出日誌檢視，伺服器仍會在背景繼續執行)"
    echo ">> (跨日重啟後 symlink 會自動指向新檔，tail -F 會跟著切換)"
    echo "------------------------------------------------------------"
    sleep 1 # 等待 Python 啟動並寫入第一行 log
    # -F = --follow=name --retry，跨日 symlink 換指向時會 reopen 新檔
    tail -F "$SERVER_LOG_LATEST"
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

    log)
        tail_logs
        ;;

    start)
        if check_status >/dev/null 2>&1; then
            echo "Server already running. Use 'restart' or 'stop' first."
            check_status
            exit 1
        fi
        start_server
        tail_logs
        ;;

    restart)
        echo "Stopping existing server (if any)..."
        stop_server
        sleep 1
        start_server
        tail_logs
        ;;
esac
