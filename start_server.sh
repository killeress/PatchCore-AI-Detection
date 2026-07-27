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
#   ./start_server.sh restart --no-tail  # 重啟後不進入 log tail（給更新腳本用）
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
TAIL_AFTER_START=1
PORT_WAIT_SECONDS=10

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
        --no-tail)
            TAIL_AFTER_START=0
            shift
            ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [start|stop|restart|status|log] [-c config.yaml] [--no-tail]"
            exit 1
            ;;
    esac
done

# ----- helper functions -----

# 讀取並驗證 PID file；無效內容不應被拿去 kill。
read_pid_file() {
    local pid=""
    if [ -f "$PID_FILE" ]; then
        pid=$(tr -d '[:space:]' < "$PID_FILE" 2>/dev/null || true)
    fi
    case "$pid" in
        ''|*[!0-9]*) return 1 ;;
        *) echo "$pid" ;;
    esac
}

# 取得 Linux process state，例如 S、R、Z。
pid_state() {
    ps -o stat= -p "$1" 2>/dev/null | awk '{print $1}'
}

pid_is_zombie() {
    local state
    state=$(pid_state "$1")
    case "$state" in
        Z*) return 0 ;;
        *) return 1 ;;
    esac
}

# 只接受仍在執行、且 command 確實是 capi_server.py 的 PID。
pid_is_server() {
    local pid="$1"
    local cmd=""
    case "$pid" in
        ''|*[!0-9]*) return 1 ;;
    esac
    kill -0 "$pid" 2>/dev/null || return 1
    pid_is_zombie "$pid" && return 1
    cmd=$(ps -o args= -p "$pid" 2>/dev/null || true)
    case "$cmd" in
        *capi_server.py*) return 0 ;;
        *) return 1 ;;
    esac
}

# 從 process table 找仍在執行的 CAPI server，排除 Zombie。
find_server_pid() {
    local candidate
    for candidate in $(pgrep -f "capi_server.py" 2>/dev/null || true); do
        if pid_is_server "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# 依實際設定檔檢查 Web/TCP port，避免啟動後才在 log 中發現衝突。
check_configured_ports() {
    "$PYTHON" - "$CONFIG_FILE" <<'PY'
import shutil
import socket
import sys
import subprocess

import yaml

config_path = sys.argv[1]
with open(config_path, encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file) or {}

server = config.get("server") or {}
web = config.get("web") or {}
ports = []
if web.get("enabled", True):
    ports.append(("Web", web.get("host", "0.0.0.0"), int(web.get("port", 8080))))
ports.append(("TCP", server.get("host", "0.0.0.0"), int(server.get("port", 7907))))

sockets = []
failed = False
for service, host, port in ports:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as exc:
        print(f"ERROR: {service} port is unavailable: {host}:{port} ({exc})", file=sys.stderr)
        owner_command = None
        if shutil.which("ss"):
            owner_command = ["ss", "-ltnp", f"sport = :{port}"]
        elif shutil.which("lsof"):
            owner_command = ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"]
        if owner_command:
            owner = subprocess.run(owner_command, capture_output=True, text=True, check=False)
            if owner.stdout.strip():
                print(owner.stdout.rstrip(), file=sys.stderr)
        else:
            print(f"Identify it with: ss -ltnp 'sport = :{port}'", file=sys.stderr)
        failed = True
        sock.close()
    else:
        sockets.append(sock)

for sock in sockets:
    sock.close()
sys.exit(1 if failed else 0)
PY
}

# Restart 時給前一個程序一點時間釋放 listener；持續占用時仍會 fail-fast。
wait_for_configured_ports() {
    local attempt=0
    while [ "$attempt" -lt "$PORT_WAIT_SECONDS" ]; do
        if check_configured_ports >/dev/null 2>&1; then
            return 0
        fi
        attempt=$((attempt + 1))
        echo "  Configured ports still busy; waiting ${attempt}/${PORT_WAIT_SECONDS}s..."
        sleep 1
    done

    check_configured_ports
    return 1
}

# 檢查狀態：印出訊息並 return 0=running / 1=not running
check_status() {
    local pid=""
    if pid=$(read_pid_file); then
        if pid_is_server "$pid"; then
            echo "  Status: RUNNING (pid=$pid)"
            return 0
        fi
    fi

    if pid=$(find_server_pid); then
        echo "  Status: RUNNING (pid=$pid)"
        return 0
    fi

    if [ -z "$pid" ]; then
        echo "  Status: NOT RUNNING"
        return 1
    fi

    if pid_is_zombie "$pid"; then
        echo "  Status: STALE zombie PID file (pid=$pid)"
    else
        echo "  Status: STALE pid file (pid=$pid is not capi_server)"
    fi
    return 1
}

# Graceful stop
stop_server() {
    local pid=""
    if pid=$(read_pid_file); then
        if pid_is_zombie "$pid"; then
            echo "  PID file points to zombie pid $pid; removing stale PID file."
            rm -f "$PID_FILE"
            pid=""
        elif ! pid_is_server "$pid"; then
            echo "  PID file does not point to a running capi_server; removing stale PID file."
            rm -f "$PID_FILE"
            pid=""
        fi
    fi
    if [ -z "$pid" ]; then
        pid=$(find_server_pid || true)
    fi

    if [ -z "$pid" ]; then
        echo "  No running server found."
        rm -f "$PID_FILE"
        return 0
    fi

    echo "  Sending SIGTERM to pid $pid..."
    kill "$pid" 2>/dev/null || true

    local i=0
    while [ $i -lt 10 ] && pid_is_server "$pid"; do
        sleep 1
        i=$((i + 1))
    done

    if pid_is_server "$pid"; then
        echo "  Still alive after 10s — sending SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi

    if pid_is_server "$pid"; then
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

    PYTHON="${CAPI_PYTHON_BIN:-}"
    if [ -z "$PYTHON" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python3" ]; then
        PYTHON="$CONDA_PREFIX/bin/python3"
    fi
    if [ -z "$PYTHON" ]; then
        for candidate in \
            /opt/miniconda3/envs/CAPI-PC/bin/python3 \
            /opt/miniconda3/envs/CAPI-PC/bin/python \
            /root/miniconda3/envs/CAPI-PC/bin/python3
        do
            if [ -x "$candidate" ]; then
                PYTHON="$candidate"
                break
            fi
        done
    fi
    if [ -z "$PYTHON" ]; then
        PYTHON=$(command -v python3 || command -v python || true)
    fi
    if [ -z "$PYTHON" ]; then
        echo "ERROR: Python not found"
        exit 1
    fi
    if ! "$PYTHON" --version >/dev/null 2>&1; then
        echo "ERROR: configured Python not executable: $PYTHON"
        exit 1
    fi
    echo "  Python      : $PYTHON ($($PYTHON --version 2>&1))"

    echo ""
    echo "Checking dependencies..."
    $PYTHON -c "import yaml" 2>/dev/null || { echo "ERROR: PyYAML not installed. Run: pip install pyyaml"; exit 1; }
    $PYTHON -c "import cv2" 2>/dev/null || { echo "ERROR: OpenCV not installed. Run: pip install opencv-python"; exit 1; }
    $PYTHON -c "import numpy" 2>/dev/null || { echo "ERROR: NumPy not installed. Run: pip install numpy"; exit 1; }
    echo "Dependencies OK"

    echo "Checking configured ports..."
    if ! wait_for_configured_ports; then
        exit 1
    fi
    echo "Configured ports are available"

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

    # 建立獨立 session，Ctrl+C 只停止目前的 log tail，不傳給背景 Server。
    if command -v setsid >/dev/null 2>&1; then
        nohup setsid "$PYTHON" capi_server.py --config "$CONFIG_FILE" \
            >> "$SERVER_LOG_DAILY" 2>&1 < /dev/null &
    else
        nohup "$PYTHON" capi_server.py --config "$CONFIG_FILE" \
            >> "$SERVER_LOG_DAILY" 2>&1 < /dev/null &
    fi

    local NEW_PID=$!
    printf '%s\n' "$NEW_PID" > "$PID_FILE"

    local i=0
    while [ $i -lt 5 ]; do
        if pid_is_server "$NEW_PID" || pid_is_zombie "$NEW_PID"; then
            break
        fi
        if ! kill -0 "$NEW_PID" 2>/dev/null; then
            break
        fi
        sleep 1
        i=$((i + 1))
    done

    if ! pid_is_server "$NEW_PID"; then
        echo "ERROR: Server exited during startup (pid=$NEW_PID)"
        rm -f "$PID_FILE"
        echo "Last server log lines:"
        tail -n 40 "$SERVER_LOG_DAILY" 2>/dev/null || true
        return 1
    fi

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
        if [ "$TAIL_AFTER_START" -eq 1 ]; then
            tail_logs
        else
            check_status
        fi
        ;;

    restart)
        echo "Stopping existing server (if any)..."
        stop_server
        sleep 1
        start_server
        if [ "$TAIL_AFTER_START" -eq 1 ]; then
            tail_logs
        else
            check_status
        fi
        ;;
esac
