#!/bin/bash
# Install one update package on this host, then publish it for other hosts.
#
# Usage:
#   ./promote_update.sh
#   ./promote_update.sh /aidata/capi_ai/update_repo/staging/patchcore_ai_release_<version>_codeonly.zip
#   ./promote_update.sh --serve-only  # recover HTTP after install/publish succeeded

set -euo pipefail

APP_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_ROOT"

UPDATE_REPO="${CAPI_UPDATE_REPO:-/aidata/capi_ai/update_repo}"
PACKAGE_DIR="${CAPI_UPDATE_PACKAGE_DIR:-$UPDATE_REPO/staging}"
PACKAGE="${1:-}"
HEALTH_URL="${CAPI_HEALTH_URL:-http://127.0.0.1/api/version}"
HTTP_PORT="${CAPI_UPDATE_HTTP_PORT:-8088}"
HTTP_LOG="${CAPI_UPDATE_HTTP_LOG:-/aidata/capi_ai/logs/update_repo_http.log}"
HTTP_START_TIMEOUT="${CAPI_UPDATE_HTTP_START_TIMEOUT_SECONDS:-120}"
SERVE_ONLY=0
if [ "$PACKAGE" = "--serve-only" ]; then
    SERVE_ONLY=1
    PACKAGE=""
fi
if ! [[ "$HTTP_START_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CAPI_UPDATE_HTTP_START_TIMEOUT_SECONDS must be a positive integer"
    exit 1
fi

if [ "$SERVE_ONLY" -eq 0 ] && [ -z "$PACKAGE" ]; then
    PACKAGE_NAME="$(find "$PACKAGE_DIR" -maxdepth 1 -type f \
        -name 'patchcore_ai_release_*_codeonly.zip' \
        -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)"
    if [ -z "$PACKAGE_NAME" ]; then
        echo "ERROR: no code-only release ZIP found in $PACKAGE_DIR"
        echo "Usage: $0 [release-zip]"
        exit 1
    fi
    PACKAGE="$PACKAGE_DIR/$PACKAGE_NAME"
    echo "Auto-selected package: $PACKAGE"
fi

if [ "$SERVE_ONLY" -eq 0 ] && [ ! -f "$PACKAGE" ]; then
    echo "ERROR: package not found: $PACKAGE"
    exit 1
fi

PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: python3/python not found"
    exit 1
fi

command -v curl >/dev/null 2>&1 || { echo "ERROR: curl not found"; exit 1; }

if [ "$SERVE_ONLY" -eq 0 ]; then
    PACKAGE="$(cd "$(dirname "$PACKAGE")" && pwd)/$(basename "$PACKAGE")"
elif [ ! -s "$UPDATE_REPO/latest.json" ]; then
    echo "ERROR: --serve-only requires existing $UPDATE_REPO/latest.json"
    exit 1
fi

echo "============================================================"
echo "  CAPI AI Update Promote"
echo "============================================================"
echo "  App root    : $APP_ROOT"
echo "  Package     : $PACKAGE"
echo "  Update repo : $UPDATE_REPO"
echo "  Health URL  : $HEALTH_URL"
echo "  HTTP port   : $HTTP_PORT"
echo "============================================================"

if [ "$SERVE_ONLY" -eq 0 ]; then
    echo "[1/4] Installing package on this host..."
    CAPI_HEALTH_URL="$HEALTH_URL" ./install_patch.sh "$PACKAGE"

    echo "[2/4] Publishing package metadata..."
    mkdir -p "$UPDATE_REPO"
    "$PYTHON_BIN" capi_update_agent.py publish \
        --package "$PACKAGE" \
        --output-dir "$UPDATE_REPO"
else
    echo "[1/4] Skipping install (--serve-only; AI service is not restarted)."
    echo "[2/4] Keeping existing published package metadata."
fi

echo "[3/4] Ensuring update HTTP server is running..."
mkdir -p "$(dirname "$HTTP_LOG")"
HTTP_URL="http://127.0.0.1:${HTTP_PORT}/latest.json"
HTTP_PROBE_ERROR=""
probe_update_http() {
    # Ignore proxy environment and ~/.curlrc for this loopback-only health check.
    # Preserve stderr so a connection failure, timeout and HTTP error are distinct.
    if HTTP_PROBE_ERROR="$(curl -q --noproxy '*' --connect-timeout 2 \
        --max-time "${1:-3}" -fsS "$HTTP_URL" 2>&1 >/dev/null)"; then
        return 0
    fi
    return 1
}
if probe_update_http; then
    echo "  HTTP server already serves latest.json on port $HTTP_PORT"
else
    echo "  Starting HTTP server on port $HTTP_PORT (timeout=${HTTP_START_TIMEOUT}s)..."
    echo "  Python: $PYTHON_BIN"
    nohup "$PYTHON_BIN" -u -m http.server "$HTTP_PORT" \
        --bind 0.0.0.0 \
        --directory "$UPDATE_REPO" \
        >"$HTTP_LOG" 2>&1 </dev/null &
    HTTP_PID=$!
    http_ready=0
    http_deadline=$((SECONDS + HTTP_START_TIMEOUT))
    while [ "$SECONDS" -lt "$http_deadline" ]; do
        remaining=$((http_deadline - SECONDS))
        probe_timeout=3
        if [ "$remaining" -lt "$probe_timeout" ]; then
            probe_timeout=$remaining
        fi
        if probe_update_http "$probe_timeout"; then
            http_ready=1
            break
        fi
        if ! kill -0 "$HTTP_PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done

    if [ "$http_ready" -ne 1 ]; then
        echo "ERROR: update HTTP server did not serve latest.json on port $HTTP_PORT"
        echo "  Last local HTTP probe: $HTTP_PROBE_ERROR"
        if kill -0 "$HTTP_PID" 2>/dev/null; then
            echo "  HTTP server process is still running (pid=$HTTP_PID); stopping it."
            kill "$HTTP_PID" 2>/dev/null || true
        else
            echo "  HTTP server process exited during startup (pid=$HTTP_PID)."
        fi
        if [ -f "$HTTP_LOG" ]; then
            echo "  HTTP server log:"
            tail -n 50 "$HTTP_LOG"
        fi
        if command -v ss >/dev/null 2>&1; then
            echo "  Listening sockets on port $HTTP_PORT:"
            ss -ltnp "sport = :$HTTP_PORT" || true
        fi
        echo "  Installation and published metadata are retained."
        echo "  Retry HTTP only: $0 --serve-only"
        exit 1
    fi
    echo "  HTTP server started (pid=$HTTP_PID)"
fi

echo "[4/4] Verifying published latest.json..."
curl -q --noproxy '*' --connect-timeout 2 --max-time 5 -fsS "$HTTP_URL"
echo ""

echo "============================================================"
echo "Promote completed."
echo "Client manifest URL:"
echo "  http://<this-host-ip>:${HTTP_PORT}/latest.json"
echo "============================================================"
