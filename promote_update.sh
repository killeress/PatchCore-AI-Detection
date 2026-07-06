#!/bin/bash
# Install one update package on this host, then publish it for other hosts.
#
# Usage:
#   ./promote_update.sh /aidata/capi_ai/update_repo/staging/patchcore_ai_release_<version>_codeonly.zip

set -euo pipefail

APP_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_ROOT"

PACKAGE="${1:-}"
UPDATE_REPO="${CAPI_UPDATE_REPO:-/aidata/capi_ai/update_repo}"
HEALTH_URL="${CAPI_HEALTH_URL:-http://127.0.0.1/api/version}"
HTTP_PORT="${CAPI_UPDATE_HTTP_PORT:-8088}"
HTTP_LOG="${CAPI_UPDATE_HTTP_LOG:-/aidata/capi_ai/logs/update_repo_http.log}"

if [ -z "$PACKAGE" ]; then
    echo "Usage: $0 <release-zip>"
    exit 1
fi

if [ ! -f "$PACKAGE" ]; then
    echo "ERROR: package not found: $PACKAGE"
    exit 1
fi

PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: python3/python not found"
    exit 1
fi

command -v curl >/dev/null 2>&1 || { echo "ERROR: curl not found"; exit 1; }

PACKAGE="$(cd "$(dirname "$PACKAGE")" && pwd)/$(basename "$PACKAGE")"

echo "============================================================"
echo "  CAPI AI Update Promote"
echo "============================================================"
echo "  App root    : $APP_ROOT"
echo "  Package     : $PACKAGE"
echo "  Update repo : $UPDATE_REPO"
echo "  Health URL  : $HEALTH_URL"
echo "  HTTP port   : $HTTP_PORT"
echo "============================================================"

echo "[1/4] Installing package on this host..."
CAPI_HEALTH_URL="$HEALTH_URL" ./install_patch.sh "$PACKAGE"

echo "[2/4] Publishing package metadata..."
mkdir -p "$UPDATE_REPO"
"$PYTHON_BIN" capi_update_agent.py publish \
    --package "$PACKAGE" \
    --output-dir "$UPDATE_REPO"

echo "[3/4] Ensuring update HTTP server is running..."
mkdir -p "$(dirname "$HTTP_LOG")"
if curl -fsS "http://127.0.0.1:${HTTP_PORT}/latest.json" >/dev/null 2>&1; then
    echo "  HTTP server already serves latest.json on port $HTTP_PORT"
else
    (
        cd "$UPDATE_REPO"
        nohup "$PYTHON_BIN" -m http.server "$HTTP_PORT" --bind 0.0.0.0 >"$HTTP_LOG" 2>&1 &
    )
    sleep 2
fi

echo "[4/4] Verifying published latest.json..."
curl -fsS "http://127.0.0.1:${HTTP_PORT}/latest.json"
echo ""

echo "============================================================"
echo "Promote completed."
echo "Client manifest URL:"
echo "  http://<this-host-ip>:${HTTP_PORT}/latest.json"
echo "============================================================"
