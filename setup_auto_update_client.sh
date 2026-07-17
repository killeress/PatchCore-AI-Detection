#!/bin/bash
# Register periodic update checks on a client host.
#
# Usage:
#   ./setup_auto_update_client.sh http://<update-host>:8088/latest.json
#   ./setup_auto_update_client.sh http://<update-host>:8088/latest.json --run-now

set -euo pipefail

APP_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_ROOT"

MANIFEST_URL="${1:-}"
RUN_NOW=0
HEALTH_URL="${CAPI_HEALTH_URL:-http://127.0.0.1/api/version}"
SCHEDULE="${CAPI_UPDATE_SCHEDULE:-*/5 * * * *}"
CRON_LOG="${CAPI_UPDATE_CRON_LOG:-update/auto_update_cron.log}"

if [ -z "$MANIFEST_URL" ]; then
    echo "Usage: $0 <manifest-url> [--run-now]"
    exit 1
fi

if [ "${2:-}" = "--run-now" ]; then
    RUN_NOW=1
elif [ -n "${2:-}" ]; then
    echo "ERROR: unknown argument: $2"
    exit 1
fi

PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: python3/python not found"
    exit 1
fi

command -v crontab >/dev/null 2>&1 || { echo "ERROR: crontab not found"; exit 1; }

mkdir -p "$(dirname "$CRON_LOG")"

BEGIN_MARK="# CAPI_AUTO_UPDATE_BEGIN"
END_MARK="# CAPI_AUTO_UPDATE_END"
CRON_LINE="$SCHEDULE cd $APP_ROOT && $PYTHON_BIN capi_update_agent.py check --manifest-url $MANIFEST_URL --health-url $HEALTH_URL >> $CRON_LOG 2>&1"

TMP_CRON="$(mktemp)"
trap 'rm -f "$TMP_CRON"' EXIT

crontab -l 2>/dev/null | sed "/$BEGIN_MARK/,/$END_MARK/d" > "$TMP_CRON" || true
{
    echo "$BEGIN_MARK"
    echo "$CRON_LINE"
    echo "$END_MARK"
} >> "$TMP_CRON"
crontab "$TMP_CRON"

echo "============================================================"
echo "  CAPI AI Auto Update Client"
echo "============================================================"
echo "  App root     : $APP_ROOT"
echo "  Manifest URL : $MANIFEST_URL"
echo "  Health URL   : $HEALTH_URL"
echo "  Schedule     : $SCHEDULE"
echo "  Cron log     : $CRON_LOG"
echo "============================================================"

echo "[1/2] Dry-run connectivity check..."
"$PYTHON_BIN" capi_update_agent.py check \
    --manifest-url "$MANIFEST_URL" \
    --health-url "$HEALTH_URL" \
    --dry-run

if [ "$RUN_NOW" -eq 1 ]; then
    echo "[2/2] Checking and staging an update now..."
    "$PYTHON_BIN" capi_update_agent.py check \
        --manifest-url "$MANIFEST_URL" \
        --health-url "$HEALTH_URL"
else
    echo "[2/2] Cron installed. Add --run-now to check immediately."
fi

echo "Update-check client setup completed. Pending updates must be applied from the frontend."
