#!/bin/bash

set -euo pipefail

TARGET_ROOT="${MARK_SHADOW_TARGET_ROOT:-/aidata/capi_ai/mark_shadow}"

echo "=== SERVICE ==="
systemctl is-active capi-mark-shadow.service
systemctl status capi-mark-shadow.service --no-pager | head -30

echo
echo "=== HEALTH ==="
curl -fsS http://127.0.0.1:8765/health
echo

echo
echo "=== STATS ==="
curl -fsS http://127.0.0.1:8765/stats
echo

echo
echo "=== STORAGE ==="
du -sh "$TARGET_ROOT"
ls -lh "$TARGET_ROOT/data/mark_shadow.db" 2>/dev/null || true

echo
echo "=== CAPI CONFIG ==="
grep -A6 '^mark_shadow:' "${CAPI_SERVER_CONFIG:-/root/Code/CAPI_AD/server_config.yaml}" || true

echo
echo "=== RECENT LOGS ==="
journalctl -u capi-mark-shadow.service -n 30 --no-pager
