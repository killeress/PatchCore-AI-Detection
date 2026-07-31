#!/bin/bash
# Apply a worker-only MARK shadow hotfix without reinstalling PaddleOCR runtime/models.

set -euo pipefail

PATCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_ROOT="${MARK_SHADOW_TARGET_ROOT:-/aidata/capi_ai/mark_shadow}"
CURRENT_LINK="$TARGET_ROOT/current"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: install_worker_hotfix.sh must run as root"
    exit 1
fi

for required in \
    "$PATCH_ROOT/SHA256SUMS" \
    "$PATCH_ROOT/worker/paddle_shadow_worker.py"
do
    if [ ! -f "$required" ]; then
        echo "ERROR: hotfix item missing: $required"
        exit 1
    fi
done

command -v sha256sum >/dev/null 2>&1 || {
    echo "ERROR: sha256sum not found"
    exit 1
}
command -v systemctl >/dev/null 2>&1 || {
    echo "ERROR: systemctl not found"
    exit 1
}
command -v curl >/dev/null 2>&1 || {
    echo "ERROR: curl not found"
    exit 1
}

echo "[1/5] Verifying hotfix checksums..."
(cd "$PATCH_ROOT" && sha256sum -c SHA256SUMS)

if [ ! -L "$CURRENT_LINK" ]; then
    echo "ERROR: MARK shadow current release link not found: $CURRENT_LINK"
    exit 1
fi
RELEASE_DIR="$(readlink -f "$CURRENT_LINK")"
case "$RELEASE_DIR" in
    "$TARGET_ROOT"/releases/*) ;;
    *)
        echo "ERROR: current release is outside $TARGET_ROOT/releases: $RELEASE_DIR"
        exit 1
        ;;
esac

TARGET_WORKER="$RELEASE_DIR/worker/paddle_shadow_worker.py"
RUNTIME_PYTHON="$RELEASE_DIR/runtime/bin/python"
if [ ! -f "$TARGET_WORKER" ] || [ ! -x "$RUNTIME_PYTHON" ]; then
    echo "ERROR: installed worker/runtime is incomplete: $RELEASE_DIR"
    exit 1
fi

echo "[2/5] Validating patched worker..."
"$RUNTIME_PYTHON" -m py_compile "$PATCH_ROOT/worker/paddle_shadow_worker.py"

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$TARGET_WORKER.before-shadow-hotfix.$STAMP.bak"
cp -a "$TARGET_WORKER" "$BACKUP"

echo "[3/5] Replacing worker (backup: $BACKUP)..."
install -m 0755 "$PATCH_ROOT/worker/paddle_shadow_worker.py" "$TARGET_WORKER.new"
mv -f "$TARGET_WORKER.new" "$TARGET_WORKER"

echo "[4/5] Restarting MARK shadow service..."
if ! systemctl restart capi-mark-shadow.service; then
    cp -a "$BACKUP" "$TARGET_WORKER"
    systemctl restart capi-mark-shadow.service || true
    echo "ERROR: service restart failed; original worker restored"
    exit 1
fi

echo "[5/5] Waiting for health check..."
health_ok=0
for _ in $(seq 1 120); do
    if curl -fsS http://127.0.0.1:8765/health \
        >/tmp/capi_mark_shadow_hotfix_health.json 2>/dev/null
    then
        health_ok=1
        break
    fi
    sleep 1
done
if [ "$health_ok" -ne 1 ]; then
    cp -a "$BACKUP" "$TARGET_WORKER"
    systemctl restart capi-mark-shadow.service || true
    echo "ERROR: patched worker did not become healthy; original worker restored"
    journalctl -u capi-mark-shadow.service -n 100 --no-pager || true
    exit 1
fi

cat /tmp/capi_mark_shadow_hotfix_health.json
echo
echo "MARK shadow worker hotfix installed."
echo "Existing error rows remain unchanged; verify a newly collected row."
