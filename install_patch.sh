#!/bin/bash
# Install a CAPI AI patch ZIP from the application root.
#
# Usage:
#   cd /root/Code/CAPI_AD
#   ./install_patch.sh /path/to/patchcore_ai_patch_YYYY.MM.DD.N.zip

set -euo pipefail

APP_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_ROOT"

PATCH_ZIP="${1:-}"
HEALTH_URL="${CAPI_HEALTH_URL:-http://127.0.0.1:8080/api/version}"
BACKUP_ROOT="${CAPI_PATCH_BACKUP_ROOT:-$APP_ROOT/.patch_backups}"

if [ -z "$PATCH_ZIP" ]; then
    echo "Usage: $0 <patch-zip>"
    exit 1
fi

if [ ! -f "$PATCH_ZIP" ]; then
    echo "ERROR: patch ZIP not found: $PATCH_ZIP"
    exit 1
fi

command -v unzip >/dev/null 2>&1 || { echo "ERROR: unzip not found"; exit 1; }

PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: python3/python not found"
    exit 1
fi

PATCH_ZIP="$(cd "$(dirname "$PATCH_ZIP")" && pwd)/$(basename "$PATCH_ZIP")"
VERSION_IN_ZIP="$("$PYTHON_BIN" - "$PATCH_ZIP" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    try:
        print(z.read("VERSION").decode("utf-8").strip() or "unknown")
    except KeyError:
        print("unknown")
PY
)"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$BACKUP_ROOT/${VERSION_IN_ZIP}_${STAMP}"

echo "============================================================"
echo "  CAPI AI Patch Installer"
echo "============================================================"
echo "  App root : $APP_ROOT"
echo "  Patch    : $PATCH_ZIP"
echo "  Version  : $VERSION_IN_ZIP"
echo "  Backup   : $BACKUP_DIR"
echo "============================================================"

echo "[1/5] Verifying ZIP checksums..."
"$PYTHON_BIN" - "$PATCH_ZIP" <<'PY'
import hashlib
import sys
import zipfile
from pathlib import PurePosixPath

zip_path = sys.argv[1]
with zipfile.ZipFile(zip_path) as z:
    names = set(z.namelist())
    if "checksums.txt" not in names:
        raise SystemExit("checksums.txt missing")
    for name in names:
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts:
            raise SystemExit(f"unsafe ZIP path: {name}")
    for line in z.read("checksums.txt").decode("utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1)
        name = name.strip()
        data = z.read(name)
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise SystemExit(f"checksum mismatch: {name}")
print("  Checksums OK")
PY

echo "[2/5] Backing up files that will be replaced..."
mkdir -p "$BACKUP_DIR/files"
: > "$BACKUP_DIR/created_files.txt"

while IFS= read -r entry; do
    [ -z "$entry" ] && continue
    case "$entry" in
        */) continue ;;
        /*|../*|*"/../"*|*".."*) echo "ERROR: unsafe ZIP path: $entry"; exit 1 ;;
    esac
    if [ -f "$entry" ]; then
        mkdir -p "$BACKUP_DIR/files/$(dirname "$entry")"
        cp -a "$entry" "$BACKUP_DIR/files/$entry"
    else
        echo "$entry" >> "$BACKUP_DIR/created_files.txt"
    fi
done < <(unzip -Z1 "$PATCH_ZIP")

echo "$PATCH_ZIP" > "$BACKUP_DIR/patch_zip.txt"

echo "[3/5] Extracting patch..."
unzip -o "$PATCH_ZIP" -d "$APP_ROOT"
chmod +x install_patch.sh rollback_patch.sh start_server.sh 2>/dev/null || true

echo "[4/5] Restarting service..."
if [ -x "./start_server.sh" ]; then
    ./start_server.sh restart --no-tail
else
    echo "WARNING: start_server.sh not executable; please restart manually."
fi

echo "[5/5] Health check..."
if command -v curl >/dev/null 2>&1; then
    ok=0
    for _ in $(seq 1 20); do
        if curl -fsS "$HEALTH_URL" >/tmp/capi_patch_health.json 2>/dev/null; then
            ok=1
            break
        fi
        sleep 1
    done
    if [ "$ok" -eq 1 ]; then
        echo "  Health check OK: $HEALTH_URL"
        cat /tmp/capi_patch_health.json
        echo ""
    else
        echo "WARNING: health check failed: $HEALTH_URL"
        echo "Rollback command:"
        echo "  ./rollback_patch.sh \"$BACKUP_DIR\""
        exit 2
    fi
else
    echo "WARNING: curl not found; skipped health check."
fi

mkdir -p "$APP_ROOT/update"
{
    echo "$(date '+%F %T') installed $PATCH_ZIP"
    echo "backup=$BACKUP_DIR"
} >> "$APP_ROOT/update/update.log"

echo "============================================================"
echo "Patch installed."
echo "Rollback command:"
echo "  ./rollback_patch.sh \"$BACKUP_DIR\""
echo "============================================================"
