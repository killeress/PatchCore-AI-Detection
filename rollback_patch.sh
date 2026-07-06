#!/bin/bash
# Roll back files saved by install_patch.sh.
#
# Usage:
#   cd /root/Code/CAPI_AD
#   ./rollback_patch.sh .patch_backups/<version_timestamp>

set -euo pipefail

APP_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_ROOT"

BACKUP_DIR="${1:-}"

if [ -z "$BACKUP_DIR" ]; then
    echo "Usage: $0 <backup-dir>"
    exit 1
fi

if [ ! -d "$BACKUP_DIR" ]; then
    echo "ERROR: backup dir not found: $BACKUP_DIR"
    exit 1
fi

BACKUP_DIR="$(cd "$BACKUP_DIR" && pwd)"

echo "============================================================"
echo "  CAPI AI Patch Rollback"
echo "============================================================"
echo "  App root : $APP_ROOT"
echo "  Backup   : $BACKUP_DIR"
echo "============================================================"

echo "[1/3] Removing files created by the patch..."
if [ -f "$BACKUP_DIR/created_files.txt" ]; then
    while IFS= read -r entry; do
        [ -z "$entry" ] && continue
        case "$entry" in
            /*|../*|*"/../"*|*".."*) echo "ERROR: unsafe path in backup: $entry"; exit 1 ;;
        esac
        [ -f "$entry" ] && rm -f "$entry"
    done < "$BACKUP_DIR/created_files.txt"
fi

echo "[2/3] Restoring previous files..."
if [ -d "$BACKUP_DIR/files" ]; then
    (cd "$BACKUP_DIR/files" && find . -type f -print0) | while IFS= read -r -d '' rel; do
        rel="${rel#./}"
        case "$rel" in
            /*|../*|*"/../"*|*".."*) echo "ERROR: unsafe backup path: $rel"; exit 1 ;;
        esac
        mkdir -p "$(dirname "$APP_ROOT/$rel")"
        cp -a "$BACKUP_DIR/files/$rel" "$APP_ROOT/$rel"
    done
fi

chmod +x install_patch.sh rollback_patch.sh start_server.sh promote_update.sh setup_auto_update_client.sh 2>/dev/null || true

echo "[3/3] Restarting service..."
if [ -x "./start_server.sh" ]; then
    ./start_server.sh restart --no-tail
else
    echo "WARNING: start_server.sh not executable; please restart manually."
fi

mkdir -p "$APP_ROOT/update"
echo "$(date '+%F %T') rolled back backup=$BACKUP_DIR" >> "$APP_ROOT/update/update.log"

echo "Rollback completed."
