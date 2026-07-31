#!/bin/bash
# Install and activate the offline CAPI PaddleOCR MARK shadow bundle on RHEL 9.

set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_ROOT="${MARK_SHADOW_TARGET_ROOT:-/aidata/capi_ai/mark_shadow}"
APP_ROOT="${CAPI_APP_ROOT:-/root/Code/CAPI_AD}"
CONFIG_FILE="${CAPI_SERVER_CONFIG:-$APP_ROOT/server_config.yaml}"
RESTART_CAPI=1

if [ "${1:-}" = "--no-restart-capi" ]; then
    RESTART_CAPI=0
elif [ -n "${1:-}" ]; then
    echo "Usage: $0 [--no-restart-capi]"
    exit 1
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: install_offline.sh must run as root"
    exit 1
fi

for required in \
    "$BUNDLE_ROOT/manifest.json" \
    "$BUNDLE_ROOT/SHA256SUMS" \
    "$BUNDLE_ROOT/runtime/mark_paddle_cpu_env.tar.gz" \
    "$BUNDLE_ROOT/models/PP-OCRv6_medium_rec" \
    "$BUNDLE_ROOT/worker/paddle_shadow_worker.py" \
    "$BUNDLE_ROOT/scripts/capi-mark-shadow.service.template"
do
    if [ ! -e "$required" ]; then
        echo "ERROR: offline bundle item missing: $required"
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
command -v python3 >/dev/null 2>&1 || {
    echo "ERROR: python3 not found"
    exit 1
}
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: CAPI server config not found: $CONFIG_FILE"
    echo "Set CAPI_APP_ROOT or CAPI_SERVER_CONFIG to the production CAPI path."
    exit 1
fi
if [ "$RESTART_CAPI" -eq 1 ] && [ ! -f "$APP_ROOT/start_server.sh" ]; then
    echo "ERROR: CAPI restart script not found: $APP_ROOT/start_server.sh"
    echo "Set CAPI_APP_ROOT to the production CAPI path."
    exit 1
fi

echo "[1/7] Verifying offline bundle checksums..."
(cd "$BUNDLE_ROOT" && sha256sum -c SHA256SUMS)

VERSION="$(
    python3 - "$BUNDLE_ROOT/manifest.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as source:
    print(json.load(source)["bundle_version"])
PY
)"
SAFE_VERSION="${VERSION//[^A-Za-z0-9_.-]/_}"
RELEASE_DIR="$TARGET_ROOT/releases/$SAFE_VERSION"
DATA_DIR="$TARGET_ROOT/data"

echo "[2/7] Installing isolated runtime and models..."
if [ -e "$RELEASE_DIR" ]; then
    if [ ! -x "$RELEASE_DIR/runtime/bin/python" ] || \
        [ ! -f "$RELEASE_DIR/worker/paddle_shadow_worker.py" ] || \
        [ ! -d "$RELEASE_DIR/models/PP-OCRv6_medium_rec" ]; then
        echo "ERROR: incomplete existing release: $RELEASE_DIR"
        echo "Move this directory aside, then run the installer again."
        exit 1
    fi
    echo "  Release already installed; reusing: $RELEASE_DIR"
    mkdir -p "$DATA_DIR"
else
    mkdir -p "$RELEASE_DIR/runtime" "$RELEASE_DIR/models" "$RELEASE_DIR/worker" "$DATA_DIR"
    tar -xzf "$BUNDLE_ROOT/runtime/mark_paddle_cpu_env.tar.gz" \
        -C "$RELEASE_DIR/runtime"
    cp -a "$BUNDLE_ROOT/models/PP-OCRv6_medium_rec" "$RELEASE_DIR/models/"
    if [ -d "$BUNDLE_ROOT/models/PP-OCRv6_small_rec" ]; then
        cp -a "$BUNDLE_ROOT/models/PP-OCRv6_small_rec" "$RELEASE_DIR/models/"
    fi
    cp -a "$BUNDLE_ROOT/worker/." "$RELEASE_DIR/worker/"
    cp -a "$BUNDLE_ROOT/manifest.json" "$RELEASE_DIR/"
fi

if [ -x "$RELEASE_DIR/runtime/bin/conda-unpack" ]; then
    PATH="$RELEASE_DIR/runtime/bin:$PATH" \
        "$RELEASE_DIR/runtime/bin/conda-unpack"
fi

echo "[3/7] Verifying Python imports without network..."
"$RELEASE_DIR/runtime/bin/python" - <<'PY'
import paddle
import paddleocr
import cv2
print("paddle", paddle.__version__)
print("paddleocr", getattr(paddleocr, "__version__", "unknown"))
print("opencv", cv2.__version__)
PY

echo "[4/7] Installing systemd worker..."
UNIT_FILE="/etc/systemd/system/capi-mark-shadow.service"
sed \
    -e "s|@RELEASE_DIR@|$RELEASE_DIR|g" \
    -e "s|@DATA_DIR@|$DATA_DIR|g" \
    "$BUNDLE_ROOT/scripts/capi-mark-shadow.service.template" \
    > "$UNIT_FILE"
systemctl daemon-reload
systemctl enable capi-mark-shadow.service
systemctl restart capi-mark-shadow.service

echo "[5/7] Waiting for shadow health check..."
health_ok=0
for _ in $(seq 1 120); do
    if curl -fsS http://127.0.0.1:8765/health >/tmp/capi_mark_shadow_health.json 2>/dev/null; then
        health_ok=1
        break
    fi
    sleep 1
done
if [ "$health_ok" -ne 1 ]; then
    echo "ERROR: MARK shadow worker did not become healthy"
    systemctl status capi-mark-shadow.service --no-pager || true
    journalctl -u capi-mark-shadow.service -n 100 --no-pager || true
    exit 1
fi
cat /tmp/capi_mark_shadow_health.json
echo

echo "[6/7] Enabling online PaddleOCR recognition in server_config.yaml..."
CONFIG_BACKUP="$CONFIG_FILE.mark-shadow.$(date +%Y%m%d_%H%M%S).bak"
cp -a "$CONFIG_FILE" "$CONFIG_BACKUP"
python3 - "$CONFIG_FILE" "$DATA_DIR/mark_shadow.db" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
database_path = sys.argv[2]
lines = path.read_text(encoding="utf-8").splitlines()
start = next(
    (index for index, line in enumerate(lines) if re.match(r"^mark_shadow\s*:", line)),
    None,
)
if start is None:
    lines.extend([
        "",
        "# PaddleOCR MARK: primary formal text with DotMatrixCV locator/fallback.",
        "mark_shadow:",
        "  enabled: true",
        "  endpoint: http://127.0.0.1:8765/infer",
        f"  database_path: {database_path}",
        "  timeout_ms: 5000",
        "  max_queue: 64",
        "  crop_padding_ratio: 0.15",
    ])
else:
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line and not line[0].isspace():
            end = index
            break
    required = {
        "enabled": "true",
        "endpoint": "http://127.0.0.1:8765/infer",
        "database_path": database_path,
        "timeout_ms": "5000",
        "max_queue": "64",
        "crop_padding_ratio": "0.15",
    }
    indexes = {}
    for index in range(start + 1, end):
        match = re.match(r"^\s+([A-Za-z0-9_]+)\s*:", lines[index])
        if match:
            indexes[match.group(1)] = index
    for key in ("enabled", "endpoint", "database_path"):
        if key in indexes:
            indent = re.match(r"^(\s*)", lines[indexes[key]]).group(1)
            lines[indexes[key]] = f"{indent}{key}: {required[key]}"
    for key, value in required.items():
        if key not in indexes:
            lines.insert(end, f"  {key}: {value}")
            end += 1
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "  Config backup: $CONFIG_BACKUP"

echo "[7/7] Activating release and restarting CAPI..."
ln -sfn "$RELEASE_DIR" "$TARGET_ROOT/current"
if [ "$RESTART_CAPI" -eq 1 ]; then
    bash "$APP_ROOT/start_server.sh" restart --no-tail
else
    echo "Skipped CAPI restart. Restart it before expecting shadow records."
fi

echo
echo "MARK PaddleOCR online recognition installed."
echo "Health: curl http://127.0.0.1:8765/health"
echo "Stats : curl http://127.0.0.1:8765/stats"
echo "DB    : $DATA_DIR/mark_shadow.db"
echo "Log   : journalctl -u capi-mark-shadow.service -f"
echo "CAPI  : curl http://127.0.0.1/api/version"
