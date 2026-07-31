#!/bin/bash
# Build a relocatable RHEL 9 compatible CPU PaddleOCR MARK shadow bundle.
#
# Run on an internet-connected Linux x86_64 host:
#   bash scripts/build_mark_shadow_offline_bundle.sh [output-directory]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${1:-$PROJECT_ROOT/deployment}"
BUILD_ROOT="${MARK_SHADOW_BUILD_ROOT:-/tmp/capi_mark_shadow_build}"
DEFAULT_BUNDLE_VERSION="$(tr -d '\r\n' < "$PROJECT_ROOT/VERSION")"
BUNDLE_VERSION="${MARK_SHADOW_BUNDLE_VERSION:-$DEFAULT_BUNDLE_VERSION}"
BUNDLE_NAME="mark_paddle_shadow_rhel9_py312_cpu_${BUNDLE_VERSION}"
BUNDLE_DIR="$OUTPUT_DIR/$BUNDLE_NAME"
ARCHIVE_PATH="$OUTPUT_DIR/$BUNDLE_NAME.tar.gz"
REUSE_BUILD="${MARK_SHADOW_REUSE_BUILD:-0}"

MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
MEDIUM_MODEL_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_medium_rec_infer.tar"
SMALL_MODEL_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_small_rec_infer.tar"

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "ERROR: build requires Linux x86_64"
    exit 1
fi

for command_name in curl tar sha256sum; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "ERROR: required command not found: $command_name"
        exit 1
    }
done

mkdir -p "$OUTPUT_DIR"
if [ "$REUSE_BUILD" != "1" ]; then
    rm -rf "$BUILD_ROOT"
fi
mkdir -p "$BUILD_ROOT"

echo "[1/8] Installing isolated Miniforge build environment..."
if [ ! -x "$BUILD_ROOT/runtime/bin/python" ]; then
    curl -fL --retry 3 "$MINIFORGE_URL" -o "$BUILD_ROOT/miniforge.sh"
    bash "$BUILD_ROOT/miniforge.sh" -b -p "$BUILD_ROOT/miniforge"
    "$BUILD_ROOT/miniforge/bin/conda" create -y \
        -p "$BUILD_ROOT/runtime" \
        -c conda-forge \
        python=3.12 pip conda-pack libgl libglib
else
    "$BUILD_ROOT/miniforge/bin/conda" install -y \
        -p "$BUILD_ROOT/runtime" \
        -c conda-forge \
        libgl libglib
fi

echo "[2/8] Installing pinned PaddleOCR CPU dependencies..."
if ! "$BUILD_ROOT/runtime/bin/python" -m pip --version >/dev/null 2>&1; then
    rm -rf \
        "$BUILD_ROOT/runtime/lib/python3.12/site-packages/pip" \
        "$BUILD_ROOT/runtime/lib/python3.12/site-packages"/pip-*.dist-info
    rm -f "$BUILD_ROOT/runtime/bin/pip" "$BUILD_ROOT/runtime/bin/pip3" \
        "$BUILD_ROOT/runtime/bin/pip3.12"
    "$BUILD_ROOT/miniforge/bin/conda" install -y \
        -p "$BUILD_ROOT/runtime" \
        -c conda-forge \
        --force-reinstall \
        pip
fi
"$BUILD_ROOT/runtime/bin/python" -m pip install \
    paddlepaddle==3.3.0 \
    --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/
"$BUILD_ROOT/runtime/bin/python" -m pip install "paddleocr==3.7.0"

echo "[3/8] Downloading PP-OCRv6 recognition models..."
mkdir -p "$BUILD_ROOT/models"
if [ ! -f "$BUILD_ROOT/medium.tar" ]; then
    curl -fL --retry 3 "$MEDIUM_MODEL_URL" -o "$BUILD_ROOT/medium.tar"
fi
if [ ! -f "$BUILD_ROOT/small.tar" ]; then
    curl -fL --retry 3 "$SMALL_MODEL_URL" -o "$BUILD_ROOT/small.tar"
fi
if [ ! -f "$BUILD_ROOT/models/PP-OCRv6_medium_rec/inference.json" ]; then
    mkdir -p "$BUILD_ROOT/models/PP-OCRv6_medium_rec"
    tar -xf "$BUILD_ROOT/medium.tar" \
        -C "$BUILD_ROOT/models/PP-OCRv6_medium_rec" \
        --strip-components=1
fi
if [ ! -f "$BUILD_ROOT/models/PP-OCRv6_small_rec/inference.json" ]; then
    mkdir -p "$BUILD_ROOT/models/PP-OCRv6_small_rec"
    tar -xf "$BUILD_ROOT/small.tar" \
        -C "$BUILD_ROOT/models/PP-OCRv6_small_rec" \
        --strip-components=1
fi

echo "[4/8] Running local-model CPU smoke test..."
"$BUILD_ROOT/runtime/bin/python" - \
    "$BUILD_ROOT/models/PP-OCRv6_medium_rec" <<'PY'
import sys

import cv2
import numpy as np
import paddle
from paddleocr import TextRecognition

model_dir = sys.argv[1]
image = np.full((96, 220, 3), 255, dtype=np.uint8)
cv2.putText(
    image,
    "BJ",
    (20, 72),
    cv2.FONT_HERSHEY_SIMPLEX,
    2.2,
    (0, 0, 0),
    5,
    cv2.LINE_AA,
)
model = TextRecognition(
    model_name="PP-OCRv6_medium_rec",
    model_dir=model_dir,
    device="cpu",
    engine="paddle_static",
    enable_hpi=False,
    enable_mkldnn=True,
    cpu_threads=4,
)
results = list(model.predict(input=image, batch_size=1))
if not results:
    raise SystemExit("PaddleOCR smoke test returned no result")
print("paddle", paddle.__version__)
print("result", results[0].json)
PY

echo "[5/8] Packing relocatable Python runtime..."
mkdir -p "$BUILD_ROOT/packed"
if [ ! -f "$BUILD_ROOT/packed/mark_paddle_cpu_env.tar.gz" ]; then
    "$BUILD_ROOT/runtime/bin/conda-pack" \
        -p "$BUILD_ROOT/runtime" \
        -o "$BUILD_ROOT/packed/mark_paddle_cpu_env.tar.gz"
fi

echo "[6/8] Assembling offline bundle..."
rm -rf "$BUNDLE_DIR"
mkdir -p \
    "$BUNDLE_DIR/runtime" \
    "$BUNDLE_DIR/models" \
    "$BUNDLE_DIR/worker" \
    "$BUNDLE_DIR/scripts"
cp -a "$BUILD_ROOT/packed/mark_paddle_cpu_env.tar.gz" "$BUNDLE_DIR/runtime/"
cp -a "$BUILD_ROOT/models/PP-OCRv6_medium_rec" "$BUNDLE_DIR/models/"
cp -a "$BUILD_ROOT/models/PP-OCRv6_small_rec" "$BUNDLE_DIR/models/"
cp -a "$PROJECT_ROOT/mark_shadow/paddle_shadow_worker.py" "$BUNDLE_DIR/worker/"
cp -a "$PROJECT_ROOT/mark_shadow/export_shadow_report.py" "$BUNDLE_DIR/worker/"
cp -a "$PROJECT_ROOT/mark_shadow/install_offline.sh" "$BUNDLE_DIR/scripts/"
cp -a "$PROJECT_ROOT/mark_shadow/verify_offline.sh" "$BUNDLE_DIR/scripts/"
cp -a "$PROJECT_ROOT/mark_shadow/capi-mark-shadow.service.template" "$BUNDLE_DIR/scripts/"
cp -a "$PROJECT_ROOT/mark_shadow/install.sh" "$BUNDLE_DIR/"
cp -a "$PROJECT_ROOT/mark_shadow/README_OFFLINE.txt" "$BUNDLE_DIR/"
chmod +x "$BUNDLE_DIR/scripts/"*.sh "$BUNDLE_DIR/worker/"*.py
chmod +x "$BUNDLE_DIR/install.sh"

"$BUILD_ROOT/runtime/bin/python" - "$BUNDLE_DIR/manifest.json" \
    "$BUNDLE_VERSION" "$MEDIUM_MODEL_URL" "$SMALL_MODEL_URL" <<'PY'
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

output, bundle_version, medium_url, small_url = sys.argv[1:]
manifest = {
    "bundle_version": bundle_version,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "target": {
        "os": "RHEL 9 compatible",
        "architecture": "x86_64",
        "python": "3.12",
        "device": "cpu",
    },
    "packages": {
        "paddlepaddle": version("paddlepaddle"),
        "paddleocr": version("paddleocr"),
        "opencv-contrib-python": version("opencv-contrib-python"),
        "numpy": version("numpy"),
    },
    "models": {
        "PP-OCRv6_medium_rec": medium_url,
        "PP-OCRv6_small_rec": small_url,
    },
    "builder": {
        "platform": platform.platform(),
        "python": platform.python_version(),
    },
}
Path(output).write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

echo "[7/8] Generating checksums..."
(
    cd "$BUNDLE_DIR"
    find . -type f ! -name SHA256SUMS -print0 |
        sort -z |
        xargs -0 sha256sum > SHA256SUMS
)

echo "[8/8] Creating delivery archive..."
rm -f "$ARCHIVE_PATH"
tar -czf "$ARCHIVE_PATH" -C "$OUTPUT_DIR" "$BUNDLE_NAME"
(
    cd "$OUTPUT_DIR"
    sha256sum "$BUNDLE_NAME.tar.gz" > "$BUNDLE_NAME.tar.gz.sha256"
)

echo
echo "Offline bundle: $ARCHIVE_PATH"
echo "SHA256 file  : $ARCHIVE_PATH.sha256"
du -h "$ARCHIVE_PATH"
