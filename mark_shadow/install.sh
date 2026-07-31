#!/bin/bash
# One-command entry point for the offline MARK PaddleOCR installer.

set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$BUNDLE_ROOT/scripts/install_offline.sh" "$@"
