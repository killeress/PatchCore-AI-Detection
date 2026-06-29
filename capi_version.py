"""Application release version helpers."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent
VERSION_PATH = PROJECT_ROOT / "VERSION"
CHANGELOG_PATH = PROJECT_ROOT / "CHANGELOG.md"
RELEASE_MANIFEST_PATH = PROJECT_ROOT / "release_manifest.json"


def read_app_version() -> str:
    """Return the deployed release version."""
    try:
        version = VERSION_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        version = ""
    return version or "unknown"


def read_release_manifest() -> Dict[str, Any]:
    try:
        return json.loads(RELEASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_changelog() -> str:
    try:
        return CHANGELOG_PATH.read_text(encoding="utf-8")
    except OSError:
        return "# 更新紀錄\n\n目前找不到更新紀錄。"


def get_version_info() -> Dict[str, Any]:
    manifest = read_release_manifest()
    version = str(manifest.get("version") or read_app_version())
    return {
        "version": version,
        "git_commit": manifest.get("git_commit") or "",
        "built_at": manifest.get("built_at") or "",
        "artifact": manifest.get("artifact") or "",
        "manifest_available": bool(manifest),
        "reported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
