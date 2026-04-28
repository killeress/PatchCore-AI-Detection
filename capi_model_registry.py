"""模型庫 CRUD：列表、啟用/停用、刪除、ZIP 匯出。"""
from __future__ import annotations
import io
import json
import shutil
import zipfile
import yaml
from pathlib import Path
from typing import Dict, List, Optional


def list_bundles_grouped(db) -> Dict[str, List[dict]]:
    """所有 bundle 依 machine_id 分組。"""
    bundles = db.list_model_bundles()
    grouped: Dict[str, List[dict]] = {}
    for b in bundles:
        grouped.setdefault(b["machine_id"], []).append(b)
    return grouped


def get_bundle_detail(db, bundle_id: int) -> Optional[dict]:
    """讀 manifest.json + thresholds.json 並合併 DB row。"""
    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        return None
    bundle_path = Path(bundle["bundle_path"])
    manifest_p = bundle_path / "manifest.json"
    thresholds_p = bundle_path / "thresholds.json"

    bundle["manifest"] = json.loads(manifest_p.read_text(encoding="utf-8")) if manifest_p.exists() else None
    bundle["thresholds"] = json.loads(thresholds_p.read_text(encoding="utf-8")) if thresholds_p.exists() else None
    return bundle
