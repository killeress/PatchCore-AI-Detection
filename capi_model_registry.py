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


def activate_bundle(db, bundle_id: int, server_config_path: Path) -> dict:
    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        raise ValueError(f"bundle {bundle_id} not found")
    yaml_rel = str(Path(bundle["bundle_path"]) / "machine_config.yaml")

    # 同 machine 其他 bundle 從 server_config 移除 + 設 inactive
    for other in db.list_model_bundles(machine_id=bundle["machine_id"]):
        if other["id"] == bundle_id:
            continue
        other_yaml = str(Path(other["bundle_path"]) / "machine_config.yaml")
        _remove_from_model_configs(server_config_path, other_yaml)
        db.set_bundle_active(other["id"], False)

    # 加入此 bundle 的 yaml 到 server_config.model_configs
    _add_to_model_configs(server_config_path, yaml_rel)
    db.set_bundle_active(bundle_id, True)

    return {"ok": True, "message": "啟用成功，請重啟 server 才會生效"}


def deactivate_bundle(db, bundle_id: int, server_config_path: Path) -> dict:
    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        raise ValueError(f"bundle {bundle_id} not found")
    yaml_rel = str(Path(bundle["bundle_path"]) / "machine_config.yaml")
    _remove_from_model_configs(server_config_path, yaml_rel)
    db.set_bundle_active(bundle_id, False)
    return {"ok": True, "message": "已停用，請重啟 server 才會生效"}


def _add_to_model_configs(server_config_path: Path, yaml_rel: str) -> None:
    cfg = yaml.safe_load(server_config_path.read_text(encoding="utf-8")) or {}
    configs = cfg.setdefault("model_configs", [])
    if yaml_rel not in configs:
        configs.append(yaml_rel)
    server_config_path.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _remove_from_model_configs(server_config_path: Path, yaml_rel: str) -> None:
    cfg = yaml.safe_load(server_config_path.read_text(encoding="utf-8")) or {}
    configs = cfg.get("model_configs", [])
    cfg["model_configs"] = [p for p in configs if p != yaml_rel]
    server_config_path.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def delete_bundle(db, bundle_id: int, server_config_path: Path) -> dict:
    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        raise ValueError(f"bundle {bundle_id} not found")
    if bundle["is_active"]:
        raise ValueError("bundle is active; deactivate first")

    bundle_path = Path(bundle["bundle_path"])
    yaml_rel = str(bundle_path / "machine_config.yaml")
    _remove_from_model_configs(server_config_path, yaml_rel)

    if bundle_path.exists():
        shutil.rmtree(bundle_path, ignore_errors=False)
    db.delete_model_bundle(bundle_id)
    return {"ok": True}
