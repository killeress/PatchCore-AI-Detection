"""模型庫 CRUD：掃描/同步、列表、啟用/停用、刪除、ZIP 匯出。"""
from __future__ import annotations
import io
import json
import logging
import re
import shutil
import sqlite3
import zipfile
import yaml
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from capi_train_new import ZONES

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _line_body_and_ending(line: str) -> Tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


def _yaml_indent(body: str) -> int:
    return len(body) - len(body.lstrip(" "))


def _is_yaml_blank_or_comment(body: str) -> bool:
    stripped = body.strip()
    return not stripped or stripped.startswith("#")


def _replace_yaml_scalar_line(body: str, key: str, value: float) -> str:
    m = re.match(rf"^(?P<prefix>\s*{re.escape(key)}\s*:\s*)(?P<rest>.*)$", body)
    if not m:
        raise ValueError(f"yaml 中找不到 threshold_mapping 對應行: {key}")

    value_text = str(value)
    rest = m.group("rest")
    comment_idx = rest.find("#")
    if comment_idx >= 0:
        before_comment = rest[:comment_idx]
        trailing_spaces = re.search(r"\s*$", before_comment).group(0)
        if not trailing_spaces:
            trailing_spaces = " "
        rest = value_text + trailing_spaces + rest[comment_idx:]
    else:
        trailing_spaces = re.search(r"\s*$", rest).group(0)
        rest = value_text + trailing_spaces
    return m.group("prefix") + rest


def _update_threshold_mapping_text(text: str, lighting: str, zone: str, value: float) -> str:
    """只替換 threshold_mapping 內指定值，避免 yaml.dump 洗掉註解。"""
    lines = text.splitlines(keepends=True)
    threshold_idx = -1
    threshold_indent = 0

    for idx, line in enumerate(lines):
        body, _ = _line_body_and_ending(line)
        if re.match(r"^\s*threshold_mapping\s*:\s*(?:#.*)?$", body):
            threshold_idx = idx
            threshold_indent = _yaml_indent(body)
            break
    if threshold_idx < 0:
        raise ValueError("yaml 中找不到 threshold_mapping")

    lighting_idx = -1
    lighting_indent = 0
    lighting_pat = re.compile(rf"^\s*{re.escape(lighting)}\s*:\s*(?:#.*)?$")
    for idx in range(threshold_idx + 1, len(lines)):
        body, _ = _line_body_and_ending(lines[idx])
        if _is_yaml_blank_or_comment(body):
            continue
        indent = _yaml_indent(body)
        if indent <= threshold_indent:
            break
        if lighting_pat.match(body):
            lighting_idx = idx
            lighting_indent = indent
            break
    if lighting_idx < 0:
        raise ValueError(f"yaml 中找不到 threshold_mapping[{lighting}][{zone}]")

    zone_pat = re.compile(rf"^\s*{re.escape(zone)}\s*:")
    for idx in range(lighting_idx + 1, len(lines)):
        body, ending = _line_body_and_ending(lines[idx])
        if _is_yaml_blank_or_comment(body):
            continue
        indent = _yaml_indent(body)
        if indent <= lighting_indent:
            break
        if zone_pat.match(body):
            lines[idx] = _replace_yaml_scalar_line(body, zone, value) + ending
            return "".join(lines)

    raise ValueError(f"yaml 中找不到 threshold_mapping[{lighting}][{zone}]")


def invalidate_score_cache(db, scoring_bundle_id: int = None,
                            tile_ids: list = None,
                            lighting: str = None, zone: str = None) -> int:
    """tile_score_cache 失效統一入口。回傳刪除筆數。

    用例：
    - 重訓 submodel 完成 → invalidate(scoring_bundle_id=B, lighting=L, zone=Z)
    - 刪訓練資料 → invalidate(tile_ids=[...])
    - 刪 bundle → invalidate(scoring_bundle_id=B)

    為避免誤操作意外擴大刪除範圍，此 helper 拒絕模糊的參數組合：
    - lighting 與 zone 必須同時提供或同時省略
    - 指定 lighting/zone 時必須同時指定 scoring_bundle_id
    """
    if (lighting is None) != (zone is None):
        raise ValueError("lighting 與 zone 必須同時提供或同時省略")
    if lighting is not None and scoring_bundle_id is None:
        raise ValueError("指定 lighting+zone 時必須同時指定 scoring_bundle_id")
    return db.delete_score_cache(
        scoring_bundle_id=scoring_bundle_id,
        tile_ids=tile_ids,
        lighting=lighting, zone=zone,
    )


def list_bundles_grouped(db) -> Dict[str, List[dict]]:
    """所有 bundle 依 machine_id 分組。"""
    bundles = db.list_model_bundles()
    grouped: Dict[str, List[dict]] = {}
    for b in bundles:
        grouped.setdefault(b["machine_id"], []).append(b)
    return grouped


_BUNDLE_MARKER_FILES = ("manifest.json", "machine_config.yaml", "thresholds.json")


def _resolve_model_root(server_config_path: Path) -> Path:
    """Resolve the training output root used for folder-based bundle discovery."""
    server_config_path = Path(server_config_path).resolve()
    try:
        config = _load_yaml(server_config_path)
    except (OSError, ValueError, yaml.YAMLError):
        config = {}
    training = config.get("training") or {}
    raw_root = training.get("output_root", "model")
    root = Path(str(raw_root))
    return (root if root.is_absolute() else server_config_path.parent / root).resolve()


def _resolve_bundle_path(server_config_path: Path, raw_path: str) -> Path:
    path = Path(str(raw_path))
    return (path if path.is_absolute() else Path(server_config_path).resolve().parent / path).resolve()


def _path_is_under(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _iter_model_mapping_paths(model_mapping: Any):
    if not isinstance(model_mapping, dict):
        return
    for value in model_mapping.values():
        if isinstance(value, dict):
            for nested in value.values():
                if isinstance(nested, str) and nested.strip():
                    yield nested
        elif isinstance(value, str) and value.strip():
            yield value


def _inspect_bundle(bundle_path: Path, model_root: Path) -> dict:
    """Validate one directory and derive the model_registry metadata."""
    bundle_path = Path(bundle_path).resolve()
    model_root = Path(model_root).resolve()
    if not _path_is_under(model_root, bundle_path) or bundle_path == model_root:
        raise ValueError("bundle 路徑不在模型根目錄內")
    if not bundle_path.is_dir():
        raise ValueError("bundle 不是資料夾")

    missing = [name for name in _BUNDLE_MARKER_FILES if not (bundle_path / name).is_file()]
    if missing:
        raise ValueError(f"缺少必要檔案：{', '.join(missing)}")

    try:
        manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
        config = _load_yaml(bundle_path / "machine_config.yaml")
        json.loads((bundle_path / "thresholds.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"bundle metadata 無法解析：{exc}") from exc

    if not isinstance(manifest, dict) or not isinstance(config, dict):
        raise ValueError("manifest.json 或 machine_config.yaml 格式無效")

    machine_id = str(manifest.get("machine_id") or config.get("machine_id") or "").strip()
    if not machine_id:
        raise ValueError("找不到 machine_id")
    config_machine_id = str(config.get("machine_id") or "").strip()
    if config_machine_id and config_machine_id != machine_id:
        raise ValueError(
            f"machine_id 不一致：manifest={machine_id}，yaml={config_machine_id}"
        )

    mapping_paths = list(_iter_model_mapping_paths(config.get("model_mapping")))
    if not mapping_paths:
        raise ValueError("machine_config.yaml 沒有 model_mapping")

    path_mismatches = []
    missing_models = []
    for raw_path in mapping_paths:
        model_name = Path(raw_path).name
        local_path = bundle_path / model_name
        if not local_path.is_file():
            missing_models.append(model_name)
            continue
        expected = str(local_path.resolve())
        if str(Path(raw_path).resolve()) != expected:
            path_mismatches.append({"source": raw_path, "target": expected})

    if missing_models:
        raise ValueError(f"model_mapping 指向的模型不存在：{', '.join(sorted(set(missing_models)))}")

    model_files = list(bundle_path.glob("*.pt"))
    if not model_files:
        raise ValueError("找不到 .pt 模型檔")

    tiles_per_unit = manifest.get("tiles_per_unit") or {}
    if not isinstance(tiles_per_unit, dict):
        tiles_per_unit = {}

    def _unit_total(field: str, suffix: str = "") -> int:
        return sum(
            int(value.get(field, 0) or 0)
            for key, value in tiles_per_unit.items()
            if isinstance(value, dict) and (not suffix or str(key).endswith(suffix))
        )

    trained_at = str(manifest.get("trained_at") or config.get("trained_at") or "").strip()
    if not trained_at:
        raise ValueError("找不到 trained_at")

    return {
        "path": str(bundle_path),
        "name": bundle_path.name,
        "machine_id": machine_id,
        "trained_at": trained_at,
        "panel_count": int(manifest.get("panel_count", 0) or 0),
        "inner_tile_count": _unit_total("train", "-inner"),
        "edge_tile_count": _unit_total("train", "-edge"),
        "ng_tile_count": _unit_total("ng"),
        "bundle_size_bytes": sum(path.stat().st_size for path in model_files),
        "source_job_id": str(manifest.get("trained_with_job_id") or "").strip(),
        "path_mismatches": path_mismatches,
    }


def discover_model_bundles(db, server_config_path: Path) -> dict:
    """Find valid, not-yet-registered bundle folders under training.output_root."""
    model_root = _resolve_model_root(server_config_path)
    registered_paths = {
        _resolve_bundle_path(server_config_path, bundle["bundle_path"])
        for bundle in db.list_model_bundles()
    }
    discovered = []
    invalid = []

    if not model_root.is_dir():
        return {
            "model_root": str(model_root),
            "bundles": discovered,
            "invalid": invalid,
        }

    for bundle_path in sorted(model_root.iterdir(), key=lambda path: path.name.lower()):
        if not bundle_path.is_dir():
            continue
        if not any((bundle_path / marker).exists() for marker in _BUNDLE_MARKER_FILES):
            continue
        try:
            info = _inspect_bundle(bundle_path, model_root)
        except ValueError as exc:
            invalid.append({"path": str(bundle_path.resolve()), "error": str(exc)})
            continue
        if bundle_path.resolve() in registered_paths:
            continue
        discovered.append(info)

    return {
        "model_root": str(model_root),
        "bundles": discovered,
        "invalid": invalid,
    }


def _rewrite_bundle_model_paths(bundle: dict) -> None:
    """Repair copied absolute model paths after the bundle was validated."""
    mismatches = bundle.get("path_mismatches") or []
    if not mismatches:
        return
    yaml_path = Path(bundle["path"]) / "machine_config.yaml"
    text = yaml_path.read_text(encoding="utf-8")
    for mismatch in mismatches:
        text = text.replace(mismatch["source"], mismatch["target"])
    yaml_path.write_text(text, encoding="utf-8")


def sync_discovered_bundles(
    db,
    server_config_path: Path,
    bundle_paths: Optional[List[str]] = None,
) -> dict:
    """Register discovered bundles without activating or changing model_configs."""
    discovery = discover_model_bundles(db, server_config_path)
    selected = None
    if bundle_paths is not None:
        model_root = Path(discovery["model_root"]).resolve()
        selected = {
            _resolve_bundle_path(server_config_path, raw_path)
            for raw_path in bundle_paths
            if _path_is_under(model_root, _resolve_bundle_path(server_config_path, raw_path))
        }

    imported = []
    skipped = []
    for bundle in discovery["bundles"]:
        if selected is not None and Path(bundle["path"]).resolve() not in selected:
            continue
        _rewrite_bundle_model_paths(bundle)
        source_job_id = bundle.get("source_job_id") or ""
        linked_job_id = None
        if source_job_id:
            try:
                linked_job_id = source_job_id if db.get_training_job(source_job_id) else None
            except Exception:
                linked_job_id = None
        notes = "資料夾掃描匯入"
        if source_job_id and not linked_job_id:
            notes += f"；原始 job {source_job_id} 的訓練資料未同步"
        try:
            bundle_id = db.register_model_bundle({
                "machine_id": bundle["machine_id"],
                "bundle_path": bundle["path"],
                "trained_at": bundle["trained_at"],
                "panel_count": bundle["panel_count"],
                "inner_tile_count": bundle["inner_tile_count"],
                "edge_tile_count": bundle["edge_tile_count"],
                "ng_tile_count": bundle["ng_tile_count"],
                "bundle_size_bytes": bundle["bundle_size_bytes"],
                "job_id": linked_job_id,
                "notes": notes,
            })
        except sqlite3.IntegrityError:
            skipped.append({"path": bundle["path"], "reason": "已存在或同步競速"})
            continue
        imported.append({
            "id": bundle_id,
            "machine_id": bundle["machine_id"],
            "path": bundle["path"],
            "path_rewritten": bool(bundle.get("path_mismatches")),
        })

    return {
        "model_root": discovery["model_root"],
        "imported": imported,
        "skipped": skipped,
        "invalid": discovery["invalid"],
    }


def get_bundle_detail(db, bundle_id: int) -> Optional[dict]:
    """讀 manifest.json + machine_config.yaml + thresholds.json 並合併 DB row。

    threshold 顯示優先順序：machine_config.yaml > thresholds.json > manifest 快照。
    inference 引擎只讀 yaml，所以 yaml 就是 source of truth；thresholds.json 退化成
    向後相容 fallback（手動改 yaml 沒同步 thresholds.json 時，UI 仍能顯示真值）。
    """
    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        return None
    bundle_path = Path(bundle["bundle_path"])
    manifest_p = bundle_path / "manifest.json"
    thresholds_p = bundle_path / "thresholds.json"
    yaml_p = bundle_path / "machine_config.yaml"

    bundle["manifest"] = json.loads(manifest_p.read_text(encoding="utf-8")) if manifest_p.exists() else None
    thresholds_json = json.loads(thresholds_p.read_text(encoding="utf-8")) if thresholds_p.exists() else {}
    bundle["thresholds_json"] = thresholds_json or None  # 留給 debug / 對比用
    bundle["thresholds_source"] = "thresholds.json"

    # yaml 優先：解析 threshold_mapping 並覆蓋顯示值
    if yaml_p.exists():
        try:
            cfg = _load_yaml(yaml_p)
            yaml_thr = cfg.get("threshold_mapping") or {}
            if isinstance(yaml_thr, dict) and yaml_thr:
                # 正規化成 {lighting: {inner: float, edge: float}} 格式（與 thresholds.json 同型）
                normalized = {}
                for lighting, val in yaml_thr.items():
                    if isinstance(val, dict):
                        normalized[lighting] = {
                            zone: float(zv) for zone, zv in val.items()
                        }
                    else:
                        # legacy flat 格式：把單值同時當 inner / edge 顯示
                        normalized[lighting] = {"inner": float(val), "edge": float(val)}
                bundle["thresholds"] = normalized
                bundle["thresholds_source"] = "machine_config.yaml"
            else:
                bundle["thresholds"] = thresholds_json or None
        except Exception as e:
            logger.warning("get_bundle_detail: 讀 machine_config.yaml 失敗，fallback 到 thresholds.json: %s", e)
            bundle["thresholds"] = thresholds_json or None
    else:
        bundle["thresholds"] = thresholds_json or None

    bundle["training_panel_modes"] = None
    job_id = bundle.get("job_id") or ""
    if job_id:
        try:
            job = db.get_training_job(job_id)
        except sqlite3.Error as e:
            logger.warning("get_bundle_detail: 讀訓練 PANEL 設定失敗 (%s): %s", job_id, e)
            job = None
        if isinstance(job, dict):
            panel_paths = job.get("panel_paths") or []
            panel_modes = job.get("panel_modes") or ["full"] * len(panel_paths)
            manifest_names = (bundle.get("manifest") or {}).get("panel_glass_ids") or []
            bundle["training_panel_modes"] = [
                {
                    "panel_name": (
                        manifest_names[index]
                        if index < len(manifest_names)
                        else Path(str(panel_path)).name
                    ),
                    "mode": panel_modes[index] if index < len(panel_modes) else "full",
                }
                for index, panel_path in enumerate(panel_paths)
            ]

    bundle["training_data"] = get_training_data_summary(db, bundle)
    bundle["pending_changes"] = get_pending_change_summary_for_bundle(db, bundle)
    return bundle


def _training_data_dir(job_id: str) -> Path:
    """job_id 對應的訓練圖片目錄（thumb / tiles / preview / ng 都在底下）。"""
    return Path(".tmp/train_new_thumbs") / job_id


def _training_staging_dir(job_id: str) -> Path:
    return Path(".tmp/training_staging") / job_id


def _training_runs_dir(job_id: str) -> Path:
    return Path(".tmp/training_runs") / job_id


def _safe_training_job_path(root: Path, job_id: str) -> Optional[Path]:
    """Return a job directory only when job_id cannot escape the temp root."""
    value = str(job_id or "").strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        logger.warning("skip training temp cleanup for unsafe job_id=%r", job_id)
        return None

    base = root.resolve()
    target = (base / value).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        logger.warning("skip training temp cleanup outside root: %s", target)
        return None
    return target


def _remove_training_temp_dir(path: Optional[Path]) -> Tuple[int, int]:
    if path is None or not path.exists():
        return 0, 0
    if path.is_symlink():
        logger.warning("skip training temp symlink: %s", path)
        return 0, 0

    freed, file_count = _dir_walk_stats(path)
    try:
        shutil.rmtree(path)
    except OSError as exc:
        logger.warning("training temp cleanup failed for %s: %s", path, exc)
        return 0, 0
    return freed, file_count


def cleanup_training_job_artifacts(
    db,
    job_id: str,
    *,
    remove_training_data: bool = True,
    remove_ephemeral: bool = True,
) -> dict:
    """清理訓練 job 暫存資料。

    `remove_training_data=False` 只清理每次訓練必須的 staging/run 輸出，
    保留 review/completed job 的 tile pool 與 thumbnails。
    """
    value = str(job_id or "").strip()
    staging_path = _training_staging_dir(value)
    runs_path = _training_runs_dir(value)
    thumbs_path = _training_data_dir(value)
    roots = {
        "staging": _safe_training_job_path(staging_path.parent, value),
        "runs": _safe_training_job_path(runs_path.parent, value),
        "thumbs": _safe_training_job_path(thumbs_path.parent, value),
    }
    if any(path is None for path in roots.values()):
        return {
            "ok": False,
            "job_id": value,
            "deleted_tile_rows": 0,
            "deleted_files": 0,
            "freed_bytes": 0,
        }

    deleted_tile_rows = 0
    if remove_training_data and db is not None:
        pool_rows = db.list_tile_pool(value)
        tile_ids = [row["id"] for row in pool_rows if row.get("id") is not None]
        if tile_ids:
            invalidate_score_cache(db, tile_ids=tile_ids)
        deleted_tile_rows = len(pool_rows)
        db.cleanup_tile_pool(value)

    deleted_files = 0
    freed_bytes = 0
    if remove_ephemeral:
        for key in ("staging", "runs"):
            freed, count = _remove_training_temp_dir(roots[key])
            freed_bytes += freed
            deleted_files += count
    if remove_training_data:
        freed, count = _remove_training_temp_dir(roots["thumbs"])
        freed_bytes += freed
        deleted_files += count

    return {
        "ok": True,
        "job_id": value,
        "deleted_tile_rows": deleted_tile_rows,
        "deleted_files": deleted_files,
        "freed_bytes": freed_bytes,
    }


def _dir_walk_stats(path: Path) -> Tuple[int, int]:
    """遞迴累加目錄大小與檔案數，目錄不存在回 (0, 0)。"""
    if not path.exists():
        return 0, 0
    total_size = 0
    file_count = 0
    for p in path.rglob("*"):
        if p.is_file():
            file_count += 1
            try:
                total_size += p.stat().st_size
            except OSError:
                pass
    return total_size, file_count


def _dir_size_bytes(path: Path) -> int:
    """遞迴累加目錄大小，目錄不存在回 0。"""
    return _dir_walk_stats(path)[0]


def get_training_data_summary(db, bundle: dict) -> dict:
    """回傳訓練資料概況：DB tile 數量 + 磁碟大小。

    `exists` 由 caller 自行判斷（任何欄位 > 0 即代表有資料）。
    """
    job_id = bundle.get("job_id") or ""
    summary = {
        "job_id": job_id,
        "ok_count": 0,
        "ng_count": 0,
        "size_bytes": 0,
        "thumb_dir": "",
    }
    if not job_id:
        return summary

    try:
        summary["ok_count"] = len(db.list_tile_pool(job_id, source="ok"))
        summary["ng_count"] = len(db.list_tile_pool(job_id, source="ng"))
    except sqlite3.Error as e:
        logger.warning("get_training_data_summary: DB query failed for %s: %s", job_id, e)

    thumb_dir = _training_data_dir(job_id)
    summary["thumb_dir"] = str(thumb_dir)
    summary["size_bytes"] = _dir_size_bytes(thumb_dir)
    return summary


def delete_training_data(db, bundle_id: int) -> dict:
    """清空指定 bundle 對應 job 的訓練資料：DB tile_pool + thumbnails 目錄。

    bundle 本身（model_registry row、bundle_path 內容）不動，inference 不受影響。
    """
    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        raise ValueError(f"bundle {bundle_id} not found")
    job_id = bundle.get("job_id") or ""
    if not job_id:
        return {"ok": True, "message": "此 bundle 沒有關聯 job_id，無訓練資料可清",
                "deleted_files": 0, "freed_bytes": 0, "deleted_tile_rows": 0}

    cleanup = cleanup_training_job_artifacts(
        db,
        job_id,
        remove_training_data=True,
        remove_ephemeral=False,
    )
    deleted_rows = cleanup["deleted_tile_rows"]
    deleted_files = cleanup["deleted_files"]
    freed = cleanup["freed_bytes"]

    return {
        "ok": True,
        "message": f"已清除 {deleted_rows} 筆 DB 紀錄、{deleted_files} 個檔案，"
                   f"釋放 {freed/1e6:.1f} MB",
        "deleted_tile_rows": deleted_rows,
        "deleted_files": deleted_files,
        "freed_bytes": freed,
    }


def activate_bundle(db, bundle_id: int, server_config_path: Path) -> dict:
    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        raise ValueError(f"bundle {bundle_id} not found")
    target_yaml = _resolve_model_config_path(
        server_config_path,
        Path(bundle["bundle_path"]) / "machine_config.yaml",
    )

    # 全域單一 active：一次讀寫 server_config，移除所有其他 bundle 的 yaml。
    # configs/capi_3f.yaml 不在 DB 內，不會被踢掉，保留作為最後 fallback。
    other_yamls = {
        _resolve_model_config_path(
            server_config_path,
            Path(o["bundle_path"]) / "machine_config.yaml",
        )
        for o in db.list_model_bundles() if o["id"] != bundle_id
    }
    _rewrite_model_configs(
        server_config_path,
        keep=lambda p: _resolve_model_config_path(server_config_path, p) not in other_yamls,
        ensure_present=str(target_yaml),
    )
    db.deactivate_all_bundles(except_id=bundle_id)
    db.set_bundle_active(bundle_id, True)

    return {"ok": True, "message": "啟用成功，請重啟 server 才會生效"}


def deactivate_bundle(db, bundle_id: int, server_config_path: Path) -> dict:
    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        raise ValueError(f"bundle {bundle_id} not found")
    yaml_path = str(Path(bundle["bundle_path"]) / "machine_config.yaml")
    _ensure_model_configs_remains_non_empty(server_config_path, removing={yaml_path})
    _remove_from_model_configs(server_config_path, yaml_path)
    db.set_bundle_active(bundle_id, False)
    return {"ok": True, "message": "已停用，請重啟 server 才會生效"}


def update_bundle_notes(db, bundle_id: int, notes: str) -> dict:
    """更新 bundle 的使用者備註。"""
    if not isinstance(notes, str):
        raise ValueError("notes 必須是文字")
    if not db.update_model_bundle_notes(bundle_id, notes):
        raise ValueError(f"bundle {bundle_id} not found")
    return {"ok": True, "notes": notes, "message": "備註已儲存"}


def _rewrite_model_configs(
    server_config_path: Path,
    keep: Callable[[str], bool],
    ensure_present: Optional[str] = None,
) -> None:
    """單次讀寫 server_config.yaml 的 model_configs。

    keep(p) → True 保留；ensure_present 為非 None 時保證該 path 在清單中。
    """
    cfg = _load_yaml(server_config_path)
    configs = [p for p in cfg.get("model_configs", []) if keep(p)]
    if ensure_present is not None and ensure_present not in configs:
        configs.append(ensure_present)
    cfg["model_configs"] = configs
    _dump_yaml(server_config_path, cfg)


def _resolve_model_config_path(server_config_path: Path, config_path) -> Path:
    server_config_path = Path(server_config_path).resolve()
    path = Path(str(config_path))
    if not path.is_absolute():
        path = server_config_path.parent / path
    return path.resolve()


def _remove_from_model_configs(server_config_path: Path, yaml_path: str) -> None:
    resolved = _resolve_model_config_path(server_config_path, yaml_path)
    _rewrite_model_configs(
        server_config_path,
        keep=lambda p: _resolve_model_config_path(server_config_path, p) != resolved,
    )


def _ensure_model_configs_remains_non_empty(
    server_config_path: Path, removing: set
) -> None:
    """擋住會把 model_configs 清空的 deactivate / delete。

    server 啟動時 model_configs=[] 會直接 raise（capi_server.py:836），
    在這裡先擋；若呼叫端要移除的 yaml 被移走後仍至少剩一個 entry 就放行。
    """
    cfg = _load_yaml(server_config_path)
    resolved_removing = {
        _resolve_model_config_path(server_config_path, path)
        for path in removing
    }
    remaining = [
        p for p in cfg.get("model_configs", [])
        if _resolve_model_config_path(server_config_path, p) not in resolved_removing
    ]
    if not remaining:
        raise ValueError(
            "不能停用 / 刪除最後一個 active bundle —— "
            "server 重啟後將找不到任何模型可載入。"
            "請先啟用其他 bundle 再執行此操作。"
        )


def delete_bundle(db, bundle_id: int, server_config_path: Path) -> dict:
    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        raise ValueError(f"bundle {bundle_id} not found")
    if bundle["is_active"]:
        raise ValueError("bundle is active; deactivate first")

    server_config_path = Path(server_config_path).resolve()
    raw_bundle_path = Path(bundle["bundle_path"])
    bundle_path = raw_bundle_path if raw_bundle_path.is_absolute() else server_config_path.parent / raw_bundle_path
    bundle_path = bundle_path.resolve()
    model_root = (server_config_path.parent / "model").resolve()
    try:
        bundle_path.relative_to(model_root)
    except ValueError:
        raise ValueError(f"bundle path is outside model root: {bundle_path}")
    if bundle_path == model_root:
        raise ValueError("refusing to delete model root")
    if bundle_path.exists() and not (bundle_path / "machine_config.yaml").is_file():
        raise ValueError(f"bundle marker missing: {bundle_path / 'machine_config.yaml'}")

    raw_yaml = str(raw_bundle_path / "machine_config.yaml")
    resolved_yaml = str(bundle_path / "machine_config.yaml")
    _ensure_model_configs_remains_non_empty(
        server_config_path, removing={raw_yaml, resolved_yaml}
    )
    _remove_from_model_configs(server_config_path, raw_yaml)
    if resolved_yaml != raw_yaml:
        _remove_from_model_configs(server_config_path, resolved_yaml)

    if bundle_path.exists():
        shutil.rmtree(bundle_path, ignore_errors=False)
    # 該 bundle 作為 scoring bundle 算過的所有分都失效
    invalidate_score_cache(db, scoring_bundle_id=bundle_id)
    db.delete_model_bundle(bundle_id)
    return {"ok": True}


def update_threshold(db, bundle_id: int, lighting: str, zone: str, value: float) -> dict:
    """改 bundle 的 machine_config.yaml + thresholds.json 內某個 (lighting, zone) 的 threshold。

    回傳 dict 包含 machine_id，呼叫端用以觸發 server in-place reload。
    thresholds.json 必須同步更新——它是模型庫細節 modal 顯示的來源
    （capi_model_registry.get_bundle_detail / capi_web step5_done 都會讀）。
    """
    if zone not in ZONES:
        raise ValueError(f"zone 必須是 {ZONES}，收到 {zone!r}")
    if not (0.0 <= value <= 10.0):
        raise ValueError(f"threshold 範圍應在 0.0~10.0，收到 {value}")

    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        raise ValueError(f"bundle {bundle_id} not found")

    bundle_dir = Path(bundle["bundle_path"])
    yaml_path = bundle_dir / "machine_config.yaml"
    thr_path = bundle_dir / "thresholds.json"
    rounded = round(float(value), 4)

    try:
        cfg = _load_yaml(yaml_path)
    except FileNotFoundError:
        raise ValueError(f"machine_config.yaml 不存在: {yaml_path}")
    light_map = (cfg.get("threshold_mapping") or {}).get(lighting)
    if not isinstance(light_map, dict) or zone not in light_map:
        raise ValueError(f"yaml 中找不到 threshold_mapping[{lighting}][{zone}]")
    yaml_text = yaml_path.read_text(encoding="utf-8")
    yaml_path.write_text(
        _update_threshold_mapping_text(yaml_text, lighting, zone, rounded),
        encoding="utf-8",
    )

    try:
        thresholds = json.loads(thr_path.read_text(encoding="utf-8"))
        thresholds.setdefault(lighting, {})[zone] = rounded
        thr_path.write_text(json.dumps(thresholds, indent=2, ensure_ascii=False), encoding="utf-8")
    except FileNotFoundError:
        pass
    except json.JSONDecodeError as e:
        logger.warning("update_threshold: thresholds.json 解析失敗，跳過同步: %s", e)

    return {
        "ok": True,
        "machine_id": bundle["machine_id"],
        "lighting": lighting, "zone": zone, "value": rounded,
        "message": "已更新 threshold，請重啟 server 才會生效",
    }


def _read_manifest(bundle_dir: Path) -> dict:
    """讀 bundle 的 manifest.json；不存在或解析錯誤回空 dict。"""
    p = bundle_dir / "manifest.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("manifest.json 解析失敗：%s", p)
        return {}


def _write_manifest(bundle_dir: Path, data: dict) -> None:
    p = bundle_dir / "manifest.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    import os
    os.replace(tmp, p)


def append_submodel_history(
    bundle_dir: Path, lighting: str, zone: str, entry: dict,
) -> None:
    """把單次訓練 entry 追加到 manifest.submodel_history[lighting-zone]。

    若 manifest 不存在 submodel_history 欄位則新建。entry 預期至少包含：
    trained_at、tile_count_used、auroc、used_tile_ids、kind。
    """
    unit_label = f"{lighting}-{zone}"
    manifest = _read_manifest(bundle_dir)
    history = manifest.setdefault("submodel_history", {})
    history.setdefault(unit_label, []).append(entry)
    manifest["last_retrained_at"] = entry.get("trained_at", manifest.get("last_retrained_at"))
    _write_manifest(bundle_dir, manifest)


def get_used_tile_ids(bundle_dir: Path, lighting: str, zone: str) -> Optional[set]:
    """讀 manifest 取得該 unit「上次訓練時使用的 tile_pool.id 集合」。

    優先順序：
    1. submodel_history[unit_label] 最新 entry 的 used_tile_ids
    2. 退回 manifest.unit_metrics[unit_label].used_tile_ids（初次訓練的記錄）
    3. 都沒有 → None（表示舊 bundle，無法判斷差異）
    """
    unit_label = f"{lighting}-{zone}"
    manifest = _read_manifest(bundle_dir)

    history = (manifest.get("submodel_history") or {}).get(unit_label) or []
    if history:
        ids = history[-1].get("used_tile_ids")
        if ids is not None:
            return set(int(x) for x in ids)

    unit_metrics = (manifest.get("unit_metrics") or {}).get(unit_label) or {}
    ids = unit_metrics.get("used_tile_ids")
    if ids is not None:
        return set(int(x) for x in ids)
    return None


def get_pending_change_count(
    db, bundle: dict, lighting: str, zone: str,
) -> int:
    """回傳該 unit 目前「decision=accept」tile id 集合與上次訓練集合的差異數。

    差異 = (新增 accept 的) ∪ (上次訓練用過但現在被 reject 的)。
    舊 bundle（manifest 沒記錄 used_tile_ids）退化策略：回傳目前 reject 的 tile 數。
    無 job_id（訓練資料已刪）回 0。
    """
    job_id = bundle.get("job_id") or ""
    bundle_dir = Path(bundle["bundle_path"])
    manifest = _read_manifest(bundle_dir)
    unit_label = f"{lighting}-{zone}"
    history = (manifest.get("submodel_history") or {}).get(unit_label) or []
    if history:
        history_job_id = history[-1].get("job_id") or history[-1].get("trained_with_job_id")
        if history_job_id:
            job_id = history_job_id

    if not job_id:
        return 0

    current_accept = {
        int(t["id"]) for t in db.list_tile_pool(
            job_id, lighting=lighting, zone=zone, source="ok", decision="accept",
        )
    }
    last_used = get_used_tile_ids(bundle_dir, lighting, zone)

    if last_used is None:
        # 舊 bundle 退化路徑：用「現在被 reject 的數量」當差異訊號
        rejected = db.list_tile_pool(
            job_id, lighting=lighting, zone=zone, source="ok", decision="reject",
        )
        return len(rejected)

    return len(current_accept.symmetric_difference(last_used))


def get_pending_change_summary_for_bundle(db, bundle: dict) -> dict:
    """所有 lighting+zone 的 pending change 數量，給列表頁徽章用。

    回傳 {(lighting, zone): count, ...}，過濾掉 count == 0 的。
    """
    out = {}
    for lighting in ("G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD"):
        for zone in ("inner", "edge"):
            n = get_pending_change_count(db, bundle, lighting, zone)
            if n > 0:
                out[(lighting, zone)] = n
    return out


def export_bundle_zip(bundle_path: Path, machine_id: str) -> bytes:
    """打包成 ZIP（內含 README）。回 bytes 給 streaming response。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 整個 bundle 目錄
        for p in bundle_path.rglob("*"):
            if p.is_file():
                arcname = Path("model") / bundle_path.name / p.relative_to(bundle_path)
                zf.write(p, str(arcname))
        # README
        readme = _build_readme(machine_id, bundle_path)
        zf.writestr(str(Path("model") / bundle_path.name / "README.txt"), readme)
    return buf.getvalue()


def _build_readme(machine_id: str, bundle_path: Path) -> str:
    return f"""新機種 PatchCore Bundle 部署說明
────────────────────────────────────────
機種：{machine_id}
Bundle：{bundle_path.name}

部署步驟：
1. 解壓本 ZIP，保留路徑結構
2. FTP 上傳整個 bundle 目錄到 production：
     model/{bundle_path.name}/  → /capi_ai/model/{bundle_path.name}/
3. 編輯 production 的 server_config.yaml，在 model_configs 列表加入：
     - model/{bundle_path.name}/machine_config.yaml
4. （可選）若同機種有舊 bundle 想停用，從 model_configs 移除舊 bundle 的 yaml
5. 重啟 capi_server 服務

驗證：傳送該機種 panel 給 inference，confirm 走新架構（log 顯示 "load 10 models"）。
"""
