"""模型庫 CRUD：列表、啟用/停用、刪除、ZIP 匯出。"""
from __future__ import annotations
import io
import json
import logging
import shutil
import sqlite3
import zipfile
import yaml
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from capi_train_new import ZONES

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def invalidate_score_cache(db, scoring_bundle_id: int = None,
                            tile_ids: list = None,
                            lighting: str = None, zone: str = None) -> int:
    """tile_score_cache 失效統一入口。回傳刪除筆數。

    用例：
    - 重訓 submodel 完成 → invalidate(scoring_bundle_id=B, lighting=L, zone=Z)
    - 刪訓練資料 → invalidate(tile_ids=[...])
    - 刪 bundle → invalidate(scoring_bundle_id=B)
    """
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

    bundle["training_data"] = get_training_data_summary(db, bundle)
    bundle["pending_changes"] = get_pending_change_summary_for_bundle(db, bundle)
    return bundle


def _training_data_dir(job_id: str) -> Path:
    """job_id 對應的訓練圖片目錄（thumb / tiles / preview / ng 都在底下）。"""
    return Path(".tmp/train_new_thumbs") / job_id


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

    pool_ok = db.list_tile_pool(job_id, source="ok")
    pool_ng = db.list_tile_pool(job_id, source="ng")
    deleted_rows = len(pool_ok) + len(pool_ng)
    tile_ids_to_clean = [t["id"] for t in pool_ok] + [t["id"] for t in pool_ng]

    # 先清 score cache 再砍 tile_pool（順序很重要：cleanup 後 list_tile_pool 會回空）
    invalidate_score_cache(db, tile_ids=tile_ids_to_clean)
    db.cleanup_tile_pool(job_id)

    thumb_dir = _training_data_dir(job_id)
    freed, deleted_files = _dir_walk_stats(thumb_dir)
    shutil.rmtree(thumb_dir, ignore_errors=True)

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
    yaml_rel = str(Path(bundle["bundle_path"]) / "machine_config.yaml")

    # 全域單一 active：一次讀寫 server_config，移除所有其他 bundle 的 yaml。
    # configs/capi_3f.yaml 不在 DB 內，不會被踢掉，保留作為最後 fallback。
    other_yamls = {
        str(Path(o["bundle_path"]) / "machine_config.yaml")
        for o in db.list_model_bundles() if o["id"] != bundle_id
    }
    _rewrite_model_configs(
        server_config_path,
        keep=lambda p: p not in other_yamls,
        ensure_present=yaml_rel,
    )
    db.deactivate_all_bundles(except_id=bundle_id)
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


def _remove_from_model_configs(server_config_path: Path, yaml_rel: str) -> None:
    _rewrite_model_configs(server_config_path, keep=lambda p: p != yaml_rel)


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
    light_map = cfg.setdefault("threshold_mapping", {}).get(lighting)
    if not isinstance(light_map, dict) or zone not in light_map:
        raise ValueError(f"yaml 中找不到 threshold_mapping[{lighting}][{zone}]")
    light_map[zone] = rounded
    _dump_yaml(yaml_path, cfg)

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
    if not job_id:
        return 0
    bundle_dir = Path(bundle["bundle_path"])

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
