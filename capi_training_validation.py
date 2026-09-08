"""Independent, panel-grouped calibration and acceptance for the training wizard.

All scores are exported-model tile scores, before production post-filters.
Acceptance samples never participate in threshold selection or model fitting.
"""
from bisect import bisect_left
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil


ROLES = ("train", "calibration", "acceptance")


def path_key(value):
    return str(Path(value).resolve()).casefold()


def assign_batch_roles(panels):
    """Stable batch split; keep repeated physical panels in one component."""
    parents = {item["group"].casefold(): item["group"].casefold() for item in panels.values()}
    def root(group):
        while parents[group] != group:
            group = parents[group]
        return group
    physical = {}
    for path, item in panels.items():
        group = item["group"].casefold()
        name = Path(path).name.casefold()
        if name in physical:
            a, b = root(group), root(physical[name])
            parents[max(a, b)] = min(a, b)
        physical[name] = group
    groups = sorted({root(g) for g in parents}, key=lambda g: hashlib.sha256(("validation-v1:" + g).encode()).hexdigest())
    roles = {g: "train" for g in groups}
    if len(groups) >= 3:
        held_out = max(1, len(groups) // 5)
        roles.update({g: "calibration" for g in groups[:held_out]})
        roles.update({g: "acceptance" for g in groups[held_out:held_out * 2]})
    for item in panels.values():
        item["role"] = roles[root(item["group"].casefold())]
    return len(groups)


def normalize_validation_config(raw, panel_paths=None):
    if raw is None or raw == {}:
        return {}
    if not isinstance(raw, dict) or set(raw) - {"panels", "max_false_positive_rate", "max_miss_rate", "split_mode"}:
        raise ValueError("validation_config 格式錯誤")
    auto = raw.get("split_mode") == "auto_batch"
    if "split_mode" in raw and not auto:
        raise ValueError("不支援的資料分組方式")
    panels = raw.get("panels")
    if not isinstance(panels, dict) or not panels:
        raise ValueError("請指定訓練、校正、驗收 panel 與批次")
    clean, groups, physical_panels = {}, {}, {}
    for path, item in panels.items():
        if not isinstance(path, str) or not path.strip() or not isinstance(item, dict):
            raise ValueError("validation_config.panels 格式錯誤")
        role = "train" if auto else item.get("role")
        group = item.get("group")
        if role not in ROLES or not isinstance(group, str) or not group.strip():
            raise ValueError("請確認每片圖片的批次名稱" if auto else "每片 panel 必須指定用途與批次 ID")
        group = group.strip()
        key = path_key(path)
        physical = Path(path).name.casefold()
        if key in clean:
            raise ValueError("同一 panel 重複選取")
        for mapping, identity in ((groups, group.casefold()), (physical_panels, physical)):
            if identity in mapping and mapping[identity] != role:
                raise ValueError("同一批次或同一 panel 不可跨訓練／校正／驗收用途")
            mapping[identity] = role
        clean[key] = {"role": role, "group": group}
    if auto:
        assign_batch_roles(clean)
    if not auto and {p["role"] for p in clean.values()} != set(ROLES):
        raise ValueError("獨立驗收需要三組不同批次：訓練、校正、驗收")
    if panel_paths is not None and set(clean) != {path_key(p) for p in panel_paths}:
        raise ValueError("驗收用途設定必須與選取的 panel 完全一致")
    result = {"panels": clean}
    if auto:
        result["split_mode"] = "auto_batch"
    for key in ("max_false_positive_rate", "max_miss_rate"):
        value = raw.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} 必須介於 0 與 1")
            result[key] = float(value)
    return result


def training_tiles(tiles, require_confirmed=False):
    return [t for t in tiles if t.get("dataset_role", "train") == "train"
            and (t.get("validation_label") == "ok" if require_confirmed else t.get("validation_label") != "ng")]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def recipe_fingerprint(bundle_dir):
    import yaml
    recipe = yaml.safe_load((Path(bundle_dir) / "machine_config.yaml").read_text(encoding="utf-8"))
    for key in ("threshold_mapping", "trained_at", "bundle_path"):
        recipe.pop(key, None)
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def seal_reports(bundle_dir, unit_metrics):
    fingerprint = recipe_fingerprint(bundle_dir)
    for metrics in unit_metrics.values():
        relative = (metrics.get("validation") or {}).get("report_path")
        if relative:
            path = Path(bundle_dir) / relative
            report = json.loads(path.read_text(encoding="utf-8"))
            report["recipe_sha256"] = fingerprint
            write_json(path, report)


def load_report(bundle_dir, unit_label):
    # Unit comes from manifest/model mapping, never an arbitrary relative path.
    if not unit_label or any(c in unit_label for c in "/\\."):
        raise ValueError("invalid unit")
    bundle_dir = Path(bundle_dir)
    report = json.loads((bundle_dir / "validation_reports" / unit_label / "report.json").read_text(encoding="utf-8"))
    model = bundle_dir / Path(report["model_file"]).name
    report["stale"] = (sha256_file(model) != report["model_sha256"] or
                       recipe_fingerprint(bundle_dir) != report.get("recipe_sha256"))
    return report


def adopt_threshold(db, bundle_id, unit_label):
    from capi_model_registry import update_threshold
    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        raise ValueError("找不到模型")
    if bundle.get("is_active"):
        raise ValueError("此模型已啟用；請先停用，再採用建議門檻")
    root = Path(bundle["bundle_path"])
    report = load_report(root, unit_label)
    if report["stale"]:
        raise ValueError("模型或前處理設定已變更，此份報告僅供歷史參考")
    if report["suggested_threshold"] is None or report["status"] == "insufficient":
        raise ValueError("資料不足，沒有可採用的建議門檻")
    lighting, zone = unit_label.rsplit("-", 1)
    audit_path = root / "validation_reports" / unit_label / "adoptions.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else []
    import yaml
    config = yaml.safe_load((root / "machine_config.yaml").read_text(encoding="utf-8"))
    previous = config["threshold_mapping"][lighting][zone]
    yaml_path, thresholds_path = root / "machine_config.yaml", root / "thresholds.json"
    originals = {path: path.read_bytes() for path in (yaml_path, thresholds_path)}
    thresholds = json.loads(originals[thresholds_path])
    if not isinstance(thresholds, dict) or not isinstance(thresholds.get(lighting), dict):
        raise ValueError("thresholds.json 格式錯誤，未更新門檻")
    audit.append({"at": datetime.now(timezone.utc).isoformat(), "previous_threshold": previous,
                  "threshold": report["suggested_threshold"], "model_sha256": report["model_sha256"],
                  "recipe_sha256": report["recipe_sha256"], "report_created_at": report["created_at"],
                  "acceptance_status": report["status"]})
    try:
        result = update_threshold(db, bundle_id, lighting=lighting, zone=zone, value=report["suggested_threshold"])
        write_json(audit_path, audit)
    except (OSError, ValueError):
        for path, content in originals.items():
            path.write_bytes(content)
        raise
    result["message"] = "已採用建議門檻並保存紀錄；模型尚未啟用"
    return result


def rates(samples, threshold):
    ok = [s for s in samples if s["label"] == "ok"]
    ng = [s for s in samples if s["label"] == "ng"]
    fp = sum(s["score"] >= threshold for s in ok)
    misses = sum(s["score"] < threshold for s in ng)
    return {"ok_count": len(ok), "ng_count": len(ng), "false_positives": fp,
            "misses": misses, "false_positive_rate": fp / len(ok) if ok else None,
            "miss_rate": misses / len(ng) if ng else None}


def meets_targets(metrics, config):
    return all(config.get(key) is None or (
        metrics[metric] is not None and metrics[metric] <= config[key]
    ) for key, metric in (("max_false_positive_rate", "false_positive_rate"), ("max_miss_rate", "miss_rate")))


def suggest_threshold(calibration, config):
    """Search deployable four-decimal thresholds using calibration data only."""
    ok = sorted(s["score"] for s in calibration if s["label"] == "ok")
    ng = sorted(s["score"] for s in calibration if s["label"] == "ng")
    if not ok or not ng:
        return None
    candidates = {0.0, 0.35, 10.0}
    for score in ok + ng:
        tick = math.floor(score * 10000)
        candidates.update((max(0, tick) / 10000, min(100000, tick + 1) / 10000))
    ranked = []
    for threshold in sorted(candidates):
        fp = (len(ok) - bisect_left(ok, threshold)) / len(ok)
        miss = bisect_left(ng, threshold) / len(ng)
        feasible = meets_targets({"false_positive_rate": fp, "miss_rate": miss}, config)
        ranked.append(((not feasible, fp + miss, miss, abs(threshold - 0.35), threshold), threshold))
    return min(ranked)[1]


def build_report(samples, config, *, complete=True, issues=None):
    calibration = [s for s in samples if s["role"] == "calibration"]
    acceptance = [s for s in samples if s["role"] == "acceptance"]
    suggested = suggest_threshold(calibration, config) if complete else None
    baseline = rates(acceptance, 0.35)
    proposed = rates(acceptance, suggested) if suggested is not None else None
    calibrated = rates(calibration, suggested) if suggested is not None else None
    reasons = list(issues or [])
    if config.get("split_mode") == "auto_batch" and not any(p["role"] == "acceptance" for p in config["panels"].values()):
        reasons.append("不足三組互不重疊的批次，本次全部用於正常樣本訓練，尚無獨立校正／驗收。")
    if suggested is None or not baseline["ok_count"] or not baseline["ng_count"] or not complete:
        status = "insufficient"
        reasons.append("校正與驗收各需人工確認的 OK、NG；資料缺漏時不判定合格。")
    elif any(config.get(key) is None for key in ("max_false_positive_rate", "max_miss_rate")):
        status = "unassessed"
        reasons.append("尚未同時設定可接受誤判率與漏檢率，僅提供建議與實測值。")
    else:
        status = "pass" if meets_targets(calibrated, config) and meets_targets(proposed, config) else "fail"
        if not meets_targets(calibrated, config):
            reasons.append("校正資料中找不到同時符合目標的門檻。")
    grouped = {}
    for group in sorted({s["group"] for s in acceptance}):
        items = [s for s in acceptance if s["group"] == group]
        grouped[group] = {"baseline": rates(items, 0.35), "suggested": rates(items, suggested) if suggested is not None else None}
    return {"schema_version": 1, "scope": "exported_model_tile_score_before_production_filters",
            "created_at": datetime.now(timezone.utc).isoformat(), "status": status, "reasons": reasons,
            "baseline_threshold": 0.35, "suggested_threshold": suggested,
            "selection_rule": "calibration_only: satisfy configured limits, then minimize FPR + miss rate",
            "targets": {k: config.get(k) for k in ("max_false_positive_rate", "max_miss_rate")},
            "calibration": calibrated, "baseline": baseline, "suggested": proposed,
            "acceptance_by_batch": grouped, "samples": samples}


def validation_tiles(db, job_id, lighting, zone):
    return [t for t in db.list_tile_pool(job_id, lighting=lighting, zone=zone)
            if t.get("dataset_role") in ("calibration", "acceptance")]


def freeze_inputs(tiles, train_tiles, bundle_dir, unit_label):
    """Persist labels, input hashes and evaluation images before model fitting."""
    report_dir = Path(bundle_dir) / "validation_reports" / unit_label
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    frozen, issues = [], []
    train_inputs = [{**{k: t.get(k) for k in ("id", "source_path", "panel_path", "tile_index", "tile_x", "tile_y", "tile_width", "tile_height")},
                     "sha256": sha256_file(t["source_path"])} for t in train_tiles]
    train_hashes = {t["sha256"] for t in train_inputs}
    seen = {h: "train" for h in train_hashes}
    for tile in tiles:
        if tile.get("decision") != "accept":
            continue
        if tile.get("validation_label") not in ("ok", "ng"):
            issues.append(f"tile #{tile['id']} 尚未確認 OK／NG")
            continue
        source = Path(tile["source_path"])
        try:
            digest = sha256_file(source)
            role = tile["dataset_role"]
            if digest in seen and seen[digest] != role:
                issues.append(f"tile #{tile['id']} 與其他用途影像完全相同，已排除")
                continue
            seen[digest] = role
            asset = assets_dir / f"{tile['id']}.png"
            shutil.copy2(source, asset)
        except OSError as exc:
            issues.append(f"tile #{tile['id']} 無法保存：{exc}")
            continue
        frozen.append({"tile_id": tile["id"], "role": role, "label": tile["validation_label"],
                       "group": tile["validation_group"], "panel_path": tile.get("panel_path"),
                       "source_path": str(source), "sha256": digest,
                       "asset_path": asset.relative_to(bundle_dir).as_posix()})
    write_json(report_dir / "inputs.json", {"training": train_inputs, "samples": frozen,
               "decisions": [{k: t.get(k) for k in ("id", "panel_path", "dataset_role", "validation_group", "validation_label", "decision")} for t in tiles]})
    return frozen, issues


def evaluate_model(model_path, tiles, train_tiles, config, bundle_dir, unit_label, log, cancel_event=None, frozen_inputs=None):
    """Stream already-preprocessed tiles through one exported inferencer."""
    import cv2
    from anomalib.deploy import TorchInferencer

    frozen, input_issues = frozen_inputs if frozen_inputs is not None else freeze_inputs(tiles, train_tiles, bundle_dir, unit_label)
    issues = list(input_issues)
    report_dir = Path(bundle_dir) / "validation_reports" / unit_label
    scores = []
    inferencer = TorchInferencer(path=str(model_path)) if frozen else None
    try:
        for index, item in enumerate(frozen):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("training cancelled by user")
            try:
                img = cv2.imread(str(Path(bundle_dir) / item["asset_path"]))
                if img is None:
                    raise ValueError("圖片無法讀取")
                result = inferencer.predict(img)
                score = float(result.pred_score.item()) if hasattr(result.pred_score, "item") else float(result.pred_score)
                if not math.isfinite(score) or not 0 <= score <= 10:
                    raise ValueError("模型分數無效或超出門檻可設定範圍")
                scores.append({**item, "score": score})
            except Exception as exc:
                issues.append(f"tile #{item['tile_id']} 推論失敗：{exc}")
            if (index + 1) % 100 == 0:
                log(f"{unit_label}: 獨立校正／驗收 {index + 1}/{len(frozen)}")
    finally:
        del inferencer
    report = build_report(scores, config, complete=not issues, issues=issues)
    report.update(unit_label=unit_label, model_sha256=sha256_file(model_path),
                  model_file=Path(model_path).name, validation_config=config)
    write_json(report_dir / "report.json", report)
    return {"report_path": (report_dir / "report.json").relative_to(bundle_dir).as_posix(),
            "status": report["status"], "suggested_threshold": report["suggested_threshold"]}
