"""Independent, panel-grouped calibration and acceptance for the training wizard.

All scores are exported-model tile scores, before production post-filters.
Acceptance samples never participate in threshold selection or model fitting.
"""
from bisect import bisect_left
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import shutil


ROLES = ("train", "calibration", "acceptance")


def path_key(value):
    return str(Path(value).resolve()).casefold()


def assign_batch_roles(panels, merge_physical=True):
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
        if merge_physical and name in physical:
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


def assign_panel_roles(panels):
    """Cover each selected zone while keeping every PANEL ID in one role."""
    masks = {}
    for panel in panels.values():
        group = panel["group"].casefold()
        masks[group] = masks.get(group, 0) | sum(1 if z == "inner" else 2 for z in panel["zones"])
    members = {mask: deque() for mask in (0, 1, 2, 3)}  # Outside scope, INNER, EDGE, both
    for group in sorted(masks, key=lambda g: hashlib.sha256(("validation-v1:" + g).encode()).hexdigest()):
        members[masks[group]].append(group)
    totals = {bit: sum(bool(mask & bit) for mask in masks.values()) for bit in (1, 2)}
    targets = {bit: max(1, count // 5) if count >= 3 else 0 for bit, count in totals.items()}
    required = sum(bit for bit in totals if targets[bit])
    options = {0: [()], 1: [(1,), (3,)], 2: [(2,), (3,)], 3: [(3,), (1, 2)]}[required]
    plans = []
    for calibration, acceptance in product(options, repeat=2):
        used = Counter(calibration + acceptance)
        if any(count > len(members[mask]) for mask, count in used.items()):
            continue
        if any(count and count <= sum(n for mask, n in used.items() if mask & bit)
               for bit, count in totals.items()):
            continue  # Always leave training panels for every selected zone.
        scarce = sum(n for mask, n in used.items() for bit in totals if mask & bit and not targets[bit])
        plans.append(((sum(used.values()), scarce), calibration, acceptance))
    # With two zones a covering plan exists whenever each requested zone has 3 IDs.
    _, calibration, acceptance = min(plans, key=lambda p: p[0])
    roles = {group: "train" for group in masks}
    counts = {"train": totals.copy(), "calibration": {1: 0, 2: 0}, "acceptance": {1: 0, 2: 0}}

    def take(mask, role):
        roles[members[mask].popleft()] = role
        for bit in totals:
            if mask & bit:
                counts["train"][bit] -= 1
                counts[role][bit] += 1

    for role, selected in (("calibration", calibration), ("acceptance", acceptance)):
        for mask in selected:
            take(mask, role)
    for role in ("calibration", "acceptance"):
        while True:
            choices = []
            for mask, remaining in members.items():
                gain = sum(1 if counts[role][bit] < targets[bit] else -1 for bit in totals if mask & bit)
                if remaining and gain > 0 and all(counts["train"][bit] > 1 for bit in totals if mask & bit):
                    choices.append((-gain, mask))
            if not choices:
                break
            take(min(choices)[1], role)
    for panel in panels.values():
        panel["role"] = roles[panel["group"].casefold()]


def normalize_validation_config(raw, panel_paths=None, panel_modes=None):
    if raw is None or raw == {}:
        return {}
    if not isinstance(raw, dict) or set(raw) - {"panels", "max_false_positive_rate", "max_miss_rate", "split_mode", "selected_zones"}:
        raise ValueError("validation_config 格式錯誤")
    auto = raw.get("split_mode") in ("auto_batch", "auto_panel")
    if "split_mode" in raw and not auto:
        raise ValueError("不支援的資料分組方式")
    selected_zones = raw.get("selected_zones", ["inner", "edge"])
    if not isinstance(selected_zones, list) or not selected_zones or any(z not in ("inner", "edge") for z in selected_zones):
        raise ValueError("重訓區域必須為 INNER／EDGE")
    panels = raw.get("panels")
    if not isinstance(panels, dict) or not panels:
        raise ValueError("請指定訓練、校正、驗收 panel 與批次")
    zone_selection = {}
    if panel_modes is not None:
        from capi_train_new import normalize_panel_modes, panel_mode_zones
        modes = normalize_panel_modes(panel_modes, len(panel_paths))
        zone_selection = {path_key(path): sorted(panel_mode_zones(mode)) for path, mode in zip(panel_paths, modes)}
    # Old running jobs without zone metadata keep their original assignment.
    zone_aware = raw.get("split_mode") == "auto_panel" and (panel_modes is not None or any("zones" in p for p in panels.values() if isinstance(p, dict)))
    clean, groups, physical_panels = {}, {}, {}
    for path, item in panels.items():
        if not isinstance(path, str) or not path.strip() or not isinstance(item, dict):
            raise ValueError("validation_config.panels 格式錯誤")
        role = "train" if auto else item.get("role")
        group = item.get("group")
        if raw.get("split_mode") == "auto_panel":
            group = group or Path(path).name
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
        if zone_aware:
            zones = zone_selection.get(key, item.get("zones", ["inner", "edge"]))
            if not isinstance(zones, list) or (not zones and "selected_zones" not in raw) or any(z not in ("inner", "edge") for z in zones):
                raise ValueError("PANEL 訓練區域必須為 INNER／EDGE")
            clean[key]["zones"] = sorted(set(zones) & set(selected_zones))
    if zone_aware:
        assign_panel_roles(clean)
    elif auto:
        assign_batch_roles(clean, merge_physical=raw.get("split_mode") != "auto_panel")
    if not auto and {p["role"] for p in clean.values()} != set(ROLES):
        raise ValueError("獨立驗收需要三組不同批次：訓練、校正、驗收")
    if panel_paths is not None and set(clean) != {path_key(p) for p in panel_paths}:
        raise ValueError("驗收用途設定必須與選取的 panel 完全一致")
    result = {"panels": clean}
    if "selected_zones" in raw:
        result["selected_zones"] = sorted(set(selected_zones))
    if auto:
        result["split_mode"] = raw["split_mode"]
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


def review_decision_labels(tiles, config):
    """In the simplified wizard, the final include/exclude decision is the label."""
    if config.get("split_mode") != "auto_panel":
        return tiles
    return [{**t, "validation_label": "ok" if t["decision"] == "accept" else "ng",
             "label_source": "review_decision"} for t in tiles]


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


def run_manifest_path(bundle_dir, job_id):
    if not job_id or job_id in (".", "..") or any(c in job_id for c in "/\\:"):
        raise ValueError("invalid job id")
    return Path(bundle_dir) / "validation_reports" / job_id / "manifest.json"


def load_report(bundle_dir, unit_label, report_path=None):
    # Unit comes from manifest/model mapping, never an arbitrary relative path.
    if not unit_label or any(c in unit_label for c in "/\\."):
        raise ValueError("invalid unit")
    bundle_dir = Path(bundle_dir)
    from capi_model_registry import _read_manifest
    metrics = (_read_manifest(bundle_dir).get("unit_metrics") or {}).get(unit_label) or {}
    current_path = (metrics.get("validation") or {}).get("report_path")
    if report_path is None:
        report_path = current_path or f"validation_reports/{unit_label}/report.json"
    path = (bundle_dir / report_path).resolve()
    path.relative_to((bundle_dir / "validation_reports").resolve())
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["unit_label"] != unit_label:
        raise ValueError("驗收報告與子模型不一致")
    model = bundle_dir / Path(report["model_file"]).name
    report["stale"] = (not model.is_file() or sha256_file(model) != report["model_sha256"] or
                       recipe_fingerprint(bundle_dir) != report.get("recipe_sha256") or
                       bool(current_path and (bundle_dir / current_path).resolve() != path))
    return report


def adopt_threshold(db, bundle_id, unit_label, report_path=None, expected_job_id=None):
    from capi_model_registry import update_threshold
    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        raise ValueError("找不到模型")
    if bundle.get("is_active"):
        raise ValueError("此模型已啟用；請先停用，再採用建議門檻")
    root = Path(bundle["bundle_path"])
    report = load_report(root, unit_label, report_path)
    if expected_job_id is not None and report.get("job_id") != expected_job_id:
        raise ValueError("驗收報告不屬於本次訓練")
    if report["stale"]:
        raise ValueError("模型或前處理設定已變更，此份報告僅供歷史參考")
    if report["suggested_threshold"] is None or report["status"] == "insufficient":
        raise ValueError("資料不足，沒有可採用的建議門檻")
    lighting, zone = unit_label.rsplit("-", 1)
    report_dir = ((root / report_path).parent if report_path else
                  (run_manifest_path(root, report["job_id"]).parent if report.get("job_id") else root / "validation_reports") / unit_label)
    audit_path = report_dir / "adoptions.json"
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


def suggest_threshold(calibration, config, baseline_threshold=0.35):
    """Search deployable four-decimal thresholds using calibration data only."""
    ok = sorted(s["score"] for s in calibration if s["label"] == "ok")
    ng = sorted(s["score"] for s in calibration if s["label"] == "ng")
    if not ok or not ng:
        return None
    candidates = {0.0, round(baseline_threshold, 4), 10.0}
    for score in ok + ng:
        tick = math.floor(score * 10000)
        candidates.update((max(0, tick) / 10000, min(100000, tick + 1) / 10000))
    ranked = []
    for threshold in sorted(candidates):
        fp = (len(ok) - bisect_left(ok, threshold)) / len(ok)
        miss = bisect_left(ng, threshold) / len(ng)
        feasible = meets_targets({"false_positive_rate": fp, "miss_rate": miss}, config)
        ranked.append(((not feasible, fp + miss, miss, abs(threshold - baseline_threshold), threshold), threshold))
    return min(ranked)[1]


def build_report(samples, config, *, complete=True, issues=None, baseline_threshold=0.35):
    calibration = [s for s in samples if s["role"] == "calibration"]
    acceptance = [s for s in samples if s["role"] == "acceptance"]
    suggested = suggest_threshold(calibration, config, baseline_threshold) if complete else None
    baseline = rates(acceptance, baseline_threshold)
    proposed = rates(acceptance, suggested) if suggested is not None else None
    calibrated = rates(calibration, suggested) if suggested is not None else None
    reasons = list(issues or [])
    if config.get("split_mode") in ("auto_batch", "auto_panel") and not any(p["role"] == "acceptance" for p in config["panels"].values()):
        group_name = "PANEL ID" if config["split_mode"] == "auto_panel" else "批次"
        reasons.append(f"不足三組互不重疊的 {group_name}，本次全部用於正常樣本訓練，尚無獨立校正／驗收。")
    if suggested is None or not baseline["ok_count"] or not baseline["ng_count"] or not complete:
        status = "insufficient"
        reasons.append("校正與驗收各需保留（OK）及排除（NG）切片；資料不足時不判定合格。" if config.get("split_mode") == "auto_panel"
                       else "校正與驗收各需人工確認的 OK、NG；資料缺漏時不判定合格。")
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
        grouped[group] = {"baseline": rates(items, baseline_threshold), "suggested": rates(items, suggested) if suggested is not None else None}
    return {"schema_version": 1, "scope": "exported_model_tile_score_before_production_filters",
            "label_source": "review_decision" if config.get("split_mode") == "auto_panel" else "manual_label",
            "grouping": "panel_id" if config.get("split_mode") == "auto_panel" else "batch",
            "created_at": datetime.now(timezone.utc).isoformat(), "status": status, "reasons": reasons,
            "baseline_threshold": baseline_threshold, "suggested_threshold": suggested,
            "selection_rule": "calibration_only: satisfy configured limits, then minimize FPR + miss rate",
            "targets": {k: config.get(k) for k in ("max_false_positive_rate", "max_miss_rate")},
            "calibration": calibrated, "baseline": baseline, "suggested": proposed,
            "acceptance_by_batch": grouped, "samples": samples}


def validation_tiles(db, job_id, lighting, zone, config=None):
    return review_decision_labels([t for t in db.list_tile_pool(job_id, lighting=lighting, zone=zone)
            if t.get("dataset_role") in ("calibration", "acceptance")], config or {})


def freeze_inputs(tiles, train_tiles, bundle_dir, unit_label, job_id=None):
    """Persist labels, input hashes and evaluation images before model fitting."""
    report_dir = (run_manifest_path(bundle_dir, job_id).parent if job_id else Path(bundle_dir) / "validation_reports") / unit_label
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    frozen, issues = [], []
    train_inputs = [{**{k: t.get(k) for k in ("id", "source_path", "panel_path", "tile_index", "tile_x", "tile_y", "tile_width", "tile_height")},
                     "sha256": sha256_file(t["source_path"])} for t in train_tiles]
    train_hashes = {t["sha256"] for t in train_inputs}
    seen = {h: "train" for h in train_hashes}
    for tile in tiles:
        if tile.get("decision") != "accept" and tile.get("label_source") != "review_decision":
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
                       "label_source": tile.get("label_source", "manual_label"), "decision": tile.get("decision"),
                       "group": tile["validation_group"], "panel_path": tile.get("panel_path"),
                       "source_path": str(source), "sha256": digest,
                       "asset_path": asset.relative_to(bundle_dir).as_posix()})
    write_json(report_dir / "inputs.json", {"training": train_inputs, "samples": frozen,
               "decisions": [{k: t.get(k) for k in ("id", "panel_path", "dataset_role", "validation_group", "validation_label", "decision")} for t in tiles]})
    return frozen, issues


def evaluate_model(model_path, tiles, train_tiles, config, bundle_dir, unit_label, log, cancel_event=None, frozen_inputs=None, *, baseline_threshold=0.35, job_id=None, fail_on_error=False):
    """Stream already-preprocessed tiles through one exported inferencer."""
    import cv2
    from anomalib.deploy import TorchInferencer

    frozen, input_issues = frozen_inputs if frozen_inputs is not None else freeze_inputs(tiles, train_tiles, bundle_dir, unit_label, job_id)
    issues = list(input_issues)
    report_dir = (run_manifest_path(bundle_dir, job_id).parent if job_id else Path(bundle_dir) / "validation_reports") / unit_label
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
                if fail_on_error or "out of memory" in str(exc).lower():
                    raise
                issues.append(f"tile #{item['tile_id']} 推論失敗：{exc}")
            if (index + 1) % 100 == 0:
                log(f"{unit_label}: 獨立校正／驗收 {index + 1}/{len(frozen)}")
    finally:
        del inferencer
    report = build_report(scores, config, complete=not issues, issues=issues, baseline_threshold=baseline_threshold)
    report.update(unit_label=unit_label, model_sha256=sha256_file(model_path),
                  model_file=Path(model_path).name, validation_config=config)
    if job_id:
        report["job_id"] = job_id
    write_json(report_dir / "report.json", report)
    return {"report_path": (report_dir / "report.json").relative_to(bundle_dir).as_posix(),
            "status": report["status"], "suggested_threshold": report["suggested_threshold"], "job_id": job_id}
