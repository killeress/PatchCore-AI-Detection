"""CPU-only exclusion rules for the training wizard; never modify image pixels."""
import json
import math

import cv2
import numpy as np


def normalize_rules(value):
    if not isinstance(value, dict):
        raise ValueError("排除規則格式錯誤")
    if type(value.get("dark_spots", False)) is not bool:
        raise ValueError("黑點開關格式錯誤")
    result = {"regions": [], "dark_spots": value.get("dark_spots", False)}
    regions = value.get("regions", [])
    if not isinstance(regions, list) or len(regions) > 30:
        raise ValueError("最多可設定 30 個排除區域")
    for rect in regions:
        if (not isinstance(rect, list) or len(rect) != 4
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in rect)):
            raise ValueError("排除區域座標錯誤")
        x, y, w, h = rect
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.000001 or y + h > 1.000001:
            raise ValueError("排除區域必須位於 PANEL 內")
        result["regions"].append(rect)
    for key, default, low, high in (("contrast", 25, 1, 255), ("min_area", 8, 1, 4096),
                                   ("max_area", 800, 1, 16384)):
        val = value.get(key, default)
        if type(val) is not int or not low <= val <= high:
            raise ValueError(f"{key} 範圍必須是 {low}–{high}")
        result[key] = val
    if result["min_area"] > result["max_area"]:
        raise ValueError("最小面積不可大於最大面積")
    return result


def geometry(tile):
    raw = tile.get("review_geometry")
    return (json.loads(raw) or {}) if isinstance(raw, str) and raw else raw or {}


def exclusion_reason(tile, rules):
    """Return evidence or None. Bbox-normalized ROI supports translation/scale, not rotation."""
    bbox = geometry(tile).get("panel_bbox")
    if bbox and tile.get("tile_x") is not None:
        bx, by, bx2, by2 = bbox
        x, y = tile["tile_x"], tile["tile_y"]
        w, h = tile["tile_width"], tile["tile_height"]
        for rx, ry, rw, rh in rules["regions"]:
            left, top = bx + rx * (bx2 - bx), by + ry * (by2 - by)
            right, bottom = left + rw * (bx2 - bx), top + rh * (by2 - by)
            if x < right and x + w > left and y < bottom and y + h > top:
                return {"kind": "region", "region": [rx, ry, rw, rh]}
    if not rules["dark_spots"]:
        return None
    # imdecode handles Unicode paths on Windows. Full tile, never the 96px thumbnail.
    gray = cv2.imdecode(np.fromfile(tile["source_path"], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("切片無法讀取")
    response = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    mask = (response >= rules["contrast"]).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    height, width = gray.shape
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        # Open components at crop boundaries and long edges are not isolated specks.
        margin = 4 if tile.get("zone") == "edge" else 1
        if (x < margin or y < margin or x + w > width - margin or y + h > height - margin
                or max(w, h) > 5 * min(w, h)):
            continue
        if rules["min_area"] <= area <= rules["max_area"]:
            return {"kind": "dark_spot", "bbox": [x, y, w, h], "area": area,
                    "contrast": int(response[labels == i].max())}
    return None


def apply_rules(db, job, lighting, zone, rules, *, save=False, state="review"):
    """Analyze one unit sequentially; commit decisions/evidence/config together if still editable."""
    rules = normalize_rules(rules)
    scope = (job.get("training_scope") or {}).get("selected_units")
    if scope is not None and f"{lighting}-{zone}" not in scope:
        raise ValueError("此 PT 不在本次訓練範圍")
    tiles = db.list_tile_pool(job["job_id"], lighting=lighting, zone=zone, source="ok")
    changes, errors, missing = [], [], 0
    for tile in tiles:
        if tile["decision"] != "accept" or tile.get("auto_exclusion"):
            continue  # Preserve manual restoration, including subsequent automatic runs.
        if rules["regions"] and not geometry(tile).get("panel_bbox"):
            missing += 1
        try:
            evidence = exclusion_reason(tile, rules)
            if evidence:
                changes.append((tile["id"], json.dumps({**evidence, "rules": rules})))
        except (OSError, ValueError, cv2.error):
            errors.append(tile["id"])
    updated = db.apply_review_exclusions(job["job_id"], lighting, zone, rules,
                                         changes, save=save, expected_state=state)
    return {"excluded": updated, "read_errors": len(errors), "missing_geometry": missing}


def apply_saved_rules(db, job_id, log):
    job = db.get_training_job(job_id)
    units = {(t["lighting"], t["zone"]) for t in db.list_tile_pool(job_id, source="ok")
             if t.get("zone") in ("inner", "edge")}
    selected = (job.get("training_scope") or {}).get("selected_units")
    for lighting, zone in sorted(units):
        if selected is not None and f"{lighting}-{zone}" not in selected:
            continue
        rules = db.get_review_rules(job["machine_id"], lighting, zone)
        if isinstance(rules, dict) and rules:
            summary = apply_rules(db, job, lighting, zone, rules, state="preprocess")
            log(f"自動排除 {lighting}-{zone}: {summary['excluded']} 張；"
                f"讀取失敗 {summary['read_errors']} 張，缺少區域座標 {summary['missing_geometry']} 張")
