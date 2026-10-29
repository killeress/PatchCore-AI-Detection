"""Read-only AOI/side-camera comparison for human validation, never a verdict."""

import json
from pathlib import Path

import cv2
import numpy as np


REVIEW_DECISIONS = {
    "unreviewed": "待驗證", "confirmed": "檢出正確", "correct_clear": "無缺陷確認",
    "false_positive": "系統過檢", "missed": "系統漏檢", "mapping_error": "座標映射不正確",
    "uncertain": "無法確認",
}

OMIT_COMPARISON_STATES = {
    "unavailable": "OMIT 無法比對", "disabled": "OMIT 比對關閉", "no_candidate": "無候選可比對",
    "clear": "OMIT 未達重疊門檻", "suspected": "OMIT 符合／疑似表面異物",
    "partial": "OMIT 部分屏蔽，仍有候選", "suppressed": "OMIT 已屏蔽候選",
}

# Shared by generated previews, historical previews and the HTML legend.
SIDE_ANNOTATION_LEGEND = [
    {"key": "aoi", "color": "#F43F5E", "tag": "+ A1", "label": "AOI 報點",
     "hint": "中空十字標示 AOI TXT 映射位置，中心保留原始像素；A1 對應下方完整座標。"},
    {"key": "candidate", "color": "#FACC15", "tag": "#1", "label": "保留檢查候選",
     "hint": "黃色只表示 AOI 報點實際命中且仍需保留檢查的候選，編號沿用原始明細。"},
    {"key": "reference", "color": "#94A3B8", "tag": "#", "label": "參考候選",
     "hint": "灰色候選未涵蓋 AOI 報點，僅供影像對照，不影響該點轉 OK。"},
    {"key": "mark", "color": "#22D3EE", "tag": "MARK / PRINT", "label": "MARK／印字屏蔽",
     "hint": "MARK 為定位標記，PRINT 為角落印字；框線表示屏蔽範圍。"},
    {"key": "zone", "color": "#A78BFA", "tag": "ZONE", "label": "設定不檢測區",
     "hint": "由正拍不檢測區設定映射而來，不代表已辨識到易撕貼。"},
    {"key": "tape", "color": "#34D399", "tag": "TAPE", "label": "角落膠貼屏蔽",
     "hint": "側拍偵測到的角落膠貼範圍；易撕貼結論另須符合方向與位置條件。"},
    {"key": "bomb", "color": "#F97316", "tag": "B1", "label": "炸彈驗證範圍",
     "hint": "炸彈座標與容許誤差範圍，用於驗證檢出能力。"},
]


def annotation_color(key):
    color = next(item["color"] for item in SIDE_ANNOTATION_LEGEND if item["key"] == key)
    return tuple(int(color[i:i + 2], 16) for i in (5, 3, 1))  # OpenCV BGR


def omit_sticker_candidate(payload, candidate):
    """Recognize a fully covered dark object even when its surrounding CV halo remains.

    This classifies the surface object; it does not rewrite the saved pixel mask.
    A partial core match, missing evidence, or an observation-only run is insufficient.
    """
    omit, dust = payload.get("omit", {}), candidate.get("dust", {})
    params = payload.get("parameters", {})
    return bool(
        params.get("dust_mode") == "suppress"
        and omit.get("status") == "available"
        and omit.get("coordinate_space") == "side_detection_pixels"
        and omit.get("size") and omit["size"] == payload.get("side_size")
        and dust.get("status") == "suppressed" and not candidate.get("bomb_ids")
        and dust.get("core_overlap_ratio") == 1.0
        and (dust.get("core_area_px") or 0) >= params.get("min_area_px", 20)
        and dust.get("suppressed_regions")
    )


CONVERSION_FILTERS = {
    "mark": "含 MARK 可轉 OK 點", "omit": "含貼紙／表面異物可轉 OK 點", "tape": "含易撕貼可轉 OK 點",
    "both": "同時含 MARK 與貼紙可轉 OK 點", "mixed": "含多類可轉 OK 點",
    "either": "含任一類可轉 OK 點", "eligible": "三類合併可整筆轉 OK",
    "eligible_mark": "僅靠 MARK 可整筆轉 OK", "eligible_omit": "僅靠 OMIT 可整筆轉 OK",
    "eligible_tape": "僅靠易撕貼可整筆轉 OK",
    "blocked": "有可轉 OK 點，但整筆仍需保留",
}


def _covered_csjbl_ng(defect, points):
    """Match a formal NG reason to a saved AOI point from the same screen."""
    from capi_image_naming import canonical_image_prefix
    if (not isinstance(defect, dict) or defect.get("type") != "aoi_passthrough"
            or str(defect.get("defect_code", "")).upper() != "CSJBL"):
        return False
    try:
        xy = np.asarray([defect["product_x"], defect["product_y"]], np.float64)
        prefix = canonical_image_prefix(str(defect.get("image", ""))).upper()
        return bool(prefix and np.isfinite(xy).all() and any(
            canonical_image_prefix(p["image_prefix"]).upper() == prefix
            and np.allclose(xy, p["product_xy"], atol=.001, rtol=0) for p in points))
    except (ValueError, TypeError, KeyError):
        return False


def record_conversion(row, comparison):
    """A point suggestion only removes a rejection when every formal NG is covered."""
    points = comparison["points"]
    mark = sum(p["status"] == "mark_suppressed" for p in points)
    omit = sum(p["status"] == "omit_suppressed" for p in points)
    tape = sum(p["status"] == "tape_suppressed" for p in points)
    kinds = [kind for kind, count in (("mark", mark), ("omit", omit), ("tape", tape)) if count]
    result = {"mark_points": mark, "omit_points": omit, "tape_points": tape, "eligible": False,
              "category": "mixed" if len(kinds) > 1 else kinds[0] if kinds else "none",
              "state": "none", "reason": "沒有 MARK／貼紙／易撕貼可轉 OK 點"}
    if not kinds:
        return result
    judgment = str(row.get("ai_judgment") or "")
    result["state"] = "blocked"
    if judgment in {"OK", "OK-i"}:
        result.update(state="already_ok", reason="正式結果已是 OK，不重複計算排片減少")
    elif not judgment.startswith("NG"):
        result["reason"] = "正式結果不是有效 NG，不納入排片減少"
    elif mark + omit + tape != len(points):
        result["reason"] = "仍有 CSJBL 點未符合 MARK／貼紙／易撕貼轉 OK 條件"
    elif row.get("review_decision") in {"confirmed", "missed", "mapping_error", "uncertain"}:
        result["reason"] = "既有紀錄仍有缺陷、定位疑慮或尚無法確認，暫不計入整筆轉 OK"
    else:
        details = _json(row.get("ng_details"), None)
        covered = isinstance(details, list) and bool(details) and all(
            _covered_csjbl_ng(defect, points) for defect in details)
        if covered:
            result.update(eligible=True, state="eligible", reason="全部 CSJBL 點及正式 NG 原因均由 MARK／貼紙／易撕貼涵蓋")
        else:
            result["reason"] = "另有正式 NG 原因，或 NG 明細不足以確認全部已涵蓋"
    return result


def panel_projection(row, comparison, conversion):
    """Explain the same record-wide projection used by rejection-rate statistics."""
    from capi_image_naming import canonical_image_prefix
    formal = str(row.get("ai_judgment") or "")
    panel = {"formal": formal, "projected": None, "state": "unavailable", "tone": "muted",
             "text": "無法試算", "reason": "尚無有效正式判定", "remaining_sources": []}
    if formal in {"OK", "OK-i"}:
        panel.update(projected=formal, state="already_ok", tone="ok", text=f"原本已 {formal}",
                     reason="沿用同次檢測的整片正式結果")
    elif formal.startswith("NG"):
        panel.update(projected="NG", state="retained", tone="warn", text="整片仍為 NG",
                     reason=conversion["reason"])
        if conversion["eligible"]:
            panel.update(projected="OK", state="eligible", tone="ok", text="整片可轉 OK",
                         reason="各畫面的全部正式 NG 原因均已由側拍處理涵蓋")
        else:
            details = _json(row.get("ng_details"), None)
            sources = []
            for defect in details if isinstance(details, list) else []:
                if isinstance(defect, dict) and not _covered_csjbl_ng(defect, comparison["points"]):
                    prefix = canonical_image_prefix(str(defect.get("image") or "")).upper()
                    if prefix and prefix not in sources:
                        sources.append(prefix)
            panel["remaining_sources"] = sources
            if sources:
                panel["reason"] = "仍有正式 NG：" + "、".join(sources)
            elif not isinstance(details, list) or not details:
                panel["reason"] = "正式 NG 明細不足，無法確認整片可轉 OK"
    elif formal:
        panel["reason"] = f"正式結果：{formal}，不納入轉 OK 試算"
    return panel


def matches_conversion_filter(result, value):
    return not value or {
        "mark": result["mark_points"] > 0, "omit": result["omit_points"] > 0,
        "tape": result["tape_points"] > 0,
        "both": result["mark_points"] > 0 and result["omit_points"] > 0,
        "mixed": result["category"] == "mixed", "either": result["category"] != "none",
        "eligible": result["eligible"], "eligible_mark": result["eligible"] and result["category"] == "mark",
        "eligible_omit": result["eligible"] and result["category"] == "omit", "blocked": result["state"] == "blocked",
        "eligible_tape": result["eligible"] and result["category"] == "tape",
    }.get(value, False)


def conversion_rate_projection(counts, eligible):
    """Fixed denominator; records covered by multiple treatments are removed once."""
    denominator = counts["ok"] + counts["ng"]
    percentage = lambda value: round(100 * value / denominator, 6) if denominator else None
    scenarios = {}
    for key, removed in (("mark", eligible["mark"]), ("omit", eligible["omit"]),
                         ("tape", eligible.get("tape", 0)),
                         ("combined", sum(eligible.values()))):
        scenarios[key] = {"converted": removed, "remaining_ng": counts["ng"] - removed,
                          "after_pct": percentage(counts["ng"] - removed), "drop_pp": percentage(removed),
                          "relative_drop_pct": round(100 * removed / counts["ng"], 6) if counts["ng"] else None}
    return {**counts, "denominator": denominator, "before_pct": percentage(counts["ng"]), "scenarios": scenarios}


def _point_omit_evidence(payload, candidates):
    """Expose saved same-camera overlap evidence without reclassifying the image."""
    omit = payload.get("omit", {})
    evidence = {"status": "unavailable", "reason": omit.get("reason") or "此筆未保存側拍 OMIT 比對證據",
                "candidates": []}
    if payload.get("parameters", {}).get("dust_mode") == "off":
        return {**evidence, "status": "disabled", "reason": "OMIT 表面異物比對未啟用"}
    if (omit.get("status") != "available" or omit.get("coordinate_space") != "side_detection_pixels"
            or not omit.get("size") or omit["size"] != payload.get("side_size")):
        return evidence
    relevant = [c for c in candidates if not c.get("bomb_ids") and c.get("disposition") != "mark_suppressed"]
    if not relevant:
        return {**evidence, "status": "no_candidate", "reason": "沒有可比對的非 MARK／炸彈候選；不代表 OK"}
    for candidate in relevant:
        dust = candidate.get("dust", {})
        evidence["candidates"].append({"id": candidate["id"], "status": dust.get("status", "unavailable"),
            "overlap_ratio": dust.get("overlap_ratio"), "mask_overlap_ratio": dust.get("mask_overlap_ratio"),
            "compared_area_px": dust.get("compared_area_px"), "overlap_area_px": dust.get("overlap_area_px"),
            "uncovered_area_px": dust.get("uncovered_area_px"), "remaining_area_px": candidate.get("remaining_area_px"),
            "core_overlap_ratio": dust.get("core_overlap_ratio"), "core_area_px": dust.get("core_area_px"),
            "halo_removed_area_px": dust.get("halo_removed_area_px", 0), "match_basis": dust.get("match_basis"),
            "sticker": omit_sticker_candidate(payload, candidate),
            "reason": dust.get("reason", "此候選未保存 OMIT 證據")})
    states = {c["status"] for c in evidence["candidates"]}
    if states - {"clear", "suspected", "suppressed"}:
        evidence["reason"] = "部分候選未完成 OMIT 比對，保留人工確認"
    elif any(c.get("dust", {}).get("status") == "suppressed" and c.get("disposition") == "retained"
             and not omit_sticker_candidate(payload, c) for c in relevant):
        evidence.update(status="partial", reason="已移除 OMIT 重疊像素，範圍外仍有異常候選")
    elif "suspected" in states:
        evidence.update(status="suspected", reason="OMIT 表面特徵達重疊門檻；此筆保存為僅標示結果，候選保留")
    elif "suppressed" in states:
        evidence.update(status="suppressed", reason=("貼紙本體已由 OMIT 完整覆蓋，該貼紙周圍差分不再阻擋轉 OK"
                        if any(c["sticker"] for c in evidence["candidates"]) else
                        "候選已依 OMIT 表面證據屏蔽；仍需確認其他 AOI 點"))
    else:
        evidence.update(status="clear", reason="OMIT 表面特徵未達重疊門檻，候選保留")
    return evidence


def _inside_omit_regions(point, regions):
    xy = tuple(map(float, point))
    for region in regions:
        outer = np.asarray(region.get("contour", []), np.float32)
        if len(outer) < 3 or cv2.pointPolygonTest(outer, xy, False) < 0:
            continue
        if not any(len(hole) >= 3 and cv2.pointPolygonTest(np.asarray(hole, np.float32), xy, False) >= 0
                   for hole in region.get("holes", [])):
            return True
    return False


def _json(value, default):
    try:
        return json.loads(value) if isinstance(value, str) and value else value or default
    except (ValueError, TypeError):
        return default


def csjbl_points(coords, ng_details=""):
    """Prefer the original AOI snapshot; old records may only have passthrough details."""
    stored = _json(coords, None)
    if isinstance(stored, dict) and stored:
        return [{**item, "image_prefix": prefix}
                for prefix, items in stored.items() if isinstance(items, list)
                for item in items if isinstance(item, dict)
                and str(item.get("defect_code", "")).upper() == "CSJBL"], "aoi"
    legacy = _json(ng_details, [])
    points = [{**item, "image_prefix": item.get("image", ""), "coordinate_space": "product"}
              for item in legacy if isinstance(item, dict)
              and item.get("type") == "aoi_passthrough"
              and str(item.get("defect_code", "")).upper() == "CSJBL"] if isinstance(legacy, list) else []
    return points, "legacy" if points else "empty"


def has_csjbl(coords, ng_details):
    return int(bool(csjbl_points(coords, ng_details)[0]))


def _side_record_source(row, detail=None):
    source = {**(detail or {}), **row}
    context = (source.get("payload") or {}).get("evidence_context", {})
    if not source.get("aoi_machine_coords") and not source.get("ng_details"):
        coords = {}
        for point in context.get("aoi_csjbl_points", []):
            coords.setdefault(point.get("image_prefix", ""), []).append(point)
        source["aoi_machine_coords"] = coords
    if "inference_marks" not in source:
        source["inference_marks"] = inference_mark_evidence(source.get("images", []))
    return source


def inference_mark_evidence(images):
    """Copy the locator regions, including underlines, from this inference only."""
    evidence = []
    for image in images or []:
        source = image.get("mark_source_image")
        if not source:
            continue
        regions = image.get("mark_exclusion_regions")
        if regions is None:  # Older DB rows saved the main bbox only.
            try:
                box = image.get("mark_bbox", "")
                x, y, w, h = map(float, box.split(",") if isinstance(box, str) else box)
                regions = [{"name": "mark_binary", "x1": x, "y1": y, "x2": x + w, "y2": y + h}]
            except (TypeError, ValueError):
                regions = []
        for region in regions:
            try:
                x1, y1, x2, y2 = [float(region[k]) for k in ("x1", "y1", "x2", "y2")]
                if not np.isfinite([x1, y1, x2, y2]).all() or min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
                    continue
                item = {"source_image": str(source), "name": str(region.get("name", "mark_binary")),
                        "front_bbox": [x1, y1, x2 - x1, y2 - y1]}
                if item not in evidence:
                    evidence.append(item)
            except (KeyError, TypeError, ValueError):
                continue
    return evidence


def mapped_inference_marks(payload, fallback=()):
    """Reuse only exact-source MARK regions in the saved detection orientation.

    AOI matching tolerance must never enlarge a MARK exclusion. Missing or invalid
    registration supplies no exclusion, including for historical report rows.
    """
    context = payload.get("evidence_context", {})
    if context.get("mark_rotation_applied", payload.get("rotation_applied", False)) != payload.get("rotation_applied", False):
        return []
    try:
        mapping = payload.get("mapping", {})
        if mapping.get("status") != "estimated":
            return []
        inverse = np.linalg.inv(np.asarray(mapping["matrix"], np.float64))
        width, height = payload["front_size"]
        if inverse.shape != (3, 3) or not np.isfinite(inverse).all() or min(width, height) <= 0:
            return []
    except (KeyError, ValueError, TypeError, np.linalg.LinAlgError):
        return []
    marks = []
    for evidence in context.get("inference_marks", fallback):
        try:
            source = str(evidence["source_image"])
            if not source or source.casefold() != str(payload.get("front_image", "")).casefold():
                continue
            x, y, w, h = map(float, evidence["front_bbox"])
            if (not np.isfinite([x, y, w, h]).all() or min(x, y) < 0 or min(w, h) <= 0
                    or x + w > width or y + h > height):
                continue
            polygon = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            side_polygon = _transform(polygon, inverse)
            if not cv2.isContourConvex(side_polygon) or abs(cv2.contourArea(side_polygon)) < 1:
                continue
            marks.append({**evidence, "front_polygon": polygon, "side_polygon": side_polygon.tolist()})
        except (KeyError, ValueError, TypeError, cv2.error):
            continue
    return marks


def _transform(points, matrix):
    result = cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), matrix).reshape(-1, 2)
    if not np.isfinite(result).all():
        raise ValueError("映射座標無效")
    return result


def _report_mark_regions(payload, fallback=()):
    """Keep exact formal and saved side-detection masks, without ROI padding."""
    marks = mapped_inference_marks(payload, fallback)
    try:
        width, height = map(float, payload["side_size"])
        if not np.isfinite([width, height]).all() or min(width, height) <= 0:
            return marks
    except (KeyError, TypeError, ValueError):
        return marks
    for exclusion in payload.get("exclusions", []):
        if exclusion.get("type") not in {"mark", "corner_print"}:
            continue
        try:
            x, y, w, h = map(float, exclusion["bbox"])
            if (not np.isfinite([x, y, w, h]).all() or min(x, y) < 0 or min(w, h) <= 0
                    or x + w > width or y + h > height):
                continue
            polygon = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            if not any(np.allclose(polygon, m["side_polygon"], atol=.001, rtol=0) for m in marks):
                marks.append({"name": exclusion["type"], "side_bbox": [x, y, w, h],
                              "side_polygon": polygon})
        except (KeyError, TypeError, ValueError):
            continue
    return marks


def peel_tape_regions(payload):
    """Reuse bounded corner-tape detections, including saved v5 observations.

    The detector has already removed these components before finding candidates.
    Position alone, or the AOI matching tolerance, must never create a tape mask.
    The physical bottom-right corner becomes top-left after a 180 degree rotation.
    """
    try:
        width, height = map(float, payload["side_size"])
        quad = np.asarray(payload["side_polygon"], np.float32)
        if (not np.isfinite([width, height]).all() or min(width, height) <= 0
                or quad.shape != (4, 2) or not np.isfinite(quad).all()
                or not cv2.isContourConvex(quad) or abs(cv2.contourArea(quad)) < 1):
            return []
        panel_width, panel_height = np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0])
        to_unit = cv2.getPerspectiveTransform(quad, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32))
    except (KeyError, ValueError, TypeError, cv2.error):
        return []
    regions = []
    for exclusion in payload.get("exclusions", []):
        if exclusion.get("type") != "corner_tape":
            continue
        try:
            x, y, w, h = map(float, exclusion["bbox"])
            if (not np.isfinite([x, y, w, h]).all() or min(x, y) < 0
                    or x + w > width or y + h > height or not 20 <= w <= panel_width * .12
                    or not 10 <= h <= panel_height * .1 or w <= h * 1.2):
                continue
            u, v = _transform([[x + w / 2, y + h / 2]], to_unit)[0]
            in_corner = (0 <= u < .12 and 0 <= v < .08) if payload.get("rotation_applied") else (
                .88 < u <= 1 and .92 < v <= 1)
            if not in_corner:
                continue
            bbox = [x, y, w, h]
            if any(r["bbox"] == bbox for r in regions):
                continue
            regions.append({"id": len(regions) + 1, "bbox": bbox,
                            "side_polygon": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]})
        except (KeyError, ValueError, TypeError, cv2.error):
            continue
    return regions


def compare_side_record(row, tolerance=50):
    """Map front product coordinates to the saved side-camera detection orientation.

    Proximity selects reference candidates only. Only remaining contours containing
    the actual AOI point can block it. Filtered and bomb candidates stay distinct.
    A MARK inside the point's ROI removes only candidates covered by its exact mask.
    """
    row = _side_record_source(row)
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 500:
        raise ValueError("比對容許誤差須為 0～500 產品 px")
    points, source = csjbl_points(row.get("aoi_machine_coords"), row.get("ng_details"))
    payload = row.get("payload") or {}
    result = {"has_csjbl": bool(points), "source": source, "points": [],
              "tolerance": tolerance, "matched": 0, "mapped": 0, "mark_suppressed": 0, "omit_suppressed": 0,
              "tape_suppressed": 0,
              "incomplete": bool(payload.get("truncated")), "reason": ""}
    if not points:
        return result
    marks = _report_mark_regions(payload, row.get("inference_marks", []))
    tapes = peel_tape_regions(payload)
    to_front = to_side = to_product = None
    resolution = payload.get("evidence_context", {}).get("product_resolution") or [
        row.get("resolution_x", 0), row.get("resolution_y", 0)]
    try:
        width, height = map(float, resolution)
        if not np.isfinite([width, height]).all() or min(width, height) <= 0:
            raise ValueError("未保存有效產品解析度")
        mapping = payload.get("mapping", {})
        if mapping.get("status") != "estimated":
            raise ValueError(mapping.get("reason") or "沒有可用的正側拍映射")
        side_to_front = np.asarray(mapping.get("matrix"), np.float64)
        front_quad = np.asarray(mapping.get("front_polygon"), np.float32)
        if (side_to_front.shape != (3, 3) or front_quad.shape != (4, 2)
                or not np.isfinite(side_to_front).all() or not np.isfinite(front_quad).all()
                or abs(np.linalg.det(side_to_front)) < 1e-12 or abs(cv2.contourArea(front_quad)) < 1):
            raise ValueError("保存的面板映射資料無效")
        product_quad = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
        to_front = cv2.getPerspectiveTransform(product_quad, front_quad)
        to_side = np.linalg.inv(side_to_front) @ to_front
        to_product = np.linalg.inv(to_side)
    except (ValueError, TypeError, cv2.error, np.linalg.LinAlgError) as exc:
        result["reason"] = str(exc)
        to_side = None

    for index, point in enumerate(points, 1):
        item = {"id": index, "defect_code": point.get("defect_code", "CSJBL"),
                "image_prefix": point.get("image_prefix", ""),
                "source_xy": [point.get("source_x", point.get("product_x")),
                              point.get("source_y", point.get("product_y"))],
                "coordinate_space": point.get("coordinate_space", "product"),
                "status": "unavailable", "reason": result["reason"],
                "candidate_ids": [], "aoi_candidate_ids": [], "retained_candidate_ids": [], "sticker_candidate_ids": [],
                "tape_candidate_ids": [], "mark_candidate_ids": [],
                "omit": {"status": "unavailable", "label": OMIT_COMPARISON_STATES["unavailable"],
                         "reason": "尚無有效 AOI 映射", "candidates": []}}
        result["points"].append(item)
        if to_side is None:
            continue
        try:
            if item["coordinate_space"] == "product":
                product_xy = np.array([point.get("product_x"), point.get("product_y")], np.float32)
            elif item["coordinate_space"] == "image":
                # Image coordinates belong to a specific exposure, unlike product coordinates.
                from capi_image_naming import canonical_image_prefix
                if canonical_image_prefix(item["image_prefix"]).upper() != "W0F00000":
                    raise ValueError("此 CSJBL 為其他正拍畫面的像素座標，無法套用白畫面映射")
                front_xy = np.array(item["source_xy"], np.float32)
                if payload.get("rotation_applied"):
                    front_xy = np.array(payload["front_size"]) - 1 - front_xy
                product_xy = _transform([front_xy], np.linalg.inv(to_front))[0]
            else:
                raise ValueError("不支援的 AOI 座標種類")
            if (not np.isfinite(product_xy).all() or np.any(product_xy < 0)
                    or np.any(product_xy > [width, height])):
                raise ValueError("AOI 座標超出產品範圍或未保存")
            front_xy, side_xy = _transform([product_xy], to_front)[0], _transform([product_xy], to_side)[0]
            item.update(product_xy=product_xy.tolist(), front_xy=front_xy.tolist(), side_xy=side_xy.tolist())
            for kind, xy in (("front", front_xy), ("side", side_xy)):
                item[kind + "_raw_xy"] = ((np.array(payload[kind + "_size"]) - 1 - xy)
                                         if payload.get("rotation_applied") else xy).tolist()
            angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
            circle = product_xy + np.column_stack((np.cos(angles), np.sin(angles))) * tolerance
            item["side_roi"] = _transform(circle, to_side).tolist()
            item["front_roi"] = _transform(circle, to_front).tolist()
            # Pixel bounds are half-open, matching the formal inference mask.
            point_marks = []
            for mark in marks:
                x, y, w, h = mark.get("front_bbox", mark.get("side_bbox"))
                mark_xy = front_xy if "front_bbox" in mark else side_xy
                if x <= mark_xy[0] < x + w and y <= mark_xy[1] < y + h:
                    point_marks.append(mark)
            roi_marks = [mark for mark in marks if mark in point_marks or (tolerance > 0 and
                cv2.intersectConvexConvex(np.asarray(mark["side_polygon"], np.float32),
                                         np.asarray(item["side_roi"], np.float32))[0] > 0)]
            point_tapes = [t for t in tapes if t["bbox"][0] <= side_xy[0] < t["bbox"][0] + t["bbox"][2]
                           and t["bbox"][1] <= side_xy[1] < t["bbox"][1] + t["bbox"][3]]
            nearby, omit_covers_point = [], False
            for candidate in payload.get("candidates", []):
                contour = candidate.get("side_contour")
                if not contour or len(contour) < 3:
                    continue
                product_contour = _transform(contour, to_product)
                distance = cv2.pointPolygonTest(product_contour, tuple(map(float, product_xy)), True)
                if distance < -tolerance:
                    continue
                item["candidate_ids"].append(candidate["id"])
                if distance >= 0:
                    item["aoi_candidate_ids"].append(candidate["id"])
                nearby.append(candidate)
                sticker = omit_sticker_candidate(payload, candidate)
                # AOI can land on the object's surrounding contrast halo. Use
                # the actual candidate contour, never the report tolerance or bbox.
                in_hole = any(len(hole) >= 3 and cv2.pointPolygonTest(
                    np.asarray(hole, np.float32), tuple(map(float, side_xy)), False) >= 0
                    for region in candidate.get("dust", {}).get("suppressed_regions", [])
                    for hole in region.get("holes", []))
                if sticker and distance >= 0 and not in_hole:
                    omit_covers_point = True
                    item["sticker_candidate_ids"].append(candidate["id"])
                if (distance >= 0 and payload.get("parameters", {}).get("dust_mode") == "suppress"
                        and candidate.get("dust", {}).get("status") == "suppressed" and not candidate.get("bomb_ids")
                        and _inside_omit_regions(side_xy, candidate["dust"].get("suppressed_regions", []))):
                    omit_covers_point = True
                if (candidate.get("disposition", "retained") == "retained"
                        and not candidate.get("bomb_ids") and not sticker):
                    remaining = candidate.get("remaining_contours", [contour])
                    # Old results may still contain a MARK candidate. Only remove
                    # whole contours contained in a confirmed MARK, never neighbours.
                    outside_mark = [c for c in remaining if not (roi_marks and len(c) >= 3 and any(
                        all(cv2.pointPolygonTest(np.asarray(m["side_polygon"], np.float32),
                                                tuple(map(float, xy)), False) >= 0 for xy in c)
                        for m in roi_marks))]
                    if remaining and not outside_mark:
                        item["mark_candidate_ids"].append(candidate["id"])
                    remaining = outside_mark
                    if point_tapes:
                        outside_tape = [c for c in remaining if not any(
                            len(c) >= 3 and all(cv2.pointPolygonTest(np.asarray(t["side_polygon"], np.float32),
                                tuple(map(float, xy)), False) >= 0 for xy in c) for t in point_tapes)]
                        if remaining and not outside_tape:
                            item["tape_candidate_ids"].append(candidate["id"])
                        remaining = outside_tape
                    if any(len(c) >= 3 and cv2.pointPolygonTest(_transform(c, to_product),
                           tuple(map(float, product_xy)), False) >= 0 for c in remaining):
                        item["retained_candidate_ids"].append(candidate["id"])
            item["status"] = ("matched" if item["retained_candidate_ids"] else
                              "filtered" if item["aoi_candidate_ids"] else "unmatched")
            item["reason"] = ("AOI 報點未命中候選；周邊候選僅供參考，不影響此點判斷。"
                              if item["candidate_ids"] and not item["aoi_candidate_ids"] else "")
            item["omit"] = _point_omit_evidence(payload, [c for c in nearby
                if c["id"] in item["aoi_candidate_ids"] and c["id"] not in item["mark_candidate_ids"]])
            item["omit"]["label"] = OMIT_COMPARISON_STATES[item["omit"]["status"]]
            if (omit_covers_point and not item["retained_candidate_ids"]
                    and item["omit"]["status"] in {"suppressed", "partial"}):
                item["status"] = "omit_suppressed"
                item["reason"] = ("AOI 命中貼紙，OMIT 已完整覆蓋暗色本體，該點可轉 OK"
                                  if item["sticker_candidate_ids"] else
                                  "AOI 報點已由側拍 OMIT 屏蔽，該點可轉 OK")
            if point_tapes:
                item["tape_regions"] = [t["id"] for t in point_tapes]
                bomb_overlap = any(c.get("bomb_ids") for c in nearby if c["id"] in item["aoi_candidate_ids"]) or any(
                    len(b.get("side_tolerance_polygon", [])) >= 3 and cv2.pointPolygonTest(
                        np.asarray(b["side_tolerance_polygon"], np.float32), tuple(map(float, side_xy)), False) >= 0
                    for b in payload.get("bombs", []))
                if not item["retained_candidate_ids"] and not bomb_overlap:
                    item["status"] = "tape_suppressed"
                    item["reason"] = "AOI 點命中已辨識的右下角易撕貼範圍，該點可轉 OK（側拍建議）"
                else:
                    if bomb_overlap:
                        item["status"] = "unverified"
                    item["reason"] = ("AOI 點位於易撕貼，但涉及炸彈驗證範圍，保留檢查" if bomb_overlap else
                                      "AOI 報點仍命中易撕貼範圍外的保留候選，保留檢查")
            if roi_marks and (point_marks or item["status"] not in {"omit_suppressed", "tape_suppressed"}):
                item["mark_regions"] = [m["name"] for m in roi_marks]
                item["mark_match_basis"] = "point" if point_marks else "roi"
                bomb_overlap = any(c.get("bomb_ids") for c in nearby if c["id"] in item["aoi_candidate_ids"]) or any(
                    len(b.get("side_tolerance_polygon", [])) >= 3 and (
                        cv2.pointPolygonTest(np.asarray(b["side_tolerance_polygon"], np.float32),
                                             tuple(map(float, side_xy)), False) >= 0)
                    for b in payload.get("bombs", []))
                if not item["retained_candidate_ids"] and not bomb_overlap:
                    item["status"] = "mark_suppressed"
                    item["reason"] = ("AOI 點位於已確認的 MARK 範圍，該點可轉 OK（側拍建議）" if point_marks else
                                      "AOI ROI 內的 MARK 已屏蔽，報點未命中其他保留候選，該點可轉 OK（側拍建議）")
                else:
                    if bomb_overlap:
                        item["status"] = "unverified"
                    item["reason"] = ("AOI ROI 內有 MARK，但涉及炸彈驗證範圍，保留檢查" if bomb_overlap else
                                      "AOI ROI 內的 MARK 已屏蔽，但報點仍命中 MARK 範圍外的保留候選，保留檢查")
            if row.get("status") not in {"CANDIDATES", "NO_CANDIDATES", "FILTERED"} or payload.get("truncated"):
                item["status"] = "unverified"
                item["reason"] = "檢測未完成或候選資料曾截斷，僅供位置對照"
            result["mapped"] += 1
            result["matched"] += item["status"] == "matched"
            result["mark_suppressed"] += item["status"] == "mark_suppressed"
            result["omit_suppressed"] += item["status"] == "omit_suppressed"
            result["tape_suppressed"] += item["status"] == "tape_suppressed"
        except (ValueError, TypeError, KeyError, cv2.error, np.linalg.LinAlgError) as exc:
            item["reason"] = str(exc)
    return result


def retained_candidate_contours(payload, inference_marks=()):
    """Return only unsuppressed side-pixel contours for yellow annotations."""
    masks = [m["side_polygon"] for m in _report_mark_regions(payload, inference_marks)]
    masks.extend(t["side_polygon"] for t in peel_tape_regions(payload))
    retained = {}
    for candidate in payload.get("candidates", []):
        if (candidate.get("disposition", "retained") != "retained" or candidate.get("bomb_ids")
                or omit_sticker_candidate(payload, candidate)):
            continue
        contours = candidate.get("remaining_contours", [candidate.get("side_contour", [])])
        contours = [c for c in contours if len(c) >= 3 and not any(
            all(cv2.pointPolygonTest(np.asarray(mask, np.float32), tuple(map(float, xy)), False) >= 0
                for xy in c) for mask in masks)]
        if contours:
            retained[candidate["id"]] = contours
    return retained


def side_white_view(row, detail=None):
    """One presentation for record details, DEBUG, and the statistics list."""
    source = _side_record_source(row, detail)
    payload = source.get("payload") or {}
    comparison = source.get("comparison") or compare_side_record(
        source, payload.get("parameters", {}).get("csjbl_preview_radius_product_px", 50))
    points = comparison["points"]
    candidates = payload.get("candidates", [])
    nearby_ids = {i for point in points for i in point["candidate_ids"]}
    yellow_nearby_ids = {i for point in points for i in point["retained_candidate_ids"]}
    aoi_ids = {i for point in points for i in point.get("aoi_candidate_ids", [])}
    retained_ids = set(retained_candidate_contours(payload, source.get("inference_marks", [])))
    yellow_ids = retained_ids & yellow_nearby_ids
    mark_ids = {i for point in points for i in point.get("mark_candidate_ids", [])}
    nearby = [c for c in candidates if c["id"] in aoi_ids] if points else []
    omit = _point_omit_evidence(payload, [c for c in nearby if c["id"] not in mark_ids])
    closed = comparison["mark_suppressed"] + comparison["omit_suppressed"] + comparison["tape_suppressed"]
    statuses = {"CANDIDATES": "保留可疑區域", "FILTERED": "候選已排除", "NO_CANDIDATES": "未發現候選",
                "NO_IMAGE": "無側拍白畫面", "ERROR": "檢測失敗", "NOT_RUN": "尚未檢測"}
    headline, tone = statuses.get(source.get("status"), "尚無結論"), "neutral"
    if source.get("status") in {"CANDIDATES", "FILTERED", "NO_CANDIDATES"} and points:
        if closed == len(points):
            headline, tone = f"AOI {closed} 點可轉 OK", "ok"
        elif closed:
            headline, tone = f"已排除 {closed}／{len(points)} 個 AOI 點", "warn"
        else:
            headline, tone = "AOI 仍有候選需確認" if comparison["matched"] else "AOI 尚無可轉 OK 結論", "warn"

    sticker = {"title": "貼紙／表面異物", "tone": "neutral", "text": "無相關候選", "reason": ""}
    if comparison["omit_suppressed"]:
        sticker.update(tone="ok", text=f"已排除 · {comparison['omit_suppressed']} 點可轉 OK",
                       reason="AOI 對應的貼紙／表面異物已由 OMIT 證據確認。")
    elif omit["status"] == "suppressed":
        sticker.update(tone="ok", text="已排除表面異物", reason="其他 AOI 點仍依各自證據判斷。")
    elif omit["status"] == "partial":
        sticker.update(tone="warn", text="部分屏蔽 · 仍有候選", reason="本體未完整覆蓋，或報點仍命中保留候選。")
    elif omit["status"] == "suspected":
        sticker.update(tone="warn", text="已辨識 · 僅標示", reason="此筆保存的是僅標示結果；重新側拍推論後套用貼紙／灰塵過濾。")
    elif omit["status"] in {"unavailable", "disabled"}:
        sticker.update(text="無法比對" if omit["status"] == "unavailable" else "未啟用", reason=omit["reason"])
    elif omit["status"] == "clear":
        sticker.update(tone="warn", text="未符合排除條件", reason="OMIT 未完整覆蓋貼紙本體或未達重疊門檻。")
    if comparison["incomplete"] or source.get("status") in {"ERROR", "NO_IMAGE", "NOT_RUN"}:
        sticker.update(tone="warn", text="檢測未完成", reason="需完成檢測後確認處理結果。")

    marks = _report_mark_regions(payload, source.get("inference_marks", []))
    mark = {"title": "MARK", "tone": "muted", "text": "未取得可用範圍", "reason": "本次沒有可套用的 MARK 屏蔽證據。"}
    if comparison["mark_suppressed"]:
        mark.update(tone="ok", text=f"已屏蔽 · {comparison['mark_suppressed']} 點可轉 OK",
                    reason="AOI 點或 ROI 內已確認的 MARK 已屏蔽，報點未命中其他保留候選。")
    elif marks:
        label = f"MARK 屏蔽範圍 {len(marks)} 區"
        mark.update(text=f"{label}，AOI 未命中",
                    reason="AOI 點及 ROI 未涵蓋已確認的 MARK 範圍。")
        hit_points = [p for p in points if p.get("mark_regions")]
        if hit_points:
            mark.update(tone="warn", text=f"{label}，AOI 命中但未排除" if any(
                        p.get("mark_match_basis") == "point" for p in hit_points) else f"{label}，ROI 有 MARK 但仍需保留",
                        reason=next((p["reason"] for p in hit_points if p.get("reason")), "尚無可轉 OK 結論。"))
        elif not points:
            mark.update(text=f"{label}，無 AOI 報點")
        elif any(p["status"] == "unavailable" for p in points):
            mark.update(tone="warn", text=f"{label}，AOI 命中狀態未確認",
                        reason="AOI 位置無法映射，無法確認是否命中 MARK。")

    tapes = peel_tape_regions(payload)
    tape = {"title": "易撕貼", "tone": "neutral", "text": "未辨識到易撕貼",
            "reason": "尚無右下角易撕貼範圍證據。"}
    if comparison["tape_suppressed"]:
        tape.update(tone="ok", text=f"已排除 · {comparison['tape_suppressed']} 點可轉 OK",
                    reason="AOI 命中已辨識的易撕貼範圍。")
    elif tapes:
        tape.update(tone="ok", text=f"已套用屏蔽 · {len(tapes)} 區", reason="僅屏蔽已辨識的易撕貼範圍。")
        if any(p.get("tape_regions") for p in points):
            tape.update(tone="warn", text="已辨識 · 仍需檢查", reason="附近仍有其他候選或驗證尚未完成。")
    if comparison["incomplete"] or source.get("status") in {"ERROR", "NO_IMAGE", "NOT_RUN"}:
        tape.update(tone="warn", text="檢測未完成", reason="需完成檢測後確認易撕貼處理結果。")

    labels = {"retained": "保留檢查", "excluded": "不檢測區域", "mark_suppressed": "MARK 已屏蔽",
              "dust_suppressed": "表面異物已排除"}
    candidate_rows = []
    tape_ids = {i for point in points for i in point.get("tape_candidate_ids", [])}
    for candidate in candidates:
        dust = candidate.get("dust", {})
        sticker_match = omit_sticker_candidate(payload, candidate)
        candidate_rows.append({"id": candidate["id"], "label": "MARK 已屏蔽" if candidate["id"] in mark_ids else
            "易撕貼已排除" if candidate["id"] in tape_ids else
            "貼紙已排除" if sticker_match else
            ("報點無保留輪廓 · 其餘僅供參考" if candidate["id"] in aoi_ids else "僅供參考 · AOI 未命中")
                if candidate.get("disposition", "retained") == "retained"
                and candidate["id"] not in yellow_ids and not candidate.get("bomb_ids") else
            labels.get(candidate.get("disposition"), "舊版未記錄"),
            "core_overlap_ratio": dust.get("core_overlap_ratio"), "overlap_ratio": dust.get("overlap_ratio"),
            "composite_key": candidate.get("composite_key"), "nearby": candidate["id"] in nearby_ids,
            "aoi_hit": candidate["id"] in aoi_ids,
            "area_px": candidate.get("area_px"), "remaining_area_px": candidate.get("remaining_area_px")})
    legacy_preview = bool(payload.get("artifacts", {}).get("side")
                          and not payload.get("artifacts", {}).get("side_clean")
                          and not payload.get("preview_annotation_version"))
    missing_bounds = bool(payload.get("artifacts", {}).get("side")
                          and not payload.get("preview_bounds", {}).get("side"))
    conversion = source.get("conversion") or record_conversion(source, comparison)
    return {"headline": headline, "tone": tone, "sticker": sticker, "mark": mark, "tape": tape,
            "legend": SIDE_ANNOTATION_LEGEND, "legacy_preview": legacy_preview,
            "missing_preview_bounds": missing_bounds,
            "comparison": comparison, "candidates": candidate_rows,
            "yellow_candidates": {"nearby": len(yellow_ids & yellow_nearby_ids), "total": len(yellow_ids)},
            "reference_candidates": {"nearby": len((retained_ids - yellow_ids) & nearby_ids),
                                     "total": len(retained_ids - yellow_ids)},
            "conversion": conversion, "panel": panel_projection(source, comparison, conversion)}


def draw_side_annotations(image, payload, comparison, kind, bounds, *, inference_marks=(),
                          all_candidates=False, full=True):
    """Draw in saved-preview pixels, after resizing; never alter evidence or coordinates."""
    coordinate_kind = "front" if kind == "front" else "side"
    height, width = image.shape[:2]
    scale = np.array([width / bounds[2], height / bounds[3]])
    origin = np.array(bounds[:2])
    unit = min(1., max(.75, width / 1800))
    thickness = max(1, round(unit))
    font_scale = min(.6, max(.35, width / 2400))
    labels, markers = [], []

    def preview_points(points):
        points = np.asarray(points, np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            return None
        return np.rint((points - origin) * scale).astype(np.int32)

    def polygon(points, key, label=None):
        points = preview_points(points)
        if points is None or len(points) < 3:
            return
        if np.any(points.max(axis=0) < 0) or np.any(points.min(axis=0) >= [width, height]):
            return
        color = annotation_color(key)
        cv2.polylines(image, [points], True, color, max(2, thickness) if key == "candidate" else thickness, cv2.LINE_AA)
        if label:
            anchor = np.clip(points.min(axis=0), [0, 0], [width - 1, height - 1])
            labels.append((anchor, label, color))

    def box_points(box):
        x, y, w, h = box
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    if full:
        if coordinate_kind == "side":
            for exclusion in payload.get("exclusions", []):
                tape = exclusion.get("type") == "corner_tape"
                polygon(box_points(exclusion["bbox"]), "tape" if tape else "mark")
        for zone in payload.get("configured_zones", []):
            points = (box_points(zone["front_bbox"]) if coordinate_kind == "front"
                      else zone["side_polygon"])
            polygon(points, "zone")
        for mark in mapped_inference_marks(payload, inference_marks):
            polygon(mark[coordinate_kind + "_polygon"], "mark")
        for bomb in payload.get("bombs", []):
            points = bomb.get("side_tolerance_polygon", [])
            if len(points) < 3:
                continue
            if coordinate_kind == "front":
                points = _transform(points, np.asarray(payload["mapping"]["matrix"], np.float64))
            polygon(points, "bomb", f"B{bomb['id']}")
        selected = {i for point in comparison["points"] for i in point["candidate_ids"]}
        aoi_selected = {i for point in comparison["points"] for i in point["retained_candidate_ids"]}
        retained = retained_candidate_contours(payload, inference_marks)
        for candidate in payload.get("candidates", []):
            if candidate["id"] not in retained or (not all_candidates and candidate["id"] not in selected):
                continue
            for index, points in enumerate(retained[candidate["id"]]):
                if coordinate_kind == "front":
                    try:
                        matrix = np.asarray(payload.get("mapping", {}).get("matrix"), np.float64)
                        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
                            raise ValueError("沒有可用的正側拍映射")
                        points = _transform(points, matrix)
                    except (ValueError, TypeError, cv2.error):
                        if len(retained[candidate["id"]]) != 1 or not np.array_equal(points, candidate.get("side_contour")):
                            continue
                        points = candidate.get("front_contour")
                if points is not None:
                    polygon(points, "candidate" if candidate["id"] in aoi_selected else "reference",
                            f"#{candidate['id']}" if index == 0 else None)

    for point in comparison["points"]:
        if point["status"] == "unavailable" or coordinate_kind + "_xy" not in point:
            continue
        xy = preview_points([point[coordinate_kind + "_xy"]])[0]
        if np.any(xy < 0) or np.any(xy >= [width, height]):
            continue
        color = annotation_color("aoi")
        # Detailed coordinates and region names belong outside the image.
        labels.insert(0, (xy, f"A{point['id']}", color))
        markers.append(xy)

    occupied = []
    radius = max(3, round(5 * unit))
    protected = [(int(x - radius), int(y - radius), int(x + radius + 1), int(y + radius + 1))
                 for x, y in markers]

    def overlaps(rect, others):
        return any(rect[0] < r[2] and rect[2] > r[0] and rect[1] < r[3] and rect[3] > r[1]
                   for r in others)

    for anchor, text, color in labels:
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness + 1)
        box_w, box_h = text_w + 2, text_h + baseline + 2
        gap = radius + 3
        x, y = map(int, anchor)
        choices = [(x + gap, y - box_h - gap), (x - box_w - gap, y - box_h - gap),
                   (x + gap, y + gap), (x - box_w - gap, y + gap)]
        # Never move a label across the picture or cover an AOI centre to fit it.
        for left, top in choices:
            rect = (left, top, left + box_w, top + box_h)
            if (left >= 0 and top >= 0 and rect[2] < width and rect[3] < height
                    and not overlaps(rect, occupied + protected)):
                break
        else:
            continue
        occupied.append(rect)
        # A thin glyph outline keeps short IDs readable on white without a filled label box.
        cv2.putText(image, text, (rect[0] + 1, rect[1] + 1 + text_h),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (24, 24, 24), thickness + 1, cv2.LINE_AA)
        cv2.putText(image, text, (rect[0] + 1, rect[1] + 1 + text_h),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    for xy in markers:
        color = annotation_color("aoi")
        gap = max(2, round(1.5 * unit))
        for axis in (np.array([1, 0]), np.array([0, 1])):
            for sign in (-1, 1):
                cv2.line(image, tuple(xy + axis * sign * gap), tuple(xy + axis * sign * radius),
                         color, thickness)
    return image


def render_comparison_image(row, comparison, root, kind="side", point_id=None):
    """Annotate saved previews, respecting the crop and resize used by the inspector."""
    if kind not in {"side", "front", "omit"}:
        raise ValueError("不支援的圖片種類")
    coordinate_kind = "side" if kind == "omit" else kind
    payload = row.get("payload") or {}
    artifact = kind
    clean = bool(payload.get("artifacts", {}).get(kind + "_clean"))
    if clean:
        # Rebuild old overview colours from clean pixels, keeping point crops untouched.
        artifact = kind + "_clean"
    stored = payload.get("artifacts", {}).get(artifact)
    bounds = payload.get("preview_bounds", {}).get(artifact)
    if not stored or not bounds or len(bounds) != 4 or min(bounds[2:]) <= 0:
        raise FileNotFoundError("預覽或裁切座標未保存")
    path = Path(stored).resolve()
    if not path.is_relative_to(Path(root).resolve()) or path.suffix.lower() != ".jpg" or not path.is_file():
        raise FileNotFoundError("預覽已過期或遺失")
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("預覽無法讀取")
    scale = np.array([image.shape[1] / bounds[2], image.shape[0] / bounds[3]])
    origin = np.array(bounds[:2])
    points = comparison["points"]
    if point_id is not None:
        points = [p for p in points if p["id"] == point_id]
        if not points or points[0]["status"] == "unavailable":
            raise FileNotFoundError("此 AOI 點沒有可用映射")
    if point_id is None:
        # A legacy marked JPEG cannot be recoloured safely without its clean copy.
        if clean and kind != "omit":
            # Match the inspector's display brightness; evidence crops stay unadjusted.
            level = max(1, float(np.percentile(image, 99)))
            image = np.clip(image.astype(np.float32) * (225 / level), 0, 255).astype(np.uint8)
        return draw_side_annotations(image, payload, comparison, kind, bounds,
                                     inference_marks=row.get("inference_marks", []),
                                     full=clean and kind != "omit")
    if point_id is not None:
        point = points[0]
        roi = (np.array(point[coordinate_kind + "_roi"]) - origin) * scale
        xy = (np.array(point[coordinate_kind + "_xy"]) - origin) * scale
        if np.any(xy < 0) or np.any(xy >= image.shape[1::-1]):
            raise FileNotFoundError("AOI 映射位置不在保存的預覽範圍")
        low = np.maximum(0, np.floor(np.minimum(roi.min(axis=0) - 20, xy - 48))).astype(int)
        high = np.minimum(image.shape[1::-1], np.ceil(np.maximum(roi.max(axis=0) + 20, xy + 48))).astype(int)
        image = image[low[1]:high[1], low[0]:high[0]]
    return image


def crop_original_image(row, comparison, folder, kind="side", point_id=None):
    """Cut the AOI point region straight from the original image file.

    No rotation, resizing, brightness change, colour conversion or annotation:
    the returned array keeps the file's own orientation, bit depth and channels.
    """
    if kind not in {"side", "front", "omit"} or point_id is None:
        raise ValueError("原始圖只提供 AOI 點局部圖")
    payload = row.get("payload") or {}
    coordinate_kind = "side" if kind == "omit" else kind
    name = str((payload.get("omit", {}).get("image") if kind == "omit" else payload.get(kind + "_image")) or "")
    size = payload.get(coordinate_kind + "_size")
    if kind == "omit" and (payload.get("omit", {}).get("coordinate_space") != "side_detection_pixels"
                           or payload["omit"].get("size") != size):
        raise FileNotFoundError("沒有同座標與尺寸的側拍 OMIT")
    if not name or Path(name).name != name or not folder or not size or len(size) != 2:
        raise FileNotFoundError("沒有原始圖檔資訊")
    path = Path(folder) / name
    if not path.is_file():
        raise FileNotFoundError("原始圖檔已不存在")
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError("原始圖檔無法讀取")
    width, height = int(size[0]), int(size[1])
    if image.shape[1] != width or image.shape[0] != height:
        raise FileNotFoundError("原始圖檔尺寸與檢測紀錄不符")
    points = [p for p in comparison["points"] if p["id"] == point_id]
    if not points or points[0]["status"] == "unavailable":
        raise FileNotFoundError("此 AOI 點沒有可用映射")
    point = points[0]
    # Same crop rule as the preview crop, in detection (possibly rotated) pixels.
    roi = np.array(point[coordinate_kind + "_roi"], np.float64)
    xy = np.array(point[coordinate_kind + "_xy"], np.float64)
    if np.any(xy < 0) or xy[0] >= width or xy[1] >= height:
        raise FileNotFoundError("AOI 映射位置不在原始圖範圍")
    low = np.maximum(0, np.floor(np.minimum(roi.min(axis=0) - 20, xy - 48))).astype(int)
    high = np.minimum([width, height], np.ceil(np.maximum(roi.max(axis=0) + 20, xy + 48))).astype(int)
    if payload.get("rotation_applied"):
        # Detection pixels were rotated 180 degrees; map the box back to file pixels.
        low, high = np.array([width, height]) - high, np.array([width, height]) - low
    return image[low[1]:high[1], low[0]:high[0]].copy()
