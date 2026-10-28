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

CONVERSION_FILTERS = {
    "mark": "含 MARK 可轉 OK 點", "omit": "含 OMIT 可轉 OK 點", "both": "同時含兩類可轉 OK 點",
    "either": "含任一類可轉 OK 點", "eligible": "兩類合併可整筆轉 OK",
    "eligible_mark": "僅靠 MARK 可整筆轉 OK", "eligible_omit": "僅靠 OMIT 可整筆轉 OK",
    "blocked": "有可轉 OK 點，但整筆仍需保留",
}


def record_conversion(row, comparison):
    """A point suggestion only removes a rejection when every formal NG is covered."""
    points = comparison["points"]
    mark = sum(p["status"] == "mark_suppressed" for p in points)
    omit = sum(p["status"] == "omit_suppressed" for p in points)
    result = {"mark_points": mark, "omit_points": omit, "eligible": False,
              "category": "mixed" if mark and omit else "mark" if mark else "omit" if omit else "none",
              "state": "none", "reason": "沒有 MARK／OMIT 可轉 OK 點"}
    if not (mark or omit):
        return result
    judgment = str(row.get("ai_judgment") or "")
    result["state"] = "blocked"
    if judgment in {"OK", "OK-i"}:
        result.update(state="already_ok", reason="正式結果已是 OK，不重複計算排片減少")
    elif not judgment.startswith("NG"):
        result["reason"] = "正式結果不是有效 NG，不納入排片減少"
    elif mark + omit != len(points):
        result["reason"] = "仍有 CSJBL 點未符合 MARK／OMIT 轉 OK 條件"
    elif row.get("review_decision") in {"confirmed", "missed", "mapping_error", "uncertain"}:
        result["reason"] = "人工驗證仍有缺陷、定位疑慮或尚無法確認，暫不計入整筆轉 OK"
    else:
        from capi_image_naming import canonical_image_prefix
        details = _json(row.get("ng_details"), None)
        covered = isinstance(details, list) and bool(details)
        for defect in details if isinstance(details, list) else []:
            if (not isinstance(defect, dict) or defect.get("type") != "aoi_passthrough"
                    or str(defect.get("defect_code", "")).upper() != "CSJBL"):
                covered = False
                break
            try:
                xy = np.asarray([defect["product_x"], defect["product_y"]], np.float64)
                prefix = canonical_image_prefix(str(defect.get("image", ""))).upper()
                if not prefix or not np.isfinite(xy).all() or not any(
                        canonical_image_prefix(p["image_prefix"]).upper() == prefix
                        and np.allclose(xy, p["product_xy"], atol=.001, rtol=0) for p in points):
                    covered = False
                    break
            except (ValueError, TypeError, KeyError):
                covered = False
                break
        if covered:
            result.update(eligible=True, state="eligible", reason="全部 CSJBL 點及正式 NG 原因均由這兩類涵蓋")
        else:
            result["reason"] = "另有正式 NG 原因，或 NG 明細不足以確認全部已涵蓋"
    return result


def matches_conversion_filter(result, value):
    return not value or {
        "mark": result["mark_points"] > 0, "omit": result["omit_points"] > 0,
        "both": result["category"] == "mixed", "either": result["category"] != "none",
        "eligible": result["eligible"], "eligible_mark": result["eligible"] and result["category"] == "mark",
        "eligible_omit": result["eligible"] and result["category"] == "omit", "blocked": result["state"] == "blocked",
    }.get(value, False)


def conversion_rate_projection(counts, eligible):
    """Fixed denominator; mixed MARK/OMIT records are removed only once, together."""
    denominator = counts["ok"] + counts["ng"]
    percentage = lambda value: round(100 * value / denominator, 6) if denominator else None
    scenarios = {}
    for key, removed in (("mark", eligible["mark"]), ("omit", eligible["omit"]),
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
            "reason": dust.get("reason", "此候選未保存 OMIT 證據")})
    states = {c["status"] for c in evidence["candidates"]}
    if states - {"clear", "suspected", "suppressed"}:
        evidence["reason"] = "部分候選未完成 OMIT 比對，保留人工確認"
    elif any(c.get("dust", {}).get("status") == "suppressed" and c.get("disposition") == "retained" for c in relevant):
        evidence.update(status="partial", reason="已移除 OMIT 重疊像素，範圍外仍有異常候選")
    elif "suspected" in states:
        evidence.update(status="suspected", reason="OMIT 表面特徵達重疊門檻；目前僅標示，候選保留")
    elif "suppressed" in states:
        evidence.update(status="suppressed", reason="候選已依 OMIT 表面證據屏蔽；仍需確認其他 AOI 點")
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


def compare_side_record(row, tolerance=50):
    """Map front product coordinates to the saved side-camera detection orientation.

    Matching is proximity to candidate contours in product pixels, not a statement
    that AOI or the detector is correct. Filtered and bomb candidates stay distinct.
    """
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 500:
        raise ValueError("比對容許誤差須為 0～500 產品 px")
    points, source = csjbl_points(row.get("aoi_machine_coords"), row.get("ng_details"))
    payload = row.get("payload") or {}
    result = {"has_csjbl": bool(points), "source": source, "points": [],
              "tolerance": tolerance, "matched": 0, "mapped": 0, "mark_suppressed": 0, "omit_suppressed": 0,
              "incomplete": bool(payload.get("truncated")), "reason": ""}
    if not points:
        return result
    marks = mapped_inference_marks(payload, row.get("inference_marks", []))
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
        item = {"id": index, "image_prefix": point.get("image_prefix", ""),
                "source_xy": [point.get("source_x", point.get("product_x")),
                              point.get("source_y", point.get("product_y"))],
                "coordinate_space": point.get("coordinate_space", "product"),
                "status": "unavailable", "reason": result["reason"],
                "candidate_ids": [], "retained_candidate_ids": [],
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
                x, y, w, h = mark["front_bbox"]
                if x <= front_xy[0] < x + w and y <= front_xy[1] < y + h:
                    point_marks.append(mark)
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
                nearby.append(candidate)
                if (distance >= 0 and candidate.get("disposition") == "dust_suppressed"
                        and candidate.get("dust", {}).get("status") == "suppressed" and not candidate.get("bomb_ids")
                        and _inside_omit_regions(side_xy, candidate["dust"].get("suppressed_regions", []))):
                    omit_covers_point = True
                if candidate.get("disposition", "retained") == "retained" and not candidate.get("bomb_ids"):
                    remaining = candidate.get("remaining_contours", [contour])
                    # Old results may still contain a MARK candidate. Only remove
                    # whole contours contained in a confirmed MARK, never neighbours.
                    remaining = [c for c in remaining if not (point_marks and any(
                        all(cv2.pointPolygonTest(np.asarray(m["side_polygon"], np.float32),
                                                tuple(map(float, xy)), False) >= 0 for xy in c)
                        for m in point_marks))]
                    if any(len(c) >= 3 and cv2.pointPolygonTest(_transform(c, to_product),
                           tuple(map(float, product_xy)), True) >= -tolerance for c in remaining):
                        item["retained_candidate_ids"].append(candidate["id"])
            item["status"] = ("matched" if item["retained_candidate_ids"] else
                              "filtered" if item["candidate_ids"] else "unmatched")
            item["reason"] = ""
            item["omit"] = _point_omit_evidence(payload, nearby)
            item["omit"]["label"] = OMIT_COMPARISON_STATES[item["omit"]["status"]]
            if (omit_covers_point and not item["retained_candidate_ids"]
                    and item["omit"]["status"] == "suppressed"):
                item["status"] = "omit_suppressed"
                item["reason"] = "AOI 點對應候選已由側拍 OMIT 屏蔽，該點可轉 OK（側拍建議，非貼紙種類確認）"
            if point_marks:
                item["mark_regions"] = [m["name"] for m in point_marks]
                if not item["retained_candidate_ids"]:
                    item["status"] = "mark_suppressed"
                    item["reason"] = "AOI 點位於正式推論 MARK 範圍，該點可轉 OK（側拍建議）"
                else:
                    item["reason"] = "AOI 點位於 MARK，但附近仍有範圍外候選，需人工確認"
            if row.get("status") not in {"CANDIDATES", "NO_CANDIDATES", "FILTERED"} or payload.get("truncated"):
                item["status"] = "unverified"
                item["reason"] = "檢測未完成或候選資料曾截斷，僅供位置對照"
            result["mapped"] += 1
            result["matched"] += item["status"] == "matched"
            result["mark_suppressed"] += item["status"] == "mark_suppressed"
            result["omit_suppressed"] += item["status"] == "omit_suppressed"
        except (ValueError, TypeError, KeyError, cv2.error, np.linalg.LinAlgError) as exc:
            item["reason"] = str(exc)
    return result


def render_comparison_image(row, comparison, root, kind="side", point_id=None):
    """Annotate saved previews, respecting the crop and resize used by the inspector."""
    if kind not in {"side", "front", "omit"}:
        raise ValueError("不支援的圖片種類")
    coordinate_kind = "side" if kind == "omit" else kind
    payload = row.get("payload") or {}
    artifact = kind
    if point_id is not None and payload.get("artifacts", {}).get(kind + "_clean"):
        # Point crops use the unannotated preview; older records fall back to the marked one.
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
    for point in points:
        # The overview locates AOI points; the crop must leave weak defects visible.
        if point["status"] == "unavailable" or point_id is not None:
            continue
        xy = np.rint((np.array(point[coordinate_kind + "_xy"]) - origin) * scale).astype(int)
        roi = np.rint((np.array(point[coordinate_kind + "_roi"]) - origin) * scale).astype(np.int32)
        cv2.polylines(image, [roi], True, (255, 255, 0), 2)
        cv2.drawMarker(image, tuple(xy), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(image, f"A{point['id']}", tuple(xy + [10, -10]),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 0), 2)
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
