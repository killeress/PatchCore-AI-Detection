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
              "tolerance": tolerance, "matched": 0, "mapped": 0,
              "incomplete": bool(payload.get("truncated")), "reason": ""}
    if not points:
        return result
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
                "candidate_ids": [], "retained_candidate_ids": []}
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
            for candidate in payload.get("candidates", []):
                contour = candidate.get("side_contour")
                if not contour or len(contour) < 3:
                    continue
                product_contour = _transform(contour, to_product)
                distance = cv2.pointPolygonTest(product_contour, tuple(map(float, product_xy)), True)
                if distance < -tolerance:
                    continue
                item["candidate_ids"].append(candidate["id"])
                if candidate.get("disposition", "retained") == "retained" and not candidate.get("bomb_ids"):
                    remaining = candidate.get("remaining_contours", [contour])
                    if any(len(c) >= 3 and cv2.pointPolygonTest(_transform(c, to_product),
                           tuple(map(float, product_xy)), True) >= -tolerance for c in remaining):
                        item["retained_candidate_ids"].append(candidate["id"])
            item["status"] = ("matched" if item["retained_candidate_ids"] else
                              "filtered" if item["candidate_ids"] else "unmatched")
            item["reason"] = ""
            if row.get("status") not in {"CANDIDATES", "NO_CANDIDATES", "FILTERED"} or payload.get("truncated"):
                item["status"] = "unverified"
                item["reason"] = "檢測未完成或候選資料曾截斷，僅供位置對照"
            result["mapped"] += 1
            result["matched"] += item["status"] == "matched"
        except (ValueError, TypeError, KeyError, cv2.error, np.linalg.LinAlgError) as exc:
            item["reason"] = str(exc)
    return result


def render_comparison_image(row, comparison, root, kind="side", point_id=None):
    """Annotate saved previews, respecting the crop and resize used by the inspector."""
    if kind not in {"side", "front"}:
        raise ValueError("不支援的圖片種類")
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
        xy = np.rint((np.array(point[kind + "_xy"]) - origin) * scale).astype(int)
        roi = np.rint((np.array(point[kind + "_roi"]) - origin) * scale).astype(np.int32)
        cv2.polylines(image, [roi], True, (255, 255, 0), 2)
        cv2.drawMarker(image, tuple(xy), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(image, f"A{point['id']}", tuple(xy + [10, -10]),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 0), 2)
    if point_id is not None:
        point = points[0]
        roi = (np.array(point[kind + "_roi"]) - origin) * scale
        xy = (np.array(point[kind + "_xy"]) - origin) * scale
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
    if kind not in {"side", "front"} or point_id is None:
        raise ValueError("原始圖只提供 AOI 點局部圖")
    payload = row.get("payload") or {}
    name = str(payload.get(kind + "_image") or "")
    size = payload.get(kind + "_size")
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
    roi = np.array(point[kind + "_roi"], np.float64)
    xy = np.array(point[kind + "_xy"], np.float64)
    if np.any(xy < 0) or xy[0] >= width or xy[1] >= height:
        raise FileNotFoundError("AOI 映射位置不在原始圖範圍")
    low = np.maximum(0, np.floor(np.minimum(roi.min(axis=0) - 20, xy - 48))).astype(int)
    high = np.minimum([width, height], np.ceil(np.maximum(roi.max(axis=0) + 20, xy + 48))).astype(int)
    if payload.get("rotation_applied"):
        # Detection pixels were rotated 180 degrees; map the box back to file pixels.
        low, high = np.array([width, height]) - high, np.array([width, height]) - low
    return image[low[1]:high[1], low[0]:high[0]].copy()
