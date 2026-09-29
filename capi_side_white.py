"""CPU-only side-white observations; never produces a formal verdict."""

from pathlib import Path
import re
import time
import copy
from typing import Optional

import cv2
import numpy as np

from capi_config import normalize_side_white_params

ALGORITHM = "side-white-cv-v2"
_SIDE_NAME = re.compile(r"^(.*?)SW0F00000(_?\d{6})?$", re.IGNORECASE)
_IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def find_side_white_pair(folder: Path):
    """Select the newest white side shot and its exact acquisition-name partner."""
    files = [p for p in Path(folder).iterdir()
             if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS]
    sides = [p for p in files if _SIDE_NAME.fullmatch(p.stem)]
    if not sides:
        return None, None
    side = max(sides, key=lambda p: (p.stat().st_mtime_ns, p.name))
    match = _SIDE_NAME.fullmatch(side.stem)
    front_stem = (match[1] + "W0F00000" + (match[2] or "")).upper()
    fronts = [p for p in files if p.stem.upper() == front_stem]
    front = max(fronts, key=lambda p: (p.stat().st_mtime_ns, p.name)) if fronts else None
    return side, front


def detect_panel_quad(gray: np.ndarray) -> np.ndarray:
    """Fit the four bright-panel boundaries, returning TL/TR/BR/BL pixels."""
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    if contour is None or cv2.contourArea(contour) < gray.size * .05:
        raise ValueError("找不到完整白畫面面板")
    coarse = cv2.approxPolyDP(contour, .015 * cv2.arcLength(contour, True), True).reshape(-1, 2)
    if len(coarse) != 4 or not cv2.isContourConvex(coarse):
        raise ValueError("面板邊界不是完整四邊形")
    top, bottom = np.array_split(coarse[np.argsort(coarse[:, 1])], 2)
    quad = np.array([top[np.argmin(top[:, 0])], top[np.argmax(top[:, 0])],
                     bottom[np.argmax(bottom[:, 0])], bottom[np.argmin(bottom[:, 0])]], np.float32)
    lines = []
    radius = max(8, min(65, min(gray.shape) * .025))
    offsets = np.linspace(-radius, radius, int(radius * 4) + 1)
    for index in range(4):
        start, end = quad[index], quad[(index + 1) % 4]
        direction = end - start
        direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])
        centers = start + (end - start) * np.linspace(.07, .93, 400)[:, None]
        samples = centers[:, None, :] + offsets[None, :, None] * normal
        profiles = cv2.remap(blurred.astype(np.float32), samples[:, :, 0].astype(np.float32),
                             samples[:, :, 1].astype(np.float32), cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        gradient = np.abs(np.gradient(profiles, axis=1))
        peaks = gradient[:, 4:-4].argmax(axis=1) + 4
        edge = samples[np.arange(len(samples)), peaks].astype(np.float32)
        vx, vy, x, y = cv2.fitLine(edge, cv2.DIST_HUBER, 0, .01, .01).ravel()
        line = np.array([-vy, vx, vy * x - vx * y])
        if np.percentile(np.abs(edge @ line[:2] + line[2]), 95) > max(3, radius * .1):
            raise ValueError("面板邊線不穩定，無法可靠定位")
        lines.append(line)
    intersections = [np.cross(lines[(i - 1) % 4], lines[i]) for i in range(4)]
    if any(abs(point[2]) < 1e-6 for point in intersections):
        raise ValueError("面板邊線無法相交")
    quad = np.array([point[:2] / point[2] for point in intersections], np.float32)
    if (not np.isfinite(quad).all() or not cv2.isContourConvex(quad)
            or np.any(quad < -2) or np.any(quad[:, 0] > gray.shape[1] + 1)
            or np.any(quad[:, 1] > gray.shape[0] + 1)):
        raise ValueError("面板邊界超出影像或形狀無效")
    return quad


def _candidates(gray, quad, params, *, limit=100, force_mask=None):
    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
    # Only remove the immediate transition; defects near the border remain eligible.
    kernel_size = 2 * params["edge_margin_px"] + 1
    valid = cv2.erode(mask, np.ones((kernel_size, kernel_size), np.uint8)) > 0
    x, y, w, h = cv2.boundingRect(quad)
    from capi_mark_detector import detect_panel_mark
    mark = detect_panel_mark(gray[y:y + h, x:x + w], include_debug=False)
    exclusions = []
    if mark.get("found"):
        box = mark["bbox"]
        mx, my, mw, mh = [int(box[k]) for k in ("x", "y", "width", "height")]
        mx, my = mx + x, my + y
        valid[max(0, my - 5):my + mh + 5, max(0, mx - 5):mx + mw + 5] = False
        exclusions.append({"type": "mark", "bbox": [mx, my, mw, mh]})

    source = gray.astype(np.float32)
    smooth = cv2.GaussianBlur(source, (0, 0), 2)
    foreground = (mask > 0).astype(np.float32)
    background = cv2.GaussianBlur(source * foreground, (0, 0), 24) / np.maximum(
        cv2.GaussianBlur(foreground, (0, 0), 24), 1e-6)
    residual = smooth - background
    # Estimate each edge's common illumination profile in panel coordinates.
    # Only the background correction is warped; detection retains source pixels.
    width = max(2, int(np.linalg.norm(quad[1] - quad[0])))
    height = max(2, int(np.linalg.norm(quad[3] - quad[0])))
    rect = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    to_rect = cv2.getPerspectiveTransform(quad, rect)
    flat = cv2.warpPerspective(residual, to_rect, (width, height))
    correction = np.zeros_like(flat)
    band = min(80, height // 5, width // 5)
    for row in list(range(band)) + list(range(height - band, height)):
        correction[row, :] = np.median(flat[row, width // 10:width * 9 // 10])
    adjusted = flat - correction
    for col in list(range(band)) + list(range(width - band, width)):
        correction[:, col] += np.median(adjusted[height // 10:height * 9 // 10, col])
    residual -= cv2.warpPerspective(correction, np.linalg.inv(to_rect), (gray.shape[1], gray.shape[0]))
    # High-contrast printing / tape in the two conventional corner pairs.
    # Exclude bounded components, never an entire corner ROI or edge strip.
    dark = ((-residual > 7) & valid).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 15), np.uint8))
    nc, _, boxes, centers = cv2.connectedComponentsWithStats(dark)
    uv = cv2.perspectiveTransform(centers.astype(np.float32).reshape(-1, 1, 2), to_rect).reshape(-1, 2)
    for i in range(1, nc):
        bx, by, bw, bh, area = [int(v) for v in boxes[i]]
        u, v = uv[i] / [width, height]
        mark_corner = (u < .18 and v > .65) or (u > .82 and v < .35)
        tape_corner = (u > .88 and v > .92) or (u < .12 and v < .08)
        if (area >= 50 and 20 <= bw <= width * .12 and 10 <= bh <= height * .1
                and bw > bh * 1.2 and (mark_corner or tape_corner)):
            padding = 12 if mark_corner else 5
            valid[max(0, by - padding):by + bh + padding, max(0, bx - padding):bx + bw + padding] = False
            exclusions.append({"type": "corner_print" if mark_corner else "corner_tape",
                               "bbox": [bx, by, bw, bh]})
    if force_mask is not None:
        valid |= (force_mask > 0) & (mask > 0)
    # Robust noise estimate prevents ordinary texture from becoming thousands of candidates.
    values = residual[valid]
    if not values.size:
        raise ValueError("白畫面沒有有效檢測區域")
    noise = float(np.median(np.abs(values - np.median(values))) * 1.4826)
    threshold = max(params["min_contrast_gray"], noise * params["noise_sigma_factor"])
    binary = ((np.abs(residual) > threshold) & valid).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    binary[~valid] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    found = []
    for index in range(1, count):
        bx, by, bw, bh, area = [int(v) for v in stats[index]]
        forced = force_mask is not None and np.any(force_mask[by:by + bh, bx:bx + bw][labels[by:by + bh, bx:bx + bw] == index])
        if area < params["min_area_px"] or (area > mask.size * .008 and not forced):
            continue
        region = labels[by:by + bh, bx:bx + bw] == index
        local = residual[by:by + bh, bx:bx + bw]
        weights = np.abs(local) * region
        yy, xx = np.indices(weights.shape)
        center = [float(bx + (xx * weights).sum() / weights.sum()),
                  float(by + (yy * weights).sum() / weights.sum())]
        context = residual[max(0, by - 16):by + bh + 16, max(0, bx - 16):bx + bw + 16]
        positive, negative = float(context.max()), float(-context.min())
        kind = "bright_dark_pair" if min(positive, negative) > threshold else (
            "bright_spot" if float(local[region].mean()) > 0 else "dark_spot")
        if max(bw, bh) / max(1, min(bw, bh)) >= 4:
            kind = "line"
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        contour = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2) + [bx, by]
        found.append({"kind": kind, "side_xy": center, "side_bbox": [bx, by, bw, bh],
                      "side_contour": contour.tolist(), "area_px": area,
                      "contrast_gray": round(float(np.abs(local[region]).max()), 2)})
        if limit is None:
            found[-1]["_mask"] = region.astype(np.uint8)
    found.sort(key=lambda item: -item["contrast_gray"])
    truncated = limit is not None and len(found) > limit
    if limit is not None:
        found = found[:limit]
    for index, item in enumerate(found, 1):
        item["id"] = index
    return found, residual, exclusions, threshold, truncated


def _preview(gray, quad):
    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
    level = max(1, float(np.percentile(gray[mask > 0], 99)))
    return cv2.cvtColor(np.clip(gray.astype(np.float32) * (225 / level), 0, 255).astype(np.uint8),
                        cv2.COLOR_GRAY2BGR)


def _save_preview(path, image):
    scale = min(1, 1800 / max(image.shape[:2]))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OSError("無法產生側拍檢測預覽")
    encoded.tofile(str(path))
    return str(path.resolve())


def snapshot_context(config, edge_config, parsed):
    """Copy mutable per-product settings before scheduling background work."""
    from capi_inference import resolve_product_resolution
    model_id = parsed.get("model_id", "")
    zones, warning = [], ""
    if edge_config is not None:
        grouped = getattr(edge_config, "all_exclude_zones_by_product", {})
        if grouped:
            code = model_id[5].upper() if len(model_id) > 5 else ""
            zones = copy.deepcopy(grouped.get(code, []))
            if code not in grouped:
                warning = "此產品沒有對應的不檢測區域設定，未套用其他產品區域"
        else:
            zones = [vars(z).copy() for z in getattr(edge_config, "exclude_zones", [])]
    bombs = ([copy.deepcopy(parsed["bomb_info"])] if parsed.get("bomb_info") is not None else
             [b.to_dict() for b in getattr(config, "bomb_defects", [])])
    return {"config": copy.deepcopy(config), "zones": zones, "zone_warning": warning,
            "bombs": bombs, "bomb_source": "client" if parsed.get("bomb_info") is not None else "config",
            "zone_coordinate_space": "front_detection_pixels", "bomb_coordinate_space": "product_detection_coordinates",
            "product_resolution": list(resolve_product_resolution(model_id, getattr(config, "model_resolution_map", None)))}


def load_omit_evidence(folder, side_path, context, rotate_180):
    """Select an exact acquisition or unique OMIT; ambiguous files fail open."""
    from capi_inference import CAPIInferencer
    from capi_station_adapter import create_station_adapter
    config = context.get("config")
    if config is None:
        return None, None, {"status": "unavailable", "reason": "沒有 OMIT 偵測設定快照"}
    detector = CAPIInferencer.__new__(CAPIInferencer)
    detector.config = config
    detector.station_adapter = create_station_adapter(context.get("station_profile", "capi"))
    files = sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS)
    matches = [p for p in files if detector.station_adapter.is_omit_image(p.name)]
    acquisition = _SIDE_NAME.fullmatch(side_path.stem)[2] or ""
    exact = [p for p in matches if acquisition and p.stem.endswith(acquisition)]
    choices = exact or matches
    if len(choices) != 1:
        return None, None, {"status": "unavailable", "reason": "缺少 OMIT 或同次拍攝的 OMIT 無法唯一配對"}
    path = choices[0]
    raw = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None, None, {"status": "unavailable", "image": path.name, "reason": "OMIT 無法讀取"}
    if rotate_180:
        raw = cv2.rotate(raw, cv2.ROTATE_180)
    overexposed, _, _, reason = detector.check_omit_overexposure(
        raw, image_files=files, product_resolution=context.get("product_resolution"))
    info = {"status": "unavailable" if overexposed else "available", "image": path.name,
            "size": [raw.shape[1], raw.shape[0]], "reason": reason if overexposed else "沿用正拍／OMIT 同相機像素對齊；側拍位置為面板邊界估算"}
    return raw, None if overexposed else detector.check_dust_or_scratch_feature, info


def _transform(points, matrix):
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), matrix).reshape(-1, 2)


def _raw_points(points, shape, rotated):
    points = np.asarray(points, np.float32)
    return ((np.array(shape[:2][::-1]) - 1 - points) if rotated else points).tolist()


def _bomb_geometry(context, front_quad, transform, shape, params):
    """Bomb coordinates are product coordinates in the existing detection orientation."""
    bombs, force = [], np.zeros(shape, np.uint8)
    resolution = context.get("product_resolution", [0, 0])
    matrix = None
    if transform is not None and min(resolution) > 0:
        w, h = resolution
        product = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        matrix = np.linalg.inv(transform) @ cv2.getPerspectiveTransform(product, front_quad)
    for definition in context.get("bombs", []):
        coords = definition.get("coordinates", [])
        kind = definition.get("defect_type", "")
        groups = [[p] for p in coords] if kind == "point" else [coords[:2]]
        for points in groups:
            bomb = {"id": len(bombs) + 1, "type": kind, "product_coordinates": points,
                    "image_prefix": definition.get("image_prefix", ""),
                    "defect_code": definition.get("defect_code", ""),
                    "status": "unavailable", "candidate_ids": [], "reason": "無法映射正拍炸彈座標"}
            bombs.append(bomb)
            if not params["bomb_check_enabled"]:
                bomb["reason"] = "炸彈比對未啟用"
                continue
            # Only front-camera definitions can be a source; never interpret side pixels as product coordinates.
            from capi_image_naming import canonical_image_prefix, CANONICAL_IMAGE_PREFIXES
            prefix = canonical_image_prefix(bomb["image_prefix"]).upper()
            if prefix not in CANONICAL_IMAGE_PREFIXES:
                bomb["reason"] = "炸彈來源不是正拍畫面"
                continue
            if matrix is None or kind not in ("point", "line") or len(points) != (1 if kind == "point" else 2):
                continue
            pts = np.asarray(points, np.float32)
            if pts.shape != (len(points), 2) or not np.isfinite(pts).all() or np.any(pts < 0) or np.any(pts > resolution):
                bomb["reason"] = "炸彈產品座標超出範圍"
                continue
            mapped = _transform(pts, matrix)
            tolerance = params["bomb_tolerance_product_px"]
            offsets = np.array([[-tolerance, -tolerance], [tolerance, -tolerance],
                                [tolerance, tolerance], [-tolerance, tolerance]], np.float32)
            hull = cv2.convexHull(_transform((pts[:, None, :] + offsets).reshape(-1, 2), matrix)).reshape(-1, 2)
            bomb.update(status="missed", reason="側拍未檢出符合位置與形態的候選",
                        side_points=mapped.tolist(), side_tolerance_polygon=hull.tolist())
            if params["bomb_force_detection_enabled"]:
                cv2.fillConvexPoly(force, np.rint(hull).astype(np.int32), 255)
    return bombs, force


def _match_bombs(candidates, bombs):
    for candidate in candidates:
        candidate["bomb_ids"] = []
        contour = np.asarray(candidate["side_contour"], np.float32)
        for bomb in bombs:
            if bomb["status"] == "unavailable":
                continue
            polygon = np.asarray(bomb["side_tolerance_polygon"], np.float32)
            # Use the candidate center (same contract as formal bomb matching).
            # Testing contour vertices misses bombs fully enclosed by a candidate,
            # and lets a long neighbouring region match on a single touching vertex.
            if cv2.pointPolygonTest(polygon, tuple(map(float, candidate["side_xy"])), False) < 0:
                continue
            if bomb["type"] == "line":
                # Position alone must not turn a nearby bright dot into a line bomb.
                rect = cv2.minAreaRect(contour)
                long, short = max(rect[1]), min(rect[1])
                if long / max(1, short) < 3:
                    continue
                _, vectors = cv2.PCACompute(contour, mean=None)
                direction = np.diff(np.asarray(bomb["side_points"]), axis=0)[0]
                if abs(float(vectors[0] @ direction)) / max(1e-6, float(np.linalg.norm(direction))) < .85:
                    continue
            candidate["bomb_ids"].append(bomb["id"])
            bomb["candidate_ids"].append(candidate["id"])
            bomb.update(status="matched", reason="側拍異常證據符合炸彈位置及形態")


def _zone_mask(context, shape, transform, params):
    mask, zones = np.zeros(shape, np.uint8), []
    if transform is None or not params["apply_exclusions"]:
        return mask, zones
    inverse = np.linalg.inv(transform)
    for index, zone in enumerate(context.get("zones", []), 1):
        if not zone.get("enabled", True):
            continue
        x, y, w, h = [float(zone.get(k, 0)) for k in ("x", "y", "w", "h")]
        if w <= 0 or h <= 0:
            continue
        # Existing CV zones use detection-image pixels, already in the configured orientation.
        polygon = _transform([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], inverse)
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
        zones.append({"name": zone.get("name", f"不檢測區域 {index}"), "side_polygon": polygon.tolist(),
                      "front_bbox": [x, y, w, h]})
    return mask, zones


def _crop_bounds(points, shape, padding):
    x, y, w, h = cv2.boundingRect(np.asarray(points, np.float32))
    return max(0, x - padding), max(0, y - padding), min(shape[1], x + w + padding), min(shape[0], y + h + padding)


def _panel(image, bounds, title, contour=None, mask=None, mask_color=(0, 220, 220), remaining_contours=()):
    """Fixed-size letterboxed panel; image pixels and annotations share one transform."""
    canvas = np.full((350, 400, 3), 20, np.uint8)
    cv2.putText(canvas, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (230, 230, 230), 1)
    if image is None or bounds is None:
        cv2.putText(canvas, "Unavailable", (95, 185), cv2.FONT_HERSHEY_SIMPLEX, .7, (150, 150, 150), 1)
        return canvas
    x1, y1, x2, y2 = bounds
    crop = image[y1:y2, x1:x2].copy()
    if not crop.size:
        return canvas
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    if mask is not None:
        colored = crop.copy()
        colored[mask > 0] = mask_color
        crop = cv2.addWeighted(crop, .55, colored, .45, 0)
    if contour is not None:
        cv2.polylines(crop, [np.rint(np.asarray(contour) - [x1, y1]).astype(np.int32)], True, (0, 180, 255), 1)
    for remaining in remaining_contours:
        cv2.polylines(crop, [np.rint(np.asarray(remaining) - [x1, y1]).astype(np.int32)], True, (80, 240, 80), 1)
    scale = min(400 / crop.shape[1], 315 / crop.shape[0])
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))))
    left, top = (400 - crop.shape[1]) // 2, 35 + (315 - crop.shape[0]) // 2
    canvas[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    return canvas


def _candidate_evidence(item, side, front, omit, detector, omit_info, transform, zone_mask, residual, params):
    """Evaluate actual CV support pixels, not the candidate bounding rectangle."""
    x, y, w, h = item["side_bbox"]
    support = item.pop("_mask").astype(bool)
    remaining = support & (zone_mask[y:y + h, x:x + w] == 0)
    item["excluded_area_px"] = int(np.count_nonzero(support & ~remaining))
    item["dust"] = {"status": "unavailable", "reason": omit_info.get("reason", "缺少 OMIT"), "overlap_ratio": None}
    item["crop_bounds"] = {"side": list(_crop_bounds(item["side_contour"], side.shape, params["crop_padding_px"]))}
    mask, front_bounds = None, None
    dust_local = np.zeros((h, w), bool)
    if front is not None and transform is not None:
        front_bounds = _crop_bounds(item["front_contour"], front.shape, params["crop_padding_px"])
        item["crop_bounds"]["front"] = list(front_bounds)
    aligned = omit is not None and front is not None and omit.shape[:2] == front.shape[:2]
    if aligned and front_bounds is not None:
        item["crop_bounds"]["omit"] = list(front_bounds)
    if params["dust_mode"] == "off":
        item["dust"].update(status="disabled", reason="灰塵判斷未啟用")
    elif aligned and detector is not None and front_bounds is not None:
        fx1, fy1, fx2, fy2 = front_bounds
        if fx2 > fx1 and fy2 > fy1:
            try:
                _, mask, _, detail = detector(omit[fy1:fy2, fx1:fx2])
                if mask is None or mask.shape != (fy2 - fy1, fx2 - fx1):
                    raise ValueError("OMIT 遮罩尺寸無效")
                margin = params["mapping_margin_px"]
                comparison = cv2.dilate(mask, np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)) if margin else mask
                yy, xx = np.nonzero(support)
                mapped = _transform(np.column_stack((xx + x, yy + y)), transform)
                mx, my = np.rint(mapped - [fx1, fy1]).astype(int).T
                in_crop = (mx >= 0) & (my >= 0) & (mx < mask.shape[1]) & (my < mask.shape[0])
                if not in_crop.all():
                    raise ValueError("候選映射超出 OMIT 有效範圍")
                dust_local[yy, xx] = comparison[my, mx] > 0
                ratio = float(np.count_nonzero(dust_local & remaining) / max(1, np.count_nonzero(remaining)))
                suspected = ratio >= params["dust_overlap_ratio"]
                item["dust"].update(status="suspected" if suspected else "clear", reason=str(detail),
                                    overlap_ratio=round(ratio, 4), margin_px=margin)
                if suspected and params["dust_mode"] == "suppress":
                    remaining &= ~dust_local
                    item["dust"]["status"] = "suppressed"
            except Exception as exc:
                item["dust"].update(status="unavailable", reason=f"灰塵證據無法產生：{exc}")
    elif omit is not None and not aligned:
        item["dust"]["reason"] = "正拍／OMIT 尺寸不一致或缺少映射，未自動縮放屏蔽"
    count, labels, stats, _ = cv2.connectedComponentsWithStats(remaining.astype(np.uint8))
    accepted = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= params["min_area_px"]]
    remaining = np.isin(labels, accepted) if accepted else np.zeros_like(remaining)
    item["remaining_area_px"] = int(remaining.sum())
    item["remaining_contrast_gray"] = round(float(np.abs(residual[y:y + h, x:x + w][remaining]).max()), 2) if remaining.any() else 0
    contours, _ = cv2.findContours(remaining.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    item["remaining_contours"] = [(c.reshape(-1, 2) + [x, y]).tolist() for c in contours]
    item["disposition"] = ("retained" if remaining.any() else
                           "dust_suppressed" if item["dust"]["status"] == "suppressed" else "excluded")
    sx1, sy1, sx2, sy2 = item["crop_bounds"]["side"]
    panels = [_panel(side, item["crop_bounds"]["side"], f"Side #{item['id']}", item["side_contour"],
                     zone_mask[sy1:sy2, sx1:sx2], (255, 140, 50), item["remaining_contours"]),
              _panel(front, front_bounds, "Front (estimated)", item.get("front_contour")),
              _panel(omit if aligned else None, front_bounds, "OMIT", item.get("front_contour")),
              _panel(omit if aligned and mask is not None else None, front_bounds, "OMIT surface mask", item.get("front_contour"), mask)]
    return np.hstack(panels)


def inspect_side_white_image(side_path: Path, front_path: Optional[Path], output_dir: Path,
                             *, rotate_180: bool = False, params: Optional[dict] = None,
                             context: Optional[dict] = None, omit_image=None, dust_detector=None,
                             omit_info: Optional[dict] = None) -> dict:
    """Detect in camera pixels, then map contours; missing mapping never implies OK."""
    started = time.perf_counter()
    payload = {"algorithm": ALGORITHM, "shadow_only": True, "status": "ERROR",
               "side_image": Path(side_path).name, "front_image": Path(front_path).name if front_path else "",
               "rotation_applied": bool(rotate_180), "coordinate_space": "detection_image_pixels",
               "candidates": [], "mapping": {"status": "unavailable"}, "artifacts": {}}
    context = context or {}
    omit_info = dict(omit_info or {"status": "unavailable", "reason": "缺少 OMIT"})
    payload["evidence_context"] = {k: v for k, v in context.items() if k != "config"}
    payload["omit"] = omit_info
    config = context.get("config")
    payload["dust_parameters"] = {k: v for k, v in vars(config).items() if k.startswith("dust_")} if config else {}
    try:
        params = normalize_side_white_params(params)
        payload["parameters"] = dict(params)
        payload["bombs"], _ = _bomb_geometry(context, None, None, (1, 1), params)
        side = cv2.imdecode(np.fromfile(str(side_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if side is None:
            raise ValueError("側拍圖片無法讀取")
        if rotate_180:
            side = cv2.rotate(side, cv2.ROTATE_180)
        quad = detect_panel_quad(side)
        payload.update(side_size=[side.shape[1], side.shape[0]], side_polygon=quad.tolist())
        front, front_quad, transform = None, None, None
        if front_path:
            try:
                front = cv2.imdecode(np.fromfile(str(front_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if front is None:
                    raise ValueError("正拍圖片無法讀取")
                if rotate_180:
                    front = cv2.rotate(front, cv2.ROTATE_180)
                front_quad = detect_panel_quad(front)
                transform = cv2.getPerspectiveTransform(quad, front_quad)
                payload["front_size"] = [front.shape[1], front.shape[0]]
                payload["mapping"] = {"status": "estimated", "method": "panel_border_homography",
                                      "matrix": transform.tolist(), "front_polygon": front_quad.tolist(),
                                      "reason": "面板邊界估算；尚未做同設備多點校正，無全域精度保證"}
            except (ValueError, OSError, cv2.error) as exc:
                payload["mapping"]["reason"] = str(exc)
        else:
            payload["mapping"]["reason"] = "缺少同次拍攝的正拍白畫面，僅保留側拍座標"
        bombs, force_mask = _bomb_geometry(context, front_quad, transform, side.shape, params)
        candidates, residual, exclusions, threshold, _ = _candidates(
            side, quad, params, limit=None, force_mask=force_mask)
        _match_bombs(candidates, bombs)
        zone_mask, zones = _zone_mask(context, side.shape, transform, params)
        payload.update(candidates=candidates, exclusions=exclusions, configured_zones=zones,
                       bombs=bombs, threshold_gray=round(threshold, 3))
        payload["exclusion_status"] = ("disabled" if not params["apply_exclusions"] else
                                       "unavailable" if transform is None else "applied")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Filtering and bomb matching run on every candidate. Only image generation is capped.
        composites = {}
        for item in candidates:
            point = np.array(item["side_xy"], np.float32)
            item["side_raw_xy"] = ((np.array(side.shape[::-1]) - 1 - point) if rotate_180 else point).tolist()
            item["front_xy"] = None
            item["front_raw_xy"] = None
            if transform is not None:
                mapped = cv2.perspectiveTransform(point.reshape(1, 1, 2), transform)[0, 0]
                item["front_xy"] = mapped.tolist()
                item["front_raw_xy"] = ((np.array(front.shape[::-1]) - 1 - mapped) if rotate_180 else mapped).tolist()
                contour = np.array(item["side_contour"], np.float32).reshape(-1, 1, 2)
                item["front_contour"] = cv2.perspectiveTransform(contour, transform).reshape(-1, 2).tolist()
            support = item["_mask"]
            composite = _candidate_evidence(item, side, front, omit_image, dust_detector, omit_info,
                                            transform, zone_mask, residual, params)
            item["excluded_zone_names"] = []
            if item["excluded_area_px"]:
                bx, by, bw, bh = item["side_bbox"]
                for zone in zones:
                    local_mask = np.zeros((bh, bw), np.uint8)
                    polygon = np.asarray(zone["side_polygon"], np.float32) - [bx, by]
                    cv2.fillConvexPoly(local_mask, np.rint(polygon).astype(np.int32), 255)
                    if np.any((support > 0) & (local_mask > 0)):
                        item["excluded_zone_names"].append(zone["name"])
            if len(composites) < 100:
                key = f"candidate_{item['id']}"
                composites[key] = _save_preview(output_dir / f"{key}.jpg", composite)
                item["composite_key"] = key
        for bomb in bombs:
            matched = [c for c in candidates if bomb["id"] in c["bomb_ids"]]
            bomb["overlaps_exclusion"] = any(c["excluded_area_px"] > 0 for c in matched)
            bomb["overlaps_dust"] = any((c["dust"].get("overlap_ratio") or 0) > 0 for c in matched)
            if bomb.get("side_points"):
                bomb["side_raw_points"] = _raw_points(bomb["side_points"], side.shape, rotate_180)
                # Report zone conflict even when there is no detected candidate.
                path_mask = np.zeros_like(zone_mask)
                points = np.rint(bomb["side_points"]).astype(int)
                if bomb["type"] == "point":
                    cv2.circle(path_mask, tuple(points[0]), 1, 255, -1)
                else:
                    cv2.line(path_mask, tuple(points[0]), tuple(points[1]), 255, 1)
                bomb["overlaps_exclusion"] |= bool(np.any((path_mask > 0) & (zone_mask > 0)))
        summary = {"raw": len(candidates), "retained": sum(c["disposition"] == "retained" and not c["bomb_ids"] for c in candidates),
                   "excluded": sum(c["disposition"] == "excluded" for c in candidates),
                   "dust_suppressed": sum(c["disposition"] == "dust_suppressed" for c in candidates),
                   "dust_suspected": sum(c["dust"]["status"] == "suspected" for c in candidates),
                   "bomb_candidates": sum(bool(c["bomb_ids"]) for c in candidates),
                   "bomb_matched": sum(b["status"] == "matched" for b in bombs), "bomb_total": len(bombs)}
        payload.update(summary=summary, truncated=False, composites_truncated=len(candidates) > 100,
                       status="CANDIDATES" if summary["retained"] else "FILTERED" if candidates else "NO_CANDIDATES")
        payload["artifacts"].update(composites)
        side_view = _preview(side, quad)
        front_view = _preview(front, front_quad) if transform is not None else None
        residual_view = cv2.cvtColor(np.clip(128 + residual * 10, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for exclusion in exclusions:
            bx, by, bw, bh = exclusion["bbox"]
            cv2.rectangle(side_view, (bx, by), (bx + bw, by + bh), (230, 150, 60), 2)
            cv2.putText(side_view, "MASK", (bx, max(18, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, .6, (230, 150, 60), 2)
        for zone in zones:
            cv2.polylines(side_view, [np.rint(zone["side_polygon"]).astype(np.int32)], True, (255, 150, 50), 3)
        for bomb in bombs:
            if bomb.get("side_points"):
                pts = np.rint(bomb["side_points"]).astype(np.int32)
                cv2.polylines(side_view, [np.rint(bomb["side_tolerance_polygon"]).astype(np.int32)], True, (255, 0, 255), 2)
                cv2.putText(side_view, f"B{bomb['id']}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 255), 2)
        for item in candidates:
            bx, by, bw, bh = item["side_bbox"]
            for view in (side_view, residual_view):
                cv2.rectangle(view, (bx - 5, by - 5), (bx + bw + 5, by + bh + 5), (0, 210, 255), 2)
                cv2.putText(view, str(item["id"]), (bx, max(18, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 210, 255), 2)
            if front_view is not None:
                contour = np.rint(item["front_contour"]).astype(np.int32)
                cv2.polylines(front_view, [contour], True, (0, 210, 255), 3)
                px, py = np.rint(item["front_xy"]).astype(int)
                cv2.putText(front_view, str(item["id"]), (px + 8, py), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 210, 255), 3)
        for name, view in (("side", side_view), ("residual", residual_view), ("front", front_view)):
            if view is not None:
                # Show the panel at a useful size; coordinates remain those of the full image.
                preview_quad = front_quad if name == "front" else quad
                bx, by, bw, bh = cv2.boundingRect(preview_quad)
                left, top = max(0, bx - 20), max(0, by - 20)
                crop = view[top:min(view.shape[0], by + bh + 20), left:min(view.shape[1], bx + bw + 20)]
                payload.setdefault("preview_bounds", {})[name] = [left, top, crop.shape[1], crop.shape[0]]
                payload["artifacts"][name] = _save_preview(output_dir / (name + ".jpg"), crop)
    except (ValueError, OSError, cv2.error) as exc:
        payload["status"] = "ERROR"
        payload["reason"] = str(exc)
    for item in payload["candidates"]:
        item.pop("_mask", None)
        if "side_raw_xy" not in item:
            point = np.asarray(item["side_xy"], np.float32)
            item["side_raw_xy"] = ((np.array(payload["side_size"]) - 1 - point) if rotate_180 else point).tolist()
        item.setdefault("front_raw_xy", None)
    payload["processing_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return payload
