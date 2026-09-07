"""CPU-only side-white inspection for review; never produces a formal verdict."""

from pathlib import Path
import re
import time
from typing import Optional

import cv2
import numpy as np

from capi_config import normalize_side_white_params

ALGORITHM = "side-white-cv-v1"
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


def _candidates(gray, quad, params):
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
        if area < params["min_area_px"] or area > mask.size * .008:
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
    found.sort(key=lambda item: -item["contrast_gray"])
    truncated = len(found) > 100
    found = found[:100]
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


def inspect_side_white_image(side_path: Path, front_path: Optional[Path], output_dir: Path,
                             *, rotate_180: bool = False, params: Optional[dict] = None) -> dict:
    """Detect in camera pixels, then map contours; missing mapping never implies OK."""
    started = time.perf_counter()
    payload = {"algorithm": ALGORITHM, "shadow_only": True, "status": "ERROR",
               "side_image": Path(side_path).name, "front_image": Path(front_path).name if front_path else "",
               "rotation_applied": bool(rotate_180), "coordinate_space": "detection_image_pixels",
               "candidates": [], "mapping": {"status": "unavailable"}, "artifacts": {}}
    try:
        params = normalize_side_white_params(params)
        payload["parameters"] = dict(params)
        side = cv2.imdecode(np.fromfile(str(side_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if side is None:
            raise ValueError("側拍圖片無法讀取")
        if rotate_180:
            side = cv2.rotate(side, cv2.ROTATE_180)
        quad = detect_panel_quad(side)
        candidates, residual, exclusions, threshold, truncated = _candidates(side, quad, params)
        payload.update(side_size=[side.shape[1], side.shape[0]], side_polygon=quad.tolist(),
                       candidates=candidates, exclusions=exclusions, threshold_gray=round(threshold, 3),
                       truncated=truncated, status="CANDIDATES" if candidates else "NO_CANDIDATES")
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
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        side_view = _preview(side, quad)
        front_view = _preview(front, front_quad) if transform is not None else None
        residual_view = cv2.cvtColor(np.clip(128 + residual * 10, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for exclusion in exclusions:
            bx, by, bw, bh = exclusion["bbox"]
            cv2.rectangle(side_view, (bx, by), (bx + bw, by + bh), (230, 150, 60), 2)
            cv2.putText(side_view, "MASK", (bx, max(18, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, .6, (230, 150, 60), 2)
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
    payload["processing_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return payload
