"""
實驗：panel_polygon 取代 axis-aligned otsu_bounds 建 fg_mask，
驗證能否消除 AOI 座標 #6 / #7 在左下角歪斜邊緣的過檢。

對 test_images/WGF50500_034149.tif 針對 AOI 座標 (38,1076) / (78,1075) 比較:
  - 舊方法：fg_mask = 矩形填充 (raw_bounds)
  - 新方法：fg_mask = cv2.fillPoly(panel_polygon)

不載入 GPU 模型；直接抽出 _find_raw_bounds + _find_panel_polygon + inspect_roi 的
純 CV 邏輯獨立跑。

輸出:
  reports/edge_polygon_real_vis/overview.png
  reports/edge_polygon_real_vis/aoi_<idx>_quad.png
  reports/edge_polygon_real_vis/summary.txt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

ROOT = Path(r"C:/Users/rh.syu/Desktop/CAPI01_AD")
IMG_PATH = ROOT / "test_images" / "WGF50500_034149.tif"
OUT_DIR = ROOT / "reports" / "edge_polygon_real_vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 參數（對齊 production default） ---
TILE_SIZE = 512
BLUR_KERNEL = 3
MEDIAN_KERNEL = 65
AOI_THRESHOLD = 4
AOI_MIN_AREA = 10
BOUNDARY_PADDING = 15
BOUNDARY_MIN_BRIGHTNESS = 15

# AOI 座標（product resolution space, 1920x1080 預設）
AOI_POINTS = [
    {"idx": 6, "x": 38, "y": 1076, "tag": "LowerLeft"},
    {"idx": 7, "x": 78, "y": 1075, "tag": "LowerLeft"},
]
# 炸彈座標（僅在 overview 標示）
BOMB_POINTS = [
    {"idx": 1, "x": 400, "y": 200},
    {"idx": 2, "x": 800, "y": 400},
    {"idx": 3, "x": 1200, "y": 600},
    {"idx": 4, "x": 1600, "y": 800},
    {"idx": 5, "x": 960, "y": 540},
]

DEFAULT_PRODUCT_RESOLUTION = (1920, 1080)


# ─────────────────────────────────────────────
# 純 CV 邏輯（複製自 capi_inference / capi_edge_cv）
# ─────────────────────────────────────────────

def find_raw_bounds(image: np.ndarray) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    img_h, img_w = image.shape[:2]
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((15, 15), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_min, y_min = np.inf, np.inf
    x_max, y_max = -np.inf, -np.inf
    MIN_AREA = 1000
    for c in contours:
        if cv2.contourArea(c) > MIN_AREA:
            x, y, w, h = cv2.boundingRect(c)
            x_min = min(x_min, x); y_min = min(y_min, y)
            x_max = max(x_max, x + w); y_max = max(y_max, y + h)
    if x_min == np.inf:
        return (0, 0, img_w, img_h), closing
    return (int(x_min), int(y_min), int(x_max), int(y_max)), closing


def find_panel_polygon(binary_mask: np.ndarray, bbox, tile_size: int = TILE_SIZE) -> Optional[np.ndarray]:
    EDGE_MARGIN = 20
    SAMPLE_STEP = 50
    OUTLIER_SIGMA = 3.0
    MIN_EDGE_LEN_RATIO = 1.0
    MIN_POLYGON_AREA_RATIO = 0.9
    MIN_SAMPLES_PER_EDGE = 5

    if binary_mask is None or binary_mask.size == 0:
        return None
    H, W = binary_mask.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    if xmax - xmin < 2 * EDGE_MARGIN or ymax - ymin < 2 * EDGE_MARGIN:
        return None

    tops, bots = [], []
    for x in range(xmin + EDGE_MARGIN, xmax - EDGE_MARGIN, SAMPLE_STEP):
        if x < 0 or x >= W:
            continue
        col = binary_mask[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) > 0:
            tops.append((x, int(ys[0])))
            bots.append((x, int(ys[-1])))
    lefts, rights = [], []
    for y in range(ymin + EDGE_MARGIN, ymax - EDGE_MARGIN, SAMPLE_STEP):
        if y < 0 or y >= H:
            continue
        row = binary_mask[y, :]
        xs = np.where(row > 0)[0]
        if len(xs) > 0:
            lefts.append((int(xs[0]), y))
            rights.append((int(xs[-1]), y))

    if (len(tops) < MIN_SAMPLES_PER_EDGE or len(bots) < MIN_SAMPLES_PER_EDGE
            or len(lefts) < MIN_SAMPLES_PER_EDGE or len(rights) < MIN_SAMPLES_PER_EDGE):
        return None

    def fit_line_robust(pts, horizontal: bool):
        arr = np.array(pts, dtype=float)
        if horizontal:
            ind, dep = arr[:, 0], arr[:, 1]
        else:
            ind, dep = arr[:, 1], arr[:, 0]
        try:
            a, b = np.polyfit(ind, dep, 1)
        except Exception:
            return None
        res = dep - (a * ind + b)
        s = float(res.std())
        if s > 0:
            keep = np.abs(res) < OUTLIER_SIGMA * s
            if keep.sum() >= 3:
                try:
                    a, b = np.polyfit(ind[keep], dep[keep], 1)
                except Exception:
                    pass
        return float(a), float(b)

    tl_ = fit_line_robust(tops, True)
    bl_ = fit_line_robust(bots, True)
    ll_ = fit_line_robust(lefts, False)
    rl_ = fit_line_robust(rights, False)
    if None in (tl_, bl_, ll_, rl_):
        return None

    def intersect(h, v):
        a_h, b_h = h; a_v, b_v = v
        denom = 1.0 - a_h * a_v
        if abs(denom) < 1e-9:
            return None
        y = (a_h * b_v + b_h) / denom
        x = a_v * y + b_v
        return (x, y)

    TL = intersect(tl_, ll_); TR = intersect(tl_, rl_)
    BR = intersect(bl_, rl_); BL = intersect(bl_, ll_)
    if None in (TL, TR, BR, BL):
        return None

    polygon = np.array([TL, TR, BR, BL], dtype=np.float32)
    tol = 50
    if (polygon[:, 0].min() < -tol or polygon[:, 0].max() > W + tol or
            polygon[:, 1].min() < -tol or polygon[:, 1].max() > H + tol):
        return None
    min_edge = tile_size * MIN_EDGE_LEN_RATIO
    for i in range(4):
        if float(np.linalg.norm(polygon[(i + 1) % 4] - polygon[i])) < min_edge:
            return None
    bbox_area = float((xmax - xmin) * (ymax - ymin))
    poly_area = float(cv2.contourArea(polygon))
    if bbox_area <= 0 or poly_area < bbox_area * MIN_POLYGON_AREA_RATIO:
        return None
    return polygon


def map_aoi_coords(px, py, raw_bounds, product_res=DEFAULT_PRODUCT_RESOLUTION):
    PW, PH = product_res
    x1, y1, x2, y2 = raw_bounds
    sx = (x2 - x1) / PW
    sy = (y2 - y1) / PH
    return int(px * sx + x1), int(py * sy + y1)


def clamp_median_kernel(k: int, max_dim: int) -> int:
    mk = min(k, max_dim)
    if mk % 2 == 0:
        mk -= 1
    return max(3, mk)


def inspect_roi_with_mask(
    roi: np.ndarray,
    offset_x: int,
    offset_y: int,
    fg_mask: Optional[np.ndarray],
    *,
    boundary_padding: int = BOUNDARY_PADDING,
    boundary_min_brightness: int = BOUNDARY_MIN_BRIGHTNESS,
    threshold: int = AOI_THRESHOLD,
    min_area: int = AOI_MIN_AREA,
):
    """複製 capi_edge_cv.inspect_roi + _inspect_side 的核心邏輯。
    直接吃現成 fg_mask（可為 rect 或 polygon）。"""
    if roi is None or roi.size == 0:
        return [], {"max_diff": 0, "max_area": 0}, None, None

    gray = roi if roi.ndim == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_h, roi_w = gray.shape[:2]

    if fg_mask is not None:
        # 邊界擴展
        if boundary_padding > 0 and np.any(fg_mask > 0):
            dil = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (boundary_padding * 2 + 1, boundary_padding * 2 + 1),
            )
            fg_expanded = cv2.dilate(fg_mask, dil, iterations=1)
            zone = (fg_expanded > 0) & (fg_mask == 0)
            valid = zone & (gray >= boundary_min_brightness)
            fg_mask = fg_mask.copy()
            fg_mask[valid] = 255

    blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
    if fg_mask is not None and np.any(fg_mask > 0):
        fg_pix = blurred[fg_mask > 0]
        fg_med = int(np.median(fg_pix)) if fg_pix.size > 0 else 0
        blurred_bg = blurred.copy()
        blurred_bg[fg_mask == 0] = fg_med
    else:
        blurred_bg = blurred
    mk = clamp_median_kernel(MEDIAN_KERNEL, min(gray.shape[:2]) - 1)
    bg = cv2.medianBlur(blurred_bg, mk)
    diff = cv2.absdiff(blurred, bg)
    if fg_mask is not None:
        diff[fg_mask == 0] = 0

    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    defects = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[i]
            comp_mask = (labels == i)
            mdiff = int(np.max(diff[comp_mask])) if np.any(comp_mask) else 0
            defects.append({
                "area": area,
                "bbox": (offset_x + x, offset_y + y, w, h),
                "center": (int(offset_x + cx), int(offset_y + cy)),
                "max_diff": mdiff,
                "local_bbox": (x, y, w, h),
            })

    stats_out = {
        "max_diff": int(np.max(diff)) if diff.size > 0 else 0,
        "max_area": max((d["area"] for d in defects), default=0),
        "n_defects": len(defects),
        "fg_coverage_pct": float(np.count_nonzero(fg_mask) / fg_mask.size * 100) if fg_mask is not None else 100.0,
    }
    return defects, stats_out, diff, fg_mask


# ─────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────

def load_image_8bit(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def draw_overview(gray: np.ndarray, raw_bounds, polygon, aoi_mapped, bomb_mapped, out_path: Path):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # raw_bounds (blue)
    x1, y1, x2, y2 = raw_bounds
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 4)
    # polygon (red)
    if polygon is not None:
        pts = polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=4)
        for i, p in enumerate(polygon):
            cv2.circle(vis, (int(p[0]), int(p[1])), 15, (0, 0, 255), -1)
            cv2.putText(vis, ["TL", "TR", "BR", "BL"][i], (int(p[0]) + 20, int(p[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    # bomb (yellow)
    for b in bomb_mapped:
        cv2.circle(vis, (b["img_x"], b["img_y"]), 12, (0, 255, 255), -1)
        cv2.putText(vis, f"B{b['idx']}", (b["img_x"] + 15, b["img_y"]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    # AOI test points (green)
    for a in aoi_mapped:
        cv2.circle(vis, (a["img_x"], a["img_y"]), 18, (0, 255, 0), 3)
        cv2.putText(vis, f"#{a['idx']} raw({a['raw_x']},{a['raw_y']})->({a['img_x']},{a['img_y']})",
                    (a["img_x"] + 20, a["img_y"] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # scale down for output
    h, w = vis.shape[:2]
    scale = 2400 / max(h, w)
    if scale < 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), vis)


def draw_quad_panel(roi: np.ndarray, fg_old, fg_new, diff_old, diff_new,
                    defects_old, defects_new, title: str, out_path: Path):
    """四宮格: ROI / old mask overlay / new mask overlay / old vs new defect highlight"""
    if roi.ndim == 2:
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    else:
        roi_bgr = roi.copy()
    H, W = roi_bgr.shape[:2]

    def overlay_mask(base, mask, color):
        out = base.copy()
        if mask is not None:
            colored = np.zeros_like(out); colored[:] = color
            alpha = 0.35
            m3 = (mask > 0).astype(np.uint8)[:, :, None]
            out = np.where(m3 > 0, (out * (1 - alpha) + colored * alpha).astype(np.uint8), out)
        return out

    p1 = roi_bgr.copy()
    cv2.putText(p1, "ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    p2 = overlay_mask(roi_bgr, fg_old, (255, 0, 0))  # blue = rect mask
    cv2.putText(p2, f"OLD mask (rect) defects={len(defects_old)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 2)
    for d in defects_old:
        x, y, w, h = d["local_bbox"]
        cv2.rectangle(p2, (x, y), (x + w, y + h), (0, 0, 255), 2)

    p3 = overlay_mask(roi_bgr, fg_new, (0, 0, 255))  # red = poly mask
    cv2.putText(p3, f"NEW mask (poly) defects={len(defects_new)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
    for d in defects_new:
        x, y, w, h = d["local_bbox"]
        cv2.rectangle(p3, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # diff comparison side-by-side normalised
    def norm(d):
        if d is None:
            return np.zeros((H, W), np.uint8)
        mx = float(d.max()) if d.size > 0 else 1.0
        mx = max(mx, 1.0)
        return np.clip(d.astype(np.float32) / mx * 255.0, 0, 255).astype(np.uint8)

    dn_old = norm(diff_old); dn_new = norm(diff_new)
    dc_old = cv2.applyColorMap(dn_old, cv2.COLORMAP_JET)
    dc_new = cv2.applyColorMap(dn_new, cv2.COLORMAP_JET)
    half_w = W // 2
    combined = np.hstack([
        cv2.resize(dc_old, (half_w, H)),
        cv2.resize(dc_new, (W - half_w, H)),
    ])
    cv2.putText(combined, "diff OLD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "diff NEW", (half_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    p4 = combined

    top = np.hstack([p1, p2])
    bot = np.hstack([p3, p4])
    grid = np.vstack([top, bot])
    cv2.putText(grid, title, (10, grid.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imwrite(str(out_path), grid)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print(f"Loading {IMG_PATH}")
    gray = load_image_8bit(IMG_PATH)
    H, W = gray.shape[:2]
    print(f"Image: {W}x{H}, dtype={gray.dtype}")

    # 1. raw_bounds + binary mask
    raw_bounds, binary_mask = find_raw_bounds(gray)
    print(f"raw_bounds = {raw_bounds}")

    # 2. panel polygon
    polygon = find_panel_polygon(binary_mask, raw_bounds)
    if polygon is None:
        print("WARNING: panel polygon detection FAILED")
    else:
        print(f"polygon corners (TL/TR/BR/BL):")
        for n, p in zip(["TL", "TR", "BR", "BL"], polygon):
            print(f"  {n}: ({p[0]:.1f}, {p[1]:.1f})")

    # 3. 產品解析度：WGF50500 未在 resolution map，用 default 1920x1080；
    #    若 raw_bounds 明顯不是 16:9 直接 fallback 到 raw_bounds 尺寸當產品解析度
    rb_w = raw_bounds[2] - raw_bounds[0]
    rb_h = raw_bounds[3] - raw_bounds[1]
    print(f"raw_bounds size: {rb_w}x{rb_h} (ratio={rb_w/rb_h:.3f})")
    product_res = DEFAULT_PRODUCT_RESOLUTION
    print(f"using product_resolution = {product_res}")

    aoi_mapped = []
    for a in AOI_POINTS:
        ix, iy = map_aoi_coords(a["x"], a["y"], raw_bounds, product_res)
        aoi_mapped.append({"idx": a["idx"], "raw_x": a["x"], "raw_y": a["y"],
                           "img_x": ix, "img_y": iy, "tag": a["tag"]})
        print(f"AOI #{a['idx']} raw=({a['x']},{a['y']}) -> img=({ix},{iy})")

    bomb_mapped = []
    for b in BOMB_POINTS:
        ix, iy = map_aoi_coords(b["x"], b["y"], raw_bounds, product_res)
        bomb_mapped.append({"idx": b["idx"], "img_x": ix, "img_y": iy})

    # 4. overview 圖
    overview_path = OUT_DIR / "overview.png"
    draw_overview(gray, raw_bounds, polygon, aoi_mapped, bomb_mapped, overview_path)
    print(f"Saved overview -> {overview_path}")

    # 5. 逐 AOI 點比較 old vs new
    summary_lines = []
    summary_lines.append(f"Image: {IMG_PATH.name} ({W}x{H})")
    summary_lines.append(f"raw_bounds: {raw_bounds}")
    summary_lines.append(f"panel_polygon: {'OK' if polygon is not None else 'FAILED'}")
    if polygon is not None:
        for n, p in zip(["TL", "TR", "BR", "BL"], polygon):
            summary_lines.append(f"  {n}: ({p[0]:.1f}, {p[1]:.1f})")
    summary_lines.append("")

    for a in aoi_mapped:
        cx, cy = a["img_x"], a["img_y"]
        x1 = max(0, cx - TILE_SIZE // 2); y1 = max(0, cy - TILE_SIZE // 2)
        x2 = min(W, x1 + TILE_SIZE); y2 = min(H, y1 + TILE_SIZE)
        # 若貼邊，回推 x1/y1 使 ROI 完整
        x1 = max(0, x2 - TILE_SIZE); y1 = max(0, y2 - TILE_SIZE)
        roi = gray[y1:y2, x1:x2].copy()
        offset_x, offset_y = x1, y1
        rh, rw = roi.shape[:2]

        # --- OLD: 矩形 mask ---
        fg_old = np.zeros((rh, rw), np.uint8)
        ox1, oy1, ox2, oy2 = raw_bounds
        lx1 = max(0, ox1 - offset_x); ly1 = max(0, oy1 - offset_y)
        lx2 = min(rw, ox2 - offset_x); ly2 = min(rh, oy2 - offset_y)
        if lx2 > lx1 and ly2 > ly1:
            fg_old[ly1:ly2, lx1:lx2] = 255

        # --- NEW: polygon mask ---
        fg_new = np.zeros((rh, rw), np.uint8)
        if polygon is not None:
            local_poly = polygon.copy()
            local_poly[:, 0] -= offset_x
            local_poly[:, 1] -= offset_y
            cv2.fillPoly(fg_new, [local_poly.astype(np.int32)], 255)
        else:
            fg_new = fg_old.copy()

        d_old, s_old, diff_old, fg_old_used = inspect_roi_with_mask(
            roi, offset_x, offset_y, fg_old)
        d_new, s_new, diff_new, fg_new_used = inspect_roi_with_mask(
            roi, offset_x, offset_y, fg_new)

        ng_old = any(dd["area"] >= AOI_MIN_AREA for dd in d_old)
        ng_new = any(dd["area"] >= AOI_MIN_AREA for dd in d_new)

        line = (f"AOI #{a['idx']} img=({cx},{cy}) ROI={rw}x{rh}@({offset_x},{offset_y}) | "
                f"OLD: n={s_old['n_defects']} max_diff={s_old['max_diff']} max_area={s_old['max_area']} "
                f"fg_cov={s_old['fg_coverage_pct']:.1f}% NG={ng_old} | "
                f"NEW: n={s_new['n_defects']} max_diff={s_new['max_diff']} max_area={s_new['max_area']} "
                f"fg_cov={s_new['fg_coverage_pct']:.1f}% NG={ng_new}")
        print(line)
        summary_lines.append(line)

        quad_path = OUT_DIR / f"aoi_{a['idx']}_quad.png"
        title = f"AOI #{a['idx']} ({a['raw_x']},{a['raw_y']})->img({cx},{cy})  OLD NG={ng_old} NEW NG={ng_new}"
        draw_quad_panel(roi, fg_old_used, fg_new_used, diff_old, diff_new,
                        d_old, d_new, title, quad_path)
        print(f"  saved {quad_path}")

    (OUT_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved summary -> {OUT_DIR/'summary.txt'}")


if __name__ == "__main__":
    main()
