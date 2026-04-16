"""
驗證：移除 fg_median 填充後，邊緣漏光真缺陷是否仍能偵測

兩張圖 × 各自 AOI 點，三路 diff 對比：
  OLD   — fg_median 填充 → medianBlur → diff 前景外歸零
  NEW   — 直接 medianBlur → diff 前景外歸零
  DEBUG — 直接 medianBlur → 不歸零

輸出:
  reports/edge_leak_verify/<image>_<point>_compare.png  六宮格
  reports/edge_leak_verify/summary.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np

ROOT = Path(r"C:/Users/rh.syu/Desktop/CAPI01_AD")
OUT_DIR = ROOT / "reports" / "edge_leak_verify"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 參數 ---
TILE_SIZE = 512
BLUR_KERNEL = 3
MEDIAN_KERNEL = 65
AOI_THRESHOLD = 4
AOI_MIN_AREA = 10
BOUNDARY_PADDING = 15
BOUNDARY_MIN_BRIGHTNESS = 15

DEFAULT_PRODUCT_RESOLUTION = (1920, 1080)

# --- 測試案例 ---
TEST_CASES = [
    {
        "label": "edge_leak",
        "image": ROOT / "test_images" / "WGF50500_074519.tif",
        "points": [
            {"id": "A", "x": 243, "y": 1080},
            {"id": "B", "x": 250, "y": 1080},
        ],
    },
    {
        "label": "L_shape_false",
        "image": ROOT / "test_images" / "WGF50500_034149.tif",
        "points": [
            {"id": "#6", "x": 38, "y": 1076},
            {"id": "#7", "x": 78, "y": 1075},
        ],
    },
]


# ─── CV helpers (copied from diag_edge_no_fg_fill.py) ────────

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
    for c in contours:
        if cv2.contourArea(c) > 1000:
            x, y, w, h = cv2.boundingRect(c)
            x_min = min(x_min, x); y_min = min(y_min, y)
            x_max = max(x_max, x + w); y_max = max(y_max, y + h)
    if x_min == np.inf:
        return (0, 0, img_w, img_h), closing
    return (int(x_min), int(y_min), int(x_max), int(y_max)), closing


def find_panel_polygon(binary_mask: np.ndarray, bbox, tile_size: int = TILE_SIZE):
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


def expand_fg_mask(fg_mask: np.ndarray, gray: np.ndarray,
                   padding: int, min_brightness: int) -> np.ndarray:
    if padding <= 0 or not np.any(fg_mask > 0):
        return fg_mask
    dil = cv2.getStructuringElement(
        cv2.MORPH_RECT, (padding * 2 + 1, padding * 2 + 1)
    )
    fg_expanded = cv2.dilate(fg_mask, dil, iterations=1)
    zone = (fg_expanded > 0) & (fg_mask == 0)
    valid = zone & (gray >= min_brightness)
    out = fg_mask.copy()
    out[valid] = 255
    return out


def _detect_defects(diff: np.ndarray, threshold: int, min_area: int,
                    offset_x: int, offset_y: int) -> List[Dict]:
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    defs = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
            comp_mask = (labels == i)
            mdiff = int(np.max(diff[comp_mask])) if np.any(comp_mask) else 0
            defs.append({
                "area": area,
                "local_bbox": (x, y, w, h),
                "bbox": (offset_x + x, offset_y + y, w, h),
                "max_diff": mdiff,
            })
    return defs


# ─── 三路徑 ──────────────────────────────────────

def run_old(roi_gray, fg_mask, mk, threshold, min_area, ox, oy):
    blurred = cv2.GaussianBlur(roi_gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
    if np.any(fg_mask > 0):
        fg_pix = blurred[fg_mask > 0]
        fg_med = int(np.median(fg_pix)) if fg_pix.size > 0 else 0
        blurred_bg = blurred.copy()
        blurred_bg[fg_mask == 0] = fg_med
    else:
        blurred_bg = blurred
    bg = cv2.medianBlur(blurred_bg, mk)
    diff = cv2.absdiff(blurred, bg)
    diff[fg_mask == 0] = 0
    return diff, _detect_defects(diff, threshold, min_area, ox, oy)


def run_new(roi_gray, fg_mask, mk, threshold, min_area, ox, oy):
    blurred = cv2.GaussianBlur(roi_gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
    bg = cv2.medianBlur(blurred, mk)
    diff = cv2.absdiff(blurred, bg)
    diff[fg_mask == 0] = 0
    return diff, _detect_defects(diff, threshold, min_area, ox, oy)


def run_debug_like(roi_gray, mk, threshold, min_area, ox, oy):
    blurred = cv2.GaussianBlur(roi_gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
    bg = cv2.medianBlur(blurred, mk)
    diff = cv2.absdiff(blurred, bg)
    return diff, _detect_defects(diff, threshold, min_area, ox, oy)


# ─── 可視化 ──────────────────────────────────────

def load_image_8bit(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def diff_to_jet(diff: np.ndarray, gain: float = 10.0) -> np.ndarray:
    d = np.clip(diff.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_JET)


def draw_defects(img, defs, color=(0, 0, 255)):
    out = img.copy()
    for d in defs:
        x, y, w, h = d["local_bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, f"A={d['area']} m={d['max_diff']}",
                    (x, max(12, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


def put_title(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def make_six_panel(roi_gray, fg_mask, diff_old, diff_new, diff_dbg,
                   defs_old, defs_new, defs_dbg, title, out_path):
    H, W = roi_gray.shape[:2]
    roi_bgr = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    fg_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

    p1 = put_title(roi_bgr, "ROI gray")
    p2 = put_title(fg_bgr, "fg_mask (expanded)")

    d_old = draw_defects(diff_to_jet(diff_old), defs_old, (255, 255, 255))
    p3 = put_title(d_old, f"OLD diff  defects={len(defs_old)}")

    d_new = draw_defects(diff_to_jet(diff_new), defs_new, (255, 255, 255))
    p4 = put_title(d_new, f"NEW diff  defects={len(defs_new)}")

    d_dbg = draw_defects(diff_to_jet(diff_dbg), defs_dbg, (255, 255, 255))
    p5 = put_title(d_dbg, f"DEBUG-like diff  defects={len(defs_dbg)}")

    hi = roi_bgr.copy()
    for d in defs_old:
        x, y, w, h = d["local_bbox"]
        cv2.rectangle(hi, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for d in defs_new:
        x, y, w, h = d["local_bbox"]
        cv2.rectangle(hi, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(hi, "OLD=red NEW=green", (6, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    p6 = put_title(hi, "OLD vs NEW bbox on ROI")

    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    grid = np.vstack([row1, row2])
    banner = np.zeros((30, grid.shape[1], 3), np.uint8)
    cv2.putText(banner, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    grid = np.vstack([banner, grid])
    cv2.imwrite(str(out_path), grid)


# ─── Main ────────────────────────────────────────

def main():
    summary: List[str] = []
    summary.append("=" * 80)
    summary.append("Edge Leak Verify: fg_median removal impact")
    summary.append(f"params: blur={BLUR_KERNEL} median={MEDIAN_KERNEL} "
                   f"thr={AOI_THRESHOLD} min_area={AOI_MIN_AREA} "
                   f"pad={BOUNDARY_PADDING} min_bright={BOUNDARY_MIN_BRIGHTNESS}")
    summary.append("=" * 80)
    summary.append("")

    # Table header
    table: List[str] = []
    table.append(f"{'Image':<20} {'Point':<14} {'Method':<8} {'n_def':>5} "
                 f"{'max_diff':>8} {'max_area':>8} {'NG?':>4}")
    table.append("-" * 72)

    for case in TEST_CASES:
        img_path = case["image"]
        label = case["label"]
        print(f"\n{'='*60}")
        print(f"Loading {img_path.name} ({label})")

        gray = load_image_8bit(img_path)
        H, W = gray.shape[:2]
        print(f"  Image: {W}x{H}")

        raw_bounds, binary_mask = find_raw_bounds(gray)
        print(f"  raw_bounds = {raw_bounds}")

        polygon = find_panel_polygon(binary_mask, raw_bounds)
        print(f"  polygon: {'OK' if polygon is not None else 'FAILED'}")

        summary.append(f"Image: {img_path.name}  {W}x{H}  label={label}")
        summary.append(f"  raw_bounds: {raw_bounds}")
        summary.append(f"  polygon: {'OK' if polygon is not None else 'FAILED'}")
        summary.append("")

        for pt in case["points"]:
            cx, cy = map_aoi_coords(pt["x"], pt["y"], raw_bounds)
            x1 = max(0, cx - TILE_SIZE // 2)
            y1 = max(0, cy - TILE_SIZE // 2)
            x2 = min(W, x1 + TILE_SIZE)
            y2 = min(H, y1 + TILE_SIZE)
            x1 = max(0, x2 - TILE_SIZE)
            y1 = max(0, y2 - TILE_SIZE)
            roi = gray[y1:y2, x1:x2].copy()
            ox, oy = x1, y1
            rh, rw = roi.shape[:2]

            # fg_mask: polygon or rect fallback
            fg_mask = np.zeros((rh, rw), np.uint8)
            if polygon is not None:
                loc = polygon.copy()
                loc[:, 0] -= ox
                loc[:, 1] -= oy
                cv2.fillPoly(fg_mask, [loc.astype(np.int32)], 255)
            else:
                rx1, ry1, rx2, ry2 = raw_bounds
                lx1 = max(0, rx1 - ox); ly1 = max(0, ry1 - oy)
                lx2 = min(rw, rx2 - ox); ly2 = min(rh, ry2 - oy)
                if lx2 > lx1 and ly2 > ly1:
                    fg_mask[ly1:ly2, lx1:lx2] = 255

            fg_mask = expand_fg_mask(fg_mask, roi, BOUNDARY_PADDING, BOUNDARY_MIN_BRIGHTNESS)
            mk = clamp_median_kernel(MEDIAN_KERNEL, min(rh, rw) - 1)

            diff_old, defs_old = run_old(roi, fg_mask, mk, AOI_THRESHOLD, AOI_MIN_AREA, ox, oy)
            diff_new, defs_new = run_new(roi, fg_mask, mk, AOI_THRESHOLD, AOI_MIN_AREA, ox, oy)
            diff_dbg, defs_dbg = run_debug_like(roi, mk, AOI_THRESHOLD, AOI_MIN_AREA, ox, oy)

            def _stats(defs):
                n = len(defs)
                md = max((d["max_diff"] for d in defs), default=0)
                ma = max((d["area"] for d in defs), default=0)
                ng = n > 0
                return n, md, ma, ng

            n_o, md_o, ma_o, ng_o = _stats(defs_old)
            n_n, md_n, ma_n, ng_n = _stats(defs_new)
            n_d, md_d, ma_d, ng_d = _stats(defs_dbg)

            pt_label = f"{pt['id']}({pt['x']},{pt['y']})"
            img_short = img_path.stem

            for method, n, md, ma, ng in [
                ("OLD", n_o, md_o, ma_o, ng_o),
                ("NEW", n_n, md_n, ma_n, ng_n),
                ("DEBUG", n_d, md_d, ma_d, ng_d),
            ]:
                table.append(f"{img_short:<20} {pt_label:<14} {method:<8} {n:>5} "
                             f"{md:>8} {ma:>8} {'Y' if ng else 'N':>4}")

            line = (f"  Point {pt_label} img=({cx},{cy}) ROI={rw}x{rh}@({ox},{oy})\n"
                    f"    OLD   : n={n_o:>3} max_diff={md_o:>3} max_area={ma_o:>6} NG={ng_o}\n"
                    f"    NEW   : n={n_n:>3} max_diff={md_n:>3} max_area={ma_n:>6} NG={ng_n}\n"
                    f"    DEBUG : n={n_d:>3} max_diff={md_d:>3} max_area={ma_d:>6} NG={ng_d}")
            print(line)
            summary.append(line)
            summary.append("")

            out_name = f"{img_short}_{pt['id']}_compare.png"
            out_path = OUT_DIR / out_name
            title = (f"{img_short} {pt_label}  "
                     f"OLD:{n_o}def  NEW:{n_n}def  DBG:{n_d}def")
            make_six_panel(roi, fg_mask, diff_old, diff_new, diff_dbg,
                           defs_old, defs_new, defs_dbg, title, out_path)
            print(f"    -> {out_path}")

    summary.append("")
    summary.append("=" * 80)
    summary.append("Table:")
    summary.extend(table)

    out_txt = OUT_DIR / "summary.txt"
    out_txt.write_text("\n".join(summary), encoding="utf-8")
    print(f"\nSummary saved -> {out_txt}")


if __name__ == "__main__":
    main()
