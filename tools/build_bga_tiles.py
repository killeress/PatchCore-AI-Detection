"""Build BGA training tiles from BGA_Train/*.tif.

Pipeline:
- Foreground: Otsu -> largest contour bbox
- MARK exclusion:        template match bga_mark.png (multi-scale)
- Right-bottom exclusion: template match mark2.png  (multi-scale)
- Tile 512x512 non-overlapping, edge flush-back
- Keep only if foreground coverage >= 0.9 AND not intersecting exclusion rects
- Output: <out>/<prefix>/<stem>_tile_<idx>.png
- Debug overlay per source image for verification
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

TILE_SIZE = 512
COVERAGE_MIN = 0.9
BBOX_INSET = 60  # 前景 bbox 四邊內縮，避免 tile 碰到黑邊 / panel 外緣

PREFIX_RULES = [
    ("STANDARD", "STANDARD"),
    ("WGF", "WGF"),
    ("W0F", "W0F"),
    ("G0F", "G0F"),
    ("R0F", "R0F"),
]

TEMPLATE_SCALES = [0.75, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]
MARK_MATCH_THRESHOLD = 0.30
RB_MATCH_THRESHOLD = 0.35

MARK_ROI_X_RATIO = 0.40
MARK_ROI_Y_START_RATIO = 0.65


def detect_prefix(name: str) -> Optional[str]:
    up = name.upper()
    for key, folder in PREFIX_RULES:
        if up.startswith(key):
            return folder
    return None


def match_template_multiscale(
    image: np.ndarray,
    template: np.ndarray,
    scales: List[float],
    threshold: float,
    search_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[int, int, int, int, float]]:
    if search_bbox is not None:
        sx1, sy1, sx2, sy2 = search_bbox
        roi = image[sy1:sy2, sx1:sx2]
    else:
        sx1, sy1 = 0, 0
        roi = image
    roi_h, roi_w = roi.shape[:2]
    best = None
    for s in scales:
        tw = int(template.shape[1] * s)
        th = int(template.shape[0] * s)
        if tw < 8 or th < 8 or tw > roi_w or th > roi_h:
            continue
        tpl = cv2.resize(template, (tw, th))
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if best is None or max_val > best[0]:
            x = max_loc[0] + sx1
            y = max_loc[1] + sy1
            best = (max_val, x, y, x + tw, y + th)
    if best is None or best[0] < threshold:
        return None
    return (best[1], best[2], best[3], best[4], best[0])


def generate_tile_positions(lo: int, hi: int, tile: int) -> List[int]:
    if hi - lo < tile:
        return []
    positions = list(range(lo, hi - tile + 1, tile))
    last = hi - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


def rect_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def process_image(
    img_path: Path,
    out_root: Path,
    mark_tpl: np.ndarray,
    rb_tpl: np.ndarray,
    debug_dir: Optional[Path],
) -> dict:
    prefix = detect_prefix(img_path.stem)
    if prefix is None:
        return {"file": img_path.name, "status": "skip_no_prefix"}

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"file": img_path.name, "status": "read_fail"}

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"file": img_path.name, "status": "no_foreground"}
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    fg_x1, fg_y1, fg_x2, fg_y2 = x + BBOX_INSET, y + BBOX_INSET, x + w - BBOX_INSET, y + h - BBOX_INSET

    fg_w = fg_x2 - fg_x1
    fg_h = fg_y2 - fg_y1
    mark_roi = (
        fg_x1,
        fg_y1 + int(fg_h * MARK_ROI_Y_START_RATIO),
        fg_x1 + int(fg_w * MARK_ROI_X_RATIO),
        fg_y2,
    )
    mark = match_template_multiscale(img, mark_tpl, TEMPLATE_SCALES, MARK_MATCH_THRESHOLD, mark_roi)
    rb = match_template_multiscale(img, rb_tpl, TEMPLATE_SCALES, RB_MATCH_THRESHOLD)

    exclusion_rects: List[Tuple[str, Tuple[int, int, int, int]]] = []
    if mark is not None:
        exclusion_rects.append(("mark", mark[:4]))
    if rb is not None:
        exclusion_rects.append(("rb", rb[:4]))

    xs = generate_tile_positions(fg_x1, fg_x2, TILE_SIZE)
    ys = generate_tile_positions(fg_y1, fg_y2, TILE_SIZE)

    out_dir = out_root / prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    kept_rects: List[Tuple[int, int, int, int]] = []
    dropped_rects: List[Tuple[str, Tuple[int, int, int, int]]] = []
    kept = 0
    dropped_coverage = 0
    dropped_excl = 0
    idx = 0
    for ty in ys:
        for tx in xs:
            tile_rect = (tx, ty, tx + TILE_SIZE, ty + TILE_SIZE)
            if any(rect_intersect(tile_rect, er[1]) for er in exclusion_rects):
                dropped_excl += 1
                dropped_rects.append(("excl", tile_rect))
                continue
            mask_tile = binary[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE]
            coverage = float((mask_tile > 0).sum()) / (TILE_SIZE * TILE_SIZE)
            if coverage < COVERAGE_MIN:
                dropped_coverage += 1
                dropped_rects.append(("cov", tile_rect))
                continue
            tile_img = img[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE]
            out_path = out_dir / f"{img_path.stem}_tile_{idx:03d}.png"
            cv2.imwrite(str(out_path), tile_img)
            kept_rects.append(tile_rect)
            kept += 1
            idx += 1

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(overlay, (fg_x1, fg_y1), (fg_x2, fg_y2), (255, 255, 0), 3)
        for name, (rx1, ry1, rx2, ry2) in exclusion_rects:
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 0, 255), 3)
            cv2.putText(overlay, name, (rx1, max(ry1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        for r in kept_rects:
            cv2.rectangle(overlay, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
        for reason, r in dropped_rects:
            color = (128, 128, 128) if reason == "cov" else (0, 128, 255)
            cv2.rectangle(overlay, (r[0], r[1]), (r[2], r[3]), color, 1)
        cv2.imwrite(str(debug_dir / f"{img_path.stem}_debug.jpg"), overlay)

    return {
        "file": img_path.name,
        "prefix": prefix,
        "fg_bbox": (fg_x1, fg_y1, fg_x2, fg_y2),
        "mark": mark,
        "rb": rb,
        "kept": kept,
        "dropped_coverage": dropped_coverage,
        "dropped_excl": dropped_excl,
        "total_positions": len(xs) * len(ys),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="BGA_Train")
    ap.add_argument("--out", default="dataset_v2/BGA")
    ap.add_argument("--mark-tpl", default="bga_mark.png")
    ap.add_argument("--rb-tpl", default="mark2.png")
    ap.add_argument("--debug-dir", default="dataset_v2/BGA/_debug")
    ap.add_argument("--no-debug", action="store_true")
    ap.add_argument("--only", default=None, help="只處理檔名含此字串的圖片 (驗證用)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    mark_tpl = cv2.imread(args.mark_tpl, cv2.IMREAD_GRAYSCALE)
    rb_tpl = cv2.imread(args.rb_tpl, cv2.IMREAD_GRAYSCALE)
    assert mark_tpl is not None, f"Cannot load {args.mark_tpl}"
    assert rb_tpl is not None, f"Cannot load {args.rb_tpl}"
    debug_dir = None if args.no_debug else Path(args.debug_dir)

    files = sorted(list(src.glob("*.tif")) + list(src.glob("*.png")) + list(src.glob("*.jpg")))
    if args.only:
        files = [f for f in files if args.only in f.name]
    print(f"Found {len(files)} image(s) in {src}")

    summary = []
    for f in files:
        r = process_image(f, out, mark_tpl, rb_tpl, debug_dir)
        summary.append(r)
        mark_flag = "Y" if r.get("mark") else "N"
        rb_flag = "Y" if r.get("rb") else "N"
        mark_score = f"{r['mark'][4]:.2f}" if r.get("mark") else "----"
        rb_score = f"{r['rb'][4]:.2f}" if r.get("rb") else "----"
        print(
            f"  {f.name:42s} prefix={r.get('prefix'):>8s} "
            f"kept={r.get('kept'):>3} drop_cov={r.get('dropped_coverage'):>3} "
            f"drop_excl={r.get('dropped_excl'):>3} "
            f"mark={mark_flag}({mark_score}) rb={rb_flag}({rb_score})"
        )

    total_kept = sum(r.get("kept", 0) or 0 for r in summary)
    print(f"\nTotal tiles saved: {total_kept}")
    print(f"Output dir: {out}")
    if debug_dir:
        print(f"Debug overlays: {debug_dir}")


if __name__ == "__main__":
    main()
