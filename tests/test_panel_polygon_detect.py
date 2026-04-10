"""
Panel polygon 偵測測試 (方案 F 前置驗證)

目的：驗證「用每邊 polyfit 直線相交得到 4 角」能否穩定偵測真實面板邊界，
並與目前 axis-aligned Otsu bbox 做視覺比對。

輸出：
  - test_output/panel_polygon/<image>_full.png   : 整張圖 + bbox (藍) + polygon (紅)
  - test_output/panel_polygon/<image>_corner_*.png : 4 個角的放大比對
  - test_output/panel_polygon/<image>_tilemask.png : 模擬 tile polygon mask 效果

注意：這是 stand-alone 測試，不 import capi_inference，不改任何生產程式。
"""
import cv2
import numpy as np
import os
from pathlib import Path

TEST_IMAGES = [
    r"C:\Users\rh.syu\Desktop\CAPI01_AD\test_images\W0F00000_110022.tif",
    r"C:\Users\rh.syu\Desktop\CAPI01_AD\test_images\G0F00000_151955.tif",
]

OUT_DIR = Path(r"C:\Users\rh.syu\Desktop\CAPI01_AD\test_output\panel_polygon")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def current_otsu_bbox(gray: np.ndarray) -> tuple:
    """複製 _find_raw_object_bounds 的邏輯。"""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((15, 15), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    xmin, ymin = np.inf, np.inf
    xmax, ymax = -np.inf, -np.inf
    for c in contours:
        if cv2.contourArea(c) > 1000:
            x, y, w, h = cv2.boundingRect(c)
            xmin = min(xmin, x)
            ymin = min(ymin, y)
            xmax = max(xmax, x + w)
            ymax = max(ymax, y + h)
    return int(xmin), int(ymin), int(xmax), int(ymax), closing


def find_panel_polygon(gray: np.ndarray, binary_mask: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    每邊 polyfit 直線 → 4 線相交 → 4 角。
    使用 binary_mask (Otsu 後 close 的結果) 做邊緣點掃描。

    Args:
        gray: 原始灰階圖
        binary_mask: Otsu 後 morphology close 的 binary 圖 (255=前景)
        bbox: (x_min, y_min, x_max, y_max) 作為搜尋範圍

    Returns:
        4x2 np.ndarray [TL, TR, BR, BL] (float)
    """
    H, W = gray.shape
    xmin, ymin, xmax, ymax = bbox

    # 在 bbox 範圍內掃描每一條邊
    # 邊緣點定義：binary_mask 上「第一個 255 的位置」
    margin = 20  # 避開角落
    sample_step = 50

    # --- TOP: 每個 x 找最小 y ---
    tops = []
    for x in range(xmin + margin, xmax - margin, sample_step):
        col = binary_mask[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) > 0:
            tops.append((x, int(ys[0])))

    # --- BOTTOM: 每個 x 找最大 y ---
    bots = []
    for x in range(xmin + margin, xmax - margin, sample_step):
        col = binary_mask[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) > 0:
            bots.append((x, int(ys[-1])))

    # --- LEFT: 每個 y 找最小 x ---
    lefts = []
    for y in range(ymin + margin, ymax - margin, sample_step):
        row = binary_mask[y, :]
        xs = np.where(row > 0)[0]
        if len(xs) > 0:
            lefts.append((int(xs[0]), y))

    # --- RIGHT: 每個 y 找最大 x ---
    rights = []
    for y in range(ymin + margin, ymax - margin, sample_step):
        row = binary_mask[y, :]
        xs = np.where(row > 0)[0]
        if len(xs) > 0:
            rights.append((int(xs[-1]), y))

    # 多數點擬合直線 (最小平方) — 對 outlier 做一次 RANSAC-like filter
    def fit_line_robust(pts, horizontal: bool):
        """
        horizontal=True: 回傳 y = a*x + b  → (a, b, 'h')
        horizontal=False: 回傳 x = a*y + b → (a, b, 'v')
        """
        if len(pts) < 3:
            return None
        pts = np.array(pts, dtype=float)
        if horizontal:
            xs, ys = pts[:, 0], pts[:, 1]
            a, b = np.polyfit(xs, ys, 1)
            residuals = ys - (a * xs + b)
        else:
            xs, ys = pts[:, 0], pts[:, 1]
            a, b = np.polyfit(ys, xs, 1)
            residuals = xs - (a * ys + b)
        # 剔除 |residual| > 3*sigma 的點再 fit 一次
        sigma = residuals.std()
        if sigma > 0:
            keep = np.abs(residuals) < 3 * sigma
            if keep.sum() >= 3:
                if horizontal:
                    a, b = np.polyfit(xs[keep], ys[keep], 1)
                else:
                    a, b = np.polyfit(ys[keep], xs[keep], 1)
        return (a, b, 'h' if horizontal else 'v')

    top_line = fit_line_robust(tops, horizontal=True)
    bot_line = fit_line_robust(bots, horizontal=True)
    left_line = fit_line_robust(lefts, horizontal=False)
    right_line = fit_line_robust(rights, horizontal=False)

    def intersect(h_line, v_line):
        # h: y = a_h*x + b_h; v: x = a_v*y + b_v
        a_h, b_h, _ = h_line
        a_v, b_v, _ = v_line
        denom = 1 - a_h * a_v
        if abs(denom) < 1e-9:
            return None
        y = (a_h * b_v + b_h) / denom
        x = a_v * y + b_v
        return (x, y)

    TL = intersect(top_line, left_line)
    TR = intersect(top_line, right_line)
    BR = intersect(bot_line, right_line)
    BL = intersect(bot_line, left_line)

    return np.array([TL, TR, BR, BL], dtype=np.float32)


def draw_visualization(gray: np.ndarray, bbox: tuple, polygon: np.ndarray, label: str) -> np.ndarray:
    """產生可比對的視覺化圖 (下採樣 + bbox + polygon 疊加)。"""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    xmin, ymin, xmax, ymax = bbox

    # 目前 bbox — 藍色
    cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (255, 0, 0), 6)

    # 新的 polygon — 紅色
    poly_int = polygon.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [poly_int], True, (0, 0, 255), 6)

    # 4 個角標記
    for i, name in enumerate(['TL', 'TR', 'BR', 'BL']):
        pt = tuple(polygon[i].astype(int))
        cv2.circle(vis, pt, 20, (0, 255, 255), 4)
        cv2.putText(vis, name, (pt[0] + 30, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)

    cv2.putText(vis, label, (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 6)
    cv2.putText(vis, "BLUE = current Otsu bbox", (50, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 4)
    cv2.putText(vis, "RED = polygon (4 corner fit)", (50, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

    return vis


def crop_corner(vis: np.ndarray, cx: int, cy: int, size: int = 500, label: str = "") -> np.ndarray:
    H, W = vis.shape[:2]
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(W, cx + size // 2)
    y2 = min(H, cy + size // 2)
    crop = vis[y1:y2, x1:x2].copy()
    # 標上 label
    cv2.putText(crop, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return crop


def simulate_tile_mask(gray: np.ndarray, polygon: np.ndarray, bbox: tuple,
                       tile_size: int = 512) -> np.ndarray:
    """
    模擬「每個 tile 套用 polygon mask 後」的效果。
    產生一張圖：綠色區域 = tile 網格內有效 (在 polygon 內)，
              紅色區域 = tile 網格內但被 polygon 排除 (非產品區)。
    """
    H, W = gray.shape
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 建立 polygon mask (整張圖尺寸)
    panel_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(panel_mask, [polygon.astype(np.int32)], 255)

    xmin, ymin, xmax, ymax = bbox

    # 產生 tile 位置 (simulate generate_tile_positions with stride=tile_size)
    def gen_positions(start, end, size):
        positions = []
        pos = start
        while pos + size <= end:
            positions.append(pos)
            pos += size
        if positions and positions[-1] + size < end:
            edge_pos = end - size
            if edge_pos > positions[-1]:
                positions.append(edge_pos)
        elif not positions and end - start >= size:
            positions.append(start)
        return positions

    xs = gen_positions(xmin, xmax, tile_size)
    ys = gen_positions(ymin, ymax, tile_size)

    # 對每個 tile：找出「tile 在 panel_mask 外的區域」並以紅色塗上
    overlay = vis.copy()
    for ty in ys:
        for tx in xs:
            tile_mask = panel_mask[ty:ty + tile_size, tx:tx + tile_size]
            excluded = tile_mask == 0  # 這是被排除的區塊
            if excluded.any():
                # 把這塊塗成紅色 (alpha blend)
                roi = overlay[ty:ty + tile_size, tx:tx + tile_size]
                roi[excluded] = (0, 0, 255)
            # 畫 tile 框
            cv2.rectangle(vis, (tx, ty), (tx + tile_size, ty + tile_size),
                          (0, 255, 0), 3)

    # alpha blend
    vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)

    # 畫 polygon
    cv2.polylines(vis, [polygon.astype(np.int32).reshape(-1, 1, 2)],
                  True, (0, 255, 255), 4)
    cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (255, 0, 0), 4)
    return vis


def process_one(image_path: str):
    name = Path(image_path).stem
    print(f"\n=== {name} ===")

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"  ERROR: cannot read {image_path}")
        return

    H, W = gray.shape
    print(f"  size {W}x{H}")

    xmin, ymin, xmax, ymax, binary_mask = current_otsu_bbox(gray)
    bbox = (xmin, ymin, xmax, ymax)
    print(f"  current bbox: ({xmin},{ymin})-({xmax},{ymax})  size={xmax-xmin}x{ymax-ymin}")

    polygon = find_panel_polygon(gray, binary_mask, bbox)
    print(f"  fitted polygon corners:")
    for i, nm in enumerate(['TL', 'TR', 'BR', 'BL']):
        print(f"    {nm} = ({polygon[i][0]:.1f}, {polygon[i][1]:.1f})")

    # 每個角的誤差
    corners_bbox = np.array([(xmin, ymin), (xmax, ymin),
                             (xmax, ymax), (xmin, ymax)], dtype=float)
    diffs = np.linalg.norm(polygon - corners_bbox, axis=1)
    print(f"  corner errors vs bbox: TL={diffs[0]:.1f}  TR={diffs[1]:.1f}  BR={diffs[2]:.1f}  BL={diffs[3]:.1f}")

    # 面積比
    bbox_area = (xmax - xmin) * (ymax - ymin)
    poly_area = cv2.contourArea(polygon.astype(np.float32))
    print(f"  bbox area: {bbox_area}  polygon area: {poly_area:.0f}  diff: {(bbox_area - poly_area) / bbox_area * 100:.2f}%")

    # 整張視覺化 (下採樣輸出)
    vis_full = draw_visualization(gray, bbox, polygon, name)
    out_full = OUT_DIR / f"{name}_full.png"
    scale = 0.25
    vis_small = cv2.resize(vis_full, (0, 0), fx=scale, fy=scale)
    cv2.imwrite(str(out_full), vis_small)
    print(f"  saved {out_full}")

    # 4 個角的放大圖
    corners_names = ['TL', 'TR', 'BR', 'BL']
    for i, nm in enumerate(corners_names):
        px, py = int(polygon[i][0]), int(polygon[i][1])
        crop = crop_corner(vis_full, px, py, size=600, label=f"{name} {nm}")
        out_corner = OUT_DIR / f"{name}_corner_{nm}.png"
        cv2.imwrite(str(out_corner), crop)
    print(f"  saved 4 corner crops")

    # Tile mask 模擬圖 (整張下採樣)
    vis_tile = simulate_tile_mask(gray, polygon, bbox, tile_size=512)
    out_tile = OUT_DIR / f"{name}_tilemask.png"
    vis_tile_small = cv2.resize(vis_tile, (0, 0), fx=scale, fy=scale)
    cv2.imwrite(str(out_tile), vis_tile_small)
    print(f"  saved {out_tile}")

    # Tile mask 的右下角放大
    br_x, br_y = int(polygon[2][0]), int(polygon[2][1])
    tile_crop = crop_corner(vis_tile, br_x - 100, br_y - 100, size=1200,
                            label=f"{name} BR zoom (green=tile, red=excluded by polygon)")
    out_tile_br = OUT_DIR / f"{name}_tilemask_BR_zoom.png"
    cv2.imwrite(str(out_tile_br), tile_crop)
    print(f"  saved {out_tile_br}")


if __name__ == "__main__":
    for img in TEST_IMAGES:
        process_one(img)
    print(f"\nAll outputs in: {OUT_DIR}")
