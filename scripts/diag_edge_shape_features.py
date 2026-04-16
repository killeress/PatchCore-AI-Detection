"""
實驗 A: 邊緣 CV 過檢 vs 真 NG 形狀特徵分佈分析

用途:
    以「形狀特徵」嘗試把 CV 邊緣檢測在 L 形歪斜產品上的外輪廓過檢 (false positive)
    和真正的邊緣缺陷 (positive) 切開。

資料來源:
    FP  (false positive / over_edge_false_positive):
        datasets/over_review/over_edge_false_positive/WGF50500/crop/   (99 張)
    POS (真 NG):
        datasets/over_review/true_ng/WGF50500/crop/   (只取 *_edge*.png)

邏輯 (模擬 capi_edge_cv._inspect_side):
    1. BGR -> Gray
    2. Gaussian blur 3x3
    3. Foreground-aware diff (median bg + absdiff)
    4. threshold=5 二值化
    5. connectedComponentsWithStats
    6. 對每個面積 >= min_area(60) 的 component 算形狀特徵

輸出:
    reports/edge_shape_features.csv
    reports/edge_shape_vis/*.png
"""

import os
import sys
import glob
import math
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

# ---------------- 設定 ----------------
ROOT = Path(r"C:/Users/rh.syu/Desktop/CAPI01_AD")
FP_DIR = ROOT / "datasets/over_review/over_edge_false_positive/WGF50500/crop"
POS_DIR = ROOT / "datasets/over_review/true_ng/WGF50500/crop"
OUT_DIR = ROOT / "reports"
VIS_DIR = OUT_DIR / "edge_shape_vis"
CSV_PATH = OUT_DIR / "edge_shape_features.csv"

# 模擬 _inspect_side 參數 (與 WGF50500 類似產品的右 / 下邊預設值相近)
BLUR_KERNEL = 3
MEDIAN_KERNEL = 65       # clamp for 512x512 OK
THRESHOLD = 5
MIN_AREA = 60
FG_BRIGHTNESS_THRESHOLD = 15

OUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


def clamp_median_kernel(k: int, max_dim: int) -> int:
    mk = min(k, max_dim)
    if mk % 2 == 0:
        mk -= 1
    return max(3, mk)


def compute_fg_aware_diff(gray: np.ndarray):
    blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
    fg_mask = np.zeros_like(gray)
    fg_mask[gray >= FG_BRIGHTNESS_THRESHOLD] = 255

    fg_pixels = blurred[fg_mask > 0]
    if fg_pixels.size > 0:
        blurred_for_bg = blurred.copy()
        blurred_for_bg[fg_mask == 0] = int(np.median(fg_pixels))
    else:
        blurred_for_bg = blurred

    mk = clamp_median_kernel(MEDIAN_KERNEL, min(gray.shape[:2]) - 1)
    bg = cv2.medianBlur(blurred_for_bg, mk)
    diff = cv2.absdiff(blurred, bg)
    diff[fg_mask == 0] = 0
    return fg_mask, diff


def perimeter_touching_fg_boundary(component_mask: np.ndarray, fg_mask: np.ndarray) -> float:
    """回傳 component 邊界中貼著 fg_mask 邊緣或影像邊緣的比例 (0~1)"""
    # component 輪廓
    cnts, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    pts = cnt.reshape(-1, 2)
    if len(pts) == 0:
        return 0.0

    H, W = fg_mask.shape
    # 3x3 擴張的非前景 mask，用來判斷像素是否「貼著 fg 邊界」
    non_fg = (fg_mask == 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    non_fg_dil = cv2.dilate(non_fg, kernel, iterations=1)

    touching = 0
    for (x, y) in pts:
        on_img_border = (x <= 0 or y <= 0 or x >= W - 1 or y >= H - 1)
        near_fg_boundary = non_fg_dil[y, x] > 0
        if on_img_border or near_fg_boundary:
            touching += 1
    return touching / max(1, len(pts))


def extract_components(img_path: Path, label: str):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    fg_mask, diff = compute_fg_aware_diff(gray)
    _, mask = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    rows = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        component_mask = (labels == i).astype(np.uint8) * 255
        # 輪廓 / 周長 / 凸包面積
        cnts, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        perim = float(cv2.arcLength(cnt, True))
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))

        aspect_ratio = max(w, h) / max(1, min(w, h))
        solidity = area / hull_area if hull_area > 0 else 0.0
        extent = area / float(w * h) if w * h > 0 else 0.0
        perim_area_ratio = perim / math.sqrt(max(1, area))
        edge_touch = perimeter_touching_fg_boundary(component_mask // 255, fg_mask)
        max_diff = int(np.max(diff[component_mask > 0]))

        rows.append({
            "file": img_path.name,
            "label": label,
            "component_idx": i,
            "area": area,
            "aspect_ratio": round(aspect_ratio, 4),
            "solidity": round(solidity, 4),
            "extent": round(extent, 4),
            "edge_touch_ratio": round(edge_touch, 4),
            "perimeter_area_ratio": round(perim_area_ratio, 4),
            "max_diff": max_diff,
        })
    return rows


def collect(folder: Path, label: str):
    if not folder.exists():
        print(f"[WARN] folder not found: {folder}")
        return []
    all_rows = []
    files = [p for p in sorted(folder.glob("*.png")) if "_edge" in p.name or label == "fp"]
    # fp 資料夾裡一些樣本是 tile82，也一併納入
    if label == "fp":
        files = sorted(folder.glob("*.png"))
    print(f"[INFO] {label}: {len(files)} files from {folder}")
    for p in files:
        all_rows.extend(extract_components(p, label))
    return all_rows


def describe(df: pd.DataFrame, col: str):
    s = df[col].dropna()
    if s.empty:
        return {}
    return {
        "n": int(s.size),
        "mean": round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
        "p10": round(float(s.quantile(0.10)), 4),
        "p90": round(float(s.quantile(0.90)), 4),
        "p95": round(float(s.quantile(0.95)), 4),
    }


def plot_hist(df: pd.DataFrame, col: str, out_path: Path, clip=None):
    plt.figure(figsize=(7, 4))
    data_fp = df[df.label == "fp"][col].dropna()
    data_pos = df[df.label == "pos"][col].dropna()
    if clip is not None:
        lo, hi = clip
        data_fp = data_fp.clip(lo, hi)
        data_pos = data_pos.clip(lo, hi)
    bins = 40
    if not data_fp.empty:
        plt.hist(data_fp, bins=bins, alpha=0.5, label=f"fp (n={len(data_fp)})", color="tab:red", density=True)
    if not data_pos.empty:
        plt.hist(data_pos, bins=bins, alpha=0.5, label=f"pos (n={len(data_pos)})", color="tab:blue", density=True)
    plt.title(f"{col}  —  fp vs pos")
    plt.xlabel(col)
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


def plot_scatter(df: pd.DataFrame, xcol: str, ycol: str, out_path: Path):
    plt.figure(figsize=(6.5, 5.5))
    for lbl, color in [("fp", "tab:red"), ("pos", "tab:blue")]:
        sub = df[df.label == lbl]
        if sub.empty:
            continue
        plt.scatter(sub[xcol], sub[ycol], s=14, alpha=0.55,
                    label=f"{lbl} (n={len(sub)})", color=color, edgecolors="none")
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(f"{xcol}  vs  {ycol}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


def best_threshold_1d(df, col, higher_is_fp=True):
    """掃描單特徵單閾值，回傳 (thr, tpr_pos_kept, fpr_fp_kept, youden)。
    higher_is_fp=True 表示 高值 => 判定為 fp (會被濾掉)，目標：保留 pos (低值)"""
    s_fp = df[df.label == "fp"][col].dropna().values
    s_pos = df[df.label == "pos"][col].dropna().values
    if len(s_fp) == 0 or len(s_pos) == 0:
        return None
    # 候選 threshold: 全部值 union 的分位
    all_v = np.concatenate([s_fp, s_pos])
    cands = np.quantile(all_v, np.linspace(0.01, 0.99, 60))
    best = None
    for t in cands:
        if higher_is_fp:
            # 判定為 fp if value >= t
            fp_filtered = (s_fp >= t).mean()       # fp 被濾掉的比例 (想要高)
            pos_kept = (s_pos < t).mean()          # pos 被保留比例 (想要高)
        else:
            fp_filtered = (s_fp <= t).mean()
            pos_kept = (s_pos > t).mean()
        youden = fp_filtered + pos_kept - 1
        if best is None or youden > best[3]:
            best = (round(float(t), 4), round(float(pos_kept), 4),
                    round(float(fp_filtered), 4), round(float(youden), 4))
    return best


def main():
    rows = []
    rows.extend(collect(FP_DIR, "fp"))
    rows.extend(collect(POS_DIR, "pos"))

    if not rows:
        print("[ERROR] no components extracted")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"[OK] csv -> {CSV_PATH} ({len(df)} rows)")

    feats = ["area", "aspect_ratio", "solidity", "extent",
             "edge_touch_ratio", "perimeter_area_ratio", "max_diff"]

    # --- 統計 ---
    print("\n=== 分佈統計 ===")
    summary_lines = []
    for lbl in ("fp", "pos"):
        sub = df[df.label == lbl]
        print(f"\n[{lbl}] n_components = {len(sub)}, n_files = {sub['file'].nunique()}")
        summary_lines.append(f"[{lbl}] n_components={len(sub)}, n_files={sub['file'].nunique()}")
        for f in feats:
            d = describe(sub, f)
            line = f"  {f:<22s} {d}"
            print(line)
            summary_lines.append(line)

    # --- 直方圖 ---
    clip_map = {
        "area": (0, 5000),
        "aspect_ratio": (1, 30),
        "perimeter_area_ratio": (0, 20),
        "max_diff": (0, 60),
    }
    for f in feats:
        plot_hist(df, f, VIS_DIR / f"hist_{f}.png", clip=clip_map.get(f))
    # --- 散佈圖 (重點配對) ---
    pairs = [
        ("solidity", "aspect_ratio"),
        ("edge_touch_ratio", "perimeter_area_ratio"),
        ("solidity", "edge_touch_ratio"),
        ("aspect_ratio", "perimeter_area_ratio"),
        ("extent", "solidity"),
        ("area", "max_diff"),
    ]
    for a, b in pairs:
        plot_scatter(df, a, b, VIS_DIR / f"scatter_{a}_vs_{b}.png")

    # --- 單特徵最佳切分 (Youden J) ---
    print("\n=== 單特徵最佳閾值 (掃描 Youden J) ===")
    print("格式: feature  thr  pos_kept  fp_filtered  youden  (方向)")
    single_results = []
    directions = {
        "area": True,                  # fp 面積大 (外輪廓), pos 可能小 → 高值=fp
        "aspect_ratio": True,          # 外輪廓是細長條 → 高值=fp
        "solidity": False,             # 外輪廓 solidity 低 (彎曲) → 低值=fp
        "extent": False,               # 外輪廓 extent 低 → 低值=fp
        "edge_touch_ratio": True,      # 外輪廓貼著邊界 → 高值=fp
        "perimeter_area_ratio": True,  # 細長 → 高周長面積比 → 高值=fp
        "max_diff": False,             # 真缺陷對比強 → 低值=fp ?
    }
    for f in feats:
        res = best_threshold_1d(df, f, higher_is_fp=directions[f])
        if res is None:
            continue
        thr, pos_kept, fp_filtered, y = res
        dir_txt = "val>=thr=>fp" if directions[f] else "val<=thr=>fp"
        line = f"  {f:<22s} thr={thr:<10} pos_kept={pos_kept:<6} fp_filtered={fp_filtered:<6} youden={y:<6} {dir_txt}"
        print(line)
        single_results.append((f, thr, pos_kept, fp_filtered, y, dir_txt))

    # --- 儲存 summary ---
    with open(OUT_DIR / "edge_shape_features_summary.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines))
        fh.write("\n\n=== 單特徵最佳閾值 (Youden J) ===\n")
        for f, thr, pk, fpf, y, d in single_results:
            fh.write(f"{f:<22s} thr={thr}  pos_kept={pk}  fp_filtered={fpf}  youden={y}  {d}\n")

    print(f"\n[OK] vis -> {VIS_DIR}")
    print(f"[OK] summary -> {OUT_DIR/'edge_shape_features_summary.txt'}")


if __name__ == "__main__":
    main()
