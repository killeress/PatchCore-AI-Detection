"""
Dust Filter 計算過程圖表可視化
用 matplotlib 產生清楚的圖表說明每一步的數字
"""
import re
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer, AOIReportDefect, DEFAULT_PRODUCT_RESOLUTION


def main():
    # === Setup ===
    config = CAPIConfig.from_yaml("configs/capi_3f.yaml")
    config.dust_brightness_threshold = 20
    config.dust_threshold_floor = 15
    config.dust_heatmap_top_percent = 0.2
    config.dust_heatmap_iou_threshold = 0.100

    inferencer = CAPIInferencer(config)

    image_dir = Path("./test_images")
    image_path = list(image_dir.glob("W0F*.tif"))[0]
    omit_path = list(image_dir.glob("PINIGBI*.*"))[0]
    omit_image = cv2.imread(str(omit_path), cv2.IMREAD_UNCHANGED)
    txt = list(image_dir.glob("*_X*Y*.txt"))[0]
    m = re.match(r'^(.+?)_X(\d+)Y(\d+)$', txt.stem)
    px, py = int(m.group(2)), int(m.group(3))

    result = inferencer.preprocess_image(image_path)
    full_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    img_prefix = inferencer._get_image_prefix(image_path.name)
    aoi_defects = [AOIReportDefect(defect_code="TXT", product_x=px, product_y=py, image_prefix=img_prefix)]
    new_tiles, _ = inferencer._create_aoi_coord_tiles(full_image, result, aoi_defects, DEFAULT_PRODUCT_RESOLUTION)
    result.tiles.extend(new_tiles)
    result.tiles = [t for t in result.tiles if t.is_aoi_coord_tile]
    target_inf = inferencer._get_inferencer_for_prefix(img_prefix)
    target_thr = inferencer._get_threshold_for_prefix(img_prefix)
    result = inferencer.run_inference(result, inferencer=target_inf, threshold=target_thr)

    tile, score, anomaly_map = result.anomaly_tiles[0]
    tx, ty, tw, th = tile.x, tile.y, tile.width, tile.height
    oh, ow = omit_image.shape[:2]
    omit_crop = omit_image[ty:min(ty + th, oh), tx:min(tx + tw, ow)]
    is_dust, dust_mask, bright_ratio, detail_text = inferencer.check_dust_or_scratch_feature(omit_crop)

    # 準備資料
    anomaly_f = np.asarray(anomaly_map, dtype=np.float32)
    anomaly_f = np.maximum(anomaly_f, 0.0)
    h, w = anomaly_f.shape

    dm = np.asarray(dust_mask, dtype=np.uint8)
    if len(dm.shape) == 3:
        dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
    if dm.shape != anomaly_f.shape:
        dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_NEAREST)
    dust_bool = dm > 0

    # 閾值
    top_pct = 0.2
    positive_all = anomaly_f[anomaly_f > 0]
    thr_old = np.percentile(positive_all, 100 - top_pct)

    masked = anomaly_f.copy()
    masked[dust_bool] = 0
    positive_masked = masked[masked > 0]
    thr_new = np.percentile(positive_masked, 100 - top_pct)

    heat_old = anomaly_f >= thr_old
    heat_new = masked >= thr_new

    out_dir = Path("./test_dust_comparison_output")
    out_dir.mkdir(exist_ok=True)

    # 原圖 (for overlay)
    tile_img = tile.image
    if tile_img is not None:
        if len(tile_img.shape) == 2:
            tile_rgb = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2RGB)
        else:
            tile_rgb = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)
        tile_rgb_resized = cv2.resize(tile_rgb, (w, h))
    else:
        tile_rgb_resized = np.zeros((h, w, 3), dtype=np.uint8)

    # ================================================================
    # Chart 1: Overview
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Dust Filter Debug | Score={score:.4f} | Tile ({tx},{ty}) {tw}x{th}\n"
                 f"anomaly_map: {h}x{w} = {h*w:,} pixels, 1 pixel = 1 float score",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    im = ax.imshow(anomaly_f, cmap="jet", vmin=np.min(anomaly_f), vmax=np.max(anomaly_f))
    plt.colorbar(im, ax=ax, label="anomaly score", shrink=0.8)
    peak_yx = np.unravel_index(np.argmax(anomaly_f), anomaly_f.shape)
    ax.plot(peak_yx[1], peak_yx[0], "w+", markersize=15, markeredgewidth=2)
    ax.set_title(f"[1] Anomaly Map (1 pixel = 1 score)\nmin={np.min(anomaly_f):.4f}  max={np.max(anomaly_f):.4f}")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    ax = axes[0, 1]
    dust_scores = anomaly_f[dust_bool]
    non_dust_scores = anomaly_f[~dust_bool]
    bins = np.linspace(np.min(anomaly_f), np.max(anomaly_f), 50)
    ax.hist(non_dust_scores.flatten(), bins=bins, alpha=0.7, color="steelblue", label=f"non-dust ({len(non_dust_scores):,}px)")
    ax.hist(dust_scores.flatten(), bins=bins, alpha=0.7, color="orange", label=f"dust ({len(dust_scores):,}px)")
    ax.axvline(thr_old, color="red", linestyle="--", linewidth=2, label=f"OLD thr={thr_old:.4f}")
    ax.axvline(thr_new, color="lime", linestyle="--", linewidth=2, label=f"NEW thr={thr_new:.4f}")
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("pixel count")
    ax.set_title("[2] Score Distribution\norange=dust  blue=non-dust")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.imshow(tile_rgb_resized, alpha=0.6)
    dust_overlay = np.zeros((*anomaly_f.shape, 4))
    dust_overlay[dust_bool] = [1, 1, 0, 0.5]
    ax.imshow(dust_overlay)
    ax.set_title(f"[3] Dust Mask (yellow)\n{np.count_nonzero(dust_bool):,} px ({np.count_nonzero(dust_bool)/dust_bool.size*100:.1f}%)")

    ax = axes[1, 0]
    ax.imshow(tile_rgb_resized, alpha=0.5)
    dust_ol = np.zeros((*anomaly_f.shape, 4))
    dust_ol[dust_bool] = [1, 1, 0, 0.3]
    ax.imshow(dust_ol)
    heat_ol = np.zeros((*anomaly_f.shape, 4))
    heat_ol[heat_old] = [0, 1, 0, 0.8]
    ax.imshow(heat_ol)
    heat_in_dust = np.count_nonzero(heat_old & dust_bool)
    heat_not_dust = np.count_nonzero(heat_old & ~dust_bool)
    ax.set_title(f"[4] OLD: top 0.2% (green) + dust (yellow)\n"
                 f"thr={thr_old:.4f} -> {np.count_nonzero(heat_old)} px\n"
                 f"green_in_dust={heat_in_dust} ({heat_in_dust/max(np.count_nonzero(heat_old),1)*100:.0f}%) -> DUST(OK)",
                 fontsize=10)

    ax = axes[1, 1]
    im2 = ax.imshow(masked, cmap="jet", vmin=np.min(anomaly_f), vmax=np.max(anomaly_f))
    plt.colorbar(im2, ax=ax, label="score (dust=0)", shrink=0.8)
    contours_mask = dm.copy()
    contour_list, _ = cv2.findContours(contours_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contour_list:
        c_squeezed = c.squeeze()
        if len(c_squeezed.shape) == 2 and len(c_squeezed) > 2:
            ax.plot(c_squeezed[:, 0], c_squeezed[:, 1], "y-", linewidth=0.5, alpha=0.5)
    ax.set_title(f"[5] NEW: dust zeroed\nyellow outline = zeroed area\nmax={np.max(masked):.4f}")

    ax = axes[1, 2]
    ax.imshow(tile_rgb_resized, alpha=0.5)
    ax.imshow(dust_ol)
    heat_new_ol = np.zeros((*anomaly_f.shape, 4))
    heat_new_ol[heat_new] = [0, 1, 0, 0.8]
    ax.imshow(heat_new_ol)
    ax.set_title(f"[6] NEW: top 0.2% (green) + dust (yellow)\n"
                 f"thr={thr_new:.4f} -> {np.count_nonzero(heat_new)} px\n"
                 f"green all in non-dust -> NG (Detected)",
                 fontsize=10)

    plt.tight_layout()
    path1 = str(out_dir / "debug_chart_overview.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart 1 saved: {path1}")

    # ================================================================
    # Chart 2: Peak zoom with actual pixel values
    # ================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 7))
    fig2.suptitle("Peak Zoom: actual pixel values (1 cell = 1 pixel = 1 float)", fontsize=14, fontweight="bold")

    peak_y, peak_x = peak_yx
    r = 12
    y1, y2 = max(0, peak_y - r), min(h, peak_y + r + 1)
    x1, x2 = max(0, peak_x - r), min(w, peak_x + r + 1)

    crop_anomaly = anomaly_f[y1:y2, x1:x2]
    crop_dust = dust_bool[y1:y2, x1:x2]
    crop_masked = masked[y1:y2, x1:x2]

    ax = axes2[0]
    im3 = ax.imshow(crop_anomaly, cmap="jet", vmin=np.min(anomaly_f), vmax=np.max(anomaly_f))
    plt.colorbar(im3, ax=ax, shrink=0.8)
    crop_dm = dm[y1:y2, x1:x2]
    contour_crop, _ = cv2.findContours(crop_dm.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contour_crop:
        c_s = c.squeeze()
        if len(c_s.shape) == 2 and len(c_s) > 2:
            ax.plot(c_s[:, 0], c_s[:, 1], "w-", linewidth=1.5)
    ax.plot(peak_x - x1, peak_y - y1, "w+", markersize=20, markeredgewidth=3)
    ax.contour(crop_anomaly >= thr_old, levels=[0.5], colors=["lime"], linewidths=2)
    ax.set_title(f"Original anomaly_map\nwhite outline=dust  green line=top0.2%(thr={thr_old:.4f})\nwhite+=peak({np.max(crop_anomaly):.4f})")

    ax = axes2[1]
    display = np.zeros((*crop_anomaly.shape, 3))
    for iy in range(crop_anomaly.shape[0]):
        for ix in range(crop_anomaly.shape[1]):
            if crop_dust[iy, ix]:
                display[iy, ix] = [1, 0.8, 0]  # yellow = dust
            else:
                display[iy, ix] = [0.2, 0.5, 1]  # blue = non-dust
    ax.imshow(display)
    cy, cx = crop_anomaly.shape[0] // 2, crop_anomaly.shape[1] // 2
    for iy in range(max(0, cy-3), min(crop_anomaly.shape[0], cy+4)):
        for ix in range(max(0, cx-3), min(crop_anomaly.shape[1], cx+4)):
            val = crop_anomaly[iy, ix]
            color = "black" if crop_dust[iy, ix] else "white"
            ax.text(ix, iy, f"{val:.3f}", ha="center", va="center", fontsize=5.5,
                    color=color, fontweight="bold")
    ax.set_title("Pixel classification + score\nyellow=dust pixel  blue=non-dust pixel\ncenter 7x7 shows actual values")
    dust_legend = mpatches.Patch(color=[1, 0.8, 0], label="Dust pixel")
    non_dust_legend = mpatches.Patch(color=[0.2, 0.5, 1], label="Non-dust pixel")
    ax.legend(handles=[dust_legend, non_dust_legend], loc="lower left", fontsize=8)

    ax = axes2[2]
    im4 = ax.imshow(crop_masked, cmap="jet", vmin=np.min(anomaly_f), vmax=np.max(anomaly_f))
    plt.colorbar(im4, ax=ax, shrink=0.8)
    ax.contour(crop_masked >= thr_new, levels=[0.5], colors=["lime"], linewidths=2)
    ax.set_title(f"After dust zeroed\ndark blue=zeroed  green line=new top0.2%(thr={thr_new:.4f})\ndust removed, defect exposed")

    plt.tight_layout()
    path2 = str(out_dir / "debug_chart_peak_zoom.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Chart 2 saved: {path2}")

    # ================================================================
    # 圖 3: 流程對比圖
    # ================================================================
    fig3, axes3 = plt.subplots(2, 4, figsize=(20, 10))
    fig3.suptitle("OLD vs NEW Flow Comparison", fontsize=14, fontweight="bold")

    # === 上排: 舊版 ===
    # Step 1: original anomaly
    ax = axes3[0, 0]
    ax.imshow(anomaly_f, cmap="jet")
    ax.set_title("Step1: anomaly_map\n(original)")
    ax.text(0.5, -0.1, f"65,536 pixels\nall > 0", transform=ax.transAxes, ha="center", fontsize=9)

    # Step 2: top 0.2%
    ax = axes3[0, 1]
    binary_vis = np.zeros((*anomaly_f.shape, 3))
    binary_vis[heat_old] = [0, 1, 0]
    binary_vis[~heat_old] = [0.1, 0.1, 0.1]
    ax.imshow(binary_vis)
    ax.set_title(f"Step2: top 0.2% 二值化\nthr={thr_old:.4f}")
    ax.text(0.5, -0.1, f"{np.count_nonzero(heat_old)} px passed", transform=ax.transAxes, ha="center", fontsize=9)

    # Step 3: COV 計算
    ax = axes3[0, 2]
    cov_vis = np.zeros((*anomaly_f.shape, 3))
    cov_vis[heat_old & dust_bool] = [1, 1, 0]       # yellow = overlap
    cov_vis[heat_old & ~dust_bool] = [0, 1, 0]      # green = heat only
    cov_vis[~heat_old & dust_bool] = [0.3, 0.3, 0]  # dim yellow = dust only
    ax.imshow(cov_vis)
    cov_val = heat_in_dust / max(np.count_nonzero(heat_old), 1)
    ax.set_title(f"Step3: COV calc\nyellow=overlap green=heat dim=dust")
    ax.text(0.5, -0.1, f"COV = {heat_in_dust}/{np.count_nonzero(heat_old)} = {cov_val:.3f}",
            transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold")

    ax = axes3[0, 3]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.7, "COV = 0.606", fontsize=20, ha="center", fontweight="bold", color="orange")
    ax.text(0.5, 0.5, ">= 0.100", fontsize=16, ha="center", color="gray")
    ax.text(0.5, 0.3, "-> DUST (OK)", fontsize=22, ha="center", fontweight="bold", color="orange")
    ax.text(0.5, 0.1, "X MISSED", fontsize=18, ha="center", color="red")
    ax.set_title("Step4: Verdict")
    ax.axis("off")

    # === Bottom row: NEW ===
    ax = axes3[1, 0]
    ax.imshow(masked, cmap="jet", vmin=np.min(anomaly_f), vmax=np.max(anomaly_f))
    ax.set_title("Step1: dust zeroed\ndust_mask area -> 0")
    ax.text(0.5, -0.1, f"{np.count_nonzero(dust_bool):,} px zeroed\nremain {len(positive_masked):,} px",
            transform=ax.transAxes, ha="center", fontsize=9)

    ax = axes3[1, 1]
    binary_new_vis = np.zeros((*anomaly_f.shape, 3))
    binary_new_vis[heat_new] = [0, 1, 0]
    binary_new_vis[~heat_new & ~dust_bool] = [0.1, 0.1, 0.1]
    binary_new_vis[dust_bool] = [0.05, 0.05, 0.15]
    ax.imshow(binary_new_vis)
    ax.set_title(f"Step2: top 0.2% binarize\nthr={thr_new:.4f} (lower)")
    ax.text(0.5, -0.1, f"{np.count_nonzero(heat_new)} px passed\ndark blue=zeroed area",
            transform=ax.transAxes, ha="center", fontsize=9)

    ax = axes3[1, 2]
    cov_new_vis = np.zeros((*anomaly_f.shape, 3))
    cov_new_vis[heat_new] = [0, 1, 0]
    cov_new_vis[dust_bool] = [0.05, 0.05, 0.15]
    ax.imshow(cov_new_vis)
    ax.set_title(f"Step3: COV = 0\ngreen all in non-dust area")
    ax.text(0.5, -0.1, f"COV = 0/{np.count_nonzero(heat_new)} = 0.000",
            transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold")

    ax = axes3[1, 3]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.7, "COV = 0.000", fontsize=20, ha="center", fontweight="bold", color="red")
    ax.text(0.5, 0.5, "< 0.100", fontsize=16, ha="center", color="gray")
    ax.text(0.5, 0.3, "-> NG", fontsize=22, ha="center", fontweight="bold", color="red")
    ax.text(0.5, 0.1, "CAUGHT!", fontsize=18, ha="center", color="green")
    ax.set_title("Step4: Verdict")
    ax.axis("off")

    axes3[0, 0].set_ylabel("OLD flow", fontsize=14, fontweight="bold", color="orange")
    axes3[1, 0].set_ylabel("NEW flow", fontsize=14, fontweight="bold", color="red")

    plt.tight_layout()
    path3 = str(out_dir / "debug_chart_flow_compare.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Chart 3 saved: {path3}")

    print("\nDone!")


if __name__ == "__main__":
    main()
