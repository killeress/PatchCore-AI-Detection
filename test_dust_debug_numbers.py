"""
Dust Filter 計算過程完整 Debug
把每一步的數字、像素分布、閾值計算全部可視化
"""
import re
from pathlib import Path

import cv2
import numpy as np

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer, AOIReportDefect, DEFAULT_PRODUCT_RESOLUTION


def main():
    # === Setup (同 test_aoi_coord) ===
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

    # 推論
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

    # OMIT crop + dust mask
    tx, ty, tw, th = tile.x, tile.y, tile.width, tile.height
    oh, ow = omit_image.shape[:2]
    omit_crop = omit_image[ty:min(ty + th, oh), tx:min(tx + tw, ow)]
    is_dust, dust_mask, bright_ratio, detail_text = inferencer.check_dust_or_scratch_feature(omit_crop)

    # ================================================================
    print("=" * 70)
    print("1. anomaly_map 基本資訊")
    print("=" * 70)

    anomaly_f = np.asarray(anomaly_map, dtype=np.float32)
    anomaly_f = np.maximum(anomaly_f, 0.0)

    print(f"   type        : {type(anomaly_map)}")
    print(f"   dtype       : {anomaly_map.dtype}")
    print(f"   shape       : {anomaly_f.shape}  ← (高, 寬), 每個 pixel 一個 float 分數")
    print(f"   total pixels: {anomaly_f.size:,}")
    print(f"   min         : {np.min(anomaly_f):.6f}")
    print(f"   max         : {np.max(anomaly_f):.6f}")
    print(f"   mean        : {np.mean(anomaly_f):.6f}")
    print(f"   median      : {np.median(anomaly_f):.6f}")
    print(f"   std         : {np.std(anomaly_f):.6f}")

    print(f"\n   Percentile 分布:")
    for p in [50, 75, 90, 95, 99, 99.5, 99.8, 99.9, 100]:
        val = np.percentile(anomaly_f, p)
        print(f"     P{p:>5.1f}: {val:.6f}")

    positive = anomaly_f[anomaly_f > 0]
    zero_count = np.count_nonzero(anomaly_f == 0)
    print(f"\n   零值像素: {zero_count:,} ({zero_count / anomaly_f.size * 100:.1f}%)")
    print(f"   正值像素: {len(positive):,} ({len(positive) / anomaly_f.size * 100:.1f}%)")

    # 印一小塊實際數值讓使用者看
    h, w = anomaly_f.shape
    peak_idx = np.unravel_index(np.argmax(anomaly_f), anomaly_f.shape)
    print(f"\n   最大值位置: pixel ({peak_idx[1]}, {peak_idx[0]})  ← (x, y)")
    print(f"   最大值附近 5x5 像素的實際數值:")
    py_c, px_c = peak_idx
    for row in range(max(0, py_c - 2), min(h, py_c + 3)):
        vals = []
        for col in range(max(0, px_c - 2), min(w, px_c + 3)):
            marker = " *" if (row == py_c and col == px_c) else "  "
            vals.append(f"{anomaly_f[row, col]:.4f}{marker}")
        print(f"     row {row:3d}: {' | '.join(vals)}")

    # ================================================================
    print(f"\n{'=' * 70}")
    print("2. dust_mask 基本資訊")
    print("=" * 70)

    dm = np.asarray(dust_mask, dtype=np.uint8)
    if len(dm.shape) == 3:
        dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
    if dm.shape != anomaly_f.shape:
        dm = cv2.resize(dm, (anomaly_f.shape[1], anomaly_f.shape[0]), interpolation=cv2.INTER_NEAREST)
    dust_bool = dm > 0

    dust_px = np.count_nonzero(dust_bool)
    non_dust_px = anomaly_f.size - dust_px
    print(f"   shape       : {dm.shape}")
    print(f"   灰塵像素    : {dust_px:,} ({dust_px / dm.size * 100:.1f}%)")
    print(f"   非灰塵像素  : {non_dust_px:,} ({non_dust_px / dm.size * 100:.1f}%)")

    # 灰塵區域 vs 非灰塵區域的 anomaly 分數對比
    dust_scores = anomaly_f[dust_bool]
    non_dust_scores = anomaly_f[~dust_bool]
    print(f"\n   灰塵區域的 anomaly score:")
    print(f"     min={np.min(dust_scores):.4f}  max={np.max(dust_scores):.4f}  mean={np.mean(dust_scores):.4f}")
    print(f"   非灰塵區域的 anomaly score:")
    print(f"     min={np.min(non_dust_scores):.4f}  max={np.max(non_dust_scores):.4f}  mean={np.mean(non_dust_scores):.4f}")

    # ================================================================
    print(f"\n{'=' * 70}")
    print("3. 現有流程: 直接 top 0.2% 二值化")
    print("=" * 70)

    top_pct = 0.2
    positive_all = anomaly_f[anomaly_f > 0]
    thr_old = np.percentile(positive_all, 100 - top_pct)
    heat_old = anomaly_f >= thr_old
    heat_old_count = np.count_nonzero(heat_old)

    print(f"   正值像素數  : {len(positive_all):,}")
    print(f"   top {top_pct}%     : 取最高的 {top_pct}% = {int(len(positive_all) * top_pct / 100)} 像素")
    print(f"   percentile  : np.percentile(positive, {100 - top_pct}) = {thr_old:.6f}")
    print(f"   threshold   : {thr_old:.6f}")
    print(f"   >= threshold: {heat_old_count:,} 像素")

    # 這些像素有多少在灰塵區域?
    heat_in_dust = np.count_nonzero(heat_old & dust_bool)
    heat_not_dust = np.count_nonzero(heat_old & ~dust_bool)
    print(f"\n   top 0.2% 像素的分布:")
    print(f"     在灰塵區域: {heat_in_dust:,} ({heat_in_dust / heat_old_count * 100:.1f}%)")
    print(f"     在非灰塵區域: {heat_not_dust:,} ({heat_not_dust / heat_old_count * 100:.1f}%)")

    # connectedComponents
    heat_old_u8 = (heat_old.astype(np.uint8)) * 255
    num_labels_old, labels_old = cv2.connectedComponents(heat_old_u8, connectivity=8)
    print(f"\n   connectedComponents: {num_labels_old - 1} 個區域")
    for lid in range(1, num_labels_old):
        region_mask = labels_old == lid
        region_area = np.count_nonzero(region_mask)
        region_dust = np.count_nonzero(region_mask & dust_bool)
        cov = region_dust / region_area if region_area > 0 else 0
        region_scores = anomaly_f[region_mask]
        print(f"     Region {lid}: area={region_area}px, dust_overlap={region_dust}px, COV={cov:.3f}, "
              f"score_max={np.max(region_scores):.4f}, score_mean={np.mean(region_scores):.4f}")
        print(f"       → COV {cov:.3f} {'≥' if cov >= 0.1 else '<'} 0.1 → {'DUST' if cov >= 0.1 else 'REAL'}")

    # ================================================================
    print(f"\n{'=' * 70}")
    print("4. 方法4: 灰塵歸零後 top 0.2% 二值化")
    print("=" * 70)

    masked = anomaly_f.copy()
    masked[dust_bool] = 0

    positive_masked = masked[masked > 0]
    thr_new = np.percentile(positive_masked, 100 - top_pct)
    heat_new = masked >= thr_new
    # 灰塵區域已經是 0，不可能 >= threshold，但為了清楚還是確認
    heat_new_count = np.count_nonzero(heat_new)

    print(f"   歸零前正值像素: {len(positive_all):,}")
    print(f"   灰塵像素歸零  : {dust_px:,} 像素 → 0")
    print(f"   歸零後正值像素: {len(positive_masked):,}  (少了 {len(positive_all) - len(positive_masked):,})")
    print(f"   top {top_pct}%      : 取最高的 {top_pct}% = {int(len(positive_masked) * top_pct / 100)} 像素")
    print(f"   percentile   : np.percentile(positive_masked, {100 - top_pct}) = {thr_new:.6f}")
    print(f"   threshold    : {thr_new:.6f}")
    print(f"   >= threshold : {heat_new_count:,} 像素")

    print(f"\n   閾值比較:")
    print(f"     舊版 threshold: {thr_old:.6f}")
    print(f"     新版 threshold: {thr_new:.6f}")
    print(f"     差異          : {thr_old - thr_new:.6f} ({'舊版較高' if thr_old > thr_new else '新版較高'})")

    # connectedComponents
    heat_new_u8 = (heat_new.astype(np.uint8)) * 255
    num_labels_new, labels_new = cv2.connectedComponents(heat_new_u8, connectivity=8)
    print(f"\n   connectedComponents: {num_labels_new - 1} 個區域")
    for lid in range(1, num_labels_new):
        region_mask = labels_new == lid
        region_area = np.count_nonzero(region_mask)
        region_dust = np.count_nonzero(region_mask & dust_bool)
        cov = region_dust / region_area if region_area > 0 else 0
        region_scores = anomaly_f[region_mask]  # 用原始分數
        print(f"     Region {lid}: area={region_area}px, dust_overlap={region_dust}px, COV={cov:.3f}, "
              f"score_max={np.max(region_scores):.4f}, score_mean={np.mean(region_scores):.4f}")

    # ================================================================
    print(f"\n{'=' * 70}")
    print("5. 可視化: anomaly_map 分數分布直方圖 (文字版)")
    print("=" * 70)

    # 用文字畫直方圖
    bins = np.linspace(0, float(np.max(anomaly_f)), 21)
    hist_all, _ = np.histogram(anomaly_f[anomaly_f > 0], bins=bins)
    hist_dust, _ = np.histogram(anomaly_f[dust_bool & (anomaly_f > 0)], bins=bins)
    hist_non_dust, _ = np.histogram(anomaly_f[~dust_bool & (anomaly_f > 0)], bins=bins)

    max_count = max(hist_all.max(), 1)
    bar_width = 40

    print(f"\n   {'Score Range':>20s}  {'All':>7s}  {'Dust':>7s}  {'!Dust':>7s}  Bar (■=All, ▓=Dust)")
    print(f"   {'─' * 20}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * bar_width}")
    for i in range(len(hist_all)):
        lo, hi = bins[i], bins[i + 1]
        n_all = hist_all[i]
        n_dust = hist_dust[i]
        n_non = hist_non_dust[i]
        bar_len = int(n_all / max_count * bar_width)
        dust_bar_len = int(n_dust / max_count * bar_width)
        bar = "▓" * dust_bar_len + "■" * (bar_len - dust_bar_len)

        marker = ""
        if lo <= thr_old <= hi:
            marker += " ← OLD threshold"
        if lo <= thr_new <= hi:
            marker += " ← NEW threshold"

        print(f"   {lo:8.4f}~{hi:8.4f}  {n_all:7,}  {n_dust:7,}  {n_non:7,}  {bar}{marker}")

    # ================================================================
    print(f"\n{'=' * 70}")
    print("6. 視覺化圖片輸出")
    print("=" * 70)

    out_dir = Path("./test_dust_comparison_output")
    out_dir.mkdir(exist_ok=True)

    tile_img = tile.image
    if tile_img is not None:
        base = tile_img.copy()
        if len(base.shape) == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        base = np.zeros((h, w, 3), dtype=np.uint8)

    panel_size = (400, 400)
    font = cv2.FONT_HERSHEY_SIMPLEX
    panels = []

    # P1: 原圖
    p1 = cv2.resize(base, panel_size)
    cv2.putText(p1, "[1] Original", (8, 25), font, 0.6, (255, 255, 255), 2)
    panels.append(p1)

    # P2: anomaly_map heatmap overlay (follow capi_heatmap.py)
    norm_map = cv2.normalize(anomaly_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hm_color = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    if hm_color.shape[:2] != base.shape[:2]:
        hm_color = cv2.resize(hm_color, (base.shape[1], base.shape[0]))
    hm_overlay = cv2.addWeighted(base, 0.5, hm_color, 0.5, 0)
    p2 = cv2.resize(hm_overlay, panel_size)
    cv2.putText(p2, "[2] Heatmap", (8, 25), font, 0.6, (255, 255, 255), 2)
    cv2.putText(p2, f"max={np.max(anomaly_f):.4f}", (8, 50), font, 0.5, (200, 200, 200), 1)
    panels.append(p2)

    # P3: dust mask overlay
    p3 = cv2.resize(base.copy(), panel_size)
    dm_resized = cv2.resize(dm, panel_size, interpolation=cv2.INTER_NEAREST)
    dust_ol = np.zeros((*panel_size, 3), dtype=np.uint8)
    dust_ol[dm_resized > 0] = (0, 255, 255)
    p3 = cv2.addWeighted(p3, 0.6, dust_ol, 0.4, 0)
    cv2.putText(p3, "[3] Dust Mask", (8, 25), font, 0.6, (0, 255, 255), 2)
    cv2.putText(p3, f"{dust_px} px ({dust_px/dm.size*100:.1f}%)", (8, 50), font, 0.5, (200, 200, 200), 1)
    panels.append(p3)

    # P4: 舊版 top% binary (綠) + dust (黃) overlay on orig
    p4 = cv2.resize(base.copy(), panel_size)
    # dust yellow
    p4 = cv2.addWeighted(p4, 0.6, dust_ol, 0.4, 0)
    # binary green
    heat_old_resized = cv2.resize(heat_old_u8, panel_size, interpolation=cv2.INTER_NEAREST)
    green_ol = np.zeros((*panel_size, 3), dtype=np.uint8)
    green_ol[heat_old_resized > 0] = (0, 255, 0)
    p4 = cv2.addWeighted(p4, 0.8, green_ol, 0.5, 0)
    cv2.putText(p4, "[4] OLD: top 0.2% (green)", (8, 25), font, 0.55, (0, 200, 255), 2)
    cv2.putText(p4, f"thr={thr_old:.4f} {heat_old_count}px", (8, 50), font, 0.5, (200, 200, 200), 1)
    cv2.putText(p4, f"in_dust={heat_in_dust} not_dust={heat_not_dust}", (8, 75), font, 0.45, (200, 200, 200), 1)
    cv2.putText(p4, f"-> DUST(OK)", (8, 100), font, 0.6, (0, 200, 255), 2)
    panels.append(p4)

    # P5: 新版 masked heatmap overlay
    masked_norm = cv2.normalize(masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    masked_hm = cv2.applyColorMap(masked_norm, cv2.COLORMAP_JET)
    if masked_hm.shape[:2] != base.shape[:2]:
        masked_hm = cv2.resize(masked_hm, (base.shape[1], base.shape[0]))
    masked_overlay = cv2.addWeighted(base, 0.5, masked_hm, 0.5, 0)
    p5 = cv2.resize(masked_overlay, panel_size)
    cv2.putText(p5, "[5] NEW: dust zeroed", (8, 25), font, 0.55, (0, 0, 255), 2)
    cv2.putText(p5, f"max={np.max(masked):.4f}", (8, 50), font, 0.5, (200, 200, 200), 1)
    panels.append(p5)

    # P6: 新版 top% binary (綠) on orig
    p6 = cv2.resize(base.copy(), panel_size)
    p6 = cv2.addWeighted(p6, 0.6, dust_ol, 0.4, 0)
    heat_new_resized = cv2.resize(heat_new_u8, panel_size, interpolation=cv2.INTER_NEAREST)
    green_ol2 = np.zeros((*panel_size, 3), dtype=np.uint8)
    green_ol2[heat_new_resized > 0] = (0, 255, 0)
    p6 = cv2.addWeighted(p6, 0.8, green_ol2, 0.5, 0)
    cv2.putText(p6, "[6] NEW: top 0.2% (green)", (8, 25), font, 0.55, (0, 0, 255), 2)
    cv2.putText(p6, f"thr={thr_new:.4f} {heat_new_count}px", (8, 50), font, 0.5, (200, 200, 200), 1)
    cv2.putText(p6, f"-> NG(Detected)", (8, 75), font, 0.6, (0, 0, 255), 2)
    panels.append(p6)

    # 拼接 2x3
    row1 = np.hstack(panels[:3])
    row2 = np.hstack(panels[3:])

    header_h = 45
    row_w = panel_size[0] * 3
    header = np.zeros((header_h, row_w, 3), dtype=np.uint8)
    cv2.putText(header, f"Dust Filter Debug | Score={score:.4f} | Tile ({tx},{ty}) {tw}x{th}",
                (10, 32), font, 0.7, (200, 200, 200), 2)

    combined = np.vstack([header, row1, row2])
    out_path = str(out_dir / "debug_numbers.png")
    cv2.imwrite(out_path, combined)
    print(f"   圖片已存: {out_path}")

    print(f"\n{'=' * 70}")
    print("完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
