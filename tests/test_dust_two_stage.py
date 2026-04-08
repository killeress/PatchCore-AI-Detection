"""
Two-Stage Approach Prototype v2
Stage 1: Heatmap -> find hot zones (approximate location)
Stage 2: Original image -> find precise feature points -> compare with OMIT dust_mask

v2 improvements:
  - Detect dark spots AND bright spots (not just dark)
  - Use actual feature pixel mask for dust comparison (not just center point)
  - Add area_min filter to ignore tiny noise
  - Fallback: if no features found but heatmap is strong -> keep NG
  - Better visualization with per-feature detail
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re, cv2, numpy as np
from capi_config import CAPIConfig
from capi_inference import CAPIInferencer, AOIReportDefect, DEFAULT_PRODUCT_RESOLUTION
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_features_in_zone(crop_gray, crop_dust, min_area=3):
    """
    In a small crop of the original image, find feature points
    (dark spots, bright spots, or high-contrast anomalies)
    and check each against dust_mask.

    Returns list of feature dicts.
    """
    ch, cw = crop_gray.shape
    features = []

    # Local background estimation
    blur = cv2.GaussianBlur(crop_gray, (31, 31), 0)
    diff_dark = blur.astype(np.float32) - crop_gray.astype(np.float32)   # positive = darker than bg
    diff_bright = crop_gray.astype(np.float32) - blur.astype(np.float32) # positive = brighter than bg

    # Adaptive thresholds
    for diff, spot_type in [(diff_dark, "dark"), (diff_bright, "bright")]:
        diff_pos = diff[diff > 0]
        if len(diff_pos) < 10:
            continue
        thr = max(float(np.percentile(diff_pos, 90)), 3.0)
        binary = (diff >= thr).astype(np.uint8) * 255

        # Morphological cleanup
        morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_k)

        # Connected components
        n_feat, feat_labels = cv2.connectedComponents(binary, connectivity=8)

        for fid in range(1, n_feat):
            fm = feat_labels == fid
            farea = int(np.count_nonzero(fm))
            if farea < min_area:
                continue

            fys, fxs = np.where(fm)
            fcy, fcx = int(np.mean(fys)), int(np.mean(fxs))

            # Dust check: use ALL feature pixels, not just center
            feature_dust = crop_dust[fm]
            dust_overlap = int(np.count_nonzero(feature_dust > 0))
            dust_ratio = dust_overlap / farea

            # Contrast strength
            contrast = float(np.mean(diff[fm]))

            features.append({
                "local_pos": (fcx, fcy),
                "area": farea,
                "type": spot_type,
                "dust_overlap": dust_overlap,
                "dust_ratio": dust_ratio,
                "contrast": contrast,
                "mask": fm,  # for visualization
            })

    return features


def main():
    project_root = Path(__file__).resolve().parent.parent
    config = CAPIConfig.from_yaml(str(project_root / "configs" / "capi_3f.yaml"))
    config.dust_brightness_threshold = 20
    config.dust_threshold_floor = 15
    config.dust_heatmap_top_percent = 0.2
    config.dust_heatmap_iou_threshold = 0.100
    inferencer = CAPIInferencer(config)

    image_dir = project_root / "test_images"
    image_path = list(image_dir.glob("W0F*.tif"))[0]
    omit_image = cv2.imread(str(list(image_dir.glob("PINIGBI*.*"))[0]), cv2.IMREAD_UNCHANGED)
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

    anomaly_f = np.maximum(np.asarray(anomaly_map, dtype=np.float32), 0.0)
    h_am, w_am = anomaly_f.shape

    tile_img = tile.image.copy()
    if len(tile_img.shape) == 3:
        tile_gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
    else:
        tile_gray = tile_img.copy()
    tile_h, tile_w = tile_gray.shape

    # dust mask - NO dilation (ext=0), pixel-precise
    _, dust_mask, _, _ = inferencer.check_dust_or_scratch_feature(omit_crop, extension_override=0)
    dm = np.asarray(dust_mask, dtype=np.uint8)
    if len(dm.shape) == 3:
        dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
    if dm.shape != (tile_h, tile_w):
        dm = cv2.resize(dm, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)

    # ================================================================
    # Per-Region Check (with residual sub-peak rescue)
    # ================================================================
    is_dust_full, dust_mask_full, _, detail_full = inferencer.check_dust_or_scratch_feature(omit_crop)
    if is_dust_full and anomaly_map is not None:
        has_real, real_peak, overall_iou, region_details, hm_bin, _ = \
            inferencer.check_dust_per_region(
                dust_mask_full, anomaly_map,
                top_percent=config.dust_heatmap_top_percent,
                metric=config.dust_heatmap_metric,
                iou_threshold=config.dust_heatmap_iou_threshold,
            )
        print("=" * 60)
        print(f"PER_REGION (with residual rescue): has_real={has_real}")
        for r in region_details:
            res_info = f" residual={r.get('residual_ratio', 0):.2f}" if r.get('residual_ratio', 0) > 0 else ""
            print(f"  Region {r['label_id']}: area={r['area']} cov={r['coverage']:.3f} "
                  f"peak_in_dust={r['peak_in_dust']} is_dust={r['is_dust']}{res_info}")
        verdict_pr = "REAL_NG" if has_real else "DUST"
        print(f"  -> {verdict_pr}")
        print("=" * 60)

    print("=" * 60)
    print("Two-Stage Approach v2")
    print("=" * 60)

    # ================================================================
    # Stage 1: Heatmap -> hot zones
    # ================================================================
    print("\n--- Stage 1: Heatmap -> hot zones ---")
    pos = anomaly_f[anomaly_f > 0]
    hot_thr = np.percentile(pos, 95)
    hot_mask = (anomaly_f >= hot_thr).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    hot_mask = cv2.dilate(hot_mask, kernel, iterations=2)
    n_labels, labels = cv2.connectedComponents(hot_mask, connectivity=8)
    print(f"  hot zones: {n_labels - 1}")

    zones = []
    scale = tile_w / w_am
    for lid in range(1, n_labels):
        rm = labels == lid
        ys, xs = np.where(rm)
        y1, y2 = int(np.min(ys)), int(np.max(ys))
        x1, x2 = int(np.min(xs)), int(np.max(xs))
        max_score = float(np.max(anomaly_f[rm]))
        pad = 20
        ty1 = max(0, int(y1 * scale) - pad)
        ty2 = min(tile_h, int((y2 + 1) * scale) + pad)
        tx1 = max(0, int(x1 * scale) - pad)
        tx2 = min(tile_w, int((x2 + 1) * scale) + pad)
        zones.append({
            "hm_bbox": (x1, y1, x2, y2),
            "tile_bbox": (tx1, ty1, tx2, ty2),
            "max_score": max_score,
        })
        print(f"  Zone {lid}: max={max_score:.4f} tile=({tx1},{ty1})-({tx2},{ty2})")

    # ================================================================
    # Stage 2: Find features on original
    # ================================================================
    print("\n--- Stage 2: Find features on original ---")

    DUST_THRESHOLD = 0.3  # dust_ratio >= this -> DUST

    all_features = []
    for zi, zone in enumerate(zones):
        tx1, ty1, tx2, ty2 = zone["tile_bbox"]
        crop_orig = tile_gray[ty1:ty2, tx1:tx2].copy()
        crop_dust = dm[ty1:ty2, tx1:tx2].copy()

        features = find_features_in_zone(crop_orig, crop_dust, min_area=3)

        # add absolute position
        for feat in features:
            fcx, fcy = feat["local_pos"]
            feat["abs_pos"] = (tx1 + fcx, ty1 + fcy)
            feat["zone"] = zi

        print(f"\n  Zone {zi+1}: ({tx1},{ty1})-({tx2},{ty2})")
        print(f"    features: {len(features)}")
        for fi, feat in enumerate(features):
            tag = "DUST" if feat["dust_ratio"] >= DUST_THRESHOLD else "REAL"
            print(f"      F{fi+1}: ({feat['abs_pos'][0]},{feat['abs_pos'][1]}) "
                  f"{feat['type']:>6s} area={feat['area']:3d}px "
                  f"dust={feat['dust_overlap']}/{feat['area']}={feat['dust_ratio']:.2f} "
                  f"contrast={feat['contrast']:.1f} -> {tag}")

        all_features.extend(features)

    # ================================================================
    # Verdict
    # ================================================================
    real_features = [f for f in all_features if f["dust_ratio"] < DUST_THRESHOLD]
    dust_features = [f for f in all_features if f["dust_ratio"] >= DUST_THRESHOLD]

    # Fallback: no features found but heatmap is strong -> suspicious, keep NG
    if not all_features and zones and zones[0]["max_score"] > 0.7:
        verdict = "NG (fallback: no features but strong heatmap)"
        fallback = True
    elif real_features:
        verdict = "NG"
        fallback = False
    else:
        verdict = "DUST(OK)"
        fallback = False

    print(f"\n{'='*60}")
    print(f"Verdict: {verdict}")
    print(f"  total={len(all_features)} dust={len(dust_features)} real={len(real_features)}")
    if real_features:
        for f in real_features:
            print(f"  REAL: ({f['abs_pos'][0]},{f['abs_pos'][1]}) {f['type']} "
                  f"area={f['area']} dust_ratio={f['dust_ratio']:.2f}")
    print("=" * 60)

    # ================================================================
    # Per-Region Residual Rescue (NEW)
    # ================================================================
    is_dust_full, dust_mask_full, _, _ = inferencer.check_dust_or_scratch_feature(omit_crop)
    pr_has_real = False
    pr_region_details = []
    pr_hm_bin = None
    residual_map = None
    if is_dust_full and anomaly_map is not None:
        pr_has_real, pr_real_peak, pr_iou, pr_region_details, pr_hm_bin, _ = \
            inferencer.check_dust_per_region(
                dust_mask_full, anomaly_map,
                top_percent=config.dust_heatmap_top_percent,
                metric=config.dust_heatmap_metric,
                iou_threshold=config.dust_heatmap_iou_threshold,
            )
        # Build residual map for visualization
        dm_full = np.asarray(dust_mask_full, dtype=np.uint8)
        if len(dm_full.shape) == 3:
            dm_full = cv2.cvtColor(dm_full, cv2.COLOR_BGR2GRAY)
        if dm_full.shape != anomaly_f.shape:
            dm_full = cv2.resize(dm_full, (anomaly_f.shape[1], anomaly_f.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        residual_map = anomaly_f.copy()
        residual_map[dm_full > 0] = 0

    # ================================================================
    # Visualization (3 rows)
    # ================================================================
    tile_rgb = (cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB) if len(tile_img.shape) == 3
                else cv2.cvtColor(tile_img, cv2.COLOR_GRAY2RGB))
    tile_sm = cv2.resize(tile_rgb, (w_am, h_am))
    norm = cv2.normalize(anomaly_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hm_rgb = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    fig.suptitle("Dust Filtering Debug: Two-Stage + Residual Rescue\n"
                 "Row1: Heatmap & Two-Stage | Row2: Zone Zoom | Row3: Residual Rescue (NEW)",
                 fontsize=13, fontweight="bold")

    # ---- Row 1: Heatmap + Two-Stage ----
    ax = axes[0, 0]
    ax.imshow(cv2.addWeighted(tile_sm, 0.5, hm_rgb, 0.5, 0))
    for zone in zones:
        x1, y1, x2, y2 = zone["hm_bbox"]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                     linewidth=2, edgecolor="lime", facecolor="none", linestyle="--"))
    ax.set_title("Stage 1: Heatmap + Hot Zone")

    ax = axes[0, 1]
    ax.imshow(tile_sm)
    for feat in all_features:
        x, y = feat["abs_pos"]
        dx, dy = x * w_am / tile_w, y * h_am / tile_h
        is_dust = feat["dust_ratio"] >= DUST_THRESHOLD
        color = "yellow" if is_dust else "red"
        marker = "s" if feat["type"] == "bright" else "o"
        ax.plot(dx, dy, marker, color=color, markersize=8,
                markeredgecolor="white", markeredgewidth=1.5)
    ax.set_title("Stage 2: Features on Original\nyellow=DUST red=REAL")

    ax = axes[0, 2]
    ax.imshow(tile_sm)
    dm_sm = cv2.resize(dm, (w_am, h_am), interpolation=cv2.INTER_NEAREST)
    dust_ol = np.zeros((*tile_sm.shape[:2], 4))
    dust_ol[dm_sm > 0] = [1, 1, 0, 0.5]
    ax.imshow(dust_ol)
    for feat in all_features:
        x, y = feat["abs_pos"]
        dx, dy = x * w_am / tile_w, y * h_am / tile_h
        is_dust = feat["dust_ratio"] >= DUST_THRESHOLD
        color = "yellow" if is_dust else "red"
        marker = "s" if feat["type"] == "bright" else "o"
        ax.plot(dx, dy, marker, color=color, markersize=8,
                markeredgecolor="white", markeredgewidth=1.5)
    ax.set_title("Features vs Dust Mask (ext=0)")

    ax = axes[0, 3]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.85, f"Score: {score:.4f}", fontsize=16, ha="center")
    ax.text(0.5, 0.70, f"Two-Stage Features: {len(all_features)}", fontsize=13, ha="center")
    ax.text(0.5, 0.55, f"Dust: {len(dust_features)}  Real: {len(real_features)}", fontsize=14, ha="center")
    v_color = "red" if "NG" in verdict else "orange"
    v_short = "NG" if "NG" in verdict else "DUST(OK)"
    ax.text(0.5, 0.35, f"Two-Stage: {v_short}", fontsize=22, ha="center", fontweight="bold", color=v_color)
    ax.text(0.5, 0.15, "feature detection missed\nsubtle defect", fontsize=10, ha="center", color="gray")
    ax.axis("off")
    ax.set_title("Two-Stage Verdict")

    # ---- Row 2: Zoom per zone ----
    for zi, zone in enumerate(zones[:4]):
        ax = axes[1, zi]
        tx1, ty1, tx2, ty2 = zone["tile_bbox"]
        crop_rgb = tile_rgb[ty1:ty2, tx1:tx2]
        ax.imshow(crop_rgb)
        crop_dm = dm[ty1:ty2, tx1:tx2]
        dust_ol_z = np.zeros((*crop_rgb.shape[:2], 4))
        dust_ol_z[crop_dm > 0] = [1, 1, 0, 0.3]
        ax.imshow(dust_ol_z)
        zone_feats = [f for f in all_features if f["zone"] == zi]
        for feat in zone_feats:
            lx = feat["abs_pos"][0] - tx1
            ly = feat["abs_pos"][1] - ty1
            is_dust = feat["dust_ratio"] >= DUST_THRESHOLD
            color = "yellow" if is_dust else "red"
            marker = "s" if feat["type"] == "bright" else "o"
            tag = f"D{feat['dust_ratio']:.0%}" if is_dust else f"R{feat['dust_ratio']:.0%}"
            ax.plot(lx, ly, marker, color=color, markersize=14,
                    markeredgecolor="white", markeredgewidth=2)
            ax.text(lx + 8, ly - 5, f"{tag}\n{feat['type']}\n{feat['area']}px",
                    fontsize=7, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
        for feat in zone_feats:
            fm = feat["mask"]
            is_dust_f = feat["dust_ratio"] >= DUST_THRESHOLD
            feat_ol = np.zeros((*crop_rgb.shape[:2], 4))
            feat_ol[fm] = [0, 1, 0, 0.4] if not is_dust_f else [1, 0.5, 0, 0.3]
            ax.imshow(feat_ol)
        n_d = sum(1 for f in zone_feats if f["dust_ratio"] >= DUST_THRESHOLD)
        n_r = sum(1 for f in zone_feats if f["dust_ratio"] < DUST_THRESHOLD)
        ax.set_title(f"Zone {zi+1} Zoom | D={n_d} R={n_r}")
    for zi in range(len(zones), 4):
        axes[1, zi].axis("off")

    # ---- Row 3: Residual Rescue (NEW) ----
    # Col 0: Original heatmap
    ax = axes[2, 0]
    ax.imshow(hm_rgb)
    ax.set_title(f"Original Heatmap\nmax={float(np.max(anomaly_f)):.4f}")

    # Col 1: Dust mask overlay on heatmap
    ax = axes[2, 1]
    ax.imshow(hm_rgb)
    if dust_mask_full is not None:
        dm_full_vis = np.asarray(dust_mask_full, dtype=np.uint8)
        if len(dm_full_vis.shape) == 3:
            dm_full_vis = cv2.cvtColor(dm_full_vis, cv2.COLOR_BGR2GRAY)
        if dm_full_vis.shape != (h_am, w_am):
            dm_full_vis = cv2.resize(dm_full_vis, (w_am, h_am), interpolation=cv2.INTER_NEAREST)
        dust_hm_ol = np.zeros((*hm_rgb.shape[:2], 4))
        dust_hm_ol[dm_full_vis > 0] = [1, 1, 0, 0.6]
        ax.imshow(dust_hm_ol)
    ax.set_title("Heatmap + Dust Mask (yellow)\npeak may land on dust")

    # Col 2: Residual heatmap (dust masked out)
    ax = axes[2, 2]
    if residual_map is not None:
        res_max = float(np.max(residual_map))
        res_norm = cv2.normalize(residual_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        res_rgb = cv2.cvtColor(cv2.applyColorMap(res_norm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        # Gray out the masked (dust) areas
        gray_bg = np.full_like(res_rgb, 40)
        dm_full_bool = dm_full_vis > 0 if dust_mask_full is not None else np.zeros((h_am, w_am), dtype=bool)
        res_vis = res_rgb.copy()
        res_vis[dm_full_bool] = gray_bg[dm_full_bool]
        ax.imshow(res_vis)
        ax.set_title(f"Residual Heatmap (dust=gray)\nresidual max={res_max:.4f}")
    else:
        ax.text(0.5, 0.5, "N/A", fontsize=20, ha="center", va="center")
        ax.axis("off")
        ax.set_title("Residual Heatmap")

    # Col 3: Residual Rescue Verdict
    ax = axes[2, 3]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.90, "Residual Rescue Check", fontsize=14, ha="center", fontweight="bold")
    if pr_region_details:
        y_pos = 0.75
        for r in pr_region_details:
            res_r = r.get('residual_ratio', 0)
            res_thr = config.dust_residual_ratio
            tag = "RESCUED -> REAL_NG" if (not r['is_dust'] and r['peak_in_dust']) else \
                  ("DUST" if r['is_dust'] else "REAL_NG")
            tag_color = "green" if "REAL" in tag else "orange"
            ax.text(0.5, y_pos,
                    f"Region {r['label_id']}: cov={r['coverage']:.2f}  "
                    f"peak_in_dust={r['peak_in_dust']}",
                    fontsize=10, ha="center")
            y_pos -= 0.08
            if res_r > 0:
                ax.text(0.5, y_pos,
                        f"residual={res_r:.2f} (thr={res_thr})  -> {tag}",
                        fontsize=11, ha="center", fontweight="bold", color=tag_color)
            else:
                ax.text(0.5, y_pos, f"-> {tag}",
                        fontsize=11, ha="center", fontweight="bold", color=tag_color)
            y_pos -= 0.12
    pr_verdict = "REAL_NG" if pr_has_real else "DUST(OK)"
    pr_color = "red" if pr_has_real else "orange"
    ax.text(0.5, 0.25, f"Final: {pr_verdict}", fontsize=26, ha="center",
            fontweight="bold", color=pr_color)
    if pr_has_real and not real_features:
        ax.text(0.5, 0.08, "Two-Stage missed it,\nResidual Rescue caught it!",
                fontsize=11, ha="center", color="green", fontweight="bold")
    ax.axis("off")
    ax.set_title("Residual Rescue Verdict (NEW)")

    plt.tight_layout()
    out_dir = project_root / "test_dust_comparison_output"
    out_dir.mkdir(exist_ok=True)
    out = str(out_dir / "debug_two_stage.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nChart saved: {out}")


if __name__ == "__main__":
    main()
