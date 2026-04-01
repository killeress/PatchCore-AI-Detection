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
import re, cv2, numpy as np
from pathlib import Path
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
    config = CAPIConfig.from_yaml("configs/capi_3f.yaml")
    config.dust_brightness_threshold = 20
    config.dust_threshold_floor = 15
    config.dust_heatmap_top_percent = 0.2
    config.dust_heatmap_iou_threshold = 0.100
    inferencer = CAPIInferencer(config)

    image_dir = Path("./test_images")
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
    # Visualization
    # ================================================================
    tile_rgb = (cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB) if len(tile_img.shape) == 3
                else cv2.cvtColor(tile_img, cv2.COLOR_GRAY2RGB))
    tile_sm = cv2.resize(tile_rgb, (w_am, h_am))
    norm = cv2.normalize(anomaly_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hm_rgb = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle("Two-Stage Approach v2\n"
                 "Stage1: heatmap->hot zone | Stage2: original->features (dark+bright) | dust_mask(ext=0)->verdict",
                 fontsize=13, fontweight="bold")

    # Row 1 Col 0: heatmap + zones
    ax = axes[0, 0]
    ax.imshow(cv2.addWeighted(tile_sm, 0.5, hm_rgb, 0.5, 0))
    for zone in zones:
        x1, y1, x2, y2 = zone["hm_bbox"]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                     linewidth=2, edgecolor="lime", facecolor="none", linestyle="--"))
    ax.set_title("Stage 1: Heatmap + Hot Zone")

    # Row 1 Col 1: original + features
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
    ax.set_title("Stage 2: Features on Original\no=dark s=bright | yellow=DUST red=REAL")

    # Row 1 Col 2: dust mask + features
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
    ax.set_title("Features vs Dust Mask (ext=0)\nred NOT in yellow = REAL")

    # Row 1 Col 3: verdict
    ax = axes[0, 3]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.85, f"Score: {score:.4f}", fontsize=16, ha="center")
    ax.text(0.5, 0.70, f"Features: {len(all_features)} (dark+bright)", fontsize=13, ha="center")
    ax.text(0.5, 0.55, f"Dust: {len(dust_features)}  Real: {len(real_features)}", fontsize=14, ha="center")
    v_color = "red" if "NG" in verdict else "orange"
    v_short = "NG" if "NG" in verdict else "DUST(OK)"
    ax.text(0.5, 0.35, f"-> {v_short}", fontsize=28, ha="center", fontweight="bold", color=v_color)
    if fallback:
        ax.text(0.5, 0.15, "(fallback: strong heatmap)", fontsize=10, ha="center", color="gray")
    else:
        ax.text(0.5, 0.15, "pixel-level comparison\next=0, no diffusion problem", fontsize=10, ha="center", color="gray")
    ax.axis("off")
    ax.set_title("Verdict")

    # Row 2: Zoom per zone with detailed feature overlay
    for zi, zone in enumerate(zones[:4]):
        ax = axes[1, zi]
        tx1, ty1, tx2, ty2 = zone["tile_bbox"]
        crop_rgb = tile_rgb[ty1:ty2, tx1:tx2]
        ax.imshow(crop_rgb)

        # dust mask overlay
        crop_dm = dm[ty1:ty2, tx1:tx2]
        dust_ol_z = np.zeros((*crop_rgb.shape[:2], 4))
        dust_ol_z[crop_dm > 0] = [1, 1, 0, 0.3]
        ax.imshow(dust_ol_z)

        # features in this zone
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

        # also show feature pixel masks
        for feat in zone_feats:
            fm = feat["mask"]
            is_dust = feat["dust_ratio"] >= DUST_THRESHOLD
            feat_ol = np.zeros((*crop_rgb.shape[:2], 4))
            feat_ol[fm] = [0, 1, 0, 0.4] if not is_dust else [1, 0.5, 0, 0.3]
            ax.imshow(feat_ol)

        n_d = sum(1 for f in zone_feats if f["dust_ratio"] >= DUST_THRESHOLD)
        n_r = sum(1 for f in zone_feats if f["dust_ratio"] < DUST_THRESHOLD)
        ax.set_title(f"Zone {zi+1} Zoom | D={n_d} R={n_r}\ngreen highlight = REAL feature pixels")

    for zi in range(len(zones), 4):
        axes[1, zi].axis("off")

    plt.tight_layout()
    out = "test_dust_comparison_output/debug_two_stage.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nChart saved: {out}")


if __name__ == "__main__":
    main()
