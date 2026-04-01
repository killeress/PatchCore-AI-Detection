"""
Dust Filter 改善方案 Prototype 比較測試
========================================
針對 issue01.png 案例，比較三種 dust filtering 方法：
  - 現有流程：top% 二值化 → connectedComponents → 逐區域 COV
  - 方法 4：OMIT 訊號相減後再做 top% 二值化
  - 方法 6：在原始 heatmap 上做 peak_local_max，逐 peak 獨立判定

使用方式：
  python test_dust_methods_comparison.py
"""
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import peak_local_max

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer


# ============================================================
# 方法 4 改良: Dust Mask 遮罩 (先 mask 再二值化)
# ============================================================
def method4_dust_mask_then_binarize(
    anomaly_map: np.ndarray,
    dust_mask: np.ndarray,
    top_percent: float = 0.2,
    iou_threshold: float = 0.1,
) -> dict:
    """
    將 anomaly_map 中 dust_mask 覆蓋的像素直接歸零，
    對剩餘區域做 top% 二值化 + connectedComponents 判定。
    """
    anomaly_f = np.asarray(anomaly_map, dtype=np.float32)
    anomaly_f = np.maximum(anomaly_f, 0.0)

    dm = np.asarray(dust_mask, dtype=np.uint8)
    if len(dm.shape) == 3:
        dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
    if dm.shape != anomaly_f.shape:
        dm = cv2.resize(dm, (anomaly_f.shape[1], anomaly_f.shape[0]),
                        interpolation=cv2.INTER_NEAREST)

    dust_bool = dm > 0

    # 核心：灰塵區域歸零
    masked = anomaly_f.copy()
    masked[dust_bool] = 0

    # 對剩餘區域做 top% 二值化
    positive_values = masked[masked > 0]
    if len(positive_values) == 0:
        return {
            "method": "4_mask",
            "has_real_defect": False,
            "regions": [],
            "num_regions": 0,
            "detail": "no positive values after masking",
            "masked_anomaly": masked,
            "heatmap_binary": None,
        }

    threshold = np.percentile(positive_values, 100 - top_percent)
    heat_bool = masked >= threshold
    heatmap_binary = (heat_bool.astype(np.uint8)) * 255

    # connectedComponents
    num_labels, labels = cv2.connectedComponents(heatmap_binary, connectivity=8)

    regions = []
    has_real_defect = False

    for label_id in range(1, num_labels):
        region_mask = labels == label_id
        region_area = np.count_nonzero(region_mask)

        region_vals = anomaly_f[region_mask]  # 原始分數
        max_score = float(np.max(region_vals))
        masked_vals = masked[region_mask]
        max_masked = float(np.max(masked_vals))

        # 這些區域已經在非灰塵區域，COV 理論上很低
        region_dust_overlap = np.count_nonzero(region_mask & dust_bool)
        coverage = float(region_dust_overlap / region_area) if region_area > 0 else 0.0
        is_dust = coverage >= iou_threshold

        regions.append({
            "label_id": label_id,
            "area": region_area,
            "coverage": coverage,
            "is_dust": is_dust,
            "max_score": max_score,
            "max_masked": max_masked,
        })

        if not is_dust:
            has_real_defect = True

    return {
        "method": "4_mask",
        "has_real_defect": has_real_defect,
        "regions": regions,
        "num_regions": len(regions),
        "masked_anomaly": masked,
        "heatmap_binary": heatmap_binary,
    }


# ============================================================
# 方法 6: Peak detection on raw heatmap
# ============================================================
def method6_peak_detection(
    anomaly_map: np.ndarray,
    dust_mask: np.ndarray,
    iou_threshold: float = 0.1,
    min_distance: int = 15,
    threshold_rel: float = 0.3,
    peak_radius: int = 20,
    score_floor: float = 0.1,
) -> dict:
    """
    在原始 anomaly_map 上做 peak_local_max，
    對每個 peak 建立局部區域，獨立與 dust_mask 交叉驗證。
    """
    anomaly_f = np.asarray(anomaly_map, dtype=np.float32)
    anomaly_f = np.maximum(anomaly_f, 0.0)

    dm = np.asarray(dust_mask, dtype=np.uint8)
    if len(dm.shape) == 3:
        dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
    if dm.shape != anomaly_f.shape:
        dm = cv2.resize(dm, (anomaly_f.shape[1], anomaly_f.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    dust_bool = dm > 0

    # peak_local_max
    max_val = np.max(anomaly_f)
    if max_val <= 0:
        return {"method": "6_peak_detection", "has_real_defect": True, "peaks": [], "detail": "no positive values"}

    coords = peak_local_max(
        anomaly_f,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        num_peaks=20,
    )

    h, w = anomaly_f.shape[:2]
    peaks = []
    has_real_defect = False

    for (py, px) in coords:
        score = float(anomaly_f[py, px])
        if score < score_floor:
            continue

        # 建立局部區域 (以 peak 為中心)
        y1, y2 = max(0, py - peak_radius), min(h, py + peak_radius)
        x1, x2 = max(0, px - peak_radius), min(w, px + peak_radius)

        # 局部區域用 flood-fill 概念：anomaly > peak_score * 0.5
        local_anomaly = anomaly_f[y1:y2, x1:x2]
        local_dust = dust_bool[y1:y2, x1:x2]
        peak_region = local_anomaly > (score * 0.3)

        region_area = np.count_nonzero(peak_region)
        if region_area == 0:
            continue

        overlap = np.count_nonzero(peak_region & local_dust)
        coverage = float(overlap / region_area)

        is_dust = coverage >= iou_threshold

        peaks.append({
            "position": (int(py), int(px)),
            "score": score,
            "region_area": region_area,
            "dust_overlap": overlap,
            "coverage": coverage,
            "is_dust": is_dust,
        })

        if not is_dust:
            has_real_defect = True

    return {
        "method": "6_peak_detection",
        "has_real_defect": has_real_defect,
        "peaks": peaks,
        "num_peaks": len(peaks),
        "min_distance": min_distance,
        "threshold_rel": threshold_rel,
        "peak_radius": peak_radius,
    }


# ============================================================
# 現有流程 (baseline)
# ============================================================
def method_current(
    inferencer: CAPIInferencer,
    dust_mask: np.ndarray,
    anomaly_map: np.ndarray,
    top_percent: float = 0.2,
    iou_threshold: float = 0.1,
) -> dict:
    """呼叫現有 check_dust_per_region"""
    has_real_defect, real_peak_yx, overall_iou, region_details, heatmap_binary, labels = \
        inferencer.check_dust_per_region(
            dust_mask, anomaly_map,
            top_percent=top_percent,
            metric="coverage",
            iou_threshold=iou_threshold,
        )

    return {
        "method": "current",
        "has_real_defect": has_real_defect,
        "overall_iou": overall_iou,
        "regions": region_details,
        "num_regions": len(region_details),
        "heatmap_binary": heatmap_binary,
    }


# ============================================================
# 視覺化比較圖
# ============================================================
def generate_comparison_image(
    anomaly_map: np.ndarray,
    dust_mask: np.ndarray,
    tile_image: np.ndarray,
    current_result: dict,
    m4_result: dict,
    m6_result: dict,
    output_path: str,
    score: float = 0.0,
    iou_threshold: float = 0.1,
):
    """產生三種方法的比較圖 (follow 原始 capi_heatmap.py 風格)"""
    anomaly_f = np.asarray(anomaly_map, dtype=np.float32)
    anomaly_f = np.maximum(anomaly_f, 0.0)
    h, w = anomaly_f.shape[:2]

    # --- 原圖 base (follow capi_heatmap.py) ---
    if tile_image is not None:
        base = tile_image.copy()
        if len(base.shape) == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        base = np.zeros((h, w, 3), dtype=np.uint8)

    # --- heatmap overlay (follow capi_heatmap.py L324-327) ---
    norm_map = cv2.normalize(anomaly_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    if heatmap_color.shape[:2] != base.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (base.shape[1], base.shape[0]))
    anomaly_color = cv2.addWeighted(base, 0.5, heatmap_color, 0.5, 0)

    panel_size = (400, 400)
    panels = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    pw, ph = panel_size

    # --- Panel 1: 原始 Heatmap ---
    p1 = cv2.resize(anomaly_color, panel_size)
    cv2.putText(p1, "[1] Anomaly Heatmap", (8, 28), font, 0.7, (255, 255, 255), 2)
    amax = float(np.max(anomaly_f))
    cv2.putText(p1, f"max={amax:.3f}", (8, 55), font, 0.55, (200, 200, 200), 1)
    panels.append(p1)

    # --- Panel 2: Dust Mask (follow capi_heatmap.py L360-365: OMIT + dust yellow overlay) ---
    dm = np.asarray(dust_mask, dtype=np.uint8)
    if len(dm.shape) == 3:
        dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
    if dm.shape != base.shape[:2]:
        dm = cv2.resize(dm, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
    dm_overlay = base.copy()
    dust_colored = np.zeros_like(dm_overlay)
    dust_colored[dm > 0] = (0, 255, 255)  # 黃色 = 灰塵
    dm_overlay = cv2.addWeighted(dm_overlay, 0.6, dust_colored, 0.4, 0)
    p2 = cv2.resize(dm_overlay, panel_size)
    dust_px = np.count_nonzero(dm)
    cv2.putText(p2, "[2] Dust Mask (yellow)", (8, 28), font, 0.65, (0, 255, 255), 2)
    cv2.putText(p2, f"dust pixels: {dust_px}", (8, 55), font, 0.55, (200, 200, 200), 1)
    panels.append(p2)

    # --- Panel 3: 現有流程 binary (綠=top% region, 黃=dust) ---
    p3_base = base.copy()
    # 先疊 dust mask (黃)
    dust_ol3 = np.zeros_like(p3_base)
    dust_ol3[dm > 0] = (0, 255, 255)
    p3_base = cv2.addWeighted(p3_base, 0.6, dust_ol3, 0.4, 0)
    # 再疊 binary (綠)
    if current_result.get("heatmap_binary") is not None:
        cb_resized = cv2.resize(current_result["heatmap_binary"],
                                (p3_base.shape[1], p3_base.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        binary_ol = np.zeros_like(p3_base)
        binary_ol[cb_resized > 0] = (0, 255, 0)
        p3_base = cv2.addWeighted(p3_base, 0.7, binary_ol, 0.5, 0)
    p3 = cv2.resize(p3_base, panel_size)
    verdict = "NG" if current_result["has_real_defect"] else "DUST(OK)"
    color = (0, 0, 255) if current_result["has_real_defect"] else (0, 200, 255)
    cv2.putText(p3, f"[3] Current: {verdict}", (8, 28), font, 0.7, color, 2)
    n_regions = current_result.get("num_regions", 0)
    cv2.putText(p3, f"top 0.2% -> {n_regions} region (green)", (8, 55), font, 0.5, (200, 200, 200), 1)
    for i, r in enumerate(current_result.get("regions", [])):
        tag = "DUST" if r["is_dust"] else "REAL"
        cv2.putText(p3, f"R{i+1}: {tag} COV={r['coverage']:.3f} score={r['max_score']:.3f}",
                    (8, 80 + i * 25), font, 0.5, (200, 200, 200), 1)
    panels.append(p3)

    # --- Panel 4: 方法 4 masked heatmap (灰塵區域歸零後的 heatmap) ---
    if m4_result.get("masked_anomaly") is not None:
        masked = m4_result["masked_anomaly"]
        mmax = float(np.max(masked))
        if mmax > 0:
            m_norm = cv2.normalize(masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            m_norm = np.zeros_like(masked, dtype=np.uint8)
        m_heatmap = cv2.applyColorMap(m_norm, cv2.COLORMAP_JET)
        if m_heatmap.shape[:2] != base.shape[:2]:
            m_heatmap = cv2.resize(m_heatmap, (base.shape[1], base.shape[0]))
        m_overlay = cv2.addWeighted(base, 0.5, m_heatmap, 0.5, 0)
        p4 = cv2.resize(m_overlay, panel_size)
    else:
        p4 = np.zeros((ph, pw, 3), dtype=np.uint8)
        mmax = 0
    verdict4 = "NG" if m4_result["has_real_defect"] else "DUST(OK)"
    color4 = (0, 0, 255) if m4_result["has_real_defect"] else (0, 200, 255)
    cv2.putText(p4, f"[4] M4 Masked: {verdict4}", (8, 28), font, 0.65, color4, 2)
    n4 = m4_result.get("num_regions", 0)
    cv2.putText(p4, f"dust zeroed, then top% -> {n4} region(s)", (8, 55), font, 0.5, (200, 200, 200), 1)
    for i, r in enumerate(m4_result.get("regions", [])):
        tag = "DUST" if r["is_dust"] else "REAL"
        cv2.putText(p4, f"R{i+1}: {tag} COV={r['coverage']:.3f} score={r['max_score']:.3f}",
                    (8, 80 + i * 25), font, 0.5, (200, 200, 200), 1)
    panels.append(p4)

    # --- Panel 5: 方法 4 binary (綠=masked top% region, 黃=dust) ---
    p5_base = base.copy()
    dust_ol5 = np.zeros_like(p5_base)
    dust_ol5[dm > 0] = (0, 255, 255)
    p5_base = cv2.addWeighted(p5_base, 0.6, dust_ol5, 0.4, 0)
    if m4_result.get("heatmap_binary") is not None:
        m4b_resized = cv2.resize(m4_result["heatmap_binary"],
                                 (p5_base.shape[1], p5_base.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        binary_ol5 = np.zeros_like(p5_base)
        binary_ol5[m4b_resized > 0] = (0, 255, 0)
        p5_base = cv2.addWeighted(p5_base, 0.7, binary_ol5, 0.5, 0)
    p5 = cv2.resize(p5_base, panel_size)
    cv2.putText(p5, "[5] M4 Binary", (8, 28), font, 0.65, color4, 2)
    cv2.putText(p5, "green=non-dust region yellow=dust", (8, 55), font, 0.45, (200, 200, 200), 1)
    panels.append(p5)

    # --- Panel 6: 方法 6 peak overlay ---
    p6 = cv2.resize(anomaly_color.copy(), panel_size)
    scale_y = ph / h
    scale_x = pw / w
    for pk in m6_result.get("peaks", []):
        py_disp = int(pk["position"][0] * scale_y)
        px_disp = int(pk["position"][1] * scale_x)
        if pk["is_dust"]:
            pk_color = (0, 200, 255)  # 黃 = dust
            label = f"D {pk['score']:.2f}"
        else:
            pk_color = (0, 0, 255)    # 紅 = real
            label = f"R {pk['score']:.2f}"
        cv2.circle(p6, (px_disp, py_disp), 12, pk_color, 2)
        cv2.putText(p6, label, (px_disp + 14, py_disp + 5), font, 0.4, pk_color, 1)
    verdict6 = "NG" if m6_result["has_real_defect"] else "DUST(OK)"
    color6 = (0, 0, 255) if m6_result["has_real_defect"] else (0, 200, 255)
    n6 = m6_result.get("num_peaks", 0)
    dust_n = sum(1 for p in m6_result.get("peaks", []) if p["is_dust"])
    real_n = sum(1 for p in m6_result.get("peaks", []) if not p["is_dust"])
    cv2.putText(p6, f"[6] M6 Peaks: {verdict6}", (8, 28), font, 0.65, color6, 2)
    cv2.putText(p6, f"{n6} peaks (D={dust_n} R={real_n})", (8, 55), font, 0.5, (200, 200, 200), 1)
    cv2.putText(p6, "yellow=DUST  red=REAL", (8, 80), font, 0.5, (200, 200, 200), 1)
    panels.append(p6)

    # --- Header (follow capi_heatmap.py style) ---
    row_w = pw * 3
    header_h = 50
    header = np.zeros((header_h, row_w, 3), dtype=np.uint8)
    header_text = f"Score: {score:.4f} | Dust Filter Comparison | iou_thr={iou_threshold}"
    cv2.putText(header, header_text, (10, 35), font, 0.8, (200, 200, 200), 2)

    # 拼接 2x3
    row1 = np.hstack(panels[:3])
    row2 = np.hstack(panels[3:])
    combined = np.vstack([header, row1, row2])
    cv2.imwrite(output_path, combined)
    print(f"📊 比較圖已存: {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    image_dir = Path("./test_images")
    config_path = "configs/capi_3f.yaml"
    output_dir = Path("./test_dust_comparison_output")
    output_dir.mkdir(exist_ok=True)

    # Config (與 test_aoi_coord 一致)
    config = CAPIConfig.from_yaml(config_path)
    config.dust_brightness_threshold = 20
    config.dust_threshold_floor = 15
    config.dust_heatmap_top_percent = 0.2
    config.dust_heatmap_iou_threshold = 0.100

    inferencer = CAPIInferencer(config)

    # 找圖片
    image_path = None
    for ext in ['*.tif', '*.tiff', '*.bmp', '*.png']:
        for f in image_dir.glob(ext):
            if not f.name.startswith('PINIGBI') and not f.name.startswith('OMIT0000'):
                image_path = f
                break
        if image_path:
            break

    if image_path is None:
        print("❌ 找不到測試圖片")
        return

    print(f"🖼️ 圖片: {image_path.name}")

    # OMIT
    omit_image = None
    for pattern in ["PINIGBI*.*", "OMIT0000*.*"]:
        matches = list(image_dir.glob(pattern))
        if matches:
            omit_image = cv2.imread(str(matches[0]), cv2.IMREAD_UNCHANGED)
            print(f"🔍 OMIT: {matches[0].name}")
            break

    if omit_image is None:
        print("❌ 找不到 OMIT 圖片")
        return

    # 座標
    txt_files = list(image_dir.glob("*_X*Y*.txt"))
    if not txt_files:
        print("❌ 找不到座標 TXT")
        return

    import re
    m = re.match(r'^(.+?)_X(\d+)Y(\d+)$', txt_files[0].stem)
    px, py = int(m.group(2)), int(m.group(3))
    print(f"📍 座標: ({px}, {py})")

    # ================================================================
    # Phase 1: 預處理 + 推論 (取得 anomaly_map)
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 1: 預處理 + 推論")
    print("=" * 60)

    result = inferencer.preprocess_image(image_path)
    if result is None:
        print("❌ 預處理失敗")
        return

    from capi_inference import AOIReportDefect, DEFAULT_PRODUCT_RESOLUTION
    full_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    img_prefix = inferencer._get_image_prefix(image_path.name)

    aoi_defects = [AOIReportDefect(
        defect_code="TXT", product_x=px, product_y=py,
        image_prefix=img_prefix,
    )]

    new_tiles, edge_defs = inferencer._create_aoi_coord_tiles(
        full_image, result, aoi_defects, DEFAULT_PRODUCT_RESOLUTION
    )
    result.tiles.extend(new_tiles)
    result.tiles = [t for t in result.tiles if t.is_aoi_coord_tile]

    # PatchCore 推論
    target_inferencer = inferencer._get_inferencer_for_prefix(img_prefix)
    target_threshold = inferencer._get_threshold_for_prefix(img_prefix)
    print(f"  🔥 PatchCore: prefix={img_prefix}, threshold={target_threshold}")

    t0 = time.time()
    result = inferencer.run_inference(result, inferencer=target_inferencer, threshold=target_threshold)
    print(f"  ✅ 推論完成 ({time.time() - t0:.2f}s)")

    if not result.anomaly_tiles:
        print("❌ 沒有異常 tile")
        return

    # ================================================================
    # Phase 2: 取得 dust_mask + anomaly_map，跑三種方法
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Dust Filter 方法比較")
    print("=" * 60)

    for idx, (tile, score, anomaly_map) in enumerate(result.anomaly_tiles):
        print(f"\n--- Tile {idx}: ({tile.x},{tile.y}) Score={score:.4f} ---")

        if anomaly_map is None:
            print("  ⚠️ 無 anomaly_map，跳過")
            continue

        # 裁切 OMIT 對應區域
        tx, ty, tw, th = tile.x, tile.y, tile.width, tile.height
        oh, ow = omit_image.shape[:2]
        x2 = min(tx + tw, ow)
        y2 = min(ty + th, oh)
        omit_crop = omit_image[ty:y2, tx:x2]

        # Step A: 灰塵偵測 (共用)
        is_dust, dust_mask, bright_ratio, detail_text = inferencer.check_dust_or_scratch_feature(omit_crop)
        print(f"  灰塵偵測: is_dust={is_dust}, {detail_text}")

        if not is_dust:
            print("  ✅ OMIT 未偵測到灰塵，三種方法都會判 REAL_NG")
            continue

        # ============================================================
        # 現有流程
        # ============================================================
        print("\n  [現有流程]")
        r_current = method_current(
            inferencer, dust_mask, anomaly_map,
            top_percent=config.dust_heatmap_top_percent,
            iou_threshold=config.dust_heatmap_iou_threshold,
        )
        print(f"    判定: {'REAL_NG ✅' if r_current['has_real_defect'] else 'DUST(OK) ❌ 漏放'}")
        print(f"    區域數: {r_current['num_regions']}")
        for r in r_current.get("regions", []):
            tag = "DUST" if r["is_dust"] else "REAL"
            print(f"      Region {r['label_id']}: {tag} area={r['area']} COV={r['coverage']:.3f} score={r['max_score']:.3f}")

        # ============================================================
        # 方法 4: Dust Mask 遮罩 (先 mask 再二值化)
        # ============================================================
        print("\n  [方法 4: Dust Mask 遮罩]")
        r_m4 = method4_dust_mask_then_binarize(
            anomaly_map, dust_mask,
            top_percent=config.dust_heatmap_top_percent,
            iou_threshold=config.dust_heatmap_iou_threshold,
        )
        verdict = "REAL_NG ✅" if r_m4["has_real_defect"] else "DUST(OK) ❌"
        print(f"    {verdict}, 區域數={r_m4.get('num_regions', 0)}")
        for r in r_m4.get("regions", []):
            tag = "DUST" if r["is_dust"] else "REAL"
            print(f"      Region {r['label_id']}: {tag} area={r['area']} COV={r['coverage']:.3f} score={r['max_score']:.3f}")

        # ============================================================
        # 方法 6: Peak Detection
        # ============================================================
        print("\n  [方法 6: Peak Detection]")
        # 測試多組參數
        for min_dist in [10, 15, 20]:
            for thr_rel in [0.2, 0.3, 0.5]:
                r_m6 = method6_peak_detection(
                    anomaly_map, dust_mask,
                    iou_threshold=config.dust_heatmap_iou_threshold,
                    min_distance=min_dist,
                    threshold_rel=thr_rel,
                    peak_radius=20,
                )
                verdict = "REAL_NG ✅" if r_m6["has_real_defect"] else "DUST(OK) ❌"
                n_peaks = r_m6.get("num_peaks", 0)
                dust_peaks = sum(1 for p in r_m6.get("peaks", []) if p["is_dust"])
                real_peaks = sum(1 for p in r_m6.get("peaks", []) if not p["is_dust"])
                print(f"    min_dist={min_dist}, thr_rel={thr_rel}: {verdict}, peaks={n_peaks} (dust={dust_peaks}, real={real_peaks})")
                for pk in r_m6.get("peaks", []):
                    tag = "DUST" if pk["is_dust"] else "REAL"
                    print(f"      Peak@{pk['position']}: {tag} score={pk['score']:.3f} COV={pk['coverage']:.3f}")

        # ============================================================
        # 產生比較圖 (用預設參數)
        # ============================================================
        r_m4_default = method4_dust_mask_then_binarize(
            anomaly_map, dust_mask,
            top_percent=config.dust_heatmap_top_percent,
            iou_threshold=config.dust_heatmap_iou_threshold,
        )
        r_m6_default = method6_peak_detection(
            anomaly_map, dust_mask,
            iou_threshold=config.dust_heatmap_iou_threshold,
            min_distance=15,
            threshold_rel=0.3,
            peak_radius=20,
        )
        output_path = str(output_dir / f"comparison_tile{idx}.png")
        generate_comparison_image(
            anomaly_map, dust_mask, tile.image,
            r_current, r_m4_default, r_m6_default,
            output_path,
            score=score,
            iou_threshold=config.dust_heatmap_iou_threshold,
        )

    print("\n" + "=" * 60)
    print("✅ 比較測試完成")
    print(f"📂 輸出目錄: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
