"""
單張圖片推論腳本（含灰塵比對）
對指定圖片進行 PatchCore 異常檢測推論，並交叉比對 PINIGBI/OMIT 圖片進行灰塵判定
輸出：
  1. 標記後的全圖 (_result.jpg)
  2. 每個異常 tile 的組合圖：Original | Heatmap | OMIT Crop | Dust Mask | IOU Debug (_tile_N.jpg)
"""
import sys
import time
import glob
from pathlib import Path

import cv2
import numpy as np

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer
from capi_heatmap import build_region_zoom_panels


def find_omit_image(image_path: Path) -> Path | None:
    """
    自動搜尋與推論圖片同目錄的 PINIGBI / OMIT0000 圖片
    支援帶空格和時間戳的檔名 (e.g. "PINIGBI _133222.tif")
    """
    parent = image_path.parent
    # 搜尋所有可能的灰塵檢查圖片
    for pattern in ["PINIGBI*.*", "OMIT0000*.*"]:
        matches = list(parent.glob(pattern))
        if matches:
            return matches[0]
    return None


def generate_tile_combined_image(
    tile,
    score,
    anomaly_map,
    omit_image,
    inferencer,
    tile_index: int,
) -> np.ndarray:
    """
    產生單個異常 tile 的組合視覺化圖
    排列與 capi_heatmap.py 完全同步: Original | Heatmap | OMIT Crop | Dust Mask | IOU Debug
    """
    tile_size = 512
    panels = []

    # --- Panel 1: Original Tile ---
    orig = tile.image.copy()
    if len(orig.shape) == 2:
        orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    elif len(orig.shape) == 3 and orig.shape[2] == 1:
        orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    orig = cv2.resize(orig, (tile_size, tile_size))

    # --- Panel 2: Heatmap Overlay ---
    if anomaly_map is not None:
        norm_map = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        heatmap_color = cv2.resize(heatmap_color, (tile_size, tile_size))
        heatmap_panel = cv2.addWeighted(orig, 0.5, heatmap_color, 0.5, 0)
    else:
        heatmap_panel = orig.copy()
        cv2.putText(heatmap_panel, "No Heatmap", (150, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)

    # --- 取得 OMIT Crop ---
    omit_crop = None
    if omit_image is not None:
        tx, ty, tw, th = tile.x, tile.y, tile.width, tile.height
        oh, ow = omit_image.shape[:2]
        if tx < ow and ty < oh:
            x2 = min(tx + tw, ow)
            y2 = min(ty + th, oh)
            omit_crop = omit_image[ty:y2, tx:x2].copy()

    # --- Panel 3: OMIT Crop ---
    if omit_crop is not None:
        omit_panel = omit_crop.copy()
        if len(omit_panel.shape) == 2:
            omit_panel = cv2.cvtColor(omit_panel, cv2.COLOR_GRAY2BGR)
        elif len(omit_panel.shape) == 3 and omit_panel.shape[2] == 1:
            omit_panel = cv2.cvtColor(omit_panel, cv2.COLOR_GRAY2BGR)
        omit_panel = cv2.resize(omit_panel, (tile_size, tile_size))
    else:
        omit_panel = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        cv2.putText(omit_panel, "No OMIT", (170, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)

    # --- 灰塵偵測與後續分析 ---
    dust_mask = None
    iou = 0.0
    metric_name = "COV"
    detail_text = ""
    is_dust_final = False
    dust_iou_debug = None
    region_details = None
    heatmap_binary = None
    is_bomb = getattr(tile, 'is_bomb', False)
    bomb_code = getattr(tile, 'bomb_defect_code', '')

    if omit_crop is not None:
        is_dust, dust_mask, bright_ratio, detail_text = inferencer.check_dust_or_scratch_feature(omit_crop)
        
        # heatmap coverage/iou cross-validation
        top_pct = inferencer.config.dust_heatmap_top_percent
        metric_mode = inferencer.config.dust_heatmap_metric
        metric_name = "COV" if metric_mode == "coverage" else "IOU"
        if is_dust and anomaly_map is not None:
            iou_threshold = inferencer.config.dust_heatmap_iou_threshold
            has_real_defect, real_peak_yx, iou, region_details, heatmap_binary, region_labels = \
                inferencer.check_dust_per_region(
                    dust_mask, anomaly_map,
                    top_percent=top_pct,
                    metric=metric_mode,
                    iou_threshold=iou_threshold,
                )
            tile.dust_heatmap_iou = iou
            tile.dust_region_details = region_details
            tile.dust_heatmap_binary = heatmap_binary
            if region_details:
                tile.dust_region_max_cov = max(r["coverage"] for r in region_details)

            dust_regions = [r for r in region_details if r["is_dust"]]
            real_regions = [r for r in region_details if not r["is_dust"]]
            if not has_real_defect:
                is_dust_final = True
                tile.is_suspected_dust_or_scratch = True
                detail_text += f" PER_REGION: 0real+{len(dust_regions)}dust -> DUST"
            else:
                detail_text += f" PER_REGION: {len(real_regions)}real+{len(dust_regions)}dust -> REAL_NG"

            try:
                dust_iou_debug = inferencer.generate_dust_iou_debug_image(
                    tile.image, anomaly_map, dust_mask,
                    heatmap_binary, iou, top_pct,
                    tile.is_suspected_dust_or_scratch,
                    region_details=region_details,
                    region_labels=region_labels,
                )
            except Exception:
                pass

        elif is_dust:
            is_dust_final = True
            tile.is_suspected_dust_or_scratch = True
            detail_text += " (no heatmap, marked as dust)"
        else:
            detail_text += " NO_DUST -> REAL_NG"
        
        tile.dust_detail_text = detail_text
        print(f"   {'🧹' if is_dust_final else '🔴'} Tile#{tile_index} → {detail_text}")

    # --- Panel 4: Dust Mask ---
    if dust_mask is not None and omit_crop is not None:
        dust_panel = omit_panel.copy()
        dust_colored = np.zeros_like(dust_panel)
        dust_resized = cv2.resize(dust_mask, (tile_size, tile_size))
        dust_colored[dust_resized > 0] = (0, 255, 255)
        dust_panel = cv2.addWeighted(dust_panel, 0.6, dust_colored, 0.4, 0)

    else:
        dust_panel = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        cv2.putText(dust_panel, "No Dust Data", (140, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)

    # --- Panel 5: IOU Debug ---
    if dust_iou_debug is not None:
        iou_debug_panel = cv2.resize(dust_iou_debug, (tile_size, tile_size))
    else:
        iou_debug_panel = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        cv2.putText(iou_debug_panel, "No IOU Debug", (130, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)

    # --- Panel 6+: Region Zoom (逐區域放大，最多 3 張) ---
    zoom_results = []
    if is_dust_final and heatmap_binary is not None:
        iou_threshold = inferencer.config.dust_heatmap_iou_threshold
        zoom_results = build_region_zoom_panels(
            heatmap_binary, dust_mask, region_details,
            tile_size=tile_size,
            metric_name=metric_name,
            iou_threshold=iou_threshold,
        )

    # --- 橫向拼接（不在面板上畫標籤）---
    panels = [orig, heatmap_panel, omit_panel, dust_panel, iou_debug_panel]
    labels = ["Original", "Heatmap", "OMIT Crop", f"Dust Mask (Overall{metric_name}:{iou:.3f})", f"{metric_name} Debug (G=Dust R=RealNG B=DustOnly)"]
    for zoom_img, zoom_label in zoom_results:
        panels.append(zoom_img)
        labels.append(zoom_label)
    composite = np.hstack(panels)
    comp_h, comp_w = composite.shape[:2]

    # --- 底部獨立標籤列（不蓋到面板內容）---
    label_h = 40
    label_bar = np.zeros((label_h, comp_w, 3), dtype=np.uint8)
    for i, lbl in enumerate(labels):
        lx = i * tile_size + 10
        cv2.putText(label_bar, lbl, (lx, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # --- 頂部 Header ---
    header_h = 60
    header = np.zeros((header_h, comp_w, 3), dtype=np.uint8)

    score_threshold = inferencer.threshold
    iou_threshold = inferencer.config.dust_heatmap_iou_threshold

    if is_bomb:
        verdict = f"BOMB: {bomb_code} (Filtered as OK)"
        verdict_color = (255, 0, 255)  # 洋紅色
    elif is_dust_final:
        dust_rcov = getattr(tile, 'dust_region_max_cov', 0.0)
        verdict = f"DUST (Filtered as OK) | Region{metric_name}:{dust_rcov:.3f}>={iou_threshold:.3f}"
        verdict_color = (0, 200, 255)
    elif score >= score_threshold:
        verdict = "NG (Detected)"
        verdict_color = (0, 0, 255)
    else:
        verdict = f"Score < THR ({score_threshold:.4f})"
        verdict_color = (0, 255, 255)

    header_text = f"Score: {score:.4f} | {verdict}"
    cv2.putText(header, header_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, verdict_color, 2)

    if detail_text:
        detail_line = detail_text[:120].replace('\u2192', '->').replace('\u2190', '<-')
        detail_line = detail_line.replace(f'>={metric_name}_THR', f'>={iou_threshold:.3f}')
        detail_line = detail_line.replace(f'<{metric_name}_THR', f'<{iou_threshold:.3f}')
    else:
        detail_line = f"Tile#{tile_index} | Unknown"
        
    cv2.putText(header, detail_line, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    final = np.vstack([header, composite, label_bar])

    return final


def run_single_inference(
    image_path: str,
    config_path: str = "configs/capi_3f.yaml",
    omit_path: str = None,
):
    """
    對單張圖片執行推論並輸出結果圖片（含灰塵比對）
    
    Args:
        image_path: 推論圖片路徑
        config_path: 配置檔路徑
        omit_path: PINIGBI/OMIT 圖片路徑 (若為 None 則自動搜尋同目錄)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ 圖片不存在: {image_path}")
        return

    print(f"📸 圖片: {image_path.name}")
    print(f"📁 配置: {config_path}")
    print("=" * 60)

    # 1. 載入配置
    config = CAPIConfig.from_yaml(config_path)
    print(f"✅ 配置已載入: {config.machine_id}")

    # 2. 建立推論器 (會自動載入對應模型)
    print("🔄 正在載入模型...")
    start_time = time.time()
    inferencer = CAPIInferencer(config)
    load_time = time.time() - start_time
    print(f"✅ 模型載入完成 ({load_time:.2f}s)")

    # 3. 載入 OMIT/PINIGBI 圖片
    omit_image = None
    if omit_path:
        omit_file = Path(omit_path)
    else:
        omit_file = find_omit_image(image_path)
    
    if omit_file and omit_file.exists():
        omit_image = cv2.imread(str(omit_file), cv2.IMREAD_UNCHANGED)
        if omit_image is not None:
            print(f"✅ OMIT/PINIGBI 圖片已載入: {omit_file.name}")
            # 過曝檢查
            is_overexposed, oe_mean, oe_ratio, oe_detail = inferencer.check_omit_overexposure(omit_image)
            if is_overexposed:
                print(f"⚠️ OMIT 過曝: {oe_detail}")
                print("   → 灰塵檢測將被跳過")
                omit_image = None  # 過曝時禁用灰塵檢測
            else:
                print(f"✅ OMIT 曝光正常: {oe_detail}")
        else:
            print(f"⚠️ 無法載入 OMIT 圖片: {omit_file}")
    else:
        print("⚠️ 未找到 OMIT/PINIGBI 圖片，將跳過灰塵檢測")

    # 4. 預處理
    print("\n🔄 預處理中...")
    result = inferencer.preprocess_image(image_path)
    if result is None:
        print("❌ 預處理失敗")
        return

    print(f"   - 圖片尺寸: {result.image_size}")
    print(f"   - Otsu 邊界: {result.otsu_bounds}")
    print(f"   - 排除區域: {len(result.exclusion_regions)}")
    print(f"   - 有效 Tiles: {result.processed_tile_count}")
    print(f"   - 排除 Tiles: {result.excluded_tile_count}")

    # 5. 推論
    print("\n🔥 推論中...")
    img_prefix = inferencer._get_image_prefix(image_path.name)
    target_inferencer = inferencer._get_inferencer_for_prefix(img_prefix)
    target_threshold = inferencer._get_threshold_for_prefix(img_prefix)
    print(f"   - 前綴: {img_prefix}")
    print(f"   - 閾值: {target_threshold}")

    if target_inferencer is None:
        print("❌ 找不到對應的模型")
        return

    infer_start = time.time()
    result = inferencer.run_inference(
        result,
        inferencer=target_inferencer,
        threshold=target_threshold,
    )
    infer_time = time.time() - infer_start
    print(f"✅ 推論完成 ({infer_time:.2f}s)")

    # 6. 結果摘要
    print("\n" + "=" * 60)
    if result.anomaly_tiles:
        print(f"🔴 判定: NG — 發現 {len(result.anomaly_tiles)} 個異常 tile")
        for tile, score, _ in result.anomaly_tiles:
            print(f"   - Tile@({tile.x},{tile.y}) Score: {score:.4f}")
    else:
        print("🟢 判定: OK — 無異常")

    # 7. 灰塵比對 + 組合圖輸出
    output_paths = []
    
    if result.anomaly_tiles:
        print(f"\n🧹 灰塵比對中 (OMIT: {'有' if omit_image is not None else '無'})...")
        
        for idx, (tile, score, anomaly_map) in enumerate(result.anomaly_tiles):
            # 產生組合圖 (Original | Heatmap | OMIT Crop | Dust Mask | IOU Debug)
            combined = generate_tile_combined_image(
                tile, score, anomaly_map, omit_image, inferencer, idx
            )
            
            tile_output = image_path.parent / f"{image_path.stem}_tile_{idx}.jpg"
            cv2.imwrite(str(tile_output), combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
            output_paths.append(tile_output)
            print(f"   ✅ Tile #{idx} 組合圖已儲存: {tile_output.name}")

    # 8. 視覺化全圖輸出 (含灰塵標記)
    print("\n🎨 生成視覺化結果...")
    vis = inferencer.visualize_inference_result(image_path, result)

    output_path = image_path.parent / f"{image_path.stem}_result.jpg"
    cv2.imwrite(str(output_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    output_paths.append(output_path)
    print(f"✅ 結果全圖已儲存: {output_path}")
    print(f"   - 輸出尺寸: {vis.shape[1]}x{vis.shape[0]}")

    total_time = time.time() - start_time
    print(f"\n⏱️ 總耗時: {total_time:.2f}s")
    print(f"\n📂 輸出檔案:")
    for p in output_paths:
        print(f"   📄 {p}")

    return output_paths


if __name__ == "__main__":
    # 預設圖片路徑
    default_image = r"R0F00000_133232.tif"

    if len(sys.argv) > 1:
        img = sys.argv[1]
    else:
        img = default_image

    # 第二個參數可選：OMIT/PINIGBI 圖片路徑
    omit = sys.argv[2] if len(sys.argv) > 2 else None

    run_single_inference(img, omit_path=omit)
