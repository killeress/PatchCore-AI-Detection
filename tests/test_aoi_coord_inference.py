"""
用 AOI 機台產品座標推論 → 與 Server process_panel 完全一致的流程
支援兩種座標來源:
  1. TXT 檔名嵌入座標 (如 W0F00000_X1651Y710.txt)
  2. 直接傳入座標列表

Pipeline (與 server 一致):
  Phase 1  : preprocess_image
  Phase 1.5: _create_aoi_coord_tiles (以 AOI 座標為中心切 512x512)
  Phase 2  : run_inference (PatchCore) 或 _detect_bright_spots (skip_files)
  Phase 3  : 灰塵交叉驗證 (OMIT)
  Phase 4  : 炸彈比對 + peak 計算
  Output   : save_panel_heatmaps (overview + tile detail + edge defect)
"""
import sys
import re
import time
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer, AOIReportDefect, TileInfo, DEFAULT_PRODUCT_RESOLUTION
from capi_edge_cv import EdgeDefect
from capi_heatmap import HeatmapManager


def parse_coord_from_txt(txt_path: Path) -> Tuple[str, int, int]:
    """
    從 TXT 檔名解析圖片前綴與座標
    例: W0F00000_X1651Y710.txt → ("W0F00000", 1651, 710)
    """
    stem = txt_path.stem  # W0F00000_X1651Y710
    m = re.match(r'^(.+?)_X(\d+)Y(\d+)$', stem)
    if not m:
        raise ValueError(f"無法從檔名解析座標: {txt_path.name} (預期格式: PREFIX_X數字Y數字.txt)")
    return m.group(1), int(m.group(2)), int(m.group(3))


def find_image_for_prefix(image_dir: Path, prefix: str) -> Optional[Path]:
    """在目錄中找到對應前綴的圖片"""
    for ext in ['*.tif', '*.tiff', '*.bmp', '*.png', '*.jpg']:
        for f in image_dir.glob(ext):
            if f.name.startswith(prefix) and not f.name.startswith('PINIGBI') and not f.name.startswith('OMIT0000'):
                return f
    return None


def find_omit_image(image_dir: Path) -> Optional[np.ndarray]:
    """自動搜尋 PINIGBI/OMIT0000 圖片"""
    for pattern in ["PINIGBI*.*", "OMIT0000*.*"]:
        matches = list(image_dir.glob(pattern))
        if matches:
            img = cv2.imread(str(matches[0]), cv2.IMREAD_UNCHANGED)
            if img is not None:
                print(f"✅ OMIT 圖片: {matches[0].name}")
                return img
    return None


def test_aoi_coord(
    image_path: str = None,
    image_dir: str = "./test_images",
    aoi_coords: list = None,
    config_path: str = "configs/capi_3f.yaml",
    product_resolution: tuple = None,
    glass_id: str = "TEST_AOI",
    heatmap_dir: str = "./test_heatmaps",
    config_overrides: dict = None,
    disable_edge: bool = False,
):
    """
    以 AOI 機檢座標執行推論，與 Server process_panel 一致的完整 pipeline。

    如果 image_path 和 aoi_coords 未指定，會自動從 image_dir 中的 TXT 檔解析。
    """
    image_dir = Path(image_dir)
    config = CAPIConfig.from_yaml(config_path)

    # 覆寫參數 (與 Server DB config_params 一致)
    if config_overrides:
        for key, val in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, val)
                print(f"  📝 Config override: {key} = {val}")

    inferencer = CAPIInferencer(config)
    heatmap_mgr = HeatmapManager(base_dir=heatmap_dir)

    # 關閉 CV Edge (與 production 一致，邊緣檢測由 DB 控制)
    if disable_edge:
        inferencer.edge_inspector.config.enabled = False
        print("  🚫 CV Edge 已關閉")

    if product_resolution is None:
        product_resolution = DEFAULT_PRODUCT_RESOLUTION

    # ----------------------------------------------------------------
    # 自動解析: 從 TXT 檔名取得座標，找到對應圖片
    # ----------------------------------------------------------------
    if image_path is None or aoi_coords is None:
        txt_files = list(image_dir.glob("*_X*Y*.txt"))
        if not txt_files:
            print("❌ 找不到座標 TXT 檔 (格式: PREFIX_X數字Y數字.txt)")
            return
        print(f"📂 找到 {len(txt_files)} 個座標 TXT 檔")

        # 按圖片前綴分組
        coord_map = {}  # {prefix: [(px, py, defect_code), ...]}
        for txt in txt_files:
            prefix, px, py = parse_coord_from_txt(txt)
            coord_map.setdefault(prefix, []).append((px, py, "TXT"))
            print(f"  📍 {txt.name} → prefix={prefix}, coord=({px},{py})")

        # 找對應圖片
        tasks = []  # [(image_path, [(px, py, code), ...])]
        for prefix, coords in coord_map.items():
            img_path = find_image_for_prefix(image_dir, prefix)
            if img_path:
                tasks.append((img_path, coords))
                print(f"  🖼️ {prefix} → {img_path.name} ({len(coords)} 個座標)")
            else:
                print(f"  ⚠️ {prefix} → 找不到對應圖片，跳過")
    else:
        image_path = Path(image_path)
        tasks = [(image_path, [(px, py, "MANUAL") for px, py in aoi_coords])]

    if not tasks:
        print("❌ 沒有可推論的任務")
        return

    # ----------------------------------------------------------------
    # OMIT 圖片
    # ----------------------------------------------------------------
    omit_image = find_omit_image(image_dir)
    omit_overexposed = False
    if omit_image is not None:
        is_oe, oe_mean, oe_ratio, oe_detail = inferencer.check_omit_overexposure(omit_image)
        if is_oe:
            print(f"⚠️ OMIT 過曝: {oe_detail} → 灰塵檢測停用")
            omit_image = None
            omit_overexposed = True
        else:
            print(f"✅ OMIT 曝光正常: {oe_detail}")
    else:
        print("⚠️ 未找到 OMIT 圖片，灰塵檢測停用")

    # ----------------------------------------------------------------
    # 對每張圖片執行完整 Pipeline
    # ----------------------------------------------------------------
    all_results = []

    for img_path, coords in tasks:
        print(f"\n{'='*60}")
        print(f"🖼️ 圖片: {img_path.name}")
        print(f"📍 AOI 座標: {[(px, py) for px, py, _ in coords]}")
        print(f"{'='*60}")

        # === Phase 1: 預處理 ===
        result = inferencer.preprocess_image(img_path)
        if result is None:
            print("❌ 預處理失敗")
            continue

        print(f"  Otsu: {result.otsu_bounds}, Raw: {result.raw_bounds}")
        print(f"  Grid tiles: {len(result.tiles)}")

        # === Phase 1.5: AOI 座標切塊 (與 server _create_aoi_coord_tiles 一致) ===
        aoi_defects = [
            AOIReportDefect(
                defect_code=code,
                product_x=px,
                product_y=py,
                image_prefix=inferencer._get_image_prefix(img_path.name),
            )
            for px, py, code in coords
        ]

        full_image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if full_image is None:
            print("❌ 無法讀取圖片")
            continue

        new_tiles, edge_defs = inferencer._create_aoi_coord_tiles(
            full_image, result, aoi_defects, product_resolution
        )
        result.tiles.extend(new_tiles)
        print(f"  AOI tiles: {len(new_tiles)}, 邊緣 defects: {len(edge_defs)}")

        # 移除 grid tiles，只保留 AOI coord tiles (與 server grid_tiling_enabled=False 一致)
        original_count = len(result.tiles)
        result.tiles = [t for t in result.tiles if t.is_aoi_coord_tile]
        removed = original_count - len(result.tiles)
        if removed > 0:
            print(f"  ⏭️ 移除 {removed} 個 grid tiles，保留 {len(result.tiles)} 個 AOI tiles")

        # === 邊緣 defects: CV Edge 處理 (與 server 一致，disable_edge 時跳過) ===
        img_h, img_w = full_image.shape[:2]
        for edef in (edge_defs if not disable_edge else []):
            img_x, img_y = inferencer._map_aoi_coords(
                edef.product_x, edef.product_y,
                result.raw_bounds, product_resolution
            )
            roi_half = config.tile_size // 2
            rx1 = max(0, img_x - roi_half)
            ry1 = max(0, img_y - roi_half)
            rx2 = min(img_w, img_x + roi_half)
            ry2 = min(img_h, img_y + roi_half)
            roi = full_image[ry1:ry2, rx1:rx2]

            cv_detected = False
            roi_stats = {"max_diff": 0, "max_area": 0, "threshold": 0, "min_area": 0}
            if roi.size > 0 and getattr(inferencer, 'edge_inspector', None):
                try:
                    edge_results, roi_stats = inferencer.edge_inspector.inspect_roi(
                        roi, offset_x=rx1, offset_y=ry1,
                        otsu_bounds=result.otsu_bounds,
                    )
                    if edge_results:
                        unified_bbox = (rx1, ry1, rx2 - rx1, ry2 - ry1)
                        total_area = sum(ed.area for ed in edge_results)
                        worst_diff = max(ed.max_diff for ed in edge_results)
                        merged = EdgeDefect(
                            side="aoi_edge",
                            area=total_area,
                            bbox=unified_bbox,
                            center=(img_x, img_y),
                            max_diff=worst_diff,
                            threshold_used=roi_stats.get("threshold", 0),
                            min_area_used=roi_stats.get("min_area", 0),
                        )
                        result.edge_defects.append(merged)
                        cv_detected = True
                        print(f"  🔍 Edge ({edef.defect_code}) @ ({img_x},{img_y}): CV NG (area={total_area}, diff={worst_diff})")
                except Exception as e:
                    print(f"  ⚠️ CV Edge error: {e}")

            if not cv_detected:
                ok_defect = EdgeDefect(
                    side="aoi_coord_ok",
                    area=roi_stats.get("max_area", 0),
                    bbox=(rx1, ry1, rx2 - rx1, ry2 - ry1),
                    center=(img_x, img_y),
                    max_diff=roi_stats.get("max_diff", 0),
                    is_cv_ok=True,
                    threshold_used=roi_stats.get("threshold", 0),
                    min_area_used=roi_stats.get("min_area", 0),
                )
                result.edge_defects.append(ok_defect)
                print(f"  ✅ Edge ({edef.defect_code}) @ ({img_x},{img_y}): CV OK")

        # === Phase 2: GPU 推論 (與 server 一致) ===
        img_prefix = inferencer._get_image_prefix(img_path.name)
        is_skip_file = config.should_skip_file(img_path.name)

        if is_skip_file:
            # B0F 等黑圖: 二值化亮點偵測
            print(f"  💡 {img_path.name} (skip_file) → 二值化偵測")
            anomaly_tiles = []
            for tile in result.tiles:
                score, anomaly_map = inferencer._detect_bright_spots(tile)
                if score <= 0:
                    tile.is_aoi_coord_below_threshold = True
                anomaly_tiles.append((tile, score, anomaly_map))
            result.anomaly_tiles = anomaly_tiles
        else:
            # PatchCore 推論
            target_inferencer = inferencer._get_inferencer_for_prefix(img_prefix)
            target_threshold = inferencer._get_threshold_for_prefix(img_prefix)
            print(f"  🔥 PatchCore 推論: prefix={img_prefix}, threshold={target_threshold}")

            if target_inferencer is None:
                print(f"  ❌ 無可用模型 (prefix={img_prefix})，跳過")
                continue

            infer_start = time.time()
            result = inferencer.run_inference(
                result,
                inferencer=target_inferencer,
                threshold=target_threshold,
            )
            print(f"  ✅ 推論完成 ({time.time() - infer_start:.2f}s)")

        # === Phase 3: 灰塵交叉驗證 (與 server _dust_check_one 一致) ===
        if not is_skip_file and result.anomaly_tiles and omit_image is not None and not omit_overexposed:
            print(f"  🧹 灰塵檢測中...")
            for tile, score, anomaly_map in result.anomaly_tiles:
                tx, ty, tw, th = tile.x, tile.y, tile.width, tile.height
                oh, ow = omit_image.shape[:2]
                if tx < ow and ty < oh:
                    x2 = min(tx + tw, ow)
                    y2 = min(ty + th, oh)
                    omit_crop = omit_image[ty:y2, tx:x2]
                    tile.omit_crop_image = omit_crop.copy()

                    is_dust, dust_mask, bright_ratio, detail_text = inferencer.check_dust_or_scratch_feature(omit_crop)
                    tile.dust_mask = dust_mask
                    tile.dust_bright_ratio = bright_ratio

                    iou = 0.0
                    top_pct = config.dust_heatmap_top_percent
                    metric_mode = config.dust_heatmap_metric

                    if is_dust and anomaly_map is not None:
                        has_real_defect, real_peak_yx, overall_iou, region_details, heatmap_binary, region_labels = \
                            inferencer.check_dust_per_region(
                                dust_mask, anomaly_map,
                                top_percent=top_pct,
                                metric=metric_mode,
                                iou_threshold=config.dust_heatmap_iou_threshold,
                            )
                        iou = overall_iou
                        tile.dust_heatmap_iou = iou
                        tile.dust_region_details = region_details
                        tile.dust_heatmap_binary = heatmap_binary
                        if region_details:
                            tile.dust_region_max_cov = max(r["coverage"] for r in region_details)

                        dust_regions = [r for r in region_details if r["is_dust"]]
                        real_regions = [r for r in region_details if not r["is_dust"]]

                        if has_real_defect:
                            tile.is_suspected_dust_or_scratch = False
                            detail_text += f" PER_REGION: {len(real_regions)}real+{len(dust_regions)}dust -> REAL_NG"
                            if real_peak_yx is not None:
                                amap_h, amap_w = anomaly_map.shape[:2]
                                tile.anomaly_peak_y = tile.y + int(real_peak_yx[0] * tile.height / amap_h)
                                tile.anomaly_peak_x = tile.x + int(real_peak_yx[1] * tile.width / amap_w)
                        else:
                            # Two-stage second opinion
                            if config.dust_two_stage_enabled:
                                dust_mask_no_ext = None
                                if omit_crop is not None:
                                    _, dust_mask_no_ext, _, _ = inferencer.check_dust_or_scratch_feature(
                                        omit_crop, extension_override=0)
                                ts_has_real, ts_peak_yx, ts_features, ts_detail = \
                                    inferencer.check_dust_two_stage(
                                        tile.image, anomaly_map,
                                        dust_mask_no_ext if dust_mask_no_ext is not None else dust_mask,
                                        score,
                                    )
                                if ts_has_real:
                                    tile.is_suspected_dust_or_scratch = False
                                    detail_text += f" PER_REGION: 0real+{len(dust_regions)}dust -> {ts_detail}"
                                    if ts_peak_yx is not None:
                                        amap_h, amap_w = anomaly_map.shape[:2]
                                        tile.anomaly_peak_y = tile.y + int(ts_peak_yx[0] * tile.height / amap_h)
                                        tile.anomaly_peak_x = tile.x + int(ts_peak_yx[1] * tile.width / amap_w)
                                else:
                                    tile.is_suspected_dust_or_scratch = True
                                    detail_text += f" PER_REGION: 0real+{len(dust_regions)}dust -> {ts_detail}"
                            else:
                                tile.is_suspected_dust_or_scratch = True
                                detail_text += f" PER_REGION: 0real+{len(dust_regions)}dust -> DUST"

                        try:
                            if config.dust_two_stage_enabled and 'ts_features' in locals():
                                dm_dbg = dust_mask_no_ext if 'dust_mask_no_ext' in locals() and dust_mask_no_ext is not None else dust_mask
                                tile.dust_iou_debug_image = inferencer.generate_two_stage_debug_image(
                                    tile.image, anomaly_map, dm_dbg,
                                    ts_features, tile.is_suspected_dust_or_scratch,
                                )
                            else:
                                tile.dust_iou_debug_image = inferencer.generate_dust_iou_debug_image(
                                    tile.image, anomaly_map, dust_mask,
                                    heatmap_binary, iou, top_pct,
                                    tile.is_suspected_dust_or_scratch,
                                    region_details=region_details,
                                    region_labels=region_labels,
                                )
                        except Exception:
                            pass
                    elif is_dust:
                        tile.is_suspected_dust_or_scratch = True
                        detail_text += " (no heatmap, marked as dust)"
                    else:
                        detail_text += " NO_DUST -> REAL_NG"

                    tile.dust_detail_text = detail_text
                    icon = "🧹" if tile.is_suspected_dust_or_scratch else "🔴"
                    print(f"    {icon} Tile@({tx},{ty}) Score:{score:.4f} → {detail_text}")

        # === Phase 4: Peak 計算 + 炸彈比對 (與 server 一致) ===
        if result.anomaly_tiles and result.raw_bounds is not None:
            for tile, score, anomaly_map in result.anomaly_tiles:
                if anomaly_map is not None and anomaly_map.size > 0:
                    try:
                        amap_h, amap_w = anomaly_map.shape[:2]
                        if tile.is_bright_spot_detection:
                            ys, xs = np.where(anomaly_map > 0.5)
                            if len(xs) > 0:
                                cx = int(np.mean(xs))
                                cy = int(np.mean(ys))
                            else:
                                cx, cy = amap_w // 2, amap_h // 2
                            tile.anomaly_peak_x = tile.x + int(cx * tile.width / amap_w)
                            tile.anomaly_peak_y = tile.y + int(cy * tile.height / amap_h)
                        elif tile.anomaly_peak_x < 0:
                            peak_local = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
                            tile.anomaly_peak_y = tile.y + int(peak_local[0] * tile.height / amap_h)
                            tile.anomaly_peak_x = tile.x + int(peak_local[1] * tile.width / amap_w)
                    except Exception as e:
                        tile.anomaly_peak_x, tile.anomaly_peak_y = tile.center

        # === 結果摘要 ===
        effective = [t for t in result.anomaly_tiles
                     if not getattr(t[0], 'is_aoi_coord_below_threshold', False)]
        ng_count = len([t for t in effective if not t[0].is_suspected_dust_or_scratch])
        dust_count = len([t for t in effective if t[0].is_suspected_dust_or_scratch])
        below_count = len(result.anomaly_tiles) - len(effective)
        edge_count = len(result.edge_defects)

        print(f"\n  📊 結果: NG={ng_count}, DUST={dust_count}, Below_THR={below_count}, Edge={edge_count}")

        all_results.append(result)

    # ----------------------------------------------------------------
    # Output: save_panel_heatmaps (與 server 一致)
    # ----------------------------------------------------------------
    if all_results:
        print(f"\n🎨 儲存結果圖...")
        heatmap_result = heatmap_mgr.save_panel_heatmaps(
            glass_id=glass_id,
            results=all_results,
            inferencer=inferencer,
            save_overview=True,
            save_tile_detail=True,
            omit_image=omit_image,
        )
        print(f"\n📂 輸出目錄: {heatmap_result['dir']}")
        for f in heatmap_result['files']:
            print(f"  📄 {f}")
    else:
        print("\n⚠️ 無結果可輸出")

    print("\n✅ 完成")


if __name__ == "__main__":
    test_aoi_coord(
        image_dir="./test_images",
        disable_edge=True,
        config_overrides={
            "dust_brightness_threshold": 20,
            "dust_threshold_floor": 15,
            "dust_heatmap_top_percent": 0.2,
            "dust_heatmap_iou_threshold": 0.100,
        },
    )
