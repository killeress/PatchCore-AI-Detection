"""
STANDARD 炸彈診斷工具

在指定的 STANDARD 圖片上：
1. 載入 AI 模型，執行推論
2. 標記 YAML 設定中的炸彈座標
3. 標記 AD 偵測到的異常 tile
4. 分析炸彈是否被正確匹配、是否被灰塵過濾誤殺
5. 輸出結果到 PNG 檔案

使用方式:
    python diagnose_bomb.py <image_path>
    python diagnose_bomb.py <image_path> --model-id GN140JCAL010S
    python diagnose_bomb.py <image_path> --resolution 1920x1200
"""

import os
os.environ["TRUST_REMOTE_CODE"] = "1"

import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# 確保模組路徑
sys.path.insert(0, str(Path(__file__).parent))

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer, resolve_product_resolution, DEFAULT_PRODUCT_RESOLUTION


# ── 設定 ──────────────────────────────────────────
CONFIG_PATH = "configs/capi_3f.yaml"
MODEL_PATH = "./model.pt"
THRESHOLD = 0.56   # Debug 用低門檻


def parse_resolution_arg(resolution_str: str):
    """解析 WxH 或 W,H 格式的解析度字串"""
    for sep in ['x', 'X', ',']:
        if sep in resolution_str:
            parts = resolution_str.split(sep)
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))
    raise ValueError(f"無法解析解析度: {resolution_str} (格式應為 WxH 或 W,H)")


def main():
    parser = argparse.ArgumentParser(description="STANDARD 炸彈診斷工具")
    parser.add_argument("image_path", help="圖片檔案路徑")
    parser.add_argument("--model-id", default="", help="機種 ID (第六碼判定解析度，例如 GN140JCAL010S)")
    parser.add_argument("--resolution", default="", help="手動指定解析度 WxH (例如 1920x1200)")
    parser.add_argument("--config", default=CONFIG_PATH, help=f"設定檔路徑 (default: {CONFIG_PATH})")
    parser.add_argument("--model", default=MODEL_PATH, help=f"模型路徑 (default: {MODEL_PATH})")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help=f"異常門檻 (default: {THRESHOLD})")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"❌ 檔案不存在: {image_path}")
        sys.exit(1)

    # ── 1. 載入設定 & 模型 ──
    print("\n[1/5] 載入設定與模型...")
    config = CAPIConfig.from_yaml(args.config)

    # 決定產品解析度 (優先順序: --resolution > --model-id > 預設值)
    if args.resolution:
        product_resolution = parse_resolution_arg(args.resolution)
    elif args.model_id:
        product_resolution = resolve_product_resolution(args.model_id, config.model_resolution_map)
    else:
        product_resolution = DEFAULT_PRODUCT_RESOLUTION
    
    PRODUCT_W, PRODUCT_H = product_resolution

    print("=" * 70)
    print(f"🔍 STANDARD 炸彈診斷工具")
    print(f"   圖片: {image_path}")
    print(f"   產品解析度: {PRODUCT_W}x{PRODUCT_H}")
    if args.model_id:
        print(f"   機種 ID: {args.model_id} (第六碼 '{args.model_id[5] if len(args.model_id) >= 6 else '?'}')")
    print("=" * 70)

    inferencer = CAPIInferencer(config, model_path=args.model, threshold=args.threshold)
    print(f"  ✅ 模型已載入: {args.model} (threshold={args.threshold})")
    print(f"  ✅ 炸彈數量: {len(config.bomb_defects)} 組設定")

    # ── 2. 預處理 & 推論 ──
    print("\n[2/5] 預處理 & 推論...")
    result = inferencer.preprocess_image(image_path)
    if result is None:
        print("❌ 無法預處理圖片")
        sys.exit(1)

    result = inferencer.run_inference(result)
    print(f"  ✅ 推論完成: {len(result.tiles)} 個 tile, {len(result.anomaly_tiles)} 個異常")

    # ── 3. 載入原圖並繪製 ──
    print("\n[3/5] 繪製診斷圖...")
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        print("❌ 無法載入圖片")
        sys.exit(1)

    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif len(vis.shape) == 3 and vis.shape[2] == 1:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    raw_bounds = result.raw_bounds
    if raw_bounds is None:
        raw_bounds = (0, 0, vis.shape[1], vis.shape[0])

    x_start, y_start, x_end, y_end = raw_bounds
    scale_x = (x_end - x_start) / PRODUCT_W
    scale_y = (y_end - y_start) / PRODUCT_H

    # 顏色定義
    BOMB_COLOR = (255, 0, 255)      # 洋紅 = 炸彈設定座標
    AD_NG_COLOR = (0, 0, 255)       # 紅色 = AD 偵測到的真 NG
    AD_DUST_COLOR = (0, 165, 255)   # 橘色 = AD 偵測到但被判灰塵
    MATCH_COLOR = (0, 255, 0)       # 綠色 = 炸彈匹配成功
    OTSU_COLOR = (255, 200, 0)      # 藍色 = Otsu 邊界

    # 畫 Otsu 邊界
    ox1, oy1, ox2, oy2 = result.otsu_bounds
    cv2.rectangle(vis, (ox1, oy1), (ox2, oy2), OTSU_COLOR, 3)

    # ── 畫 AD 偵測結果 ──
    tolerance = config.bomb_match_tolerance
    img_tol_x = int(tolerance * scale_x)
    img_tol_y = int(tolerance * scale_y)

    ad_tiles_info = []  # 收集所有偵測結果

    for tile, score, anomaly_map in result.anomaly_tiles:
        # 取得熱力圖峰值
        if anomaly_map is not None and anomaly_map.size > 0:
            try:
                peak_local = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
                # anomaly_map 尺寸可能和 tile 不同，需要縮放
                amap_h, amap_w = anomaly_map.shape[:2]
                peak_x = tile.x + int(peak_local[1] * tile.width / amap_w)
                peak_y = tile.y + int(peak_local[0] * tile.height / amap_h)
            except Exception:
                peak_x = tile.x + tile.width // 2
                peak_y = tile.y + tile.height // 2
        else:
            peak_x = tile.x + tile.width // 2
            peak_y = tile.y + tile.height // 2

        # 嘗試炸彈匹配 (不做灰塵判定，直接座標比對)
        img_prefix = image_path.stem
        is_bomb, bomb_code = inferencer.check_bomb_match(
            img_prefix, peak_x, peak_y, raw_bounds,
            anomaly_map=anomaly_map, product_resolution=product_resolution
        )

        # 反推產品座標
        prod_x = int((peak_x - x_start) / scale_x) if scale_x > 0 else 0
        prod_y = int((peak_y - y_start) / scale_y) if scale_y > 0 else 0

        ad_tiles_info.append({
            "tile": tile, "score": score,
            "peak_x": peak_x, "peak_y": peak_y,
            "prod_x": prod_x, "prod_y": prod_y,
            "is_bomb": is_bomb, "bomb_code": bomb_code,
            "anomaly_map": anomaly_map,
        })

        # 繪製 tile 框
        if is_bomb:
            color = MATCH_COLOR
        else:
            color = AD_NG_COLOR  # 未被匹配的異常
        cv2.rectangle(vis, (tile.x, tile.y),
                      (tile.x + tile.width, tile.y + tile.height), color, 4)
        # 畫峰值點
        cv2.circle(vis, (peak_x, peak_y), 15, color, -1)
        cv2.line(vis, (peak_x - 30, peak_y), (peak_x + 30, peak_y), (0, 0, 255), 2)
        cv2.line(vis, (peak_x, peak_y - 30), (peak_x, peak_y + 30), (0, 0, 255), 2)

        # 標籤
        label = f"S:{score:.2f}"
        if is_bomb:
            label += f" BOMB({bomb_code})"
        label += f" P({prod_x},{prod_y})"
        cv2.putText(vis, label, (tile.x + 5, tile.y + tile.height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # ── 畫 YAML 設定的炸彈座標 ──
    bomb_points_img = []  # 收集轉換後的炸彈座標
    img_prefix = image_path.stem

    for bomb in config.bomb_defects:
        if not (img_prefix == bomb.image_prefix or
                img_prefix.startswith(bomb.image_prefix + "_")):
            continue

        if bomb.defect_type == "point":
            for idx, coord in enumerate(bomb.coordinates):
                img_bx = int(coord[0] * scale_x + x_start)
                img_by = int(coord[1] * scale_y + y_start)
                bomb_points_img.append((img_bx, img_by, coord[0], coord[1], bomb.defect_code))

                # 找最近的 AD tile
                nearest_dist = float('inf')
                nearest_ad = None
                is_matched = False
                for ad in ad_tiles_info:
                    dx = abs(img_bx - ad["peak_x"])
                    dy = abs(img_by - ad["peak_y"])
                    dist = (dx**2 + dy**2) ** 0.5
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_ad = ad
                    if dx <= img_tol_x and dy <= img_tol_y:
                        is_matched = True

                pt_color = MATCH_COLOR if is_matched else BOMB_COLOR

                # 十字 + 圓
                size = 50
                cv2.circle(vis, (img_bx, img_by), size, pt_color, 4)
                cv2.line(vis, (img_bx - size, img_by), (img_bx + size, img_by), pt_color, 3)
                cv2.line(vis, (img_bx, img_by - size), (img_bx, img_by + size), pt_color, 3)

                # 容錯範圍框 (虛線效果: 用短線段代替)
                tl = (img_bx - img_tol_x, img_by - img_tol_y)
                br = (img_bx + img_tol_x, img_by + img_tol_y)
                cv2.rectangle(vis, tl, br, pt_color, 2)

                # label
                status = "✅ MATCHED" if is_matched else "❌ NOT MATCHED"
                label = f"BOMB#{idx+1} ({coord[0]},{coord[1]}) {status}"
                cv2.putText(vis, label, (img_bx + size + 10, img_by - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, pt_color, 3)

                # 距離連線
                if nearest_ad is not None:
                    line_c = MATCH_COLOR if is_matched else (128, 128, 128)
                    cv2.line(vis, (img_bx, img_by),
                             (nearest_ad["peak_x"], nearest_ad["peak_y"]),
                             line_c, 2, cv2.LINE_AA)
                    mid_x = (img_bx + nearest_ad["peak_x"]) // 2
                    mid_y = (img_by + nearest_ad["peak_y"]) // 2
                    cv2.putText(vis, f"d={nearest_dist:.0f}px", (mid_x + 10, mid_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_c, 2)

        elif bomb.defect_type == "line" and len(bomb.coordinates) >= 2:
            pt1 = bomb.coordinates[0]
            pt2 = bomb.coordinates[1]
            img_x1 = int(pt1[0] * scale_x + x_start)
            img_y1 = int(pt1[1] * scale_y + y_start)
            img_x2 = int(pt2[0] * scale_x + x_start)
            img_y2 = int(pt2[1] * scale_y + y_start)
            cv2.line(vis, (img_x1, img_y1), (img_x2, img_y2), BOMB_COLOR, 5)
            mid_x = (img_x1 + img_x2) // 2
            mid_y = (img_y1 + img_y2) // 2
            cv2.putText(vis, f"BOMB LINE ({bomb.defect_code})", (mid_x + 20, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, BOMB_COLOR, 3)

    # ── 標題區 ──
    header_h = 160
    cv2.rectangle(vis, (0, 0), (vis.shape[1], header_h), (0, 0, 0), -1)
    title = f"BOMB Diagnosis: {image_path.name}"
    cv2.putText(vis, title, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)

    info1 = f"Size: {vis.shape[1]}x{vis.shape[0]} | RawBounds: {raw_bounds} | Threshold: {args.threshold} | Resolution: {PRODUCT_W}x{PRODUCT_H}"
    cv2.putText(vis, info1, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    matched = sum(1 for a in ad_tiles_info if a["is_bomb"])
    total_bombs = len(bomb_points_img)
    info2 = (f"AD Anomalies: {len(ad_tiles_info)} | "
             f"YAML Bombs: {total_bombs} | "
             f"Matched: {matched}/{total_bombs} | "
             f"Tolerance: {tolerance}px (product) -> {img_tol_x}x{img_tol_y}px (image)")
    cv2.putText(vis, info2, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    # ── 圖例 ──
    legend_h = 60
    legend_y = vis.shape[0] - legend_h
    cv2.rectangle(vis, (0, legend_y), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
    ly = legend_y + 35
    # 炸彈座標 (洋紅)
    cv2.circle(vis, (30, ly), 12, BOMB_COLOR, -1)
    cv2.putText(vis, "YAML Bomb (not matched)", (55, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BOMB_COLOR, 2)
    # 匹配 (綠色)
    cv2.circle(vis, (500, ly), 12, MATCH_COLOR, -1)
    cv2.putText(vis, "Matched", (525, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, MATCH_COLOR, 2)
    # AD NG (紅色)
    cv2.rectangle(vis, (730, ly - 12), (754, ly + 12), AD_NG_COLOR, 3)
    cv2.putText(vis, "AD Detected (NG)", (765, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, AD_NG_COLOR, 2)
    # Otsu (藍色)
    cv2.rectangle(vis, (1100, ly - 12), (1124, ly + 12), OTSU_COLOR, 3)
    cv2.putText(vis, "Otsu Bounds", (1135, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, OTSU_COLOR, 2)

    # ── 4. 儲存結果 ──
    output_path = image_path.parent / f"bomb_diagnosis_{image_path.stem}.png"
    # 縮小存檔 (原圖太大)
    max_dim = 3000
    h, w = vis.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))
    cv2.imwrite(str(output_path), vis)
    print(f"  ✅ 診斷圖已儲存: {output_path}")

    # ── 5. Console 報告 ──
    print("\n[4/5] 分析報告")
    print("-" * 70)
    print(f"{'#':>3} {'產品座標':>14} {'圖片座標':>16} {'AD分數':>8} {'匹配':>6} {'狀態'}")
    print("-" * 70)

    # 先列出 YAML 設定的炸彈座標
    for i, (bx, by, px, py, code) in enumerate(bomb_points_img):
        # 找最近的 AD 偵測
        best_ad = None
        best_dist = float('inf')
        for ad in ad_tiles_info:
            dx = abs(bx - ad["peak_x"])
            dy = abs(by - ad["peak_y"])
            dist = (dx**2 + dy**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_ad = ad

        if best_ad and best_dist < 500:  # 500px 以內才列出
            match_flag = "✅" if best_ad["is_bomb"] else "❌"
            status = "MATCHED" if best_ad["is_bomb"] else f"MISS (d={best_dist:.0f}px)"
            print(f"{i+1:>3} ({px:>4},{py:>4})  ({bx:>5},{by:>5})  "
                  f"{best_ad['score']:>7.3f}  {match_flag:>4}  {status}")
        else:
            print(f"{i+1:>3} ({px:>4},{py:>4})  ({bx:>5},{by:>5})  "
                  f"{'N/A':>7}  {'❌':>4}  NO AD DETECTION")

    # 列出未匹配到炸彈的 AD 異常
    unmatched = [a for a in ad_tiles_info if not a["is_bomb"]]
    if unmatched:
        print(f"\n⚠️  未匹配到炸彈的 AD 異常 ({len(unmatched)} 個):")
        for a in unmatched:
            print(f"   Tile@({a['tile'].x},{a['tile'].y}) "
                  f"Peak@({a['peak_x']},{a['peak_y']}) "
                  f"Prod@({a['prod_x']},{a['prod_y']}) "
                  f"Score={a['score']:.3f}")

    print("-" * 70)
    print(f"\n[5/5] 完成！")
    print(f"  📊 總異常 Tile: {len(ad_tiles_info)}")
    print(f"  💣 YAML 炸彈座標: {total_bombs}")
    print(f"  ✅ 匹配成功: {matched}")
    print(f"  ❌ 未匹配: {total_bombs - matched}")
    print(f"  📁 診斷圖: {output_path}")

    # 建議
    if matched < total_bombs:
        print(f"\n💡 建議:")
        print(f"   部分炸彈未被匹配，可能原因:")
        print(f"   1. YAML 座標與實際炸彈位置不符 → 更新 capi_3f.yaml 中的 coordinates")
        print(f"   2. bomb_match_tolerance ({tolerance}) 太小 → 可嘗試增加到 150~200")
        print(f"   3. 炸彈未被 AD 模型偵測到 (Score < {args.threshold})")

    # ── 5.5 儲存每個異常 tile 的熱力圖 ──
    print(f"\n[5.5] 儲存異常 tile 熱力圖...")
    heatmap_dir = image_path.parent / f"heatmap_tiles_{image_path.stem}"
    heatmap_dir.mkdir(exist_ok=True)

    for idx, ad in enumerate(ad_tiles_info):
        tile = ad["tile"]
        amap = ad["anomaly_map"]
        tid = tile.tile_id

        # 裁切原圖對應區域
        crop = image[tile.y:tile.y + tile.height, tile.x:tile.x + tile.width].copy()
        if len(crop.shape) == 2:
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        elif crop.shape[2] == 1:
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        else:
            crop_bgr = crop.copy()

        if amap is not None and amap.size > 0:
            # 正規化熱力圖到 0-255
            amap_vis = amap.copy()
            if amap_vis.max() > amap_vis.min():
                amap_vis = (amap_vis - amap_vis.min()) / (amap_vis.max() - amap_vis.min())
            else:
                amap_vis = np.zeros_like(amap_vis)
            amap_uint8 = (amap_vis * 255).astype(np.uint8)

            # resize 到 tile 大小
            amap_resized = cv2.resize(amap_uint8, (tile.width, tile.height))
            heatmap_color = cv2.applyColorMap(amap_resized, cv2.COLORMAP_JET)

            # 疊加
            blended = cv2.addWeighted(crop_bgr, 0.5, heatmap_color, 0.5, 0)

            # 標記峰值
            peak_lx = ad["peak_x"] - tile.x
            peak_ly = ad["peak_y"] - tile.y
            cv2.circle(blended, (peak_lx, peak_ly), 15, (0, 255, 0), 3)
            cv2.line(blended, (peak_lx - 20, peak_ly), (peak_lx + 20, peak_ly), (0, 255, 0), 2)
            cv2.line(blended, (peak_lx, peak_ly - 20), (peak_lx, peak_ly + 20), (0, 255, 0), 2)

            # 拼接: 左=原圖, 右=熱力圖疊加
            combined = np.hstack([crop_bgr, blended])
        else:
            combined = crop_bgr

        # 加標題列
        header = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)
        info = f"Tile#{tid} S:{ad['score']:.3f} Peak:({ad['peak_x']},{ad['peak_y']}) Prod:({ad['prod_x']},{ad['prod_y']})"
        cv2.putText(header, info, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        combined = np.vstack([header, combined])

        fname = heatmap_dir / f"tile_{tid:03d}_s{ad['score']:.2f}.png"
        cv2.imwrite(str(fname), combined)

    print(f"  ✅ 已儲存 {len(ad_tiles_info)} 張熱力圖到: {heatmap_dir}")

    # ── 6. 單獨畫炸彈座標圖 (格子線 + tile 編號 + YAML/AD 對照) ──
    print(f"\n[6/6] 繪製炸彈座標獨立圖...")

    # 收集當前圖片的所有 bomb 座標 (產品座標系)
    bomb_coords_product = []
    img_prefix = image_path.stem
    for bomb in config.bomb_defects:
        if not (img_prefix == bomb.image_prefix or
                img_prefix.startswith(bomb.image_prefix + "_")):
            continue
        if bomb.defect_type == "point":
            for coord in bomb.coordinates:
                bomb_coords_product.append((coord[0], coord[1], bomb.defect_code))

    if bomb_coords_product:
        vis2 = image.copy()
        if len(vis2.shape) == 2:
            vis2 = cv2.cvtColor(vis2, cv2.COLOR_GRAY2BGR)
        elif len(vis2.shape) == 3 and vis2.shape[2] == 1:
            vis2 = cv2.cvtColor(vis2, cv2.COLOR_GRAY2BGR)

        # 用戶確認的炸彈 tile 編號
        bomb_tile_ids = {38, 42, 23, 43, 18, 12}

        YAML_COLOR = (255, 0, 255)      # 洋紅 = YAML 設定座標
        AD_COLOR = (0, 255, 255)        # 黃色 = AD 實際偵測位置
        LINE_COLOR = (128, 128, 128)    # 灰色 = 偏差連線
        GRID_COLOR = (80, 80, 80)       # 暗灰 = 一般格子線
        BOMB_TILE_COLOR = (0, 200, 255) # 橘黃 = 炸彈 tile 高亮
        TILE_NUM_COLOR = (120, 120, 120)  # 灰 = 一般 tile 編號

        # ── 畫 tile 格子線 + 編號 ──
        for tile in result.tiles:
            tx, ty = tile.x, tile.y
            tw, th = tile.width, tile.height
            tid = tile.tile_id

            if tid in bomb_tile_ids:
                # 炸彈 tile: 橘黃高亮框 + 半透明背景
                overlay = vis2.copy()
                cv2.rectangle(overlay, (tx, ty), (tx + tw, ty + th), BOMB_TILE_COLOR, -1)
                cv2.addWeighted(overlay, 0.15, vis2, 0.85, 0, vis2)
                cv2.rectangle(vis2, (tx, ty), (tx + tw, ty + th), BOMB_TILE_COLOR, 4)
                # 大編號
                cv2.putText(vis2, f"#{tid}", (tx + 10, ty + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, BOMB_TILE_COLOR, 4)
            else:
                # 一般 tile: 暗灰格子線 + 小編號
                cv2.rectangle(vis2, (tx, ty), (tx + tw, ty + th), GRID_COLOR, 1)
                cv2.putText(vis2, f"#{tid}", (tx + 5, ty + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TILE_NUM_COLOR, 1)

        # ── 畫 YAML 炸彈座標 + AD 偵測對照 ──
        max_match_dist = 200
        for i, (px, py, code) in enumerate(bomb_coords_product):
            img_bx = int(px * scale_x + x_start)
            img_by = int(py * scale_y + y_start)

            # YAML 座標 (洋紅十字圓)
            size = 50
            cv2.circle(vis2, (img_bx, img_by), size, YAML_COLOR, 4)
            cv2.line(vis2, (img_bx - size, img_by), (img_bx + size, img_by), YAML_COLOR, 3)
            cv2.line(vis2, (img_bx, img_by - size), (img_bx, img_by + size), YAML_COLOR, 3)
            label_yaml = f"YAML#{i+1} ({px},{py})"
            cv2.putText(vis2, label_yaml, (img_bx + size + 10, img_by - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, YAML_COLOR, 3)

            # 找最近的 AD 偵測點
            best_ad = None
            best_prod_dist = float('inf')
            for ad in ad_tiles_info:
                dpx = abs(px - ad["prod_x"])
                dpy = abs(py - ad["prod_y"])
                prod_dist = (dpx**2 + dpy**2) ** 0.5
                if prod_dist < best_prod_dist:
                    best_prod_dist = prod_dist
                    best_ad = ad

            if best_ad and best_prod_dist < max_match_dist:
                ax, ay = best_ad["peak_x"], best_ad["peak_y"]
                apx, apy = best_ad["prod_x"], best_ad["prod_y"]

                # AD 偵測位置 (黃色菱形)
                diamond_size = 40
                pts = np.array([
                    [ax, ay - diamond_size],
                    [ax + diamond_size, ay],
                    [ax, ay + diamond_size],
                    [ax - diamond_size, ay],
                ], np.int32)
                cv2.polylines(vis2, [pts], True, AD_COLOR, 4)
                label_ad = f"AD ({apx},{apy}) S:{best_ad['score']:.2f}"
                cv2.putText(vis2, label_ad, (ax + diamond_size + 10, ay + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, AD_COLOR, 3)

                # 偏差連線
                cv2.line(vis2, (img_bx, img_by), (ax, ay), LINE_COLOR, 2, cv2.LINE_AA)
                mid_x = (img_bx + ax) // 2
                mid_y = (img_by + ay) // 2
                cv2.putText(vis2, f"d={best_prod_dist:.0f}", (mid_x + 10, mid_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, LINE_COLOR, 2)

        # 標題區
        header_h = 120
        cv2.rectangle(vis2, (0, 0), (vis2.shape[1], header_h), (0, 0, 0), -1)
        title = f"Bomb Tiles: {image_path.name}"
        cv2.putText(vis2, title, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
        # 圖例
        ly = 90
        cv2.rectangle(vis2, (20, ly - 12), (44, ly + 12), BOMB_TILE_COLOR, 3)
        cv2.putText(vis2, "Bomb Tile", (55, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BOMB_TILE_COLOR, 2)
        cv2.circle(vis2, (300, ly), 12, YAML_COLOR, -1)
        cv2.putText(vis2, "YAML", (320, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, YAML_COLOR, 2)
        diamond_pts = np.array([[510, ly-12], [522, ly], [510, ly+12], [498, ly]], np.int32)
        cv2.polylines(vis2, [diamond_pts], True, AD_COLOR, 3)
        cv2.putText(vis2, "AD Peak", (535, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, AD_COLOR, 2)

        # 縮小存檔
        max_dim = 3000
        h2, w2 = vis2.shape[:2]
        if max(h2, w2) > max_dim:
            s = max_dim / max(h2, w2)
            vis2 = cv2.resize(vis2, (int(w2 * s), int(h2 * s)))

        coord_output = image_path.parent / f"bomb_coordinates_only_{image_path.stem}.png"
        cv2.imwrite(str(coord_output), vis2)
        print(f"  ✅ 炸彈座標圖已儲存: {coord_output}")
    else:
        print(f"  ⚠️  找不到匹配的炸彈座標設定")


if __name__ == "__main__":
    main()
