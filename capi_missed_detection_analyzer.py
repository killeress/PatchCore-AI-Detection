"""
漏檢分析工具 (Missed Detection Analyzer)

針對 GT=NG 但 AD=OK 的 Panel，重新推論並嘗試多種閾值，
讀取人工標註的 Defect 座標進行可視化比對，生成 HTML 報告。

使用方式:
    python capi_missed_detection_analyzer.py --input_dir ./CAPI_20260212
"""

import argparse
import cv2
import numpy as np
import time
import json
import os
import pathlib
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

# 強制在 Linux 下將 WindowsPath 視為 PosixPath
if os.name == 'posix':
    pathlib.WindowsPath = pathlib.PosixPath

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer, ImageResult

def load_original_scores_from_html(html_path: Optional[Path]) -> Tuple[Dict[str, float], List[str]]:
    """
    從原始 HTML 報告中讀取 rawData 並提取各 Panel 的原始分數。
    同時找出「漏檢」的 Panel (ground_truth=NG 但 ad_result=Pass)。
    
    Returns:
        (scores_dict, missed_panel_ids)
    """
    scores = {}
    missed_ids = []
    if not html_path or not html_path.exists():
        print(f"  ⚠️ 找不到或未指定 HTML 報告 ({html_path})，將使用預設原始分數 0.0")
        return scores, missed_ids
    
    try:
        import re
        import json
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 尋找 rawData 變數
        match = re.search(r'const\s+rawData\s*=\s*(\[.*?\]);', content, re.DOTALL)
        if match:
            json_str = match.group(1)
            data = json.loads(json_str)
            for item in data:
                panel_id = item.get('panel_id')
                score = item.get('score', 0.0)
                if panel_id:
                    scores[panel_id] = score
                    # 找出漏檢: GT=NG 但 AD 判定為 Pass (OK)
                    gt = item.get('ground_truth', '')
                    ad = item.get('ad_result', '')
                    if gt == 'NG' and ad == 'Pass':
                        missed_ids.append(panel_id)
            print(f"  📋 從 HTML 報告 ({html_path.name}) 載入 {len(scores)} 筆原始分數")
            print(f"  🎯 偵測到 {len(missed_ids)} 個漏檢 Panel (GT=NG, AD=OK)")
        else:
            print(f"  ⚠️ 無法在 HTML 報告中找到 rawData 變數，請確認報告格式")
    except Exception as e:
        print(f"  ⚠️ 解析 HTML 報告失敗: {e}")
        
    return scores, missed_ids

THRESHOLDS = [0.50, 0.60]
DUST_THRESHOLDS = [0.0, 0.02, 0.05, 0.10]  # 模擬灰塵 IOU 閾值


@dataclass
class PanelAnalysis:
    """單一 Panel 的分析結果"""
    panel_id: str
    found: bool = False
    original_score: float = 0.0
    # 重新推論的結果
    max_score: float = 0.0
    all_tile_scores: list = field(default_factory=list)
    threshold_detection: dict = field(default_factory=dict)  # {threshold: detected}
    # GT Defect 資訊
    gt_defects: list = field(default_factory=list)  # [{defect_code, x, y, img_x, img_y, crop_path}]
    gt_in_tile: bool = False  # GT 缺陷是否在任何 tile 範圍內
    gt_near_anomaly: bool = False  # GT 缺陷位置附近是否有異常熱區
    # 推論結果
    image_results: list = field(default_factory=list)
    # 漏檢原因分類
    miss_reason: str = ""
    # 可視化圖片路徑
    vis_paths: list = field(default_factory=list)
    # GT 缺陷裁切圖路徑
    gt_crop_paths: list = field(default_factory=list)
    # 被後處理過濾
    filtered_by_dust: bool = False
    filtered_by_bomb: bool = False
    filter_detail: str = ""
    # 灰塵過濾分析
    max_dust_iou: float = 0.0  # 該 Panel 中被過濾的灰塵最大 IOU
    dust_threshold_recovery: dict = field(default_factory=dict)  # {iou_thr: recovered}


def parse_gt_defect_txt(defect_file: Path) -> Dict[str, List[Dict]]:
    # ... (unchanged) ...
    # (keeps existing code)

# ... (inside analyze_panel) ...

    # --- 檢查是否被後處理過濾 (OMIT Dust / Bomb) ---
    # 這是漏檢的重要原因：score >= threshold 但被判為灰塵或炸彈而歸類為 OK
    panel_has_real_ng = False
    max_dust_iou = 0.0

    for result in panel_results:
        if result.anomaly_tiles:
            for tile, score, amap in result.anomaly_tiles:
                if tile.is_suspected_dust_or_scratch:
                    analysis.filtered_by_dust = True
                    max_dust_iou = max(max_dust_iou, tile.dust_heatmap_iou)
                    analysis.filter_detail += (
                        f"{result.image_path.stem} Tile@({tile.x},{tile.y}): "
                        f"Score={score:.4f}, Dust IOU={tile.dust_heatmap_iou:.3f}, "
                        f"BrightRatio={tile.dust_bright_ratio:.3f}; "
                    )
                elif tile.is_bomb:
                    analysis.filtered_by_bomb = True
                    analysis.filter_detail += (
                        f"{result.image_path.stem} Tile@({tile.x},{tile.y}): "
                        f"Score={score:.4f}, Bomb={tile.bomb_defect_code}; "
                    )
                else:
                    panel_has_real_ng = True
    
    analysis.max_dust_iou = max_dust_iou

    # 模擬灰塵閾值調整：如果只被灰塵過濾，且 max_iou 小於新閾值，則視為救回
    # 前提是 max_score 必須 >= 當前測試的最低閾值 (0.5)，否則就算不過濾也抓不到
    min_score_thr = min(THRESHOLDS)
    if analysis.filtered_by_dust and not analysis.filtered_by_bomb and not panel_has_real_ng and analysis.max_score >= min_score_thr:
        for dthr in DUST_THRESHOLDS:
            # 如果最大灰塵 IOU 小於新閾值，表示該灰塵不會被過濾 -> 變回 NG -> 檢出
            if analysis.max_dust_iou < dthr:
                analysis.dust_threshold_recovery[dthr] = True
            else:
                analysis.dust_threshold_recovery[dthr] = False
    else:
        # 非單純灰塵過濾導致的漏檢 (可能是分數太低、或是炸彈、或是已經有其他 NG)
        # 對於已經 detected 的 (panel_has_real_ng)，recovery 也是 True (原本就檢出)
        # 但這裡我們專注於 "因灰塵過濾而漏檢" 的救回分析，所以預設 False
        for dthr in DUST_THRESHOLDS:
            analysis.dust_threshold_recovery[dthr] = False

    # ... (rest of function) ...

# ... (inside generate_html_report) ...
    # 灰塵閾值模擬分析
    dust_recovery_stats = {}
    total_dust_filtered = sum(1 for a in analyses if a.filtered_by_dust and not a.filtered_by_bomb)
    
    for dthr in DUST_THRESHOLDS:
        recovered = sum(1 for a in analyses if a.dust_threshold_recovery.get(dthr, False))
        dust_recovery_stats[dthr] = recovered

    # ... (HTML generation) ...

    # --- 灰塵閾值分析 ---
    html_parts.append(f"""
<div class="card">
<h2>🧹 灰塵過濾閾值模擬 (Dust IOU Simulation)</h2>
<p style="color:#8b949e; margin-bottom: 15px;">
  目前共有 {total_dust_filtered} 個 Panel 因被判定為灰塵而漏檢。
  <br>如果調高 Dust IOU 閾值 (容許更多重疊)，能救回多少？
</p>
<div class="thr-chart">
""")
    for dthr in DUST_THRESHOLDS:
        cnt = dust_recovery_stats[dthr]
        pct = int(cnt / total_dust_filtered * 100) if total_dust_filtered > 0 else 0
        bar_h = max(5, pct * 1.5)
        html_parts.append(f"""
  <div class="thr-bar">
    <div class="bar-value">{cnt}</div>
    <div class="bar-fill" style="height: {bar_h}px; background: linear-gradient(180deg, #e3b341, #b08800);"></div>
    <div class="bar-label">{dthr}</div>
  </div>""")

    html_parts.append("""
</div>
<table>
<tr><th>Dust IOU 閾值</th><th>可救回 Panel 數</th><th>救回率 (針對灰塵漏檢)</th></tr>
""")
    for dthr in DUST_THRESHOLDS:
        cnt = dust_recovery_stats[dthr]
        pct = cnt / total_dust_filtered * 100 if total_dust_filtered > 0 else 0
        html_parts.append(f"<tr><td>{dthr}</td><td>{cnt}/{total_dust_filtered}</td><td>{pct:.1f}%</td></tr>")
    html_parts.append("</table></div>")



def parse_gt_defect_txt(defect_file: Path) -> Dict[str, List[Dict]]:
    """解析實際 NG 坐標的 Defect.txt"""
    defects_map = {}
    if not defect_file.exists():
        return defects_map
    try:
        with open(defect_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            return defects_map
        records = content.split(';')
        for record in records:
            record = record.strip()
            if not record:
                continue
            parts = record.split(',')
            if len(parts) >= 4:
                filename = parts[0].strip()
                if filename not in defects_map:
                    defects_map[filename] = []
                defects_map[filename].append({
                    'defect_code': parts[1].strip(),
                    'x': int(parts[2].strip()),
                    'y': int(parts[3].strip())
                })
    except Exception as e:
        print(f"  ⚠️ 解析 Defect.txt 失敗: {e}")
    return defects_map


def analyze_panel(
    inferencer: CAPIInferencer,
    panel_dir: Path,
    gt_defect_dir: Path,
    output_dir: Path,
    panel_id: str,
    original_scores: Dict[str, float],
) -> PanelAnalysis:
    """分析單一 Panel 的漏檢原因"""
    analysis = PanelAnalysis(
        panel_id=panel_id,
        original_score=original_scores.get(panel_id, 0.0),
    )

    if not panel_dir.exists():
        print(f"  ❌ Panel 資料夾不存在: {panel_dir}")
        analysis.miss_reason = "PANEL_NOT_FOUND"
        return analysis

    analysis.found = True
    panel_out = output_dir / panel_id
    panel_out.mkdir(parents=True, exist_ok=True)

    # --- 讀取 GT Defect 座標 ---
    gt_dir = gt_defect_dir / panel_id
    gt_defect_map = {}
    if gt_dir.exists():
        defect_file = gt_dir / "Defect.txt"
        if not defect_file.exists():
            # 嘗試不區分大小寫
            for f in gt_dir.iterdir():
                if f.name.lower() == "defect.txt":
                    defect_file = f
                    break
        gt_defect_map = parse_gt_defect_txt(defect_file)
        print(f"  📋 GT Defect 載入: {sum(len(v) for v in gt_defect_map.values())} 筆")
    else:
        print(f"  ⚠️ 無 GT Defect 資料: {gt_dir}")

    # --- 執行推論 (使用原始閾值 0.6) ---
    print(f"  🔍 正在推論...")
    try:
        panel_results, omit_vis, omit_oe, omit_oe_info, is_dup = inferencer.process_panel(panel_dir)
    except Exception as e:
        print(f"  ❌ 推論失敗: {e}")
        analysis.miss_reason = "INFERENCE_ERROR"
        return analysis

    if is_dup:
        analysis.miss_reason = "DUPLICATE_PANEL"
        return analysis

    analysis.image_results = panel_results

    # --- 收集所有 tile 的分數 (包含低於閾值的) ---
    # 需要重新推論以取得所有 tile 分數
    all_scores = []
    tile_details = []  # [(img_name, tile_id, x, y, score, anomaly_map)]

    for result in panel_results:
        for tile in result.tiles:
            score, anomaly_map = inferencer.predict_tile(tile)
            all_scores.append(score)
            tile_details.append({
                'img_name': result.image_path.name,
                'tile_id': tile.tile_id,
                'x': tile.x, 'y': tile.y,
                'w': tile.width, 'h': tile.height,
                'score': score,
                'anomaly_map': anomaly_map,
                'tile': tile,
                'raw_bounds': result.raw_bounds,
            })

    analysis.all_tile_scores = sorted(all_scores, reverse=True)
    analysis.max_score = max(all_scores) if all_scores else 0.0

    # --- 多閾值檢出分析 ---
    for thr in THRESHOLDS:
        detected = any(s >= thr for s in all_scores)
        analysis.threshold_detection[thr] = detected

    # --- 檢查是否被後處理過濾 (OMIT Dust / Bomb) ---
    # 這是漏檢的重要原因：score >= threshold 但被判為灰塵或炸彈而歸類為 OK
    panel_has_real_ng = False
    for result in panel_results:
        if result.anomaly_tiles:
            for tile, score, amap in result.anomaly_tiles:
                if tile.is_suspected_dust_or_scratch:
                    analysis.filtered_by_dust = True
                    analysis.filter_detail += (
                        f"{result.image_path.stem} Tile@({tile.x},{tile.y}): "
                        f"Score={score:.4f}, Dust IOU={tile.dust_heatmap_iou:.3f}, "
                        f"BrightRatio={tile.dust_bright_ratio:.3f}; "
                    )
                elif tile.is_bomb:
                    analysis.filtered_by_bomb = True
                    analysis.filter_detail += (
                        f"{result.image_path.stem} Tile@({tile.x},{tile.y}): "
                        f"Score={score:.4f}, Bomb={tile.bomb_defect_code}; "
                    )
                else:
                    panel_has_real_ng = True

    # 檢查 OMIT 過曝狀態 (過曝時灰塵檢測被停用，不會被 dust 過濾)
    omit_was_overexposed = omit_oe
    if omit_was_overexposed:
        analysis.filter_detail += f"OMIT_OVEREXPOSED ({omit_oe_info}); "

    # --- GT Defect 映射與比對 ---
    # Defect.txt 的 filename 可能是 "G0F00000" (不含時間戳)
    # 而實際圖片檔名是 "G0F00000_085634.tif" (含時間戳)
    # 需要用前綴比對而非精確比對
    for result in panel_results:
        if result.raw_bounds is None:
            continue
        img_stem = result.image_path.stem  # e.g. "G0F00000_085634"
        img_name_full = result.image_path.name  # e.g. "G0F00000_085634.tif"
        
        # 嘗試匹配 GT 中的每個 filename
        for gt_fname, gt_list in gt_defect_map.items():
            # 移除副檔名 (如果有)
            gt_base = gt_fname.replace('.tif', '').replace('.png', '').replace('.jpg', '')
            
            # 匹配方式: 精確匹配 或 前綴匹配 (GT basename 是圖片 stem 的前綴)
            matched = (img_stem == gt_base or 
                       img_stem.startswith(gt_base + "_") or
                       img_name_full == gt_fname)
            
            if not matched:
                continue
                
            for d in gt_list:
                img_x, img_y = inferencer._map_aoi_coords(d['x'], d['y'], result.raw_bounds)
                gt_info = {
                    'defect_code': d['defect_code'],
                    'product_x': d['x'], 'product_y': d['y'],
                    'image_x': img_x, 'image_y': img_y,
                    'img_name': img_name_full,
                }
                # 檢查 GT 是否在某個 tile 範圍內
                found_in_tile = False
                for td in tile_details:
                    if td['img_name'] == img_name_full:
                        if (td['x'] <= img_x <= td['x'] + td['w'] and
                                td['y'] <= img_y <= td['y'] + td['h']):
                            gt_info['in_tile'] = True
                            gt_info['tile_score'] = td['score']
                            analysis.gt_in_tile = True
                            found_in_tile = True
                            break
                if not found_in_tile:
                    gt_info['in_tile'] = False
                    gt_info['tile_score'] = 0.0
                analysis.gt_defects.append(gt_info)
    
    if analysis.gt_defects:
        print(f"  🎯 GT 座標成功映射: {len(analysis.gt_defects)} 筆 (分布在 {len(set(g['img_name'] for g in analysis.gt_defects))} 張圖)")

    # --- 漏檢原因分類 ---
    if analysis.filtered_by_dust and analysis.filtered_by_bomb:
        analysis.miss_reason = "FILTERED_BY_DUST_AND_BOMB"
    elif analysis.filtered_by_dust:
        analysis.miss_reason = "FILTERED_BY_DUST"
    elif analysis.filtered_by_bomb:
        analysis.miss_reason = "FILTERED_BY_BOMB"
    elif analysis.max_score < 0.30:
        analysis.miss_reason = "MODEL_UNDETECTABLE"
    elif analysis.max_score < 0.60:
        analysis.miss_reason = "THRESHOLD_TOO_HIGH"
    elif analysis.gt_defects and not analysis.gt_in_tile:
        analysis.miss_reason = "GT_OUTSIDE_TILES"
    else:
        analysis.miss_reason = "THRESHOLD_TOO_HIGH"

    # --- 生成可視化圖片 ---
    for result in panel_results:
        img_stem = result.image_path.stem
        if img_stem.startswith("PINIGBI") or "OMIT0000" in img_stem:
            continue

        # 判斷此圖片是否有值得可視化的內容
        # 1. 有 GT defect
        # 2. 有高分 tile (>= 0.3)
        # 3. 有被過濾的 tile (dust/bomb)
        has_gt = any(gt['img_name'] == result.image_path.name for gt in analysis.gt_defects)
        has_anomaly_tile = any(
            td['img_name'] == result.image_path.name and 
            (td['score'] >= 0.3 or td['tile'].is_suspected_dust_or_scratch or td['tile'].is_bomb)
            for td in tile_details
        )
        
        if not has_gt and not has_anomaly_tile:
            continue

        try:
            image = cv2.imread(str(result.image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            vis = image.copy()
            if len(vis.shape) == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            elif len(vis.shape) == 3 and vis.shape[2] == 1:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

            # 1. 畫 Otsu 邊界 (藍色)
            x1, y1, x2, y2 = result.otsu_bounds
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 4)

            # 2. 畫高分 tile 框 + 分數 (只標示 >= 0.5)
            for td in tile_details:
                if td['img_name'] != result.image_path.name:
                    continue
                s = td['score']
                if s < 0.5:
                    continue
                if s >= 0.6:
                    color = (0, 0, 255)  # 紅 (>=0.6)
                else:
                    color = (0, 165, 255)  # 橘 (0.5~0.6)
                cv2.rectangle(vis, (td['x'], td['y']),
                              (td['x'] + td['w'], td['y'] + td['h']),
                              color, 4)
                cv2.putText(vis, f"{s:.3f}", (td['x'] + 5, td['y'] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # 3. 畫 GT Defect 位置 (大紅色星號) + 512x512 裁切 (僅當有 GT 資料時)
            if has_gt:
                for gt_idx, gt in enumerate(analysis.gt_defects):
                    if gt['img_name'] != result.image_path.name:
                        continue
                    gx, gy = gt['image_x'], gt['image_y']
                    # 大紅色十字 + 圓圈
                    cv2.circle(vis, (gx, gy), 100, (0, 0, 255), 8)
                    cv2.line(vis, (gx - 120, gy), (gx + 120, gy), (0, 0, 255), 6)
                    cv2.line(vis, (gx, gy - 120), (gx, gy + 120), (0, 0, 255), 6)
                    # 標籤
                    label = f"GT: ({gt['product_x']},{gt['product_y']}) {gt['defect_code']}"
                    cv2.putText(vis, label, (gx + 130, gy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    in_label = f"In Tile: {'YES' if gt['in_tile'] else 'NO'} | Score: {gt['tile_score']:.3f}"
                    cv2.putText(vis, in_label, (gx + 130, gy + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                    # --- 512x512 裁切 (以 GT Defect 位置為中心) ---
                    crop_size = 512
                    img_h, img_w = image.shape[:2]
                    cx1 = max(0, gx - crop_size // 2)
                    cy1 = max(0, gy - crop_size // 2)
                    cx2 = min(img_w, cx1 + crop_size)
                    cy2 = min(img_h, cy1 + crop_size)
                    if cx2 - cx1 < crop_size:
                        cx1 = max(0, cx2 - crop_size)
                    if cy2 - cy1 < crop_size:
                        cy1 = max(0, cy2 - crop_size)

                    crop_img = image[cy1:cy2, cx1:cx2].copy()
                    if len(crop_img.shape) == 2:
                        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)
                    elif len(crop_img.shape) == 3 and crop_img.shape[2] == 1:
                        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)

                    # 在裁切圖上標示 GT 位置 (用角落方框，不遮擋缺陷中心)
                    rel_x = gx - cx1
                    rel_y = gy - cy1
                    r = 35
                    corner_len = 12
                    marker_color = (0, 255, 0)
                    cv2.line(crop_img, (rel_x - r, rel_y - r), (rel_x - r + corner_len, rel_y - r), marker_color, 2)
                    cv2.line(crop_img, (rel_x - r, rel_y - r), (rel_x - r, rel_y - r + corner_len), marker_color, 2)
                    cv2.line(crop_img, (rel_x + r, rel_y - r), (rel_x + r - corner_len, rel_y - r), marker_color, 2)
                    cv2.line(crop_img, (rel_x + r, rel_y - r), (rel_x + r, rel_y - r + corner_len), marker_color, 2)
                    cv2.line(crop_img, (rel_x - r, rel_y + r), (rel_x - r + corner_len, rel_y + r), marker_color, 2)
                    cv2.line(crop_img, (rel_x - r, rel_y + r), (rel_x - r, rel_y + r - corner_len), marker_color, 2)
                    cv2.line(crop_img, (rel_x + r, rel_y + r), (rel_x + r - corner_len, rel_y + r), marker_color, 2)
                    cv2.line(crop_img, (rel_x + r, rel_y + r), (rel_x + r, rel_y + r - corner_len), marker_color, 2)

                    # 標題 (頂部黑底)
                    cv2.rectangle(crop_img, (0, 0), (crop_size, 50), (0, 0, 0), -1)
                    crop_label = f"GT#{gt_idx+1} {gt['defect_code']} ({gt['product_x']},{gt['product_y']}) Score:{gt.get('tile_score',0):.3f}"
                    cv2.putText(crop_img, crop_label, (5, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                    crop_path = panel_out / f"{img_stem}_GT_crop{gt_idx+1}_{gt['defect_code']}.jpg"
                    cv2.imwrite(str(crop_path), crop_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    gt['crop_path'] = str(crop_path.relative_to(output_dir))
                    analysis.gt_crop_paths.append(gt['crop_path'])

            # 4. 標題列
            header_h = 120
            cv2.rectangle(vis, (0, 0), (vis.shape[1], header_h), (0, 0, 0), -1)
            title = f"Panel: {panel_id} | Image: {img_stem} | Max Score: {analysis.max_score:.4f} | THR=0.6: {'MISS' if analysis.max_score < 0.6 else 'DETECT'}"
            cv2.putText(vis, title, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            reason_text = f"Miss Reason: {analysis.miss_reason} | Top3 Scores: {analysis.all_tile_scores[:3]}"
            cv2.putText(vis, reason_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)

            # 儲存
            scale = 0.5
            vis_small = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))
            save_path = panel_out / f"{img_stem}_analysis.jpg"
            cv2.imwrite(str(save_path), vis_small, [cv2.IMWRITE_JPEG_QUALITY, 85])
            analysis.vis_paths.append(str(save_path.relative_to(output_dir)))

            # --- 組合圖：Original | Heatmap | OMIT | Dust Mask ---
            # 對每個有異常/被過濾/含GT的 tile 產出四格組合圖
            for td in tile_details:
                if td['img_name'] != result.image_path.name:
                    continue
                min_thr = min(THRESHOLDS)
                is_gt_tile = False
                for gt in analysis.gt_defects:
                    if gt['img_name'] == result.image_path.name:
                        gx, gy = gt['image_x'], gt['image_y']
                        if td['x'] <= gx <= td['x'] + td['w'] and td['y'] <= gy <= td['y'] + td['h']:
                            is_gt_tile = True
                            break
                
                # 只產出包含 GT Defect 的 Tile 組合圖
                if not is_gt_tile:
                    continue
                tile_size = 512

                # Panel 1: Original Tile
                orig = tile.image.copy()
                if len(orig.shape) == 2:
                    orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
                elif len(orig.shape) == 3 and orig.shape[2] == 1:
                    orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
                orig = cv2.resize(orig, (tile_size, tile_size))

                # 如果此 tile 包含 GT defect，在 Original 和 Heatmap 上標示位置
                gt_in_this_tile = []
                for gt in analysis.gt_defects:
                    if gt['img_name'] == result.image_path.name:
                        gx, gy = gt['image_x'], gt['image_y']
                        if td['x'] <= gx <= td['x'] + td['w'] and td['y'] <= gy <= td['y'] + td['h']:
                            # 計算 GT 在 512x512 tile 中的相對位置
                            rel_x = int((gx - td['x']) / td['w'] * tile_size)
                            rel_y = int((gy - td['y']) / td['h'] * tile_size)
                            gt_in_this_tile.append((rel_x, rel_y, gt))

                # 在 Original 上畫 GT 綠色十字準心
                for rel_x, rel_y, gt in gt_in_this_tile:
                    cv2.drawMarker(orig, (rel_x, rel_y), (0, 255, 0), 
                                   cv2.MARKER_CROSS, 60, 2)
                    cv2.putText(orig, f"GT:{gt['defect_code']}", (rel_x + 15, rel_y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Panel 2: Heatmap Overlay
                amap = td['anomaly_map']
                if amap is not None:
                    norm_map = cv2.normalize(amap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
                    heatmap_color = cv2.resize(heatmap_color, (tile_size, tile_size))
                    heatmap_panel = cv2.addWeighted(orig, 0.5, heatmap_color, 0.5, 0)
                else:
                    heatmap_panel = orig.copy()
                    cv2.putText(heatmap_panel, "No Heatmap", (150, 260),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)

                # 在 Heatmap 上也畫 GT 標記 (白色，以便在熱圖上可見)
                for rel_x, rel_y, gt in gt_in_this_tile:
                    cv2.drawMarker(heatmap_panel, (rel_x, rel_y), (255, 255, 255),
                                   cv2.MARKER_CROSS, 60, 2)

                # Panel 3: OMIT Crop
                if tile.omit_crop_image is not None:
                    omit_panel = tile.omit_crop_image.copy()
                    if len(omit_panel.shape) == 2:
                        omit_panel = cv2.cvtColor(omit_panel, cv2.COLOR_GRAY2BGR)
                    omit_panel = cv2.resize(omit_panel, (tile_size, tile_size))
                else:
                    omit_panel = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    cv2.putText(omit_panel, "No OMIT", (170, 260),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)

                # Panel 4: Dust Mask Overlay
                if tile.dust_mask is not None and tile.omit_crop_image is not None:
                    dust_panel = omit_panel.copy()
                    dust_colored = np.zeros_like(dust_panel)
                    dust_resized = cv2.resize(tile.dust_mask, (tile_size, tile_size))
                    dust_colored[dust_resized > 0] = (0, 255, 255)
                    dust_panel = cv2.addWeighted(dust_panel, 0.6, dust_colored, 0.4, 0)
                    cv2.putText(dust_panel, f"IOU: {tile.dust_heatmap_iou:.3f}", (10, 490),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    dust_panel = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    cv2.putText(dust_panel, "No Dust Data", (140, 260),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)

                # 各 Panel 加標題 (底部)
                labels = ["Original", "Heatmap", "OMIT Crop", "Dust Mask"]
                panels = [orig, heatmap_panel, omit_panel, dust_panel]
                for lbl, p in zip(labels, panels):
                    cv2.rectangle(p, (0, tile_size - 35), (tile_size, tile_size), (0, 0, 0), -1)
                    cv2.putText(p, lbl, (10, tile_size - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                # 橫向拼接
                composite = np.hstack(panels)
                comp_h, comp_w = composite.shape[:2]

                # 頂部 Header
                header_h = 60
                header = np.zeros((header_h, comp_w, 3), dtype=np.uint8)

                if tile.is_suspected_dust_or_scratch:
                    verdict = "DUST (Filtered as OK)"
                    verdict_color = (0, 200, 255)
                elif tile.is_bomb:
                    verdict = f"BOMB: {tile.bomb_defect_code} (Filtered as OK)"
                    verdict_color = (0, 100, 255)
                elif td['score'] >= 0.6:
                    verdict = "REAL NG (Detected)"
                    verdict_color = (0, 0, 255)
                else:
                    verdict = "BELOW THRESHOLD (Missed)"
                    verdict_color = (0, 255, 255)

                # 如果含 GT，在 header 加上 GT 標記
                gt_tag = ""
                if gt_in_this_tile:
                    gt_codes = ",".join(g['defect_code'] for _, _, g in gt_in_this_tile)
                    gt_tag = f" | [GT: {gt_codes}]"

                header_text = f"Score: {td['score']:.4f} | {verdict}{gt_tag}"
                cv2.putText(header, header_text, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, verdict_color, 2)

                if tile.dust_detail_text:
                    # 替換可能殘留的 Unicode 字元
                    detail_line = tile.dust_detail_text[:120].replace('\u2192', '->').replace('\u2190', '<-')
                else:
                    detail_line = f"Tile@({tile.x},{tile.y}) BrightRatio:{tile.dust_bright_ratio:.3f} IOU:{tile.dust_heatmap_iou:.3f}"
                cv2.putText(header, detail_line, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

                final = np.vstack([header, composite])

                hm_path = panel_out / f"{img_stem}_heatmap_tile{td['tile_id']}.jpg"
                cv2.imwrite(str(hm_path), final, [cv2.IMWRITE_JPEG_QUALITY, 90])
                analysis.vis_paths.append(str(hm_path.relative_to(output_dir)))

            # --- GT 不在任何 Tile 中的情況：以 GT 為中心做 512x512 裁切 + 推論 ---
            for gt_idx, gt in enumerate(analysis.gt_defects):
                if gt['img_name'] != result.image_path.name:
                    continue
                if gt.get('in_tile'):
                    continue  # 已經由上面的 tile 迴圈處理過了

                gx, gy = gt['image_x'], gt['image_y']
                crop_size = 512
                img_h, img_w = image.shape[:2]
                cx1 = max(0, gx - crop_size // 2)
                cy1 = max(0, gy - crop_size // 2)
                cx2 = min(img_w, cx1 + crop_size)
                cy2 = min(img_h, cy1 + crop_size)
                if cx2 - cx1 < crop_size:
                    cx1 = max(0, cx2 - crop_size)
                if cy2 - cy1 < crop_size:
                    cy1 = max(0, cy2 - crop_size)

                gt_crop = image[cy1:cy2, cx1:cx2].copy()
                if gt_crop.size == 0:
                    continue

                # 轉 BGR
                if len(gt_crop.shape) == 2:
                    gt_crop_bgr = cv2.cvtColor(gt_crop, cv2.COLOR_GRAY2BGR)
                elif len(gt_crop.shape) == 3 and gt_crop.shape[2] == 1:
                    gt_crop_bgr = cv2.cvtColor(gt_crop, cv2.COLOR_GRAY2BGR)
                else:
                    gt_crop_bgr = gt_crop.copy()

                gt_crop_resized = cv2.resize(gt_crop_bgr, (crop_size, crop_size))

                # 對這個 GT 裁切做推論
                from capi_inference import TileInfo
                gt_tile = TileInfo(
                    image=gt_crop,
                    x=cx1, y=cy1,
                    width=cx2 - cx1, height=cy2 - cy1,
                    tile_id=9000 + gt_idx,
                )
                gt_score, gt_amap = inferencer.predict_tile(gt_tile)

                # Panel 1: Original + GT 標記
                orig_panel = gt_crop_resized.copy()
                rel_x = gx - cx1
                rel_y = gy - cy1
                # 縮放到 512
                rel_x_s = int(rel_x / (cx2 - cx1) * crop_size)
                rel_y_s = int(rel_y / (cy2 - cy1) * crop_size)
                cv2.drawMarker(orig_panel, (rel_x_s, rel_y_s), (0, 255, 0),
                               cv2.MARKER_CROSS, 60, 2)
                cv2.putText(orig_panel, f"GT:{gt['defect_code']}", (rel_x_s + 15, rel_y_s - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Panel 2: Heatmap
                if gt_amap is not None:
                    norm_map = cv2.normalize(gt_amap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    hm_color = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
                    hm_color = cv2.resize(hm_color, (crop_size, crop_size))
                    hm_panel = cv2.addWeighted(orig_panel, 0.5, hm_color, 0.5, 0)
                    cv2.drawMarker(hm_panel, (rel_x_s, rel_y_s), (255, 255, 255),
                                   cv2.MARKER_CROSS, 60, 2)
                else:
                    hm_panel = orig_panel.copy()
                    cv2.putText(hm_panel, "No Heatmap", (150, 260),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)

                # 底部標籤
                for lbl, p in [("Original (GT center)", orig_panel), ("Heatmap", hm_panel)]:
                    cv2.rectangle(p, (0, crop_size - 35), (crop_size, crop_size), (0, 0, 0), -1)
                    cv2.putText(p, lbl, (10, crop_size - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                composite_gt = np.hstack([orig_panel, hm_panel])
                comp_h, comp_w = composite_gt.shape[:2]

                # Header
                header_h = 60
                header = np.zeros((header_h, comp_w, 3), dtype=np.uint8)
                header_text = f"GT NOT IN TILE | Score: {gt_score:.4f} | GT: {gt['defect_code']} ({gt['product_x']},{gt['product_y']})"
                cv2.putText(header, header_text, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                detail = f"Image: {img_stem} | GT@({gx},{gy}) -> Crop({cx1},{cy1})-({cx2},{cy2})"
                cv2.putText(header, detail, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

                final_gt = np.vstack([header, composite_gt])
                gt_hm_path = panel_out / f"{img_stem}_GT_heatmap{gt_idx+1}_{gt['defect_code']}.jpg"
                cv2.imwrite(str(gt_hm_path), final_gt, [cv2.IMWRITE_JPEG_QUALITY, 90])
                analysis.vis_paths.append(str(gt_hm_path.relative_to(output_dir)))

        except Exception as e:
            print(f"  ⚠️ 可視化失敗 ({img_stem}): {e}")

    return analysis


def generate_html_report(analyses: List[PanelAnalysis], output_dir: Path, inferencer: Any, ok_results: List[Dict]=None):
    """生成 HTML 分析報告"""
    # --- 統計 ---
    total = len(analyses)
    found = sum(1 for a in analyses if a.found)
    reasons = {}
    for a in analyses:
        reasons[a.miss_reason] = reasons.get(a.miss_reason, 0) + 1

    # 閾值分析
    thr_stats = {}
    for thr in THRESHOLDS:
        detected = sum(1 for a in analyses if a.found and a.threshold_detection.get(thr, False))
        thr_stats[thr] = detected

    # 分數分布
    score_ranges = {"0.0": 0, "0.0~0.3": 0, "0.3~0.4": 0, "0.4~0.5": 0, "0.5~0.6": 0, "≥0.6": 0}
    for a in analyses:
        if not a.found:
            continue
        s = a.max_score
        if s == 0:
            score_ranges["0.0"] += 1
        elif s < 0.3:
            score_ranges["0.0~0.3"] += 1
        elif s < 0.4:
            score_ranges["0.3~0.4"] += 1
        elif s < 0.5:
            score_ranges["0.4~0.5"] += 1
        elif s < 0.6:
            score_ranges["0.5~0.6"] += 1
        else:
            score_ranges["≥0.6"] += 1

    reason_labels = {
        "MODEL_UNDETECTABLE": "🔵 模型完全未偵測 (score < 0.3)",
        "THRESHOLD_TOO_HIGH": "🟡 閾值過高 (0.3 ≤ score < 0.6)",
        "FILTERED_BY_DUST": "🧹 被 OMIT 灰塵過濾 (Dust)",
        "FILTERED_BY_BOMB": "💣 被炸彈系統過濾 (Bomb)",
        "FILTERED_BY_DUST_AND_BOMB": "🟠 同時被灰塵+炸彈過濾",
        "GT_OUTSIDE_TILES": "🔴 GT 缺陷不在 Tile 範圍",
        "PANEL_NOT_FOUND": "⚪ Panel 資料夾不存在",
        "DUPLICATE_PANEL": "⚪ 重複投片",
        "INFERENCE_ERROR": "❌ 推論錯誤",
    }

    # --- HTML 生成 ---
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>漏檢分析報告 - Missed Detection Analysis</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
h1 {{ color: #58a6ff; margin-bottom: 20px; text-align: center; font-size: 28px; }}
h2 {{ color: #79c0ff; margin: 30px 0 15px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }}
h3 {{ color: #d2a8ff; margin: 15px 0 10px; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
th, td {{ padding: 8px 12px; text-align: left; border: 1px solid #30363d; }}
th {{ background: #21262d; color: #79c0ff; font-weight: 600; }}
tr:hover {{ background: #1c2128; }}
.score-high {{ color: #f85149; font-weight: bold; }}
.score-mid {{ color: #d29922; }}
.score-low {{ color: #3fb950; }}
.detected {{ color: #3fb950; font-weight: bold; }}
.missed {{ color: #f85149; }}
.reason-tag {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin: 2px; }}
.reason-MODEL_UNDETECTABLE {{ background: #1f6feb33; color: #58a6ff; }}
.reason-THRESHOLD_TOO_HIGH {{ background: #d2992233; color: #d29922; }}
.reason-FILTERED_BY_DUST {{ background: #d2992233; color: #e3b341; }}
.reason-FILTERED_BY_BOMB {{ background: #da3c1f33; color: #f0883e; }}
.reason-FILTERED_BY_DUST_AND_BOMB {{ background: #da3c1f33; color: #f0883e; }}
.reason-GT_OUTSIDE_TILES {{ background: #f8514933; color: #f85149; }}
.bar {{ display: inline-block; height: 20px; border-radius: 3px; margin-right: 5px; }}
.panel-detail {{ margin: 15px 0; padding: 15px; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; }}
.img-container {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }}
.img-container img {{ max-width: 800px; border-radius: 4px; border: 1px solid #30363d; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
.summary-item {{ background: #21262d; padding: 15px; border-radius: 8px; text-align: center; }}
.summary-item .value {{ font-size: 32px; font-weight: bold; color: #58a6ff; }}
.summary-item .label {{ color: #8b949e; font-size: 14px; margin-top: 5px; }}
.thr-chart {{ display: flex; align-items: flex-end; gap: 8px; height: 200px; padding: 10px; }}
.thr-bar {{ display: flex; flex-direction: column; align-items: center; flex: 1; }}
.thr-bar .bar-fill {{ width: 100%; background: linear-gradient(180deg, #3fb950, #238636); border-radius: 4px 4px 0 0; transition: height 0.3s; }}
.thr-bar .bar-label {{ margin-top: 5px; font-size: 12px; color: #8b949e; }}
.thr-bar .bar-value {{ font-size: 14px; font-weight: bold; color: #c9d1d9; margin-bottom: 3px; }}
.gt-info {{ background: #f8514922; border: 1px solid #f85149; padding: 10px; border-radius: 6px; margin: 5px 0; }}
.sweet-spot-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
.sweet-spot-table th, .sweet-spot-table td {{ padding: 8px; border: 1px solid #444; text-align: center; }}
.sweet-spot-table th {{ background: #21262d; }}
.best-choice {{ background: #23863644; font-weight: bold; color: #7ee787; }}
</style>
</head>
<body>
<h1>🔍 漏檢分析報告 (Missed Detection Analysis)</h1>
<p style="text-align:center; color:#8b949e; margin-bottom: 30px;">
  產生時間: {time.strftime('%Y-%m-%d %H:%M:%S')} | 分析 Panels: {total} (NG) + {len(ok_results) if ok_results else 0} (OK) | 原始閾值: 0.6 | Dust IOU: {inferencer.config.dust_heatmap_iou_threshold}
</p>
""")

    # --- 1. Sweet Spot 分析 (如果有跑 OK Panel) ---
    if ok_results:
        total_ok = len(ok_results)
        html_parts.append(f"""
<div class="card">
<h2>🎯 閾值甜蜜點分析 (Sweet Spot Analysis)</h2>
<p style="color:#8b949e; margin-bottom: 15px;">
  結合 NG 漏檢分析與 OK 假陽性測試，尋找最佳平衡點。
  <br>Recall: 在 {total} 個漏檢樣本中能抓回的比例。
  <br>FPR (False Positive Rate): 在 {total_ok} 個 OK 樣本中誤判為 NG 的比例。
</p>
<table class="sweet-spot-table">
<tr>
    <th>閾值 (Threshold)</th>
    <th>Recall (救回率)</th>
    <th>FPR (誤報率)</th>
    <th>建議</th>
</tr>
""")
        # 計算每個閾值的 Recall & FPR
        for thr in THRESHOLDS:
            # Recall: 原本漏檢的 58 個中，現在抓到了幾個
            # 注意: 這裡的 analyses 全都是原本漏檢的，所以 "detected" = 救回
            # 但要注意 filtered_by_dust 的情況，如果被 filter 了就不算 detected
            # analyze_panel 中已經算好了 threshold_detection[thr] (有考慮 postprocess filter)
            recovered_count = sum(1 for a in analyses if a.found and a.threshold_detection.get(thr, False))
            recall = recovered_count / total if total > 0 else 0
            
            # FPR: OK Panel 中被誤判的個數
            fp_count = sum(1 for r in ok_results if r['threshold_fp'].get(thr, False))
            fpr = fp_count / total_ok if total_ok > 0 else 0
            
            # 簡單建議邏輯
            suggestion = ""
            row_class = ""
            if recall > 0.95 and fpr < 0.05:
                suggestion = "🌟 推薦 (高Recall, 低FPR)"
                row_class = "best-choice"
            elif recall > 0.9 and fpr < 0.1:
                suggestion = "✅ 可用 (平衡)"
            elif fpr > 0.2:
                suggestion = "⚠️ 誤報過高"
            elif recall < 0.5:
                suggestion = "❌ 漏檢過多"
                
            html_parts.append(f"""
<tr class="{row_class}">
    <td>{thr}</td>
    <td>{recovered_count}/{total} ({recall*100:.1f}%)</td>
    <td>{fp_count}/{total_ok} ({fpr*100:.1f}%)</td>
    <td>{suggestion}</td>
</tr>
""")
        html_parts.append("</table></div>")
        
        # --- 灰塵閾值 FPR ---
        html_parts.append(f"""
<div class="card">
<h2>🧹 灰塵 IOU 甜蜜點 (Dust Sweet Spot)</h2>
<p style="color:#8b949e; margin-bottom: 15px;">
  調高 Dust IOU 閾值可以救回被誤殺的 NG，但也可能導致灰塵被誤判為 NG (FP)。
</p>
<table class="sweet-spot-table">
<tr>
    <th>Dust IOU 閾值</th>
    <th>NG 救回數 (Recall Gain)</th>
    <th>OK 誤報數 (FP Increase)</th>
</tr>
""")
        for dthr in DUST_THRESHOLDS:
            # NG 救回
            ng_recovered = sum(1 for a in analyses if a.dust_threshold_recovery.get(dthr, False))
            
            # OK 誤報增量 (原本是 OK/Filtered，現在變成 FP)
            ok_fp_increase = sum(1 for r in ok_results if r['dust_recovery_fp'].get(dthr, False))
            
            html_parts.append(f"""
<tr>
    <td>{dthr}</td>
    <td>+{ng_recovered}</td>
    <td>+{ok_fp_increase}</td>
</tr>
""")
        html_parts.append("</table></div>")

    # --- 總覽 ---
    html_parts.append(f"""
<div class="card">
<h2>📊 總覽</h2>
<div class="summary-grid">
  <div class="summary-item"><div class="value">{total}</div><div class="label">漏檢 Panels</div></div>
  <div class="summary-item"><div class="value">{found}</div><div class="label">已找到資料</div></div>
  <div class="summary-item"><div class="value">{reasons.get('MODEL_UNDETECTABLE', 0)}</div><div class="label">模型完全未偵測</div></div>
  <div class="summary-item"><div class="value">{reasons.get('THRESHOLD_TOO_HIGH', 0)}</div><div class="label">閾值過高可修復</div></div>
  <div class="summary-item"><div class="value">{reasons.get('FILTERED_BY_DUST', 0)}</div><div class="label">🧹 灰塵過濾</div></div>
  <div class="summary-item"><div class="value">{reasons.get('FILTERED_BY_BOMB', 0)}</div><div class="label">💣 炸彈過濾</div></div>
  <div class="summary-item"><div class="value">{reasons.get('PANEL_NOT_FOUND', 0)}</div><div class="label">資料缺失</div></div>
</div>
</div>
""")

    # --- 閾值分析 ---
    html_parts.append(f"""
<div class="card">
<h2>📈 閾值調整分析</h2>
<p style="color:#8b949e; margin-bottom: 15px;">如果調整 anomaly_threshold，能多檢出多少漏檢 Panel？</p>
<div class="thr-chart">
""")
    for thr in THRESHOLDS:
        cnt = thr_stats[thr]
        pct = int(cnt / found * 100) if found > 0 else 0
        bar_h = max(5, pct * 1.5)
        html_parts.append(f"""
  <div class="thr-bar">
    <div class="bar-value">{cnt}/{found}</div>
    <div class="bar-fill" style="height: {bar_h}px;"></div>
    <div class="bar-label">{thr}</div>
  </div>""")

    html_parts.append("""
</div>
<table>
<tr><th>閾值</th><th>可檢出數</th><th>檢出率</th><th>新增檢出</th></tr>
""")
    prev = 0
    for thr in THRESHOLDS:
        cnt = thr_stats[thr]
        pct = cnt / found * 100 if found > 0 else 0
        delta = cnt - prev
        prev = cnt
        html_parts.append(f"<tr><td>{thr}</td><td>{cnt}/{found}</td><td>{pct:.1f}%</td><td>+{delta}</td></tr>")
    html_parts.append("</table></div>")

    # --- 分數分布 ---
    html_parts.append("""
<div class="card">
<h2>📊 最高分數分布</h2>
<table>
<tr><th>分數範圍</th><th>Panel 數</th><th>比例</th></tr>
""")
    for rng, cnt in score_ranges.items():
        pct = cnt / found * 100 if found > 0 else 0
        html_parts.append(f"<tr><td>{rng}</td><td>{cnt}</td><td>{pct:.1f}%</td></tr>")
    html_parts.append("</table></div>")

    # --- 漏檢原因統計 ---
    html_parts.append("""
<div class="card">
<h2>🔎 漏檢原因分類</h2>
<table>
<tr><th>原因</th><th>數量</th><th>比例</th></tr>
""")
    for reason, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        label = reason_labels.get(reason, reason)
        pct = cnt / total * 100 if total > 0 else 0
        html_parts.append(f"<tr><td>{label}</td><td>{cnt}</td><td>{pct:.1f}%</td></tr>")
    html_parts.append("</table></div>")

    # --- Panel 總表 ---
    html_parts.append("""
<div class="card">
<h2>📋 Panel 詳細列表</h2>
<table>
<tr><th>#</th><th>Panel ID</th><th>原始 Score</th><th>重新推論 Max Score</th><th>漏檢原因</th>
<th>THR=0.3</th><th>THR=0.4</th><th>THR=0.5</th><th>THR=0.6</th><th>GT Defects</th></tr>
""")
    for i, a in enumerate(sorted(analyses, key=lambda x: -x.max_score)):
        score_class = "score-high" if a.max_score >= 0.6 else ("score-mid" if a.max_score >= 0.3 else "score-low")
        thr_cells = ""
        for thr in [0.3, 0.4, 0.5, 0.6]:
            det = a.threshold_detection.get(thr, False)
            cls = "detected" if det else "missed"
            txt = "✅" if det else "❌"
            thr_cells += f'<td class="{cls}">{txt}</td>'
        gt_count = len(a.gt_defects)
        reason_cls = f"reason-{a.miss_reason}"
        html_parts.append(f"""
<tr>
  <td>{i+1}</td>
  <td><a href="#{a.panel_id}" style="color:#58a6ff">{a.panel_id}</a></td>
  <td>{a.original_score:.4f}</td>
  <td class="{score_class}">{a.max_score:.4f}</td>
  <td><span class="reason-tag {reason_cls}">{a.miss_reason}</span></td>
  {thr_cells}
  <td>{gt_count}</td>
</tr>""")
    html_parts.append("</table></div>")

    # --- 每個 Panel 的詳細分析 ---
    html_parts.append('<h2>🔬 Panel 詳細分析</h2>')
    for a in sorted(analyses, key=lambda x: -x.max_score):
        if not a.found:
            continue
        html_parts.append(f"""
<div class="panel-detail" id="{a.panel_id}">
<h3>📦 {a.panel_id}</h3>
<table>
<tr><th>項目</th><th>值</th></tr>
<tr><td>原始 Score (THR=0.6)</td><td>{a.original_score:.4f}</td></tr>
<tr><td>重新推論 Max Score</td><td class="{'score-high' if a.max_score >= 0.6 else 'score-mid'}">{a.max_score:.4f}</td></tr>
<tr><td>漏檢原因</td><td><span class="reason-tag reason-{a.miss_reason}">{a.miss_reason}</span></td></tr>
<tr><td>Top 5 Tile Scores</td><td>{[f'{s:.4f}' for s in a.all_tile_scores[:5]]}</td></tr>
<tr><td>灰塵過濾</td><td>{'🧹 ' + a.filter_detail if a.filtered_by_dust else '否'}</td></tr>
<tr><td>炸彈過濾</td><td>{'💣 ' + a.filter_detail if a.filtered_by_bomb else '否'}</td></tr>
</table>
""")
        # GT Defect 資訊
        if a.gt_defects:
            html_parts.append('<div class="gt-info"><h4>🎯 GT Defect 位置 (人工標註)</h4><table>')
            html_parts.append('<tr><th>Image</th><th>Defect Code</th><th>產品座標</th><th>圖片座標</th><th>在 Tile 內?</th><th>Tile Score</th><th>512x512 裁切</th></tr>')
            for gt in a.gt_defects:
                in_tile = "✅ YES" if gt.get('in_tile') else "❌ NO"
                crop_html = ""
                if gt.get('crop_path'):
                    crop_html = f'<img src="{gt["crop_path"]}" style="max-width:256px; border-radius:4px;">'
                html_parts.append(f"""<tr>
<td>{gt['img_name']}</td><td>{gt['defect_code']}</td>
<td>({gt['product_x']}, {gt['product_y']})</td>
<td>({gt['image_x']}, {gt['image_y']})</td>
<td>{in_tile}</td><td>{gt.get('tile_score', 0):.4f}</td>
<td>{crop_html}</td></tr>""")
            html_parts.append('</table></div>')

        # 可視化圖片
        if a.vis_paths:
            html_parts.append('<div class="img-container">')
            for vp in a.vis_paths:
                html_parts.append(f'<img src="{vp}" alt="{a.panel_id}" loading="lazy">')
            html_parts.append('</div>')

        html_parts.append('</div>')

    html_parts.append('</body></html>')

    report_path = output_dir / "missed_detection_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    print(f"\n📄 HTML 報告已生成: {report_path}")
    return report_path


def analyze_ok_panel(inferencer: Any, panel_dir: Path) -> Optional[Dict]:
    """分析單一 OK Panel (用於計算 FPR)"""
    try:
        # 執行推論 (不需存圖，不需 GT)
        # 為了加速，可以設 cpu_workers=0 (主執行緒執行) 或 2
        results = inferencer.process_panel(panel_dir, cpu_workers=2)
        
        p_max_score = 0.0
        p_filtered = False
        p_dust_iou = 0.0
        
        for r in results:
            if r.anomaly_tiles:
                for t, s, _ in r.anomaly_tiles:
                    # 檢查過濾狀態
                    if t.is_suspected_dust_or_scratch:
                        p_filtered = True # 只要有一個 tile 被過濾就算有灰塵
                        p_dust_iou = max(p_dust_iou, t.dust_heatmap_iou)
                    elif t.is_bomb:
                        p_filtered = True
                    else:
                        p_max_score = max(p_max_score, s)
        
        res = {
            'id': panel_dir.name,
            'max_score': p_max_score,
            'max_dust_iou': p_dust_iou,
            'is_dust_filtered': p_filtered,
            'threshold_fp': {},     # {thr: is_fp}
            'dust_recovery_fp': {}  # {dthr: would_be_fp}
        }
        
        # 1. 閾值假陽性分析 (Anomaly Threshold FPR)
        for thr in THRESHOLDS:
            # FP 定義：Score >= Thr 且 "最後判定為 NG" (即未被過濾)
            if p_max_score >= thr:
                res['threshold_fp'][thr] = True
            else:
                res['threshold_fp'][thr] = False
                
        # 2. 灰塵閾值假陽性分析 (Dust IOU FPR)
        # 如果調高 Dust IOU，原本因灰塵被過濾的 OK Panel 可能變成 FP
        # 條件：原本被灰塵過濾 (p_filtered) 且 max_dust_iou < new_thr -> 變回 NG -> FP
        # 注意：如果 p_max_score 已經很高 (Real NG)，那已經是 FP 了，這裡只算 "新增的 FP"
        # 但為了簡化，我們先算出 "因灰塵救回而導致的 FP"
        for dthr in DUST_THRESHOLDS:
            if p_filtered and p_dust_iou < dthr:
                pass # 這會變成 NG
                # 但要看它是否真的會變成 NG? 
                # 假設被灰塵過濾的 tile 分數通常很高 (>0.6)，所以只要不過濾就是 NG
                res['dust_recovery_fp'][dthr] = True
            else:
                res['dust_recovery_fp'][dthr] = False
        
        return res
        
    except Exception as e:
        print(f"⚠️ Error analyzing OK panel {panel_dir.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="漏檢分析工具 - 針對 GT=NG 但 AD=OK 的 Panel 進行深入分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python capi_missed_detection_analyzer.py --input_dir ./CAPI_20260212
  python capi_missed_detection_analyzer.py --input_dir ./CAPI_20260212 --gt_defect_dir ./CAPI_20260212/實際NG坐標
        """)
    parser.add_argument("--input_dir", required=True, help="圖片資料夾路徑 (包含 Panel 子目錄)")
    parser.add_argument("--gt_defect_dir", default=None, help="GT Defect 座標目錄 (預設: input_dir/實際NG坐標)")
    parser.add_argument("--ok_dir", default=None, help="OK Panel 目錄 (用於計算假陽性率 FPR)")
    parser.add_argument("--report_html", default=None, help="原始檢測產出的 HTML 報告路徑 (用於讀取原始分數)")
    parser.add_argument("--config", default="configs/capi_3f.yaml", help="配置檔路徑")
    parser.add_argument("--output_dir", default=None, help="輸出目錄 (預設: input_dir_missed_report)")
    parser.add_argument("--device", default="auto", help="運算裝置 (auto/cpu/cuda)")
    parser.add_argument("--panels", nargs="*", default=None, help="指定要分析的 Panel ID (若未指定則掃描 input_dir 所有子目錄)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ 路徑不存在: {input_dir}")
        sys.exit(1)

    gt_defect_dir = Path(args.gt_defect_dir) if args.gt_defect_dir else input_dir / "實際NG坐標"
    ok_dir = Path(args.ok_dir) if args.ok_dir else None
    
    output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / (input_dir.name + "_missed_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入原始分數與漏檢列表
    report_html_path = None
    if args.report_html:
        report_html_path = Path(args.report_html)
    else:
        # 嘗試自動尋找同層目錄的 _report/anomaly_report.html
        auto_report = input_dir.parent / f"{input_dir.name}_report" / "anomaly_report.html"
        if auto_report.exists():
            report_html_path = auto_report
    
    original_scores, html_missed_ids = load_original_scores_from_html(report_html_path)

    # 取得 Panel 列表
    # 排除的特殊資料夾名稱
    EXCLUDED_DIRS = {'實際NG坐標', '__pycache__', '.git'}
    
    if args.panels:
        panel_ids = args.panels
    elif html_missed_ids:
        # 優先：從 HTML 報告解析出漏檢 Panel (GT=NG, AD=OK)
        # 但需確認它們在 input_dir 中存在
        panel_ids = [pid for pid in html_missed_ids if (input_dir / pid).is_dir()]
        print(f"  📋 從 HTML 報告篩選出 {len(panel_ids)} 個漏檢 Panel (GT=NG, AD=OK)")
    else:
        # fallback：掃描 實際NG坐標 資料夾下有哪些 Panel
        if gt_defect_dir.exists():
            gt_panel_ids = {p.name for p in gt_defect_dir.iterdir() if p.is_dir()}
            # 只分析有 GT 資料且在 input_dir 中存在的 Panel
            panel_ids = [pid for pid in gt_panel_ids 
                         if (input_dir / pid).is_dir() and pid not in EXCLUDED_DIRS]
            print(f"  📋 從 實際NG坐標 資料夾篩選出 {len(panel_ids)} 個有 GT 的 Panel")
        else:
            # 最後 fallback：掃描所有目錄
            panel_ids = [p.name for p in input_dir.iterdir() 
                         if p.is_dir() and p.name not in EXCLUDED_DIRS 
                         and not p.name.endswith("_report")]

    print("=" * 70)
    print("🔍 漏檢分析工具 (Missed Detection Analyzer)")
    print("=" * 70)
    print(f"  📁 圖片來源:   {input_dir}")
    print(f"  📋 GT Defect:  {gt_defect_dir}")
    print(f"  📄 輸出目錄:   {output_dir}")
    print(f"  🎯 分析 Panels: {len(panel_ids)} 個")
    print(f"  ⚙️ 配置檔:     {args.config}")
    print()

    # --- 載入配置與模型 ---
    print("📦 載入配置...")
    base_dir = Path(__file__).resolve().parent
    config = CAPIConfig.from_yaml(str(base_dir / args.config))
    print(f"  ✅ 配置: {config.machine_id} | 閾值: {config.anomaly_threshold}")

    print("🧠 載入模型...")
    model_path = Path(config.model_path)
    if not model_path.is_absolute():
        model_path = base_dir / model_path
    inferencer = CAPIInferencer(
        config=config,
        model_path=str(model_path),
        device=args.device,
        threshold=config.anomaly_threshold,
        base_dir=base_dir,
    )
    print(f"  ✅ 模型已載入 (裝置: {inferencer.device})")
    print()

    # --- 逐一分析 ---
    analyses = []
    start_time = time.time()

    for i, panel_id in enumerate(panel_ids):
        panel_dir = input_dir / panel_id
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (len(panel_ids) - i - 1) if i > 0 else 0

        print(f"\n[{i+1}/{len(panel_ids)}] 📦 {panel_id} (耗時: {elapsed:.0f}s, ETA: {eta:.0f}s)")

        analysis = analyze_panel(inferencer, panel_dir, gt_defect_dir, output_dir, panel_id, original_scores)
        analyses.append(analysis)

        print(f"  → Max Score: {analysis.max_score:.4f} | Reason: {analysis.miss_reason}")

    # --- OK Panel 分析 (FPR 計算) ---
    ok_results = []
    if ok_dir and ok_dir.exists():
        print(f"\n🚀 開始分析 OK Panels (計算 FPR)... 目錄: {ok_dir}")
        ok_panels = [p for p in ok_dir.iterdir() if p.is_dir()]
        print(f"📋 發現 {len(ok_panels)} 個 OK Panels")
        
        # 依序執行
        for i, p_dir in enumerate(ok_panels):
            if i % 10 == 0:
                print(f"  ⏳ Progress: {i}/{len(ok_panels)}...")
            res = analyze_ok_panel(inferencer, p_dir)
            if res:
                ok_results.append(res)
        
        print(f"✅ OK Panel 分析完成。")
    elif args.ok_dir:
        print(f"⚠️ OK Panel 目錄不存在: {args.ok_dir}")

    # --- 生成報告 ---
    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"⏱️ 分析完成! 總耗時: {total_time:.1f}s")
    report_path = generate_html_report(analyses, output_dir, inferencer, ok_results)

    # --- 總結 ---
    print(f"\n{'=' * 70}")
    print("📊 漏檢原因總結:")
    reasons = {}
    for a in analyses:
        reasons[a.miss_reason] = reasons.get(a.miss_reason, 0) + 1
    for reason, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {cnt} 個")
    print(f"\n閾值調整影響:")
    found = sum(1 for a in analyses if a.found)
    for thr in THRESHOLDS:
        detected = sum(1 for a in analyses if a.found and a.threshold_detection.get(thr, False))
        print(f"  THR={thr}: 可檢出 {detected}/{found} ({detected/found*100:.1f}%)" if found > 0 else f"  THR={thr}: 0")
    print(f"\n📄 報告: {report_path}")


if __name__ == "__main__":
    main()
