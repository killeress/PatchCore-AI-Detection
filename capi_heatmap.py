"""
CAPI 熱力圖管理模組

負責生成、儲存和管理推論結果的熱力圖。
儲存的熱力圖可透過 Web 服務查閱。

目錄結構:
    {base_dir}/{YYYYMMDD}/{glass_id}/
        overview_{image_name}.png   — 全圖異常標記總覽
        heatmap_{image_name}_tile{N}.png — 單一 tile 熱力圖
"""

import cv2
import numpy as np
from pathlib import Path
from capi_edge_cv import clamp_median_kernel
from typing import List, Dict, Optional, Tuple, Any
import time


def build_region_zoom_panel(
    heatmap_panel: np.ndarray,
    anomaly_map: np.ndarray,
    dust_mask: Optional[np.ndarray],
    tile_size: int = 512,
) -> Optional[np.ndarray]:
    """以 anomaly_map 峰值為中心，裁切 heatmap+dust 疊加後放大至 tile_size。

    Returns:
        放大面板 (BGR) 或 None（失敗時）
    """
    try:
        amap = anomaly_map
        if len(amap.shape) == 3:
            amap = amap[:, :, 0]

        # 在原始解析度找峰值再縮放座標（避免冗餘 resize）
        peak_yx = np.unravel_index(np.argmax(amap), amap.shape)
        peak_y = int(peak_yx[0] * tile_size / amap.shape[0])
        peak_x = int(peak_yx[1] * tile_size / amap.shape[1])

        crop_sz = tile_size // 2
        y1 = max(0, peak_y - crop_sz // 2)
        x1 = max(0, peak_x - crop_sz // 2)
        y2 = min(tile_size, y1 + crop_sz)
        x2 = min(tile_size, x1 + crop_sz)
        if y2 - y1 < crop_sz:
            y1 = max(0, y2 - crop_sz)
        if x2 - x1 < crop_sz:
            x1 = max(0, x2 - crop_sz)

        zoom_crop = heatmap_panel[y1:y2, x1:x2].copy()

        if dust_mask is not None:
            dm = dust_mask
            if len(dm.shape) == 3:
                dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
            dm_resized = cv2.resize(dm, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
            dm_crop = dm_resized[y1:y2, x1:x2]
            dust_overlay = np.zeros_like(zoom_crop)
            dust_overlay[dm_crop > 0] = (0, 255, 255)
            zoom_crop = cv2.addWeighted(zoom_crop, 0.7, dust_overlay, 0.3, 0)

        panel = cv2.resize(zoom_crop, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        ctr = tile_size // 2
        cv2.drawMarker(panel, (ctr, ctr), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
        return panel
    except Exception as e:
        print(f"⚠️ Region zoom panel failed: {e}")
        return None


class HeatmapManager:
    """熱力圖生成與儲存管理"""

    def __init__(self, base_dir: str, save_format: str = "png"):
        """
        Args:
            base_dir: 熱力圖儲存根目錄
            save_format: 儲存格式 (png / jpg)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.save_format = save_format.lower()

    @staticmethod
    def _draw_split_color_header(header, info_text, verdict, verdict_color, x=10, y=30, font_scale=0.7):
        """繪製分色 header: 資訊段白色 + verdict 段跟隨判定顏色"""
        cv2.putText(header, info_text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 2)
        (info_w, _), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.putText(header, verdict, (x + info_w, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, verdict_color, 2)

    def get_save_dir(self, glass_id: str, date_str: str = "") -> Path:
        """
        取得儲存目錄路徑

        Args:
            glass_id: 玻璃 ID
            date_str: 日期字串 (YYYYMMDD)，空值則使用今天
        """
        if not date_str:
            date_str = time.strftime("%Y%m%d")
        save_dir = self.base_dir / date_str / glass_id
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def generate_heatmap_overlay(
        self,
        tile_image: np.ndarray,
        anomaly_map: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        將 anomaly_map 以熱力圖方式疊加到原始 tile 上

        Args:
            tile_image: 原始 tile 圖片 (BGR 或灰階)
            anomaly_map: 異常熱圖 (2D float)
            alpha: 熱力圖透明度

        Returns:
            疊加後的 BGR 圖片
        """
        # 確保 tile 為 BGR
        if len(tile_image.shape) == 2:
            base = cv2.cvtColor(tile_image, cv2.COLOR_GRAY2BGR)
        elif tile_image.shape[2] == 1:
            base = cv2.cvtColor(tile_image, cv2.COLOR_GRAY2BGR)
        else:
            base = tile_image.copy()

        # 將 anomaly_map 正規化到 0-255
        if anomaly_map.max() > anomaly_map.min():
            norm_map = ((anomaly_map - anomaly_map.min()) /
                        (anomaly_map.max() - anomaly_map.min()) * 255).astype(np.uint8)
        else:
            norm_map = np.zeros_like(anomaly_map, dtype=np.uint8)

        # 調整大小匹配 tile
        if norm_map.shape != base.shape[:2]:
            norm_map = cv2.resize(norm_map, (base.shape[1], base.shape[0]))

        # 套用 colormap
        heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

        # 疊加
        result = cv2.addWeighted(heatmap, alpha, base, 1 - alpha, 0)

        return result

    def save_tile_heatmap(
        self,
        save_dir: Path,
        image_name: str,
        tile_id: int,
        tile_image: np.ndarray,
        anomaly_map: np.ndarray,
        score: float,
        tile_info: Any = None,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.01,
    ) -> str:
        """
        儲存單一 tile 的組合圖 (Original | Heatmap | OMIT Crop | Dust Mask)

        完全參照 capi_missed_detection_analyzer.py 的四面板格式。
        如果沒有 OMIT/Dust 資料，則退化為 2-panel (Original | Heatmap)。

        Args:
            tile_info: TileInfo 物件 (可選)，含 omit_crop_image, dust_mask 等
            score_threshold: 判定異常的分數門檻值
            iou_threshold: 判定灰塵的 IOU 門檻值

        Returns:
            儲存的檔案路徑
        """
        tile_size = 512
        is_bright_spot = getattr(tile_info, 'is_bright_spot_detection', False) if tile_info else False

        # --- Panel 1: Original Tile ---
        orig = tile_image.copy()
        if len(orig.shape) == 2:
            orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        elif len(orig.shape) == 3 and orig.shape[2] == 1:
            orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        orig = cv2.resize(orig, (tile_size, tile_size))

        # === B0F 二值化偵測模式：2-panel (Original + Binarization Overlay) ===
        if is_bright_spot:
            is_bomb = getattr(tile_info, 'is_bomb', False) if tile_info else False
            bomb_code = getattr(tile_info, 'bomb_defect_code', '') if tile_info else ''

            # Panel 2: 二值化結果疊加在原圖上（紅色標記亮點）
            if anomaly_map is not None:
                binary_vis = orig.copy()
                binary_resized = cv2.resize(
                    (anomaly_map * 255).astype(np.uint8), (tile_size, tile_size)
                )
                # 紅色高亮標記偵測到的亮點區域
                binary_vis[binary_resized > 0] = (0, 0, 255)
                # 半透明混合
                binary_panel = cv2.addWeighted(orig, 0.5, binary_vis, 0.5, 0)
            else:
                binary_panel = orig.copy()
                cv2.putText(binary_panel, "No Binary", (150, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)

            labels = ["Original", "Binarization"]
            panels = [orig, binary_panel]

            # --- 橫向拼接 ---
            composite = np.hstack(panels)
            comp_h, comp_w = composite.shape[:2]

            # --- 標籤列 ---
            label_h = 40
            label_bar = np.zeros((label_h, comp_w, 3), dtype=np.uint8)
            for i_lbl, lbl in enumerate(labels):
                lx = i_lbl * tile_size + 10
                cv2.putText(label_bar, lbl, (lx, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # --- 頂部 Header ---
            header_h = 60
            header = np.zeros((header_h, comp_w, 3), dtype=np.uint8)

            # 取得偵測統計
            max_diff = getattr(tile_info, 'bright_spot_max_diff', 0) if tile_info else 0
            diff_thr = getattr(tile_info, 'bright_spot_diff_threshold', 0) if tile_info else 0
            spot_area = getattr(tile_info, 'bright_spot_area', 0) if tile_info else 0
            spot_min_area = getattr(tile_info, 'bright_spot_min_area', 0) if tile_info else 0

            if is_bomb:
                verdict = f"BOMB: {bomb_code} (Filtered as OK)"
                verdict_color = (255, 0, 255)
            elif score > 0:
                verdict = "NG"
                verdict_color = (0, 0, 255)
            else:
                verdict = "OK"
                verdict_color = (0, 255, 0)

            # 資訊段 + 判定段分色顯示
            info_text = f"B0F | Diff:{max_diff} Area:{spot_area}px | DiffThr:{diff_thr} MinArea:{spot_min_area} | "
            self._draw_split_color_header(header, info_text, verdict, verdict_color, y=25, font_scale=0.6)
            detail_line = f"Tile#{tile_id} | {image_name}"
            cv2.putText(header, detail_line, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            final = np.vstack([header, composite, label_bar])

            filename = f"heatmap_{image_name}_tile{tile_id}.{self.save_format}"
            filepath = save_dir / filename
            cv2.imwrite(str(filepath), final)
            return str(filepath)

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

        # --- Panel 3 & 4 & 5: OMIT Crop & Dust Mask & IOU Debug (如果有 tile_info) ---
        has_omit = False
        iou_debug_panel = None
        if tile_info is not None:
            omit_crop = getattr(tile_info, 'omit_crop_image', None)
            dust_mask = getattr(tile_info, 'dust_mask', None)
            dust_iou = getattr(tile_info, 'dust_heatmap_iou', 0.0)
            dust_detail = getattr(tile_info, 'dust_detail_text', '')
            is_dust = getattr(tile_info, 'is_suspected_dust_or_scratch', False)
            is_bomb = getattr(tile_info, 'is_bomb', False)
            bomb_code = getattr(tile_info, 'bomb_defect_code', '')
            dust_iou_debug = getattr(tile_info, 'dust_iou_debug_image', None)

            if omit_crop is not None:
                has_omit = True
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
                has_omit = True  # 仍然顯示多面板

            if dust_mask is not None:
                dust_panel = omit_panel.copy()
                dust_colored = np.zeros_like(dust_panel)
                dust_resized = cv2.resize(dust_mask, (tile_size, tile_size))
                dust_colored[dust_resized > 0] = (0, 255, 255)
                dust_panel = cv2.addWeighted(dust_panel, 0.6, dust_colored, 0.4, 0)

            else:
                dust_panel = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                cv2.putText(dust_panel, "No Dust Data", (140, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)

            # Panel 5: IOU Debug 可視化 (2x2 重疊分析圖)
            if dust_iou_debug is not None:
                iou_debug_panel = cv2.resize(dust_iou_debug, (tile_size, tile_size))
            else:
                iou_debug_panel = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                cv2.putText(iou_debug_panel, "No IOU Debug", (130, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
        else:
            is_dust = False
            is_bomb = False
            bomb_code = ''
            dust_detail = ''
            dust_iou = 0.0

        # --- 決定 Metric Name ---
        metric_name = "COV"
        if dust_detail and "IOU:" in dust_detail:
            metric_name = "IOU"

        # --- Panel 6: Region Zoom (當灰塵判定超過閾值時，放大異常區域) ---
        zoom_panel = None
        if has_omit and is_dust and anomaly_map is not None:
            zoom_panel = build_region_zoom_panel(
                heatmap_panel, anomaly_map, dust_mask, tile_size
            )

        # --- 底部獨立標籤列（不蓋到面板內容）---
        if has_omit:
            labels = ["Original", "Heatmap", "OMIT Crop", f"Dust Mask (Overall{metric_name}:{dust_iou:.3f})", f"{metric_name} Debug (G=Overlap R=Heat B=Dust)"]
            panels = [orig, heatmap_panel, omit_panel, dust_panel, iou_debug_panel]
            if zoom_panel is not None:
                labels.append("Region Zoom (Heatmap+Dust)")
                panels.append(zoom_panel)
        else:
            labels = ["Original", "Heatmap"]
            panels = [orig, heatmap_panel]

        # --- 橫向拼接 ---
        composite = np.hstack(panels)
        comp_h, comp_w = composite.shape[:2]

        # --- 標籤列（獨立一行）---
        label_h = 40
        label_bar = np.zeros((label_h, comp_w, 3), dtype=np.uint8)
        for i, lbl in enumerate(labels):
            lx = i * tile_size + 10
            cv2.putText(label_bar, lbl, (lx, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # --- 頂部 Header ---
        header_h = 60
        header = np.zeros((header_h, comp_w, 3), dtype=np.uint8)

        dust_region_cov = getattr(tile_info, 'dust_region_max_cov', 0.0) if tile_info else 0.0

        if is_bomb:
            verdict = f"BOMB: {bomb_code} (Filtered as OK)"
            verdict_color = (255, 0, 255)  # 洋紅色
        elif is_dust:
            verdict = f"DUST (Filtered as OK) | Region{metric_name}:{dust_region_cov:.3f}>={iou_threshold:.3f}"
            verdict_color = (0, 200, 255)
        elif score >= score_threshold:
            verdict = "NG (Detected)"
            verdict_color = (0, 0, 255)
        else:
            verdict = f"Score < THR"
            verdict_color = (0, 255, 255)

        header_text = f"Score: {score:.4f} | {verdict}"
        cv2.putText(header, header_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, verdict_color, 2)

        if dust_detail:
            detail_line = str(dust_detail)[:120].replace('\u2192', '->').replace('\u2190', '<-')
            detail_line = detail_line.replace(f'>={metric_name}_THR', f'>={iou_threshold:.3f}')
            detail_line = detail_line.replace(f'<{metric_name}_THR', f'<{iou_threshold:.3f}')
        else:
            detail_line = f"Tile#{tile_id} | {image_name}"
        cv2.putText(header, detail_line, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        final = np.vstack([header, composite, label_bar])

        filename = f"heatmap_{image_name}_tile{tile_id}.{self.save_format}"
        filepath = save_dir / filename
        cv2.imwrite(str(filepath), final)

        return str(filepath)

    def save_edge_defect_image(
        self,
        save_dir: Path,
        image_name: str,
        edge_index: int,
        edge_defect: Any,
        full_image: np.ndarray,
        omit_image: np.ndarray = None,
        edge_config: Any = None,
        dust_check_fn: Any = None,
        dust_iou_fn: Any = None,
        dust_iou_threshold: float = 0.3,
        dust_heatmap_top_percent: float = 5.0,
        dust_metric: str = "coverage",
    ) -> str:
        """
        儲存 CV 邊緣缺陷比較圖 (Original ROI | Defect Highlight | OMIT ROI | Defect vs OMIT Overlay)

        Panel 1: 原始 ROI
        Panel 2: 原圖 + 缺陷 BBox 標註 (不使用熱力圖疊加)
        Panel 3: OMIT 圖同座標 ROI 裁切 (無圖像處理)
        Panel 4: 缺陷 mask 與 OMIT 異物 mask 疊加比對 (IOU/COV 位置交叉驗證)

        Args:
            save_dir: 儲存目錄
            image_name: 圖片名稱
            edge_index: 缺陷序號
            edge_defect: EdgeDefect 物件
            full_image: 原始完整圖片
            omit_image: OMIT 完整圖片 (與原圖尺寸一致)
            edge_config: EdgeInspectionConfig (用於取得 blur/median 參數)
            dust_check_fn: check_dust_or_scratch_feature 函數引用
            dust_iou_fn: compute_dust_heatmap_iou 函數引用
            dust_iou_threshold: IOU/COV 閾值
            dust_heatmap_top_percent: 灰塵熱區取前百分比
            dust_metric: "coverage" 或 "iou"

        Returns:
            儲存的檔案路徑
        """
        bx, by, bw, bh = edge_defect.bbox
        max_diff = edge_defect.max_diff
        area = edge_defect.area
        side = edge_defect.side
        is_dust = getattr(edge_defect, 'is_suspected_dust_or_scratch', False)
        is_bomb = getattr(edge_defect, 'is_bomb', False)
        is_cv_ok = getattr(edge_defect, 'is_cv_ok', False)
        bomb_code = getattr(edge_defect, 'bomb_defect_code', '')
        img_h, img_w = full_image.shape[:2]

        # 計算 ROI 區域 (擴大一些以提供上下文)
        padding = 100
        roi_x1 = max(0, bx - padding)
        roi_y1 = max(0, by - padding)
        roi_x2 = min(img_w, bx + bw + padding)
        roi_y2 = min(img_h, by + bh + padding)

        roi_raw = full_image[roi_y1:roi_y2, roi_x1:roi_x2].copy()

        # 灰階用於計算 diff mask
        if len(roi_raw.shape) == 3:
            roi_gray = cv2.cvtColor(roi_raw, cv2.COLOR_BGR2GRAY)
        elif len(roi_raw.shape) == 2:
            roi_gray = roi_raw
        else:
            roi_gray = roi_raw.reshape(roi_raw.shape[0], roi_raw.shape[1])

        # ROI 轉 BGR 用於顯示
        roi = roi_raw.copy()
        if len(roi.shape) == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        elif len(roi.shape) == 3 and roi.shape[2] == 1:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        # ── Panel 1: 原始 ROI ──
        panel_orig = roi.copy()

        # ── Panel 2: Defect Highlight (原圖 + BBox 標註，不使用熱力圖) ──
        panel_highlight = roi.copy()

        # 在原圖上繪製缺陷 bbox 邊框
        rel_x = bx - roi_x1
        rel_y = by - roi_y1
        if is_bomb:
            box_color = (255, 0, 255)   # 洋紅色
        elif is_dust:
            box_color = (0, 200, 255)   # 橘色
        else:
            box_color = (0, 0, 255)     # 紅色
        cv2.rectangle(panel_highlight, (rel_x, rel_y), (rel_x + bw, rel_y + bh), box_color, 3)

        # 產生 defect binary mask (與 _inspect_side 相同邏輯)，供 Panel 4 使用
        k = 3
        mk = 65
        if edge_config is not None:
            k = getattr(edge_config, 'blur_kernel', 3)
            mk = getattr(edge_config, 'median_kernel', 65)
        blurred = cv2.GaussianBlur(roi_gray, (k, k), 0)
        mk = clamp_median_kernel(mk, min(roi_gray.shape[:2]) - 1)
        bg = cv2.medianBlur(blurred, mk)
        diff = cv2.absdiff(blurred, bg)

        # 取得 threshold (根據邊的 config)
        edge_threshold = 5  # 預設值
        if edge_config is not None:
            side_cfg = getattr(edge_config, side, None)
            if side_cfg is not None:
                edge_threshold = getattr(side_cfg, 'threshold', 5)
        _, defect_mask = cv2.threshold(diff, edge_threshold, 255, cv2.THRESH_BINARY)

        # 只保留缺陷 BBox 範圍內的像素（避免整個 ROI 的紋理雜訊被納入 IOU/COV 計算）
        bbox_only_mask = np.zeros_like(defect_mask)
        bbox_only_mask[rel_y:rel_y + bh, rel_x:rel_x + bw] = 255
        defect_mask = cv2.bitwise_and(defect_mask, bbox_only_mask)

        # 統一面板大小
        panel_h = 400
        scale = panel_h / max(panel_orig.shape[0], 1)
        panel_w = int(panel_orig.shape[1] * scale)
        panel_w = max(panel_w, 200)
        panel_orig = cv2.resize(panel_orig, (panel_w, panel_h))
        panel_highlight = cv2.resize(panel_highlight, (panel_w, panel_h))

        panels = [panel_orig, panel_highlight]
        labels = ["Original ROI", "Defect Highlight"]

        # ── Panel 3 & 4: OMIT ROI + Defect vs OMIT Overlay ──
        metric_name = "COV" if dust_metric == "coverage" else "IOU"
        surface_iou = 0.0
        is_surface = False

        if omit_image is not None:
            oh, ow = omit_image.shape[:2]

            # Panel 3: OMIT 同座標 ROI 裁切 (無圖像處理)
            omit_roi_x1 = max(0, min(roi_x1, ow))
            omit_roi_y1 = max(0, min(roi_y1, oh))
            omit_roi_x2 = max(0, min(roi_x2, ow))
            omit_roi_y2 = max(0, min(roi_y2, oh))

            omit_roi_raw = omit_image[omit_roi_y1:omit_roi_y2, omit_roi_x1:omit_roi_x2].copy()

            # 轉 BGR 用於顯示
            omit_roi = omit_roi_raw.copy()
            if len(omit_roi.shape) == 2:
                omit_roi = cv2.cvtColor(omit_roi, cv2.COLOR_GRAY2BGR)
            elif len(omit_roi.shape) == 3 and omit_roi.shape[2] == 1:
                omit_roi = cv2.cvtColor(omit_roi, cv2.COLOR_GRAY2BGR)

            omit_panel = cv2.resize(omit_roi, (panel_w, panel_h))

            # Panel 4: Defect vs OMIT Overlay (IOU/COV 位置交叉驗證)
            # Step 1: 在 OMIT ROI 上偵測異物 (灰塵/刮痕)
            dust_mask_omit = None
            if dust_check_fn is not None:
                try:
                    is_dust_detected, dust_mask_omit, bright_ratio, detail_text = dust_check_fn(omit_roi_raw)
                except Exception as e:
                    print(f"⚠️ Edge OMIT dust check failed: {e}")
                    is_dust_detected = False

            # Step 2: 計算 defect_mask 與 dust_mask 的 IOU/COV
            if dust_mask_omit is not None and is_dust_detected:
                # 將 defect_mask resize 到與 dust_mask_omit 相同尺寸
                if defect_mask.shape != dust_mask_omit.shape[:2]:
                    defect_mask_resized = cv2.resize(defect_mask,
                                                      (dust_mask_omit.shape[1] if len(dust_mask_omit.shape) > 1 else dust_mask_omit.shape[0],
                                                       dust_mask_omit.shape[0]),
                                                      interpolation=cv2.INTER_NEAREST)
                else:
                    defect_mask_resized = defect_mask

                # 確保 dust_mask 為單通道
                dust_mask_single = dust_mask_omit
                if len(dust_mask_single.shape) == 3:
                    dust_mask_single = cv2.cvtColor(dust_mask_single, cv2.COLOR_BGR2GRAY)

                defect_bool = defect_mask_resized > 0
                dust_bool = dust_mask_single > 0

                intersection = np.count_nonzero(defect_bool & dust_bool)

                if dust_metric == "coverage":
                    defect_area_px = np.count_nonzero(defect_bool)
                    surface_iou = float(intersection / defect_area_px) if defect_area_px > 0 else 0.0
                else:
                    union = np.count_nonzero(defect_bool | dust_bool)
                    surface_iou = float(intersection / union) if union > 0 else 0.0

                is_surface = surface_iou >= dust_iou_threshold

            # 產生 Panel 4 可視化圖 (R=僅缺陷, B=僅異物, G=重疊)
            overlay_panel = omit_panel.copy()
            if dust_mask_omit is not None and is_dust_detected:
                # resize masks to panel size
                defect_vis = cv2.resize(defect_mask, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
                dust_vis_mask = dust_mask_single if dust_mask_single is not None else dust_mask_omit
                if len(dust_vis_mask.shape) == 3:
                    dust_vis_mask = cv2.cvtColor(dust_vis_mask, cv2.COLOR_BGR2GRAY)
                dust_vis = cv2.resize(dust_vis_mask, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)

                # 只顯示 BBox 周圍區域的 mask (加 margin 以顯示附近異物)
                margin = 30
                vis_rel_x = int(rel_x * scale)
                vis_rel_y = int(rel_y * scale)
                vis_bw = int(bw * scale)
                vis_bh = int(bh * scale)
                vis_x1 = max(0, vis_rel_x - margin)
                vis_y1 = max(0, vis_rel_y - margin)
                vis_x2 = min(panel_w, vis_rel_x + vis_bw + margin)
                vis_y2 = min(panel_h, vis_rel_y + vis_bh + margin)

                # 在 bbox 區域外清除 mask（避免散佈的雜點）
                bbox_vis_mask = np.zeros((panel_h, panel_w), dtype=np.uint8)
                bbox_vis_mask[vis_y1:vis_y2, vis_x1:vis_x2] = 255
                defect_vis = cv2.bitwise_and(defect_vis, bbox_vis_mask)
                dust_vis = cv2.bitwise_and(dust_vis, bbox_vis_mask)

                defect_b = defect_vis > 0
                dust_b = dust_vis > 0
                overlap_b = defect_b & dust_b
                only_defect = defect_b & ~dust_b
                only_dust = dust_b & ~defect_b

                color_layer = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                color_layer[overlap_b] = (0, 255, 0)    # 綠色 = 重疊
                color_layer[only_defect] = (0, 0, 255)   # 紅色 = 僅缺陷
                color_layer[only_dust] = (255, 0, 0)     # 藍色 = 僅異物

                # 畫 BBox 外框以利辨識
                cv2.rectangle(overlay_panel,
                              (vis_rel_x, vis_rel_y),
                              (vis_rel_x + vis_bw, vis_rel_y + vis_bh),
                              (255, 255, 0), 1)

                overlay_panel = cv2.addWeighted(overlay_panel, 0.5, color_layer, 0.5, 0)
            else:
                cv2.putText(overlay_panel, "No OMIT Dust Data", (10, panel_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

            panels.extend([omit_panel, overlay_panel])
            labels.extend(["OMIT ROI", f"Overlay ({metric_name}:{surface_iou:.3f})"])

            # 判定結果僅用於 heatmap 顯示，不回寫到 edge_defect
            # (推論結果的 is_suspected_dust_or_scratch 應在 Phase 3 中設定，
            #  避免 async heatmap 儲存與同步判定之間的時序不一致)
            if is_surface and not is_bomb:
                is_dust = True

        # 橫向拼接 (加入明顯深灰間隔，避免相連處被誤認為缺陷線)
        gap_w = 10
        gap = np.full((panel_h, gap_w, 3), 80, dtype=np.uint8)
        
        spaced_panels = []
        for i, p in enumerate(panels):
            spaced_panels.append(p)
            if i < len(panels) - 1:
                spaced_panels.append(gap)

        composite = np.hstack(spaced_panels)
        comp_h, comp_w = composite.shape[:2]

        # Header
        header_h = 50
        header = np.zeros((header_h, comp_w, 3), dtype=np.uint8)

        # 取得判定參數 (用於 header 顯示)
        threshold_used = getattr(edge_defect, 'threshold_used', 0)
        min_area_used = getattr(edge_defect, 'min_area_used', 0)

        if is_cv_ok:
            verdict = "CV OK"
            verdict_color = (0, 255, 0)
        elif is_bomb:
            verdict = f"BOMB: {bomb_code} (Filtered as OK)"
            verdict_color = (255, 0, 255)
        elif is_dust:
            verdict = f"SURFACE ({metric_name}:{surface_iou:.3f}) (Filtered as OK)"
            verdict_color = (0, 200, 255)
        else:
            verdict = "NG"
            verdict_color = (0, 0, 255)

        # 組合 header: 實際數值 + 判定閾值 + 結果
        threshold_info = ""
        if threshold_used > 0 or min_area_used > 0:
            threshold_info = f" | Thr:{threshold_used} MinArea:{min_area_used}"
        header_text = f"CV Edge [v4]: {side} | Diff:{max_diff:.0f} Area:{area}px{threshold_info} | {verdict}"

        # 分段繪製: 數值部分白色，verdict 部分跟隨判定顏色
        verdict_start = header_text.rfind("| " + verdict)
        info_part = header_text[:verdict_start + 2] if verdict_start > 0 else ""
        if info_part:
            self._draw_split_color_header(header, info_part, verdict, verdict_color, y=30, font_scale=0.7)
        else:
            cv2.putText(header, header_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, verdict_color, 2)

        # Label bar
        label_h = 40
        label_bar = np.zeros((label_h, comp_w, 3), dtype=np.uint8)
        for i, lbl in enumerate(labels):
            lx = i * (panel_w + gap_w) + 10
            cv2.putText(label_bar, lbl, (lx, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        final = np.vstack([header, composite, label_bar])

        filename = f"edge_{image_name}_{side}_{edge_index}.{self.save_format}"
        filepath = save_dir / filename
        cv2.imwrite(str(filepath), final)

        return str(filepath)

    def save_overview(
        self,
        save_dir: Path,
        image_name: str,
        overview_image: np.ndarray,
    ) -> str:
        """
        儲存全圖總覽

        Args:
            save_dir: 儲存目錄
            image_name: 圖片名稱 (不含副檔名)
            overview_image: 已標記異常的總覽圖 (由 CAPIInferencer.visualize_inference_result 產生)

        Returns:
            儲存的檔案路徑
        """
        filename = f"overview_{image_name}.{self.save_format}"
        filepath = save_dir / filename

        # 縮小儲存 (原圖可能非常大)
        max_dim = 2000
        h, w = overview_image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(overview_image, (new_w, new_h))
            cv2.imwrite(str(filepath), resized)
        else:
            cv2.imwrite(str(filepath), overview_image)

        return str(filepath)

    def save_panel_heatmaps(
        self,
        glass_id: str,
        results: list,
        inferencer: Any,
        save_overview: bool = True,
        save_tile_detail: bool = True,
        date_str: str = "",
        omit_image: np.ndarray = None,
    ) -> Dict:
        """
        儲存整個 Panel 的所有熱力圖

        Args:
            glass_id: 玻璃 ID
            results: List[ImageResult] - 推論結果
            inferencer: CAPIInferencer 實例 (用於生成 overview)
            save_overview: 是否儲存全圖總覽
            save_tile_detail: 是否儲存 tile 細節
            date_str: 日期字串
            omit_image: OMIT 原圖 (用於邊緣缺陷 OMIT 交叉驗證)

        Returns:
            {"dir": save_dir, "files": [...]}
        """
        save_dir = self.get_save_dir(glass_id, date_str)
        saved_files = []

        for result in results:
            image_name = result.image_path.stem
            
            # 取得當前圖片實際的 threshold
            active_threshold = getattr(inferencer, 'threshold', 0.5)
            if hasattr(inferencer, '_get_image_prefix') and hasattr(inferencer, '_get_threshold_for_prefix'):
                try:
                    img_prefix = inferencer._get_image_prefix(result.image_path.name)
                    active_threshold = inferencer._get_threshold_for_prefix(img_prefix)
                except Exception:
                    pass

            # 儲存全圖總覽 (有 tile 異常或 edge 缺陷時)
            has_edge = hasattr(result, 'edge_defects') and result.edge_defects
            if save_overview and (result.anomaly_tiles or has_edge):
                try:
                    overview = inferencer.visualize_inference_result(
                        result.image_path, result
                    )
                    path = self.save_overview(save_dir, image_name, overview)
                    saved_files.append(path)
                except Exception as e:
                    print(f"⚠️ 儲存 overview 失敗 ({image_name}): {e}")

            # 儲存每個 anomaly tile 的組合圖
            if save_tile_detail:
                for tile, score, anomaly_map in result.anomaly_tiles:
                    if anomaly_map is not None:
                        try:
                            path = self.save_tile_heatmap(
                                save_dir, image_name, tile.tile_id,
                                tile.image, anomaly_map, score,
                                tile_info=tile,
                                score_threshold=active_threshold,
                                iou_threshold=inferencer.config.dust_heatmap_iou_threshold,
                            )
                            saved_files.append(path)
                        except Exception as e:
                            print(f"⚠️ 儲存 tile heatmap 失敗 ({image_name} tile{tile.tile_id}): {e}")

            # 儲存 CV 邊緣缺陷比較圖
            if save_tile_detail and hasattr(result, 'edge_defects') and result.edge_defects:
                try:
                    full_img = cv2.imread(str(result.image_path), cv2.IMREAD_UNCHANGED)
                    if full_img is not None:
                        # 取得 edge config 和 dust 相關函數
                        edge_config = None
                        dust_check_fn = None
                        dust_iou_threshold = 0.3
                        dust_metric = "coverage"
                        if hasattr(inferencer, 'edge_inspector') and inferencer.edge_inspector:
                            edge_config = inferencer.edge_inspector.config
                        if hasattr(inferencer, 'check_dust_or_scratch_feature'):
                            dust_check_fn = inferencer.check_dust_or_scratch_feature
                        if hasattr(inferencer, 'config'):
                            dust_iou_threshold = getattr(inferencer.config, 'dust_heatmap_iou_threshold', 0.3)
                            dust_metric = getattr(inferencer.config, 'dust_heatmap_metric', 'coverage')

                        for ei, edge in enumerate(result.edge_defects):
                            try:
                                edge_path = self.save_edge_defect_image(
                                    save_dir,
                                    image_name,
                                    ei,
                                    edge,
                                    full_img,
                                    omit_image=omit_image,
                                    edge_config=edge_config,
                                    dust_check_fn=dust_check_fn,
                                    dust_iou_threshold=dust_iou_threshold,
                                    dust_metric=dust_metric,
                                )
                                saved_files.append(edge_path)
                                # 回寫路徑到 edge 物件 (供後續 DB 儲存使用)
                                edge._heatmap_path = edge_path
                            except Exception as e:
                                print(f"⚠️ 儲存 edge defect 圖失敗 ({image_name} {edge.side}_{ei}): {e}")
                except Exception as e:
                    print(f"⚠️ 載入原圖失敗 ({image_name}): {e}")

        return {
            "dir": str(save_dir),
            "files": saved_files,
        }

    def get_heatmap_url(self, filepath: str, web_base: str = "/heatmaps") -> str:
        """
        將檔案路徑轉換為 Web URL

        Args:
            filepath: 熱力圖檔案的絕對路徑
            web_base: Web 服務的靜態檔案基礎路徑
        """
        try:
            rel_path = Path(filepath).relative_to(self.base_dir)
            return f"{web_base}/{rel_path.as_posix()}"
        except ValueError:
            return filepath


if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("Heatmap Manager Test")
    print("=" * 60)

    # 建立測試用 manager
    test_dir = Path(tempfile.mkdtemp()) / "heatmaps"
    manager = HeatmapManager(str(test_dir))

    # 建立假的 tile 和 anomaly_map
    tile_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    anomaly_map = np.random.rand(512, 512).astype(np.float32)
    anomaly_map[200:300, 200:300] = 1.0  # 模擬異常區域

    # 測試 overlay 生成
    overlay = manager.generate_heatmap_overlay(tile_img, anomaly_map)
    print(f"✅ Overlay shape: {overlay.shape}")

    # 測試儲存
    save_dir = manager.get_save_dir("TEST_GLASS_001")
    path = manager.save_tile_heatmap(save_dir, "G0F00000", 15, tile_img, anomaly_map, 0.85)
    print(f"✅ Tile heatmap saved: {path}")

    # 測試 URL 生成
    url = manager.get_heatmap_url(path)
    print(f"✅ Heatmap URL: {url}")

    # 清理
    import shutil
    shutil.rmtree(str(test_dir))
    print("✅ All tests passed!")
