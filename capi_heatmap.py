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
from capi_edge_cv import clamp_median_kernel, compute_fg_aware_diff
from typing import List, Dict, Optional, Tuple, Any
import time


def _blend_color_on_mask(image: np.ndarray, mask: np.ndarray, color, alpha: float = 0.5):
    """在 mask > 0 的像素上疊加半透明顏色 (in-place 修改 image)"""
    pixels = mask > 0
    if np.any(pixels):
        image[pixels] = (
            image[pixels].astype(np.float32) * (1 - alpha) +
            np.array(color, dtype=np.float32) * alpha
        ).clip(0, 255).astype(np.uint8)


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    """將 grayscale / 單通道 ndarray 統一轉為 BGR，多通道直接 copy"""
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def render_pc_masked_roi(roi_bgr: np.ndarray, fg_mask: Optional[np.ndarray]) -> np.ndarray:
    """Panel 1: Original ROI，panel 外區塗暗紅 + 對角斜線標示 masked 區"""
    panel = roi_bgr.copy()
    if fg_mask is None:
        return panel
    panel[fg_mask == 0] = (0, 0, 60)
    # 對角斜線 mask (向量化，不用 Python loop)
    h, w = panel.shape[:2]
    yy, xx = np.indices((h, w))
    hatch = ((xx - yy) % 12 == 0)
    stripe = hatch & (fg_mask == 0)
    panel[stripe] = (0, 0, 120)
    return panel


def render_pc_overlay(
    roi_bgr: np.ndarray,
    fg_mask: Optional[np.ndarray],
    anomaly_map: Optional[np.ndarray],
) -> np.ndarray:
    """Panel 2: 將 anomaly_map 歸一化 + JET colormap 半透明疊加於 ROI 上，panel 外塗暗紅"""
    h, w = roi_bgr.shape[:2]
    panel = roi_bgr.copy()
    if anomaly_map is None or anomaly_map.size == 0:
        cv2.putText(panel, "No anomaly map", (20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return panel
    amap = np.asarray(anomaly_map, dtype=np.float32)
    if amap.shape[:2] != (h, w):
        amap = cv2.resize(amap, (w, h))
    if fg_mask is not None:
        amap = amap * (fg_mask.astype(np.float32) / 255.0)
    peak = float(np.max(amap)) if amap.size > 0 else 0.0
    if peak > 1e-6:
        amap_u8 = (amap / peak * 255.0).clip(0, 255).astype(np.uint8)
    else:
        amap_u8 = np.zeros((h, w), dtype=np.uint8)
    heatmap = cv2.applyColorMap(amap_u8, cv2.COLORMAP_JET)
    panel = cv2.addWeighted(roi_bgr, 0.5, heatmap, 0.5, 0)
    if fg_mask is not None:
        panel[fg_mask == 0] = (0, 0, 60)
    return panel


def build_region_zoom_panels(
    heatmap_binary: Optional[np.ndarray],
    dust_mask: Optional[np.ndarray],
    region_details: Optional[list],
    tile_size: int = 512,
    metric_name: str = "COV",
    iou_threshold: float = 0.01,
    max_panels: int = 3,
) -> List[Tuple[np.ndarray, str]]:
    """為每個異常區域生成放大面板（二值化 heatmap + 灰塵遮罩疊加 + 計算數值標註）。

    Args:
        heatmap_binary: 二值化 heatmap（來自 check_dust_per_region）
        dust_mask: 灰塵遮罩（原始尺寸）
        region_details: per-region 判定結果列表
        tile_size: 面板大小
        metric_name: "COV" 或 "IOU"
        iou_threshold: 判定閾值
        max_panels: 最多生成幾張

    Returns:
        [(panel_image, label_text), ...] 最多 max_panels 筆
    """
    if not region_details or heatmap_binary is None:
        return []

    # 按 max_score 降序，取前 max_panels 個
    sorted_regions = sorted(region_details, key=lambda r: r["max_score"], reverse=True)[:max_panels]

    # 準備 tile_size 尺寸的二值化圖與灰塵遮罩
    heat_bin = heatmap_binary
    if len(heat_bin.shape) == 3:
        heat_bin = heat_bin[:, :, 0]
    heat_resized = cv2.resize(heat_bin, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)

    dm_resized = None
    if dust_mask is not None:
        dm = dust_mask
        if len(dm.shape) == 3:
            dm = cv2.cvtColor(dm, cv2.COLOR_BGR2GRAY)
        dm_resized = cv2.resize(dm, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)

    h_src, w_src = heatmap_binary.shape[:2]
    results = []

    for region in sorted_regions:
        try:
            # 峰值座標縮放到 tile_size 空間
            peak_y = int(region["peak_yx"][0] * tile_size / h_src)
            peak_x = int(region["peak_yx"][1] * tile_size / w_src)

            crop_sz = tile_size // 2
            y1 = max(0, peak_y - crop_sz // 2)
            x1 = max(0, peak_x - crop_sz // 2)
            y2 = min(tile_size, y1 + crop_sz)
            x2 = min(tile_size, x1 + crop_sz)
            if y2 - y1 < crop_sz:
                y1 = max(0, y2 - crop_sz)
            if x2 - x1 < crop_sz:
                x1 = max(0, x2 - crop_sz)

            # 底圖：二值化 heatmap 區域（白色）在黑色背景上
            heat_crop = heat_resized[y1:y2, x1:x2]
            base = np.zeros((crop_sz, crop_sz, 3), dtype=np.uint8)
            base[heat_crop > 0] = (255, 255, 255)  # 白色 = 熱區

            # 疊加灰塵遮罩（黃色）與重疊區域（綠色）
            if dm_resized is not None:
                dm_crop = dm_resized[y1:y2, x1:x2]
                dust_only = (dm_crop > 0) & (heat_crop == 0)
                overlap = (heat_crop > 0) & (dm_crop > 0)
                base[dust_only] = (0, 255, 255)   # 黃色 (BGR) = 僅灰塵
                base[overlap] = (0, 255, 0)        # 綠色 = 重疊

            panel = cv2.resize(base, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)

            # 標註計算過程
            cov = region["coverage"]
            area = region["area"]
            dust_ol = region["dust_overlap"]
            is_dust = region["is_dust"]
            tag = "DUST" if is_dust else "REAL"
            tag_color = (0, 200, 255) if is_dust else (0, 0, 255)

            cv2.putText(panel, f"{tag}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, tag_color, 2)
            cv2.putText(panel, f"{metric_name}: {dust_ol}/{area} = {cov:.4f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            peak_in = region.get("peak_in_dust", True)
            thr_str = f"Thr:{iou_threshold}"
            if is_dust:
                reason = f"PeakInDust ({thr_str}) -> DUST"
            elif cov < iou_threshold:
                reason = f"{metric_name}<{thr_str} -> REAL"
            else:
                reason = f"PeakNotInDust ({thr_str}) -> REAL"
            cv2.putText(panel, reason, (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, tag_color, 1)

            # 圖例
            legend_y = tile_size - 15
            cv2.putText(panel, "White=Heat  Yellow=Dust  Green=Overlap", (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

            label = f"Region#{region['label_id']} {tag} {metric_name}:{cov:.3f}"
            results.append((panel, label))
        except Exception as e:
            print(f"⚠️ Region zoom panel #{region.get('label_id', '?')} failed: {e}")

    return results


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

        # --- Draw two-stage feature markers on Original & Heatmap ---
        ts_features = getattr(tile_info, 'dust_two_stage_features', None) if tile_info else None
        if ts_features:
            tile_h_orig, tile_w_orig = getattr(tile_info, 'height', tile_size), getattr(tile_info, 'width', tile_size)
            sx = tile_size / tile_w_orig
            sy = tile_size / tile_h_orig
            tile_x0 = getattr(tile_info, 'x', 0)
            tile_y0 = getattr(tile_info, 'y', 0)
            for feat in ts_features:
                abs_x, abs_y = feat["abs_pos"]
                # abs_pos is relative to tile origin, convert to panel coords
                dx = int(abs_x * sx)
                dy = int(abs_y * sy)
                if not (0 <= dx < tile_size and 0 <= dy < tile_size):
                    continue
                is_dust_feat = feat.get("is_dust", False)
                color = (0, 200, 0) if is_dust_feat else (0, 0, 255)
                marker_r = max(8, int(feat.get("area", 5) ** 0.5 * 2))
                # Draw on both Original and Heatmap panels
                for panel in [orig, heatmap_panel]:
                    cv2.circle(panel, (dx, dy), marker_r, color, 2)
                    label = "D" if is_dust_feat else "R"
                    cv2.putText(panel, label, (dx + marker_r + 3, dy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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

        # --- Panel 6+: Region Zoom (逐區域放大，最多 3 張) ---
        zoom_results = []
        has_region_details = tile_info is not None and getattr(tile_info, 'dust_region_details', None)
        if has_omit and has_region_details:
            region_details = getattr(tile_info, 'dust_region_details', None)
            heatmap_binary = getattr(tile_info, 'dust_heatmap_binary', None)
            zoom_results = build_region_zoom_panels(
                heatmap_binary, dust_mask, region_details,
                tile_size=tile_size,
                metric_name=metric_name,
                iou_threshold=iou_threshold,
            )

        # --- 底部獨立標籤列（不蓋到面板內容）---
        if has_omit:
            is_two_stage = "TWO_STAGE" in dust_detail
            if is_two_stage:
                debug_label = "TwoStage Debug (G=Dust R=RealNG B=DustMask)"
            else:
                debug_label = f"{metric_name} Debug (G=Dust R=RealNG B=DustOnly)"
            labels = ["Original", "Heatmap", "OMIT Crop", f"Dust Mask (Overall{metric_name}:{dust_iou:.3f})", debug_label]
            panels = [orig, heatmap_panel, omit_panel, dust_panel, iou_debug_panel]
            for zoom_img, zoom_label in zoom_results:
                panels.append(zoom_img)
                labels.append(zoom_label)
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

        scratch_score_val = float(getattr(tile_info, 'scratch_score', 0.0) or 0.0) if tile_info else 0.0
        scratch_filtered_val = bool(getattr(tile_info, 'scratch_filtered', False)) if tile_info else False

        if scratch_filtered_val and not is_bomb and not is_dust:
            verdict = f"[SCR] Scratch Filter OK (score={scratch_score_val:.3f})"
            verdict_color = (180, 220, 100)  # 青綠色

        header_text = f"Score: {score:.4f} | {verdict}"
        cv2.putText(header, header_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, verdict_color, 2)

        if dust_detail:
            detail_line = str(dust_detail)[:120].replace('\u2192', '->').replace('\u2190', '<-')
            detail_line = detail_line.replace(f'>={metric_name}_THR', f'>={iou_threshold:.3f}')
            detail_line = detail_line.replace(f'<{metric_name}_THR', f'<{iou_threshold:.3f}')
        else:
            detail_line = f"Tile#{tile_id} | {image_name}"
        if scratch_score_val > 0 and not scratch_filtered_val:
            detail_line = f"{detail_line} | SCR:{scratch_score_val:.3f}"
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
        # Phase 5: inspector_mode='patchcore' → PC renderer
        # Phase 6: inspector_mode='fusion' + source_inspector='patchcore' → PC renderer
        inspector_mode = getattr(edge_defect, 'inspector_mode', 'cv')
        source_inspector = getattr(edge_defect, 'source_inspector', '')
        if inspector_mode == 'patchcore' or (
            inspector_mode == 'fusion' and source_inspector == 'patchcore'
        ):
            return self._save_patchcore_edge_image(
                save_dir, image_name, edge_index, edge_defect,
                full_image, omit_image,
                dust_check_fn=dust_check_fn,
            )
        # Phase 7.2-B: fusion 的 CV defect 走新 3 板 renderer
        if inspector_mode == 'fusion' and source_inspector == 'cv':
            return self._save_cv_fusion_edge_image(
                save_dir, image_name, edge_index, edge_defect,
                full_image, omit_image,
                edge_config=edge_config,
                dust_check_fn=dust_check_fn,
                dust_iou_threshold=dust_iou_threshold,
                dust_metric=dust_metric,
                panel_polygon=getattr(edge_defect, 'panel_polygon', None),
            )
        # 四邊 CV / 非 fusion CV 繼續走舊主分支（保留現有 4 板邏輯）

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

        # Defect BBox 相對 ROI 的座標
        rel_x = bx - roi_x1
        rel_y = by - roi_y1

        # ── 產生 defect binary mask (供 Panel 2 像素標記和 Panel 4 IOU/COV 計算) ──
        # 優先使用 inspect_roi 產生的真實過濾後 mask，避免 heatmap 路徑與判定路徑分歧
        cv_mask = getattr(edge_defect, 'cv_filtered_mask', None)
        cv_mask_offset = getattr(edge_defect, 'cv_mask_offset', None)
        defect_mask = None
        if cv_mask is not None and cv_mask_offset is not None:
            mo_x, mo_y = cv_mask_offset
            mh, mw = cv_mask.shape[:2]
            # 貼到 panel (padding 後 roi) 的對位 offset
            paste_x = mo_x - roi_x1
            paste_y = mo_y - roi_y1
            if (0 <= paste_x < roi_gray.shape[1] and 0 <= paste_y < roi_gray.shape[0]
                    and paste_x + mw <= roi_gray.shape[1] and paste_y + mh <= roi_gray.shape[0]):
                defect_mask = np.zeros(roi_gray.shape[:2], dtype=np.uint8)
                defect_mask[paste_y:paste_y + mh, paste_x:paste_x + mw] = cv_mask

        if defect_mask is None:
            # Fallback：舊紀錄 / 四邊檢測 / cv_mask 缺失時重算 (保持向下相容)
            k = 3
            mk = 65
            if edge_config is not None:
                k = getattr(edge_config, 'blur_kernel', 3)
                mk = getattr(edge_config, 'median_kernel', 65)
            blurred = cv2.GaussianBlur(roi_gray, (k, k), 0)
            mk = clamp_median_kernel(mk, min(roi_gray.shape[:2]) - 1)

            if side == "aoi_edge":
                _, diff = compute_fg_aware_diff(blurred, roi_gray, mk)
            else:
                bg = cv2.medianBlur(blurred, mk)
                diff = cv2.absdiff(blurred, bg)

            edge_threshold = edge_config.get_threshold_for_side(side) if edge_config is not None else 5
            _, defect_mask = cv2.threshold(diff, edge_threshold, 255, cv2.THRESH_BINARY)

            # 只保留缺陷 BBox 範圍內的像素（避免整個 ROI 的紋理雜訊被納入 IOU/COV 計算）
            bbox_only_mask = np.zeros_like(defect_mask)
            bbox_only_mask[rel_y:rel_y + bh, rel_x:rel_x + bw] = 255
            defect_mask = cv2.bitwise_and(defect_mask, bbox_only_mask)

        # ── Panel 1: 原始 ROI ──
        panel_orig = roi.copy()

        # ── Panel 2: Defect Highlight (缺陷像素紅色標記，取代方形框) ──
        # vis_mask = defect_mask 做 3×3 dilate，只用於視覺塗色 (讓 1-3 px 寬薄線
        # resize 到 panel_h=400 後不被 INTER_LINEAR 淡化成看不見)。
        # defect_mask 本身保持原狀，後續 COV/IOU 計算仍用精確面積。
        vis_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        vis_mask = cv2.dilate(defect_mask, vis_kernel, iterations=1)
        panel_highlight = roi.copy()
        if is_bomb:
            highlight_color = (255, 0, 255)   # 洋紅
        elif is_dust:
            highlight_color = (0, 200, 255)   # 橘色
        else:
            highlight_color = (0, 0, 255)     # 紅色
        _blend_color_on_mask(panel_highlight, vis_mask, highlight_color)

        # 統一面板大小
        panel_h = 400
        scale = panel_h / max(panel_orig.shape[0], 1)
        panel_w = int(panel_orig.shape[1] * scale)
        panel_w = max(panel_w, 200)
        panel_orig = cv2.resize(panel_orig, (panel_w, panel_h))
        panel_highlight = cv2.resize(panel_highlight, (panel_w, panel_h))

        # Phase 6: fusion 模式下 CV 來源 → 左上角 [CV] 角標
        if inspector_mode == 'fusion' and source_inspector == 'cv':
            cv2.rectangle(panel_highlight, (5, 5), (60, 28), (0, 0, 0), -1)
            cv2.putText(panel_highlight, "[CV]", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        # 取得判定參數 + 推算 CV OK 原因（corner 標記與下方 header verdict 共用，確保同步）
        threshold_used = getattr(edge_defect, 'threshold_used', 0)
        min_area_used = getattr(edge_defect, 'min_area_used', 0)
        min_max_diff_used = getattr(edge_defect, 'min_max_diff_used', 0)
        cv_ok_reason = ""
        if is_cv_ok:
            # 推算 OK 原因：由淺入深依序排除
            #   1. diff 未達 threshold → Diff<Thr
            #   2. area 未達 min_area → Area<Min
            #   3. diff 達 threshold 但 < min_max_diff (低對比紋理雜訊) → Diff<MinMaxDiff
            #   4. 以上皆不成立 → 必是 solidity/morph_open 過濾 → Shape filtered
            if threshold_used > 0 and max_diff < threshold_used:
                cv_ok_reason = "Diff<Thr"
            elif min_area_used > 0 and area < min_area_used:
                cv_ok_reason = "Area<Min"
            elif min_max_diff_used > 0 and max_diff < min_max_diff_used:
                cv_ok_reason = f"Diff<MinMaxDiff({min_max_diff_used})"
            elif threshold_used > 0 or min_area_used > 0 or min_max_diff_used > 0:
                cv_ok_reason = "Shape filtered"

        # CV OK 時在 Defect Highlight 角落加標：表明紅色像素是被過濾的候選，非真缺陷
        if is_cv_ok:
            corner_text = f"[!] {cv_ok_reason}" if cv_ok_reason else "[!] CV OK"
            corner_font = cv2.FONT_HERSHEY_SIMPLEX
            corner_scale = 0.5
            corner_thickness = 1
            (tw, th), _ = cv2.getTextSize(corner_text, corner_font, corner_scale, corner_thickness)
            pad = 5
            bx1 = panel_w - tw - 2 * pad - 6
            by1 = 6
            bx2 = panel_w - 6
            by2 = by1 + th + 2 * pad
            overlay = panel_highlight.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.65, panel_highlight, 0.35, 0, panel_highlight)
            cv2.rectangle(panel_highlight, (bx1, by1), (bx2, by2), (0, 165, 255), 1)
            cv2.putText(panel_highlight, corner_text,
                        (bx1 + pad, by1 + pad + th),
                        corner_font, corner_scale, (0, 165, 255), corner_thickness)

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

            # Panel 3: 在 OMIT 上標記偵測到的異物（藍色半透明覆蓋）
            if dust_mask_omit is not None and is_dust_detected:
                dm_for_omit = dust_mask_omit
                if len(dm_for_omit.shape) == 3:
                    dm_for_omit = cv2.cvtColor(dm_for_omit, cv2.COLOR_BGR2GRAY)
                dm_vis_omit = cv2.resize(dm_for_omit, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
                _blend_color_on_mask(omit_panel, dm_vis_omit, (255, 100, 0))

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

            # 產生 Panel 4 可視化圖 (R=僅缺陷, G=重疊, B=僅異物)
            overlay_panel = omit_panel.copy()
            if dust_mask_omit is not None and is_dust_detected:
                # resize masks to panel size (用 vis_mask 保薄線可見度)
                defect_vis = cv2.resize(vis_mask, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
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

        # threshold_used / min_area_used / min_max_diff_used / cv_ok_reason 已於 Panel 2 前計算
        if is_cv_ok:
            verdict = f"CV OK ({cv_ok_reason})" if cv_ok_reason else "CV OK"
            verdict_color = (0, 255, 0)
        elif is_bomb:
            verdict = f"BOMB: {bomb_code} (Filtered as OK)"
            verdict_color = (255, 0, 255)
        elif is_dust:
            iou_cmp = ">=" if surface_iou >= dust_iou_threshold else "<"
            verdict = f"SURFACE ({metric_name}:{surface_iou:.3f}{iou_cmp}Thr:{dust_iou_threshold:.2f}) (Filtered as OK)"
            verdict_color = (0, 200, 255)
        else:
            # 非灰塵 NG：若有 COV/IOU 數值也顯示與閾值的比較
            if surface_iou > 0 or (omit_image is not None and dust_check_fn is not None):
                iou_cmp = ">=" if surface_iou >= dust_iou_threshold else "<"
                verdict = f"NG ({metric_name}:{surface_iou:.3f}{iou_cmp}Thr:{dust_iou_threshold:.2f})"
            else:
                verdict = "NG"
            verdict_color = (0, 0, 255)

        # 組合 header: 實際數值 + 閾值判定符號 + 結果
        threshold_info = ""
        if threshold_used > 0 or min_area_used > 0:
            diff_cmp = ">=" if max_diff >= threshold_used else "<"
            area_cmp = ">=" if area >= min_area_used else "<"
            threshold_info = f" | Diff:{max_diff:.0f}{diff_cmp}Thr:{threshold_used} Area:{area}{area_cmp}MinArea:{min_area_used}"
        else:
            threshold_info = f" | Diff:{max_diff:.0f} Area:{area}px"
        header_text = f"CV Edge [v4]: {side}{threshold_info} | {verdict}"

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

    def _save_cv_fusion_edge_image(
        self,
        save_dir: Path,
        image_name: str,
        edge_index: int,
        edge_defect: Any,
        full_image: np.ndarray,
        omit_image: np.ndarray = None,
        edge_config: Any = None,
        dust_check_fn: Any = None,
        dust_iou_threshold: float = 0.3,
        dust_metric: str = "coverage",
        panel_polygon: Optional[np.ndarray] = None,
    ) -> str:
        """Phase 7.2-B: Fusion 模式 CV defect 3 板組合圖
        (Panel 1: Detection / Panel 2: OMIT+Dust / Panel 3: Overlap)"""
        bx, by, bw, bh = edge_defect.bbox
        side = edge_defect.side
        max_diff = int(getattr(edge_defect, 'max_diff', 0))
        area = int(edge_defect.area)
        is_dust = bool(getattr(edge_defect, 'is_suspected_dust_or_scratch', False))
        is_bomb = bool(getattr(edge_defect, 'is_bomb', False))
        is_cv_ok = bool(getattr(edge_defect, 'is_cv_ok', False))
        img_h, img_w = full_image.shape[:2]

        padding = 100
        rx1 = max(0, bx - padding); ry1 = max(0, by - padding)
        rx2 = min(img_w, bx + bw + padding); ry2 = min(img_h, by + bh + padding)
        roi_raw = full_image[ry1:ry2, rx1:rx2].copy()
        roi_bgr = ensure_bgr(roi_raw)

        # --- Panel 1: Detection ---
        panel1 = roi_bgr.copy()

        # 1-a: 藍虛線畫 band 輪廓（polygon 邊往內縮 band_px 的等距線）
        band_px = 40
        if edge_config is not None:
            band_px = int(getattr(edge_config, 'aoi_edge_boundary_band_px', 40))
        if panel_polygon is not None and len(panel_polygon) >= 3:
            poly_local = panel_polygon.astype(np.float32).copy()
            poly_local[:, 0] -= rx1; poly_local[:, 1] -= ry1
            # 原 polygon 邊
            cv2.polylines(panel1, [poly_local.astype(np.int32)], isClosed=True,
                          color=(255, 100, 0), thickness=1, lineType=cv2.LINE_AA)
            # 內縮 band_px 虛線（純視覺，不影響判定）
            fg_mask = np.zeros(panel1.shape[:2], dtype=np.uint8)
            cv2.fillPoly(fg_mask, [poly_local.astype(np.int32)], 255)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * band_px + 1, 2 * band_px + 1))
            fg_inner = cv2.erode(fg_mask, kernel)
            band_contour = cv2.subtract(fg_mask, fg_inner)
            dash_pattern = np.zeros_like(band_contour)
            dash_pattern[::4, :] = 255  # 每 4 行取 1 行製造虛線
            band_dash = cv2.bitwise_and(band_contour, dash_pattern)
            panel1[band_dash > 0] = (255, 100, 0)

        # 1-b: 紅色 defect 像素 (cv_filtered_mask)
        cv_mask = getattr(edge_defect, 'cv_filtered_mask', None)
        cv_mask_offset = getattr(edge_defect, 'cv_mask_offset', None)
        if cv_mask is not None and cv_mask_offset is not None:
            mo_x, mo_y = cv_mask_offset
            paste_x = mo_x - rx1; paste_y = mo_y - ry1
            mh, mw = cv_mask.shape[:2]
            if (0 <= paste_x and 0 <= paste_y
                    and paste_x + mw <= panel1.shape[1] and paste_y + mh <= panel1.shape[0]):
                defect_mask = np.zeros(panel1.shape[:2], dtype=np.uint8)
                defect_mask[paste_y:paste_y + mh, paste_x:paste_x + mw] = cv_mask
                vis_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                vis_mask = cv2.dilate(defect_mask, vis_kernel, iterations=1)
                if is_bomb:
                    highlight = (255, 0, 255)
                elif is_dust:
                    highlight = (0, 200, 255)
                else:
                    highlight = (0, 0, 255)
                _blend_color_on_mask(panel1, vis_mask, highlight, alpha=0.55)

        # --- Panel 2: OMIT + Dust Overlay ---
        # 變數保留給 B4 Panel 3 / B5 header 使用
        dust_mask_omit = None
        omit_sub = None
        is_dust_detected = False
        if omit_image is not None:
            oh, ow = omit_image.shape[:2]
            osx1 = max(0, rx1); osy1 = max(0, ry1)
            osx2 = min(ow, rx2); osy2 = min(oh, ry2)
            omit_sub = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
            if osx2 > osx1 and osy2 > osy1:
                omit_crop = omit_image[osy1:osy2, osx1:osx2]
                dx1 = osx1 - rx1; dy1 = osy1 - ry1
                dx2 = dx1 + (osx2 - osx1); dy2 = dy1 + (osy2 - osy1)
                omit_sub[dy1:dy2, dx1:dx2] = omit_crop if omit_crop.ndim == 2 else (
                    cv2.cvtColor(omit_crop, cv2.COLOR_BGR2GRAY)
                )
            panel2 = ensure_bgr(omit_sub)
            if dust_check_fn is not None:
                try:
                    is_dust_detected, dust_mask_raw, _br, _dt = dust_check_fn(omit_sub)
                    if dust_mask_raw is not None:
                        if dust_mask_raw.ndim == 3:
                            dust_mask_raw = cv2.cvtColor(dust_mask_raw, cv2.COLOR_BGR2GRAY)
                        dust_mask_omit = dust_mask_raw
                        if is_dust_detected:
                            _blend_color_on_mask(panel2, dust_mask_omit,
                                                  (255, 100, 0), alpha=0.5)
                except Exception as e:
                    print(f"⚠️ CV Fusion Panel 2 dust check 失敗: {e}")
        else:
            panel2 = np.full(roi_bgr.shape, 40, dtype=np.uint8)
            cv2.putText(panel2, "No OMIT", (10, panel2.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Panel 3: Overlap (OMIT 底 + 紅 defect + 藍 dust + 紫交集) ---
        if omit_image is not None and omit_sub is not None:
            panel3 = ensure_bgr(omit_sub).copy()
        else:
            panel3 = roi_bgr.copy()

        # 組 defect mask（在 panel3 尺寸）
        defect_mask_p3 = None
        if cv_mask is not None and cv_mask_offset is not None:
            mo_x, mo_y = cv_mask_offset
            paste_x = mo_x - rx1; paste_y = mo_y - ry1
            mh, mw = cv_mask.shape[:2]
            if (0 <= paste_x and 0 <= paste_y
                    and paste_x + mw <= panel3.shape[1] and paste_y + mh <= panel3.shape[0]):
                defect_mask_p3 = np.zeros(panel3.shape[:2], dtype=np.uint8)
                defect_mask_p3[paste_y:paste_y + mh, paste_x:paste_x + mw] = cv_mask
                defect_mask_p3 = cv2.dilate(
                    defect_mask_p3,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                    iterations=1,
                )

        # 畫紅色 defect
        if defect_mask_p3 is not None:
            _blend_color_on_mask(panel3, defect_mask_p3, (0, 0, 255), alpha=0.55)
        # 畫藍色 dust（與 Panel 2 一致：is_dust_detected=True 且 mask 非 None 才畫）
        if dust_mask_omit is not None and is_dust_detected:
            _blend_color_on_mask(panel3, dust_mask_omit, (255, 100, 0), alpha=0.5)
        # 畫紫色交集（純色覆蓋）— 需 defect + is_dust + dust_mask 三者都有
        if (defect_mask_p3 is not None and dust_mask_omit is not None
                and is_dust_detected):
            # 尺寸對齊
            if dust_mask_omit.shape != panel3.shape[:2]:
                dust_resized = cv2.resize(dust_mask_omit, (panel3.shape[1], panel3.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
            else:
                dust_resized = dust_mask_omit
            overlap = cv2.bitwise_and(defect_mask_p3, dust_resized)
            panel3[overlap > 0] = (220, 0, 180)  # 紫

        panel_h = 400
        scale = panel_h / max(panel1.shape[0], 1)
        panel_w = max(int(panel1.shape[1] * scale), 200)
        panel1_resized = cv2.resize(panel1, (panel_w, panel_h))
        panel2_resized = cv2.resize(panel2, (panel_w, panel_h))
        panel3_resized = cv2.resize(panel3, (panel_w, panel_h))
        panels = [panel1_resized, panel2_resized, panel3_resized]
        labels = ["CV Detection (band)", "OMIT + Dust", "Overlap"]

        gap_w = 10
        gap = np.full((panel_h, gap_w, 3), 80, dtype=np.uint8)
        spaced = []
        for i, p in enumerate(panels):
            spaced.append(p)
            if i < len(panels) - 1:
                spaced.append(gap)
        composite = np.hstack(spaced)
        comp_h, comp_w = composite.shape[:2]

        # Verdict 分類
        if is_bomb:
            verdict = "BOMB (filter OK)"; v_color = (255, 0, 255)
        elif is_cv_ok:
            verdict = "OK (CV filtered)"; v_color = (0, 255, 0)
        elif is_dust:
            verdict = "OK (dust)"; v_color = (0, 255, 0)
        else:
            verdict = "NG"; v_color = (0, 0, 255)

        # COV 值（若有跑 dust_check + 有 defect + is_dust_detected）
        cov_text = ""
        if (defect_mask_p3 is not None
                and dust_mask_omit is not None
                and is_dust_detected):
            if dust_mask_omit.shape != defect_mask_p3.shape[:2]:
                dust_cmp = cv2.resize(dust_mask_omit, defect_mask_p3.shape[::-1],
                                       interpolation=cv2.INTER_NEAREST)
            else:
                dust_cmp = dust_mask_omit
            inter = int(np.count_nonzero((defect_mask_p3 > 0) & (dust_cmp > 0)))
            defect_area = max(1, int(np.count_nonzero(defect_mask_p3 > 0)))
            cov_val = inter / defect_area
            metric_label = "COV" if dust_metric == "coverage" else "IOU"
            cov_text = f" | {metric_label}={cov_val:.2f}"

        header_h = 50
        header = np.zeros((header_h, comp_w, 3), dtype=np.uint8)
        info_part = f"CV Edge: {side} | MaxDiff:{max_diff} | Area:{area}px{cov_text} | "
        self._draw_split_color_header(header, info_part, verdict, v_color,
                                       y=30, font_scale=0.65)

        label_h = 40
        label_bar = np.zeros((label_h, comp_w, 3), dtype=np.uint8)
        for i, lbl in enumerate(labels):
            lx = i * (panel_w + gap_w) + 10
            cv2.putText(label_bar, lbl, (lx, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        final = np.vstack([header, composite, label_bar])
        filename = f"edge_cvfusion_{image_name}_{side}_{edge_index}.{self.save_format}"
        filepath = save_dir / filename
        cv2.imwrite(str(filepath), final)
        return str(filepath)

    def _save_patchcore_edge_image(
        self,
        save_dir: Path,
        image_name: str,
        edge_index: int,
        edge_defect: Any,
        full_image: np.ndarray,
        omit_image: np.ndarray = None,
        dust_check_fn: Any = None,
    ) -> str:
        """PatchCore inspector 邊緣缺陷比較圖。

        Panel 1: Original ROI (AOI centered raw crop，不疊 mask / marker)
        Panel 2: Anomaly Heatmap Overlay (半透明 colormap 疊於 ROI 上)
        Panel 3: OMIT ROI (shifted) + dust mask 藍色 overlay (如果 dust_check_fn 提供)

        Args:
            dust_check_fn: 灰塵偵測函數（通常 CAPIInferencer.check_dust_or_scratch_feature），
                           回傳 (is_dust, dust_mask, bright_ratio, detail_text)。
                           None 則跳過 overlay；回傳 is_dust=False 也不 overlay。
        """
        side = edge_defect.side
        center = edge_defect.center
        bbox = edge_defect.bbox
        score = float(getattr(edge_defect, 'patchcore_score', 0.0))
        threshold = float(getattr(edge_defect, 'patchcore_threshold', 0.0))
        area = int(edge_defect.area)
        is_cv_ok = bool(getattr(edge_defect, 'is_cv_ok', False))
        is_bomb = bool(getattr(edge_defect, 'is_bomb', False))
        bomb_code = getattr(edge_defect, 'bomb_defect_code', '')
        ok_reason = getattr(edge_defect, 'patchcore_ok_reason', '')

        roi = edge_defect.pc_roi
        fg_mask = edge_defect.pc_fg_mask
        anomaly_map = edge_defect.pc_anomaly_map

        # Fallback: 推論 artifact 遺失時用 bbox 擷取避免爆錯
        img_h, img_w = full_image.shape[:2]
        if roi is None:
            bx, by, bw, bh = bbox
            rx1 = max(0, bx); ry1 = max(0, by)
            rx2 = min(img_w, bx + bw); ry2 = min(img_h, by + bh)
            roi = full_image[ry1:ry2, rx1:rx2].copy()
            fg_mask = np.ones(roi.shape[:2], dtype=np.uint8) * 255

        roi_bgr = ensure_bgr(roi)
        # Phase 7.2 A1: Panel 1 改 AOI centered raw crop，不疊 mask / marker
        cx, cy = int(center[0]), int(center[1])
        tile_size = roi.shape[0] if roi is not None else 512
        half = tile_size // 2
        img_h, img_w = full_image.shape[:2]
        shape = (tile_size, tile_size, 3) if full_image.ndim == 3 else (tile_size, tile_size)
        raw_canvas = np.zeros(shape, dtype=full_image.dtype)
        sx1 = max(0, cx - half); sy1 = max(0, cy - half)
        sx2 = min(img_w, cx + half); sy2 = min(img_h, cy + half)
        if sx2 > sx1 and sy2 > sy1:
            dx1 = sx1 - (cx - half); dy1 = sy1 - (cy - half)
            dx2 = dx1 + (sx2 - sx1); dy2 = dy1 + (sy2 - sy1)
            raw_canvas[dy1:dy2, dx1:dx2] = full_image[sy1:sy2, sx1:sx2]
        panel_orig = ensure_bgr(raw_canvas)

        panel_heatmap = render_pc_overlay(roi_bgr, fg_mask, anomaly_map)

        panel_h = 400
        scale = panel_h / max(panel_orig.shape[0], 1)
        panel_w = max(int(panel_orig.shape[1] * scale), 200)
        panel_orig = cv2.resize(panel_orig, (panel_w, panel_h))
        panel_heatmap = cv2.resize(panel_heatmap, (panel_w, panel_h))

        # Phase 6: fusion 模式 PC 來源 → 左上角 [PC] 角標
        # Phase 7: 加 shift / fallback 資訊
        edge_inspector_mode = getattr(edge_defect, 'inspector_mode', 'cv')
        edge_source = getattr(edge_defect, 'source_inspector', '')
        if edge_inspector_mode == 'fusion' and edge_source == 'patchcore':
            shift_dx = int(getattr(edge_defect, 'pc_roi_shift_dx', 0))
            shift_dy = int(getattr(edge_defect, 'pc_roi_shift_dy', 0))
            pc_fb = str(getattr(edge_defect, 'pc_roi_fallback_reason', ''))
            if shift_dx or shift_dy:
                badge_text = f"[PC dx={shift_dx:+d} dy={shift_dy:+d}]"
                badge_w = 200
            elif pc_fb == "shift_insufficient":
                badge_text = "[PC FB: shift_insufficient]"
                badge_w = 310
            elif pc_fb == "concave_polygon":
                badge_text = "[PC FB: concave_polygon]"
                badge_w = 290
            elif pc_fb == "shift_disabled":
                badge_text = "[PC FB: shift_disabled]"
                badge_w = 280
            elif pc_fb:
                badge_text = f"[PC FB: {pc_fb}]"
                badge_w = 250
            else:
                badge_text = "[PC]"
                badge_w = 60
            cv2.rectangle(panel_heatmap, (5, 5), (badge_w, 28), (0, 0, 0), -1)
            cv2.putText(panel_heatmap, badge_text, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        panels = [panel_orig, panel_heatmap]
        labels = ["Original ROI (AOI)", "PatchCore Heatmap"]

        if omit_image is not None:
            try:
                # Phase 7.2 A2: 先把 OMIT 標準化成 BGR 再擷取（避免 gray vs BGR broadcast 失敗）
                omit_src = ensure_bgr(omit_image)
                oh, ow = omit_src.shape[:2]
                tile_size = roi.shape[0]
                # Phase 7.2 A2: OMIT 擷取改以 shifted ROI origin 為起點（pc_roi_origin_x/y）
                ox_origin = int(getattr(edge_defect, "pc_roi_origin_x", 0))
                oy_origin = int(getattr(edge_defect, "pc_roi_origin_y", 0))
                # 舊 record 無 origin（預設 0,0）→ fallback 用 AOI center
                if ox_origin == 0 and oy_origin == 0:
                    cx, cy = int(center[0]), int(center[1])
                    ox_origin = cx - tile_size // 2
                    oy_origin = cy - tile_size // 2
                omit_canvas = np.zeros((tile_size, tile_size, 3), dtype=omit_src.dtype)
                osx1 = max(0, ox_origin); osy1 = max(0, oy_origin)
                osx2 = min(ow, ox_origin + tile_size); osy2 = min(oh, oy_origin + tile_size)
                if osx2 > osx1 and osy2 > osy1:
                    odx1 = osx1 - ox_origin; ody1 = osy1 - oy_origin
                    odx2 = odx1 + (osx2 - osx1); ody2 = ody1 + (osy2 - osy1)
                    omit_canvas[ody1:ody2, odx1:odx2] = omit_src[osy1:osy2, osx1:osx2]

                # Phase 7.2 A3: 套 dust_check_fn 的藍色 overlay（BGR 255,100,0）
                # 與 CV 路徑（capi_heatmap.py:800）行為一致：直接傳 omit_canvas，由 check_fn 自行處理通道
                dust_mask_panel = None
                is_dust_detected = False
                if dust_check_fn is not None:
                    try:
                        is_dust_detected, dust_mask_raw, _br, _dt = dust_check_fn(omit_canvas)
                        if dust_mask_raw is not None:
                            if dust_mask_raw.ndim == 3:
                                dust_mask_raw = cv2.cvtColor(dust_mask_raw, cv2.COLOR_BGR2GRAY)
                            dust_mask_panel = dust_mask_raw
                    except Exception as e:
                        print(f"⚠️ PC Panel 3 dust check 失敗: {e}")

                omit_panel_bgr = ensure_bgr(omit_canvas)
                # 與 CV 路徑 (capi_heatmap.py:806) 一致：要 dust_mask 有值且 is_dust_detected=True 才 overlay
                if dust_mask_panel is not None and is_dust_detected:
                    _blend_color_on_mask(omit_panel_bgr, dust_mask_panel,
                                         (255, 100, 0), alpha=0.5)
                omit_panel = cv2.resize(omit_panel_bgr, (panel_w, panel_h))
                panels.append(omit_panel)
                labels.append("OMIT ROI (shifted)" if (
                    edge_defect.pc_roi_shift_dx or edge_defect.pc_roi_shift_dy
                ) else "OMIT ROI")
            except Exception as e:
                print(f"⚠️ PatchCore edge OMIT panel 失敗: {e}")

        # 拼接
        gap_w = 10
        gap = np.full((panel_h, gap_w, 3), 80, dtype=np.uint8)
        spaced = []
        for i, p in enumerate(panels):
            spaced.append(p)
            if i < len(panels) - 1:
                spaced.append(gap)
        composite = np.hstack(spaced)
        comp_h, comp_w = composite.shape[:2]

        # Header
        header_h = 50
        header = np.zeros((header_h, comp_w, 3), dtype=np.uint8)

        if is_bomb:
            verdict = f"BOMB: {bomb_code} (Filtered as OK)"
            verdict_color = (255, 0, 255)
        elif is_cv_ok:
            reason_txt = ok_reason if ok_reason else ""
            verdict = f"PC OK ({reason_txt})" if reason_txt else "PC OK"
            verdict_color = (0, 255, 0)
        else:
            verdict = "NG"
            verdict_color = (0, 0, 255)

        score_cmp = ">=" if score >= threshold else "<"
        info_part = f"PC Edge [v1]: {side} | Score:{score:.3f}{score_cmp}Thr:{threshold:.3f} | Area:{area}px | "
        self._draw_split_color_header(header, info_part, verdict, verdict_color, y=30, font_scale=0.65)

        # Phase 7.2 A4: Header 右側加 shift / fallback 資訊
        shift_dx = int(getattr(edge_defect, 'pc_roi_shift_dx', 0))
        shift_dy = int(getattr(edge_defect, 'pc_roi_shift_dy', 0))
        pc_fb = str(getattr(edge_defect, 'pc_roi_fallback_reason', ''))
        extra_text = ""
        if shift_dx or shift_dy:
            extra_text = f"PC dx={shift_dx:+d} dy={shift_dy:+d}"
        elif pc_fb == "shift_insufficient":
            extra_text = "PC-FB=shift_insufficient(offset short)"
        elif pc_fb == "concave_polygon":
            extra_text = "PC-FB=concave_polygon(concave)"
        elif pc_fb == "shift_disabled":
            extra_text = "PC-FB=shift_disabled"
        elif pc_fb:
            extra_text = f"PC-FB={pc_fb}"
        if extra_text:
            (et_w, _), _ = cv2.getTextSize(extra_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(header, extra_text, (comp_w - et_w - 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)

        # Label bar
        label_h = 40
        label_bar = np.zeros((label_h, comp_w, 3), dtype=np.uint8)
        for i, lbl in enumerate(labels):
            lx = i * (panel_w + gap_w) + 10
            cv2.putText(label_bar, lbl, (lx, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        final = np.vstack([header, composite, label_bar])

        filename = f"edge_pc_{image_name}_{side}_{edge_index}.{self.save_format}"
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
