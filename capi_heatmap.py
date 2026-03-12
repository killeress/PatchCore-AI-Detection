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
from typing import List, Dict, Optional, Tuple, Any
import time


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

        # --- Panel 1: Original Tile ---
        orig = tile_image.copy()
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

        # --- 底部獨立標籤列（不蓋到面板內容）---
        if has_omit:
            labels = ["Original", "Heatmap", "OMIT Crop", f"Dust Mask ({metric_name}:{dust_iou:.3f})", f"{metric_name} Debug (G=Overlap R=Heat B=Dust)"]
            panels = [orig, heatmap_panel, omit_panel, dust_panel, iou_debug_panel]
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

        if is_bomb:
            verdict = f"BOMB: {bomb_code} (Filtered as OK)"
            verdict_color = (255, 0, 255)  # 洋紅色
        elif is_dust:
            verdict = "DUST (Filtered as OK)"
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

        Returns:
            {"dir": save_dir, "files": [...]}
        """
        save_dir = self.get_save_dir(glass_id, date_str)
        saved_files = []

        for result in results:
            image_name = result.image_path.stem

            # 儲存全圖總覽
            if save_overview and result.anomaly_tiles:
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
                                score_threshold=inferencer.threshold,
                                iou_threshold=inferencer.config.dust_heatmap_iou_threshold,
                            )
                            saved_files.append(path)
                        except Exception as e:
                            print(f"⚠️ 儲存 tile heatmap 失敗 ({image_name} tile{tile.tile_id}): {e}")

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
