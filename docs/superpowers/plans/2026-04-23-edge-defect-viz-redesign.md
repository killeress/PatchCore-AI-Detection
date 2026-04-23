# Edge Defect 組合圖視覺化重構 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重構 `capi_heatmap.py` 的 edge defect 組合圖：PC 組 3 板改 Panel 1 raw + Panel 3 shifted OMIT + dust overlay；CV fusion 組新增 3 板（Detection / OMIT+Dust / Overlap）；header verdict 大字 + 翻 OK 原因。

**Architecture:** 只動視覺化層（`capi_heatmap.py`）。兩個 renderer：`_save_patchcore_edge_image`（既有，改版）與 `_save_cv_fusion_edge_image`（新）。四邊 CV (`side in {left/right/top/bottom}`) 維持 4 板 `save_edge_defect_image` 主分支。Phase A / B / C 依序獨立 deploy。

**Tech Stack:** OpenCV (drawing/blending)、NumPy、pytest + unittest.mock

**Spec:** `docs/superpowers/specs/2026-04-23-edge-defect-viz-redesign-design.md`

---

## File Structure

| File | Responsibility | 改動類型 |
|---|---|---|
| `capi_heatmap.py::_save_patchcore_edge_image` | PC 組 3 板渲染 | 改 |
| `capi_heatmap.py::_save_cv_fusion_edge_image` | CV fusion 3 板渲染（新） | 建 |
| `capi_heatmap.py::save_edge_defect_image` 分派 | fusion+cv 改走新 renderer | 改 |
| `capi_heatmap.py::_render_edge_header`（新） | Header 統一 helper（Phase C） | 建 |
| `tests/test_edge_viz_pc_panel.py` | Phase A 單測 | 建 |
| `tests/test_edge_viz_cv_fusion_panel.py` | Phase B 單測 | 建 |
| `tests/test_edge_viz_header.py` | Phase C helper 單測 | 建 |

---

## Phase A — PC 組 3 板 refactor

修改 `_save_patchcore_edge_image`（`capi_heatmap.py` 約 line 972–1122）。

### Task A1: Panel 1 改 raw AOI-centered crop（移除 mask + marker）

**Files:**
- Create: `tests/test_edge_viz_pc_panel.py`
- Modify: `capi_heatmap.py` Panel 1 建構段（line 1011-1015）

- [ ] **Step 1: Write the failing test**

```python
# tests/test_edge_viz_pc_panel.py (新建檔)
"""Phase 7.2 Phase A — PC 組 3 板視覺化單元測試"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from capi_edge_cv import EdgeDefect
from capi_heatmap import HeatmapSaver


def _make_edge_defect(
    center=(1000, 500),
    pc_roi=None,
    pc_fg_mask=None,
    pc_anomaly_map=None,
    shift=(0, 0),
    pc_roi_origin=None,
    fallback_reason="",
    source="patchcore",
    inspector_mode="fusion",
):
    """建立測試用 EdgeDefect（PC fusion 情境）"""
    roi_size = 512
    if pc_roi is None:
        pc_roi = np.full((roi_size, roi_size, 3), 128, dtype=np.uint8)
    if pc_fg_mask is None:
        pc_fg_mask = np.full((roi_size, roi_size), 255, dtype=np.uint8)
    if pc_anomaly_map is None:
        pc_anomaly_map = np.zeros((roi_size, roi_size), dtype=np.float32)
    if pc_roi_origin is None:
        pc_roi_origin = (center[0] - roi_size // 2 + shift[0],
                         center[1] - roi_size // 2 + shift[1])

    ed = EdgeDefect(
        side="aoi_edge", area=100,
        bbox=(pc_roi_origin[0], pc_roi_origin[1], roi_size, roi_size),
        center=center, max_diff=0,
    )
    ed.source_inspector = source
    ed.inspector_mode = inspector_mode
    ed.patchcore_score = 0.8
    ed.patchcore_threshold = 0.5
    ed.pc_roi = pc_roi
    ed.pc_fg_mask = pc_fg_mask
    ed.pc_anomaly_map = pc_anomaly_map
    ed.pc_roi_origin_x = pc_roi_origin[0]
    ed.pc_roi_origin_y = pc_roi_origin[1]
    ed.pc_roi_shift_dx = shift[0]
    ed.pc_roi_shift_dy = shift[1]
    ed.pc_roi_fallback_reason = fallback_reason
    return ed


class TestPCGroupPanel1:
    """Panel 1: AOI centered raw ROI，不加 mask 不加 marker"""

    def test_panel1_is_raw_aoi_centered_crop(self, tmp_path):
        """Panel 1 像素 = full_image AOI 中心 512×512 區塊（不被 fg_mask 蓋紅）"""
        # full_image 建一個有明確紋理的 2000×2000 灰階 (轉 BGR)
        full = np.zeros((2000, 2000, 3), dtype=np.uint8)
        full[400:600, 900:1100] = 200  # AOI 區塊一塊亮色方塊

        ed = _make_edge_defect(center=(1000, 500),
                                shift=(0, -192))  # shifted up
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        path = saver._save_patchcore_edge_image(
            save_dir=tmp_path, image_name="test", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=None,
        )
        out = cv2.imread(path)
        assert out is not None, "output image should exist"

        # Panel 1 位於 header 下方左側第一塊 (header_h + ~0, 0..panel_w)
        # 簡單策略：取左上角 Panel 1 中心點像素，應該接近原圖 AOI 中心（bright 200）
        # 不能精確斷言尺寸，但可以斷言：左側第一塊 Panel 的色調偏亮（raw），不是暗紅 mask 色 (0,0,60)
        h, w = out.shape[:2]
        # Header 50px + Panel_h 400px + label 40px → 總高約 490px
        # Panel 1 水平位於 0..panel_w（約 1/3 寬度）
        panel_w_approx = w // 3
        panel1_center_y = 50 + 400 // 2
        panel1_center_x = panel_w_approx // 2
        px = out[panel1_center_y, panel1_center_x]
        # 不能是 (0,0,60) 或 (0,0,120) 暗紅 mask
        assert not (px[0] < 30 and px[1] < 30 and px[2] > 40 and px[2] < 150), \
            f"Panel 1 中心像素 {px.tolist()} 看起來是暗紅 mask，應為 raw"

    def test_panel1_has_no_yellow_marker(self, tmp_path):
        """Panel 1 中央不應有黃色圓圈 (0, 255, 255)"""
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        ed = _make_edge_defect(center=(1000, 500))
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        path = saver._save_patchcore_edge_image(
            save_dir=tmp_path, image_name="test", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=None,
        )
        out = cv2.imread(path)
        h, w = out.shape[:2]
        panel_w_approx = w // 3
        # 在 Panel 1 中心 40×40 區塊掃描是否有黃色像素
        cy, cx = 50 + 200, panel_w_approx // 2
        region = out[cy - 20:cy + 20, cx - 20:cx + 20]
        # 黃色：B<=50, G>=180, R>=180
        yellow_mask = (region[:, :, 0] <= 50) & (region[:, :, 1] >= 180) & (region[:, :, 2] >= 180)
        assert not np.any(yellow_mask), "Panel 1 中心 40×40 區域不應出現黃色 marker 像素"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_edge_viz_pc_panel.py::TestPCGroupPanel1 -v`
Expected: FAIL — `test_panel1_has_no_yellow_marker` 因為現行實作會畫黃色 circle

- [ ] **Step 3: Implement — 把 Panel 1 改成 raw crop，移除 mask + marker**

在 `capi_heatmap.py` 的 `_save_patchcore_edge_image` 中：

找到：
```python
        roi_bgr = ensure_bgr(roi)
        panel_orig = render_pc_masked_roi(roi_bgr, fg_mask)
        h_roi, w_roi = panel_orig.shape[:2]
        cv2.circle(panel_orig, (w_roi // 2, h_roi // 2), 15, (0, 255, 255),
                   thickness=2, lineType=cv2.LINE_AA)
```

替換成：
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_edge_viz_pc_panel.py::TestPCGroupPanel1 -v`
Expected: PASS — 2 個測試綠

- [ ] **Step 5: Commit**

```bash
git add tests/test_edge_viz_pc_panel.py capi_heatmap.py
git commit -m "feat(edge-viz): Phase 7.2-A1 — PC Panel 1 改 AOI centered raw crop (移除 mask/marker)"
```

---

### Task A2: Panel 3 OMIT 位置對齊 shifted origin

**Files:**
- Modify: `tests/test_edge_viz_pc_panel.py` (加新 class)
- Modify: `capi_heatmap.py` Panel 3 OMIT 擷取段（約 line 1058-1076）

- [ ] **Step 1: Write the failing test**

在 `tests/test_edge_viz_pc_panel.py` 加：

```python
class TestPCGroupPanel3OMITAlign:
    """Panel 3: OMIT 擷取位置 = shifted ROI origin，不是 AOI center"""

    def test_omit_crop_uses_shifted_origin(self, tmp_path):
        """shift=(0,-192) 時 OMIT 擷取應從 pc_roi_origin_y 開始（比 AOI center 上移 192 px）"""
        # OMIT 影像：畫兩條水平分帶，一帶在 y=[300, 400] 為 255 亮、其餘 0 暗
        omit = np.zeros((2000, 2000), dtype=np.uint8)
        omit[300:400, :] = 255

        # 若 crop 以 AOI center (cy=500) 為中心 → crop y=[244, 756]，亮帶 y=[300,400] 在 crop 頂部
        # 若 crop 以 shifted origin (pc_roi_origin_y = 500 - 256 + (-192) = 52) 為起點 → crop y=[52, 564]
        #   亮帶 y=[300,400] 在 crop 中段（比率 ~0.5）
        ed = _make_edge_defect(center=(1000, 500), shift=(0, -192))
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        path = saver._save_patchcore_edge_image(
            save_dir=tmp_path, image_name="test_omit", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
        )
        out = cv2.imread(path)
        h, w = out.shape[:2]

        # Panel 3 位於最右側第三塊（Panel 1 | gap | Panel 2 | gap | Panel 3）
        # Panel width 大約 w/3，取 Panel 3 中心區塊向下分析
        panel_w_approx = w // 3
        panel3_left = panel_w_approx * 2 + 20  # 加 gap 餘裕
        panel3_right = w - 5

        # 取 Panel 3 垂直中間帶的亮度分佈（y from header_h 到 header_h+panel_h）
        header_h = 50
        panel_h = 400
        col_mid = (panel3_left + panel3_right) // 2
        vertical_profile = cv2.cvtColor(
            out[header_h:header_h + panel_h, col_mid:col_mid + 5], cv2.COLOR_BGR2GRAY
        ).mean(axis=1)

        # 亮峰所在 y 比率
        peak_y_ratio = int(np.argmax(vertical_profile)) / panel_h
        # shifted crop: 亮帶 y=300-400 在 crop y=[52,564] 的 (248,348)/(564-52)=(248,348)/512 ≈ 0.484~0.680 中段
        assert 0.35 <= peak_y_ratio <= 0.75, \
            f"shifted crop 亮帶應該在 Panel 3 中段 (0.35~0.75)，實 ratio={peak_y_ratio}"

    def test_omit_crop_no_shift_matches_centered(self, tmp_path):
        """shift=(0,0) 時 OMIT crop 仍以 centered 位置擷取（等同 pc_roi_origin）"""
        omit = np.zeros((2000, 2000), dtype=np.uint8)
        omit[244:500, :] = 255  # AOI center crop 上半部亮

        ed = _make_edge_defect(center=(1000, 500), shift=(0, 0))
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        path = saver._save_patchcore_edge_image(
            save_dir=tmp_path, image_name="test_omit_nc", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
        )
        out = cv2.imread(path)
        h, w = out.shape[:2]
        panel_w_approx = w // 3
        panel3_left = panel_w_approx * 2 + 20
        col_mid = (panel3_left + (w - 5)) // 2
        header_h = 50; panel_h = 400
        vp = cv2.cvtColor(
            out[header_h:header_h + panel_h, col_mid:col_mid + 5], cv2.COLOR_BGR2GRAY
        ).mean(axis=1)
        peak_y_ratio = int(np.argmax(vp)) / panel_h
        # 亮帶 y=[244, 500] 在 crop y=[244, 756] 映射到 ratio [0, 0.5]
        assert peak_y_ratio <= 0.5, \
            f"無 shift 時亮帶應該在 Panel 3 上半 (<=0.5)，實 ratio={peak_y_ratio}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_edge_viz_pc_panel.py::TestPCGroupPanel3OMITAlign -v`
Expected: FAIL — `test_omit_crop_uses_shifted_origin` 因為現行以 AOI center 擷取

- [ ] **Step 3: Implement — OMIT 擷取改用 shifted origin**

在 `_save_patchcore_edge_image` 找到 Panel 3 OMIT 建構段：

```python
        if omit_image is not None:
            try:
                oh, ow = omit_image.shape[:2]
                tile_size = roi.shape[0]
                cx, cy = int(center[0]), int(center[1])
                half = tile_size // 2
                shape = (tile_size, tile_size, 3) if roi.ndim == 3 else (tile_size, tile_size)
                omit_canvas = np.zeros(shape, dtype=roi.dtype)
                osx1 = max(0, cx - half); osy1 = max(0, cy - half)
                osx2 = min(ow, cx + half); osy2 = min(oh, cy + half)
                if osx2 > osx1 and osy2 > osy1:
                    odx1 = osx1 - (cx - half); ody1 = osy1 - (cy - half)
                    odx2 = odx1 + (osx2 - osx1); ody2 = ody1 + (osy2 - osy1)
                    omit_canvas[ody1:ody2, odx1:odx2] = omit_image[osy1:osy2, osx1:osx2]
                omit_panel = cv2.resize(ensure_bgr(omit_canvas), (panel_w, panel_h))
                panels.append(omit_panel)
                labels.append("OMIT ROI")
            except Exception as e:
                print(f"⚠️ PatchCore edge OMIT panel 失敗: {e}")
```

替換成：

```python
        if omit_image is not None:
            try:
                oh, ow = omit_image.shape[:2]
                tile_size = roi.shape[0]
                # Phase 7.2 A2: OMIT 擷取改以 shifted ROI origin 為起點（pc_roi_origin_x/y）
                ox_origin = int(getattr(edge_defect, "pc_roi_origin_x", 0))
                oy_origin = int(getattr(edge_defect, "pc_roi_origin_y", 0))
                # 舊 record 無 origin（預設 0,0）→ fallback 用 AOI center
                if ox_origin == 0 and oy_origin == 0:
                    cx, cy = int(center[0]), int(center[1])
                    ox_origin = cx - tile_size // 2
                    oy_origin = cy - tile_size // 2
                shape = (tile_size, tile_size, 3) if roi.ndim == 3 else (tile_size, tile_size)
                omit_canvas = np.zeros(shape, dtype=roi.dtype)
                osx1 = max(0, ox_origin); osy1 = max(0, oy_origin)
                osx2 = min(ow, ox_origin + tile_size); osy2 = min(oh, oy_origin + tile_size)
                if osx2 > osx1 and osy2 > osy1:
                    odx1 = osx1 - ox_origin; ody1 = osy1 - oy_origin
                    odx2 = odx1 + (osx2 - osx1); ody2 = ody1 + (osy2 - osy1)
                    omit_canvas[ody1:ody2, odx1:odx2] = omit_image[osy1:osy2, osx1:osx2]
                omit_panel = cv2.resize(ensure_bgr(omit_canvas), (panel_w, panel_h))
                panels.append(omit_panel)
                labels.append("OMIT ROI (shifted)" if (
                    edge_defect.pc_roi_shift_dx or edge_defect.pc_roi_shift_dy
                ) else "OMIT ROI")
            except Exception as e:
                print(f"⚠️ PatchCore edge OMIT panel 失敗: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_edge_viz_pc_panel.py::TestPCGroupPanel3OMITAlign -v`
Expected: PASS — 2 個測試綠

- [ ] **Step 5: Commit**

```bash
git add tests/test_edge_viz_pc_panel.py capi_heatmap.py
git commit -m "feat(edge-viz): Phase 7.2-A2 — PC Panel 3 OMIT 對齊 shifted ROI origin"
```

---

### Task A3: Panel 3 加 dust mask 藍色 overlay

**Files:**
- Modify: `tests/test_edge_viz_pc_panel.py`
- Modify: `capi_heatmap.py::save_edge_defect_image` 把 `dust_check_fn` 傳到 `_save_patchcore_edge_image`；`_save_patchcore_edge_image` 簽章加參數

- [ ] **Step 1: Write the failing test**

加到 `tests/test_edge_viz_pc_panel.py`：

```python
class TestPCGroupPanel3DustOverlay:
    """Panel 3: 套 dust_check_fn 回傳的藍色 overlay (BGR 255,100,0)"""

    def test_dust_overlay_applied_when_fn_returns_mask(self, tmp_path):
        """dust_check_fn 回傳 dust_mask 時，Panel 3 中 dust 區塊偏藍色 (B 高 R 低)"""
        omit = np.full((2000, 2000), 128, dtype=np.uint8)

        # 建一個 stub dust_check_fn：永遠偵到 OMIT 中間 100×100 區塊為 dust
        def stub_dust_check(img):
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[h // 2 - 50:h // 2 + 50, w // 2 - 50:w // 2 + 50] = 255
            return True, mask, 0.1, "stub dust"

        ed = _make_edge_defect(center=(1000, 500), shift=(0, 0))
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        path = saver._save_patchcore_edge_image(
            save_dir=tmp_path, image_name="test_dust", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
            dust_check_fn=stub_dust_check,
        )
        out = cv2.imread(path)
        h_out, w_out = out.shape[:2]
        panel_w_approx = w_out // 3
        panel3_left = panel_w_approx * 2 + 20
        header_h = 50; panel_h = 400
        # Panel 3 中心
        center_x = (panel3_left + w_out - 5) // 2
        center_y = header_h + panel_h // 2
        px = out[center_y, center_x]
        # 藍色 overlay 後 B > R 明顯
        assert px[0] > px[2] + 30, \
            f"dust 中心像素應偏藍 (B>R+30)，實為 B={px[0]} G={px[1]} R={px[2]}"

    def test_no_dust_overlay_when_fn_is_none(self, tmp_path):
        """dust_check_fn 為 None 時 Panel 3 保持純 OMIT (灰)，無藍色覆蓋"""
        omit = np.full((2000, 2000), 128, dtype=np.uint8)
        ed = _make_edge_defect(center=(1000, 500))
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        path = saver._save_patchcore_edge_image(
            save_dir=tmp_path, image_name="test_nodust", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
            dust_check_fn=None,
        )
        out = cv2.imread(path)
        h_out, w_out = out.shape[:2]
        panel3_left = (w_out // 3) * 2 + 20
        center_x = (panel3_left + w_out - 5) // 2
        center_y = 50 + 200
        px = out[center_y, center_x]
        # 純灰，B/G/R 差不多
        assert abs(int(px[0]) - int(px[2])) < 20, \
            f"無 dust_check_fn 時 Panel 3 應為灰 (B≈R)，實為 B={px[0]} R={px[2]}"

    def test_dust_check_exception_does_not_crash(self, tmp_path):
        """dust_check_fn 丟 exception 時 Panel 3 仍能產圖（不爆，無 overlay）"""
        omit = np.full((2000, 2000), 128, dtype=np.uint8)
        def failing_fn(img):
            raise RuntimeError("dust check failed")
        ed = _make_edge_defect(center=(1000, 500))
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        path = saver._save_patchcore_edge_image(
            save_dir=tmp_path, image_name="test_except", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
            dust_check_fn=failing_fn,
        )
        assert Path(path).exists(), "即使 dust_check_fn 丟 exception 也要能產出圖"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_edge_viz_pc_panel.py::TestPCGroupPanel3DustOverlay -v`
Expected: FAIL — `_save_patchcore_edge_image` 不接受 `dust_check_fn` 參數 + 沒有藍色 overlay

- [ ] **Step 3: Implement — 簽章加 `dust_check_fn`, Panel 3 加 overlay**

先改 `_save_patchcore_edge_image` 簽章（約 line 972）：

```python
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
```

再把 Panel 3 OMIT 建構段（Task A2 改完的那段）進階加 dust overlay。在 `omit_canvas[ody1:ody2, odx1:odx2] = omit_image[...]` 後、`omit_panel = cv2.resize(...)` 前加：

```python
                # Phase 7.2 A3: 套 dust_check_fn 的藍色 overlay
                dust_mask_panel = None
                if dust_check_fn is not None:
                    try:
                        _is_dust, dust_mask_raw, _br, _dt = dust_check_fn(omit_canvas)
                        if dust_mask_raw is not None:
                            if dust_mask_raw.ndim == 3:
                                dust_mask_raw = cv2.cvtColor(dust_mask_raw, cv2.COLOR_BGR2GRAY)
                            dust_mask_panel = dust_mask_raw
                    except Exception as e:
                        print(f"⚠️ PC Panel 3 dust check 失敗: {e}")

                omit_panel_bgr = ensure_bgr(omit_canvas)
                if dust_mask_panel is not None:
                    _blend_color_on_mask(omit_panel_bgr, dust_mask_panel,
                                          (255, 100, 0), alpha=0.5)
                omit_panel = cv2.resize(omit_panel_bgr, (panel_w, panel_h))
```

（取代原本 `omit_panel = cv2.resize(ensure_bgr(omit_canvas), (panel_w, panel_h))` 這一行）

最後在 `save_edge_defect_image`（約 line 608-614 分派處）把 `dust_check_fn` 傳進去：

```python
        if inspector_mode == 'patchcore' or (
            inspector_mode == 'fusion' and source_inspector == 'patchcore'
        ):
            return self._save_patchcore_edge_image(
                save_dir, image_name, edge_index, edge_defect,
                full_image, omit_image,
                dust_check_fn=dust_check_fn,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_edge_viz_pc_panel.py::TestPCGroupPanel3DustOverlay -v`
Expected: PASS — 3 個測試綠

- [ ] **Step 5: Commit**

```bash
git add tests/test_edge_viz_pc_panel.py capi_heatmap.py
git commit -m "feat(edge-viz): Phase 7.2-A3 — PC Panel 3 加 dust_check_fn 藍色 overlay"
```

---

### Task A4: PC Header 加 shift / fallback info

**Files:**
- Modify: `tests/test_edge_viz_pc_panel.py`
- Modify: `capi_heatmap.py::_save_patchcore_edge_image` header 建構段（約 line 1089-1106）

- [ ] **Step 1: Write the failing test**

加到 `tests/test_edge_viz_pc_panel.py`：

```python
class TestPCHeaderShiftInfo:
    """PC header 加 shift / fallback info 欄位"""

    def _extract_header_text_check(self, out_img, needle: str) -> bool:
        """暴力 OCR 替代：直接用 cv2.putText 寫過的字會影響 header 區，
        但單測不做 OCR，改用『渲染時 call 紀錄』驗證。
        因此這些測試用 mock patch cv2.putText 攔截文字內容。
        """
        raise NotImplementedError("測試改用 mock patch cv2.putText 攔截")

    def test_header_contains_shift_info(self, tmp_path, monkeypatch):
        """有 shift 時 header 文字應含 'PC dx=' 字串"""
        captured = []
        _real_put = cv2.putText
        def _capture(img, text, *args, **kwargs):
            captured.append(text)
            return _real_put(img, text, *args, **kwargs)
        monkeypatch.setattr("capi_heatmap.cv2.putText", _capture)

        ed = _make_edge_defect(center=(1000, 500), shift=(0, -192))
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        saver._save_patchcore_edge_image(
            save_dir=tmp_path, image_name="h1", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=None,
        )
        all_text = " ".join(captured)
        assert "PC dx=" in all_text or "dy=-192" in all_text, \
            f"header 應含 PC dx/dy 資訊，實際 putText 字串: {captured}"

    def test_header_contains_fallback_reason(self, tmp_path, monkeypatch):
        """fallback=shift_insufficient 時 header 應含 'PC-FB=shift_insufficient'"""
        captured = []
        _real_put = cv2.putText
        def _capture(img, text, *args, **kwargs):
            captured.append(text)
            return _real_put(img, text, *args, **kwargs)
        monkeypatch.setattr("capi_heatmap.cv2.putText", _capture)

        ed = _make_edge_defect(center=(1000, 500), shift=(0, 0),
                                fallback_reason="shift_insufficient")
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        saver._save_patchcore_edge_image(
            save_dir=tmp_path, image_name="h2", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=None,
        )
        all_text = " ".join(captured)
        assert "PC-FB=shift_insufficient" in all_text, \
            f"header 應含 PC-FB=shift_insufficient，實際 putText: {captured}"

    def test_header_no_shift_info_when_none(self, tmp_path, monkeypatch):
        """無 shift 且無 fallback 時 header 不含 PC dx= 或 PC-FB= 字串"""
        captured = []
        _real_put = cv2.putText
        def _capture(img, text, *args, **kwargs):
            captured.append(text)
            return _real_put(img, text, *args, **kwargs)
        monkeypatch.setattr("capi_heatmap.cv2.putText", _capture)

        ed = _make_edge_defect(center=(1000, 500), shift=(0, 0),
                                fallback_reason="")
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        saver._save_patchcore_edge_image(
            save_dir=tmp_path, image_name="h3", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=None,
        )
        all_text = " ".join(captured)
        assert "PC dx=" not in all_text and "PC-FB=" not in all_text, \
            f"無 shift/fallback 時 header 不應出現 dx/FB，實 putText: {captured}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_edge_viz_pc_panel.py::TestPCHeaderShiftInfo -v`
Expected: FAIL — header 現行無 shift info

- [ ] **Step 3: Implement — Header 加 shift / fallback info**

在 `_save_patchcore_edge_image` 找到 header 建構段（約 line 1104-1106）：

```python
        score_cmp = ">=" if score >= threshold else "<"
        info_part = f"PC Edge [v1]: {side} | Score:{score:.3f}{score_cmp}Thr:{threshold:.3f} | Area:{area}px | "
        self._draw_split_color_header(header, info_part, verdict, verdict_color, y=30, font_scale=0.65)
```

替換成：

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_edge_viz_pc_panel.py::TestPCHeaderShiftInfo -v`
Expected: PASS — 3 個測試綠

- [ ] **Step 5: Commit**

```bash
git add tests/test_edge_viz_pc_panel.py capi_heatmap.py
git commit -m "feat(edge-viz): Phase 7.2-A4 — PC header 加 shift / fallback info"
```

---

### Task A5: Phase A 全套 regression + 視覺抽檢

**Files:** none（純跑測試）

- [ ] **Step 1: 跑 Phase A 全部新測試**

Run: `python -m pytest tests/test_edge_viz_pc_panel.py -v`
Expected: 所有 PC 組測試（~8 個）綠

- [ ] **Step 2: 跑 Edge / Fusion 迴歸**

Run: `python -m pytest tests/test_aoi_edge_fusion.py tests/test_aoi_edge_pc_roi_shift.py -v`
Expected: 44 passed

- [ ] **Step 3: 跑全套測試**

Run: `python -m pytest tests/`
Expected: >= 177 passed, 2 skipped（與現況相同或更多）

- [ ] **Step 4: 更新 tuning log 與 commit 壓軸**

在 `docs/edge-cv-tuning-log.md` Phase 7.1c 下面加一段（簡短記錄）：

```markdown
### Phase 7.2-A — PC 組組合圖重構 (2026-04-XX)

- Panel 1: 改 AOI centered raw crop，移除 fg_mask 紅斜線與黃色 marker（現場回饋遮蓋 defect）
- Panel 3: OMIT 擷取改以 `pc_roi_origin_x/y` 為起點，對齊 shifted ROI 位置；無 shift 則 fallback centered
- Panel 3: 加 `dust_check_fn` 回傳的藍色 dust_mask overlay（BGR 255,100,0, alpha 0.5）
- Header: 右側加 `PC dx=+0 dy=-192` 或 `PC-FB=shift_insufficient` 資訊
- 測試：`tests/test_edge_viz_pc_panel.py` 新增 8 項；全套 177+ green
```

```bash
git add docs/edge-cv-tuning-log.md
git commit -m "docs: Phase 7.2-A tuning log — PC 組組合圖 3 板重構紀錄"
```

---

## Phase B — CV fusion 組 3 板 refactor

修改 `capi_heatmap.py`：新增 `_save_cv_fusion_edge_image`，並在 `save_edge_defect_image` 加分派。四邊 CV 繼續走現有 4 板主分支。

### Task B1: 新增分派到 `_save_cv_fusion_edge_image`（空殼回傳占位圖）

**Files:**
- Create: `tests/test_edge_viz_cv_fusion_panel.py`
- Modify: `capi_heatmap.py::save_edge_defect_image` 分派
- Modify: `capi_heatmap.py` 加 `_save_cv_fusion_edge_image` 空殼

- [ ] **Step 1: Write the failing test**

```python
# tests/test_edge_viz_cv_fusion_panel.py (新建)
"""Phase 7.2 Phase B — CV fusion 組 3 板視覺化單元測試"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from capi_edge_cv import EdgeDefect
from capi_heatmap import HeatmapSaver


def _make_cv_fusion_defect(
    bbox=(950, 450, 40, 40),
    center=(970, 470),
    cv_filtered_mask=None,
    cv_mask_offset=(850, 350),
    is_dust=False,
    is_bomb=False,
):
    ed = EdgeDefect(
        side="aoi_edge", area=80,
        bbox=bbox, center=center, max_diff=15,
    )
    ed.source_inspector = "cv"
    ed.inspector_mode = "fusion"
    ed.threshold_used = 4
    ed.min_area_used = 40
    ed.min_max_diff_used = 20
    ed.is_suspected_dust_or_scratch = is_dust
    ed.is_bomb = is_bomb
    if cv_filtered_mask is None:
        cv_filtered_mask = np.zeros((200, 200), dtype=np.uint8)
        cv_filtered_mask[95:105, 95:105] = 255
    ed.cv_filtered_mask = cv_filtered_mask
    ed.cv_mask_offset = cv_mask_offset
    return ed


class TestCVFusionDispatch:
    """dispatch: fusion+cv 走新 renderer，其他 CV 走舊 renderer"""

    def test_fusion_cv_routes_to_new_renderer(self, tmp_path):
        ed = _make_cv_fusion_defect()
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        with patch.object(saver, "_save_cv_fusion_edge_image",
                          return_value=str(tmp_path / "stub.png")) as mock_new, \
             patch.object(saver, "_save_patchcore_edge_image") as mock_pc:
            saver.save_edge_defect_image(
                tmp_path, "img", 0, ed, full, omit_image=None,
            )
            assert mock_new.called, "fusion+cv 應呼叫 _save_cv_fusion_edge_image"
            assert not mock_pc.called, "fusion+cv 不該走 PC renderer"

    def test_four_side_cv_still_uses_old_renderer(self, tmp_path):
        ed = _make_cv_fusion_defect(bbox=(100, 100, 40, 40), center=(120, 120))
        ed.side = "left"
        ed.inspector_mode = "cv"
        ed.source_inspector = ""
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        with patch.object(saver, "_save_cv_fusion_edge_image") as mock_new:
            saver.save_edge_defect_image(
                tmp_path, "img", 0, ed, full, omit_image=None,
            )
            assert not mock_new.called, "四邊 CV 不該走 fusion renderer"

    def test_non_fusion_cv_aoi_edge_uses_old_renderer(self, tmp_path):
        """inspector_mode='cv' + side='aoi_edge' 仍走舊 4 板路徑"""
        ed = _make_cv_fusion_defect()
        ed.inspector_mode = "cv"
        ed.source_inspector = ""
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        with patch.object(saver, "_save_cv_fusion_edge_image") as mock_new:
            saver.save_edge_defect_image(
                tmp_path, "img", 0, ed, full, omit_image=None,
            )
            assert not mock_new.called, "非 fusion 的 aoi_edge CV 應繼續走舊 renderer"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_edge_viz_cv_fusion_panel.py::TestCVFusionDispatch -v`
Expected: FAIL — `_save_cv_fusion_edge_image` 不存在

- [ ] **Step 3: Implement — 加空殼 + 分派**

先在 `capi_heatmap.py` 的 `_save_patchcore_edge_image` 定義之前（約 line 970）加一個空殼：

```python
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
        """Phase 7.2-B: Fusion 模式 CV defect 3 板組合圖（Detection / OMIT+Dust / Overlap）"""
        # 空殼：先回傳占位圖，後續 task 實作 panels
        h, w = 50 + 400 + 40, 1200
        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(placeholder, "CV Fusion Renderer (TBD)", (20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        filename = f"edge_cvfusion_{image_name}_{edge_defect.side}_{edge_index}.{self.save_format}"
        filepath = save_dir / filename
        cv2.imwrite(str(filepath), placeholder)
        return str(filepath)
```

然後在 `save_edge_defect_image` 分派處（約 line 608）調整：

```python
        inspector_mode = getattr(edge_defect, 'inspector_mode', 'cv')
        source_inspector = getattr(edge_defect, 'source_inspector', '')
        # PC 路徑
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_edge_viz_cv_fusion_panel.py::TestCVFusionDispatch -v`
Expected: PASS — 3 個測試綠

- [ ] **Step 5: Commit**

```bash
git add tests/test_edge_viz_cv_fusion_panel.py capi_heatmap.py
git commit -m "feat(edge-viz): Phase 7.2-B1 — 新增 _save_cv_fusion_edge_image 分派 (空殼)"
```

---

### Task B2: Panel 1 — CV Detection (藍 band 輪廓 + 紅 defect)

**Files:**
- Modify: `tests/test_edge_viz_cv_fusion_panel.py`
- Modify: `capi_heatmap.py::_save_cv_fusion_edge_image` 實作 Panel 1

- [ ] **Step 1: Write the failing test**

加到 `tests/test_edge_viz_cv_fusion_panel.py`：

```python
class TestCVFusionPanel1Detection:
    """Panel 1: 原圖 + 藍色 band 虛線輪廓 + 紅色 defect 像素"""

    def test_panel1_has_red_defect_overlay(self, tmp_path):
        """cv_filtered_mask=255 的像素在 Panel 1 應呈紅色 (R 高 G/B 低)"""
        ed = _make_cv_fusion_defect()
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        path = saver._save_cv_fusion_edge_image(
            save_dir=tmp_path, image_name="b2", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=None,
            dust_check_fn=None,
        )
        out = cv2.imread(path)
        h, w = out.shape[:2]
        header_h = 50
        panel_h = 400
        # Panel 1 在左側（0..panel_w），掃整個 Panel 1 區域找紅色像素（容忍位置偏移）
        panel_w_approx = w // 3
        panel1 = out[header_h:header_h + panel_h, 0:panel_w_approx]
        red_pixels = (panel1[:, :, 2] > 150) & (panel1[:, :, 0] < 100) & (panel1[:, :, 1] < 100)
        assert np.sum(red_pixels) >= 10, \
            f"Panel 1 應有紅色 defect overlay 像素 (>=10 px)，實 {np.sum(red_pixels)}"

    def test_panel1_has_blue_band_contour(self, tmp_path):
        """有 panel_polygon 與 band_px 時，Panel 1 沿 polygon 邊畫藍虛線"""
        # 建立一個大 polygon，讓 bbox ROI 確實有 polygon edge 在附近
        polygon = np.array([[900, 400], [1100, 400], [1100, 600], [900, 600]], dtype=np.int32)
        ed = _make_cv_fusion_defect(bbox=(920, 420, 40, 40), center=(940, 440))
        ed.panel_polygon = polygon  # Phase 7.2 新欄位（或透過 kwargs 傳入）
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        # Panel 1 掃描藍色像素（BGR B高 R低）
        class _Cfg:
            aoi_edge_boundary_band_px = 40
        path = saver._save_cv_fusion_edge_image(
            save_dir=tmp_path, image_name="b2band", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=None,
            dust_check_fn=None, edge_config=_Cfg(), panel_polygon=polygon,
        )
        out = cv2.imread(path)
        h, w = out.shape[:2]
        panel_w_approx = w // 3
        header_h = 50; panel_h = 400
        panel1 = out[header_h:header_h + panel_h, 0:panel_w_approx]
        blue_pixels = (panel1[:, :, 0] > 150) & (panel1[:, :, 2] < 100)
        assert np.sum(blue_pixels) > 10, \
            f"Panel 1 應該有藍色 band 輪廓像素（>10 px），實 {np.sum(blue_pixels)}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_edge_viz_cv_fusion_panel.py::TestCVFusionPanel1Detection -v`
Expected: FAIL — 空殼沒有 red/blue 像素

- [ ] **Step 3: Implement — Panel 1 的 Detection 圖**

改 `_save_cv_fusion_edge_image`，把空殼替換成：

```python
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
        """Phase 7.2-B: Fusion CV defect 3 板 (Detection / OMIT+Dust / Overlap)"""
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
            # 用 cv2.erode on fg_mask 做近似
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

        # Panel 2 / 3 先占位，下一 task 實作
        panel_h = 400
        scale = panel_h / max(panel1.shape[0], 1)
        panel_w = max(int(panel1.shape[1] * scale), 200)
        panel1_resized = cv2.resize(panel1, (panel_w, panel_h))
        placeholder2 = np.full((panel_h, panel_w, 3), 40, dtype=np.uint8)
        placeholder3 = np.full((panel_h, panel_w, 3), 40, dtype=np.uint8)
        panels = [panel1_resized, placeholder2, placeholder3]
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

        header_h = 50
        header = np.zeros((header_h, comp_w, 3), dtype=np.uint8)
        header_text = f"CV Edge: {side} | MaxDiff:{max_diff} | Area:{area}px"
        cv2.putText(header, header_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_edge_viz_cv_fusion_panel.py::TestCVFusionPanel1Detection -v`
Expected: PASS — 2 個測試綠

- [ ] **Step 5: Commit**

```bash
git add tests/test_edge_viz_cv_fusion_panel.py capi_heatmap.py
git commit -m "feat(edge-viz): Phase 7.2-B2 — CV fusion Panel 1 (藍 band + 紅 defect)"
```

---

### Task B3: Panel 2 — OMIT + Dust Overlay

**Files:**
- Modify: `tests/test_edge_viz_cv_fusion_panel.py`
- Modify: `capi_heatmap.py::_save_cv_fusion_edge_image` — placeholder2 換實作

- [ ] **Step 1: Write the failing test**

加到 `tests/test_edge_viz_cv_fusion_panel.py`：

```python
class TestCVFusionPanel2OMITDust:
    """Panel 2: OMIT 同 ROI + 藍色 dust overlay"""

    def test_panel2_shows_omit_gray(self, tmp_path):
        """Panel 2 中心像素應反映 OMIT 原圖灰度（128），不是黑 placeholder (40)"""
        omit = np.full((2000, 2000), 180, dtype=np.uint8)  # OMIT 均勻亮
        ed = _make_cv_fusion_defect()
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        path = saver._save_cv_fusion_edge_image(
            save_dir=tmp_path, image_name="b3omit", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
            dust_check_fn=None,
        )
        out = cv2.imread(path)
        h, w = out.shape[:2]
        panel_w_approx = w // 3
        header_h = 50; panel_h = 400
        p2_cx = panel_w_approx + panel_w_approx // 2 + 5
        p2_cy = header_h + panel_h // 2
        px = out[p2_cy, p2_cx]
        assert 120 < px[0] < 200 and 120 < px[1] < 200 and 120 < px[2] < 200, \
            f"Panel 2 應為灰 OMIT (~180)，實 {px.tolist()}"

    def test_panel2_has_blue_dust_overlay(self, tmp_path):
        """dust_check_fn 回傳 mask 時 Panel 2 對應區塊偏藍"""
        omit = np.full((2000, 2000), 180, dtype=np.uint8)

        def stub_dust(img):
            h, w = img.shape[:2]
            m = np.zeros((h, w), dtype=np.uint8)
            m[h // 2 - 30:h // 2 + 30, w // 2 - 30:w // 2 + 30] = 255
            return True, m, 0.1, ""

        ed = _make_cv_fusion_defect()
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        path = saver._save_cv_fusion_edge_image(
            save_dir=tmp_path, image_name="b3dust", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
            dust_check_fn=stub_dust,
        )
        out = cv2.imread(path)
        h, w = out.shape[:2]
        panel_w_approx = w // 3
        header_h = 50; panel_h = 400
        p2_cx = panel_w_approx + panel_w_approx // 2 + 5
        p2_cy = header_h + panel_h // 2
        px = out[p2_cy, p2_cx]
        assert px[0] > px[2] + 30, \
            f"Panel 2 dust 區應偏藍 B>R+30，實 B={px[0]} R={px[2]}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_edge_viz_cv_fusion_panel.py::TestCVFusionPanel2OMITDust -v`
Expected: FAIL — Panel 2 還是 placeholder 深灰

- [ ] **Step 3: Implement Panel 2**

在 `_save_cv_fusion_edge_image`，把 `placeholder2` 建構段替換成：

```python
        # --- Panel 2: OMIT + Dust Overlay ---
        dust_mask_omit = None
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
                    _is_d, dust_mask_raw, _br, _dt = dust_check_fn(omit_sub)
                    if dust_mask_raw is not None:
                        if dust_mask_raw.ndim == 3:
                            dust_mask_raw = cv2.cvtColor(dust_mask_raw, cv2.COLOR_BGR2GRAY)
                        dust_mask_omit = dust_mask_raw
                        _blend_color_on_mask(panel2, dust_mask_omit, (255, 100, 0), alpha=0.5)
                except Exception as e:
                    print(f"⚠️ CV Fusion Panel 2 dust check 失敗: {e}")
        else:
            panel2 = np.full(roi_bgr.shape, 40, dtype=np.uint8)
            cv2.putText(panel2, "No OMIT", (10, panel2.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        panel2_resized = cv2.resize(panel2, (panel_w, panel_h))
```

再把 `panels = [panel1_resized, placeholder2, placeholder3]` 改為 `panels = [panel1_resized, panel2_resized, placeholder3]`。

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_edge_viz_cv_fusion_panel.py::TestCVFusionPanel2OMITDust -v`
Expected: PASS — 2 個測試綠

- [ ] **Step 5: Commit**

```bash
git add tests/test_edge_viz_cv_fusion_panel.py capi_heatmap.py
git commit -m "feat(edge-viz): Phase 7.2-B3 — CV fusion Panel 2 (OMIT + dust overlay)"
```

---

### Task B4: Panel 3 — Overlap (紅/藍/紫)

**Files:**
- Modify: `tests/test_edge_viz_cv_fusion_panel.py`
- Modify: `capi_heatmap.py::_save_cv_fusion_edge_image` — placeholder3 換實作

- [ ] **Step 1: Write the failing test**

加到 `tests/test_edge_viz_cv_fusion_panel.py`：

```python
class TestCVFusionPanel3Overlap:
    """Panel 3: OMIT 底 + 紅 defect + 藍 dust + 紫色交集"""

    def test_panel3_has_purple_intersection(self, tmp_path):
        """defect mask 與 dust mask 交集處出現紫色像素 (B 高 R 高 G 低)"""
        omit = np.full((2000, 2000), 160, dtype=np.uint8)

        def stub_dust(img):
            h, w = img.shape[:2]
            m = np.zeros((h, w), dtype=np.uint8)
            # dust mask 覆蓋整塊 ROI 中央
            m[h // 2 - 50:h // 2 + 50, w // 2 - 50:w // 2 + 50] = 255
            return True, m, 0.1, ""

        # cv_filtered_mask 也在中央，與 dust mask 重疊
        cv_mask = np.zeros((200, 200), dtype=np.uint8)
        cv_mask[95:115, 95:115] = 255
        ed = _make_cv_fusion_defect(cv_filtered_mask=cv_mask, cv_mask_offset=(850, 350))
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        path = saver._save_cv_fusion_edge_image(
            save_dir=tmp_path, image_name="b4ov", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
            dust_check_fn=stub_dust,
        )
        out = cv2.imread(path)
        h, w = out.shape[:2]
        panel_w_approx = w // 3
        header_h = 50; panel_h = 400
        p3_left = panel_w_approx * 2 + 20
        panel3 = out[header_h:header_h + panel_h, p3_left:w - 5]
        # 紫色像素：B > 150, G < 80, R > 150
        purple = (panel3[:, :, 0] > 150) & (panel3[:, :, 1] < 80) & (panel3[:, :, 2] > 150)
        assert np.sum(purple) > 5, \
            f"Panel 3 應有紫色像素 (紅∩藍交集)，實 {np.sum(purple)} px"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_edge_viz_cv_fusion_panel.py::TestCVFusionPanel3Overlap -v`
Expected: FAIL — Panel 3 還是 placeholder

- [ ] **Step 3: Implement Panel 3**

在 `_save_cv_fusion_edge_image`，把 `placeholder3` 建構段替換成：

```python
        # --- Panel 3: Overlap (OMIT 底 + 紅 defect + 藍 dust + 紫交集) ---
        if omit_image is not None:
            panel3 = ensure_bgr(omit_sub).copy() if 'omit_sub' in locals() else roi_bgr.copy()
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
        # 畫藍色 dust
        if dust_mask_omit is not None:
            _blend_color_on_mask(panel3, dust_mask_omit, (255, 100, 0), alpha=0.5)
        # 畫紫色交集（純色覆蓋）
        if defect_mask_p3 is not None and dust_mask_omit is not None:
            # 尺寸對齊：dust_mask_omit 可能與 panel3 不同尺寸
            if dust_mask_omit.shape != panel3.shape[:2]:
                dust_resized = cv2.resize(dust_mask_omit, (panel3.shape[1], panel3.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
            else:
                dust_resized = dust_mask_omit
            overlap = cv2.bitwise_and(defect_mask_p3, dust_resized)
            panel3[overlap > 0] = (220, 0, 180)  # 紫

        panel3_resized = cv2.resize(panel3, (panel_w, panel_h))
```

再把 `panels = [panel1_resized, panel2_resized, placeholder3]` 改為 `panels = [panel1_resized, panel2_resized, panel3_resized]`。

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_edge_viz_cv_fusion_panel.py::TestCVFusionPanel3Overlap -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_edge_viz_cv_fusion_panel.py capi_heatmap.py
git commit -m "feat(edge-viz): Phase 7.2-B4 — CV fusion Panel 3 (Overlap 紅/藍/紫)"
```

---

### Task B5: CV fusion Header 含 COV → VERDICT

**Files:**
- Modify: `tests/test_edge_viz_cv_fusion_panel.py`
- Modify: `capi_heatmap.py::_save_cv_fusion_edge_image` — header 段

- [ ] **Step 1: Write the failing test**

加到 `tests/test_edge_viz_cv_fusion_panel.py`：

```python
class TestCVFusionHeader:
    """CV fusion header: MaxDiff / Area / COV → VERDICT"""

    def test_header_shows_ok_dust_verdict(self, tmp_path, monkeypatch):
        """is_suspected_dust_or_scratch=True 時 header 寫 '✓ OK (dust)'"""
        captured = []
        _real = cv2.putText
        def cap(img, text, *a, **kw):
            captured.append(text)
            return _real(img, text, *a, **kw)
        monkeypatch.setattr("capi_heatmap.cv2.putText", cap)

        ed = _make_cv_fusion_defect(is_dust=True)
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        saver._save_cv_fusion_edge_image(
            save_dir=tmp_path, image_name="b5ok", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=None,
            dust_check_fn=None,
        )
        all_text = " ".join(captured)
        assert "OK (dust)" in all_text, \
            f"header 應含 'OK (dust)'，實際 putText: {captured}"

    def test_header_shows_ng_verdict(self, tmp_path, monkeypatch):
        """普通 NG case header 寫 'NG'"""
        captured = []
        _real = cv2.putText
        def cap(img, text, *a, **kw):
            captured.append(text)
            return _real(img, text, *a, **kw)
        monkeypatch.setattr("capi_heatmap.cv2.putText", cap)

        ed = _make_cv_fusion_defect()
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        saver._save_cv_fusion_edge_image(
            save_dir=tmp_path, image_name="b5ng", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=None,
            dust_check_fn=None,
        )
        all_text = " ".join(captured)
        assert " NG" in all_text, f"header 應含 'NG'，實 putText: {captured}"

    def test_header_shows_cov_when_dust_check_runs(self, tmp_path, monkeypatch):
        """OMIT + dust_check_fn 有跑 → header 含 'COV=' 字串"""
        captured = []
        _real = cv2.putText
        def cap(img, text, *a, **kw):
            captured.append(text)
            return _real(img, text, *a, **kw)
        monkeypatch.setattr("capi_heatmap.cv2.putText", cap)

        def stub(img):
            m = np.zeros(img.shape[:2], dtype=np.uint8)
            m[50:100, 50:100] = 255
            return True, m, 0.1, ""

        ed = _make_cv_fusion_defect()
        full = np.full((2000, 2000, 3), 128, dtype=np.uint8)
        omit = np.full((2000, 2000), 160, dtype=np.uint8)
        saver = HeatmapSaver(base_dir=str(tmp_path), save_format="png")
        saver._save_cv_fusion_edge_image(
            save_dir=tmp_path, image_name="b5cov", edge_index=0,
            edge_defect=ed, full_image=full, omit_image=omit,
            dust_check_fn=stub,
        )
        all_text = " ".join(captured)
        assert "COV=" in all_text, f"header 應含 'COV='，實 putText: {captured}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_edge_viz_cv_fusion_panel.py::TestCVFusionHeader -v`
Expected: FAIL — 目前 header 沒有 verdict / COV

- [ ] **Step 3: Implement — Header verdict 邏輯**

把 `_save_cv_fusion_edge_image` 的 header 建構段（`header_text = ...`）替換成：

```python
        # Verdict 分類
        if is_bomb:
            verdict = "BOMB (filter OK)"; v_color = (255, 0, 255)
        elif is_cv_ok:
            verdict = "OK (CV filtered)"; v_color = (0, 255, 0)
        elif is_dust:
            verdict = "✓ OK (dust)"; v_color = (0, 255, 0)
        else:
            verdict = "NG"; v_color = (0, 0, 255)

        # COV 值（若有跑 dust_check_fn + 有 defect）
        cov_text = ""
        if defect_mask_p3 is not None and dust_mask_omit is not None:
            if dust_mask_omit.shape != defect_mask_p3.shape[:2]:
                dust_cmp = cv2.resize(dust_mask_omit, defect_mask_p3.shape[::-1],
                                       interpolation=cv2.INTER_NEAREST)
            else:
                dust_cmp = dust_mask_omit
            inter = np.count_nonzero((defect_mask_p3 > 0) & (dust_cmp > 0))
            defect_area = max(1, np.count_nonzero(defect_mask_p3 > 0))
            cov_val = inter / defect_area
            metric_label = "COV" if dust_metric == "coverage" else "IOU"
            cov_text = f" | {metric_label}={cov_val:.2f}"

        header_text_info = f"CV Edge: {side} | MaxDiff:{max_diff} | Area:{area}px{cov_text} | "
        self._draw_split_color_header(header, header_text_info, verdict, v_color,
                                       y=30, font_scale=0.65)
```

（取代原本單行 `cv2.putText(header, header_text, ...)`）

注意：`defect_mask_p3` / `dust_mask_omit` 要在 Panel 3 段建構完成後才存在 — 若 header 建構順序在它們之前要調整。把 Panel 3 建構移到 header 建構之前即可（若目前不是，就調整順序）。

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_edge_viz_cv_fusion_panel.py::TestCVFusionHeader -v`
Expected: PASS — 3 個測試綠

- [ ] **Step 5: Commit**

```bash
git add tests/test_edge_viz_cv_fusion_panel.py capi_heatmap.py
git commit -m "feat(edge-viz): Phase 7.2-B5 — CV fusion header COV → verdict"
```

---

### Task B6: Phase B 全套 regression + tuning log

**Files:**
- Modify: `docs/edge-cv-tuning-log.md`

- [ ] **Step 1: 跑 Phase B 全部測試**

Run: `python -m pytest tests/test_edge_viz_cv_fusion_panel.py -v`
Expected: ~9 個測試綠

- [ ] **Step 2: 跑 Edge / Fusion 迴歸**

Run: `python -m pytest tests/test_aoi_edge_fusion.py tests/test_aoi_edge_pc_roi_shift.py tests/test_edge_viz_pc_panel.py -v`
Expected: 全綠

- [ ] **Step 3: 跑全套**

Run: `python -m pytest tests/`
Expected: >= 177 + 新增測試 passed

- [ ] **Step 4: tuning log + commit**

加到 `docs/edge-cv-tuning-log.md`：

```markdown
### Phase 7.2-B — CV fusion 組 3 板重構 (2026-04-XX)

- 新增 `_save_cv_fusion_edge_image`：fusion + cv 走新 renderer；四邊 CV / 非 fusion CV 仍走 4 板
- Panel 1 Detection：ROI 原圖 + 藍色 polygon band 虛線輪廓 + 紅色 cv_filtered_mask
- Panel 2 OMIT + Dust：OMIT 同 ROI 範圍 + 藍色 dust_mask overlay (BGR 255,100,0, α=0.5)
- Panel 3 Overlap：OMIT 底 + 紅 defect + 藍 dust + 紫色交集 (BGR 220,0,180 純色)
- Header：`CV Edge: {side} | MaxDiff | Area | COV=x.xx | <VERDICT>`，verdict 含 `NG / ✓ OK (dust) / OK (CV filtered) / BOMB`
- 測試：`tests/test_edge_viz_cv_fusion_panel.py` 新增 9 項
```

```bash
git add docs/edge-cv-tuning-log.md
git commit -m "docs: Phase 7.2-B tuning log — CV fusion 組 3 板重構紀錄"
```

---

## Phase C — Header helper 統一

抽出 `_render_edge_header` helper，PC / CV 兩個 renderer 都呼叫。驗收點：PC / CV header 排版不跑版、字體大小一致、verdict 大字突出。

### Task C1: 建 `_render_edge_header` helper + 單測

**Files:**
- Create: `tests/test_edge_viz_header.py`
- Modify: `capi_heatmap.py` 加 `_render_edge_header` method

- [ ] **Step 1: Write the failing test**

```python
# tests/test_edge_viz_header.py (新建)
"""Phase 7.2 Phase C — Edge header helper 單元測試"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from capi_heatmap import HeatmapSaver


class TestRenderEdgeHeader:
    """helper：產生 header image，含 info 白字 + verdict 彩字 + 可選 extra"""

    def test_returns_image_of_correct_size(self):
        saver = HeatmapSaver(base_dir=".", save_format="png")
        img = saver._render_edge_header(
            width=1200, info="PC Edge | Score:0.5 | ", verdict="NG",
            verdict_color=(0, 0, 255),
        )
        assert img.shape[0] == 50 and img.shape[1] == 1200
        assert img.dtype == np.uint8

    def test_verdict_uses_correct_color(self):
        """NG 區塊應有紅色像素、OK 區塊應有綠色像素"""
        saver = HeatmapSaver(base_dir=".", save_format="png")
        img_ng = saver._render_edge_header(
            width=800, info="X | ", verdict="NG", verdict_color=(0, 0, 255),
        )
        img_ok = saver._render_edge_header(
            width=800, info="X | ", verdict="OK", verdict_color=(0, 255, 0),
        )
        # NG header 右側區塊掃 red pixel 數量
        right_ng = img_ng[:, 400:]
        red = (right_ng[:, :, 2] > 150) & (right_ng[:, :, 0] < 80)
        assert np.sum(red) > 5, "NG verdict 應畫紅字"
        right_ok = img_ok[:, 400:]
        green = (right_ok[:, :, 1] > 150) & (right_ok[:, :, 0] < 80) & (right_ok[:, :, 2] < 80)
        assert np.sum(green) > 5, "OK verdict 應畫綠字"

    def test_extra_right_side_rendered(self, monkeypatch):
        """extra_right 文字應渲染（透過 putText 攔截驗證內容）"""
        captured = []
        _real = cv2.putText
        def cap(img, text, *a, **kw):
            captured.append(text)
            return _real(img, text, *a, **kw)
        monkeypatch.setattr("capi_heatmap.cv2.putText", cap)

        saver = HeatmapSaver(base_dir=".", save_format="png")
        saver._render_edge_header(
            width=1200, info="PC Edge | ", verdict="NG", verdict_color=(0, 0, 255),
            extra_right="PC dx=+0 dy=-192",
        )
        all_text = " ".join(captured)
        assert "PC dx=+0 dy=-192" in all_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_edge_viz_header.py -v`
Expected: FAIL — `_render_edge_header` 不存在

- [ ] **Step 3: Implement helper**

在 `capi_heatmap.py` 的 `HeatmapSaver` class 裡，`_draw_split_color_header` 定義之後加：

```python
    def _render_edge_header(
        self,
        width: int,
        info: str,
        verdict: str,
        verdict_color: Tuple[int, int, int],
        extra_right: str = "",
        height: int = 50,
        font_scale: float = 0.65,
    ) -> np.ndarray:
        """Phase 7.2-C: Edge defect 組合圖通用 header helper。

        Layout:
          [左側 info 白字] [verdict 大字彩色] ............ [右側 extra_right 灰字]

        Verdict 字大小 = font_scale × 1.5，thickness=2 突出。
        """
        header = np.zeros((height, width, 3), dtype=np.uint8)
        # info 白字
        cv2.putText(header, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), 2)
        (info_w, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        # verdict 大字
        verdict_scale = font_scale * 1.5
        cv2.putText(header, verdict, (10 + info_w, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, verdict_scale, verdict_color, 2)
        # 右側 extra 字（若有）
        if extra_right:
            (ex_w, _), _ = cv2.getTextSize(extra_right, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(header, extra_right, (width - ex_w - 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)
        return header
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_edge_viz_header.py -v`
Expected: PASS — 3 個測試綠

- [ ] **Step 5: Commit**

```bash
git add tests/test_edge_viz_header.py capi_heatmap.py
git commit -m "feat(edge-viz): Phase 7.2-C1 — _render_edge_header helper"
```

---

### Task C2: PC / CV renderer 改呼叫 helper

**Files:**
- Modify: `capi_heatmap.py::_save_patchcore_edge_image` header 段
- Modify: `capi_heatmap.py::_save_cv_fusion_edge_image` header 段

- [ ] **Step 1: 先確認 A4/B5 的 header 測試仍會通過（對照基準）**

Run: `python -m pytest tests/test_edge_viz_pc_panel.py::TestPCHeaderShiftInfo tests/test_edge_viz_cv_fusion_panel.py::TestCVFusionHeader -v`
Expected: 現況綠（因為 A4/B5 已完成）

- [ ] **Step 2: 改 PC renderer 呼叫 helper**

在 `_save_patchcore_edge_image` 找到 header 建構段（由 A4 產生的版本）：

```python
        score_cmp = ">=" if score >= threshold else "<"
        info_part = f"PC Edge [v1]: {side} | Score:{score:.3f}{score_cmp}Thr:{threshold:.3f} | Area:{area}px | "
        self._draw_split_color_header(header, info_part, verdict, verdict_color, y=30, font_scale=0.65)

        # ... A4 寫的 extra_text 段 ...
```

整塊替換成：

```python
        score_cmp = ">=" if score >= threshold else "<"
        info_part = f"PC Edge [v1]: {side} | Score:{score:.3f}{score_cmp}Thr:{threshold:.3f} | Area:{area}px | "

        shift_dx = int(getattr(edge_defect, 'pc_roi_shift_dx', 0))
        shift_dy = int(getattr(edge_defect, 'pc_roi_shift_dy', 0))
        pc_fb = str(getattr(edge_defect, 'pc_roi_fallback_reason', ''))
        extra = ""
        if shift_dx or shift_dy:
            extra = f"PC dx={shift_dx:+d} dy={shift_dy:+d}"
        elif pc_fb == "shift_insufficient":
            extra = "PC-FB=shift_insufficient(offset short)"
        elif pc_fb == "concave_polygon":
            extra = "PC-FB=concave_polygon(concave)"
        elif pc_fb == "shift_disabled":
            extra = "PC-FB=shift_disabled"
        elif pc_fb:
            extra = f"PC-FB={pc_fb}"

        header = self._render_edge_header(
            width=comp_w, info=info_part, verdict=verdict, verdict_color=verdict_color,
            extra_right=extra,
        )
```

（並刪除原本 `header = np.zeros(...)` 那幾行）

- [ ] **Step 3: 改 CV renderer 呼叫 helper**

在 `_save_cv_fusion_edge_image` 找到 B5 產生的 header 段：

```python
        header_text_info = f"CV Edge: {side} | MaxDiff:{max_diff} | Area:{area}px{cov_text} | "
        self._draw_split_color_header(header, header_text_info, verdict, v_color,
                                       y=30, font_scale=0.65)
```

替換成：

```python
        header_text_info = f"CV Edge: {side} | MaxDiff:{max_diff} | Area:{area}px{cov_text} | "
        header = self._render_edge_header(
            width=comp_w, info=header_text_info, verdict=verdict, verdict_color=v_color,
        )
```

（並刪除原本 `header = np.zeros(...)` 那幾行）

- [ ] **Step 4: 跑既有 header 測試 + 新 helper 測試**

Run: `python -m pytest tests/test_edge_viz_pc_panel.py tests/test_edge_viz_cv_fusion_panel.py tests/test_edge_viz_header.py -v`
Expected: 全綠

- [ ] **Step 5: Commit**

```bash
git add capi_heatmap.py
git commit -m "refactor(edge-viz): Phase 7.2-C2 — PC/CV renderer 共用 _render_edge_header"
```

---

### Task C3: Phase C 全套 regression + tuning log

**Files:**
- Modify: `docs/edge-cv-tuning-log.md`

- [ ] **Step 1: 跑 Phase C 相關測試**

Run: `python -m pytest tests/test_edge_viz_header.py tests/test_edge_viz_pc_panel.py tests/test_edge_viz_cv_fusion_panel.py -v`
Expected: 全綠

- [ ] **Step 2: 跑全套**

Run: `python -m pytest tests/`
Expected: >= 177 + 新增測試全綠

- [ ] **Step 3: tuning log + 視覺抽檢**

加到 `docs/edge-cv-tuning-log.md`：

```markdown
### Phase 7.2-C — Header helper 統一 (2026-04-XX)

- 抽出 `HeatmapSaver._render_edge_header(width, info, verdict, verdict_color, extra_right)`
- PC / CV fusion renderer 皆呼叫 helper；verdict 字大小 ×1.5 突出（font_scale 0.65 × 1.5 = 0.975）
- 測試：`tests/test_edge_viz_header.py` 新增 3 項
```

視覺抽檢：人工打開 web dashboard 看 3-5 個實際 record（PC NG / PC OK dust / CV NG / CV OK dust），確認 header / panel 排版無跑版。

- [ ] **Step 4: Final commit**

```bash
git add docs/edge-cv-tuning-log.md
git commit -m "docs: Phase 7.2-C tuning log — header helper 統一"
```

---

## 風險與 Mitigation

| 風險 | Mitigation |
|---|---|
| OMIT shifted 後超出 OMIT 影像邊界 | Task A2 的 `osx1/osy1/osx2/osy2 = max(0,...)/min(ow,...)` clamp + black pad，與現況一致 |
| 舊 record 沒 `pc_roi_origin_x/y`（=0 預設值） | Task A2 實作中檢測 `ox_origin==0 and oy_origin==0` → fallback 用 AOI center |
| `dust_check_fn` 對 OMIT sub-ROI 丟 exception | Task A3 / B3 的 try/except 包起來，仍產圖（Panel 3 / Panel 2 無 overlay） |
| 四邊 CV (`side in {left,right,top,bottom}`) 意外被 routed 到新 renderer | Task B1 第 2/3 測試驗證 dispatch 正確；`side` 不檢查是因 `inspector_mode != 'fusion'` 已擋掉 |
| Phase C helper 改動 font_scale 造成原 header 排版跑版 | Task C2 後跑 A4/B5 現有 header 測試（放入 C2 Step 4）；人工抽檢實際 record |
| `panel_polygon` 沒傳進 `_save_cv_fusion_edge_image` | Task B1 dispatch 段從 `edge_defect.panel_polygon` 取（若 EdgeDefect 無此欄位則為 None，Panel 1 就不畫藍線） |

---

## Self-Review Check

- [x] Spec coverage: Layout 1 (PC 3 板) = Tasks A1-A4；Layout 2 (CV 3 板) = Tasks B1-B5；Layout 3 (header verdict) = Tasks A4 + B5 + C1-C2；Layout 4 (尺寸) 沿用現況
- [x] No placeholders: 每個 step 有具體 code、命令、檔案
- [x] Type consistency: `dust_check_fn`, `pc_roi_origin_x/y`, `cv_filtered_mask`, `cv_mask_offset` 名稱前後一致
- [x] File paths: 全部絕對引用（`capi_heatmap.py::xxx` / `tests/test_xxx.py`）

## 時序建議

每個 Task 結束即 commit（避免半成品堆積）。Phase A / B / C 各自最後一個 task 做 regression + tuning log + commit，該 commit 作為 phase 部署點。
