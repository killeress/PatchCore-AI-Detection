# Edge Defect 組合圖視覺化重構設計

**日期**：2026-04-23
**作者**：Ray（設計）/ Claude（記錄）
**前置**：
  - `2026-04-22-aoi-edge-fusion-inspector-design.md`（Phase 6 fusion）
  - `2026-04-23-aoi-edge-fusion-pc-roi-inward-shift-design.md`（Phase 7 PC ROI shift）
**相關程式**：`capi_heatmap.py` (`_save_patchcore_edge_image` / `save_edge_defect_image`)

## 問題陳述

現場對 `record_detail` 內的 edge defect 組合圖回饋「難理解」，具體痛點：

1. **PC 組 Panel 1 蓋太多 overlay**：原始 ROI 被紅色斜線 mask + 黃色十字/圓圈蓋住中心，defect 細節看不清。
2. **PC Panel 3 OMIT 位置沒跟著 shift**：Phase 7 起 PC ROI 會內移（例：AOI 近底邊時上移 192 px），但 OMIT panel 仍以 AOI 中心擷取，兩者不對齊 → 使用者以為 OMIT 看的是 PC 分析的同一塊，實際不是。
3. **CV 組的表面翻 OK 邏輯不明**：現有 4 板（Original / Highlight / OMIT / Defect vs OMIT IOU）把判定理由埋在 Panel 4 的 R/G/B overlay，但沒有明確文字寫出「因為 COV=0.82 ≥ 0.3，判定是表面灰塵 → 翻 OK」。

## 目標

1. PC 組 Panel 1 純呈現「AOI 本來指到的位置」的 raw ROI，不加任何判定 overlay / marker。
2. PC 組 Panel 2/3 皆對齊 shifted ROI 位置（Heatmap 與 OMIT 同框），讓使用者明確看到「PC 實際搬去哪分析」。
3. OMIT 顯示欄位統一加上 dust / scratch 偵測結果 overlay，呼應 `check_dust_or_scratch_feature` 回傳的 mask。
4. CV 組以 3 板明確呈現「偵測 → 表面檢查 → 判定翻轉」流程，header 大字寫出最終判定與依據（`COV=0.82 → ✓ OK (dust)` 或 `NG`）。
5. PC / CV 組的 header 格式統一，讓使用者一眼知道是哪個 inspector 在講話。

## 非目標

- 不改變判定邏輯（PC threshold、CV COV/IOU dust match、bomb match 全部不動）。
- 不調整四邊（left / right / top / bottom）CV defect 的 4 板視覺化（物理上沒有 polygon band 概念，維持現況）。
- 不處理 debug 頁（`/debug`）的獨立視覺化 — 那套與 production record 分開，另議。

## 現況摘要

### `capi_heatmap.py::save_edge_defect_image` 分派邏輯（line 608-614）

```python
if inspector_mode == 'patchcore' or (
    inspector_mode == 'fusion' and source_inspector == 'patchcore'
):
    return self._save_patchcore_edge_image(...)   # 3 板 PC
# 以下是 CV 4 板路徑
```

### PC 路徑現況（`_save_patchcore_edge_image`, line 972-）

| Panel | 內容 | 問題 |
|---|---|---|
| 1 | `render_pc_masked_roi(roi, fg_mask)` + 黃色圓圈 marker 中心 | 紅色斜線 mask + marker 遮蓋 defect |
| 2 | `render_pc_overlay(roi, fg_mask, anomaly_map)` + `[PC dx= dy=]` badge | OK |
| 3 | `omit_image[cx-256 : cx+256, cy-256 : cy+256]` 擷取 **以 AOI center** | 不對齊 shifted ROI |

header 是 `PC Edge [v1]: {side} | Score:{score}>=Thr:{thr} | Area:{area}px | NG/OK`，已有 Score/Thr/Area/Verdict，但 shift/fallback 資訊在 Panel 2 badge 而不在 header。

### CV 路徑現況（`save_edge_defect_image` 主分支, line 616-）

4 panels：
1. Original ROI
2. Defect Highlight（紅色 defect 像素，或橘色=dust、洋紅=bomb）
3. OMIT ROI + dust mask 藍色 overlay
4. Defect vs OMIT overlay（R=defect only, G=overlap, B=dust only）+ COV/IOU 文字

Header 現已包含 MaxDiff / Area / COV 等，但 verdict 沒有「OK (dust)」這種帶原因的標記。

## 設計

### Layout 1：PC 組 3 板

```
┌─────────────────────── Header ───────────────────────────┐
│ PC Edge [v1]: aoi_edge | Score:0.764>=Thr:0.500          │
│ | Area:95288px | <NG>   PC dx=0 dy=-192                  │
└──────────────────────────────────────────────────────────┘
┌─────────────────┬─────────────────┬─────────────────────┐
│ Panel 1         │ Panel 2         │ Panel 3             │
│ Original ROI    │ PatchCore       │ OMIT + Dust         │
│ (AOI centered)  │ Heatmap         │ (shifted)           │
│                 │ (shifted)       │                     │
│ ← raw pixels    │ ← heatmap       │ ← OMIT 原像素       │
│   no mask       │   colormap      │  + 藍色 dust mask   │
│   no marker     │   + polygon     │    overlay          │
│                 │   mask overlay  │                     │
└─────────────────┴─────────────────┴─────────────────────┘
```

**Panel 1 — Original ROI (AOI centered)**

- 以 AOI 座標為中心擷取 `tile_size × tile_size` raw pixels
- **不套 fg_mask 紅斜線**、不畫任何 marker（去除當前黃色圓圈）
- Label：`Original ROI (AOI)`

**Panel 2 — PatchCore Heatmap (shifted ROI)**

- 沿用現有 `render_pc_overlay(pc_roi_bgr, pc_fg_mask, pc_anomaly_map)`
- `pc_roi` / `pc_fg_mask` / `pc_anomaly_map` 來自 inspector 存的 shifted 版本（`edge_defect.pc_roi` 已是 shifted）
- `[PC dx=X dy=Y]` 或 `[PC FB: shift_insufficient]` 等 badge 保留左上角
- Label：`PatchCore Heatmap (shifted)` 若有 shift，否則 `PatchCore Heatmap`

**Panel 3 — OMIT + Dust Overlay (shifted 位置)**

- 以 **shifted ROI 左上角** `(pc_roi_origin_x, pc_roi_origin_y)` 擷取 OMIT，而非 AOI center
- 在 OMIT BGR 上套用 `check_dust_or_scratch_feature` 回傳的 `dust_mask` 藍色半透明 overlay（BGR `(255, 100, 0)`，同 CV path 既有色）
- 若 `omit_overexposed` 則 label 標注 `OMIT + Dust (overexposed)`；若 OMIT 缺失則維持現況 fallback（文字 "No OMIT"）

### Layout 2：CV 組 3 板（fusion 模式，source_inspector=cv）

```
┌─────────────────────── Header ───────────────────────────┐
│ CV Edge: aoi_edge | MaxDiff:12 | Area:85px |             │
│ COV=0.82 → <✓ OK (dust)>                                 │
└──────────────────────────────────────────────────────────┘
┌─────────────────┬─────────────────┬─────────────────────┐
│ Panel 1         │ Panel 2         │ Panel 3             │
│ CV Detection    │ OMIT + Dust     │ Overlap             │
│                 │                 │                     │
│ 原 ROI          │ OMIT 原像素     │ defect(紅) +        │
│ + 藍色 band     │ + 藍色 dust     │ dust(藍) 疊圖       │
│   輪廓虛線      │   overlay       │ + 紫=交集           │
│ + 紅色 defect   │                 │ → 翻 OK 的視覺      │
│   像素          │                 │   憑據              │
└─────────────────┴─────────────────┴─────────────────────┘
```

**Panel 1 — CV Detection**

- 以 `bbox + padding=100` 擷取 ROI（同現況）
- 藍色虛線繪製 polygon edge 沿 `band_px` 內縮的管轄帶輪廓（fusion 模式下才有；若 `inspector_mode != 'fusion'` 或沒 polygon，不畫藍線）
- 紅色半透明 overlay `cv_filtered_mask`（dilate 3×3 保薄線 resize 可見度，同現況）
- Dust/bomb 情境的顏色沿用現況（`is_bomb=洋紅 (255,0,255)` / `is_dust=橘 (0,200,255)` / 普通=紅 `(0,0,255)`）
- Label：`CV Detection (band)` 於 fusion 模式，`CV Detection` 於非 fusion 模式

**Panel 2 — OMIT + Dust Overlay**

- 以 **Panel 1 ROI 的同座標範圍**擷取 OMIT（不是 AOI center，是 CV defect bbox 的 ROI）
- 藍色 overlay 套 dust_mask（與 PC 組 Panel 3 同色 `(255, 100, 0)`）
- Label：`OMIT + Dust`

**Panel 3 — Overlap**

- 基底為 **OMIT（不是原 ROI）**，強化「表面判定」語意：紅色 defect 疊在表面光路上，視覺上直接說「defect 落在表面有灰塵的位置」
- 紅色：`vis_mask`（CV defect dilated）半透明覆蓋
- 藍色：`dust_mask_omit` 半透明覆蓋
- **紫色**：紅 mask ∩ 藍 mask 的交集區，BGR `(180, 0, 220)` 純色覆蓋（不再半透明，強調重疊）
- Label：`Overlap`

### Layout 3：Header Verdict 規格

統一格式：

```
<INSPECTOR> Edge[ [v1]: {side}] | <metric fields> | {badges} <VERDICT> {extras}
```

#### PC header

```
PC Edge [v1]: aoi_edge | Score:0.764>=Thr:0.500 | Area:95288px | <VERDICT> {shift_info}
```

- `<VERDICT>`：紅色 `NG` / 綠色 `✓ OK` / 綠色 `✓ OK (dust)`（現有 `is_cv_ok` / `is_suspected_dust_or_scratch` 判斷）
- `{shift_info}` 放最右側（可選）：
  - 有 shift → `PC dx={dx:+d} dy={dy:+d}`
  - `pc_roi_fallback_reason="shift_insufficient"` → `PC-FB=shift_insufficient(偏移不足)`
  - `"concave_polygon"` → `PC-FB=concave_polygon(凹角)`
  - `"shift_disabled"` → `PC-FB=shift_disabled`
  - 無 shift 且無 fallback → 省略

#### CV header

```
CV Edge: aoi_edge | MaxDiff:12 | Area:85px | COV=0.82 → <VERDICT>
```

- CV 無 patchcore score，用 MaxDiff / Area
- `COV=X.XX`（或 `IOU=X.XX`，依 `dust_metric` 設定）— 只在「有比對到 OMIT」時顯示；OMIT 缺失或未觸發 dust check 時省略
- `<VERDICT>`：
  - `NG` 紅字 — CV defect 未被 dust match flip
  - `✓ OK (dust)` 綠字 — `is_suspected_dust_or_scratch=True` 且原因是灰塵
  - `✓ OK (scratch)` 綠字 — 刮痕來源（若 classifier 標記）
  - `✓ OK (CV filtered)` 綠字 — `is_cv_ok=True` 的被濾原因（`Diff<Thr` / `Area<Min` 等列在 Panel 1 corner）

#### 顏色規範

- NG 紅：BGR `(0, 0, 255)`
- OK 綠：BGR `(0, 255, 0)` 或 `(80, 220, 80)`（視對比調整）
- Header 背景：黑底（現況）
- Verdict 字大小：原 header 字 size ×1.5（現況 0.6 → 0.9），bold effect 用 `thickness=2`

### Layout 4：Panel 尺寸與拼接

沿用現況：

- `panel_h = 400`
- Panel width 按 ROI aspect ratio scale，最小 200
- Panel 間距 `gap_w = 10`
- Header height 按行數動態；verdict 大字那行多 10-15 px
- Label bar 底部高度 40 px

### 資料流

```
capi_inference._inspect_roi_fusion
  ↓ 產生 EdgeDefect（含 pc_roi / pc_fg_mask / pc_anomaly_map / cv_filtered_mask 等）
  ↓ 各欄位已具備：
  │   - center = AOI 座標 (img_x, img_y)
  │   - pc_roi_origin_x/y = shifted ROI 左上角
  │   - pc_roi_shift_dx/dy
  │   - pc_roi_fallback_reason
  ↓
capi_heatmap.save_edge_defect_image
  ├→ source='patchcore' → _save_patchcore_edge_image（改版）
  │   ├→ Panel 1: full_image[cy-256:cy+256, cx-256:cx+256] 純 raw crop
  │   ├→ Panel 2: pc_roi + pc_fg_mask + pc_anomaly_map (shifted 已包含)
  │   └→ Panel 3: omit_image[pc_roi_origin_y:+512, pc_roi_origin_x:+512]
  │              ⊕ check_dust_or_scratch_feature(omit_sub) → 藍 overlay
  │
  └→ source='cv' 且 inspector_mode='fusion' → _save_cv_fusion_edge_image（新）
      ├→ Panel 1: bbox ROI + polygon band 藍虛線 + cv_filtered_mask 紅
      ├→ Panel 2: OMIT 同 bbox ROI 範圍 ⊕ dust_mask 藍
      └→ Panel 3: Panel 2 底圖 ⊕ defect(紅) ⊕ dust(藍) ⊕ overlap(紫)
```

四邊 CV（`side ∈ {left, right, top, bottom}`）維持 `save_edge_defect_image` 現有 4 板，不走新路徑。

## 實作切分（便於分段 review）

### Phase A：PC 組重構（優先）

1. `capi_heatmap.py::_save_patchcore_edge_image`
   - Panel 1 改 `full_image` AOI 中心 raw crop（移除 `render_pc_masked_roi` / `cv2.circle`）
   - Panel 3 OMIT 擷取改從 `(edge_defect.pc_roi_origin_x, pc_roi_origin_y)` 起點
   - Panel 3 加 dust_mask 藍色 overlay（呼叫 `check_dust_or_scratch_feature`，沿用 CV path 現有寫法）
   - Header 加 shift/fallback 資訊字段
2. 新增測試 `tests/test_edge_viz_pc_panel.py`：
   - Panel 1 是 AOI centered crop（比對像素區塊）
   - Panel 3 OMIT 位置 = shifted origin（不是 AOI center）
   - dust_check_fn 為 None 時 Panel 3 無 overlay 且不爆錯
   - Header 含 `PC dx= dy=` 或 `PC-FB=` 字串

### Phase B：CV 組重構（fusion mode only）

1. 新增 `capi_heatmap.py::_save_cv_fusion_edge_image`
   - 分派規則：`inspector_mode=='fusion' and source_inspector=='cv'` 走新函式
   - 其他 CV 情境繼續走舊的 4 板 `save_edge_defect_image` 主分支
2. Panel 1：bbox + padding ROI → 原圖底 + polygon band 藍虛線（用 `panel_polygon` + `aoi_edge_boundary_band_px`）+ 紅色 defect overlay
3. Panel 2：OMIT 同 bbox ROI 範圍 + dust_mask 藍 overlay
4. Panel 3：OMIT 底 + 紅 / 藍 / 紫色 overlap overlay
5. Header 加 `COV=X.XX → <VERDICT>` 含 verdict 分類邏輯
6. 新增測試 `tests/test_edge_viz_cv_fusion_panel.py`：
   - fusion + cv 路徑分派正確（patch 到新函式）
   - 4 邊 CV 仍走舊函式（patch 確認未進新函式）
   - Panel 1 藍虛線在 polygon edge 沿 band_px 內縮處
   - Panel 3 紫色像素 = 紅 ∩ 藍

### Phase C：header 樣式統一

1. 抽出 `_render_header(inspector, verdict, metrics, extras)` helper
   - PC / CV 皆用
   - Verdict 字大小 ×1.5 + 顏色（紅/綠）
2. 各 callsite 改呼叫 helper

## 風險與 open questions

| 風險 | 影響 | Mitigation |
|---|---|---|
| OMIT shifted 後的座標超出 OMIT 影像邊界 | Panel 3 擷取失敗 | 沿用現況 dx1/dy1/dx2/dy2 clamp + black pad 邏輯 |
| `check_dust_or_scratch_feature` 對 OMIT sub-ROI 行為 | dust mask 可能異常 | 參數與 CV path 一致，回歸測試覆蓋 |
| 舊 record 沒有 `pc_roi_origin_x/y`（Phase 7 前） | Panel 3 無法 shifted 擷取 | `getattr(edge_defect, 'pc_roi_origin_x', None)` → None 則回退 AOI center 擷取 |
| 四邊 CV 視覺化未動但 header helper 統一 | 格式跑版 | helper 維持 CV 4 板呼叫方式不變，僅 fusion 新路徑用新 verdict 欄位 |

## 測試計畫

- 新增 Phase A unit tests（~5 項）
- 新增 Phase B unit tests（~6 項）
- Regression：現有 `test_aoi_edge_fusion.py` + `test_aoi_edge_pc_roi_shift.py`（44 項）全綠
- 全套 `pytest tests/`（目標 ≥ 177 passed / 2 skipped 維持）
- 人工：在 web dashboard 抓 3-5 個實際 record（PC NG / PC OK dust / CV NG / CV OK dust），比對新版組合圖與判定結果一致

## 時序對齊

此重構僅動視覺化層，不動判定邏輯，可獨立 deploy：
1. Phase A 做完 → commit → 可部署（PC 組改版）
2. Phase B 做完 → commit → 可部署（CV fusion 改版）
3. Phase C header 統一 → commit → 可部署

每 phase 結束都跑全套 pytest 再 commit。
