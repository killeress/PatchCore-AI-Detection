# AOI 邊緣 CV+PatchCore 空間分權 Fusion Inspector 設計

**日期**：2026-04-22
**作者**：Ray（設計）/ Claude（記錄）
**相關文件**：`docs/edge-cv-tuning-log.md`（Phase 1–5 歷程）
**替代**：原 router 版 spec（已刪除）— 設計從「per-AOI 座標二選一 router」改為「單 ROI 內空間分權 fusion」

## 問題陳述

AOI 座標邊緣檢測（`_inspect_side side="aoi_edge"` / `_inspect_roi_patchcore`）目前 Phase 5 做了 CV / PatchCore 全域互斥切換，兩個 inspector 各有一個無法用 component-level 參數繞開的弱點：

| Inspector | 弱點 | 發作位置 |
|---|---|---|
| CV | `aoi_min_max_diff=20` 誤殺 faint 真 point defect（max_diff 15–19），見 Phase 3 Issue 1 | 全區；真 defect 與低對比紋理雜訊 max_diff 重疊，component-level metric 無法區分 |
| PatchCore（Phase 5） | backbone 有效感受野 ~30–50 px 被 panel 邊界黑 pad 污染，產生近邊假陽 | **只在 ROI 內的 polygon 邊界帶內側 ~40 px 發作** |

時序觀察（2026-04-17 → 2026-04-21）：
- 04-17 切到 CV + `min_max_diff=20` → 漏檢增加（Issue 1 發作）
- 04-21 切到 PatchCore → 漏檢改善（PC 語意判 faint 成功），但過檢變多（近邊感受野污染）

**關鍵觀察**：兩個弱點**空間不重疊於同一區域**。PatchCore 弱點**只在 ROI 內的 polygon 邊界帶**，CV 弱點（`min_max_diff=20`）雖全區都在，但在 AOI 邊緣樣本裡**真 defect 多為中遠區的 faint**。

因此做**單 ROI 內空間分權 fusion**：在同一個 AOI 座標的 512×512 ROI 裡，把 polygon 邊往 panel 內部延伸 `boundary_band_px`（預設 40）的「邊界帶」劃為 CV 管轄區，其餘 ROI 內部為 PatchCore 管轄區。兩個 inspector 同時跑同一個 ROI，各自只保留管轄區內的 defect 輸出，聯集後得到 fusion defect list。該 AOI 座標的 ROI **只有兩邊管轄區都乾淨才算 OK**。

## 目標

1. 減少 PatchCore 近邊假陽造成的過檢（PC 管轄區避開受污染的 polygon 邊界帶）
2. 保留 PatchCore 對中遠區 faint defect 的語意判別能力（不回到 Phase 3 的 `min_max_diff=20` 誤殺）
3. **決策邏輯必須在 UI 可視化**：現場人員能看到「這顆 defect 是哪個 inspector 在哪個管轄區抓到的」，訊號用 polygon 邊 + boundary band 疊層呈現
4. OMIT 灰塵屏蔽邏輯在 fusion 下統一套用、不分 source

## 非目標

- **不解決 Phase 3 faint defect 漏檢殘留**：CV 在 band 內仍受 `min_max_diff=20` 限制。若真 faint defect 恰好落在 band 內 CV 管轄區、max_diff 15–19 → 仍會被 CV 擋。先觀察線上發生率，日後若需要再做「band 內 CV 獨立放寬 min_max_diff」。
- **不改 PatchCore ROI 策略 / pad 填充方式**：保留 Phase 5 的黑 pad + fg_mask 遮罩 anomaly_map 行為；fusion 是獨立於 ROI 策略的輸出層空間分權。
- **不動中央 tile / 四邊全掃 pipeline**：只動 AOI 座標邊緣路徑。
- **不改 Bomb 比對邏輯**：純座標 match 沿用。
- **不做 inspector 層 OR / AND 重疊投票**：CV / PC 各自管自己的空間區，不會在同一 pixel 兩邊都判（空間分權=exclusive authority per pixel）。
- **不改 OMIT 內部判定參數**（CLAHE kernel, Otsu, top-hat 等）：沿用既有 `check_dust_or_scratch_feature` + per-region IoU/Coverage 邏輯。
- **不改 `/v3/record/<id>` v3 版頁面**：UI 改動只改 `/record/<id>` 預設版（`record_detail.html`）。

## 設計

### 核心邏輯

```python
# 擴增 inspector_mode 第三選項 "fusion"；"cv" / "patchcore" 行為與 Phase 5 完全一致
if inspector_mode == "fusion":
    defects = _inspect_roi_fusion(roi, fg_mask, panel_polygon, aoi_coord, ...)

def _inspect_roi_fusion(roi, fg_mask, panel_polygon, aoi_coord, ...):
    # 1. 計算 boundary band mask（fg_mask 內、距 polygon 邊 ≤ boundary_band_px 的 pixel）
    band_mask = compute_boundary_band_mask(
        roi_shape=roi.shape,
        roi_origin=(rx1, ry1),
        panel_polygon=panel_polygon,    # 若為 None → 退回 CV only
        band_px=config.boundary_band_px,
        fg_mask=fg_mask,
    )
    interior_mask = fg_mask & (~band_mask)

    # 2. CV 路徑：照常跑 inspect_roi 得 defect list；空間過濾到 band 內
    cv_defects_all, cv_stats = edge_inspector.inspect_roi(roi, ...)
    cv_defects_kept = [d for d in cv_defects_all if band_mask[d.center_y, d.center_x] > 0]
    for d in cv_defects_kept:
        d.source_inspector = "cv"
        d.d_edge_px = compute_defect_center_distance_to_polygon(d, panel_polygon)

    # 3. PatchCore 路徑：跑 _inspect_roi_patchcore 後從 pc_stats 拿 anomaly_map；
    #    把 band 內歸零、再 threshold+CC 抽 defect
    _, pc_stats = _inspect_roi_patchcore(roi, img_x, img_y, img_prefix, panel_polygon=..., return_raw=True)
    # 註：_inspect_roi_patchcore 目前已把 anomaly_map 塞進 pc_stats（見 capi_inference.py:3833 用法）
    # 本次新增 return_raw 參數（預設 False）：True 時跳過 defect 抽取、只回 (None, pc_stats including anomaly_map)
    # 沿用既存 pc_stats 欄位結構，avoid 新演算法
    pc_anomaly_map = pc_stats["anomaly_map"]
    pc_anomaly_map_interior = pc_anomaly_map.copy()
    pc_anomaly_map_interior[band_mask > 0] = 0.0
    pc_defects = extract_defects_from_anomaly_map(
        pc_anomaly_map_interior,
        threshold=pc_stats["threshold"],
        fg_mask=fg_mask,
        min_area=config.pc_min_area,
    )
    for d in pc_defects:
        d.source_inspector = "patchcore"
        d.d_edge_px = compute_defect_center_distance_to_polygon(d, panel_polygon)

    fusion_defects = cv_defects_kept + pc_defects

    # 4. OMIT 灰塵屏蔽 — 不分 source 統一套
    fusion_defects = apply_omit_dust_filter(
        fusion_defects, omit_image, omit_overexposed,
    )
    # apply_omit_dust_filter 對每顆 defect：
    #   - omit_crop = OMIT[d.bbox]
    #   - 沿用 check_dust_or_scratch_feature(omit_crop) + per-region IoU/Coverage
    #   - 命中 dust → d.is_suspected_dust_or_scratch = True
    #   - OMIT 過曝 → defect 保留 + dust_detail_text="OMIT_OVEREXPOSED → REAL_NG"（沿用既有 tile 路徑邏輯）
    #   - OMIT 缺失 → skip dust check

    return fusion_defects, {
        "band_mask": band_mask, "interior_mask": interior_mask,
        "pc_anomaly_map": pc_anomaly_map, "pc_anomaly_map_interior": pc_anomaly_map_interior,
        "cv_stats": cv_stats, "pc_stats": pc_stats,
    }
```

**聚合規則**：
- 該 AOI 座標 ROI 的最終 NG 判定 = fusion_defects 中「非 is_suspected_dust_or_scratch 的 defect」是否為空
- 空 → ROI OK（「CV band 乾淨 AND PC interior 乾淨」同時成立才到這）
- 非空 → ROI NG
- 沿用 `capi_inference.py:1829` 既有 `is_suspected_dust_or_scratch` 過濾規則，**聚合層不用改**

### Boundary Band Mask 計算

```python
def compute_boundary_band_mask(roi_shape, roi_origin, panel_polygon, band_px, fg_mask):
    h, w = roi_shape[:2]
    roi_ox, roi_oy = roi_origin

    # 1. 把 panel polygon 頂點轉到 ROI 座標
    poly_roi = np.array(
        [(int(x - roi_ox), int(y - roi_oy)) for x, y in panel_polygon],
        dtype=np.int32,
    )

    # 2. 畫 polygon edge（closed, 1px 寬線）到 ROI 大小空白圖
    edge_img = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(edge_img, [poly_roi], isClosed=True, color=255, thickness=1)

    # 3. ROI 內無 polygon edge（deep interior 情況）→ 回空 band
    if not edge_img.any():
        return np.zeros((h, w), dtype=np.uint8)

    # 4. Dilate edge 到 band_px 寬度
    kernel_size = 2 * band_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    band_near_edge = cv2.dilate(edge_img, kernel)

    # 5. 限定在 fg_mask 內（band 只在 panel 內側；外側是 panel 外、fg_mask=0 天然排除）
    return cv2.bitwise_and(band_near_edge, fg_mask)
```

**語意**：Band = 「ROI 內所有離 polygon 邊 ≤ band_px 的像素」且「在 fg_mask 內」。

- AOI 座標 deep interior（polygon 邊未進入 ROI）→ edge_img 全零 → band 空 → interior_mask = fg_mask（PC 管全部；CV 管轄區空、其 defect 全被過濾 = 變相 PC only）
- AOI 座標近邊 → band 是 polygon 沿線 panel 內側 40 px 寬帶 → CV 管帶內、PC 管帶外
- AOI 座標貼邊 / 角落 → band 覆蓋 ROI 多數 fg 區 → CV 管大部分、PC 管剩餘少數中心區

### 輔助函式語意說明

**`compute_defect_center_distance_to_polygon(defect, panel_polygon)`**
直接呼叫 `cv2.pointPolygonTest(polygon_np, (cx, cy), measureDist=True)`，回傳值 clip 到 ≥ 0（polygon 外視為 d=0）。`(cx, cy)` 為 defect center（panel 座標系）。

**`extract_defects_from_anomaly_map(anomaly_map, threshold, fg_mask, min_area)`**
**不是全新演算法**——直接提煉 `_inspect_roi_patchcore` 既有後處理邏輯成獨立函式：
- `mask = (anomaly_map >= threshold) & (fg_mask > 0)`
- CC + `_anomaly_max_cc_area`（既存 helper）取 CC
- 面積過濾後，每個 CC 產一個 defect（bbox / center / area / score = 該 CC 內 anomaly_map 最大值）
Fusion path 用「已對 band 歸零的 anomaly_map_interior」呼叫此函式即可，其餘 PC 後處理行為（threshold 來源 / area 過濾 / score 計算）與 Phase 5 一致，避免兩套邏輯分歧。

**每顆 defect 的 mask（OMIT per-region IoU/Coverage 用）**
- **CV defect**：取 ROI 內 `diff >= threshold` 的 binary mask 裁到該 defect 的 bbox 範圍；等同該 CC 的 thresholded 輪廓 mask
- **PC defect**：取 `anomaly_map_interior >= threshold` 的 binary mask 裁到該 defect 的 bbox 範圍
兩者都是 bbox 範圍內的 binary mask，跟 tile 路徑 `_dust_check_one` 傳給 `compute_dust_heatmap_iou` 的 `heatmap_binary` 結構一致，可直接餵進既有 function 不用改簽名。

### Config 新增 / 修改

| Key | 型別 | 預設 | 說明 |
|---|---|---|---|
| `aoi_edge_inspector` | string | `"cv"` | **擴增第三選項** `"fusion"`；`"cv"` / `"patchcore"` 行為與 Phase 5 完全一致（fusion 不啟用） |
| `aoi_edge_boundary_band_px` | int | `40` | Fusion 下 boundary band 寬度（px）；只在 `inspector="fusion"` 時生效。基於 PatchCore backbone 有效感受野 ~30–50 px 取保守中值 |

**預設 inspector 維持 `"cv"`**（線上遷移不變更既有行為；使用者 DB 手動設 `"patchcore"` 亦保持）。

**Hot-reload**：兩個 key 都納入 `capi_web.py` 的 hot-reload 觸發條件，`/settings` 改完即時生效、不用重啟。

**`cv_edge_dust_filter_enabled` 在 fusion 下強制關閉**：
- `/settings` AOI tab 當 `inspector="fusion"` 時，`cv_edge_dust_filter_enabled` 欄位 UI disable 並顯示提示「Fusion 模式使用統一 OMIT 屏蔽，CV dust filter 不啟用」
- 後端 `_inspect_roi_fusion` 即使 `cv_edge_dust_filter_enabled=True` 也不呼叫 CV 內部 dust filter；唯一 dust 判定來源是 fusion 後的 `apply_omit_dust_filter`
- 切回 `cv` / `patchcore` 模式時 `cv_edge_dust_filter_enabled` 恢復可編輯

### OMIT 屏蔽整合

**統一邏輯**（fusion 模式專用新 helper `apply_omit_dust_filter`）：

```python
def apply_omit_dust_filter(defects, omit_image, omit_overexposed, config):
    """
    對 fusion_defects 逐顆做 OMIT 灰塵檢查，沿用 tile 路徑的 check_dust_or_scratch_feature
    + per-region IoU/Coverage 邏輯，不分 source_inspector。
    """
    if omit_image is None:
        # OMIT 缺失 → skip dust check，defect 全保留（與 tile 路徑 _dust_check_one 一致）
        return defects

    if omit_overexposed:
        # OMIT 過曝 → 不判 dust，defect 全保留，加 debug 文字（沿用 line 3981 行為）
        for d in defects:
            d.dust_detail_text = "OMIT_OVEREXPOSED → REAL_NG"
        return defects

    for d in defects:
        bx, by, bw, bh = d.bbox
        oh, ow = omit_image.shape[:2]
        if bx >= ow or by >= oh:
            continue
        x2 = min(bx + bw, ow)
        y2 = min(by + bh, oh)
        omit_crop = omit_image[by:y2, bx:x2]

        is_dust, dust_mask, bright_ratio, detail = inferencer.check_dust_or_scratch_feature(omit_crop)
        d.dust_bright_ratio = bright_ratio
        d.dust_detail_text = detail

        if is_dust and dust_mask is not None:
            # 沿用 per-region IoU/Coverage 驗證（既有邏輯）
            # defect 的 mask 用 bbox 區 thresholded diff / anomaly 取得
            defect_mask = _get_defect_mask_within_bbox(d)
            iou_or_cov = compute_dust_heatmap_iou(dust_mask, defect_mask, metric=config.dust_heatmap_metric)
            if iou_or_cov >= config.dust_iou_threshold:
                d.is_suspected_dust_or_scratch = True
                d.dust_mask = dust_mask
    return defects
```

**整合要點**：
- 每個 defect 獨立判定，不分 `source_inspector="cv"` / `"patchcore"`
- 既有 `check_dust_or_scratch_feature`（CLAHE + Otsu + area filter）與 `compute_dust_heatmap_iou` 直接複用
- `dust_iou_threshold` / `dust_heatmap_metric` 沿用既有 config 不改
- 輸出欄位（`dust_mask`, `dust_bright_ratio`, `dust_detail_text`, `is_suspected_dust_or_scratch`）跟 tile 路徑一致 → DB / UI 不用新增 dust 欄位

**既有 `is_suspected_dust_or_scratch` 聚合過濾（`capi_inference.py:1829`）維持不動**：
```python
real_edge_defects = [ed for ed in edge_defects if not ed.is_suspected_dust_or_scratch]
```
→ Fusion 產出的 defect 走同一條路。

### Fallback / 邊界情況

| 情境 | 處理 |
|---|---|
| `inspector="cv"` 或 `"patchcore"` | 完全忽略 fusion 邏輯，走該 inspector（向下相容、Phase 5 行為不變） |
| `inspector="fusion"` + `panel_polygon=None`（polygon 偵測失敗） | **退回 CV only**（無 polygon → 無法算 band；PC 在 polygon 缺失下 fg_mask 精確度不足）；defects 都標 `source_inspector="cv"`, `fusion_fallback_reason="polygon_unavailable"` |
| `inspector="fusion"` + AOI 座標 deep interior（ROI 內無 polygon edge） | band 空；CV defects 全部空間過濾 → 丟棄；PC 管全 ROI；**流程仍跑完整 fusion**（統一 pipeline 原則，不做 skip 優化） |
| `inspector="fusion"` + AOI 座標貼邊（d_edge≈0） | band 覆蓋 ROI 多數 fg；CV 管大部分；PC 管少數中心區 |
| `boundary_band_px=0` | band 永遠空 → CV defect 全部被過濾 → 等同 `inspector="patchcore"` |
| `boundary_band_px ≥ 256` | band 可能覆蓋整個 ROI fg_mask → PC 管轄區幾乎為空 → 等同 `inspector="cv"` |
| OMIT 缺失（`omit_image is None`） | fusion_defects 全保留，不做 dust check（既有 tile 路徑 fallback） |
| OMIT 過曝 | fusion_defects 全保留 + `dust_detail_text="OMIT_OVEREXPOSED → REAL_NG"`（既有 tile 路徑邏輯） |

### DB Schema 變更

`edge_defect_results` 表新增欄位：

| 欄位 | 型別 | 說明 |
|---|---|---|
| `source_inspector` | TEXT | defect 來源 inspector：`"cv"` / `"patchcore"` / `NULL`（非 fusion 模式紀錄或 CV OK row） |
| `d_edge_px` | REAL | defect center 到 panel polygon 邊的距離（px）；`NULL` for 非 fusion 紀錄或 polygon 不可用 |
| `fusion_fallback_reason` | TEXT | fusion 模式下退回 fallback 的原因（例：`polygon_unavailable`）；`NULL` 為正常 fusion |

**`inspector_mode` vs `source_inspector` 語意分層**（兩個欄位並存，不重複）：

| 欄位 | 層級 | 取值 | 用途 |
|---|---|---|---|
| `inspector_mode`（Phase 5 既有） | ROI 層（該顆 AOI 座標用什麼模式跑） | `"cv"` / `"patchcore"` / `"fusion"` | 統計時 group by：這個 AOI 座標當時跑哪個模式 |
| `source_inspector`（本 spec 新增） | Defect 層（個別 defect 來自哪個 inspector） | `"cv"` / `"patchcore"` / `NULL` | fusion 模式下區分 defect 來自 band（CV）還是 interior（PC）；非 fusion 模式 NULL |

範例：fusion 模式下一顆 AOI 座標 ROI 抓到 2 個 defect（1 個 CV band + 1 個 PC interior），DB 會寫入 2 筆 edge_defect_results，兩筆 `inspector_mode="fusion"`，其中一筆 `source_inspector="cv"`、另一筆 `source_inspector="patchcore"`。

Migration 策略：`ALTER TABLE ADD COLUMN`，既有列預設 NULL，向下相容。

### UI 可視化

#### `/debug` 角落測試頁

1. Inspector radio 擴三顆：`cv` / `patchcore` / **`fusion`**
2. 選 `fusion` 時：
   - 下方出現 `boundary_band_px` 數字輸入框（預設讀 DB，本次試跑可 override）
   - CV 專屬參數（threshold / min_area / solidity / min_max_diff / line detection）與 PC 專屬參數全部可編輯（fusion 同時用到兩邊）
   - `cv_edge_dust_filter_enabled` 欄位 **disable** 並顯示提示
3. 結果區新增 **Fusion 分權區塊**：
   ```
   Boundary band width: 40 px
   Band 內 CV defect: 2 顆
   Interior PC defect: 1 顆
   OMIT 屏蔽: 1 顆灰塵（已排除）
   → 最終 ROI NG（2 顆真 defect）
   ```
4. **Panel 預覽圖疊畫**（canvas overlay，可 toggle 顯示/隱藏）：
   - polygon 邊界：青色實線
   - boundary band：青色半透明填充（標示 CV 管轄區）
   - interior：無填色（PC 管轄區）
   - CV defect bbox：🟠 橘色實線 + 左上角 `[CV]` 標籤
   - PC defect bbox：🟣 洋紅實線 + 左上角 `[PC]` 標籤
   - Dust 屏蔽 defect：虛線 + 灰階 + `[DUST]` 標籤
5. Defect 明細表格每列顯示 `Source` / `d_edge` / `OMIT 狀態` 欄

#### `/record/<id>`（預設版，**非** `/v3/record/<id>`）邊緣表格

每筆 AOI 邊緣紀錄每 defect 列新增 3 欄：

| 欄位 | 內容 |
|---|---|
| **Source** | 🟠 `CV (band)` 徽章 / 🟣 `PC (interior)` 徽章 |
| **d_edge** | `12.3 px`（defect center 到 polygon 邊距離；fusion 模式才有值、其餘顯示 `—`） |
| **OMIT 狀態** | 🟢 `OMIT 比對：非灰塵` / ⚫ `OMIT 判灰塵（已屏蔽）` / 🟡 `OMIT 過曝 → REAL_NG` / ⚪ `OMIT 缺失` |

AOI 座標標題列（一顆 AOI 座標可能有多 defect 列）新增 `Inspector` 欄顯示 `fusion` / `cv` / `patchcore` 總模式。

**熱力圖 / Defect Highlight 圖片**（production record 存檔）：
- 每 defect bbox 顏色照 Source 區分（CV 橘 / PC 洋紅 / DUST 虛線灰）
- 圖片左上角加 `🎯 Fusion` 標籤（fusion 模式時）+ band overlay（青色半透明填充）
- 這是現場人員看的主要產出，要 10 秒內看懂「哪裡 NG、是哪個 inspector 抓的、有沒有灰塵屏蔽」

**不改** `record_detail_v3.html`。

#### `/settings` 🎯 AOI tab

1. `aoi_edge_inspector` radio 擴三顆（加 `fusion`）
2. 選 `fusion` 時 `aoi_edge_boundary_band_px` 欄位啟用、可編輯
3. 選 `fusion` 時 `cv_edge_dust_filter_enabled` 欄位 disable + 提示文字
4. 欄位旁放小型 SVG 示意圖：polygon + boundary band（青色半透明）+ interior + 示例 CV defect（橘）/ PC defect（洋紅），標註 band_px 寬度

### 實作範圍

| 檔案 | 改動摘要 |
|---|---|
| `capi_edge_cv.py` | `EdgeDefect` 加 `source_inspector` / `d_edge_px` / `fusion_fallback_reason` 欄位；`EdgeInspectionConfig` 加 `boundary_band_px` + `from_db_params` 對應；新 helper `compute_boundary_band_mask` |
| `capi_inference.py` | 新 method `_inspect_roi_fusion`（CV + PC 跑 ROI、空間分權、OMIT 屏蔽）；AOI edge loop（L3800 附近）加 `inspector_mode=="fusion"` 分支；新 helper `apply_omit_dust_filter`（fusion 用）+ `extract_defects_from_anomaly_map`（提煉自 `_inspect_roi_patchcore` 既有後處理邏輯）；`_inspect_roi_patchcore` 加 `return_raw: bool = False` 參數（True 時跳過 defect 抽取、只回 pc_stats 含 anomaly_map，供 fusion 呼叫） |
| `capi_database.py` | migration 加 `source_inspector` / `d_edge_px` / `fusion_fallback_reason` 欄位；seed `aoi_edge_boundary_band_px=40`；`aoi_edge_inspector` 寫入值擴 `"fusion"` |
| `capi_server.py` | dict 轉換補 3 個新欄位（與 Phase 5 同 pattern） |
| `capi_heatmap.py` | defect highlight 渲染：color code by source（CV 橘 / PC 洋紅 / DUST 虛線灰）+ `[CV]` / `[PC]` / `[DUST]` 角標；fusion 模式時畫 boundary band 疊層（青色半透明）；新 helper 供 debug + production 共用 |
| `capi_web.py` | `/settings` 表單欄位 + fusion radio 啟用後 `cv_edge_dust_filter_enabled` disable 邏輯；`/debug` API 接 `inspector="fusion"` + `boundary_band_px`；hot-reload 觸發條件擴充；新增 API 回傳 band overlay contour 給前端 canvas 畫 |
| `templates/debug_inference.html` | 角落測試區加 fusion radio + boundary_band_px 欄位 + band overlay toggle + Fusion 分權結果區塊 + defect 明細新增 Source / d_edge / OMIT 欄 |
| `templates/record_detail.html` | 邊緣表格每 defect 列加 Source / d_edge / OMIT 欄；AOI 座標標題列加 Inspector 欄（`record_detail_v3.html` **不改**） |
| `templates/settings.html` | `AOI_COORD_PARAMS` 加 `aoi_edge_boundary_band_px` + radio 擴 fusion 選項 + SVG 示意圖 + fusion 時 `cv_edge_dust_filter_enabled` disable JS |
| `tests/test_aoi_edge_fusion.py` | **新檔**：fusion 空間分權分支、band mask 計算、OMIT 整合、fallback、各 inspector 模式切換（見測試計劃） |
| `tests/test_aoi_coord_inference.py` | 擴既有測試涵蓋 fusion 模式 |
| `docs/edge-cv-tuning-log.md` | 加 Phase 6 section 記錄 fusion 設計、trade-off、OMIT 整合 |

### 測試計劃

新單元測試 `tests/test_aoi_edge_fusion.py`，涵蓋：

**Band mask 幾何**
1. `test_band_mask_deep_interior`：AOI 在 panel 中心（d_edge > 256+band），ROI 內無 polygon edge → band 空
2. `test_band_mask_near_edge`：AOI 距 polygon 100 px，band=40 → band 為 polygon 沿線帶，面積 ≈ perimeter × 40
3. `test_band_mask_on_edge`：AOI 在 polygon 上（d_edge=0），band 覆蓋 ROI 多數 fg
4. `test_band_mask_band_px_zero`：`boundary_band_px=0` → band 永遠空
5. `test_band_mask_polygon_none`：panel_polygon=None → 觸發 fallback，不進入 band 計算

**空間分權**
6. `test_fusion_cv_defect_in_band_kept`：模擬 CV 在 band 內抓到 defect → `source_inspector="cv"` 保留
7. `test_fusion_cv_defect_in_interior_dropped`：模擬 CV 在 interior 抓到 defect → 空間過濾丟棄
8. `test_fusion_pc_defect_in_interior_kept`：模擬 PC anomaly 在 interior → `source_inspector="patchcore"` 保留
9. `test_fusion_pc_defect_in_band_dropped`：模擬 PC anomaly 在 band（感受野污染假陽）→ band mask 歸零 → 不產 defect

**OMIT 屏蔽**
10. `test_fusion_omit_missing_all_kept`：omit_image=None → fusion_defects 全保留
11. `test_fusion_omit_overexposed_all_kept`：omit_overexposed=True → fusion_defects 全保留 + `dust_detail_text` 填
12. `test_fusion_cv_defect_omit_dust_filtered`：CV band defect 命中 OMIT dust → `is_suspected_dust_or_scratch=True`
13. `test_fusion_pc_defect_omit_dust_filtered`：PC interior defect 命中 OMIT dust → `is_suspected_dust_or_scratch=True`

**Inspector 模式**
14. `test_inspector_mode_cv_skips_fusion`：`inspector="cv"` 模式不跑 fusion，沿用 Phase 5 CV 行為
15. `test_inspector_mode_patchcore_skips_fusion`：`inspector="patchcore"` 模式不跑 fusion
16. `test_inspector_mode_fusion_polygon_unavailable_fallback`：fusion + polygon=None → fallback CV only，`fusion_fallback_reason="polygon_unavailable"`

**向下相容**
17. `test_existing_cv_mode_regression`：既有 CV mode 測試全綠（不受本次改動影響）
18. `test_existing_patchcore_mode_regression`：既有 PC mode 測試全綠

**聚合**
19. `test_fusion_ng_aggregation_cv_only`：fusion 只有 CV band defect → ROI NG，`is_suspected_dust_or_scratch=False`
20. `test_fusion_ng_aggregation_pc_only`：fusion 只有 PC interior defect → ROI NG
21. `test_fusion_ok_all_dust`：fusion 有 defect 但全部 `is_suspected_dust_or_scratch=True` → ROI OK（等同 is_dust_only）

**既有 regression 全綠**：
- `tests/test_aoi_coord_inference.py`
- `tests/test_aoi_edge_patchcore.py`（Phase 5 PC inspector 8 項）
- `tests/test_cv_edge.py`
- log.txt / log2 / log3 / log4 對照案例

### 風險

| 風險 | 機率 | 影響 | 緩解 |
|---|---|---|---|
| `boundary_band_px=40` 太小 → PC 在靠近 band 邊緣處仍產假陽 | 中 | 過檢殘留 | 熱更新調高（50 / 60）；每顆過檢 defect 有 `d_edge_px` → 可統計分布決定最佳 band 寬度 |
| `boundary_band_px=40` 太大 → PC 管轄區縮太多、中遠區 faint defect 召回下降 | 低 | 漏檢增加 | 熱更新調低（30 / 25）；PC 是本版主力，不建議 band 超過 60 |
| 每顆 AOI 座標從單 inspector 變雙 inspector，算力 ~2x | 高（一定發生） | 延遲增加 | CV 推論本身輕量（~幾十 ms / ROI）；PC 推論本來就有；整體 per-panel 延遲增加可忽略。實測 latency 後若不可接受再做 per-coord skip 優化 |
| `compute_boundary_band_mask` 對破碎 / 非凸 polygon 行為不穩 | 低 | band 過大或過小 | polygon 來源已是 `_find_raw_object_bounds` 主要 contour（單一凸近似）；異常時 band 計算失敗 → 退回 CV only（polygon_unavailable fallback） |
| OMIT 屏蔽對 CV band defect 敏感度差異未知（CLAHE + Otsu 原本調在 tile 中央區） | 中 | band 內真 defect 被誤屏蔽或 dust 未屏蔽 | 觀察上線 1–2 週，若 band 內 OMIT 判定偏差大再考慮 band 專屬 OMIT 參數（Phase 7） |
| 畫面視覺元素增多（band overlay + CV/PC 顏色 + DUST 虛線）導致熱力圖擁擠 | 低 | 現場辨讀負擔 | 可 toggle 顯示/隱藏 band overlay；defect 顏色語義明確、已在 debug 頁先驗證 |

## 已知開放問題（不在本 spec scope）

- **Issue 1 延續（band 內 CV 仍受 `min_max_diff=20` 限制）**：若 faint defect 恰落在 CV 管轄區（band 內）且 max_diff 15–19 → 仍漏檢。觀察線上發生率後，若顯著可做「band 內 CV 獨立放寬 min_max_diff」作為 Phase 7 議題。
- **band 內 OMIT 判定差異**：如上「風險」列項，觀察後再決定是否做 band 專屬 OMIT 參數。
- **Fusion 算力優化**：若實測 per-panel latency 不可接受，可評估「ROI 內無 polygon edge 時 skip CV」（違反本次「統一 pipeline」原則；需明確 trade-off 討論）。

## 檢核點

交付完成條件：

- [ ] 所有新 unit test 通過（21 項 fusion 測試涵蓋 band geometry / 空間分權 / OMIT / fallback / 聚合）
- [ ] 既有 AOI / CV / PC edge regression 全綠
- [ ] `/debug` 角落測試 fusion 模式能呈現 band overlay + 分權結果區塊 + CV / PC / DUST 顏色區分
- [ ] `/record/<id>`（預設版）邊緣表格顯示 Source / d_edge / OMIT 欄
- [ ] Production heatmap 圖片 CV / PC / DUST bbox 顏色區分 + band overlay，現場 10 秒可辨讀
- [ ] `/settings` 切 fusion 後 `boundary_band_px` 可熱更新生效；`cv_edge_dust_filter_enabled` 自動 disable
- [ ] DB migration 向下相容既有列（舊列 `source_inspector=NULL`, `d_edge_px=NULL`, `fusion_fallback_reason=NULL`）
- [ ] `docs/edge-cv-tuning-log.md` Phase 6 section 寫完，含 fusion 設計 / trade-off / OMIT 整合 / 已知 open issue
