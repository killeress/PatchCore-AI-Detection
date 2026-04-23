# AOI 座標邊緣 CV 檢測調校記錄

起始日期：2026-04-17（持續更新）

## 背景

AOI 機台會把疑似 defect 座標報給 CAPI 系統，走 `_inspect_side` (side="aoi_edge") 做 CV 二次驗證。早期演算法為 `GaussianBlur → fg_median 常數填充 → medianBlur → absdiff → threshold → CC → solidity 過濾`，在多種 panel 類型上出現過檢/漏檢，需要持續調參與改演算法。本文件記錄每次改動的動機、trade-off、測試案例與已知 open issue，便於未來 session 快速回到狀態。

## Production 生效路徑

```
capi_server.py:772  →  EdgeInspectionConfig.from_db_params(db_dict)
                    →  update_edge_config
                    →  self.edge_inspector = CVEdgeInspector(config)

每筆 request:
  capi_inference.py:3518  →  inspect_roi(...)
                          →  _inspect_side (side="aoi_edge" path)
```

- 參數優先級：DB `config_params` 的 `cv_edge_aoi_*` > `EdgeInspectionConfig` default
- `/settings` 改參數後 `capi_web.py:3008-3014` 會 hot-reload（`update_edge_config` 重建 inspector），**不用重啟 server**
- `init_config_from_yaml` (`capi_database.py:2042`) 在啟動時 seed 新 key，既有 key 不覆寫

## 演算法演進時間軸

### Phase 1 — Inpaint 填充取代 fg_median

- **問題**：polygon 邊界切進 ROI（fg_mask coverage < 100%）時，`fg_median` 常數填充 + `medianBlur(65)` 會在 fg 邊界內側產生約 30 px 寬的長條 diff streak。ROI 左右平移 5 px 讓邊界離開 ROI 時 streak 就消失（user 實證）。
- **根因**：填充值是全域中位數，跟真實邊緣漸暗亮度不匹配；medianBlur kernel=65 取樣半徑 ~32 px 會把常數填充值擴散到 fg 邊界內側，bg 估計被拉偏。
- **解法**：`cv2.inpaint(blurred, non_fg_mask, 3, cv2.INPAINT_NS)`。NS 擴散填充使 bg 估計在邊界內側跟真實局部亮度一致。
- **Code**：`capi_edge_cv.py` `inpaint_non_fg_region()` helper；`_inspect_side`、`inspect_roi` 統計重算、`compute_fg_aware_diff`、`capi_web.py` debug page 重建 都統一用此。
- **驗證**：人工模擬 9216 px 偽影 → 0 px；實機 x=5765 case 8132 px streak → 43 px（幾乎消失）。
- **Production 狀態**：永久 ON，無 toggle。

### Phase 2 — Morphological Opening (kernel=3)

- **問題**：面板 sub-pixel / column pattern 偽影（log2.txt 11 個 1×12 垂直條紋）；threshold=3 時雜訊像素形成 bridges 把真 defect 吞進 134K px 巨塊 component。
- **解法**：二值化後 CC 前做 `cv2.MORPH_OPEN` 3×3。1-px 寬結構被 erode 歸零，雜訊橋被切斷，compact 真 defect (area ≥ 9) 完整保留。
- **Config**：`aoi_morph_open_kernel: int = 3` (0=停用)
- **Trade-off**：會殺掉 1-px 寬的真實細線（由 Phase 4 的 line detection path 補償）
- **測試驗證**：log2.txt 原本 13 個（含 134K 巨塊 + 11 條紋）→ 3 個 (只剩真黑點附近兩小塊 + 真黑點 #3 max_diff=51)。

### Phase 3 — Max Diff 下限 (min_max_diff=20)

- **問題**：面板紋理（織紋）在 threshold=5 下會產生 max_diff 9-18 的 low-contrast component（log3.txt 27 個）。area / solidity / shape 都區分不出。
- **解法**：`component.max_diff < aoi_min_max_diff` 直接過濾。放在 max_diff 計算後、solidity 之前（便宜的先過濾）。
- **Config**：`aoi_min_max_diff: int = 20` (0=停用)
- **Trade-off（重要）**：**會誤殺 max_diff 15-19 的 faint 真 point defect**。實例：
  - x=3008 ROI 一顆肉眼可見黑點 max_diff=17 → 被 20 擋 → 判 CV OK (Shape filtered)
  - 2026-04-17 open issue，詳見下方「已知問題 1」
- **Production 狀態**：預設 20（2026-04-17 起）

### Phase 4 — 薄線偵測（投影法）

- **問題**：1-px 寬真實線性 defect（log4.txt 垂直虛線 max_diff=14）在 binary mask 呈現**斷點狀**，morph_open 3×3 全部殺光；單個 fragment 又 < min_area=20。
- **解法**：二值化後 morph_open 前先跑**獨立的線偵測 path**：
  1. 1D 方向 close（kernel 1×5 垂直 / 5×1 水平）橋接虛線斷點
  2. Column / row projection 找活化像素數 ≥ `aoi_line_min_length` 的窄帶
  3. 限制寬度 ≤ `aoi_line_max_width` 排除一般 compact component
  4. 產生 EdgeDefect **旁路 min_area / solidity / min_max_diff**（避免 faint 線再被 Phase 3 擋）
- **Config**：`aoi_line_min_length: int = 30`、`aoi_line_max_width: int = 3`
- **Code**：`_detect_thin_lines()` + `_group_true_runs()` in `capi_edge_cv.py`
- **驗證**：模擬測試：垂直虛線 (30 點, 間距 5) 被抓；1×12 雜訊條紋 (< 30) 被擋；低對比紋理塊 (max_diff=6) 被 min_max_diff 擋。三路各司其職。

### Phase 7.1 — Fusion 結果 collapse 成每 AOI 1 筆代表 defect (2026-04-23)

- **動機**：Phase 7 上線後現場觀察：2 個 AOI 座標吐 8 行結果
  - CV `inspect_roi` 單 ROI 內可吐多個 component（band 內 3 個黑點 → 3 筆）
  - 相鄰 AOI 座標（距 67 px）的 512×512 ROI 重疊 → 同一個 defect 被兩顆 AOI 各抓一次
  - UI 8 行難解讀，與使用者「1 AOI 座標 = 1 NG」的心智模型不符
- **設計決策**：`_inspect_roi_fusion` 加 `collapse_to_representative=True` 參數（預設開）
  - 優先序：real NG > dust；real NG 內 PC > CV；CV 內 max area（tiebreak max_diff）
  - 全 dust：同優先序取代表，保留 dust 旗標
  - 空 list：回空（OK 由 caller 處理）
  - Debug endpoint 傳 `collapse_to_representative=False` 保留全細節供診斷
- **Trade-off**：
  - **(+)** 1 AOI 座標 = 1 result row，UI 清爽符合機台語意
  - **(+)** stats 保留 `pre_collapse_count` / `cv_band_count` / `pc_interior_count` 供 log / 診斷
  - **(−)** 遺失 band 內其他 component 明細（可在 debug 頁重跑看全部）
- **Code**：
  - `capi_inference.py::_inspect_roi_fusion` 末端加 collapse + stats 4 個診斷欄位
  - `capi_web.py::_handle_api_debug_edge_corner_fusion` 傳 `collapse_to_representative=False`
- **測試**：`tests/test_aoi_edge_pc_roi_shift.py::TestFusionCollapseToRepresentative` 4 項：
  - 多 CV → 取最大 area / CV+PC → PC wins / 空 → 空 / debug 模式保留全部

### Phase 7 — Fusion PC ROI 內移避開 polygon 邊 (2026-04-23)

- **動機**：Phase 6 上線觀察：fusion 雖在 scoring 層把 band 區從 PC 歸零，但 PC backbone
  的 feature extraction 仍跑在整張含 polygon edge 的 512×512 上，receptive field
  ~30-50 px 讓 interior 靠近 band 側的 feature map 殘留「偽邊緣」activation →
  d_edge 偏小的 PC INTERIOR 命中仍有過檢殘留。
- **設計決策**：
  - 把 **PC 跑的 ROI 從 AOI 中心往 panel 內側偏移**，使 PC ROI 距 polygon 邊 ≥ band_px
  - CV ROI 不變（仍居中於 AOI 座標），band 區覆蓋不變
  - 偏移方向 = AOI 到最近 polygon 邊的 inward normal（axis-aligned polygon 為水平或垂直）
  - 偏移量 = `max(0, band_px - (d_edge - roi_size/2))`，clamp 到 `[0, roi_size/2 - aoi_margin_px]`
  - **AOI margin 約束**：AOI 座標距 PC ROI 邊 ≥ `aoi_edge_aoi_margin_px` (預設 64 px)
  - **凹角 polygon fallback**：偏移後用 `verify_polygon_clear_of_pc_roi` 檢查，若其他邊
    仍在 PC ROI 內 < band_px → 退回 Phase 6 centered + band_mask 行為，標
    `pc_roi_fallback_reason="concave_polygon"`
  - **Config `aoi_edge_pc_roi_inward_shift_enabled=True`**：總開關，off 時走 Phase 6
  - **Defect center 強制 = AOI 座標**（Bomb match 一致性沿用 Phase 6）
- **Trade-off**：
  - **(+)** PC feature map 完全脫離 polygon 邊 discontinuity，進一步壓抑近邊過檢
  - **(+)** 算力不變（PC 仍跑 1×，只是輸入裁切位置換了；helper 純幾何計算）
  - **(+)** CV 覆蓋不變，band 區照跑 → 零漏網
  - **(−)** UI 需顯示 shift 量（record_detail 加 PC Shift 欄、heatmap badge 加 `[PC dx=+N]`）
  - **(−)** 凹角 polygon 會 fallback 回 Phase 6（fallback_reason 可追蹤）
- **Code**：
  - `capi_edge_cv.py::compute_pc_roi_offset` 純幾何 offset 計算
  - `capi_edge_cv.py::verify_polygon_clear_of_pc_roi` 偏移後驗證（凹角 fallback）
  - `capi_inference.py::_inspect_roi_fusion` PC 路徑改走 shifted origin（L3294 附近）
  - `capi_database.py` migration 加 `pc_roi_origin_x/y`、`pc_roi_shift_dx/dy`、
    `pc_roi_fallback_reason` 五欄 + seed `aoi_edge_pc_roi_inward_shift_enabled=True`、
    `aoi_edge_aoi_margin_px=64`
  - `capi_heatmap.py` fusion PC heatmap badge 擴充：`[PC dx=+192 dy=0]` / `[PC FB:...]`
  - `capi_server.py` to_dict 加 5 個新欄位
  - `capi_web.py` debug fusion endpoint 支援 shift / margin override + 回應含 shift 資訊
  - UI: `/settings` AOI tab 加 `aoi_edge_pc_roi_inward_shift_enabled` / `aoi_edge_aoi_margin_px`；
    `/record/<id>` edge 表格加 PC Shift 欄
- **測試**：`tests/test_aoi_edge_pc_roi_shift.py` 22 項：
  - `compute_pc_roi_offset` 12 項（deep interior / near 4 邊 / corner 取最近邊 / clamped /
    polygon None / 頂點<3 / AOI 外 / threshold boundary）
  - `verify_polygon_clear_of_pc_roi` 5 項（clean / polygon inside / too close / 凹角 / None）
  - `_inspect_roi_fusion` 整合 5 項（deep interior 無 shift / 近邊 shift / defect fields /
    disabled / fallback concave）
  - Phase 6 (17 項) + AOI coord (3 項) + CV edge (46 項) regression 全綠
- **Config 預設值表更新**:

  | Key | 預設 | 角色 |
  |---|---|---|
  | `aoi_edge_pc_roi_inward_shift_enabled` | **True** | Phase 7 新增；fusion 下 PC ROI 自動內移 |
  | `aoi_edge_aoi_margin_px` | **64** | Phase 7 新增；AOI 座標距 PC ROI 邊最小 margin |

### Phase 6 — CV+PatchCore 空間分權 Fusion Inspector (2026-04-22)

- **動機**：Phase 5 互斥切換實際上線後觀察到：
  - `inspector="cv"` + `min_max_diff=20` → faint 真黑點 (max_diff 15-19) 漏檢增加 (Issue 1)
  - `inspector="patchcore"` → faint 漏檢救回，但 backbone 感受野 (~30-50 px) 被 panel 邊界
    黑 pad 污染，AOI 座標貼邊時產生大量近邊假陽 (過檢)
  - 兩個弱點 **空間不重疊**：PC 弱點只在 polygon 邊內側 ~40 px，CV 弱點 (低對比過濾)
    在中遠區的 faint defect 才關鍵。
- **設計決策**：
  - 第三選項 `aoi_edge_inspector="fusion"`，新增 `aoi_edge_boundary_band_px=40` (px)
  - 同一顆 AOI 座標 ROI **同時跑 CV + PC**，依 boundary_band_mask 做空間分權：
    - CV 管「polygon 邊往 panel 內 boundary_band_px 寬帶」
    - PC 管「band 外的 interior」(把 anomaly_map 在 band 區歸零)
  - 聚合：兩管轄區都乾淨才 OK；任一找到非 dust defect 即 NG
  - **統一 pipeline 不加 skip 優化**：deep interior (ROI 內無 polygon edge) 仍跑完整 fusion
    流程 (CV defect 全被 band mask 過濾 = 變相 PC only)，避免條件分支複雜度
  - OMIT 灰塵屏蔽 fusion 後**統一**套一次 (`apply_omit_dust_filter`)，不分 source；
    沿用既存 `check_dust_or_scratch_feature` + per-region IoU/Coverage
  - **fusion 模式下 `cv_edge_dust_filter_enabled` 強制視為關閉** (UI disable + 後端忽略)
    避免雙層 dust filter 邏輯分歧
  - Polygon 失敗 → fallback CV only，標 `fusion_fallback_reason="polygon_unavailable"`
- **Trade-off**：
  - **(+)** 同時解 PC 近邊過檢 與 CV 中遠區 faint 漏檢
  - **(+)** UI 完全可解釋：每個 defect 帶 `source_inspector` (cv/patchcore) +
    `d_edge_px` (defect center 到 polygon 邊距離) + Source 徽章
  - **(−)** 算力 ~2x (CV+PC 都跑)，CV 部分輕量，per-panel 影響可忽略
  - **(−)** 殘留 Issue 1 在 band 內：若 faint defect 落在 band 內 CV 管轄區、max_diff 15-19
    仍會被 `min_max_diff=20` 擋；觀察線上發生率後再決定是否做 band 專屬 CV 放寬
- **Code**：
  - `capi_edge_cv.py::compute_boundary_band_mask` 幾何 helper
  - `capi_inference.py::_inspect_roi_fusion` 主流程 (L3163 附近)
  - `capi_inference.py::_apply_omit_dust_filter_to_edge_defects` OMIT 整合
  - `capi_inference.py::_inspect_roi_patchcore` 加 `return_raw=True` 參數
    (跳過 defect 抽取、回 anomaly_map 給 fusion)
  - AOI loop 加 `inspector_mode=="fusion"` 分支 (L4070 附近)
  - `capi_database.py` migration 加 `source_inspector` / `d_edge_px` /
    `fusion_fallback_reason` 三欄 + seed `aoi_edge_boundary_band_px=40`
  - `capi_heatmap.py` defect highlight 渲染：fusion 模式 PC 來源走 PC renderer，
    兩條路徑加 [CV] / [PC] 角標
  - `capi_web.py::_handle_api_debug_edge_corner_fusion` 新 endpoint
  - UI: `/settings` AOI tab radio 加 fusion；`/debug` 角落測試 radio 加 fusion；
    `/record/<id>` (預設版，不動 v3) 邊緣表格加 Source / d_edge / OMIT 三欄
- **測試**：`tests/test_aoi_edge_fusion.py` 17 項：
  - Boundary band geometry 5 項 (deep interior / near edge / on edge / band_px=0 / polygon=None)
  - `_inspect_roi_patchcore return_raw` 2 項
  - `apply_omit_dust_filter` 4 項 (omit None / overexposed / dust hit / clean)
  - `_inspect_roi_fusion` 6 項 (fallback / CV in/out band / PC in/out band / deep interior)
  - 既有 Phase 5 (8 項) + AOI coord (3 項) regression 全綠
- **Config 預設值表更新**:

  | Key | 預設 | 角色 |
  |---|---|---|
  | `aoi_edge_inspector` | `"cv"` | 改：`"cv"` / `"patchcore"` / **`"fusion"`** 三選項 |
  | `aoi_edge_boundary_band_px` | **40** | Phase 6 新增；fusion 下 CV 管轄帶寬度 |

### Phase 5 — PatchCore Inspector 切換 (2026-04-21)

- **動機**：Phase 3 的 `min_max_diff=20` 與 faint 黑點 max_diff 15-19 衝突（Issue 1），靠
  component-level CV metric 無法區分，且持續累積的 CV 參數維護負擔高。改以 PatchCore
  同時處理中央 tile 與 AOI 邊緣，利用模型對異常的語意學習直接繞開這個困境。
- **設計決策**：
  - 互斥切換（不做 OR / AND 串接），DB key `aoi_edge_inspector: "cv" | "patchcore"`
  - 只動 AOI 座標邊緣路徑；四邊全掃 / 中央 tile 路徑不動
  - ROI 策略 (a')：中心對齊 AOI 座標 + 黑 pad (panel 外本來就是黑色，非合成 OOD) +
    `tile.mask=fg_mask` 遮罩 anomaly_map (panel 外分數歸零，沿用 `predict_tile` 現有機制)
  - Inspector 建立的 TileInfo `is_*_edge=False` → 不觸發 `_apply_edge_margin` 衰減
  - threshold / PatchCore 後處理 / OMIT 灰塵 / Bomb 比對：全部沿用中央 tile pipeline
  - Bomb 比對邏輯不改 — 純座標 match (center=AOI 座標強制保留)
- **EdgeDefect 擴充**：`inspector_mode` / `patchcore_score` / `patchcore_threshold` /
  `patchcore_ok_reason` 四欄，向下相容 (CV path 預設值)
- **Config key**：`aoi_edge_inspector: str = "cv"` (default)；hot-reload 觸發條件擴充
- **UI**：
  - `/settings` → 🎯 AOI tab radio (cv / patchcore)
  - `/debug` → 📐 角落測試 top 加 inspector radio，PatchCore 時停用 CV 專屬參數；
    呼叫 `/api/debug/edge-inspect-corner` 帶 `inspector` 參數分派到 `_inspect_roi_patchcore`
  - `record_detail` 邊緣表格：根據 mode 分支顯示「score vs Thr」或「max_diff vs Thr」；
    PC OK 帶「分數未達/面積未達」原因
  - Heatmap：PatchCore path 走獨立渲染 (`_save_patchcore_edge_image`)，
    Panel 1 = Original ROI（panel 外用暗紅 + 斜線標示遭 mask），
    Panel 2 = Anomaly Heatmap Overlay，Panel 3 = OMIT ROI (如有)
- **Trade-off**：
  - **(+)** 省 min_max_diff 的 trade-off；faint 真 defect 由模型語意判定
  - **(+)** 統一中央 / 邊緣的異常判定邏輯，降低 CV 啟發式調校負擔
  - **(−)** ROI 含 panel 邊界，backbone 感受野 (~30-50 px) 會污染近邊 feature，
    可能產生不同 pattern 的假陽性；但 training 資料裡邊角 tile 也有相同現象，
    模型已學到大部分。實際效果待線上驗證
  - **(−)** GPU 多一次推論/AOI 座標。以 per-panel 3~10 個座標估算，影響可忽略
- **Code**：`capi_inference.py::_inspect_roi_patchcore`；AOI loop 分支在 L3656 附近；
  `capi_edge_cv.py` EdgeDefect / EdgeInspectionConfig 欄位；
  `capi_heatmap.py::_save_patchcore_edge_image`；
  `capi_database.py` migration + seed
- **測試**：`tests/test_aoi_edge_patchcore.py` 8 項 (ROI 建構 / fg_mask / EdgeDefect mapping /
  OK 原因 / 邊界外 / 前綴未載入 / 向下相容 / config 讀取)

## 目前的完整 Filter 順序（`_inspect_side` aoi_edge path）

```
roi → GaussianBlur(3) → blurred
blurred_for_bg = inpaint_non_fg_region(blurred, fg_mask)   [Phase 1]
bg = medianBlur(blurred_for_bg, 65)
diff = absdiff(blurred, bg)
diff[fg_mask==0] = 0

mask = threshold(diff, cfg.threshold)

# Line detection path (parallel, bypass later filters)        [Phase 4]
line_defects = _detect_thin_lines(mask, diff, offset_x, offset_y)

# Compact path
mask = morph_open(mask, kernel=aoi_morph_open_kernel)         [Phase 2]
components = CC(mask)
for component:
    if area < min_area: skip
    max_diff = max(diff within component)
    if max_diff < aoi_min_max_diff: skip                       [Phase 3]
    solidity = area / convex_hull_area
    if solidity < solidity_min: skip
    defects.append(EdgeDefect)

defects.extend(line_defects)
return defects
```

## Config 參數完整列表（AOI edge 相關）

| Key | Default | 角色 |
|---|---|---|
| `cv_edge_aoi_threshold` | 4 | 二值化閾值 |
| `cv_edge_aoi_min_area` | 40 | CC path min_area |
| `cv_edge_aoi_solidity_min` | 0.2 | Solidity 下限（擋 L 形偽影）|
| `cv_edge_aoi_polygon_erode_px` | 3 | polygon fg_mask 內縮 px（避開邊緣亮帶轉換區）|
| `cv_edge_aoi_morph_open_kernel` | **3** | Phase 2 opening kernel（0=停用）|
| `cv_edge_aoi_min_max_diff` | **20** | Phase 3 max_diff 下限（0=停用）|
| `cv_edge_aoi_line_min_length` | **30** | Phase 4 薄線最小長度（0=停用）|
| `cv_edge_aoi_line_max_width` | **3** | Phase 4 薄線最大寬度 |
| `aoi_edge_inspector` | **"cv"** | Phase 5 AOI 座標邊緣 inspector：`"cv"` / `"patchcore"` |

粗體 = 本次討論新增。Defaults 有三處需同步：
- dataclass `EdgeInspectionConfig` (`capi_edge_cv.py:116` 附近)
- DB seed `init_config_from_yaml` (`capi_database.py:2109-2116`)
- Debug 頁 fallback (`capi_web.py:1020-1028`)

## 測試案例對照（log.txt 樣本）

| Case | 症狀 | 解法 Phase | 關鍵特徵 |
|---|---|---|---|
| x=5765 (log) | 邊界 8132 px 長條偽影，ROI 左移 5 px 就消失 | 1 (inpaint) | polygon 切進 ROI, fg_coverage=99.1% |
| log2 | 134K 巨塊吞真黑點 + 11 條 1×12 垂直雜訊 | 1 + 2 | panel sub-pixel column pattern |
| log3 | 27 個低對比紋理 component 全 max_diff 9-18 | 3 (min_max_diff=20) | 機構邊緣 / 紋理漸暗群聚 |
| log4 | 中央 1-px 垂直虛線 max_diff=14 漏檢 | 4 (line detection) | faint 真線、morph_open 會殺掉 |

## 已知開放問題

### Issue 1：Faint point defect 漏檢 (max_diff 15-19)

- **案例**：2026-04-17 使用者用紅圈標出一顆肉眼可見黑點，max_diff=17, area=36，被 `min_max_diff=20` 擋
- **核心衝突**：這類真 defect 的 max_diff 跟 log3 紋理雜訊（max_diff 9-18）幾乎完全重疊，**單靠 component-level metrics 無法區分**
- **候選方案**（尚未實作）：
  - **(A) 全域降 min_max_diff 到 15**：log3 重新開 3-5 個雜訊 component
  - **(B) Adaptive filter（推薦）**：若 ROI 內 candidate 總數 < N (例 5) → 套寬鬆 min_max_diff=12；若 ≥ N → 套嚴格 20。利用「真 defect 通常孤立 vs 紋理雜訊通常群聚」的 signature
  - **(C) 接受漏檢**：承認 min_max_diff=20 下 max_diff<20 是偵測極限
- **使用者傾向**：這類黑點算 defect（明確說「是defect」），但方案未拍板

### Issue 2：`CV OK (Shape filtered)` 標籤誤導

- **位置**：`capi_heatmap.py:867-880`
- **問題**：原本「Shape filtered」只指 solidity 過濾；新加的 min_max_diff / morph_open / line_detect 過濾也全部落到這個 catch-all 標籤，使用者會以為是 solidity 但其實是 min_max_diff 擋的
- **已修 (2026-04-21)**：
  - `EdgeDefect` 加 `min_max_diff_used` 欄位 (`capi_edge_cv.py:98`)
  - `inspect_roi` 的 `roi_stats` 帶出 `min_max_diff` (`capi_edge_cv.py:617+626`)
  - `capi_inference.py:3869+3889` 建 merged/ok_defect 時填入
  - `capi_server.py:652` dict 轉換補欄位；`capi_database.py` schema + 兩處 INSERT 同步補
  - `capi_heatmap.py:875` 判定推斷加 `elif min_max_diff_used > 0 and max_diff < min_max_diff_used: cv_ok_reason = "Diff<MinMaxDiff({min_max_diff_used})"`
  - `templates/record_detail.html:514` 同步加「低對比過濾 (Diff<N)」分支

## 除錯/驗證慣用工具

| 工具 | 用途 |
|---|---|
| `/debug` 頁 → 📐 角落測試 | 重現 production `inspect_roi`，UI 可調所有 aoi_edge 參數 |
| 「判定為異常 component 明細」表格 | 看每個 defect 的 center / bbox / area / max_diff / solidity，立刻判斷分離性 |
| `/settings` → 🎯 AOI 機檢座標 tab | 改 DB config 熱更新 |
| Log 訊息 `🔲 _inspect_side`、`🔲 _detect_thin_lines` | 追蹤哪個 filter 擋掉 component |
| 使用者匯出的 log*.txt | 紀錄一次 debug run 的完整 component 明細，便於重現/對比 |

## 繼續工作的方法（for future sessions）

每次拿到新 case（使用者貼 log.txt / 截圖）：

1. **讀 component 明細表格**，觀察：
   - 總數量：少（< 5）= 乾淨面板；多（> 20）= 有雜訊源
   - max_diff 分佈：真 defect vs 雜訊是否有 gap？gap 有多大？
   - Shape signature：有沒有 1-px 寬、極長、規則排列的 outlier？
2. **跟 Phase 歷程裡的 case 比對**，判斷屬於：
   - 已處理類型 → 是不是 config 沒到位？
   - 新種類過檢/漏檢 → 需要新演算法
3. **提出解法時先問**：動 config 還是改演算法？
   - Config → 建議先 `/debug` 頁驗證，過 3 個不同 panel 再決定是否改 default
   - 演算法 → **必須在 log / log2 / log3 / log4 全部 regression 跑一次**確認不會退回以前修好的 case
4. **Trade-off 誠實告知**：每個 filter 都有誤殺可能，讓使用者基於 AOI 驗收標準決定

## 檔案定位快速索引

- **主演算法**：`capi_edge_cv.py`
  - `EdgeInspectionConfig` dataclass (~L116)
  - `from_db_params` (~L177)
  - `inpaint_non_fg_region` (~L36)
  - `_inspect_side` (~L666)
  - `_detect_thin_lines` + `_group_true_runs` (~L756)
  - `inspect_roi` (~L437)
- **Production 呼叫點**：`capi_inference.py:3518`（`inspect_roi`）、L3547-3556（建 is_cv_ok defect）
- **Debug API**：`capi_web.py:2085` (`_handle_api_debug_edge_inspect_corner`)
- **Hot-reload**：`capi_web.py:3005-3016`
- **DB seed**：`capi_database.py:2109-2116`
- **Debug UI**：`templates/debug_inference.html`（角落測試 section + 結果表格）
- **Settings UI**：`templates/settings.html`（`AOI_COORD_PARAMS` 列表）
- **誤導標籤**：`capi_heatmap.py:810-820`
