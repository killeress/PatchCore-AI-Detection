# AOI 邊緣 Fusion — PC ROI 內移設計（Phase 7）

**日期**：2026-04-23
**作者**：Ray（設計）/ Claude（記錄）
**前置**：`docs/superpowers/specs/2026-04-22-aoi-edge-fusion-inspector-design.md`（Phase 6 fusion）
**相關文件**：`docs/edge-cv-tuning-log.md`

## 問題陳述

Phase 6 上線後觀察到：fusion 雖然已在 **scoring 層**把 band 區從 PC 分數排除，但 PC backbone 的 feature extraction 仍然跑在整張含 polygon edge 的 512×512 ROI 上；backbone receptive field ~30–50 px 會被 polygon discontinuity（panel 邊外是全黑 pad）汙染，導致 PC interior 區的 feature map 在靠近 band 側仍有殘留「偽邊緣」activation。

**結果**：PC INTERIOR 命中中，d_edge 偏小的案例仍有疑似過檢殘留。

## 目標

1. 進一步抑制 PC 過檢殘留：讓 PC 輸入的 ROI **完全脫離 polygon 邊**，feature map 不含邊緣 discontinuity
2. 保持 CV band 覆蓋不變：CV ROI 仍居中於 AOI 座標（Phase 6 行為）
3. 不產生檢測空白區：PC 沒看到的區域由 CV 補位
4. UI 能清楚呈現雙 ROI 布局，讓現場人員看得懂 shift 在做什麼

## 非目標

- Phase 6 CV / PC 分權的核心邏輯不動（band 區 CV、interior 區 PC）
- 不調整 PatchCore backbone / threshold / CV 參數
- Bomb match 座標一致性不破壞（defect center 仍 = AOI 座標）

## 設計決策（已對齊）

### 內移策略：C — 動態偏移到 PC ROI 距 polygon ≥ N px

- N = `aoi_edge_boundary_band_px`（共用，預設 40 px）
- 偏移方向：AOI 座標到 **最近 polygon 邊**的 inward normal
- 偏移量：`max(0, band_px - (d_edge - 256))`，使得偏移後 PC ROI 距 polygon ≥ band_px

### AOI 座標位置約束：距 PC ROI 邊 ≥ 64 px

- `aoi_edge_aoi_margin_px = 64`（可配置）
- 偏移量 clamp 到 `[0, 256 - aoi_margin_px] = [0, 192] px`

### 偏移方向衝突（凹角 / 多邊同時逼近）：用最近單一邊的 normal

- 取與 AOI 座標最近的單一 polygon 線段
- 沿該線段 inward normal 偏移
- 偏移後若仍有其他 polygon 邊侵入 PC ROI（`verify_polygon_clear_of_pc_roi` fail）→ fallback 到 Phase 6 原行為（centered PC ROI + band_mask）

### CV 覆蓋：跑整個原 ROI band 區（含 PC ROI 重疊處）

- CV 邏輯與 Phase 6 完全一致，**不需要感知 PC ROI 位置**
- 重疊區 CV + PC 都看，任一 NG 即 NG，零漏網

### Defect center 一致性

- 沿用 Phase 6：fusion 內所有 defect 的 `center` 強制設為 AOI 座標，確保 Bomb match 對齊

## 核心演算法

### 新 helper 1：`compute_pc_roi_offset`

純函式，位置：`capi_edge_cv.py`

```python
def compute_pc_roi_offset(
    aoi_xy: Tuple[int, int],
    polygon: np.ndarray,  # Nx2 panel 座標系
    band_px: int = 40,
    aoi_margin_px: int = 64,
    roi_size: int = 512,
) -> Tuple[Tuple[int, int], Tuple[int, int], float]:
    """計算 PC ROI 應有的偏移量。

    Returns:
        (pc_roi_origin, shift_vec, d_edge)
        - pc_roi_origin: (ox, oy) 偏移後 PC ROI 左上角絕對座標
        - shift_vec: (dx, dy) 相對於 centered ROI 的偏移量 (px)
        - d_edge: AOI 座標到 polygon 的有號距離（內側 > 0）

    邊界情境：
        - polygon=None / 頂點 < 3 → shift=(0,0) d_edge=-inf（caller 會 fallback）
        - AOI 座標在 polygon 外（d_edge < 0）→ shift=(0,0)
        - 偏移量 = max(0, band_px - (d_edge - roi_size/2))
          * d_edge >= roi_size/2 + band_px 時，不需偏移
          * 偏移量 clamp 到 [0, roi_size/2 - aoi_margin_px]
    """
```

### 新 helper 2：`verify_polygon_clear_of_pc_roi`

純函式，位置：`capi_edge_cv.py`

```python
def verify_polygon_clear_of_pc_roi(
    pc_roi_origin: Tuple[int, int],
    roi_size: int,
    polygon: np.ndarray,
    band_px: int,
) -> bool:
    """檢查偏移後 PC ROI 是否「完全乾淨」——所有 polygon 邊距 PC ROI ≥ band_px。

    做法：
        1. 建立 PC ROI 範圍內 distance transform 到 polygon 邊
        2. 若最小距離 < band_px（任何 polygon 邊侵入 ROI 或太近）→ False

    凹角 / 複雜 polygon 情境：
        即使偏移到最近邊夠遠，其他邊可能從另一側逼近 → verify False → caller fallback
    """
```

### 修改 `_inspect_roi_fusion`（`capi_inference.py`）

```
if panel_polygon is None:
    # 既有 fallback 不動
    ...
    return

# 1) Build CV ROI (centered, 不變)
cv_roi = build_centered_roi(image, aoi_xy, tile_size)
cv_fg_mask = build_fg_mask_from_polygon(cv_roi, panel_polygon)
band_mask = compute_boundary_band_mask(cv_roi, ..., band_px, cv_fg_mask)

# 2) 計算 PC ROI 偏移
pc_roi_origin, shift_vec, d_edge = compute_pc_roi_offset(
    aoi_xy, panel_polygon, band_px, aoi_margin_px, tile_size
)

# 3) 驗證並決定是否內移
use_shifted = (shift_vec != (0, 0)) and verify_polygon_clear_of_pc_roi(
    pc_roi_origin, tile_size, panel_polygon, band_px
)

if use_shifted:
    # shifted: PC 完全乾淨，不需要 band_mask 排除
    pc_stats = run_pc_at_origin(image, pc_roi_origin, img_prefix, panel_polygon)
    pc_anomaly_map_for_scoring = pc_stats["anomaly_map"] * (fg_mask>0)  # 只過 fg_mask
    fallback_reason = ""
else:
    # fallback: 與 Phase 6 相同，centered + band_mask 排除
    pc_stats = run_pc_at_origin(image, centered_origin, img_prefix, panel_polygon)
    pc_anomaly_map_for_scoring = pc_stats["anomaly_map"]
    pc_anomaly_map_for_scoring[band_mask > 0] = 0
    pc_anomaly_map_for_scoring[fg_mask == 0] = 0
    fallback_reason = "concave_polygon" or "no_shift_needed"
    pc_roi_origin = centered_origin
    shift_vec = (0, 0)

# 4) CV 路徑同 Phase 6（跑原 CV ROI + band_mask 保留）
cv_defects_kept = run_cv_and_filter_to_band(cv_roi, band_mask, ...)

# 5) PC threshold / 生成 defect（與 Phase 6 一致，只是座標系用 pc_roi_origin）
pc_defect = threshold_and_build_defect(pc_anomaly_map_for_scoring, pc_roi_origin, aoi_xy)

# 6) 每個 fusion defect 寫入 shift 欄位
for d in fusion_defects:
    d.pc_roi_origin = pc_roi_origin
    d.pc_roi_shift_dx = shift_vec[0]
    d.pc_roi_shift_dy = shift_vec[1]
    d.pc_roi_fallback_reason = fallback_reason

# 7) OMIT dust filter 不變
```

## 新增欄位

### `EdgeDefect`（`capi_edge_cv.py`）

```python
pc_roi_origin: Tuple[int, int] = (0, 0)    # PC 實際跑的 ROI 左上角 (panel 絕對座標)
pc_roi_shift_dx: int = 0                    # 相對 centered ROI 的偏移 x
pc_roi_shift_dy: int = 0                    # 相對 centered ROI 的偏移 y
pc_roi_fallback_reason: str = ""            # "" / "concave_polygon" / "polygon_unavailable"
```

### Config（`EdgeInspectionConfig`）

```python
aoi_edge_pc_roi_inward_shift_enabled: bool = True   # Phase 7 開關，預設 True
aoi_edge_aoi_margin_px: int = 64                    # AOI 座標距 PC ROI 邊最小 margin
```

對應 DB `config_params` seed：
- `aoi_edge_pc_roi_inward_shift_enabled` (bool, default True)
- `aoi_edge_aoi_margin_px` (int, default 64)

### DB schema `edge_defect_results`（ALTER TABLE）

```sql
ALTER TABLE edge_defect_results ADD COLUMN pc_roi_origin_x INTEGER DEFAULT 0;
ALTER TABLE edge_defect_results ADD COLUMN pc_roi_origin_y INTEGER DEFAULT 0;
ALTER TABLE edge_defect_results ADD COLUMN pc_roi_shift_dx INTEGER DEFAULT 0;
ALTER TABLE edge_defect_results ADD COLUMN pc_roi_shift_dy INTEGER DEFAULT 0;
ALTER TABLE edge_defect_results ADD COLUMN pc_roi_fallback_reason TEXT DEFAULT '';
```

透過 `add_column_if_not_exists` 做可重入 migration。

## UI 視覺化

### `record_detail.html` — fusion 缺陷列

在既有 `Source` / `d_edge` / `OMIT` 欄位之後再加：
- **Shift 欄**：顯示 `(dx, dy) px`，若 `(0, 0)` 顯示 `—`，fallback 時顯示 badge `fallback: concave_polygon`

### `debug_inference.html` + heatmap renderer — 雙 ROI 並排

```
[ CV ROI (centered)             ]   [ PC ROI (shifted)               ]   [ OMIT ROI ]
  紅虛框：PC ROI 在此圖中位置         黃十字：AOI 座標在此圖中位置
  黃十字：AOI 座標（= ROI 中心）      紅虛框：band 區在此 ROI 殘留（fallback 時才有）
  橘色 band mask 疊層                 heatmap 疊層（全 ROI 或 band-masked 視 fallback）
```

右上角 badge：
- `shift=(dx, dy) d_edge=Xpx`（shifted）
- `FALLBACK: concave_polygon`（fallback）
- `FALLBACK: polygon_unavailable`（無 polygon）

### `settings.html` AOI_COORD_PARAMS

- 加 `aoi_edge_pc_roi_inward_shift_enabled`
- 加 `aoi_edge_aoi_margin_px`

## 測試計畫（TDD）

### Unit tests (`tests/test_aoi_edge_pc_roi_shift.py`)

**`TestComputePcRoiOffset`**
- `test_deep_interior_no_shift` — AOI 座標 d_edge > 256+band → shift=(0,0)
- `test_near_left_edge_shifts_right` — AOI 近左邊，inward normal 指右 → shift=(+N, 0)
- `test_near_top_edge_shifts_down` — AOI 近上邊 → shift=(0, +N)
- `test_near_corner_uses_nearest_single_edge` — AOI 在角落附近，取最近的那條邊
- `test_clamped_to_aoi_margin` — 極近邊（d_edge=5）時，shift 不超過 192
- `test_polygon_none_returns_zero_shift` — polygon None → (0,0)
- `test_aoi_outside_polygon` — AOI 座標在 polygon 外 → (0,0)

**`TestVerifyPolygonClearOfPcRoi`**
- `test_clean_shifted_roi` — 簡單矩形 polygon，shift 後完全 clear → True
- `test_polygon_still_inside_pc_roi` — 偏移不足、polygon 仍在 PC ROI → False
- `test_concave_polygon_with_second_edge_intruding` — L 形 polygon，偏移後另一邊靠近 → False

**`TestInspectRoiFusionShifted`**（mock PC inferencer）
- `test_centered_path_when_deep_interior` — shift=0，行為同 Phase 6
- `test_shifted_path_near_edge` — 近邊時 PC ROI 確實從偏移位置取樣、band_mask 不套用
- `test_fallback_when_verify_fails` — mock verify 回 False，退回 Phase 6 centered + band_mask
- `test_defect_center_forced_to_aoi_coord` — shifted 命中，defect.center == AOI 座標
- `test_shift_fields_populated` — EdgeDefect.pc_roi_origin / shift_dx / shift_dy / fallback_reason 正確

### 迴歸
- 既有 `tests/test_aoi_edge_fusion.py`（Phase 6, 17 個）全綠
- 既有 `tests/test_aoi_coord_inference.py` / `tests/test_cv_edge.py` 全綠
- 全套 unit tests ≥ 150 + 新增

## 風險與 fallback

| 風險 | 處置 |
|---|---|
| 凹角 polygon 偏移後仍有 polygon 邊侵入 | `verify_polygon_clear_of_pc_roi` 檢測 fail → fallback 到 Phase 6 行為 |
| 偏移後 PC feature map 在 shifted ROI 邊緣 padding 區仍有污染 | shifted ROI 本來就全在 fg_mask 內（polygon 離 ≥ band_px），padding 影響小 |
| 算力增加 | 幾乎無（PC 仍跑 1×，只是輸入裁切位置換了；helper 是純幾何計算） |
| UI 混淆：shifted PC ROI 在 CV ROI 外時難以可視化 | heatmap 雙圖並排 + 紅虛框標位置，badge 顯示 shift 量 |
| Bomb match 座標對不上 | `defect.center = AOI 座標` 強制 — 不變 |

## Rollout

1. 寫 DB migration + seed（可重入、預設 `shift_enabled=True`）
2. TDD 實作 helper（Slice 2、3）
3. 改造 `_inspect_roi_fusion`（Slice 4）
4. Persistence（Slice 5）
5. Heatmap 雙 ROI 渲染（Slice 6）
6. UI templates（Slice 7）
7. docs 更新 + 全套迴歸（Slice 8）
8. 上線後觀察一週 d_edge < 100 px 的 PC 命中數下降比例

## 預計命中改善

- 目標：d_edge < 100 px 的 PC INTERIOR 過檢 ≥ 50% 下降
- 若命中沒下降 → 說明 PC 過檢主因不是邊緣汙染 → 下一階段考慮 model drift / OMIT 規則強化
