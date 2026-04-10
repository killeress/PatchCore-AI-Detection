# 面板 4 角 Polygon Mask 精準裁切 設計規格

**日期**: 2026-04-10
**相關問題**: 目前 Otsu axis-aligned bbox 無法貼合傾斜/keystone 變形的面板，導致邊緣 tile 包含非產品黑區，可能造成 PatchCore 誤判或 bomb 驗證不準。
**作者**: Ray (brainstorming with Claude)

---

## 1. 目標與背景

CAPI AI 目前用 `_find_raw_object_bounds` (`capi_inference.py:444`) 以 Otsu 二值化 + findContours + **axis-aligned boundingRect** 定位面板，結果存在 `ImageResult.otsu_bounds`。但實測兩張正式圖片 (`test_images/W0F00000_110022.tif`、`test_images/G0F00000_151955.tif`) 發現：

1. **面板在影像中不是純剛體旋轉**，而是「輕微旋轉 + 梯形 (keystone) 變形」
2. `cv2.minAreaRect` 對 keystone 無效 (它假設矩形)
3. Axis-aligned bbox 在右下角最嚴重時**誤差 37 px** (G0F00000_151955)，整體邊偏移最大 **40 px**
4. 這個誤差落在邊緣 tile 上，會把真實的黑色背景當作 panel 內部，後續:
   - PatchCore 在非產品區打分 → 可能假 NG
   - Bomb 座標驗證失準
   - MARK 排除區的 `relative_bottom_right` 錨點偏位

本設計**不引入任何影像 warp / 插值 / 透視校正**，而是：

- 新增 4 角 polygon 偵測 (逐邊直線 fit + 相交)
- 在既有的 `TileInfo.mask` 基礎設施上套用「tile ∩ panel polygon」
- `otsu_bounds` 保持不變 (向後相容)
- 下游 PatchCore 推論、dust filter、bomb 驗證、edge decay **全部無需修改**，因為它們早就已經支援 `tile.mask is not None` 的路徑

---

## 2. 設計決策摘要

| 決策點 | 選擇 | 理由 |
|---|---|---|
| 偵測演算法 | **逐邊 polyfit + 直線相交** | minAreaRect 對 keystone 無效；template match 需自製新模板且 panel 沒有對稱 4 角特徵 |
| 是否做影像 warp | **否** | warp 會引入插值、影響 PatchCore score；增加 CPU 成本；座標反向轉換出錯風險高 |
| 產品座標傳遞方式 | **透過 `tile.mask`** (既有基礎設施) | `capi_inference.py:1126、1164、1230` 已全面支援 `tile.mask`，只是目前設成 None |
| `otsu_bounds` 是否改定義 | **不改**，仍為 polygon 的外接 axis-aligned bbox | 下游所有 `otsu_bounds` 的 consumer 不動；tile 網格仍 axis-aligned |
| 4 角順序 | **TL, TR, BR, BL** (順時針，從左上開始) | 與 AIFunction.py 及 OpenCV 慣例一致 |
| 偵測失敗 fallback | **退回目前 axis-aligned bbox，polygon = bbox 4 角** | 零風險回退路徑，不會讓推論中斷 |
| B0F 黑圖處理 | **與現有 `reference_raw_bounds` 並行**，新增 `reference_polygon` | 沿用同一套「從同資料夾白圖繼承」邏輯 |
| Config 開關 | **新增 `enable_panel_polygon: true`** | 回退用；若線上出問題可一鍵關閉回到 axis-aligned bbox |
| AOI 座標映射 (`_map_aoi_coords`) 是否改用 polygon | **不改** (第一階段) | 超出本設計 scope；0.25° 旋轉僅造成 ~0.5% 誤差，留作未來優化 |
| Bomb 驗證是否用 polygon | **不改** (第一階段) | 同上；現行邏輯在 axis-aligned bbox 內已能運作 |
| MARK `relative_bottom_right` 錨點 | **改用 polygon BR 角** | 這是使用者明確提到的痛點 (bbox 右下偏就連帶 MARK 排除區偏)，成本極低 |

---

## 3. 系統架構

```
preprocess_image(image_path)
   │
   ├─ _find_raw_object_bounds(image)            ← 既有，產 binary+bbox
   │
   ├─ _find_panel_polygon(image, binary, bbox)  ← 【新增】
   │     ├─ 逐邊掃描 (top/bottom/left/right)
   │     ├─ np.polyfit 每邊一條直線 (帶 3σ robust filter)
   │     ├─ 4 線相交 → TL/TR/BR/BL
   │     ├─ 品質檢查 (角度 / 邊長 / 退化)
   │     └─ 失敗 → 回傳 bbox 4 角
   │
   ├─ calculate_otsu_bounds(...)
   │     └─ 不變，仍回傳 axis-aligned bbox
   │       (但現在 bbox = polygon 的外接 bbox)
   │
   ├─ calculate_exclusion_regions(image, bbox, polygon)
   │     └─ relative_bottom_right 錨點改為 polygon BR
   │
   └─ tile_image(image, bbox, polygon, exclusion_regions)
         │
         └─ 對每個 tile:
              ├─ 產生 tile 矩形 mask (預設 full 255)
              ├─ 【新增】在 mask 上 fillPoly(panel polygon) 做交集
              ├─ 若 tile 完全在 polygon 外 → 跳過 tile
              ├─ 若 tile 完全在 polygon 內 → mask=None (優化)
              └─ 若部分交集 → tile.mask = 該 tile 大小的 uint8 mask
```

下游既有流程**完全不動**：

- `capi_inference.py:1126` PatchCore anomaly_map × tile.mask (既有)
- `capi_inference.py:1164` decay ratio 自動納入 mask 效果 (既有)
- `capi_inference.py:1230` B0F 亮點偵測 binary & tile.mask (既有)
- Edge margin decay 讀 `is_*_edge` 旗標 (既有)

---

## 4. 資料結構變更

### 4.1 `ImageResult` 新欄位

```python
@dataclass
class ImageResult:
    ...
    otsu_bounds: Tuple[int, int, int, int]             # 不變 (仍為 axis-aligned bbox)
    raw_bounds: Optional[Tuple[int, int, int, int]]    # 不變
    # 【新增】
    panel_polygon: Optional[np.ndarray] = None
    """面板 4 角 polygon，shape (4, 2) float32，順序為 [TL, TR, BR, BL]。
    若偵測失敗則為 None (下游應 fallback 回 axis-aligned bbox 行為)。"""
```

### 4.2 `TileInfo.mask` 語意重新啟用

```python
@dataclass
class TileInfo:
    ...
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    """遮罩: 255=有效 (tile 內落在 panel polygon 內的像素), 0=排除 (非產品)。
    如果 tile 完全落在 panel polygon 內，mask 為 None 以節省記憶體。
    舊有的 '不再使用遮罩' 註記移除。"""
```

### 4.3 Config 新增欄位

```yaml
# configs/capi_3f.yaml
# 面板 polygon 偵測設定 (新增)
enable_panel_polygon: true        # 主開關，false 則回到純 axis-aligned bbox 行為
```

**注意**: 不加 `polygon_sample_step`、`polygon_edge_margin` 等參數 — 演算法用的常數直接 hardcode 在 `_find_panel_polygon` 裡，只有當線上證明需要 per-model 微調時才提出到 config。

---

## 5. 新增 / 修改的函式

### 5.1 `_find_panel_polygon` (新增)

位置：`capi_inference.py`，放在 `_find_raw_object_bounds` 後方。

```python
def _find_panel_polygon(
    self,
    binary_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    """
    在既有 Otsu binary mask 上用逐邊 polyfit 找 4 角 panel polygon。

    Args:
        binary_mask: Otsu + morphology close 後的 binary 圖 (與 _find_raw_object_bounds
                     使用的 closing 相同)
        bbox: 粗略 bbox (x_min, y_min, x_max, y_max)，作為邊緣掃描範圍

    Returns:
        np.ndarray shape (4,2) float32, 順序 [TL, TR, BR, BL]；
        若偵測失敗回傳 None。
    """
```

**演算法**:

1. **逐邊取樣邊緣點**
   - 取樣範圍: 四周內縮 `EDGE_MARGIN=20` px 避開角落
   - 取樣步距: `SAMPLE_STEP=50` px
   - Top: 對每個 x 找 `binary_mask[:, x]` 中 **最小** 的前景 y
   - Bottom: 對每個 x 找 **最大** 的前景 y
   - Left: 對每個 y 找 `binary_mask[y, :]` 中 **最小** 的前景 x
   - Right: 對每個 y 找 **最大** 的前景 x
   - 任一邊若取樣點 < 5 → fallback
2. **每邊用 `np.polyfit` 一次直線擬合**
   - Top/Bottom: `y = a*x + b`
   - Left/Right: `x = a*y + b`
3. **Robust filter**: 計算殘差 σ，剔除 |residual| > 3σ 的點後再 fit 一次；若剔除後剩 < 3 點則用第一次 fit
4. **4 線相交**
   - `TL = intersect(top, left)`, `TR = intersect(top, right)`
   - `BR = intersect(bottom, right)`, `BL = intersect(bottom, left)`
   - 分母接近 0 (線近乎平行) → fallback
5. **品質檢查** (任一失敗則 fallback)
   - 4 角必須在 image 範圍內 (可接受最多 `-50` px 溢出容忍數值誤差)
   - 4 邊長度必須 > `tile_size` (512)，避免退化成細長帶
   - 4 角構成的凸四邊形面積 ≥ bbox 面積 × 0.9 (避免偵測到內部雜訊)
6. **回傳** `np.array([TL, TR, BR, BL], dtype=np.float32)`，或 `None`

**關鍵常數** (hardcode，不入 config):
```python
EDGE_MARGIN = 20
SAMPLE_STEP = 50
OUTLIER_SIGMA = 3.0
MIN_EDGE_LEN_RATIO = 1.0  # relative to tile_size
MIN_POLYGON_AREA_RATIO = 0.9  # vs bbox area
```

### 5.2 `_find_raw_object_bounds` (輕微修改)

讓它回傳 binary mask (目前是內部變數丟掉)，供 `_find_panel_polygon` 重複使用，避免重算 Otsu。

```python
def _find_raw_object_bounds(
    self, image: np.ndarray
) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    """回傳 ((x_min, y_min, x_max, y_max), binary_mask)"""
```

**向後相容**: 只有 `calculate_otsu_bounds` 和 `preprocess_image` 會呼叫 `_find_raw_object_bounds`，兩處都在本設計中會同步改動。

### 5.3 `calculate_otsu_bounds` (修改簽名)

```python
def calculate_otsu_bounds(
    self,
    image: np.ndarray,
    otsu_offset_override: Optional[int] = None,
    reference_raw_bounds: Optional[Tuple[int, int, int, int]] = None,
    reference_polygon: Optional[np.ndarray] = None,  # 【新增】
) -> Tuple[
    Tuple[int, int, int, int],  # bbox (同舊)
    Optional[int],               # original_y2 (同舊)
    Optional[np.ndarray],        # 【新增】panel_polygon
]:
```

行為:
- 若 `reference_polygon is not None` → 直接使用它 (B0F 黑圖路徑)
- 否則: 呼叫 `_find_panel_polygon(binary_mask, bbox)` 取得 polygon
- 若 `self.config.enable_panel_polygon is False` → polygon = None (完全 fallback 到舊行為)
- bbox 計算邏輯**不變** (仍是 `_find_raw_object_bounds` 的 bbox + `otsu_offset` 內縮 + `otsu_bottom_crop`)
- polygon 也依 `otsu_offset` 做**同向內縮** (避免 bbox 和 polygon 不同步)
- `otsu_bottom_crop > 0` 時 polygon 下半部也截掉 (與 bbox 邏輯同步)

### 5.4 `preprocess_image` (修改簽名)

```python
def preprocess_image(
    self,
    image_path: Path,
    cached_mark: Optional[ExclusionRegion] = None,
    otsu_offset_override: Optional[int] = None,
    reference_raw_bounds: Optional[Tuple[int, int, int, int]] = None,
    reference_polygon: Optional[np.ndarray] = None,  # 【新增】
) -> Optional[ImageResult]:
```

- 接 `calculate_otsu_bounds` 的三值回傳
- 把 polygon 傳給 `tile_image` 和 `calculate_exclusion_regions`
- 把 polygon 寫入 `ImageResult.panel_polygon`

### 5.5 `tile_image` (修改簽名與內部邏輯)

```python
def tile_image(
    self,
    image: np.ndarray,
    otsu_bounds: Tuple[int, int, int, int],
    exclusion_regions: List[ExclusionRegion],
    panel_polygon: Optional[np.ndarray] = None,  # 【新增】
    exclusion_threshold: float = 0.0,
) -> Tuple[List[TileInfo], int]:
```

**新增的處理流程** (放在原本建 `TileInfo` 前):

1. 若 `panel_polygon is None` → 退回舊行為，`mask=None`
2. 否則 for each candidate tile:
   - 建立 `tile_rect_mask = np.full((tile_size, tile_size), 255, np.uint8)`
   - 建立 `panel_mask_in_tile = np.zeros((tile_size, tile_size), np.uint8)`
     - 把 panel_polygon 的座標減去 `(x, y)` (tile 的左上角) → tile 座標系下的 polygon
     - `cv2.fillPoly(panel_mask_in_tile, [polygon_in_tile.astype(np.int32)], 255)`
   - `tile.mask = panel_mask_in_tile`
   - **優化**: 若 `panel_mask_in_tile.min() == 255` (全 255) → `tile.mask = None` 省記憶體
   - **跳過條件**: 若 `panel_mask_in_tile.max() == 0` (全 0)，**整個 tile 不納入**（與現行 exclusion 等同處理，`excluded_count += 1`）

**注意**: `is_*_edge` 旗標仍然以 axis-aligned bbox 判定 (不改)，因為 edge margin decay 假設 tile 貼齊 bbox 邊。polygon 的傾斜部分由 mask 處理，不影響 edge flag。

### 5.6 `calculate_exclusion_regions` (修改簽名)

```python
def calculate_exclusion_regions(
    self,
    image: np.ndarray,
    otsu_bounds: Tuple[int, int, int, int],
    cached_mark: Optional[ExclusionRegion] = None,
    panel_polygon: Optional[np.ndarray] = None,  # 【新增】
) -> List[ExclusionRegion]:
```

`relative_bottom_right` 路徑改為:
```python
elif zone.type == "relative_bottom_right":
    # 以 polygon BR 角為錨點 (若 polygon 存在)，否則 fallback 到 bbox 右下
    if panel_polygon is not None:
        br_x, br_y = int(round(panel_polygon[2][0])), int(round(panel_polygon[2][1]))
    else:
        br_x, br_y = otsu_x2, otsu_y2
    regions.append(ExclusionRegion(
        name=zone.name,
        x1=max(otsu_x1, br_x - zone.width),
        y1=max(otsu_y1, br_y - zone.height),
        x2=br_x,
        y2=br_y,
    ))
```

MARK template-match 路徑不變 (`find_mark_region` 是獨立的，與 polygon 無關)。

### 5.7 B0F 黑圖 fallback (修改 `process_panel` 或其周邊呼叫者)

目前 `capi_inference.py:2979` 區塊計算 `reference_raw_bounds_for_dark`。新增並行邏輯:

```python
reference_raw_bounds_for_dark = None
reference_polygon_for_dark = None  # 【新增】

# ... 找到白圖 ref_img 時:
ref_bounds, ref_binary = self._find_raw_object_bounds(ref_img)
reference_raw_bounds_for_dark = ref_bounds
reference_polygon_for_dark = self._find_panel_polygon(ref_binary, ref_bounds)
# 兩個都可能是 None → 都 pass 下去

# 呼叫 preprocess_image 時:
ref_bounds = reference_raw_bounds_for_dark if _is_dark_image(img_path.name) else None
ref_poly = reference_polygon_for_dark if _is_dark_image(img_path.name) else None
result = self.preprocess_image(
    img_path,
    cached_mark=cached_mark,
    reference_raw_bounds=ref_bounds,
    reference_polygon=ref_poly,
)
```

---

## 6. 視覺化變更

### 6.1 `capi_inference.py` 內部 debug 視覺化 (約 line 1385、3805)

目前畫 bbox (藍色)。新增:
- Polygon 用紅色線畫上 (`cv2.polylines`)
- 4 個角用黃色圓圈標註
- 標題列標 "bbox (blue) + polygon (red)"

此為 debug 用，效能允許 (只在視覺化路徑執行)。

### 6.2 Web Dashboard 熱圖 overlay

目前 `capi_web.py` / `capi_heatmap.py` 的 overlay 顯示 bbox。**第一階段不改**，維持顯示 axis-aligned bbox (因為 frontend 的 tile grid 仍 axis-aligned，多畫 polygon 反而混亂)。若後續需要可在 follow-up 補。

---

## 7. B0F 黑圖路徑細節

| 階段 | 目前做法 | 新設計 |
|---|---|---|
| B0F bbox | 從同資料夾白圖的 `_find_raw_object_bounds` 繼承 | 不變 |
| B0F polygon | (無) | 從同白圖的 `_find_panel_polygon` 繼承 |
| B0F 若找不到白圖 | 用自身 OTSU bbox (可能不準) | polygon = None，退回純 bbox 行為 |
| B0F 的 `_detect_bright_spots` | 讀 `tile.mask` (既有) | 自動生效 — mask 由新路徑產生 |

---

## 8. 相容性與回退

### 8.1 Config 開關

```yaml
enable_panel_polygon: true
```

- **`true` (預設)**: 啟用 polygon 偵測與 mask
- **`false`**: `_find_panel_polygon` 一律回傳 None，所有 `tile.mask` 為 None，行為與改動前完全一致

這允許線上若發現 PatchCore score 有非預期漂移，可一鍵回退，不需 rollback 程式碼。

### 8.2 偵測失敗的 fallback

`_find_panel_polygon` 內部任一品質檢查失敗 → 回傳 None → 下游與 `enable_panel_polygon=false` 等價。**不會中斷推論**。

### 8.3 Panel polygon 為 None 時的下游行為

| 下游函式 | polygon=None 行為 |
|---|---|
| `tile_image` | 產生全 axis-aligned tile，`mask=None` (同現行) |
| `calculate_exclusion_regions` | `relative_bottom_right` 用 bbox 右下角 (同現行) |
| PatchCore (`capi_inference.py:1126`) | `if tile.mask is not None` → 跳過 mask 步驟 (同現行) |
| B0F bright spot (`:1230`) | 同上 |
| Debug 視覺化 | 不畫 polygon (只畫 bbox) |

---

## 9. 測試策略

### 9.1 單元測試 — `tests/test_panel_polygon_detect.py`

現行已有的 stand-alone 腳本 **提升為 pytest**，加入斷言:

1. **W0F00000_110022.tif**: 4 角誤差 vs 目前 bbox
   - 預期 TL=6.6, TR=15.7, BR=15.0, BL=0.6 (±2 px)
2. **G0F00000_151955.tif**: 4 角誤差 vs 目前 bbox
   - 預期 TL=16.7, TR=19.0, BR=36.9, BL=3.8 (±2 px)
3. **退化情境測試**: 餵一張全黑圖 → polygon 應為 None
4. **退化情境測試**: 餵一張只有小雜點的圖 → polygon 應為 None (MIN_POLYGON_AREA_RATIO 檢查)
5. **理想矩形測試**: 餵一張完美 axis-aligned 矩形 → polygon 應為 bbox 的 4 角 (誤差 < 1 px)
6. **順序正確性**: TL.y < BL.y, TR.y < BR.y, TL.x < TR.x, BL.x < BR.x

### 9.2 Tile mask 視覺回歸

`tests/test_panel_polygon_tile_mask.py` (新增):
- 對 2 張測試圖跑 `tile_image` 並檢查:
  - 每個 tile 的 `mask` 要不是 None，要不是 shape (512, 512) uint8
  - Tile 內 mask 為 0 的像素必定落在 polygon 外
  - Tile 內 mask 為 255 的像素必定落在 polygon 內
- 產生視覺化 PNG 存到 `test_output/panel_polygon/` (與現行 `test_panel_polygon_detect.py` 的輸出共用資料夾)

### 9.3 Inference 回歸測試

在 2 張測試圖上跑 **完整 `process_panel`**:
- 比較 `enable_panel_polygon=false` vs `true` 的 `pred_score`
- 對非邊緣 tile: score 差異應 < 0.1% (mask 全 255 → 無影響)
- 對邊緣 tile: 允許 score 下降 (mask 部分為 0 → 正確排除黑區)，但**不應上升**
- 任一 tile score 暴增視為 regression

### 9.4 既有測試不得 regression

```bash
python tests/test_inference.py           # TCP protocol test
python tests/test_cv_edge.py              # CV edge detection
python tests/test_aoi_coord_inference.py  # AOI coord pipeline
python tests/test_dust_two_stage.py       # Two-stage dust filter
```

全部要通過且 pred_score 漂移 < 0.5%。

---

## 10. Scope 明確界定

### 10.1 本次 **會** 改的

- `capi_inference.py`: 新增 `_find_panel_polygon`, 修改 `_find_raw_object_bounds`/`calculate_otsu_bounds`/`preprocess_image`/`tile_image`/`calculate_exclusion_regions`/B0F fallback 路徑
- `capi_inference.py`: `ImageResult` 新增 `panel_polygon` 欄位
- `capi_inference.py`: `TileInfo.mask` 重新啟用 (改掉「不再使用遮罩」的註記)
- `configs/capi_3f.yaml`: 新增 `enable_panel_polygon: true`
- `capi_config.py`: 加 `enable_panel_polygon: bool = True` 欄位
- `tests/test_panel_polygon_detect.py`: 從 stand-alone 提升為 pytest
- `tests/test_panel_polygon_tile_mask.py`: 新增

### 10.2 本次 **不會** 改的 (留作未來優化)

- `_map_aoi_coords`: 仍用 raw_bounds axis-aligned 映射 (0.25° 旋轉僅 ~0.5% 誤差)
- `check_bomb_defect`: 仍用 axis-aligned 邏輯
- Web Dashboard heatmap overlay: 仍畫 bbox，不畫 polygon
- 資料庫 schema: `inference_records`, `image_results`, `tile_results` 都不動
- `capi_edge_cv.py`: CV 邊緣檢查不動 (它已經在做自己的 edge 掃描)
- RIC 報表、dust filter 主流程: 不動

---

## 11. 風險與緩解

| 風險 | 機率 | 衝擊 | 緩解 |
|---|---|---|---|
| 某些光源下 Otsu binary 邊緣有雜訊，polygon 偵測失敗率高 | 中 | 中 | Robust 3σ filter；失敗 fallback 回 bbox；`enable_panel_polygon` 開關 |
| Tile 內 fillPoly CPU 成本累加 (panel 有 ~80 個 tile) | 低 | 低 | fillPoly 對 512×512 uint8 極快 (<0.1 ms/tile)；全 panel 約 +10 ms |
| PatchCore score 因 mask 下降而出現 false OK | 低 | 高 | 測試 9.3 已明確檢查；mask 只在真正 panel 外像素上，panel 內像素完全不變 |
| `relative_bottom_right` 錨點改動導致 MARK 排除區位置變化 | 低 | 中 | 原本就有 offset 5 px 誤差，改動後 ≤ 40 px；人工比對 2 張測試圖後驗證 |
| B0F 黑圖沒對應白圖時，polygon 為 None | 低 | 低 | 已處理: fallback 退回 bbox-only 行為 |
| 梯形極嚴重時 (單邊偏差 > 100 px) polygon 仍有殘差 | 極低 | 低 | 本次目標是吃掉 ~40 px 誤差，更極端情境超出 scope |

---

## 12. 分階段交付建議

本 spec 為 **單階段交付**，但實作時建議按以下順序提交 (方便 review):

1. **Commit 1**: `_find_panel_polygon` 函式 + `_find_raw_object_bounds` 回傳 binary mask + 單元測試
2. **Commit 2**: `ImageResult.panel_polygon` 欄位 + `calculate_otsu_bounds` / `preprocess_image` 整合 (polygon 已計算但尚未使用)
3. **Commit 3**: `tile_image` 套用 polygon mask + tile mask 視覺回歸測試
4. **Commit 4**: `calculate_exclusion_regions` 用 polygon BR 角
5. **Commit 5**: B0F 黑圖 `reference_polygon` fallback
6. **Commit 6**: Debug 視覺化 (`:1385`, `:3805`) 畫 polygon
7. **Commit 7**: Config 開關 `enable_panel_polygon` + 文件更新

每個 commit 後都可跑 `python tests/test_inference.py` 確認無 regression。

---

## 13. 驗收條件

- [ ] `W0F00000_110022.tif`、`G0F00000_151955.tif` 兩張測試圖:
  - 4 角偵測誤差與 section 9.1 預期值相符 (±2 px)
  - 右下角 tile 的 mask 正確排除非產品區 (視覺回歸)
- [ ] `process_panel` 在兩張圖上的 `pred_score` 相對於 `enable_panel_polygon=false`:
  - 非邊緣 tile 漂移 < 0.1%
  - 邊緣 tile 只會下降不會上升
- [ ] 現有測試全數通過 (`test_inference.py`, `test_cv_edge.py`, `test_aoi_coord_inference.py`, `test_dust_two_stage.py`)
- [ ] `enable_panel_polygon: false` 可完全回退到現行行為，任何 pred_score 差異 = 0
- [ ] `tests/test_panel_polygon_detect.py` 從 stand-alone 升為 pytest 且通過
- [ ] `tests/test_panel_polygon_tile_mask.py` 新增並通過

---

## 14. 參考

- `test_output/panel_polygon/*.png` — 視覺驗證已通過 (2026-04-10 使用者確認)
- `AIFunction.py` — 靈感來源 (4 角 template match 做座標轉換)，但本設計不移植其 template 路線
- `CLAUDE.md` §Inference Pipeline — 現行流程的 high-level 描述
- `capi_inference.py:1126, 1164, 1230` — 現有 `tile.mask` 支援點
