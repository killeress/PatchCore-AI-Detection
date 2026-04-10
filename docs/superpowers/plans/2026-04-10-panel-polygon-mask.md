# 面板 4 角 Polygon Mask 精準裁切 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 取代現行 axis-aligned Otsu bbox 的非產品區裁切，改用「逐邊 polyfit 相交得 4 角 panel polygon + 在既有 `TileInfo.mask` 基礎設施上做 polygon intersection」的方式，讓傾斜/keystone 變形的面板邊緣 tile 能精準排除黑色非產品區。

**Architecture:** 保留 `otsu_bounds` (axis-aligned bbox) 不變以維持向後相容。新增 `ImageResult.panel_polygon` 存 4 角座標，並在 `tile_image()` 為每個 tile 產生 `tile.mask`（tile 矩形 ∩ polygon）。下游 PatchCore 推論、B0F 亮點偵測、edge decay 等**都不需改動**，因為它們早已支援 `tile.mask is not None` 路徑（`capi_inference.py:1126, 1164, 1230`）。

**Tech Stack:** Python 3, OpenCV (`cv2.fillPoly`, `cv2.threshold`, `cv2.findContours`), NumPy (`np.polyfit`), dataclass/YAML config, 既有 CAPI 推論 pipeline

**Spec:** [docs/superpowers/specs/2026-04-10-panel-polygon-mask-design.md](../specs/2026-04-10-panel-polygon-mask-design.md)

---

## File Structure

### 會修改的檔案

| 檔案 | 職責 | 變更 |
|---|---|---|
| `capi_config.py` | Config dataclass + YAML 解析/序列化 | 新增 `enable_panel_polygon` 欄位；更新 `from_dict`/`to_dict` |
| `configs/capi_3f.yaml` | Runtime config | 新增 `enable_panel_polygon: true` |
| `capi_inference.py` | 推論主流程 | `TileInfo` 註記移除「不再使用遮罩」；`ImageResult` 新增 `panel_polygon`；新增 `_find_panel_polygon`；修改 `_find_raw_object_bounds`/`calculate_otsu_bounds`/`preprocess_image`/`tile_image`/`calculate_exclusion_regions`；修改 B0F 黑圖 fallback 路徑；debug 視覺化多畫 polygon |

### 會新增的檔案

| 檔案 | 職責 |
|---|---|
| `tests/test_panel_polygon_unit.py` | `_find_panel_polygon` 的單元測試（退化/理想/真實圖三類） |
| `tests/test_panel_polygon_tile_mask.py` | `tile_image` 套用 polygon mask 後的視覺回歸 |

### 不動的檔案

- `capi_server.py`, `capi_database.py`, `capi_web.py`, `capi_heatmap.py`, `capi_edge_cv.py`
- 其他 `tests/*`（既有測試不得 regression，但無需修改）

---

## Task 1: Foundation — Config 開關 + `_find_raw_object_bounds` 回傳 binary mask

**目的**: 先把最底層的 config toggle 加上，並讓 `_find_raw_object_bounds` 回傳 binary mask（後續 Task 2 的 `_find_panel_polygon` 會重用這個 mask，避免重算 Otsu）。這個 task 不改任何行為，只是鋪路。

**Files:**
- Modify: `capi_config.py:80-105` (CAPIConfig dataclass)
- Modify: `capi_config.py:226-227` (from_dict)
- Modify: `capi_config.py:298-299` (to_dict)
- Modify: `configs/capi_3f.yaml:17-19` (Otsu 裁剪設定區塊)
- Modify: `capi_inference.py:444-474` (`_find_raw_object_bounds`)
- Modify: `capi_inference.py:488-492` (`calculate_otsu_bounds` 呼叫端)

---

- [ ] **Step 1.1: 新增 `enable_panel_polygon` 欄位到 `CAPIConfig`**

修改 `capi_config.py`，在 `otsu_bottom_crop` 之後加一行（約 line 97 之後）：

```python
    # Otsu 裁剪設定
    otsu_offset: int = 5
    otsu_bottom_crop: int = 1000  # Otsu 後裁切底部像素

    # 面板 4 角 polygon 偵測（新增）
    enable_panel_polygon: bool = True  # 啟用後在 tile.mask 上套用 polygon 做精準裁切
```

同檔案 `from_dict`（約 line 226 附近）新增：

```python
            otsu_offset=data.get("otsu_offset", 5),
            otsu_bottom_crop=data.get("otsu_bottom_crop", 1000),
            enable_panel_polygon=data.get("enable_panel_polygon", True),
```

同檔案 `to_dict`（約 line 298 附近）新增：

```python
            "otsu_offset": self.otsu_offset,
            "otsu_bottom_crop": self.otsu_bottom_crop,
            "enable_panel_polygon": self.enable_panel_polygon,
```

- [ ] **Step 1.2: 新增 `enable_panel_polygon: true` 到 YAML**

修改 `configs/capi_3f.yaml`，在 `otsu_bottom_crop` 之後：

```yaml
# Otsu 裁剪設定
otsu_offset: 5  # 四周偏移 px
otsu_bottom_crop: 0  # Otsu 後裁切底部像素 (若設置 > 0，則自動禁用排除區域)

# 面板 4 角 polygon 偵測
enable_panel_polygon: true  # false 則完全退回 axis-aligned bbox 行為
```

- [ ] **Step 1.3: 寫 config round-trip 測試確保欄位不掉**

建立 `tests/test_panel_polygon_unit.py`（初始版本，後續 Task 2 會再補內容）：

```python
"""面板 4 角 polygon 功能的單元測試"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np
from capi_config import CAPIConfig


def test_config_enable_panel_polygon_default_true():
    """enable_panel_polygon 預設必須為 True"""
    cfg = CAPIConfig()
    assert cfg.enable_panel_polygon is True, \
        f"expected default True, got {cfg.enable_panel_polygon}"
    print("✅ test_config_enable_panel_polygon_default_true")


def test_config_roundtrip_enable_panel_polygon():
    """from_dict / to_dict 必須保留 enable_panel_polygon 欄位"""
    cfg1 = CAPIConfig()
    cfg1.enable_panel_polygon = False
    d = cfg1.to_dict()
    assert "enable_panel_polygon" in d
    assert d["enable_panel_polygon"] is False

    cfg2 = CAPIConfig.from_dict(d)
    assert cfg2.enable_panel_polygon is False
    print("✅ test_config_roundtrip_enable_panel_polygon")


def test_config_yaml_loads_enable_panel_polygon():
    """從實際 capi_3f.yaml 讀取應該能抓到 enable_panel_polygon"""
    yaml_path = Path(__file__).resolve().parent.parent / "configs" / "capi_3f.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "enable_panel_polygon" in data, \
        f"capi_3f.yaml 缺少 enable_panel_polygon 欄位"
    assert data["enable_panel_polygon"] is True
    print("✅ test_config_yaml_loads_enable_panel_polygon")


if __name__ == "__main__":
    test_config_enable_panel_polygon_default_true()
    test_config_roundtrip_enable_panel_polygon()
    test_config_yaml_loads_enable_panel_polygon()
    print("\n✅ 所有 config 測試通過")
```

- [ ] **Step 1.4: 跑測試確認 PASS**

```bash
cd /c/Users/rh.syu/Desktop/CAPI01_AD
python tests/test_panel_polygon_unit.py
```

Expected output:
```
✅ test_config_enable_panel_polygon_default_true
✅ test_config_roundtrip_enable_panel_polygon
✅ test_config_yaml_loads_enable_panel_polygon

✅ 所有 config 測試通過
```

- [ ] **Step 1.5: 修改 `_find_raw_object_bounds` 回傳 binary mask**

修改 `capi_inference.py:444-474`，將簽名與回傳改為：

```python
    def _find_raw_object_bounds(
        self, image: np.ndarray
    ) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
        """找尋物件的原始邊界 (不含 Offset)

        Returns:
            ((x_min, y_min, x_max, y_max), binary_mask) — binary_mask 是 Otsu +
            morphology close 後的 uint8 前景圖（255=前景），供後續 polygon 偵測重用。
        """
        img_height, img_width = image.shape[:2]

        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((15, 15), np.uint8)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf

        MIN_AREA = 1000
        for contour in contours:
            if cv2.contourArea(contour) > MIN_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x + w, x_max)
                y_max = max(y + h, y_max)

        if x_min == np.inf:
            return (0, 0, img_width, img_height), closing

        return (int(x_min), int(y_min), int(x_max), int(y_max)), closing
```

- [ ] **Step 1.6: 更新 `_find_raw_object_bounds` 的呼叫端**

`calculate_otsu_bounds` 約 line 492：

```python
        if reference_raw_bounds is not None:
            x_min, y_min, x_max, y_max = reference_raw_bounds
            binary_mask = None  # 使用參考邊界時無 binary mask
        else:
            (x_min, y_min, x_max, y_max), binary_mask = self._find_raw_object_bounds(image)
```

同檔案搜尋其他 `_find_raw_object_bounds` 呼叫（B0F 路徑約 line 2995）：

```bash
grep -n "_find_raw_object_bounds" capi_inference.py
```

將 B0F 路徑改成：

```python
                        ref_bounds, _ref_binary = self._find_raw_object_bounds(ref_img)
                        reference_raw_bounds_for_dark = ref_bounds
                        print(f"📐 黑圖參考邊界已從 {ref_path.name} 計算 → {reference_raw_bounds_for_dark}")
```

（`_ref_binary` 用底線前綴先接著，Task 6 才真正使用）

- [ ] **Step 1.7: 跑既有測試確認無 regression**

```bash
cd /c/Users/rh.syu/Desktop/CAPI01_AD
python tests/test_panel_polygon_unit.py
python tests/test_cv_edge.py
```

兩者都要成功跑完無錯誤。

- [ ] **Step 1.8: Commit**

```bash
git add capi_config.py configs/capi_3f.yaml capi_inference.py tests/test_panel_polygon_unit.py
git commit -m "$(cat <<'EOF'
feat: 加入 enable_panel_polygon config + _find_raw_object_bounds 回傳 binary

- capi_config.py 新增 enable_panel_polygon bool 欄位 (預設 True)
- configs/capi_3f.yaml 新增 enable_panel_polygon: true
- _find_raw_object_bounds 改為回傳 (bounds, binary_mask) 供後續 polygon 偵測重用
- tests/test_panel_polygon_unit.py 初始版本 (config round-trip 測試)

尚無行為變化，為 Task 2-8 鋪路。
EOF
)"
```

---

## Task 2: 新增 `_find_panel_polygon` 函式 + 單元測試

**目的**: 實作逐邊 polyfit + 相交的 4 角偵測演算法，完整覆蓋退化/理想/真實圖 3 類情境。此 task 產出的函式在 Task 3 才被 wire up 到 pipeline。

**Files:**
- Modify: `capi_inference.py` (在 `_find_raw_object_bounds` 後方新增 `_find_panel_polygon`)
- Modify: `tests/test_panel_polygon_unit.py` (擴充單元測試)

---

- [ ] **Step 2.1: 撰寫失敗的單元測試**

擴充 `tests/test_panel_polygon_unit.py`，在檔案底部（`if __name__ == "__main__":` 之前）新增：

```python
import cv2
from capi_inference import CAPIInferencer
from capi_config import CAPIConfig


def _make_inferencer():
    """建立一個不需要模型載入的 inferencer instance"""
    cfg = CAPIConfig()
    cfg.tile_size = 512
    return CAPIInferencer(cfg)


def test_polygon_detect_ideal_rectangle():
    """完美 axis-aligned 矩形 → 4 角應該與 bbox 4 角幾乎相同 (< 2 px 誤差)"""
    inf = _make_inferencer()
    # 建立 4000x3000 黑底，中心 (500,400)-(3500,2600) 白矩形
    binary = np.zeros((3000, 4000), dtype=np.uint8)
    binary[400:2600, 500:3500] = 255
    bbox = (500, 400, 3500, 2600)

    polygon = inf._find_panel_polygon(binary, bbox)
    assert polygon is not None, "理想矩形偵測不應該失敗"
    assert polygon.shape == (4, 2)
    assert polygon.dtype == np.float32

    expected = np.array([
        [500, 400],   # TL
        [3500, 400],  # TR
        [3500, 2600], # BR
        [500, 2600],  # BL
    ], dtype=np.float32)
    diff = np.abs(polygon - expected).max()
    assert diff < 2.0, f"ideal rect 誤差過大: {diff:.1f}px (per-corner max)"
    print(f"✅ test_polygon_detect_ideal_rectangle (max err={diff:.2f}px)")


def test_polygon_detect_degenerate_all_black():
    """全黑圖 → 應該回傳 None"""
    inf = _make_inferencer()
    binary = np.zeros((3000, 4000), dtype=np.uint8)
    bbox = (0, 0, 4000, 3000)
    polygon = inf._find_panel_polygon(binary, bbox)
    assert polygon is None, f"全黑圖應該回傳 None，實際 {polygon}"
    print("✅ test_polygon_detect_degenerate_all_black")


def test_polygon_detect_degenerate_tiny_noise():
    """只有小雜點 (面積太小) → 應該回傳 None (MIN_POLYGON_AREA_RATIO 檢查)"""
    inf = _make_inferencer()
    binary = np.zeros((3000, 4000), dtype=np.uint8)
    # 一個 100x100 白點，遠小於 bbox
    binary[1000:1100, 1000:1100] = 255
    bbox = (0, 0, 4000, 3000)  # 故意給大 bbox
    polygon = inf._find_panel_polygon(binary, bbox)
    assert polygon is None, f"小雜點應回傳 None，實際 {polygon}"
    print("✅ test_polygon_detect_degenerate_tiny_noise")


def test_polygon_detect_real_W0F():
    """真實影像 W0F00000_110022.tif 的 4 角誤差應該符合 spec 預期值"""
    img_path = Path(__file__).resolve().parent.parent / "test_images" / "W0F00000_110022.tif"
    if not img_path.exists():
        print(f"⚠️  跳過 (測試圖不存在): {img_path}")
        return

    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    inf = _make_inferencer()
    bbox, binary = inf._find_raw_object_bounds(gray)
    polygon = inf._find_panel_polygon(binary, bbox)
    assert polygon is not None

    # Spec 9.1 預期: TL=6.6, TR=15.7, BR=15.0, BL=0.6 (±2 px 容忍)
    bbox_corners = np.array([
        [bbox[0], bbox[1]], [bbox[2], bbox[1]],
        [bbox[2], bbox[3]], [bbox[0], bbox[3]],
    ], dtype=np.float32)
    errs = np.linalg.norm(polygon - bbox_corners, axis=1)
    expected = np.array([6.6, 15.7, 15.0, 0.6])
    diff = np.abs(errs - expected).max()
    assert diff < 2.0, \
        f"W0F 4 角誤差與 spec 不符: expected {expected}, got {errs}, max diff {diff:.2f}px"
    print(f"✅ test_polygon_detect_real_W0F (errs={errs.round(1).tolist()})")


def test_polygon_detect_real_G0F():
    """真實影像 G0F00000_151955.tif 的 4 角誤差應該符合 spec 預期值"""
    img_path = Path(__file__).resolve().parent.parent / "test_images" / "G0F00000_151955.tif"
    if not img_path.exists():
        print(f"⚠️  跳過 (測試圖不存在): {img_path}")
        return

    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    inf = _make_inferencer()
    bbox, binary = inf._find_raw_object_bounds(gray)
    polygon = inf._find_panel_polygon(binary, bbox)
    assert polygon is not None

    # Spec 9.1 預期: TL=16.7, TR=19.0, BR=36.9, BL=3.8 (±2 px 容忍)
    bbox_corners = np.array([
        [bbox[0], bbox[1]], [bbox[2], bbox[1]],
        [bbox[2], bbox[3]], [bbox[0], bbox[3]],
    ], dtype=np.float32)
    errs = np.linalg.norm(polygon - bbox_corners, axis=1)
    expected = np.array([16.7, 19.0, 36.9, 3.8])
    diff = np.abs(errs - expected).max()
    assert diff < 2.0, \
        f"G0F 4 角誤差與 spec 不符: expected {expected}, got {errs}, max diff {diff:.2f}px"
    print(f"✅ test_polygon_detect_real_G0F (errs={errs.round(1).tolist()})")


def test_polygon_corner_ordering():
    """4 角順序必須是 TL, TR, BR, BL"""
    inf = _make_inferencer()
    binary = np.zeros((2000, 3000), dtype=np.uint8)
    binary[300:1700, 500:2500] = 255
    bbox = (500, 300, 2500, 1700)
    polygon = inf._find_panel_polygon(binary, bbox)
    assert polygon is not None
    TL, TR, BR, BL = polygon
    assert TL[0] < TR[0], f"TL.x ({TL[0]}) 必須 < TR.x ({TR[0]})"
    assert BL[0] < BR[0], f"BL.x ({BL[0]}) 必須 < BR.x ({BR[0]})"
    assert TL[1] < BL[1], f"TL.y ({TL[1]}) 必須 < BL.y ({BL[1]})"
    assert TR[1] < BR[1], f"TR.y ({TR[1]}) 必須 < BR.y ({BR[1]})"
    print("✅ test_polygon_corner_ordering")
```

同時把 `if __name__ == "__main__":` 區塊更新成：

```python
if __name__ == "__main__":
    test_config_enable_panel_polygon_default_true()
    test_config_roundtrip_enable_panel_polygon()
    test_config_yaml_loads_enable_panel_polygon()

    test_polygon_detect_ideal_rectangle()
    test_polygon_detect_degenerate_all_black()
    test_polygon_detect_degenerate_tiny_noise()
    test_polygon_detect_real_W0F()
    test_polygon_detect_real_G0F()
    test_polygon_corner_ordering()

    print("\n✅ 所有測試通過")
```

- [ ] **Step 2.2: 跑測試確認全部失敗 (因為函式尚未實作)**

```bash
cd /c/Users/rh.syu/Desktop/CAPI01_AD
python tests/test_panel_polygon_unit.py
```

Expected: `AttributeError: 'CAPIInferencer' object has no attribute '_find_panel_polygon'`

- [ ] **Step 2.3: 實作 `_find_panel_polygon`**

在 `capi_inference.py` 的 `_find_raw_object_bounds` 函式之後（約 line 475），新增：

```python
    def _find_panel_polygon(
        self,
        binary_mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        在既有 Otsu binary mask 上用逐邊 polyfit 找 4 角 panel polygon。

        演算法:
          1. 逐邊 (top/bottom/left/right) 取樣前景邊緣點
          2. 每邊用 np.polyfit 擬合直線 + 3σ robust filter
          3. 4 線兩兩相交求 4 角 (TL/TR/BR/BL)
          4. 品質檢查 (面積、邊長、線近乎平行) — 任一失敗回傳 None

        Args:
            binary_mask: Otsu + morphology close 後的 uint8 前景圖 (255=前景)
            bbox: 粗略 bbox (x_min, y_min, x_max, y_max)，作為邊緣掃描範圍

        Returns:
            np.ndarray shape (4,2) float32，順序 [TL, TR, BR, BL]；
            偵測失敗或品質不足回傳 None。
        """
        # 常數 (hardcode，不入 config)
        EDGE_MARGIN = 20
        SAMPLE_STEP = 50
        OUTLIER_SIGMA = 3.0
        MIN_EDGE_LEN_RATIO = 1.0          # 相對 tile_size
        MIN_POLYGON_AREA_RATIO = 0.9      # 相對 bbox 面積
        MIN_SAMPLES_PER_EDGE = 5

        if binary_mask is None or binary_mask.size == 0:
            return None

        H, W = binary_mask.shape[:2]
        xmin, ymin, xmax, ymax = bbox
        if xmax - xmin < 2 * EDGE_MARGIN or ymax - ymin < 2 * EDGE_MARGIN:
            return None

        # --- Step 1: 逐邊掃描 ---
        tops, bots = [], []
        for x in range(xmin + EDGE_MARGIN, xmax - EDGE_MARGIN, SAMPLE_STEP):
            if x < 0 or x >= W:
                continue
            col = binary_mask[:, x]
            ys = np.where(col > 0)[0]
            if len(ys) > 0:
                tops.append((x, int(ys[0])))
                bots.append((x, int(ys[-1])))

        lefts, rights = [], []
        for y in range(ymin + EDGE_MARGIN, ymax - EDGE_MARGIN, SAMPLE_STEP):
            if y < 0 or y >= H:
                continue
            row = binary_mask[y, :]
            xs = np.where(row > 0)[0]
            if len(xs) > 0:
                lefts.append((int(xs[0]), y))
                rights.append((int(xs[-1]), y))

        if (len(tops) < MIN_SAMPLES_PER_EDGE or len(bots) < MIN_SAMPLES_PER_EDGE
                or len(lefts) < MIN_SAMPLES_PER_EDGE or len(rights) < MIN_SAMPLES_PER_EDGE):
            return None

        # --- Step 2: 每邊 polyfit + 3σ robust filter ---
        def fit_line_robust(pts, horizontal: bool) -> Optional[Tuple[float, float]]:
            """horizontal=True: 回傳 (a, b) 代表 y = a*x + b；否則代表 x = a*y + b"""
            arr = np.array(pts, dtype=float)
            if horizontal:
                ind = arr[:, 0]; dep = arr[:, 1]
            else:
                ind = arr[:, 1]; dep = arr[:, 0]
            try:
                a, b = np.polyfit(ind, dep, 1)
            except (np.linalg.LinAlgError, ValueError):
                return None
            residuals = dep - (a * ind + b)
            sigma = float(residuals.std())
            if sigma > 0:
                keep = np.abs(residuals) < OUTLIER_SIGMA * sigma
                if keep.sum() >= 3:
                    try:
                        a, b = np.polyfit(ind[keep], dep[keep], 1)
                    except (np.linalg.LinAlgError, ValueError):
                        pass
            return (float(a), float(b))

        top_line = fit_line_robust(tops, horizontal=True)
        bot_line = fit_line_robust(bots, horizontal=True)
        left_line = fit_line_robust(lefts, horizontal=False)
        right_line = fit_line_robust(rights, horizontal=False)
        if None in (top_line, bot_line, left_line, right_line):
            return None

        # --- Step 3: 4 線相交 ---
        def intersect_hv(h_line, v_line):
            # h: y = a_h*x + b_h ; v: x = a_v*y + b_v
            a_h, b_h = h_line
            a_v, b_v = v_line
            denom = 1.0 - a_h * a_v
            if abs(denom) < 1e-9:
                return None
            y = (a_h * b_v + b_h) / denom
            x = a_v * y + b_v
            return (x, y)

        TL = intersect_hv(top_line, left_line)
        TR = intersect_hv(top_line, right_line)
        BR = intersect_hv(bot_line, right_line)
        BL = intersect_hv(bot_line, left_line)
        if None in (TL, TR, BR, BL):
            return None

        polygon = np.array([TL, TR, BR, BL], dtype=np.float32)

        # --- Step 4: 品質檢查 ---
        # 4a. 所有角必須大致在 image 範圍內 (容忍 50 px 溢出)
        tol = 50
        if (polygon[:, 0].min() < -tol or polygon[:, 0].max() > W + tol or
                polygon[:, 1].min() < -tol or polygon[:, 1].max() > H + tol):
            return None

        # 4b. 邊長必須 > tile_size
        tile_size = self.config.tile_size
        min_edge_len = tile_size * MIN_EDGE_LEN_RATIO
        for i in range(4):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % 4]
            edge_len = float(np.linalg.norm(p2 - p1))
            if edge_len < min_edge_len:
                return None

        # 4c. polygon 面積必須 >= bbox 面積 * 0.9
        bbox_area = float((xmax - xmin) * (ymax - ymin))
        poly_area = float(cv2.contourArea(polygon))
        if bbox_area <= 0 or poly_area < bbox_area * MIN_POLYGON_AREA_RATIO:
            return None

        return polygon
```

- [ ] **Step 2.4: 跑測試確認全部通過**

```bash
cd /c/Users/rh.syu/Desktop/CAPI01_AD
python tests/test_panel_polygon_unit.py
```

Expected output (略去前面已通過的):
```
✅ test_polygon_detect_ideal_rectangle (max err=...)
✅ test_polygon_detect_degenerate_all_black
✅ test_polygon_detect_degenerate_tiny_noise
✅ test_polygon_detect_real_W0F (errs=[6.x, 15.x, 15.x, 0.x])
✅ test_polygon_detect_real_G0F (errs=[16.x, 19.x, 36.x, 3.x])
✅ test_polygon_corner_ordering

✅ 所有測試通過
```

若 W0F/G0F 測試因容忍度失敗，**不要調整函式的常數** — 先驗證數字是否與之前 `test_panel_polygon_detect.py` 跑出的一致（先前測試是 TL=6.6, TR=15.7 等），確認是容忍度問題還是演算法差異。

- [ ] **Step 2.5: Commit**

```bash
git add capi_inference.py tests/test_panel_polygon_unit.py
git commit -m "$(cat <<'EOF'
feat: 實作 _find_panel_polygon 4 角面板偵測

逐邊掃描前景邊緣點 → np.polyfit 擬合 + 3σ robust filter → 4 線相交
得到 [TL, TR, BR, BL] 4 角。品質檢查含 4 角範圍、邊長、面積比例，
任一失敗回傳 None 供下游 fallback 到 axis-aligned bbox 行為。

單元測試涵蓋: 理想矩形、全黑、小雜點、真實 W0F/G0F 影像、4 角順序。
尚未接入 pipeline — Task 3 才 wire up 到 ImageResult / tile_image。
EOF
)"
```

---

## Task 3: 把 polygon 接到 `ImageResult` + `calculate_otsu_bounds` / `preprocess_image` 路徑

**目的**: 讓 polygon 在 pipeline 裡被計算並存到 `ImageResult`，但還沒被 `tile_image` 用到。此 task 驗證 polygon 能正確傳遞。

**Files:**
- Modify: `capi_inference.py:192-228` (`ImageResult` dataclass)
- Modify: `capi_inference.py:476-515` (`calculate_otsu_bounds`)
- Modify: `capi_inference.py:517-523` (`find_panel_boundaries` 向後相容別名)
- Modify: `capi_inference.py:734-790` (`preprocess_image`)
- Modify: `tests/test_panel_polygon_unit.py` (新增 pipeline 整合測試)

---

- [ ] **Step 3.1: 新增 `ImageResult.panel_polygon` 欄位**

修改 `capi_inference.py:192-228`，在 `raw_bounds` 之後新增：

```python
    # 原始物件邊界 (用於 AOI 座標映射，避免重複讀取圖片)
    raw_bounds: Optional[Tuple[int, int, int, int]] = None

    # 面板 4 角 polygon (shape (4,2) float32，順序 TL/TR/BR/BL)
    # None 代表 polygon 偵測失敗或未啟用，下游應 fallback 回 axis-aligned bbox
    panel_polygon: Optional[np.ndarray] = field(default=None, repr=False)
```

- [ ] **Step 3.2: 修改 `calculate_otsu_bounds` 回傳 polygon**

修改 `capi_inference.py:476-515`：

```python
    def calculate_otsu_bounds(
        self,
        image: np.ndarray,
        otsu_offset_override: Optional[int] = None,
        reference_raw_bounds: Optional[Tuple[int, int, int, int]] = None,
        reference_polygon: Optional[np.ndarray] = None,
    ) -> Tuple[Tuple[int, int, int, int], Optional[int], Optional[np.ndarray]]:
        """
        計算 Otsu 前景邊界與 panel polygon。

        Args:
            reference_raw_bounds: 參考用的原始邊界 (來自同資料夾的白圖)。
                                  黑圖 (B0F) OTSU 無法正確偵測邊界時使用。
            reference_polygon: 參考用的 panel polygon (來自同資料夾的白圖)，
                               與 reference_raw_bounds 同時使用於 B0F fallback。
        Returns:
            (final_bounds, original_y2, panel_polygon)
        """
        img_height, img_width = image.shape[:2]

        # 取得原始物件邊界與 binary mask
        if reference_raw_bounds is not None:
            x_min, y_min, x_max, y_max = reference_raw_bounds
            binary_mask = None
        else:
            (x_min, y_min, x_max, y_max), binary_mask = self._find_raw_object_bounds(image)

        offset = otsu_offset_override if otsu_offset_override is not None else self.config.otsu_offset
        x_start = max(0, int(x_min) + offset)
        y_start = max(0, int(y_min) + offset)
        x_end = min(img_width, int(x_max) - offset)
        y_end = min(img_height, int(y_max) - offset)

        if x_start >= x_end or y_start >= y_end:
            x_start, y_start = 0, 0
            x_end, y_end = img_width, img_height

        # 應用底部裁切 (otsu_bottom_crop)
        original_y2 = None
        if self.config.otsu_bottom_crop > 0:
            h = y_end - y_start
            desired_height = max(self.config.tile_size, h - self.config.otsu_bottom_crop)
            final_height = min(h, desired_height)

            if final_height < h:
                original_y2 = y_end
                y_end = y_start + final_height

        bounds = (x_start, y_start, x_end, y_end)

        # 計算 panel polygon
        panel_polygon: Optional[np.ndarray] = None
        if self.config.enable_panel_polygon:
            if reference_polygon is not None:
                panel_polygon = reference_polygon.copy()
            elif binary_mask is not None:
                # 使用原始 (未內縮) bbox 做邊緣掃描
                raw_bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
                panel_polygon = self._find_panel_polygon(binary_mask, raw_bbox)

            # 若 polygon 存在且使用者有設 otsu_offset，對 polygon 做同向內縮
            if panel_polygon is not None and offset != 0:
                cx = (panel_polygon[:, 0].mean())
                cy = (panel_polygon[:, 1].mean())
                # 朝中心點內縮 offset px
                for i in range(4):
                    dx = panel_polygon[i, 0] - cx
                    dy = panel_polygon[i, 1] - cy
                    length = float(np.hypot(dx, dy))
                    if length > 1e-6:
                        shrink = offset / length
                        panel_polygon[i, 0] -= dx * shrink
                        panel_polygon[i, 1] -= dy * shrink

            # 若 polygon 存在且啟用 otsu_bottom_crop，截掉下半部
            if panel_polygon is not None and original_y2 is not None:
                new_bottom = float(y_end)
                for i in (2, 3):  # BR, BL
                    if panel_polygon[i, 1] > new_bottom:
                        panel_polygon[i, 1] = new_bottom

        return bounds, original_y2, panel_polygon
```

- [ ] **Step 3.3: 修改向後相容別名 `find_panel_boundaries`**

修改 `capi_inference.py:517-523`：

```python
    def find_panel_boundaries(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        向後相容別名，對應舊版呼叫。
        注意：此版本僅回傳四元組邊界，不回傳 original_y2 / polygon。
        """
        bounds, _, _ = self.calculate_otsu_bounds(image)
        return bounds
```

- [ ] **Step 3.4: 修改 `preprocess_image` 傳遞 polygon**

修改 `capi_inference.py:734-790`：

```python
    def preprocess_image(
        self,
        image_path: Path,
        cached_mark: Optional[ExclusionRegion] = None,
        otsu_offset_override: Optional[int] = None,
        reference_raw_bounds: Optional[Tuple[int, int, int, int]] = None,
        reference_polygon: Optional[np.ndarray] = None,
    ) -> Optional[ImageResult]:
        """預處理圖片：Otsu + 排除區域 + 切塊

        Args:
            image_path: 圖片路徑
            cached_mark: 快取的 MARK 區域
            otsu_offset_override: Debug 用 Otsu 內縮覆寫值 (px)
            reference_raw_bounds: 參考用的原始邊界 (來自同資料夾的白圖)。
                                  黑圖 (B0F) OTSU 無法正確偵測邊界時使用。
            reference_polygon: 參考用的 panel polygon，與 reference_raw_bounds 同
                               時使用於 B0F fallback。
        """
```

在函式內部找到 `otsu_bounds, original_y2 = self.calculate_otsu_bounds(...)` 這一行（約 line 767），改為：

```python
        # 黑圖 (B0F) 使用參考邊界，因為 OTSU 無法正確偵測全黑畫面的邊界
        if reference_raw_bounds is not None:
            raw_bounds = reference_raw_bounds
        else:
            raw_bounds, _ = self._find_raw_object_bounds(image)

        # Otsu 裁切 (同樣使用參考邊界)
        otsu_bounds, original_y2, panel_polygon = self.calculate_otsu_bounds(
            image,
            otsu_offset_override=otsu_offset_override,
            reference_raw_bounds=reference_raw_bounds,
            reference_polygon=reference_polygon,
        )
```

在 `ImageResult(...)` 的建構位置（約 line 786），新增 `panel_polygon=panel_polygon`：

```python
        result = ImageResult(
            image_path=image_path,
            image_size=(image.shape[1], image.shape[0]),
            otsu_bounds=otsu_bounds,
            exclusion_regions=exclusion_regions,
            tiles=tiles,
            excluded_tile_count=excluded_count,
            processed_tile_count=len(tiles),
            processing_time=0.0,
            cropped_region=cropped_region,
            raw_bounds=raw_bounds,
            panel_polygon=panel_polygon,
        )
```

- [ ] **Step 3.5: 修改單元測試，新增 pipeline 整合測試**

在 `tests/test_panel_polygon_unit.py` 底部（`if __name__ == "__main__":` 之前）新增：

```python
def test_preprocess_image_populates_panel_polygon():
    """preprocess_image 跑完後 result.panel_polygon 必須是 (4,2) float32"""
    img_path = Path(__file__).resolve().parent.parent / "test_images" / "G0F00000_151955.tif"
    if not img_path.exists():
        print(f"⚠️  跳過 (測試圖不存在): {img_path}")
        return

    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.tile_stride = 512
    cfg.otsu_bottom_crop = 0  # 不做 bottom crop 以便直接比對
    inf = CAPIInferencer(cfg)
    result = inf.preprocess_image(img_path)
    assert result is not None
    assert result.panel_polygon is not None, "polygon 應該要被計算"
    assert result.panel_polygon.shape == (4, 2)
    assert result.panel_polygon.dtype == np.float32
    print(f"✅ test_preprocess_image_populates_panel_polygon "
          f"(polygon={result.panel_polygon.round(1).tolist()})")


def test_preprocess_image_polygon_disabled_when_toggle_off():
    """enable_panel_polygon=False 時 panel_polygon 必須為 None"""
    img_path = Path(__file__).resolve().parent.parent / "test_images" / "G0F00000_151955.tif"
    if not img_path.exists():
        print(f"⚠️  跳過 (測試圖不存在): {img_path}")
        return

    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.tile_stride = 512
    cfg.otsu_bottom_crop = 0
    cfg.enable_panel_polygon = False
    inf = CAPIInferencer(cfg)
    result = inf.preprocess_image(img_path)
    assert result is not None
    assert result.panel_polygon is None, \
        f"toggle off 時 polygon 應為 None，實際 {result.panel_polygon}"
    print("✅ test_preprocess_image_polygon_disabled_when_toggle_off")
```

並把這兩個測試加到 `if __name__ == "__main__":` block。

- [ ] **Step 3.6: 跑測試確認**

```bash
cd /c/Users/rh.syu/Desktop/CAPI01_AD
python tests/test_panel_polygon_unit.py
```

全部通過 (若 test_images 不存在會跳過真實圖測試並 print 警告)。

- [ ] **Step 3.7: Commit**

```bash
git add capi_inference.py tests/test_panel_polygon_unit.py
git commit -m "$(cat <<'EOF'
feat: ImageResult 新增 panel_polygon 欄位並接入 preprocess_image 路徑

- ImageResult 新增 panel_polygon: Optional[np.ndarray] 欄位
- calculate_otsu_bounds 回傳第三值 polygon，並對它做 otsu_offset 內縮
- preprocess_image 接受 reference_polygon 並把 polygon 寫入 ImageResult
- enable_panel_polygon=False 時 polygon 為 None (完全向後相容)

尚未影響 tile.mask — 下個 task (tile_image) 才會套用。
EOF
)"
```

---

## Task 4: `tile_image` 套用 polygon mask + tile mask 視覺回歸測試

**目的**: 最關鍵的一步 — 讓每個 tile 根據 polygon 交集產生正確的 `tile.mask`，並透過現有的 `capi_inference.py:1126-1133` 自動生效。

**Files:**
- Modify: `capi_inference.py:88` (`TileInfo.mask` comment)
- Modify: `capi_inference.py:632-732` (`tile_image`)
- Modify: `capi_inference.py:779-783` (`preprocess_image` 呼叫 `tile_image`)
- Create: `tests/test_panel_polygon_tile_mask.py`

---

- [ ] **Step 4.1: 撰寫失敗的視覺回歸測試**

建立 `tests/test_panel_polygon_tile_mask.py`：

```python
"""tile_image 套用 panel polygon mask 的視覺回歸測試"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from capi_config import CAPIConfig
from capi_inference import CAPIInferencer


TEST_IMAGES = [
    "test_images/W0F00000_110022.tif",
    "test_images/G0F00000_151955.tif",
]

OUT_DIR = Path(__file__).resolve().parent.parent / "test_output" / "panel_polygon_tile_mask"


def _run_one(rel_path: str):
    img_path = Path(__file__).resolve().parent.parent / rel_path
    if not img_path.exists():
        print(f"⚠️  跳過 (測試圖不存在): {img_path}")
        return None, None

    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.tile_stride = 512
    cfg.otsu_bottom_crop = 0
    cfg.otsu_offset = 0  # 測試用: 不內縮以便精準比對 polygon 邊緣
    inf = CAPIInferencer(cfg)
    result = inf.preprocess_image(img_path)
    assert result is not None
    return result, img_path


def test_tile_mask_shapes_and_dtypes():
    """每個 tile 的 mask 要不是 None，要不是 shape (tile_size, tile_size) uint8"""
    for rel in TEST_IMAGES:
        result, img_path = _run_one(rel)
        if result is None:
            continue
        if result.panel_polygon is None:
            print(f"⚠️  {rel} polygon 偵測失敗，跳過 mask 檢查")
            continue
        for tile in result.tiles:
            if tile.mask is None:
                continue
            assert tile.mask.dtype == np.uint8, \
                f"tile {tile.tile_id} mask dtype {tile.mask.dtype}"
            assert tile.mask.shape == (tile.height, tile.width), \
                f"tile {tile.tile_id} mask shape {tile.mask.shape}"
        print(f"✅ test_tile_mask_shapes_and_dtypes / {Path(rel).name}")


def test_tile_mask_matches_polygon():
    """
    對每個有 mask 的 tile: mask==0 的像素必須在 polygon 外；
    mask==255 的像素必須在 polygon 內。
    """
    for rel in TEST_IMAGES:
        result, img_path = _run_one(rel)
        if result is None or result.panel_polygon is None:
            continue

        # 建立整張圖尺寸的 panel_mask ground truth
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        H, W = gray.shape
        gt_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(gt_mask, [result.panel_polygon.astype(np.int32)], 255)

        masked_tiles = 0
        for tile in result.tiles:
            if tile.mask is None:
                continue
            masked_tiles += 1
            gt_sub = gt_mask[tile.y:tile.y + tile.height,
                             tile.x:tile.x + tile.width]
            # Mask 應該與 ground truth 完全一致 (fillPoly 是 deterministic)
            mismatch = int(np.count_nonzero(tile.mask != gt_sub))
            total = tile.mask.size
            assert mismatch / total < 0.001, \
                f"tile {tile.tile_id} mask mismatch {mismatch}/{total} " \
                f"({mismatch/total:.3%})"

        print(f"✅ test_tile_mask_matches_polygon / {Path(rel).name} "
              f"({masked_tiles} 個邊緣 tile 有 mask)")


def test_tile_mask_visualization():
    """產生視覺化 PNG 供人工確認 (非 assertion 測試)"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rel in TEST_IMAGES:
        result, img_path = _run_one(rel)
        if result is None or result.panel_polygon is None:
            continue
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        overlay = vis.copy()

        for tile in result.tiles:
            # 畫 tile 框 (綠)
            cv2.rectangle(vis, (tile.x, tile.y),
                          (tile.x + tile.width, tile.y + tile.height),
                          (0, 255, 0), 3)
            if tile.mask is not None:
                excluded = tile.mask == 0
                if excluded.any():
                    roi = overlay[tile.y:tile.y + tile.height,
                                  tile.x:tile.x + tile.width]
                    roi[excluded] = (0, 0, 255)

        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
        # Polygon (黃) + bbox (藍)
        poly_int = result.panel_polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly_int], True, (0, 255, 255), 4)
        x1, y1, x2, y2 = result.otsu_bounds
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 4)

        scale = 0.25
        vis_small = cv2.resize(vis, (0, 0), fx=scale, fy=scale)
        out_path = OUT_DIR / f"{Path(rel).stem}_tile_mask.png"
        cv2.imwrite(str(out_path), vis_small)
        print(f"✅ 視覺化輸出: {out_path}")


if __name__ == "__main__":
    test_tile_mask_shapes_and_dtypes()
    test_tile_mask_matches_polygon()
    test_tile_mask_visualization()
    print("\n✅ 所有 tile mask 測試通過")
```

- [ ] **Step 4.2: 跑測試確認失敗（因為 `tile_image` 尚未套 polygon）**

```bash
cd /c/Users/rh.syu/Desktop/CAPI01_AD
python tests/test_panel_polygon_tile_mask.py
```

Expected: `test_tile_mask_matches_polygon` 失敗（目前所有 tile 的 `mask` 都是 None，或 mismatch 很大），或看到 `masked_tiles = 0` 的 assertion。

- [ ] **Step 4.3: 更新 `TileInfo.mask` 註記**

修改 `capi_inference.py:88`，把註記從「不再使用遮罩」改為反映新用途：

```python
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # 遮罩: 255=panel 內, 0=panel 外 (tile 完全在 polygon 內時為 None)
```

同時搜尋檔案內是否有「不再使用遮罩」註解並一併更新或刪除：

```bash
grep -n "不再使用遮罩" capi_inference.py
```

- [ ] **Step 4.4: 修改 `tile_image` 接受 polygon 並產生 mask**

修改 `capi_inference.py:632-732`，簽名加上 `panel_polygon`，並在迴圈中為每個 tile 計算 mask。

簽名：

```python
    def tile_image(
        self,
        image: np.ndarray,
        otsu_bounds: Tuple[int, int, int, int],
        exclusion_regions: List[ExclusionRegion],
        panel_polygon: Optional[np.ndarray] = None,
        exclusion_threshold: float = 0.0,
    ) -> Tuple[List[TileInfo], int]:
```

在 `tile_size = self.config.tile_size` 附近新增「準備 polygon in tile coords 時用到的資料」：

```python
        otsu_x1, otsu_y1, otsu_x2, otsu_y2 = otsu_bounds
        tile_size = self.config.tile_size
        stride = self.config.tile_stride

        # 預先把 polygon 轉成 int32 供 cv2.fillPoly 使用
        polygon_int: Optional[np.ndarray] = None
        if panel_polygon is not None:
            polygon_int = panel_polygon.astype(np.int32)
```

在 for loop 建 `TileInfo` 之前（`tile_img = image[y:tile_y2, x:tile_x2].copy()` 之後，`tiles.append(TileInfo(...))` 之前），新增：

```python
                # 擷取 tile 圖片
                tile_img = image[y:tile_y2, x:tile_x2].copy()

                # 計算 tile 的 panel mask (polygon 交集)
                tile_mask: Optional[np.ndarray] = None
                if polygon_int is not None:
                    mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
                    # 把 polygon 平移到 tile 座標系
                    shifted = polygon_int - np.array([x, y], dtype=np.int32)
                    cv2.fillPoly(mask, [shifted], 255)

                    if mask.max() == 0:
                        # Tile 完全在 polygon 外 → 跳過
                        excluded_count += 1
                        continue
                    if mask.min() == 255:
                        # Tile 完全在 polygon 內 → 省記憶體
                        tile_mask = None
                    else:
                        tile_mask = mask
```

然後把 `TileInfo(...)` 的 `mask=None` 改為 `mask=tile_mask`：

```python
                tiles.append(TileInfo(
                    tile_id=tile_id,
                    x=x,
                    y=y,
                    width=tile_size,
                    height=tile_size,
                    image=tile_img,
                    mask=tile_mask,
                    has_exclusion=False,
                    is_bottom_edge=is_bottom,
                    is_top_edge=is_top,
                    is_left_edge=is_left,
                    is_right_edge=is_right,
                ))
```

- [ ] **Step 4.5: 更新 `preprocess_image` 裡 `tile_image` 的呼叫**

修改 `capi_inference.py` 中的 `preprocess_image`，找到 `tile_image(...)` 的呼叫（約 line 779），改為傳入 polygon：

```python
        tiles, excluded_count = self.tile_image(
            image, otsu_bounds, exclusion_regions,
            panel_polygon=panel_polygon,
        )
```

- [ ] **Step 4.6: 跑視覺回歸測試**

```bash
cd /c/Users/rh.syu/Desktop/CAPI01_AD
python tests/test_panel_polygon_tile_mask.py
```

Expected: 全部測試通過，`test_output/panel_polygon_tile_mask/*.png` 產出 2 張視覺化。**人工打開這兩張 PNG 確認紅色 (被 mask 掉的區域) 在 panel 之外**。

- [ ] **Step 4.7: 跑既有單元測試確認沒壞**

```bash
python tests/test_panel_polygon_unit.py
```

- [ ] **Step 4.8: Commit**

```bash
git add capi_inference.py tests/test_panel_polygon_tile_mask.py
git commit -m "$(cat <<'EOF'
feat: tile_image 套用 panel polygon mask 做精準裁切

- tile_image 接受新參數 panel_polygon
- 每個 tile 用 cv2.fillPoly(polygon_in_tile_coords) 產生 tile.mask
- tile 完全在 polygon 外 → 跳過 (excluded_count++)
- tile 完全在 polygon 內 → mask=None 省記憶體
- 部分交集 → mask 為 uint8 255/0，由下游 :1126/:1164/:1230 自動套用

下游 PatchCore 推論、B0F 亮點、dust filter 等既有邏輯透過
`if tile.mask is not None` 路徑自動生效，無需修改。

視覺回歸測試輸出到 test_output/panel_polygon_tile_mask/
EOF
)"
```

---

## Task 5: `calculate_exclusion_regions` 用 polygon BR 做錨點

**目的**: 讓 MARK 的 `relative_bottom_right` 排除區改用 polygon BR 角，避免 bbox 右下偏差導致排除區歪掉。

**Files:**
- Modify: `capi_inference.py:586-630` (`calculate_exclusion_regions`)
- Modify: `capi_inference.py` 內 `preprocess_image` 裡 `calculate_exclusion_regions` 的呼叫
- Modify: `tests/test_panel_polygon_unit.py` (新增錨點測試)

---

- [ ] **Step 5.1: 撰寫失敗的測試**

在 `tests/test_panel_polygon_unit.py` 底部新增：

```python
def test_exclusion_region_uses_polygon_br_anchor():
    """
    relative_bottom_right 排除區應以 polygon BR 為錨點，
    而不是 bbox 右下角。
    """
    from capi_config import ExclusionZone

    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.tile_stride = 512
    cfg.otsu_bottom_crop = 0
    cfg.otsu_offset = 0
    cfg.exclusion_zones = [
        ExclusionZone(
            name="test_br",
            type="relative_bottom_right",
            width=300,
            height=200,
            enabled=True,
        ),
    ]
    inf = CAPIInferencer(cfg)

    img_path = Path(__file__).resolve().parent.parent / "test_images" / "G0F00000_151955.tif"
    if not img_path.exists():
        print(f"⚠️  跳過 (測試圖不存在): {img_path}")
        return

    result = inf.preprocess_image(img_path)
    assert result is not None
    assert result.panel_polygon is not None

    # 找到 test_br 排除區
    br_regions = [r for r in result.exclusion_regions if r.name == "test_br"]
    assert len(br_regions) == 1, f"應該找到 1 個 test_br 排除區，實際 {len(br_regions)}"
    br = br_regions[0]

    # 排除區的 x2/y2 必須接近 polygon BR，不能是 bbox 右下
    poly_br_x, poly_br_y = int(round(result.panel_polygon[2][0])), int(round(result.panel_polygon[2][1]))
    bbox_x2, bbox_y2 = result.otsu_bounds[2], result.otsu_bounds[3]

    # G0F 這張 polygon BR 與 bbox 右下差 ~37 px
    assert abs(br.x2 - poly_br_x) <= 1, \
        f"排除區 x2={br.x2} 應接近 polygon BR x={poly_br_x}，而非 bbox x2={bbox_x2}"
    assert abs(br.y2 - poly_br_y) <= 1, \
        f"排除區 y2={br.y2} 應接近 polygon BR y={poly_br_y}，而非 bbox y2={bbox_y2}"
    print(f"✅ test_exclusion_region_uses_polygon_br_anchor "
          f"(br=({br.x2},{br.y2}), poly BR=({poly_br_x},{poly_br_y}), "
          f"bbox BR=({bbox_x2},{bbox_y2}))")
```

並加入 `if __name__ == "__main__":` block。

- [ ] **Step 5.2: 跑測試確認失敗**

```bash
cd /c/Users/rh.syu/Desktop/CAPI01_AD
python tests/test_panel_polygon_unit.py
```

Expected: `test_exclusion_region_uses_polygon_br_anchor` 失敗，`br.x2` 等於 bbox x2 而非 polygon BR x。

- [ ] **Step 5.3: 修改 `calculate_exclusion_regions`**

修改 `capi_inference.py:586-630`，加上 `panel_polygon` 參數並改寫 `relative_bottom_right` 區段：

```python
    def calculate_exclusion_regions(
        self,
        image: np.ndarray,
        otsu_bounds: Tuple[int, int, int, int],
        cached_mark: Optional[ExclusionRegion] = None,
        panel_polygon: Optional[np.ndarray] = None,
    ) -> List[ExclusionRegion]:
        """計算所有排除區域

        Args:
            image: 原始圖片
            otsu_bounds: Otsu 邊界
            cached_mark: 快取的 MARK 區域（Panel 級共用），若提供則跳過模板匹配
            panel_polygon: 面板 4 角 polygon，若提供則 relative_bottom_right 以
                           polygon BR 為錨點
        """
        if self.config.otsu_bottom_crop > 0:
            return []

        regions = []
        otsu_x1, otsu_y1, otsu_x2, otsu_y2 = otsu_bounds

        for zone in self.config.get_enabled_exclusion_zones():
            if zone.type == "template_match" and zone.name == "mark_area":
                if cached_mark is not None:
                    regions.append(cached_mark)
                else:
                    mark_region = self.find_mark_region(image)
                    if mark_region:
                        regions.append(mark_region)

            elif zone.type == "relative_bottom_right":
                # 錨點: polygon BR 優先，否則回退到 bbox 右下
                if panel_polygon is not None:
                    anchor_x = int(round(float(panel_polygon[2][0])))
                    anchor_y = int(round(float(panel_polygon[2][1])))
                else:
                    anchor_x = otsu_x2
                    anchor_y = otsu_y2

                br_x1 = max(otsu_x1, anchor_x - zone.width)
                br_y1 = max(otsu_y1, anchor_y - zone.height)
                regions.append(ExclusionRegion(
                    name=zone.name,
                    x1=br_x1,
                    y1=br_y1,
                    x2=anchor_x,
                    y2=anchor_y,
                ))

        return regions
```

- [ ] **Step 5.4: 更新 `preprocess_image` 裡的呼叫**

`capi_inference.py` 內 `preprocess_image` 呼叫 `calculate_exclusion_regions` 的位置（約 line 776）：

```python
        exclusion_regions = self.calculate_exclusion_regions(
            image, otsu_bounds,
            cached_mark=cached_mark,
            panel_polygon=panel_polygon,
        )
```

- [ ] **Step 5.5: 跑測試確認通過**

```bash
python tests/test_panel_polygon_unit.py
python tests/test_panel_polygon_tile_mask.py
```

- [ ] **Step 5.6: Commit**

```bash
git add capi_inference.py tests/test_panel_polygon_unit.py
git commit -m "$(cat <<'EOF'
feat: relative_bottom_right 排除區改用 polygon BR 角為錨點

- calculate_exclusion_regions 接受 panel_polygon 參數
- relative_bottom_right 類型現在以 polygon[2] (BR) 為錨點
- polygon 為 None 時完全退回原行為 (bbox 右下)
- 修正傾斜面板下 MARK 排除區位置歪掉的問題

G0F 測試圖中排除區從 bbox BR (6266, 3692) 移到 polygon BR
(~6268, ~3655)，差距 ~37px 與 spec 一致。
EOF
)"
```

---

## Task 6: B0F 黑圖 `reference_polygon` fallback

**目的**: 讓 B0F 黑圖也能套用 polygon mask — 它本身 OTSU 抓不到 panel，但可以繼承同資料夾白圖的 polygon。

**Files:**
- Modify: `capi_inference.py:2979-3026` (B0F fallback 計算區塊)

---

- [ ] **Step 6.1: 找到 B0F fallback 路徑**

```bash
grep -n "reference_raw_bounds_for_dark" capi_inference.py
```

確認位置，應該在 `process_panel` 附近（line 2979 上下）。

- [ ] **Step 6.2: 修改 B0F fallback 邏輯**

在 B0F 參考邊界計算區塊裡，同時計算 reference_polygon。找到類似這段：

```python
        reference_raw_bounds_for_dark = None
        ...
                        reference_raw_bounds_for_dark = self._find_raw_object_bounds(ref_img)
                        print(f"📐 黑圖參考邊界已從 {ref_path.name} 計算 → {reference_raw_bounds_for_dark}")
```

注意 Task 1 已經把 `_find_raw_object_bounds` 改為回傳 `(bounds, binary)`。修改為：

```python
        reference_raw_bounds_for_dark = None
        reference_polygon_for_dark: Optional[np.ndarray] = None
        ...
                        ref_bounds, ref_binary = self._find_raw_object_bounds(ref_img)
                        reference_raw_bounds_for_dark = ref_bounds
                        if self.config.enable_panel_polygon:
                            reference_polygon_for_dark = self._find_panel_polygon(
                                ref_binary, ref_bounds
                            )
                        print(f"📐 黑圖參考邊界已從 {ref_path.name} 計算 → {reference_raw_bounds_for_dark}"
                              f" (polygon: {'有' if reference_polygon_for_dark is not None else '無'})")
```

- [ ] **Step 6.3: 更新把 reference 傳給 `preprocess_image` 的位置**

找到 `preprocess_image(...)` 的呼叫（約 line 3026 和 3115）：

```bash
grep -n "preprocess_image.*reference_raw_bounds" capi_inference.py
```

每處都加上 `reference_polygon`：

```python
            ref_bounds = reference_raw_bounds_for_dark if reference_raw_bounds_for_dark and _is_dark_image(img_path.name) else None
            ref_poly = reference_polygon_for_dark if reference_polygon_for_dark is not None and _is_dark_image(img_path.name) else None
            result = self.preprocess_image(
                img_path,
                cached_mark=cached_mark,
                reference_raw_bounds=ref_bounds,
                reference_polygon=ref_poly,
            )
```

另一處（約 line 3115 skip_result）：

```python
                            skip_ref_bounds = reference_raw_bounds_for_dark if reference_raw_bounds_for_dark and _is_dark_image(matched_file.name) else None
                            skip_ref_poly = reference_polygon_for_dark if reference_polygon_for_dark is not None and _is_dark_image(matched_file.name) else None
                            skip_result = self.preprocess_image(
                                matched_file,
                                cached_mark=cached_mark,
                                reference_raw_bounds=skip_ref_bounds,
                                reference_polygon=skip_ref_poly,
                            )
```

- [ ] **Step 6.4: 冒煙測試 — 確認引用路徑沒錯**

沒有 B0F 測試圖可用時可跑 syntax 檢查：

```bash
cd /c/Users/rh.syu/Desktop/CAPI01_AD
python -c "from capi_inference import CAPIInferencer; print('import ok')"
```

並重跑既有測試：

```bash
python tests/test_panel_polygon_unit.py
python tests/test_panel_polygon_tile_mask.py
```

- [ ] **Step 6.5: Commit**

```bash
git add capi_inference.py
git commit -m "$(cat <<'EOF'
feat: B0F 黑圖 polygon fallback 使用同資料夾白圖

- process_panel 內 B0F 參考路徑除了 reference_raw_bounds_for_dark
  外同時計算 reference_polygon_for_dark
- preprocess_image 的兩處呼叫都加上 reference_polygon 參數
- 白圖無法算出 polygon 時 B0F 自動退回純 bbox 行為
EOF
)"
```

---

## Task 7: Debug 視覺化多畫 polygon

**目的**: 讓 debug output 裡的 overlay 同時顯示 bbox (藍) 與 polygon (紅)，便於後續人工檢視。

**Files:**
- Modify: `capi_inference.py:1385-1390` (debug 視覺化)
- Modify: `capi_inference.py:3805-3810` (另一處視覺化)

---

- [ ] **Step 7.1: 定位所有視覺化點**

```bash
grep -n "Otsu Bounds\|otsu_bounds.*cv2.rectangle\|x1, y1, x2, y2 = result.otsu_bounds" capi_inference.py
```

預期看到 2 處（line 1385、3805 附近）。

- [ ] **Step 7.2: 修改第一處 (line ~1385)**

閱讀當前程式碼確認上下文：

```bash
sed -n '1380,1395p' capi_inference.py
```

在 `cv2.putText(vis, "Otsu Bounds", ...)` 那一行之後，追加畫 polygon：

```python
        # Otsu 邊界（藍色）
        x1, y1, x2, y2 = result.otsu_bounds
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(vis, "Otsu Bounds", (x1 + 10, y1 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        # Panel polygon（紅色）
        if result.panel_polygon is not None:
            poly_int = result.panel_polygon.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [poly_int], True, (0, 0, 255), 3)
            cv2.putText(vis, "Panel Polygon", (x1 + 10, y1 + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
```

- [ ] **Step 7.3: 修改第二處 (line ~3805)**

```bash
sed -n '3800,3815p' capi_inference.py
```

同樣在 bbox 繪圖後追加 polygon 繪圖（注意第二處可能沒有 putText，保持與現場 style 一致）：

```python
        # Otsu 邊界（藍色）
        x1, y1, x2, y2 = result.otsu_bounds
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Panel polygon（紅色）
        if result.panel_polygon is not None:
            poly_int = result.panel_polygon.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [poly_int], True, (0, 0, 255), 3)
```

- [ ] **Step 7.4: 跑既有測試**

```bash
python tests/test_panel_polygon_unit.py
python tests/test_panel_polygon_tile_mask.py
python -c "from capi_inference import CAPIInferencer; print('import ok')"
```

- [ ] **Step 7.5: Commit**

```bash
git add capi_inference.py
git commit -m "$(cat <<'EOF'
feat: debug 視覺化疊加 panel polygon (紅色)

在既有的 Otsu bbox 藍色繪圖旁追加 panel polygon 紅色輪廓，
polygon 為 None 時不畫。不影響 web dashboard 顯示。
EOF
)"
```

---

## Task 8: 整合回歸驗證 — toggle off 要完全等同舊行為

**目的**: 最後一層安全網 — 確認 `enable_panel_polygon=false` 時所有行為都與改動前一致。這是我們的 rollback 路徑，必須驗證有效。

**Files:**
- Modify: `tests/test_panel_polygon_unit.py` (新增 toggle 等價性測試)

---

- [ ] **Step 8.1: 撰寫 toggle 等價性測試**

在 `tests/test_panel_polygon_unit.py` 底部新增：

```python
def test_toggle_off_bbox_unchanged():
    """
    enable_panel_polygon=False 時 otsu_bounds 與 toggle=True 時完全一致
    (polygon 可能改變 bbox 計算路徑時，此測試會抓到)
    """
    img_path = Path(__file__).resolve().parent.parent / "test_images" / "G0F00000_151955.tif"
    if not img_path.exists():
        print(f"⚠️  跳過 (測試圖不存在): {img_path}")
        return

    cfg_on = CAPIConfig()
    cfg_on.tile_size = 512
    cfg_on.tile_stride = 512
    cfg_on.otsu_bottom_crop = 0
    cfg_on.enable_panel_polygon = True

    cfg_off = CAPIConfig()
    cfg_off.tile_size = 512
    cfg_off.tile_stride = 512
    cfg_off.otsu_bottom_crop = 0
    cfg_off.enable_panel_polygon = False

    inf_on = CAPIInferencer(cfg_on)
    inf_off = CAPIInferencer(cfg_off)

    r_on = inf_on.preprocess_image(img_path)
    r_off = inf_off.preprocess_image(img_path)

    assert r_on.otsu_bounds == r_off.otsu_bounds, \
        f"otsu_bounds 應相同, on={r_on.otsu_bounds} off={r_off.otsu_bounds}"
    assert r_off.panel_polygon is None
    assert r_on.panel_polygon is not None
    print(f"✅ test_toggle_off_bbox_unchanged (bbox={r_on.otsu_bounds})")


def test_toggle_off_all_tile_masks_are_none():
    """
    enable_panel_polygon=False 時每個 tile 的 mask 都必須是 None
    (否則代表有未受 toggle 控制的程式碼在設 mask)
    """
    img_path = Path(__file__).resolve().parent.parent / "test_images" / "G0F00000_151955.tif"
    if not img_path.exists():
        print(f"⚠️  跳過 (測試圖不存在): {img_path}")
        return

    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.tile_stride = 512
    cfg.otsu_bottom_crop = 0
    cfg.enable_panel_polygon = False
    inf = CAPIInferencer(cfg)
    result = inf.preprocess_image(img_path)

    none_count = sum(1 for t in result.tiles if t.mask is None)
    assert none_count == len(result.tiles), \
        f"toggle off 時所有 tile mask 必須為 None，實際 {none_count}/{len(result.tiles)}"
    print(f"✅ test_toggle_off_all_tile_masks_are_none ({len(result.tiles)} tiles, all mask=None)")


def test_toggle_on_some_edge_tiles_have_mask():
    """
    enable_panel_polygon=True 時至少有一個邊緣 tile 有 mask (非 None)
    (否則代表 polygon 沒實際生效)
    """
    img_path = Path(__file__).resolve().parent.parent / "test_images" / "G0F00000_151955.tif"
    if not img_path.exists():
        print(f"⚠️  跳過 (測試圖不存在): {img_path}")
        return

    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.tile_stride = 512
    cfg.otsu_bottom_crop = 0
    cfg.enable_panel_polygon = True
    inf = CAPIInferencer(cfg)
    result = inf.preprocess_image(img_path)

    masked = [t for t in result.tiles if t.mask is not None]
    assert len(masked) >= 1, \
        f"toggle on 時應至少 1 個邊緣 tile 有 mask，實際 {len(masked)}"
    print(f"✅ test_toggle_on_some_edge_tiles_have_mask ({len(masked)} 個 tile 有 mask)")
```

加入 `if __name__ == "__main__":` block。

- [ ] **Step 8.2: 跑測試**

```bash
cd /c/Users/rh.syu/Desktop/CAPI01_AD
python tests/test_panel_polygon_unit.py
python tests/test_panel_polygon_tile_mask.py
```

Expected: 全通過。

- [ ] **Step 8.3: 跑既有測試檢查無 regression**

```bash
python tests/test_cv_edge.py
```

若專案內還有其他測試，也一併跑：

```bash
ls tests/*.py
```

對每一個可 stand-alone 的 `test_*.py` 跑一次，確認無 import 錯誤或 regression。

**注意**: `tests/test_inference.py`、`tests/test_aoi_coord_inference.py`、`tests/test_dust_two_stage.py` 可能需要 server 運行或真實模型，若環境不允許跑請跳過並在 commit message 註記。

- [ ] **Step 8.4: 人工視覺確認 (建議但非必要)**

```bash
ls test_output/panel_polygon_tile_mask/
```

打開 `W0F00000_110022_tile_mask.png` 與 `G0F00000_151955_tile_mask.png`，確認：

1. 黃色 polygon 貼齊真實面板邊界
2. 藍色 bbox 與 polygon 大致重疊，但在右下角明顯偏離
3. 紅色「被 mask 掉」的區塊**全部在 polygon 外**，**不在** panel 內

- [ ] **Step 8.5: Commit**

```bash
git add tests/test_panel_polygon_unit.py
git commit -m "$(cat <<'EOF'
test: 新增 enable_panel_polygon toggle 等價性測試

- toggle=False: otsu_bounds 與 toggle=True 完全一致、所有 tile.mask 為 None
- toggle=True: 至少有一個邊緣 tile 擁有 mask (確認實際生效)

這組測試是 rollback 路徑的安全網，若未來有人不小心把 polygon 邏輯
綁在 toggle 之外的路徑上，這些測試會立刻抓到。
EOF
)"
```

---

## 驗收清單 (對照 spec §13)

所有 task 完成後逐項確認：

- [ ] `W0F00000_110022.tif` / `G0F00000_151955.tif` 兩張測試圖 4 角偵測誤差與 spec 9.1 相符 → **Task 2 覆蓋**
- [ ] 右下角 tile mask 正確排除非產品區 → **Task 4 覆蓋** (視覺 + assertion)
- [ ] `enable_panel_polygon=false` 可完全回退 → **Task 8 覆蓋**
- [ ] `tests/test_panel_polygon_unit.py` 全數通過 → **Tasks 1-3, 5, 8**
- [ ] `tests/test_panel_polygon_tile_mask.py` 全數通過 → **Task 4**
- [ ] 既有測試 `test_cv_edge.py` 無 regression → **Task 4.7、8.3**
- [ ] `_map_aoi_coords` / `check_bomb_defect` / 資料庫 schema / web dashboard heatmap **未改動** → **本 plan 不含相關 task，對照 spec §10.2**
- [ ] PatchCore pred_score 漂移驗證 → **保留給執行期 smoke test**，本 plan 未含自動化 (需要 server + 模型，超出 stand-alone 範圍)

---

## 自審筆記

1. **Spec 覆蓋度**:
   - §4 資料結構: Tasks 1 (`TileInfo` comment), 3 (`ImageResult.panel_polygon`), 1 (config)
   - §5.1 `_find_panel_polygon`: Task 2
   - §5.2 `_find_raw_object_bounds`: Task 1.5
   - §5.3 `calculate_otsu_bounds`: Task 3.2
   - §5.4 `preprocess_image`: Task 3.4
   - §5.5 `tile_image`: Task 4
   - §5.6 `calculate_exclusion_regions`: Task 5
   - §5.7 B0F fallback: Task 6
   - §6.1 debug 視覺化: Task 7
   - §8 Config 開關: Task 1 + 驗證於 Task 8
   - §9 測試: 每個 task 都有 test

2. **Placeholder 掃描**: 已人工檢查無 TBD/TODO，除「若環境不允許跑請跳過」是明確指示

3. **類型/簽名一致性**:
   - `_find_raw_object_bounds` 新簽名 `→ (bounds, binary_mask)` 在 Task 1.5 定義，Task 3/6 依此呼叫
   - `calculate_otsu_bounds` 新簽名 `→ (bounds, original_y2, polygon)` 在 Task 3.2 定義
   - `tile_image(..., panel_polygon=None)` 在 Task 4.4 定義，Task 4.5 呼叫
   - `calculate_exclusion_regions(..., panel_polygon=None)` 在 Task 5.3 定義，Task 5.4 呼叫
   - `ImageResult.panel_polygon: Optional[np.ndarray]` 在 Task 3.1 定義
