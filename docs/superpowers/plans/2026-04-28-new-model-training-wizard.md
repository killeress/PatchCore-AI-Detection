# 新機種 PatchCore 訓練 Wizard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 為新機種訓練 PatchCore 模型 bundle 的 5 步驟 web wizard，含共用前處理模組、10 模型架構（5 lighting × inner+edge）、模型庫管理、離線 backbone 預載、新舊機種架構並存。

**Architecture:** 抽取現有 `capi_inference.py` 的 polygon / tile 邏輯到新模組 `capi_preprocess.py`，訓練 / 推論共用。Wizard 走 `threading.Thread` + DB 持久化 state（不再用記憶體 dict，避免 server crash 失資料）。模型庫的「啟用」改 `server_config.yaml` 的 `model_configs` 列表，不自動重啟。Inference 端依 `model_id` → bundle yaml → tile 位置 routing inner/edge 模型。

**Tech Stack:** Python stdlib threading, SQLite (既有 capi_database), Jinja2 templates, anomalib PatchCore (既有), vanilla JS fetch + setInterval, OpenCV polygon ops。

**參考 spec:** `docs/superpowers/specs/2026-04-28-new-model-training-wizard-design.md`

---

## File Map

| Phase | File | Action |
|-------|------|--------|
| 1 | `capi_preprocess.py` | Create — Otsu/polygon/tile 抽出共用 |
| 1 | `tests/test_capi_preprocess_unit.py` | Create — 純函式測試 |
| 1 | `tests/test_capi_preprocess_integration.py` | Create — fixture image 端到端 |
| 1 | `tests/fixtures/preprocess/*.png` | Create — 1 張 panel 縮圖 + 預期 polygon JSON |
| 2 | `capi_config.py` | Modify — 新增 `machine_id`、`is_new_architecture`、`model_mapping` 巢狀 dict 支援 |
| 2 | `capi_server.py` | Modify — 啟動載多個 yaml；request dispatcher 依 model_id 選 cfg |
| 2 | `server_config.yaml` / `server_config_local.yaml` | Modify — 加 `model_configs`、`fallback_model_config`、`training` 區塊 |
| 3 | `capi_database.py` | Modify — 新增 3 表 schema + CRUD |
| 3 | `tests/test_capi_database_train.py` | Create — DB CRUD 測試 |
| 4 | `capi_train_new.py` | Create — preprocess worker + training worker |
| 4 | `tests/test_capi_train_new_preprocess.py` | Create — preprocess worker 測試 |
| 4 | `tests/test_capi_train_new_training.py` | Create — training worker 測試（mock anomalib） |
| 5 | `capi_web.py` | Modify — 加 wizard 路由（共 ~9 個 handler） |
| 5 | `tests/test_capi_web_train_new.py` | Create — API 路由測試 |
| 6 | `templates/train_new/step1_select.html` | Create |
| 6 | `templates/train_new/step2_progress.html` | Create |
| 6 | `templates/train_new/step3_review.html` | Create |
| 6 | `templates/train_new/step4_progress.html` | Create |
| 6 | `templates/train_new/step5_done.html` | Create |
| 6 | `capi_web.py` | Modify — `_handle_train_new_page` 入口 + resume 邏輯 |
| 7 | `capi_model_registry.py` | Create — 列表 / 啟用 / 停用 / 刪除 / ZIP |
| 7 | `tests/test_capi_model_registry.py` | Create |
| 8 | `templates/models.html` | Create — 模型庫頁 |
| 8 | `capi_web.py` | Modify — 加 `/models` 路由（共 ~6 個 handler） |
| 8 | `capi_web.py` | Modify — `_build_training_cards()` 加新機種卡 |
| 9 | `capi_inference.py` | Modify — 移除 polygon/MARK 內部邏輯 → import preprocess；加 `_process_panel_v2` 雙路徑；inner/edge lazy load |
| 9 | `tests/test_panel_polygon_*.py` | Modify — import path 改 `capi_preprocess` |
| 9 | `tests/test_capi_inference_v2.py` | Create — 新架構 inference 整合測試 |
| 10 | `tools/build_bga_tiles.py` | Modify — 加 deprecated header |
| 10 | `tools/train_bga_all.py` | Modify — 加 deprecated header |
| 10 | `CLAUDE.md` | Modify — 反映新架構 |

---

## Phase 1：`capi_preprocess.py` 共用前處理模組（foundation）

無 inference 行為變化、純抽出與重構，先做建基礎。

### Task 1.1 — 建立模組骨架 + dataclasses

**Files:**
- Create: `capi_preprocess.py`
- Create: `tests/test_capi_preprocess_unit.py`

- [ ] **Step 1：寫 dataclass 結構失敗測試**

```python
# tests/test_capi_preprocess_unit.py
from capi_preprocess import PreprocessConfig, TileResult, PanelPreprocessResult

def test_preprocess_config_defaults():
    cfg = PreprocessConfig()
    assert cfg.tile_size == 512
    assert cfg.tile_stride == 512
    assert cfg.otsu_offset == 5
    assert cfg.enable_panel_polygon is True
    assert cfg.edge_threshold_px == 768
    assert cfg.coverage_min == 0.3

def test_tile_result_zone_values():
    import numpy as np
    t = TileResult(tile_id=0, x1=0, y1=0, x2=512, y2=512,
                   image=np.zeros((512,512), np.uint8),
                   mask=None, coverage=1.0, zone="inner",
                   center_dist_to_edge=999.0)
    assert t.zone == "inner"
```

- [ ] **Step 2：跑測試確認 fail**

Run: `pytest tests/test_capi_preprocess_unit.py -v`
Expected: ImportError / ModuleNotFoundError

- [ ] **Step 3：建模組 + dataclass**

```python
# capi_preprocess.py
"""共用前處理：訓練 / 推論皆使用此模組。

從 capi_inference.py 抽出 Otsu / panel polygon / tile 切分 / zone 分類，
讓訓練端與推論端走同一套邏輯。
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np


@dataclass
class PreprocessConfig:
    tile_size: int = 512
    tile_stride: int = 512
    otsu_offset: int = 5
    enable_panel_polygon: bool = True
    edge_threshold_px: int = 768
    coverage_min: float = 0.3


@dataclass
class TileResult:
    tile_id: int
    x1: int; y1: int; x2: int; y2: int
    image: np.ndarray
    mask: Optional[np.ndarray]
    coverage: float
    zone: str  # "inner" | "edge" | "outside"
    center_dist_to_edge: float


@dataclass
class PanelPreprocessResult:
    image_path: Path
    lighting: str
    foreground_bbox: Tuple[int, int, int, int]
    panel_polygon: Optional[np.ndarray]
    tiles: List[TileResult] = field(default_factory=list)
    polygon_detection_failed: bool = False
```

- [ ] **Step 4：跑測試確認 pass**

Run: `pytest tests/test_capi_preprocess_unit.py -v`
Expected: 2 passed

- [ ] **Step 5：commit**

```bash
git add capi_preprocess.py tests/test_capi_preprocess_unit.py
git commit -m "feat(preprocess): scaffold capi_preprocess module with dataclasses"
```

### Task 1.2 — `filter_panel_lighting_files`

**Files:**
- Modify: `capi_preprocess.py`
- Modify: `tests/test_capi_preprocess_unit.py`

- [ ] **Step 1：寫過濾規則測試**

```python
# tests/test_capi_preprocess_unit.py 加入：
import tempfile

def test_filter_panel_lighting_files_keeps_5():
    from capi_preprocess import filter_panel_lighting_files
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for name in [
            "G0F00000_164039.tif", "R0F00000_164041.tif",
            "W0F00000_164033.tif", "WGF50500_164035.tif",
            "STANDARD_164038.tif",
            # filter out:
            "B0F00000_164043.tif", "PINIGBI _164030.tif",
            "SG0F00000_164039.tif", "SR0F00000_164041.tif",
            "SSTANDARD_164038.tif",
            "Optics.log",
        ]:
            (base / name).write_bytes(b"x")
        result = filter_panel_lighting_files(base)
        assert set(result.keys()) == {"G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD"}
        assert result["G0F00000"].name == "G0F00000_164039.tif"

def test_filter_panel_lighting_files_partial_panel():
    from capi_preprocess import filter_panel_lighting_files
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "G0F00000_x.tif").write_bytes(b"x")
        (base / "STANDARD_x.tif").write_bytes(b"x")
        result = filter_panel_lighting_files(base)
        assert set(result.keys()) == {"G0F00000", "STANDARD"}
```

- [ ] **Step 2：跑測試確認 fail**

Run: `pytest tests/test_capi_preprocess_unit.py::test_filter_panel_lighting_files_keeps_5 -v`
Expected: ImportError

- [ ] **Step 3：實作 filter**

```python
# 加到 capi_preprocess.py
LIGHTING_PREFIXES = ("G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD")
SKIP_PREFIXES = ("S", "B0F", "PINIGBI", "OMIT")
SKIP_EXACT = ("Optics.log",)


def filter_panel_lighting_files(folder: Path) -> Dict[str, Path]:
    """從 panel folder 過濾出 5 個有效 lighting 圖。
    
    跳過：S* (側拍) / B0F (黑屏) / PINIGBI (點燈狀態檔) / OMIT (光源圖) /
          Optics.log。
    
    Returns: {"G0F00000": Path, ...}，缺哪個 lighting 就少哪個 key。
    """
    result: Dict[str, Path] = {}
    for entry in folder.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name in SKIP_EXACT:
            continue
        if any(name.startswith(p) for p in SKIP_PREFIXES):
            continue
        for lighting in LIGHTING_PREFIXES:
            if name.startswith(lighting):
                if lighting not in result:
                    result[lighting] = entry
                break
    return result
```

- [ ] **Step 4：跑測試 pass**

Run: `pytest tests/test_capi_preprocess_unit.py -v`
Expected: 4 passed

- [ ] **Step 5：commit**

```bash
git add capi_preprocess.py tests/test_capi_preprocess_unit.py
git commit -m "feat(preprocess): add filter_panel_lighting_files"
```

### Task 1.3 — `detect_panel_polygon`（從 inference 抽出）

**Files:**
- Modify: `capi_preprocess.py`
- Modify: `tests/test_capi_preprocess_unit.py`

- [ ] **Step 1：寫測試**

```python
# tests/test_capi_preprocess_unit.py 加入：
import cv2

def test_detect_panel_polygon_simple_rect():
    from capi_preprocess import detect_panel_polygon, PreprocessConfig
    img = np.zeros((1000, 1500), np.uint8)
    cv2.rectangle(img, (200, 100), (1300, 900), 200, -1)  # 白色 panel
    bbox, poly = detect_panel_polygon(img, PreprocessConfig())
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert 195 <= x1 <= 215 and 95 <= y1 <= 115
    assert 1295 <= x2 <= 1315 and 895 <= y2 <= 915
    assert poly is not None
    assert poly.shape == (4, 2)

def test_detect_panel_polygon_disabled_returns_no_polygon():
    from capi_preprocess import detect_panel_polygon, PreprocessConfig
    img = np.zeros((500, 500), np.uint8)
    cv2.rectangle(img, (100, 100), (400, 400), 200, -1)
    cfg = PreprocessConfig(enable_panel_polygon=False)
    bbox, poly = detect_panel_polygon(img, cfg)
    assert bbox is not None
    assert poly is None
```

- [ ] **Step 2：跑 fail**

Run: `pytest tests/test_capi_preprocess_unit.py::test_detect_panel_polygon_simple_rect -v`
Expected: ImportError

- [ ] **Step 3：把 `_find_panel_polygon` 從 capi_inference.py 抽出搬到 capi_preprocess.py，重命名為 `detect_panel_polygon` 並接受 config**

```python
# 加到 capi_preprocess.py
import cv2

EDGE_MARGIN = 20
SAMPLE_STEP = 50
OUTLIER_SIGMA = 3.0
MIN_EDGE_LEN_RATIO = 1.0
MIN_POLYGON_AREA_RATIO = 0.9
MIN_SAMPLES_PER_EDGE = 5


def detect_panel_polygon(
    image: np.ndarray,
    config: PreprocessConfig,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """Otsu binarize → 最大連通輪廓 bbox → polyfit 4 角 polygon。
    
    Returns:
        (bbox, polygon)
        bbox = (x1, y1, x2, y2)，若 binarize 失敗回 (None, None)
        polygon = (4,2) float32 [TL, TR, BR, BL]，偵測失敗或 enable_panel_polygon=False 回 None
    """
    if image is None or image.size == 0:
        return None, None
    
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    offset = config.otsu_offset
    bbox = (x + offset, y + offset, x + w - offset, y + h - offset)
    
    if not config.enable_panel_polygon:
        return bbox, None
    
    polygon = _polyfit_polygon(binary, bbox, config.tile_size)
    return bbox, polygon


def _polyfit_polygon(
    binary_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    tile_size: int,
) -> Optional[np.ndarray]:
    """從 capi_inference._find_panel_polygon 抽出，邏輯不變。"""
    H, W = binary_mask.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    if xmax - xmin < 2 * EDGE_MARGIN or ymax - ymin < 2 * EDGE_MARGIN:
        return None
    
    tops, bots, lefts, rights = [], [], [], []
    for x in range(xmin + EDGE_MARGIN, xmax - EDGE_MARGIN, SAMPLE_STEP):
        if 0 <= x < W:
            ys = np.where(binary_mask[:, x] > 0)[0]
            if len(ys):
                tops.append((x, int(ys[0])))
                bots.append((x, int(ys[-1])))
    for y in range(ymin + EDGE_MARGIN, ymax - EDGE_MARGIN, SAMPLE_STEP):
        if 0 <= y < H:
            xs = np.where(binary_mask[y, :] > 0)[0]
            if len(xs):
                lefts.append((int(xs[0]), y))
                rights.append((int(xs[-1]), y))
    
    if min(len(tops), len(bots), len(lefts), len(rights)) < MIN_SAMPLES_PER_EDGE:
        return None
    
    def fit(pts, horizontal):
        arr = np.array(pts, dtype=float)
        ind, dep = (arr[:, 0], arr[:, 1]) if horizontal else (arr[:, 1], arr[:, 0])
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
        return float(a), float(b)
    
    top_l, bot_l, left_l, right_l = fit(tops, True), fit(bots, True), fit(lefts, False), fit(rights, False)
    if None in (top_l, bot_l, left_l, right_l):
        return None
    
    def intersect(h, v):
        a_h, b_h = h; a_v, b_v = v
        denom = 1.0 - a_h * a_v
        if abs(denom) < 1e-9:
            return None
        y = (a_h * b_v + b_h) / denom
        x = a_v * y + b_v
        return (x, y)
    
    TL, TR, BR, BL = intersect(top_l, left_l), intersect(top_l, right_l), intersect(bot_l, right_l), intersect(bot_l, left_l)
    if None in (TL, TR, BR, BL):
        return None
    
    polygon = np.array([TL, TR, BR, BL], dtype=np.float32)
    
    tol = 50
    if (polygon[:, 0].min() < -tol or polygon[:, 0].max() > W + tol or
            polygon[:, 1].min() < -tol or polygon[:, 1].max() > H + tol):
        return None
    
    min_edge_len = tile_size * MIN_EDGE_LEN_RATIO
    for i in range(4):
        if float(np.linalg.norm(polygon[(i + 1) % 4] - polygon[i])) < min_edge_len:
            return None
    
    bbox_area = float((xmax - xmin) * (ymax - ymin))
    poly_area = float(cv2.contourArea(polygon))
    if bbox_area <= 0 or poly_area < bbox_area * MIN_POLYGON_AREA_RATIO:
        return None
    
    return polygon
```

- [ ] **Step 4：跑測試 pass**

Run: `pytest tests/test_capi_preprocess_unit.py -v`
Expected: 6 passed

- [ ] **Step 5：commit**

```bash
git add capi_preprocess.py tests/test_capi_preprocess_unit.py
git commit -m "feat(preprocess): extract detect_panel_polygon from capi_inference"
```

### Task 1.4 — `classify_tile_zone`

**Files:**
- Modify: `capi_preprocess.py`
- Modify: `tests/test_capi_preprocess_unit.py`

- [ ] **Step 1：寫測試（5 個 case 覆蓋 zone 規則）**

```python
def test_classify_tile_zone_inner():
    from capi_preprocess import classify_tile_zone, PreprocessConfig
    poly = np.array([[100, 100], [4000, 100], [4000, 3000], [100, 3000]], np.float32)
    cfg = PreprocessConfig(tile_size=512, edge_threshold_px=768)
    # tile fully inside, distance from edge >= 768
    zone, cov, dist, mask = classify_tile_zone((1500, 1200, 2012, 1712), poly, cfg)
    assert zone == "inner"
    assert cov == 1.0
    assert mask is None

def test_classify_tile_zone_edge_close_to_boundary():
    from capi_preprocess import classify_tile_zone, PreprocessConfig
    poly = np.array([[100, 100], [4000, 100], [4000, 3000], [100, 3000]], np.float32)
    cfg = PreprocessConfig(tile_size=512, edge_threshold_px=768)
    # tile fully inside but center close to top edge
    zone, cov, dist, mask = classify_tile_zone((1500, 200, 2012, 712), poly, cfg)
    assert zone == "edge"
    assert cov == 1.0

def test_classify_tile_zone_edge_partial_coverage():
    from capi_preprocess import classify_tile_zone, PreprocessConfig
    poly = np.array([[100, 100], [4000, 100], [4000, 3000], [100, 3000]], np.float32)
    cfg = PreprocessConfig()
    # tile crosses top boundary
    zone, cov, dist, mask = classify_tile_zone((1500, 0, 2012, 512), poly, cfg)
    assert zone == "edge"
    assert 0.3 < cov < 1.0
    assert mask is not None
    assert mask.shape == (512, 512)

def test_classify_tile_zone_outside():
    from capi_preprocess import classify_tile_zone, PreprocessConfig
    poly = np.array([[100, 100], [4000, 100], [4000, 3000], [100, 3000]], np.float32)
    cfg = PreprocessConfig()
    zone, cov, dist, mask = classify_tile_zone((4500, 1500, 5012, 2012), poly, cfg)
    assert zone == "outside"
    assert cov < 0.3

def test_classify_tile_zone_no_polygon_fallback_inner():
    from capi_preprocess import classify_tile_zone, PreprocessConfig
    cfg = PreprocessConfig()
    zone, cov, dist, mask = classify_tile_zone((0, 0, 512, 512), None, cfg)
    assert zone == "inner"
    assert cov == 1.0
    assert mask is None
```

- [ ] **Step 2：跑 fail**

- [ ] **Step 3：實作 classify**

```python
# 加到 capi_preprocess.py
def classify_tile_zone(
    tile_rect: Tuple[int, int, int, int],
    polygon: Optional[np.ndarray],
    config: PreprocessConfig,
) -> Tuple[str, float, float, Optional[np.ndarray]]:
    """根據 polygon 與 tile 幾何決定 zone + 計算 coverage / center_dist。
    
    Returns: (zone, coverage, center_dist_to_edge, mask)
        - zone: "inner" | "edge" | "outside"
        - mask: tile 內 polygon 的 binary mask（uint8 0/255），fully inside 時 None
        - polygon=None → fallback ("inner", 1.0, inf, None)
    """
    x1, y1, x2, y2 = tile_rect
    tile_size = max(x2 - x1, y2 - y1)
    
    if polygon is None:
        return "inner", 1.0, float("inf"), None
    
    # tile 內生成 polygon mask
    mask = np.zeros((tile_size, tile_size), np.uint8)
    shifted = polygon.copy()
    shifted[:, 0] -= x1
    shifted[:, 1] -= y1
    cv2.fillPoly(mask, [shifted.astype(np.int32)], 255)
    coverage = float((mask > 0).sum()) / (tile_size * tile_size)
    
    if coverage < config.coverage_min:
        return "outside", coverage, 0.0, mask
    
    # 計算 tile 中心到 polygon 邊的最短距離
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dists = []
    for i in range(4):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % 4]
        # 點到線段距離
        d = _point_segment_dist((cx, cy), tuple(p1), tuple(p2))
        dists.append(d)
    center_dist = float(min(dists))
    
    # 決定 zone：完全在內 + 距邊 >= threshold → inner，其他 → edge
    if coverage >= 1.0 - 1e-6 and center_dist >= config.edge_threshold_px:
        return "inner", 1.0, center_dist, None
    return "edge", coverage, center_dist, mask if coverage < 1.0 - 1e-6 else None


def _point_segment_dist(p, a, b):
    px, py = p; ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    seg_sq = dx * dx + dy * dy
    if seg_sq < 1e-9:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_sq))
    qx, qy = ax + t * dx, ay + t * dy
    return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5
```

- [ ] **Step 4：跑 pass**

Run: `pytest tests/test_capi_preprocess_unit.py -v`
Expected: 11 passed

- [ ] **Step 5：commit**

```bash
git add capi_preprocess.py tests/test_capi_preprocess_unit.py
git commit -m "feat(preprocess): add classify_tile_zone with edge_threshold logic"
```

### Task 1.5 — `preprocess_panel_image`（單張圖完整 pipeline）

**Files:**
- Modify: `capi_preprocess.py`
- Create: `tests/test_capi_preprocess_integration.py`
- Create: `tests/fixtures/preprocess/synthetic_panel.png`

- [ ] **Step 1：產生測試 fixture**

```python
# 一次性 script: scripts/gen_preprocess_fixture.py
import cv2, numpy as np
from pathlib import Path

img = np.zeros((1000, 1500), np.uint8)
cv2.rectangle(img, (150, 100), (1350, 900), 180, -1)
# 加雜訊讓 Otsu 工作
img = img + np.random.randint(0, 20, img.shape, dtype=np.uint8)
out = Path("tests/fixtures/preprocess")
out.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out / "synthetic_panel.png"), img)
```

執行：
```bash
mkdir -p tests/fixtures/preprocess
python scripts/gen_preprocess_fixture.py  # 一次性
git add tests/fixtures/preprocess/synthetic_panel.png
```

- [ ] **Step 2：寫整合測試**

```python
# tests/test_capi_preprocess_integration.py
from pathlib import Path
import pytest
from capi_preprocess import preprocess_panel_image, PreprocessConfig

FIXTURE = Path(__file__).parent / "fixtures" / "preprocess" / "synthetic_panel.png"

def test_preprocess_panel_image_basic():
    cfg = PreprocessConfig(tile_size=256, edge_threshold_px=384)
    result = preprocess_panel_image(FIXTURE, "STANDARD", cfg)
    assert result.lighting == "STANDARD"
    assert not result.polygon_detection_failed
    assert result.panel_polygon is not None
    assert len(result.tiles) > 0
    zones = {t.zone for t in result.tiles}
    assert "inner" in zones
    assert "edge" in zones
    # outside tile 不應出現
    assert "outside" not in zones

def test_preprocess_panel_image_with_reference_polygon():
    import numpy as np
    cfg = PreprocessConfig(tile_size=256)
    ref = np.array([[200, 150], [1300, 150], [1300, 850], [200, 850]], np.float32)
    result = preprocess_panel_image(FIXTURE, "G0F00000", cfg, reference_polygon=ref)
    # 應該直接套 reference 不重新偵測
    np.testing.assert_array_almost_equal(result.panel_polygon, ref)
```

- [ ] **Step 3：實作 preprocess_panel_image**

```python
# 加到 capi_preprocess.py
def preprocess_panel_image(
    image_path: Path,
    lighting: str,
    config: PreprocessConfig,
    reference_polygon: Optional[np.ndarray] = None,
) -> PanelPreprocessResult:
    """單張 lighting 圖完整前處理。"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"無法讀取圖片: {image_path}")
    
    if reference_polygon is not None:
        # 沿用 reference polygon，仍要偵測 bbox
        bbox, _ = detect_panel_polygon(img, config)
        polygon = reference_polygon
        polygon_failed = False
    else:
        bbox, polygon = detect_panel_polygon(img, config)
        polygon_failed = (config.enable_panel_polygon and polygon is None)
    
    if bbox is None:
        return PanelPreprocessResult(
            image_path=image_path, lighting=lighting,
            foreground_bbox=(0, 0, 0, 0),
            panel_polygon=None, tiles=[],
            polygon_detection_failed=True,
        )
    
    tiles = _generate_tiles(img, bbox, polygon, config)
    return PanelPreprocessResult(
        image_path=image_path, lighting=lighting,
        foreground_bbox=bbox, panel_polygon=polygon,
        tiles=tiles, polygon_detection_failed=polygon_failed,
    )


def _generate_tiles(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    polygon: Optional[np.ndarray],
    config: PreprocessConfig,
) -> List[TileResult]:
    x1, y1, x2, y2 = bbox
    ts = config.tile_size
    
    def positions(lo, hi):
        if hi - lo < ts:
            return []
        out = list(range(lo, hi - ts + 1, config.tile_stride))
        if out and out[-1] != hi - ts:
            out.append(hi - ts)
        return out
    
    xs, ys = positions(x1, x2), positions(y1, y2)
    tiles: List[TileResult] = []
    tid = 0
    for ty in ys:
        for tx in xs:
            tile_rect = (tx, ty, tx + ts, ty + ts)
            zone, cov, dist, mask = classify_tile_zone(tile_rect, polygon, config)
            if zone == "outside":
                continue
            tile_img = img[ty:ty + ts, tx:tx + ts].copy()
            tiles.append(TileResult(
                tile_id=tid, x1=tx, y1=ty, x2=tx + ts, y2=ty + ts,
                image=tile_img, mask=mask, coverage=cov,
                zone=zone, center_dist_to_edge=dist,
            ))
            tid += 1
    return tiles
```

- [ ] **Step 4：跑 pass**

Run: `pytest tests/test_capi_preprocess_integration.py -v`
Expected: 2 passed

- [ ] **Step 5：commit**

```bash
git add capi_preprocess.py tests/test_capi_preprocess_integration.py tests/fixtures/preprocess/
git commit -m "feat(preprocess): add preprocess_panel_image full pipeline + fixture"
```

### Task 1.6 — `preprocess_panel_folder`（多 lighting 對齊）

**Files:**
- Modify: `capi_preprocess.py`
- Modify: `tests/test_capi_preprocess_integration.py`

- [ ] **Step 1：寫測試**

```python
def test_preprocess_panel_folder_uses_reference_polygon(tmp_path):
    import shutil
    from capi_preprocess import preprocess_panel_folder, PreprocessConfig
    # 複製 fixture 5 份模擬不同 lighting
    for lighting in ["STANDARD", "G0F00000", "R0F00000", "W0F00000", "WGF50500"]:
        shutil.copy(FIXTURE, tmp_path / f"{lighting}_x.tif")
    cfg = PreprocessConfig(tile_size=256)
    results = preprocess_panel_folder(tmp_path, cfg)
    assert set(results.keys()) == {"STANDARD", "G0F00000", "R0F00000", "W0F00000", "WGF50500"}
    # 所有 lighting 應共用同一 polygon
    ref_poly = results["STANDARD"].panel_polygon
    for lighting, r in results.items():
        np.testing.assert_array_almost_equal(r.panel_polygon, ref_poly)
```

- [ ] **Step 2：跑 fail**

- [ ] **Step 3：實作**

```python
# 加到 capi_preprocess.py
def preprocess_panel_folder(
    folder: Path,
    config: PreprocessConfig,
) -> Dict[str, PanelPreprocessResult]:
    """處理整個 panel folder 的 5 lighting 圖。
    
    流程：filter 出 5 lighting → STANDARD 先處理取 reference polygon →
          其他 4 lighting 套 reference。STANDARD 失敗 fallback G0F00000。
    """
    files = filter_panel_lighting_files(folder)
    if not files:
        return {}
    
    # 決定 reference image：STANDARD > G0F00000 > 任意
    ref_lighting = None
    for cand in ("STANDARD", "G0F00000", "W0F00000", "R0F00000", "WGF50500"):
        if cand in files:
            ref_lighting = cand
            break
    if ref_lighting is None:
        return {}
    
    ref_result = preprocess_panel_image(files[ref_lighting], ref_lighting, config)
    if ref_result.polygon_detection_failed and ref_lighting != "G0F00000" and "G0F00000" in files:
        ref_lighting = "G0F00000"
        ref_result = preprocess_panel_image(files[ref_lighting], ref_lighting, config)
    
    results: Dict[str, PanelPreprocessResult] = {ref_lighting: ref_result}
    ref_poly = ref_result.panel_polygon
    for lighting, path in files.items():
        if lighting == ref_lighting:
            continue
        results[lighting] = preprocess_panel_image(path, lighting, config, reference_polygon=ref_poly)
    return results
```

- [ ] **Step 4：跑 pass**

- [ ] **Step 5：commit**

```bash
git add capi_preprocess.py tests/test_capi_preprocess_integration.py
git commit -m "feat(preprocess): add preprocess_panel_folder with reference polygon"
```

### Task 1.7 — 既有 polygon 測試 import path 搬遷

**Files:**
- Modify: `tests/test_panel_polygon_unit.py`
- Modify: `tests/test_panel_polygon_detect.py`
- Modify: `tests/test_panel_polygon_tile_mask.py`

- [ ] **Step 1：grep 看 import 用法**

Run: `grep -n "from capi_inference\|import capi_inference\|_find_panel_polygon\|calculate_otsu_bounds" tests/test_panel_polygon_*.py`
記下所有引用點。

- [ ] **Step 2：把 import path 改 `capi_preprocess`**

對每個檔，例如 `tests/test_panel_polygon_unit.py`：
- `from capi_inference import CAPIInferencer` → 維持（若需要 inferencer）
- 或調用 `inferencer._find_panel_polygon(...)` → 改為 `from capi_preprocess import detect_panel_polygon, PreprocessConfig` 並改寫呼叫

詳細做法：先看每個測試到底測什麼，若是純 polygon 邏輯就改成直接 import `capi_preprocess.detect_panel_polygon`；若涉及 inferencer 內部 state 則維持但加 deprecation note。

- [ ] **Step 3：跑舊測試確認仍 pass**

Run: `pytest tests/test_panel_polygon_unit.py tests/test_panel_polygon_detect.py tests/test_panel_polygon_tile_mask.py -v`
Expected: all pass

- [ ] **Step 4：commit**

```bash
git add tests/test_panel_polygon_*.py
git commit -m "refactor(tests): migrate polygon tests to import capi_preprocess"
```

---

**Phase 1 完成 checkpoint**：`capi_preprocess.py` 是獨立可用模組，不影響 inference。所有測試 pass。

---

## Phase 2：`capi_config` / `capi_server` 多機種 yaml 支援

### Task 2.1 — `CAPIConfig` 新增新架構欄位

**Files:**
- Modify: `capi_config.py`
- Create: `tests/test_capi_config_v2.py`

- [ ] **Step 1：寫測試**

```python
# tests/test_capi_config_v2.py
import tempfile
from pathlib import Path
import yaml
from capi_config import CAPIConfig

def test_capi_config_legacy_yaml_no_machine_id():
    cfg_data = {
        "model_path": "model.pt",
        "model_mapping": {"G0F00000": "g.pt"},
        "threshold_mapping": {"G0F00000": 0.75},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg_data, f)
        path = f.name
    cfg = CAPIConfig.from_yaml(path)
    assert cfg.machine_id is None
    assert cfg.is_new_architecture is False
    assert cfg.model_mapping == {"G0F00000": "g.pt"}
    Path(path).unlink()

def test_capi_config_new_arch_yaml():
    cfg_data = {
        "machine_id": "GN160JCEL250S",
        "edge_threshold_px": 768,
        "model_mapping": {
            "G0F00000": {"inner": "g_inner.pt", "edge": "g_edge.pt"},
            "STANDARD": {"inner": "s_inner.pt", "edge": "s_edge.pt"},
        },
        "threshold_mapping": {
            "G0F00000": {"inner": 0.62, "edge": 0.71},
        },
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg_data, f)
        path = f.name
    cfg = CAPIConfig.from_yaml(path)
    assert cfg.machine_id == "GN160JCEL250S"
    assert cfg.is_new_architecture is True
    assert cfg.edge_threshold_px == 768
    assert cfg.model_mapping["G0F00000"]["inner"] == "g_inner.pt"
    Path(path).unlink()
```

- [ ] **Step 2：跑 fail**

- [ ] **Step 3：在 `CAPIConfig` 加欄位 + from_yaml 改造**

修改 `capi_config.py`：

```python
# 在 @dataclass 內加（從現有 100 行附近的欄位區塊找位置）：
machine_id: Optional[str] = None
is_new_architecture: bool = False
edge_threshold_px: int = 768

# from_yaml 內判斷新架構：
@classmethod
def from_yaml(cls, yaml_path: str) -> "CAPIConfig":
    path = Path(yaml_path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    # 判斷新架構：machine_id 存在 且 model_mapping value 為 dict
    model_mapping = data.get("model_mapping", {})
    is_new = bool(data.get("machine_id")) and any(
        isinstance(v, dict) and {"inner", "edge"}.issubset(v.keys())
        for v in model_mapping.values()
    )
    
    # ... 既有的所有 cls(...) 參數 ...
    cfg = cls(
        # ... 既有所有欄位 ...
        machine_id=data.get("machine_id"),
        is_new_architecture=is_new,
        edge_threshold_px=int(data.get("edge_threshold_px", 768)),
        model_mapping=model_mapping,
        threshold_mapping=data.get("threshold_mapping", {}),
    )
    cfg.config_path = path
    return cfg
```

⚠️ 注意：閱讀現有 from_yaml 完整實作後再 inline 修改，不要重寫。

- [ ] **Step 4：跑 pass，且現有 capi_3f.yaml 也應 work**

Run: `pytest tests/test_capi_config_v2.py -v`
Run: `python -c "from capi_config import CAPIConfig; cfg = CAPIConfig.from_yaml('configs/capi_3f.yaml'); print(cfg.is_new_architecture)"`
Expected: `False`

- [ ] **Step 5：commit**

```bash
git add capi_config.py tests/test_capi_config_v2.py
git commit -m "feat(config): support new-arch yaml with machine_id + nested model_mapping"
```

### Task 2.2 — `server_config.yaml` schema 擴充

**Files:**
- Modify: `server_config.yaml`
- Modify: `server_config_local.yaml`

- [ ] **Step 1：先讀現況**

Run: `cat server_config_local.yaml`
記下現有結構（注意縮排）。

- [ ] **Step 2：加入 `model_configs` / `fallback_model_config` / `training` 區塊**

加在檔尾（兩份檔案都改）：

```yaml
# 新架構：多機種 yaml 並存（legacy capi_3f.yaml 仍維持）
model_configs:
  - configs/capi_3f.yaml
fallback_model_config: configs/capi_3f.yaml

# 訓練 wizard 設定
training:
  backbone_cache_dir: deployment/torch_hub_cache
  required_backbones:
    - wide_resnet50_2-32ee1156.pth
```

⚠️ 不要刪除既有 `model_path` / 類似 legacy 欄位 — 那些對應既有 server_config 載入路徑。

- [ ] **Step 3：commit**

```bash
git add server_config.yaml server_config_local.yaml
git commit -m "config: add model_configs/training sections to server config"
```

### Task 2.3 — `capi_server.py` 載多 yaml + dispatcher

**Files:**
- Modify: `capi_server.py`

- [ ] **Step 1：先讀 capi_server.py 找現有 config 載入處**

Run: `grep -n "CAPIConfig.from_yaml\|self.config\|configs_by\|model_id" capi_server.py | head -30`
找到 server 啟動時載 config 的位置（通常在 `__init__` 或 `start` 方法）。

- [ ] **Step 2：在啟動時改載多個 yaml**

加程式碼（改 server 啟動 path）：

```python
# capi_server.py 的 server 初始化位置
self.configs_by_machine = {}
server_cfg_dict = yaml.safe_load(open(server_config_path))

for cfg_path in server_cfg_dict.get("model_configs", [server_config_path]):
    cfg = CAPIConfig.from_yaml(cfg_path)
    key = cfg.machine_id if cfg.machine_id else "3F"
    self.configs_by_machine[key] = cfg

fallback_path = server_cfg_dict.get("fallback_model_config", server_config_path)
self.fallback_config = CAPIConfig.from_yaml(fallback_path)

# 將 self.config 設為「主要的」legacy 或 fallback，繼續餵舊路徑
self.config = self.fallback_config
```

- [ ] **Step 3：加 dispatcher 方法依 model_id 選 config**

```python
def get_config_for(self, model_id: str) -> CAPIConfig:
    return self.configs_by_machine.get(model_id, self.fallback_config)
```

- [ ] **Step 4：手動驗證**

啟 server `python capi_server.py --config server_config_local.yaml`
觀察啟動 log：應顯示「Loaded N model configs」之類訊息（自己加 print）。
連到 `http://localhost:8080/api/status` 確認沒掛。

- [ ] **Step 5：commit**

```bash
git add capi_server.py
git commit -m "feat(server): load multiple model_configs at startup, add machine dispatcher"
```

---

**Phase 2 完成 checkpoint**：Server 啟動時可載多個 yaml，但 inference 端尚未使用 dispatcher（Phase 9 才接）。`capi_3f.yaml` 仍走 legacy path，行為不變。

---

## Phase 3：DB Schema 擴充

### Task 3.1 — 加入 3 張新表 schema migration

**Files:**
- Modify: `capi_database.py`
- Create: `tests/test_capi_database_train.py`

- [ ] **Step 1：先讀 capi_database 找 schema migration 位置**

Run: `grep -n "CREATE TABLE\|_init_schema\|migration\|def __init__" capi_database.py | head -30`
找到 `_init_schema` 或類似初始化函式，新表加在這裡。

- [ ] **Step 2：寫 schema 測試**

```python
# tests/test_capi_database_train.py
import tempfile
from pathlib import Path
from capi_database import CAPIDatabase

def test_training_jobs_schema():
    with tempfile.TemporaryDirectory() as tmp:
        db = CAPIDatabase(Path(tmp) / "test.db")
        cur = db._conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        assert "training_jobs" in tables
        assert "model_registry" in tables
        assert "training_tile_pool" in tables
        cur.execute("PRAGMA table_info(training_jobs)")
        cols = {row[1] for row in cur.fetchall()}
        assert {"job_id", "machine_id", "state", "panel_paths", "output_bundle"}.issubset(cols)
```

- [ ] **Step 3：加表 schema**

在 `capi_database.py` 的 `_init_schema`（或對應位置）加入：

```python
self._conn.executescript("""
CREATE TABLE IF NOT EXISTS training_jobs (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id          TEXT UNIQUE,
  machine_id      TEXT NOT NULL,
  state           TEXT NOT NULL,
  started_at      TEXT,
  completed_at    TEXT,
  panel_paths     TEXT,
  output_bundle   TEXT,
  error_message   TEXT
);

CREATE TABLE IF NOT EXISTS model_registry (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  machine_id        TEXT NOT NULL,
  bundle_path       TEXT UNIQUE NOT NULL,
  trained_at        TEXT NOT NULL,
  panel_count       INTEGER,
  inner_tile_count  INTEGER,
  edge_tile_count   INTEGER,
  ng_tile_count     INTEGER,
  bundle_size_bytes INTEGER,
  is_active         INTEGER DEFAULT 0,
  job_id            TEXT,
  notes             TEXT
);

CREATE TABLE IF NOT EXISTS training_tile_pool (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id      TEXT NOT NULL,
  lighting    TEXT NOT NULL,
  zone        TEXT NOT NULL,
  source      TEXT NOT NULL,
  source_path TEXT NOT NULL,
  thumb_path  TEXT,
  decision    TEXT DEFAULT 'accept'
);
CREATE INDEX IF NOT EXISTS idx_tile_pool_job ON training_tile_pool(job_id, lighting, zone, source);
""")
```

- [ ] **Step 4：跑 pass + 確認既有測試不壞**

Run: `pytest tests/test_capi_database_train.py -v`
Run: `pytest tests/ -k database -v` （確認既有 db 測試仍 pass）

- [ ] **Step 5：commit**

```bash
git add capi_database.py tests/test_capi_database_train.py
git commit -m "feat(db): add training_jobs/model_registry/training_tile_pool schema"
```

### Task 3.2 — `training_jobs` CRUD 方法

**Files:**
- Modify: `capi_database.py`
- Modify: `tests/test_capi_database_train.py`

- [ ] **Step 1：寫測試**

```python
def test_training_jobs_crud():
    with tempfile.TemporaryDirectory() as tmp:
        db = CAPIDatabase(Path(tmp) / "test.db")
        # create
        job_id = "train_GN160_20260428_153045"
        db.create_training_job(
            job_id=job_id, machine_id="GN160JCEL250S",
            panel_paths=["/p/a", "/p/b"],
        )
        # read
        job = db.get_training_job(job_id)
        assert job["machine_id"] == "GN160JCEL250S"
        assert job["state"] == "preprocess"
        assert job["panel_paths"] == ["/p/a", "/p/b"]
        # update state
        db.update_training_job_state(job_id, "review")
        assert db.get_training_job(job_id)["state"] == "review"
        # update with error
        db.update_training_job_state(job_id, "failed", error_message="OOM")
        job = db.get_training_job(job_id)
        assert job["state"] == "failed"
        assert job["error_message"] == "OOM"
```

- [ ] **Step 2：跑 fail**

- [ ] **Step 3：實作**

```python
# 加到 CAPIDatabase 類
import json

def create_training_job(self, job_id: str, machine_id: str, panel_paths: list) -> int:
    cur = self._conn.cursor()
    cur.execute(
        """INSERT INTO training_jobs (job_id, machine_id, state, started_at, panel_paths)
           VALUES (?, ?, 'preprocess', datetime('now'), ?)""",
        (job_id, machine_id, json.dumps(panel_paths)),
    )
    self._conn.commit()
    return cur.lastrowid

def get_training_job(self, job_id: str) -> dict | None:
    cur = self._conn.cursor()
    cur.execute("SELECT * FROM training_jobs WHERE job_id = ?", (job_id,))
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    job = dict(zip(cols, row))
    if job.get("panel_paths"):
        job["panel_paths"] = json.loads(job["panel_paths"])
    return job

def update_training_job_state(self, job_id: str, state: str,
                              error_message: str | None = None,
                              output_bundle: str | None = None) -> None:
    fields = ["state = ?"]
    args = [state]
    if state in ("completed", "failed"):
        fields.append("completed_at = datetime('now')")
    if error_message is not None:
        fields.append("error_message = ?")
        args.append(error_message)
    if output_bundle is not None:
        fields.append("output_bundle = ?")
        args.append(output_bundle)
    args.append(job_id)
    self._conn.cursor().execute(
        f"UPDATE training_jobs SET {', '.join(fields)} WHERE job_id = ?", tuple(args)
    )
    self._conn.commit()

def get_active_training_job(self) -> dict | None:
    """回傳目前進行中的 job（preprocess/review/train）。"""
    cur = self._conn.cursor()
    cur.execute(
        """SELECT * FROM training_jobs WHERE state IN ('preprocess','review','train')
           ORDER BY started_at DESC LIMIT 1"""
    )
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    job = dict(zip(cols, row))
    if job.get("panel_paths"):
        job["panel_paths"] = json.loads(job["panel_paths"])
    return job
```

- [ ] **Step 4：跑 pass**

- [ ] **Step 5：commit**

```bash
git add capi_database.py tests/test_capi_database_train.py
git commit -m "feat(db): add training_jobs CRUD methods"
```

### Task 3.3 — `training_tile_pool` CRUD 方法

**Files:**
- Modify: `capi_database.py`
- Modify: `tests/test_capi_database_train.py`

- [ ] **Step 1：寫測試**

```python
def test_tile_pool_crud():
    with tempfile.TemporaryDirectory() as tmp:
        db = CAPIDatabase(Path(tmp) / "test.db")
        db.create_training_job(job_id="j1", machine_id="M", panel_paths=[])
        # bulk insert
        tiles = [
            {"lighting": "G0F00000", "zone": "inner", "source": "ok",
             "source_path": "/t/1.png", "thumb_path": "/t/thumb_1.png"},
            {"lighting": "G0F00000", "zone": "edge", "source": "ok",
             "source_path": "/t/2.png", "thumb_path": "/t/thumb_2.png"},
            {"lighting": "G0F00000", "zone": None, "source": "ng",
             "source_path": "/t/n1.png", "thumb_path": "/t/thumb_n1.png"},
        ]
        ids = db.insert_tile_pool("j1", tiles)
        assert len(ids) == 3
        # query
        all_g0f = db.list_tile_pool("j1", lighting="G0F00000")
        assert len(all_g0f) == 3
        inner = db.list_tile_pool("j1", lighting="G0F00000", zone="inner")
        assert len(inner) == 1
        # update decision
        db.update_tile_decisions("j1", [ids[0]], "reject")
        assert db.list_tile_pool("j1", decision="reject") == [
            t for t in db.list_tile_pool("j1") if t["id"] == ids[0]
        ]
```

- [ ] **Step 2：跑 fail**

- [ ] **Step 3：實作**

```python
def insert_tile_pool(self, job_id: str, tiles: list[dict]) -> list[int]:
    cur = self._conn.cursor()
    ids = []
    for t in tiles:
        cur.execute(
            """INSERT INTO training_tile_pool
               (job_id, lighting, zone, source, source_path, thumb_path)
               VALUES (?,?,?,?,?,?)""",
            (job_id, t["lighting"], t.get("zone"), t["source"],
             t["source_path"], t.get("thumb_path")),
        )
        ids.append(cur.lastrowid)
    self._conn.commit()
    return ids

def list_tile_pool(self, job_id: str, lighting: str = None, zone: str = None,
                   source: str = None, decision: str = None) -> list[dict]:
    sql = "SELECT * FROM training_tile_pool WHERE job_id = ?"
    args = [job_id]
    for fld, val in [("lighting", lighting), ("zone", zone), ("source", source), ("decision", decision)]:
        if val is not None:
            sql += f" AND {fld} = ?"
            args.append(val)
    sql += " ORDER BY id"
    cur = self._conn.cursor()
    cur.execute(sql, tuple(args))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]

def update_tile_decisions(self, job_id: str, tile_ids: list[int], decision: str) -> None:
    if not tile_ids:
        return
    placeholders = ",".join("?" * len(tile_ids))
    self._conn.cursor().execute(
        f"UPDATE training_tile_pool SET decision = ? WHERE job_id = ? AND id IN ({placeholders})",
        (decision, job_id, *tile_ids),
    )
    self._conn.commit()

def cleanup_tile_pool(self, job_id: str) -> None:
    """job 完成後清此 pool（thumb 檔不刪，由 caller 處理）。"""
    self._conn.cursor().execute("DELETE FROM training_tile_pool WHERE job_id = ?", (job_id,))
    self._conn.commit()
```

- [ ] **Step 4：跑 pass**

- [ ] **Step 5：commit**

```bash
git add capi_database.py tests/test_capi_database_train.py
git commit -m "feat(db): add training_tile_pool CRUD"
```

### Task 3.4 — `model_registry` CRUD 方法

**Files:**
- Modify: `capi_database.py`
- Modify: `tests/test_capi_database_train.py`

- [ ] **Step 1：寫測試**

```python
def test_model_registry_crud():
    with tempfile.TemporaryDirectory() as tmp:
        db = CAPIDatabase(Path(tmp) / "test.db")
        # register
        bid = db.register_model_bundle({
            "machine_id": "GN160", "bundle_path": "model/GN160-20260428",
            "trained_at": "2026-04-28T15:30:45",
            "panel_count": 5, "inner_tile_count": 2400,
            "edge_tile_count": 900, "ng_tile_count": 150,
            "bundle_size_bytes": 478_000_000, "job_id": "j1",
        })
        # list
        bundles = db.list_model_bundles(machine_id="GN160")
        assert len(bundles) == 1
        assert bundles[0]["bundle_path"] == "model/GN160-20260428"
        assert bundles[0]["is_active"] == 0
        # activate
        db.set_bundle_active(bid, True)
        assert db.list_model_bundles()[0]["is_active"] == 1
        # deactivate others when activating new
        bid2 = db.register_model_bundle({
            "machine_id": "GN160", "bundle_path": "model/GN160-20260501",
            "trained_at": "2026-05-01T10:00:00", "panel_count": 5,
            "inner_tile_count": 2500, "edge_tile_count": 950,
            "ng_tile_count": 150, "bundle_size_bytes": 480_000_000, "job_id": "j2",
        })
        db.deactivate_other_bundles_for_machine("GN160", except_id=bid2)
        db.set_bundle_active(bid2, True)
        bundles = db.list_model_bundles(machine_id="GN160")
        actives = [b for b in bundles if b["is_active"] == 1]
        assert len(actives) == 1
        assert actives[0]["id"] == bid2
        # delete
        db.delete_model_bundle(bid)
        assert len(db.list_model_bundles(machine_id="GN160")) == 1
```

- [ ] **Step 2：跑 fail**

- [ ] **Step 3：實作**

```python
def register_model_bundle(self, info: dict) -> int:
    cur = self._conn.cursor()
    cur.execute(
        """INSERT INTO model_registry
           (machine_id, bundle_path, trained_at, panel_count, inner_tile_count,
            edge_tile_count, ng_tile_count, bundle_size_bytes, is_active, job_id, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (info["machine_id"], info["bundle_path"], info["trained_at"],
         info.get("panel_count"), info.get("inner_tile_count"),
         info.get("edge_tile_count"), info.get("ng_tile_count"),
         info.get("bundle_size_bytes"), 0, info.get("job_id"), info.get("notes")),
    )
    self._conn.commit()
    return cur.lastrowid

def list_model_bundles(self, machine_id: str = None) -> list[dict]:
    sql = "SELECT * FROM model_registry"
    args = ()
    if machine_id:
        sql += " WHERE machine_id = ?"
        args = (machine_id,)
    sql += " ORDER BY trained_at DESC"
    cur = self._conn.cursor()
    cur.execute(sql, args)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]

def get_model_bundle(self, bundle_id: int) -> dict | None:
    cur = self._conn.cursor()
    cur.execute("SELECT * FROM model_registry WHERE id = ?", (bundle_id,))
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))

def set_bundle_active(self, bundle_id: int, active: bool) -> None:
    self._conn.cursor().execute(
        "UPDATE model_registry SET is_active = ? WHERE id = ?",
        (1 if active else 0, bundle_id),
    )
    self._conn.commit()

def deactivate_other_bundles_for_machine(self, machine_id: str, except_id: int) -> None:
    self._conn.cursor().execute(
        "UPDATE model_registry SET is_active = 0 WHERE machine_id = ? AND id != ?",
        (machine_id, except_id),
    )
    self._conn.commit()

def delete_model_bundle(self, bundle_id: int) -> None:
    self._conn.cursor().execute("DELETE FROM model_registry WHERE id = ?", (bundle_id,))
    self._conn.commit()
```

- [ ] **Step 4：跑 pass**

- [ ] **Step 5：commit**

```bash
git add capi_database.py tests/test_capi_database_train.py
git commit -m "feat(db): add model_registry CRUD methods"
```

---

**Phase 3 完成 checkpoint**：DB schema 完整，測試 pass。

---

## Phase 4：`capi_train_new.py` 訓練 Worker

### Task 4.1 — Module 骨架 + State 結構

**Files:**
- Create: `capi_train_new.py`
- Create: `tests/test_capi_train_new_preprocess.py`

- [ ] **Step 1：寫骨架測試**

```python
# tests/test_capi_train_new_preprocess.py
def test_module_imports():
    from capi_train_new import (
        TrainingConfig, generate_job_id,
        preprocess_panels_to_pool, sample_ng_tiles,
        run_training_pipeline,
    )
    assert callable(generate_job_id)

def test_generate_job_id_format():
    from capi_train_new import generate_job_id
    job_id = generate_job_id("GN160JCEL250S")
    assert job_id.startswith("train_GN160JCEL250S_")
    assert len(job_id.split("_")) >= 4
```

- [ ] **Step 2：跑 fail**

- [ ] **Step 3：建模組骨架**

```python
# capi_train_new.py
"""新機種 PatchCore 訓練 Wizard 後端 worker。

提供：
- preprocess_panels_to_pool: Step 2 切 tile + 寫 DB
- sample_ng_tiles: 從 over_review 抽 NG
- run_training_pipeline: Step 4 訓 10 模型 + 寫 bundle
"""
from __future__ import annotations
import os
import json
import logging
import platform
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable
import numpy as np
import cv2

logger = logging.getLogger("capi.train_new")

LIGHTINGS = ("G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD")
ZONES = ("inner", "edge")
TRAINING_UNITS = [(l, z) for l in LIGHTINGS for z in ZONES]  # 10 個

MIN_TRAIN_TILES = 30
NG_TILES_PER_LIGHTING = 30


@dataclass
class TrainingConfig:
    machine_id: str
    panel_paths: List[Path]
    over_review_root: Path
    output_root: Path = Path("model")
    backbone_cache_dir: Path = Path("deployment/torch_hub_cache")
    
    batch_size: int = 8
    image_size: tuple = (512, 512)
    coreset_ratio: float = 0.1
    max_epochs: int = 1


def generate_job_id(machine_id: str) -> str:
    return f"train_{machine_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def preprocess_panels_to_pool(*args, **kwargs):
    raise NotImplementedError("Phase 4.2")


def sample_ng_tiles(*args, **kwargs):
    raise NotImplementedError("Phase 4.3")


def run_training_pipeline(*args, **kwargs):
    raise NotImplementedError("Phase 4.4-4.6")
```

- [ ] **Step 4：跑 pass**

- [ ] **Step 5：commit**

```bash
git add capi_train_new.py tests/test_capi_train_new_preprocess.py
git commit -m "feat(train_new): scaffold module skeleton"
```

### Task 4.2 — `preprocess_panels_to_pool`：Step 2 切 tile

**Files:**
- Modify: `capi_train_new.py`
- Modify: `tests/test_capi_train_new_preprocess.py`

- [ ] **Step 1：寫測試（用 fixture panel folder）**

```python
def test_preprocess_panels_to_pool_writes_tiles(tmp_path):
    """需要 fixture panel folder，每個 lighting 各 1 張圖。"""
    from capi_preprocess import PreprocessConfig
    from capi_train_new import preprocess_panels_to_pool, TrainingConfig
    
    # 準備 fake panel folders（用 Phase 1 的 fixture image 複製）
    fixture_img = Path("tests/fixtures/preprocess/synthetic_panel.png")
    panel_dir = tmp_path / "panel_a"
    panel_dir.mkdir()
    for lighting in ["G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD"]:
        target = panel_dir / f"{lighting}_x.png"
        target.write_bytes(fixture_img.read_bytes())
    
    # mock DB
    class MockDB:
        def __init__(self):
            self.tiles = []
        def insert_tile_pool(self, job_id, tiles):
            self.tiles.extend(tiles)
            return list(range(len(tiles)))
    
    db = MockDB()
    cfg = TrainingConfig(
        machine_id="TEST", panel_paths=[panel_dir],
        over_review_root=tmp_path / "or_unused",
    )
    job_id = "j_test"
    pre_cfg = PreprocessConfig(tile_size=256, edge_threshold_px=384)
    
    stats = preprocess_panels_to_pool(
        job_id=job_id, cfg=cfg, preprocess_cfg=pre_cfg,
        db=db, thumb_dir=tmp_path / "thumbs",
        log=lambda msg: None,
    )
    
    assert stats["panel_success"] == 1
    assert stats["total_tiles"] > 0
    # 應有 inner 和 edge 兩種
    zones = {t["zone"] for t in db.tiles}
    assert "inner" in zones and "edge" in zones
    # 5 lighting 都應有 tile
    lightings = {t["lighting"] for t in db.tiles}
    assert lightings == set(["G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD"])
```

- [ ] **Step 2：跑 fail**

- [ ] **Step 3：實作**

```python
# capi_train_new.py 改寫 preprocess_panels_to_pool
from capi_preprocess import (
    PreprocessConfig, preprocess_panel_folder, PanelPreprocessResult,
)


def preprocess_panels_to_pool(
    job_id: str,
    cfg: TrainingConfig,
    preprocess_cfg: PreprocessConfig,
    db,
    thumb_dir: Path,
    log: Callable[[str], None],
) -> dict:
    """將 cfg.panel_paths 全部前處理 + 切 tile + 寫 DB。"""
    thumb_dir.mkdir(parents=True, exist_ok=True)
    panel_success = 0
    panel_fail = 0
    total_tiles = 0
    
    for idx, panel_dir in enumerate(cfg.panel_paths, 1):
        log(f"[{idx}/{len(cfg.panel_paths)}] panel {panel_dir.name}")
        try:
            results = preprocess_panel_folder(panel_dir, preprocess_cfg)
        except Exception as e:
            log(f"  ✗ 處理失敗: {e}")
            panel_fail += 1
            continue
        if not results:
            log(f"  ✗ 無有效 lighting 圖")
            panel_fail += 1
            continue
        
        polygon_failed_count = sum(1 for r in results.values() if r.polygon_detection_failed)
        if polygon_failed_count > 0:
            log(f"  ⚠ {polygon_failed_count} lighting polygon 偵測失敗")
        
        # 為每張 tile 存 .png + 縮圖 + 寫 DB
        tile_records = []
        for lighting, result in results.items():
            for tile in result.tiles:
                tile_filename = f"{job_id}_{panel_dir.name}_{lighting}_t{tile.tile_id:04d}.png"
                tile_path = thumb_dir / "tiles" / tile_filename
                tile_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(tile_path), tile.image)
                
                thumb_path = thumb_dir / "thumb" / tile_filename
                thumb_path.parent.mkdir(parents=True, exist_ok=True)
                thumb = cv2.resize(tile.image, (96, 96))
                cv2.imwrite(str(thumb_path), thumb)
                
                tile_records.append({
                    "lighting": lighting, "zone": tile.zone,
                    "source": "ok",
                    "source_path": str(tile_path),
                    "thumb_path": str(thumb_path),
                })
        
        if tile_records:
            db.insert_tile_pool(job_id, tile_records)
            total_tiles += len(tile_records)
            panel_success += 1
            log(f"  ✓ 切出 {len(tile_records)} tile")
    
    return {
        "panel_success": panel_success,
        "panel_fail": panel_fail,
        "total_tiles": total_tiles,
    }
```

- [ ] **Step 4：跑 pass**

- [ ] **Step 5：commit**

```bash
git add capi_train_new.py tests/test_capi_train_new_preprocess.py
git commit -m "feat(train_new): preprocess_panels_to_pool slices tiles into DB"
```

### Task 4.3 — `sample_ng_tiles`：從 over_review 抽 NG

**Files:**
- Modify: `capi_train_new.py`
- Modify: `tests/test_capi_train_new_preprocess.py`

- [ ] **Step 1：寫測試**

```python
def test_sample_ng_tiles(tmp_path):
    from capi_train_new import sample_ng_tiles
    
    # 模擬 over_review 結構：snapshot/true_ng/<lighting>/crop/<files>.png
    or_root = tmp_path / "over_review"
    snap_a = or_root / "20260415_104812" / "true_ng"
    for lighting in ["G0F00000", "R0F00000", "STANDARD"]:
        crop_dir = snap_a / lighting / "crop"
        crop_dir.mkdir(parents=True)
        for i in range(50):
            (crop_dir / f"img_{i}.png").write_bytes(b"x")
    
    class MockDB:
        def __init__(self): self.tiles = []
        def insert_tile_pool(self, job_id, tiles):
            self.tiles.extend(tiles)
            return list(range(len(tiles)))
    
    db = MockDB()
    stats = sample_ng_tiles(
        job_id="j1", over_review_root=or_root, db=db,
        per_lighting=10, log=lambda m: None,
    )
    assert stats["sampled"] == 30  # 3 lighting × 10
    assert stats["missing_lightings"] == ["W0F00000", "WGF50500"]
    by_lighting = {}
    for t in db.tiles:
        by_lighting.setdefault(t["lighting"], 0)
        by_lighting[t["lighting"]] += 1
    assert by_lighting == {"G0F00000": 10, "R0F00000": 10, "STANDARD": 10}
    # NG zone 應為 None（不分 inner/edge）
    assert all(t["zone"] is None for t in db.tiles)
    assert all(t["source"] == "ng" for t in db.tiles)
```

- [ ] **Step 2：跑 fail**

- [ ] **Step 3：實作**

```python
def sample_ng_tiles(
    job_id: str,
    over_review_root: Path,
    db,
    per_lighting: int = NG_TILES_PER_LIGHTING,
    log: Callable[[str], None] = print,
) -> dict:
    """從 over_review/{*}/true_ng/{lighting}/crop/ 隨機抽 NG tile。"""
    if not over_review_root.exists():
        log(f"⚠ over_review 不存在: {over_review_root}，跳過 NG 抽樣")
        return {"sampled": 0, "missing_lightings": list(LIGHTINGS)}
    
    sampled = 0
    missing = []
    snapshots = [d for d in over_review_root.iterdir() if d.is_dir() and (d / "true_ng").exists()]
    
    for lighting in LIGHTINGS:
        all_files = []
        for snap in snapshots:
            crop_dir = snap / "true_ng" / lighting / "crop"
            if crop_dir.exists():
                all_files.extend(crop_dir.glob("*.png"))
        if not all_files:
            missing.append(lighting)
            log(f"⚠ {lighting}: 無 NG 樣本")
            continue
        chosen = random.sample(all_files, min(per_lighting, len(all_files)))
        records = [{
            "lighting": lighting, "zone": None, "source": "ng",
            "source_path": str(p), "thumb_path": str(p),  # NG 用原圖當縮圖
        } for p in chosen]
        db.insert_tile_pool(job_id, records)
        sampled += len(records)
        log(f"  ✓ {lighting}: 抽 {len(chosen)} 個 NG")
    
    return {"sampled": sampled, "missing_lightings": missing}
```

- [ ] **Step 4：跑 pass**

- [ ] **Step 5：commit**

```bash
git add capi_train_new.py tests/test_capi_train_new_preprocess.py
git commit -m "feat(train_new): sample_ng_tiles from over_review snapshots"
```

### Task 4.4 — `run_training_pipeline` Step A：Stage dataset + train one PatchCore

**Files:**
- Modify: `capi_train_new.py`
- Create: `tests/test_capi_train_new_training.py`

- [ ] **Step 1：寫 stage_dataset 測試**

```python
# tests/test_capi_train_new_training.py
import os
import platform
import tempfile
from pathlib import Path

def test_stage_dataset_creates_links_or_copies(tmp_path):
    from capi_train_new import stage_dataset
    
    # 假設 train tile 路徑列表
    train_dir = tmp_path / "src_train"
    train_dir.mkdir()
    for i in range(5):
        (train_dir / f"t{i}.png").write_bytes(b"x")
    train_paths = list(train_dir.glob("*.png"))
    
    ng_dir = tmp_path / "src_ng"
    ng_dir.mkdir()
    for i in range(3):
        (ng_dir / f"n{i}.png").write_bytes(b"y")
    ng_paths = list(ng_dir.glob("*.png"))
    
    staging = tmp_path / "staging" / "G0F-inner"
    stage_dataset(staging, train_paths, ng_paths)
    
    assert (staging / "train").exists()
    assert (staging / "test" / "anormal").exists()
    train_files = list((staging / "train").glob("*"))
    ng_files = list((staging / "test" / "anormal").glob("*"))
    assert len(train_files) == 5
    assert len(ng_files) == 3
```

- [ ] **Step 2：實作 stage_dataset（mklink/symlink hybrid）**

```python
def stage_dataset(staging_dir: Path, train_paths: List[Path], ng_paths: List[Path]) -> None:
    """為一個 (lighting, zone) unit 準備訓練 staging。
    
    結構：
      staging_dir/
        train/         (個別 file 的 hardlink/copy)
        test/anormal/  (個別 file)
    
    為避免 anomalib Folder 對 symlink 行為不一致，用個別檔案 link / copy
    （不是整目錄 mklink）。
    """
    train_dir = staging_dir / "train"
    ng_dir = staging_dir / "test" / "anormal"
    train_dir.mkdir(parents=True, exist_ok=True)
    ng_dir.mkdir(parents=True, exist_ok=True)
    
    for src in train_paths:
        dst = train_dir / src.name
        _link_or_copy(src, dst)
    for src in ng_paths:
        dst = ng_dir / src.name
        _link_or_copy(src, dst)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hardlink，跨平台
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
```

- [ ] **Step 3：跑 pass**

- [ ] **Step 4：寫 train_one_patchcore 測試（mock anomalib）**

```python
def test_train_one_patchcore_smoke(tmp_path, monkeypatch):
    """smoke test：mock anomalib，確認 orchestration 順序正確。"""
    from capi_train_new import train_one_patchcore
    
    staging = tmp_path / "staging"
    (staging / "train").mkdir(parents=True)
    (staging / "test" / "anormal").mkdir(parents=True)
    
    calls = []
    
    class FakeEngine:
        def __init__(self, *a, **kw): calls.append(("Engine.__init__", kw))
        def fit(self, **kw): calls.append(("fit", kw))
        def export(self, **kw):
            calls.append(("export", kw))
            # 假裝 export 會丟個 .pt 出來
            run_root = Path(kw.get("model_path", ""))
            (run_root / "weights" / "torch").mkdir(parents=True, exist_ok=True)
            (run_root / "weights" / "torch" / "model.pt").write_bytes(b"fake")
    
    class FakePatchcore:
        def __init__(self, *a, **kw): calls.append(("Patchcore", kw))
        @staticmethod
        def configure_pre_processor(image_size): return None
        pre_processor = None
    
    monkeypatch.setattr("capi_train_new._import_anomalib", lambda: (
        type("Folder", (), {"__init__": lambda s, **kw: None}),
        FakePatchcore,
        FakeEngine,
        type("ExportType", (), {"TORCH": "torch"}),
        "same_as_test",
    ))
    
    out = train_one_patchcore(
        staging_dir=staging,
        run_root=tmp_path / "run",
        unit_label="G0F-inner",
    )
    assert "fit" in [c[0] for c in calls]
    assert "export" in [c[0] for c in calls]
```

- [ ] **Step 5：實作 train_one_patchcore**

```python
def _import_anomalib():
    """延後 import，方便 unit test mock。"""
    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    try:
        from anomalib.data.utils import ValSplitMode
        val_mode = ValSplitMode.SAME_AS_TEST
    except ImportError:
        val_mode = "same_as_test"
    return Folder, Patchcore, Engine, ExportType, val_mode


def train_one_patchcore(
    staging_dir: Path,
    run_root: Path,
    unit_label: str,
    cfg: TrainingConfig = None,
) -> Path:
    """訓練一個 (lighting, zone) unit。回傳 model.pt 路徑。"""
    cfg = cfg or TrainingConfig(machine_id="?", panel_paths=[], over_review_root=Path("?"))
    Folder, Patchcore, Engine, ExportType, val_mode = _import_anomalib()
    
    if run_root.exists():
        shutil.rmtree(run_root, ignore_errors=True)
    run_root.mkdir(parents=True, exist_ok=True)
    
    datamodule = Folder(
        name=f"unit_{unit_label}",
        root=staging_dir,
        normal_dir="train",
        abnormal_dir="test/anormal",
        train_batch_size=cfg.batch_size,
        eval_batch_size=cfg.batch_size,
        num_workers=0,
        val_split_mode=val_mode,
    )
    try:
        datamodule.image_size = cfg.image_size
    except Exception:
        pass
    
    model = Patchcore(coreset_sampling_ratio=cfg.coreset_ratio)
    model.pre_processor = Patchcore.configure_pre_processor(image_size=cfg.image_size)
    
    engine = Engine(
        max_epochs=cfg.max_epochs,
        default_root_dir=str(run_root),
        callbacks=None,
    )
    engine.fit(datamodule=datamodule, model=model, model_path=str(run_root))
    engine.export(model=model, export_type=ExportType.TORCH, model_path=str(run_root))
    
    candidates = list(run_root.rglob("weights/torch/model.pt"))
    if not candidates:
        candidates = list(run_root.rglob("model.pt"))
    if not candidates:
        raise RuntimeError(f"訓練後找不到 model.pt under {run_root}")
    return candidates[0]
```

- [ ] **Step 6：跑 pass**

- [ ] **Step 7：commit**

```bash
git add capi_train_new.py tests/test_capi_train_new_training.py
git commit -m "feat(train_new): stage_dataset + train_one_patchcore"
```

### Task 4.5 — `calibrate_threshold` + manifest / yaml 寫入

**Files:**
- Modify: `capi_train_new.py`
- Modify: `tests/test_capi_train_new_training.py`

- [ ] **Step 1：寫測試**

```python
def test_calibrate_threshold_uses_p10_and_train_max():
    from capi_train_new import calibrate_threshold
    # 假設 NG scores 與 train_max
    ng_scores = sorted([0.5, 0.55, 0.6, 0.7, 0.8, 0.9])  # P10 ≈ 0.5
    threshold = calibrate_threshold(ng_scores=ng_scores, train_max_score=0.4)
    # max(P10=0.5, 0.4*1.05=0.42) → 0.5
    assert threshold == 0.5
    # train_max_score 較高的情況
    threshold = calibrate_threshold(ng_scores=ng_scores, train_max_score=0.6)
    # max(0.5, 0.63) → 0.63
    assert abs(threshold - 0.63) < 1e-6


def test_write_manifest_yaml(tmp_path):
    from capi_train_new import write_manifest, write_machine_config_yaml
    bundle = tmp_path / "GN160-20260428"
    bundle.mkdir()
    
    write_manifest(bundle, {
        "machine_id": "GN160", "trained_at": "2026-04-28T15:30:45",
        "trained_with_job_id": "j1", "panel_count": 5,
        "panel_glass_ids": ["YQ21KU218E45"],
        "edge_threshold_px": 768,
        "patchcore_params": {"batch_size": 8, "image_size": [512, 512],
                             "coreset_ratio": 0.1, "max_epochs": 1},
        "tiles_per_unit": {"G0F00000-inner": {"train": 480, "ng": 30}},
        "model_files": {"G0F00000-inner": {"path": "G0F00000-inner.pt", "size_bytes": 47340000}},
    })
    
    import json as _json
    m = _json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert m["machine_id"] == "GN160"
    assert m["version_schema"] == 1
    
    write_machine_config_yaml(bundle, "GN160", {
        "G0F00000": {"inner": 0.62, "edge": 0.71},
    })
    import yaml
    y = yaml.safe_load((bundle / "machine_config.yaml").read_text(encoding="utf-8"))
    assert y["machine_id"] == "GN160"
    assert y["model_mapping"]["G0F00000"]["inner"].endswith("G0F00000-inner.pt")
    assert y["threshold_mapping"]["G0F00000"]["inner"] == 0.62
```

- [ ] **Step 2：跑 fail**

- [ ] **Step 3：實作**

```python
def calibrate_threshold(ng_scores: List[float], train_max_score: float) -> float:
    """取 max(NG P10, train_max × 1.05)。"""
    if not ng_scores:
        return float(train_max_score) * 1.05
    sorted_scores = sorted(ng_scores)
    p10_idx = max(0, int(len(sorted_scores) * 0.10))
    p10 = float(sorted_scores[p10_idx])
    return float(max(p10, train_max_score * 1.05))


def write_manifest(bundle_dir: Path, info: dict) -> None:
    info_full = dict(info)
    info_full["version_schema"] = 1
    (bundle_dir / "manifest.json").write_text(
        json.dumps(info_full, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_thresholds(bundle_dir: Path, thresholds: Dict[str, Dict[str, float]]) -> None:
    (bundle_dir / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_machine_config_yaml(bundle_dir: Path, machine_id: str,
                              thresholds: Dict[str, Dict[str, float]]) -> None:
    """產出 bundle 內的 inference yaml。"""
    import yaml
    
    model_mapping = {}
    threshold_mapping = {}
    for lighting in LIGHTINGS:
        model_mapping[lighting] = {
            "inner": str(bundle_dir / f"{lighting}-inner.pt"),
            "edge":  str(bundle_dir / f"{lighting}-edge.pt"),
        }
        threshold_mapping[lighting] = thresholds.get(lighting, {"inner": 0.75, "edge": 0.75})
    
    cfg = {
        "machine_id": machine_id,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "bundle_path": str(bundle_dir),
        "edge_threshold_px": 768,
        "otsu_offset": 5,
        "enable_panel_polygon": True,
        "model_mapping": model_mapping,
        "threshold_mapping": threshold_mapping,
    }
    (bundle_dir / "machine_config.yaml").write_text(
        yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
```

- [ ] **Step 4：跑 pass**

- [ ] **Step 5：commit**

```bash
git add capi_train_new.py tests/test_capi_train_new_training.py
git commit -m "feat(train_new): calibrate_threshold + manifest/yaml writers"
```

### Task 4.6 — `run_training_pipeline` 整合 10 unit 訓練

**Files:**
- Modify: `capi_train_new.py`
- Modify: `tests/test_capi_train_new_training.py`

- [ ] **Step 1：寫整合測試（mock train_one_patchcore）**

```python
def test_run_training_pipeline_orchestrates_10_units(tmp_path, monkeypatch):
    from capi_train_new import run_training_pipeline, TrainingConfig, LIGHTINGS, ZONES
    
    # 準備假 tile pool
    pool = []
    for lighting in LIGHTINGS:
        for zone in ZONES:
            for i in range(50):
                p = tmp_path / "tiles" / f"{lighting}_{zone}_{i}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")
                pool.append({
                    "id": len(pool), "lighting": lighting, "zone": zone,
                    "source": "ok", "source_path": str(p), "decision": "accept",
                })
        for i in range(30):
            p = tmp_path / "tiles_ng" / f"{lighting}_ng_{i}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
            pool.append({
                "id": len(pool), "lighting": lighting, "zone": None,
                "source": "ng", "source_path": str(p), "decision": "accept",
            })
    
    class MockDB:
        def list_tile_pool(self, job_id, **filt):
            res = list(pool)
            for k, v in filt.items():
                res = [r for r in res if r.get(k) == v]
            return res
    
    db = MockDB()
    
    trained_units = []
    def fake_train(staging_dir, run_root, unit_label, cfg=None):
        trained_units.append(unit_label)
        out = run_root / "weights" / "torch" / "model.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fakemodel")
        return out
    monkeypatch.setattr("capi_train_new.train_one_patchcore", fake_train)
    monkeypatch.setattr("capi_train_new._compute_train_max_score", lambda *a, **kw: 0.5)
    monkeypatch.setattr("capi_train_new._predict_ng_scores", lambda *a, **kw: [0.6, 0.7, 0.8])
    
    cfg = TrainingConfig(
        machine_id="GN160TEST", panel_paths=[Path("p1")],
        over_review_root=tmp_path / "or",
        output_root=tmp_path / "model",
    )
    bundle_dir = run_training_pipeline(
        job_id="j1", cfg=cfg, db=db,
        gpu_lock=__import__("threading").Lock(),
        log=lambda m: None,
    )
    
    assert bundle_dir.exists()
    assert len(trained_units) == 10
    # 應有 10 個 .pt
    pts = list(bundle_dir.glob("*.pt"))
    assert len(pts) == 10
    # 應有 manifest + thresholds + yaml
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "thresholds.json").exists()
    assert (bundle_dir / "machine_config.yaml").exists()
```

- [ ] **Step 2：跑 fail**

- [ ] **Step 3：實作**

```python
def run_training_pipeline(
    job_id: str,
    cfg: TrainingConfig,
    db,
    gpu_lock,
    log: Callable[[str], None] = print,
) -> Path:
    """執行 10 unit 訓練，輸出 bundle 目錄。"""
    # 1. 環境檢查
    _setup_offline_env(cfg.backbone_cache_dir, log)
    
    bundle_dir = cfg.output_root / f"{cfg.machine_id}-{datetime.now().strftime('%Y%m%d')}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    
    thresholds: Dict[str, Dict[str, float]] = {l: {} for l in LIGHTINGS}
    tiles_per_unit: Dict[str, Dict[str, int]] = {}
    model_files: Dict[str, Dict] = {}
    success_units = 0
    
    for idx, (lighting, zone) in enumerate(TRAINING_UNITS, 1):
        unit_label = f"{lighting}-{zone}"
        log(f"[{idx}/10] {unit_label}: 載 tile")
        
        train_tiles = db.list_tile_pool(job_id, lighting=lighting, zone=zone, source="ok", decision="accept")
        ng_tiles = db.list_tile_pool(job_id, lighting=lighting, source="ng", decision="accept")
        
        if len(train_tiles) < MIN_TRAIN_TILES:
            log(f"[{idx}/10] ⚠ 跳過：tile 不足 ({len(train_tiles)} < {MIN_TRAIN_TILES})")
            continue
        
        with gpu_lock:
            staging = Path(".tmp/training_staging") / job_id / unit_label
            stage_dataset(staging,
                          [Path(t["source_path"]) for t in train_tiles],
                          [Path(t["source_path"]) for t in ng_tiles])
            run_root = Path(".tmp/training_runs") / job_id / unit_label
            try:
                model_pt = train_one_patchcore(staging, run_root, unit_label, cfg)
                # threshold
                train_max = _compute_train_max_score(model_pt, [Path(t["source_path"]) for t in train_tiles])
                ng_scores = _predict_ng_scores(model_pt, [Path(t["source_path"]) for t in ng_tiles])
                threshold = calibrate_threshold(ng_scores, train_max)
                
                # 移到 bundle
                dst_pt = bundle_dir / f"{unit_label}.pt"
                shutil.copy2(model_pt, dst_pt)
                size = dst_pt.stat().st_size
                
                thresholds[lighting][zone] = round(threshold, 4)
                tiles_per_unit[unit_label] = {"train": len(train_tiles), "ng": len(ng_tiles)}
                model_files[unit_label] = {"path": dst_pt.name, "size_bytes": size}
                success_units += 1
                log(f"[{idx}/10] ✓ done, threshold={threshold:.4f}, size={size/1e6:.1f}MB")
            finally:
                shutil.rmtree(run_root, ignore_errors=True)
                shutil.rmtree(staging, ignore_errors=True)
    
    if success_units < 5:
        raise RuntimeError(f"成功 unit 數 {success_units} < 5，訓練失敗")
    
    # 寫 bundle metadata
    write_thresholds(bundle_dir, thresholds)
    write_machine_config_yaml(bundle_dir, cfg.machine_id, thresholds)
    write_manifest(bundle_dir, {
        "machine_id": cfg.machine_id,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "trained_with_job_id": job_id,
        "panel_count": len(cfg.panel_paths),
        "panel_glass_ids": [p.name for p in cfg.panel_paths],
        "edge_threshold_px": 768,
        "patchcore_params": {
            "batch_size": cfg.batch_size,
            "image_size": list(cfg.image_size),
            "coreset_ratio": cfg.coreset_ratio,
            "max_epochs": cfg.max_epochs,
        },
        "tiles_per_unit": tiles_per_unit,
        "model_files": model_files,
        "success_units": success_units,
    })
    return bundle_dir


def _setup_offline_env(backbone_cache_dir: Path, log: Callable):
    backbone_cache_dir = Path(backbone_cache_dir).resolve()
    required = backbone_cache_dir / "hub" / "checkpoints" / "wide_resnet50_2-32ee1156.pth"
    if not required.exists():
        raise RuntimeError(f"backbone 缺檔：{required}（請先在開發機 stage 後 FTP 上傳）")
    os.environ["TORCH_HOME"] = str(backbone_cache_dir)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    log(f"✓ backbone cache: {backbone_cache_dir}")


def _compute_train_max_score(model_pt: Path, train_paths: List[Path]) -> float:
    """讀 model 對 train tile 跑一次推論取 max score（用於 threshold 1.05× 校準）。
    
    為簡化，採 sample 100 張取 max；若 anomalib API 變動可能需更新。
    """
    from anomalib.deploy import TorchInferencer
    inferencer = TorchInferencer(path=str(model_pt))
    sample = random.sample(train_paths, min(100, len(train_paths)))
    max_score = 0.0
    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            continue
        result = inferencer.predict(img)
        score = float(getattr(result, "pred_score", 0.0))
        if score > max_score:
            max_score = score
    return max_score


def _predict_ng_scores(model_pt: Path, ng_paths: List[Path]) -> List[float]:
    from anomalib.deploy import TorchInferencer
    inferencer = TorchInferencer(path=str(model_pt))
    scores = []
    for p in ng_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        result = inferencer.predict(img)
        scores.append(float(getattr(result, "pred_score", 0.0)))
    return scores
```

- [ ] **Step 4：跑 pass**

- [ ] **Step 5：commit**

```bash
git add capi_train_new.py tests/test_capi_train_new_training.py
git commit -m "feat(train_new): run_training_pipeline orchestrates 10 PatchCore units"
```

---

**Phase 4 完成 checkpoint**：核心訓練邏輯完整且可單元測試（用 mock）。實際跑訓練還需 GPU 環境。

---

## Phase 5：Wizard API Routes（`capi_web.py`）

### Task 5.1 — `_handle_train_new_panels`（DB 撈 OK panel 清單）

**Files:**
- Modify: `capi_web.py`
- Create: `tests/test_capi_web_train_new.py`

- [ ] **Step 1：寫測試**

```python
# tests/test_capi_web_train_new.py
import json
from unittest.mock import MagicMock

def test_handle_train_new_panels_returns_recent_ok():
    from capi_web import CAPIWebHandler
    
    # mock 一個 fake server instance + database
    fake_db = MagicMock()
    fake_db.query.return_value = [
        {"glass_id": "G1", "machine_no": "CAPI07", "image_path": "/p/G1",
         "ai_judgment": "OK", "timestamp": "2026-04-28 16:40:00"},
        {"glass_id": "G2", "machine_no": "CAPI07", "image_path": "/p/G2",
         "ai_judgment": "OK", "timestamp": "2026-04-28 14:22:00"},
    ]
    fake_server = MagicMock()
    fake_server.database = fake_db
    
    # 用 helper 模擬 request — 既有測試應有類似 helper
    from tests.helpers import call_route_handler  # 可能要寫
    resp = call_route_handler(
        fake_server, "GET", "/api/train/new/panels?machine_id=GN160&days=7"
    )
    body = json.loads(resp["body"])
    assert len(body["panels"]) == 2
    assert body["panels"][0]["glass_id"] == "G1"
```

⚠️ 若 `tests/helpers.py` 不存在，先建一個 minimal helper 或直接用 `BaseHTTPServer.HTTPServer` + 真 socket 呼叫。

- [ ] **Step 2：跑 fail**

- [ ] **Step 3：實作 route**

在 `capi_web.py` `do_GET` dispatch 內加：

```python
elif path == "/api/train/new/panels":
    self._handle_train_new_panels()
```

新增方法：

```python
def _handle_train_new_panels(self):
    """GET /api/train/new/panels?machine_id=X&days=7"""
    from urllib.parse import parse_qs, urlparse
    qs = parse_qs(urlparse(self.path).query)
    machine_id = (qs.get("machine_id") or [""])[0]
    days = int((qs.get("days") or ["7"])[0])
    
    if not machine_id:
        self._send_json({"error": "machine_id required"}, status=400)
        return
    
    # 用既有 database query 介面
    db = self._capi_server_instance.database
    rows = db.query(
        """SELECT glass_id, machine_no, image_path, ai_judgment, timestamp
           FROM inference_records
           WHERE model_id = ? AND machine_judgment = 'OK'
           AND timestamp >= datetime('now', ? || ' days')
           ORDER BY timestamp DESC LIMIT 100""",
        (machine_id, f"-{days}"),
    )
    self._send_json({"panels": rows})
```

⚠️ 若既有 `database.query` API 不同（可能是 `cursor()` 或 namespaced），讀 `capi_database.py` 找對應方法後調用。

- [ ] **Step 4：跑 pass**

- [ ] **Step 5：commit**

```bash
git add capi_web.py tests/test_capi_web_train_new.py
git commit -m "feat(web): add /api/train/new/panels route"
```

### Task 5.2 — `_handle_train_new_start`（建 job + 起 preprocess thread）

**Files:**
- Modify: `capi_web.py`

- [ ] **Step 1：在 `CAPIWebHandler` 類加 `_train_new_state`（class-level dict）**

```python
# 類別頂部（同 _retrain_state 風格）
_train_new_state = {
    "lock": threading.Lock(),
    "active_job_id": None,
    "thread": None,
    "log_lines": [],
    "log_lock": threading.Lock(),
}
```

- [ ] **Step 2：實作 `_handle_train_new_start`**

```python
def _handle_train_new_start(self):
    """POST /api/train/new/start
    
    body: {"machine_id": "...", "panel_paths": [...]}
    """
    try:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        params = json.loads(body) if body else {}
    except Exception:
        self._send_json({"error": "invalid JSON body"}, status=400)
        return
    
    machine_id = params.get("machine_id", "").strip()
    panel_paths = params.get("panel_paths", [])
    if not machine_id or not panel_paths:
        self._send_json({"error": "machine_id and panel_paths required"}, status=400)
        return
    if len(panel_paths) > 20:
        self._send_json({"error": "panel_paths too many (max 20)"}, status=400)
        return
    
    db = self._capi_server_instance.database
    state = self._train_new_state
    with state["lock"]:
        existing = db.get_active_training_job()
        if existing:
            self._send_json({
                "error": "job_already_running",
                "active_job_id": existing["job_id"],
                "state": existing["state"],
            }, status=409)
            return
        
        from capi_train_new import generate_job_id
        job_id = generate_job_id(machine_id)
        db.create_training_job(job_id=job_id, machine_id=machine_id, panel_paths=panel_paths)
        state["active_job_id"] = job_id
        state["log_lines"] = []
    
    # 起 preprocess thread
    thread = threading.Thread(
        target=CAPIWebHandler._train_new_preprocess_worker,
        args=(job_id, machine_id, panel_paths, self._capi_server_instance),
        daemon=True, name=f"train_new_pre-{job_id}",
    )
    thread.start()
    self._send_json({"job_id": job_id, "state": "preprocess"})
```

- [ ] **Step 3：實作 worker**

```python
@staticmethod
def _train_new_preprocess_worker(job_id, machine_id, panel_paths, server_inst):
    """背景 thread：preprocess + 抽 NG → state=review。"""
    import traceback
    from pathlib import Path as _Path
    from capi_train_new import (
        TrainingConfig, preprocess_panels_to_pool, sample_ng_tiles,
    )
    from capi_preprocess import PreprocessConfig
    
    db = server_inst.database
    state = CAPIWebHandler._train_new_state
    
    def log(msg):
        with state["log_lock"]:
            state["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    
    try:
        thumb_root = _Path(".tmp/train_new_thumbs") / job_id
        cfg = TrainingConfig(
            machine_id=machine_id,
            panel_paths=[_Path(p) for p in panel_paths],
            over_review_root=_Path("/aidata/capi_ai/datasets/over_review"),
        )
        pre_cfg = PreprocessConfig()
        
        log(f"開始前處理 {len(panel_paths)} panel")
        stats = preprocess_panels_to_pool(
            job_id=job_id, cfg=cfg, preprocess_cfg=pre_cfg,
            db=db, thumb_dir=thumb_root, log=log,
        )
        if stats["panel_success"] < 4:
            raise RuntimeError(f"成功 panel < 4 ({stats['panel_success']})")
        
        log(f"抽 NG tile（每 lighting {30} 個）")
        ng_stats = sample_ng_tiles(
            job_id=job_id, over_review_root=cfg.over_review_root,
            db=db, per_lighting=30, log=log,
        )
        
        db.update_training_job_state(job_id, "review")
        log("✓ 進入 review 階段")
    except Exception as e:
        traceback.print_exc()
        db.update_training_job_state(job_id, "failed", error_message=str(e))
        log(f"✗ 失敗: {e}")
    finally:
        with state["lock"]:
            if state["active_job_id"] == job_id:
                # preprocess 完成不解鎖（仍在 review）；失敗才解鎖
                job = db.get_training_job(job_id)
                if job and job["state"] in ("failed",):
                    state["active_job_id"] = None
```

- [ ] **Step 4：手動驗證（無 GPU）**

啟 server、丟空 panel_paths：應回 400。
丟有效 panel_paths（fixture）：應回 200 + job_id。

- [ ] **Step 5：commit**

```bash
git add capi_web.py
git commit -m "feat(web): add /api/train/new/start + preprocess worker"
```

### Task 5.3 — `_handle_train_new_status`（poll endpoint）

**Files:**
- Modify: `capi_web.py`

- [ ] **Step 1：實作**

```python
def _handle_train_new_status(self):
    """GET /api/train/new/status?job_id=X"""
    from urllib.parse import parse_qs, urlparse
    qs = parse_qs(urlparse(self.path).query)
    job_id = (qs.get("job_id") or [""])[0]
    
    db = self._capi_server_instance.database
    state = self._train_new_state
    
    if not job_id:
        # 無 job_id 回最近 active job 狀態
        job = db.get_active_training_job()
        if not job:
            self._send_json({"state": "idle"})
            return
        job_id = job["job_id"]
    else:
        job = db.get_training_job(job_id)
        if not job:
            self._send_json({"error": "job not found"}, status=404)
            return
    
    with state["log_lock"]:
        log_lines = list(state["log_lines"][-100:])
    
    resp = {
        "job_id": job["job_id"], "machine_id": job["machine_id"],
        "state": job["state"],
        "started_at": job["started_at"], "completed_at": job["completed_at"],
        "output_bundle": job["output_bundle"], "error_message": job["error_message"],
        "log_lines": log_lines,
    }
    self._send_json(resp)
```

- [ ] **Step 2：dispatch 加 entry**

```python
elif path.startswith("/api/train/new/status"):
    self._handle_train_new_status()
```

- [ ] **Step 3：commit**

```bash
git add capi_web.py
git commit -m "feat(web): add /api/train/new/status route"
```

### Task 5.4 — `_handle_train_new_tiles` + decision API

**Files:**
- Modify: `capi_web.py`

- [ ] **Step 1：實作 GET tiles**

```python
def _handle_train_new_tiles(self):
    """GET /api/train/new/tiles?job_id=X&lighting=Y"""
    from urllib.parse import parse_qs, urlparse
    qs = parse_qs(urlparse(self.path).query)
    job_id = (qs.get("job_id") or [""])[0]
    lighting = (qs.get("lighting") or [""])[0]
    if not job_id or not lighting:
        self._send_json({"error": "job_id and lighting required"}, status=400)
        return
    db = self._capi_server_instance.database
    tiles = db.list_tile_pool(job_id, lighting=lighting)
    self._send_json({"tiles": tiles})
```

- [ ] **Step 2：實作 POST decision**

```python
def _handle_train_new_tiles_decision(self):
    """POST /api/train/new/tiles/decision
    body: {"job_id": "...", "tile_ids": [int, ...], "decision": "accept"|"reject"}
    """
    try:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
    except Exception:
        self._send_json({"error": "invalid JSON"}, status=400)
        return
    
    job_id = body.get("job_id")
    tile_ids = body.get("tile_ids", [])
    decision = body.get("decision")
    
    if not job_id or not tile_ids or decision not in ("accept", "reject"):
        self._send_json({"error": "job_id, tile_ids, decision required"}, status=400)
        return
    
    db = self._capi_server_instance.database
    db.update_tile_decisions(job_id, tile_ids, decision)
    self._send_json({"ok": True, "updated": len(tile_ids)})
```

- [ ] **Step 3：dispatch 加 entries**

```python
# do_GET：
elif path == "/api/train/new/tiles":
    self._handle_train_new_tiles()
# do_POST：
elif path == "/api/train/new/tiles/decision":
    self._handle_train_new_tiles_decision()
```

- [ ] **Step 4：靜態 file serve**

`thumb_path` 是相對路徑 `.tmp/train_new_thumbs/...`。需 route 服務這些 png：

```python
# do_GET 內加：
elif path.startswith("/api/train/new/thumb/"):
    self._handle_train_new_thumb()

def _handle_train_new_thumb(self):
    """GET /api/train/new/thumb/<job_id>/<filename>"""
    from urllib.parse import unquote
    parts = self.path.split("/api/train/new/thumb/", 1)[1].split("?")[0]
    parts = unquote(parts)
    safe = Path(".tmp/train_new_thumbs").resolve()
    target = (safe / parts).resolve()
    if not str(target).startswith(str(safe)):
        self._send_response(403, "")
        return
    if not target.exists():
        self._send_response(404, "")
        return
    self._send_file(target, content_type="image/png")
```

⚠️ 確認 `_send_file` helper 存在；若無則用既有 thumb serve 模式。

- [ ] **Step 5：commit**

```bash
git add capi_web.py
git commit -m "feat(web): add /api/train/new/tiles + decision + thumb serve"
```

### Task 5.5 — `_handle_train_new_start_training`（觸發訓練 thread）

**Files:**
- Modify: `capi_web.py`

- [ ] **Step 1：實作**

```python
def _handle_train_new_start_training(self):
    """POST /api/train/new/start_training/<job_id>"""
    job_id = self.path.rsplit("/", 1)[-1].split("?")[0]
    db = self._capi_server_instance.database
    job = db.get_training_job(job_id)
    if not job:
        self._send_json({"error": "job not found"}, status=404)
        return
    if job["state"] != "review":
        self._send_json({"error": f"job state must be 'review', currently '{job['state']}'"}, status=409)
        return
    
    db.update_training_job_state(job_id, "train")
    
    thread = threading.Thread(
        target=CAPIWebHandler._train_new_training_worker,
        args=(job_id, job["machine_id"], job["panel_paths"], self._capi_server_instance),
        daemon=True, name=f"train_new-{job_id}",
    )
    thread.start()
    self._send_json({"ok": True, "state": "train"})


@staticmethod
def _train_new_training_worker(job_id, machine_id, panel_paths, server_inst):
    """背景 thread：跑 10 unit 訓練 + 註冊 bundle。"""
    import traceback
    from pathlib import Path as _Path
    from capi_train_new import TrainingConfig, run_training_pipeline
    
    db = server_inst.database
    state = CAPIWebHandler._train_new_state
    
    def log(msg):
        with state["log_lock"]:
            state["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    
    try:
        cfg = TrainingConfig(
            machine_id=machine_id,
            panel_paths=[_Path(p) for p in panel_paths],
            over_review_root=_Path("/aidata/capi_ai/datasets/over_review"),
            output_root=_Path("model"),
        )
        bundle_dir = run_training_pipeline(
            job_id=job_id, cfg=cfg, db=db,
            gpu_lock=server_inst._gpu_lock,
            log=log,
        )
        
        # 註冊到 model_registry
        sizes = sum(p.stat().st_size for p in bundle_dir.glob("*.pt"))
        manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        inner_count = sum(v["train"] for k, v in manifest["tiles_per_unit"].items() if k.endswith("-inner"))
        edge_count = sum(v["train"] for k, v in manifest["tiles_per_unit"].items() if k.endswith("-edge"))
        ng_count = sum(v["ng"] for v in manifest["tiles_per_unit"].values())
        
        db.register_model_bundle({
            "machine_id": machine_id,
            "bundle_path": str(bundle_dir),
            "trained_at": manifest["trained_at"],
            "panel_count": manifest["panel_count"],
            "inner_tile_count": inner_count,
            "edge_tile_count": edge_count,
            "ng_tile_count": ng_count,
            "bundle_size_bytes": sizes,
            "job_id": job_id,
        })
        
        db.update_training_job_state(job_id, "completed", output_bundle=str(bundle_dir))
        log(f"✓ 訓練完成，bundle={bundle_dir}")
    except Exception as e:
        traceback.print_exc()
        db.update_training_job_state(job_id, "failed", error_message=str(e))
        log(f"✗ 訓練失敗: {e}")
    finally:
        with state["lock"]:
            if state["active_job_id"] == job_id:
                state["active_job_id"] = None
```

- [ ] **Step 2：dispatch 加 entry**

```python
elif path.startswith("/api/train/new/start_training/"):
    self._handle_train_new_start_training()
```

- [ ] **Step 3：commit**

```bash
git add capi_web.py
git commit -m "feat(web): add /api/train/new/start_training + training worker"
```

---

**Phase 5 完成 checkpoint**：所有 wizard 後端 API 完成，可用 curl 測試完整流程。前端尚未做。

---

## Phase 6：Wizard 前端 Templates

### Task 6.1 — `step1_select.html`（panel picker）

**Files:**
- Create: `templates/train_new/step1_select.html`
- Modify: `capi_web.py`

- [ ] **Step 1：建模板（Jinja2，extends base.html）**

整體結構參考 brainstorm mockup `.superpowers/brainstorm/.../step1-mockup.html`，搬到 Jinja2：

```html
{% extends "base.html" %}
{% block title %}訓練新機種 - Step 1{% endblock %}
{% block content %}
<div style="max-width:900px;margin:0 auto;padding:24px 16px;">
  <div style="margin-bottom:14px;">
    <a href="/training" style="color:#a6adc8;text-decoration:none;font-size:.9rem;">← 返回模型訓練</a>
  </div>
  <h1 style="color:#cdd6f4;margin:0 0 6px 0;font-size:1.4rem;">Step 1 / 5 · 選擇訓練資料</h1>
  <p style="color:#a6adc8;font-size:.9rem;">勾選 5 片 AOI 判 OK 的 panel 作為 OK 訓練樣本來源</p>

  <div style="background:#313244;border-radius:10px;padding:18px;margin-top:16px;">
    <div style="display:flex;gap:10px;align-items:center;margin-bottom:12px;">
      <span style="color:#a6adc8;font-size:.85rem;width:90px;">機種 ID</span>
      <input id="machine-id" type="text" placeholder="例：GN160JCEL250S"
        style="flex:1;background:#1e1e2e;border:1px solid #45475a;border-radius:5px;color:#cdd6f4;padding:7px 10px;font-family:monospace;">
      <button id="search-btn" onclick="searchPanels()"
        style="background:#89b4fa;color:#1e1e2e;border:none;border-radius:5px;padding:7px 16px;font-weight:700;">查詢</button>
    </div>
    <div style="display:flex;gap:6px;align-items:center;">
      <span style="color:#a6adc8;font-size:.82rem;">時間範圍</span>
      <span class="day-pill" data-d="1" onclick="setDays(1)">1 天</span>
      <span class="day-pill active" data-d="7" onclick="setDays(7)">7 天</span>
      <span class="day-pill" data-d="30" onclick="setDays(30)">30 天</span>
    </div>
  </div>

  <div id="panel-list" style="margin-top:16px;"></div>

  <div style="margin-top:18px;display:flex;justify-content:space-between;align-items:center;">
    <span id="sel-count" style="color:#a6e3a1;font-weight:600;">已選 0/5 片</span>
    <span>
      <a href="/training" style="background:#313244;color:#a6adc8;padding:8px 18px;border-radius:5px;text-decoration:none;">取消</a>
      <button id="next-btn" onclick="submitStart()" disabled
        style="background:#89b4fa;color:#1e1e2e;border:none;border-radius:5px;padding:8px 22px;font-weight:700;opacity:.4;">
        下一步：前處理 →
      </button>
    </span>
  </div>
</div>

<style>
  .day-pill { background:#313244;padding:5px 11px;border-radius:13px;color:#a6adc8;font-size:.78rem;cursor:pointer; }
  .day-pill.active { background:#89b4fa;color:#1e1e2e;font-weight:700; }
</style>

<script>
let _days = 7;
let _selected = new Set();

function setDays(d) {
  _days = d;
  document.querySelectorAll('.day-pill').forEach(p => p.classList.toggle('active', parseInt(p.dataset.d) === d));
  if (document.getElementById('machine-id').value.trim()) searchPanels();
}

function searchPanels() {
  const mid = document.getElementById('machine-id').value.trim();
  if (!mid) return;
  fetch(`/api/train/new/panels?machine_id=${encodeURIComponent(mid)}&days=${_days}`)
    .then(r => r.json())
    .then(d => renderPanels(d.panels || []));
}

function renderPanels(panels) {
  _selected = new Set();
  const root = document.getElementById('panel-list');
  if (!panels.length) {
    root.innerHTML = '<div style="background:#313244;padding:24px;border-radius:10px;color:#a6adc8;text-align:center;">無符合條件的 panel</div>';
    updateNext();
    return;
  }
  let html = '<div style="background:#313244;border-radius:10px;overflow:hidden;"><table style="width:100%;border-collapse:collapse;">';
  html += '<thead><tr style="background:#11111b;color:#a6adc8;font-size:.78rem;text-align:left;"><th style="padding:8px;width:30px;"></th><th style="padding:8px;">Glass ID</th><th>日期</th><th>機台</th><th>AI 判定</th></tr></thead><tbody>';
  for (const p of panels) {
    html += `<tr data-path="${p.image_path}" style="border-top:1px solid #1e1e2e;cursor:pointer;" onclick="toggleSelect(this)">
      <td style="padding:8px;"><span class="chk" style="width:14px;height:14px;border:2px solid #45475a;border-radius:3px;display:inline-block;"></span></td>
      <td style="padding:8px;font-family:monospace;color:#cdd6f4;font-size:.82rem;">${p.glass_id}</td>
      <td style="padding:8px;color:#a6adc8;font-size:.82rem;">${p.timestamp}</td>
      <td style="padding:8px;color:#a6adc8;font-size:.82rem;">${p.machine_no}</td>
      <td style="padding:8px;font-size:.82rem;color:${p.ai_judgment === 'OK' ? '#a6e3a1' : '#fab387'};">${p.ai_judgment || '-'}</td>
    </tr>`;
  }
  html += '</tbody></table></div>';
  root.innerHTML = html;
  updateNext();
}

function toggleSelect(tr) {
  const path = tr.dataset.path;
  if (_selected.has(path)) {
    _selected.delete(path);
    tr.style.background = '';
    tr.querySelector('.chk').style.background = '';
  } else {
    if (_selected.size >= 5) return;
    _selected.add(path);
    tr.style.background = '#1e3a4d';
    const chk = tr.querySelector('.chk');
    chk.style.background = '#89b4fa';
    chk.style.borderColor = '#89b4fa';
  }
  updateNext();
}

function updateNext() {
  document.getElementById('sel-count').textContent = `已選 ${_selected.size}/5 片`;
  const btn = document.getElementById('next-btn');
  btn.disabled = _selected.size !== 5;
  btn.style.opacity = btn.disabled ? .4 : 1;
}

function submitStart() {
  const mid = document.getElementById('machine-id').value.trim();
  fetch('/api/train/new/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({machine_id: mid, panel_paths: Array.from(_selected)}),
  }).then(r => r.json()).then(d => {
    if (d.error) { alert('錯誤: ' + d.error + (d.active_job_id ? ` (進行中: ${d.active_job_id})` : '')); return; }
    window.location.href = '/train/new/progress?job_id=' + encodeURIComponent(d.job_id);
  });
}
</script>
{% endblock %}
```

- [ ] **Step 2：route handler**

在 `capi_web.py`：

```python
def _handle_train_new_page(self):
    """GET /train/new"""
    db = self._capi_server_instance.database
    active = db.get_active_training_job()
    if active:
        # 提示繼續或拋棄
        # 為 MVP：直接 redirect 到對應 step page
        if active["state"] == "review":
            self.send_response(302)
            self.send_header("Location", f"/train/new/review/{active['job_id']}")
            self.end_headers()
            return
        elif active["state"] in ("preprocess", "train"):
            self.send_response(302)
            self.send_header("Location", f"/train/new/progress?job_id={active['job_id']}")
            self.end_headers()
            return
    
    template = self.jinja_env.get_template("train_new/step1_select.html")
    html = template.render(request_path="/train/new")
    self._send_response(200, html)

# do_GET 加：
elif path == "/train/new":
    self._handle_train_new_page()
```

- [ ] **Step 3：手動驗證**

啟 server，連 `http://localhost:8080/train/new`：
- 顯示空表 panel picker
- 輸入機種 ID 點查詢 → 應撈 DB 有資料

- [ ] **Step 4：commit**

```bash
git add templates/train_new/step1_select.html capi_web.py
git commit -m "feat(web): step1 panel picker page"
```

### Task 6.2 — `step2_progress.html`（前處理進度頁）

**Files:**
- Create: `templates/train_new/step2_progress.html`
- Modify: `capi_web.py`

- [ ] **Step 1：建模板**

```html
{% extends "base.html" %}
{% block title %}前處理 - Step 2{% endblock %}
{% block content %}
<div style="max-width:900px;margin:0 auto;padding:24px 16px;">
  <h1 style="color:#cdd6f4;margin:0 0 6px 0;font-size:1.3rem;">Step 2 / 5 · 前處理 + 切 tile</h1>
  <p style="color:#a6adc8;font-size:.85rem;">系統正在處理 panel 圖片，這需要幾分鐘。</p>

  <div id="status-pill" style="display:inline-block;background:#fab387;color:#1e1e2e;padding:5px 14px;border-radius:14px;font-weight:700;margin-top:14px;">前處理中</div>

  <div style="background:#313244;border-radius:10px;padding:14px;margin-top:16px;">
    <div style="color:#a6adc8;font-size:.82rem;margin-bottom:8px;">處理 log</div>
    <pre id="log-box" style="background:#1e1e2e;border-radius:6px;padding:12px;margin:0;font-size:.78rem;color:#cdd6f4;max-height:420px;overflow-y:auto;white-space:pre-wrap;"></pre>
  </div>
</div>

<script>
const params = new URLSearchParams(location.search);
const jobId = params.get('job_id');
let _timer = null;

function poll() {
  fetch(`/api/train/new/status?job_id=${encodeURIComponent(jobId)}`)
    .then(r => r.json())
    .then(d => {
      if (d.log_lines) document.getElementById('log-box').textContent = d.log_lines.join('\n');
      const pill = document.getElementById('status-pill');
      pill.textContent = stateText(d.state);
      if (d.state === 'review') {
        clearInterval(_timer);
        location.href = `/train/new/review/${encodeURIComponent(jobId)}`;
      } else if (d.state === 'failed') {
        clearInterval(_timer);
        pill.style.background = '#f38ba8';
        pill.textContent = '失敗：' + (d.error_message || '未知');
      } else if (d.state === 'completed') {
        clearInterval(_timer);
        location.href = `/train/new/done/${encodeURIComponent(jobId)}`;
      }
    });
}

function stateText(s) {
  return {preprocess:'前處理中', review:'準備 review', train:'訓練中', completed:'完成', failed:'失敗'}[s] || s;
}

_timer = setInterval(poll, 3000);
poll();
</script>
{% endblock %}
```

- [ ] **Step 2：route**

```python
elif path == "/train/new/progress":
    self._handle_train_new_progress_page()

def _handle_train_new_progress_page(self):
    """GET /train/new/progress?job_id=X"""
    template = self.jinja_env.get_template("train_new/step2_progress.html")
    html = template.render(request_path="/train/new/progress")
    self._send_response(200, html)
```

- [ ] **Step 3：commit**

```bash
git add templates/train_new/step2_progress.html capi_web.py
git commit -m "feat(web): step2 preprocess progress page"
```

### Task 6.3 — `step3_review.html`（tile review）

**Files:**
- Create: `templates/train_new/step3_review.html`
- Modify: `capi_web.py`

- [ ] **Step 1：建模板（簡化版，主要 layout + tab + tile grid + decision API 連接）**

模板較長，採以下結構（跟 brainstorm mockup 對齊）：
- 5 個 lighting tab，點擊切換
- 4 個 sub-group：內部 OK / 內部 NG / 邊緣 OK / 邊緣 NG
- 每 tile 縮圖點擊切換 accept/reject，立刻 POST decision API
- 「開始訓練」按鈕送 POST start_training

詳細模板：

```html
{% extends "base.html" %}
{% block title %}Tile 審核 - Step 3{% endblock %}
{% block content %}
<div style="max-width:1100px;margin:0 auto;padding:24px 16px;">
  <h1 style="color:#cdd6f4;margin:0 0 12px 0;font-size:1.3rem;">Step 3 / 5 · 訓練 tile 審核</h1>

  <div style="display:flex;gap:4px;border-bottom:1px solid #45475a;padding-bottom:8px;margin-bottom:14px;overflow-x:auto;">
    {% for lighting in ["G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD"] %}
    <span class="lighting-tab" data-lighting="{{ lighting }}" onclick="switchLighting('{{ lighting }}')"
      style="background:#313244;padding:6px 14px;border-radius:6px 6px 0 0;color:#a6adc8;font-size:.82rem;cursor:pointer;white-space:nowrap;">
      {{ lighting }} <span class="badge" style="background:#1e1e2e;padding:1px 7px;border-radius:8px;font-size:.72rem;margin-left:3px;">0/0</span>
    </span>
    {% endfor %}
  </div>

  <div id="content-area"></div>

  <div style="margin-top:18px;display:flex;justify-content:space-between;align-items:center;padding-top:12px;border-top:1px solid #45475a;">
    <a href="/train/new" style="background:#313244;color:#a6adc8;padding:8px 18px;border-radius:5px;text-decoration:none;">返回</a>
    <button onclick="startTraining()"
      style="background:#89b4fa;color:#1e1e2e;border:none;border-radius:5px;padding:8px 22px;font-weight:700;">
      開始訓練 →
    </button>
  </div>
</div>

<style>
  .lighting-tab.active { background:#89b4fa;color:#1e1e2e;font-weight:700; }
  .grp-head { background:#313244;padding:7px 12px;margin:8px 0 5px 0;border-radius:5px;border-left:3px solid #89b4fa;color:#cdd6f4;font-size:.85rem;display:flex;justify-content:space-between; }
  .grp-head.edge { border-left-color:#f9e2af; }
  .grp-head.ng { border-left-color:#f38ba8; }
  .grid { display:grid;grid-template-columns:repeat(12,1fr);gap:4px; }
  .tile { aspect-ratio:1;border-radius:3px;background:#3b4257;border:2px solid transparent;cursor:pointer;background-size:cover;background-position:center; }
  .tile.accept { border-color:#a6e3a1; }
  .tile.reject { border-color:#f38ba8;opacity:.5; }
</style>

<script>
const jobId = "{{ job_id }}";
let _currentLighting = 'G0F00000';
let _tilesByLighting = {};  // {lighting: [tile, ...]}

function loadLighting(lighting) {
  return fetch(`/api/train/new/tiles?job_id=${encodeURIComponent(jobId)}&lighting=${lighting}`)
    .then(r => r.json())
    .then(d => { _tilesByLighting[lighting] = d.tiles || []; });
}

function switchLighting(lighting) {
  _currentLighting = lighting;
  document.querySelectorAll('.lighting-tab').forEach(t => t.classList.toggle('active', t.dataset.lighting === lighting));
  if (!_tilesByLighting[lighting]) {
    loadLighting(lighting).then(render);
  } else {
    render();
  }
}

function render() {
  const tiles = _tilesByLighting[_currentLighting] || [];
  const groups = {
    'inner_ok': tiles.filter(t => t.zone === 'inner' && t.source === 'ok'),
    'inner_ng': tiles.filter(t => t.source === 'ng'),  // NG 不分 zone
    'edge_ok':  tiles.filter(t => t.zone === 'edge' && t.source === 'ok'),
  };
  let html = '';
  for (const [key, list] of Object.entries(groups)) {
    if (!list.length) continue;
    const accept = list.filter(t => t.decision === 'accept').length;
    const isEdge = key.startsWith('edge'), isNg = key.endsWith('ng');
    const cls = isNg ? 'ng' : (isEdge ? 'edge' : '');
    const title = {inner_ok:'內部 OK', inner_ng:'NG (來自 over_review)', edge_ok:'邊緣 OK'}[key];
    html += `<div class="grp-head ${cls}"><span>📦 ${title}</span><span style="color:#a6adc8;font-size:.78rem;">${accept}/${list.length} 加入</span></div>`;
    html += '<div class="grid">';
    for (const t of list) {
      const cls = t.decision === 'reject' ? 'reject' : 'accept';
      const thumb = `/api/train/new/thumb${t.thumb_path.replace(/^.*\.tmp\/train_new_thumbs/, '')}`;
      html += `<div class="tile ${cls}" data-id="${t.id}" style="background-image:url('${thumb}');" onclick="toggleTile(this)"></div>`;
    }
    html += '</div>';
  }
  document.getElementById('content-area').innerHTML = html;
  updateBadge(_currentLighting);
}

function toggleTile(el) {
  const id = parseInt(el.dataset.id);
  const tile = _tilesByLighting[_currentLighting].find(t => t.id === id);
  const newDec = tile.decision === 'accept' ? 'reject' : 'accept';
  fetch('/api/train/new/tiles/decision', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({job_id: jobId, tile_ids: [id], decision: newDec}),
  }).then(() => {
    tile.decision = newDec;
    el.className = 'tile ' + newDec;
    render();
  });
}

function updateBadge(lighting) {
  const tiles = _tilesByLighting[lighting] || [];
  const accept = tiles.filter(t => t.decision === 'accept').length;
  const tab = document.querySelector(`.lighting-tab[data-lighting="${lighting}"] .badge`);
  if (tab) tab.textContent = `${accept}/${tiles.length}`;
}

function startTraining() {
  if (!confirm('開始訓練 10 個 PatchCore 模型？預計 60-120 分鐘。')) return;
  fetch(`/api/train/new/start_training/${encodeURIComponent(jobId)}`, {method: 'POST'})
    .then(r => r.json())
    .then(d => {
      if (d.error) { alert('錯誤: ' + d.error); return; }
      location.href = `/train/new/progress?job_id=${encodeURIComponent(jobId)}`;
    });
}

// 初始載入 5 個 lighting（一次先預載）
Promise.all(['G0F00000','R0F00000','W0F00000','WGF50500','STANDARD'].map(loadLighting))
  .then(() => switchLighting('G0F00000'));
</script>
{% endblock %}
```

- [ ] **Step 2：route**

```python
elif path.startswith("/train/new/review/"):
    self._handle_train_new_review_page()

def _handle_train_new_review_page(self):
    job_id = self.path.split("/")[-1]
    db = self._capi_server_instance.database
    job = db.get_training_job(job_id)
    if not job:
        self._send_response(404, "Job not found")
        return
    template = self.jinja_env.get_template("train_new/step3_review.html")
    html = template.render(request_path="/train/new/review", job_id=job_id)
    self._send_response(200, html)
```

- [ ] **Step 3：commit**

```bash
git add templates/train_new/step3_review.html capi_web.py
git commit -m "feat(web): step3 tile review page"
```

### Task 6.4 — `step4_progress.html`（訓練進度）

**Files:**
- Create: `templates/train_new/step4_progress.html`

- [ ] **Step 1：建模板（複用 step2 結構，加 10 步驟 chip）**

實際上 step2 與 step4 邏輯非常相似（poll status、show log），可共用一個模板叫 `progress.html`。為簡化，**用同一份 step2_progress.html 服務兩階段**，模板內依 state 顯示對應 chip。已經在 task 6.2 中實作 — 不需要新模板。

- [ ] **Step 2：commit（無檔案變更，跳過）**

實際工作併入 task 6.2，本 task 用於文件確認。

### Task 6.5 — `step5_done.html`（完成頁）

**Files:**
- Create: `templates/train_new/step5_done.html`
- Modify: `capi_web.py`

- [ ] **Step 1：建模板**

```html
{% extends "base.html" %}
{% block title %}訓練完成{% endblock %}
{% block content %}
<div style="max-width:900px;margin:0 auto;padding:24px 16px;">
  <h1 style="color:#a6e3a1;margin:0 0 6px 0;">✓ 訓練完成</h1>
  
  <div style="background:#313244;border-radius:10px;padding:18px;margin-top:14px;">
    <table style="width:100%;border-collapse:collapse;font-size:.9rem;">
      <tr><td style="color:#a6adc8;padding:6px 0;width:140px;">機種</td><td style="color:#cdd6f4;font-family:monospace;">{{ machine_id }}</td></tr>
      <tr><td style="color:#a6adc8;padding:6px 0;">Bundle</td><td style="color:#cdd6f4;font-family:monospace;font-size:.82rem;">{{ bundle_path }}</td></tr>
      <tr><td style="color:#a6adc8;padding:6px 0;">訓練 Job</td><td style="color:#cdd6f4;font-family:monospace;font-size:.82rem;">{{ job_id }}</td></tr>
    </table>
  </div>
  
  <h2 style="color:#cdd6f4;margin:18px 0 8px 0;font-size:1.05rem;">10 個子模型摘要</h2>
  <div style="background:#313244;border-radius:10px;padding:14px;">
    <table style="width:100%;border-collapse:collapse;font-size:.82rem;">
      <thead><tr style="background:#11111b;color:#a6adc8;text-align:left;">
        <th style="padding:6px 8px;">Lighting</th><th>Zone</th><th>Train</th><th>NG</th><th>Threshold</th><th>Size</th>
      </tr></thead>
      <tbody>
        {% for unit_label, info in units %}
        <tr style="border-top:1px solid #1e1e2e;">
          <td style="padding:6px 8px;color:#cdd6f4;font-family:monospace;">{{ info.lighting }}</td>
          <td style="color:#cdd6f4;">{{ info.zone }}</td>
          <td style="color:#cdd6f4;">{{ info.train }}</td>
          <td style="color:#cdd6f4;">{{ info.ng }}</td>
          <td style="color:#cdd6f4;font-family:monospace;">{{ "%.4f"|format(info.threshold) }}</td>
          <td style="color:#a6adc8;">{{ "%.1f"|format(info.size_mb) }} MB</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  
  <div style="margin-top:18px;padding:12px;background:#1e1e2e;border-left:3px solid #89b4fa;border-radius:6px;color:#a6adc8;font-size:.88rem;">
    部署：到模型庫頁點啟用 + 匯出 ZIP → FTP 上傳到 production
  </div>
  
  <div style="margin-top:18px;text-align:right;">
    <a href="/" style="background:#313244;color:#a6adc8;padding:8px 18px;border-radius:5px;text-decoration:none;">回首頁</a>
    <a href="/models" style="background:#89b4fa;color:#1e1e2e;padding:8px 22px;border-radius:5px;text-decoration:none;font-weight:700;margin-left:6px;">前往模型庫 →</a>
  </div>
</div>
{% endblock %}
```

- [ ] **Step 2：route**

```python
elif path.startswith("/train/new/done/"):
    self._handle_train_new_done_page()

def _handle_train_new_done_page(self):
    job_id = self.path.split("/")[-1]
    db = self._capi_server_instance.database
    job = db.get_training_job(job_id)
    if not job or job["state"] != "completed":
        self._send_response(404, "Job not done")
        return
    
    bundle_path = Path(job["output_bundle"])
    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    thresholds = json.loads((bundle_path / "thresholds.json").read_text(encoding="utf-8"))
    
    units = []
    for unit_label, tile_info in manifest["tiles_per_unit"].items():
        lighting, zone = unit_label.rsplit("-", 1)
        size_bytes = manifest["model_files"][unit_label]["size_bytes"]
        units.append((unit_label, {
            "lighting": lighting, "zone": zone,
            "train": tile_info["train"], "ng": tile_info["ng"],
            "threshold": thresholds.get(lighting, {}).get(zone, 0.0),
            "size_mb": size_bytes / 1e6,
        }))
    
    template = self.jinja_env.get_template("train_new/step5_done.html")
    html = template.render(
        request_path="/train/new/done",
        machine_id=job["machine_id"],
        bundle_path=str(bundle_path),
        job_id=job_id, units=units,
    )
    self._send_response(200, html)
```

- [ ] **Step 3：commit**

```bash
git add templates/train_new/step5_done.html capi_web.py
git commit -m "feat(web): step5 done summary page"
```

### Task 6.6 — `/training` hub 加新機種卡

**Files:**
- Modify: `capi_web.py`

- [ ] **Step 1：找 `_build_training_cards` 修改**

在現有 `_build_training_cards` 末尾 append：

```python
# 新機種 PatchCore card
new_arch_count = len(db.list_model_bundles())  # 全部 bundle 數
active_count = sum(1 for b in db.list_model_bundles() if b["is_active"])

cards.append({
    "title": "新機種 PatchCore",
    "subtitle": "C-10 (5 lighting × inner+edge)",
    "description": f"為新機種訓練 10 個模型 bundle。已啟用 {active_count} 個 / 共 {new_arch_count} bundle。",
    "bundle_path": "model/<機種>-<日期>",
    "trained_at": "—" if new_arch_count == 0 else "詳見模型庫",
    "status": "ok" if active_count > 0 else "warning",
    "status_text": f"{active_count} 啟用" if active_count else "未訓練",
    "target_url": "/train/new",
})
```

- [ ] **Step 2：手動驗證**

`http://localhost:8080/training` 應看到 2 張卡（刮痕 + 新機種）。

- [ ] **Step 3：commit**

```bash
git add capi_web.py
git commit -m "feat(web): add new-arch PatchCore card to /training hub"
```

---

**Phase 6 完成 checkpoint**：完整 wizard 5 step UI 可用，可從 `/training` → `/train/new` → step1-5 走完整流程。

---

## Phase 7：`capi_model_registry.py` 模型庫後端

### Task 7.1 — Module 骨架 + 列表查詢

**Files:**
- Create: `capi_model_registry.py`
- Create: `tests/test_capi_model_registry.py`

- [ ] **Step 1：寫測試**

```python
# tests/test_capi_model_registry.py
import tempfile
from pathlib import Path

def test_list_bundles_grouped(tmp_path):
    from capi_database import CAPIDatabase
    from capi_model_registry import list_bundles_grouped
    
    db = CAPIDatabase(tmp_path / "test.db")
    db.register_model_bundle({
        "machine_id": "GN160", "bundle_path": "model/GN160-20260428",
        "trained_at": "2026-04-28T15:30:45", "panel_count": 5,
        "inner_tile_count": 2400, "edge_tile_count": 900,
        "ng_tile_count": 150, "bundle_size_bytes": 478_000_000, "job_id": "j1",
    })
    db.register_model_bundle({
        "machine_id": "OTHER", "bundle_path": "model/OTHER-20260420",
        "trained_at": "2026-04-20T10:00:00", "panel_count": 3,
        "inner_tile_count": 1200, "edge_tile_count": 400,
        "ng_tile_count": 100, "bundle_size_bytes": 300_000_000, "job_id": "j2",
    })
    
    grouped = list_bundles_grouped(db)
    assert "GN160" in grouped
    assert "OTHER" in grouped
    assert len(grouped["GN160"]) == 1
```

- [ ] **Step 2：實作**

```python
# capi_model_registry.py
"""模型庫 CRUD：列表、啟用/停用、刪除、ZIP 匯出。"""
from __future__ import annotations
import io
import json
import shutil
import zipfile
import yaml
from pathlib import Path
from typing import Dict, List, Optional


def list_bundles_grouped(db) -> Dict[str, List[dict]]:
    """所有 bundle 依 machine_id 分組。"""
    bundles = db.list_model_bundles()
    grouped: Dict[str, List[dict]] = {}
    for b in bundles:
        grouped.setdefault(b["machine_id"], []).append(b)
    return grouped


def get_bundle_detail(db, bundle_id: int) -> Optional[dict]:
    """讀 manifest.json + thresholds.json 並合併 DB row。"""
    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        return None
    bundle_path = Path(bundle["bundle_path"])
    manifest_p = bundle_path / "manifest.json"
    thresholds_p = bundle_path / "thresholds.json"
    
    bundle["manifest"] = json.loads(manifest_p.read_text(encoding="utf-8")) if manifest_p.exists() else None
    bundle["thresholds"] = json.loads(thresholds_p.read_text(encoding="utf-8")) if thresholds_p.exists() else None
    return bundle
```

- [ ] **Step 3：跑 pass**

- [ ] **Step 4：commit**

```bash
git add capi_model_registry.py tests/test_capi_model_registry.py
git commit -m "feat(registry): module skeleton + list_bundles_grouped"
```

### Task 7.2 — 啟用 / 停用（修改 `server_config.yaml`）

**Files:**
- Modify: `capi_model_registry.py`
- Modify: `tests/test_capi_model_registry.py`

- [ ] **Step 1：寫測試**

```python
def test_activate_bundle_writes_server_config(tmp_path):
    from capi_database import CAPIDatabase
    from capi_model_registry import activate_bundle, deactivate_bundle
    
    # 設定 fake server_config.yaml
    sc_path = tmp_path / "server_config.yaml"
    sc_path.write_text(yaml.dump({"model_configs": ["configs/capi_3f.yaml"]}))
    
    bundle_dir = tmp_path / "model" / "GN160-20260428"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "machine_config.yaml").write_text(yaml.dump({"machine_id": "GN160"}))
    
    db = CAPIDatabase(tmp_path / "test.db")
    bid = db.register_model_bundle({
        "machine_id": "GN160", "bundle_path": str(bundle_dir),
        "trained_at": "2026-04-28T15:30:45", "panel_count": 5,
        "inner_tile_count": 2400, "edge_tile_count": 900,
        "ng_tile_count": 150, "bundle_size_bytes": 478_000_000, "job_id": "j1",
    })
    
    activate_bundle(db, bid, server_config_path=sc_path)
    
    sc = yaml.safe_load(sc_path.read_text())
    assert any("machine_config.yaml" in p for p in sc["model_configs"])
    assert db.get_model_bundle(bid)["is_active"] == 1
    
    deactivate_bundle(db, bid, server_config_path=sc_path)
    sc = yaml.safe_load(sc_path.read_text())
    assert not any("GN160-20260428" in p for p in sc["model_configs"])
    assert db.get_model_bundle(bid)["is_active"] == 0
```

- [ ] **Step 2：實作**

```python
def activate_bundle(db, bundle_id: int, server_config_path: Path) -> dict:
    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        raise ValueError(f"bundle {bundle_id} not found")
    yaml_rel = str(Path(bundle["bundle_path"]) / "machine_config.yaml")
    
    # 同 machine 其他 bundle 從 server_config 移除 + 設 inactive
    for other in db.list_model_bundles(machine_id=bundle["machine_id"]):
        if other["id"] == bundle_id:
            continue
        other_yaml = str(Path(other["bundle_path"]) / "machine_config.yaml")
        _remove_from_model_configs(server_config_path, other_yaml)
        db.set_bundle_active(other["id"], False)
    
    # 加入此 yaml
    _add_to_model_configs(server_config_path, yaml_rel)
    db.set_bundle_active(bundle_id, True)
    return {"ok": True, "message": "已啟用，請重啟 server 才會生效"}


def deactivate_bundle(db, bundle_id: int, server_config_path: Path) -> dict:
    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        raise ValueError(f"bundle {bundle_id} not found")
    yaml_rel = str(Path(bundle["bundle_path"]) / "machine_config.yaml")
    _remove_from_model_configs(server_config_path, yaml_rel)
    db.set_bundle_active(bundle_id, False)
    return {"ok": True, "message": "已停用，請重啟 server 才會生效"}


def _add_to_model_configs(server_config_path: Path, yaml_rel: str) -> None:
    cfg = yaml.safe_load(server_config_path.read_text(encoding="utf-8")) or {}
    configs = cfg.setdefault("model_configs", [])
    if yaml_rel not in configs:
        configs.append(yaml_rel)
    server_config_path.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _remove_from_model_configs(server_config_path: Path, yaml_rel: str) -> None:
    cfg = yaml.safe_load(server_config_path.read_text(encoding="utf-8")) or {}
    configs = cfg.get("model_configs", [])
    cfg["model_configs"] = [p for p in configs if p != yaml_rel]
    server_config_path.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
```

- [ ] **Step 3：跑 pass**

- [ ] **Step 4：commit**

```bash
git add capi_model_registry.py tests/test_capi_model_registry.py
git commit -m "feat(registry): activate/deactivate with server_config.yaml edits"
```

### Task 7.3 — 刪除 + 安全檢查

**Files:**
- Modify: `capi_model_registry.py`
- Modify: `tests/test_capi_model_registry.py`

- [ ] **Step 1：寫測試**

```python
def test_delete_active_bundle_rejected(tmp_path):
    from capi_database import CAPIDatabase
    from capi_model_registry import delete_bundle
    db = CAPIDatabase(tmp_path / "test.db")
    bid = db.register_model_bundle({
        "machine_id": "GN160", "bundle_path": str(tmp_path / "model/x"),
        "trained_at": "2026-04-28T15:30:45", "panel_count": 5,
        "inner_tile_count": 0, "edge_tile_count": 0, "ng_tile_count": 0,
        "bundle_size_bytes": 0, "job_id": "j1",
    })
    db.set_bundle_active(bid, True)
    sc = tmp_path / "server_config.yaml"
    sc.write_text("model_configs: []")
    
    import pytest
    with pytest.raises(ValueError, match="active"):
        delete_bundle(db, bid, server_config_path=sc)


def test_delete_inactive_bundle_removes_dir(tmp_path):
    from capi_database import CAPIDatabase
    from capi_model_registry import delete_bundle
    db = CAPIDatabase(tmp_path / "test.db")
    bdir = tmp_path / "model" / "y"
    bdir.mkdir(parents=True)
    (bdir / "x.pt").write_bytes(b"x")
    bid = db.register_model_bundle({
        "machine_id": "G", "bundle_path": str(bdir),
        "trained_at": "2026-04-28T15:30:45", "panel_count": 5,
        "inner_tile_count": 0, "edge_tile_count": 0, "ng_tile_count": 0,
        "bundle_size_bytes": 0, "job_id": "j1",
    })
    sc = tmp_path / "server_config.yaml"
    sc.write_text("model_configs: []")
    delete_bundle(db, bid, server_config_path=sc)
    assert not bdir.exists()
    assert db.get_model_bundle(bid) is None
```

- [ ] **Step 2：實作**

```python
def delete_bundle(db, bundle_id: int, server_config_path: Path) -> dict:
    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        raise ValueError(f"bundle {bundle_id} not found")
    if bundle["is_active"]:
        raise ValueError("bundle is active; deactivate first")
    
    bundle_path = Path(bundle["bundle_path"])
    yaml_rel = str(bundle_path / "machine_config.yaml")
    _remove_from_model_configs(server_config_path, yaml_rel)
    
    if bundle_path.exists():
        shutil.rmtree(bundle_path, ignore_errors=False)
    db.delete_model_bundle(bundle_id)
    return {"ok": True}
```

- [ ] **Step 3：跑 pass**

- [ ] **Step 4：commit**

```bash
git add capi_model_registry.py tests/test_capi_model_registry.py
git commit -m "feat(registry): delete bundle with safety check"
```

### Task 7.4 — Export ZIP

**Files:**
- Modify: `capi_model_registry.py`
- Modify: `tests/test_capi_model_registry.py`

- [ ] **Step 1：寫測試**

```python
def test_export_zip_streams(tmp_path):
    from capi_model_registry import export_bundle_zip
    bundle = tmp_path / "model" / "GN160-20260428"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text('{"machine_id":"GN160"}')
    (bundle / "machine_config.yaml").write_text("machine_id: GN160")
    (bundle / "G0F00000-inner.pt").write_bytes(b"\x00" * 1024)
    
    zip_bytes = export_bundle_zip(bundle, machine_id="GN160")
    assert isinstance(zip_bytes, bytes)
    
    import zipfile, io
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = z.namelist()
        assert any("manifest.json" in n for n in names)
        assert any("machine_config.yaml" in n for n in names)
        assert any(n.endswith(".pt") for n in names)
        assert any("README.txt" in n for n in names)
```

- [ ] **Step 2：實作**

```python
def export_bundle_zip(bundle_path: Path, machine_id: str) -> bytes:
    """打包成 ZIP（內含 README）。回 bytes 給 streaming response。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 整個 bundle 目錄
        for p in bundle_path.rglob("*"):
            if p.is_file():
                arcname = Path(bundle_path.parent.name) / bundle_path.name / p.relative_to(bundle_path)
                # bundle_path may be `model/<machine>-<date>` → arcname `model/<machine>-<date>/...`
                arcname = Path("model") / bundle_path.name / p.relative_to(bundle_path)
                zf.write(p, str(arcname))
        # README
        readme = _build_readme(machine_id, bundle_path)
        zf.writestr(str(Path("model") / bundle_path.name / "README.txt"), readme)
    return buf.getvalue()


def _build_readme(machine_id: str, bundle_path: Path) -> str:
    return f"""新機種 PatchCore Bundle 部署說明
────────────────────────────────────────
機種：{machine_id}
Bundle：{bundle_path.name}

部署步驟：
1. 解壓本 ZIP，保留路徑結構
2. FTP 上傳整個 bundle 目錄到 production：
     model/{bundle_path.name}/  → /capi_ai/model/{bundle_path.name}/
3. 編輯 production 的 server_config.yaml，在 model_configs 列表加入：
     - model/{bundle_path.name}/machine_config.yaml
4. （可選）若同機種有舊 bundle 想停用，從 model_configs 移除舊 bundle 的 yaml
5. 重啟 capi_server 服務

驗證：傳送該機種 panel 給 inference，confirm 走新架構（log 顯示 "load 10 models"）。
"""
```

- [ ] **Step 3：跑 pass**

- [ ] **Step 4：commit**

```bash
git add capi_model_registry.py tests/test_capi_model_registry.py
git commit -m "feat(registry): export_bundle_zip with deployment README"
```

---

**Phase 7 完成 checkpoint**：模型庫後端可呼叫，可單元測試。

---

## Phase 8：模型庫前端 + 路由整合

### Task 8.1 — `templates/models.html`

**Files:**
- Create: `templates/models.html`

- [ ] **Step 1：建模板**

參考 brainstorm mockup `.superpowers/brainstorm/.../models-mockup.html`，搬到 Jinja2：

```html
{% extends "base.html" %}
{% block title %}模型庫{% endblock %}
{% block content %}
<div style="max-width:1100px;margin:0 auto;padding:24px 16px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
    <h1 style="color:#cdd6f4;margin:0;">模型庫</h1>
    <a href="/train/new" style="background:#89b4fa;color:#1e1e2e;padding:7px 16px;border-radius:5px;font-weight:700;text-decoration:none;font-size:.85rem;">+ 新機種訓練</a>
  </div>

  {% for machine_id, bundles in grouped.items() %}
  <div style="background:#11111b;border-radius:8px;padding:14px;margin-bottom:14px;">
    <h2 style="color:#cdd6f4;margin:0 0 10px 0;font-size:1rem;">{{ machine_id }} <span style="color:#a6adc8;font-size:.78rem;font-weight:400;">({{ bundles|length }} bundle)</span></h2>
    {% for b in bundles %}
    <div style="background:#313244;border-radius:8px;padding:14px 16px;margin-bottom:8px;{% if b.is_active %}border-left:3px solid #a6e3a1;{% endif %}">
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:14px;align-items:center;">
        <div>
          <h3 style="color:#cdd6f4;margin:0 0 3px 0;font-size:.95rem;font-family:monospace;">{{ b.bundle_path|replace("model/", "") }}</h3>
          <div style="color:#a6adc8;font-size:.74rem;">{{ b.trained_at }} · job {{ b.job_id }}</div>
          {% if b.is_active %}<div style="color:#a6e3a1;font-size:.78rem;margin-top:3px;">● 啟用中</div>{% endif %}
        </div>
        <div style="font-size:.78rem;color:#a6adc8;line-height:1.5;">
          <div>Panel：<b style="color:#cdd6f4;">{{ b.panel_count }}</b> · Inner：<b style="color:#cdd6f4;">{{ b.inner_tile_count }}</b> · Edge：<b style="color:#cdd6f4;">{{ b.edge_tile_count }}</b></div>
          <div>NG：<b style="color:#cdd6f4;">{{ b.ng_tile_count }}</b> · 大小：<b style="color:#cdd6f4;">{{ "%.1f"|format(b.bundle_size_bytes / 1e6) }} MB</b></div>
        </div>
        <div style="font-size:.78rem;color:#a6adc8;">
          <div>路徑：<code style="font-size:.7rem;color:#cdd6f4;">{{ b.bundle_path }}</code></div>
        </div>
        <div style="display:flex;flex-direction:column;gap:4px;">
          <button onclick="showDetail({{ b.id }})" style="background:#1e1e2e;color:#a6adc8;border:1px solid #45475a;padding:5px 10px;border-radius:4px;font-size:.74rem;">📋 細節</button>
          {% if b.is_active %}
            <button onclick="doAction({{ b.id }}, 'deactivate')" style="background:#fab387;color:#1e1e2e;border:none;padding:5px 10px;border-radius:4px;font-size:.74rem;font-weight:700;">⏸ 停用</button>
          {% else %}
            <button onclick="doAction({{ b.id }}, 'activate')" style="background:#a6e3a1;color:#1e1e2e;border:none;padding:5px 10px;border-radius:4px;font-size:.74rem;font-weight:700;">▶ 啟用</button>
          {% endif %}
          <a href="/api/models/{{ b.id }}/export" style="background:#1e1e2e;color:#a6adc8;border:1px solid #45475a;padding:5px 10px;border-radius:4px;font-size:.74rem;text-decoration:none;text-align:center;">📦 匯出 ZIP</a>
          <button onclick="doDelete({{ b.id }})" style="background:#1e1e2e;color:#f38ba8;border:1px solid #f38ba8;padding:5px 10px;border-radius:4px;font-size:.74rem;">🗑 刪除</button>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>
  {% endfor %}

  {% if not grouped %}
  <div style="background:#313244;padding:32px;border-radius:10px;text-align:center;color:#a6adc8;">
    尚未訓練任何模型。點右上「+ 新機種訓練」開始。
  </div>
  {% endif %}
</div>

<script>
function doAction(id, action) {
  fetch(`/api/models/${id}/${action}`, {method:'POST'})
    .then(r => r.json())
    .then(d => {
      if (d.error) alert(d.error);
      else { alert(d.message || '已完成'); location.reload(); }
    });
}

function doDelete(id) {
  if (!confirm('確定要刪除這個 bundle？此操作不可恢復。')) return;
  fetch(`/api/models/${id}/delete`, {method:'POST'})
    .then(r => r.json())
    .then(d => {
      if (d.error) alert(d.error);
      else location.reload();
    });
}

function showDetail(id) {
  fetch(`/api/models/${id}/detail`)
    .then(r => r.json())
    .then(d => {
      // 簡化：用 alert 顯示 manifest（後續可改 modal）
      alert(JSON.stringify(d.manifest, null, 2));
    });
}
</script>
{% endblock %}
```

- [ ] **Step 2：commit**

```bash
git add templates/models.html
git commit -m "feat(web): models registry page template"
```

### Task 8.2 — Model registry routes

**Files:**
- Modify: `capi_web.py`

- [ ] **Step 1：實作 routes**

```python
def _handle_models_page(self):
    """GET /models"""
    db = self._capi_server_instance.database
    from capi_model_registry import list_bundles_grouped
    grouped = list_bundles_grouped(db)
    template = self.jinja_env.get_template("models.html")
    html = template.render(request_path="/models", grouped=grouped)
    self._send_response(200, html)


def _handle_models_detail(self):
    """GET /api/models/<id>/detail"""
    parts = self.path.split("/")
    bundle_id = int(parts[3])
    from capi_model_registry import get_bundle_detail
    detail = get_bundle_detail(self._capi_server_instance.database, bundle_id)
    if not detail:
        self._send_json({"error": "not found"}, status=404)
        return
    self._send_json(detail)


def _handle_models_activate(self):
    parts = self.path.split("/")
    bundle_id = int(parts[3])
    from capi_model_registry import activate_bundle
    try:
        result = activate_bundle(
            self._capi_server_instance.database,
            bundle_id,
            server_config_path=Path(self._capi_server_instance.server_config_path),
        )
        self._send_json(result)
    except ValueError as e:
        self._send_json({"error": str(e)}, status=400)


def _handle_models_deactivate(self):
    parts = self.path.split("/")
    bundle_id = int(parts[3])
    from capi_model_registry import deactivate_bundle
    try:
        result = deactivate_bundle(
            self._capi_server_instance.database, bundle_id,
            server_config_path=Path(self._capi_server_instance.server_config_path),
        )
        self._send_json(result)
    except ValueError as e:
        self._send_json({"error": str(e)}, status=400)


def _handle_models_delete(self):
    parts = self.path.split("/")
    bundle_id = int(parts[3])
    from capi_model_registry import delete_bundle
    try:
        result = delete_bundle(
            self._capi_server_instance.database, bundle_id,
            server_config_path=Path(self._capi_server_instance.server_config_path),
        )
        self._send_json(result)
    except ValueError as e:
        self._send_json({"error": str(e)}, status=409)


def _handle_models_export(self):
    """GET /api/models/<id>/export → 串流 ZIP"""
    parts = self.path.split("/")
    bundle_id = int(parts[3])
    from capi_model_registry import export_bundle_zip
    db = self._capi_server_instance.database
    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        self._send_response(404, "")
        return
    
    zip_bytes = export_bundle_zip(Path(bundle["bundle_path"]), bundle["machine_id"])
    filename = f"{Path(bundle['bundle_path']).name}.zip"
    self.send_response(200)
    self.send_header("Content-Type", "application/zip")
    self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
    self.send_header("Content-Length", str(len(zip_bytes)))
    self.end_headers()
    self.wfile.write(zip_bytes)
```

- [ ] **Step 2：dispatch entries**

```python
# do_GET：
elif path == "/models":
    self._handle_models_page()
elif path.startswith("/api/models/") and path.endswith("/detail"):
    self._handle_models_detail()
elif path.startswith("/api/models/") and path.endswith("/export"):
    self._handle_models_export()

# do_POST：
elif path.startswith("/api/models/") and path.endswith("/activate"):
    self._handle_models_activate()
elif path.startswith("/api/models/") and path.endswith("/deactivate"):
    self._handle_models_deactivate()
elif path.startswith("/api/models/") and path.endswith("/delete"):
    self._handle_models_delete()
```

- [ ] **Step 3：確保 `_capi_server_instance.server_config_path` 屬性存在**

在 `capi_server.py` 啟動時記下：

```python
self.server_config_path = server_config_path  # __init__ 收的參數
```

- [ ] **Step 4：手動驗證**

啟 server，連 `/models` → 看到列表頁。

- [ ] **Step 5：commit**

```bash
git add capi_web.py capi_server.py
git commit -m "feat(web): /models page + 5 action API routes"
```

---

**Phase 8 完成 checkpoint**：模型庫頁可瀏覽 / 啟用 / 停用 / 刪除 / 匯出。

---

## Phase 9：Inference 端整合（`capi_inference.py`）

⚠️ 這 phase 改動 inference 行為，需謹慎，每個 task 完成後跑既有 inference 測試。

### Task 9.1 — `capi_inference` import `capi_preprocess`，內部 polygon 邏輯改用共用模組

**Files:**
- Modify: `capi_inference.py`

- [ ] **Step 1：把 `_find_panel_polygon` 改成 thin wrapper**

```python
# capi_inference.py 內，把 _find_panel_polygon 全部 body 替換為：
from capi_preprocess import _polyfit_polygon as _pf_polygon

def _find_panel_polygon(self, binary_mask, bbox):
    return _pf_polygon(binary_mask, bbox, self.config.tile_size)
```

⚠️ 留 wrapper 而非直接刪除：避免別處 reference 失敗。

- [ ] **Step 2：跑既有 polygon 測試**

Run: `pytest tests/test_panel_polygon_*.py tests/test_aoi_*.py -v`
Expected: all pass

- [ ] **Step 3：commit**

```bash
git add capi_inference.py
git commit -m "refactor(inference): _find_panel_polygon delegates to capi_preprocess"
```

### Task 9.2 — `capi_inference` 加 `_process_panel_v2` 雙路徑分發

**Files:**
- Modify: `capi_inference.py`

- [ ] **Step 1：在 CAPIInferencer 構造階段判斷分發**

```python
# capi_inference.py CAPIInferencer.__init__ 內加：
def __init__(self, config: CAPIConfig, ...):
    # ... 既有 ...
    if config.is_new_architecture:
        self._dispatch_process_panel = self._process_panel_v2
    else:
        self._dispatch_process_panel = self._process_panel_v1  # 既有 process_panel 改名

# 把舊 process_panel 改名 _process_panel_v1
# 新 process_panel 變 dispatcher：
def process_panel(self, panel_dir, ...):
    return self._dispatch_process_panel(panel_dir, ...)
```

⚠️ 重命名要小心：grep 既有 callers，確保都 reference 到 dispatcher。

- [ ] **Step 2：實作 v2 path（先放 stub，下個 task 補完整邏輯）**

```python
def _process_panel_v2(self, panel_dir, *args, **kwargs):
    """新架構：依 tile zone routing inner/edge model。"""
    raise NotImplementedError("Phase 9.3")
```

- [ ] **Step 3：跑既有測試確認 legacy 路徑不變**

Run: `pytest tests/test_aoi_coord_inference.py tests/test_inference.py -v` （或對應既有測試）
Expected: all pass

- [ ] **Step 4：commit**

```bash
git add capi_inference.py
git commit -m "refactor(inference): split process_panel into v1/v2 dispatcher"
```

### Task 9.3 — 實作 `_process_panel_v2` 完整邏輯

**Files:**
- Modify: `capi_inference.py`

- [ ] **Step 1：實作 v2**

```python
def _process_panel_v2(self, panel_dir, *args, **kwargs):
    """新架構流程：
    1. capi_preprocess.preprocess_panel_folder 切 tile + 分流 inner/edge
    2. 對每張 tile 呼叫對應 model（inner.pt / edge.pt）
    3. 後處理（dust filter / bomb / heatmap）沿用既有 v1 helper 函式
    """
    from capi_preprocess import preprocess_panel_folder, PreprocessConfig
    
    pre_cfg = PreprocessConfig(
        tile_size=self.config.tile_size,
        otsu_offset=self.config.otsu_offset,
        enable_panel_polygon=self.config.enable_panel_polygon,
        edge_threshold_px=self.config.edge_threshold_px,
    )
    panel_results = preprocess_panel_folder(Path(panel_dir), pre_cfg)
    if not panel_results:
        return self._build_empty_panel_result(panel_dir, "no valid lighting files")
    
    # 對每 lighting 跑推論
    image_results = []
    for lighting, result in panel_results.items():
        inner_path = self.config.model_mapping.get(lighting, {}).get("inner")
        edge_path  = self.config.model_mapping.get(lighting, {}).get("edge")
        if not inner_path or not edge_path:
            logger.warning(f"{lighting}: 缺對應 model，跳過")
            continue
        
        inner_thr = self.config.threshold_mapping.get(lighting, {}).get("inner", 0.75)
        edge_thr  = self.config.threshold_mapping.get(lighting, {}).get("edge",  0.75)
        
        anomaly_tiles = []
        for tile in result.tiles:
            if tile.zone == "inner":
                model = self._get_or_load_model_v2(lighting, "inner", inner_path)
                threshold = inner_thr
            else:
                model = self._get_or_load_model_v2(lighting, "edge", edge_path)
                threshold = edge_thr
            
            score, anomaly_map = self._predict_tile(model, tile.image, tile.mask)
            if score > threshold:
                anomaly_tiles.append((tile, score, anomaly_map))
        
        # 後處理：dust filter / bomb check / heatmap — 沿用既有 helpers
        # ... 這裡需要把 v1 的後處理 helpers 提取出來給 v2 共用
        # （見 step 2）
        image_results.append(self._postprocess_image(
            lighting, result, anomaly_tiles, panel_dir,
        ))
    
    return self._aggregate_panel_result(panel_dir, image_results)


def _get_or_load_model_v2(self, lighting: str, zone: str, model_path: str):
    """新架構 lazy loading，cache key 含 zone。"""
    if not hasattr(self, "_model_cache_v2"):
        self._model_cache_v2 = {}
    key = (lighting, zone)
    if key not in self._model_cache_v2:
        from anomalib.deploy import TorchInferencer
        self._model_cache_v2[key] = TorchInferencer(path=model_path)
    return self._model_cache_v2[key]


def _predict_tile(self, model, tile_img, mask=None):
    """跑 PatchCore 推論 + 應用 polygon mask（如有）。"""
    result = model.predict(tile_img)
    score = float(getattr(result, "pred_score", 0.0))
    anomaly_map = getattr(result, "anomaly_map", None)
    if mask is not None and anomaly_map is not None:
        # mask 標 panel 內外，外部分數歸 0
        anomaly_map = anomaly_map * (mask.astype(np.float32) / 255.0)
        score = float(anomaly_map.max())
    return score, anomaly_map
```

⚠️ `_postprocess_image` / `_aggregate_panel_result` / `_build_empty_panel_result` 需從 v1 流程抽出共用 helper（refactor，可能要 ~50 行調整）。詳細做法：閱讀 `_process_panel_v1` 後段（dust filter / bomb / heatmap / aggregate），把這些步驟拆出 `def _postprocess_image(self, lighting, result, anomaly_tiles, panel_dir)` 共用方法。

- [ ] **Step 2：實作 helpers + v1 也改用同 helper 確保一致**

這是 refactor task，主要是把 v1 末段邏輯封裝。詳細方式由 implementer 看現有 code 決定。

- [ ] **Step 3：手動驗證**

準備一個 fixture 機種 yaml + 假 bundle（10 個 fake .pt），跑 inference：
- 既有 3F 走 v1 → 行為不變
- 新機種走 v2 → 不 crash，能拿到 result

- [ ] **Step 4：commit**

```bash
git add capi_inference.py
git commit -m "feat(inference): _process_panel_v2 with inner/edge model routing"
```

### Task 9.4 — 移除 v2 路徑的 MARK / exclusion 邏輯

**Files:**
- Modify: `capi_inference.py`

- [ ] **Step 1：確認 v2 不依賴 MARK / exclusion**

讀 `_process_panel_v2` 與 helpers，confirm 沒有 reference：
- `find_mark_region`
- `calculate_exclusion_regions`
- `mark_template`
- `exclusion_zones`

若有，把這些呼叫 wrap 在 `if not config.is_new_architecture:` 條件內。

- [ ] **Step 2：v1 路徑保留 MARK / exclusion 邏輯不動**

確認 `_process_panel_v1` 仍呼叫上述函式（legacy 行為）。

- [ ] **Step 3：commit**

```bash
git add capi_inference.py
git commit -m "feat(inference): v2 path skips MARK template / exclusion zones"
```

### Task 9.5 — `capi_server.py` request dispatcher 整合

**Files:**
- Modify: `capi_server.py`

- [ ] **Step 1：找處理 request 的位置（dispatch model_id 到 inferencer）**

```python
# 之前的 inferencer 取得方式（既有 lazy load by prefix）需擴充：
def _get_or_create_inferencer(self, model_id: str) -> CAPIInferencer:
    if model_id not in self.inferencers:
        cfg = self.configs_by_machine.get(model_id, self.fallback_config)
        self.inferencers[model_id] = CAPIInferencer(cfg, ...)
    return self.inferencers[model_id]

# request 處理：
def handle_aoi_request(self, req):
    inferencer = self._get_or_create_inferencer(req.model_id)
    return inferencer.process_panel(req.image_dir, ...)
```

⚠️ 既有 server inferencer 結構可能單一實例 — 改成 dict by machine。讀現有 `capi_server.py` 結構後再改。

- [ ] **Step 2：啟動 health check**

```python
def _health_check_models(self):
    for cfg in self.configs_by_machine.values():
        if not cfg.is_new_architecture:
            continue
        for lighting, mapping in cfg.model_mapping.items():
            for zone in ("inner", "edge"):
                p = mapping.get(zone)
                if p and not Path(p).exists():
                    raise RuntimeError(f"模型缺檔：{p}（machine={cfg.machine_id}, lighting={lighting}, zone={zone}）")

# __init__ 末段：
self._health_check_models()
```

- [ ] **Step 3：手動驗證**

啟 server，連舊機種 panel：走 v1，行為不變。
連新機種 panel（fake bundle）：走 v2，不 crash。

- [ ] **Step 4：commit**

```bash
git add capi_server.py
git commit -m "feat(server): per-machine inferencer dispatch + startup health check"
```

---

**Phase 9 完成 checkpoint**：Inference 端支援新舊架構並存，舊機種行為不變，新機種 routing 到 inner/edge model。

---

## Phase 10：Cleanup + E2E 驗證

### Task 10.1 — 標 `tools/build_bga_tiles.py` 與 `tools/train_bga_all.py` deprecated

**Files:**
- Modify: `tools/build_bga_tiles.py`
- Modify: `tools/train_bga_all.py`

- [ ] **Step 1：在 docstring 頂部加 deprecation warning**

兩個檔案都加：

```python
"""[DEPRECATED] 此腳本將被 capi_train_new 取代。
新機種訓練請用 web wizard `/train/new`，舊機種（CAPI 3F）遷移完成後此檔可刪除。
"""

import warnings
warnings.warn(
    "tools/build_bga_tiles.py is deprecated; use capi_train_new + /train/new wizard",
    DeprecationWarning, stacklevel=2,
)
```

- [ ] **Step 2：commit**

```bash
git add tools/build_bga_tiles.py tools/train_bga_all.py
git commit -m "docs: mark legacy training scripts as deprecated"
```

### Task 10.2 — 更新 `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1：更新「Architecture」章節**

在「Inference Pipeline」區塊後加：

```markdown
## Multi-Architecture Support

System now supports two model architectures concurrently:

- **Legacy 5-model (CAPI 3F)**: Single model per lighting, MARK/exclusion mask in inference. Config: `configs/capi_3f.yaml`.
- **New C-10 (per-machine)**: 10 models per machine = 5 lighting × (inner + edge). Config auto-generated at `model/<machine>-<date>/machine_config.yaml`. No MARK template.

`server_config.yaml.model_configs` lists active configs. Inference dispatches by `model_id` from request.

## Training

- **Legacy** (deprecated): `tools/build_bga_tiles.py` + `tools/train_bga_all.py`
- **New**: Web wizard at `/train/new`, 5 steps (select panels → preprocess → review tiles → train 10 PatchCore → done). Backed by `capi_train_new.py` + `capi_preprocess.py`.
- **Backbone offline**: Pre-stage `wide_resnet50_2-32ee1156.pth` to `deployment/torch_hub_cache/hub/checkpoints/`.
```

- [ ] **Step 2：commit**

```bash
git add CLAUDE.md
git commit -m "docs: document new C-10 architecture in CLAUDE.md"
```

### Task 10.3 — End-to-end 手動驗證

**Files:** 無

- [ ] **Step 1：準備測試環境**

```bash
# 確保 backbone 預載
mkdir -p deployment/torch_hub_cache/hub/checkpoints
# (從有網路機器下載 wide_resnet50_2 後 FTP 過來)

# 啟 server
python capi_server.py --config server_config_local.yaml
```

- [ ] **Step 2：跑 wizard 完整流程**

1. 開 `http://localhost:8080/training` → 看到 2 張卡
2. 點「新機種 PatchCore」→ step 1，輸入測試機種
3. 勾 5 片 panel → 下一步
4. step 2 看 preprocess log 跑完
5. step 3 review tile，隨意 reject 幾個
6. 點「開始訓練」→ step 4，看 10 unit 跑（GPU 機才能跑完）
7. step 5 看摘要，點「前往模型庫」
8. `/models` 點「啟用」→ 重啟 server
9. 傳該機種 panel inference → log 顯示 "load 10 models"

- [ ] **Step 3：跑全部測試**

```bash
pytest tests/ -v
```
Expected: all pass

- [ ] **Step 4：建 release commit**

```bash
git log --oneline | head -30  # 確認所有 commits 在
git status  # 應 clean
git tag wizard-v1-20260428  # 標 release
```

---

## Self-Review

完成 plan 寫作後對照 spec 檢查：

### 1. Spec coverage check

| Spec 段落 | 對應 Phase / Task |
|-----------|-------------------|
| Key Decision #1 (C-10 架構) | Phase 4.6, 9.3 |
| Key Decision #2 (MARK 進訓練不 mask) | Phase 1, 9.4 |
| Key Decision #3 (Tile review UI 5 tab) | Phase 6.3 |
| Key Decision #4 (lighting 過濾規則) | Phase 1.2 |
| Key Decision #5 (5 panel 預設) | Phase 6.1 (UI 預設) |
| Key Decision #6 (NG over_review 跨機種) | Phase 4.3 |
| Key Decision #7 (DB panel picker) | Phase 5.1 |
| Key Decision #8 (超參數隱藏) | Phase 4.6 (hard-coded) |
| Key Decision #9 (registry 動作) | Phase 7.2-7.4 |
| Key Decision #10 (3F 不遷移) | Phase 9 (legacy 路徑保留) |
| Key Decision #11 (yaml 在 bundle 內) | Phase 4.5 |
| Key Decision #12 (edge_threshold 768 可調) | Phase 1.4 + Task 4.5 yaml |
| Key Decision #13 (orientation/易斯貼固定) | 無需特別程式碼 |
| Key Decision #14 (offline backbone) | Phase 4.6 _setup_offline_env |
| 共用前處理模組 | Phase 1 |
| 多機種 yaml | Phase 2 |
| DB schema | Phase 3 |
| Wizard backend | Phase 4 + 5 |
| Wizard frontend | Phase 6 |
| Model registry | Phase 7 + 8 |
| Inference 修改 | Phase 9 |
| Cleanup | Phase 10 |

### 2. Placeholder scan

無「TBD」「TODO」「fill in details」。所有 step 都含實際代碼或具體指令。Inline comment 註明 ⚠️ 之處（如 v1/v2 helpers 抽出）為 implementer 必須查看現有 code 再決定的合理彈性，不算 placeholder。

### 3. Type consistency

- `TileResult.zone` ∈ {"inner", "edge", "outside"} 全文一致
- `training_jobs.state` ∈ {"preprocess", "review", "train", "completed", "failed"} 一致
- `training_tile_pool.decision` ∈ {"accept", "reject"} 一致（spec 已修正不使用 pending）
- `model_registry.is_active` ∈ {0, 1} 一致
- `LIGHTINGS` tuple 與 `TRAINING_UNITS` 計算一致

### 4. 已知 caveats

- Phase 9 涉及對 `capi_inference.py` 大量 refactor，task 描述用「⚠️ 閱讀現有 code 後再決定」彈性處理 — 因為現有 code 行為複雜，過細描述反而可能誤導。Implementer 需 grep / read 既有 callers 後完成。
- Phase 5 / 6 假設既有 `capi_web.py` 有 `_send_response` / `_send_json` / `jinja_env` / `_send_file` 這些 helpers — 實際上以 retrain plan 為證已存在。
- Phase 4.6 `_compute_train_max_score` 與 `_predict_ng_scores` 是用 `TorchInferencer.predict()` 跑 inference，性能上 NG 30 張× 10 unit = 300 inference call，每 unit 估 1-2 分。可接受。

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-28-new-model-training-wizard.md` (~50 個 tasks, 10 phases)。**

**兩個執行選項：**

**1. Subagent-Driven（建議）** — 每 task 派一個 fresh subagent + two-stage review，每 phase 結束 checkpoint。
適合本次規模，phase-by-phase 推進、隔離 context。

**2. Inline Execution** — 在當前 session 連續跑，使用 executing-plans skill 批次執行 + checkpoint。
較快但 context 累積快。

**選哪種？**
