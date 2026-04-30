# AOI 座標邊緣 ROI 新架構 edge.pt 路由 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓 C-10 新架構 (`is_new_architecture=True`) 下 AOI 座標邊緣 ROI 推論真的跑起來，且使用該 lighting 的 edge.pt + edge_threshold；同時把 fusion 模式在新架構下停用（強制走 PC = edge.pt）。

**Architecture:** 把 v1 inline 的 AOI 座標 inspection 區塊（~250 行）抽成 `_apply_aoi_coord_inspection(...)` helper，v1 / v2 都呼叫；新增 zone-aware 的 model / threshold lookup helper，新架構走 nested mapping 的 edge slot；新架構強制 inspector mode = `"patchcore"`，UI 灰掉 fusion 選項。

**Tech Stack:** Python 3.11, pytest, anomalib, OpenCV，動到 `capi_inference.py` / `capi_web.py` / `templates/settings.html`。

---

## File Structure

**Modify:**
- `capi_inference.py`
  - 新增 `_get_inferencer_for_zone(prefix, zone)` 與 `_get_threshold_for_zone(prefix, zone)`（zone-aware lookup）
  - 新增 `_resolve_aoi_edge_inspector_mode()`（新架構強制 "patchcore"）
  - `_inspect_roi_patchcore` 加 `zone="edge"` 參數
  - `_inspect_roi_fusion` 加 `zone="edge"` 參數，PC half 走 zone-aware lookup
  - 新增 `_apply_aoi_coord_inspection(panel_dir, preprocessed_results, omit_image, omit_overexposed, product_resolution)` helper（從 v1 inline 區塊抽出）
  - v1 (`_process_panel_v1`) 改用 helper（保留 skip-file 重新 preprocess 的 v1-only 分支）
  - v2 (`_process_panel_v2_per_zone`) 加上 `_apply_aoi_coord_inspection` 呼叫
- `capi_web.py`
  - `/api/settings` response 增加 `is_new_architecture` 欄位
- `templates/settings.html`
  - inspector radio 在 `is_new_architecture=true` 時鎖定為 `"patchcore"`、灰掉 cv / fusion，並顯示提示

**Test:**
- `tests/test_aoi_edge_zone_routing.py`（新）— zone-aware lookup + inspector mode resolver
- `tests/test_aoi_coord_inspection_helper.py`（新）— `_apply_aoi_coord_inspection` 在 v1 / v2 路徑都跑得起來
- `tests/test_aoi_edge_patchcore.py`（修改）— 既有測試帶上 `zone` 參數
- `tests/test_aoi_edge_fusion.py`（修改）— 既有測試帶上 `zone` 參數

---

## Task 1: zone-aware inferencer / threshold lookup

**Files:**
- Modify: `capi_inference.py`（在 `_get_inferencer_for_prefix` / `_get_threshold_for_prefix` 之後新增）
- Test: `tests/test_aoi_edge_zone_routing.py`（新）

- [ ] **Step 1: Write the failing tests**

建立 `tests/test_aoi_edge_zone_routing.py`：

```python
"""新架構 AOI 座標邊緣 zone-aware routing 單元測試。"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer


@pytest.fixture
def legacy_inferencer():
    """舊架構 (5-model)：is_new_architecture=False，flat model_mapping。"""
    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.is_new_architecture = False
    cfg.model_mapping = {"G0F00000": "/fake/legacy.pt"}
    cfg.threshold_mapping = {"G0F00000": 0.55}
    inf = CAPIInferencer.__new__(CAPIInferencer)
    inf.config = cfg
    inf.threshold = 0.5
    inf._model_mapping = {"G0F00000": Path("/fake/legacy.pt")}
    inf._threshold_mapping = {"G0F00000": 0.55}
    inf._inferencers = {}
    inf.inferencer = None
    return inf


@pytest.fixture
def new_arch_inferencer():
    """新架構 (C-10)：nested model_mapping with inner/edge slots。"""
    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.is_new_architecture = True
    cfg.machine_id = "M1"
    cfg.model_mapping = {
        "G0F00000": {"inner": "/fake/G0F-inner.pt", "edge": "/fake/G0F-edge.pt"},
    }
    cfg.threshold_mapping = {
        "G0F00000": {"inner": 0.40, "edge": 0.65},
    }
    inf = CAPIInferencer.__new__(CAPIInferencer)
    inf.config = cfg
    inf.threshold = 0.5
    inf.base_dir = Path(".")
    inf._model_mapping = {}
    inf._threshold_mapping = {}
    inf._model_cache_v2 = {}
    inf._inferencers = {}
    inf.inferencer = None
    return inf


def test_zone_inferencer_legacy_ignores_zone(legacy_inferencer):
    """舊架構：zone 參數應該被忽略，走 prefix-only lookup。"""
    legacy_inferencer._inferencers[str(Path("/fake/legacy.pt"))] = "LEGACY_MODEL"
    assert legacy_inferencer._get_inferencer_for_zone("G0F00000", "edge") == "LEGACY_MODEL"
    assert legacy_inferencer._get_inferencer_for_zone("G0F00000", "inner") == "LEGACY_MODEL"


def test_zone_inferencer_new_arch_routes_to_edge(new_arch_inferencer):
    """新架構：zone='edge' 應路由到 edge.pt。"""
    with patch.object(new_arch_inferencer, "_load_model_from_path", return_value="EDGE_MODEL") as mock_load:
        result = new_arch_inferencer._get_inferencer_for_zone("G0F00000", "edge")
        assert result == "EDGE_MODEL"
        loaded_path = mock_load.call_args[0][0]
        assert loaded_path.name == "G0F-edge.pt"


def test_zone_inferencer_new_arch_routes_to_inner(new_arch_inferencer):
    """新架構：zone='inner' 應路由到 inner.pt。"""
    with patch.object(new_arch_inferencer, "_load_model_from_path", return_value="INNER_MODEL") as mock_load:
        result = new_arch_inferencer._get_inferencer_for_zone("G0F00000", "inner")
        assert result == "INNER_MODEL"
        loaded_path = mock_load.call_args[0][0]
        assert loaded_path.name == "G0F-inner.pt"


def test_zone_threshold_legacy_returns_flat(legacy_inferencer):
    assert legacy_inferencer._get_threshold_for_zone("G0F00000", "edge") == 0.55
    assert legacy_inferencer._get_threshold_for_zone("G0F00000", "inner") == 0.55


def test_zone_threshold_new_arch_picks_zone_value(new_arch_inferencer):
    assert new_arch_inferencer._get_threshold_for_zone("G0F00000", "edge") == 0.65
    assert new_arch_inferencer._get_threshold_for_zone("G0F00000", "inner") == 0.40


def test_zone_threshold_new_arch_unknown_prefix_falls_back(new_arch_inferencer):
    """新架構：prefix 不在 mapping → fallback 到 self.threshold。"""
    assert new_arch_inferencer._get_threshold_for_zone("UNKNOWN", "edge") == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_aoi_edge_zone_routing.py -v`
Expected: FAIL — `AttributeError: 'CAPIInferencer' object has no attribute '_get_inferencer_for_zone'`

- [ ] **Step 3: Implement the helpers**

在 `capi_inference.py` 中、緊接 `_get_threshold_for_prefix` 定義之後（約 line 575）插入：

```python
def _get_inferencer_for_zone(self, prefix: str, zone: str) -> Optional[Any]:
    """新架構：依 (prefix, zone) 走 nested model_mapping；舊架構：fallback 到 prefix。

    新架構 (is_new_architecture=True) 的 model_mapping 是
    ``{prefix: {"inner": path, "edge": path}}``，需要 zone 才能解析。
    舊架構 ``{prefix: path}``，zone 參數忽略。
    """
    if getattr(self.config, "is_new_architecture", False):
        return self._get_model_for(self.config.machine_id, prefix, zone)
    return self._get_inferencer_for_prefix(prefix)

def _get_threshold_for_zone(self, prefix: str, zone: str) -> float:
    """同上，threshold 版本。新架構 threshold_mapping 是 ``{prefix: {inner, edge}}``。"""
    if getattr(self.config, "is_new_architecture", False):
        thr_map = self.config.threshold_mapping.get(prefix)
        if isinstance(thr_map, dict):
            return float(thr_map.get(zone, self.threshold))
        if thr_map is not None:
            return float(thr_map)
        return self.threshold
    return self._get_threshold_for_prefix(prefix)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_aoi_edge_zone_routing.py -v`
Expected: PASS（6 tests）

- [ ] **Step 5: Commit**

```bash
git add capi_inference.py tests/test_aoi_edge_zone_routing.py
git commit -m "feat(infer): zone-aware model/threshold lookup for new arch"
```

---

## Task 2: inspector mode resolver（新架構強制 patchcore）

**Files:**
- Modify: `capi_inference.py`
- Test: `tests/test_aoi_edge_zone_routing.py`（追加）

- [ ] **Step 1: Write the failing tests**

在 `tests/test_aoi_edge_zone_routing.py` 末尾追加：

```python
def test_inspector_mode_legacy_reads_config(legacy_inferencer):
    """舊架構：照舊讀 edge_inspector.config.aoi_edge_inspector。"""
    legacy_inferencer.edge_inspector = MagicMock()
    legacy_inferencer.edge_inspector.config.aoi_edge_inspector = "fusion"
    assert legacy_inferencer._resolve_aoi_edge_inspector_mode() == "fusion"


def test_inspector_mode_new_arch_forces_patchcore(new_arch_inferencer):
    """新架構：無視 config，強制回 'patchcore'（edge.pt 已專為 edge 訓練，
    fusion / cv 不再有理論基礎）。"""
    new_arch_inferencer.edge_inspector = MagicMock()
    new_arch_inferencer.edge_inspector.config.aoi_edge_inspector = "fusion"
    assert new_arch_inferencer._resolve_aoi_edge_inspector_mode() == "patchcore"

    new_arch_inferencer.edge_inspector.config.aoi_edge_inspector = "cv"
    assert new_arch_inferencer._resolve_aoi_edge_inspector_mode() == "patchcore"


def test_inspector_mode_no_edge_inspector_default_cv(legacy_inferencer):
    """舊架構 + edge_inspector 不存在 → 'cv' (與既有 fallback 一致)。"""
    legacy_inferencer.edge_inspector = None
    assert legacy_inferencer._resolve_aoi_edge_inspector_mode() == "cv"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_aoi_edge_zone_routing.py::test_inspector_mode_new_arch_forces_patchcore -v`
Expected: FAIL — `AttributeError: ... '_resolve_aoi_edge_inspector_mode'`

- [ ] **Step 3: Implement the resolver**

在 `capi_inference.py` 中、緊接 `_get_threshold_for_zone` 之後新增：

```python
def _resolve_aoi_edge_inspector_mode(self) -> str:
    """回傳實際使用的 inspector mode。

    新架構 (is_new_architecture=True) 強制 'patchcore'：edge.pt 已專為 edge zone
    訓練，CV+PC 空間分權的 fusion 失去理論基礎；同步把 cv 路徑停用以統一行為。
    舊架構讀 ``edge_inspector.config.aoi_edge_inspector`` (cv / patchcore / fusion)。
    """
    if getattr(self.config, "is_new_architecture", False):
        return "patchcore"
    if not getattr(self, "edge_inspector", None):
        return "cv"
    return getattr(self.edge_inspector.config, "aoi_edge_inspector", "cv")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_aoi_edge_zone_routing.py -v`
Expected: PASS（9 tests）

- [ ] **Step 5: Commit**

```bash
git add capi_inference.py tests/test_aoi_edge_zone_routing.py
git commit -m "feat(infer): force aoi_edge_inspector=patchcore in new arch"
```

---

## Task 3: `_inspect_roi_patchcore` 接受 zone 參數

**Files:**
- Modify: `capi_inference.py:3600-3735`（`_inspect_roi_patchcore` 函式）
- Modify: `tests/test_aoi_edge_patchcore.py`（既有測試保護向後相容）
- Test: `tests/test_aoi_edge_zone_routing.py`（追加 zone 串通測試）

- [ ] **Step 1: Write the failing test**

在 `tests/test_aoi_edge_zone_routing.py` 末尾追加：

```python
import numpy as np


def test_inspect_roi_patchcore_new_arch_uses_edge_model(new_arch_inferencer):
    """_inspect_roi_patchcore(zone='edge') 在新架構下應呼叫 edge.pt 對應的 inferencer。"""
    edge_model = MagicMock()
    edge_model.predict.return_value = MagicMock(pred_score=0.1, anomaly_map=np.zeros((512, 512), dtype=np.float32))

    new_arch_inferencer.edge_inspector = MagicMock()
    new_arch_inferencer.config.patchcore_min_area = 10
    new_arch_inferencer.config.patchcore_filter_enabled = False

    with patch.object(new_arch_inferencer, "_get_model_for", return_value=edge_model) as mock_for:
        with patch.object(new_arch_inferencer, "predict_tile", return_value=(0.1, np.zeros((512, 512), dtype=np.float32))):
            image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            defects, stats = new_arch_inferencer._inspect_roi_patchcore(
                image, img_x=200, img_y=200, img_prefix="G0F00000",
                panel_polygon=None, zone="edge",
            )

    mock_for.assert_called_with("M1", "G0F00000", "edge")
    assert stats["threshold"] == 0.65, "新架構應使用 edge_threshold=0.65"


def test_inspect_roi_patchcore_legacy_default_zone_unchanged(legacy_inferencer):
    """舊架構 + 預設 zone='edge'：行為與既有 prefix-only lookup 完全一致。"""
    legacy_model = MagicMock()
    legacy_inferencer.edge_inspector = MagicMock()
    legacy_inferencer.config.patchcore_min_area = 10
    legacy_inferencer.config.patchcore_filter_enabled = False

    with patch.object(legacy_inferencer, "_get_inferencer_for_prefix", return_value=legacy_model) as mock_p:
        with patch.object(legacy_inferencer, "predict_tile", return_value=(0.1, np.zeros((512, 512), dtype=np.float32))):
            image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            legacy_inferencer._inspect_roi_patchcore(
                image, img_x=200, img_y=200, img_prefix="G0F00000",
                panel_polygon=None,  # zone 預設 "edge"
            )

    mock_p.assert_called_with("G0F00000")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_aoi_edge_zone_routing.py::test_inspect_roi_patchcore_new_arch_uses_edge_model -v`
Expected: FAIL — `_inspect_roi_patchcore` 內部仍呼叫 `_get_inferencer_for_prefix`，新架構下 mock_for 不會被叫到、threshold 為 0.5（fallback）。

- [ ] **Step 3: Modify `_inspect_roi_patchcore` 簽章與內部 lookup**

在 `capi_inference.py:3600`，把：

```python
def _inspect_roi_patchcore(
    self,
    image: np.ndarray,
    img_x: int,
    img_y: int,
    img_prefix: str,
    panel_polygon: Optional[np.ndarray] = None,
    return_raw: bool = False,
) -> Tuple[List[EdgeDefect], Dict[str, Any]]:
```

改為：

```python
def _inspect_roi_patchcore(
    self,
    image: np.ndarray,
    img_x: int,
    img_y: int,
    img_prefix: str,
    panel_polygon: Optional[np.ndarray] = None,
    return_raw: bool = False,
    zone: str = "edge",
) -> Tuple[List[EdgeDefect], Dict[str, Any]]:
```

並在 docstring 的 ``Args:`` 區塊內加一行：

```
            zone: "inner" | "edge"，新架構 (C-10) 用來路由到對應的 inner.pt / edge.pt。
                AOI 座標邊緣路徑預設 "edge"。舊架構忽略此值。
```

接著在同一函式中、line 3674 附近：

```python
        # 取 inferencer / threshold (沿用中央 tile pipeline 的 prefix 路由)
        inferencer = self._get_inferencer_for_prefix(img_prefix)
        threshold = self._get_threshold_for_prefix(img_prefix)
```

替換為：

```python
        # 取 inferencer / threshold：新架構走 zone-aware (走 edge.pt)，舊架構走 prefix-only
        inferencer = self._get_inferencer_for_zone(img_prefix, zone)
        threshold = self._get_threshold_for_zone(img_prefix, zone)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_aoi_edge_zone_routing.py tests/test_aoi_edge_patchcore.py -v`
Expected: PASS（既有 test_aoi_edge_patchcore.py 的測試也應全綠 — 預設 zone="edge" 不破壞向後相容）

- [ ] **Step 5: Commit**

```bash
git add capi_inference.py tests/test_aoi_edge_zone_routing.py
git commit -m "feat(infer): _inspect_roi_patchcore accepts zone parameter"
```

---

## Task 4: `_inspect_roi_fusion` 加 zone 參數（PC half 走 zone-aware）

**Files:**
- Modify: `capi_inference.py:3024-3425`（`_inspect_roi_fusion` 函式）
- Test: `tests/test_aoi_edge_fusion.py`（既有測試保護向後相容）

- [ ] **Step 1: 確認 zone 在 fusion 內如何流通**

`_inspect_roi_fusion` 自己不直接 call `_get_inferencer_for_prefix`；它在 line 3319 呼叫 `_inspect_roi_patchcore(return_raw=True)`，由後者做 model lookup。所以 fusion 只需把 `zone` 透傳。

Read: `capi_inference.py:3024-3030, 3315-3325` 確認上述呼叫點。

- [ ] **Step 2: Write the failing test**

在 `tests/test_aoi_edge_zone_routing.py` 末尾追加：

```python
def test_inspect_roi_fusion_new_arch_pc_half_uses_edge_model(new_arch_inferencer):
    """新架構下 fusion (理論上不會走到，但行為要正確) PC half 應走 edge.pt。"""
    edge_model = MagicMock()
    new_arch_inferencer.edge_inspector = MagicMock()
    new_arch_inferencer.edge_inspector.config.aoi_edge_boundary_band_px = 40
    new_arch_inferencer.edge_inspector.config.aoi_edge_pc_roi_inward_shift_enabled = False
    new_arch_inferencer.edge_inspector.inspect_roi.return_value = ([], {})

    with patch.object(new_arch_inferencer, "_inspect_roi_patchcore") as mock_pc:
        mock_pc.return_value = ([], {"score": 0.1, "threshold": 0.65,
                                      "anomaly_map": np.zeros((512, 512), np.float32),
                                      "fg_mask": np.zeros((512, 512), np.uint8),
                                      "roi": np.zeros((512, 512, 3), np.uint8),
                                      "area": 0, "min_area": 10})
        new_arch_inferencer._inspect_roi_fusion(
            image=np.zeros((1080, 1920, 3), np.uint8),
            img_x=200, img_y=200, img_prefix="G0F00000",
            panel_polygon=np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], dtype=np.float32),
            zone="edge",
        )

    # 確認 fusion 內部呼叫 _inspect_roi_patchcore 時帶上 zone="edge"
    pc_kwargs = mock_pc.call_args.kwargs
    assert pc_kwargs.get("zone") == "edge", \
        f"fusion 內部 _inspect_roi_patchcore 未帶 zone='edge'，實際 kwargs={pc_kwargs}"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_aoi_edge_zone_routing.py::test_inspect_roi_fusion_new_arch_pc_half_uses_edge_model -v`
Expected: FAIL — `_inspect_roi_fusion` 內部 PC 呼叫沒有 `zone` kwarg。

- [ ] **Step 4: Add zone param to `_inspect_roi_fusion`**

在 `capi_inference.py:3024`，把：

```python
def _inspect_roi_fusion(
    self,
    image: np.ndarray,
    img_x: int,
    img_y: int,
    img_prefix: str,
    panel_polygon: Optional[np.ndarray] = None,
    omit_image: Optional[np.ndarray] = None,
    omit_overexposed: bool = False,
    otsu_bounds: Optional[Tuple[int, int, int, int]] = None,
    collapse_to_representative: bool = True,
    group_cv_band: bool = False,
) -> Tuple[List[EdgeDefect], Dict[str, Any]]:
```

改為：

```python
def _inspect_roi_fusion(
    self,
    image: np.ndarray,
    img_x: int,
    img_y: int,
    img_prefix: str,
    panel_polygon: Optional[np.ndarray] = None,
    omit_image: Optional[np.ndarray] = None,
    omit_overexposed: bool = False,
    otsu_bounds: Optional[Tuple[int, int, int, int]] = None,
    collapse_to_representative: bool = True,
    group_cv_band: bool = False,
    zone: str = "edge",
) -> Tuple[List[EdgeDefect], Dict[str, Any]]:
```

接著在 line 3319 附近找出所有呼叫 `_inspect_roi_patchcore(...)` 的位置，全部加上 `zone=zone` kwarg。先 grep 出位置：

Run: `grep -n '_inspect_roi_patchcore(' capi_inference.py`

對 `_inspect_roi_fusion` 內部（範圍 3024-3425）的每一處 `self._inspect_roi_patchcore(...)` 呼叫加 `zone=zone`。範例（line 3319 附近）：

```python
        _, pc_stats = self._inspect_roi_patchcore(
            image, img_x, img_y, img_prefix,
            panel_polygon=panel_polygon, return_raw=True,
        )
```

改為：

```python
        _, pc_stats = self._inspect_roi_patchcore(
            image, img_x, img_y, img_prefix,
            panel_polygon=panel_polygon, return_raw=True,
            zone=zone,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_aoi_edge_zone_routing.py tests/test_aoi_edge_fusion.py tests/test_aoi_edge_pc_roi_shift.py -v`
Expected: PASS — 既有 fusion 測試行為不變（zone 預設 "edge"），新測試通過。

- [ ] **Step 6: Commit**

```bash
git add capi_inference.py tests/test_aoi_edge_zone_routing.py
git commit -m "feat(infer): _inspect_roi_fusion threads zone through PC half"
```

---

## Task 5: 抽 `_apply_aoi_coord_inspection` helper

**Files:**
- Modify: `capi_inference.py:4197-4451`（v1 inline AOI coord 區塊）
- Test: `tests/test_aoi_coord_inspection_helper.py`（新）

> **重要**：本 task 是「行為保留 refactor」。先抽出 helper、再讓 v1 改用 helper，最後跑既有 test suite 驗證 v1 行為不變。**Skip-file 重新 preprocess 的分支保留在 v1 inline**（v2 不需要這分支）。

- [ ] **Step 1: Write the failing test**

建立 `tests/test_aoi_coord_inspection_helper.py`：

```python
"""_apply_aoi_coord_inspection helper 在 v1 / v2 都能直接呼叫的測試。"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer, ImageResult


@pytest.fixture
def new_arch_inferencer(tmp_path):
    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.is_new_architecture = True
    cfg.machine_id = "M1"
    cfg.aoi_coord_inspection_enabled = True
    cfg.aoi_report_path_replace_from = ""
    cfg.aoi_report_path_replace_to = ""
    cfg.model_mapping = {
        "G0F00000": {"inner": "/fake/inner.pt", "edge": "/fake/edge.pt"},
    }
    cfg.threshold_mapping = {"G0F00000": {"inner": 0.4, "edge": 0.65}}
    inf = CAPIInferencer.__new__(CAPIInferencer)
    inf.config = cfg
    inf.threshold = 0.5
    inf.base_dir = tmp_path
    inf.edge_inspector = MagicMock()
    inf.edge_inspector.config.aoi_edge_inspector = "fusion"  # 應被 new_arch override
    inf._model_mapping = {}
    inf._threshold_mapping = {}
    inf._model_cache_v2 = {}
    inf._inferencers = {}
    inf.inferencer = None
    return inf


def test_helper_returns_zero_when_disabled(new_arch_inferencer, tmp_path):
    new_arch_inferencer.config.aoi_coord_inspection_enabled = False
    stats = new_arch_inferencer._apply_aoi_coord_inspection(
        panel_dir=tmp_path,
        preprocessed_results=[],
        omit_image=None, omit_overexposed=False,
        product_resolution=(1920, 1080),
    )
    assert stats == {"aoi_tile_count": 0, "aoi_edge_count": 0}


def test_helper_returns_zero_when_no_aoi_report(new_arch_inferencer, tmp_path):
    """panel_dir 內沒有 aoi report → helper 回傳 0/0，不丟例外。"""
    stats = new_arch_inferencer._apply_aoi_coord_inspection(
        panel_dir=tmp_path,
        preprocessed_results=[],
        omit_image=None, omit_overexposed=False,
        product_resolution=(1920, 1080),
    )
    assert stats == {"aoi_tile_count": 0, "aoi_edge_count": 0}


def test_helper_calls_patchcore_with_edge_zone(new_arch_inferencer, tmp_path):
    """新架構下 helper 對 AOI 邊緣 defect 呼叫 _inspect_roi_patchcore(zone='edge')。"""
    img_path = tmp_path / "G0F00000_001.png"
    np.ones((1080, 1920, 3), dtype=np.uint8).tofile(img_path)  # placeholder file

    fake_result = ImageResult(
        image_path=img_path,
        image_size=(1920, 1080),
        otsu_bounds=(0, 0, 1920, 1080),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        anomaly_tiles=[],
        raw_bounds=(0, 0, 1920, 1080),
        panel_polygon=np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], dtype=np.float32),
    )

    fake_edge_def = MagicMock(product_x=1900, product_y=540, defect_code="L01")

    with patch.object(new_arch_inferencer, "_parse_aoi_report_txt",
                      return_value={"G0F00000": [fake_edge_def]}), \
         patch.object(new_arch_inferencer, "_create_aoi_coord_tiles",
                      return_value=([], [fake_edge_def])), \
         patch("cv2.imread", return_value=np.ones((1080, 1920, 3), dtype=np.uint8)), \
         patch.object(new_arch_inferencer, "_inspect_roi_patchcore",
                      return_value=([], {"score": 0.1, "threshold": 0.65, "area": 0,
                                          "ok_reason": "", "roi": None, "fg_mask": None,
                                          "anomaly_map": None})) as mock_pc:

        stats = new_arch_inferencer._apply_aoi_coord_inspection(
            panel_dir=tmp_path,
            preprocessed_results=[fake_result],
            omit_image=None, omit_overexposed=False,
            product_resolution=(1920, 1080),
        )

    assert stats["aoi_edge_count"] == 1
    pc_kwargs = mock_pc.call_args.kwargs
    assert pc_kwargs.get("zone") == "edge", \
        f"new arch 下 helper 應呼叫 _inspect_roi_patchcore(zone='edge')；實際 kwargs={pc_kwargs}"


def test_helper_skips_lighting_without_aoi_report(new_arch_inferencer, tmp_path):
    img_path = tmp_path / "R0F00000_001.png"
    img_path.touch()

    other_result = ImageResult(
        image_path=img_path,
        image_size=(1920, 1080),
        otsu_bounds=(0, 0, 1920, 1080),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        anomaly_tiles=[],
        raw_bounds=(0, 0, 1920, 1080),
        panel_polygon=None,
    )

    with patch.object(new_arch_inferencer, "_parse_aoi_report_txt",
                      return_value={"G0F00000": [MagicMock()]}), \
         patch.object(new_arch_inferencer, "_create_aoi_coord_tiles") as mock_create:
        stats = new_arch_inferencer._apply_aoi_coord_inspection(
            panel_dir=tmp_path,
            preprocessed_results=[other_result],
            omit_image=None, omit_overexposed=False,
            product_resolution=(1920, 1080),
        )

    mock_create.assert_not_called()
    assert stats == {"aoi_tile_count": 0, "aoi_edge_count": 0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_aoi_coord_inspection_helper.py -v`
Expected: FAIL — `AttributeError: ... '_apply_aoi_coord_inspection'`

- [ ] **Step 3: Read existing v1 inline block to understand exact behavior**

Read: `capi_inference.py:4197-4451`. Inline 區塊負責四件事：
1. `aoi_report = self._parse_aoi_report_txt(panel_dir)`（4203）
2. Skip-file 重新 preprocess 分支（4214-4238）— **v1-only**
3. 對每個 `result` 跑 `_create_aoi_coord_tiles` + `inspector_mode` 分派（4240-4450）— **共用部分**
4. 結尾 print stats

Helper 只負責 (1)、(3)、(4)。Skip-file 分支留在 v1 inline。

- [ ] **Step 4: Add `_apply_aoi_coord_inspection` method**

在 `capi_inference.py` 中、`_inspect_roi_fusion` 之後（約 line 3450 之後、`_inspect_roi_patchcore` 之前合適的位置）新增：

```python
def _apply_aoi_coord_inspection(
    self,
    panel_dir: Path,
    preprocessed_results: List[Any],  # ImageResult
    omit_image: Optional[np.ndarray],
    omit_overexposed: bool,
    product_resolution: Optional[Tuple[int, int]],
) -> Dict[str, int]:
    """執行 AOI 機檢座標切塊 + 邊緣 ROI inspection；mutates ``preprocessed_results`` in place。

    v1 / v2 共用入口。Inspector mode 由 ``_resolve_aoi_edge_inspector_mode``
    決定（新架構強制 'patchcore'）。新架構下 PC ROI 走 zone='edge' → edge.pt。

    **Note**：v1 在呼叫此 helper 之前還會做「skip-file 重新 preprocess」的分支
    （把 should_skip_file 的圖片補進 preprocessed_results 以便對它跑 AOI coord
    inspection），那是 v1 專屬流程，不在此 helper 內。

    Args:
        panel_dir: 面板目錄（用來找 aoi_report.txt）
        preprocessed_results: 已預處理的 ImageResult list
        omit_image, omit_overexposed: OMIT 灰塵屏蔽參數
        product_resolution: 產品解析度 (w, h)，用於 AOI 座標映射

    Returns:
        ``{"aoi_tile_count": int, "aoi_edge_count": int}``
    """
    if not self.config.aoi_coord_inspection_enabled:
        return {"aoi_tile_count": 0, "aoi_edge_count": 0}

    aoi_report = self._parse_aoi_report_txt(panel_dir)
    if not aoi_report:
        return {"aoi_tile_count": 0, "aoi_edge_count": 0}

    inspector_mode = self._resolve_aoi_edge_inspector_mode()
    aoi_tile_count = 0
    aoi_edge_count = 0

    for result in preprocessed_results:
        img_prefix = self._get_image_prefix(result.image_path.name)
        if img_prefix not in aoi_report:
            continue

        aoi_image = cv2.imread(str(result.image_path), cv2.IMREAD_UNCHANGED)
        if aoi_image is None:
            continue

        new_tiles, edge_defs = self._create_aoi_coord_tiles(
            aoi_image, result, aoi_report[img_prefix], product_resolution,
        )
        result.tiles.extend(new_tiles)
        aoi_tile_count += len(new_tiles)
        aoi_edge_count += len(edge_defs)

        for edef in edge_defs:
            self._inspect_aoi_edge_defect(
                edef=edef,
                aoi_image=aoi_image,
                result=result,
                product_resolution=product_resolution,
                inspector_mode=inspector_mode,
                img_prefix=img_prefix,
                omit_image=omit_image,
                omit_overexposed=omit_overexposed,
            )

    return {"aoi_tile_count": aoi_tile_count, "aoi_edge_count": aoi_edge_count}
```

- [ ] **Step 5: 抽出 `_inspect_aoi_edge_defect` 方法**

把 v1 inline 區塊中、line 4258-4450 那段「for edef in edge_defs:」整個迴圈體抽成新 method `_inspect_aoi_edge_defect`，放在 `_apply_aoi_coord_inspection` 之後。簽章：

```python
def _inspect_aoi_edge_defect(
    self,
    edef,
    aoi_image: np.ndarray,
    result,  # ImageResult
    product_resolution: Optional[Tuple[int, int]],
    inspector_mode: str,
    img_prefix: str,
    omit_image: Optional[np.ndarray],
    omit_overexposed: bool,
) -> None:
    """對單一 AOI 座標 edge defect 跑 fusion / patchcore / cv 任一 inspector。

    把 result.edge_defects mutate（append NG 或 OK record）。新架構走 zone='edge'。
    """
    img_x, img_y = self._map_aoi_coords(
        edef.product_x, edef.product_y,
        result.raw_bounds, product_resolution,
    )
    roi_size = self.config.tile_size
    roi_half = roi_size // 2
    img_h, img_w = aoi_image.shape[:2]
    rx1 = max(0, img_x - roi_half)
    ry1 = max(0, img_y - roi_half)
    rx2 = min(img_w, img_x + roi_half)
    ry2 = min(img_h, img_y + roi_half)

    detected = False

    # === fusion 路徑 ===
    if inspector_mode == "fusion":
        # ...原 v1 line 4274-4350 的內容 verbatim 複製進來，把所有
        # `_inspect_roi_fusion(...)` 加上 ``zone="edge"``，
        # 把 `_inspect_roi_patchcore(...)` 加上 ``zone="edge"`` （已在 Task 3/4 完成）
        # 結尾保留 `return` 而非 `continue`
        ...
        return

    # === patchcore 路徑 ===
    if inspector_mode == "patchcore":
        # ...原 v1 line 4353-4392 的內容 verbatim，把 `_inspect_roi_patchcore(...)` 加上 ``zone="edge"``
        ...
        return

    # === CV 路徑 (default) ===
    # ...原 v1 line 4395-4450 的內容 verbatim
```

> 實作時：把原 v1 inline `for edef in edge_defs:` 整段內容（含所有 print、所有 if/else、所有 EdgeDefect 構造）一字不漏複製進這個 method；**唯一改動**是把 inline 中的 `continue` 改成 `return`、indent 重排，以及讓 `_inspect_roi_patchcore` / `_inspect_roi_fusion` 帶上 `zone="edge"`。不要重寫邏輯。

- [ ] **Step 6: 把 v1 inline 區塊改用 helper**

把 `capi_inference.py:4197-4451` 替換為以下骨架（保留 skip-file 分支與外層 ``aoi_report`` parse — 因為印 log / skip-file rebuild 必須在 helper 之前；但實際 inspection 改 delegate）：

```python
        # ================================================================
        # Phase 1.5: AOI 機檢座標目標切塊
        # 解析 AOI 機台 NG 報告，以缺陷座標為中心建立額外的 512x512 tiles
        # ================================================================
        aoi_report = {}
        if self.config.aoi_coord_inspection_enabled:
            aoi_report = self._parse_aoi_report_txt(panel_dir)
            if aoi_report:
                # 收集已有的圖片前綴
                existing_prefixes = set()
                for result in preprocessed_results:
                    existing_prefixes.add(self._get_image_prefix(result.image_path.name))

                # v1-only: 對 skip_files 中有 AOI 報告的圖片，預處理後加入
                for report_prefix in aoi_report:
                    if report_prefix not in existing_prefixes:
                        matched_file = None
                        skipped_files = [f for f in image_files
                                         if self.config.should_skip_file(f.name)
                                         and not is_dust_check_image(f)]
                        for f in skipped_files:
                            if self._get_image_prefix(f.name) == report_prefix:
                                matched_file = f
                                break
                        if matched_file is not None:
                            print(f"🎯 AOI Coord: 為跳過的圖片 {matched_file.name} 建立預處理 "
                                  f"(有 {len(aoi_report[report_prefix])} 筆 AOI 座標)")
                            skip_result = self.preprocess_image(
                                matched_file,
                                cached_mark=cached_mark,
                                reference_raw_bounds=panel_reference_raw_bounds,
                                reference_polygon=panel_reference_polygon,
                            )
                            if skip_result is not None:
                                skip_result.tiles = []
                                skip_result.excluded_tile_count = 0
                                skip_result.processed_tile_count = 0
                                preprocessed_results.append(skip_result)
                                existing_prefixes.add(report_prefix)

                stats = self._apply_aoi_coord_inspection(
                    panel_dir=panel_dir,
                    preprocessed_results=preprocessed_results,
                    omit_image=omit_image,
                    omit_overexposed=omit_overexposed,
                    product_resolution=product_resolution,
                )
                print(f"🎯 Phase 1.5 完成: AOI 座標新增 {stats['aoi_tile_count']} 個 tiles, "
                      f"{stats['aoi_edge_count']} 個邊緣 defects")
```

- [ ] **Step 7: Run all related tests**

Run: `pytest tests/test_aoi_coord_inspection_helper.py tests/test_aoi_edge_patchcore.py tests/test_aoi_edge_fusion.py tests/test_aoi_edge_pc_roi_shift.py tests/test_aoi_coord_inference.py -v`
Expected: 全部 PASS。新 helper 測試通過；既有 v1 整合測試（`test_aoi_coord_inference.py`）行為不變。

> 若 `test_aoi_coord_inference.py` 失敗，先 read 失敗原因 — 通常是 helper 漏 mutate 某個欄位。**先 diff 比對抽出前後的迴圈體**，不要憑空調整 helper。

- [ ] **Step 8: Commit**

```bash
git add capi_inference.py tests/test_aoi_coord_inspection_helper.py
git commit -m "refactor(infer): extract AOI coord inspection helper"
```

---

## Task 6: v2 (`_process_panel_v2_per_zone`) 呼叫 helper

**Files:**
- Modify: `capi_inference.py:5320-5498`（v2 路徑）
- Test: `tests/test_aoi_coord_inspection_helper.py`（追加 v2 整合測試）

- [ ] **Step 1: Write the failing test**

在 `tests/test_aoi_coord_inspection_helper.py` 末尾追加：

```python
def test_v2_process_panel_invokes_aoi_coord_helper(new_arch_inferencer, tmp_path):
    """新架構 _process_panel_v2_per_zone 應呼叫 _apply_aoi_coord_inspection."""
    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    # 建立最小可運行的 panel：一張 G0F00000 圖
    img_file = panel_dir / "G0F00000_001.png"
    import cv2
    cv2.imwrite(str(img_file), np.ones((1080, 1920, 3), dtype=np.uint8) * 128)

    new_arch_inferencer.config.tile_size = 512
    new_arch_inferencer.config.otsu_offset = 5
    new_arch_inferencer.config.enable_panel_polygon = False
    new_arch_inferencer.config.edge_threshold_px = 768

    with patch("capi_preprocess.preprocess_panel_folder", return_value={}), \
         patch.object(new_arch_inferencer, "_load_omit_context",
                      return_value=(None, False, "", None)), \
         patch.object(new_arch_inferencer, "_apply_aoi_coord_inspection",
                      return_value={"aoi_tile_count": 0, "aoi_edge_count": 0}) as mock_helper, \
         patch.object(new_arch_inferencer, "_apply_omit_dust_postprocess"), \
         patch.object(new_arch_inferencer, "_apply_bomb_postprocess"), \
         patch.object(new_arch_inferencer, "_apply_exclude_zone_postprocess"), \
         patch.object(new_arch_inferencer, "_apply_scratch_postprocess"):

        new_arch_inferencer._process_panel_v2_per_zone(
            panel_dir=panel_dir,
            product_resolution=(1920, 1080),
        )

    assert mock_helper.call_count == 1
    kwargs = mock_helper.call_args.kwargs
    assert kwargs["panel_dir"] == panel_dir
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aoi_coord_inspection_helper.py::test_v2_process_panel_invokes_aoi_coord_helper -v`
Expected: FAIL — `_apply_aoi_coord_inspection` 沒被 v2 呼叫。

- [ ] **Step 3: Read v2 to find insertion point**

Read: `capi_inference.py:5466-5500`（v2 結尾，現有 `_apply_cv_edge_inspection` / postprocess 呼叫鏈）。

- [ ] **Step 4: 在 v2 加上 helper 呼叫**

在 `capi_inference.py` v2 路徑中、緊接 `_apply_cv_edge_inspection(image_result, model_id=model_id)` 整段迴圈結束後（約 line 5471 之後、`post_start = time.time()` 之前）插入：

```python
        # AOI 機檢座標 inspection（v1 / v2 共用 helper；新架構 PC half 走 edge.pt）
        aoi_stats = self._apply_aoi_coord_inspection(
            panel_dir=Path(panel_dir),
            preprocessed_results=results,
            omit_image=omit_image,
            omit_overexposed=omit_overexposed,
            product_resolution=product_resolution,
        )
        if aoi_stats["aoi_tile_count"] or aoi_stats["aoi_edge_count"]:
            print(f"[v2] AOI 座標 inspection: {aoi_stats['aoi_tile_count']} tiles, "
                  f"{aoi_stats['aoi_edge_count']} edge defects")
```

> v2 不需要 v1 的 skip-file 重新 preprocess 分支：v2 已透過 `preprocess_panel_folder` 把 LIGHTINGS 全部處理完，不存在 should_skip_file 概念。

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_aoi_coord_inspection_helper.py -v`
Expected: PASS（5 tests）

- [ ] **Step 6: Run full test suite to catch regressions**

Run: `pytest tests/test_aoi_edge_patchcore.py tests/test_aoi_edge_fusion.py tests/test_aoi_edge_pc_roi_shift.py tests/test_aoi_coord_inference.py tests/test_aoi_edge_zone_routing.py tests/test_aoi_coord_inspection_helper.py -v`
Expected: 全部 PASS。

- [ ] **Step 7: Commit**

```bash
git add capi_inference.py tests/test_aoi_coord_inspection_helper.py
git commit -m "feat(infer): wire v2 to AOI coord inspection helper"
```

---

## Task 7: UI gate — 新架構鎖定 `aoi_edge_inspector="patchcore"`

**Files:**
- Modify: `capi_web.py:3561-3565`（`/api/settings` response 加 `is_new_architecture`）
- Modify: `templates/settings.html:1358-1395`（fusion / cv 選項在新架構下灰掉）

- [ ] **Step 1: Read current `/api/settings` response shape**

Read: `capi_web.py:3540-3570`。確認 `_send_json` 內容是 `{"params": ..., "model_resolution_map": ...}`。

- [ ] **Step 2: 在 `/api/settings` response 加 `is_new_architecture`**

Edit `capi_web.py:3561-3565`，把：

```python
            # 附帶 model_resolution_map 給前端產品選擇器使用
            resolution_map = {}
            if self.inferencer and hasattr(self.inferencer, 'config') and self.inferencer.config:
                resolution_map = getattr(self.inferencer.config, 'model_resolution_map', {})
            self._send_json({"params": params, "model_resolution_map": resolution_map})
```

改為：

```python
            # 附帶 model_resolution_map 給前端產品選擇器使用
            resolution_map = {}
            is_new_arch = False
            if self.inferencer and hasattr(self.inferencer, 'config') and self.inferencer.config:
                resolution_map = getattr(self.inferencer.config, 'model_resolution_map', {})
                is_new_arch = bool(getattr(self.inferencer.config, 'is_new_architecture', False))
            self._send_json({
                "params": params,
                "model_resolution_map": resolution_map,
                "is_new_architecture": is_new_arch,
            })
```

- [ ] **Step 3: 在前端用 `is_new_architecture` flag 鎖 inspector**

Read: `templates/settings.html:1305-1320`（loadSettings 接 response 處）。

Edit `templates/settings.html`，把 `loadSettings` 中：

```javascript
            allParams = data.params || [];
            modelResolutionMap = data.model_resolution_map || {};
            renderSettings();
```

改為：

```javascript
            allParams = data.params || [];
            modelResolutionMap = data.model_resolution_map || {};
            window.isNewArchitecture = !!data.is_new_architecture;
            renderSettings();
```

接著在 `renderSettings()` 計算 `paramLockMap` 的區段（約 line 1357 之後、`if (!aoiCoordEnabled) {` 之前）插入：

```javascript
        // 新架構：edge.pt 已專為 edge zone 訓練，CV+PC fusion 失去理論基礎；
        // 強制 inspector_mode = 'patchcore'，UI 把其他選項灰掉。
        if (window.isNewArchitecture) {
            paramLockMap['aoi_edge_inspector'] =
                '新架構 (C-10) 強制 patchcore：edge.pt 已專為邊緣訓練，' +
                'CV / fusion 在新架構下不再使用';
        }
```

並在 `inspectorVal === 'patchcore'` 那個分支（約 line 1387）的 lock reason 內保留現行行為。

- [ ] **Step 4: 手動驗證（無自動測試）**

由於這是 UI 行為變更，手動測：

1. 啟動 `python capi_server.py --config server_config_local.yaml`（用新架構 config）
2. 開 `http://localhost:8080/settings`
3. 切到 AOI 座標檢測 tab
4. 確認 `aoi_edge_inspector` radio 被鎖在 `patchcore`，hover 顯示「新架構 (C-10) 強制 patchcore...」提示
5. 換到舊架構 config 重啟，確認 radio 正常可切換

- [ ] **Step 5: Commit**

```bash
git add capi_web.py templates/settings.html
git commit -m "feat(ui): lock aoi_edge_inspector to patchcore in new arch"
```

---

## Task 8: end-to-end smoke test 與 docs note

**Files:**
- Modify: `docs/edge-cv-tuning-log.md`（在最末加一段註記新架構行為）
- Test: `tests/test_aoi_coord_inspection_helper.py`（追加 e2e patchcore zone 串通測試）

- [ ] **Step 1: 加上 e2e 測試確保新架構整條路徑都使用 edge.pt**

在 `tests/test_aoi_coord_inspection_helper.py` 末尾追加：

```python
def test_v2_e2e_aoi_edge_routes_through_edge_pt(new_arch_inferencer, tmp_path):
    """E2E：新架構下，AOI 座標邊緣 defect → _apply_aoi_coord_inspection
    → _inspect_aoi_edge_defect → _inspect_roi_patchcore(zone='edge')
    → _get_inferencer_for_zone → _get_model_for(_, _, 'edge')。"""
    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()

    fake_result = ImageResult(
        image_path=panel_dir / "G0F00000_001.png",
        image_size=(1920, 1080),
        otsu_bounds=(0, 0, 1920, 1080),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        anomaly_tiles=[],
        raw_bounds=(0, 0, 1920, 1080),
        panel_polygon=np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], dtype=np.float32),
    )

    fake_edge_def = MagicMock(product_x=1900, product_y=540, defect_code="L01")
    edge_model = MagicMock()

    with patch.object(new_arch_inferencer, "_parse_aoi_report_txt",
                      return_value={"G0F00000": [fake_edge_def]}), \
         patch.object(new_arch_inferencer, "_create_aoi_coord_tiles",
                      return_value=([], [fake_edge_def])), \
         patch("cv2.imread", return_value=np.ones((1080, 1920, 3), dtype=np.uint8)), \
         patch.object(new_arch_inferencer, "_get_model_for", return_value=edge_model) as mock_for, \
         patch.object(new_arch_inferencer, "predict_tile",
                      return_value=(0.1, np.zeros((512, 512), dtype=np.float32))):

        new_arch_inferencer._apply_aoi_coord_inspection(
            panel_dir=panel_dir,
            preprocessed_results=[fake_result],
            omit_image=None, omit_overexposed=False,
            product_resolution=(1920, 1080),
        )

    # 串通驗證：_get_model_for 被叫，且 zone='edge'
    assert mock_for.called, "_get_model_for 應被呼叫（新架構走 edge.pt 路徑）"
    args = mock_for.call_args.args
    assert args[0] == "M1"
    assert args[1] == "G0F00000"
    assert args[2] == "edge", f"zone 應為 'edge'，實際 {args[2]!r}"
```

- [ ] **Step 2: Run e2e test**

Run: `pytest tests/test_aoi_coord_inspection_helper.py::test_v2_e2e_aoi_edge_routes_through_edge_pt -v`
Expected: PASS。

- [ ] **Step 3: Update docs/edge-cv-tuning-log.md**

Read: `docs/edge-cv-tuning-log.md`（看現有結構，找最末「Phase X」段落）。

Append 一個新段（保留現行 phase 編號慣例）：

```markdown
## 新架構 (C-10) AOI 座標邊緣 routing — 2026-04-30

新架構 (`is_new_architecture=True`, 5 lighting × 2 zone = 10 模型) 上線後，AOI
座標邊緣 ROI 推論強制走該 lighting 的 `edge.pt` (zone='edge') 與
`edge_threshold`。實作要點：

- `_apply_aoi_coord_inspection` helper 統一 v1 / v2 入口（`_process_panel_v2_per_zone`
  原本完全沒呼叫 AOI 座標 inspection，本 patch 補上）。
- `_resolve_aoi_edge_inspector_mode()` 在新架構下回 `"patchcore"`，
  忽略 `aoi_edge_inspector` config 值。Settings UI 同步把該選項灰掉。
- `_inspect_roi_patchcore` / `_inspect_roi_fusion` 加 `zone="edge"` 預設參數，
  AOI 座標路徑顯式傳遞。

Fusion 模式不再進入新架構執行路徑（理論基礎是「PC 對邊緣訓練不足」，
edge.pt 已專為邊緣訓練）；保留實作以供舊架構繼續使用。
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_aoi_coord_inspection_helper.py docs/edge-cv-tuning-log.md
git commit -m "test(infer): e2e edge.pt routing + docs note"
```

---

## Self-Review Checklist

執行前自查（已內化到計畫，動手前再核對一次）：

1. **Spec 覆蓋**
   - ✅ Zone-aware lookup（Task 1）
   - ✅ 新架構強制 patchcore（Task 2 + 7）
   - ✅ ROI inspector 走 edge.pt（Task 3 + 4）
   - ✅ V1 / V2 共用 helper（Task 5 + 6）
   - ✅ UI gate（Task 7）
   - ✅ E2E 串通驗證（Task 8）

2. **Placeholder 掃描**：無 TBD / TODO / "similar to" — 每段有實際 code。Task 5 Step 5 的 ellipsis (`...原 v1 line ... 的內容 verbatim 複製`) 是「行為保留 refactor」的標準作法，明確指出複製來源行範圍與唯一允許的差異點。

3. **型別一致性**
   - `_get_inferencer_for_zone(prefix: str, zone: str)` — 一致使用
   - `_get_threshold_for_zone(prefix: str, zone: str) -> float` — 一致
   - `_resolve_aoi_edge_inspector_mode() -> str` — 回 `"cv" | "patchcore" | "fusion"`
   - `_apply_aoi_coord_inspection(panel_dir, preprocessed_results, omit_image, omit_overexposed, product_resolution) -> Dict[str, int]`
   - `_inspect_aoi_edge_defect(...)` — return None，mutate `result.edge_defects`
   - `zone="edge"` — 跨 `_inspect_roi_patchcore` / `_inspect_roi_fusion` 一致預設值

4. **Risk gates**
   - Task 5 是大塊 refactor，若 `test_aoi_coord_inference.py` 紅燈先 diff 比對複製出的迴圈體，不要憑空調整 helper（Step 7 已 inline 此提醒）。
   - Task 7 UI 改動沒有自動測試，必須手動驗證新／舊架構切換行為。
