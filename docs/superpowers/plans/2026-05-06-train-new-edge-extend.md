# Train New Wizard — 固定 3 片 + Edge 外推取樣 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 訓練 wizard panel 數從 5 改為固定 3，全部 inner+edge；`_generate_tiles` 加入「中心位於 panel 邊緣」的外推 tile 取樣以涵蓋 v2 推論 AOI-centered tile 情境。

**Architecture:** `PreprocessConfig.outer_edge_extend`（預設 256）控制每邊往外推距離，`_generate_tiles` 在原 bbox grid 之外加 4 邊 + 4 角的外推 tile，push 距離夾到 image 邊界以保留自然黑邊。`TrainingConfig.inner_panels` 完全移除（統一流程不留條件分支）。

**Tech Stack:** Python 3 + OpenCV + dataclass，pytest（含現有 fixture `tests/fixtures/preprocess/synthetic_panel.png`）。

---

## Spec 對照

設計文件：`docs/superpowers/specs/2026-05-06-train-new-edge-extend-design.md`

| Spec 區塊 | 對應 Task |
|-----------|-----------|
| §1 Edge 外推取樣 | Task 1 |
| §2 移除 inner_panels | Task 2 |
| §3 web 驗證 5→3 | Task 3 |
| §4 Step1 UI | Task 4 |
| §5 顯示層 / docstring | Task 5 |
| §5 docs / scripts | Task 6 |
| §6 Tests（修改現有） | 散在 Task 1-3 內 |
| §6 Tests（新增） | Task 1 |
| 驗收 | Task 7 |

---

## File Map

**新增：**
- `tests/test_capi_preprocess_outer_extend.py` — Task 1 新測試（外推取樣）

**修改：**
- `capi_preprocess.py` — Task 1（PreprocessConfig + _generate_tiles）
- `capi_train_new.py` — Task 2（移除 inner_panels 全部痕跡）
- `capi_web.py` — Task 3（驗證 + docstring + USER_TRAINABLE_PARAM_SPECS）
- `templates/train_new/step1_select.html` — Task 4
- `templates/train_new/step5_done.html` — Task 5
- `templates/train_new/step2_progress.html` — Task 5
- `templates/models.html` — Task 5
- `tests/test_capi_train_new_preprocess.py` — Task 2
- `tests/test_capi_train_new_training.py` — Task 2
- `tests/test_capi_web_train_new.py` — Task 3
- `tests/test_capi_database_train.py` — Task 3
- `docs/patchcore_training_architecture.zh-TW.md` — Task 6
- `scripts/build_deploy_zip.py` — Task 6

---

## Task 1：Preprocess — `outer_edge_extend` 與外推 tile 產生

**Files:**
- Modify: `capi_preprocess.py:14-21` (PreprocessConfig), `capi_preprocess.py:358-399` (_generate_tiles)
- Create: `tests/test_capi_preprocess_outer_extend.py`

### Step 1.1：寫第一個失敗測試（PreprocessConfig 預設值）

- [ ] 建立 `tests/test_capi_preprocess_outer_extend.py`，內容：

```python
"""capi_preprocess.outer_edge_extend：訓練 edge tile 外推取樣。"""
from __future__ import annotations
import numpy as np
import cv2
import pytest

from capi_preprocess import (
    PreprocessConfig,
    _generate_tiles,
    detect_panel_polygon,
)


def _make_panel_image(img_h: int, img_w: int, panel_xyxy):
    """建一張黑底 + 灰 panel 矩形的合成圖。"""
    px1, py1, px2, py2 = panel_xyxy
    img = np.zeros((img_h, img_w), dtype=np.uint8)
    img[py1:py2, px1:px2] = 180
    return img


def _detect(img, cfg=None):
    cfg = cfg or PreprocessConfig(enable_panel_polygon=False)
    bbox, polygon = detect_panel_polygon(img, cfg)
    return bbox, polygon


def test_default_outer_edge_extend_is_256():
    cfg = PreprocessConfig()
    assert cfg.outer_edge_extend == 256
```

- [ ] 執行：`pytest tests/test_capi_preprocess_outer_extend.py::test_default_outer_edge_extend_is_256 -v`
- [ ] 預期：FAIL — `AttributeError: 'PreprocessConfig' object has no attribute 'outer_edge_extend'`

### Step 1.2：加 PreprocessConfig 欄位

- [ ] 修改 `capi_preprocess.py:14-21`：

```python
@dataclass
class PreprocessConfig:
    tile_size: int = 512
    tile_stride: int = 512
    otsu_offset: int = 5
    enable_panel_polygon: bool = True
    edge_threshold_px: int = 768  # retained for config compatibility; zone split is coverage-based
    coverage_min: float = 0.3
    # Edge tile 中心可往 panel bbox 邊外推的最大距離（px）。預設 256（= half tile_size），
    # 讓「中心位於 panel 邊」的 tile 進入訓練集。0 = 關閉外推（向後相容舊行為）。
    # 實際 push 距離會夾到 image 邊界，使 tile 始終落在 image 內。
    outer_edge_extend: int = 256
```

- [ ] 執行：`pytest tests/test_capi_preprocess_outer_extend.py::test_default_outer_edge_extend_is_256 -v`
- [ ] 預期：PASS

### Step 1.3：寫外推 tile 產生測試（panel 在中央，full push）

- [ ] 在 `tests/test_capi_preprocess_outer_extend.py` 加入：

```python
def test_outer_edge_extend_adds_extension_tiles_for_centered_panel():
    """panel 在 image 中央、各邊距 >= 256，外推應該每邊多出一排 tile + 4 個角。"""
    img = _make_panel_image(2000, 2400, (500, 500, 1900, 1500))
    cfg_off = PreprocessConfig(enable_panel_polygon=False, outer_edge_extend=0)
    cfg_on = PreprocessConfig(enable_panel_polygon=False, outer_edge_extend=256)
    bbox, _ = _detect(img, cfg_off)

    base = _generate_tiles(img, bbox, polygon=None, config=cfg_off)
    extended = _generate_tiles(img, bbox, polygon=None, config=cfg_on)

    # 全部多出來的 tile 都該歸 edge zone。
    base_pos = {(t.x1, t.y1) for t in base}
    extra = [t for t in extended if (t.x1, t.y1) not in base_pos]
    assert len(extra) > 0
    assert all(t.zone == "edge" for t in extra), "extension tiles 必須是 edge"

    # 至少要有上下左右四側 + 四個角各一個 tile（外推）。
    x1, y1, x2, y2 = bbox
    ts = cfg_on.tile_size
    extend = cfg_on.outer_edge_extend
    expected_top_y = y1 - extend
    expected_bot_y = y2 - ts + extend
    expected_left_x = x1 - extend
    expected_right_x = x2 - ts + extend
    extra_xy = {(t.x1, t.y1) for t in extra}
    assert any(y == expected_top_y for _, y in extra_xy), "缺 top 外推行"
    assert any(y == expected_bot_y for _, y in extra_xy), "缺 bottom 外推行"
    assert any(x == expected_left_x for x, _ in extra_xy), "缺 left 外推列"
    assert any(x == expected_right_x for x, _ in extra_xy), "缺 right 外推列"
    # 4 個角的 tile 同時出現於 top/bottom 行 ∩ left/right 列
    corners = {
        (expected_left_x, expected_top_y),
        (expected_right_x, expected_top_y),
        (expected_left_x, expected_bot_y),
        (expected_right_x, expected_bot_y),
    }
    assert corners.issubset(extra_xy), f"缺角落 tile，extra_xy={extra_xy}"


def test_outer_edge_extend_clamps_to_image_boundary():
    """panel 上邊離 image 上邊只有 100px 時，push_top 必須夾到 100，
    使 top 外推 tile 完全落在 image 內。"""
    img = _make_panel_image(1200, 2400, (500, 100, 1900, 1100))
    cfg = PreprocessConfig(enable_panel_polygon=False, outer_edge_extend=256)
    bbox, _ = _detect(img, cfg)
    x1, y1, x2, y2 = bbox

    # margin = y1 - 0；push_top 應為 min(256, y1)。
    expected_push_top = min(256, max(0, y1))
    expected_top_ty = y1 - expected_push_top
    assert expected_top_ty >= 0, "外推 tile 不可超出 image 上邊"

    tiles = _generate_tiles(img, bbox, polygon=None, config=cfg)
    top_extra = [t for t in tiles if t.y1 == expected_top_ty]
    assert top_extra, f"應產生 ty={expected_top_ty} 的外推 tile"
    # tile 完全在 image 內：tile size 切片應是 ts × ts
    for t in top_extra:
        assert t.image.shape[0] == cfg.tile_size
        assert t.image.shape[1] == cfg.tile_size


def test_outer_edge_extend_skips_side_with_no_margin():
    """panel 緊貼 image 上邊（y1=0）時，不產生 top 外推 tile。"""
    img = _make_panel_image(1200, 2400, (500, 0, 1900, 1100))
    cfg = PreprocessConfig(enable_panel_polygon=False, outer_edge_extend=256)
    bbox, _ = _detect(img, cfg)
    x1, y1, x2, y2 = bbox

    tiles = _generate_tiles(img, bbox, polygon=None, config=cfg)
    # bbox 上邊 y1 加上 otsu_offset 後仍很小（5），push_top = 5
    assert y1 < 256, "fixture 應使 y1 接近 0"
    # ty < y1 - 5 的 tile（即真的往外推超過 otsu_offset 的）不應存在
    extension_top_tiles = [t for t in tiles if t.y1 < 0]
    assert not extension_top_tiles, "panel 緊貼 image 上邊時不可產生 ty<0 的 tile"


def test_outer_edge_extend_zero_matches_legacy_behavior():
    """outer_edge_extend=0 → 結果與舊行為一致（無外推 tile）。"""
    img = _make_panel_image(2000, 2400, (500, 500, 1900, 1500))
    cfg_legacy = PreprocessConfig(enable_panel_polygon=False, outer_edge_extend=0)
    bbox, _ = _detect(img, cfg_legacy)
    tiles = _generate_tiles(img, bbox, polygon=None, config=cfg_legacy)

    x1, y1, x2, y2 = bbox
    # 任何 tile 不應在 bbox 之外。
    for t in tiles:
        assert t.x1 >= x1 and t.y1 >= y1
        assert t.x2 <= x2 and t.y2 <= y2


def test_outer_edge_extend_corner_tile_forced_to_edge_zone():
    """角落外推 tile coverage 約 25%（小於 coverage_min=0.3），
    若不強制 edge 會被歸 outside 跳掉；本測驗證強制成功。"""
    img = _make_panel_image(2000, 2400, (500, 500, 1900, 1500))
    # 啟用 polygon，否則 zone 會 fallback 為 inner（polygon=None 路徑）
    cfg = PreprocessConfig(enable_panel_polygon=True, outer_edge_extend=256)
    bbox, polygon = _detect(img, cfg)
    assert polygon is not None, "fixture 應能 fit polygon"

    tiles = _generate_tiles(img, bbox, polygon=polygon, config=cfg)
    x1, y1, x2, y2 = bbox
    ts = cfg.tile_size
    corner_tl = (x1 - cfg.outer_edge_extend, y1 - cfg.outer_edge_extend)
    corner_tile = next((t for t in tiles if (t.x1, t.y1) == corner_tl), None)
    assert corner_tile is not None, "缺左上角外推 tile"
    assert corner_tile.zone == "edge"
    # coverage 應該 < 0.3 但 tile 仍存在 → 證明有強制 edge bypass
    assert corner_tile.coverage < cfg.coverage_min, (
        f"corner coverage {corner_tile.coverage} 應小於 coverage_min "
        f"才能驗證 zone 強制 edge 的 bypass"
    )
```

- [ ] 執行：`pytest tests/test_capi_preprocess_outer_extend.py -v`
- [ ] 預期：5 個測試全部 FAIL（外推邏輯尚未實作）

### Step 1.4：實作 `_generate_tiles` 外推邏輯

- [ ] 修改 `capi_preprocess.py:358-399`，整段 `_generate_tiles` 改為：

```python
def _generate_tiles(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    polygon: Optional[np.ndarray],
    config: PreprocessConfig,
) -> List[TileResult]:
    """在 bbox 範圍內走格子 + 外推一圈，切 tile 並分類 zone。

    內圈 tile 走原本邏輯（outside 跳掉，最外圈 inner 改 edge）。
    外圈 tile 來自 ``outer_edge_extend``：每邊往外推到 image 邊界內，
    強制 zone="edge"（避免角落 coverage<0.3 被歸 outside 跳掉），
    讓「中心位於 panel 邊」的 tile 進入訓練集。
    """
    x1, y1, x2, y2 = bbox
    ts = config.tile_size
    img_h, img_w = img.shape[:2]

    def positions(lo: int, hi: int) -> List[int]:
        if hi - lo < ts:
            return []
        out = list(range(lo, hi - ts + 1, config.tile_stride))
        if out and out[-1] != hi - ts:
            out.append(hi - ts)
        return out

    xs = positions(x1, x2)
    ys = positions(y1, y2)

    # 外推：push 距離夾到 image 邊界，確保 tile 完全在 image 內。
    extend = max(0, int(config.outer_edge_extend))
    push_top    = min(extend, max(0, y1))
    push_bottom = min(extend, max(0, img_h - y2))
    push_left   = min(extend, max(0, x1))
    push_right  = min(extend, max(0, img_w - x2))

    # 外推位置：top/bottom 行的 xs 含左右外推 col；left/right 列的 ys 不含 top/bottom 行避免角落重複。
    top_ty    = (y1 - push_top)             if push_top > 0    else None
    bottom_ty = (y2 - ts + push_bottom)     if push_bottom > 0 else None
    left_tx   = (x1 - push_left)            if push_left > 0   else None
    right_tx  = (x2 - ts + push_right)      if push_right > 0  else None

    extra_xs = []
    if left_tx is not None:
        extra_xs.append(left_tx)
    extra_xs.extend(xs)
    if right_tx is not None:
        extra_xs.append(right_tx)

    extension_positions: List[Tuple[int, int]] = []
    if top_ty is not None:
        extension_positions.extend((tx, top_ty) for tx in extra_xs)
    if bottom_ty is not None:
        extension_positions.extend((tx, bottom_ty) for tx in extra_xs)
    if left_tx is not None:
        extension_positions.extend((left_tx, ty) for ty in ys)
    if right_tx is not None:
        extension_positions.extend((right_tx, ty) for ty in ys)
    # 去重（理論上不會重複，但 stride 邊界湊巧時防一手）
    extension_positions = list(dict.fromkeys(extension_positions))

    tiles: List[TileResult] = []
    tid = 0

    # 1) 內圈 grid（原邏輯）
    for ty in ys:
        for tx in xs:
            tile_rect = (tx, ty, tx + ts, ty + ts)
            zone, cov, dist, mask = classify_tile_zone(tile_rect, polygon, config)
            if zone == "outside":
                continue
            if zone == "inner" and (tx == xs[0] or tx == xs[-1] or ty == ys[0] or ty == ys[-1]):
                zone = "edge"
            tile_img = img[ty:ty + ts, tx:tx + ts].copy()
            tiles.append(TileResult(
                tile_id=tid,
                x1=tx, y1=ty, x2=tx + ts, y2=ty + ts,
                image=tile_img,
                mask=mask,
                coverage=cov,
                zone=zone,
                center_dist_to_edge=dist,
            ))
            tid += 1

    # 2) 外圈外推 tile（強制 zone="edge"）
    for tx, ty in extension_positions:
        tile_rect = (tx, ty, tx + ts, ty + ts)
        _zone, cov, dist, mask = classify_tile_zone(tile_rect, polygon, config)
        # push 已夾到 image 內，img 切片必為 ts × ts
        tile_img = img[ty:ty + ts, tx:tx + ts].copy()
        tiles.append(TileResult(
            tile_id=tid,
            x1=tx, y1=ty, x2=tx + ts, y2=ty + ts,
            image=tile_img,
            mask=mask,
            coverage=cov,
            zone="edge",  # 強制：corner tile coverage<0.3 也算 edge
            center_dist_to_edge=dist,
        ))
        tid += 1

    return tiles
```

- [ ] 執行：`pytest tests/test_capi_preprocess_outer_extend.py -v`
- [ ] 預期：5 個測試全部 PASS

### Step 1.5：跑既有 preprocess 測試確認沒回歸

- [ ] 執行：`pytest tests/test_capi_preprocess.py tests/test_capi_train_new_preprocess.py -v`
- [ ] 預期：全 PASS（既有測試使用 `outer_edge_extend=256` 預設值，多出來的 edge tile 不影響原本 assertion；若 fixture margin 太小可能無外推）

> 若 `tests/test_capi_train_new_preprocess.py::test_preprocess_panels_to_pool_writes_tiles` 因外推 tile 數變多而失敗，**只調整 assertion 至大於等於原本下限**，不要關閉 outer_edge_extend；外推就是新預設行為。

### Step 1.6：commit

- [ ] 執行：

```bash
git add capi_preprocess.py tests/test_capi_preprocess_outer_extend.py
git commit -m "feat(preprocess): 加 outer_edge_extend，每邊外推一排 edge tile

push 距離夾到 image 邊界，確保 tile 完全落在 image 內、用真實
自然黑邊像素而非 zero-pad。角落外推 tile coverage<0.3 強制 edge，
不被 classify_tile_zone 當 outside 跳掉。"
```

---

## Task 2：`capi_train_new` — 移除 `inner_panels`

**Files:**
- Modify: `capi_train_new.py:58-87` (TrainingConfig + USER_TRAINABLE_PARAM_SPECS), `capi_train_new.py:116-186` (preprocess_panels_to_pool), `capi_train_new.py:1037-1057` (run_training_pipeline manifest)
- Modify: `tests/test_capi_train_new_preprocess.py:69-115`
- Modify: `tests/test_capi_train_new_training.py:9-32`

### Step 2.1：先改測試（紅 → 綠 順序）— `test_capi_train_new_training.py`

- [ ] 修改 `tests/test_capi_train_new_training.py:9-33`：

```python
def test_apply_user_training_params_none_is_noop():
    from capi_train_new import TrainingConfig, apply_user_training_params
    cfg = TrainingConfig(
        machine_id="M", panel_paths=[], over_review_root=Path("/r"),
    )
    snapshot = (cfg.batch_size, cfg.coreset_ratio, cfg.max_epochs)
    apply_user_training_params(cfg, None)
    apply_user_training_params(cfg, {})
    assert (cfg.batch_size, cfg.coreset_ratio, cfg.max_epochs) == snapshot


def test_apply_user_training_params_overrides_match_keys():
    from capi_train_new import TrainingConfig, apply_user_training_params
    cfg = TrainingConfig(
        machine_id="M", panel_paths=[], over_review_root=Path("/r"),
    )
    apply_user_training_params(cfg, {
        "batch_size": 16, "coreset_ratio": 0.05, "max_epochs": 2,
    })
    assert cfg.batch_size == 16
    assert cfg.coreset_ratio == 0.05
    assert cfg.max_epochs == 2
```

- [ ] 執行：`pytest tests/test_capi_train_new_training.py::test_apply_user_training_params_none_is_noop tests/test_capi_train_new_training.py::test_apply_user_training_params_overrides_match_keys -v`
- [ ] 預期：FAIL（`inner_panels` 仍是 TrainingConfig 欄位、仍在 USER_TRAINABLE_PARAM_SPECS 內，但測試本身不再 reference 它，反而會把 `apply_user_training_params({"inner_panels": 4})` 移除導致原 assertion 通過。實際上若把 inner_panels 從 SPECS 拿掉之後，下一個 unknown_key_raises 測試才會 fail）

> 此 step 僅 reset baseline；真正的 fail 發生在 step 2.2 的 unknown_key_raises 測試（移除 SPECS 後 inner_panels 變 unknown）。

### Step 2.2：改 `tests/test_capi_train_new_preprocess.py:69-115`

- [ ] 把 `test_preprocess_panels_to_pool_skips_inner_after_inner_panels` 整段替換為：

```python
def test_preprocess_panels_to_pool_all_panels_have_inner_and_edge(tmp_path):
    """每片 panel 都應同時有 inner + edge tile（移除 inner_panels 條件分支後）。"""
    from pathlib import Path
    from capi_preprocess import PreprocessConfig
    from capi_train_new import preprocess_panels_to_pool, TrainingConfig

    fixture_img = Path("tests/fixtures/preprocess/synthetic_panel.png")
    panel_dirs = []
    for i in range(3):
        d = tmp_path / f"panel_{i + 1}"
        d.mkdir()
        for lighting in ["G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD"]:
            (d / f"{lighting}_x.png").write_bytes(fixture_img.read_bytes())
        panel_dirs.append(d)

    class MockDB:
        def __init__(self):
            self.tiles_per_call = []

        def insert_tile_pool(self, job_id, tiles):
            self.tiles_per_call.append(list(tiles))
            return list(range(len(tiles)))

    db = MockDB()
    cfg = TrainingConfig(
        machine_id="TEST", panel_paths=panel_dirs,
        over_review_root=tmp_path / "or_unused",
    )
    pre_cfg = PreprocessConfig(tile_size=256, edge_threshold_px=384, tile_stride=256)

    preprocess_panels_to_pool(
        job_id="j_split", cfg=cfg, preprocess_cfg=pre_cfg,
        db=db, thumb_dir=tmp_path / "thumbs", log=lambda m: None,
    )

    assert len(db.tiles_per_call) == 3
    for i, batch in enumerate(db.tiles_per_call):
        zones = {t["zone"] for t in batch}
        assert "inner" in zones, f"panel {i + 1} 應該包含 inner tile"
        assert "edge" in zones, f"panel {i + 1} 應該包含 edge tile"
```

- [ ] 執行：`pytest tests/test_capi_train_new_preprocess.py::test_preprocess_panels_to_pool_all_panels_have_inner_and_edge -v`
- [ ] 預期：PASS（既有 `inner_panels=3` 預設下，3 片都該有 inner+edge）。若 fixture 太小導致某 lighting 沒 inner，**改用 tile_size=128** 避免無 inner tile。

### Step 2.3：實作 `capi_train_new.py` 改動 — TrainingConfig + SPECS

- [ ] 修改 `capi_train_new.py:58-87`：

```python
@dataclass
class TrainingConfig:
    machine_id: str
    panel_paths: List[Path]
    over_review_root: Path
    output_root: Path = Path("model")
    backbone_cache_dir: Path = Path("deployment/torch_hub_cache")
    required_backbones: List[str] = field(
        default_factory=lambda: ["wide_resnet50_2-32ee1156.pth"]
    )

    batch_size: int = 8
    image_size: tuple = (512, 512)
    coreset_ratio: float = 0.1
    max_epochs: int = 1


# 使用者可從 step1 表單覆寫的 PatchCore 超參數，與其合法值範圍。
# 同時做為前後端的單一資料來源：capi_web 的請求驗證、
# step1 的前端表單、capi_train_runner 套用、以及未知 key 防呆都讀此表。
USER_TRAINABLE_PARAM_SPECS: Dict[str, Dict] = {
    "batch_size":    {"type": int,   "min": 1,    "max": 32},
    "coreset_ratio": {"type": float, "min": 0.01, "max": 0.5},
    "max_epochs":    {"type": int,   "min": 1,    "max": 10},
}
USER_TRAINABLE_PARAM_NAMES: Tuple[str, ...] = tuple(USER_TRAINABLE_PARAM_SPECS.keys())
```

### Step 2.4：實作改動 — `preprocess_panels_to_pool`

- [ ] 修改 `capi_train_new.py:132-180`（panel 迴圈內），把 inner_panels 邏輯整段拔掉：

```python
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
                cv2.imwrite(str(tile_path), tile.image)

                thumb_path = thumb_dir / "thumb" / tile_filename
                thumb = cv2.resize(tile.image, (96, 96))
                cv2.imwrite(str(thumb_path), thumb)

                tile_records.append({
                    "lighting": lighting,
                    "zone": tile.zone,
                    "source": "ok",
                    "source_path": str(tile_path.resolve()),
                    "thumb_path": str(thumb_path.resolve()),
                })

        if tile_records:
            db.insert_tile_pool(job_id, tile_records)
            total_tiles += len(tile_records)
            panel_success += 1
            log(f"  ✓ 切出 {len(tile_records)} tile")
```

### Step 2.5：實作改動 — `run_training_pipeline` manifest 移除 inner_panels

- [ ] 修改 `capi_train_new.py:1043-1052`（patchcore_params 區塊）：

```python
        "patchcore_params": {
            "batch_size": cfg.batch_size,
            "image_size": list(cfg.image_size),
            "coreset_ratio": cfg.coreset_ratio,
            "max_epochs": cfg.max_epochs,
        },
```

### Step 2.6：執行所有 capi_train_new 相關測試

- [ ] 執行：`pytest tests/test_capi_train_new_training.py tests/test_capi_train_new_preprocess.py -v`
- [ ] 預期：全 PASS

### Step 2.7：commit

- [ ] 執行：

```bash
git add capi_train_new.py tests/test_capi_train_new_training.py tests/test_capi_train_new_preprocess.py
git commit -m "refactor(train): 移除 inner_panels，所有 panel 都收 inner+edge

簡化訓練流程：固定 3 片，無條件分支。配合 outer_edge_extend
新增的外推 tile，edge 樣本量已足夠。manifest patchcore_params
不再寫入 inner_panels 欄位。"
```

---

## Task 3：`capi_web` — Panel 數驗證 5→3 + 移除 inner_panels SPEC

**Files:**
- Modify: `capi_web.py:5021-5022` (panel count check), `capi_web.py:4985-4998` (docstring)
- Modify: `tests/test_capi_web_train_new.py:235, 253, 276-282, 298-311, 338, 380-389`
- Modify: `tests/test_capi_database_train.py:185-199`

### Step 3.1：改 web 驗證測試

- [ ] 修改 `tests/test_capi_web_train_new.py`：把所有 `[f"/p{i}" for i in range(5)]` 改 `range(3)`（line 235, 253, 338, 383）

- [ ] 修改 `tests/test_capi_web_train_new.py:276-282`：

```python
    def test_full_valid_dict(self):
        from capi_web import CAPIWebHandler
        raw = {"batch_size": 16, "coreset_ratio": 0.05, "max_epochs": 2}
        params, err = CAPIWebHandler._validate_training_params(raw)
        assert err is None
        assert params == raw
```

- [ ] 修改 `tests/test_capi_web_train_new.py:298-311`（移除 inner_panels 越界 case）：

```python
    def test_out_of_range_rejected(self):
        from capi_web import CAPIWebHandler
        for raw in [
            {"batch_size": 0},
            {"batch_size": 64},
            {"coreset_ratio": 0.0},
            {"coreset_ratio": 0.6},
            {"max_epochs": 0},
            {"max_epochs": 100},
        ]:
            _, err = CAPIWebHandler._validate_training_params(raw)
            assert err and "out of range" in err, f"expected error for {raw}"
```

- [ ] 修改 `tests/test_capi_web_train_new.py:380-389`：

```python
    payload = {
        "machine_id": "M",
        "panel_paths": [f"/p{i}" for i in range(3)],
        "training_params": {
            "batch_size": 16, "coreset_ratio": 0.05, "max_epochs": 2,
        },
    }
```

### Step 3.2：新增測試 — 確認 5/2 panels 被拒、3 panels 接受

- [ ] 在 `tests/test_capi_web_train_new.py` 末尾加入：

```python
def test_handle_train_new_start_rejects_wrong_panel_count():
    """非 3 片 panel 一律拒絕。"""
    server = MagicMock()
    server.database.get_active_training_job.return_value = None

    for n in (0, 1, 2, 4, 5, 6):
        h = _make_handler_with_server(server, "/api/train/new/start")
        body = json.dumps({"machine_id": "M", "panel_paths": [f"/p{i}" for i in range(n)]}).encode()
        h.headers.get = MagicMock(return_value=str(len(body)))
        h.rfile = io.BytesIO(body)

        h._handle_train_new_start()
        assert h._sent_response[0]["status"] == 400, f"n={n} 應該被拒"
        if n > 0:
            err_body = json.loads(h._sent_response[0]["body"])
            assert "exactly 3" in err_body.get("error", ""), f"n={n} error: {err_body}"
```

### Step 3.3：改 DB round-trip 測試

- [ ] 修改 `tests/test_capi_database_train.py:185-199`：

```python
    def test_training_params_round_trip(self, tmp_path):
        """create 帶 dict → get 反序列化回 dict。"""
        db = _make_db(tmp_path)
        params = {
            "batch_size": 16,
            "coreset_ratio": 0.05,
            "max_epochs": 2,
        }
        db.create_training_job(
            job_id="j_with_params", machine_id="M", panel_paths=[],
            training_params=params,
        )
        job = db.get_training_job("j_with_params")
        assert job["training_params"] == params
```

### Step 3.4：執行測試確認 fail（panel 數測試應該 fail；training_params 測試 PASS）

- [ ] 執行：`pytest tests/test_capi_web_train_new.py tests/test_capi_database_train.py -v`
- [ ] 預期：`test_handle_train_new_start_rejects_wrong_panel_count` 中 n=5 時 status 仍會通過 → 因為現在的程式接受 5；其他測試應 PASS（前提 Task 2 已合）

### Step 3.5：實作 web 改動 — 驗證 5→3 與 docstring

- [ ] 修改 `capi_web.py:5021-5022`：

```python
        if len(clean_panel_paths) != 3:
            self._send_json({"error": "panel_paths must contain exactly 3 panels"}, status=400)
            return
```

- [ ] 修改 `capi_web.py:4985-4998`（docstring 移除 inner_panels）：

```python
    def _handle_train_new_start(self):
        """POST /api/train/new/start

        body: {
            "machine_id": "...",
            "panel_paths": [...],   # 必須是 3 片
            "training_params": {    # optional
                "batch_size": 8,
                "coreset_ratio": 0.1,
                "max_epochs": 1
            }
        }
        """
```

### Step 3.6：執行測試確認全 PASS

- [ ] 執行：`pytest tests/test_capi_web_train_new.py tests/test_capi_database_train.py -v`
- [ ] 預期：全 PASS

### Step 3.7：commit

- [ ] 執行：

```bash
git add capi_web.py tests/test_capi_web_train_new.py tests/test_capi_database_train.py
git commit -m "feat(web): /api/train/new/start 改為要求 3 片 panel

對應 inner_panels 移除：USER_TRAINABLE_PARAM_SPECS 已不含此 key，
training_params 帶 inner_panels 會被當 unknown 拒絕。"
```

---

## Task 4：Step1 UI — 文案 / 進階參數 / 選取上限

**Files:**
- Modify: `templates/train_new/step1_select.html`

> 這個 task 沒有 unit test 可寫（純 HTML/JS）。改完後在 step 4.4 做手動 smoke test。

### Step 4.1：改文案區塊（line 8-14）

- [ ] 修改 `templates/train_new/step1_select.html:8-14`：

```html
  <h1 style="color:#cdd6f4;margin:0 0 6px 0;font-size:1.4rem;">Step 1 / 5 · 選擇訓練資料</h1>
  <p style="color:#a6adc8;font-size:.9rem;line-height:1.55;">
    從最近 3 天 AOI 判 OK 紀錄勾選
    <span style="color:#a6e3a1;font-weight:600;">3 片 panel</span>，皆收 inner + edge tile。
    Edge 取樣會自動往 panel 邊緣外推（涵蓋邊緣鄰近的 ROI 切塊情境）。
    <span style="color:#cdd6f4;font-size:.82rem;display:block;margin-top:2px;">勾選順序就是 panel 順位 — 取消重選會重排。</span>
  </p>
```

### Step 4.2：移除進階參數的 inner_panels 欄位（line 53-57）

- [ ] 在 `templates/train_new/step1_select.html` 進階參數 div（`<details id="train-params">` 內）刪除整段 `<label class="tp-field">` 的 inner_panels 欄位：

刪除這段：
```html
      <label class="tp-field">
        <span class="tp-label">inner_panels <span class="tp-default">預設 3</span></span>
        <input type="number" id="tp-inner_panels" min="1" max="5" step="1" placeholder="3">
        <span class="tp-hint">前 N 片收 inner+edge tile，第 N+1~5 片只收 edge</span>
      </label>
```

### Step 4.3：JS — 選取上限與 role color 簡化

- [ ] 修改 `templates/train_new/step1_select.html:80`：

```html
    <span id="sel-count" style="color:#a6e3a1;font-weight:600;">已選 0/3 片</span>
    <span id="sel-roles" style="color:#a6adc8;font-size:.78rem;display:none;"></span>
```

- [ ] 修改 `templates/train_new/step1_select.html:182-185`（INNER_PANELS / _roleColor）：

```javascript
// 全部 panel 都收 inner+edge，不再依順位分色
const PANEL_LIMIT = 3;
function _roleColor(_idx) { return '#a6e3a1'; }
```

- [ ] 修改 `templates/train_new/step1_select.html:194`：

```javascript
    if (_selected.size >= PANEL_LIMIT) return;
```

- [ ] 修改 `templates/train_new/step1_select.html:217-229`（refreshSelectedVisuals 內 role 色塊邏輯，改為單色）：

```javascript
  const ordered = Array.from(_selected);
  ordered.forEach((path, idx) => {
    const tr = document.querySelector(`#panel-list tr[data-path="${CSS.escape(path)}"]`);
    if (!tr) return;
    const color = _roleColor(idx);
    tr.style.background = '#1e3a2d';
    const chk = tr.querySelector('.chk');
    chk.textContent = String(idx + 1);
    chk.style.background = color;
    chk.style.color = '#1e1e2e';
    chk.style.borderColor = color;
    chk.style.fontWeight = '700';
  });
```

- [ ] 修改 `templates/train_new/step1_select.html:232-249`（updateNext）：

```javascript
function updateNext() {
  const machineText = _selectedMachineId ? `（機種 ${_selectedMachineId}）` : '';
  document.getElementById('sel-count').textContent = `已選 ${_selected.size}/${PANEL_LIMIT} 片 ${machineText}`;
  // 不再依 inner/edge 分組顯示，roles 區塊永遠隱藏
  document.getElementById('sel-roles').style.display = 'none';
  const btn = document.getElementById('next-btn');
  btn.disabled = _selected.size !== PANEL_LIMIT;
  btn.style.opacity = btn.disabled ? .4 : 1;
}
```

- [ ] 修改 `templates/train_new/step1_select.html:252-257`（TRAIN_PARAM_KEYS 移除 inner_panels）：

```javascript
const TRAIN_PARAM_KEYS = [
  {key: 'batch_size',    type: 'int',   min: 1,    max: 32},
  {key: 'coreset_ratio', type: 'float', min: 0.01, max: 0.5},
  {key: 'max_epochs',    type: 'int',   min: 1,    max: 10},
];
```

### Step 4.4：手動 smoke test

- [ ] 起 server：`python capi_server.py --config server_config_local.yaml`
- [ ] 開瀏覽器：`http://localhost:8080/train/new`
- [ ] 確認：
  - 標題下方文案改為「3 片」
  - sticky bar 顯示「已選 0/3 片」
  - 進階設定展開後**沒有** inner_panels 欄位
  - 選 3 片後「下一步」按鈕亮起；嘗試選第 4 片應被擋下
  - 選滿 3 片時 chk 顏色全為綠色（沒有橘色 4-5 片區別）
- [ ] 截圖（可選）

### Step 4.5：commit

- [ ] 執行：

```bash
git add templates/train_new/step1_select.html
git commit -m "feat(ui): Step1 改為固定 3 片，移除 inner_panels 進階設定

文案、選取上限、role color、TRAIN_PARAM_KEYS 全部對齊
新的 3 片皆 inner+edge 流程。"
```

---

## Task 5：顯示層 — Step5 / Models / Step2 regex

**Files:**
- Modify: `templates/train_new/step5_done.html:76`
- Modify: `templates/models.html:285`
- Modify: `templates/train_new/step2_progress.html:6, 372`

### Step 5.1：Step5 done 頁 — 條件顯示 inner_panels（舊 bundle 才有）

- [ ] 修改 `templates/train_new/step5_done.html:76`：

```html
        <dl class="d5-params">
          {% if patchcore_params.get("inner_panels") is not none %}
            <dt>inner_panels</dt><dd>{{ patchcore_params["inner_panels"] }}</dd>
          {% endif %}
          <dt>batch_size</dt><dd>{{ patchcore_params.get("batch_size", "—") }}</dd>
```

### Step 5.2：Models 頁 — 條件顯示 inner_panels

- [ ] 修改 `templates/models.html:280-287`：

```javascript
    rows.push(
      ['─── PatchCore 參數 ───', '─────'],
      ['batch_size', pp.batch_size != null ? pp.batch_size : '—'],
      ['coreset_ratio', pp.coreset_ratio != null ? pp.coreset_ratio : '—'],
      ['max_epochs', pp.max_epochs != null ? pp.max_epochs : '—'],
      ['image_size', Array.isArray(pp.image_size) ? pp.image_size.join('×') : '—'],
    );
    if (pp.inner_panels != null) {
      // 舊 bundle 才會有此欄
      rows.push(['inner_panels (legacy)', pp.inner_panels]);
    }
```

### Step 5.3：Step2 進度頁 regex — 拿掉 (inner+edge|edge only)

- [ ] 修改 `templates/train_new/step2_progress.html:5-8`（註解）：

```html
{# Step 2 前處理進度頁。
   後端 log 由 capi_train_new.py 寫，可解析的訊號：
     [N/M] panel <id>                          ← panel 進度
     ✓ 切出 N tile                              ← tile 累加
     ⚠ ... polygon 偵測失敗                     ← 警告計數
```

- [ ] 修改 `templates/train_new/step2_progress.html:372`：

```javascript
const RE_PANEL = /^\s*\[(\d+)\/(\d+)\]\s+panel\s+(\S+)/;
```

### Step 5.4：手動 smoke test（可選）

- [ ] 跑一次完整訓練（或載入既有舊 bundle）→ 確認 Step5 / Models 頁面顯示正確

### Step 5.5：commit

- [ ] 執行：

```bash
git add templates/train_new/step5_done.html templates/models.html templates/train_new/step2_progress.html
git commit -m "fix(ui): inner_panels 顯示改為條件 render，step2 log regex 簡化

舊 bundle manifest 仍含 inner_panels，新 bundle 不會有；UI 兩種
情況都要相容。step2 panel 進度行不再附 (inner+edge|edge only)
標籤，regex 對應簡化。"
```

---

## Task 6：Docs / Scripts

**Files:**
- Modify: `docs/patchcore_training_architecture.zh-TW.md:84`
- Modify: `scripts/build_deploy_zip.py:120`

### Step 6.1：架構文件改 5→3

- [ ] 修改 `docs/patchcore_training_architecture.zh-TW.md` line 84（用 Read 確認當前文字後再 Edit）：

行內容約：
> Wizard 要求選滿 5 片。

改為：
> Wizard 要求選滿 3 片，每片皆收 inner + edge tile（含 edge 外推取樣）。

### Step 6.2：deploy zip release notes 改 5→3

- [ ] 修改 `scripts/build_deploy_zip.py:120`（用 Read 確認上下文後 Edit）：

把含「勾選 5 片」字串改成「勾選 3 片」。

### Step 6.3：commit

- [ ] 執行：

```bash
git add docs/patchcore_training_architecture.zh-TW.md scripts/build_deploy_zip.py
git commit -m "docs: 訓練 wizard panel 數 5→3 對齊新流程"
```

---

## Task 7：整合驗收

### Step 7.1：跑全部 unit test

- [ ] 執行：

```bash
pytest tests/ -v --ignore=tests/test_inference.py --ignore=tests/test_aoi_coord_inference.py 2>&1 | tail -40
```

(忽略需要實際 server 的 integration test)

- [ ] 預期：相關修改的測試全 PASS；不相關測試保持原狀

### Step 7.2：手動驗收清單

- [ ] Step 1 頁：顯示「3 片」、無 inner_panels 進階欄位、第 4 片無法勾、3 片皆綠色
- [ ] Step 3 預覽：panel polygon 外圍多一圈橘色 edge tile（需訓 1 個 job 走到 step3 才能看）
- [ ] 訓練後 Step 5：`PATCHCORE PARAMS` 區塊不出現 `inner_panels`（新 bundle）
- [ ] Models 頁：載入舊 bundle 仍顯示 `inner_panels (legacy)`，新 bundle 沒有此欄

### Step 7.3：sanity check（standalone python）

- [ ] 執行：

```bash
python -c "
from capi_preprocess import PreprocessConfig, _generate_tiles, detect_panel_polygon
import numpy as np
img = np.zeros((2000, 2400), dtype=np.uint8)
img[500:1500, 500:1900] = 180
cfg = PreprocessConfig(enable_panel_polygon=True, outer_edge_extend=256)
bbox, poly = detect_panel_polygon(img, cfg)
tiles = _generate_tiles(img, bbox, poly, cfg)
edge = [t for t in tiles if t.zone == 'edge']
inner = [t for t in tiles if t.zone == 'inner']
print(f'bbox={bbox} | edge tiles={len(edge)} inner tiles={len(inner)}')
print(f'edge x ranges: {sorted({(t.x1, t.y1) for t in edge})[:6]} ...')
"
```

- [ ] 預期：edge tile 數明顯超過原本（含外推一圈），bbox 為 (505, 505, 1895, 1495) 左右

### Step 7.4：（選做）回頭重訓 1 個 job 看 step5 metrics

- [ ] 用相同 panel 重訓一次，比對 step5 各 unit 的 train_count（應比舊 5 片版本略增 ~外推 tile 數）、auroc 不應顯著下降

---

## Self-Review

### 1. Spec 覆蓋率

| Spec 區塊 | Task |
|-----------|------|
| `outer_edge_extend` 預設 256 | 1.2 |
| 4 邊外推、push 夾到 image | 1.3, 1.4 |
| 角落強制 zone=edge | 1.3 (corner_tile test), 1.4 |
| 移除 `inner_panels` | 2.3 |
| `preprocess_panels_to_pool` 拿掉條件 | 2.4 |
| log 簡化 | 2.4 |
| manifest 移除 inner_panels | 2.5 |
| web 驗證 5→3 | 3.5 |
| docstring example | 3.5 |
| step1_select.html 全套改動 | 4.1-4.3 |
| models.html / step5_done.html 條件 render | 5.1, 5.2 |
| step2_progress.html regex | 5.3 |
| docs / scripts | 6.1, 6.2 |
| 既有 tests 修改 | 散在 1.5, 2.1-2.2, 3.1-3.3 |
| 新增 tests | 1.3 |

✓ 全部覆蓋。

### 2. Placeholder scan

✓ 無 TBD/TODO。所有 step 都有具體程式碼或具體指令。

### 3. Type / 命名一致性

- `outer_edge_extend` 名稱整份統一
- `PANEL_LIMIT = 3` 在 step1_select.html 取代 `INNER_PANELS = 3`
- `RE_PANEL` regex 在 Task 5 改寫，前面 Task 2 也對應簡化了 log 訊息（移除 `(inner+edge)` 後綴）

✓ 一致。

### 4. 風險點檢核

- Task 1.5 提到 fixture 太小可能讓既有測試失敗 → 已給 fallback（調 tile_size）
- Task 5 的條件 render 對舊/新 bundle 都相容
