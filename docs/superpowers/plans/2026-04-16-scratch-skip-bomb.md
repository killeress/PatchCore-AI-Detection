# ScratchFilter 跳過炸彈 tile — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `ScratchFilter.apply_to_image_result` 迴圈頭部跳過 `tile.is_bomb=True` 的 tile，避免對已判定為炸彈的 tile 再跑 DINOv2 分類器。

**Architecture:** 單點修改 `scratch_filter.py` 迴圈內加入 `if tile.is_bomb: continue`；保留 bomb tile 的 `scratch_score=0.0` / `scratch_filtered=False` 預設值，下游（判定聚合、DB schema、UI badge）不需改動。

**Tech Stack:** Python 3, pytest, numpy, dataclasses (TileInfo / ImageResult)。

**Spec reference:** `docs/superpowers/specs/2026-04-16-scratch-skip-bomb-design.md`

---

## Task 1: 新增 bomb-skip 單元測試（紅燈）

**Files:**
- Modify: `tests/test_scratch_filter.py`（檔尾新增一個測試函式）

- [ ] **Step 1: 寫失敗測試**

在 `tests/test_scratch_filter.py` 檔尾追加：

```python
def test_filter_skips_bomb_tile():
    """C2 regression: bomb tile 不應進入 classifier；scratch 欄位保持預設值。"""
    from scratch_filter import ScratchFilter
    clf = _MockClassifier(fixed_score=0.99, conformal_threshold=0.7)
    sf = ScratchFilter(clf, safety_multiplier=1.0)   # threshold = 0.7
    ir = _fake_image_result_with_tiles(2)
    # 第一顆 tile 標記為炸彈
    ir.tiles[0].is_bomb = True

    sf.apply_to_image_result(ir)

    # Bomb tile: classifier 未被呼叫，scratch 欄位為預設值
    assert ir.tiles[0].scratch_score == 0.0
    assert ir.tiles[0].scratch_filtered is False
    # Non-bomb tile: 正常跑，高分被翻回 OK
    assert ir.tiles[1].scratch_score == pytest.approx(0.99)
    assert ir.tiles[1].scratch_filtered is True
    # scratch_filter_count 只計非 bomb tile
    assert ir.scratch_filter_count == 1
    # anomaly_tiles 仍保留兩顆 tile（C1 不變性）
    assert len(ir.anomaly_tiles) == 2
```

- [ ] **Step 2: 跑測試確認失敗**

Run:
```
cd C:/Users/rh.syu/Desktop/CAPI01_AD && python -m pytest tests/test_scratch_filter.py::test_filter_skips_bomb_tile -v
```

Expected output（FAIL）:
```
AssertionError: assert 0.99 == 0.0
```
理由：目前 `scratch_filter.py` 對 bomb tile 也會跑 classifier，`scratch_score` 會被寫入 0.99。

- [ ] **Step 3: 不 commit（紅燈狀態 commit 會讓 CI 炸）**

直接進 Task 2。

---

## Task 2: 在 ScratchFilter 迴圈內跳過 bomb tile（綠燈）

**Files:**
- Modify: `scratch_filter.py:36-37`（迴圈頭部加入 skip 條件）

- [ ] **Step 1: 修改 `scratch_filter.py`**

找到 `apply_to_image_result` 的迴圈（目前在第 36–37 行）：

```python
        for entry in image_result.anomaly_tiles:
            tile = entry[0]
            t0 = time.perf_counter()
```

改為：

```python
        for entry in image_result.anomaly_tiles:
            tile = entry[0]
            if getattr(tile, "is_bomb", False):
                continue
            t0 = time.perf_counter()
```

僅插入兩行（`if` + `continue`），其餘不動。

- [ ] **Step 2: 跑新測試確認通過**

Run:
```
cd C:/Users/rh.syu/Desktop/CAPI01_AD && python -m pytest tests/test_scratch_filter.py::test_filter_skips_bomb_tile -v
```

Expected output: `PASSED`

- [ ] **Step 3: 跑整個 scratch 測試集合確認無回歸**

Run:
```
cd C:/Users/rh.syu/Desktop/CAPI01_AD && python -m pytest tests/test_scratch_filter.py tests/test_scratch_integration.py tests/test_scratch_aggregate.py -v
```

Expected output: 全部 PASS。重點檢查：
- `test_filter_flips_high_score` — 仍 PASS（無 bomb tile）
- `test_filter_keeps_low_score` — 仍 PASS
- `test_filter_keeps_tile_in_anomaly_tiles_when_flipping` — 仍 PASS（C1 不變性）
- `test_filter_skips_bomb_tile` — 新增，PASS

- [ ] **Step 4: Commit**

```bash
cd C:/Users/rh.syu/Desktop/CAPI01_AD && git add scratch_filter.py tests/test_scratch_filter.py && git commit -m "$(cat <<'EOF'
fix(scratch): 已判定為炸彈的 tile 不再走 scratch 分類器

ScratchFilter.apply_to_image_result 迴圈頭部新增 is_bomb 跳過條件。
原本 bomb tile 會同時被標「炸彈」與「Scratch 救回 OK」（UI 誤導）
且每顆都浪費一次 DINOv2 推論。判定聚合本就先於 scratch_filtered
過濾 bomb，故此修正不影響 OK/NG 結論。
EOF
)"
```

---

## Task 3: 人工回歸驗證（瀏覽器）

**Files:** 無（僅檢查執行效果）

- [ ] **Step 1: 重跑一個已知有炸彈的 record**

在 `/debug` 或使用 `run_single_inference.py` 對包含炸彈的 panel 做推論，或重跑截圖 (2026-04-16) 中顯示 #82–#100 這筆 record。

- [ ] **Step 2: 開啟 `/record/<id>` 檢查顯示**

Expected: 所有標「炸彈」的 tile 不再同時顯示「Scratch 救回 OK (score=…)」badge；顯示內容僅為「炸彈」。

- [ ] **Step 3: 檢查 log**

確認 inference log 中 bomb tile 不再出現 `[scratch] … FLIP NG→OK` 或 `KEEP NG` 記錄。

- [ ] **Step 4: 若驗證通過，結束**

無額外 commit。若有發現異常則回到 Task 2 修正。

---

## Self-Review

**Spec coverage:**
- Spec「設計」段的一行 `if getattr(tile, "is_bomb", False): continue` → Task 2 Step 1 ✓
- Spec「測試」段的 `test_filter_skips_bomb_tile` → Task 1 Step 1 ✓
- Spec「作用域限制 (YAGNI)」：只 skip `is_bomb`，不處理 dust/exclude zone → 計劃內完全沒提到擴大範圍 ✓
- Spec「變更清單」：`scratch_filter.py` + `tests/test_scratch_filter.py`，無 DB/config/前端 → 計劃任務僅涉及這兩個檔案 ✓

**Placeholder scan:** 無 TBD/TODO/"appropriate error handling"/"similar to"。所有程式碼 blocks 完整列出。✓

**Type consistency:** `is_bomb`（TileInfo 欄位，`capi_inference.py:110` 定義）、`scratch_score` / `scratch_filtered` / `scratch_filter_count`（與現行 `test_scratch_filter.py` 測試一致）、`_MockClassifier` / `_fake_image_result_with_tiles`（兩者皆已在 `tests/test_scratch_filter.py` 定義，新測試可直接重用）。✓
