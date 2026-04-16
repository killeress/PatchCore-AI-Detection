---
title: ScratchFilter 跳過已判定為炸彈的 tile
date: 2026-04-16
status: draft
---

# ScratchFilter 跳過已判定為炸彈的 tile

## 問題

推論後處理流程：

1. `capi_inference.py` 先跑 bomb 比對，命中的 tile 設 `tile.is_bomb=True` + `bomb_defect_code`
2. 接著呼叫 `ScratchFilter.apply_to_image_result(result)` 對**所有** `anomaly_tiles` 跑 DINOv2 分類器

第 2 步沒有考慮 `tile.is_bomb`，導致：

- **誤導性 UI / log**：同一 tile 同時被標成「炸彈」與「Scratch 救回 OK」（見 2026-04-16 截圖 #94–#100）
- **浪費算力**：每個 bomb tile 都跑一次 DINOv2 classifier，100 顆炸彈就是 100 次無謂呼叫
- **資料雜訊**：DB `tile_results.scratch_score` 寫入不具意義的分數，干擾 over-review 統計

判定正確性**不受影響**——`capi_server.py` 判定聚合（`build_ng_details` 第 452 行、`results_to_db_data` 第 551 行）已經先於 `scratch_filtered` 過濾 `is_bomb`，所以 bomb 分類結果不會影響 OK/NG 結論。

## 設計

在 `scratch_filter.py` `ScratchFilter.apply_to_image_result` 迴圈頭部加入 bomb 跳過條件：

```python
for entry in image_result.anomaly_tiles:
    tile = entry[0]
    if getattr(tile, "is_bomb", False):
        continue   # 已判定為炸彈，不走 scratch 分類器
    ...
```

### 為什麼改在 ScratchFilter 內部

- ScratchFilter 自己的契約：「已被分類的 tile 不該進我的流程」
- 呼叫端 (`capi_inference.py:4153`) 不需改動，loop 結構單純
- bomb tile 保留 `scratch_score=0.0`、`scratch_filtered=False` 預設值——等同「classifier 未啟用或未跑」的 schema 語意，向後相容

### 副作用確認

| 位置 | 現行行為 | 改動後 |
|------|----------|--------|
| `capi_server.py:452` 判定聚合 | 先 skip bomb，再 skip scratch_filtered | 不變（bomb 本就先 skip） |
| `capi_server.py:551` DB NG 聚合 | 同上 | 不變 |
| `record_detail.html` badge | `scratch_filtered=True` → 顯示「Scratch 救回 OK」 | bomb 不再設此旗標，改回只顯示「炸彈」 |
| DB `tile_results.scratch_score` | bomb tile 寫入實際分數 | bomb tile 寫入 0.0（等同未啟用） |
| DINOv2 呼叫次數 | 每 bomb tile 1 次 | 0 次 |

### 作用域限制（YAGNI）

本次**只**跳過 `is_bomb`。其他可能的 skip 目標（`is_suspected_dust_or_scratch`、`is_in_exclude_zone`、`is_aoi_coord_below_threshold`）**不在本次範圍**——理由：

- 使用者回報只針對炸彈
- 灰塵/排除區域本來就是 over-review 常客，scratch classifier 的目標就是協助把這類 tile 救回 OK，跑 classifier 反而有幫助
- AOI-below-threshold tile 可能是 false-OK 樣本，留著跑 classifier 可作未來分析

## 測試

在 `tests/test_scratch_filter.py` 新增單元測試：

```python
def test_filter_skips_bomb_tile():
    """Bomb tile 不應進入 classifier；scratch 欄位保持預設值。"""
    clf = _MockClassifier(fixed_score=0.99, conformal_threshold=0.7)
    sf = ScratchFilter(clf, safety_multiplier=1.0)
    ir = _fake_image_result_with_tiles(2)
    ir.tiles[0].is_bomb = True
    ir.anomaly_tiles[0][0].is_bomb = True   # same object, 保險起見

    sf.apply_to_image_result(ir)

    # Bomb tile: 未跑 classifier
    assert ir.tiles[0].scratch_score == 0.0
    assert ir.tiles[0].scratch_filtered is False
    # Non-bomb tile: 正常跑，高分被翻回 OK
    assert ir.tiles[1].scratch_score == pytest.approx(0.99)
    assert ir.tiles[1].scratch_filtered is True
    # scratch_filter_count 只計非 bomb tile
    assert ir.scratch_filter_count == 1
```

## 變更清單

- `scratch_filter.py`：1 行新增（`if getattr(tile, "is_bomb", False): continue`）
- `tests/test_scratch_filter.py`：新增 `test_filter_skips_bomb_tile`
- 無需 DB migration、無需 config 變更、無需前端改動
