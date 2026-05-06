# Train New Wizard：固定 3 片 + Edge 外推取樣

**狀態：** Draft
**日期：** 2026-05-06
**範圍：** 訓練 wizard Step 1 panel 數固定為 3、`_generate_tiles` 加入 edge 外推取樣以涵蓋「ROI 中心剛好在 panel 邊緣」的推論情境。

## 背景

現行訓練 wizard：
- Step 1 勾 5 片，前 3 片收 inner+edge，後 2 片只收 edge（補 edge 樣本）
- `capi_preprocess._generate_tiles` 從 OTSU bbox 內走 grid，最外圈 tile 緊貼 panel 邊（中心離 panel 邊緣 256px，tile 完全在 panel 內）

問題：
- v2 推論已能切「以 AOI 機檢座標為中心的 512×512 tile」（`capi_inference._build_aoi_centered_tiles_v2`），OOB 區域用 `cv2.copyMakeBorder(BORDER_CONSTANT, value=0)` 黑邊填補
- 但訓練資料**沒有**「panel 邊緣 + 自然黑邊背景」的 tile
- 結果：當 AOI 座標剛好落在 panel 邊緣，推論 tile 50% 是 panel、50% 是黑邊 → PatchCore 把整片黑邊當 anomaly，產生大量過殺（Image #16 即為此情境）

## 目標

1. 訓練資料補上「panel 邊緣 + 自然黑邊」的 tile
2. 簡化 wizard：3 片，全部 inner+edge，砍掉 `inner_panels` 進階參數

## 非目標

- 不改 v2 推論的 centered-tile 切法
- 不改 NG zone 分類邏輯
- 不動既有 5-model（CAPI 3F）legacy 流程

## 設計

### 1. `capi_preprocess.py`：Edge 外推取樣

`PreprocessConfig` 加新欄位：
```python
outer_edge_extend: int = 256
```

語意：edge tile 中心可以離 panel bbox 邊往外推到 `outer_edge_extend` px。預設 256（= half tile_size），剛好讓「中心位於 panel 邊」的 tile 進入訓練集（50% panel + 50% 黑邊）。

`_generate_tiles` 在原 bbox grid 之外，每邊多加一排 tile：

```
push_top    = min(outer_edge_extend, max(0, y1))
push_bottom = min(outer_edge_extend, max(0, img_h - y2))
push_left   = min(outer_edge_extend, max(0, x1))
push_right  = min(outer_edge_extend, max(0, img_w - x2))
```

關鍵點：**push 距離夾到 image 邊界**。如果某側 panel 邊離 image 邊不夠 256（例：panel 上緣 y1=100，push_top=100），就把該側推出距離縮到 image 邊剛好為止，使 tile 始終落在 image 內、用真實的自然黑邊像素。

如果某側 push=0（panel 緊貼 image 邊），跳過該側外推。

外推 tile 位置：
- Top row    : `ty = y1 - push_top`           （xs 包含左右外推 col 對應的 tx）
- Bottom row : `ty = y2 - tile_size + push_bottom`
- Left col   : `tx = x1 - push_left`           （ys 不含 top/bottom 行，避免角落重複）
- Right col  : `tx = x2 - tile_size + push_right`

四個角落的外推 tile 同時 push x 和 y（含在 top/bottom 行內）。

對外推 tile：
- **強制 zone="edge"**：不走 `classify_tile_zone` 的 coverage 判斷（角落 tile 可能 coverage < 0.3 而被分類為 outside 跳掉，外推 tile 應一律當 edge）
- coverage 仍照算（用於診斷 / DB 紀錄）
- 因為 push 已被夾到 image 內，**不需要 `cv2.copyMakeBorder`**，直接用 `img[ty:ty+ts, tx:tx+ts].copy()` 即可

### 2. `capi_train_new.py`：簡化參數

- 移除 `TrainingConfig.inner_panels` 欄位
- 移除 `USER_TRAINABLE_PARAM_SPECS["inner_panels"]`
- `preprocess_panels_to_pool`：拿掉 `inner_allowed = idx <= cfg.inner_panels` 邏輯，所有 panel 一律收 inner+edge
- log 簡化為 `[idx/total] panel <name>`（移除 `(inner+edge|edge only)` 後綴）
- `apply_user_training_params` 不需動，因為 unknown key 已 raise

### 3. `capi_web.py`：驗證層

- `_handle_train_new_start`：`len(clean_panel_paths) != 5` → `!= 3`
- 錯誤訊息：`"panel_paths must contain exactly 3 panels"`

### 4. `templates/train_new/step1_select.html`：UI

- 文案：「勾選 5 片 panel」→「勾選 3 片 panel」
- 移除「第 1-3 片提供 inner+edge tile，第 4-5 片只取 edge tile」說明（改成單句「3 片皆收 inner+edge tile」）
- `INNER_PANELS = 3` 常數移除（用不到了）
- `_roleColor` 移除（單一綠色）
- `_selected.size >= 5` → `>= 3`
- sticky bar：「已選 0/5 片」→「已選 0/3 片」
- 進階設定移除 `inner_panels` 欄位
- `TRAIN_PARAM_KEYS` 移除 `inner_panels`

### 5. 受影響的 templates / 顯示層 / 文件

- `templates/models.html` line 285：`inner_panels` 欄位顯示 → 改為條件顯示（舊 bundle 仍有此欄，新 bundle 沒有）
- `templates/train_new/step5_done.html` line 76：同上
- `templates/train_new/step2_progress.html` line 6 & 372：log regex `(inner\+edge|edge only)` 改為單純 panel name 比對
- `capi_web.py` line 4995：`_handle_train_new_start` docstring body example 移除 `"inner_panels": 3`
- `capi_web.py:2047` 的 `f"{img_prefix} (inner+edge)"` **不動**（它指 model 架構名稱「lighting × inner+edge」，與 panel 取樣 inner_panels 不同）
- `docs/patchcore_training_architecture.zh-TW.md` line 84：「選滿 5 片」→「選滿 3 片」
- `scripts/build_deploy_zip.py` line 120：release notes 中「勾選 5 片」→「勾選 3 片」

### 6. Tests

- **修改：**
  - `tests/test_capi_train_new_preprocess.py::test_preprocess_panels_to_pool_skips_inner_after_inner_panels`：邏輯已不存在 → 改為 `test_preprocess_panels_to_pool_all_panels_inner_and_edge`，驗證 3 片都有 inner + edge
  - `tests/test_capi_web_train_new.py`：把 `panel_paths = [f"/p{i}" for i in range(5)]` (line 235, 253, 338, 383) 全部改為 `range(3)`；`{"inner_panels": 4}` 等 assertion 移除
  - `tests/test_capi_train_new_training.py` (line 14, 17, 27, 32)：snapshot tuple 移除 `inner_panels`，覆寫測試只 assert `batch_size/coreset_ratio/max_epochs`
  - `tests/test_capi_database_train.py:192`：移除 `inner_panels` 欄位
- **新增：**
  - `tests/test_capi_preprocess_outer_extend.py`：
    - `test_outer_edge_extend_adds_extension_tiles`：在中央 panel 圖上跑 preprocess，驗證 zone=edge 的 tile 數量比未開外推時多（含 4 邊 + 4 角）
    - `test_outer_edge_extend_clamps_to_image_boundary`：panel 緊貼 image 上邊（y1=50），驗證 push_top=50（不是 256），tile 仍完全在 image 內
    - `test_outer_edge_extend_skips_when_no_margin`：panel y1=0 → 不產生 top 外推 tile

## 取捨

| 選項 | 取 | 不取 | 理由 |
|------|----|------|------|
| `outer_edge_extend` 是 config 還是常數 | config（預設 256） | 常數 | 方便 test 與未來調 |
| 外推 tile push=0 時是否強制建 1 個 | 跳過 | 強建 | push=0 等於原本最外圈 tile，重複 |
| OOB 像素處理 | 夾 push 距離 → 不需 padding | `copyMakeBorder` | 訓練資料用真實像素更可靠，避免 zero-pad 偽資料污染 normal memory bank |
| `inner_panels` 處理 | 完全移除 | 保留並改 max=3 | 統一流程不留條件分支（user feedback memory） |

## 風險

- **舊 bundle 顯示**：models.html / step5_done.html 顯示 `inner_panels` 的位置在新 bundle 是 `None`，需改成 conditional render（舊 bundle 仍能看到該欄）
- **首次訓練 tile 數變化**：3 片 × (inner+edge+外推) 的總 tile 數 vs 舊 5 片 × 部分 inner+edge — 預期相當（待 step5 metrics 驗）
- **角落外推 tile**：coverage 約 0.25，過去會被歸 outside 跳掉，現在強制 edge。如果 corner tile 真的全黑（OTSU bbox 比實際 panel 大很多）可能成為高 score sample，需用 step3 review UI 人工剔除

## 驗收

- Web wizard：Step 1 顯示「勾選 3 片」，第 4 片無法勾選；不出現 inner_panels 進階欄位
- Step 3 預覽：panel polygon 外圍多一圈橘色 edge tile（4 邊 + 4 角）
- Step 5 metrics：所有 10 個 unit 都正常出 model.pt + threshold + auroc
- 訓練後手動測：拿一張原本會在 panel 邊緣過殺的圖跑 v2 推論，比對 threshold-過 NG 數應降低（人工驗）
