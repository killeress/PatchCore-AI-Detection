# Tile Self-Scoring — Design Spec

**Date:** 2026-05-07
**Status:** Draft

---

## Context

PatchCore 是 one-class anomaly detection：memory bank 只存正常樣本的 patch features，分數 = 測試 patch 到 memory bank 的最近鄰距離。**訓練集裡如果混入帶缺陷的 tile，那些 tile 的特徵會被 coreset subsampling 收進 memory bank**，導致日後同型缺陷在 inference 時也找得到很近的鄰居 → 分數偏低 → 漏判。

CAPI 既有路徑只能靠操作員肉眼從 step3 review 把可疑 tile 一張張看出來；單一 lighting+zone 動輒 100~300 tile，目視效率低、容易漏。

本 spec 提出兩個共用底層的功能：

1. **Bundle 自掃**：在 `/models` 的 bundle detail，用 bundle 自己的 `<lighting>-<zone>.pt` 對該 bundle 的訓練 tile 反向算分。**理論依據**：典型 OK tile 的 patch 多被 coreset 採樣或近鄰即在 bank → score 低；含污染的 tile 因 coreset 偏好「典型」representative，污染 patch 通常未被選中、跑分時找最近鄰會跨到距離較大的正常 patch → score 自然偏高。**排序前段就是優先目視確認的候選。**

2. **Step3 預篩**：訓練精靈 step3 review 階段，使用者可選一個既有 trained bundle 對當前 in-progress job 的 tile 做同樣的相對排序，作為人工 review 的視覺輔助。新機種若無同 machine 上一版，可選跨 machine 的 bundle，分數僅當「相對突兀程度」用。

---

## Scope

**In scope:**

- 新增 `tile_score_cache` 表（DB schema migration）
- 新增 `SubmodelScorer`：傳入 `(bundle_id, lighting, zone, tile_ids)`、跑分、寫 cache
- Bundle detail 頁 OK tiles 區新增「掃描可疑 tile」按鈕（單一 lighting+zone 為單位）
- Step3 review 頁頂部新增「預篩模型」下拉，選後切 lighting+zone tab 觸發 lazy compute
- Tile thumb 加分數 badge（依該批 top 5% / 5–20% 分位上色）
- 兩個入口共用：grid 依 score desc 排序、相同 badge 配色、相同 preview modal
- Cache 在重訓 submodel / 刪訓練資料 / 刪 bundle 時失效

**Out of scope:**

- 自動 reject（任何 threshold 都不自動砍 tile，全部由人決定）
- 批次 reject UI（marquee select / shift-click / "reject top N"）—— 維持既有 click→preview→Del 流程
- 在 step2 preprocess 完成時自動預先 scan（沒選 bundle 不該為了潛在使用浪費 GPU）
- 多 bundle ensemble 跑分後再聚合（一次一個 scoring bundle）
- Cache 過期時間 / TTL（除了明確 invalidation 觸發點外不過期）
- Score 校準 / 跨 bundle 可比性（明確標示「相對排序」用）

---

## Architecture

### 共用 scoring core

```
┌─────────────────────────────────────────────────────┐
│  capi_inference.SubmodelScorer (新)                  │
│  - input: bundle_id, lighting, zone, tile_ids[]      │
│  - 從 bundle_dir 載入 <lighting>-<zone>.pt           │
│    （沿用既有 inferencer cache）                       │
│  - 對每張 tile 跑 predict_tile() (raw score 即可)     │
│  - 增量寫入 tile_score_cache                         │
└─────────────────────────────────────────────────────┘
        ▲                           ▲
        │                           │
   ┌────┴──────┐            ┌──────┴────────┐
   │ 入口 A     │            │ 入口 B          │
   │ Bundle    │            │ Step3 review   │
   │ Detail    │            │ wizard         │
   │ 自掃       │            │ 用任一 bundle   │
   │           │            │ 預篩            │
   │ scoring_  │            │ scoring_       │
   │ bundle =  │            │ bundle = 使用者  │
   │ 該 bundle │            │ 選的            │
   └───────────┘            └───────────────┘
```

兩個入口的差別只在「scoring bundle 是誰」、「被掃的 tile 來自哪個 job_id」，背後都是同一個 scorer 跑同一個 cache 表。

### GPU 序列化

`SubmodelScorer` 走既有的 `_gpu_lock`，跟 production inference / 重訓 submodel 互相 serialize。同一台機器同時只允許一個 scoring job（server-level 單例旗標，沿用 `train_new_state` pattern）。

---

## DB Schema

新表，不動現有表：

```sql
CREATE TABLE IF NOT EXISTS tile_score_cache (
    tile_id           INTEGER NOT NULL,    -- training_tile_pool.id
    scoring_bundle_id INTEGER NOT NULL,    -- model_bundles.id
    score             REAL    NOT NULL,
    computed_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tile_id, scoring_bundle_id)
);
CREATE INDEX IF NOT EXISTS idx_score_cache_bundle
    ON tile_score_cache(scoring_bundle_id);
```

設計理由：

- key = `(tile_id, scoring_bundle_id)`：自掃（scoring_bundle = 該 bundle 自己）和跨 bundle 預篩用同一張表，不用分開
- 不動 `training_tile_pool` 主表，將來功能拔除乾淨
- `computed_at` 留作 debug / 手動清快取用

### Cache invalidation 觸發點

| 觸發點 | 清除條件 |
|---|---|
| 重訓 submodel 完成（覆蓋 `<lighting>-<zone>.pt`） | `WHERE scoring_bundle_id = ? AND tile_id IN (SELECT id FROM training_tile_pool WHERE lighting=? AND zone=?)`——清掉「這份重訓後的 .pt」對「所有 job 該 lighting+zone tile」算過的 row（不只該 bundle 自己的訓練 tile，也包含 step3 預篩用過該 bundle 的舊分） |
| `delete_training_data(bundle_id)` | `WHERE tile_id IN (被刪的 tile_id 清單)` |
| 刪整個 bundle | `WHERE scoring_bundle_id = ?` |

不需要清的場景：
- 新 tile insert（cache 本來就沒它）
- step3 切換不同 bundle（key 不同，並存）
- bundle is_active 切換（.pt 沒變）

---

## Backend Components

### `capi_inference.SubmodelScorer`

```python
class SubmodelScorer:
    """單純跑分寫 cache，不做 reject 決定。"""
    def __init__(self, gpu_lock, db, log_fn):
        ...

    def score_tiles(
        self,
        scoring_bundle_id: int,
        lighting: str,
        zone: str,
        tile_ids: list[int],
        cancel_event: threading.Event,
        progress_cb: Callable[[int, int], None],  # (done, total)
    ) -> ScoreScanResult:
        # 1. 取 scoring bundle 的 bundle_dir
        # 2. 載入 <lighting>-<zone>.pt 透過既有 inferencer cache
        # 3. for tile_id in tile_ids:
        #    if cancel_event.is_set(): break
        #    讀 source_path 圖片
        #    predict_tile() 取 pred_score（用 production 預設 config，
        #      不傳 patchcore_overrides / edge_margin_override，
        #      讓 scan 的分數和線上 inference 同口徑）
        #    寫一筆 tile_score_cache (UPSERT)
        #    progress_cb(done, total)
        # 4. 回傳 {scanned, skipped, cancelled}
```

讀 tile 圖片用 `training_tile_pool.source_path`（preprocess 階段已存的 512×512 raw tile png）。讀失敗的 tile 跳過、log warn、不寫 cache。

### `capi_model_registry` 補強

- 新增 `invalidate_score_cache(scoring_bundle_id, lighting=None, zone=None)` helper
- `retrain_submodel` 收尾呼叫上面 helper（指定 lighting+zone）
- `delete_training_data` 收尾刪 `tile_score_cache WHERE tile_id IN (...)`
- `delete_bundle`（如有）收尾刪 `tile_score_cache WHERE scoring_bundle_id = ?`

### `capi_database` 補強

- `insert_score_cache(rows)`：UPSERT 批次寫
- `get_score_cache(scoring_bundle_id, tile_ids)`：批次查、回 `{tile_id: score}`
- `delete_score_cache(scoring_bundle_id=None, tile_ids=None)`：彈性清除

---

## API

### Bundle detail 自掃

```
POST /api/models/<bundle_id>/scan_self_score
body: {"lighting": "WGF50500", "zone": "inner"}
→ 啟動後台 job
→ 200 {"job_id": "scan_<uuid>", "total": 180}
→ 409 if 已有 scan job 在跑

GET  /api/models/<bundle_id>/scan_status?lighting=WGF50500&zone=inner
→ 200 {"state": "running"|"done"|"failed"|"cancelled"|"empty",
       "done": 45, "total": 180, "error": null}

POST /api/models/<bundle_id>/scan_cancel
body: {"lighting": "WGF50500", "zone": "inner"}
→ 200 {"cancelled": true}
```

### Step3 預篩

```
GET  /api/train/new/eligible_scoring_bundles
→ 200 {"bundles": [
     {"id": 12, "machine_id": "GN160JCEL250S",
      "trained_at": "2026-04-15T14:10:00", "is_active": true,
      "label": "GN160JCEL250S / 20260415_141000 ●active"},
     ...
   ]}

POST /api/train/new/scan_prefilter_score
body: {"job_id": "...", "scoring_bundle_id": 12,
       "lighting": "WGF50500", "zone": "inner"}
→ 200 {"job_id": "scan_<uuid>", "total": 180, "cached_hit": false}
→ 若全部都已在 cache: 200 {"cached_hit": true, "scores": {tile_id: score, ...}}

GET  /api/train/new/scan_status?job_id=...
（同上 scan_status 結構）
```

### 既有 training_tiles API 擴充

`/api/models/<bundle_id>/training_tiles` 與 step3 對應 endpoint 都加 query：

```
?score_from_bundle=<bundle_id>&sort_by=score_desc
```

回 tile list 時 join `tile_score_cache`：

```json
{"tiles": [
  {"id": 1234, "thumb_url": "...", "decision": "accept",
   "score": 0.842, "score_quartile": "top5"},
  ...
]}
```

`score_quartile` 在後端依該批分數計算（top 5% / top 5–20% / rest），前端不算分位。

---

## Frontend

### Bundle detail 自掃

`templates/models.html` OK tiles tab 既有的 lighting / zone 下拉旁加：

```html
<button id="scanSelfScoreBtn">🔍 掃描可疑 tile</button>
```

點按鈕：
1. POST `/scan_self_score` → 拿 job_id
2. 按鈕變「掃描中... 23/180」+ small spinner，輪詢 `/scan_status`
3. 跑完 reload tile grid（帶 `?score_from_bundle=<self>&sort_by=score_desc`）
4. Grid 重排為 score desc，每張 thumb 加 badge

如該 (bundle, lighting, zone) 已有 cache，按鈕 label 變「🔄 重新掃描」，點下去 confirm 後清 cache 再跑。

### Step3 預篩

`templates/train_new/step3_review.html` 頂部新增控制列：

```html
<div class="prefilter-bar">
  預篩模型:
  <select id="prefilterBundleSelect">
    <option value="">不啟用</option>
    {% for b in eligible_bundles %}
      <option value="{{ b.id }}" data-machine="{{ b.machine_id }}">
        {{ b.label }}
      </option>
    {% endfor %}
  </select>
  <span id="prefilterStatus"></span>  <!-- 跨 machine 提示 banner 等 -->
</div>
```

切 lighting / zone tab 時：
- 沒選 bundle → 載 tile 一切照舊
- 有選 → 先查該 (job, lighting, zone, scoring_bundle) cache
  - 全 hit → 直接拿分數 sort + render badge
  - miss / 部分 miss → POST `scan_prefilter_score` → 進度條 → 完成 reload

如選了非該 machine 的 bundle，grid 上方顯示 `⚠ 用跨 machine 的模型預篩，分數只能當相對排序參考`。

### Tile thumb badge 樣式

CSS：

```css
.tile-thumb .score-badge {
  position: absolute; top: 4px; right: 4px;
  padding: 2px 6px; border-radius: 8px;
  font-size: 0.72rem; font-family: monospace;
  background: #45475a; color: #cdd6f4;  /* 灰，預設 */
}
.tile-thumb[data-score-quartile="top5"]   .score-badge { background: #f38ba8; color: #1e1e2e; }
.tile-thumb[data-score-quartile="top20"]  .score-badge { background: #fab387; color: #1e1e2e; }
```

### 既有 reject 流程不動

維持 click → preview modal → Del / A。Badge 只是排序輔助標。

---

## Error Handling

| 情境 | 行為 |
|---|---|
| GPU 鎖被 production inference / 重訓佔用 | 序列化等待，UI 顯示「等待 GPU 中」 |
| 同一個 bundle 已有 scan 在跑 | API 409，前端按鈕禁用 |
| 使用者按 cancel / 關瀏覽器 | 後台跑完當下 tile 後停，已寫 cache 保留 |
| `<lighting>-<zone>.pt` 不存在 / 壞掉 | job fail，狀態 `failed` + error message，不寫 cache |
| Tile source_path 失效 | 跳過該 tile、log warn、其他 tile 照常跑 |
| 切 bundle 下拉時舊 scan 仍在跑 | 不取消舊 job（讓它跑完寫 cache 留著），新 bundle 排隊 |
| 空 tile pool | API 直接回 `state: empty`，不啟 job |
| Server restart 中途 | 已寫 row 持久（WAL），下次點按鈕只算剩下沒算的 |
| 下拉 bundle 列表為空 | step3 下拉 disabled 顯示「目前無可用模型」 |

---

## Testing Strategy

依專案慣例，不主動跑 pytest，但提供下列 sanity check 路徑供需要時手動驗證：

**Unit-level（standalone python -c 快速確認）：**

- `tile_score_cache` schema migration 在新舊 DB 都跑得起來
- `SubmodelScorer.score_tiles()` 對 mock inferencer 跑 5 張 tile，cache 寫入 5 筆、score desc 排序正確
- `invalidate_score_cache()` 三條觸發路徑各自只清掉預期的 row

**Integration-level（真環境跑）：**

- bundle detail 對 GN160JCEL250S 該 bundle 跑 WGF50500-inner 自掃，確認：
  - top 5% 紅色 badge 對應的 tile 視覺上明顯有缺陷 / 異常
  - reject 那幾顆 → 重訓 → 同位置缺陷 inference score 上升
- step3 用上一版 bundle 預篩新訓練 job，確認：
  - 跨 machine 提示 banner 出現條件正確
  - cache hit 的後續切換是秒開
  - 進度條跑完後 grid 確實依 score desc 排序

---

## Open Questions

無——主要設計選擇都已在 brainstorming 階段確認：

- 結果呈現：grid 重排 + badge（不做 modal、不做 top-N 截斷）
- Cache 策略：on-demand 觸發 + DB 持久化 cache
- Step3 模型來源：下拉列所有 trained bundle，使用者自選
- 跨 machine 用法：開放使用，但 UI 加相對排序 caveat
- 不做批次 reject

---

## Implementation Order

1. DB migration（`tile_score_cache` 表）+ database CRUD
2. `SubmodelScorer` 核心 + 單獨 sanity check
3. Cache invalidation hooks 加進 `retrain_submodel` / `delete_training_data` / `delete_bundle`
4. Bundle detail 自掃 API + 前端按鈕 / 進度條 / badge
5. Step3 預篩 API + 前端下拉 / lazy compute / badge
6. 整合測：跑一輪自掃 → reject → 重訓 → 確認 score 上升

---
