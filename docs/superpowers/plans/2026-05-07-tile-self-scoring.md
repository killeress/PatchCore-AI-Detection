# Tile Self-Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 PatchCore bundle 對 tile pool 反向算分，bundle detail 提供「自掃可疑 tile」、step3 review 提供「用既有 bundle 預篩」兩個入口，加速人工 reject 流程。

**Architecture:** 共用 `SubmodelScorer` 跑分 → 寫 `tile_score_cache` 表（key = `(tile_id, scoring_bundle_id)`）。前端依 score desc 排序 + 顯示 badge。重訓 / 刪訓練資料 / 刪 bundle 時自動失效對應 cache。

**Tech Stack:** Python 3.12 / SQLite (WAL) / anomalib `TorchInferencer` / threading（既有 `_gpu_lock` + `_scan_state` 單例旗標）/ Vanilla JS + Jinja2。

**Spec:** `docs/superpowers/specs/2026-05-07-tile-self-scoring-design.md`

**Testing convention:** 此專案不主動跑 pytest（user feedback memory），plan 內仍寫 pytest 測試以支援 TDD，但「執行測試」這步驟由工程師手動進行；驗證亦可改用 `python -c "..."` 簡易 sanity check。

---

## File Structure

| File | Role |
|---|---|
| `capi_database.py` | 加 `tile_score_cache` schema 與 CRUD |
| `capi_inference.py` | 加 `SubmodelScorer` class（純跑分寫 cache，不做決策） |
| `capi_model_registry.py` | 加 `invalidate_score_cache()`，hook 進 `delete_training_data` / `delete_bundle` |
| `capi_web.py` | 4 個新 endpoint：scan_self_score / scan_prefilter_score / scan_status / scan_cancel；既有 retrain_submodel + training_tiles + train_new_tiles 擴充 |
| `templates/models.html` | bundle detail OK tiles 區加掃描按鈕、進度顯示、score badge |
| `templates/train_new/step3_review.html` | step3 頂部加 prefilter dropdown、lazy compute、score badge |
| `tests/test_score_cache_db.py` | DB schema + CRUD 測試 |
| `tests/test_submodel_scorer.py` | Scorer 單元測試 |
| `tests/test_score_cache_invalidation.py` | 三條失效路徑測試 |

---

## Task 1: tile_score_cache DB schema + CRUD

**Files:**
- Modify: `capi_database.py:285-296`（schema 區段）
- Modify: `capi_database.py:2520-2528`（CRUD 接 `cleanup_tile_pool` 後）
- Test: `tests/test_score_cache_db.py`（新檔）

- [ ] **Step 1.1：寫失敗測試**

`tests/test_score_cache_db.py`:

```python
"""tile_score_cache 表 schema + CRUD 測試。"""
import pytest
from pathlib import Path

from capi_database import CAPIDatabase


@pytest.fixture
def db(tmp_path: Path) -> CAPIDatabase:
    return CAPIDatabase(str(tmp_path / "test.db"))


def test_tile_score_cache_schema_exists(db):
    conn = db._get_conn()
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tile_score_cache'"
        )
        assert cur.fetchone() is not None, "tile_score_cache 表應該存在"
    finally:
        conn.close()


def test_insert_and_get_score_cache(db):
    db.insert_score_cache([
        {"tile_id": 1, "scoring_bundle_id": 10, "score": 0.42},
        {"tile_id": 2, "scoring_bundle_id": 10, "score": 0.88},
        {"tile_id": 3, "scoring_bundle_id": 11, "score": 0.31},
    ])
    got = db.get_score_cache(scoring_bundle_id=10, tile_ids=[1, 2, 3])
    assert got == {1: 0.42, 2: 0.88}  # tile 3 是 bundle 11，不在結果


def test_insert_score_cache_upsert(db):
    db.insert_score_cache([{"tile_id": 1, "scoring_bundle_id": 10, "score": 0.5}])
    db.insert_score_cache([{"tile_id": 1, "scoring_bundle_id": 10, "score": 0.9}])
    got = db.get_score_cache(scoring_bundle_id=10, tile_ids=[1])
    assert got == {1: 0.9}, "同 key 重複插入應 upsert 為新分數"


def test_delete_score_cache_by_bundle(db):
    db.insert_score_cache([
        {"tile_id": 1, "scoring_bundle_id": 10, "score": 0.5},
        {"tile_id": 2, "scoring_bundle_id": 11, "score": 0.5},
    ])
    db.delete_score_cache(scoring_bundle_id=10)
    assert db.get_score_cache(10, [1, 2]) == {}
    assert db.get_score_cache(11, [1, 2]) == {2: 0.5}


def test_delete_score_cache_by_tile_ids(db):
    db.insert_score_cache([
        {"tile_id": 1, "scoring_bundle_id": 10, "score": 0.5},
        {"tile_id": 2, "scoring_bundle_id": 10, "score": 0.5},
        {"tile_id": 1, "scoring_bundle_id": 11, "score": 0.5},
    ])
    db.delete_score_cache(tile_ids=[1])
    assert db.get_score_cache(10, [1, 2]) == {2: 0.5}
    assert db.get_score_cache(11, [1, 2]) == {}


def test_get_score_cache_empty_tile_ids_returns_empty(db):
    assert db.get_score_cache(scoring_bundle_id=10, tile_ids=[]) == {}
```

- [ ] **Step 1.2：跑測試確認 fail**

```bash
pytest tests/test_score_cache_db.py -v
```
Expected: 全 FAIL（表不存在 / 方法不存在）

- [ ] **Step 1.3：加 schema**

在 `capi_database.py` 既有 schema 區段（`training_tile_pool` 之後，line 296 `CREATE INDEX idx_tile_pool_job` 之後）加：

```python
                CREATE TABLE IF NOT EXISTS tile_score_cache (
                    tile_id           INTEGER NOT NULL,
                    scoring_bundle_id INTEGER NOT NULL,
                    score             REAL    NOT NULL,
                    computed_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (tile_id, scoring_bundle_id)
                );
                CREATE INDEX IF NOT EXISTS idx_score_cache_bundle
                    ON tile_score_cache(scoring_bundle_id);
```

- [ ] **Step 1.4：加 CRUD 方法**

在 `capi_database.py` `cleanup_tile_pool`（line 2527）之後加：

```python
    # ------------------------------------------------------------------
    # tile_score_cache CRUD
    # ------------------------------------------------------------------

    def insert_score_cache(self, rows: list) -> None:
        """批次 UPSERT (tile_id, scoring_bundle_id) → score。空清單 no-op。"""
        if not rows:
            return
        conn = self._get_conn()
        try:
            conn.executemany(
                """INSERT INTO tile_score_cache
                       (tile_id, scoring_bundle_id, score, computed_at)
                   VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(tile_id, scoring_bundle_id)
                   DO UPDATE SET score = excluded.score,
                                 computed_at = CURRENT_TIMESTAMP""",
                [(r["tile_id"], r["scoring_bundle_id"], r["score"]) for r in rows],
            )
            conn.commit()
        finally:
            conn.close()

    def get_score_cache(self, scoring_bundle_id: int, tile_ids: list) -> dict:
        """回傳 {tile_id: score}，只包含 cache 中存在的 row。空 tile_ids → 空 dict。"""
        if not tile_ids:
            return {}
        placeholders = ",".join("?" * len(tile_ids))
        conn = self._get_conn()
        try:
            cur = conn.execute(
                f"""SELECT tile_id, score FROM tile_score_cache
                    WHERE scoring_bundle_id = ? AND tile_id IN ({placeholders})""",
                (scoring_bundle_id, *tile_ids),
            )
            return {row[0]: row[1] for row in cur.fetchall()}
        finally:
            conn.close()

    def delete_score_cache(self, scoring_bundle_id: int = None,
                           tile_ids: list = None,
                           lighting: str = None, zone: str = None) -> int:
        """彈性刪除 cache。回傳刪除筆數。

        - scoring_bundle_id only: 清該 bundle 全部
        - tile_ids only: 清這些 tile 在所有 bundle 的分
        - scoring_bundle_id + lighting + zone: 清該 bundle 對該 lighting+zone tile 的分
          （join training_tile_pool 過濾）
        """
        conn = self._get_conn()
        try:
            if scoring_bundle_id is not None and lighting is not None and zone is not None:
                cur = conn.execute(
                    """DELETE FROM tile_score_cache
                       WHERE scoring_bundle_id = ?
                         AND tile_id IN (
                           SELECT id FROM training_tile_pool
                           WHERE lighting = ? AND zone = ?
                         )""",
                    (scoring_bundle_id, lighting, zone),
                )
            elif tile_ids:
                placeholders = ",".join("?" * len(tile_ids))
                cur = conn.execute(
                    f"DELETE FROM tile_score_cache WHERE tile_id IN ({placeholders})",
                    tuple(tile_ids),
                )
            elif scoring_bundle_id is not None:
                cur = conn.execute(
                    "DELETE FROM tile_score_cache WHERE scoring_bundle_id = ?",
                    (scoring_bundle_id,),
                )
            else:
                return 0
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()
```

- [ ] **Step 1.5：跑測試確認 pass**

```bash
pytest tests/test_score_cache_db.py -v
```
Expected: 6/6 PASS

- [ ] **Step 1.6：commit**

```bash
git add capi_database.py tests/test_score_cache_db.py
git commit -m "feat(db): tile_score_cache 表 schema 與 CRUD"
```

---

## Task 2: SubmodelScorer 核心

**Files:**
- Modify: `capi_inference.py`（檔末尾，CAPIInferencer class 之後加新 class）
- Test: `tests/test_submodel_scorer.py`（新檔）

- [ ] **Step 2.1：寫失敗測試**

`tests/test_submodel_scorer.py`:

```python
"""SubmodelScorer 單元測試（用 mock inferencer，不跑真 GPU）。"""
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from capi_database import CAPIDatabase


@pytest.fixture
def fake_tile_pool(tmp_path: Path):
    """產 5 張假 tile png + 對應 training_tile_pool row。"""
    db = CAPIDatabase(str(tmp_path / "t.db"))
    job_id = "j1"
    rows = []
    for i in range(5):
        p = tmp_path / f"tile_{i}.png"
        img = np.full((512, 512), 128 + i * 20, dtype=np.uint8)
        cv2.imwrite(str(p), img)
        rows.append({
            "lighting": "WGF50500", "zone": "inner", "source": "ok",
            "source_path": str(p), "thumb_path": str(p),
        })
    ids = db.insert_tile_pool(job_id, rows)
    return db, job_id, ids


def test_scorer_writes_cache_for_all_tiles(fake_tile_pool, tmp_path):
    from capi_inference import SubmodelScorer

    db, job_id, tile_ids = fake_tile_pool
    bundle_dir = tmp_path / "bundle"; bundle_dir.mkdir()
    (bundle_dir / "WGF50500-inner.pt").write_bytes(b"FAKE")

    fake_scores = [0.10, 0.85, 0.42, 0.99, 0.21]

    def fake_predict_tile(tile, **kwargs):
        idx = fake_predict_tile.call_count
        fake_predict_tile.call_count += 1
        return fake_scores[idx], np.zeros((28, 28), dtype=np.float32)
    fake_predict_tile.call_count = 0

    scorer = SubmodelScorer(
        gpu_lock=threading.Lock(),
        db=db,
        log_fn=lambda msg: None,
    )

    cancel_event = threading.Event()
    progress = []

    with patch.object(scorer, "_load_inferencer_for_pt", return_value=MagicMock()), \
         patch.object(scorer, "_score_one_tile", side_effect=fake_scores):
        result = scorer.score_tiles(
            scoring_bundle_id=99, bundle_dir=bundle_dir,
            lighting="WGF50500", zone="inner",
            tile_pool_job_id=job_id, tile_ids=tile_ids,
            cancel_event=cancel_event,
            progress_cb=lambda d, t: progress.append((d, t)),
        )

    assert result["scanned"] == 5
    assert result["cancelled"] is False
    cache = db.get_score_cache(99, tile_ids)
    assert len(cache) == 5
    assert progress[-1] == (5, 5)


def test_scorer_respects_cancel(fake_tile_pool, tmp_path):
    from capi_inference import SubmodelScorer

    db, job_id, tile_ids = fake_tile_pool
    bundle_dir = tmp_path / "bundle"; bundle_dir.mkdir()
    (bundle_dir / "WGF50500-inner.pt").write_bytes(b"FAKE")

    cancel_event = threading.Event()

    def cancel_after_two(score_):
        if cancel_after_two.calls == 2:
            cancel_event.set()
        cancel_after_two.calls += 1
        return 0.5
    cancel_after_two.calls = 0

    scorer = SubmodelScorer(
        gpu_lock=threading.Lock(), db=db, log_fn=lambda m: None,
    )
    with patch.object(scorer, "_load_inferencer_for_pt", return_value=MagicMock()), \
         patch.object(scorer, "_score_one_tile", side_effect=cancel_after_two):
        result = scorer.score_tiles(
            scoring_bundle_id=99, bundle_dir=bundle_dir,
            lighting="WGF50500", zone="inner",
            tile_pool_job_id=job_id, tile_ids=tile_ids,
            cancel_event=cancel_event, progress_cb=lambda d, t: None,
        )
    assert result["cancelled"] is True
    assert result["scanned"] < 5


def test_scorer_skips_missing_pt(fake_tile_pool, tmp_path):
    from capi_inference import SubmodelScorer

    db, job_id, tile_ids = fake_tile_pool
    bundle_dir = tmp_path / "bundle"; bundle_dir.mkdir()
    # 故意不放 .pt

    scorer = SubmodelScorer(
        gpu_lock=threading.Lock(), db=db, log_fn=lambda m: None,
    )
    with pytest.raises(FileNotFoundError):
        scorer.score_tiles(
            scoring_bundle_id=99, bundle_dir=bundle_dir,
            lighting="WGF50500", zone="inner",
            tile_pool_job_id=job_id, tile_ids=tile_ids,
            cancel_event=threading.Event(), progress_cb=lambda d, t: None,
        )


def test_scorer_skips_missing_tile_image(fake_tile_pool, tmp_path):
    from capi_inference import SubmodelScorer

    db, job_id, tile_ids = fake_tile_pool
    # 砍掉第一張圖
    pool = db.list_tile_pool(job_id)
    Path(pool[0]["source_path"]).unlink()

    bundle_dir = tmp_path / "bundle"; bundle_dir.mkdir()
    (bundle_dir / "WGF50500-inner.pt").write_bytes(b"FAKE")

    scorer = SubmodelScorer(
        gpu_lock=threading.Lock(), db=db, log_fn=lambda m: None,
    )
    with patch.object(scorer, "_load_inferencer_for_pt", return_value=MagicMock()), \
         patch.object(scorer, "_score_one_tile", return_value=0.5):
        result = scorer.score_tiles(
            scoring_bundle_id=99, bundle_dir=bundle_dir,
            lighting="WGF50500", zone="inner",
            tile_pool_job_id=job_id, tile_ids=tile_ids,
            cancel_event=threading.Event(), progress_cb=lambda d, t: None,
        )
    assert result["scanned"] == 4
    assert result["skipped"] == 1
```

- [ ] **Step 2.2：跑測試確認 fail**

```bash
pytest tests/test_submodel_scorer.py -v
```
Expected: 全 FAIL (`SubmodelScorer` 不存在)

- [ ] **Step 2.3：實作 SubmodelScorer**

在 `capi_inference.py` 檔尾加：

```python
class SubmodelScorer:
    """對既有 PatchCore .pt + 一批 tile 跑分，結果寫進 tile_score_cache。

    純跑分 + 寫 cache，不做 reject 判斷、不動 anomaly map 後處理（用線上預設）。
    """

    def __init__(self, gpu_lock, db, log_fn):
        self.gpu_lock = gpu_lock
        self.db = db
        self.log = log_fn
        self._inferencer_cache = {}  # path_str → TorchInferencer

    def _load_inferencer_for_pt(self, pt_path: Path):
        """Lazy load TorchInferencer，路徑為 key 做 cache。"""
        key = str(pt_path)
        if key in self._inferencer_cache:
            return self._inferencer_cache[key]
        if not pt_path.exists():
            raise FileNotFoundError(f"模型檔不存在: {pt_path}")
        from anomalib.deploy import TorchInferencer
        inf = TorchInferencer(path=str(pt_path), device="auto")
        self._inferencer_cache[key] = inf
        return inf

    def _score_one_tile(self, image: np.ndarray, inferencer) -> float:
        """跑一張 tile，回 pred_score（線上預設後處理）。"""
        # 用既有 CAPIInferencer.predict_tile 邏輯太重；scorer 只要 raw pred_score。
        # 包成 TileInfo 物件，呼叫 anomalib inferencer，取 pred_score.max()。
        h, w = image.shape[:2]
        tile = TileInfo(
            tile_id=0, x=0, y=0, width=w, height=h, image=image,
        )
        # 直接呼叫 anomalib inferencer 的 predict
        pred = inferencer.predict(image=image)
        score = float(pred.pred_score.item()) if hasattr(pred.pred_score, "item") \
                else float(pred.pred_score)
        return score

    def score_tiles(
        self,
        scoring_bundle_id: int,
        bundle_dir: Path,
        lighting: str,
        zone: str,
        tile_pool_job_id: str,
        tile_ids: list,
        cancel_event: "threading.Event",
        progress_cb,
    ) -> dict:
        """主 entry。回 {scanned, skipped, cancelled, total}。

        Raises FileNotFoundError 如果 <lighting>-<zone>.pt 不存在。
        """
        pt_path = bundle_dir / f"{lighting}-{zone}.pt"
        inferencer = self._load_inferencer_for_pt(pt_path)

        # 一次性查 source_path
        pool = self.db.list_tile_pool(tile_pool_job_id, lighting=lighting, zone=zone)
        path_by_id = {row["id"]: row["source_path"] for row in pool}

        total = len(tile_ids)
        scanned = 0
        skipped = 0
        cancelled = False
        progress_cb(0, total)

        rows_to_write = []
        BATCH_FLUSH = 20

        for tile_id in tile_ids:
            if cancel_event.is_set():
                cancelled = True
                break
            src = path_by_id.get(tile_id)
            if not src or not Path(src).exists():
                self.log(f"[scorer] tile {tile_id} source 失效，跳過")
                skipped += 1
                progress_cb(scanned + skipped, total)
                continue
            try:
                img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    self.log(f"[scorer] tile {tile_id} 讀圖失敗，跳過")
                    skipped += 1
                    progress_cb(scanned + skipped, total)
                    continue
                with self.gpu_lock:
                    score = self._score_one_tile(img, inferencer)
                rows_to_write.append({
                    "tile_id": tile_id,
                    "scoring_bundle_id": scoring_bundle_id,
                    "score": score,
                })
                scanned += 1
                if len(rows_to_write) >= BATCH_FLUSH:
                    self.db.insert_score_cache(rows_to_write)
                    rows_to_write = []
                progress_cb(scanned + skipped, total)
            except Exception as e:
                self.log(f"[scorer] tile {tile_id} 跑分失敗：{e}")
                skipped += 1
                progress_cb(scanned + skipped, total)

        if rows_to_write:
            self.db.insert_score_cache(rows_to_write)

        return {
            "scanned": scanned,
            "skipped": skipped,
            "cancelled": cancelled,
            "total": total,
        }
```

- [ ] **Step 2.4：跑測試確認 pass**

```bash
pytest tests/test_submodel_scorer.py -v
```
Expected: 4/4 PASS

- [ ] **Step 2.5：commit**

```bash
git add capi_inference.py tests/test_submodel_scorer.py
git commit -m "feat(inference): SubmodelScorer 對 PatchCore .pt 反向跑 tile 分數"
```

---

## Task 3: Cache invalidation hooks

**Files:**
- Modify: `capi_model_registry.py`（既有 `delete_training_data` line 140-170 + `delete_bundle` line 227+）
- Modify: `capi_web.py:6177`（`_submodel_retrain_worker` 寫完 manifest 後）
- Test: `tests/test_score_cache_invalidation.py`（新檔）

- [ ] **Step 3.1：寫失敗測試**

`tests/test_score_cache_invalidation.py`:

```python
"""三條 cache invalidation 路徑測試。"""
from pathlib import Path
import pytest
from capi_database import CAPIDatabase


@pytest.fixture
def db_with_pool(tmp_path: Path):
    db = CAPIDatabase(str(tmp_path / "t.db"))
    job_id = "j1"
    rows = []
    for lighting in ("W0F00000", "G0F00000"):
        for zone in ("inner", "edge"):
            rows.append({
                "lighting": lighting, "zone": zone, "source": "ok",
                "source_path": "x", "thumb_path": "x",
            })
    ids = db.insert_tile_pool(job_id, rows)
    # 對 bundle 10 / 11 各寫 4 筆分數
    cache_rows = []
    for bid in (10, 11):
        for tid in ids:
            cache_rows.append({"tile_id": tid, "scoring_bundle_id": bid, "score": 0.5})
    db.insert_score_cache(cache_rows)
    return db, job_id, ids


def test_invalidate_score_cache_helper_by_lighting_zone(db_with_pool):
    from capi_model_registry import invalidate_score_cache
    db, job_id, ids = db_with_pool
    invalidate_score_cache(db, scoring_bundle_id=10,
                            lighting="W0F00000", zone="inner")
    # 只有 bundle=10 + W0F00000+inner 那筆被清
    remaining_10 = db.get_score_cache(10, ids)
    assert len(remaining_10) == 3  # 原 4 筆扣掉 W0F00000+inner 1 筆
    remaining_11 = db.get_score_cache(11, ids)
    assert len(remaining_11) == 4  # bundle 11 原封不動


def test_invalidate_on_delete_training_data(db_with_pool, monkeypatch, tmp_path):
    from capi_model_registry import delete_training_data
    db, job_id, ids = db_with_pool

    # 假裝有對應 bundle row
    bid = db.register_model_bundle({
        "machine_id": "M1", "bundle_path": str(tmp_path / "b"),
        "trained_at": "2026-01-01", "panel_count": 5, "ng_tile_count": 0,
        "inner_tile_count": 4, "edge_tile_count": 0,
        "bundle_size_bytes": 0, "is_active": 0, "job_id": job_id,
    })

    delete_training_data(db, bundle_id=bid)
    # 該 job 的 tile 都從 pool 消失，相關 score cache 也該清
    assert db.get_score_cache(10, ids) == {}
    assert db.get_score_cache(11, ids) == {}


def test_invalidate_on_delete_bundle(db_with_pool, tmp_path, monkeypatch):
    from capi_model_registry import delete_bundle
    db, job_id, ids = db_with_pool

    bundle_dir = tmp_path / "b"; bundle_dir.mkdir()
    bid = db.register_model_bundle({
        "machine_id": "M1", "bundle_path": str(bundle_dir),
        "trained_at": "2026-01-01", "panel_count": 5, "ng_tile_count": 0,
        "inner_tile_count": 4, "edge_tile_count": 0,
        "bundle_size_bytes": 0, "is_active": 0, "job_id": job_id,
    })
    # 對該 bundle 寫一些 score
    db.insert_score_cache([{"tile_id": ids[0], "scoring_bundle_id": bid, "score": 0.7}])

    # delete_bundle 需要 server_config_path，給個假路徑（讓它走 deactivate 路徑時不炸）
    cfg = tmp_path / "server.yaml"; cfg.write_text("model_configs: []\n")
    delete_bundle(db, bundle_id=bid, server_config_path=cfg)

    # 該 bundle 為 scoring_bundle_id 的所有 row 應該被清
    assert db.get_score_cache(bid, [ids[0]]) == {}
```

- [ ] **Step 3.2：跑測試確認 fail**

```bash
pytest tests/test_score_cache_invalidation.py -v
```
Expected: 全 FAIL

- [ ] **Step 3.3：加 helper + 接入 delete_training_data**

在 `capi_model_registry.py` 檔頭 import 區之後（既有 helper 區）加：

```python
def invalidate_score_cache(db, scoring_bundle_id: int = None,
                            tile_ids: list = None,
                            lighting: str = None, zone: str = None) -> int:
    """tile_score_cache 失效統一入口。回傳刪除筆數。

    用例：
    - 重訓 submodel 完成 → invalidate(scoring_bundle_id=B, lighting=L, zone=Z)
    - 刪訓練資料 → invalidate(tile_ids=[...])
    - 刪 bundle → invalidate(scoring_bundle_id=B)
    """
    return db.delete_score_cache(
        scoring_bundle_id=scoring_bundle_id,
        tile_ids=tile_ids,
        lighting=lighting, zone=zone,
    )
```

修改 `delete_training_data`（line 140-170）。**在 `db.cleanup_tile_pool(job_id)` 之前**先撈 tile_ids 再清 cache：

```python
def delete_training_data(db, bundle_id: int) -> dict:
    bundle = db.get_model_bundle(bundle_id)
    if not bundle:
        raise ValueError(f"bundle {bundle_id} not found")
    job_id = bundle.get("job_id") or ""
    if not job_id:
        return {"ok": True, "message": "此 bundle 沒有關聯 job_id，無訓練資料可清",
                "deleted_files": 0, "freed_bytes": 0, "deleted_tile_rows": 0}

    pool_ok = db.list_tile_pool(job_id, source="ok")
    pool_ng = db.list_tile_pool(job_id, source="ng")
    deleted_rows = len(pool_ok) + len(pool_ng)
    tile_ids_to_clean = [t["id"] for t in pool_ok] + [t["id"] for t in pool_ng]

    # 先清 score cache 再砍 tile_pool（順序很重要：cleanup 後 list_tile_pool 會回空）
    invalidate_score_cache(db, tile_ids=tile_ids_to_clean)
    db.cleanup_tile_pool(job_id)

    thumb_dir = _training_data_dir(job_id)
    freed, deleted_files = _dir_walk_stats(thumb_dir)
    shutil.rmtree(thumb_dir, ignore_errors=True)

    return {
        "ok": True,
        "message": f"已清除 {deleted_rows} 筆 DB 紀錄、{deleted_files} 個檔案，"
                   f"釋放 {freed/1e6:.1f} MB",
        "deleted_tile_rows": deleted_rows,
        "deleted_files": deleted_files,
        "freed_bytes": freed,
    }
```

- [ ] **Step 3.4：接入 delete_bundle**

找 `delete_bundle`（line 227+）在 `db.delete_model_bundle(bundle_id)` 之前加：

```python
    # 該 bundle 作為 scoring bundle 算過的所有分都失效
    invalidate_score_cache(db, scoring_bundle_id=bundle_id)
```

- [ ] **Step 3.5：接入 retrain_submodel worker**

在 `capi_web.py:6177`（`append_submodel_history(...)` 之後、`_set_step("reload")` 之前）加：

```python
            # 該 bundle 對該 lighting+zone 的舊分全失效
            from capi_model_registry import invalidate_score_cache
            cleared = invalidate_score_cache(
                db, scoring_bundle_id=bundle_id, lighting=lighting, zone=zone,
            )
            _log(f"清除 {cleared} 筆 score cache（lighting={lighting}, zone={zone}）")
```

- [ ] **Step 3.6：跑測試確認 pass**

```bash
pytest tests/test_score_cache_invalidation.py -v
```
Expected: 3/3 PASS

- [ ] **Step 3.7：commit**

```bash
git add capi_model_registry.py capi_web.py tests/test_score_cache_invalidation.py
git commit -m "feat(registry): tile_score_cache 失效 hook 接入 retrain/delete 流程"
```

---

## Task 4: Scan 後台 worker 與單例 state

**Files:**
- Modify: `capi_web.py`（既有 `_submodel_retrain_state` 旁邊，line ~210）
- Modify: `capi_web.py`（worker function 區）

- [ ] **Step 4.1：加 server-level scan_state 與 worker**

在 `capi_web.py` `_submodel_retrain_state = {...}`（搜 `_submodel_retrain_state`）下方加：

```python
    # 全 server 同時只能一個 score scan job（共用 GPU，序列化）
    _scan_state = {
        "lock": threading.Lock(),
        "job": None,  # Optional[dict]，欄位見 _start_scan_job
    }
```

在 `capi_web.py` `_submodel_retrain_worker` 之後加共用 worker 與 state helper：

```python
    @classmethod
    def _start_scan_job(
        cls,
        kind: str,                     # "self" | "prefilter"
        scoring_bundle_id: int,
        bundle_dir: "Path",
        tile_pool_job_id: str,
        lighting: str,
        zone: str,
        tile_ids: list,
        server_inst,
    ) -> tuple:
        """嘗試啟動 scan job。回傳 (started: bool, response_dict)。"""
        import uuid
        state = cls._scan_state
        with state["lock"]:
            current = state["job"]
            if current and current.get("state") == "running":
                return False, {"error": "已有 scan job 進行中", "job": current}
            scan_id = "scan_" + uuid.uuid4().hex[:12]
            cancel_event = threading.Event()
            state["job"] = {
                "scan_id": scan_id,
                "kind": kind,
                "scoring_bundle_id": scoring_bundle_id,
                "tile_pool_job_id": tile_pool_job_id,
                "lighting": lighting,
                "zone": zone,
                "total": len(tile_ids),
                "done": 0,
                "skipped": 0,
                "state": "running",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "error": None,
                "cancel_event": cancel_event,
            }

        thread = threading.Thread(
            target=cls._scan_worker,
            args=(scan_id, scoring_bundle_id, bundle_dir, tile_pool_job_id,
                  lighting, zone, tile_ids, cancel_event, server_inst),
            daemon=True,
            name=f"scan-{scan_id}",
        )
        thread.start()
        return True, {"scan_id": scan_id, "total": len(tile_ids)}

    @classmethod
    def _scan_worker(cls, scan_id, scoring_bundle_id, bundle_dir,
                      tile_pool_job_id, lighting, zone, tile_ids,
                      cancel_event, server_inst):
        from capi_inference import SubmodelScorer
        state = cls._scan_state

        def _progress(done, total):
            with state["lock"]:
                if state["job"] and state["job"]["scan_id"] == scan_id:
                    state["job"]["done"] = done

        def _log(msg):
            logger.info("[scan %s] %s", scan_id, msg)

        try:
            scorer = SubmodelScorer(
                gpu_lock=server_inst._gpu_lock,
                db=server_inst.database,
                log_fn=_log,
            )
            result = scorer.score_tiles(
                scoring_bundle_id=scoring_bundle_id,
                bundle_dir=bundle_dir,
                lighting=lighting, zone=zone,
                tile_pool_job_id=tile_pool_job_id,
                tile_ids=tile_ids,
                cancel_event=cancel_event,
                progress_cb=_progress,
            )
            with state["lock"]:
                if state["job"] and state["job"]["scan_id"] == scan_id:
                    state["job"]["state"] = "cancelled" if result["cancelled"] else "done"
                    state["job"]["done"] = result["scanned"] + result["skipped"]
                    state["job"]["skipped"] = result["skipped"]
        except FileNotFoundError as e:
            with state["lock"]:
                if state["job"] and state["job"]["scan_id"] == scan_id:
                    state["job"]["state"] = "failed"
                    state["job"]["error"] = str(e)
        except Exception as e:
            logger.exception("scan worker crashed")
            with state["lock"]:
                if state["job"] and state["job"]["scan_id"] == scan_id:
                    state["job"]["state"] = "failed"
                    state["job"]["error"] = str(e)
```

- [ ] **Step 4.2：commit**

```bash
git add capi_web.py
git commit -m "feat(web): scan job 單例 state 與共用 worker"
```

---

## Task 5: Scan API endpoints (start/status/cancel)

**Files:**
- Modify: `capi_web.py`（router + handler）

- [ ] **Step 5.1：加 router**

在 `capi_web.py` POST router 區（搜 `/retrain_submodel`）加：

```python
            elif path.startswith("/api/models/") and path.endswith("/scan_self_score"):
                self._handle_scan_self_score()
            elif path == "/api/train/new/scan_prefilter_score":
                self._handle_scan_prefilter_score()
            elif path == "/api/scan/cancel":
                self._handle_scan_cancel()
```

GET router 區加：

```python
            elif path == "/api/scan/status":
                self._handle_scan_status()
            elif path == "/api/train/new/eligible_scoring_bundles":
                self._handle_eligible_scoring_bundles()
```

- [ ] **Step 5.2：加 handler**

在 `_submodel_retrain_worker` 之後加：

```python
    def _handle_scan_self_score(self):
        """POST /api/models/<bundle_id>/scan_self_score
        body: {"lighting": str, "zone": "inner"|"edge"}
        """
        from capi_train_new import LIGHTINGS, ZONES
        parts = self.path.split("/")
        try:
            bundle_id = int(parts[3])
        except (ValueError, IndexError):
            self._send_json({"error": "invalid bundle id"}, status=400); return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            self._send_json({"error": "invalid JSON"}, status=400); return

        lighting = body.get("lighting"); zone = body.get("zone")
        if lighting not in LIGHTINGS or zone not in ZONES:
            self._send_json({"error": "lighting/zone 不合法"}, status=400); return

        db = self._capi_server_instance.database
        bundle = db.get_model_bundle(bundle_id)
        if not bundle:
            self._send_json({"error": "bundle not found"}, status=404); return
        if not bundle.get("job_id"):
            self._send_json({"error": "此 bundle 無關聯 job_id（訓練資料已刪）"}, status=400)
            return

        # 自掃 = 該 bundle 對「自己訓練資料」算分
        pool = db.list_tile_pool(
            bundle["job_id"], lighting=lighting, zone=zone, source="ok",
        )
        if not pool:
            self._send_json({"state": "empty", "scanned": 0}); return
        tile_ids = [t["id"] for t in pool]

        # 已有 cache 全命中？告知前端不需重算
        cached = db.get_score_cache(bundle_id, tile_ids)
        if len(cached) == len(tile_ids):
            self._send_json({"cached_hit": True, "total": len(tile_ids)})
            return

        started, resp = CAPIWebHandler._start_scan_job(
            kind="self",
            scoring_bundle_id=bundle_id,
            bundle_dir=Path(bundle["bundle_path"]),
            tile_pool_job_id=bundle["job_id"],
            lighting=lighting, zone=zone, tile_ids=tile_ids,
            server_inst=self._capi_server_instance,
        )
        self._send_json(resp, status=200 if started else 409)

    def _handle_scan_prefilter_score(self):
        """POST /api/train/new/scan_prefilter_score
        body: {"job_id", "scoring_bundle_id", "lighting", "zone"}
        """
        from capi_train_new import LIGHTINGS, ZONES
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            self._send_json({"error": "invalid JSON"}, status=400); return

        tile_pool_job_id = body.get("job_id")
        scoring_bundle_id = body.get("scoring_bundle_id")
        lighting = body.get("lighting"); zone = body.get("zone")
        if not all([tile_pool_job_id, scoring_bundle_id, lighting, zone]):
            self._send_json({"error": "missing required fields"}, status=400); return
        if lighting not in LIGHTINGS or zone not in ZONES:
            self._send_json({"error": "lighting/zone 不合法"}, status=400); return

        db = self._capi_server_instance.database
        scoring_bundle = db.get_model_bundle(int(scoring_bundle_id))
        if not scoring_bundle:
            self._send_json({"error": "scoring bundle not found"}, status=404); return

        pool = db.list_tile_pool(tile_pool_job_id, lighting=lighting, zone=zone, source="ok")
        if not pool:
            self._send_json({"state": "empty"}); return
        tile_ids = [t["id"] for t in pool]

        cached = db.get_score_cache(int(scoring_bundle_id), tile_ids)
        if len(cached) == len(tile_ids):
            self._send_json({"cached_hit": True, "total": len(tile_ids)})
            return

        started, resp = CAPIWebHandler._start_scan_job(
            kind="prefilter",
            scoring_bundle_id=int(scoring_bundle_id),
            bundle_dir=Path(scoring_bundle["bundle_path"]),
            tile_pool_job_id=tile_pool_job_id,
            lighting=lighting, zone=zone, tile_ids=tile_ids,
            server_inst=self._capi_server_instance,
        )
        self._send_json(resp, status=200 if started else 409)

    def _handle_scan_status(self):
        """GET /api/scan/status — 回目前唯一 scan job 狀態（沒有則 idle）。"""
        state = CAPIWebHandler._scan_state
        with state["lock"]:
            job = state["job"]
            if job is None:
                self._send_json({"state": "idle"})
                return
            # 不回傳 cancel_event 物件
            payload = {k: v for k, v in job.items() if k != "cancel_event"}
            self._send_json(payload)

    def _handle_scan_cancel(self):
        """POST /api/scan/cancel — 對當前 running job 設 cancel_event。"""
        state = CAPIWebHandler._scan_state
        with state["lock"]:
            job = state["job"]
            if not job or job["state"] != "running":
                self._send_json({"cancelled": False, "reason": "no running job"})
                return
            job["cancel_event"].set()
        self._send_json({"cancelled": True})
```

- [ ] **Step 5.3：sanity check**

```bash
python -c "import capi_web; print(hasattr(capi_web.CAPIWebHandler, '_scan_state'))"
```
Expected: `True`

- [ ] **Step 5.4：commit**

```bash
git add capi_web.py
git commit -m "feat(web): scan_self_score / scan_prefilter_score / status / cancel API"
```

---

## Task 6: Eligible scoring bundles API

**Files:**
- Modify: `capi_web.py`（加 handler）

- [ ] **Step 6.1：加 handler**

```python
    def _handle_eligible_scoring_bundles(self):
        """GET /api/train/new/eligible_scoring_bundles
        回所有「.pt 檔健全的 trained bundle」清單，給 step3 prefilter 下拉用。
        """
        db = self._capi_server_instance.database
        bundles = db.list_model_bundles() or []
        from capi_train_new import LIGHTINGS, ZONES
        out = []
        for b in bundles:
            bundle_dir = Path(b["bundle_path"])
            # 至少要存在 1 個 .pt 才算可用（細項 lighting+zone 由 frontend 切 tab 才知）
            has_any_pt = any(
                (bundle_dir / f"{l}-{z}.pt").exists()
                for l in LIGHTINGS for z in ZONES
            )
            if not has_any_pt:
                continue
            label = (
                f"{b['machine_id']} / "
                f"{Path(b['bundle_path']).name}"
                f"{' ●active' if b.get('is_active') else ''}"
            )
            out.append({
                "id": b["id"],
                "machine_id": b["machine_id"],
                "trained_at": b.get("trained_at"),
                "is_active": bool(b.get("is_active")),
                "label": label,
            })
        # 排序：active 優先 → trained_at desc
        out.sort(key=lambda x: (not x["is_active"], x["trained_at"] or ""), reverse=False)
        self._send_json({"bundles": out})
```

- [ ] **Step 6.2：sanity check**

啟動 server，curl 測：

```bash
curl http://localhost:8080/api/train/new/eligible_scoring_bundles
```
Expected: `{"bundles": [...]}` 列出所有 trained bundle

- [ ] **Step 6.3：commit**

```bash
git add capi_web.py
git commit -m "feat(web): eligible_scoring_bundles API 給 step3 prefilter 下拉"
```

---

## Task 7: Tile listing API 擴充 score 排序

**Files:**
- Modify: `capi_web.py`（既有 `_handle_models_training_tiles` + `_handle_train_new_tiles`）

- [ ] **Step 7.1：找既有 _handle_models_training_tiles**

```bash
grep -n "_handle_models_training_tiles\|/training_tiles" capi_web.py
```

- [ ] **Step 7.2：擴充 bundle detail 的 training_tiles**

在 `_handle_models_training_tiles` 內，**讀 tile list 之後、回傳之前**加 score join：

```python
        # 既有：tiles = db.list_tile_pool(...)
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(self.path).query)
        score_from = qs.get("score_from_bundle", [None])[0]
        sort_by = qs.get("sort_by", ["default"])[0]

        if score_from:
            tile_ids = [t["id"] for t in tiles]
            scores = db.get_score_cache(int(score_from), tile_ids)
            # 標分數 + 分位
            scored = [(tid, scores.get(tid)) for tid in tile_ids]
            present = [s for _, s in scored if s is not None]
            if present:
                present_sorted = sorted(present, reverse=True)
                top5_cut = present_sorted[max(0, int(len(present_sorted) * 0.05) - 1)]
                top20_cut = present_sorted[max(0, int(len(present_sorted) * 0.20) - 1)]
            else:
                top5_cut = top20_cut = float("inf")
            for tile in tiles:
                s = scores.get(tile["id"])
                tile["score"] = s
                if s is None:
                    tile["score_quartile"] = None
                elif s >= top5_cut:
                    tile["score_quartile"] = "top5"
                elif s >= top20_cut:
                    tile["score_quartile"] = "top20"
                else:
                    tile["score_quartile"] = "rest"
            if sort_by == "score_desc":
                tiles.sort(key=lambda t: (t.get("score") is None, -(t.get("score") or 0)))
```

- [ ] **Step 7.3：擴充 step3 的 train_new tiles**

`_handle_train_new_tiles`（line 5241）加同樣的邏輯（複製貼上即可，或抽 helper）。為 DRY，抽出一個 module-level helper：

在 capi_web.py 找一個合適位置（例如靠近 `_train_new_thumb_url` 旁）加：

```python
    @staticmethod
    def _decorate_tiles_with_scores(tiles: list, db, score_from_bundle_id: int,
                                     sort_by: str) -> list:
        """In-place 為 tile dict 加 score / score_quartile，並依 sort_by 排序。"""
        if not score_from_bundle_id:
            return tiles
        tile_ids = [t["id"] for t in tiles]
        scores = db.get_score_cache(int(score_from_bundle_id), tile_ids)
        present = sorted(
            [s for tid, s in scores.items()], reverse=True,
        )
        if present:
            top5_idx = max(0, int(len(present) * 0.05) - 1)
            top20_idx = max(0, int(len(present) * 0.20) - 1)
            top5_cut = present[top5_idx]
            top20_cut = present[top20_idx]
        else:
            top5_cut = top20_cut = float("inf")
        for tile in tiles:
            s = scores.get(tile["id"])
            tile["score"] = s
            if s is None:
                tile["score_quartile"] = None
            elif s >= top5_cut:
                tile["score_quartile"] = "top5"
            elif s >= top20_cut:
                tile["score_quartile"] = "top20"
            else:
                tile["score_quartile"] = "rest"
        if sort_by == "score_desc":
            tiles.sort(key=lambda t: (t.get("score") is None, -(t.get("score") or 0)))
        return tiles
```

把 Step 7.2 的 inline 邏輯改呼叫此 helper。`_handle_train_new_tiles` 也用同樣 helper：

```python
    def _handle_train_new_tiles(self):
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(self.path).query)
        job_id = (qs.get("job_id") or [""])[0]
        lighting = (qs.get("lighting") or [""])[0]
        score_from = qs.get("score_from_bundle", [None])[0]
        sort_by = qs.get("sort_by", ["default"])[0]
        if not job_id or not lighting:
            self._send_json({"error": "job_id and lighting required"}, status=400); return
        db = self._capi_server_instance.database
        tiles = db.list_tile_pool(job_id, lighting=lighting)
        for tile in tiles:
            tile["thumb_url"] = self._train_new_thumb_url(tile.get("thumb_path"))
            tile["image_url"] = self._train_new_thumb_url(tile.get("source_path"))
        if score_from:
            CAPIWebHandler._decorate_tiles_with_scores(
                tiles, db, int(score_from), sort_by,
            )
        self._send_json({"tiles": tiles})
```

- [ ] **Step 7.4：sanity check**

```bash
python -c "from capi_web import CAPIWebHandler; print(callable(CAPIWebHandler._decorate_tiles_with_scores))"
```
Expected: `True`

- [ ] **Step 7.5：commit**

```bash
git add capi_web.py
git commit -m "feat(web): training_tiles / train_new_tiles 支援 score_from_bundle 排序"
```

---

## Task 8: Bundle detail 前端按鈕 + 進度 + badge

**Files:**
- Modify: `templates/models.html`（CSS + tile loadTiles 邏輯 + 新增 scan 按鈕）

- [ ] **Step 8.1：加 CSS**

`templates/models.html:127`（既有 `</style>` 之前）加：

```css
  /* Score badge for self-scoring */
  .tile-thumb { position: relative; }
  .tile-thumb .score-badge {
    position: absolute; top: 4px; right: 4px;
    padding: 2px 6px; border-radius: 8px;
    font-size: 0.7rem; font-family: 'JetBrains Mono', Consolas, monospace;
    background: #45475a; color: #cdd6f4; pointer-events: none;
    z-index: 2;
  }
  .tile-thumb[data-score-quartile="top5"]  .score-badge {
    background: #f38ba8; color: #1e1e2e; font-weight: 700;
  }
  .tile-thumb[data-score-quartile="top20"] .score-badge {
    background: #fab387; color: #1e1e2e;
  }
  /* Scan button + progress */
  .scan-self-btn {
    background: #89b4fa; color: #1e1e2e; border: none;
    padding: 5px 12px; border-radius: 4px; font-size: .78rem;
    font-weight: 700; cursor: pointer; margin-left: 8px;
  }
  .scan-self-btn[disabled] { background: #45475a; color: #6c7086; cursor: wait; }
```

- [ ] **Step 8.2：加掃描按鈕**

找 OK tiles tab 的 lighting/zone 控制列（搜 `mlibOkLighting`），在 `mlibOkCount` span 旁邊加按鈕：

```html
        <button id="scanSelfBtn" class="scan-self-btn" onclick="startSelfScan()">
          🔍 掃描可疑 tile
        </button>
        <span id="scanSelfStatus" style="margin-left:8px;color:#a6adc8;font-size:.78rem;"></span>
```

- [ ] **Step 8.3：加 JS（startSelfScan + 輪詢 + 完成 reload）**

在 `loadTiles` function 之前加：

```javascript
let _scanPollTimer = null;
let _activeScoreBundleId = null;  // grid 目前用哪個 bundle 的分數排序

async function startSelfScan() {
  if (!currentBundleId) return;
  const lighting = document.getElementById('mlibOkLighting').value;
  const zone = document.getElementById('mlibOkZone').value;
  if (!lighting || !zone) { alert("請先選 lighting/zone"); return; }

  const btn = document.getElementById('scanSelfBtn');
  btn.disabled = true; btn.textContent = '掃描中...';

  let resp;
  try {
    resp = await fetch(`/api/models/${currentBundleId}/scan_self_score`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({lighting, zone}),
    }).then(r => r.json());
  } catch (e) { alert("啟動失敗: " + e); resetScanBtn(); return; }

  if (resp.error) { alert(resp.error); resetScanBtn(); return; }
  if (resp.cached_hit || resp.state === 'empty') {
    onScanFinished(currentBundleId);
    return;
  }
  // 進入輪詢
  pollScanStatus(currentBundleId);
}

function resetScanBtn() {
  const btn = document.getElementById('scanSelfBtn');
  btn.disabled = false; btn.textContent = '🔍 掃描可疑 tile';
  document.getElementById('scanSelfStatus').textContent = '';
}

function pollScanStatus(bundleId) {
  if (_scanPollTimer) clearInterval(_scanPollTimer);
  _scanPollTimer = setInterval(async () => {
    const s = await fetch('/api/scan/status').then(r => r.json());
    const status = document.getElementById('scanSelfStatus');
    if (s.state === 'idle') { clearInterval(_scanPollTimer); resetScanBtn(); return; }
    status.textContent = `${s.done || 0}/${s.total || 0}`;
    if (s.state === 'done' || s.state === 'cancelled' || s.state === 'failed') {
      clearInterval(_scanPollTimer);
      if (s.state === 'failed') { alert("掃描失敗: " + s.error); resetScanBtn(); return; }
      onScanFinished(bundleId);
    }
  }, 800);
}

function onScanFinished(bundleId) {
  _activeScoreBundleId = bundleId;  // 自掃結果用該 bundle 自己的分
  resetScanBtn();
  loadTiles('ok');  // 重 fetch 帶 score_from_bundle
}
```

- [ ] **Step 8.4：改 loadTiles 帶 score_from_bundle**

找既有 `loadTiles`（line 601），把 `params` 建構處改為：

```javascript
  const params = new URLSearchParams({source, limit: '500'});
  if (lighting) params.set('lighting', lighting);
  if (zone) params.set('zone', zone);
  if (source === 'ok' && _activeScoreBundleId === currentBundleId) {
    params.set('score_from_bundle', String(currentBundleId));
    params.set('sort_by', 'score_desc');
  }
```

並把 tile innerHTML 模板改加 badge：

```javascript
      grid.innerHTML = d.tiles.map(t =>
        t.thumb_url
          ? `<div class="tile-thumb" data-tile-id="${t.id}" data-decision="${t.decision || 'accept'}" data-image-url="${t.image_url || t.thumb_url}" data-score-quartile="${t.score_quartile || ''}">`
            + `<img src="${t.thumb_url}" title="${t.lighting} ${t.zone || ''} #${t.id}${t.score != null ? ' score=' + t.score.toFixed(3) : ''}" loading="lazy">`
            + (t.score != null ? `<span class="score-badge">${t.score.toFixed(2)}</span>` : '')
            + `</div>`
          : `<div title="${t.lighting} ${t.zone || ''} (no thumb)" style="background:#313244;border-radius:4px;aspect-ratio:1;"></div>`
      ).join('');
```

切 lighting/zone 時 reset score state：在 `mlibOkLighting` / `mlibOkZone` change handler（搜 `loadTiles('ok')`）前加：

```javascript
  // 切 lighting/zone → 清除「目前以什麼 bundle 排序」的記憶
  _activeScoreBundleId = null;
```

- [ ] **Step 8.5：手動 browser test**

開 server，瀏覽 `http://localhost:8080/models`，點任一 bundle → OK tiles → 選 WGF50500/inner → 按「🔍 掃描可疑 tile」。

驗證：
- 按鈕變灰、status 顯示 `n/180`
- 算完 grid 重排，前面幾張帶紅 / 橘 badge
- 切 lighting → grid 順序回正常（無 badge）
- 重新進該 lighting → badge 還在（cache hit 路徑）

- [ ] **Step 8.6：commit**

```bash
git add templates/models.html
git commit -m "feat(web): bundle detail OK tiles 加自掃按鈕、score badge、依分排序"
```

---

## Task 9: Step3 prefilter 前端

**Files:**
- Modify: `templates/train_new/step3_review.html`（CSS + 頂部下拉 + JS lazy compute）

- [ ] **Step 9.1：加 CSS（與 models.html 同一套 badge）**

`templates/train_new/step3_review.html` 既有 `<style>` 區塊加：

```css
  .tile-thumb .score-badge {
    position: absolute; top: 4px; right: 4px;
    padding: 2px 6px; border-radius: 8px;
    font-size: 0.7rem; font-family: 'JetBrains Mono', Consolas, monospace;
    background: #45475a; color: #cdd6f4; pointer-events: none; z-index: 2;
  }
  .tile-thumb[data-score-quartile="top5"]  .score-badge {
    background: #f38ba8; color: #1e1e2e; font-weight: 700;
  }
  .tile-thumb[data-score-quartile="top20"] .score-badge {
    background: #fab387; color: #1e1e2e;
  }
  .prefilter-bar {
    background: #313244; padding: 8px 12px; border-radius: 6px;
    margin-bottom: 10px; display: flex; align-items: center; gap: 10px;
    color: #cdd6f4; font-size: .82rem;
  }
  .prefilter-bar select { background: #1e1e2e; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 4px; padding: 4px 8px;
    font-size: .82rem; }
  .prefilter-cross-warn { color: #f9e2af; font-size: .76rem; }
```

- [ ] **Step 9.2：加 prefilter 下拉控制列**

在 step3 主 panel 上方（lighting tab 之前）加：

```html
<div class="prefilter-bar">
  <span>預篩模型：</span>
  <select id="prefilterBundleSelect" onchange="onPrefilterBundleChange()">
    <option value="">不啟用（手動 review）</option>
  </select>
  <span id="prefilterStatus" style="color:#a6adc8;"></span>
  <span id="prefilterCrossWarn" class="prefilter-cross-warn" style="display:none;">
    ⚠ 跨 machine 模型，分數僅當相對排序參考
  </span>
</div>
```

- [ ] **Step 9.3：加 JS：載 bundle 清單、切 tab 觸發 lazy scan、輪詢、reload**

在 step3 既有 JS 區末加：

```javascript
let _prefilterBundleId = null;
let _prefilterScanCache = {};  // key=`${lighting}|${zone}` → "done" | "running"
let _prefilterPoll = null;
let _currentJobId = (typeof JOB_ID !== 'undefined') ? JOB_ID : null;
let _currentMachineId = (typeof MACHINE_ID !== 'undefined') ? MACHINE_ID : null;

async function loadEligibleScoringBundles() {
  const sel = document.getElementById('prefilterBundleSelect');
  const r = await fetch('/api/train/new/eligible_scoring_bundles').then(r => r.json());
  for (const b of (r.bundles || [])) {
    const opt = document.createElement('option');
    opt.value = b.id;
    opt.dataset.machine = b.machine_id;
    opt.textContent = b.label;
    sel.appendChild(opt);
  }
  if (!r.bundles || r.bundles.length === 0) {
    sel.disabled = true;
    sel.options[0].textContent = '目前無可用模型';
  }
}

function onPrefilterBundleChange() {
  const sel = document.getElementById('prefilterBundleSelect');
  _prefilterBundleId = sel.value ? parseInt(sel.value, 10) : null;
  _prefilterScanCache = {};  // 換 bundle 清除 lazy cache 記憶（DB cache 仍在）
  // 跨 machine 警示
  const opt = sel.options[sel.selectedIndex];
  const cross = opt && opt.dataset.machine && opt.dataset.machine !== _currentMachineId;
  document.getElementById('prefilterCrossWarn').style.display = cross ? '' : 'none';
  // 重載目前 tab
  reloadCurrentTabWithPrefilter();
}

async function ensurePrefilterScoreForCurrentTab(lighting, zone) {
  if (!_prefilterBundleId) return false;
  const key = `${lighting}|${zone}`;
  if (_prefilterScanCache[key] === 'done') return true;
  // 觸發 scan
  document.getElementById('prefilterStatus').textContent = '預篩中...';
  const resp = await fetch('/api/train/new/scan_prefilter_score', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      job_id: _currentJobId, scoring_bundle_id: _prefilterBundleId,
      lighting, zone,
    }),
  }).then(r => r.json());
  if (resp.error) {
    alert("預篩失敗: " + resp.error);
    document.getElementById('prefilterStatus').textContent = '';
    return false;
  }
  if (resp.cached_hit || resp.state === 'empty') {
    _prefilterScanCache[key] = 'done';
    document.getElementById('prefilterStatus').textContent = '';
    return true;
  }
  // 輪詢
  return await new Promise(resolve => {
    if (_prefilterPoll) clearInterval(_prefilterPoll);
    _prefilterPoll = setInterval(async () => {
      const s = await fetch('/api/scan/status').then(r => r.json());
      const status = document.getElementById('prefilterStatus');
      status.textContent = (s.state === 'running')
        ? `預篩中 ${s.done}/${s.total}` : '';
      if (s.state === 'done' || s.state === 'cancelled' || s.state === 'idle') {
        clearInterval(_prefilterPoll);
        _prefilterScanCache[key] = 'done';
        status.textContent = '';
        resolve(true);
      } else if (s.state === 'failed') {
        clearInterval(_prefilterPoll);
        alert("預篩失敗: " + (s.error || ''));
        status.textContent = '';
        resolve(false);
      }
    }, 800);
  });
}

async function reloadCurrentTabWithPrefilter() {
  // 假設 step3 既有 loadStep3Tiles(lighting, zone) function
  const lighting = currentLighting();  // 既有
  const zone = currentZone();          // 既有
  if (_prefilterBundleId) {
    const ok = await ensurePrefilterScoreForCurrentTab(lighting, zone);
    if (!ok) { loadStep3Tiles(lighting, zone, null); return; }
    loadStep3Tiles(lighting, zone, _prefilterBundleId);
  } else {
    loadStep3Tiles(lighting, zone, null);
  }
}

// page 載入時 hook
document.addEventListener('DOMContentLoaded', loadEligibleScoringBundles);
```

- [ ] **Step 9.4：改 step3 既有 loadStep3Tiles 帶 score_from_bundle**

找 step3 載 tile 的 fetch（搜 `/api/train/new/tiles`），加 query 與 badge render：

```javascript
async function loadStep3Tiles(lighting, zone, scoreFromBundle) {
  const params = new URLSearchParams({job_id: _currentJobId, lighting});
  if (scoreFromBundle) {
    params.set('score_from_bundle', String(scoreFromBundle));
    params.set('sort_by', 'score_desc');
  }
  const r = await fetch('/api/train/new/tiles?' + params).then(r => r.json());
  const tiles = (r.tiles || []).filter(t => !zone || t.zone === zone);
  // render（沿用既有 grid template + 加 badge 同 models.html）
  renderTilesGrid(tiles);
}
```

`renderTilesGrid` 是 step3 既有函式，把 thumb 模板加上 badge（與 Task 8.4 同邏輯）：

```javascript
  grid.innerHTML = tiles.map(t => `
    <div class="tile-thumb" data-tile-id="${t.id}" data-decision="${t.decision || 'accept'}"
         data-score-quartile="${t.score_quartile || ''}">
      <img src="${t.thumb_url}" loading="lazy">
      ${t.score != null ? `<span class="score-badge">${t.score.toFixed(2)}</span>` : ''}
    </div>
  `).join('');
```

- [ ] **Step 9.5：切 tab 時觸發 prefilter**

step3 切 lighting / zone 既有 handler（搜 `onLightingChange` 或類似）末尾加：

```javascript
  reloadCurrentTabWithPrefilter();
```

取代既有直接呼叫 `loadStep3Tiles(...)` 那一行。

- [ ] **Step 9.6：手動 browser test**

啟動 server，跑一次 `/train/new` wizard 到 step3：
- 預篩下拉預設「不啟用」→ tile grid 跟現狀一樣
- 選同 machine 一個 active bundle → status 顯示「預篩中 n/m」→ 跑完 grid 重排有 badge
- 切到別的 lighting tab → 該 tab 也跑 prefilter（可能有等待 GPU）
- 切 bundle 為跨 machine → `⚠ 跨 machine` 警示出現
- DB cache hit 時切回原 tab → 秒開、無等待

- [ ] **Step 9.7：commit**

```bash
git add templates/train_new/step3_review.html
git commit -m "feat(web): step3 review 加預篩模型下拉、lazy scan、score badge"
```

---

## Task 10: 整合驗證

**Files:** 不改檔，純驗證

- [ ] **Step 10.1：bundle 自掃 → reject → 重訓 → 確認分數變化**

1. 啟 server、跑 GN160JCEL250S 對 WGF50500-inner 自掃
2. 看 top 紅 badge tile，挑一張肉眼確認是污染（有缺陷紋理）
3. 進 preview modal 按 Del reject
4. 點「重訓 WGF50500-inner」
5. 等重訓完成 → 自掃按鈕變「🔍 掃描可疑 tile」（cache 已清）
6. 重新自掃 → 該 tile 應該（如果它沒被 reject 進不了訓練集）跑分時不在 pool 內；其他剩餘 tile 的 score 分布應該明顯不同

- [ ] **Step 10.2：step3 預篩端到端**

1. 跑 `/train/new` 到 step3
2. 選 active bundle 預篩
3. 切到 WGF50500/inner，觀察進度條跑完、grid 排序、badge 顯示
4. 切到 R0F00000/inner → 也觸發 scan（GPU 序列化）
5. 切回 WGF50500/inner → cache hit、秒開
6. 換選跨 machine bundle → 跨 machine 警示出現、scan 重跑

- [ ] **Step 10.3：邊界 case**

1. 自掃 .pt 不存在的 lighting+zone（例如某舊 bundle 訓練失敗的）→ status 顯示 failed + error
2. 自掃中途按 cancel → grid 不變、cache 已寫的 row 仍保留
3. 同時開兩個 tab 都點掃描 → 第二個 409 顯示「已有進行中」

- [ ] **Step 10.4：驗證完成 commit checkpoint**

```bash
git log --oneline -10
```
Expected: 看到 Task 1–9 的 9 個 commit 依序排列。

---

## Self-Review Checklist

- [x] **Spec coverage**：spec 每節都有對應 task（DB→T1, scorer→T2, hooks→T3, scan worker→T4, scan API→T5, eligible→T6, tile listing→T7, bundle UI→T8, step3 UI→T9, integration→T10）
- [x] **Placeholder scan**：每個 step 都有具體 code / 命令，無 TBD / TODO
- [x] **Type consistency**：`tile_score_cache` 欄位、`SubmodelScorer.score_tiles` 簽章、`_decorate_tiles_with_scores` 介面在各 task 一致
- [x] **Cache invalidation 三條路徑**：T3 都涵蓋（retrain / delete_training_data / delete_bundle）
- [x] **GPU 鎖**：T2 scorer 內、T4 worker 共用 `server_inst._gpu_lock`，序列化既有 retrain / inference
- [x] **Error handling**：T2 跳過 missing image、T5 status 回 failed + error、T8/T9 alert 給使用者

---
