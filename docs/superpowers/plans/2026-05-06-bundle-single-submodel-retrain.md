# Bundle 單一子模型重新訓練 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓使用者在 `/models` bundle detail 頁直接標記要排除的訓練 tile，重訓單一 lighting+zone 子模型並就地覆蓋同 bundle 內的 `.pt` 檔。

**Architecture:** 沿用既有 `train_one_patchcore` + `_calibrate_from_model` 抽出可單 unit 呼叫的 `train_single_submodel`；UI 沿用 step3 review 的 accept/reject 鍵盤約定；後台 worker 沿用 `_train_new_state` 的 dict + lock + log_lines 模式；模型 reload 用 `inferencer._model_cache_v2` pop 該 key 觸發 lazy reload。**threshold/yaml 完全不動**——`calibrate_threshold` 已寫死回 0.5，threshold 由使用者在模型庫手動調整。

**Tech Stack:** Python 3.11+, anomalib, sqlite (WAL), Jinja2 templates, vanilla JS（沿用既有 inline pattern）。

**Spec：** `docs/superpowers/specs/2026-05-06-bundle-single-submodel-retrain-design.md`

---

## File Structure

| 檔案 | 動作 | 責任 |
|------|------|------|
| `capi_train_new.py` | 修改 | 抽出 `train_single_submodel(...)`；初次訓練在 manifest 寫入 `used_tile_ids` |
| `capi_model_registry.py` | 修改 | 加 `append_submodel_history`、`get_used_tile_ids`、`get_pending_change_count`、`get_pending_change_summary_for_bundle`；`get_bundle_detail` 多回 `pending_changes` |
| `capi_inference.py` | 修改 | 加 `reload_submodel(machine_id, lighting, zone)` |
| `capi_web.py` | 修改 | 加 `_submodel_retrain_state` + 4 個 handler + worker；`_handle_models_training_tiles` 回傳加 `decision`；`_handle_models_detail` 多回 `pending_changes` |
| `templates/models.html` | 修改 | tile 區改成可標 reject；變更指示條 + 重訓按鈕 + 進度面板；列表頁加未訓練修改徽章 |
| `tests/test_submodel_retrain.py` | 新建 | 純函式單元測試（manifest 操作、pending 計算） |

---

## Task 1: capi_train_new.py — 抽出 `train_single_submodel`

**目的：** 讓 `_submodel_retrain_worker` 不重寫一遍訓練邏輯，沿用同樣的 stage / train / calibrate / metrics 計算。

**Files:**
- Modify: `capi_train_new.py:879-1058`（主要在 `run_training_pipeline` 的 unit 迴圈內抽出共用步驟為新函式 `train_single_submodel`）

**設計：** 保持 `run_training_pipeline` 行為不變，新函式接受 `(db, job_id, lighting, zone, cfg, output_pt_path, gpu_lock, log, cancel_event) -> dict`，內部做 stage→train→calibrate→copy→metrics，最後把 .pt 寫到 `output_pt_path`。

`run_training_pipeline` 的 unit 迴圈改為呼叫新函式，把 `bundle_dir / f"{unit_label}.pt"` 當 `output_pt_path` 傳入。

- [ ] **Step 1: 在 `capi_train_new.py` 加新函式 `train_single_submodel`**

放在 `run_training_pipeline` 上方（line 879 之前），先讀完當前檔案找好位置：

```python
def train_single_submodel(
    db: TrainingDB,
    job_id: str,
    lighting: str,
    zone: str,
    cfg: TrainingConfig,
    output_pt_path: Path,
    gpu_lock=None,
    log: Callable[[str], None] = print,
    cancel_event=None,
) -> Dict:
    """訓練單一 (lighting, zone) unit。

    回傳 dict 包含：
      - threshold: float (永遠 = DEFAULT_THRESHOLD = 0.5，calibrate 寫死)
      - metrics: dict (compute_unit_metrics 結果)
      - tile_count: int (訓練用 tile 數)
      - ng_count: int (NG 數)
      - ng_used: "zone" | "fallback"
      - used_tile_ids: list[int] (該次訓練實際送進的 tile_pool.id)
      - elapsed_seconds: int
      - size_bytes: int (.pt 檔大小)

    output_pt_path 會被原子覆蓋（同目錄 .pt.tmp → os.replace）。失敗時不動到原檔。
    """
    from contextlib import nullcontext
    import os
    gpu_ctx = gpu_lock if gpu_lock is not None else nullcontext()
    unit_label = f"{lighting}-{zone}"

    train_tiles = db.list_tile_pool(job_id, lighting=lighting, zone=zone,
                                    source="ok", decision="accept")
    ng_all = db.list_tile_pool(job_id, lighting=lighting,
                               source="ng", decision="accept")
    ng_for_zone = [t for t in ng_all if t.get("zone") in (zone, None)]
    if len(ng_for_zone) < MIN_NG_PER_ZONE:
        log(f"{unit_label}: zone NG 僅 {len(ng_for_zone)} (<{MIN_NG_PER_ZONE})，"
            f"退回全部 NG ({len(ng_all)})")
        ng_tiles = ng_all
        ng_used = "fallback"
    else:
        ng_tiles = ng_for_zone
        ng_used = "zone"

    if len(train_tiles) < MIN_TRAIN_TILES:
        raise RuntimeError(
            f"{unit_label}: tile 不足 ({len(train_tiles)} < {MIN_TRAIN_TILES})"
        )

    used_tile_ids = sorted(int(t["id"]) for t in train_tiles)
    unit_start = time.monotonic()

    with gpu_ctx:
        staging = Path(".tmp/training_staging") / job_id / unit_label
        run_root = Path(".tmp/training_runs") / job_id / unit_label
        try:
            stage_dataset(staging,
                          [Path(t["source_path"]) for t in train_tiles],
                          [Path(t["source_path"]) for t in ng_tiles])
            model_pt = train_one_patchcore(staging, run_root, unit_label, cfg, log=log)

            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("training cancelled by user")

            train_max, train_scores, ng_scores = _calibrate_from_model(
                model_pt,
                [Path(t["source_path"]) for t in train_tiles],
                [Path(t["source_path"]) for t in ng_tiles],
            )
            threshold = calibrate_threshold(ng_scores, train_max)

            output_pt_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = output_pt_path.with_suffix(output_pt_path.suffix + ".tmp")
            shutil.copy2(model_pt, tmp_path)
            os.replace(tmp_path, output_pt_path)
            size = output_pt_path.stat().st_size

            metrics = compute_unit_metrics(
                train_max, ng_scores, threshold, train_scores=train_scores,
            )
            metrics["train_count"] = len(train_tiles)
            metrics["ng_used"] = ng_used
            elapsed = time.monotonic() - unit_start
            metrics["elapsed_seconds"] = int(elapsed)

            return {
                "threshold": round(threshold, 4),
                "metrics": metrics,
                "tile_count": len(train_tiles),
                "ng_count": len(ng_tiles),
                "ng_used": ng_used,
                "used_tile_ids": used_tile_ids,
                "elapsed_seconds": int(elapsed),
                "size_bytes": size,
            }
        finally:
            shutil.rmtree(run_root, ignore_errors=True)
            shutil.rmtree(staging, ignore_errors=True)
            tmp_leftover = output_pt_path.with_suffix(output_pt_path.suffix + ".tmp")
            if tmp_leftover.exists():
                try:
                    tmp_leftover.unlink()
                except OSError:
                    pass
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
```

- [ ] **Step 2: 重構 `run_training_pipeline` 用新函式**

把 `run_training_pipeline` 的 unit 迴圈內 `with gpu_ctx:` 區塊（line 952-1018）改為呼叫 `train_single_submodel`。

替換 line 952-1018 為：

```python
        try:
            output_pt = bundle_dir / f"{unit_label}.pt"
            result = train_single_submodel(
                db=db, job_id=job_id, lighting=lighting, zone=zone,
                cfg=cfg, output_pt_path=output_pt,
                gpu_lock=gpu_lock, log=log, cancel_event=cancel_event,
            )

            thresholds[lighting][zone] = result["threshold"]
            tiles_per_unit[unit_label] = {"train": result["tile_count"], "ng": result["ng_count"]}
            model_files[unit_label] = {"path": output_pt.name, "size_bytes": result["size_bytes"]}

            metrics = result["metrics"]
            metrics["used_tile_ids"] = result["used_tile_ids"]
            unit_metrics[unit_label] = metrics

            success_units += 1
            succeeded_units.add((lighting, zone))
            completed_durations.append(result["elapsed_seconds"])
            eta = _eta_text()
            caught = metrics.get("ng_caught_count", 0)
            ng_n = metrics.get("ng_count", 0)
            auroc = metrics.get("auroc")
            auroc_str = f", AUROC={auroc:.3f}({metrics.get('auroc_grade','')})" if auroc is not None else ""
            log(
                f"[{idx}/10] {unit_label}: ✓ done | {result['elapsed_seconds']}s, "
                f"threshold={result['threshold']:.4f}, size={result['size_bytes']/1e6:.1f}MB, "
                f"ng_caught={caught}/{ng_n}{auroc_str}"
                + (f" | {eta}" if eta else "")
            )
        except Exception as e:
            completed_durations.append(time.monotonic() - unit_start)
            log(f"[{idx}/10] {unit_label}: ✗ 訓練失敗: {e}")
            for line in traceback.format_exc().rstrip().splitlines()[-8:]:
                log(f"  {line}")
            # 不增加 success_units，繼續下一個 unit
```

注意：`unit_start` 變數仍要在迴圈頂部保留供 except 計算。原 line 930 `unit_start = time.monotonic()` 保留。

- [ ] **Step 3: Sanity check：訓練 wizard 跑通**

完整重訓會吃 GPU 數十分鐘，不適合每次都跑。改用 Python import sanity check：

```bash
python -c "from capi_train_new import train_single_submodel, run_training_pipeline; print('imports ok')"
```

預期 Output：`imports ok`

如果 import 失敗，立即修。

- [ ] **Step 4: Commit**

```bash
git add capi_train_new.py
git commit -m "refactor(train): 抽出 train_single_submodel 公用函式

run_training_pipeline 的 unit 迴圈改為呼叫新函式；新函式可被
單獨呼叫，給 bundle 子模型重訓用。
unit metrics 新增 used_tile_ids 欄位（後續 UI 判斷『有未訓練修改』
要靠它對比目前 tile_pool.decision='accept' 的 id 集合）。"
```

---

## Task 2: capi_model_registry.py — manifest 操作公用函式

**目的：** 給 worker 與 UI 用的 manifest 讀寫工具，封裝 `submodel_history` 與 `pending_changes` 計算。

**Files:**
- Modify: `capi_model_registry.py`（檔尾加新函式）

- [ ] **Step 1: 加 `append_submodel_history`**

在 `update_threshold` 函式之後（line 304 之後）加：

```python
def _read_manifest(bundle_dir: Path) -> dict:
    """讀 bundle 的 manifest.json；不存在或解析錯誤回空 dict。"""
    p = bundle_dir / "manifest.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("manifest.json 解析失敗：%s", p)
        return {}


def _write_manifest(bundle_dir: Path, data: dict) -> None:
    p = bundle_dir / "manifest.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    import os
    os.replace(tmp, p)


def append_submodel_history(
    bundle_dir: Path, lighting: str, zone: str, entry: dict,
) -> None:
    """把單次訓練 entry 追加到 manifest.submodel_history[lighting-zone]。

    若 manifest 不存在 submodel_history 欄位則新建。entry 預期至少包含：
    trained_at、tile_count_used、auroc、used_tile_ids、kind。
    """
    unit_label = f"{lighting}-{zone}"
    manifest = _read_manifest(bundle_dir)
    history = manifest.setdefault("submodel_history", {})
    history.setdefault(unit_label, []).append(entry)
    manifest["last_retrained_at"] = entry.get("trained_at", manifest.get("last_retrained_at"))
    _write_manifest(bundle_dir, manifest)


def get_used_tile_ids(bundle_dir: Path, lighting: str, zone: str) -> Optional[set]:
    """讀 manifest 取得該 unit「上次訓練時使用的 tile_pool.id 集合」。

    優先順序：
    1. submodel_history[unit_label] 最新 entry 的 used_tile_ids
    2. 退回 manifest.unit_metrics[unit_label].used_tile_ids（初次訓練的記錄）
    3. 都沒有 → None（表示舊 bundle，無法判斷差異）
    """
    unit_label = f"{lighting}-{zone}"
    manifest = _read_manifest(bundle_dir)

    history = (manifest.get("submodel_history") or {}).get(unit_label) or []
    if history:
        ids = history[-1].get("used_tile_ids")
        if ids is not None:
            return set(int(x) for x in ids)

    unit_metrics = (manifest.get("unit_metrics") or {}).get(unit_label) or {}
    ids = unit_metrics.get("used_tile_ids")
    if ids is not None:
        return set(int(x) for x in ids)
    return None


def get_pending_change_count(
    db, bundle: dict, lighting: str, zone: str,
) -> int:
    """回傳該 unit 目前「decision=accept」tile id 集合與上次訓練集合的差異數。

    差異 = (新增 accept 的) ∪ (上次訓練用過但現在被 reject 的)。
    舊 bundle（manifest 沒記錄 used_tile_ids）退化策略：回傳目前 reject 的 tile 數。
    無 job_id（訓練資料已刪）回 0。
    """
    job_id = bundle.get("job_id") or ""
    if not job_id:
        return 0
    bundle_dir = Path(bundle["bundle_path"])

    current_accept = {
        int(t["id"]) for t in db.list_tile_pool(
            job_id, lighting=lighting, zone=zone, source="ok", decision="accept",
        )
    }
    last_used = get_used_tile_ids(bundle_dir, lighting, zone)

    if last_used is None:
        # 舊 bundle 退化路徑：用「現在被 reject 的數量」當差異訊號
        rejected = db.list_tile_pool(
            job_id, lighting=lighting, zone=zone, source="ok", decision="reject",
        )
        return len(rejected)

    return len(current_accept.symmetric_difference(last_used))


def get_pending_change_summary_for_bundle(db, bundle: dict) -> dict:
    """所有 lighting+zone 的 pending change 數量，給列表頁徽章用。

    回傳 {(lighting, zone): count, ...}，過濾掉 count == 0 的。
    """
    out = {}
    for lighting in ("G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD"):
        for zone in ("inner", "edge"):
            n = get_pending_change_count(db, bundle, lighting, zone)
            if n > 0:
                out[(lighting, zone)] = n
    return out
```

注意 file 開頭已 import `from capi_train_new import ZONES`。如果新函式內部用到 `LIGHTINGS`，請改為直接從 `capi_train_new import LIGHTINGS`，或在新函式內 hardcode（如上）。`LIGHTINGS` 與 `ZONES` 集合穩定不需動態化。

- [ ] **Step 2: 修改 `get_bundle_detail` 多回 `pending_changes`**

在 `get_bundle_detail` 函式內，line 81 `bundle["training_data"] = ...` 之後加一行：

```python
    bundle["pending_changes"] = get_pending_change_summary_for_bundle(db, bundle)
```

- [ ] **Step 3: 寫單元測試**

新建 `tests/test_submodel_retrain.py`：

```python
"""capi_model_registry 與 capi_train_new 中與單子模型重訓相關的純函式測試。

無需啟動 web server / GPU；用 tempdir 與假 DB 物件做 isolated 測試。
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_model_registry import (
    append_submodel_history,
    get_used_tile_ids,
    get_pending_change_count,
)


def _write_manifest(bundle_dir: Path, data: dict) -> None:
    (bundle_dir / "manifest.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8",
    )


def test_append_submodel_history_creates_field(tmp_path):
    _write_manifest(tmp_path, {"machine_id": "M1"})

    entry = {"trained_at": "2026-05-06T10:00:00", "tile_count_used": 100,
             "auroc": 0.95, "used_tile_ids": [1, 2, 3], "kind": "retrain"}
    append_submodel_history(tmp_path, "G0F00000", "edge", entry)

    data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert data["submodel_history"]["G0F00000-edge"] == [entry]
    assert data["last_retrained_at"] == "2026-05-06T10:00:00"


def test_append_submodel_history_appends_existing(tmp_path):
    initial = {"trained_at": "2026-05-01T10:00:00", "tile_count_used": 100,
               "auroc": 0.93, "used_tile_ids": [1, 2], "kind": "initial"}
    _write_manifest(tmp_path, {
        "submodel_history": {"G0F00000-edge": [initial]},
    })

    new_entry = {"trained_at": "2026-05-06T10:00:00", "tile_count_used": 95,
                 "auroc": 0.96, "used_tile_ids": [1, 3], "kind": "retrain"}
    append_submodel_history(tmp_path, "G0F00000", "edge", new_entry)

    data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    history = data["submodel_history"]["G0F00000-edge"]
    assert len(history) == 2
    assert history[1] == new_entry


def test_get_used_tile_ids_from_history(tmp_path):
    _write_manifest(tmp_path, {
        "submodel_history": {
            "G0F00000-edge": [
                {"used_tile_ids": [1, 2]},
                {"used_tile_ids": [1, 3, 5]},
            ]
        }
    })
    assert get_used_tile_ids(tmp_path, "G0F00000", "edge") == {1, 3, 5}


def test_get_used_tile_ids_fallback_to_unit_metrics(tmp_path):
    _write_manifest(tmp_path, {
        "unit_metrics": {
            "G0F00000-edge": {"used_tile_ids": [10, 20]},
        }
    })
    assert get_used_tile_ids(tmp_path, "G0F00000", "edge") == {10, 20}


def test_get_used_tile_ids_none_when_missing(tmp_path):
    _write_manifest(tmp_path, {"machine_id": "M1"})
    assert get_used_tile_ids(tmp_path, "G0F00000", "edge") is None


def test_pending_change_count_diff(tmp_path):
    _write_manifest(tmp_path, {
        "submodel_history": {
            "G0F00000-edge": [{"used_tile_ids": [1, 2, 3]}],
        }
    })
    db = MagicMock()
    # 目前 accept = {1, 2, 4}：相比上次 {1, 2, 3} 差異是 {3, 4}（2 張）
    db.list_tile_pool.return_value = [{"id": 1}, {"id": 2}, {"id": 4}]

    bundle = {"job_id": "j1", "bundle_path": str(tmp_path)}
    assert get_pending_change_count(db, bundle, "G0F00000", "edge") == 2


def test_pending_change_count_legacy_uses_reject_count(tmp_path):
    """舊 bundle 沒有 used_tile_ids → 退化用 reject 數量。"""
    _write_manifest(tmp_path, {"machine_id": "M1"})
    db = MagicMock()

    def fake_list(job_id, **filters):
        if filters.get("decision") == "reject":
            return [{"id": 5}, {"id": 6}]
        return [{"id": 1}, {"id": 2}]
    db.list_tile_pool.side_effect = fake_list

    bundle = {"job_id": "j1", "bundle_path": str(tmp_path)}
    assert get_pending_change_count(db, bundle, "G0F00000", "edge") == 2


def test_pending_change_count_no_job_id(tmp_path):
    db = MagicMock()
    bundle = {"job_id": "", "bundle_path": str(tmp_path)}
    assert get_pending_change_count(db, bundle, "G0F00000", "edge") == 0
```

- [ ] **Step 2.5: 跑 unit tests**

```bash
python -m pytest tests/test_submodel_retrain.py -v
```

預期：8 個測試全 pass。任何 fail 都立即修。

- [ ] **Step 3: Commit**

```bash
git add capi_model_registry.py tests/test_submodel_retrain.py
git commit -m "feat(registry): manifest submodel_history 與 pending_changes 工具

加 append_submodel_history、get_used_tile_ids、get_pending_change_count、
get_pending_change_summary_for_bundle。
get_bundle_detail 多回 pending_changes 給 UI 渲染徽章用。
舊 bundle 沒記錄 used_tile_ids 時退化為『目前 reject 數量』判定。"
```

---

## Task 3: capi_inference.py — `reload_submodel`

**目的：** 提供「重訓完一個 .pt 後叫 inferencer 從快取丟掉舊模型」的 API。

**Files:**
- Modify: `capi_inference.py:6228-6241`（`_get_model_for` 周邊）

- [ ] **Step 1: 加 `reload_submodel` method**

在 `_get_model_for` 之後（找到 line 6241 後）加：

```python
    def reload_submodel(self, machine_id: str, lighting: str, zone: str) -> bool:
        """重訓完成後丟掉 cache 中的舊 model，下次 inference 自動 lazy reload。

        回 True 表示有被踢掉舊 cache；False 代表本來就沒載入過（也不需要 reload）。
        """
        key = (machine_id, lighting, zone)
        if key in self._model_cache_v2:
            del self._model_cache_v2[key]
            logger.info("[v2] 已將 cache key %s pop，下次推論會 lazy reload", key)
            return True
        logger.info("[v2] cache 中無 key %s，不需要 reload", key)
        return False
```

- [ ] **Step 2: Sanity check**

```bash
python -c "from capi_inference import CAPIInferencer; assert hasattr(CAPIInferencer, 'reload_submodel'), 'missing reload_submodel'; print('ok')"
```

預期：`ok`

- [ ] **Step 3: Commit**

```bash
git add capi_inference.py
git commit -m "feat(infer): reload_submodel 支援單子模型熱替換

從 _model_cache_v2 pop 對應 key，下次 _get_model_for 觸發 lazy reload。
給 bundle 重訓完成後的 .pt 替換用。"
```

---

## Task 4: capi_web.py — `/api/models/<id>/tiles/decision`

**目的：** 讓使用者在 bundle detail 頁切換 OK tile 的 accept/reject。

**Files:**
- Modify: `capi_web.py`（POST routing 區 + 加 handler）

- [ ] **Step 1: 在 POST router 加路由**

定位 `_handle_train_new_tiles_decision` 的 routing 行（搜 `tiles/decision`）。在它之後（同 `do_POST` 函式內）加：

```python
            elif path.startswith("/api/models/") and path.endswith("/tiles/decision"):
                self._handle_models_tiles_decision()
```

請在現有 models routing 區塊（看 `/api/models/.../delete` 那附近）加，便於閱讀。

- [ ] **Step 2: 加 handler**

放在 `_handle_models_training_data_delete` 之後（line ~5912 之後）：

```python
    def _handle_models_tiles_decision(self):
        """POST /api/models/<id>/tiles/decision
        body: {"tile_ids": [int, ...], "decision": "accept"|"reject"}

        只允許切換 OK tile 的 decision；NG tile 嘗試操作 → 400。
        """
        parts = self.path.split("/")
        try:
            bundle_id = int(parts[3])
        except (ValueError, IndexError):
            self._send_json({"error": "invalid bundle id"}, status=400)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            self._send_json({"error": "invalid JSON"}, status=400)
            return

        tile_ids = body.get("tile_ids", [])
        decision = body.get("decision")
        if not tile_ids or decision not in ("accept", "reject"):
            self._send_json({"error": "tile_ids 與 decision 必填"}, status=400)
            return
        try:
            tile_ids = [int(x) for x in tile_ids]
        except (TypeError, ValueError):
            self._send_json({"error": "tile_ids 必須為整數陣列"}, status=400)
            return

        db = self._capi_server_instance.database
        bundle = db.get_model_bundle(bundle_id)
        if not bundle:
            self._send_json({"error": "bundle not found"}, status=404)
            return
        job_id = bundle.get("job_id") or ""
        if not job_id:
            self._send_json({"error": "此 bundle 無關聯 job_id"}, status=400)
            return

        # 只允許動 OK tile：用 source='ok' 撈一次當前 job 的所有 OK tile id 做白名單
        ok_ids = {int(t["id"]) for t in db.list_tile_pool(job_id, source="ok")}
        bad = [i for i in tile_ids if i not in ok_ids]
        if bad:
            self._send_json({"error": f"tile_ids 含非 OK tile（NG tile 不可動）: {bad[:5]}"},
                            status=400)
            return

        db.update_tile_decisions(job_id, tile_ids, decision)
        self._send_json({"ok": True, "updated": len(tile_ids)})
```

- [ ] **Step 3: 修 `_handle_models_training_tiles` 回傳加 `decision`**

定位 `_handle_models_training_tiles` 函式內 `for t in page` 迴圈（line 5879-5888）。在每個 tile dict 內加 `"decision": t.get("decision")`：

```python
        for t in page:
            out.append({
                "id": t.get("id"),
                "lighting": t.get("lighting"),
                "zone": t.get("zone"),
                "source": t.get("source"),
                "decision": t.get("decision"),
                "thumb_url": self._train_new_thumb_url(t.get("thumb_path")),
            })
```

（`decision` 可能已經回傳，看現有代碼確認；若有就不用改。讀取後確定是否已經包含 `decision` 欄位再決定。）

- [ ] **Step 4: Sanity check（手動）**

啟動 server：
```bash
python capi_server.py --config server_config_local.yaml
```

用 curl 測 happy path（要先有 bundle，例如取 bundle_id=1）：
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"tile_ids":[],"decision":"accept"}' \
  http://localhost:8080/api/models/1/tiles/decision
```

預期：`{"error":"tile_ids 與 decision 必填"}`，status 400。

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"tile_ids":[1,2],"decision":"foo"}' \
  http://localhost:8080/api/models/1/tiles/decision
```

預期：同上 400。

正常路徑（用真實 tile id；若沒有測試環境跳過）：用 GET `/api/models/<id>/training_tiles?source=ok&lighting=G0F00000&zone=inner` 撈 id，再 POST decision。

- [ ] **Step 5: Commit**

```bash
git add capi_web.py
git commit -m "feat(web): /api/models/<id>/tiles/decision 切 OK tile accept/reject

NG tile 嘗試操作回 400；訓練資料已刪（無 job_id）回 400。
training_tiles 回傳補 decision 欄位給前端渲染狀態。"
```

---

## Task 5: capi_web.py — `_submodel_retrain_state` 與啟動 API

**目的：** 加 worker 共用 state，提供啟動 retrain job 的 API（含併發保護）。

**Files:**
- Modify: `capi_web.py`（class state 區、POST router、handler）

- [ ] **Step 1: 加 class-level state**

在 `_train_new_state = {...}`（line 176-190）下方加：

```python
    # 單子模型重訓 state（一次只允許一個 job）
    _submodel_retrain_state: dict = {
        "lock": threading.Lock(),
        "job": None,
        # job dict 結構（running 中時）：
        # {
        #   bundle_id: int,
        #   lighting: str,
        #   zone: str,
        #   state: "running" | "completed" | "failed",
        #   step: "stage" | "train" | "metrics" | "swap" | "reload" | "done",
        #   started_at: str (ISO),
        #   log_lines: list[str],
        #   summary: dict | None,    # {auroc_old, auroc_new, tile_count_old, tile_count_new}
        #   error: str | None,
        # }
    }
```

- [ ] **Step 2: 加 POST router**

在 `_handle_models_tiles_decision` 路由之後加：

```python
            elif path.startswith("/api/models/") and path.endswith("/retrain_submodel"):
                self._handle_models_retrain_submodel()
            elif path.startswith("/api/models/") and path.endswith("/retrain_status"):
                # 注意：retrain_status 是 GET，這個 elif 是錯的！見下面正確 GET routing。
                pass
```

—— 上面是錯誤示範，**回到 GET router 處**（搜 `/api/models/.../detail`）加：

```python
            elif path.startswith("/api/models/") and path.endswith("/retrain_status"):
                self._handle_models_retrain_status()
```

- [ ] **Step 3: 加 `_handle_models_retrain_submodel` handler**

放在 `_handle_models_tiles_decision` 之後：

```python
    def _handle_models_retrain_submodel(self):
        """POST /api/models/<id>/retrain_submodel
        body: {"lighting": str, "zone": "inner"|"edge"}

        啟動 worker thread 重訓單一子模型。已有 retrain job 跑 → 409。
        """
        from capi_train_new import LIGHTINGS, ZONES

        parts = self.path.split("/")
        try:
            bundle_id = int(parts[3])
        except (ValueError, IndexError):
            self._send_json({"error": "invalid bundle id"}, status=400)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            self._send_json({"error": "invalid JSON"}, status=400)
            return

        lighting = body.get("lighting")
        zone = body.get("zone")
        if lighting not in LIGHTINGS:
            self._send_json({"error": f"lighting 必須為 {LIGHTINGS}"}, status=400)
            return
        if zone not in ZONES:
            self._send_json({"error": f"zone 必須為 {ZONES}"}, status=400)
            return

        db = self._capi_server_instance.database
        bundle = db.get_model_bundle(bundle_id)
        if not bundle:
            self._send_json({"error": "bundle not found"}, status=404)
            return
        if not bundle.get("job_id"):
            self._send_json({"error": "此 bundle 無關聯 job_id（訓練資料已刪），無法重訓"},
                            status=400)
            return

        state = CAPIWebHandler._submodel_retrain_state
        with state["lock"]:
            current = state.get("job")
            if current and current.get("state") == "running":
                self._send_json({"error": "已有重訓 job 進行中，請等待完成",
                                 "job": current}, status=409)
                return

            state["job"] = {
                "bundle_id": bundle_id,
                "lighting": lighting,
                "zone": zone,
                "state": "running",
                "step": "stage",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "log_lines": [],
                "summary": None,
                "error": None,
            }

        thread = threading.Thread(
            target=self._submodel_retrain_worker,
            args=(bundle_id, lighting, zone),
            daemon=True,
            name=f"submodel-retrain-{bundle_id}-{lighting}-{zone}",
        )
        thread.start()

        self._send_json({"ok": True, "bundle_id": bundle_id,
                         "lighting": lighting, "zone": zone})
```

- [ ] **Step 4: 加 `_handle_models_retrain_status` handler**

放在 `_handle_models_retrain_submodel` 之後：

```python
    def _handle_models_retrain_status(self):
        """GET /api/models/<id>/retrain_status?tail=200

        回目前 retrain job 的狀態與末 N 行 log。job 為空時回 {"job": null}。
        """
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(self.path).query)
        try:
            tail = max(0, min(int((qs.get("tail") or ["200"])[0]), 1000))
        except ValueError:
            tail = 200

        state = CAPIWebHandler._submodel_retrain_state
        with state["lock"]:
            job = state.get("job")
            if job is None:
                self._send_json({"job": None})
                return
            # 淺拷貝 + 截斷 log
            out = dict(job)
            out["log_lines"] = list(job["log_lines"][-tail:])

        self._send_json({"job": out})
```

- [ ] **Step 5: Sanity check**

啟動 server，測試：

```bash
# 沒有 bundle 1：
curl -X POST -H "Content-Type: application/json" \
  -d '{"lighting":"G0F00000","zone":"edge"}' \
  http://localhost:8080/api/models/9999/retrain_submodel
```

預期：`{"error":"bundle not found"}`，status 404。

```bash
# 不合法 lighting：
curl -X POST -H "Content-Type: application/json" \
  -d '{"lighting":"FOO","zone":"edge"}' \
  http://localhost:8080/api/models/1/retrain_submodel
```

預期：400 含 lighting 錯誤訊息。

```bash
# Status 還沒跑過：
curl http://localhost:8080/api/models/1/retrain_status
```

預期：`{"job":null}`。

- [ ] **Step 6: Commit**

```bash
git add capi_web.py
git commit -m "feat(web): bundle 單子模型 retrain 啟動與狀態 API

新增 _submodel_retrain_state、POST /api/models/<id>/retrain_submodel、
GET /api/models/<id>/retrain_status。已有 job 跑 → 409；訓練資料已刪 → 400。
worker 實作在下個 commit。"
```

---

## Task 6: capi_web.py — `_submodel_retrain_worker`

**目的：** 把訓練→替換→reload→寫 manifest 全跑完。沿用 Task 1 的 `train_single_submodel`。

**Files:**
- Modify: `capi_web.py`

- [ ] **Step 1: 加 worker method**

放在 `_handle_models_retrain_status` 之後：

```python
    def _submodel_retrain_worker(self, bundle_id: int, lighting: str, zone: str):
        """背景 thread：執行單子模型重訓全流程。

        步驟：stage → train → metrics → swap → reload → done。任一步失敗
        都更新 state["job"] state="failed" + error，並保留 .pt 與 manifest 不動。
        """
        import traceback
        from capi_train_new import train_single_submodel, TrainingConfig
        from capi_model_registry import append_submodel_history, get_used_tile_ids

        state = CAPIWebHandler._submodel_retrain_state

        def _set_step(step: str):
            with state["lock"]:
                if state["job"] is not None:
                    state["job"]["step"] = step

        def _log(msg: str):
            ts = datetime.now().strftime("%H:%M:%S")
            with state["lock"]:
                if state["job"] is not None:
                    state["job"]["log_lines"].append(f"[{ts}] {msg}")
                    if len(state["job"]["log_lines"]) > 500:
                        state["job"]["log_lines"] = state["job"]["log_lines"][-500:]

        try:
            db = self._capi_server_instance.database
            bundle = db.get_model_bundle(bundle_id)
            if not bundle:
                raise RuntimeError(f"bundle {bundle_id} 已不存在")
            job_id = bundle["job_id"]
            bundle_dir = Path(bundle["bundle_path"])
            machine_id = bundle["machine_id"]
            unit_label = f"{lighting}-{zone}"
            output_pt = bundle_dir / f"{unit_label}.pt"

            _log(f"開始重訓 {unit_label} (bundle_id={bundle_id})")

            # 取舊 AUROC / tile 數做 summary 比對
            from capi_model_registry import _read_manifest
            old_manifest = _read_manifest(bundle_dir)
            old_unit_metrics = (old_manifest.get("unit_metrics") or {}).get(unit_label) or {}
            old_history = (old_manifest.get("submodel_history") or {}).get(unit_label) or []
            if old_history:
                old_auroc = old_history[-1].get("auroc")
                old_tile_count = old_history[-1].get("tile_count_used")
            else:
                old_auroc = old_unit_metrics.get("auroc")
                old_tile_count = old_unit_metrics.get("train_count")

            # 取 TrainingConfig：用既有 bundle 訓練時的 patchcore_params 與 backbone_cache
            patchcore_params = (old_manifest.get("patchcore_params") or {})
            cfg = TrainingConfig(
                machine_id=machine_id,
                panel_paths=[],
                over_review_root=Path(".tmp/_unused"),
                batch_size=patchcore_params.get("batch_size", 32),
                image_size=tuple(patchcore_params.get("image_size", (256, 256))),
                coreset_ratio=patchcore_params.get("coreset_ratio", 0.1),
                max_epochs=patchcore_params.get("max_epochs", 1),
                inner_panels=patchcore_params.get("inner_panels", []),
            )
            # backbone_cache_dir / required_backbones / output_root 沿用 dataclass 預設值

            _set_step("stage")
            _log("準備訓練資料...")

            _set_step("train")
            _log("訓練中（含 stage_dataset → train_one_patchcore → calibrate）...")
            gpu_lock = self._capi_server_instance._gpu_lock
            result = train_single_submodel(
                db=db, job_id=job_id, lighting=lighting, zone=zone,
                cfg=cfg, output_pt_path=output_pt,
                gpu_lock=gpu_lock, log=_log,
            )

            _set_step("metrics")
            new_auroc = result["metrics"].get("auroc")
            new_tile_count = result["tile_count"]
            _log(f"訓練完成：tile={new_tile_count}, AUROC={new_auroc}")

            _set_step("swap")
            # train_single_submodel 已 atomic 寫好 output_pt，不需另做 swap
            _log(f"已替換 {output_pt}")

            # 寫 manifest history
            entry = {
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "tile_count_used": new_tile_count,
                "auroc": new_auroc,
                "used_tile_ids": result["used_tile_ids"],
                "kind": "retrain",
                "ng_used": result["ng_used"],
            }
            append_submodel_history(bundle_dir, lighting, zone, entry)
            _log("manifest history 已更新")

            _set_step("reload")
            inferencer = self._capi_server_instance.inferencers.get(machine_id)
            if inferencer is None:
                _log(f"[v2] 機台 {machine_id} 無 inferencer cache，跳過 reload（下次首次推論會載入新模型）")
            else:
                inferencer.reload_submodel(machine_id, lighting, zone)
                _log(f"[v2] 已通知 inferencer reload {machine_id}/{lighting}/{zone}")

            _set_step("done")
            with state["lock"]:
                state["job"]["state"] = "completed"
                state["job"]["summary"] = {
                    "auroc_old": old_auroc,
                    "auroc_new": new_auroc,
                    "tile_count_old": old_tile_count,
                    "tile_count_new": new_tile_count,
                }
            _log(f"✓ 重訓完成")

        except Exception as e:
            tb = traceback.format_exc()
            _log(f"✗ 失敗: {e}")
            for line in tb.rstrip().splitlines()[-8:]:
                _log(f"  {line}")
            with state["lock"]:
                if state["job"] is not None:
                    state["job"]["state"] = "failed"
                    state["job"]["error"] = str(e)
            logger.error("submodel retrain worker failed: %s", e, exc_info=True)
```

注意：
- `_capi_server_instance._gpu_lock` 是 `capi_server.py:734` 之後初始化的；確認該屬性名（搜 `_gpu_lock` 在 `capi_server.py`）。如果命名不同（例如 `gpu_lock` 或 `self.gpu_lock`），改用實際名稱。
- `_read_manifest` 是 Task 2 加在 `capi_model_registry.py` 的私有函式，這裡 import 用 `from capi_model_registry import _read_manifest`，是 ok 的（Python convention 允許）。若想更乾淨，把它改 public。

- [ ] **Step 2: 確認 `_gpu_lock` 命名**

```bash
python -c "from capi_server import CAPIServer; import inspect; print([a for a in dir(CAPIServer) if 'lock' in a.lower()])"
```

如果輸出包含 `_gpu_lock` 即可；若是其他名稱，把 worker 內 `self._capi_server_instance._gpu_lock` 改為實際名稱。

如果 CAPIServer 用 `__init__` 動態設置 attr（class-level 看不到），改檢查實例：可以暫時跳過此 sanity，留待 step 3 整合驗證。

- [ ] **Step 3: 整合驗證（手動，需要既有 bundle 與訓練資料）**

啟動 server：
```bash
python capi_server.py --config server_config_local.yaml
```

如果你的本機沒有可用 bundle，跳過此 step，留待 Task 12 整合驗收。

如果有可用 bundle（bundle_id=1，G0F00000-edge）：
```bash
# 起動重訓
curl -X POST -H "Content-Type: application/json" \
  -d '{"lighting":"G0F00000","zone":"edge"}' \
  http://localhost:8080/api/models/1/retrain_submodel

# 輪詢狀態
watch -n 2 'curl -s http://localhost:8080/api/models/1/retrain_status | python -m json.tool | head -40'
```

預期：state 從 running → completed，summary 含 AUROC/tile_count 對照。

- [ ] **Step 4: Commit**

```bash
git add capi_web.py
git commit -m "feat(web): submodel retrain worker 完整流程

stage → train → metrics → swap → reload → done。
沿用 capi_server.inferencers[machine_id].reload_submodel 熱替換。
失敗時 .pt 與 manifest 完全不動，job state 標 failed + error。"
```

---

## Task 7: capi_web.py — `_handle_models_detail` 回傳 `pending_changes`

**目的：** UI 渲染變更指示條與列表頁徽章需要這些數字。

**Files:**
- Modify: `capi_web.py`（找 `_handle_models_detail`）

- [ ] **Step 1: 確認 detail handler 行為**

```bash
grep -n "_handle_models_detail\|def _handle_models_detail" capi_web.py | head
```

讀該函式（約 line 5723 附近）。`get_bundle_detail` 在 Task 2 已修改成回傳含 `pending_changes` 的 dict，但 detail handler 可能會 filter 欄位再吐出去。檢查並確保 `pending_changes` 透出。

- [ ] **Step 2: 修改 handler（如有需要）**

如果 detail handler 直接回傳 `bundle` dict：不用改任何東西。

如果它選擇性挑欄位回傳，補上 `pending_changes` 欄位。`pending_changes` 結構是 `{(lighting, zone): count}`，JSON 不能用 tuple 當 key，要轉為：

```python
def _serialize_pending_changes(pc: dict) -> list:
    return [{"lighting": l, "zone": z, "count": c}
            for (l, z), c in pc.items()]
```

放在 handler 旁，回傳前對 `pending_changes` 做轉換：

```python
detail["pending_changes"] = _serialize_pending_changes(detail.get("pending_changes") or {})
```

也同步修改 `get_bundle_detail` 的呼叫處，或在 handler 內處理。**選一個地方，不要兩處重複處理**。建議在 handler 處理（避免改變 `get_bundle_detail` 的 dict 結構讓其他用法 break）。

- [ ] **Step 3: Sanity check**

```bash
curl -s http://localhost:8080/api/models/1/detail | python -m json.tool | grep -A3 pending_changes
```

預期：看到 `pending_changes: []` 或 `pending_changes: [{"lighting": ..., "zone": ..., "count": ...}]`。

- [ ] **Step 4: Commit**

```bash
git add capi_web.py
git commit -m "feat(web): bundle detail 回傳 pending_changes 給 UI

格式：[{lighting, zone, count}, ...]。供前端在 tile 區與列表頁渲染
『有 N 張 tile 修改未訓練』指示。"
```

---

## Task 8: templates/models.html — tile accept/reject 切換

**目的：** 既有 tile 縮圖網格加上點擊或鍵盤切換 decision 狀態的能力。

**Files:**
- Modify: `templates/models.html`

- [ ] **Step 1: 讀目前 tile 區實作**

```bash
grep -n "training_tiles\|training-tiles\|tile-grid\|fetch.*models" templates/models.html | head -30
```

理解現在 tile 怎麼呈現（看似是 modal 內透過 `/api/models/<id>/training_tiles` 動態載入）。讀相關函式約 100 行。

- [ ] **Step 2: 修改 tile 縮圖渲染加 data-id / data-decision**

找到渲染 tile 縮圖的函式（可能名為 `renderTrainingTiles` 或在 detail modal js 內 inline）。每張 tile 改為：

```html
<div class="tile-thumb" data-tile-id="${t.id}" data-decision="${t.decision || 'accept'}">
  <img src="${t.thumb_url}" alt="">
</div>
```

對應 CSS（加在現有 style 區）：

```css
.tile-thumb { position: relative; cursor: pointer; }
.tile-thumb[data-decision="reject"] {
  filter: brightness(0.4) saturate(0.5);
  outline: 2px solid #f38ba8;
}
.tile-thumb.modified::after {
  content: "✎"; position: absolute; top: 2px; right: 4px;
  background: rgba(243,139,168,0.9); color: #1e1e2e;
  border-radius: 50%; width: 16px; height: 16px;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px;
}
```

- [ ] **Step 3: 加切換邏輯**

在 tile 區 JS 加：

```javascript
async function toggleTileDecision(bundleId, tileEl) {
  const id = parseInt(tileEl.dataset.tileId, 10);
  const cur = tileEl.dataset.decision || 'accept';
  const next = cur === 'accept' ? 'reject' : 'accept';

  try {
    const resp = await fetch(`/api/models/${bundleId}/tiles/decision`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({tile_ids: [id], decision: next}),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      alert('切換失敗：' + (err.error || resp.status));
      return;
    }
    tileEl.dataset.decision = next;
    tileEl.classList.add('modified');
    refreshPendingIndicator(bundleId);  // 見 Task 9
  } catch (e) {
    alert('切換失敗：' + e);
  }
}

// 綁定點擊
document.addEventListener('click', (e) => {
  const tile = e.target.closest('.tile-thumb');
  if (tile && tile.closest('[data-bundle-id]')) {
    const bundleId = parseInt(tile.closest('[data-bundle-id]').dataset.bundleId, 10);
    toggleTileDecision(bundleId, tile);
  }
});
```

注意 `closest('[data-bundle-id]')` 假設 tile grid 容器有 `data-bundle-id` 屬性，務必加上：

```html
<div class="tile-grid" data-bundle-id="${bundleId}">
  ...tile thumbs...
</div>
```

- [ ] **Step 4: 鍵盤捷徑（沿用 step3 約定）**

如果 modal 已支援放大預覽：在 modal 內加 Del / A 鍵綁定。否則跳過——點擊切換已能完成主要操作。

- [ ] **Step 5: 手動驗證**

啟動 server，到 `/models` 開 bundle detail，找到 tile 區。點任意 OK tile：
- 縮圖應立即變暗（reject 樣式）
- 再點一次回到正常

開 DevTools Network tab，確認 POST `/api/models/<id>/tiles/decision` 200。

NG tile（如果可見）點擊應彈 alert 顯示「NG tile 不可動」（後端 400）。

- [ ] **Step 6: Commit**

```bash
git add templates/models.html
git commit -m "feat(ui): bundle detail 訓練 tile 支援切換 reject 狀態

點擊 OK tile 縮圖切換 accept/reject；reject 加暗色濾鏡 + 紅框。
NG tile 後端拒絕（400），前端不另作前置判斷。"
```

---

## Task 9: templates/models.html — 變更指示條與重訓按鈕

**目的：** 偵測「該 lighting+zone 有未訓練的修改」並提供啟動重訓的入口。

**Files:**
- Modify: `templates/models.html`

- [ ] **Step 1: 讀 lighting/zone 切換的現有結構**

```bash
grep -n "lighting\|G0F00000\|switchLighting\|tile-filter" templates/models.html | head -30
```

理解切換 lighting/zone 後 tile grid 怎麼重新載入。

- [ ] **Step 2: 加變更指示條 HTML**

在 lighting/zone 切換列下方插入：

```html
<div class="pending-bar" id="pending-bar" style="display:none;
     padding:8px 12px; background:#313244; border-radius:6px;
     margin:8px 0; display:flex; justify-content:space-between; align-items:center;">
  <span id="pending-text" style="color:#f9e2af;"></span>
  <button id="retrain-btn" onclick="startSubmodelRetrain()"
          style="background:#89b4fa; color:#1e1e2e; border:none;
                 border-radius:5px; padding:6px 14px; font-weight:700; cursor:pointer;">
    重訓此子模型 →
  </button>
</div>
```

- [ ] **Step 3: 加 `refreshPendingIndicator` 與 `startSubmodelRetrain` JS**

```javascript
let _bundleDetail = null;

async function refreshPendingIndicator(bundleId) {
  // 重新拉 detail 取得最新 pending_changes
  const resp = await fetch(`/api/models/${bundleId}/detail`);
  if (!resp.ok) return;
  _bundleDetail = await resp.json();
  updatePendingBar();
}

function updatePendingBar() {
  const bar = document.getElementById('pending-bar');
  const text = document.getElementById('pending-text');
  if (!_bundleDetail) { bar.style.display = 'none'; return; }

  const lighting = currentLighting();   // 既有切換狀態 getter；若不同名請用實際的
  const zone = currentZone();
  const pc = (_bundleDetail.pending_changes || []).find(
    p => p.lighting === lighting && p.zone === zone
  );
  if (!pc) { bar.style.display = 'none'; return; }

  text.textContent = `${lighting}-${zone}：有 ${pc.count} 張 tile 修改未訓練`;
  bar.style.display = 'flex';
  bar.dataset.lighting = lighting;
  bar.dataset.zone = zone;
  bar.dataset.bundleId = _bundleDetail.id;
}

async function startSubmodelRetrain() {
  const bar = document.getElementById('pending-bar');
  const bundleId = parseInt(bar.dataset.bundleId, 10);
  const lighting = bar.dataset.lighting;
  const zone = bar.dataset.zone;

  if (!confirm(`確認重訓 ${lighting}-${zone}？\n（threshold 不會被改動）`)) return;

  const resp = await fetch(`/api/models/${bundleId}/retrain_submodel`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({lighting, zone}),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    alert('啟動失敗：' + (err.error || resp.status));
    return;
  }
  document.getElementById('retrain-btn').disabled = true;
  startRetrainStatusPolling(bundleId);   // 見 Task 10
}
```

`currentLighting()` / `currentZone()` 要對應現有切換實作。若切換以 `data-lighting` 屬性記錄目前選擇，可寫成 `document.querySelector('.lighting-tab.active').dataset.lighting`。

在切換 lighting/zone 的事件處理結尾呼叫 `updatePendingBar()`，讓切換後即時反映。

- [ ] **Step 4: 在開啟 detail modal 時呼叫一次 `refreshPendingIndicator`**

找到開啟 detail modal 的函式（可能是 `openBundleDetail` 之類），結尾加：

```javascript
await refreshPendingIndicator(bundleId);
checkTrainingDataAvailability();
```

- [ ] **Step 4.5: 加訓練資料已刪禁用提示**

在 pending-bar HTML 上方加一個 banner：

```html
<div id="training-data-missing" style="display:none; padding:10px 14px;
     background:#3d1c25; border:1px solid #f38ba8; border-radius:6px;
     color:#f9e2af; margin:8px 0;">
  此 bundle 的訓練資料已清除，無法在此重訓。
  <a href="/train/new" style="color:#89dceb;">→ 至 /train/new 重新訓練</a>
</div>
```

JS：

```javascript
function checkTrainingDataAvailability() {
  const td = (_bundleDetail && _bundleDetail.training_data) || {};
  const ok = (td.ok_count || 0) > 0;
  const missing = document.getElementById('training-data-missing');
  const pendingBar = document.getElementById('pending-bar');
  const tileGrid = document.querySelector('.tile-grid');

  if (ok) {
    missing.style.display = 'none';
  } else {
    missing.style.display = 'block';
    pendingBar.style.display = 'none';
    if (tileGrid) tileGrid.classList.add('readonly');  // CSS .readonly { pointer-events: none; opacity: .6; }
  }
}
```

CSS：

```css
.tile-grid.readonly { pointer-events: none; opacity: 0.6; }
```

`updatePendingBar()` 開頭也加防護：

```javascript
function updatePendingBar() {
  const bar = document.getElementById('pending-bar');
  const text = document.getElementById('pending-text');
  if (!_bundleDetail) { bar.style.display = 'none'; return; }
  // 訓練資料已刪 → checkTrainingDataAvailability 會藏掉 bar
  const td = _bundleDetail.training_data || {};
  if ((td.ok_count || 0) === 0) { bar.style.display = 'none'; return; }
  // ... 原邏輯不變
}
```

- [ ] **Step 5: 手動驗證**

驗證訓練資料已刪場景：先對某 bundle 跑 delete_training_data，再進 detail 頁，預期看到紅色禁用 banner、tile grid 變半透明且不可點。

到另一個有訓練資料的 bundle detail 頁，標 reject 一張 tile：
- pending bar 應出現顯示「有 1 張 tile 修改未訓練」
- 切到別的 lighting+zone：bar 應消失（因為那個 unit 沒修改）
- 切回原 unit：bar 再出現

點「重訓此子模型」彈 confirm，按取消不送 request；按確認送 POST。

- [ ] **Step 6: Commit**

```bash
git add templates/models.html
git commit -m "feat(ui): 變更指示條與重訓單子模型按鈕

切到對應 lighting+zone 時顯示『有 N 張 tile 修改未訓練』+ 重訓按鈕。
按下後送 /api/models/<id>/retrain_submodel，禁用按鈕等進度。"
```

---

## Task 10: templates/models.html — 進度面板

**目的：** 啟動 retrain 後輪詢狀態並顯示 step/log/summary。失敗顯示 traceback。

**Files:**
- Modify: `templates/models.html`

- [ ] **Step 1: 加進度面板 HTML**

在 pending-bar 之後插入：

```html
<div id="retrain-progress" style="display:none; margin-top:12px;
     padding:12px; background:#1e1e2e; border:1px solid #45475a; border-radius:6px;">
  <div style="display:flex; gap:8px; margin-bottom:10px;">
    <span class="step-pill" data-step="stage">Stage</span>
    <span class="step-pill" data-step="train">Train</span>
    <span class="step-pill" data-step="metrics">Metrics</span>
    <span class="step-pill" data-step="swap">Swap</span>
    <span class="step-pill" data-step="reload">Reload</span>
    <span class="step-pill" data-step="done">Done</span>
  </div>
  <pre id="retrain-log" style="max-height:200px; overflow:auto;
       background:#11111b; color:#cdd6f4; padding:8px; border-radius:4px;
       font-size:.82rem; margin:0;"></pre>
  <div id="retrain-summary" style="display:none; margin-top:12px;
       padding:10px; background:#313244; border-radius:5px;"></div>
</div>
```

CSS：
```css
.step-pill {
  padding: 4px 10px; background: #313244; color: #6c7086;
  border-radius: 12px; font-size: .78rem;
}
.step-pill.active { background: #89b4fa; color: #1e1e2e; font-weight: 700; }
.step-pill.done { background: #a6e3a1; color: #1e1e2e; }
.step-pill.failed { background: #f38ba8; color: #1e1e2e; }
```

- [ ] **Step 2: 加狀態輪詢 JS**

```javascript
let _retrainPollTimer = null;

function startRetrainStatusPolling(bundleId) {
  document.getElementById('retrain-progress').style.display = 'block';
  document.getElementById('retrain-summary').style.display = 'none';

  if (_retrainPollTimer) clearInterval(_retrainPollTimer);
  _retrainPollTimer = setInterval(() => pollRetrainStatus(bundleId), 2000);
  pollRetrainStatus(bundleId);
}

async function pollRetrainStatus(bundleId) {
  const resp = await fetch(`/api/models/${bundleId}/retrain_status?tail=200`);
  if (!resp.ok) return;
  const { job } = await resp.json();
  if (!job) {
    clearInterval(_retrainPollTimer);
    return;
  }

  // 步驟高亮
  const STEPS = ['stage', 'train', 'metrics', 'swap', 'reload', 'done'];
  const curIdx = STEPS.indexOf(job.step);
  document.querySelectorAll('.step-pill').forEach(el => {
    const idx = STEPS.indexOf(el.dataset.step);
    el.classList.remove('active', 'done', 'failed');
    if (job.state === 'failed' && idx === curIdx) el.classList.add('failed');
    else if (idx < curIdx) el.classList.add('done');
    else if (idx === curIdx) el.classList.add('active');
  });

  // log
  const logEl = document.getElementById('retrain-log');
  logEl.textContent = (job.log_lines || []).join('\n');
  logEl.scrollTop = logEl.scrollHeight;

  if (job.state === 'completed' || job.state === 'failed') {
    clearInterval(_retrainPollTimer);
    document.getElementById('retrain-btn').disabled = false;
    showRetrainSummary(job);
    if (job.state === 'completed') {
      // 重新拉 detail 更新 pending bar（應該變空）
      refreshPendingIndicator(bundleId);
    }
  }
}

function showRetrainSummary(job) {
  const el = document.getElementById('retrain-summary');
  el.style.display = 'block';
  if (job.state === 'failed') {
    el.style.background = '#3d1c25';
    el.innerHTML = `<strong style="color:#f38ba8;">✗ 失敗</strong><br>
      <span style="color:#cdd6f4;">${(job.error || '').replace(/</g, '&lt;')}</span>`;
    return;
  }
  const s = job.summary || {};
  el.style.background = '#1e3025';
  el.innerHTML = `
    <strong style="color:#a6e3a1;">✓ 重訓完成</strong>
    <table style="width:100%; margin-top:8px; color:#cdd6f4;">
      <tr><td>AUROC</td><td>${s.auroc_old ?? '-'} → <strong>${s.auroc_new ?? '-'}</strong></td></tr>
      <tr><td>訓練 tile 數</td><td>${s.tile_count_old ?? '-'} → <strong>${s.tile_count_new ?? '-'}</strong></td></tr>
    </table>
    <p style="color:#fab387; margin:8px 0 0; font-size:.82rem;">
      Threshold 維持原值（沿用模型庫頁面手動調整的設定）。
    </p>`;
}
```

- [ ] **Step 3: 手動驗證（需要可重訓的 bundle）**

如果有可用 bundle：
1. 標 reject 一張 tile
2. 按重訓
3. 進度面板顯示 step 從 stage 漸進到 done，log 即時滾動
4. 完成後 summary 顯示新舊 AUROC

如果失敗（例如故意改一個錯的 lighting）：summary 顯示紅色失敗 + error 訊息。

- [ ] **Step 4: Commit**

```bash
git add templates/models.html
git commit -m "feat(ui): 重訓進度面板：step pill + log + summary

每 2 秒輪詢 /retrain_status；完成顯示 AUROC/tile 數對照與
『threshold 維持原值』提示。失敗顯示 error message 而非 traceback。"
```

---

## Task 11: templates/models.html — 列表頁徽章

**目的：** bundle 列表頁顯示「N 個子模型有未訓練修改」徽章，引導使用者進 detail 處理。

**Files:**
- Modify: `templates/models.html`

- [ ] **Step 1: 找列表渲染處**

```bash
grep -n "list_bundles\|api/models\?\|/api/models$" templates/models.html | head
```

理解列表怎麼來。可能是 `GET /api/models?machine_id=X` 回 bundle list，但**只有 detail 才回 pending_changes**。徽章需求要在列表階段就知道，這意味要麼：

(a) 修改列表 API 也回 `pending_changes` 摘要（簡單但 N+1 query 風險）
(b) 列表頁載入後對顯示中的 bundle 平行 fetch detail 取 pending（多 N 次 round trip）
(c) 加新 API `/api/models/pending_changes_summary` 一次回所有 bundle 的 count

**選 (a)**：列表頁本來就少（每機台幾個），N+1 影響小。

- [ ] **Step 2: 修 `_handle_models_list`**

找到列表 handler（搜 `_handle_models_list` 或 `def _handle_models`，line ~5715）：

```python
    def _handle_models_list(self):
        """GET /api/models?machine_id=X"""
        from urllib.parse import parse_qs, urlparse
        from capi_model_registry import get_pending_change_summary_for_bundle
        qs = parse_qs(urlparse(self.path).query)
        machine_id = (qs.get("machine_id") or [None])[0]
        db = self._capi_server_instance.database
        bundles = db.list_model_bundles(machine_id=machine_id) if machine_id else db.list_model_bundles()
        out = []
        for b in bundles:
            pc = get_pending_change_summary_for_bundle(db, b)
            row = dict(b)
            row["pending_unit_count"] = len(pc)   # 有未訓練修改的 unit 數量
            out.append(row)
        self._send_json({"bundles": out})
```

注意：此函式現有實作可能不同，要先讀現況再改。**保留現有回傳結構，只加 `pending_unit_count` 欄位**。

- [ ] **Step 3: 在列表 UI 加徽章**

找 bundle 列表的 row 渲染（HTML/JS）。在 bundle 名稱旁加：

```html
${b.pending_unit_count > 0
  ? `<span style="background:#f9e2af; color:#1e1e2e; padding:2px 8px;
       border-radius:10px; font-size:.7rem; margin-left:6px;"
       title="有 ${b.pending_unit_count} 個子模型有未訓練的修改">
       ⚠ ${b.pending_unit_count}</span>`
  : ''}
```

- [ ] **Step 4: Sanity check**

進 `/models`：
- 沒有任何 reject → 不顯示徽章
- 標 1 張 tile reject → 重整列表，看到對應 bundle 出現「⚠ 1」徽章
- 重訓完成 → 列表徽章消失

- [ ] **Step 5: Commit**

```bash
git add capi_web.py templates/models.html
git commit -m "feat(ui): bundle 列表加『N 個子模型有未訓練修改』徽章

列表 API 多回 pending_unit_count，前端用黃色徽章呈現。"
```

---

## Task 12: 文件與整合驗收

**目的：** 更新文件、跑端對端流程驗收、確認沒踩到既有功能。

**Files:**
- Modify: `CLAUDE.md`（功能章節補一行）
- Verify: 手動 happy path + 邊界 case

- [ ] **Step 1: CLAUDE.md 更新**

找「Multi-Architecture Support」或「Web Dashboard」章節，加一段：

```
- bundle detail 頁支援單子模型重訓：標記 OK tile 為 reject → 「重訓此子模型」按鈕觸發後台訓練 →
  就地覆蓋 `<lighting>-<zone>.pt`、寫 manifest history、reload inferencer cache。
  threshold 與 yaml 完全不動（threshold 由使用者在模型庫手動調整）。
```

- [ ] **Step 2: 全套手動驗收**

跟著 spec `Verification` 章節的 7 個項目逐一跑過：

1. ✅ 正常重訓路徑
2. ✅ inference 即時生效（送一個對應機台 panel，分數應改變）
3. ✅ 失敗回滾（傳 lighting=FOO → 400；無 job_id → 400）
4. ✅ GPU lock 排隊（重訓中送 AOI request，inference 等待）
5. ✅ 訓練資料已刪場景：跑「刪訓練資料」後重訓 API 回 400
6. ✅ NG tile 唯讀：對 NG tile 呼叫 decision API 回 400
7. ✅ 同時兩個重訓：A 跑中送 B，B 收到 409

每項記下實際結果（pass/fail + 觀察）。

- [ ] **Step 3: 整合驗收 commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md 補單子模型重訓功能說明"
```

如果驗收過程發現任何 bug，立即修並另開 commit。

---

## 完成檢查表

- [ ] Task 1：`train_single_submodel` 抽出，`run_training_pipeline` 行為不變
- [ ] Task 2：manifest 操作公用函式 + 8 個單元測試 pass
- [ ] Task 3：`reload_submodel` 加在 inferencer
- [ ] Task 4：tiles/decision API（NG tile 拒絕）
- [ ] Task 5：retrain 啟動/狀態 API（409 / 400 處理）
- [ ] Task 6：worker 完整流程（含 manifest history）
- [ ] Task 7：detail handler 透出 `pending_changes`
- [ ] Task 8：UI tile accept/reject 切換
- [ ] Task 9：變更指示條 + 重訓按鈕
- [ ] Task 10：進度面板 + 完成 summary
- [ ] Task 11：列表頁徽章
- [ ] Task 12：CLAUDE.md 更新 + 7 項手動驗收 pass

**Done definition：** spec 中所有 In Scope 項目實作完成，Verification 7 項全 pass，新增的單元測試全 pass，既有 train_new 流程不受影響（5 步精靈仍能跑通）。
