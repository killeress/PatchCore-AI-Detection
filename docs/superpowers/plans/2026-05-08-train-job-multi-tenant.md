# Train Wizard Multi-Tenant Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓 `/train/new` 流程支援同時多筆 job 共存（preprocess + review 平行），只在 GPU-bound 的 train 階段做 singleton 序列化。消除目前 TOCTOU race 與跨 job cancel 互打的事故根因。

**Architecture:** 把 `CAPIWebHandler._train_new_state` 全域單槽 dict 拆成 `_train_new_jobs: Dict[str, dict]` per-job runtime 註冊表，加 `_train_slot` 訓練槽（singleton lock）。所有 helper / worker / handler 透過 `job_id` 在註冊表查 runtime；`/api/train/new/start` 不再做全 wizard singleton 檢查；`/api/train/new/start_training/<id>` 拿不到 train slot 就 409；cancel 永遠以 `job_id` 反推自己的 cancel flag 檔，不讀全域。Step 1 banner 從「最近一筆 review」改成「列出所有 open jobs」。

**Tech Stack:** Python 3, http.server.ThreadingHTTPServer, threading.{Lock,Event}, subprocess.Popen, SQLite (training_jobs table), Jinja2 templates, pytest + ThreadPoolExecutor for race regression test.

---

## File Structure

**Modify:**
- `capi_web.py` — replace `_train_new_state` with per-job registry + train slot; rewrite ~9 handlers/workers/helpers
- `capi_database.py` — add `list_active_training_jobs()`
- `templates/train_new/step1_select.html` — banner 改為列表
- `tests/test_capi_web_train_new.py` — 既有 6 個用 `_train_new_state` 的測試遷移到新結構；新增 race / cancel-isolation / status-isolation / training-slot 測試

**No new files.**

## Files Map

| Concern | File | Lines (approx) |
|---|---|---|
| Per-job runtime registry + train slot 定義 | `capi_web.py` | 191-260 |
| Log / cancel_event / worker_alive helper | `capi_web.py` | 246-295 |
| Stale guard | `capi_web.py` | 274-292 |
| `_handle_train_new_start` | `capi_web.py` | 5052-5130 |
| `_handle_train_new_start_training` | `capi_web.py` | 5471-5500 |
| `_handle_train_new_cancel` | `capi_web.py` | 5414-5469 |
| `_handle_train_new_status` | `capi_web.py` | 5224-5264 |
| `_train_new_preprocess_worker` | `capi_web.py` | 5159-5222 |
| `_train_new_training_worker` | `capi_web.py` | 5502-5635 |
| `_handle_train_new_page` | `capi_web.py` | 4754-4771 |
| Server bootstrap reset | `capi_web.py` | 6620-6640 |
| Step 1 banner template | `templates/train_new/step1_select.html` | 16-33 |
| DB query | `capi_database.py` | 後接 `get_active_training_job` |

---

### Task 1: Add `list_active_training_jobs()` to DB layer

**Files:**
- Modify: `capi_database.py` — 在 `get_active_training_job` 後面加新 method
- Test: `tests/test_capi_database_train.py`

- [ ] **Step 1: Write failing test**

加到 `tests/test_capi_database_train.py` 末尾：

```python
def test_list_active_training_jobs_returns_all_open(tmp_path):
    db = TrainingDB(tmp_path / "t.db")
    db.create_training_job("j_pre", "M", ["/a", "/b", "/c"])
    db.create_training_job("j_rev", "M", ["/a", "/b", "/c"])
    db.update_training_job_state("j_rev", "review")
    db.create_training_job("j_done", "M", ["/a", "/b", "/c"])
    db.update_training_job_state("j_done", "completed")

    rows = db.list_active_training_jobs()
    ids = [r["job_id"] for r in rows]
    assert sorted(ids) == ["j_pre", "j_rev"]
    # 驗證 newest first（started_at DESC）
    assert rows[0]["job_id"] == "j_rev" or rows[0]["job_id"] == "j_pre"
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_capi_database_train.py::test_list_active_training_jobs_returns_all_open -v
```
Expected: FAIL with `AttributeError: 'TrainingDB' object has no attribute 'list_active_training_jobs'`

- [ ] **Step 3: Add method in `capi_database.py`**

於 `get_active_training_job` 結束後（line ~2469）插入：

```python
    def list_active_training_jobs(self) -> List[Dict]:
        """回傳所有 state 在 (preprocess, review, train) 的 job，依 started_at DESC 排序。"""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT * FROM training_jobs
                   WHERE state IN ('preprocess', 'review', 'train')
                   ORDER BY started_at DESC"""
            )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            jobs = []
            for row in rows:
                job = dict(zip(cols, row))
                if job.get("panel_paths"):
                    job["panel_paths"] = json.loads(job["panel_paths"])
                else:
                    job["panel_paths"] = []
                raw_params = job.get("training_params")
                job["training_params"] = json.loads(raw_params) if raw_params else None
                jobs.append(job)
            return jobs
        finally:
            conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_capi_database_train.py::test_list_active_training_jobs_returns_all_open -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```
git add capi_database.py tests/test_capi_database_train.py
git commit -m "feat(db): list_active_training_jobs 回傳所有 open job"
```

---

### Task 2: 加入 `_train_new_jobs` 註冊表 + `_train_slot`，先不切換 callers

**Files:**
- Modify: `capi_web.py:191-205`（替換 `_train_new_state` class attr 為新結構）
- Modify: `capi_web.py:6627-6633`（server bootstrap 重設）

**這一步是純結構引入 + helper 加上，行為不變。** 既有 callers 還用舊路徑，下一個 task 才開始遷移。為避免 callers 全部失效，本 task 先把 `_train_new_state` 保留並改成「動態 view 到 default job」適配層（callers 不會壞）；新加的 helper 走新註冊表。

具體做法：把舊 `_train_new_state` 留著（行為不變），在它旁邊加 `_train_new_jobs` 與 `_train_slot`、加新 helper。

- [ ] **Step 1: 在 class 屬性區插入新結構**

在 `capi_web.py:205`（舊 `_train_new_state` 結束 `}` 之後）插入：

```python
    # ── 多 job 註冊表（refactor：取代 _train_new_state） ──────────────────────
    # _train_new_jobs key = job_id；value = per-job runtime dict，欄位：
    #   thread:        Thread (preprocess supervisor / training supervisor)
    #   proc:          Optional[Popen] (only training)
    #   cancel_flag:   Optional[str] (training subprocess 的 cancel flag 檔路徑)
    #   log_file:      Optional[str] (training subprocess 的 log 檔路徑)
    #   cancel_event:  threading.Event
    #   log_lines:     List[str] (ring buffer 500)
    #   log_lock:      threading.Lock
    #   unit_status:   Dict[unit_label, status]
    #   phase:         "preprocess" | "review" | "train"  (last known)
    _train_new_jobs: dict = {}
    _train_new_jobs_lock: threading.Lock = threading.Lock()

    # 訓練槽（單機 GPU 序列化）：持有 lock 的 job_id 才能跑 training subprocess
    _train_slot: dict = {
        "lock": threading.Lock(),
        "active_job_id": None,
    }
```

- [ ] **Step 2: 加 helper `_make_job_runtime` / `_get_job_runtime` / `_drop_job_runtime`**

在 `_append_train_new_log`（line 246）之前插入：

```python
    @classmethod
    def _make_job_runtime(cls, job_id: str, phase: str) -> dict:
        runtime = {
            "thread": None,
            "proc": None,
            "cancel_flag": None,
            "log_file": None,
            "cancel_event": threading.Event(),
            "log_lines": [],
            "log_lock": threading.Lock(),
            "unit_status": {},
            "phase": phase,
        }
        with cls._train_new_jobs_lock:
            cls._train_new_jobs[job_id] = runtime
        return runtime

    @classmethod
    def _get_job_runtime(cls, job_id: str) -> Optional[dict]:
        with cls._train_new_jobs_lock:
            return cls._train_new_jobs.get(job_id)

    @classmethod
    def _drop_job_runtime(cls, job_id: str) -> None:
        with cls._train_new_jobs_lock:
            cls._train_new_jobs.pop(job_id, None)
```

- [ ] **Step 3: 在 `create_web_server` 末尾把新結構也重設**

於 `capi_web.py:6633` 後（`CAPIWebHandler._train_new_state = {...}` 結束 `}` 之後）插入：

```python
    CAPIWebHandler._train_new_jobs = {}
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()
    CAPIWebHandler._train_slot = {
        "lock": threading.Lock(),
        "active_job_id": None,
    }
```

- [ ] **Step 4: Smoke test — import 不壞**

```
python -c "import capi_web; print(capi_web.CAPIWebHandler._train_new_jobs, capi_web.CAPIWebHandler._train_slot)"
```
Expected: `{} {'lock': ..., 'active_job_id': None}`

- [ ] **Step 5: Run existing test suite to verify no regression**

```
pytest tests/test_capi_web_train_new.py -v
```
Expected: ALL PASS（純結構新增）

- [ ] **Step 6: Commit**

```
git add capi_web.py
git commit -m "refactor(web): 加入 _train_new_jobs 註冊表 + _train_slot 結構，尚未接通 callers"
```

---

### Task 3: 把 `_append_train_new_log` / `_train_new_cancel_event` / `_train_new_worker_alive` 改成 by job_id

**Files:**
- Modify: `capi_web.py:246-292`

舊簽名 take `state: dict` 參數，改成 take `job_id: str`，內部查 `_train_new_jobs[job_id]`。為了讓既有 callers（workers + handlers）都還能編譯通過，**這 task 先讓函式接受兩種呼叫方式**：傳 dict 走 legacy 路徑，傳 str 走新註冊表。等所有 callers 切完才移除 legacy。

- [ ] **Step 1: 改 `_append_train_new_log` 簽名**

替換 `capi_web.py:245-253`：

```python
    @classmethod
    def _append_train_new_log(cls, state_or_job_id, msg: str) -> None:
        """寫一行 log + 解析 unit_status。

        state_or_job_id 可以是 legacy state dict 或新 job_id (str)。
        過渡期支援兩種，全部 caller 切完後改為只接受 str。
        """
        if isinstance(state_or_job_id, str):
            runtime = cls._get_job_runtime(state_or_job_id)
            if runtime is None:
                return
            state = runtime
        else:
            state = state_or_job_id
        with state["log_lock"]:
            state["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            if len(state["log_lines"]) > 500:
                state["log_lines"] = state["log_lines"][-500:]
            m = _TRAIN_UNIT_LOG_RE.search(msg)
            if m:
                state.setdefault("unit_status", {})[m.group(1)] = _TRAIN_UNIT_STATUS_MAP[m.group(2)]
```

- [ ] **Step 2: 改 `_train_new_cancel_event` 簽名**

替換 `capi_web.py:255-257`：

```python
    @classmethod
    def _train_new_cancel_event(cls, state_or_job_id) -> threading.Event:
        if isinstance(state_or_job_id, str):
            runtime = cls._get_job_runtime(state_or_job_id)
            if runtime is None:
                return threading.Event()  # 無 runtime 回 dummy event
            return runtime.setdefault("cancel_event", threading.Event())
        return state_or_job_id.setdefault("cancel_event", threading.Event())
```

- [ ] **Step 3: 改 `_train_new_worker_alive`**

替換 `capi_web.py:259-271`：

```python
    @classmethod
    def _train_new_worker_alive(cls, job_id: str) -> bool:
        runtime = cls._get_job_runtime(job_id)
        if runtime is None:
            # legacy fallback：在切換完成前還會被既有舊路徑 caller 呼叫
            state = cls._train_new_state
            with state["lock"]:
                if state.get("active_job_id") != job_id:
                    return False
                thread = state.get("thread")
                if thread is not None and thread.is_alive():
                    return True
                proc = state.get("proc")
                if proc is not None and proc.poll() is None:
                    return True
                return False
        thread = runtime.get("thread")
        if thread is not None and thread.is_alive():
            return True
        proc = runtime.get("proc")
        if proc is not None and proc.poll() is None:
            return True
        return False
```

- [ ] **Step 4: 改 `_mark_train_new_stale_if_needed`**

替換 `capi_web.py:273-292`：

```python
    @classmethod
    def _mark_train_new_stale_if_needed(cls, db, job: Optional[dict]) -> Optional[dict]:
        if not job or job.get("state") not in ("preprocess", "train"):
            return job
        if cls._train_new_worker_alive(job["job_id"]):
            return job

        error = "interrupted: server restarted or training worker is not running"
        db.update_training_job_state(job["job_id"], "failed", error_message=error)
        # 若 runtime 還在註冊表（且沒 thread）→ 順手清掉
        cls._drop_job_runtime(job["job_id"])
        # legacy 路徑相容
        state = cls._train_new_state
        with state["lock"]:
            if state.get("active_job_id") == job["job_id"] or not state.get("active_job_id"):
                state["active_job_id"] = None
                state["thread"] = None
                cls._train_new_cancel_event(state).clear()
        cls._append_train_new_log(state, f"✗ {job['job_id']} 已標記失敗：{error}")
        updated = dict(job)
        updated["state"] = "failed"
        updated["error_message"] = error
        return updated
```

- [ ] **Step 5: Run existing tests to verify no regression**

```
pytest tests/test_capi_web_train_new.py -v
```
Expected: ALL PASS（過渡期 helper 同時支援兩種呼叫方式）

- [ ] **Step 6: Commit**

```
git add capi_web.py
git commit -m "refactor(web): helper 同時支援 by-job-id 與 legacy state dict 兩種簽名"
```

---

### Task 4: 遷移 `_train_new_preprocess_worker` 到 per-job runtime

**Files:**
- Modify: `capi_web.py:5159-5222`

- [ ] **Step 1: 改 worker 內所有 `state` 引用為 `runtime`**

替換 `_train_new_preprocess_worker` 整個函式（line 5159-5222）為：

```python
    @staticmethod
    def _train_new_preprocess_worker(
        job_id, machine_id, panel_paths, server_inst, training_params=None,
    ):
        """背景 thread：preprocess + 抽 NG → state=review。"""
        import traceback
        from pathlib import Path as _Path
        from capi_train_new import (
            TrainingConfig, apply_user_training_params,
            preprocess_panels_to_pool, sample_ng_tiles, NG_TILES_PER_LIGHTING,
        )
        from capi_preprocess import PreprocessConfig

        db = server_inst.database
        runtime = CAPIWebHandler._get_job_runtime(job_id)
        if runtime is None:
            # 防呆：理論上 _handle_train_new_start 已建好
            runtime = CAPIWebHandler._make_job_runtime(job_id, "preprocess")

        def log(msg):
            CAPIWebHandler._append_train_new_log(job_id, msg)

        try:
            train_cfg = CAPIWebHandler._load_train_new_config(server_inst)

            thumb_root = _Path(".tmp/train_new_thumbs") / job_id
            cfg = TrainingConfig(
                machine_id=machine_id,
                panel_paths=[_Path(p) for p in panel_paths],
                over_review_root=train_cfg["over_review_root"],
                backbone_cache_dir=train_cfg["backbone_cache_dir"],
                output_root=train_cfg["output_root"],
                required_backbones=train_cfg["required_backbones"],
            )
            apply_user_training_params(cfg, training_params, log_fn=log)
            pre_cfg = PreprocessConfig()

            log(f"開始前處理 {len(panel_paths)} panel")
            stats = preprocess_panels_to_pool(
                job_id=job_id, cfg=cfg, preprocess_cfg=pre_cfg,
                db=db, thumb_dir=thumb_root, log=log,
            )
            if stats["panel_success"] < 3:
                raise RuntimeError(f"成功 panel < 3 ({stats['panel_success']})")

            log(f"抽 NG tile（每 lighting 上限 {NG_TILES_PER_LIGHTING} 個）")
            ng_stats = sample_ng_tiles(
                job_id=job_id, over_review_root=cfg.over_review_root,
                db=db, thumb_dir=thumb_root, log=log,
            )

            db.update_training_job_state(job_id, "review")
            runtime["phase"] = "review"
            log("✓ 進入 review 階段")
        except Exception as e:
            traceback.print_exc()
            db.update_training_job_state(job_id, "failed", error_message=str(e))
            log(f"✗ 失敗: {e}")
            CAPIWebHandler._drop_job_runtime(job_id)
```

注意：成功路徑（state→review）**保留** runtime 在註冊表，因為 review 階段不需要 thread/proc 但 log_lines 還會被 status 端點查；review→train 切換時由 start_training 補上 thread/proc。失敗路徑才 `_drop_job_runtime`。

- [ ] **Step 2: Smoke import**

```
python -c "import capi_web; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Run existing tests**

```
pytest tests/test_capi_web_train_new.py -v
```
Expected: 既有測試仍 PASS（preprocess worker 未被任何 unit test 直接 mock 過 state）

- [ ] **Step 4: Commit**

```
git add capi_web.py
git commit -m "refactor(web): preprocess worker 改寫到 per-job runtime"
```

---

### Task 5: 遷移 `_train_new_training_worker` 到 per-job runtime + 釋放 train slot

**Files:**
- Modify: `capi_web.py:5502-5635`

- [ ] **Step 1: 替換整個 `_train_new_training_worker`**

```python
    @staticmethod
    def _train_new_training_worker(job_id, machine_id, panel_paths, server_inst):
        """Supervisor thread：launch 訓練 subprocess、tail log、偵測退出、清狀態。"""
        import os as _os
        import subprocess as _subprocess
        import sys as _sys
        import time as _time
        import traceback as _traceback
        from pathlib import Path as _Path

        runtime = CAPIWebHandler._get_job_runtime(job_id)
        if runtime is None:
            runtime = CAPIWebHandler._make_job_runtime(job_id, "train")
        runtime["phase"] = "train"
        db = server_inst.database

        def log(msg):
            CAPIWebHandler._append_train_new_log(job_id, msg)

        proc = None
        try:
            train_cfg = CAPIWebHandler._load_train_new_config(server_inst)
            output_root = _Path(train_cfg["output_root"])
            log_dir = output_root / "training_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{job_id}.log"
            cancel_flag = log_dir / f"{job_id}.cancel"
            log_file.write_text("", encoding="utf-8")
            try:
                cancel_flag.unlink()
            except FileNotFoundError:
                pass

            server_cfg_path = _Path(server_inst.server_config_path).resolve()
            project_root = _Path(__file__).resolve().parent

            cmd = [
                _sys.executable, "-u", "-m", "capi_train_runner",
                "--job-id", job_id,
                "--server-config", str(server_cfg_path),
                "--log-file", str(log_file),
                "--cancel-flag", str(cancel_flag),
            ]
            log(f"啟動訓練 subprocess（VRAM 隔離模式）")

            env = {**_os.environ, "PYTHONUNBUFFERED": "1"}
            proc = _subprocess.Popen(
                cmd,
                cwd=str(project_root),
                env=env,
                stdout=_subprocess.DEVNULL,
                stderr=_subprocess.STDOUT,
            )

            runtime["proc"] = proc
            runtime["log_file"] = str(log_file)
            runtime["cancel_flag"] = str(cancel_flag)

            log(f"訓練 subprocess pid={proc.pid}，log={log_file}")

            tail_pos = 0

            def _drain_log():
                nonlocal tail_pos
                try:
                    with open(log_file, "rb") as f:
                        f.seek(tail_pos)
                        data = f.read()
                        tail_pos = f.tell()
                except FileNotFoundError:
                    return
                if not data:
                    return
                text = data.decode("utf-8", errors="replace")
                for line in text.splitlines():
                    line = line.rstrip()
                    if line:
                        CAPIWebHandler._append_train_new_log(job_id, line)

            while True:
                _drain_log()
                ret = proc.poll()
                if ret is not None:
                    _drain_log()
                    if ret == 0:
                        log(f"✓ 訓練 subprocess 結束 (exit=0)")
                    else:
                        log(f"✗ 訓練 subprocess 結束 (exit={ret})")
                    break
                _time.sleep(1.0)

            if proc.returncode != 0:
                try:
                    job = db.get_training_job(job_id)
                    if job and job.get("state") == "train":
                        db.update_training_job_state(
                            job_id, "failed",
                            error_message=f"runner exited with code {proc.returncode}",
                        )
                except Exception:
                    pass

        except Exception as e:
            _traceback.print_exc()
            log(f"✗ 訓練監看失敗: {e}")
            try:
                db.update_training_job_state(job_id, "failed", error_message=str(e))
            except Exception:
                pass
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        finally:
            # 釋放訓練槽（讓下一個排隊的 job 能進入 train）
            slot = CAPIWebHandler._train_slot
            with slot["lock"]:
                if slot.get("active_job_id") == job_id:
                    slot["active_job_id"] = None
            # 訓練 done/failed 都把 runtime 清掉（log 還想留可改成只清 thread/proc）
            CAPIWebHandler._drop_job_runtime(job_id)
```

- [ ] **Step 2: Smoke import**

```
python -c "import capi_web; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Run existing tests**

```
pytest tests/test_capi_web_train_new.py -v
```
Expected: ALL PASS（worker 沒被直接 unit test）

- [ ] **Step 4: Commit**

```
git add capi_web.py
git commit -m "refactor(web): training worker 改寫到 per-job runtime + 釋放 train slot"
```

---

### Task 6: 重寫 `_handle_train_new_start` — 拿掉 wizard singleton 檢查

**Files:**
- Modify: `capi_web.py:5052-5130`
- Test: `tests/test_capi_web_train_new.py`（含 race regression）

- [ ] **Step 1: Write failing test — 既有 review job 不再阻擋新 start**

替換 `tests/test_capi_web_train_new.py:212-245` 整個 `test_handle_train_new_start_rejects_concurrent` 為：

```python
def test_handle_train_new_start_allows_concurrent_with_review_job():
    """有別人在 review state 不再阻擋新 start。"""
    from capi_web import CAPIWebHandler

    server = MagicMock()
    server.database.list_active_training_jobs.return_value = [
        {"job_id": "j_old", "state": "review"},
    ]
    server.database.create_training_job = MagicMock()
    CAPIWebHandler._train_new_jobs = {}
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()

    h = _make_handler_with_server(server, "/api/train/new/start")
    body = json.dumps({"machine_id": "M", "panel_paths": [f"/p{i}" for i in range(3)]}).encode()
    h.headers.get = MagicMock(return_value=str(len(body)))
    h.rfile = io.BytesIO(body)

    # 不讓 worker thread 真的跑
    with patch("capi_web.threading.Thread") as MockThread:
        MockThread.return_value.start = MagicMock()
        h._handle_train_new_start()

    assert h._sent_response[0]["status"] == 200
    body = json.loads(h._sent_response[0]["body"])
    assert body["state"] == "preprocess"
    assert body["job_id"]  # 新 job_id 已配發
    server.database.create_training_job.assert_called_once()


def test_handle_train_new_start_parallel_creates_two_jobs():
    """TOCTOU race regression：兩個 thread 同時 start 都成功。"""
    from concurrent.futures import ThreadPoolExecutor
    from capi_web import CAPIWebHandler

    created = []
    db_lock = threading.Lock()

    server = MagicMock()
    server.database.list_active_training_jobs.return_value = []

    def record_create(job_id, machine_id, panel_paths, training_params=None):
        with db_lock:
            created.append(job_id)

    server.database.create_training_job = MagicMock(side_effect=record_create)

    CAPIWebHandler._train_new_jobs = {}
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()

    def run_start():
        h = _make_handler_with_server(server, "/api/train/new/start")
        body = json.dumps({"machine_id": "M", "panel_paths": [f"/p{i}" for i in range(3)]}).encode()
        h.headers.get = MagicMock(return_value=str(len(body)))
        h.rfile = io.BytesIO(body)
        with patch("capi_web.threading.Thread") as MockThread:
            MockThread.return_value.start = MagicMock()
            h._handle_train_new_start()
        return h._sent_response[0]

    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(run_start) for _ in range(2)]
        results = [f.result() for f in futs]

    statuses = sorted(r["status"] for r in results)
    assert statuses == [200, 200]
    assert len(set(created)) == 2  # 兩個不同 job_id
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_capi_web_train_new.py::test_handle_train_new_start_allows_concurrent_with_review_job tests/test_capi_web_train_new.py::test_handle_train_new_start_parallel_creates_two_jobs -v
```
Expected: FAIL（舊 handler 還在做 singleton 檢查）

- [ ] **Step 3: 重寫 `_handle_train_new_start`**

替換 `capi_web.py:5052-5130`：

```python
    def _handle_train_new_start(self):
        """POST /api/train/new/start

        body: {
            "machine_id": "...",
            "panel_paths": [...],
            "training_params": {...}  # optional
        }
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
        if not isinstance(panel_paths, list):
            self._send_json({"error": "panel_paths must be a list"}, status=400)
            return
        clean_panel_paths = []
        for p in panel_paths:
            if not isinstance(p, str) or not p.strip() or p.strip() in ("undefined", "null"):
                self._send_json({"error": "panel_paths contains invalid path"}, status=400)
                return
            clean_panel_paths.append(p.strip())
        if len(clean_panel_paths) != 3:
            self._send_json({"error": "panel_paths must contain exactly 3 panels"}, status=400)
            return

        training_params, err = self._validate_training_params(params.get("training_params"))
        if err:
            self._send_json({"error": err}, status=400)
            return

        db = self._capi_server_instance.database
        from capi_train_new import generate_job_id
        job_id = generate_job_id(machine_id)

        # 註冊 runtime + 寫 DB（沒有 wizard singleton 檢查；多 job 可共存）
        runtime = CAPIWebHandler._make_job_runtime(job_id, "preprocess")
        try:
            db.create_training_job(
                job_id=job_id, machine_id=machine_id,
                panel_paths=clean_panel_paths,
                training_params=training_params,
            )
        except Exception:
            CAPIWebHandler._drop_job_runtime(job_id)
            raise

        thread = threading.Thread(
            target=CAPIWebHandler._train_new_preprocess_worker,
            args=(job_id, machine_id, clean_panel_paths,
                  self._capi_server_instance, training_params),
            daemon=True, name=f"train_new_pre-{job_id}",
        )
        runtime["thread"] = thread
        thread.start()
        self._send_json({"job_id": job_id, "state": "preprocess"})
```

- [ ] **Step 4: Run new tests**

```
pytest tests/test_capi_web_train_new.py::test_handle_train_new_start_allows_concurrent_with_review_job tests/test_capi_web_train_new.py::test_handle_train_new_start_parallel_creates_two_jobs -v
```
Expected: PASS

- [ ] **Step 5: 順手修舊測試 fixture（如果還有 reference 到 `_train_new_state` 的初始化）**

跑：
```
pytest tests/test_capi_web_train_new.py -v
```
若有失敗，把該 test 的 `CAPIWebHandler._train_new_state = {...}` 改成 `CAPIWebHandler._train_new_jobs = {}`、`CAPIWebHandler._train_new_jobs_lock = threading.Lock()`，並把測試斷言裡的 `_train_new_state["active_job_id"]` 改用 `_get_job_runtime`。

- [ ] **Step 6: Commit**

```
git add capi_web.py tests/test_capi_web_train_new.py
git commit -m "feat(web): /api/train/new/start 支援多 job 共存，移除 wizard singleton 檢查"
```

---

### Task 7: 重寫 `_handle_train_new_start_training` — 用 `_train_slot` 做 GPU 序列化

**Files:**
- Modify: `capi_web.py:5471-5500`
- Test: `tests/test_capi_web_train_new.py`

- [ ] **Step 1: Write failing test — 第二個 start_training 應 409**

加到 `tests/test_capi_web_train_new.py`：

```python
def test_handle_train_new_start_training_rejects_when_slot_held():
    """另一個 job 已在 train → 第二個 start_training 收 409。"""
    from capi_web import CAPIWebHandler

    server = MagicMock()
    server.database.get_training_job.return_value = {
        "job_id": "j2", "machine_id": "M", "state": "review", "panel_paths": []
    }

    CAPIWebHandler._train_new_jobs = {}
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()
    CAPIWebHandler._train_slot = {
        "lock": threading.Lock(),
        "active_job_id": "j1",  # 模擬 j1 已佔用
    }
    # j1 也佔了 slot lock
    assert CAPIWebHandler._train_slot["lock"].acquire(blocking=False)

    h = _make_handler_with_server(server, "/api/train/new/start_training/j2")
    h._handle_train_new_start_training()

    assert h._sent_response[0]["status"] == 409
    body = json.loads(h._sent_response[0]["body"])
    assert body["error"] == "another_job_training"
    assert body["training_job_id"] == "j1"

    CAPIWebHandler._train_slot["lock"].release()
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_capi_web_train_new.py::test_handle_train_new_start_training_rejects_when_slot_held -v
```
Expected: FAIL

- [ ] **Step 3: 重寫 handler**

替換 `capi_web.py:5471-5500`：

```python
    def _handle_train_new_start_training(self):
        """POST /api/train/new/start_training/<job_id>

        以 _train_slot 序列化：拿不到 → 409 並回 currently_training_job_id。
        """
        job_id = self.path.rsplit("/", 1)[-1].split("?")[0]
        db = self._capi_server_instance.database
        job = db.get_training_job(job_id)
        if not job:
            self._send_json({"error": "job not found"}, status=404)
            return
        job = self._mark_train_new_stale_if_needed(db, job)
        if job["state"] != "review":
            self._send_json({"error": f"job state must be 'review', currently '{job['state']}'"}, status=409)
            return

        slot = CAPIWebHandler._train_slot
        if not slot["lock"].acquire(blocking=False):
            self._send_json({
                "error": "another_job_training",
                "training_job_id": slot.get("active_job_id"),
            }, status=409)
            return
        slot["active_job_id"] = job_id

        # 確保 runtime 存在（從 review 接續，多半已存在；服器重啟後可能沒有）
        runtime = CAPIWebHandler._get_job_runtime(job_id)
        if runtime is None:
            runtime = CAPIWebHandler._make_job_runtime(job_id, "train")
        runtime["phase"] = "train"
        runtime["proc"] = None
        runtime["cancel_flag"] = None
        runtime["log_file"] = None
        runtime["unit_status"] = {}
        runtime["cancel_event"].clear()

        thread = threading.Thread(
            target=CAPIWebHandler._train_new_training_worker,
            args=(job_id, job["machine_id"], job["panel_paths"], self._capi_server_instance),
            daemon=True, name=f"train_new-{job_id}",
        )
        runtime["thread"] = thread

        # slot 的 release 由 worker 在 finally 處（透過 _train_slot["active_job_id"] = None
        # 與 lock.release()）一起做。這裡先把 release 排入 worker 結束流程。
        # → worker `finally` block 不釋放 lock？需確認設計
        # 為避免兩處鎖管，這裡 acquire 完不留 lock 給 worker：改用 active_job_id 旗標
        slot["lock"].release()

        db.update_training_job_state(job_id, "train")
        thread.start()
        self._send_json({"ok": True, "state": "train"})
```

**設計選擇說明：** `slot["lock"]` 只用來保證「同一時刻只有一個 thread 進入起 training 的關鍵段落」，不長期持有。GPU singleton 真正的判斷靠 `active_job_id` 是否為 None。worker `finally` 把 `active_job_id` 設回 None，下次 start_training 拿 lock 後檢查 `active_job_id != None` → 409。

修正：把 handler 內 acquire 後的判斷再嚴格一點。重寫此段：

```python
        slot = CAPIWebHandler._train_slot
        with slot["lock"]:
            if slot.get("active_job_id") is not None:
                self._send_json({
                    "error": "another_job_training",
                    "training_job_id": slot["active_job_id"],
                }, status=409)
                return
            slot["active_job_id"] = job_id
```

把上面 task step 3 中 `if not slot["lock"].acquire(blocking=False): ...` 與後續 `slot["lock"].release()` 整段替換成這個 `with slot["lock"]:` block。

- [ ] **Step 4: 修正測試以對齊新設計**

把 step 1 的測試改為直接設 `active_job_id = "j1"`、不要 acquire lock：

```python
def test_handle_train_new_start_training_rejects_when_slot_held():
    from capi_web import CAPIWebHandler

    server = MagicMock()
    server.database.get_training_job.return_value = {
        "job_id": "j2", "machine_id": "M", "state": "review", "panel_paths": []
    }
    CAPIWebHandler._train_new_jobs = {}
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()
    CAPIWebHandler._train_slot = {
        "lock": threading.Lock(),
        "active_job_id": "j1",
    }

    h = _make_handler_with_server(server, "/api/train/new/start_training/j2")
    h._handle_train_new_start_training()

    assert h._sent_response[0]["status"] == 409
    body = json.loads(h._sent_response[0]["body"])
    assert body["error"] == "another_job_training"
    assert body["training_job_id"] == "j1"
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_capi_web_train_new.py::test_handle_train_new_start_training_rejects_when_slot_held -v
pytest tests/test_capi_web_train_new.py -v
```
Expected: 全 PASS

- [ ] **Step 6: Commit**

```
git add capi_web.py tests/test_capi_web_train_new.py
git commit -m "feat(web): start_training 用 _train_slot 序列化 GPU 訓練"
```

---

### Task 8: 重寫 `_handle_train_new_cancel` — per-job 路徑

**Files:**
- Modify: `capi_web.py:5414-5469`
- Test: `tests/test_capi_web_train_new.py`

- [ ] **Step 1: Write failing test — 取消 A job 不會 touch B 的 cancel flag**

加到 `tests/test_capi_web_train_new.py`：

```python
def test_handle_train_new_cancel_isolates_flags(tmp_path):
    """取消 j1 不應 touch j2 的 cancel flag 檔。"""
    from capi_web import CAPIWebHandler

    flag_a = tmp_path / "a.cancel"
    flag_b = tmp_path / "b.cancel"

    class AliveProc:
        def poll(self):
            return None

    class AliveThread:
        def is_alive(self):
            return True

    CAPIWebHandler._train_new_jobs = {
        "j1": {
            "thread": AliveThread(), "proc": AliveProc(),
            "cancel_flag": str(flag_a), "log_file": None,
            "cancel_event": threading.Event(),
            "log_lines": [], "log_lock": threading.Lock(),
            "unit_status": {}, "phase": "train",
        },
        "j2": {
            "thread": AliveThread(), "proc": AliveProc(),
            "cancel_flag": str(flag_b), "log_file": None,
            "cancel_event": threading.Event(),
            "log_lines": [], "log_lock": threading.Lock(),
            "unit_status": {}, "phase": "train",
        },
    }
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()

    server = MagicMock()
    server.database.get_training_job.return_value = {
        "job_id": "j1", "machine_id": "M", "state": "train", "panel_paths": []
    }
    h = _make_handler_with_server(server, "/api/train/new/cancel/j1")
    h._handle_train_new_cancel()

    assert flag_a.exists()
    assert not flag_b.exists()
    assert CAPIWebHandler._train_new_jobs["j1"]["cancel_event"].is_set()
    assert not CAPIWebHandler._train_new_jobs["j2"]["cancel_event"].is_set()
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_capi_web_train_new.py::test_handle_train_new_cancel_isolates_flags -v
```
Expected: FAIL

- [ ] **Step 3: 重寫 handler**

替換 `capi_web.py:5414-5469`：

```python
    def _handle_train_new_cancel(self):
        """POST /api/train/new/cancel/<job_id>

        Cancel a job by job_id. 訓練中 job 透過該 job 自己的 cancel flag 檔通知；
        review 階段 job 直接 mark failed；server restart 後 stale job 也標 failed。
        """
        job_id = self.path.rsplit("/", 1)[-1].split("?")[0]
        db = self._capi_server_instance.database
        job = db.get_training_job(job_id)
        if not job:
            self._send_json({"error": "job not found"}, status=404)
            return
        job = self._mark_train_new_stale_if_needed(db, job)

        if job["state"] == "failed":
            self._send_json({"ok": True, "job_id": job_id, "state": "failed"})
            return

        runtime = CAPIWebHandler._get_job_runtime(job_id)

        if job["state"] in ("preprocess", "train"):
            if not self._train_new_worker_alive(job_id):
                db.update_training_job_state(job_id, "failed", error_message="cancelled stale job")
                CAPIWebHandler._drop_job_runtime(job_id)
                # train slot 也清掉
                slot = CAPIWebHandler._train_slot
                with slot["lock"]:
                    if slot.get("active_job_id") == job_id:
                        slot["active_job_id"] = None
                self._send_json({"ok": True, "job_id": job_id, "state": "failed"})
                return

            # 觸發該 job 的 cancel event 並 touch 該 job 的 cancel flag 檔
            if runtime is not None:
                runtime["cancel_event"].set()
                cancel_flag = runtime.get("cancel_flag")
                if cancel_flag:
                    try:
                        Path(cancel_flag).touch()
                    except Exception:
                        pass
                CAPIWebHandler._append_train_new_log(
                    job_id, "收到取消要求，會在目前訓練階段結束後停止"
                )
            self._send_json({"ok": True, "job_id": job_id, "state": job["state"], "cancel_requested": True})
            return

        if job["state"] != "review":
            self._send_json({
                "error": f"job state must be 'review', currently '{job['state']}'"
            }, status=409)
            return

        db.update_training_job_state(job_id, "failed", error_message="cancelled by user")
        CAPIWebHandler._drop_job_runtime(job_id)
        self._send_json({"ok": True, "job_id": job_id, "state": "failed"})
```

- [ ] **Step 4: 修正既有 cancel 測試（line 547-626）**

既有三個測試 (`test_handle_train_new_cancel_marks_review_job_failed`, `_marks_stale_running_job_failed`, `_requests_live_training_stop`) 用了舊 `_train_new_state`。改寫 fixture 為新註冊表：

範本 1（review job）：
```python
def test_handle_train_new_cancel_marks_review_job_failed():
    from capi_web import CAPIWebHandler

    server = MagicMock()
    server.database.get_training_job.return_value = {
        "job_id": "j1", "machine_id": "M", "state": "review", "panel_paths": []
    }
    CAPIWebHandler._train_new_jobs = {
        "j1": {
            "thread": None, "proc": None, "cancel_flag": None,
            "log_file": None, "cancel_event": threading.Event(),
            "log_lines": [], "log_lock": threading.Lock(),
            "unit_status": {}, "phase": "review",
        }
    }
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()

    h = _make_handler_with_server(server, "/api/train/new/cancel/j1")
    h._handle_train_new_cancel()

    server.database.update_training_job_state.assert_called_once_with(
        "j1", "failed", error_message="cancelled by user"
    )
    body = json.loads(h._sent_response[0]["body"])
    assert body["ok"] is True
    assert "j1" not in CAPIWebHandler._train_new_jobs
```

範本 2（stale running，runtime 不存在）：
```python
def test_handle_train_new_cancel_marks_stale_running_job_failed():
    server = MagicMock()
    server.database.get_training_job.return_value = {
        "job_id": "j1", "machine_id": "M", "state": "train", "panel_paths": []
    }
    from capi_web import CAPIWebHandler
    CAPIWebHandler._train_new_jobs = {}
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()
    CAPIWebHandler._train_slot = {"lock": threading.Lock(), "active_job_id": None}
    h = _make_handler_with_server(server, "/api/train/new/cancel/j1")

    h._handle_train_new_cancel()

    server.database.update_training_job_state.assert_called_with(
        "j1", "failed",
        error_message="interrupted: server restarted or training worker is not running",
    )
    body = json.loads(h._sent_response[0]["body"])
    assert body["ok"] is True
    assert body["state"] == "failed"
```

範本 3（live training）：
```python
def test_handle_train_new_cancel_requests_live_training_stop():
    from capi_web import CAPIWebHandler

    class AliveThread:
        def is_alive(self):
            return True

    server = MagicMock()
    server.database.get_training_job.return_value = {
        "job_id": "j1", "machine_id": "M", "state": "train", "panel_paths": []
    }
    cancel_event = threading.Event()
    CAPIWebHandler._train_new_jobs = {
        "j1": {
            "thread": AliveThread(), "proc": None, "cancel_flag": None,
            "log_file": None, "cancel_event": cancel_event,
            "log_lines": [], "log_lock": threading.Lock(),
            "unit_status": {}, "phase": "train",
        }
    }
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()
    h = _make_handler_with_server(server, "/api/train/new/cancel/j1")

    h._handle_train_new_cancel()

    server.database.update_training_job_state.assert_not_called()
    assert cancel_event.is_set()
    body = json.loads(h._sent_response[0]["body"])
    assert body["cancel_requested"] is True
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_capi_web_train_new.py -v
```
Expected: 全 PASS

- [ ] **Step 6: Commit**

```
git add capi_web.py tests/test_capi_web_train_new.py
git commit -m "feat(web): cancel handler 改成 per-job runtime 路徑，修跨 job cancel 互打事故"
```

---

### Task 9: 重寫 `_handle_train_new_status` 與 `_handle_train_new_page` 用 per-job log

**Files:**
- Modify: `capi_web.py:5224-5264`（status）
- Modify: `capi_web.py:4754-4771`（train_new page）
- Test: `tests/test_capi_web_train_new.py`

- [ ] **Step 1: Write failing test — 兩個 job 的 log 不會互串**

加到 `tests/test_capi_web_train_new.py`：

```python
def test_handle_train_new_status_returns_per_job_log():
    from capi_web import CAPIWebHandler

    CAPIWebHandler._train_new_jobs = {
        "jA": {
            "thread": None, "proc": None, "cancel_flag": None,
            "log_file": None, "cancel_event": threading.Event(),
            "log_lines": ["[hh:mm:ss] A line"], "log_lock": threading.Lock(),
            "unit_status": {}, "phase": "review",
        },
        "jB": {
            "thread": None, "proc": None, "cancel_flag": None,
            "log_file": None, "cancel_event": threading.Event(),
            "log_lines": ["[hh:mm:ss] B line"], "log_lock": threading.Lock(),
            "unit_status": {}, "phase": "review",
        },
    }
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()

    server = MagicMock()
    server.database.get_training_job.side_effect = lambda jid: {
        "job_id": jid, "machine_id": "M", "state": "review",
        "started_at": None, "completed_at": None,
        "output_bundle": None, "error_message": None,
        "panel_paths": [],
    }

    h = _make_handler_with_server(server, "/api/train/new/status?job_id=jA")
    h._handle_train_new_status()
    body = json.loads(h._sent_response[0]["body"])
    assert body["log_lines"] == ["[hh:mm:ss] A line"]

    h2 = _make_handler_with_server(server, "/api/train/new/status?job_id=jB")
    h2._handle_train_new_status()
    body2 = json.loads(h2._sent_response[0]["body"])
    assert body2["log_lines"] == ["[hh:mm:ss] B line"]
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_capi_web_train_new.py::test_handle_train_new_status_returns_per_job_log -v
```
Expected: FAIL（舊 status 讀全域 `state["log_lines"]`）

- [ ] **Step 3: 重寫 status handler**

替換 `capi_web.py:5223-5264`：

```python
    def _handle_train_new_status(self):
        """GET /api/train/new/status?job_id=X

        無 job_id：列出最新 active job 的狀態（fallback to idle）。
        有 job_id：查 per-job runtime 的 log_lines / unit_status。
        """
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(self.path).query)
        job_id = (qs.get("job_id") or [""])[0]

        db = self._capi_server_instance.database

        if not job_id:
            job = db.get_active_training_job()
            if not job:
                self._send_json({"state": "idle"})
                return
            job = self._mark_train_new_stale_if_needed(db, job)
            job_id = job["job_id"]
        else:
            job = db.get_training_job(job_id)
            if not job:
                self._send_json({"error": "job not found"}, status=404)
                return
            job = self._mark_train_new_stale_if_needed(db, job)

        runtime = CAPIWebHandler._get_job_runtime(job_id)
        if runtime is None:
            log_lines = []
            unit_status = {}
        else:
            with runtime["log_lock"]:
                log_lines = list(runtime["log_lines"][-100:])
                unit_status = dict(runtime.get("unit_status") or {})

        resp = {
            "job_id": job["job_id"], "machine_id": job["machine_id"],
            "state": job["state"],
            "started_at": job["started_at"], "completed_at": job["completed_at"],
            "output_bundle": job["output_bundle"], "error_message": job["error_message"],
            "log_lines": log_lines,
            "unit_status": unit_status,
            "worker_alive": self._train_new_worker_alive(job_id),
        }
        self._send_json(resp)
```

- [ ] **Step 4: 重寫 `_handle_train_new_page` 顯示多 job**

替換 `capi_web.py:4754-4771`：

```python
    def _handle_train_new_page(self):
        """GET /train/new — Step 1 with optional list of open jobs."""
        db = self._capi_server_instance.database
        all_active = db.list_active_training_jobs()
        # 把 stale job 補刀（preprocess/train 但 worker 死）
        cleaned = []
        for j in all_active:
            j = self._mark_train_new_stale_if_needed(db, j)
            if j and j["state"] in ("preprocess", "review", "train"):
                cleaned.append(j)

        template = self.jinja_env.get_template("train_new/step1_select.html")
        html = template.render(
            request_path="/train/new",
            active_jobs=cleaned,
        )
        self._send_response(200, html)
```

- [ ] **Step 5: 更新 step1_select.html banner**

替換 `templates/train_new/step1_select.html:16-33`（`{% if active_review_job %}` 區塊）為：

```html
{% if active_jobs %}
<div style="background:#2f3347;border:1px solid #89b4fa;border-radius:10px;padding:14px 16px;margin-top:16px;">
  <div style="color:#cdd6f4;font-weight:700;font-size:.92rem;margin-bottom:8px;">進行中的訓練 job ({{ active_jobs|length }})</div>
  {% for j in active_jobs %}
  <div style="display:flex;justify-content:space-between;gap:14px;align-items:center;flex-wrap:wrap;padding:8px 0;border-top:1px solid #45475a;">
    <div>
      <div style="color:#cdd6f4;font-size:.86rem;">
        <span style="font-family:monospace;">{{ j.job_id }}</span>
        <span style="margin-left:8px;color:#a6adc8;">{{ j.machine_id }}</span>
      </div>
      <div style="color:#a6adc8;font-size:.78rem;margin-top:2px;">
        state: <span style="color:{% if j.state == 'review' %}#a6e3a1{% elif j.state == 'train' %}#fab387{% else %}#89b4fa{% endif %};">{{ j.state }}</span>
      </div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;">
      {% if j.state == 'review' %}
      <a href="/train/new/review/{{ j.job_id }}"
         style="background:#89b4fa;color:#1e1e2e;border:none;border-radius:5px;padding:6px 12px;font-weight:700;text-decoration:none;font-size:.82rem;">繼續審核 →</a>
      {% elif j.state in ('preprocess', 'train') %}
      <a href="/train/new/progress?job_id={{ j.job_id }}"
         style="background:#89b4fa;color:#1e1e2e;border:none;border-radius:5px;padding:6px 12px;font-weight:700;text-decoration:none;font-size:.82rem;">查看進度 →</a>
      {% endif %}
      <button onclick="cancelReviewJob('{{ j.job_id }}')"
        style="background:#313244;color:#f38ba8;border:1px solid #f38ba8;border-radius:5px;padding:6px 12px;font-weight:700;font-size:.82rem;cursor:pointer;">
        取消
      </button>
    </div>
  </div>
  {% endfor %}
</div>
{% endif %}
```

- [ ] **Step 6: 修舊測試 `test_handle_train_new_page_renders_step1_when_active_job_is_review`**

該測試（line ~663）原本 mock `get_active_training_job` 回 review job。改成 mock `list_active_training_jobs` 回 list：

```python
def test_handle_train_new_page_renders_step1_when_active_job_is_review():
    server = MagicMock()
    server.database.list_active_training_jobs.return_value = [
        {"job_id": "j1", "machine_id": "M", "state": "review", "panel_paths": []}
    ]
    h = _make_handler_with_server(server, "/train/new")
    template = MagicMock()
    template.render.return_value = "<html>step1</html>"
    h.jinja_env = MagicMock()
    h.jinja_env.get_template.return_value = template

    h._handle_train_new_page()

    h.jinja_env.get_template.assert_called_with("train_new/step1_select.html")
    # 驗證 active_jobs 有傳入
    _, kwargs = template.render.call_args
    assert any(j["job_id"] == "j1" for j in kwargs.get("active_jobs", []))
```

- [ ] **Step 7: Run tests**

```
pytest tests/test_capi_web_train_new.py -v
```
Expected: 全 PASS

- [ ] **Step 8: Commit**

```
git add capi_web.py templates/train_new/step1_select.html tests/test_capi_web_train_new.py
git commit -m "feat(web): status / train-new page 改 per-job 路徑，Step 1 banner 列出多 job"
```

---

### Task 10: 移除 `_train_new_state` legacy 結構與 fallback 路徑

**Files:**
- Modify: `capi_web.py`（移除舊 class attr + helper 內 legacy 分支 + bootstrap 重設）
- Test: `tests/test_capi_web_train_new.py`

- [ ] **Step 1: 全文檢查 `_train_new_state` 殘留**

```
grep -n "_train_new_state" capi_web.py
```

- [ ] **Step 2: 移除 class attribute**

刪除 `capi_web.py:191-205` 整個 `_train_new_state: dict = {...}` 定義 + 上方註解（line 186-205）。

- [ ] **Step 3: 移除 helper 的 legacy 分支**

把 `_append_train_new_log` 改回簽名只接 `job_id`：

```python
    @classmethod
    def _append_train_new_log(cls, job_id: str, msg: str) -> None:
        runtime = cls._get_job_runtime(job_id)
        if runtime is None:
            return
        with runtime["log_lock"]:
            runtime["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            if len(runtime["log_lines"]) > 500:
                runtime["log_lines"] = runtime["log_lines"][-500:]
            m = _TRAIN_UNIT_LOG_RE.search(msg)
            if m:
                runtime.setdefault("unit_status", {})[m.group(1)] = _TRAIN_UNIT_STATUS_MAP[m.group(2)]
```

`_train_new_cancel_event` 改回只接 `job_id`：

```python
    @classmethod
    def _train_new_cancel_event(cls, job_id: str) -> threading.Event:
        runtime = cls._get_job_runtime(job_id)
        if runtime is None:
            return threading.Event()
        return runtime.setdefault("cancel_event", threading.Event())
```

`_train_new_worker_alive` 移除 legacy fallback：

```python
    @classmethod
    def _train_new_worker_alive(cls, job_id: str) -> bool:
        runtime = cls._get_job_runtime(job_id)
        if runtime is None:
            return False
        thread = runtime.get("thread")
        if thread is not None and thread.is_alive():
            return True
        proc = runtime.get("proc")
        if proc is not None and proc.poll() is None:
            return True
        return False
```

`_mark_train_new_stale_if_needed` 移除 legacy fallback：

```python
    @classmethod
    def _mark_train_new_stale_if_needed(cls, db, job: Optional[dict]) -> Optional[dict]:
        if not job or job.get("state") not in ("preprocess", "train"):
            return job
        if cls._train_new_worker_alive(job["job_id"]):
            return job

        error = "interrupted: server restarted or training worker is not running"
        db.update_training_job_state(job["job_id"], "failed", error_message=error)
        cls._drop_job_runtime(job["job_id"])
        slot = cls._train_slot
        with slot["lock"]:
            if slot.get("active_job_id") == job["job_id"]:
                slot["active_job_id"] = None
        updated = dict(job)
        updated["state"] = "failed"
        updated["error_message"] = error
        return updated
```

- [ ] **Step 4: 移除 bootstrap 中舊重設**

刪除 `capi_web.py:6627-6633` 的 `CAPIWebHandler._train_new_state = {...}` 整段（保留下方剛加的 `_train_new_jobs` / `_train_slot` 重設）。

- [ ] **Step 5: 全測試套件**

```
pytest tests/test_capi_web_train_new.py tests/test_capi_database_train.py -v
```
Expected: 全 PASS

- [ ] **Step 6: Smoke server start**

```
python -c "from capi_web import CAPIWebHandler; print(CAPIWebHandler._train_new_jobs, CAPIWebHandler._train_slot)"
```
Expected: `{} {'lock': ..., 'active_job_id': None}`

- [ ] **Step 7: Commit**

```
git add capi_web.py
git commit -m "refactor(web): 移除 _train_new_state legacy 結構與 fallback 路徑"
```

---

## Self-Review

**Spec coverage:**
- ✅ 多 job 共存 → Task 6 移除 wizard singleton
- ✅ 訓練槽序列化 → Task 7 `_train_slot`
- ✅ 跨 job cancel 互打修正 → Task 8 per-job cancel flag
- ✅ Per-job log → Task 9 status handler
- ✅ Step 1 列表式 banner → Task 9 + template 改寫
- ✅ DB 多 job 查詢 → Task 1 `list_active_training_jobs`
- ✅ Race regression test → Task 6 step 1
- ✅ Cancel isolation test → Task 8 step 1
- ✅ Status isolation test → Task 9 step 1
- ✅ Training slot test → Task 7 step 1
- ✅ Legacy 清理 → Task 10

**Placeholder scan:** 通過。所有 step 都有具體 code / 指令。

**Type consistency:** runtime dict shape 在 Task 2/4/5/8/9 一致（thread / proc / cancel_flag / log_file / cancel_event / log_lines / log_lock / unit_status / phase）。Slot dict shape (`lock` + `active_job_id`) 在 Task 2/7/8/10 一致。

**Risks:**
- Task 6 race test 使用 `MagicMock` 模擬 DB，真實 SQLite 在 ThreadingHTTPServer 下也應該無 race（因為 `create_training_job` 用 SQLite 鎖 + `INSERT OR IGNORE`／PRIMARY KEY 已可避免重複 job_id）。但 panel_paths 重複新增到不同 job 的 tile_pool 沒有 unique 約束，多 job 會各自切 tile，正常。
- 既有 cancel/preprocess/training tests 未直接 unit test（thread 不會跑），refactor 後這部分覆蓋率不變；race 與 slot test 是新加的。

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-08-train-job-multi-tenant.md`.** 進入實作。
