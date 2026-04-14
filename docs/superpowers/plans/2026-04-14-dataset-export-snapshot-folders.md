# Dataset Export Snapshot Folders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `DatasetExporter.run` 由「全部累積進單一 `base_dir`、帶 stale cleanup」改成「每次 run 產生 `base_dir/<YYYYMMDD_HHMMSS>/` 獨立 snapshot、只輸出相對於歷史所有 job 新增的 sample」。Web Gallery 相應支援 job 切換。

**Architecture:**
- `base_dir` 語意由「資料夾本體」變成「所有 snapshot 的父目錄」
- 每次 export 掃 `base_dir/*/manifest.csv` 建立「歷史 sample_id 集合」，本次只處理集合外的新 candidate
- 舊 job 資料夾從不被後續 run 修改／刪除，使用者手動分好的檔案不會被擾動
- 不刪 stale、不跨 job move；`_process_candidate` 移除 `existing_row` 分支
- Gallery / file API / sample move / sample delete 全部加 `job` 參數，作用於單一 job 資料夾

**Tech Stack:** Python 3.11, pytest, stdlib http.server, Jinja2（既有）

**約定：** sample 內部目錄結構（`{label}/{prefix}/crop/{filename}`）維持不變，本次只新增 job 層。

---

## File Structure

**Modify:**
- `capi_dataset_export.py` — `DatasetExporter.run` 改寫、新增 job 輔助函式、精簡 `_process_candidate`
- `capi_web.py` — gallery 路由 / file / sample delete / sample move 都接 `job` 參數；新增 job 列表 helper
- `templates/dataset_gallery.html` — 上方加 job 下拉選單；URL builder 帶 `job`
- `tests/test_dataset_export.py` — 更新現有測試、刪除不再適用的測試、新增 snapshot 行為測試

**Do NOT modify:**
- Manifest schema (MANIFEST_FIELDS) — 維持原欄位，避免 breaking 歷史 manifest
- `move_sample_files` / `relabel_sample` / `delete_sample` — 這些仍供 gallery 單一 job 內編輯使用，但呼叫端改傳 `base_dir/<job>` 而非 `base_dir`
- 檔名規則 (`build_sample_filename`) 與 crop 目錄結構 (`{label}/{prefix}/crop/{filename}`)

---

## 設計決策（實作前須知）

### 何為「job 資料夾」
`base_dir` 下任何直接子目錄只要包含 `manifest.csv` 就視為 job 資料夾。名稱慣例為 `YYYYMMDD_HHMMSS`，但程式不強制 — 掃描時以「子目錄 + manifest.csv 存在」為準，排序用目錄名 lexicographic（YYYYMMDD_HHMMSS 格式天然遞增）。

### skip_existing 新語意
- `skip_existing=True`（預設）：sample_id 出現在任何 prior job 的 manifest → skip，不輸出到本次 job
- `skip_existing=False`：忽略 prior job，照樣處理（會在新 job 資料夾裡產生同 sample_id 的副本）

### 空 job 處理
若本次沒有任何 candidate 被實際寫入（全部 skip 或 candidate 集合為空），**不**建立 job 資料夾、**不**寫 manifest。`JobSummary.output_dir` 填 `base_dir`，`total=0`。

### Legacy 扁平結構
若 `base_dir` 根目錄（非子目錄）存在 `manifest.csv`（舊版輸出），本次改動**不做自動遷移**。
- Export 的 skip-set 掃描只看子目錄，不讀根目錄 manifest
- Gallery job 列表也不顯示根目錄 — 使用者若需要繼續瀏覽舊資料，手動把舊資料移到 `base_dir/legacy_YYYYMMDD/` 即可

這個選擇的理由：讓改動單純、避免在自動遷移路徑上寫隱藏複雜度。

### 路徑 traversal 安全
Gallery file API 接 `job` 參數時，先組 `base_dir/<job>/<path>`，再用既有的 `resolve() + relative_to(base_dir)` 守衛。`job` 參數僅允許 `[A-Za-z0-9_]` 字元（dropdown 只會給 `YYYYMMDD_HHMMSS`，但 API 仍做白名單驗證）。

---

## Task 1: Add `list_job_dirs` helper to `capi_dataset_export.py`

**Files:**
- Modify: `capi_dataset_export.py` — 在 `write_manifest` 附近（約 L201 之後）新增
- Test: `tests/test_dataset_export.py`

- [ ] **Step 1: Write the failing tests**

在 `tests/test_dataset_export.py` 檔尾新增：

```python
def test_list_job_dirs_returns_sorted_subdirs_with_manifest(tmp_path):
    from capi_dataset_export import list_job_dirs
    # 三個合法 job，一個沒 manifest，一個 legacy root-level manifest
    for name in ["20260414_100000", "20260414_120000", "20260413_090000"]:
        d = tmp_path / name
        d.mkdir()
        (d / "manifest.csv").write_text("sample_id\n", encoding="utf-8")
    (tmp_path / "no_manifest_dir").mkdir()
    (tmp_path / "manifest.csv").write_text("sample_id\n", encoding="utf-8")  # legacy root

    jobs = list_job_dirs(tmp_path)
    assert [j.name for j in jobs] == ["20260413_090000", "20260414_100000", "20260414_120000"]


def test_list_job_dirs_empty_when_base_missing(tmp_path):
    from capi_dataset_export import list_job_dirs
    assert list_job_dirs(tmp_path / "does_not_exist") == []


def test_load_known_sample_ids_unions_across_jobs(tmp_path):
    from capi_dataset_export import load_known_sample_ids, write_manifest
    job_a = tmp_path / "20260414_100000"
    job_b = tmp_path / "20260414_110000"
    write_manifest(job_a / "manifest.csv", {
        "sid_1": {"sample_id": "sid_1", "status": "ok"},
        "sid_2": {"sample_id": "sid_2", "status": "skipped_no_source"},
    })
    write_manifest(job_b / "manifest.csv", {
        "sid_3": {"sample_id": "sid_3", "status": "ok"},
    })
    assert load_known_sample_ids(tmp_path) == {"sid_1", "sid_2", "sid_3"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /c/Users/rh.syu/Desktop/CAPI01_AD
python -m pytest tests/test_dataset_export.py::test_list_job_dirs_returns_sorted_subdirs_with_manifest tests/test_dataset_export.py::test_list_job_dirs_empty_when_base_missing tests/test_dataset_export.py::test_load_known_sample_ids_unions_across_jobs -v
```
Expected: FAIL — `list_job_dirs` / `load_known_sample_ids` not defined.

- [ ] **Step 3: Add the two helpers**

在 `capi_dataset_export.py` 的 `write_manifest` 函式之後（約 L201 結束處）插入：

```python
def list_job_dirs(base_dir: Path) -> List[Path]:
    """列出 base_dir 下的 job 資料夾（子目錄且含 manifest.csv），依目錄名升冪排序。

    不含 base_dir 根目錄層（legacy 扁平結構）的 manifest.csv。
    不存在的 base_dir 回空 list。
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []
    out: List[Path] = []
    for child in sorted(base_dir.iterdir(), key=lambda p: p.name):
        if child.is_dir() and (child / "manifest.csv").exists():
            out.append(child)
    return out


def load_known_sample_ids(base_dir: Path) -> set:
    """Union 所有 job 資料夾 manifest.csv 內的 sample_id。

    供 DatasetExporter.run 判定新 candidate 用：只要任一 prior job 有此 sample_id → skip。
    status 不限制（即使 skipped_* 也算已處理過），避免使用者明明看過該 sample 卻
    被下次 run 重新蒐集而產生重複干擾。
    """
    ids: set = set()
    for job_dir in list_job_dirs(base_dir):
        rows = read_manifest(job_dir / "manifest.csv")
        ids.update(rows.keys())
    return ids
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_dataset_export.py::test_list_job_dirs_returns_sorted_subdirs_with_manifest tests/test_dataset_export.py::test_list_job_dirs_empty_when_base_missing tests/test_dataset_export.py::test_load_known_sample_ids_unions_across_jobs -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add capi_dataset_export.py tests/test_dataset_export.py
git commit -m "feat(dataset_export): add list_job_dirs / load_known_sample_ids helpers"
```

---

## Task 2: Rewrite `DatasetExporter.run` to produce dated snapshot folders

**Files:**
- Modify: `capi_dataset_export.py` — 改寫 `run` (L663~L767) 與 `_process_candidate` (L780~L868)

- [ ] **Step 1: Read current `run` end (lines 755-770) to confirm JobSummary build site**

```bash
python -c "
with open('capi_dataset_export.py', encoding='utf-8') as f:
    lines = f.readlines()
for i in range(755, 775):
    print(f'{i+1}: {lines[i]}', end='')
"
```

- [ ] **Step 2: Replace `DatasetExporter.run` body**

找到 L663 `def run(self, days: int, include_true_ng: bool, skip_existing: bool,` 開頭，取代整個 method 內容為：

```python
    def run(self, days: int, include_true_ng: bool, skip_existing: bool,
            status_callback=None, cancel_event=None) -> JobSummary:
        """執行一次 export job。

        與舊版差異：
          - 輸出到 self.base_dir/<YYYYMMDD_HHMMSS>/（每次跑產生獨立 snapshot）
          - skip 基準改成「所有歷史 job 的 sample_id 聯集」
          - 不做 stale cleanup（舊 job 永遠不動）
          - skip_existing=True 時跳過歷史已匯出的 sample；False 時照常處理（會在新 job 產生副本）
          - 本次沒有任何 row 要寫 → 不建立 job 資料夾、output_dir 填 base_dir、total=0
        """
        from capi_server import resolve_unc_path  # lazy import 避免 circular

        started_at = datetime.now()
        job_id = started_at.strftime("job_%Y%m%d_%H%M%S")
        job_folder_name = started_at.strftime("%Y%m%d_%H%M%S")
        job_dir = self.base_dir / job_folder_name
        manifest_path = job_dir / "manifest.csv"

        self.base_dir.mkdir(parents=True, exist_ok=True)

        # 1. 掃歷史 job，建 known sample_id 集合
        known_ids = load_known_sample_ids(self.base_dir) if skip_existing else set()

        # 2. 蒐集 candidates
        candidates, diag = self.collect_candidates_with_diagnostics(
            days=days, include_true_ng=include_true_ng
        )
        total = len(candidates)
        logger.info(
            "Collected %d candidates (days=%d, include_true_ng=%s, known_prior=%d)",
            total, days, include_true_ng, len(known_ids),
        )

        new_rows: Dict[str, Dict[str, str]] = {}
        labels_count: Dict[str, int] = {}
        skipped_count: Dict[str, int] = {}

        # 3. 處理每個 candidate
        for idx, cand in enumerate(candidates, start=1):
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Job cancelled at %d/%d", idx, total)
                break

            if status_callback:
                try:
                    status_callback(idx, total, cand.glass_id)
                except Exception:
                    logger.exception("status_callback error")

            if cand.sample_id in known_ids:
                skipped_count["already_exists"] = skipped_count.get("already_exists", 0) + 1
                continue

            source_path = self._resolve_source_path(cand.image_path, resolve_unc_path)
            new_row = self._process_candidate(cand=cand, source_path=source_path, job_dir=job_dir)
            new_rows[cand.sample_id] = new_row

            status = new_row["status"]
            if status == "ok":
                labels_count[new_row["label"]] = labels_count.get(new_row["label"], 0) + 1
            else:
                skipped_count[status] = skipped_count.get(status, 0) + 1

        # 4. 空 job：不建資料夾、不寫 manifest
        if not new_rows:
            logger.info("No new samples for this job; skipping folder creation")
            finished_at = datetime.now()
            return JobSummary(
                job_id=job_id,
                started_at=started_at.isoformat(timespec="seconds"),
                finished_at=finished_at.isoformat(timespec="seconds"),
                duration_sec=round((finished_at - started_at).total_seconds(), 2),
                total=0,
                labels=labels_count,
                skipped=skipped_count,
                output_dir=str(self.base_dir),
                diagnostics=diag,
            )

        # 5. 寫 manifest
        write_manifest(manifest_path, new_rows)
        finished_at = datetime.now()
        return JobSummary(
            job_id=job_id,
            started_at=started_at.isoformat(timespec="seconds"),
            finished_at=finished_at.isoformat(timespec="seconds"),
            duration_sec=round((finished_at - started_at).total_seconds(), 2),
            total=len(new_rows),
            labels=labels_count,
            skipped=skipped_count,
            output_dir=str(job_dir),
            diagnostics=diag,
        )
```

- [ ] **Step 3: Replace `_process_candidate` body**

找到 L780 `def _process_candidate(` 開頭，取代整個 method 為：

```python
    def _process_candidate(
        self, cand: SampleCandidate, source_path: Path, job_dir: Path,
    ) -> Dict[str, str]:
        """處理單一 candidate，回傳 manifest row。

        新架構不再處理「既有 row 的 label 變動 / 強制重做」分支 —
        已存在於任何歷史 job 的 sample_id 早在 run() 入口就被 known_ids skip 掉。
        """
        row = self._build_row_stub(cand)

        if not source_path.exists():
            row["status"] = "skipped_no_source"
            logger.warning("Source not found: %s (sample_id=%s)", source_path, cand.sample_id)
            return row

        img = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            row["status"] = "skipped_no_source"
            return row
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if cand.source_type == "patchcore_tile":
            crop = crop_patchcore_tile(img, cand.tile_x, cand.tile_y, cand.tile_w, cand.tile_h)
            defect_x = cand.tile_x + cand.tile_w // 2
            defect_y = cand.tile_y + cand.tile_h // 2
            sample_key = f"tile{cand.tile_idx}"
        else:  # edge_defect
            H, W = img.shape[:2]
            half = CROP_SIZE // 2
            valid_w = min(W, cand.edge_center_x + half) - max(0, cand.edge_center_x - half)
            valid_h = min(H, cand.edge_center_y + half) - max(0, cand.edge_center_y - half)
            if max(0, valid_w) * max(0, valid_h) < (CROP_SIZE * CROP_SIZE) * 0.25:
                row["status"] = "skipped_out_of_bounds"
                return row
            crop = crop_edge_defect(img, cand.edge_center_x, cand.edge_center_y)
            defect_x = cand.edge_center_x
            defect_y = cand.edge_center_y
            sample_key = f"edge{cand.edge_defect_id}"

        filename = build_sample_filename(
            glass_id=cand.glass_id, image_name=cand.image_name,
            sample_key=sample_key, inference_timestamp=cand.inference_timestamp,
        )
        crop_rel = f"{cand.label}/{cand.prefix}/crop/{filename}"
        crop_dst = job_dir / crop_rel
        crop_dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_dst), crop)

        row["crop_path"] = crop_rel
        row["heatmap_path"] = ""
        row["defect_x"] = str(defect_x)
        row["defect_y"] = str(defect_y)
        row["status"] = "ok"
        return row
```

- [ ] **Step 4: Quick sanity run**

```bash
python -c "from capi_dataset_export import DatasetExporter, list_job_dirs, load_known_sample_ids; print('import ok')"
```
Expected: prints `import ok`, no syntax error.

- [ ] **Step 5: Commit**

```bash
git add capi_dataset_export.py
git commit -m "refactor(dataset_export): write snapshots into base_dir/<YYYYMMDD_HHMMSS>/ per run"
```

---

## Task 3: Update existing tests; delete obsolete ones; add snapshot tests

**Files:**
- Modify: `tests/test_dataset_export.py`

舊測試編碼了已移除的行為，直接改寫為新架構斷言。

- [ ] **Step 1: Delete obsolete tests**

移除 `tests/test_dataset_export.py` 裡這三個（grep 定位）：
- `test_exporter_run_second_pass_moves_on_label_change` (L478 附近)
- `test_exporter_run_cleans_stale_manifest_entries` (L635 附近)
- `test_exporter_run_counts_already_exists_on_second_pass` (L600 附近) — 將在 Step 3 以新架構重寫

Run: `python -m pytest tests/test_dataset_export.py -v --collect-only 2>&1 | grep -E "(moves_on_label_change|cleans_stale_manifest|counts_already_exists)"`
Expected: 無輸出。

- [ ] **Step 2: Update `test_exporter_run_end_to_end` to assert snapshot subfolder**

在 `tests/test_dataset_export.py` 找到 `test_exporter_run_end_to_end` (L408)，把期望的 crop 路徑與 manifest 位置改為 `<base_dir>/<YYYYMMDD_HHMMSS>/...`。具體：

找到既有的 `summary = exporter.run(days=3, include_true_ng=True, skip_existing=True)` 那行之後，把原本對 `tmp_path / "manifest.csv"` / `tmp_path / "true_ng" / ...` 的 assertion 改為：

```python
    assert summary.total == 1  # 或原本的數字，以實測為準
    job_dir = Path(summary.output_dir)
    assert job_dir.parent == tmp_path.resolve()
    assert job_dir.name.count("_") == 1 and len(job_dir.name) == 15  # YYYYMMDD_HHMMSS
    assert (job_dir / "manifest.csv").exists()
    rows = read_manifest(job_dir / "manifest.csv")
    assert len(rows) == 1
    row = next(iter(rows.values()))
    assert (job_dir / row["crop_path"]).exists()
```

（保留測試原本注入 DB fake 的 setup，只改輸出斷言）

- [ ] **Step 3: Add new snapshot-behaviour tests**

在檔尾新增：

```python
def test_exporter_run_second_pass_writes_to_new_job_and_skips_known(tmp_path, monkeypatch):
    """第二次跑：已在前一 job 出現的 sample_id 不會再被寫入新 job。"""
    from capi_dataset_export import DatasetExporter, read_manifest, list_job_dirs
    import capi_dataset_export as de

    # 用 monkeypatch 建 fake DB 與 source image（沿用本檔其他測試的 fixture pattern）
    # 若現有檔案已有可複用的 _make_fake_db / fake image 工具則直接 reuse
    db, src_dir = _setup_fake_db_single_candidate(tmp_path)  # helper assumed in this file
    monkeypatch.setattr("capi_server.resolve_unc_path", lambda p, m: p)

    exporter = DatasetExporter(db=db, base_dir=str(tmp_path / "out"), path_mapping={})

    s1 = exporter.run(days=3, include_true_ng=True, skip_existing=True)
    assert s1.total == 1

    # 第二次 run 須產生不同的 job_dir（強制至少 1 秒差）
    import time as _t; _t.sleep(1.1)
    s2 = exporter.run(days=3, include_true_ng=True, skip_existing=True)
    assert s2.total == 0
    assert s2.skipped.get("already_exists", 0) == 1

    jobs = list_job_dirs(tmp_path / "out")
    assert len(jobs) == 1, "第二次 run 沒新增 sample → 不該建立空資料夾"


def test_exporter_run_does_not_touch_prior_job_folder(tmp_path, monkeypatch):
    """使用者手動在舊 job 資料夾裡新增/刪檔，下一次 run 都不會動它。"""
    from capi_dataset_export import DatasetExporter, list_job_dirs

    db, _ = _setup_fake_db_single_candidate(tmp_path)
    monkeypatch.setattr("capi_server.resolve_unc_path", lambda p, m: p)
    exporter = DatasetExporter(db=db, base_dir=str(tmp_path / "out"), path_mapping={})

    s1 = exporter.run(days=3, include_true_ng=True, skip_existing=True)
    first_job = Path(s1.output_dir)
    user_file = first_job / "USER_ADDED.txt"
    user_file.write_text("human touched this", encoding="utf-8")
    old_manifest = (first_job / "manifest.csv").read_bytes()

    import time as _t; _t.sleep(1.1)
    exporter.run(days=3, include_true_ng=True, skip_existing=True)

    assert user_file.exists(), "第一次 job 的使用者檔案被動到"
    assert (first_job / "manifest.csv").read_bytes() == old_manifest


def test_exporter_run_skip_existing_false_forces_reprocess(tmp_path, monkeypatch):
    """skip_existing=False 時，即使 prior job 有相同 sample_id 也要寫進新 job。"""
    from capi_dataset_export import DatasetExporter, list_job_dirs

    db, _ = _setup_fake_db_single_candidate(tmp_path)
    monkeypatch.setattr("capi_server.resolve_unc_path", lambda p, m: p)
    exporter = DatasetExporter(db=db, base_dir=str(tmp_path / "out"), path_mapping={})

    exporter.run(days=3, include_true_ng=True, skip_existing=True)
    import time as _t; _t.sleep(1.1)
    s2 = exporter.run(days=3, include_true_ng=True, skip_existing=False)

    assert s2.total == 1
    assert len(list_job_dirs(tmp_path / "out")) == 2
```

**若 `_setup_fake_db_single_candidate` 不存在** — 在 test 檔頂部加：

```python
def _setup_fake_db_single_candidate(tmp_path):
    """最小 fake DB + 一個可讀取的原圖檔，供 snapshot 測試重複使用。"""
    # 照現有 test_exporter_run_end_to_end 的模式實作
    # （讀 L408~L460 區段後複製其中 DB stub + cv2.imwrite 的樣板）
    raise NotImplementedError("請以 test_exporter_run_end_to_end 作範例提取")
```

實作時把 `test_exporter_run_end_to_end` 內建 DB stub 與原圖生成的樣板抽成此 helper。

- [ ] **Step 4: Run full dataset_export test file**

```bash
python -m pytest tests/test_dataset_export.py -v
```
Expected: 全綠（無 FAIL、無 ERROR）。若有剩餘舊測試失敗，逐個更新 assertion；不要回填被刻意移除的行為。

- [ ] **Step 5: Commit**

```bash
git add tests/test_dataset_export.py
git commit -m "test(dataset_export): assert snapshot-folder behavior; drop obsolete stale/label-move tests"
```

---

## Task 4: Wire gallery to show per-job view (`capi_web.py`)

**Files:**
- Modify: `capi_web.py` — `_dataset_export_base_dir` 下方新增 job helper；改寫 `_handle_dataset_gallery_page`、`_handle_dataset_export_file`、`_handle_dataset_sample_delete`、`_handle_dataset_sample_move`

- [ ] **Step 1: Add job helper methods below `_dataset_export_base_dir`**

在 `capi_web.py` L3015 之後（緊接 `_dataset_export_base_dir` 結尾）新增：

```python
    _JOB_ID_RE = __import__("re").compile(r"^[A-Za-z0-9_]+$")

    def _dataset_list_jobs(self) -> list:
        """回傳 base_dir 下所有 job 資料夾名稱（字串），依名稱降冪（最新在前）"""
        from capi_dataset_export import list_job_dirs
        base = self._dataset_export_base_dir()
        return [p.name for p in reversed(list_job_dirs(base))]

    def _dataset_resolve_job_dir(self, job_id: str) -> Optional[Path]:
        """驗證 job_id 字元集 + 必須存在 + 必須有 manifest.csv；無效回 None"""
        if not job_id or not self._JOB_ID_RE.match(job_id):
            return None
        base = self._dataset_export_base_dir()
        cand = (base / job_id).resolve()
        try:
            cand.relative_to(base)
        except ValueError:
            return None
        if not cand.is_dir() or not (cand / "manifest.csv").exists():
            return None
        return cand
```

- [ ] **Step 2: Replace `_handle_dataset_gallery_page` body**

找到 `_handle_dataset_gallery_page` (L3017)，整段換成：

```python
    def _handle_dataset_gallery_page(self, query: dict):
        """GET /dataset_gallery — 樣本瀏覽頁（按 job 資料夾切換）"""
        from capi_dataset_export import read_manifest

        def _q(key, default=None):
            v = query.get(key)
            if isinstance(v, list):
                return v[0] if v else default
            return v if v is not None else default

        jobs = self._dataset_list_jobs()
        current_job = _q("job", "") or (jobs[0] if jobs else "")
        current_label = _q("label", "") or ""
        current_prefix = _q("prefix", "") or ""
        try:
            page = max(1, int(_q("page", "1") or 1))
        except (TypeError, ValueError):
            page = 1
        try:
            limit = int(_q("limit", "48") or 48)
            limit = max(1, min(limit, 500))
        except (TypeError, ValueError):
            limit = 48

        base_dir = self._dataset_export_base_dir()
        items_all: list = []
        manifest_error = ""
        label_counts: dict = {}
        prefixes_set: set = set()
        job_dir = None

        if not jobs:
            manifest_error = f"尚未有任何 export job 資料夾：{base_dir}"
        elif not current_job:
            manifest_error = "請選擇 job 資料夾"
        else:
            job_dir = self._dataset_resolve_job_dir(current_job)
            if job_dir is None:
                manifest_error = f"指定的 job 不存在：{current_job}"
            else:
                try:
                    manifest = read_manifest(job_dir / "manifest.csv")
                except Exception as e:
                    manifest_error = f"讀 manifest.csv 失敗：{e}"
                    manifest = {}
                for sid, row in manifest.items():
                    if row.get("status") != "ok":
                        continue
                    label = row.get("label", "")
                    prefix = row.get("prefix", "")
                    label_counts[label] = label_counts.get(label, 0) + 1
                    if prefix:
                        prefixes_set.add(prefix)
                    items_all.append(row)

                def _match(r):
                    if current_label and r.get("label") != current_label:
                        return False
                    if current_prefix and r.get("prefix") != current_prefix:
                        return False
                    return True

                items_all = [r for r in items_all if _match(r)]
                items_all.sort(key=lambda r: r.get("collected_at", ""), reverse=True)

        total_count = sum(label_counts.values())
        filtered_count = len(items_all)
        total_pages = max(1, (filtered_count + limit - 1) // limit)
        page = min(page, total_pages)
        start_idx = (page - 1) * limit
        end_idx = min(start_idx + limit, filtered_count)
        page_items = items_all[start_idx:end_idx]

        import urllib.parse as _up
        for it in page_items:
            crop_rel = it.get("crop_path", "")
            hm_rel = it.get("heatmap_path", "")
            q = {"job": current_job, "path": crop_rel}
            it["crop_url"] = "/api/dataset_export/file?" + _up.urlencode(q)
            q_hm = {"job": current_job, "path": hm_rel}
            it["heatmap_url"] = "/api/dataset_export/file?" + _up.urlencode(q_hm)

        def _page_url(p):
            qs = {"page": p, "limit": limit}
            if current_job:
                qs["job"] = current_job
            if current_label:
                qs["label"] = current_label
            if current_prefix:
                qs["prefix"] = current_prefix
            return "/dataset_gallery?" + _up.urlencode(qs)

        has_prev = page > 1
        has_next = page < total_pages
        prev_url = _page_url(page - 1) if has_prev else ""
        next_url = _page_url(page + 1) if has_next else ""

        template = self.jinja_env.get_template("dataset_gallery.html")
        html = template.render(
            request_path="/dataset_gallery",
            base_dir=str(base_dir),
            jobs=jobs,
            current_job=current_job,
            manifest_error=manifest_error,
            total_count=total_count,
            filtered_count=filtered_count,
            label_counts=dict(sorted(label_counts.items())),
            prefixes=sorted(prefixes_set),
            current_label=current_label,
            current_prefix=current_prefix,
            items=page_items,
            page=page,
            limit=limit,
            total_pages=total_pages,
            start_idx=start_idx,
            end_idx=end_idx,
            has_prev=has_prev,
            has_next=has_next,
            prev_url=prev_url,
            next_url=next_url,
            label_zh=LABEL_ZH,
            valid_labels=get_valid_labels(),
        )
        self._send_response(200, html)
```

- [ ] **Step 3: Replace `_handle_dataset_export_file`**

找到 `_handle_dataset_export_file` (L3136)，整段換成：

```python
    def _handle_dataset_export_file(self, query: dict):
        """GET /api/dataset_export/file?job=<job_id>&path=<rel>

        path traversal 防護：resolve 後必須 is_relative_to base_dir/<job>
        """
        def _q(key, default=None):
            v = query.get(key)
            if isinstance(v, list):
                return v[0] if v else default
            return v if v is not None else default

        job_id = _q("job", "") or ""
        rel = _q("path", "") or ""
        if not job_id:
            self._send_error(400, "missing job parameter")
            return
        if not rel:
            self._send_error(400, "missing path parameter")
            return

        job_dir = self._dataset_resolve_job_dir(job_id)
        if job_dir is None:
            self._send_error(404, "invalid job")
            return

        try:
            target = (job_dir / rel).resolve()
        except (OSError, ValueError):
            self._send_404()
            return

        try:
            target.relative_to(job_dir)
        except ValueError:
            self._send_error(403, "path outside job_dir")
            return

        if not target.exists() or not target.is_file():
            self._send_404()
            return

        self._send_binary(str(target))
```

- [ ] **Step 4: Replace `_handle_dataset_sample_delete` and `_handle_dataset_sample_move`**

找到 `_handle_dataset_sample_delete` (L3183) 與 `_handle_dataset_sample_move` (L3206)，整段換成：

```python
    def _handle_dataset_sample_delete(self):
        """POST /api/dataset_export/sample/delete  body: {job, sample_id}"""
        data = self._read_json_body()
        if data is None:
            return
        job_id = (data.get("job") or "").strip()
        sample_id = (data.get("sample_id") or "").strip()
        if not job_id or not sample_id:
            self._send_json({"error": "missing job or sample_id"}, status=400)
            return

        job_dir = self._dataset_resolve_job_dir(job_id)
        if job_dir is None:
            self._send_json({"error": "invalid job"}, status=404)
            return

        manifest_path = job_dir / "manifest.csv"
        state = self._dataset_export_state
        with state["manifest_lock"]:
            manifest = read_manifest(manifest_path)
            ok = delete_sample(job_dir, manifest, sample_id)
            if ok:
                write_manifest(manifest_path, manifest)
        if not ok:
            self._send_json({"error": "sample_id not found"}, status=404)
            return
        self._send_json({"ok": True, "sample_id": sample_id})

    def _handle_dataset_sample_move(self):
        """POST /api/dataset_export/sample/move  body: {job, sample_id, new_label}"""
        data = self._read_json_body()
        if data is None:
            return
        job_id = (data.get("job") or "").strip()
        sample_id = (data.get("sample_id") or "").strip()
        new_label = (data.get("new_label") or "").strip()
        if not job_id or not sample_id or not new_label:
            self._send_json({"error": "missing job, sample_id or new_label"}, status=400)
            return
        if new_label not in get_valid_labels():
            self._send_json({
                "error": "invalid new_label",
                "valid_labels": get_valid_labels(),
            }, status=400)
            return

        job_dir = self._dataset_resolve_job_dir(job_id)
        if job_dir is None:
            self._send_json({"error": "invalid job"}, status=404)
            return

        manifest_path = job_dir / "manifest.csv"
        state = self._dataset_export_state
        with state["manifest_lock"]:
            manifest = read_manifest(manifest_path)
            try:
                updated = relabel_sample(job_dir, manifest, sample_id, new_label)
            except ValueError as e:
                self._send_json({"error": str(e)}, status=400)
                return
            if updated is None:
                self._send_json({"error": "sample_id not found"}, status=404)
                return
            write_manifest(manifest_path, manifest)
        self._send_json({
            "ok": True,
            "sample_id": sample_id,
            "new_label": updated["label"],
            "crop_path": updated["crop_path"],
        })
```

- [ ] **Step 5: Syntax sanity check**

```bash
python -c "import ast; ast.parse(open('capi_web.py', encoding='utf-8').read()); print('syntax ok')"
```
Expected: `syntax ok`.

- [ ] **Step 6: Commit**

```bash
git add capi_web.py
git commit -m "feat(web): dataset gallery & sample APIs operate on per-job snapshot folder"
```

---

## Task 5: Gallery template — job selector + job-aware links

**Files:**
- Modify: `templates/dataset_gallery.html`

- [ ] **Step 1: Read template around the existing filter row**

```bash
python -c "
with open('templates/dataset_gallery.html', encoding='utf-8') as f:
    lines = f.readlines()
for i, line in enumerate(lines, 1):
    if 'current_label' in line or 'current_prefix' in line or '<select' in line:
        print(f'{i}: {line}', end='')
" | head -40
```

- [ ] **Step 2: Add job selector before the label selector**

在 template 中找到 label 下拉那段（`{% for lbl, cnt in label_counts.items() %}` 附近的 `<select>`，約 L88-L92），**在它前面**插入一個新的 job 下拉：

```html
        <label class="text-sm text-slate-400 mr-1">Job 資料夾</label>
        <select name="job" onchange="this.form.submit()"
                class="bg-slate-800 border border-slate-700 text-slate-200 rounded-lg px-3 py-1.5 text-sm">
          {% for j in jobs %}
          <option value="{{ j }}" {% if j == current_job %}selected{% endif %}>{{ j }}</option>
          {% endfor %}
          {% if not jobs %}<option value="">（無 job）</option>{% endif %}
        </select>
```

（jobs 已由後端傳入並排序為新的在前）

- [ ] **Step 3: Ensure the form GET keeps `job` when switching filters**

如果 template 的篩選 form 是 `<form method="get" action="/dataset_gallery">`，確認它有 label/prefix select 時 — 需要在 form 內加一個隱藏欄位避免切 label 時丟掉 job。若 job select 在同一個 form 且 `onchange=this.form.submit()` 會隨表單送出，則無需額外處理。檢查 template 現況決定是否需在 label/prefix form 加：

```html
<input type="hidden" name="job" value="{{ current_job }}">
```

（若 label 與 job 在同一個 form 則不需要；若在兩個不同的 form 則 label form 必須帶 hidden `job`）

- [ ] **Step 4: Update JS move/delete calls to include `job`**

Grep template 內的 fetch/AJAX 呼叫 `/api/dataset_export/sample/delete` 與 `/sample/move`：

```bash
python -c "
with open('templates/dataset_gallery.html', encoding='utf-8') as f:
    content = f.read()
for keyword in ['sample/delete', 'sample/move']:
    idx = 0
    while True:
        j = content.find(keyword, idx)
        if j == -1: break
        print(f'{keyword} at offset {j}: {content[max(0,j-80):j+120]!r}')
        idx = j + 1
"
```

找到對應的 `fetch(...)` / `body: JSON.stringify({...})`，把 body 從 `{sample_id: ...}` / `{sample_id, new_label}` 改成帶 `job: CURRENT_JOB`（JS 端從 Jinja 讀）。

在 template 的 `<script>` 區塊開頭（找到 `const VALID_LABELS = {{ valid_labels|tojson }};` 那一行附近）加：

```javascript
const CURRENT_JOB = {{ current_job|tojson }};
```

然後把 delete / move fetch 的 body 改成類似：

```javascript
body: JSON.stringify({ job: CURRENT_JOB, sample_id: sid })
// 或
body: JSON.stringify({ job: CURRENT_JOB, sample_id: sid, new_label: lbl })
```

- [ ] **Step 5: Manually smoke-test in browser**

```bash
# terminal A
cd /c/Users/rh.syu/Desktop/CAPI01_AD
python capi_server.py --config server_config_local.yaml
```

開 http://localhost:8080/dataset_gallery
- 檢查 job 下拉顯示既有 snapshot（若還沒 run 過 job，瀏覽 base_dir 先手動建一個 `20260414_100000/manifest.csv` 假資料測試）
- 切換 job 後 URL 應帶 `?job=...`，圖片能正常載入
- delete / move 按鈕的 fetch 在 DevTools Network tab 檢查 request body 有 `job` 欄位

- [ ] **Step 6: Commit**

```bash
git add templates/dataset_gallery.html
git commit -m "feat(web): gallery template adds job selector + job-aware sample actions"
```

---

## Task 6: Verify full test suite green

**Files:** none (verification only)

- [ ] **Step 1: Run dataset tests**

```bash
cd /c/Users/rh.syu/Desktop/CAPI01_AD
python -m pytest tests/test_dataset_export.py -v
```
Expected: ALL PASS.

- [ ] **Step 2: Run entire test dir to catch collateral breakage**

```bash
python -m pytest tests/ -v
```
Expected: ALL PASS (或既有 failing 不在本次改動範圍內 — 事先 `git stash pop` 確認 baseline)。

- [ ] **Step 3: Smoke-test `DatasetExporter.run` end-to-end via web**

```bash
# terminal A
python capi_server.py --config server_config_local.yaml
# terminal B
curl -X POST http://localhost:8080/api/dataset_export/start \
     -H 'Content-Type: application/json' \
     -d '{"days":3,"include_true_ng":true,"skip_existing":true}'
```

然後 `ls` 看 `base_dir/`（從 `server_config_local.yaml` 的 `dataset_export.base_dir` 取值），應該有一個 `YYYYMMDD_HHMMSS/manifest.csv`。再 POST 一次 — 第二次 `total=0` 且不新增資料夾。

- [ ] **Step 4: No commit needed; task 6 is verification only**

---

## Self-Review Results

**Spec coverage:**
- ✅ `base_dir/<YYYYMMDD_HHMMSS>/` snapshot output — Task 2
- ✅ Skip samples already in any prior job — Task 2 (`load_known_sample_ids`)
- ✅ Remove stale cleanup — Task 2 (rewrite removes L730-752)
- ✅ Don't touch prior job folders — Task 3 (`test_exporter_run_does_not_touch_prior_job_folder`)
- ✅ Empty-job short circuit — Task 2 (`if not new_rows`)
- ✅ Gallery per-job view — Tasks 4 & 5
- ✅ File / delete / move APIs accept `job` — Task 4
- ✅ Path traversal defense includes `job` validation — Task 4 (`_JOB_ID_RE` + `_dataset_resolve_job_dir`)

**Placeholder scan:** 僅 Task 3 Step 3 的 `_setup_fake_db_single_candidate` helper 在 stub 階段提示「以 `test_exporter_run_end_to_end` 為範例實作」— 這不是 placeholder，是指定實作依據（讀既有測試樣板移植）。

**Type consistency:**
- `list_job_dirs` 回 `List[Path]`；`_dataset_list_jobs` 回 `list[str]`；`_dataset_resolve_job_dir` 回 `Optional[Path]` — 一致
- `run` 簽名參數名（days, include_true_ng, skip_existing, status_callback, cancel_event）與呼叫端 `_dataset_export_worker` 一致

---

Plan complete and saved to `docs/superpowers/plans/2026-04-14-dataset-export-snapshot-folders.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
