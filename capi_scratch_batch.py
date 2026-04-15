"""批次 scratch 分類器驗證：掃整個 dataset export job，
對每筆樣本跑 ScratchClassifier.predict，回報混淆矩陣與分數分佈。

陽性 (should flip) = over_surface_scratch
陰性 (should keep NG) = 其他全部 (true_ng + 其他 over_*)
"""
from __future__ import annotations

import csv
import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

POSITIVE_LABEL = "over_surface_scratch"
MAX_SAMPLES = 2000

STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_CANCELLED = "cancelled"
STATE_FAILED = "failed"


@dataclass
class SampleResult:
    sample_id: str
    label: str
    is_positive: bool
    score: float
    crop_path: str
    glass_id: str
    image_name: str
    over_review_category: str


@dataclass
class BatchTask:
    task_id: str
    job_id: str
    state: str
    done: int = 0
    total: int = 0
    skipped: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    effective_threshold: float = 0.0
    conformal_threshold: float = 0.0
    safety_multiplier: float = 0.0
    error: str = ""
    results: list = field(default_factory=list)

    def to_status_dict(self) -> dict:
        now = time.time()
        end = self.finished_at if self.finished_at else now
        elapsed = max(0.0, end - self.started_at) if self.started_at else 0.0
        return {
            "task_id": self.task_id,
            "job_id": self.job_id,
            "state": self.state,
            "done": self.done,
            "total": self.total,
            "skipped": self.skipped,
            "effective_threshold": self.effective_threshold,
            "conformal_threshold": self.conformal_threshold,
            "safety_multiplier": self.safety_multiplier,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "elapsed": elapsed,
        }


class ScratchBatchRunner:
    """單任務背景執行器，同一時間只允許一個 task 執行。結果同時快取到
    cache_dir/<task_id>.json，server 重啟後可由 get() 回存。"""

    def __init__(self, inferencer, gpu_lock, cache_dir):
        self.inferencer = inferencer
        self.gpu_lock = gpu_lock
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._current: Optional[BatchTask] = None
        self._cancel_event = threading.Event()
        self._tasks: dict[str, BatchTask] = {}

    def start(self, job_id: str, job_dir: Path) -> BatchTask:
        with self._lock:
            if self._current is not None and self._current.state == STATE_RUNNING:
                raise RuntimeError("已有批次任務執行中，請等待或取消")
            task_id = time.strftime("batch_%Y%m%d_%H%M%S")
            task = BatchTask(
                task_id=task_id,
                job_id=job_id,
                state=STATE_RUNNING,
                started_at=time.time(),
            )
            self._current = task
            self._cancel_event.clear()
            self._tasks[task_id] = task

        t = threading.Thread(
            target=self._run,
            args=(task, job_dir),
            name=f"scratch-batch-{task_id}",
            daemon=True,
        )
        t.start()
        return task

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            if (self._current
                    and self._current.task_id == task_id
                    and self._current.state == STATE_RUNNING):
                self._cancel_event.set()
                return True
        return False

    def current(self) -> Optional[BatchTask]:
        return self._current

    def get(self, task_id: str) -> Optional[BatchTask]:
        if task_id in self._tasks:
            return self._tasks[task_id]
        cache = self.cache_dir / f"{task_id}.json"
        if cache.exists():
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                task = self._restore(data)
                self._tasks[task_id] = task
                return task
            except Exception as e:
                logger.warning("讀取快取任務 %s 失敗：%s", task_id, e)
        return None

    def list_recent(self, limit: int = 10) -> list:
        """列出最近的任務（記憶體 + 快取檔），依 started_at 反序。"""
        merged: dict[str, BatchTask] = dict(self._tasks)
        try:
            for p in self.cache_dir.glob("batch_*.json"):
                tid = p.stem
                if tid in merged:
                    continue
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    merged[tid] = self._restore(data)
                except Exception:
                    pass
        except Exception:
            pass
        items = list(merged.values())
        items.sort(key=lambda t: t.started_at, reverse=True)
        return items[:limit]

    def _restore(self, data: dict) -> BatchTask:
        task = BatchTask(
            task_id=data["task_id"],
            job_id=data.get("job_id", ""),
            state=data.get("state", STATE_DONE),
            done=int(data.get("done", 0)),
            total=int(data.get("total", 0)),
            skipped=int(data.get("skipped", 0)),
            started_at=float(data.get("started_at", 0) or 0),
            finished_at=float(data.get("finished_at", 0) or 0),
            effective_threshold=float(data.get("effective_threshold", 0) or 0),
            conformal_threshold=float(data.get("conformal_threshold", 0) or 0),
            safety_multiplier=float(data.get("safety_multiplier", 0) or 0),
            error=data.get("error", "") or "",
        )
        task.results = [SampleResult(**r) for r in data.get("results", [])]
        return task

    def _persist(self, task: BatchTask) -> None:
        try:
            payload = {
                **task.to_status_dict(),
                "results": [asdict(r) for r in task.results],
            }
            target = self.cache_dir / f"{task.task_id}.json"
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(target)
        except Exception as e:
            logger.warning("寫入快取任務 %s 失敗：%s", task.task_id, e)

    def _run(self, task: BatchTask, job_dir: Path) -> None:
        try:
            manifest_path = job_dir / "manifest.csv"
            if not manifest_path.exists():
                raise FileNotFoundError(f"找不到 manifest：{manifest_path}")

            rows: list[dict] = []
            with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("status") != "ok":
                        continue
                    rows.append(row)

            truncated = False
            if len(rows) > MAX_SAMPLES:
                rows = rows[:MAX_SAMPLES]
                truncated = True
            task.total = len(rows)

            with self.gpu_lock:
                sf = self.inferencer._get_scratch_filter()
            if sf is None:
                raise RuntimeError(
                    "ScratchClassifier 未能載入（請確認 scratch_classifier_enabled=true 且 bundle / DINOv2 權重檔存在）"
                )

            task.effective_threshold = float(sf.effective_threshold)
            task.conformal_threshold = float(sf._classifier.conformal_threshold)
            task.safety_multiplier = float(sf._safety)
            if truncated:
                task.error = f"樣本過多，僅取前 {MAX_SAMPLES} 筆"

            for row in rows:
                if self._cancel_event.is_set():
                    task.state = STATE_CANCELLED
                    break

                sid = row.get("sample_id", "")
                label = row.get("label", "")
                crop_rel = row.get("crop_path", "") or ""

                if not crop_rel:
                    task.skipped += 1
                    task.done += 1
                    continue

                crop_abs = job_dir / crop_rel
                if not crop_abs.exists():
                    task.skipped += 1
                    task.done += 1
                    continue

                try:
                    img = Image.open(crop_abs).convert("RGB")
                except Exception as e:
                    logger.warning("開啟 %s 失敗：%s", crop_abs, e)
                    task.skipped += 1
                    task.done += 1
                    continue

                try:
                    with self.gpu_lock:
                        score = float(sf._classifier.predict(img))
                except Exception as e:
                    logger.warning("predict %s 失敗：%s", sid, e)
                    task.skipped += 1
                    task.done += 1
                    continue

                task.results.append(SampleResult(
                    sample_id=sid,
                    label=label,
                    is_positive=(label == POSITIVE_LABEL),
                    score=score,
                    crop_path=crop_rel,
                    glass_id=row.get("glass_id", ""),
                    image_name=row.get("image_name", ""),
                    over_review_category=row.get("over_review_category", ""),
                ))
                task.done += 1

                if task.done % 50 == 0:
                    self._persist(task)

            if task.state == STATE_RUNNING:
                task.state = STATE_DONE
        except Exception as e:
            logger.error("批次分類器任務失敗：%s", e, exc_info=True)
            task.state = STATE_FAILED
            task.error = str(e)
        finally:
            task.finished_at = time.time()
            self._persist(task)


def compute_summary(results: list, threshold: float) -> dict:
    """給定已算好分數的 results 與一個 threshold，重算混淆矩陣與指標。
    陽性 flip 條件：score > threshold。"""
    tp = fp = tn = fn = 0
    pos_scores = []
    neg_scores = []
    for r in results:
        flipped = r.score > threshold
        if r.is_positive:
            pos_scores.append(r.score)
            if flipped:
                tp += 1
            else:
                fn += 1
        else:
            neg_scores.append(r.score)
            if flipped:
                fp += 1
            else:
                tn += 1
    p = tp + fn
    n = fp + tn
    recall = (tp / p) if p else 0.0
    leak_rate = (fp / n) if n else 0.0
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "positive_total": p,
        "negative_total": n,
        "recall": recall,
        "leak_rate": leak_rate,
        "pos_score_count": len(pos_scores),
        "neg_score_count": len(neg_scores),
    }
