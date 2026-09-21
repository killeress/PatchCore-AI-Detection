"""CUDA tracing enabled by default. Never synchronize, clear cache, or reset peak counters."""
import contextvars
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import inspect
import json
import logging
import os
import threading
import time
import uuid

logger = logging.getLogger("capi.inference")
_context = contextvars.ContextVar("cuda_trace_context", default={})


class FlightRecorder:
    """Bounded host-memory history shared by the service's GPU work.

    Flush immediately on a trigger, then stream a bounded number of subsequent
    samples. A sustained high reading does not repeatedly dump the history.
    """
    def __init__(self, max_records=256, max_bytes=1024 * 1024,
                 growth_mib=512, high_ratio=0.85, post_records=64, post_seconds=30):
        self.history = deque()
        self.history_bytes = 0
        self.max_records = max_records
        self.max_bytes = max_bytes
        self.growth_mib = growth_mib
        self.high_ratio = high_ratio
        self.post_records = post_records
        self.post_seconds = post_seconds
        self.remaining = 0
        self.deadline = 0
        self.anchor = None
        self.was_high = False
        self.retries = 0
        self.ooms = 0
        self.sequence = 0
        self.lock = threading.Lock()

    def accept(self, payload):
        with self.lock:
            self.sequence += 1
            payload = dict(payload, sequence=self.sequence)
            encoded = json.dumps(payload, ensure_ascii=False)
            size = len(encoded.encode("utf-8"))
            reasons = []
            reserved = payload.get("reserved_mib")
            if reserved is not None:
                if self.anchor is None:
                    self.anchor = reserved
                if reserved - self.anchor >= self.growth_mib:
                    reasons.append("reserved_growth")
                    self.anchor = reserved
                else:
                    self.anchor = min(self.anchor, reserved)
            used = payload.get("device_used_mib", 0)
            total = used + payload.get("device_free_mib", 0)
            high = total > 0 and used / total >= self.high_ratio
            if high and not self.was_high:
                reasons.append("device_usage_high")
            # Hysteresis avoids repeated triggers around the 85% boundary.
            if high:
                self.was_high = True
            elif total > 0 and used / total < self.high_ratio - 0.05:
                self.was_high = False
            for field, attr in (("allocation_retries_total", "retries"), ("oom_total", "ooms")):
                value = payload.get(field, getattr(self, attr))
                if value > getattr(self, attr):
                    reasons.append(field)
                setattr(self, attr, value)
            if payload.get("outcome") == "error" or "snapshot_error" in payload:
                reasons.append("stage_error")

            now = time.monotonic()
            active = self.remaining > 0 and now <= self.deadline
            if active:
                logger.info("[CUDA-TRACE] %s", encoded)
                self.remaining -= 1
            elif reasons:
                logger.warning("[CUDA-TRIGGER] %s", json.dumps({
                    "reasons": reasons, "sequence": self.sequence,
                    "request": payload.get("request"), "glass": payload.get("glass"),
                    "history_records": len(self.history), "history_bytes": self.history_bytes,
                }, ensure_ascii=False))
                for old, _ in self.history:
                    logger.info("[CUDA-TRACE] %s", old)
                logger.info("[CUDA-TRACE] %s", encoded)
                self.history.clear()
                self.history_bytes = 0
                self.remaining = self.post_records
                self.deadline = now + self.post_seconds
            else:
                # Never retain tensors, images, or arbitrary object references.
                if size <= self.max_bytes:
                    self.history.append((encoded, size))
                    self.history_bytes += size
                while len(self.history) > self.max_records or self.history_bytes > self.max_bytes:
                    _, removed = self.history.popleft()
                    self.history_bytes -= removed
            if payload.get("stage") == "panel" and payload.get("phase") == "after":
                logger.info("[CUDA-SUMMARY] %s", encoded)


_recorder = FlightRecorder()


def enabled():
    return os.environ.get("CAPI_CUDA_MEMORY_TRACE", "1").lower() in {"1", "true", "yes"}


def _snapshot():
    import torch
    if not torch.cuda.is_initialized():
        return None
    stats = torch.cuda.memory_stats()
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "inactive_split_mib": stats.get("inactive_split_bytes.all.current", 0) / 2**20,
        "allocation_retries_total": stats.get("num_alloc_retries", 0),
        "oom_total": stats.get("num_ooms", 0),
        "device_used_mib": (total - free) / 2**20,
        "device_free_mib": free / 2**20,
    }


def _record(phase, stage, span, before=None, outcome=None):
    try:
        values = _snapshot()
        if values is None:
            return None
        details = {key: value[:512] if isinstance(value, str) else value
                   for key, value in _context.get().items()}
        payload = dict(details, **values, phase=phase, stage=stage,
                       span=span, pid=os.getpid(),
                       sampled_at=datetime.now(timezone.utc).isoformat())
        if outcome:
            payload["outcome"] = outcome
        if before is not None:
            for key in ("allocated_mib", "reserved_mib", "inactive_split_mib",
                        "allocation_retries_total", "oom_total"):
                payload["delta_" + key] = values[key] - before[key]
        _recorder.accept(payload)
        return values
    except Exception as exc:
        logger.warning("[CUDA-TRACE] %s %s unavailable: %s", stage, phase, exc)
        try:
            _recorder.accept(dict(_context.get(), stage=stage, phase=phase, span=span,
                                  pid=os.getpid(), outcome="error", snapshot_error=str(exc)[:512],
                                  sampled_at=datetime.now(timezone.utc).isoformat()))
        except Exception:
            pass  # Diagnostics must not replace the inference exception.
        return None


@contextmanager
def cuda_stage(stage, **details):
    if not enabled():
        yield
        return
    token = _context.set(dict(_context.get(), **details))
    span = uuid.uuid4().hex[:12]
    before = _record("before", stage, span)
    outcome = "error"
    try:
        yield
        outcome = "ok"
    finally:
        try:
            _record("after", stage, span, before, outcome)
        finally:
            _context.reset(token)


def trace_call(stage):
    """Trace complete calls so post-call snapshots include local tensor release."""
    def decorate(func):
        signature = inspect.signature(func)

        @wraps(func)
        def wrapped(*args, **kwargs):
            if not enabled():
                return func(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            values = bound.arguments
            if stage == "patchcore-tile" and values.get("raw_prediction") is not None:
                return func(*args, **kwargs)
            details = {}
            try:
                if stage == "panel":
                    details = {"glass": str(values.get("glass_id") or values.get("panel_dir")),
                               "machine": values.get("machine_no"),
                               "model_id": values.get("model_id"),
                               "request": uuid.uuid4().hex[:12]}
                result = values.get("result") or values.get("image_result")
                if result is not None:
                    details.update(image=str(result.image_path),
                                   screen=getattr(result, "image_prefix", ""))
                if values.get("image_path") is not None:
                    details["image"] = str(values["image_path"])
                tile = values.get("tile")
                if tile is not None:
                    details.update(tile=getattr(tile, "tile_id", None),
                                   zone=getattr(tile, "zone", ""),
                                   input_shape=list(tile.image.shape), batch=1)
                model = values.get("inferencer") or values.get("model")
                if model is None and stage == "patchcore-tile":
                    model = getattr(values.get("self"), "inferencer", None)
                if model is not None:
                    details["model"] = str(getattr(model, "_cuda_trace_model_path", type(model).__name__))
                image = values.get("tile_img")
                if image is not None:
                    details.update(input_shape=list(image.shape), batch=1)
            except Exception as exc:
                logger.warning("[CUDA-TRACE] metadata unavailable: %s", exc)
            with cuda_stage(stage, **details):
                return func(*args, **kwargs)
        return wrapped
    return decorate
