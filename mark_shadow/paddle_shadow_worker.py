#!/usr/bin/env python3
"""Local PaddleOCR worker for MARK comparison and online recognition."""

from __future__ import annotations

import argparse
import base64
from collections import Counter, deque
import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Callable, Dict

import cv2
import numpy as np


VERSION = "4"
TECHNIQUE = "PaddleOCR"
LOGGER = logging.getLogger("mark_shadow.worker")
VALID_MARK = re.compile(r"[A-Z0-9]{2}")
DEFAULT_FORCED_CHAR_CONVERSIONS = (("U", "V"),)


def installed_package_version(package_name: str) -> str:
    try:
        return package_version(package_name)
    except PackageNotFoundError:
        return "unknown"


def normalize_mark_text(value: Any) -> str:
    text = "".join(str(value or "").upper().split())
    return text if VALID_MARK.fullmatch(text) else ""


def normalize_forced_char_conversions(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return DEFAULT_FORCED_CHAR_CONVERSIONS
    if not isinstance(value, (list, tuple)) or len(value) > 100:
        raise ValueError("invalid forced_char_conversions")

    normalized = []
    seen = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("invalid forced_char_conversions rule")
        paddle = str(item.get("paddle") or "").strip().upper()
        dotmatrix = str(item.get("dotmatrix") or "").strip().upper()
        if not re.fullmatch(r"[A-Z0-9]", paddle):
            raise ValueError("invalid Paddle character conversion")
        if not re.fullmatch(r"[A-Z0-9]", dotmatrix):
            raise ValueError("invalid DotMatrix character conversion")
        if paddle == dotmatrix or (paddle, dotmatrix) in seen:
            raise ValueError("duplicate or ineffective character conversion")
        seen.add((paddle, dotmatrix))
        normalized.append((paddle, dotmatrix))
    return tuple(normalized)


def apply_forced_char_conversions(
    paddle_text: Any,
    dotmatrix_text: Any,
    rules: Any = None,
) -> tuple[str, tuple[int, ...], tuple[tuple[str, str], ...]]:
    """Apply configured same-position Paddle/DotMatrix conflict conversions."""
    paddle = normalize_mark_text(paddle_text)
    dotmatrix = normalize_mark_text(dotmatrix_text)
    if not paddle or not dotmatrix:
        return paddle, (), ()

    configured = set(normalize_forced_char_conversions(rules))
    applied = tuple(
        (index, paddle_char, dotmatrix_char)
        for index, (paddle_char, dotmatrix_char) in enumerate(
            zip(paddle, dotmatrix)
        )
        if (paddle_char, dotmatrix_char) in configured
    )
    rescued_positions = tuple(item[0] for item in applied)
    if not rescued_positions:
        return paddle, (), ()

    corrected = list(paddle)
    for index, _paddle_char, dotmatrix_char in applied:
        corrected[index] = dotmatrix_char
    applied_rules = tuple(
        (paddle_char, dotmatrix_char)
        for _, paddle_char, dotmatrix_char in applied
    )
    return "".join(corrected), rescued_positions, applied_rules


def rescue_paddle_u_with_dotmatrix_v(
    paddle_text: Any,
    dotmatrix_text: Any,
) -> tuple[str, tuple[int, ...]]:
    """Backward-compatible helper for the original default U/V rescue."""
    corrected, positions, _rules = apply_forced_char_conversions(
        paddle_text,
        dotmatrix_text,
        [{"paddle": "U", "dotmatrix": "V"}],
    )
    return corrected, positions


def prepare_paddle_image(image: np.ndarray) -> np.ndarray:
    """Convert decoded MARK crops to the 3-channel input PaddleOCR expects."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"unsupported MARK crop shape: {image.shape}")
    return np.ascontiguousarray(image)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[index])


class MarkTemporalStabilizer:
    """Keep a short-lived MARK OCR glitch from changing the formal result.

    The history is keyed by the caller-provided stream key.  It is deliberately
    based on valid PaddleOCR text after configured same-position conflict
    conversions. Other DotMatrix/CV disagreements never enter the vote. A new
    value must be seen three times in a row before it replaces the stable value.
    """

    HISTORY_LIMIT = 100
    INITIAL_MIN_SUPPORT = 3
    INITIAL_MODE_RATIO = 0.60
    SWITCH_CONSECUTIVE = 3

    def __init__(
        self,
        history_loader: Callable[[str, int], list[str]] | None = None,
    ):
        self._history_loader = history_loader
        self._states: Dict[str, Dict[str, Any]] = {}
        self._fresh_context_prefixes: set[str] = set()
        self._lock = threading.Lock()

    @staticmethod
    def _key(stream_key: Any) -> str:
        return str(stream_key or "default").strip() or "default"

    def _new_state(self, key: str) -> Dict[str, Any]:
        history = deque(maxlen=self.HISTORY_LIMIT)
        load_history = not any(
            key.startswith(prefix) for prefix in self._fresh_context_prefixes
        )
        if load_history and self._history_loader is not None:
            try:
                loaded = self._history_loader(key, self.HISTORY_LIMIT) or []
            except Exception as exc:
                LOGGER.warning("MARK temporal history load failed: %s", exc)
                loaded = []
            for value in loaded:
                text = normalize_mark_text(value)
                if text:
                    history.append(text)

        stable_text = ""
        if history:
            mode, count = Counter(history).most_common(1)[0]
            if count >= self.INITIAL_MIN_SUPPORT and (
                count / len(history) >= self.INITIAL_MODE_RATIO
            ):
                stable_text = mode
        return {
            "history": history,
            "stable_text": stable_text,
            "candidate_text": "",
            "candidate_count": 0,
        }

    @staticmethod
    def _snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
        history = state["history"]
        stable_text = str(state.get("stable_text") or "")
        support = history.count(stable_text) if stable_text else 0
        return {
            "temporal_stable_text": stable_text,
            "temporal_history_count": len(history),
            "temporal_stable_support_count": support,
            "temporal_candidate_text": str(state.get("candidate_text") or ""),
            "temporal_candidate_count": int(state.get("candidate_count") or 0),
        }

    def snapshot(self, stream_key: Any) -> Dict[str, Any]:
        key = self._key(stream_key)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = self._new_state(key)
                self._states[key] = state
            return self._snapshot(state)

    def reset(self, stream_key: Any) -> None:
        """Start a new history session without reloading older DB rows."""
        key = self._key(stream_key)
        with self._lock:
            self._states[key] = {
                "history": deque(maxlen=self.HISTORY_LIMIT),
                "stable_text": "",
                "candidate_text": "",
                "candidate_count": 0,
            }

    def reset_all(self) -> None:
        """Reload every stream under the currently configured conversion rules."""
        with self._lock:
            self._states.clear()
            self._fresh_context_prefixes.clear()

    def reset_context(self, stream_key: Any) -> None:
        """Reset every ROI/orientation state for one machine/model session."""
        key = self._key(stream_key)
        parts = key.split("|", 2)
        prefix = f"{parts[0]}|{parts[1]}|" if len(parts) == 3 else key
        with self._lock:
            self._fresh_context_prefixes.add(prefix)
            for state_key in list(self._states):
                if state_key.startswith(prefix):
                    del self._states[state_key]

    def observe(self, stream_key: Any, raw_text: Any) -> Dict[str, Any]:
        key = self._key(stream_key)
        raw = normalize_mark_text(raw_text)
        if not raw:
            result = self.snapshot(key)
            result.update({
                "final_text": "",
                "adoption_reason": "no_valid_paddle",
            })
            return result

        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = self._new_state(key)
                self._states[key] = state
            history = state["history"]
            history.append(raw)
            stable_text = str(state.get("stable_text") or "")
            reason = "warmup"

            if not stable_text:
                mode, count = Counter(history).most_common(1)[0]
                if count >= self.INITIAL_MIN_SUPPORT and (
                    count / len(history) >= self.INITIAL_MODE_RATIO
                ):
                    stable_text = mode
                    state["stable_text"] = stable_text

            if stable_text:
                if raw == stable_text:
                    state["candidate_text"] = ""
                    state["candidate_count"] = 0
                    final_text = stable_text
                    reason = "stable_match"
                else:
                    if state.get("candidate_text") == raw:
                        state["candidate_count"] = int(
                            state.get("candidate_count") or 0
                        ) + 1
                    else:
                        state["candidate_text"] = raw
                        state["candidate_count"] = 1

                    if int(state["candidate_count"]) >= self.SWITCH_CONSECUTIVE:
                        stable_text = raw
                        state["stable_text"] = stable_text
                        state["candidate_text"] = ""
                        state["candidate_count"] = 0
                        final_text = stable_text
                        reason = "temporal_switch"
                    else:
                        final_text = stable_text
                        reason = "temporal_outlier"
            else:
                final_text = raw

            result = self._snapshot(state)
            result.update({
                "final_text": final_text,
                "adoption_reason": reason,
            })
            return result


class ShadowStore:
    def __init__(self, db_path: Path, disagreement_dir: Path):
        self.db_path = db_path
        self.disagreement_dir = disagreement_dir
        self._forced_char_conversions = DEFAULT_FORCED_CHAR_CONVERSIONS
        self._forced_char_conversions_lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.disagreement_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(str(self.db_path), timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS mark_shadow_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    captured_at TEXT DEFAULT '',
                    source_path TEXT DEFAULT '',
                    source_image TEXT DEFAULT '',
                    inference_record_id INTEGER DEFAULT 0,
                    crop_sha256 TEXT NOT NULL,
                    current_text TEXT DEFAULT '',
                    current_confidence REAL DEFAULT 0,
                    current_profile_version INTEGER DEFAULT 0,
                    current_roi TEXT DEFAULT '',
                    current_orientation TEXT DEFAULT '',
                    stream_key TEXT DEFAULT '',
                    paddle_raw_text TEXT DEFAULT '',
                    paddle_text TEXT DEFAULT '',
                    paddle_confidence REAL DEFAULT 0,
                    valid_two_chars INTEGER DEFAULT 0,
                    agreed INTEGER DEFAULT 0,
                    final_text TEXT DEFAULT '',
                    adoption_reason TEXT DEFAULT '',
                    temporal_stable_text TEXT DEFAULT '',
                    temporal_history_count INTEGER DEFAULT 0,
                    temporal_stable_support_count INTEGER DEFAULT 0,
                    latency_ms REAL DEFAULT 0,
                    model_name TEXT NOT NULL,
                    crop_path TEXT DEFAULT '',
                    expected_text TEXT DEFAULT '',
                    error TEXT DEFAULT ''
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(mark_shadow_results)"
                ).fetchall()
            }
            migrations = {
                "inference_record_id": "INTEGER DEFAULT 0",
                "stream_key": "TEXT DEFAULT ''",
                "final_text": "TEXT DEFAULT ''",
                "adoption_reason": "TEXT DEFAULT ''",
                "temporal_stable_text": "TEXT DEFAULT ''",
                "temporal_history_count": "INTEGER DEFAULT 0",
                "temporal_stable_support_count": "INTEGER DEFAULT 0",
            }
            for column, definition in migrations.items():
                if column not in columns:
                    connection.execute(
                        f"ALTER TABLE mark_shadow_results ADD COLUMN {column} {definition}"
                    )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_mark_shadow_created
                ON mark_shadow_results(created_at)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_mark_shadow_agreed
                ON mark_shadow_results(agreed, valid_two_chars)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_mark_shadow_inference_record
                ON mark_shadow_results(inference_record_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_mark_shadow_stream
                ON mark_shadow_results(stream_key, id)
                """
            )

    def recent_paddle_texts(self, stream_key: str, limit: int = 100) -> list[str]:
        """Return effective Paddle/CV texts, oldest first, for one stream."""
        with self._forced_char_conversions_lock:
            rules = tuple(self._forced_char_conversions)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT paddle_text, current_text
                FROM mark_shadow_results
                WHERE stream_key = ? AND valid_two_chars = 1
                ORDER BY id DESC
                LIMIT ?
                """,
                (str(stream_key or ""), max(1, int(limit))),
            ).fetchall()
        rule_payload = [
            {"paddle": paddle, "dotmatrix": dotmatrix}
            for paddle, dotmatrix in rules
        ]
        return [
            apply_forced_char_conversions(
                row["paddle_text"],
                row["current_text"],
                rule_payload,
            )[0]
            for row in reversed(rows)
        ]

    def set_forced_char_conversions(self, value: Any) -> bool:
        normalized = normalize_forced_char_conversions(value)
        with self._forced_char_conversions_lock:
            changed = normalized != self._forced_char_conversions
            self._forced_char_conversions = normalized
        return changed

    def save(
        self,
        request_data: Dict[str, Any],
        result: Dict[str, Any],
        crop_png: bytes,
    ) -> int:
        agreed = bool(result.get("paddle_text")) and (
            str(result.get("paddle_text")) == str(request_data.get("current_text") or "")
        )
        crop_root = (
            self.disagreement_dir
            if not agreed
            else self.disagreement_dir.parent / "crops"
        )
        date_dir = crop_root / datetime.now().strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(request_data.get("source_image") or "W0F"),
        )[:80]
        crop_file = date_dir / (
            f"{safe_name}_{request_data['crop_sha256'][:12]}.png"
        )
        if not crop_file.exists():
            crop_file.write_bytes(crop_png)
        crop_path = str(crop_file)

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO mark_shadow_results (
                    created_at, captured_at, source_path, source_image,
                    crop_sha256, current_text, current_confidence,
                    current_profile_version, current_roi, current_orientation,
                    stream_key, paddle_raw_text, paddle_text, paddle_confidence,
                    valid_two_chars, agreed, final_text, adoption_reason,
                    temporal_stable_text, temporal_history_count,
                    temporal_stable_support_count, latency_ms, model_name,
                    crop_path, error
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?
                )
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    str(request_data.get("captured_at") or ""),
                    str(request_data.get("source_path") or ""),
                    str(request_data.get("source_image") or ""),
                    str(request_data.get("crop_sha256") or ""),
                    str(request_data.get("current_text") or ""),
                    float(request_data.get("current_confidence") or 0.0),
                    int(request_data.get("current_profile_version") or 0),
                    str(request_data.get("current_roi") or ""),
                    str(request_data.get("current_orientation") or ""),
                    str(request_data.get("stream_key") or ""),
                    str(result.get("paddle_raw_text") or ""),
                    str(result.get("paddle_text") or ""),
                    float(result.get("paddle_confidence") or 0.0),
                    int(bool(result.get("paddle_text"))),
                    int(agreed),
                    str(result.get("final_text") or result.get("paddle_text") or ""),
                    str(result.get("adoption_reason") or ""),
                    str(result.get("temporal_stable_text") or ""),
                    int(result.get("temporal_history_count") or 0),
                    int(result.get("temporal_stable_support_count") or 0),
                    float(result.get("latency_ms") or 0.0),
                    str(result.get("model_name") or ""),
                    crop_path,
                    str(result.get("error") or ""),
                ),
            )
            return int(cursor.lastrowid)

    def stats(self) -> Dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT valid_two_chars, agreed, latency_ms, error
                FROM mark_shadow_results
                ORDER BY id
                """
            ).fetchall()
        latencies = [float(row["latency_ms"] or 0.0) for row in rows]
        total = len(rows)
        valid = sum(int(row["valid_two_chars"] or 0) for row in rows)
        agreed = sum(int(row["agreed"] or 0) for row in rows)
        errors = sum(1 for row in rows if str(row["error"] or ""))
        return {
            "total": total,
            "valid_two_chars": valid,
            "no_read": total - valid,
            "agreed": agreed,
            "disagreed": total - agreed,
            "agreement_rate": (agreed / total) if total else 0.0,
            "error_count": errors,
            "latency_ms": {
                "average": (sum(latencies) / total) if total else 0.0,
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
            },
        }


class PaddleRecognizer:
    def __init__(
        self,
        model_dir: Path,
        model_name: str,
        device: str,
        cpu_threads: int,
    ):
        from paddleocr import TextRecognition

        self.model_name = model_name
        self.engine_version = installed_package_version("paddleocr")
        self._lock = threading.Lock()
        self._model = TextRecognition(
            model_name=model_name,
            model_dir=str(model_dir),
            device=device,
            engine="paddle_static",
            enable_hpi=False,
            enable_mkldnn=True,
            cpu_threads=max(1, int(cpu_threads)),
        )

    @staticmethod
    def _result_dict(result: Any) -> Dict[str, Any]:
        data = getattr(result, "json", {})
        if callable(data):
            data = data()
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            return {}
        nested = data.get("res")
        return nested if isinstance(nested, dict) else data

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        started = time.perf_counter()
        model_input = prepare_paddle_image(image)
        with self._lock:
            outputs = self._model.predict(input=model_input, batch_size=1)
            output = next(iter(outputs), None)
        latency_ms = (time.perf_counter() - started) * 1000.0
        data = self._result_dict(output) if output is not None else {}
        raw_text = str(data.get("rec_text") or "")
        return {
            "paddle_raw_text": raw_text,
            "paddle_text": normalize_mark_text(raw_text),
            "paddle_confidence": float(data.get("rec_score") or 0.0),
            "latency_ms": latency_ms,
            "model_name": self.model_name,
            "technique": TECHNIQUE,
            "engine_version": getattr(self, "engine_version", "unknown"),
            "worker_version": VERSION,
            "error": "",
        }


class ShadowApplication:
    def __init__(self, recognizer: PaddleRecognizer, store: ShadowStore):
        self.recognizer = recognizer
        self.store = store
        self.temporal = MarkTemporalStabilizer(store.recent_paddle_texts)
        self._context_lock = threading.Lock()
        self._active_model_by_machine: Dict[str, str] = {}

    def _reset_for_model_context(
        self,
        request_data: Dict[str, Any],
        stream_key: str,
    ) -> str:
        machine_no = str(request_data.get("machine_no") or "").strip().casefold()
        model_id = str(request_data.get("model_id") or "").strip().casefold()
        if not machine_no or not model_id:
            return ""

        with self._context_lock:
            previous_model = self._active_model_by_machine.get(machine_no)
            if previous_model == model_id:
                return ""
            self._active_model_by_machine[machine_no] = model_id

        self.temporal.reset_context(stream_key)
        return "context_start" if previous_model is None else "model_switch_reset"

    def infer(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        png_base64 = str(request_data.get("image_png_base64") or "")
        if not png_base64:
            raise ValueError("image_png_base64 is required")
        crop_png = base64.b64decode(png_base64, validate=True)
        if not crop_png or len(crop_png) > 8 * 1024 * 1024:
            raise ValueError("MARK crop size is invalid")
        crop_sha256 = hashlib.sha256(crop_png).hexdigest()
        expected_sha256 = str(request_data.get("crop_sha256") or "")
        if expected_sha256 and crop_sha256 != expected_sha256:
            raise ValueError("MARK crop checksum mismatch")
        request_data["crop_sha256"] = crop_sha256

        image = cv2.imdecode(
            np.frombuffer(crop_png, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        if image is None:
            raise ValueError("MARK crop cannot be decoded")

        try:
            result = self.recognizer.predict(image)
        except Exception as exc:
            result = {
                "paddle_raw_text": "",
                "paddle_text": "",
                "paddle_confidence": 0.0,
                "latency_ms": 0.0,
                "model_name": self.recognizer.model_name,
                "technique": TECHNIQUE,
                "engine_version": getattr(
                    self.recognizer,
                    "engine_version",
                    "unknown",
                ),
                "worker_version": VERSION,
                "error": str(exc),
            }
        result.setdefault("technique", TECHNIQUE)
        result.setdefault(
            "engine_version",
            getattr(self.recognizer, "engine_version", "unknown"),
        )
        result.setdefault("worker_version", VERSION)
        raw_text = normalize_mark_text(result.get("paddle_text"))
        result["paddle_text"] = raw_text
        requested_rules = (
            request_data.get("forced_char_conversions")
            if "forced_char_conversions" in request_data
            else None
        )
        stream_key = str(request_data.get("stream_key") or "default").strip() or "default"
        request_data["stream_key"] = stream_key
        rules_changed = self.store.set_forced_char_conversions(requested_rules)
        if rules_changed:
            self.temporal.reset_all()
        recognition_text, rescued_positions, applied_rules = (
            apply_forced_char_conversions(
                raw_text,
                request_data.get("current_text"),
                requested_rules,
            )
        )
        context_reason = self._reset_for_model_context(request_data, stream_key)
        if raw_text:
            result.update(self.temporal.observe(stream_key, recognition_text))
            if context_reason:
                result["adoption_reason"] = context_reason
            if rescued_positions:
                positions = ",".join(str(index + 1) for index in rescued_positions)
                rule_names = ",".join(
                    f"{paddle}>{dotmatrix}"
                    for paddle, dotmatrix in dict.fromkeys(applied_rules)
                )
                result["adoption_reason"] = (
                    f"forced_char_conversion[pos={positions};rules={rule_names}];"
                    f"{result.get('adoption_reason') or 'direct'}"
                )
        else:
            result.update(self.temporal.observe(stream_key, ""))
        result["id"] = self.store.save(request_data, result, crop_png)
        result["agreed"] = bool(result.get("paddle_text")) and (
            result["paddle_text"] == str(request_data.get("current_text") or "")
        )
        return result


def make_handler(application: ShadowApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CAPI-Mark-Shadow/" + VERSION

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "version": VERSION,
                        "technique": TECHNIQUE,
                        "engine_version": getattr(
                            application.recognizer,
                            "engine_version",
                            "unknown",
                        ),
                        "model_name": application.recognizer.model_name,
                    },
                )
                return
            if self.path == "/stats":
                self._send_json(200, application.store.stats())
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/infer":
                self._send_json(404, {"error": "not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 12 * 1024 * 1024:
                    raise ValueError("request size is invalid")
                request_data = json.loads(
                    self.rfile.read(content_length).decode("utf-8")
                )
                if not isinstance(request_data, dict):
                    raise ValueError("JSON object is required")
                result = application.infer(request_data)
                self._send_json(200, {"success": True, **result})
            except Exception as exc:
                LOGGER.warning("Shadow inference rejected: %s", exc)
                self._send_json(400, {"success": False, "error": str(exc)})

        def log_message(self, format_string, *args):
            LOGGER.info("%s - %s", self.address_string(), format_string % args)

    return Handler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", default="PP-OCRv6_medium_rec")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--db",
        default="/aidata/capi_ai/mark_shadow/mark_shadow.db",
    )
    parser.add_argument(
        "--disagreement-dir",
        default="/aidata/capi_ai/mark_shadow/disagreements",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {model_dir}")
    recognizer = PaddleRecognizer(
        model_dir,
        args.model_name,
        args.device,
        args.cpu_threads,
    )
    store = ShadowStore(
        Path(args.db).resolve(),
        Path(args.disagreement_dir).resolve(),
    )
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(ShadowApplication(recognizer, store)),
    )
    LOGGER.info(
        "MARK shadow listening on http://%s:%s model=%s device=%s",
        args.host,
        args.port,
        args.model_name,
        args.device,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
