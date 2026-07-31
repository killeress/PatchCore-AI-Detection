#!/usr/bin/env python3
"""Local PaddleOCR worker for MARK comparison and online recognition."""

from __future__ import annotations

import argparse
import base64
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
from typing import Any, Dict

import cv2
import numpy as np


VERSION = "2"
TECHNIQUE = "PaddleOCR"
LOGGER = logging.getLogger("mark_shadow.worker")
VALID_MARK = re.compile(r"[A-Z0-9]{2}")


def installed_package_version(package_name: str) -> str:
    try:
        return package_version(package_name)
    except PackageNotFoundError:
        return "unknown"


def normalize_mark_text(value: Any) -> str:
    text = "".join(str(value or "").upper().split())
    return text if VALID_MARK.fullmatch(text) else ""


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


class ShadowStore:
    def __init__(self, db_path: Path, disagreement_dir: Path):
        self.db_path = db_path
        self.disagreement_dir = disagreement_dir
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
                    crop_sha256 TEXT NOT NULL,
                    current_text TEXT DEFAULT '',
                    current_confidence REAL DEFAULT 0,
                    current_profile_version INTEGER DEFAULT 0,
                    current_roi TEXT DEFAULT '',
                    current_orientation TEXT DEFAULT '',
                    paddle_raw_text TEXT DEFAULT '',
                    paddle_text TEXT DEFAULT '',
                    paddle_confidence REAL DEFAULT 0,
                    valid_two_chars INTEGER DEFAULT 0,
                    agreed INTEGER DEFAULT 0,
                    latency_ms REAL DEFAULT 0,
                    model_name TEXT NOT NULL,
                    crop_path TEXT DEFAULT '',
                    expected_text TEXT DEFAULT '',
                    error TEXT DEFAULT ''
                )
                """
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
                    paddle_raw_text, paddle_text, paddle_confidence,
                    valid_two_chars, agreed, latency_ms, model_name,
                    crop_path, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    str(result.get("paddle_raw_text") or ""),
                    str(result.get("paddle_text") or ""),
                    float(result.get("paddle_confidence") or 0.0),
                    int(bool(result.get("paddle_text"))),
                    int(agreed),
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
