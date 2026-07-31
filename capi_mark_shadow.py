"""Client for local PaddleOCR MARK comparison and online recognition."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import queue
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import cv2


logger = logging.getLogger("capi.mark_shadow")

_CLIENT_LOCK = threading.Lock()
_CLIENT: Optional["MarkShadowClient"] = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_mark_shadow_payload(
    image,
    detection: Dict[str, Any],
    source_path: str | Path,
    *,
    padding_ratio: float = 0.15,
) -> Dict[str, Any]:
    """Build the exact upright MARK crop sent to the shadow recognizer."""
    if image is None or not detection.get("found"):
        raise ValueError("MARK shadow requires a successful detection")

    bbox = detection.get("bbox") or {}
    try:
        x = int(bbox["x"])
        y = int(bbox["y"])
        width = int(bbox["width"])
        height = int(bbox["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("MARK shadow bbox is invalid") from exc
    if width <= 0 or height <= 0:
        raise ValueError("MARK shadow bbox must have positive size")

    image_height, image_width = image.shape[:2]
    pad_x = max(4, int(round(width * max(0.0, float(padding_ratio)))))
    pad_y = max(4, int(round(height * max(0.0, float(padding_ratio)))))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image_width, x + width + pad_x)
    y2 = min(image_height, y + height + pad_y)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("MARK shadow crop is empty")

    crop = image[y1:y2, x1:x2].copy()
    if str(detection.get("orientation") or "").lower() == "rot180":
        crop = cv2.rotate(crop, cv2.ROTATE_180)

    encoded, png = cv2.imencode(".png", crop)
    if not encoded:
        raise ValueError("MARK shadow crop could not be encoded")
    png_bytes = png.tobytes()

    return {
        "source_path": str(source_path),
        "source_image": Path(source_path).name,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "current_text": str(detection.get("text") or ""),
        "current_confidence": float(detection.get("confidence") or 0.0),
        "current_profile_version": int(detection.get("profile_version") or 0),
        "current_roi": str(detection.get("roi") or ""),
        "current_orientation": str(detection.get("orientation") or ""),
        "bbox": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        },
        "crop_bbox": {
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
        },
        "crop_width": int(crop.shape[1]),
        "crop_height": int(crop.shape[0]),
        "crop_sha256": hashlib.sha256(png_bytes).hexdigest(),
        "image_png_base64": base64.b64encode(png_bytes).decode("ascii"),
    }


class MarkShadowClient:
    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float = 5.0,
        max_queue: int = 64,
        padding_ratio: float = 0.15,
    ):
        self.endpoint = endpoint
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.padding_ratio = max(0.0, float(padding_ratio))
        self._queue: queue.Queue[Optional[Dict[str, Any]]] = queue.Queue(
            maxsize=max(1, int(max_queue))
        )
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="mark-shadow-client",
            daemon=True,
        )
        self._dropped = 0
        self._thread.start()

    def submit(
        self,
        image,
        detection: Dict[str, Any],
        source_path: str | Path,
    ) -> bool:
        try:
            payload = build_mark_shadow_payload(
                image,
                detection,
                source_path,
                padding_ratio=self.padding_ratio,
            )
            self._queue.put_nowait(payload)
            return True
        except queue.Full:
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 100 == 0:
                logger.warning(
                    "MARK shadow queue full; dropped=%s",
                    self._dropped,
                )
        except Exception as exc:
            logger.warning("MARK shadow submission skipped: %s", exc)
        return False

    def _send(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(
            request,
            timeout=self.timeout_seconds,
        ) as response:
            response_data = json.loads(response.read().decode("utf-8"))
        if not isinstance(response_data, dict):
            raise RuntimeError("PaddleOCR worker returned a non-object response")
        if not response_data.get("success"):
            raise RuntimeError(
                str(response_data.get("error") or "PaddleOCR worker inference failed")
            )
        return response_data

    def recognize(
        self,
        image,
        detection: Dict[str, Any],
        source_path: str | Path,
    ) -> Dict[str, Any]:
        payload = build_mark_shadow_payload(
            image,
            detection,
            source_path,
            padding_ratio=self.padding_ratio,
        )
        return self._send(payload)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if payload is None:
                self._queue.task_done()
                return
            try:
                self._send(payload)
            except Exception as exc:
                logger.warning("MARK shadow worker unavailable: %s", exc)
            finally:
                self._queue.task_done()

    def stop(self, timeout_seconds: float = 1.0) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=max(0.0, float(timeout_seconds)))


def configure_mark_shadow(config: Optional[Dict[str, Any]]) -> bool:
    """Configure the process-global shadow client from server_config.yaml."""
    global _CLIENT
    cfg = config or {}
    enabled = _env_bool(
        "CAPI_MARK_SHADOW_ENABLED",
        bool(cfg.get("enabled", False)),
    )
    endpoint = os.environ.get(
        "CAPI_MARK_SHADOW_ENDPOINT",
        str(cfg.get("endpoint") or "http://127.0.0.1:8765/infer"),
    )
    timeout_ms = int(cfg.get("timeout_ms", 5000) or 5000)
    max_queue = int(cfg.get("max_queue", 64) or 64)
    padding_ratio = float(cfg.get("crop_padding_ratio", 0.15) or 0.15)

    with _CLIENT_LOCK:
        old_client = _CLIENT
        _CLIENT = None
        if old_client is not None:
            old_client.stop()
        if enabled:
            _CLIENT = MarkShadowClient(
                endpoint,
                timeout_seconds=timeout_ms / 1000.0,
                max_queue=max_queue,
                padding_ratio=padding_ratio,
            )
            logger.info(
                "MARK PaddleOCR online recognition enabled: endpoint=%s queue=%s",
                endpoint,
                max_queue,
            )
    return enabled


def submit_mark_shadow(
    image,
    detection: Dict[str, Any],
    source_path: str | Path,
) -> bool:
    with _CLIENT_LOCK:
        client = _CLIENT
    if client is None:
        return False
    return client.submit(image, detection, source_path)


def recognize_mark_online(
    image,
    detection: Dict[str, Any],
    source_path: str | Path,
) -> Dict[str, Any]:
    """Run the configured PaddleOCR worker synchronously for the formal MARK text."""
    started = time.perf_counter()
    with _CLIENT_LOCK:
        client = _CLIENT
    if client is None:
        return {
            "success": False,
            "error": "PaddleOCR client is disabled",
            "round_trip_ms": 0.0,
        }
    try:
        result = client.recognize(image, detection, source_path)
        result.setdefault("technique", "PaddleOCR")
        result.setdefault("engine_version", "3.7.0")
        result.setdefault("worker_version", "1")
        result["round_trip_ms"] = (time.perf_counter() - started) * 1000.0
        return result
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "round_trip_ms": (time.perf_counter() - started) * 1000.0,
        }


def stop_mark_shadow() -> None:
    global _CLIENT
    with _CLIENT_LOCK:
        client = _CLIENT
        _CLIENT = None
    if client is not None:
        client.stop()
