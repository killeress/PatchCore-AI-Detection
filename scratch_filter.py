"""ScratchFilter — glue between ScratchClassifier and CAPIInferencer pipeline.

Takes a trained ScratchClassifier and a per-image anomaly_tiles list, flips
high-confidence scratch tiles from NG to OK (removes them from anomaly_tiles)
and records audit fields on each TileInfo.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from capi_inference import ImageResult

logger = logging.getLogger(__name__)

_THR_CLAMP = 0.9999


class ScratchFilter:
    def __init__(self, classifier, safety_multiplier: float = 1.1):
        self._classifier = classifier
        self._safety = float(safety_multiplier)
        raw = classifier.conformal_threshold * self._safety
        self.effective_threshold = min(raw, _THR_CLAMP)

    def apply_to_image_result(self, image_result: ImageResult) -> None:
        if not image_result.anomaly_tiles:
            return
        img_name = getattr(image_result, "image_name", "?")
        n = len(image_result.anomaly_tiles)
        filtered = 0
        infer_sum_ms = 0.0
        t_image_start = time.perf_counter()
        for entry in image_result.anomaly_tiles:
            tile = entry[0]
            t0 = time.perf_counter()
            try:
                score = float(self._classifier.predict(tile.image))
            except Exception as e:
                logger.warning("[scratch] %s tile#%s classifier failed: %s",
                               img_name, getattr(tile, "tile_id", "?"), e)
                continue
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            infer_sum_ms += elapsed_ms
            tile.scratch_score = score
            tile.scratch_infer_ms = elapsed_ms
            if score > self.effective_threshold:
                tile.scratch_filtered = True
                filtered += 1
                logger.debug("[scratch] %s tile#%s FLIP NG→OK score=%.6f (%.1fms)",
                             img_name, getattr(tile, "tile_id", "?"), score, elapsed_ms)
            else:
                logger.debug("[scratch] %s tile#%s KEEP NG   score=%.6f (%.1fms)",
                             img_name, getattr(tile, "tile_id", "?"), score, elapsed_ms)
        image_result.scratch_filter_count = filtered
        total_image_ms = (time.perf_counter() - t_image_start) * 1000.0
        image_result.scratch_elapsed_ms = total_image_ms
        logger.info(
            "[scratch] %s: filtered=%d/%d | image=%.1fms infer_sum=%.1fms avg/tile=%.1fms (thr=%.6f)",
            img_name, filtered, n, total_image_ms, infer_sum_ms,
            infer_sum_ms / n if n else 0.0, self.effective_threshold,
        )
