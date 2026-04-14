"""ScratchFilter — glue between ScratchClassifier and CAPIInferencer pipeline.

Takes a trained ScratchClassifier and a per-image anomaly_tiles list, flips
high-confidence scratch tiles from NG to OK (removes them from anomaly_tiles)
and records audit fields on each TileInfo.
"""
from __future__ import annotations

import logging

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
        keep: list = []
        filtered = 0
        for entry in image_result.anomaly_tiles:
            tile = entry[0]
            try:
                score = float(self._classifier.predict(tile.image))
            except Exception as e:
                logger.warning("Scratch classifier failed on tile %s: %s",
                               getattr(tile, "tile_id", "?"), e)
                # Safety default: keep NG, score stays 0
                keep.append(entry)
                continue
            tile.scratch_score = score
            if score > self.effective_threshold:
                tile.scratch_filtered = True
                filtered += 1
            else:
                keep.append(entry)
        image_result.anomaly_tiles = keep
        image_result.scratch_filter_count = filtered
