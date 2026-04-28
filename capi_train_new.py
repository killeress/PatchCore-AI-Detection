"""新機種 PatchCore 訓練 Wizard 後端 worker。

提供：
- preprocess_panels_to_pool: Step 2 切 tile + 寫 DB
- sample_ng_tiles: 從 over_review 抽 NG
- run_training_pipeline: Step 4 訓 10 模型 + 寫 bundle
"""
from __future__ import annotations
import os
import json
import logging
import platform
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable
import numpy as np
import cv2

logger = logging.getLogger("capi.train_new")

LIGHTINGS = ("G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD")
ZONES = ("inner", "edge")
TRAINING_UNITS = [(l, z) for l in LIGHTINGS for z in ZONES]  # 10 個

MIN_TRAIN_TILES = 30
NG_TILES_PER_LIGHTING = 30


@dataclass
class TrainingConfig:
    machine_id: str
    panel_paths: List[Path]
    over_review_root: Path
    output_root: Path = Path("model")
    backbone_cache_dir: Path = Path("deployment/torch_hub_cache")

    batch_size: int = 8
    image_size: tuple = (512, 512)
    coreset_ratio: float = 0.1
    max_epochs: int = 1


def generate_job_id(machine_id: str) -> str:
    return f"train_{machine_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def preprocess_panels_to_pool(*args, **kwargs):
    raise NotImplementedError("Phase 4.2")


def sample_ng_tiles(*args, **kwargs):
    raise NotImplementedError("Phase 4.3")


def run_training_pipeline(*args, **kwargs):
    raise NotImplementedError("Phase 4.4-4.6")
