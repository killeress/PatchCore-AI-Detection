"""新機種 PatchCore 訓練 Wizard 後端 worker。

提供：
- preprocess_panels_to_pool: Step 2 切 tile + 寫 DB
- sample_ng_tiles: 從推論紀錄抽 Client AOI 炸彈 crop
- run_training_pipeline: Step 4 訓 10 模型 + 寫 bundle
"""
from __future__ import annotations
import bisect
import gc
import os
import json
import logging
import random
import re
import shutil
import time
import traceback
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Callable, Protocol, runtime_checkable, Iterable, Any
import cv2

from capi_image_naming import canonical_image_prefix
from capi_image_orientation import read_detection_image
from capi_preprocess import (
    PreprocessConfig, preprocess_panel_folder, PanelPreprocessResult,
    classify_anchor_zone, detect_panel_polygon,
    image_preprocess_pipeline_for_zone, map_product_coord_to_image,
    rect_polygon_from_bounds, resolve_aoi_inward_shift_axes,
    resolve_inward_polygon_tile,
)

logger = logging.getLogger("capi.train_new")


@runtime_checkable
class TrainingDB(Protocol):
    """Database interface required by train worker."""
    def insert_tile_pool(self, job_id: str, tiles: List[dict]) -> List[int]: ...
    def list_tile_pool(self, job_id: str, **filters) -> List[dict]: ...
    def list_training_bomb_candidates(
        self, machine_id: str, lightings: Optional[Tuple[str, ...]] = None,
    ) -> List[dict]: ...
    def list_training_bomb_validation_samples(
        self, *, machine_id: str, lightings: Optional[Tuple[str, ...]] = None,
    ) -> List[dict]: ...
    def save_training_bomb_validation_samples(self, samples: List[dict]) -> int: ...

LIGHTINGS = ("G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD")
ZONE_INNER = "inner"
ZONE_EDGE = "edge"
ZONES = (ZONE_INNER, ZONE_EDGE)
TRAINING_UNITS = [(l, z) for l in LIGHTINGS for z in ZONES]  # 10 個

MIN_TRAIN_TILES = 30
NG_TILES_PER_LIGHTING = 100
TRAIN_ZERO_SCORE_EPSILON = 1e-6

PANEL_MODE_FULL = "full"
PANEL_MODE_INNER_ONLY = "inner_only"
PANEL_MODE_EDGE_ONLY = "edge_only"
PANEL_MODE_CORNERS_ONLY = "corners_only"
PANEL_MODE_CHOICES = (
    PANEL_MODE_FULL,
    PANEL_MODE_INNER_ONLY,
    PANEL_MODE_EDGE_ONLY,
    PANEL_MODE_CORNERS_ONLY,
)
DEFAULT_TRAIN_TILE_STRIDE = 256
LEGACY_TRAIN_TILE_STRIDE = 512
PATCHCORE_FEATURE_LAYERS_DEFAULT = "layer2_layer3"
PATCHCORE_FEATURE_LAYERS_LAYER3 = "layer3"
PATCHCORE_FEATURE_LAYERS_CHOICES = (
    PATCHCORE_FEATURE_LAYERS_DEFAULT,
    PATCHCORE_FEATURE_LAYERS_LAYER3,
)
PATCHCORE_FEATURE_LAYER_MAP = {
    PATCHCORE_FEATURE_LAYERS_DEFAULT: ("layer2", "layer3"),
    PATCHCORE_FEATURE_LAYERS_LAYER3: ("layer3",),
}
PATCHCORE_FEATURE_POOL_KERNEL_DEFAULT = 3
PATCHCORE_FEATURE_POOL_KERNEL_CHOICES = (1, 3, 5)
FEATURE_CLEANING_MODE_OFF = "off"
# Stable recipe id for existing jobs/bundles; the actual keep ratio is stored separately.
FEATURE_CLEANING_MODE_KNN_Q99 = "knn_cosine_q99_v1"
FEATURE_CLEANING_MODE_CONTEXT_OVERLAP_ADAPTIVE = "context_overlap_adaptive_v1"
FEATURE_CLEANING_MODE_CHOICES = (
    FEATURE_CLEANING_MODE_OFF,
    FEATURE_CLEANING_MODE_KNN_Q99,
    FEATURE_CLEANING_MODE_CONTEXT_OVERLAP_ADAPTIVE,
)
FEATURE_CLEANING_K = 30
FEATURE_CLEANING_K_MIN = 1
FEATURE_CLEANING_K_MAX = 200
FEATURE_CLEANING_KEEP_RATIO_DEFAULT = 0.99
FEATURE_CLEANING_KEEP_RATIO_MIN = 0.90
FEATURE_CLEANING_KEEP_RATIO_MAX = 1.00
FEATURE_CLEANING_CENTER_SIZE_DEFAULT = 384
FEATURE_CLEANING_CENTER_SIZE_MIN = 64
FEATURE_CLEANING_CENTER_SIZE_MAX = 512
FEATURE_CLEANING_SEED = 42
FEATURE_CLEANING_REFERENCE_SIZE = 20_000
FEATURE_CLEANING_QUERY_CHUNK = 1_024
FEATURE_CLEANING_ADAPTIVE_MAD_Z = 6.0
FEATURE_CLEANING_SCOPE_INNER_ONLY = "inner_only"
FEATURE_CLEANING_SCOPE_INNER_AND_EDGE = "inner_and_edge"
FEATURE_CLEANING_SCOPE_CHOICES = (
    FEATURE_CLEANING_SCOPE_INNER_ONLY,
    FEATURE_CLEANING_SCOPE_INNER_AND_EDGE,
)

# 舊版訓練 wizard 曾使用固定 3 full + 5 corners-only 的選片策略。
# 目前改為使用者自行選片，所有選到的 panel 都完整切 tile。
WIZARD_FULL_PANEL_COUNT = 3
WIZARD_CORNERS_ONLY_PANEL_COUNT = 5
WIZARD_TOTAL_PANEL_COUNT = WIZARD_FULL_PANEL_COUNT + WIZARD_CORNERS_ONLY_PANEL_COUNT  # 8


def derive_panel_modes(panel_count: int) -> List[str]:
    """依目前 wizard 行為推 panel_modes：所有使用者選到的 panel 都完整切 tile。"""
    return [PANEL_MODE_FULL] * max(0, panel_count)


def normalize_panel_modes(panel_modes: Optional[List[str]], panel_count: int) -> List[str]:
    """驗證每片 panel 的切片模式；未提供時維持舊版全 full 行為。"""
    if panel_modes is None:
        return derive_panel_modes(panel_count)
    if not isinstance(panel_modes, list) or len(panel_modes) != panel_count:
        raise ValueError("panel_modes must be a list with the same length as panel_paths")
    for mode in panel_modes:
        if not isinstance(mode, str) or mode not in PANEL_MODE_CHOICES:
            raise ValueError(f"invalid panel mode: {mode}")
    return list(panel_modes)


def panel_mode_zones(mode: str) -> set:
    """回傳指定 panel 模式可寫入的 tile zone。"""
    if mode == PANEL_MODE_FULL:
        return {ZONE_INNER, ZONE_EDGE}
    if mode == PANEL_MODE_INNER_ONLY:
        return {ZONE_INNER}
    if mode in (PANEL_MODE_EDGE_ONLY, PANEL_MODE_CORNERS_ONLY):
        return {ZONE_EDGE}
    raise ValueError(f"invalid panel mode: {mode}")

# AOI 炸彈依映射後的 crop 中心位置判定 inner/edge。
# 該 zone 的 NG 樣本少於此閾值時，訓練端退回該 lighting 全部 NG（避免 calibration 失準）。
MIN_NG_PER_ZONE = 5


def _classify_ng_crop_zone(
    center_x: int,
    center_y: int,
    bounds: Tuple[int, int, int, int],
    tile_size: int = 512,
) -> str:
    """依炸彈座標到 panel 四邊的距離決定應校正 inner 或 edge 模型。"""
    x1, y1, x2, y2 = bounds
    edge_distance = max(1, int(tile_size) // 2)
    distance = min(center_x - x1, x2 - center_x, center_y - y1, y2 - center_y)
    return ZONE_EDGE if distance <= edge_distance else ZONE_INNER


@dataclass
class TrainingConfig:
    machine_id: str
    panel_paths: List[Path]
    over_review_root: Path
    output_root: Path = Path("model")
    backbone_cache_dir: Path = Path("deployment/torch_hub_cache")
    required_backbones: List[str] = field(
        default_factory=lambda: ["wide_resnet50_2-32ee1156.pth"]
    )

    batch_size: int = 8
    image_size: tuple = (512, 512)
    tile_stride: int = DEFAULT_TRAIN_TILE_STRIDE
    coreset_ratio: float = 0.1
    max_epochs: int = 1
    precision: str = "float16"
    feature_layers: str = PATCHCORE_FEATURE_LAYERS_DEFAULT
    feature_pool_kernel_size: int = PATCHCORE_FEATURE_POOL_KERNEL_DEFAULT
    feature_cleaning_mode: str = FEATURE_CLEANING_MODE_OFF
    feature_cleaning_scope: str = FEATURE_CLEANING_SCOPE_INNER_ONLY
    feature_cleaning_keep_ratio: float = FEATURE_CLEANING_KEEP_RATIO_DEFAULT
    feature_cleaning_center_size: int = FEATURE_CLEANING_CENTER_SIZE_DEFAULT
    image_preprocess_pipeline: List[Dict[str, Any]] = field(default_factory=list)
    preprocess_after_tiling: bool = False
    training_data_source: Dict[str, Any] = field(
        default_factory=lambda: {"type": "inference_records"}
    )
    feature_cleaning_by_zone: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    image_preprocess_pipelines: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


# 使用者可從 step1 表單覆寫的 PatchCore 超參數。
# 同時做為前後端的單一資料來源：capi_web 的請求驗證、
# step1 的前端表單、capi_train_runner 套用、以及未知 key 防呆都讀此表。
# spec 兩種形態：
#   數值型 → {"type": int/float, "min": x, "max": y}
#   選項型 → {"type": str, "choices": [...]}
USER_TRAINABLE_PARAM_SPECS: Dict[str, Dict] = {
    "batch_size":    {"type": int,   "min": 1,    "max": 32},
    "coreset_ratio": {"type": float, "min": 0.01, "max": 0.5},
    "max_epochs":    {"type": int,   "min": 1,    "max": 10},
    "precision":     {"type": str,   "choices": ["float32", "float16"]},
    "feature_layers": {"type": str,   "choices": list(PATCHCORE_FEATURE_LAYERS_CHOICES)},
    "feature_pool_kernel_size": {"type": int, "choices": list(PATCHCORE_FEATURE_POOL_KERNEL_CHOICES)},
    "feature_cleaning_mode": {"type": str, "choices": list(FEATURE_CLEANING_MODE_CHOICES)},
    "feature_cleaning_scope": {"type": str, "choices": list(FEATURE_CLEANING_SCOPE_CHOICES)},
    "feature_cleaning_keep_ratio": {
        "type": float,
        "min": FEATURE_CLEANING_KEEP_RATIO_MIN,
        "max": FEATURE_CLEANING_KEEP_RATIO_MAX,
    },
    "feature_cleaning_center_size": {
        "type": int,
        "min": FEATURE_CLEANING_CENTER_SIZE_MIN,
        "max": FEATURE_CLEANING_CENTER_SIZE_MAX,
    },
    "feature_cleaning_by_zone": {"type": dict},
}
USER_TRAINABLE_PARAM_NAMES: Tuple[str, ...] = tuple(USER_TRAINABLE_PARAM_SPECS.keys())


def normalize_feature_cleaning_by_zone(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Validate the optional INNER/EDGE-specific feature-cleaning recipe."""
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict):
        raise ValueError("feature_cleaning_by_zone must be an object")
    unknown_zones = set(raw) - set(ZONES)
    missing_zones = set(ZONES) - set(raw)
    if unknown_zones or missing_zones:
        raise ValueError("feature_cleaning_by_zone must contain inner and edge only")

    normalized: Dict[str, Dict[str, Any]] = {}
    for zone in ZONES:
        item = raw.get(zone)
        if not isinstance(item, dict):
            raise ValueError(f"feature_cleaning_by_zone.{zone} must be an object")
        unknown = set(item) - {"mode", "k", "keep_ratio"}
        if unknown:
            raise ValueError(
                f"feature_cleaning_by_zone.{zone} has unknown keys: {sorted(unknown)}"
            )
        mode = item.get("mode", FEATURE_CLEANING_MODE_OFF)
        if mode not in FEATURE_CLEANING_MODE_CHOICES:
            raise ValueError(
                f"feature_cleaning_by_zone.{zone}.mode must be one of "
                f"{list(FEATURE_CLEANING_MODE_CHOICES)}"
            )
        k_raw = item.get("k", FEATURE_CLEANING_K)
        if isinstance(k_raw, bool):
            raise ValueError(f"feature_cleaning_by_zone.{zone}.k must be an integer")
        try:
            k_numeric = float(k_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"feature_cleaning_by_zone.{zone}.k must be an integer"
            ) from exc
        if not k_numeric.is_integer():
            raise ValueError(f"feature_cleaning_by_zone.{zone}.k must be an integer")
        k_value = int(k_numeric)
        if not FEATURE_CLEANING_K_MIN <= k_value <= FEATURE_CLEANING_K_MAX:
            raise ValueError(
                f"feature_cleaning_by_zone.{zone}.k must be between "
                f"{FEATURE_CLEANING_K_MIN} and {FEATURE_CLEANING_K_MAX}"
            )
        ratio_raw = item.get("keep_ratio", FEATURE_CLEANING_KEEP_RATIO_DEFAULT)
        if isinstance(ratio_raw, bool):
            raise ValueError(
                f"feature_cleaning_by_zone.{zone}.keep_ratio must be a number"
            )
        try:
            keep_ratio = float(ratio_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"feature_cleaning_by_zone.{zone}.keep_ratio must be a number"
            ) from exc
        if not FEATURE_CLEANING_KEEP_RATIO_MIN <= keep_ratio <= FEATURE_CLEANING_KEEP_RATIO_MAX:
            raise ValueError(
                f"feature_cleaning_by_zone.{zone}.keep_ratio must be between "
                f"{FEATURE_CLEANING_KEEP_RATIO_MIN} and {FEATURE_CLEANING_KEEP_RATIO_MAX}"
            )
        normalized[zone] = {
            "mode": mode,
            "k": k_value,
            "keep_ratio": keep_ratio,
        }
    return normalized


def normalize_image_preprocess_pipelines(raw: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Validate optional per-zone tile preprocessing pipelines."""
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict):
        raise ValueError("image_preprocess_pipelines must be an object")
    unknown_zones = set(raw) - set(ZONES)
    missing_zones = set(ZONES) - set(raw)
    if unknown_zones or missing_zones:
        raise ValueError("image_preprocess_pipelines must contain inner and edge only")
    from capi_image_preprocess_lab import normalize_preprocess_pipeline

    return {
        zone: normalize_preprocess_pipeline(raw.get(zone) or [])
        for zone in ZONES
    }


def feature_cleaning_config_for_zone(
    cfg: TrainingConfig,
    zone: str,
) -> Dict[str, Any]:
    """Resolve a zone recipe, falling back to the legacy shared settings."""
    by_zone = normalize_feature_cleaning_by_zone(cfg.feature_cleaning_by_zone)
    if by_zone:
        return dict(by_zone[zone])
    enabled = (
        cfg.feature_cleaning_mode != FEATURE_CLEANING_MODE_OFF
        and (
            zone == ZONE_INNER
            or cfg.feature_cleaning_scope == FEATURE_CLEANING_SCOPE_INNER_AND_EDGE
        )
    )
    return {
        "mode": cfg.feature_cleaning_mode if enabled else FEATURE_CLEANING_MODE_OFF,
        "k": FEATURE_CLEANING_K,
        "keep_ratio": float(cfg.feature_cleaning_keep_ratio),
    }


def _patchcore_layers_for_mode(feature_layers: Optional[str]) -> Tuple[str, ...]:
    mode = feature_layers or PATCHCORE_FEATURE_LAYERS_DEFAULT
    try:
        return PATCHCORE_FEATURE_LAYER_MAP[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported PatchCore feature_layers: {mode}") from exc


def apply_user_training_params(
    cfg: TrainingConfig,
    params: Optional[Dict],
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """把 step1 表單覆寫的 PatchCore 超參數套到 TrainingConfig。

    None / 空 dict 直接 return，cfg 維持 dataclass 預設值。
    含 USER_TRAINABLE_PARAM_SPECS 之外的 key 會 raise，避免 DB 內髒資料 silent
    fall-through 到訓練（caller 在寫入 DB 前已驗證過，這裡是第二層防線）。
    """
    if not params:
        return
    unknown = set(params.keys()) - set(USER_TRAINABLE_PARAM_SPECS.keys())
    if unknown:
        raise ValueError(f"unknown user training params: {sorted(unknown)}")
    for key, val in params.items():
        if key == "feature_cleaning_by_zone":
            val = normalize_feature_cleaning_by_zone(val)
        setattr(cfg, key, val)
    if log_fn is not None:
        log_fn(f"使用者覆寫訓練參數: {params}")


def generate_job_id(machine_id: str) -> str:
    # 加 4-char 隨機後綴避免兩個並行 start 在同一秒撞 id（多 job 共存後是真的會發生；
    # Windows datetime resolution 不足以保證微秒唯一）
    import secrets
    suffix = secrets.token_hex(2)
    return f"train_{machine_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"


def preprocess_panels_to_pool(
    job_id: str,
    cfg: "TrainingConfig",
    preprocess_cfg: "PreprocessConfig",
    db: TrainingDB,
    thumb_dir: Path,
    log: Callable[[str], None],
    panel_modes: Optional[List[str]] = None,
    target_lightings: Optional[Iterable[str]] = None,
    target_units: Optional[Iterable[str]] = None,
) -> dict:
    """將 cfg.panel_paths 全部前處理 + 切 tile + 寫 DB。

    panel_modes 與 cfg.panel_paths 同長度：
      - "full"          : 收所有 inner / edge tile（原行為）
      - "inner_only"    : 只收 inner tile
      - "edge_only"     : 只收 edge tile
      - "corners_only"  : 只收 edge 且 is_corner=True 的 tile（4 outer-extension 角 +
                          4 inner-edge 角 / lighting）。長邊與 inner 不取樣，
                          目的是補強 edge 模型的角落樣本。
    None 視同全 full，與舊呼叫者相容。
    """
    panel_modes = normalize_panel_modes(panel_modes, len(cfg.panel_paths))
    target_lighting_set = set(target_lightings) if target_lightings is not None else None
    target_unit_set = set(target_units) if target_units is not None else None

    thumb_dir.mkdir(parents=True, exist_ok=True)
    (thumb_dir / "tiles").mkdir(parents=True, exist_ok=True)
    (thumb_dir / "thumb").mkdir(parents=True, exist_ok=True)
    panel_success_full = 0
    panel_success_inner_only = 0
    panel_success_edge_only = 0
    panel_success_corner = 0
    panel_fail = 0
    total_tiles = 0
    preprocess_desc = ""
    preprocess_after_tiling = bool(getattr(preprocess_cfg, "preprocess_after_tiling", False))
    if (
        preprocess_cfg.image_preprocess_pipeline
        or getattr(preprocess_cfg, "image_preprocess_pipelines", None)
    ):
        from capi_image_preprocess_lab import describe_preprocess_pipeline
        if getattr(preprocess_cfg, "image_preprocess_pipelines", None):
            preprocess_desc = "INNER/EDGE 分區前處理"
        else:
            preprocess_desc = describe_preprocess_pipeline(
                preprocess_cfg.image_preprocess_pipeline
            )
        preprocess_mode = (
            "先切分後處理（每個 tile 套用）"
            if preprocess_after_tiling
            else "先處理整張圖再切片"
        )
        log(
            f"影像前處理模式: {preprocess_mode}；流程: "
            f"{preprocess_desc}"
        )

    for idx, (panel_dir, mode) in enumerate(zip(cfg.panel_paths, panel_modes), 1):
        mode_label = {
            PANEL_MODE_FULL: "INNER + EDGE",
            PANEL_MODE_INNER_ONLY: "僅 INNER",
            PANEL_MODE_EDGE_ONLY: "僅 EDGE",
            PANEL_MODE_CORNERS_ONLY: "僅 EDGE 4 角",
        }[mode]
        allowed_zones = panel_mode_zones(mode)
        log(f"[{idx}/{len(cfg.panel_paths)}] panel {panel_dir.name} ({mode_label})")
        try:
            results = preprocess_panel_folder(panel_dir, preprocess_cfg)
        except Exception as e:
            log(f"  ✗ 處理失敗: {e}")
            panel_fail += 1
            continue
        if not results:
            log(f"  ✗ 無有效 lighting 圖")
            panel_fail += 1
            continue

        polygon_failed_count = sum(1 for r in results.values() if r.polygon_detection_failed)
        if polygon_failed_count > 0:
            log(f"  ⚠ {polygon_failed_count} lighting polygon 偵測失敗")

        # 先依 panel mode 過濾 zone，再寫 .png + 縮圖 + DB，避免浪費 IO。
        tile_records = []
        for lighting, result in results.items():
            if target_lighting_set is not None and lighting not in target_lighting_set:
                continue
            for tile in result.tiles:
                if target_unit_set is not None and f"{lighting}-{tile.zone}" not in target_unit_set:
                    continue
                if tile.zone not in allowed_zones:
                    continue
                if mode == PANEL_MODE_CORNERS_ONLY and not tile.is_corner:
                    continue
                tile_filename = f"{job_id}_{panel_dir.name}_{lighting}_t{tile.tile_id:04d}.png"
                tile_path = thumb_dir / "tiles" / tile_filename
                cv2.imwrite(str(tile_path), tile.image)

                thumb_path = thumb_dir / "thumb" / tile_filename
                thumb = cv2.resize(tile.image, (96, 96))
                cv2.imwrite(str(thumb_path), thumb)

                tile_x = getattr(tile, "x1", None)
                tile_y = getattr(tile, "y1", None)
                tile_x2 = getattr(tile, "x2", None)
                tile_y2 = getattr(tile, "y2", None)

                tile_records.append({
                    "lighting": lighting,
                    "zone": tile.zone,
                    "source": "ok",
                    "source_path": str(tile_path.resolve()),
                    "thumb_path": str(thumb_path.resolve()),
                    "panel_path": str(panel_dir.resolve()),
                    "tile_index": int(tile.tile_id),
                    "tile_x": int(tile_x) if tile_x is not None else None,
                    "tile_y": int(tile_y) if tile_y is not None else None,
                    "tile_width": (
                        int(tile_x2 - tile_x)
                        if tile_x is not None and tile_x2 is not None
                        else None
                    ),
                    "tile_height": (
                        int(tile_y2 - tile_y)
                        if tile_y is not None and tile_y2 is not None
                        else None
                    ),
                })

        if tile_records:
            db.insert_tile_pool(job_id, tile_records)
            total_tiles += len(tile_records)
            if mode == PANEL_MODE_FULL:
                panel_success_full += 1
            elif mode == PANEL_MODE_INNER_ONLY:
                panel_success_inner_only += 1
            elif mode == PANEL_MODE_EDGE_ONLY:
                panel_success_edge_only += 1
            else:
                panel_success_corner += 1
            log(f"  ✓ 切出 {len(tile_records)} tile")
            if preprocess_desc:
                if preprocess_after_tiling:
                    log(f"  ↳ 前處理: 已對這 {len(tile_records)} 個 tile 套用 {preprocess_desc}")
                else:
                    log(f"  ↳ 前處理: 已先對原圖套用 {preprocess_desc}，再切出這批 tile")
        else:
            panel_fail += 1
            log("  ✗ 無 tile 寫入")

    return {
        # panel_success 維持原 key 給舊呼叫者用（= 所有有 tile 寫入的 panel）。
        "panel_success": (
            panel_success_full
            + panel_success_inner_only
            + panel_success_edge_only
            + panel_success_corner
        ),
        "panel_success_full": panel_success_full,
        "panel_success_inner_only": panel_success_inner_only,
        "panel_success_edge_only": panel_success_edge_only,
        "panel_success_corner": panel_success_corner,
        "panel_fail": panel_fail,
        "total_tiles": total_tiles,
    }


def sample_ng_tiles(
    job_id: str,
    over_review_root: Path,
    db: TrainingDB,
    thumb_dir: Optional[Path] = None,
    per_lighting: int = NG_TILES_PER_LIGHTING,
    log: Callable[[str], None] = print,
    lightings: Optional[Iterable[str]] = None,
    preprocess_cfg: Optional[PreprocessConfig] = None,
    machine_id: str = "",
    rotate_180: bool = False,
    ng_validation_base_dir: Optional[Path] = None,
) -> dict:
    """準備同機種 Client AOI 炸彈 crop，作為 PatchCore 驗證 NG。

    ``over_review_root`` 僅保留舊版呼叫介面相容；新版不再讀取該目錄。
    已有訓練炸彈快取時直接從 NG 驗證庫載入；缺少的 lighting 才讀取
    推論原圖裁切，並將未前處理的 512 crop 持久化供下次重用。
    B0F00000 黑畫面固定排除。這些 crop 只會以 ``source=ng`` 進入
    ``test/anormal``，不會加入 normal memory bank。
    AOI 映射、panel polygon 內縮、zone 判定與前處理順序和正式 v2
    推論共用，避免 NG 校正樣本與實際驗證 tile 不同 ROI。
    """
    del over_review_root
    requested_lightings = tuple(lightings) if lightings is not None else LIGHTINGS
    target_lightings = tuple(dict.fromkeys(
        str(lighting).strip().upper()
        for lighting in requested_lightings
        if str(lighting).strip() and str(lighting).strip().upper() != "B0F00000"
    ))
    result_stats = {
        "sampled": 0,
        "missing_lightings": [],
        "black_skipped": 0,
        "invalid_skipped": 0,
        "cache_reused": 0,
        "cache_saved": 0,
    }
    if not target_lightings:
        return result_stats
    if not machine_id:
        log("⚠ 未提供 machine_id，無法從推論紀錄抽 AOI 炸彈 crop")
        result_stats["missing_lightings"] = list(target_lightings)
        return result_stats

    tile_size = int(
        getattr(preprocess_cfg, "tile_size", 512)
        if preprocess_cfg is not None else 512
    )
    validation_root = (
        Path(ng_validation_base_dir).resolve()
        if ng_validation_base_dir is not None else None
    )
    cached_by_lighting: Dict[str, List[dict]] = {
        lighting: [] for lighting in target_lightings
    }
    if validation_root is not None:
        try:
            cached_samples = db.list_training_bomb_validation_samples(
                machine_id=str(machine_id).strip(),
                lightings=target_lightings,
            )
        except AttributeError:
            cached_samples = []
            log("⚠ 資料庫版本不支援訓練 NG 快取，改走推論原圖裁切")
        for sample in cached_samples:
            lighting = str(sample.get("lighting") or "").strip().upper()
            zone = str(sample.get("zone") or "").strip().lower()
            if lighting not in cached_by_lighting or zone not in ZONES:
                continue
            try:
                crop_path = Path(str(sample.get("crop_path") or "")).resolve()
                crop_path.relative_to(validation_root)
            except (OSError, ValueError):
                result_stats["invalid_skipped"] += 1
                continue
            if not crop_path.is_file():
                result_stats["invalid_skipped"] += 1
                continue
            cached_probe = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
            if cached_probe is None or cached_probe.shape[:2] != (tile_size, tile_size):
                result_stats["invalid_skipped"] += 1
                continue
            cached_by_lighting[lighting].append({
                **sample,
                "crop_path": str(crop_path),
            })

    uncached_lightings = tuple(
        lighting for lighting in target_lightings
        if not cached_by_lighting.get(lighting)
    )
    candidates = []
    if uncached_lightings:
        try:
            candidates = db.list_training_bomb_candidates(
                machine_id=str(machine_id).strip(),
                lightings=uncached_lightings,
            )
        except AttributeError:
            log("⚠ 資料庫版本不支援 AOI 炸彈 crop 查詢，未使用 over_review 舊資料")

    by_lighting: Dict[str, List[dict]] = {lighting: [] for lighting in target_lightings}
    for source_row in candidates:
        image_lighting = canonical_image_prefix(source_row.get("image_name") or "").upper()
        try:
            bomb_info = json.loads(str(source_row.get("client_bomb_info") or ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            result_stats["invalid_skipped"] += 1
            continue
        bomb_lighting = canonical_image_prefix(bomb_info.get("image_prefix") or "").upper()
        if image_lighting == "B0F00000" or bomb_lighting == "B0F00000":
            result_stats["black_skipped"] += 1
            continue
        if image_lighting != bomb_lighting or image_lighting not in by_lighting:
            continue

        valid_coords = []
        for coord in bomb_info.get("coordinates") or []:
            try:
                valid_coords.append((int(coord[0]), int(coord[1])))
            except (TypeError, ValueError, IndexError):
                continue
        defect_type = str(bomb_info.get("defect_type") or "point").strip().lower()
        if defect_type == "line" and len(valid_coords) >= 2:
            pt1, pt2 = valid_coords[0], valid_coords[1]
            crop_centers = [((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)]
        else:
            crop_centers = valid_coords
            defect_type = "point"
        if not crop_centers:
            result_stats["invalid_skipped"] += 1
            continue
        for coord_index, (product_x, product_y) in enumerate(crop_centers):
            by_lighting[image_lighting].append({
                **source_row,
                "source_type": defect_type,
                "coord_index": coord_index,
                "product_x": product_x,
                "product_y": product_y,
            })

    sampled = 0
    missing: List[str] = []
    work_root = thumb_dir or (Path(".tmp/train_new_thumbs") / job_id)
    apply_preprocess_pipeline = None
    preprocess_desc = ""
    if (
        preprocess_cfg is not None
        and (
            preprocess_cfg.image_preprocess_pipeline
            or getattr(preprocess_cfg, "image_preprocess_pipelines", None)
        )
    ):
        from capi_image_preprocess_lab import (
            apply_preprocess_pipeline as _apply_preprocess_pipeline,
            describe_preprocess_pipeline,
        )
        apply_preprocess_pipeline = _apply_preprocess_pipeline
        if getattr(preprocess_cfg, "image_preprocess_pipelines", None):
            preprocess_desc = "INNER/EDGE 分區前處理"
        else:
            preprocess_desc = describe_preprocess_pipeline(preprocess_cfg.image_preprocess_pipeline)

    def write_job_tile(
        *,
        lighting: str,
        zone: str,
        stem: str,
        crop,
        panel_path: Path,
        tile_index: int,
        tile_x: int,
        tile_y: int,
        error_label: str,
    ) -> Optional[dict]:
        source_path = work_root / "tiles" / "ng" / lighting / f"{stem}.png"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(source_path), crop):
            result_stats["invalid_skipped"] += 1
            log(f"  ⚠ {lighting}: {error_label}寫入失敗 {source_path}")
            return None

        thumb_path = work_root / "thumb" / "ng" / lighting / f"{stem}.png"
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        thumb = cv2.resize(crop, (96, 96))
        if not cv2.imwrite(str(thumb_path), thumb):
            thumb_path = source_path
        return {
            "lighting": lighting,
            "zone": zone,
            "source": "ng",
            "source_path": str(source_path.resolve()),
            "thumb_path": str(thumb_path.resolve()),
            "panel_path": str(panel_path),
            "tile_index": int(tile_index),
            "tile_x": int(tile_x),
            "tile_y": int(tile_y),
            "tile_width": tile_size,
            "tile_height": tile_size,
        }

    def safe_path_token(value: Any, fallback: str, max_length: int) -> str:
        token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or fallback))
        return token[:max_length] or fallback

    # Cache geometry/preprocessed pixels once per source image.  A single
    # inference record can contain several AOI coordinates; re-running Otsu and
    # polygon fitting for every coordinate made the old sampler both slower and
    # more likely to drift from formal inference.
    geometry_cache: Dict[
        str,
        Tuple[Any, Any, Tuple[int, int, int, int], Optional[Any]],
    ] = {}

    for lighting in target_lightings:
        cached_samples = list(cached_by_lighting.get(lighting) or [])
        if cached_samples:
            random.shuffle(cached_samples)
            cached_records = []
            cached_edge_n = cached_inner_n = 0
            cached_preprocessed_n = 0
            for sample in cached_samples[:per_lighting]:
                crop_path = Path(str(sample.get("crop_path") or ""))
                crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
                if crop is None or crop.shape[:2] != (tile_size, tile_size):
                    result_stats["invalid_skipped"] += 1
                    log(f"  ⚠ {lighting}: NG 驗證庫 crop 無法讀取或尺寸錯誤 {crop_path}")
                    continue
                zone = str(sample.get("zone") or "").strip().lower()
                if apply_preprocess_pipeline is not None and preprocess_cfg is not None:
                    try:
                        pipeline = image_preprocess_pipeline_for_zone(preprocess_cfg, zone)
                        if pipeline:
                            crop = apply_preprocess_pipeline(crop, pipeline)["image"]
                            cached_preprocessed_n += 1
                    except Exception as exc:
                        log(f"  ⚠ {lighting}: NG 快取前處理失敗 {crop_path.name}: {exc}")

                source_image_path = str(sample.get("source_image_path") or "").strip()
                panel_path = Path(source_image_path) if source_image_path else crop_path
                record = write_job_tile(
                    lighting=lighting,
                    zone=zone,
                    stem=f"{job_id}_{lighting}_ngcache_s{int(sample.get('id') or 0)}",
                    crop=crop,
                    panel_path=panel_path,
                    tile_index=int(sample.get("id") or 0),
                    tile_x=int(sample.get("tile_x") or 0),
                    tile_y=int(sample.get("tile_y") or 0),
                    error_label="NG 快取 crop ",
                )
                if record is None:
                    continue
                cached_records.append(record)
                if zone == ZONE_EDGE:
                    cached_edge_n += 1
                else:
                    cached_inner_n += 1

            if cached_records:
                db.insert_tile_pool(job_id, cached_records)
                reused = len(cached_records)
                sampled += reused
                result_stats["cache_reused"] += reused
                preprocess_text = (
                    f" / 前處理={cached_preprocessed_n}: {preprocess_desc}"
                    if cached_preprocessed_n else ""
                )
                log(
                    f"  ✓ {lighting}: 從 NG 驗證庫重用 {reused} 個 NG "
                    f"(不重讀推論原圖 / edge={cached_edge_n} / inner={cached_inner_n}"
                    f"{preprocess_text})"
                )
                continue
            log(f"  ⚠ {lighting}: NG 驗證庫快取不可用，改走推論原圖裁切")

        lighting_candidates = list(by_lighting.get(lighting) or [])
        if not lighting_candidates:
            missing.append(lighting)
            log(f"⚠ {lighting}: 推論紀錄無可用 AOI 炸彈 crop")
            continue
        random.shuffle(lighting_candidates)
        records = []
        validation_rows = []
        edge_n = inner_n = unknown_n = 0
        preprocessed_n = 0
        for candidate in lighting_candidates:
            if len(records) >= per_lighting:
                break

            raw_source_path = Path(str(candidate.get("image_path") or ""))
            if not raw_source_path.is_file():
                fallback_path = (
                    Path(str(candidate.get("image_dir") or ""))
                    / Path(str(candidate.get("image_name") or "")).name
                )
                raw_source_path = fallback_path if fallback_path.is_file() else raw_source_path
            if not raw_source_path.is_file():
                result_stats["invalid_skipped"] += 1
                log(f"  ⚠ {lighting}: 炸彈原圖不存在 {raw_source_path}")
                continue

            source_key = str(raw_source_path.resolve())
            geometry = geometry_cache.get(source_key)
            if geometry is None:
                raw_image = read_detection_image(
                    raw_source_path,
                    cv2.IMREAD_GRAYSCALE,
                    rotate_180=bool(rotate_180),
                )
                if raw_image is None:
                    result_stats["invalid_skipped"] += 1
                    log(f"  ⚠ {lighting}: 炸彈原圖讀取失敗 {raw_source_path}")
                    continue

                # This is intentionally the same ordering as
                # preprocess_panel_image / _create_aoi_centered_tiles_v2:
                # global pipeline first, then boundary detection and crop.
                geometry_image = raw_image
                if (
                    preprocess_cfg is not None
                    and preprocess_cfg.image_preprocess_pipeline
                    and not getattr(preprocess_cfg, "preprocess_after_tiling", False)
                    and apply_preprocess_pipeline is not None
                ):
                    try:
                        geometry_image = apply_preprocess_pipeline(
                            raw_image, preprocess_cfg.image_preprocess_pipeline,
                        )["image"]
                    except Exception as exc:
                        log(f"  ⚠ {lighting}: NG 整圖前處理失敗 {raw_source_path.name}: {exc}")
                        geometry_image = raw_image

                detected_bounds = None
                detected_polygon = None
                if preprocess_cfg is not None:
                    try:
                        detected_bounds, detected_polygon = detect_panel_polygon(
                            geometry_image, preprocess_cfg,
                        )
                    except Exception as exc:
                        log(f"  ⚠ {lighting}: panel 邊界偵測失敗 {raw_source_path.name}: {exc}")

                try:
                    db_bounds = tuple(
                        int(part.strip())
                        for part in str(candidate.get("otsu_bounds") or "").split(",")
                    )
                    if (
                        len(db_bounds) != 4
                        or db_bounds[2] <= db_bounds[0]
                        or db_bounds[3] <= db_bounds[1]
                    ):
                        raise ValueError("invalid otsu_bounds")
                except (TypeError, ValueError, cv2.error) as exc:
                    db_bounds = (0, 0, raw_image.shape[1], raw_image.shape[0])
                    log(f"  ⚠ {lighting}: {raw_source_path.name} 無有效 Otsu bounds，改用全圖映射 ({exc})")

                bounds = detected_bounds or db_bounds
                polygon = (
                    detected_polygon
                    if detected_polygon is not None
                    else rect_polygon_from_bounds(bounds)
                )
                geometry = (
                    raw_image,
                    geometry_image,
                    tuple(int(v) for v in bounds),
                    polygon,
                )
                geometry_cache[source_key] = geometry

            raw_image, image, bounds, polygon = geometry
            product_width = int(
                getattr(preprocess_cfg, "product_resolution", None)[0]
                if preprocess_cfg is not None
                and getattr(preprocess_cfg, "product_resolution", None)
                else candidate.get("resolution_x") or 0
            )
            product_height = int(
                getattr(preprocess_cfg, "product_resolution", None)[1]
                if preprocess_cfg is not None
                and getattr(preprocess_cfg, "product_resolution", None)
                else candidate.get("resolution_y") or 0
            )
            if product_width <= 0:
                product_width = max(1, image.shape[1])
            if product_height <= 0:
                product_height = max(1, image.shape[0])
            product_resolution = (product_width, product_height)
            product_x = int(candidate.get("product_x") or 0)
            product_y = int(candidate.get("product_y") or 0)
            img_x, img_y = map_product_coord_to_image(
                product_x, product_y, bounds, product_resolution, polygon,
            )

            half = tile_size // 2
            centered_tile_x = img_x - half
            centered_tile_y = img_y - half
            tile_x, tile_y, _coverage, _shifted = resolve_inward_polygon_tile(
                anchor_xy=(img_x, img_y),
                polygon=polygon,
                image_shape=image.shape[:2],
                tile_size=tile_size,
                initial_origin=(centered_tile_x, centered_tile_y),
                keep_anchor_inside=True,
                shift_axes=resolve_aoi_inward_shift_axes(
                    img_x, img_y, bounds, tile_size,
                ),
            )
            raw_crop = raw_image[
                tile_y:tile_y + tile_size,
                tile_x:tile_x + tile_size,
            ].copy()
            crop = image[
                tile_y:tile_y + tile_size,
                tile_x:tile_x + tile_size,
            ].copy()
            if (
                raw_crop.shape[:2] != (tile_size, tile_size)
                or crop.shape[:2] != (tile_size, tile_size)
            ):
                result_stats["invalid_skipped"] += 1
                log(
                    f"  ⚠ {lighting}: 炸彈 crop 尺寸不是 {tile_size}x{tile_size}: "
                    f"{crop.shape[:2]}"
                )
                continue

            zone, _anchor_distance = classify_anchor_zone(
                (img_x, img_y), polygon, half,
            )
            if (
                apply_preprocess_pipeline is not None
                and preprocess_cfg is not None
                and getattr(preprocess_cfg, "preprocess_after_tiling", False)
            ):
                try:
                    pipeline = image_preprocess_pipeline_for_zone(preprocess_cfg, zone)
                    if pipeline:
                        crop = apply_preprocess_pipeline(crop, pipeline)["image"]
                        preprocessed_n += 1
                except Exception as exc:
                    log(f"  ⚠ {lighting}: NG 前處理失敗 {raw_source_path.name}: {exc}")

            record_id = int(candidate.get("inference_record_id") or 0)
            source_result_id = int(candidate.get("source_result_id") or 0)
            source_type = str(candidate.get("source_type") or "point")
            coord_index = int(candidate.get("coord_index") or 0)
            stem = (
                f"{job_id}_{lighting}_bomb_r{record_id}_"
                f"img{source_result_id}_{source_type}{coord_index}"
            )
            record = write_job_tile(
                lighting=lighting,
                zone=zone,
                stem=stem,
                crop=crop,
                panel_path=raw_source_path.resolve(),
                tile_index=source_result_id,
                tile_x=tile_x,
                tile_y=tile_y,
                error_label="炸彈 crop ",
            )
            if record is None:
                continue
            records.append(record)

            if zone == ZONE_EDGE:
                edge_n += 1
            elif zone == ZONE_INNER:
                inner_n += 1
            else:
                unknown_n += 1

            if validation_root is not None:
                safe_model = safe_path_token(machine_id, "unknown", 80)
                safe_lighting = safe_path_token(lighting, "unknown", 40)
                safe_zone = safe_path_token(zone, "unknown", 40)
                safe_glass = safe_path_token(candidate.get("glass_id"), "panel", 80)
                request_day = re.sub(
                    r"[^0-9]+", "", str(candidate.get("request_time") or "")[:10]
                ) or datetime.now().strftime("%Y%m%d")
                validation_dir = (
                    validation_root / safe_model / safe_lighting / safe_zone / "crop"
                )
                validation_path = validation_dir / (
                    f"{request_day}_{safe_glass}_r{record_id}_"
                    f"img{source_result_id}_{source_type}{coord_index}.png"
                )
                validation_error_logged = False
                try:
                    validation_dir.mkdir(parents=True, exist_ok=True)
                    validation_written = cv2.imwrite(str(validation_path), raw_crop)
                except (OSError, cv2.error) as exc:
                    validation_written = False
                    validation_error_logged = True
                    log(f"  ⚠ {lighting}: NG 驗證庫 crop 寫入失敗 {validation_path}: {exc}")
                if validation_written:
                    validation_rows.append({
                        "inference_record_id": record_id,
                        "image_result_id": source_result_id,
                        "coord_index": coord_index,
                        "glass_id": str(candidate.get("glass_id") or ""),
                        "model_id": str(machine_id).strip(),
                        "machine_no": str(candidate.get("machine_no") or ""),
                        "request_time": str(candidate.get("request_time") or ""),
                        "image_name": str(candidate.get("image_name") or ""),
                        "source_image_path": str(raw_source_path.resolve()),
                        "lighting": lighting,
                        "zone": zone,
                        "source_type": source_type,
                        "aoi_product_x": product_x,
                        "aoi_product_y": product_y,
                        "aoi_image_x": img_x,
                        "aoi_image_y": img_y,
                        "tile_x": tile_x,
                        "tile_y": tile_y,
                        "tile_w": tile_size,
                        "tile_h": tile_size,
                        "crop_path": str(validation_path.resolve()),
                    })
                elif not validation_error_logged:
                    log(f"  ⚠ {lighting}: NG 驗證庫 crop 寫入失敗 {validation_path}")
        if records:
            db.insert_tile_pool(job_id, records)
            if validation_rows:
                try:
                    saved = db.save_training_bomb_validation_samples(validation_rows)
                    result_stats["cache_saved"] += int(saved or 0)
                    log(f"  ✓ {lighting}: 新增 {int(saved or 0)} 張至 NG 驗證庫")
                except AttributeError:
                    log("  ⚠ 資料庫版本不支援寫入訓練 NG 快取")
                except Exception as exc:
                    log(f"  ⚠ {lighting}: NG 驗證庫寫入失敗，仍繼續本次訓練: {exc}")
        else:
            missing.append(lighting)
        sampled += len(records)
        preprocess_text = (
            f" / 前處理={preprocessed_n}: {preprocess_desc}"
            if preprocessed_n else ""
        )
        if records:
            log(
                f"  ✓ {lighting}: 抽 {len(records)} 個 NG "
                f"(來源=推論紀錄 AOI 炸彈 / edge={edge_n} / inner={inner_n} "
                f"/ 未分類={unknown_n}{preprocess_text})"
            )

    result_stats["sampled"] = sampled
    result_stats["missing_lightings"] = missing
    return result_stats


def _link_or_copy(src: Path, dst: Path) -> None:
    """建立 hardlink，跨 filesystem 或不支援時退回 copy2。"""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def stage_dataset(
    staging_dir: Path,
    train_paths: List[Path],
    ng_paths: List[Path],
) -> List[Path]:
    """為一個 (lighting, zone) unit 準備訓練 staging。

    結構：
      staging_dir/
        train/         (個別 file 的 hardlink/copy)
        test/anormal/  (個別 file)

    為避免 anomalib Folder 對 symlink 行為不一致，用個別檔案 hardlink / copy
    （不是整目錄 mklink）。
    """
    train_dir = staging_dir / "train"
    ng_dir = staging_dir / "test" / "anormal"
    train_dir.mkdir(parents=True, exist_ok=True)
    ng_dir.mkdir(parents=True, exist_ok=True)

    def destination_names(paths: List[Path]) -> List[str]:
        counts: Dict[str, int] = {}
        for src in paths:
            counts[src.name] = counts.get(src.name, 0) + 1
        return [
            f"{index:06d}_{src.name}" if counts[src.name] > 1 else src.name
            for index, src in enumerate(paths)
        ]

    staged_train_paths = []
    for src, name in zip(train_paths, destination_names(train_paths)):
        dst = train_dir / name
        _link_or_copy(src, dst)
        staged_train_paths.append(dst.resolve())
    for src, name in zip(ng_paths, destination_names(ng_paths)):
        dst = ng_dir / name
        _link_or_copy(src, dst)
    return staged_train_paths


def _import_anomalib():
    """延後 import anomalib，方便 unit test monkeypatch。"""
    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    try:
        from anomalib.data.utils import ValSplitMode
        val_mode = ValSplitMode.SAME_AS_TEST
    except ImportError:
        val_mode = "same_as_test"
    return Folder, Patchcore, Engine, ExportType, val_mode


def train_one_patchcore(
    staging_dir: Path,
    run_root: Path,
    unit_label: str,
    cfg: "TrainingConfig" = None,
    log: Optional[Callable[[str], None]] = None,
    experiment_stats_out: Optional[Dict[str, Any]] = None,
    trace_sources: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Path:
    """訓練一個 (lighting, zone) unit。回傳 model.pt 路徑。

    mirrors tools/train_bga_all.py train_one() 的 anomalib 呼叫方式：
    - engine.fit(datamodule=..., model=...)  # 無 model_path
    - engine.export(model=..., export_type=...)  # 無 model_path
    - default_root_dir 控制輸出路徑
    """
    cfg = cfg or TrainingConfig(
        machine_id="?", panel_paths=[], over_review_root=Path("?"),
    )
    Folder, Patchcore, Engine, ExportType, val_mode = _import_anomalib()

    if run_root.exists():
        shutil.rmtree(run_root, ignore_errors=True)
    run_root.mkdir(parents=True, exist_ok=True)

    if log:
        log(f"{unit_label}: 建立 Folder datamodule")
    abnormal_dir = staging_dir / "test" / "anormal"
    has_abnormal_images = abnormal_dir.is_dir() and any(
        path.is_file() for path in abnormal_dir.iterdir()
    )
    folder_kwargs = {}
    if has_abnormal_images:
        folder_kwargs["abnormal_dir"] = "test/anormal"
    else:
        folder_kwargs.update(abnormal_dir=None, test_split_mode="synthetic")
        if log:
            log(f"{unit_label}: 無 NG 樣本，改用 synthetic anomaly 建立驗證集")

    datamodule = Folder(
        name=f"unit_{unit_label}",
        root=staging_dir,
        normal_dir="train",
        train_batch_size=cfg.batch_size,
        eval_batch_size=cfg.batch_size,
        num_workers=16,
        val_split_mode=val_mode,
        **folder_kwargs,
    )
    try:
        datamodule.image_size = cfg.image_size
    except Exception:
        pass

    if log:
        log(f"{unit_label}: 建立 PatchCore model")
    feature_layers = _patchcore_layers_for_mode(cfg.feature_layers)
    if log:
        log(f"{unit_label}: PatchCore feature layers={'+'.join(feature_layers)}")
    model = Patchcore(
        layers=feature_layers,
        coreset_sampling_ratio=cfg.coreset_ratio,
        precision=cfg.precision,
    )
    model.pre_processor = Patchcore.configure_pre_processor(image_size=cfg.image_size)

    pool_kernel = int(cfg.feature_pool_kernel_size)
    if pool_kernel not in PATCHCORE_FEATURE_POOL_KERNEL_CHOICES:
        raise ValueError(f"unsupported feature_pool_kernel_size: {pool_kernel}")
    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "feature_pooler"):
        import torch
        inner_model.feature_pooler = torch.nn.AvgPool2d(
            kernel_size=pool_kernel,
            stride=1,
            padding=pool_kernel // 2,
        )
    elif pool_kernel != PATCHCORE_FEATURE_POOL_KERNEL_DEFAULT:
        raise RuntimeError("PatchCore model does not expose feature_pooler")
    elif log:
        log(f"{unit_label}: feature_pooler unavailable; keep PatchCore default aggregation")
    if log:
        log(f"{unit_label}: feature aggregation={pool_kernel}x{pool_kernel}")

    cleaning_mode = cfg.feature_cleaning_mode
    if cleaning_mode not in FEATURE_CLEANING_MODE_CHOICES:
        raise ValueError(f"unsupported feature_cleaning_mode: {cleaning_mode}")
    cleaning_scope = cfg.feature_cleaning_scope
    if cleaning_scope not in FEATURE_CLEANING_SCOPE_CHOICES:
        raise ValueError(f"unsupported feature_cleaning_scope: {cleaning_scope}")
    cleaning_keep_ratio = float(cfg.feature_cleaning_keep_ratio)
    if not FEATURE_CLEANING_KEEP_RATIO_MIN <= cleaning_keep_ratio <= FEATURE_CLEANING_KEEP_RATIO_MAX:
        raise ValueError(
            "feature_cleaning_keep_ratio must be between "
            f"{FEATURE_CLEANING_KEEP_RATIO_MIN} and {FEATURE_CLEANING_KEEP_RATIO_MAX}"
        )
    cleaning_center_size = int(cfg.feature_cleaning_center_size)
    max_center_size = min(int(cfg.image_size[0]), int(cfg.image_size[1]))
    if not FEATURE_CLEANING_CENTER_SIZE_MIN <= cleaning_center_size <= max_center_size:
        raise ValueError(
            "feature_cleaning_center_size must be between "
            f"{FEATURE_CLEANING_CENTER_SIZE_MIN} and {max_center_size}"
        )
    zone = unit_label.rsplit("-", 1)[-1].lower()
    zone_cleaning = feature_cleaning_config_for_zone(cfg, zone)
    zone_cleaning_mode = zone_cleaning["mode"]
    zone_cleaning_k = int(zone_cleaning["k"])
    zone_cleaning_keep_ratio = float(zone_cleaning["keep_ratio"])
    per_zone_cleaning = bool(cfg.feature_cleaning_by_zone)
    cleaning_stats: Dict[str, Any] = {
        "mode": zone_cleaning_mode,
        "scope": "per_zone" if per_zone_cleaning else cleaning_scope,
        "zone": zone,
        "k": zone_cleaning_k,
        "keep_ratio": zone_cleaning_keep_ratio,
        "center_size": cleaning_center_size,
        "applied": False,
        "reason": "disabled",
    }
    callbacks = None
    cleaning_callback = None
    if zone_cleaning_mode in (
        FEATURE_CLEANING_MODE_KNN_Q99,
        FEATURE_CLEANING_MODE_CONTEXT_OVERLAP_ADAPTIVE,
    ):
        from capi_patchcore_feature_cleaning import FeatureDensityCleaningCallback
        context_adaptive = (
            zone_cleaning_mode == FEATURE_CLEANING_MODE_CONTEXT_OVERLAP_ADAPTIVE
        )
        cleaning_callback = FeatureDensityCleaningCallback(
            k=zone_cleaning_k,
            keep_ratio=zone_cleaning_keep_ratio,
            center_size=None if context_adaptive else cleaning_center_size,
            seed=FEATURE_CLEANING_SEED,
            reference_size=FEATURE_CLEANING_REFERENCE_SIZE,
            query_chunk=FEATURE_CLEANING_QUERY_CHUNK,
            trace_sources=trace_sources,
            strategy=(
                "context_overlap_adaptive"
                if context_adaptive
                else "quantile"
            ),
            adaptive_mad_z=FEATURE_CLEANING_ADAPTIVE_MAD_Z,
        )
        callbacks = [cleaning_callback]
        cleaning_stats["reason"] = "pending"
        if log:
            cleaning_detail = (
                "context=overlap/adaptive"
                if context_adaptive
                else f"center={cleaning_center_size}x{cleaning_center_size}"
            )
            log(
                f"{unit_label}: feature cleaning enabled "
                f"(scope={'per_zone' if per_zone_cleaning else cleaning_scope}, "
                f"mode={zone_cleaning_mode}, "
                f"cosine k={zone_cleaning_k}, keep={zone_cleaning_keep_ratio:.1%}, "
                f"{cleaning_detail})"
            )
    elif (
        not per_zone_cleaning
        and cleaning_mode != FEATURE_CLEANING_MODE_OFF
        and zone == ZONE_EDGE
        and cleaning_scope == FEATURE_CLEANING_SCOPE_INNER_ONLY
    ):
        cleaning_stats["reason"] = "edge_scope_excluded"
        if log:
            log(f"{unit_label}: feature cleaning skipped (scope={cleaning_scope})")

    engine = Engine(
        max_epochs=cfg.max_epochs,
        default_root_dir=str(run_root),
        callbacks=callbacks,
    )

    if log:
        log(f"{unit_label}: engine.fit 開始")
    engine.fit(datamodule=datamodule, model=model)
    if cleaning_callback is not None:
        if not cleaning_callback.stats:
            raise RuntimeError("feature cleaning callback did not run")
        cleaning_stats = dict(cleaning_callback.stats)
        cleaning_stats.update({
            "mode": zone_cleaning_mode,
            "scope": "per_zone" if per_zone_cleaning else cleaning_scope,
            "zone": zone,
        })
        if log:
            log(
                f"{unit_label}: feature cleaning "
                f"{cleaning_stats.get('total', 0)} -> {cleaning_stats.get('kept', 0)} "
                f"(removed={cleaning_stats.get('removed_ratio', 0.0):.2%})"
            )
    if experiment_stats_out is not None:
        experiment_stats_out.clear()
        experiment_stats_out.update({
            "feature_pool_kernel_size": pool_kernel,
            "feature_cleaning": cleaning_stats,
        })
    if log:
        log(f"{unit_label}: engine.fit 完成，開始 export")
    engine.export(model=model, export_type=ExportType.TORCH)

    candidates = list(run_root.rglob("weights/torch/model.pt"))
    if not candidates:
        candidates = list(run_root.rglob("model.pt"))
    if not candidates:
        raise RuntimeError(f"訓練後找不到 model.pt under {run_root}")
    return candidates[0]


DEFAULT_THRESHOLD = 0.35


def calibrate_threshold(ng_scores: List[float], train_max_score: float) -> float:
    """所有 unit 統一回傳 DEFAULT_THRESHOLD。

    舊版用 max(NG P10, train_max × 1.05) 但 NG 抽樣未分 zone（inner/edge
    共用同一批），導致校準不準（見 docs）。改為固定預設值，由使用者在
    模型庫頁面依誤判情況微調。

    參數保留是因為呼叫端仍傳入這兩個值，且 ng_scores 仍用於計算 metrics
    （AUROC、ng_caught_rate）給 UI 顯示。
    """
    return DEFAULT_THRESHOLD


def _compute_auroc(train_scores: List[float], ng_scores: List[float]) -> Optional[float]:
    """Mann-Whitney U 計算 AUROC，不引入 sklearn 依賴。

    AUROC = P(NG_score > train_score)，兩任意樣本中 NG 分數較高的機率。
    平手算 0.5。沒樣本時回 None。
    """
    if not train_scores or not ng_scores:
        return None
    n_t = len(train_scores)
    n_n = len(ng_scores)
    sorted_t = sorted(train_scores)
    wins = 0.0
    for s in ng_scores:
        # 嚴格小於 s 的 train 個數 = bisect_left
        # 等於 s 的 train 個數 = bisect_right - bisect_left
        lo = bisect.bisect_left(sorted_t, s)
        hi = bisect.bisect_right(sorted_t, s)
        wins += lo + 0.5 * (hi - lo)
    return round(wins / (n_t * n_n), 4)


def _auroc_grade(auroc: Optional[float]) -> str:
    """把 AUROC 對應到中文簡評。"""
    if auroc is None:
        return "n/a"
    if auroc >= 0.95:
        return "excellent"
    if auroc >= 0.85:
        return "good"
    if auroc >= 0.70:
        return "fair"
    if auroc >= 0.55:
        return "poor"
    return "fail"


def compute_unit_metrics(
    train_max: float,
    ng_scores: List[float],
    threshold: float,
    train_scores: List[float],
) -> Dict[str, Any]:
    """從 calibrate 用的數字算出 unit 品質指標。純函式，沒 I/O。

    回傳欄位：
      train_max          訓練樣本最大分數（已抽樣 100）
      train_count_eval   評估時用到的 train sample 數
      train_zero_score_* OK 評估分數為 0 的數量、比例與全為 0 警告
      ng_count           實際算到分數的 NG 樣本數
      ng_min/median/max  NG 分布
      ng_p10             NG 第 10 百分位
      threshold          最終 threshold（同 thresholds.json）
      separation         ng_median - train_max
      ng_caught_count    NG 中 score >= threshold 的個數
      ng_caught_rate     ng_caught_count / ng_count
      auroc              異常檢測 AUROC
      auroc_grade        excellent / good / fair / poor / fail / n/a
    """
    train_zero_score_count = sum(
        1 for score in train_scores
        if abs(float(score)) <= TRAIN_ZERO_SCORE_EPSILON
    )
    train_zero_score_warning = bool(train_scores) and (
        train_zero_score_count == len(train_scores)
    )
    metrics = {
        "train_max": round(float(train_max), 4),
        "ng_count": len(ng_scores),
        "threshold": round(float(threshold), 4),
        "train_count_eval": len(train_scores),
        "train_zero_score_count": train_zero_score_count,
        "train_zero_score_rate": round(
            train_zero_score_count / len(train_scores), 4
        ) if train_scores else None,
        "train_zero_score_warning": train_zero_score_warning,
    }
    if not ng_scores:
        metrics.update({
            "ng_min": None, "ng_p10": None, "ng_median": None, "ng_max": None,
            "separation": None, "ng_caught_count": 0, "ng_caught_rate": None,
            "auroc": None, "auroc_grade": "n/a",
        })
        return metrics

    sorted_scores = sorted(ng_scores)
    n = len(sorted_scores)
    p10_idx = max(0, int(n * 0.10))
    median_idx = n // 2
    ng_median = float(sorted_scores[median_idx])

    caught = sum(1 for s in ng_scores if s >= threshold)
    auroc = _compute_auroc(train_scores, ng_scores)
    metrics.update({
        "ng_min": round(float(sorted_scores[0]), 4),
        "ng_p10": round(float(sorted_scores[p10_idx]), 4),
        "ng_median": round(ng_median, 4),
        "ng_max": round(float(sorted_scores[-1]), 4),
        "separation": round(ng_median - float(train_max), 4),
        "ng_caught_count": caught,
        "ng_caught_rate": round(caught / n, 4),
        "auroc": auroc,
        "auroc_grade": _auroc_grade(auroc),
    })
    return metrics


def write_manifest(bundle_dir: Path, info: dict) -> None:
    info_full = dict(info)
    info_full["version_schema"] = 1
    (bundle_dir / "manifest.json").write_text(
        json.dumps(info_full, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_thresholds(bundle_dir: Path, thresholds: Dict[str, Dict[str, float]]) -> None:
    (bundle_dir / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_machine_config_yaml(bundle_dir: Path, machine_id: str,
                              thresholds: Dict[str, Dict[str, float]],
                              succeeded_units: Optional[Set[Tuple[str, str]]] = None,
                              image_preprocess_pipeline: Optional[List[Dict[str, Any]]] = None,
                              preprocess_after_tiling: bool = False,
                              tile_stride: int = LEGACY_TRAIN_TILE_STRIDE,
                              image_preprocess_pipelines: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> None:
    """產出 bundle 內的 inference yaml。

    若提供 succeeded_units，只寫入 inner/edge 都成功訓練的 lighting；
    None 表示寫入全部 5×2=10 組（舊行為，測試用）。
    """
    import yaml
    from capi_image_preprocess_lab import normalize_preprocess_pipeline

    model_mapping = {}
    threshold_mapping = {}
    image_preprocess_pipeline = normalize_preprocess_pipeline(image_preprocess_pipeline or [])
    image_preprocess_pipelines = {
        zone: normalize_preprocess_pipeline((image_preprocess_pipelines or {}).get(zone) or [])
        for zone in ZONES
        if zone in (image_preprocess_pipelines or {})
    }
    tile_stride = int(tile_stride or LEGACY_TRAIN_TILE_STRIDE)
    for lighting in LIGHTINGS:
        if succeeded_units is not None and not all(
            (lighting, zone) in succeeded_units for zone in ("inner", "edge")
        ):
            continue
        unit_paths = {}
        unit_thr = {}
        for zone in ("inner", "edge"):
            if succeeded_units is None or (lighting, zone) in succeeded_units:
                unit_paths[zone] = str(bundle_dir / f"{lighting}-{zone}.pt")
                unit_thr[zone] = thresholds.get(lighting, {}).get(zone, DEFAULT_THRESHOLD)
        if unit_paths:
            model_mapping[lighting] = unit_paths
        if unit_thr:
            threshold_mapping[lighting] = unit_thr

    # 維護注意：此處所有 production 值對齊 configs/capi_3f.yaml。
    # yaml 沒寫的欄位會 fallback 到 CAPIConfig.from_dict 的預設值；
    # 多數預設與 production 不同（過去曾因 patchcore_concentration_enabled、
    # edge_margin_px、otsu_bottom_crop 等漏寫導致分數異常偏低）。
    # 改 configs/capi_3f.yaml 時請同步檢查此處。

    def _emit(value) -> str:
        """yaml.dump 任意 value 並 strip 尾端換行；用於嵌進 template。"""
        return yaml.dump(
            value, allow_unicode=True, sort_keys=False, default_flow_style=False
        ).rstrip("\n")

    header_block = _emit({
        "machine_id": machine_id,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "bundle_path": str(bundle_dir),
    })
    model_mapping_block = _emit({"model_mapping": model_mapping})
    threshold_mapping_block = _emit({"threshold_mapping": threshold_mapping})
    image_preprocess_block = _emit({"image_preprocess_pipeline": image_preprocess_pipeline})
    image_preprocess_zones_block = _emit({"image_preprocess_pipelines": image_preprocess_pipelines})

    content = f"""\
# ============================================================================
# CAPI 新架構 bundle 推論設定（machine_config.yaml）
# 由 capi_train_new.write_machine_config_yaml 自動產生
#
# 所有 production 值對齊 configs/capi_3f.yaml；yaml 缺欄位會 fallback 到
# CAPIConfig.from_dict 的預設值（多數預設與 production 不同，故顯式寫出）。
# ============================================================================

{header_block}

# === 切塊與面板偵測 ===
tile_size: 512
tile_stride: {tile_stride}
edge_threshold_px: 768
otsu_offset: 5
# default=1000 會切掉 panel 底部 1000px，新架構 panel polygon 完全失準
otsu_bottom_crop: 0
enable_panel_polygon: true

# === 影像前處理（共用或 INNER/EDGE 分區；套用時機見下方）===
{image_preprocess_block}
{image_preprocess_zones_block}
preprocess_after_tiling: {str(preprocess_after_tiling).lower()}

# === 模型映射（lighting → inner/edge 模型路徑 + threshold）===
{model_mapping_block}

{threshold_mapping_block}

# === PatchCore 後處理過濾 ===
# 兩個 enabled 的 dataclass default=True，每個 tile score 會被乘 <=1.0 兩次
# → production 分數偏低；configs/capi_3f.yaml 全部關閉
patchcore_concentration_enabled: false
patchcore_concentration_min_ratio: 2.0
patchcore_concentration_penalty: 0.5
patchcore_diffuse_area_enabled: false
patchcore_diffuse_area_threshold: 0.3
patchcore_diffuse_area_penalty: 0.5

# === 邊緣衰減 ===
# px=0 → capi_inference.py:1569 整段衰減邏輯 short-circuit，衰減完全停用。
# 產線實測啟用衰減反而造成邊緣 NG 漏檢/分數偏低，故停用。
# 若要重新啟用，把 px 改 >0 並把對應邊 sides 設 true。
edge_margin_px: 0
edge_margin_sides:
  top: false
  bottom: false
  left: false
  right: false

# === OMIT 灰塵偵測 ===
dust_brightness_threshold: 40
dust_threshold_floor: 25
dust_bright_rescue_threshold: 180
dust_area_min: 5
dust_area_max: 50000
dust_extension: 3
# default 0.02 / 5.0 比 production 寬鬆很多，dust 判定會跑掉
dust_heatmap_iou_threshold: 0.007
dust_heatmap_top_percent: 0.4
dust_heatmap_metric: coverage

# === OMIT 過曝偵測 ===
omit_overexposure_mean_threshold: 82
omit_overexposure_ratio_threshold: 0.05

# === 畫異預檢 ===
# 推論前依 AOI Report 涉及畫面檢查平均亮度；低於下限或高於上限時跳過 AI 推論並回報 PCO05。
# 預設關閉，避免未校準門檻造成產線誤擋。
image_abnormal_detection_enabled: false
image_abnormal_standard_mean_lower: 68
image_abnormal_standard_mean_upper: 88
image_abnormal_wgf50500_mean_lower: 72
image_abnormal_wgf50500_mean_upper: 92
image_abnormal_g0f00000_mean_lower: 67
image_abnormal_g0f00000_mean_upper: 87
image_abnormal_r0f00000_mean_lower: 71
image_abnormal_r0f00000_mean_upper: 91
image_abnormal_w0f00000_mean_lower: 70
image_abnormal_w0f00000_mean_upper: 90
image_abnormal_b0f00000_mean_lower: 0
image_abnormal_b0f00000_mean_upper: 13

# === B0F 黑畫面亮點偵測（無 PatchCore 模型，走二值化）===
bright_spot_threshold: 200
bright_spot_min_area: 5
bright_spot_median_kernel: 21
bright_spot_diff_threshold: 10

# === 檔案過濾 ===
# B0F00000 改走亮點偵測；side_shot 前綴 S* 自動跳過（無模型訓練資料）
skip_files:
- B0F00000
side_shot_prefixes:
- SG0F00000
- SR0F00000
- SW0F00000
- SB0F00000
- SWGF50500
- SSTANDARD
- 'SPINIGBI '
# from_dict default=7（與 dataclass default=20 不一致），不寫會把 panel
# 圖片數限制砍到 7 張
max_images_per_panel: 20

# === AOI 機檢座標 attribution ===
# 新架構 helper 改走「找既存 grid tile 標屬性」，幾乎零成本，預設開啟。
# 缺此欄會走 CAPIConfig 預設 False，記錄頁的 🎯 區塊永遠不出現。
aoi_coord_inspection_enabled: true
aoi_report_path_replace_from: yuantu
aoi_report_path_replace_to: Report

# === 炸彈匹配 ===
# 新訓練 bundle 預設使用較窄距離，避免鄰近 AOI 真實缺陷被同一炸彈座標吸住。
# bomb_line_min_aspect_ratio default=3.0 太嚴，line bomb 熱力圖抓不到。
bomb_match_tolerance: 20
bomb_line_min_aspect_ratio: 1.2

# === 機種第六碼 → 產品解析度（本地工具如 diagnose_bomb 備用）===
model_resolution_map:
  B: [1366, 768]
  H: [1920, 1080]
  J: [1920, 1200]
  K: [2560, 1440]
  G: [2560, 1600]

# === Scratch classifier 後濾（DINOv2 LoRA + LogReg）===
# 預設承襲 configs/capi_3f.yaml；缺欄位會走 dataclass 預設空字串，
# 產線載入時 timm 會撞網路。
scratch_classifier_enabled: true
scratch_safety_multiplier: 1.5
scratch_bundle_path: deployment/scratch_classifier_v3.pkl
scratch_dinov2_weights_path: deployment/dinov2_vitb14.pth
scratch_dinov2_repo_path: deployment/dinov2_repo
"""

    (bundle_dir / "machine_config.yaml").write_text(content, encoding="utf-8")


def _setup_offline_env(
    backbone_cache_dir: Path,
    log: Callable,
    required_backbones: Optional[List[str]] = None,
) -> None:
    """Set torch / huggingface offline env vars + verify backbone is cached.

    anomalib's PatchCore uses `timm.create_model('wide_resnet50_2', pretrained=True)`
    which downloads from HuggingFace Hub. We redirect both TORCH_HOME and HF cache
    env vars to deployment/torch_hub_cache/.
    """
    backbone_cache_dir = Path(backbone_cache_dir).resolve()
    hf_cache = backbone_cache_dir / "huggingface"

    _configure_backbone_cache_runtime(backbone_cache_dir)
    _repair_hf_snapshot_symlinks(hf_cache, log)

    required_backbones = required_backbones or ["wide_resnet50_2-32ee1156.pth"]

    # Verify timm wide_resnet50_2 weights are present in HF cache. Older
    # deployments may also stage the raw torch hub checkpoint by filename.
    missing = []
    has_raw_cache = False
    has_hf_cache = False
    for backbone in required_backbones:
        cache_hits = [
            p for p in backbone_cache_dir.rglob(backbone)
            if p.is_file() and p.stat().st_size > 1024 * 1024
        ]
        if cache_hits:
            has_raw_cache = True
            continue

        if backbone.startswith("wide_resnet50_2"):
            timm_dirs = list(hf_cache.glob("models--timm--wide_resnet50_2*"))
            has_hf_cache = any(_has_valid_hf_snapshot_weights(d) for d in timm_dirs)
            if has_hf_cache:
                continue

        missing.append(backbone)

    if missing:
        raise RuntimeError(
            f"backbone 缺檔：未找到 {', '.join(missing)}。\n"
            f"已檢查 cache: {backbone_cache_dir}\n"
            f"請在有網路的開發機執行：\n"
            f"  HF_HOME={backbone_cache_dir} python -c \"import timm; "
            f"timm.create_model('wide_resnet50_2', pretrained=True)\"\n"
            f"然後把整個 {backbone_cache_dir} 目錄 FTP 上傳到 production。"
        )

    if has_raw_cache and not has_hf_cache:
        _enable_timm_old_cache()
    _patch_hf_hub_local_files_only()

    _preflight_timm_backbone(log, cache_dir=hf_cache if has_hf_cache else None)
    log(f"✓ backbone cache 已就緒: {hf_cache}")


def _configure_backbone_cache_runtime(backbone_cache_dir: Path) -> None:
    """Point torch/timm/HF runtime state at the staged offline cache."""
    backbone_cache_dir = Path(backbone_cache_dir).resolve()
    hf_cache = backbone_cache_dir / "huggingface"

    # Redirect both torch hub and HuggingFace cache to deployment dir.
    os.environ["TORCH_HOME"] = str(backbone_cache_dir)
    os.environ["HF_HOME"] = str(backbone_cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hf_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
    os.environ["HF_XET_CACHE"] = str(backbone_cache_dir / "xet")
    # Force offline mode (no network calls during training).
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["TRUST_REMOTE_CODE"] = "1"

    # capi_server imports anomalib for inference before server_config is loaded.
    # If that already imported huggingface_hub, its constants were frozen from the
    # process environment at import time. Patch them so later timm calls still use
    # this staged cache and offline mode.
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HOME = str(backbone_cache_dir)
        hf_constants.HF_HUB_CACHE = str(hf_cache)
        hf_constants.HUGGINGFACE_HUB_CACHE = str(hf_cache)
        hf_constants.HF_XET_CACHE = str(backbone_cache_dir / "xet")
        hf_constants.HF_HUB_OFFLINE = True
    except Exception:
        pass


def _enable_timm_old_cache() -> None:
    """Allow timm to use TORCH_HOME/hub/checkpoints/*.pth fallback weights."""
    os.environ["TIMM_USE_OLD_CACHE"] = "1"
    try:
        import timm.models._builder as timm_builder

        timm_builder._USE_OLD_CACHE = True
    except Exception:
        pass


def _make_local_only_hf_download(download_fn: Callable) -> Callable:
    """Wrap hf_hub_download so timm cannot open an HTTP client during training."""
    if getattr(download_fn, "_capi_forces_local_files_only", False):
        return download_fn

    @wraps(download_fn)
    def _local_only_download(*args, **kwargs):
        kwargs["local_files_only"] = True
        return download_fn(*args, **kwargs)

    _local_only_download._capi_forces_local_files_only = True
    return _local_only_download


def _patch_hf_hub_local_files_only() -> None:
    """Force HF downloads used by timm to resolve only from local cache."""
    try:
        import huggingface_hub

        huggingface_hub.hf_hub_download = _make_local_only_hf_download(
            huggingface_hub.hf_hub_download
        )
    except Exception:
        pass

    try:
        import huggingface_hub.file_download as hf_file_download

        hf_file_download.hf_hub_download = _make_local_only_hf_download(
            hf_file_download.hf_hub_download
        )
    except Exception:
        pass

    try:
        import timm.models._hub as timm_hub

        timm_hub.hf_hub_download = _make_local_only_hf_download(
            timm_hub.hf_hub_download
        )
    except Exception:
        pass


def _has_valid_hf_snapshot_weights(model_dir: Path) -> bool:
    snapshot_root = model_dir / "snapshots"
    if not snapshot_root.exists():
        return False

    for snapshot_file in snapshot_root.rglob("*"):
        if snapshot_file.suffix not in {".safetensors", ".bin", ".pth"}:
            continue
        try:
            if (
                (snapshot_file.is_file() or snapshot_file.is_symlink())
                and snapshot_file.stat().st_size > 1024 * 1024
            ):
                return True
        except OSError:
            continue
    return False


def _repair_hf_snapshot_symlinks(hf_cache: Path, log: Callable) -> None:
    """Repair HF cache snapshots copied by tools that do not preserve symlinks.

    HuggingFace snapshots usually store files as symlinks into `blobs/`. Some
    FTP/copy workflows turn those symlinks into zero-byte regular files. timm
    then falls through to HuggingFace Hub loading and can fail with opaque HTTP
    client errors during every PatchCore unit. Restore empty/broken snapshot
    weight files from the matching large blob when possible.
    """
    if not hf_cache.exists():
        return

    for model_dir in hf_cache.glob("models--timm--*"):
        blob_candidates = sorted(
            [
                p for p in (model_dir / "blobs").glob("*")
                if p.is_file() and p.stat().st_size > 1024 * 1024
            ],
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        if not blob_candidates:
            continue

        for snapshot_file in (model_dir / "snapshots").rglob("*"):
            if snapshot_file.suffix not in {".safetensors", ".bin", ".pth"}:
                continue
            if _is_valid_weight_file(snapshot_file):
                continue

            blob = _select_hf_blob_for_snapshot(snapshot_file, blob_candidates)
            if blob is None:
                log(f"  ! 無法自動修復 HF cache 權重檔: {snapshot_file}")
                continue

            if snapshot_file.is_symlink():
                snapshot_file.unlink()
            snapshot_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(blob, snapshot_file)
            log(f"  ✓ 修復 HF cache 權重檔: {snapshot_file}")


def _is_valid_weight_file(path: Path) -> bool:
    try:
        return (
            (path.is_file() or path.is_symlink())
            and path.stat().st_size > 1024 * 1024
        )
    except OSError:
        return False


def _select_hf_blob_for_snapshot(
    snapshot_file: Path,
    blob_candidates: List[Path],
) -> Optional[Path]:
    if snapshot_file.is_symlink():
        try:
            linked_blob = (snapshot_file.parent / snapshot_file.readlink()).resolve()
            for blob in blob_candidates:
                if blob.resolve() == linked_blob:
                    return blob
        except OSError:
            pass

    if len(blob_candidates) == 1:
        return blob_candidates[0]

    # timm weight repositories normally have one large weight blob. If a cache
    # contains multiple large blobs, use the largest one as a conservative
    # recovery path instead of leaving an empty snapshot pointer in place.
    return blob_candidates[0]


def _preflight_timm_backbone(log: Callable, cache_dir: Optional[Path] = None) -> None:
    """Verify timm can load the PatchCore backbone from local cache only."""
    try:
        import timm

        kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
        timm.create_model(
            "wide_resnet50_2",
            pretrained=True,
            features_only=True,
            exportable=True,
            **kwargs,
        )
    except Exception as exc:
        raise RuntimeError(
            "backbone cache 無法離線載入 wide_resnet50_2。"
            "請確認 huggingface snapshot 內的 model.safetensors 不是 0 byte，"
            "且 blobs 目錄已完整上傳。原始錯誤: "
            f"{exc}"
        ) from exc


def _calibrate_from_model(
    model_pt: Path, train_paths: List[Path], ng_paths: List[Path]
) -> Tuple[float, List[float], List[float]]:
    """單次載入模型，回傳 (train_max_score, train_scores, ng_scores)。

    train_scores 是抽樣 100 張訓練圖跑分的完整列表（用於算 AUROC、分布指標）；
    train_max 是其中最大值（保留供 calibrate_threshold 使用）。
    """
    from anomalib.deploy import TorchInferencer
    inferencer = TorchInferencer(path=str(model_pt))

    sample = random.sample(train_paths, min(100, len(train_paths)))
    train_scores: List[float] = []
    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            continue
        result = inferencer.predict(img)
        train_scores.append(float(getattr(result, "pred_score", 0.0)))
    train_max = max(train_scores) if train_scores else 0.0

    ng_scores = []
    for p in ng_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        result = inferencer.predict(img)
        ng_scores.append(float(getattr(result, "pred_score", 0.0)))

    return train_max, train_scores, ng_scores


def train_single_submodel(
    db: TrainingDB,
    job_id: str,
    lighting: str,
    zone: str,
    cfg: TrainingConfig,
    output_pt_path: Path,
    gpu_lock=None,
    log: Callable[[str], None] = print,
    cancel_event=None,
    unit_prefix: str = "",
) -> Dict:
    """訓練單一 (lighting, zone) unit。

    回傳 dict 包含：
      - threshold: float (永遠 = DEFAULT_THRESHOLD = 0.35，calibrate 寫死)
      - metrics: dict (compute_unit_metrics 結果)
      - tile_count: int (訓練用 tile 數)
      - ng_count: int (NG 數)
      - ng_used: "zone" | "fallback" | "none"
      - used_tile_ids: list[int] (該次訓練實際送進的 tile_pool.id)
      - elapsed_seconds: int
      - size_bytes: int (.pt 檔大小)

    output_pt_path 會被原子覆蓋（同目錄 .pt.tmp → os.replace）。失敗時不動到原檔。
    """
    from contextlib import nullcontext
    import os

    # train_single_submodel 同時被 wizard run_training_pipeline 與子模型
    # retrain worker（同 process thread）呼叫，後者不會先設離線 env；
    # 在此 ensure，避免 PatchCore 建立時 timm 走線上 huggingface_hub
    # 撞到 server process 殘留的已關閉 httpx client。內部冪等。
    _setup_offline_env(cfg.backbone_cache_dir, log, cfg.required_backbones)

    gpu_ctx = gpu_lock if gpu_lock is not None else nullcontext()
    unit_label = f"{lighting}-{zone}"

    train_tiles = db.list_tile_pool(job_id, lighting=lighting, zone=zone,
                                    source="ok", decision="accept")
    ng_all = db.list_tile_pool(job_id, lighting=lighting,
                               source="ng", decision="accept")
    ng_for_zone = [t for t in ng_all if t.get("zone") in (zone, None)]
    if not ng_all:
        log(f"{unit_prefix}{unit_label}: 無可用 NG，僅以 OK tile 訓練")
        ng_tiles = []
        ng_used = "none"
    elif len(ng_for_zone) < MIN_NG_PER_ZONE:
        log(f"{unit_prefix}{unit_label}: zone NG 僅 {len(ng_for_zone)} (<{MIN_NG_PER_ZONE})，"
            f"退回全部 NG ({len(ng_all)})")
        ng_tiles = ng_all
        ng_used = "fallback"
    else:
        ng_tiles = ng_for_zone
        ng_used = "zone"

    if len(train_tiles) < MIN_TRAIN_TILES:
        raise RuntimeError(
            f"{unit_label}: tile 不足 ({len(train_tiles)} < {MIN_TRAIN_TILES})"
        )

    used_tile_ids = sorted(int(t["id"]) for t in train_tiles)
    unit_start = time.monotonic()

    with gpu_ctx:
        staging = Path(".tmp/training_staging") / job_id / unit_label
        run_root = Path(".tmp/training_runs") / job_id / unit_label
        try:
            staged_train_paths = stage_dataset(
                staging,
                [Path(t["source_path"]) for t in train_tiles],
                [Path(t["source_path"]) for t in ng_tiles],
            )
            trace_sources: Dict[str, Dict[str, Any]] = {}
            for tile, staged_path in zip(train_tiles, staged_train_paths):
                trace_sources[str(staged_path)] = {
                    "tile_pool_id": int(tile["id"]),
                    "source_path": str(Path(tile["source_path"]).resolve()),
                    "panel_path": tile.get("panel_path"),
                    "tile_index": tile.get("tile_index"),
                    "tile_x": tile.get("tile_x"),
                    "tile_y": tile.get("tile_y"),
                    "tile_width": tile.get("tile_width"),
                    "tile_height": tile.get("tile_height"),
                }
            experiment_stats: Dict[str, Any] = {}
            trace_kwargs = {}
            if feature_cleaning_config_for_zone(cfg, zone)["mode"] != FEATURE_CLEANING_MODE_OFF:
                trace_kwargs["trace_sources"] = trace_sources
            model_pt = train_one_patchcore(
                staging,
                run_root,
                unit_label,
                cfg,
                log=log,
                experiment_stats_out=experiment_stats,
                **trace_kwargs,
            )

            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("training cancelled by user")

            train_max, train_scores, ng_scores = _calibrate_from_model(
                model_pt,
                [Path(t["source_path"]) for t in train_tiles],
                [Path(t["source_path"]) for t in ng_tiles],
            )
            threshold = calibrate_threshold(ng_scores, train_max)

            metrics = compute_unit_metrics(
                train_max, ng_scores, threshold, train_scores=train_scores,
            )
            metrics["train_count"] = len(train_tiles)
            metrics["ng_used"] = ng_used
            metrics["feature_pool_kernel_size"] = experiment_stats.get(
                "feature_pool_kernel_size", cfg.feature_pool_kernel_size,
            )
            fallback_cleaning = feature_cleaning_config_for_zone(cfg, zone)
            metrics["feature_cleaning"] = experiment_stats.get("feature_cleaning", {
                **fallback_cleaning,
                "scope": "per_zone" if cfg.feature_cleaning_by_zone else cfg.feature_cleaning_scope,
                "zone": zone,
                "applied": False,
                "reason": "stats_unavailable",
            })
            patch_trace = metrics["feature_cleaning"].pop("patch_trace", [])
            removed_patch_trace = metrics["feature_cleaning"].pop(
                "removed_patch_trace", []
            )
            cleaning_patch_trace = patch_trace or removed_patch_trace
            elapsed = time.monotonic() - unit_start
            metrics["elapsed_seconds"] = int(elapsed)

            output_pt_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = output_pt_path.with_suffix(output_pt_path.suffix + ".tmp")
            shutil.copy2(model_pt, tmp_path)
            os.replace(tmp_path, output_pt_path)
            size = output_pt_path.stat().st_size

            if cleaning_patch_trace:
                try:
                    report_rel = Path("feature_cleaning_reports") / f"{unit_label}.json"
                    report_path = output_pt_path.parent / report_rel
                    assets_dir = (
                        output_pt_path.parent
                        / "feature_cleaning_reports"
                        / "assets"
                        / unit_label
                    )
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.rmtree(assets_dir, ignore_errors=True)
                    assets_dir.mkdir(parents=True, exist_ok=True)
                    report_tiles = []
                    affected_indices = [
                        index
                        for index, trace_item in enumerate(cleaning_patch_trace)
                        if int(trace_item.get("removed_count") or 0) > 0
                    ]
                    visual_indices = set(sorted(
                        affected_indices,
                        key=lambda item_index: int(
                            cleaning_patch_trace[item_index].get("removed_count") or 0
                        ),
                        reverse=True,
                    )[:12])
                    for index, trace_item in enumerate(cleaning_patch_trace):
                        source_path = Path(trace_item["source_path"])
                        report_item = {
                            key: value
                            for key, value in trace_item.items()
                            if key != "source_path"
                        }
                        report_item["source_name"] = source_path.name
                        if index in visual_indices:
                            asset_path = assets_dir / f"{index:04d}_{source_path.name}"
                            asset_tmp = asset_path.with_suffix(asset_path.suffix + ".tmp")
                            shutil.copy2(source_path, asset_tmp)
                            os.replace(asset_tmp, asset_path)
                            report_item["asset_path"] = asset_path.relative_to(
                                output_pt_path.parent
                            ).as_posix()
                        report_tiles.append(report_item)

                    report_tmp = report_path.with_suffix(report_path.suffix + ".tmp")
                    report_payload = {
                        "schema_version": 2,
                        "unit_label": unit_label,
                        "k": metrics["feature_cleaning"].get("k"),
                        "keep_ratio": metrics["feature_cleaning"].get("keep_ratio"),
                        "center_size": metrics["feature_cleaning"].get("center_size"),
                        "cleaning_candidates": metrics["feature_cleaning"].get(
                            "cleaning_candidates"
                        ),
                        "threshold": metrics["feature_cleaning"].get("threshold"),
                        "removed": metrics["feature_cleaning"].get("removed", 0),
                        "distance_removed": metrics["feature_cleaning"].get(
                            "distance_removed", 0
                        ),
                        "coreset_selected": metrics["feature_cleaning"].get(
                            "coreset_selected", 0
                        ),
                        "reason_legend": metrics["feature_cleaning"].get(
                            "trace_reason_legend"
                        ) or {
                            "0": "kept_normal",
                            "1": "removed_distance_outlier",
                            "2": "kept_overlap_disagreement",
                            "3": "protected_tile_boundary",
                            "5": "missing_coordinate_metadata",
                            "6": "protected_outside_cleaning_scope",
                        },
                        "tiles": report_tiles,
                    }
                    report_tmp.write_text(
                        json.dumps(report_payload, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    os.replace(report_tmp, report_path)
                    metrics["feature_cleaning"]["report_path"] = report_rel.as_posix()
                    metrics["feature_cleaning"]["affected_tiles"] = len(affected_indices)
                    metrics["feature_cleaning"]["traced_tiles"] = len(report_tiles)
                except Exception as exc:
                    logger.warning(
                        "feature cleaning visualization report failed for %s: %s",
                        unit_label,
                        exc,
                        exc_info=True,
                    )
                    if log:
                        log(f"{unit_label}: feature cleaning 可視化報告寫入失敗: {exc}")

            return {
                "threshold": round(threshold, 4),
                "metrics": metrics,
                "tile_count": len(train_tiles),
                "ng_count": len(ng_tiles),
                "ng_used": ng_used,
                "used_tile_ids": used_tile_ids,
                "elapsed_seconds": int(elapsed),
                "size_bytes": size,
            }
        finally:
            shutil.rmtree(run_root, ignore_errors=True)
            shutil.rmtree(staging, ignore_errors=True)
            tmp_leftover = output_pt_path.with_suffix(output_pt_path.suffix + ".tmp")
            if tmp_leftover.exists():
                try:
                    tmp_leftover.unlink()
                except OSError:
                    pass
            report_tmp = (
                output_pt_path.parent
                / "feature_cleaning_reports"
                / f"{unit_label}.json.tmp"
            )
            if report_tmp.exists():
                try:
                    report_tmp.unlink()
                except OSError:
                    pass
            report_assets = (
                output_pt_path.parent
                / "feature_cleaning_reports"
                / "assets"
                / unit_label
            )
            if report_assets.exists():
                for asset_tmp in report_assets.glob("*.tmp"):
                    try:
                        asset_tmp.unlink()
                    except OSError:
                        pass
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def run_training_pipeline(
    job_id: str,
    cfg: TrainingConfig,
    db: TrainingDB,
    gpu_lock=None,
    log: Callable[[str], None] = print,
    cancel_event=None,
) -> Path:
    """執行 10 unit 訓練，輸出 bundle 目錄。

    gpu_lock: 同 process 多 thread 共享 GPU 時用於序列化的 lock；subprocess
        模式下傳 None 即可（VRAM 已透過 set_per_process_memory_fraction 隔離）。
    cancel_event: 任意提供 .is_set() 的物件（threading.Event 或 file-flag wrapper）。
    """
    # 1. 環境檢查
    _setup_offline_env(cfg.backbone_cache_dir, log, cfg.required_backbones)

    # 路徑格式：<machine_id>-<YYYYMMDD_HHMMSS>。
    # job_id 已存在 manifest.json.trained_with_job_id 與 DB model_bundles.job_id，
    # 不再放入路徑（避免 machine_id 在路徑與 job_id 中重複出現）。
    bundle_dir = cfg.output_root / f"{cfg.machine_id}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    thresholds: Dict[str, Dict[str, float]] = {l: {} for l in LIGHTINGS}
    tiles_per_unit: Dict[str, Dict[str, int]] = {}
    model_files: Dict[str, Dict] = {}
    unit_metrics: Dict[str, Dict] = {}
    success_units = 0
    succeeded_units: Set[Tuple[str, str]] = set()
    completed_durations: List[float] = []  # 已完成 unit 的耗時，用來算 ETA
    pipeline_start = time.monotonic()

    def _eta_text() -> str:
        if not completed_durations:
            return ""
        avg = sum(completed_durations) / len(completed_durations)
        remaining_units = len(TRAINING_UNITS) - len(completed_durations)
        if remaining_units <= 0:
            return ""
        eta_s = avg * remaining_units
        m, s = divmod(int(eta_s), 60)
        return f"預估剩 {m}m{s:02d}s（平均 {int(avg)}s/unit）"

    for idx, (lighting, zone) in enumerate(TRAINING_UNITS, 1):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("training cancelled by user")

        unit_label = f"{lighting}-{zone}"
        log(f"[{idx}/10] {unit_label}: 載 tile")
        unit_start = time.monotonic()

        train_tiles = db.list_tile_pool(job_id, lighting=lighting, zone=zone,
                                        source="ok", decision="accept")
        if len(train_tiles) < MIN_TRAIN_TILES:
            log(f"[{idx}/10] {unit_label}: 跳過：tile 不足 ({len(train_tiles)} < {MIN_TRAIN_TILES})")
            continue

        try:
            output_pt = bundle_dir / f"{unit_label}.pt"
            result = train_single_submodel(
                db=db, job_id=job_id, lighting=lighting, zone=zone,
                cfg=cfg, output_pt_path=output_pt,
                gpu_lock=gpu_lock, log=log, cancel_event=cancel_event,
                unit_prefix=f"[{idx}/10] ",
            )

            thresholds[lighting][zone] = result["threshold"]
            tiles_per_unit[unit_label] = {"train": result["tile_count"], "ng": result["ng_count"]}
            model_files[unit_label] = {"path": output_pt.name, "size_bytes": result["size_bytes"]}

            metrics = result["metrics"]
            metrics["used_tile_ids"] = result["used_tile_ids"]
            unit_metrics[unit_label] = metrics

            success_units += 1
            succeeded_units.add((lighting, zone))
            completed_durations.append(result["elapsed_seconds"])
            eta = _eta_text()
            caught = metrics.get("ng_caught_count", 0)
            ng_n = metrics.get("ng_count", 0)
            auroc = metrics.get("auroc")
            auroc_str = f", AUROC={auroc:.3f}({metrics.get('auroc_grade','')})" if auroc is not None else ""
            log(
                f"[{idx}/10] {unit_label}: ✓ done | {result['elapsed_seconds']}s, "
                f"threshold={result['threshold']:.4f}, size={result['size_bytes']/1e6:.1f}MB, "
                f"ng_caught={caught}/{ng_n}{auroc_str}"
                + (f" | {eta}" if eta else "")
            )
        except Exception as e:
            completed_durations.append(time.monotonic() - unit_start)
            log(f"[{idx}/10] {unit_label}: ✗ 訓練失敗: {e}")
            for line in traceback.format_exc().rstrip().splitlines()[-8:]:
                log(f"  {line}")
            # 不增加 success_units，繼續下一個 unit

    if success_units != len(TRAINING_UNITS):
        missing = [
            f"{lighting}-{zone}"
            for lighting, zone in TRAINING_UNITS
            if (lighting, zone) not in succeeded_units
        ]
        shutil.rmtree(bundle_dir, ignore_errors=True)
        raise RuntimeError(
            f"成功 unit 數 {success_units}/{len(TRAINING_UNITS)}，缺少: {', '.join(missing)}"
        )

    auroc_values = [u["auroc"] for u in unit_metrics.values() if u.get("auroc") is not None]
    overall_auroc = round(sum(auroc_values) / len(auroc_values), 4) if auroc_values else None
    overall_auroc_grade = _auroc_grade(overall_auroc)

    write_thresholds(bundle_dir, thresholds)
    write_machine_config_yaml(
        bundle_dir,
        cfg.machine_id,
        thresholds,
        succeeded_units=succeeded_units,
        image_preprocess_pipeline=cfg.image_preprocess_pipeline,
        image_preprocess_pipelines=cfg.image_preprocess_pipelines,
        preprocess_after_tiling=cfg.preprocess_after_tiling,
        tile_stride=cfg.tile_stride,
    )
    write_manifest(bundle_dir, {
        "machine_id": cfg.machine_id,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "trained_with_job_id": job_id,
        "experimental_training": bool(
            cfg.feature_pool_kernel_size != PATCHCORE_FEATURE_POOL_KERNEL_DEFAULT
            or cfg.feature_cleaning_mode != FEATURE_CLEANING_MODE_OFF
            or any(
                item.get("mode") != FEATURE_CLEANING_MODE_OFF
                for item in normalize_feature_cleaning_by_zone(
                    cfg.feature_cleaning_by_zone
                ).values()
            )
        ),
        "panel_count": len(cfg.panel_paths),
        "panel_glass_ids": [p.name for p in cfg.panel_paths],
        "training_data_source": cfg.training_data_source,
        "edge_threshold_px": 768,
        "tile_stride": cfg.tile_stride,
        "preprocess_after_tiling": cfg.preprocess_after_tiling,
        "patchcore_params": {
            "batch_size": cfg.batch_size,
            "image_size": list(cfg.image_size),
            "coreset_ratio": cfg.coreset_ratio,
            "max_epochs": cfg.max_epochs,
            "precision": cfg.precision,
            "feature_layers": cfg.feature_layers,
            "feature_pool_kernel_size": cfg.feature_pool_kernel_size,
            "feature_cleaning_mode": cfg.feature_cleaning_mode,
            "feature_cleaning_scope": cfg.feature_cleaning_scope,
            "feature_cleaning_k": FEATURE_CLEANING_K,
            "feature_cleaning_keep_ratio": cfg.feature_cleaning_keep_ratio,
            "feature_cleaning_center_size": cfg.feature_cleaning_center_size,
            "feature_cleaning_seed": FEATURE_CLEANING_SEED,
            "feature_cleaning_reference_size": FEATURE_CLEANING_REFERENCE_SIZE,
            "feature_cleaning_query_chunk": FEATURE_CLEANING_QUERY_CHUNK,
            "feature_cleaning_adaptive_mad_z": FEATURE_CLEANING_ADAPTIVE_MAD_Z,
            "feature_cleaning_by_zone": normalize_feature_cleaning_by_zone(
                cfg.feature_cleaning_by_zone
            ),
        },
        "image_preprocess_pipeline": cfg.image_preprocess_pipeline,
        "image_preprocess_pipelines": cfg.image_preprocess_pipelines,
        "tiles_per_unit": tiles_per_unit,
        "model_files": model_files,
        "unit_metrics": unit_metrics,
        "overall_auroc": overall_auroc,
        "overall_auroc_grade": overall_auroc_grade,
        "success_units": success_units,
    })
    return bundle_dir
