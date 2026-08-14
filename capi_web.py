"""
CAPI AI Web 查閱介面

提供推論結果的 Web 查閱功能，包含：
- 推論記錄列表 (首頁)
- 記錄詳情 (含熱力圖)
- 搜尋功能
- 統計 API
- 熱力圖靜態檔案服務

使用 Python 內建 http.server + 簡單路由，無需額外依賴。
"""

import os
import csv
import socket
import subprocess
import sys
import tempfile
import gzip
import json
import hashlib
import html
import http.client
import ipaddress
import inspect
import secrets
import sqlite3
import time
import urllib.parse
import mimetypes
from contextlib import contextmanager
from datetime import datetime, timedelta
from http.cookies import SimpleCookie
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
import threading
import logging
from jinja2 import Environment, FileSystemLoader
import shutil
from capi_dataset_export import (
    DatasetExporter, JobSummary,
    JOB_STATE_IDLE, JOB_STATE_RUNNING, JOB_STATE_COMPLETED,
    JOB_STATE_FAILED, JOB_STATE_CANCELLED,
    read_manifest, write_manifest, delete_sample, relabel_sample,
    get_valid_labels, LABEL_ZH, list_job_dirs,
    parse_datastr_per_prefix, extract_prefix, resolve_source_path,
    crop_patchcore_tile,
)
from capi_scratch_batch import (
    ScratchBatchRunner, compute_summary as scratch_batch_summary,
    POSITIVE_LABEL as SCRATCH_POSITIVE_LABEL,
    STATE_RUNNING as SCRATCH_STATE_RUNNING,
)
from capi_version import get_version_info, read_changelog
from capi_image_naming import canonical_image_prefix, image_prefix_display_labels, source_image_prefix
from capi_image_orientation import read_detection_image

logger = logging.getLogger("capi.web")

_MES_REVIEW_LIGHTINGS = {
    "G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD",
}

CENTRAL_ACCOUNT_LOCATION_PARAM = "central_account_location"
CENTRAL_ACCOUNT_DEFAULT_IPS = {
    "MOD1": "10.172.25.105",
    "MOD2": "10.174.37.81",
}
CENTRAL_ACCOUNT_LOCATION_DESCRIPTION = (
    "中央帳號中心位置；依廠區帶入預設 IP，IP 可依現場需求修改"
)
CENTRAL_ACCOUNT_AUTH_HEADER = "X-CAPI-Central-Auth"
CENTRAL_ACCOUNT_AUTH_PATH = "/api/settings/central-auth"
CENTRAL_ACCOUNT_AUTH_TIMEOUT_SECONDS = 3.0
CENTRAL_UPDATE_AUTH_HEADER = "X-CAPI-Central-Update"
CENTRAL_UPDATE_USER_HEADER = "X-CAPI-Central-User"
CENTRAL_UPDATE_APPLY_PATH = "/api/update/apply-central"
CENTRAL_UPDATE_TIMEOUT_SECONDS = 8.0


def _default_central_account_location(
    server_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    mes_report = (server_config or {}).get("mes_report") or {}
    facility = str(mes_report.get("facility") or "MOD2").strip().upper()
    if facility not in CENTRAL_ACCOUNT_DEFAULT_IPS:
        facility = "MOD2"
    return {
        "facility": facility,
        "ip": CENTRAL_ACCOUNT_DEFAULT_IPS[facility],
    }


def _normalize_central_account_location(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("中心位置格式錯誤")

    facility = str(value.get("facility") or "").strip().upper()
    if facility not in CENTRAL_ACCOUNT_DEFAULT_IPS:
        raise ValueError("中心廠區只能選擇 MOD1 或 MOD2")

    ip_text = str(value.get("ip") or "").strip()
    try:
        address = ipaddress.ip_address(ip_text)
    except ValueError as exc:
        raise ValueError("請輸入有效的 IPv4 位址") from exc
    if address.version != 4:
        raise ValueError("請輸入有效的 IPv4 位址")

    return {
        "facility": facility,
        "ip": str(address),
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coord_debug_point_payload(
    *,
    map_x: int,
    map_y: int,
    map_width: int,
    map_height: int,
    tile_x: int,
    tile_y: int,
    tile_width: int,
    tile_height: int,
) -> Dict[str, List[int]]:
    """Convert an anomaly-map point into tile-local and full-image coordinates."""
    local_x = int(round(
        int(map_x) * max(int(tile_width) - 1, 0) / max(int(map_width) - 1, 1)
    ))
    local_y = int(round(
        int(map_y) * max(int(tile_height) - 1, 0) / max(int(map_height) - 1, 1)
    ))
    return {
        "map_coord": [int(map_x), int(map_y)],
        "tile_coord": [local_x, local_y],
        "image_coord": [int(tile_x) + local_x, int(tile_y) + local_y],
    }


def _coord_debug_region_payload(
    region_details: List[Dict[str, Any]],
    anomaly_map: Any,
    *,
    tile_score: float,
    tile_info: Any,
) -> List[Dict[str, Any]]:
    """Serialize every Top-% connected region with an operator-facing explanation."""
    if anomaly_map is None:
        return []

    import numpy as np

    anomaly = np.asarray(anomaly_map, dtype=np.float32)
    if anomaly.ndim < 2 or anomaly.size == 0:
        return []
    map_height, map_width = anomaly.shape[:2]
    global_peak = max(float(np.max(anomaly)), 0.0)
    payload = []
    for index, region in enumerate(region_details or [], start=1):
        peak_yx = region.get("peak_yx") or (0, 0)
        peak_y, peak_x = int(peak_yx[0]), int(peak_yx[1])
        peak_value = float(region.get("max_score") or 0.0)
        relative = peak_value / global_peak if global_peak > 0 else 0.0
        estimated_score = float(tile_score) * relative
        is_dust = bool(region.get("is_dust", False))
        peak_in_dust = bool(region.get("peak_in_dust", False))
        coverage = float(region.get("coverage") or 0.0)
        point = _coord_debug_point_payload(
            map_x=peak_x,
            map_y=peak_y,
            map_width=map_width,
            map_height=map_height,
            tile_x=int(getattr(tile_info, "x", 0)),
            tile_y=int(getattr(tile_info, "y", 0)),
            tile_width=int(getattr(tile_info, "width", map_width)),
            tile_height=int(getattr(tile_info, "height", map_height)),
        )
        if is_dust:
            verdict_zh = "灰塵／氣泡熱區"
            reason_zh = (
                "此熱區與灰塵遮罩重疊，正式灰塵流程會把它當成非產品缺陷。"
            )
        else:
            verdict_zh = "候選真缺陷"
            reason_zh = (
                "此熱區未被灰塵遮罩充分覆蓋，正式灰塵流程會保留為 NG 證據。"
            )
        payload.append({
            "rank": index,
            "label_id": int(region.get("label_id") or index),
            **point,
            "area": int(region.get("area") or 0),
            "dust_overlap": int(region.get("dust_overlap") or 0),
            "coverage": round(coverage, 6),
            "peak_value": round(peak_value, 6),
            "relative_to_global": round(relative, 6),
            "estimated_score": round(estimated_score, 6),
            "is_dust": is_dust,
            "peak_in_dust": peak_in_dust,
            "dust_sub_peak_rescue": bool(
                region.get("dust_sub_peak_rescue", False)
            ),
            "verdict_zh": verdict_zh,
            "reason_zh": reason_zh,
        })
    payload.sort(key=lambda item: item["peak_value"], reverse=True)
    for rank, item in enumerate(payload, start=1):
        item["rank"] = rank
    return payload


def _coord_debug_two_stage_payload(
    features: List[Dict[str, Any]],
    detail_text: str,
    *,
    tile_info: Any,
) -> Dict[str, Any]:
    """Translate two-stage feature evidence and counters into stable JSON."""
    support_labels = {
        "original_core": "原始 Top % 核心",
        "dust_rerank": "排除灰塵後重新排名",
        "local_score": "局部等效分數達門檻",
    }
    dust_reason_labels = {
        "feature_overlap": "特徵本身與灰塵重疊",
        "zone_dominated": "所在熱區主要由灰塵／氣泡構成",
        "clean": "未被灰塵覆蓋",
    }
    serialized = []
    for index, feature in enumerate(features or [], start=1):
        abs_pos = feature.get("abs_pos") or (0, 0)
        local_x, local_y = int(abs_pos[0]), int(abs_pos[1])
        is_dust = bool(feature.get("is_dust", False))
        support = str(feature.get("support_source") or "")
        dust_reason = str(feature.get("dust_reason") or "")
        serialized.append({
            "rank": index,
            "tile_coord": [local_x, local_y],
            "image_coord": [
                int(getattr(tile_info, "x", 0)) + local_x,
                int(getattr(tile_info, "y", 0)) + local_y,
            ],
            "type": str(feature.get("type") or ""),
            "type_zh": "暗點" if feature.get("type") == "dark" else "亮點",
            "area": int(feature.get("area") or 0),
            "dust_ratio": round(float(feature.get("dust_ratio") or 0.0), 6),
            "zone_dust_coverage": round(
                float(feature.get("zone_dust_cov") or 0.0), 6
            ),
            "local_peak": round(float(feature.get("local_peak") or 0.0), 6),
            "estimated_score": round(
                float(feature.get("local_equiv_score") or 0.0), 6
            ),
            "hot_core_supported": bool(
                feature.get("hot_core_supported", False)
            ),
            "dust_rerank_supported": bool(
                feature.get("dust_rerank_supported", False)
            ),
            "local_score_supported": bool(
                feature.get("local_score_supported", False)
            ),
            "support_source": support,
            "support_source_zh": support_labels.get(support, "未取得判定資格"),
            "is_dust": is_dust,
            "dust_reason": dust_reason,
            "dust_reason_zh": dust_reason_labels.get(dust_reason, "未記錄"),
            "verdict_zh": "灰塵／氣泡特徵" if is_dust else "候選真缺陷特徵",
        })

    counters = {}
    counter_labels = {
        "ignored_border": "位於切塊邊界，未採用",
        "ignored_outside_hot_core": (
            "原圖找到特徵，但未進入 Top % 核心、灰塵重排核心，"
            "且局部等效分數也未達門檻"
        ),
        "reranked_after_dust": "排除灰塵／氣泡後重新排名而救回",
        "local_score_rescued": "靠局部等效分數達門檻而救回",
    }
    for key, label_zh in counter_labels.items():
        match = re.search(rf"\b{re.escape(key)}=(\d+)", str(detail_text or ""))
        count = int(match.group(1)) if match else 0
        counters[key] = {"count": count, "label_zh": label_zh}

    return {
        "ran": "TWO_STAGE:" in str(detail_text or ""),
        "detail_raw": str(detail_text or ""),
        "features": serialized,
        "counters": counters,
    }


def _get_host_identity() -> str:
    try:
        return socket.gethostname().strip() or "unknown-host"
    except Exception:
        return "unknown-host"


_HARDWARE_STATUS_CACHE_SECONDS = 30.0
_hardware_status_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_hardware_status_lock = threading.Lock()


def _metric_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _read_gpu_status() -> Dict[str, Any]:
    """Read GPU 0 metrics from the NVIDIA driver without importing torch."""
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False}

    if completed.returncode != 0 or not completed.stdout.strip():
        return {"available": False}

    try:
        row = next(csv.reader(completed.stdout.splitlines()))
    except (StopIteration, csv.Error):
        return {"available": False}
    if len(row) < 5:
        return {"available": False}

    vram_used_mib = _metric_float(row[2])
    vram_total_mib = _metric_float(row[3])
    return {
        "available": True,
        "name": row[0].strip(),
        "utilization_percent": _metric_float(row[1]),
        "vram_used_gb": round(vram_used_mib / 1024.0, 2) if vram_used_mib is not None else None,
        "vram_total_gb": round(vram_total_mib / 1024.0, 2) if vram_total_mib is not None else None,
        "temperature_c": _metric_float(row[4]),
    }


def _read_memory_status() -> Dict[str, Any]:
    """Read host RAM usage using only the Python standard library."""
    total_bytes = 0
    available_bytes = 0

    try:
        if os.name == "nt":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            memory = _MemoryStatusEx()
            memory.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                return {}
            total_bytes = int(memory.ullTotalPhys)
            available_bytes = int(memory.ullAvailPhys)
        else:
            meminfo = {}
            with open("/proc/meminfo", "r", encoding="ascii") as handle:
                for line in handle:
                    key, value = line.split(":", 1)
                    meminfo[key] = int(value.strip().split()[0]) * 1024
            total_bytes = meminfo.get("MemTotal", 0)
            available_bytes = meminfo.get(
                "MemAvailable",
                meminfo.get("MemFree", 0) + meminfo.get("Buffers", 0) + meminfo.get("Cached", 0),
            )
    except (OSError, ValueError):
        return {}

    if total_bytes <= 0:
        return {}
    used_bytes = max(0, total_bytes - available_bytes)
    return {
        "used_gb": round(used_bytes / (1024 ** 3), 2),
        "total_gb": round(total_bytes / (1024 ** 3), 2),
        "used_percent": round((used_bytes / total_bytes) * 100.0, 1),
    }


def _read_disk_status(path: Any) -> Dict[str, Any]:
    probe_path = Path(path or Path.cwd()).expanduser()
    while not probe_path.exists() and probe_path != probe_path.parent:
        probe_path = probe_path.parent
    if not probe_path.exists():
        probe_path = Path.cwd()

    try:
        usage = shutil.disk_usage(str(probe_path))
    except OSError:
        return {}
    return {
        "path": str(probe_path),
        "free_gb": round(usage.free / (1024 ** 3), 2),
        "used_gb": round(usage.used / (1024 ** 3), 2),
        "total_gb": round(usage.total / (1024 ** 3), 2),
        "used_percent": round((usage.used / usage.total) * 100.0, 1) if usage.total else 0.0,
    }


def _collect_hardware_status(disk_path: Any) -> Dict[str, Any]:
    return {
        "gpu": _read_gpu_status(),
        "memory": _read_memory_status(),
        "disk": _read_disk_status(disk_path),
    }


def _get_cached_hardware_status(disk_path: Any) -> Dict[str, Any]:
    cache_key = str(Path(disk_path or Path.cwd()).expanduser())
    now = time.monotonic()
    with _hardware_status_lock:
        cached = _hardware_status_cache.get(cache_key)
        if cached and now - cached[0] < _HARDWARE_STATUS_CACHE_SECONDS:
            return cached[1]
        status = _collect_hardware_status(disk_path)
        _hardware_status_cache[cache_key] = (now, status)
        return status


class _AppVersionProxy:
    def __getitem__(self, key: str) -> Any:
        return get_version_info().get(key, "")

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        return self[key]


# 幫 Jinja2 準備一些好用的過濾器
def ai_simple(ai_judgment):
    if not ai_judgment: return ""
    if ai_judgment.startswith("ERR:HY"): return "HY"
    return "OK-i" if ai_judgment == "OK-i" else ("OK" if ai_judgment == "OK" else ("NG" if ai_judgment.startswith("NG") else ("ERR" if ai_judgment.startswith("ERR") else ai_judgment)))

def ai_badge(ai_judgment):
    simple = ai_simple(ai_judgment)
    return "badge-ok-i" if simple == "OK-i" else ("badge-ok" if simple == "OK" else ("badge-ng" if simple == "NG" else "badge-err"))

def mj_badge(machine_judgment):
    if machine_judgment == "OK":
        return "badge-ok"
    elif machine_judgment == "HY":
        return "badge-err"
    else:
        return "badge-ng"

def img_status_info(img):
    if img.get("is_dust_only"): return "灰塵 (DUST)", "badge-err"
    if img.get("is_bomb"): return "炸彈 (BOMB)", "badge-err"
    if img["is_ng"]: return "NG", "badge-ng"
    return "OK", "badge-ok"

def tile_info(t):
    badge = "badge-ng"
    info = f"Score: {t['score']:.3f}"
    if t.get("is_aoi_coord"):
        code = t.get('aoi_defect_code', '')
        if t.get("is_anomaly"):
            info += f" | 🎯 AOI座標 ({code}) AI也判NG"
        else:
            badge = "badge-ok"
            info += f" | 🎯 AOI座標 ({code}) AI判OK"
    if t.get("is_exclude_zone"):
        badge = "badge-ok"
        info += " | 不檢測排除區域"
    elif t.get("is_dust"):
        badge = "badge-err"
        info += f" | 灰塵 Region COV: {t.get('dust_iou',0):.3f}"
    if t.get("is_bomb") and t.get("is_anomaly"):
        badge = "badge-err"
        info += f" | 炸彈代碼: {t.get('bomb_code','')}"
    if t.get("scratch_filtered") and not t.get("is_bomb") and not t.get("is_dust"):
        badge = "badge-ok"
        info += f" | 🧽 DINOv2 救回 (scratch={t.get('scratch_score', 0):.3f})"
    return badge, info

def get_img_stem(img):
    img_path_str = img.get("image_path", "")
    img_stem = Path(img_path_str).stem if img_path_str else ""
    raw_name = img.get("image_name", "")
    if raw_name.startswith("overview_"):
        img_stem = raw_name.replace("overview_", "").replace(".png", "").replace(".jpg", "")
    return img_stem or raw_name

def hm_relative(path_str, base_dir):
    if not path_str or not base_dir: return ""
    try:
        rel = Path(path_str).relative_to(base_dir)
        return rel.as_posix()
    except (ValueError, TypeError):
        return ""


def _within_spec_auto_visual_output(base_dir: str, glass_id: str = "", inference_record_id: int = 0) -> Tuple[Optional[Path], str]:
    if not base_dir:
        return None, ""
    base = Path(base_dir)
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(glass_id or "").strip())[:80]
    if not safe_key:
        safe_key = f"record_{inference_record_id or 'unknown'}"
    run_dir = base / "within_spec_inference" / f"{safe_key}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    rel = hm_relative(str(run_dir), base)
    if not rel:
        return None, ""
    return run_dir, f"/heatmaps/{rel}"


def _serialize_pending_changes(pc) -> list:
    """把 {(lighting, zone): count} 轉為 [{lighting, zone, count}] 給 JSON 用。

    無項目或 None 時回 []。
    """
    if not pc:
        return []
    out = []
    for key, count in pc.items():
        if isinstance(key, tuple) and len(key) == 2:
            lighting, zone = key
            out.append({"lighting": lighting, "zone": zone, "count": int(count)})
    return out


_DOT_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DOT_RULER_MM_PER_PX = 0.0245
DOT_RULER_CALIBRATION_POINTS = []
DOT_RULER_CALIBRATION_SOURCE = "default mm/px ratio 0.0245"
DOT_PREPROCESS_METHOD = "gaussian"
DOT_PREPROCESS_PARAMS = {"kernel_size": 7, "sigma": 1.0}
WITHIN_SPEC_PARAM = "within_spec_judgment_rules"


def _preprocess_dot_image_for_detection(image_bgr, method: str = None, params: dict = None) -> tuple:
    """Apply dot-detection preprocessing used before residual detection."""
    from capi_image_preprocess_lab import apply_preprocess_method

    result = apply_preprocess_method(
        image_bgr,
        method or DOT_PREPROCESS_METHOD,
        params or DOT_PREPROCESS_PARAMS,
    )
    return result["image"], {
        "method": result["method"],
        "method_label": result["method_label"],
        "applied_params": result["applied_params"],
        "notes": result.get("notes", []),
    }


def _list_debug_dot_samples(base_dir: Path) -> dict:
    """List bundled black/white dot samples for the debug page."""
    samples = {}
    for polarity in ("black", "white"):
        folder = base_dir / polarity
        items = []
        if folder.exists() and folder.is_dir():
            for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
                if path.is_file() and path.suffix.lower() in _DOT_IMAGE_EXTS:
                    items.append({"name": path.name, "path": str(path)})
        if polarity == "black":
            ruler_path = base_dir.parent.parent / "dot_size.png"
            if ruler_path.is_file():
                items.insert(0, {
                    "name": f"{ruler_path.name} (點線規)",
                    "path": str(ruler_path),
                })
        samples[polarity] = items
    return samples


def _odd_kernel(value, default: int = 31, min_value: int = 3) -> int:
    try:
        kernel = int(value)
    except (TypeError, ValueError):
        kernel = default
    kernel = max(min_value, kernel)
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def _dot_hysteresis_kwargs(dot_cfg: Dict[str, Any]) -> Dict[str, Any]:
    dot_cfg = dot_cfg or {}
    low = max(0, _as_int(dot_cfg.get("hysteresis_low_threshold"), 2))
    high = max(low, max(0, _as_int(dot_cfg.get("hysteresis_high_threshold"), 4)))
    second_low = max(0, _as_int(dot_cfg.get("hysteresis_second_low_threshold"), 3))
    second_high = max(second_low, max(0, _as_int(dot_cfg.get("hysteresis_second_high_threshold"), 4)))
    return {
        "hysteresis_low_threshold": low,
        "hysteresis_high_threshold": high,
        "hysteresis_edge_width_percent": max(0.0, _as_float(dot_cfg.get("hysteresis_edge_width_percent"), 3.0)),
        "hysteresis_edge_extra_threshold": max(0, _as_int(dot_cfg.get("hysteresis_edge_extra_threshold"), 2)),
        "hysteresis_second_low_threshold": second_low,
        "hysteresis_second_high_threshold": second_high,
        "hysteresis_second_edge_width_percent": max(0.0, _as_float(dot_cfg.get("hysteresis_second_edge_width_percent"), 9.5)),
        "hysteresis_second_edge_extra_threshold": max(0, _as_int(dot_cfg.get("hysteresis_second_edge_extra_threshold"), 2)),
        "hysteresis_switch_count_threshold": max(0, _as_int(dot_cfg.get("hysteresis_switch_count_threshold"), 5)),
        "hysteresis_second_max_count": max(0, _as_int(dot_cfg.get("hysteresis_second_max_count"), 5)),
        "hysteresis_edge_suppress_percent": max(0.0, _as_float(dot_cfg.get("hysteresis_edge_suppress_percent"), 0.0)),
    }


def _detect_dot_components(
    image_bgr,
    *,
    polarity: str,
    diff_threshold: int,
    background_kernel: int,
    min_area: int,
    max_area: int,
    morph_open: int,
    size_metric: str,
    unit_per_px: float,
    defect_threshold: float,
    min_aspect_ratio: float = 0.0,
    edge_margin: int = 0,
    segmentation_method: str = "background_diff",
    hysteresis_low_threshold: Optional[int] = None,
    hysteresis_high_threshold: Optional[int] = None,
    hysteresis_edge_width_percent: float = 3.0,
    hysteresis_edge_extra_threshold: int = 2,
    hysteresis_second_low_threshold: Optional[int] = None,
    hysteresis_second_high_threshold: Optional[int] = None,
    hysteresis_second_edge_width_percent: float = 9.5,
    hysteresis_second_edge_extra_threshold: int = 2,
    hysteresis_switch_count_threshold: int = 5,
    hysteresis_second_max_count: int = 5,
    hysteresis_edge_suppress_percent: float = 0.0,
    include_visuals: bool = True,
) -> dict:
    """Detect dot-like dark/bright components and measure their visible size."""
    import cv2
    import numpy as np

    if image_bgr.ndim == 2:
        gray = image_bgr
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if include_visuals else None
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        overlay = image_bgr.copy() if include_visuals else None

    background_kernel = _odd_kernel(background_kernel)
    bg = cv2.medianBlur(gray, background_kernel)
    if polarity == "white":
        diff = cv2.subtract(gray, bg)
    else:
        diff = cv2.subtract(bg, gray)

    segmentation_method = str(segmentation_method or "background_diff").strip().lower()
    if segmentation_method not in ("background_diff", "hysteresis", "morph_hat", "adaptive_mean"):
        segmentation_method = "background_diff"

    if segmentation_method == "morph_hat":
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (background_kernel, background_kernel),
        )
        if polarity == "white":
            bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            diff = cv2.subtract(gray, bg)
        else:
            bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            diff = cv2.subtract(bg, gray)

    diff_threshold = max(0, int(diff_threshold))
    if segmentation_method == "adaptive_mean":
        low_threshold = diff_threshold
        high_threshold = diff_threshold
        threshold_type = cv2.THRESH_BINARY if polarity == "white" else cv2.THRESH_BINARY_INV
        adaptive_c = -diff_threshold if polarity == "white" else diff_threshold
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            threshold_type,
            background_kernel,
            adaptive_c,
        )
    elif segmentation_method == "hysteresis":
        import math

        high_threshold = diff_threshold if hysteresis_high_threshold is None else int(hysteresis_high_threshold)
        low_threshold = diff_threshold if hysteresis_low_threshold is None else int(hysteresis_low_threshold)
        low_threshold = max(0, low_threshold)
        high_threshold = max(low_threshold, max(0, high_threshold))

        second_low_threshold = low_threshold if hysteresis_second_low_threshold is None else int(hysteresis_second_low_threshold)
        second_high_threshold = high_threshold if hysteresis_second_high_threshold is None else int(hysteresis_second_high_threshold)
        second_low_threshold = max(0, second_low_threshold)
        second_high_threshold = max(second_low_threshold, max(0, second_high_threshold))
        hysteresis_switch_count_threshold = max(0, int(hysteresis_switch_count_threshold or 0))
        hysteresis_second_max_count = max(0, int(hysteresis_second_max_count or 0))
        hysteresis_edge_suppress_percent = max(0.0, float(hysteresis_edge_suppress_percent or 0.0))

        min_dim = max(1, min(gray.shape[:2]))
        sigma_bg = max(2.0, float(min_dim) / 25.0)
        bg_kernel = max(3, int(math.ceil(sigma_bg * 6.0 + 1.0)))
        if bg_kernel % 2 == 0:
            bg_kernel += 1
        bg = cv2.GaussianBlur(
            gray,
            (bg_kernel, bg_kernel),
            sigma_bg,
            borderType=cv2.BORDER_REPLICATE,
        )
        diff = cv2.absdiff(gray, bg)
        diff = cv2.medianBlur(diff, 3)

        def hysteresis_mask_for_params(
            high: int,
            low: int,
            edge_width_percent: float,
            edge_extra_threshold: int,
        ):
            edge_width_percent = max(0.0, float(edge_width_percent or 0.0))
            edge_extra_threshold = max(0, int(edge_extra_threshold or 0))
            h, w = gray.shape[:2]
            border_x = int(w * edge_width_percent / 100.0)
            border_y = int(h * edge_width_percent / 100.0)
            threshold_extra = np.zeros_like(diff, dtype=np.uint8)
            if edge_extra_threshold > 0 and (border_x > 0 or border_y > 0):
                if border_x > 0:
                    threshold_extra[:, :border_x] = edge_extra_threshold
                    threshold_extra[:, w - border_x:] = edge_extra_threshold
                if border_y > 0:
                    threshold_extra[:border_y, :] = edge_extra_threshold
                    threshold_extra[h - border_y:, :] = edge_extra_threshold
            high_map = np.minimum(255, int(high) + threshold_extra.astype(np.uint16)).astype(np.uint8)
            low_map = np.minimum(255, int(low) + threshold_extra.astype(np.uint16)).astype(np.uint8)
            low_mask = (diff >= low_map).astype("uint8") * 255
            high_mask = (diff >= high_map).astype("uint8") * 255
            low_count, low_labels = cv2.connectedComponents(low_mask, 8)
            if low_count <= 1 or not np.any(high_mask):
                return np.zeros_like(low_mask)
            keep_labels = np.unique(low_labels[high_mask > 0])
            keep_labels = keep_labels[keep_labels != 0]
            chosen = np.zeros_like(low_mask)
            if keep_labels.size:
                chosen[np.isin(low_labels, keep_labels)] = 255
            return chosen

        def _hysteresis_count_mask(mask_to_count):
            count_mask = mask_to_count.copy()
            open_size = int(morph_open or 0)
            if open_size > 1:
                open_kernel = np.ones((open_size, open_size), dtype=np.uint8)
                count_mask = cv2.morphologyEx(count_mask, cv2.MORPH_OPEN, open_kernel)
            if hysteresis_edge_suppress_percent > 0:
                h, w = gray.shape[:2]
                border_x = int(w * hysteresis_edge_suppress_percent / 100.0)
                border_y = int(h * hysteresis_edge_suppress_percent / 100.0)
                if border_x > 0:
                    count_mask[:, :border_x] = 0
                    count_mask[:, w - border_x:] = 0
                if border_y > 0:
                    count_mask[:border_y, :] = 0
                    count_mask[h - border_y:, :] = 0
            return count_mask

        def count_mask_components(mask_to_count):
            count_mask = _hysteresis_count_mask(mask_to_count)
            count, labels_to_count, stats_to_count, _ = cv2.connectedComponentsWithStats(count_mask, 8)
            total = 0
            min_area_for_count = max(1, int(min_area))
            max_area_for_count = int(max_area or 0)
            if max_area_for_count <= 0:
                max_area_for_count = gray.shape[0] * gray.shape[1]
            min_aspect_for_count = max(0.0, float(min_aspect_ratio or 0.0))
            edge_margin_for_count = max(0, int(edge_margin or 0))
            background_gray_for_count = float(np.mean(bg))
            for idx in range(1, count):
                x, y, w, h, area = [int(v) for v in stats_to_count[idx]]
                if area < min_area_for_count:
                    continue
                if area > max_area_for_count:
                    continue
                bbox_max = float(max(w, h))
                aspect_ratio = float(min(w, h) / bbox_max) if bbox_max > 0 else 0.0
                if min_aspect_for_count > 0 and aspect_ratio < min_aspect_for_count:
                    continue
                if edge_margin_for_count > 0:
                    if (
                        x < edge_margin_for_count
                        or y < edge_margin_for_count
                        or x + w > gray.shape[1] - edge_margin_for_count
                        or y + h > gray.shape[0] - edge_margin_for_count
                    ):
                        continue
                component_labels = labels_to_count[y:y + h, x:x + w]
                component_mask = component_labels == idx
                component_values = gray[y:y + h, x:x + w][component_mask]
                if component_values.size:
                    component_polarity = "black" if float(component_values.mean()) < background_gray_for_count else "white"
                    if component_polarity != polarity:
                        continue
                else:
                    continue
                total += 1
            return total

        group1_mask = hysteresis_mask_for_params(
            high_threshold,
            low_threshold,
            hysteresis_edge_width_percent,
            hysteresis_edge_extra_threshold,
        )
        group1_count = count_mask_components(group1_mask)
        group2_count = 0
        selected_group = 1
        group2_attempted = False
        switch_reason = ""
        group2_reject_reason = ""
        mask = group1_mask
        if group1_count == 0 or group1_count > hysteresis_switch_count_threshold:
            group2_attempted = True
            switch_reason = (
                "group1_empty"
                if group1_count == 0
                else "group1_count_above_switch"
            )
            group2_mask = hysteresis_mask_for_params(
                second_high_threshold,
                second_low_threshold,
                hysteresis_second_edge_width_percent,
                hysteresis_second_edge_extra_threshold,
            )
            group2_count = count_mask_components(group2_mask)
            if group2_count <= hysteresis_second_max_count:
                mask = group2_mask
                selected_group = 2
            else:
                group2_reject_reason = "group2_count_above_max2"

        if hysteresis_edge_suppress_percent > 0:
            h, w = gray.shape[:2]
            border_x = int(w * hysteresis_edge_suppress_percent / 100.0)
            border_y = int(h * hysteresis_edge_suppress_percent / 100.0)
            if border_x > 0:
                mask[:, :border_x] = 0
                mask[:, w - border_x:] = 0
            if border_y > 0:
                mask[:border_y, :] = 0
                mask[h - border_y:, :] = 0
    elif segmentation_method == "morph_hat":
        high_threshold = diff_threshold if hysteresis_high_threshold is None else int(hysteresis_high_threshold)
        low_threshold = diff_threshold if hysteresis_low_threshold is None else int(hysteresis_low_threshold)
        low_threshold = max(0, low_threshold)
        high_threshold = max(low_threshold, max(0, high_threshold))

        low_mask = (diff >= low_threshold).astype("uint8") * 255
        high_mask = (diff >= high_threshold).astype("uint8") * 255
        low_count, low_labels = cv2.connectedComponents(low_mask, 8)
        if low_count <= 1 or not np.any(high_mask):
            mask = np.zeros_like(low_mask)
        else:
            keep_labels = np.unique(low_labels[high_mask > 0])
            keep_labels = keep_labels[keep_labels != 0]
            mask = np.zeros_like(low_mask)
            if keep_labels.size:
                mask[np.isin(low_labels, keep_labels)] = 255
    else:
        low_threshold = diff_threshold
        high_threshold = diff_threshold
        mask = (diff >= diff_threshold).astype("uint8") * 255

    morph_open = int(morph_open or 0)
    if morph_open > 1:
        kernel = np.ones((morph_open, morph_open), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(1, int(min_area))
    max_area = int(max_area or 0)
    if max_area <= 0:
        max_area = gray.shape[0] * gray.shape[1]
    min_aspect_ratio = max(0.0, float(min_aspect_ratio or 0.0))
    edge_margin = max(0, int(edge_margin or 0))

    candidates = []
    rejected_candidates = []
    calibrated = unit_per_px > 0
    background_gray = float(np.mean(bg)) if segmentation_method == "hysteresis" else 0.0

    def add_rejected(
        *,
        reason: str,
        label: int,
        x: int,
        y: int,
        w: int,
        h: int,
        area: int,
        aspect_ratio: float,
        detected_polarity: str = "",
    ):
        if len(rejected_candidates) >= 50:
            return
        component_labels = labels[y:y + h, x:x + w]
        component_mask = component_labels == label
        component_diff = diff[y:y + h, x:x + w][component_mask]
        rejected_candidates.append({
            "reason": reason,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area_px": area,
            "aspect_ratio": round(float(aspect_ratio), 3),
            "max_diff": int(component_diff.max()) if component_diff.size else 0,
            "mean_diff": round(float(component_diff.mean()), 2) if component_diff.size else 0.0,
            "expected_polarity": polarity,
            "detected_polarity": detected_polarity,
        })

    for label in range(1, num_labels):
        x, y, w, h, area = [int(v) for v in stats[label]]
        bbox_max = float(max(w, h))
        aspect_ratio = float(min(w, h) / bbox_max) if bbox_max > 0 else 0.0
        if area < min_area:
            add_rejected(
                reason="area_too_small",
                label=label,
                x=x,
                y=y,
                w=w,
                h=h,
                area=area,
                aspect_ratio=aspect_ratio,
            )
            continue
        if area > max_area:
            add_rejected(
                reason="area_too_large",
                label=label,
                x=x,
                y=y,
                w=w,
                h=h,
                area=area,
                aspect_ratio=aspect_ratio,
            )
            continue
        if min_aspect_ratio > 0 and aspect_ratio < min_aspect_ratio:
            add_rejected(
                reason="aspect_ratio_below_min",
                label=label,
                x=x,
                y=y,
                w=w,
                h=h,
                area=area,
                aspect_ratio=aspect_ratio,
            )
            continue
        if edge_margin > 0:
            if (
                x < edge_margin
                or y < edge_margin
                or x + w > gray.shape[1] - edge_margin
                or y + h > gray.shape[0] - edge_margin
            ):
                add_rejected(
                    reason="edge_margin",
                    label=label,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    area=area,
                    aspect_ratio=aspect_ratio,
                )
                continue

        component_labels = labels[y:y + h, x:x + w]
        component_mask = (component_labels == label).astype("uint8") * 255
        if segmentation_method == "hysteresis":
            component_values = gray[y:y + h, x:x + w][component_mask > 0]
            component_polarity = "black" if float(component_values.mean()) < background_gray else "white"
            if component_polarity != polarity:
                add_rejected(
                    reason="polarity_mismatch",
                    label=label,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    area=area,
                    aspect_ratio=aspect_ratio,
                    detected_polarity=component_polarity,
                )
                continue
        else:
            component_polarity = polarity
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        enclosing_diameter = float(max(w, h))
        if contours:
            contour = max(contours, key=cv2.contourArea)
            _, radius = cv2.minEnclosingCircle(contour)
            enclosing_diameter = float(radius * 2.0)

        equivalent_diameter = float((4.0 * area / np.pi) ** 0.5)
        bbox_diagonal = float(np.hypot(w, h))
        if size_metric == "equivalent":
            size_px = equivalent_diameter
            size_mode = "equivalent"
        elif size_metric == "enclosing":
            size_px = enclosing_diameter
            size_mode = "enclosing"
        elif size_metric == "bbox_diagonal":
            size_px = bbox_diagonal
            size_mode = "bbox_diagonal"
        else:
            size_px = bbox_max
            size_mode = "bbox_max"

        size_units = float(size_px * unit_per_px) if calibrated else None
        size_mm = size_units if calibrated else None
        component_diff = diff[y:y + h, x:x + w][component_mask > 0]
        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area_px": area,
            "center_x": round(float(centroids[label][0]), 2),
            "center_y": round(float(centroids[label][1]), 2),
            "aspect_ratio": round(aspect_ratio, 3),
            "bbox_max_diameter_px": round(bbox_max, 2),
            "bbox_diagonal_px": round(bbox_diagonal, 2),
            "equivalent_diameter_px": round(equivalent_diameter, 2),
            "enclosing_diameter_px": round(enclosing_diameter, 2),
            "size_mode": size_mode,
            "size_px": round(float(size_px), 2),
            "size_units": round(size_units, 4) if size_units is not None else None,
            "size_mm": round(size_mm, 4) if size_mm is not None else None,
            "max_diff": int(component_diff.max()),
            "mean_diff": round(float(component_diff.mean()), 2),
            "polarity": component_polarity,
            "is_defect": bool(calibrated and size_units >= defect_threshold),
        })

    candidates.sort(key=lambda c: (c["is_defect"], c["size_px"]), reverse=True)
    rejected_candidates.sort(key=lambda c: (c["area_px"], c["max_diff"]), reverse=True)
    for idx, candidate in enumerate(candidates, 1):
        candidate["id"] = idx
        if not include_visuals:
            continue
        color = (0, 0, 255) if candidate["is_defect"] else (0, 200, 0)
        x, y, w, h = candidate["x"], candidate["y"], candidate["w"], candidate["h"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
        cv2.circle(
            overlay,
            (int(round(candidate["center_x"])), int(round(candidate["center_y"]))),
            3,
            color,
            -1,
        )
        label = f"#{idx} {candidate['size_px']:.1f}px"
        if candidate["size_units"] is not None:
            label = f"#{idx} {candidate['size_units']:.3g}mm"
        cv2.putText(
            overlay,
            label,
            (x, max(12, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    result = {
        "gray": gray,
        "diff": diff,
        "mask": mask,
        "candidates": candidates,
        "rejected_candidates": rejected_candidates,
        "calibrated": calibrated,
        "segmentation_method": segmentation_method,
        "thresholds": {
            "diff_threshold": diff_threshold,
            "hysteresis_low_threshold": low_threshold,
            "hysteresis_high_threshold": high_threshold,
        },
    }
    if segmentation_method == "hysteresis":
        result["thresholds"].update({
            "hysteresis_second_low_threshold": second_low_threshold,
            "hysteresis_second_high_threshold": second_high_threshold,
            "hysteresis_edge_width_percent": round(float(hysteresis_edge_width_percent), 4),
            "hysteresis_edge_extra_threshold": int(hysteresis_edge_extra_threshold),
            "hysteresis_second_edge_width_percent": round(float(hysteresis_second_edge_width_percent), 4),
            "hysteresis_second_edge_extra_threshold": int(hysteresis_second_edge_extra_threshold),
            "hysteresis_switch_count_threshold": int(hysteresis_switch_count_threshold),
            "hysteresis_second_max_count": int(hysteresis_second_max_count),
            "hysteresis_edge_suppress_percent": round(float(hysteresis_edge_suppress_percent), 4),
            "hysteresis_selected_group": int(selected_group),
            "hysteresis_group1_count": int(group1_count),
            "hysteresis_group2_count": int(group2_count),
            "hysteresis_group2_attempted": bool(group2_attempted),
            "hysteresis_group2_adopted": bool(selected_group == 2),
            "hysteresis_switch_reason": switch_reason,
            "hysteresis_group2_reject_reason": group2_reject_reason,
        })
    if include_visuals:
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        result.update({
            "overlay": overlay,
            "diff_color": cv2.applyColorMap(diff_norm.astype("uint8"), cv2.COLORMAP_JET),
            "mask_color": cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
        })
    return result


def _detect_white_halo_components(
    image_bgr,
    *,
    diff_threshold: int,
    background_kernel: int,
    min_area: int,
    max_area: int,
    morph_open: int,
    size_metric: str,
    unit_per_px: float,
    defect_threshold: float,
    min_aspect_ratio: float = 0.0,
    edge_margin: int = 0,
    include_visuals: bool = True,
) -> dict:
    """Detect broad bright halo around the strongest dark seed."""
    import cv2
    import numpy as np

    dark = _detect_dot_components(
        image_bgr,
        polarity="black",
        diff_threshold=max(4, int(diff_threshold)),
        background_kernel=background_kernel,
        min_area=5,
        max_area=max_area,
        morph_open=0,
        size_metric=size_metric,
        unit_per_px=unit_per_px,
        defect_threshold=defect_threshold,
        min_aspect_ratio=max(0.45, min_aspect_ratio),
        edge_margin=edge_margin,
        include_visuals=False,
    )
    if image_bgr.ndim == 2:
        gray = image_bgr
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if include_visuals else None
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        overlay = image_bgr.copy() if include_visuals else None

    blank = np.zeros(gray.shape, dtype=np.uint8)
    if not dark["candidates"]:
        result = {
            "gray": gray,
            "diff": blank,
            "mask": blank,
            "candidates": [],
            "calibrated": unit_per_px > 0,
            "segmentation_method": "halo",
            "thresholds": {
                "diff_threshold": int(diff_threshold),
                "hysteresis_low_threshold": int(diff_threshold),
                "hysteresis_high_threshold": int(diff_threshold),
            },
        }
        if include_visuals:
            result.update({
                "overlay": overlay,
                "diff_color": cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR),
                "mask_color": cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR),
            })
        return result

    seed = dark["candidates"][0]
    cx = float(seed["center_x"])
    cy = float(seed["center_y"])
    yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    outer_radius = max(18.0, min(96.0, float(max(seed["w"], seed["h"])) * 8.0))
    inner_radius = max(2.0, float(max(seed["w"], seed["h"])) * 0.7)
    search_mask = (dist <= outer_radius) & (dist >= inner_radius)
    background_mask = (dist > outer_radius * 0.75) & (dist <= outer_radius)
    if not np.any(search_mask) or not np.any(background_mask):
        background_level = float(np.median(gray))
    else:
        background_level = float(np.median(gray[background_mask]))
    halo_threshold = int(round(background_level + max(1, int(diff_threshold))))
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[search_mask & (gray >= halo_threshold)] = 255
    if morph_open > 1:
        kernel = np.ones((int(morph_open), int(morph_open)), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(1, int(min_area))
    max_area = int(max_area or 0)
    if max_area <= 0:
        max_area = gray.shape[0] * gray.shape[1]
    min_aspect_ratio = max(0.0, float(min_aspect_ratio or 0.0))
    edge_margin = max(0, int(edge_margin or 0))
    calibrated = unit_per_px > 0
    candidates = []
    rejected_candidates = []
    diff = np.zeros_like(gray, dtype=np.uint8)
    brighter = gray.astype(np.int16) - int(round(background_level))
    diff[brighter > 0] = np.clip(brighter[brighter > 0], 0, 255).astype(np.uint8)

    for label in range(1, num_labels):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < min_area or area > max_area:
            continue
        if not (x <= cx <= x + w and y <= cy <= y + h):
            continue
        bbox_max = float(max(w, h))
        aspect_ratio = float(min(w, h) / bbox_max) if bbox_max > 0 else 0.0
        if min_aspect_ratio > 0 and aspect_ratio < min_aspect_ratio:
            continue
        if edge_margin > 0:
            if (
                x < edge_margin
                or y < edge_margin
                or x + w > gray.shape[1] - edge_margin
                or y + h > gray.shape[0] - edge_margin
            ):
                continue

        component_labels = labels[y:y + h, x:x + w]
        component_mask = (component_labels == label).astype("uint8") * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        enclosing_diameter = float(max(w, h))
        if contours:
            contour = max(contours, key=cv2.contourArea)
            _, radius = cv2.minEnclosingCircle(contour)
            enclosing_diameter = float(radius * 2.0)

        equivalent_diameter = float((4.0 * area / np.pi) ** 0.5)
        bbox_diagonal = float(np.hypot(w, h))
        if size_metric == "equivalent":
            size_px = equivalent_diameter
            size_mode = "equivalent"
        elif size_metric == "enclosing":
            size_px = enclosing_diameter
            size_mode = "enclosing"
        elif size_metric == "bbox_diagonal":
            size_px = bbox_diagonal
            size_mode = "bbox_diagonal"
        else:
            size_px = bbox_max
            size_mode = "bbox_max"

        size_units = float(size_px * unit_per_px) if calibrated else None
        component_values = gray[y:y + h, x:x + w][component_mask > 0]
        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area_px": area,
            "center_x": round(float(centroids[label][0]), 2),
            "center_y": round(float(centroids[label][1]), 2),
            "aspect_ratio": round(aspect_ratio, 3),
            "bbox_max_diameter_px": round(bbox_max, 2),
            "bbox_diagonal_px": round(bbox_diagonal, 2),
            "equivalent_diameter_px": round(equivalent_diameter, 2),
            "enclosing_diameter_px": round(enclosing_diameter, 2),
            "size_mode": size_mode,
            "size_px": round(float(size_px), 2),
            "size_units": round(size_units, 4) if size_units is not None else None,
            "size_mm": round(size_units, 4) if size_units is not None else None,
            "max_diff": int(max(0, int(component_values.max()) - int(round(background_level)))),
            "mean_diff": round(float(component_values.mean() - background_level), 2),
            "halo_seed_x": round(cx, 2),
            "halo_seed_y": round(cy, 2),
            "is_defect": bool(calibrated and size_units >= defect_threshold),
        })

    candidates.sort(key=lambda c: (c["is_defect"], c["size_px"]), reverse=True)
    rejected_candidates.sort(key=lambda c: (c["area_px"], c["max_diff"]), reverse=True)
    for idx, candidate in enumerate(candidates, 1):
        candidate["id"] = idx
        if not include_visuals:
            continue
        color = (0, 0, 255) if candidate["is_defect"] else (0, 200, 0)
        x, y, w, h = candidate["x"], candidate["y"], candidate["w"], candidate["h"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
        cv2.circle(overlay, (int(round(cx)), int(round(cy))), 3, (255, 0, 0), -1)
        cv2.putText(overlay, f"#{idx} {candidate['size_px']:.1f}px", (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    result = {
        "gray": gray,
        "diff": diff,
        "mask": mask,
        "candidates": candidates,
        "rejected_candidates": rejected_candidates,
        "calibrated": calibrated,
        "segmentation_method": "halo",
        "thresholds": {
            "diff_threshold": int(diff_threshold),
            "hysteresis_low_threshold": int(diff_threshold),
            "hysteresis_high_threshold": int(diff_threshold),
        },
    }
    if include_visuals:
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        result.update({
            "overlay": overlay,
            "diff_color": cv2.applyColorMap(diff_norm.astype("uint8"), cv2.COLORMAP_JET),
            "mask_color": cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
        })
    return result


def _detect_dot_components_auto(
    image_bgr,
    *,
    polarity: str,
    segmentation_method: str,
    diff_threshold: int,
    background_kernel: int,
    min_area: int,
    max_area: int,
    morph_open: int,
    size_metric: str,
    unit_per_px: float,
    defect_threshold: float,
    min_aspect_ratio: float = 0.0,
    edge_margin: int = 0,
    hysteresis_low_threshold: Optional[int] = None,
    hysteresis_high_threshold: Optional[int] = None,
    hysteresis_edge_width_percent: float = 3.0,
    hysteresis_edge_extra_threshold: int = 2,
    hysteresis_second_low_threshold: Optional[int] = None,
    hysteresis_second_high_threshold: Optional[int] = None,
    hysteresis_second_edge_width_percent: float = 9.5,
    hysteresis_second_edge_extra_threshold: int = 2,
    hysteresis_switch_count_threshold: int = 5,
    hysteresis_second_max_count: int = 5,
    hysteresis_edge_suppress_percent: float = 0.0,
    include_visuals: bool = True,
) -> dict:
    """Select the best dot detector for the configured mode."""
    mode = str(segmentation_method or "background_diff").strip().lower()
    common = {
        "diff_threshold": diff_threshold,
        "background_kernel": background_kernel,
        "min_area": min_area,
        "max_area": max_area,
        "morph_open": morph_open,
        "size_metric": size_metric,
        "unit_per_px": unit_per_px,
        "defect_threshold": defect_threshold,
        "min_aspect_ratio": min_aspect_ratio,
        "edge_margin": edge_margin,
        "include_visuals": include_visuals,
    }
    hysteresis_common = {
        "hysteresis_low_threshold": hysteresis_low_threshold,
        "hysteresis_high_threshold": hysteresis_high_threshold,
        "hysteresis_edge_width_percent": hysteresis_edge_width_percent,
        "hysteresis_edge_extra_threshold": hysteresis_edge_extra_threshold,
        "hysteresis_second_low_threshold": hysteresis_second_low_threshold,
        "hysteresis_second_high_threshold": hysteresis_second_high_threshold,
        "hysteresis_second_edge_width_percent": hysteresis_second_edge_width_percent,
        "hysteresis_second_edge_extra_threshold": hysteresis_second_edge_extra_threshold,
        "hysteresis_switch_count_threshold": hysteresis_switch_count_threshold,
        "hysteresis_second_max_count": hysteresis_second_max_count,
        "hysteresis_edge_suppress_percent": hysteresis_edge_suppress_percent,
    }

    if polarity == "white" and mode == "halo":
        return _detect_white_halo_components(image_bgr, **common)
    if mode != "auto":
        return _detect_dot_components(
            image_bgr,
            polarity=polarity,
            segmentation_method=mode,
            **hysteresis_common,
            **common,
        )

    candidates = []
    base_modes = ["background_diff", "hysteresis", "adaptive_mean"]
    if polarity == "black":
        base_modes.append("morph_hat")
    for candidate_mode in base_modes:
        detected = _detect_dot_components(
            image_bgr,
            polarity=polarity,
            segmentation_method=candidate_mode,
            **hysteresis_common,
            **common,
        )
        candidates.append(detected)

    if polarity == "white":
        halo = _detect_white_halo_components(
            image_bgr,
            diff_threshold=max(1, int(diff_threshold // 2) if diff_threshold > 2 else int(diff_threshold)),
            background_kernel=background_kernel,
            min_area=max(min_area, 50),
            max_area=max_area,
            morph_open=morph_open,
            size_metric=size_metric,
            unit_per_px=unit_per_px,
            defect_threshold=defect_threshold,
            min_aspect_ratio=min(min_aspect_ratio, 0.25) if min_aspect_ratio > 0 else 0.25,
            edge_margin=edge_margin,
            include_visuals=include_visuals,
        )
        candidates.append(halo)

    def score(result: Dict[str, Any]) -> Tuple[int, float, int]:
        top = (result.get("candidates") or [{}])[0]
        return (
            1 if top.get("is_defect") else 0,
            float(top.get("size_px") or 0.0),
            len(result.get("candidates") or []),
        )

    best = max(candidates, key=score)
    best["auto_candidates"] = [
        {
            "segmentation_method": item.get("segmentation_method", ""),
            "count": len(item.get("candidates") or []),
            "max_size_px": max((float(c.get("size_px") or 0.0) for c in item.get("candidates") or []), default=0.0),
            "max_size_mm": max((float(c.get("size_mm") or 0.0) for c in item.get("candidates") or []), default=0.0),
            "defect_count": sum(1 for c in item.get("candidates") or [] if c.get("is_defect")),
        }
        for item in candidates
    ]
    best["segmentation_method"] = f"auto:{best.get('segmentation_method', '')}"
    return best


def _detect_dot_components_debug_polarity(
    image_bgr,
    *,
    polarity: str,
    segmentation_method: str,
    diff_threshold: int,
    background_kernel: int,
    min_area: int,
    max_area: int,
    morph_open: int,
    size_metric: str,
    unit_per_px: float,
    defect_threshold: float,
    min_aspect_ratio: float = 0.0,
    edge_margin: int = 0,
    hysteresis_low_threshold: Optional[int] = None,
    hysteresis_high_threshold: Optional[int] = None,
    hysteresis_edge_width_percent: float = 3.0,
    hysteresis_edge_extra_threshold: int = 2,
    hysteresis_second_low_threshold: Optional[int] = None,
    hysteresis_second_high_threshold: Optional[int] = None,
    hysteresis_second_edge_width_percent: float = 9.5,
    hysteresis_second_edge_extra_threshold: int = 2,
    hysteresis_switch_count_threshold: int = 5,
    hysteresis_second_max_count: int = 5,
    hysteresis_edge_suppress_percent: float = 0.0,
    include_visuals: bool = True,
) -> dict:
    """Run dot debug detection for one polarity or auto-select black/white."""
    hysteresis_common = {
        "hysteresis_low_threshold": hysteresis_low_threshold,
        "hysteresis_high_threshold": hysteresis_high_threshold,
        "hysteresis_edge_width_percent": hysteresis_edge_width_percent,
        "hysteresis_edge_extra_threshold": hysteresis_edge_extra_threshold,
        "hysteresis_second_low_threshold": hysteresis_second_low_threshold,
        "hysteresis_second_high_threshold": hysteresis_second_high_threshold,
        "hysteresis_second_edge_width_percent": hysteresis_second_edge_width_percent,
        "hysteresis_second_edge_extra_threshold": hysteresis_second_edge_extra_threshold,
        "hysteresis_switch_count_threshold": hysteresis_switch_count_threshold,
        "hysteresis_second_max_count": hysteresis_second_max_count,
        "hysteresis_edge_suppress_percent": hysteresis_edge_suppress_percent,
    }
    if polarity != "auto":
        result = _detect_dot_components_auto(
            image_bgr,
            polarity=polarity,
            segmentation_method=segmentation_method,
            diff_threshold=diff_threshold,
            background_kernel=background_kernel,
            min_area=min_area,
            max_area=max_area,
            morph_open=morph_open,
            size_metric=size_metric,
            unit_per_px=unit_per_px,
            defect_threshold=defect_threshold,
            min_aspect_ratio=min_aspect_ratio,
            edge_margin=edge_margin,
            **hysteresis_common,
            include_visuals=include_visuals,
        )
        result["detected_polarity"] = polarity
        return result

    detected_by_polarity = []
    for candidate_polarity in ("black", "white"):
        if segmentation_method == "halo" and candidate_polarity != "white":
            continue
        detected = _detect_dot_components_auto(
            image_bgr,
            polarity=candidate_polarity,
            segmentation_method=segmentation_method,
            diff_threshold=diff_threshold,
            background_kernel=background_kernel,
            min_area=min_area,
            max_area=max_area,
            morph_open=morph_open,
            size_metric=size_metric,
            unit_per_px=unit_per_px,
            defect_threshold=defect_threshold,
            min_aspect_ratio=min_aspect_ratio,
            edge_margin=edge_margin,
            **hysteresis_common,
            include_visuals=include_visuals,
        )
        detected["detected_polarity"] = candidate_polarity
        detected_by_polarity.append(detected)

    def score(result: Dict[str, Any]) -> Tuple[int, float, int]:
        top = (result.get("candidates") or [{}])[0]
        return (
            1 if top.get("is_defect") else 0,
            float(top.get("size_px") or 0.0),
            len(result.get("candidates") or []),
        )

    best = max(detected_by_polarity, key=score)
    chosen_polarity = best.get("detected_polarity", "")
    if best.get("segmentation_method", "").startswith("auto:"):
        best["segmentation_method"] = f"auto:{chosen_polarity}:{best['segmentation_method'][5:]}"
    else:
        best["segmentation_method"] = f"auto:{chosen_polarity}:{best.get('segmentation_method', '')}"

    merged_auto_candidates = []
    for item in detected_by_polarity:
        item_polarity = item.get("detected_polarity", "")
        source_candidates = item.get("auto_candidates") or [{
            "segmentation_method": item.get("segmentation_method", ""),
            "count": len(item.get("candidates") or []),
            "max_size_px": max((float(c.get("size_px") or 0.0) for c in item.get("candidates") or []), default=0.0),
            "max_size_mm": max((float(c.get("size_mm") or 0.0) for c in item.get("candidates") or []), default=0.0),
            "defect_count": sum(1 for c in item.get("candidates") or [] if c.get("is_defect")),
        }]
        for candidate in source_candidates:
            merged = dict(candidate)
            merged["polarity"] = item_polarity
            merged["segmentation_method"] = f"{item_polarity}:{candidate.get('segmentation_method', '')}"
            merged_auto_candidates.append(merged)
    best["auto_candidates"] = merged_auto_candidates
    merged_rejected_candidates = []
    for item in detected_by_polarity:
        item_polarity = item.get("detected_polarity", "")
        for rejected in item.get("rejected_candidates") or []:
            merged = dict(rejected)
            merged.setdefault("expected_polarity", item_polarity)
            merged["source_polarity"] = item_polarity
            merged_rejected_candidates.append(merged)
    merged_rejected_candidates.sort(
        key=lambda c: (_as_int(c.get("area_px"), 0), _as_int(c.get("max_diff"), 0)),
        reverse=True,
    )
    best["rejected_candidates"] = merged_rejected_candidates[:100]
    for candidate in best.get("candidates") or []:
        candidate["polarity"] = chosen_polarity
    output = dict(best)
    output["polarity_results"] = {
        item.get("detected_polarity", ""): item
        for item in detected_by_polarity
        if item.get("detected_polarity")
    }
    return output


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_safe_snapshot(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return str(value)


def _within_spec_screen_code(image_name: str, screens: Dict[str, Any]) -> Optional[str]:
    stem = Path(str(image_name or "")).stem
    if stem.startswith("overview_"):
        stem = stem[len("overview_"):]
    if stem in screens:
        return stem
    for code in screens:
        if stem.startswith(code):
            return code
    return "STANDARD" if "STANDARD" in screens else None


def _within_spec_tile_skip_reason(tile: Dict[str, Any]) -> str:
    if not tile.get("is_anomaly"):
        return "not_ng_tile"
    if tile.get("is_bomb"):
        return "bomb"
    if tile.get("is_dust"):
        return "dust"
    if tile.get("is_exclude_zone"):
        return "exclude_zone"
    if tile.get("scratch_filtered"):
        return "scratch_filtered"
    return ""


def _target_tiles_for_within_spec(image: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        tile for tile in (image.get("tiles") or [])
        if not _within_spec_tile_skip_reason(tile)
    ]


def _crop_tile(image_bgr, tile: Dict[str, Any]):
    h, w = image_bgr.shape[:2]
    x1 = max(0, _as_int(tile.get("x"), 0))
    y1 = max(0, _as_int(tile.get("y"), 0))
    x2 = min(w, x1 + max(0, _as_int(tile.get("width"), 0)))
    y2 = min(h, y1 + max(0, _as_int(tile.get("height"), 0)))
    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)
    return image_bgr[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def _dot_rule_limits(rule: Dict[str, Any]) -> Tuple[float, int, int]:
    threshold_mm = _as_float(rule.get("area_threshold_mm"), 0.0)
    screen_limit = _as_int(rule.get("screen_count_limit"), 0)
    tile_limit = _as_int(rule.get("tile_count_threshold"), screen_limit)
    return threshold_mm, screen_limit, tile_limit


def _non_dot_residue_config(dot_cfg: Dict[str, Any]) -> Dict[str, Any]:
    reasons = dot_cfg.get("non_dot_residue_reasons")
    if not isinstance(reasons, list):
        reasons = ["aspect_ratio_below_min", "edge_margin", "area_too_large"]
    reason_set = {
        str(reason).strip()
        for reason in reasons
        if str(reason or "").strip()
    }
    return {
        "enabled": bool(dot_cfg.get("non_dot_residue_enabled", True)),
        "min_area_px": max(0, _as_int(dot_cfg.get("non_dot_residue_min_area_px"), 500)),
        "min_long_side_px": max(0, _as_int(dot_cfg.get("non_dot_residue_min_long_side_px"), 80)),
        "min_long_side_ratio": max(0.0, _as_float(dot_cfg.get("non_dot_residue_min_long_side_ratio"), 0.15)),
        "min_max_diff": max(0, _as_int(dot_cfg.get("non_dot_residue_min_max_diff"), 12)),
        "reasons": sorted(reason_set),
    }


def _non_dot_residue_match(
    rejected: Dict[str, Any],
    cfg: Dict[str, Any],
    crop_box: Tuple[int, int, int, int],
) -> Optional[Dict[str, Any]]:
    if not cfg.get("enabled"):
        return None
    reason = str(rejected.get("reason") or "")
    if reason not in set(cfg.get("reasons") or []):
        return None

    area_px = _as_int(rejected.get("area_px"), 0)
    max_diff = _as_int(rejected.get("max_diff"), 0)
    w = _as_int(rejected.get("w"), 0)
    h = _as_int(rejected.get("h"), 0)
    long_side = max(w, h)
    x1, y1, x2, y2 = crop_box
    crop_long_side = max(1, max(x2 - x1, y2 - y1))
    long_side_ratio = float(long_side / crop_long_side)

    if area_px < _as_int(cfg.get("min_area_px"), 0):
        return None
    if max_diff < _as_int(cfg.get("min_max_diff"), 0):
        return None
    if long_side < _as_int(cfg.get("min_long_side_px"), 0):
        return None
    if long_side_ratio < _as_float(cfg.get("min_long_side_ratio"), 0.0):
        return None

    item = dict(rejected)
    item["long_side_px"] = int(long_side)
    item["long_side_ratio"] = round(long_side_ratio, 4)
    item["thresholds"] = {
        "min_area_px": int(cfg.get("min_area_px", 0)),
        "min_long_side_px": int(cfg.get("min_long_side_px", 0)),
        "min_long_side_ratio": round(float(cfg.get("min_long_side_ratio", 0.0)), 4),
        "min_max_diff": int(cfg.get("min_max_diff", 0)),
    }
    return item


def _attach_runtime_dust_masks_to_within_spec_detail(detail: Dict[str, Any], results: List[Any]) -> Dict[str, Any]:
    """Attach in-memory dust masks to live within-spec detail; masks are not persisted."""
    if not detail or not results:
        return detail

    result_by_name = {}
    for result in results:
        image_path = Path(str(getattr(result, "image_path", "")))
        if image_path.name:
            result_by_name[image_path.name] = result
        if image_path.stem:
            result_by_name[image_path.stem] = result

    for image in detail.get("images") or []:
        image_name = str(image.get("image_name") or image.get("image_path") or "")
        image_path = Path(image_name)
        result = (
            result_by_name.get(image_path.name)
            or result_by_name.get(image_path.stem)
            or result_by_name.get(Path(str(image.get("image_path") or "")).name)
            or result_by_name.get(Path(str(image.get("image_path") or "")).stem)
        )
        if result is None:
            continue

        runtime_tiles = [item[0] for item in (getattr(result, "anomaly_tiles", []) or [])]
        by_id = {getattr(tile, "tile_id", None): tile for tile in runtime_tiles}
        by_xy = {
            (getattr(tile, "x", None), getattr(tile, "y", None)): tile
            for tile in runtime_tiles
        }
        for tile_data in image.get("tiles") or []:
            runtime_tile = by_id.get(tile_data.get("tile_id"))
            if runtime_tile is None:
                runtime_tile = by_xy.get((tile_data.get("x"), tile_data.get("y")))
            if runtime_tile is None:
                continue

            dust_mask = getattr(runtime_tile, "dust_two_stage_dust_mask", None)
            if dust_mask is None:
                dust_mask = getattr(runtime_tile, "dust_mask", None)
            if dust_mask is not None:
                tile_data["_runtime_dust_mask"] = dust_mask
                tile_data["dust_detail_text"] = getattr(runtime_tile, "dust_detail_text", "")
    return detail


def _attach_no_detect_regions_to_within_spec_detail(
    detail: Dict[str, Any],
    inferencer: Any,
    model_id: str = "",
) -> Dict[str, Any]:
    """Attach MARK padding and configured no-detect zones for within-spec OpenCV filtering."""
    if not detail:
        return detail

    config = getattr(inferencer, "config", None) if inferencer is not None else None
    mark_padding = max(0, _as_int(getattr(config, "mark_exclusion_padding_px", 32), 32))
    zones = []
    if inferencer is not None and hasattr(inferencer, "_configured_exclude_regions_for_model"):
        try:
            for region in inferencer._configured_exclude_regions_for_model(model_id) or []:
                zones.append({
                    "enabled": True,
                    "name": getattr(region, "name", "cv_edge_exclude"),
                    "x1": int(getattr(region, "x1", 0)),
                    "y1": int(getattr(region, "y1", 0)),
                    "x2": int(getattr(region, "x2", 0)),
                    "y2": int(getattr(region, "y2", 0)),
                })
        except Exception:
            zones = []

    for image in detail.get("images") or []:
        no_detect_cfg = image.get("within_spec_no_detect")
        if not isinstance(no_detect_cfg, dict):
            no_detect_cfg = {}
        else:
            no_detect_cfg = dict(no_detect_cfg)
        no_detect_cfg["mark_padding_px"] = mark_padding
        no_detect_cfg["cv_edge_exclude_zones"] = zones
        image["within_spec_no_detect"] = no_detect_cfg
    return detail


def _runtime_dust_mask_for_tile(tile: Dict[str, Any], shape: Tuple[int, int]) -> Optional[Any]:
    mask = tile.get("_runtime_dust_mask")
    if mask is None:
        return None

    import cv2
    import numpy as np

    dust_mask = np.asarray(mask)
    if dust_mask.ndim == 3:
        dust_mask = cv2.cvtColor(dust_mask, cv2.COLOR_BGR2GRAY)
    target_h, target_w = int(shape[0]), int(shape[1])
    if dust_mask.shape[:2] != (target_h, target_w):
        dust_mask = cv2.resize(dust_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return dust_mask > 0


def _parse_within_spec_rect(value: Any, *, bbox_is_size: bool = False) -> Optional[Tuple[int, int, int, int]]:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) != 4:
            return None
        try:
            nums = [int(float(p)) for p in parts]
        except ValueError:
            return None
    elif isinstance(value, dict):
        try:
            if {"x1", "y1", "x2", "y2"}.issubset(value.keys()):
                nums = [int(value["x1"]), int(value["y1"]), int(value["x2"]), int(value["y2"])]
                bbox_is_size = False
            else:
                nums = [
                    int(value.get("x", 0)),
                    int(value.get("y", 0)),
                    int(value.get("w", value.get("width", 0))),
                    int(value.get("h", value.get("height", 0))),
                ]
                bbox_is_size = True
        except (TypeError, ValueError):
            return None
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            nums = [int(float(v)) for v in value]
        except (TypeError, ValueError):
            return None
    else:
        return None

    x1, y1, a, b = nums
    x2, y2 = (x1 + a, y1 + b) if bbox_is_size else (a, b)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _within_spec_no_detect_regions_for_image(image: Dict[str, Any]) -> List[Tuple[int, int, int, int, str]]:
    regions: List[Tuple[int, int, int, int, str]] = []
    no_detect_cfg = image.get("within_spec_no_detect") if isinstance(image.get("within_spec_no_detect"), dict) else {}
    mark_padding = max(
        0,
        _as_int(
            no_detect_cfg.get("mark_padding_px", image.get("mark_exclusion_padding_px")),
            0,
        ),
    )

    mark_rect = _parse_within_spec_rect(image.get("mark_bbox"), bbox_is_size=True)
    if mark_rect:
        x1, y1, x2, y2 = mark_rect
        regions.append((x1 - mark_padding, y1 - mark_padding, x2 + mark_padding, y2 + mark_padding, "mark"))

    for raw in image.get("mark_exclusion_regions") or []:
        rect = _parse_within_spec_rect(raw)
        if not rect:
            continue
        x1, y1, x2, y2 = rect
        regions.append((x1 - mark_padding, y1 - mark_padding, x2 + mark_padding, y2 + mark_padding, "mark"))

    zone_values = []
    zone_values.extend(no_detect_cfg.get("cv_edge_exclude_zones") or [])
    zone_values.extend(image.get("cv_edge_exclude_zones") or [])
    for raw in zone_values:
        if isinstance(raw, dict) and raw.get("enabled") is False:
            continue
        zone_is_size = False
        if isinstance(raw, dict):
            zone_is_size = not {"x1", "y1", "x2", "y2"}.issubset(raw.keys())
        rect = _parse_within_spec_rect(raw, bbox_is_size=zone_is_size)
        if rect:
            regions.append((*rect, "exclude_zone"))

    return regions


def _within_spec_no_detect_mask_for_tile(
    image: Dict[str, Any],
    tile: Dict[str, Any],
    shape: Tuple[int, int],
    crop_box: Tuple[int, int, int, int],
):
    import cv2
    import numpy as np

    regions = _within_spec_no_detect_regions_for_image(image)
    if not regions:
        return None

    target_h, target_w = int(shape[0]), int(shape[1])
    if target_h <= 0 or target_w <= 0:
        return None
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    crop_w = max(1, crop_x2 - crop_x1)
    crop_h = max(1, crop_y2 - crop_y1)

    mask = np.zeros((target_h, target_w), dtype=np.uint8)
    for rx1, ry1, rx2, ry2, _source in regions:
        ix1 = max(crop_x1, int(rx1))
        iy1 = max(crop_y1, int(ry1))
        ix2 = min(crop_x2, int(rx2))
        iy2 = min(crop_y2, int(ry2))
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        mx1 = max(0, min(target_w, int(np.floor((ix1 - crop_x1) * target_w / crop_w))))
        my1 = max(0, min(target_h, int(np.floor((iy1 - crop_y1) * target_h / crop_h))))
        mx2 = max(0, min(target_w, int(np.ceil((ix2 - crop_x1) * target_w / crop_w))))
        my2 = max(0, min(target_h, int(np.ceil((iy2 - crop_y1) * target_h / crop_h))))
        if mx2 > mx1 and my2 > my1:
            mask[my1:my2, mx1:mx2] = 255

    if not np.any(mask > 0):
        return None
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def _candidate_dust_overlap(candidate: Dict[str, Any], dust_mask) -> Optional[Dict[str, Any]]:
    import numpy as np

    if dust_mask is None:
        return None
    mask_h, mask_w = dust_mask.shape[:2]
    x = max(0, min(mask_w - 1, _as_int(candidate.get("x"), 0)))
    y = max(0, min(mask_h - 1, _as_int(candidate.get("y"), 0)))
    w = max(1, _as_int(candidate.get("w"), 1))
    h = max(1, _as_int(candidate.get("h"), 1))
    x2 = max(x + 1, min(mask_w, x + w))
    y2 = max(y + 1, min(mask_h, y + h))
    patch = dust_mask[y:y2, x:x2]
    if patch.size <= 0:
        return None

    overlap_ratio = float(np.count_nonzero(patch) / patch.size)
    cx = max(0, min(mask_w - 1, int(round(_as_float(candidate.get("center_x"), x + w / 2.0)))))
    cy = max(0, min(mask_h - 1, int(round(_as_float(candidate.get("center_y"), y + h / 2.0)))))
    center_in_dust = bool(dust_mask[cy, cx])
    if not center_in_dust and overlap_ratio < 0.25:
        return None

    rejected = {
        k: v for k, v in candidate.items()
        if k not in {"is_defect", "id"}
    }
    rejected.update({
        "reason": "dust_mask_overlap",
        "dust_overlap_ratio": round(overlap_ratio, 4),
        "center_in_dust": center_in_dust,
    })
    return rejected


def _candidate_no_detect_overlap(candidate: Dict[str, Any], no_detect_mask) -> Optional[Dict[str, Any]]:
    import numpy as np

    if no_detect_mask is None:
        return None
    mask_h, mask_w = no_detect_mask.shape[:2]
    x = max(0, min(mask_w - 1, _as_int(candidate.get("x"), 0)))
    y = max(0, min(mask_h - 1, _as_int(candidate.get("y"), 0)))
    w = max(1, _as_int(candidate.get("w"), 1))
    h = max(1, _as_int(candidate.get("h"), 1))
    x2 = max(x + 1, min(mask_w, x + w))
    y2 = max(y + 1, min(mask_h, y + h))
    patch = no_detect_mask[y:y2, x:x2]
    if patch.size <= 0:
        return None

    overlap_ratio = float(np.count_nonzero(patch) / patch.size)
    cx = max(0, min(mask_w - 1, int(round(_as_float(candidate.get("center_x"), x + w / 2.0)))))
    cy = max(0, min(mask_h - 1, int(round(_as_float(candidate.get("center_y"), y + h / 2.0)))))
    center_in_no_detect = bool(no_detect_mask[cy, cx])
    if not center_in_no_detect and overlap_ratio < 0.25:
        return None

    rejected = {
        k: v for k, v in candidate.items()
        if k not in {"is_defect", "id"}
    }
    rejected.update({
        "reason": "no_detect_mask_overlap",
        "no_detect_overlap_ratio": round(overlap_ratio, 4),
        "center_in_no_detect": center_in_no_detect,
    })
    return rejected


def _remove_dust_overlap_candidates(detected: Dict[str, Any], dust_mask) -> Dict[str, Any]:
    if dust_mask is None:
        return detected

    kept = []
    rejected = []
    original_rejected = detected.get("rejected_candidates") or []
    original_count = len(detected.get("candidates") or [])
    dust_filtered_rejected_count = 0
    for candidate in original_rejected:
        dust_rejected = _candidate_dust_overlap(candidate, dust_mask)
        if dust_rejected:
            dust_rejected["source_reason"] = candidate.get("reason", "")
            rejected.append(dust_rejected)
            dust_filtered_rejected_count += 1
        else:
            rejected.append(candidate)
    for candidate in detected.get("candidates") or []:
        dust_rejected = _candidate_dust_overlap(candidate, dust_mask)
        if dust_rejected:
            rejected.append(dust_rejected)
        else:
            kept.append(candidate)

    for idx, candidate in enumerate(kept, 1):
        candidate["id"] = idx
    detected["candidates"] = kept
    detected["rejected_candidates"] = rejected[:50]
    detected["dust_mask_filtered_count"] = original_count - len(kept)
    detected["dust_mask_filtered_rejected_count"] = dust_filtered_rejected_count
    return detected


def _remove_no_detect_overlap_candidates(detected: Dict[str, Any], no_detect_mask) -> Dict[str, Any]:
    if no_detect_mask is None:
        return detected

    kept = []
    rejected = []
    original_rejected = detected.get("rejected_candidates") or []
    original_count = len(detected.get("candidates") or [])
    no_detect_filtered_rejected_count = 0
    for candidate in original_rejected:
        no_detect_rejected = _candidate_no_detect_overlap(candidate, no_detect_mask)
        if no_detect_rejected:
            no_detect_rejected["source_reason"] = candidate.get("reason", "")
            rejected.append(no_detect_rejected)
            no_detect_filtered_rejected_count += 1
        else:
            rejected.append(candidate)
    for candidate in detected.get("candidates") or []:
        no_detect_rejected = _candidate_no_detect_overlap(candidate, no_detect_mask)
        if no_detect_rejected:
            rejected.append(no_detect_rejected)
        else:
            kept.append(candidate)

    for idx, candidate in enumerate(kept, 1):
        candidate["id"] = idx
    detected["candidates"] = kept
    detected["rejected_candidates"] = rejected[:50]
    detected["no_detect_mask_filtered_count"] = original_count - len(kept)
    detected["no_detect_mask_filtered_rejected_count"] = no_detect_filtered_rejected_count
    return detected


def _format_within_spec_panel_summary(detail: Dict[str, Any], fallback: str = "") -> str:
    parts = _format_within_spec_panel_summary_parts(detail)
    return "；".join(parts) if parts else fallback


def _format_within_spec_panel_summary_parts(detail: Dict[str, Any]) -> List[str]:
    parts = []

    def cmp_text(label: str, value: Any, limit: Any, ok: bool, suffix: str = "") -> str:
        op = "<=" if ok else ">"
        status = "OK" if ok else "NG"
        return f"{label} {value}{suffix} {op} {limit}{suffix} [{status}]"

    for item in (detail.get("panel_totals") or []):
        screen = item.get("screen") or "UNKNOWN"
        dot_label = item.get("dot_label") or item.get("dot_type") or "點"
        max_size = _as_float(item.get("max_size_mm"), 0.0)
        threshold = _as_float(item.get("threshold_mm"), 0.0)
        total_count = _as_int(item.get("total_count"), 0)
        screen_limit = _as_int(item.get("screen_count_limit"), 0)
        max_tile_count = _as_int(item.get("max_tile_count"), 0)
        tile_limit = _as_int(item.get("tile_count_threshold"), 0)
        size_ok = max_size <= threshold
        screen_count_ok = total_count <= screen_limit
        tile_count_ok = max_tile_count <= tile_limit
        max_size_text = f"{max_size:.4f}"
        threshold_text = f"{threshold:.4f}"
        status = "OK" if item.get("within") else "NG"
        parts.append(
            f"{screen} {dot_label}："
            f"{cmp_text('最大尺寸', max_size_text, threshold_text, size_ok, 'mm')}；"
            f"{cmp_text('畫面總點數', total_count, screen_limit, screen_count_ok)}；"
            f"{cmp_text('單Tile最大點數', max_tile_count, tile_limit, tile_count_ok)}；"
            f"結果={status}"
        )

    residues_by_screen: Dict[str, List[Dict[str, Any]]] = {}
    for item in detail.get("non_dot_residues") or []:
        screen = item.get("screen") or "UNKNOWN"
        residues_by_screen.setdefault(screen, []).append(item)
    for screen, items in residues_by_screen.items():
        max_area = max((_as_int(item.get("area_px"), 0) for item in items), default=0)
        max_long_side = max((_as_int(item.get("long_side_px"), 0) for item in items), default=0)
        max_diff = max((_as_int(item.get("max_diff"), 0) for item in items), default=0)
        parts.append(
            f"{screen} 非點狀殘留："
            f"數量 {len(items)} > 0 [NG]；"
            f"最大面積 {max_area}px；最大長邊 {max_long_side}px；"
            f"最大diff {max_diff}；結果=NG"
        )

    missed_by_screen: Dict[str, List[Dict[str, Any]]] = {}
    for item in detail.get("missed_dot_tiles") or []:
        screen = item.get("screen") or "UNKNOWN"
        missed_by_screen.setdefault(screen, []).append(item)
    for screen, items in missed_by_screen.items():
        aoi_count = sum(1 for item in items if item.get("is_aoi_coord"))
        label = "AOI點未檢出" if aoi_count else "NG Tile未檢出點"
        details = []
        for item in items[:5]:
            tile_id = item.get("tile_id")
            tile_text = f"tile {tile_id}" if tile_id is not None else "tile ?"
            if item.get("is_aoi_coord"):
                ax = _as_int(item.get("aoi_product_x"), -1)
                ay = _as_int(item.get("aoi_product_y"), -1)
                if ax >= 0 and ay >= 0:
                    tile_text = f"{tile_text} AOI({ax},{ay})"
            details.append(tile_text)
        if len(items) > 5:
            details.append(f"...共{len(items)}個")
        detail_text = f"；{', '.join(details)}" if details else ""
        parts.append(
            f"{screen} {label}："
            f"數量 {len(items)} > 0 [NG]{detail_text}；結果=NG"
        )
    return parts


def _format_within_spec_inference_note(within_spec_info: Dict[str, Any], detail_url: str) -> str:
    detail = within_spec_info.get("detail") or {}
    summary = detail.get("panel_summary") or {}
    rule_selection = detail.get("rule_selection") or {}
    status = within_spec_info.get("status", "unknown")
    reason = within_spec_info.get("reason", "")
    result_text = (
        "符合規格內，最終判定 OK-i"
        if within_spec_info.get("converted")
        else f"已執行規格內檢查，結果={status}"
    )

    lines = [
        f"[WITHIN_SPEC_INFERENCE] 原始 AI=NG，{result_text}",
        (
            f"  matched_machine={rule_selection.get('matched_machine_key') or 'N/A'}；"
            f"target_tiles={summary.get('target_tile_count', 0)}；"
            f"evaluated_tiles={summary.get('evaluated_tile_count', 0)}"
        ),
    ]

    panel_parts = _format_within_spec_panel_summary_parts(detail)
    if panel_parts:
        lines.extend(f"  - {part}" for part in panel_parts)
    elif reason:
        lines.append(f"  reason：{reason}")
    lines.append(f"  明細：{detail_url}")
    return "\n".join(lines)


def _within_spec_machine_candidates(detail: Dict[str, Any], machine_id: str = "") -> List[str]:
    candidates: List[str] = []
    for value in (
        detail.get("model_id"),
        machine_id,
        detail.get("machine_id"),
        detail.get("machine_no"),
    ):
        key = str(value or "").strip()
        if key and key not in candidates:
            candidates.append(key)
    return candidates


def _select_within_spec_machine_rules(
    detail: Dict[str, Any],
    rules: Dict[str, Any],
    machine_id: str = "",
) -> Tuple[str, Dict[str, Any], List[str], bool]:
    candidates = _within_spec_machine_candidates(detail, machine_id)
    for key in candidates:
        machine_rules = rules.get(key)
        if isinstance(machine_rules, dict):
            return key, machine_rules, candidates, False

    default_rules = rules.get("default")
    if isinstance(default_rules, dict):
        return "default", default_rules, candidates, bool(candidates)
    return "", {}, candidates, False


def _save_within_spec_dot_visuals(
    *,
    tile_crop,
    processed_crop,
    chosen: Dict[str, Any],
    dot_cfg: Dict[str, Any],
    runtime_dust_mask=None,
    no_detect_mask=None,
    non_dot_residues: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[Path],
    url_prefix: str,
    image_name: str,
    tile_id: Any,
    crop_box: Tuple[int, int, int, int],
) -> Optional[Dict[str, Any]]:
    if not output_dir or not url_prefix:
        return None

    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_image = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(str(image_name or "image")).stem)[:80]
    safe_tile = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(tile_id if tile_id is not None else "tile"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = f"within_spec_{safe_image}_tile{safe_tile}_{chosen['dot_type']}_{ts}"

    detected = _detect_dot_components_auto(
        processed_crop,
        polarity=chosen["polarity"],
        segmentation_method=str(dot_cfg.get("segmentation_method") or "background_diff"),
        diff_threshold=_as_int(dot_cfg.get("diff_threshold"), 4),
        background_kernel=_odd_kernel(_as_int(dot_cfg.get("background_kernel"), 33)),
        min_area=max(1, _as_int(dot_cfg.get("min_area_px"), 2)),
        max_area=max(0, _as_int(dot_cfg.get("max_area_px"), 50000)),
        morph_open=max(0, _as_int(dot_cfg.get("morph_open"), 0)),
        size_metric=str(dot_cfg.get("size_metric") or "bbox_diagonal"),
        unit_per_px=max(0.0, _as_float(dot_cfg.get("unit_per_px"), DOT_RULER_MM_PER_PX)),
        defect_threshold=chosen["threshold_mm"],
        min_aspect_ratio=max(0.0, _as_float(dot_cfg.get("min_aspect_ratio"), 0.45)),
        edge_margin=max(0, _as_int(dot_cfg.get("edge_margin_px"), 4)),
        **_dot_hysteresis_kwargs(dot_cfg),
        include_visuals=True,
    )
    detected = _remove_dust_overlap_candidates(detected, runtime_dust_mask)
    detected = _remove_no_detect_overlap_candidates(detected, no_detect_mask)
    dust_mask_color = None
    dust_overlay = None
    if runtime_dust_mask is not None:
        import numpy as np

        dust_pixels = np.asarray(runtime_dust_mask) > 0
        dust_mask_u8 = (dust_pixels.astype("uint8") * 255)
        dust_mask_color = cv2.cvtColor(dust_mask_u8, cv2.COLOR_GRAY2BGR)
        if np.any(dust_pixels):
            dust_mask_color[dust_pixels] = (0, 255, 255)
        base_overlay = detected.get("overlay")
        if base_overlay is not None:
            dust_overlay = base_overlay.copy()
            if np.any(dust_pixels):
                yellow = np.zeros_like(dust_overlay)
                yellow[:, :] = (0, 255, 255)
                dust_overlay[dust_pixels] = cv2.addWeighted(
                    dust_overlay[dust_pixels],
                    0.45,
                    yellow[dust_pixels],
                    0.55,
                    0,
                )
    no_detect_mask_color = None
    no_detect_overlay = None
    if no_detect_mask is not None:
        import numpy as np

        no_detect_pixels = np.asarray(no_detect_mask) > 0
        no_detect_mask_u8 = (no_detect_pixels.astype("uint8") * 255)
        no_detect_mask_color = cv2.cvtColor(no_detect_mask_u8, cv2.COLOR_GRAY2BGR)
        if np.any(no_detect_pixels):
            no_detect_mask_color[no_detect_pixels] = (255, 0, 255)
        base_overlay = dust_overlay if dust_overlay is not None else detected.get("overlay")
        if base_overlay is not None:
            no_detect_overlay = base_overlay.copy()
            if np.any(no_detect_pixels):
                magenta = np.zeros_like(no_detect_overlay)
                magenta[:, :] = (255, 0, 255)
                no_detect_overlay[no_detect_pixels] = cv2.addWeighted(
                    no_detect_overlay[no_detect_pixels],
                    0.45,
                    magenta[no_detect_pixels],
                    0.55,
                    0,
                )

    visual_residues = [
        {k: v for k, v in residue.items() if k != "_key"}
        for residue in (non_dot_residues or [])
    ]
    if visual_residues and detected.get("overlay") is not None:
        h_img, w_img = detected["overlay"].shape[:2]
        for idx, residue in enumerate(visual_residues, 1):
            x = max(0, min(w_img - 1, _as_int(residue.get("x"), 0)))
            y = max(0, min(h_img - 1, _as_int(residue.get("y"), 0)))
            w = max(1, _as_int(residue.get("w"), 1))
            h = max(1, _as_int(residue.get("h"), 1))
            x2 = max(x + 1, min(w_img - 1, x + w))
            y2 = max(y + 1, min(h_img - 1, y + h))
            label_y = max(13, min(h_img - 5, y + 14))
            label = f"NG residue #{idx}"
            for key in ("overlay", "mask_color", "diff_color"):
                canvas = detected.get(key)
                if canvas is None:
                    continue
                cv2.rectangle(canvas, (x, y), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    canvas,
                    label,
                    (x + 4, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

    files = {
        "crop_url": f"{prefix}_crop.png",
        "preprocessed_url": f"{prefix}_preprocessed.png",
        "overlay_url": f"{prefix}_overlay.png",
        "mask_url": f"{prefix}_mask.png",
        "diff_url": f"{prefix}_diff.png",
    }
    if dust_mask_color is not None:
        files["dust_mask_url"] = f"{prefix}_dust_mask.png"
    if dust_overlay is not None:
        files["dust_overlay_url"] = f"{prefix}_dust_overlay.png"
    if no_detect_mask_color is not None:
        files["no_detect_mask_url"] = f"{prefix}_no_detect_mask.png"
    if no_detect_overlay is not None:
        files["no_detect_overlay_url"] = f"{prefix}_no_detect_overlay.png"
    cv2.imwrite(str(output_dir / files["crop_url"]), tile_crop)
    cv2.imwrite(str(output_dir / files["preprocessed_url"]), processed_crop)
    cv2.imwrite(str(output_dir / files["overlay_url"]), detected["overlay"])
    cv2.imwrite(str(output_dir / files["mask_url"]), detected["mask_color"])
    cv2.imwrite(str(output_dir / files["diff_url"]), detected["diff_color"])
    if dust_mask_color is not None:
        cv2.imwrite(str(output_dir / files["dust_mask_url"]), dust_mask_color)
    if dust_overlay is not None:
        cv2.imwrite(str(output_dir / files["dust_overlay_url"]), dust_overlay)
    if no_detect_mask_color is not None:
        cv2.imwrite(str(output_dir / files["no_detect_mask_url"]), no_detect_mask_color)
    if no_detect_overlay is not None:
        cv2.imwrite(str(output_dir / files["no_detect_overlay_url"]), no_detect_overlay)

    return {
        "image_name": image_name,
        "tile_id": tile_id,
        "dot_type": chosen["dot_type"],
        "dot_label": chosen["label"],
        "segmentation_method": detected.get("segmentation_method", ""),
        "auto_candidates": detected.get("auto_candidates", []),
        "fallback_source": chosen.get("fallback_source", ""),
        "crop_box": crop_box,
        "count": len(detected["candidates"]),
        "max_size_mm": max((_as_float(c.get("size_mm"), 0.0) for c in detected["candidates"]), default=0.0),
        "threshold_mm": chosen["threshold_mm"],
        "candidates": detected["candidates"][:30],
        "non_dot_residues": visual_residues[:20],
        "dust_mask_filtered_count": _as_int(detected.get("dust_mask_filtered_count"), 0),
        "dust_mask_filtered_rejected_count": _as_int(detected.get("dust_mask_filtered_rejected_count"), 0),
        "no_detect_mask_filtered_count": _as_int(detected.get("no_detect_mask_filtered_count"), 0),
        "no_detect_mask_filtered_rejected_count": _as_int(detected.get("no_detect_mask_filtered_rejected_count"), 0),
        "thresholds": detected.get("thresholds", {}),
        "urls": {key: f"{url_prefix}/{filename}" for key, filename in files.items()},
    }


def _evaluate_within_spec_suggestion_detail(
    detail: Dict[str, Any],
    rules: Dict[str, Any],
    machine_id: str = "",
    visual_output_dir: Optional[Path] = None,
    visual_url_prefix: str = "",
    rotate_180: bool = False,
) -> Dict[str, Any]:
    """Evaluate within-spec suggestion on NG tiles only and collect traceable steps."""
    import cv2

    steps: List[Dict[str, Any]] = []

    def add_step(message: str, **data):
        if len(steps) < 1000:
            item = {"message": message}
            item.update(data)
            steps.append(item)

    result = {
        "suggestion": None,
        "matches": [],
        "steps": steps,
        "skipped_tiles": {},
        "target_tile_count": 0,
        "evaluated_tile_count": 0,
        "panel_totals": [],
        "panel_summary": {},
        "candidate_summary": {
            "raw_candidate_count": 0,
            "final_candidate_count": 0,
            "dust_mask_filtered_count": 0,
            "no_detect_mask_filtered_count": 0,
            "dust_mask_filtered_rejected_count": 0,
            "no_detect_mask_filtered_rejected_count": 0,
        },
        "visuals": [],
        "non_dot_residues": [],
        "missed_dot_tiles": [],
        "rule_selection": {},
        "parameter_snapshot": {},
    }
    if not detail or not isinstance(rules, dict):
        add_step("缺少推論 detail 或規格內規則，停止判定")
        return result

    machine_key, machine_rules, machine_candidates, fallback_used = _select_within_spec_machine_rules(
        detail,
        rules,
        machine_id,
    )
    result["rule_selection"] = {
        "candidate_keys": machine_candidates,
        "matched_machine_key": machine_key or "",
        "fallback_used": fallback_used,
    }
    add_step(
        "選擇規格內設定",
        candidate_keys=machine_candidates,
        matched_machine=machine_key or "",
        fallback_default=fallback_used,
    )
    screens = machine_rules.get("screens") or {}
    dot_cfg = machine_rules.get("dot_detection") or {}
    preprocess_params = dot_cfg.get("preprocess_params") if isinstance(dot_cfg.get("preprocess_params"), dict) else DOT_PREPROCESS_PARAMS
    effective_segmentation_method = str(dot_cfg.get("segmentation_method") or "background_diff").strip().lower()
    if effective_segmentation_method not in ("background_diff", "hysteresis", "morph_hat", "adaptive_mean", "halo", "auto", "off"):
        effective_segmentation_method = "background_diff"
    effective_hysteresis = _dot_hysteresis_kwargs(dot_cfg)
    non_dot_cfg = _non_dot_residue_config(dot_cfg)
    result["parameter_snapshot"] = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "input_rotation_degrees": 180 if rotate_180 else 0,
        "candidate_keys": machine_candidates,
        "matched_machine_key": machine_key or "",
        "fallback_used": fallback_used,
        "full_rules": _json_safe_snapshot(rules),
        "matched_machine_rules": _json_safe_snapshot(machine_rules),
        "dot_detection": {
            "raw": _json_safe_snapshot(dot_cfg),
            "effective": {
                "diff_threshold": _as_int(dot_cfg.get("diff_threshold"), 4),
                "segmentation_method": effective_segmentation_method,
                **effective_hysteresis,
                "background_kernel": _odd_kernel(_as_int(dot_cfg.get("background_kernel"), 33)),
                "min_area_px": max(1, _as_int(dot_cfg.get("min_area_px"), 2)),
                "max_area_px": max(0, _as_int(dot_cfg.get("max_area_px"), 50000)),
                "morph_open": max(0, _as_int(dot_cfg.get("morph_open"), 0)),
                "min_aspect_ratio": max(0.0, _as_float(dot_cfg.get("min_aspect_ratio"), 0.45)),
                "edge_margin_px": max(0, _as_int(dot_cfg.get("edge_margin_px"), 4)),
                "non_dot_residue": _json_safe_snapshot(non_dot_cfg),
                "size_metric": str(dot_cfg.get("size_metric") or "bbox_diagonal"),
                "unit_per_px": max(0.0, _as_float(dot_cfg.get("unit_per_px"), DOT_RULER_MM_PER_PX)),
            },
        },
        "preprocess": {
            "method": str(dot_cfg.get("preprocess_method") or DOT_PREPROCESS_METHOD),
            "params": _json_safe_snapshot(preprocess_params),
        },
        "screen_rules_used": {},
    }
    if effective_segmentation_method == "off":
        add_step("規格內判定已關閉，停止判定", segmentation_method=effective_segmentation_method)
        return result
    if not screens:
        add_step("找不到 screen 規則，停止判定", machine=machine_key or "default")
        return result

    dot_types = (
        ("black_dot", "black", "黑點"),
        ("white_dot", "white", "白點"),
    )
    aggregates: Dict[Tuple[str, str], Dict[str, Any]] = {}
    non_dot_residues: List[Dict[str, Any]] = []
    non_dot_keys = set()

    for image in detail.get("images") or []:
        screen_code = _within_spec_screen_code(image.get("image_name") or image.get("image_path"), screens)
        if not screen_code:
            add_step("略過圖片：找不到對應 screen 規則", image=image.get("image_name") or image.get("image_path") or "")
            continue
        screen_rules = screens.get(screen_code) or {}
        result["parameter_snapshot"]["screen_rules_used"][screen_code] = _json_safe_snapshot(screen_rules)

        target_tiles = []
        skipped = {}
        for tile in image.get("tiles") or []:
            reason = _within_spec_tile_skip_reason(tile)
            if reason:
                skipped[reason] = skipped.get(reason, 0) + 1
                if reason == "bomb":
                    add_step(
                        "略過 bomb tile",
                        image=image.get("image_name") or "",
                        tile_id=tile.get("tile_id"),
                        defect_code=tile.get("bomb_code") or tile.get("aoi_defect_code") or "",
                    )
                continue
            target_tiles.append(tile)
        for reason, count in skipped.items():
            result["skipped_tiles"][reason] = result["skipped_tiles"].get(reason, 0) + count

        if not target_tiles:
            add_step(
                "圖片沒有需要判定的 NG tile",
                image=image.get("image_name") or "",
                skipped=skipped,
            )
            continue
        result["target_tile_count"] += len(target_tiles)

        image_path = Path(str(image.get("image_path") or ""))
        if not image_path.is_file():
            add_step("原圖不存在，略過圖片", image=image.get("image_name") or "", path=str(image_path))
            continue
        image_bgr = read_detection_image(image_path, cv2.IMREAD_COLOR, rotate_180)
        if image_bgr is None:
            add_step("原圖讀取失敗，略過圖片", image=image.get("image_name") or "", path=str(image_path))
            continue

        add_step(
            "開始判定圖片",
            image=image.get("image_name") or image_path.name,
            screen=screen_code,
            target_tiles=len(target_tiles),
            skipped=skipped,
        )

        for tile in target_tiles:
            tile_crop, crop_box = _crop_tile(image_bgr, tile)
            if tile_crop is None:
                add_step(
                    "略過 tile：座標無效",
                    image=image.get("image_name") or "",
                    tile_id=tile.get("tile_id"),
                    crop_box=crop_box,
                )
                continue

            processed_crop, _ = _preprocess_dot_image_for_detection(
                tile_crop,
                method=str(dot_cfg.get("preprocess_method") or DOT_PREPROCESS_METHOD),
                params=preprocess_params,
            )
            runtime_dust_mask = _runtime_dust_mask_for_tile(tile, processed_crop.shape[:2])
            no_detect_mask = _within_spec_no_detect_mask_for_tile(
                image,
                tile,
                processed_crop.shape[:2],
                crop_box,
            )

            detections = []
            for dot_type, polarity, label in dot_types:
                rule = screen_rules.get(dot_type) or {}
                threshold_mm, screen_limit, tile_limit = _dot_rule_limits(rule)
                if rule.get("enabled") is False:
                    continue
                if threshold_mm <= 0 or screen_limit < 0 or tile_limit < 0:
                    add_step(
                        "略過點類規則：門檻或數量設定無效",
                        screen=screen_code,
                        dot_type=dot_type,
                        threshold_mm=threshold_mm,
                        screen_limit=screen_limit,
                        tile_limit=tile_limit,
                    )
                    continue

                detected = _detect_dot_components_auto(
                    processed_crop,
                    polarity=polarity,
                    segmentation_method=str(dot_cfg.get("segmentation_method") or "background_diff"),
                    diff_threshold=_as_int(dot_cfg.get("diff_threshold"), 4),
                    background_kernel=_odd_kernel(_as_int(dot_cfg.get("background_kernel"), 33)),
                    min_area=max(1, _as_int(dot_cfg.get("min_area_px"), 2)),
                    max_area=max(0, _as_int(dot_cfg.get("max_area_px"), 50000)),
                    morph_open=max(0, _as_int(dot_cfg.get("morph_open"), 0)),
                    size_metric=str(dot_cfg.get("size_metric") or "bbox_diagonal"),
                    unit_per_px=max(0.0, _as_float(dot_cfg.get("unit_per_px"), DOT_RULER_MM_PER_PX)),
                    defect_threshold=threshold_mm,
                    min_aspect_ratio=max(0.0, _as_float(dot_cfg.get("min_aspect_ratio"), 0.45)),
                    edge_margin=max(0, _as_int(dot_cfg.get("edge_margin_px"), 4)),
                    **_dot_hysteresis_kwargs(dot_cfg),
                    include_visuals=False,
                )
                detected = _remove_dust_overlap_candidates(detected, runtime_dust_mask)
                detected = _remove_no_detect_overlap_candidates(detected, no_detect_mask)
                candidates = detected["candidates"]
                max_size_mm = max((_as_float(c.get("size_mm"), 0.0) for c in candidates), default=0.0)
                detections.append({
                    "dot_type": dot_type,
                    "polarity": polarity,
                    "label": label,
                    "segmentation_method": detected.get("segmentation_method", ""),
                    "auto_candidates": detected.get("auto_candidates", []),
                    "thresholds": detected.get("thresholds", {}),
                    "rule": rule,
                    "threshold_mm": threshold_mm,
                    "screen_limit": screen_limit,
                    "tile_limit": tile_limit,
                    "candidates": candidates,
                    "rejected_candidates": detected.get("rejected_candidates", []),
                    "dust_mask_filtered_count": detected.get("dust_mask_filtered_count", 0),
                    "dust_mask_filtered_rejected_count": detected.get("dust_mask_filtered_rejected_count", 0),
                    "no_detect_mask_filtered_count": detected.get("no_detect_mask_filtered_count", 0),
                    "no_detect_mask_filtered_rejected_count": detected.get("no_detect_mask_filtered_rejected_count", 0),
                    "count": len(candidates),
                    "max_size_mm": max_size_mm,
                })

            if not detections:
                add_step(
                    "略過 tile：沒有可用的黑點/白點規則",
                    image=image.get("image_name") or "",
                    tile_id=tile.get("tile_id"),
                    crop_box=crop_box,
                )
                continue

            chosen = max(detections, key=lambda d: (d["max_size_mm"], d["count"]))
            tile_candidate_summary = {
                "final_candidate_count": sum(_as_int(d.get("count"), 0) for d in detections),
                "dust_mask_filtered_count": sum(_as_int(d.get("dust_mask_filtered_count"), 0) for d in detections),
                "no_detect_mask_filtered_count": sum(_as_int(d.get("no_detect_mask_filtered_count"), 0) for d in detections),
                "dust_mask_filtered_rejected_count": sum(_as_int(d.get("dust_mask_filtered_rejected_count"), 0) for d in detections),
                "no_detect_mask_filtered_rejected_count": sum(_as_int(d.get("no_detect_mask_filtered_rejected_count"), 0) for d in detections),
            }
            tile_candidate_summary["raw_candidate_count"] = (
                tile_candidate_summary["final_candidate_count"]
                + tile_candidate_summary["dust_mask_filtered_count"]
                + tile_candidate_summary["no_detect_mask_filtered_count"]
            )
            dust_only = (
                tile_candidate_summary["raw_candidate_count"] > 0
                and tile_candidate_summary["final_candidate_count"] == 0
                and tile_candidate_summary["dust_mask_filtered_count"]
                == tile_candidate_summary["raw_candidate_count"]
            )
            if dust_only:
                chosen = max(
                    detections,
                    key=lambda d: _as_int(d.get("dust_mask_filtered_count"), 0),
                )
            for key, value in tile_candidate_summary.items():
                result["candidate_summary"][key] = result["candidate_summary"].get(key, 0) + value
            tile_non_dot_residues: List[Dict[str, Any]] = []
            tile_non_dot_keys = set()
            detection_summary = {
                d["dot_type"]: {
                    "count": d["count"],
                    "max_size_mm": round(float(d["max_size_mm"]), 4),
                    "segmentation_method": d["segmentation_method"],
                    "rejected_count": len(d.get("rejected_candidates") or []),
                    "dust_mask_filtered_count": _as_int(d.get("dust_mask_filtered_count"), 0),
                    "dust_mask_filtered_rejected_count": _as_int(d.get("dust_mask_filtered_rejected_count"), 0),
                    "no_detect_mask_filtered_count": _as_int(d.get("no_detect_mask_filtered_count"), 0),
                    "no_detect_mask_filtered_rejected_count": _as_int(d.get("no_detect_mask_filtered_rejected_count"), 0),
                    "thresholds": d.get("thresholds", {}),
                }
                for d in detections
            }
            rejected_summary = {
                d["dot_type"]: (d.get("rejected_candidates") or [])[:12]
                for d in detections
                if d.get("rejected_candidates")
            }
            if rejected_summary:
                add_step(
                    "點候選被過濾",
                    image=image.get("image_name") or "",
                    tile_id=tile.get("tile_id"),
                    crop_box=crop_box,
                    rejected=rejected_summary,
                )
                for detection in detections:
                    dot_type = detection["dot_type"]
                    for rejected in detection.get("rejected_candidates") or []:
                        matched = _non_dot_residue_match(rejected, non_dot_cfg, crop_box)
                        if not matched:
                            continue
                        key = (
                            image.get("image_name") or image_path.name,
                            tile.get("tile_id"),
                            matched.get("reason"),
                            matched.get("x"),
                            matched.get("y"),
                            matched.get("w"),
                            matched.get("h"),
                        )
                        if key in non_dot_keys:
                            for item in non_dot_residues:
                                if item.get("_key") == key and dot_type not in item["dot_types"]:
                                    item["dot_types"].append(dot_type)
                                if item.get("_key") == key and key not in tile_non_dot_keys:
                                    tile_non_dot_keys.add(key)
                                    tile_non_dot_residues.append(item)
                            continue
                        non_dot_keys.add(key)
                        matched.update({
                            "_key": key,
                            "image": image.get("image_name") or image_path.name,
                            "screen": screen_code,
                            "tile_id": tile.get("tile_id"),
                            "crop_box": crop_box,
                            "dot_types": [dot_type],
                        })
                        non_dot_residues.append(matched)
                        tile_non_dot_keys.add(key)
                        tile_non_dot_residues.append(matched)
            if dust_only:
                add_step(
                    "tile 點候選皆落在灰塵遮罩：以 0 點納入 Panel 判定",
                    image=image.get("image_name") or "",
                    tile_id=tile.get("tile_id"),
                    crop_box=crop_box,
                    filtered_count=tile_candidate_summary["dust_mask_filtered_count"],
                    detection=detection_summary,
                )
            if chosen["count"] <= 0 and not dust_only:
                no_detect_filtered = sum(
                    _as_int(d.get("no_detect_mask_filtered_count"), 0)
                    for d in detections
                )
                if no_detect_filtered > 0:
                    result["skipped_tiles"]["no_detect_mask"] = (
                        result["skipped_tiles"].get("no_detect_mask", 0) + 1
                    )
                    add_step(
                        "tile 點候選落在 MARK/不檢測區：略過規格內判定",
                        image=image.get("image_name") or "",
                        tile_id=tile.get("tile_id"),
                        crop_box=crop_box,
                        filtered_count=no_detect_filtered,
                        detection=detection_summary,
                    )
                    visual = _save_within_spec_dot_visuals(
                        tile_crop=tile_crop,
                        processed_crop=processed_crop,
                        chosen=chosen,
                        dot_cfg=dot_cfg,
                        runtime_dust_mask=runtime_dust_mask,
                        no_detect_mask=no_detect_mask,
                        non_dot_residues=tile_non_dot_residues,
                        output_dir=visual_output_dir,
                        url_prefix=visual_url_prefix,
                        image_name=image.get("image_name") or image_path.name,
                        tile_id=tile.get("tile_id"),
                        crop_box=crop_box,
                    )
                    if visual:
                        result["visuals"].append(visual)
                    continue

                missed_tile = {
                    "image": image.get("image_name") or image_path.name,
                    "screen": screen_code,
                    "tile_id": tile.get("tile_id"),
                    "crop_box": crop_box,
                    "is_aoi_coord": bool(tile.get("is_aoi_coord")),
                    "aoi_defect_code": tile.get("aoi_defect_code") or "",
                    "aoi_product_x": _as_int(tile.get("aoi_product_x"), -1),
                    "aoi_product_y": _as_int(tile.get("aoi_product_y"), -1),
                    "aoi_image_x": _as_int(tile.get("aoi_image_x"), -1),
                    "aoi_image_y": _as_int(tile.get("aoi_image_y"), -1),
                }
                result["missed_dot_tiles"].append(missed_tile)
                add_step(
                    "tile 點偵測未命中：不符合規格內",
                    image=image.get("image_name") or "",
                    tile_id=tile.get("tile_id"),
                    crop_box=crop_box,
                    is_aoi_coord=missed_tile["is_aoi_coord"],
                    aoi_product_x=missed_tile["aoi_product_x"],
                    aoi_product_y=missed_tile["aoi_product_y"],
                    detection=detection_summary,
                )
                visual = _save_within_spec_dot_visuals(
                    tile_crop=tile_crop,
                    processed_crop=processed_crop,
                    chosen=chosen,
                    dot_cfg=dot_cfg,
                    runtime_dust_mask=runtime_dust_mask,
                    no_detect_mask=no_detect_mask,
                    non_dot_residues=tile_non_dot_residues,
                    output_dir=visual_output_dir,
                    url_prefix=visual_url_prefix,
                    image_name=image.get("image_name") or image_path.name,
                    tile_id=tile.get("tile_id"),
                    crop_box=crop_box,
                )
                if visual:
                    missed_tile["visual"] = visual
                    result["visuals"].append(visual)
                continue

            result["evaluated_tile_count"] += 1
            key = (screen_code, chosen["dot_type"])
            state = aggregates.setdefault(key, {
                "screen": screen_code,
                "image_name": image.get("image_name") or image_path.name,
                "dot_type": chosen["dot_type"],
                "dot_label": chosen["label"],
                "rule": _json_safe_snapshot(chosen["rule"]),
                "threshold_mm": chosen["threshold_mm"],
                "screen_count_limit": chosen["screen_limit"],
                "tile_count_threshold": chosen["tile_limit"],
                "screen_count": 0,
                "max_tile_count": 0,
                "max_size_mm": 0.0,
                "target_tile_count": 0,
                "fallback_count": 0,
                "aoi_defect_codes": set(),
                "tiles": [],
            })
            state["screen_count"] += chosen["count"]
            state["max_tile_count"] = max(state["max_tile_count"], chosen["count"])
            state["max_size_mm"] = max(state["max_size_mm"], chosen["max_size_mm"])
            state["target_tile_count"] += 1
            if tile.get("aoi_defect_code"):
                state["aoi_defect_codes"].add(str(tile.get("aoi_defect_code")))
            tile_detail = {
                "tile_id": tile.get("tile_id"),
                "count": chosen["count"],
                "max_size_mm": round(float(chosen["max_size_mm"]), 4),
                "segmentation_method": chosen["segmentation_method"],
                "auto_candidates": chosen["auto_candidates"],
                "thresholds": chosen.get("thresholds", {}),
                "crop_box": crop_box,
            }
            state["tiles"].append(tile_detail)
            visual = _save_within_spec_dot_visuals(
                tile_crop=tile_crop,
                processed_crop=processed_crop,
                chosen=chosen,
                dot_cfg=dot_cfg,
                runtime_dust_mask=runtime_dust_mask,
                no_detect_mask=no_detect_mask,
                non_dot_residues=tile_non_dot_residues,
                output_dir=visual_output_dir,
                url_prefix=visual_url_prefix,
                image_name=image.get("image_name") or image_path.name,
                tile_id=tile.get("tile_id"),
                crop_box=crop_box,
            )
            if visual:
                state["tiles"][-1]["visual"] = visual
                result["visuals"].append(visual)
            add_step(
                "tile 點類分類完成",
                image=image.get("image_name") or "",
                tile_id=tile.get("tile_id"),
                crop_box=crop_box,
                classified_as=chosen["dot_type"],
                detection=detection_summary,
            )

    matches = []
    panel_totals = []
    for state in aggregates.values():
        panel_totals.append({
            "screen": state["screen"],
            "dot_type": state["dot_type"],
            "dot_label": state["dot_label"],
            "total_count": int(state["screen_count"]),
            "max_size_mm": round(float(state["max_size_mm"]), 4),
            "evaluated_tiles": int(state["target_tile_count"]),
            "threshold_mm": round(float(state["threshold_mm"]), 4),
            "screen_count_limit": int(state["screen_count_limit"]),
            "max_tile_count": int(state["max_tile_count"]),
            "tile_count_threshold": int(state["tile_count_threshold"]),
            "fallback_count": int(state["fallback_count"]),
            "rule": state["rule"],
        })
        within = (
            state["max_size_mm"] <= state["threshold_mm"]
            and state["screen_count"] <= state["screen_count_limit"]
            and state["max_tile_count"] <= state["tile_count_threshold"]
        )
        panel_totals[-1]["within"] = within
        add_step(
            "規格內規則比對",
            screen=state["screen"],
            dot_type=state["dot_type"],
            max_size_mm=round(float(state["max_size_mm"]), 4),
            threshold_mm=round(float(state["threshold_mm"]), 4),
            screen_count=state["screen_count"],
            screen_count_limit=state["screen_count_limit"],
            max_tile_count=state["max_tile_count"],
            tile_count_threshold=state["tile_count_threshold"],
            within=within,
        )
        if not within:
            continue

        matches.append({
            "screen": state["screen"],
            "image_name": state["image_name"],
            "dot_type": state["dot_type"],
            "dot_label": state["dot_label"],
            "max_size_mm": round(float(state["max_size_mm"]), 4),
            "threshold_mm": round(float(state["threshold_mm"]), 4),
            "screen_count": int(state["screen_count"]),
            "screen_count_limit": int(state["screen_count_limit"]),
            "max_tile_count": int(state["max_tile_count"]),
            "tile_count_threshold": int(state["tile_count_threshold"]),
            "target_tile_count": int(state["target_tile_count"]),
            "fallback_count": int(state["fallback_count"]),
            "aoi_defect_codes": sorted(state["aoi_defect_codes"]),
            "rule": state["rule"],
            "tiles": state["tiles"][:20],
        })

    result["panel_totals"] = panel_totals
    result["non_dot_residues"] = [
        {k: v for k, v in item.items() if k != "_key"}
        for item in non_dot_residues[:50]
    ]
    result["panel_summary"] = {
        "total_dot_count": sum(item["total_count"] for item in panel_totals),
        "total_visuals": len(result["visuals"]),
        "target_tile_count": result["target_tile_count"],
        "evaluated_tile_count": result["evaluated_tile_count"],
        "fallback_count": sum(item.get("fallback_count", 0) for item in panel_totals),
        "non_dot_residue_count": len(non_dot_residues),
        "missed_dot_tile_count": len(result["missed_dot_tiles"]),
        "skipped_tiles": result["skipped_tiles"],
        "candidate_summary": result["candidate_summary"],
    }
    result["matches"] = matches[:5]
    if non_dot_residues:
        add_step(
            "非點狀殘留命中：不建議規格內",
            count=len(non_dot_residues),
            residues=result["non_dot_residues"][:12],
        )
    if result["missed_dot_tiles"]:
        add_step(
            "點偵測未命中：不建議規格內",
            count=len(result["missed_dot_tiles"]),
            tiles=result["missed_dot_tiles"][:12],
        )
    if non_dot_residues or result["missed_dot_tiles"]:
        return result
    if not matches:
        add_step("判定結果：未符合規格內建議條件")
        return result

    first = matches[0]
    result["suggestion"] = {
        "suggested": True,
        "category": "within_spec",
        "reason": (
            f"{first['screen']} {first['dot_label']} "
            f"{first['max_size_mm']:.4g}mm <= {first['threshold_mm']:.4g}mm，"
            f"畫面 {first['screen_count']} <= {first['screen_count_limit']}，"
            f"Tile {first['max_tile_count']} <= {first['tile_count_threshold']}"
        ),
        "matches": matches[:5],
    }
    add_step("判定結果：建議規格內", reason=result["suggestion"]["reason"])
    return result


def _evaluate_within_spec_suggestion(
    detail: Dict[str, Any],
    rules: Dict[str, Any],
    machine_id: str = "",
    rotate_180: bool = False,
) -> Optional[Dict[str, Any]]:
    return _evaluate_within_spec_suggestion_detail(
        detail,
        rules,
        machine_id,
        rotate_180=rotate_180,
    ).get("suggestion")


class _CallbackLogHandler(logging.Handler):
    """Forwards selected library log records into the training wizard log."""

    def __init__(self, callback):
        super().__init__(level=logging.INFO)
        self.callback = callback
        self.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.callback(self.format(record))
        except Exception:
            pass


# 解析 train runner log 行抽出 unit 狀態。例：
#   [16:13:38] INFO [capi.train_runner] [3/10] R0F00000-inner: 載 tile
#   [15:32:50] INFO [capi.train_runner] [1/10] G0F00000-inner: ✓ done | 360s, ...
_TRAIN_UNIT_LOG_RE = re.compile(
    r"\[\d+/\d+\]\s+(\S+):\s+(✓ done|✗ 訓練失敗|跳過：tile 不足|載 tile)"
)
_TRAIN_COMPLETED_LOG_RE = re.compile(
    r"✓\s+(?:局部重訓|訓練)完成，bundle=(.+?)\s*$"
)
_TRAIN_UNIT_STATUS_MAP = {
    "✓ done": "done",
    "✗ 訓練失敗": "failed",
    "跳過：tile 不足": "skipped",
    "載 tile": "running",
}


class CAPIWebHandler(BaseHTTPRequestHandler):
    """CAPI Web 請求處理器"""

    # 類別變數，由 create_web_server 設定
    db = None
    # Jinja2 環境
    jinja_env = None
    # Debug 推論用 (可選)
    inferencer = None
    heatmap_manager = None
    _debug_heatmap_dir = None  # Debug 推論暫存目錄
    _capi_server_instance = None  # CAPIServer 實例 (用於 hot-reload)
    _log_file = None  # 日誌檔案路徑 (用於 Log Viewer)
    _settings_session_cookie = "capi_settings_session"
    _settings_session_ttl_seconds = 12 * 60 * 60
    _settings_sessions: Dict[str, Dict[str, Any]] = {}
    _settings_session_lock: threading.Lock = threading.Lock()
    _update_state_file = Path(__file__).resolve().parent / "update" / "auto_update_state.json"
    _update_apply_lock: threading.Lock = threading.Lock()
    _mes_comparison_lock: threading.Lock = threading.Lock()
    _mark_calibration_lock: threading.Lock = threading.Lock()
    # ── 訓練 wizard 多 job 註冊表（由 create_web_server 完整初始化） ─────────
    # _train_new_jobs key = job_id；value = per-job runtime dict，欄位：
    #   thread:        Thread (preprocess supervisor / training supervisor)
    #   proc:          Optional[Popen] (only training)
    #   cancel_flag:   Optional[str] (training subprocess 的 cancel flag 檔路徑)
    #   log_file:      Optional[str] (training subprocess 的 log 檔路徑)
    #   cancel_event:  threading.Event
    #   log_lines:     List[str] (ring buffer 500)
    #   log_lock:      threading.Lock
    #   unit_status:   Dict[unit_label, status]
    #   completed_bundle: Optional[str] (runner 完成訊息中的 bundle path)
    #   phase:         "preprocess" | "review" | "train"  (last known)
    _train_new_jobs: dict = {}
    _train_new_jobs_lock: threading.Lock = threading.Lock()

    # 訓練槽（單機 GPU 序列化）：active_job_id 不為 None 即代表槽被佔用。
    # lock 只保護「進入 train phase 的關鍵段落」，worker 不長期持鎖。
    _train_slot: dict = {
        "lock": threading.Lock(),
        "active_job_id": None,
    }
    TRAIN_NEW_MACHINE_PREFIX_LEN = 8
    TRAIN_NEW_DEFAULT_TILE_STRIDE = 256
    TRAIN_NEW_MIN_TILE_STRIDE = 64
    TRAIN_NEW_MAX_TILE_STRIDE = 512
    PATCHCORE_BUNDLE_LOCKED_TRAINING_PARAMS = frozenset({
        "feature_pool_kernel_size",
        "feature_cleaning_mode",
        "feature_cleaning_scope",
        "feature_cleaning_keep_ratio",
        "feature_cleaning_center_size",
        "feature_cleaning_by_zone",
    })

    # 單子模型重訓 state（一次只允許一個 job）
    _submodel_retrain_state: dict = {
        "lock": threading.Lock(),
        "job": None,
        # job dict 結構（running 中時）：
        # {
        #   bundle_id: int,
        #   lighting: str,
        #   zone: str,
        #   state: "running" | "completed" | "failed",
        #   step: "stage" | "train" | "metrics" | "swap" | "reload" | "done",
        #   started_at: str (ISO),
        #   log_lines: list[str],
        #   summary: dict | None,    # {auroc_old, auroc_new, tile_count_old, tile_count_new}
        #   error: str | None,
        # }
    }

    # 全 server 同時只能一個 score scan job（共用 GPU，序列化）
    _scan_state = {
        "lock": threading.Lock(),
        "job": None,  # Optional[dict]，欄位見 _start_scan_job
    }

    @classmethod
    def init_jinja(cls):
        if cls.jinja_env is None:
            templates_dir = Path(__file__).parent / "templates"
            cls.jinja_env = Environment(loader=FileSystemLoader(templates_dir))
            cls.jinja_env.filters['ai_simple'] = ai_simple
            cls.jinja_env.filters['ai_badge'] = ai_badge
            cls.jinja_env.filters['mj_badge'] = mj_badge
            cls.jinja_env.filters['img_status_info'] = img_status_info
            cls.jinja_env.filters['tile_info'] = tile_info
            cls.jinja_env.filters['get_img_stem'] = get_img_stem
            cls.jinja_env.filters['fromjson'] = lambda s: json.loads(s) if s else {}
            cls.jinja_env.globals['hm_relative'] = hm_relative
            cls.jinja_env.globals['app_version'] = _AppVersionProxy()
            cls.jinja_env.globals['host_identity'] = _get_host_identity()

    @classmethod
    def _make_job_runtime(cls, job_id: str, phase: str) -> dict:
        runtime = {
            "thread": None,
            "proc": None,
            "cancel_flag": None,
            "log_file": None,
            "cancel_event": threading.Event(),
            "log_lines": [],
            "log_lock": threading.Lock(),
            "unit_status": {},
            "completed_bundle": None,
            "phase": phase,
        }
        with cls._train_new_jobs_lock:
            cls._train_new_jobs[job_id] = runtime
        return runtime

    @classmethod
    def _get_job_runtime(cls, job_id: str) -> Optional[dict]:
        with cls._train_new_jobs_lock:
            return cls._train_new_jobs.get(job_id)

    @classmethod
    def _drop_job_runtime(cls, job_id: str) -> None:
        with cls._train_new_jobs_lock:
            cls._train_new_jobs.pop(job_id, None)

    @classmethod
    def _append_train_new_log(cls, job_id: str, msg: str) -> None:
        runtime = cls._get_job_runtime(job_id)
        if runtime is None:
            return
        with runtime["log_lock"]:
            runtime["log_lines"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            if len(runtime["log_lines"]) > 500:
                runtime["log_lines"] = runtime["log_lines"][-500:]
            m = _TRAIN_UNIT_LOG_RE.search(msg)
            if m:
                runtime.setdefault("unit_status", {})[m.group(1)] = _TRAIN_UNIT_STATUS_MAP[m.group(2)]
            completed_match = _TRAIN_COMPLETED_LOG_RE.search(msg)
            if completed_match:
                runtime["completed_bundle"] = completed_match.group(1).strip()

    @classmethod
    def _sync_train_new_completed_state(
        cls,
        db,
        job: Optional[dict],
        completed_bundle: Optional[str],
    ) -> Optional[dict]:
        """以 runner 成功完成訊息校正 Web process 看到的 training job 狀態。"""
        bundle_path = str(completed_bundle or "").strip()
        if not job or job.get("state") != "train" or not bundle_path:
            return job
        try:
            db.update_training_job_state(
                job["job_id"],
                "completed",
                output_bundle=bundle_path,
            )
        except Exception:
            logger.warning(
                "cannot sync completed training state: job_id=%s bundle=%s",
                job.get("job_id"),
                bundle_path,
                exc_info=True,
            )
            return job
        updated = dict(job)
        updated["state"] = "completed"
        updated["output_bundle"] = bundle_path
        return updated

    @classmethod
    def _train_new_cancel_event(cls, job_id: str) -> threading.Event:
        runtime = cls._get_job_runtime(job_id)
        if runtime is None:
            return threading.Event()
        return runtime.setdefault("cancel_event", threading.Event())

    @classmethod
    def _train_new_worker_alive(cls, job_id: str) -> bool:
        runtime = cls._get_job_runtime(job_id)
        if runtime is None:
            return False
        thread = runtime.get("thread")
        if thread is not None and thread.is_alive():
            return True
        proc = runtime.get("proc")
        if proc is not None and proc.poll() is None:
            return True
        return False

    @classmethod
    def _cleanup_train_new_job_artifacts(
        cls,
        db,
        job_id: str,
        *,
        remove_training_data: bool = True,
        reason: str = "",
    ) -> Optional[dict]:
        from capi_model_registry import cleanup_training_job_artifacts

        try:
            result = cleanup_training_job_artifacts(
                db,
                job_id,
                remove_training_data=remove_training_data,
            )
        except Exception:
            logger.warning(
                "training temp cleanup failed: job_id=%s reason=%s",
                job_id,
                reason,
                exc_info=True,
            )
            return None

        if result.get("deleted_files") or result.get("deleted_tile_rows"):
            logger.info(
                "training temp cleanup: job_id=%s reason=%s files=%s rows=%s freed=%s",
                job_id,
                reason or "unspecified",
                result.get("deleted_files", 0),
                result.get("deleted_tile_rows", 0),
                result.get("freed_bytes", 0),
            )
        return result

    @classmethod
    def _reconcile_train_new_artifacts(cls, db) -> None:
        """Server 啟動時回收 kill/失敗 job 的暫存資料。

        review / completed job 的 thumbnails 是後續檢視與重訓資料，保留；
        staging / runner 輸出則一律視為單次訓練暫存。
        """
        processed_job_ids = set()

        try:
            active_jobs = db.list_active_training_jobs()
        except Exception:
            logger.warning("cannot inspect active training jobs for temp cleanup", exc_info=True)
            active_jobs = []

        for job in active_jobs or []:
            job_id = str(job.get("job_id") or "")
            if not job_id:
                continue
            if job.get("state") in ("preprocess", "train"):
                processed_job_ids.add(job_id)
                cls._mark_train_new_stale_if_needed(db, job)

        roots = (
            Path(".tmp/training_staging"),
            Path(".tmp/training_runs"),
            Path(".tmp/train_new_thumbs"),
        )
        job_ids = set()
        for root in roots:
            try:
                if root.is_dir():
                    job_ids.update(
                        child.name
                        for child in root.iterdir()
                        if child.is_dir() and not child.is_symlink()
                    )
            except OSError:
                logger.warning("cannot scan training temp root: %s", root, exc_info=True)

        for job_id in sorted(job_ids - processed_job_ids):
            try:
                job = db.get_training_job(job_id)
            except Exception:
                logger.warning("cannot inspect training job: %s", job_id, exc_info=True)
                continue

            if not job:
                cls._cleanup_train_new_job_artifacts(
                    db, job_id, reason="orphan temp directory"
                )
                continue

            state = job.get("state")
            if state == "failed":
                cls._cleanup_train_new_job_artifacts(
                    db, job_id, reason="failed job"
                )
            elif state in ("review", "completed"):
                cls._cleanup_train_new_job_artifacts(
                    db,
                    job_id,
                    remove_training_data=False,
                    reason=f"{state} ephemeral cleanup",
                )
            elif state not in ("preprocess", "train"):
                cls._cleanup_train_new_job_artifacts(
                    db, job_id, reason=f"unexpected state={state}"
                )

    @classmethod
    def _mark_train_new_stale_if_needed(cls, db, job: Optional[dict]) -> Optional[dict]:
        if not job or job.get("state") not in ("preprocess", "train"):
            return job
        if cls._train_new_worker_alive(job["job_id"]):
            return job

        error = "interrupted: server restarted or training worker is not running"
        db.update_training_job_state(job["job_id"], "failed", error_message=error)
        cls._cleanup_train_new_job_artifacts(db, job["job_id"], reason=error)
        cls._drop_job_runtime(job["job_id"])
        slot = cls._train_slot
        with slot["lock"]:
            if slot.get("active_job_id") == job["job_id"]:
                slot["active_job_id"] = None
        updated = dict(job)
        updated["state"] = "failed"
        updated["error_message"] = error
        return updated

    @staticmethod
    @contextmanager
    def _capture_train_new_library_logs(log):
        handler = _CallbackLogHandler(log)
        logger_names = (
            "lightning",
            "lightning_fabric",
            "pytorch_lightning",
            "anomalib",
        )
        old_levels = []
        attached = []
        try:
            for name in logger_names:
                lib_logger = logging.getLogger(name)
                old_levels.append((lib_logger, lib_logger.level))
                if lib_logger.level == logging.NOTSET or lib_logger.level > logging.INFO:
                    lib_logger.setLevel(logging.INFO)
                lib_logger.addHandler(handler)
                attached.append(lib_logger)
            yield
        finally:
            for lib_logger in attached:
                lib_logger.removeHandler(handler)
            for lib_logger, old_level in old_levels:
                lib_logger.setLevel(old_level)
            
    def log_message(self, format, *args):
        """靜默 Web HTTP 存取日誌，避免污染 server.log 與 CMD"""
        pass

    def do_GET(self):
        """處理 GET 請求"""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if path == "/" or path == "":
                self._handle_index(query, path)
            elif path == "/dashboard":
                self._handle_dashboard(query, path)
            elif path == "/dashboard_v2":
                self._handle_dashboard_v2(query, path)
            elif path == "/v3/dashboard":
                self._handle_dashboard_v3(query, path)
            elif path.startswith("/api/record/") and path.endswith("/rerun/status"):
                record_id_str = path.split("/api/record/")[1].split("/rerun/status")[0]
                self._handle_rerun_status_sse(record_id_str)
            elif path.startswith("/v3/record/"):
                record_id = path.split("/v3/record/")[1].rstrip("/")
                self._handle_record_detail_v3(record_id, path)
            elif path.startswith("/record/"):
                record_id = path.split("/record/")[1].rstrip("/")
                self._handle_record_detail(record_id, path)
            elif path == "/overexposed":
                self._handle_overexposed(query, path)
            elif path == "/search":
                self._handle_search(query, path)
            elif path == "/search/export":
                self._handle_search_export(query)
            elif path == "/logs":
                self._handle_logs_page(query, path)
            elif path == "/api/logs":
                self._handle_api_logs(query)
            elif path == "/release-notes":
                self._handle_release_notes_page(path)
            elif path == "/debug":
                self._handle_debug_page(path)
            elif path == "/training":
                self._handle_training_page()
            elif path == "/train/new":
                self._handle_train_new_scope_page()
            elif path == "/train/new/select":
                self._handle_train_new_select_page()
            elif path == "/train/new/progress":
                self._handle_train_new_progress_page()
            elif path.startswith("/train/new/review/"):
                self._handle_train_new_review_page()
            elif path.startswith("/train/new/done/"):
                self._handle_train_new_done_page()
            elif path == "/retrain":
                self._handle_retrain_page()
            elif path == "/api/retrain/status":
                self._handle_retrain_status()
            elif path == "/api/train/new/panels":
                self._handle_train_new_panels()
            elif path.startswith("/api/train/new/status"):
                self._handle_train_new_status()
            elif path == "/api/train/new/tiles":
                self._handle_train_new_tiles()
            elif path == "/api/train/new/preprocess_pipeline_preview":
                self._handle_train_new_preprocess_pipeline_preview()
            elif path == "/api/train/new/preprocess_preview":
                self._handle_train_new_preprocess_preview()
            elif path.startswith("/api/train/new/thumb/"):
                self._handle_train_new_thumb()
            elif path.startswith("/api/train/new/bundle-asset/"):
                self._handle_train_new_bundle_asset()
            elif path == "/ric":
                self._handle_ric_page(query, path)
            elif path == "/ric/within-spec-logs":
                self._handle_within_spec_report_page(query, path)
            elif path.startswith("/ric/within-spec-log/"):
                self._handle_within_spec_log_page(path)
            elif path == "/api/ric/report":
                self._handle_ric_report_api(query)
            elif path == "/api/ric/client-data":
                self._handle_client_data_api(query)
            elif path == "/api/ric/within-spec-suggestion":
                self._handle_within_spec_suggestion_api(query)
            elif path == "/api/ric/within-spec-suggestion/log":
                self._handle_within_spec_suggestion_log_api(query)
            elif path == "/api/ric/inference-stats":
                self._handle_inference_stats_api(query)
            elif path == "/api/ric/mes-comparison":
                self._handle_mes_comparison_api(query)
            elif path == "/api/ric/mes-report-detail":
                self._handle_mes_report_detail_api(query)
            elif path == "/api/ric/mes-review/candidates":
                self._handle_mes_review_candidates_api(query)
            elif path == "/api/ric/mes-review/crop":
                self._handle_mes_review_crop_api(query)
            elif path == "/api/ric/ng-validation":
                self._handle_ng_validation_api(query)
            elif path == "/api/ric/ng-validation/file":
                self._handle_ng_validation_file_api(query)
            elif path == "/scratch-review":
                self._handle_scratch_review_page(path)
            elif path == "/api/scratch-review/list":
                self._handle_scratch_review_list_api(query)
            elif path == "/api/stats":
                self._handle_api_stats(query)
            elif path == "/api/status":
                self._handle_api_status()
            elif path == "/api/version":
                self._handle_api_version()
            elif path == "/api/update/status":
                self._handle_api_update_status()
            elif path == "/api/central-dashboard/config/all":
                if self._require_settings_user(api=True):
                    self._handle_api_central_dashboard_config_all()
            elif path == "/api/central-dashboard/config":
                self._handle_api_central_dashboard_config()
            elif path == "/settings/login":
                self._handle_settings_login_page(path)
            elif path == "/settings/logout":
                self._handle_settings_logout()
            elif path == "/api/settings/me":
                user = self._require_settings_user(api=True)
                if user:
                    self._send_json({"user": user})
            elif path == "/api/settings/users":
                if self._require_settings_user(api=True, admin=True):
                    self._handle_api_settings_users()
            elif path == "/api/settings/mark":
                if self._require_settings_user(api=True, admin=True):
                    self._handle_api_settings_mark()
            elif path == "/api/settings/mark/sample-image":
                if self._require_settings_user(api=True, admin=True):
                    self._handle_api_settings_mark_sample_image(query)
            elif path == "/api/settings/mark-shadow":
                if self._require_settings_user(api=True, admin=True):
                    self._handle_api_settings_mark_shadow(query)
            elif path == "/api/settings/mark-shadow/crop":
                if self._require_settings_user(api=True, admin=True):
                    self._handle_api_settings_mark_shadow_crop(query)
            elif path == "/settings":
                if self._require_settings_user(next_path=path):
                    self._handle_settings_page(path)
            elif path == "/settings_v2":
                if self._require_settings_user(next_path=path):
                    self._handle_settings_v2_page(path)
            elif path == "/api/settings":
                if self._require_settings_user(api=True):
                    self._handle_api_settings()
            elif path == "/api/settings/scratch-bundles":
                if self._require_settings_user(api=True):
                    self._handle_api_settings_scratch_bundles()
            elif path == "/api/settings/history":
                if self._require_settings_user(api=True):
                    self._handle_api_settings_history(query)
            elif path == "/api/auto-model-switch":
                if self._require_settings_user(api=True):
                    self._handle_auto_model_switch_api()
                return
            elif path == "/api/auto-model-switch/history":
                if self._require_settings_user(api=True):
                    self._handle_auto_model_switch_history_api(query)
                return
            elif path.startswith("/heatmaps/"):
                self._handle_static_file(path)
            elif path.startswith("/debug/heatmaps/"):
                self._handle_debug_heatmap_file(path)
            elif path == "/api/debug/serve-image":
                self._handle_debug_serve_image(query)
            elif path in (
                "/central_dashboard/settings",
                "/central_dashboard/settings/",
                "/central_dashboard/settings.html",
            ):
                if self._require_settings_user(next_path="/central_dashboard/settings"):
                    self._handle_central_dashboard_file(
                        "/central_dashboard/settings.html"
                    )
            elif path == "/central_dashboard" or path == "/central_dashboard/":
                self._handle_central_dashboard_file("/central_dashboard/index.html")
            elif path.startswith("/central_dashboard/"):
                self._handle_central_dashboard_file(path)
            elif path.startswith("/images/"):
                self._handle_source_image(path)
            elif path in ("/favicon.ico", "/favicon.svg"):
                self._handle_static_assets("/static/favicon.svg")
            elif path.startswith("/imgs/"):
                self._handle_imgs_file(path)
            elif path.startswith("/static/"):
                self._handle_static_assets(path)
            elif path == "/api/dataset_export/status":
                self._handle_dataset_export_status()
                return
            elif path.startswith("/api/dataset_export/summary/"):
                job_id = path.split("/api/dataset_export/summary/", 1)[1]
                self._handle_dataset_export_summary(job_id)
                return
            elif path == "/dataset_gallery":
                self._handle_dataset_gallery_page(query)
                return
            elif path == "/api/dataset_export/file":
                self._handle_dataset_export_file(query)
                return
            elif path == "/models/retrain-pool":
                self._handle_retrain_pool_page()
                return
            elif path == "/api/retrain-pool":
                self._handle_retrain_pool_list(query)
                return
            elif path == "/api/retrain-pool/file":
                self._handle_retrain_pool_file(query)
                return
            elif path == "/debug/scratch-batch":
                self._handle_scratch_batch_page(path, query)
                return
            elif path == "/api/debug/scratch-batch/jobs":
                self._handle_scratch_batch_jobs()
                return
            elif path == "/api/debug/scratch-batch/status":
                self._handle_scratch_batch_status(query)
                return
            elif path == "/api/debug/scratch-batch/result":
                self._handle_scratch_batch_result(query)
                return
            elif path == "/api/debug/scratch-batch/export":
                self._handle_scratch_batch_export(query)
                return
            elif path == "/models":
                self._handle_models_page()
                return
            elif path == "/api/models/discover":
                self._handle_models_discover()
                return
            elif path == "/api/models":
                self._handle_models_list()
                return
            elif path.startswith("/api/models/") and path.endswith("/detail"):
                self._handle_models_detail()
                return
            elif path.startswith("/api/models/") and path.endswith("/validation"):
                self._handle_models_validation()
                return
            elif path.startswith("/api/models/") and path.endswith("/training_tiles"):
                self._handle_models_training_tiles()
                return
            elif path.startswith("/api/models/") and path.endswith("/export"):
                self._handle_models_export()
                return
            elif path.startswith("/api/models/") and path.endswith("/retrain_status"):
                self._handle_models_retrain_status()
                return
            elif path == "/api/scan/status":
                self._handle_scan_status()
                return
            elif path == "/api/train/new/eligible_scoring_bundles":
                self._handle_eligible_scoring_bundles()
                return
            elif path == "/favicon.ico":
                self._send_response(204, "")
            else:
                self._send_404(path)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # 客戶端已斷線，靜默忽略即可
            pass
        except Exception as e:
            logger.error(f"Error handling {path}: {e}", exc_info=True)
            try:
                self._send_error(500, str(e))
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    def do_POST(self):
        """處理 POST 請求"""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        try:
            if path == "/api/debug/inference":
                self._handle_debug_inference_run()
            elif path == "/api/debug/mark-detection":
                self._handle_debug_mark_detection()
            elif path == "/api/debug/preprocess-lab":
                self._handle_debug_preprocess_lab()
            elif path == "/api/debug/coord-inference":
                self._handle_debug_coord_inference()
            elif path == "/api/debug/edge-inspect":
                self._handle_api_debug_edge_inspect()
            elif path == "/api/debug/edge-inspect-corner":
                self._handle_api_debug_edge_inspect_corner()
            elif path == "/api/debug/bright-spot-inference":
                self._handle_debug_bright_spot_inference()
            elif path == "/api/debug/dot-detection":
                self._handle_debug_dot_detection()
            elif path == "/api/ric/upload":
                self._handle_ric_upload()
            elif path == "/api/ric/delete":
                self._handle_ric_delete()
            elif path == "/api/ric/import-client":
                self._handle_client_import()
            elif path == "/api/ric/clear-client":
                self._handle_client_clear()
            elif path == "/api/ric/miss-review":
                self._handle_miss_review_save()
            elif path == "/api/ric/miss-review/delete":
                self._handle_miss_review_delete()
            elif path == "/api/ric/over-review":
                self._handle_over_review_save()
            elif path == "/api/ric/over-review/delete":
                self._handle_over_review_delete()
            elif path == "/api/ric/mes-review":
                self._handle_mes_review_save()
            elif path == "/api/ric/mes-review/delete":
                self._handle_mes_review_delete()
            elif path == "/api/ric/ng-validation/delete":
                self._handle_ng_validation_delete()
            elif path == "/api/ric/over-retrain-pool/add":
                self._handle_over_retrain_pool_add()
            elif path == "/api/ric/within-spec-log/regenerate":
                self._handle_within_spec_log_regenerate()
            elif path == "/api/ric/within-spec-suggestion/run":
                self._handle_within_spec_suggestion_run()
            elif path == "/api/scratch-review/mark":
                self._handle_scratch_review_mark()
            elif path == "/api/scratch-review/unmark":
                self._handle_scratch_review_unmark()
            elif path == "/api/scratch-review/export":
                self._handle_scratch_review_export()
            elif path == "/api/settings/login":
                self._handle_api_settings_login()
            elif path == CENTRAL_ACCOUNT_AUTH_PATH:
                self._handle_api_settings_central_auth()
            elif path == "/api/settings/logout":
                self._handle_api_settings_logout()
            elif path == "/api/central-dashboard/config":
                if self._require_settings_user(api=True):
                    self._handle_api_central_dashboard_config_update()
            elif path == "/api/central-dashboard/update/apply":
                user = self._require_settings_user(api=True, admin=True)
                if user:
                    self._handle_api_central_dashboard_update_apply(user)
            elif path == "/api/update/apply":
                if self._require_settings_user(api=True, admin=True):
                    self._handle_api_update_apply()
            elif path == CENTRAL_UPDATE_APPLY_PATH:
                self._handle_api_central_update_apply()
            elif path == "/api/settings/users/create":
                if self._require_settings_user(api=True, admin=True):
                    self._handle_api_settings_user_create()
            elif path == "/api/settings/users/update":
                if self._require_settings_user(api=True, admin=True):
                    self._handle_api_settings_user_update()
            elif path == "/api/settings/users/delete":
                if self._require_settings_user(api=True, admin=True):
                    self._handle_api_settings_user_delete()
            elif path == "/api/settings/mark/correct":
                if self._require_settings_user(api=True, admin=True):
                    self._handle_api_settings_mark_correct()
            elif path == "/api/settings/mark/rollback":
                if self._require_settings_user(api=True, admin=True):
                    self._handle_api_settings_mark_rollback()
            elif path == "/api/settings/update":
                if self._require_settings_user(api=True):
                    self._handle_api_settings_update()
            elif path == "/api/settings/reload":
                if self._require_settings_user(api=True):
                    self._handle_api_settings_reload()
            elif path == "/api/auto-model-switch/rules/upsert":
                if self._require_settings_user(api=True):
                    self._handle_auto_model_switch_rule_upsert()
                return
            elif path == "/api/auto-model-switch/rules/delete":
                if self._require_settings_user(api=True):
                    self._handle_auto_model_switch_rule_delete()
                return
            elif path.startswith("/api/record/") and path.endswith("/rerun"):
                record_id_str = path.split("/api/record/")[1].split("/rerun")[0]
                self._handle_rerun_trigger(record_id_str)
            elif path == "/api/dataset_export/start":
                self._handle_dataset_export_start()
                return
            elif path == "/api/retrain/start":
                self._handle_retrain_start()
                return
            elif path == "/api/dataset_export/cancel":
                self._handle_dataset_export_cancel()
                return
            elif path == "/api/dataset_export/sample/delete":
                self._handle_dataset_sample_delete()
                return
            elif path == "/api/dataset_export/sample/batch_delete":
                self._handle_dataset_sample_batch_delete()
                return
            elif path == "/api/dataset_export/sample/move":
                self._handle_dataset_sample_move()
                return
            elif path == "/api/debug/scratch-batch/start":
                self._handle_scratch_batch_start()
                return
            elif path == "/api/debug/scratch-batch/cancel":
                self._handle_scratch_batch_cancel()
                return
            elif path == "/api/train/new/start":
                self._handle_train_new_start()
                return
            elif path == "/api/train/new/manual-panels":
                self._handle_train_new_manual_panels()
                return
            elif path == "/api/train/new/preprocess_pipeline_preview":
                self._handle_train_new_preprocess_pipeline_preview()
                return
            elif path == "/api/train/new/tiles/decision":
                self._handle_train_new_tiles_decision()
                return
            elif path.startswith("/api/train/new/cancel/"):
                self._handle_train_new_cancel()
                return
            elif path.startswith("/api/train/new/start_training/"):
                self._handle_train_new_start_training()
                return
            elif path == "/api/models/sync":
                self._handle_models_sync()
                return
            elif path.startswith("/api/models/") and path.endswith("/activate"):
                self._handle_models_activate()
                return
            elif path.startswith("/api/models/") and path.endswith("/deactivate"):
                self._handle_models_deactivate()
                return
            elif path.startswith("/api/models/") and path.endswith("/notes"):
                self._handle_models_update_notes()
                return
            elif path.startswith("/api/models/") and path.endswith("/training_data/delete"):
                self._handle_models_training_data_delete()
                return
            elif path.startswith("/api/models/") and path.endswith("/tiles/decision"):
                self._handle_models_tiles_decision()
                return
            elif path.startswith("/api/models/") and path.endswith("/retrain_pool/add"):
                self._handle_models_retrain_pool_add()
                return
            elif path.startswith("/api/models/") and path.endswith("/retrain_submodel_with_panels"):
                self._handle_models_retrain_submodel_with_panels()
                return
            elif path.startswith("/api/models/") and path.endswith("/retrain_submodel"):
                self._handle_models_retrain_submodel()
                return
            elif path == "/api/retrain-pool/unadd":
                self._handle_retrain_pool_unadd()
                return
            elif path == "/api/retrain-pool/delete":
                self._handle_retrain_pool_delete()
                return
            elif path.startswith("/api/models/") and path.endswith("/scan_self_score"):
                self._handle_scan_self_score()
                return
            elif path.startswith("/api/models/") and path.endswith("/validation/start"):
                self._handle_models_validation_start()
                return
            elif path == "/api/train/new/scan_prefilter_score":
                self._handle_scan_prefilter_score()
                return
            elif path == "/api/scan/cancel":
                self._handle_scan_cancel()
                return
            elif path.startswith("/api/models/") and path.endswith("/threshold"):
                self._handle_models_update_threshold()
                return
            elif path.startswith("/api/models/") and path.endswith("/delete"):
                self._handle_models_delete()
                return
            else:
                self._send_404(path)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            logger.error(f"Error handling POST {path}: {e}", exc_info=True)
            try:
                self._send_json({"error": str(e)})
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    def _send_response(
        self,
        code: int,
        content: str,
        content_type: str = "text/html; charset=utf-8",
        headers: Optional[Dict[str, str]] = None,
    ):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content.encode("utf-8"))))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def _send_json(
        self,
        data,
        status=200,
        headers: Optional[Dict[str, str]] = None,
        *,
        compact: bool = False,
        compress: bool = False,
        server_timing: Optional[Dict[str, float]] = None,
    ):
        serialize_started = time.monotonic()
        dump_kwargs = (
            {"separators": (",", ":")}
            if compact else
            {"indent": 2}
        )
        content = json.dumps(data, ensure_ascii=False, default=str, **dump_kwargs)
        content_bytes = content.encode("utf-8")
        serialize_seconds = time.monotonic() - serialize_started
        uncompressed_size = len(content_bytes)

        response_headers = dict(headers or {})
        compression_seconds = 0.0
        accept_encoding = str(self.headers.get("Accept-Encoding", "") if self.headers else "")
        compressed = (
            compress
            and len(content_bytes) >= 1024
            and "gzip" in accept_encoding.lower()
        )
        if compress:
            response_headers["Vary"] = "Accept-Encoding"
        if compressed:
            compression_started = time.monotonic()
            content_bytes = gzip.compress(content_bytes, compresslevel=5)
            compression_seconds = time.monotonic() - compression_started
            response_headers["Content-Encoding"] = "gzip"

        if server_timing:
            timing_parts = [
                f"{name};dur={max(0.0, float(seconds)) * 1000.0:.1f}"
                for name, seconds in server_timing.items()
            ]
            timing_parts.append(f"json;dur={serialize_seconds * 1000.0:.1f}")
            if compressed:
                timing_parts.append(f"gzip;dur={compression_seconds * 1000.0:.1f}")
            response_headers["Server-Timing"] = ", ".join(timing_parts)

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content_bytes)))
        for name, value in response_headers.items():
            self.send_header(name, value)
        self.end_headers()
        write_started = time.monotonic()
        self.wfile.write(content_bytes)
        return {
            "serialize_seconds": serialize_seconds,
            "compression_seconds": compression_seconds,
            "write_seconds": time.monotonic() - write_started,
            "uncompressed_bytes": uncompressed_size,
            "response_bytes": len(content_bytes),
            "compressed": compressed,
        }

    def _redirect(self, location: str, headers: Optional[Dict[str, str]] = None):
        body = f"<html><body>Redirecting to {html.escape(location)}</body></html>"
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body.encode("utf-8"))))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _settings_cookie_token(self) -> str:
        raw_cookie = self.headers.get("Cookie", "") if self.headers else ""
        try:
            cookie = SimpleCookie(raw_cookie)
            morsel = cookie.get(self._settings_session_cookie)
            return morsel.value if morsel else ""
        except Exception:
            return ""

    def _settings_cookie_header(self, token: str) -> str:
        return (
            f"{self._settings_session_cookie}={token}; Path=/; HttpOnly; "
            f"SameSite=Lax; Max-Age={self._settings_session_ttl_seconds}"
        )

    def _settings_clear_cookie_header(self) -> str:
        return (
            f"{self._settings_session_cookie}=; Path=/; HttpOnly; "
            "SameSite=Lax; Max-Age=0"
        )

    @staticmethod
    def _safe_settings_next_path(next_path: str) -> str:
        next_path = str(next_path or "/settings")
        if not next_path.startswith("/") or next_path.startswith("//"):
            return "/settings"
        return next_path

    def _current_settings_user(self) -> Optional[Dict[str, Any]]:
        token = self._settings_cookie_token()
        if not token:
            return None
        now = datetime.now()
        with self._settings_session_lock:
            session = self._settings_sessions.get(token)
            if not session or session.get("expires_at", now) < now:
                self._settings_sessions.pop(token, None)
                return None
            username = session.get("username", "")
            if session.get("auth_source") == "central":
                user = session.get("user")
                if not isinstance(user, dict) or user.get("username") != username:
                    self._settings_sessions.pop(token, None)
                    return None
                return dict(user)
            user = self.db.get_settings_user_by_username(username) if self.db else None
            if not user:
                self._settings_sessions.pop(token, None)
                return None
            session["expires_at"] = now + timedelta(seconds=self._settings_session_ttl_seconds)
            return user

    def _create_settings_session(self, user: Dict[str, Any]) -> str:
        token = secrets.token_urlsafe(32)
        auth_source = (
            "central" if user.get("auth_source") == "central" else "local"
        )
        session = {
            "username": user["username"],
            "auth_source": auth_source,
            "expires_at": (
                datetime.now()
                + timedelta(seconds=self._settings_session_ttl_seconds)
            ),
        }
        if auth_source == "central":
            session["user"] = dict(user)
        with self._settings_session_lock:
            self._settings_sessions[token] = session
        return token

    def _drop_settings_session(self) -> None:
        token = self._settings_cookie_token()
        if token:
            with self._settings_session_lock:
                self._settings_sessions.pop(token, None)

    def _require_settings_user(
        self,
        *,
        api: bool = False,
        admin: bool = False,
        next_path: str = "/settings",
    ) -> Optional[Dict[str, Any]]:
        user = self._current_settings_user()
        if not user:
            if api:
                self._send_json({"error": "請先登入參數設定"}, status=401)
            else:
                next_qs = urllib.parse.quote(next_path or "/settings", safe="")
                self._redirect(f"/settings/login?next={next_qs}")
            return None
        if admin and not user.get("can_manage_accounts"):
            self._send_json({"error": "只有 admin 可以執行此操作"}, status=403)
            return None
        return user

    def _send_404(self, path=""):
        self._send_error(404, "Page Not Found", path)

    def _send_error(self, code: int, message: str, path=""):
        html = """
        {% extends "base.html" %}
        {% block content %}
        <div class="card" style="border-color:var(--err)">
            <h2 style="color:var(--err)">Error """ + str(code) + """</h2>
            <p>""" + message + """</p>
        </div>
        {% endblock %}
        """
        try:
            template = self.jinja_env.from_string(html)
            rendered = template.render(request_path=path)
        except Exception:
            rendered = f"<h2>Error {code}</h2><p>{message}</p>"
        self._send_response(code, rendered)

    def _send_binary(self, filepath: str):
        """發送二進位檔案 (圖片等)"""
        path = Path(filepath)
        if not path.exists():
            self._send_404()
            return
        mime_type, _ = mimetypes.guess_type(str(path))
        
        # Fallback for .mjs if not in standard mimetypes
        if mime_type is None and str(path).endswith(".mjs"):
            mime_type = "application/javascript"
            
        if mime_type is None:
            mime_type = "application/octet-stream"
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    # ── Route Handlers ─────────────────────────────────

    def _handle_index(self, query: dict, path: str):
        """首頁 — 最近推論記錄（支援分頁）"""
        try:
            limit = int(query.get("limit", [50])[0])
            limit = max(1, min(limit, 500))  # 限制 1~500
        except (ValueError, TypeError):
            limit = 50
        try:
            page = int(query.get("page", [1])[0])
            page = max(1, page)
        except (ValueError, TypeError):
            page = 1
        auto_refresh_seconds = 0
        if "auto_refresh" in query:
            try:
                auto_refresh_seconds = int(query.get("auto_refresh", ["300"])[0])
            except (ValueError, TypeError, IndexError):
                auto_refresh_seconds = 300
            auto_refresh_seconds = max(60, min(auto_refresh_seconds, 3600))

        offset = (page - 1) * limit
        records, total_count = self.db.query_paged(limit, offset) if self.db else ([], 0)
        shift_stats = self.db.get_shift_statistics() if self.db else {}

        # 計算 OK/NG 比率（異常 HY 不列入分母）
        s_total = shift_stats.get('total', 0) or 0
        s_ok = shift_stats.get('ok_count', 0) or 0
        s_ng = shift_stats.get('ng_count', 0) or 0
        s_err = shift_stats.get('err_count', 0) or 0
        s_denom = s_total - s_err
        ok_rate = (s_ok / s_denom * 100) if s_denom > 0 else 0
        ng_rate = (s_ng / s_denom * 100) if s_denom > 0 else 0

        import math
        total_pages = max(1, math.ceil(total_count / limit))

        template = self.jinja_env.get_template("index.html")
        html = template.render(
            records=records,
            stats=shift_stats,
            ok_rate=ok_rate,
            ng_rate=ng_rate,
            request_path=path,
            page=page,
            limit=limit,
            total_count=total_count,
            total_pages=total_pages,
            auto_refresh_seconds=auto_refresh_seconds,
        )
        self._send_response(200, html)

    @staticmethod
    def _render_changelog_html(markdown_text: str) -> str:
        lines = markdown_text.splitlines()
        parts = []
        in_list = False

        def close_list():
            nonlocal in_list
            if in_list:
                parts.append("</ul>")
                in_list = False

        for raw in lines:
            line = raw.strip()
            if not line:
                close_list()
                continue
            if line.startswith("### "):
                close_list()
                parts.append(f"<h3>{html.escape(line[4:])}</h3>")
            elif line.startswith("## "):
                close_list()
                parts.append(f"<h2>{html.escape(line[3:])}</h2>")
            elif line.startswith("# "):
                close_list()
                parts.append(f"<h1>{html.escape(line[2:])}</h1>")
            elif line.startswith("- "):
                if not in_list:
                    parts.append("<ul>")
                    in_list = True
                parts.append(f"<li>{html.escape(line[2:])}</li>")
            else:
                close_list()
                parts.append(f"<p>{html.escape(line)}</p>")

        close_list()
        return "\n".join(parts)

    def _handle_release_notes_page(self, path: str):
        template = self.jinja_env.get_template("release_notes.html")
        html_text = self._render_changelog_html(read_changelog())
        rendered = template.render(
            request_path=path,
            release_notes_html=html_text,
        )
        self._send_response(200, rendered)

    def _handle_overexposed(self, query: dict, path: str):
        """過曝記錄列表"""
        try:
            limit = int(query.get("limit", [50])[0])
            limit = max(1, min(limit, 500))
        except (ValueError, TypeError):
            limit = 50
        try:
            page = int(query.get("page", [1])[0])
            page = max(1, page)
        except (ValueError, TypeError):
            page = 1

        offset = (page - 1) * limit
        records, total_count = self.db.query_overexposed(limit, offset) if self.db else ([], 0)

        import math
        total_pages = max(1, math.ceil(total_count / limit))

        template = self.jinja_env.get_template("overexposed.html")
        html = template.render(
            records=records,
            total_count=total_count,
            request_path=path,
            page=page,
            limit=limit,
            total_pages=total_pages,
        )
        self._send_response(200, html)

    def _handle_record_detail(self, record_id_str: str, path: str):
        """記錄詳情頁"""
        try:
            record_id = int(record_id_str)
        except ValueError:
            self._send_404(path)
            return

        detail = self.db.get_record_detail(record_id) if self.db else None
        if not detail:
            self._send_404(path)
            return
        self._decorate_record_image_prefix_labels(detail)
        self._decorate_record_preprocess_info(detail)

        template = self.jinja_env.get_template("record_detail.html")
        html = template.render(
            detail=detail,
            heatmap_base_dir=self.heatmap_base_dir,
            request_path=path
        )
        self._send_response(200, html)

    def _handle_search(self, query: dict, path: str):
        """搜尋頁面（含日期篩選、分頁）"""
        record_id = query.get("record_id", [""])[0]
        glass_id = query.get("glass_id", [""])[0]
        machine_no = query.get("machine_no", [""])[0]
        ai_judgment = query.get("ai_judgment", [""])[0]
        start_date = query.get("start_date", [""])[0]
        end_date = query.get("end_date", [""])[0]
        cross_filter = query.get("cross_filter", [""])[0]

        # 預設顯示近 7 天（首次進入頁面時）
        if not any([record_id, glass_id, machine_no, ai_judgment, start_date, end_date, cross_filter]):
            today = datetime.now()
            end_date = today.strftime("%Y-%m-%d")
            start_date = (today - timedelta(days=6)).strftime("%Y-%m-%d")

        per_page = 50
        try:
            page = max(1, int(query.get("page", ["1"])[0]))
        except (ValueError, IndexError):
            page = 1

        # 先查總數以便在查詢前校正頁碼
        records = []
        total_count = 0
        if self.db:
            records, total_count = self.db.search_records(
                glass_id=glass_id,
                machine_no=machine_no,
                ai_judgment=ai_judgment,
                start_date=start_date,
                end_date=end_date,
                cross_filter=cross_filter,
                record_id=record_id,
                limit=per_page,
                offset=(page - 1) * per_page,
            )

        total_pages = max(1, (total_count + per_page - 1) // per_page)
        if page > total_pages:
            page = total_pages

        template = self.jinja_env.get_template("search.html")
        html = template.render(
            record_id=record_id,
            glass_id=glass_id,
            machine_no=machine_no,
            ai_judgment=ai_judgment,
            start_date=start_date,
            end_date=end_date,
            cross_filter=cross_filter,
            records=records,
            total_count=total_count,
            page=page,
            total_pages=total_pages,
            request_path=path
        )
        self._send_response(200, html)

    def _handle_search_export(self, query: dict):
        """匯出搜尋結果為 CSV"""
        import csv
        import io
        from datetime import datetime as _dt

        record_id  = query.get("record_id",   [""])[0]
        glass_id   = query.get("glass_id",    [""])[0]
        machine_no = query.get("machine_no",  [""])[0]
        ai_judgment = query.get("ai_judgment", [""])[0]
        start_date = query.get("start_date",  [""])[0]
        end_date   = query.get("end_date",    [""])[0]
        cross_filter = query.get("cross_filter", [""])[0]

        records, _ = self.db.search_records(
            glass_id=glass_id,
            machine_no=machine_no,
            ai_judgment=ai_judgment,
            start_date=start_date,
            end_date=end_date,
            cross_filter=cross_filter,
            record_id=record_id,
            limit=10000,
        ) if self.db else ([], 0)

        # 建立 CSV 內容（UTF-8 BOM，讓 Excel 正常顯示中文）
        buf = io.StringIO()
        buf.write("\ufeff")  # BOM
        writer = csv.writer(buf)
        writer.writerow(["ID", "玻璃編號", "機種", "機台", "機檢判定", "AI判定", "耗時(s)", "建立時間", "圖片數", "NG圖片數"])
        for r in records:
            writer.writerow([
                r.get("id", ""),
                r.get("glass_id", ""),
                r.get("model_id", ""),
                r.get("machine_no", ""),
                r.get("machine_judgment", ""),
                r.get("ai_judgment", ""),
                r.get("processing_seconds", ""),
                r.get("created_at", ""),
                r.get("total_images", ""),
                r.get("ng_images", ""),
            ])

        csv_bytes = buf.getvalue().encode("utf-8-sig")

        # 組出有意義的檔名
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        date_part = ""
        if start_date or end_date:
            date_part = f"_{start_date or ''}~{end_date or ''}"
        filename = f"capi_records{date_part}_{ts}.csv"

        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8-sig")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(csv_bytes)))
        self.end_headers()
        self.wfile.write(csv_bytes)

    def _handle_dashboard(self, query: dict, path: str):
        """即時儀表板頁面"""
        template = self.jinja_env.get_template("dashboard.html")
        html = template.render(request_path=path)
        self._send_response(200, html)

    def _handle_dashboard_v2(self, query: dict, path: str):
        """舊版 V2 路由相容別名，改用現存的 V3 儀表板。"""
        self._handle_dashboard_v3(query, path)
        
    def _handle_dashboard_v3(self, query: dict, path: str):
        """V3 高階 UI 即時儀表板頁面"""
        template = self.jinja_env.get_template("dashboard_v3.html")
        html = template.render(request_path=path)
        self._send_response(200, html)

    def _handle_record_detail_v3(self, record_id_str: str, path: str):
        """V3 高階 UI 記錄詳情頁"""
        try:
            record_id = int(record_id_str)
        except ValueError:
            self._send_404(path)
            return

        detail = self.db.get_record_detail(record_id) if self.db else None
        if not detail:
            self._send_404(path)
            return
        self._decorate_record_image_prefix_labels(detail)
        self._decorate_record_preprocess_info(detail)

        template = self.jinja_env.get_template("record_detail_v3.html")
        html = template.render(
            detail=detail,
            heatmap_base_dir=self.heatmap_base_dir,
            request_path=path
        )
        self._send_response(200, html)

    @staticmethod
    def _decorate_record_image_prefix_labels(detail: dict) -> None:
        detail["image_prefix_labels"] = image_prefix_display_labels(
            image.get("image_name") or image.get("image_path") or ""
            for image in detail.get("images") or []
        )

    @staticmethod
    def _decorate_record_preprocess_info(detail: dict) -> None:
        """Add display-ready preprocessing metadata to an inference record."""
        raw = detail.get("image_preprocess_pipeline")
        raw_zones = detail.get("image_preprocess_pipelines")
        detail["image_preprocess_pipeline_recorded"] = False
        detail["image_preprocess_pipeline_steps"] = []
        detail["image_preprocess_pipeline_summary"] = "舊紀錄未記錄"
        detail["image_preprocess_timing_recorded"] = False
        detail["image_preprocess_total_seconds_text"] = ""

        if raw_zones not in (None, ""):
            try:
                pipelines = (
                    json.loads(raw_zones)
                    if isinstance(raw_zones, str)
                    else raw_zones
                )
                if not isinstance(pipelines, dict) or set(pipelines) != {"inner", "edge"}:
                    raise ValueError("invalid zone pipeline map")
                from capi_image_preprocess_lab import get_method_specs, normalize_preprocess_pipeline
                normalized_zones = {
                    zone: normalize_preprocess_pipeline(pipelines.get(zone) or [])
                    for zone in ("inner", "edge")
                }
                specs = {spec["id"]: spec for spec in get_method_specs()}
            except Exception:
                detail["image_preprocess_pipeline_recorded"] = True
                detail["image_preprocess_pipeline_summary"] = "分區前處理設定解析失敗"
                return

            steps = []
            zone_summaries = []
            for zone in ("inner", "edge"):
                summary_parts = []
                for idx, step in enumerate(normalized_zones[zone], 1):
                    method = step["method"]
                    spec = specs.get(method, {})
                    label = spec.get("label") or method
                    param_labels = {
                        item.get("key"): item.get("name") or item.get("key")
                        for item in spec.get("params", [])
                        if item.get("key")
                    }
                    params = [
                        {
                            "key": key,
                            "label": param_labels.get(key, key),
                            "value": value,
                        }
                        for key, value in step.get("params", {}).items()
                    ]
                    params_text = ", ".join(
                        f"{item['label']}={item['value']}" for item in params
                    )
                    steps.append({
                        "index": idx,
                        "zone_label": zone.upper(),
                        "method": method,
                        "method_label": label,
                        "params": params,
                        "params_text": params_text,
                        "timing_text": "",
                        "elapsed_ms_total": 0.0,
                        "elapsed_ms_avg": 0.0,
                        "calls": 0,
                    })
                    summary_parts.append(f"{idx}.{label}({params_text})")
                zone_summaries.append(
                    f"{zone.upper()}: "
                    + (" -> ".join(summary_parts) if summary_parts else "未啟用")
                )
            detail["image_preprocess_pipeline_recorded"] = True
            detail["image_preprocess_pipeline_steps"] = steps
            detail["image_preprocess_pipeline_summary"] = "；".join(zone_summaries)
            timing_raw = detail.get("image_preprocess_timing")
            if timing_raw:
                try:
                    timing = (
                        json.loads(timing_raw)
                        if isinstance(timing_raw, str)
                        else timing_raw
                    )
                    detail["image_preprocess_timing_recorded"] = True
                    total_ms = float((timing or {}).get("total_elapsed_ms") or 0.0)
                    detail["image_preprocess_total_seconds_text"] = (
                        f"{total_ms / 1000.0:.3f}s"
                    )
                except Exception:
                    detail["image_preprocess_timing_recorded"] = False
            return

        if raw is None or raw == "":
            return

        try:
            pipeline = json.loads(raw) if isinstance(raw, str) else raw
            from capi_image_preprocess_lab import get_method_specs, normalize_preprocess_pipeline
            normalized = normalize_preprocess_pipeline(pipeline)
            specs = {spec["id"]: spec for spec in get_method_specs()}
        except Exception:
            detail["image_preprocess_pipeline_recorded"] = True
            detail["image_preprocess_pipeline_summary"] = "前處理設定解析失敗"
            return

        detail["image_preprocess_pipeline_recorded"] = True
        if not normalized:
            detail["image_preprocess_pipeline_summary"] = "未啟用"
            return

        timing_by_key = {}
        timing_raw = detail.get("image_preprocess_timing")
        if timing_raw:
            try:
                timing = json.loads(timing_raw) if isinstance(timing_raw, str) else timing_raw
                detail["image_preprocess_timing_recorded"] = True
                total_ms = float((timing or {}).get("total_elapsed_ms") or 0.0)
                detail["image_preprocess_total_seconds_text"] = f"{total_ms / 1000.0:.3f}s"
                for item in (timing or {}).get("steps", []) or []:
                    method = str(item.get("method") or "")
                    try:
                        params_key = json.dumps(
                            item.get("applied_params") or {},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    except TypeError:
                        params_key = str(item.get("applied_params") or {})
                    timing_by_key[(int(item.get("index") or 0), method, params_key)] = item
            except Exception:
                detail["image_preprocess_timing_recorded"] = False

        steps = []
        summary_parts = []
        for idx, step in enumerate(normalized, 1):
            method = step["method"]
            spec = specs.get(method, {})
            label = spec.get("label") or method
            param_labels = {
                p.get("key"): p.get("name") or p.get("key")
                for p in spec.get("params", [])
                if p.get("key")
            }
            params = []
            for key, value in step.get("params", {}).items():
                params.append({
                    "key": key,
                    "label": param_labels.get(key, key),
                    "value": value,
                })
            params_text = ", ".join(f"{p['label']}={p['value']}" for p in params)
            try:
                params_key = json.dumps(step.get("params", {}), ensure_ascii=False, sort_keys=True)
            except TypeError:
                params_key = str(step.get("params", {}))
            timing_info = timing_by_key.get((idx, method, params_key), {})
            elapsed_ms = float(timing_info.get("elapsed_ms_total") or 0.0)
            avg_ms = float(timing_info.get("elapsed_ms_avg") or 0.0)
            calls = int(timing_info.get("calls") or 0)
            timing_text = ""
            if calls:
                timing_text = f"耗時 {elapsed_ms / 1000.0:.3f}s"
                if calls > 1:
                    timing_text += f" / {calls} 次 / 平均 {avg_ms:.2f}ms"
            steps.append({
                "index": idx,
                "method": method,
                "method_label": label,
                "params": params,
                "params_text": params_text,
                "timing_text": timing_text,
                "elapsed_ms_total": elapsed_ms,
                "elapsed_ms_avg": avg_ms,
                "calls": calls,
            })
            summary_parts.append(f"{idx}.{label}({params_text})")

        detail["image_preprocess_pipeline_steps"] = steps
        detail["image_preprocess_pipeline_summary"] = " -> ".join(summary_parts)
        
    def _handle_logs_page(self, query: dict, path: str):
        """Log Viewer 頁面"""
        template = self.jinja_env.get_template("logs.html")
        # 列出可用的 log 檔案 (current + rotated)
        log_files = []
        if self._log_file:
            log_path = Path(self._log_file)
            if log_path.exists():
                log_files.append({"name": log_path.name, "path": str(log_path), "size": log_path.stat().st_size})
            # rotated files: server.log.1, server.log.2, ...
            for i in range(1, 10):
                rotated = log_path.parent / f"{log_path.name}.{i}"
                if rotated.exists():
                    log_files.append({"name": rotated.name, "path": str(rotated), "size": rotated.stat().st_size})
        html = template.render(request_path=path, log_files=log_files, log_configured=bool(self._log_file))
        self._send_response(200, html)

    def _handle_api_logs(self, query: dict):
        """API: 讀取 log 檔案內容"""
        if not self._log_file:
            self._send_json({"error": "未設定日誌檔案路徑", "lines": []})
            return

        # 選擇要讀取的 log 檔案 (支援 rotated files)
        file_index = int(query.get("file", [0])[0])  # 0=current, 1=.1, 2=.2, ...
        tail_lines = int(query.get("lines", [500])[0])
        tail_lines = min(tail_lines, 5000)  # 上限 5000 行
        search = query.get("search", [""])[0]
        level_filter = query.get("level", [""])[0].upper()

        log_path = Path(self._log_file)
        if file_index > 0:
            log_path = log_path.parent / f"{log_path.name}.{file_index}"

        if not log_path.exists():
            self._send_json({"error": f"日誌檔案不存在: {log_path.name}", "lines": []})
            return

        try:
            # 讀取最後 N 行 (高效能 tail)
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                # 快速 tail: 讀取末尾 chunk
                f.seek(0, 2)
                file_size = f.tell()
                # 估算每行約 200 bytes, 多讀一些
                chunk_size = min(file_size, tail_lines * 300)
                f.seek(max(0, file_size - chunk_size))
                if f.tell() > 0:
                    f.readline()  # 跳過不完整的第一行
                all_lines = f.readlines()
                lines = all_lines[-tail_lines:]

            # 過濾
            if level_filter:
                lines = [l for l in lines if level_filter in l]
            if search:
                search_lower = search.lower()
                lines = [l for l in lines if search_lower in l.lower()]

            self._send_json({
                "file": log_path.name,
                "total_lines": len(lines),
                "lines": [l.rstrip("\n\r") for l in lines],
            })
        except Exception as e:
            self._send_json({"error": str(e), "lines": []})

    def _handle_api_status(self):
        """API: 即時伺服器狀態"""
        cors_headers = {"Access-Control-Allow-Origin": "*"}
        try:
            if hasattr(self, 'status_tracker') and self.status_tracker:
                status = self.status_tracker.get_status()
            else:
                from capi_server import server_status
                status = server_status.get_status()

            status.setdefault("server", {})["hostname"] = _get_host_identity()
            status["update"] = self._get_update_status_payload()

            # 將當班統計數據替換為 DB (支援重啟後恢復)
            if self.db:
                shift_stats = self.db.get_shift_statistics()
                status.setdefault("stats", {})
                status["stats"]["total_requests"] = shift_stats.get("total", 0) or 0
                status["stats"]["total_ok"] = shift_stats.get("ok_count", 0) or 0
                status["stats"]["total_ng"] = shift_stats.get("ng_count", 0) or 0
                status["stats"]["aoi_ng_count"] = shift_stats.get("aoi_ng_count", 0) or 0
                status["stats"]["ai_ng_count"] = shift_stats.get("ng_count", 0) or 0
                status["stats"]["total_err"] = shift_stats.get("err_count", 0) or 0
                status["stats"]["shift_name"] = shift_stats.get("shift_name", "當班")
                status["stats"]["time_range"] = shift_stats.get("time_range", "")
                status["stats"]["avg_time"] = shift_stats.get("avg_time")
                status["stats"]["overexposed_count"] = shift_stats.get("overexposed_count", 0) or 0

                # 取最近 1 筆 image_results (有熱力圖)
                try:
                    import sqlite3 as _sqlite3
                    from pathlib import Path as _Path

                    conn = _sqlite3.connect(str(self.db.db_path), timeout=5)
                    conn.row_factory = _sqlite3.Row

                    # 按時間最新，取最新 1 筆 (無論 OK/NG)
                    row = conn.execute(
                        """SELECT
                               ir.id             AS img_id,
                               ir.record_id,
                               ir.heatmap_path,
                               ir.is_ng,
                               ir.image_name,
                               rec.glass_id,
                               rec.ai_judgment,
                               rec.created_at
                           FROM image_results ir
                           JOIN inference_records rec ON rec.id = ir.record_id
                           WHERE ir.heatmap_path != ''
                           ORDER BY rec.created_at DESC, ir.id DESC
                           LIMIT 1"""
                    ).fetchone()

                    recent_heatmaps = []
                    base = self.heatmap_base_dir
                    
                    if row:
                        # 1. 放入 Overview
                        hm_abs = row["heatmap_path"]
                        url = None
                        if hm_abs and base:
                            try:
                                rel = str(_Path(hm_abs).relative_to(base)).replace("\\", "/")
                                url = f"/heatmaps/{rel}"
                            except ValueError:
                                url = None
                        if url:
                            recent_heatmaps.append({
                                "url":        url,
                                "glass_id":   row["glass_id"] or "",
                                "image_name": row["image_name"] or "",
                                "is_ng":      bool(row["is_ng"]),
                                "status":     "NG" if row["is_ng"] else "OK",
                                "label":      "Overview",
                                "judgment":   row["ai_judgment"] or "",
                                "created_at": row["created_at"] or "",
                                "record_id":  row["record_id"],
                            })
                            
                        # 2. 抓取區域熱力圖 (Tiles)，最多取 8 張以填好 9 宮格
                        img_id = row["img_id"]
                        tiles = conn.execute(
                            """SELECT tile_id, heatmap_path, is_anomaly, is_dust, is_bomb, is_aoi_coord
                               FROM tile_results
                               WHERE image_result_id = ? AND heatmap_path != ''
                               ORDER BY is_anomaly DESC, score DESC
                               LIMIT 8""",
                            (img_id,)
                        ).fetchall()
                        
                        for t in tiles:
                            thm_abs = t["heatmap_path"]
                            t_url = None
                            if thm_abs and base:
                                try:
                                    t_rel = str(_Path(thm_abs).relative_to(base)).replace("\\", "/")
                                    t_url = f"/heatmaps/{t_rel}"
                                except ValueError:
                                    t_url = None
                            if t_url:
                                is_bomb = bool(t["is_bomb"])
                                is_dust = bool(t["is_dust"])
                                is_ng = bool(t["is_anomaly"])
                                
                                is_exclude_zone = bool(t.get("is_exclude_zone", 0))
                                is_aoi_coord = bool(t.get("is_aoi_coord", 0))

                                tile_status = "OK"
                                if is_exclude_zone:
                                    tile_status = "EXCLUDED"
                                elif is_bomb:
                                    tile_status = "BOMB"
                                elif is_dust:
                                    tile_status = "DUST"
                                elif is_ng:
                                    tile_status = "NG"

                                tile_label = f"Tile #{t['tile_id']}"
                                if is_aoi_coord:
                                    tile_label = f"🎯 AOI #{t['tile_id']}"

                                recent_heatmaps.append({
                                    "url":        t_url,
                                    "glass_id":   row["glass_id"] or "",
                                    "image_name": row["image_name"] or "",
                                    "is_ng":      is_ng or is_bomb or is_dust,
                                    "status":     tile_status,
                                    "label":      tile_label,
                                    "judgment":   row["ai_judgment"] or "",
                                    "created_at": row["created_at"] or "",
                                    "record_id":  row["record_id"],
                                })

                    conn.close()
                    status["recent_heatmaps"] = recent_heatmaps
                except Exception:
                    status["recent_heatmaps"] = []

                threshold_mapping = status.get("server", {}).get("threshold_mapping") or {}
                status.setdefault("server", {})["image_prefix_labels"] = \
                    self._dashboard_lighting_labels(self.db, list(threshold_mapping))

            db_path = getattr(self.db, "db_path", None) if self.db else None
            if db_path:
                disk_path = Path(db_path).expanduser().parent
            else:
                disk_path = getattr(self, "heatmap_base_dir", None) or Path.cwd()
            status["hardware"] = _get_cached_hardware_status(disk_path)

            self._send_json(status, headers=cors_headers)
        except Exception as e:
            self._send_json(
                {"error": f"Cannot get server status: {e}"},
                status=500,
                headers=cors_headers,
            )

    def _handle_api_version(self):
        """API: deployed release version."""
        self._send_json(get_version_info())

    def _get_update_status_payload(self) -> Dict[str, Any]:
        """Return the sanitized update state shared by local and central dashboards."""
        from capi_update_agent import _load_state

        state = _load_state(self._update_state_file)
        pending = state.get("pending_update") if isinstance(state.get("pending_update"), dict) else {}
        failed = state.get("last_failed") if isinstance(state.get("last_failed"), dict) else {}
        current_version = str(get_version_info().get("version") or "unknown")
        pending_version = str(pending.get("version") or "")
        status = str(state.get("status") or ("pending" if pending_version else "current"))

        if pending_version == current_version:
            pending_version = ""
            status = "current"
        elif not pending_version and status not in {"failed", "apply_requested", "installing"}:
            status = "current"

        return {
            "status": status,
            "current_version": current_version,
            "pending_version": pending_version,
            "detected_at": pending.get("detected_at"),
            "can_apply": bool(pending_version) and status in {"pending", "failed"},
            "failure_reason": str(failed.get("reason") or "")[:240],
            "central_apply_supported": True,
        }

    def _handle_api_update_status(self):
        """API: sanitized pending-update status for the frontend notice."""
        self._send_json(self._get_update_status_payload())

    @classmethod
    def _active_training_update_blocker(cls) -> str:
        """Return the active training job that makes a service restart unsafe."""
        for state_name, label in (
            ("_retrain_state", "刮痕分類器重訓"),
            ("_submodel_retrain_state", "PatchCore 子模型重訓"),
        ):
            state = getattr(cls, state_name, None)
            if not isinstance(state, dict):
                continue
            lock = state.get("lock")
            if lock is None:
                job = state.get("job")
            else:
                with lock:
                    job = state.get("job")
            if isinstance(job, dict) and job.get("state") == "running":
                job_id = str(job.get("job_id") or "").strip()
                return f"{label} {job_id}".strip()

        with cls._train_new_jobs_lock:
            runtimes = list(cls._train_new_jobs.items())
        for job_id, runtime in runtimes:
            if runtime.get("phase") not in {"preprocess", "train"}:
                continue
            thread = runtime.get("thread")
            proc = runtime.get("proc")
            thread_alive = thread is not None and thread.is_alive()
            proc_alive = proc is not None and proc.poll() is None
            if thread_alive or proc_alive:
                return f"PatchCore 訓練 {job_id}"

        slot = cls._train_slot
        with slot["lock"]:
            active_job_id = str(slot.get("active_job_id") or "").strip()
        if active_job_id:
            return f"PatchCore 訓練 {active_job_id}"
        return ""

    def _handle_api_central_update_apply(self):
        """Apply an update requested by the configured central dashboard host."""
        source_ip = str(getattr(self, "client_address", ("", 0))[0] or "").strip()
        center_ip = str(self._load_central_account_location().get("ip") or "").strip()
        trusted_header = str(
            self.headers.get(CENTRAL_UPDATE_AUTH_HEADER, "") if self.headers else ""
        )
        if trusted_header != "1" or not source_ip or source_ip != center_ip:
            logger.warning(
                "Rejected central update request source=%s configured_center=%s",
                source_ip or "unknown",
                center_ip or "unknown",
            )
            self._send_json({"error": "不允許的中央更新來源"}, status=403)
            return

        data = self._read_json_body()
        if data is None:
            return
        expected_version = str(data.get("expectedVersion") or "").strip()
        if not expected_version:
            self._send_json({"error": "缺少預期更新版本"}, status=400)
            return
        requested_by = urllib.parse.unquote(
            str(self.headers.get(CENTRAL_UPDATE_USER_HEADER, "") or "")
        ).strip()[:64]
        logger.info(
            "Accepted central update request source=%s requested_by=%s version=%s",
            source_ip,
            requested_by or "unknown",
            expected_version,
        )
        self._handle_api_update_apply(
            expected_version=expected_version,
            requested_by=requested_by,
            request_source="central_dashboard",
        )

    def _handle_api_update_apply(
        self,
        *,
        expected_version: str = "",
        requested_by: str = "",
        request_source: str = "local",
    ):
        """Launch the staged updater outside the server process, then return 202."""
        from capi_update_agent import _load_state, _write_state

        tracker = getattr(self, "status_tracker", None)
        if tracker is not None:
            runtime_status = tracker.get_status()
            active_inferences = int(runtime_status.get("traffic", {}).get("active_inferences") or 0)
            if active_inferences > 0:
                self._send_json({
                    "error": f"目前仍有 {active_inferences} 筆檢測進行中，請稍後再試",
                }, status=409)
                return

        state_file = self._update_state_file
        app_root = Path(__file__).resolve().parent
        with self._update_apply_lock:
            active_training = self._active_training_update_blocker()
            if active_training:
                logger.warning(
                    "Rejected update while training is active: %s",
                    active_training,
                )
                self._send_json({
                    "error": (
                        f"目前有訓練工作進行中（{active_training}），"
                        "請等待完成或取消後再更新"
                    ),
                }, status=409)
                return

            state = _load_state(state_file)
            pending = state.get("pending_update")
            if not isinstance(pending, dict):
                self._send_json({"error": "目前沒有待套用的更新"}, status=409)
                return
            if state.get("status") in {"apply_requested", "installing"}:
                self._send_json({"error": "更新已在執行中"}, status=409)
                return

            version = str(pending.get("version") or "").strip()
            expected_version = str(expected_version or "").strip()
            if expected_version and version != expected_version:
                self._send_json({
                    "error": (
                        f"待更新版本已變更（目前 {version or '未知'}），"
                        "請重新整理後再試"
                    ),
                }, status=409)
                return
            state["status"] = "apply_requested"
            state["apply_requested"] = {
                "version": version,
                "at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "requested_by": str(requested_by or "").strip()[:64],
                "source": str(request_source or "local").strip()[:32],
            }
            _write_state(state_file, state)
            logger.info(
                "Update apply requested source=%s requested_by=%s version=%s",
                request_source or "local",
                requested_by or "unknown",
                version,
            )

            command = [
                sys.executable,
                str(app_root / "capi_update_agent.py"),
                "apply",
                "--app-root",
                str(app_root),
                "--state-file",
                str(state_file),
                "--delay",
                "2",
            ]
            apply_log = state_file.parent / "manual_apply.log"
            apply_log.parent.mkdir(parents=True, exist_ok=True)
            try:
                with apply_log.open("a", encoding="utf-8") as output:
                    popen_kwargs = {
                        "cwd": app_root,
                        "stdout": output,
                        "stderr": subprocess.STDOUT,
                    }
                    if os.name != "nt":
                        popen_kwargs["start_new_session"] = True
                    subprocess.Popen(command, **popen_kwargs)
            except Exception as exc:
                logger.error("Cannot start manual update process: %s", exc)
                state.pop("apply_requested", None)
                state["status"] = "failed"
                state["last_failed"] = {
                    "version": version,
                    "at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "reason": "cannot start manual update process",
                }
                _write_state(state_file, state)
                self._send_json({"error": "無法啟動更新程序"}, status=500)
                return

        self._send_json({
            "status": "apply_requested",
            "version": version,
            "message": "更新程序已啟動，服務即將重新啟動",
        }, status=202)

    def _handle_api_stats(self, query: dict):
        """API: 統計資料"""
        try:
            days = int(query.get("days", [7])[0])
        except (ValueError, TypeError):
            days = 7
        try:
            limit = int(query.get("limit", [15])[0])
        except (ValueError, TypeError):
            limit = 15
        
        stats = self.db.get_statistics(days) if self.db else {}
        
        # 附加最近的一批記錄給 dashboard_v2
        recent_records = []
        if self.db:
            recent_list = self.db.query_recent(limit)
            for r in recent_list:
                rec_dict = dict(r)
                hm_path = rec_dict.get("first_heatmap_path")
                if hm_path and self.heatmap_base_dir:
                    rec_dict["hm_url"] = hm_relative(hm_path, self.heatmap_base_dir)
                recent_records.append(rec_dict)
                
        stats["recent_records"] = recent_records
        
        self._send_json(stats)

    def _handle_static_file(self, path: str):
        """靜態檔案服務 (熱力圖)"""
        # /heatmaps/20260225/GLASS001/overview_G0F00000.png
        rel_path = path[len("/heatmaps/"):]
        # 安全檢查：防止路徑穿越
        rel_path = rel_path.replace("..", "").lstrip("/")
        full_path = Path(self.heatmap_base_dir) / rel_path
        if full_path.exists() and full_path.is_file():
            self._send_binary(str(full_path))
        else:
            self._send_404()

    def _inference_rotate_180_enabled(self) -> bool:
        config = getattr(getattr(self, "inferencer", None), "config", None)
        return bool(getattr(config, "inference_rotate_180_enabled", False))

    def _read_inference_image(self, image_path: Path, flags: int):
        return read_detection_image(
            image_path,
            flags,
            rotate_180=self._inference_rotate_180_enabled(),
        )

    def _send_image_array_png(self, image) -> None:
        import cv2
        import numpy as np

        if image.dtype != np.uint8:
            max_value = float(image.max())
            if max_value > 0:
                image = (image.astype(np.float32) / max_value * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            self._send_error(500, "PNG encode failed")
            return
        data = buf.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _handle_source_image(self, path: str):
        """靜態檔案服務 (原始圖片)"""
        # /images/{record_id}/{image_name}
        try:
            parts = path.strip("/").split("/")
            if len(parts) != 3:
                self._send_404()
                return
            record_id = int(parts[1])
            image_name = parts[2]
            
            # 安全檢查：防止路徑穿越
            image_name = image_name.replace("..", "").replace("/", "").replace("\\", "")
            
            detail = self.db.get_record_detail(record_id) if self.db else None
            if not detail or not detail.get("image_dir"):
                self._send_404()
                return
                
            full_path = Path(detail["image_dir"]) / image_name
            if full_path.exists() and full_path.is_file():
                if self._inference_rotate_180_enabled():
                    import cv2

                    image = self._read_inference_image(full_path, cv2.IMREAD_UNCHANGED)
                    if image is None:
                        self._send_404()
                        return
                    self._send_image_array_png(image)
                else:
                    self._send_binary(str(full_path))
            else:
                self._send_404()
        except Exception as e:
            logger.error(f"Error serving source image {path}: {e}")
            self._send_404()

    def _handle_imgs_file(self, path: str):
        """靜態檔案服務 (UI 圖片/影片)"""
        rel_path = path[len("/imgs/"):]
        rel_path = rel_path.replace("..", "").lstrip("/")
        full_path = Path(__file__).parent / "templates" / "imgs" / rel_path
        if full_path.exists() and full_path.is_file():
            self._send_binary(str(full_path))
        else:
            self._send_404()

    def _handle_static_assets(self, path: str):
        """靜態檔案服務 (CSS/JS)"""
        rel_path = path[len("/static/"):]
        rel_path = rel_path.replace("..", "").lstrip("/")
        full_path = Path(__file__).parent / "static" / rel_path
        if full_path.exists() and full_path.is_file():
            self._send_binary(str(full_path))
        else:
            self._send_404()

    def _handle_central_dashboard_file(self, path: str):
        """靜態檔案服務 (central_dashboard)"""
        rel_path = path[len("/central_dashboard/"):]
        rel_path = urllib.parse.unquote(rel_path).replace("..", "").replace("\\", "").lstrip("/")
        if not rel_path:
            rel_path = "index.html"
        full_path = Path(__file__).parent / "central_dashboard" / rel_path
        if full_path.exists() and full_path.is_file():
            self._send_binary(str(full_path))
        elif full_path.exists() and full_path.is_dir():
            index_path = full_path / "index.html"
            if index_path.exists() and index_path.is_file():
                self._send_binary(str(index_path))
            else:
                self._send_404()
        else:
            self._send_404()

    def _handle_api_central_dashboard_config(self):
        """公開提供與目前 Web Server 同廠區網段的中控看板設定。"""
        try:
            config = self.db.get_central_dashboard_config(
                self._load_central_dashboard_file_config()
            )
            webserver_ip = self._detect_central_dashboard_webserver_ip()
            prefix = self._central_dashboard_network_prefix(webserver_ip)
            configured_count = len(config.get("lines") or [])
            if prefix:
                config = dict(config)
                config["lines"] = [
                    line
                    for line in config.get("lines") or []
                    if self._central_dashboard_line_network_prefix(line) == prefix
                ]
            config["webServerIp"] = webserver_ip
            config["networkPrefix"] = prefix
            config["networkFilterApplied"] = bool(prefix)
            config["configuredLineCount"] = configured_count
            self._send_json(config)
        except Exception as exc:
            logger.error("Failed to load central dashboard config: %s", exc)
            self._send_json(
                {"error": "無法讀取中控看板設定"},
                status=500,
            )

    def _handle_api_central_dashboard_config_all(self):
        """設定頁使用：回傳所有廠區設備，不套用 Web Server 網段過濾。"""
        try:
            self._send_json(
                self.db.get_central_dashboard_config(
                    self._load_central_dashboard_file_config()
                )
            )
        except Exception as exc:
            logger.error("Failed to load all central dashboard config: %s", exc)
            self._send_json(
                {"error": "無法讀取完整中控看板設定"},
                status=500,
            )

    @staticmethod
    def _central_dashboard_network_prefix(value: str) -> str:
        try:
            address = ipaddress.ip_address(str(value or "").strip())
        except ValueError:
            return ""
        if address.version != 4 or address.packed[0] != 10:
            return ""
        return f"{address.packed[0]}.{address.packed[1]}"

    @classmethod
    def _central_dashboard_line_network_prefix(
        cls, line: Dict[str, Any]
    ) -> str:
        try:
            hostname = urllib.parse.urlparse(
                str(line.get("apiUrl") or "")
            ).hostname
        except (TypeError, ValueError):
            return ""
        return cls._central_dashboard_network_prefix(hostname or "")

    def _detect_central_dashboard_webserver_ip(self) -> str:
        candidates = []
        try:
            candidates.append(self.connection.getsockname()[0])
        except (AttributeError, OSError, TypeError):
            pass

        host_header = str(
            self.headers.get("Host", "") if self.headers else ""
        ).strip()
        if host_header:
            try:
                hostname = urllib.parse.urlsplit(f"//{host_header}").hostname
                if hostname:
                    candidates.append(hostname)
                    try:
                        candidates.append(socket.gethostbyname(hostname))
                    except OSError:
                        pass
            except ValueError:
                pass

        try:
            candidates.append(self.server.server_address[0])
        except (AttributeError, IndexError, TypeError):
            pass

        for candidate in candidates:
            if self._central_dashboard_network_prefix(candidate):
                return str(candidate)
        return ""

    @staticmethod
    def _load_central_dashboard_file_config() -> Optional[Dict[str, Any]]:
        """讀取既有 config.js，供 SQLite 第一次初始化時保留現場設定。"""
        config_path = Path(__file__).parent / "central_dashboard" / "config.js"
        try:
            source = config_path.read_text(encoding="utf-8")
            _, separator, assigned = source.partition("=")
            if not separator:
                return None
            object_source = assigned.strip()
            if object_source.endswith(";"):
                object_source = object_source[:-1]
            object_source = re.sub(
                r"([{,]\s*)([A-Za-z_$][A-Za-z0-9_$]*)(\s*:)",
                r'\1"\2"\3',
                object_source,
            )
            config = json.loads(object_source)
            return config if isinstance(config, dict) else None
        except Exception as exc:
            logger.warning(
                "Cannot import central_dashboard/config.js; using defaults: %s",
                exc,
            )
            return None

    def _handle_api_central_dashboard_config_update(self):
        """更新中控看板顯示設定；路由層已要求參數設定登入。"""
        data = self._read_json_body()
        if data is None:
            return
        try:
            user = self._current_settings_user() or {}
            config = self.db.save_central_dashboard_config(
                data,
                changed_by=user.get("username", ""),
            )
            self._send_json(
                {
                    "success": True,
                    "config": config,
                }
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            logger.error("Failed to update central dashboard config: %s", exc)
            self._send_json(
                {"error": "無法儲存中控看板設定"},
                status=500,
            )

    def _handle_api_central_dashboard_update_apply(self, user: Dict[str, Any]):
        """Proxy an admin-approved update request to one configured CAPI device."""
        data = self._read_json_body()
        if data is None:
            return
        line_id = str(data.get("lineId") or "").strip()
        expected_version = str(data.get("expectedVersion") or "").strip()
        if not line_id or not expected_version:
            self._send_json({"error": "缺少線體或更新版本"}, status=400)
            return

        config = self.db.get_central_dashboard_config(
            self._load_central_dashboard_file_config()
        )
        line = next(
            (
                item
                for item in config.get("lines") or []
                if item.get("id") == line_id and item.get("enabled") is not False
            ),
            None,
        )
        if not line:
            self._send_json({"error": "找不到已啟用的線體設定"}, status=404)
            return

        api_url = str(line.get("apiUrl") or "").strip()
        try:
            parsed = urllib.parse.urlparse(api_url)
            target_ip = ipaddress.ip_address(parsed.hostname or "")
            if (
                parsed.scheme not in {"http", "https"}
                or target_ip.version != 4
                or target_ip.packed[0] != 10
            ):
                raise ValueError
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except (ValueError, TypeError):
            self._send_json({"error": "設備更新目標必須是 10.x.x.x 的 HTTP(S) 位址"}, status=400)
            return

        webserver_ip = self._detect_central_dashboard_webserver_ip()
        webserver_prefix = self._central_dashboard_network_prefix(webserver_ip)
        target_prefix = self._central_dashboard_network_prefix(str(target_ip))
        if webserver_prefix and target_prefix != webserver_prefix:
            self._send_json({"error": "設備不屬於此中控看板網段"}, status=403)
            return

        body = json.dumps(
            {"expectedVersion": expected_version},
            ensure_ascii=False,
        ).encode("utf-8")
        requested_by = str(user.get("username") or "").strip()[:64]
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            CENTRAL_UPDATE_AUTH_HEADER: "1",
            CENTRAL_UPDATE_USER_HEADER: urllib.parse.quote(requested_by, safe=""),
        }
        timeout_seconds = min(
            15.0,
            max(
                3.0,
                float(config.get("requestTimeoutSeconds") or CENTRAL_UPDATE_TIMEOUT_SECONDS),
            ),
        )
        connection_class = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = None
        try:
            connection = connection_class(
                str(target_ip),
                port,
                timeout=timeout_seconds,
            )
            connection.request(
                "POST",
                CENTRAL_UPDATE_APPLY_PATH,
                body=body,
                headers=headers,
            )
            response = connection.getresponse()
            response_status = int(response.status)
            response_body = response.read()
        except (OSError, http.client.HTTPException) as exc:
            logger.warning(
                "Central dashboard cannot reach update endpoint line=%s target=%s: %s",
                line_id,
                target_ip,
                exc,
            )
            self._send_json({"error": "無法連線至設備更新服務"}, status=502)
            return
        finally:
            if connection is not None:
                connection.close()

        try:
            response_payload = json.loads(response_body.decode("utf-8"))
            if not isinstance(response_payload, dict):
                response_payload = {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            response_payload = {}

        if response_status == 202:
            logger.info(
                "Central dashboard update started line=%s target=%s requested_by=%s version=%s",
                line_id,
                target_ip,
                requested_by or "unknown",
                expected_version,
            )
            self._send_json({
                "success": True,
                "lineId": line_id,
                "status": "apply_requested",
                "version": str(response_payload.get("version") or expected_version),
                "message": "更新程序已啟動，設備即將重新啟動",
            }, status=202)
            return

        if response_status == 404:
            self._send_json({
                "error": "設備版本尚未支援中央直接更新，需先安裝本版一次",
            }, status=409)
            return
        error = str(response_payload.get("error") or "").strip()
        if response_status in {400, 403, 409}:
            self._send_json(
                {"error": error or "設備拒絕更新要求"},
                status=response_status,
            )
            return
        self._send_json({"error": error or "設備更新服務回應異常"}, status=502)

    # ── Debug 推論功能 ─────────────────────────────────

    def _handle_debug_serve_image(self, query):
        """API: 以絕對路徑提供原始圖片 (僅 Debug 用)
        瀏覽器不支援 TIF/TIFF/BMP，自動轉為 PNG 回傳。
        """
        try:
            # query 已由 do_GET 透過 parse_qs 解析為 dict
            params = query if isinstance(query, dict) else urllib.parse.parse_qs(query)
            img_path = params.get("path", [None])[0]
            serve_raw = str(params.get("raw", ["0"])[0]).strip().lower() in ("1", "true", "yes")
            if not img_path:
                self._send_error(400, "missing path parameter")
                return
            p = Path(img_path)
            if not p.exists() or not p.is_file():
                self._send_error(404, f"file not found: {img_path}")
                return

            suffix = p.suffix.lower()
            # 瀏覽器原生可顯示的格式直接回傳
            if (
                (serve_raw or not self._inference_rotate_180_enabled())
                and suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")
            ):
                self._send_binary(str(p))
                return

            # 其餘格式 (tif, tiff, bmp …) 用 cv2 轉 PNG
            import cv2
            img = (
                cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if serve_raw
                else self._read_inference_image(p, cv2.IMREAD_UNCHANGED)
            )
            if img is None:
                self._send_error(400, f"cv2 cannot read: {img_path}")
                return
            self._send_image_array_png(img)
        except Exception as e:
            logger.error(f"Error serving debug image: {e}")
            self._send_error(500, str(e))

    def _handle_debug_preprocess_lab(self):
        """API: Debug image preprocessing lab."""
        import time as _time
        import uuid

        import cv2

        from capi_image_preprocess_lab import apply_preprocess_method, make_diff_image

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode('utf-8'))
        except Exception:
            self._send_json({"error": "Invalid JSON body"})
            return

        image_path_str = data.get("image_path", "").strip()
        if not image_path_str:
            self._send_json({"error": "請提供圖片路徑 (image_path)"})
            return

        image_path = Path(image_path_str)
        if not image_path.exists():
            self._send_json({"error": f"檔案不存在: {image_path}"})
            return
        if not image_path.is_file():
            self._send_json({"error": f"不是檔案: {image_path}"})
            return

        method = str(data.get("method", "median")).strip()
        params = data.get("params") or {}
        if not isinstance(params, dict):
            self._send_json({"error": "params 必須是物件"})
            return

        try:
            start = _time.time()
            image = self._read_inference_image(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                self._send_json({"error": f"無法讀取圖片: {image_path}"})
                return

            result = apply_preprocess_method(image, method, params)
            processed = result["image"]
            diff = make_diff_image(image, processed)

            if CAPIWebHandler._debug_heatmap_dir is None:
                CAPIWebHandler._debug_heatmap_dir = Path(tempfile.mkdtemp(prefix="capi_debug_hm_"))
            debug_dir = CAPIWebHandler._debug_heatmap_dir
            debug_dir.mkdir(parents=True, exist_ok=True)

            safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem)[:80] or "image"
            token = uuid.uuid4().hex[:8]
            safe_method = result["method"]
            processed_filename = f"preprocess_lab_{safe_stem}_{safe_method}_{token}.png"
            diff_filename = f"preprocess_lab_{safe_stem}_{safe_method}_{token}_diff.png"
            processed_path = debug_dir / processed_filename
            diff_path = debug_dir / diff_filename

            if not cv2.imwrite(str(processed_path), processed):
                self._send_json({"error": "處理後圖片寫入失敗"})
                return
            if not cv2.imwrite(str(diff_path), diff):
                self._send_json({"error": "差異圖寫入失敗"})
                return

            elapsed = _time.time() - start
            h, w = image.shape[:2]
            channels = 1 if image.ndim == 2 else image.shape[2]

            self._send_json({
                "success": True,
                "image_path": str(image_path),
                "image_name": image_path.name,
                "image_size": [w, h],
                "channels": channels,
                "input_dtype": str(image.dtype),
                "method": result["method"],
                "method_label": result["method_label"],
                "applied_params": result["applied_params"],
                "notes": result["notes"],
                "conversion": result["conversion"],
                "stats": result["stats"],
                "processing_time": round(elapsed, 3),
                "original_url": "/api/debug/serve-image?path=" + urllib.parse.quote(str(image_path)),
                "processed_url": f"/debug/heatmaps/{processed_filename}",
                "diff_url": f"/debug/heatmaps/{diff_filename}",
                "output_path": str(processed_path),
                "diff_path": str(diff_path),
            })
        except Exception as e:
            logger.error(f"[DEBUG] Preprocess lab error: {e}", exc_info=True)
            self._send_json({"error": f"影像前處理失敗: {str(e)}"})

    def _handle_debug_page(self, path: str):
        """Debug 推論頁面"""
        from capi_image_preprocess_lab import get_method_specs

        # 從 DB 讀取最新設定，若無則 fallback 到推論器的 config
        db_params = {}
        if self.db:
            try:
                for p in self.db.get_all_config_params():
                    db_params[p["param_name"]] = p["decoded_value"]
            except Exception as e:
                logger.error(f"無法讀取 DB 設定: {e}")

        def get_val(name, default_val):
            if name in db_params:
                return db_params[name]
            if self.inferencer and hasattr(self.inferencer.config, name):
                return getattr(self.inferencer.config, name)
            return default_val

        from capi_config import CAPIConfig

        try:
            within_spec_rules = CAPIConfig._normalize_within_spec_judgment_rules(
                get_val(WITHIN_SPEC_PARAM, CAPIConfig().within_spec_judgment_rules)
            )
            dot_detection_defaults = (
                within_spec_rules.get("default", {}).get("dot_detection", {})
                if isinstance(within_spec_rules, dict)
                else {}
            )
        except Exception:
            within_spec_rules = CAPIConfig().within_spec_judgment_rules
            dot_detection_defaults = within_spec_rules["default"]["dot_detection"]

        dot_machine_configs = {}
        if isinstance(within_spec_rules, dict):
            for machine_key, machine_cfg in within_spec_rules.items():
                if not isinstance(machine_cfg, dict):
                    continue
                dot_cfg = machine_cfg.get("dot_detection") or {}
                if not isinstance(dot_cfg, dict):
                    continue
                dot_machine_configs[str(machine_key)] = {
                    "display_name": str(machine_cfg.get("display_name") or machine_key),
                    "dot_detection": _json_safe_snapshot(dot_cfg),
                }
        
        dot_sample_dir = Path(__file__).resolve().parent / "templates" / "imgs"
        template = self.jinja_env.get_template("debug_inference.html")
        html = template.render(
            request_path=path,
            default_threshold=get_val('anomaly_threshold', 0.5),
            default_edge_margin=get_val('edge_margin_px', 0),
            default_dust_extension=get_val('dust_extension', 0),
            default_dust_metric=get_val('dust_heatmap_metric', 'iou'),
            default_dust_iou_thr=get_val('dust_heatmap_iou_threshold', 0.01),
            default_dust_top_pct=get_val('dust_heatmap_top_percent', 5.0),
            model_resolution_map=get_val('model_resolution_map', {}),
            default_patchcore_filter_enabled=get_val('patchcore_filter_enabled', False),
            default_patchcore_blur_sigma=get_val('patchcore_blur_sigma', 1.5),
            default_patchcore_min_area=get_val('patchcore_min_area', 10),
            default_patchcore_score_metric=get_val('patchcore_score_metric', 'max'),
            default_otsu_offset=get_val('otsu_offset', 5),
            default_bs_diff_threshold=get_val('bright_spot_diff_threshold', 10),
            default_bs_median_kernel=get_val('bright_spot_median_kernel', 21),
            default_bs_min_area=get_val('bright_spot_min_area', 5),
            default_bs_threshold=get_val('bright_spot_threshold', 200),
            default_aoi_threshold=get_val('cv_edge_aoi_threshold', 4),
            default_aoi_min_area=get_val('cv_edge_aoi_min_area', 40),
            default_aoi_solidity_min=get_val('cv_edge_aoi_solidity_min', 0.2),
            default_aoi_polygon_erode_px=get_val('cv_edge_aoi_polygon_erode_px', 3),
            default_aoi_morph_open_kernel=get_val('cv_edge_aoi_morph_open_kernel', 3),
            default_aoi_min_max_diff=get_val('cv_edge_aoi_min_max_diff', 20),
            default_aoi_line_min_length=get_val('cv_edge_aoi_line_min_length', 30),
            default_aoi_line_max_width=get_val('cv_edge_aoi_line_max_width', 3),
            default_aoi_boundary_padding=15,
            default_aoi_boundary_min_bright=15,
            default_aoi_edge_inspector=get_val('aoi_edge_inspector', 'cv'),
            preprocess_methods=get_method_specs(),
            dot_samples=_list_debug_dot_samples(dot_sample_dir),
            dot_detection_defaults=dot_detection_defaults,
            dot_default_mm_per_px=DOT_RULER_MM_PER_PX,
            dot_calibration_points=DOT_RULER_CALIBRATION_POINTS,
            dot_calibration_source=DOT_RULER_CALIBRATION_SOURCE,
            dot_machine_configs=dot_machine_configs,
        )
        self._send_response(200, html)

    # ── RIC 人工檢驗報表功能 ─────────────────────────

    def _handle_ric_page(self, query: dict, path: str):
        """人工檢驗 (RIC) 比對報表頁面"""
        batches = self.db.get_ric_batches() if self.db else []
        template = self.jinja_env.get_template("ric_report.html")
        html = template.render(request_path=path, batches=batches)
        self._send_response(200, html)

    def _handle_within_spec_report_page(self, query: dict, path: str):
        """規格內建議計算紀錄清單。"""
        try:
            start_date = query.get("start_date", [""])[0] or None
            end_date = query.get("end_date", [""])[0] or None
            keyword = query.get("keyword", [""])[0] or ""
            status = query.get("status", ["all"])[0] or "all"
            limit = _as_int(query.get("limit", ["200"])[0], 200)
            suggested = None
            if status == "suggested":
                suggested = True
            elif status == "not_suggested":
                suggested = False

            rows = self.db.list_within_spec_review_log_report(
                start_date=start_date,
                end_date=end_date,
                keyword=keyword,
                suggested=suggested,
                limit=limit,
            ) if self.db else []
            summary = {
                "total": len(rows),
                "suggested": sum(1 for row in rows if row.get("suggested")),
                "not_suggested": sum(1 for row in rows if not row.get("suggested")),
                "fallback": sum(1 for row in rows if row.get("fallback_used")),
            }
            template = self.jinja_env.get_template("within_spec_report.html")
            html = template.render(
                request_path=path,
                rows=rows,
                summary=summary,
                filters={
                    "start_date": start_date or "",
                    "end_date": end_date or "",
                    "keyword": keyword,
                    "status": status,
                    "limit": max(1, min(limit, 500)),
                },
            )
            self._send_response(200, html)
        except ValueError as ve:
            self._send_response(400, str(ve))
        except Exception as e:
            logger.error("Within-spec report page error: %s", e, exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_ric_upload(self):
        """上傳 XLS 檔案並匯入 RIC 資料"""
        import cgi
        import io

        content_type = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in content_type:
            self._send_json({"error": "請使用 multipart/form-data 上傳檔案"})
            return

        # 解析 multipart form data
        try:
            boundary = content_type.split('boundary=')[1]
        except IndexError:
            self._send_json({"error": "無法解析 boundary"})
            return

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        # 用 cgi 解析 multipart
        environ = {
            'REQUEST_METHOD': 'POST',
            'CONTENT_TYPE': content_type,
            'CONTENT_LENGTH': str(content_length),
        }
        fs = cgi.FieldStorage(
            fp=io.BytesIO(body),
            environ=environ,
            keep_blank_values=True
        )

        if 'file' not in fs:
            self._send_json({"error": "請提供檔案 (field name: file)"})
            return

        file_item = fs['file']
        filename = file_item.filename or 'unknown.xlsx'
        file_data = file_item.file.read()

        if not file_data:
            self._send_json({"error": "檔案為空"})
            return

        try:
            # 先偵測是否為 HTML 格式 (很多舊系統匯出的 .xls 其實是 HTML)
            is_html = file_data[:100].strip().startswith(b'<')

            if is_html:
                # HTML Table 格式 → 用 html.parser
                from html.parser import HTMLParser
                import html as html_module

                class TableParser(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.in_th = False
                        self.in_td = False
                        self.headers = []
                        self.rows = []
                        self.current_row = []
                        self.current_data = ''

                    def handle_starttag(self, tag, attrs):
                        if tag == 'th':
                            self.in_th = True
                            self.current_data = ''
                        elif tag == 'td':
                            self.in_td = True
                            self.current_data = ''
                        elif tag == 'tr':
                            self.current_row = []

                    def handle_endtag(self, tag):
                        if tag == 'th':
                            self.in_th = False
                            self.headers.append(self.current_data.strip())
                        elif tag == 'td':
                            self.in_td = False
                            val = self.current_data.strip()
                            if val == '\xa0' or val == '&nbsp;':
                                val = ''
                            self.current_row.append(val)
                        elif tag == 'tr':
                            if self.current_row:
                                self.rows.append(self.current_row)

                    def handle_data(self, data):
                        if self.in_th or self.in_td:
                            self.current_data += data

                    def handle_entityref(self, name):
                        if self.in_th or self.in_td:
                            ch = html_module.unescape(f'&{name};')
                            self.current_data += ch

                # 嘗試多種編碼
                text = ''
                for enc in ['utf-8', 'gb2312', 'gbk', 'big5', 'latin-1']:
                    try:
                        text = file_data.decode(enc)
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue

                parser = TableParser()
                parser.feed(text)

                headers = parser.headers
                records_data = []
                for row in parser.rows:
                    if not any(row):
                        continue
                    rec = {}
                    for i, header in enumerate(headers):
                        if i < len(row):
                            rec[header] = row[i]
                        else:
                            rec[header] = ''
                    if rec.get('PNL_ID'):
                        records_data.append(rec)

            elif filename.lower().endswith('.xls') and not filename.lower().endswith('.xlsx'):
                # .xls 格式 (Excel 97-2003) → 用 xlrd
                import xlrd
                wb = xlrd.open_workbook(file_contents=file_data)
                ws = wb.sheet_by_index(0)

                if ws.nrows == 0:
                    self._send_json({"error": "檔案無資料"})
                    return

                headers = [str(ws.cell_value(0, c)).strip() for c in range(ws.ncols)]

                records_data = []
                for r in range(1, ws.nrows):
                    row_vals = [ws.cell_value(r, c) for c in range(ws.ncols)]
                    if not any(row_vals):
                        continue
                    rec = {}
                    for i, header in enumerate(headers):
                        if i < len(row_vals):
                            val = row_vals[i]
                            rec[header] = str(val) if val is not None else ''
                        else:
                            rec[header] = ''
                    if rec.get('PNL_ID'):
                        records_data.append(rec)

            else:
                self._send_json({"error": "不支援的檔案格式，請上傳 .xls 檔案"})
                return

            if not records_data:
                self._send_json({"error": "檔案中無有效 RIC 記錄 (缺少 PNL_ID)"})
                return

            batch_id = self.db.save_ric_batch(filename, records_data)

            self._send_json({
                "success": True,
                "batch_id": batch_id,
                "filename": filename,
                "total_records": len(records_data),
                "message": f"成功匯入 {len(records_data)} 筆 RIC 記錄"
            })

        except ImportError as ie:
            self._send_json({"error": f"缺少套件，請執行: pip install xlrd ({ie})"})
        except Exception as e:
            logger.error(f"RIC upload error: {e}", exc_info=True)
            self._send_json({"error": f"解析檔案失敗: {str(e)}"})

    def _handle_ric_delete(self):
        """API: 刪除 RIC 匯入批次"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            batch_id = data.get('batch_id')

            if not batch_id:
                self._send_json({"error": "缺少 batch_id 參數"})
                return

            batch_id = int(batch_id)
            deleted_count = self.db.delete_ric_batch(batch_id) if self.db else 0

            self._send_json({
                "success": True,
                "batch_id": batch_id,
                "deleted_records": deleted_count,
                "message": f"已刪除批次 #{batch_id}，共 {deleted_count} 筆記錄"
            })
        except (ValueError, TypeError) as e:
            self._send_json({"error": f"參數錯誤: {e}"})
        except Exception as e:
            logger.error(f"RIC delete error: {e}", exc_info=True)
            self._send_json({"error": f"刪除失敗: {str(e)}"})

    def _handle_client_import(self):
        """API: 匯入 client accuracy records 至資料庫"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            records = data.get("records", [])
            if not records:
                self._send_json({"error": "無資料可匯入"})
                return

            result = self.db.save_client_accuracy_records(records)
            self._send_json({
                "success": True,
                "inserted": result["inserted"],
                "skipped": result["skipped"],
                "message": f"新增 {result['inserted']} 筆，略過 {result['skipped']} 筆重複資料"
            })
        except Exception as e:
            logger.error(f"Client import error: {e}", exc_info=True)
            self._send_json({"error": f"匯入失敗: {str(e)}"})

    def _handle_client_data_api(self, query: dict):
        """API: 取得已儲存的 client accuracy records（支援日期篩選 + summary 統計）"""
        try:
            start_date = query.get('start_date', [''])[0] or None
            end_date = query.get('end_date', [''])[0] or None
            records = self.db.get_client_accuracy_records(start_date, end_date)
            inf_ids = list({r["inference_record_id"] for r in records if r.get("inference_record_id")})
            dust_ids = self.db.get_dust_affected_record_ids(inf_ids) if inf_ids else set()
            scratch_stats = self.db.get_scratch_rescue_stats(inf_ids) if inf_ids else {}
            summary, out_records = self._compute_client_summary(records, dust_ids, scratch_stats)

            self._send_json({
                "success": True,
                "total": summary["total"],
                "summary": summary,
                "records": out_records,
            })
        except ValueError as ve:
            self._send_json({"success": False, "error": str(ve)})
        except Exception as e:
            logger.error(f"Client data API error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _load_within_spec_rules_for_review(self) -> Dict[str, Any]:
        rules = None
        if self.db:
            try:
                param = self.db.get_config_param(WITHIN_SPEC_PARAM)
                if param:
                    rules = param.get("decoded_value")
            except Exception as e:
                logger.warning("Failed to load within-spec rules from DB: %s", e)
        if rules is None and self.inferencer and getattr(self.inferencer, "config", None):
            rules = getattr(self.inferencer.config, "within_spec_judgment_rules", None)
        if rules is None:
            from capi_config import CAPIConfig
            rules = CAPIConfig().within_spec_judgment_rules
        from capi_config import CAPIConfig
        return CAPIConfig._normalize_within_spec_judgment_rules(rules)

    def _within_spec_visual_output(self, detail: Dict[str, Any], client_record_id: int, inference_record_id: int) -> Tuple[Optional[Path], str]:
        if not self.heatmap_base_dir:
            return None, ""
        base_dir = Path(self.heatmap_base_dir)
        heatmap_dir = Path(str(detail.get("heatmap_dir") or ""))
        if heatmap_dir and hm_relative(str(heatmap_dir), base_dir):
            root = heatmap_dir / "within_spec_review"
        else:
            root = base_dir / "within_spec_review" / f"record_{inference_record_id}"
        run_dir = root / f"client_{client_record_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        rel = hm_relative(str(run_dir), base_dir)
        if not rel:
            return None, ""
        return run_dir, f"/heatmaps/{rel}"

    def _handle_within_spec_log_page(self, path: str):
        try:
            log_id = _as_int(path.rstrip("/").split("/")[-1], 0)
            if log_id <= 0:
                self._send_404(path)
                return
            log = self.db.get_within_spec_review_log(log_id) if self.db else None
            if not log:
                self._send_404(path)
                return
            template = self.jinja_env.get_template("within_spec_detail.html")
            html = template.render(log=log, request_path=path)
            self._send_response(200, html)
        except Exception as e:
            logger.error("Within-spec log page error: %s", e, exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_within_spec_log_regenerate(self):
        """API: regenerate a saved within-spec log with persisted dot-detection visuals."""
        import time as _time

        started = _time.time()
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8')) if body else {}

            log_id = _as_int(data.get("log_id"), 0)
            if log_id <= 0:
                self._send_json({"success": False, "error": "缺少 log_id"})
                return
            log = self.db.get_within_spec_review_log(log_id) if self.db else None
            if not log:
                self._send_json({"success": False, "error": f"找不到規格內明細: {log_id}"})
                return

            inference_record_id = _as_int(log.get("inference_record_id"), 0)
            if inference_record_id <= 0:
                self._send_json({"success": False, "error": "這筆 log 沒有對應推論紀錄"})
                return
            detail = self.db.get_record_detail(inference_record_id) if self.db else None
            if not detail:
                self._send_json({"success": False, "error": f"找不到推論紀錄: {inference_record_id}"})
                return

            client_record_id = _as_int(log.get("client_record_id"), 0)
            if client_record_id > 0:
                visual_dir, visual_prefix = self._within_spec_visual_output(detail, client_record_id, inference_record_id)
            else:
                visual_dir, visual_prefix = _within_spec_auto_visual_output(
                    str(self.heatmap_base_dir or ""),
                    detail.get("glass_id") or "",
                    inference_record_id,
                )

            eval_result = _evaluate_within_spec_suggestion_detail(
                detail,
                self._load_within_spec_rules_for_review(),
                machine_id=str(detail.get("model_id") or detail.get("machine_no") or ""),
                visual_output_dir=visual_dir,
                visual_url_prefix=visual_prefix,
                rotate_180=self._inference_rotate_180_enabled(),
            )

            source = "inference" if log.get("source") == "inference" else "review"
            suggestion = eval_result.get("suggestion")
            error_message = ""
            if source == "inference":
                panel_totals = eval_result.get("panel_totals") or []
                panel_within = bool(panel_totals) and all(bool(item.get("within")) for item in panel_totals)
                converted = bool(suggestion and suggestion.get("suggested") and panel_within)
                panel_reason = _format_within_spec_panel_summary(eval_result)
                if converted:
                    status = "within_spec"
                    reason = panel_reason or suggestion.get("reason", "")
                    if suggestion and reason:
                        suggestion = dict(suggestion)
                        suggestion["reason"] = reason
                elif suggestion and suggestion.get("suggested"):
                    status = "not_within_spec"
                    reason = panel_reason or "部分項目符合規格內，但整片 PANEL 尚有項目未符合"
                    suggestion = None
                    error_message = reason
                elif panel_totals:
                    status = "not_within_spec"
                    reason = panel_reason or "未符合規格內條件"
                    error_message = reason
                else:
                    status = "not_evaluable"
                    reason = "未取得可比對的規格內點數結果"
                    error_message = reason
                eval_result["source"] = "inference"
                eval_result["inference_context"] = {
                    "glass_id": detail.get("glass_id", ""),
                    "model_id": detail.get("model_id", ""),
                    "machine_no": detail.get("machine_no", ""),
                    "machine_judgment": detail.get("machine_judgment", ""),
                }
                eval_result["inference_auto_decision"] = {
                    "converted_to_ok_i": converted,
                    "status": status,
                    "reason": reason,
                    "requires_all_panel_totals_within": True,
                }

            saved = self.db.save_within_spec_review_log(
                client_record_id=client_record_id if client_record_id > 0 else None,
                inference_record_id=inference_record_id,
                suggestion=suggestion,
                detail=eval_result,
                processing_seconds=_time.time() - started,
                error_message=error_message,
                source=source,
            )
            self._send_json({
                "success": True,
                "log": saved,
                "redirect_url": f"/ric/within-spec-log/{saved.get('id')}",
                "visuals_saved": len(eval_result.get("visuals") or []),
            })
        except ValueError as ve:
            self._send_json({"success": False, "error": str(ve)})
        except Exception as e:
            logger.error("Within-spec log regenerate error: %s", e, exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_within_spec_suggestion_api(self, query: dict):
        """API: preview one non-mutating within-spec suggestion without saving a log."""
        import time as _time

        started = _time.time()
        try:
            record_id = _as_int((query.get("inference_record_id") or [""])[0], 0)
            if record_id <= 0:
                self._send_json({"success": False, "error": "missing inference_record_id"})
                return

            detail = self.db.get_record_detail(record_id) if self.db else None
            eval_result = {"suggestion": None, "steps": []}
            if detail:
                eval_result = _evaluate_within_spec_suggestion_detail(
                    detail,
                    self._load_within_spec_rules_for_review(),
                    machine_id=(query.get("mach_id") or [""])[0] or "",
                    rotate_180=self._inference_rotate_180_enabled(),
                )

            elapsed = _time.time() - started
            if elapsed > 5:
                logger.warning("Within-spec suggestion %s took %.2fs", record_id, elapsed)
            self._send_json({
                "success": True,
                "suggestion": eval_result.get("suggestion"),
                "detail": eval_result,
                "processing_time": round(elapsed, 3),
            })
        except Exception as e:
            logger.error("Within-spec suggestion API error: %s", e, exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_within_spec_suggestion_run(self):
        """API: manually calculate and save one within-spec suggestion log."""
        import time as _time

        started = _time.time()
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8')) if body else {}

            client_record_id = _as_int(data.get("client_record_id"), 0)
            inference_record_id = _as_int(data.get("inference_record_id"), 0)
            mach_id = str(data.get("mach_id") or "")
            if client_record_id <= 0:
                self._send_json({"success": False, "error": "缺少 client_record_id"})
                return
            if inference_record_id <= 0:
                self._send_json({"success": False, "error": "缺少 inference_record_id"})
                return

            detail = self.db.get_record_detail(inference_record_id) if self.db else None
            if not detail:
                self._send_json({"success": False, "error": f"找不到推論紀錄: {inference_record_id}"})
                return

            visual_dir, visual_prefix = self._within_spec_visual_output(detail, client_record_id, inference_record_id)
            eval_result = _evaluate_within_spec_suggestion_detail(
                detail,
                self._load_within_spec_rules_for_review(),
                machine_id=mach_id,
                visual_output_dir=visual_dir,
                visual_url_prefix=visual_prefix,
                rotate_180=self._inference_rotate_180_enabled(),
            )
            elapsed = _time.time() - started
            log = self.db.save_within_spec_review_log(
                client_record_id=client_record_id,
                inference_record_id=inference_record_id,
                suggestion=eval_result.get("suggestion"),
                detail=eval_result,
                processing_seconds=elapsed,
                error_message="",
            )
            if elapsed > 5:
                logger.warning("Within-spec suggestion run client=%s inference=%s took %.2fs", client_record_id, inference_record_id, elapsed)
            self._send_json({
                "success": True,
                "suggestion": eval_result.get("suggestion"),
                "log": log,
                "processing_time": round(elapsed, 3),
            })
        except ValueError as ve:
            self._send_json({"success": False, "error": str(ve)})
        except Exception as e:
            logger.error("Within-spec suggestion run error: %s", e, exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_within_spec_suggestion_log_api(self, query: dict):
        """API: fetch saved within-spec suggestion logs for one client record."""
        try:
            client_record_id = _as_int((query.get("client_record_id") or [""])[0], 0)
            if client_record_id <= 0:
                self._send_json({"success": False, "error": "missing client_record_id"})
                return
            limit = _as_int((query.get("limit") or ["10"])[0], 10)
            logs = self.db.get_within_spec_review_logs(client_record_id, limit=limit) if self.db else []
            self._send_json({
                "success": True,
                "log": logs[0] if logs else None,
                "logs": logs,
            })
        except Exception as e:
            logger.error("Within-spec suggestion log API error: %s", e, exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _compute_client_summary(self, records: list, dust_affected_ids: set = None, scratch_stats: dict = None):
        """從 client accuracy records 計算統計摘要並格式化 records，單次遍歷。
        Returns: (summary_dict, out_records_list)
        """
        from capi_database import CAPIDatabase

        _empty_miss_cats = lambda: {c: 0 for c in CAPIDatabase.VALID_MISS_CATEGORIES}
        _empty_over_cats = lambda: {c: 0 for c in CAPIDatabase.VALID_OVER_CATEGORIES}
        def _production_day_key(value: str) -> str:
            text = str(value or "").strip()
            try:
                dt = datetime.fromisoformat(text.replace("T", " "))
                if dt.hour < 7 or (dt.hour == 7 and dt.minute < 30):
                    dt -= timedelta(days=1)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                return (text or "unknown")[:10]

        actual_ok_review_categories = {"ric_misjudge", "data_error_actually_ok", "within_spec_misjudge"}
        counted_ai_miss_categories = {"threshold_high", "dust_misfilter"}
        total = len(records)
        if total == 0:
            return {
                "total": 0, "aoiOK": 0, "aoiNG": 0, "aiNG": 0, "ricNG": 0,
                "aoiCorrect": 0, "aiCorrect": 0,
                "aoiOver": 0, "aoiOverRate": 0,
                "aiOver": 0, "aiOverRate": 0,
                "aiMiss": 0, "aiMissRate": 0,
                "revival": 0, "revivalRate": 0,
                "revivalWithinSpec": 0, "revivalWithinSpecRate": 0,
                "combos": {}, "daily": {},
                "missReviewStats": {
                    "total": 0, "reviewed": 0, "unreviewed": 0,
                    "byCategory": _empty_miss_cats(),
                },
                "manualTruthAdjustments": {
                    "total": 0,
                    "byCategory": {c: 0 for c in actual_ok_review_categories},
                },
                "overReviewStats": {
                    "total": 0, "reviewed": 0, "unreviewed": 0,
                    "poolRecords": 0,
                    "poolTiles": 0,
                    "byCategory": _empty_over_cats(),
                },
                "scratchRescueStats": {
                    "panels": 0, "images": 0, "tiles": 0,
                },
            }, []

        aoiNG = aiNG = ricNG = 0
        aoiCorrect = aiCorrect = 0
        aoiOver = aiOver = aiMiss = 0
        revival = 0
        revival_within_spec = 0
        combos = {}
        daily = {}
        manual_adjustments = 0
        manual_adjustments_by_cat = {c: 0 for c in actual_ok_review_categories}
        miss_reviewed = 0
        miss_total = 0
        miss_by_cat = _empty_miss_cats()
        over_reviewed = 0
        over_pool_records = 0
        over_pool_tiles = 0
        over_by_cat = _empty_over_cats()
        scratch_stats = scratch_stats or {}
        scratch_panels = 0
        scratch_images_total = 0
        scratch_tiles_total = 0
        out_records = []

        for rec in records:
            eqp = rec["result_eqp"] or "OK"
            raw_ai = rec["result_ai"] or "OK"
            inference_ai = rec.get("inference_ai_judgment") or ""
            ai = "OK" if raw_ai == "OK-i" else raw_ai
            within_spec_converted_ok = raw_ai == "OK-i" or inference_ai == "OK-i"
            raw_ric = CAPIDatabase.parse_ric_judgment(rec.get("datastr", ""))
            review_cat = rec.get("review_category")
            manual_actual_ok = (
                ai == "OK"
                and raw_ric == "NG"
                and review_cat in actual_ok_review_categories
            )
            counted_ai_miss = (
                ai == "OK"
                and raw_ric == "NG"
                and review_cat in counted_ai_miss_categories
            )
            ric = "OK" if manual_actual_ok else raw_ric

            # Build formatted output record in the same pass
            out_rec = {
                "id": rec["id"],
                "time_stamp": rec["time_stamp"],
                "pnl_id": rec["pnl_id"],
                "mach_id": rec["mach_id"],
                "result_eqp": eqp,
                "result_ai": raw_ai,
                "inference_ai_judgment": inference_ai,
                "within_spec_converted_to_ok": within_spec_converted_ok,
                "result_ric": rec["result_ric"],
                "datastr": rec["datastr"] or "",
                "actual_judgment": ric,
                "truth_adjusted_by_review": manual_actual_ok,
                "inference_record_id": rec.get("inference_record_id"),
                "has_dust_filtering": bool(
                    dust_affected_ids
                    and rec.get("inference_record_id")
                    and rec["inference_record_id"] in dust_affected_ids
                ),
                "miss_review": None,
                "over_review": None,
                "over_retrain_pool": {
                    "count": int(rec.get("over_retrain_pool_count") or 0),
                    "latest_at": rec.get("over_retrain_pool_latest_at"),
                },
                "scratch_rescue": None,
                "within_spec_log": None,
            }
            rid = rec.get("inference_record_id")
            sr = scratch_stats.get(rid) if rid else None
            if sr:
                out_rec["scratch_rescue"] = {"tiles": sr["tiles"], "images": sr["images"]}
                scratch_panels += 1
                scratch_images_total += sr["images"]
                scratch_tiles_total += sr["tiles"]
            if rec.get("review_id"):
                out_rec["miss_review"] = {
                    "id": rec["review_id"],
                    "category": rec["review_category"],
                    "note": rec["review_note"] or "",
                    "updated_at": rec["review_updated_at"],
                }
            if rec.get("over_review_id"):
                out_rec["over_review"] = {
                    "id": rec["over_review_id"],
                    "category": rec["over_review_category"],
                    "note": rec["over_review_note"] or "",
                    "updated_at": rec["over_review_updated_at"],
                }
            if rec.get("within_spec_log_id"):
                suggestion = None
                if rec.get("within_spec_suggested"):
                    suggestion = {
                        "suggested": True,
                        "category": rec.get("within_spec_category") or "within_spec",
                        "reason": rec.get("within_spec_reason") or "",
                    }
                    out_rec["within_spec_suggestion"] = suggestion
                out_rec["within_spec_log"] = {
                    "id": rec["within_spec_log_id"],
                    "suggested": bool(rec.get("within_spec_suggested")),
                    "category": rec.get("within_spec_category") or "",
                    "reason": rec.get("within_spec_reason") or "",
                    "error_message": rec.get("within_spec_error") or "",
                    "processing_seconds": rec.get("within_spec_processing_seconds") or 0,
                    "created_at": rec.get("within_spec_created_at"),
                    "suggestion": suggestion,
                }
            out_records.append(out_rec)

            if eqp == "NG":
                aoiNG += 1
            if ai == "NG":
                aiNG += 1
            if ric == "NG":
                ricNG += 1
            if manual_actual_ok:
                manual_adjustments += 1
                manual_adjustments_by_cat[review_cat] += 1

            if eqp == ric:
                aoiCorrect += 1
            if ai == ric:
                aiCorrect += 1

            if eqp == "NG" and ric == "OK":
                aoiOver += 1
            if ai == "NG" and ric == "OK":
                aiOver += 1
                pool_count = int(rec.get("over_retrain_pool_count") or 0)
                if pool_count > 0:
                    over_pool_records += 1
                    over_pool_tiles += pool_count
                if rec.get("over_review_category"):
                    over_reviewed += 1
                    cat = rec["over_review_category"]
                    if cat in over_by_cat:
                        over_by_cat[cat] += 1
            if ai == "OK" and raw_ric == "NG":
                miss_total += 1
                if rec.get("review_category"):
                    miss_reviewed += 1
                    if review_cat in miss_by_cat:
                        miss_by_cat[review_cat] += 1
            if counted_ai_miss:
                aiMiss += 1
            if eqp == "NG" and ai == "OK" and ric == "OK":
                revival += 1
                if within_spec_converted_ok:
                    revival_within_spec += 1

            combo = f"{eqp}/{ai}/{ric}"
            combos[combo] = combos.get(combo, 0) + 1

            day = _production_day_key(rec.get("time_stamp"))
            if day not in daily:
                daily[day] = {
                    "total": 0, "aoiCorrect": 0, "aiCorrect": 0,
                    "aiMiss": 0, "aiOver": 0, "aoiOver": 0,
                    "ricMisjudge": 0, "withinSpecMisjudge": 0,
                }
            daily[day]["total"] += 1
            if eqp == ric:
                daily[day]["aoiCorrect"] += 1
            if ai == ric:
                daily[day]["aiCorrect"] += 1
            if counted_ai_miss:
                daily[day]["aiMiss"] += 1
            if ai == "NG" and ric == "OK":
                daily[day]["aiOver"] += 1
            if eqp == "NG" and ric == "OK":
                daily[day]["aoiOver"] += 1
            if manual_actual_ok:
                daily[day]["ricMisjudge"] += 1
                if review_cat == "within_spec_misjudge":
                    daily[day]["withinSpecMisjudge"] += 1

        aoiOK = total - aoiNG
        aoiOverRate = round(aoiOver / total * 100, 2) if total > 0 else 0
        aiOverRate = round(aiOver / total * 100, 2) if total > 0 else 0
        aiMissRate = round(aiMiss / total * 100, 2) if total > 0 else 0
        revivalRate = round(revival / aoiOver * 100, 2) if aoiOver > 0 else 0
        revivalWithinSpecRate = round(revival_within_spec / revival * 100, 2) if revival > 0 else 0

        return {
            "total": total,
            "aoiOK": aoiOK, "aoiNG": aoiNG, "aiNG": aiNG, "ricNG": ricNG,
            "aoiCorrect": aoiCorrect, "aiCorrect": aiCorrect,
            "aoiOver": aoiOver, "aoiOverRate": aoiOverRate,
            "aiOver": aiOver, "aiOverRate": aiOverRate,
            "aiMiss": aiMiss, "aiMissRate": aiMissRate,
            "revival": revival, "revivalRate": revivalRate,
            "revivalWithinSpec": revival_within_spec,
            "revivalWithinSpecRate": revivalWithinSpecRate,
            "combos": combos, "daily": daily,
            "missReviewStats": {
                "total": miss_total,
                "reviewed": miss_reviewed,
                "unreviewed": miss_total - miss_reviewed,
                "byCategory": miss_by_cat,
            },
            "manualTruthAdjustments": {
                "total": manual_adjustments,
                "byCategory": manual_adjustments_by_cat,
            },
            "overReviewStats": {
                "total": aiOver,
                "reviewed": over_reviewed,
                "unreviewed": aiOver - over_reviewed,
                "poolRecords": over_pool_records,
                "poolTiles": over_pool_tiles,
                "byCategory": over_by_cat,
            },
            "scratchRescueStats": {
                "panels": scratch_panels,
                "images": scratch_images_total,
                "tiles": scratch_tiles_total,
            },
        }, out_records

    def _handle_client_clear(self):
        """API: 清除所有 client accuracy records"""
        try:
            count = self.db.clear_client_accuracy_records()
            self._send_json({
                "success": True,
                "deleted": count,
                "message": f"已清除 {count} 筆資料"
            })
        except Exception as e:
            logger.error(f"Client clear error: {e}", exc_info=True)
            self._send_json({"error": str(e)})

    def _handle_miss_review_save(self):
        """API: 儲存/更新漏檢 Review"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            client_record_id = data.get("client_record_id")
            category = data.get("category", "")
            note = data.get("note", "")

            if not client_record_id:
                self._send_json({"success": False, "error": "缺少 client_record_id"})
                return

            review_id = self.db.save_miss_review(int(client_record_id), category, note)
            self._send_json({"success": True, "id": review_id, "message": "Review 已儲存"})
        except ValueError as ve:
            self._send_json({"success": False, "error": str(ve)})
        except Exception as e:
            logger.error(f"Miss review save error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_miss_review_delete(self):
        """API: 刪除漏檢 Review"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            client_record_id = data.get("client_record_id")
            if not client_record_id:
                self._send_json({"success": False, "error": "缺少 client_record_id"})
                return

            deleted = self.db.delete_miss_review(int(client_record_id))
            if deleted:
                self._send_json({"success": True, "message": "Review 已刪除"})
            else:
                self._send_json({"success": False, "error": "Review 不存在"})
        except Exception as e:
            logger.error(f"Miss review delete error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_over_review_save(self):
        """API: 儲存/更新過檢 Review"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            client_record_id = data.get("client_record_id")
            category = data.get("category", "")
            note = data.get("note", "")

            if not client_record_id:
                self._send_json({"success": False, "error": "缺少 client_record_id"})
                return

            review_id = self.db.save_over_review(int(client_record_id), category, note)
            self._send_json({"success": True, "id": review_id, "message": "Review 已儲存"})
        except ValueError as ve:
            self._send_json({"success": False, "error": str(ve)})
        except Exception as e:
            logger.error(f"Over review save error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_over_review_delete(self):
        """API: 刪除過檢 Review"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            client_record_id = data.get("client_record_id")
            if not client_record_id:
                self._send_json({"success": False, "error": "缺少 client_record_id"})
                return

            deleted = self.db.delete_over_review(int(client_record_id))
            if deleted:
                self._send_json({"success": True, "message": "Review 已刪除"})
            else:
                self._send_json({"success": False, "error": "Review 不存在"})
        except Exception as e:
            logger.error(f"Over review delete error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    # ── Scratch 救回審查頁面 ────────────────────────

    def _handle_scratch_review_page(self, path: str):
        """渲染 Scratch 救回審查頁面"""
        template = self.jinja_env.get_template("scratch_review.html")
        html = template.render(request_path=path)
        self._send_response(200, html)

    def _handle_scratch_review_list_api(self, query: dict):
        """API: 列出被 scratch filter 救回的 tile"""
        try:
            start_date = query.get('start_date', [''])[0] or None
            end_date = query.get('end_date', [''])[0] or None
            order_by = query.get('order', ['latest'])[0] or 'latest'
            filter_state = query.get('filter', ['pending'])[0] or 'pending'
            if filter_state not in self.db._SCRATCH_REVIEW_FILTER:
                filter_state = 'pending'
            try:
                limit = int(query.get('limit', ['24'])[0])
            except (ValueError, TypeError):
                limit = 24
            try:
                offset = int(query.get('offset', ['0'])[0])
            except (ValueError, TypeError):
                offset = 0

            items = self.db.list_scratch_rescued_tiles(
                start_date=start_date,
                end_date=end_date,
                order_by=order_by,
                limit=limit,
                offset=offset,
                filter_state=filter_state,
            )
            base = self.heatmap_base_dir
            for it in items:
                hm = it.pop("heatmap_path", "")
                it["heatmap_url"] = f"/heatmaps/{hm_relative(hm, base)}" if (hm and base and hm_relative(hm, base)) else ""

            counts = self.db.count_scratch_rescued_tiles(start_date=start_date, end_date=end_date)
            total_all = counts["total"]
            marked = counts["marked"]
            filtered_total = {
                "pending": max(0, total_all - marked),
                "marked": marked,
                "all": total_all,
            }[filter_state]
            self._send_json({
                "success": True,
                "total": filtered_total,
                "total_all": total_all,
                "marked": marked,
                "filter": filter_state,
                "items": items,
                "limit": limit,
                "offset": offset,
            })
        except Exception as e:
            logger.error(f"Scratch review list error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_scratch_review_mark(self):
        """API: 標記 tile 為誤救"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            tile_id = data.get("tile_id")
            note = data.get("note", "") or ""
            if not tile_id:
                self._send_json({"success": False, "error": "缺少 tile_id"})
                return

            review_id = self.db.mark_scratch_misrescue(int(tile_id), note)
            self._send_json({"success": True, "id": review_id, "message": "已標記為誤救"})
        except ValueError as ve:
            self._send_json({"success": False, "error": str(ve)})
        except Exception as e:
            logger.error(f"Scratch review mark error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_scratch_review_unmark(self):
        """API: 取消誤救標記"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            tile_id = data.get("tile_id")
            if not tile_id:
                self._send_json({"success": False, "error": "缺少 tile_id"})
                return

            deleted = self.db.unmark_scratch_misrescue(int(tile_id))
            self._send_json({"success": True, "deleted": deleted, "message": "已取消標記"})
        except Exception as e:
            logger.error(f"Scratch review unmark error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_scratch_review_export(self):
        """API: 匯出已標記誤救 tile 為 DINOv2 hard-negative 訓練樣本。

        Body (JSON, 可選): {start_date, end_date}
        """
        try:
            content_length = int(self.headers.get('Content-Length', 0) or 0)
            data = {}
            if content_length > 0:
                body = self.rfile.read(content_length)
                if body:
                    try:
                        data = json.loads(body.decode('utf-8'))
                    except json.JSONDecodeError:
                        data = {}

            start_date = data.get("start_date") or None
            end_date = data.get("end_date") or None

            from capi_scratch_export import export_misrescue_samples, DEFAULT_BASE_DIR
            server_inst = self._capi_server_instance
            path_mapping = getattr(server_inst, "path_mapping", {}) if server_inst else {}
            base_dir = self._export_base_dir("scratch_export", DEFAULT_BASE_DIR)
            base_dir.mkdir(parents=True, exist_ok=True)

            summary = export_misrescue_samples(
                db=self.db,
                base_dir=base_dir,
                path_mapping=path_mapping,
                start_date=start_date,
                end_date=end_date,
                rotate_180=self._inference_rotate_180_enabled(),
            )
            summary["success"] = True
            summary["base_dir"] = str(base_dir)
            self._send_json(summary)
        except Exception as e:
            logger.error(f"Scratch review export error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_ric_report_api(self, query: dict):
        """API: 取得 RIC 比對報表資料"""
        try:
            batch_id_str = query.get('batch_id', [''])[0]
            batch_id = int(batch_id_str) if batch_id_str else None
        except (ValueError, TypeError):
            batch_id = None

        stats = self.db.get_ric_accuracy_stats(batch_id) if self.db else {}
        self._send_json(stats)

    def _handle_inference_stats_api(self, query: dict):
        """API: 取得 AI 推論紀錄統計資料"""
        try:
            start_date = query.get('start_date', [''])[0] or None
            end_date = query.get('end_date', [''])[0] or None
            stats = self.db.get_inference_stats(start_date, end_date) if self.db else {"success": False, "error": "DB not available"}
            self._send_json(stats)
        except Exception as e:
            logger.error(f"Inference stats API error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_mes_comparison_api(self, query: dict):
        """API: 將 AI 推論結果與 MES Report 人工不良判定比對。"""
        if not self._mes_comparison_lock.acquire(blocking=False):
            logger.warning("[MES Report] Duplicate query rejected: another comparison is still running")
            self._send_json(
                {
                    "success": False,
                    "error": "MES Report 查詢進行中，請等待目前查詢完成。",
                },
                status=409,
            )
            return

        request_started = time.monotonic()
        stage_timings = {}
        try:
            if not self.db:
                self._send_json({"success": False, "error": "DB not available"}, status=503)
                return

            start_date = query.get("start_date", [""])[0] or None
            end_date = query.get("end_date", [""])[0] or None
            ignore_aoi_ok = query.get("ignore_aoi_ok", ["0"])[0] == "1"
            panel_id = query.get("panel_id", [""])[0].strip()
            stage_started = time.monotonic()
            records = self.db.get_mes_comparison_records(
                start_date,
                end_date,
                ignore_aoi_ok=ignore_aoi_ok,
                panel_id=panel_id or None,
            )
            stage_timings["sqlite"] = time.monotonic() - stage_started

            from capi_mes_report import (
                OracleMESRepository,
                apply_mes_review_miss_policy,
                build_mes_comparison,
                build_mes_review_summary,
                _parse_datetime,
            )

            server_inst = self._capi_server_instance
            server_config = getattr(server_inst, "server_config", {}) if server_inst else {}
            mes_report_config = server_config.get("mes_report") or {}
            repository = OracleMESRepository(mes_report_config)
            defects = {}
            stage_started = time.monotonic()
            if records:
                cutoffs = [_parse_datetime(row.get("request_time")) for row in records]
                valid_cutoffs = [value for value in cutoffs if value is not None]
                if valid_cutoffs:
                    panel_ids = [row.get("glass_id", "") for row in records]
                    logger.info(
                        "[MES Report] Local records ready: range=%s..%s, records=%d, unique_panels=%d, ignore_aoi_ok=%s, panel_id=%s",
                        start_date, end_date, len(records),
                        len({str(value or "").strip().upper() for value in panel_ids if str(value or "").strip()}),
                        ignore_aoi_ok, panel_id or "(all)",
                    )
                    defects = repository.fetch_defects(
                        panel_ids,
                        min(valid_cutoffs),
                    )
            stage_timings["oracle"] = time.monotonic() - stage_started

            stage_started = time.monotonic()
            report = build_mes_comparison(
                records,
                defects,
                host_name=_get_host_identity(),
            )
            stage_timings["comparison"] = time.monotonic() - stage_started
            stage_started = time.monotonic()
            review_rows = self.db.get_mes_comparison_reviews([
                row.get("id") for row in report["records"] if row.get("id") is not None
            ])
            reviews_by_record = {
                int(row["inference_record_id"]): row for row in review_rows
            }
            for row in report["records"]:
                record_id = row.get("id")
                row["review"] = (
                    reviews_by_record.get(int(record_id))
                    if record_id is not None else None
                )
            apply_mes_review_miss_policy(report)
            report["review_summary"] = build_mes_review_summary(report["records"])
            report["ng_validation_summary"] = self.db.get_ng_validation_summary()
            report.update({
                "success": True,
                "source": repository.source_label,
                "rule": "DEFT_OPER=1600、IF_NEWER=Y、推論時間後、排除 PCK21、X/Y 皆有值",
                "ignore_aoi_ok": ignore_aoi_ok,
                "panel_id": panel_id,
            })
            stage_timings["review"] = time.monotonic() - stage_started
            stage_timings["backend"] = time.monotonic() - request_started
            report["timing"] = {
                f"{name}_seconds": round(seconds, 3)
                for name, seconds in stage_timings.items()
            }
            logger.info(
                "[MES Report] Comparison response ready: records=%d, sqlite=%.2fs, oracle=%.2fs, comparison=%.2fs, review=%.2fs, backend=%.2fs",
                len(report["records"]),
                stage_timings["sqlite"],
                stage_timings["oracle"],
                stage_timings["comparison"],
                stage_timings["review"],
                stage_timings["backend"],
            )
            send_metrics = self._send_json(
                report,
                compact=True,
                compress=True,
                server_timing=stage_timings,
            )
            logger.info(
                "[MES Report] Response sent: json=%.2fs, gzip=%.2fs, write=%.2fs, size=%d/%d bytes, compressed=%s",
                send_metrics["serialize_seconds"],
                send_metrics["compression_seconds"],
                send_metrics["write_seconds"],
                send_metrics["response_bytes"],
                send_metrics["uncompressed_bytes"],
                send_metrics["compressed"],
            )
        except ValueError as e:
            self._send_json({"success": False, "error": str(e)}, status=400)
        except Exception as e:
            logger.error("MES comparison API error: %s", e, exc_info=True)
            self._send_json({"success": False, "error": str(e)}, status=502)
        finally:
            self._mes_comparison_lock.release()

    def _handle_mes_report_detail_api(self, query: dict):
        """API: 按需取得一筆 AI 推論對應的完整 MES Report 欄位。"""
        try:
            if not self.db:
                self._send_json({"success": False, "error": "DB not available"}, status=503)
                return

            record_id_text = query.get("record_id", [""])[0]
            try:
                record_id = int(record_id_text)
            except (TypeError, ValueError):
                self._send_json({"success": False, "error": "record_id 格式錯誤"}, status=400)
                return

            record = self.db.get_mes_comparison_record(record_id)
            if not record:
                self._send_json({"success": False, "error": "找不到該筆 AI 推論紀錄"}, status=404)
                return

            from capi_mes_report import OracleMESRepository, WP_DEFTHIS_COLUMNS, _parse_datetime

            panel_id = str(record.get("glass_id") or "").strip()
            cutoff = _parse_datetime(record.get("request_time"))
            if not panel_id or cutoff is None:
                self._send_json({"success": False, "error": "該筆推論缺少玻璃 ID 或推論時間"}, status=400)
                return

            server_inst = self._capi_server_instance
            server_config = getattr(server_inst, "server_config", {}) if server_inst else {}
            repository = OracleMESRepository(server_config.get("mes_report") or {})
            rows = repository.fetch_report_details(panel_id, cutoff)
            detail_columns = list(rows[0]) if rows else list(WP_DEFTHIS_COLUMNS)
            self._send_json({
                "success": True,
                "source": repository.source_label,
                "rule": "同玻璃 ID、DEFT_OPER=1600、IF_NEWER=Y、推論時間後",
                "columns": detail_columns,
                "inference": record,
                "rows": rows,
            })
        except Exception as e:
            logger.error("MES report detail API error: %s", e, exc_info=True)
            self._send_json({"success": False, "error": str(e)}, status=502)

    @staticmethod
    def _ng_validation_base_dir_for_server(server_inst) -> Path:
        server_config = getattr(server_inst, "server_config", {}) if server_inst else {}
        ng_config = server_config.get("ng_validation") or {}
        configured = str(ng_config.get("base_dir") or "").strip()
        if configured:
            return Path(configured).resolve()

        dataset_config = server_config.get("dataset_export") or {}
        dataset_base = str(dataset_config.get("base_dir") or "").strip()
        if dataset_base:
            return (Path(dataset_base).parent / "ng_validation").resolve()
        return (Path.cwd() / "datasets" / "ng_validation").resolve()

    def _ng_validation_base_dir(self) -> Path:
        return self._ng_validation_base_dir_for_server(self._capi_server_instance)

    @staticmethod
    def _query_int(query: dict, key: str) -> Optional[int]:
        value = query.get(key, [""])
        value = value[0] if isinstance(value, list) else value
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _mes_review_resolve_source(self, image_path: str) -> Path:
        server_inst = self._capi_server_instance
        path_mapping = getattr(server_inst, "path_mapping", {}) if server_inst else {}
        return resolve_source_path(str(image_path or ""), path_mapping)

    def _handle_mes_review_candidates_api(self, query: dict):
        """GET: 取得指定推論紀錄中由 AOI 座標產生的五光源候選 tile。"""
        record_id = self._query_int(query, "record_id")
        if record_id is None:
            self._send_json({"success": False, "error": "record_id 格式錯誤"}, status=400)
            return
        record = self.db.get_mes_comparison_record(record_id) if self.db else None
        if not record:
            self._send_json({"success": False, "error": "找不到該筆 AI 推論紀錄"}, status=404)
            return

        candidates = []
        for row in self.db.get_mes_review_aoi_candidates(record_id):
            lighting = canonical_image_prefix(row.get("image_name") or "")
            if lighting not in _MES_REVIEW_LIGHTINGS:
                continue
            source_path = self._mes_review_resolve_source(row.get("image_path") or "")
            item = dict(row)
            item["lighting"] = lighting
            item["source_available"] = source_path.is_file()
            if not source_path.is_file():
                item["collectable_reason"] = "原圖不存在，無法保存 NG crop"
            elif int(row.get("image_is_bomb") or 0) or int(row.get("is_bomb") or 0):
                item["collectable_reason"] = "BOMB 模擬缺陷，不納入真實 NG 驗證庫"
            else:
                item["collectable_reason"] = ""
            item["collectable"] = bool(
                source_path.is_file()
                and not int(row.get("image_is_bomb") or 0)
                and not int(row.get("is_bomb") or 0)
            )
            item["crop_url"] = (
                f"/api/ric/mes-review/crop?tile_result_id={int(row['tile_result_id'])}"
            )
            candidates.append(item)

        self._send_json({
            "success": True,
            "record": record,
            "review": self.db.get_mes_comparison_review(record_id),
            "candidates": candidates,
            "message": (
                f"找到 {len(candidates)} 個 AOI 座標候選 tile"
                if candidates else
                "找不到 AOI 座標 tile；可能未提供 AOI 座標或 tile 明細已超過保留期"
            ),
        })

    def _handle_mes_review_crop_api(self, query: dict):
        """GET: 讀取 AOI 候選 tile 的原始 512 crop。"""
        tile_result_id = self._query_int(query, "tile_result_id")
        if tile_result_id is None:
            self._send_error(400, "tile_result_id 格式錯誤")
            return
        candidate = self.db.get_mes_review_aoi_candidate(tile_result_id)
        if not candidate:
            self._send_404()
            return

        lighting = canonical_image_prefix(candidate.get("image_name") or "")
        if lighting not in _MES_REVIEW_LIGHTINGS:
            self._send_error(400, "此光源不納入 PatchCore NG 驗證庫")
            return
        source_path = self._mes_review_resolve_source(candidate.get("image_path") or "")
        if not source_path.is_file():
            self._send_404()
            return

        import cv2

        image = self._read_inference_image(source_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            self._send_404()
            return
        crop = crop_patchcore_tile(
            image,
            int(candidate.get("tile_x") or 0),
            int(candidate.get("tile_y") or 0),
            max(1, int(candidate.get("tile_w") or 512)),
            max(1, int(candidate.get("tile_h") or 512)),
        )
        self._send_image_array_png(crop)

    def _prepare_ng_validation_samples(
        self,
        record: Dict,
        candidates: List[Dict],
    ) -> List[Dict]:
        """裁切並保存人工勾選的 AOI tile，回傳 DB snapshot rows。"""
        import cv2

        base_dir = self._ng_validation_base_dir()
        image_cache: Dict[str, Any] = {}
        rows = []
        for candidate in candidates:
            lighting = canonical_image_prefix(candidate.get("image_name") or "")
            if lighting not in _MES_REVIEW_LIGHTINGS:
                raise ValueError(f"光源不納入 NG 驗證庫: {lighting}")
            if int(candidate.get("image_is_bomb") or 0) or int(candidate.get("is_bomb") or 0):
                raise ValueError(f"炸彈候選不可加入 NG 驗證庫: tile {candidate['tile_result_id']}")

            source_path = self._mes_review_resolve_source(candidate.get("image_path") or "")
            cache_key = str(source_path)
            if cache_key not in image_cache:
                image_cache[cache_key] = self._read_inference_image(
                    source_path, cv2.IMREAD_UNCHANGED
                ) if source_path.is_file() else None
            image = image_cache[cache_key]
            if image is None:
                raise ValueError(f"原圖不存在或無法讀取: {source_path}")

            x = int(candidate.get("tile_x") or 0)
            y = int(candidate.get("tile_y") or 0)
            w = max(1, int(candidate.get("tile_w") or 512))
            h = max(1, int(candidate.get("tile_h") or 512))
            crop = crop_patchcore_tile(image, x, y, w, h)
            if crop.size == 0:
                raise ValueError(f"AOI tile 裁切為空: {candidate['tile_result_id']}")

            zone = str(candidate.get("zone") or "unknown").strip().lower() or "unknown"
            safe_model = re.sub(
                r"[^A-Za-z0-9_.-]+", "_", str(record.get("model_id") or "unknown")
            )[:80] or "unknown"
            safe_lighting = re.sub(r"[^A-Za-z0-9_.-]+", "_", lighting)[:40]
            safe_zone = re.sub(r"[^A-Za-z0-9_.-]+", "_", zone)[:40] or "unknown"
            safe_glass = re.sub(
                r"[^A-Za-z0-9_.-]+", "_", str(record.get("glass_id") or "panel")
            )[:80] or "panel"
            request_day = re.sub(
                r"[^0-9]+", "", str(record.get("request_time") or "")[:10]
            ) or datetime.now().strftime("%Y%m%d")
            out_dir = base_dir / safe_model / safe_lighting / safe_zone / "crop"
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = (
                f"{request_day}_{safe_glass}_r{int(record['id'])}"
                f"_t{int(candidate['tile_result_id'])}.png"
            )
            crop_path = out_dir / filename
            if not cv2.imwrite(str(crop_path), crop):
                raise ValueError(f"NG crop 寫入失敗: {crop_path}")

            rows.append({
                "tile_result_id": int(candidate["tile_result_id"]),
                "image_result_id": int(candidate["image_result_id"]),
                "image_name": str(candidate.get("image_name") or ""),
                "source_image_path": str(source_path),
                "lighting": lighting,
                "zone": zone,
                "aoi_defect_code": str(candidate.get("aoi_defect_code") or ""),
                "aoi_product_x": int(
                    candidate["aoi_product_x"]
                    if candidate.get("aoi_product_x") is not None else -1
                ),
                "aoi_product_y": int(
                    candidate["aoi_product_y"]
                    if candidate.get("aoi_product_y") is not None else -1
                ),
                "aoi_image_x": int(
                    candidate["aoi_image_x"]
                    if candidate.get("aoi_image_x") is not None else -1
                ),
                "aoi_image_y": int(
                    candidate["aoi_image_y"]
                    if candidate.get("aoi_image_y") is not None else -1
                ),
                "tile_x": x,
                "tile_y": y,
                "tile_w": w,
                "tile_h": h,
                "ai_score": float(candidate.get("ai_score") or 0.0),
                "crop_path": str(crop_path.resolve()),
            })
        return rows

    def _handle_mes_review_save(self):
        """POST: 儲存 Report 人工 Review，必要時同步 NG 驗證樣本。"""
        data = self._read_json_body()
        if data is None:
            return
        try:
            record_id = int(data.get("record_id"))
        except (TypeError, ValueError):
            self._send_json({"success": False, "error": "record_id 格式錯誤"}, status=400)
            return
        record = self.db.get_mes_comparison_record(record_id) if self.db else None
        if not record:
            self._send_json({"success": False, "error": "找不到該筆 AI 推論紀錄"}, status=404)
            return

        review_type = str(data.get("review_type") or "").strip()
        category = str(data.get("category") or "").strip()
        mes_judgment = str(data.get("mes_judgment") or "").strip().upper()
        confirmed_ng = bool(data.get("confirmed_ng"))
        if mes_judgment not in {"OK", "NG"}:
            self._send_json({"success": False, "error": "MES 判定格式錯誤"}, status=400)
            return

        raw_tile_ids = data.get("selected_tile_ids") or []
        if not isinstance(raw_tile_ids, list):
            self._send_json({"success": False, "error": "selected_tile_ids 必須是陣列"}, status=400)
            return
        try:
            selected_tile_ids = sorted({int(value) for value in raw_tile_ids})
        except (TypeError, ValueError):
            self._send_json({"success": False, "error": "selected_tile_ids 格式錯誤"}, status=400)
            return

        all_candidates = self.db.get_mes_review_aoi_candidates(record_id)
        candidates_by_id = {
            int(row["tile_result_id"]): row
            for row in all_candidates
            if canonical_image_prefix(row.get("image_name") or "") in _MES_REVIEW_LIGHTINGS
        }
        missing_ids = [tile_id for tile_id in selected_tile_ids if tile_id not in candidates_by_id]
        if missing_ids:
            self._send_json({
                "success": False,
                "error": f"選取的 AOI tile 不屬於此推論紀錄: {missing_ids[:10]}",
            }, status=400)
            return
        if confirmed_ng and not selected_tile_ids:
            self._send_json({
                "success": False,
                "error": "加入 NG 驗證庫時，至少要勾選一張肉眼確認可見的 AOI 圖片",
            }, status=400)
            return

        try:
            samples = self._prepare_ng_validation_samples(
                record,
                [candidates_by_id[tile_id] for tile_id in selected_tile_ids],
            ) if confirmed_ng else []
            review = self.db.save_mes_comparison_review(
                inference_record_id=record_id,
                glass_id=record.get("glass_id") or "",
                model_id=record.get("model_id") or "",
                machine_no=record.get("machine_no") or "",
                request_time=record.get("request_time") or "",
                ai_judgment=record.get("ai_judgment") or "",
                mes_judgment=mes_judgment,
                review_type=review_type,
                category=category,
                note=str(data.get("note") or "").strip(),
                reviewer="",
                confirmed_ng=confirmed_ng,
                samples=samples,
            )
        except ValueError as exc:
            self._send_json({"success": False, "error": str(exc)}, status=400)
            return
        except Exception as exc:
            logger.error("Save MES review failed: %s", exc, exc_info=True)
            self._send_json({"success": False, "error": str(exc)}, status=500)
            return

        self._send_json({
            "success": True,
            "review": review,
            "ng_validation_summary": self.db.get_ng_validation_summary(),
            "message": (
                f"Review 已儲存，NG 驗證庫同步 {review.get('ng_sample_count', 0)} 張"
            ),
        })

    def _handle_mes_review_delete(self):
        data = self._read_json_body()
        if data is None:
            return
        try:
            record_id = int(data.get("record_id"))
        except (TypeError, ValueError):
            self._send_json({"success": False, "error": "record_id 格式錯誤"}, status=400)
            return
        deleted = self.db.delete_mes_comparison_review(record_id) if self.db else False
        if not deleted:
            self._send_json({"success": False, "error": "找不到 Review"}, status=404)
            return
        self._send_json({
            "success": True,
            "ng_validation_summary": self.db.get_ng_validation_summary(),
        })

    def _handle_ng_validation_api(self, query: dict):
        def _query_text(key: str) -> str:
            value = query.get(key, [""])
            return str(value[0] if isinstance(value, list) and value else value or "").strip()

        limit = self._query_int(query, "limit") or 100
        offset = self._query_int(query, "offset") or 0
        samples, total = self.db.list_ng_validation_samples(
            machine_no=_query_text("machine_no"),
            model_id=_query_text("model_id"),
            lighting=_query_text("lighting"),
            zone=_query_text("zone"),
            limit=limit,
            offset=offset,
        )
        for sample in samples:
            sample["file_url"] = f"/api/ric/ng-validation/file?id={int(sample['id'])}"
        self._send_json({
            "success": True,
            "samples": samples,
            "total": total,
            "summary": self.db.get_ng_validation_summary(),
            "base_dir": str(self._ng_validation_base_dir()),
        })

    def _handle_ng_validation_delete(self):
        """POST: 刪除單筆 NG 驗證 crop，保留原始推論與 Review。"""
        data = self._read_json_body()
        if data is None:
            return
        try:
            sample_id = int(data.get("sample_id"))
        except (TypeError, ValueError):
            self._send_json({"success": False, "error": "sample_id 格式錯誤"}, status=400)
            return

        sample = self.db.get_ng_validation_sample(sample_id) if self.db else None
        if not sample:
            self._send_json({"success": False, "error": "找不到 NG 驗證樣本"}, status=404)
            return
        try:
            base_dir = self._ng_validation_base_dir()
            crop_path = Path(sample.get("crop_path") or "").resolve()
            crop_path.relative_to(base_dir)
        except (OSError, ValueError):
            self._send_json({
                "success": False,
                "error": "NG crop 路徑不在設定的驗證資料庫目錄內",
            }, status=403)
            return

        file_deleted = False
        try:
            if crop_path.exists():
                if not crop_path.is_file():
                    raise OSError(f"NG crop 不是一般檔案: {crop_path}")
                crop_path.unlink()
                file_deleted = True
        except OSError as exc:
            logger.error("Delete NG validation crop failed: %s", exc, exc_info=True)
            self._send_json({"success": False, "error": f"NG 圖片刪除失敗: {exc}"}, status=500)
            return

        deleted = self.db.remove_ng_validation_sample(sample_id)
        if not deleted:
            self._send_json({"success": False, "error": "NG 驗證樣本已不存在"}, status=404)
            return
        self._send_json({
            "success": True,
            "sample_id": sample_id,
            "file_deleted": file_deleted,
            "summary": self.db.get_ng_validation_summary(),
        })

    def _handle_ng_validation_file_api(self, query: dict):
        sample_id = self._query_int(query, "id")
        if sample_id is None:
            self._send_error(400, "id 格式錯誤")
            return
        sample = self.db.get_ng_validation_sample(sample_id) if self.db else None
        if not sample:
            self._send_404()
            return
        try:
            base_dir = self._ng_validation_base_dir()
            crop_path = Path(sample.get("crop_path") or "").resolve()
            crop_path.relative_to(base_dir)
        except (OSError, ValueError):
            self._send_error(403, "NG crop path outside configured base_dir")
            return
        if not crop_path.is_file():
            self._send_404()
            return
        self._send_binary(str(crop_path))

    def _handle_debug_inference_run(self):
        """API: 執行 Debug 單圖推論"""
        import time as _time
        import cv2
        import numpy as np

        # 讀取 POST body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode('utf-8'))
        except Exception:
            self._send_json({"error": "Invalid JSON body"})
            return

        image_path_str = data.get("image_path", "").strip()
        if not image_path_str:
            self._send_json({"error": "請提供圖片路徑 (image_path)"})
            return

        image_path = Path(image_path_str)
        if not image_path.exists():
            self._send_json({"error": f"檔案不存在: {image_path}"})
            return
        if not image_path.is_file():
            self._send_json({"error": f"不是檔案: {image_path}"})
            return

        if self.inferencer is None:
            self._send_json({"error": "推論器尚未載入 (inferencer is None)"})
            return

        # Debug 門檻：預設 0.5（比正式環境低，用於漏檢排查）
        debug_threshold = float(data.get("threshold", 0.5))
        # 邊緣衰減覆寫：None = 使用 config 預設值，0 = 停用，>0 = 自訂寬度
        edge_margin_raw = data.get("edge_margin_px")
        edge_margin_override = int(edge_margin_raw) if edge_margin_raw is not None and str(edge_margin_raw).strip() != "" else None
        
        # 灰塵檢測參數覆寫
        dust_ext_raw = data.get("dust_extension")
        dust_ext_override = int(dust_ext_raw) if dust_ext_raw is not None and str(dust_ext_raw).strip() != "" else None
        
        dust_iou_thr_raw = data.get("dust_heatmap_iou_threshold")
        dust_iou_thr_override = float(dust_iou_thr_raw) if dust_iou_thr_raw is not None and str(dust_iou_thr_raw).strip() != "" else None
        
        dust_top_pct_raw = data.get("dust_heatmap_top_percent")
        dust_top_pct_override = float(dust_top_pct_raw) if dust_top_pct_raw is not None and str(dust_top_pct_raw).strip() != "" else None
        
        dust_metric_override = data.get("dust_heatmap_metric")
        
        otsu_offset_raw = data.get("otsu_offset")
        otsu_offset_override = int(otsu_offset_raw) if otsu_offset_raw is not None and str(otsu_offset_raw).strip() != "" else None

        patchcore_overrides = {}
        if "patchcore_filter_enabled" in data:
            patchcore_overrides["patchcore_filter_enabled"] = bool(data["patchcore_filter_enabled"])
        if "patchcore_blur_sigma" in data and str(data["patchcore_blur_sigma"]).strip() != "":
            patchcore_overrides["patchcore_blur_sigma"] = float(data["patchcore_blur_sigma"])
        if "patchcore_min_area" in data and str(data["patchcore_min_area"]).strip() != "":
            patchcore_overrides["patchcore_min_area"] = int(data["patchcore_min_area"])
        if "patchcore_score_metric" in data and str(data["patchcore_score_metric"]).strip() != "":
            patchcore_overrides["patchcore_score_metric"] = str(data["patchcore_score_metric"])

        try:
            total_start = _time.time()

            img_prefix = self.inferencer._get_image_prefix(image_path.name)
            is_v2 = getattr(self.inferencer.config, "is_new_architecture", False)

            if is_v2:
                # 新架構：單圖 zone-aware 推論 (preprocess + per-tile inner/edge model 由 helper 內部處理)
                lighting_map = self.inferencer.config.model_mapping.get(img_prefix, {})
                if not isinstance(lighting_map, dict) or "inner" not in lighting_map or "edge" not in lighting_map:
                    self._send_json({"error": f"找不到 {image_path.name} 對應的模型 (新架構: model_mapping 缺 {img_prefix} 的 inner/edge)"})
                    return

                if hasattr(self, '_gpu_lock') and self._gpu_lock:
                    with self._gpu_lock:
                        result = self.inferencer.run_inference_v2_single_image(
                            image_path,
                            threshold=debug_threshold,
                            edge_margin_override=edge_margin_override,
                            patchcore_overrides=patchcore_overrides if patchcore_overrides else None,
                            otsu_offset_override=otsu_offset_override,
                        )
                else:
                    result = self.inferencer.run_inference_v2_single_image(
                        image_path,
                        threshold=debug_threshold,
                        edge_margin_override=edge_margin_override,
                        patchcore_overrides=patchcore_overrides if patchcore_overrides else None,
                        otsu_offset_override=otsu_offset_override,
                    )

                if result is None:
                    self._send_json({"error": f"無法載入或預處理圖片: {image_path}"})
                    return

                model_name = f"{img_prefix} (inner+edge)"
            else:
                # 舊架構：單一 inferencer 路由 (preprocess + run_inference)
                result = self.inferencer.preprocess_image(image_path, otsu_offset_override=otsu_offset_override)
                if result is None:
                    self._send_json({"error": f"無法載入或預處理圖片: {image_path}"})
                    return

                target_inferencer = self.inferencer._get_inferencer_for_prefix(img_prefix)

                model_name = "預設模型"
                if img_prefix in self.inferencer._model_mapping:
                    model_name = self.inferencer._model_mapping[img_prefix].name

                if target_inferencer is None:
                    self._send_json({"error": f"找不到 {image_path.name} 對應的模型"})
                    return

                # model_id: 從 POST 資料取得（可選），用於推導產品解析度
                debug_model_id = data.get("model_id")

                if hasattr(self, '_gpu_lock') and self._gpu_lock:
                    with self._gpu_lock:
                        result = self.inferencer.run_inference(
                            result,
                            inferencer=target_inferencer,
                            threshold=debug_threshold,
                            edge_margin_override=edge_margin_override,
                            patchcore_overrides=patchcore_overrides if patchcore_overrides else None,
                            model_id=debug_model_id,
                        )
                else:
                    result = self.inferencer.run_inference(
                        result,
                        inferencer=target_inferencer,
                        threshold=debug_threshold,
                        edge_margin_override=edge_margin_override,
                        patchcore_overrides=patchcore_overrides if patchcore_overrides else None,
                        model_id=debug_model_id,
                    )

            total_time = _time.time() - total_start

            # 3. 建立 Debug heatmap 暫存目錄
            if CAPIWebHandler._debug_heatmap_dir is None:
                CAPIWebHandler._debug_heatmap_dir = Path(tempfile.mkdtemp(prefix="capi_debug_hm_"))
            debug_dir = CAPIWebHandler._debug_heatmap_dir
            debug_dir.mkdir(parents=True, exist_ok=True)

            image_name = image_path.stem

            # 4. 產生 Overview 圖
            overview_img = self.inferencer.visualize_inference_result(image_path, result)
            overview_filename = f"debug_overview_{image_name}.png"
            overview_path = debug_dir / overview_filename
            # 縮小存檔
            max_dim = 2000
            h, w = overview_img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                overview_img = cv2.resize(overview_img, (int(w * scale), int(h * scale)))
            cv2.imwrite(str(overview_path), overview_img)
            overview_url = f"/debug/heatmaps/{overview_filename}"

            # 5. 產生各 Tile 組合圖 (與推論記錄格式一致)
            tiles_data = []
            image_dir = image_path.parent
            
            # 先找是否有 OMIT 圖片 (Panel 級別共用)
            omit_candidates = []
            for pattern in ["PINIGBI*.*", "OMIT0000*.*"]:
                omit_candidates.extend(list(image_dir.glob(pattern)))
            
            omit_full = None
            if omit_candidates:
                omit_full = self._read_inference_image(omit_candidates[0], cv2.IMREAD_UNCHANGED)
                if omit_full is not None:
                    logger.info(f"[DEBUG] Found OMIT image for dust check: {omit_candidates[0].name}")

            for tile, score, anomaly_map in result.anomaly_tiles:
                # 準備 TileInfo 擴充資訊 (灰塵檢查)
                if omit_full is not None:
                    try:
                        tx, ty, tw, th = tile.x, tile.y, tile.width, tile.height
                        oh, ow = omit_full.shape[:2]
                        if tx < ow and ty < oh:
                            x2_o = min(tx + tw, ow)
                            y2_o = min(ty + th, oh)
                            omit_crop = omit_full[ty:y2_o, tx:x2_o].copy()
                            tile.omit_crop_image = omit_crop
                            
                            # A. 灰塵偵測
                            is_dust, dust_mask, bright_ratio, detail_text = self.inferencer.check_dust_or_scratch_feature(omit_crop, extension_override=dust_ext_override)
                            tile.dust_mask = dust_mask
                            tile.dust_bright_ratio = bright_ratio
                            
                            # B. IOU 計算
                            top_pct = dust_top_pct_override if dust_top_pct_override is not None else self.inferencer.config.dust_heatmap_top_percent
                            metric_mode = dust_metric_override if dust_metric_override else self.inferencer.config.dust_heatmap_metric
                            dust_iou_thr = dust_iou_thr_override if dust_iou_thr_override is not None else self.inferencer.config.dust_heatmap_iou_threshold
                            if is_dust and anomaly_map is not None:
                                iou, heatmap_binary = self.inferencer.compute_dust_heatmap_iou(
                                    dust_mask, anomaly_map, top_percent=top_pct, metric=metric_mode
                                )
                                tile.dust_heatmap_iou = iou
                                # 判定灰塵 (與正式路徑 _dust_check_one 一致)
                                metric_name = "COV" if metric_mode == "coverage" else "IOU"
                                if iou >= dust_iou_thr:
                                    tile.is_suspected_dust_or_scratch = True
                                    detail_text += f" {metric_name}:{iou:.3f}>={metric_name}_THR -> DUST"
                                else:
                                    detail_text += f" {metric_name}:{iou:.3f}<{metric_name}_THR -> REAL_NG"
                                # 產生 Debug 圖
                                tile.dust_iou_debug_image = self.inferencer.generate_dust_iou_debug_image(
                                    tile.image, anomaly_map, dust_mask, heatmap_binary, iou, top_pct, tile.is_suspected_dust_or_scratch
                                )
                            elif is_dust:
                                tile.is_suspected_dust_or_scratch = True
                                detail_text += " (no heatmap, marked as dust)"
                            else:
                                detail_text += " NO_DUST -> REAL_NG"
                            tile.dust_detail_text = detail_text
                    except Exception as e:
                        logger.warning(f"[DEBUG] Dust check processing failed for tile {tile.tile_id}: {e}")

                # 產生組合圖
                if self.heatmap_manager:
                    try:
                        composite_path = self.heatmap_manager.save_tile_heatmap(
                            save_dir=debug_dir,
                            image_name=f"debug_{image_name}",
                            tile_id=tile.tile_id,
                            tile_image=tile.image,
                            anomaly_map=anomaly_map,
                            score=score,
                            tile_info=tile,
                            score_threshold=debug_threshold,
                            iou_threshold=dust_iou_thr_override if dust_iou_thr_override is not None else getattr(self.inferencer.config, 'dust_heatmap_iou_threshold', 0.01),
                            dust_metric=getattr(self.inferencer.config, 'dust_heatmap_metric', 'coverage'),
                            dust_high_cov_threshold=getattr(self.inferencer.config, 'dust_high_cov_threshold', None),
                        )
                        tile_url = f"/debug/heatmaps/{Path(composite_path).name}"
                    except Exception as e:
                        logger.error(f"[DEBUG] Composite image generation failed for tile {tile.tile_id}: {e}")
                        # Fallback to simple overlay if composite fails
                        overlay = self.heatmap_manager.generate_heatmap_overlay(tile.image, anomaly_map, alpha=0.5)
                        tile_filename = f"debug_tile_{image_name}_t{tile.tile_id}_fallback.png"
                        cv2.imwrite(str(debug_dir / tile_filename), overlay)
                        tile_url = f"/debug/heatmaps/{tile_filename}"
                else:
                    # Fallback (無 HeatmapManager)
                    tile_img = tile.image.copy()
                    if len(tile_img.shape) == 2:
                        tile_img = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2BGR)
                    norm_map = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
                    if heatmap_color.shape[:2] != tile_img.shape[:2]:
                        heatmap_color = cv2.resize(heatmap_color, (tile_img.shape[1], tile_img.shape[0]))
                    overlay = cv2.addWeighted(tile_img, 0.5, heatmap_color, 0.5, 0)
                    tile_filename = f"debug_tile_{image_name}_t{tile.tile_id}_simple.png"
                    cv2.imwrite(str(debug_dir / tile_filename), overlay)
                    tile_url = f"/debug/heatmaps/{tile_filename}"

                tile_status = "NG"
                if tile.is_in_exclude_zone:
                    tile_status = "EXCLUDED"
                elif tile.is_bomb:
                    tile_status = "BOMB"
                elif tile.is_suspected_dust_or_scratch:
                    tile_status = "DUST"

                tiles_data.append({
                    "tile_id": tile.tile_id,
                    "x": tile.x,
                    "y": tile.y,
                    "width": tile.width,
                    "height": tile.height,
                    "score": round(score, 4),
                    "status": tile_status,
                    "is_dust": tile.is_suspected_dust_or_scratch,
                    "dust_iou": round(getattr(tile, 'dust_region_max_cov', tile.dust_heatmap_iou), 4),
                    "is_bomb": tile.is_bomb,
                    "bomb_code": tile.bomb_defect_code,
                    "is_exclude_zone": tile.is_in_exclude_zone,
                    "heatmap_url": tile_url,
                })

            # 6. 判定結果
            has_real_ng = any(
                not t.is_suspected_dust_or_scratch and not t.is_bomb and not t.is_in_exclude_zone
                for t, s, m in result.anomaly_tiles
            )
            all_dust = (
                len(result.anomaly_tiles) > 0 and
                all(t.is_suspected_dust_or_scratch for t, s, m in result.anomaly_tiles)
            )

            if has_real_ng:
                judgment = "NG"
            elif all_dust:
                judgment = "OK (DUST Filtered)"
            elif len(result.anomaly_tiles) > 0:
                judgment = "NG"
            else:
                judgment = "OK"

            response_data = {
                "success": True,
                "image_path": str(image_path),
                "image_name": image_path.name,
                "image_size": list(result.image_size),
                "judgment": judgment,
                "total_tiles": result.processed_tile_count,
                "excluded_tiles": result.excluded_tile_count,
                "anomaly_count": len(result.anomaly_tiles),
                "processing_time": round(total_time, 3),
                "threshold": debug_threshold,
                "edge_margin_px": edge_margin_override if edge_margin_override is not None else self.inferencer.config.edge_margin_px,
                "overview_url": overview_url,
                "tiles": tiles_data,
                "image_prefix": img_prefix,
                "image_prefix_label": source_image_prefix(image_path.name),
                "model_name": model_name,
            }

            self._send_json(response_data)
            logger.info(f"[DEBUG] Inference {image_path.name}: {judgment} ({total_time:.2f}s, {len(result.anomaly_tiles)} anomalies)")

        except Exception as e:
            logger.error(f"[DEBUG] Inference error: {e}", exc_info=True)
            self._send_json({"error": f"推論失敗: {str(e)}"})

    def _handle_debug_mark_detection(self):
        """API: 執行 Debug Mark 檢測（不跑 PatchCore 推論）"""
        import time as _time
        import cv2
        from capi_mark_detector import detect_panel_mark

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode('utf-8'))
        except Exception:
            self._send_json({"error": "Invalid JSON body"})
            return

        image_path_str = data.get("image_path", "").strip()
        if not image_path_str:
            self._send_json({"error": "請提供圖片路徑 (image_path)"})
            return

        image_path = Path(image_path_str)
        if not image_path.exists():
            self._send_json({"error": f"檔案不存在: {image_path}"})
            return
        if not image_path.is_file():
            self._send_json({"error": f"不是檔案: {image_path}"})
            return

        try:
            start = _time.time()
            image = self._read_inference_image(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                self._send_json({"error": f"無法讀取圖片: {image_path}"})
                return

            if CAPIWebHandler._debug_heatmap_dir is None:
                CAPIWebHandler._debug_heatmap_dir = Path(tempfile.mkdtemp(prefix="capi_debug_hm_"))
            debug_dir = CAPIWebHandler._debug_heatmap_dir
            debug_dir.mkdir(parents=True, exist_ok=True)

            mark_detection = detect_panel_mark(image, include_debug=True)
            mark_debug_images = mark_detection.pop("_debug_images", None)
            if mark_debug_images:
                image_name = image_path.stem
                ts = int(_time.time() * 1000)
                for key, debug_image in mark_debug_images.items():
                    mark_filename = f"debug_mark_{key}_{image_name}_{ts}.png"
                    cv2.imwrite(str(debug_dir / mark_filename), debug_image)
                    mark_detection[f"{key}_url"] = f"/debug/heatmaps/{mark_filename}"

            h, w = image.shape[:2]
            mark_detection.update({
                "success": True,
                "image_path": str(image_path),
                "image_name": image_path.name,
                "image_size": [w, h],
                "processing_time": round(_time.time() - start, 3),
            })
            self._send_json(mark_detection)
            logger.info(f"[DEBUG] Mark detection {image_path.name}: {mark_detection.get('text', 'NOT_FOUND')}")
        except Exception as e:
            logger.error(f"[DEBUG] Mark detection error: {e}", exc_info=True)
            self._send_json({"error": f"Mark 檢測失敗: {str(e)}"})

    def _handle_api_debug_edge_inspect(self):
        """API: 測試單邊 CV 邊緣檢測"""
        import cv2
        import numpy as np
        import base64
        from capi_edge_cv import CVEdgeInspector, EdgeSideConfig
        
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode('utf-8'))
        except Exception:
            self._send_json({"error": "Invalid JSON body"})
            return

        image_path_str = data.get("image_path", "").strip()
        side = data.get("side", "left")
        
        if not image_path_str:
            self._send_json({"error": "請提供圖片路徑 (image_path)"})
            return

        image_path = Path(image_path_str)
        if not image_path.exists():
            self._send_json({"error": f"檔案不存在: {image_path}"})
            return

        try:
            # 讀取參數
            cfg = EdgeSideConfig(
                width=int(data.get("width", 450)),
                threshold=int(data.get("threshold", 5)),
                min_area=int(data.get("min_area", 70)),
                exclude_top=int(data.get("exclude_top", 80)),
                exclude_bottom=int(data.get("exclude_bottom", 80)),
                exclude_left=int(data.get("exclude_left", 10)),
                exclude_right=int(data.get("exclude_right", 10)),
            )

            # 準備推論器 (拿掉這段檢查，因為我們自己算邊界)
            # if self.inferencer is None:
            #     self._send_json({"error": "AI 推論器尚未載入，無法取得邊界參數"})
            #     return
                
            # 讀取圖片並自行找範圍，不依賴 inferencer
            image = self._read_inference_image(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                self._send_json({"error": "無法讀取圖片"})
                return
                
            image_size = (image.shape[1], image.shape[0])
            
            def _fast_otsu_bounds(img: np.ndarray) -> Tuple[int, int, int, int]:
                """輕量版 Otsu 邊界尋找，不載入模型"""
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = np.ones((15, 15), np.uint8)
                closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                img_h, img_w = img.shape[:2]
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = -float('inf'), -float('inf')
                
                for contour in contours:
                    if cv2.contourArea(contour) > 1000:
                        x, y, w, h = cv2.boundingRect(contour)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x + w, x_max)
                        y_max = max(y + h, y_max)
                
                if x_min == float('inf'):
                    return 0, 0, img_w, img_h
                return int(x_min), int(y_min), int(x_max), int(y_max)

            # 放棄使用 self.inferencer 的 calculate_otsu_bounds，因為這會觸發卡死
            # 直接強制使用 _fast_otsu_bounds 算出來的邊界
            otsu_bounds = _fast_otsu_bounds(image)

            # 準備 EdgeInspector (從自己 init)
            inspector = CVEdgeInspector()
            defects, debug_imgs = inspector.inspect_single_side(
                image, otsu_bounds, side, config_override=cfg
            )

            # 將 debug 圖片轉為 base64
            encoded_imgs = {}
            for k, img in debug_imgs.items():
                if img is not None:
                    _, buffer = cv2.imencode('.png', img)
                    b64 = base64.b64encode(buffer).decode('utf-8')
                    encoded_imgs[k] = f"data:image/png;base64,{b64}"

            self._send_json({
                "success": True,
                "defects": [
                    {
                        "area": d.area,
                        "bbox": d.bbox,
                        "center": d.center,
                        "max_diff": d.max_diff
                    } for d in defects
                ],
                "images": encoded_imgs,
                "otsu_bounds": otsu_bounds,
                "image_size": image_size,
            })

        except Exception as e:
            logger.error(f"[DEBUG] Edge Inspect error: {e}", exc_info=True)
            self._send_json({"error": f"邊緣檢測失敗: {str(e)}"})

    def _handle_api_debug_edge_inspect_corner(self):
        """API: 測試 AOI 座標角落邊緣檢測（CV 或 PatchCore 可切換）"""
        import cv2
        import numpy as np
        import base64
        from capi_edge_cv import CVEdgeInspector, EdgeInspectionConfig, clamp_median_kernel, inpaint_non_fg_region

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode('utf-8'))
        except Exception:
            self._send_json({"error": "Invalid JSON body"})
            return

        image_path_str = data.get("image_path", "").strip()
        if not image_path_str:
            self._send_json({"error": "請提供圖片路徑 (image_path)"})
            return

        image_path = Path(image_path_str)
        if not image_path.exists():
            self._send_json({"error": f"檔案不存在: {image_path}"})
            return

        inspector_mode = str(data.get("inspector", "cv")).lower().strip()
        if inspector_mode == "patchcore":
            return self._handle_api_debug_edge_corner_patchcore(data, image_path)
        if inspector_mode == "fusion":
            return self._handle_api_debug_edge_corner_fusion(data, image_path)

        try:
            roi_x = int(data.get("roi_x", 0))
            roi_y = int(data.get("roi_y", 0))
            tile_size = int(data.get("tile_size", 512))
            threshold = int(data.get("threshold", 4))
            min_area = int(data.get("min_area", 40))
            solidity_min = float(data.get("solidity_min", 0.2))
            polygon_erode_px = int(data.get("polygon_erode_px", 3))
            morph_open_kernel = int(data.get("morph_open_kernel", 3))
            min_max_diff = int(data.get("min_max_diff", 20))
            line_min_length = int(data.get("line_min_length", 30))
            line_max_width = int(data.get("line_max_width", 3))
            boundary_padding = int(data.get("boundary_padding", 15))
            boundary_min_brightness = int(data.get("boundary_min_brightness", 15))

            image = self._read_inference_image(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                self._send_json({"error": "無法讀取圖片"})
                return

            if image.dtype == np.uint16:
                image = (image / 256).astype(np.uint8)

            img_h, img_w = image.shape[:2]

            # Fast Otsu bounds (與 /api/debug/edge-inspect 一致)
            # 同時回傳 closing mask 供 polygon 偵測使用
            def _fast_otsu_bounds(img: np.ndarray) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
                if len(img.shape) == 3:
                    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    g = img
                blurred = cv2.GaussianBlur(g, (5, 5), 0)
                _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = np.ones((15, 15), np.uint8)
                closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = -float('inf'), -float('inf')
                for contour in contours:
                    if cv2.contourArea(contour) > 1000:
                        x, y, w, h = cv2.boundingRect(contour)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x + w, x_max)
                        y_max = max(y + h, y_max)
                if x_min == float('inf'):
                    return (0, 0, img.shape[1], img.shape[0]), closing
                return (int(x_min), int(y_min), int(x_max), int(y_max)), closing

            otsu_bounds, otsu_closing = _fast_otsu_bounds(image)

            # 用 inferencer 的 polygon 偵測 (對齊 production)，無 inferencer 時 fallback 為 None
            panel_polygon = None
            if self.inferencer is not None:
                try:
                    panel_polygon = self.inferencer._find_panel_polygon(otsu_closing, otsu_bounds)
                except Exception as poly_err:
                    logger.warning(f"[DEBUG corner] panel_polygon 偵測失敗，fallback 用矩形: {poly_err}")

            # ROI 擷取（對齊 capi_inference 切法）
            rx1 = max(0, roi_x)
            ry1 = max(0, roi_y)
            rx2 = min(img_w, roi_x + tile_size)
            ry2 = min(img_h, roi_y + tile_size)
            roi = image[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                self._send_json({"error": f"ROI 超出影像範圍: ({roi_x},{roi_y}) size={tile_size}"})
                return

            # 用前端傳入參數 override inspector config
            cfg = EdgeInspectionConfig()
            cfg.aoi_threshold = threshold
            cfg.aoi_min_area = min_area
            cfg.aoi_solidity_min = solidity_min
            cfg.aoi_polygon_erode_px = polygon_erode_px
            cfg.aoi_morph_open_kernel = morph_open_kernel
            cfg.aoi_min_max_diff = min_max_diff
            cfg.aoi_line_min_length = line_min_length
            cfg.aoi_line_max_width = line_max_width
            inspector = CVEdgeInspector(cfg)

            defects, roi_stats = inspector.inspect_roi(
                roi, offset_x=rx1, offset_y=ry1,
                otsu_bounds=otsu_bounds,
                boundary_padding=boundary_padding,
                boundary_min_brightness=boundary_min_brightness,
                panel_polygon=panel_polygon,
            )

            # 手動重建 debug 影像（對齊 _inspect_side 完整流程，含 inpaint 填充）
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            roi_h, roi_w = gray.shape[:2]

            fg_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            if panel_polygon is not None:
                # 與 inspect_roi 同步：polygon 優先 (轉 ROI 局部座標 rasterize)
                local_poly = panel_polygon.copy().astype(np.float32)
                local_poly[:, 0] -= rx1
                local_poly[:, 1] -= ry1
                cv2.fillPoly(fg_mask, [local_poly.astype(np.int32)], 255)
                # Polygon 內縮：避開面板邊緣亮帶轉換區
                if polygon_erode_px > 0 and np.any(fg_mask > 0):
                    ek = cv2.getStructuringElement(cv2.MORPH_RECT,
                            (polygon_erode_px * 2 + 1, polygon_erode_px * 2 + 1))
                    fg_mask = cv2.erode(fg_mask, ek)
            else:
                ox1, oy1, ox2, oy2 = otsu_bounds
                lx1 = max(0, ox1 - rx1); ly1 = max(0, oy1 - ry1)
                lx2 = min(roi_w, ox2 - rx1); ly2 = min(roi_h, oy2 - ry1)
                if lx2 > lx1 and ly2 > ly1:
                    fg_mask[ly1:ly2, lx1:lx2] = 255
                # 矩形 bbox 模式才做 boundary_padding 外擴 (polygon 模式跳過)
                if boundary_padding > 0 and np.any(fg_mask > 0):
                    dk = cv2.getStructuringElement(cv2.MORPH_RECT,
                            (boundary_padding * 2 + 1, boundary_padding * 2 + 1))
                    fg_mask_expanded = cv2.dilate(fg_mask, dk, iterations=1)
                    expansion_zone = (fg_mask_expanded > 0) & (fg_mask == 0)
                    expansion_valid = expansion_zone & (gray >= boundary_min_brightness)
                    fg_mask[expansion_valid] = 255

            k = cfg.blur_kernel
            blurred = cv2.GaussianBlur(gray, (k, k), 0)
            blurred_for_bg = inpaint_non_fg_region(blurred, fg_mask)
            mk = clamp_median_kernel(cfg.median_kernel, min(gray.shape[:2]) - 1)
            bg = cv2.medianBlur(blurred_for_bg, mk)
            diff = cv2.absdiff(blurred, bg)
            diff[fg_mask == 0] = 0
            _, mask_bin = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            if morph_open_kernel > 0:
                ko = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_open_kernel, morph_open_kernel))
                mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, ko)

            # Result 圖：純灰階底 + pixel mask 紅色 blend (底圖 / 顏色 / alpha
            # 與 production Defect Highlight 完全一致；fg_mask 覆蓋另有獨立 panel)
            result_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            filtered_mask_local = roi_stats.get('filtered_mask')
            if filtered_mask_local is not None and filtered_mask_local.shape == result_img.shape[:2]:
                vis_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                vis_mask_local = cv2.dilate(filtered_mask_local, vis_kernel, iterations=1)
                mp = vis_mask_local > 0
                if mp.any():
                    result_img[mp] = (
                        result_img[mp].astype(np.float32) * 0.5
                        + np.array([0, 0, 255], dtype=np.float32) * 0.5
                    ).clip(0, 255).astype(np.uint8)

            # fg_mask 疊到 ROI
            fg_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            fg_overlay[fg_mask == 0] = (0, 0, 60)  # 非前景塗暗紅

            diff_colored = cv2.applyColorMap(
                np.clip(diff * 10, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)

            debug_imgs = {
                "roi": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                "fg_mask": fg_overlay,
                "background": cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR),
                "diff": diff_colored,
                "mask": cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR),
                "result": result_img,
            }

            encoded_imgs = {}
            for kname, img in debug_imgs.items():
                _, buffer = cv2.imencode('.png', img)
                encoded_imgs[kname] = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

            fg_coverage_pct = float(np.count_nonzero(fg_mask) / fg_mask.size * 100) if fg_mask.size > 0 else 0.0

            self._send_json({
                "success": True,
                "defects": [
                    {
                        "area": d.area,
                        "bbox": d.bbox,
                        "center": d.center,
                        "max_diff": d.max_diff,
                        "solidity": float(d.solidity),
                    } for d in defects
                ],
                "stats": {
                    "max_diff": roi_stats.get("max_diff", 0),
                    "max_area": roi_stats.get("max_area", 0),
                    "threshold": roi_stats.get("threshold", threshold),
                    "min_area": roi_stats.get("min_area", min_area),
                    "solidity_min": solidity_min,
                    "morph_open_kernel": morph_open_kernel,
                    "min_max_diff": min_max_diff,
                    "line_min_length": line_min_length,
                    "line_max_width": line_max_width,
                    "fg_coverage_pct": fg_coverage_pct,
                },
                "images": encoded_imgs,
                "otsu_bounds": otsu_bounds,
                "panel_polygon": panel_polygon.astype(int).tolist() if panel_polygon is not None else None,
                "roi_bbox": [rx1, ry1, rx2, ry2],
                "image_size": [img_w, img_h],
            })

        except Exception as e:
            logger.error(f"[DEBUG] Edge Inspect Corner error: {e}", exc_info=True)
            self._send_json({"error": f"角落邊緣檢測失敗: {str(e)}"})

    def _handle_api_debug_edge_corner_patchcore(self, data: dict, image_path: Path):
        """API: 角落 PatchCore 推論 (走 production _inspect_roi_patchcore)"""
        import cv2
        import numpy as np
        import base64
        from capi_heatmap import ensure_bgr, render_pc_masked_roi, render_pc_overlay

        if self.inferencer is None:
            self._send_json({"error": "推論器尚未載入 (inferencer is None)"})
            return

        try:
            roi_x = int(data.get("roi_x", 0))
            roi_y = int(data.get("roi_y", 0))
            tile_size = int(data.get("tile_size", self.inferencer.config.tile_size))

            image = self._read_inference_image(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                self._send_json({"error": "無法讀取圖片"})
                return
            if image.dtype == np.uint16:
                image = (image / 256).astype(np.uint8)

            img_h, img_w = image.shape[:2]

            # UI 的 ROI X/Y 指左上角，轉回中心供 _inspect_roi_patchcore
            img_cx = roi_x + tile_size // 2
            img_cy = roi_y + tile_size // 2

            panel_polygon = None
            try:
                _, _, panel_polygon = self.inferencer.calculate_otsu_bounds(image)
            except Exception as poly_err:
                logger.warning(f"[DEBUG corner PC] polygon 偵測失敗: {poly_err}")

            img_prefix = self.inferencer._get_image_prefix(image_path.name)

            defects, stats = self.inferencer._inspect_roi_patchcore(
                image, img_cx, img_cy, img_prefix,
                panel_polygon=panel_polygon,
            )

            roi = stats.get("roi")
            fg_mask = stats.get("fg_mask")
            anomaly_map = stats.get("anomaly_map")

            def _to_data_url(img: np.ndarray) -> str:
                _, buf = cv2.imencode('.png', img)
                return f"data:image/png;base64,{base64.b64encode(buf).decode('utf-8')}"

            encoded_imgs = {}
            if roi is not None:
                roi_bgr = ensure_bgr(roi)
                encoded_imgs["roi"] = _to_data_url(render_pc_masked_roi(roi_bgr, fg_mask))
                encoded_imgs["heatmap"] = _to_data_url(render_pc_overlay(roi_bgr, fg_mask, anomaly_map))
            if fg_mask is not None:
                encoded_imgs["fg_mask"] = _to_data_url(cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR))

            self._send_json({
                "success": True,
                "inspector": "patchcore",
                "defects": [
                    {
                        "area": d.area,
                        "bbox": list(d.bbox),
                        "center": list(d.center),
                        "patchcore_score": getattr(d, 'patchcore_score', 0.0),
                        "patchcore_threshold": getattr(d, 'patchcore_threshold', 0.0),
                    } for d in defects
                ],
                "stats": {
                    "score": float(stats.get("score", 0.0)),
                    "threshold": float(stats.get("threshold", 0.0)),
                    "area": int(stats.get("area", 0)),
                    "min_area": int(stats.get("min_area", 0)),
                    "ok_reason": str(stats.get("ok_reason", "")),
                    "prefix": img_prefix,
                },
                "images": encoded_imgs,
                "panel_polygon": panel_polygon.astype(int).tolist() if panel_polygon is not None else None,
                "image_size": [img_w, img_h],
            })
        except Exception as e:
            logger.error(f"[DEBUG] Edge Inspect Corner (PatchCore) error: {e}", exc_info=True)
            self._send_json({"error": f"PatchCore 角落檢測失敗: {str(e)}"})

    def _handle_api_debug_edge_corner_fusion(self, data: dict, image_path: Path):
        """API: 角落 Fusion 推論 (走 production _inspect_roi_fusion)"""
        import cv2
        import numpy as np
        import base64
        from capi_heatmap import ensure_bgr, render_pc_masked_roi, render_pc_overlay, HeatmapManager

        if self.inferencer is None:
            self._send_json({"error": "推論器尚未載入 (inferencer is None)"})
            return

        try:
            roi_x = int(data.get("roi_x", 0))
            roi_y = int(data.get("roi_y", 0))
            tile_size = int(data.get("tile_size", self.inferencer.config.tile_size))
            band_px = int(data.get("boundary_band_px",
                                    getattr(self.inferencer.edge_inspector.config,
                                            "aoi_edge_boundary_band_px", 40)))
            # Phase 7: shift override（debug 頁可獨立切換，預設沿用 config）
            cfg = self.inferencer.edge_inspector.config
            shift_enabled_override = data.get("pc_roi_inward_shift_enabled", None)
            aoi_margin_override = data.get("aoi_margin_px", None)

            image = self._read_inference_image(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                self._send_json({"error": "無法讀取圖片"})
                return
            if image.dtype == np.uint16:
                image = (image / 256).astype(np.uint8)

            # 自動找同目錄的 OMIT 圖（OMIT0000* 或 PINIGBI*）
            omit_image = None
            try:
                for f in image_path.parent.iterdir():
                    if f.is_file() and (f.stem.startswith("PINIGBI") or "OMIT0000" in f.name):
                        _omit_raw = self._read_inference_image(f, cv2.IMREAD_UNCHANGED)
                        if _omit_raw is not None:
                            omit_image = (_omit_raw / 256).astype(np.uint8) if _omit_raw.dtype == np.uint16 else _omit_raw
                            break
            except Exception as omit_err:
                logger.warning(f"[DEBUG Fusion] OMIT 自動偵測失敗: {omit_err}")

            img_h, img_w = image.shape[:2]
            img_cx = roi_x + tile_size // 2
            img_cy = roi_y + tile_size // 2

            panel_polygon = None
            try:
                _, _, panel_polygon = self.inferencer.calculate_otsu_bounds(image)
            except Exception as poly_err:
                logger.warning(f"[DEBUG corner Fusion] polygon 偵測失敗: {poly_err}")

            img_prefix = self.inferencer._get_image_prefix(image_path.name)

            # Override config on inspector for this single test
            orig_band_px = cfg.aoi_edge_boundary_band_px
            orig_shift_enabled = getattr(cfg, "aoi_edge_pc_roi_inward_shift_enabled", True)
            orig_aoi_margin = getattr(cfg, "aoi_edge_aoi_margin_px", 64)
            cfg.aoi_edge_boundary_band_px = band_px
            if shift_enabled_override is not None:
                cfg.aoi_edge_pc_roi_inward_shift_enabled = bool(shift_enabled_override)
            if aoi_margin_override is not None:
                cfg.aoi_edge_aoi_margin_px = int(aoi_margin_override)
            try:
                defects, stats = self.inferencer._inspect_roi_fusion(
                    image, img_cx, img_cy, img_prefix,
                    panel_polygon=panel_polygon,
                    omit_image=omit_image,
                    omit_overexposed=False,
                    collapse_to_representative=False,
                )
            finally:
                cfg.aoi_edge_boundary_band_px = orig_band_px
                cfg.aoi_edge_pc_roi_inward_shift_enabled = orig_shift_enabled
                cfg.aoi_edge_aoi_margin_px = orig_aoi_margin

            band_mask = stats.get("band_mask")
            interior_mask = stats.get("interior_mask")
            pc_anomaly_map = stats.get("pc_anomaly_map")
            pc_anomaly_map_interior = stats.get("pc_anomaly_map_interior")
            pc_stats = stats.get("pc_stats", {})
            cv_stats = stats.get("cv_stats", {})
            pc_roi = pc_stats.get("roi")
            pc_fg_mask = pc_stats.get("fg_mask")

            def _to_data_url(img: np.ndarray) -> str:
                _, buf = cv2.imencode('.png', img)
                return f"data:image/png;base64,{base64.b64encode(buf).decode('utf-8')}"

            encoded_imgs = {}
            if pc_roi is not None:
                roi_bgr = ensure_bgr(pc_roi)
                encoded_imgs["roi"] = _to_data_url(render_pc_masked_roi(roi_bgr, pc_fg_mask))
                encoded_imgs["heatmap_full"] = _to_data_url(
                    render_pc_overlay(roi_bgr, pc_fg_mask, pc_anomaly_map))
                encoded_imgs["heatmap_interior"] = _to_data_url(
                    render_pc_overlay(roi_bgr, pc_fg_mask, pc_anomaly_map_interior))
            if band_mask is not None:
                encoded_imgs["band_mask"] = _to_data_url(
                    cv2.cvtColor(band_mask, cv2.COLOR_GRAY2BGR))
            if interior_mask is not None:
                encoded_imgs["interior_mask"] = _to_data_url(
                    cv2.cvtColor(interior_mask, cv2.COLOR_GRAY2BGR))

            # 產生 production 同款 composite 圖（全部 defect 各自一張，collapse 已移除）
            if defects:
                hm = HeatmapManager(base_dir=".", save_format="png")
                ecfg = self.inferencer.edge_inspector.config
                dust_fn = getattr(self.inferencer, 'check_dust_or_scratch_feature', None)
                per_region_fn = getattr(self.inferencer, 'check_dust_per_region', None)
                dust_debug_fn = getattr(self.inferencer, 'generate_dust_iou_debug_image', None)
                top_pct = getattr(self.inferencer.config, 'dust_heatmap_top_percent', 5.0)
                iou_thr = getattr(self.inferencer.config, 'dust_heatmap_iou_threshold', 0.3)
                dust_metric = getattr(self.inferencer.config, 'dust_heatmap_metric', 'coverage')
                composites = []
                for ei, d in enumerate(defects):
                    try:
                        src = getattr(d, 'source_inspector', '')
                        if src == 'patchcore':
                            arr = hm._save_patchcore_edge_image(
                                None, "debug", ei, d, image,
                                omit_image=omit_image, dust_check_fn=dust_fn,
                                edge_config=ecfg, return_array=True,
                                check_dust_per_region_fn=per_region_fn,
                                generate_dust_debug_fn=dust_debug_fn,
                                dust_heatmap_top_percent=top_pct,
                                dust_iou_threshold=iou_thr,
                                dust_metric=dust_metric,
                            )
                        else:
                            arr = hm._save_cv_fusion_edge_image(
                                None, "debug", ei, d, image,
                                omit_image=omit_image, edge_config=ecfg,
                                dust_check_fn=dust_fn,
                                panel_polygon=panel_polygon,
                                return_array=True,
                            )
                        if arr is not None:
                            composites.append(_to_data_url(arr))
                    except Exception as comp_err:
                        logger.warning(f"[DEBUG Fusion] composite 圖 #{ei} 生成失敗: {comp_err}")
                if composites:
                    encoded_imgs["composite"] = composites[0]       # 向後相容
                    encoded_imgs["composites"] = composites         # 全部

            cv_kept = [d for d in defects if getattr(d, 'source_inspector', '') == 'cv']
            pc_kept = [d for d in defects if getattr(d, 'source_inspector', '') == 'patchcore']

            self._send_json({
                "success": True,
                "inspector": "fusion",
                "boundary_band_px": band_px,
                "fusion_fallback_reason": stats.get("fusion_fallback_reason", ""),
                # Phase 7: shift info
                "pc_roi_origin": list(stats.get("pc_roi_origin", (0, 0))),
                "pc_roi_shift": list(stats.get("pc_roi_shift", (0, 0))),
                "pc_roi_fallback_reason": stats.get("pc_roi_fallback_reason", ""),
                "defects": [
                    {
                        "area": d.area,
                        "bbox": list(d.bbox),
                        "center": list(d.center),
                        "source_inspector": getattr(d, 'source_inspector', ''),
                        "d_edge_px": getattr(d, 'd_edge_px', 0.0),
                        "max_diff": d.max_diff,
                        "patchcore_score": getattr(d, 'patchcore_score', 0.0),
                        "patchcore_threshold": getattr(d, 'patchcore_threshold', 0.0),
                        "is_dust": bool(getattr(d, 'is_suspected_dust_or_scratch', False)),
                        "dust_detail_text": getattr(d, 'dust_detail_text', ''),
                        # Phase 7
                        "pc_roi_origin_x": int(getattr(d, 'pc_roi_origin_x', 0)),
                        "pc_roi_origin_y": int(getattr(d, 'pc_roi_origin_y', 0)),
                        "pc_roi_shift_dx": int(getattr(d, 'pc_roi_shift_dx', 0)),
                        "pc_roi_shift_dy": int(getattr(d, 'pc_roi_shift_dy', 0)),
                        "pc_roi_fallback_reason": str(getattr(d, 'pc_roi_fallback_reason', '')),
                    } for d in defects
                ],
                "summary": {
                    "cv_band_count": stats.get("cv_band_count", 0),  # pre-collapse 真實數量
                    "pc_interior_count": stats.get("pc_interior_count", 0),
                    "ng_count": sum(1 for d in defects
                                     if not getattr(d, 'is_suspected_dust_or_scratch', False)),
                    "dust_count": sum(1 for d in defects
                                       if getattr(d, 'is_suspected_dust_or_scratch', False)),
                },
                # CV band 前置診斷
                "cv_band_debug": {
                    "total_found": len(stats.get("cv_defects_all_debug", [])),
                    "kept_in_band": stats.get("cv_band_count", 0),
                    "band_mask_pixels": stats.get("band_mask_pixels", 0),
                    "fg_mask_pixels": stats.get("fg_mask_pixels", 0),
                    "defects": stats.get("cv_defects_all_debug", []),
                    "aoi_threshold": getattr(self.inferencer.edge_inspector.config, "aoi_threshold", "?"),
                    "aoi_min_area": getattr(self.inferencer.edge_inspector.config, "aoi_min_area", "?"),
                    "aoi_min_max_diff": getattr(self.inferencer.edge_inspector.config, "aoi_min_max_diff", "?"),
                    "polygon_erode_px": getattr(self.inferencer.edge_inspector.config, "aoi_polygon_erode_px", "?"),
                },
                "images": encoded_imgs,
                "panel_polygon": panel_polygon.astype(int).tolist() if panel_polygon is not None else None,
                "image_size": [img_w, img_h],
            })
        except Exception as e:
            logger.error(f"[DEBUG] Edge Inspect Corner (Fusion) error: {e}", exc_info=True)
            self._send_json({"error": f"Fusion 角落檢測失敗: {str(e)}"})

    def _run_debug_coord_dust_pipeline(
        self,
        *,
        tile_info,
        tile_image,
        anomaly_map,
        score,
        score_threshold,
        omit_image,
        omit_crop,
        product_resolution,
        model_id,
    ):
        """
        Reproduce the production OMIT/per-region/two-stage path for coordinate debug.

        The returned payload is diagnostic only.  It never changes model/config
        settings and keeps the raw PatchCore threshold judgment separately.
        """
        import numpy as np

        inferencer = self.inferencer
        config = inferencer.config
        anomaly_map_for_dust = anomaly_map
        dust_mask = None
        detail_text = ""
        region_details = []
        heatmap_binary = None
        region_labels = None
        two_stage_features = []
        two_stage_ran = False
        final_is_dust = False
        overall_metric = 0.0
        exclude_zone_masked = False
        seed_yx = None
        seed_radius = 0
        seed_min_score = None

        metric_mode = str(
            getattr(config, "dust_heatmap_metric", "coverage") or "coverage"
        ).lower()
        metric_name = "COV" if metric_mode == "coverage" else "IOU"
        top_percent = float(
            getattr(config, "dust_heatmap_top_percent", 5.0) or 5.0
        )
        overlap_threshold_raw = getattr(
            config, "dust_heatmap_iou_threshold", 0.01
        )
        overlap_threshold = float(
            0.01 if overlap_threshold_raw is None else overlap_threshold_raw
        )
        center_seed_enabled = bool(
            getattr(config, "aoi_heatmap_center_seed_enabled", True)
        )

        payload = {
            "available": omit_image is not None and omit_crop is not None,
            "diagnostic_only": True,
            "metric": metric_mode,
            "metric_label": metric_name,
            "top_percent": top_percent,
            "overlap_threshold": overlap_threshold,
            "omit_overexposed": False,
            "dust_feature_detected": False,
            "dust_filter_result": "NOT_CHECKED",
            "dust_filter_result_zh": "尚未執行灰塵檢查",
            "overall_metric": 0.0,
            "regions": [],
            "center_seed": {
                "enabled": center_seed_enabled,
                "applied": False,
                "map_coord": None,
                "radius": 0,
                "min_score": None,
                "explanation_zh": (
                    "現場已開啟 AOI 中心弱熱區保留。"
                    if center_seed_enabled
                    else "現場設定為關閉；本次不會強制把座標中心塞回 Top % 熱區。"
                ),
            },
            "exclude_zone_heatmap_masked": False,
            "two_stage": _coord_debug_two_stage_payload(
                [], "", tile_info=tile_info
            ),
            "detail_raw": "",
        }

        raw_judgment = "NG" if float(score) >= float(score_threshold) else "OK"
        payload["raw_patchcore_judgment"] = raw_judgment

        def _finish(final_reason_zh):
            tile_info.dust_detail_text = detail_text
            # Below-threshold tiles remain AI OK.  Dust evidence is still exposed
            # in the diagnostic payload but must not be presented as a live filter.
            tile_info.is_suspected_dust_or_scratch = bool(
                final_is_dust and raw_judgment == "NG"
            )
            if raw_judgment == "OK":
                final_judgment = "OK"
                final_reason = "PatchCore 分數低於門檻；灰塵分析僅供排查。"
            elif final_is_dust:
                final_judgment = "OK"
                final_reason = "PatchCore 初判 NG，但正式灰塵流程會過濾為 OK。"
            else:
                final_judgment = "NG"
                final_reason = final_reason_zh
            payload.update({
                "final_judgment": final_judgment,
                "final_reason_zh": final_reason,
                "detail_raw": detail_text,
                "overall_metric": round(float(overall_metric), 6),
                "exclude_zone_heatmap_masked": bool(exclude_zone_masked),
                "regions": _coord_debug_region_payload(
                    region_details,
                    anomaly_map_for_dust,
                    tile_score=float(score),
                    tile_info=tile_info,
                ),
                "two_stage": _coord_debug_two_stage_payload(
                    two_stage_features,
                    detail_text,
                    tile_info=tile_info,
                ),
            })
            return payload, anomaly_map_for_dust, dust_mask

        if omit_image is None or omit_crop is None:
            detail_text = "找不到 OMIT 圖片，無法驗證灰塵／氣泡"
            payload["dust_filter_result"] = "NO_OMIT"
            payload["dust_filter_result_zh"] = "沒有 OMIT，無法排除灰塵／氣泡"
            return _finish("沒有 OMIT 可交叉驗證，保守維持 PatchCore NG。")

        overexposure_check = getattr(inferencer, "check_omit_overexposure", None)
        if callable(overexposure_check):
            try:
                is_overexposed, _mean, _ratio, overexposure_detail = \
                    overexposure_check(omit_image)
            except Exception as exc:
                logger.warning("[DEBUG-COORD] OMIT overexposure check failed: %s", exc)
                is_overexposed = False
                overexposure_detail = ""
            if is_overexposed:
                payload["omit_overexposed"] = True
                payload["dust_filter_result"] = "OMIT_OVEREXPOSED"
                payload["dust_filter_result_zh"] = "OMIT 過曝，灰塵檢查停用"
                detail_text = f"OMIT_OVEREXPOSED ({overexposure_detail})"
                return _finish("OMIT 過曝，無法可靠排除灰塵，保守維持 NG。")

        context_check = getattr(
            inferencer, "_check_dust_or_scratch_feature_with_context", None
        )
        focus_x = int(getattr(tile_info, "aoi_image_x", -1)) - int(
            getattr(tile_info, "x", 0)
        )

        def _detect_dust(extension_override=None):
            if callable(context_check):
                return context_check(
                    omit_image,
                    int(getattr(tile_info, "x", 0)),
                    int(getattr(tile_info, "y", 0)),
                    int(getattr(tile_info, "width", omit_crop.shape[1])),
                    int(getattr(tile_info, "height", omit_crop.shape[0])),
                    omit_crop,
                    extension_override=extension_override,
                    focus_x=focus_x,
                    product_resolution=product_resolution,
                )
            basic_check = inferencer.check_dust_or_scratch_feature
            try:
                return basic_check(
                    omit_crop,
                    extension_override,
                    product_resolution=product_resolution,
                )
            except TypeError:
                return basic_check(omit_crop, extension_override)

        is_dust_feature, dust_mask, bright_ratio, detail_text = _detect_dust()
        tile_info.omit_crop_image = omit_crop.copy()
        tile_info.dust_mask = dust_mask
        tile_info.dust_bright_ratio = float(bright_ratio or 0.0)
        payload["dust_feature_detected"] = bool(is_dust_feature)
        payload["bright_ratio"] = round(float(bright_ratio or 0.0), 6)

        if not is_dust_feature:
            detail_text = f"{detail_text} NO_DUST -> REAL_NG".strip()
            payload["dust_filter_result"] = "NO_DUST"
            payload["dust_filter_result_zh"] = "OMIT 未找到灰塵／氣泡，保留為真缺陷"
            return _finish("OMIT 未找到可解釋此熱區的灰塵／氣泡，維持 NG。")

        if anomaly_map is None:
            final_is_dust = True
            detail_text = f"{detail_text} (no heatmap, marked as dust)".strip()
            payload["dust_filter_result"] = "DUST"
            payload["dust_filter_result_zh"] = "找到灰塵，但沒有熱力圖可逐區比對"
            return _finish("沒有熱力圖可交叉驗證，依現行規則視為灰塵。")

        mask_exclude = getattr(
            inferencer, "_mask_aoi_exclude_zones_for_dust", None
        )
        if callable(mask_exclude):
            anomaly_map_for_dust, exclude_zone_masked = mask_exclude(
                tile_info, anomaly_map, model_id
            )
        if exclude_zone_masked:
            detail_text = f"{detail_text} EXCLUDE_ZONE_HEATMAP_ZEROED".strip()

        center_seed = getattr(inferencer, "_aoi_center_seed_for_tile", None)
        if callable(center_seed):
            seed_yx, seed_radius, seed_min_score = center_seed(
                tile_info, anomaly_map_for_dust
            )
        if seed_yx is not None:
            payload["center_seed"].update({
                "applied": True,
                "map_coord": [int(seed_yx[1]), int(seed_yx[0])],
                "radius": int(seed_radius),
                "min_score": (
                    round(float(seed_min_score), 6)
                    if seed_min_score is not None else None
                ),
                "explanation_zh": (
                    "本次已把指定座標附近、且達最低熱力的像素額外納入 Top %。"
                ),
            })

        has_real, real_peak_yx, overall_metric, region_details, \
            heatmap_binary, region_labels = inferencer.check_dust_per_region(
                dust_mask,
                anomaly_map_for_dust,
                top_percent=top_percent,
                metric=metric_mode,
                iou_threshold=overlap_threshold,
                force_include_yx=seed_yx,
                force_include_radius=seed_radius,
                force_include_min_score=seed_min_score,
            )
        tile_info.dust_heatmap_iou = float(overall_metric)
        tile_info.dust_region_details = region_details
        tile_info.dust_heatmap_binary = heatmap_binary
        if region_details:
            tile_info.dust_region_max_cov = max(
                float(region.get("coverage") or 0.0)
                for region in region_details
            )
        dust_regions = [
            region for region in region_details if region.get("is_dust")
        ]
        real_regions = [
            region for region in region_details if not region.get("is_dust")
        ]

        if has_real:
            final_is_dust = False
            detail_text += (
                f" PER_REGION: {len(real_regions)}real+"
                f"{len(dust_regions)}dust -> REAL_NG"
            )
            payload["dust_filter_result"] = "REAL_NG"
            payload["dust_filter_result_zh"] = "Top % 內仍有非灰塵熱區，保留 NG"
            if real_peak_yx is not None:
                map_height, map_width = np.asarray(
                    anomaly_map_for_dust
                ).shape[:2]
                tile_info.anomaly_peak_y = int(getattr(tile_info, "y", 0)) + int(
                    int(real_peak_yx[0])
                    * int(getattr(tile_info, "height", map_height))
                    / max(map_height, 1)
                )
                tile_info.anomaly_peak_x = int(getattr(tile_info, "x", 0)) + int(
                    int(real_peak_yx[1])
                    * int(getattr(tile_info, "width", map_width))
                    / max(map_width, 1)
                )
        elif bool(getattr(config, "dust_two_stage_enabled", False)):
            _precise_is_dust, precise_dust_mask, _precise_ratio, _precise_detail = \
                _detect_dust(extension_override=0)
            if precise_dust_mask is None:
                precise_dust_mask = dust_mask
            ts_has_real, ts_peak_yx, two_stage_features, ts_detail = \
                inferencer.check_dust_two_stage(
                    tile_image,
                    anomaly_map_for_dust,
                    precise_dust_mask,
                    float(score),
                    score_threshold=float(score_threshold),
                    candidate_dust_mask=dust_mask,
                )
            two_stage_ran = True
            tile_info.dust_two_stage_features = two_stage_features
            tile_info.dust_two_stage_dust_mask = precise_dust_mask
            detail_text += (
                f" PER_REGION: 0real+{len(dust_regions)}dust -> {ts_detail}"
            )
            if ts_has_real:
                final_is_dust = False
                payload["dust_filter_result"] = "REAL_NG"
                payload["dust_filter_result_zh"] = "二階段找到非灰塵特徵，保留 NG"
                if ts_peak_yx is not None:
                    map_height, map_width = np.asarray(
                        anomaly_map_for_dust
                    ).shape[:2]
                    tile_info.anomaly_peak_y = int(
                        getattr(tile_info, "y", 0)
                    ) + int(
                        int(ts_peak_yx[0])
                        * int(getattr(tile_info, "height", map_height))
                        / max(map_height, 1)
                    )
                    tile_info.anomaly_peak_x = int(
                        getattr(tile_info, "x", 0)
                    ) + int(
                        int(ts_peak_yx[1])
                        * int(getattr(tile_info, "width", map_width))
                        / max(map_width, 1)
                    )
            else:
                final_is_dust = True
                payload["dust_filter_result"] = "DUST"
                payload["dust_filter_result_zh"] = "二階段未找到可保留的真缺陷特徵"
        else:
            final_is_dust = True
            detail_text += (
                f" PER_REGION: 0real+{len(dust_regions)}dust -> DUST"
            )
            payload["dust_filter_result"] = "DUST"
            payload["dust_filter_result_zh"] = "Top % 熱區全部可由灰塵／氣泡解釋"

        try:
            if two_stage_ran:
                tile_info.dust_iou_debug_image = \
                    inferencer.generate_two_stage_debug_image(
                        tile_image,
                        anomaly_map_for_dust,
                        tile_info.dust_two_stage_dust_mask,
                        two_stage_features,
                        final_is_dust,
                    )
            else:
                tile_info.dust_iou_debug_image = \
                    inferencer.generate_dust_iou_debug_image(
                        tile_image,
                        anomaly_map_for_dust,
                        dust_mask,
                        heatmap_binary,
                        overall_metric,
                        top_percent,
                        final_is_dust,
                        region_details=region_details,
                        region_labels=region_labels,
                    )
        except Exception as exc:
            logger.warning("[DEBUG-COORD] Dust debug image failed: %s", exc)

        if final_is_dust:
            return _finish(
                "Top % 與二階段證據都可由灰塵／氣泡解釋，正式流程會過濾。"
            )
        return _finish("灰塵流程仍找到非灰塵異常證據，維持 NG。")

    def _resolve_debug_coord_mark_exclusion(self, image_path: Path):
        """Locate the panel MARK for coordinate debug without running OCR side effects."""
        status = {
            "applied": False,
            "source_image": "",
            "bbox": None,
            "region_count": 0,
            "reason_zh": "同資料夾找不到 W0F0000 MARK 來源圖",
        }

        try:
            prepare_files = getattr(
                self.inferencer, "_prepare_panel_image_files", None
            )
            detect_mark = getattr(
                self.inferencer, "_detect_panel_mark_binary_region", None
            )
            if not callable(prepare_files) or not callable(detect_mark):
                return [], status

            image_files, _is_duplicate = prepare_files(image_path.parent)
            detection, regions = detect_mark(
                image_files,
                apply_recognition=False,
            )
            detection = detection or {}
            status["source_image"] = str(detection.get("source_image") or "")
            if not detection.get("found") or not regions:
                status["reason_zh"] = str(
                    detection.get("message")
                    or detection.get("error")
                    or "MARK 未定位"
                )
                return [], status

            bbox = detection.get("mark_bbox_tuple")
            if bbox is None:
                raise ValueError("MARK 定位結果缺少 bbox")
            status.update({
                "bbox": [int(value) for value in bbox],
                "region_count": len(regions),
                "reason_zh": "MARK 已定位，等待套用不檢測區",
            })
            return list(regions), status
        except Exception as exc:
            logger.warning("[DEBUG-COORD] MARK exclusion failed: %s", exc)
            status["reason_zh"] = f"MARK 不檢測區建立失敗：{exc}"
            return [], status

    def _handle_debug_coord_inference(self):
        """API: 人工座標推論 — 以指定產品座標為中心裁切 512x512 做推論"""
        import time as _time
        import cv2
        import numpy as np
        from capi_inference import TileInfo, score_normalization_diagnostic

        # 讀取 POST body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode('utf-8'))
        except Exception:
            self._send_json({"error": "Invalid JSON body"})
            return

        image_path_str = data.get("image_path", "").strip()
        if not image_path_str:
            self._send_json({"error": "請提供圖片路徑 (image_path)"})
            return

        image_path = Path(image_path_str)
        if not image_path.exists():
            self._send_json({"error": f"檔案不存在: {image_path}"})
            return

        if self.inferencer is None:
            self._send_json({"error": "推論器尚未載入 (inferencer is None)"})
            return

        # 解析參數
        try:
            product_x = int(data.get("product_x", 0))
            product_y = int(data.get("product_y", 0))
            product_w = int(data.get("product_w", 1920))
            product_h = int(data.get("product_h", 1080))
        except (ValueError, TypeError) as e:
            self._send_json({"error": f"座標或解析度參數無效: {e}"})
            return

        debug_threshold = float(data.get("threshold", 0.5))
        edge_margin_raw = data.get("edge_margin_px")
        edge_margin_override = int(edge_margin_raw) if edge_margin_raw is not None else None
        try:
            peak_window_px = max(
                1, min(512, int(data.get("peak_window_px", 50)))
            )
        except (ValueError, TypeError):
            self._send_json({"error": "座標搜尋半徑必須是 1～512 的整數"})
            return

        try:
            total_start = _time.time()

            # 1. 載入圖片
            image = self._read_inference_image(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                self._send_json({"error": f"無法載入圖片: {image_path}"})
                return

            img_h, img_w = image.shape[:2]

            # 座標推論必須與正式推論使用同一組模型訓練前處理。
            # preprocess_after_tiling=False：先處理整張圖再裁切座標 tile；
            # True：先裁切座標 tile，再套用前處理。
            from capi_image_preprocess_lab import (
                apply_preprocess_pipeline,
                normalize_preprocess_pipeline,
            )
            configured_pipeline = normalize_preprocess_pipeline(
                getattr(self.inferencer.config, "image_preprocess_pipeline", [])
            )
            configured_zone_pipelines = getattr(
                self.inferencer.config, "image_preprocess_pipelines", {}
            ) or {}
            preprocess_after_tiling = bool(
                getattr(self.inferencer.config, "preprocess_after_tiling", False)
            )
            is_skip_file = bool(
                getattr(self.inferencer.config, "should_skip_file", lambda _name: False)(
                    image_path.name
                )
            )
            processed_panel_image = image
            preprocess_steps = []
            preprocess_total_ms = 0.0
            if configured_pipeline and not preprocess_after_tiling and not is_skip_file:
                pipeline_result = apply_preprocess_pipeline(image, configured_pipeline)
                processed_panel_image = pipeline_result["image"]
                configured_pipeline = pipeline_result["pipeline"]
                preprocess_steps = pipeline_result["steps"]
                preprocess_total_ms = float(pipeline_result.get("total_elapsed_ms") or 0.0)

            # 2. 計算 raw_bounds (面板在圖片中的實際邊界) 與 otsu_bounds
            raw_bounds, _ = self.inferencer._find_raw_object_bounds(image)
            if raw_bounds is None:
                raw_bounds = (0, 0, img_w, img_h)
            
            otsu_bounds, _, _ = self.inferencer.calculate_otsu_bounds(processed_panel_image)
            if otsu_bounds is None:
                otsu_bounds = raw_bounds

            x_start, y_start, x_end, y_end = raw_bounds

            # 3. 產品座標 → 圖片座標
            scale_x = (x_end - x_start) / product_w if product_w > 0 else 1.0
            scale_y = (y_end - y_start) / product_h if product_h > 0 else 1.0
            img_cx = int(product_x * scale_x + x_start)
            img_cy = int(product_y * scale_y + y_start)

            # 4. 以 (img_cx, img_cy) 為 anchor 裁切 tile，v2 會再依 polygon 往內推
            tile_size = self.inferencer.config.tile_size
            half = tile_size // 2
            centered_crop_x1 = img_cx - half
            centered_crop_y1 = img_cy - half
            crop_x1 = max(0, centered_crop_x1)
            crop_y1 = max(0, centered_crop_y1)
            crop_x2 = crop_x1 + tile_size
            crop_y2 = crop_y1 + tile_size

            # 如果超出右/下邊界，向前推
            if crop_x2 > img_w:
                crop_x2 = img_w
                crop_x1 = max(0, crop_x2 - tile_size)
            if crop_y2 > img_h:
                crop_y2 = img_h
                crop_y1 = max(0, crop_y2 - tile_size)

            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            raw_tile_image = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            tile_source_image = (
                image if preprocess_after_tiling or is_skip_file else processed_panel_image
            )
            tile_image = tile_source_image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            crop_shift_dx = crop_x1 - centered_crop_x1
            crop_shift_dy = crop_y1 - centered_crop_y1

            # 5. 建立 TileInfo + 推論
            tile_info = TileInfo(
                tile_id=0,
                x=crop_x1, y=crop_y1,
                width=crop_w, height=crop_h,
                image=tile_image,
                original_image=raw_tile_image,
                is_aoi_coord_tile=True,
                aoi_product_x=product_x,
                aoi_product_y=product_y,
                aoi_image_x=img_cx,
                aoi_image_y=img_cy,
                aoi_tile_shift_dx=crop_shift_dx,
                aoi_tile_shift_dy=crop_shift_dy,
                score_threshold=debug_threshold,
            )

            # 多模型路由：v1 走 prefix lookup；v2 依 polygon 分類 zone 後走 (lighting, zone) 取模型
            img_prefix = self.inferencer._get_image_prefix(image_path.name)
            is_v2 = getattr(self.inferencer.config, "is_new_architecture", False)

            if is_v2:
                lighting_map = self.inferencer.config.model_mapping.get(img_prefix, {})
                if not isinstance(lighting_map, dict) or "inner" not in lighting_map or "edge" not in lighting_map:
                    self._send_json({"error": f"找不到 {image_path.name} 對應的模型 (新架構: model_mapping 缺 {img_prefix} 的 inner/edge)"})
                    return

                from capi_preprocess import (
                    classify_tile_zone, detect_panel_polygon, PreprocessConfig,
                    resolve_inward_polygon_tile,
                )
                pre_cfg = PreprocessConfig(
                    tile_size=self.inferencer.config.tile_size,
                    tile_stride=getattr(
                        self.inferencer.config,
                        "tile_stride",
                        self.inferencer.config.tile_size,
                    ),
                    otsu_offset=self.inferencer.config.otsu_offset,
                    enable_panel_polygon=self.inferencer.config.enable_panel_polygon,
                    edge_threshold_px=self.inferencer.config.edge_threshold_px,
                    image_preprocess_pipeline=getattr(self.inferencer.config, "image_preprocess_pipeline", []),
                    image_preprocess_pipelines=configured_zone_pipelines,
                )
                _, polygon = detect_panel_polygon(processed_panel_image, pre_cfg)
                if polygon is None and hasattr(self.inferencer, "_rect_polygon_from_bounds"):
                    polygon = self.inferencer._rect_polygon_from_bounds(raw_bounds)
                if polygon is not None:
                    # Use the same polygon-aware product-coordinate mapping as
                    # formal v2 AOI inference before resolving the ROI.  The
                    # old debug path only used raw-bounds linear mapping, so a
                    # contaminated bound could make the diagnostic inspect a
                    # different tile than production.
                    img_cx, img_cy = self.inferencer._map_aoi_coords(
                        product_x,
                        product_y,
                        raw_bounds,
                        (product_w, product_h),
                        panel_polygon=polygon,
                    )
                    centered_crop_x1 = img_cx - half
                    centered_crop_y1 = img_cy - half
                    crop_x1 = max(0, centered_crop_x1)
                    crop_y1 = max(0, centered_crop_y1)
                    crop_x2 = crop_x1 + tile_size
                    crop_y2 = crop_y1 + tile_size
                    if crop_x2 > img_w:
                        crop_x2 = img_w
                        crop_x1 = max(0, crop_x2 - tile_size)
                    if crop_y2 > img_h:
                        crop_y2 = img_h
                        crop_y1 = max(0, crop_y2 - tile_size)
                    shift_axes = self.inferencer._resolve_aoi_inward_shift_axes(
                        img_cx, img_cy, raw_bounds, tile_size,
                    )
                    crop_x1, crop_y1, _cov, _shifted = resolve_inward_polygon_tile(
                        anchor_xy=(img_cx, img_cy),
                        polygon=polygon,
                        image_shape=(img_h, img_w),
                        tile_size=tile_size,
                        initial_origin=(crop_x1, crop_y1),
                        keep_anchor_inside=True,
                        shift_axes=shift_axes,
                    )
                    crop_x2 = min(img_w, crop_x1 + tile_size)
                    crop_y2 = min(img_h, crop_y1 + tile_size)
                    crop_w = crop_x2 - crop_x1
                    crop_h = crop_y2 - crop_y1
                    raw_tile_image = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                    tile_image = tile_source_image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                    crop_shift_dx = crop_x1 - centered_crop_x1
                    crop_shift_dy = crop_y1 - centered_crop_y1
                    tile_info.x = crop_x1
                    tile_info.y = crop_y1
                    tile_info.width = crop_w
                    tile_info.height = crop_h
                    tile_info.image = tile_image
                    tile_info.original_image = raw_tile_image
                    tile_info.aoi_image_x = img_cx
                    tile_info.aoi_image_y = img_cy
                    tile_info.aoi_tile_shift_dx = crop_shift_dx
                    tile_info.aoi_tile_shift_dy = crop_shift_dy

                tile_rect = (crop_x1, crop_y1, crop_x2, crop_y2)
                zone, _, _, tile_mask = classify_tile_zone(tile_rect, polygon, pre_cfg)
                if zone == "outside":
                    zone = "edge"  # debug fallback：邊外座標仍用 edge model 看一下分數
                if polygon is not None:
                    d_edge = float(cv2.pointPolygonTest(
                        np.asarray(polygon, dtype=np.float32),
                        (float(img_cx), float(img_cy)), True,
                    ))
                    if d_edge <= tile_size // 2:
                        zone = "edge"
                tile_info.zone = zone
                tile_info.mask = tile_mask

                try:
                    target_inferencer = self.inferencer._get_model_for(
                        self.inferencer.config.machine_id, img_prefix, zone,
                    )
                except Exception as exc:
                    self._send_json({"error": f"新架構模型載入失敗 ({img_prefix}/{zone}): {exc}"})
                    return

                if target_inferencer is None:
                    self._send_json({"error": f"找不到 {image_path.name} ({img_prefix}/{zone}) 對應的模型"})
                    return

                model_name = f"{img_prefix}/{zone}"
            else:
                target_inferencer = self.inferencer._get_inferencer_for_prefix(img_prefix)
                model_name = "預設模型"
                if img_prefix in self.inferencer._model_mapping:
                    model_name = self.inferencer._model_mapping[img_prefix].name

                if target_inferencer is None:
                    self._send_json({"error": f"找不到 {image_path.name} 對應的模型"})
                    return

            active_pipeline = configured_pipeline
            if is_v2 and configured_zone_pipelines:
                active_pipeline = configured_zone_pipelines.get(zone, [])
            if active_pipeline and preprocess_after_tiling and not is_skip_file:
                pipeline_result = apply_preprocess_pipeline(raw_tile_image, active_pipeline)
                tile_image = pipeline_result["image"]
                configured_pipeline = pipeline_result["pipeline"]
                preprocess_steps = pipeline_result["steps"]
                preprocess_total_ms = float(pipeline_result.get("total_elapsed_ms") or 0.0)
                tile_info.image = tile_image
            elif preprocess_after_tiling:
                configured_pipeline = list(active_pipeline or [])

            mark_regions, mark_exclusion = \
                self._resolve_debug_coord_mark_exclusion(image_path)
            tile_info.mark_exclusion_regions = list(mark_regions)
            tile_info.mark_exclusion_region_count = len(mark_regions)

            # 推論 (含 GPU lock)
            debug_model_id = getattr(self.inferencer.config, "machine_id", None)
            predict_kwargs = {
                "inferencer": target_inferencer,
                "edge_margin_override": edge_margin_override,
                "threshold": debug_threshold,
                "model_id": debug_model_id,
            }
            # Older test doubles / deployed workers may still expose the old
            # predict_tile signature.  Only the real implementation receives
            # the opt-in second pass for raw-score diagnostics.
            try:
                import inspect
                if "capture_raw_diagnostics" in inspect.signature(
                    self.inferencer.predict_tile
                ).parameters:
                    predict_kwargs["capture_raw_diagnostics"] = True
            except (TypeError, ValueError):
                pass
            if hasattr(self, '_gpu_lock') and self._gpu_lock:
                with self._gpu_lock:
                    score, anomaly_map = self.inferencer.predict_tile(
                        tile_info, **predict_kwargs,
                    )
            else:
                score, anomaly_map = self.inferencer.predict_tile(
                    tile_info, **predict_kwargs,
                )

            if mark_regions:
                mark_exclusion["applied"] = bool(
                    getattr(tile_info, "mark_exclusion_masked", False)
                )
                mark_exclusion["reason_zh"] = (
                    "已套用正式 MARK 不檢測區"
                    if mark_exclusion["applied"]
                    else "MARK 已定位，但未影響本次座標 tile"
                )

            total_time = _time.time() - total_start

            # 6. 建立 Debug heatmap 暫存目錄
            if CAPIWebHandler._debug_heatmap_dir is None:
                CAPIWebHandler._debug_heatmap_dir = Path(tempfile.mkdtemp(prefix="capi_debug_hm_"))
            debug_dir = CAPIWebHandler._debug_heatmap_dir
            debug_dir.mkdir(parents=True, exist_ok=True)

            image_name = image_path.stem
            ts = int(_time.time() * 1000) % 100000  # 避免快取

            # 7. 儲存原始裁切圖
            crop_bgr = raw_tile_image.copy()
            if len(crop_bgr.shape) == 2:
                crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
            elif len(crop_bgr.shape) == 3 and crop_bgr.shape[2] == 1:
                crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
            crop_filename = f"debug_coord_crop_{image_name}_{ts}.png"
            cv2.imwrite(str(debug_dir / crop_filename), crop_bgr)
            crop_url = f"/debug/heatmaps/{crop_filename}"

            # 8. 儲存熱力圖
            heatmap_url = ""
            if anomaly_map is not None:
                if self.heatmap_manager:
                    overlay = self.heatmap_manager.generate_heatmap_overlay(
                        tile_image, anomaly_map, alpha=0.5
                    )
                else:
                    norm_map = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
                    if heatmap_color.shape[:2] != crop_bgr.shape[:2]:
                        heatmap_color = cv2.resize(heatmap_color, (crop_bgr.shape[1], crop_bgr.shape[0]))
                    overlay = cv2.addWeighted(crop_bgr, 0.5, heatmap_color, 0.5, 0)
                hm_filename = f"debug_coord_hm_{image_name}_{ts}.png"
                cv2.imwrite(str(debug_dir / hm_filename), overlay)
                heatmap_url = f"/debug/heatmaps/{hm_filename}"

            # 8. 產生 Overview 圖 (加上裁切框)
            overview_img = image.copy()
            if len(overview_img.shape) == 2:
                overview_img = cv2.cvtColor(overview_img, cv2.COLOR_GRAY2BGR)
            elif len(overview_img.shape) == 3 and overview_img.shape[2] == 1:
                overview_img = cv2.cvtColor(overview_img, cv2.COLOR_GRAY2BGR)
            
            # 使用半透明遮罩凸顯區域
            overlay_bg = overview_img.copy()
            cv2.rectangle(overlay_bg, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay_bg, 0.3, overview_img, 0.7, 0, overview_img)
            # 畫 Otsu 範圍 (黃色框)
            ox1, oy1, ox2, oy2 = otsu_bounds
            cv2.rectangle(overview_img, (ox1, oy1), (ox2, oy2), (0, 255, 255), 4)
            # 畫紅框 + 中心點
            cv2.rectangle(overview_img, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 0, 255), 6)
            cv2.circle(overview_img, (img_cx, img_cy), 10, (0, 255, 0), -1)

            # 在前景有效區域 (Otsu bounds 黃框) 左上與右下標上圖片座標
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            font_thickness = 3
            # 左上座標 (置於黃框上方或內部)
            tl_text = f"({ox1}, {oy1})"
            cv2.putText(overview_img, tl_text, (ox1, max(30, oy1 - 10)), font, font_scale, (0, 255, 255), font_thickness)
            # 右下座標 (置於黃框下方或內部)
            br_text = f"({ox2}, {oy2})"
            cv2.putText(overview_img, br_text, (max(0, ox2 - 300), min(img_h - 10, oy2 + 40)), font, font_scale, (0, 255, 255), font_thickness)

            # 縮小 Overview 存檔 (最多 2000px)
            max_dim = 2000
            oh, ow = overview_img.shape[:2]
            if max(oh, ow) > max_dim:
                scale_o = max_dim / max(oh, ow)
                overview_img = cv2.resize(overview_img, (int(ow * scale_o), int(oh * scale_o)))
            ov_filename = f"debug_coord_overview_{image_name}_{ts}.png"
            cv2.imwrite(str(debug_dir / ov_filename), overview_img)
            overview_url = f"/debug/heatmaps/{ov_filename}"

            # 9. 嘗試尋找並裁切 OMIT 圖片
            omit_url = ""
            omit_name = ""
            omit_full = None
            omit_crop = None
            image_dir = image_path.parent
            omit_candidates = []
            for pattern in ["PINIGBI*.*", "OMIT0000*.*"]:
                omit_candidates.extend(
                    candidate
                    for candidate in image_dir.glob(pattern)
                    if not candidate.name.startswith("S")
                )
            if omit_candidates:
                omit_path = sorted(omit_candidates, key=lambda path: path.name)[0]
                omit_name = omit_path.name
                omit_full = self._read_inference_image(omit_path, cv2.IMREAD_UNCHANGED)
                if omit_full is not None:
                    try:
                        omit_crop = omit_full[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                        if len(omit_crop.shape) == 2:
                            omit_crop = cv2.cvtColor(omit_crop, cv2.COLOR_GRAY2BGR)
                        elif len(omit_crop.shape) == 3 and omit_crop.shape[2] == 1:
                            omit_crop = cv2.cvtColor(omit_crop, cv2.COLOR_GRAY2BGR)
                        omit_filename = f"debug_coord_omit_{image_name}_{ts}.png"
                        cv2.imwrite(str(debug_dir / omit_filename), omit_crop)
                        omit_url = f"/debug/heatmaps/{omit_filename}"
                    except Exception as e:
                        logger.warning(f"OMIT crop failed: {e}")

            # 9.5 依正式流程執行逐區域與 two-stage 灰塵診斷
            dust_analysis, anomaly_map_for_dust, diagnostic_dust_mask = \
                self._run_debug_coord_dust_pipeline(
                    tile_info=tile_info,
                    tile_image=tile_image,
                    anomaly_map=anomaly_map,
                    score=score,
                    score_threshold=debug_threshold,
                    omit_image=omit_full,
                    omit_crop=omit_crop,
                    product_resolution=(product_w, product_h),
                    model_id=debug_model_id,
                )
            dust_analysis["omit_name"] = omit_name

            # 9.6 Local-maxima 診斷：不受 Top % 配額限制，專門排查強氣泡搶分
            peak_diagnostic_url = ""
            heatmap_analysis = {
                "available": False,
                "diagnostic_only": True,
                "peaks": [],
                "conclusion_zh": "本次沒有可分析的熱力圖。",
            }
            if anomaly_map is not None:
                try:
                    from capi_heatmap_diagnostics import analyze_heatmap_peaks

                    anomaly_array = np.asarray(anomaly_map, dtype=np.float32)
                    map_h, map_w = anomaly_array.shape[:2]
                    local_aoi_x = min(
                        max(int(img_cx - crop_x1), 0), max(crop_w - 1, 0)
                    )
                    local_aoi_y = min(
                        max(int(img_cy - crop_y1), 0), max(crop_h - 1, 0)
                    )
                    aoi_map_x = int(round(
                        local_aoi_x * max(map_w - 1, 0) / max(crop_w - 1, 1)
                    ))
                    aoi_map_y = int(round(
                        local_aoi_y * max(map_h - 1, 0) / max(crop_h - 1, 1)
                    ))
                    map_scale = min(
                        map_w / max(crop_w, 1),
                        map_h / max(crop_h, 1),
                    )
                    aoi_window_map = max(
                        1, int(round(peak_window_px * map_scale))
                    )
                    heatmap_analysis = analyze_heatmap_peaks(
                        anomaly_array,
                        diagnostic_dust_mask,
                        aoi_xy=(aoi_map_x, aoi_map_y),
                        aoi_window=aoi_window_map,
                        top_percent=float(dust_analysis["top_percent"]),
                        min_distance=max(2, int(round(10 * map_scale))),
                        threshold_rel=0.3,
                        aoi_threshold_rel=0.3,
                        global_score=float(score),
                        # 正常平滑 heatmap 通常只有個位數峰值；安全上限避免
                        # 嚴重雜訊圖讓 region-grow 診斷拖慢整個 Debug API。
                        max_peaks=100,
                    )
                    heatmap_analysis["available"] = True
                    heatmap_analysis["dust_mask_available"] = \
                        diagnostic_dust_mask is not None
                    peak_limit_reached = bool(
                        int(heatmap_analysis.get("global_peak_count") or 0) >= 100
                        or int(heatmap_analysis.get("aoi_peak_count") or 0) >= 100
                    )
                    heatmap_analysis["peak_limit_per_scan"] = 100
                    heatmap_analysis["peak_limit_reached"] = peak_limit_reached
                    heatmap_analysis["all_peaks_returned"] = \
                        not peak_limit_reached
                    heatmap_analysis["aoi_window_tile_px"] = peak_window_px
                    heatmap_analysis["analysis_heatmap_zh"] = (
                        "正式灰塵判定用熱力圖（已套用 MARK 不檢測區）"
                        if getattr(tile_info, "mark_exclusion_masked", False)
                        else (
                            "正式灰塵判定用熱力圖（已套用不檢測區遮罩）"
                            if dust_analysis.get("exclude_zone_heatmap_masked")
                            else "PatchCore 衰減後熱力圖"
                        )
                    )
                    global_stats = heatmap_analysis.get("global_stats") or {}
                    top_info = heatmap_analysis.get("top_percent") or {}
                    heatmap_analysis["statistics"] = {
                        **global_stats,
                        "global_peak": global_stats.get("max"),
                        "top_percent": top_info.get("percent"),
                        "top_cutoff": top_info.get("cutoff"),
                        "retained_pixel_count": top_info.get(
                            "retained_pixel_count", 0
                        ),
                    }

                    effective_top = getattr(
                        tile_info, "dust_heatmap_binary", None
                    )
                    if effective_top is not None:
                        effective_top = np.asarray(effective_top)
                        if effective_top.shape[:2] != (map_h, map_w):
                            effective_top = cv2.resize(
                                effective_top,
                                (map_w, map_h),
                                interpolation=cv2.INTER_NEAREST,
                            )

                    for peak in heatmap_analysis.get("peaks", []):
                        map_x = int(peak.get("x", 0))
                        map_y = int(peak.get("y", 0))
                        peak.update(_coord_debug_point_payload(
                            map_x=map_x,
                            map_y=map_y,
                            map_width=map_w,
                            map_height=map_h,
                            tile_x=crop_x1,
                            tile_y=crop_y1,
                            tile_width=crop_w,
                            tile_height=crop_h,
                        ))
                        peak["peak_value"] = round(
                            float(peak.get("raw_peak") or 0.0), 6
                        )
                        peak["relative_to_global"] = round(
                            float(
                                peak.get("relative_to_global_max") or 0.0
                            ),
                            6,
                        )
                        peak["in_top_percent"] = bool(
                            peak.get("kept_by_top_percent", False)
                        )
                        if effective_top is not None:
                            in_effective_top = bool(
                                0 <= map_y < effective_top.shape[0]
                                and 0 <= map_x < effective_top.shape[1]
                                and effective_top[map_y, map_x] > 0
                            )
                        else:
                            in_effective_top = peak["in_top_percent"]
                        peak["in_effective_top_percent"] = in_effective_top
                        peak["retained_by_center_seed"] = bool(
                            in_effective_top and not peak["in_top_percent"]
                        )
                        peak["local_dust_coverage"] = peak.get(
                            "local_dust_cov_11x11"
                        )
                        peak["region_grow_coverage"] = peak.get(
                            "region_grow_cov"
                        )
                        estimated_score = peak.get("estimated_score")
                        if estimated_score is not None:
                            peak["estimated_score"] = round(
                                float(estimated_score), 6
                            )
                            estimated_score = peak["estimated_score"]
                        peak["passes_score_threshold"] = bool(
                            estimated_score is not None
                            and float(estimated_score) >= debug_threshold
                        )
                        is_aoi_peak = "aoi" in (peak.get("sources") or [])
                        if peak.get("in_dust") is True:
                            verdict_zh = "灰塵／氣泡高分來源"
                        elif is_aoi_peak and not in_effective_top:
                            verdict_zh = "座標附近候選；被 Top % 排除"
                        elif is_aoi_peak:
                            verdict_zh = "座標附近候選；已進入熱區"
                        elif not in_effective_top:
                            verdict_zh = "局部熱點；被 Top % 排除"
                        else:
                            verdict_zh = "正式熱區候選"
                        peak["verdict_zh"] = verdict_zh
                        peak["reason_zh"] = str(
                            peak.get("interpretation_zh") or ""
                        )

                    dominant_peak = heatmap_analysis.get("dominant_peak")
                    aoi_best_peak = heatmap_analysis.get("aoi_best_peak")
                    heatmap_analysis["score_competition_detected"] = bool(
                        dominant_peak
                        and dominant_peak.get("in_dust") is True
                        and aoi_best_peak
                        and not aoi_best_peak.get(
                            "in_effective_top_percent",
                            aoi_best_peak.get("in_top_percent", False),
                        )
                    )

                    # 產生可與表格排名對照的峰值圖；文字說明留在中文 UI。
                    norm_map = cv2.normalize(
                        anomaly_array, None, 0, 255, cv2.NORM_MINMAX
                    ).astype(np.uint8)
                    peak_overlay = cv2.applyColorMap(
                        norm_map, cv2.COLORMAP_JET
                    )
                    peak_overlay = cv2.resize(
                        peak_overlay, (crop_w, crop_h)
                    )
                    for peak in heatmap_analysis.get("peaks", []):
                        px, py = peak["tile_coord"]
                        if peak.get("in_dust") is True:
                            color = (0, 215, 255)
                        elif not peak.get("in_effective_top_percent", False):
                            color = (255, 0, 255)
                        else:
                            color = (0, 0, 255)
                        cv2.circle(peak_overlay, (px, py), 10, color, 2)
                        cv2.putText(
                            peak_overlay,
                            f"#{peak['rank']}",
                            (min(crop_w - 36, px + 12), max(16, py - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            color,
                            1,
                            cv2.LINE_AA,
                        )
                    cv2.circle(
                        peak_overlay,
                        (local_aoi_x, local_aoi_y),
                        min(peak_window_px, max(1, min(crop_w, crop_h) // 2)),
                        (255, 255, 0),
                        1,
                    )
                    cv2.drawMarker(
                        peak_overlay,
                        (local_aoi_x, local_aoi_y),
                        (255, 255, 0),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=18,
                        thickness=2,
                    )
                    peak_filename = (
                        f"debug_coord_peaks_{image_name}_{ts}.png"
                    )
                    cv2.imwrite(str(debug_dir / peak_filename), peak_overlay)
                    peak_diagnostic_url = (
                        f"/debug/heatmaps/{peak_filename}"
                    )
                except Exception as diag_err:
                    logger.warning(
                        "[DEBUG-COORD] Local peak diagnostic failed: %s",
                        diag_err,
                        exc_info=True,
                    )
                    heatmap_analysis = {
                        "available": False,
                        "diagnostic_only": True,
                        "peaks": [],
                        "error_zh": f"局部峰值診斷失敗: {diag_err}",
                        "conclusion_zh": "局部峰值診斷失敗，正式推論分數不受影響。",
                    }

            # 9.7 產生組合圖 (Coordinate 推論專屬)
            composite_url = ""
            if self.heatmap_manager:
                try:
                    composite_path = self.heatmap_manager.save_tile_heatmap(
                        save_dir=debug_dir,
                        image_name=f"coord_{image_name}_{ts}",
                        tile_id=0,
                        tile_image=tile_image,
                        anomaly_map=anomaly_map,
                        score=score,
                        tile_info=tile_info,
                        score_threshold=debug_threshold,
                        iou_threshold=getattr(self.inferencer.config, 'dust_heatmap_iou_threshold', 0.01),
                        dust_metric=getattr(self.inferencer.config, 'dust_heatmap_metric', 'coverage'),
                        dust_high_cov_threshold=getattr(self.inferencer.config, 'dust_high_cov_threshold', None),
                    )
                    composite_filename = Path(composite_path).name
                    composite_url = f"/debug/heatmaps/{composite_filename}"
                except Exception as comp_err:
                    logger.warning(f"[DEBUG-COORD] 組合圖產生失敗: {comp_err}")

            # 10. 判定結果
            actual_threshold = debug_threshold
            judgment = "NG" if score >= actual_threshold else "OK"
            total_time = _time.time() - total_start
            def _finite_tile_diag(name: str) -> Optional[float]:
                value = getattr(tile_info, name, None)
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    return None
                return value if np.isfinite(value) else None

            raw_model_score = _finite_tile_diag("raw_model_score")
            model_image_min = _finite_tile_diag("model_image_min")
            model_image_max = _finite_tile_diag("model_image_max")
            model_image_threshold = _finite_tile_diag("model_image_threshold")
            model_normalization_enabled = getattr(
                tile_info, "model_normalization_enabled", None
            )
            normalization_diag = score_normalization_diagnostic(
                raw_model_score,
                model_image_min,
                model_image_max,
                model_image_threshold,
                normalization_enabled=model_normalization_enabled,
            )
            score_breakdown = {
                "raw_patchcore_score": round(
                    float(getattr(tile_info, "raw_pred_score", score)), 6
                ),
                # raw_patchcore_score is the normalized score retained for
                # backward compatibility; these fields expose the distance
                # before Anomalib normalization so a displayed 0 is explainable.
                "raw_model_score": (
                    round(raw_model_score, 6) if raw_model_score is not None else None
                ),
                "model_image_min": (
                    round(model_image_min, 6) if model_image_min is not None else None
                ),
                "model_image_max": (
                    round(model_image_max, 6) if model_image_max is not None else None
                ),
                "model_image_threshold": (
                    round(model_image_threshold, 6)
                    if model_image_threshold is not None else None
                ),
                "model_normalization_enabled": model_normalization_enabled,
                "raw_anomaly_map_max": (
                    round(_finite_tile_diag("raw_anomaly_map_max"), 6)
                    if _finite_tile_diag("raw_anomaly_map_max") is not None else None
                ),
                "normalized_anomaly_map_max": (
                    round(_finite_tile_diag("normalized_anomaly_map_max"), 6)
                    if _finite_tile_diag("normalized_anomaly_map_max") is not None else None
                ),
                **normalization_diag,
                "pre_decay_map_max": round(
                    float(getattr(tile_info, "pre_decay_map_max", 0.0)), 6
                ),
                "post_decay_map_max": round(
                    float(getattr(tile_info, "post_decay_map_max", 0.0)), 6
                ),
                "decay_ratio": round(
                    float(getattr(tile_info, "score_decay_ratio", 1.0)), 6
                ),
                "final_tile_score": round(float(score), 6),
                "threshold": round(float(actual_threshold), 6),
                "mask_valid_ratio": round(
                    float(getattr(tile_info, "score_mask_valid_ratio", 1.0)), 6
                ),
                "edge_margin_sides": str(
                    getattr(tile_info, "score_edge_margin_sides", "") or ""
                ),
                "mark_exclusion_masked": bool(
                    getattr(tile_info, "mark_exclusion_masked", False)
                ),
                "mark_exclusion_region_count": int(
                    getattr(tile_info, "mark_exclusion_region_count", 0) or 0
                ),
                "mark_patch_score_applied": bool(
                    getattr(tile_info, "mark_patch_score_applied", False)
                ),
                "mark_patchcore_score": round(
                    float(getattr(tile_info, "mark_patchcore_score", 0.0)), 6
                ),
                "mark_patch_valid_count": int(
                    getattr(tile_info, "mark_patch_valid_count", 0) or 0
                ),
                "mark_patch_total_count": int(
                    getattr(tile_info, "mark_patch_total_count", 0) or 0
                ),
                "mark_patch_peak": [
                    int(getattr(tile_info, "mark_patch_peak_x", -1)),
                    int(getattr(tile_info, "mark_patch_peak_y", -1)),
                ],
                "mark_patch_score_reason": str(
                    getattr(tile_info, "mark_patch_score_reason", "") or ""
                ),
            }

            response_data = {
                "success": True,
                "product_coord": [product_x, product_y],
                "product_resolution": [product_w, product_h],
                "image_coord": [img_cx, img_cy],
                "centered_crop_origin": [centered_crop_x1, centered_crop_y1],
                "tile_shift": [crop_shift_dx, crop_shift_dy],
                "crop_region": [crop_x1, crop_y1, crop_x2, crop_y2],
                "raw_bounds": list(raw_bounds),
                "scale": [round(scale_x, 4), round(scale_y, 4)],
                "image_size": [img_w, img_h],
                "score": round(score, 4),
                "threshold": actual_threshold,
                "judgment": judgment,
                "final_judgment": dust_analysis.get("final_judgment", judgment),
                "final_reason_zh": dust_analysis.get("final_reason_zh", ""),
                "processing_time": round(total_time, 3),
                "crop_url": crop_url,
                "heatmap_url": heatmap_url,
                "overview_url": overview_url,
                "omit_url": omit_url,
                "composite_url": composite_url,
                "peak_diagnostic_url": peak_diagnostic_url,
                "image_prefix": img_prefix,
                "image_prefix_label": source_image_prefix(image_path.name),
                "model_name": model_name,
                "mark_exclusion": mark_exclusion,
                "edge_margin_px": edge_margin_override if edge_margin_override is not None else self.inferencer.config.edge_margin_px,
                "peak_window_px": peak_window_px,
                "score_breakdown": score_breakdown,
                "dust_analysis": dust_analysis,
                "heatmap_analysis": heatmap_analysis,
                "preprocess": {
                    "enabled": bool(configured_pipeline),
                    "applied": bool(preprocess_steps),
                    "pipeline": configured_pipeline,
                    "steps": preprocess_steps,
                    "total_time_ms": round(preprocess_total_ms, 3),
                    "after_tiling": preprocess_after_tiling,
                    "scope": "座標 tile（切塊後）" if preprocess_after_tiling else "整張影像（切塊前）",
                    "skipped": is_skip_file,
                },
            }

            self._send_json(response_data)
            logger.info(f"[DEBUG-COORD] ({product_x},{product_y})→({img_cx},{img_cy}) "
                        f"shift=({crop_shift_dx:+d},{crop_shift_dy:+d}) "
                        f"Score={score:.4f} {judgment} ({total_time:.2f}s)")

        except Exception as e:
            logger.error(f"[DEBUG-COORD] Error: {e}", exc_info=True)
            self._send_json({"error": f"座標推論失敗: {str(e)}"})

    def _handle_debug_bright_spot_inference(self):
        """API: 黑畫面亮點偵測 — 以指定產品座標為中心裁切 512x512 做 B0F 偵測"""
        import time as _time
        import cv2
        import numpy as np
        from capi_inference import TileInfo

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode('utf-8'))
        except Exception:
            self._send_json({"error": "Invalid JSON body"})
            return

        image_path_str = data.get("image_path", "").strip()
        if not image_path_str:
            self._send_json({"error": "請提供圖片路徑 (image_path)"})
            return

        image_path = Path(image_path_str)
        if not image_path.exists():
            self._send_json({"error": f"檔案不存在: {image_path}"})
            return

        if self.inferencer is None:
            self._send_json({"error": "推論器尚未載入 (inferencer is None)"})
            return

        # 解析座標參數
        try:
            product_x = int(data.get("product_x", 0))
            product_y = int(data.get("product_y", 0))
            product_w = int(data.get("product_w", 1920))
            product_h = int(data.get("product_h", 1080))
        except (ValueError, TypeError) as e:
            self._send_json({"error": f"座標或解析度參數無效: {e}"})
            return

        # 解析 bright_spot 參數覆蓋
        bs_diff_threshold = int(data.get("bs_diff_threshold", self.inferencer.config.bright_spot_diff_threshold))
        bs_median_kernel = int(data.get("bs_median_kernel", self.inferencer.config.bright_spot_median_kernel))
        bs_min_area = int(data.get("bs_min_area", self.inferencer.config.bright_spot_min_area))
        bs_threshold = int(data.get("bs_threshold", self.inferencer.config.bright_spot_threshold))

        try:
            total_start = _time.time()

            # 1. 載入圖片
            image = self._read_inference_image(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                self._send_json({"error": f"無法載入圖片: {image_path}"})
                return

            img_h, img_w = image.shape[:2]

            # 2. 計算 raw_bounds / otsu_bounds
            #    黑圖 Otsu 無法正確偵測邊界，從同資料夾找白圖計算參考邊界
            #    (與 process_panel 中 reference_raw_bounds_for_dark 邏輯一致)
            reference_bounds = None
            ref_image_name = None
            image_dir = image_path.parent
            _DARK_PREFIXES = ("B0F",)
            is_dark = image_path.name.upper().startswith(_DARK_PREFIXES)
            if is_dark:
                # 找同資料夾的白圖 (W0F00000_ 開頭優先，其次任何非 B0F/OMIT/PINIGBI 圖)
                _IMG_EXTS = ('.bmp', '.tif', '.tiff', '.png', '.jpg', '.jpeg')
                all_files = sorted(image_dir.iterdir())
                # 第一輪：優先找 W0F00000_ 開頭
                for candidate in all_files:
                    if not candidate.is_file() or candidate.suffix.lower() not in _IMG_EXTS:
                        continue
                    if candidate.name.upper().startswith("W0F00000"):
                        try:
                            ref_img = self._read_inference_image(candidate, cv2.IMREAD_UNCHANGED)
                            if ref_img is not None:
                                reference_bounds, _ = self.inferencer._find_raw_object_bounds(ref_img)
                                ref_image_name = candidate.name
                                break
                        except Exception:
                            continue
                # 第二輪 fallback：任何非暗色、非 OMIT 圖片
                if reference_bounds is None:
                    for candidate in all_files:
                        if not candidate.is_file() or candidate.suffix.lower() not in _IMG_EXTS:
                            continue
                        cname = candidate.name.upper()
                        if cname.startswith(_DARK_PREFIXES) or cname.startswith("OMIT0000") or cname.startswith("PINIGBI"):
                            continue
                        try:
                            ref_img = self._read_inference_image(candidate, cv2.IMREAD_UNCHANGED)
                            if ref_img is not None:
                                reference_bounds, _ = self.inferencer._find_raw_object_bounds(ref_img)
                                ref_image_name = candidate.name
                                break
                        except Exception:
                            continue
                if reference_bounds is not None:
                    logger.info(f"[DEBUG-BS] 黑圖參考邊界已從 {ref_image_name} 計算 → {reference_bounds}")

            if reference_bounds is not None:
                raw_bounds = reference_bounds
                otsu_bounds, _, _ = self.inferencer.calculate_otsu_bounds(image, reference_raw_bounds=reference_bounds)
                if otsu_bounds is None:
                    otsu_bounds = raw_bounds
            else:
                raw_bounds, _ = self.inferencer._find_raw_object_bounds(image)
                if raw_bounds is None:
                    raw_bounds = (0, 0, img_w, img_h)
                otsu_bounds, _, _ = self.inferencer.calculate_otsu_bounds(image)
                if otsu_bounds is None:
                    otsu_bounds = raw_bounds
                if is_dark:
                    logger.warning(f"[DEBUG-BS] 無法找到白圖計算參考邊界，使用自身 Otsu 邊界 (可能不準確)")

            x_start, y_start, x_end, y_end = raw_bounds

            # 3. 產品座標 → 圖片座標
            scale_x = (x_end - x_start) / product_w if product_w > 0 else 1.0
            scale_y = (y_end - y_start) / product_h if product_h > 0 else 1.0
            img_cx = int(product_x * scale_x + x_start)
            img_cy = int(product_y * scale_y + y_start)

            # 4. 以 (img_cx, img_cy) 為中心裁切 512x512
            tile_size = 512
            half = tile_size // 2
            crop_x1 = max(0, img_cx - half)
            crop_y1 = max(0, img_cy - half)
            crop_x2 = crop_x1 + tile_size
            crop_y2 = crop_y1 + tile_size

            if crop_x2 > img_w:
                crop_x2 = img_w
                crop_x1 = max(0, crop_x2 - tile_size)
            if crop_y2 > img_h:
                crop_y2 = img_h
                crop_y1 = max(0, crop_y2 - tile_size)

            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            tile_image = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            # 5. 建立 TileInfo
            tile_info = TileInfo(
                tile_id=0,
                x=crop_x1, y=crop_y1,
                width=crop_w, height=crop_h,
                image=tile_image,
            )

            # 6. 暫時覆蓋 config 值，呼叫 _detect_bright_spots，再還原
            cfg = self.inferencer.config
            orig_diff = cfg.bright_spot_diff_threshold
            orig_kernel = cfg.bright_spot_median_kernel
            orig_area = cfg.bright_spot_min_area
            orig_thr = cfg.bright_spot_threshold
            try:
                cfg.bright_spot_diff_threshold = bs_diff_threshold
                cfg.bright_spot_median_kernel = bs_median_kernel
                cfg.bright_spot_min_area = bs_min_area
                cfg.bright_spot_threshold = bs_threshold
                score, anomaly_map = self.inferencer._detect_bright_spots(tile_info)
            finally:
                cfg.bright_spot_diff_threshold = orig_diff
                cfg.bright_spot_median_kernel = orig_kernel
                cfg.bright_spot_min_area = orig_area
                cfg.bright_spot_threshold = orig_thr

            total_time = _time.time() - total_start

            # 7. 準備暫存目錄
            if CAPIWebHandler._debug_heatmap_dir is None:
                CAPIWebHandler._debug_heatmap_dir = Path(tempfile.mkdtemp(prefix="capi_debug_hm_"))
            debug_dir = CAPIWebHandler._debug_heatmap_dir
            debug_dir.mkdir(parents=True, exist_ok=True)

            image_name = image_path.stem
            ts = int(_time.time() * 1000) % 100000

            # 8. 儲存原始裁切圖
            crop_bgr = tile_image.copy()
            if len(crop_bgr.shape) == 2:
                crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
            elif len(crop_bgr.shape) == 3 and crop_bgr.shape[2] == 1:
                crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
            crop_filename = f"debug_bs_crop_{image_name}_{ts}.png"
            cv2.imwrite(str(debug_dir / crop_filename), crop_bgr)
            crop_url = f"/debug/heatmaps/{crop_filename}"

            # 9. 產生亮點偵測結果圖 (binary overlay)
            detect_url = ""
            if anomaly_map is not None:
                binary_mask = (anomaly_map * 255).astype(np.uint8)
                # 紅色 overlay 標記亮點
                overlay = crop_bgr.copy()
                red_mask = np.zeros_like(overlay)
                red_mask[:, :, 2] = binary_mask  # Red channel
                overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.8, 0)
                # 畫亮點輪廓
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
                detect_filename = f"debug_bs_detect_{image_name}_{ts}.png"
                cv2.imwrite(str(debug_dir / detect_filename), overlay)
                detect_url = f"/debug/heatmaps/{detect_filename}"

            # 10. 產生差異圖 (diff visualization)
            diff_url = ""
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(tile_image.shape) == 3 else tile_image.copy()
            from capi_edge_cv import clamp_median_kernel
            mk = clamp_median_kernel(bs_median_kernel, min(gray.shape[:2]) - 1)
            bg = cv2.medianBlur(gray, mk)
            diff = cv2.subtract(gray, bg)
            diff_color = cv2.applyColorMap(
                cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            diff_filename = f"debug_bs_diff_{image_name}_{ts}.png"
            cv2.imwrite(str(debug_dir / diff_filename), diff_color)
            diff_url = f"/debug/heatmaps/{diff_filename}"

            # 11. 產生 Overview 圖 (加上裁切框)
            overview_img = image.copy()
            if len(overview_img.shape) == 2:
                overview_img = cv2.cvtColor(overview_img, cv2.COLOR_GRAY2BGR)
            elif len(overview_img.shape) == 3 and overview_img.shape[2] == 1:
                overview_img = cv2.cvtColor(overview_img, cv2.COLOR_GRAY2BGR)

            overlay_bg = overview_img.copy()
            cv2.rectangle(overlay_bg, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay_bg, 0.3, overview_img, 0.7, 0, overview_img)
            ox1, oy1, ox2, oy2 = otsu_bounds
            cv2.rectangle(overview_img, (ox1, oy1), (ox2, oy2), (0, 255, 255), 4)
            cv2.rectangle(overview_img, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 0, 255), 6)
            cv2.circle(overview_img, (img_cx, img_cy), 10, (0, 255, 0), -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(overview_img, f"({ox1}, {oy1})", (ox1, max(30, oy1 - 10)), font, 1.5, (0, 255, 255), 3)
            cv2.putText(overview_img, f"({ox2}, {oy2})", (max(0, ox2 - 300), min(img_h - 10, oy2 + 40)), font, 1.5, (0, 255, 255), 3)

            max_dim = 2000
            oh, ow = overview_img.shape[:2]
            if max(oh, ow) > max_dim:
                scale_o = max_dim / max(oh, ow)
                overview_img = cv2.resize(overview_img, (int(ow * scale_o), int(oh * scale_o)))
            ov_filename = f"debug_bs_overview_{image_name}_{ts}.png"
            cv2.imwrite(str(debug_dir / ov_filename), overview_img)
            overview_url = f"/debug/heatmaps/{ov_filename}"

            # 12. 判定結果
            judgment = "NG" if score >= 1.0 else "OK"

            # 從 tile_info 取出偵測統計
            bright_spot_area = getattr(tile_info, 'bright_spot_area', 0)
            bright_spot_max_diff = getattr(tile_info, 'bright_spot_max_diff', 0)

            response_data = {
                "success": True,
                "product_coord": [product_x, product_y],
                "product_resolution": [product_w, product_h],
                "image_coord": [img_cx, img_cy],
                "crop_region": [crop_x1, crop_y1, crop_x2, crop_y2],
                "raw_bounds": list(raw_bounds),
                "otsu_bounds": [ox1, oy1, ox2, oy2],
                "scale": [round(scale_x, 4), round(scale_y, 4)],
                "image_size": [img_w, img_h],
                "score": round(score, 4),
                "judgment": judgment,
                "processing_time": round(total_time, 3),
                "crop_url": crop_url,
                "detect_url": detect_url,
                "diff_url": diff_url,
                "overview_url": overview_url,
                "bright_spot_area": bright_spot_area,
                "bright_spot_max_diff": bright_spot_max_diff,
                "ref_image": ref_image_name,
                "params_used": {
                    "diff_threshold": bs_diff_threshold,
                    "median_kernel": bs_median_kernel,
                    "min_area": bs_min_area,
                    "threshold": bs_threshold,
                },
            }

            self._send_json(response_data)
            logger.info(f"[DEBUG-BS] ({product_x},{product_y})→({img_cx},{img_cy}) "
                        f"Score={score:.1f} {judgment} area={bright_spot_area} ({total_time:.2f}s)")

        except Exception as e:
            logger.error(f"[DEBUG-BS] Error: {e}", exc_info=True)
            self._send_json({"error": f"黑畫面推論失敗: {str(e)}"})

    def _handle_debug_dot_detection(self):
        """API: 點圖偵測實驗，量測黑點/白點在影像中的可見尺寸。"""
        import time as _time
        import cv2
        import numpy as np

        data = self._read_json_body()
        if data is None:
            return

        image_path_str = (data.get("image_path") or "").strip()
        if not image_path_str:
            self._send_json({"error": "請提供圖片路徑 (image_path)"}, status=400)
            return

        image_path = Path(image_path_str)
        if not image_path.exists():
            self._send_json({"error": f"檔案不存在: {image_path}"}, status=400)
            return
        if not image_path.is_file():
            self._send_json({"error": f"不是檔案: {image_path}"}, status=400)
            return

        polarity = (data.get("polarity") or "black").strip().lower()
        if polarity not in ("black", "white", "auto"):
            self._send_json({"error": "polarity 必須是 black、white 或 auto"}, status=400)
            return

        size_metric = (data.get("size_metric") or "bbox_diagonal").strip().lower()
        if size_metric not in ("bbox_diagonal", "bbox_max", "equivalent", "enclosing"):
            self._send_json({"error": "size_metric 必須是 bbox_diagonal、bbox_max、equivalent 或 enclosing"}, status=400)
            return

        segmentation_method = (data.get("segmentation_method") or "background_diff").strip().lower()
        if segmentation_method not in ("background_diff", "hysteresis", "morph_hat", "adaptive_mean", "halo", "auto", "off"):
            self._send_json({"error": "segmentation_method 必須是 background_diff、hysteresis、morph_hat、adaptive_mean、halo、auto 或 off"}, status=400)
            return

        def as_int(name, default):
            try:
                return int(data.get(name, default))
            except (TypeError, ValueError):
                return default

        def as_float(name, default):
            try:
                return float(data.get(name, default))
            except (TypeError, ValueError):
                return default

        def as_bool(name, default):
            value = data.get(name, default)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() not in ("0", "false", "no", "off")
            return bool(value)

        diff_threshold = as_int("diff_threshold", 4)
        hysteresis_low_threshold = max(0, as_int("hysteresis_low_threshold", 2))
        hysteresis_high_threshold = max(
            hysteresis_low_threshold,
            max(0, as_int("hysteresis_high_threshold", 4)),
        )
        hysteresis_second_low_threshold = max(0, as_int("hysteresis_second_low_threshold", 3))
        hysteresis_second_high_threshold = max(
            hysteresis_second_low_threshold,
            max(0, as_int("hysteresis_second_high_threshold", 4)),
        )
        hysteresis_edge_width_percent = max(0.0, as_float("hysteresis_edge_width_percent", 3.0))
        hysteresis_edge_extra_threshold = max(0, as_int("hysteresis_edge_extra_threshold", 2))
        hysteresis_second_edge_width_percent = max(0.0, as_float("hysteresis_second_edge_width_percent", 9.5))
        hysteresis_second_edge_extra_threshold = max(0, as_int("hysteresis_second_edge_extra_threshold", 2))
        hysteresis_switch_count_threshold = max(0, as_int("hysteresis_switch_count_threshold", 5))
        hysteresis_second_max_count = max(0, as_int("hysteresis_second_max_count", 5))
        hysteresis_edge_suppress_percent = max(0.0, as_float("hysteresis_edge_suppress_percent", 0.0))
        background_kernel = _odd_kernel(as_int("background_kernel", 33))
        min_area = max(1, as_int("min_area", 2))
        max_area = max(0, as_int("max_area", 50000))
        morph_open = max(0, as_int("morph_open", 0))
        min_aspect_ratio = max(0.0, as_float("min_aspect_ratio", 0.45))
        edge_margin = max(0, as_int("edge_margin", 4))
        use_default_calibration = as_bool("use_default_calibration", True)
        unit_per_px = max(0.0, as_float("unit_per_px", DOT_RULER_MM_PER_PX))
        if unit_per_px <= 0 and use_default_calibration:
            unit_per_px = DOT_RULER_MM_PER_PX
        defect_threshold = max(0.0, as_float("defect_threshold", 0.3))
        unit_label = str(data.get("unit_label") or "mm").strip()[:16] or "mm"
        preprocess_method = str(data.get("preprocess_method") or DOT_PREPROCESS_METHOD).strip() or DOT_PREPROCESS_METHOD
        preprocess_params = data.get("preprocess_params")
        if not isinstance(preprocess_params, dict):
            preprocess_params = DOT_PREPROCESS_PARAMS
        non_dot_cfg = _non_dot_residue_config({
            "non_dot_residue_enabled": as_bool("non_dot_residue_enabled", True),
            "non_dot_residue_min_area_px": as_int("non_dot_residue_min_area_px", 500),
            "non_dot_residue_min_long_side_px": as_int("non_dot_residue_min_long_side_px", 80),
            "non_dot_residue_min_long_side_ratio": as_float("non_dot_residue_min_long_side_ratio", 0.15),
            "non_dot_residue_min_max_diff": as_int("non_dot_residue_min_max_diff", 12),
            "non_dot_residue_reasons": data.get(
                "non_dot_residue_reasons",
                ["aspect_ratio_below_min", "edge_margin", "area_too_large"],
            ),
        })

        try:
            started = _time.time()
            image = self._read_inference_image(image_path, cv2.IMREAD_COLOR)
            if image is None:
                self._send_json({"error": f"無法讀取圖片: {image_path}"}, status=400)
                return
            processed_image, preprocess_info = _preprocess_dot_image_for_detection(
                image,
                method=preprocess_method,
                params=preprocess_params,
            )

            if segmentation_method == "off":
                blank = np.zeros(processed_image.shape[:2], dtype=np.uint8)
                overlay = processed_image.copy()
                if overlay.ndim == 2:
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
                detected = {
                    "overlay": overlay,
                    "mask_color": cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR),
                    "diff_color": cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR),
                    "candidates": [],
                    "rejected_candidates": [],
                    "calibrated": unit_per_px > 0,
                    "segmentation_method": "off",
                    "detected_polarity": polarity,
                    "thresholds": {
                        "diff_threshold": diff_threshold,
                        "hysteresis_low_threshold": hysteresis_low_threshold,
                        "hysteresis_high_threshold": hysteresis_high_threshold,
                        "hysteresis_second_low_threshold": hysteresis_second_low_threshold,
                        "hysteresis_second_high_threshold": hysteresis_second_high_threshold,
                        "hysteresis_edge_width_percent": hysteresis_edge_width_percent,
                        "hysteresis_edge_extra_threshold": hysteresis_edge_extra_threshold,
                        "hysteresis_second_edge_width_percent": hysteresis_second_edge_width_percent,
                        "hysteresis_second_edge_extra_threshold": hysteresis_second_edge_extra_threshold,
                        "hysteresis_switch_count_threshold": hysteresis_switch_count_threshold,
                        "hysteresis_second_max_count": hysteresis_second_max_count,
                        "hysteresis_edge_suppress_percent": hysteresis_edge_suppress_percent,
                    },
                }
            else:
                detected = _detect_dot_components_debug_polarity(
                    processed_image,
                    polarity=polarity,
                    segmentation_method=segmentation_method,
                    diff_threshold=diff_threshold,
                    background_kernel=background_kernel,
                    min_area=min_area,
                    max_area=max_area,
                    morph_open=morph_open,
                    size_metric=size_metric,
                    unit_per_px=unit_per_px,
                    defect_threshold=defect_threshold,
                    min_aspect_ratio=min_aspect_ratio,
                    edge_margin=edge_margin,
                    hysteresis_low_threshold=hysteresis_low_threshold,
                    hysteresis_high_threshold=hysteresis_high_threshold,
                    hysteresis_edge_width_percent=hysteresis_edge_width_percent,
                    hysteresis_edge_extra_threshold=hysteresis_edge_extra_threshold,
                    hysteresis_second_low_threshold=hysteresis_second_low_threshold,
                    hysteresis_second_high_threshold=hysteresis_second_high_threshold,
                    hysteresis_second_edge_width_percent=hysteresis_second_edge_width_percent,
                    hysteresis_second_edge_extra_threshold=hysteresis_second_edge_extra_threshold,
                    hysteresis_switch_count_threshold=hysteresis_switch_count_threshold,
                    hysteresis_second_max_count=hysteresis_second_max_count,
                    hysteresis_edge_suppress_percent=hysteresis_edge_suppress_percent,
                )

            def prepare_non_dot_result(detection):
                residues = []
                if non_dot_cfg.get("enabled"):
                    h_img, w_img = processed_image.shape[:2]
                    crop_box = (0, 0, int(w_img), int(h_img))
                    seen_residues = set()
                    for rejected in detection.get("rejected_candidates") or []:
                        matched = _non_dot_residue_match(rejected, non_dot_cfg, crop_box)
                        if not matched:
                            continue
                        key = (
                            matched.get("reason"),
                            matched.get("x"),
                            matched.get("y"),
                            matched.get("w"),
                            matched.get("h"),
                        )
                        if key in seen_residues:
                            continue
                        seen_residues.add(key)
                        residues.append(matched)

                if not residues:
                    return residues

                h_img, w_img = detection["overlay"].shape[:2]
                for idx, residue in enumerate(residues[:20], 1):
                    x = max(0, min(w_img - 1, _as_int(residue.get("x"), 0)))
                    y = max(0, min(h_img - 1, _as_int(residue.get("y"), 0)))
                    w = max(1, _as_int(residue.get("w"), 1))
                    h = max(1, _as_int(residue.get("h"), 1))
                    x2 = max(x + 1, min(w_img - 1, x + w))
                    y2 = max(y + 1, min(h_img - 1, y + h))
                    label_y = max(13, min(h_img - 5, y + 14))
                    label = f"NG residue #{idx}"
                    for key in ("overlay", "mask_color", "diff_color"):
                        canvas = detection.get(key)
                        if canvas is None:
                            continue
                        cv2.rectangle(canvas, (x, y), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(
                            canvas,
                            label,
                            (x + 4, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
                return residues

            polarity_detections = detected.get("polarity_results") or {}
            polarity_non_dot_residues = {
                key: prepare_non_dot_result(item)
                for key, item in polarity_detections.items()
            }
            if polarity_detections:
                non_dot_residues = polarity_non_dot_residues.get(
                    detected.get("detected_polarity", ""),
                    [],
                )
            else:
                non_dot_residues = prepare_non_dot_result(detected)

            if CAPIWebHandler._debug_heatmap_dir is None:
                CAPIWebHandler._debug_heatmap_dir = Path(tempfile.mkdtemp(prefix="capi_debug_hm_"))
            debug_dir = CAPIWebHandler._debug_heatmap_dir
            debug_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem)[:80]
            preprocessed_filename = f"debug_dot_preprocessed_{safe_name}_{ts}.png"
            overlay_filename = f"debug_dot_overlay_{safe_name}_{ts}.png"
            mask_filename = f"debug_dot_mask_{safe_name}_{ts}.png"
            diff_filename = f"debug_dot_diff_{safe_name}_{ts}.png"
            cv2.imwrite(str(debug_dir / preprocessed_filename), processed_image)
            cv2.imwrite(str(debug_dir / overlay_filename), detected["overlay"])
            cv2.imwrite(str(debug_dir / mask_filename), detected["mask_color"])
            cv2.imwrite(str(debug_dir / diff_filename), detected["diff_color"])

            polarity_visual_urls = {}
            for result_polarity, polarity_result in polarity_detections.items():
                polarity_overlay_filename = f"debug_dot_overlay_{result_polarity}_{safe_name}_{ts}.png"
                polarity_mask_filename = f"debug_dot_mask_{result_polarity}_{safe_name}_{ts}.png"
                polarity_diff_filename = f"debug_dot_diff_{result_polarity}_{safe_name}_{ts}.png"
                cv2.imwrite(str(debug_dir / polarity_overlay_filename), polarity_result["overlay"])
                cv2.imwrite(str(debug_dir / polarity_mask_filename), polarity_result["mask_color"])
                cv2.imwrite(str(debug_dir / polarity_diff_filename), polarity_result["diff_color"])
                polarity_visual_urls[result_polarity] = {
                    "overlay_url": f"/debug/heatmaps/{polarity_overlay_filename}",
                    "mask_url": f"/debug/heatmaps/{polarity_mask_filename}",
                    "diff_url": f"/debug/heatmaps/{polarity_diff_filename}",
                }

            candidates = detected["candidates"]
            defect_count = sum(1 for c in candidates if c["is_defect"])
            max_size_px = max((c["size_px"] for c in candidates), default=0.0)
            max_size_units = None
            defect_threshold_px = None
            if detected["calibrated"]:
                max_size_units = max((c["size_units"] or 0.0 for c in candidates), default=0.0)
                defect_threshold_px = defect_threshold / unit_per_px if unit_per_px > 0 else None
            max_size_mm = max_size_units if unit_label.lower() == "mm" else None
            detected_thresholds = detected.get("thresholds", {})

            polarity_response_results = {}
            for result_polarity, polarity_result in polarity_detections.items():
                polarity_candidates = polarity_result.get("candidates") or []
                polarity_max_size_px = max(
                    (candidate.get("size_px") or 0.0 for candidate in polarity_candidates),
                    default=0.0,
                )
                polarity_max_size_units = None
                if polarity_result.get("calibrated"):
                    polarity_max_size_units = max(
                        (candidate.get("size_units") or 0.0 for candidate in polarity_candidates),
                        default=0.0,
                    )
                polarity_residues = polarity_non_dot_residues.get(result_polarity, [])
                polarity_response_results[result_polarity] = {
                    "detected_polarity": result_polarity,
                    "segmentation_method": polarity_result.get("segmentation_method", segmentation_method),
                    "thresholds": polarity_result.get("thresholds", {}),
                    "calibrated": bool(polarity_result.get("calibrated")),
                    "count": len(polarity_candidates),
                    "defect_count": sum(1 for candidate in polarity_candidates if candidate.get("is_defect")),
                    "max_size_px": round(float(polarity_max_size_px), 2),
                    "max_size_units": (
                        round(float(polarity_max_size_units), 4)
                        if polarity_max_size_units is not None
                        else None
                    ),
                    "max_size_mm": (
                        round(float(polarity_max_size_units), 4)
                        if polarity_max_size_units is not None and unit_label.lower() == "mm"
                        else None
                    ),
                    "non_dot_residue": {
                        "enabled": bool(non_dot_cfg.get("enabled")),
                        "count": len(polarity_residues),
                        "blocks_within_spec": bool(polarity_residues),
                        "params": non_dot_cfg,
                        "residues": polarity_residues[:50],
                    },
                    "auto_candidates": polarity_result.get("auto_candidates", []),
                    "candidates": polarity_candidates[:300],
                    **polarity_visual_urls.get(result_polarity, {}),
                }

            response_data = {
                "success": True,
                "image_path": str(image_path),
                "image_shape": [int(image.shape[1]), int(image.shape[0])],
                "polarity": polarity,
                "detected_polarity": detected.get("detected_polarity", polarity),
                "method": "dot_detection",
                "segmentation_method": detected.get("segmentation_method", segmentation_method),
                "preprocess": preprocess_info,
                "calibrated": detected["calibrated"],
                "count": len(candidates),
                "defect_count": defect_count,
                "max_size_px": round(float(max_size_px), 2),
                "max_size_units": round(float(max_size_units), 4) if max_size_units is not None else None,
                "max_size_mm": round(float(max_size_mm), 4) if max_size_mm is not None else None,
                "defect_threshold_px": round(float(defect_threshold_px), 2) if defect_threshold_px is not None else None,
                "unit_label": unit_label,
                "calibration": {
                    "unit_per_px": round(float(unit_per_px), 6),
                    "mm_per_px": round(float(unit_per_px), 6) if unit_label.lower() == "mm" else None,
                    "source": DOT_RULER_CALIBRATION_SOURCE if use_default_calibration else "manual mm/px input",
                    "points": DOT_RULER_CALIBRATION_POINTS if use_default_calibration else [],
                    "formula": f"size_mm = {size_metric}_px * {unit_per_px:.6f}",
                },
                "processing_time": round(_time.time() - started, 3),
                "params_used": {
                    "segmentation_method": detected.get("segmentation_method", segmentation_method),
                    "diff_threshold": diff_threshold,
                    "hysteresis_low_threshold": detected_thresholds.get("hysteresis_low_threshold", hysteresis_low_threshold),
                    "hysteresis_high_threshold": detected_thresholds.get("hysteresis_high_threshold", hysteresis_high_threshold),
                    "hysteresis_second_low_threshold": detected_thresholds.get("hysteresis_second_low_threshold", hysteresis_second_low_threshold),
                    "hysteresis_second_high_threshold": detected_thresholds.get("hysteresis_second_high_threshold", hysteresis_second_high_threshold),
                    "hysteresis_edge_width_percent": detected_thresholds.get("hysteresis_edge_width_percent", hysteresis_edge_width_percent),
                    "hysteresis_edge_extra_threshold": detected_thresholds.get("hysteresis_edge_extra_threshold", hysteresis_edge_extra_threshold),
                    "hysteresis_second_edge_width_percent": detected_thresholds.get("hysteresis_second_edge_width_percent", hysteresis_second_edge_width_percent),
                    "hysteresis_second_edge_extra_threshold": detected_thresholds.get("hysteresis_second_edge_extra_threshold", hysteresis_second_edge_extra_threshold),
                    "hysteresis_switch_count_threshold": detected_thresholds.get("hysteresis_switch_count_threshold", hysteresis_switch_count_threshold),
                    "hysteresis_second_max_count": detected_thresholds.get("hysteresis_second_max_count", hysteresis_second_max_count),
                    "hysteresis_edge_suppress_percent": detected_thresholds.get("hysteresis_edge_suppress_percent", hysteresis_edge_suppress_percent),
                    "hysteresis_selected_group": detected_thresholds.get("hysteresis_selected_group"),
                    "hysteresis_group1_count": detected_thresholds.get("hysteresis_group1_count"),
                    "hysteresis_group2_count": detected_thresholds.get("hysteresis_group2_count"),
                    "background_kernel": background_kernel,
                    "min_area": min_area,
                    "max_area": max_area,
                    "morph_open": morph_open,
                    "min_aspect_ratio": min_aspect_ratio,
                    "edge_margin": edge_margin,
                    "size_metric": size_metric,
                    "unit_per_px": unit_per_px,
                    "defect_threshold": defect_threshold,
                    "use_default_calibration": use_default_calibration,
                    "non_dot_residue_enabled": bool(non_dot_cfg.get("enabled")),
                    "non_dot_residue_min_area_px": int(non_dot_cfg.get("min_area_px", 0)),
                    "non_dot_residue_min_long_side_px": int(non_dot_cfg.get("min_long_side_px", 0)),
                    "non_dot_residue_min_long_side_ratio": float(non_dot_cfg.get("min_long_side_ratio", 0.0)),
                    "non_dot_residue_min_max_diff": int(non_dot_cfg.get("min_max_diff", 0)),
                },
                "non_dot_residue": {
                    "enabled": bool(non_dot_cfg.get("enabled")),
                    "count": len(non_dot_residues),
                    "blocks_within_spec": bool(non_dot_residues),
                    "params": non_dot_cfg,
                    "residues": non_dot_residues[:50],
                },
                "auto_candidates": detected.get("auto_candidates", []),
                "candidates": candidates[:300],
                "polarity_results": polarity_response_results,
                "source_url": "/api/debug/serve-image?path=" + urllib.parse.quote(str(image_path)),
                "preprocessed_url": f"/debug/heatmaps/{preprocessed_filename}",
                "overlay_url": f"/debug/heatmaps/{overlay_filename}",
                "mask_url": f"/debug/heatmaps/{mask_filename}",
                "diff_url": f"/debug/heatmaps/{diff_filename}",
            }
            self._send_json(response_data)
            logger.info(
                "[DEBUG-DOT] %s polarity=%s count=%s defects=%s max_px=%.2f",
                image_path.name,
                polarity,
                len(candidates),
                defect_count,
                max_size_px,
            )
        except Exception as e:
            logger.error(f"[DEBUG-DOT] Error: {e}", exc_info=True)
            self._send_json({"error": f"點圖偵測失敗: {str(e)}"}, status=500)

    def _handle_debug_heatmap_file(self, path: str):
        """靜態檔案服務 (Debug 推論熱力圖)"""
        if self._debug_heatmap_dir is None:
            self._send_404()
            return
        rel_path = path[len("/debug/heatmaps/"):]
        rel_path = rel_path.replace("..", "").lstrip("/")
        full_path = self._debug_heatmap_dir / rel_path
        if full_path.exists() and full_path.is_file():
            self._send_binary(str(full_path))
        else:
            self._send_404()

    # ── 設定管理功能 ─────────────────────────────────────

    def _handle_settings_login_page(self, path: str):
        """參數設定登入頁。"""
        if self._current_settings_user():
            self._redirect("/settings")
            return
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        next_path = self._safe_settings_next_path(query.get("next", ["/settings"])[0])
        template = self.jinja_env.get_template("settings_login.html")
        html = template.render(request_path=path, next_path=next_path)
        self._send_response(200, html)

    def _handle_settings_logout(self):
        self._drop_settings_session()
        self._redirect(
            "/settings/login",
            headers={"Set-Cookie": self._settings_clear_cookie_header()},
        )

    def _load_central_account_location(self) -> Dict[str, str]:
        server_config = (
            getattr(self._capi_server_instance, "server_config", {}) or {}
        )
        default_location = _default_central_account_location(server_config)
        if not self.db:
            return default_location
        try:
            param = self.db.get_config_param(CENTRAL_ACCOUNT_LOCATION_PARAM)
            if not param:
                return default_location
            value = param.get("decoded_value")
            if value is None:
                value = json.loads(param.get("param_value") or "{}")
            return _normalize_central_account_location(value)
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "Invalid central account location; using facility default: %s",
                exc,
            )
            return default_location

    def _verify_central_settings_user(
        self,
        username: str,
        password: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        location = self._load_central_account_location()
        center_ip = location["ip"]
        request_body = json.dumps(
            {"username": username, "password": password},
            ensure_ascii=False,
        ).encode("utf-8")
        request_headers = {
            "Content-Type": "application/json",
            CENTRAL_ACCOUNT_AUTH_HEADER: "1",
        }

        for auth_path in (CENTRAL_ACCOUNT_AUTH_PATH, "/api/settings/login"):
            connection = None
            try:
                connection = http.client.HTTPConnection(
                    center_ip,
                    80,
                    timeout=CENTRAL_ACCOUNT_AUTH_TIMEOUT_SECONDS,
                )
                connection.request(
                    "POST",
                    auth_path,
                    body=request_body,
                    headers=request_headers,
                )
                response = connection.getresponse()
                status = int(response.status)
                response_body = response.read()
            except (OSError, http.client.HTTPException) as exc:
                logger.warning(
                    "Central account authentication unavailable at %s: %s",
                    center_ip,
                    exc,
                )
                return (
                    None,
                    "中心帳號服務無法連線，請改用本機帳號或聯絡管理員",
                )
            finally:
                if connection is not None:
                    connection.close()

            if status == 404 and auth_path == CENTRAL_ACCOUNT_AUTH_PATH:
                continue
            if status == 401:
                return None, None
            if status != 200:
                logger.warning(
                    "Central account authentication returned HTTP %s from %s",
                    status,
                    center_ip,
                )
                return (
                    None,
                    "中心帳號服務回應異常，請改用本機帳號或聯絡管理員",
                )

            try:
                payload = json.loads(response_body.decode("utf-8"))
                remote_user = payload.get("user")
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                remote_user = None
            if (
                not isinstance(remote_user, dict)
                or str(remote_user.get("username") or "") != username
            ):
                logger.warning(
                    "Central account authentication returned invalid user data from %s",
                    center_ip,
                )
                return (
                    None,
                    "中心帳號服務回應異常，請改用本機帳號或聯絡管理員",
                )

            try:
                user_id = int(remote_user.get("id") or 0)
            except (TypeError, ValueError):
                user_id = 0
            is_admin = bool(remote_user.get("is_admin"))
            user = {
                "id": user_id,
                "username": username,
                "is_admin": is_admin,
                "can_manage_accounts": bool(
                    remote_user.get("can_manage_accounts") or is_admin
                ),
                "created_at": str(remote_user.get("created_at") or ""),
                "updated_at": str(remote_user.get("updated_at") or ""),
                "auth_source": "central",
                "central_facility": location["facility"],
            }
            logger.info(
                "Settings login authenticated by central account service %s",
                center_ip,
            )
            return user, None

        return (
            None,
            "中心帳號服務版本不支援中央登入，請先更新中心主機",
        )

    def _handle_api_settings_central_auth(self):
        if (
            str(self.headers.get(CENTRAL_ACCOUNT_AUTH_HEADER, "") or "")
            != "1"
        ):
            self._send_json({"error": "不允許的驗證來源"}, status=403)
            return
        data = self._read_json_body()
        if data is None:
            return
        username = str(data.get("username", "") or "").strip()
        password = str(data.get("password", "") or "")
        user = self.db.verify_settings_user(username, password) if self.db else None
        if not user:
            self._send_json({"error": "帳號或密碼錯誤"}, status=401)
            return
        self._send_json({"success": True, "user": user})

    def _handle_api_settings_login(self):
        try:
            data = self._read_json_body()
            if data is None:
                return
            username = str(data.get("username", "") or "").strip()
            password = str(data.get("password", "") or "")
            next_path = self._safe_settings_next_path(data.get("next", "") or "/settings")
            if not username or not password:
                self._send_json({"error": "帳號或密碼錯誤"}, status=401)
                return
            user = self.db.verify_settings_user(username, password) if self.db else None
            central_error = None
            forwarded_auth = (
                str(self.headers.get(CENTRAL_ACCOUNT_AUTH_HEADER, "") or "")
                == "1"
            )
            if not user and not forwarded_auth:
                local_user = (
                    self.db.get_settings_user_by_username(username)
                    if self.db
                    else None
                )
                if not local_user:
                    user, central_error = self._verify_central_settings_user(
                        username,
                        password,
                    )
            if not user:
                if central_error:
                    self._send_json({"error": central_error}, status=503)
                else:
                    self._send_json({"error": "帳號或密碼錯誤"}, status=401)
                return
            token = self._create_settings_session(user)
            self._send_json(
                {"success": True, "user": user, "redirect": next_path or "/settings"},
                headers={"Set-Cookie": self._settings_cookie_header(token)},
            )
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_api_settings_logout(self):
        self._drop_settings_session()
        self._send_json(
            {"success": True},
            headers={"Set-Cookie": self._settings_clear_cookie_header()},
        )

    def _handle_api_settings_users(self):
        users = self.db.list_settings_users() if self.db else []
        self._send_json({"users": users})

    @staticmethod
    def _mark_sample_public(sample: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            key: value
            for key, value in sample.items()
            if key not in {"image_path", "prototypes"}
        }
        result["image_url"] = (
            f"/api/settings/mark/sample-image?id={int(sample.get('id') or 0)}"
        )
        return result

    @staticmethod
    def _mark_profile_public(profile: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in profile.items()
            if key != "profile"
        }

    @staticmethod
    def _mark_detection_public(detection: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "found": bool(detection.get("found")),
            "text": str(detection.get("text") or ""),
            "confidence": float(detection.get("confidence") or 0.0),
            "bbox": detection.get("bbox") or {},
            "roi": str(detection.get("roi") or ""),
            "orientation": str(detection.get("orientation") or ""),
            "profile_version": int(detection.get("profile_version") or 0),
            "chars": [
                {
                    "char": str(item.get("char") or ""),
                    "score": float(item.get("score") or 0.0),
                    "base_score": float(item.get("base_score") or 0.0),
                    "prototype_similarity": item.get("prototype_similarity"),
                }
                for item in (detection.get("chars") or [])
            ],
        }

    def _handle_api_settings_mark(self):
        try:
            active = self.db.get_active_mark_profile()
            profiles = self.db.list_mark_profiles(limit=50)
            samples = self.db.list_mark_calibration_samples(limit=200)
            self._send_json(
                {
                    "active_profile": self._mark_profile_public(active),
                    "profiles": [
                        self._mark_profile_public(profile)
                        for profile in profiles
                    ],
                    "samples": [
                        self._mark_sample_public(sample)
                        for sample in samples
                    ],
                }
            )
        except Exception as exc:
            logger.error("MARK calibration list failed: %s", exc, exc_info=True)
            self._send_json({"error": str(exc)}, status=500)

    def _handle_api_settings_mark_sample_image(self, query: dict):
        try:
            sample_id = int((query.get("id") or ["0"])[0])
        except (TypeError, ValueError):
            self._send_json({"error": "校正樣本 id 格式錯誤"}, status=400)
            return
        sample = self.db.get_mark_calibration_sample(sample_id)
        if not sample:
            self._send_json({"error": "找不到校正樣本"}, status=404)
            return
        image_path = Path(str(sample.get("image_path") or ""))
        if not image_path.is_file():
            self._send_json({"error": "校正樣本圖片遺失"}, status=404)
            return
        import cv2

        image = read_detection_image(
            image_path,
            cv2.IMREAD_UNCHANGED,
            bool(sample.get("rotation_applied")),
        )
        if image is None:
            self._send_json({"error": "校正樣本圖片無法讀取"}, status=500)
            return
        height, width = image.shape[:2]
        max_side = max(height, width)
        if max_side > 1800:
            scale = 1800.0 / max_side
            image = cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        self._send_image_array_png(image)

    def _mark_shadow_db_path(self) -> Path:
        server_config = getattr(self._capi_server_instance, "server_config", {}) or {}
        shadow_config = server_config.get("mark_shadow", {}) or {}
        configured = (
            os.environ.get("CAPI_MARK_SHADOW_DB_PATH")
            or shadow_config.get("database_path")
            or "/aidata/capi_ai/mark_shadow/data/mark_shadow.db"
        )
        return Path(str(configured)).expanduser().resolve()

    @staticmethod
    def _mark_shadow_percentile(values: List[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(
            len(ordered) - 1,
            max(0, int(round((len(ordered) - 1) * fraction))),
        )
        return float(ordered[index])

    def _open_mark_shadow_db(self):
        db_path = self._mark_shadow_db_path()
        if not db_path.is_file():
            raise FileNotFoundError(f"MARK Shadow DB 不存在：{db_path}")
        connection = sqlite3.connect(str(db_path), timeout=3)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    def _handle_api_settings_mark_shadow(self, query: dict):
        db_path = self._mark_shadow_db_path()
        try:
            limit = max(1, min(500, int((query.get("limit") or ["100"])[0])))
        except (TypeError, ValueError):
            limit = 100
        filter_name = str((query.get("filter") or ["all"])[0]).strip().lower()
        filters = {
            "all": ("", ()),
            "agreed": ("WHERE agreed = 1 AND error = ''", ()),
            "disagreed": ("WHERE agreed = 0", ()),
            "errors": ("WHERE error <> ''", ()),
            "no_read": ("WHERE valid_two_chars = 0", ()),
        }
        if filter_name not in filters:
            self._send_json({"error": "MARK Shadow 篩選條件錯誤"}, status=400)
            return

        if not db_path.is_file():
            self._send_json(
                {
                    "available": False,
                    "db_path": str(db_path),
                    "filter": filter_name,
                    "limit": limit,
                    "rows": [],
                    "stats": {
                        "total": 0,
                        "valid_two_chars": 0,
                        "no_read": 0,
                        "agreed": 0,
                        "disagreed": 0,
                        "agreement_rate": 0.0,
                        "error_count": 0,
                        "latency_ms": {"average": 0.0, "p50": 0.0, "p95": 0.0},
                    },
                    "error": f"MARK Shadow DB 不存在：{db_path}",
                }
            )
            return

        try:
            where_sql, params = filters[filter_name]
            with self._open_mark_shadow_db() as connection:
                shadow_columns = {
                    str(row["name"])
                    for row in connection.execute(
                        "PRAGMA table_info(mark_shadow_results)"
                    ).fetchall()
                }
                inference_link_column = (
                    "inference_record_id"
                    if "inference_record_id" in shadow_columns
                    else "0 AS inference_record_id"
                )
                stream_key_column = (
                    "stream_key"
                    if "stream_key" in shadow_columns
                    else "'' AS stream_key"
                )
                final_text_column = (
                    "final_text"
                    if "final_text" in shadow_columns
                    else "'' AS final_text"
                )
                adoption_reason_column = (
                    "adoption_reason"
                    if "adoption_reason" in shadow_columns
                    else "'' AS adoption_reason"
                )
                temporal_stable_text_column = (
                    "temporal_stable_text"
                    if "temporal_stable_text" in shadow_columns
                    else "'' AS temporal_stable_text"
                )
                temporal_history_count_column = (
                    "temporal_history_count"
                    if "temporal_history_count" in shadow_columns
                    else "0 AS temporal_history_count"
                )
                temporal_support_count_column = (
                    "temporal_stable_support_count"
                    if "temporal_stable_support_count" in shadow_columns
                    else "0 AS temporal_stable_support_count"
                )
                rows = connection.execute(
                    f"""
                    SELECT
                        id, created_at, captured_at, source_path, source_image,
                        current_text, current_confidence,
                        current_profile_version, current_roi,
                        current_orientation, paddle_raw_text, paddle_text,
                        paddle_confidence, valid_two_chars, agreed,
                        {stream_key_column}, {final_text_column},
                        {adoption_reason_column}, {temporal_stable_text_column},
                        {temporal_history_count_column},
                        {temporal_support_count_column},
                        latency_ms, model_name, crop_path, expected_text, error,
                        {inference_link_column}
                    FROM mark_shadow_results
                    {where_sql}
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (*params, limit),
                ).fetchall()
                stat_rows = connection.execute(
                    """
                    SELECT valid_two_chars, agreed, latency_ms, error
                    FROM mark_shadow_results
                    """
                ).fetchall()

            total = len(stat_rows)
            valid = sum(int(row["valid_two_chars"] or 0) for row in stat_rows)
            agreed = sum(int(row["agreed"] or 0) for row in stat_rows)
            errors = sum(1 for row in stat_rows if str(row["error"] or ""))
            latencies = [
                float(row["latency_ms"])
                for row in stat_rows
                if not str(row["error"] or "") and float(row["latency_ms"] or 0) > 0
            ]
            record_ids = [
                int(row["inference_record_id"])
                if int(row["inference_record_id"] or 0) > 0
                else None
                for row in rows
            ]
            database = getattr(self, "db", None)
            unresolved_indexes = [
                index for index, record_id in enumerate(record_ids)
                if record_id is None
            ]
            if database is not None and unresolved_indexes:
                try:
                    fallback_ids = database.find_inference_record_ids_for_images(
                        [
                            (
                                str(rows[index]["source_path"] or ""),
                                str(rows[index]["source_image"] or ""),
                            )
                            for index in unresolved_indexes
                        ]
                    )
                    for index, fallback_id in zip(
                        unresolved_indexes,
                        fallback_ids,
                    ):
                        if fallback_id:
                            record_ids[index] = int(fallback_id)
                except Exception as exc:
                    logger.warning("MARK Shadow inference record link lookup failed: %s", exc)

            public_rows = []
            for index, row in enumerate(rows):
                item = dict(row)
                item.pop("source_path", None)
                crop_path = str(item.pop("crop_path", "") or "")
                item["crop_url"] = (
                    f"/api/settings/mark-shadow/crop?id={int(item['id'])}"
                    if crop_path
                    else ""
                )
                item["valid_two_chars"] = bool(item.get("valid_two_chars"))
                item["agreed"] = bool(item.get("agreed"))
                raw_final_text = str(item.get("final_text") or "").strip().upper()
                if not raw_final_text and item["valid_two_chars"]:
                    raw_final_text = str(item.get("paddle_text") or "").strip().upper()
                if not raw_final_text and not item["valid_two_chars"]:
                    raw_final_text = str(item.get("current_text") or "").strip().upper()
                    if raw_final_text and not item.get("adoption_reason"):
                        item["adoption_reason"] = "dotmatrix_fallback"
                item["final_text"] = raw_final_text
                current_text = str(item.get("current_text") or "").strip().upper()
                item["final_agreed"] = bool(
                    raw_final_text and current_text and raw_final_text == current_text
                )
                record_id = record_ids[index] if index < len(record_ids) else None
                item["inference_record_id"] = int(record_id) if record_id else None
                item["record_url"] = f"/record/{int(record_id)}" if record_id else ""
                public_rows.append(item)

            self._send_json(
                {
                    "available": True,
                    "db_path": str(db_path),
                    "filter": filter_name,
                    "limit": limit,
                    "rows": public_rows,
                    "stats": {
                        "total": total,
                        "valid_two_chars": valid,
                        "no_read": total - valid,
                        "agreed": agreed,
                        "disagreed": total - agreed,
                        "agreement_rate": (agreed / total) if total else 0.0,
                        "error_count": errors,
                        "latency_ms": {
                            "average": (
                                sum(latencies) / len(latencies)
                                if latencies
                                else 0.0
                            ),
                            "p50": self._mark_shadow_percentile(latencies, 0.50),
                            "p95": self._mark_shadow_percentile(latencies, 0.95),
                        },
                    },
                }
            )
        except Exception as exc:
            logger.error("MARK Shadow list failed: %s", exc, exc_info=True)
            self._send_json({"error": str(exc)}, status=500)

    def _handle_api_settings_mark_shadow_crop(self, query: dict):
        try:
            result_id = int((query.get("id") or ["0"])[0])
        except (TypeError, ValueError):
            self._send_json({"error": "MARK Shadow id 格式錯誤"}, status=400)
            return
        if result_id <= 0:
            self._send_json({"error": "MARK Shadow id 格式錯誤"}, status=400)
            return

        try:
            with self._open_mark_shadow_db() as connection:
                row = connection.execute(
                    "SELECT crop_path FROM mark_shadow_results WHERE id = ?",
                    (result_id,),
                ).fetchone()
            if not row:
                self._send_json({"error": "找不到 MARK Shadow 紀錄"}, status=404)
                return

            raw_crop_path = str(row["crop_path"] or "")
            if not raw_crop_path:
                self._send_json({"error": "此紀錄未保存 crop 圖片"}, status=404)
                return

            crop_path = Path(raw_crop_path).expanduser().resolve()
            data_root = self._mark_shadow_db_path().parent
            allowed_roots = [
                (data_root / "disagreements").resolve(),
                (data_root / "crops").resolve(),
            ]
            if not any(crop_path.is_relative_to(root) for root in allowed_roots):
                self._send_json({"error": "MARK Shadow crop 路徑不在允許範圍"}, status=403)
                return
            if not crop_path.is_file():
                self._send_json({"error": "MARK Shadow crop 圖片遺失"}, status=404)
                return

            import cv2

            image = read_detection_image(
                crop_path,
                cv2.IMREAD_UNCHANGED,
                rotate_180=False,
            )
            if image is None:
                self._send_json({"error": "MARK Shadow crop 圖片無法讀取"}, status=500)
                return
            self._send_image_array_png(image)
        except Exception as exc:
            logger.error("MARK Shadow crop failed: %s", exc, exc_info=True)
            self._send_json({"error": str(exc)}, status=500)

    def _read_mark_correction_form(self):
        import cgi
        import io

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("請使用 multipart/form-data 上傳圖片")

        content_length = int(self.headers.get("Content-Length", 0) or 0)
        max_upload_bytes = 128 * 1024 * 1024
        if content_length <= 0:
            raise ValueError("上傳內容不可空白")
        if content_length > max_upload_bytes:
            raise ValueError("圖片上傳大小不可超過 128 MB")

        body = self.rfile.read(content_length)
        fs = cgi.FieldStorage(
            fp=io.BytesIO(body),
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(content_length),
            },
            keep_blank_values=True,
        )
        image_path_text = str(
            fs.getfirst("image_path", "") or ""
        ).strip()
        file_item = fs["file"] if "file" in fs else None
        if isinstance(file_item, list):
            file_item = file_item[0]
        upload_filename = Path(
            str(getattr(file_item, "filename", "") or "")
        ).name
        has_upload = bool(upload_filename)
        if has_upload and image_path_text:
            raise ValueError("圖片上傳與圖片路徑只能擇一")

        if has_upload:
            filename = upload_filename
            file_data = file_item.file.read(max_upload_bytes + 1)
            if not file_data:
                raise ValueError("上傳圖片不可空白")
            if len(file_data) > max_upload_bytes:
                raise ValueError("圖片上傳大小不可超過 128 MB")
        elif image_path_text:
            if "\x00" in image_path_text:
                raise ValueError("圖片文件路徑格式錯誤")
            source_path = Path(image_path_text)
            if not source_path.is_absolute():
                raise ValueError("請輸入伺服器上的絕對圖片文件路徑")
            try:
                source_path = source_path.resolve(strict=True)
                if not source_path.is_file():
                    raise ValueError("圖片文件路徑不是檔案")
                if source_path.stat().st_size > max_upload_bytes:
                    raise ValueError("圖片文件大小不可超過 128 MB")
                file_data = source_path.read_bytes()
            except ValueError:
                raise
            except (OSError, RuntimeError) as exc:
                raise ValueError(f"無法讀取圖片文件路徑：{exc}") from exc
            if not file_data:
                raise ValueError("圖片文件不可空白")
            if len(file_data) > max_upload_bytes:
                raise ValueError("圖片文件大小不可超過 128 MB")
            filename = source_path.name
        else:
            raise ValueError("請選擇 MARK 圖片或輸入伺服器圖片文件路徑")

        return (
            filename,
            file_data,
            str(fs.getfirst("correct_text", "") or "").strip().upper(),
            str(fs.getfirst("reason", "") or "").strip(),
        )

    def _handle_api_settings_mark_correct(self):
        import cv2
        import numpy as np

        try:
            filename, file_data, expected_text, reason = (
                self._read_mark_correction_form()
            )
            if not re.fullmatch(r"[A-Z0-9]{2}", expected_text):
                raise ValueError("正確 MARK 必須是兩碼英數字")
            if not reason:
                raise ValueError("請填寫校正原因")

            suffix = Path(filename).suffix.lower()
            allowed_suffixes = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
            if suffix not in allowed_suffixes:
                raise ValueError("只接受 TIF、TIFF、PNG、JPG 或 BMP 圖片")
            if not filename.upper().startswith("W0F0000"):
                raise ValueError("MARK 校正只接受 W0F0000 畫面圖片")

            image = cv2.imdecode(
                np.frombuffer(file_data, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            if image is None or image.ndim not in (2, 3):
                raise ValueError("圖片格式無法解碼")
            if image.ndim == 3 and image.shape[2] not in (3, 4):
                raise ValueError("圖片色彩通道格式不支援")
            if int(image.shape[0]) * int(image.shape[1]) > 80_000_000:
                raise ValueError("圖片像素數不可超過 8,000 萬")

            user = self._current_settings_user() or {}
            actor = str(user.get("username") or "")
            file_sha256 = hashlib.sha256(file_data).hexdigest()

            with CAPIWebHandler._mark_calibration_lock:
                duplicate = self.db.get_mark_calibration_sample_by_hash(
                    file_sha256
                )
                is_revision = bool(
                    duplicate
                    and str(duplicate.get("expected_text") or "")
                    != expected_text
                )

                active = self.db.get_active_mark_profile()
                rotate_180 = (
                    bool(duplicate.get("rotation_applied"))
                    if duplicate
                    else bool(
                        self.inferencer
                        and getattr(
                            getattr(self.inferencer, "config", None),
                            "inference_rotate_180_enabled",
                            False,
                        )
                    )
                )
                oriented_image = (
                    cv2.rotate(image, cv2.ROTATE_180)
                    if rotate_180
                    else image
                )

                from capi_mark_calibration import (
                    build_mark_profile,
                    mark_sample_set_sha256,
                    run_mark_profile_regression,
                )
                from capi_mark_detector import (
                    build_mark_calibration_prototypes,
                    detect_panel_mark,
                    set_active_mark_profile,
                )

                original_detection = detect_panel_mark(
                    oriented_image,
                    include_debug=False,
                    profile=active["profile"],
                    profile_id=active["id"],
                )
                if is_revision:
                    previous_text = str(duplicate.get("expected_text") or "")
                    force_positions = [
                        position
                        for position in (0, 1)
                        if previous_text[position] != expected_text[position]
                    ]
                    changed_prototypes = build_mark_calibration_prototypes(
                        original_detection,
                        expected_text,
                        force_positions=force_positions,
                    )
                    prototypes = [
                        item
                        for item in (duplicate.get("prototypes") or [])
                        if int(item.get("position", -1)) not in force_positions
                    ] + changed_prototypes
                elif duplicate:
                    prototypes = duplicate.get("prototypes") or []
                else:
                    prototypes = build_mark_calibration_prototypes(
                        original_detection,
                        expected_text,
                    )

                if not duplicate or is_revision:
                    existing_samples = self.db.list_mark_calibration_samples()
                    provisional_id = max(
                        (
                            int(item.get("id") or 0)
                            for item in existing_samples
                        ),
                        default=0,
                    ) + 1
                    provisional_sample = {
                        "id": provisional_id,
                        "prototypes": prototypes,
                    }
                    if is_revision:
                        provisional_sample = {
                            **duplicate,
                            "expected_text": expected_text,
                            "prototypes": prototypes,
                        }
                        existing_samples = [
                            item
                            for item in existing_samples
                            if int(item.get("id") or 0)
                            != int(duplicate.get("id") or 0)
                        ]
                    build_mark_profile(
                        [
                            *existing_samples,
                            provisional_sample,
                        ]
                    )

                storage_dir = self.db.get_mark_calibration_storage_dir()
                if duplicate:
                    image_path = Path(str(duplicate.get("image_path") or ""))
                    if image_path.parent.resolve() != storage_dir.resolve():
                        raise RuntimeError("既有 MARK 樣本路徑不在校正資料夾內")
                else:
                    image_path = storage_dir / f"{file_sha256}.img"
                needs_write = not image_path.exists()
                if not needs_write:
                    needs_write = _sha256_path(image_path) != file_sha256
                if needs_write:
                    fd, temp_name = tempfile.mkstemp(
                        prefix=f"{file_sha256}.",
                        suffix=".tmp",
                        dir=str(storage_dir),
                    )
                    try:
                        with os.fdopen(fd, "wb") as output:
                            output.write(file_data)
                            output.flush()
                            os.fsync(output.fileno())
                        os.replace(temp_name, image_path)
                    finally:
                        if os.path.exists(temp_name):
                            os.unlink(temp_name)

                if is_revision:
                    sample = self.db.revise_mark_calibration_sample(
                        duplicate["id"],
                        expected_text=expected_text,
                        prototypes=prototypes,
                        changed_by=actor,
                        reason=reason,
                    )
                elif duplicate:
                    sample = duplicate
                else:
                    try:
                        sample = self.db.add_mark_calibration_sample(
                            {
                                "file_sha256": file_sha256,
                                "image_path": str(image_path),
                                "original_filename": filename,
                                "expected_text": expected_text,
                                "original_text": original_detection.get("text", ""),
                                "original_confidence": original_detection.get(
                                    "confidence",
                                    0.0,
                                ),
                                "original_roi": original_detection.get("roi", ""),
                                "original_orientation": original_detection.get(
                                    "orientation",
                                    "",
                                ),
                                "original_bbox": original_detection.get("bbox") or {},
                                "prototypes": prototypes,
                                "rotation_applied": rotate_180,
                                "profile_id_before": active["id"],
                                "created_by": actor,
                                "reason": reason,
                            }
                        )
                    except Exception:
                        try:
                            if (
                                needs_write
                                and not self.db.get_mark_calibration_sample_by_hash(
                                    file_sha256
                                )
                                and image_path.is_file()
                                and _sha256_path(image_path) == file_sha256
                            ):
                                image_path.unlink()
                        except OSError:
                            logger.warning(
                                "Failed to clean unused MARK upload %s",
                                image_path,
                                exc_info=True,
                            )
                        raise

                all_samples = self.db.list_mark_calibration_samples()
                sample_set_sha256 = mark_sample_set_sha256(all_samples)
                active_report = active.get("regression_report") or {}
                if (
                    duplicate
                    and not is_revision
                    and str(active.get("sample_set_sha256") or "")
                    == sample_set_sha256
                    and bool(active_report.get("success"))
                    and int(active.get("regression_failed") or 0) == 0
                ):
                    self._send_json(
                        {
                            "success": True,
                            "activated": True,
                            "already_applied": True,
                            "revised": False,
                            "sample": self._mark_sample_public(sample),
                            "original_detection": self._mark_detection_public(
                                original_detection
                            ),
                            "candidate_profile": self._mark_profile_public(
                                active
                            ),
                            "regression": active_report,
                            "message": (
                                f"這張圖片的校正已包含在啟用版 "
                                f"v{active['id']}"
                            ),
                        }
                    )
                    return
                candidate_data = build_mark_profile(all_samples)
                candidate = self.db.create_mark_profile(
                    candidate_data,
                    parent_profile_id=active["id"],
                    sample_count=len(all_samples),
                    sample_set_sha256=sample_set_sha256,
                    created_by=actor,
                    reason=reason,
                    triggering_sample_id=sample["id"],
                )
                regression = None
                try:
                    regression = run_mark_profile_regression(
                        all_samples,
                        candidate_data,
                        profile_id=candidate["id"],
                    )
                    activated = bool(regression.get("success"))
                    finalized = self.db.finalize_mark_profile(
                        candidate["id"],
                        regression,
                        activate=activated,
                    )
                except Exception as exc:
                    rejected_report = dict(
                        regression
                        or {
                            "total": len(all_samples),
                            "passed": 0,
                            "failed": len(all_samples),
                            "sample_set_sha256": sample_set_sha256,
                            "failures": [
                                {
                                    "reason": f"回歸執行失敗: {exc}",
                                    "actual_text": "",
                                }
                            ],
                        }
                    )
                    rejected_report["success"] = False
                    rejected_report["activation_error"] = str(exc)
                    try:
                        self.db.finalize_mark_profile(
                            candidate["id"],
                            rejected_report,
                            activate=False,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to reject stale MARK profile v%s",
                            candidate["id"],
                            exc_info=True,
                        )
                    raise
                if activated:
                    set_active_mark_profile(
                        finalized["profile"],
                        finalized["id"],
                    )

            logger.info(
                "[MARK CALIBRATION] sample=%s expected=%s revised=%s "
                "profile=%s activated=%s regression=%s/%s admin=%s",
                sample["id"],
                expected_text,
                is_revision,
                finalized["id"],
                activated,
                regression["passed"],
                regression["total"],
                actor,
            )
            if activated:
                response_message = (
                    (
                        f"已修訂樣本 #{sample['id']}；"
                        if is_revision
                        else ""
                    )
                    + f"全 {regression['total']} 筆回歸通過，"
                    + f"已自動啟用 MARK profile v{finalized['id']}"
                )
            else:
                saved_action = "已修訂" if is_revision else "已保存"
                response_message = (
                    f"樣本 #{sample['id']} {saved_action}，但回歸有 "
                    f"{regression['failed']} 筆失敗，維持原啟用版本"
                )
            self._send_json(
                {
                    "success": True,
                    "activated": activated,
                    "revised": is_revision,
                    "sample": self._mark_sample_public(sample),
                    "original_detection": self._mark_detection_public(
                        original_detection
                    ),
                    "candidate_profile": self._mark_profile_public(finalized),
                    "regression": regression,
                    "message": response_message,
                }
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, status=409)
        except Exception as exc:
            logger.error("MARK calibration failed: %s", exc, exc_info=True)
            self._send_json({"error": str(exc)}, status=500)

    def _handle_api_settings_mark_rollback(self):
        try:
            data = self._read_json_body()
            if data is None:
                return
            target_profile_id = int(data.get("profile_id") or 0)
            reason = str(data.get("reason") or "").strip()
            allow_known_regressions = (
                data.get("allow_known_regressions") is True
            )
            if not target_profile_id:
                raise ValueError("缺少回滾 profile id")
            user = self._current_settings_user() or {}
            actor = str(user.get("username") or "")

            with CAPIWebHandler._mark_calibration_lock:
                from capi_mark_calibration import (
                    mark_sample_set_sha256,
                    run_mark_profile_regression,
                )

                current = self.db.get_active_mark_profile()
                target = self.db.get_mark_profile(target_profile_id)
                if not target or target.get("status") != "retired":
                    raise ValueError("只能回滾到曾啟用過的 MARK profile")
                samples = self.db.list_mark_calibration_samples()
                sample_set_sha256 = mark_sample_set_sha256(samples)
                regression = run_mark_profile_regression(
                    samples,
                    target["profile"],
                    profile_id=target["id"],
                )
                rollback_safe = (
                    regression["total"] == 0
                    or bool(regression.get("success"))
                )
                if not rollback_safe and not allow_known_regressions:
                    self._send_json(
                        {
                            "error": (
                                f"目標 v{target_profile_id} 對目前題庫有 "
                                f"{regression['failed']} 筆失敗"
                            ),
                            "requires_force": True,
                            "regression": regression,
                        },
                        status=409,
                    )
                    return
                active = self.db.rollback_mark_profile(
                    target_profile_id,
                    expected_active_profile_id=current["id"],
                    regression_report=regression,
                    sample_count=len(samples),
                    sample_set_sha256=sample_set_sha256,
                    allow_known_regressions=allow_known_regressions,
                    changed_by=actor,
                    reason=reason,
                )
                already_applied = int(active["id"]) == int(current["id"])
                from capi_mark_detector import set_active_mark_profile

                set_active_mark_profile(active["profile"], active["id"])

            logger.warning(
                "[MARK CALIBRATION] rollback target=%s new_profile=%s "
                "failed=%s forced=%s already_applied=%s admin=%s",
                target_profile_id,
                active["id"],
                regression["failed"],
                not rollback_safe,
                already_applied,
                actor,
            )
            self._send_json(
                {
                    "success": True,
                    "active_profile": self._mark_profile_public(active),
                    "regression": regression,
                    "forced": not rollback_safe,
                    "already_applied": already_applied,
                    "message": (
                        f"目前已是回滾版 v{active['id']}"
                        if already_applied
                        else f"已回滾並建立啟用版 v{active['id']}"
                    ),
                }
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, status=409)
        except Exception as exc:
            logger.error("MARK rollback failed: %s", exc, exc_info=True)
            self._send_json({"error": str(exc)}, status=500)

    def _handle_api_settings_user_create(self):
        try:
            data = self._read_json_body()
            if data is None:
                return
            user = self.db.create_settings_user(
                data.get("username", ""),
                data.get("password", ""),
                is_admin=False,
            )
            self._send_json({"success": True, "user": user})
        except ValueError as e:
            self._send_json({"error": str(e)}, status=400)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_api_settings_user_update(self):
        try:
            data = self._read_json_body()
            if data is None:
                return
            user_id = int(data.get("id") or 0)
            if not user_id:
                self._send_json({"error": "缺少帳號 id"}, status=400)
                return
            password = data.get("password")
            user = self.db.update_settings_user(
                user_id,
                username=data.get("username"),
                password=password if password not in (None, "") else None,
            )
            if not user:
                self._send_json({"error": "找不到帳號"}, status=404)
                return
            self._send_json({"success": True, "user": user})
        except ValueError as e:
            self._send_json({"error": str(e)}, status=400)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_api_settings_user_delete(self):
        try:
            data = self._read_json_body()
            if data is None:
                return
            user_id = int(data.get("id") or 0)
            if not user_id:
                self._send_json({"error": "缺少帳號 id"}, status=400)
                return
            if not self.db.delete_settings_user(user_id):
                self._send_json({"error": "找不到帳號"}, status=404)
                return
            self._send_json({"success": True})
        except ValueError as e:
            self._send_json({"error": str(e)}, status=400)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_settings_page(self, path: str):
        """設定管理頁面 (舊版)"""
        template = self.jinja_env.get_template("settings.html")
        html = template.render(
            request_path=path,
            settings_user=self._current_settings_user(),
        )
        self._send_response(200, html)

    def _handle_settings_v2_page(self, path: str):
        """舊版 V2 路由相容別名，改用現存的設定頁。"""
        self._handle_settings_page(path)

    def _handle_api_settings(self):
        """API: 取得所有設定參數"""
        try:
            params = self.db.get_all_config_params() if self.db else []
            params = [
                p for p in params
                if p.get("param_name") != "dust_pixel_grid_max_mask_ratio"
            ]
            server_config = (
                getattr(self._capi_server_instance, "server_config", {}) or {}
            )
            default_location = _default_central_account_location(server_config)
            location_param = next(
                (
                    p for p in params
                    if p.get("param_name") == CENTRAL_ACCOUNT_LOCATION_PARAM
                ),
                None,
            )
            if location_param:
                try:
                    location = _normalize_central_account_location(
                        location_param.get("decoded_value")
                    )
                except ValueError:
                    location = default_location
                location_param.update({
                    "param_value": json.dumps(location, ensure_ascii=False),
                    "param_type": "dict",
                    "description": CENTRAL_ACCOUNT_LOCATION_DESCRIPTION,
                    "decoded_value": location,
                })
            else:
                params.append({
                    "param_name": CENTRAL_ACCOUNT_LOCATION_PARAM,
                    "param_value": json.dumps(default_location, ensure_ascii=False),
                    "param_type": "dict",
                    "description": CENTRAL_ACCOUNT_LOCATION_DESCRIPTION,
                    "updated_at": None,
                    "decoded_value": default_location,
                })
            # 補上 config 中有但 DB 沒有的參數（用目前執行值作為預設）
            if self.inferencer and hasattr(self.inferencer, 'config') and self.inferencer.config:
                import dataclasses
                field_type_by_name = {
                    f.name: f.type
                    for f in dataclasses.fields(self.inferencer.config)
                }
                for p in params:
                    if field_type_by_name.get(p.get("param_name")) is bool:
                        p["param_type"] = "bool"
                existing_names = {p["param_name"] for p in params}
                for f in dataclasses.fields(self.inferencer.config):
                    if f.name in existing_names:
                        continue
                    val = getattr(self.inferencer.config, f.name)
                    # 只處理 JSON 可序列化的基本型別
                    if isinstance(val, (str, int, float, bool)):
                        val_str = str(val)
                        if isinstance(val, bool):
                            param_type = "bool"
                        elif isinstance(val, int):
                            param_type = "int"
                        elif isinstance(val, float):
                            param_type = "float"
                        else:
                            param_type = "str"
                    elif isinstance(val, (dict, list)):
                        try:
                            val_str = json.dumps(val)
                        except (TypeError, ValueError):
                            continue
                        param_type = "dict" if isinstance(val, dict) else "list"
                    else:
                        continue
                    params.append({
                        "param_name": f.name,
                        "param_value": val_str,
                        "param_type": param_type,
                        "updated_at": None,
                    })
            # 附帶 model_resolution_map 給前端產品選擇器使用
            resolution_map = {}
            is_new_arch = False
            if self.inferencer and hasattr(self.inferencer, 'config') and self.inferencer.config:
                resolution_map = getattr(self.inferencer.config, 'model_resolution_map', {})
                is_new_arch = bool(getattr(self.inferencer.config, 'is_new_architecture', False))
            self._send_json({
                "params": params,
                "model_resolution_map": resolution_map,
                "is_new_architecture": is_new_arch,
                "user": self._current_settings_user(),
            })
        except Exception as e:
            self._send_json({"error": str(e)})

    def _handle_api_settings_update(self):
        """API: 更新設定參數"""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)

            param_name = data.get("param_name", "")
            new_value = data.get("new_value")
            reason = data.get("reason", "")

            if not param_name:
                self._send_json({"error": "缺少 param_name"})
                return
            if new_value is None:
                self._send_json({"error": "缺少 new_value"})
                return
            if not reason.strip():
                self._send_json({"error": "請填寫修改原因"})
                return
            user = self._current_settings_user() or {}
            if param_name == CENTRAL_ACCOUNT_LOCATION_PARAM:
                if not user.get("can_manage_accounts"):
                    self._send_json(
                        {"error": "只有 admin 可以修改中心位置"},
                        status=403,
                    )
                    return
                try:
                    new_value = _normalize_central_account_location(new_value)
                except ValueError as e:
                    self._send_json({"error": str(e)}, status=400)
                    return

            # 新架構：threshold_mapping / model_mapping 屬於 bundle yaml 自包含設定，
            # 不接受 /settings 介面動 DB（避免重啟時被 DB 蓋掉 yaml）。請改 /models
            # 介面或直接編輯 machine_config.yaml。
            if param_name in ("threshold_mapping", "model_mapping"):
                is_new_arch = bool(
                    self.inferencer
                    and getattr(self.inferencer.config, "is_new_architecture", False)
                )
                if is_new_arch:
                    self._send_json({
                        "error": (
                            f"新架構 (v2) 不支援透過 /settings 修改 {param_name}。"
                            f"請改 /models 介面，或直接編輯 model/<bundle>/machine_config.yaml。"
                        )
                    })
                    return

            success = self.db.update_config_param(
                param_name,
                new_value,
                reason,
                changed_by=user.get("username", ""),
            )
            if success:
                # Hot-reload 1：把 DB 同步到所有 inferencer.config 屬性（包含單機與
                # 多機新架構 inferencers）。apply_db_overrides 純 setattr、不重載
                # 模型，重複呼叫安全。沒這段時，UI 改完 settings 必須重啟 server
                # 才會生效（aoi_coord_inspection_enabled、grid_tiling_enabled 等
                # 不在下面 edge hot-reload 名單裡的都會卡）。
                try:
                    db_params_list = self.db.get_all_config_params()
                    synced_inferencers = []
                    if hasattr(self, 'inferencer') and self.inferencer is not None:
                        self.inferencer.config.apply_db_overrides(db_params_list)
                        synced_inferencers.append(self.inferencer)
                    server_inst = self._capi_server_instance
                    if server_inst is not None and getattr(server_inst, 'inferencers', None):
                        for inf in server_inst.inferencers.values():
                            if inf in synced_inferencers:
                                continue
                            inf.config.apply_db_overrides(db_params_list)
                            synced_inferencers.append(inf)
                    if synced_inferencers:
                        logger.info(
                            f"[Config Hot-Reload] '{param_name}' synced to "
                            f"{len(synced_inferencers)} inferencer(s)"
                        )
                except Exception as e:
                    logger.warning(f"[Config Hot-Reload] Failed to sync '{param_name}': {e}")

                # Hot-reload 2：Edge inspector 設定有獨立物件樹（EdgeInspectionConfig
                # → EdgeInspector），apply_db_overrides 涵蓋不到，仍需重建。
                edge_triggers = (
                    param_name.startswith("cv_edge")
                    or param_name == "aoi_edge_inspector"
                    or param_name == "aoi_edge_boundary_band_px"
                    or param_name == "aoi_edge_pc_roi_inward_shift_enabled"
                )
                if edge_triggers and hasattr(self, 'inferencer') and self.inferencer:
                    try:
                        from capi_edge_cv import EdgeInspectionConfig
                        db_params = {r["param_name"]: r for r in self.db.get_all_config_params()}
                        edge_cfg = EdgeInspectionConfig.from_db_params(db_params)
                        self.inferencer.update_edge_config(edge_cfg)
                        logger.info(f"[Edge Hot-Reload] CV Edge config synced after updating '{param_name}'")
                    except Exception as e:
                        logger.warning(f"[Edge Hot-Reload] Failed to sync edge config: {e}")
                self._send_json({"success": True, "message": f"已更新 {param_name}"})
            else:
                self._send_json({"error": f"找不到參數: {param_name}"})
        except json.JSONDecodeError:
            self._send_json({"error": "無效的 JSON 格式"})
        except Exception as e:
            self._send_json({"error": str(e)})

    def _handle_api_settings_history(self, query: dict):
        """API: 取得設定修改紀錄"""
        try:
            param_name = query.get("param_name", [""])[0]
            limit = int(query.get("limit", [50])[0])
            history = self.db.get_config_change_history(param_name, limit) if self.db else []
            self._send_json({"history": history})
        except Exception as e:
            self._send_json({"error": str(e)})

    def _handle_api_settings_scratch_bundles(self):
        """API: 取得服務器上已有的刮痕分類器 bundle 檔案"""
        try:
            project_root = Path(__file__).resolve().parent
            current_bundle = ""
            if self.inferencer and getattr(self.inferencer, "config", None):
                current_bundle = str(getattr(self.inferencer.config, "scratch_bundle_path", "") or "")

            scan_dirs = [project_root / "deployment"]
            if current_bundle:
                current_path = Path(current_bundle)
                if not current_path.is_absolute():
                    current_path = project_root / current_path
                scan_dirs.append(current_path.parent)

            bundles = []
            seen = set()
            for scan_dir in scan_dirs:
                if not scan_dir.exists() or not scan_dir.is_dir():
                    continue
                for path in scan_dir.glob("scratch_classifier*.pkl"):
                    resolved = path.resolve()
                    if resolved in seen or not path.is_file():
                        continue
                    seen.add(resolved)
                    try:
                        rel_path = path.relative_to(project_root).as_posix()
                    except ValueError:
                        rel_path = path.as_posix()
                    stat = path.stat()
                    bundles.append({
                        "path": rel_path,
                        "name": path.name,
                        "size": stat.st_size,
                        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                    })

            version_re = re.compile(r"scratch_classifier_v(\d+)\.pkl$", re.IGNORECASE)

            def sort_key(item):
                match = version_re.match(item["name"])
                version = int(match.group(1)) if match else -1
                return (version, item["mtime"], item["name"])

            bundles.sort(key=sort_key, reverse=True)
            self._send_json({"bundles": bundles, "current": current_bundle})
        except Exception as e:
            self._send_json({"error": str(e)})

    @staticmethod
    def _bundle_payload(bundle: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(bundle)
        payload["label"] = Path(str(bundle.get("bundle_path", "") or "")).name or str(bundle.get("machine_id", "") or "")
        return payload

    def _handle_auto_model_switch_api(self):
        """GET /api/auto-model-switch"""
        try:
            db = self._capi_server_instance.database
            bundles = [self._bundle_payload(b) for b in db.list_model_bundles()]
            rules = [dict(r) for r in db.list_auto_model_switch_rules()]
            for rule in rules:
                rule["bundle_label"] = Path(str(rule.get("bundle_path", "") or "")).name
            active = db.get_active_model_bundle()
            self._send_json({
                "bundles": bundles,
                "rules": rules,
                "active_bundle_id": active["id"] if active else None,
            })
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_auto_model_switch_history_api(self, query: dict):
        """GET /api/auto-model-switch/history"""
        try:
            db = self._capi_server_instance.database
            limit = max(1, min(500, int(query.get("limit", [100])[0])))
            series_prefix = (query.get("series_prefix", [""])[0] or "").strip().upper()
            status = (query.get("status", [""])[0] or "").strip().lower()
            history = db.list_auto_model_switch_history(
                limit=limit,
                series_prefix=series_prefix,
                status=status,
            )
            self._send_json({"history": history})
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_auto_model_switch_rule_upsert(self):
        """POST /api/auto-model-switch/rules/upsert"""
        try:
            from capi_auto_model_switch import DEFAULT_SERIES_PREFIX, normalize_series_prefix

            payload = self._read_json_body()
            if payload is None:
                return
            rule_id = payload.get("id")
            series_prefix = str(payload.get("series_prefix", "") or "").strip()
            if payload.get("is_default"):
                series_prefix = DEFAULT_SERIES_PREFIX
            series_prefix = normalize_series_prefix(series_prefix)
            bundle_id = int(payload.get("bundle_id") or 0)
            notes = str(payload.get("notes", "") or "")

            db = self._capi_server_instance.database
            rule = db.upsert_auto_model_switch_rule(
                series_prefix=series_prefix,
                bundle_id=bundle_id,
                notes=notes,
                rule_id=int(rule_id) if rule_id else None,
            )
            rule["bundle_label"] = Path(str(rule.get("bundle_path", "") or "")).name
            self._send_json({"success": True, "rule": rule})
        except json.JSONDecodeError:
            self._send_json({"error": "無效的 JSON 格式"}, status=400)
        except ValueError as e:
            self._send_json({"error": str(e)}, status=400)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_auto_model_switch_rule_delete(self):
        """POST /api/auto-model-switch/rules/delete"""
        try:
            payload = self._read_json_body()
            if payload is None:
                return
            rule_id = int(payload.get("id") or 0)
            if not rule_id:
                self._send_json({"error": "缺少規則 id"}, status=400)
                return
            db = self._capi_server_instance.database
            deleted = db.delete_auto_model_switch_rule(rule_id)
            if not deleted:
                self._send_json({"error": "找不到規則"}, status=404)
                return
            self._send_json({"success": True})
        except json.JSONDecodeError:
            self._send_json({"error": "無效的 JSON 格式"}, status=400)
        except ValueError as e:
            self._send_json({"error": str(e)}, status=400)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _handle_api_settings_reload(self):
        """API: 重新載入設定 (Hot-reload inferencer)"""
        try:
            if not self._capi_server_instance:
                self._send_json({"error": "Server 實例未設定，無法重載"})
                return

            server_inst = self._capi_server_instance
            gpu_lock = self._gpu_lock

            is_new_arch = bool(
                getattr(getattr(server_inst, "fallback_config", None), "is_new_architecture", False)
            )
            logger.info(
                "Settings reload: %s",
                "syncing runtime config from DB" if is_new_arch else "re-initializing inferencer from DB config",
            )

            # 使用 GPU lock 阻止推論期間重建
            if gpu_lock:
                gpu_lock.acquire()

            try:
                if is_new_arch:
                    synced = server_inst.reload_runtime_config_from_db()
                    message = f"設定已重新載入，已同步 {synced} 個推論器（模型未重載）"
                else:
                    server_inst._load_inferencer()
                    message = "設定已重新載入，推論器已重建"
                # 更新 Web handler 的 inferencer 參照
                CAPIWebHandler.inferencer = server_inst.inferencer
                logger.info("Settings reload completed")
                self._send_json({"success": True, "message": message})
            finally:
                if gpu_lock:
                    gpu_lock.release()

        except Exception as e:
            logger.error(f"Settings reload failed: {e}", exc_info=True)
            self._send_json({"error": f"重載失敗: {str(e)}"})

    # ── Rerun inference endpoints ──────────────────────────────────────

    def _handle_rerun_trigger(self, record_id_str: str):
        """API: 觸發重新推論"""
        try:
            record_id = int(record_id_str)
        except ValueError:
            self._send_json({"status": "error", "message": "invalid record_id"})
            return

        if not self.inferencer and not self._capi_server_instance:
            self._send_json({"status": "error", "message": "推論器未載入"})
            return

        with self._rerun_lock:
            task = self._rerun_tasks.get(record_id)
            if task and task["status"] == "running":
                self._send_json({"status": "already_running"})
                return

        detail = self.db.get_record_detail(record_id) if self.db else None
        if not detail:
            self._send_json({"status": "error", "message": f"找不到紀錄 #{record_id}"})
            return

        image_dir = detail.get("image_dir", "")
        if not image_dir or not Path(image_dir).is_dir():
            self._send_json({"status": "error", "message": f"圖片目錄不存在: {image_dir}"})
            return

        with self._rerun_lock:
            # 再次檢查防止 TOCTOU race
            task = self._rerun_tasks.get(record_id)
            if task and task["status"] == "running":
                self._send_json({"status": "already_running"})
                return
            self._rerun_tasks[record_id] = {"status": "running", "message": "正在準備推論..."}

        thread = threading.Thread(
            target=CAPIWebHandler._rerun_worker,
            args=(record_id, detail),
            daemon=True,
        )
        thread.start()

        self._send_json({"status": "started", "record_id": record_id})

    @classmethod
    def _rerun_worker(cls, record_id: int, detail: dict):
        """背景執行緒：重新推論並覆蓋紀錄"""
        import sys as _sys
        import time as _time
        from capi_server import (
            results_to_db_data, aggregate_judgment, append_cv_edge_to_judgment,
            InferenceLogCapture, WITHIN_SPEC_LOGS_URL,
            _stored_machine_judgment_for_record,
        )

        def _update_status(msg, *_):
            with cls._rerun_lock:
                task = cls._rerun_tasks.get(record_id)
                if task:
                    if isinstance(msg, int) and _:
                        task["message"] = f"推論中 {msg}/{_[0]}..."
                    else:
                        task["message"] = str(msg)

        try:
            panel_dir = Path(detail["image_dir"])
            model_id = detail.get("model_id", "")
            inferencer = cls.inferencer
            server_inst = cls._capi_server_instance
            if server_inst is not None and hasattr(server_inst, "_get_or_create_inferencer"):
                inferencer = server_inst._get_or_create_inferencer(model_id) or inferencer
            if inferencer is None:
                raise RuntimeError("推論器未載入")

            resolution = None
            if detail.get("resolution_x") and detail.get("resolution_y"):
                resolution = (detail["resolution_x"], detail["resolution_y"])

            bomb_info = None
            if detail.get("client_bomb_info"):
                try:
                    bomb_info = json.loads(detail["client_bomb_info"])
                except (json.JSONDecodeError, TypeError):
                    pass

            _update_status("正在等待 GPU...")
            start_time = _time.time()

            InferenceLogCapture.start_capture()
            if cls._gpu_lock:
                with cls._gpu_lock:
                    _update_status("正在推論中...")
                    panel_result = inferencer.process_panel(
                        panel_dir,
                        progress_callback=_update_status,
                        product_resolution=resolution,
                        bomb_info=bomb_info,
                        model_id=model_id,
                        machine_no=detail.get("machine_no"),
                        machine_judgment=detail.get("machine_judgment"),
                    )
            else:
                _update_status("正在推論中...")
                panel_result = inferencer.process_panel(
                    panel_dir,
                    progress_callback=_update_status,
                    product_resolution=resolution,
                    bomb_info=bomb_info,
                    model_id=model_id,
                    machine_no=detail.get("machine_no"),
                    machine_judgment=detail.get("machine_judgment"),
                )

            processing_seconds = _time.time() - start_time

            results = panel_result[0]
            omit_overexposed = panel_result[2] if len(panel_result) > 2 else False
            omit_overexposure_info = panel_result[3] if len(panel_result) > 3 else ""
            is_duplicate = panel_result[4] if len(panel_result) > 4 else False
            omit_image_raw = panel_result[5] if len(panel_result) > 5 else None
            aoi_report = panel_result[6] if len(panel_result) > 6 else {}

            if is_duplicate:
                logger.warning(
                    f"[RERUN] [DUPLICATE_PANEL] record_id={record_id} "
                    f"重複投片，已依建立時間選取最新圖片推論"
                )

            if not results:
                InferenceLogCapture.stop_capture()
                with cls._rerun_lock:
                    cls._rerun_tasks[record_id] = {"status": "error", "message": "推論完成但無圖片結果"}
                return

            ai_judgment, ng_details = aggregate_judgment(results)
            for result in results:
                if hasattr(result, 'edge_defects') and result.edge_defects:
                    ai_judgment, ng_details = append_cv_edge_to_judgment(
                        ai_judgment, ng_details, result.edge_defects, result.image_path.stem
                    )

            within_spec_info = None
            if ai_judgment.startswith("NG"):
                parsed_for_within_spec = {
                    "glass_id": detail.get("glass_id", ""),
                    "model_id": model_id,
                    "machine_no": detail.get("machine_no", ""),
                    "machine_judgment": detail.get("machine_judgment", ""),
                }
                if server_inst is not None and hasattr(server_inst, "_evaluate_within_spec_for_inference"):
                    within_spec_info = server_inst._evaluate_within_spec_for_inference(
                        parsed_for_within_spec,
                        results,
                        inferencer,
                    )
                else:
                    from capi_config import CAPIConfig

                    started = _time.time()
                    eval_detail = {
                        "model_id": model_id,
                        "machine_id": model_id,
                        "machine_no": detail.get("machine_no", ""),
                        "images": results_to_db_data(results, {}),
                        "source": "inference",
                    }
                    _attach_runtime_dust_masks_to_within_spec_detail(eval_detail, results)
                    _attach_no_detect_regions_to_within_spec_detail(eval_detail, inferencer, model_id)
                    rules = getattr(getattr(inferencer, "config", None), "within_spec_judgment_rules", None)
                    if not rules:
                        rules = CAPIConfig().within_spec_judgment_rules
                    visual_dir, visual_prefix = _within_spec_auto_visual_output(
                        str(getattr(getattr(cls, "heatmap_manager", None), "base_dir", "") or cls.heatmap_base_dir or ""),
                        detail.get("glass_id", ""),
                        record_id,
                    )
                    eval_result = _evaluate_within_spec_suggestion_detail(
                        eval_detail,
                        CAPIConfig._normalize_within_spec_judgment_rules(rules),
                        model_id,
                        visual_output_dir=visual_dir,
                        visual_url_prefix=visual_prefix,
                        rotate_180=bool(getattr(inferencer, "_rotate_detection_images_180", False)),
                    )
                    suggestion = eval_result.get("suggestion")
                    panel_totals = eval_result.get("panel_totals") or []
                    panel_within = bool(panel_totals) and all(bool(item.get("within")) for item in panel_totals)
                    converted = bool(suggestion and suggestion.get("suggested") and panel_within)
                    panel_reason = _format_within_spec_panel_summary(eval_result)
                    if converted:
                        status = "within_spec"
                        reason = panel_reason or suggestion.get("reason", "")
                    elif suggestion and suggestion.get("suggested"):
                        status = "not_within_spec"
                        reason = panel_reason or "部分項目符合規格內，但整片 PANEL 尚有項目未符合"
                    elif panel_totals:
                        status = "not_within_spec"
                        reason = panel_reason or "未符合規格內條件"
                    else:
                        status = "not_evaluable"
                        reason = "未取得可比對的規格內點數結果"
                    saved_suggestion = suggestion if converted else None
                    if saved_suggestion and reason:
                        saved_suggestion = dict(saved_suggestion)
                        saved_suggestion["reason"] = reason
                    eval_result["source"] = "inference"
                    eval_result["inference_context"] = parsed_for_within_spec
                    eval_result["inference_auto_decision"] = {
                        "converted_to_ok_i": converted,
                        "status": status,
                        "reason": reason,
                        "requires_all_panel_totals_within": True,
                    }
                    within_spec_info = {
                        "suggestion": saved_suggestion,
                        "raw_suggestion": suggestion,
                        "detail": eval_result,
                        "processing_seconds": _time.time() - started,
                        "converted": converted,
                        "status": status,
                        "reason": reason,
                    }

                if within_spec_info and within_spec_info.get("converted"):
                    ai_judgment = "OK-i"

            _update_status("正在儲存 heatmap...")
            heatmap_info = {}
            if cls.heatmap_manager:
                old_heatmap_dir = detail.get("heatmap_dir", "")
                if old_heatmap_dir and Path(old_heatmap_dir).is_dir():
                    import shutil
                    try:
                        shutil.rmtree(old_heatmap_dir)
                    except Exception:
                        pass

                heatmap_info = cls.heatmap_manager.save_panel_heatmaps(
                    glass_id=detail["glass_id"],
                    results=results,
                    inferencer=inferencer,
                    save_overview=True,
                    save_tile_detail=True,
                    omit_image=omit_image_raw,
                )

            _update_status("正在更新資料庫...")
            image_results_data = results_to_db_data(results, heatmap_info) if results else []
            total_images = len(image_results_data)
            ng_images = sum(1 for d in image_results_data if d.get("is_ng"))
            stored_machine_judgment = _stored_machine_judgment_for_record(
                detail.get("machine_judgment", ""),
                results,
                aoi_report,
            )
            aoi_machine_coords = ""
            if aoi_report:
                aoi_coords_data = {}
                for prefix, defects in aoi_report.items():
                    aoi_coords_data[prefix] = [
                        {"defect_code": d.defect_code, "product_x": d.product_x, "product_y": d.product_y}
                        for d in defects
                    ]
                aoi_machine_coords = json.dumps(aoi_coords_data, ensure_ascii=False)
            if is_duplicate:
                dup_note = "[DUPLICATE_PANEL] 重複投片，已依建立時間選取最新圖片推論"
                err_suffix = ai_judgment if ai_judgment.startswith("ERR") else ""
                error_message = f"{dup_note}\n{err_suffix}".strip()
            else:
                error_message = ai_judgment if ai_judgment.startswith("ERR") else ""

            inference_log = InferenceLogCapture.stop_capture()
            if within_spec_info:
                within_spec_note = _format_within_spec_inference_note(within_spec_info, WITHIN_SPEC_LOGS_URL)
                inference_log = f"{(inference_log or '').rstrip()}\n{within_spec_note}".strip()
            from capi_image_preprocess_lab import summarize_preprocess_timings
            preprocess_timing = summarize_preprocess_timings(
                getattr(r, "preprocess_steps", []) for r in results
            )

            cls.db.update_record_for_rerun(
                record_id=record_id,
                ai_judgment=ai_judgment,
                total_images=total_images,
                ng_images=ng_images,
                ng_details=ng_details,
                processing_seconds=processing_seconds,
                heatmap_dir=heatmap_info.get("dir", ""),
                error_message=error_message,
                machine_judgment=stored_machine_judgment,
                aoi_machine_coords=aoi_machine_coords if aoi_report else None,
                image_results_data=image_results_data,
                inference_log=inference_log,
                omit_overexposed=int(omit_overexposed),
                omit_overexposure_info=omit_overexposure_info if omit_overexposure_info else "",
                image_preprocess_pipeline=getattr(
                    getattr(inferencer, "config", None),
                    "image_preprocess_pipeline",
                    [],
                ),
                image_preprocess_pipelines=getattr(
                    getattr(inferencer, "config", None),
                    "image_preprocess_pipelines",
                    {},
                ),
                image_preprocess_timing=preprocess_timing,
            )

            if within_spec_info:
                cls.db.save_within_spec_review_log(
                    client_record_id=None,
                    inference_record_id=record_id,
                    suggestion=within_spec_info.get("suggestion"),
                    detail=within_spec_info.get("detail") or {},
                    processing_seconds=within_spec_info.get("processing_seconds", 0.0),
                    error_message=within_spec_info.get("error_message", ""),
                    source="inference",
                )

            with cls._rerun_lock:
                cls._rerun_tasks[record_id] = {"status": "done", "message": "完成"}

        except Exception as e:
            import traceback
            InferenceLogCapture.stop_capture()
            traceback.print_exc()
            with cls._rerun_lock:
                cls._rerun_tasks[record_id] = {"status": "error", "message": f"推論失敗: {e}"}

    # ==== Dataset Export Endpoints ====

    @classmethod
    def _dataset_export_worker(cls, days: int, include_true_ng: bool,
                               skip_existing: bool, output_dir: str):
        """背景執行 DatasetExporter.run()"""
        state = cls._dataset_export_state
        try:
            server_inst = cls._capi_server_instance
            path_mapping = getattr(server_inst, "path_mapping", {}) if server_inst else {}

            def status_callback(current, total, last_glass_id):
                with state["lock"]:
                    if state["current_job"]:
                        state["current_job"]["current"] = current
                        state["current_job"]["total"] = total
                        state["current_job"]["last_glass_id"] = last_glass_id

            exporter = DatasetExporter(
                db=cls.db,
                base_dir=output_dir,
                path_mapping=path_mapping,
                rotate_180=bool(
                    getattr(getattr(cls.inferencer, "config", None), "inference_rotate_180_enabled", False)
                ),
            )
            summary = exporter.run(
                days=days, include_true_ng=include_true_ng,
                skip_existing=skip_existing,
                status_callback=status_callback,
                cancel_event=state["cancel_event"],
            )
            with state["lock"]:
                if state["current_job"]:
                    state["current_job"]["state"] = (
                        JOB_STATE_CANCELLED if state["cancel_event"].is_set() else JOB_STATE_COMPLETED
                    )
                state["last_summary"] = summary
        except Exception as e:
            logger.exception("dataset_export worker failed")
            with state["lock"]:
                if state["current_job"]:
                    state["current_job"]["state"] = JOB_STATE_FAILED
                    state["current_job"]["error"] = str(e)
        finally:
            with state["lock"]:
                state["cancel_event"].clear()

    def _handle_dataset_export_start(self):
        """POST /api/dataset_export/start"""
        import json as _json
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            data = _json.loads(body) if body else {}
        except Exception:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return

        days = int(data.get("days", 3))
        include_true_ng = bool(data.get("include_true_ng", True))
        skip_existing = bool(data.get("skip_existing", True))

        # 決定 output_dir
        server_inst = self._capi_server_instance
        default_cfg = {}
        if server_inst:
            default_cfg = server_inst.server_config.get("dataset_export", {})
        output_dir = data.get("output_dir") or default_cfg.get("base_dir") or "./datasets/over_review"
        min_free_gb = float(default_cfg.get("min_free_space_gb", 1))

        state = self._dataset_export_state
        with state["lock"]:
            if state["current_job"] and state["current_job"].get("state") == JOB_STATE_RUNNING:
                self._send_json({
                    "error": "job_already_running",
                    "current_job_id": state["current_job"].get("job_id"),
                }, status=409)
                return

            # 磁碟空間檢查
            try:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                free_bytes = shutil.disk_usage(str(output_path)).free
                if free_bytes < min_free_gb * (1024 ** 3):
                    self._send_json({
                        "error": "insufficient_disk_space",
                        "free_gb": round(free_bytes / (1024 ** 3), 2),
                        "required_gb": min_free_gb,
                    }, status=409)
                    return
            except Exception as e:
                self._send_json({"error": f"cannot access output_dir: {e}"}, status=400)
                return

            job_id = datetime.now().strftime("job_%Y%m%d_%H%M%S")
            state["cancel_event"].clear()
            state["current_job"] = {
                "job_id": job_id,
                "state": JOB_STATE_RUNNING,
                "current": 0,
                "total": 0,
                "last_glass_id": "",
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }

        thread = threading.Thread(
            target=CAPIWebHandler._dataset_export_worker,
            args=(days, include_true_ng, skip_existing, output_dir),
            daemon=True,
            name=f"dataset-export-{job_id}",
        )
        thread.start()
        self._send_json({"job_id": job_id, "started_at": state["current_job"]["started_at"]})

    def _handle_dataset_export_status(self):
        """GET /api/dataset_export/status"""
        state = self._dataset_export_state
        with state["lock"]:
            job = state["current_job"]
            if not job:
                self._send_json({"state": JOB_STATE_IDLE})
                return
            resp = dict(job)
            if resp.get("started_at"):
                try:
                    started = datetime.fromisoformat(resp["started_at"])
                    resp["elapsed_sec"] = round((datetime.now() - started).total_seconds(), 1)
                except Exception:
                    resp["elapsed_sec"] = 0
            self._send_json(resp)

    def _handle_dataset_export_summary(self, job_id: str):
        """GET /api/dataset_export/summary/<job_id>"""
        from dataclasses import asdict as _asdict
        state = self._dataset_export_state
        with state["lock"]:
            summary = state["last_summary"]
            if not summary or summary.job_id != job_id:
                self._send_json({"error": "not_found"}, status=404)
                return
            self._send_json(_asdict(summary))

    def _handle_dataset_export_cancel(self):
        """POST /api/dataset_export/cancel"""
        state = self._dataset_export_state
        with state["lock"]:
            if not state["current_job"] or state["current_job"].get("state") != JOB_STATE_RUNNING:
                self._send_json({"error": "no_running_job"}, status=404)
                return
            state["cancel_event"].set()
        self._send_json({"ok": True})

    def _export_base_dir(self, config_key: str, default: str) -> Path:
        """讀 server_config.<config_key>.base_dir，fallback 至 default，回傳 resolve 後的 Path"""
        server_inst = self._capi_server_instance
        cfg = {}
        if server_inst:
            cfg = server_inst.server_config.get(config_key, {})
        return Path(cfg.get("base_dir") or default).resolve()

    def _dataset_export_base_dir(self) -> Path:
        return self._export_base_dir("dataset_export", "./datasets/over_review")

    _JOB_ID_RE = __import__("re").compile(r"^[A-Za-z0-9_]+$")

    def _dataset_list_jobs(self) -> list:
        """回傳 base_dir 下所有 job 資料夾名稱（字串），依名稱降冪（最新在前）"""
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

    # ── Debug: 批次 Scratch 分類器驗證 ─────────────────
    _scratch_batch_runner = None
    _scratch_batch_lock = threading.Lock()

    @classmethod
    def _get_scratch_batch_runner(cls):
        with cls._scratch_batch_lock:
            if cls._scratch_batch_runner is None:
                cache_dir = Path("reports") / "scratch_batch"
                cls._scratch_batch_runner = ScratchBatchRunner(
                    inferencer=cls.inferencer,
                    gpu_lock=cls._gpu_lock,
                    cache_dir=cache_dir,
                )
            return cls._scratch_batch_runner

    def _handle_scratch_batch_page(self, path: str, query: dict):
        jobs = self._dataset_list_jobs()
        runner = self._get_scratch_batch_runner()
        recent = runner.list_recent(limit=10)
        template = self.jinja_env.get_template("debug_scratch_batch.html")
        html = template.render(
            request_path=path,
            jobs=jobs,
            recent_tasks=[t.to_status_dict() for t in recent],
            positive_label=SCRATCH_POSITIVE_LABEL,
        )
        self._send_response(200, html)

    def _handle_scratch_batch_jobs(self):
        self._send_json({"jobs": self._dataset_list_jobs()})

    def _handle_scratch_batch_start(self):
        data = self._read_json_body()
        if data is None:
            return
        job_id = (data.get("job_id") or "").strip()
        if not job_id:
            self._send_json({"error": "缺少 job_id"}, status=400)
            return
        job_dir = self._dataset_resolve_job_dir(job_id)
        if job_dir is None:
            self._send_json({"error": f"無效的 job：{job_id}"}, status=404)
            return
        if self.inferencer is None:
            self._send_json({"error": "Inferencer 未初始化，無法執行分類器"}, status=500)
            return

        runner = self._get_scratch_batch_runner()
        try:
            task = runner.start(job_id, job_dir)
        except RuntimeError as e:
            self._send_json({"error": str(e)}, status=409)
            return
        self._send_json(task.to_status_dict())

    def _handle_scratch_batch_cancel(self):
        data = self._read_json_body()
        if data is None:
            return
        task_id = (data.get("task_id") or "").strip()
        if not task_id:
            self._send_json({"error": "缺少 task_id"}, status=400)
            return
        ok = self._get_scratch_batch_runner().cancel(task_id)
        self._send_json({"cancelled": bool(ok)})

    def _handle_scratch_batch_status(self, query: dict):
        tid = (query.get("task_id", [""])[0] if isinstance(query.get("task_id"), list) else query.get("task_id")) or ""
        runner = self._get_scratch_batch_runner()
        task = runner.get(tid) if tid else runner.current()
        if task is None:
            self._send_json({"state": "not_found"})
            return
        self._send_json(task.to_status_dict())

    def _handle_scratch_batch_result(self, query: dict):
        tid = (query.get("task_id", [""])[0] if isinstance(query.get("task_id"), list) else query.get("task_id")) or ""
        if not tid:
            self._send_json({"error": "缺少 task_id"}, status=400)
            return
        runner = self._get_scratch_batch_runner()
        task = runner.get(tid)
        if task is None:
            self._send_json({"error": "找不到該任務"}, status=404)
            return
        status = task.to_status_dict()
        results = [
            {
                "sample_id": r.sample_id,
                "label": r.label,
                "is_positive": r.is_positive,
                "score": r.score,
                "crop_path": r.crop_path,
                "glass_id": r.glass_id,
                "image_name": r.image_name,
                "over_review_category": r.over_review_category,
            }
            for r in task.results
        ]
        default_summary = scratch_batch_summary(task.results, task.effective_threshold)
        self._send_json({
            "status": status,
            "results": results,
            "default_summary": default_summary,
            "positive_label": SCRATCH_POSITIVE_LABEL,
        })

    def _handle_scratch_batch_export(self, query: dict):
        import csv as _csv
        import io as _io
        tid = (query.get("task_id", [""])[0] if isinstance(query.get("task_id"), list) else query.get("task_id")) or ""
        try:
            thr = float((query.get("threshold", [""])[0] if isinstance(query.get("threshold"), list) else query.get("threshold")) or 0)
        except Exception:
            thr = 0.0
        if not tid:
            self._send_json({"error": "缺少 task_id"}, status=400)
            return
        task = self._get_scratch_batch_runner().get(tid)
        if task is None:
            self._send_json({"error": "找不到該任務"}, status=404)
            return
        if thr <= 0:
            thr = task.effective_threshold

        buf = _io.StringIO()
        w = _csv.writer(buf)
        w.writerow([
            "sample_id", "label", "is_positive", "score", "flipped",
            "judgment", "glass_id", "image_name", "over_review_category", "crop_path",
        ])
        for r in task.results:
            flipped = r.score > thr
            if r.is_positive:
                judgment = "TP" if flipped else "FN"
            else:
                judgment = "FP (leak)" if flipped else "TN"
            w.writerow([
                r.sample_id, r.label, int(r.is_positive), f"{r.score:.6f}",
                int(flipped), judgment, r.glass_id, r.image_name,
                r.over_review_category, r.crop_path,
            ])
        content = buf.getvalue()
        content_bytes = ("\ufeff" + content).encode("utf-8")
        filename = f"scratch_batch_{tid}.csv"
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", f"attachment; filename=\"{filename}\"")
        self.send_header("Content-Length", str(len(content_bytes)))
        self.end_headers()
        self.wfile.write(content_bytes)

    def _read_json_body(self) -> Optional[dict]:
        """讀 POST JSON body；失敗回 None 並已送出 400 response"""
        import json as _json
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            return _json.loads(body) if body else {}
        except Exception:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return None

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

    def _handle_dataset_sample_batch_delete(self):
        """POST /api/dataset_export/sample/batch_delete  body: {job, sample_ids: [...]}"""
        data = self._read_json_body()
        if data is None:
            return
        job_id = (data.get("job") or "").strip()
        sample_ids = data.get("sample_ids")
        if not job_id or not isinstance(sample_ids, list) or not sample_ids:
            self._send_json({"error": "missing job or sample_ids"}, status=400)
            return

        job_dir = self._dataset_resolve_job_dir(job_id)
        if job_dir is None:
            self._send_json({"error": "invalid job"}, status=404)
            return

        manifest_path = job_dir / "manifest.csv"
        state = self._dataset_export_state
        deleted = []
        not_found = []
        with state["manifest_lock"]:
            manifest = read_manifest(manifest_path)
            for sid in sample_ids:
                sid = (sid or "").strip()
                if not sid:
                    continue
                if delete_sample(job_dir, manifest, sid):
                    deleted.append(sid)
                else:
                    not_found.append(sid)
            if deleted:
                write_manifest(manifest_path, manifest)
        self._send_json({
            "ok": True,
            "deleted": deleted,
            "not_found": not_found,
            "deleted_count": len(deleted),
        })

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

    def _handle_rerun_status_sse(self, record_id_str: str):
        """SSE: 串流重跑進度"""
        import time as _time

        try:
            record_id = int(record_id_str)
        except ValueError:
            self._send_json({"status": "error", "message": "invalid record_id"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def sse_send(event_type, data):
            msg = f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            self.wfile.write(msg.encode("utf-8"))
            self.wfile.flush()

        last_msg = ""
        try:
            while True:
                with self._rerun_lock:
                    task = self._rerun_tasks.get(record_id)

                if not task:
                    sse_send("status", {"message": "idle"})
                    break

                if task["message"] != last_msg:
                    sse_send("status", {"message": task["message"]})
                    last_msg = task["message"]

                if task["status"] == "done":
                    sse_send("done", {"message": "完成", "record_id": record_id})
                    with self._rerun_lock:
                        self._rerun_tasks.pop(record_id, None)
                    break
                elif task["status"] == "error":
                    sse_send("error", {"message": task["message"]})
                    with self._rerun_lock:
                        self._rerun_tasks.pop(record_id, None)
                    break

                _time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _build_training_cards(self):
        """Build list of model card dicts for the /training hub."""
        cards = []
        server_inst = self._capi_server_instance
        cfg = getattr(server_inst, "config", None) if server_inst else None

        bundle_path = ""
        if cfg:
            bundle_path = getattr(cfg, "scratch_bundle_path", "") or ""

        status = "warning"
        status_text = "未設定"
        trained_at = "—"
        if bundle_path:
            p = Path(bundle_path)
            if not p.exists():
                status_text = "找不到檔案"
            else:
                try:
                    import pickle
                    with open(p, "rb") as f:
                        bundle = pickle.load(f)
                    trained_at = bundle.get("metadata", {}).get("trained_at") or "未知"
                    status = "ok"
                    status_text = "已部署"
                except Exception:
                    status_text = "讀取失敗"

        cards.append({
            "title": "刮痕分類器",
            "subtitle": "Scratch Classifier",
            "description": "DINOv2 + LoRA，用於 over-review 的二次刮痕判定。從歷史紀錄收集標註樣本後重訓。",
            "bundle_path": bundle_path or "(未設定)",
            "trained_at": trained_at,
            "status": status,
            "status_text": status_text,
            "target_url": "/retrain",
        })

        # 新機種 PatchCore card
        db = server_inst.database if server_inst else None
        bundles = db.list_model_bundles() if db else []
        new_arch_count = len(bundles)
        active_count = sum(1 for b in bundles if b.get("is_active"))

        cards.append({
            "title": "新機種 PatchCore",
            "subtitle": "C-10 (5 lighting × inner+edge)",
            "description": f"訓練完整 10 個 PT，或針對既有 bundle 選擇指定 PT 重新選圖重訓。已啟用 {active_count} 個 / 共 {new_arch_count} bundle。",
            "bundle_path": "model/<機種>-<日期>",
            "trained_at": "—" if new_arch_count == 0 else "詳見模型庫",
            "status": "ok" if active_count > 0 else "warning",
            "status_text": f"{active_count} 啟用" if active_count else "未訓練",
            "target_url": "/train/new",
        })

        return cards

    def _build_submodel_retrain_choices(self):
        """Build bundle/unit choices for single PatchCore submodel retraining."""
        from capi_train_new import (
            TRAINING_UNITS,
            WIZARD_FULL_PANEL_COUNT,
            WIZARD_TOTAL_PANEL_COUNT,
        )

        server_inst = self._capi_server_instance
        db = server_inst.database if server_inst else None

        bundles = []
        if db:
            for b in db.list_model_bundles():
                bundle_path = str(b.get("bundle_path") or "")
                bundle_name = Path(bundle_path).name or f"bundle-{b.get('id')}"
                is_active = bool(b.get("is_active"))
                bundles.append({
                    "id": b.get("id"),
                    "machine_id": b.get("machine_id") or "",
                    "bundle_path": bundle_path,
                    "bundle_name": bundle_name,
                    "trained_at": b.get("trained_at") or "",
                    "panel_count": b.get("panel_count") or 0,
                    "inner_tile_count": b.get("inner_tile_count") or 0,
                    "edge_tile_count": b.get("edge_tile_count") or 0,
                    "ng_tile_count": b.get("ng_tile_count") or 0,
                    "is_active": is_active,
                    "job_id": b.get("job_id") or "",
                    "label": (
                        f"{b.get('machine_id') or '(unknown)'} / {bundle_name}"
                        + (" (啟用中)" if is_active else "")
                    ),
                })

        # 保持 DB 的 trained_at DESC，再把 active bundle 穩定排到前面。
        bundles.sort(key=lambda x: x["trained_at"] or "", reverse=True)
        bundles.sort(key=lambda x: not x["is_active"])

        return {
            "bundles": bundles,
            "units": [
                {"lighting": lighting, "zone": zone, "label": f"{lighting}-{zone}"}
                for lighting, zone in TRAINING_UNITS
            ],
            "full_panel_count": WIZARD_FULL_PANEL_COUNT,
            "total_panel_count": WIZARD_TOTAL_PANEL_COUNT,
        }

    @staticmethod
    def _all_train_unit_labels() -> list:
        from capi_train_new import TRAINING_UNITS
        return [f"{lighting}-{zone}" for lighting, zone in TRAINING_UNITS]

    @classmethod
    def _machine_id_prefix(cls, machine_id: str) -> str:
        return str(machine_id or "").strip()[:cls.TRAIN_NEW_MACHINE_PREFIX_LEN]

    @staticmethod
    def _product_resolution_for_machine(machine_id: str, server_inst=None) -> Tuple[int, int]:
        default_resolution = (1920, 1080)
        resolution_map = {
            "B": (1366, 768),
            "H": (1920, 1080),
            "J": (1920, 1200),
            "K": (2560, 1440),
            "G": (2560, 1600),
        }
        if server_inst is not None:
            inferencers = getattr(server_inst, "inferencers", None)
            inferencer = inferencers.get(machine_id) if hasattr(inferencers, "get") else None
            if inferencer is not None:
                configured_map = getattr(inferencer.config, "model_resolution_map", None)
                if configured_map:
                    resolution_map = configured_map

        if machine_id and len(machine_id) >= 6:
            code = machine_id[5].upper()
            try:
                res = resolution_map.get(code)
            except AttributeError:
                res = None
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                return int(res[0]), int(res[1])
        return default_resolution

    @staticmethod
    def _training_bomb_rotate_180(machine_id: str, server_inst=None) -> bool:
        """Return the source-image orientation used by the matching machine inferencer."""
        if server_inst is None:
            return False
        inferencers = getattr(server_inst, "inferencers", None)
        inferencer = inferencers.get(machine_id) if hasattr(inferencers, "get") else None
        config = getattr(inferencer, "config", None)
        return bool(getattr(config, "inference_rotate_180_enabled", False))

    @staticmethod
    def _sample_ng_tiles_compat(sample_ng_tiles_fn, preprocess_cfg=None, log=print, **kwargs):
        """Call sample_ng_tiles across mixed capi_web/capi_train_new deployments."""
        import inspect

        call_kwargs = dict(kwargs)
        call_kwargs["log"] = log
        supports_preprocess_cfg = False
        try:
            signature = inspect.signature(sample_ng_tiles_fn)
            parameters = signature.parameters
            supports_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            supports_preprocess_cfg = "preprocess_cfg" in parameters
            if not supports_kwargs:
                for optional_name in (
                    "machine_id", "rotate_180", "ng_validation_base_dir",
                ):
                    if optional_name not in parameters:
                        call_kwargs.pop(optional_name, None)
        except (TypeError, ValueError):
            supports_preprocess_cfg = False
        if supports_preprocess_cfg:
            call_kwargs["preprocess_cfg"] = preprocess_cfg
        elif preprocess_cfg is not None and getattr(preprocess_cfg, "image_preprocess_pipeline", None):
            log("⚠ NG 前處理略過：capi_train_new.sample_ng_tiles 版本不支援 preprocess_cfg，請同步更新 capi_train_new.py")
        return sample_ng_tiles_fn(**call_kwargs)

    @classmethod
    def _same_machine_family(cls, machine_a: str, machine_b: str) -> bool:
        prefix_a = cls._machine_id_prefix(machine_a)
        prefix_b = cls._machine_id_prefix(machine_b)
        return bool(prefix_a and prefix_a == prefix_b)

    @classmethod
    def _normalize_train_new_scope(cls, raw, db=None) -> tuple:
        """Validate and normalize train scope for the 6-step wizard.

        Returns (scope, error).  The default is full 10-unit training.
        """
        all_units = cls._all_train_unit_labels()
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            return None, "training_scope must be an object"

        mode = str(raw.get("mode") or raw.get("training_mode") or "full")
        if mode not in ("full", "partial"):
            return None, "training_scope.mode must be 'full' or 'partial'"

        if mode == "full":
            return {
                "mode": "full",
                "selected_units": all_units,
                "target_bundle_id": None,
            }, None

        selected = raw.get("selected_units")
        if not isinstance(selected, list) or not selected:
            return None, "training_scope.selected_units must be a non-empty list"

        clean_units = []
        for unit in selected:
            unit = str(unit)
            if unit not in all_units:
                return None, f"invalid selected unit: {unit}"
            if unit not in clean_units:
                clean_units.append(unit)

        try:
            target_bundle_id = int(raw.get("target_bundle_id"))
        except (TypeError, ValueError):
            return None, "training_scope.target_bundle_id is required for partial training"

        if db is not None and not db.get_model_bundle(target_bundle_id):
            return None, "target bundle not found"

        return {
            "mode": "partial",
            "selected_units": clean_units,
            "target_bundle_id": target_bundle_id,
        }, None

    @staticmethod
    def _scope_selected_units(scope: Optional[dict]) -> list:
        from capi_train_new import TRAINING_UNITS
        labels = (scope or {}).get("selected_units") or [
            f"{lighting}-{zone}" for lighting, zone in TRAINING_UNITS
        ]
        out = []
        for label in labels:
            try:
                lighting, zone = str(label).rsplit("-", 1)
            except ValueError:
                continue
            out.append((lighting, zone))
        return out

    @classmethod
    def _scope_selected_lightings(cls, scope: Optional[dict]) -> list:
        out = []
        for lighting, _zone in cls._scope_selected_units(scope):
            if lighting not in out:
                out.append(lighting)
        return out

    @staticmethod
    def _train_new_lighting_labels(selected_lightings: list, panel_paths: list) -> dict:
        labels = {lighting: lighting for lighting in selected_lightings}
        if "STANDARD" not in labels:
            return labels

        for panel_path in panel_paths:
            try:
                source_labels = image_prefix_display_labels(
                    entry.name for entry in Path(panel_path).iterdir() if entry.is_file()
                )
            except OSError:
                continue
            if source_labels.get("STANDARD") != "STANDARD":
                for lighting in labels:
                    if lighting in source_labels:
                        labels[lighting] = source_labels[lighting]
                return labels
        return labels

    @classmethod
    def _dashboard_lighting_labels(cls, db, selected_lightings: list) -> dict:
        labels = {lighting: lighting for lighting in selected_lightings}
        if db is None:
            return labels
        try:
            active_bundle = db.get_active_model_bundle()
            job_id = (active_bundle or {}).get("job_id")
            job = db.get_training_job(job_id) if job_id else None
        except Exception:
            return labels
        if not job:
            return labels
        return cls._train_new_lighting_labels(
            selected_lightings,
            job.get("panel_paths") or [],
        )

    def _handle_training_page(self):
        """GET /training - hub page listing trainable models."""
        template = self.jinja_env.get_template("training.html")
        html = template.render(
            request_path="/training",
            model_cards=self._build_training_cards(),
        )
        self._send_response(200, html)

    def _list_open_train_new_jobs(self):
        db = self._capi_server_instance.database
        all_active = db.list_active_training_jobs()
        # 把 stale job 補刀（preprocess/train 但 worker 已死）
        cleaned = []
        for j in all_active:
            j = self._mark_train_new_stale_if_needed(db, j)
            if j and j["state"] in ("preprocess", "review", "train"):
                cleaned.append(j)
        return cleaned

    def _handle_train_new_scope_page(self):
        """GET /train/new — Step 1 / 6: choose full training or selected PTs."""
        choices = self._build_submodel_retrain_choices()
        template = self.jinja_env.get_template("train_new/step1_scope.html")
        html = template.render(
            request_path="/train/new",
            active_jobs=self._list_open_train_new_jobs(),
            bundles=choices["bundles"],
            units=choices["units"],
        )
        self._send_response(200, html)

    def _handle_train_new_page(self):
        """Backward-compatible alias for tests/callers that still invoke the old method."""
        self._handle_train_new_scope_page()

    def _handle_train_new_select_page(self):
        """GET /train/new/select — Step 2 / 6: choose panel training data."""
        from urllib.parse import parse_qs, urlparse
        from capi_image_preprocess_lab import get_default_pipeline, get_method_specs
        qs = parse_qs(urlparse(self.path).query)
        mode = (qs.get("mode") or ["full"])[0]
        target_bundle_id = (qs.get("target_bundle_id") or [""])[0]
        units_raw = (qs.get("units") or [""])[0]
        scope_raw = {"mode": mode}
        if target_bundle_id:
            scope_raw["target_bundle_id"] = target_bundle_id
        if units_raw:
            scope_raw["selected_units"] = [u for u in units_raw.split(",") if u]

        scope, err = self._normalize_train_new_scope(scope_raw, self._capi_server_instance.database)
        if err:
            self._send_response(400, f"Invalid training scope: {err}")
            return

        target_bundle = None
        target_patchcore_params = {}
        target_image_preprocess_pipeline = []
        target_image_preprocess_pipelines = {}
        target_preprocess_after_tiling = False
        target_tile_stride = 256
        if scope["mode"] == "partial":
            target_bundle = self._capi_server_instance.database.get_model_bundle(scope["target_bundle_id"])
            try:
                from capi_model_registry import _read_manifest
                target_manifest = _read_manifest(Path(target_bundle["bundle_path"]))
                target_patchcore_params = target_manifest.get("patchcore_params") or {}
                target_image_preprocess_pipeline = target_manifest.get(
                    "image_preprocess_pipeline"
                ) or []
                target_image_preprocess_pipelines = target_manifest.get(
                    "image_preprocess_pipelines"
                ) or {}
                target_preprocess_after_tiling = bool(
                    target_manifest.get("preprocess_after_tiling", False)
                )
                target_tile_stride = int(target_manifest.get("tile_stride") or 512)
            except Exception:
                target_patchcore_params = {}

        template = self.jinja_env.get_template("train_new/step1_select.html")
        html = template.render(
            request_path="/train/new/select",
            active_jobs=self._list_open_train_new_jobs(),
            training_scope=scope,
            target_bundle=target_bundle,
            target_patchcore_params=target_patchcore_params,
            target_image_preprocess_pipeline=target_image_preprocess_pipeline,
            target_image_preprocess_pipelines=target_image_preprocess_pipelines,
            target_preprocess_after_tiling=target_preprocess_after_tiling,
            target_tile_stride=target_tile_stride,
            selected_lightings=self._scope_selected_lightings(scope),
            machine_prefix_len=self.TRAIN_NEW_MACHINE_PREFIX_LEN,
            preprocess_methods=get_method_specs(),
            default_preprocess_pipeline=get_default_pipeline(),
        )
        self._send_response(200, html)

    def _handle_train_new_progress_page(self):
        """GET /train/new/progress?job_id=X"""
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(self.path).query)
        job_id = (qs.get("job_id") or [""])[0]
        template_name = "train_new/step2_progress.html"
        job = None
        if job_id:
            job = self._capi_server_instance.database.get_training_job(job_id)
            job = self._mark_train_new_stale_if_needed(
                self._capi_server_instance.database,
                job,
            )
            if job and job.get("state") == "train":
                template_name = "train_new/step4_progress.html"
        template = self.jinja_env.get_template(template_name)
        scope = (job or {}).get("training_scope") if job else None
        selected_units = self._scope_selected_units(scope)
        unit_labels = [f"{lighting}-{zone}" for lighting, zone in selected_units]
        selected_lightings = self._scope_selected_lightings(scope)
        lighting_labels = self._train_new_lighting_labels(
            selected_lightings,
            (job or {}).get("panel_paths") or [],
        )
        html = template.render(
            request_path="/train/new/progress",
            job_id=job_id,
            training_scope=scope,
            unit_labels=unit_labels,
            display_unit_labels=[
                f"{lighting_labels.get(lighting, lighting)}-{zone}"
                for lighting, zone in selected_units
            ],
            selected_lightings=selected_lightings,
        )
        self._send_response(200, html)

    def _handle_train_new_review_page(self):
        """GET /train/new/review/<job_id>"""
        job_id = self.path.split("/")[-1]
        db = self._capi_server_instance.database
        job = db.get_training_job(job_id)
        if not job:
            self._send_response(404, "Job not found")
            return
        template = self.jinja_env.get_template("train_new/step3_review.html")
        scope = job.get("training_scope")
        selected_lightings = self._scope_selected_lightings(scope)
        html = template.render(
            request_path="/train/new/review",
            job_id=job_id,
            machine_id=job.get("machine_id", ""),
            training_scope=scope,
            selected_lightings=selected_lightings,
            lighting_labels=self._train_new_lighting_labels(
                selected_lightings,
                job.get("panel_paths") or [],
            ),
        )
        self._send_response(200, html)

    def _handle_train_new_done_page(self):
        """GET /train/new/done/<job_id>"""
        job_id = self.path.split("/")[-1]
        db = self._capi_server_instance.database
        job = db.get_training_job(job_id)
        if not job or job["state"] != "completed":
            self._send_response(404, "Job not done")
            return

        bundle_path = Path(job["output_bundle"])
        try:
            manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
            thresholds = json.loads((bundle_path / "thresholds.json").read_text(encoding="utf-8"))
        except Exception as e:
            self._send_response(500, f"Failed to read manifest/thresholds: {str(e)}")
            return

        # threshold 顯示以 machine_config.yaml 為準（inference 引擎讀的 source of truth）；
        # thresholds.json 退化成 fallback。手動改 yaml 沒同步 json 時也能正確顯示。
        yaml_p = bundle_path / "machine_config.yaml"
        if yaml_p.exists():
            try:
                import yaml as _yaml
                _cfg = _yaml.safe_load(yaml_p.read_text(encoding="utf-8")) or {}
                _yaml_thr = _cfg.get("threshold_mapping") or {}
                if isinstance(_yaml_thr, dict) and _yaml_thr:
                    _norm = {}
                    for _lt, _v in _yaml_thr.items():
                        if isinstance(_v, dict):
                            _norm[_lt] = {_z: float(_zv) for _z, _zv in _v.items()}
                        else:
                            _norm[_lt] = {"inner": float(_v), "edge": float(_v)}
                    thresholds = _norm
            except Exception as _e:
                logger.warning("[train/done] 讀 machine_config.yaml 失敗，fallback thresholds.json: %s", _e)

        # AUROC grade 可能是 "n/a"，斜線不能當 CSS class；slugify 給模板用
        def _grade_slug(g):
            return (g or "n/a").replace("/", "")

        unit_metrics = manifest.get("unit_metrics", {}) or {}
        manifest_lightings = []
        for unit_label in (manifest.get("tiles_per_unit") or {}):
            lighting, _zone = unit_label.rsplit("-", 1)
            if lighting not in manifest_lightings:
                manifest_lightings.append(lighting)
        lighting_labels = self._train_new_lighting_labels(
            manifest_lightings,
            job.get("panel_paths") or [],
        )
        units = []
        total_size_bytes = 0
        total_elapsed = 0
        for unit_label, tile_info in manifest["tiles_per_unit"].items():
            lighting, zone = unit_label.rsplit("-", 1)
            size_bytes = manifest["model_files"][unit_label]["size_bytes"]
            total_size_bytes += size_bytes
            m = unit_metrics.get(unit_label, {}) or {}
            elapsed = int(m.get("elapsed_seconds") or 0)
            total_elapsed += elapsed
            grade = m.get("auroc_grade") or "n/a"
            if "train_zero_score_warning" in m:
                train_zero_score_warning = bool(m.get("train_zero_score_warning"))
            else:
                # Legacy manifests only persisted the rounded train_max.  Use
                # it as a compatibility hint; new manifests use exact scores.
                train_max_value = m.get("train_max")
                train_zero_score_warning = bool(
                    int(m.get("train_count_eval") or 0) > 0
                    and train_max_value is not None
                    and float(train_max_value) == 0.0
                )
            feature_cleaning = dict(m.get("feature_cleaning") or {})
            report_rel = str(feature_cleaning.get("report_path") or "")
            feature_cleaning_visuals = []
            if report_rel:
                try:
                    report_root = (bundle_path / "feature_cleaning_reports").resolve()
                    report_path = (bundle_path / report_rel).resolve()
                    report_path.relative_to(report_root)
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    report_tiles = sorted(
                        report.get("tiles") or [],
                        key=lambda item: int(item.get("removed_count") or 0),
                        reverse=True,
                    )
                    for item in report_tiles[:12]:
                        asset_path = str(item.get("asset_path") or "")
                        if asset_path:
                            source_url = (
                                "/api/train/new/bundle-asset/"
                                + urllib.parse.quote(job_id, safe="")
                                + "/"
                                + urllib.parse.quote(asset_path, safe="/")
                            )
                        else:
                            source_url = self._train_new_thumb_url(
                                item.get("source_path")
                            )
                        if not source_url:
                            continue
                        feature_cleaning_visuals.append({
                            "source_url": source_url,
                            "source_name": (
                                item.get("source_name")
                                or Path(item.get("source_path") or "").name
                            ),
                            "grid_size": item.get("grid_size") or [],
                            "removed_indices": item.get("removed_indices") or [],
                            "removed_count": int(item.get("removed_count") or 0),
                            "distance_removed_count": int(
                                item.get("distance_removed_count") or 0
                            ),
                            "rejected_overlap_count": int(
                                item.get("rejected_overlap_count") or 0
                            ),
                            "protected_count": int(item.get("protected_count") or 0),
                            "distances": item.get("distances") or [],
                            "reason_codes": item.get("reason_codes") or [],
                            "overlap_view_counts": item.get("overlap_view_counts") or [],
                            "outlier_vote_counts": item.get("outlier_vote_counts") or [],
                            "outlier_vote_required": item.get("outlier_vote_required") or [],
                            "rejected_overlap_counts": item.get(
                                "rejected_overlap_counts"
                            ) or [],
                            "coreset_indices": item.get("coreset_indices") or [],
                            "coreset_count": int(item.get("coreset_count") or 0),
                            "rejected_neighbor_tile_ids": item.get(
                                "rejected_neighbor_tile_ids"
                            ) or [],
                            "tile_x": item.get("tile_x"),
                            "tile_y": item.get("tile_y"),
                            "tile_width": item.get("tile_width"),
                            "tile_height": item.get("tile_height"),
                            "threshold": report.get("threshold"),
                        })
                except (OSError, ValueError, json.JSONDecodeError):
                    logger.warning(
                        "[train/done] feature cleaning report unavailable: %s",
                        report_rel,
                        exc_info=True,
                    )
            feature_cleaning["visualizations"] = feature_cleaning_visuals
            units.append((unit_label, {
                "lighting": lighting,
                "lighting_label": lighting_labels.get(lighting, lighting),
                "zone": zone,
                "train": tile_info["train"], "ng": tile_info["ng"],
                "threshold": thresholds.get(lighting, {}).get(zone, 0.0),
                "size_mb": size_bytes / 1e6,
                "auroc": m.get("auroc"),
                "auroc_grade": grade,
                "auroc_grade_slug": _grade_slug(grade),
                "ng_caught_count": int(m.get("ng_caught_count") or 0),
                "ng_caught_rate": m.get("ng_caught_rate"),
                "ng_fallback": m.get("ng_used") == "fallback",
                "separation": m.get("separation"),
                "train_max": m.get("train_max"),
                "train_count_eval": int(m.get("train_count_eval") or 0),
                "train_zero_score_count": int(
                    m.get("train_zero_score_count") or 0
                ),
                "train_zero_score_rate": m.get("train_zero_score_rate"),
                "train_zero_score_warning": train_zero_score_warning,
                "ng_median": m.get("ng_median"),
                "ng_max": m.get("ng_max"),
                "elapsed_seconds": elapsed,
                "feature_pool_kernel_size": m.get(
                    "feature_pool_kernel_size",
                    (manifest.get("patchcore_params") or {}).get("feature_pool_kernel_size", 3),
                ),
                "feature_cleaning": feature_cleaning,
            }))

        overall_grade = manifest.get("overall_auroc_grade") or "n/a"
        zero_score_units = [
            unit_label
            for unit_label, info in units
            if info.get("train_zero_score_warning")
        ]
        template = self.jinja_env.get_template("train_new/step5_done.html")
        html = template.render(
            request_path="/train/new/done",
            machine_id=job["machine_id"],
            bundle_path=str(bundle_path),
            job_id=job_id, units=units,
            overall_auroc=manifest.get("overall_auroc"),
            overall_auroc_grade=overall_grade,
            overall_auroc_grade_slug=_grade_slug(overall_grade),
            trained_at=manifest.get("trained_at") or "",
            panel_count=manifest.get("panel_count") or 0,
            panel_glass_ids=manifest.get("panel_glass_ids") or [],
            patchcore_params=manifest.get("patchcore_params") or {},
            experimental_training=bool(manifest.get("experimental_training", False)),
            total_size_mb=total_size_bytes / 1e6,
            total_elapsed_seconds=total_elapsed,
            success_units=manifest.get("success_units") or len(units),
            zero_score_units=zero_score_units,
        )
        self._send_response(200, html)

    @staticmethod
    def _normalize_project_path(path_value) -> str:
        raw = str(path_value or "").strip()
        if not raw:
            return ""
        path = Path(raw)
        if path.is_absolute():
            try:
                path = path.resolve().relative_to(Path.cwd().resolve())
            except Exception:
                pass
        norm = path.as_posix()
        while norm.startswith("./"):
            norm = norm[2:]
        return norm.rstrip("/")

    @staticmethod
    def _same_project_path(path_a, path_b) -> bool:
        return CAPIWebHandler._normalize_project_path(path_a) == \
               CAPIWebHandler._normalize_project_path(path_b)

    @staticmethod
    def _next_scratch_bundle_path(current_bundle="", deployment_dir="deployment") -> str:
        deployment_dir = Path(deployment_dir)
        version_re = re.compile(r"^scratch_classifier_v(\d+)\.pkl$", re.IGNORECASE)
        versions = []
        if deployment_dir.exists():
            for path in deployment_dir.glob("scratch_classifier_v*.pkl"):
                m = version_re.match(path.name)
                if m:
                    versions.append(int(m.group(1)))

        current_norm = CAPIWebHandler._normalize_project_path(current_bundle)
        current_name = current_norm.replace("\\", "/").rsplit("/", 1)[-1]
        m = version_re.match(current_name)
        if m:
            versions.append(int(m.group(1)))

        next_version = max(versions, default=0) + 1
        return (deployment_dir / f"scratch_classifier_v{next_version}.pkl").as_posix()
    def _handle_retrain_page(self):
        """GET /retrain"""
        current_bundle = ""
        trained_at = "未知"
        server_inst = self._capi_server_instance
        if server_inst:
            cfg = getattr(server_inst, "config", None)
            if cfg:
                current_bundle = getattr(cfg, "scratch_bundle_path", "")
        if current_bundle:
            try:
                import pickle
                with open(current_bundle, "rb") as f:
                    bundle = pickle.load(f)
                trained_at = bundle.get("metadata", {}).get("trained_at", "未知")
            except Exception:
                pass

        template = self.jinja_env.get_template("retrain.html")
        html = template.render(
            request_path="/retrain",
            current_bundle=current_bundle or "(未設定)",
            trained_at=trained_at,
            default_manifest_base="/aidata/capi_ai/datasets/over_review/",
            default_output_path=self._next_scratch_bundle_path(current_bundle),
            default_epochs=15,
            default_rank=16,
            default_calib_frac=0.2,
            default_dinov2_repo="deployment/dinov2_repo",
            default_dinov2_weights="deployment/dinov2_vitb14.pth",
        )
        self._send_response(200, html)

    def _handle_retrain_start(self):
        """POST /api/retrain/start"""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            params = json.loads(body) if body else {}
        except Exception:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return

        if not params.get("manifest_base") or not params.get("output_path"):
            self._send_json({"error": "manifest_base and output_path are required"}, status=400)
            return

        # Reject absolute paths and path traversal to keep writes within project dir
        output_path_str = str(params["output_path"])
        if Path(output_path_str).is_absolute() or ".." in output_path_str:
            self._send_json({"error": "output_path must be a relative path within the project"}, status=400)
            return

        current_bundle = ""
        server_inst = self._capi_server_instance
        if server_inst:
            cfg = getattr(server_inst, "config", None)
            if cfg:
                current_bundle = getattr(cfg, "scratch_bundle_path", "")
        suggested_output = self._next_scratch_bundle_path(current_bundle)
        if current_bundle and self._same_project_path(output_path_str, current_bundle):
            self._send_json({
                "error": (
                    "output_path points to the currently deployed scratch bundle; "
                    f"choose a new versioned path such as {suggested_output}"
                )
            }, status=400)
            return
        if Path(output_path_str).exists():
            self._send_json({
                "error": (
                    f"output_path already exists: {output_path_str}; "
                    f"choose a new versioned path such as {suggested_output}"
                )
            }, status=400)
            return
        state = self._retrain_state
        with state["lock"]:
            job = state["job"]
            if job and job.get("state") == "running":
                self._send_json({
                    "error": "job_already_running",
                    "job_id": job["job_id"],
                }, status=409)
                return

            job_id = datetime.now().strftime("retrain_%Y%m%d_%H%M%S")
            new_job = {
                "job_id": job_id,
                "state": "running",
                "step": "merge",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "output_path": params["output_path"],
                "log_lines": [],
                "_log_lock": threading.Lock(),
                "summary": None,
                "error": None,
            }
            state["job"] = new_job

        thread = threading.Thread(
            target=CAPIWebHandler._retrain_worker,
            args=(new_job, params),
            daemon=True,
            name=f"retrain-{job_id}",
        )
        thread.start()
        self._send_json({"job_id": job_id, "started_at": new_job["started_at"]})

    def _handle_train_new_panels(self):
        """GET /api/train/new/panels?machine_id=X&days=3

        回傳近 3 天內 machine_judgment='OK'
        的 inference_records，供訓練 wizard 第一步選擇訓練樣本使用。
        machine_id 可省略，省略時回傳所有機種的最近紀錄。
        machine_id_prefix 可省略，局部重訓用來查同前綴料號。
        """
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(self.path).query)
        machine_id = (qs.get("machine_id") or [""])[0].strip()
        machine_id_prefix = (qs.get("machine_id_prefix") or [""])[0].strip()
        try:
            days = int((qs.get("days") or ["3"])[0])
        except (ValueError, TypeError):
            days = 3
        days = max(1, min(days, 3))

        if not self.db:
            self._send_json({"error": "database not available"}, status=503)
            return

        try:
            if machine_id_prefix:
                panels = self.db.list_ok_panels_for_machine(
                    machine_id,
                    days=days,
                    machine_id_prefix=machine_id_prefix,
                )
            else:
                panels = self.db.list_ok_panels_for_machine(machine_id, days=days)
            for panel in panels:
                image_dir = panel.get("image_dir", "")
                panel["image_path"] = image_dir
                panel["preview_image_path"] = self._resolve_train_new_preview_image_path(image_dir)
            self._send_json({"panels": panels, "days": days})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    @staticmethod
    def _resolve_train_new_input_path(path_value, server_inst) -> Path:
        """Resolve an operator-entered path to the path readable by this host."""
        from capi_server import resolve_unc_path

        raw = str(path_value or "").strip()
        path_mapping = getattr(server_inst, "path_mapping", {}) if server_inst else {}
        return Path(resolve_unc_path(raw, path_mapping))

    @staticmethod
    def _scan_train_new_manual_batch(batch_root: Path, machine_id: str) -> list:
        """List valid first-level panel folders from a manually prepared batch."""
        from capi_preprocess import LIGHTING_PREFIXES, filter_panel_lighting_files

        panels = []
        for panel_dir in sorted(batch_root.iterdir(), key=lambda p: p.name.lower()):
            if not panel_dir.is_dir():
                continue
            lighting_files = filter_panel_lighting_files(panel_dir)
            if not lighting_files:
                continue
            lightings = [lighting for lighting in LIGHTING_PREFIXES if lighting in lighting_files]
            preview_path = CAPIWebHandler._resolve_train_new_preview_image_path(panel_dir)
            panels.append({
                "glass_id": panel_dir.name,
                "model_id": machine_id,
                "machine_no": "手動資料夾",
                "machine_judgment": "OK",
                "ai_judgment": "人工確認 OK",
                "image_dir": str(panel_dir),
                "image_path": str(panel_dir),
                "preview_image_path": preview_path,
                "available_lightings": lightings,
                "lighting_count": len(lightings),
                "expected_lighting_count": len(LIGHTING_PREFIXES),
                "source_type": "manual_folder",
            })
        return panels

    def _handle_train_new_manual_panels(self):
        """POST /api/train/new/manual-panels — scan one batch directory level."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            data = json.loads(body) if body else {}
        except Exception:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return

        machine_id = str(data.get("machine_id") or "").strip()
        batch_root_raw = str(data.get("batch_root") or "").strip()
        if not machine_id:
            self._send_json({"error": "請提供機種 ID"}, status=400)
            return
        if not batch_root_raw:
            self._send_json({"error": "請提供 batch 根目錄"}, status=400)
            return

        batch_root = self._resolve_train_new_input_path(
            batch_root_raw, self._capi_server_instance
        )
        if not batch_root.is_dir():
            self._send_json({"error": f"batch 根目錄不存在或無法讀取: {batch_root}"}, status=400)
            return

        try:
            panels = self._scan_train_new_manual_batch(batch_root, machine_id)
        except OSError as exc:
            self._send_json({"error": f"無法讀取 batch 根目錄: {exc}"}, status=400)
            return
        if not panels:
            self._send_json({
                "error": "batch 根目錄第一層找不到含有效 lighting 圖的 panel 資料夾"
            }, status=400)
            return

        self._send_json({
            "panels": panels,
            "machine_id": machine_id,
            "batch_root": str(batch_root),
            "source_type": "manual_folder",
        })

    @staticmethod
    def _resolve_train_new_preview_image_path(path_value) -> str:
        """Resolve a panel folder to the W0F00000_* image used by Step 2 preview."""
        raw = str(path_value or "").strip()
        if not raw:
            return ""
        path = Path(raw)
        if path.is_file():
            return str(path)
        if not path.is_dir():
            return ""

        for candidate in sorted(path.iterdir(), key=lambda p: p.name):
            if candidate.is_file() and candidate.name.upper().startswith("W0F00000_"):
                return str(candidate)

        try:
            from capi_preprocess import filter_panel_lighting_files
            files = filter_panel_lighting_files(path)
            preferred = ("W0F00000", "STANDARD", "G0F00000", "R0F00000", "WGF50500")
            for lighting in preferred:
                if lighting in files:
                    return str(files[lighting])
        except Exception:
            return ""
        return ""

    @staticmethod
    def _validate_training_params(raw):
        """驗證並 normalize 使用者送上來的 training_params。

        回傳 (params_dict_or_None, error_msg_or_None)。
        - raw 為 None / 空 dict → params_dict_or_None=None（之後吃 dataclass 預設）
        - 含未知 key → error
        - 數值越界 / 型別錯 → error
        """
        from capi_train_new import (
            USER_TRAINABLE_PARAM_SPECS,
            normalize_feature_cleaning_by_zone,
        )
        if raw is None:
            return None, None
        if not isinstance(raw, dict):
            return None, "training_params must be an object"
        if not raw:
            return None, None
        unknown = set(raw.keys()) - set(USER_TRAINABLE_PARAM_SPECS.keys())
        if unknown:
            return None, f"unknown training_params keys: {sorted(unknown)}"
        cleaned = {}
        for key, spec in USER_TRAINABLE_PARAM_SPECS.items():
            if key not in raw:
                continue
            val = raw[key]
            if key == "feature_cleaning_by_zone":
                try:
                    cleaned[key] = normalize_feature_cleaning_by_zone(val)
                except ValueError as exc:
                    return None, f"training_params.{exc}"
                continue
            if "choices" in spec:
                if val not in spec["choices"]:
                    return None, (
                        f"training_params.{key} must be one of "
                        f"{spec['choices']}"
                    )
                cleaned[key] = val
                continue
            try:
                if isinstance(val, bool):
                    raise TypeError
                if spec["type"] is int:
                    val = int(val)
                else:
                    val = float(val)
            except (TypeError, ValueError):
                return None, f"training_params.{key} must be {spec['type'].__name__}"
            if val < spec["min"] or val > spec["max"]:
                return None, (
                    f"training_params.{key} out of range "
                    f"[{spec['min']}, {spec['max']}]"
                )
            cleaned[key] = val
        return (cleaned or None), None

    @staticmethod
    def _validate_image_preprocess_pipeline(raw):
        """Normalize image preprocessing pipeline from Step 2.

        Missing value means the current recommended default pipeline. An empty
        list is valid and explicitly disables image preprocessing.
        """
        from capi_image_preprocess_lab import get_default_pipeline, normalize_preprocess_pipeline
        try:
            if raw is None:
                return normalize_preprocess_pipeline(get_default_pipeline()), None
            return normalize_preprocess_pipeline(raw), None
        except Exception as exc:
            return None, f"image_preprocess_pipeline invalid: {exc}"

    @staticmethod
    def _validate_image_preprocess_pipelines(raw):
        """Normalize optional INNER/EDGE tile preprocessing pipelines."""
        from capi_train_new import normalize_image_preprocess_pipelines
        try:
            return normalize_image_preprocess_pipelines(raw), None
        except Exception as exc:
            return None, f"image_preprocess_pipelines invalid: {exc}"

    @staticmethod
    def _validate_train_tile_stride(raw):
        """Validate training tile step/stride. 512 keeps legacy non-overlap tiling."""
        if raw is None or raw == "":
            return CAPIWebHandler.TRAIN_NEW_DEFAULT_TILE_STRIDE, None
        if isinstance(raw, bool):
            return None, "tile_stride must be an integer"
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            return None, "tile_stride must be an integer"
        if not parsed.is_integer():
            return None, "tile_stride must be an integer"
        value = int(parsed)
        if not (
            CAPIWebHandler.TRAIN_NEW_MIN_TILE_STRIDE
            <= value
            <= CAPIWebHandler.TRAIN_NEW_MAX_TILE_STRIDE
        ):
            return None, (
                "tile_stride out of range "
                f"({CAPIWebHandler.TRAIN_NEW_MIN_TILE_STRIDE}.."
                f"{CAPIWebHandler.TRAIN_NEW_MAX_TILE_STRIDE})"
            )
        return value, None

    def _handle_train_new_start(self):
        """POST /api/train/new/start

        body: {
            "machine_id": "...",
            "panel_paths": [...],   # 至少 1 片
            "panel_modes": [...],   # optional; full / inner_only / edge_only
            "tile_stride": 256,     # optional; 512 means legacy non-overlap
            "training_params": {  # optional
                "batch_size": 8,
                "coreset_ratio": 0.1,
                "max_epochs": 1,
                "feature_pool_kernel_size": 3,
                "feature_cleaning_mode": "off",
                "feature_cleaning_scope": "inner_only",
                "feature_cleaning_keep_ratio": 0.99,
                "feature_cleaning_center_size": 384
            },
            "image_preprocess_pipeline": [  # optional; omitted means recommended default
                {"method": "bilateral", "params": {"diameter": 9}}
            ],
            "training_scope": {   # optional; omitted means full 10-unit training
                "mode": "full" | "partial",
                "target_bundle_id": 123,            # partial only
                "selected_units": ["G0F00000-inner"] # partial only
            }
        }

        panel_modes 未提供時由後端補成全 full，維持舊版呼叫相容。
        """
        from capi_train_new import (
            generate_job_id, normalize_panel_modes, panel_mode_zones,
        )

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            params = json.loads(body) if body else {}
        except Exception:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return

        machine_id = params.get("machine_id", "").strip()
        panel_paths = params.get("panel_paths", [])
        if not machine_id or not panel_paths:
            self._send_json({"error": "machine_id and panel_paths required"}, status=400)
            return
        if not isinstance(panel_paths, list):
            self._send_json({"error": "panel_paths must be a list"}, status=400)
            return
        clean_panel_paths = []
        for p in panel_paths:
            if not isinstance(p, str) or not p.strip() or p.strip() in ("undefined", "null"):
                self._send_json({"error": "panel_paths contains invalid path"}, status=400)
                return
            clean_panel_paths.append(p.strip())
        try:
            panel_modes = normalize_panel_modes(
                params.get("panel_modes"), len(clean_panel_paths)
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        training_data_source = params.get("training_data_source")
        if training_data_source is None:
            training_data_source = {"type": "inference_records"}
        if not isinstance(training_data_source, dict):
            self._send_json({"error": "training_data_source must be an object"}, status=400)
            return
        source_type = str(training_data_source.get("type") or "").strip()
        if source_type not in ("inference_records", "manual_folder"):
            self._send_json({"error": "unsupported training_data_source.type"}, status=400)
            return
        if source_type == "manual_folder":
            if training_data_source.get("confirmed_normal") is not True:
                self._send_json({"error": "手動資料必須確認為正常訓練樣本"}, status=400)
                return
            batch_root_raw = str(training_data_source.get("batch_root") or "").strip()
            if not batch_root_raw:
                self._send_json({"error": "請提供 batch 根目錄"}, status=400)
                return
            batch_root = self._resolve_train_new_input_path(
                batch_root_raw, self._capi_server_instance
            )
            if not batch_root.is_dir():
                self._send_json({"error": f"batch 根目錄不存在或無法讀取: {batch_root}"}, status=400)
                return
            try:
                valid_panels = {
                    panel["image_path"]
                    for panel in self._scan_train_new_manual_batch(batch_root, machine_id)
                }
            except OSError as exc:
                self._send_json({"error": f"無法讀取 batch 根目錄: {exc}"}, status=400)
                return
            invalid_paths = [p for p in clean_panel_paths if p not in valid_panels]
            if invalid_paths:
                self._send_json({
                    "error": f"panel_paths 含不屬於此 batch 或無有效 lighting 的路徑: {invalid_paths[0]}"
                }, status=400)
                return
            training_data_source = {
                "type": "manual_folder",
                "batch_root": str(batch_root),
                "confirmed_normal": True,
            }
        else:
            training_data_source = {"type": "inference_records"}

        training_params, err = self._validate_training_params(params.get("training_params"))
        if err:
            self._send_json({"error": err}, status=400)
            return
        image_preprocess_pipeline, err = self._validate_image_preprocess_pipeline(
            params.get("image_preprocess_pipeline")
        )
        if err:
            self._send_json({"error": err}, status=400)
            return
        image_preprocess_pipelines, err = self._validate_image_preprocess_pipelines(
            params.get("image_preprocess_pipelines")
        )
        if err:
            self._send_json({"error": err}, status=400)
            return

        preprocess_after_tiling = bool(params.get("preprocess_after_tiling", False))
        if (
            image_preprocess_pipelines
            and params.get("preprocess_after_tiling") is not True
        ):
            self._send_json({
                "error": "INNER/EDGE 分區前處理只支援先切分後處理"
            }, status=400)
            return
        tile_stride, err = self._validate_train_tile_stride(params.get("tile_stride"))
        if err:
            self._send_json({"error": err}, status=400)
            return

        db = self._capi_server_instance.database
        training_scope, err = self._normalize_train_new_scope(params.get("training_scope"), db)
        if err:
            self._send_json({"error": err}, status=400)
            return
        required_zones = {
            zone for _lighting, zone in self._scope_selected_units(training_scope)
        }
        provided_zones = set()
        for panel_mode in panel_modes:
            provided_zones.update(panel_mode_zones(panel_mode))
        missing_zones = required_zones - provided_zones
        if missing_zones:
            missing_label = "、".join(zone.upper() for zone in sorted(missing_zones))
            self._send_json({
                "error": (
                    f"已選 PANEL 沒有提供 {missing_label} 訓練切片；"
                    "請至少在一片 PANEL 勾選該區域"
                )
            }, status=400)
            return
        if training_scope["mode"] == "partial":
            locked_params = sorted(
                set(training_params or {})
                & self.PATCHCORE_BUNDLE_LOCKED_TRAINING_PARAMS
            )
            if locked_params:
                self._send_json({
                    "error": (
                        "partial training must inherit bundle-level PatchCore params: "
                        + ", ".join(locked_params)
                    )
                }, status=400)
                return
            target_bundle = db.get_model_bundle(training_scope["target_bundle_id"])
            if not target_bundle:
                self._send_json({"error": "target bundle not found"}, status=404)
                return
            target_machine_id = str(target_bundle.get("machine_id") or "")
            if not target_machine_id:
                self._send_json({
                    "error": "partial training target bundle has empty machine_id"
                }, status=400)
                return
            if target_machine_id != machine_id:
                if self._same_machine_family(target_machine_id, machine_id):
                    machine_id = target_machine_id
                else:
                    target_prefix = self._machine_id_prefix(target_machine_id)
                    self._send_json({
                        "error": (
                            "partial training machine_id must match target bundle "
                            f"prefix ({target_prefix})"
                        )
                    }, status=400)
                    return
            try:
                from capi_model_registry import _read_manifest
                target_manifest = _read_manifest(Path(target_bundle["bundle_path"]))
                image_preprocess_pipeline = target_manifest.get(
                    "image_preprocess_pipeline"
                ) or []
                image_preprocess_pipelines = target_manifest.get(
                    "image_preprocess_pipelines"
                ) or {}
                preprocess_after_tiling = bool(
                    target_manifest.get("preprocess_after_tiling", False)
                )
            except Exception as exc:
                self._send_json({
                    "error": f"cannot inherit target bundle preprocessing: {exc}"
                }, status=400)
                return

        job_id = generate_job_id(machine_id)

        # 註冊 runtime + 寫 DB（沒有 wizard singleton 檢查；多 job 可共存）。
        # GPU singleton 留給 _handle_train_new_start_training 用 _train_slot 把關。
        runtime = CAPIWebHandler._make_job_runtime(job_id, "preprocess")
        try:
            db.create_training_job(
                job_id=job_id, machine_id=machine_id,
                panel_paths=clean_panel_paths,
                training_params=training_params,
                panel_modes=panel_modes,
                training_scope=training_scope,
                training_data_source=training_data_source,
                image_preprocess_pipeline=image_preprocess_pipeline,
                image_preprocess_pipelines=image_preprocess_pipelines,
                preprocess_after_tiling=preprocess_after_tiling,
                tile_stride=tile_stride,
            )
        except Exception:
            CAPIWebHandler._drop_job_runtime(job_id)
            raise

        thread = threading.Thread(
            target=CAPIWebHandler._train_new_preprocess_worker,
            args=(job_id, machine_id, clean_panel_paths,
                  self._capi_server_instance, training_params, panel_modes,
                  training_scope, image_preprocess_pipeline, preprocess_after_tiling,
                  tile_stride, training_data_source, image_preprocess_pipelines),
            daemon=True, name=f"train_new_pre-{job_id}",
        )
        runtime["thread"] = thread
        thread.start()
        self._send_json({"job_id": job_id, "state": "preprocess"})

    @staticmethod
    def _load_train_new_config(server_inst) -> dict:
        """Load training wizard config and resolve relative paths from server_config.yaml."""
        import yaml as _yaml
        from pathlib import Path as _Path

        server_config_path = _Path(server_inst.server_config_path).resolve()
        base_dir = server_config_path.parent
        try:
            raw = _yaml.safe_load(server_config_path.read_text(encoding="utf-8")) or {}
            training_cfg = raw.get("training", {}) or {}
        except Exception:
            training_cfg = {}

        def resolve_path(key: str, default: str) -> _Path:
            value = training_cfg.get(key, default)
            path = _Path(value)
            return path if path.is_absolute() else base_dir / path

        required_backbones = training_cfg.get("required_backbones") or ["wide_resnet50_2-32ee1156.pth"]
        return {
            "over_review_root": resolve_path("over_review_root", "/aidata/capi_ai/datasets/over_review"),
            "backbone_cache_dir": resolve_path("backbone_cache_dir", "deployment/torch_hub_cache"),
            "output_root": resolve_path("output_root", "model"),
            "required_backbones": list(required_backbones),
        }

    @staticmethod
    def _train_new_preprocess_worker(
        job_id, machine_id, panel_paths, server_inst, training_params=None,
        panel_modes=None, training_scope=None, image_preprocess_pipeline=None,
        preprocess_after_tiling=False, tile_stride=None, training_data_source=None,
        image_preprocess_pipelines=None,
    ):
        """背景 thread：preprocess + 從推論紀錄抽 AOI 炸彈 NG → state=review。

        panel_modes 與 panel_paths 同長度；None 視同全 full（向下相容舊呼叫者）。
        失敗條件：至少要有 1 片 panel 成功寫入 tile。
        """
        import traceback
        from pathlib import Path as _Path
        from capi_train_new import (
            TrainingConfig, apply_user_training_params,
            preprocess_panels_to_pool, sample_ng_tiles, NG_TILES_PER_LIGHTING,
        )
        from capi_preprocess import PreprocessConfig

        db = server_inst.database
        runtime = CAPIWebHandler._get_job_runtime(job_id)
        if runtime is None:
            # 防呆：理論上 _handle_train_new_start 已建好
            runtime = CAPIWebHandler._make_job_runtime(job_id, "preprocess")

        def log(msg):
            CAPIWebHandler._append_train_new_log(job_id, msg)

        try:
            train_cfg = CAPIWebHandler._load_train_new_config(server_inst)

            thumb_root = _Path(".tmp/train_new_thumbs") / job_id
            cfg = TrainingConfig(
                machine_id=machine_id,
                panel_paths=[_Path(p) for p in panel_paths],
                over_review_root=train_cfg["over_review_root"],
                backbone_cache_dir=train_cfg["backbone_cache_dir"],
                output_root=train_cfg["output_root"],
                required_backbones=train_cfg["required_backbones"],
                image_preprocess_pipeline=image_preprocess_pipeline or [],
                image_preprocess_pipelines=image_preprocess_pipelines or {},
                preprocess_after_tiling=preprocess_after_tiling,
                tile_stride=int(tile_stride or CAPIWebHandler.TRAIN_NEW_DEFAULT_TILE_STRIDE),
                training_data_source=training_data_source or {"type": "inference_records"},
            )
            apply_user_training_params(cfg, training_params, log_fn=log)
            pre_cfg = PreprocessConfig(
                tile_stride=cfg.tile_stride,
                image_preprocess_pipeline=cfg.image_preprocess_pipeline,
                image_preprocess_pipelines=cfg.image_preprocess_pipelines,
                preprocess_after_tiling=cfg.preprocess_after_tiling,
                product_resolution=CAPIWebHandler._product_resolution_for_machine(machine_id, server_inst),
            )
            target_lightings = None
            target_units = None
            if training_scope and training_scope.get("mode") == "partial":
                target_lightings = CAPIWebHandler._scope_selected_lightings(training_scope)
                target_units = training_scope.get("selected_units") or None
                log(
                    "局部重訓 scope: "
                    + ", ".join(training_scope.get("selected_units") or [])
                )

            log(f"開始前處理 {len(panel_paths)} panel（tile=512, stride={cfg.tile_stride}）")
            stats = preprocess_panels_to_pool(
                job_id=job_id, cfg=cfg, preprocess_cfg=pre_cfg,
                db=db, thumb_dir=thumb_root, log=log,
                panel_modes=panel_modes,
                target_lightings=target_lightings,
                target_units=target_units,
            )
            if stats["panel_success"] <= 0:
                raise RuntimeError("沒有任何 panel 前處理成功")

            log(
                f"準備 NG 驗證 crop（優先重用 NG 驗證庫；缺少才從推論紀錄裁切，"
                f"每 lighting 上限 {NG_TILES_PER_LIGHTING} 個，排除 B0F 黑畫面）"
            )
            ng_stats = CAPIWebHandler._sample_ng_tiles_compat(
                sample_ng_tiles,
                job_id=job_id, over_review_root=cfg.over_review_root,
                db=db, thumb_dir=thumb_root, log=log,
                lightings=target_lightings,
                preprocess_cfg=pre_cfg,
                machine_id=machine_id,
                rotate_180=CAPIWebHandler._training_bomb_rotate_180(machine_id, server_inst),
                ng_validation_base_dir=CAPIWebHandler._ng_validation_base_dir_for_server(
                    server_inst
                ),
            )

            db.update_training_job_state(job_id, "review")
            runtime["phase"] = "review"
            log("✓ 進入 review 階段")
        except Exception as e:
            traceback.print_exc()
            db.update_training_job_state(job_id, "failed", error_message=str(e))
            CAPIWebHandler._cleanup_train_new_job_artifacts(
                db, job_id, reason="preprocess failed"
            )
            log(f"✗ 失敗: {e}")
            CAPIWebHandler._drop_job_runtime(job_id)

    def _handle_train_new_status(self):
        """GET /api/train/new/status?job_id=X

        若無 job_id 則回傳最近的 active job；若無 active job 則回傳 idle。
        若指定 job_id 則查詢該 job；若不存在則回傳 404。
        """
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(self.path).query)
        job_id = (qs.get("job_id") or [""])[0]
        no_cache_headers = {
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        }

        db = self._capi_server_instance.database

        if not job_id:
            # 無 job_id 回最近 active job 狀態
            job = db.get_active_training_job()
            if not job:
                self._send_json({"state": "idle"}, headers=no_cache_headers)
                return
            job = self._mark_train_new_stale_if_needed(db, job)
            job_id = job["job_id"]
        else:
            job = db.get_training_job(job_id)
            if not job:
                self._send_json(
                    {"error": "job not found"},
                    status=404,
                    headers=no_cache_headers,
                )
                return
            job = self._mark_train_new_stale_if_needed(db, job)

        runtime = CAPIWebHandler._get_job_runtime(job_id)
        if runtime is None:
            log_lines = []
            unit_status = {}
            completed_bundle = None
        else:
            with runtime["log_lock"]:
                log_lines = list(runtime["log_lines"][-100:])
                unit_status = dict(runtime.get("unit_status") or {})
                completed_bundle = runtime.get("completed_bundle")

        job = self._sync_train_new_completed_state(db, job, completed_bundle)

        resp = {
            "job_id": job["job_id"], "machine_id": job["machine_id"],
            "state": job["state"],
            "started_at": job["started_at"], "completed_at": job["completed_at"],
            "output_bundle": job["output_bundle"], "error_message": job["error_message"],
            "training_scope": job.get("training_scope"),
            "tile_stride": job.get("tile_stride"),
            "log_lines": log_lines,
            "unit_status": unit_status,
            "worker_alive": self._train_new_worker_alive(job_id),
        }
        self._send_json(resp, headers=no_cache_headers)

    def _handle_train_new_tiles(self):
        """GET /api/train/new/tiles?job_id=X&lighting=Y[&score_from_bundle=N&sort_by=score_desc]"""
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(self.path).query)
        job_id = (qs.get("job_id") or [""])[0]
        lighting = (qs.get("lighting") or [""])[0]
        score_from = qs.get("score_from_bundle", [None])[0]
        sort_by = qs.get("sort_by", ["default"])[0]
        if not job_id or not lighting:
            self._send_json({"error": "job_id and lighting required"}, status=400)
            return
        db = self._capi_server_instance.database
        tiles = db.list_tile_pool(job_id, lighting=lighting)
        for tile in tiles:
            tile["thumb_url"] = self._train_new_thumb_url(tile.get("thumb_path"))
            tile["image_url"] = self._train_new_thumb_url(tile.get("source_path"))
        if score_from:
            CAPIWebHandler._decorate_tiles_with_scores(tiles, db, score_from, sort_by)
        self._send_json({"tiles": tiles})

    def _handle_train_new_preprocess_pipeline_preview(self):
        """POST /api/train/new/preprocess_pipeline_preview

        Run the Step 2 image preprocessing pipeline once against either an image
        path or the first training lighting image in a panel folder.
        """
        import time as _time
        import uuid

        import cv2

        from capi_image_preprocess_lab import (
            apply_preprocess_pipeline,
            make_diff_image,
            normalize_preprocess_pipeline,
        )

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            data = json.loads(body) if body else {}
        except Exception:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return

        raw_path = str(data.get("image_path") or data.get("panel_path") or "").strip()
        if not raw_path:
            self._send_json({"error": "請提供圖片路徑或 panel 資料夾路徑"}, status=400)
            return

        try:
            pipeline = normalize_preprocess_pipeline(data.get("image_preprocess_pipeline", []))
        except Exception as exc:
            self._send_json({"error": f"前處理流程無效: {exc}"}, status=400)
            return
        preprocess_after_tiling = bool(data.get("preprocess_after_tiling", False))
        preview_zone = str(data.get("zone") or "inner").strip().lower()
        if preview_zone not in ("inner", "edge"):
            self._send_json({"error": "zone must be inner or edge"}, status=400)
            return
        try:
            from capi_train_new import normalize_image_preprocess_pipelines
            zone_pipelines = normalize_image_preprocess_pipelines(
                data.get("image_preprocess_pipelines")
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        if zone_pipelines and data.get("preprocess_after_tiling") is not True:
            self._send_json({
                "error": "INNER/EDGE 分區前處理只支援先切分後處理"
            }, status=400)
            return
        preview_pipeline = zone_pipelines.get(preview_zone, pipeline)
        tile_stride, err = self._validate_train_tile_stride(data.get("tile_stride"))
        if err:
            self._send_json({"error": err}, status=400)
            return
        machine_id = str(data.get("machine_id") or "").strip()

        source_path = Path(raw_path)
        if not source_path.exists():
            self._send_json({"error": f"路徑不存在: {source_path}"}, status=400)
            return

        if source_path.is_dir():
            resolved_preview = self._resolve_train_new_preview_image_path(source_path)
            if not resolved_preview:
                self._send_json({"error": f"資料夾內找不到可訓練 lighting 圖: {source_path}"}, status=400)
                return
            image_path = Path(resolved_preview)
        elif source_path.is_file():
            image_path = source_path
        else:
            self._send_json({"error": f"不是圖片檔或資料夾: {source_path}"}, status=400)
            return

        try:
            start = _time.time()
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                self._send_json({"error": f"無法讀取圖片: {image_path}"}, status=400)
                return

            if CAPIWebHandler._debug_heatmap_dir is None:
                CAPIWebHandler._debug_heatmap_dir = Path(tempfile.mkdtemp(prefix="capi_debug_hm_"))
            debug_dir = CAPIWebHandler._debug_heatmap_dir
            debug_dir.mkdir(parents=True, exist_ok=True)

            safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem)[:80] or "image"
            token = uuid.uuid4().hex[:8]
            original_filename = None
            original_path = None
            original_url = "/api/debug/serve-image?raw=1&path=" + urllib.parse.quote(str(image_path))
            preview_mode = "image"
            extra_payload = {
                "preprocess_after_tiling": False,
                "preview_mode": preview_mode,
                "preview_size": None,
            }

            if preprocess_after_tiling:
                from capi_preprocess import LIGHTING_PREFIXES, PreprocessConfig, preprocess_panel_image

                lighting = canonical_image_prefix(image_path.name)
                if lighting not in LIGHTING_PREFIXES:
                    lighting = "STANDARD"
                pre_cfg = PreprocessConfig(
                    tile_stride=tile_stride,
                    image_preprocess_pipeline=preview_pipeline,
                    preprocess_after_tiling=True,
                    product_resolution=self._product_resolution_for_machine(
                        machine_id,
                        self._capi_server_instance,
                    ),
                )
                panel_result = preprocess_panel_image(image_path, lighting, pre_cfg)
                if not panel_result.tiles:
                    self._send_json({"error": "先切分後處理預覽無法產生 tile"}, status=400)
                    return

                tile = next(
                    (t for t in panel_result.tiles if t.zone == preview_zone),
                    None,
                )
                if tile is None:
                    self._send_json({
                        "error": f"這張圖片沒有可預覽的 {preview_zone.upper()} tile"
                    }, status=400)
                    return
                original = tile.original_image
                if original is None:
                    original = image[tile.y1:tile.y2, tile.x1:tile.x2].copy()
                result = apply_preprocess_pipeline(original, preview_pipeline)
                processed = result["image"]
                diff = make_diff_image(original, processed)

                preview_mode = "tile"
                original_filename = f"train_preprocess_preview_{safe_stem}_{token}_tile_orig.png"
                processed_filename = f"train_preprocess_preview_{safe_stem}_{token}_tile.png"
                diff_filename = f"train_preprocess_preview_{safe_stem}_{token}_tile_diff.png"
                original_path = debug_dir / original_filename
                original_url = f"/debug/heatmaps/{original_filename}"
                extra_payload = {
                    "preprocess_after_tiling": True,
                    "preview_mode": preview_mode,
                    "preview_size": [int(original.shape[1]), int(original.shape[0])],
                    "tile_id": int(tile.tile_id),
                    "tile_rect": [int(tile.x1), int(tile.y1), int(tile.x2), int(tile.y2)],
                    "tile_zone": tile.zone,
                    "requested_zone": preview_zone,
                }
            else:
                result = apply_preprocess_pipeline(image, pipeline)
                processed = result["image"]
                diff = make_diff_image(image, processed)
                processed_filename = f"train_preprocess_preview_{safe_stem}_{token}.png"
                diff_filename = f"train_preprocess_preview_{safe_stem}_{token}_diff.png"

            processed_path = debug_dir / processed_filename
            diff_path = debug_dir / diff_filename

            if original_path is not None and not cv2.imwrite(str(original_path), original):
                self._send_json({"error": "原始 tile 圖片寫入失敗"}, status=500)
                return
            if not cv2.imwrite(str(processed_path), processed):
                self._send_json({"error": "處理後圖片寫入失敗"}, status=500)
                return
            if not cv2.imwrite(str(diff_path), diff):
                self._send_json({"error": "差異圖寫入失敗"}, status=500)
                return

            h, w = image.shape[:2]
            channels = 1 if image.ndim == 2 else image.shape[2]
            payload = {
                "success": True,
                "input_path": str(source_path),
                "image_path": str(image_path),
                "image_name": image_path.name,
                "image_size": [w, h],
                "channels": channels,
                "input_dtype": str(image.dtype),
                "pipeline": result["pipeline"],
                "steps": result["steps"],
                "processing_time": round(_time.time() - start, 3),
                "original_url": original_url,
                "processed_url": f"/debug/heatmaps/{processed_filename}",
                "diff_url": f"/debug/heatmaps/{diff_filename}",
                "output_path": str(processed_path),
                "diff_path": str(diff_path),
            }
            if original_path is not None:
                payload["original_path"] = str(original_path)
            payload.update(extra_payload)
            self._send_json(payload)
        except Exception as exc:
            logger.error("[train/new] preprocessing pipeline preview failed: %s", exc, exc_info=True)
            self._send_json({"error": f"前處理預覽失敗: {exc}"}, status=500)

    def _handle_train_new_preprocess_preview(self):
        """GET /api/train/new/preprocess_preview?job_id=X&lighting=Y

        Rebuilds a compact visual of the actual preprocessing geometry using the
        same preprocessing code path: foreground/panel boundary and generated
        inner/edge tile rectangles.
        """
        from urllib.parse import parse_qs, urlparse
        import cv2
        from capi_preprocess import PreprocessConfig, filter_panel_lighting_files, preprocess_panel_folder

        qs = parse_qs(urlparse(self.path).query)
        job_id = (qs.get("job_id") or [""])[0]
        lighting = (qs.get("lighting") or ["G0F00000"])[0]
        if not job_id:
            self._send_response(400, "")
            return

        db = self._capi_server_instance.database
        job = db.get_training_job(job_id)
        if not job:
            self._send_response(404, "")
            return

        preprocess_cfg = PreprocessConfig(
            tile_stride=int(job.get("tile_stride") or 512),
            image_preprocess_pipeline=job.get("image_preprocess_pipeline") or [],
            image_preprocess_pipelines=job.get("image_preprocess_pipelines") or {},
            preprocess_after_tiling=bool(job.get("preprocess_after_tiling", False)),
            product_resolution=self._product_resolution_for_machine(
                job.get("machine_id", ""),
                self._capi_server_instance,
            ),
        )
        panel_paths = [Path(p) for p in job.get("panel_paths", [])]
        panel_modes = job.get("panel_modes") or ["full"] * len(panel_paths)
        # 預覽優先挑 full panel，能同時看到 inner / edge；沒有 full 時 fallback 全部 panel。
        full_panel_paths = [
            p for p, m in zip(panel_paths, panel_modes) if m == "full"
        ] or panel_paths

        preview_panel_dir = None
        preview_files = None
        for panel_dir in full_panel_paths:
            if not panel_dir.exists():
                continue
            files = filter_panel_lighting_files(panel_dir)
            if lighting not in files:
                continue
            preview_panel_dir = panel_dir
            preview_files = files
            break

        if preview_panel_dir is None or preview_files is None:
            self._send_response(404, "")
            return

        def _file_fingerprint(path: Path) -> dict:
            try:
                st = path.stat()
                return {
                    "name": path.name,
                    "mtime_ns": int(st.st_mtime_ns),
                    "size": int(st.st_size),
                }
            except OSError:
                return {"name": path.name, "mtime_ns": 0, "size": 0}

        cache_payload = {
            "version": 13,
            "lighting": lighting,
            "panel_dir": str(preview_panel_dir.resolve()),
            "files": {
                key: _file_fingerprint(path)
                for key, path in sorted(preview_files.items())
            },
            "image_preprocess_pipeline": job.get("image_preprocess_pipeline") or [],
            "preprocess_after_tiling": bool(job.get("preprocess_after_tiling", False)),
            "tile_stride": preprocess_cfg.tile_stride,
            "product_resolution": preprocess_cfg.product_resolution,
        }
        cache_key = hashlib.sha1(
            json.dumps(cache_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:12]

        preview_dir = Path(".tmp/train_new_thumbs") / job_id / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"{lighting}_v13_{cache_key}.jpg"
        if preview_path.exists():
            self._send_binary(str(preview_path))
            return

        result = None
        panel_name = preview_panel_dir.name
        results = preprocess_panel_folder(
            preview_panel_dir,
            preprocess_cfg,
            image_files=preview_files.values(),
        )
        result = results.get(lighting)

        if result is None:
            self._send_response(404, "")
            return

        img = cv2.imread(str(result.image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            self._send_response(404, "")
            return
        canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        canvas = cv2.convertScaleAbs(canvas, alpha=0.55, beta=18)
        overlay = canvas.copy()

        # BGR: edge=peach (#fab387), inner=blue (#89b4fa) — 與 step3 legend 同步
        edge_color = (135, 179, 250)
        inner_color = (250, 180, 137)

        for tile in result.tiles:
            color = edge_color if tile.zone == "edge" else inner_color
            cv2.rectangle(overlay, (tile.x1, tile.y1), (tile.x2, tile.y2), color, -1)
        canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)

        def draw_rect(rect, color, thickness):
            xa, ya, xb, yb = rect
            cv2.rectangle(canvas, (xa, ya), (xb, yb), (0, 0, 0), thickness + 8)
            cv2.rectangle(canvas, (xa, ya), (xb, yb), color, thickness)

        def draw_poly(points, color, thickness):
            cv2.polylines(canvas, [points], isClosed=True, color=(0, 0, 0), thickness=thickness + 8)
            cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=thickness)

        if result.panel_polygon is not None:
            poly = result.panel_polygon.astype("int32").reshape((-1, 1, 2))
            draw_poly(poly, (80, 255, 120), 10)
            boundary_label = "boundary=green"
        else:
            # Polygon detection fallback: bbox becomes the only available foreground boundary.
            x1, y1, x2, y2 = result.foreground_bbox
            draw_rect((x1, y1, x2, y2), (80, 255, 120), 10)
            boundary_label = "boundary=green (bbox fallback)"

        for tile in result.tiles:
            color = edge_color if tile.zone == "edge" else inner_color
            draw_rect((tile.x1, tile.y1, tile.x2, tile.y2), color, 4)

        label = f"{panel_name} / {lighting}  {boundary_label}  inner=blue  edge=orange"
        cv2.rectangle(canvas, (20, 20), (min(canvas.shape[1] - 20, 1400), 102), (18, 18, 30), -1)
        cv2.rectangle(canvas, (20, 20), (min(canvas.shape[1] - 20, 1400), 102), (137, 180, 250), 3)
        cv2.putText(canvas, label, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (205, 214, 244), 3, cv2.LINE_AA)

        max_side = 1400
        h, w = canvas.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            canvas = cv2.resize(canvas, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(preview_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        self._send_binary(str(preview_path))

    def _handle_train_new_tiles_decision(self):
        """POST /api/train/new/tiles/decision
        body: {"job_id": "...", "tile_ids": [int, ...], "decision": "accept"|"reject"}
        """
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            self._send_json({"error": "invalid JSON"}, status=400)
            return

        job_id = body.get("job_id")
        tile_ids = body.get("tile_ids", [])
        decision = body.get("decision")

        if not job_id or not tile_ids or decision not in ("accept", "reject"):
            self._send_json({"error": "job_id, tile_ids, decision required"}, status=400)
            return

        db = self._capi_server_instance.database
        db.update_tile_decisions(job_id, tile_ids, decision)
        self._send_json({"ok": True, "updated": len(tile_ids)})

    def _handle_train_new_cancel(self):
        """POST /api/train/new/cancel/<job_id>

        Cancel by job_id：訓練中 job 透過該 job 自己的 cancel flag 檔通知 runner；
        review 階段 job 直接 mark failed；server restart 後 stale job 也標 failed。
        cancel flag 的路徑從 per-job runtime 取，不讀全域，避免跨 job 互打。
        """
        job_id = self.path.rsplit("/", 1)[-1].split("?")[0]
        db = self._capi_server_instance.database
        job = db.get_training_job(job_id)
        if not job:
            self._send_json({"error": "job not found"}, status=404)
            return
        job = self._mark_train_new_stale_if_needed(db, job)

        if job["state"] == "failed":
            self._send_json({"ok": True, "job_id": job_id, "state": "failed"})
            return

        runtime = CAPIWebHandler._get_job_runtime(job_id)

        if job["state"] in ("preprocess", "train"):
            if not self._train_new_worker_alive(job_id):
                db.update_training_job_state(job_id, "failed", error_message="cancelled stale job")
                CAPIWebHandler._drop_job_runtime(job_id)
                slot = CAPIWebHandler._train_slot
                with slot["lock"]:
                    if slot.get("active_job_id") == job_id:
                        slot["active_job_id"] = None
                self._send_json({"ok": True, "job_id": job_id, "state": "failed"})
                return

            # 觸發該 job 的 cancel event 並 touch 該 job 的 cancel flag 檔
            if runtime is not None:
                runtime["cancel_event"].set()
                cancel_flag = runtime.get("cancel_flag")
                if cancel_flag:
                    try:
                        Path(cancel_flag).touch()
                    except Exception:
                        pass
                CAPIWebHandler._append_train_new_log(
                    job_id, "收到取消要求，會在目前訓練階段結束後停止"
                )
            self._send_json({"ok": True, "job_id": job_id, "state": job["state"], "cancel_requested": True})
            return

        if job["state"] != "review":
            self._send_json({
                "error": f"job state must be 'review', currently '{job['state']}'"
            }, status=409)
            return

        db.update_training_job_state(job_id, "failed", error_message="cancelled by user")
        CAPIWebHandler._cleanup_train_new_job_artifacts(
            db, job_id, reason="cancelled from review"
        )
        CAPIWebHandler._drop_job_runtime(job_id)
        self._send_json({"ok": True, "job_id": job_id, "state": "failed"})

    def _handle_train_new_start_training(self):
        """POST /api/train/new/start_training/<job_id>"""
        job_id = self.path.rsplit("/", 1)[-1].split("?")[0]
        db = self._capi_server_instance.database
        job = db.get_training_job(job_id)
        if not job:
            self._send_json({"error": "job not found"}, status=404)
            return
        job = self._mark_train_new_stale_if_needed(db, job)
        logger.info(
            "[train/new] start request received: job_id=%s state=%s",
            job_id,
            job.get("state"),
        )
        if job["state"] in ("train", "completed"):
            logger.info(
                "[train/new] start request already applied: job_id=%s state=%s",
                job_id,
                job["state"],
            )
            self._send_json({
                "ok": True,
                "job_id": job_id,
                "state": job["state"],
                "already_started": True,
            })
            return
        if job["state"] != "review":
            self._send_json({"error": f"job state must be 'review', currently '{job['state']}'"}, status=409)
            return

        # GPU singleton：拿不到 train slot 就 409 並回對方 job_id
        slot = CAPIWebHandler._train_slot
        with slot["lock"]:
            if slot.get("active_job_id") is not None:
                self._send_json({
                    "error": "another_job_training",
                    "training_job_id": slot["active_job_id"],
                }, status=409)
                return
            slot["active_job_id"] = job_id

        # 確保 runtime 存在（從 review 接續，多半已存在；server 重啟後可能沒有）
        runtime = CAPIWebHandler._get_job_runtime(job_id)
        if runtime is None:
            runtime = CAPIWebHandler._make_job_runtime(job_id, "train")
        runtime["phase"] = "train"
        runtime["proc"] = None
        runtime["cancel_flag"] = None
        runtime["log_file"] = None
        runtime["unit_status"] = {}
        runtime["cancel_event"].clear()

        scope = job.get("training_scope") or {}
        if scope.get("mode") == "partial":
            target = CAPIWebHandler._train_new_partial_training_worker
            args = (job_id, self._capi_server_instance)
            thread_name = f"train_new_partial-{job_id}"
        else:
            target = CAPIWebHandler._train_new_training_worker
            args = (job_id, job["machine_id"], job["panel_paths"], self._capi_server_instance)
            thread_name = f"train_new-{job_id}"

        # 先建立 worker thread，但等 HTTP 啟動回應送完才放行耗資源的 GPU 清理／subprocess。
        # 即使 Client 在收回應時斷線，finally 仍會放行，避免 job 卡在 train state。
        worker_start_gate = threading.Event()

        def run_after_start_ack():
            worker_start_gate.wait()
            target(*args)

        thread = threading.Thread(
            target=run_after_start_ack,
            daemon=True, name=thread_name,
        )
        runtime["thread"] = thread

        db.update_training_job_state(job_id, "train")
        thread.start()
        try:
            logger.info(
                "[train/new] start state committed; worker waiting for ack: job_id=%s",
                job_id,
            )
            self._send_json({"ok": True, "job_id": job_id, "state": "train"})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as exc:
            logger.warning(
                "[train/new] start response connection lost: job_id=%s error=%s",
                job_id,
                exc,
            )
            raise
        finally:
            worker_start_gate.set()
        logger.info("[train/new] start response sent: job_id=%s", job_id)

    @staticmethod
    def _train_new_partial_training_worker(job_id, server_inst):
        """Train selected PTs for an existing bundle using the reviewed tile pool."""
        import traceback as _traceback
        from capi_train_new import TrainingConfig, apply_user_training_params, train_single_submodel, _auroc_grade
        from capi_model_registry import append_submodel_history, _read_manifest, _write_manifest, invalidate_score_cache

        db = server_inst.database
        runtime = CAPIWebHandler._get_job_runtime(job_id)
        if runtime is None:
            runtime = CAPIWebHandler._make_job_runtime(job_id, "train")
        runtime["phase"] = "train"

        def log(msg):
            CAPIWebHandler._append_train_new_log(job_id, msg)

        try:
            log("準備訓練資源：正在停止模型掃描並釋放 GPU")
            CAPIWebHandler._cancel_and_wait_scan_idle(timeout_s=15.0)
            CAPIWebHandler._free_server_gpu_cache()

            job = db.get_training_job(job_id)
            if not job:
                raise RuntimeError(f"找不到 job_id={job_id}")
            scope = job.get("training_scope") or {}
            if scope.get("mode") != "partial":
                raise RuntimeError("partial training worker received non-partial job")

            bundle_id = int(scope["target_bundle_id"])
            bundle = db.get_model_bundle(bundle_id)
            if not bundle:
                raise RuntimeError(f"target bundle {bundle_id} 已不存在")

            bundle_dir = Path(bundle["bundle_path"])
            manifest = _read_manifest(bundle_dir)
            patchcore_params = manifest.get("patchcore_params") or {}
            train_cfg = CAPIWebHandler._load_train_new_config(server_inst)
            cfg = TrainingConfig(
                machine_id=job["machine_id"],
                panel_paths=[Path(p) for p in job.get("panel_paths", [])],
                over_review_root=train_cfg["over_review_root"],
                output_root=train_cfg["output_root"],
                backbone_cache_dir=train_cfg["backbone_cache_dir"],
                required_backbones=train_cfg["required_backbones"],
                image_preprocess_pipeline=job.get("image_preprocess_pipeline") or [],
                image_preprocess_pipelines=job.get("image_preprocess_pipelines") or {},
                preprocess_after_tiling=bool(job.get("preprocess_after_tiling", False)),
                tile_stride=int(job.get("tile_stride") or 512),
                batch_size=patchcore_params.get("batch_size", 8),
                image_size=tuple(patchcore_params.get("image_size", (512, 512))),
                coreset_ratio=patchcore_params.get("coreset_ratio", 0.1),
                max_epochs=patchcore_params.get("max_epochs", 1),
                precision=patchcore_params.get("precision", "float32"),
                feature_layers=patchcore_params.get("feature_layers", "layer2_layer3"),
                feature_pool_kernel_size=patchcore_params.get("feature_pool_kernel_size", 3),
                feature_cleaning_mode=patchcore_params.get("feature_cleaning_mode", "off"),
                feature_cleaning_scope=patchcore_params.get(
                    "feature_cleaning_scope", "inner_only",
                ),
                feature_cleaning_keep_ratio=patchcore_params.get(
                    "feature_cleaning_keep_ratio", 0.99,
                ),
                feature_cleaning_center_size=patchcore_params.get(
                    "feature_cleaning_center_size", 512,
                ),
                feature_cleaning_by_zone=patchcore_params.get(
                    "feature_cleaning_by_zone"
                ) or {},
            )
            partial_training_params = dict(job.get("training_params") or {})
            for key in CAPIWebHandler.PATCHCORE_BUNDLE_LOCKED_TRAINING_PARAMS:
                partial_training_params.pop(key, None)
            apply_user_training_params(cfg, partial_training_params, log_fn=log)

            selected_units = CAPIWebHandler._scope_selected_units(scope)
            if not selected_units:
                raise RuntimeError("selected_units is empty")

            log(
                f"局部重訓開始：bundle_id={bundle_id}, units="
                + ", ".join(f"{l}-{z}" for l, z in selected_units)
            )

            summaries = {}
            total = len(selected_units)
            for idx, (lighting, zone) in enumerate(selected_units, 1):
                if runtime["cancel_event"].is_set():
                    raise RuntimeError("training cancelled by user")

                unit_label = f"{lighting}-{zone}"
                output_pt = bundle_dir / f"{unit_label}.pt"
                old_manifest = _read_manifest(bundle_dir)
                old_unit_metrics = (old_manifest.get("unit_metrics") or {}).get(unit_label) or {}
                old_history = (old_manifest.get("submodel_history") or {}).get(unit_label) or []
                if old_history:
                    old_auroc = old_history[-1].get("auroc")
                    old_tile_count = old_history[-1].get("tile_count_used")
                else:
                    old_auroc = old_unit_metrics.get("auroc")
                    old_tile_count = old_unit_metrics.get("train_count")

                log(f"[{idx}/{total}] {unit_label}: 載 tile")
                result = train_single_submodel(
                    db=db,
                    job_id=job_id,
                    lighting=lighting,
                    zone=zone,
                    cfg=cfg,
                    output_pt_path=output_pt,
                    gpu_lock=server_inst._gpu_lock,
                    log=log,
                    cancel_event=runtime["cancel_event"],
                    unit_prefix=f"[{idx}/{total}] ",
                )

                metrics = result["metrics"]
                metrics["used_tile_ids"] = result["used_tile_ids"]
                new_auroc = metrics.get("auroc")
                new_tile_count = result["tile_count"]
                entry = {
                    "trained_at": datetime.now().isoformat(timespec="seconds"),
                    "job_id": job_id,
                    "trained_with_job_id": job_id,
                    "tile_count_used": new_tile_count,
                    "auroc": new_auroc,
                    "used_tile_ids": result["used_tile_ids"],
                    "kind": "partial_retrain",
                    "ng_used": result["ng_used"],
                    "panel_count": len(job.get("panel_paths") or []),
                    "panel_glass_ids": [Path(p).name for p in job.get("panel_paths", [])],
                    "feature_pool_kernel_size": metrics.get(
                        "feature_pool_kernel_size", cfg.feature_pool_kernel_size,
                    ),
                    "feature_cleaning_mode": cfg.feature_cleaning_mode,
                    "feature_cleaning": metrics.get("feature_cleaning") or {},
                }
                append_submodel_history(bundle_dir, lighting, zone, entry)
                refreshed_manifest = _read_manifest(bundle_dir)
                refreshed_manifest.setdefault("unit_metrics", {})[unit_label] = metrics
                refreshed_manifest.setdefault("tiles_per_unit", {})[unit_label] = {
                    "train": result["tile_count"],
                    "ng": result["ng_count"],
                }
                refreshed_manifest.setdefault("model_files", {})[unit_label] = {
                    "path": output_pt.name,
                    "size_bytes": result["size_bytes"],
                }
                auroc_values = [
                    m.get("auroc") for m in refreshed_manifest.get("unit_metrics", {}).values()
                    if m.get("auroc") is not None
                ]
                overall_auroc = round(sum(auroc_values) / len(auroc_values), 4) if auroc_values else None
                refreshed_manifest["overall_auroc"] = overall_auroc
                refreshed_manifest["overall_auroc_grade"] = _auroc_grade(overall_auroc)
                _write_manifest(bundle_dir, refreshed_manifest)
                cleared = invalidate_score_cache(
                    db, scoring_bundle_id=bundle_id, lighting=lighting, zone=zone,
                )
                log(f"[{idx}/{total}] {unit_label}: 清除 {cleared} 筆 score cache")

                summaries[unit_label] = {
                    "auroc_old": old_auroc,
                    "auroc_new": new_auroc,
                    "tile_count_old": old_tile_count,
                    "tile_count_new": new_tile_count,
                }

                caught = metrics.get("ng_caught_count", 0)
                ng_n = metrics.get("ng_count", 0)
                auroc_str = f", AUROC={new_auroc:.3f}({metrics.get('auroc_grade','')})" if new_auroc is not None else ""
                log(
                    f"[{idx}/{total}] {unit_label}: ✓ done | {result['elapsed_seconds']}s, "
                    f"threshold={result['threshold']:.4f}, size={result['size_bytes']/1e6:.1f}MB, "
                    f"ng_caught={caught}/{ng_n}{auroc_str}"
                )

            inferencer = server_inst.inferencers.get(job["machine_id"])
            if inferencer is None:
                log(f"[v2] 機台 {job['machine_id']} 無 inferencer cache，跳過 reload")
            else:
                for lighting, zone in selected_units:
                    try:
                        inferencer.reload_submodel(job["machine_id"], lighting, zone)
                        log(f"[v2] 已通知 inferencer reload {job['machine_id']}/{lighting}/{zone}")
                    except Exception as reload_err:
                        log(f"[v2] reload 失敗（不影響重訓結果）：{reload_err}")
                        logger.warning("reload_submodel raised: %s", reload_err, exc_info=True)

            db.update_training_job_state(job_id, "completed", output_bundle=str(bundle_dir))
            log(f"✓ 局部重訓完成，bundle={bundle_dir}")

        except Exception as e:
            _traceback.print_exc()
            msg = "取消" if runtime["cancel_event"].is_set() else str(e)
            try:
                db.update_training_job_state(job_id, "failed", error_message=msg)
            except Exception:
                pass
            log(f"✗ 局部重訓失敗: {msg}")
            for line in _traceback.format_exc().rstrip().splitlines()[-8:]:
                log(f"  {line}")
        finally:
            try:
                finished_job = db.get_training_job(job_id)
                if finished_job and finished_job.get("state") == "failed":
                    CAPIWebHandler._cleanup_train_new_job_artifacts(
                        db, job_id, reason="partial training failed"
                    )
            except Exception:
                logger.warning(
                    "cannot finalize partial training cleanup: job_id=%s",
                    job_id,
                    exc_info=True,
                )
            slot = CAPIWebHandler._train_slot
            with slot["lock"]:
                if slot.get("active_job_id") == job_id:
                    slot["active_job_id"] = None
            CAPIWebHandler._drop_job_runtime(job_id)

    @staticmethod
    def _train_new_training_worker(job_id, machine_id, panel_paths, server_inst):
        """Supervisor thread：launch 訓練 subprocess、tail log、偵測退出、清狀態。

        實際訓練在 capi_train_runner.py 的獨立 Python process 執行，使其能與
        推論 server 共用 GPU 而不互鎖（兩邊各自設 set_per_process_memory_fraction）。
        Subprocess 自行寫 DB（model_registry + training_jobs.state）；supervisor
        搬運 log、校正成功完成狀態，並在退出後釋放 train slot。
        """
        import os as _os
        import subprocess as _subprocess
        import sys as _sys
        import time as _time
        import traceback as _traceback
        from pathlib import Path as _Path

        runtime = CAPIWebHandler._get_job_runtime(job_id)
        if runtime is None:
            runtime = CAPIWebHandler._make_job_runtime(job_id, "train")
        runtime["phase"] = "train"
        db = server_inst.database

        def log(msg):
            CAPIWebHandler._append_train_new_log(job_id, msg)

        proc = None
        try:
            log("準備訓練資源：正在停止模型掃描並釋放 GPU")
            CAPIWebHandler._cancel_and_wait_scan_idle(timeout_s=15.0)
            CAPIWebHandler._free_server_gpu_cache()

            train_cfg = CAPIWebHandler._load_train_new_config(server_inst)
            output_root = _Path(train_cfg["output_root"])
            log_dir = output_root / "training_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{job_id}.log"
            cancel_flag = log_dir / f"{job_id}.cancel"
            log_file.write_text("", encoding="utf-8")
            try:
                cancel_flag.unlink()
            except FileNotFoundError:
                pass

            server_cfg_path = _Path(server_inst.server_config_path).resolve()
            project_root = _Path(__file__).resolve().parent

            cmd = [
                _sys.executable, "-u", "-m", "capi_train_runner",
                "--job-id", job_id,
                "--server-config", str(server_cfg_path),
                "--log-file", str(log_file),
                "--cancel-flag", str(cancel_flag),
            ]
            log(f"啟動訓練 subprocess（VRAM 隔離模式）")

            env = {**_os.environ, "PYTHONUNBUFFERED": "1"}
            proc = _subprocess.Popen(
                cmd,
                cwd=str(project_root),
                env=env,
                stdout=_subprocess.DEVNULL,
                stderr=_subprocess.STDOUT,
            )

            runtime["proc"] = proc
            runtime["log_file"] = str(log_file)
            runtime["cancel_flag"] = str(cancel_flag)

            log(f"訓練 subprocess pid={proc.pid}，log={log_file}")

            tail_pos = 0

            def _drain_log():
                nonlocal tail_pos
                try:
                    with open(log_file, "rb") as f:
                        f.seek(tail_pos)
                        data = f.read()
                        tail_pos = f.tell()
                except FileNotFoundError:
                    return
                if not data:
                    return
                text = data.decode("utf-8", errors="replace")
                for line in text.splitlines():
                    line = line.rstrip()
                    if line:
                        CAPIWebHandler._append_train_new_log(job_id, line)

            while True:
                _drain_log()
                ret = proc.poll()
                if ret is not None:
                    _drain_log()
                    if ret == 0:
                        log(f"✓ 訓練 subprocess 結束 (exit=0)")
                    else:
                        log(f"✗ 訓練 subprocess 結束 (exit={ret})")
                    break
                _time.sleep(1.0)

            if proc.returncode == 0:
                try:
                    with runtime["log_lock"]:
                        completed_bundle = runtime.get("completed_bundle")
                    finished_job = db.get_training_job(job_id)
                    synced_job = CAPIWebHandler._sync_train_new_completed_state(
                        db,
                        finished_job,
                        completed_bundle,
                    )
                    if (
                        finished_job
                        and finished_job.get("state") == "train"
                        and synced_job
                        and synced_job.get("state") == "completed"
                    ):
                        log("✓ Web 訓練狀態已同步為 completed")
                    elif finished_job and finished_job.get("state") == "train":
                        log("⚠ runner 已成功結束，但未取得完成 bundle，保留 train 狀態")
                except Exception:
                    logger.warning(
                        "cannot reconcile successful training state: job_id=%s",
                        job_id,
                        exc_info=True,
                    )
            else:
                try:
                    job = db.get_training_job(job_id)
                    if job and job.get("state") == "train":
                        db.update_training_job_state(
                            job_id, "failed",
                            error_message=f"runner exited with code {proc.returncode}",
                        )
                except Exception:
                    pass

        except Exception as e:
            _traceback.print_exc()
            log(f"✗ 訓練監看失敗: {e}")
            try:
                db.update_training_job_state(job_id, "failed", error_message=str(e))
            except Exception:
                pass
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        finally:
            try:
                finished_job = db.get_training_job(job_id)
                if finished_job and finished_job.get("state") == "failed":
                    CAPIWebHandler._cleanup_train_new_job_artifacts(
                        db, job_id, reason="training worker failed"
                    )
            except Exception:
                logger.warning(
                    "cannot finalize failed training cleanup: job_id=%s",
                    job_id,
                    exc_info=True,
                )
            # 釋放訓練槽（讓下一個排隊的 job 能進入 train）
            slot = CAPIWebHandler._train_slot
            with slot["lock"]:
                if slot.get("active_job_id") == job_id:
                    slot["active_job_id"] = None
            # 訓練 done/failed 都把 runtime 清掉
            CAPIWebHandler._drop_job_runtime(job_id)

    def _handle_train_new_thumb(self):
        """GET /api/train/new/thumb/<rest_of_path>"""
        from urllib.parse import unquote
        parts = self.path.split("/api/train/new/thumb/", 1)[1].split("?")[0]
        parts = unquote(parts)
        safe = Path(".tmp/train_new_thumbs").resolve()
        target = (safe / parts).resolve()
        try:
            target.relative_to(safe)
        except ValueError:
            self._send_response(403, "")
            return
        if not target.is_file():
            self._send_response(404, "")
            return
        self._send_binary(str(target))

    def _handle_train_new_bundle_asset(self):
        """Serve persisted feature-cleaning images from a completed bundle."""
        from urllib.parse import unquote

        encoded = self.path.split("/api/train/new/bundle-asset/", 1)[1].split("?", 1)[0]
        parts = unquote(encoded).split("/", 1)
        if len(parts) != 2:
            self._send_response(404, "")
            return
        job_id, relative_path = parts
        job = self._capi_server_instance.database.get_training_job(job_id)
        if not job or not job.get("output_bundle"):
            self._send_response(404, "")
            return
        bundle_path = Path(job["output_bundle"]).resolve()
        asset_root = (bundle_path / "feature_cleaning_reports" / "assets").resolve()
        target = (bundle_path / relative_path).resolve()
        try:
            target.relative_to(asset_root)
        except ValueError:
            self._send_response(403, "")
            return
        if not target.is_file():
            self._send_response(404, "")
            return
        self._send_binary(str(target))

    def _train_new_thumb_url(self, thumb_path: str) -> str:
        """Convert a stored thumbnail path to the confined thumbnail route."""
        if not thumb_path:
            return ""
        safe = Path(".tmp/train_new_thumbs").resolve()
        target = Path(thumb_path)
        if not target.is_absolute():
            target = target.resolve()
        else:
            target = target.resolve()
        try:
            rel = target.relative_to(safe)
        except ValueError:
            return ""
        return "/api/train/new/thumb/" + urllib.parse.quote(rel.as_posix(), safe="/")

    def _retrain_pool_base_dir(self) -> Path:
        server_inst = self._capi_server_instance
        cfg = server_inst.server_config.get("retrain_pool", {}) if server_inst else {}
        if cfg.get("base_dir"):
            return Path(cfg["base_dir"]).resolve()
        return (self._dataset_export_base_dir().parent / "retrain_pool").resolve()

    def _retrain_pool_file_url(self, pool_id: int, kind: str = "thumb") -> str:
        return f"/api/retrain-pool/file?id={int(pool_id)}&kind={urllib.parse.quote(kind)}"

    def _handle_retrain_pool_file(self, query: dict):
        def _q(key, default=""):
            value = query.get(key, default)
            if isinstance(value, list):
                return value[0] if value else default
            return value

        try:
            pool_id = int(_q("id", "0"))
        except (TypeError, ValueError):
            self._send_error(400, "invalid id")
            return
        kind = _q("kind", "thumb")
        items = self._capi_server_instance.database.get_over_retrain_pool_items([pool_id])
        if not items:
            self._send_404()
            return
        item = items[0]
        path_str = item.get("thumb_path") if kind == "thumb" else item.get("source_path")
        if not path_str:
            path_str = item.get("source_path")
        target = Path(path_str).resolve()
        base = self._retrain_pool_base_dir()
        try:
            target.relative_to(base)
        except ValueError:
            self._send_error(403, "path outside retrain pool")
            return
        if not target.is_file():
            self._send_404()
            return
        self._send_binary(str(target))

    def _handle_retrain_pool_page(self):
        from capi_train_new import LIGHTINGS
        today = datetime.now().strftime("%Y-%m-%d")
        template = self.jinja_env.get_template("retrain_pool.html")
        html = template.render(
            request_path="/models/retrain-pool",
            lightings=list(LIGHTINGS),
            today=today,
        )
        self._send_response(200, html)

    def _handle_retrain_pool_list(self, query: dict):
        def _q(key, default=""):
            value = query.get(key, default)
            if isinstance(value, list):
                return value[0] if value else default
            return value

        try:
            limit = max(1, min(int(_q("limit", "500")), 1000))
            offset = max(0, int(_q("offset", "0")))
            client_record_id = _q("client_record_id", "")
            client_record_id = int(client_record_id) if client_record_id else None
            rows, total = self._capi_server_instance.database.list_over_retrain_pool(
                start_date=_q("start_date", "") or None,
                end_date=_q("end_date", "") or None,
                machine_id=_q("machine_id", "") or None,
                lighting=_q("lighting", "") or None,
                zone=_q("zone", None),
                client_record_id=client_record_id,
                limit=limit,
                offset=offset,
            )
        except ValueError as e:
            self._send_json({"error": str(e)}, status=400)
            return

        items = []
        for row in rows:
            item = dict(row)
            item["thumb_url"] = self._retrain_pool_file_url(item["id"], "thumb")
            item["image_url"] = self._retrain_pool_file_url(item["id"], "source")
            item["created_date"] = (item.get("created_at") or "")[:10]
            items.append(item)
        self._send_json({"items": items, "total": total, "limit": limit, "offset": offset})

    def _safe_unlink_retrain_pool_file(self, path_str: str) -> bool:
        if not path_str:
            return False
        target = Path(path_str).resolve()
        base = self._retrain_pool_base_dir()
        try:
            target.relative_to(base)
        except ValueError:
            return False
        if target.is_file():
            target.unlink()
            return True
        return False

    def _safe_unlink_train_new_file(self, path_str: str) -> bool:
        if not path_str:
            return False
        target = Path(path_str).resolve()
        base = Path(".tmp/train_new_thumbs").resolve()
        try:
            target.relative_to(base)
        except ValueError:
            return False
        if target.is_file():
            target.unlink()
            return True
        return False

    @staticmethod
    def _retrain_pool_training_paths(item: dict) -> Tuple[Path, Path]:
        unit = str(item.get("added_to_unit") or "")
        if "-" in unit:
            lighting, zone = unit.rsplit("-", 1)
        else:
            lighting = str(item.get("lighting") or "")
            zone = str(item.get("zone") or "")
        job_id = str(item.get("added_to_job_id") or "")
        src_path = Path(item.get("source_path") or "")
        filename = f"pool_{int(item['id']):06d}_{src_path.name}"
        train_pool_dir = Path(".tmp/train_new_thumbs") / job_id / "retrain_pool" / lighting / zone
        return (
            (train_pool_dir / "tiles" / filename).resolve(),
            (train_pool_dir / "thumb" / filename).resolve(),
        )

    def _handle_retrain_pool_unadd(self):
        data = self._read_json_body()
        if data is None:
            return
        try:
            pool_id = int(data.get("id"))
        except (TypeError, ValueError):
            self._send_json({"error": "id 必須是整數"}, status=400)
            return

        db = self._capi_server_instance.database
        items = db.get_over_retrain_pool_items([pool_id])
        if not items:
            self._send_json({"error": "pool item not found"}, status=404)
            return
        item = items[0]
        job_id = str(item.get("added_to_job_id") or "")
        if not job_id:
            self._send_json({"ok": True, "message": "此 Pool item 尚未加入訓練清單"})
            return
        if self._train_new_worker_alive(job_id):
            self._send_json({"error": "此訓練 job 仍在執行中，請先停止訓練後再移出清單"}, status=409)
            return

        train_source_path, train_thumb_path = self._retrain_pool_training_paths(item)
        deleted_rows = db.delete_tile_pool_by_source_paths(job_id, [str(train_source_path)])
        deleted_files = 0
        deleted_files += int(self._safe_unlink_train_new_file(str(train_thumb_path)))
        deleted_files += int(self._safe_unlink_train_new_file(str(train_source_path)))
        updated_rows = db.clear_over_retrain_pool_added([pool_id])
        self._send_json({
            "ok": True,
            "pool_id": pool_id,
            "deleted_training_rows": deleted_rows,
            "deleted_files": deleted_files,
            "updated_rows": updated_rows,
            "message": "已移出訓練清單；Pool 原始資料仍保留",
        })

    def _handle_retrain_pool_delete(self):
        data = self._read_json_body()
        if data is None:
            return
        try:
            pool_id = int(data.get("id"))
        except (TypeError, ValueError):
            self._send_json({"error": "id 必須是整數"}, status=400)
            return

        db = self._capi_server_instance.database
        items = db.get_over_retrain_pool_items([pool_id])
        if not items:
            self._send_json({"error": "pool item not found"}, status=404)
            return
        if items[0].get("added_to_job_id"):
            self._send_json({
                "error": "此 Pool item 已加入訓練清單，請先移出清單後再刪除",
            }, status=409)
            return

        item = db.delete_over_retrain_pool_item(pool_id)
        if not item:
            self._send_json({"error": "pool item not found"}, status=404)
            return

        deleted_files = 0
        deleted_files += int(self._safe_unlink_retrain_pool_file(item.get("thumb_path") or ""))
        deleted_files += int(self._safe_unlink_retrain_pool_file(item.get("source_path") or ""))
        self._send_json({
            "ok": True,
            "deleted_id": pool_id,
            "deleted_files": deleted_files,
            "kept_files": False,
        })

    @staticmethod
    def _pool_ok_prefixes(datastr: str) -> set:
        parsed = parse_datastr_per_prefix(datastr or "")
        return {prefix for prefix, result in parsed.items() if result == "OK"}

    def _handle_over_retrain_pool_add(self):
        data = self._read_json_body()
        if data is None:
            return
        try:
            client_record_id = int(data.get("client_record_id"))
        except (TypeError, ValueError):
            self._send_json({"success": False, "error": "client_record_id 必須是整數"}, status=400)
            return

        from capi_database import CAPIDatabase
        import cv2

        db = self._capi_server_instance.database
        record = db.get_client_accuracy_record(client_record_id)
        if not record:
            self._send_json({"success": False, "error": "找不到 client record"}, status=404)
            return

        ai = str(record.get("result_ai") or "")
        ric = CAPIDatabase.parse_ric_judgment(record.get("datastr") or "")
        if not ai.startswith("NG") or ric != "OK":
            self._send_json({"success": False, "error": "只有 AI=NG 且 RIC=OK 的過檢紀錄可加入重訓 Pool"}, status=400)
            return

        inference_record_id = record.get("inference_record_id")
        if not inference_record_id:
            self._send_json({"success": False, "error": "此筆沒有對應推論紀錄，無法裁 tile"}, status=400)
            return

        ok_prefixes = self._pool_ok_prefixes(record.get("datastr") or "")
        if not ok_prefixes:
            self._send_json({"success": False, "error": "DATASTR 解析不到 OK 畫面，未自動加入"}, status=400)
            return

        detail = db.get_record_detail(int(inference_record_id))
        if not detail:
            self._send_json({"success": False, "error": "找不到推論明細"}, status=404)
            return

        server_inst = self._capi_server_instance
        path_mapping = getattr(server_inst, "path_mapping", {}) if server_inst else {}
        base_dir = self._retrain_pool_base_dir()
        client_date = (record.get("time_stamp") or datetime.now().strftime("%Y-%m-%d"))[:10]
        machine_id = detail.get("model_id") or record.get("mach_id") or ""
        machine_no = detail.get("machine_no") or record.get("mach_id") or ""
        safe_machine_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", machine_id or "unknown")[:80] or "unknown"

        rows = []
        skipped = []
        for image in detail.get("images", []):
            image_name = image.get("image_name") or Path(image.get("image_path") or "").name
            screen_prefix = extract_prefix(image_name)
            if screen_prefix not in ok_prefixes:
                continue
            if int(image.get("is_bomb") or 0):
                skipped.append({"image": image_name, "reason": "bomb image"})
                continue
            source_image_path = resolve_source_path(image.get("image_path") or "", path_mapping)
            image_bgr = self._read_inference_image(source_image_path, cv2.IMREAD_UNCHANGED)
            if image_bgr is None:
                skipped.append({"image": image_name, "reason": f"read failed: {source_image_path}"})
                continue

            for tile in image.get("tiles", []):
                if not int(tile.get("is_anomaly") or 0):
                    continue
                if int(tile.get("is_bomb") or 0) or int(tile.get("is_exclude_zone") or 0):
                    continue
                tile_id = int(tile.get("id"))
                x = max(0, int(tile.get("x") or 0))
                y = max(0, int(tile.get("y") or 0))
                w = max(1, int(tile.get("width") or 0))
                h = max(1, int(tile.get("height") or 0))
                crop = image_bgr[y:y + h, x:x + w].copy()
                if crop.size == 0:
                    skipped.append({"tile_result_id": tile_id, "reason": "empty crop"})
                    continue
                pad_bottom = max(0, h - crop.shape[0])
                pad_right = max(0, w - crop.shape[1])
                if pad_bottom or pad_right:
                    crop = cv2.copyMakeBorder(
                        crop, 0, pad_bottom, 0, pad_right,
                        cv2.BORDER_REPLICATE,
                )

                zone = tile.get("zone") or ""
                safe_screen = re.sub(r"[^A-Za-z0-9_.-]+", "_", screen_prefix or "unknown")[:80] or "unknown"
                safe_zone = re.sub(r"[^A-Za-z0-9_.-]+", "_", zone or "unknown")[:40] or "unknown"
                out_dir = base_dir / client_date / safe_machine_id / safe_screen / safe_zone
                tile_dir = out_dir / "tiles"
                thumb_dir = out_dir / "thumb"
                tile_dir.mkdir(parents=True, exist_ok=True)
                thumb_dir.mkdir(parents=True, exist_ok=True)
                filename = f"c{client_record_id}_r{inference_record_id}_i{image.get('id')}_t{tile_id}.png"
                tile_path = tile_dir / filename
                thumb_path = thumb_dir / filename
                if not tile_path.exists() and not cv2.imwrite(str(tile_path), crop):
                    skipped.append({"tile_result_id": tile_id, "reason": "write crop failed"})
                    continue
                if not thumb_path.exists():
                    thumb = cv2.resize(crop, (96, 96))
                    cv2.imwrite(str(thumb_path), thumb)

                rows.append({
                    "client_record_id": client_record_id,
                    "inference_record_id": int(inference_record_id),
                    "tile_result_id": tile_id,
                    "image_result_id": int(image.get("id")),
                    "machine_id": machine_id,
                    "machine_no": machine_no,
                    "pnl_id": record.get("pnl_id") or "",
                    "client_time_stamp": record.get("time_stamp") or "",
                    "datastr": record.get("datastr") or "",
                    "screen_prefix": screen_prefix,
                    "lighting": screen_prefix,
                    "zone": zone,
                    "source_path": str(tile_path.resolve()),
                    "thumb_path": str(thumb_path.resolve()),
                    "tile_x": x,
                    "tile_y": y,
                    "tile_w": w,
                    "tile_h": h,
                    "score": float(tile.get("score") or 0.0),
                })

        if not rows:
            self._send_json({
                "success": False,
                "error": "沒有可加入的過檢 tile",
                "skipped": skipped[:20],
            }, status=400)
            return

        result = db.insert_over_retrain_pool_rows(rows)
        _, total_for_record = db.list_over_retrain_pool(
            client_record_id=client_record_id,
            limit=1,
            offset=0,
        )
        self._send_json({
            "success": True,
            "inserted": len(result["inserted_ids"]),
            "existing": len(result["existing_ids"]),
            "pool_ids": result["inserted_ids"] + result["existing_ids"],
            "total_for_record": total_for_record,
            "skipped": skipped[:20],
            "message": f"已加入 {len(result['inserted_ids'])} 個 tile，既有 {len(result['existing_ids'])} 個",
        })

    def _handle_models_retrain_pool_add(self):
        from capi_train_new import LIGHTINGS, ZONES

        parts = self.path.split("/")
        try:
            bundle_id = int(parts[3])
        except (ValueError, IndexError):
            self._send_json({"error": "invalid bundle id"}, status=400)
            return
        data = self._read_json_body()
        if data is None:
            return
        pool_ids = data.get("pool_ids")
        lighting = data.get("lighting")
        zone = data.get("zone")
        if not isinstance(pool_ids, list) or not pool_ids:
            self._send_json({"error": "pool_ids 必須是非空陣列"}, status=400)
            return
        if lighting not in LIGHTINGS:
            self._send_json({"error": f"lighting 必須為 {LIGHTINGS}"}, status=400)
            return
        if zone not in ZONES:
            self._send_json({"error": f"zone 必須為 {ZONES}"}, status=400)
            return

        db = self._capi_server_instance.database
        bundle = db.get_model_bundle(bundle_id)
        if not bundle:
            self._send_json({"error": "bundle not found"}, status=404)
            return
        job_id = bundle.get("job_id") or ""
        if not job_id:
            self._send_json({"error": "此 bundle 無關聯 job_id（訓練資料已刪），無法匯入 Pool"}, status=400)
            return

        try:
            pool_ids = [int(x) for x in pool_ids]
        except (TypeError, ValueError):
            self._send_json({"error": "pool_ids 必須為整數陣列"}, status=400)
            return
        items = db.get_over_retrain_pool_items(pool_ids)
        if len(items) != len(set(pool_ids)):
            self._send_json({"error": "pool_ids 含不存在項目"}, status=404)
            return

        already_added = [i for i in items if i.get("added_to_job_id")]
        if already_added:
            units = sorted({
                str(i.get("added_to_unit") or i.get("added_to_job_id") or "")
                for i in already_added
            })
            suffix = f"（{', '.join(u for u in units if u)})" if units else ""
            self._send_json({
                "error": f"有 {len(already_added)} 筆 Pool tile 已加入過訓練清單，不能重複加入{suffix}",
                "already_added_ids": [int(i["id"]) for i in already_added],
            }, status=409)
            return

        wrong_machine = [i for i in items if i.get("machine_id") != bundle.get("machine_id")]
        if wrong_machine:
            self._send_json({"error": f"有 {len(wrong_machine)} 筆 Pool 機種不屬於此 bundle"}, status=400)
            return
        wrong_lighting = [i for i in items if i.get("lighting") != lighting]
        if wrong_lighting:
            self._send_json({"error": f"有 {len(wrong_lighting)} 筆 Pool lighting 不是 {lighting}"}, status=400)
            return
        wrong_zone = [i for i in items if i.get("zone") and i.get("zone") != zone]
        if wrong_zone:
            self._send_json({"error": f"有 {len(wrong_zone)} 筆 Pool zone 不是 {zone}"}, status=400)
            return

        missing_files = [i for i in items if not Path(i.get("source_path") or "").is_file()]
        if missing_files:
            self._send_json({"error": f"有 {len(missing_files)} 筆 Pool 圖檔不存在，請先刪除或重建"}, status=400)
            return

        import shutil as _shutil
        train_pool_dir = Path(".tmp/train_new_thumbs") / job_id / "retrain_pool" / lighting / zone
        train_tile_dir = train_pool_dir / "tiles"
        train_thumb_dir = train_pool_dir / "thumb"
        train_tile_dir.mkdir(parents=True, exist_ok=True)
        train_thumb_dir.mkdir(parents=True, exist_ok=True)
        existing_paths = {
            str(Path(t["source_path"]).resolve())
            for t in db.list_tile_pool(job_id, lighting=lighting, zone=zone, source="ok")
        }
        new_tiles = []
        inserted_pool_ids = []
        existing_count = 0
        for item in items:
            src_path = Path(item["source_path"]).resolve()
            src_thumb = Path(item.get("thumb_path") or item["source_path"]).resolve()
            filename = f"pool_{int(item['id']):06d}_{src_path.name}"
            train_source_path = (train_tile_dir / filename).resolve()
            train_thumb_path = (train_thumb_dir / filename).resolve()
            if str(train_source_path) in existing_paths:
                existing_count += 1
                continue
            if not train_source_path.exists():
                _shutil.copy2(src_path, train_source_path)
            if not train_thumb_path.exists():
                _shutil.copy2(src_thumb if src_thumb.is_file() else src_path, train_thumb_path)
            new_tiles.append({
                "lighting": lighting,
                "zone": zone,
                "source": "ok",
                "source_path": str(train_source_path),
                "thumb_path": str(train_thumb_path),
            })
            inserted_pool_ids.append(int(item["id"]))
            existing_paths.add(str(train_source_path))

        if new_tiles:
            db.insert_tile_pool(job_id, new_tiles)
        db.mark_over_retrain_pool_added(pool_ids, bundle_id, job_id, f"{lighting}-{zone}")
        self._send_json({
            "ok": True,
            "inserted": len(new_tiles),
            "existing": existing_count,
            "pool_ids": pool_ids,
            "inserted_pool_ids": inserted_pool_ids,
            "unit": f"{lighting}-{zone}",
            "message": f"已加入訓練清單 {len(new_tiles)} 個 Pool tile，既有 {existing_count} 個",
        })

    @staticmethod
    def _decorate_tiles_with_scores(tiles: list, db, score_from_bundle_id, sort_by: str) -> list:
        """In-place 為 tile dict 加 score / score_quartile，並依 sort_by 排序。"""
        try:
            score_from_bundle_id = int(score_from_bundle_id) if score_from_bundle_id else None
        except (TypeError, ValueError):
            return tiles
        if not score_from_bundle_id:
            return tiles
        tile_ids = [t["id"] for t in tiles]
        scores = db.get_score_cache(score_from_bundle_id, tile_ids)
        present = sorted(
            [s for tid, s in scores.items()], reverse=True,
        )
        if present:
            top5_idx = max(0, int(len(present) * 0.05) - 1)
            top20_idx = max(0, int(len(present) * 0.20) - 1)
            top5_cut = present[top5_idx]
            top20_cut = present[top20_idx]
        else:
            top5_cut = top20_cut = float("inf")
        for tile in tiles:
            s = scores.get(tile["id"])
            tile["score"] = s
            if s is None:
                tile["score_quartile"] = None
            elif s >= top5_cut:
                tile["score_quartile"] = "top5"
            elif s >= top20_cut:
                tile["score_quartile"] = "top20"
            else:
                tile["score_quartile"] = "rest"
        if sort_by == "score_desc":
            tiles.sort(key=lambda t: (t.get("score") is None, -(t.get("score") or 0)))
        return tiles

    def _handle_retrain_status(self):
        """GET /api/retrain/status"""
        state = self._retrain_state
        with state["lock"]:
            job = state["job"]
            if not job:
                self._send_json({"state": "idle"})
                return
            snapshot = {k: job[k] for k in
                        ("job_id", "state", "step", "started_at", "output_path", "summary", "error")}
            log_lock = job["_log_lock"]

        with log_lock:
            log_lines = list(job["log_lines"][-100:])

        resp = {**snapshot, "log_lines": log_lines}
        try:
            started = datetime.fromisoformat(resp["started_at"])
            resp["elapsed_sec"] = round((datetime.now() - started).total_seconds(), 1)
        except Exception:
            resp["elapsed_sec"] = 0

        self._send_json(resp)

    @staticmethod
    def _retrain_worker(job: dict, params: dict) -> None:
        """Background thread: merge manifests, then supervise training subprocess."""
        import os as _os
        import subprocess as _subprocess
        import sys as _sys
        import time as _time
        import traceback
        from tools.merge_over_review_manifests import run as merge_run

        log_lines = job["log_lines"]
        log_lock = job["_log_lock"]

        def _append_log(message: str) -> None:
            with log_lock:
                log_lines.append(message)

        try:
            # Step 1: merge manifests
            with CAPIWebHandler._retrain_state["lock"]:
                job["step"] = "merge"

            base = Path(params["manifest_base"])
            _append_log(f"[merge] 掃描資料目錄 {base} ...")

            def _log_merge_progress(message: str) -> None:
                _append_log(f"[merge] {message}")

            if "progress" in inspect.signature(merge_run).parameters:
                merge_stats = merge_run(base, set(), progress=_log_merge_progress)
            else:
                _log_merge_progress("目前 merge 工具不支援進度回報，改用舊版合併流程")
                merge_stats = merge_run(base, set())
            manifest_path = Path(merge_stats["out_path"])
            _append_log(
                f"[merge] 完成：{merge_stats['total_rows']} 筆，"
                f"共 {len(merge_stats['batches'])} 批次"
            )

            # Step 2: train
            with CAPIWebHandler._retrain_state["lock"]:
                job["step"] = "train"

            project_root = Path(__file__).resolve().parent
            output_path = Path(params["output_path"])
            output_file = output_path if output_path.is_absolute() else project_root / output_path
            log_dir = output_file.parent / "retrain_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            train_log_file = log_dir / f"{job['job_id']}.log"
            summary_json = log_dir / f"{job['job_id']}.summary.json"
            train_log_file.write_text("", encoding="utf-8")
            try:
                summary_json.unlink()
            except FileNotFoundError:
                pass

            train_argv = [
                _sys.executable, "-u", "-m", "scripts.over_review_poc.train_final_model",
                "--manifest", str(manifest_path),
                "--transform", "clahe",
                "--clahe-clip", str(params.get("clahe_clip", 4.0)),
                "--rank", str(params.get("rank", 16)),
                "--alpha", str(params.get("alpha", params.get("rank", 16))),
                "--n-lora-blocks", "2",
                "--epochs", str(params.get("epochs", 15)),
                "--calib-frac", str(params.get("calib_frac", 0.2)),
                "--output", str(params["output_path"]),
                "--summary-json", str(summary_json),
            ]
            if params.get("dinov2_repo"):
                train_argv += ["--dinov2-repo", str(params["dinov2_repo"])]
            if params.get("dinov2_weights"):
                train_argv += ["--dinov2-weights", str(params["dinov2_weights"])]

            _append_log(
                f"[train] 開始訓練 subprocess：epochs={params.get('epochs', 15)}, "
                f"rank={params.get('rank', 16)}, output={params['output_path']}"
            )
            _append_log(f"[train] log={train_log_file}")

            env = {**_os.environ, "PYTHONUNBUFFERED": "1"}
            with open(train_log_file, "ab") as log_f:
                proc = _subprocess.Popen(
                    train_argv,
                    cwd=str(project_root),
                    env=env,
                    stdout=log_f,
                    stderr=_subprocess.STDOUT,
                )
            _append_log(f"[train] subprocess pid={proc.pid}")

            tail_pos = 0
            last_subprocess_log_at = _time.monotonic()
            next_heartbeat_at = last_subprocess_log_at + 60.0

            def _drain_train_log() -> bool:
                nonlocal tail_pos
                try:
                    with open(train_log_file, "rb") as f:
                        f.seek(tail_pos)
                        data = f.read()
                        tail_pos = f.tell()
                except FileNotFoundError:
                    return False
                if not data:
                    return False
                text = data.decode("utf-8", errors="replace")
                for line in text.splitlines():
                    line = line.rstrip()
                    if line:
                        _append_log(line)
                return True

            while True:
                now = _time.monotonic()
                if _drain_train_log():
                    last_subprocess_log_at = now
                    next_heartbeat_at = now + 60.0
                ret = proc.poll()
                if ret is not None:
                    _drain_train_log()
                    break
                if now >= next_heartbeat_at:
                    quiet_sec = int(now - last_subprocess_log_at)
                    _append_log(
                        f"[train] subprocess pid={proc.pid} still running; "
                        f"no new train log for {quiet_sec}s"
                    )
                    next_heartbeat_at = now + 60.0
                _time.sleep(1.0)

            if proc.returncode != 0:
                raise RuntimeError(
                    f"train subprocess exited with code {proc.returncode}; "
                    f"see {train_log_file}"
                )

            if summary_json.is_file():
                summary = json.loads(summary_json.read_text(encoding="utf-8"))
            elif output_file.is_file():
                summary = {"output_path": str(params["output_path"])}
            else:
                raise RuntimeError(
                    f"train subprocess exited with code 0 but output was not created: {output_file}"
                )

            # Done
            with CAPIWebHandler._retrain_state["lock"]:
                job["step"] = "done"
                job["state"] = "completed"
                job["summary"] = summary

        except Exception:
            err = traceback.format_exc()
            with CAPIWebHandler._retrain_state["lock"]:
                job["state"] = "failed"
                job["error"] = err
            with log_lock:
                log_lines.append(f"[ERROR] {err}")

    # ------------------------------------------------------------------ #
    #  Model registry handlers                                            #
    # ------------------------------------------------------------------ #

    def _handle_models_page(self):
        """GET /models"""
        db = self._capi_server_instance.database
        from capi_model_registry import list_bundles_grouped, get_pending_change_summary_for_bundle
        from capi_train_new import LIGHTINGS, TRAINING_UNITS
        grouped = list_bundles_grouped(db)
        for bundles in grouped.values():
            for b in bundles:
                pc = get_pending_change_summary_for_bundle(db, b)
                b["pending_unit_count"] = len(pc)
        template = self.jinja_env.get_template("models.html")
        html = template.render(
            request_path="/models",
            grouped=grouped,
            lightings=list(LIGHTINGS),
            unit_labels=[f"{l}-{z}" for l, z in TRAINING_UNITS],
        )
        self._send_response(200, html)

    def _handle_models_list(self):
        """GET /api/models?machine_id=X"""
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(self.path).query)
        machine_id = (qs.get("machine_id") or [""])[0].strip() or None
        bundles = self._capi_server_instance.database.list_model_bundles(machine_id=machine_id)
        self._send_json({"bundles": bundles})

    def _handle_models_discover(self):
        """GET /api/models/discover → 掃描模型根目錄但不寫入 DB。"""
        from capi_model_registry import discover_model_bundles
        result = discover_model_bundles(
            self._capi_server_instance.database,
            server_config_path=Path(self._capi_server_instance.server_config_path),
        )
        self._send_json(result)

    def _handle_models_sync(self):
        """POST /api/models/sync → 將指定或全部發現的 bundle 寫入 model_registry。"""
        content_length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json({"error": f"無效 JSON: {exc}"}, status=400)
            return

        bundle_paths = payload.get("bundle_paths")
        if bundle_paths is not None:
            if not isinstance(bundle_paths, list) or not all(
                isinstance(path, str) for path in bundle_paths
            ):
                self._send_json({"error": "bundle_paths 必須是字串陣列"}, status=400)
                return

        from capi_model_registry import sync_discovered_bundles
        try:
            result = sync_discovered_bundles(
                self._capi_server_instance.database,
                server_config_path=Path(self._capi_server_instance.server_config_path),
                bundle_paths=bundle_paths,
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        result["message"] = f"已同步 {len(result['imported'])} 個模型"
        self._send_json(result)

    def _handle_models_detail(self):
        """GET /api/models/<id>/detail"""
        parts = self.path.split("/")
        bundle_id = int(parts[3])
        from capi_model_registry import get_bundle_detail
        detail = get_bundle_detail(self._capi_server_instance.database, bundle_id)
        if not detail:
            self._send_json({"error": "not found"}, status=404)
            return
        detail["pending_changes"] = _serialize_pending_changes(detail.get("pending_changes"))
        self._send_json(detail)

    @staticmethod
    def _model_validation_baseline(db, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        active = db.get_active_model_bundle()
        if (
            not active
            or int(active["id"]) == int(candidate["id"])
            or str(active.get("machine_id") or "") != str(candidate.get("machine_id") or "")
        ):
            return None
        from capi_model_registry import get_bundle_detail
        return get_bundle_detail(db, int(active["id"]))

    @staticmethod
    def _model_validation_run_payload(run: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not run:
            return None
        from capi_model_validation import (
            build_model_validation_summary,
            classify_model_validation_result,
        )

        payload = dict(run)
        results = [dict(row) for row in payload.get("results") or []]
        has_baseline = payload.get("baseline_bundle_id") is not None
        if results and not payload.get("summary"):
            payload["summary"] = build_model_validation_summary(
                results,
                has_baseline=has_baseline,
            )
        for result in results:
            result["file_url"] = (
                f"/api/ric/ng-validation/file?id={int(result['sample_id'])}"
            )
            result["comparison"] = classify_model_validation_result(
                result,
                has_baseline=has_baseline,
            )
        payload["results"] = results
        return payload

    def _handle_models_validation(self):
        """GET /api/models/<id>/validation[?run_id=N]."""
        parsed = urllib.parse.urlparse(self.path)
        try:
            bundle_id = int(parsed.path.split("/")[3])
        except (IndexError, ValueError):
            self._send_json({"error": "invalid bundle id"}, status=400)
            return

        db = self._capi_server_instance.database
        from capi_model_registry import get_bundle_detail
        from capi_model_validation import bundle_validation_snapshot

        candidate = get_bundle_detail(db, bundle_id)
        if not candidate:
            self._send_json({"error": "bundle not found"}, status=404)
            return
        baseline = self._model_validation_baseline(db, candidate)
        machine_id = str(candidate.get("machine_id") or "").strip()
        if machine_id:
            _, sample_count = db.list_ng_validation_samples(
                model_id=machine_id,
                limit=1,
            )
        else:
            sample_count = 0
        runs = db.list_model_validation_runs(bundle_id, limit=20)

        query = urllib.parse.parse_qs(parsed.query)
        raw_run_id = (query.get("run_id") or [""])[0]
        selected_run = None
        if raw_run_id:
            try:
                selected_run = db.get_model_validation_run(int(raw_run_id))
            except ValueError:
                self._send_json({"error": "run_id 格式錯誤"}, status=400)
                return
            if (
                not selected_run
                or int(selected_run["candidate_bundle_id"]) != bundle_id
            ):
                self._send_json({"error": "validation run not found"}, status=404)
                return
        elif runs:
            selected_run = db.get_model_validation_run(int(runs[0]["id"]))

        if selected_run and selected_run.get("state") in {"pending", "running"}:
            with CAPIWebHandler._scan_state["lock"]:
                active_job = CAPIWebHandler._scan_state.get("job")
                matching_job = bool(
                    active_job
                    and active_job.get("kind") == "model_validation"
                    and int(active_job.get("run_id") or 0) == int(selected_run["id"])
                )
            if not matching_job:
                db.finish_model_validation_run(
                    int(selected_run["id"]),
                    state="failed",
                    summary=selected_run.get("summary") or {},
                    error_message="考試因 server 重啟或工作中斷而停止",
                )
                selected_run = db.get_model_validation_run(int(selected_run["id"]))
                runs = db.list_model_validation_runs(bundle_id, limit=20)

        self._send_json({
            "candidate": bundle_validation_snapshot(candidate),
            "baseline": bundle_validation_snapshot(baseline) if baseline else None,
            "sample_count": sample_count,
            "runs": runs,
            "run": self._model_validation_run_payload(selected_run),
        })

    def _handle_models_validation_start(self):
        """POST /api/models/<id>/validation/start."""
        try:
            bundle_id = int(urllib.parse.urlparse(self.path).path.split("/")[3])
        except (IndexError, ValueError):
            self._send_json({"error": "invalid bundle id"}, status=400)
            return

        db = self._capi_server_instance.database
        from capi_model_registry import get_bundle_detail

        candidate = get_bundle_detail(db, bundle_id)
        if not candidate:
            self._send_json({"error": "bundle not found"}, status=404)
            return
        machine_id = str(candidate.get("machine_id") or "").strip()
        if not machine_id:
            self._send_json({"error": "bundle 未設定機種 ID"}, status=400)
            return

        samples = []
        offset = 0
        total = 0
        while True:
            page, total = db.list_ng_validation_samples(
                model_id=machine_id,
                limit=1000,
                offset=offset,
            )
            samples.extend(page)
            offset += len(page)
            if not page or offset >= total:
                break
        if not samples:
            self._send_json({
                "error": f"機種 {candidate.get('machine_id') or '-'} 尚無 NG 驗證樣本",
            }, status=400)
            return

        baseline = self._model_validation_baseline(db, candidate)
        started, response = CAPIWebHandler._start_model_validation_job(
            candidate=candidate,
            baseline=baseline,
            samples=samples,
            validation_base_dir=self._ng_validation_base_dir(),
            server_inst=self._capi_server_instance,
        )
        self._send_json(response, status=202 if started else 409)

    def _handle_models_activate(self):
        """POST /api/models/<id>/activate"""
        parts = self.path.split("/")
        bundle_id = int(parts[3])
        from capi_model_registry import activate_bundle
        try:
            result = activate_bundle(
                self._capi_server_instance.database,
                bundle_id,
                server_config_path=Path(self._capi_server_instance.server_config_path),
            )
            self._send_json(result)
        except ValueError as e:
            self._send_json({"error": str(e)}, status=400)

    def _handle_models_deactivate(self):
        """POST /api/models/<id>/deactivate"""
        parts = self.path.split("/")
        bundle_id = int(parts[3])
        from capi_model_registry import deactivate_bundle
        try:
            result = deactivate_bundle(
                self._capi_server_instance.database, bundle_id,
                server_config_path=Path(self._capi_server_instance.server_config_path),
            )
            self._send_json(result)
        except ValueError as e:
            self._send_json({"error": str(e)}, status=400)

    def _handle_models_update_notes(self):
        """POST /api/models/<id>/notes  body: {notes}"""
        bundle_id = int(self.path.split("/")[3])
        content_length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8") or "{}")
        except json.JSONDecodeError as e:
            self._send_json({"error": f"無效 JSON: {e}"}, status=400)
            return
        notes = payload.get("notes", "")
        if not isinstance(notes, str):
            self._send_json({"error": "notes 必須是文字"}, status=400)
            return
        from capi_model_registry import update_bundle_notes
        try:
            self._send_json(update_bundle_notes(
                self._capi_server_instance.database, bundle_id, notes,
            ))
        except ValueError as e:
            self._send_json({"error": str(e)}, status=404)

    def _handle_models_update_threshold(self):
        """POST /api/models/<id>/threshold  body: {lighting, zone, value}"""
        bundle_id = int(self.path.split("/")[3])
        content_length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8") or "{}")
        except json.JSONDecodeError as e:
            self._send_json({"error": f"無效 JSON: {e}"}, status=400)
            return
        lighting = str(payload.get("lighting", ""))
        zone = str(payload.get("zone", ""))
        try:
            value = float(payload.get("value"))
        except (TypeError, ValueError):
            self._send_json({"error": "value 必須是數字"}, status=400)
            return
        from capi_model_registry import update_threshold
        try:
            result = update_threshold(
                self._capi_server_instance.database,
                bundle_id, lighting=lighting, zone=zone, value=value,
            )
            # active bundle 才能 in-place reload；未 active 只寫檔
            applied = self._capi_server_instance.apply_threshold_inplace(
                machine_id=result["machine_id"], lighting=lighting, zone=zone, value=value,
            )
            result["hot_reloaded"] = applied
            result["message"] = (
                "已即時生效（無需重啟）" if applied
                else "已寫入 yaml；此 bundle 未 active，下次啟用後生效"
            )
            self._send_json(result)
        except ValueError as e:
            self._send_json({"error": str(e)}, status=400)

    def _handle_models_delete(self):
        """POST /api/models/<id>/delete"""
        parts = self.path.split("/")
        bundle_id = int(parts[3])
        from capi_model_registry import delete_bundle
        try:
            result = delete_bundle(
                self._capi_server_instance.database, bundle_id,
                server_config_path=Path(self._capi_server_instance.server_config_path),
            )
            self._send_json(result)
        except ValueError as e:
            self._send_json({"error": str(e)}, status=409)

    def _handle_models_export(self):
        """GET /api/models/<id>/export → 串流 ZIP"""
        parts = self.path.split("/")
        bundle_id = int(parts[3])
        from capi_model_registry import export_bundle_zip
        db = self._capi_server_instance.database
        bundle = db.get_model_bundle(bundle_id)
        if not bundle:
            self._send_response(404, "")
            return

        zip_bytes = export_bundle_zip(Path(bundle["bundle_path"]), bundle["machine_id"])
        filename = f"{Path(bundle['bundle_path']).name}.zip"
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(zip_bytes)))
        self.end_headers()
        self.wfile.write(zip_bytes)

    def _handle_models_training_tiles(self):
        """GET /api/models/<id>/training_tiles?source=ok|ng&lighting=X&zone=Y&limit=N&offset=M[&score_from_bundle=N&sort_by=score_desc]

        回傳該 bundle 對應 job_id 的 tile pool 縮圖列表。zone 預設不過濾（NG 沒 zone）。
        """
        from urllib.parse import parse_qs, urlparse
        parts = self.path.split("/")
        try:
            bundle_id = int(parts[3])
        except (ValueError, IndexError):
            self._send_json({"error": "invalid bundle id"}, status=400)
            return

        qs = parse_qs(urlparse(self.path).query)
        source = (qs.get("source") or ["ok"])[0]
        if source not in ("ok", "ng"):
            self._send_json({"error": "source must be ok or ng"}, status=400)
            return
        lighting = (qs.get("lighting") or [""])[0] or None
        zone = (qs.get("zone") or [""])[0] or None
        score_from = qs.get("score_from_bundle", [None])[0]
        sort_by = qs.get("sort_by", ["default"])[0]
        try:
            limit = max(1, min(int((qs.get("limit") or ["200"])[0]), 1000))
            offset = max(0, int((qs.get("offset") or ["0"])[0]))
        except ValueError:
            self._send_json({"error": "limit/offset must be int"}, status=400)
            return

        db = self._capi_server_instance.database
        bundle = db.get_model_bundle(bundle_id)
        if not bundle:
            self._send_json({"error": "bundle not found"}, status=404)
            return
        job_id = bundle.get("job_id") or ""
        if not job_id:
            self._send_json({"tiles": [], "total": 0,
                             "message": "此 bundle 沒有關聯 job_id"})
            return

        kwargs = {"source": source}
        if lighting:
            kwargs["lighting"] = lighting
        # NG 樣本 zone 為 NULL，不傳 zone 過濾；OK 才依 zone 篩
        if source == "ok" and zone:
            kwargs["zone"] = zone

        all_tiles = db.list_tile_pool(job_id, **kwargs)
        # 加 score 並排序（在分頁前，確保排序後再取 page）
        if score_from:
            CAPIWebHandler._decorate_tiles_with_scores(all_tiles, db, score_from, sort_by)
        total = len(all_tiles)
        page = all_tiles[offset:offset + limit]
        out = []
        for t in page:
            out.append({
                "id": t.get("id"),
                "lighting": t.get("lighting"),
                "zone": t.get("zone"),
                "source": t.get("source"),
                "decision": t.get("decision"),
                "score": t.get("score"),
                "score_quartile": t.get("score_quartile"),
                "thumb_url": self._train_new_thumb_url(t.get("thumb_path")),
                "image_url": self._train_new_thumb_url(t.get("source_path")),
            })
        self._send_json({"tiles": out, "total": total, "limit": limit, "offset": offset})

    def _handle_models_training_data_delete(self):
        """POST /api/models/<id>/training_data/delete

        清空此 bundle 對應 job 的訓練資料（DB tile_pool + thumb dir）。
        bundle 本身與 inference 不受影響。
        """
        parts = self.path.split("/")
        try:
            bundle_id = int(parts[3])
        except (ValueError, IndexError):
            self._send_json({"error": "invalid bundle id"}, status=400)
            return
        from capi_model_registry import delete_training_data
        try:
            result = delete_training_data(self._capi_server_instance.database, bundle_id)
            self._send_json(result)
        except ValueError as e:
            self._send_json({"error": str(e)}, status=404)
        except Exception as e:
            logger.error(f"delete_training_data error: {e}", exc_info=True)
            self._send_json({"error": str(e)}, status=500)

    def _handle_models_tiles_decision(self):
        """POST /api/models/<id>/tiles/decision
        body: {"tile_ids": [int, ...], "decision": "accept"|"reject"}

        只允許切換 OK tile 的 decision；NG tile 嘗試操作 → 400。
        """
        parts = self.path.split("/")
        try:
            bundle_id = int(parts[3])
        except (ValueError, IndexError):
            self._send_json({"error": "invalid bundle id"}, status=400)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            self._send_json({"error": "invalid JSON"}, status=400)
            return

        tile_ids = body.get("tile_ids", [])
        decision = body.get("decision")
        if not tile_ids or decision not in ("accept", "reject"):
            self._send_json({"error": "tile_ids 與 decision 必填"}, status=400)
            return
        try:
            tile_ids = [int(x) for x in tile_ids]
        except (TypeError, ValueError):
            self._send_json({"error": "tile_ids 必須為整數陣列"}, status=400)
            return

        db = self._capi_server_instance.database
        bundle = db.get_model_bundle(bundle_id)
        if not bundle:
            self._send_json({"error": "bundle not found"}, status=404)
            return
        job_id = bundle.get("job_id") or ""
        if not job_id:
            self._send_json({"error": "此 bundle 無關聯 job_id"}, status=400)
            return

        # 只允許動 OK tile：用 source='ok' 撈一次當前 job 的所有 OK tile id 做白名單
        ok_ids = {int(t["id"]) for t in db.list_tile_pool(job_id, source="ok")}
        bad = [i for i in tile_ids if i not in ok_ids]
        if bad:
            self._send_json({"error": f"tile_ids 含非 OK tile（NG tile 不可動）: {bad[:5]}"},
                            status=400)
            return

        db.update_tile_decisions(job_id, tile_ids, decision)
        self._send_json({"ok": True, "updated": len(tile_ids)})

    def _handle_models_retrain_submodel_with_panels(self):
        """POST /api/models/<id>/retrain_submodel_with_panels
        body: {
            "lighting": str,
            "zone": "inner"|"edge",
            "panel_paths": [str, ...]  # 至少 1 片；所有 panel 都完整切 tile
            "training_params": {...}   # optional
        }

        重新選圖後只重訓單一子模型，不跑完整 10-unit pipeline。
        """
        from capi_train_new import (
            LIGHTINGS, ZONES,
            derive_panel_modes, generate_job_id,
        )

        parts = self.path.split("/")
        try:
            bundle_id = int(parts[3])
        except (ValueError, IndexError):
            self._send_json({"error": "invalid bundle id"}, status=400)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            self._send_json({"error": "invalid JSON"}, status=400)
            return

        lighting = body.get("lighting")
        zone = body.get("zone")
        if lighting not in LIGHTINGS:
            self._send_json({"error": f"lighting 必須為 {LIGHTINGS}"}, status=400)
            return
        if zone not in ZONES:
            self._send_json({"error": f"zone 必須為 {ZONES}"}, status=400)
            return

        panel_paths = body.get("panel_paths", [])
        if not isinstance(panel_paths, list):
            self._send_json({"error": "panel_paths must be a list"}, status=400)
            return
        clean_panel_paths = []
        for p in panel_paths:
            if not isinstance(p, str) or not p.strip() or p.strip() in ("undefined", "null"):
                self._send_json({"error": "panel_paths contains invalid path"}, status=400)
                return
            clean_panel_paths.append(p.strip())
        if not clean_panel_paths:
            self._send_json({"error": "panel_paths must contain at least 1 panel"}, status=400)
            return

        training_params, err = self._validate_training_params(body.get("training_params"))
        if err:
            self._send_json({"error": err}, status=400)
            return
        locked_params = sorted(
            set(training_params or {})
            & self.PATCHCORE_BUNDLE_LOCKED_TRAINING_PARAMS
        )
        if locked_params:
            self._send_json({
                "error": (
                    "single-unit retraining must inherit bundle-level PatchCore params: "
                    + ", ".join(locked_params)
                )
            }, status=400)
            return

        db = self._capi_server_instance.database
        bundle = db.get_model_bundle(bundle_id)
        if not bundle:
            self._send_json({"error": "bundle not found"}, status=404)
            return

        from pathlib import Path
        from capi_model_registry import _read_manifest
        bundle_dir = Path(bundle["bundle_path"])
        try:
            old_manifest = _read_manifest(bundle_dir)
        except Exception:
            old_manifest = {}
        image_preprocess_pipeline = old_manifest.get("image_preprocess_pipeline") or []
        image_preprocess_pipelines = old_manifest.get("image_preprocess_pipelines") or {}
        preprocess_after_tiling = bool(old_manifest.get("preprocess_after_tiling", False))
        tile_stride = int(old_manifest.get("tile_stride") or 512)

        machine_id = str(bundle.get("machine_id") or "").strip()
        if not machine_id:
            self._send_json({"error": "bundle missing machine_id"}, status=400)
            return

        job_id = generate_job_id(machine_id).replace("train_", "subtrain_", 1)
        panel_modes = derive_panel_modes(len(clean_panel_paths))

        state = CAPIWebHandler._submodel_retrain_state
        with state["lock"]:
            current = state.get("job")
            if current and current.get("state") == "running":
                self._send_json({
                    "error": "已有重訓 job 進行中，請等待完成",
                    "job": current,
                }, status=409)
                return

            state["job"] = {
                "bundle_id": bundle_id,
                "job_id": job_id,
                "machine_id": machine_id,
                "lighting": lighting,
                "zone": zone,
                "state": "running",
                "step": "preprocess",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "log_lines": [],
                "summary": None,
                "error": None,
                "source": "panels",
                "panel_count": len(clean_panel_paths),
            }

        try:
            db.create_training_job(
                job_id=job_id,
                machine_id=machine_id,
                panel_paths=clean_panel_paths,
                training_params=training_params,
                panel_modes=panel_modes,
                image_preprocess_pipeline=image_preprocess_pipeline,
                image_preprocess_pipelines=image_preprocess_pipelines,
                preprocess_after_tiling=preprocess_after_tiling,
                tile_stride=tile_stride,
            )
        except Exception:
            with state["lock"]:
                state["job"] = None
            raise

        thread = threading.Thread(
            target=self._submodel_retrain_with_panels_worker,
            args=(bundle_id, lighting, zone, job_id, clean_panel_paths, training_params, panel_modes),
            daemon=True,
            name=f"submodel-retrain-panels-{bundle_id}-{lighting}-{zone}",
        )
        thread.start()

        self._send_json({
            "ok": True,
            "bundle_id": bundle_id,
            "job_id": job_id,
            "lighting": lighting,
            "zone": zone,
        })

    def _submodel_retrain_with_panels_worker(
        self,
        bundle_id: int,
        lighting: str,
        zone: str,
        job_id: str,
        panel_paths: list,
        training_params=None,
        panel_modes=None,
    ):
        """背景 thread：重新選 panel 後，只訓練單一子模型並覆蓋原 .pt。"""
        import traceback
        from capi_preprocess import PreprocessConfig
        from capi_train_new import (
            TrainingConfig, apply_user_training_params,
            preprocess_panels_to_pool, sample_ng_tiles, train_single_submodel,
            NG_TILES_PER_LIGHTING,
        )
        from capi_model_registry import append_submodel_history, _read_manifest, _write_manifest

        state = CAPIWebHandler._submodel_retrain_state
        server_inst = self._capi_server_instance
        db = server_inst.database
        slot_acquired = False

        def _set_step(step: str):
            with state["lock"]:
                if state["job"] is not None:
                    state["job"]["step"] = step

        def _log(msg: str):
            ts = datetime.now().strftime("%H:%M:%S")
            with state["lock"]:
                if state["job"] is not None:
                    state["job"]["log_lines"].append(f"[{ts}] {msg}")
                    if len(state["job"]["log_lines"]) > 500:
                        state["job"]["log_lines"] = state["job"]["log_lines"][-500:]

        try:
            bundle = db.get_model_bundle(bundle_id)
            if not bundle:
                raise RuntimeError(f"bundle {bundle_id} 已不存在")

            job = db.get_training_job(job_id)
            if not job:
                raise RuntimeError(f"job {job_id} 已不存在")

            bundle_dir = Path(bundle["bundle_path"])
            machine_id = bundle["machine_id"]
            unit_label = f"{lighting}-{zone}"
            output_pt = bundle_dir / f"{unit_label}.pt"

            _log(f"開始單一重訓 {unit_label} (bundle_id={bundle_id}, job_id={job_id})")
            _log(f"重新選圖 panel 數: {len(panel_paths)}")

            old_manifest = _read_manifest(bundle_dir)
            old_unit_metrics = (old_manifest.get("unit_metrics") or {}).get(unit_label) or {}
            old_history = (old_manifest.get("submodel_history") or {}).get(unit_label) or []
            if old_history:
                old_auroc = old_history[-1].get("auroc")
                old_tile_count = old_history[-1].get("tile_count_used")
            else:
                old_auroc = old_unit_metrics.get("auroc")
                old_tile_count = old_unit_metrics.get("train_count")

            train_cfg = CAPIWebHandler._load_train_new_config(server_inst)
            patchcore_params = old_manifest.get("patchcore_params") or {}
            cfg = TrainingConfig(
                machine_id=machine_id,
                panel_paths=[Path(p) for p in panel_paths],
                over_review_root=train_cfg["over_review_root"],
                output_root=train_cfg["output_root"],
                backbone_cache_dir=train_cfg["backbone_cache_dir"],
                required_backbones=train_cfg["required_backbones"],
                image_preprocess_pipeline=job.get("image_preprocess_pipeline") or [],
                image_preprocess_pipelines=job.get("image_preprocess_pipelines") or {},
                preprocess_after_tiling=bool(job.get("preprocess_after_tiling", False)),
                tile_stride=int(job.get("tile_stride") or 512),
                batch_size=patchcore_params.get("batch_size", 8),
                image_size=tuple(patchcore_params.get("image_size", (512, 512))),
                coreset_ratio=patchcore_params.get("coreset_ratio", 0.1),
                max_epochs=patchcore_params.get("max_epochs", 1),
                precision=patchcore_params.get("precision", "float32"),
                feature_layers=patchcore_params.get("feature_layers", "layer2_layer3"),
                feature_pool_kernel_size=patchcore_params.get("feature_pool_kernel_size", 3),
                feature_cleaning_mode=patchcore_params.get("feature_cleaning_mode", "off"),
                feature_cleaning_scope=patchcore_params.get(
                    "feature_cleaning_scope", "inner_only",
                ),
                feature_cleaning_keep_ratio=patchcore_params.get(
                    "feature_cleaning_keep_ratio", 0.99,
                ),
                feature_cleaning_center_size=patchcore_params.get(
                    "feature_cleaning_center_size", 512,
                ),
                feature_cleaning_by_zone=patchcore_params.get(
                    "feature_cleaning_by_zone"
                ) or {},
            )
            inherited_training_params = dict(training_params or {})
            for key in CAPIWebHandler.PATCHCORE_BUNDLE_LOCKED_TRAINING_PARAMS:
                inherited_training_params.pop(key, None)
            apply_user_training_params(cfg, inherited_training_params, log_fn=_log)

            _set_step("preprocess")
            thumb_root = Path(".tmp/train_new_thumbs") / job_id
            _log(f"前處理 selected panels（只保留 lighting={lighting}）")
            preprocess_cfg = PreprocessConfig(
                tile_stride=cfg.tile_stride,
                image_preprocess_pipeline=cfg.image_preprocess_pipeline,
                image_preprocess_pipelines=cfg.image_preprocess_pipelines,
                preprocess_after_tiling=cfg.preprocess_after_tiling,
                product_resolution=CAPIWebHandler._product_resolution_for_machine(machine_id, server_inst),
            )
            stats = preprocess_panels_to_pool(
                job_id=job_id,
                cfg=cfg,
                preprocess_cfg=preprocess_cfg,
                db=db,
                thumb_dir=thumb_root,
                log=_log,
                panel_modes=panel_modes,
                target_lightings=(lighting,),
            )
            if stats["panel_success_full"] <= 0:
                raise RuntimeError("沒有任何 full panel 前處理成功")
            _log(f"OK tile 寫入完成：{stats['total_tiles']} tiles")

            _set_step("ng")
            _log(
                f"準備 NG 驗證 crop（優先重用 NG 驗證庫；缺少才從推論紀錄裁切，"
                f"lighting={lighting}，上限 {NG_TILES_PER_LIGHTING} 個，排除 B0F 黑畫面）"
            )
            CAPIWebHandler._sample_ng_tiles_compat(
                sample_ng_tiles,
                job_id=job_id,
                over_review_root=cfg.over_review_root,
                db=db,
                thumb_dir=thumb_root,
                per_lighting=NG_TILES_PER_LIGHTING,
                log=_log,
                lightings=(lighting,),
                preprocess_cfg=preprocess_cfg,
                machine_id=machine_id,
                rotate_180=CAPIWebHandler._training_bomb_rotate_180(machine_id, server_inst),
                ng_validation_base_dir=CAPIWebHandler._ng_validation_base_dir_for_server(
                    server_inst
                ),
            )

            db.update_training_job_state(job_id, "train")

            _set_step("train")
            slot = CAPIWebHandler._train_slot
            with slot["lock"]:
                if slot.get("active_job_id") is not None:
                    raise RuntimeError(f"另一個訓練 job 正在執行: {slot['active_job_id']}")
                slot["active_job_id"] = job_id
                slot_acquired = True

            CAPIWebHandler._cancel_and_wait_scan_idle(timeout_s=15.0)
            CAPIWebHandler._free_server_gpu_cache()
            _log("訓練中（只訓練選定子模型）...")
            result = train_single_submodel(
                db=db,
                job_id=job_id,
                lighting=lighting,
                zone=zone,
                cfg=cfg,
                output_pt_path=output_pt,
                gpu_lock=server_inst._gpu_lock,
                log=_log,
            )

            _set_step("metrics")
            new_auroc = result["metrics"].get("auroc")
            new_tile_count = result["tile_count"]
            _log(f"訓練完成：tile={new_tile_count}, AUROC={new_auroc}")

            _set_step("swap")
            _log(f"已替換 {output_pt}")

            entry = {
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "job_id": job_id,
                "trained_with_job_id": job_id,
                "tile_count_used": new_tile_count,
                "auroc": new_auroc,
                "used_tile_ids": result["used_tile_ids"],
                "kind": "retrain_from_panels",
                "ng_used": result["ng_used"],
                "panel_count": len(panel_paths),
                "panel_glass_ids": [Path(p).name for p in panel_paths],
                "feature_pool_kernel_size": result["metrics"].get(
                    "feature_pool_kernel_size", cfg.feature_pool_kernel_size,
                ),
                "feature_cleaning_mode": cfg.feature_cleaning_mode,
                "feature_cleaning": result["metrics"].get("feature_cleaning") or {},
            }
            append_submodel_history(bundle_dir, lighting, zone, entry)
            refreshed_manifest = _read_manifest(bundle_dir)
            refreshed_metrics = dict(result["metrics"])
            refreshed_metrics["used_tile_ids"] = result["used_tile_ids"]
            refreshed_manifest.setdefault("unit_metrics", {})[unit_label] = refreshed_metrics
            refreshed_manifest.setdefault("tiles_per_unit", {})[unit_label] = {
                "train": result["tile_count"],
                "ng": result["ng_count"],
            }
            refreshed_manifest.setdefault("model_files", {})[unit_label] = {
                "path": output_pt.name,
                "size_bytes": result["size_bytes"],
            }
            _write_manifest(bundle_dir, refreshed_manifest)
            _log("manifest history 已更新")

            from capi_model_registry import invalidate_score_cache
            cleared = invalidate_score_cache(
                db, scoring_bundle_id=bundle_id, lighting=lighting, zone=zone,
            )
            _log(f"清除 {cleared} 筆 score cache（lighting={lighting}, zone={zone}）")

            _set_step("reload")
            inferencer = server_inst.inferencers.get(machine_id)
            if inferencer is None:
                _log(f"[v2] 機台 {machine_id} 無 inferencer cache，跳過 reload")
            else:
                try:
                    inferencer.reload_submodel(machine_id, lighting, zone)
                    _log(f"[v2] 已通知 inferencer reload {machine_id}/{lighting}/{zone}")
                except Exception as reload_err:
                    _log(f"[v2] reload 失敗（不影響重訓結果）：{reload_err}")
                    logger.warning("reload_submodel raised: %s", reload_err, exc_info=True)

            db.update_training_job_state(job_id, "completed", output_bundle=str(bundle_dir))
            with state["lock"]:
                state["job"]["step"] = "done"
                state["job"]["state"] = "completed"
                state["job"]["summary"] = {
                    "auroc_old": old_auroc,
                    "auroc_new": new_auroc,
                    "tile_count_old": old_tile_count,
                    "tile_count_new": new_tile_count,
                    "job_id": job_id,
                    "panel_count": len(panel_paths),
                }
            _log("✓ 單一子模型重訓完成")

        except Exception as e:
            tb = traceback.format_exc()
            _log(f"✗ 失敗: {e}")
            for line in tb.rstrip().splitlines()[-8:]:
                _log(f"  {line}")
            try:
                db.update_training_job_state(job_id, "failed", error_message=str(e))
            except Exception:
                pass
            with state["lock"]:
                if state["job"] is not None:
                    state["job"]["state"] = "failed"
                    state["job"]["error"] = str(e)
            logger.error("submodel retrain with panels worker failed: %s", e, exc_info=True)
        finally:
            if slot_acquired:
                slot = CAPIWebHandler._train_slot
                with slot["lock"]:
                    if slot.get("active_job_id") == job_id:
                        slot["active_job_id"] = None
            try:
                finished_job = db.get_training_job(job_id)
                if finished_job and finished_job.get("state") == "failed":
                    CAPIWebHandler._cleanup_train_new_job_artifacts(
                        db, job_id, reason="panel retrain failed"
                    )
            except Exception:
                logger.warning(
                    "cannot finalize panel retrain cleanup: job_id=%s",
                    job_id,
                    exc_info=True,
                )

    def _handle_models_retrain_submodel(self):
        """POST /api/models/<id>/retrain_submodel
        body: {"lighting": str, "zone": "inner"|"edge"}

        啟動 worker thread 重訓單一子模型。已有 retrain job 跑 → 409。
        """
        from capi_train_new import LIGHTINGS, ZONES

        parts = self.path.split("/")
        try:
            bundle_id = int(parts[3])
        except (ValueError, IndexError):
            self._send_json({"error": "invalid bundle id"}, status=400)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            self._send_json({"error": "invalid JSON"}, status=400)
            return

        lighting = body.get("lighting")
        zone = body.get("zone")
        if lighting not in LIGHTINGS:
            self._send_json({"error": f"lighting 必須為 {LIGHTINGS}"}, status=400)
            return
        if zone not in ZONES:
            self._send_json({"error": f"zone 必須為 {ZONES}"}, status=400)
            return

        db = self._capi_server_instance.database
        bundle = db.get_model_bundle(bundle_id)
        if not bundle:
            self._send_json({"error": "bundle not found"}, status=404)
            return
        if not bundle.get("job_id"):
            self._send_json({"error": "此 bundle 無關聯 job_id（訓練資料已刪），無法重訓"},
                            status=400)
            return

        state = CAPIWebHandler._submodel_retrain_state
        with state["lock"]:
            current = state.get("job")
            if current and current.get("state") == "running":
                self._send_json({"error": "已有重訓 job 進行中，請等待完成",
                                 "job": current}, status=409)
                return

            state["job"] = {
                "bundle_id": bundle_id,
                "lighting": lighting,
                "zone": zone,
                "state": "running",
                "step": "stage",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "log_lines": [],
                "summary": None,
                "error": None,
            }

        thread = threading.Thread(
            target=self._submodel_retrain_worker,
            args=(bundle_id, lighting, zone),
            daemon=True,
            name=f"submodel-retrain-{bundle_id}-{lighting}-{zone}",
        )
        thread.start()

        self._send_json({"ok": True, "bundle_id": bundle_id,
                         "lighting": lighting, "zone": zone})

    def _submodel_retrain_worker(self, bundle_id: int, lighting: str, zone: str):
        """背景 thread：執行單子模型重訓全流程。

        步驟：stage → train → metrics → swap → reload → done。任一步失敗
        都更新 state["job"] state="failed" + error，並保留 .pt 與 manifest 不動。
        """
        import traceback
        from capi_train_new import train_single_submodel, TrainingConfig
        from capi_model_registry import append_submodel_history, _read_manifest, _write_manifest

        state = CAPIWebHandler._submodel_retrain_state

        def _set_step(step: str):
            with state["lock"]:
                if state["job"] is not None:
                    state["job"]["step"] = step

        def _log(msg: str):
            ts = datetime.now().strftime("%H:%M:%S")
            with state["lock"]:
                if state["job"] is not None:
                    state["job"]["log_lines"].append(f"[{ts}] {msg}")
                    if len(state["job"]["log_lines"]) > 500:
                        state["job"]["log_lines"] = state["job"]["log_lines"][-500:]

        try:
            db = self._capi_server_instance.database
            bundle = db.get_model_bundle(bundle_id)
            if not bundle:
                raise RuntimeError(f"bundle {bundle_id} 已不存在")
            job_id = bundle["job_id"]
            bundle_dir = Path(bundle["bundle_path"])
            machine_id = bundle["machine_id"]
            unit_label = f"{lighting}-{zone}"
            output_pt = bundle_dir / f"{unit_label}.pt"

            _log(f"開始重訓 {unit_label} (bundle_id={bundle_id})")

            # 取舊 AUROC / tile 數做 summary 比對
            old_manifest = _read_manifest(bundle_dir)
            old_unit_metrics = (old_manifest.get("unit_metrics") or {}).get(unit_label) or {}
            old_history = (old_manifest.get("submodel_history") or {}).get(unit_label) or []
            if old_history:
                old_auroc = old_history[-1].get("auroc")
                old_tile_count = old_history[-1].get("tile_count_used")
            else:
                old_auroc = old_unit_metrics.get("auroc")
                old_tile_count = old_unit_metrics.get("train_count")

            # 取 TrainingConfig：用既有 bundle 訓練時的 patchcore_params
            patchcore_params = (old_manifest.get("patchcore_params") or {})
            cfg = TrainingConfig(
                machine_id=machine_id,
                panel_paths=[],
                over_review_root=Path(".tmp/_unused"),
                image_preprocess_pipeline=old_manifest.get("image_preprocess_pipeline") or [],
                batch_size=patchcore_params.get("batch_size", 32),
                image_size=tuple(patchcore_params.get("image_size", (512, 512))),
                coreset_ratio=patchcore_params.get("coreset_ratio", 0.1),
                max_epochs=patchcore_params.get("max_epochs", 1),
                precision=patchcore_params.get("precision", "float32"),
                feature_layers=patchcore_params.get("feature_layers", "layer2_layer3"),
                feature_pool_kernel_size=patchcore_params.get("feature_pool_kernel_size", 3),
                feature_cleaning_mode=patchcore_params.get("feature_cleaning_mode", "off"),
                feature_cleaning_scope=patchcore_params.get(
                    "feature_cleaning_scope", "inner_only",
                ),
                feature_cleaning_keep_ratio=patchcore_params.get(
                    "feature_cleaning_keep_ratio", 0.99,
                ),
                feature_cleaning_center_size=patchcore_params.get(
                    "feature_cleaning_center_size", 512,
                ),
                feature_cleaning_by_zone=patchcore_params.get(
                    "feature_cleaning_by_zone"
                ) or {},
            )
            # backbone_cache_dir / required_backbones / output_root 沿用 dataclass 預設值

            _set_step("stage")
            _log("準備訓練資料...")

            _set_step("train")
            _log("訓練中（含 stage_dataset → train_one_patchcore → calibrate）...")
            gpu_lock = self._capi_server_instance._gpu_lock
            result = train_single_submodel(
                db=db, job_id=job_id, lighting=lighting, zone=zone,
                cfg=cfg, output_pt_path=output_pt,
                gpu_lock=gpu_lock, log=_log,
            )

            _set_step("metrics")
            new_auroc = result["metrics"].get("auroc")
            new_tile_count = result["tile_count"]
            _log(f"訓練完成：tile={new_tile_count}, AUROC={new_auroc}")

            _set_step("swap")
            # train_single_submodel 已 atomic 寫好 output_pt，不需另做 swap
            _log(f"已替換 {output_pt}")

            # 寫 manifest history
            entry = {
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "tile_count_used": new_tile_count,
                "auroc": new_auroc,
                "used_tile_ids": result["used_tile_ids"],
                "kind": "retrain",
                "ng_used": result["ng_used"],
                "feature_pool_kernel_size": result["metrics"].get(
                    "feature_pool_kernel_size", cfg.feature_pool_kernel_size,
                ),
                "feature_cleaning_mode": cfg.feature_cleaning_mode,
                "feature_cleaning": result["metrics"].get("feature_cleaning") or {},
            }
            append_submodel_history(bundle_dir, lighting, zone, entry)
            refreshed_manifest = _read_manifest(bundle_dir)
            refreshed_metrics = dict(result["metrics"])
            refreshed_metrics["used_tile_ids"] = result["used_tile_ids"]
            refreshed_manifest.setdefault("unit_metrics", {})[unit_label] = refreshed_metrics
            refreshed_manifest.setdefault("tiles_per_unit", {})[unit_label] = {
                "train": result["tile_count"],
                "ng": result["ng_count"],
            }
            refreshed_manifest.setdefault("model_files", {})[unit_label] = {
                "path": output_pt.name,
                "size_bytes": result["size_bytes"],
            }
            _write_manifest(bundle_dir, refreshed_manifest)
            _log("manifest history 已更新")

            # 該 bundle 對該 lighting+zone 的舊分全失效
            from capi_model_registry import invalidate_score_cache
            cleared = invalidate_score_cache(
                db, scoring_bundle_id=bundle_id, lighting=lighting, zone=zone,
            )
            _log(f"清除 {cleared} 筆 score cache（lighting={lighting}, zone={zone}）")

            _set_step("reload")
            # reload 失敗不能讓整個 job 標 failed：.pt 與 manifest 已成功落地，
            # 下次推論本來就會 lazy-load 新模型；reload 只是即時生效的優化。
            inferencer = self._capi_server_instance.inferencers.get(machine_id)
            if inferencer is None:
                _log(f"[v2] 機台 {machine_id} 無 inferencer cache，跳過 reload（下次首次推論會載入新模型）")
            else:
                try:
                    inferencer.reload_submodel(machine_id, lighting, zone)
                    _log(f"[v2] 已通知 inferencer reload {machine_id}/{lighting}/{zone}")
                except Exception as reload_err:
                    _log(f"[v2] reload 失敗（不影響重訓結果）：{reload_err}")
                    logger.warning("reload_submodel raised: %s", reload_err, exc_info=True)

            with state["lock"]:
                state["job"]["step"] = "done"
                state["job"]["state"] = "completed"
                state["job"]["summary"] = {
                    "auroc_old": old_auroc,
                    "auroc_new": new_auroc,
                    "tile_count_old": old_tile_count,
                    "tile_count_new": new_tile_count,
                }
            _log(f"✓ 重訓完成")

        except Exception as e:
            tb = traceback.format_exc()
            _log(f"✗ 失敗: {e}")
            for line in tb.rstrip().splitlines()[-8:]:
                _log(f"  {line}")
            with state["lock"]:
                if state["job"] is not None:
                    state["job"]["state"] = "failed"
                    state["job"]["error"] = str(e)
            logger.error("submodel retrain worker failed: %s", e, exc_info=True)

    @staticmethod
    def _free_server_gpu_cache() -> None:
        """主動釋放 server process 內 PyTorch caching allocator 的閒置 VRAM。

        prefilter / self-scan 用過的 TorchInferencer 即使被 GC，PyTorch 仍把
        freed blocks 留在 cache 裡不還給 driver；後續訓練 subprocess 因此擠
        不出 fraction 上限的 VRAM。呼叫 empty_cache() 把這些閒置 block 還回去。
        """
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @classmethod
    def _cancel_and_wait_scan_idle(cls, timeout_s: float = 15.0) -> None:
        """若有 prefilter/self/validation scan 還在跑，cancel 並等 worker 結束。

        worker 的 finally 會清 _inferencer_cache + empty_cache；join 確保這段
        跑完再進訓練 subprocess，避免兩 process 同時搶 GPU。
        """
        import time
        state = cls._scan_state
        with state["lock"]:
            job = state["job"]
            cancel_event = job.get("cancel_event") if job else None
            thread = job.get("thread") if job else None
            running = bool(job and job.get("state") == "running")
        if running and cancel_event is not None:
            cancel_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)
        # polling fallback：worker 可能還在寫 state，等到非 running 為止
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with state["lock"]:
                job = state["job"]
                if not job or job.get("state") != "running":
                    return
            time.sleep(0.2)

    @classmethod
    def _start_model_validation_job(
        cls,
        *,
        candidate: Dict[str, Any],
        baseline: Optional[Dict[str, Any]],
        samples: List[Dict[str, Any]],
        validation_base_dir: Path,
        server_inst,
    ) -> Tuple[bool, Dict[str, Any]]:
        from capi_model_validation import bundle_validation_snapshot

        state = cls._scan_state
        db = server_inst.database
        with state["lock"]:
            current = state["job"]
            if current and current.get("state") == "running":
                return False, {
                    "error": "已有模型掃描或 NG 能力考試進行中",
                    "job": {
                        key: value
                        for key, value in current.items()
                        if key not in {"cancel_event", "thread"}
                    },
                }

            run_id = db.create_model_validation_run(
                candidate_bundle_id=int(candidate["id"]),
                baseline_bundle_id=int(baseline["id"]) if baseline else None,
                machine_id=str(candidate.get("machine_id") or ""),
                sample_count=len(samples),
                candidate_snapshot=bundle_validation_snapshot(candidate),
                baseline_snapshot=bundle_validation_snapshot(baseline) if baseline else None,
            )
            cancel_event = threading.Event()
            state["job"] = {
                "scan_id": f"model_validation_{run_id}",
                "run_id": run_id,
                "kind": "model_validation",
                "scoring_bundle_id": int(candidate["id"]),
                "baseline_bundle_id": int(baseline["id"]) if baseline else None,
                "total": len(samples),
                "done": 0,
                "skipped": 0,
                "state": "running",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "error": None,
                "cancel_event": cancel_event,
            }

        thread = threading.Thread(
            target=cls._model_validation_worker,
            args=(
                run_id,
                candidate,
                baseline,
                samples,
                Path(validation_base_dir),
                cancel_event,
                server_inst,
            ),
            daemon=True,
            name=f"model-validation-{run_id}",
        )
        with state["lock"]:
            if state["job"] and state["job"].get("run_id") == run_id:
                state["job"]["thread"] = thread
        try:
            thread.start()
        except Exception as exc:
            db.finish_model_validation_run(
                run_id,
                state="failed",
                error_message=str(exc),
            )
            with state["lock"]:
                if state["job"] and state["job"].get("run_id") == run_id:
                    state["job"]["state"] = "failed"
                    state["job"]["error"] = str(exc)
            return False, {"error": f"NG 能力考試啟動失敗: {exc}"}
        return True, {"run_id": run_id, "total": len(samples), "state": "pending"}

    @classmethod
    def _model_validation_worker(
        cls,
        run_id: int,
        candidate: Dict[str, Any],
        baseline: Optional[Dict[str, Any]],
        samples: List[Dict[str, Any]],
        validation_base_dir: Path,
        cancel_event: threading.Event,
        server_inst,
    ) -> None:
        from capi_inference import SubmodelScorer
        from capi_model_validation import (
            build_model_validation_summary,
            score_bundle_sample,
        )

        db = server_inst.database
        state = cls._scan_state
        scorer = None
        final_state = "completed"
        final_error = ""

        def _empty_result(sample: Dict[str, Any]) -> Dict[str, Any]:
            review_id = int(sample.get("review_id") or 0)
            return {
                "sample_id": int(sample["id"]),
                "review_id": review_id if review_id > 0 else None,
                "glass_id": str(sample.get("glass_id") or ""),
                "model_id": str(sample.get("model_id") or ""),
                "lighting": str(sample.get("lighting") or ""),
                "zone": str(sample.get("zone") or ""),
                "image_name": str(sample.get("image_name") or ""),
                "aoi_defect_code": str(sample.get("aoi_defect_code") or ""),
                "category": str(sample.get("category") or ""),
                "candidate_score": None,
                "candidate_threshold": None,
                "candidate_caught": None,
                "candidate_error": "",
                "baseline_score": None,
                "baseline_threshold": None,
                "baseline_caught": None,
                "baseline_error": "",
            }

        def _score_into(
            result: Dict[str, Any],
            sample: Dict[str, Any],
            bundle: Dict[str, Any],
            prefix: str,
        ) -> None:
            try:
                scored = score_bundle_sample(
                    bundle,
                    sample,
                    scorer,
                    validation_base_dir=validation_base_dir,
                )
                result[f"{prefix}_score"] = scored["score"]
                result[f"{prefix}_threshold"] = scored["threshold"]
                result[f"{prefix}_caught"] = scored["caught"]
            except Exception as exc:
                result[f"{prefix}_error"] = str(exc)[:1000]

        def _release_loaded_models() -> None:
            if scorer is not None:
                scorer._inferencer_cache.clear()
            cls._free_server_gpu_cache()

        try:
            db.start_model_validation_run(run_id)
            scorer = SubmodelScorer(
                gpu_lock=server_inst._gpu_lock,
                db=db,
                log_fn=lambda message: logger.info(
                    "[model validation %s] %s", run_id, message
                ),
            )

            grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
            for sample in samples:
                key = (
                    str(sample.get("lighting") or ""),
                    str(sample.get("zone") or "").lower(),
                )
                grouped.setdefault(key, []).append(sample)

            done = 0
            for unit_key in sorted(grouped):
                unit_samples = grouped[unit_key]
                result_by_id = {
                    int(sample["id"]): _empty_result(sample)
                    for sample in unit_samples
                }

                for sample in unit_samples:
                    if cancel_event.is_set():
                        break
                    _score_into(
                        result_by_id[int(sample["id"])],
                        sample,
                        candidate,
                        "candidate",
                    )
                _release_loaded_models()

                if baseline and not cancel_event.is_set():
                    for sample in unit_samples:
                        if cancel_event.is_set():
                            break
                        _score_into(
                            result_by_id[int(sample["id"])],
                            sample,
                            baseline,
                            "baseline",
                        )
                    _release_loaded_models()

                for sample in unit_samples:
                    result = result_by_id[int(sample["id"])]
                    if (
                        not result["candidate_error"]
                        and result["candidate_caught"] is None
                    ):
                        continue
                    db.save_model_validation_result(
                        run_id,
                        result,
                        progress=done + 1,
                    )
                    done += 1
                    with state["lock"]:
                        if state["job"] and state["job"].get("run_id") == run_id:
                            state["job"]["done"] = done
                if cancel_event.is_set():
                    final_state = "cancelled"
                    break

            results = db.list_model_validation_results(run_id)
            summary = build_model_validation_summary(
                results,
                has_baseline=baseline is not None,
            )
            db.finish_model_validation_run(
                run_id,
                state=final_state,
                summary=summary,
            )
        except Exception as exc:
            final_state = "failed"
            final_error = str(exc)
            logger.exception("model validation worker crashed: run=%s", run_id)
            try:
                partial_results = db.list_model_validation_results(run_id)
                partial_summary = build_model_validation_summary(
                    partial_results,
                    has_baseline=baseline is not None,
                )
                db.finish_model_validation_run(
                    run_id,
                    state="failed",
                    summary=partial_summary,
                    error_message=final_error,
                )
            except Exception:
                logger.exception(
                    "failed to persist model validation failure: run=%s",
                    run_id,
                )
        finally:
            if scorer is not None:
                scorer._inferencer_cache.clear()
            scorer = None
            cls._free_server_gpu_cache()
            with state["lock"]:
                if state["job"] and state["job"].get("run_id") == run_id:
                    state["job"]["state"] = final_state
                    state["job"]["done"] = len(
                        db.list_model_validation_results(run_id)
                    )
                    state["job"]["error"] = final_error or None

    @classmethod
    def _start_scan_job(
        cls,
        kind: str,                     # "self" | "prefilter"
        scoring_bundle_id: int,
        bundle_dir: "Path",
        tile_pool_job_id: str,
        lighting: str,
        zone: str,
        tile_ids: list,
        server_inst,
    ) -> tuple:
        """嘗試啟動 scan job。回傳 (started: bool, response_dict)。"""
        import uuid
        state = cls._scan_state
        with state["lock"]:
            current = state["job"]
            if current and current.get("state") == "running":
                return False, {"error": "已有 scan job 進行中", "job": current}
            scan_id = "scan_" + uuid.uuid4().hex[:12]
            cancel_event = threading.Event()
            state["job"] = {
                "scan_id": scan_id,
                "kind": kind,
                "scoring_bundle_id": scoring_bundle_id,
                "tile_pool_job_id": tile_pool_job_id,
                "lighting": lighting,
                "zone": zone,
                "total": len(tile_ids),
                "done": 0,
                "skipped": 0,
                "state": "running",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "error": None,
                "cancel_event": cancel_event,
            }

        thread = threading.Thread(
            target=cls._scan_worker,
            args=(scan_id, scoring_bundle_id, bundle_dir, tile_pool_job_id,
                  lighting, zone, tile_ids, cancel_event, server_inst),
            daemon=True,
            name=f"scan-{scan_id}",
        )
        # 存 thread 物件，讓 _cancel_and_wait_scan_idle 可以 join，確保
        # finally 區塊（含 GPU cache 清理）跑完再進訓練 subprocess。
        with state["lock"]:
            if state["job"] and state["job"].get("scan_id") == scan_id:
                state["job"]["thread"] = thread
        thread.start()
        return True, {"scan_id": scan_id, "total": len(tile_ids)}

    @classmethod
    def _scan_worker(cls, scan_id, scoring_bundle_id, bundle_dir,
                      tile_pool_job_id, lighting, zone, tile_ids,
                      cancel_event, server_inst):
        from capi_inference import SubmodelScorer
        state = cls._scan_state

        def _progress(done, total):
            with state["lock"]:
                if state["job"] and state["job"]["scan_id"] == scan_id:
                    state["job"]["done"] = done

        def _log(msg):
            logger.info("[scan %s] %s", scan_id, msg)

        scorer = None
        try:
            scorer = SubmodelScorer(
                gpu_lock=server_inst._gpu_lock,
                db=server_inst.database,
                log_fn=_log,
            )
            result = scorer.score_tiles(
                scoring_bundle_id=scoring_bundle_id,
                bundle_dir=bundle_dir,
                lighting=lighting, zone=zone,
                tile_pool_job_id=tile_pool_job_id,
                tile_ids=tile_ids,
                cancel_event=cancel_event,
                progress_cb=_progress,
            )
            with state["lock"]:
                if state["job"] and state["job"]["scan_id"] == scan_id:
                    state["job"]["state"] = "cancelled" if result["cancelled"] else "done"
                    state["job"]["done"] = result["scanned"] + result["skipped"]
                    state["job"]["skipped"] = result["skipped"]
        except FileNotFoundError as e:
            with state["lock"]:
                if state["job"] and state["job"]["scan_id"] == scan_id:
                    state["job"]["state"] = "failed"
                    state["job"]["error"] = str(e)
        except Exception as e:
            logger.exception("scan worker crashed")
            with state["lock"]:
                if state["job"] and state["job"]["scan_id"] == scan_id:
                    state["job"]["state"] = "failed"
                    state["job"]["error"] = str(e)
        finally:
            # 釋放 SubmodelScorer 載入的 TorchInferencer，避免 PyTorch caching
            # allocator 把 prefilter 用過的 VRAM 留在 server process 裡，
            # 影響後續訓練 subprocess 的 GPU 配額。
            if scorer is not None:
                try:
                    scorer._inferencer_cache.clear()
                except Exception:
                    pass
            scorer = None
            cls._free_server_gpu_cache()

    def _handle_scan_self_score(self):
        """POST /api/models/<bundle_id>/scan_self_score
        body: {"lighting": str, "zone": "inner"|"edge"}
        """
        from capi_train_new import LIGHTINGS, ZONES
        parts = self.path.split("/")
        try:
            bundle_id = int(parts[3])
        except (ValueError, IndexError):
            self._send_json({"error": "invalid bundle id"}, status=400); return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            self._send_json({"error": "invalid JSON"}, status=400); return

        lighting = body.get("lighting"); zone = body.get("zone")
        if lighting not in LIGHTINGS or zone not in ZONES:
            self._send_json({"error": "lighting/zone 不合法"}, status=400); return

        db = self._capi_server_instance.database
        bundle = db.get_model_bundle(bundle_id)
        if not bundle:
            self._send_json({"error": "bundle not found"}, status=404); return
        if not bundle.get("job_id"):
            self._send_json({"error": "此 bundle 無關聯 job_id（訓練資料已刪）"}, status=400)
            return

        # 自掃 = 該 bundle 對「自己訓練資料」算分
        pool = db.list_tile_pool(
            bundle["job_id"], lighting=lighting, zone=zone, source="ok",
        )
        if not pool:
            self._send_json({"state": "empty", "scanned": 0}); return
        tile_ids = [t["id"] for t in pool]

        # 已有 cache 全命中？告知前端不需重算
        cached = db.get_score_cache(bundle_id, tile_ids)
        if len(cached) == len(tile_ids):
            self._send_json({"cached_hit": True, "total": len(tile_ids)})
            return

        started, resp = CAPIWebHandler._start_scan_job(
            kind="self",
            scoring_bundle_id=bundle_id,
            bundle_dir=Path(bundle["bundle_path"]),
            tile_pool_job_id=bundle["job_id"],
            lighting=lighting, zone=zone, tile_ids=tile_ids,
            server_inst=self._capi_server_instance,
        )
        self._send_json(resp, status=200 if started else 409)

    def _handle_scan_prefilter_score(self):
        """POST /api/train/new/scan_prefilter_score
        body: {"job_id", "scoring_bundle_id", "lighting", "zone"}
        """
        from capi_train_new import LIGHTINGS, ZONES
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            self._send_json({"error": "invalid JSON"}, status=400); return

        tile_pool_job_id = body.get("job_id")
        scoring_bundle_id = body.get("scoring_bundle_id")
        lighting = body.get("lighting"); zone = body.get("zone")
        if not all([tile_pool_job_id, scoring_bundle_id, lighting, zone]):
            self._send_json({"error": "missing required fields"}, status=400); return
        try:
            scoring_bundle_id = int(scoring_bundle_id)
        except (TypeError, ValueError):
            self._send_json({"error": "scoring_bundle_id 必須是整數"}, status=400); return
        if lighting not in LIGHTINGS or zone not in ZONES:
            self._send_json({"error": "lighting/zone 不合法"}, status=400); return

        db = self._capi_server_instance.database
        scoring_bundle = db.get_model_bundle(scoring_bundle_id)
        if not scoring_bundle:
            self._send_json({"error": "scoring bundle not found"}, status=404); return

        pool = db.list_tile_pool(tile_pool_job_id, lighting=lighting, zone=zone, source="ok")
        if not pool:
            self._send_json({"state": "empty"}); return
        tile_ids = [t["id"] for t in pool]

        cached = db.get_score_cache(scoring_bundle_id, tile_ids)
        if len(cached) == len(tile_ids):
            self._send_json({"cached_hit": True, "total": len(tile_ids)})
            return

        started, resp = CAPIWebHandler._start_scan_job(
            kind="prefilter",
            scoring_bundle_id=scoring_bundle_id,
            bundle_dir=Path(scoring_bundle["bundle_path"]),
            tile_pool_job_id=tile_pool_job_id,
            lighting=lighting, zone=zone, tile_ids=tile_ids,
            server_inst=self._capi_server_instance,
        )
        self._send_json(resp, status=200 if started else 409)

    def _handle_scan_status(self):
        """GET /api/scan/status — 回目前唯一 scan job 狀態（沒有則 idle）。"""
        state = CAPIWebHandler._scan_state
        with state["lock"]:
            job = state["job"]
            if job is None:
                self._send_json({"state": "idle"})
                return
            # 不回傳 cancel_event / thread 等不可序列化物件
            payload = {k: v for k, v in job.items()
                       if k not in ("cancel_event", "thread")}
            self._send_json(payload)

    def _handle_scan_cancel(self):
        """POST /api/scan/cancel — 對當前 running job 設 cancel_event。"""
        state = CAPIWebHandler._scan_state
        with state["lock"]:
            job = state["job"]
            if not job or job["state"] != "running":
                self._send_json({"cancelled": False, "reason": "no running job"})
                return
            job["cancel_event"].set()
        self._send_json({"cancelled": True})

    def _handle_eligible_scoring_bundles(self):
        """GET /api/train/new/eligible_scoring_bundles
        回所有「.pt 檔健全的 trained bundle」清單，給 step3 prefilter 下拉用。
        """
        db = self._capi_server_instance.database
        bundles = db.list_model_bundles() or []
        from capi_train_new import LIGHTINGS, ZONES
        out = []
        for b in bundles:
            bundle_dir = Path(b["bundle_path"])
            # 至少要存在 1 個 .pt 才算可用（細項 lighting+zone 由 frontend 切 tab 才知）
            has_any_pt = any(
                (bundle_dir / f"{l}-{z}.pt").exists()
                for l in LIGHTINGS for z in ZONES
            )
            if not has_any_pt:
                continue
            label = (
                f"{b['machine_id']} / "
                f"{Path(b['bundle_path']).name}"
                f"{' ●active' if b.get('is_active') else ''}"
            )
            out.append({
                "id": b["id"],
                "machine_id": b["machine_id"],
                "trained_at": b.get("trained_at"),
                "is_active": bool(b.get("is_active")),
                "label": label,
            })
        # Active 優先；同一組內 trained_at 由新到舊
        out.sort(key=lambda x: x["trained_at"] or "", reverse=True)
        out.sort(key=lambda x: not x["is_active"])  # stable sort: active group first
        self._send_json({"bundles": out})

    def _handle_models_retrain_status(self):
        """GET /api/models/<id>/retrain_status?tail=200

        回目前 retrain job 的狀態與末 N 行 log。
        若沒有 job，或 job 對應的 bundle_id 與 path 中的 id 不符，回 {"job": null}。
        """
        from urllib.parse import parse_qs, urlparse
        parts = self.path.split("/")
        try:
            bundle_id = int(parts[3])
        except (ValueError, IndexError):
            self._send_json({"error": "invalid bundle id"}, status=400)
            return

        qs = parse_qs(urlparse(self.path).query)
        try:
            tail = max(0, min(int((qs.get("tail") or ["200"])[0]), 1000))
        except ValueError:
            tail = 200

        state = CAPIWebHandler._submodel_retrain_state
        with state["lock"]:
            job = state.get("job")
            if job is None or job.get("bundle_id") != bundle_id:
                # 沒有 job 或別的 bundle 的 job 跑著 → 對目前頁面而言視為無 job
                self._send_json({"job": None})
                return
            # 淺拷貝 + 截斷 log
            out = dict(job)
            out["log_lines"] = list(job["log_lines"][-tail:])

        self._send_json({"job": out})


def create_web_server(
    host: str,
    port: int,
    db,
    heatmap_base_dir: str,
    status_tracker=None,
    inferencer=None,
    heatmap_manager=None,
    gpu_lock=None,
    capi_server_instance=None,
    log_file=None,
) -> ThreadingHTTPServer:
    """
    建立 Web 伺服器

    Args:
        host: 綁定地址
        port: 綁定端口
        db: CAPIDatabase 實例
        heatmap_base_dir: 熱力圖儲存根目錄
        status_tracker: 伺服器狀態追蹤物件
        inferencer: CAPIInferencer 實例 (Optional, for debug inference)
        heatmap_manager: HeatmapManager 實例 (Optional, for debug inference)
        gpu_lock: GPU 排隊鎖 (Optional, for debug inference)
        capi_server_instance: CAPIServer 實例 (Optional, for config hot-reload)
        log_file: 日誌檔案路徑 (Optional, for log viewer)
    """
    CAPIWebHandler.db = db
    CAPIWebHandler.heatmap_base_dir = heatmap_base_dir
    CAPIWebHandler.status_tracker = status_tracker
    CAPIWebHandler.inferencer = inferencer
    CAPIWebHandler.heatmap_manager = heatmap_manager
    CAPIWebHandler._gpu_lock = gpu_lock
    CAPIWebHandler._capi_server_instance = capi_server_instance
    CAPIWebHandler._log_file = log_file
    CAPIWebHandler._rerun_tasks = {}
    CAPIWebHandler._rerun_lock = threading.Lock()
    CAPIWebHandler._dataset_export_state = {
        "lock": threading.Lock(),
        "current_job": None,         # dict: job_id, state, current, total, last_glass_id, started_at
        "cancel_event": threading.Event(),
        "last_summary": None,        # JobSummary instance
        "manifest_lock": threading.Lock(),  # 保護 manifest.csv 手動 curation 的 read-modify-write
    }
    CAPIWebHandler._retrain_state = {
        "lock": threading.Lock(),
        "job": None,
    }
    CAPIWebHandler._settings_sessions = {}
    CAPIWebHandler._settings_session_lock = threading.Lock()
    CAPIWebHandler._update_apply_lock = threading.Lock()
    CAPIWebHandler._mark_calibration_lock = threading.Lock()
    CAPIWebHandler._train_new_jobs = {}
    CAPIWebHandler._train_new_jobs_lock = threading.Lock()
    CAPIWebHandler._train_slot = {
        "lock": threading.Lock(),
        "active_job_id": None,
    }
    CAPIWebHandler._scan_state = {
        "lock": threading.Lock(),
        "job": None,
    }
    try:
        from capi_mark_detector import (
            set_active_mark_profile,
            set_active_mark_profile_loader,
        )

        set_active_mark_profile_loader(
            lambda: db.get_active_mark_profile(timeout=1)
        )
        active_mark_profile = db.get_active_mark_profile()
        set_active_mark_profile(
            active_mark_profile["profile"],
            active_mark_profile["id"],
        )
    except Exception as exc:
        logger.error("Failed to load active MARK profile for web server: %s", exc)
        from capi_mark_detector import (
            set_active_mark_profile,
            set_active_mark_profile_loader,
        )

        set_active_mark_profile_loader(
            lambda: db.get_active_mark_profile(timeout=1)
        )
        set_active_mark_profile(None, 0)
    CAPIWebHandler._reconcile_train_new_artifacts(db)
    CAPIWebHandler.init_jinja()

    class ReusableThreadingHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = True

    server = ReusableThreadingHTTPServer((host, port), CAPIWebHandler)
    return server


def start_web_server_thread(
    host: str,
    port: int,
    db,
    heatmap_base_dir: str,
    status_tracker=None,
    inferencer=None,
    heatmap_manager=None,
    gpu_lock=None,
    capi_server_instance=None,
    log_file=None,
    ) -> threading.Thread:
    """
    在背景執行緒啟動 Web 伺服器

    Returns:
        Web 伺服器執行緒
    """
    server = create_web_server(
        host, port, db, heatmap_base_dir, status_tracker,
        inferencer=inferencer,
        heatmap_manager=heatmap_manager,
        gpu_lock=gpu_lock,
        capi_server_instance=capi_server_instance,
        log_file=log_file,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="web-server")
    thread.web_server = server
    thread.start()
    logger.info(f"Web server started on http://{host}:{port}")
    return thread


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from capi_database import CAPIDatabase

    logging.basicConfig(level=logging.INFO)

    # 使用 test_results.db 測試 (含真實推論資料)
    db_path = os.path.join(str(Path(__file__).parent), "test_results.db")
    db = CAPIDatabase(db_path)

    # 熱力圖目錄
    test_heatmap_dir = os.path.join(str(Path(__file__).parent), "test_heatmaps")

    print(f"Test DB: {db_path}")
    print(f"Starting web server on http://localhost:8080")
    print("Press Ctrl+C to stop")

    server = create_web_server("0.0.0.0", 8080, db, test_heatmap_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        print("\nStopped.")


