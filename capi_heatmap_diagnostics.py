"""PatchCore heatmap peak diagnostics (CPU-only).

This module intentionally contains no model/GPU or scikit-image dependency.  It
explains a heatmap that has already been produced; it must not be used as the
production OK/NG decision.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DIAGNOSTIC_DISCLAIMER_ZH = (
    "此結果只用於排查熱力圖與 Top% 搶分現象，不可取代正式 OK/NG 判定。"
)


def _as_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(anomaly_map, dtype=np.float32)
    if heatmap.ndim == 3 and heatmap.shape[2] == 1:
        heatmap = heatmap[:, :, 0]
    if heatmap.ndim != 2 or heatmap.size == 0:
        raise ValueError("anomaly_map 必須是非空的 2D 陣列")
    # A diagnostic should remain usable even if an upstream debug dump contains
    # a non-finite value.  Production dust processing also ignores negative heat.
    return np.maximum(np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def _as_dust_mask(
    dust_mask: Optional[np.ndarray], shape: Tuple[int, int]
) -> Tuple[np.ndarray, bool]:
    if dust_mask is None:
        return np.zeros(shape, dtype=bool), False
    mask = np.asarray(dust_mask)
    if mask.ndim == 3:
        if mask.shape[2] == 1:
            mask = mask[:, :, 0]
        else:
            mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("dust_mask 必須是 2D（或可轉灰階的 BGR）陣列")
    if mask.shape != shape:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask > 0, True


def detect_local_maxima(
    anomaly_map: np.ndarray,
    *,
    min_distance: int = 10,
    threshold_rel: Optional[float] = None,
    threshold_abs: Optional[float] = None,
    num_peaks: Optional[int] = None,
) -> np.ndarray:
    """Find local maxima with OpenCV/Numpy and return ``(y, x)`` coordinates.

    ``threshold_rel`` is relative to the maximum of the supplied map.  When both
    thresholds are given, the stricter one is used.  Equal-valued plateaus are
    collapsed to one deterministic point.  Results are sorted by descending
    heatmap value.
    """

    heatmap = _as_heatmap(anomaly_map)
    distance = int(min_distance)
    if distance < 0:
        raise ValueError("min_distance 不可小於 0")
    if threshold_rel is not None and float(threshold_rel) < 0:
        raise ValueError("threshold_rel 不可小於 0")
    if num_peaks is not None and int(num_peaks) < 0:
        raise ValueError("num_peaks 不可小於 0")

    map_max = float(np.max(heatmap))
    if map_max <= 0.0 or num_peaks == 0:
        return np.empty((0, 2), dtype=np.int32)

    thresholds = [0.0]
    if threshold_abs is not None:
        thresholds.append(float(threshold_abs))
    if threshold_rel is not None:
        thresholds.append(float(threshold_rel) * map_max)
    threshold = max(thresholds)

    kernel_size = distance * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(heatmap, kernel)
    candidate_mask = (heatmap >= dilated) & (heatmap > threshold)
    if not np.any(candidate_mask):
        return np.empty((0, 2), dtype=np.int32)

    # Collapse a flat maximum to one point.  Choosing the point nearest the
    # plateau centroid makes the result stable and easier to explain visually.
    count, labels = cv2.connectedComponents(candidate_mask.astype(np.uint8), 8)
    candidates: List[Tuple[float, int, int]] = []
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        if ys.size == 0:
            continue
        values = heatmap[ys, xs]
        best_value = float(np.max(values))
        best_indices = np.flatnonzero(values == best_value)
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        chosen = min(
            (int(i) for i in best_indices),
            key=lambda i: (
                (float(ys[i]) - cy) ** 2 + (float(xs[i]) - cx) ** 2,
                int(ys[i]),
                int(xs[i]),
            ),
        )
        candidates.append((best_value, int(ys[chosen]), int(xs[chosen])))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    # Dilation already enforces the usual square min-distance neighbourhood.
    # Keep a greedy guard because two disconnected plateau fragments can still
    # sit exactly on a neighbourhood boundary.
    selected: List[Tuple[int, int]] = []
    for _value, y, x in candidates:
        if distance and any(
            abs(y - old_y) <= distance and abs(x - old_x) <= distance
            for old_y, old_x in selected
        ):
            continue
        selected.append((y, x))
        if num_peaks is not None and len(selected) >= int(num_peaks):
            break
    return np.asarray(selected, dtype=np.int32).reshape((-1, 2))


def local_dust_coverage(
    dust_mask: np.ndarray, peak_y: int, peak_x: int, window: int = 11
) -> float:
    """Return dust coverage in a clipped square window around a peak."""

    mask = np.asarray(dust_mask)
    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=2)
    else:
        mask = mask > 0
    if mask.ndim != 2:
        raise ValueError("dust_mask 必須是 2D 陣列")
    size = int(window)
    if size <= 0 or size % 2 == 0:
        raise ValueError("window 必須是正奇數")
    y, x = int(peak_y), int(peak_x)
    h, w = mask.shape
    if not (0 <= y < h and 0 <= x < w):
        raise ValueError("peak 座標超出 dust_mask")
    half = size // 2
    region = mask[max(0, y - half) : min(h, y + half + 1),
                  max(0, x - half) : min(w, x + half + 1)]
    return float(np.count_nonzero(region) / region.size) if region.size else 0.0


def region_grow_dust_coverage(
    anomaly_map: np.ndarray,
    dust_mask: np.ndarray,
    peak_y: int,
    peak_x: int,
    drop_ratio: float = 0.5,
) -> Tuple[float, int]:
    """Measure dust coverage of the connected ``peak * drop_ratio`` region."""

    heatmap = _as_heatmap(anomaly_map)
    mask, _ = _as_dust_mask(dust_mask, heatmap.shape)
    ratio = float(drop_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("drop_ratio 必須介於 0 與 1 之間")
    y, x = int(peak_y), int(peak_x)
    if not (0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]):
        raise ValueError("peak 座標超出 anomaly_map")
    peak_value = float(heatmap[y, x])
    if peak_value <= 0.0:
        return 0.0, 0

    binary = (heatmap >= peak_value * ratio).astype(np.uint8)
    _count, labels = cv2.connectedComponents(binary, connectivity=8)
    peak_label = int(labels[y, x])
    if peak_label == 0:
        return 0.0, 0
    region = labels == peak_label
    area = int(np.count_nonzero(region))
    overlap = int(np.count_nonzero(region & mask))
    return (float(overlap / area) if area else 0.0), area


def _peak_interpretation_zh(
    *,
    dust_available: bool,
    in_dust: bool,
    local_coverage: float,
    grow_coverage: float,
    kept_by_top_percent: bool,
    from_aoi_scan: bool,
) -> str:
    if dust_available and (in_dust or grow_coverage >= 0.5 or local_coverage >= 0.5):
        base = "此峰與灰塵遮罩高度重疊，強訊號較可能來自灰塵或氣泡"
    elif dust_available:
        base = "此峰與灰塵遮罩重疊低，需回看原圖確認是否為真缺陷"
    else:
        base = "未提供灰塵遮罩，無法判讀此峰是否由灰塵或氣泡造成"
    if not kept_by_top_percent:
        base += "；它低於 Top% 切點，正式流程可能在二值化時看不到"
    if from_aoi_scan:
        base += "；AOI 區域獨立搜尋仍找到此局部峰值"
    return base + "。"


def _stats(heatmap: np.ndarray) -> Dict[str, Any]:
    positive = heatmap[heatmap > 0]
    return {
        "width": int(heatmap.shape[1]),
        "height": int(heatmap.shape[0]),
        "pixel_count": int(heatmap.size),
        "positive_pixel_count": int(positive.size),
        "min": float(np.min(heatmap)),
        "max": float(np.max(heatmap)),
        "mean": float(np.mean(heatmap)),
        "positive_mean": float(np.mean(positive)) if positive.size else 0.0,
        "p50": float(np.percentile(positive, 50)) if positive.size else 0.0,
        "p95": float(np.percentile(positive, 95)) if positive.size else 0.0,
        "p99": float(np.percentile(positive, 99)) if positive.size else 0.0,
        "p99_5": float(np.percentile(positive, 99.5)) if positive.size else 0.0,
        "p99_8": float(np.percentile(positive, 99.8)) if positive.size else 0.0,
    }


def _top_percent_info(heatmap: np.ndarray, top_percent: float) -> Dict[str, Any]:
    percent = float(top_percent)
    if not 0.0 < percent <= 100.0:
        raise ValueError("top_percent 必須大於 0 且不超過 100")
    positive = heatmap[heatmap > 0]
    if not positive.size:
        return {
            "percent": percent,
            "cutoff": None,
            "retained_pixel_count": 0,
            "retained_positive_ratio": 0.0,
            "description_zh": f"熱力圖沒有正值，無法計算 Top {percent:g}% 切點。",
        }
    cutoff = float(np.percentile(positive, 100.0 - percent))
    retained = int(np.count_nonzero(heatmap >= cutoff))
    return {
        "percent": percent,
        "cutoff": cutoff,
        "retained_pixel_count": retained,
        "retained_positive_ratio": float(retained / positive.size),
        "description_zh": (
            f"Top {percent:g}% 切點為 {cutoff:.6g}；低於此值的熱點會在 "
            "Top% 二值化階段被排除。相同分數像素可能使實際保留比例略高。"
        ),
    }


def analyze_heatmap_peaks(
    anomaly_map: np.ndarray,
    dust_mask: Optional[np.ndarray] = None,
    *,
    aoi_xy: Optional[Sequence[float]] = None,
    aoi_window: int = 64,
    top_percent: float = 0.5,
    min_distance: int = 10,
    threshold_rel: Optional[float] = 0.3,
    threshold_abs: Optional[float] = None,
    aoi_threshold_rel: Optional[float] = 0.3,
    aoi_threshold_abs: Optional[float] = None,
    global_score: Optional[float] = None,
    max_peaks: int = 100,
) -> Dict[str, Any]:
    """Explain global and AOI-local heatmap peaks as JSON-serializable data.

    AOI-local detection computes ``threshold_rel`` against the AOI window itself.
    Its peaks are unioned with global peaks, so a weaker AOI defect can remain
    visible even when a strong dust bubble consumes the global Top% quota.

    ``estimated_score`` is a linear diagnostic estimate:
    ``global_score * raw_peak / global_heatmap_max``.  It is not a PatchCore
    re-inference score and is deliberately labelled as an estimate.
    """

    heatmap = _as_heatmap(anomaly_map)
    dust, dust_available = _as_dust_mask(dust_mask, heatmap.shape)
    dust_has_pixels = bool(np.any(dust)) if dust_available else False
    top_info = _top_percent_info(heatmap, top_percent)
    cutoff = top_info["cutoff"]
    map_max = float(np.max(heatmap))
    peak_limit = int(max_peaks)
    if peak_limit <= 0:
        raise ValueError("max_peaks 必須大於 0")

    global_coords = detect_local_maxima(
        heatmap,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        threshold_abs=threshold_abs,
        num_peaks=peak_limit,
    )

    aoi_center: Optional[Tuple[float, float]] = None
    aoi_bounds: Optional[Dict[str, int]] = None
    aoi_coords: List[Tuple[int, int]] = []
    if aoi_xy is not None:
        if len(aoi_xy) != 2:
            raise ValueError("aoi_xy 必須是 (x, y)")
        aoi_center = (float(aoi_xy[0]), float(aoi_xy[1]))
        radius = int(aoi_window)
        if radius < 0:
            raise ValueError("aoi_window 不可小於 0")
        cx, cy = int(round(aoi_center[0])), int(round(aoi_center[1]))
        y1, y2 = max(0, cy - radius), min(heatmap.shape[0], cy + radius + 1)
        x1, x2 = max(0, cx - radius), min(heatmap.shape[1], cx + radius + 1)
        aoi_bounds = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        if x1 < x2 and y1 < y2:
            local_coords = detect_local_maxima(
                heatmap[y1:y2, x1:x2],
                min_distance=min_distance,
                threshold_rel=aoi_threshold_rel,
                threshold_abs=aoi_threshold_abs,
                num_peaks=peak_limit,
            )
            aoi_coords = [
                (int(local_y + y1), int(local_x + x1))
                for local_y, local_x in local_coords
            ]

    sources: Dict[Tuple[int, int], set] = {}
    for y, x in global_coords:
        sources.setdefault((int(y), int(x)), set()).add("global")
    for y, x in aoi_coords:
        sources.setdefault((y, x), set()).add("aoi")

    peaks: List[Dict[str, Any]] = []
    for (y, x), source_set in sources.items():
        raw_peak = float(heatmap[y, x])
        relative = float(raw_peak / map_max) if map_max > 0 else 0.0
        estimated_score = (
            float(global_score) * relative if global_score is not None else None
        )
        kept = bool(cutoff is not None and raw_peak >= float(cutoff))
        if dust_has_pixels:
            local_cov = local_dust_coverage(dust, y, x, 11)
            grow_cov, grow_area = region_grow_dust_coverage(
                heatmap, dust, y, x, 0.5
            )
        else:
            local_cov, grow_cov, grow_area = 0.0, 0.0, 0
        distance = (
            math.hypot(float(x) - aoi_center[0], float(y) - aoi_center[1])
            if aoi_center is not None
            else None
        )
        source_names = sorted(source_set)
        from_aoi = "aoi" in source_set
        peaks.append(
            {
                "x": x,
                "y": y,
                # Compatibility aliases used by validate_peak_detection.py.
                "score": raw_peak,
                "raw_peak": raw_peak,
                "relative_to_global_max": relative,
                "relative_percent": relative * 100.0,
                "estimated_score": estimated_score,
                "estimated_score_note_zh": (
                    "依全域分數按熱力峰值比例線性估算，非重新推論分數。"
                    if estimated_score is not None
                    else "未提供全域分數，無法估算此峰的分數。"
                ),
                "kept_by_top_percent": kept,
                "top_percent_status_zh": (
                    "高於或等於 Top% 切點，會保留"
                    if kept
                    else "低於 Top% 切點，會被排除"
                ),
                "sources": source_names,
                "source_zh": (
                    "全圖與 AOI 區域皆找到"
                    if len(source_names) == 2
                    else ("AOI 區域獨立找到" if from_aoi else "全圖找到")
                ),
                "in_dust": bool(dust[y, x]) if dust_available else None,
                "local_dust_cov_11x11": local_cov if dust_available else None,
                "region_grow_cov": grow_cov if dust_available else None,
                "region_grow_area": grow_area,
                "aoi_distance_px": distance,
                "interpretation_zh": _peak_interpretation_zh(
                    dust_available=dust_available,
                    in_dust=bool(dust[y, x]),
                    local_coverage=local_cov,
                    grow_coverage=grow_cov,
                    kept_by_top_percent=kept,
                    from_aoi_scan=from_aoi,
                ),
            }
        )
    peaks.sort(key=lambda peak: (-peak["raw_peak"], peak["y"], peak["x"]))
    for index, peak in enumerate(peaks, 1):
        peak["rank"] = index

    dominant = peaks[0] if peaks else None
    aoi_candidates = [peak for peak in peaks if "aoi" in peak["sources"]]
    aoi_best = max(aoi_candidates, key=lambda peak: peak["raw_peak"], default=None)

    if not peaks:
        conclusion = "沒有找到符合門檻的局部熱點，請降低診斷門檻或確認熱力圖內容。"
    elif aoi_center is None:
        conclusion = (
            "已列出全圖主要熱點；未提供 AOI 座標，因此無法比較 AOI 黑點與全圖搶分來源。"
        )
    elif aoi_best is None:
        conclusion = (
            "AOI 視窗內沒有找到符合門檻的局部峰值；這只代表目前診斷參數未找到訊號，"
            "不能直接判定 OK。"
        )
    elif (
        dominant is not None
        and dominant["in_dust"] is True
        and not aoi_best["kept_by_top_percent"]
    ):
        conclusion = (
            "全圖最高峰與灰塵遮罩重疊，可能是氣泡/灰塵搶走高分；AOI 附近仍有較弱"
            "局部峰值，但低於 Top% 切點，可能因此在正式 dust 分區前被丟掉。請回看"
            "原圖確認 AOI 黑點。"
        )
    elif aoi_best["in_dust"] is False:
        conclusion = (
            "AOI 附近找到與灰塵遮罩重疊低的局部峰值，需回看原圖確認真缺陷；"
            "本診斷不直接改寫正式判定。"
        )
    else:
        conclusion = (
            "AOI 附近最強峰與灰塵遮罩重疊，較可能受灰塵/氣泡影響；仍需回看原圖。"
        )

    return {
        "diagnostic_only": True,
        "disclaimer_zh": DIAGNOSTIC_DISCLAIMER_ZH,
        "global_stats": _stats(heatmap),
        "global_score": float(global_score) if global_score is not None else None,
        "top_percent": top_info,
        "parameters": {
            "min_distance": int(min_distance),
            "threshold_rel": (
                float(threshold_rel) if threshold_rel is not None else None
            ),
            "threshold_abs": (
                float(threshold_abs) if threshold_abs is not None else None
            ),
            "aoi_threshold_rel": (
                float(aoi_threshold_rel) if aoi_threshold_rel is not None else None
            ),
            "aoi_threshold_abs": (
                float(aoi_threshold_abs) if aoi_threshold_abs is not None else None
            ),
            "aoi_window": int(aoi_window),
        },
        "aoi": {
            "x": aoi_center[0] if aoi_center is not None else None,
            "y": aoi_center[1] if aoi_center is not None else None,
            "window_bounds": aoi_bounds,
        },
        "peaks": peaks,
        "global_peak_count": int(len(global_coords)),
        "aoi_peak_count": int(len(aoi_coords)),
        "union_peak_count": len(peaks),
        "dominant_peak": dominant,
        "aoi_best_peak": aoi_best,
        "conclusion_zh": conclusion,
    }


def run_peak_trial(
    anomaly_map: np.ndarray,
    dust_mask: np.ndarray,
    *,
    min_distance: int,
    threshold_rel: Optional[float] = None,
    threshold_abs: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Compatibility helper for the standalone validation script."""

    heatmap = _as_heatmap(anomaly_map)
    dust, _ = _as_dust_mask(dust_mask, heatmap.shape)
    results: List[Dict[str, Any]] = []
    for y, x in detect_local_maxima(
        heatmap,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        threshold_abs=threshold_abs,
    ):
        y, x = int(y), int(x)
        grow_cov, grow_area = region_grow_dust_coverage(
            heatmap, dust, y, x, 0.5
        )
        results.append(
            {
                "y": y,
                "x": x,
                "score": float(heatmap[y, x]),
                "in_dust": bool(dust[y, x]),
                "local_dust_cov_11x11": local_dust_coverage(dust, y, x, 11),
                "region_grow_cov": grow_cov,
                "region_grow_area": grow_area,
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results


__all__ = [
    "DIAGNOSTIC_DISCLAIMER_ZH",
    "analyze_heatmap_peaks",
    "detect_local_maxima",
    "local_dust_coverage",
    "region_grow_dust_coverage",
    "run_peak_trial",
]
