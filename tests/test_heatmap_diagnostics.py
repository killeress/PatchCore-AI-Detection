import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_heatmap_diagnostics import (
    analyze_heatmap_peaks,
    detect_local_maxima,
    local_dust_coverage,
    region_grow_dust_coverage,
)


def test_detect_local_maxima_supports_relative_absolute_and_distance():
    heatmap = np.zeros((40, 40), dtype=np.float32)
    heatmap[5, 5] = 1.0
    heatmap[6, 6] = 0.9  # suppressed by min_distance
    heatmap[30, 30] = 0.6
    heatmap[20, 20] = 0.2

    rel = detect_local_maxima(heatmap, min_distance=3, threshold_rel=0.5)
    assert rel.tolist() == [[5, 5], [30, 30]]

    absolute = detect_local_maxima(
        heatmap, min_distance=3, threshold_abs=0.7
    )
    assert absolute.tolist() == [[5, 5]]


def test_dust_coverage_uses_11x11_and_peak_half_region():
    heatmap = np.zeros((31, 31), dtype=np.float32)
    heatmap[13:18, 13:18] = 0.6
    heatmap[15, 15] = 1.0
    dust = np.zeros_like(heatmap, dtype=np.uint8)
    dust[13:18, 13:16] = 255

    assert local_dust_coverage(dust, 15, 15, 11) == 15 / 121
    coverage, area = region_grow_dust_coverage(
        heatmap, dust, 15, 15, drop_ratio=0.5
    )
    assert area == 25
    assert coverage == 15 / 25


def test_strong_dust_bubble_consumes_top_percent_but_aoi_peak_is_recovered():
    heatmap = np.full((100, 100), 0.01, dtype=np.float32)
    yy, xx = np.ogrid[:100, :100]

    # Strong, broad bubble at bottom-right: it consumes more than the Top 0.5%
    # quota, so the weaker AOI black-dot peak is below the percentile cutoff.
    bubble = (xx - 85) ** 2 + (yy - 85) ** 2 <= 6 ** 2
    heatmap[bubble] = 0.92
    heatmap[85, 85] = 1.0

    # Genuine visual defect around the AOI coordinate, weaker but still a local
    # maximum when the AOI window is evaluated independently.
    defect = (xx - 30) ** 2 + (yy - 30) ** 2 <= 3 ** 2
    heatmap[defect] = 0.32
    heatmap[30, 30] = 0.40

    dust = np.zeros((100, 100), dtype=np.uint8)
    dust[bubble] = 255

    report = analyze_heatmap_peaks(
        heatmap,
        dust,
        aoi_xy=(30, 30),
        aoi_window=10,
        top_percent=0.5,
        min_distance=5,
        threshold_rel=0.5,       # global threshold is 0.5: misses 0.40
        aoi_threshold_rel=0.5,   # AOI-local threshold is 0.20: finds 0.40
        global_score=1.0,
    )

    dominant = report["dominant_peak"]
    aoi_best = report["aoi_best_peak"]
    assert (dominant["x"], dominant["y"]) == (85, 85)
    assert dominant["in_dust"] is True
    assert dominant["kept_by_top_percent"] is True

    assert (aoi_best["x"], aoi_best["y"]) == (30, 30)
    assert aoi_best["sources"] == ["aoi"]
    assert aoi_best["in_dust"] is False
    assert aoi_best["kept_by_top_percent"] is False
    assert aoi_best["relative_to_global_max"] == pytest.approx(0.4)
    assert aoi_best["estimated_score"] == pytest.approx(0.4)
    assert "Top%" in aoi_best["interpretation_zh"]
    assert "氣泡/灰塵搶走高分" in report["conclusion_zh"]
    assert report["diagnostic_only"] is True
    assert "不可取代正式 OK/NG 判定" in report["disclaimer_zh"]


def test_aoi_and_global_peaks_are_unioned_without_duplicate():
    heatmap = np.zeros((30, 30), dtype=np.float32)
    heatmap[15, 15] = 1.0

    report = analyze_heatmap_peaks(
        heatmap,
        aoi_xy=(15, 15),
        aoi_window=5,
        min_distance=2,
        threshold_rel=0.2,
        aoi_threshold_rel=0.2,
        top_percent=1.0,
    )

    assert report["global_peak_count"] == 1
    assert report["aoi_peak_count"] == 1
    assert report["union_peak_count"] == 1
    assert report["peaks"][0]["sources"] == ["aoi", "global"]
