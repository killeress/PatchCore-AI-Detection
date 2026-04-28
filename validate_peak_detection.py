"""
peak_detection 驗證腳本 — 對「指定產品座標」直接觸發 AOI Edge fusion，
然後在 PC anomaly map 上跑 local maxima peak detection，驗證能否救回
被 Top X% 二值化擠掉的弱缺陷。

背景：
  現行 dust filter 流程是「全圖 Top 0.2% 二值化 → CC → per-region 判定」。
  當圖內存在強訊號（MARK / 強灰塵 / 另一個明顯缺陷），會把 0.2% 配額吃光，
  弱缺陷在二值化階段就消失，後面任何 dust 判定都救不到。

  方法 6（見 docs/2026-03-31-dust-filter-miss-analysis.md）改用 local maxima：
  不做配額式二值化，每個 peak 獨立判定。

使用方式：
    python validate_peak_detection.py <panel_dir> \
        --image <prefix_or_filename> \
        --coord <px,py> [--coord <px2,py2>] \
        [--config configs/capi_3f.yaml] \
        [--out <output_dir>]

範例（你的失敗 case）:
    mkdir -p /tmp/peak_check
    python validate_peak_detection.py \
        /CAPI07/TIANMU/yuantu/GN160JCEL250S/20260427/YQ21HE207H38 \
        --image WGF50500 \
        --coord 88,1153 \
        --config configs/capi_3f.yaml \
        --out /tmp/peak_check 2>&1 | tee /tmp/peak_check/run.log

座標說明：
  --coord 是「產品座標」(AOI 機檢回傳的 px, py)。腳本會用 result.raw_bounds 跟
  config.product_resolution 把它換算成影像座標。

輸出：
  - Console: 既有 per-region 判定 + 多組 peak detection 試驗結果
  - <out>/<image>_aoi_<px>_<py>_peaks.png       4-panel 視覺化
  - <out>/<image>_aoi_<px>_<py>_anomaly.npy     原始 anomaly map
  - <out>/<image>_aoi_<px>_<py>_dust_mask.npy   dust mask
"""
import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

try:
    from skimage.feature import peak_local_max
except ImportError:
    print("❌ 需要 scikit-image: pip install scikit-image")
    sys.exit(1)

from capi_config import CAPIConfig
from capi_inference import (
    CAPIInferencer, DEFAULT_PRODUCT_RESOLUTION,
    resolve_product_resolution, MODEL_RESOLUTION_MAP,
)


# ============================================================
# Peak 周圍 dust 覆蓋計算
# ============================================================

def _local_dust_coverage(dust_mask, peak_y, peak_x, window=11):
    h, w = dust_mask.shape[:2]
    half = window // 2
    y1 = max(0, peak_y - half); y2 = min(h, peak_y + half + 1)
    x1 = max(0, peak_x - half); x2 = min(w, peak_x + half + 1)
    region = dust_mask[y1:y2, x1:x2] > 0
    return float(np.count_nonzero(region) / region.size) if region.size else 0.0


def _region_grow_coverage(anomaly_map, dust_mask, peak_y, peak_x, drop_ratio=0.5):
    peak_score = float(anomaly_map[peak_y, peak_x])
    if peak_score <= 0:
        return 0.0, 0
    threshold = peak_score * drop_ratio
    binary = (anomaly_map >= threshold).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    peak_label = labels[peak_y, peak_x]
    if peak_label == 0:
        return 0.0, 0
    region_mask = labels == peak_label
    region_area = int(np.count_nonzero(region_mask))
    dust_overlap = int(np.count_nonzero(region_mask & (dust_mask > 0)))
    return float(dust_overlap / region_area) if region_area else 0.0, region_area


def _peak_in_dust(dust_mask, peak_y, peak_x):
    return bool(dust_mask[peak_y, peak_x] > 0)


def _run_peak_trial(anomaly_map, dust_mask, *, min_distance, threshold_rel=None,
                    threshold_abs=None):
    coords = peak_local_max(
        anomaly_map,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        threshold_abs=threshold_abs,
        exclude_border=False,
    )
    results = []
    for (py, px) in coords:
        score = float(anomaly_map[py, px])
        in_dust = _peak_in_dust(dust_mask, py, px)
        local_cov = _local_dust_coverage(dust_mask, py, px, 11)
        rg_cov, rg_area = _region_grow_coverage(anomaly_map, dust_mask, py, px, 0.5)
        results.append({
            "y": int(py), "x": int(px), "score": score,
            "in_dust": in_dust,
            "local_dust_cov_11x11": local_cov,
            "region_grow_cov": rg_cov, "region_grow_area": rg_area,
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ============================================================
# 視覺化
# ============================================================

def _to_bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def _normalize_to_u8(arr):
    a = np.asarray(arr, dtype=np.float32)
    a = np.maximum(a, 0)
    if a.max() > 0:
        a = a / a.max() * 255
    return a.astype(np.uint8)


def _draw_peaks(base_img, peaks):
    """🟡黃=in_dust  🔴紅=候選真缺陷"""
    out = _to_bgr(base_img.copy())
    for i, p in enumerate(peaks):
        py, px = p["y"], p["x"]
        color = (0, 255, 255) if p["in_dust"] else (0, 0, 255)
        cv2.circle(out, (px, py), 8, color, 2)
        cv2.putText(out, f"{i+1}:{p['score']:.2f}", (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


def _build_visualization(pc_roi_img, omit_crop, anomaly_map, dust_mask, peaks, title,
                          aoi_xy=None, aoi_window=0):
    panel_size = 384
    panels, labels = [], []

    # Panel 1: PC ROI
    if pc_roi_img is not None:
        p1 = cv2.resize(_to_bgr(pc_roi_img), (panel_size, panel_size))
    else:
        p1 = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
        cv2.putText(p1, "no PC ROI", (20, panel_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
    panels.append(p1); labels.append("PC ROI")

    # Panel 2: OMIT crop
    if omit_crop is not None:
        p2 = cv2.resize(_to_bgr(omit_crop), (panel_size, panel_size))
    else:
        p2 = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
        cv2.putText(p2, "no OMIT", (20, panel_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
    panels.append(p2); labels.append("OMIT Crop")

    # AOI window 框 + 中心點繪圖 helper
    def _draw_aoi(img):
        if aoi_xy is None:
            return img
        ax, ay = aoi_xy
        # 中心紫色十字
        cv2.line(img, (ax - 12, ay), (ax + 12, ay), (255, 0, 255), 2)
        cv2.line(img, (ax, ay - 12), (ax, ay + 12), (255, 0, 255), 2)
        # window 框
        if aoi_window > 0:
            x1 = max(0, ax - aoi_window); y1 = max(0, ay - aoi_window)
            x2 = min(img.shape[1] - 1, ax + aoi_window)
            y2 = min(img.shape[0] - 1, ay + aoi_window)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
        return img

    # Panel 3: Heatmap + peaks + AOI 中心
    heat_u8 = _normalize_to_u8(anomaly_map)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    peaks_overlay = _draw_aoi(_draw_peaks(heat_color, peaks))
    panels.append(cv2.resize(peaks_overlay, (panel_size, panel_size)))
    labels.append(f"Heatmap+Peaks ({len(peaks)})")

    # Panel 4: Heatmap + dust + peaks + AOI 中心
    dust_overlay = heat_color.copy()
    dust_bool = dust_mask > 0
    if np.any(dust_bool):
        dust_overlay[dust_bool] = (
            0.5 * dust_overlay[dust_bool].astype(np.float32) +
            0.5 * np.array([255, 200, 0], dtype=np.float32)
        ).astype(np.uint8)
    panels.append(cv2.resize(_draw_aoi(_draw_peaks(dust_overlay, peaks)),
                              (panel_size, panel_size)))
    labels.append("Heatmap+Dust+Peaks")

    gap = np.full((panel_size, 8, 3), 80, dtype=np.uint8)
    composite = panels[0]
    for p in panels[1:]:
        composite = np.hstack([composite, gap, p])

    header_h, label_h = 40, 28
    header = np.zeros((header_h, composite.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    label_bar = np.zeros((label_h, composite.shape[1], 3), dtype=np.uint8)
    for i, lbl in enumerate(labels):
        x = i * (panel_size + 8) + 10
        cv2.putText(label_bar, lbl, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)
    return np.vstack([header, composite, label_bar])


# ============================================================
# 找圖 / 找 OMIT
# ============================================================

def _find_image(panel_dir: Path, prefix_or_filename: str) -> Optional[Path]:
    """支援『完整檔名』、『含副檔名』、『prefix only』"""
    p = Path(prefix_or_filename)
    direct = panel_dir / p
    if direct.exists():
        return direct
    # 用 prefix 搜
    matches = []
    for ext in ("*.tif", "*.tiff", "*.bmp", "*.png", "*.jpg"):
        for f in panel_dir.glob(ext):
            if f.name.startswith("PINIGBI") or f.name.startswith("OMIT0000"):
                continue
            if f.name.startswith(prefix_or_filename):
                matches.append(f)
    if not matches:
        return None
    # 優先非 S 開頭（S 開頭通常是縮圖）
    non_s = [f for f in matches if not f.name.startswith("S")]
    return (non_s or matches)[0]


def _find_omit(panel_dir: Path) -> Optional[Path]:
    for pattern in ("PINIGBI*.*", "OMIT0000*.*"):
        for f in panel_dir.glob(pattern):
            if f.name.startswith("S"):  # 跳過 S 開頭
                continue
            return f
    return None


# ============================================================
# 分析單一 EdgeDefect
# ============================================================

def analyze_edge_defect(edge_defect, image_name, px, py, img_x, img_y,
                         aoi_window, output_dir):
    side = getattr(edge_defect, 'side', 'unknown')
    bbox = getattr(edge_defect, 'bbox', (0, 0, 0, 0))
    is_dust_filtered = bool(getattr(edge_defect, 'is_suspected_dust_or_scratch', False))
    detail_text = getattr(edge_defect, 'dust_detail_text', '')
    pc_score = getattr(edge_defect, 'patchcore_score', 0.0)
    src = getattr(edge_defect, 'source_inspector', '')

    # AOI 影像座標 → ROI 內部座標
    rx1, ry1 = int(bbox[0]), int(bbox[1])
    aoi_local_x = int(img_x) - rx1
    aoi_local_y = int(img_y) - ry1

    anomaly_map = getattr(edge_defect, 'pc_anomaly_map', None)
    dust_mask = getattr(edge_defect, 'dust_mask', None)
    omit_crop = getattr(edge_defect, 'omit_crop_image', None)
    pc_roi_img = getattr(edge_defect, 'pc_roi', None)

    print()
    print("=" * 78)
    print(f"📍 EdgeDefect | image={image_name} | AOI 產品=({px},{py}) "
          f"→ 影像=({img_x},{img_y}) → ROI 內部=({aoi_local_x},{aoi_local_y})")
    print(f"   side={side} | source={src or '<n/a>'}")
    print("-" * 78)
    print(f"   bbox: {bbox}")
    print(f"   PC score: {pc_score:.4f}")
    print(f"   既有判定: {'DUST_FILTERED' if is_dust_filtered else 'REAL_NG'}")
    print(f"   detail: {detail_text}")

    if src != "patchcore":
        print(f"   ⚠️ source_inspector={src!r}，非 patchcore 路徑，沒有 anomaly map")
        return False

    if anomaly_map is None:
        print("   ⚠️ pc_anomaly_map 為 None，跳過")
        return False
    if dust_mask is None:
        print("   ⚠️ dust_mask 為 None — 沒進 dust filter（可能 OMIT 缺失或過曝）")
        return False

    if dust_mask.ndim == 3:
        dust_mask = cv2.cvtColor(dust_mask, cv2.COLOR_BGR2GRAY)
    if dust_mask.shape[:2] != anomaly_map.shape[:2]:
        print(f"   ⚠️ resize dust_mask {dust_mask.shape} → {anomaly_map.shape}")
        dust_mask = cv2.resize(dust_mask,
                               (anomaly_map.shape[1], anomaly_map.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    h, w = anomaly_map.shape[:2]
    a_max = float(anomaly_map.max())
    a_p99 = float(np.percentile(anomaly_map, 99))
    a_p99_8 = float(np.percentile(anomaly_map, 99.8))
    dust_total = int(np.count_nonzero(dust_mask > 0))

    print(f"   anomaly_map shape={anomaly_map.shape} max={a_max:.4f} "
          f"p99={a_p99:.4f} p99.8(Top0.2%切點)={a_p99_8:.4f}")
    print(f"   dust_mask 總覆蓋 {dust_total/(h*w)*100:.1f}% ({dust_total}/{h*w} px)")

    region_details = getattr(edge_defect, 'dust_region_details', []) or []
    print(f"   既有 per-region: {len(region_details)} 個 region")
    for ri, r in enumerate(region_details):
        print(f"      R{ri}: area={r['area']} cov={r['coverage']:.3f} "
              f"max_score={r['max_score']:.3f} is_dust={r['is_dust']} "
              f"peak_in_dust={r.get('peak_in_dust', '?')}")

    # 跑 3 組全圖 + 1 組 AOI window peak 試驗
    print()
    print("   🔍 Peak Local Max 試驗 (全圖):")
    trials = [
        ("Trial1: rel>=0.3 dist=10",
         dict(min_distance=10, threshold_rel=0.3)),
        ("Trial2: rel>=0.5 dist=10",
         dict(min_distance=10, threshold_rel=0.5)),
        ("Trial3: abs>=0.25 dist=15",
         dict(min_distance=15, threshold_abs=0.25)),
    ]

    def _print_peaks(peaks, max_lines=None):
        for i, p in enumerate(peaks):
            if max_lines is not None and i >= max_lines:
                print(f"        ... (省略 {len(peaks) - max_lines} 個更多 peak)")
                break
            tag = "🟡DUST" if p["in_dust"] else "🔴REAL?"
            print(f"        Peak {i+1} {tag} @ ({p['x']:3d},{p['y']:3d}) "
                  f"score={p['score']:.4f} | "
                  f"local(11x11)={p['local_dust_cov_11x11']:.3f} | "
                  f"region_grow(50%)={p['region_grow_cov']:.3f} "
                  f"(area={p['region_grow_area']}px)")

    trial_results = []
    for label, params in trials:
        peaks = _run_peak_trial(anomaly_map, dust_mask, **params)
        trial_results.append((label, peaks))
        n_in_dust = sum(1 for p in peaks if p["in_dust"])
        print(f"   ── {label}")
        print(f"      → 找到 {len(peaks)} 個 peak "
              f"({n_in_dust} 在 dust 上, {len(peaks)-n_in_dust} 候選真缺陷)")
        _print_peaks(peaks, max_lines=5)  # 全圖只列前 5 個

    # === 關鍵：AOI window 內 peak detection ===
    aoi_peaks_all = []
    aoi_peaks_real = []
    if aoi_window > 0:
        print()
        print(f"   🎯 AOI 座標 ROI=({aoi_local_x},{aoi_local_y}) "
              f"±{aoi_window}px 範圍內 peak detection:")

        # 在 anomaly_map 上把 window 外的部分歸零，再做 peak detection
        h, w = anomaly_map.shape[:2]
        y1 = max(0, aoi_local_y - aoi_window)
        y2 = min(h, aoi_local_y + aoi_window + 1)
        x1 = max(0, aoi_local_x - aoi_window)
        x2 = min(w, aoi_local_x + aoi_window + 1)

        masked_amap = np.zeros_like(anomaly_map, dtype=np.float32)
        masked_amap[y1:y2, x1:x2] = anomaly_map[y1:y2, x1:x2]

        aoi_peaks_all = _run_peak_trial(
            masked_amap, dust_mask,
            min_distance=10, threshold_rel=0.5,
        )
        aoi_peaks_real = [p for p in aoi_peaks_all if not p["in_dust"]]

        n_in_dust = len(aoi_peaks_all) - len(aoi_peaks_real)
        print(f"      window=({x1},{y1})~({x2-1},{y2-1}) "
              f"→ {len(aoi_peaks_all)} 個 peak "
              f"({n_in_dust} 在 dust, {len(aoi_peaks_real)} 候選真缺陷)")
        _print_peaks(aoi_peaks_all, max_lines=20)

        # 結論
        print()
        if aoi_peaks_real:
            best = aoi_peaks_real[0]  # score 排序最大
            dist = ((best['x'] - aoi_local_x) ** 2 + (best['y'] - aoi_local_y) ** 2) ** 0.5
            print(f"   ✅ AOI 結論: window 內找到 {len(aoi_peaks_real)} 個非 dust peak")
            print(f"      最強候選: ({best['x']},{best['y']}) score={best['score']:.4f} "
                  f"距離 AOI 點 {dist:.1f}px → **判 NG**（方法 6 + AOI window 救回）")
        else:
            print(f"   ❌ AOI 結論: window 內無非 dust peak → 維持 DUST 判定")
            if aoi_peaks_all:
                strongest = aoi_peaks_all[0]
                print(f"      window 內最強 peak: ({strongest['x']},{strongest['y']}) "
                      f"score={strongest['score']:.4f} 但在 dust mask 上")
            else:
                print(f"      window 內完全沒 peak (anomaly 訊號不夠強或被 mask 為 0)")

    # 視覺化用「AOI window 內 peak」(若沒開 window 退回 Trial1)
    output_dir.mkdir(parents=True, exist_ok=True)
    if aoi_window > 0 and aoi_peaks_all:
        viz_peaks = aoi_peaks_all
        viz_label = f"AOI window peaks ({len(aoi_peaks_real)} real)"
    else:
        viz_peaks = trial_results[0][1]
        viz_label = "Trial1 peaks"

    title = (f"{image_name} AOI=({px},{py}) ROI=({aoi_local_x},{aoi_local_y}) | "
             f"既有={'DUST' if is_dust_filtered else 'NG'} | "
             f"window={aoi_window} | {viz_label}")
    vis = _build_visualization(
        pc_roi_img, omit_crop, anomaly_map, dust_mask,
        viz_peaks, title,
        aoi_xy=(aoi_local_x, aoi_local_y),
        aoi_window=aoi_window,
    )
    base = f"{Path(image_name).stem}_aoi_{px}_{py}"
    vis_path = output_dir / f"{base}_peaks.png"
    cv2.imwrite(str(vis_path), vis)
    print(f"   📊 視覺化: {vis_path}")

    np.save(output_dir / f"{base}_anomaly.npy", anomaly_map)
    np.save(output_dir / f"{base}_dust_mask.npy", dust_mask)
    print(f"   💾 anomaly/dust mask npy: {output_dir}")
    return True


# ============================================================
# 主流程：直接注入座標觸發 fusion
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="對指定產品座標跑 AOI Edge fusion → peak detection 驗證",
    )
    ap.add_argument("panel_dir", type=str, help="panel 資料夾")
    ap.add_argument("--image", required=True,
                    help="目標圖片 prefix 或檔名 (例: WGF50500 或 WGF50500_205045.tif)")
    ap.add_argument("--coord", action="append", required=True,
                    help="產品座標，格式 px,py，可重複 (例: --coord 88,1153)")
    ap.add_argument("--config", default="configs/capi_3f.yaml",
                    help="inference yaml")
    ap.add_argument("--server-config", default="server_config.yaml",
                    help="server_config.yaml — 用來找 DB 路徑套 config overrides")
    ap.add_argument("--db-path", default=None,
                    help="直接指定 SQLite DB 路徑 (覆寫 server-config 推導)")
    ap.add_argument("--no-db", action="store_true",
                    help="不套用 DB overrides (只用 yaml)")
    ap.add_argument("--model-id", default=None,
                    help="機種 ID (例 GN160JCEL250S)；未提供時從 panel_dir 路徑自動推導")
    ap.add_argument("--product-res", default=None,
                    help="強制覆寫產品解析度，格式 W,H (例 1920,1200)")
    ap.add_argument("--inspector-mode", default=None, choices=[None, "cv", "patchcore", "fusion"],
                    help="強制覆寫 aoi_edge_inspector 模式")
    ap.add_argument("--aoi-window", type=int, default=50,
                    help="AOI 座標附近搜尋半徑 (px)，0 = 全圖搜尋 (預設 50)")
    ap.add_argument("--out", default="peak_validation_out")
    args = ap.parse_args()

    panel_dir = Path(args.panel_dir)
    if not panel_dir.is_dir():
        print(f"❌ 找不到資料夾: {panel_dir}")
        sys.exit(1)

    # 解析座標
    coords: List[Tuple[int, int]] = []
    for c in args.coord:
        try:
            px, py = c.split(",")
            coords.append((int(px), int(py)))
        except ValueError:
            print(f"❌ --coord 格式錯誤: {c} (應為 px,py)")
            sys.exit(1)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"📂 Panel: {panel_dir}")
    print(f"🖼️  Image: {args.image}")
    print(f"📍 Coords (產品座標): {coords}")
    print(f"⚙️  Config: {args.config}")
    print(f"📤 Output: {output_dir}")
    print("=" * 78)

    # 找圖
    img_path = _find_image(panel_dir, args.image)
    if img_path is None:
        print(f"❌ 找不到對應圖片: {args.image}")
        sys.exit(1)
    print(f"✅ 圖片: {img_path.name}")

    omit_path = _find_omit(panel_dir)
    if omit_path:
        print(f"✅ OMIT: {omit_path.name}")
    else:
        print("⚠️ 找不到 OMIT，灰塵屏蔽會被跳過 → 沒法看 peak vs dust 比對")

    # 載入 config
    config = CAPIConfig.from_yaml(args.config)
    print(f"✅ Config: {config.machine_id}")

    # === DB overrides (生產環境設定主要存在 DB 不在 yaml) ===
    db_dict_for_edge = None  # 給 EdgeInspectionConfig.from_db_params 用
    if not args.no_db:
        db_path = args.db_path
        if db_path is None:
            try:
                import yaml
                with open(args.server_config, "r", encoding="utf-8") as f:
                    sc = yaml.safe_load(f) or {}
                db_path = (sc.get("database") or {}).get("path")
            except Exception as e:
                print(f"⚠️ 讀 {args.server_config} 失敗: {e}")
        if db_path and Path(db_path).exists():
            try:
                from capi_database import CAPIDatabase
                db = CAPIDatabase(db_path)
                db_params = db.get_all_config_params()
                if db_params:
                    config.apply_db_overrides(db_params)
                    db_dict_for_edge = {p["param_name"]: p for p in db_params}
                    print(f"✅ 套用 {len(db_params)} 筆 DB config overrides "
                          f"(from {db_path})")
            except Exception as e:
                print(f"⚠️ DB overrides 載入失敗: {e}")
        else:
            print(f"⚠️ DB 路徑不存在 ({db_path!r})，跳過 DB overrides")

    print("🔄 初始化 inferencer (含模型載入)...")
    t0 = time.time()
    inferencer = CAPIInferencer(config)
    print(f"✅ Inferencer ready ({time.time()-t0:.1f}s)")

    # === 套 EdgeInspectionConfig (from DB)，跟 server 流程一致 ===
    if db_dict_for_edge is not None:
        try:
            from capi_edge_cv import EdgeInspectionConfig
            edge_cfg = EdgeInspectionConfig.from_db_params(db_dict_for_edge)
            inferencer.update_edge_config(edge_cfg)
            print(f"✅ 套用 EdgeInspectionConfig from DB "
                  f"(aoi_edge_inspector={edge_cfg.aoi_edge_inspector!r})")
        except Exception as e:
            print(f"⚠️ EdgeInspectionConfig 套用失敗: {e}")

    # === Inspector mode CLI 強制覆寫 (CLI > DB > yaml) ===
    if args.inspector_mode and getattr(inferencer, 'edge_inspector', None):
        inferencer.edge_inspector.config.aoi_edge_inspector = args.inspector_mode
        print(f"   ▶ 強制 inspector_mode={args.inspector_mode}")

    # 預處理（取 raw_bounds / panel_polygon / otsu_bounds）
    print("🔄 preprocess_image...")
    pp = inferencer.preprocess_image(img_path)
    if pp is None:
        print("❌ 預處理失敗")
        sys.exit(1)
    print(f"   raw_bounds={pp.raw_bounds} otsu={pp.otsu_bounds} "
          f"polygon={'有' if pp.panel_polygon is not None else '無'}")

    # 載 OMIT
    omit_image = None
    omit_overexposed = False
    if omit_path:
        omit_image = cv2.imread(str(omit_path), cv2.IMREAD_UNCHANGED)
        if omit_image is not None:
            is_oe, _, _, oe_detail = inferencer.check_omit_overexposure(omit_image)
            if is_oe:
                print(f"⚠️ OMIT 過曝: {oe_detail} → dust 檢測停用")
                omit_image = None
                omit_overexposed = True

    # 載原圖
    full_image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if full_image is None:
        print(f"❌ 無法讀取 {img_path}")
        sys.exit(1)

    img_prefix = inferencer._get_image_prefix(img_path.name)

    # === Resolve product_resolution ===
    # 優先序：CLI --product-res > --model-id 推導 > panel_dir 路徑推導 > DEFAULT
    product_resolution = None
    if args.product_res:
        try:
            w, h = args.product_res.split(",")
            product_resolution = (int(w), int(h))
            print(f"   ▶ CLI 強制 product_resolution={product_resolution}")
        except ValueError:
            print(f"⚠️ --product-res 格式錯誤: {args.product_res}")

    if product_resolution is None:
        model_id = args.model_id
        if model_id is None:
            # 從 panel_dir 推導 — 慣例 .../yuantu/<MODEL_ID>/<DATE>/<GLASS>
            try:
                model_id = panel_dir.parent.parent.name
                print(f"   ⓘ 從 panel_dir 路徑推導 model_id={model_id!r}")
            except Exception:
                pass
        if model_id:
            res_map = getattr(config, 'model_resolution_map', None) or MODEL_RESOLUTION_MAP
            product_resolution = resolve_product_resolution(model_id, res_map)
            code = model_id[5].upper() if len(model_id) >= 6 else "?"
            print(f"   ⓘ resolve_product_resolution({model_id!r}) → "
                  f"第6碼 '{code}' → {product_resolution}")

    if product_resolution is None:
        product_resolution = DEFAULT_PRODUCT_RESOLUTION
        print(f"   ⓘ 用預設 product_resolution={product_resolution}")

    inspector_mode = getattr(inferencer.edge_inspector.config,
                             "aoi_edge_inspector", "cv") \
        if getattr(inferencer, 'edge_inspector', None) else "cv"
    print(f"   prefix={img_prefix}  inspector_mode={inspector_mode}  "
          f"product_resolution={product_resolution}")

    if inspector_mode != "fusion":
        print(f"⚠️ aoi_edge_inspector={inspector_mode!r}，"
              "本腳本針對 fusion 模式設計 — 仍會嘗試但 dust filter 路徑可能不同")

    # 對每個座標跑 fusion
    success_count = 0
    for (px, py) in coords:
        img_x, img_y = inferencer._map_aoi_coords(
            px, py, pp.raw_bounds, product_resolution
        )
        print()
        print("█" * 78)
        print(f"█ AOI 產品座標 ({px}, {py}) → 影像座標 ({img_x}, {img_y})")
        print("█" * 78)

        try:
            fusion_defects, fusion_stats = inferencer._inspect_roi_fusion(
                full_image, img_x, img_y, img_prefix,
                panel_polygon=pp.panel_polygon,
                omit_image=omit_image,
                omit_overexposed=omit_overexposed,
                otsu_bounds=pp.otsu_bounds,
                collapse_to_representative=False,
                group_cv_band=True,
            )
        except Exception as e:
            print(f"❌ _inspect_roi_fusion 失敗: {e}")
            import traceback; traceback.print_exc()
            continue

        print(f"📊 fusion 回傳 {len(fusion_defects)} 個 defect "
              f"(fallback={fusion_stats.get('fusion_fallback_reason', '')!r})")

        # 找有 patchcore anomaly map 的 defect
        analyzed = 0
        for d in fusion_defects:
            ok = analyze_edge_defect(
                d, img_path.name, px, py, img_x, img_y,
                args.aoi_window, output_dir,
            )
            if ok:
                analyzed += 1
                success_count += 1

        if analyzed == 0:
            print(f"   ⚠️ 此座標 ({px},{py}) 無 patchcore-source defect 可分析")
            print(f"      → 可能 fusion 把它分到 cv source；可看 fusion_defects "
                  f"檢查每個的 source_inspector")
            for i, d in enumerate(fusion_defects):
                print(f"      defect {i}: source={getattr(d, 'source_inspector', '?')!r} "
                      f"side={getattr(d, 'side', '?')} "
                      f"is_dust={getattr(d, 'is_suspected_dust_or_scratch', False)}")

    # 結尾
    print()
    print("=" * 78)
    print(f"✅ 共分析 {success_count} 個 patchcore edge defect")
    if success_count == 0:
        print("⚠️ 沒有可分析的 case。可能原因：")
        print("   1. 座標不在 panel polygon 邊緣 (沒走 fusion 路徑)")
        print("   2. fusion 因 polygon 偵測失敗走 fallback (CV only)")
        print("   3. inspector_mode 非 'fusion'")
    else:
        print(f"📁 視覺化在: {output_dir}")
        print()
        print("📖 解讀:")
        print("   🔴 紅色 peak (in_dust=False) = 候選真缺陷")
        print("   🟡 黃色 peak (in_dust=True)  = 灰塵 / MARK 等已知強訊號")
        print()
        print("   ➤ 找到 ≥1 紅色 peak 對應目視缺陷 → 方法 6 可救漏放")
        print("   ➤ 全黃 → 缺陷 peak 被 dust mask 蓋住，問題在 OMIT 過敏感")


if __name__ == "__main__":
    main()
