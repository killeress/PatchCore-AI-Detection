"""Debug-only white-mura enhancement and a paired PatchCore experiment.

The source image, production configuration and saved judgments are never changed.
DoG is evaluated on surrounding pixels before cropping a model-sized tile.
"""
from __future__ import annotations

from contextlib import nullcontext
import logging
import math
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

logger = logging.getLogger("capi.white_spot_debug")

DEFAULT_PARAMS = {
    "signal_sigma_x": 20.0, "signal_sigma_y": 6.0,
    "background_sigma_x": 96.0, "background_sigma_y": 32.0,
    "gain": 40.0, "offset": 128.0,
}
PRESETS = {
    "recommended": {"label": "白丸：背景抑制", "params": dict(DEFAULT_PARAMS)},
    "balanced": {"label": "白丸：保留對比", "params": {**DEFAULT_PARAMS, "signal_sigma_x": 16.0}},
    "isotropic": {"label": "一般 DoG 8 / 64", "params": {
        **DEFAULT_PARAMS, "signal_sigma_x": 8.0, "signal_sigma_y": 8.0,
        "background_sigma_x": 64.0, "background_sigma_y": 64.0,
    }},
    "baseline": {"label": "原先 DoG 4 / 48", "params": {
        **DEFAULT_PARAMS, "signal_sigma_x": 4.0, "signal_sigma_y": 4.0,
        "background_sigma_x": 48.0, "background_sigma_y": 48.0,
    }},
}


def _number(value, name, minimum, maximum, *, integer=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} 必須是數值")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必須是數值") from None
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} 必須介於 {minimum}～{maximum}")
    if integer and not number.is_integer():
        raise ValueError(f"{name} 必須是整數")
    return int(number) if integer else number


def normalize_params(params=None):
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("params 必須是物件")
    if set(params) - set(DEFAULT_PARAMS):
        raise ValueError("包含不支援的 DoG 參數")
    limits = {
        "signal_sigma_x": (.5, 64), "signal_sigma_y": (.5, 64),
        "background_sigma_x": (1, 192), "background_sigma_y": (1, 192),
        "gain": (1, 100), "offset": (0, 255),
    }
    result = {k: _number(params.get(k, v), k, *limits[k]) for k, v in DEFAULT_PARAMS.items()}
    for axis in ("x", "y"):
        if result[f"background_sigma_{axis}"] <= result[f"signal_sigma_{axis}"]:
            raise ValueError(f"背景 sigma {axis.upper()} 必須大於平滑 sigma {axis.upper()}")
    return result


def prepare_white_spot(gray, center=None, tile_size=512, params=None):
    """Filter padded context, then return aligned raw/enhanced tiles and geometry."""
    params = normalize_params(params)
    if gray is None or gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("白丸檢測需要 8-bit 灰階影像")
    tile_size = _number(tile_size, "模型切塊大小", 32, 1024, integer=True)
    h, w = gray.shape
    if min(h, w) < tile_size:
        raise ValueError(f"影像寬高須至少 {tile_size} px，才能建立模型切塊")
    if center is None:
        cx, cy = w // 2, h // 2
    else:
        cx = _number(center[0], "中心 X", 0, w - 1, integer=True)
        cy = _number(center[1], "中心 Y", 0, h - 1, integer=True)
    x1 = min(max(cx - tile_size // 2, 0), w - tile_size)
    y1 = min(max(cy - tile_size // 2, 0), h - tile_size)
    x2, y2 = x1 + tile_size, y1 + tile_size
    pad_x = math.ceil(4 * params["background_sigma_x"])
    pad_y = math.ceil(4 * params["background_sigma_y"])
    bx1, by1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    bx2, by2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    work = gray[by1:by2, bx1:bx2].astype(np.float32)
    start = time.perf_counter()
    signal = cv2.GaussianBlur(work, (0, 0), sigmaX=params["signal_sigma_x"],
                              sigmaY=params["signal_sigma_y"], borderType=cv2.BORDER_REFLECT_101)
    background = cv2.GaussianBlur(work, (0, 0), sigmaX=params["background_sigma_x"],
                                  sigmaY=params["background_sigma_y"], borderType=cv2.BORDER_REFLECT_101)
    residual = (signal - background)[y1-by1:y2-by1, x1-bx1:x2-bx1].copy()
    enhanced = np.rint(np.clip(params["offset"] + params["gain"] * residual, 0, 255)).astype(np.uint8)
    return {
        "original": gray[y1:y2, x1:x2].copy(), "enhanced": enhanced, "residual": residual,
        "params": params, "center": [cx, cy], "center_in_tile": [cx-x1, cy-y1],
        "crop_bounds": [x1, y1, x2, y2], "context_bounds": [bx1, by1, bx2, by2],
        "padding_px": [pad_x, pad_y], "elapsed_ms": (time.perf_counter()-start)*1000,
        "clipped_fraction": float(np.mean((enhanced == 0) | (enhanced == 255))),
    }


def summarize_prediction(score, anomaly_map, tile_shape, center, radius, threshold):
    """Use the existing AOI max-ratio convention, with an image-pixel square ROI."""
    score = float(score)
    if not math.isfinite(score) or score < 0:
        raise ValueError("PC 回傳無效分數")
    h, w = tile_shape
    cx, cy = center
    roi = [max(0, cx-radius), max(0, cy-radius), min(w, cx+radius+1), min(h, cy+radius+1)]
    result = {"tile_score": score, "roi_score": None, "roi_bounds": roi,
              "roi_passes_threshold": None, "threshold": threshold, "score_method": "max_ratio"}
    if anomaly_map is None:
        result.update(status="no_heatmap", message="模型未回傳熱圖，無法判斷白丸中心區域")
        return result
    amap = np.asarray(anomaly_map, dtype=np.float32)
    if amap.ndim != 2 or not amap.size or not np.isfinite(amap).all():
        raise ValueError("PC 回傳無效熱力圖")
    mh, mw = amap.shape
    mx1, my1 = int(math.floor(roi[0]*mw/w)), int(math.floor(roi[1]*mh/h))
    mx2, my2 = int(math.ceil(roi[2]*mw/w)), int(math.ceil(roi[3]*mh/h))
    peak = float(amap.max())
    roi_peak = float(amap[my1:my2, mx1:mx2].max())
    equivalent = score * max(0.0, roi_peak) / peak if peak > 0 else 0.0
    result.update(status="ok", roi_score=equivalent, map_max=peak, roi_map_max=roi_peak,
                  roi_passes_threshold=bool(equivalent >= threshold and roi_peak > 0))
    return result


def _resolve_model(inferencer, image_path, zone):
    prefix = inferencer._get_image_prefix(image_path.name)
    config = inferencer.config
    machine_id = getattr(config, "machine_id", "")
    if getattr(config, "grid_canonicalization_enabled", False):
        raise ValueError("目前模型使用 Pixel Grid 座標；白丸原圖座標試跑尚不支援此模型，仍可查看強化圖")
    if getattr(config, "is_new_architecture", False):
        model_prefix = inferencer._get_model_prefix(prefix)
        model = inferencer._get_model_for(machine_id, model_prefix, zone)
        mapping = getattr(config, "threshold_mapping", {}) or {}
        threshold = mapping.get(model_prefix, {})
        if isinstance(threshold, dict):
            threshold = threshold.get(zone)
        name = f"{machine_id} / {model_prefix} / {zone}"
    else:
        model_prefix = prefix
        model = inferencer._get_inferencer_for_prefix(prefix)
        threshold = None
        name = f"{machine_id} / {prefix}"
    if model is None:
        raise ValueError(f"找不到 {name} 的 PC 模型")
    if threshold is None:
        threshold = getattr(inferencer, "threshold", getattr(config, "anomaly_threshold", .5))
    return model, {"name": name, "machine_id": machine_id, "prefix": model_prefix, "zone": zone}, float(threshold)


def compare_patchcore(inferencer, image_path, prepared, radius, zone="inner", threshold=None, gpu_lock=None):
    """Run both inputs through the same selected model under the shared GPU lock."""
    if inferencer is None:
        return {"status": "unavailable", "message": "推論器尚未載入；已保留 DoG 強化結果"}
    try:
        from capi_inference import TileInfo
        with gpu_lock if gpu_lock is not None else nullcontext():
            model, info, default_threshold = _resolve_model(inferencer, image_path, zone)
            threshold = default_threshold if threshold is None else threshold
            threshold = _number(threshold, "PC 門檻", 0, 1e6)
            output = {"status": "ok", "model": info, "threshold": threshold, "variants": {}}
            for key in ("original", "enhanced"):
                try:
                    tile_image = prepared[key]
                    h, w = tile_image.shape
                    tile = TileInfo(tile_id=0, x=prepared["crop_bounds"][0], y=prepared["crop_bounds"][1],
                                    width=w, height=h, image=tile_image.copy(), zone=zone,
                                    original_image=prepared["original"].copy())
                    start = time.perf_counter()
                    score, amap = inferencer.predict_tile(
                        tile, inferencer=model, edge_margin_override=0,
                        patchcore_overrides={"patchcore_filter_enabled": False},
                        threshold=threshold, model_id=info["machine_id"],
                    )
                    result = summarize_prediction(score, amap, (h,w), prepared["center_in_tile"], radius, threshold)
                    result["elapsed_ms"] = (time.perf_counter()-start)*1000
                    result["anomaly_map"] = None if amap is None else np.asarray(amap, dtype=np.float32).copy()
                    output["variants"][key] = result
                except Exception as exc:
                    logger.exception("White spot PC experiment failed: %s", key)
                    output["variants"][key] = {"status": "error", "message": str(exc)}
            if any(r["status"] != "ok" for r in output["variants"].values()):
                output["status"] = "partial"
            return output
    except Exception as exc:
        logger.warning("White spot model unavailable: %s", exc)
        return {"status": "unavailable", "message": f"PC 試跑無法執行：{exc}"}


def run_debug(handler, data, output_dir):
    """Return a JSON-safe preview and optional PatchCore comparison."""
    if not isinstance(data, dict):
        raise ValueError("請提供 JSON 物件")
    path_text = data.get("image_path")
    if not isinstance(path_text, str) or not path_text.strip():
        raise ValueError("請提供圖片檔案路徑")
    path = Path(path_text.strip())
    if not path.is_file():
        raise ValueError(f"圖片檔案不存在：{path}")
    params = normalize_params(data.get("params"))
    run_pc = data.get("run_patchcore", False)
    if not isinstance(run_pc, bool):
        raise ValueError("run_patchcore 必須是布林值")
    zone = data.get("zone", "inner")
    if zone not in ("inner", "edge"):
        raise ValueError("模型區域必須是 inner 或 edge")
    threshold = data.get("threshold")
    if threshold is not None:
        threshold = _number(threshold, "PC 門檻", 0, 1e6)
    center_x, center_y = data.get("center_x"), data.get("center_y")
    if (center_x is None) != (center_y is None):
        raise ValueError("中心 X、Y 必須同時提供，或同時留空")
    center = None if center_x is None else (center_x, center_y)
    inferencer = getattr(handler, "inferencer", None)
    tile_size = getattr(getattr(inferencer, "config", None), "tile_size", 512)
    tile_size = _number(tile_size, "模型切塊大小", 32, 1024, integer=True)
    radius = _number(data.get("radius_px", min(64, tile_size//2)), "中心檢查半徑", 1, tile_size//2, integer=True)
    start = time.perf_counter()
    image = handler._read_inference_image(path, cv2.IMREAD_GRAYSCALE)
    prepared = prepare_white_spot(image, center, tile_size, params)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    token = "white_spot_" + uuid.uuid4().hex

    def save(label, pixels):
        filename = f"{token}_{label}.png"
        # imencode/tofile also handles Unicode output directories on Windows.
        ok, encoded = cv2.imencode(".png", pixels)
        if not ok:
            raise RuntimeError("白丸檢測圖片編碼失敗")
        encoded.tofile(str(output_dir / filename))
        return f"/debug/heatmaps/{filename}"

    h, w = image.shape
    scale = min(1.0, 1600/max(h,w))
    overview = cv2.resize(image, (max(1,round(w*scale)),max(1,round(h*scale))), interpolation=cv2.INTER_AREA)
    overview = cv2.cvtColor(overview, cv2.COLOR_GRAY2BGR)
    x1,y1,x2,y2 = prepared["crop_bounds"]
    cv2.rectangle(overview, (round(x1*scale),round(y1*scale)), (round((x2-1)*scale),round((y2-1)*scale)), (80,220,255), 2)
    cv2.drawMarker(overview, tuple(round(c*scale) for c in prepared["center"]), (100,255,100), cv2.MARKER_CROSS, 13, 2)
    urls = {"overview": save("overview",overview), "original": save("original",prepared["original"]),
            "enhanced": save("enhanced",prepared["enhanced"])}
    cx,cy = prepared["center_in_tile"]
    focus = [max(0,cx-radius),max(0,cy-radius),min(tile_size,cx+radius+1),min(tile_size,cy+radius+1)]
    pc = compare_patchcore(inferencer, path, prepared, radius, zone, threshold, getattr(handler,"_gpu_lock",None)) if run_pc else {"status":"not_requested"}
    variants = pc.get("variants", {})
    maps = [v["anomaly_map"] for v in variants.values() if v.get("anomaly_map") is not None]
    # Use a shared scale; independently stretching each map would mislead comparison.
    map_max = max((float(a.max()) for a in maps), default=0)
    pc["heatmap_display_range"] = [0.0, max(0.0,map_max)]
    for key, result in variants.items():
        amap = result.pop("anomaly_map", None)
        if amap is None:
            continue
        resized = cv2.resize(amap, (tile_size,tile_size), interpolation=cv2.INTER_LINEAR)
        normalized = np.clip(resized/max(map_max,1e-12)*255,0,255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        bgr = cv2.cvtColor(prepared[key], cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(bgr,.55,colored,.45,0)
        cv2.rectangle(overlay, tuple(focus[:2]), (focus[2]-1,focus[3]-1), (90,255,90), 1)
        result["heatmap_url"] = save(f"{key}_pc",overlay)
    return {
        "success": True, "image_path": str(path), "image_name": path.name,
        "image_size": [w,h], "rotation_applied": handler._inference_rotate_180_enabled(),
        "center": prepared["center"], "center_in_tile": prepared["center_in_tile"],
        "tile_size": tile_size, "crop_bounds": prepared["crop_bounds"],
        "context_bounds": prepared["context_bounds"], "padding_px": prepared["padding_px"],
        "params": params, "radius_px": radius, "focus_bounds": focus,
        "clipped_fraction": prepared["clipped_fraction"], "urls": urls, "pc": pc,
        "preprocess_ms": prepared["elapsed_ms"], "total_ms": (time.perf_counter()-start)*1000,
    }
