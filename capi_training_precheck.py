"""Mandatory, whole-panel image screening before training tiles are created.

The measurement helpers and limits are shared with the production image-abnormal
precheck. Training checks every supported source image (including retakes and
black screens), independently of the inference switch and AOI report.
"""
from pathlib import Path

import cv2

from capi_image_orientation import read_detection_image
from capi_server import (
    _detect_image_abnormal_product_polygon,
    _image_abnormal_limits,
    _image_abnormal_preprocess_config,
    _image_abnormal_reference_polygon,
    _image_abnormal_screen_path_map,
    _image_mean_brightness,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class TrainingPanelPrecheckError(RuntimeError):
    def __init__(self, rejected_panels):
        self.rejected_panels = rejected_panels
        names = "、".join(panel["panel_id"] for panel in rejected_panels)
        super().__init__(f"PANEL 影像檢查未通過：{names}；整片排除，請重選新的 Panel ID。")


def _panel_files(panel_dir):
    return sorted(
        (p for p in panel_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )


def _file_snapshot(files):
    records = []
    for path in files:
        stat = path.stat()
        records.append({"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return records


def check_training_panels(panel_paths, config, station_adapter, *,
                          product_resolution=None, rotate_180=False, log=lambda _: None):
    """Return a checked source snapshot, or reject the selection before any slicing.

    Missing/unreadable sources fail closed. A rejected panel never contributes
    even its otherwise healthy lighting images. No source files are modified.
    """
    limits = _image_abnormal_limits(config)
    for screen, (lower, upper) in limits.items():
        if not 0 <= lower <= upper <= 255:
            raise ValueError(f"畫異設定 {screen} 亮度範圍無效：{lower}–{upper}")
    pre_cfg = _image_abnormal_preprocess_config(
        config, product_resolution, rotate_180, station_adapter.profile,
    )
    panels = []
    for raw_path in panel_paths:
        panel_dir = Path(raw_path).resolve()
        panel = {"panel_path": str(panel_dir), "panel_id": panel_dir.name,
                 "passed": False, "images": [], "issues": []}
        panels.append(panel)
        log(f"畫異檢查 PANEL {panel_dir.name}")
        try:
            files = _panel_files(panel_dir)
            before = _file_snapshot(files)
            targets = []
            for path in files:
                source = station_adapter.image_prefix(path.name)
                screen = station_adapter.model_prefix(source)
                if screen in limits:
                    targets.append((path, screen))
            if not targets:
                raise ValueError("找不到可依畫異設定檢查的來源圖片")
            # Use the same reference priority and polygon/bbox fallback as inference.
            cache = {}
            screen_paths = _image_abnormal_screen_path_map(files, station_adapter.image_prefix)
            polygon, source, shape = _image_abnormal_reference_polygon(
                screen_paths, pre_cfg, cache, station_adapter.boundary_reference_priority,
            )
            cache.clear()  # Do not retain full-resolution panels while reading all screens.
            for path, screen in targets:
                lower, upper = limits[screen]
                measurement = {"screen": screen, "image_name": path.name,
                               "image_path": str(path), "lower": lower, "upper": upper}
                panel["images"].append(measurement)
                image = read_detection_image(path, cv2.IMREAD_UNCHANGED, rotate_180)
                if image is None or image.size == 0:
                    measurement.update(passed=False, reason="圖片無法讀取，無法完成畫異檢查")
                else:
                    image_polygon, mean_source = polygon, source
                    if image_polygon is None or shape != image.shape[:2]:
                        image_polygon, mean_source = _detect_image_abnormal_product_polygon(image, pre_cfg)
                    mean, mean_source, pixels = _image_mean_brightness(image, image_polygon, mean_source)
                    passed = lower <= mean <= upper
                    measurement.update(mean_brightness=mean, mean_source=mean_source,
                                       mean_pixels=pixels, passed=passed)
                    measurement["reason"] = (
                        f"產品區域平均亮度 {mean:.3f}，允許範圍 {lower}–{upper}"
                    )
                    del image
                if not measurement["passed"]:
                    panel["issues"].append(dict(measurement))
            if before != _file_snapshot(_panel_files(panel_dir)):
                raise ValueError("檢查期間來源圖片已變更，請重選新的 Panel ID")
            panel["files"] = before
            panel["passed"] = not panel["issues"]
        except (OSError, ValueError, cv2.error) as exc:
            panel["issues"].append({"reason": str(exc)})
        log(f"  {'✓ 通過' if panel['passed'] else '✗ 整片排除'}：{panel_dir.name}")
    rejected = [panel for panel in panels if not panel["passed"]]
    if rejected:
        raise TrainingPanelPrecheckError(rejected)
    return {"panels": panels, "limits": {k: list(v) for k, v in limits.items()}}


def verify_training_panel_sources(precheck, panel_paths):
    """Validate the complete selection before writing any tiles, then pin its files."""
    panels = {panel["panel_path"]: panel for panel in precheck["panels"]}
    if set(panels) != {str(Path(p).resolve()) for p in panel_paths}:
        raise ValueError("PANEL 選擇與畫異檢查不一致，請重新選圖並檢查")
    pinned = {}
    rejected = []
    for path, panel in panels.items():
        try:
            unchanged = panel["passed"] and panel["files"] == _file_snapshot(_panel_files(Path(path)))
        except OSError:
            unchanged = False
        if not unchanged:
            rejected.append({"panel_id": panel["panel_id"], "panel_path": path,
                             "issues": [{"reason": "畫異檢查後來源圖片已變更或無法讀取"}]})
        else:
            pinned[path] = [Path(path) / record["name"] for record in panel["files"]]
    if rejected:
        raise TrainingPanelPrecheckError(rejected)
    return pinned
