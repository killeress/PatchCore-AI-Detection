import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from capi_edge_cv import EdgeDefect
from capi_heatmap import HeatmapManager


def _capture_put_text(monkeypatch):
    captured = []
    real_put_text = cv2.putText

    def capture(img, text, *args, **kwargs):
        captured.append(str(text))
        return real_put_text(img, text, *args, **kwargs)

    monkeypatch.setattr("capi_heatmap.cv2.putText", capture)
    return captured


def test_tile_header_prefers_two_stage_reason_over_region_text(tmp_path, monkeypatch):
    captured = _capture_put_text(monkeypatch)

    tile_img = np.full((512, 512, 3), 128, dtype=np.uint8)
    anomaly_map = np.zeros((512, 512), dtype=np.float32)
    anomaly_map[200:240, 200:240] = 1.0
    tile = SimpleNamespace(
        is_bright_spot_detection=False,
        omit_crop_image=None,
        dust_mask=np.zeros((512, 512), dtype=np.uint8),
        dust_heatmap_iou=0.25,
        dust_detail_text="PER_REGION: 0real+2dust -> TWO_STAGE: 0real+2dust -> DUST",
        is_suspected_dust_or_scratch=True,
        is_bomb=False,
        bomb_defect_code="",
        dust_iou_debug_image=None,
        dust_region_max_cov=0.25,
        dust_region_details=None,
        dust_heatmap_binary=(anomaly_map > 0).astype(np.uint8) * 255,
        dust_two_stage_features=None,
        height=512,
        width=512,
        scratch_score=0.0,
        scratch_filtered=False,
    )

    saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
    saver.save_tile_heatmap(
        save_dir=tmp_path,
        image_name="tile_ts",
        tile_id=1,
        tile_image=tile_img,
        anomaly_map=anomaly_map,
        score=0.9,
        tile_info=tile,
        score_threshold=0.5,
        iou_threshold=0.1,
        dust_metric="coverage",
    )

    all_text = " ".join(captured)
    assert "TWO_STAGE -> DUST" in all_text
    assert "RegionCOV" not in all_text


def test_cv_edge_header_uses_inference_dust_flag_not_render_overlap(tmp_path, monkeypatch):
    captured = _capture_put_text(monkeypatch)

    full = np.full((220, 220, 3), 128, dtype=np.uint8)
    omit = np.full((220, 220), 128, dtype=np.uint8)
    cv_mask = np.full((20, 20), 255, dtype=np.uint8)
    edge = EdgeDefect(
        side="left",
        area=400,
        bbox=(80, 80, 20, 20),
        center=(90, 90),
        max_diff=10,
        is_suspected_dust_or_scratch=False,
        cv_filtered_mask=cv_mask,
        cv_mask_offset=(80, 80),
    )

    def dust_check(img):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[80:100, 80:100] = 255
        return True, mask, 0.1, "stub dust"

    saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
    saver.save_edge_defect_image(
        save_dir=tmp_path,
        image_name="edge_flag",
        edge_index=0,
        edge_defect=edge,
        full_image=full,
        omit_image=omit,
        dust_check_fn=dust_check,
        dust_iou_threshold=0.5,
        dust_metric="coverage",
    )

    all_text = " ".join(captured)
    assert "DustFlag=False" in all_text
    assert "SURFACE" not in all_text


def test_cv_fusion_header_uses_iou_union_denominator(tmp_path, monkeypatch):
    captured = _capture_put_text(monkeypatch)

    full = np.full((300, 300, 3), 128, dtype=np.uint8)
    omit = np.full((300, 300), 128, dtype=np.uint8)
    cv_mask = np.zeros((220, 220), dtype=np.uint8)
    cv_mask[50:70, 50:70] = 255
    edge = EdgeDefect(
        side="aoi_edge",
        area=400,
        bbox=(100, 100, 20, 20),
        center=(110, 110),
        max_diff=10,
        inspector_mode="fusion",
        source_inspector="cv",
        cv_filtered_mask=cv_mask,
        cv_mask_offset=(0, 0),
    )

    def dust_check(img):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[60:82, 49:71] = 255
        return True, mask, 0.1, "stub dust"

    saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
    saver._save_cv_fusion_edge_image(
        save_dir=tmp_path,
        image_name="fusion_iou",
        edge_index=0,
        edge_defect=edge,
        full_image=full,
        omit_image=omit,
        dust_check_fn=dust_check,
        dust_metric="iou",
    )

    all_text = " ".join(captured)
    assert "IOU=0.33" in all_text
    assert "IOU=0.50" not in all_text


def test_patchcore_edge_dust_flag_header_is_filtered_ok(tmp_path, monkeypatch):
    captured = _capture_put_text(monkeypatch)

    roi = np.full((512, 512, 3), 128, dtype=np.uint8)
    anomaly_map = np.zeros((512, 512), dtype=np.float32)
    anomaly_map[240:270, 240:270] = 1.0
    edge = EdgeDefect(
        side="aoi_edge",
        area=900,
        bbox=(0, 0, 512, 512),
        center=(256, 256),
        max_diff=0,
        is_suspected_dust_or_scratch=True,
        inspector_mode="fusion",
        source_inspector="patchcore",
        patchcore_score=0.9,
        patchcore_threshold=0.5,
    )
    edge.pc_roi = roi
    edge.pc_fg_mask = np.full((512, 512), 255, dtype=np.uint8)
    edge.pc_anomaly_map = anomaly_map

    saver = HeatmapManager(base_dir=str(tmp_path), save_format="png")
    saver._save_patchcore_edge_image(
        save_dir=tmp_path,
        image_name="pc_dust",
        edge_index=0,
        edge_defect=edge,
        full_image=roi,
        omit_image=None,
    )

    all_text = " ".join(captured)
    assert "DUST (Filtered as OK)" in all_text
    assert " NG " not in f" {all_text} "
