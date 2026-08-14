import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import capi_server
from capi_database import CAPIDatabase
from capi_inference import ImageResult
from capi_server import (
    CAPIServer,
    aggregate_judgment,
    build_dual_protocol_response,
    results_to_db_data,
)
from capi_white_frame import (
    WhiteFrameInspection,
    find_white_frame_image,
    inspect_white_frame_image,
)


def _write_frame(
    path: Path,
    *,
    angle: float = 0.0,
    missing_side: str = "",
    break_side: str = "",
    open_corners: int = 35,
) -> Path:
    width, height = 1200, 800
    x1, y1, x2, y2 = 180, 140, 1020, 660
    image = np.zeros((height, width), dtype=np.uint8)
    thickness = 7

    if missing_side != "top":
        cv2.line(image, (x1 + open_corners, y1), (x2 - open_corners, y1), 255, thickness)
    if missing_side != "right":
        cv2.line(image, (x2, y1 + open_corners), (x2, y2 - open_corners), 255, thickness)
    if missing_side != "bottom":
        cv2.line(image, (x1 + open_corners, y2), (x2 - open_corners, y2), 255, thickness)
    if missing_side != "left":
        cv2.line(image, (x1, y1 + open_corners), (x1, y2 - open_corners), 255, thickness)

    if break_side == "top":
        cv2.rectangle(image, (555, y1 - 12), (645, y1 + 12), 0, -1)
    elif break_side == "right":
        cv2.rectangle(image, (x2 - 12, 355), (x2 + 12, 445), 0, -1)
    elif break_side == "bottom":
        cv2.rectangle(image, (555, y2 - 12), (645, y2 + 12), 0, -1)
    elif break_side == "left":
        cv2.rectangle(image, (x1 - 12, 355), (x1 + 12, 445), 0, -1)

    if angle:
        transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR)

    assert cv2.imwrite(str(path), image)
    return path


def _white_result(path: Path, payload: dict) -> ImageResult:
    return ImageResult(
        image_path=path,
        image_size=(1200, 800),
        otsu_bounds=(180, 140, 1021, 661),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.02,
        anomaly_tiles=[],
        inference_time=0.02,
        white_frame_result=payload,
    )


def test_find_white_frame_image_is_optional_and_selects_newest(tmp_path):
    assert find_white_frame_image(tmp_path) is None
    (tmp_path / "W0F00000_100000.png").write_bytes(b"not selected")
    older = tmp_path / "WHITEFRA_100000.png"
    newer = tmp_path / "whitefra_100001.TIF"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    assert find_white_frame_image(tmp_path) == newer


def test_rotated_frame_and_open_corners_are_ok(tmp_path):
    image_path = _write_frame(tmp_path / "WHITEFRA_OK.png", angle=2.4, open_corners=42)

    result = inspect_white_frame_image(image_path)

    assert result.payload["status"] == "OK"
    assert result.payload["ng_sides"] == []
    assert all(side["status"] == "OK" for side in result.payload["sides"].values())
    assert abs(result.payload["angle_deg"]) == pytest.approx(2.4, abs=0.25)


def test_rotated_top_break_reports_only_top_and_keeps_future_coordinate(tmp_path):
    image_path = _write_frame(tmp_path / "WHITEFRA_TOP_NG.png", angle=-1.7, break_side="top")

    result = inspect_white_frame_image(image_path)
    top = result.payload["sides"]["top"]

    assert result.payload["status"] == "NG"
    assert result.payload["ng_sides"] == ["top"]
    assert top["gap_count"] == 1
    assert top["largest_gap_px"] >= 70
    assert isinstance(top["gaps"][0]["center_x"], int)
    assert isinstance(top["gaps"][0]["center_y"], int)


def test_missing_top_reports_top_without_corner_false_positives(tmp_path):
    image_path = _write_frame(tmp_path / "WHITEFRA_TOP_MISSING.png", angle=1.1, missing_side="top")

    result = inspect_white_frame_image(image_path)

    assert result.payload["status"] == "NG"
    assert result.payload["ng_sides"] == ["top"]
    assert result.payload["sides"]["top"]["largest_gap_px"] > 700


def test_blank_white_frame_is_unreadable_not_exception(tmp_path):
    image_path = tmp_path / "WHITEFRA_BLANK.png"
    assert cv2.imwrite(str(image_path), np.zeros((800, 1200), dtype=np.uint8))

    result = inspect_white_frame_image(image_path)

    assert result.payload["status"] == "UNREADABLE"
    assert result.payload["affects_judgment"] is False


def test_shadow_result_does_not_change_judgment_or_tcp_and_persists(tmp_path):
    payload = {
        "algorithm": "white-frame-cv-v1",
        "shadow_mode": True,
        "affects_judgment": False,
        "status": "NG",
        "ng_sides": ["top"],
        "sides": {"top": {"label": "上邊", "status": "NG", "gap_count": 1}},
        "processing_ms": 20.0,
    }
    image_path = tmp_path / "WHITEFRA_123456.png"
    result = _white_result(image_path, payload)
    parsed = {
        "glass_id": "G-WHITE",
        "model_id": "MODEL-A",
        "machine_no": "CAPI01",
        "machine_judgment": "OK",
        "resolution": (1200, 800),
    }

    assert aggregate_judgment([result]) == ("OK", "[]")
    assert build_dual_protocol_response(parsed, "OK", [result]) == build_dual_protocol_response(
        parsed, "OK", []
    )

    db_data = results_to_db_data([result], {})
    assert db_data[0]["is_ng"] == 0
    assert db_data[0]["anomaly_count"] == 0
    assert json.loads(db_data[0]["white_frame_result"])["ng_sides"] == ["top"]

    db = CAPIDatabase(tmp_path / "white_frame.db")
    record_id = db.save_inference_record(
        glass_id="G-WHITE",
        model_id="MODEL-A",
        machine_no="CAPI01",
        resolution=(1200, 800),
        machine_judgment="OK",
        ai_judgment="OK",
        image_dir=str(tmp_path),
        total_images=1,
        ng_images=0,
        ng_details="[]",
        request_time="2026-08-14 10:00:00",
        response_time="2026-08-14 10:00:01",
        processing_seconds=0.02,
        image_results_data=db_data,
    )
    detail = db.get_record_detail(record_id)

    assert detail["ai_judgment"] == "OK"
    assert detail["images"][0]["is_ng"] == 0
    assert detail["images"][0]["white_frame_result"]["status"] == "NG"


def test_server_persists_white_frame_in_post_response_worker(tmp_path, monkeypatch):
    normal_result = _white_result(tmp_path / "W0F00000.png", {})
    normal_result.white_frame_result = None
    fake_inferencer = SimpleNamespace(
        config=SimpleNamespace(
            image_preprocess_pipeline=[],
            image_preprocess_pipelines={},
        ),
        _rotate_detection_images_180=False,
    )
    inspection = WhiteFrameInspection(
        image_path=tmp_path / "WHITEFRA_123456.png",
        image_size=(1200, 800),
        bounds=(180, 140, 1021, 661),
        payload={
            "status": "NG",
            "angle_deg": 1.0,
            "processing_ms": 10.0,
            "sides": {
                "top": {"label": "上邊", "status": "NG"},
                "right": {"label": "右邊", "status": "OK"},
                "bottom": {"label": "下邊", "status": "OK"},
                "left": {"label": "左邊", "status": "OK"},
            },
        },
    )

    def fake_inspection(panel_dir, **_kwargs):
        assert panel_dir == tmp_path
        return inspection

    monkeypatch.setattr(capi_server, "inspect_white_frame_panel", fake_inspection)
    server = CAPIServer.__new__(CAPIServer)
    server.path_mapping = {}
    server._get_or_create_inferencer = lambda _model_id: fake_inferencer
    server.heatmap_manager = SimpleNamespace(save_panel_heatmaps=lambda **_kwargs: {})
    server.save_overview = True
    server.save_tile_detail = True
    server.db = MagicMock()
    server.db.save_inference_record.return_value = 7
    parsed = {
        "glass_id": "G-WHITE",
        "model_id": "MODEL-A",
        "machine_no": "CAPI01",
        "machine_judgment": "OK",
        "resolution": (1200, 800),
        "image_dir": str(tmp_path),
        "bomb_info": None,
    }

    server._save_results_async(
        ("test", 1),
        parsed,
        [normal_result],
        "OK",
        "[]",
        "2026-08-14 10:00:00.000",
        "2026-08-14 10:00:01.000",
        0.1,
    )

    saved = server.db.save_inference_record.call_args.kwargs
    assert saved["ai_judgment"] == "OK"
    assert saved["ng_images"] == 0
    assert len(saved["image_results_data"]) == 2
    assert json.loads(saved["image_results_data"][1]["white_frame_result"])["status"] == "NG"
    assert "FormalJudgment=unchanged TCP=unchanged" in saved["inference_log"]


@pytest.mark.skipif(not Path("WBF/OK.png").exists(), reason="local WBF samples not available")
def test_local_wbf_reference_samples():
    expected_ng_sides = {
        "OK.png": set(),
        "NG.png": {"top"},
        "NG1.png": {"top", "bottom"},
        "NG2.png": {"top", "right", "bottom"},
    }

    for name, expected in expected_ng_sides.items():
        result = inspect_white_frame_image(Path("WBF") / name)
        assert set(result.payload["ng_sides"]) == expected, (name, result.payload)


def test_record_pages_include_shadow_result_panel():
    partial = Path("templates/_white_frame_result.html").read_text(encoding="utf-8")
    assert "白色外框檢測" in partial
    assert "不影響正式 OK／NG 與 TCP 回傳" in partial
    assert "四角不列入判斷" in partial
    for template_name in ("record_detail.html", "record_detail_v3.html"):
        template = Path("templates") / template_name
        assert '{% include "_white_frame_result.html" %}' in template.read_text(encoding="utf-8")
