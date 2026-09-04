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
    _white_frame_image_result,
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
        raw_bounds=(180, 140, 1021, 661),
        inference_time=0.02,
        white_frame_result=payload,
        report_image_prefix="WHITEFRA",
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
    assert result.payload["affects_judgment"] is True


def test_white_frame_ng_changes_formal_judgment_and_qjpg_uses_its_own_frame_coords(tmp_path):
    payload = {
        "algorithm": "white-frame-cv-v1",
        "affects_judgment": True,
        "status": "NG",
        "ng_sides": ["top"],
        "sides": {
            "top": {
                "label": "上邊",
                "status": "NG",
                "gap_count": 1,
                "gaps": [{"center_x": 600, "center_y": 140}],
            }
        },
        "processing_ms": 20.0,
    }
    image_path = tmp_path / "WHITEFRA_123456.png"
    result = _white_frame_image_result(WhiteFrameInspection(
        image_path=image_path,
        image_size=(1200, 800),
        bounds=(180, 140, 1021, 661),
        payload=payload,
    ))
    parsed = {
        "glass_id": "G-WHITE",
        "model_id": "MODEL-A",
        "machine_no": "CAPI01",
        "machine_judgment": "OK",
        "resolution": (1200, 800),
    }

    ai_judgment, ng_details = aggregate_judgment([result])
    assert ai_judgment == "NG"
    assert json.loads(ng_details)[0]["type"] == "white_frame_gap"
    response = build_dual_protocol_response(
        parsed,
        ai_judgment,
        [result],
        SimpleNamespace(report_white_dot_defect_code="WHT01"),
    )
    # (600, 140) is converted with WHITEFRA bounds (180,140)-(1021,661),
    # not with any black-screen image bounds.
    assert "NGWHT010059900000WHITEFRA" in response

    db_data = results_to_db_data([result], {})
    assert db_data[0]["is_ng"] == 1
    assert db_data[0]["anomaly_count"] == 1
    assert json.loads(db_data[0]["white_frame_result"])["ng_sides"] == ["top"]

    db = CAPIDatabase(tmp_path / "white_frame.db")
    record_id = db.save_inference_record(
        glass_id="G-WHITE",
        model_id="MODEL-A",
        machine_no="CAPI01",
        resolution=(1200, 800),
        machine_judgment="OK",
        ai_judgment="NG",
        image_dir=str(tmp_path),
        total_images=1,
        ng_images=1,
        ng_details=ng_details,
        request_time="2026-08-14 10:00:00",
        response_time="2026-08-14 10:00:01",
        processing_seconds=0.02,
        image_results_data=db_data,
    )
    detail = db.get_record_detail(record_id)

    assert detail["ai_judgment"] == "NG"
    assert detail["images"][0]["is_ng"] == 1
    assert detail["images"][0]["white_frame_result"]["status"] == "NG"


def test_white_frame_summary_query_returns_images_sides_coordinates_and_filters(tmp_path):
    db = CAPIDatabase(tmp_path / "white_frame_summary.db")

    def save(name, status, machine, request_time, sides):
        payload = {
            "status": status,
            "sides": sides,
            "ng_sides": [key for key, value in sides.items() if value.get("status") == "NG"],
        }
        return db.save_inference_record(
            glass_id=f"G-{name}",
            model_id="MODEL-A",
            machine_no=machine,
            resolution=(1200, 800),
            machine_judgment="OK",
            ai_judgment="OK",
            image_dir=str(tmp_path),
            total_images=1,
            ng_images=0,
            ng_details="[]",
            request_time=request_time,
            response_time=request_time,
            processing_seconds=0.02,
            image_results_data=[{
                "image_path": str(tmp_path / name),
                "image_name": name,
                "image_width": 1200,
                "image_height": 800,
                "white_frame_result": json.dumps(payload),
            }],
        )

    top_ng = {
        "top": {
            "label": "上邊",
            "status": "NG",
            "gap_count": 1,
            "largest_gap_px": 90,
            "gaps": [{"center_x": 610, "center_y": 142}],
        },
        "right": {"label": "右邊", "status": "OK", "gaps": []},
        "bottom": {"label": "下邊", "status": "OK", "gaps": []},
        "left": {"label": "左邊", "status": "OK", "gaps": []},
    }
    all_ok = {
        side: {"label": side, "status": "OK", "gaps": []}
        for side in ("上邊", "右邊", "下邊", "左邊")
    }
    save("WHITEFRA_NG.png", "NG", "CAPI01", "2026-08-14 10:00:00", top_ng)
    save("WHITEFRA_OK.png", "OK", "CAPI02", "2026-08-15 10:00:00", all_ok)

    rows, total, summary = db.query_white_frame_paged(limit=10)
    assert total == 2
    assert summary == {"total": 2, "ok": 1, "ng": 1, "unreadable": 0}
    ng_row = next(row for row in rows if row["white_frame_status"] == "NG")
    assert ng_row["white_frame_ng_sides"] == ["top"]
    assert ng_row["white_frame_ng_sides_detail"][0]["coordinates"] == [(610, 142)]

    rows, total, summary = db.query_white_frame_paged(status="NG", edge="top", machine_no="CAPI01")
    assert total == 1
    assert summary["ng"] == 1
    assert rows[0]["glass_id"] == "G-WHITEFRA_NG.png"

    rows, total, _summary = db.query_white_frame_paged(status="NG", edge="left")
    assert rows == []
    assert total == 0


def test_server_runs_white_frame_before_formal_response_and_does_not_allow_ok_i(tmp_path, monkeypatch):
    normal_result = _white_result(tmp_path / "W0F00000.png", {})
    normal_result.white_frame_result = None
    fake_inferencer = SimpleNamespace(
        config=SimpleNamespace(
            image_abnormal_detection_enabled=False,
            image_preprocess_pipeline=[],
            image_preprocess_pipelines={},
        ),
        _rotate_detection_images_180=False,
        process_panel=lambda *_args, **_kwargs: (
            [normal_result], None, False, "", False, None, {}
        ),
    )
    inspection = WhiteFrameInspection(
        image_path=tmp_path / "WHITEFRA_123456.png",
        image_size=(1200, 800),
        bounds=(180, 140, 1021, 661),
        payload={
            "status": "NG",
            "affects_judgment": True,
            "ng_sides": ["top"],
            "angle_deg": 1.0,
            "processing_ms": 10.0,
            "sides": {
                "top": {
                    "label": "上邊",
                    "status": "NG",
                    "gap_count": 1,
                    "gaps": [{"center_x": 600, "center_y": 140}],
                },
                "right": {"label": "右邊", "status": "OK"},
                "bottom": {"label": "下邊", "status": "OK"},
                "left": {"label": "左邊", "status": "OK"},
            },
        },
    )

    def fake_inspection(image_path, **_kwargs):
        assert image_path == inspection.image_path
        return inspection

    monkeypatch.setattr(capi_server, "inspect_white_frame_image", fake_inspection)
    server = CAPIServer.__new__(CAPIServer)
    server.path_mapping = {}
    server._get_or_create_inferencer = lambda _model_id: fake_inferencer
    server.station_adapter = SimpleNamespace(
        find_white_frame_image=lambda panel_dir: inspection.image_path,
    )
    server.cpu_workers = 1
    import threading
    server._gpu_lock = threading.Lock()
    server._evaluate_within_spec_for_inference = MagicMock(
        side_effect=AssertionError("white-frame NG must not be converted to OK-i")
    )
    parsed = {
        "glass_id": "G-WHITE",
        "model_id": "MODEL-A",
        "machine_no": "CAPI01",
        "machine_judgment": "OK",
        "resolution": (1200, 800),
        "image_dir": str(tmp_path),
        "bomb_info": None,
    }

    ai_judgment, ng_details, results, *_rest = server._process_request(parsed)

    assert ai_judgment == "NG"
    assert len(results) == 2
    assert results[-1].report_image_prefix == "WHITEFRA"
    assert json.loads(ng_details)[0]["type"] == "white_frame_gap"
    server._evaluate_within_spec_for_inference.assert_not_called()


def test_async_save_does_not_reinspect_or_change_formal_white_frame_result(tmp_path, monkeypatch):
    payload = {
        "status": "NG",
        "affects_judgment": True,
        "ng_sides": ["top"],
        "sides": {
            "top": {
                "label": "上邊",
                "status": "NG",
                "gap_count": 1,
                "gaps": [{"center_x": 600, "center_y": 140}],
            }
        },
    }
    formal_result = _white_result(tmp_path / "WHITEFRA_123456.png", payload)
    fake_inferencer = SimpleNamespace(
        config=SimpleNamespace(image_preprocess_pipeline=[], image_preprocess_pipelines={}),
    )
    monkeypatch.setattr(
        capi_server,
        "inspect_white_frame_image",
        MagicMock(side_effect=AssertionError("post-response inspection is forbidden")),
    )
    server = CAPIServer.__new__(CAPIServer)
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
        ("test", 1), parsed, [formal_result], "NG", "[]",
        "2026-08-21 10:00:00.000", "2026-08-21 10:00:01.000", 0.1,
    )

    saved = server.db.save_inference_record.call_args.kwargs
    assert saved["ai_judgment"] == "NG"
    assert saved["ng_images"] == 1
    assert len(saved["image_results_data"]) == 1


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


def test_record_pages_include_white_frame_result_panel():
    partial = Path("templates/_white_frame_result.html").read_text(encoding="utf-8")
    assert "白色外框檢測" in partial
    assert "不影響正式 OK／NG" not in partial
    assert "?white_frame=1" in partial
    assert "紅色標記為斷線中心" in partial
    for template_name in ("record_detail.html", "record_detail_v3.html"):
        template = Path("templates") / template_name
        assert '{% include "_white_frame_result.html" %}' in template.read_text(encoding="utf-8")


def test_white_frame_annotation_marks_scaled_gap_center_only_for_ng():
    from capi_web import CAPIWebHandler

    image = np.zeros((100, 200), dtype=np.uint8)
    payload = {
        "status": "NG",
        "sides": {
            "top": {
                "status": "NG",
                "gaps": [{"center_x": 500, "center_y": 250}],
            },
        },
    }

    annotated = CAPIWebHandler._annotate_white_frame_gaps(
        image,
        payload,
        source_size=(1000, 500),
    )

    assert annotated.shape == (100, 200, 3)
    assert tuple(annotated[50, 100]) == (0, 0, 255)
    assert tuple(annotated[50, 199]) == (0, 0, 0)

    untouched = CAPIWebHandler._annotate_white_frame_gaps(
        image,
        {"status": "OK", "sides": {}},
    )
    assert untouched is image


def test_source_image_preview_applies_white_frame_annotation(tmp_path):
    from capi_web import CAPIWebHandler

    image_name = "WHITEFRA_TEST.png"
    assert cv2.imwrite(str(tmp_path / image_name), np.zeros((500, 1000), dtype=np.uint8))
    payload = {
        "status": "NG",
        "sides": {
            "top": {
                "status": "NG",
                "gaps": [{"center_x": 500, "center_y": 250}],
            },
        },
    }
    detail = {
        "image_dir": str(tmp_path),
        "images": [{"image_name": image_name, "white_frame_result": payload}],
    }
    handler = object.__new__(CAPIWebHandler)
    handler.db = SimpleNamespace(get_record_detail=lambda _record_id: detail)
    sent_images = []
    handler._send_image_array_png = lambda image: sent_images.append(image)
    handler._send_404 = lambda *_args: pytest.fail("unexpected 404")
    handler._send_binary = lambda *_args: pytest.fail("unexpected raw image response")

    handler._handle_source_image(
        f"/images/7/{image_name}",
        {"preview": ["1"], "white_frame": ["1"]},
    )

    assert len(sent_images) == 1
    assert sent_images[0].shape == (320, 640, 3)
    assert tuple(sent_images[0][160, 320]) == (0, 0, 255)


def test_white_frame_summary_page_template_and_nav_are_present():
    page = Path("templates/white_frame.html").read_text(encoding="utf-8")
    nav = Path("templates/base.html").read_text(encoding="utf-8")
    settings = Path("templates/settings.html").read_text(encoding="utf-8")
    assert "白色外框檢測總表" in page
    assert "r.image_url" in page
    assert "r.white_frame_ng_sides_detail" in page
    assert 'href="/white-frame"' not in nav
    assert 'href=\"/white-frame\"' in settings
