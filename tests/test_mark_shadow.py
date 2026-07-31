import base64
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import capi_mark_shadow
from capi_database import CAPIDatabase
from capi_mark_shadow import build_mark_shadow_payload, recognize_mark_online
from capi_web import CAPIWebHandler
from mark_shadow.paddle_shadow_worker import (
    PaddleRecognizer,
    ShadowApplication,
    ShadowStore,
    normalize_mark_text,
    prepare_paddle_image,
)


def test_build_mark_shadow_payload_crops_and_rotates_upright():
    image = np.arange(20 * 30, dtype=np.uint16).reshape(20, 30)
    detection = {
        "found": True,
        "text": "BJ",
        "confidence": 0.481,
        "profile_version": 3,
        "roi": "bottom_left",
        "orientation": "rot180",
        "bbox": {"x": 5, "y": 6, "width": 4, "height": 3},
    }

    payload = build_mark_shadow_payload(
        image,
        detection,
        "/images/W0F00000_074058.tif",
        padding_ratio=0,
    )

    png = base64.b64decode(payload["image_png_base64"])
    actual = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    expected = cv2.rotate(image[2:13, 1:13], cv2.ROTATE_180)
    np.testing.assert_array_equal(actual, expected)
    assert payload["source_image"] == "W0F00000_074058.tif"
    assert payload["current_text"] == "BJ"
    assert payload["current_profile_version"] == 3


def test_normalize_mark_text_rejects_non_two_character_results():
    assert normalize_mark_text(" b j ") == "BJ"
    assert normalize_mark_text("B1") == "B1"
    assert normalize_mark_text("B") == ""
    assert normalize_mark_text("BJ1") == ""
    assert normalize_mark_text("B-") == ""


def test_prepare_paddle_image_converts_grayscale_and_bgra_to_bgr():
    grayscale = np.full((12, 20), 180, dtype=np.uint8)
    bgra = np.full((12, 20, 4), 180, dtype=np.uint8)

    prepared_gray = prepare_paddle_image(grayscale)
    prepared_bgra = prepare_paddle_image(bgra)

    assert prepared_gray.shape == (12, 20, 3)
    assert prepared_bgra.shape == (12, 20, 3)
    assert prepared_gray.flags.c_contiguous
    assert prepared_bgra.flags.c_contiguous


def test_paddle_recognizer_sends_three_channel_image_to_model():
    class FakeResult:
        json = {"res": {"rec_text": "EJ", "rec_score": 0.93}}

    class FakeModel:
        def predict(self, *, input, batch_size):
            assert input.shape == (12, 20, 3)
            assert batch_size == 1
            return iter([FakeResult()])

    import threading

    recognizer = PaddleRecognizer.__new__(PaddleRecognizer)
    recognizer.model_name = "fake"
    recognizer._lock = threading.Lock()
    recognizer._model = FakeModel()

    result = recognizer.predict(np.full((12, 20), 180, dtype=np.uint8))

    assert result["paddle_text"] == "EJ"
    assert result["paddle_confidence"] == pytest.approx(0.93)
    assert result["technique"] == "PaddleOCR"
    assert result["worker_version"] == "2"
    assert result["error"] == ""


def test_shadow_application_saves_comparison_and_disagreement_crop(tmp_path):
    class FakeRecognizer:
        model_name = "fake"

        def predict(self, image):
            assert image.shape == (12, 20)
            return {
                "paddle_raw_text": "B1",
                "paddle_text": "B1",
                "paddle_confidence": 0.91,
                "latency_ms": 12.5,
                "model_name": self.model_name,
                "error": "",
            }

    image = np.full((12, 20), 180, dtype=np.uint8)
    encoded, png = cv2.imencode(".png", image)
    assert encoded
    png_bytes = png.tobytes()
    import hashlib

    request_data = {
        "source_path": "/images/W0F00000_074058.tif",
        "source_image": "W0F00000_074058.tif",
        "crop_sha256": hashlib.sha256(png_bytes).hexdigest(),
        "image_png_base64": base64.b64encode(png_bytes).decode("ascii"),
        "current_text": "BJ",
        "current_confidence": 0.481,
        "current_profile_version": 3,
        "current_roi": "bottom_left",
        "current_orientation": "rot180",
    }
    store = ShadowStore(tmp_path / "shadow.db", tmp_path / "disagreements")

    result = ShadowApplication(FakeRecognizer(), store).infer(request_data)

    assert result["agreed"] is False
    assert result["technique"] == "PaddleOCR"
    assert result["worker_version"] == "2"
    assert store.stats()["total"] == 1
    assert store.stats()["disagreed"] == 1
    assert len(list((tmp_path / "disagreements").rglob("*.png"))) == 1


def test_shadow_store_saves_agreed_crop_for_admin_comparison(tmp_path):
    image = np.full((12, 20), 180, dtype=np.uint8)
    encoded, png = cv2.imencode(".png", image)
    assert encoded
    png_bytes = png.tobytes()
    request_data = {
        "source_image": "W0F00000_080000.tif",
        "crop_sha256": "a" * 64,
        "current_text": "EJ",
    }
    result = {
        "paddle_raw_text": "EJ",
        "paddle_text": "EJ",
        "paddle_confidence": 0.95,
        "latency_ms": 11.2,
        "model_name": "fake",
        "error": "",
    }
    store = ShadowStore(
        tmp_path / "data" / "mark_shadow.db",
        tmp_path / "data" / "disagreements",
    )

    result_id = store.save(request_data, result, png_bytes)

    with sqlite3.connect(store.db_path) as connection:
        crop_path = connection.execute(
            "SELECT crop_path FROM mark_shadow_results WHERE id = ?",
            (result_id,),
        ).fetchone()[0]
    assert Path(crop_path).is_file()
    assert Path(crop_path).is_relative_to(tmp_path / "data" / "crops")


def _save_shadow_row(store, *, current_text, paddle_text, latency_ms, error=""):
    image = np.full((12, 20), 180, dtype=np.uint8)
    encoded, png = cv2.imencode(".png", image)
    assert encoded
    png_bytes = png.tobytes()
    suffix = f"{current_text}-{paddle_text}-{latency_ms}-{error}"
    import hashlib

    crop_hash = hashlib.sha256((suffix.encode("utf-8") + png_bytes)).hexdigest()
    return store.save(
        {
            "source_image": f"W0F00000_{crop_hash[:6]}.tif",
            "crop_sha256": crop_hash,
            "current_text": current_text,
            "current_confidence": 0.8,
            "current_profile_version": 2,
            "current_roi": "bottom_left",
            "current_orientation": "rot180",
        },
        {
            "paddle_raw_text": paddle_text,
            "paddle_text": paddle_text,
            "paddle_confidence": 0.9 if paddle_text else 0.0,
            "latency_ms": latency_ms,
            "model_name": "fake",
            "error": error,
        },
        png_bytes,
    )


def test_inference_database_matches_mark_shadow_image_to_record(tmp_path):
    db = CAPIDatabase(tmp_path / "capi.db")
    image_path = tmp_path / "panel" / "W0F00000_101010.tif"
    image_path.parent.mkdir()
    record_id = db.save_inference_record(
        glass_id="G-LINK",
        model_id="MODEL-LINK",
        machine_no="M-LINK",
        resolution=(100, 100),
        machine_judgment="OK",
        ai_judgment="OK",
        image_dir=str(image_path.parent),
        total_images=1,
        ng_images=0,
        ng_details="[]",
        request_time="2026-07-31 10:10:10",
        response_time="2026-07-31 10:10:11",
        processing_seconds=0.1,
        image_results_data=[
            {"image_path": str(image_path), "image_name": image_path.name}
        ],
    )

    assert db.find_inference_record_ids_for_images(
        [(str(image_path), image_path.name)]
    ) == [record_id]
    assert db.find_inference_record_ids_for_images(
        [("", image_path.name)]
    ) == [record_id]


def test_mark_shadow_admin_api_returns_comparisons_and_success_latency(tmp_path, monkeypatch):
    db_path = tmp_path / "data" / "mark_shadow.db"
    store = ShadowStore(db_path, tmp_path / "data" / "disagreements")
    _save_shadow_row(
        store,
        current_text="EJ",
        paddle_text="EJ",
        latency_ms=12.5,
    )
    _save_shadow_row(
        store,
        current_text="EJ",
        paddle_text="B1",
        latency_ms=20.0,
    )
    error_id = _save_shadow_row(
        store,
        current_text="EJ",
        paddle_text="",
        latency_ms=0.0,
        error="tuple index out of range",
    )
    source_path = "/images/W0F00000_error.tif"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE mark_shadow_results SET source_path = ? WHERE id = ?",
            (source_path, error_id),
        )
    monkeypatch.setattr(
        CAPIWebHandler,
        "_capi_server_instance",
        SimpleNamespace(
            server_config={"mark_shadow": {"database_path": str(db_path)}}
        ),
    )
    handler = object.__new__(CAPIWebHandler)
    linked_refs = []
    handler.db = SimpleNamespace(
        find_inference_record_ids_for_images=lambda refs: (
            linked_refs.extend(refs) or [456]
        )
    )
    sent = []
    handler._send_json = (
        lambda payload, status=200, **kwargs: sent.append((payload, status))
    )

    handler._handle_api_settings_mark_shadow(
        {"filter": ["errors"], "limit": ["50"]}
    )

    payload, status = sent[0]
    assert status == 200
    assert payload["available"] is True
    assert [row["id"] for row in payload["rows"]] == [error_id]
    assert "crop_path" not in payload["rows"][0]
    assert payload["rows"][0]["crop_url"].endswith(f"id={error_id}")
    assert linked_refs[0][0] == source_path
    assert linked_refs[0][1].startswith("W0F00000_")
    assert payload["rows"][0]["inference_record_id"] == 456
    assert payload["rows"][0]["record_url"] == "/record/456"
    assert payload["stats"]["total"] == 3
    assert payload["stats"]["agreed"] == 1
    assert payload["stats"]["error_count"] == 1
    assert payload["stats"]["latency_ms"]["average"] == pytest.approx(16.25)


def test_mark_shadow_crop_api_only_serves_recorded_shadow_crop(tmp_path, monkeypatch):
    db_path = tmp_path / "data" / "mark_shadow.db"
    store = ShadowStore(db_path, tmp_path / "data" / "disagreements")
    result_id = _save_shadow_row(
        store,
        current_text="EJ",
        paddle_text="EJ",
        latency_ms=12.5,
    )
    monkeypatch.setattr(
        CAPIWebHandler,
        "_capi_server_instance",
        SimpleNamespace(
            server_config={"mark_shadow": {"database_path": str(db_path)}}
        ),
    )
    handler = object.__new__(CAPIWebHandler)
    images = []
    handler._send_image_array_png = lambda image: images.append(image)
    handler._send_json = lambda payload, status=200, **kwargs: pytest.fail(
        f"unexpected JSON response {status}: {payload}"
    )

    handler._handle_api_settings_mark_shadow_crop({"id": [str(result_id)]})

    assert len(images) == 1
    assert images[0].shape == (12, 20)


def test_mark_shadow_crop_api_rejects_path_outside_shadow_data(tmp_path, monkeypatch):
    db_path = tmp_path / "data" / "mark_shadow.db"
    store = ShadowStore(db_path, tmp_path / "data" / "disagreements")
    result_id = _save_shadow_row(
        store,
        current_text="EJ",
        paddle_text="B1",
        latency_ms=12.5,
    )
    outside_crop = tmp_path / "outside.png"
    outside_crop.write_bytes(b"not served")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE mark_shadow_results SET crop_path = ? WHERE id = ?",
            (str(outside_crop), result_id),
        )
    monkeypatch.setattr(
        CAPIWebHandler,
        "_capi_server_instance",
        SimpleNamespace(
            server_config={"mark_shadow": {"database_path": str(db_path)}}
        ),
    )
    handler = object.__new__(CAPIWebHandler)
    sent = []
    handler._send_json = (
        lambda payload, status=200, **kwargs: sent.append((payload, status))
    )

    handler._handle_api_settings_mark_shadow_crop({"id": [str(result_id)]})

    assert sent[0][1] == 403
    assert "允許範圍" in sent[0][0]["error"]


def test_mark_shadow_settings_ui_has_admin_comparison_fields():
    root = Path(__file__).resolve().parent.parent
    template = (root / "templates" / "settings.html").read_text(encoding="utf-8")
    web_source = (root / "capi_web.py").read_text(encoding="utf-8")

    assert "Mark PPOCR檢查" in template
    assert "/api/settings/mark-shadow" in template
    for label in ("原辨識", "Paddle 辨識", "信心", "耗時", "錯誤", "Crop", "推論紀錄", "row.record_url"):
        assert label in template
    assert 'path == "/api/settings/mark-shadow"' in web_source
    assert "_require_settings_user(api=True, admin=True)" in web_source


def test_online_client_adds_versions_for_legacy_worker_response(monkeypatch):
    class FakeClient:
        def recognize(self, image, detection, source_path):
            return {
                "success": True,
                "paddle_text": "B1",
                "paddle_confidence": 0.94,
                "latency_ms": 18.0,
                "model_name": "PP-OCRv6_medium_rec",
            }

    monkeypatch.setattr(capi_mark_shadow, "_CLIENT", FakeClient())

    result = recognize_mark_online(
        np.zeros((20, 30), dtype=np.uint8),
        {"found": True},
        Path("W0F00000_000000.tif"),
    )

    assert result["success"] is True
    assert result["technique"] == "PaddleOCR"
    assert result["engine_version"] == "3.7.0"
    assert result["worker_version"] == "1"
    assert result["round_trip_ms"] >= 0


def _legacy_mark_detection():
    return {
        "found": True,
        "text": "EJ",
        "confidence": 0.61,
        "profile_version": 5,
        "roi": "bottom_left",
        "orientation": "rot180",
        "bbox": {"x": 5, "y": 6, "width": 12, "height": 8},
    }


def test_formal_mark_uses_paddle_text_and_logs_technology(
    tmp_path,
    monkeypatch,
    capsys,
):
    from capi_inference import CAPIInferencer, ImageResult

    image_path = tmp_path / "W0F00000_000000.tif"
    assert cv2.imwrite(str(image_path), np.full((40, 60), 180, dtype=np.uint8))
    monkeypatch.setattr(
        "capi_mark_detector.detect_panel_mark",
        lambda image, include_debug=False: _legacy_mark_detection(),
    )
    monkeypatch.setattr(
        "capi_mark_shadow.recognize_mark_online",
        lambda image, detection, source_path: {
            "success": True,
            "paddle_text": "B1",
            "paddle_confidence": 0.94,
            "latency_ms": 18.5,
            "round_trip_ms": 21.2,
            "model_name": "PP-OCRv6_medium_rec",
            "engine_version": "3.7.0",
            "worker_version": "2",
            "error": "",
        },
    )
    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = SimpleNamespace(inference_rotate_180_enabled=False)

    detection, regions = inferencer._detect_panel_mark_binary_region([image_path])

    assert detection["text"] == "B1"
    assert detection["confidence"] == pytest.approx(0.94)
    assert detection["legacy_text"] == "EJ"
    assert detection["recognition_technique"] == "PaddleOCR"
    assert detection["recognition_fallback"] is False
    assert [(r.x1, r.y1, r.x2, r.y2) for r in regions] == [(5, 6, 17, 14)]
    image_result = ImageResult(
        image_path=image_path,
        image_size=(60, 40),
        otsu_bounds=(0, 0, 60, 40),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
    )
    inferencer._attach_panel_mark_binary_to_results(
        [image_result],
        detection,
        regions,
    )
    assert image_result.mark_text == "B1"
    assert image_result.mark_confidence == pytest.approx(0.94)
    log = capsys.readouterr().out
    assert "technique=PaddleOCR" in log
    assert "engine_version=3.7.0" in log
    assert "model=PP-OCRv6_medium_rec" in log
    assert "worker_api=v2" in log
    assert "decision=primary text=B1" in log


def test_formal_mark_falls_back_and_logs_reason_when_paddle_fails(
    tmp_path,
    monkeypatch,
    capsys,
):
    from capi_inference import CAPIInferencer

    image_path = tmp_path / "W0F00000_000001.tif"
    assert cv2.imwrite(str(image_path), np.full((40, 60), 180, dtype=np.uint8))
    monkeypatch.setattr(
        "capi_mark_detector.detect_panel_mark",
        lambda image, include_debug=False: _legacy_mark_detection(),
    )
    monkeypatch.setattr(
        "capi_mark_shadow.recognize_mark_online",
        lambda image, detection, source_path: {
            "success": False,
            "error": "connection refused",
            "round_trip_ms": 2.1,
        },
    )
    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = SimpleNamespace(inference_rotate_180_enabled=False)

    detection, _regions = inferencer._detect_panel_mark_binary_region([image_path])

    assert detection["text"] == "EJ"
    assert detection["confidence"] == pytest.approx(0.61)
    assert detection["recognition_technique"] == "DotMatrixCV"
    assert detection["recognition_fallback"] is True
    log = capsys.readouterr().out
    assert "technique=DotMatrixCV" in log
    assert "version=profile-v5" in log
    assert "decision=fallback text=EJ" in log
    assert "reason=connection refused" in log


def test_shadow_service_template_uses_localhost_only():
    template = (
        Path(__file__).resolve().parent.parent
        / "mark_shadow"
        / "capi-mark-shadow.service.template"
    ).read_text(encoding="utf-8")

    assert "--host 127.0.0.1" in template
    assert "--device cpu" in template


def test_offline_installer_has_one_command_entrypoint_and_reuses_release():
    root = Path(__file__).resolve().parent.parent
    entrypoint = (root / "mark_shadow" / "install.sh").read_text(encoding="utf-8")
    installer = (
        root / "mark_shadow" / "install_offline.sh"
    ).read_text(encoding="utf-8")

    assert 'exec "$BUNDLE_ROOT/scripts/install_offline.sh" "$@"' in entrypoint
    assert "Release already installed; reusing" in installer
    assert 'bash "$APP_ROOT/start_server.sh" restart --no-tail' in installer


def test_offline_installer_enables_shadow_and_sets_actual_database_path(tmp_path):
    root = Path(__file__).resolve().parent.parent
    installer = (
        root / "mark_shadow" / "install_offline.sh"
    ).read_text(encoding="utf-8")
    match = re.search(
        r'python3 - "\$CONFIG_FILE" "\$DATA_DIR/mark_shadow\.db" <<\'PY\'\n'
        r"(.*?)\nPY",
        installer,
        flags=re.DOTALL,
    )
    assert match
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        "server:\n"
        "  port: 7890\n"
        "mark_shadow:\n"
        "  enabled: false\n"
        "  endpoint: http://127.0.0.1:9999/infer\n"
        "  max_queue: 99\n"
        "\n"
        "# cleanup settings\n"
        "cleanup:\n"
        "  enabled: true\n",
        encoding="utf-8",
    )
    database_path = "/custom/mark_shadow/data/mark_shadow.db"

    subprocess.run(
        [
            sys.executable,
            "-c",
            match.group(1),
            str(config_path),
            database_path,
        ],
        check=True,
    )

    updated = config_path.read_text(encoding="utf-8")
    assert "  enabled: true" in updated
    assert "  endpoint: http://127.0.0.1:8765/infer" in updated
    assert f"  database_path: {database_path}" in updated
    assert "  max_queue: 99" in updated
    assert updated.index("  database_path:") < updated.index("# cleanup settings")
