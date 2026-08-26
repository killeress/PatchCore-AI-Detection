import base64
import io
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
import capi_web
from capi_inference import CAPIInferencer
from capi_database import CAPIDatabase
from capi_mark_shadow import (
    MARK_FORCED_CHAR_CONVERSIONS_PARAM,
    build_mark_shadow_payload,
    normalize_forced_char_conversions,
    recognize_mark_online,
    set_forced_char_conversions,
)
from capi_web import CAPIWebHandler
from mark_shadow.paddle_shadow_worker import (
    apply_forced_char_conversions,
    PaddleRecognizer,
    MarkTemporalStabilizer,
    ShadowApplication,
    ShadowStore,
    normalize_mark_text,
    prepare_paddle_image,
    rescue_paddle_u_with_dotmatrix_v,
)


@pytest.fixture(autouse=True)
def _reset_forced_char_conversions():
    set_forced_char_conversions(None)
    yield
    set_forced_char_conversions(None)


def test_build_mark_shadow_payload_keeps_full_mark_envelope_and_rotates_upright():
    image = np.arange(20 * 30, dtype=np.uint16).reshape(20, 30)
    detection = {
        "found": True,
        "text": "BJ",
        "confidence": 0.481,
        "profile_version": 3,
        "roi": "bottom_left",
        "orientation": "rot180",
        "mark_machine_no": "CAPI13",
        "mark_model_id": "MODEL-A",
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
    expected = cv2.rotate(image[2:13, 0:14], cv2.ROTATE_180)
    np.testing.assert_array_equal(actual, expected)
    assert payload["source_image"] == "W0F00000_074058.tif"
    assert payload["current_text"] == "BJ"
    assert payload["current_profile_version"] == 3
    assert payload["machine_no"] == "CAPI13"
    assert payload["model_id"] == "MODEL-A"
    assert payload["forced_char_conversions"] == [
        {"paddle": "U", "dotmatrix": "V"}
    ]


def test_mark_payload_uses_runtime_forced_char_conversions():
    set_forced_char_conversions([
        {"paddle": "0", "dotmatrix": "O"},
    ])
    payload = build_mark_shadow_payload(
        np.zeros((20, 30), dtype=np.uint8),
        {
            "found": True,
            "text": "00",
            "bbox": {"x": 5, "y": 6, "width": 12, "height": 8},
        },
        "W0F00000_000000.tif",
    )

    assert payload["forced_char_conversions"] == [
        {"paddle": "0", "dotmatrix": "O"}
    ]


def test_normalize_forced_char_conversions_rejects_invalid_or_duplicate_rules():
    assert normalize_forced_char_conversions([]) == []
    assert normalize_forced_char_conversions([
        {"paddle": "u", "dotmatrix": "v"},
    ]) == [{"paddle": "U", "dotmatrix": "V"}]
    with pytest.raises(ValueError):
        normalize_forced_char_conversions([
            {"paddle": "U", "dotmatrix": "V"},
            {"paddle": "U", "dotmatrix": "V"},
        ])
    with pytest.raises(ValueError):
        normalize_forced_char_conversions([
            {"paddle": "U", "dotmatrix": "U"},
        ])


@pytest.mark.parametrize("inference_rotate_180_enabled", [False, True])
def test_mark_crop_always_rotates_after_optional_full_image_rotation(
    tmp_path,
    inference_rotate_180_enabled,
):
    raw_image = np.arange(12 * 20, dtype=np.uint8).reshape(12, 20)
    image_path = tmp_path / "W0F00000_080000.tif"
    assert cv2.imwrite(str(image_path), raw_image)

    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = SimpleNamespace(
        inference_rotate_180_enabled=inference_rotate_180_enabled,
    )
    detection_image = inferencer._read_detection_image(image_path)
    detection = {
        "found": True,
        "text": "EJ",
        "roi": "bottom_left",
        # The locator result must not control the PPOCR crop direction.
        "orientation": "normal",
        "bbox": {"x": 0, "y": 0, "width": 20, "height": 12},
    }

    payload = build_mark_shadow_payload(
        detection_image,
        detection,
        image_path,
        padding_ratio=0,
    )

    png = base64.b64decode(payload["image_png_base64"])
    actual = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    expected = cv2.rotate(detection_image, cv2.ROTATE_180)
    np.testing.assert_array_equal(actual, expected)


def test_mark_stream_key_uses_fixed_crop_rotation_not_locator_orientation():
    normal_key = CAPIInferencer._build_mark_stream_key(
        "CAPI13",
        "MODEL-A",
        {"roi": "bottom_left", "orientation": "normal"},
    )
    locator_rotated_key = CAPIInferencer._build_mark_stream_key(
        "CAPI13",
        "MODEL-A",
        {"roi": "bottom_left", "orientation": "rot180"},
    )

    assert normal_key == "CAPI13|MODEL-A|bottom_left|rot180"
    assert locator_rotated_key == normal_key


def test_normalize_mark_text_rejects_non_two_character_results():
    assert normalize_mark_text(" b j ") == "BJ"
    assert normalize_mark_text("B1") == "B1"
    assert normalize_mark_text("B") == ""
    assert normalize_mark_text("BJ1") == ""
    assert normalize_mark_text("B-") == ""


@pytest.mark.parametrize(
    ("paddle_text", "dotmatrix_text", "expected_text", "expected_positions"),
    [
        ("UU", "VV", "VV", (0, 1)),
        ("U1", "V1", "V1", (0,)),
        ("1U", "1V", "1V", (1,)),
        ("UU", "UU", "UU", ()),
        ("VV", "UU", "VV", ()),
        ("UU", "0V", "UV", (1,)),
    ],
)
def test_uv_rescue_only_replaces_paddle_u_conflicting_with_dotmatrix_v(
    paddle_text,
    dotmatrix_text,
    expected_text,
    expected_positions,
):
    assert rescue_paddle_u_with_dotmatrix_v(
        paddle_text,
        dotmatrix_text,
    ) == (expected_text, expected_positions)


def test_user_defined_conflict_rules_replace_only_matching_positions():
    assert apply_forced_char_conversions(
        "00",
        "O0",
        [{"paddle": "0", "dotmatrix": "O"}],
    ) == ("O0", (0,), (("0", "O"),))
    assert apply_forced_char_conversions("UU", "VV", []) == ("UU", (), ())


def test_mark_temporal_stabilizer_uses_recent_stream_and_rejects_isolated_values():
    stabilizer = MarkTemporalStabilizer()
    key = "M1|MODEL-A|bottom_left|rot180"

    assert stabilizer.observe(key, "K5")["final_text"] == "K5"
    assert stabilizer.observe(key, "K5")["final_text"] == "K5"
    assert stabilizer.observe(key, "K5")["adoption_reason"] == "stable_match"

    isolated = stabilizer.observe(key, "K6")
    assert isolated["final_text"] == "K5"
    assert isolated["adoption_reason"] == "temporal_outlier"
    assert isolated["temporal_history_count"] == 4

    # A different one-off spelling must not join the K6 candidate run.
    assert stabilizer.observe(key, "KS")["final_text"] == "K5"
    assert stabilizer.observe(key, "K6")["final_text"] == "K5"
    assert stabilizer.observe(key, "K6")["final_text"] == "K5"
    switched = stabilizer.observe(key, "K6")
    assert switched["final_text"] == "K6"
    assert switched["adoption_reason"] == "temporal_switch"


def test_model_switch_resets_history_even_when_switching_back(tmp_path):
    store = ShadowStore(
        tmp_path / "shadow.db",
        tmp_path / "disagreements",
    )
    application = ShadowApplication(SimpleNamespace(model_name="fake"), store)
    key_a = "CAPI13|MODEL-A|bottom_left|rot180"
    key_a_other_orientation = "CAPI13|MODEL-A|top_right|normal"
    key_b = "CAPI13|MODEL-B|bottom_left|rot180"

    assert application._reset_for_model_context(
        {"machine_no": "CAPI13", "model_id": "MODEL-A"},
        key_a,
    ) == "context_start"
    application.temporal.observe(key_a, "K5")
    application.temporal.observe(key_a, "K5")
    assert application.temporal.observe(key_a, "K5")["temporal_stable_text"] == "K5"
    application.temporal.observe(key_a_other_orientation, "K5")
    application.temporal.observe(key_a_other_orientation, "K5")
    assert application.temporal.observe(
        key_a_other_orientation,
        "K5",
    )["temporal_stable_text"] == "K5"

    assert application._reset_for_model_context(
        {"machine_no": "CAPI13", "model_id": "MODEL-B"},
        key_b,
    ) == "model_switch_reset"
    model_b = application.temporal.observe(key_b, "EJ")
    assert model_b["final_text"] == "EJ"
    assert model_b["temporal_history_count"] == 1

    assert application._reset_for_model_context(
        {"machine_no": "CAPI13", "model_id": "MODEL-A"},
        key_a,
    ) == "model_switch_reset"
    switched_back = application.temporal.observe(key_a, "N5")
    assert switched_back["final_text"] == "N5"
    assert switched_back["temporal_stable_text"] == ""
    assert switched_back["temporal_history_count"] == 1
    other_orientation = application.temporal.observe(
        key_a_other_orientation,
        "N5",
    )
    assert other_orientation["final_text"] == "N5"
    assert other_orientation["temporal_stable_text"] == ""
    assert other_orientation["temporal_history_count"] == 1


def test_formal_mark_uses_temporal_final_but_keeps_paddle_raw(monkeypatch):
    import capi_mark_shadow

    monkeypatch.setattr(
        capi_mark_shadow,
        "recognize_mark_online",
        lambda image, detection, source_path: {
            "success": True,
            "paddle_text": "K6",
            "final_text": "K5",
            "adoption_reason": "temporal_outlier",
            "paddle_confidence": 0.648,
            "model_name": "PP-OCRv6_medium_rec",
            "engine_version": "3.7.0",
            "worker_version": "2",
            "latency_ms": 30.0,
            "round_trip_ms": 31.0,
            "temporal_stable_text": "K5",
            "temporal_history_count": 100,
            "temporal_stable_support_count": 94,
        },
    )
    detection = {
        "found": True,
        "text": "K5",
        "confidence": 0.4,
        "profile_version": 5,
    }

    CAPIInferencer._apply_online_paddle_mark_recognition(
        np.zeros((12, 20), dtype=np.uint8),
        detection,
        Path("/images/W0F00000_101653.tif"),
    )

    assert detection["paddle_text"] == "K6"
    assert detection["final_text"] == "K5"
    assert detection["text"] == "K5"
    assert detection["recognition_reason"] == "temporal_outlier"


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
    assert result["worker_version"] == "4"
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
    application = ShadowApplication(FakeRecognizer(), store)

    result = application.infer(request_data)

    assert result["agreed"] is False
    assert result["technique"] == "PaddleOCR"
    assert result["worker_version"] == "4"
    assert store.stats()["total"] == 1
    assert store.stats()["disagreed"] == 1
    assert len(list((tmp_path / "disagreements").rglob("*.png"))) == 1


def test_shadow_application_uses_uv_rescue_before_temporal_decision(tmp_path):
    class FakeRecognizer:
        model_name = "fake"

        def predict(self, image):
            return {
                "paddle_raw_text": "UU",
                "paddle_text": "UU",
                "paddle_confidence": 0.995,
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
        "source_path": "/images/W0F00000_134042.tif",
        "source_image": "W0F00000_134042.tif",
        "crop_sha256": hashlib.sha256(png_bytes).hexdigest(),
        "image_png_base64": base64.b64encode(png_bytes).decode("ascii"),
        "current_text": "VV",
        "current_confidence": 0.918,
        "stream_key": "CAPI13|MODEL-A|bottom_left|rot180",
    }
    store = ShadowStore(tmp_path / "shadow.db", tmp_path / "disagreements")
    application = ShadowApplication(FakeRecognizer(), store)

    result = application.infer(request_data)

    assert result["paddle_text"] == "UU"
    assert result["final_text"] == "VV"
    assert result["adoption_reason"] == (
        "forced_char_conversion[pos=1,2;rules=U>V];warmup"
    )
    with sqlite3.connect(store.db_path) as connection:
        saved = connection.execute(
            "SELECT paddle_text, final_text, adoption_reason "
            "FROM mark_shadow_results WHERE id = ?",
            (result["id"],),
        ).fetchone()
    assert saved == (
        "UU",
        "VV",
        "forced_char_conversion[pos=1,2;rules=U>V];warmup",
    )

    request_data["forced_char_conversions"] = []
    disabled_result = application.infer(request_data)
    assert disabled_result["paddle_text"] == "UU"
    assert disabled_result["final_text"] == "UU"
    assert disabled_result["adoption_reason"] == "warmup"


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


def test_shadow_history_reapplies_uv_rescue_to_existing_raw_rows(tmp_path):
    store = ShadowStore(
        tmp_path / "shadow.db",
        tmp_path / "disagreements",
    )
    stream_key = "CAPI13|MODEL-A|bottom_left|rot180"
    with sqlite3.connect(store.db_path) as connection:
        connection.executemany(
            """
            INSERT INTO mark_shadow_results (
                created_at, crop_sha256, current_text, paddle_text,
                valid_two_chars, model_name, stream_key
            ) VALUES (?, ?, ?, ?, 1, 'fake', ?)
            """,
            [
                ("2026-08-25T01:00:00Z", "a", "VV", "UU", stream_key),
                ("2026-08-25T01:00:01Z", "b", "UU", "UU", stream_key),
                ("2026-08-25T01:00:02Z", "c", "V1", "U1", stream_key),
            ],
        )

    assert store.recent_paddle_texts(stream_key) == ["VV", "UU", "V1"]


def test_shadow_store_migrates_existing_database_for_inference_link(tmp_path):
    db_path = tmp_path / "data" / "mark_shadow.db"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE mark_shadow_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                valid_two_chars INTEGER DEFAULT 0,
                agreed INTEGER DEFAULT 0
            )
            """
        )

    ShadowStore(db_path, tmp_path / "data" / "disagreements")

    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(mark_shadow_results)"
            ).fetchall()
        }
    assert "inference_record_id" in columns


def test_mark_shadow_result_can_be_linked_to_exact_inference_record(
    tmp_path,
    monkeypatch,
):
    store = ShadowStore(
        tmp_path / "data" / "mark_shadow.db",
        tmp_path / "data" / "disagreements",
    )
    result_id = _save_shadow_row(
        store,
        current_text="EJ",
        paddle_text="EJ",
        latency_ms=12.5,
    )
    monkeypatch.setattr(capi_mark_shadow, "_DATABASE_PATH", store.db_path)

    linked = capi_mark_shadow.link_mark_shadow_results_to_inference(
        [result_id],
        387220,
    )

    with sqlite3.connect(store.db_path) as connection:
        row = connection.execute(
            "SELECT inference_record_id FROM mark_shadow_results WHERE id = ?",
            (result_id,),
        ).fetchone()
    assert linked == 1
    assert row[0] == 387220


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

    skipped_mark_path = tmp_path / "panel-2" / "W0F00000_101011.tif"
    processed_path = skipped_mark_path.parent / "B0F00000_101011.tif"
    skipped_mark_path.parent.mkdir()
    directory_record_id = db.save_inference_record(
        glass_id="G-DIR-LINK",
        model_id="MODEL-LINK",
        machine_no="M-LINK",
        resolution=(100, 100),
        machine_judgment="OK",
        ai_judgment="OK",
        image_dir=str(skipped_mark_path.parent),
        total_images=1,
        ng_images=0,
        ng_details="[]",
        request_time="2026-07-31 10:10:12",
        response_time="2026-07-31 10:10:13",
        processing_seconds=0.1,
        image_results_data=[
            {"image_path": str(processed_path), "image_name": processed_path.name}
        ],
    )

    assert db.find_inference_record_ids_for_images(
        [(str(skipped_mark_path), skipped_mark_path.name)]
    ) == [directory_record_id]


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
    assert payload["forced_char_conversions"] == [
        {"paddle": "U", "dotmatrix": "V"}
    ]


def _mark_rules_update_handler(*, admin=True):
    class FakeDB:
        def __init__(self):
            self.updated = []

        def update_config_param(
            self,
            param_name,
            new_value,
            reason,
            changed_by="",
        ):
            self.updated.append((param_name, new_value, reason, changed_by))
            return True

        def get_all_config_params(self):
            return []

    handler = object.__new__(CAPIWebHandler)
    handler.db = FakeDB()
    handler.inferencer = None
    handler._capi_server_instance = SimpleNamespace(inferencers={})
    handler._current_settings_user = lambda: {
        "username": "admin" if admin else "operator",
        "can_manage_accounts": admin,
    }
    responses = []
    handler._send_json = (
        lambda payload, status=200, **kwargs: responses.append((status, payload))
    )
    return handler, responses


def test_admin_can_save_mark_forced_char_conversions(monkeypatch):
    handler, responses = _mark_rules_update_handler()
    applied = []
    monkeypatch.setattr(
        capi_web,
        "set_forced_char_conversions",
        lambda value: applied.append(value),
    )
    body = json.dumps({
        "param_name": MARK_FORCED_CHAR_CONVERSIONS_PARAM,
        "new_value": [{"paddle": "0", "dotmatrix": "o"}],
        "reason": "新增 0/O 衝突規則",
    }).encode("utf-8")
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)

    handler._handle_api_settings_update()

    expected = [{"paddle": "0", "dotmatrix": "O"}]
    assert responses[-1][0] == 200
    assert handler.db.updated == [(
        MARK_FORCED_CHAR_CONVERSIONS_PARAM,
        expected,
        "新增 0/O 衝突規則",
        "admin",
    )]
    assert applied == [expected]


def test_non_admin_cannot_save_mark_forced_char_conversions():
    handler, responses = _mark_rules_update_handler(admin=False)
    body = json.dumps({
        "param_name": MARK_FORCED_CHAR_CONVERSIONS_PARAM,
        "new_value": [],
        "reason": "清空規則",
    }).encode("utf-8")
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)

    handler._handle_api_settings_update()

    assert responses[-1] == (
        403,
        {"error": "只有 admin 可以修改 MARK 強制轉換規則"},
    )
    assert handler.db.updated == []


def test_server_restores_mark_forced_char_conversions_from_database():
    from capi_server import CAPIServer

    server = object.__new__(CAPIServer)
    server.db = SimpleNamespace(
        get_config_param=lambda name: {
            "param_name": name,
            "decoded_value": [{"paddle": "8", "dotmatrix": "B"}],
        }
    )

    assert server._load_mark_forced_char_conversions() == [
        {"paddle": "8", "dotmatrix": "B"}
    ]


def test_mark_shadow_admin_api_prefers_persisted_inference_link(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "data" / "mark_shadow.db"
    store = ShadowStore(db_path, tmp_path / "data" / "disagreements")
    result_id = _save_shadow_row(
        store,
        current_text="EJ",
        paddle_text="B1",
        latency_ms=12.5,
    )
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE mark_shadow_results
            SET inference_record_id = 789
            WHERE id = ?
            """,
            (result_id,),
        )
    monkeypatch.setattr(
        CAPIWebHandler,
        "_capi_server_instance",
        SimpleNamespace(
            server_config={"mark_shadow": {"database_path": str(db_path)}}
        ),
    )
    handler = object.__new__(CAPIWebHandler)
    handler.db = SimpleNamespace(
        find_inference_record_ids_for_images=lambda refs: pytest.fail(
            "persisted MARK link should not use path fallback"
        )
    )
    sent = []
    handler._send_json = (
        lambda payload, status=200, **kwargs: sent.append((payload, status))
    )

    handler._handle_api_settings_mark_shadow(
        {"filter": ["disagreed"], "limit": ["50"]}
    )

    payload, status = sent[0]
    assert status == 200
    assert payload["rows"][0]["inference_record_id"] == 789
    assert payload["rows"][0]["record_url"] == "/record/789"


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
    for label in (
        "原辨識",
        "Paddle 辨識",
        "信心",
        "耗時",
        "錯誤",
        "Crop",
        "推論紀錄",
        "row.record_url",
        "推論紀錄寫入中",
        "字元衝突強制轉換",
        "儲存並立即套用",
        "saveMarkForcedConversions",
        "mark_forced_char_conversions",
    ):
        assert label in template
    assert 'path == "/api/settings/mark-shadow"' in web_source
    assert "_require_settings_user(api=True, admin=True)" in web_source


def test_online_client_adds_versions_for_legacy_worker_response(monkeypatch):
    class FakeClient:
        def recognize(self, image, detection, source_path):
            return {
                "success": True,
                "id": 654,
                "paddle_text": "B1",
                "paddle_confidence": 0.94,
                "latency_ms": 18.0,
                "model_name": "PP-OCRv6_medium_rec",
            }

    monkeypatch.setattr(capi_mark_shadow, "_CLIENT", FakeClient())
    capi_mark_shadow.reset_mark_shadow_request_results()

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
    assert capi_mark_shadow.consume_mark_shadow_request_results() == [654]


def test_async_save_links_captured_mark_when_inference_results_are_empty(
    monkeypatch,
):
    from capi_server import CAPIServer

    server = object.__new__(CAPIServer)
    server.db = SimpleNamespace(save_inference_record=lambda **kwargs: 77)
    linked = []
    monkeypatch.setattr(
        capi_mark_shadow,
        "link_mark_shadow_results_to_inference",
        lambda result_ids, record_id: (
            linked.append((list(result_ids), record_id)) or len(result_ids)
        ),
    )

    server._save_results_async(
        client_addr=("127.0.0.1", 12345),
        parsed={
            "glass_id": "G-EMPTY",
            "model_id": "MODEL-A",
            "machine_no": "AOI-1",
            "resolution": (100, 100),
            "machine_judgment": "OK",
            "image_dir": "/images/panel",
            "bomb_info": None,
        },
        results=[],
        ai_judgment="ERR:NO_IMAGES_FOUND",
        ng_details="[]",
        request_time="2026-07-31 10:10:10",
        response_time="2026-07-31 10:10:11",
        processing_seconds=0.1,
        mark_shadow_result_ids=[321],
    )

    assert linked == [([321], 77)]


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
            "id": 321,
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

    detection, regions = inferencer._detect_panel_mark_binary_region(
        [image_path],
        machine_no="CAPI13",
        model_id="MODEL-A",
    )

    assert detection["text"] == "B1"
    assert detection["confidence"] == pytest.approx(0.94)
    assert detection["legacy_text"] == "EJ"
    assert detection["recognition_technique"] == "PaddleOCR"
    assert detection["recognition_fallback"] is False
    assert detection["mark_machine_no"] == "CAPI13"
    assert detection["mark_model_id"] == "MODEL-A"
    assert detection["mark_shadow_result_id"] == 321
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
    assert image_result.mark_shadow_result_id == 321
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
