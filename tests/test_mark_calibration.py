import hashlib
import io
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import capi_mark_detector
from capi_database import CAPIDatabase
from capi_mark_calibration import (
    build_mark_profile,
    mark_sample_set_sha256,
    run_mark_profile_regression,
)
from capi_mark_detector import (
    build_mark_calibration_prototypes,
    detect_panel_mark,
    set_active_mark_profile,
    set_active_mark_profile_loader,
)


_PATTERNS = {
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "J": ("01111", "01111", "00110", "00110", "00110", "11110", "11100"),
}


def _multipart_mark_handler(fields, upload=None):
    from capi_web import CAPIWebHandler

    boundary = "----CapiMarkCalibrationBoundary"
    parts = []
    for name, value in fields.items():
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                ).encode(),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    if upload is not None:
        filename, content = upload
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    'Content-Disposition: form-data; name="file"; '
                    f'filename="{filename}"\r\n'
                ).encode(),
                b"Content-Type: image/png\r\n\r\n",
                content,
                b"\r\n",
            ]
        )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    handler = object.__new__(CAPIWebHandler)
    handler.headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    handler.rfile = io.BytesIO(body)
    return handler


def _draw_mark(image, text, x=790, y=150, cell=10, gap=8, radius=3):
    cursor_x = x
    for char in text:
        for row_idx, row in enumerate(_PATTERNS[char]):
            for col_idx, value in enumerate(row):
                if value == "1":
                    cv2.circle(
                        image,
                        (cursor_x + col_idx * cell, y + row_idx * cell),
                        radius,
                        45,
                        -1,
                    )
        cursor_x += 5 * cell + gap


@pytest.fixture(autouse=True)
def _reset_active_profile():
    set_active_mark_profile_loader(None)
    set_active_mark_profile(None, 0)
    yield
    set_active_mark_profile_loader(None)
    set_active_mark_profile(None, 0)


def _reviewed_sample(db, image_path, expected_text="E1"):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    baseline = detect_panel_mark(
        image,
        profile={"schema_version": 1, "prototypes": []},
        profile_id=1,
    )
    assert baseline["text"] == "EJ"
    prototypes = build_mark_calibration_prototypes(baseline, expected_text)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    sample = db.add_mark_calibration_sample(
        {
            "file_sha256": digest,
            "image_path": str(image_path),
            "original_filename": image_path.name,
            "expected_text": expected_text,
            "original_text": baseline["text"],
            "original_confidence": baseline["confidence"],
            "original_roi": baseline["roi"],
            "original_orientation": baseline["orientation"],
            "original_bbox": baseline["bbox"],
            "prototypes": prototypes,
            "rotation_applied": False,
            "profile_id_before": 1,
            "created_by": "admin",
            "reason": "測試誤判校正",
        }
    )
    return baseline, sample


def test_single_reviewed_sample_can_correct_without_waiting_for_more_samples(tmp_path):
    image = np.full((768, 1024), 160, dtype=np.uint8)
    _draw_mark(image, "EJ")
    image_path = tmp_path / "W0F00000_test.png"
    assert cv2.imwrite(str(image_path), image)

    db = CAPIDatabase(tmp_path / "mark.db")
    baseline, sample = _reviewed_sample(db, image_path)
    samples = db.list_mark_calibration_samples()
    profile = build_mark_profile(samples)

    candidate = detect_panel_mark(
        image,
        profile=profile,
        profile_id=2,
    )

    assert baseline["text"] == "EJ"
    assert candidate["text"] == "E1"
    assert candidate["profile_version"] == 2
    assert len(profile["prototypes"]) == 1
    assert profile["prototypes"][0]["position"] == 1
    assert profile["prototypes"][0]["sample_id"] == sample["id"]


def test_profile_regression_activation_and_rollback_are_versioned(tmp_path):
    image = np.full((768, 1024), 160, dtype=np.uint8)
    _draw_mark(image, "EJ")
    image_path = tmp_path / "W0F00000_test.png"
    assert cv2.imwrite(str(image_path), image)

    db = CAPIDatabase(tmp_path / "mark.db")
    _, sample = _reviewed_sample(db, image_path)
    samples = db.list_mark_calibration_samples()
    profile_data = build_mark_profile(samples)
    candidate = db.create_mark_profile(
        profile_data,
        parent_profile_id=1,
        sample_count=1,
        sample_set_sha256=mark_sample_set_sha256(samples),
        created_by="admin",
        reason="修正 EJ 為 E1",
        triggering_sample_id=sample["id"],
    )

    regression = run_mark_profile_regression(
        samples,
        profile_data,
        profile_id=candidate["id"],
    )
    assert regression["total"] == 1
    assert regression["passed"] == 1
    assert regression["failed"] == 0
    assert regression["success"] is True
    assert regression["failures"] == []

    active = db.finalize_mark_profile(
        candidate["id"],
        regression,
        activate=True,
    )
    assert active["status"] == "active"
    assert db.get_active_mark_profile()["id"] == candidate["id"]
    assert sum(p["status"] == "active" for p in db.list_mark_profiles()) == 1
    mirror_db = CAPIDatabase(tmp_path / "mark.db")
    set_active_mark_profile_loader(mirror_db.get_active_mark_profile)
    hot_loaded = detect_panel_mark(image)
    assert hot_loaded["profile_version"] == active["id"]
    assert hot_loaded["text"] == "E1"

    baseline_profile = db.get_mark_profile(1)
    rollback_regression = run_mark_profile_regression(
        samples,
        baseline_profile["profile"],
        profile_id=baseline_profile["id"],
    )
    assert rollback_regression["failed"] == 1
    with pytest.raises(ValueError, match="已知回歸失敗"):
        db.rollback_mark_profile(
            1,
            expected_active_profile_id=active["id"],
            regression_report=rollback_regression,
            sample_count=len(samples),
            sample_set_sha256=mark_sample_set_sha256(samples),
            allow_known_regressions=False,
            changed_by="admin",
            reason="不可靜默回滾",
        )
    rolled_back = db.rollback_mark_profile(
        1,
        expected_active_profile_id=active["id"],
        regression_report=rollback_regression,
        sample_count=len(samples),
        sample_set_sha256=mark_sample_set_sha256(samples),
        allow_known_regressions=True,
        changed_by="admin",
        reason="緊急恢復內建模板",
    )
    assert rolled_back["status"] == "active"
    assert rolled_back["rollback_of_profile_id"] == 1
    assert rolled_back["profile"]["prototypes"] == []
    assert rolled_back["regression_failed"] == 1
    assert sum(p["status"] == "active" for p in db.list_mark_profiles()) == 1
    profile_count = len(db.list_mark_profiles())
    same_rollback = db.rollback_mark_profile(
        1,
        expected_active_profile_id=rolled_back["id"],
        regression_report=rollback_regression,
        sample_count=len(samples),
        sample_set_sha256=mark_sample_set_sha256(samples),
        allow_known_regressions=True,
        changed_by="admin",
        reason="重送相同回滾",
    )
    assert same_rollback["id"] == rolled_back["id"]
    assert len(db.list_mark_profiles()) == profile_count
    after_rollback = detect_panel_mark(image)
    assert after_rollback["profile_version"] == rolled_back["id"]
    assert after_rollback["text"] == "EJ"


def test_missing_historical_image_blocks_activation(tmp_path):
    image = np.full((768, 1024), 160, dtype=np.uint8)
    _draw_mark(image, "EJ")
    image_path = tmp_path / "W0F00000_test.png"
    assert cv2.imwrite(str(image_path), image)

    db = CAPIDatabase(tmp_path / "mark.db")
    _, sample = _reviewed_sample(db, image_path)
    samples = db.list_mark_calibration_samples()
    profile_data = build_mark_profile(samples)
    candidate = db.create_mark_profile(
        profile_data,
        parent_profile_id=1,
        sample_count=1,
        sample_set_sha256=mark_sample_set_sha256(samples),
        created_by="admin",
        reason="缺圖回歸測試",
        triggering_sample_id=sample["id"],
    )
    image_path.unlink()

    regression = run_mark_profile_regression(
        samples,
        profile_data,
        profile_id=candidate["id"],
    )

    assert regression["success"] is False
    assert regression["failed"] == 1
    assert regression["failures"][0]["sample_id"] == sample["id"]
    assert regression["failures"][0]["reason"] == "圖片檔案不存在"
    with pytest.raises(ValueError, match="未全數通過"):
        db.finalize_mark_profile(
            candidate["id"],
            regression,
            activate=True,
        )
    rejected = db.finalize_mark_profile(
        candidate["id"],
        regression,
        activate=False,
    )
    assert rejected["status"] == "rejected"
    assert db.get_active_mark_profile()["id"] == 1


def test_changed_historical_image_hash_blocks_regression(tmp_path):
    image = np.full((768, 1024), 160, dtype=np.uint8)
    _draw_mark(image, "EJ")
    image_path = tmp_path / "W0F00000_test.png"
    assert cv2.imwrite(str(image_path), image)

    db = CAPIDatabase(tmp_path / "mark.db")
    _reviewed_sample(db, image_path)
    samples = db.list_mark_calibration_samples()
    profile_data = build_mark_profile(samples)
    image_path.write_bytes(image_path.read_bytes() + b"changed")

    regression = run_mark_profile_regression(
        samples,
        profile_data,
        profile_id=2,
    )

    assert regression["success"] is False
    assert regression["failed"] == 1
    assert regression["failures"][0]["reason"] == "圖片 SHA-256 與標註紀錄不符"


def test_partial_regression_report_cannot_activate(tmp_path):
    image = np.full((768, 1024), 160, dtype=np.uint8)
    _draw_mark(image, "EJ")
    first_path = tmp_path / "W0F00000_first.png"
    second_path = tmp_path / "W0F00000_second.png"
    assert cv2.imwrite(str(first_path), image)
    image[0, 0] = 159
    assert cv2.imwrite(str(second_path), image)

    db = CAPIDatabase(tmp_path / "mark.db")
    _, first = _reviewed_sample(db, first_path)
    _reviewed_sample(db, second_path)
    samples = db.list_mark_calibration_samples()
    profile_data = build_mark_profile(samples)
    sample_set_sha256 = mark_sample_set_sha256(samples)
    candidate = db.create_mark_profile(
        profile_data,
        parent_profile_id=1,
        sample_count=2,
        sample_set_sha256=sample_set_sha256,
        created_by="admin",
        reason="部分報告不得啟用",
        triggering_sample_id=first["id"],
    )

    partial_report = {
        "profile_id": candidate["id"],
        "total": 1,
        "passed": 1,
        "failed": 0,
        "success": True,
        "sample_set_sha256": sample_set_sha256,
        "failures": [],
    }
    with pytest.raises(ValueError, match="未全數通過"):
        db.finalize_mark_profile(
            candidate["id"],
            partial_report,
            activate=True,
        )


def test_duplicate_image_cannot_create_conflicting_ground_truth(tmp_path):
    image = np.full((768, 1024), 160, dtype=np.uint8)
    _draw_mark(image, "EJ")
    image_path = tmp_path / "W0F00000_test.png"
    assert cv2.imwrite(str(image_path), image)

    db = CAPIDatabase(tmp_path / "mark.db")
    _, sample = _reviewed_sample(db, image_path)
    duplicate = dict(sample)
    duplicate["expected_text"] = "E2"
    duplicate["original_bbox"] = duplicate.pop("bbox")
    duplicate["prototypes"] = duplicate["prototypes"]

    with pytest.raises(ValueError, match="已經保存過校正"):
        db.add_mark_calibration_sample(duplicate)


def test_identical_character_features_cannot_have_conflicting_labels():
    densities = [[0.5] * 5 for _ in range(7)]
    samples = [
        {
            "id": 1,
            "prototypes": [
                {"char": "1", "position": 1, "densities": densities}
            ],
        },
        {
            "id": 2,
            "prototypes": [
                {"char": "2", "position": 1, "densities": densities}
            ],
        },
    ]

    with pytest.raises(ValueError, match="標註衝突"):
        build_mark_profile(samples)


def test_database_atomically_rejects_concurrent_conflicting_samples(tmp_path):
    db_path = tmp_path / "mark.db"
    first_db = CAPIDatabase(db_path)
    second_db = CAPIDatabase(db_path)
    densities = [[0.5] * 5 for _ in range(7)]
    start = threading.Barrier(2)
    successes = []
    errors = []

    def insert(db, index, expected_char):
        image_path = tmp_path / f"W0F00000_{index}.img"
        image_path.write_bytes(f"image-{index}".encode())
        payload = {
            "file_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "image_path": str(image_path),
            "original_filename": image_path.name,
            "expected_text": f"E{expected_char}",
            "original_text": "EJ",
            "prototypes": [
                {
                    "char": expected_char,
                    "position": 1,
                    "densities": densities,
                }
            ],
            "created_by": "admin",
            "reason": "併發衝突測試",
        }
        start.wait()
        try:
            successes.append(db.add_mark_calibration_sample(payload))
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=insert, args=(first_db, 1, "1")),
        threading.Thread(target=insert, args=(second_db, 2, "2")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(successes) == 1
    assert len(errors) == 1
    assert "標註衝突" in str(errors[0])
    samples = first_db.list_mark_calibration_samples()
    assert len(samples) == 1
    assert build_mark_profile(samples)["prototypes"]


def test_slow_profile_refresh_cannot_overwrite_newer_active_profile():
    loader_started = threading.Event()
    release_loader = threading.Event()
    stale_profile = {"id": 2, "profile": {"schema_version": 1, "prototypes": []}}

    def stale_loader():
        loader_started.set()
        assert release_loader.wait(timeout=2)
        return stale_profile

    set_active_mark_profile(None, 1)
    set_active_mark_profile_loader(stale_loader)
    result = {}

    def refresh():
        result["snapshot"] = capi_mark_detector._active_mark_profile_snapshot()

    thread = threading.Thread(target=refresh)
    thread.start()
    assert loader_started.wait(timeout=2)
    cached_result = {}
    cached_thread = threading.Thread(
        target=lambda: cached_result.setdefault(
            "snapshot",
            capi_mark_detector._active_mark_profile_snapshot(),
        )
    )
    cached_thread.start()
    cached_thread.join(timeout=0.5)
    assert not cached_thread.is_alive()
    assert cached_result["snapshot"][1] == 1
    set_active_mark_profile(None, 3)
    release_loader.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert result["snapshot"][1] == 3


def test_mark_calibration_schema_survives_database_reopen(tmp_path):
    db_path = tmp_path / "mark.db"
    first = CAPIDatabase(db_path)
    assert first.get_active_mark_profile()["id"] == 1

    reopened = CAPIDatabase(db_path)
    profiles = reopened.list_mark_profiles()

    assert len(profiles) == 1
    assert profiles[0]["status"] == "active"
    assert json.dumps(profiles[0]["profile"], sort_keys=True)


def test_mark_correction_form_accepts_absolute_server_image_path(tmp_path):
    source_path = tmp_path / "W0F00000_path_source.tif"
    source_path.write_bytes(b"path-image-bytes")
    handler = _multipart_mark_handler(
        {
            "image_path": source_path,
            "correct_text": "b1",
            "reason": "伺服器路徑測試",
        }
    )

    filename, file_data, expected_text, reason = (
        handler._read_mark_correction_form()
    )

    assert filename == source_path.name
    assert file_data == b"path-image-bytes"
    assert expected_text == "B1"
    assert reason == "伺服器路徑測試"


def test_mark_correction_form_requires_exactly_one_image_source(tmp_path):
    source_path = tmp_path / "W0F00000_path_source.png"
    source_path.write_bytes(b"path-image-bytes")
    both_handler = _multipart_mark_handler(
        {
            "image_path": source_path,
            "correct_text": "B1",
            "reason": "來源衝突測試",
        },
        upload=("W0F00000_upload.png", b"upload-image-bytes"),
    )
    with pytest.raises(ValueError, match="只能擇一"):
        both_handler._read_mark_correction_form()

    relative_handler = _multipart_mark_handler(
        {
            "image_path": "W0F00000_relative.png",
            "correct_text": "B1",
            "reason": "相對路徑測試",
        }
    )
    with pytest.raises(ValueError, match="絕對圖片文件路徑"):
        relative_handler._read_mark_correction_form()


def test_mark_calibration_tab_is_removed_from_settings_ui():
    html = Path("templates/settings.html").read_text(encoding="utf-8")

    assert 'data-target="mark-calibration"' not in html
    assert "switchTab('mark-calibration')" not in html
    assert "${renderMarkCalibrationPane()}" not in html


def test_admin_correction_handler_runs_regression_and_auto_activates(tmp_path):
    from capi_web import CAPIWebHandler

    image = np.full((768, 1024), 160, dtype=np.uint8)
    _draw_mark(image, "EJ")
    ok, encoded = cv2.imencode(".tif", image)
    assert ok

    db = CAPIDatabase(tmp_path / "mark.db")
    handler = object.__new__(CAPIWebHandler)
    handler.db = db
    handler.inferencer = SimpleNamespace(
        config=SimpleNamespace(inference_rotate_180_enabled=False)
    )
    handler._read_mark_correction_form = lambda: (
        "W0F00000_test.tif",
        encoded.tobytes(),
        "A1",
        "現場人工確認",
    )
    handler._current_settings_user = lambda: {
        "username": "admin",
        "can_manage_accounts": True,
    }
    responses = []
    handler._send_json = lambda payload, status=200, headers=None: responses.append(
        (status, payload)
    )
    CAPIWebHandler._mark_calibration_lock = threading.Lock()

    handler._handle_api_settings_mark_correct()

    assert responses[-1][0] == 200
    assert responses[-1][1]["activated"] is True
    assert responses[-1][1]["regression"]["passed"] == 1
    assert db.get_active_mark_profile()["id"] == 2
    assert db.list_mark_calibration_samples()[0]["created_by"] == "admin"

    handler._read_mark_correction_form = lambda: (
        "W0F00000_test.tif",
        encoded.tobytes(),
        "A2",
        "更正先前人工答案",
    )
    handler._handle_api_settings_mark_correct()

    assert responses[-1][0] == 200
    assert responses[-1][1]["revised"] is True
    assert responses[-1][1]["activated"] is True
    revised = db.list_mark_calibration_samples()[0]
    assert revised["expected_text"] == "A2"
    assert revised["revision_count"] == 1
    assert revised["updated_by"] == "admin"
    assert {item["position"] for item in revised["prototypes"]} == {0, 1}
    conn = db._get_conn()
    try:
        revision = conn.execute(
            "SELECT * FROM mark_calibration_sample_revisions"
        ).fetchone()
        stored_image_path = conn.execute(
            "SELECT image_path FROM mark_calibration_samples"
        ).fetchone()[0]
    finally:
        conn.close()
    assert revision["previous_expected_text"] == "A1"
    assert revision["new_expected_text"] == "A2"
    assert not Path(stored_image_path).is_absolute()
    assert Path(stored_image_path).suffix == ".img"

    profile_count = len(db.list_mark_profiles())
    handler._handle_api_settings_mark_correct()
    assert responses[-1][0] == 200
    assert responses[-1][1]["already_applied"] is True
    assert len(db.list_mark_profiles()) == profile_count


def test_regression_exception_rejects_candidate_profile(tmp_path, monkeypatch):
    import capi_mark_calibration
    from capi_web import CAPIWebHandler

    image = np.full((768, 1024), 160, dtype=np.uint8)
    _draw_mark(image, "EJ")
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    db = CAPIDatabase(tmp_path / "mark.db")
    handler = object.__new__(CAPIWebHandler)
    handler.db = db
    handler.inferencer = SimpleNamespace(
        config=SimpleNamespace(inference_rotate_180_enabled=False)
    )
    handler._read_mark_correction_form = lambda: (
        "W0F00000_test.png",
        encoded.tobytes(),
        "E1",
        "回歸例外測試",
    )
    handler._current_settings_user = lambda: {"username": "admin"}
    responses = []
    handler._send_json = lambda payload, status=200, headers=None: responses.append(
        (status, payload)
    )
    CAPIWebHandler._mark_calibration_lock = threading.Lock()

    def fail_regression(*args, **kwargs):
        raise RuntimeError("simulated regression failure")

    monkeypatch.setattr(
        capi_mark_calibration,
        "run_mark_profile_regression",
        fail_regression,
    )
    handler._handle_api_settings_mark_correct()

    assert responses[-1][0] == 409
    statuses = [profile["status"] for profile in db.list_mark_profiles()]
    assert statuses == ["rejected", "active"]


def test_rollback_handler_requires_explicit_force_and_is_idempotent(tmp_path):
    from capi_web import CAPIWebHandler

    image = np.full((768, 1024), 160, dtype=np.uint8)
    _draw_mark(image, "EJ")
    image_path = tmp_path / "W0F00000_test.png"
    assert cv2.imwrite(str(image_path), image)

    db = CAPIDatabase(tmp_path / "mark.db")
    _, sample = _reviewed_sample(db, image_path)
    samples = db.list_mark_calibration_samples()
    profile_data = build_mark_profile(samples)
    candidate = db.create_mark_profile(
        profile_data,
        parent_profile_id=1,
        sample_count=1,
        sample_set_sha256=mark_sample_set_sha256(samples),
        created_by="admin",
        reason="建立回滾測試版",
        triggering_sample_id=sample["id"],
    )
    regression = run_mark_profile_regression(
        samples,
        profile_data,
        profile_id=candidate["id"],
    )
    db.finalize_mark_profile(candidate["id"], regression, activate=True)

    handler = object.__new__(CAPIWebHandler)
    handler.db = db
    handler._current_settings_user = lambda: {"username": "admin"}
    payload = {
        "profile_id": 1,
        "reason": "緊急回滾測試",
        "allow_known_regressions": False,
    }
    handler._read_json_body = lambda: dict(payload)
    responses = []
    handler._send_json = lambda body, status=200, headers=None: responses.append(
        (status, body)
    )
    CAPIWebHandler._mark_calibration_lock = threading.Lock()

    handler._handle_api_settings_mark_rollback()
    assert responses[-1][0] == 409
    assert responses[-1][1]["requires_force"] is True
    assert db.get_active_mark_profile()["id"] == candidate["id"]

    payload["allow_known_regressions"] = True
    handler._handle_api_settings_mark_rollback()
    assert responses[-1][0] == 200
    assert responses[-1][1]["forced"] is True
    rollback_id = db.get_active_mark_profile()["id"]
    profile_count = len(db.list_mark_profiles())

    handler._handle_api_settings_mark_rollback()
    assert responses[-1][0] == 200
    assert responses[-1][1]["already_applied"] is True
    assert db.get_active_mark_profile()["id"] == rollback_id
    assert len(db.list_mark_profiles()) == profile_count
