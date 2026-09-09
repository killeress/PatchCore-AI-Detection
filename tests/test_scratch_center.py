import base64
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import capi_scratch_center as center
from capi_database import CAPIDatabase
from capi_dataset_export import read_manifest, write_manifest
from capi_web import CAPIWebHandler
from tools.merge_over_review_manifests import run as merge
from test_mes_review import _insert_aoi_candidates


def payload(tile_id=1):
    ok, png = cv2.imencode(".png", np.full((512, 512, 3), 90, dtype=np.uint8))
    assert ok
    raw = png.tobytes()
    return {"action": "save", "sample_id": f"manual_{'a' * 32}_{tile_id}",
            "png": base64.b64encode(raw).decode(), "sha256": hashlib.sha256(raw).hexdigest(),
            "metadata": {"glass_id": "PANEL", "image_name": "G0F00000_test.tif"}}


def test_collection_retries_merge_with_legacy_and_withdraw(tmp_path):
    data = payload()
    for _ in range(2):
        center.store_sample(tmp_path, data, "10.174.1.20")
    center.store_sample(tmp_path, payload(2), "10.174.1.20")
    assert len(list(tmp_path.glob("*/manifest.csv"))) == 1
    (tmp_path / "old.png").write_bytes(b"legacy")
    write_manifest(tmp_path / "manifest.csv", {"old": {
        "sample_id": "old", "crop_path": "old.png", "label": "true_ng", "status": "ok",
    }})
    stats = merge(tmp_path, set())
    assert stats["total_rows"] == 3
    assert stats["label_counts"] == {"true_ng": 1, "over_surface_scratch": 2}
    center.store_sample(tmp_path, {"sample_id": data["sample_id"], "action": "remove"}, "10.174.1.20")
    assert merge(tmp_path, set())["total_rows"] == 2
    assert len(list(tmp_path.rglob("*.png"))) == 3  # removal retains evidence


@pytest.mark.parametrize("mutation", [
    {"sample_id": "../../escape"}, {"sha256": "wrong"},
    {"metadata": {"image_name": "B0F00000.tif"}}, {"png": "not-base64"},
])
def test_invalid_transfer_cannot_create_sample(tmp_path, mutation):
    data = payload()
    data.update(mutation)
    with pytest.raises(ValueError):
        center.store_sample(tmp_path, data, "10.174.1.20")
    assert not list(tmp_path.rglob("manifest.csv"))


def test_other_line_cannot_remove_sample(tmp_path):
    data = payload()
    center.store_sample(tmp_path, data, "10.174.1.20")
    with pytest.raises(ValueError, match="其他來源"):
        center.store_sample(tmp_path, {"sample_id": data["sample_id"], "action": "remove"}, "10.174.1.21")
    assert merge(tmp_path, set())["total_rows"] == 1


def make_handler(tmp_path):
    source = tmp_path / "source.png"
    assert cv2.imwrite(str(source), np.full((1100, 1600), 80, np.uint8))
    db = CAPIDatabase(tmp_path / "review.db")
    record_id, _, tile_id = _insert_aoi_candidates(db, str(source))
    handler = object.__new__(CAPIWebHandler)
    handler.db = db
    handler.inferencer = SimpleNamespace(config=SimpleNamespace(inference_rotate_180_enabled=False))
    handler._capi_server_instance = SimpleNamespace(
        server_config={"dataset_export": {"base_dir": str(tmp_path / "dataset")}}, path_mapping={})
    handler._scratch_is_center = lambda: True
    handler._scratch_center_ip = lambda: "10.174.37.81"
    responses = []
    handler._send_json = lambda data, status=200: responses.append((status, data))
    return handler, record_id, tile_id, responses


def test_classification_durable_idempotent_and_remove(tmp_path):
    handler, record_id, tile_id, responses = make_handler(tmp_path)
    data = {"classification": "scratch", "tile_result_id": tile_id}
    handler._handle_record_sample_classification(data)
    assert responses[-1][0] == 200
    receipt = handler.db.get_scratch_sample_classification(tile_id)
    assert receipt and receipt["center_ip"] == "10.174.37.81"
    handler._handle_record_sample_classification(data)
    assert responses[-1][1]["already_exists"] is True
    # Refreshing the detail page can reconstruct the category from SQLite.
    detail = handler.db.get_record_detail(record_id)
    tiles = [tile for image in detail["images"] for tile in image["tiles"]]
    assert next(tile for tile in tiles if tile["id"] == tile_id)["scratch_sample_id"] == receipt["sample_id"]
    handler._handle_record_sample_classification({**data, "classification": "none"})
    assert responses[-1][0] == 200
    assert handler.db.get_scratch_sample_classification(tile_id) is None
    assert merge(tmp_path / "dataset", set())["total_rows"] == 0


def test_center_unavailable_does_not_mark_success(tmp_path, monkeypatch):
    handler, _, tile_id, responses = make_handler(tmp_path)
    handler._scratch_is_center = lambda: False
    def unavailable(*args, **kwargs):
        raise OSError("中心離線")
    monkeypatch.setattr(center, "post_json", unavailable)
    handler._handle_record_sample_classification({"classification": "scratch", "tile_result_id": tile_id})
    assert responses[-1][0] == 500
    assert handler.db.get_scratch_sample_classification(tile_id) is None
    assert not list((tmp_path / "dataset").glob("*/manifest.csv"))


def test_remote_ack_loss_retry_uses_same_source_identity(tmp_path, monkeypatch):
    handler, _, tile_id, responses = make_handler(tmp_path)
    handler._scratch_is_center = lambda: False
    remote = tmp_path / "remote-center"
    requests = []
    def receive(url, data):
        assert url == "http://10.174.37.81/api/scratch/samples"
        requests.append(data)
        sid = center.store_sample(remote, data, "10.174.1.20")
        if len(requests) == 1:
            raise OSError("response lost")
        return {"success": True, "sample_id": sid}
    monkeypatch.setattr(center, "post_json", receive)
    data = {"classification": "scratch", "tile_result_id": tile_id}
    handler._handle_record_sample_classification(data)
    assert responses[-1][0] == 500
    handler._handle_record_sample_classification(data)
    assert responses[-1][0] == 200
    assert requests[0]["sample_id"] == requests[1]["sample_id"]
    assert merge(remote, set())["total_rows"] == 1
    assert not list((tmp_path / "dataset").glob("*/manifest.csv"))


def test_network_retry_preserves_central_relabel(tmp_path):
    data = payload()
    center.store_sample(tmp_path, data, "10.174.1.20")
    manifest = next(tmp_path.glob("*/manifest.csv"))
    rows = read_manifest(manifest)
    rows[data["sample_id"]]["label"] = "true_ng"
    write_manifest(manifest, rows)
    center.store_sample(tmp_path, data, "10.174.1.20")
    assert read_manifest(manifest)[data["sample_id"]]["label"] == "true_ng"


def test_failed_withdraw_keeps_receipt(tmp_path, monkeypatch):
    handler, _, tile_id, responses = make_handler(tmp_path)
    handler._handle_record_sample_classification({"classification": "scratch", "tile_result_id": tile_id})
    handler._scratch_is_center = lambda: False
    monkeypatch.setattr(center, "post_json", lambda *args: (_ for _ in ()).throw(OSError("offline")))
    handler._handle_record_sample_classification({"classification": "none", "tile_result_id": tile_id})
    assert responses[-1][0] == 500
    assert handler.db.get_scratch_sample_classification(tile_id)


def test_receive_rejects_unknown_peer_before_reading_body(tmp_path):
    handler, _, _, responses = make_handler(tmp_path)
    handler.client_address = ("10.174.1.99", 2000)
    handler.headers = {center.TRANSFER_HEADER: "1"}
    handler._scratch_lines = lambda: [{"apiUrl": "http://10.174.1.20/api/status"}]
    handler._handle_scratch_sample_receive()
    assert responses[-1][0] == 403
    handler._handle_scratch_model_receive()
    assert responses[-1][0] == 403


def test_receive_registered_peer(tmp_path):
    handler, _, _, responses = make_handler(tmp_path)
    raw = json.dumps(payload()).encode()
    handler.client_address = ("10.174.1.20", 2000)
    handler.headers = {center.TRANSFER_HEADER: "1", "Content-Length": str(len(raw))}
    handler.rfile = io.BytesIO(raw)
    handler._scratch_lines = lambda: [{"apiUrl": "http://10.174.1.20/api/status"}]
    handler._handle_scratch_sample_receive()
    assert responses[-1][0] == 200
    assert responses[-1][1]["success"]


@pytest.mark.parametrize("valid", [True, False])
def test_model_install_validates_before_switch_and_keeps_old(tmp_path, monkeypatch, valid):
    handler, _, _, _ = make_handler(tmp_path)
    monkeypatch.setattr(center, "__file__", str(tmp_path / "capi_scratch_center.py"))
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    old = deployment / "scratch_classifier_v1.pkl"
    old.write_bytes(b"old")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"base")
    config = SimpleNamespace(scratch_bundle_path=str(old), scratch_classifier_enabled=True,
                             scratch_dinov2_repo_path=str(tmp_path), scratch_dinov2_weights_path=str(weights))
    handler._capi_server_instance.config = config
    handler._capi_server_instance.inferencers = {}
    class Classifier:
        def __init__(self, *args, **kwargs):
            if not valid:
                raise ValueError("bad model")
            self._model = SimpleNamespace(state_dict=lambda: {"lora_A.weight": 1})
        def predict(self, image):
            return 0.1
    monkeypatch.setitem(sys.modules, "scratch_classifier", SimpleNamespace(
        ScratchClassifier=Classifier,
        load_bundle=lambda path: ({"lora_A.weight": 1}, None, SimpleNamespace(conformal_threshold=0.5), None),
    ))
    raw = b"new"
    data = {"name": "scratch_classifier_v1.pkl", "bundle": base64.b64encode(raw).decode(),
            "sha256": hashlib.sha256(raw).hexdigest()}
    if valid:
        result = handler._install_scratch_model(data)
        assert result["success"] and result["previous_bundle"] == str(old)
        assert config.scratch_bundle_path != str(old)
        assert handler.db.get_config_param("scratch_bundle_path")["decoded_value"] == config.scratch_bundle_path
        assert handler.inferencer.config.scratch_bundle_path == config.scratch_bundle_path
    else:
        with pytest.raises(ValueError, match="bad model"):
            handler._install_scratch_model(data)
        assert config.scratch_bundle_path == str(old)
        assert handler.db.get_config_param("scratch_bundle_path") is None
        assert len(list(deployment.glob("*.pkl"))) == 1
    assert old.read_bytes() == b"old"
    assert not list(deployment.glob("*.pending"))


def test_model_hash_mismatch_never_loads_pickle(tmp_path, monkeypatch):
    handler, _, _, _ = make_handler(tmp_path)
    monkeypatch.setitem(sys.modules, "scratch_classifier", SimpleNamespace(ScratchClassifier=None, load_bundle=None))
    with pytest.raises(ValueError, match="校驗"):
        handler._install_scratch_model({"bundle": "bmV3", "sha256": "wrong"})


def test_distribute_requires_admin(tmp_path):
    handler, _, _, responses = make_handler(tmp_path)
    handler._require_settings_user = lambda **kwargs: None
    handler._handle_scratch_distribute()  # does not read/send any model
    assert not responses


def test_distribution_targets_registered_line_only(tmp_path, monkeypatch):
    handler, _, _, responses = make_handler(tmp_path)
    monkeypatch.setattr(center, "__file__", str(tmp_path / "capi_scratch_center.py"))
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    (deployment / "scratch_classifier_v2.pkl").write_bytes(b"model")
    handler._require_settings_user = lambda **kwargs: {"username": "admin"}
    handler._scratch_lines = lambda: [{"id": "line1", "apiUrl": "http://10.174.1.20:8080/api/status"}]
    data = {"bundle": "scratch_classifier_v2.pkl", "line_id": "line1", "url": "http://10.174.9.99"}
    handler._read_json_body = lambda: data
    calls = []
    def post(url, payload, **kwargs):
        calls.append((url, payload))
        return {"success": True, "message": "loaded"}
    monkeypatch.setattr(center, "post_json", post)
    handler._handle_scratch_distribute()
    assert responses[-1][0] == 200
    assert calls[0][0] == "http://10.174.1.20:8080/api/scratch/model"
    assert hashlib.sha256(base64.b64decode(calls[0][1]["bundle"])).hexdigest() == calls[0][1]["sha256"]
    data["line_id"] = "unknown"
    handler._handle_scratch_distribute()
    assert responses[-1][0] == 400 and len(calls) == 1


def test_legacy_root_available_in_gallery(tmp_path):
    handler, _, _, _ = make_handler(tmp_path)
    root = handler._dataset_export_base_dir()
    write_manifest(root / "manifest.csv", {"old": {"sample_id": "old", "status": "ok"}})
    assert "legacy_root" in handler._dataset_list_jobs()
    assert handler._dataset_resolve_job_dir("legacy_root") == root


@pytest.mark.parametrize("batch", [False, True])
def test_gallery_last_sample_delete_hides_batch(tmp_path, batch):
    import threading
    handler, _, _, responses = make_handler(tmp_path)
    root = handler._dataset_export_base_dir()
    data = payload()
    center.store_sample(root, data, "10.174.1.20")
    job = data["sample_id"].rsplit("_", 1)[0]
    handler._dataset_export_state = {"manifest_lock": threading.Lock()}
    handler._read_json_body = lambda: {"job": job, "sample_id": data["sample_id"],
                                       "sample_ids": [data["sample_id"]]}
    assert job in handler._dataset_list_jobs()
    if batch:
        handler._handle_dataset_sample_batch_delete()
    else:
        handler._handle_dataset_sample_delete()
    assert responses[-1][0] == 200
    assert responses[-1][1]["batch_empty"] is True
    assert job not in handler._dataset_list_jobs()
    assert (root / job / "manifest.csv").is_file()
    center.store_sample(root, payload(2), "10.174.1.20")
    assert job in handler._dataset_list_jobs()


def test_gallery_hides_withdrawn_and_empty_legacy_batches(tmp_path):
    handler, _, _, _ = make_handler(tmp_path)
    root = handler._dataset_export_base_dir()
    data = payload()
    center.store_sample(root, data, "10.174.1.20")
    center.store_sample(root, {"sample_id": data["sample_id"], "action": "remove"}, "10.174.1.20")
    write_manifest(root / "manifest.csv", {})
    assert handler._dataset_list_jobs() == []


def test_gallery_stale_batch_selection_falls_back(tmp_path):
    handler, _, _, _ = make_handler(tmp_path)
    root = handler._dataset_export_base_dir()
    write_manifest(root / "empty" / "manifest.csv", {})
    write_manifest(root / "older" / "manifest.csv", {
        "old": {"sample_id": "old", "status": "ok", "label": "true_ng", "prefix": "G0F"},
    })
    rendered = {}
    def render(**kwargs):
        rendered.update(kwargs)
        return "page"
    handler.jinja_env = SimpleNamespace(get_template=lambda name: SimpleNamespace(render=render))
    handler._send_response = lambda *args: None
    handler._handle_dataset_gallery_page({"job": ["empty"], "label": ["over_surface_scratch"]})
    assert rendered["jobs"] == ["older"]
    assert rendered["current_job"] == ""
    assert rendered["current_label"] == ""
    assert rendered["filtered_count"] == 1


def test_activation_db_failure_rolls_back_both_settings(tmp_path):
    import sqlite3
    db = CAPIDatabase(tmp_path / "settings.db")
    db.update_config_param("scratch_bundle_path", "old.pkl")
    db.update_config_param("scratch_classifier_enabled", False)
    with sqlite3.connect(db.db_path) as connection:
        connection.execute("CREATE TRIGGER reject_activation BEFORE INSERT ON config_change_history "
                           "WHEN NEW.param_name = 'scratch_classifier_enabled' "
                           "BEGIN SELECT RAISE(ABORT, 'failed'); END")
    with pytest.raises(sqlite3.IntegrityError):
        db.activate_scratch_bundle("new.pkl", "test")
    assert db.get_config_param("scratch_bundle_path")["decoded_value"] == "old.pkl"
    assert db.get_config_param("scratch_classifier_enabled")["decoded_value"] is False
