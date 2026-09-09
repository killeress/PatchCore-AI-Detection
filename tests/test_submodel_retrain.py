"""capi_model_registry 與 capi_train_new 中與單子模型重訓相關的純函式測試。

無需啟動 web server / GPU；用 tempdir 與假 DB 物件做 isolated 測試。
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_model_registry import (
    append_submodel_history,
    get_used_tile_ids,
    get_pending_change_count,
)


def _write_manifest(bundle_dir: Path, data: dict) -> None:
    (bundle_dir / "manifest.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8",
    )


def test_append_submodel_history_creates_field(tmp_path):
    _write_manifest(tmp_path, {"machine_id": "M1"})

    entry = {"trained_at": "2026-05-06T10:00:00", "tile_count_used": 100,
             "auroc": 0.95, "used_tile_ids": [1, 2, 3], "kind": "retrain"}
    append_submodel_history(tmp_path, "G0F00000", "edge", entry)

    data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert data["submodel_history"]["G0F00000-edge"] == [entry]
    assert data["last_retrained_at"] == "2026-05-06T10:00:00"


def test_append_submodel_history_appends_existing(tmp_path):
    initial = {"trained_at": "2026-05-01T10:00:00", "tile_count_used": 100,
               "auroc": 0.93, "used_tile_ids": [1, 2], "kind": "initial"}
    _write_manifest(tmp_path, {
        "submodel_history": {"G0F00000-edge": [initial]},
    })

    new_entry = {"trained_at": "2026-05-06T10:00:00", "tile_count_used": 95,
                 "auroc": 0.96, "used_tile_ids": [1, 3], "kind": "retrain"}
    append_submodel_history(tmp_path, "G0F00000", "edge", new_entry)

    data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    history = data["submodel_history"]["G0F00000-edge"]
    assert len(history) == 2
    assert history[1] == new_entry


def test_get_used_tile_ids_from_history(tmp_path):
    _write_manifest(tmp_path, {
        "submodel_history": {
            "G0F00000-edge": [
                {"used_tile_ids": [1, 2]},
                {"used_tile_ids": [1, 3, 5]},
            ]
        }
    })
    assert get_used_tile_ids(tmp_path, "G0F00000", "edge") == {1, 3, 5}


def test_get_used_tile_ids_fallback_to_unit_metrics(tmp_path):
    _write_manifest(tmp_path, {
        "unit_metrics": {
            "G0F00000-edge": {"used_tile_ids": [10, 20]},
        }
    })
    assert get_used_tile_ids(tmp_path, "G0F00000", "edge") == {10, 20}


def test_get_used_tile_ids_none_when_missing(tmp_path):
    _write_manifest(tmp_path, {"machine_id": "M1"})
    assert get_used_tile_ids(tmp_path, "G0F00000", "edge") is None


def test_pending_change_count_diff(tmp_path):
    _write_manifest(tmp_path, {
        "submodel_history": {
            "G0F00000-edge": [{"used_tile_ids": [1, 2, 3]}],
        }
    })
    db = MagicMock()
    # 目前 accept = {1, 2, 4}：相比上次 {1, 2, 3} 差異是 {3, 4}（2 張）
    db.list_tile_pool.return_value = [{"id": 1}, {"id": 2}, {"id": 4}]

    bundle = {"job_id": "j1", "bundle_path": str(tmp_path)}
    assert get_pending_change_count(db, bundle, "G0F00000", "edge") == 2


def test_pending_change_count_legacy_uses_reject_count(tmp_path):
    """舊 bundle 沒有 used_tile_ids → 退化用 reject 數量。"""
    _write_manifest(tmp_path, {"machine_id": "M1"})
    db = MagicMock()

    def fake_list(job_id, **filters):
        if filters.get("decision") == "reject":
            return [{"id": 5}, {"id": 6}]
        return [{"id": 1}, {"id": 2}]
    db.list_tile_pool.side_effect = fake_list

    bundle = {"job_id": "j1", "bundle_path": str(tmp_path)}
    assert get_pending_change_count(db, bundle, "G0F00000", "edge") == 2


def test_pending_change_count_no_job_id(tmp_path):
    db = MagicMock()
    bundle = {"job_id": "", "bundle_path": str(tmp_path)}
    assert get_pending_change_count(db, bundle, "G0F00000", "edge") == 0


@pytest.fixture
def partial_source(tmp_path):
    _write_manifest(tmp_path, {
        "submodel_history": {"WGF50500-inner": [
            {"job_id": "new", "used_tile_ids": [2]},
        ]},
        "unit_metrics": {"WGF50500-edge": {"used_tile_ids": [3]}},
    })
    rows = [
        dict(id=1, job_id="old", lighting="WGF50500", zone="inner", source="ok", decision="accept"),
        dict(id=2, job_id="new", lighting="WGF50500", zone="inner", source="ok", decision="accept"),
        dict(id=3, job_id="old", lighting="WGF50500", zone="edge", source="ok", decision="accept"),
        dict(id=4, job_id="new", lighting="WGF50500", zone="edge", source="ok", decision="accept"),
    ]
    db = MagicMock()
    db.list_tile_pool.side_effect = lambda job, **filters: [
        r for r in rows if r["job_id"] == job and all(r.get(k) == v for k, v in filters.items())
    ]
    bundle = dict(id=13, job_id="old", bundle_path=str(tmp_path))
    db.get_model_bundle.return_value = bundle
    return db, bundle, rows


def test_partial_source_list_and_repeated_retrain(partial_source):
    from capi_model_registry import list_bundle_training_tiles, get_submodel_job_id
    db, bundle, rows = partial_source
    assert [t["id"] for t in list_bundle_training_tiles(db, bundle, source="ok")] == [2, 3]
    assert get_submodel_job_id(bundle, "WGF50500", "inner") == "new"
    assert get_submodel_job_id(bundle, "WGF50500", "edge") == "old"
    rows[1]["decision"] = "reject"
    assert get_pending_change_count(db, bundle, "WGF50500", "inner") == 1
    append_submodel_history(Path(bundle["bundle_path"]), "WGF50500", "inner", {
        "kind": "retrain", "job_id": "new", "used_tile_ids": [],
    })
    assert get_pending_change_count(db, bundle, "WGF50500", "inner") == 0
    rows[1]["decision"] = "accept"
    assert get_pending_change_count(db, bundle, "WGF50500", "inner") == 1
    assert get_pending_change_count(db, bundle, "WGF50500", "edge") == 0


def test_partial_source_decision_api(partial_source):
    import io
    from capi_web import CAPIWebHandler
    db, bundle, rows = partial_source
    h = CAPIWebHandler.__new__(CAPIWebHandler)
    h._capi_server_instance = MagicMock(database=db)
    h.path = "/api/models/13/tiles/decision"
    h._send_json = MagicMock()
    for ids, status in [([2, 3], 200), ([1], 400), ([4], 400)]:
        body = json.dumps(dict(tile_ids=ids, decision="reject")).encode()
        h.headers = {"Content-Length": str(len(body))}
        h.rfile = io.BytesIO(body)
        h._handle_models_tiles_decision()
        assert h._send_json.call_args.kwargs.get("status", 200) == status
    assert db.update_tile_decisions.call_count == 2
    db.update_tile_decisions.assert_any_call("new", [2], "reject")
    db.update_tile_decisions.assert_any_call("old", [3], "reject")



def test_retrain_worker_keeps_partial_source(partial_source, monkeypatch):
    import threading
    import capi_train_new
    import capi_model_registry
    from capi_web import CAPIWebHandler
    db, bundle, rows = partial_source
    bundle["machine_id"] = "M1"
    validation = {"split_mode": "auto_panel", "panels": {}}
    db.get_training_job.return_value = {"training_params": {"validation_config": validation}}
    train = MagicMock(return_value={
        "metrics": {"auroc": 0.9}, "tile_count": 1, "used_tile_ids": [2],
        "ng_used": "none", "ng_count": 0, "size_bytes": 10,
    })
    monkeypatch.setattr(capi_train_new, "train_single_submodel", train)
    monkeypatch.setattr(capi_model_registry, "invalidate_score_cache", lambda *a, **k: 0)
    state = {"lock": threading.Lock(), "job": {"log_lines": []}}
    monkeypatch.setattr(CAPIWebHandler, "_submodel_retrain_state", state)
    h = CAPIWebHandler.__new__(CAPIWebHandler)
    h._capi_server_instance = MagicMock(database=db, inferencers={})
    h._submodel_retrain_worker(13, "WGF50500", "inner")
    assert state["job"]["state"] == "completed"
    assert train.call_args.kwargs["job_id"] == "new"
    assert train.call_args.kwargs["cfg"].validation_config == validation
    manifest = json.loads((Path(bundle["bundle_path"]) / "manifest.json").read_text())
    assert manifest["submodel_history"]["WGF50500-inner"][-1]["job_id"] == "new"



def test_partial_source_tiles_api(partial_source):
    from capi_web import CAPIWebHandler
    db, bundle, rows = partial_source
    h = CAPIWebHandler.__new__(CAPIWebHandler)
    h._capi_server_instance = MagicMock(database=db)
    h._send_json = MagicMock()
    h._train_new_thumb_url = lambda path: path
    for query, expected in [("", [2, 3]), ("&zone=inner", [2]), ("&zone=edge", [3])]:
        h.path = "/api/models/13/training_tiles?source=ok&lighting=WGF50500" + query
        h._handle_models_training_tiles()
        data = h._send_json.call_args.args[0]
        assert [t["id"] for t in data["tiles"]] == expected
        assert data["total"] == len(expected)


def test_partial_source_self_scan(partial_source, monkeypatch):
    import io
    from capi_web import CAPIWebHandler
    db, bundle, rows = partial_source
    db.get_score_cache.return_value = {}
    start = MagicMock(return_value=(True, {"state": "running"}))
    monkeypatch.setattr(CAPIWebHandler, "_start_scan_job", start)
    h = CAPIWebHandler.__new__(CAPIWebHandler)
    h._capi_server_instance = MagicMock(database=db)
    h._send_json = MagicMock()
    h.path = "/api/models/13/self_scan"
    body = json.dumps(dict(lighting="WGF50500", zone="inner")).encode()
    h.headers = {"Content-Length": str(len(body))}
    h.rfile = io.BytesIO(body)
    h._handle_scan_self_score()
    assert start.call_args.kwargs["tile_pool_job_id"] == "new"
    assert start.call_args.kwargs["tile_ids"] == [2]
