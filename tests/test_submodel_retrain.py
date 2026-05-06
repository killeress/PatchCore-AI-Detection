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
