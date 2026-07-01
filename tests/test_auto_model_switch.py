from pathlib import Path

import pytest


def _register_bundle(db, tmp_path: Path, machine_id: str, name: str) -> int:
    bundle_dir = tmp_path / "model" / name
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "machine_config.yaml").write_text(f"machine_id: {machine_id}\n", encoding="utf-8")
    return db.register_model_bundle({
        "machine_id": machine_id,
        "bundle_path": str(bundle_dir),
        "trained_at": "2026-07-01T10:00:00",
        "panel_count": 1,
        "inner_tile_count": 0,
        "edge_tile_count": 0,
        "ng_tile_count": 0,
        "bundle_size_bytes": 0,
        "job_id": f"job_{machine_id}",
    })


def test_auto_model_switch_rule_crud_and_history(tmp_path):
    from capi_auto_model_switch import DEFAULT_SERIES_PREFIX
    from capi_database import CAPIDatabase

    db = CAPIDatabase(tmp_path / "test.db")
    bundle_id = _register_bundle(db, tmp_path, "GN156HRAAPF0S", "GN156HRAAPF0S-20260701_100000")

    rule = db.upsert_auto_model_switch_rule("GN156HRA", bundle_id, notes="line A")
    assert rule["series_prefix"] == "GN156HRA"
    assert rule["bundle_id"] == bundle_id
    assert rule["notes"] == "line A"

    default_rule = db.upsert_auto_model_switch_rule(DEFAULT_SERIES_PREFIX, bundle_id)
    rules = db.list_auto_model_switch_rules()
    assert [r["series_prefix"] for r in rules[:2]] == [DEFAULT_SERIES_PREFIX, "GN156HRA"]
    assert db.get_default_auto_model_switch_rule()["id"] == default_rule["id"]

    history_id = db.add_auto_model_switch_history({
        "requested_model_id": "GN156HRA9999",
        "series_prefix": "GN156HRA",
        "previous_bundle_id": None,
        "previous_bundle_label": "",
        "target_bundle_id": bundle_id,
        "target_bundle_label": "GN156HRAAPF0S-20260701_100000",
        "action": "switched",
        "status": "success",
        "message": "ok",
    })
    history = db.list_auto_model_switch_history(limit=10)
    assert history[0]["id"] == history_id
    assert history[0]["status"] == "success"


def test_select_target_bundle_match_and_default(tmp_path):
    from capi_auto_model_switch import DEFAULT_SERIES_PREFIX, select_target_bundle
    from capi_database import CAPIDatabase

    db = CAPIDatabase(tmp_path / "test.db")
    mapped_id = _register_bundle(db, tmp_path, "GN156HRAAPF0S", "GN156HRAAPF0S-20260701_100000")
    default_id = _register_bundle(db, tmp_path, "GN140HGAA390S", "GN140HGAA390S-20260701_110000")
    db.upsert_auto_model_switch_rule("GN156HRA", mapped_id)
    db.upsert_auto_model_switch_rule(DEFAULT_SERIES_PREFIX, default_id)

    matched = select_target_bundle(db, "GN156HRA9999")
    assert matched["reason"] == "matched"
    assert matched["series_prefix"] == "GN156HRA"
    assert matched["bundle"]["id"] == mapped_id

    fallback = select_target_bundle(db, "GN999ZZZ0000")
    assert fallback["reason"] == "fallback_default"
    assert fallback["used_default"] is True
    assert fallback["bundle"]["id"] == default_id


def test_series_prefix_validation():
    from capi_auto_model_switch import DEFAULT_SERIES_PREFIX, normalize_series_prefix

    assert normalize_series_prefix("gn156hra") == "GN156HRA"
    assert normalize_series_prefix(DEFAULT_SERIES_PREFIX) == DEFAULT_SERIES_PREFIX
    with pytest.raises(ValueError):
        normalize_series_prefix("GN156")
