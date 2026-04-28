"""測試 capi_model_registry 模型庫 CRUD 功能。"""
import tempfile
from pathlib import Path


def test_list_bundles_grouped(tmp_path):
    from capi_database import CAPIDatabase
    from capi_model_registry import list_bundles_grouped

    db = CAPIDatabase(tmp_path / "test.db")
    db.register_model_bundle({
        "machine_id": "GN160", "bundle_path": "model/GN160-20260428",
        "trained_at": "2026-04-28T15:30:45", "panel_count": 5,
        "inner_tile_count": 2400, "edge_tile_count": 900,
        "ng_tile_count": 150, "bundle_size_bytes": 478_000_000, "job_id": "j1",
    })
    db.register_model_bundle({
        "machine_id": "OTHER", "bundle_path": "model/OTHER-20260420",
        "trained_at": "2026-04-20T10:00:00", "panel_count": 3,
        "inner_tile_count": 1200, "edge_tile_count": 400,
        "ng_tile_count": 100, "bundle_size_bytes": 300_000_000, "job_id": "j2",
    })

    grouped = list_bundles_grouped(db)
    assert "GN160" in grouped
    assert "OTHER" in grouped
    assert len(grouped["GN160"]) == 1
