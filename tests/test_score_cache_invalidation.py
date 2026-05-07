"""三條 cache invalidation 路徑測試。"""
from pathlib import Path
import pytest
from capi_database import CAPIDatabase


@pytest.fixture
def db_with_pool(tmp_path: Path):
    db = CAPIDatabase(str(tmp_path / "t.db"))
    job_id = "j1"
    rows = []
    for lighting in ("W0F00000", "G0F00000"):
        for zone in ("inner", "edge"):
            rows.append({
                "lighting": lighting, "zone": zone, "source": "ok",
                "source_path": "x", "thumb_path": "x",
            })
    ids = db.insert_tile_pool(job_id, rows)
    # 對 bundle 10 / 11 各寫 4 筆分數
    cache_rows = []
    for bid in (10, 11):
        for tid in ids:
            cache_rows.append({"tile_id": tid, "scoring_bundle_id": bid, "score": 0.5})
    db.insert_score_cache(cache_rows)
    return db, job_id, ids


def test_invalidate_score_cache_helper_by_lighting_zone(db_with_pool):
    from capi_model_registry import invalidate_score_cache
    db, job_id, ids = db_with_pool
    invalidate_score_cache(db, scoring_bundle_id=10,
                            lighting="W0F00000", zone="inner")
    # 只有 bundle=10 + W0F00000+inner 那筆被清
    remaining_10 = db.get_score_cache(10, ids)
    assert len(remaining_10) == 3  # 原 4 筆扣掉 W0F00000+inner 1 筆
    remaining_11 = db.get_score_cache(11, ids)
    assert len(remaining_11) == 4  # bundle 11 原封不動


def test_invalidate_on_delete_training_data(db_with_pool, monkeypatch, tmp_path):
    from capi_model_registry import delete_training_data
    db, job_id, ids = db_with_pool

    # 假裝有對應 bundle row
    bid = db.register_model_bundle({
        "machine_id": "M1", "bundle_path": str(tmp_path / "b"),
        "trained_at": "2026-01-01", "panel_count": 5, "ng_tile_count": 0,
        "inner_tile_count": 4, "edge_tile_count": 0,
        "bundle_size_bytes": 0, "is_active": 0, "job_id": job_id,
    })

    delete_training_data(db, bundle_id=bid)
    # 該 job 的 tile 都從 pool 消失，相關 score cache 也該清
    assert db.get_score_cache(10, ids) == {}
    assert db.get_score_cache(11, ids) == {}


def test_invalidate_on_delete_bundle(db_with_pool, tmp_path, monkeypatch):
    from capi_model_registry import delete_bundle
    db, job_id, ids = db_with_pool

    # delete_bundle 驗證 bundle_path 必須在 server_config_path.parent/model/ 之下
    cfg = tmp_path / "server.yaml"; cfg.write_text("model_configs: []\n")
    bundle_dir = tmp_path / "model" / "b"; bundle_dir.mkdir(parents=True)
    # bundle marker 必須存在
    (bundle_dir / "machine_config.yaml").write_text("threshold_mapping: {}\n")

    bid = db.register_model_bundle({
        "machine_id": "M1", "bundle_path": str(bundle_dir),
        "trained_at": "2026-01-01", "panel_count": 5, "ng_tile_count": 0,
        "inner_tile_count": 4, "edge_tile_count": 0,
        "bundle_size_bytes": 0, "is_active": 0, "job_id": job_id,
    })
    # 對該 bundle 寫一些 score
    db.insert_score_cache([{"tile_id": ids[0], "scoring_bundle_id": bid, "score": 0.7}])

    delete_bundle(db, bundle_id=bid, server_config_path=cfg)

    # 該 bundle 為 scoring_bundle_id 的所有 row 應該被清
    assert db.get_score_cache(bid, [ids[0]]) == {}


def test_invalidate_score_cache_rejects_partial_lighting_zone(db_with_pool):
    """缺一半的 lighting/zone 應該 raise，不能默默變成「清整個 bundle」。"""
    from capi_model_registry import invalidate_score_cache
    db, _, _ = db_with_pool
    with pytest.raises(ValueError):
        invalidate_score_cache(db, scoring_bundle_id=10, lighting="W0F00000")
    with pytest.raises(ValueError):
        invalidate_score_cache(db, scoring_bundle_id=10, zone="inner")
    with pytest.raises(ValueError):
        invalidate_score_cache(db, lighting="W0F00000", zone="inner")  # 沒 bundle_id
