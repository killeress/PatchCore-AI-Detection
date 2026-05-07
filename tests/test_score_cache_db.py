"""tile_score_cache 表 schema + CRUD 測試。"""
import pytest
from pathlib import Path

from capi_database import CAPIDatabase


@pytest.fixture
def db(tmp_path: Path) -> CAPIDatabase:
    return CAPIDatabase(str(tmp_path / "test.db"))


def test_tile_score_cache_schema_exists(db):
    conn = db._get_conn()
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tile_score_cache'"
        )
        assert cur.fetchone() is not None, "tile_score_cache 表應該存在"
    finally:
        conn.close()


def test_insert_and_get_score_cache(db):
    db.insert_score_cache([
        {"tile_id": 1, "scoring_bundle_id": 10, "score": 0.42},
        {"tile_id": 2, "scoring_bundle_id": 10, "score": 0.88},
        {"tile_id": 3, "scoring_bundle_id": 11, "score": 0.31},
    ])
    got = db.get_score_cache(scoring_bundle_id=10, tile_ids=[1, 2, 3])
    assert got == {1: 0.42, 2: 0.88}  # tile 3 是 bundle 11，不在結果


def test_insert_score_cache_upsert(db):
    db.insert_score_cache([{"tile_id": 1, "scoring_bundle_id": 10, "score": 0.5}])
    db.insert_score_cache([{"tile_id": 1, "scoring_bundle_id": 10, "score": 0.9}])
    got = db.get_score_cache(scoring_bundle_id=10, tile_ids=[1])
    assert got == {1: 0.9}, "同 key 重複插入應 upsert 為新分數"


def test_delete_score_cache_by_bundle(db):
    db.insert_score_cache([
        {"tile_id": 1, "scoring_bundle_id": 10, "score": 0.5},
        {"tile_id": 2, "scoring_bundle_id": 11, "score": 0.5},
    ])
    db.delete_score_cache(scoring_bundle_id=10)
    assert db.get_score_cache(10, [1, 2]) == {}
    assert db.get_score_cache(11, [1, 2]) == {2: 0.5}


def test_delete_score_cache_by_tile_ids(db):
    db.insert_score_cache([
        {"tile_id": 1, "scoring_bundle_id": 10, "score": 0.5},
        {"tile_id": 2, "scoring_bundle_id": 10, "score": 0.5},
        {"tile_id": 1, "scoring_bundle_id": 11, "score": 0.5},
    ])
    db.delete_score_cache(tile_ids=[1])
    assert db.get_score_cache(10, [1, 2]) == {2: 0.5}
    assert db.get_score_cache(11, [1, 2]) == {}


def test_get_score_cache_empty_tile_ids_returns_empty(db):
    assert db.get_score_cache(scoring_bundle_id=10, tile_ids=[]) == {}
