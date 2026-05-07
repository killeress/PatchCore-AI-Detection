"""SubmodelScorer 單元測試（用 mock inferencer，不跑真 GPU）。"""
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from capi_database import CAPIDatabase


@pytest.fixture
def fake_tile_pool(tmp_path: Path):
    """產 5 張假 tile png + 對應 training_tile_pool row。"""
    db = CAPIDatabase(str(tmp_path / "t.db"))
    job_id = "j1"
    rows = []
    for i in range(5):
        p = tmp_path / f"tile_{i}.png"
        img = np.full((512, 512), 128 + i * 20, dtype=np.uint8)
        cv2.imwrite(str(p), img)
        rows.append({
            "lighting": "WGF50500", "zone": "inner", "source": "ok",
            "source_path": str(p), "thumb_path": str(p),
        })
    ids = db.insert_tile_pool(job_id, rows)
    return db, job_id, ids


def test_scorer_writes_cache_for_all_tiles(fake_tile_pool, tmp_path):
    from capi_inference import SubmodelScorer

    db, job_id, tile_ids = fake_tile_pool
    bundle_dir = tmp_path / "bundle"; bundle_dir.mkdir()
    (bundle_dir / "WGF50500-inner.pt").write_bytes(b"FAKE")

    fake_scores = [0.10, 0.85, 0.42, 0.99, 0.21]

    scorer = SubmodelScorer(
        gpu_lock=threading.Lock(),
        db=db,
        log_fn=lambda msg: None,
    )

    cancel_event = threading.Event()
    progress = []

    with patch.object(scorer, "_load_inferencer_for_pt", return_value=MagicMock()), \
         patch.object(scorer, "_score_one_tile", side_effect=fake_scores):
        result = scorer.score_tiles(
            scoring_bundle_id=99, bundle_dir=bundle_dir,
            lighting="WGF50500", zone="inner",
            tile_pool_job_id=job_id, tile_ids=tile_ids,
            cancel_event=cancel_event,
            progress_cb=lambda d, t: progress.append((d, t)),
        )

    assert result["scanned"] == 5
    assert result["cancelled"] is False
    cache = db.get_score_cache(99, tile_ids)
    assert len(cache) == 5
    assert progress[-1] == (5, 5)


def test_scorer_respects_cancel(fake_tile_pool, tmp_path):
    from capi_inference import SubmodelScorer

    db, job_id, tile_ids = fake_tile_pool
    bundle_dir = tmp_path / "bundle"; bundle_dir.mkdir()
    (bundle_dir / "WGF50500-inner.pt").write_bytes(b"FAKE")

    cancel_event = threading.Event()

    def cancel_after_two(*args, **kwargs):
        if cancel_after_two.calls == 2:
            cancel_event.set()
        cancel_after_two.calls += 1
        return 0.5
    cancel_after_two.calls = 0

    scorer = SubmodelScorer(
        gpu_lock=threading.Lock(), db=db, log_fn=lambda m: None,
    )
    with patch.object(scorer, "_load_inferencer_for_pt", return_value=MagicMock()), \
         patch.object(scorer, "_score_one_tile", side_effect=cancel_after_two):
        result = scorer.score_tiles(
            scoring_bundle_id=99, bundle_dir=bundle_dir,
            lighting="WGF50500", zone="inner",
            tile_pool_job_id=job_id, tile_ids=tile_ids,
            cancel_event=cancel_event, progress_cb=lambda d, t: None,
        )
    assert result["cancelled"] is True
    assert result["scanned"] < 5


def test_scorer_skips_missing_pt(fake_tile_pool, tmp_path):
    from capi_inference import SubmodelScorer

    db, job_id, tile_ids = fake_tile_pool
    bundle_dir = tmp_path / "bundle"; bundle_dir.mkdir()
    # 故意不放 .pt

    scorer = SubmodelScorer(
        gpu_lock=threading.Lock(), db=db, log_fn=lambda m: None,
    )
    with pytest.raises(FileNotFoundError):
        scorer.score_tiles(
            scoring_bundle_id=99, bundle_dir=bundle_dir,
            lighting="WGF50500", zone="inner",
            tile_pool_job_id=job_id, tile_ids=tile_ids,
            cancel_event=threading.Event(), progress_cb=lambda d, t: None,
        )


def test_scorer_skips_missing_tile_image(fake_tile_pool, tmp_path):
    from capi_inference import SubmodelScorer

    db, job_id, tile_ids = fake_tile_pool
    # 砍掉第一張圖
    pool = db.list_tile_pool(job_id)
    Path(pool[0]["source_path"]).unlink()

    bundle_dir = tmp_path / "bundle"; bundle_dir.mkdir()
    (bundle_dir / "WGF50500-inner.pt").write_bytes(b"FAKE")

    scorer = SubmodelScorer(
        gpu_lock=threading.Lock(), db=db, log_fn=lambda m: None,
    )
    with patch.object(scorer, "_load_inferencer_for_pt", return_value=MagicMock()), \
         patch.object(scorer, "_score_one_tile", return_value=0.5):
        result = scorer.score_tiles(
            scoring_bundle_id=99, bundle_dir=bundle_dir,
            lighting="WGF50500", zone="inner",
            tile_pool_job_id=job_id, tile_ids=tile_ids,
            cancel_event=threading.Event(), progress_cb=lambda d, t: None,
        )
    assert result["scanned"] == 4
    assert result["skipped"] == 1
