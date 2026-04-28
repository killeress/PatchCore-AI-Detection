def test_module_imports():
    from capi_train_new import (
        TrainingConfig, generate_job_id,
        preprocess_panels_to_pool, sample_ng_tiles,
        run_training_pipeline,
    )
    assert callable(generate_job_id)


def test_generate_job_id_format():
    from capi_train_new import generate_job_id
    job_id = generate_job_id("GN160JCEL250S")
    assert job_id.startswith("train_GN160JCEL250S_")
    assert len(job_id.split("_")) >= 4


def test_preprocess_panels_to_pool_writes_tiles(tmp_path):
    """需要 fixture panel folder，每個 lighting 各 1 張圖。"""
    from pathlib import Path
    from capi_preprocess import PreprocessConfig
    from capi_train_new import preprocess_panels_to_pool, TrainingConfig

    # 準備 fake panel folders（用 Phase 1 的 fixture image 複製）
    fixture_img = Path("tests/fixtures/preprocess/synthetic_panel.png")
    panel_dir = tmp_path / "panel_a"
    panel_dir.mkdir()
    for lighting in ["G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD"]:
        target = panel_dir / f"{lighting}_x.png"
        target.write_bytes(fixture_img.read_bytes())

    # mock DB
    class MockDB:
        def __init__(self):
            self.tiles = []

        def insert_tile_pool(self, job_id, tiles):
            self.tiles.extend(tiles)
            return list(range(len(tiles)))

    db = MockDB()
    cfg = TrainingConfig(
        machine_id="TEST", panel_paths=[panel_dir],
        over_review_root=tmp_path / "or_unused",
    )
    job_id = "j_test"
    pre_cfg = PreprocessConfig(tile_size=256, edge_threshold_px=384, tile_stride=256)

    stats = preprocess_panels_to_pool(
        job_id=job_id, cfg=cfg, preprocess_cfg=pre_cfg,
        db=db, thumb_dir=tmp_path / "thumbs",
        log=lambda msg: None,
    )

    assert stats["panel_success"] == 1
    assert stats["total_tiles"] > 0
    # 應有 inner 和 edge 兩種
    zones = {t["zone"] for t in db.tiles}
    assert "inner" in zones and "edge" in zones
    # 5 lighting 都應有 tile
    lightings = {t["lighting"] for t in db.tiles}
    assert lightings == set(["G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD"])
    # tile 和 thumb 檔案應確實存在磁碟
    assert len(db.tiles) > 0
    first_tile = db.tiles[0]
    assert Path(first_tile["source_path"]).exists()
    assert Path(first_tile["thumb_path"]).exists()
