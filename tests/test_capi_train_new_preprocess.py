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
    # grid 最外圈應歸為 edge，其餘完整在 panel 內的 tile 才是 inner。
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


def test_preprocess_panels_to_pool_all_panels_have_inner_and_edge(tmp_path):
    """每片 panel 都應同時有 inner + edge tile。"""
    from pathlib import Path
    from capi_preprocess import PreprocessConfig
    from capi_train_new import preprocess_panels_to_pool, TrainingConfig

    fixture_img = Path("tests/fixtures/preprocess/synthetic_panel.png")
    panel_dirs = []
    for i in range(3):
        d = tmp_path / f"panel_{i + 1}"
        d.mkdir()
        for lighting in ["G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD"]:
            (d / f"{lighting}_x.png").write_bytes(fixture_img.read_bytes())
        panel_dirs.append(d)

    class MockDB:
        def __init__(self):
            self.tiles_per_call = []

        def insert_tile_pool(self, job_id, tiles):
            self.tiles_per_call.append(list(tiles))
            return list(range(len(tiles)))

    db = MockDB()
    cfg = TrainingConfig(
        machine_id="TEST", panel_paths=panel_dirs,
        over_review_root=tmp_path / "or_unused",
    )
    pre_cfg = PreprocessConfig(tile_size=256, edge_threshold_px=384, tile_stride=256)

    preprocess_panels_to_pool(
        job_id="j_split", cfg=cfg, preprocess_cfg=pre_cfg,
        db=db, thumb_dir=tmp_path / "thumbs", log=lambda m: None,
    )

    assert len(db.tiles_per_call) == 3
    for i, batch in enumerate(db.tiles_per_call):
        zones = {t["zone"] for t in batch}
        assert "inner" in zones, f"panel {i + 1} 應該包含 inner tile"
        assert "edge" in zones, f"panel {i + 1} 應該包含 edge tile"


def test_preprocess_panels_to_pool_respects_per_panel_zone_modes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from capi_preprocess import PreprocessConfig
    from capi_train_new import preprocess_panels_to_pool, TrainingConfig

    panel_dirs = [tmp_path / "panel_inner", tmp_path / "panel_edge"]
    for panel_dir in panel_dirs:
        panel_dir.mkdir()

    tiles = [
        SimpleNamespace(
            tile_id=1,
            image=np.zeros((16, 16), dtype=np.uint8),
            zone="inner",
            is_corner=False,
        ),
        SimpleNamespace(
            tile_id=2,
            image=np.zeros((16, 16), dtype=np.uint8),
            zone="edge",
            is_corner=True,
        ),
    ]
    result = SimpleNamespace(polygon_detection_failed=False, tiles=tiles)
    monkeypatch.setattr(
        "capi_train_new.preprocess_panel_folder",
        lambda _panel, _cfg: {"G0F00000": result},
    )

    class MockDB:
        def __init__(self):
            self.tiles_per_call = []

        def insert_tile_pool(self, job_id, tile_records):
            self.tiles_per_call.append(list(tile_records))
            return []

    db = MockDB()
    cfg = TrainingConfig(
        machine_id="TEST",
        panel_paths=panel_dirs,
        over_review_root=tmp_path / "or_unused",
    )

    stats = preprocess_panels_to_pool(
        job_id="j_modes",
        cfg=cfg,
        preprocess_cfg=PreprocessConfig(),
        db=db,
        thumb_dir=tmp_path / "thumbs",
        log=lambda _msg: None,
        panel_modes=["inner_only", "edge_only"],
    )

    assert [[tile["zone"] for tile in batch] for batch in db.tiles_per_call] == [
        ["inner"],
        ["edge"],
    ]
    assert stats["panel_success"] == 2
    assert stats["panel_success_inner_only"] == 1
    assert stats["panel_success_edge_only"] == 1


def test_preprocess_panels_to_pool_logs_after_tiling_mode(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from capi_preprocess import PreprocessConfig
    from capi_train_new import preprocess_panels_to_pool, TrainingConfig

    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    tile = SimpleNamespace(
        tile_id=1,
        image=np.zeros((16, 16), dtype=np.uint8),
        zone="inner",
        is_corner=False,
    )
    result = SimpleNamespace(polygon_detection_failed=False, tiles=[tile])
    monkeypatch.setattr(
        "capi_train_new.preprocess_panel_folder",
        lambda _panel, _cfg: {"G0F00000": result},
    )

    class MockDB:
        def insert_tile_pool(self, job_id, tiles):
            return []

    logs = []
    cfg = TrainingConfig(
        machine_id="TEST",
        panel_paths=[panel_dir],
        over_review_root=tmp_path / "or_unused",
    )
    pre_cfg = PreprocessConfig(
        image_preprocess_pipeline=[
            {"method": "gaussian", "params": {"kernel_size": 5, "sigma": 1.0}},
        ],
        preprocess_after_tiling=True,
    )

    preprocess_panels_to_pool(
        job_id="j_log",
        cfg=cfg,
        preprocess_cfg=pre_cfg,
        db=MockDB(),
        thumb_dir=tmp_path / "thumbs",
        log=logs.append,
    )

    assert any("影像前處理模式: 先切分後處理" in msg for msg in logs)
    assert any("已對這 1 個 tile 套用 1.高斯平滑" in msg for msg in logs)


def test_sample_ng_tiles(tmp_path):
    from capi_train_new import sample_ng_tiles

    # 模擬 over_review 結構：snapshot/true_ng/<lighting>/crop/<files>.png
    or_root = tmp_path / "over_review"
    snap_a = or_root / "20260415_104812" / "true_ng"
    for lighting in ["G0F00000", "R0F00000", "STANDARD"]:
        crop_dir = snap_a / lighting / "crop"
        crop_dir.mkdir(parents=True)
        for i in range(50):
            (crop_dir / f"img_{i}.png").write_bytes(b"x")

    class MockDB:
        def __init__(self): self.tiles = []
        def insert_tile_pool(self, job_id, tiles):
            self.tiles.extend(tiles)
            return list(range(len(tiles)))

    db = MockDB()
    stats = sample_ng_tiles(
        job_id="j1", over_review_root=or_root, db=db,
        per_lighting=10, log=lambda m: None,
    )
    assert stats["sampled"] == 30  # 3 lighting × 10
    assert set(stats["missing_lightings"]) == {"W0F00000", "WGF50500"}
    by_lighting = {}
    for t in db.tiles:
        by_lighting.setdefault(t["lighting"], 0)
        by_lighting[t["lighting"]] += 1
    assert by_lighting == {"G0F00000": 10, "R0F00000": 10, "STANDARD": 10}
    # 沒有 manifest.csv → zone heuristic 全部回 None（不分 inner/edge）
    assert all(t["zone"] is None for t in db.tiles)
    assert all(t["source"] == "ng" for t in db.tiles)


def test_sample_ng_tiles_applies_preprocess_pipeline_to_ng_crops(tmp_path):
    import cv2
    import numpy as np
    from pathlib import Path
    from capi_preprocess import PreprocessConfig
    from capi_train_new import sample_ng_tiles

    or_root = tmp_path / "over_review"
    crop_dir = or_root / "20260415_104812" / "true_ng" / "G0F00000" / "crop"
    crop_dir.mkdir(parents=True)
    original = np.zeros((512, 512), dtype=np.uint8)
    original[:, 256:] = 255
    source = crop_dir / "img_0.png"
    assert cv2.imwrite(str(source), original)

    class MockDB:
        def __init__(self): self.tiles = []
        def insert_tile_pool(self, job_id, tiles):
            self.tiles.extend(tiles)
            return list(range(len(tiles)))

    db = MockDB()
    logs = []
    thumb_dir = tmp_path / "thumbs"
    sample_ng_tiles(
        job_id="j_ng_pre",
        over_review_root=or_root,
        db=db,
        thumb_dir=thumb_dir,
        per_lighting=1,
        log=logs.append,
        lightings=("G0F00000",),
        preprocess_cfg=PreprocessConfig(
            preprocess_after_tiling=True,
            image_preprocess_pipeline=[
                {"method": "gaussian", "params": {"kernel_size": 5, "sigma": 1.0}},
            ],
        ),
    )

    assert len(db.tiles) == 1
    processed_path = Path(db.tiles[0]["source_path"])
    assert processed_path.exists()
    assert processed_path != source.resolve()
    processed = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)
    assert processed is not None
    assert not np.array_equal(processed, original)
    assert any("前處理=1: 1.高斯平滑" in msg for msg in logs)


def test_sample_ng_tiles_classifies_zone_from_manifest(tmp_path):
    """有 manifest.csv 時依 defect_y 標 zone：< EDGE_BAND_PX → edge，否則 inner。"""
    import csv
    from pathlib import Path
    from capi_train_new import sample_ng_tiles, EDGE_BAND_PX

    or_root = tmp_path / "over_review"
    snap = or_root / "20260420_120000"
    crop_dir = snap / "true_ng" / "G0F00000" / "crop"
    crop_dir.mkdir(parents=True)
    rows = []
    for i in range(6):
        fname = f"img_{i}.png"
        (crop_dir / fname).write_bytes(b"x")
        # 前 3 張落在 top edge band，後 3 張在 inner
        defect_y = 100 if i < 3 else EDGE_BAND_PX + 500
        rows.append({"sample_id": f"s{i}",
                     "crop_path": f"true_ng/G0F00000/crop/{fname}",
                     "defect_y": str(defect_y)})
    with open(snap / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "crop_path", "defect_y"])
        w.writeheader()
        w.writerows(rows)

    class MockDB:
        def __init__(self): self.tiles = []
        def insert_tile_pool(self, job_id, tiles):
            self.tiles.extend(tiles)
            return list(range(len(tiles)))

    db = MockDB()
    sample_ng_tiles(job_id="j2", over_review_root=or_root, db=db,
                    per_lighting=10, log=lambda m: None)

    zones = {Path(t["source_path"]).name: t["zone"] for t in db.tiles}
    for i in range(3):
        assert zones[f"img_{i}.png"] == "edge", f"img_{i} 應為 edge"
    for i in range(3, 6):
        assert zones[f"img_{i}.png"] == "inner", f"img_{i} 應為 inner"


def test_sample_ng_tiles_classifies_zone_with_manifest_height_768(tmp_path):
    """1366x768 機種不應因 EDGE_BAND_PX=768 導致全部 NG 都被標成 edge。"""
    import csv
    from pathlib import Path
    from capi_train_new import sample_ng_tiles

    or_root = tmp_path / "over_review"
    snap = or_root / "20260420_130000"
    crop_dir = snap / "true_ng" / "G0F00000" / "crop"
    crop_dir.mkdir(parents=True)
    rows = [
        ("top.png", 100, "edge"),
        ("middle.png", 384, "inner"),
        ("bottom.png", 760, "edge"),
    ]
    manifest_rows = []
    for fname, defect_y, _zone in rows:
        (crop_dir / fname).write_bytes(b"x")
        manifest_rows.append({
            "sample_id": fname,
            "crop_path": f"true_ng/G0F00000/crop/{fname}",
            "defect_y": str(defect_y),
            "image_height": "768",
        })
    with open(snap / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "crop_path", "defect_y", "image_height"])
        w.writeheader()
        w.writerows(manifest_rows)

    class MockDB:
        def __init__(self): self.tiles = []
        def insert_tile_pool(self, job_id, tiles):
            self.tiles.extend(tiles)
            return list(range(len(tiles)))

    db = MockDB()
    sample_ng_tiles(job_id="j3", over_review_root=or_root, db=db,
                    per_lighting=10, log=lambda m: None)

    zones = {Path(t["source_path"]).name: t["zone"] for t in db.tiles}
    assert zones == {fname: zone for fname, _y, zone in rows}


def test_sample_ng_tiles_writes_confined_thumbnails(tmp_path):
    from pathlib import Path
    import cv2
    import numpy as np
    from capi_train_new import sample_ng_tiles

    or_root = tmp_path / "over_review"
    crop_dir = or_root / "20260415_104812" / "true_ng" / "G0F00000" / "crop"
    crop_dir.mkdir(parents=True)
    for i in range(2):
        cv2.imwrite(str(crop_dir / f"img_{i}.png"), np.full((32, 32), 128, dtype=np.uint8))

    class MockDB:
        def __init__(self): self.tiles = []
        def insert_tile_pool(self, job_id, tiles):
            self.tiles.extend(tiles)
            return list(range(len(tiles)))

    db = MockDB()
    thumb_dir = tmp_path / ".tmp" / "train_new_thumbs" / "j1"
    sample_ng_tiles(
        job_id="j1", over_review_root=or_root, db=db,
        thumb_dir=thumb_dir, per_lighting=2, log=lambda m: None,
    )

    assert db.tiles
    for tile in db.tiles:
        thumb = Path(tile["thumb_path"])
        assert thumb.exists()
        thumb.resolve().relative_to(thumb_dir.resolve())
