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


def test_image_preprocess_pipeline_for_zone_prefers_zone_and_falls_back_to_shared():
    from capi_preprocess import PreprocessConfig, image_preprocess_pipeline_for_zone

    shared = [{"method": "gaussian", "params": {"kernel_size": 3, "sigma": 1.0}}]
    inner = [{"method": "bilateral", "params": {"diameter": 5}}]
    edge = [{"method": "median", "params": {"kernel_size": 5}}]
    zone_cfg = PreprocessConfig(
        image_preprocess_pipeline=shared,
        image_preprocess_pipelines={"inner": inner, "edge": edge},
        preprocess_after_tiling=True,
    )

    assert image_preprocess_pipeline_for_zone(zone_cfg, "inner") == inner
    assert image_preprocess_pipeline_for_zone(zone_cfg, "edge") == edge
    assert image_preprocess_pipeline_for_zone(PreprocessConfig(image_preprocess_pipeline=shared), "inner") == shared


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
    assert first_tile["panel_path"] == str(panel_dir.resolve())
    assert isinstance(first_tile["tile_index"], int)
    assert isinstance(first_tile["tile_x"], int)
    assert isinstance(first_tile["tile_y"], int)
    assert first_tile["tile_width"] == 256
    assert first_tile["tile_height"] == 256


def test_preprocess_panels_to_pool_accepts_aapi_glass_prefixed_images(tmp_path):
    from pathlib import Path

    from capi_preprocess import PreprocessConfig
    from capi_station_adapter import AAPIStationAdapter
    from capi_train_new import TrainingConfig, preprocess_panels_to_pool

    fixture_img = Path("tests/fixtures/preprocess/synthetic_panel.png")
    panel_dir = tmp_path / "YQ52TR205A41"
    panel_dir.mkdir()
    image_names = (
        "YQ52TR205A41G0F00000073955.tif",
        "YQ52TR205A41R0F00000073956.tif",
        "YQ52TR205A41W0F00000073951.tif",
        "YQ52TR205A41W0F00010073959.tif",
        "YQ52TR205A41WGF25250073954.tif",
        "YQ52TR205A41WGF50500073958.tif",
        "YQ52TR205A41U0F00000073953.tif",
        "YQ52TR205A41Windows_BG073957.tif",
    )
    for image_name in image_names:
        (panel_dir / image_name).write_bytes(fixture_img.read_bytes())

    class MockDB:
        def __init__(self):
            self.tiles = []

        def insert_tile_pool(self, job_id, tiles):
            self.tiles.extend(tiles)
            return list(range(len(tiles)))

    db = MockDB()
    adapter = AAPIStationAdapter()
    stats = preprocess_panels_to_pool(
        job_id="j_aapi",
        cfg=TrainingConfig(
            machine_id="GN160TEST",
            panel_paths=[panel_dir],
            over_review_root=tmp_path / "unused",
        ),
        preprocess_cfg=PreprocessConfig(tile_size=256, tile_stride=256),
        db=db,
        thumb_dir=tmp_path / "thumbs",
        log=lambda _msg: None,
        target_lightings=adapter.training_prefixes,
        prefix_resolver=adapter.image_prefix,
        allowed_prefixes=adapter.inference_prefixes,
        boundary_reference_priority=adapter.boundary_reference_priority,
        training_lighting_resolver=adapter.model_prefix,
    )

    assert stats["panel_success"] == 1
    assert {tile["lighting"] for tile in db.tiles} == {
        "G0F00000", "R0F00000", "W0F00000", "WGF25250",
        "WGF50500", "U0F00000", "STANDARD",
    }
    tile_names = {Path(tile["source_path"]).name for tile in db.tiles}
    assert any("W0F00010" in name for name in tile_names)
    assert any("WGF25250" in name for name in tile_names)
    assert any("WGF50500" in name for name in tile_names)
    assert any("U0F00000" in name for name in tile_names)


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
    import cv2
    import numpy as np
    from pathlib import Path
    from capi_train_new import sample_ng_tiles

    source = tmp_path / "WGF50500_100000.png"
    image = np.zeros((900, 1000), dtype=np.uint8)
    image[200:712, 300:812] = 177
    assert cv2.imwrite(str(source), image)

    class MockDB:
        def __init__(self):
            self.tiles = []
            self.query = None

        def list_training_bomb_candidates(self, machine_id, lightings):
            self.query = (machine_id, tuple(lightings))
            return [
                {
                    "inference_record_id": 10,
                    "source_result_id": 20,
                    "client_bomb_info": '{"image_prefix":"WGF50500","defect_type":"point","coordinates":[[556,456]]}',
                    "image_path": str(source),
                    "image_dir": str(tmp_path),
                    "image_name": source.name,
                    "image_width": 1000,
                    "image_height": 900,
                    "resolution_x": 1000,
                    "resolution_y": 900,
                    "otsu_bounds": "0,0,1000,900",
                },
                {
                    "inference_record_id": 11,
                    "source_result_id": 21,
                    "client_bomb_info": '{"image_prefix":"B0F00000","defect_type":"point","coordinates":[[256,256]]}',
                    "image_path": str(tmp_path / "B0F00000_100000.png"),
                    "image_dir": str(tmp_path),
                    "image_name": "B0F00000_100000.png",
                    "image_width": 1000,
                    "image_height": 900,
                    "resolution_x": 1000,
                    "resolution_y": 900,
                    "otsu_bounds": "0,0,1000,900",
                },
            ]

        def insert_tile_pool(self, job_id, tiles):
            self.tiles.extend(tiles)
            return list(range(len(tiles)))

    db = MockDB()
    thumb_dir = tmp_path / ".tmp" / "train_new_thumbs" / "j1"
    stats = sample_ng_tiles(
        job_id="j1", machine_id="MODEL-A", over_review_root=tmp_path / "unused", db=db,
        thumb_dir=thumb_dir, per_lighting=10, log=lambda m: None,
        lightings=("WGF50500", "B0F00000"),
    )

    assert db.query == ("MODEL-A", ("WGF50500",))
    assert stats["sampled"] == 1
    assert stats["missing_lightings"] == []
    assert stats["black_skipped"] == 1
    assert len(db.tiles) == 1
    tile = db.tiles[0]
    assert tile["lighting"] == "WGF50500"
    assert tile["zone"] == "inner"
    assert tile["source"] == "ng"
    assert tile["panel_path"] == str(source.resolve())
    crop_path = Path(tile["source_path"])
    thumb_path = Path(tile["thumb_path"])
    assert crop_path.exists() and cv2.imread(str(crop_path)).shape[:2] == (512, 512)
    assert int(cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE).mean()) == 177
    crop_path.resolve().relative_to(thumb_dir.resolve())
    thumb_path.resolve().relative_to(thumb_dir.resolve())


def test_sample_ng_tiles_aggregates_missing_recent_source_warnings(tmp_path, monkeypatch):
    from pathlib import Path

    from capi_train_new import sample_ng_tiles

    class MockDB:
        def list_training_bomb_candidates(self, machine_id, lightings):
            assert machine_id == "MODEL-A"
            assert tuple(lightings) == ("G0F00000",)
            return [
                {
                    "inference_record_id": index,
                    "source_result_id": index,
                    "client_bomb_info": (
                        '{"image_prefix":"G0F00000","defect_type":"point",'
                        '"coordinates":[[350,350]]}'
                    ),
                    "image_path": str(tmp_path / f"G0F00000_missing_{index}.png"),
                    "image_dir": str(tmp_path),
                    "image_name": f"G0F00000_missing_{index}.png",
                }
                for index in (1, 2)
            ]

        def insert_tile_pool(self, _job_id, _tiles):
            raise AssertionError("missing sources must not create NG tiles")

    probed_paths = []
    original_is_file = Path.is_file

    def tracked_is_file(path):
        probed_paths.append(path)
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", tracked_is_file)
    logs = []
    stats = sample_ng_tiles(
        job_id="j_missing_recent",
        machine_id="MODEL-A",
        over_review_root=tmp_path / "unused",
        db=MockDB(),
        thumb_dir=tmp_path / "thumbs",
        per_lighting=10,
        log=logs.append,
        lightings=("G0F00000",),
    )

    missing_logs = [line for line in logs if "炸彈原圖不存在" in line]
    missing_probes = [path for path in probed_paths if path.name.startswith("G0F00000_missing_")]
    assert stats["sampled"] == 0
    assert stats["invalid_skipped"] == 2
    assert missing_logs == ["  ⚠ G0F00000: 最近 3 天炸彈原圖不存在 2 筆，已略過"]
    assert len(missing_probes) == 2


def test_sample_ng_tiles_routes_aapi_prefixed_image_to_model_lighting(tmp_path):
    import cv2
    import numpy as np

    from capi_station_adapter import AAPIStationAdapter
    from capi_train_new import sample_ng_tiles

    source = tmp_path / "YQ52TR205A41Windows_BG073957.tif"
    image = np.full((900, 1000), 177, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)

    class MockDB:
        def __init__(self):
            self.tiles = []
            self.query_lightings = "unset"

        def list_training_bomb_candidates(self, machine_id, lightings):
            self.query_lightings = lightings
            return [{
                "inference_record_id": 10,
                "source_result_id": 20,
                "client_bomb_info": (
                    '{"image_prefix":"Windows_BG","defect_type":"point",'
                    '"coordinates":[[500,450]]}'
                ),
                "image_path": str(source),
                "image_dir": str(tmp_path),
                "image_name": source.name,
                "image_width": 1000,
                "image_height": 900,
                "resolution_x": 1000,
                "resolution_y": 900,
                "otsu_bounds": "0,0,1000,900",
            }]

        def insert_tile_pool(self, _job_id, tiles):
            self.tiles.extend(tiles)
            return list(range(len(tiles)))

    db = MockDB()
    adapter = AAPIStationAdapter()
    stats = sample_ng_tiles(
        job_id="j_aapi_ng",
        machine_id="MODEL-A",
        over_review_root=tmp_path / "unused",
        db=db,
        thumb_dir=tmp_path / "thumbs",
        per_lighting=10,
        log=lambda _msg: None,
        lightings=("STANDARD",),
        image_prefix_resolver=adapter.image_prefix,
        model_prefix_resolver=adapter.model_prefix,
    )

    assert db.query_lightings is None
    assert stats["sampled"] == 1
    assert db.tiles[0]["lighting"] == "STANDARD"


def test_sample_ng_tiles_saves_first_crop_then_reuses_ng_validation_cache(tmp_path):
    import cv2
    import numpy as np
    from pathlib import Path
    from capi_train_new import sample_ng_tiles

    source = tmp_path / "G0F00000_100000.png"
    assert cv2.imwrite(str(source), np.full((700, 700), 180, dtype=np.uint8))

    class MockDB:
        def __init__(self):
            self.cache = []
            self.raw_queries = 0
            self.tiles_by_job = {}

        def list_training_bomb_validation_samples(self, *, machine_id, lightings):
            assert machine_id == "MODEL-A"
            return [row for row in self.cache if row["lighting"] in lightings]

        def list_training_bomb_candidates(self, machine_id, lightings):
            self.raw_queries += 1
            assert machine_id == "MODEL-A"
            assert tuple(lightings) == ("G0F00000",)
            return [{
                "inference_record_id": 10,
                "source_result_id": 20,
                "glass_id": "PANEL-10",
                "model_id": "MODEL-A",
                "machine_no": "HM01",
                "request_time": "2026-08-14 10:00:00",
                "client_bomb_info": (
                    '{"image_prefix":"G0F00000","defect_type":"point",'
                    '"coordinates":[[350,350]]}'
                ),
                "image_path": str(source),
                "image_dir": str(tmp_path),
                "image_name": source.name,
                "resolution_x": 700,
                "resolution_y": 700,
                "otsu_bounds": "0,0,700,700",
            }]

        def save_training_bomb_validation_samples(self, samples):
            self.cache = [
                {**sample, "id": index + 1, "sample_source": "training_bomb"}
                for index, sample in enumerate(samples)
            ]
            return len(samples)

        def insert_tile_pool(self, job_id, tiles):
            self.tiles_by_job.setdefault(job_id, []).extend(tiles)
            return list(range(len(tiles)))

    db = MockDB()
    validation_root = tmp_path / "ng-validation"
    first = sample_ng_tiles(
        job_id="j_first",
        machine_id="MODEL-A",
        over_review_root=tmp_path / "unused",
        db=db,
        thumb_dir=tmp_path / "first",
        per_lighting=10,
        lightings=("G0F00000",),
        ng_validation_base_dir=validation_root,
        log=lambda _message: None,
    )

    assert first["sampled"] == 1
    assert first["cache_saved"] == 1
    assert first["cache_reused"] == 0
    assert db.raw_queries == 1
    assert len(db.cache) == 1
    cache_path = Path(db.cache[0]["crop_path"])
    cache_path.resolve().relative_to(validation_root.resolve())
    assert cache_path.is_file()

    source.unlink()
    second = sample_ng_tiles(
        job_id="j_second",
        machine_id="MODEL-A",
        over_review_root=tmp_path / "unused",
        db=db,
        thumb_dir=tmp_path / "second",
        per_lighting=10,
        lightings=("G0F00000",),
        ng_validation_base_dir=validation_root,
        log=lambda _message: None,
    )

    assert second["sampled"] == 1
    assert second["cache_reused"] == 1
    assert second["cache_saved"] == 0
    assert db.raw_queries == 1
    reused_path = Path(db.tiles_by_job["j_second"][0]["source_path"])
    assert reused_path.is_file()
    assert cv2.imread(str(reused_path), cv2.IMREAD_GRAYSCALE).shape == (512, 512)


def test_sample_ng_tiles_applies_preprocess_pipeline_to_ng_crops(tmp_path):
    import cv2
    import numpy as np
    from pathlib import Path
    from capi_preprocess import PreprocessConfig
    from capi_train_new import sample_ng_tiles

    original = np.zeros((700, 700), dtype=np.uint8)
    original[:, 350:] = 255
    source = tmp_path / "G0F00000_100000.png"
    assert cv2.imwrite(str(source), original)

    class MockDB:
        def __init__(self):
            self.tiles = []
            self.saved = []
        def list_training_bomb_validation_samples(self, **_kwargs):
            return []
        def list_training_bomb_candidates(self, machine_id, lightings):
            return [{
                "inference_record_id": 1, "source_result_id": 2,
                "client_bomb_info": '{"image_prefix":"G0F00000","defect_type":"point","coordinates":[[350,350]]}',
                "image_path": str(source), "image_dir": str(tmp_path), "image_name": source.name,
                "image_width": 700, "image_height": 700,
                "resolution_x": 700, "resolution_y": 700, "otsu_bounds": "0,0,700,700",
            }]
        def insert_tile_pool(self, job_id, tiles):
            self.tiles.extend(tiles)
            return list(range(len(tiles)))
        def save_training_bomb_validation_samples(self, samples):
            self.saved.extend(samples)
            return len(samples)

    db = MockDB()
    logs = []
    thumb_dir = tmp_path / "thumbs"
    sample_ng_tiles(
        job_id="j_ng_pre",
        machine_id="MODEL-A",
        over_review_root=tmp_path / "unused",
        db=db,
        thumb_dir=thumb_dir,
        per_lighting=1,
        log=logs.append,
        lightings=("G0F00000",),
        ng_validation_base_dir=tmp_path / "ng-validation",
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
    processed = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)
    assert processed is not None
    tile = db.tiles[0]
    raw_crop = original[
        tile["tile_y"]:tile["tile_y"] + tile["tile_height"],
        tile["tile_x"]:tile["tile_x"] + tile["tile_width"],
    ]
    assert not np.array_equal(processed, raw_crop)
    cached_raw = cv2.imread(db.saved[0]["crop_path"], cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(cached_raw, raw_crop)
    assert any("前處理=1: 1.高斯平滑" in msg for msg in logs)


def test_sample_ng_tiles_uses_formal_polygon_mapping_and_inward_roi(tmp_path, monkeypatch):
    """NG crop must use the same polygon mapping/inward shift as v2 inference."""
    import cv2
    import numpy as np
    from pathlib import Path
    from capi_preprocess import PreprocessConfig
    from capi_train_new import sample_ng_tiles

    source = tmp_path / "R0F00000_edge.png"
    image = np.zeros((900, 1000), dtype=np.uint8)
    image[100:800, 100:900] = 200
    assert cv2.imwrite(str(source), image)

    polygon = np.array(
        [[100, 100], [899, 100], [899, 799], [100, 799]],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        "capi_train_new.detect_panel_polygon",
        lambda _image, _config: ((100, 100, 900, 800), polygon),
    )

    class MockDB:
        def __init__(self):
            self.tiles = []

        def list_training_bomb_candidates(self, machine_id, lightings):
            return [{
                "inference_record_id": 1,
                "source_result_id": 2,
                "client_bomb_info": (
                    '{"image_prefix":"R0F00000","defect_type":"point",'
                    '"coordinates":[[1800,540]]}'
                ),
                "image_path": str(source),
                "image_dir": str(tmp_path),
                "image_name": source.name,
                "resolution_x": 1920,
                "resolution_y": 1080,
                # Deliberately contaminated/raw bounds: formal mapping must
                # correct this through the detected panel polygon.
                "otsu_bounds": "0,0,1000,900",
            }]

        def insert_tile_pool(self, job_id, tiles):
            self.tiles.extend(tiles)
            return list(range(len(tiles)))

    db = MockDB()
    stats = sample_ng_tiles(
        job_id="j_polygon",
        machine_id="MODEL-A",
        over_review_root=tmp_path / "unused",
        db=db,
        thumb_dir=tmp_path / "thumbs",
        per_lighting=1,
        lightings=("R0F00000",),
        preprocess_cfg=PreprocessConfig(
            tile_size=512,
            product_resolution=(1920, 1080),
            enable_panel_polygon=True,
        ),
        log=lambda _message: None,
    )

    assert stats["sampled"] == 1
    tile = db.tiles[0]
    # Linear mapping would center at x=681? The formal polygon correction maps
    # the point to x=850, then shifts the tile inward to the panel right edge.
    assert tile["zone"] == "edge"
    assert tile["tile_x"] < 594
    assert tile["tile_x"] + tile["tile_width"] <= 900
    crop = cv2.imread(str(Path(tile["source_path"])), cv2.IMREAD_GRAYSCALE)
    assert crop is not None and crop.shape == (512, 512)
    assert int(crop.mean()) > 190


def test_sample_ng_tiles_crops_line_bomb_at_segment_midpoint(tmp_path):
    import cv2
    import numpy as np
    from pathlib import Path
    from capi_train_new import sample_ng_tiles

    source = tmp_path / "WGF50500_line.png"
    image = np.zeros((1024, 1024), dtype=np.uint8)
    image[510:515, 256:769] = 255
    assert cv2.imwrite(str(source), image)

    class MockDB:
        def __init__(self): self.tiles = []
        def list_training_bomb_candidates(self, machine_id, lightings):
            return [{
                "inference_record_id": 8, "source_result_id": 9,
                "client_bomb_info": '{"image_prefix":"WGF50500","defect_type":"line","coordinates":[[256,512],[768,512]]}',
                "image_path": str(source), "image_dir": str(tmp_path), "image_name": source.name,
                "image_width": 1024, "image_height": 1024,
                "resolution_x": 1024, "resolution_y": 1024, "otsu_bounds": "0,0,1024,1024",
            }]
        def insert_tile_pool(self, job_id, tiles):
            self.tiles.extend(tiles)
            return list(range(len(tiles)))

    db = MockDB()
    stats = sample_ng_tiles(
        job_id="j_line", machine_id="MODEL-A", over_review_root=tmp_path / "unused", db=db,
        thumb_dir=tmp_path / "thumbs", per_lighting=10, log=lambda m: None,
        lightings=("WGF50500",),
    )

    assert stats["sampled"] == 1
    crop_path = Path(db.tiles[0]["source_path"])
    crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    assert crop is not None
    assert crop.shape == (512, 512)
    assert int(crop[256, 256]) == 255
    assert "line0" in crop_path.name


def test_sample_ng_tiles_black_only_scope_does_not_query_database(tmp_path):
    from capi_train_new import sample_ng_tiles

    class MockDB:
        def list_training_bomb_candidates(self, **_kwargs):
            raise AssertionError("B0F-only scope must not query bomb candidates")

    stats = sample_ng_tiles(
        job_id="j_black", machine_id="MODEL-A", over_review_root=tmp_path / "unused",
        db=MockDB(), lightings=("B0F00000",), log=lambda _msg: None,
    )

    assert stats == {
        "sampled": 0,
        "missing_lightings": [],
        "black_skipped": 0,
        "invalid_skipped": 0,
        "cache_reused": 0,
        "cache_saved": 0,
    }


def test_sample_ng_tiles_classifies_zone_from_inference_crop(tmp_path):
    """依 AOI 炸彈座標映射後的 crop 中心與影像高度判定 zone。"""
    import cv2
    import numpy as np
    from capi_train_new import sample_ng_tiles

    source = tmp_path / "W0F00000_100000.png"
    assert cv2.imwrite(str(source), np.full((1200, 1000), 128, dtype=np.uint8))

    class MockDB:
        def __init__(self): self.tiles = []
        def list_training_bomb_candidates(self, machine_id, lightings):
            return [{
                "inference_record_id": 1, "source_result_id": 1,
                "client_bomb_info": '{"image_prefix":"W0F00000","defect_type":"point","coordinates":[[500,600],[10,40]]}',
                "image_path": str(source), "image_dir": str(tmp_path), "image_name": source.name,
                "image_width": 1000, "image_height": 1200,
                "resolution_x": 1000, "resolution_y": 1200, "otsu_bounds": "0,0,1000,1200",
            }]
        def insert_tile_pool(self, job_id, tiles):
            self.tiles.extend(tiles)
            return list(range(len(tiles)))

    db = MockDB()
    sample_ng_tiles(
        job_id="j2", machine_id="MODEL-A", over_review_root=tmp_path / "unused", db=db,
        thumb_dir=tmp_path / "thumbs", per_lighting=10, log=lambda m: None,
        lightings=("W0F00000",),
    )

    assert {tile["zone"] for tile in db.tiles} == {"inner", "edge"}


def test_sample_ng_tiles_classifies_zone_against_all_panel_edges():
    from capi_train_new import _classify_ng_crop_zone

    bounds = (0, 0, 1366, 768)
    assert _classify_ng_crop_zone(683, 100, bounds) == "edge"
    assert _classify_ng_crop_zone(683, 384, bounds) == "inner"
    assert _classify_ng_crop_zone(10, 384, bounds) == "edge"
    assert _classify_ng_crop_zone(1350, 384, bounds) == "edge"
    assert _classify_ng_crop_zone(683, 760, bounds) == "edge"
