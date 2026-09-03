"""capi_preprocess 模組的整合測試：preprocess_panel_image 完整 pipeline"""
import sys
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import capi_preprocess
from capi_preprocess import (
    PanelPreprocessResult,
    PreprocessConfig,
    preprocess_panel_folder,
    preprocess_panel_image,
)

FIXTURE = Path(__file__).parent / "fixtures" / "preprocess" / "synthetic_panel.png"


def test_preprocess_panel_image_basic():
    cfg = PreprocessConfig(tile_size=256, tile_stride=256, edge_threshold_px=384)
    result = preprocess_panel_image(FIXTURE, "STANDARD", cfg)
    assert result.lighting == "STANDARD"
    assert not result.polygon_detection_failed
    assert result.panel_polygon is not None
    assert len(result.tiles) > 0
    zones = {t.zone for t in result.tiles}
    assert "inner" in zones
    assert "edge" in zones
    # outside tile 不應出現
    assert "outside" not in zones


def test_preprocess_panel_image_with_reference_polygon():
    cfg = PreprocessConfig(tile_size=256)
    ref = np.array([[200, 150], [1300, 150], [1300, 850], [200, 850]], np.float32)
    result = preprocess_panel_image(FIXTURE, "G0F00000", cfg, reference_polygon=ref)
    # 應該直接套 reference 不重新偵測
    np.testing.assert_array_almost_equal(result.panel_polygon, ref)


def test_preprocess_panel_image_keeps_original_tiles_when_pipeline_enabled():
    cfg = PreprocessConfig(
        tile_size=256,
        image_preprocess_pipeline=[
            {"method": "bilateral", "params": {"diameter": 9, "sigma_color": 35.0, "sigma_space": 35.0}},
            {"method": "gaussian", "params": {"kernel_size": 5, "sigma": 1.0}},
        ],
    )
    result = preprocess_panel_image(FIXTURE, "STANDARD", cfg)
    assert result.tiles
    tile = result.tiles[0]
    assert tile.original_image is not None
    assert tile.preprocess_pipeline
    assert tile.original_image.shape == tile.image.shape
    assert len(result.preprocess_steps) == 2
    assert result.preprocess_total_ms >= 0.0
    assert all("elapsed_ms" in step for step in result.preprocess_steps)


def test_preprocess_panel_image_detects_boundary_from_raw_image(monkeypatch):
    from capi_image_preprocess_lab import apply_preprocess_pipeline as real_apply_pipeline

    captured = {}
    expected_polygon = np.array(
        [[200, 150], [1300, 150], [1300, 850], [200, 850]],
        np.float32,
    )

    def fake_detect_fast_panel_boundary(image, config, *, source_name=""):
        captured["boundary_image"] = image.copy()
        return (200, 150, 1300, 850), expected_polygon, True

    def destructive_model_pipeline(image, pipeline):
        result = real_apply_pipeline(image, pipeline)
        result["image"] = np.zeros_like(image)
        return result

    monkeypatch.setattr(
        capi_preprocess,
        "_detect_fast_panel_boundary",
        fake_detect_fast_panel_boundary,
    )
    monkeypatch.setattr(
        "capi_image_preprocess_lab.apply_preprocess_pipeline",
        destructive_model_pipeline,
    )

    cfg = PreprocessConfig(
        tile_size=256,
        generate_grid_tiles=False,
        cache_processed_image=True,
        fast_raw_boundary_enabled=True,
        image_preprocess_pipeline=[
            {"method": "mean", "params": {"kernel_size": 3}},
        ],
    )
    result = preprocess_panel_image(FIXTURE, "STANDARD", cfg)
    raw = capi_preprocess.cv2.imread(str(FIXTURE), capi_preprocess.cv2.IMREAD_GRAYSCALE)

    np.testing.assert_array_equal(captured["boundary_image"], raw)
    np.testing.assert_array_equal(result.processed_image, np.zeros_like(raw))
    np.testing.assert_array_equal(result.panel_polygon, expected_polygon)


def test_preprocess_panel_image_default_keeps_legacy_boundary_order(monkeypatch):
    from capi_image_preprocess_lab import apply_preprocess_pipeline as real_apply_pipeline

    captured = {}
    expected_polygon = np.array(
        [[200, 150], [1300, 150], [1300, 850], [200, 850]],
        np.float32,
    )

    def reject_fast_boundary(*args, **kwargs):
        raise AssertionError("default CAPI flow must not use fast raw boundary")

    def fake_detect_panel_polygon(image, config):
        captured["boundary_image"] = image.copy()
        return (200, 150, 1300, 850), expected_polygon

    def destructive_model_pipeline(image, pipeline):
        result = real_apply_pipeline(image, pipeline)
        result["image"] = np.zeros_like(image)
        return result

    monkeypatch.setattr(
        capi_preprocess,
        "_detect_fast_panel_boundary",
        reject_fast_boundary,
    )
    monkeypatch.setattr(
        capi_preprocess,
        "detect_panel_polygon",
        fake_detect_panel_polygon,
    )
    monkeypatch.setattr(
        "capi_image_preprocess_lab.apply_preprocess_pipeline",
        destructive_model_pipeline,
    )

    cfg = PreprocessConfig(
        tile_size=256,
        generate_grid_tiles=False,
        image_preprocess_pipeline=[
            {"method": "mean", "params": {"kernel_size": 3}},
        ],
    )
    result = preprocess_panel_image(FIXTURE, "STANDARD", cfg)

    np.testing.assert_array_equal(
        captured["boundary_image"],
        np.zeros_like(captured["boundary_image"]),
    )
    np.testing.assert_array_equal(result.panel_polygon, expected_polygon)


def test_fast_boundary_small_occupancy_falls_back_to_legacy(monkeypatch):
    from capi_image_preprocess_lab import apply_preprocess_pipeline as real_apply_pipeline

    captured = {}
    fast_polygon = np.array(
        [[400, 300], [1100, 300], [1100, 700], [400, 700]],
        np.float32,
    )
    legacy_polygon = np.array(
        [[200, 150], [1300, 150], [1300, 850], [200, 850]],
        np.float32,
    )

    def fake_fast_boundary(image, config, *, source_name=""):
        captured["fast_image"] = image.copy()
        return (400, 300, 1100, 700), fast_polygon, False

    def fake_legacy_boundary(image, config):
        captured["legacy_image"] = image.copy()
        return (200, 150, 1300, 850), legacy_polygon

    def destructive_model_pipeline(image, pipeline):
        result = real_apply_pipeline(image, pipeline)
        result["image"] = np.zeros_like(image)
        return result

    monkeypatch.setattr(
        capi_preprocess,
        "_detect_fast_panel_boundary",
        fake_fast_boundary,
    )
    monkeypatch.setattr(capi_preprocess, "detect_panel_polygon", fake_legacy_boundary)
    monkeypatch.setattr(
        "capi_image_preprocess_lab.apply_preprocess_pipeline",
        destructive_model_pipeline,
    )

    cfg = PreprocessConfig(
        tile_size=256,
        generate_grid_tiles=False,
        fast_raw_boundary_enabled=True,
        image_preprocess_pipeline=[
            {"method": "mean", "params": {"kernel_size": 3}},
        ],
    )
    result = preprocess_panel_image(FIXTURE, "STANDARD", cfg)
    raw = capi_preprocess.cv2.imread(str(FIXTURE), capi_preprocess.cv2.IMREAD_GRAYSCALE)

    np.testing.assert_array_equal(captured["fast_image"], raw)
    np.testing.assert_array_equal(
        captured["legacy_image"],
        np.zeros_like(captured["legacy_image"]),
    )
    np.testing.assert_array_equal(result.panel_polygon, legacy_polygon)


def test_preprocess_panel_image_aggregates_after_tiling_zone_timings():
    cfg = PreprocessConfig(
        tile_size=256,
        tile_stride=256,
        edge_threshold_px=384,
        preprocess_after_tiling=True,
        image_preprocess_pipelines={
            "inner": [{"method": "mean", "params": {"kernel_size": 3}}],
            "edge": [{"method": "median", "params": {"kernel_size": 3}}],
        },
    )

    result = preprocess_panel_image(FIXTURE, "STANDARD", cfg)

    assert {tile.zone for tile in result.tiles} == {"inner", "edge"}
    assert all(len(tile.preprocess_steps) == 1 for tile in result.tiles)
    assert all(
        tile.preprocess_steps[0]["method"]
        == ("mean" if tile.zone == "inner" else "median")
        for tile in result.tiles
    )
    assert len(result.preprocess_steps) == len(result.tiles)
    assert result.preprocess_total_ms == pytest.approx(
        sum(tile.preprocess_total_ms for tile in result.tiles)
    )


def test_preprocess_panel_image_can_skip_grid_tiles_and_cache_image():
    cfg = PreprocessConfig(
        tile_size=256,
        cache_processed_image=True,
        generate_grid_tiles=False,
        image_preprocess_pipeline=[
            {"method": "mean", "params": {"kernel_size": 3}},
        ],
    )
    result = preprocess_panel_image(FIXTURE, "STANDARD", cfg)
    assert result.foreground_bbox != (0, 0, 0, 0)
    assert result.panel_polygon is not None
    assert result.tiles == []
    assert result.processed_image is not None
    assert len(result.preprocess_steps) == 1


def test_preprocess_timing_summary_aggregates_steps():
    from capi_image_preprocess_lab import summarize_preprocess_timings

    summary = summarize_preprocess_timings([
        [
            {"index": 1, "method": "gaussian", "method_label": "高斯平滑", "applied_params": {"kernel_size": 5, "sigma": 1.0}, "elapsed_ms": 2.0},
            {"index": 2, "method": "laplace_sharpen", "method_label": "Laplace 銳化", "applied_params": {"kernel_size": 3, "strength": 0.5}, "elapsed_ms": 3.0},
        ],
        [
            {"index": 1, "method": "gaussian", "method_label": "高斯平滑", "applied_params": {"kernel_size": 5, "sigma": 1.0}, "elapsed_ms": 4.0},
        ],
    ])

    assert summary["total_elapsed_ms"] == pytest.approx(9.0)
    assert summary["steps"][0]["calls"] == 2
    assert summary["steps"][0]["elapsed_ms_total"] == pytest.approx(6.0)
    assert summary["steps"][0]["elapsed_ms_avg"] == pytest.approx(3.0)


def test_preprocess_panel_folder_uses_reference_polygon(tmp_path):
    # 複製 fixture 5 份模擬不同 lighting
    for lighting in ["STANDARD", "G0F00000", "R0F00000", "W0F00000", "WGF50500"]:
        shutil.copy(FIXTURE, tmp_path / f"{lighting}_x.png")
    cfg = PreprocessConfig(tile_size=256)
    results = preprocess_panel_folder(tmp_path, cfg)
    assert set(results.keys()) == {"STANDARD", "G0F00000", "R0F00000", "W0F00000", "WGF50500"}
    # 所有 lighting 應共用同一 polygon
    ref_poly = results["W0F00000"].panel_polygon
    for lighting, r in results.items():
        np.testing.assert_array_almost_equal(r.panel_polygon, ref_poly)


def test_preprocess_panel_folder_default_keeps_legacy_candidate_flow(monkeypatch, tmp_path):
    target = tmp_path / "G0F00000_x.png"
    w0f = tmp_path / "W0F00000_x.png"
    standard = tmp_path / "STANDARD_x.png"
    for path in (target, w0f, standard):
        path.write_bytes(b"stub")

    expected_polygon = np.array(
        [[1, 2], [11, 2], [11, 12], [1, 12]],
        np.float32,
    )
    preprocess_calls = []

    def reject_fast_boundary(*args, **kwargs):
        raise AssertionError("default CAPI folder flow must not probe raw boundary")

    def fake_preprocess_panel_image(
        image_path,
        lighting,
        config,
        reference_polygon=None,
        reference_bbox=None,
    ):
        preprocess_calls.append((lighting, reference_polygon is not None))
        detected_polygon = (
            reference_polygon
            if reference_polygon is not None
            else (expected_polygon if lighting == "STANDARD" else None)
        )
        return PanelPreprocessResult(
            image_path=Path(image_path),
            lighting=lighting,
            foreground_bbox=(0, 0, 10, 10),
            panel_polygon=detected_polygon,
            tiles=[],
            polygon_detection_failed=detected_polygon is None,
        )

    monkeypatch.setattr(
        capi_preprocess,
        "_detect_panel_boundary_file",
        reject_fast_boundary,
    )
    monkeypatch.setattr(capi_preprocess, "preprocess_panel_image", fake_preprocess_panel_image)

    results = preprocess_panel_folder(
        tmp_path,
        PreprocessConfig(tile_size=256),
        image_files=[target],
        boundary_reference_files=[target, w0f, standard],
    )

    assert preprocess_calls == [
        ("W0F00000", False),
        ("STANDARD", False),
        ("G0F00000", True),
    ]
    np.testing.assert_array_equal(results["G0F00000"].panel_polygon, expected_polygon)


def test_preprocess_panel_folder_prioritizes_w0f_reference(monkeypatch, tmp_path):
    for lighting in ["STANDARD", "G0F00000", "R0F00000", "W0F00000", "WGF50500"]:
        (tmp_path / f"{lighting}_x.png").write_bytes(b"stub")

    boundary_calls = []
    preprocess_calls = []
    polygons = {
        lighting: np.array([[idx, 0], [idx + 10, 0], [idx + 10, 10], [idx, 10]], np.float32)
        for idx, lighting in enumerate(["STANDARD", "G0F00000", "R0F00000", "W0F00000", "WGF50500"], 1)
    }

    def fake_detect_panel_boundary_file(image_path, config):
        lighting = capi_preprocess.canonical_image_prefix(Path(image_path).name)
        boundary_calls.append(lighting)
        return (0, 0, 10, 10), polygons[lighting], True

    def fake_preprocess_panel_image(
        image_path,
        lighting,
        config,
        reference_polygon=None,
        reference_bbox=None,
    ):
        preprocess_calls.append((lighting, reference_polygon is not None, reference_bbox))
        polygon = reference_polygon if reference_polygon is not None else polygons[lighting]
        return PanelPreprocessResult(
            image_path=Path(image_path),
            lighting=lighting,
            foreground_bbox=reference_bbox or (0, 0, 10, 10),
            panel_polygon=polygon,
            tiles=[],
            polygon_detection_failed=False,
        )

    monkeypatch.setattr(
        capi_preprocess,
        "_detect_panel_boundary_file",
        fake_detect_panel_boundary_file,
    )
    monkeypatch.setattr(capi_preprocess, "preprocess_panel_image", fake_preprocess_panel_image)

    results = preprocess_panel_folder(
        tmp_path,
        PreprocessConfig(tile_size=256, fast_raw_boundary_enabled=True),
    )

    assert boundary_calls == ["W0F00000"]
    assert preprocess_calls[0] == ("W0F00000", True, (0, 0, 10, 10))
    assert all(call[2] is None for call in preprocess_calls[1:])
    np.testing.assert_array_equal(results["W0F00000"].panel_polygon, polygons["W0F00000"])
    for lighting, result in results.items():
        if lighting != "W0F00000":
            np.testing.assert_array_equal(result.panel_polygon, polygons["W0F00000"])


def test_preprocess_panel_folder_uses_boundary_only_w0f_reference(monkeypatch, tmp_path):
    target = tmp_path / "G0F00000_x.png"
    reference = tmp_path / "W0F00000_x.png"
    target.write_bytes(b"stub")
    reference.write_bytes(b"stub")

    w0f_polygon = np.array([[1, 2], [11, 2], [11, 12], [1, 12]], np.float32)
    boundary_calls = []
    preprocess_calls = []

    def fake_detect_panel_boundary_file(image_path, config):
        boundary_calls.append(Path(image_path).name)
        return (0, 0, 10, 10), w0f_polygon, True

    def fake_preprocess_panel_image(
        image_path,
        lighting,
        config,
        reference_polygon=None,
        reference_bbox=None,
    ):
        preprocess_calls.append(
            (lighting, reference_polygon is not None, reference_bbox)
        )
        polygon = reference_polygon if reference_polygon is not None else w0f_polygon
        return PanelPreprocessResult(
            image_path=Path(image_path),
            lighting=lighting,
            foreground_bbox=reference_bbox or (0, 0, 10, 10),
            panel_polygon=polygon,
            tiles=[],
            polygon_detection_failed=False,
        )

    monkeypatch.setattr(
        capi_preprocess,
        "_detect_panel_boundary_file",
        fake_detect_panel_boundary_file,
    )
    monkeypatch.setattr(capi_preprocess, "preprocess_panel_image", fake_preprocess_panel_image)

    results = preprocess_panel_folder(
        tmp_path,
        PreprocessConfig(tile_size=256, fast_raw_boundary_enabled=True),
        image_files=[target],
        boundary_reference_files=[target, reference],
    )

    assert boundary_calls == ["W0F00000_x.png"]
    assert preprocess_calls == [("G0F00000", True, None)]
    assert set(results) == {"G0F00000"}
    np.testing.assert_array_equal(results["G0F00000"].panel_polygon, w0f_polygon)


def test_preprocess_panel_folder_small_occupancy_uses_legacy_flow(monkeypatch, tmp_path):
    target = tmp_path / "G0F00000_x.png"
    reference = tmp_path / "W0F00000_x.png"
    target.write_bytes(b"stub")
    reference.write_bytes(b"stub")

    polygon = np.array([[1, 2], [11, 2], [11, 12], [1, 12]], np.float32)
    boundary_calls = []
    preprocess_calls = []

    def fake_detect_panel_boundary_file(image_path, config):
        boundary_calls.append(Path(image_path).name)
        return (0, 0, 10, 10), polygon, False

    def fake_preprocess_panel_image(
        image_path,
        lighting,
        config,
        reference_polygon=None,
        reference_bbox=None,
    ):
        preprocess_calls.append(
            (
                lighting,
                reference_polygon is not None,
                config.fast_raw_boundary_enabled,
            )
        )
        result_polygon = (
            reference_polygon if reference_polygon is not None else polygon
        )
        return PanelPreprocessResult(
            image_path=Path(image_path),
            lighting=lighting,
            foreground_bbox=(0, 0, 10, 10),
            panel_polygon=result_polygon,
            tiles=[],
            polygon_detection_failed=False,
        )

    monkeypatch.setattr(
        capi_preprocess,
        "_detect_panel_boundary_file",
        fake_detect_panel_boundary_file,
    )
    monkeypatch.setattr(capi_preprocess, "preprocess_panel_image", fake_preprocess_panel_image)

    results = preprocess_panel_folder(
        tmp_path,
        PreprocessConfig(tile_size=256, fast_raw_boundary_enabled=True),
        image_files=[target],
        boundary_reference_files=[target, reference],
    )

    assert boundary_calls == ["W0F00000_x.png"]
    assert preprocess_calls == [
        ("W0F00000", False, False),
        ("G0F00000", True, False),
    ]
    np.testing.assert_array_equal(results["G0F00000"].panel_polygon, polygon)


def test_preprocess_panel_folder_fallbacks_are_boundary_only(monkeypatch, tmp_path):
    target = tmp_path / "G0F00000_x.png"
    w0f = tmp_path / "W0F00000_x.png"
    standard = tmp_path / "STANDARD_x.png"
    for path in (target, w0f, standard):
        path.write_bytes(b"stub")

    standard_polygon = np.array(
        [[1, 2], [11, 2], [11, 12], [1, 12]],
        np.float32,
    )
    boundary_calls = []
    preprocess_calls = []

    def fake_detect_panel_boundary_file(image_path, config):
        lighting = capi_preprocess.canonical_image_prefix(Path(image_path).name)
        boundary_calls.append(lighting)
        polygon = standard_polygon if lighting == "STANDARD" else None
        return (0, 0, 10, 10), polygon, True

    def fake_preprocess_panel_image(
        image_path,
        lighting,
        config,
        reference_polygon=None,
        reference_bbox=None,
    ):
        preprocess_calls.append(lighting)
        return PanelPreprocessResult(
            image_path=Path(image_path),
            lighting=lighting,
            foreground_bbox=reference_bbox,
            panel_polygon=reference_polygon,
            tiles=[],
            polygon_detection_failed=False,
        )

    monkeypatch.setattr(
        capi_preprocess,
        "_detect_panel_boundary_file",
        fake_detect_panel_boundary_file,
    )
    monkeypatch.setattr(capi_preprocess, "preprocess_panel_image", fake_preprocess_panel_image)

    results = preprocess_panel_folder(
        tmp_path,
        PreprocessConfig(tile_size=256, fast_raw_boundary_enabled=True),
        image_files=[target],
        boundary_reference_files=[target, w0f, standard],
    )

    assert boundary_calls == ["W0F00000", "STANDARD"]
    assert preprocess_calls == ["G0F00000"]
    np.testing.assert_array_equal(results["G0F00000"].panel_polygon, standard_polygon)


def test_preprocess_panel_image_with_preprocess_after_tiling():
    cfg = PreprocessConfig(
        tile_size=256,
        preprocess_after_tiling=True,
        image_preprocess_pipeline=[
            {"method": "gaussian", "params": {"kernel_size": 5, "sigma": 2.0}},
        ],
    )
    result = preprocess_panel_image(FIXTURE, "STANDARD", cfg)
    assert result.tiles
    
    # 大圖前處理被跳過，但每個 tile 的實際步驟仍須彙整供推論紀錄顯示。
    assert len(result.preprocess_steps) == len(result.tiles)
    assert result.preprocess_total_ms == pytest.approx(
        sum(tile.preprocess_total_ms for tile in result.tiles)
    )
    
    # 驗證每個 tile 的 image 確實套用了前處理，而 original_image 是原來的
    tile = result.tiles[0]
    assert tile.original_image is not None
    # 由於套用了模糊，處理後的 image 應與 original_image 不同
    assert not np.array_equal(tile.image, tile.original_image)
