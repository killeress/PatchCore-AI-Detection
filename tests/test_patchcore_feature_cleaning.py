from types import SimpleNamespace

import pytest
import torch

from capi_patchcore_feature_cleaning import FeatureDensityCleaningCallback


def _model_with_store(*embeddings: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(embedding_store=list(embeddings)))


def test_default_recipe():
    callback = FeatureDensityCleaningCallback()

    assert callback.k == 30
    assert callback.keep_ratio == 0.99
    assert callback.center_size is None
    assert callback.seed == 42
    assert callback.stats == {}


def _context_callback_with_two_overlapping_tiles(tmp_path, *, keep_ratio=0.75):
    first = tmp_path / "tile_a.png"
    second = tmp_path / "tile_b.png"
    panel = tmp_path / "panel"
    panel.mkdir()
    callback = FeatureDensityCleaningCallback(
        k=1,
        keep_ratio=keep_ratio,
        reference_size=8,
        strategy="context_overlap_adaptive",
        trace_sources={
            str(first.resolve()): {
                "source_path": str(first.resolve()),
                "panel_path": str(panel.resolve()),
                "tile_x": 0,
                "tile_y": 0,
                "tile_width": 8,
                "tile_height": 8,
            },
            str(second.resolve()): {
                "source_path": str(second.resolve()),
                "panel_path": str(panel.resolve()),
                "tile_x": 4,
                "tile_y": 0,
                "tile_width": 8,
                "tile_height": 8,
            },
        },
    )
    callback._batch_layouts = [{
        "image_paths": [str(first.resolve()), str(second.resolve())],
        "input_size": [8, 8],
        "grid_size": [1, 4],
        "embedding_count": 8,
    }]
    return callback


def test_context_plan_uses_best_overlapping_view_and_auto_guard(tmp_path):
    callback = _context_callback_with_two_overlapping_tiles(tmp_path)

    plan = callback._build_context_cleaning_plan(8)

    assert plan["candidate_mask"].tolist() == [False, False, True, True, True, True, False, False]
    assert plan["reference_indices"].tolist() == [0, 1, 2, 5, 6, 7]
    assert plan["stats"]["context_overlap_groups"] == 2
    assert plan["stats"]["context_auto_guard_px"] == 4.0


def test_rejected_neighbor_overlap_is_forced_out_of_accepted_tile(tmp_path):
    accepted = tmp_path / "t0011.png"
    accepted.write_bytes(b"tile")
    panel = tmp_path / "panel"
    panel.mkdir()
    callback = FeatureDensityCleaningCallback(
        k=1,
        keep_ratio=0.94,
        reference_size=4096,
        strategy="context_overlap_adaptive",
        trace_sources={
            str(accepted.resolve()): {
                "tile_pool_id": 141168,
                "source_path": str(accepted.resolve()),
                "panel_path": str(panel.resolve()),
                "tile_x": 4694,
                "tile_y": 736,
                "tile_width": 512,
                "tile_height": 512,
            },
        },
        rejected_trace_sources=[{
            "tile_pool_id": 141167,
            "panel_path": str(panel.resolve()),
            "tile_x": 4331,
            "tile_y": 735,
            "tile_width": 512,
            "tile_height": 512,
        }],
    )
    callback._batch_layouts = [{
        "image_paths": [str(accepted.resolve())],
        "input_size": [512, 512],
        "grid_size": [64, 64],
        "embedding_count": 64 * 64,
    }]

    plan = callback._build_context_cleaning_plan(64 * 64)
    rejected = plan["forced_exclude_mask"].reshape(64, 64)

    # t0010/t0011 overlap is 149 px. Feature centers 4..148 (cols 0..18)
    # are covered by the rejected t0010 and must not reach KNN or coreset.
    assert bool(rejected[:, :19].all())
    assert not bool(rejected[:, 19:].any())
    assert not bool(plan["candidate_mask"][plan["forced_exclude_mask"]].any())
    assert not set(torch.nonzero(plan["forced_exclude_mask"]).flatten().tolist()) & set(
        plan["reference_indices"].tolist()
    )
    assert plan["stats"]["context_rejected_tiles"] == 1
    assert plan["stats"]["context_rejected_overlap_features"] == 64 * 19


def test_rejected_overlap_applies_even_when_no_distance_candidates(tmp_path):
    accepted = tmp_path / "accepted.png"
    accepted.write_bytes(b"tile")
    panel = tmp_path / "panel"
    panel.mkdir()
    raw = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    model = _model_with_store(raw.clone())
    callback = FeatureDensityCleaningCallback(
        k=1,
        keep_ratio=0.75,
        reference_size=4,
        strategy="context_overlap_adaptive",
        trace_sources={
            str(accepted.resolve()): {
                "tile_pool_id": 11,
                "source_path": str(accepted.resolve()),
                "panel_path": str(panel.resolve()),
                "tile_x": 4,
                "tile_y": 0,
                "tile_width": 8,
                "tile_height": 8,
            },
        },
        rejected_trace_sources=[{
            "tile_pool_id": 10,
            "panel_path": str(panel.resolve()),
            "tile_x": 0,
            "tile_y": 0,
            "tile_width": 8,
            "tile_height": 8,
        }],
    )
    callback._batch_layouts = [{
        "image_paths": [str(accepted.resolve())],
        "input_size": [8, 8],
        "grid_size": [1, 4],
        "embedding_count": 4,
    }]

    callback.on_train_epoch_end(None, model)
    callback.record_coreset_indices([0])

    assert torch.equal(torch.cat(model.model.embedding_store), raw[2:])
    assert callback.stats["reason"] == "rejected_overlap_only"
    assert callback.stats["rejected_overlap_excluded"] == 2
    trace = callback.stats["patch_trace"][0]
    assert trace["reason_codes"][:2] == [4, 4]
    assert trace["removed_indices"] == [0, 1]
    # Coreset index 0 after filtering maps back to original feature index 2.
    assert trace["coreset_indices"] == [2]


def test_rejected_overlap_applies_when_total_does_not_exceed_k(tmp_path):
    accepted = tmp_path / "accepted-small.png"
    accepted.write_bytes(b"tile")
    panel = tmp_path / "panel-small"
    panel.mkdir()
    raw = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    model = _model_with_store(raw.clone())
    callback = FeatureDensityCleaningCallback(
        k=4,
        keep_ratio=0.75,
        reference_size=5,
        strategy="context_overlap_adaptive",
        trace_sources={
            str(accepted.resolve()): {
                "tile_pool_id": 11,
                "source_path": str(accepted.resolve()),
                "panel_path": str(panel.resolve()),
                "tile_x": 4,
                "tile_y": 0,
                "tile_width": 8,
                "tile_height": 8,
            },
        },
        rejected_trace_sources=[{
            "tile_pool_id": 10,
            "panel_path": str(panel.resolve()),
            "tile_x": 0,
            "tile_y": 0,
            "tile_width": 8,
            "tile_height": 8,
        }],
    )
    callback._batch_layouts = [{
        "image_paths": [str(accepted.resolve())],
        "input_size": [8, 8],
        "grid_size": [1, 4],
        "embedding_count": 4,
    }]

    callback.on_train_epoch_end(None, model)

    assert torch.equal(torch.cat(model.model.embedding_store), raw[2:])
    assert callback.stats["reason"] == "rejected_overlap_only"
    assert callback.stats["rejected_overlap_excluded"] == 2


def test_coreset_wrapper_records_exact_original_cell_indices(tmp_path, monkeypatch):
    source = tmp_path / "tile.png"
    source.write_bytes(b"tile")
    raw = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.8, 0.2],
        [-1.0, 0.0],
    ])
    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_store = [raw.clone()]
            self.register_buffer("memory_bank", torch.empty(0))

        def subsample_embedding(self, _ratio):
            pass

    inner = Inner()
    model = SimpleNamespace(model=inner)
    callback = FeatureDensityCleaningCallback(
        k=1,
        keep_ratio=0.75,
        reference_size=4,
        trace_sources={
            str(source.resolve()): {
                "tile_pool_id": 7,
                "source_path": str(source.resolve()),
            },
        },
    )
    callback._batch_layouts = [{
        "image_paths": [str(source.resolve())],
        "input_size": [8, 8],
        "grid_size": [2, 2],
        "embedding_count": 4,
    }]
    distances = torch.tensor([0.0, 0.1, 0.2, 1.0])
    monkeypatch.setattr(
        callback,
        "_kth_cosine_distances",
        lambda *args, **kwargs: (distances, torch.device("cpu")),
    )
    monkeypatch.setattr(
        callback,
        "_select_coreset_indices",
        lambda _memory_bank, _ratio: [0, 2],
    )

    callback._install_coreset_trace(model)
    callback.on_train_epoch_end(None, model)
    inner.subsample_embedding(0.5)

    assert torch.equal(inner.memory_bank, raw[[0, 2]])
    assert callback.stats["coreset_selected"] == 2
    assert callback.stats["patch_trace"][0]["coreset_indices"] == [0, 2]


def test_overlap_consensus_rejects_tile_edge_only_outlier(tmp_path, monkeypatch):
    raw = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    model = _model_with_store(raw.clone())
    callback = _context_callback_with_two_overlapping_tiles(tmp_path)
    # Physical x=5 is indices 2/4: only the Tile-edge view (4) is high.
    # Physical x=7 is indices 3/5: both overlapping views are high.
    distances = torch.tensor([0.0, 0.0, 0.1, 1.0, 1.0, 1.0, 0.0, 0.0])
    monkeypatch.setattr(
        callback,
        "_kth_cosine_distances",
        lambda *args, **kwargs: (distances, torch.device("cpu")),
    )
    monkeypatch.setattr(
        callback,
        "_adaptive_threshold",
        lambda values: (torch.tensor(0.5), {"threshold_method": "test"}),
    )

    callback.on_train_epoch_end(None, model)

    assert torch.equal(torch.cat(model.model.embedding_store), raw[[0, 1, 2, 4, 6, 7]])
    assert callback.stats["raw_outlier_count"] == 3
    assert callback.stats["consensus_removed_count"] == 2
    assert callback.stats["removed"] == 2


def test_adaptive_threshold_allows_zero_removal(tmp_path, monkeypatch):
    raw = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    model = _model_with_store(raw.clone())
    callback = _context_callback_with_two_overlapping_tiles(tmp_path)
    distances = torch.full((8,), 0.2)
    monkeypatch.setattr(
        callback,
        "_kth_cosine_distances",
        lambda *args, **kwargs: (distances, torch.device("cpu")),
    )

    callback.on_train_epoch_end(None, model)

    assert torch.equal(torch.cat(model.model.embedding_store), raw)
    assert callback.stats["threshold"] == pytest.approx(0.2)
    assert callback.stats["adaptive_mad"] == 0.0
    assert callback.stats["raw_outlier_count"] == 0
    assert callback.stats["removed"] == 0


def test_context_mode_protects_tiles_without_coordinates(tmp_path):
    source = tmp_path / "legacy_tile.png"
    raw = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    model = _model_with_store(raw.clone())
    callback = FeatureDensityCleaningCallback(
        k=1,
        keep_ratio=0.75,
        reference_size=4,
        strategy="context_overlap_adaptive",
        trace_sources={
            str(source.resolve()): {"source_path": str(source.resolve())},
        },
    )
    callback._batch_layouts = [{
        "image_paths": [str(source.resolve())],
        "input_size": [4, 4],
        "grid_size": [2, 2],
        "embedding_count": 4,
    }]

    callback.on_train_epoch_end(None, model)

    assert torch.equal(torch.cat(model.model.embedding_store), raw)
    assert callback.stats["reason"] == "no_cleaning_candidates"
    assert callback.stats["context_missing_metadata"] == 4
    assert callback.stats["removed"] == 0


def test_validation_hook_runs_before_epoch_end_fallback():
    raw = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [-1.0, 0.0]])
    model = _model_with_store(raw.clone())
    callback = FeatureDensityCleaningCallback(k=2, keep_ratio=0.75, reference_size=4)

    callback.on_validation_start(SimpleNamespace(sanity_checking=False), model)
    cleaned_after_validation = torch.cat(model.model.embedding_store).clone()
    callback.on_train_epoch_end(None, model)

    assert cleaned_after_validation.shape[0] == 3
    assert torch.equal(torch.cat(model.model.embedding_store), cleaned_after_validation)


def test_sanity_validation_does_not_consume_cleaning_hook():
    raw = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [-1.0, 0.0]])
    model = _model_with_store(raw.clone())
    callback = FeatureDensityCleaningCallback(k=2, keep_ratio=0.75, reference_size=4)

    callback.on_validation_start(SimpleNamespace(sanity_checking=True), model)

    assert callback.stats == {}
    assert torch.equal(model.model.embedding_store[0], raw)
    callback.on_validation_start(SimpleNamespace(sanity_checking=False), model)
    assert callback.stats["applied"] is True


def test_keep_ratio_one_skips_distance_cleaning():
    raw = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [-1.0, 0.0]])
    model = _model_with_store(raw.clone())
    callback = FeatureDensityCleaningCallback(k=2, keep_ratio=1.0, reference_size=4)

    callback.on_train_epoch_end(None, model)

    assert torch.equal(model.model.embedding_store[0], raw)
    assert callback.stats["applied"] is False
    assert callback.stats["removed"] == 0
    assert callback.stats["reason"] == "keep_all"


def test_removes_isolated_feature_and_preserves_raw_embeddings():
    raw = torch.tensor(
        [[10.0, 0.0], [9.0, 0.1], [11.0, -0.1], [8.0, 0.05], [-3.0, 0.0]],
        dtype=torch.float64,
    )
    model = _model_with_store(raw[:3].clone(), raw[3:].clone())
    callback = FeatureDensityCleaningCallback(k=2, keep_ratio=0.8, reference_size=5, query_chunk=2)

    callback.on_train_epoch_end(None, model)

    cleaned = torch.cat(model.model.embedding_store)
    assert torch.equal(cleaned, raw[:4])
    assert cleaned.dtype == torch.float64
    assert callback.stats["total"] == 5
    assert callback.stats["kept"] == 4
    assert callback.stats["removed"] == 1
    assert callback.stats["removed_ratio"] == pytest.approx(0.2)
    assert callback.stats["reference_size"] == 5
    assert callback.stats["threshold"] is not None
    assert callback.stats["elapsed_seconds"] >= 0
    assert callback.stats["applied"] is True
    assert callback.stats["reason"] == "completed"


def test_removed_patch_trace_maps_keep_mask_back_to_source_grid(tmp_path):
    source = tmp_path / "tile.png"
    source.write_bytes(b"tile")
    raw = torch.tensor([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.98, -0.01],
        [-1.0, 0.0],
    ])
    model = _model_with_store(raw.clone())
    callback = FeatureDensityCleaningCallback(
        k=2,
        keep_ratio=0.75,
        reference_size=4,
        trace_sources={
            str(source.resolve()): {
                "tile_pool_id": 7,
                "source_path": str(source.resolve()),
            }
        },
    )
    callback._current_grid_shape = (2, 2)
    batch = SimpleNamespace(
        image_path=[str(source)],
        image=torch.zeros(1, 3, 8, 8),
    )

    callback.on_train_batch_end(None, model, None, batch, 0)
    callback.on_train_epoch_end(None, model)

    trace = callback.stats["removed_patch_trace"]
    assert trace == [{
        "tile_pool_id": 7,
        "source_path": str(source.resolve()),
        "input_size": [8, 8],
        "grid_size": [2, 2],
        "removed_indices": [3],
        "removed_count": 1,
    }]
    assert callback.stats["removed"] == 1


def test_center_cleaning_removes_only_center_candidates(monkeypatch):
    raw = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    model = _model_with_store(raw.clone())
    callback = FeatureDensityCleaningCallback(
        k=2,
        keep_ratio=0.75,
        center_size=4,
        reference_size=16,
    )
    callback._batch_layouts = [{
        "image_paths": ["tile.png"],
        "input_size": [8, 8],
        "grid_size": [4, 4],
        "embedding_count": 16,
    }]
    distances = torch.zeros(16)
    distances[0] = 10.0  # 外圍即使最離群也必須保留
    distances[10] = 9.0  # 中央候選區的離群 feature 應被移除
    monkeypatch.setattr(
        callback,
        "_kth_cosine_distances",
        lambda *args, **kwargs: (distances, torch.device("cpu")),
    )

    callback.on_train_epoch_end(None, model)

    cleaned = torch.cat(model.model.embedding_store)
    assert torch.equal(cleaned, torch.cat([raw[:10], raw[11:]]))
    assert torch.equal(cleaned[0], raw[0])
    assert callback.stats["cleaning_candidates"] == 4
    assert callback.stats["protected"] == 12
    assert callback.stats["removed"] == 1
    assert callback.stats["center_size"] == 4


def test_default_384_center_maps_to_48_by_48_features_on_64_grid():
    callback = FeatureDensityCleaningCallback(center_size=384)
    callback._batch_layouts = [{
        "image_paths": ["tile.png"],
        "input_size": [512, 512],
        "grid_size": [64, 64],
        "embedding_count": 64 * 64,
    }]

    mask = callback._build_cleaning_candidate_mask(64 * 64).reshape(64, 64)

    assert int(mask.sum().item()) == 48 * 48
    assert bool(mask[8:56, 8:56].all())
    assert not bool(mask[:8].any())
    assert not bool(mask[56:].any())
    assert not bool(mask[:, :8].any())
    assert not bool(mask[:, 56:].any())


def test_sampled_reference_is_deterministic_across_query_chunks():
    generator = torch.Generator().manual_seed(7)
    raw = torch.randn(40, 8, generator=generator) * torch.arange(1, 41).unsqueeze(1)
    first_model = _model_with_store(raw[:17].clone(), raw[17:].clone())
    second_model = _model_with_store(raw[:17].clone(), raw[17:].clone())
    first = FeatureDensityCleaningCallback(
        k=4, keep_ratio=0.8, seed=42, reference_size=13, query_chunk=3
    )
    second = FeatureDensityCleaningCallback(
        k=4, keep_ratio=0.8, seed=42, reference_size=13, query_chunk=11
    )

    first.on_train_epoch_end(None, first_model)
    second.on_train_epoch_end(None, second_model)

    assert torch.equal(torch.cat(first_model.model.embedding_store), torch.cat(second_model.model.embedding_store))
    for key in first.stats.keys() - {"elapsed_seconds"}:
        assert first.stats[key] == second.stats[key]


def test_non_finite_features_fail_without_changing_store():
    raw = torch.tensor([[1.0, 0.0], [0.9, 0.1], [float("nan"), 0.0], [0.8, 0.2]])
    model = _model_with_store(raw.clone())
    callback = FeatureDensityCleaningCallback(k=2, keep_ratio=0.75, reference_size=4)

    with pytest.raises(RuntimeError, match="non-finite PatchCore embeddings"):
        callback.on_train_epoch_end(None, model)

    assert torch.allclose(model.model.embedding_store[0], raw, equal_nan=True)
    assert callback.stats == {}


@pytest.mark.parametrize("count", [0, 3])
def test_too_few_features_are_left_unchanged(count):
    raw = torch.arange(count * 2, dtype=torch.float32).reshape(count, 2)
    model = _model_with_store(raw.clone())
    callback = FeatureDensityCleaningCallback(k=3, keep_ratio=0.5, reference_size=10, query_chunk=2)

    callback.on_train_epoch_end(None, model)

    assert torch.equal(model.model.embedding_store[0], raw)
    assert callback.stats["total"] == count
    assert callback.stats["kept"] == count
    assert callback.stats["removed"] == 0
    assert callback.stats["threshold"] is None
    assert callback.stats["applied"] is False
    assert callback.stats["reason"] == "insufficient_features"
