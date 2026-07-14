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
    assert callback.seed == 42
    assert callback.stats == {}


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
