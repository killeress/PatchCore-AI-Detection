"""Focused SoftPatch+ checks without downloading a backbone or training data."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sklearn.neighbors import LocalOutlierFactor
from torch import nn

from capi_patchcore_softpatch import (
    SoftPatchPlusCleaningCallback, SoftPatchPlusModel, enable_softpatch_model,
    percentile_rank, score_features,
)
from capi_softpatch_config import normalize_softpatch_config
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


class ToyExtractor(nn.Module):
    def forward(self, image):
        return {"layer2": image}


class ToyMap(nn.Module):
    def forward(self, scores, output_size):
        return torch.nn.functional.interpolate(scores, output_size)


def toy_model():
    model = PatchcoreModel.__new__(PatchcoreModel)
    nn.Module.__init__(model)
    model.register_buffer("memory_bank", torch.tensor([[0., 0.], [2., 0.], [0., 2.]]))
    model.feature_extractor = ToyExtractor()
    model.feature_pooler = nn.Identity()
    model.anomaly_map_generator = ToyMap()
    model.layers = ["layer2"]
    model.num_neighbors = 1
    model.tiler = None
    model.embedding_store = []
    return model.eval()


@pytest.mark.parametrize("raw", [
    {"soft_weight": "false"}, {"context_overlap": 1}, {"projection_dim": 2},
    {"reference_size": 200}, {"reference_size": 1024.5}, {"weight_strength": float("nan")},
    {"weight_strength": float("inf")}, {"weight_strength": True}, {"extra": 1},
    {"discriminator": "unknown"}, [],
])
def test_invalid_softpatch_options(raw):
    with pytest.raises(ValueError, match="softpatch_plus_config"):
        normalize_softpatch_config(raw)


def test_rank_ties_are_stable_and_constant_is_neutral():
    assert percentile_rank(torch.ones(4)).tolist() == [0.] * 4
    assert percentile_rank(torch.tensor([1., 2., 2., 4.])).tolist() == [0., .5, .5, 1.]


def test_lof_matches_independent_sklearn_calculation_and_preserves_input():
    x = torch.randn(35, 5, generator=torch.Generator().manual_seed(77))
    x[-1] += 20
    before = x.clone()
    options = normalize_softpatch_config({"discriminator": "lof"})
    actual, size, _ = score_features([x], torch.arange(len(x)), k=6, seed=42,
                                     options=options, query_chunk=7)
    estimator = LocalOutlierFactor(n_neighbors=6).fit(x.numpy())
    expected = percentile_rank(torch.from_numpy(-estimator.negative_outlier_factor_))
    torch.testing.assert_close(actual, expected)
    assert size == len(x)
    assert actual[-1] == 1
    assert torch.equal(before, x)


def test_projection_sampling_and_fusion_are_deterministic_across_chunks():
    x = torch.randn(350, 40, generator=torch.Generator().manual_seed(13))
    options = normalize_softpatch_config({"reference_size": 256, "projection_dim": 8})
    args = dict(reference_indices=torch.arange(350), k=6, seed=42, options=options)
    a, count, _ = score_features([x], query_chunk=13, **args)
    b, _, _ = score_features([x[:170], x[170:]], query_chunk=41, **args)
    torch.testing.assert_close(a, b, atol=.004, rtol=0)
    assert count == 256
    assert torch.isfinite(a).all() and (a >= 0).all() and (a <= 1).all()


@pytest.mark.parametrize("soft_weight,strength", [(True, 1.0), (False, 1.0), (True, 0.0)])
def test_keep_all_still_scores_and_coreset_weights_follow_selected_order(soft_weight, strength):
    model = toy_model()
    enable_softpatch_model(model)
    x = torch.randn(35, 2, generator=torch.Generator().manual_seed(8))
    model.embedding_store = [x.clone()]
    callback = SoftPatchPlusCleaningCallback(k=6, keep_ratio=1, options={
        "context_overlap": False, "soft_weight": soft_weight, "weight_strength": strength,
    })
    callback._select_coreset_indices = lambda bank, ratio: [30, 1, 12]
    module = SimpleNamespace(model=model)
    callback.on_train_start(None, module)
    callback.on_train_epoch_end(None, module)
    expected_weights = callback._soft_weights[[30, 1, 12]]
    model.subsample_embedding(.1)
    torch.testing.assert_close(model.memory_bank, x[[30, 1, 12]])
    torch.testing.assert_close(model.softpatch_weights, expected_weights)
    assert callback.stats["removed"] == 0
    if soft_weight and strength:
        assert model.softpatch_weights.max() > 1
    else:
        assert model.softpatch_weights.tolist() == [1., 1., 1.]


def test_cleaned_indices_and_weights_remain_aligned(monkeypatch):
    model = toy_model()
    enable_softpatch_model(model)
    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    model.embedding_store = [x.clone()]
    callback = SoftPatchPlusCleaningCallback(k=2, keep_ratio=.8, options={"context_overlap": False})
    callback._select_coreset_indices = lambda bank, ratio: [7, 0]
    scores = torch.arange(10, dtype=torch.float32) / 9
    monkeypatch.setattr("capi_patchcore_softpatch.score_features", lambda *a, **kw: (scores, 10, torch.device("cpu")))
    module = SimpleNamespace(model=model)
    callback.on_train_start(None, module)
    callback.on_train_epoch_end(None, module)
    model.subsample_embedding(.2)
    torch.testing.assert_close(model.memory_bank, x[[7, 0]])
    torch.testing.assert_close(model.softpatch_weights, (1 + scores)[[7, 0]])
    assert callback.stats["removed"] == 2


def test_overlap_disagreement_and_missing_coordinates_cannot_be_bypassed_by_weights(monkeypatch):
    model = SimpleNamespace(embedding_store=[torch.ones(6, 2)])
    callback = SoftPatchPlusCleaningCallback(k=1, keep_ratio=.8)
    plan = {
        "candidate_mask": torch.tensor([True, True, True, True, False, False]),
        "reference_indices": torch.arange(6), "overlap_groups": [[0, 1]],
        "stats": {},
    }
    monkeypatch.setattr(callback, "_build_context_cleaning_plan", lambda total: plan)
    scores = torch.tensor([1., .1, .2, .3, .8, .9])
    monkeypatch.setattr("capi_patchcore_softpatch.score_features", lambda *a, **kw: (scores, 6, torch.device("cpu")))
    callback.on_train_epoch_end(None, SimpleNamespace(model=model))
    assert callback.stats["removed"] == 0
    assert callback._soft_weights[[0, 1, 4, 5]].tolist() == [1.] * 4
    assert callback._soft_weights[3] > 1


@pytest.mark.parametrize("num_neighbors", [1, 3])
def test_unity_weights_match_existing_patchcore_and_persist(tmp_path, num_neighbors):
    model = toy_model()
    model.num_neighbors = num_neighbors
    image = torch.tensor([[[[.2, 1.5]], [[.1, .1]]]])
    expected = model(image)
    enable_softpatch_model(model)
    model.softpatch_weights = torch.ones(3)
    actual = model(image)
    torch.testing.assert_close(actual.pred_score, expected.pred_score)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map)
    path = tmp_path / "model.pt"
    torch.save({"model": model}, path)
    loaded = torch.load(path, weights_only=False)["model"]
    assert isinstance(loaded, SoftPatchPlusModel)
    torch.testing.assert_close(loaded(image).pred_score, actual.pred_score)
    reloaded = toy_model()
    enable_softpatch_model(reloaded)
    reloaded.load_state_dict(model.state_dict())
    torch.testing.assert_close(reloaded.softpatch_weights, model.softpatch_weights)


def test_weights_affect_map_and_mark_score_input_but_not_neighbor_selection():
    model = toy_model()
    enable_softpatch_model(model)
    model.softpatch_weights = torch.tensor([1., 3., 1.])
    image = torch.tensor([[[[.2, 1.5]], [[.1, .1]]]])
    embedding = image.permute(0, 2, 3, 1).reshape(-1, 2)
    raw, locations = model.nearest_neighbors(embedding, 1)
    expected = raw * model.softpatch_weights[locations]
    captured = {}

    def mark_score(scores, indices, features):
        captured["scores"] = scores
        return scores[:, 0]  # Simulate excluding a MARK-affected second patch.

    model.compute_anomaly_score = mark_score
    result = model(image)
    torch.testing.assert_close(captured["scores"].flatten(), expected)
    torch.testing.assert_close(result.anomaly_map.flatten(), expected)
    torch.testing.assert_close(result.pred_score, expected[:1])
    torch.testing.assert_close(model.nearest_neighbors(embedding, 1)[0], raw)


def test_missing_softpatch_weights_fail_explicitly():
    model = toy_model()
    enable_softpatch_model(model)
    with pytest.raises(RuntimeError, match="misaligned"):
        model(torch.ones(1, 2, 1, 1))


def test_weights_work_with_existing_fp16_knn_optimization():
    from capi_inference import CAPIInferencer

    class IsolatedSoftPatch(SoftPatchPlusModel):
        _fp16_patched = False

    model = toy_model()
    enable_softpatch_model(model)
    model.__class__ = IsolatedSoftPatch
    model.softpatch_weights = torch.tensor([1., 2., 1.])
    engine = CAPIInferencer.__new__(CAPIInferencer)
    engine._optimize_model_fp16(SimpleNamespace(model=model))
    image = torch.tensor([[[[.2, 1.5]], [[.1, .1]]]])
    result = model(image)
    query = image.half().permute(0, 2, 3, 1).reshape(-1, 2)
    raw, index = model.nearest_neighbors(query, 1)
    expected = raw * model.softpatch_weights[index]
    torch.testing.assert_close(result.anomaly_map.flatten(), expected)
    torch.testing.assert_close(result.pred_score, expected.max().reshape(1))
    assert model.memory_bank.dtype == torch.float16


@pytest.mark.parametrize("count,k", [(20, 6), (3, 6)])
def test_constant_and_insufficient_features_remain_neutral(count, k):
    callback = SoftPatchPlusCleaningCallback(k=k, keep_ratio=.5, options={"context_overlap": False})
    model = SimpleNamespace(embedding_store=[torch.full((count, 4), 3.)])
    callback.on_train_epoch_end(None, SimpleNamespace(model=model))
    assert callback.stats["removed"] == 0
    assert callback._soft_weights.tolist() == [1.] * count
    assert torch.equal(model.embedding_store[0], torch.full((count, 4), 3.))


def test_training_wiring_exports_weighted_model(tmp_path, monkeypatch):
    from capi_train_new import TrainingConfig, train_one_patchcore
    captured = {}

    class FakePatchcore:
        def __init__(self, **kwargs):
            self.model = toy_model()
            self.model.embedding_store = [torch.randn(40, 2, generator=torch.Generator().manual_seed(15))]

        @staticmethod
        def configure_pre_processor(**kwargs):
            return None

    class FakeEngine:
        def __init__(self, **kwargs):
            self.callbacks = kwargs["callbacks"]
            self.root = kwargs["default_root_dir"]

        def fit(self, *, model, **kwargs):
            callback = self.callbacks[0]
            assert isinstance(callback, SoftPatchPlusCleaningCallback)
            captured["callback"] = callback
            callback._select_coreset_indices = lambda bank, ratio: [0, 3, 7]
            callback.on_train_start(None, model)
            callback.on_validation_start(SimpleNamespace(sanity_checking=False), model)
            model.model.subsample_embedding(.1)

        def export(self, *, model, **kwargs):
            path = Path(self.root) / "weights/torch/model.pt"
            path.parent.mkdir(parents=True)
            torch.save({"model": model.model}, path)

    monkeypatch.setattr("capi_train_new._import_anomalib", lambda: (
        lambda **kw: SimpleNamespace(), FakePatchcore, FakeEngine,
        SimpleNamespace(TORCH="torch"), "same_as_test",
    ))
    staging = tmp_path / "staging"
    (staging / "train").mkdir(parents=True)
    config = TrainingConfig(machine_id="M", panel_paths=[], over_review_root=tmp_path,
        feature_cleaning_mode="softpatch_plus_v1", feature_cleaning_k=6,
        feature_cleaning_keep_ratio=.85, softpatch_plus_config={"context_overlap": False})
    stats = {}
    path = train_one_patchcore(staging, tmp_path / "run", "G0F00000-inner", cfg=config,
                              experiment_stats_out=stats)
    loaded = torch.load(path, weights_only=False)["model"]
    assert isinstance(loaded, SoftPatchPlusModel)
    assert loaded.softpatch_weights.shape == (3,)
    assert stats["feature_cleaning"]["mode"] == "softpatch_plus_v1"
    assert stats["feature_cleaning"]["k"] == 6
    assert stats["feature_cleaning"]["removed"] > 0
    assert torch.isfinite(loaded(torch.ones(1, 2, 1, 2)).pred_score).all()


@pytest.mark.parametrize("num_neighbors", [1, 9])
def test_real_backbone_coreset_torch_export_and_inferencer_reload(tmp_path, monkeypatch, num_neighbors):
    import numpy as np
    from anomalib.models import Patchcore
    from anomalib.deploy import TorchInferencer
    from capi_inference import CAPIInferencer

    # No pretrained download or production images. Exercise the actual
    # feature grid, coreset sampler, export format and deployed loader on CPU.
    monkeypatch.setenv("TRUST_REMOTE_CODE", "1")
    outer = Patchcore(backbone="resnet18", pre_trained=False, num_neighbors=num_neighbors,
                      pre_processor=False, post_processor=False, evaluator=False,
                      visualizer=False, coreset_sampling_ratio=.1)
    enable_softpatch_model(outer.model)
    callback = SoftPatchPlusCleaningCallback(k=6, keep_ratio=.95, options={"context_overlap": False})
    callback.on_train_start(None, outer)
    outer.model.train()
    training = torch.randn(4, 3, 64, 64, generator=torch.Generator().manual_seed(42))
    outer.model(training)
    callback.on_train_epoch_end(None, outer)
    outer.model.subsample_embedding(.1)
    outer.eval()
    query = training[:1] + .05
    with torch.no_grad():
        expected = outer.model(query)
    path = outer.to_torch(tmp_path)
    inferencer = TorchInferencer(path=path, device="cpu")
    actual = inferencer.predict(query)
    torch.testing.assert_close(actual.pred_score, expected.pred_score)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map.squeeze(1))
    assert inferencer.model.model.softpatch_weights.shape[0] == outer.model.memory_bank.shape[0]

    # A second exported model makes cache replacement observable in real scores.
    original_weights = outer.model.softpatch_weights.clone()
    outer.model.softpatch_weights *= 2
    updated_path = outer.to_torch(tmp_path / "updated")

    # The production optimizer patches the class; restore it after this test.
    for name in ("euclidean_dist", "nearest_neighbors", "_fp16_patched"):
        monkeypatch.setattr(SoftPatchPlusModel, name, getattr(SoftPatchPlusModel, name, False), raising=False)
    engine = CAPIInferencer.__new__(CAPIInferencer)
    engine.device = "cpu"
    engine.base_dir = tmp_path
    engine.config = SimpleNamespace(model_mapping={"G0F00000": {"inner": str(path)}})
    engine._model_cache_v2 = {}
    deployed = engine._get_model_for("M", "G0F00000", "inner")
    inner = deployed.model.model
    assert isinstance(inner, SoftPatchPlusModel)
    assert inner.memory_bank.dtype == torch.float16
    torch.testing.assert_close(inner.softpatch_weights, original_weights)

    # Exercise the production uint8 tile entry point, precision hook and KNN.
    tile = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    score, anomaly_map = engine._predict_tile(deployed, tile)
    assert np.isfinite(score) and score > 0
    assert torch.isfinite(torch.as_tensor(anomaly_map)).all()

    engine.config.model_mapping["G0F00000"]["inner"] = str(updated_path)
    assert engine._get_model_for("M", "G0F00000", "inner") is deployed
    assert engine.reload_submodel("M", "G0F00000", "inner") is True
    updated = engine._get_model_for("M", "G0F00000", "inner")
    assert updated is not deployed
    updated_score, updated_map = engine._predict_tile(updated, tile)
    torch.testing.assert_close(updated.model.model.softpatch_weights, original_weights * 2)
    assert updated_score == pytest.approx(score * 2, rel=1e-5)
    torch.testing.assert_close(torch.as_tensor(updated_map), torch.as_tensor(anomaly_map) * 2)


def test_api_accepts_softpatch_settings_and_preserves_legacy_limits():
    from capi_web import CAPIWebHandler
    from capi_train_new import TrainingConfig, apply_user_training_params, feature_cleaning_config_for_zone
    raw = {"feature_cleaning_mode": "softpatch_plus_v1", "feature_cleaning_k": 6,
           "feature_cleaning_keep_ratio": .85, "softpatch_plus_config": {"soft_weight": False}}
    parsed, error = CAPIWebHandler._validate_training_params(raw)
    assert error is None
    cfg = TrainingConfig(machine_id="M", panel_paths=[], over_review_root=Path("."))
    apply_user_training_params(cfg, parsed)
    assert cfg.softpatch_plus_config["discriminator"] == "lof_gaussian"
    assert cfg.softpatch_plus_config["soft_weight"] is False
    assert feature_cleaning_config_for_zone(cfg, "inner")["k"] == 6
    assert feature_cleaning_config_for_zone(cfg, "edge")["mode"] == "off"
    assert CAPIWebHandler._validate_training_params({"feature_cleaning_keep_ratio": .85})[1]
    assert CAPIWebHandler._validate_training_params({"feature_cleaning_k": 6.5})[1]
    assert CAPIWebHandler._validate_training_params({"feature_cleaning_k": float("inf")})[1]
    zones = {"inner": {"mode": "softpatch_plus_v1", "k": 6, "keep_ratio": .5},
             "edge": {"mode": "context_overlap_adaptive_v1", "k": 30, "keep_ratio": .999}}
    parsed, error = CAPIWebHandler._validate_training_params({"feature_cleaning_by_zone": zones})
    assert error is None and parsed["feature_cleaning_by_zone"] == zones
    assert {"feature_cleaning_k", "softpatch_plus_config"} <= CAPIWebHandler.PATCHCORE_BUNDLE_LOCKED_TRAINING_PARAMS
