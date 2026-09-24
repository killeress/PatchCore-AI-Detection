"""Precision compatibility for exported models used during retraining."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from anomalib.data import InferenceBatch
from anomalib.deploy import TorchInferencer

# Install the same compatibility patch as the server before loading candidates.
import capi_inference  # noqa: F401


class TinyExportedModel(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.feature_extractor = torch.nn.Conv2d(3, 1, 1, bias=False).to(dtype)
        torch.nn.init.constant_(self.feature_extractor.weight, 0.1)
        # Legacy metadata and KNN buffers need not match the backbone weights.
        self.precision = "float16"
        self.register_buffer("memory_bank", torch.zeros(1, dtype=torch.float16))

    def forward(self, batch):
        image = batch.image if hasattr(batch, "image") else batch
        with torch.no_grad():
            anomaly_map = self.feature_extractor(image).abs().squeeze(1)
        return InferenceBatch(
            pred_score=anomaly_map.flatten(1).amax(1), anomaly_map=anomaly_map,
        )


class WrappedExportedModel(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.model = TinyExportedModel(dtype)

    def forward(self, batch):
        return self.model(batch)


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("model_type", [TinyExportedModel, WrappedExportedModel])
@pytest.mark.parametrize("input_type", ["numpy", "float32", "float16"])
def test_exported_inferencer_matches_weight_precision(tmp_path, device, dtype, model_type, input_type):
    model = model_type(dtype)
    path = tmp_path / "model.pt"
    torch.save({"model": model}, path)
    inferencer = TorchInferencer(path, device=device)
    original_forward = inferencer.model.forward
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    if input_type != "numpy":
        image = torch.full((1, 3, 8, 8), 128 / 255, dtype=getattr(torch, input_type))

    result = inferencer.predict(image)

    assert result.pred_score.item() == pytest.approx(0.3 * 128 / 255, abs=0.001)
    assert next(inferencer.model.parameters()).dtype == dtype
    assert inferencer.model.forward == original_forward


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_training_calibration_loads_exported_model(tmp_path, dtype, monkeypatch):
    from capi_train_new import _calibrate_from_model

    # A standalone training process has not imported capi_inference.
    monkeypatch.setattr(TorchInferencer, "predict", TorchInferencer.predict.__wrapped__)

    path = tmp_path / "model.pt"
    torch.save({"model": WrappedExportedModel(dtype)}, path)
    ok = tmp_path / "ok.png"
    ng = tmp_path / "ng.png"
    assert cv2.imwrite(str(ok), np.full((8, 8, 3), 64, dtype=np.uint8))
    assert cv2.imwrite(str(ng), np.full((8, 8, 3), 192, dtype=np.uint8))

    maximum, train_scores, ng_scores = _calibrate_from_model(path, [ok], [ng])

    assert train_scores == [maximum]
    assert maximum == pytest.approx(0.3 * 64 / 255, abs=0.001)
    assert ng_scores == pytest.approx([0.3 * 192 / 255], abs=0.001)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_partial_training_validation_scores_exported_candidate(tmp_path, dtype, monkeypatch):
    from capi_training_validation import evaluate_model

    monkeypatch.setattr(TorchInferencer, "predict", TorchInferencer.predict.__wrapped__)

    path = tmp_path / "U0F00000-inner.pt"
    torch.save({"model": WrappedExportedModel(dtype)}, path)
    samples = []
    for role in ("calibration", "acceptance"):
        for label, brightness in (("ok", 64), ("ng", 192)):
            name = f"{role}_{label}.png"
            assert cv2.imwrite(str(tmp_path / name), np.full((8, 8, 3), brightness, np.uint8))
            samples.append({"tile_id": len(samples) + 1, "asset_path": name,
                            "role": role, "label": label, "group": role})

    result = evaluate_model(
        path, [], [], {}, tmp_path, "U0F00000-inner", lambda _: None,
        frozen_inputs=(samples, []), job_id="partial", fail_on_error=True,
    )

    report = json.loads((tmp_path / result["report_path"]).read_text(encoding="utf-8"))
    assert report["reasons"] == ["尚未同時設定可接受誤判率與漏檢率，僅提供建議與實測值。"]
    assert [item["score"] for item in report["samples"]] == pytest.approx(
        [0.3 * 64 / 255, 0.3 * 192 / 255] * 2, abs=0.001,
    )


def test_batch_input_precision_and_hooks_restored_on_failure():
    from capi_torch_compat import patch_torch_inferencer_precision
    model = TinyExportedModel(torch.float16)
    original_forward = model.forward

    class BatchInferencer:
        def __init__(self, model):
            self.model = model

        def predict(self, batch):
            result = self.model(batch)
            assert result.pred_score.item() > 0
            raise RuntimeError("post-processing failed")

    patch_torch_inferencer_precision(BatchInferencer)
    first_patch = BatchInferencer.predict
    patch_torch_inferencer_precision(BatchInferencer)
    assert BatchInferencer.predict is first_patch
    batch = SimpleNamespace(image=torch.ones(1, 3, 8, 8))
    with pytest.raises(RuntimeError, match="post-processing failed"):
        BatchInferencer(model).predict(batch)
    assert model.forward == original_forward
    assert not model._forward_pre_hooks
    assert not model.feature_extractor._forward_pre_hooks


class RecastingModel(TinyExportedModel):
    def forward(self, batch):
        return super().forward(batch.to(dtype=self.memory_bank.dtype))


@pytest.mark.parametrize("dtype,bank_dtype", [(torch.float32, torch.float16), (torch.float16, torch.float32)])
def test_internal_memory_bank_cast_cannot_override_backbone_precision(tmp_path, dtype, bank_dtype):
    model = RecastingModel(dtype)
    model.memory_bank = model.memory_bank.to(dtype=bank_dtype)
    path = tmp_path / "recasting.pt"
    torch.save({"model": model}, path)
    inferencer = TorchInferencer(path, device="cpu")
    result = inferencer.predict(torch.ones(1, 3, 8, 8))
    assert result.pred_score.item() == pytest.approx(.3, abs=.001)
    assert inferencer.model.memory_bank.dtype == bank_dtype


class MixedBankModel(TinyExportedModel):
    def __init__(self):
        super().__init__(torch.float32)
        self.memory_bank = torch.ones(1, 64, dtype=torch.float16)

    def forward(self, batch):
        with torch.no_grad():
            features = self.feature_extractor(batch).squeeze(1)
            scores = features.flatten(1) @ self.memory_bank.T
        return InferenceBatch(pred_score=scores.flatten(), anomaly_map=features)


def test_scorer_repairs_half_memory_bank_even_with_float_weights(tmp_path):
    from capi_inference import SubmodelScorer

    path = tmp_path / "mixed.pt"
    torch.save({"model": MixedBankModel()}, path)
    scorer = SubmodelScorer(gpu_lock=None, db=None, log_fn=lambda _: None)
    inferencer = scorer._load_inferencer_for_pt(path)
    score = scorer._score_one_tile(np.full((8, 8, 3), 128, np.uint8), inferencer)
    assert score == pytest.approx(64 * .3 * 128 / 255, abs=.001)
    assert inferencer.model.memory_bank.dtype == torch.float32
