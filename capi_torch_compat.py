"""Shared inference precision compatibility for server and training processes."""
from functools import wraps

import torch


def patch_torch_inferencer_precision(inferencer_class):
    """Align images with backbone weights, regardless of checkpoint metadata."""
    original = getattr(inferencer_class, "predict", None)
    if original is None or getattr(original, "_capi_weight_precision", False):
        return

    @wraps(original)
    def predict(self, *args, **kwargs):
        model = self.model
        inner = getattr(model, "model", model)
        backbone = getattr(inner, "feature_extractor", inner)
        if not isinstance(backbone, torch.nn.Module):
            return original(self, *args, **kwargs)
        dtype = next((p.dtype for p in backbone.parameters() if p.is_floating_point()), None)
        if dtype is None:
            return original(self, *args, **kwargs)

        def align_input(_module, inputs):
            if not inputs:
                return inputs
            batch, *rest = inputs
            if isinstance(batch, torch.Tensor):
                batch = batch.to(dtype=dtype)
            elif isinstance(getattr(batch, "image", None), torch.Tensor):
                batch.image = batch.image.to(dtype=dtype)
            return (batch, *rest)

        handles = []
        try:
            handles.append(model.register_forward_pre_hook(align_input))
            if backbone is not model:
                # Some versions cast again inside PatchCore using memory-bank
                # precision. Align once more immediately before the backbone.
                handles.append(backbone.register_forward_pre_hook(align_input))
            return original(self, *args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()

    predict._capi_weight_precision = True
    inferencer_class.predict = predict
