"""Snapshot training provenance from the loaded model for persisted inference logs."""

import json
import math
from pathlib import Path


MODES = {"off", "knn_cosine_q99_v1", "context_overlap_adaptive_v1", "softpatch_plus_v1"}


def snapshot_model_training(inferencer, model_path=None):
    outer = getattr(inferencer, "model", None)
    model = getattr(outer, "model", outer)
    is_softpatch = any(
        cls.__name__ == "SoftPatchPlusModel" and cls.__module__ == "capi_patchcore_softpatch"
        for cls in type(model).__mro__
    )
    metadata = getattr(model, "training_provenance", None)
    source = "model" if isinstance(metadata, dict) else "unknown"
    if not isinstance(metadata, dict):
        metadata = {}
        if model_path:
            path = Path(model_path)
            try:
                manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
                params = manifest.get("patchcore_params") or {}
                zone = path.stem.rsplit("-", 1)[-1]
                zone_config = (params.get("feature_cleaning_by_zone") or {}).get(zone)
                mode = params.get("feature_cleaning_mode", "unknown")
                if isinstance(zone_config, dict):
                    mode = zone_config.get("mode", "unknown")
                elif zone == "edge" and params.get("feature_cleaning_scope") == "inner_only":
                    mode = "off"
                metadata = {"mode": mode, "softpatch_plus_config": params.get("softpatch_plus_config") or {}}
                source = "manifest"
            except (OSError, ValueError, TypeError, AttributeError):
                pass
    mode = metadata.get("mode", "unknown")
    if is_softpatch:
        mode = "softpatch_plus_v1"
    elif not isinstance(mode, str) or mode == "softpatch_plus_v1" or mode not in MODES:
        # A recipe on disk must not label a plain loaded model as SoftPatch+.
        mode = "unknown"
    result = {"model_path": str(model_path or ""), "mode": mode, "source": source,
              "runtime_softpatch": is_softpatch}
    if is_softpatch:
        options = metadata.get("softpatch_plus_config") or {}
        if isinstance(options, dict):
            if options.get("discriminator") in ("lof", "lof_gaussian"):
                result["discriminator"] = options["discriminator"]
            if isinstance(options.get("soft_weight"), bool):
                result["soft_weight"] = options["soft_weight"]
            strength = options.get("weight_strength")
            if isinstance(strength, (int, float)) and not isinstance(strength, bool) and math.isfinite(strength):
                result["weight_strength"] = strength
        weights = getattr(model, "softpatch_weights", None)
        if weights is not None and weights.numel():
            minimum, maximum = float(weights.min()), float(weights.max())
            if math.isfinite(minimum) and math.isfinite(maximum):
                result["weight_min"], result["weight_max"] = minimum, maximum
    return result


def log_model_training(inferencer, lighting, zone, seen=None):
    """Print once per used model/unit in a caller's inference pass, including cache hits."""
    if inferencer is None:
        return
    key = (id(inferencer), lighting, zone)
    if seen is not None:
        if key in seen:
            return
        seen.add(key)
    snapshot = getattr(inferencer, "_capi_training_provenance", None)
    if not isinstance(snapshot, dict):
        snapshot = snapshot_model_training(inferencer)
    print("[MODEL_TRAINING] " + json.dumps(
        {**snapshot, "lighting": lighting, "zone": zone}, ensure_ascii=False, allow_nan=False,
    ))
