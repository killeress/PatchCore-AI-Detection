"""SoftPatch+ inspired pooled LOF/Gaussian cleaning with optional Tile consensus.

This version uses one memory bank, rank fusion and bounded weights. It keeps
the application's PatchCore image aggregation, rather than claiming a literal
reproduction of the paper's position-wise, dual-bank evaluation protocol.
"""

from __future__ import annotations

import math
import time

import torch
from anomalib.data import InferenceBatch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

from capi_patchcore_feature_cleaning import FeatureDensityCleaningCallback
from capi_softpatch_config import normalize_softpatch_config


def percentile_rank(values: torch.Tensor) -> torch.Tensor:
    """Average ranks for ties; a constant score carries no outlier evidence."""
    if values.numel() < 2 or bool((values == values[0]).all()):
        return torch.zeros_like(values)
    _, inverse, counts = torch.unique(values, sorted=True, return_inverse=True, return_counts=True)
    ranks = (counts.cumsum(0) - (counts + 1) / 2) / (values.numel() - 1)
    return ranks[inverse].to(values.dtype)


@torch.no_grad()
def score_features(embedding_store, reference_indices, *, k, seed, options, query_chunk):
    """Compute sampled-reference LOF and regularized Mahalanobis rank fusion.

    Only reduced features are stacked. Query/reference matrices are chunked;
    identity exclusion uses original row indices, including sampled references.
    """
    first = next(x for x in embedding_store if x.shape[0])
    device, dim = first.device, first.shape[1]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    reduced_dim = min(dim, options["projection_dim"])
    projection = None
    if dim > reduced_dim:
        projection = (torch.randn(dim, reduced_dim, generator=generator) / math.sqrt(reduced_dim)).to(device)
    chunks = []
    for embedding in embedding_store:
        for start in range(0, embedding.shape[0], query_chunk):
            value = embedding[start:start + query_chunk].detach().float()
            chunks.append((value @ projection if projection is not None else value).cpu())
    features = torch.cat(chunks)
    available = reference_indices.cpu().long()
    count = min(available.numel(), options["reference_size"])
    if count <= k:
        raise ValueError("SoftPatch+ requires more reference features than k")
    if count < available.numel():
        available = available[torch.randperm(available.numel(), generator=generator)[:count]]
    reference = features[available].to(device)
    reference_positions = torch.full((features.shape[0],), -1, dtype=torch.long)
    reference_positions[available] = torch.arange(count)

    # Reference neighborhoods and local reachability densities.
    neighbor_distances, neighbor_indices = [], []
    for start in range(0, count, query_chunk):
        end = min(start + query_chunk, count)
        distances = torch.cdist(reference[start:end], reference)
        distances[torch.arange(end - start, device=device), torch.arange(start, end, device=device)] = torch.inf
        distance, index = distances.topk(k, largest=False, dim=1)
        neighbor_distances.append(distance)
        neighbor_indices.append(index)
    ref_dist = torch.cat(neighbor_distances)
    ref_neighbors = torch.cat(neighbor_indices)
    kth = ref_dist[:, -1]
    ref_lrd = torch.maximum(ref_dist, kth[ref_neighbors]).mean(1).clamp_min(1e-6).reciprocal()

    mean = reference.mean(0)
    chol = None
    if options["discriminator"] == "lof_gaussian":
        centered = reference - mean
        covariance = centered.T @ centered / max(1, count - 1)
        scale = covariance.diagonal().mean().clamp_min(1e-6)
        covariance = 0.9 * covariance + (0.1 * scale + 1e-6) * torch.eye(reduced_dim, device=device)
        chol = torch.linalg.cholesky(covariance)

    lof, gaussian = [], []
    for start in range(0, features.shape[0], query_chunk):
        end = min(start + query_chunk, features.shape[0])
        query = features[start:end].to(device)
        distances = torch.cdist(query, reference)
        positions = reference_positions[start:end].to(device)
        rows = torch.nonzero(positions >= 0, as_tuple=False).flatten()
        distances[rows, positions[rows]] = torch.inf
        distance, index = distances.topk(k, largest=False, dim=1)
        reachability = torch.maximum(distance, kth[index]).mean(1).clamp_min(1e-6)
        lof.append((ref_lrd[index].mean(1) * reachability).cpu())
        if chol is not None:
            whitened = torch.linalg.solve_triangular(chol, (query - mean).T, upper=False)
            gaussian.append(torch.linalg.vector_norm(whitened, dim=0).cpu())
    scores = percentile_rank(torch.cat(lof))
    if gaussian:
        scores = (scores + percentile_rank(torch.cat(gaussian))) / 2
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError("non-finite SoftPatch+ scores")
    return scores, count, device


class SoftPatchPlusModel(PatchcoreModel):
    """Pickleable PatchCore model; weights are persistent, device-aware buffers."""

    def weighted_patch_scores(self, scores, locations):
        weights = self.softpatch_weights
        if weights.numel() != self.memory_bank.shape[0]:
            raise RuntimeError("SoftPatch+ memory-bank weights are missing or misaligned")
        return scores * weights[locations].to(scores.dtype)

    def forward(self, input_tensor):
        if self.training:
            return super().forward(input_tensor)
        if not self.memory_bank.shape[0]:
            raise ValueError("Memory bank is empty. Cannot provide anomaly scores")
        input_tensor = input_tensor.type(self.memory_bank.dtype)
        output_size = input_tensor.shape[-2:]
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
        features = {layer: self.feature_pooler(value) for layer, value in features.items()}
        embedding = self.generate_embedding(features)
        if self.tiler:
            embedding = self.tiler.untile(embedding)
        batch_size, _, height, width = embedding.shape
        embedding = self.reshape_embedding(embedding)
        scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
        scores = self.weighted_patch_scores(scores, locations).reshape(batch_size, -1)
        locations = locations.reshape(batch_size, -1)
        # Existing aggregation (including MARK overrides) receives weighted scores;
        # internal support-neighbor searches remain ordinary distances.
        pred_score = self.compute_anomaly_score(scores, locations, embedding)
        anomaly_map = self.anomaly_map_generator(scores.reshape(batch_size, 1, height, width), output_size)
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)


def enable_softpatch_model(model):
    if not isinstance(model, PatchcoreModel):
        raise TypeError("SoftPatch+ requires an anomalib PatchcoreModel")
    if not isinstance(model, SoftPatchPlusModel):
        # Preserve the initialized backbone and model buffers without loading a
        # second network. The importable class survives Torch export/reload.
        model.__class__ = SoftPatchPlusModel
        model.register_buffer("softpatch_weights", torch.empty(0, device=model.memory_bank.device))


class SoftPatchPlusCleaningCallback(FeatureDensityCleaningCallback):
    requires_coreset_indices = True

    def __init__(self, *, options=None, **kwargs):
        self.options = normalize_softpatch_config(options)
        kwargs["strategy"] = "context_overlap_adaptive" if self.options["context_overlap"] else "quantile"
        kwargs["reference_size"] = self.options["reference_size"]
        kwargs["center_size"] = None
        super().__init__(**kwargs)
        self._soft_weights = None

    def on_train_start(self, trainer, pl_module):
        self._install_coreset_trace(pl_module)
        super().on_train_start(trainer, pl_module)

    def _on_coreset_selected(self, model, selected):
        if self._soft_weights is None:
            raise RuntimeError("SoftPatch+ cleaning did not produce weights")
        indices = self._cleaned_original_indices[torch.as_tensor(selected, dtype=torch.long)]
        model.softpatch_weights = self._soft_weights[indices].to(device=model.memory_bank.device)
        self.stats["soft_weight_min"] = float(model.softpatch_weights.min())
        self.stats["soft_weight_max"] = float(model.softpatch_weights.max())

    @torch.no_grad()
    def _clean_once(self, pl_module):
        if self._has_run:
            return
        started = time.perf_counter()
        store = pl_module.model.embedding_store
        total = sum(x.shape[0] for x in store)
        if not total:
            raise ValueError("SoftPatch+ cannot clean an empty feature set")
        if any(not bool(torch.isfinite(x).all()) for x in store):
            raise RuntimeError("non-finite PatchCore embeddings cannot be density-cleaned")
        context = self._build_context_cleaning_plan(total) if self.options["context_overlap"] else None
        candidates = context["candidate_mask"] if context is not None else torch.ones(total, dtype=torch.bool)
        references = context["reference_indices"] if context is not None else torch.arange(total)
        used = min(references.numel(), self.reference_size)
        scores = torch.zeros(total)
        applied = used > self.k
        device = store[0].device
        if applied:
            scores, used, device = score_features(store, references, k=self.k, seed=self.seed,
                                                 options=self.options, query_chunk=self.query_chunk)
        # Quantile ranking is already normalized: a MAD threshold on rank scores
        # would usually suppress all cleaning. Context still protects boundaries
        # and requires unanimous overlap votes.
        threshold = None
        raw_remove = torch.zeros(total, dtype=torch.bool)
        if applied and bool(candidates.any()) and self.keep_ratio < 1:
            threshold = float(torch.quantile(scores[candidates], self.keep_ratio))
            raw_remove = candidates & (scores > threshold)
        keep = self._apply_overlap_consensus(raw_remove, context) if context is not None else ~raw_remove
        strength = self.options["weight_strength"] if self.options["soft_weight"] else 0.0
        self._soft_weights = 1 + strength * scores
        # Protected or disagreeing views retain neutral confidence. Otherwise
        # soft weights could bypass the very protection selected by the user.
        protected = ~candidates
        if context is not None:
            for indices in context["overlap_groups"]:
                if bool(raw_remove[indices].any()) and not bool(raw_remove[indices].all()):
                    protected[indices] = True
        self._soft_weights[protected] = 1
        self._finalize_cleaning_result(
            embedding_store=store, total=total, keep_mask=keep, cleaning_candidates=candidates,
            context_plan=context, kth_distances=scores, raw_remove_mask=raw_remove,
            threshold=threshold, reference_size=used, started=started, applied=applied,
            reason=("insufficient_context_references" if not applied else
                    "completed" if bool(candidates.any()) else "no_cleaning_candidates"), device=device,
            adaptive_stats={"threshold_method": "rank_quantile_with_optional_overlap_consensus",
                            "raw_outlier_count": int(raw_remove.sum()),
                            "consensus_removed_count": int((~keep).sum())},
        )
        self.stats.update({"strategy": "softpatch_plus_v1", "score_metric": "outlier_rank",
                           "softpatch_plus_config": dict(self.options), "grouping": "pooled_training_unit",
                           "weight_formula": "1 + strength * outlier_rank", "memory_banks": 1})
