"""Density-based cleaning for PatchCore training embeddings."""

from __future__ import annotations

import time

import torch
from lightning.pytorch.callbacks import Callback


class FeatureDensityCleaningCallback(Callback):
    """Remove low-density embeddings before PatchCore builds its coreset."""

    def __init__(
        self,
        *,
        k: int = 30,
        keep_ratio: float = 0.99,
        seed: int = 42,
        reference_size: int = 20_000,
        query_chunk: int = 1_024,
    ) -> None:
        super().__init__()
        if k < 1:
            raise ValueError("k must be positive")
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        if reference_size <= k:
            raise ValueError("reference_size must be greater than k")
        if query_chunk < 1:
            raise ValueError("query_chunk must be positive")

        self.k = k
        self.keep_ratio = keep_ratio
        self.seed = seed
        self.reference_size = reference_size
        self.query_chunk = query_chunk
        self.stats: dict[str, int | float | bool | str | None] = {}
        self._has_run = False

    @torch.no_grad()
    def on_validation_start(self, trainer: object, pl_module: object) -> None:
        """Clean before anomalib fits its memory bank for validation."""
        if getattr(trainer, "sanity_checking", False):
            return
        self._clean_once(pl_module)

    @torch.no_grad()
    def on_train_epoch_end(self, trainer: object, pl_module: object) -> None:
        """Fallback for training runs without a validation loop."""
        del trainer
        self._clean_once(pl_module)

    def _clean_once(self, pl_module: object) -> None:
        if self._has_run:
            return
        started = time.perf_counter()
        embedding_store = pl_module.model.embedding_store
        total = sum(int(embedding.shape[0]) for embedding in embedding_store)
        used_reference_size = min(total, self.reference_size)

        for embedding in embedding_store:
            if embedding.numel() and not bool(torch.isfinite(embedding).all().item()):
                raise RuntimeError("non-finite PatchCore embeddings cannot be density-cleaned")

        if total <= self.k:
            self.stats = self._stats(
                total=total,
                kept=total,
                threshold=None,
                reference_size=used_reference_size,
                started=started,
                applied=False,
                reason="insufficient_features",
            )
            self._has_run = True
            return

        if self.keep_ratio == 1:
            self.stats = self._stats(
                total=total,
                kept=total,
                threshold=None,
                reference_size=used_reference_size,
                started=started,
                applied=False,
                reason="keep_all",
            )
            self._has_run = True
            return

        kth_distances, device = self._kth_cosine_distances(
            embedding_store,
            total=total,
            reference_size=used_reference_size,
        )
        threshold_tensor = torch.quantile(kth_distances, self.keep_ratio)
        keep_mask = kth_distances <= threshold_tensor
        kept = int(keep_mask.sum().item())
        threshold = float(threshold_tensor.item())
        del kth_distances, threshold_tensor

        offset = 0
        for index in range(len(embedding_store)):
            embedding = embedding_store[index]
            count = int(embedding.shape[0])
            local_mask = keep_mask[offset : offset + count].to(embedding.device)
            embedding_store[index] = embedding[local_mask]
            offset += count
        embedding_store[:] = [embedding for embedding in embedding_store if embedding.shape[0]]

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.stats = self._stats(
            total=total,
            kept=kept,
            threshold=threshold,
            reference_size=used_reference_size,
            started=started,
            applied=True,
            reason="completed",
        )
        self._has_run = True

    def _kth_cosine_distances(
        self,
        embedding_store: list[torch.Tensor],
        *,
        total: int,
        reference_size: int,
    ) -> tuple[torch.Tensor, torch.device]:
        nonempty = next(embedding for embedding in embedding_store if embedding.shape[0])
        device = nonempty.device
        feature_dim = int(nonempty.shape[1])

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        if reference_size == total:
            reference_indices = torch.arange(total, dtype=torch.long)
        else:
            reference_indices = torch.randperm(total, generator=generator)[:reference_size]

        reference = torch.empty(
            (reference_size, feature_dim),
            dtype=torch.float32,
            device=device,
        )
        offset = 0
        for embedding in embedding_store:
            count = int(embedding.shape[0])
            selected_positions = torch.nonzero(
                (reference_indices >= offset) & (reference_indices < offset + count),
                as_tuple=False,
            ).flatten()
            if selected_positions.numel():
                local_indices = (reference_indices[selected_positions] - offset).to(embedding.device)
                selected = embedding.detach().index_select(0, local_indices).to(
                    device=device,
                    dtype=torch.float32,
                )
                reference[selected_positions.to(device)] = selected
            offset += count
        reference.div_(torch.linalg.vector_norm(reference, dim=1, keepdim=True).clamp_min_(1e-12))

        reference_positions = torch.full((total,), -1, dtype=torch.long)
        reference_positions[reference_indices] = torch.arange(reference_size, dtype=torch.long)
        distances = torch.empty(total, dtype=torch.float32)
        offset = 0
        for embedding in embedding_store:
            count = int(embedding.shape[0])
            for start in range(0, count, self.query_chunk):
                end = min(start + self.query_chunk, count)
                query = embedding[start:end].detach().to(device=device, dtype=torch.float32)
                query.div_(torch.linalg.vector_norm(query, dim=1, keepdim=True).clamp_min_(1e-12))
                similarities = query @ reference.T

                global_start = offset + start
                global_end = offset + end
                positions = reference_positions[global_start:global_end]
                rows = torch.nonzero(positions >= 0, as_tuple=False).flatten()
                if rows.numel():
                    reference_columns = positions[rows].to(device)
                    rows = rows.to(device)
                    similarities[rows, reference_columns] = -torch.inf

                kth_similarity = similarities.topk(self.k, dim=1).values[:, -1]
                distances[global_start:global_end] = (
                    1 - kth_similarity
                ).clamp_(0, 2).cpu()
            offset += count
        return distances, device

    def _stats(
        self,
        *,
        total: int,
        kept: int,
        threshold: float | None,
        reference_size: int,
        started: float,
        applied: bool,
        reason: str,
    ) -> dict[str, int | float | bool | str | None]:
        removed = total - kept
        return {
            "total": total,
            "kept": kept,
            "removed": removed,
            "removed_ratio": removed / total if total else 0.0,
            "threshold": threshold,
            "k": self.k,
            "keep_ratio": self.keep_ratio,
            "seed": self.seed,
            "reference_size": reference_size,
            "elapsed_seconds": time.perf_counter() - started,
            "applied": applied,
            "reason": reason,
        }
