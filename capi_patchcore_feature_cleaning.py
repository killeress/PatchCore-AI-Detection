"""Density-based cleaning for PatchCore training embeddings."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback


class FeatureDensityCleaningCallback(Callback):
    """Remove low-density embeddings before PatchCore builds its coreset."""

    def __init__(
        self,
        *,
        k: int = 30,
        keep_ratio: float = 0.99,
        center_size: int | None = None,
        seed: int = 42,
        reference_size: int = 20_000,
        query_chunk: int = 1_024,
        trace_sources: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        if k < 1:
            raise ValueError("k must be positive")
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        if center_size is not None and center_size < 1:
            raise ValueError("center_size must be positive")
        if reference_size <= k:
            raise ValueError("reference_size must be greater than k")
        if query_chunk < 1:
            raise ValueError("query_chunk must be positive")

        self.k = k
        self.keep_ratio = keep_ratio
        self.center_size = center_size
        self.seed = seed
        self.reference_size = reference_size
        self.query_chunk = query_chunk
        self.trace_sources = trace_sources or {}
        self.stats: dict[str, Any] = {}
        self._has_run = False
        self._batch_layouts: list[dict[str, Any]] = []
        self._current_grid_shape: tuple[int, int] | None = None
        self._pool_hook_handle: Any = None

    def on_train_start(self, trainer: object, pl_module: object) -> None:
        """Capture the feature-grid shape before PatchCore flattens embeddings."""
        del trainer
        if not self._needs_layouts() or self._pool_hook_handle is not None:
            return
        pooler = getattr(getattr(pl_module, "model", None), "feature_pooler", None)
        register_hook = getattr(pooler, "register_forward_hook", None)
        if not callable(register_hook):
            raise RuntimeError("feature cleaning requires a hookable feature_pooler")

        def _capture_grid(_module: object, _inputs: object, output: object) -> None:
            if self._current_grid_shape is not None:
                return
            shape = getattr(output, "shape", None)
            if shape is None or len(shape) < 4:
                return
            self._current_grid_shape = (int(shape[-2]), int(shape[-1]))

        self._pool_hook_handle = register_hook(_capture_grid)

    def on_train_batch_start(
        self,
        trainer: object,
        pl_module: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        if self._needs_layouts():
            self._current_grid_shape = None

    def on_train_batch_end(
        self,
        trainer: object,
        pl_module: object,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Bind each flattened embedding chunk back to its source tile paths."""
        del trainer, outputs, batch_idx
        if not self._needs_layouts():
            return
        store = getattr(getattr(pl_module, "model", None), "embedding_store", None)
        if not isinstance(store, list) or len(store) != len(self._batch_layouts) + 1:
            raise RuntimeError("feature cleaning lost embedding batch alignment")
        embedding_count = int(store[-1].shape[0])
        raw_paths = getattr(batch, "image_path", None)
        if raw_paths is None:
            raise RuntimeError("feature cleaning requires batch.image_path")
        if isinstance(raw_paths, (str, Path)):
            paths = [str(raw_paths)]
        else:
            paths = [str(path) for path in raw_paths]
        if not paths:
            raise RuntimeError("feature cleaning received an empty image_path batch")

        grid_shape = self._current_grid_shape
        if grid_shape is None:
            per_image = embedding_count // len(paths)
            side = math.isqrt(per_image)
            if side * side != per_image:
                raise RuntimeError("feature cleaning could not infer feature-grid shape")
            grid_shape = (side, side)
        grid_h, grid_w = grid_shape
        if embedding_count != len(paths) * grid_h * grid_w:
            raise RuntimeError(
                "feature cleaning embedding count does not match batch/grid layout"
            )
        input_tensor = getattr(batch, "image", None)
        input_shape = getattr(input_tensor, "shape", None)
        input_size = (
            [int(input_shape[-2]), int(input_shape[-1])]
            if input_shape is not None and len(input_shape) >= 4
            else None
        )
        self._batch_layouts.append({
            "image_paths": [str(Path(path).resolve()) for path in paths],
            "input_size": input_size,
            "grid_size": [grid_h, grid_w],
            "embedding_count": embedding_count,
        })

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
            self._remove_pool_hook()
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
            self._remove_pool_hook()
            self._has_run = True
            return

        cleaning_candidates = self._build_cleaning_candidate_mask(total)
        candidate_count = int(cleaning_candidates.sum().item())
        if candidate_count == 0:
            self.stats = self._stats(
                total=total,
                kept=total,
                threshold=None,
                reference_size=used_reference_size,
                started=started,
                applied=False,
                reason="no_cleaning_candidates",
                cleaning_candidates=0,
            )
            self._remove_pool_hook()
            self._has_run = True
            return

        kth_distances, device = self._kth_cosine_distances(
            embedding_store,
            total=total,
            reference_size=used_reference_size,
        )
        threshold_tensor = torch.quantile(
            kth_distances[cleaning_candidates], self.keep_ratio
        )
        keep_mask = torch.ones(total, dtype=torch.bool)
        keep_mask[cleaning_candidates] = (
            kth_distances[cleaning_candidates] <= threshold_tensor
        )
        kept = int(keep_mask.sum().item())
        threshold = float(threshold_tensor.item())
        del kth_distances, threshold_tensor

        removed_patch_trace = self._build_removed_patch_trace(keep_mask, embedding_store)

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
            cleaning_candidates=candidate_count,
        )
        if removed_patch_trace:
            self.stats["removed_patch_trace"] = removed_patch_trace
        self._remove_pool_hook()
        self._has_run = True

    def _needs_layouts(self) -> bool:
        return self.center_size is not None or bool(self.trace_sources)

    def _build_cleaning_candidate_mask(self, total: int) -> torch.Tensor:
        if self.center_size is None:
            return torch.ones(total, dtype=torch.bool)
        if not self._batch_layouts:
            raise RuntimeError("center feature cleaning requires batch/grid layouts")

        masks: list[torch.Tensor] = []
        consumed = 0
        for layout in self._batch_layouts:
            paths = layout["image_paths"]
            input_size = layout.get("input_size")
            if not input_size or len(input_size) != 2:
                raise RuntimeError("center feature cleaning requires input image size")
            input_h, input_w = (int(input_size[0]), int(input_size[1]))
            grid_h, grid_w = (int(v) for v in layout["grid_size"])
            center_h = min(int(self.center_size), input_h)
            center_w = min(int(self.center_size), input_w)
            top = (input_h - center_h) / 2.0
            left = (input_w - center_w) / 2.0
            # 以 feature cell 中心映射回輸入座標，避免寫死特定 backbone 的 grid 大小。
            row_centers = (torch.arange(grid_h, dtype=torch.float32) + 0.5) * (
                input_h / grid_h
            )
            col_centers = (torch.arange(grid_w, dtype=torch.float32) + 0.5) * (
                input_w / grid_w
            )
            eligible_rows = (row_centers >= top) & (row_centers < top + center_h)
            eligible_cols = (col_centers >= left) & (col_centers < left + center_w)
            grid_mask = eligible_rows[:, None] & eligible_cols[None, :]
            local_mask = grid_mask.reshape(-1).repeat(len(paths))
            expected = int(layout["embedding_count"])
            if int(local_mask.numel()) != expected:
                raise RuntimeError(
                    "center feature cleaning mask does not match embedding layout"
                )
            masks.append(local_mask)
            consumed += expected
        if consumed != total:
            raise RuntimeError("center feature cleaning did not cover all embeddings")
        return torch.cat(masks)

    def _build_removed_patch_trace(
        self,
        keep_mask: torch.Tensor,
        embedding_store: list[torch.Tensor],
    ) -> list[dict[str, Any]]:
        if not self.trace_sources:
            return []
        if len(self._batch_layouts) != len(embedding_store):
            raise RuntimeError("feature cleaning trace is missing one or more batch layouts")
        records: list[dict[str, Any]] = []
        offset = 0
        for layout, embedding in zip(self._batch_layouts, embedding_store):
            count = int(embedding.shape[0])
            paths = layout["image_paths"]
            grid_h, grid_w = layout["grid_size"]
            per_image = grid_h * grid_w
            local_keep = keep_mask[offset : offset + count].reshape(
                len(paths), grid_h, grid_w
            )
            for image_idx, staged_path in enumerate(paths):
                removed = torch.nonzero(
                    ~local_keep[image_idx].reshape(-1), as_tuple=False
                ).flatten()
                if not removed.numel():
                    continue
                source = self.trace_sources.get(staged_path)
                if source is None:
                    source = self.trace_sources.get(Path(staged_path).name)
                if source is None:
                    raise RuntimeError(
                        f"feature cleaning trace source not found: {staged_path}"
                    )
                records.append({
                    "tile_pool_id": source.get("tile_pool_id"),
                    "source_path": str(source.get("source_path") or staged_path),
                    "input_size": layout.get("input_size"),
                    "grid_size": [grid_h, grid_w],
                    "removed_indices": [int(value) for value in removed.tolist()],
                    "removed_count": int(removed.numel()),
                })
            offset += count
        if offset != int(keep_mask.numel()):
            raise RuntimeError("feature cleaning trace did not consume the full keep mask")
        return records

    def _remove_pool_hook(self) -> None:
        if self._pool_hook_handle is not None:
            self._pool_hook_handle.remove()
            self._pool_hook_handle = None

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
        cleaning_candidates: int | None = None,
    ) -> dict[str, int | float | bool | str | None]:
        removed = total - kept
        candidate_count = total if cleaning_candidates is None else cleaning_candidates
        return {
            "total": total,
            "kept": kept,
            "removed": removed,
            "removed_ratio": removed / total if total else 0.0,
            "cleaning_candidates": candidate_count,
            "cleaning_candidate_kept": max(0, candidate_count - removed),
            "cleaning_candidate_removed_ratio": (
                removed / candidate_count if candidate_count else 0.0
            ),
            "protected": max(0, total - candidate_count),
            "center_size": self.center_size,
            "threshold": threshold,
            "k": self.k,
            "keep_ratio": self.keep_ratio,
            "seed": self.seed,
            "reference_size": reference_size,
            "elapsed_seconds": time.perf_counter() - started,
            "applied": applied,
            "reason": reason,
        }
