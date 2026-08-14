"""Density-based cleaning for PatchCore training embeddings."""

from __future__ import annotations

import math
import time
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback


TRACE_REASON_LEGEND = {
    0: "kept_normal",
    1: "removed_distance_outlier",
    2: "kept_overlap_disagreement",
    3: "protected_tile_boundary",
    5: "missing_coordinate_metadata",
    6: "protected_outside_cleaning_scope",
}


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
        strategy: str = "quantile",
        adaptive_mad_z: float = 6.0,
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
        if strategy not in {"quantile", "context_overlap_adaptive"}:
            raise ValueError("unsupported feature cleaning strategy")
        if adaptive_mad_z <= 0:
            raise ValueError("adaptive_mad_z must be positive")

        self.k = k
        self.keep_ratio = keep_ratio
        self.center_size = center_size
        self.seed = seed
        self.reference_size = reference_size
        self.query_chunk = query_chunk
        self.trace_sources = trace_sources or {}
        self.strategy = strategy
        self.adaptive_mad_z = adaptive_mad_z
        self.stats: dict[str, Any] = {}
        self._has_run = False
        self._batch_layouts: list[dict[str, Any]] = []
        self._current_grid_shape: tuple[int, int] | None = None
        self._pool_hook_handle: Any = None
        self._cleaned_original_indices: torch.Tensor | None = None
        self._trace_record_spans: list[tuple[dict[str, Any], int, int]] = []
        self._coreset_trace_installed = False

    def on_train_start(self, trainer: object, pl_module: object) -> None:
        """Capture the feature-grid shape before PatchCore flattens embeddings."""
        del trainer
        if not self._needs_layouts():
            return
        self._install_coreset_trace(pl_module)
        if self._pool_hook_handle is not None:
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

    def _install_coreset_trace(self, pl_module: object) -> None:
        """Wrap PatchCore subsampling so selected rows retain source-cell lineage."""
        if self._coreset_trace_installed or not self.trace_sources:
            return
        inner_model = getattr(pl_module, "model", None)
        original = getattr(inner_model, "subsample_embedding", None)
        if inner_model is None or not callable(original):
            return

        callback = self

        def _traced_subsample(
            model: object,
            sampling_ratio: float,
            embeddings: torch.Tensor | None = None,
        ) -> None:
            if embeddings is not None:
                raise RuntimeError("coreset trace does not support external embeddings")
            embedding_store = getattr(model, "embedding_store", None)
            if not isinstance(embedding_store, list) or not embedding_store:
                raise ValueError("Embedding store is empty. Cannot perform coreset selection.")
            memory_bank = torch.vstack(embedding_store)
            embedding_store.clear()
            selected = callback._select_coreset_indices(memory_bank, sampling_ratio)
            model.memory_bank = memory_bank[selected]
            callback.record_coreset_indices(selected)
            model.subsample_embedding = original

        inner_model.subsample_embedding = MethodType(_traced_subsample, inner_model)
        self._coreset_trace_installed = True

    @staticmethod
    def _select_coreset_indices(
        memory_bank: torch.Tensor,
        sampling_ratio: float,
    ) -> list[int]:
        from anomalib.models.components import KCenterGreedy

        sampler = KCenterGreedy(
            embedding=memory_bank,
            sampling_ratio=sampling_ratio,
        )
        return sampler.select_coreset_idxs()

    def record_coreset_indices(self, selected_indices: list[int]) -> None:
        """Map cleaned coreset indices back to each source Tile's original grid."""
        if self._cleaned_original_indices is None:
            raise RuntimeError("coreset trace is missing the cleaned-to-original index map")
        selected = torch.as_tensor(selected_indices, dtype=torch.long, device="cpu")
        if selected.numel() and (
            int(selected.min().item()) < 0
            or int(selected.max().item()) >= int(self._cleaned_original_indices.numel())
        ):
            raise RuntimeError("coreset trace index is outside the cleaned embedding range")
        original = self._cleaned_original_indices.index_select(0, selected)
        selected_mask = torch.zeros(
            int(self.stats.get("total") or self._cleaned_original_indices.numel()),
            dtype=torch.bool,
        )
        if original.numel():
            selected_mask[original] = True
        for record, start, count in self._trace_record_spans:
            local = torch.nonzero(
                selected_mask[start : start + count], as_tuple=False
            ).flatten()
            record["coreset_indices"] = [int(value) for value in local.tolist()]
            record["coreset_count"] = int(local.numel())
        self.stats["coreset_selected"] = int(selected.numel())
        self.stats["coreset_selected_ratio"] = (
            float(selected.numel()) / float(self._cleaned_original_indices.numel())
            if self._cleaned_original_indices.numel()
            else 0.0
        )

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

        if total <= self.k and self.strategy != "context_overlap_adaptive":
            self._cleaned_original_indices = torch.arange(total, dtype=torch.long)
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

        if self.keep_ratio == 1 and self.strategy != "context_overlap_adaptive":
            self._cleaned_original_indices = torch.arange(total, dtype=torch.long)
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

        context_plan: dict[str, Any] | None = None
        reference_indices: torch.Tensor | None = None
        if self.strategy == "context_overlap_adaptive":
            context_plan = self._build_context_cleaning_plan(total)
            cleaning_candidates = context_plan["candidate_mask"]
            reference_indices = context_plan["reference_indices"]
            used_reference_size = min(int(reference_indices.numel()), self.reference_size)
        else:
            cleaning_candidates = self._build_cleaning_candidate_mask(total)
        candidate_count = int(cleaning_candidates.sum().item())

        if self.keep_ratio == 1:
            self._finalize_cleaning_result(
                embedding_store=embedding_store,
                total=total,
                keep_mask=torch.ones(total, dtype=torch.bool),
                cleaning_candidates=cleaning_candidates,
                context_plan=context_plan,
                kth_distances=None,
                raw_remove_mask=torch.zeros(total, dtype=torch.bool),
                threshold=None,
                reference_size=used_reference_size,
                started=started,
                applied=False,
                reason="keep_all",
            )
            return

        if candidate_count == 0:
            self._finalize_cleaning_result(
                embedding_store=embedding_store,
                total=total,
                keep_mask=torch.ones(total, dtype=torch.bool),
                cleaning_candidates=cleaning_candidates,
                context_plan=context_plan,
                kth_distances=None,
                raw_remove_mask=torch.zeros(total, dtype=torch.bool),
                threshold=None,
                reference_size=used_reference_size,
                started=started,
                applied=False,
                reason="no_cleaning_candidates",
            )
            return

        if used_reference_size <= self.k:
            self._finalize_cleaning_result(
                embedding_store=embedding_store,
                total=total,
                keep_mask=torch.ones(total, dtype=torch.bool),
                cleaning_candidates=cleaning_candidates,
                context_plan=context_plan,
                kth_distances=None,
                raw_remove_mask=torch.zeros(total, dtype=torch.bool),
                threshold=None,
                reference_size=used_reference_size,
                started=started,
                applied=False,
                reason="insufficient_context_references",
            )
            return

        kth_distances, device = self._kth_cosine_distances(
            embedding_store,
            total=total,
            reference_size=used_reference_size,
            available_reference_indices=reference_indices,
        )
        candidate_distances = kth_distances[cleaning_candidates]
        adaptive_stats: dict[str, Any] = {}
        if context_plan is not None:
            threshold_tensor, adaptive_stats = self._adaptive_threshold(
                candidate_distances
            )
            raw_remove_mask = cleaning_candidates & (kth_distances > threshold_tensor)
            keep_mask = self._apply_overlap_consensus(raw_remove_mask, context_plan)
            adaptive_stats["raw_outlier_count"] = int(raw_remove_mask.sum().item())
            adaptive_stats["consensus_removed_count"] = int((~keep_mask).sum().item())
        else:
            threshold_tensor = torch.quantile(candidate_distances, self.keep_ratio)
            raw_remove_mask = cleaning_candidates & (kth_distances > threshold_tensor)
            keep_mask = torch.ones(total, dtype=torch.bool)
            keep_mask[cleaning_candidates] = (
                candidate_distances <= threshold_tensor
            )
        threshold = float(threshold_tensor.item())
        del threshold_tensor
        self._finalize_cleaning_result(
            embedding_store=embedding_store,
            total=total,
            keep_mask=keep_mask,
            cleaning_candidates=cleaning_candidates,
            context_plan=context_plan,
            kth_distances=kth_distances,
            raw_remove_mask=raw_remove_mask,
            threshold=threshold,
            reference_size=used_reference_size,
            started=started,
            applied=True,
            reason="completed",
            adaptive_stats=adaptive_stats,
            device=device,
        )

    def _finalize_cleaning_result(
        self,
        *,
        embedding_store: list[torch.Tensor],
        total: int,
        keep_mask: torch.Tensor,
        cleaning_candidates: torch.Tensor,
        context_plan: dict[str, Any] | None,
        kth_distances: torch.Tensor | None,
        raw_remove_mask: torch.Tensor,
        threshold: float | None,
        reference_size: int,
        started: float,
        applied: bool,
        reason: str,
        adaptive_stats: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ) -> None:
        patch_trace = self._build_patch_trace(
            keep_mask=keep_mask,
            embedding_store=embedding_store,
            cleaning_candidates=cleaning_candidates,
            kth_distances=kth_distances,
            raw_remove_mask=raw_remove_mask,
            context_plan=context_plan,
        )
        removed_patch_trace = [
            {
                "tile_pool_id": record.get("tile_pool_id"),
                "source_path": record.get("source_path"),
                "input_size": record.get("input_size"),
                "grid_size": record.get("grid_size"),
                "removed_indices": record.get("removed_indices") or [],
                "removed_count": int(record.get("removed_count") or 0),
            }
            for record in patch_trace
            if int(record.get("removed_count") or 0) > 0
        ]

        self._cleaned_original_indices = torch.nonzero(
            keep_mask, as_tuple=False
        ).flatten().to(device="cpu", dtype=torch.long)
        offset = 0
        for index in range(len(embedding_store)):
            embedding = embedding_store[index]
            count = int(embedding.shape[0])
            local_mask = keep_mask[offset : offset + count].to(embedding.device)
            embedding_store[index] = embedding[local_mask]
            offset += count
        embedding_store[:] = [embedding for embedding in embedding_store if embedding.shape[0]]

        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        kept = int(keep_mask.sum().item())
        candidate_count = int(cleaning_candidates.sum().item())
        self.stats = self._stats(
            total=total,
            kept=kept,
            threshold=threshold,
            reference_size=reference_size,
            started=started,
            applied=applied,
            reason=reason,
            cleaning_candidates=candidate_count,
        )
        self.stats["strategy"] = self.strategy
        self.stats["trace_reason_legend"] = {
            str(code): reason for code, reason in TRACE_REASON_LEGEND.items()
        }
        self.stats["distance_removed"] = int((~keep_mask).sum().item())
        if context_plan is not None:
            self.stats.update(context_plan["stats"])
        if adaptive_stats:
            self.stats.update(adaptive_stats)
        if patch_trace:
            self.stats["patch_trace"] = patch_trace
        if removed_patch_trace:
            self.stats["removed_patch_trace"] = removed_patch_trace
        self._remove_pool_hook()
        self._has_run = True

    def _needs_layouts(self) -> bool:
        return (
            self.strategy == "context_overlap_adaptive"
            or self.center_size is not None
            or bool(self.trace_sources)
        )

    def _trace_source_for_path(self, staged_path: str) -> dict[str, Any] | None:
        source = self.trace_sources.get(staged_path)
        if source is None:
            source = self.trace_sources.get(Path(staged_path).name)
        return source

    def _build_context_cleaning_plan(self, total: int) -> dict[str, Any]:
        """Map flattened features back to physical Panel positions.

        The same physical position may be observed by multiple overlapping Tiles.
        We keep one best-context view in the KNN reference set and require all
        overlapping views to agree before removing that position.
        """
        if not self._batch_layouts:
            raise RuntimeError("context feature cleaning requires batch/grid layouts")

        images: list[dict[str, Any]] = []
        pitch_x_values: list[float] = []
        pitch_y_values: list[float] = []
        consumed = 0
        missing_metadata = 0
        for layout in self._batch_layouts:
            paths = layout["image_paths"]
            grid_h, grid_w = (int(v) for v in layout["grid_size"])
            per_image = grid_h * grid_w
            expected = int(layout["embedding_count"])
            if expected != len(paths) * per_image:
                raise RuntimeError(
                    "context feature cleaning layout does not match embedding count"
                )
            for image_idx, staged_path in enumerate(paths):
                start = consumed + image_idx * per_image
                source = self._trace_source_for_path(staged_path)
                required = (
                    "panel_path",
                    "tile_x",
                    "tile_y",
                    "tile_width",
                    "tile_height",
                )
                if (
                    source is None
                    or not source.get("panel_path")
                    or any(source.get(name) is None for name in required[1:])
                ):
                    missing_metadata += per_image
                    images.append({"start": start, "count": per_image, "source": None})
                    continue
                tile_width = int(source["tile_width"])
                tile_height = int(source["tile_height"])
                if tile_width <= 0 or tile_height <= 0:
                    missing_metadata += per_image
                    images.append({"start": start, "count": per_image, "source": None})
                    continue
                pitch_x = tile_width / grid_w
                pitch_y = tile_height / grid_h
                pitch_x_values.append(pitch_x)
                pitch_y_values.append(pitch_y)
                images.append({
                    "start": start,
                    "count": per_image,
                    "source": source,
                    "grid_h": grid_h,
                    "grid_w": grid_w,
                    "tile_x": int(source["tile_x"]),
                    "tile_y": int(source["tile_y"]),
                    "tile_width": tile_width,
                    "tile_height": tile_height,
                    "pitch_x": pitch_x,
                    "pitch_y": pitch_y,
                })
            consumed += expected
        if consumed != total:
            raise RuntimeError("context feature cleaning did not cover all embeddings")

        candidate_mask = torch.zeros(total, dtype=torch.bool)
        missing_metadata_mask = torch.zeros(total, dtype=torch.bool)
        overlap_view_counts = torch.zeros(total, dtype=torch.int16)
        if not pitch_x_values or not pitch_y_values:
            for image in images:
                if image.get("source") is None:
                    start = int(image["start"])
                    missing_metadata_mask[start : start + int(image["count"])] = True
            return {
                "candidate_mask": candidate_mask,
                "reference_indices": torch.arange(total, dtype=torch.long),
                "overlap_groups": [],
                "missing_metadata_mask": missing_metadata_mask,
                "overlap_view_counts": overlap_view_counts,
                "stats": {
                    "context_overlap_groups": 0,
                    "context_singletons": 0,
                    "context_missing_metadata": missing_metadata,
                    "context_auto_guard_px": None,
                },
            }

        base_pitch_x = self._median(pitch_x_values)
        base_pitch_y = self._median(pitch_y_values)
        grouped: dict[tuple[str, int, int], list[tuple[int, float]]] = {}
        missing_reference_indices: list[int] = []
        rectangles_by_panel: dict[str, list[tuple[int, int, int, int]]] = {}
        for image in images:
            start = int(image["start"])
            count = int(image["count"])
            source = image.get("source")
            if source is None:
                missing_reference_indices.extend(range(start, start + count))
                missing_metadata_mask[start : start + count] = True
                continue
            panel_key = str(Path(str(source["panel_path"])).resolve()).casefold()
            tile_x = int(image["tile_x"])
            tile_y = int(image["tile_y"])
            tile_width = int(image["tile_width"])
            tile_height = int(image["tile_height"])
            pitch_x = float(image["pitch_x"])
            pitch_y = float(image["pitch_y"])
            grid_h = int(image["grid_h"])
            grid_w = int(image["grid_w"])
            rectangle = (tile_x, tile_y, tile_x + tile_width, tile_y + tile_height)
            panel_rectangles = rectangles_by_panel.setdefault(panel_key, [])
            if rectangle not in panel_rectangles:
                panel_rectangles.append(rectangle)
            for row in range(grid_h):
                local_y = (row + 0.5) * pitch_y
                physical_y = tile_y + local_y
                key_y = int(math.floor(physical_y / base_pitch_y + 0.5))
                for col in range(grid_w):
                    local_x = (col + 0.5) * pitch_x
                    physical_x = tile_x + local_x
                    key_x = int(math.floor(physical_x / base_pitch_x + 0.5))
                    index = start + row * grid_w + col
                    border_margin = min(
                        local_x,
                        tile_width - local_x,
                        local_y,
                        tile_height - local_y,
                    )
                    grouped.setdefault((panel_key, key_x, key_y), []).append(
                        (index, border_margin)
                    )

        overlap_depths: list[float] = []
        for rectangles in rectangles_by_panel.values():
            for left_index, left in enumerate(rectangles):
                for right in rectangles[left_index + 1 :]:
                    intersection_w = min(left[2], right[2]) - max(left[0], right[0])
                    intersection_h = min(left[3], right[3]) - max(left[1], right[1])
                    if intersection_w <= 0 or intersection_h <= 0:
                        continue
                    if left[0] != right[0] and intersection_w < min(
                        left[2] - left[0], right[2] - right[0]
                    ):
                        overlap_depths.append(float(intersection_w))
                    if left[1] != right[1] and intersection_h < min(
                        left[3] - left[1], right[3] - right[1]
                    ):
                        overlap_depths.append(float(intersection_h))
        auto_guard = self._median(overlap_depths) if overlap_depths else None

        overlap_groups: list[list[int]] = []
        reference_indices: list[int] = []
        overlap_group_count = 0
        singleton_count = 0
        for members in grouped.values():
            best_index, best_margin = max(members, key=lambda item: item[1])
            indices = [index for index, _margin in members]
            overlap_view_counts[indices] = len(indices)
            reference_indices.append(best_index)
            if len(indices) > 1:
                overlap_group_count += 1
                candidate_mask[indices] = True
                overlap_groups.append(indices)
            else:
                singleton_count += 1
                if auto_guard is not None and best_margin >= auto_guard:
                    candidate_mask[best_index] = True

        # Records without Panel/Tile coordinates stay protected. They may come
        # from an older job or a hard-normal pool and must not fake context.
        reference_indices.extend(missing_reference_indices)
        return {
            "candidate_mask": candidate_mask,
            "reference_indices": torch.tensor(reference_indices, dtype=torch.long),
            "overlap_groups": overlap_groups,
            "missing_metadata_mask": missing_metadata_mask,
            "overlap_view_counts": overlap_view_counts,
            "stats": {
                "context_overlap_groups": overlap_group_count,
                "context_singletons": singleton_count,
                "context_missing_metadata": missing_metadata,
                "context_auto_guard_px": auto_guard,
                "context_reference_candidates": len(reference_indices),
            },
        }

    @staticmethod
    def _median(values: list[float]) -> float:
        ordered = sorted(values)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[midpoint])
        return float((ordered[midpoint - 1] + ordered[midpoint]) / 2.0)

    def _adaptive_threshold(
        self, distances: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float | str]]:
        median = torch.median(distances)
        mad = torch.median(torch.abs(distances - median))
        robust_threshold = median + self.adaptive_mad_z * 1.4826 * mad
        removal_cap_threshold = torch.quantile(distances, self.keep_ratio)
        threshold = torch.maximum(robust_threshold, removal_cap_threshold)
        return threshold, {
            "threshold_method": "max(median+z*MAD, keep_ratio_quantile)",
            "adaptive_median": float(median.item()),
            "adaptive_mad": float(mad.item()),
            "adaptive_mad_z": self.adaptive_mad_z,
            "adaptive_robust_threshold": float(robust_threshold.item()),
            "adaptive_removal_cap_threshold": float(removal_cap_threshold.item()),
        }

    @staticmethod
    def _apply_overlap_consensus(
        raw_remove_mask: torch.Tensor,
        context_plan: dict[str, Any],
    ) -> torch.Tensor:
        keep_mask = ~raw_remove_mask
        for indices in context_plan["overlap_groups"]:
            # Safety-first: a physical position is removed only when every
            # overlapping Tile view independently says it is an outlier.
            if not bool(raw_remove_mask[indices].all().item()):
                keep_mask[indices] = True
        return keep_mask

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

    def _build_patch_trace(
        self,
        *,
        keep_mask: torch.Tensor,
        embedding_store: list[torch.Tensor],
        cleaning_candidates: torch.Tensor,
        kth_distances: torch.Tensor | None,
        raw_remove_mask: torch.Tensor,
        context_plan: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not self.trace_sources:
            return []
        if len(self._batch_layouts) != len(embedding_store):
            raise RuntimeError("feature cleaning trace is missing one or more batch layouts")
        missing_metadata_mask = (
            context_plan["missing_metadata_mask"]
            if context_plan is not None
            else torch.zeros_like(keep_mask)
        )
        overlap_view_counts = (
            context_plan["overlap_view_counts"].clone()
            if context_plan is not None
            else torch.ones(int(keep_mask.numel()), dtype=torch.int16)
        )
        outlier_vote_counts = raw_remove_mask.to(dtype=torch.int16)
        outlier_vote_required = torch.zeros(int(keep_mask.numel()), dtype=torch.int16)
        outlier_vote_required[cleaning_candidates] = 1
        overlap_disagreement_mask = torch.zeros_like(keep_mask)
        if context_plan is not None:
            for indices in context_plan["overlap_groups"]:
                votes = int(raw_remove_mask[indices].sum().item())
                required = len(indices)
                outlier_vote_counts[indices] = votes
                outlier_vote_required[indices] = required
                if 0 < votes < required:
                    overlap_disagreement_mask[indices] = True

        reason_codes = torch.zeros(int(keep_mask.numel()), dtype=torch.uint8)
        protected_mask = ~cleaning_candidates
        if context_plan is not None:
            reason_codes[protected_mask] = 3
            reason_codes[missing_metadata_mask] = 5
        else:
            reason_codes[protected_mask] = 6
        reason_codes[overlap_disagreement_mask & keep_mask] = 2
        distance_removed_mask = ~keep_mask
        reason_codes[distance_removed_mask] = 1

        records: list[dict[str, Any]] = []
        self._trace_record_spans = []
        offset = 0
        for layout, embedding in zip(self._batch_layouts, embedding_store):
            count = int(embedding.shape[0])
            paths = layout["image_paths"]
            grid_h, grid_w = layout["grid_size"]
            per_image = grid_h * grid_w
            for image_idx, staged_path in enumerate(paths):
                image_start = offset + image_idx * per_image
                image_end = image_start + per_image
                local_keep = keep_mask[image_start:image_end]
                removed = torch.nonzero(~local_keep, as_tuple=False).flatten()
                distance_removed = torch.nonzero(
                    distance_removed_mask[image_start:image_end], as_tuple=False
                ).flatten()
                protected = torch.nonzero(
                    protected_mask[image_start:image_end], as_tuple=False
                ).flatten()
                raw_outliers = torch.nonzero(
                    raw_remove_mask[image_start:image_end], as_tuple=False
                ).flatten()
                candidates = torch.nonzero(
                    cleaning_candidates[image_start:image_end], as_tuple=False
                ).flatten()
                source = self.trace_sources.get(staged_path)
                if source is None:
                    source = self.trace_sources.get(Path(staged_path).name)
                if source is None:
                    raise RuntimeError(
                        f"feature cleaning trace source not found: {staged_path}"
                    )
                local_distances = (
                    [
                        round(float(value), 6)
                        for value in kth_distances[image_start:image_end].tolist()
                    ]
                    if kth_distances is not None
                    else [None] * per_image
                )
                record = {
                    "tile_pool_id": source.get("tile_pool_id"),
                    "source_path": str(source.get("source_path") or staged_path),
                    "panel_path": source.get("panel_path"),
                    "tile_index": source.get("tile_index"),
                    "tile_x": source.get("tile_x"),
                    "tile_y": source.get("tile_y"),
                    "tile_width": source.get("tile_width"),
                    "tile_height": source.get("tile_height"),
                    "input_size": layout.get("input_size"),
                    "grid_size": [grid_h, grid_w],
                    "removed_indices": [int(value) for value in removed.tolist()],
                    "removed_count": int(removed.numel()),
                    "distance_removed_indices": [
                        int(value) for value in distance_removed.tolist()
                    ],
                    "distance_removed_count": int(distance_removed.numel()),
                    "protected_indices": [int(value) for value in protected.tolist()],
                    "protected_count": int(protected.numel()),
                    "candidate_indices": [int(value) for value in candidates.tolist()],
                    "raw_outlier_indices": [int(value) for value in raw_outliers.tolist()],
                    "distances": local_distances,
                    "reason_codes": [
                        int(value)
                        for value in reason_codes[image_start:image_end].tolist()
                    ],
                    "overlap_view_counts": [
                        int(value)
                        for value in overlap_view_counts[image_start:image_end].tolist()
                    ],
                    "outlier_vote_counts": [
                        int(value)
                        for value in outlier_vote_counts[image_start:image_end].tolist()
                    ],
                    "outlier_vote_required": [
                        int(value)
                        for value in outlier_vote_required[image_start:image_end].tolist()
                    ],
                    "coreset_indices": [],
                    "coreset_count": 0,
                }
                records.append(record)
                self._trace_record_spans.append((record, image_start, per_image))
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
        available_reference_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.device]:
        nonempty = next(embedding for embedding in embedding_store if embedding.shape[0])
        device = nonempty.device
        feature_dim = int(nonempty.shape[1])

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        if available_reference_indices is None:
            available_reference_indices = torch.arange(total, dtype=torch.long)
        else:
            available_reference_indices = available_reference_indices.to(
                device="cpu", dtype=torch.long
            )
        available_count = int(available_reference_indices.numel())
        if reference_size == available_count:
            reference_indices = available_reference_indices
        else:
            positions = torch.randperm(available_count, generator=generator)[:reference_size]
            reference_indices = available_reference_indices[positions]

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
            "strategy": self.strategy,
            "threshold": threshold,
            "k": self.k,
            "keep_ratio": self.keep_ratio,
            "seed": self.seed,
            "reference_size": reference_size,
            "elapsed_seconds": time.perf_counter() - started,
            "applied": applied,
            "reason": reason,
        }
