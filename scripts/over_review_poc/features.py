"""DINOv2 ViT-B/14 feature extraction with embedding cache."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from scripts.over_review_poc.dataset import Sample

logger = logging.getLogger(__name__)

DINOV2_REPO = "facebookresearch/dinov2"
DINOV2_MODEL = "dinov2_vitb14"
EMBEDDING_DIM = 768
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Preprocessing pipeline version — bump to invalidate embedding caches when
# the transform semantics change. Main path is v1 naive resize; v2 panel-mask
# helpers below are kept for ablation (see build_transform_otsu_aspect).
PREPROCESSING_VERSION = "v1_naive_resize"
_DARK_THRESHOLD = 30       # pixels < this are treated as "panel-exterior black"
_MIN_DARK_RATIO = 0.05     # below this → no meaningful black region, skip cropping
_MAX_DARK_RATIO = 0.95     # above this → image is mostly black, skip cropping


def load_dinov2(checkpoint_path: Path | None = None, device: str | None = None) -> torch.nn.Module:
    """Load DINOv2 ViT-B/14 via torch.hub; optional local checkpoint override."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_path is not None:
        model = torch.hub.load(DINOV2_REPO, DINOV2_MODEL, pretrained=False, source="github")
        state = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(state)
        logger.info("Loaded DINOv2 from local checkpoint: %s", checkpoint_path)
    else:
        model = torch.hub.load(DINOV2_REPO, DINOV2_MODEL, source="github")
        logger.info("Loaded DINOv2 via torch.hub (default cache)")

    model = model.to(device).eval()
    return model


def _find_panel_bbox(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Otsu-based panel bbox. Returns None if no meaningful black region.

    edge_defect crops often contain large black areas outside the panel; we
    crop to the panel region so DINOv2's CLS token isn't dominated by the
    black/gray layout edge. patchcore_tile crops are uniformly panel → Otsu
    would split arbitrarily, so we short-circuit on dark-pixel ratio.
    """
    dark_ratio = float((gray < _DARK_THRESHOLD).mean())
    if dark_ratio < _MIN_DARK_RATIO or dark_ratio > _MAX_DARK_RATIO:
        return None
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(binary > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _aspect_resize_pad(img: np.ndarray, target: int, pad_value: int) -> np.ndarray:
    """Resize longest side to `target`, pad to target×target with `pad_value`."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((target, target, 3), pad_value, dtype=np.uint8)
    scale = target / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.full((target, target, 3), pad_value, dtype=np.uint8)
    top = (target - new_h) // 2
    left = (target - new_w) // 2
    out[top:top + new_h, left:left + new_w] = resized
    return out


def preprocess_crop(pil_img: Image.Image) -> Image.Image:
    """Otsu-crop panel (if meaningful black region) → aspect-preserve resize + pad.

    Ablation helper (not on default path). Intended to reduce the layout-edge
    bias in DINOv2 CLS tokens for crops with large black background. Wire via
    `build_transform_otsu_aspect`.
    """
    rgb = np.asarray(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bbox = _find_panel_bbox(gray)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        panel = rgb[y0:y1, x0:x1]
        pad_value = int(np.median(gray[y0:y1, x0:x1]))
    else:
        panel = rgb
        pad_value = int(np.median(gray))
    return Image.fromarray(_aspect_resize_pad(panel, INPUT_SIZE, pad_value))


def build_transform() -> transforms.Compose:
    """Naive resize (v1 baseline, production-default)."""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_transform_otsu_aspect() -> transforms.Compose:
    """Experimental: Otsu panel crop + aspect-preserve pad.

    Ablation run (2026-04-14) showed this is a wash vs. the naive resize on the
    scratch POC — metric delta within 1 std. Kept for future combination with
    CLAHE or patch-level pooling experiments.
    """
    return transforms.Compose([
        transforms.Lambda(preprocess_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_clahe(pil_img: Image.Image,
                     clip_limit: float = 2.0,
                     tile_grid: int = 8) -> Image.Image:
    """Grayscale CLAHE contrast enhancement → 3-channel RGB.

    Targets the A-type misses (low-contrast small scratches) by locally
    equalizing histogram in each tile. Industrial AOI crops are effectively
    grayscale (R=G=B), so we apply CLAHE on the single luminance channel and
    stack back to 3 channels for DINOv2.
    """
    rgb = np.asarray(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    enhanced = clahe.apply(gray)
    return Image.fromarray(np.stack([enhanced] * 3, axis=-1))


def build_transform_clahe(clip_limit: float = 2.0,
                          tile_grid: int = 8) -> transforms.Compose:
    """CLAHE contrast enhancement → naive resize → normalize."""
    def _clahe(img):
        return preprocess_clahe(img, clip_limit, tile_grid)
    return transforms.Compose([
        transforms.Lambda(_clahe),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_clahe_tile_only(pil_img: Image.Image,
                               clip_limit: float = 2.0,
                               tile_grid: int = 8) -> Image.Image:
    """CLAHE only on panel-interior tiles; skip for edge crops (large black area).

    Edge-defect crops have ~50% dark pixels (area outside physical panel);
    CLAHE there would amplify noise in the useless dark region. Heuristic: if
    dark_ratio ≥ 5% then skip CLAHE and pass through unchanged.
    """
    rgb = np.asarray(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark_ratio = float((gray < _DARK_THRESHOLD).mean())
    if dark_ratio >= _MIN_DARK_RATIO:
        return Image.fromarray(rgb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    enhanced = clahe.apply(gray)
    return Image.fromarray(np.stack([enhanced] * 3, axis=-1))


def build_transform_clahe_tile_only(clip_limit: float = 2.0,
                                    tile_grid: int = 8) -> transforms.Compose:
    """CLAHE on tiles only (pass-through for edge crops) → resize → normalize."""
    def _clahe_tile(img):
        return preprocess_clahe_tile_only(img, clip_limit, tile_grid)
    return transforms.Compose([
        transforms.Lambda(_clahe_tile),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _manifest_fingerprint(samples: Sequence[Sample],
                          preprocessing_id: str | None = None) -> str:
    """Stable hash over preprocessing id + sample_ids.

    Changing `preprocessing_id` (or the module-level PREPROCESSING_VERSION
    default) invalidates the cache — new transforms produce different
    embeddings so we must not reuse a stale cache.
    """
    if preprocessing_id is None:
        preprocessing_id = PREPROCESSING_VERSION
    h = hashlib.sha256()
    h.update(preprocessing_id.encode("utf-8"))
    h.update(b"\x00")
    for s in samples:
        h.update(s.sample_id.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _pool_features(cls_token: torch.Tensor,
                   patch_tokens: torch.Tensor,
                   pool_mode: str) -> torch.Tensor:
    """Pool DINOv2 outputs into a single image representation.

    Args:
        cls_token: (B, D) normalized CLS token
        patch_tokens: (B, N, D) normalized patch tokens (N=196 for 224×14)
        pool_mode: 'cls' | 'max' | 'cls_max' | 'cls_max_anom' | 'cls_top5_anom'

    Returns:
        (B, D) or (B, 2D) tensor depending on mode.
    """
    if pool_mode == "cls":
        return cls_token
    if pool_mode == "max":
        return patch_tokens.max(dim=1).values
    if pool_mode == "cls_max":
        return torch.cat([cls_token, patch_tokens.max(dim=1).values], dim=-1)
    # Anomaly-aware: rank patches by L2 distance from per-image mean patch
    mean_patch = patch_tokens.mean(dim=1, keepdim=True)      # (B, 1, D)
    dist = (patch_tokens - mean_patch).norm(dim=-1)          # (B, N)
    if pool_mode == "cls_max_anom":
        idx = dist.argmax(dim=-1)                             # (B,)
        top_patch = patch_tokens[torch.arange(patch_tokens.size(0)), idx]
        return torch.cat([cls_token, top_patch], dim=-1)
    if pool_mode == "cls_top5_anom":
        top5_idx = dist.topk(5, dim=-1).indices               # (B, 5)
        gather_idx = top5_idx.unsqueeze(-1).expand(-1, -1, patch_tokens.size(-1))
        top5 = torch.gather(patch_tokens, 1, gather_idx)      # (B, 5, D)
        return torch.cat([cls_token, top5.mean(dim=1)], dim=-1)
    raise ValueError(f"Unknown pool_mode: {pool_mode}")


def extract_batch(
    model: torch.nn.Module,
    samples: Sequence[Sample],
    transform: transforms.Compose,
    batch_size: int = 32,
    pool_mode: str = "cls",
) -> np.ndarray:
    """Batched feature extraction. Auto-falls back on CUDA OOM: 32→16→8→4→CPU.

    pool_mode='cls' uses the fast `model(x)` call (CLS-only). Other modes call
    `model.forward_features(x)` and apply `_pool_features` to aggregate patch
    tokens; output dim may be D or 2D depending on mode.
    """
    device = next(model.parameters()).device
    all_embs: list[np.ndarray] = []

    i = 0
    current_bs = batch_size
    while i < len(samples):
        chunk_samples = samples[i:i + current_bs]
        try:
            imgs = [transform(Image.open(s.crop_path).convert("RGB")) for s in chunk_samples]
            batch = torch.stack(imgs).to(device)
            with torch.no_grad():
                if pool_mode == "cls":
                    emb = model(batch)  # (B, D)
                else:
                    out = model.forward_features(batch)
                    cls = out["x_norm_clstoken"]
                    patches = out["x_norm_patchtokens"]
                    emb = _pool_features(cls, patches, pool_mode)
            all_embs.append(emb.cpu().numpy())
            i += current_bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_bs > 1:
                current_bs = max(1, current_bs // 2)
                logger.warning("CUDA OOM; reducing batch to %d", current_bs)
            else:
                logger.warning("CUDA OOM at batch=1; falling back to CPU")
                device = torch.device("cpu")
                model.to(device)

    return np.concatenate(all_embs, axis=0)


def get_or_extract(
    samples: Sequence[Sample],
    cache_path: Path,
    transform_factory: "callable" = None,
    preprocessing_id: str | None = None,
    checkpoint_path: Path | None = None,
    batch_size: int = 32,
    pool_mode: str = "cls",
) -> np.ndarray:
    """Return embeddings (N × D). Load cache if fingerprint matches, else compute & save.

    Args:
        transform_factory: zero-arg callable returning a torchvision Compose.
            Defaults to `build_transform` (v1 naive resize).
        preprocessing_id: string key identifying the transform+pool combination;
            goes into the cache fingerprint. Defaults to module-level
            PREPROCESSING_VERSION. Callers should embed pool_mode into the id
            when they use non-cls pooling.
        pool_mode: see `_pool_features`. Non-cls modes use `forward_features`
            and may produce embeddings with dim > 768 (e.g. cls_max → 1536).
    """
    if transform_factory is None:
        transform_factory = build_transform
    cache_path = Path(cache_path)
    fingerprint = _manifest_fingerprint(samples, preprocessing_id)

    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=False)
            cached_fp = str(data["fingerprint"].item())
            if cached_fp == fingerprint:
                logger.info("Loaded embeddings from cache: %s (N=%d)",
                            cache_path, data["embeddings"].shape[0])
                return data["embeddings"]
            logger.info("Cache fingerprint mismatch; recomputing")
        except Exception as e:
            logger.warning("Failed to read cache (%s); recomputing", e)

    model = load_dinov2(checkpoint_path)
    transform = transform_factory()
    embeddings = extract_batch(model, samples, transform, batch_size=batch_size,
                               pool_mode=pool_mode)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, embeddings=embeddings, fingerprint=np.array(fingerprint))
    logger.info("Saved embeddings to cache: %s (N=%d, D=%d)",
                cache_path, embeddings.shape[0], embeddings.shape[1])
    return embeddings
