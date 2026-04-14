"""DINOv2 ViT-B/14 feature extraction with embedding cache."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Sequence

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


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _manifest_fingerprint(samples: Sequence[Sample]) -> str:
    """Stable hash over sample_ids (任何新增/移除/重排會觸發 cache 失效)。"""
    h = hashlib.sha256()
    for s in samples:
        h.update(s.sample_id.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def extract_batch(
    model: torch.nn.Module,
    samples: Sequence[Sample],
    transform: transforms.Compose,
    batch_size: int = 32,
) -> np.ndarray:
    """Batched CLS-token extraction. Auto-falls back on CUDA OOM: 32→16→8→4→CPU."""
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
                emb = model(batch)  # DINOv2 forward → CLS token (B, 768)
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
    checkpoint_path: Path | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Return embeddings (N × 768). Load cache if fingerprint matches, else compute & save."""
    cache_path = Path(cache_path)
    fingerprint = _manifest_fingerprint(samples)

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
    transform = build_transform()
    embeddings = extract_batch(model, samples, transform, batch_size=batch_size)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, embeddings=embeddings, fingerprint=np.array(fingerprint))
    logger.info("Saved embeddings to cache: %s (N=%d)", cache_path, embeddings.shape[0])
    return embeddings
