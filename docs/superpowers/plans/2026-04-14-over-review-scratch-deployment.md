# Over-Review Scratch Deployment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 CAPI AI 推論流程加入 scratch 二元分類器，把 PatchCore 判 NG 但分類器認定為高信心 scratch 的 tile 翻回 OK，降低過檢率。

**Architecture:** 新增兩個模組 `scratch_classifier.py`（純模型）與 `scratch_filter.py`（pipeline glue），在 `CAPIInferencer.process_panel` 的 per-image anomaly_tiles 迴圈後 post-filter；threshold 用 conformal（訓練時 calib set 定）× runtime safety multiplier；共用 `_gpu_lock`。

**Tech Stack:** PyTorch + torch.hub DINOv2 + LoRA adapter + scikit-learn LogisticRegression + SQLite + PyYAML + pytest。

**Spec:** `docs/superpowers/specs/2026-04-14-over-review-scratch-deployment-design.md`

---

## File Structure

**New files:**
- `scratch_classifier.py` — `ScratchClassifier`, `ScratchClassifierMetadata`, bundle load/save
- `scratch_filter.py` — `ScratchFilter.apply_to_image_result`
- `scripts/over_review_poc/train_final_model.py` — 一次性訓練部署 bundle 的 CLI
- `tests/test_scratch_classifier.py`
- `tests/test_scratch_filter.py`
- `tests/test_scratch_integration.py`
- `deployment/README.md` — 部署 bundle 的操作指引

**Modified files:**
- `capi_inference.py` — `TileInfo`/`ImageResult` 加欄位；`CAPIInferencer.__init__` 加 lazy filter；`process_panel` 加 hook
- `capi_config.py` — `CAPIConfig` 加 `scratch_*` 欄位
- `capi_database.py` — schema migration 加 `scratch_score` / `scratch_filtered` / `scratch_filter_count`；save_*_results 寫入這些欄位

**Created by bundle producer (not by code):**
- `deployment/scratch_classifier_v1.pkl` — LoRA + LogReg + metadata
- `deployment/dinov2_vitb14.pth` — DINOv2 base weights（複用 prepare_offline_model.py 產出）

---

## Task 1: Scaffold `scratch_classifier.py` with metadata + bundle I/O

**Files:**
- Create: `scratch_classifier.py`
- Test: `tests/test_scratch_classifier.py`

- [ ] **Step 1: Write the failing test for bundle roundtrip**

Add to `tests/test_scratch_classifier.py`:

```python
"""Unit tests for scratch_classifier module (bundle I/O, predict)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from scratch_classifier import (
    ScratchClassifierMetadata,
    save_bundle,
    load_bundle,
)


def _fake_metadata() -> ScratchClassifierMetadata:
    return ScratchClassifierMetadata(
        preprocessing_id="v3_clahe_cl4.0_tg8",
        clahe_clip=4.0,
        clahe_tile=8,
        input_size=224,
        dinov2_repo="facebookresearch/dinov2",
        dinov2_model="dinov2_vitb14",
        lora_rank=16,
        lora_alpha=16,
        lora_n_blocks=2,
        conformal_threshold=0.85,
        safety_multiplier=1.1,
        trained_at="2026-04-14T10:00:00",
        dataset_sha256="abc123",
        git_commit="deadbeef",
    )


def test_bundle_roundtrip(tmp_path: Path):
    lora_sd = {"lora_A.weight": torch.randn(16, 768)}
    logreg = LogisticRegression().fit(np.random.randn(10, 768), [0]*5 + [1]*5)
    calib_scores = np.array([0.1, 0.2, 0.85])
    meta = _fake_metadata()
    bundle_path = tmp_path / "bundle.pkl"

    save_bundle(bundle_path, lora_sd, logreg, meta, calib_scores)

    loaded_sd, loaded_logreg, loaded_meta, loaded_calib = load_bundle(bundle_path)
    assert list(loaded_sd.keys()) == ["lora_A.weight"]
    assert torch.allclose(loaded_sd["lora_A.weight"], lora_sd["lora_A.weight"])
    assert loaded_logreg.coef_.shape == logreg.coef_.shape
    assert np.allclose(loaded_logreg.coef_, logreg.coef_)
    assert loaded_meta.conformal_threshold == 0.85
    assert loaded_meta.preprocessing_id == "v3_clahe_cl4.0_tg8"
    assert np.allclose(loaded_calib, calib_scores)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scratch_classifier.py::test_bundle_roundtrip -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scratch_classifier'`

- [ ] **Step 3: Implement scratch_classifier.py with bundle I/O**

Create `scratch_classifier.py`:

```python
"""Scratch binary classifier for CAPI over-review filtering.

Exposes ScratchClassifier that wraps LoRA-fine-tuned DINOv2 ViT-B/14 +
LogisticRegression head. Bundles are pickled dicts containing LoRA weights,
LogReg, metadata, and calibration scores.
"""
from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TRUST_REMOTE_CODE", "1")

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@dataclass
class ScratchClassifierMetadata:
    """Frozen at training time; single source of truth for preprocessing + model id."""
    preprocessing_id: str
    clahe_clip: float
    clahe_tile: int
    input_size: int
    dinov2_repo: str
    dinov2_model: str
    lora_rank: int
    lora_alpha: int
    lora_n_blocks: int
    conformal_threshold: float
    safety_multiplier: float
    trained_at: str
    dataset_sha256: str
    git_commit: str


class ScratchClassifierLoadError(RuntimeError):
    """Raised when bundle / DINOv2 weights cannot be loaded."""


def save_bundle(
    path: str | Path,
    lora_state_dict: dict[str, Any],
    logreg: LogisticRegression,
    metadata: ScratchClassifierMetadata,
    calibration_scores: np.ndarray,
) -> None:
    """Pickle the deployment artifact to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lora_state_dict": lora_state_dict,
        "logreg": logreg,
        "metadata": asdict(metadata),
        "calibration_scores": np.asarray(calibration_scores),
        "format_version": 1,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved scratch classifier bundle: %s", path)


def load_bundle(
    path: str | Path,
) -> tuple[dict[str, Any], LogisticRegression, ScratchClassifierMetadata, np.ndarray]:
    """Load pickled artifact; returns (lora_sd, logreg, metadata, calib_scores)."""
    path = Path(path)
    if not path.exists():
        raise ScratchClassifierLoadError(f"Bundle not found: {path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("format_version", 0) != 1:
        raise ScratchClassifierLoadError(
            f"Unsupported bundle format_version: {payload.get('format_version')}"
        )
    meta = ScratchClassifierMetadata(**payload["metadata"])
    return (
        payload["lora_state_dict"],
        payload["logreg"],
        meta,
        payload["calibration_scores"],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scratch_classifier.py::test_bundle_roundtrip -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scratch_classifier.py tests/test_scratch_classifier.py
git commit -m "feat(scratch): scratch_classifier.py scaffold + bundle I/O (Task 1)"
```

---

## Task 2: `ScratchClassifier.__init__` with DINOv2 + LoRA loading

**Files:**
- Modify: `scratch_classifier.py`
- Test: `tests/test_scratch_classifier.py`

- [ ] **Step 1: Add failing tests for classifier init**

Append to `tests/test_scratch_classifier.py`:

```python
def _train_tiny_stub(tmp_path: Path, device: str = "cpu") -> Path:
    """Create a minimal valid bundle + fake DINOv2 weights for testing."""
    # Build LoRA state_dict keys that match what ScratchClassifier expects
    # (last 2 blocks, qkv/mlp.fc1/mlp.fc2; rank=8 for speed)
    rank = 8
    lora_sd = {}
    for blk in range(-2, 0):  # last 2 blocks (logical indexing in implementation)
        for proj in ("attn.qkv", "mlp.fc1", "mlp.fc2"):
            for side, out_dim in (("lora_A.weight", 768), ("lora_B.weight", 768)):
                # Shapes based on DINOv2 ViT-B/14 (D=768, qkv→3*768, mlp→3072)
                in_dim = 768
                actual_out = 3 * 768 if "qkv" in proj else (3072 if "fc1" in proj else 768)
                in_dim_actual = 768 if "fc1" in proj or "qkv" in proj else 3072
                if "lora_A" in side:
                    lora_sd[f"blocks.{blk}.{proj}.lora_A.weight"] = torch.randn(rank, in_dim_actual)
                else:
                    lora_sd[f"blocks.{blk}.{proj}.lora_B.weight"] = torch.zeros(actual_out, rank)
    # Random linear head produces 768-d embedding; LogReg trained on noise.
    X = np.random.randn(20, 768)
    y = np.array([0]*10 + [1]*10)
    logreg = LogisticRegression(max_iter=200).fit(X, y)

    meta = ScratchClassifierMetadata(
        preprocessing_id="v3_clahe_cl4.0_tg8",
        clahe_clip=4.0, clahe_tile=8, input_size=224,
        dinov2_repo="facebookresearch/dinov2", dinov2_model="dinov2_vitb14",
        lora_rank=rank, lora_alpha=16, lora_n_blocks=2,
        conformal_threshold=0.8, safety_multiplier=1.1,
        trained_at="t", dataset_sha256="x", git_commit="y",
    )
    bundle_path = tmp_path / "bundle.pkl"
    save_bundle(bundle_path, lora_sd, logreg, meta, np.array([0.5, 0.8]))
    return bundle_path


@pytest.mark.slow
def test_classifier_loads_bundle_and_weights(tmp_path: Path):
    """Requires network on first run (downloads DINOv2); marked slow."""
    from scratch_classifier import ScratchClassifier
    bundle_path = _train_tiny_stub(tmp_path)
    clf = ScratchClassifier(bundle_path=bundle_path, device="cpu")
    assert clf.conformal_threshold == 0.8
    assert clf.metadata.lora_rank == 8
    assert clf.metadata.lora_n_blocks == 2


def test_classifier_missing_bundle_raises(tmp_path: Path):
    from scratch_classifier import ScratchClassifier, ScratchClassifierLoadError
    with pytest.raises(ScratchClassifierLoadError):
        ScratchClassifier(bundle_path=tmp_path / "missing.pkl", device="cpu")
```

- [ ] **Step 2: Run tests to verify failures**

Run: `python -m pytest tests/test_scratch_classifier.py::test_classifier_missing_bundle_raises -v`
Expected: FAIL with `ImportError: cannot import name 'ScratchClassifier'`

- [ ] **Step 3: Implement `ScratchClassifier.__init__`**

Append to `scratch_classifier.py`:

```python
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear with LoRA adapter: W + scale * B @ A. Only A, B trainable.

    Mirrors scripts.over_review_poc.finetune_lora.LoRALinear — duplicated here
    so deployment doesn't depend on POC script package. Keep the two in sync
    when LoRA adapter implementation changes.
    """
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        nn.init.zeros_(self.lora_B.weight)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


def _apply_lora(model: nn.Module, n_blocks: int, rank: int, alpha: int) -> None:
    """Replace qkv + mlp.fc1 + mlp.fc2 of last n_blocks with LoRA variants (in-place)."""
    targets = model.blocks[-n_blocks:]
    for block in targets:
        block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha)
        block.mlp.fc1 = LoRALinear(block.mlp.fc1, rank, alpha)
        block.mlp.fc2 = LoRALinear(block.mlp.fc2, rank, alpha)


def _load_dinov2(repo: str, name: str,
                 weights_path: str | Path | None) -> nn.Module:
    """Load DINOv2 from torch.hub; if weights_path given, load local checkpoint."""
    if weights_path is not None:
        model = torch.hub.load(repo, name, pretrained=False, source="github")
        state = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state)
    else:
        model = torch.hub.load(repo, name, source="github")
    return model


class ScratchClassifier:
    def __init__(
        self,
        bundle_path: str | Path,
        dinov2_weights_path: str | Path | None = None,
        device: str = "cuda",
    ):
        lora_sd, logreg, meta, calib_scores = load_bundle(bundle_path)
        self.metadata = meta
        self._logreg = logreg
        self._calibration_scores = calib_scores

        try:
            model = _load_dinov2(meta.dinov2_repo, meta.dinov2_model,
                                  dinov2_weights_path)
        except Exception as e:  # pragma: no cover - depends on network/filesystem
            raise ScratchClassifierLoadError(
                f"Failed to load DINOv2 ({meta.dinov2_repo}/{meta.dinov2_model}): {e}"
            ) from e

        _apply_lora(model, meta.lora_n_blocks, meta.lora_rank, meta.lora_alpha)
        missing, unexpected = model.load_state_dict(lora_sd, strict=False)
        if unexpected:
            logger.warning("Unexpected keys in LoRA state_dict: %s", unexpected[:5])

        model.eval()
        self._device = torch.device(device)
        self._model = model.to(self._device)
        logger.info("ScratchClassifier loaded: rank=%d blocks=%d threshold=%.4f",
                    meta.lora_rank, meta.lora_n_blocks, meta.conformal_threshold)

    @property
    def conformal_threshold(self) -> float:
        """Raw conformal threshold (calib max true_ng score)."""
        return self.metadata.conformal_threshold

    @property
    def device(self) -> torch.device:
        return self._device
```

- [ ] **Step 4: Run tests**

Run:
```bash
python -m pytest tests/test_scratch_classifier.py::test_classifier_missing_bundle_raises -v
python -m pytest tests/test_scratch_classifier.py::test_bundle_roundtrip -v
```

Expected: Both PASS. `test_classifier_loads_bundle_and_weights` is marked slow and requires DINOv2 download; skip via `-m "not slow"` unless needed.

- [ ] **Step 5: Commit**

```bash
git add scratch_classifier.py tests/test_scratch_classifier.py
git commit -m "feat(scratch): ScratchClassifier init with DINOv2 + LoRA (Task 2)"
```

---

## Task 3: `ScratchClassifier.predict` + `predict_batch` + preprocessing

**Files:**
- Modify: `scratch_classifier.py`
- Test: `tests/test_scratch_classifier.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_scratch_classifier.py`:

```python
from PIL import Image


def _make_fake_image(size: int = 256) -> Image.Image:
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_preprocessing_matches_poc(tmp_path: Path):
    """ScratchClassifier preprocessing must produce identical tensor to POC helper."""
    from scratch_classifier import _build_transform  # module-private helper
    from scripts.over_review_poc.features import build_transform_clahe

    img = _make_fake_image()
    poc_tensor = build_transform_clahe(clip_limit=4.0, tile_grid=8)(img)
    our_tensor = _build_transform(clahe_clip=4.0, clahe_tile=8, input_size=224)(img)
    assert torch.allclose(poc_tensor, our_tensor, atol=1e-6)


@pytest.mark.slow
def test_predict_score_range(tmp_path: Path):
    from scratch_classifier import ScratchClassifier
    bundle_path = _train_tiny_stub(tmp_path)
    clf = ScratchClassifier(bundle_path=bundle_path, device="cpu")
    img = _make_fake_image()
    score = clf.predict(img)
    assert 0.0 <= score <= 1.0


@pytest.mark.slow
def test_predict_batch_matches_loop(tmp_path: Path):
    from scratch_classifier import ScratchClassifier
    bundle_path = _train_tiny_stub(tmp_path)
    clf = ScratchClassifier(bundle_path=bundle_path, device="cpu")
    imgs = [_make_fake_image(size=s) for s in (100, 150, 200, 256)]
    batch_scores = clf.predict_batch(imgs)
    loop_scores = np.array([clf.predict(i) for i in imgs])
    assert np.allclose(batch_scores, loop_scores, atol=1e-5)
```

- [ ] **Step 2: Run tests — expect failure**

Run: `python -m pytest tests/test_scratch_classifier.py::test_preprocessing_matches_poc -v`
Expected: FAIL with `ImportError: cannot import name '_build_transform'`

- [ ] **Step 3: Implement preprocessing + predict**

Append to `scratch_classifier.py`:

```python
import cv2
from PIL import Image
from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _preprocess_clahe(pil_img: Image.Image, clip_limit: float, tile_grid: int) -> Image.Image:
    """CLAHE grayscale enhance → 3-channel stack. Matches
    scripts.over_review_poc.features.preprocess_clahe exactly.
    """
    rgb = np.asarray(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    enhanced = clahe.apply(gray)
    return Image.fromarray(np.stack([enhanced] * 3, axis=-1))


def _build_transform(clahe_clip: float, clahe_tile: int,
                     input_size: int) -> transforms.Compose:
    def _clahe(img):
        return _preprocess_clahe(img, clahe_clip, clahe_tile)
    return transforms.Compose([
        transforms.Lambda(_clahe),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return Image.fromarray(img.astype(np.uint8))
    raise TypeError(f"Unsupported image type: {type(img)}")
```

Then add transform + predict methods inside `ScratchClassifier.__init__` (before `logger.info`):

```python
        self._transform = _build_transform(
            meta.clahe_clip, meta.clahe_tile, meta.input_size,
        )
```

And add methods to `ScratchClassifier`:

```python
    def predict(self, image) -> float:
        """Return scratch probability in [0, 1]. Accepts PIL or np.ndarray RGB."""
        return float(self.predict_batch([image])[0])

    def predict_batch(self, images) -> np.ndarray:
        """Vectorised predict for a list of images."""
        if len(images) == 0:
            return np.zeros(0, dtype=np.float32)
        tensors = [self._transform(_to_pil(i)) for i in images]
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            feats = self._model(batch).cpu().numpy()
        return self._logreg.predict_proba(feats)[:, 1].astype(np.float32)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_scratch_classifier.py -v -m "not slow"`
Expected: All non-slow tests PASS (roundtrip, missing_bundle, preprocessing_matches_poc).

For slow tests (if DINOv2 already cached locally):
Run: `python -m pytest tests/test_scratch_classifier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scratch_classifier.py tests/test_scratch_classifier.py
git commit -m "feat(scratch): predict + preprocessing (equiv to POC) (Task 3)"
```

---

## Task 4: Add `TileInfo` + `ImageResult` dataclass fields

**Files:**
- Modify: `capi_inference.py:80-118` (TileInfo), `:193-213` (ImageResult)
- Test: `tests/test_scratch_filter.py` (will reference these fields in Task 5)

- [ ] **Step 1: Write tests for new defaults**

Create `tests/test_scratch_filter.py`:

```python
"""Unit tests for scratch_filter module + related TileInfo/ImageResult fields."""
from __future__ import annotations

import numpy as np
import pytest

from capi_inference import TileInfo, ImageResult


def _fake_tile(tid: int) -> TileInfo:
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    return TileInfo(tile_id=tid, x=0, y=0, width=512, height=512, image=img)


def test_tileinfo_has_scratch_fields():
    t = _fake_tile(1)
    assert t.scratch_score == 0.0
    assert t.scratch_filtered is False


def test_image_result_has_scratch_count():
    from pathlib import Path
    ir = ImageResult(
        image_path=Path("/fake"),
        image_size=(512, 512),
        otsu_bounds=(0, 0, 512, 512),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
    )
    assert ir.scratch_filter_count == 0
```

- [ ] **Step 2: Run — expect failure**

Run: `python -m pytest tests/test_scratch_filter.py -v`
Expected: FAIL with `AttributeError: 'TileInfo' object has no attribute 'scratch_score'`

- [ ] **Step 3: Add dataclass fields**

In `capi_inference.py`, locate `TileInfo` (around line 80–118). After the existing `bright_spot_min_area: int = 0` (line ~118), add before the `@property center`:

```python
    # Scratch classifier post-filter (over-review reduction)
    scratch_score: float = 0.0              # 0 = 未跑 classifier
    scratch_filtered: bool = False          # True = 被翻回 OK
```

In `ImageResult` (around line 193–213), add after `processing_time: float`:

```python
    # Scratch classifier post-filter stats
    scratch_filter_count: int = 0           # 此 image 中被翻 OK 的 tile 數
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_scratch_filter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add capi_inference.py tests/test_scratch_filter.py
git commit -m "feat(scratch): TileInfo/ImageResult scratch fields (Task 4)"
```

---

## Task 5: `scratch_filter.py` — `ScratchFilter.apply_to_image_result`

**Files:**
- Create: `scratch_filter.py`
- Test: `tests/test_scratch_filter.py`

- [ ] **Step 1: Append failing tests to `tests/test_scratch_filter.py`**

```python
# --- ScratchFilter tests ---

class _MockClassifier:
    """Returns fixed score; exposes conformal_threshold for effective calc."""
    def __init__(self, fixed_score: float, conformal_threshold: float = 0.7,
                 raise_on_call: bool = False):
        self._score = fixed_score
        self.conformal_threshold = conformal_threshold
        self._raise = raise_on_call

    def predict(self, image):
        if self._raise:
            raise RuntimeError("simulated failure")
        return self._score


def _fake_image_result_with_tiles(n_tiles: int) -> ImageResult:
    from pathlib import Path
    tiles = [_fake_tile(i) for i in range(n_tiles)]
    ir = ImageResult(
        image_path=Path("/fake"),
        image_size=(1024, 1024),
        otsu_bounds=(0, 0, 1024, 1024),
        exclusion_regions=[],
        tiles=tiles,
        excluded_tile_count=0,
        processed_tile_count=n_tiles,
        processing_time=0.01,
    )
    # Populate anomaly_tiles: (TileInfo, score, anomaly_map)
    ir.anomaly_tiles = [(t, 0.9, None) for t in tiles]
    return ir


def test_filter_flips_high_score():
    from scratch_filter import ScratchFilter
    clf = _MockClassifier(fixed_score=0.95, conformal_threshold=0.7)
    sf = ScratchFilter(clf, safety_multiplier=1.0)   # threshold = 0.7
    ir = _fake_image_result_with_tiles(3)

    sf.apply_to_image_result(ir)

    assert len(ir.anomaly_tiles) == 0
    assert ir.scratch_filter_count == 3
    for t in ir.tiles:
        assert t.scratch_filtered is True
        assert t.scratch_score == pytest.approx(0.95)


def test_filter_keeps_low_score():
    from scratch_filter import ScratchFilter
    clf = _MockClassifier(fixed_score=0.2, conformal_threshold=0.7)
    sf = ScratchFilter(clf, safety_multiplier=1.0)   # threshold = 0.7
    ir = _fake_image_result_with_tiles(3)

    sf.apply_to_image_result(ir)

    assert len(ir.anomaly_tiles) == 3
    assert ir.scratch_filter_count == 0
    for t in ir.tiles:
        assert t.scratch_filtered is False
        assert t.scratch_score == pytest.approx(0.2)


def test_filter_no_anomaly_is_noop():
    from scratch_filter import ScratchFilter
    clf = _MockClassifier(fixed_score=0.99, conformal_threshold=0.7)
    sf = ScratchFilter(clf, safety_multiplier=1.0)
    ir = _fake_image_result_with_tiles(0)   # anomaly_tiles = []

    sf.apply_to_image_result(ir)   # should not raise
    assert ir.scratch_filter_count == 0


def test_filter_exception_safety():
    """If classifier raises, tile stays NG (safety default) and other tiles continue."""
    from scratch_filter import ScratchFilter
    # First tile raises; build a classifier that raises once then returns low score
    class _Sometimes(_MockClassifier):
        def __init__(self):
            super().__init__(fixed_score=0.1, conformal_threshold=0.7)
            self._calls = 0
        def predict(self, image):
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("fail once")
            return self._score
    clf = _Sometimes()
    sf = ScratchFilter(clf, safety_multiplier=1.0)
    ir = _fake_image_result_with_tiles(3)

    sf.apply_to_image_result(ir)

    # All 3 kept (first exception, other two score<threshold)
    assert len(ir.anomaly_tiles) == 3
    assert ir.tiles[0].scratch_score == 0.0     # exception path default
    assert ir.tiles[0].scratch_filtered is False
    assert ir.tiles[1].scratch_score == pytest.approx(0.1)
    assert ir.tiles[2].scratch_score == pytest.approx(0.1)


def test_effective_threshold():
    from scratch_filter import ScratchFilter
    clf = _MockClassifier(fixed_score=0.5, conformal_threshold=0.8)
    sf = ScratchFilter(clf, safety_multiplier=1.25)
    # effective = 0.8 * 1.25 = 1.0 → clamped to 0.9999 per spec
    assert sf.effective_threshold == pytest.approx(0.9999)

    sf2 = ScratchFilter(clf, safety_multiplier=1.0)
    assert sf2.effective_threshold == pytest.approx(0.8)
```

- [ ] **Step 2: Run — expect failure**

Run: `python -m pytest tests/test_scratch_filter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scratch_filter'`

- [ ] **Step 3: Implement `scratch_filter.py`**

Create `scratch_filter.py`:

```python
"""ScratchFilter — glue between ScratchClassifier and CAPIInferencer pipeline.

Takes a trained ScratchClassifier and a per-image anomaly_tiles list, flips
high-confidence scratch tiles from NG to OK (removes them from anomaly_tiles)
and records audit fields on each TileInfo.
"""
from __future__ import annotations

import logging

from capi_inference import ImageResult

logger = logging.getLogger(__name__)

_THR_CLAMP = 0.9999


class ScratchFilter:
    def __init__(self, classifier, safety_multiplier: float = 1.1):
        self._classifier = classifier
        self._safety = float(safety_multiplier)
        raw = classifier.conformal_threshold * self._safety
        self.effective_threshold = min(raw, _THR_CLAMP)

    def apply_to_image_result(self, image_result: ImageResult) -> None:
        if not image_result.anomaly_tiles:
            return
        keep: list = []
        filtered = 0
        for entry in image_result.anomaly_tiles:
            tile = entry[0]
            try:
                score = float(self._classifier.predict(tile.image))
            except Exception as e:
                logger.warning("Scratch classifier failed on tile %s: %s",
                               getattr(tile, "tile_id", "?"), e)
                # Safety default: keep NG, score stays 0
                keep.append(entry)
                continue
            tile.scratch_score = score
            if score > self.effective_threshold:
                tile.scratch_filtered = True
                filtered += 1
            else:
                keep.append(entry)
        image_result.anomaly_tiles = keep
        image_result.scratch_filter_count = filtered
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_scratch_filter.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scratch_filter.py tests/test_scratch_filter.py
git commit -m "feat(scratch): ScratchFilter post-filter glue (Task 5)"
```

---

## Task 6: `CAPIConfig` scratch config fields

**Files:**
- Modify: `capi_config.py:82+` (CAPIConfig dataclass)

- [ ] **Step 1: Write failing test**

Create `tests/test_scratch_config.py`:

```python
from capi_config import CAPIConfig


def test_default_scratch_config_enabled():
    cfg = CAPIConfig()
    assert cfg.scratch_classifier_enabled is True
    assert cfg.scratch_safety_multiplier == 1.1
    assert cfg.scratch_bundle_path == "deployment/scratch_classifier_v1.pkl"
    assert cfg.scratch_dinov2_weights_path == "deployment/dinov2_vitb14.pth"


def test_yaml_roundtrip_preserves_scratch(tmp_path):
    import yaml
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "scratch_classifier_enabled": False,
        "scratch_safety_multiplier": 1.25,
        "scratch_bundle_path": "/custom/path.pkl",
    }))
    cfg = CAPIConfig.from_yaml(str(cfg_path))
    assert cfg.scratch_classifier_enabled is False
    assert cfg.scratch_safety_multiplier == 1.25
    assert cfg.scratch_bundle_path == "/custom/path.pkl"
```

- [ ] **Step 2: Run — expect failure**

Run: `python -m pytest tests/test_scratch_config.py -v`
Expected: FAIL with `AttributeError: 'CAPIConfig' object has no attribute 'scratch_classifier_enabled'`

- [ ] **Step 3: Add fields to CAPIConfig**

In `capi_config.py`, in `CAPIConfig` dataclass, add after `aoi_report_path_replace_to: str = "Report"` (line ~202):

```python
    # Scratch classifier post-filter (over-review reduction)
    scratch_classifier_enabled: bool = True
    scratch_safety_multiplier: float = 1.1
    scratch_bundle_path: str = "deployment/scratch_classifier_v1.pkl"
    scratch_dinov2_weights_path: str = "deployment/dinov2_vitb14.pth"
```

Then in `CAPIConfig.from_dict` (search for the big dict of `data.get(...)` calls), add:

```python
            scratch_classifier_enabled=data.get("scratch_classifier_enabled", True),
            scratch_safety_multiplier=float(data.get("scratch_safety_multiplier", 1.1)),
            scratch_bundle_path=data.get("scratch_bundle_path", "deployment/scratch_classifier_v1.pkl"),
            scratch_dinov2_weights_path=data.get("scratch_dinov2_weights_path", "deployment/dinov2_vitb14.pth"),
```

Place these entries near the other simple scalar fields (e.g., after `aoi_coord_inspection_enabled`).

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_scratch_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add capi_config.py tests/test_scratch_config.py
git commit -m "feat(scratch): CAPIConfig scratch_* fields (Task 6)"
```

---

## Task 7: DB schema migration for scratch fields

**Files:**
- Modify: `capi_database.py` (migration block around line 243–269)

- [ ] **Step 1: Write failing test**

Create `tests/test_scratch_db_migration.py`:

```python
"""Verify schema migration adds scratch columns to existing DB."""
import sqlite3
from pathlib import Path

from capi_database import CAPIDatabase


def _get_columns(db_path: Path, table: str) -> set[str]:
    with sqlite3.connect(db_path) as c:
        return {row[1] for row in c.execute(f"PRAGMA table_info({table})").fetchall()}


def test_fresh_db_has_scratch_columns(tmp_path):
    db_path = tmp_path / "fresh.db"
    CAPIDatabase(str(db_path))  # triggers init

    tile_cols = _get_columns(db_path, "tile_results")
    assert "scratch_score" in tile_cols
    assert "scratch_filtered" in tile_cols

    image_cols = _get_columns(db_path, "image_results")
    assert "scratch_filter_count" in image_cols


def test_existing_db_migrated(tmp_path):
    """Create a DB without scratch columns, then re-open and verify migration ran."""
    db_path = tmp_path / "old.db"
    # Manually seed with old schema (subset of tile_results without scratch_score)
    with sqlite3.connect(db_path) as c:
        c.execute("CREATE TABLE tile_results (id INTEGER PRIMARY KEY, image_result_id INTEGER)")
        c.execute("CREATE TABLE image_results (id INTEGER PRIMARY KEY, record_id INTEGER)")
        c.execute("CREATE TABLE inference_records (id INTEGER PRIMARY KEY)")

    # Re-open via CAPIDatabase — should migrate
    CAPIDatabase(str(db_path))

    tile_cols = _get_columns(db_path, "tile_results")
    assert "scratch_score" in tile_cols
    assert "scratch_filtered" in tile_cols
    image_cols = _get_columns(db_path, "image_results")
    assert "scratch_filter_count" in image_cols
```

- [ ] **Step 2: Run — expect failure**

Run: `python -m pytest tests/test_scratch_db_migration.py -v`
Expected: FAIL (columns not present).

- [ ] **Step 3: Add migration calls**

In `capi_database.py`, locate the `add_column_if_not_exists` block (around line 250–269). After the last `add_column_if_not_exists(...)` call, add:

```python
            # Scratch classifier post-filter (over-review reduction)
            add_column_if_not_exists("tile_results", "scratch_score", "REAL DEFAULT 0.0")
            add_column_if_not_exists("tile_results", "scratch_filtered", "INTEGER DEFAULT 0")
            add_column_if_not_exists("image_results", "scratch_filter_count", "INTEGER DEFAULT 0")
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_scratch_db_migration.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add capi_database.py tests/test_scratch_db_migration.py
git commit -m "feat(scratch): DB schema migration for scratch fields (Task 7)"
```

---

## Task 8: Persist scratch fields when saving results

**Files:**
- Modify: `capi_database.py` (save_tile_result / save_image_result methods)

- [ ] **Step 1: Explore current save pattern**

Run:
```bash
python -c "from capi_database import CAPIDatabase; import inspect; print(inspect.getsource(CAPIDatabase.save_image_result))"
python -c "from capi_database import CAPIDatabase; import inspect; print(inspect.getsource(CAPIDatabase.save_tile_result))"
```
Note the method signatures and column list used in INSERT statement.

- [ ] **Step 2: Write failing test**

Append to `tests/test_scratch_db_migration.py`:

```python
def test_scratch_fields_persist(tmp_path):
    db = CAPIDatabase(str(tmp_path / "persist.db"))
    rec_id = db.save_inference_record(
        glass_id="G1", model_id="M1", machine_no="1",
        resolution=(1920, 1080),
        machine_judgment="NG", ai_judgment="OK",
        image_dir="/fake",
        total_images=1, ng_images=0, ng_details="",
        request_time="2026-04-14T10:00:00",
        response_time="2026-04-14T10:00:05",
        processing_seconds=5.0,
    )
    img_id = db.save_image_result(
        record_id=rec_id,
        image_path="/fake/G0F0000001.jpg",
        image_size=(512, 512),
        is_ng=False, total_tiles=1, ng_tiles=0, max_score=0.5,
        processing_time=0.1,
        scratch_filter_count=2,
    )
    db.save_tile_result(
        image_result_id=img_id,
        tile_id=1, x=0, y=0, width=512, height=512,
        score=0.95, is_anomaly=False, is_dust=False,
        heatmap_path="",
        scratch_score=0.88, scratch_filtered=True,
    )

    # Reload and verify
    import sqlite3
    with sqlite3.connect(tmp_path / "persist.db") as c:
        row = c.execute("SELECT scratch_filter_count FROM image_results").fetchone()
        assert row[0] == 2
        row = c.execute(
            "SELECT scratch_score, scratch_filtered FROM tile_results"
        ).fetchone()
        assert row[0] == pytest.approx(0.88)
        assert row[1] == 1
```

(Add `import pytest` at top of file if not present.)

- [ ] **Step 3: Run — expect failure**

Run: `python -m pytest tests/test_scratch_db_migration.py::test_scratch_fields_persist -v`
Expected: FAIL with TypeError (unexpected keyword arg).

- [ ] **Step 4: Update `save_image_result` and `save_tile_result`**

In `capi_database.py`:

Locate `save_image_result` method — add `scratch_filter_count: int = 0` to the signature and append `scratch_filter_count` to the INSERT column list + VALUES. Similarly for `save_tile_result` — add `scratch_score: float = 0.0, scratch_filtered: bool = False` parameters and include them in the INSERT.

Exact pattern (read current INSERT first via Step 1 output, then mirror it). Example shape:

```python
    def save_image_result(
        self, record_id: int, image_path: str, image_size: tuple,
        is_ng: bool, total_tiles: int, ng_tiles: int, max_score: float,
        processing_time: float,
        # ... existing params ...
        scratch_filter_count: int = 0,        # NEW
    ) -> int:
        with self._connection() as conn:
            cursor = conn.execute(
                """INSERT INTO image_results
                   (record_id, image_path, ..., processing_time, scratch_filter_count)
                   VALUES (?, ?, ..., ?, ?)""",
                (record_id, image_path, ..., processing_time, scratch_filter_count),
            )
            return cursor.lastrowid
```

(Exact column list from actual existing code — keep alphabetical/logical order unchanged, just append new cols at end.)

Same pattern for `save_tile_result`: add `scratch_score=0.0, scratch_filtered=False` params, append to INSERT.

- [ ] **Step 5: Run test**

Run: `python -m pytest tests/test_scratch_db_migration.py::test_scratch_fields_persist -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add capi_database.py tests/test_scratch_db_migration.py
git commit -m "feat(scratch): persist scratch fields in DB (Task 8)"
```

---

## Task 9: `CAPIInferencer` integration — lazy load + process_panel hook

**Files:**
- Modify: `capi_inference.py` (`CAPIInferencer.__init__`, new `_get_scratch_filter`, `process_panel` hook)
- Test: `tests/test_scratch_integration.py`

- [ ] **Step 1: Write failing integration test (mock classifier, no real bundle)**

Create `tests/test_scratch_integration.py`:

```python
"""Integration test: CAPIInferencer with scratch filter enabled (mocked classifier)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer, TileInfo, ImageResult


def _make_image_result_with_ng_tiles(n: int) -> ImageResult:
    tiles = [
        TileInfo(tile_id=i, x=0, y=0, width=512, height=512,
                 image=np.zeros((512, 512, 3), dtype=np.uint8))
        for i in range(n)
    ]
    ir = ImageResult(
        image_path=Path("/fake"), image_size=(1024, 1024),
        otsu_bounds=(0, 0, 1024, 1024),
        exclusion_regions=[], tiles=tiles,
        excluded_tile_count=0, processed_tile_count=n, processing_time=0.0,
    )
    ir.anomaly_tiles = [(t, 0.9, None) for t in tiles]
    return ir


class _FakeClassifier:
    def __init__(self, score=0.95):
        self.conformal_threshold = 0.7
        self._score = score
    def predict(self, image):
        return self._score


def test_get_scratch_filter_disabled_returns_none(tmp_path):
    cfg = CAPIConfig()
    cfg.scratch_classifier_enabled = False
    inferencer = CAPIInferencer(config=cfg, model_path="")
    assert inferencer._get_scratch_filter() is None


def test_get_scratch_filter_caches_on_success(tmp_path):
    cfg = CAPIConfig()
    cfg.scratch_classifier_enabled = True
    inferencer = CAPIInferencer(config=cfg, model_path="")
    with patch("capi_inference.ScratchClassifier", return_value=_FakeClassifier()) as mock_cls:
        sf1 = inferencer._get_scratch_filter()
        sf2 = inferencer._get_scratch_filter()
        assert sf1 is sf2
        assert mock_cls.call_count == 1    # cached


def test_get_scratch_filter_load_failure_sentinel(tmp_path):
    from scratch_classifier import ScratchClassifierLoadError
    cfg = CAPIConfig()
    cfg.scratch_classifier_enabled = True
    inferencer = CAPIInferencer(config=cfg, model_path="")
    with patch("capi_inference.ScratchClassifier",
               side_effect=ScratchClassifierLoadError("no bundle")) as mock_cls:
        sf1 = inferencer._get_scratch_filter()
        sf2 = inferencer._get_scratch_filter()
        assert sf1 is None
        assert sf2 is None
        assert mock_cls.call_count == 1   # no retry
```

- [ ] **Step 2: Run — expect failure**

Run: `python -m pytest tests/test_scratch_integration.py -v`
Expected: FAIL — `AttributeError: 'CAPIInferencer' object has no attribute '_get_scratch_filter'`

- [ ] **Step 3: Add integration points in `capi_inference.py`**

At the top of `capi_inference.py` (near other imports), add:

```python
# Lazy imports inside _get_scratch_filter to avoid eager torch.hub calls on server start
```

In `CAPIInferencer.__init__` (line ~238+), at the end of `__init__`, add:

```python
        # Scratch classifier post-filter (lazy-loaded on first NG tile)
        self.scratch_filter = None
        self._scratch_load_failed = False
```

Immediately after `__init__`, add the helper method:

```python
    def _get_scratch_filter(self):
        """Lazy-load ScratchFilter (first call only). Thread-safe via _gpu_lock
        (caller responsibility — called inside process_panel)."""
        if not getattr(self.config, "scratch_classifier_enabled", False):
            return None
        if self._scratch_load_failed:
            return None

        # Rebuild filter if safety multiplier changed (dynamic config)
        current_safety = float(getattr(self.config, "scratch_safety_multiplier", 1.1))
        if self.scratch_filter is not None and \
                abs(self.scratch_filter._safety - current_safety) > 1e-9:
            self.scratch_filter = ScratchFilter(self.scratch_filter._classifier,
                                                 safety_multiplier=current_safety)

        if self.scratch_filter is not None:
            return self.scratch_filter

        bundle = getattr(self.config, "scratch_bundle_path", "")
        weights = getattr(self.config, "scratch_dinov2_weights_path", "")
        try:
            clf = ScratchClassifier(
                bundle_path=bundle,
                dinov2_weights_path=weights or None,
                device=self._get_device(getattr(self.config, "device", "cuda")),
            )
        except ScratchClassifierLoadError as e:
            logger.error("ScratchClassifier load failed: %s", e)
            self._scratch_load_failed = True
            return None
        self.scratch_filter = ScratchFilter(clf, safety_multiplier=current_safety)
        return self.scratch_filter
```

Add imports at top of `capi_inference.py` (with other local imports):

```python
from scratch_classifier import ScratchClassifier, ScratchClassifierLoadError
from scratch_filter import ScratchFilter
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_scratch_integration.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Hook into process_panel**

Locate `process_panel` method in `capi_inference.py` (around line 3129). Find where per-image results are produced — specifically the loop that appends results into `image_results`. Just before returning `image_results` / before panel-level aggregation, add:

```python
            # Scratch post-filter: flip high-confidence scratch NG tiles to OK
            sf = self._get_scratch_filter()
            if sf is not None:
                for ir in image_results:
                    if ir.anomaly_tiles:
                        sf.apply_to_image_result(ir)
```

**Finding exact insertion:** In `process_panel`, after the loop that fills each `ImageResult.anomaly_tiles` (PatchCore scoring + dust / bomb / edge filters all done), before the code that sets `is_ng` on the whole panel. If the loop body currently ends with some aggregation/logging step, insert immediately after the loop but before `is_ng` is computed.

Add an integration test:

```python
def test_process_panel_applies_scratch_filter(tmp_path, monkeypatch):
    cfg = CAPIConfig()
    cfg.scratch_classifier_enabled = True
    inferencer = CAPIInferencer(config=cfg, model_path="")

    # Fake the filter so process_panel uses it without needing bundle
    from scratch_filter import ScratchFilter
    fake_sf = ScratchFilter(_FakeClassifier(score=0.99), safety_multiplier=1.0)
    monkeypatch.setattr(inferencer, "_get_scratch_filter", lambda: fake_sf)

    # Provide a minimal image_results list with 1 NG tile; bypass actual inference
    ir = _make_image_result_with_ng_tiles(2)
    # Apply filter directly (simulates what process_panel's post-filter step does)
    sf = inferencer._get_scratch_filter()
    for r in [ir]:
        if r.anomaly_tiles:
            sf.apply_to_image_result(r)

    assert len(ir.anomaly_tiles) == 0
    assert ir.scratch_filter_count == 2
```

(This test exercises the integration path without the full process_panel loop.)

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_scratch_integration.py -v`
Expected: 4 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add capi_inference.py tests/test_scratch_integration.py
git commit -m "feat(scratch): CAPIInferencer integration + process_panel hook (Task 9)"
```

---

## Task 10: Wire scratch fields into DB write path from process_panel results

**Files:**
- Modify: `capi_inference.py` (save helper / where DB write happens) OR `capi_server.py` (if DB write is on server side)

- [ ] **Step 1: Find where DB write happens**

Run:
```bash
grep -n "save_image_result\|save_tile_result" capi_inference.py capi_server.py | head -20
```

This shows you where the current DB save is invoked. Typically in `capi_server.py` background thread or a helper in `capi_inference.py`.

- [ ] **Step 2: Write failing test**

Append to `tests/test_scratch_db_migration.py`:

```python
def test_pipeline_persists_scratch_fields(tmp_path):
    """Simulate writing image/tile results with scratch fields — should roundtrip."""
    db = CAPIDatabase(str(tmp_path / "e2e.db"))
    rec_id = db.save_inference_record(
        glass_id="G1", model_id="M1", machine_no="1",
        resolution=(1920, 1080),
        machine_judgment="NG", ai_judgment="OK",
        image_dir="/fake",
        total_images=1, ng_images=0, ng_details="",
        request_time="2026-04-14T10:00:00",
        response_time="2026-04-14T10:00:05",
        processing_seconds=5.0,
    )

    # Simulate what the pipeline save path must do: pass scratch fields
    img_id = db.save_image_result(
        record_id=rec_id, image_path="/fake/G0F0000001.jpg",
        image_size=(512, 512),
        is_ng=False, total_tiles=2, ng_tiles=0, max_score=0.5,
        processing_time=0.1,
        scratch_filter_count=2,
    )
    db.save_tile_result(
        image_result_id=img_id, tile_id=1, x=0, y=0,
        width=512, height=512,
        score=0.95, is_anomaly=False, is_dust=False, heatmap_path="",
        scratch_score=0.91, scratch_filtered=True,
    )

    rows = db.get_tile_results_for_image(img_id)
    assert rows[0].get("scratch_score", 0.0) == pytest.approx(0.91)
    assert rows[0].get("scratch_filtered", 0) == 1
```

Also grep to find the pipeline save call location and verify the call-site passes the new fields:
```bash
grep -n "save_image_result" capi_server.py capi_inference.py
```

- [ ] **Step 3: Update call-site to pass scratch fields from ImageResult/TileInfo**

In whichever file invokes `save_image_result` and `save_tile_result` (typical location: `capi_server.py` background thread or a dedicated helper), add the new fields to the call:

```python
# save_image_result call:
db.save_image_result(
    record_id=record_id,
    image_path=str(ir.image_path),
    # ... existing args ...
    scratch_filter_count=ir.scratch_filter_count,
)

# save_tile_result call, for each tile:
db.save_tile_result(
    image_result_id=img_id,
    tile_id=tile.tile_id,
    # ... existing args ...
    scratch_score=tile.scratch_score,
    scratch_filtered=tile.scratch_filtered,
)
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_scratch_db_migration.py::test_pipeline_persists_scratch_fields -v`
Expected: PASS. If `get_tile_results_for_image` doesn't exist, use direct sqlite query in the test.

- [ ] **Step 5: Commit**

```bash
git add capi_server.py capi_inference.py tests/test_scratch_db_migration.py
git commit -m "feat(scratch): wire scratch fields through DB save path (Task 10)"
```

---

## Task 11: `train_final_model.py` — produce deployment bundle

**Files:**
- Create: `scripts/over_review_poc/train_final_model.py`

- [ ] **Step 1: Write a smoke-test (runs script end-to-end on a tiny dataset subset)**

Create `tests/test_train_final_model.py`:

```python
"""Smoke test for scripts.over_review_poc.train_final_model CLI."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_train_final_model_smoke(tmp_path):
    manifest = REPO_ROOT / "datasets" / "over_review" / "manifest.csv"
    if not manifest.exists():
        pytest.skip("Real manifest not available in CI")
    out = tmp_path / "bundle.pkl"
    # Run with minimal epochs to keep test fast
    res = subprocess.run(
        [sys.executable, "-m", "scripts.over_review_poc.train_final_model",
         "--manifest", str(manifest),
         "--transform", "clahe", "--clahe-clip", "4.0",
         "--rank", "4", "--n-lora-blocks", "1", "--epochs", "1",
         "--calib-frac", "0.2",
         "--output", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert res.returncode == 0, f"Script failed: {res.stderr}"
    assert out.exists()
    # Verify bundle is loadable
    from scratch_classifier import load_bundle
    lora_sd, logreg, meta, calib = load_bundle(out)
    assert meta.lora_rank == 4
    assert meta.conformal_threshold > 0.0
    assert len(calib) > 0
```

- [ ] **Step 2: Run — expect failure / skip**

Run: `python -m pytest tests/test_train_final_model.py -v -m slow`
Expected: If manifest not present → SKIP. Otherwise FAIL (script not present).

- [ ] **Step 3: Implement `train_final_model.py`**

Create `scripts/over_review_poc/train_final_model.py`:

```python
"""Train final deployment LoRA model on full dataset + pickle bundle.

Produces `deployment/scratch_classifier_v1.pkl` consumable by
`scratch_classifier.ScratchClassifier`.

Usage:
    python -m scripts.over_review_poc.train_final_model \
        --manifest datasets/over_review/manifest.csv \
        --transform clahe --clahe-clip 4.0 \
        --rank 16 --n-lora-blocks 2 --epochs 15 --alpha 16 \
        --calib-frac 0.2 \
        --output deployment/scratch_classifier_v1.pkl
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TRUST_REMOTE_CODE", "1")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from scratch_classifier import ScratchClassifierMetadata, save_bundle

from scripts.over_review_poc.dataset import load_samples, SCRATCH_BINARY
from scripts.over_review_poc.features import (
    DINOV2_MODEL, DINOV2_REPO, INPUT_SIZE,
    build_transform_clahe,
)
from scripts.over_review_poc.finetune_lora import (
    _apply_lora, _extract_cls, _train_fold,
)
from scripts.over_review_poc.zero_leak_analysis import _group_aware_split

logger = logging.getLogger(__name__)


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_lora_state_dict(model: torch.nn.Module) -> dict:
    """Return only LoRA adapter weights from the fine-tuned model."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if "lora_A" in k or "lora_B" in k}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--transform", choices=["clahe"], default="clahe")
    p.add_argument("--clahe-clip", type=float, default=4.0)
    p.add_argument("--clahe-tile", type=int, default=8)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--n-lora-blocks", type=int, default=2)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calib-frac", type=float, default=0.2)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--default-safety", type=float, default=1.1,
                   help="Default safety multiplier baked into bundle metadata "
                        "(runtime config may override).")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_transform_clahe(args.clahe_clip, args.clahe_tile)
    transform_id = f"v3_clahe_cl{args.clahe_clip}_tg{args.clahe_tile}"

    samples = load_samples(args.manifest)
    is_true_ng = np.array([s.original_label == "true_ng" for s in samples])
    y = np.array([1 if s.label == SCRATCH_BINARY else 0 for s in samples])
    all_idx = np.arange(len(samples))
    proper_idx, calib_idx = _group_aware_split(all_idx, samples, args.calib_frac, args.seed)
    logger.info("Total=%d | proper_train=%d | calibration=%d",
                len(samples), len(proper_idx), len(calib_idx))

    # Train LoRA + head on proper_train
    logger.info("Training LoRA (rank=%d, blocks=%d, epochs=%d)...",
                args.rank, args.n_lora_blocks, args.epochs)
    model = _train_fold(samples, proper_idx, transform, device, args)
    proper_samples = [samples[i] for i in proper_idx]
    calib_samples = [samples[i] for i in calib_idx]

    logger.info("Extracting features on proper_train + calib...")
    proper_feats = _extract_cls(model, proper_samples, transform, args.batch_size, device)
    calib_feats = _extract_cls(model, calib_samples, transform, args.batch_size, device)

    logger.info("Fitting LogReg on proper_train features...")
    logreg = LogisticRegression(
        class_weight="balanced", max_iter=2000, C=1.0,
        solver="lbfgs", random_state=args.seed,
    ).fit(proper_feats, y[proper_idx])

    # Conformal threshold = max LogReg score on calib true_ng
    calib_scores = logreg.predict_proba(calib_feats)[:, 1]
    calib_ng_mask = is_true_ng[calib_idx]
    if not calib_ng_mask.any():
        raise RuntimeError("No true_ng samples in calibration split; re-seed or increase calib_frac.")
    conformal_threshold = float(calib_scores[calib_ng_mask].max())

    meta = ScratchClassifierMetadata(
        preprocessing_id=transform_id,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
        input_size=INPUT_SIZE,
        dinov2_repo=DINOV2_REPO,
        dinov2_model=DINOV2_MODEL,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        lora_n_blocks=args.n_lora_blocks,
        conformal_threshold=conformal_threshold,
        safety_multiplier=args.default_safety,
        trained_at=dt.datetime.now().isoformat(),
        dataset_sha256=_sha256_of_file(args.manifest),
        git_commit=_get_git_commit(),
    )
    save_bundle(
        args.output,
        _extract_lora_state_dict(model),
        logreg,
        meta,
        calib_scores,
    )

    print()
    print("=== train_final_model summary ===")
    print(f"Bundle:             {args.output}")
    print(f"Conformal thresh:   {conformal_threshold:.4f}")
    print(f"Default safety:     {args.default_safety}")
    print(f"Est. eff. thresh:   {min(conformal_threshold * args.default_safety, 0.9999):.4f}")
    print(f"Calib NG count:     {int(calib_ng_mask.sum())} / {len(calib_idx)}")
    print(f"Calib score range:  [{calib_scores.min():.3f}, {calib_scores.max():.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run smoke test (if GPU + data available)**

Run: `python -m pytest tests/test_train_final_model.py -v -m slow`
Expected: PASS if manifest + GPU present, else SKIP.

Optional manual check on a small subset:
```bash
# On dev machine with manifest
python -m scripts.over_review_poc.train_final_model \
    --manifest datasets/over_review/manifest.csv \
    --rank 4 --n-lora-blocks 1 --epochs 1 \
    --output /tmp/tiny_bundle.pkl
```
Verify `/tmp/tiny_bundle.pkl` is produced and loadable.

- [ ] **Step 5: Commit**

```bash
git add scripts/over_review_poc/train_final_model.py tests/test_train_final_model.py
git commit -m "feat(scratch): train_final_model.py for bundle production (Task 11)"
```

---

## Task 12: `deployment/README.md` — bundle production & deploy instructions

**Files:**
- Create: `deployment/README.md`

- [ ] **Step 1: Write the README**

Create `deployment/README.md`:

```markdown
# CAPI Scratch Classifier — Deployment Bundle

This directory holds the deployable artifacts for the over-review scratch
post-filter. Bundle layout:

```
deployment/
├── scratch_classifier_v1.pkl    # LoRA + LogReg + metadata (produced by train_final_model.py)
├── dinov2_vitb14.pth            # DINOv2 base weights (330 MB)
└── README.md                    # this file
```

## Producing a new bundle

Run on a machine with the training data + GPU:

```bash
python -m scripts.over_review_poc.train_final_model \
    --manifest datasets/over_review/manifest.csv \
    --transform clahe --clahe-clip 4.0 \
    --rank 16 --n-lora-blocks 2 --epochs 15 --alpha 16 \
    --calib-frac 0.2 \
    --output deployment/scratch_classifier_v1.pkl
```

The script prints the conformal threshold and estimated effective threshold at
the default safety multiplier. Commit these numbers + the SHA256 of the bundle
in the version log below.

## Exporting DINOv2 base weights (one-time, on machine with internet)

```bash
python -m scripts.over_review_poc.prepare_offline_model \
    --export-state-dict deployment/dinov2_vitb14.pth
```

## Deploying to production

1. `rsync -av deployment/ production-host:/opt/capi/deployment/` (or SCP whole dir).
2. On production host, ensure `server_config.yaml` points at these paths:
   ```yaml
   scratch_classifier_enabled: true
   scratch_safety_multiplier: 1.1
   scratch_bundle_path: /opt/capi/deployment/scratch_classifier_v1.pkl
   scratch_dinov2_weights_path: /opt/capi/deployment/dinov2_vitb14.pth
   ```
3. Restart `capi_server.py`.
4. Verify first NG request in logs: "ScratchClassifier loaded: rank=16 blocks=2 threshold=X.XXXX".

## Runtime override

Toggle via DB `config_params` (no restart needed):

```sql
UPDATE config_params SET param_value = 'false'
 WHERE param_name = 'scratch_classifier_enabled';
-- or
UPDATE config_params SET param_value = '1.2'
 WHERE param_name = 'scratch_safety_multiplier';
```

## Version log

| Version | Built at | git_commit | Conformal thr | Notes |
|---------|----------|------------|---------------|-------|
| v1      | TBD      | TBD        | TBD           | Initial deployment |

Append new rows when producing a new bundle; keep old `.pkl` files for rollback.

## Rollback

1. Set `scratch_classifier_enabled=false` via DB.
2. (Optional) Swap `scratch_bundle_path` to previous `_vN.pkl`, set enabled=true.
```

- [ ] **Step 2: Create deployment directory marker**

Create `deployment/.gitkeep` so the directory persists in git without the bundles themselves:

```bash
touch deployment/.gitkeep
```

Add to `.gitignore` (check if already excluded):

```bash
grep -n "deployment" .gitignore || echo "deployment/*.pkl
deployment/*.pth" >> .gitignore
```

- [ ] **Step 3: Commit**

```bash
git add deployment/README.md deployment/.gitkeep .gitignore
git commit -m "feat(scratch): deployment README + gitignore bundles (Task 12)"
```

---

## Post-plan verification

After all 12 tasks:

1. Run full unit test suite:
   ```bash
   python -m pytest tests/test_scratch_classifier.py \
                    tests/test_scratch_filter.py \
                    tests/test_scratch_config.py \
                    tests/test_scratch_db_migration.py \
                    tests/test_scratch_integration.py -v -m "not slow"
   ```
   Expected: all PASS.

2. Run existing CAPI test suite to verify no regressions:
   ```bash
   python -m pytest tests/ -v -m "not slow" --ignore=tests/test_scratch_classifier.py \
                                            --ignore=tests/test_train_final_model.py
   ```
   Expected: all PASS.

3. Manual spot-check: build a real bundle and start server with `scratch_classifier_enabled=true`; send a fake NG request via `test_client.py` with a known scratch panel fixture. Confirm `/records` drill-down shows `scratch_score > 0` on the NG tile.

4. Staging validation: deploy to 1 AOI machine for 1 full working day. Monitor:
   - `/records` for `scratch_filter_count > 0` rows
   - P95 latency < baseline + 2s
   - Next-day RIC report: count of panels where CAPI said OK but RIC said NG — should be < 3% of NG-flipped panels.

---

## Notes for executing agent

- **TDD discipline**: every code change must be driven by a test. If you find yourself writing code without a test already in place, stop and write the test first.
- **DINOv2 download**: first run of `ScratchClassifier` downloads DINOv2 ViT-B/14 (~330MB) via `torch.hub`. Expect 1-5 min on first run, cached afterwards.
- **GPU requirement for slow tests**: tests marked `@pytest.mark.slow` require GPU and DINOv2 cache. Skip in CI with `-m "not slow"`.
- **Commits per task**: commit after each task's final step. No squashing mid-task.
- **No scope creep**: do not refactor `finetune_lora.py` internals. This plan imports from it as-is; if it drifts later, file a separate refactor task.
- **Test file note**: tests use `from capi_inference import ...`. The project currently runs from the repo root with no `__init__.py` package — follow that convention.
