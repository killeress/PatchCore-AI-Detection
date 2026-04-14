# Over-Review Scratch POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 DINOv2 ViT-B/14 凍結特徵 + LogReg/k-NN 的 5-fold POC pipeline，驗證 `over_surface_scratch` 是否能在不漏 `true_ng` 的硬門檻下被有效擋下。

**Architecture:** 新增 `scripts/over_review_poc/` 模組群（dataset → features → splits → evaluate → report），透過 CLI `run_poc.py` 串接。DINOv2 透過 `torch.hub` 載入，embedding 快取到 `.npz` 以便重跑。測試集中 glass_id group k-fold 避免洩漏，主指標為「realistic threshold（取 train `true_ng` 最大 score）下的 test scratch_recall」。

**Tech Stack:** Python 3.10+, PyTorch, DINOv2 (via `torch.hub`), scikit-learn, `umap-learn`, matplotlib, pytest.

**Spec reference:** `docs/superpowers/specs/2026-04-14-over-review-scratch-poc-design.md`

---

## File Structure

```
scripts/over_review_poc/
  __init__.py                     # 空 package marker
  dataset.py                      # Sample dataclass + load_samples + binary label mapping
  features.py                     # DINOv2 載入 + batch 抽特徵 + 快取
  splits.py                       # StratifiedGroupKFold 包裝 + fallback
  evaluate.py                     # FoldResult + LogReg/k-NN + threshold 邏輯 + breakdown
  report.py                       # UMAP / PR curves / markdown / JSON / missed CSV
  run_poc.py                      # CLI orchestration
  prepare_offline_model.py        # 離線部署輔助：下載並印出 DINOv2 cache 路徑

tests/
  test_over_review_poc.py         # 4 個關鍵單元測試

reports/over_review_scratch_poc/  # 輸出（gitignore）
  embeddings_cache.npz
  fold_{1..5}_umap.png
  fold_{1..5}_pr.png
  report.md
  report.json
  missed_scratch.csv
```

**Split 原則：** 一個檔一個職責；檔間只透過 public function 互相依賴；任何檔案可獨立 import 測試。

---

## Task 1: 建立專案骨架與 gitignore

**Files:**
- Create: `scripts/over_review_poc/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: 建立 package 空 marker**

Create `scripts/over_review_poc/__init__.py` with content:

```python
"""Over-review reduction POC: scratch-vs-not-scratch feasibility study."""
```

- [ ] **Step 2: 在 .gitignore 末尾加入 reports 目錄**

Append to `.gitignore`:

```
# Over-review POC outputs
reports/over_review_scratch_poc/
```

- [ ] **Step 3: 驗證 gitignore 生效**

Run: `git status`
Expected: `reports/over_review_scratch_poc/` 不應出現在 untracked files（若尚未建立該目錄，單純確認 `.gitignore` 有對應規則即可）。

- [ ] **Step 4: Commit**

```bash
git add scripts/over_review_poc/__init__.py .gitignore
git commit -m "feat(over_review_poc): 專案骨架 + gitignore reports 目錄"
```

---

## Task 2: dataset.py — Sample 與 load_samples

**Files:**
- Create: `scripts/over_review_poc/dataset.py`
- Create: `tests/test_over_review_poc.py`

- [ ] **Step 1: 先寫失敗的測試**

Create `tests/test_over_review_poc.py`:

```python
"""Over-review POC 單元測試。

執行方式:
    python tests/test_over_review_poc.py
    pytest tests/test_over_review_poc.py -v
"""
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from scripts.over_review_poc.dataset import (
    Sample,
    to_binary_label,
    load_samples,
    SCRATCH_BINARY,
    NOT_SCRATCH_BINARY,
)


# ---------- Task 2: label binary mapping ----------

def test_label_binary_mapping_scratch():
    assert to_binary_label("over_surface_scratch") == SCRATCH_BINARY


def test_label_binary_mapping_true_ng_is_negative():
    assert to_binary_label("true_ng") == NOT_SCRATCH_BINARY


def test_label_binary_mapping_other_over_types_are_negative():
    for original in [
        "over_overexposure", "over_within_spec", "over_edge_false_positive",
        "over_bubble", "over_dust_mask_incomplete", "over_surface_dirt",
        "over_aoi_ai_false_positive", "over_other",
    ]:
        assert to_binary_label(original) == NOT_SCRATCH_BINARY, f"{original} 應 map 到 not_scratch"


def test_load_samples_skips_missing_and_non_ok_status(tmp_path):
    """manifest 中 status != ok 或 crop 不存在的 row 應被 skip。"""
    crop_dir = tmp_path / "true_ng" / "R0F00000" / "crop"
    crop_dir.mkdir(parents=True)
    existing_crop = crop_dir / "ok_sample.png"
    existing_crop.write_bytes(b"fake_png_bytes")

    manifest = tmp_path / "manifest.csv"
    fields = [
        "sample_id", "collected_at", "label", "source_type", "prefix",
        "glass_id", "image_name", "inference_record_id", "image_result_id",
        "tile_idx", "edge_defect_id", "crop_path", "heatmap_path",
        "ai_score", "defect_x", "defect_y", "ric_judgment",
        "over_review_category", "over_review_note", "inference_timestamp", "status",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({
            "sample_id": "ok_sample", "label": "true_ng", "source_type": "patchcore_tile",
            "prefix": "R0F00000", "glass_id": "G1", "image_name": "R0F00000_1.tif",
            "crop_path": "true_ng/R0F00000/crop/ok_sample.png",
            "ai_score": "0.9", "defect_x": "100", "defect_y": "200",
            "status": "ok",
        })
        w.writerow({
            "sample_id": "missing_file", "label": "true_ng", "source_type": "patchcore_tile",
            "prefix": "R0F00000", "glass_id": "G1", "image_name": "R0F00000_2.tif",
            "crop_path": "true_ng/R0F00000/crop/does_not_exist.png",
            "ai_score": "0.8", "defect_x": "10", "defect_y": "20",
            "status": "ok",
        })
        w.writerow({
            "sample_id": "bad_status", "label": "true_ng", "source_type": "patchcore_tile",
            "prefix": "R0F00000", "glass_id": "G1", "image_name": "R0F00000_3.tif",
            "crop_path": "true_ng/R0F00000/crop/ok_sample.png",
            "ai_score": "0.7", "defect_x": "10", "defect_y": "20",
            "status": "error",
        })

    samples = load_samples(manifest, datasets_root=tmp_path)
    assert len(samples) == 1
    assert samples[0].sample_id == "ok_sample"
    assert samples[0].label == NOT_SCRATCH_BINARY
    assert samples[0].original_label == "true_ng"
    assert samples[0].glass_id == "G1"
    assert samples[0].prefix == "R0F00000"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python -m pytest tests/test_over_review_poc.py -v`
Expected: 4 個 test 全部 ImportError（`scripts.over_review_poc.dataset` 還沒建立）。

- [ ] **Step 3: 實作 dataset.py**

Create `scripts/over_review_poc/dataset.py`:

```python
"""Load manifest.csv → Sample records with binary scratch label."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SCRATCH_LABEL_KEY = "over_surface_scratch"
SCRATCH_BINARY = "scratch"
NOT_SCRATCH_BINARY = "not_scratch"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    crop_path: Path
    label: str              # "scratch" | "not_scratch"
    original_label: str     # 原始 9-way label (true_ng / over_*)
    glass_id: str
    prefix: str
    source_type: str        # "patchcore_tile" | "edge_defect"
    ai_score: float
    defect_x: int
    defect_y: int


def to_binary_label(original_label: str) -> str:
    """Map 9-way label to binary (scratch vs not_scratch)."""
    return SCRATCH_BINARY if original_label == SCRATCH_LABEL_KEY else NOT_SCRATCH_BINARY


def _safe_float(val: str) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _safe_int(val: str) -> int:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def load_samples(manifest_path: Path, datasets_root: Path | None = None) -> list[Sample]:
    """Load samples from manifest.csv.

    Args:
        manifest_path: manifest.csv 路徑
        datasets_root: crop_path 的解析根目錄；預設為 manifest 所在目錄

    Returns:
        List of Sample. Rows with status != "ok" 或 crop 檔不存在者被 skip（log warn）。
    """
    manifest_path = Path(manifest_path)
    if datasets_root is None:
        datasets_root = manifest_path.parent

    samples: list[Sample] = []
    skipped_missing = 0
    skipped_status = 0

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                skipped_status += 1
                continue

            crop_rel = row.get("crop_path", "")
            crop_path = datasets_root / crop_rel
            if not crop_path.exists():
                skipped_missing += 1
                continue

            original = row.get("label", "")
            samples.append(Sample(
                sample_id=row["sample_id"],
                crop_path=crop_path,
                label=to_binary_label(original),
                original_label=original,
                glass_id=row.get("glass_id", ""),
                prefix=row.get("prefix", ""),
                source_type=row.get("source_type", ""),
                ai_score=_safe_float(row.get("ai_score", "")),
                defect_x=_safe_int(row.get("defect_x", "")),
                defect_y=_safe_int(row.get("defect_y", "")),
            ))

    logger.info(
        "Loaded %d samples (skipped: %d missing files, %d non-ok status)",
        len(samples), skipped_missing, skipped_status,
    )
    return samples
```

- [ ] **Step 4: 跑測試確認通過**

Run: `python -m pytest tests/test_over_review_poc.py -v`
Expected: 4 個 test 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add scripts/over_review_poc/dataset.py tests/test_over_review_poc.py
git commit -m "feat(over_review_poc): dataset.py + Sample + binary label mapping (+tests)"
```

---

## Task 3: splits.py — Stratified Group K-fold

**Files:**
- Create: `scripts/over_review_poc/splits.py`
- Modify: `tests/test_over_review_poc.py`（新增 2 個 test）

- [ ] **Step 1: 先寫失敗的測試**

Append to `tests/test_over_review_poc.py` (在檔尾 `if __name__ == "__main__"` 之前插入)：

```python
# ---------- Task 3: splits ----------

from scripts.over_review_poc.splits import group_kfold_stratified


def _mk_sample(sid, label, glass, prefix, src):
    return Sample(
        sample_id=sid, crop_path=Path("/dummy.png"),
        label=to_binary_label(label), original_label=label,
        glass_id=glass, prefix=prefix, source_type=src,
        ai_score=0.5, defect_x=0, defect_y=0,
    )


def _build_balanced_fixture(n_glass=25, per_glass=4):
    """25 glasses × 4 tiles = 100 samples；每片玻璃 3 個 not_scratch + 1 個 scratch。"""
    samples = []
    for g in range(n_glass):
        prefix = ["G0F00000", "R0F00000", "W0F00000"][g % 3]
        src = "patchcore_tile" if g % 2 == 0 else "edge_defect"
        samples.append(_mk_sample(f"s{g}_scratch", "over_surface_scratch",
                                  f"glass{g}", prefix, src))
        for i in range(per_glass - 1):
            lbl = "true_ng" if i == 0 else "over_overexposure"
            samples.append(_mk_sample(f"s{g}_ng{i}", lbl, f"glass{g}", prefix, src))
    return samples


def test_group_kfold_no_leakage():
    """同 glass_id 不能同時出現在 train 與 test。"""
    samples = _build_balanced_fixture()
    folds = group_kfold_stratified(samples, k=5, seed=42)
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        train_glasses = {samples[i].glass_id for i in train_idx}
        test_glasses = {samples[i].glass_id for i in test_idx}
        assert train_glasses.isdisjoint(test_glasses), "glass_id 在 train 與 test 同時出現"


def test_group_kfold_stratify_balance():
    """每 fold test 至少 1 個 scratch、至少 1 個 true_ng、兩種 source_type 各有 ≥1。"""
    samples = _build_balanced_fixture()
    folds = group_kfold_stratified(samples, k=5, seed=42)
    for fold_idx, (_, test_idx) in enumerate(folds):
        n_scratch = sum(1 for i in test_idx if samples[i].label == SCRATCH_BINARY)
        n_true_ng = sum(1 for i in test_idx if samples[i].original_label == "true_ng")
        src_types = {samples[i].source_type for i in test_idx}
        assert n_scratch >= 1, f"fold {fold_idx} 沒有 scratch"
        assert n_true_ng >= 1, f"fold {fold_idx} 沒有 true_ng"
        assert "patchcore_tile" in src_types and "edge_defect" in src_types, \
            f"fold {fold_idx} 缺 source_type ({src_types})"
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python -m pytest tests/test_over_review_poc.py::test_group_kfold_no_leakage tests/test_over_review_poc.py::test_group_kfold_stratify_balance -v`
Expected: ImportError（`scripts.over_review_poc.splits` 尚未建立）。

- [ ] **Step 3: 實作 splits.py**

Create `scripts/over_review_poc/splits.py`:

```python
"""Stratified Group K-fold split for POC evaluation."""
from __future__ import annotations

import logging
from collections import Counter
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.over_review_poc.dataset import Sample

logger = logging.getLogger(__name__)


def _composite_stratify_key(samples: Sequence[Sample]) -> np.ndarray:
    """label|prefix|source_type 複合 key。"""
    return np.array([f"{s.label}|{s.prefix}|{s.source_type}" for s in samples])


def _coarse_stratify_key(samples: Sequence[Sample]) -> np.ndarray:
    """Fallback：label|source_type（捨棄 prefix）。"""
    return np.array([f"{s.label}|{s.source_type}" for s in samples])


def group_kfold_stratified(
    samples: Sequence[Sample],
    k: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield k folds; each fold is (train_idx, test_idx) ndarray pair.

    - Group by glass_id（同玻璃 tiles 必在同 fold，防洩漏）
    - Stratify by composite key (label, prefix, source_type)
    - 若 composite 某 bin < k，退到 (label, source_type) fallback
    - 若 fallback 仍不夠，raise ValueError
    """
    groups = np.array([s.glass_id for s in samples])
    y = _composite_stratify_key(samples)

    bin_sizes = Counter(y)
    min_bin = min(bin_sizes.values())
    if min_bin < k:
        logger.warning(
            "Composite stratify bin min=%d < k=%d; falling back to (label, source_type)",
            min_bin, k,
        )
        y = _coarse_stratify_key(samples)
        bin_sizes = Counter(y)
        min_bin = min(bin_sizes.values())
        if min_bin < k:
            raise ValueError(
                f"Fallback stratify still has bin smaller than k "
                f"(min_bin={min_bin}, k={k}). Reduce k or rethink stratify. "
                f"Bin sizes: {dict(bin_sizes)}"
            )

    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in sgkf.split(np.zeros(len(samples)), y, groups=groups):
        folds.append((np.asarray(train_idx), np.asarray(test_idx)))
    return folds
```

- [ ] **Step 4: 跑測試確認通過**

Run: `python -m pytest tests/test_over_review_poc.py -v`
Expected: 6 個 test 全部 PASS（Task 2 的 4 個 + Task 3 的 2 個）。

- [ ] **Step 5: Commit**

```bash
git add scripts/over_review_poc/splits.py tests/test_over_review_poc.py
git commit -m "feat(over_review_poc): splits.py StratifiedGroupKFold + fallback (+tests)"
```

---

## Task 4: features.py — DINOv2 特徵抽取與快取

**Files:**
- Create: `scripts/over_review_poc/features.py`

（本 task 不寫單元測試——DINOv2 推論慢且靠 pretrained 保證；改在 Task 9 end-to-end 驗證。）

- [ ] **Step 1: 實作 features.py**

Create `scripts/over_review_poc/features.py`:

```python
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
```

- [ ] **Step 2: 基本 import smoke check**

Run: `python -c "from scripts.over_review_poc.features import load_dinov2, build_transform, get_or_extract; print('ok')"`
Expected: `ok`（不實際載 DINOv2，只驗 syntax / imports）。

- [ ] **Step 3: Commit**

```bash
git add scripts/over_review_poc/features.py
git commit -m "feat(over_review_poc): features.py DINOv2 抽特徵 + fingerprint 快取 + OOM fallback"
```

---

## Task 5: evaluate.py — LogReg/k-NN、threshold、FoldResult

**Files:**
- Create: `scripts/over_review_poc/evaluate.py`
- Modify: `tests/test_over_review_poc.py`（新增 3 個 test）

- [ ] **Step 1: 先寫失敗的測試**

Append to `tests/test_over_review_poc.py`:

```python
# ---------- Task 5: evaluate ----------

from scripts.over_review_poc.evaluate import find_threshold_at_full_recall, _find_threshold


def test_find_threshold_separable_perfect_recall():
    """全分離：所有 scratch score > 所有 true_ng score → scratch_recall 應 = 1.0"""
    train_scores = np.array([0.1, 0.2, 0.15])      # 全 true_ng
    train_is_true_ng = np.array([True, True, True])
    test_scores = np.array([0.05, 0.9, 0.8, 0.1])  # 2 scratch, 2 true_ng
    test_is_true_ng = np.array([True, False, False, True])
    test_is_scratch = np.array([False, True, True, False])

    out = find_threshold_at_full_recall(
        train_scores, train_is_true_ng, test_scores, test_is_true_ng, test_is_scratch,
    )
    assert out["realistic_threshold"] == pytest.approx(0.2)
    assert out["realistic_scratch_recall"] == pytest.approx(1.0)
    assert out["realistic_true_ng_recall"] == pytest.approx(1.0)


def test_find_threshold_complete_overlap_zero_recall():
    """完全重疊：所有 scratch score ≤ 某 true_ng → scratch_recall 應 = 0.0"""
    train_scores = np.array([0.9, 0.5, 0.7])
    train_is_true_ng = np.array([True, True, True])
    test_scores = np.array([0.6, 0.5, 0.4, 0.3])
    test_is_true_ng = np.array([True, False, False, False])
    test_is_scratch = np.array([False, True, True, True])

    out = find_threshold_at_full_recall(
        train_scores, train_is_true_ng, test_scores, test_is_true_ng, test_is_scratch,
    )
    assert out["realistic_threshold"] == pytest.approx(0.9)
    assert out["realistic_scratch_recall"] == pytest.approx(0.0)


def test_find_threshold_partial_overlap():
    """部分重疊：realistic threshold = max(train true_ng) = 0.4；test 中 score > 0.4 的 scratch 才被擋下。"""
    train_scores = np.array([0.1, 0.4, 0.2])
    train_is_true_ng = np.array([True, True, True])
    test_scores = np.array([0.35, 0.9, 0.5, 0.2, 0.6])
    test_is_true_ng = np.array([True, False, False, True, False])
    test_is_scratch = np.array([False, True, True, False, True])

    out = find_threshold_at_full_recall(
        train_scores, train_is_true_ng, test_scores, test_is_true_ng, test_is_scratch,
    )
    assert out["realistic_threshold"] == pytest.approx(0.4)
    # test scratch scores: [0.9, 0.5, 0.6] → 三個都 > 0.4 → recall = 1.0
    assert out["realistic_scratch_recall"] == pytest.approx(1.0)
    # test true_ng scores: [0.35, 0.2] → 都 < 0.4 → true_ng_recall = 1.0
    assert out["realistic_true_ng_recall"] == pytest.approx(1.0)
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `python -m pytest tests/test_over_review_poc.py -v`
Expected: 新增 3 個 test ImportError（`scripts.over_review_poc.evaluate` 尚未建立）。

- [ ] **Step 3: 實作 evaluate.py**

Create `scripts/over_review_poc/evaluate.py`:

```python
"""Fold-level training (LogReg + k-NN), threshold search, breakdown metrics."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neighbors import KNeighborsClassifier

from scripts.over_review_poc.dataset import Sample, SCRATCH_BINARY

logger = logging.getLogger(__name__)


@dataclass
class ClassifierResult:
    classifier_name: str
    test_scores: np.ndarray
    realistic_threshold: float
    realistic_scratch_recall: float
    realistic_true_ng_recall: float
    oracle_threshold: float
    oracle_scratch_recall: float
    pr_auc: float
    missed_scratch_test_idx: list[int] = field(default_factory=list)


@dataclass
class FoldResult:
    fold_idx: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    logreg: ClassifierResult
    knn: ClassifierResult
    per_prefix: dict
    per_source_type: dict


def _find_threshold(scores: np.ndarray, is_true_ng: np.ndarray) -> float:
    """Max score among true_ng samples. Any sample with score > this is 'safe to flip'."""
    tn = scores[is_true_ng]
    if len(tn) == 0:
        return float("-inf")
    return float(tn.max())


def _recall_above(scores: np.ndarray, mask: np.ndarray, threshold: float) -> float:
    sel = scores[mask]
    if len(sel) == 0:
        return 0.0
    return float((sel > threshold).mean())


def find_threshold_at_full_recall(
    train_scores: np.ndarray,
    train_is_true_ng: np.ndarray,
    test_scores: np.ndarray,
    test_is_true_ng: np.ndarray,
    test_is_scratch: np.ndarray,
) -> dict:
    """Realistic + Oracle threshold metrics.

    Realistic: threshold = max train_true_ng score → apply to test
    Oracle: threshold = max test_true_ng score → by definition test_true_ng_recall=100%
    """
    realistic_thr = _find_threshold(train_scores, train_is_true_ng)
    oracle_thr = _find_threshold(test_scores, test_is_true_ng)

    realistic_scratch_recall = _recall_above(test_scores, test_is_scratch, realistic_thr)
    # true_ng 在 realistic 下的 recall = 1 - 被誤判 flip OK 的比例
    #   flipped-to-ok 比例 = test true_ng 中 score > threshold 的比例
    realistic_true_ng_miss_rate = _recall_above(test_scores, test_is_true_ng, realistic_thr)
    realistic_true_ng_recall = 1.0 - realistic_true_ng_miss_rate

    oracle_scratch_recall = _recall_above(test_scores, test_is_scratch, oracle_thr)

    return {
        "realistic_threshold": realistic_thr,
        "realistic_scratch_recall": realistic_scratch_recall,
        "realistic_true_ng_recall": realistic_true_ng_recall,
        "oracle_threshold": oracle_thr,
        "oracle_scratch_recall": oracle_scratch_recall,
    }


def _breakdown_by_attr(
    samples: Sequence[Sample],
    test_idx: np.ndarray,
    result: ClassifierResult,
    attr: str,
) -> dict:
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"flagged": 0, "total": 0})
    for local_i, sample_i in enumerate(test_idx):
        s = samples[sample_i]
        if s.label != SCRATCH_BINARY:
            continue
        key = getattr(s, attr)
        groups[key]["total"] += 1
        if result.test_scores[local_i] > result.realistic_threshold:
            groups[key]["flagged"] += 1
    out = {}
    for key, counts in groups.items():
        total = counts["total"]
        out[key] = {
            "scratch_recall": counts["flagged"] / total if total > 0 else 0.0,
            "n_scratch": total,
        }
    return out


def run_fold(
    fold_idx: int,
    embeddings: np.ndarray,
    samples: Sequence[Sample],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
) -> FoldResult:
    """Train LogReg + k-NN on fold; return FoldResult with full metrics and breakdown."""
    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]
    y_train = np.array([1 if samples[i].label == SCRATCH_BINARY else 0 for i in train_idx])
    y_test = np.array([1 if samples[i].label == SCRATCH_BINARY else 0 for i in test_idx])

    train_is_true_ng = np.array([samples[i].original_label == "true_ng" for i in train_idx])
    test_is_true_ng = np.array([samples[i].original_label == "true_ng" for i in test_idx])
    test_is_scratch = y_test.astype(bool)

    # ---- LogReg ----
    logreg = LogisticRegression(
        class_weight="balanced", max_iter=2000, C=1.0, solver="lbfgs", random_state=seed,
    ).fit(X_train, y_train)
    logreg_train_scores = logreg.predict_proba(X_train)[:, 1]
    logreg_test_scores = logreg.predict_proba(X_test)[:, 1]

    # ---- k-NN ----
    knn = KNeighborsClassifier(n_neighbors=7, metric="cosine").fit(X_train, y_train)
    knn_train_scores = knn.predict_proba(X_train)[:, 1]
    knn_test_scores = knn.predict_proba(X_test)[:, 1]

    def _make_result(name: str, train_sc: np.ndarray, test_sc: np.ndarray) -> ClassifierResult:
        thr = find_threshold_at_full_recall(
            train_sc, train_is_true_ng, test_sc, test_is_true_ng, test_is_scratch,
        )
        missed = [
            int(i) for i, s in enumerate(test_sc)
            if test_is_scratch[i] and s <= thr["realistic_threshold"]
        ]
        pr_auc = (float(average_precision_score(y_test, test_sc))
                  if y_test.sum() > 0 else 0.0)
        return ClassifierResult(
            classifier_name=name,
            test_scores=test_sc,
            realistic_threshold=thr["realistic_threshold"],
            realistic_scratch_recall=thr["realistic_scratch_recall"],
            realistic_true_ng_recall=thr["realistic_true_ng_recall"],
            oracle_threshold=thr["oracle_threshold"],
            oracle_scratch_recall=thr["oracle_scratch_recall"],
            pr_auc=pr_auc,
            missed_scratch_test_idx=missed,
        )

    logreg_result = _make_result("logreg", logreg_train_scores, logreg_test_scores)
    knn_result = _make_result("knn", knn_train_scores, knn_test_scores)

    # ---- breakdown (基於 LogReg，因為它是主指標) ----
    per_prefix = _breakdown_by_attr(samples, test_idx, logreg_result, "prefix")
    per_source_type = _breakdown_by_attr(samples, test_idx, logreg_result, "source_type")

    logger.info(
        "Fold %d: LogReg realistic scratch_recall=%.3f (true_ng_recall=%.3f) | "
        "k-NN scratch_recall=%.3f",
        fold_idx, logreg_result.realistic_scratch_recall,
        logreg_result.realistic_true_ng_recall,
        knn_result.realistic_scratch_recall,
    )

    return FoldResult(
        fold_idx=fold_idx, train_idx=train_idx, test_idx=test_idx,
        logreg=logreg_result, knn=knn_result,
        per_prefix=per_prefix, per_source_type=per_source_type,
    )
```

- [ ] **Step 4: 跑測試確認通過**

Run: `python -m pytest tests/test_over_review_poc.py -v`
Expected: 9 個 test 全部 PASS（Task 2/3/5 累計）。

- [ ] **Step 5: Commit**

```bash
git add scripts/over_review_poc/evaluate.py tests/test_over_review_poc.py
git commit -m "feat(over_review_poc): evaluate.py LogReg/k-NN + realistic/oracle threshold + breakdown (+tests)"
```

---

## Task 6: report.py — UMAP、PR、markdown/JSON/CSV 輸出

**Files:**
- Create: `scripts/over_review_poc/report.py`

（本 task 無單元測試——純視覺化與 I/O；在 Task 9 end-to-end 驗證產物正確。）

- [ ] **Step 1: 實作 report.py**

Create `scripts/over_review_poc/report.py`:

```python
"""Report aggregation: UMAP, PR curves, markdown + JSON + missed CSV."""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts.over_review_poc.dataset import Sample, SCRATCH_BINARY
from scripts.over_review_poc.evaluate import FoldResult

logger = logging.getLogger(__name__)


def plot_umap(
    fold_idx: int,
    embeddings: np.ndarray,
    samples: Sequence[Sample],
    out_path: Path,
    seed: int = 42,
) -> None:
    """2D UMAP scatter: scratch=red, true_ng=blue, other_over=gray."""
    import matplotlib.pyplot as plt
    import umap

    reducer = umap.UMAP(n_components=2, random_state=seed)
    emb2d = reducer.fit_transform(embeddings)

    colors: list[str] = []
    for s in samples:
        if s.label == SCRATCH_BINARY:
            colors.append("red")
        elif s.original_label == "true_ng":
            colors.append("blue")
        else:
            colors.append("gray")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb2d[:, 0], emb2d[:, 1], c=colors, s=8, alpha=0.6)
    ax.set_title(f"Fold {fold_idx} UMAP (red=scratch, blue=true_ng, gray=other_over)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_pr(
    fold: FoldResult,
    samples: Sequence[Sample],
    out_path: Path,
) -> None:
    """PR curve (LogReg + k-NN 疊圖) for this fold."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    y_test = np.array([1 if samples[i].label == SCRATCH_BINARY else 0 for i in fold.test_idx])

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, result in [("LogReg", fold.logreg), ("k-NN", fold.knn)]:
        p, r, _ = precision_recall_curve(y_test, result.test_scores)
        ax.plot(r, p, label=f"{name} (AUC={result.pr_auc:.3f})")
    ax.set_xlabel("Recall (scratch)")
    ax.set_ylabel("Precision (scratch)")
    ax.set_title(f"Fold {fold.fold_idx} PR Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _aggregate_classifier_metrics(
    folds: list[FoldResult],
    classifier_key: str,
) -> dict:
    """mean ± std across folds for the named classifier."""
    vals = {
        "realistic_scratch_recall": [],
        "realistic_true_ng_recall": [],
        "oracle_scratch_recall": [],
        "pr_auc": [],
    }
    for f in folds:
        cr = getattr(f, classifier_key)
        vals["realistic_scratch_recall"].append(cr.realistic_scratch_recall)
        vals["realistic_true_ng_recall"].append(cr.realistic_true_ng_recall)
        vals["oracle_scratch_recall"].append(cr.oracle_scratch_recall)
        vals["pr_auc"].append(cr.pr_auc)
    return {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in vals.items()
    }


def _aggregate_breakdown(folds: list[FoldResult], key: str) -> dict:
    """合併各 fold 的 per-prefix / per-source_type breakdown。"""
    pooled: dict[str, dict[str, int]] = defaultdict(lambda: {"flagged": 0, "total": 0})
    for f in folds:
        bd = getattr(f, key)
        for attr, counts in bd.items():
            total = counts["n_scratch"]
            flagged = int(round(counts["scratch_recall"] * total))
            pooled[attr]["total"] += total
            pooled[attr]["flagged"] += flagged
    out = {}
    for attr, c in pooled.items():
        t = c["total"]
        out[attr] = {
            "scratch_recall": c["flagged"] / t if t > 0 else 0.0,
            "n_scratch": t,
        }
    return out


def _verdict(mean_scratch_recall: float) -> str:
    if mean_scratch_recall >= 0.50:
        return "GO"
    if mean_scratch_recall >= 0.30:
        return "部分可行"
    return "NO-GO"


def _write_markdown(
    agg: dict,
    per_prefix: dict,
    per_source_type: dict,
    missed_count: int,
    metadata: dict,
    out_path: Path,
) -> None:
    lr = agg["logreg"]
    kn = agg["knn"]
    verdict = _verdict(lr["realistic_scratch_recall"]["mean"])
    warn = ""
    if lr["realistic_true_ng_recall"]["mean"] < 0.95:
        warn = (f"\n> **⚠️ 警示**: Realistic true_ng_recall 平均 "
                f"{lr['realistic_true_ng_recall']['mean']:.3f} < 0.95\n")

    lines = [
        f"# Over-Review POC: Surface Scratch Classifier",
        f"Date: {metadata.get('run_at', 'N/A')} | "
        f"Model: {metadata.get('dinov2_model', 'N/A')} | "
        f"Folds: {metadata.get('n_folds', 'N/A')}",
        "",
        "## Verdict",
        f"- Realistic scratch_recall (LogReg): "
        f"**{lr['realistic_scratch_recall']['mean']:.1%} ± "
        f"{lr['realistic_scratch_recall']['std']:.1%}** → **{verdict}**",
        f"- Realistic true_ng_recall (LogReg): "
        f"{lr['realistic_true_ng_recall']['mean']:.1%} ± "
        f"{lr['realistic_true_ng_recall']['std']:.1%}",
        warn,
        "## Main Metrics",
        "",
        "| Metric | LogReg | k-NN |",
        "|---|---|---|",
        f"| Realistic scratch_recall | {lr['realistic_scratch_recall']['mean']:.3f} ± "
        f"{lr['realistic_scratch_recall']['std']:.3f} | "
        f"{kn['realistic_scratch_recall']['mean']:.3f} ± "
        f"{kn['realistic_scratch_recall']['std']:.3f} |",
        f"| Realistic true_ng_recall | {lr['realistic_true_ng_recall']['mean']:.3f} ± "
        f"{lr['realistic_true_ng_recall']['std']:.3f} | "
        f"{kn['realistic_true_ng_recall']['mean']:.3f} ± "
        f"{kn['realistic_true_ng_recall']['std']:.3f} |",
        f"| Oracle scratch_recall | {lr['oracle_scratch_recall']['mean']:.3f} ± "
        f"{lr['oracle_scratch_recall']['std']:.3f} | "
        f"{kn['oracle_scratch_recall']['mean']:.3f} ± "
        f"{kn['oracle_scratch_recall']['std']:.3f} |",
        f"| PR-AUC | {lr['pr_auc']['mean']:.3f} ± {lr['pr_auc']['std']:.3f} | "
        f"{kn['pr_auc']['mean']:.3f} ± {kn['pr_auc']['std']:.3f} |",
        "",
        "## Per-Prefix Breakdown (LogReg, pooled over folds)",
        "",
        "| Prefix | scratch_recall | n_scratch |",
        "|---|---|---|",
    ]
    for prefix in sorted(per_prefix.keys()):
        v = per_prefix[prefix]
        lines.append(f"| {prefix} | {v['scratch_recall']:.3f} | {v['n_scratch']} |")

    lines += [
        "",
        "## Per-Source-Type Breakdown (LogReg, pooled over folds)",
        "",
        "| Source Type | scratch_recall | n_scratch |",
        "|---|---|---|",
    ]
    for src in sorted(per_source_type.keys()):
        v = per_source_type[src]
        lines.append(f"| {src} | {v['scratch_recall']:.3f} | {v['n_scratch']} |")

    lines += [
        "",
        f"## Missed Scratch Samples",
        f"總計 {missed_count} 筆 scratch 未被 LogReg realistic threshold 擋下；"
        f"詳見 `missed_scratch.csv`。",
        "",
        "## Appendix",
        "- `fold_{1..5}_umap.png`：DINOv2 embedding UMAP 2D 視覺化",
        "- `fold_{1..5}_pr.png`：各 fold 的 PR curve（LogReg + k-NN）",
        "",
        "## Metadata",
        "```json",
        json.dumps(metadata, indent=2, ensure_ascii=False),
        "```",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_missed_csv(
    folds: list[FoldResult],
    samples: Sequence[Sample],
    out_path: Path,
) -> int:
    """Export missed scratch samples (LogReg realistic threshold 沒擋下的) to CSV."""
    rows = []
    for f in folds:
        for local_i in f.logreg.missed_scratch_test_idx:
            sample_i = f.test_idx[local_i]
            s = samples[sample_i]
            rows.append({
                "fold_idx": f.fold_idx,
                "sample_id": s.sample_id,
                "glass_id": s.glass_id,
                "prefix": s.prefix,
                "source_type": s.source_type,
                "logreg_score": float(f.logreg.test_scores[local_i]),
                "realistic_threshold": float(f.logreg.realistic_threshold),
                "crop_path": str(s.crop_path),
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        if rows:
            w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        else:
            fp.write("fold_idx,sample_id,glass_id,prefix,source_type,"
                     "logreg_score,realistic_threshold,crop_path\n")
    return len(rows)


def aggregate(
    folds: list[FoldResult],
    samples: Sequence[Sample],
    embeddings: np.ndarray,
    out_dir: Path,
    metadata: dict,
) -> None:
    """Full report pipeline: UMAP + PR per fold → markdown + JSON + missed CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in folds:
        # Per-fold UMAP 用該 fold test set 的 embeddings
        test_embs = embeddings[f.test_idx]
        test_samples = [samples[i] for i in f.test_idx]
        plot_umap(f.fold_idx, test_embs, test_samples,
                  out_dir / f"fold_{f.fold_idx}_umap.png")
        plot_pr(f, samples, out_dir / f"fold_{f.fold_idx}_pr.png")

    agg = {
        "logreg": _aggregate_classifier_metrics(folds, "logreg"),
        "knn": _aggregate_classifier_metrics(folds, "knn"),
    }
    per_prefix = _aggregate_breakdown(folds, "per_prefix")
    per_source_type = _aggregate_breakdown(folds, "per_source_type")

    missed_count = _write_missed_csv(folds, samples, out_dir / "missed_scratch.csv")

    _write_markdown(agg, per_prefix, per_source_type,
                    missed_count, metadata, out_dir / "report.md")

    json_payload = {
        "metadata": metadata,
        "aggregate": agg,
        "per_prefix": per_prefix,
        "per_source_type": per_source_type,
        "missed_count": missed_count,
        "per_fold": [
            {
                "fold_idx": f.fold_idx,
                "n_train": int(len(f.train_idx)),
                "n_test": int(len(f.test_idx)),
                "logreg": {
                    "realistic_threshold": f.logreg.realistic_threshold,
                    "realistic_scratch_recall": f.logreg.realistic_scratch_recall,
                    "realistic_true_ng_recall": f.logreg.realistic_true_ng_recall,
                    "oracle_scratch_recall": f.logreg.oracle_scratch_recall,
                    "pr_auc": f.logreg.pr_auc,
                },
                "knn": {
                    "realistic_threshold": f.knn.realistic_threshold,
                    "realistic_scratch_recall": f.knn.realistic_scratch_recall,
                    "realistic_true_ng_recall": f.knn.realistic_true_ng_recall,
                    "oracle_scratch_recall": f.knn.oracle_scratch_recall,
                    "pr_auc": f.knn.pr_auc,
                },
            }
            for f in folds
        ],
    }
    (out_dir / "report.json").write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("Report written to %s", out_dir)
```

- [ ] **Step 2: 基本 import smoke check**

Run: `python -c "from scripts.over_review_poc.report import aggregate, plot_umap, plot_pr; print('ok')"`
Expected: `ok`（不實際畫圖，只驗 imports；若 `umap` 或 `matplotlib` 缺失會在此報錯）。

- [ ] **Step 3: Commit**

```bash
git add scripts/over_review_poc/report.py
git commit -m "feat(over_review_poc): report.py UMAP/PR/markdown/JSON/missed CSV 輸出"
```

---

## Task 7: run_poc.py — CLI 串接

**Files:**
- Create: `scripts/over_review_poc/run_poc.py`

- [ ] **Step 1: 實作 run_poc.py**

Create `scripts/over_review_poc/run_poc.py`:

```python
"""Over-Review Scratch POC CLI entry.

Usage:
    python -m scripts.over_review_poc.run_poc \
        --manifest datasets/over_review/manifest.csv \
        --output reports/over_review_scratch_poc
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from scripts.over_review_poc.dataset import load_samples, SCRATCH_BINARY
from scripts.over_review_poc.features import DINOV2_MODEL, get_or_extract
from scripts.over_review_poc.splits import group_kfold_stratified
from scripts.over_review_poc.evaluate import run_fold
from scripts.over_review_poc.report import aggregate

logger = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _check_optional_deps() -> None:
    """Fail fast if umap-learn / matplotlib missing, with actionable message."""
    missing = []
    try:
        import umap  # noqa: F401
    except ImportError:
        missing.append(("umap-learn", "pip install umap-learn"))
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append(("matplotlib", "pip install matplotlib"))
    if missing:
        lines = ["Missing optional dependencies:"]
        for pkg, cmd in missing:
            lines.append(f"  - {pkg}  (install with: {cmd})")
        raise SystemExit("\n".join(lines))


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "not_installed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Over-Review Scratch POC")
    parser.add_argument("--manifest", required=True, type=Path,
                        help="datasets/over_review/manifest.csv 路徑")
    parser.add_argument("--output", required=True, type=Path,
                        help="輸出報告目錄（reports/over_review_scratch_poc/）")
    parser.add_argument("--k", type=int, default=5, help="k-fold k 值")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="選填：本地 DINOv2 .pth，預設走 torch.hub")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _check_optional_deps()
    _set_seeds(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    # ---- Load ----
    logger.info("Loading samples from %s", args.manifest)
    samples = load_samples(args.manifest)
    n_scratch = sum(1 for s in samples if s.label == SCRATCH_BINARY)
    n_true_ng = sum(1 for s in samples if s.original_label == "true_ng")
    logger.info("Total samples: %d (scratch=%d, true_ng=%d, other_over=%d)",
                len(samples), n_scratch, n_true_ng,
                len(samples) - n_scratch - n_true_ng)

    if n_scratch < args.k:
        raise ValueError(f"scratch 樣本 ({n_scratch}) 不足以做 {args.k}-fold")

    # ---- Features ----
    cache_path = args.output / "embeddings_cache.npz"
    embeddings = get_or_extract(samples, cache_path, checkpoint_path=args.checkpoint,
                                batch_size=args.batch_size)
    assert embeddings.shape[0] == len(samples), "embedding count mismatch"

    # ---- Splits ----
    folds_idx = group_kfold_stratified(samples, k=args.k, seed=args.seed)

    # ---- Train / Evaluate per fold ----
    fold_results = []
    for i, (train_idx, test_idx) in enumerate(folds_idx, start=1):
        fr = run_fold(i, embeddings, samples, train_idx, test_idx, seed=args.seed)
        fold_results.append(fr)

    # ---- Aggregate / Report ----
    metadata = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "manifest_path": str(args.manifest),
        "manifest_sha256": _file_sha256(args.manifest),
        "n_samples": len(samples),
        "n_scratch": n_scratch,
        "n_true_ng": n_true_ng,
        "n_folds": args.k,
        "seed": args.seed,
        "dinov2_model": DINOV2_MODEL,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": platform.node(),
        "packages": {
            "torch": _pkg_version("torch"),
            "sklearn": _pkg_version("sklearn"),
            "umap": _pkg_version("umap"),
            "numpy": _pkg_version("numpy"),
        },
    }
    aggregate(fold_results, samples, embeddings, args.output, metadata)

    logger.info("POC complete. Report at %s/report.md", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: CLI help smoke check**

Run: `python -m scripts.over_review_poc.run_poc --help`
Expected: argparse 印出 usage / arg list（所有 import 正常）。

- [ ] **Step 3: Commit**

```bash
git add scripts/over_review_poc/run_poc.py
git commit -m "feat(over_review_poc): run_poc.py CLI 串接 dataset→features→splits→evaluate→report"
```

---

## Task 8: prepare_offline_model.py — 離線部署輔助

**Files:**
- Create: `scripts/over_review_poc/prepare_offline_model.py`

- [ ] **Step 1: 實作 prepare_offline_model.py**

Create `scripts/over_review_poc/prepare_offline_model.py`:

```python
"""Helper：在有外網的機器執行，預先下載 DINOv2 並印出 cache 路徑 / state_dict 檔。

用途：產線 Linux 無外網 → 在此機器執行 → 把印出的檔案搬到產線相同路徑
(或指定 --export-state-dict <path>，產出可搬運的 .pth 檔)。

Usage:
    python -m scripts.over_review_poc.prepare_offline_model
    python -m scripts.over_review_poc.prepare_offline_model --export-state-dict /tmp/dinov2_vitb14.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.over_review_poc.features import DINOV2_MODEL, DINOV2_REPO


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare DINOv2 for offline deployment")
    parser.add_argument("--export-state-dict", type=Path, default=None,
                        help="選填：把 state_dict 另存為 .pth（方便搬運）")
    args = parser.parse_args(argv)

    print(f"Loading {DINOV2_MODEL} via torch.hub (this will download on first run)...")
    model = torch.hub.load(DINOV2_REPO, DINOV2_MODEL, source="github")
    model.eval()

    hub_dir = torch.hub.get_dir()
    print("=" * 70)
    print(f"DINOv2 cache root: {hub_dir}")
    print("Contents to copy to offline machine (same path):")
    for entry in sorted(Path(hub_dir).rglob("*")):
        if entry.is_file():
            print(f"  {entry}")
    print("=" * 70)

    if args.export_state_dict is not None:
        args.export_state_dict.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.export_state_dict)
        print(f"State dict exported to: {args.export_state_dict}")
        print(f"On offline machine: pass --checkpoint {args.export_state_dict} to run_poc.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: CLI help smoke check**

Run: `python -m scripts.over_review_poc.prepare_offline_model --help`
Expected: argparse 印出 usage。

- [ ] **Step 3: Commit**

```bash
git add scripts/over_review_poc/prepare_offline_model.py
git commit -m "feat(over_review_poc): prepare_offline_model.py 離線部署輔助 script"
```

---

## Task 9: End-to-end 執行與報告檢視

**Files:**
- Read: `reports/over_review_scratch_poc/report.md`（執行後產出）

（本 task 無程式改動；目的是實際跑完整 pipeline、驗證產物正確、記錄首輪結果。）

- [ ] **Step 1: 執行 POC（會自動下載 DINOv2 到 `~/.cache/torch/hub/`，首次約 400MB）**

Run:
```bash
python -m scripts.over_review_poc.run_poc \
    --manifest datasets/over_review/manifest.csv \
    --output reports/over_review_scratch_poc \
    --log-level INFO
```

Expected:
- log 看到：sample count、embedding 抽取進度（OOM fallback 若發生）、5 個 fold metrics、最後 `Report written to ...`
- 無 exception

- [ ] **Step 2: 驗證產物齊全**

Run:
```bash
ls reports/over_review_scratch_poc/
```
Expected 檔案：
```
embeddings_cache.npz
fold_1_pr.png  fold_1_umap.png
fold_2_pr.png  fold_2_umap.png
fold_3_pr.png  fold_3_umap.png
fold_4_pr.png  fold_4_umap.png
fold_5_pr.png  fold_5_umap.png
missed_scratch.csv
report.json
report.md
```

- [ ] **Step 3: 檢視 report.md**

Run: `cat reports/over_review_scratch_poc/report.md`（或在編輯器開啟）
確認：
- `## Verdict` 區塊有明確 LogReg scratch_recall mean ± std 與 Go / 部分可行 / NO-GO 結論
- Main Metrics 表四列（scratch_recall / true_ng_recall / oracle / PR-AUC）均填值
- Per-Prefix / Per-Source-Type Breakdown 各 prefix / source_type 有列
- 若 Realistic true_ng_recall < 95%，頂端有 ⚠️ 警示

- [ ] **Step 4: Smoke test：重跑一次驗證 cache hit**

Run:
```bash
python -m scripts.over_review_poc.run_poc \
    --manifest datasets/over_review/manifest.csv \
    --output reports/over_review_scratch_poc \
    --log-level INFO
```
Expected: log 看到 `Loaded embeddings from cache: ...`（應在 <5 秒內跳過 DINOv2 推論）。

- [ ] **Step 5: 決定下一步（無論結果都 commit 當前狀態）**

根據 report verdict：
- **GO** → 另開 deployment integration spec + `over_overexposure` POC 延伸 spec
- **部分可行** → 討論是否升 ViT-L/14 / 加 TTA / 換 patch mean pool
- **NO-GO** → 討論改走 CV 規則路線或重新 formulate task

Commit 最終狀態（即便結果出爐，只 commit code / 可能的 readme 更新；`reports/` 不進 git）：

```bash
git status
# 若有任何 code 修復（例如 end-to-end 跑出來才發現的小 bug）
git add -p
git commit -m "fix(over_review_poc): <具體修復描述>"
```

若整個 pipeline 一次過且無需改動，則此 step 留白，流程到此結束。
