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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
