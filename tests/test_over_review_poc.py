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


# ---------- Task A-followup: preprocess_crop (panel mask + aspect-preserve resize) ----------

from PIL import Image as _PILImage

from scripts.over_review_poc.features import (
    preprocess_crop,
    _find_panel_bbox,
    INPUT_SIZE,
    PREPROCESSING_VERSION,
)


def test_preprocess_crop_removes_black_half():
    """Top half black + bottom half gray → output is panel-dominated, no large black region."""
    arr = np.zeros((400, 200, 3), dtype=np.uint8)
    arr[200:, :, :] = 128
    out = preprocess_crop(_PILImage.fromarray(arr))
    out_arr = np.asarray(out)

    assert out.size == (INPUT_SIZE, INPUT_SIZE)
    assert out_arr.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    black_ratio = float((out_arr.mean(axis=-1) < 30).mean())
    assert black_ratio < 0.10, f"black region should be cropped out, got ratio={black_ratio:.3f}"
    assert 100 < out_arr.mean() < 150, f"output should be panel-dominated, got mean={out_arr.mean():.1f}"


def test_preprocess_crop_uniform_panel_is_noop_like():
    """No significant black → Otsu skipped, aspect-resize only."""
    arr = np.full((300, 300, 3), 100, dtype=np.uint8)
    out = preprocess_crop(_PILImage.fromarray(arr))
    out_arr = np.asarray(out)

    assert out.size == (INPUT_SIZE, INPUT_SIZE)
    assert abs(out_arr.mean() - 100) < 3, f"uniform input should survive intact, got mean={out_arr.mean():.1f}"


def test_preprocess_crop_non_square_pad_preserves_aspect():
    """Non-square uniform input is padded with median gray, not distorted."""
    arr = np.full((100, 300, 3), 128, dtype=np.uint8)  # wide
    out = preprocess_crop(_PILImage.fromarray(arr))
    out_arr = np.asarray(out)

    assert out.size == (INPUT_SIZE, INPUT_SIZE)
    # No stretch: wide input → letterboxed with panel median (top/bottom pad = 128)
    assert abs(out_arr.mean() - 128) < 3


def test_find_panel_bbox_skips_uniform():
    gray = np.full((100, 100), 128, dtype=np.uint8)
    assert _find_panel_bbox(gray) is None


def test_find_panel_bbox_finds_black_half():
    gray = np.zeros((200, 200), dtype=np.uint8)
    gray[100:, :] = 128
    bbox = _find_panel_bbox(gray)
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    # Panel occupies bottom half → y0 ≈ 100
    assert 95 <= y0 <= 105, f"expected panel to start around y=100, got y0={y0}"
    assert y1 == 200


def test_preprocessing_version_in_fingerprint():
    """Fingerprint must change when PREPROCESSING_VERSION changes (cache invalidation)."""
    from scripts.over_review_poc.features import _manifest_fingerprint
    import scripts.over_review_poc.features as F

    samples = [_mk_sample("s1", "true_ng", "g1", "G0F00000", "patchcore_tile")]
    fp_a = _manifest_fingerprint(samples)

    original = F.PREPROCESSING_VERSION
    try:
        F.PREPROCESSING_VERSION = original + "_mutated"
        fp_b = _manifest_fingerprint(samples)
    finally:
        F.PREPROCESSING_VERSION = original

    assert fp_a != fp_b, "fingerprint must differ when preprocessing version changes"


def test_fingerprint_differs_by_preprocessing_id():
    """Passing a different preprocessing_id argument must produce different fingerprint."""
    from scripts.over_review_poc.features import _manifest_fingerprint

    samples = [_mk_sample("s1", "true_ng", "g1", "G0F00000", "patchcore_tile")]
    fp_naive = _manifest_fingerprint(samples, preprocessing_id="v1_naive_resize")
    fp_clahe = _manifest_fingerprint(samples, preprocessing_id="v3_clahe_cl2.0_tg8")
    assert fp_naive != fp_clahe


# ---------- CLAHE ----------

from scripts.over_review_poc.features import preprocess_clahe, build_transform_clahe


def test_preprocess_clahe_boosts_low_contrast_local_variance():
    """CLAHE should increase local contrast on a region with a faint feature."""
    # Mid-gray background (128) with a faint dark line (Δ=8)
    arr = np.full((200, 200, 3), 128, dtype=np.uint8)
    arr[95:105, 40:160, :] = 120
    img = _PILImage.fromarray(arr)
    pre_std = float(np.asarray(img.convert("L"))[80:120, 30:170].std())

    out = preprocess_clahe(img)
    out_gray = np.asarray(out.convert("L"))
    post_std = float(out_gray[80:120, 30:170].std())

    assert out.size == img.size, "preprocess_clahe must not resize"
    # clipLimit=2.0 is conservative; even a modest 1.1x bump proves it is amplifying
    # local contrast rather than just passing through.
    assert post_std > pre_std * 1.1, (
        f"CLAHE should amplify local contrast: pre_std={pre_std:.2f}, post_std={post_std:.2f}"
    )


def test_preprocess_clahe_output_is_rgb_3channel():
    """Even grayscale input should produce 3-channel RGB (for DINOv2 input)."""
    arr = np.full((100, 100), 100, dtype=np.uint8)
    img = _PILImage.fromarray(arr, mode="L")
    out = preprocess_clahe(img)
    assert out.mode == "RGB"
    out_arr = np.asarray(out)
    assert out_arr.shape == (100, 100, 3)
    # R == G == B (grayscale stacked)
    assert np.array_equal(out_arr[..., 0], out_arr[..., 1])
    assert np.array_equal(out_arr[..., 1], out_arr[..., 2])


def test_build_transform_clahe_output_tensor_shape():
    """End-to-end CLAHE transform produces 3×INPUT_SIZE×INPUT_SIZE normalized tensor."""
    arr = np.full((300, 300, 3), 120, dtype=np.uint8)
    arr[140:160, 50:250, :] = 100  # faint feature
    img = _PILImage.fromarray(arr)

    transform = build_transform_clahe()
    out = transform(img)
    assert out.shape == (3, INPUT_SIZE, INPUT_SIZE)
    # ImageNet-normalized values typically within roughly [-3, 3]
    assert out.min() > -5 and out.max() < 5


# ---------- CLAHE tile-only (v4) ----------

from scripts.over_review_poc.features import (
    preprocess_clahe_tile_only,
    build_transform_clahe_tile_only,
)


def test_clahe_tile_only_applies_on_uniform_tile():
    """Uniform panel tile (no black) → CLAHE applied → local variance increases."""
    arr = np.full((300, 300, 3), 120, dtype=np.uint8)
    arr[140:160, 50:250, :] = 110  # faint feature
    img = _PILImage.fromarray(arr)
    pre_std = float(np.asarray(img.convert("L"))[130:170, 40:260].std())

    out = preprocess_clahe_tile_only(img)
    post_std = float(np.asarray(out.convert("L"))[130:170, 40:260].std())
    assert post_std > pre_std, (
        f"tile-only CLAHE should amplify local contrast on tile crop: "
        f"pre={pre_std:.2f}, post={post_std:.2f}"
    )


def test_clahe_tile_only_passes_through_edge_crop():
    """Edge crop (50% black) → CLAHE skipped → output = input unchanged."""
    arr = np.zeros((400, 400, 3), dtype=np.uint8)
    arr[200:, :, :] = 128  # bottom half is panel gray
    img = _PILImage.fromarray(arr)
    out_arr = np.asarray(preprocess_clahe_tile_only(img))
    # Should be identical — pass-through
    assert np.array_equal(out_arr, np.asarray(img))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
