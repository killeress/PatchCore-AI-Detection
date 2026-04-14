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
