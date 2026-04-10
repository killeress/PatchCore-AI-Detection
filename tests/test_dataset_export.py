"""
過檢資料蒐集工具測試

執行方式:
    python tests/test_dataset_export.py        # 跑全部
    pytest tests/test_dataset_export.py -v     # 用 pytest
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from capi_dataset_export import (
    CROP_SIZE, OVER_LABEL_MAP, TRUE_NG_LABEL, MANIFEST_FIELDS,
    SampleCandidate, DatasetExporter,
)


def test_constants_aligned_with_db_enum():
    """OVER_LABEL_MAP 的 key 必須與 capi_database.VALID_OVER_CATEGORIES 一致"""
    from capi_database import CAPIDatabase
    assert set(OVER_LABEL_MAP.keys()) == CAPIDatabase.VALID_OVER_CATEGORIES


def test_manifest_fields_contains_required_columns():
    required = {"sample_id", "label", "crop_path", "heatmap_path", "status"}
    assert required.issubset(set(MANIFEST_FIELDS))


from capi_dataset_export import crop_patchcore_tile, crop_edge_defect


def test_crop_patchcore_tile_exact_512():
    """tile 剛好 512×512，直接切出不需要 pad"""
    img = np.full((2048, 2048, 3), 128, dtype=np.uint8)
    # 在 (512,512) 放一個白塊作為標記
    img[512:1024, 512:1024] = 255
    crop = crop_patchcore_tile(img, x=512, y=512, w=512, h=512)
    assert crop.shape == (512, 512, 3)
    assert crop[0, 0, 0] == 255  # 剛好切到白塊


def test_crop_patchcore_tile_edge_needs_pad():
    """tile 在右下角，w/h < 512，需要右/下 pad 黑邊"""
    img = np.full((600, 600, 3), 200, dtype=np.uint8)
    crop = crop_patchcore_tile(img, x=400, y=400, w=200, h=200)
    assert crop.shape == (512, 512, 3)
    # 右下 pad 區應為黑
    assert crop[500, 500, 0] == 0
    # 左上原圖區應為 200
    assert crop[0, 0, 0] == 200


def test_crop_edge_defect_center_interior():
    """center 在圖內，上下左右空間足夠，不需要 pad"""
    img = np.full((2048, 2048, 3), 100, dtype=np.uint8)
    img[1000, 1000] = [255, 0, 0]  # center 處放紅點
    crop = crop_edge_defect(img, cx=1000, cy=1000)
    assert crop.shape == (512, 512, 3)
    # 中心 pixel（256,256）應為紅點
    assert tuple(crop[256, 256]) == (255, 0, 0)


def test_crop_edge_defect_near_top_left_corner():
    """center=(50,50)，上/左會 clamp + pad 黑邊；紅點中心保持在 crop 的 (256,256)"""
    img = np.full((1024, 1024, 3), 100, dtype=np.uint8)
    img[50, 50] = [0, 255, 0]
    crop = crop_edge_defect(img, cx=50, cy=50)
    assert crop.shape == (512, 512, 3)
    # center 經 pad 後仍在 (256,256)
    assert tuple(crop[256, 256]) == (0, 255, 0)
    # 左上角 (0,0) 在 pad 黑邊區
    assert tuple(crop[0, 0]) == (0, 0, 0)


from capi_dataset_export import (
    determine_label, extract_prefix, build_sample_filename, build_sample_id,
)


def test_determine_label_true_ng():
    assert determine_label(ric="NG", over_category=None) == "true_ng"


def test_determine_label_over_review_category():
    assert determine_label(ric="OK", over_category="edge_false_positive") == "over_edge_false_positive"
    assert determine_label(ric="OK", over_category="other") == "over_other"


def test_determine_label_returns_none_for_unfilled():
    """RIC=OK 但沒回填 category → 不蒐集（回 None）"""
    assert determine_label(ric="OK", over_category=None) is None
    assert determine_label(ric="OK", over_category="") is None


def test_determine_label_unknown_category_raises():
    with pytest.raises(ValueError):
        determine_label(ric="OK", over_category="not_a_real_category")


def test_extract_prefix_with_timestamp():
    assert extract_prefix("G0F00000_114438.tif") == "G0F00000"
    assert extract_prefix("STANDARD.png") == "STANDARD"
    assert extract_prefix("WGF_0001_20260410.bmp") == "WGF_0001"


def test_build_sample_id_patchcore():
    assert build_sample_id("GLS123", "G0F0001.bmp", "patchcore_tile", tile_idx=3) == "GLS123_G0F0001_tile3"


def test_build_sample_id_edge_defect():
    assert build_sample_id("GLS123", "W0F0002.bmp", "edge_defect", edge_defect_id=7) == "GLS123_W0F0002_edge7"


def test_build_sample_filename_patchcore():
    # 日期取自 inference_timestamp 的 YYYY-MM-DD
    fn = build_sample_filename(
        glass_id="GLS123", image_name="G0F0001.bmp",
        sample_key="tile3", inference_timestamp="2026-04-08T14:22:03"
    )
    assert fn == "20260408_GLS123_G0F0001_tile3.png"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
