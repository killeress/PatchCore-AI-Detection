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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
