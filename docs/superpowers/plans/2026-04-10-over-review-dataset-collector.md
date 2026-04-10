# 過檢資料蒐集工具 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增「過檢訓練資料集蒐集工具」，自動從 RIC 回填結果蒐集 AI=NG&RIC=NG (true_ng) 與 AI=NG&RIC=OK 已回填 category 的過檢樣本，輸出 crop+heatmap 對到分類目錄 + manifest.csv，供後續分類 AI 或 CV 後處理評估，並整合到 `/ric` Web Dashboard 以背景 async job 觸發。

**Architecture:** 新增 `capi_dataset_export.py` 核心 Exporter（純邏輯、無 HTTP），在 `capi_web.py` 加入 4 個 API 端點 + job state，在 `templates/ric_report.html` 的過檢 Review tab 加按鈕 + modal + polling JS。DB 操作完全複用既有 `CAPIDatabase.get_client_accuracy_records()` 與 `get_record_detail()`，不修改 schema。

**Tech Stack:** Python 3（標準庫 `csv` / `shutil` / `threading`）、OpenCV（crop + pad）、既有 `CAPIDatabase`、`resolve_unc_path`、ThreadingHTTPServer、Bootstrap modal + vanilla JS polling。

**Spec:** `docs/superpowers/specs/2026-04-10-over-review-dataset-collector-design.md`

---

## File Structure

### 新增檔案

| 檔案 | 責任 |
|---|---|
| `capi_dataset_export.py` | 核心 Exporter class + 模組級 job state（與 web 解耦的純邏輯） |
| `tests/test_dataset_export.py` | crop / label / manifest / candidate 篩選 的單元測試 |

### 修改檔案

| 檔案 | 修改範圍 |
|---|---|
| `capi_web.py` | 新增 4 個 API handler (`/api/dataset_export/*`)、初始化 job state |
| `templates/ric_report.html` | 過檢 Review tab 頂部按鈕 + Bootstrap modal + polling JS |
| `server_config.yaml` | 新增 `dataset_export:` 區段（Linux 預設） |
| `server_config_local.yaml` | 新增 `dataset_export:` 區段（Windows 預設） |

### 不修改

- `capi_database.py`（只讀查詢，完全複用現有 method）
- `capi_inference.py`、`capi_heatmap.py`、`capi_server.py`（獨立運作）

---

## Task 1：server_config 設定 + 空模組骨架 + 常數定義

**Files:**
- Modify: `server_config.yaml`（加 `dataset_export` 區段）
- Modify: `server_config_local.yaml`（加 `dataset_export` 區段）
- Create: `capi_dataset_export.py`
- Create: `tests/test_dataset_export.py`

- [ ] **Step 1.1: 在 server_config.yaml 末尾新增 dataset_export 區段**

找到檔案最後一個頂層 key 之後附加：
```yaml
# 過檢訓練資料集蒐集
dataset_export:
  base_dir: "/data/capi_ai/datasets/over_review"
  # 啟動前檢查磁碟剩餘空間 (GB)，小於此值會拒絕啟動 job
  min_free_space_gb: 1
```

- [ ] **Step 1.2: 在 server_config_local.yaml 加相同區段但用 Windows 路徑**

```yaml
# 過檢訓練資料集蒐集
dataset_export:
  base_dir: "./datasets/over_review"
  min_free_space_gb: 1
```

- [ ] **Step 1.3: 建立 capi_dataset_export.py 骨架**

```python
"""
過檢訓練資料集蒐集工具

從 RIC 回填結果蒐集 AI=NG&RIC=NG (true_ng) 與 AI=NG&RIC=OK 已回填 category 的樣本，
輸出 crop + heatmap 對到分類目錄結構 + manifest.csv，供後續分類 AI 訓練或 CV 後處理評估。

Spec: docs/superpowers/specs/2026-04-10-over-review-dataset-collector-design.md
"""
from __future__ import annotations

import csv
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---- 常數 ----

CROP_SIZE = 512

# 與 capi_database.py VALID_OVER_CATEGORIES 對應的 label 目錄名
OVER_LABEL_MAP = {
    "edge_false_positive": "over_edge_false_positive",
    "within_spec": "over_within_spec",
    "overexposure": "over_overexposure",
    "surface_scratch": "over_surface_scratch",
    "aoi_ai_false_positive": "over_aoi_ai_false_positive",
    "other": "over_other",
}

TRUE_NG_LABEL = "true_ng"

# Manifest CSV 欄位順序（固定，避免 DictWriter 亂序）
MANIFEST_FIELDS = [
    "sample_id", "collected_at", "label", "source_type", "prefix",
    "glass_id", "image_name",
    "inference_record_id", "image_result_id",
    "tile_idx", "edge_defect_id",
    "crop_path", "heatmap_path",
    "ai_score", "defect_x", "defect_y",
    "ric_judgment", "over_review_category", "over_review_note",
    "inference_timestamp", "status",
]

# Job 狀態常數
JOB_STATE_IDLE = "idle"
JOB_STATE_RUNNING = "running"
JOB_STATE_COMPLETED = "completed"
JOB_STATE_FAILED = "failed"
JOB_STATE_CANCELLED = "cancelled"


# ---- Dataclasses ----

@dataclass
class SampleCandidate:
    """一個待處理樣本（尚未實際寫入 disk）"""
    sample_id: str          # 去重 key: f"{glass_id}_{image_stem}_{sample_key}"
    source_type: str        # "patchcore_tile" | "edge_defect"
    glass_id: str
    image_name: str
    image_path: str         # 原圖完整路徑（已做 UNC → Linux mapping）
    inference_record_id: int
    image_result_id: int
    tile_idx: Optional[int]
    edge_defect_id: Optional[int]
    prefix: str             # G0F / R0F / W0F / WGF / STANDARD / …
    label: str              # true_ng / over_edge_false_positive / …
    # Crop 座標資訊（依 source_type 填不同欄位，未用到的留 None）
    tile_x: Optional[int] = None
    tile_y: Optional[int] = None
    tile_w: Optional[int] = None
    tile_h: Optional[int] = None
    edge_center_x: Optional[int] = None
    edge_center_y: Optional[int] = None
    # 通用 metadata
    src_heatmap_path: str = ""      # DB 紀錄的原 heatmap 檔案路徑
    ai_score: float = 0.0
    ric_judgment: str = ""
    over_review_category: str = ""
    over_review_note: str = ""
    inference_timestamp: str = ""


@dataclass
class JobSummary:
    """Job 完成後的統計摘要"""
    job_id: str
    started_at: str
    finished_at: str
    duration_sec: float
    total: int = 0
    labels: Dict[str, int] = field(default_factory=dict)  # label → 數量
    skipped: Dict[str, int] = field(default_factory=dict)  # reason → 數量
    output_dir: str = ""


# ---- Exporter（實作在後續 Task 加入） ----

class DatasetExporter:
    """過檢訓練資料集蒐集器（pure logic，不含 HTTP）"""

    def __init__(self, db, base_dir: str, path_mapping: Dict[str, str]):
        self.db = db
        self.base_dir = Path(base_dir).resolve()
        self.path_mapping = path_mapping

    def run(self, days: int, include_true_ng: bool, skip_existing: bool,
            status_callback=None, cancel_event: Optional[threading.Event] = None
            ) -> JobSummary:
        """主執行入口（實作在 Task 6）"""
        raise NotImplementedError("Implemented in Task 6")
```

- [ ] **Step 1.4: 建立 tests/test_dataset_export.py 測試骨架**

```python
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
```

- [ ] **Step 1.5: 執行測試驗證骨架可 import**

Run: `cd C:/Users/rh.syu/Desktop/CAPI01_AD && python tests/test_dataset_export.py`
Expected: 兩個測試都 PASS（`test_constants_aligned_with_db_enum`、`test_manifest_fields_contains_required_columns`）

- [ ] **Step 1.6: Commit**

```bash
git add capi_dataset_export.py tests/test_dataset_export.py server_config.yaml server_config_local.yaml
git commit -m "feat(dataset_export): 新增模組骨架、常數與 server_config 設定"
```

---

## Task 2：Crop 工具函式（PatchCore tile + Edge defect）

純函式、容易測試。兩種 source_type 對應兩個 function。

**Files:**
- Modify: `capi_dataset_export.py`（新增 `crop_patchcore_tile`、`crop_edge_defect`、`_pad_to_size`）
- Modify: `tests/test_dataset_export.py`（新增 4 個 crop 測試）

- [ ] **Step 2.1: 先寫失敗測試 — patchcore tile 正常 512×512 情況**

在 `tests/test_dataset_export.py` 加：
```python
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
```

- [ ] **Step 2.2: 再寫 edge defect 的失敗測試**

```python
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
```

- [ ] **Step 2.3: 執行測試驗證全部 fail**

Run: `python tests/test_dataset_export.py`
Expected: 4 個新測試都 FAIL with `ImportError: cannot import name 'crop_patchcore_tile'`

- [ ] **Step 2.4: 實作 `_pad_to_size`、`crop_patchcore_tile`、`crop_edge_defect`**

在 `capi_dataset_export.py` 的 `DatasetExporter` class 之前新增：
```python
def _pad_to_size(img: np.ndarray, target: int = CROP_SIZE,
                 pad_top: int = 0, pad_left: int = 0) -> np.ndarray:
    """將 img pad 到 target×target，黑邊 (value=0)。
    pad_top / pad_left 允許指定「原圖左上角在 target 中的偏移」，
    其餘空間用 BORDER_CONSTANT 補黑。
    """
    h, w = img.shape[:2]
    pad_bottom = target - h - pad_top
    pad_right = target - w - pad_left
    if pad_top < 0 or pad_left < 0 or pad_bottom < 0 or pad_right < 0:
        raise ValueError(
            f"Image too large for target: shape=({h},{w}) target={target} "
            f"offset=({pad_top},{pad_left})"
        )
    return cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0
    )


def crop_patchcore_tile(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """從原圖切出 PatchCore tile 區域，不足 CROP_SIZE 用黑邊在右/下側 pad。

    Args:
        img: 原圖 (H, W, C) 或 (H, W)
        x, y, w, h: tile 在原圖的絕對座標（tile_results.x/y/width/height）

    Returns:
        CROP_SIZE × CROP_SIZE 的 crop
    """
    H, W = img.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    crop = img[y1:y2, x1:x2]
    if crop.shape[:2] == (CROP_SIZE, CROP_SIZE):
        return crop
    # 右/下 pad（tile 通常在 grid 邊緣才會不足）
    return _pad_to_size(crop, CROP_SIZE, pad_top=0, pad_left=0)


def crop_edge_defect(img: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """以 (cx, cy) 為中心切出 CROP_SIZE × CROP_SIZE，clamp 後再 pad 保持中心對齊。

    Args:
        img: 原圖
        cx, cy: edge_defect_results.center_x / center_y

    Returns:
        CROP_SIZE × CROP_SIZE 的 crop
    """
    half = CROP_SIZE // 2
    H, W = img.shape[:2]
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half

    x1_c = max(0, x1)
    y1_c = max(0, y1)
    x2_c = min(W, x2)
    y2_c = min(H, y2)
    crop = img[y1_c:y2_c, x1_c:x2_c]

    # 以 clamp 掉的量當 top/left pad，確保 defect 中心在 crop 的 (half, half)
    pad_top = y1_c - y1
    pad_left = x1_c - x1
    return _pad_to_size(crop, CROP_SIZE, pad_top=pad_top, pad_left=pad_left)
```

- [ ] **Step 2.5: 執行測試驗證全部 PASS**

Run: `python tests/test_dataset_export.py`
Expected: 所有測試 PASS（含原有 2 個與新增 4 個）

- [ ] **Step 2.6: Commit**

```bash
git add capi_dataset_export.py tests/test_dataset_export.py
git commit -m "feat(dataset_export): 新增 crop 工具函式（PatchCore tile 與 edge defect 中心裁切）"
```

---

## Task 3：Label 判定 + 檔名 + prefix 抽取輔助函式

**Files:**
- Modify: `capi_dataset_export.py`
- Modify: `tests/test_dataset_export.py`

- [ ] **Step 3.1: 先寫失敗測試**

```python
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
```

- [ ] **Step 3.2: 執行測試驗證全部 FAIL**

Run: `python tests/test_dataset_export.py`
Expected: 新增測試全部 FAIL with ImportError

- [ ] **Step 3.3: 實作四個輔助函式**

在 `capi_dataset_export.py` 常數下方新增：
```python
def determine_label(ric: str, over_category: Optional[str]) -> Optional[str]:
    """依 RIC 判定與 over_review category 決定輸出 label。

    Returns:
        - "true_ng" 若 RIC=NG
        - "over_<category>" 若 RIC=OK 且 category 在 OVER_LABEL_MAP
        - None 若 RIC=OK 且 category 未填（這類樣本不蒐集）

    Raises:
        ValueError: 若 RIC=OK 且 category 不在合法 enum
    """
    if ric == "NG":
        return TRUE_NG_LABEL
    if ric == "OK":
        if not over_category:
            return None
        if over_category not in OVER_LABEL_MAP:
            raise ValueError(f"Unknown over_review category: {over_category}")
        return OVER_LABEL_MAP[over_category]
    raise ValueError(f"Unknown RIC judgment: {ric}")


def extract_prefix(image_name: str) -> str:
    """從原圖檔名抽出光源 prefix（去掉 timestamp 尾綴）。

    Mirror of capi_inference.CAPIInferencer._get_image_prefix but stand-alone
    so 本工具不需要 inferencer 實例就能分類樣本。

    Examples:
        G0F00000_114438.tif → G0F00000
        STANDARD.png → STANDARD
        WGF_0001_20260410.bmp → WGF_0001
    """
    stem = Path(image_name).stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def build_sample_id(glass_id: str, image_name: str, source_type: str,
                    tile_idx: Optional[int] = None,
                    edge_defect_id: Optional[int] = None) -> str:
    """去重 key：glass_id + image_stem + sample_key"""
    stem = Path(image_name).stem
    if source_type == "patchcore_tile":
        if tile_idx is None:
            raise ValueError("tile_idx required for patchcore_tile")
        return f"{glass_id}_{stem}_tile{tile_idx}"
    if source_type == "edge_defect":
        if edge_defect_id is None:
            raise ValueError("edge_defect_id required for edge_defect")
        return f"{glass_id}_{stem}_edge{edge_defect_id}"
    raise ValueError(f"Unknown source_type: {source_type}")


def build_sample_filename(glass_id: str, image_name: str,
                          sample_key: str, inference_timestamp: str) -> str:
    """檔名：{YYYYMMDD}_{glass_id}_{image_stem}_{sample_key}.png

    inference_timestamp 支援 'YYYY-MM-DDTHH:MM:SS' 與 'YYYY-MM-DD HH:MM:SS'。
    """
    stem = Path(image_name).stem
    ts = inference_timestamp.replace("T", " ")[:10]  # 'YYYY-MM-DD'
    yyyymmdd = ts.replace("-", "")
    return f"{yyyymmdd}_{glass_id}_{stem}_{sample_key}.png"
```

- [ ] **Step 3.4: 執行測試驗證全部 PASS**

Run: `python tests/test_dataset_export.py`
Expected: 全部 PASS

- [ ] **Step 3.5: Commit**

```bash
git add capi_dataset_export.py tests/test_dataset_export.py
git commit -m "feat(dataset_export): 新增 label 判定 / prefix / 檔名 / sample_id 輔助函式"
```

---

## Task 4：Candidate 蒐集（DB 查詢 → SampleCandidate list）

從 DB 讀已回填 Review 的過檢紀錄 + 真 NG，展開到 tile 與 edge_defect 層級的 candidate，過濾炸彈。

**Files:**
- Modify: `capi_dataset_export.py`
- Modify: `tests/test_dataset_export.py`

- [ ] **Step 4.1: 寫失敗測試 — 使用 fake db object**

```python
class FakeDB:
    """模擬 CAPIDatabase，用靜態 dict 回傳 fixture"""

    def __init__(self, accuracy_rows, record_details):
        self._accuracy_rows = accuracy_rows
        self._record_details = record_details

    def get_client_accuracy_records(self, start_date=None, end_date=None):
        return self._accuracy_rows

    def get_record_detail(self, record_id):
        return self._record_details.get(record_id)


def _make_accuracy_row(**overrides):
    base = {
        "id": 1, "time_stamp": "2026-04-08 10:00:00", "pnl_id": "GLS123",
        "mach_id": "M01", "result_eqp": "NG", "result_ai": "NG",
        "result_ric": "OK", "datastr": "",
        "review_id": None, "review_category": None, "review_note": None,
        "review_updated_at": None,
        "over_review_id": 1, "over_review_category": "edge_false_positive",
        "over_review_note": "測試", "over_review_updated_at": "2026-04-09 15:00:00",
        "inference_record_id": 1001,
    }
    base.update(overrides)
    return base


def _make_record_detail(**overrides):
    base = {
        "id": 1001, "glass_id": "GLS123", "image_dir": "/data/panels/GLS123",
        "request_time": "2026-04-08T10:00:00",
        "images": [
            {
                "id": 5001, "image_name": "G0F00000_114438.tif",
                "image_path": "/data/panels/GLS123/G0F00000_114438.tif",
                "is_bomb": 0,
                "tiles": [
                    {"id": 9001, "tile_id": 3, "x": 0, "y": 0, "width": 512, "height": 512,
                     "score": 0.85, "is_anomaly": 1, "is_bomb": 0,
                     "heatmap_path": "/tmp/heatmap_tile3.png"},
                    {"id": 9002, "tile_id": 4, "x": 512, "y": 0, "width": 512, "height": 512,
                     "score": 0.95, "is_anomaly": 1, "is_bomb": 1,  # 炸彈 tile 要被過濾
                     "heatmap_path": "/tmp/heatmap_tile4.png"},
                ],
                "edge_defects": [
                    {"id": 7001, "center_x": 1000, "center_y": 1200,
                     "max_diff": 30.5, "heatmap_path": "/tmp/edge_7001.png",
                     "is_dust": 0},
                ],
            },
            {
                "id": 5002, "image_name": "W0F00000_114500.tif",
                "image_path": "/data/panels/GLS123/W0F00000_114500.tif",
                "is_bomb": 1,  # 整張炸彈圖 → 全部跳過
                "tiles": [
                    {"id": 9003, "tile_id": 1, "x": 0, "y": 0, "width": 512, "height": 512,
                     "score": 0.91, "is_anomaly": 1, "is_bomb": 0,
                     "heatmap_path": "/tmp/heatmap_wtile1.png"},
                ],
                "edge_defects": [],
            },
        ],
    }
    base.update(overrides)
    return base


def test_collect_candidates_filters_bomb_and_non_anomaly():
    db = FakeDB(
        accuracy_rows=[_make_accuracy_row()],
        record_details={1001: _make_record_detail()},
    )
    exporter = DatasetExporter(db, base_dir="/tmp/out", path_mapping={})
    candidates = exporter.collect_candidates(
        days=3, include_true_ng=True,
    )
    ids = [c.sample_id for c in candidates]

    # 應有：tile3 (G0F) + edge7001 (G0F)
    # 應無：tile4 (is_bomb=1), W0F image 整張 (is_bomb=1)
    assert "GLS123_G0F00000_114438_tile3" in ids
    assert "GLS123_G0F00000_114438_edge7001" in ids
    assert not any("tile4" in i for i in ids)
    assert not any("W0F" in i for i in ids)


def test_collect_candidates_skips_unfilled_over_review():
    row_unfilled = _make_accuracy_row(
        over_review_id=None, over_review_category=None, over_review_note=None
    )
    db = FakeDB(
        accuracy_rows=[row_unfilled],
        record_details={1001: _make_record_detail()},
    )
    exporter = DatasetExporter(db, base_dir="/tmp/out", path_mapping={})
    candidates = exporter.collect_candidates(days=3, include_true_ng=True)
    assert candidates == []


def test_collect_candidates_include_true_ng_false_skips_ric_ng():
    row_true_ng = _make_accuracy_row(
        result_ric="NG", over_review_id=None, over_review_category=None,
    )
    db = FakeDB(
        accuracy_rows=[row_true_ng],
        record_details={1001: _make_record_detail()},
    )
    exporter = DatasetExporter(db, base_dir="/tmp/out", path_mapping={})
    candidates = exporter.collect_candidates(days=3, include_true_ng=False)
    assert candidates == []


def test_collect_candidates_labels_true_ng():
    row_true_ng = _make_accuracy_row(
        result_ric="NG", over_review_id=None, over_review_category=None,
    )
    db = FakeDB(
        accuracy_rows=[row_true_ng],
        record_details={1001: _make_record_detail()},
    )
    exporter = DatasetExporter(db, base_dir="/tmp/out", path_mapping={})
    candidates = exporter.collect_candidates(days=3, include_true_ng=True)
    assert len(candidates) >= 1
    assert all(c.label == "true_ng" for c in candidates)


def test_collect_candidates_attaches_prefix_and_metadata():
    db = FakeDB(
        accuracy_rows=[_make_accuracy_row()],
        record_details={1001: _make_record_detail()},
    )
    exporter = DatasetExporter(db, base_dir="/tmp/out", path_mapping={})
    tile_cand = next(
        c for c in exporter.collect_candidates(days=3, include_true_ng=True)
        if c.source_type == "patchcore_tile"
    )
    assert tile_cand.prefix == "G0F00000"
    assert tile_cand.label == "over_edge_false_positive"
    assert tile_cand.ai_score == 0.85
    assert tile_cand.over_review_note == "測試"
    assert tile_cand.ric_judgment == "OK"
    assert tile_cand.tile_x == 0 and tile_cand.tile_w == 512
```

- [ ] **Step 4.2: 執行測試驗證全部 FAIL**

Run: `python tests/test_dataset_export.py`
Expected: 新測試全部 FAIL with `AttributeError: 'DatasetExporter' object has no attribute 'collect_candidates'`

- [ ] **Step 4.3: 實作 `collect_candidates` 與 `_flatten_record_to_candidates`**

在 `DatasetExporter` class 內新增：
```python
def collect_candidates(self, days: int, include_true_ng: bool) -> List[SampleCandidate]:
    """從 DB 讀最近 N 日的 client_accuracy_records，展開成 candidate 清單。

    篩選規則：
      - result_ai == "NG"
      - 若 result_ric == "NG" 且 include_true_ng → 收為 true_ng
      - 若 result_ric == "OK" 且 over_review_category 已填 → 收為 over_<category>
      - 其他情況跳過

    過濾：
      - image_results.is_bomb == 1 整張跳過
      - tile_results.is_bomb == 1 該 tile 跳過
      - tile_results.is_anomaly == 0 跳過（只蒐集 NG tile）
    """
    end = datetime.now().date()
    start = end - timedelta(days=days - 1)
    rows = self.db.get_client_accuracy_records(
        start_date=start.isoformat(), end_date=end.isoformat()
    )

    candidates: List[SampleCandidate] = []
    for row in rows:
        if row.get("result_ai") != "NG":
            continue

        ric = row.get("result_ric") or ""
        over_category = row.get("over_review_category")

        if ric == "NG":
            if not include_true_ng:
                continue
            label = TRUE_NG_LABEL
        elif ric == "OK":
            if not over_category:
                continue  # 未回填的跳過
            try:
                label = determine_label(ric, over_category)
            except ValueError as e:
                logger.warning("Skip unknown over_review category: %s", e)
                continue
            if label is None:
                continue
        else:
            continue

        inference_record_id = row.get("inference_record_id")
        if not inference_record_id:
            logger.warning("No inference_record linked: pnl_id=%s time=%s",
                           row.get("pnl_id"), row.get("time_stamp"))
            continue

        detail = self.db.get_record_detail(inference_record_id)
        if not detail:
            continue

        candidates.extend(self._flatten_record_to_candidates(
            detail=detail, label=label, row=row,
        ))

    return candidates


def _flatten_record_to_candidates(
    self, detail: Dict, label: str, row: Dict
) -> List[SampleCandidate]:
    """把一筆 inference_record 展開成多個 SampleCandidate（依 image / tile / edge）。"""
    out: List[SampleCandidate] = []
    glass_id = detail.get("glass_id") or row.get("pnl_id") or ""
    inference_timestamp = detail.get("request_time") or row.get("time_stamp") or ""
    record_id = detail.get("id")

    for img in detail.get("images") or []:
        if img.get("is_bomb"):
            continue
        image_name = img.get("image_name") or ""
        image_path = img.get("image_path") or ""
        image_result_id = img.get("id")
        prefix = extract_prefix(image_name)

        # PatchCore tile 樣本
        for tile in img.get("tiles") or []:
            if tile.get("is_bomb"):
                continue
            if not tile.get("is_anomaly"):
                continue
            tile_idx = tile.get("tile_id", 0)
            sample_id = build_sample_id(glass_id, image_name, "patchcore_tile", tile_idx=tile_idx)
            out.append(SampleCandidate(
                sample_id=sample_id,
                source_type="patchcore_tile",
                glass_id=glass_id,
                image_name=image_name,
                image_path=image_path,
                inference_record_id=record_id,
                image_result_id=image_result_id,
                tile_idx=tile_idx,
                edge_defect_id=None,
                prefix=prefix,
                label=label,
                tile_x=tile.get("x", 0),
                tile_y=tile.get("y", 0),
                tile_w=tile.get("width", CROP_SIZE),
                tile_h=tile.get("height", CROP_SIZE),
                src_heatmap_path=tile.get("heatmap_path", ""),
                ai_score=float(tile.get("score", 0.0)),
                ric_judgment=row.get("result_ric") or "",
                over_review_category=row.get("over_review_category") or "",
                over_review_note=row.get("over_review_note") or "",
                inference_timestamp=inference_timestamp,
            ))

        # Edge defect 樣本
        for edge in img.get("edge_defects") or []:
            edge_id = edge.get("id")
            sample_id = build_sample_id(glass_id, image_name, "edge_defect", edge_defect_id=edge_id)
            out.append(SampleCandidate(
                sample_id=sample_id,
                source_type="edge_defect",
                glass_id=glass_id,
                image_name=image_name,
                image_path=image_path,
                inference_record_id=record_id,
                image_result_id=image_result_id,
                tile_idx=None,
                edge_defect_id=edge_id,
                prefix=prefix,
                label=label,
                edge_center_x=edge.get("center_x", 0),
                edge_center_y=edge.get("center_y", 0),
                src_heatmap_path=edge.get("heatmap_path", ""),
                ai_score=float(edge.get("max_diff", 0.0)),
                ric_judgment=row.get("result_ric") or "",
                over_review_category=row.get("over_review_category") or "",
                over_review_note=row.get("over_review_note") or "",
                inference_timestamp=inference_timestamp,
            ))

    return out
```

- [ ] **Step 4.4: 執行測試驗證全部 PASS**

Run: `python tests/test_dataset_export.py`
Expected: 全部 PASS

- [ ] **Step 4.5: Commit**

```bash
git add capi_dataset_export.py tests/test_dataset_export.py
git commit -m "feat(dataset_export): 實作 DB candidate 蒐集，過濾炸彈/非 anomaly tile"
```

---

## Task 5：Manifest CSV I/O + 去重 / 移動邏輯

**Files:**
- Modify: `capi_dataset_export.py`
- Modify: `tests/test_dataset_export.py`

- [ ] **Step 5.1: 寫失敗測試 — 使用 tmp_path fixture**

```python
def test_manifest_read_empty_when_not_exists(tmp_path):
    from capi_dataset_export import read_manifest
    m = read_manifest(tmp_path / "manifest.csv")
    assert m == {}


def test_manifest_roundtrip(tmp_path):
    from capi_dataset_export import read_manifest, write_manifest
    manifest_path = tmp_path / "manifest.csv"
    rows = {
        "GLS123_G0F0001_tile3": {
            "sample_id": "GLS123_G0F0001_tile3",
            "collected_at": "2026-04-10T15:00:00",
            "label": "true_ng",
            "source_type": "patchcore_tile",
            "prefix": "G0F0001",
            "glass_id": "GLS123",
            "image_name": "G0F0001.bmp",
            "inference_record_id": "1001",
            "image_result_id": "5001",
            "tile_idx": "3",
            "edge_defect_id": "",
            "crop_path": "true_ng/G0F0001/crop/20260408_GLS123_G0F0001_tile3.png",
            "heatmap_path": "true_ng/G0F0001/heatmap/20260408_GLS123_G0F0001_tile3.png",
            "ai_score": "0.85",
            "defect_x": "256",
            "defect_y": "256",
            "ric_judgment": "NG",
            "over_review_category": "",
            "over_review_note": "",
            "inference_timestamp": "2026-04-08T10:00:00",
            "status": "ok",
        }
    }
    write_manifest(manifest_path, rows)
    loaded = read_manifest(manifest_path)
    assert loaded == rows


def test_move_existing_sample_to_new_label(tmp_path):
    """label 變更時，實體檔案要從舊目錄 move 到新目錄"""
    from capi_dataset_export import move_sample_files
    base = tmp_path
    old_crop = base / "over_other" / "G0F0001" / "crop" / "a.png"
    old_hm = base / "over_other" / "G0F0001" / "heatmap" / "a.png"
    old_crop.parent.mkdir(parents=True)
    old_hm.parent.mkdir(parents=True)
    old_crop.write_bytes(b"crop")
    old_hm.write_bytes(b"heatmap")

    new_crop_rel, new_hm_rel = move_sample_files(
        base_dir=base,
        old_crop_rel="over_other/G0F0001/crop/a.png",
        old_heatmap_rel="over_other/G0F0001/heatmap/a.png",
        new_label="over_edge_false_positive",
        prefix="G0F0001",
    )

    assert new_crop_rel == "over_edge_false_positive/G0F0001/crop/a.png"
    assert (base / new_crop_rel).read_bytes() == b"crop"
    assert (base / new_hm_rel).read_bytes() == b"heatmap"
    assert not old_crop.exists()
    assert not old_hm.exists()
```

- [ ] **Step 5.2: 執行測試驗證 FAIL**

Run: `python tests/test_dataset_export.py`
Expected: FAIL with ImportError

- [ ] **Step 5.3: 實作 manifest 讀寫與 move**

在 `capi_dataset_export.py` module level 新增：
```python
def read_manifest(manifest_path: Path) -> Dict[str, Dict[str, str]]:
    """讀 manifest.csv 成 {sample_id: row_dict}。檔案不存在回空 dict。"""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("sample_id")
            if sid:
                out[sid] = row
    return out


def write_manifest(manifest_path: Path, rows: Dict[str, Dict[str, str]]) -> None:
    """整批 rewrite manifest.csv（呼叫者須持有 job lock 保證單寫入者）"""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for sid in sorted(rows.keys()):
            row = rows[sid]
            # 確保所有欄位都存在（補空值）
            full = {k: str(row.get(k, "")) for k in MANIFEST_FIELDS}
            writer.writerow(full)
    tmp_path.replace(manifest_path)  # atomic on same filesystem


def move_sample_files(
    base_dir: Path, old_crop_rel: str, old_heatmap_rel: str,
    new_label: str, prefix: str,
) -> Tuple[str, str]:
    """把 crop 與 heatmap 從舊 label 目錄 move 到新 label 目錄。

    Returns:
        (new_crop_rel, new_heatmap_rel) — 兩個新的相對路徑
    """
    base_dir = Path(base_dir)
    old_crop = base_dir / old_crop_rel
    old_hm = base_dir / old_heatmap_rel

    filename = old_crop.name
    new_crop_rel = f"{new_label}/{prefix}/crop/{filename}"
    new_hm_rel = f"{new_label}/{prefix}/heatmap/{filename}"
    new_crop = base_dir / new_crop_rel
    new_hm = base_dir / new_hm_rel

    new_crop.parent.mkdir(parents=True, exist_ok=True)
    new_hm.parent.mkdir(parents=True, exist_ok=True)

    if old_crop.exists():
        shutil.move(str(old_crop), str(new_crop))
    if old_hm.exists():
        shutil.move(str(old_hm), str(new_hm))

    return new_crop_rel, new_hm_rel
```

- [ ] **Step 5.4: 執行測試驗證 PASS**

Run: `python tests/test_dataset_export.py`
Expected: PASS

- [ ] **Step 5.5: Commit**

```bash
git add capi_dataset_export.py tests/test_dataset_export.py
git commit -m "feat(dataset_export): 實作 manifest CSV 讀寫與 label 變更時的檔案 move"
```

---

## Task 6：DatasetExporter.run() 主流程

整合 candidates + crop + manifest + 狀態回報。

**Files:**
- Modify: `capi_dataset_export.py`
- Modify: `tests/test_dataset_export.py`

- [ ] **Step 6.1: 寫端對端測試（用 tmp_path + 生成 fake 原圖與 fake heatmap）**

```python
def test_exporter_run_end_to_end(tmp_path):
    """完整跑一次 run()：生成 fake 原圖與 heatmap → 驗證目錄與 manifest 正確"""
    import cv2

    # 1. 造 fake 原圖
    panel_dir = tmp_path / "panels" / "GLS123"
    panel_dir.mkdir(parents=True)
    fake_img_path = panel_dir / "G0F00000_114438.tif"
    fake_img = np.full((2048, 2048, 3), 150, dtype=np.uint8)
    cv2.imwrite(str(fake_img_path), fake_img)

    # 2. 造 fake heatmap
    heatmap_dir = tmp_path / "heatmaps"
    heatmap_dir.mkdir()
    fake_hm = heatmap_dir / "heatmap_tile3.png"
    cv2.imwrite(str(fake_hm), np.zeros((256, 1024, 3), dtype=np.uint8))
    fake_edge_hm = heatmap_dir / "edge_7001.png"
    cv2.imwrite(str(fake_edge_hm), np.zeros((256, 256, 3), dtype=np.uint8))

    # 3. 準備 fake DB
    row = _make_accuracy_row()
    detail = _make_record_detail()
    # 修正 fixture 的 image_path 指向 fake 檔案
    detail["images"][0]["image_path"] = str(fake_img_path)
    detail["images"][0]["tiles"][0]["heatmap_path"] = str(fake_hm)
    detail["images"][0]["edge_defects"][0]["heatmap_path"] = str(fake_edge_hm)
    # 把炸彈 image 移除，避免路徑失效干擾測試
    detail["images"].pop(1)

    db = FakeDB(accuracy_rows=[row], record_details={1001: detail})
    output = tmp_path / "out"
    exporter = DatasetExporter(db, base_dir=str(output), path_mapping={})

    summary = exporter.run(days=3, include_true_ng=True, skip_existing=True)

    # 4. 驗證目錄
    assert (output / "over_edge_false_positive" / "G0F00000" / "crop"
            / "20260408_GLS123_G0F00000_114438_tile3.png").exists()
    assert (output / "over_edge_false_positive" / "G0F00000" / "heatmap"
            / "20260408_GLS123_G0F00000_114438_tile3.png").exists()
    assert (output / "over_edge_false_positive" / "G0F00000" / "crop"
            / "20260408_GLS123_G0F00000_114438_edge7001.png").exists()

    # 5. 驗證 manifest
    manifest = read_manifest(output / "manifest.csv")
    assert "GLS123_G0F00000_114438_tile3" in manifest
    assert "GLS123_G0F00000_114438_edge7001" in manifest
    assert manifest["GLS123_G0F00000_114438_tile3"]["label"] == "over_edge_false_positive"
    assert manifest["GLS123_G0F00000_114438_tile3"]["status"] == "ok"

    # 6. 驗證 summary
    assert summary.total == 2
    assert summary.labels.get("over_edge_false_positive") == 2


def test_exporter_run_skip_missing_source(tmp_path):
    """原圖不存在 → 樣本寫入 manifest 但 status=skipped_no_source，實體檔不生"""
    row = _make_accuracy_row()
    detail = _make_record_detail()
    detail["images"][0]["image_path"] = str(tmp_path / "does_not_exist.tif")
    detail["images"].pop(1)  # 移除炸彈 image
    db = FakeDB(accuracy_rows=[row], record_details={1001: detail})
    output = tmp_path / "out"
    exporter = DatasetExporter(db, base_dir=str(output), path_mapping={})

    summary = exporter.run(days=3, include_true_ng=True, skip_existing=True)

    manifest = read_manifest(output / "manifest.csv")
    assert manifest["GLS123_G0F00000_114438_tile3"]["status"] == "skipped_no_source"
    assert summary.skipped.get("skipped_no_source", 0) >= 1
    # 沒有任何實體 crop 檔
    assert not list((output / "over_edge_false_positive").rglob("*.png")) if (output / "over_edge_false_positive").exists() else True


def test_exporter_run_second_pass_moves_on_label_change(tmp_path):
    """第一次跑 label=other → 第二次 DB 改成 edge_false_positive → 檔案應 move"""
    import cv2

    panel_dir = tmp_path / "panels" / "GLS123"
    panel_dir.mkdir(parents=True)
    fake_img = panel_dir / "G0F00000_114438.tif"
    cv2.imwrite(str(fake_img), np.full((2048, 2048, 3), 150, dtype=np.uint8))
    fake_hm = tmp_path / "heatmaps" / "heatmap_tile3.png"
    fake_hm.parent.mkdir()
    cv2.imwrite(str(fake_hm), np.zeros((256, 1024, 3), dtype=np.uint8))

    row = _make_accuracy_row(over_review_category="other")
    detail = _make_record_detail()
    detail["images"][0]["image_path"] = str(fake_img)
    detail["images"][0]["tiles"][0]["heatmap_path"] = str(fake_hm)
    detail["images"][0]["edge_defects"] = []  # 簡化：只測 tile 樣本
    detail["images"].pop(1)
    db = FakeDB(accuracy_rows=[row], record_details={1001: detail})
    output = tmp_path / "out"
    exporter = DatasetExporter(db, base_dir=str(output), path_mapping={})

    # 第一次：label = over_other
    exporter.run(days=3, include_true_ng=True, skip_existing=True)
    assert (output / "over_other" / "G0F00000" / "crop"
            / "20260408_GLS123_G0F00000_114438_tile3.png").exists()

    # 模擬使用者改 category
    db._accuracy_rows[0]["over_review_category"] = "edge_false_positive"
    exporter.run(days=3, include_true_ng=True, skip_existing=True)

    # 檔案應 move 到新 label 目錄
    assert (output / "over_edge_false_positive" / "G0F00000" / "crop"
            / "20260408_GLS123_G0F00000_114438_tile3.png").exists()
    assert not (output / "over_other" / "G0F00000" / "crop"
                / "20260408_GLS123_G0F00000_114438_tile3.png").exists()
    manifest = read_manifest(output / "manifest.csv")
    assert manifest["GLS123_G0F00000_114438_tile3"]["label"] == "over_edge_false_positive"
```

- [ ] **Step 6.2: 執行測試驗證 FAIL**

Run: `python tests/test_dataset_export.py`
Expected: FAIL with `NotImplementedError` in `DatasetExporter.run`

- [ ] **Step 6.3: 實作 `DatasetExporter.run()`、`_process_candidate()`、`_resolve_source_path()`**

在 `DatasetExporter` class 內新增（替換 Task 1 的 NotImplementedError stub）：
```python
def run(self, days: int, include_true_ng: bool, skip_existing: bool,
        status_callback=None, cancel_event: Optional[threading.Event] = None
        ) -> JobSummary:
    """主執行入口。

    Args:
        days: 抓最近 N 日 (含今天)
        include_true_ng: 是否包含 AI=NG & RIC=NG 樣本
        skip_existing: True → 已存在且 label 未變的樣本 skip
        status_callback: callable(current, total, last_glass_id) 可選
        cancel_event: threading.Event，set 後會在下一筆前停止
    """
    from capi_server import resolve_unc_path  # lazy import 避免 circular

    started_at = datetime.now()
    job_id = started_at.strftime("job_%Y%m%d_%H%M%S")
    self.base_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = self.base_dir / "manifest.csv"

    # 1. 讀 manifest
    existing = read_manifest(manifest_path)

    # 2. 蒐集 candidates
    candidates = self.collect_candidates(days=days, include_true_ng=include_true_ng)
    total = len(candidates)
    logger.info("Collected %d candidates (days=%d, include_true_ng=%s)",
                total, days, include_true_ng)

    labels_count: Dict[str, int] = {}
    skipped_count: Dict[str, int] = {}

    # 3. 處理每個 candidate
    for idx, cand in enumerate(candidates, start=1):
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Job cancelled at %d/%d", idx, total)
            break

        if status_callback:
            try:
                status_callback(idx, total, cand.glass_id)
            except Exception:
                logger.exception("status_callback error")

        # 解析原圖路徑（UNC → Linux）
        source_path = self._resolve_source_path(cand.image_path, resolve_unc_path)

        new_row = self._process_candidate(
            cand=cand,
            existing_row=existing.get(cand.sample_id),
            skip_existing=skip_existing,
            source_path=source_path,
        )
        if new_row is None:
            continue  # skip_existing 命中，不更動 manifest 該 row
        existing[cand.sample_id] = new_row

        status = new_row["status"]
        if status == "ok":
            labels_count[new_row["label"]] = labels_count.get(new_row["label"], 0) + 1
        else:
            skipped_count[status] = skipped_count.get(status, 0) + 1

    # 4. 寫回 manifest
    write_manifest(manifest_path, existing)

    finished_at = datetime.now()
    return JobSummary(
        job_id=job_id,
        started_at=started_at.isoformat(timespec="seconds"),
        finished_at=finished_at.isoformat(timespec="seconds"),
        duration_sec=(finished_at - started_at).total_seconds(),
        total=sum(labels_count.values()) + sum(skipped_count.values()),
        labels=labels_count,
        skipped=skipped_count,
        output_dir=str(self.base_dir),
    )


def _resolve_source_path(self, image_path: str, resolver) -> Path:
    """套用 path_mapping 轉換，回傳 Path；若 path_mapping 為空或已是本地路徑，直接回原字串"""
    if self.path_mapping:
        try:
            mapped = resolver(image_path, self.path_mapping)
            return Path(mapped)
        except Exception:
            logger.exception("resolve_unc_path failed for %s", image_path)
    return Path(image_path)


def _process_candidate(
    self, cand: SampleCandidate, existing_row: Optional[Dict[str, str]],
    skip_existing: bool, source_path: Path,
) -> Optional[Dict[str, str]]:
    """處理一個 candidate，回傳更新後的 manifest row，或 None（代表 skip 不更動 manifest）"""
    # === 去重/移動 ===
    if existing_row is not None:
        old_label = existing_row.get("label", "")
        old_status = existing_row.get("status", "")

        if old_status == "ok" and old_label == cand.label:
            if skip_existing:
                return None  # 完全相同，skip
            # 不 skip → 重做一次（往下走）

        if old_status == "ok" and old_label != cand.label:
            # label 變了 → move 實體檔 + 更新 row
            new_crop_rel, new_hm_rel = move_sample_files(
                base_dir=self.base_dir,
                old_crop_rel=existing_row.get("crop_path", ""),
                old_heatmap_rel=existing_row.get("heatmap_path", ""),
                new_label=cand.label,
                prefix=cand.prefix,
            )
            updated = dict(existing_row)
            updated["label"] = cand.label
            updated["crop_path"] = new_crop_rel
            updated["heatmap_path"] = new_hm_rel
            updated["collected_at"] = datetime.now().isoformat(timespec="seconds")
            updated["over_review_category"] = cand.over_review_category
            updated["over_review_note"] = cand.over_review_note
            return updated

        if old_status.startswith("skipped_") and skip_existing:
            return None  # 之前 skip 過，不 retry

    # === 新樣本（或強制重做） ===
    row = self._build_row_stub(cand)

    # 讀原圖
    if not source_path.exists():
        row["status"] = "skipped_no_source"
        logger.warning("Source not found: %s (sample_id=%s)", source_path, cand.sample_id)
        return row

    img = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        row["status"] = "skipped_no_source"
        return row
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Crop
    if cand.source_type == "patchcore_tile":
        crop = crop_patchcore_tile(img, cand.tile_x, cand.tile_y, cand.tile_w, cand.tile_h)
        defect_x = cand.tile_x + cand.tile_w // 2
        defect_y = cand.tile_y + cand.tile_h // 2
        sample_key = f"tile{cand.tile_idx}"
    else:  # edge_defect
        # 有效像素 < 25% → skip
        H, W = img.shape[:2]
        half = CROP_SIZE // 2
        valid_w = min(W, cand.edge_center_x + half) - max(0, cand.edge_center_x - half)
        valid_h = min(H, cand.edge_center_y + half) - max(0, cand.edge_center_y - half)
        if max(0, valid_w) * max(0, valid_h) < (CROP_SIZE * CROP_SIZE) * 0.25:
            row["status"] = "skipped_out_of_bounds"
            return row
        crop = crop_edge_defect(img, cand.edge_center_x, cand.edge_center_y)
        defect_x = cand.edge_center_x
        defect_y = cand.edge_center_y
        sample_key = f"edge{cand.edge_defect_id}"

    # Heatmap 必須存在
    src_hm = Path(cand.src_heatmap_path) if cand.src_heatmap_path else None
    if src_hm is None or not src_hm.exists():
        row["status"] = "skipped_no_heatmap"
        return row

    # 決定目的路徑
    filename = build_sample_filename(
        glass_id=cand.glass_id, image_name=cand.image_name,
        sample_key=sample_key, inference_timestamp=cand.inference_timestamp,
    )
    crop_rel = f"{cand.label}/{cand.prefix}/crop/{filename}"
    hm_rel = f"{cand.label}/{cand.prefix}/heatmap/{filename}"
    crop_dst = self.base_dir / crop_rel
    hm_dst = self.base_dir / hm_rel
    crop_dst.parent.mkdir(parents=True, exist_ok=True)
    hm_dst.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(crop_dst), crop)
    shutil.copy2(str(src_hm), str(hm_dst))

    row["crop_path"] = crop_rel
    row["heatmap_path"] = hm_rel
    row["defect_x"] = str(defect_x)
    row["defect_y"] = str(defect_y)
    row["status"] = "ok"
    return row


def _build_row_stub(self, cand: SampleCandidate) -> Dict[str, str]:
    """從 candidate 組出 manifest row 的初始值（尚未 crop/copy）"""
    return {
        "sample_id": cand.sample_id,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "label": cand.label,
        "source_type": cand.source_type,
        "prefix": cand.prefix,
        "glass_id": cand.glass_id,
        "image_name": cand.image_name,
        "inference_record_id": str(cand.inference_record_id or ""),
        "image_result_id": str(cand.image_result_id or ""),
        "tile_idx": str(cand.tile_idx) if cand.tile_idx is not None else "",
        "edge_defect_id": str(cand.edge_defect_id) if cand.edge_defect_id is not None else "",
        "crop_path": "",
        "heatmap_path": "",
        "ai_score": f"{cand.ai_score:.4f}",
        "defect_x": "",
        "defect_y": "",
        "ric_judgment": cand.ric_judgment,
        "over_review_category": cand.over_review_category,
        "over_review_note": cand.over_review_note,
        "inference_timestamp": cand.inference_timestamp,
        "status": "",
    }
```

- [ ] **Step 6.4: 執行測試驗證 PASS**

Run: `python tests/test_dataset_export.py`
Expected: 全部測試 PASS（含三個 end-to-end）

- [ ] **Step 6.5: Commit**

```bash
git add capi_dataset_export.py tests/test_dataset_export.py
git commit -m "feat(dataset_export): 實作 run() 主流程（crop + heatmap 拷貝 + manifest + label move）"
```

---

## Task 7：Web API 端點 + 背景 Job 狀態

在 `capi_web.py` 加入 4 個 endpoint + 模組級 job state，啟動背景 thread 執行 exporter。

**Files:**
- Modify: `capi_web.py`

- [ ] **Step 7.1: 在 capi_web.py 的 module top（第 30 行附近 import 區塊）新增 import**

找到 `import threading` 那行，確保已 import。在下方的 helper 區塊（約 90 行前）加入：
```python
from capi_dataset_export import DatasetExporter, JobSummary, JOB_STATE_IDLE, JOB_STATE_RUNNING, JOB_STATE_COMPLETED, JOB_STATE_FAILED, JOB_STATE_CANCELLED
```

- [ ] **Step 7.2: 在 `create_web_server` 內初始化 dataset_export job state**

找到 `capi_web.py:2908` `CAPIWebHandler._rerun_tasks = {}` 那行後方加入：
```python
CAPIWebHandler._dataset_export_state = {
    "lock": threading.Lock(),
    "current_job": None,         # dict: job_id, state, current, total, last_glass_id, started_at
    "cancel_event": threading.Event(),
    "last_summary": None,        # 最近一次完成的 JobSummary
}
```

- [ ] **Step 7.3: 實作 4 個 handler method**

在 `CAPIWebHandler` class 內加入（建議放在 `_rerun_worker` 附近，約 2770 行左右）：
```python
# ==== Dataset Export Endpoints ====

@classmethod
def _dataset_export_worker(cls, days: int, include_true_ng: bool,
                           skip_existing: bool, output_dir: str):
    """背景執行 DatasetExporter.run()"""
    state = cls._dataset_export_state
    try:
        server_inst = cls._capi_server_instance
        path_mapping = getattr(server_inst, "path_mapping", {}) if server_inst else {}

        def status_callback(current, total, last_glass_id):
            with state["lock"]:
                if state["current_job"]:
                    state["current_job"]["current"] = current
                    state["current_job"]["total"] = total
                    state["current_job"]["last_glass_id"] = last_glass_id

        exporter = DatasetExporter(
            db=cls.db, base_dir=output_dir, path_mapping=path_mapping,
        )
        summary = exporter.run(
            days=days, include_true_ng=include_true_ng,
            skip_existing=skip_existing,
            status_callback=status_callback,
            cancel_event=state["cancel_event"],
        )
        with state["lock"]:
            if state["current_job"]:
                state["current_job"]["state"] = (
                    JOB_STATE_CANCELLED if state["cancel_event"].is_set() else JOB_STATE_COMPLETED
                )
            state["last_summary"] = summary
    except Exception as e:
        logger.exception("dataset_export worker failed")
        with state["lock"]:
            if state["current_job"]:
                state["current_job"]["state"] = JOB_STATE_FAILED
                state["current_job"]["error"] = str(e)
    finally:
        with state["lock"]:
            state["cancel_event"].clear()

def _handle_dataset_export_start(self):
    """POST /api/dataset_export/start"""
    try:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        import json as _json
        data = _json.loads(body) if body else {}
    except Exception:
        self._send_json({"error": "invalid JSON body"}, status=400)
        return

    days = int(data.get("days", 3))
    include_true_ng = bool(data.get("include_true_ng", True))
    skip_existing = bool(data.get("skip_existing", True))

    # 決定 output_dir
    server_inst = self._capi_server_instance
    default_cfg = {}
    if server_inst:
        default_cfg = server_inst.server_config.get("dataset_export", {})
    output_dir = data.get("output_dir") or default_cfg.get("base_dir") or "./datasets/over_review"
    min_free_gb = float(default_cfg.get("min_free_space_gb", 1))

    state = self._dataset_export_state
    with state["lock"]:
        if state["current_job"] and state["current_job"].get("state") == JOB_STATE_RUNNING:
            self._send_json({
                "error": "job_already_running",
                "current_job_id": state["current_job"].get("job_id"),
            }, status=409)
            return

        # 磁碟空間檢查
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            free_bytes = shutil.disk_usage(str(output_path)).free
            if free_bytes < min_free_gb * (1024 ** 3):
                self._send_json({
                    "error": "insufficient_disk_space",
                    "free_gb": round(free_bytes / (1024 ** 3), 2),
                    "required_gb": min_free_gb,
                }, status=409)
                return
        except Exception as e:
            self._send_json({"error": f"cannot access output_dir: {e}"}, status=400)
            return

        job_id = datetime.now().strftime("job_%Y%m%d_%H%M%S")
        state["cancel_event"].clear()
        state["current_job"] = {
            "job_id": job_id,
            "state": JOB_STATE_RUNNING,
            "current": 0,
            "total": 0,
            "last_glass_id": "",
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }

    thread = threading.Thread(
        target=CAPIWebHandler._dataset_export_worker,
        args=(days, include_true_ng, skip_existing, output_dir),
        daemon=True,
        name=f"dataset-export-{job_id}",
    )
    thread.start()
    self._send_json({"job_id": job_id, "started_at": state["current_job"]["started_at"]})

def _handle_dataset_export_status(self):
    """GET /api/dataset_export/status"""
    state = self._dataset_export_state
    with state["lock"]:
        job = state["current_job"]
        if not job:
            self._send_json({"state": JOB_STATE_IDLE})
            return
        resp = dict(job)
        if resp.get("started_at"):
            started = datetime.fromisoformat(resp["started_at"])
            resp["elapsed_sec"] = round((datetime.now() - started).total_seconds(), 1)
        self._send_json(resp)

def _handle_dataset_export_summary(self, job_id: str):
    """GET /api/dataset_export/summary/<job_id>"""
    state = self._dataset_export_state
    with state["lock"]:
        summary = state["last_summary"]
        if not summary or summary.job_id != job_id:
            self._send_json({"error": "not_found"}, status=404)
            return
        from dataclasses import asdict as _asdict
        self._send_json(_asdict(summary))

def _handle_dataset_export_cancel(self):
    """POST /api/dataset_export/cancel"""
    state = self._dataset_export_state
    with state["lock"]:
        if not state["current_job"] or state["current_job"].get("state") != JOB_STATE_RUNNING:
            self._send_json({"error": "no_running_job"}, status=404)
            return
        state["cancel_event"].set()
    self._send_json({"ok": True})
```

- [ ] **Step 7.4: 把 4 個路徑接到 do_GET / do_POST 路由分派**

找到 `do_GET` 與 `do_POST` 中現有 `/api/ric/over-review` 的路由處理（搜尋 `over-review`）。在同一個 elif/if 鏈裡加入：

**do_POST 內新增：**
```python
elif self.path == "/api/dataset_export/start":
    self._handle_dataset_export_start()
    return
elif self.path == "/api/dataset_export/cancel":
    self._handle_dataset_export_cancel()
    return
```

**do_GET 內新增：**
```python
elif self.path == "/api/dataset_export/status":
    self._handle_dataset_export_status()
    return
elif self.path.startswith("/api/dataset_export/summary/"):
    job_id = self.path.split("/api/dataset_export/summary/", 1)[1]
    self._handle_dataset_export_summary(job_id)
    return
```

- [ ] **Step 7.5: 手動煙霧測試（啟 server 打 API）**

```bash
cd C:/Users/rh.syu/Desktop/CAPI01_AD
python capi_server.py --config server_config_local.yaml
```

另開 terminal：
```bash
# 1. 啟動 job
curl -X POST http://localhost:8080/api/dataset_export/start \
     -H "Content-Type: application/json" \
     -d '{"days":3,"include_true_ng":true,"skip_existing":true}'

# 2. 查 status
curl http://localhost:8080/api/dataset_export/status

# 3. 若有 job_id，查 summary
curl http://localhost:8080/api/dataset_export/summary/job_20260410_...
```

Expected: 
- Start 回 `{job_id, started_at}`
- Status 跑完後 `state=completed`
- Summary 回 `{labels:{...}, skipped:{...}, total, duration_sec, output_dir}`
- 第二次 start 若 job 還在跑會回 409

- [ ] **Step 7.6: Commit**

```bash
git add capi_web.py
git commit -m "feat(dataset_export): 新增 4 個 API 端點 + 背景 job 狀態管理"
```

---

## Task 8：前端 — 過檢 Review tab 加按鈕 + Modal + polling JS

**Files:**
- Modify: `templates/ric_report.html`

- [ ] **Step 8.1: 定位過檢 Review tab 容器**

用 Grep 找：
```
grep -n "overReview\|over_review\|renderOverReviewTab\|過檢" templates/ric_report.html
```
記下 tab panel 的 div id 與 render 函式名稱。

- [ ] **Step 8.2: 在過檢 Review tab 容器的 render 函式最前面加入按鈕**

在 `renderOverReviewTab()` 函式（或對應的 tab render 入口）最前面、既有內容之前插入：
```html
<div class="mb-3">
    <button type="button" class="btn btn-primary" id="btnExportDataset">
        匯出訓練資料集
    </button>
</div>
```

（若 render 函式是字串拼接 HTML，就把這段字串 prepend 到輸出；若是 DOM append，就 createElement）

- [ ] **Step 8.3: 在 template 底部（`</body>` 前）加入 Modal 容器**

```html
<!-- Dataset Export Modal -->
<div class="modal fade" id="datasetExportModal" tabindex="-1">
  <div class="modal-dialog">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title">匯出訓練資料集</h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
      </div>
      <div class="modal-body">
        <!-- 表單 -->
        <div id="dsExportForm">
          <div class="mb-2">
            <label class="form-label">天數</label>
            <input type="number" class="form-control" id="dsExportDays" value="3" min="1" max="30">
          </div>
          <div class="mb-2">
            <label class="form-label">輸出目錄（留空使用 server_config 預設）</label>
            <input type="text" class="form-control" id="dsExportOutputDir" placeholder="/data/capi_ai/datasets/over_review">
          </div>
          <div class="form-check mb-2">
            <input class="form-check-input" type="checkbox" id="dsExportIncludeTrueNg" checked>
            <label class="form-check-label" for="dsExportIncludeTrueNg">包含真 NG 樣本 (true_ng)</label>
          </div>
          <div class="form-check mb-2">
            <input class="form-check-input" type="checkbox" id="dsExportSkipExisting" checked>
            <label class="form-check-label" for="dsExportSkipExisting">跳過已存在的樣本</label>
          </div>
        </div>
        <!-- 進度 -->
        <div id="dsExportProgress" style="display:none;">
          <p>Job ID: <span id="dsExportJobId"></span></p>
          <div class="progress mb-2"><div id="dsExportBar" class="progress-bar" style="width:0%;"></div></div>
          <p id="dsExportCurrentInfo"></p>
        </div>
        <!-- Summary -->
        <div id="dsExportSummary" style="display:none;">
          <h6>匯出完成</h6>
          <pre id="dsExportSummaryText" style="background:#f8f9fa;padding:10px;"></pre>
        </div>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">關閉</button>
        <button type="button" class="btn btn-warning" id="btnDsExportCancel" style="display:none;">取消 Job</button>
        <button type="button" class="btn btn-primary" id="btnDsExportStart">開始匯出</button>
      </div>
    </div>
  </div>
</div>

<script>
(function() {
  let dsExportPollTimer = null;

  document.addEventListener("click", function(ev) {
    if (ev.target && ev.target.id === "btnExportDataset") {
      // 重置 modal 狀態
      document.getElementById("dsExportForm").style.display = "";
      document.getElementById("dsExportProgress").style.display = "none";
      document.getElementById("dsExportSummary").style.display = "none";
      document.getElementById("btnDsExportStart").style.display = "";
      document.getElementById("btnDsExportCancel").style.display = "none";
      const m = new bootstrap.Modal(document.getElementById("datasetExportModal"));
      m.show();
    }
  });

  document.getElementById("btnDsExportStart").addEventListener("click", async function() {
    const days = parseInt(document.getElementById("dsExportDays").value || "3", 10);
    const outputDir = document.getElementById("dsExportOutputDir").value.trim();
    const body = {
      days: days,
      include_true_ng: document.getElementById("dsExportIncludeTrueNg").checked,
      skip_existing: document.getElementById("dsExportSkipExisting").checked,
    };
    if (outputDir) body.output_dir = outputDir;

    try {
      const resp = await fetch("/api/dataset_export/start", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      if (!resp.ok) {
        alert("啟動失敗: " + (data.error || resp.status));
        return;
      }
      // 切換到進度畫面
      document.getElementById("dsExportForm").style.display = "none";
      document.getElementById("dsExportProgress").style.display = "";
      document.getElementById("btnDsExportStart").style.display = "none";
      document.getElementById("btnDsExportCancel").style.display = "";
      document.getElementById("dsExportJobId").textContent = data.job_id;
      startDsExportPolling(data.job_id);
    } catch (e) {
      alert("網路錯誤: " + e);
    }
  });

  document.getElementById("btnDsExportCancel").addEventListener("click", async function() {
    await fetch("/api/dataset_export/cancel", {method: "POST"});
  });

  function startDsExportPolling(jobId) {
    if (dsExportPollTimer) clearInterval(dsExportPollTimer);
    dsExportPollTimer = setInterval(async function() {
      try {
        const r = await fetch("/api/dataset_export/status");
        const s = await r.json();
        if (s.state === "running") {
          const pct = s.total > 0 ? (s.current * 100 / s.total) : 0;
          document.getElementById("dsExportBar").style.width = pct + "%";
          document.getElementById("dsExportCurrentInfo").textContent =
            `${s.current} / ${s.total}  (${s.last_glass_id || ""})  已耗時 ${s.elapsed_sec || 0}s`;
        } else if (s.state === "completed" || s.state === "cancelled" || s.state === "failed") {
          clearInterval(dsExportPollTimer);
          dsExportPollTimer = null;
          document.getElementById("btnDsExportCancel").style.display = "none";
          if (s.state === "completed") {
            const sr = await fetch("/api/dataset_export/summary/" + jobId);
            const sm = await sr.json();
            document.getElementById("dsExportProgress").style.display = "none";
            document.getElementById("dsExportSummary").style.display = "";
            document.getElementById("dsExportSummaryText").textContent = JSON.stringify(sm, null, 2);
          } else {
            document.getElementById("dsExportCurrentInfo").textContent =
              `Job ${s.state}: ${s.error || ""}`;
          }
        }
      } catch (e) {
        console.error("poll error", e);
      }
    }, 2000);
  }
})();
</script>
```

- [ ] **Step 8.4: 手動瀏覽器測試**

1. 啟動 server: `python capi_server.py --config server_config_local.yaml`
2. 開 `http://localhost:8080/ric`
3. 切到「過檢 Review」tab，應看到「匯出訓練資料集」按鈕
4. 點按鈕 → modal 出現 → 填參數 → 開始匯出
5. 進度條應更新、完成後顯示 summary JSON
6. 檢查 `./datasets/over_review/` 目錄結構正確 + `manifest.csv` 有 row

- [ ] **Step 8.5: Commit**

```bash
git add templates/ric_report.html
git commit -m "feat(dataset_export): RIC 過檢 Review tab 新增匯出按鈕 + modal + 進度 polling"
```

---

## Task 9：整合測試 + 使用者驗收

- [ ] **Step 9.1: 跑完整單元測試**

```bash
python tests/test_dataset_export.py
```
Expected: 全部 PASS

- [ ] **Step 9.2: 本地 smoke test — days=1**

1. 啟動 server
2. 瀏覽器觸發匯出，`days=1`
3. 確認：
   - manifest.csv 存在且有 row
   - 目錄結構符合 spec（label/prefix/{crop,heatmap}）
   - crop 檔是 512×512
   - heatmap 檔尺寸與 DB 紀錄一致

- [ ] **Step 9.3: 驗證增量蒐集（重跑一次）**

再點一次「匯出」（同樣參數）：
- 狀態應顯示「skip 既有」
- manifest.csv 樣本數不變
- 目錄檔案數不變

- [ ] **Step 9.4: 驗證 label 變更 move**

1. 到 `/ric` 過檢 Review tab，把某筆樣本的 category 改成不同值（例：從 other 改 edge_false_positive）
2. 重跑匯出
3. 確認：
   - 舊目錄下該 sample 檔案已消失
   - 新目錄下該 sample 檔案存在
   - manifest.csv 對應 row 的 `label` 欄位已更新

- [ ] **Step 9.5: 回報給使用者**

報告項目：
- 各 label 實際蒐集到的樣本數
- Skip 統計
- 輸出目錄路徑
- 有沒有異常 log

- [ ] **Step 9.6: Commit final notes（若有 tweak）**

---

## 自我檢查（Self-Review）

### Spec coverage
- ✅ Spec §3 架構圖 → Task 6-7（exporter run + web API）
- ✅ Spec §4 目錄結構 → Task 6（crop_rel / hm_rel 組裝）
- ✅ Spec §5 manifest schema → Task 1（MANIFEST_FIELDS）+ Task 5（讀寫）
- ✅ Spec §5.1 增量蒐集 → Task 6（`_process_candidate` 的 existing_row 分支）
- ✅ Spec §6.1 PatchCore tile crop → Task 2 `crop_patchcore_tile`
- ✅ Spec §6.2 Edge defect crop → Task 2 `crop_edge_defect`
- ✅ Spec §6.3 heatmap 複製 → Task 6 `shutil.copy2`
- ✅ Spec §7 4 個 API 端點 → Task 7
- ✅ Spec §8 前端 modal + polling → Task 8
- ✅ Spec §9 錯誤處理 → Task 6（skip 分支）+ Task 7（磁碟檢查、job lock）
- ✅ Spec §10 檔案組織 → Task 1-8 與 spec 完全一致
- ✅ Spec §11 Out of Scope → 計畫沒做分類 AI、cron、即時觸發、歷史回溯、heatmap 重生

### Type consistency
- `SampleCandidate` dataclass 欄位在 Task 1 定義，Task 4/6 使用的欄位名全對上
- `determine_label` 回傳 `Optional[str]`，Task 4 正確 handle None（`continue`）
- `JobSummary` 欄位在 Task 1 定義，Task 6 產生、Task 7 `asdict` 轉 JSON
- `build_sample_filename` / `build_sample_id` 簽名一致（Task 3 定義、Task 6 使用）
- `move_sample_files` 回 `(new_crop_rel, new_heatmap_rel)`，Task 6 正確解構

### Placeholder scan
- 無 TBD / TODO / 「write tests for the above」/ 未填 code block
- 所有 SQL、函式簽名、相對路徑都是實際可執行值

### 延後到實作階段的 spec 疑問（無阻礙）
- Spec §3.1 `client_accuracy_records ↔ inference_records` JOIN key：已在 Task 4 探索階段確認，直接複用 `get_client_accuracy_records()` 回傳的 `inference_record_id`
- Spec §5 Schema edge 樣本 `ai_score`：使用 `edge_defect_results.max_diff`（Task 4 `_flatten_record_to_candidates` 實作）

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-10-over-review-dataset-collector.md`.**
