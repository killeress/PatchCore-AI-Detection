"""
過檢訓練資料集蒐集工具

從 RIC 回填結果蒐集 AI=NG&RIC=NG (true_ng) 與 AI=NG&RIC=OK 已回填 category 的樣本，
輸出 crop 對到分類目錄結構 + manifest.csv，供後續分類 AI 訓練或 CV 後處理評估。
(heatmap 已停止蒐集；manifest 保留 heatmap_path 欄位但新樣本留空以維持向後相容)

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
    "surface_dirt": "over_surface_dirt",
    "bubble": "over_bubble",
    "aoi_ai_false_positive": "over_aoi_ai_false_positive",
    "dust_mask_incomplete": "over_dust_mask_incomplete",
    "other": "over_other",
}

TRUE_NG_LABEL = "true_ng"

# Label 中文說明（供 web UI 呈現與翻圖時快速理解類別）
LABEL_ZH = {
    "true_ng": "真實 NG",
    "over_edge_false_positive": "邊緣誤判",
    "over_within_spec": "規格內",
    "over_overexposure": "曝光過度",
    "over_surface_scratch": "表面刮痕",
    "over_surface_dirt": "表面汙",
    "over_bubble": "氣泡",
    "over_aoi_ai_false_positive": "AOI/AI 都誤判",
    "over_dust_mask_incomplete": "灰塵屏蔽需完善",
    "over_other": "其他",
}

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

# 黑畫面光源 prefix：該光源條件下面板是關燈狀態，整張圖無訓練價值，直接以檔名過濾
BLACK_IMAGE_PREFIX = "B0F"


def parse_datastr_per_prefix(datastr: str) -> Dict[str, str]:
    """解析 RIC DATASTR，回傳 {prefix: "OK"|"NG"} dict。

    例: "W0F00000,OK;WGF50500,OK;G0F00000,NG;3;"
        → {"W0F00000": "OK", "WGF50500": "OK", "G0F00000": "NG"}

    末尾純數字 (張數計數) 會被略過。空字串回傳空 dict。
    """
    out: Dict[str, str] = {}
    if not datastr:
        return out
    for part in datastr.strip().rstrip(";").split(";"):
        part = part.strip()
        if not part or part.isdigit():
            continue
        if "," not in part:
            continue
        prefix, result = part.rsplit(",", 1)
        out[prefix.strip()] = result.strip().upper()
    return out


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


def list_job_dirs(base_dir: Path) -> List[Path]:
    """列出 base_dir 下的 job 資料夾（子目錄且含 manifest.csv），依目錄名升冪排序。

    不含 base_dir 根目錄層（legacy 扁平結構）的 manifest.csv。
    不存在的 base_dir 回空 list。
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []
    out: List[Path] = []
    for child in sorted(base_dir.iterdir(), key=lambda p: p.name):
        if child.is_dir() and (child / "manifest.csv").exists():
            out.append(child)
    return out


def load_known_sample_ids(base_dir: Path) -> set:
    """Union 所有 job 資料夾 manifest.csv 內的 sample_id。

    供 DatasetExporter.run 判定新 candidate 用：只要任一 prior job 有此 sample_id → skip。
    status 不限制（即使 skipped_* 也算已處理過），避免使用者明明看過該 sample 卻
    被下次 run 重新蒐集而產生重複干擾。
    """
    ids: set = set()
    for job_dir in list_job_dirs(base_dir):
        rows = read_manifest(job_dir / "manifest.csv")
        ids.update(rows.keys())
    return ids


def move_sample_files(
    base_dir: Path, old_crop_rel: str, old_heatmap_rel: str,
    new_label: str, prefix: str,
) -> Tuple[str, str]:
    """把 crop 與 heatmap 從舊 label 目錄 move 到新 label 目錄。

    Returns:
        (new_crop_rel, new_heatmap_rel) — 兩個新的相對路徑
    """
    base_dir = Path(base_dir)

    # 空字串時 base_dir / "" == base_dir (即整個 out 目錄)，會害 shutil.move 誤移，必須先檔掉
    if not old_crop_rel:
        new_crop_rel = ""
    else:
        old_crop = base_dir / old_crop_rel
        filename = old_crop.name
        new_crop_rel = f"{new_label}/{prefix}/crop/{filename}"
        new_crop = base_dir / new_crop_rel
        new_crop.parent.mkdir(parents=True, exist_ok=True)
        if old_crop.exists():
            shutil.move(str(old_crop), str(new_crop))

    if not old_heatmap_rel:
        # heatmap 已停止蒐集，舊樣本也可能沒有 heatmap_path → 跳過 heatmap move
        new_hm_rel = ""
    else:
        old_hm = base_dir / old_heatmap_rel
        filename_hm = old_hm.name
        new_hm_rel = f"{new_label}/{prefix}/heatmap/{filename_hm}"
        new_hm = base_dir / new_hm_rel
        new_hm.parent.mkdir(parents=True, exist_ok=True)
        if old_hm.exists():
            shutil.move(str(old_hm), str(new_hm))

    return new_crop_rel, new_hm_rel


def get_valid_labels() -> List[str]:
    """回傳所有合法 label 目錄名（for 手動 relabel 驗證 + 前端下拉）"""
    return [TRUE_NG_LABEL] + sorted(OVER_LABEL_MAP.values())


def delete_sample(
    base_dir: Path, manifest: Dict[str, Dict[str, str]], sample_id: str
) -> bool:
    """從 manifest 與 disk 刪除一個樣本（用於人工 curation）。

    Caller 須在呼叫前後持 lock 並負責把 manifest 寫回 (write_manifest)。

    Returns:
        True 表示有刪除動作，False 表示 sample_id 不存在於 manifest。
    """
    base_dir = Path(base_dir)
    if sample_id not in manifest:
        return False
    row = manifest[sample_id]
    for rel_key in ("crop_path", "heatmap_path"):
        rel = row.get(rel_key) or ""
        if rel:
            full = base_dir / rel
            if full.exists():
                try:
                    full.unlink()
                except OSError:
                    logger.exception("Failed to unlink %s: %s", rel_key, full)
    del manifest[sample_id]
    return True


def relabel_sample(
    base_dir: Path, manifest: Dict[str, Dict[str, str]],
    sample_id: str, new_label: str,
) -> Optional[Dict[str, str]]:
    """把樣本的 label 改成 new_label：實體檔 move + manifest row 更新。

    Args:
        new_label: 必須在 get_valid_labels() 內，否則 raise ValueError

    Returns:
        更新後的 row dict，或 None 若 sample_id 不存在於 manifest。
    """
    if new_label not in get_valid_labels():
        raise ValueError(f"Invalid label: {new_label}")
    base_dir = Path(base_dir)
    if sample_id not in manifest:
        return None
    row = dict(manifest[sample_id])  # copy 避免 in-place 副作用
    old_label = row.get("label", "")
    if old_label == new_label:
        return row  # no-op

    new_crop_rel, new_hm_rel = move_sample_files(
        base_dir=base_dir,
        old_crop_rel=row.get("crop_path", ""),
        old_heatmap_rel=row.get("heatmap_path", ""),
        new_label=new_label,
        prefix=row.get("prefix", ""),
    )
    row["label"] = new_label
    row["crop_path"] = new_crop_rel
    row["heatmap_path"] = new_hm_rel
    row["collected_at"] = datetime.now().isoformat(timespec="seconds")
    manifest[sample_id] = row
    return row


# ---- Crop Tool Functions ----

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
    """以 (cx, cy) 為中心切出 CROP_SIZE × CROP_SIZE，clamp 後再 pad 保持中心對齐。

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
    # 診斷用：collect_candidates 每階段的過濾計數（幫助排查為何 total=0）
    diagnostics: Dict[str, int] = field(default_factory=dict)


# ---- Exporter（實作在後續 Task 加入） ----

class DatasetExporter:
    """過檢訓練資料集蒐集器（pure logic，不含 HTTP）"""

    def __init__(self, db, base_dir: str, path_mapping: Dict[str, str]):
        self.db = db
        self.base_dir = Path(base_dir).resolve()
        self.path_mapping = path_mapping

    def collect_candidates(self, days: int, include_true_ng: bool) -> List[SampleCandidate]:
        """便利包裝：只回傳 candidates，忽略診斷統計。"""
        candidates, _diag = self.collect_candidates_with_diagnostics(days, include_true_ng)
        return candidates

    def collect_candidates_with_diagnostics(
        self, days: int, include_true_ng: bool
    ) -> Tuple[List[SampleCandidate], Dict[str, int]]:
        """從 DB 讀最近 N 日的 client_accuracy_records，展開成 candidate 清單，
        同時記錄每個過濾階段掉了多少筆，方便排查為何 total=0。

        篩選規則：
          - result_ai == "NG"
          - 若 result_ric == "NG" 且 include_true_ng → 收為 true_ng
          - 若 result_ric == "OK" 且 over_review_category 已填 → 收為 over_<category>
          - 其他情況跳過

        過濾：
          - image_results.is_bomb == 1 整張跳過
          - tile_results.is_bomb == 1 該 tile 跳過
          - tile_results.is_anomaly == 0 跳過（只蒐集 NG tile）

        Returns:
            (candidates, diagnostics) — diagnostics 是 dict，key 包含：
              total_accuracy_rows, ai_not_ng, ric_ng_kept, ric_ng_skipped_opt_out,
              ric_ok_filled, ric_ok_empty, ric_other, missing_inference_id,
              missing_record_detail, bomb_record_skipped, final_candidates
        """
        end = datetime.now().date()
        start = end - timedelta(days=days - 1)
        rows = self.db.get_client_accuracy_records(
            start_date=start.isoformat(), end_date=end.isoformat()
        )

        diag: Dict[str, int] = {
            "date_start": start.isoformat(),
            "date_end": end.isoformat(),
            "total_accuracy_rows": len(rows),
            "ai_not_ng": 0,
            "ric_ng_kept": 0,
            "ric_ng_skipped_opt_out": 0,
            "ric_ok_filled": 0,
            "ric_ok_empty": 0,
            "ric_other": 0,
            "missing_inference_id": 0,
            "missing_record_detail": 0,
            "images_bomb_skipped": 0,      # image_results.is_bomb=1 被整張跳過
            "images_b0f_skipped": 0,       # 黑光源 B0F 檔名被跳過
            "images_ric_ok_skipped": 0,    # true_ng panel 內 RIC 個別判 OK 的 prefix 被跳過
            "tiles_bomb_skipped": 0,       # tile_results.is_bomb=1 被跳過
            "tiles_not_anomaly": 0,        # tile_results.is_anomaly=0 被跳過
            "edges_bomb_skipped": 0,       # edge_defect_results.is_bomb=1 被跳過
            "final_candidates": 0,
        }

        candidates: List[SampleCandidate] = []
        for row in rows:
            if row.get("result_ai") != "NG":
                diag["ai_not_ng"] += 1
                continue

            ric = row.get("result_ric") or ""
            over_category = row.get("over_review_category")

            if ric == "NG":
                if not include_true_ng:
                    diag["ric_ng_skipped_opt_out"] += 1
                    continue
                label = TRUE_NG_LABEL
                diag["ric_ng_kept"] += 1
            elif ric == "OK":
                if not over_category:
                    diag["ric_ok_empty"] += 1
                    continue  # 未回填的跳過
                try:
                    label = determine_label(ric, over_category)
                except ValueError as e:
                    logger.warning("Skip unknown over_review category: %s", e)
                    diag["ric_ok_empty"] += 1
                    continue
                if label is None:
                    diag["ric_ok_empty"] += 1
                    continue
                diag["ric_ok_filled"] += 1
            else:
                diag["ric_other"] += 1
                continue

            inference_record_id = row.get("inference_record_id")
            if not inference_record_id:
                logger.warning("No inference_record linked: pnl_id=%s time=%s",
                               row.get("pnl_id"), row.get("time_stamp"))
                diag["missing_inference_id"] += 1
                continue

            detail = self.db.get_record_detail(inference_record_id)
            if not detail:
                diag["missing_record_detail"] += 1
                continue

            flattened = self._flatten_record_to_candidates(
                detail=detail, label=label, row=row, diag=diag,
            )
            candidates.extend(flattened)

        diag["final_candidates"] = len(candidates)
        logger.info("collect_candidates diagnostics: %s", diag)
        return candidates, diag

    def _flatten_record_to_candidates(
        self, detail: Dict, label: str, row: Dict,
        diag: Optional[Dict[str, int]] = None,
    ) -> List[SampleCandidate]:
        """把一筆 inference_record 展開成多個 SampleCandidate（依 image / tile / edge）。

        炸彈過濾策略：
          - 不看 inference_records.client_bomb_info (實測正式機幾乎每筆都有值，
            這個欄位是 AOI 機台傳來的炸彈參考座標列表，非「此片為炸彈測試」指標)
          - image_results.is_bomb == 1 整張跳過 (capi_server.py:549 判定為「純炸彈」時設)
          - tile_results.is_bomb == 1 該 tile 跳過 (check_bomb_match 個別標註)
          - edge_defect_results.is_bomb == 1 該 edge 跳過 (check_bomb_match 個別標註)

        Args:
            diag: 可選的診斷計數器 dict，傳入時會更新 images_bomb_skipped /
                  images_b0f_skipped / tiles_bomb_skipped / tiles_not_anomaly
        """
        out: List[SampleCandidate] = []

        def _bump(key: str) -> None:
            if diag is not None:
                diag[key] = diag.get(key, 0) + 1

        glass_id = detail.get("glass_id") or row.get("pnl_id") or ""
        inference_timestamp = detail.get("request_time") or row.get("time_stamp") or ""
        record_id = detail.get("id")

        # 解析 DATASTR 取得 RIC 對各 prefix 的個別判定 (W0F=OK, G0F=NG ...)
        # 只在 true_ng 蒐集時才生效：panel 雖整體 NG，但某些光源 prefix 是 OK，
        # 那些 image 不該被當 NG 樣本蒐集。空 DATASTR → 不做此過濾 (回退舊行為)。
        ric_per_prefix = parse_datastr_per_prefix(row.get("datastr") or "")

        for img in detail.get("images") or []:
            if img.get("is_bomb"):
                _bump("images_bomb_skipped")
                continue
            image_name = img.get("image_name") or ""
            # 黑光源圖 (B0F) 無訓練價值，整張跳過
            if image_name.startswith(BLACK_IMAGE_PREFIX):
                _bump("images_b0f_skipped")
                continue
            image_path = img.get("image_path") or ""
            image_result_id = img.get("id")
            prefix = extract_prefix(image_name)

            # true_ng 樣本：只收 RIC 在 DATASTR 內亦判 NG 的 prefix
            if label == TRUE_NG_LABEL and ric_per_prefix:
                if ric_per_prefix.get(prefix) != "NG":
                    _bump("images_ric_ok_skipped")
                    continue

            # PatchCore tile 樣本
            for tile in img.get("tiles") or []:
                if tile.get("is_bomb"):
                    _bump("tiles_bomb_skipped")
                    continue
                if not tile.get("is_anomaly"):
                    _bump("tiles_not_anomaly")
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
                if edge.get("is_bomb"):
                    _bump("edges_bomb_skipped")
                    continue
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

    def run(self, days: int, include_true_ng: bool, skip_existing: bool,
            status_callback=None, cancel_event=None) -> JobSummary:
        """執行一次 export job。

        與舊版差異：
          - 輸出到 self.base_dir/<YYYYMMDD_HHMMSS>/（每次跑產生獨立 snapshot）
          - skip 基準改成「所有歷史 job 的 sample_id 聯集」
          - 不做 stale cleanup（舊 job 永遠不動）
          - skip_existing=True 時跳過歷史已匯出的 sample；False 時照常處理（會在新 job 產生副本）
          - 本次沒有任何 row 要寫 → 不建立 job 資料夾、output_dir 填 base_dir、total=0
        """
        from capi_server import resolve_unc_path  # lazy import 避免 circular

        started_at = datetime.now()
        job_id = started_at.strftime("job_%Y%m%d_%H%M%S")
        job_folder_name = started_at.strftime("%Y%m%d_%H%M%S")
        job_dir = self.base_dir / job_folder_name
        manifest_path = job_dir / "manifest.csv"

        self.base_dir.mkdir(parents=True, exist_ok=True)

        # 1. 掃歷史 job，建 known sample_id 集合
        known_ids = load_known_sample_ids(self.base_dir) if skip_existing else set()

        # 2. 蒐集 candidates
        candidates, diag = self.collect_candidates_with_diagnostics(
            days=days, include_true_ng=include_true_ng
        )
        total = len(candidates)
        logger.info(
            "Collected %d candidates (days=%d, include_true_ng=%s, known_prior=%d)",
            total, days, include_true_ng, len(known_ids),
        )

        new_rows: Dict[str, Dict[str, str]] = {}
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

            if cand.sample_id in known_ids:
                skipped_count["already_exists"] = skipped_count.get("already_exists", 0) + 1
                continue

            source_path = self._resolve_source_path(cand.image_path, resolve_unc_path)
            new_row = self._process_candidate(cand=cand, source_path=source_path, job_dir=job_dir)
            new_rows[cand.sample_id] = new_row

            status = new_row["status"]
            if status == "ok":
                labels_count[new_row["label"]] = labels_count.get(new_row["label"], 0) + 1
            else:
                skipped_count[status] = skipped_count.get(status, 0) + 1

        # 4. 空 job：不建資料夾、不寫 manifest
        if not new_rows:
            logger.info("No new samples for this job; skipping folder creation")
            finished_at = datetime.now()
            return JobSummary(
                job_id=job_id,
                started_at=started_at.isoformat(timespec="seconds"),
                finished_at=finished_at.isoformat(timespec="seconds"),
                duration_sec=round((finished_at - started_at).total_seconds(), 2),
                total=0,
                labels=labels_count,
                skipped=skipped_count,
                output_dir=str(self.base_dir),
                diagnostics=diag,
            )

        # 5. 寫 manifest
        write_manifest(manifest_path, new_rows)
        finished_at = datetime.now()
        return JobSummary(
            job_id=job_id,
            started_at=started_at.isoformat(timespec="seconds"),
            finished_at=finished_at.isoformat(timespec="seconds"),
            duration_sec=round((finished_at - started_at).total_seconds(), 2),
            total=len(new_rows),
            labels=labels_count,
            skipped=skipped_count,
            output_dir=str(job_dir),
            diagnostics=diag,
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
        self, cand: SampleCandidate, source_path: Path, job_dir: Path,
    ) -> Dict[str, str]:
        """處理單一 candidate，回傳 manifest row。

        新架構不再處理「既有 row 的 label 變動 / 強制重做」分支 —
        已存在於任何歷史 job 的 sample_id 早在 run() 入口就被 known_ids skip 掉。
        """
        row = self._build_row_stub(cand)

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

        if cand.source_type == "patchcore_tile":
            crop = crop_patchcore_tile(img, cand.tile_x, cand.tile_y, cand.tile_w, cand.tile_h)
            defect_x = cand.tile_x + cand.tile_w // 2
            defect_y = cand.tile_y + cand.tile_h // 2
            sample_key = f"tile{cand.tile_idx}"
        else:  # edge_defect
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

        filename = build_sample_filename(
            glass_id=cand.glass_id, image_name=cand.image_name,
            sample_key=sample_key, inference_timestamp=cand.inference_timestamp,
        )
        crop_rel = f"{cand.label}/{cand.prefix}/crop/{filename}"
        crop_dst = job_dir / crop_rel
        crop_dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_dst), crop)

        row["crop_path"] = crop_rel
        row["heatmap_path"] = ""
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
