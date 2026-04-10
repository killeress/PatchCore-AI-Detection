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
    detail["images"][0]["image_path"] = str(fake_img_path)
    detail["images"][0]["tiles"][0]["heatmap_path"] = str(fake_hm)
    detail["images"][0]["edge_defects"][0]["heatmap_path"] = str(fake_edge_hm)
    detail["images"].pop(1)  # 移除炸彈 image 避免路徑失效

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
    from capi_dataset_export import read_manifest
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
    from capi_dataset_export import read_manifest
    row = _make_accuracy_row()
    detail = _make_record_detail()
    detail["images"][0]["image_path"] = str(tmp_path / "does_not_exist.tif")
    detail["images"].pop(1)
    db = FakeDB(accuracy_rows=[row], record_details={1001: detail})
    output = tmp_path / "out"
    exporter = DatasetExporter(db, base_dir=str(output), path_mapping={})

    summary = exporter.run(days=3, include_true_ng=True, skip_existing=True)

    manifest = read_manifest(output / "manifest.csv")
    assert manifest["GLS123_G0F00000_114438_tile3"]["status"] == "skipped_no_source"
    assert summary.skipped.get("skipped_no_source", 0) >= 1


def test_exporter_run_second_pass_moves_on_label_change(tmp_path):
    """第一次跑 label=other → 第二次 DB 改成 edge_false_positive → 檔案應 move"""
    import cv2
    from capi_dataset_export import read_manifest

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
    detail["images"][0]["edge_defects"] = []
    detail["images"].pop(1)
    db = FakeDB(accuracy_rows=[row], record_details={1001: detail})
    output = tmp_path / "out"
    exporter = DatasetExporter(db, base_dir=str(output), path_mapping={})

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


# ==== 黑光源圖 (B0F) / bomb record 過濾測試 ====

from capi_dataset_export import BLACK_IMAGE_PREFIX


def test_collect_candidates_skips_b0f_images():
    """檔名以 B0F 開頭 → 黑光源圖，該 image 不蒐集（tile 與 edge 都跳過）"""
    row = _make_accuracy_row()
    detail = _make_record_detail()
    # 把第一張 image 改成黑光源
    detail["images"][0]["image_name"] = "B0F00000_114438.tif"
    detail["images"][0]["image_path"] = "/data/panels/GLS123/B0F00000_114438.tif"
    db = FakeDB(accuracy_rows=[row], record_details={1001: detail})
    exporter = DatasetExporter(db, base_dir="/tmp/out", path_mapping={})
    cands = exporter.collect_candidates(days=3, include_true_ng=True)
    # B0F 的 tile 和 edge 都不該出現
    assert not any("B0F" in c.sample_id for c in cands)
    # 其他 image（非 B0F）仍照常蒐集 — 本 fixture 另一張是 W0F 但被標為 is_bomb，
    # 所以也會被過濾，最終 candidates 應為空
    assert cands == []


def test_collect_candidates_keeps_non_b0f_images():
    """確認 B0F prefix 過濾不會誤殺 G0F / W0F 等正常光源"""
    row = _make_accuracy_row()
    detail = _make_record_detail()
    # 原本的 G0F 應保留，確認 G0F 不以 B0F 開頭
    assert BLACK_IMAGE_PREFIX == "B0F"
    db = FakeDB(accuracy_rows=[row], record_details={1001: detail})
    exporter = DatasetExporter(db, base_dir="/tmp/out", path_mapping={})
    cands = exporter.collect_candidates(days=3, include_true_ng=True)
    assert any("G0F00000" in c.sample_id for c in cands)


def test_collect_candidates_skips_record_with_client_bomb_info():
    """inference_records.client_bomb_info 非空 → 整個 record 不蒐集"""
    row = _make_accuracy_row()
    detail = _make_record_detail()
    detail["client_bomb_info"] = '{"coords":[{"x":100,"y":200}]}'
    db = FakeDB(accuracy_rows=[row], record_details={1001: detail})
    exporter = DatasetExporter(db, base_dir="/tmp/out", path_mapping={})
    assert exporter.collect_candidates(days=3, include_true_ng=True) == []


def test_exporter_run_counts_already_exists_on_second_pass(tmp_path):
    """第二次跑 (skip_existing=True, label 未變) → 樣本計入 skipped['already_exists']，total 不為 0"""
    import cv2
    from capi_dataset_export import read_manifest

    panel_dir = tmp_path / "panels" / "GLS123"
    panel_dir.mkdir(parents=True)
    fake_img = panel_dir / "G0F00000_114438.tif"
    cv2.imwrite(str(fake_img), np.full((2048, 2048, 3), 150, dtype=np.uint8))
    fake_hm = tmp_path / "heatmaps" / "heatmap_tile3.png"
    fake_hm.parent.mkdir()
    cv2.imwrite(str(fake_hm), np.zeros((256, 1024, 3), dtype=np.uint8))

    row = _make_accuracy_row()
    detail = _make_record_detail()
    detail["images"][0]["image_path"] = str(fake_img)
    detail["images"][0]["tiles"][0]["heatmap_path"] = str(fake_hm)
    detail["images"][0]["edge_defects"] = []
    detail["images"].pop(1)
    db = FakeDB(accuracy_rows=[row], record_details={1001: detail})
    output = tmp_path / "out"
    exporter = DatasetExporter(db, base_dir=str(output), path_mapping={})

    # 第一次：蒐集到一筆
    s1 = exporter.run(days=3, include_true_ng=True, skip_existing=True)
    assert s1.labels.get("over_edge_false_positive") == 1
    assert s1.skipped.get("already_exists", 0) == 0

    # 第二次：label 沒變、檔已存在 → already_exists 應計入
    s2 = exporter.run(days=3, include_true_ng=True, skip_existing=True)
    assert s2.labels == {}  # 沒有新增
    assert s2.skipped.get("already_exists") == 1
    assert s2.total == 1


def test_exporter_run_cleans_stale_manifest_entries(tmp_path):
    """舊樣本在新過濾規則下不再是 candidate → 檔案與 manifest 都應被清除"""
    import cv2
    from capi_dataset_export import read_manifest

    panel_dir = tmp_path / "panels" / "GLS123"
    panel_dir.mkdir(parents=True)
    fake_img = panel_dir / "G0F00000_114438.tif"
    cv2.imwrite(str(fake_img), np.full((2048, 2048, 3), 150, dtype=np.uint8))
    fake_hm = tmp_path / "heatmaps" / "heatmap_tile3.png"
    fake_hm.parent.mkdir()
    cv2.imwrite(str(fake_hm), np.zeros((256, 1024, 3), dtype=np.uint8))

    row = _make_accuracy_row()
    detail = _make_record_detail()
    detail["images"][0]["image_path"] = str(fake_img)
    detail["images"][0]["tiles"][0]["heatmap_path"] = str(fake_hm)
    detail["images"][0]["edge_defects"] = []
    detail["images"].pop(1)
    db = FakeDB(accuracy_rows=[row], record_details={1001: detail})
    output = tmp_path / "out"
    exporter = DatasetExporter(db, base_dir=str(output), path_mapping={})

    # 第一次：正常蒐集
    exporter.run(days=3, include_true_ng=True, skip_existing=True)
    stale_crop = output / "over_edge_false_positive" / "G0F00000" / "crop" / "20260408_GLS123_G0F00000_114438_tile3.png"
    assert stale_crop.exists()

    # 模擬：DB 這筆 record 突然被標記為 bomb panel（或手動改 record 讓它不再是 candidate）
    db._record_details[1001]["client_bomb_info"] = '{"coords":[{"x":100,"y":200}]}'

    # 第二次：沒 candidate，應清掉舊樣本
    s2 = exporter.run(days=3, include_true_ng=True, skip_existing=True)
    assert not stale_crop.exists()
    manifest = read_manifest(output / "manifest.csv")
    assert "GLS123_G0F00000_114438_tile3" not in manifest
    assert s2.skipped.get("cleaned_stale") == 1


def test_collect_candidates_keeps_record_when_client_bomb_info_empty():
    """client_bomb_info 空字串 → 照常蒐集（防 regression）"""
    row = _make_accuracy_row()
    detail = _make_record_detail()
    detail["client_bomb_info"] = ""  # 明確設為空
    db = FakeDB(accuracy_rows=[row], record_details={1001: detail})
    exporter = DatasetExporter(db, base_dir="/tmp/out", path_mapping={})
    cands = exporter.collect_candidates(days=3, include_true_ng=True)
    assert len(cands) >= 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
