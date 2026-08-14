import sqlite3
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from capi_database import CAPIDatabase
from capi_mes_report import apply_mes_review_miss_policy, build_mes_review_summary
from capi_web import CAPIWebHandler


def _insert_aoi_candidates(db: CAPIDatabase, image_path: str) -> tuple[int, int, int]:
    with sqlite3.connect(str(db.db_path)) as conn:
        record_id = conn.execute(
            """INSERT INTO inference_records
               (glass_id, model_id, machine_no, machine_judgment,
                ai_judgment, image_dir, request_time)
               VALUES ('PANEL-1', 'MODEL-1', 'HM01', 'NG', 'OK', ?, ?)""",
            (image_path, "2026-07-23 08:00:00"),
        ).lastrowid
        image_id = conn.execute(
            """INSERT INTO image_results
               (record_id, image_path, image_name, image_width, image_height)
               VALUES (?, ?, 'G0F00000_080000.tif', 6576, 4384)""",
            (record_id, image_path),
        ).lastrowid
        aoi_tile_id = conn.execute(
            """INSERT INTO tile_results
               (image_result_id, tile_id, x, y, width, height, score,
                is_anomaly, is_aoi_coord, aoi_defect_code,
                aoi_product_x, aoi_product_y, aoi_image_x, aoi_image_y, zone)
               VALUES (?, 7, 1024, 512, 512, 512, 0.23,
                       0, 1, 'PCDK2', 130, 220, 1280, 768, 'inner')""",
            (image_id,),
        ).lastrowid
        conn.execute(
            """INSERT INTO tile_results
               (image_result_id, tile_id, x, y, width, height, score,
                is_anomaly, is_aoi_coord)
               VALUES (?, 8, 1280, 512, 512, 512, 0.80, 1, 0)""",
            (image_id,),
        )
    return record_id, image_id, aoi_tile_id


def test_mes_review_candidates_include_below_threshold_aoi_tile(tmp_path):
    db = CAPIDatabase(tmp_path / "review.db")
    record_id, image_id, tile_id = _insert_aoi_candidates(db, str(tmp_path / "source.tif"))

    candidates = db.get_mes_review_aoi_candidates(record_id)

    assert len(candidates) == 1
    assert candidates[0]["tile_result_id"] == tile_id
    assert candidates[0]["image_result_id"] == image_id
    assert candidates[0]["is_anomaly"] == 0
    assert candidates[0]["ai_score"] == pytest.approx(0.23)
    assert candidates[0]["aoi_image_x"] == 1280
    assert candidates[0]["zone"] == "inner"


def test_mes_review_upsert_syncs_durable_ng_validation_samples(tmp_path):
    db = CAPIDatabase(tmp_path / "review.db")
    record_id, image_id, tile_id = _insert_aoi_candidates(db, str(tmp_path / "source.tif"))
    crop_path = tmp_path / "ng-validation" / "sample.png"
    crop_path.parent.mkdir()
    crop_path.write_bytes(b"crop")

    review = db.save_mes_comparison_review(
        inference_record_id=record_id,
        glass_id="PANEL-1",
        model_id="MODEL-1",
        machine_no="HM01",
        request_time="2026-07-23 08:00:00",
        ai_judgment="OK",
        mes_judgment="NG",
        review_type="miss_detection",
        category="score_below_threshold",
        note="肉眼可見",
        confirmed_ng=True,
        samples=[{
            "tile_result_id": tile_id,
            "image_result_id": image_id,
            "image_name": "G0F00000_080000.tif",
            "source_image_path": str(tmp_path / "source.tif"),
            "lighting": "G0F00000",
            "zone": "inner",
            "aoi_defect_code": "PCDK2",
            "aoi_product_x": 130,
            "aoi_product_y": 220,
            "aoi_image_x": 1280,
            "aoi_image_y": 768,
            "tile_x": 1024,
            "tile_y": 512,
            "tile_w": 512,
            "tile_h": 512,
            "ai_score": 0.23,
            "crop_path": str(crop_path),
        }],
    )

    assert review["confirmed_ng"] == 1
    assert review["reviewer"] == ""
    assert review["ng_sample_count"] == 1
    samples, total = db.list_ng_validation_samples()
    assert total == 1
    assert samples[0]["lighting"] == "G0F00000"
    assert samples[0]["category"] == "score_below_threshold"
    assert db.get_ng_validation_summary() == {
        "samples": 1,
        "reviews": 1,
        "by_lighting": {"G0F00000": 1},
        "by_zone": {"inner": 1},
        "by_model": {"MODEL-1": 1},
    }

    # Source inference/tile retention cleanup must not erase reviewed evidence.
    with sqlite3.connect(str(db.db_path)) as conn:
        conn.execute("DELETE FROM inference_records WHERE id = ?", (record_id,))
    assert db.get_mes_comparison_review(record_id)["ng_sample_count"] == 1
    assert db.get_ng_validation_sample(samples[0]["id"])["crop_path"] == str(crop_path)


def test_training_bomb_samples_share_ng_validation_db_without_fake_review(tmp_path):
    db = CAPIDatabase(tmp_path / "review.db")
    crop_path = tmp_path / "ng-validation" / "MODEL-A" / "G0F00000" / "inner" / "crop" / "bomb.png"
    crop_path.parent.mkdir(parents=True)
    crop_path.write_bytes(b"crop")
    sample = {
        "inference_record_id": 101,
        "image_result_id": 202,
        "coord_index": 0,
        "glass_id": "PANEL-101",
        "model_id": "MODEL-A",
        "machine_no": "HM01",
        "request_time": "2026-08-14 08:00:00",
        "image_name": "G0F00000_080000.tif",
        "source_image_path": str(tmp_path / "source.tif"),
        "lighting": "G0F00000",
        "zone": "inner",
        "source_type": "point",
        "aoi_product_x": 130,
        "aoi_product_y": 220,
        "aoi_image_x": 1280,
        "aoi_image_y": 768,
        "tile_x": 1024,
        "tile_y": 512,
        "tile_w": 512,
        "tile_h": 512,
        "crop_path": str(crop_path),
    }

    assert db.save_training_bomb_validation_samples([sample]) == 1
    assert db.save_training_bomb_validation_samples([sample]) == 1

    cached = db.list_training_bomb_validation_samples(
        machine_id="MODEL-A", lightings=("G0F00000",),
    )
    assert len(cached) == 1
    assert cached[0]["sample_source"] == "training_bomb"
    assert cached[0]["review_id"] == 0
    assert cached[0]["tile_result_id"] < 0
    samples, total = db.list_ng_validation_samples(model_id="MODEL-A")
    assert total == 1
    assert samples[0]["crop_path"] == str(crop_path)
    assert db.get_ng_validation_summary() == {
        "samples": 1,
        "reviews": 0,
        "by_lighting": {"G0F00000": 1},
        "by_zone": {"inner": 1},
        "by_model": {"MODEL-A": 1},
    }


def test_ng_validation_filters_by_model_and_deletes_only_selected_crop(tmp_path):
    db = CAPIDatabase(tmp_path / "review.db")
    base_dir = tmp_path / "ng-validation"
    base_dir.mkdir()
    crop_a = base_dir / "model-a.png"
    crop_b = base_dir / "model-b.png"
    crop_a.write_bytes(b"model-a")
    crop_b.write_bytes(b"model-b")

    for inference_id, tile_id, model_id, crop_path in (
        (101, 1001, "MODEL-A", crop_a),
        (102, 1002, "MODEL-B", crop_b),
    ):
        db.save_mes_comparison_review(
            inference_record_id=inference_id,
            glass_id=f"PANEL-{model_id}",
            model_id=model_id,
            machine_no="HM01",
            request_time="2026-07-23 08:00:00",
            ai_judgment="OK",
            mes_judgment="NG",
            review_type="miss_detection",
            category="score_below_threshold",
            confirmed_ng=True,
            samples=[{
                "tile_result_id": tile_id,
                "image_result_id": tile_id + 100,
                "image_name": "G0F00000_080000.tif",
                "lighting": "G0F00000",
                "zone": "inner",
                "crop_path": str(crop_path),
            }],
        )

    model_a_samples, model_a_total = db.list_ng_validation_samples(model_id="MODEL-A")
    assert model_a_total == 1
    assert model_a_samples[0]["model_id"] == "MODEL-A"
    assert db.get_ng_validation_summary()["by_model"] == {
        "MODEL-A": 1,
        "MODEL-B": 1,
    }

    handler = object.__new__(CAPIWebHandler)
    handler.db = db
    handler._capi_server_instance = SimpleNamespace(
        server_config={"ng_validation": {"base_dir": str(base_dir)}}
    )
    handler._read_json_body = lambda: {"sample_id": model_a_samples[0]["id"]}
    responses = []
    handler._send_json = lambda data, status=200: responses.append((status, data))

    handler._handle_ng_validation_delete()

    assert responses[0][0] == 200
    assert responses[0][1]["success"] is True
    assert responses[0][1]["file_deleted"] is True
    assert not crop_a.exists()
    assert crop_b.is_file()
    assert db.get_ng_validation_sample(model_a_samples[0]["id"]) is None
    assert db.get_mes_comparison_review(101)["category"] == "score_below_threshold"
    assert db.get_mes_comparison_review(101)["ng_sample_count"] == 0
    assert db.get_ng_validation_summary()["by_model"] == {"MODEL-B": 1}


def test_ng_validation_delete_rejects_crop_outside_configured_base(tmp_path):
    db = CAPIDatabase(tmp_path / "review.db")
    base_dir = tmp_path / "ng-validation"
    base_dir.mkdir()
    outside_crop = tmp_path / "outside.png"
    outside_crop.write_bytes(b"outside")
    db.save_mes_comparison_review(
        inference_record_id=201,
        glass_id="PANEL-OUTSIDE",
        model_id="MODEL-A",
        machine_no="HM01",
        request_time="2026-07-23 08:00:00",
        ai_judgment="OK",
        mes_judgment="NG",
        review_type="miss_detection",
        category="score_below_threshold",
        confirmed_ng=True,
        samples=[{
            "tile_result_id": 2001,
            "image_result_id": 2101,
            "image_name": "G0F00000_080000.tif",
            "lighting": "G0F00000",
            "zone": "inner",
            "crop_path": str(outside_crop),
        }],
    )
    samples, _ = db.list_ng_validation_samples()

    handler = object.__new__(CAPIWebHandler)
    handler.db = db
    handler._capi_server_instance = SimpleNamespace(
        server_config={"ng_validation": {"base_dir": str(base_dir)}}
    )
    handler._read_json_body = lambda: {"sample_id": samples[0]["id"]}
    responses = []
    handler._send_json = lambda data, status=200: responses.append((status, data))

    handler._handle_ng_validation_delete()

    assert responses[0][0] == 403
    assert outside_crop.is_file()
    assert db.get_ng_validation_sample(samples[0]["id"]) is not None


def test_mes_review_update_can_reopen_and_remove_active_ng_samples(tmp_path):
    db = CAPIDatabase(tmp_path / "review.db")
    record_id, image_id, tile_id = _insert_aoi_candidates(db, str(tmp_path / "source.tif"))
    common = {
        "inference_record_id": record_id,
        "glass_id": "PANEL-1",
        "model_id": "MODEL-1",
        "machine_no": "HM01",
        "request_time": "2026-07-23 08:00:00",
        "ai_judgment": "OK",
        "mes_judgment": "NG",
        "review_type": "miss_detection",
        "note": "",
        "reviewer": "operator-a",
    }
    db.save_mes_comparison_review(
        **common,
        category="low_contrast",
        confirmed_ng=True,
        samples=[{
            "tile_result_id": tile_id,
            "image_result_id": image_id,
            "image_name": "G0F00000_080000.tif",
            "lighting": "G0F00000",
            "zone": "inner",
            "crop_path": str(tmp_path / "sample.png"),
        }],
    )

    review = db.save_mes_comparison_review(
        **common,
        category="mes_misjudge",
        confirmed_ng=False,
        samples=[],
    )

    assert review["confirmed_ng"] == 0
    assert review["ng_sample_count"] == 0
    assert db.list_ng_validation_samples()[1] == 0


def test_mes_review_rejects_category_from_other_review_type(tmp_path):
    db = CAPIDatabase(tmp_path / "review.db")

    with pytest.raises(ValueError, match="Invalid category"):
        db.save_mes_comparison_review(
            inference_record_id=1,
            glass_id="PANEL-1",
            model_id="MODEL-1",
            machine_no="HM01",
            request_time="2026-07-23 08:00:00",
            ai_judgment="OK",
            mes_judgment="NG",
            review_type="miss_detection",
            category="edge_false_positive",
        )


def test_mes_review_accepts_within_spec_release_category(tmp_path):
    db = CAPIDatabase(tmp_path / "review.db")

    review = db.save_mes_comparison_review(
        inference_record_id=1,
        glass_id="PANEL-1",
        model_id="MODEL-1",
        machine_no="HM01",
        request_time="2026-07-23 08:00:00",
        ai_judgment="OK",
        mes_judgment="NG",
        review_type="miss_detection",
        category="within_spec_release",
    )

    assert review["category"] == "within_spec_release"


def test_build_mes_review_summary_counts_pending_reasons_and_ng_samples():
    records = [
        {"review_type": "over_detection", "review": None},
        {
            "review_type": "over_detection",
            "review": {
                "category": "within_spec",
                "confirmed_ng": 0,
                "ng_sample_count": 0,
            },
        },
        {
            "review_type": "miss_detection",
            "review": {
                "category": "low_contrast",
                "confirmed_ng": 1,
                "ng_sample_count": 2,
            },
        },
        {"review_type": "true_ng", "review": None},
        {"review_type": "", "review": None},
    ]

    summary = build_mes_review_summary(records)

    assert summary["total"] == 4
    assert summary["reviewed"] == 2
    assert summary["pending"] == 2
    assert summary["confirmed_ng_reviews"] == 1
    assert summary["ng_samples"] == 2
    assert summary["by_type"]["over_detection"]["by_category"] == {"within_spec": 1}
    assert summary["by_type"]["miss_detection"]["by_category"] == {"low_contrast": 1}


def test_mes_miss_policy_counts_pending_and_only_selected_review_categories():
    report = {
        "summary": {
            "total": 7,
            "miss_detection": 6,
            "miss_detection_rate": 85.71,
        },
        "records": [
            {"comparison": "miss_detection", "review": None},
            {"comparison": "miss_detection", "review": {"category": "score_below_threshold"}},
            {"comparison": "miss_detection", "review": {"category": "low_contrast"}},
            {"comparison": "miss_detection", "review": {"category": "dust_misfilter"}},
            {"comparison": "miss_detection", "review": {"category": "not_visible_in_image"}},
            {"comparison": "miss_detection", "review": {"category": "within_spec_release"}},
            {"comparison": "correct", "review": None},
        ],
    }

    apply_mes_review_miss_policy(report)

    assert report["summary"]["miss_detection"] == 4
    assert report["summary"]["miss_detection_rate"] == 57.14
    assert [row["counts_as_miss_detection"] for row in report["records"]] == [
        True, True, True, True, False, False, False,
    ]


def test_web_helper_saves_selected_aoi_crop_under_validation_root(tmp_path):
    source_path = tmp_path / "G0F00000_080000.png"
    image = np.arange(700 * 700, dtype=np.uint32).reshape(700, 700)
    image = (image % 256).astype(np.uint8)
    assert cv2.imwrite(str(source_path), image)

    handler = object.__new__(CAPIWebHandler)
    handler.inferencer = SimpleNamespace(
        config=SimpleNamespace(inference_rotate_180_enabled=False)
    )
    handler._capi_server_instance = SimpleNamespace(
        server_config={
            "ng_validation": {"base_dir": str(tmp_path / "ng-validation")},
        },
        path_mapping={},
    )
    record = {
        "id": 10,
        "glass_id": "PANEL-1",
        "model_id": "MODEL-1",
        "request_time": "2026-07-23 08:00:00",
    }
    candidate = {
        "tile_result_id": 22,
        "image_result_id": 33,
        "image_name": source_path.name,
        "image_path": str(source_path),
        "image_is_bomb": 0,
        "is_bomb": 0,
        "tile_x": 400,
        "tile_y": 400,
        "tile_w": 512,
        "tile_h": 512,
        "zone": "inner",
        "aoi_defect_code": "PCDK2",
        "aoi_product_x": 0,
        "aoi_product_y": 20,
        "aoi_image_x": 0,
        "aoi_image_y": 500,
        "ai_score": 0.23,
    }

    samples = handler._prepare_ng_validation_samples(record, [candidate])

    assert len(samples) == 1
    assert samples[0]["lighting"] == "G0F00000"
    assert samples[0]["aoi_product_x"] == 0
    crop_path = Path(samples[0]["crop_path"])
    assert crop_path.is_file()
    crop = cv2.imread(str(crop_path), cv2.IMREAD_UNCHANGED)
    assert crop.shape == (512, 512)
    assert str(crop_path).startswith(str((tmp_path / "ng-validation").resolve()))

    b0f_candidate = dict(candidate, image_name="B0F00000_080000.png")
    with pytest.raises(ValueError, match="光源不納入 NG 驗證庫"):
        handler._prepare_ng_validation_samples(record, [b0f_candidate])


def test_web_review_save_does_not_require_reviewer(tmp_path):
    db = CAPIDatabase(tmp_path / "review.db")
    record_id, _, _ = _insert_aoi_candidates(db, str(tmp_path / "source.tif"))
    handler = object.__new__(CAPIWebHandler)
    handler.db = db
    handler._read_json_body = lambda: {
        "record_id": record_id,
        "review_type": "miss_detection",
        "category": "not_visible_in_image",
        "note": "肉眼可見",
        "confirmed_ng": False,
        "selected_tile_ids": [],
        "mes_judgment": "NG",
    }
    responses = []
    handler._send_json = lambda data, status=200: responses.append((status, data))

    handler._handle_mes_review_save()

    assert responses[0][0] == 200
    assert responses[0][1]["success"] is True
    assert responses[0][1]["review"]["reviewer"] == ""


def test_mes_review_candidates_explain_why_bomb_tile_is_disabled(tmp_path):
    source_path = tmp_path / "G0F00000_080000.tif"
    source_path.write_bytes(b"fixture")
    db = CAPIDatabase(tmp_path / "review.db")
    record_id, _, tile_id = _insert_aoi_candidates(db, str(source_path))
    with sqlite3.connect(str(db.db_path)) as conn:
        conn.execute(
            "UPDATE tile_results SET is_bomb = 1 WHERE id = ?",
            (tile_id,),
        )

    handler = object.__new__(CAPIWebHandler)
    handler.db = db
    handler._capi_server_instance = SimpleNamespace(path_mapping={})
    responses = []
    handler._send_json = lambda data, status=200: responses.append((status, data))

    handler._handle_mes_review_candidates_api({"record_id": [str(record_id)]})

    assert responses[0][0] == 200
    candidate = responses[0][1]["candidates"][0]
    assert candidate["collectable"] is False
    assert candidate["collectable_reason"] == "BOMB 模擬缺陷，不納入真實 NG 驗證庫"


def test_report_template_contains_manual_review_and_ng_database_ui():
    template = Path("templates/ric_report.html").read_text(encoding="utf-8")

    assert 'id="mes_workflowTabs"' in template
    assert "mesReportTab.switchView('comparison')" in template
    assert "mesReportTab.switchView('review')" in template
    assert 'id="mes_reviewPanel"' in template
    assert 'id="mesReviewCandidateGrid"' in template
    assert 'id="mesReviewConfirmedNg"' in template
    assert 'id="mesNgDatabaseModal"' in template
    assert "/api/ric/mes-review/candidates" in template
    assert "/api/ric/ng-validation" in template
    assert "/api/ric/ng-validation/delete" in template
    assert 'id="mesNgModelFilter"' in template
    assert "filterNgDatabase" in template
    assert "deleteNgSample" in template
    assert "🗑 刪除圖片" in template
    assert "機種：" in template
    assert "訓練 AOI 炸彈快取" in template
    assert "人工確認事件" in template
    assert "B0F 已排除" in template
    assert "mesReviewReviewer" not in template
    assert "Review人員" not in template
    assert "非畫面檢可見不良" in template
    assert "異常區域不在 AOI 提供區域" in template
    assert "規格內釋放" in template
    assert "未 Review 的漏檢全部計入" in template
    assert "COUNTED_MES_MISS_REVIEW_CATEGORIES" in template
    assert "countsAsMissDetection(row)" in template
    assert "rows = rows.filter(countsAsMissDetection)" in template
    miss_order = template.split("miss_detection: [", 1)[1].split("]", 1)[0]
    assert miss_order.index("'not_visible_in_image'") < miss_order.index("'outside_aoi_area'")
    assert miss_order.index("'outside_aoi_area'") < miss_order.index("'score_below_threshold'")
    assert miss_order.index("'mes_misjudge'") < miss_order.index("'within_spec_release'")
    assert "mes-review-candidate-disabled" in template
    assert "不可選：" in template
    assert "prepareReasonChartData" in template
    assert "entries.slice(0, limit)" in template
    assert "其他（合併）" in template
    assert "mes_overReasonChart" in template
    assert "mes_missReasonChart" in template
    assert "indexAxis: 'y'" in template
    assert "REVIEW_REASON_COLORS" in template
    assert "outside_aoi_area: 'rgba(14,165,233,0.82)'" in template
    assert "dust_misfilter: 'rgba(234,179,8,0.82)'" in template
    assert "mes_misjudge: 'rgba(34,197,94,0.82)'" in template
    assert "within_spec_release: 'rgba(20,184,166,0.82)'" in template
    assert "backgroundColor: chartData.entries.map" in template
    assert "已 Review" in template
    assert "toggleReviewReasonFilter(type, item)" in template
    assert "onClick: (_event, elements)" in template
    assert "categories: remainingEntries.map(item => item.category)" in template
    assert "categories.has(row.review?.category)" in template
    assert "_reviewFilter = null;" in template
    assert 'id="mes_reasonFilterChip"' in template
    assert "clearReviewReasonFilter" in template
    assert "reviewReasonFilterLabel() || reviewFilterLabel()" in template
    assert "點擊長條篩選" in template
    assert "座標匹配" in template
    assert "coordinateMatchHtml(row)" in template
    assert "CAPIHM 旋轉換算" in template
    assert "'MES 有效不良', '座標匹配'" in template
    assert "只使用 AOI 座標篩選五光源圖片" not in template
    assert "新模型訓練自動保存的 AOI 炸彈快取" in template
    assert 'id="mes_comparisonTrendChart"' in template
    assert "buildComparisonDailyTrend" in template
    assert "判定正確率 (%)" in template
    assert "當日可比對數" in template
    assert "無法比對資料不納入比例" in template
    assert "mes-review-stat-action" in template
    assert "點擊查看明細 →" in template
    assert 'aria-pressed="${active}"' in template
