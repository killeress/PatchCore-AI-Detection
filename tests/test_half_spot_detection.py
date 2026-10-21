"""BWFHALF0/WBFHALF0 半屏黑底白點偵測測試。"""
from types import SimpleNamespace

from capi_image_naming import (
    AOI_REPORT_PREFIXES,
    HALF_SCREEN_PREFIXES,
    canonical_image_prefix,
    is_half_screen_prefix,
)
from capi_config import CAPIConfig


def test_half_screen_prefixes_registered():
    assert set(HALF_SCREEN_PREFIXES) == {"BWFHALF0", "WBFHALF0"}
    for p in HALF_SCREEN_PREFIXES:
        assert p in AOI_REPORT_PREFIXES  # 讓 AOI TXT 解析 regex 能匹配
        assert is_half_screen_prefix(p)
    assert not is_half_screen_prefix("B0F00000")
    assert not is_half_screen_prefix("R0F00000")
    assert not is_half_screen_prefix("")


def test_canonical_prefix_passthrough():
    # 非 canonical 前綴 fallback 為 stem 去尾段，不需改動 canonical 清單
    assert canonical_image_prefix("BWFHALF0_101831.tif") == "BWFHALF0"
    assert canonical_image_prefix("WBFHALF0_101829.tif") == "WBFHALF0"


def test_config_defaults_present():
    cfg = CAPIConfig()
    assert cfg.half_spot_diff_threshold == 10
    assert cfg.half_spot_median_kernel == 21
    assert cfg.half_spot_min_area == 5
    assert cfg.half_spot_max_area == 2000
    assert cfg.half_spot_max_aspect_ratio == 3.0
    assert cfg.half_spot_black_region_threshold == 60
    assert cfg.half_spot_abs_threshold == 200


def test_config_db_override():
    cfg = CAPIConfig()
    cfg.apply_db_overrides([
        {"param_name": "half_spot_diff_threshold", "decoded_value": "15"},
        {"param_name": "half_spot_max_aspect_ratio", "decoded_value": "2.5"},
        {"param_name": "half_spot_black_region_threshold", "decoded_value": "80"},
    ])
    assert cfg.half_spot_diff_threshold == 15
    assert cfg.half_spot_max_aspect_ratio == 2.5
    assert cfg.half_spot_black_region_threshold == 80


import numpy as np
from capi_inference import detect_half_spot_binary

PARAMS = dict(
    diff_threshold=10, median_kernel=21, min_area=5, max_area=2000,
    max_aspect_ratio=3.0, abs_threshold=200, black_region_threshold=60,
)


def _black_img(value=20):
    return np.full((512, 512, 3), value, dtype=np.uint8)


def _draw_rect(img, x, y, w, h, value=255):
    img[y:y + h, x:x + w] = value
    return img


def test_white_dot_on_black_is_ng():
    img = _draw_rect(_black_img(), 250, 250, 4, 3)  # 12 px 白點
    score, binary, st = detect_half_spot_binary(img, aoi_local_xy=(256, 256), **PARAMS)
    assert score == 1.0 and st["region"] == "black" and st["kept"] == 1
    assert binary[251, 252] == 255


def test_white_region_gate_returns_ok():
    img = _black_img()
    img[:, 256:] = 220  # 右半白區
    img = _draw_rect(img, 400, 250, 4, 3)  # 白區裡的亮點
    score, binary, st = detect_half_spot_binary(img, aoi_local_xy=(400, 256), **PARAMS)
    assert score == 0.0 and st["region"] == "white"
    assert not np.any(binary)


def test_bright_blob_on_white_background_not_counted():
    # AOI 在黑區，但 tile 內白區有一處更亮的斑 → 不得誤判
    img = _black_img()
    img[:, 400:] = 210
    img = _draw_rect(img, 450, 250, 4, 3, value=255)  # 白區亮斑
    score, _, st = detect_half_spot_binary(img, aoi_local_xy=(100, 256), **PARAMS)
    assert score == 0.0 and st["region"] == "black"


def test_long_strip_is_ok():
    img = _draw_rect(_black_img(), 156, 253, 200, 5)  # 長條漏光 1000px, 長寬比 40
    score, _, st = detect_half_spot_binary(img, aoi_local_xy=(256, 256), **PARAMS)
    assert score == 0.0 and st["too_elongated"] + st["too_large"] >= 1


def test_large_block_is_ok():
    # AOI 在黑區、附近有 80x60 大亮塊 → 大區塊內部 median 背景亦為亮，
    # 以亮底排除（或面積上限），不判 NG
    img = _draw_rect(_black_img(), 380, 226, 80, 60)  # 區塊異常 4800px
    score, _, st = detect_half_spot_binary(img, aoi_local_xy=(256, 256), **PARAMS)
    assert score == 0.0 and st["region"] == "black"
    assert st["bright_background"] + st["too_large"] >= 1


def test_large_block_covering_aoi_treated_as_white_region():
    # 大亮塊正好盖住 AOI 點 → 鄰域中位灰階達門檻，視同白區不檢測
    img = _draw_rect(_black_img(), 216, 226, 80, 60)
    score, _, st = detect_half_spot_binary(img, aoi_local_xy=(256, 256), **PARAMS)
    assert score == 0.0 and st["region"] == "white"


def test_tiny_dot_is_ok():
    img = _black_img()
    img[255:257, 255:257] = 180  # 4px 且亮度 < abs_threshold，面積 < min_area
    score, _, st = detect_half_spot_binary(img, aoi_local_xy=(256, 256), **PARAMS)
    assert score == 0.0


def test_boundary_aoi_still_detects():
    # AOI 在座標鄰域跨黑白交界、黑區佔多數 → 仍檢測
    img = _black_img()
    img[:, 280:] = 220
    img = _draw_rect(img, 250, 250, 4, 3)
    score, _, st = detect_half_spot_binary(img, aoi_local_xy=(256, 256), **PARAMS)
    assert score == 1.0


def test_get_known_image_prefixes_includes_half():
    # AOI TXT 解析 regex 由此前綴清單動態建立，須含 HALF 前綴才能解析
    # PCLH20135300821WBFHALF0 → code=PCLH2, X=01353, Y=00821, prefix=WBFHALF0
    from capi_inference import CAPIInferencer
    inst = CAPIInferencer.__new__(CAPIInferencer)
    inst.config = SimpleNamespace(model_mapping={}, skip_files=[])
    inst._model_mapping = {}
    known = inst._get_known_image_prefixes()
    assert "BWFHALF0" in known and "WBFHALF0" in known


def test_half_spot_roi_recheck_shape_filter():
    from capi_inference import filter_half_spot_components
    # 一個小點 + 一個長條
    binary = np.zeros((100, 100), dtype=np.uint8)
    binary[10:13, 10:13] = 255      # 9px 方點 → 保留
    binary[50:53, 10:90] = 255      # 3x80 長條 → 排除
    filtered, kept = filter_half_spot_components(binary, min_area=5, max_area=500, max_aspect_ratio=3.0)
    assert kept == 1
    assert filtered[11, 11] == 255 and filtered[51, 50] == 0


def _stub_half_tile(*, ng: bool, code="PCLH2", px=1353, py=821, peak=(5200, 1900)):
    return SimpleNamespace(
        is_half_spot_detection=True, is_bomb=False, is_bright_spot_detection=False,
        is_suspected_dust_or_scratch=False, is_in_exclude_zone=False,
        scratch_filtered=False, is_aoi_coord_below_threshold=not ng,
        aoi_defect_code=code, aoi_product_x=px, aoi_product_y=py,
        aoi_image_x=100, aoi_image_y=100, is_aoi_coord_tile=True,
        anomaly_peak_x=peak[0], anomaly_peak_y=peak[1], anomaly_peak_source="aoi_roi",
        x=peak[0] - 256, y=peak[1] - 256, width=512, height=512,
        center=(peak[0], peak[1]),
        bomb_remaining_points=None, dust_two_stage_features=None, dust_region_details=None,
    )


def _stub_result(tiles):
    from pathlib import Path
    return SimpleNamespace(
        image_path=Path("WBFHALF0_101829.tif"), report_image_prefix="",
        raw_bounds=(1000, 500, 5000, 3500), otsu_bounds=(1000, 500, 5000, 3500),
        panel_polygon=None, anomaly_tiles=[(t, 1.0, None) for t in tiles],
        edge_defects=[], white_frame_result=None,
    )


def test_qjpg_ok_passthrough_original_coord():
    from capi_server import _iter_qjpg_defect_records
    recs = _iter_qjpg_defect_records(
        [_stub_result([_stub_half_tile(ng=False)])], (1920, 1200), None, bomb_only=True)
    assert recs == ["PCLH20135300821WBFHALF0"]


def test_qjpg_ng_reports_centroid_with_aoi_code():
    from capi_server import _iter_qjpg_defect_records
    recs = _iter_qjpg_defect_records(
        [_stub_result([_stub_half_tile(ng=True)])], (1920, 1200), None, bomb_only=False)
    assert len(recs) == 1 and recs[0].startswith("PCLH2") and recs[0].endswith("WBFHALF0")
    assert recs[0] != "PCLH20135300821WBFHALF0"  # 重心非 AOI 原座標


def test_qjpg_ng_panel_still_passthrough_ok_half_tiles():
    # 面板因其他點位 NG 時，未命中的 half tile 仍透傳 AOI 原座標
    from capi_server import _iter_qjpg_defect_records
    tiles = [_stub_half_tile(ng=True, px=100, py=200),
             _stub_half_tile(ng=False, px=1353, py=821)]
    recs = _iter_qjpg_defect_records([_stub_result(tiles)], (1920, 1200), None, bomb_only=False)
    assert "PCLH20135300821WBFHALF0" in recs
    assert any(r.startswith("PCLH2") and r.endswith("WBFHALF0") and "0135300821" not in r for r in recs)


def test_tile_diagnostics_half_spot_evidence():
    from capi_tile_diagnostics import decision_evidence
    tile = {
        "is_bright_spot_detection": False,
        "zone": "half_spot",
        "is_anomaly": True,
        "score": 1.0,
        "decision_context": {
            "region": "black", "gate_median": 18.0, "black_region_threshold": 60,
            "max_diff": 235, "diff_threshold": 10, "max_pixel": 255, "abs_threshold": 200,
            "max_component_area": 12, "min_area": 5, "max_area": 500,
            "max_aspect_ratio": 3.0, "rejected_too_large": 0, "rejected_too_elongated": 1,
        },
    }
    ev = decision_evidence(tile)
    assert ev["status"] == "NG" and "半屏" in ev["rule_label"]
    labels = [m["label"] for m in ev["metrics"]]
    assert any("鄰域" in l or "中位" in l for l in labels)
    assert any("面積上限" in l for l in labels)


def test_tile_diagnostics_half_spot_white_region_ok():
    from capi_tile_diagnostics import decision_evidence
    tile = {
        "is_bright_spot_detection": False,
        "zone": "half_spot",
        "is_anomaly": False,
        "score": 0.0,
        "decision_context": {
            "region": "white", "gate_median": 210.0, "black_region_threshold": 60,
            "max_diff": 0, "diff_threshold": 10, "max_pixel": 240, "abs_threshold": 200,
            "max_component_area": 0, "min_area": 5, "max_area": 500, "max_aspect_ratio": 3.0,
        },
    }
    ev = decision_evidence(tile)
    assert ev["status"] == "OK"
    assert any("白區" in n for n in ev["notes"])
