"""面板 4 角 polygon 功能的單元測試"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows cp950 console 無法顯示 Unicode 檢查記號，強制 utf-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import yaml
import numpy as np
from capi_config import CAPIConfig


def test_config_enable_panel_polygon_default_true():
    """enable_panel_polygon 預設必須為 True"""
    cfg = CAPIConfig()
    assert cfg.enable_panel_polygon is True, \
        f"expected default True, got {cfg.enable_panel_polygon}"
    print("✅ test_config_enable_panel_polygon_default_true")


def test_config_roundtrip_enable_panel_polygon():
    """from_dict / to_dict 必須保留 enable_panel_polygon 欄位"""
    cfg1 = CAPIConfig()
    cfg1.enable_panel_polygon = False
    d = cfg1.to_dict()
    assert "enable_panel_polygon" in d
    assert d["enable_panel_polygon"] is False

    cfg2 = CAPIConfig.from_dict(d)
    assert cfg2.enable_panel_polygon is False
    print("✅ test_config_roundtrip_enable_panel_polygon")


def test_config_yaml_loads_enable_panel_polygon():
    """從實際 capi_3f.yaml 讀取應該能抓到 enable_panel_polygon"""
    yaml_path = Path(__file__).resolve().parent.parent / "configs" / "capi_3f.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "enable_panel_polygon" in data, \
        f"capi_3f.yaml 缺少 enable_panel_polygon 欄位"
    assert data["enable_panel_polygon"] is True
    print("✅ test_config_yaml_loads_enable_panel_polygon")


import cv2
from capi_inference import CAPIInferencer
from capi_preprocess import (
    BOUNDARY_GRAY_BAND_SHIFT_PARAMS,
    PreprocessConfig,
    _detect_aapi_large_panel_raw_boundary,
    detect_panel_boundary,
    detect_panel_polygon,
    _polyfit_polygon,
)  # 直接測試 polygon 數學邏輯


def _make_inferencer():
    """建立一個不需要模型載入的 inferencer instance"""
    cfg = CAPIConfig()
    cfg.tile_size = 512
    return CAPIInferencer(cfg)


# --- 純 polygon 數學測試：直接使用 capi_preprocess._polyfit_polygon ---

def test_polygon_detect_ideal_rectangle():
    """完美 axis-aligned 矩形 → 4 角應該與 bbox 4 角幾乎相同 (< 2 px 誤差)"""
    # 建立 4000x3000 黑底，中心 (500,400)-(3500,2600) 白矩形
    binary = np.zeros((3000, 4000), dtype=np.uint8)
    binary[400:2600, 500:3500] = 255
    bbox = (500, 400, 3500, 2600)

    polygon = _polyfit_polygon(binary, bbox, tile_size=512)
    assert polygon is not None, "理想矩形偵測不應該失敗"
    assert polygon.shape == (4, 2)
    assert polygon.dtype == np.float32

    expected = np.array([
        [500, 400],   # TL
        [3500, 400],  # TR
        [3500, 2600], # BR
        [500, 2600],  # BL
    ], dtype=np.float32)
    diff = np.abs(polygon - expected).max()
    assert diff < 2.0, f"ideal rect 誤差過大: {diff:.1f}px (per-corner max)"
    print(f"✅ test_polygon_detect_ideal_rectangle (max err={diff:.2f}px)")


def test_polygon_detect_degenerate_all_black():
    """全黑圖 → 應該回傳 None"""
    binary = np.zeros((3000, 4000), dtype=np.uint8)
    bbox = (0, 0, 4000, 3000)
    polygon = _polyfit_polygon(binary, bbox, tile_size=512)
    assert polygon is None, f"全黑圖應該回傳 None，實際 {polygon}"
    print("✅ test_polygon_detect_degenerate_all_black")


def test_polygon_detect_degenerate_tiny_noise():
    """只有小雜點 (面積太小) → 應該回傳 None (MIN_POLYGON_AREA_RATIO 檢查)"""
    binary = np.zeros((3000, 4000), dtype=np.uint8)
    # 一個 100x100 白點，遠小於 bbox
    binary[1000:1100, 1000:1100] = 255
    bbox = (0, 0, 4000, 3000)  # 故意給大 bbox
    polygon = _polyfit_polygon(binary, bbox, tile_size=512)
    assert polygon is None, f"小雜點應回傳 None，實際 {polygon}"
    print("✅ test_polygon_detect_degenerate_tiny_noise")


def test_polygon_stabilizes_near_vertical_sides_with_bad_top_rows():
    """左右邊上端局部缺角時，小尺寸 robust polygon 不應被拉成斜邊。"""
    binary = np.zeros((800, 1000), dtype=np.uint8)
    binary[100:700, 100:900] = 255
    binary[120:220, 100:260] = 0
    binary[120:220, 740:900] = 0
    bbox = (100, 100, 900, 700)

    polygon = _polyfit_polygon(
        binary,
        bbox,
        tile_size=512,
        stabilize_near_vertical_edges=True,
    )

    assert polygon is not None
    left_delta = abs(float(polygon[0, 0] - polygon[3, 0]))
    right_delta = abs(float(polygon[1, 0] - polygon[2, 0]))
    assert left_delta <= 2.0, f"左邊仍被上端缺角拉歪: {polygon.round(1).tolist()}"
    assert right_delta <= 2.0, f"右邊仍被上端缺角拉歪: {polygon.round(1).tolist()}"
    assert abs(float(polygon[:, 0].min()) - 100.0) <= 2.0
    assert abs(float(polygon[:, 0].max()) - 899.0) <= 2.0


def test_detect_panel_polygon_keeps_upper_band_connected():
    """上緣若被細小黑縫切開，detect_panel_polygon 仍應抓到完整外框。"""
    img = np.full((768, 1366), 18, dtype=np.uint8)
    img[72:112, 110:1256] = 180
    img[120:720, 100:1266] = 210
    img[112:120, 100:1266] = 18

    bbox, polygon = detect_panel_polygon(
        img,
        PreprocessConfig(tile_size=512, product_resolution=(1366, 768)),
    )
    assert bbox is not None
    assert polygon is not None
    x1, y1, x2, y2 = bbox
    assert x1 <= 110
    assert y1 <= 80, f"top band 被吃掉了: bbox={bbox}"
    assert x2 >= 1250
    assert y2 >= 710
    assert polygon[:, 1].min() <= 80, f"polygon top edge 太低: {polygon.round(1).tolist()}"
    print(f"✅ test_detect_panel_polygon_keeps_upper_band_connected (bbox={bbox})")


def test_detect_panel_polygon_includes_low_contrast_upper_band():
    """小尺寸產品上緣為暗帶時，不能只抓到下方亮區邊界。"""
    img = np.full((768, 1366), 18, dtype=np.uint8)
    img[72:120, 100:1266] = 28
    img[120:720, 100:1266] = 210

    bbox, polygon = detect_panel_polygon(
        img,
        PreprocessConfig(tile_size=512, product_resolution=(1366, 768)),
    )

    assert bbox is not None
    assert polygon is not None
    assert bbox[1] <= 80, f"低對比上緣被吃掉了: bbox={bbox}"
    assert polygon[:, 1].min() <= 80, f"polygon top edge 太低: {polygon.round(1).tolist()}"


def test_detect_panel_polygon_includes_low_contrast_side_bands():
    """小尺寸產品左右為暗帶時，不能只抓到中間亮區邊界。"""
    img = np.full((768, 1366), 18, dtype=np.uint8)
    img[100:720, 72:120] = 28
    img[100:720, 120:1266] = 210
    img[100:720, 1266:1314] = 28

    bbox, polygon = detect_panel_polygon(
        img,
        PreprocessConfig(tile_size=512, product_resolution=(1366, 768)),
    )

    assert bbox is not None
    assert polygon is not None
    assert bbox[0] <= 80, f"低對比左緣被吃掉了: bbox={bbox}"
    assert bbox[2] >= 1306, f"低對比右緣被吃掉了: bbox={bbox}"
    assert polygon[:, 0].min() <= 80, f"polygon left edge 太靠內: {polygon.round(1).tolist()}"
    assert polygon[:, 0].max() >= 1306, f"polygon right edge 太靠內: {polygon.round(1).tolist()}"


def test_detect_panel_polygon_rejects_wide_outer_side_shadow():
    """外側大塊陰影不應被當作低對比產品側邊。"""
    img = np.full((768, 1366), 18, dtype=np.uint8)
    img[100:720, 120:1000] = 210
    img[100:720, 1000:1160] = 50

    bbox, polygon = detect_panel_polygon(
        img,
        PreprocessConfig(tile_size=512, product_resolution=(1366, 768)),
    )

    assert bbox is not None
    assert polygon is not None
    assert bbox[2] <= 1025, f"右側外部陰影被誤當產品: bbox={bbox}"
    assert polygon[:, 0].max() <= 1025, f"polygon right 吃到外部陰影: {polygon.round(1).tolist()}"


def test_detect_panel_polygon_uses_gray_band_shift_for_boundary_preprocess(monkeypatch):
    """灰階分段映射若有配置，邊界偵測應先吃這一步。"""
    img = np.full((768, 1366), 18, dtype=np.uint8)
    cfg = PreprocessConfig(
        tile_size=512,
        product_resolution=(1366, 768),
        image_preprocess_pipeline=[
            {
                "method": "gray_band_shift",
                "params": {
                    "low_threshold": 105,
                    "high_threshold": 110,
                    "dark_shift": 10,
                    "bright_shift": 10,
                    "band_mode": "keep",
                },
            },
        ],
    )

    calls = []

    def fake_apply_preprocess_method(image, method, params=None):
        calls.append((method, params))
        processed = np.full_like(image, 18)
        processed[72:120, 100:1266] = 28
        processed[120:720, 100:1266] = 210
        return {
            "image": processed,
            "method": method,
            "method_label": "分段灰階映射",
            "applied_params": params or {},
            "notes": [],
            "conversion": {},
            "stats": {},
        }

    monkeypatch.setattr("capi_preprocess.apply_preprocess_method", fake_apply_preprocess_method)

    bbox, polygon = detect_panel_polygon(img, cfg)

    assert calls == [("gray_band_shift", cfg.image_preprocess_pipeline[0]["params"])]
    assert bbox is not None
    assert polygon is not None
    assert bbox[1] <= 80, f"邊界前處理未被套用: bbox={bbox}"
    assert polygon[:, 1].min() <= 80, f"polygon top edge 太低: {polygon.round(1).tolist()}"


def test_detect_panel_polygon_uses_default_boundary_gray_band_shift(monkeypatch):
    """未設定使用者前處理時，找邊仍應套 boundary-only 分段灰階映射。"""
    img = np.full((768, 1366), 18, dtype=np.uint8)
    cfg = PreprocessConfig(
        tile_size=512,
        product_resolution=(1366, 768),
        preprocess_after_tiling=True,
        image_preprocess_pipeline=[],
    )

    calls = []

    def fake_apply_preprocess_method(image, method, params=None):
        calls.append((method, params))
        processed = np.full_like(image, 18)
        processed[72:120, 100:1266] = 28
        processed[120:720, 100:1266] = 210
        return {
            "image": processed,
            "method": method,
            "method_label": "分段灰階映射",
            "applied_params": params or {},
            "notes": [],
            "conversion": {},
            "stats": {},
        }

    monkeypatch.setattr("capi_preprocess.apply_preprocess_method", fake_apply_preprocess_method)

    bbox, polygon = detect_panel_polygon(img, cfg)

    assert calls == [("gray_band_shift", BOUNDARY_GRAY_BAND_SHIFT_PARAMS)]
    assert bbox is not None
    assert polygon is not None
    assert bbox[1] <= 80, f"預設邊界前處理未被套用: bbox={bbox}"


def test_polygon_corner_ordering():
    """4 角順序必須是 TL, TR, BR, BL"""
    binary = np.zeros((2000, 3000), dtype=np.uint8)
    binary[300:1700, 500:2500] = 255
    bbox = (500, 300, 2500, 1700)
    polygon = _polyfit_polygon(binary, bbox, tile_size=512)
    assert polygon is not None
    TL, TR, BR, BL = polygon
    assert TL[0] < TR[0], f"TL.x ({TL[0]}) 必須 < TR.x ({TR[0]})"
    assert BL[0] < BR[0], f"BL.x ({BL[0]}) 必須 < BR.x ({BR[0]})"
    assert TL[1] < BL[1], f"TL.y ({TL[1]}) 必須 < BL.y ({BL[1]})"
    assert TR[1] < BR[1], f"TR.y ({TR[1]}) 必須 < BR.y ({BR[1]})"
    print("✅ test_polygon_corner_ordering")


def test_polyfit_polygon_rejects_non_linear_edge_samples():
    """彎曲/污染的邊緣點不可只因面積夠大就被當成有效 polygon。"""
    binary = np.zeros((2000, 3000), dtype=np.uint8)
    for x in range(500, 2500):
        normalized_x = (x - 1500) / 1000
        top = int(round(300 + 80 * normalized_x * normalized_x))
        binary[top:1700, x] = 255

    polygon = _polyfit_polygon(binary, (500, 300, 2500, 1700), tile_size=512)

    assert polygon is None


def test_detect_panel_boundary_half_scale_restores_full_resolution(monkeypatch):
    import capi_preprocess

    image = np.zeros((2200, 3000), dtype=np.uint8)
    captured = {}
    scaled_polygon = np.array(
        [[50, 100], [1450, 100], [1450, 1000], [50, 1000]],
        np.float32,
    )

    def fake_detect_panel_polygon(
        boundary_image,
        config,
        *,
        side_endpoint_trim_ratio,
    ):
        captured["shape"] = boundary_image.shape
        captured["tile_size"] = config.tile_size
        captured["otsu_offset"] = config.otsu_offset
        captured["side_trim"] = side_endpoint_trim_ratio
        return (50, 100, 1450, 1000), scaled_polygon

    monkeypatch.setattr(
        capi_preprocess,
        "detect_panel_polygon",
        fake_detect_panel_polygon,
    )

    bbox, polygon = detect_panel_boundary(
        image,
        PreprocessConfig(
            tile_size=512,
            tile_stride=512,
            otsu_offset=5,
            product_resolution=(1920, 1200),
        ),
    )

    assert captured == {
        "shape": (1100, 1500),
        "tile_size": 256,
        "otsu_offset": 2,
        "side_trim": 0.15,
    }
    assert bbox == (100, 200, 2900, 2000)
    np.testing.assert_array_equal(polygon, scaled_polygon * 2)


def test_detect_panel_boundary_still_rejects_curved_large_panel():
    image = np.zeros((2200, 3000), dtype=np.uint8)
    for x in range(500, 2500):
        normalized_x = (x - 1500) / 1000
        top = int(round(300 + 80 * normalized_x * normalized_x))
        image[top:1900, x] = 255

    _bbox, polygon = detect_panel_boundary(
        image,
        PreprocessConfig(
            tile_size=512,
            product_resolution=(1920, 1200),
        ),
    )

    assert polygon is None


def test_detect_panel_boundary_still_rejects_curved_vertical_edge():
    image = np.zeros((2200, 3000), dtype=np.uint8)
    for y in range(300, 1900):
        normalized_y = (y - 1100) / 800
        left = int(round(500 + 100 * normalized_y * normalized_y))
        image[y, left:2500] = 255

    _bbox, polygon = detect_panel_boundary(
        image,
        PreprocessConfig(
            tile_size=512,
            product_resolution=(1920, 1200),
        ),
    )

    assert polygon is None


def test_aapi_large_panel_raw_boundary_requires_large_frame_occupancy(monkeypatch):
    import capi_preprocess

    image = np.zeros((1000, 1000), dtype=np.uint8)
    polygon = np.array(
        [[50, 100], [950, 100], [950, 900], [50, 900]],
        np.float32,
    )
    detected = {"bbox": (100, 100, 900, 900)}

    def fake_detect_panel_boundary(image, config, *, source_name=""):
        return detected["bbox"], polygon

    monkeypatch.setattr(
        capi_preprocess,
        "detect_panel_boundary",
        fake_detect_panel_boundary,
    )

    _bbox, _polygon, large_occupancy = _detect_aapi_large_panel_raw_boundary(
        image,
        PreprocessConfig(),
    )
    assert large_occupancy is False

    detected["bbox"] = (50, 100, 950, 900)
    bbox, returned_polygon, large_occupancy = _detect_aapi_large_panel_raw_boundary(
        image,
        PreprocessConfig(),
    )
    assert large_occupancy is True
    assert bbox == detected["bbox"]
    np.testing.assert_array_equal(returned_polygon, polygon)


def test_reference_polygon_not_double_shrunk():
    """
    回歸 I1: 傳入 reference_polygon 時，calculate_otsu_bounds 不應改動 polygon。
    Task 6 的 B0F 路徑會依賴這個保證。
    """
    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.otsu_offset = 10
    cfg.otsu_bottom_crop = 0
    inf = CAPIInferencer(cfg)

    # 用一張合成圖 (4000x3000 黑底 + 大白矩形)，OTSU 會成功偵測
    synthetic = np.zeros((3000, 4000), dtype=np.uint8)
    synthetic[200:2800, 300:3700] = 200  # 亮度 200 > OTSU threshold

    # 第一次: 讓 calculate_otsu_bounds 自己算 polygon。
    bounds1, _, polygon1 = inf.calculate_otsu_bounds(synthetic)
    assert polygon1 is not None, "第一次必須算出 polygon"

    # 第二次: 傳入 polygon1 當 reference_polygon，結果應該跟 polygon1 相同
    _, _, polygon2 = inf.calculate_otsu_bounds(
        synthetic,
        reference_polygon=polygon1,
    )
    assert polygon2 is not None
    diff = float(np.abs(polygon1 - polygon2).max())
    assert diff < 0.01, \
        f"reference_polygon 被雙重內縮: max diff={diff:.3f}px (應為 0)"
    print(f"✅ test_reference_polygon_not_double_shrunk (max diff={diff:.4f}px)")


def test_calculate_otsu_bounds_keeps_polygon_on_raw_boundary():
    """推論紀錄紅框應畫產品外框；otsu_offset 只內縮 tile bounds。"""
    cfg = CAPIConfig()
    cfg.machine_id = "ABCDEB"
    cfg.tile_size = 512
    cfg.otsu_offset = 20
    cfg.otsu_bottom_crop = 0
    inf = CAPIInferencer(cfg)

    synthetic = np.zeros((768, 1366), dtype=np.uint8)
    synthetic[72:720, 100:1266] = 200

    bounds, _, polygon = inf.calculate_otsu_bounds(synthetic)

    assert bounds == (120, 92, 1246, 700)
    assert polygon is not None
    assert polygon[:, 0].min() <= 102, f"polygon left edge 被 offset 內縮: {polygon.round(1).tolist()}"
    assert polygon[:, 1].min() <= 74, f"polygon top edge 被 offset 內縮: {polygon.round(1).tolist()}"
    assert polygon[:, 0].max() >= 1264, f"polygon right edge 被 offset 內縮: {polygon.round(1).tolist()}"
    assert polygon[:, 1].max() >= 718, f"polygon bottom edge 被 offset 內縮: {polygon.round(1).tolist()}"
    print(f"✅ test_calculate_otsu_bounds_keeps_polygon_on_raw_boundary (bounds={bounds})")


def test_calculate_otsu_bounds_includes_low_contrast_outer_bands():
    """推論總覽 polygon 也要抓到小尺寸產品的低對比外框。"""
    cfg = CAPIConfig()
    cfg.machine_id = "ABCDEB"
    cfg.tile_size = 512
    cfg.otsu_offset = 20
    cfg.otsu_bottom_crop = 0
    inf = CAPIInferencer(cfg)

    synthetic = np.full((768, 1366), 18, dtype=np.uint8)
    synthetic[72:120, 72:1314] = 28
    synthetic[120:720, 72:120] = 28
    synthetic[120:720, 120:1266] = 210
    synthetic[120:720, 1266:1314] = 28

    bounds, _, polygon = inf.calculate_otsu_bounds(synthetic)

    assert bounds[0] <= 100, f"推論 bounds 左緣太靠內: {bounds}"
    assert bounds[1] <= 100, f"推論 bounds 上緣太靠內: {bounds}"
    assert bounds[2] >= 1290, f"推論 bounds 右緣太靠內: {bounds}"
    assert polygon is not None
    assert polygon[:, 0].min() <= 80, f"推論 polygon left 太靠內: {polygon.round(1).tolist()}"
    assert polygon[:, 1].min() <= 80, f"推論 polygon top 太靠內: {polygon.round(1).tolist()}"
    assert polygon[:, 0].max() >= 1306, f"推論 polygon right 太靠內: {polygon.round(1).tolist()}"


def test_calculate_otsu_bounds_legacy_machine_still_shrinks_polygon():
    """舊尺寸機種維持原本 polygon offset 行為，避免影響既有模型。"""
    cfg = CAPIConfig()
    cfg.machine_id = "ABCDEH"
    cfg.tile_size = 512
    cfg.otsu_offset = 20
    cfg.otsu_bottom_crop = 0
    inf = CAPIInferencer(cfg)

    synthetic = np.zeros((768, 1366), dtype=np.uint8)
    synthetic[72:720, 100:1266] = 200

    bounds, _, polygon = inf.calculate_otsu_bounds(synthetic)

    assert bounds == (120, 92, 1246, 700)
    assert polygon is not None
    assert polygon[:, 0].min() > 110, f"legacy polygon 未維持 offset 內縮: {polygon.round(1).tolist()}"
    assert polygon[:, 1].min() > 78, f"legacy polygon 未維持 offset 內縮: {polygon.round(1).tolist()}"
    print(f"✅ test_calculate_otsu_bounds_legacy_machine_still_shrinks_polygon (bounds={bounds})")


def test_bottom_crop_preserves_polygon_tilt():
    """
    回歸 I3: otsu_bottom_crop 觸發時，polygon 底邊應以 left/right edge 與新
    底線的交點為新 BR/BL (保留傾斜度)，而不是只改 y 保留原 x。

    用一個「左右 side edge 都斜」的平行四邊形驗證:
      - TL (400, 100), TR (3600, 100)
      - BL (200, 2800), BR (3800, 2800)  ← 底部比頂部寬 400 px
    Side edges 斜率明確 → intersection 交點 x 與原 BL/BR x 有可量化的差異
    """
    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.otsu_offset = 0
    cfg.otsu_bottom_crop = 500
    inf = CAPIInferencer(cfg)

    synthetic = np.zeros((3000, 4000), dtype=np.uint8)
    parallelogram = np.array([
        [400, 100],    # TL
        [3600, 100],   # TR
        [3800, 2800],  # BR (右下比 TR 偏右 200 px)
        [200, 2800],   # BL (左下比 TL 偏左 200 px)
    ], dtype=np.int32)
    cv2.fillPoly(synthetic, [parallelogram], 200)

    bounds, original_y2, polygon = inf.calculate_otsu_bounds(synthetic)
    assert polygon is not None
    assert original_y2 is not None, "bottom_crop 應該觸發"

    y_end = bounds[3]
    BR = polygon[2]
    BL = polygon[3]
    TR = polygon[1]
    TL = polygon[0]

    # 裁切後 BR/BL 的 y 都應等於 new_bottom (= y_end)
    assert abs(float(BR[1]) - y_end) < 0.5, \
        f"BR.y={BR[1]} 應貼齊 y_end={y_end}"
    assert abs(float(BL[1]) - y_end) < 0.5, \
        f"BL.y={BL[1]} 應貼齊 y_end={y_end}"

    # 核心驗證: 底邊的 x 應該是「side edge 與新底線的交點」
    # Left edge 方程 (TL → BL) 在 y=y_end 處的 x:
    def _interp_x(p_top, p_bot, y_line):
        dy = float(p_bot[1]) - float(p_top[1])
        if abs(dy) < 1e-9:
            return float(p_top[0])
        t = (y_line - float(p_top[1])) / dy
        return float(p_top[0]) + t * (float(p_bot[0]) - float(p_top[0]))

    expected_BL_x = _interp_x(TL, np.array([200.0, 2800.0]), y_end)
    expected_BR_x = _interp_x(TR, np.array([3800.0, 2800.0]), y_end)

    # 容忍 polyfit 偵測的數值誤差
    assert abs(float(BL[0]) - expected_BL_x) < 5.0, \
        f"BL.x={BL[0]} 應該 ≈ interp({expected_BL_x:.1f})，差距過大"
    assert abs(float(BR[0]) - expected_BR_x) < 5.0, \
        f"BR.x={BR[0]} 應該 ≈ interp({expected_BR_x:.1f})，差距過大"

    # 舊 code 會把 BR.x/BL.x 保留成原 polygon 的 BR/BL x (3800/200)
    # 新 code 應該產出明顯向內的 x (因為裁切後的 y 更靠近頂部)
    # Diff = expected 交點 vs 原 BL/BR x
    diff_BL_from_orig = abs(float(BL[0]) - 200.0)
    diff_BR_from_orig = abs(float(BR[0]) - 3800.0)
    assert diff_BL_from_orig > 5.0 or diff_BR_from_orig > 5.0, \
        f"BL/BR x 完全沒動 (BL.x={BL[0]}, BR.x={BR[0]})，疑似舊 buggy 行為"

    print(f"✅ test_bottom_crop_preserves_polygon_tilt "
          f"(BR={BR.round(1).tolist()} [expect x≈{expected_BR_x:.1f}], "
          f"BL={BL.round(1).tolist()} [expect x≈{expected_BL_x:.1f}])")


def test_exclusion_region_uses_polygon_br_anchor():
    """relative_bottom_right 排除區應以 polygon BR 為錨點。"""
    from capi_config import ExclusionZone

    cfg = CAPIConfig()
    cfg.otsu_bottom_crop = 0
    cfg.exclusion_zones = [
        ExclusionZone(
            name="test_br",
            type="relative_bottom_right",
            width=300,
            height=200,
            enabled=True,
        ),
    ]
    inf = CAPIInferencer(cfg)
    image = np.zeros((1000, 1000), dtype=np.uint8)
    polygon = np.array(
        [[100, 100], [900, 100], [850, 900], [150, 900]],
        dtype=np.float32,
    )

    regions = inf.calculate_exclusion_regions(
        image,
        otsu_bounds=(100, 100, 900, 900),
        panel_polygon=polygon,
    )

    assert len(regions) == 1
    assert (regions[0].x1, regions[0].y1, regions[0].x2, regions[0].y2) == (
        550,
        700,
        850,
        900,
    )
