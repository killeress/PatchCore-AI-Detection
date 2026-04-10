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


def _make_inferencer():
    """建立一個不需要模型載入的 inferencer instance"""
    cfg = CAPIConfig()
    cfg.tile_size = 512
    return CAPIInferencer(cfg)


def test_polygon_detect_ideal_rectangle():
    """完美 axis-aligned 矩形 → 4 角應該與 bbox 4 角幾乎相同 (< 2 px 誤差)"""
    inf = _make_inferencer()
    # 建立 4000x3000 黑底，中心 (500,400)-(3500,2600) 白矩形
    binary = np.zeros((3000, 4000), dtype=np.uint8)
    binary[400:2600, 500:3500] = 255
    bbox = (500, 400, 3500, 2600)

    polygon = inf._find_panel_polygon(binary, bbox)
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
    inf = _make_inferencer()
    binary = np.zeros((3000, 4000), dtype=np.uint8)
    bbox = (0, 0, 4000, 3000)
    polygon = inf._find_panel_polygon(binary, bbox)
    assert polygon is None, f"全黑圖應該回傳 None，實際 {polygon}"
    print("✅ test_polygon_detect_degenerate_all_black")


def test_polygon_detect_degenerate_tiny_noise():
    """只有小雜點 (面積太小) → 應該回傳 None (MIN_POLYGON_AREA_RATIO 檢查)"""
    inf = _make_inferencer()
    binary = np.zeros((3000, 4000), dtype=np.uint8)
    # 一個 100x100 白點，遠小於 bbox
    binary[1000:1100, 1000:1100] = 255
    bbox = (0, 0, 4000, 3000)  # 故意給大 bbox
    polygon = inf._find_panel_polygon(binary, bbox)
    assert polygon is None, f"小雜點應回傳 None，實際 {polygon}"
    print("✅ test_polygon_detect_degenerate_tiny_noise")


def test_polygon_detect_real_W0F():
    """真實影像 W0F00000_110022.tif 的 4 角誤差應該符合 spec 預期值"""
    img_path = Path(__file__).resolve().parent.parent / "test_images" / "W0F00000_110022.tif"
    if not img_path.exists():
        print(f"⚠️  跳過 (測試圖不存在): {img_path}")
        return

    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    inf = _make_inferencer()
    bbox, binary = inf._find_raw_object_bounds(gray)
    polygon = inf._find_panel_polygon(binary, bbox)
    assert polygon is not None

    # Spec 9.1 預期: TL=6.6, TR=15.7, BR=15.0, BL=0.6 (±2 px 容忍)
    bbox_corners = np.array([
        [bbox[0], bbox[1]], [bbox[2], bbox[1]],
        [bbox[2], bbox[3]], [bbox[0], bbox[3]],
    ], dtype=np.float32)
    errs = np.linalg.norm(polygon - bbox_corners, axis=1)
    expected = np.array([6.6, 15.7, 15.0, 0.6])
    diff = np.abs(errs - expected).max()
    assert diff < 2.0, \
        f"W0F 4 角誤差與 spec 不符: expected {expected}, got {errs}, max diff {diff:.2f}px"
    print(f"✅ test_polygon_detect_real_W0F (errs={errs.round(1).tolist()})")


def test_polygon_detect_real_G0F():
    """真實影像 G0F00000_151955.tif 的 4 角誤差應該符合 spec 預期值"""
    img_path = Path(__file__).resolve().parent.parent / "test_images" / "G0F00000_151955.tif"
    if not img_path.exists():
        print(f"⚠️  跳過 (測試圖不存在): {img_path}")
        return

    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    inf = _make_inferencer()
    bbox, binary = inf._find_raw_object_bounds(gray)
    polygon = inf._find_panel_polygon(binary, bbox)
    assert polygon is not None

    # Spec 9.1 預期: TL=16.7, TR=19.0, BR=36.9, BL=3.8 (±2 px 容忍)
    bbox_corners = np.array([
        [bbox[0], bbox[1]], [bbox[2], bbox[1]],
        [bbox[2], bbox[3]], [bbox[0], bbox[3]],
    ], dtype=np.float32)
    errs = np.linalg.norm(polygon - bbox_corners, axis=1)
    expected = np.array([16.7, 19.0, 36.9, 3.8])
    diff = np.abs(errs - expected).max()
    assert diff < 2.0, \
        f"G0F 4 角誤差與 spec 不符: expected {expected}, got {errs}, max diff {diff:.2f}px"
    print(f"✅ test_polygon_detect_real_G0F (errs={errs.round(1).tolist()})")


def test_polygon_corner_ordering():
    """4 角順序必須是 TL, TR, BR, BL"""
    inf = _make_inferencer()
    binary = np.zeros((2000, 3000), dtype=np.uint8)
    binary[300:1700, 500:2500] = 255
    bbox = (500, 300, 2500, 1700)
    polygon = inf._find_panel_polygon(binary, bbox)
    assert polygon is not None
    TL, TR, BR, BL = polygon
    assert TL[0] < TR[0], f"TL.x ({TL[0]}) 必須 < TR.x ({TR[0]})"
    assert BL[0] < BR[0], f"BL.x ({BL[0]}) 必須 < BR.x ({BR[0]})"
    assert TL[1] < BL[1], f"TL.y ({TL[1]}) 必須 < BL.y ({BL[1]})"
    assert TR[1] < BR[1], f"TR.y ({TR[1]}) 必須 < BR.y ({BR[1]})"
    print("✅ test_polygon_corner_ordering")


def test_preprocess_image_populates_panel_polygon():
    """preprocess_image 跑完後 result.panel_polygon 必須是 (4,2) float32"""
    img_path = Path(__file__).resolve().parent.parent / "test_images" / "G0F00000_151955.tif"
    if not img_path.exists():
        print(f"⚠️  跳過 (測試圖不存在): {img_path}")
        return

    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.tile_stride = 512
    cfg.otsu_bottom_crop = 0  # 不做 bottom crop 以便直接比對
    inf = CAPIInferencer(cfg)
    result = inf.preprocess_image(img_path)
    assert result is not None
    assert result.panel_polygon is not None, "polygon 應該要被計算"
    assert result.panel_polygon.shape == (4, 2)
    assert result.panel_polygon.dtype == np.float32
    print(f"✅ test_preprocess_image_populates_panel_polygon "
          f"(polygon={result.panel_polygon.round(1).tolist()})")


def test_preprocess_image_polygon_disabled_when_toggle_off():
    """enable_panel_polygon=False 時 panel_polygon 必須為 None"""
    img_path = Path(__file__).resolve().parent.parent / "test_images" / "G0F00000_151955.tif"
    if not img_path.exists():
        print(f"⚠️  跳過 (測試圖不存在): {img_path}")
        return

    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.tile_stride = 512
    cfg.otsu_bottom_crop = 0
    cfg.enable_panel_polygon = False
    inf = CAPIInferencer(cfg)
    result = inf.preprocess_image(img_path)
    assert result is not None
    assert result.panel_polygon is None, \
        f"toggle off 時 polygon 應為 None，實際 {result.panel_polygon}"
    print("✅ test_preprocess_image_polygon_disabled_when_toggle_off")


def test_reference_polygon_not_double_shrunk():
    """
    回歸 I1: 傳入 reference_polygon 時，calculate_otsu_bounds 不應再次內縮。
    Task 6 的 B0F 路徑會依賴這個保證。
    """
    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.otsu_offset = 10  # 明顯的 offset，雙重內縮會產生明顯差異
    cfg.otsu_bottom_crop = 0
    inf = CAPIInferencer(cfg)

    # 用一張合成圖 (4000x3000 黑底 + 大白矩形)，OTSU 會成功偵測
    synthetic = np.zeros((3000, 4000), dtype=np.uint8)
    synthetic[200:2800, 300:3700] = 200  # 亮度 200 > OTSU threshold

    # 第一次: 讓 calculate_otsu_bounds 自己算 polygon (會經過 offset 內縮)
    bounds1, _, polygon1 = inf.calculate_otsu_bounds(synthetic)
    assert polygon1 is not None, "第一次必須算出 polygon"

    # 第二次: 傳入 polygon1 當 reference_polygon，結果應該跟 polygon1 相同
    # (reference_polygon 已經是內縮過的，不應被再次內縮)
    _, _, polygon2 = inf.calculate_otsu_bounds(
        synthetic,
        reference_polygon=polygon1,
    )
    assert polygon2 is not None
    diff = float(np.abs(polygon1 - polygon2).max())
    assert diff < 0.01, \
        f"reference_polygon 被雙重內縮: max diff={diff:.3f}px (應為 0)"
    print(f"✅ test_reference_polygon_not_double_shrunk (max diff={diff:.4f}px)")


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
    """
    relative_bottom_right 排除區應以 polygon BR 為錨點，
    而不是 bbox 右下角。
    """
    from capi_config import ExclusionZone

    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.tile_stride = 512
    cfg.otsu_bottom_crop = 0
    cfg.otsu_offset = 0
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

    img_path = Path(__file__).resolve().parent.parent / "test_images" / "G0F00000_151955.tif"
    if not img_path.exists():
        print(f"⚠️  跳過 (測試圖不存在): {img_path}")
        return

    result = inf.preprocess_image(img_path)
    assert result is not None
    assert result.panel_polygon is not None

    # 找到 test_br 排除區
    br_regions = [r for r in result.exclusion_regions if r.name == "test_br"]
    assert len(br_regions) == 1, f"應該找到 1 個 test_br 排除區，實際 {len(br_regions)}"
    br = br_regions[0]

    # 排除區的 x2/y2 必須接近 polygon BR，不能是 bbox 右下
    poly_br_x, poly_br_y = int(round(result.panel_polygon[2][0])), int(round(result.panel_polygon[2][1]))
    bbox_x2, bbox_y2 = result.otsu_bounds[2], result.otsu_bounds[3]

    # G0F 這張 polygon BR 與 bbox 右下差 ~37 px
    assert abs(br.x2 - poly_br_x) <= 1, \
        f"排除區 x2={br.x2} 應接近 polygon BR x={poly_br_x}，而非 bbox x2={bbox_x2}"
    assert abs(br.y2 - poly_br_y) <= 1, \
        f"排除區 y2={br.y2} 應接近 polygon BR y={poly_br_y}，而非 bbox y2={bbox_y2}"
    print(f"✅ test_exclusion_region_uses_polygon_br_anchor "
          f"(br=({br.x2},{br.y2}), poly BR=({poly_br_x},{poly_br_y}), "
          f"bbox BR=({bbox_x2},{bbox_y2}))")


if __name__ == "__main__":
    test_config_enable_panel_polygon_default_true()
    test_config_roundtrip_enable_panel_polygon()
    test_config_yaml_loads_enable_panel_polygon()

    test_polygon_detect_ideal_rectangle()
    test_polygon_detect_degenerate_all_black()
    test_polygon_detect_degenerate_tiny_noise()
    test_polygon_detect_real_W0F()
    test_polygon_detect_real_G0F()
    test_polygon_corner_ordering()

    test_preprocess_image_populates_panel_polygon()
    test_preprocess_image_polygon_disabled_when_toggle_off()
    test_reference_polygon_not_double_shrunk()
    test_bottom_crop_preserves_polygon_tilt()
    test_exclusion_region_uses_polygon_br_anchor()

    print("\n✅ 所有測試通過")
