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

    print("\n✅ 所有測試通過")
