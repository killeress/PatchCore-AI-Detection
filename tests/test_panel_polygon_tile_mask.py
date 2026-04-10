"""tile_image 套用 panel polygon mask 的視覺回歸測試"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows cp950 console 對應
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import cv2
import numpy as np
from capi_config import CAPIConfig
from capi_inference import CAPIInferencer


TEST_IMAGES = [
    "test_images/W0F00000_110022.tif",
    "test_images/G0F00000_151955.tif",
]

OUT_DIR = Path(__file__).resolve().parent.parent / "test_output" / "panel_polygon_tile_mask"


def _run_one(rel_path: str):
    img_path = Path(__file__).resolve().parent.parent / rel_path
    if not img_path.exists():
        print(f"⚠️  跳過 (測試圖不存在): {img_path}")
        return None, None

    cfg = CAPIConfig()
    cfg.tile_size = 512
    cfg.tile_stride = 512
    cfg.otsu_bottom_crop = 0
    cfg.otsu_offset = 0  # 測試用: 不內縮以便精準比對 polygon 邊緣
    inf = CAPIInferencer(cfg)
    result = inf.preprocess_image(img_path)
    assert result is not None
    return result, img_path


def test_tile_mask_shapes_and_dtypes():
    """每個 tile 的 mask 要不是 None，要不是 shape (tile_size, tile_size) uint8"""
    for rel in TEST_IMAGES:
        result, img_path = _run_one(rel)
        if result is None:
            continue
        if result.panel_polygon is None:
            print(f"⚠️  {rel} polygon 偵測失敗，跳過 mask 檢查")
            continue
        for tile in result.tiles:
            if tile.mask is None:
                continue
            assert tile.mask.dtype == np.uint8, \
                f"tile {tile.tile_id} mask dtype {tile.mask.dtype}"
            assert tile.mask.shape == (tile.height, tile.width), \
                f"tile {tile.tile_id} mask shape {tile.mask.shape}"
        print(f"✅ test_tile_mask_shapes_and_dtypes / {Path(rel).name}")


def test_tile_mask_matches_polygon():
    """
    對每個有 mask 的 tile: mask==0 的像素必須在 polygon 外；
    mask==255 的像素必須在 polygon 內。
    """
    for rel in TEST_IMAGES:
        result, img_path = _run_one(rel)
        if result is None or result.panel_polygon is None:
            continue

        # 建立整張圖尺寸的 panel_mask ground truth
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        H, W = gray.shape
        gt_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(gt_mask, [result.panel_polygon.astype(np.int32)], 255)

        masked_tiles = 0
        for tile in result.tiles:
            if tile.mask is None:
                continue
            masked_tiles += 1
            gt_sub = gt_mask[tile.y:tile.y + tile.height,
                             tile.x:tile.x + tile.width]
            # Mask 應該與 ground truth 完全一致 (fillPoly 是 deterministic)
            mismatch = int(np.count_nonzero(tile.mask != gt_sub))
            total = tile.mask.size
            assert mismatch / total < 0.001, \
                f"tile {tile.tile_id} mask mismatch {mismatch}/{total} " \
                f"({mismatch/total:.3%})"

        print(f"✅ test_tile_mask_matches_polygon / {Path(rel).name} "
              f"({masked_tiles} 個邊緣 tile 有 mask)")


def test_tile_mask_visualization():
    """產生視覺化 PNG 供人工確認 (非 assertion 測試)"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rel in TEST_IMAGES:
        result, img_path = _run_one(rel)
        if result is None or result.panel_polygon is None:
            continue
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        overlay = vis.copy()

        for tile in result.tiles:
            # 畫 tile 框 (綠)
            cv2.rectangle(vis, (tile.x, tile.y),
                          (tile.x + tile.width, tile.y + tile.height),
                          (0, 255, 0), 3)
            if tile.mask is not None:
                excluded = tile.mask == 0
                if excluded.any():
                    roi = overlay[tile.y:tile.y + tile.height,
                                  tile.x:tile.x + tile.width]
                    roi[excluded] = (0, 0, 255)

        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
        # Polygon (黃) + bbox (藍)
        poly_int = result.panel_polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly_int], True, (0, 255, 255), 4)
        x1, y1, x2, y2 = result.otsu_bounds
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 4)

        scale = 0.25
        vis_small = cv2.resize(vis, (0, 0), fx=scale, fy=scale)
        out_path = OUT_DIR / f"{Path(rel).stem}_tile_mask.png"
        cv2.imwrite(str(out_path), vis_small)
        print(f"✅ 視覺化輸出: {out_path}")


if __name__ == "__main__":
    test_tile_mask_shapes_and_dtypes()
    test_tile_mask_matches_polygon()
    test_tile_mask_visualization()
    print("\n✅ 所有 tile mask 測試通過")
