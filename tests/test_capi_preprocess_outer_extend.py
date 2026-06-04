"""capi_preprocess.outer_edge_extend：edge anchor 外推 + polygon 內推取樣。"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from capi_preprocess import (
    PreprocessConfig,
    _generate_tiles,
    detect_panel_polygon,
)


def _make_panel_image(img_h: int, img_w: int, panel_xyxy):
    """建一張黑底 + 灰 panel 矩形的合成圖。"""
    px1, py1, px2, py2 = panel_xyxy
    img = np.zeros((img_h, img_w), dtype=np.uint8)
    img[py1:py2, px1:px2] = 180
    return img


def _detect(img, cfg=None):
    cfg = cfg or PreprocessConfig(enable_panel_polygon=False)
    bbox, polygon = detect_panel_polygon(img, cfg)
    return bbox, polygon


def test_default_outer_edge_extend_is_half_tile_size():
    cfg = PreprocessConfig()
    assert cfg.outer_edge_extend == cfg.tile_size // 2


def test_outer_edge_extend_adds_extension_tiles_for_centered_panel():
    """panel 在 image 中央、各邊距 >= 256，外推應該每邊多出一排 tile + 4 個角。"""
    img = _make_panel_image(2000, 2400, (500, 500, 1900, 1500))
    cfg_off = PreprocessConfig(enable_panel_polygon=False, outer_edge_extend=0)
    cfg_on = PreprocessConfig(enable_panel_polygon=False, outer_edge_extend=256)
    bbox, _ = _detect(img, cfg_off)

    base = _generate_tiles(img, bbox, polygon=None, config=cfg_off)
    extended = _generate_tiles(img, bbox, polygon=None, config=cfg_on)

    base_pos = {(t.x1, t.y1) for t in base}
    extra = [t for t in extended if (t.x1, t.y1) not in base_pos]
    assert len(extra) > 0
    assert all(t.zone == "edge" for t in extra), "extension tiles 必須是 edge"

    x1, y1, x2, y2 = bbox
    ts = cfg_on.tile_size
    extend = cfg_on.outer_edge_extend
    expected_top_y = y1 - extend
    expected_bot_y = y2 - ts + extend
    expected_left_x = x1 - extend
    expected_right_x = x2 - ts + extend
    extra_xy = {(t.x1, t.y1) for t in extra}
    assert any(y == expected_top_y for _, y in extra_xy), "缺 top 外推行"
    assert any(y == expected_bot_y for _, y in extra_xy), "缺 bottom 外推行"
    assert any(x == expected_left_x for x, _ in extra_xy), "缺 left 外推列"
    assert any(x == expected_right_x for x, _ in extra_xy), "缺 right 外推列"
    corners = {
        (expected_left_x, expected_top_y),
        (expected_right_x, expected_top_y),
        (expected_left_x, expected_bot_y),
        (expected_right_x, expected_bot_y),
    }
    assert corners.issubset(extra_xy), f"缺角落 tile，extra_xy={extra_xy}"


def test_outer_edge_extend_clamps_to_image_boundary():
    """panel 上邊離 image 上邊只有 100px 時，push_top 必須夾到 100,
    使 top 外推 tile 完全落在 image 內。"""
    img = _make_panel_image(1200, 2400, (500, 100, 1900, 1100))
    cfg = PreprocessConfig(enable_panel_polygon=False, outer_edge_extend=256)
    bbox, _ = _detect(img, cfg)
    x1, y1, x2, y2 = bbox

    expected_push_top = min(256, max(0, y1))
    expected_top_ty = y1 - expected_push_top
    assert expected_top_ty >= 0, "外推 tile 不可超出 image 上邊"

    tiles = _generate_tiles(img, bbox, polygon=None, config=cfg)
    top_extra = [t for t in tiles if t.y1 == expected_top_ty]
    assert top_extra, f"應產生 ty={expected_top_ty} 的外推 tile"
    for t in top_extra:
        assert t.image.shape[0] == cfg.tile_size
        assert t.image.shape[1] == cfg.tile_size


def test_outer_edge_extend_skips_side_with_no_margin():
    """panel 緊貼 image 上邊（y1=0）時，不產生 top 外推 tile。"""
    img = _make_panel_image(1200, 2400, (500, 0, 1900, 1100))
    cfg = PreprocessConfig(enable_panel_polygon=False, outer_edge_extend=256)
    bbox, _ = _detect(img, cfg)
    x1, y1, x2, y2 = bbox

    tiles = _generate_tiles(img, bbox, polygon=None, config=cfg)
    assert y1 < 256, "fixture 應使 y1 接近 0"
    extension_top_tiles = [t for t in tiles if t.y1 < 0]
    assert not extension_top_tiles, "panel 緊貼 image 上邊時不可產生 ty<0 的 tile"


def test_outer_edge_extend_zero_matches_legacy_behavior():
    """outer_edge_extend=0 → 結果與舊行為一致（無外推 tile）。"""
    img = _make_panel_image(2000, 2400, (500, 500, 1900, 1500))
    cfg_legacy = PreprocessConfig(enable_panel_polygon=False, outer_edge_extend=0)
    bbox, _ = _detect(img, cfg_legacy)
    tiles = _generate_tiles(img, bbox, polygon=None, config=cfg_legacy)

    x1, y1, x2, y2 = bbox
    for t in tiles:
        assert t.x1 >= x1 and t.y1 >= y1
        assert t.x2 <= x2 and t.y2 <= y2


def test_outer_edge_extend_corner_tile_shifted_inside_polygon():
    """polygon 開啟時，外推角落 anchor 只用來決定 edge 取樣位置；
    實際 tile 必須被推回產品內，不應再含黑色背景。"""
    img = _make_panel_image(2000, 2400, (500, 500, 1900, 1500))
    cfg = PreprocessConfig(enable_panel_polygon=True, outer_edge_extend=256)
    bbox, polygon = _detect(img, cfg)
    assert polygon is not None, "fixture 應能 fit polygon"

    tiles = _generate_tiles(img, bbox, polygon=polygon, config=cfg)
    corner_edges = [t for t in tiles if t.is_corner and t.zone == "edge"]
    assert corner_edges, "外推角落 anchor 應保留為 edge corner 樣本"
    assert all(t.coverage >= 0.999 for t in corner_edges)
    assert all(t.mask is None for t in corner_edges)
    assert all(int(t.image.min()) > 0 for t in corner_edges), "edge corner tile 不應包含黑色背景"


def test_inward_shifted_outer_edge_tiles_skip_overlap_duplicates():
    """外推 tile 若被 polygon 推回到既有 edge tile 上，不應產生第二層 edge 框。"""
    img = np.zeros((4480, 6720), dtype=np.uint8)
    bbox = (723, 674, 6217, 3758)
    polygon = np.array([
        [716.7, 694.5],
        [6222.0, 668.2],
        [6218.3, 3740.8],
        [740.3, 3755.7],
    ], dtype=np.float32)

    base_cfg = PreprocessConfig(tile_size=512, tile_stride=512, outer_edge_extend=0)
    ext_cfg = PreprocessConfig(tile_size=512, tile_stride=512, outer_edge_extend=256)

    base_tiles = _generate_tiles(img, bbox, polygon=polygon, config=base_cfg)
    ext_tiles = _generate_tiles(img, bbox, polygon=polygon, config=ext_cfg)

    base_positions = {(t.x1, t.y1) for t in base_tiles}
    duplicate_layer_edges = [
        t for t in ext_tiles
        if t.zone == "edge" and (t.x1, t.y1) not in base_positions
    ]

    assert not duplicate_layer_edges
    assert sum(t.zone == "edge" for t in ext_tiles) == sum(t.zone == "edge" for t in base_tiles)


def test_tile_positions_distribute_remainder_without_skinny_last_column():
    """bbox 寬度非 stride 整數倍時，不應在右側擠出一條窄重疊 tile。"""
    img = np.zeros((1200, 1600), dtype=np.uint8)
    bbox = (100, 100, 1500, 1100)
    polygon = np.array([
        [100, 100],
        [1500, 100],
        [1500, 1100],
        [100, 1100],
    ], dtype=np.float32)
    cfg = PreprocessConfig(tile_size=256, tile_stride=256, outer_edge_extend=0)

    tiles = _generate_tiles(img, bbox, polygon=polygon, config=cfg)
    top_edge_xs = sorted({
        t.x1 for t in tiles
        if t.zone == "edge" and t.y1 == bbox[1]
    })

    assert len(top_edge_xs) >= 2
    gaps = [b - a for a, b in zip(top_edge_xs, top_edge_xs[1:])]
    assert min(gaps) >= int(cfg.tile_size * 0.75), top_edge_xs
    assert top_edge_xs[0] == bbox[0]
    assert top_edge_xs[-1] == bbox[2] - cfg.tile_size
