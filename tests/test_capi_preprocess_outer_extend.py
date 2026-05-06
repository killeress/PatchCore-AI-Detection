"""capi_preprocess.outer_edge_extend：訓練 edge tile 外推取樣。"""
from __future__ import annotations
import numpy as np
import cv2
import pytest

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


def test_default_outer_edge_extend_is_256():
    cfg = PreprocessConfig()
    assert cfg.outer_edge_extend == 256


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


def test_outer_edge_extend_corner_tile_forced_to_edge_zone():
    """角落外推 tile coverage 約 25%（小於 coverage_min=0.3),
    若不強制 edge 會被歸 outside 跳掉；本測驗證強制成功。"""
    img = _make_panel_image(2000, 2400, (500, 500, 1900, 1500))
    cfg = PreprocessConfig(enable_panel_polygon=True, outer_edge_extend=256)
    bbox, polygon = _detect(img, cfg)
    assert polygon is not None, "fixture 應能 fit polygon"

    tiles = _generate_tiles(img, bbox, polygon=polygon, config=cfg)
    x1, y1, x2, y2 = bbox
    ts = cfg.tile_size
    corner_tl = (x1 - cfg.outer_edge_extend, y1 - cfg.outer_edge_extend)
    corner_tile = next((t for t in tiles if (t.x1, t.y1) == corner_tl), None)
    assert corner_tile is not None, "缺左上角外推 tile"
    assert corner_tile.zone == "edge"
    assert corner_tile.coverage < cfg.coverage_min, (
        f"corner coverage {corner_tile.coverage} 應小於 coverage_min "
        f"才能驗證 zone 強制 edge 的 bypass"
    )
