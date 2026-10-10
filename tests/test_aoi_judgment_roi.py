"""AOI-only judgment: coordinates, scores, filters, reporting and rendered evidence."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from capi_config import BombDefect, CAPIConfig
from capi_database import CAPIDatabase
from capi_heatmap import HeatmapManager
from capi_inference import CAPIInferencer
from capi_server import results_to_db_data, _iter_qjpg_defect_records
from capi_tile_diagnostics import decision_evidence
from test_bomb_aoi_mapping import (make_tile, make_result, apply_bomb_postprocess,
                                  make_overlapping_point_bomb_case, RESOLUTION)


def case(points, *, bombs=(), enabled=True, anchor=(256, 256)):
    inf = CAPIInferencer.__new__(CAPIInferencer)
    inf.config = CAPIConfig(aoi_judgment_roi_enabled=enabled, bomb_match_tolerance=20,
                           bomb_defects=[BombDefect('STANDARD', 'B01', 'point', list(bombs))] if bombs else [])
    tile = make_tile(0, anchor, anchor)
    tile.x = tile.y = 0
    tile.score_threshold = .5
    tile.image[:] = 100
    amap = np.zeros((512, 512), dtype=np.float32)
    for x, y, value in points:
        amap[y, x] = value
    result = make_result([tile])
    result.raw_bounds = result.otsu_bounds = (0, 0, *RESOLUTION)
    result.panel_polygon = None
    result.anomaly_tiles = [(tile, float(amap.max()), amap)]
    return inf, result, tile


@pytest.mark.parametrize('version', ['v1', 'v2'])
@pytest.mark.parametrize('points,bombs,ng,bomb', [
    ([(400, 400, 1)], [], False, False),
    ([(256, 256, .7), (400, 400, 1)], [], True, False),
    ([(256, 256, .3), (400, 400, 1)], [], False, False),
    ([(256, 256, .7), (400, 400, 1)], [(256, 256)], False, True),
    ([(256, 256, .7), (400, 400, 1)], [(400, 400)], True, False),
    ([(256, 256, 1), (280, 256, .7)], [(256, 256)], False, True),
    ([(280, 256, .7)], [(256, 256)], False, True),  # AOI coordinate has priority
    ([(281, 256, .7)], [], True, False),
    ([(282, 256, .7)], [], False, False),
])
def test_roi_decision_and_reporting(version, points, bombs, ng, bomb, monkeypatch):
    inf, result, tile = case(points, bombs=bombs)
    original = result.anomaly_tiles[0][2].copy()
    if version == 'v2':
        inf._apply_aoi_judgment_roi([result], RESOLUTION)
    apply_bomb_postprocess(inf, result, None, version, monkeypatch)
    assert tile.is_bomb is bomb
    stored = results_to_db_data([result], {})[0]
    assert stored['is_ng'] == int(ng)
    records = _iter_qjpg_defect_records([result], RESOLUTION, inf.config)
    assert any(r.startswith('PCDK2') for r in records) is ng
    assert np.array_equal(tile.aoi_roi_full_map, original)
    assert not np.any(result.anomaly_tiles[0][2][tile.aoi_roi_mask == 0])
    ctx = json.loads(stored['tiles'][0]['decision_context'])
    assert ctx['aoi_judgment_roi']['radius_product_px'] == 25
    assert decision_evidence(stored['tiles'][0])['title'].startswith('AOI 範圍內：')


def test_switch_off_keeps_full_tile_and_grid_unaffected():
    inf, result, tile = case([(400, 400, 1)], enabled=False)
    inf._apply_aoi_judgment_roi([result], RESOLUTION)
    assert tile.aoi_roi_mask is None
    assert result.anomaly_tiles[0][1] == 1
    inf.config.aoi_judgment_roi_enabled = True
    tile.is_aoi_coord_tile = False
    inf._apply_aoi_judgment_roi([result], RESOLUTION)
    assert tile.aoi_roi_mask is None


@pytest.mark.parametrize('version', ['v1', 'v2'])
@pytest.mark.parametrize('source', ['client', 'config'])
def test_five_logged_aoi_bombs_ignore_spreading_heat(version, source, monkeypatch):
    from capi_web import _target_tiles_for_within_spec
    aoi_points = [(358, 136), (1569, 326), (289, 552), (1014, 567), (1681, 724)]
    bomb_points = [(353, 134), (1565, 323), (286, 550), (1011, 565), (1677, 720)]
    inf, result, _ = case([])
    result.image_path = Path('WGF50500_134729.tif')
    result.raw_bounds = (844, 898, 5920, 4062)
    result.tiles, result.anomaly_tiles = [], []
    for index, aoi in enumerate(aoi_points):
        anchor = inf._map_aoi_coords(*aoi, result.raw_bounds, RESOLUTION)
        tile = make_tile(index, aoi, anchor)
        tile.score_threshold = .35
        # Synthetic diffuse heat spanning the ROI exceeds the bomb tolerance.
        amap = np.full((512, 512), .6, dtype=np.float32)
        result.tiles.append(tile)
        result.anomaly_tiles.append((tile, .6, amap))
    inf.config.bomb_defects = [BombDefect('WGF50500', 'B01', 'point', bomb_points)]
    bomb_info = dict(image_prefix='WGF50500', defect_type='point', coordinates=bomb_points)
    if version == 'v2':
        inf._apply_aoi_judgment_roi([result], RESOLUTION)
    apply_bomb_postprocess(inf, result, bomb_info if source == 'client' else None,
                           version, monkeypatch)
    assert all(t.is_bomb for t in result.tiles)
    assert all(t.bomb_remaining_points == [] for t in result.tiles)
    stored = results_to_db_data([result], {})[0]
    assert stored['is_ng'] == 0
    assert _target_tiles_for_within_spec(stored) == []
    assert not any(r.startswith('PCDK2') for r in
                   _iter_qjpg_defect_records([result], RESOLUTION, inf.config))
    for row in stored['tiles']:
        ctx = json.loads(row['decision_context'])
        assert ctx['aoi_bomb_priority_basis'] == 'aoi_coordinate'
        assert any('原始 AOI 產品座標' in note for note in decision_evidence(row)['notes'])
    assert all(r['bomb_remaining_px'] == 0 for t in result.tiles
               for r in t.bomb_region_diagnostics['regions'])


@pytest.mark.parametrize('bomb_x,prefix,expected', [
    (236, 'STANDARD', True),  # Inclusive 20 px tolerance.
    (235, 'STANDARD', False),
    (256, 'WGF50500', False),  # Different lighting cannot match.
])
def test_aoi_bomb_priority_respects_tolerance_and_lighting(bomb_x, prefix, expected, monkeypatch):
    inf, result, tile = case([(280, 256, .8)])
    inf.config.bomb_defects = [BombDefect(prefix, 'B01', 'point', [(bomb_x, 256)])]
    inf._apply_aoi_judgment_roi([result], RESOLUTION)
    apply_bomb_postprocess(inf, result, None, 'v2', monkeypatch)
    assert tile.is_bomb is expected
    assert results_to_db_data([result], {})[0]['is_ng'] == int(not expected)


@pytest.mark.parametrize('map_size', [128, 512])
def test_shifted_corner_uses_original_aoi_and_product_scale(map_size):
    inf, result, _ = make_overlapping_point_bomb_case('config', map_size)
    inf.config.aoi_judgment_roi_enabled = True
    inf._apply_aoi_judgment_roi([result], RESOLUTION)
    tile = result.tiles[1]
    mask = tile.aoi_roi_mask
    assert mask[int(33 * map_size / 512), 0]
    assert not mask[map_size // 2, map_size // 2]
    assert tile.decision_context['aoi_judgment_roi']['product_bounds'] == [0, 0, 36, 36]


def test_invalid_geometry_does_not_silently_pass():
    inf, result, tile = case([(400, 400, 1)])
    result.raw_bounds = None
    with pytest.raises(ValueError, match='geometry'):
        inf._apply_aoi_judgment_roi([result], RESOLUTION)


@pytest.mark.parametrize('radius,expected_count,converted', [(10, 1, True), (25, 2, False)])
def test_roi_snapshot_reaches_live_within_spec_conversion(tmp_path, radius, expected_count, converted):
    import cv2
    from capi_server import CAPIServer
    from test_within_spec_suggestion import _rules

    inf, result, tile = case([(256, 256, .8), (276, 256, .7), (400, 400, 1)])
    image = np.full((512, 512, 3), 128, np.uint8)
    for point in [(256, 256), (276, 256), (400, 400)]:
        cv2.circle(image, point, 3, (60, 60, 60), -1)
    result.image_path = tmp_path / 'W0F00000.png'
    cv2.imwrite(str(result.image_path), image)
    inf.config.aoi_judgment_radius_px = radius
    inf._apply_aoi_judgment_roi([result], RESOLUTION)
    # Later settings must not silently replace this tile's captured scope.
    inf.config.aoi_judgment_radius_px = 100
    server = CAPIServer.__new__(CAPIServer)
    server._load_within_spec_rules_for_inference = lambda _: _rules(threshold_mm=1)
    outcome = server._evaluate_within_spec_for_inference(
        {'glass_id': 'ROI_TEST', 'model_id': '', 'machine_no': 'MOD1'}, [result], inf)
    assert outcome['converted'] is converted, outcome
    total = outcome['detail']['panel_totals'][0]
    assert total['total_count'] == expected_count
    scope_step = next(step for step in outcome['detail']['steps'] if 'aoi_judgment_roi' in step)
    assert scope_step['aoi_judgment_roi']['radius_product_px'] == radius


def test_aoi_bomb_coordinate_has_priority_over_two_stage_feature(monkeypatch):
    inf, result, tile = case([(256, 256, 1)], bombs=[(256, 256)])
    inf._apply_aoi_judgment_roi([result], RESOLUTION)
    tile.dust_two_stage_features = [dict(abs_pos=(280, 256), is_dust=False, area=1)]
    apply_bomb_postprocess(inf, result, None, 'v2', monkeypatch)
    assert tile.is_bomb


def test_bright_spot_outside_roi_is_not_ng():
    inf, result, tile = case([(400, 400, 1)])
    tile.is_bright_spot_detection = True
    tile.bright_spot_min_area = 1
    inf._apply_aoi_judgment_roi([result], RESOLUTION)
    assert result.anomaly_tiles[0][1] == 0
    assert tile.is_aoi_coord_below_threshold


def test_bright_spot_aoi_bomb_coordinate_has_priority(monkeypatch):
    inf, result, tile = case([(280, 256, 1)], bombs=[(256, 256)])
    tile.is_bright_spot_detection = True
    tile.bright_spot_min_area = 1
    inf._apply_aoi_judgment_roi([result], RESOLUTION)
    apply_bomb_postprocess(inf, result, None, 'v2', monkeypatch)
    assert tile.is_bomb
    assert results_to_db_data([result], {})[0]['is_ng'] == 0


def test_old_setting_migrates_once_and_radius_persists(tmp_path):
    db = CAPIDatabase(str(tmp_path / 'settings.db'))
    db.update_config_param('aoi_bomb_priority_enabled', True)
    db.init_config_from_yaml(CAPIConfig())
    cfg = CAPIConfig()
    cfg.apply_db_overrides(db.get_all_config_params())
    assert cfg.aoi_judgment_roi_enabled
    assert cfg.aoi_judgment_radius_px == 25
    assert not any(p['param_name'] == 'aoi_bomb_priority_enabled' for p in db.get_all_config_params())
    db.update_config_param('aoi_judgment_roi_enabled', False)
    db.update_config_param('aoi_judgment_radius_px', 30)
    db.init_config_from_yaml(CAPIConfig())
    cfg.apply_db_overrides(db.get_all_config_params())
    assert not cfg.aoi_judgment_roi_enabled
    assert cfg.aoi_judgment_radius_px == 30
    assert CAPIConfig.from_dict({'aoi_bomb_priority_enabled': True}).aoi_judgment_roi_enabled


@pytest.mark.parametrize('radius', [25, 30])
def test_roi_heatmap_banner_box_preserves_full_tile_display(tmp_path, monkeypatch, radius):
    inf, result, tile = case([(256, 256, .7), (400, 400, 1)], bombs=[(256, 256)])
    inf.config.apply_db_overrides([{'param_name': 'aoi_judgment_radius_px', 'decoded_value': radius}])
    tile.image[240, 256 - radius] = 20  # A dark defect directly under the ROI outline.
    inf._apply_aoi_judgment_roi([result], RESOLUTION)
    apply_bomb_postprocess(inf, result, None, 'v2', monkeypatch)
    captured = []
    original = cv2.putText
    def put_text(img, text, *args, **kwargs):
        captured.append(text)
        return original(img, text, *args, **kwargs)
    monkeypatch.setattr(cv2, 'putText', put_text)
    _, score, amap = result.anomaly_tiles[0]
    output = HeatmapManager(tmp_path, save_format='png').save_tile_heatmap(
        tmp_path, 'ROI_EXAMPLE', 0, tile.image, amap, score, tile_info=tile)
    image = cv2.imread(output)
    assert image is not None
    assert any(f'AOI ONLY +/-{radius} product px' in s and 'ROI RESULT: BOMB -> OK' in s for s in captured)
    assert any('Full tile score: 1.0000 (reference only)' in s for s in captured)
    assert not any('DIMMED' in s for s in captured)
    assert sum('CYAN BOX = EVALUATED' in s for s in captured) == 1
    assert tuple(image[88 + 60 + 400, 400]) == (100, 100, 100)
    border_defect = image[88 + 60 + 240, 256 - radius]
    assert tuple(border_defect) == (20, 20, 20)
    assert tuple(image[88 + 60 + 240, 256 - radius - 1]) == (255, 255, 0)
    # The AOI center remains unobstructed on the original-image panel.
    assert tuple(image[88 + 60 + 256, 256]) == (100, 100, 100)
    # Compare all full-tile panels against the unannotated rendering, including
    # the heatmap's colors outside the judgment ROI.
    tile.decision_context['aoi_judgment_roi']['enabled'] = False
    plain_path = HeatmapManager(tmp_path, save_format='png').save_tile_heatmap(
        tmp_path, 'ROI_PLAIN', 0, tile.image, amap, score, tile_info=tile)
    plain = cv2.imread(plain_path)
    for panel_index in range(5):
        x = panel_index * 512 + 400
        np.testing.assert_array_equal(image[88 + 60 + 400, x], plain[60 + 400, x])
    heatmap_pixel = image[88 + 60 + 400, 2 * 512 + 400]
    assert len(set(heatmap_pixel)) > 1


@pytest.mark.parametrize('has_inner_ng', [False, True])
def test_dust_inside_roi_cannot_hide_another_valid_inner_region(has_inner_ng, monkeypatch):
    inf, result, tile = case([(256, 256, 1), (280, 256, .7), (400, 400, 1)])
    inf.config.dust_two_stage_enabled = False
    inf._apply_aoi_judgment_roi([result], RESOLUTION)
    dust = np.full((512, 512), 255, dtype=np.uint8)
    if has_inner_ng:
        dust[256, 280] = 0
    monkeypatch.setattr(inf, '_check_dust_or_scratch_feature_with_context',
                        lambda *a, **k: (True, dust, .9, 'OMIT'))
    inf._apply_omit_dust_postprocess([result], np.full((512, 512), 100, dtype=np.uint8),
                                   False, '', cpu_workers=1, product_resolution=RESOLUTION)
    assert tile.is_suspected_dust_or_scratch is (not has_inner_ng)
    assert results_to_db_data([result], {})[0]['is_ng'] == int(has_inner_ng)
    assert not np.any(tile.dust_heatmap_binary[tile.aoi_roi_mask == 0])


def test_two_stage_dilation_does_not_restore_outside_features():
    inf, _, _ = case([])
    inf.config.dust_two_stage_min_area = 3
    image = np.full((64, 64), 100, dtype=np.uint8)
    image[30:35, 42:47] = 20
    amap = np.zeros((64, 64), dtype=np.float32)
    amap[30:35, 35:40] = 1
    allowed = np.zeros_like(amap, dtype=np.uint8)
    allowed[20:41, 20:41] = 1
    has_real, peak, features, _ = inf.check_dust_two_stage(
        image, amap, np.zeros_like(image), 1., score_threshold=.5, evaluation_mask=allowed)
    assert all(allowed[f['abs_pos'][1], f['abs_pos'][0]] for f in features)
    assert not any(f['feature_bbox'][0] >= 42 for f in features)


@pytest.mark.parametrize('has_original', [False, True])
@pytest.mark.parametrize('scratch_score,filtered', [(.1, False), (.95, True)])
def test_scratch_classifier_sees_full_tile_with_roi_enabled(has_original, scratch_score, filtered):
    from scratch_filter import ScratchFilter
    from types import SimpleNamespace
    inf, result, tile = case([(256, 256, .8), (400, 400, 1)])
    tile.image[350:, 350:] = 255
    if has_original:
        tile.original_image = np.full((512, 512, 3), 80, dtype=np.uint8)
        tile.original_image[350:, 350:] = 220
    expected = (tile.original_image if has_original else tile.image).copy()
    inf._apply_aoi_judgment_roi([result], RESOLUTION)
    roi_mask = tile.aoi_roi_mask.copy()
    limited_map = result.anomaly_tiles[0][2].copy()
    seen = []
    def predict(image):
        seen.append(image)
        return scratch_score
    ScratchFilter(SimpleNamespace(conformal_threshold=.7, predict=predict)).apply_to_image_result(result)
    assert len(seen) == 1
    assert seen[0].shape[:2] == (512, 512)
    np.testing.assert_array_equal(seen[0], expected)
    np.testing.assert_array_equal(tile.aoi_roi_mask, roi_mask)
    np.testing.assert_array_equal(result.anomaly_tiles[0][2], limited_map)
    assert tile.scratch_filtered is filtered
    assert tile.scratch_score == scratch_score
    assert results_to_db_data([result], {})[0]['is_ng'] == int(not filtered)


def test_edge_rescue_only_uses_roi_evidence():
    from test_edge_light_leak import _synthetic_panel, _light_leak_config
    from capi_edge_cv import inspect_aoi_edge_light_leak
    image, dust, polygon, aoi, _ = _synthetic_panel('top')
    allowed = np.zeros(image.shape, dtype=np.uint8)
    allowed[:, :30] = 1  # Bright band lies outside this evaluation area.
    outcome = inspect_aoi_edge_light_leak(
        tile_image=image, tile_origin=(0, 0), panel_polygon=polygon,
        raw_bounds=(0, 0, 160, 120), aoi_product_xy=aoi, product_resolution=(160, 120),
        dust_mask=dust, config=_light_leak_config(), evaluation_mask=allowed)
    assert not outcome['detected']
