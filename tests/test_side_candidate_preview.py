"""AOI-only annotations keep full inspection evidence available to DEBUG."""
import copy

import cv2
import numpy as np
import pytest

from capi_config import normalize_side_white_params
from capi_side_white import _candidate_preview_scope, inspect_side_white_image, prepare_aoi_context


def mapped_payload(points):
    return {'status': 'CANDIDATES', 'front_size': [1200, 800], 'side_size': [1200, 800],
            'mapping': {'status': 'estimated', 'matrix': np.eye(3).tolist(),
                        'front_polygon': [[100, 100], [1100, 100], [1100, 600], [100, 600]]},
            'evidence_context': {'product_resolution': [1000, 500], 'aoi_csjbl_points': points},
            'candidates': [
                {'id': 1, 'side_contour': [[490, 290], [510, 290], [510, 310], [490, 310]]},
                {'id': 2, 'side_contour': [[890, 490], [910, 490], [910, 510], [890, 510]]},
            ]}


def point(x=400, y=200, **extra):
    return {'defect_code': 'CSJBL', 'image_prefix': 'W0F00000', 'product_x': x, 'product_y': y, **extra}


def test_preview_only_counts_aoi_hits_and_keeps_neighbours_as_references():
    payload = mapped_payload([point(460, 200), point(800, 400, defect_code='OTHER')])
    before = copy.deepcopy(payload)
    scope = _candidate_preview_scope(payload)
    assert scope['candidate_ids'] == []  # Nearby is not an AOI point hit.
    assert scope['total'] == 0 and scope['reference_total'] == 2 and scope['tolerance_product_px'] == 50
    assert payload == before
    payload['evidence_context']['aoi_csjbl_points'][0]['product_x'] = 461
    assert _candidate_preview_scope(payload)['candidate_ids'] == []
    payload['evidence_context']['aoi_csjbl_points'] = [point(), point(800, 400)]
    assert _candidate_preview_scope(payload)['candidate_ids'] == [1, 2]


@pytest.mark.parametrize('disposition', ['mark_suppressed', 'dust_suppressed', 'excluded'])
def test_preview_counts_only_retained_yellow_candidates_and_keeps_original_ids(disposition):
    payload = mapped_payload([point()])
    payload['candidates'][0]['disposition'] = disposition
    before = copy.deepcopy(payload)
    scope = _candidate_preview_scope(payload)
    assert scope['candidate_ids'] == [] and scope['total'] == 0 and scope['reference_total'] == 1
    assert payload == before
    payload['evidence_context']['aoi_csjbl_points'] = [point(800, 400)]
    assert _candidate_preview_scope(payload)['candidate_ids'] == [2]


@pytest.mark.parametrize('radius,x,expected', [(0, 400, [1]), (0, 411, []),
    (20, 430, []), (20, 431, []), (100, 500, []), (500, 400, [1])])
def test_preview_radius_does_not_expand_aoi_point_hits(radius, x, expected):
    payload = mapped_payload([point(x, 200)])
    payload['parameters'] = {'csjbl_preview_radius_product_px': radius}
    before = copy.deepcopy(payload)
    scope = _candidate_preview_scope(payload)
    assert scope['candidate_ids'] == expected
    assert scope['tolerance_product_px'] == radius
    assert payload == before


@pytest.mark.parametrize('value', [-1, 501, 1.5, True, '50', None, float('nan'), float('inf')])
def test_preview_radius_rejects_invalid_values(value):
    with pytest.raises(ValueError, match='CSJBL 顯示範圍'):
        normalize_side_white_params({'csjbl_preview_radius_product_px': value})


@pytest.mark.parametrize('rotated', [False, True])
def test_preview_honors_image_coordinate_rotation(rotated):
    xy = [699, 499] if rotated else [500, 300]
    payload = mapped_payload([point(source_x=xy[0], source_y=xy[1], coordinate_space='image')])
    payload['rotation_applied'] = rotated
    assert _candidate_preview_scope(payload)['candidate_ids'] == [1]
    payload['evidence_context']['aoi_csjbl_points'][0]['image_prefix'] = 'G0F00000'
    scope = _candidate_preview_scope(payload)
    assert scope['candidate_ids'] == [] and '無法映射' in scope['reason']


def test_missing_aoi_or_mapping_never_falls_back_to_all_candidates():
    payload = mapped_payload([])
    payload['bombs'] = [{'candidate_ids': [1, 2], 'defect_code': 'CSJBL'}]
    scope = _candidate_preview_scope(payload)
    assert scope['candidate_ids'] == [] and '沒有 AOI CSJBL' in scope['reason']
    payload['evidence_context']['aoi_csjbl_points'] = [point()]
    payload['mapping'] = {'status': 'unavailable'}
    scope = _candidate_preview_scope(payload)
    assert scope['candidate_ids'] == [] and '無法映射' in scope['reason']


def test_automatic_context_keeps_an_independent_csjbl_snapshot():
    coords = {'W0F00000': [point(), point(defect_code='OTHER')]}
    context = prepare_aoi_context({}, coords)
    coords['W0F00000'][0]['product_x'] = 999
    assert len(context['aoi_csjbl_points']) == 1
    assert context['aoi_csjbl_points'][0]['product_x'] == 400
    legacy = [{**point(), 'type': 'aoi_passthrough', 'image': 'W0F00000'}]
    assert prepare_aoi_context({}, None, legacy)['aoi_csjbl_points'][0]['coordinate_space'] == 'product'


@pytest.mark.parametrize('rotated', [False, True])
def test_real_previews_only_draw_nearby_but_debug_can_show_all(tmp_path, rotated):
    front = np.zeros((500, 700), np.uint8)
    cv2.rectangle(front, (50, 50), (650, 450), 90, -1)
    front_path, side_path = tmp_path / 'W0F00000.tif', tmp_path / 'SW0F00000.tif'
    cv2.imwrite(str(front_path), front)
    side = front.copy()
    cv2.circle(side, (350, 250), 8, 125, -1)
    cv2.circle(side, (150, 150), 8, 125, -1)
    cv2.imwrite(str(side_path), side)
    context = {'product_resolution': [600, 400], 'aoi_csjbl_points': [point(300, 200)]}
    payload = inspect_side_white_image(side_path, front_path, tmp_path / 'debug',
        context=context, rotate_180=rotated, include_all_candidates=True)
    assert payload['status'] == 'CANDIDATES', payload.get('reason')
    assert len(payload['candidates']) == 2
    selected = payload['candidate_display']['candidate_ids']
    assert len(selected) == 1
    far = next(c for c in payload['candidates'] if c['id'] not in selected)
    for kind in ['side', 'front', 'residual']:
        near_view = cv2.imread(payload['artifacts'][kind])
        all_view = cv2.imread(payload['artifacts'][kind + '_all'])
        assert near_view.shape == all_view.shape
        xy = far['front_xy'] if kind == 'front' else far['side_xy']
        x, y = np.rint(np.array(xy) - payload['preview_bounds'][kind][:2]).astype(int)
        def yellow_count(image):
            roi = image[y-25:y+26, x-25:x+26]
            return np.count_nonzero((roi[:, :, 0] < 100) & (roi[:, :, 1] > 130) & (roi[:, :, 2] > 170))
        assert yellow_count(near_view) == 0
        assert yellow_count(all_view) == 0
        # DEBUG can still inspect the non-AOI candidate, using reference gray.
        def colored_count(image):
            return np.count_nonzero(np.ptp(image[y-25:y+26, x-25:x+26], axis=2) > 10)
        assert colored_count(near_view) == 0 and colored_count(all_view) > 20
    formal = inspect_side_white_image(side_path, front_path, tmp_path / 'formal',
        context=context, rotate_180=rotated)
    assert formal['candidates'] == payload['candidates']
    assert formal['summary'] == payload['summary']
    assert formal['candidate_display'] == payload['candidate_display']
    assert not any(key.endswith('_all') for key in formal['artifacts'])
