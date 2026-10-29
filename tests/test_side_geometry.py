"""Same-inference front geometry reuse, provenance and failure diagnostics."""
import copy
import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

import capi_side_white as sw
from capi_side_report import compare_side_record


@pytest.fixture
def pair(tmp_path):
    front = np.zeros((500, 700), np.uint8)
    cv2.rectangle(front, (50, 50), (650, 450), 90, -1)
    quad = sw.detect_panel_quad(front)
    front_path = tmp_path / 'W0F00000_140512.tif'
    side_path = tmp_path / 'SW0F00000_140513.tif'
    assert cv2.imwrite(str(front_path), front)
    side = front.copy()
    cv2.circle(side, (350, 250), 9, 125, -1)
    assert cv2.imwrite(str(side_path), side)
    result = SimpleNamespace(image_path=front_path, image_size=(700, 500), panel_polygon=quad)
    return SimpleNamespace(side=side_path, front=front_path, result=result, root=tmp_path)


def context_for(pair, rotated=False):
    return {'front_geometries': sw.snapshot_front_geometries([pair.result], rotate_180=rotated),
            'product_resolution': [600, 400],
            'aoi_csjbl_points': [{'defect_code': 'CSJBL', 'image_prefix': 'W0F00000',
                                  'product_x': 300, 'product_y': 200, 'coordinate_space': 'product'}]}


@pytest.mark.parametrize('rotated', [False, True])
def test_reuses_formal_quad_when_independent_front_localizer_would_fail(pair, monkeypatch, rotated):
    if rotated:
        pair.result.panel_polygon = (np.array([699, 499]) - pair.result.panel_polygon)[[2, 3, 0, 1]]
    context = context_for(pair, rotated)
    expected = pair.result.panel_polygon.copy()
    pair.result.panel_polygon[:] = 0  # queued evidence must be independent of live results
    original = sw.detect_panel_quad

    def localize(gray, **kwargs):
        if kwargs.get('view') == 'front':
            raise ValueError('independent front localization failed')
        return original(gray, **kwargs)

    monkeypatch.setattr(sw, 'detect_panel_quad', localize)
    payload = sw.inspect_side_white_image(pair.side, pair.front, pair.root / 'out',
                                          context=context, rotate_180=rotated)
    assert payload['mapping']['status'] == 'estimated'
    assert payload['mapping']['front_geometry_source'] == 'formal_inference'
    assert np.allclose(payload['mapping']['front_polygon'], expected)
    assert Path(payload['artifacts']['front']).is_file()
    comparison = compare_side_record({'payload': payload, 'status': payload['status']})
    assert comparison['mapped'] == 1
    assert np.allclose(comparison['points'][0]['side_xy'], [350, 250], atol=2)
    json.dumps(payload)


@pytest.mark.parametrize('field,value,reason', [
    ('source_image', 'W0F00000_140511.tif', '來源圖不一致'),
    ('source_path', '/another-record/W0F00000_140512.tif', '來源圖不一致'),
    ('image_size', [1400, 1000], '尺寸'),
    ('rotation_applied', True, '旋轉方向'),
    ('rotation_applied', None, '旋轉方向'),
    ('coordinate_space', 'raw_pixels', '座標系'),
    ('file_signature', {}, '來源檔已變更'),
    ('polygon', [[50, 50], [650, 450], [650, 50], [50, 450]], '四角'),
])
def test_mismatched_evidence_never_falls_back_to_another_localizer(pair, field, value, reason, caplog):
    context = context_for(pair)
    context['front_geometries'][0][field] = value
    with caplog.at_level(logging.WARNING, logger='capi.side_white'):
        payload = sw.inspect_side_white_image(pair.side, pair.front, pair.root / 'out', context=context)
    assert payload['mapping']['status'] == 'unavailable'
    assert reason in payload['mapping']['reason']
    assert 'front' not in payload['artifacts']
    assert compare_side_record({'payload': payload, 'status': payload['status']})['mapped'] == 0
    assert '[SIDE_WHITE_MAPPING]' in caplog.text and pair.front.name in caplog.text
    assert payload['candidates'] and all(c['front_xy'] is None for c in payload['candidates'])


def test_changed_file_and_ambiguous_acquisition_are_rejected(pair):
    context = context_for(pair)
    with pair.front.open('ab') as f:
        f.write(b'changed after formal inference')
    payload = sw.inspect_side_white_image(pair.side, pair.front, pair.root / 'changed', context=context)
    assert '來源檔已變更' in payload['mapping']['reason']
    # Another front at a different exposure makes the sequential pair ambiguous.
    other = pair.front.with_name('W0F00000_140514.tif')
    other.write_bytes(pair.front.read_bytes())
    payload = sw.inspect_side_white_image(pair.side, pair.front, pair.root / 'ambiguous', context=context)
    assert '同次配對' in payload['mapping']['reason']


@pytest.mark.parametrize('view,failed_call', [('side', 1), ('front', 5)])
def test_localization_log_identifies_image_edge_residual_and_limit(pair, monkeypatch, caplog, view, failed_call):
    original = cv2.fitLine
    calls = 0

    def bad_front_edge(*args, **kwargs):
        nonlocal calls
        calls += 1
        line = original(*args, **kwargs)
        if calls == failed_call:
            line[3] += 50
        return line

    monkeypatch.setattr(cv2, 'fitLine', bad_front_edge)
    with caplog.at_level(logging.WARNING, logger='capi.side_white'):
        payload = sw.inspect_side_white_image(pair.side, pair.front, pair.root / 'out')
    diagnostics = (payload if view == 'side' else payload['mapping'])['diagnostics']
    assert diagnostics['view'] == view and diagnostics['edge'] == 'top'
    assert diagnostics['source_image'] == (pair.side if view == 'side' else pair.front).name
    assert diagnostics['residual_p95_px'] > diagnostics['limit_px']
    assert diagnostics['samples'] == 400
    assert '[SIDE_WHITE_LOCALIZATION]' in caplog.text
    assert 'residual_p95_px' in caplog.text and 'limit_px' in caplog.text


def test_queue_copies_geometry_before_background_work(pair):
    from capi_server import CAPIServer
    server = CAPIServer.__new__(CAPIServer)
    server._async_executor_lock = threading.Lock()
    server._async_executor_shutdown = False
    server._async_executor = MagicMock()
    server._queue_save_results_async(None, {}, [pair.result], side_white_context={},
                                     side_white_rotate_180=True)
    queued = server._async_executor.submit.call_args.kwargs['side_white_context']
    expected = copy.deepcopy(queued)
    pair.result.panel_polygon[:] = 0
    assert queued == expected
    assert queued['front_geometries'][0]['rotation_applied'] is True


def test_only_formal_four_corners_are_snapshotted(pair):
    pair.result.panel_polygon = None
    pair.result.raw_bounds = (50, 50, 650, 450)
    assert sw.snapshot_front_geometries([pair.result]) == []
