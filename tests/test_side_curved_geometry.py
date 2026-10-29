"""Curved side borders must be distinguished from unreliable edge samples."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import capi_side_white as sw


def bowed_panel(bow=18, disturbance=None):
    image = np.zeros((1600, 2400), np.uint8)
    t = np.linspace(0, 1, 2001)
    y = 1200 + bow * 4 * t * (1 - t)
    if disturbance == 'ripple':
        y += 16 * np.sin(t * 6 * np.pi)
    elif disturbance == 'step':
        y += 20 * (t > .5)
    elif disturbance == 'local':
        y += 12 * ((t > .4) & (t < .48))
    bottom = np.column_stack([2200 - t * 2000, y])
    boundary = np.vstack([[200, 300], [2200, 300], bottom])
    cv2.fillPoly(image, [np.rint(boundary).astype(np.int32)], 100)
    return image


@pytest.mark.parametrize('bow', [-18, 18])
@pytest.mark.parametrize('rotated', [False, True])
def test_smooth_bow_recovers_corners_and_actual_boundary(bow, rotated):
    image = bowed_panel(bow)
    expected = np.array([[200, 300], [2200, 300], [2200, 1200], [200, 1200]], np.float32)
    midpoint = np.array([1200, 1200 + bow])
    if rotated:
        image = cv2.rotate(image, cv2.ROTATE_180)
        expected = (np.array([2399, 1599]) - expected)[[2, 3, 0, 1]]
        midpoint = np.array([2399, 1599]) - midpoint
    evidence = {}
    quad = sw.detect_panel_quad(image, localization=evidence, rotate_180=rotated)
    assert np.linalg.norm(quad - expected, axis=1).max() < 2
    assert evidence['method'] == 'bounded_curved_edges'
    curved = [edge for edge in evidence['edges'] if edge['model'] == 'quadratic']
    assert len(curved) == 1 and curved[0]['edge'] == ('top' if rotated else 'bottom')
    assert curved[0]['residual_p95_px'] > curved[0]['limit_px']
    assert curved[0]['curve_residual_p95_px'] < curved[0]['curve_limit_px']
    assert np.linalg.norm(np.array(evidence['boundary']) - midpoint, axis=1).min() < 2
    # A rigid perspective mapping still recovers interior points from known
    # camera corners; bow validation must not shift the endpoints into the panel.
    transform = cv2.getPerspectiveTransform(quad.astype(np.float32), expected.astype(np.float32))
    points = np.array([[[600, 600]], [[1800, 1000]]], np.float32)
    assert np.linalg.norm(cv2.perspectiveTransform(points, transform) - points, axis=2).max() < 2


@pytest.mark.parametrize('bow,disturbance', [(18, 'ripple'), (18, 'step'), (18, 'local'), (45, None)])
def test_unreliable_or_excessive_bows_are_still_rejected(bow, disturbance):
    with pytest.raises(sw.PanelLocalizationError) as caught:
        sw.detect_panel_quad(bowed_panel(bow, disturbance), view='side')
    info = caught.value.diagnostics
    assert info['stage'] == 'edge_fit' and info['edge'] == 'bottom'
    assert not info['curve_accepted']


def test_bowed_boundary_keeps_edge_defect_and_excludes_background():
    image = bowed_panel(18)
    evidence = {}
    quad = sw.detect_panel_quad(image, localization=evidence)
    # This defect lies beyond the corner chord but inside the real curved panel.
    cv2.circle(image, (1200, 1207), 3, 210, -1)
    params = sw.normalize_side_white_params({'edge_margin_px': 3, 'min_area_px': 3})
    candidates, *_ = sw._candidates(image, quad, params, panel_boundary=evidence['boundary'])
    assert any(np.linalg.norm(np.array(c['side_xy']) - [1200, 1207]) < 6 for c in candidates)
    boundary = np.array(evidence['boundary'], np.float32)
    assert all(cv2.pointPolygonTest(boundary, tuple(c['side_xy']), False) >= 0 for c in candidates)


@pytest.mark.parametrize('rotated', [False, True])
def test_real_side_regression_when_local_capture_is_available(tmp_path, rotated):
    # Production captures are intentionally not checked into the repository.
    source = Path(__file__).resolve().parents[1] / 'SW0F00000_145218.tif'
    if not source.is_file():
        pytest.skip('optional local production capture')
    payload = sw.inspect_side_white_image(source, None, tmp_path / 'out', rotate_180=rotated)
    assert payload['status'] == 'CANDIDATES', payload.get('reason')
    bottom = payload['localization']['edges'][0 if rotated else 2]
    assert bottom['edge'] == ('top' if rotated else 'bottom') and bottom['model'] == 'quadratic'
    assert bottom['residual_p95_px'] > bottom['limit_px']
    if not rotated:
        assert bottom['residual_p95_px'] == pytest.approx(7.33, abs=.1)
    else:
        original = sw.detect_panel_quad(cv2.imread(str(source), cv2.IMREAD_GRAYSCALE))
        expected = (np.array([3295, 2471]) - original)[[2, 3, 0, 1]]
        assert np.linalg.norm(np.array(payload['side_polygon']) - expected, axis=1).max() < 2
    assert bottom['curve_residual_p95_px'] < 2
    assert payload['mapping']['status'] == 'unavailable'  # No fabricated front evidence.
    assert Path(payload['artifacts']['side']).is_file()
    json.dumps(payload)
