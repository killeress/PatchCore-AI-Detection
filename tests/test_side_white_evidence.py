"""Coordinate, masking and evidence checks without loading any ML models."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from capi_config import normalize_side_white_params
from capi_side_white import (
    _bomb_geometry, _candidate_evidence, _match_bombs, _zone_mask,
    inspect_side_white_image,
)


def candidate():
    return {"id": 1, "kind": "line", "side_xy": [39.5, 29.5],
            "side_bbox": [20, 20, 40, 20], "side_contour": [[20,20],[59,20],[59,39],[20,39]],
            "front_contour": [[20,20],[59,20],[59,39],[20,39]],
            "area_px": 800, "contrast_gray": 10, "_mask": np.ones((20,40), np.uint8)}


def evaluate(item, *, zones=None, mode="observe", detector=None, omit=None, params=None):
    image = np.full((100,100), 90, np.uint8)
    residual = np.full(image.shape, 10, np.float32)
    options = normalize_side_white_params({"dust_mode":mode,"mapping_margin_px":0, **(params or {})})
    return _candidate_evidence(item, image, image, omit, detector,
                               {"reason":"test"}, np.eye(3),
                               np.zeros_like(image) if zones is None else zones, residual, options)


def test_partial_exclusion_keeps_outside_evidence():
    item = candidate()
    zones = np.zeros((100,100), np.uint8)
    zones[:, :40] = 255
    composite = evaluate(item, zones=zones)
    assert item["excluded_area_px"] == 400
    assert item["remaining_area_px"] == 400
    assert item["disposition"] == "retained"
    assert item["remaining_contrast_gray"] == 10
    assert composite.shape == (350,1600,3)
    assert min(p[0] for c in item["remaining_contours"] for p in c) == 40
    json.dumps(item)


@pytest.mark.parametrize("mode,disposition,area", [("observe","retained",800),("suppress","dust_suppressed",0),("off","retained",800)])
def test_dust_modes_use_support_pixels(mode, disposition, area):
    item = candidate()
    omit = np.ones((100,100), np.uint8)
    detector = lambda crop: (True, np.full(crop.shape,255,np.uint8), 1, "surface evidence")
    evaluate(item,mode=mode,omit=omit,detector=detector)
    assert item["disposition"] == disposition
    assert item["remaining_area_px"] == area


def test_dust_partial_overlap_preserves_adjacent_anomaly():
    item = candidate()
    omit = np.zeros((100,100), np.uint8)
    omit[:, :40] = 255
    evaluate(item, mode="suppress", omit=omit,
             detector=lambda crop: (True,crop,1,"partial"), params={"dust_overlap_ratio":.5})
    assert item["dust"]["overlap_ratio"] == .5
    assert item["remaining_area_px"] == 400
    assert item["disposition"] == "retained"


@pytest.mark.parametrize("with_front", [False, True])
def test_side_dust_coordinates_are_independent_of_front_mapping(with_front):
    item = candidate()
    side = np.full((100, 100), 90, np.uint8)
    omit = np.zeros_like(side)
    omit[20:40, 20:60] = 255
    transform = np.diag([2., 2., 1.]) if with_front else None
    front = np.full((200, 200), 90, np.uint8) if with_front else None
    item['front_contour'] = [[40, 40], [118, 40], [118, 78], [40, 78]] if with_front else None
    composite = _candidate_evidence(
        item, side, front, omit, lambda crop: (True, crop, 1, 'side dust'), {},
        transform, np.zeros_like(side), np.full(side.shape, 10, np.float32),
        normalize_side_white_params({'dust_mode': 'suppress', 'mapping_margin_px': 0}))
    assert item['dust']['overlap_ratio'] == 1
    assert item['disposition'] == 'dust_suppressed'
    assert item['crop_bounds']['omit'] == item['crop_bounds']['side']
    assert composite[:, 800:1200].max() == 255


def test_dust_in_bbox_hole_does_not_suppress_candidate():
    item = candidate()
    item["_mask"][:, 10:30] = 0
    item["area_px"] = 400
    omit = np.zeros((100,100), np.uint8)
    omit[20:40,30:50] = 255
    evaluate(item, mode="suppress", omit=omit,detector=lambda crop:(True,crop,1,"hole"))
    assert item["dust"]["overlap_ratio"] == 0
    assert item["remaining_area_px"] == 400


@pytest.mark.parametrize("shape,detector", [((90,100),None), ((100,100),None), ((100,100),lambda c: (_ for _ in ()).throw(ValueError("bad mask")))])
def test_unavailable_dust_preserves_candidates(shape, detector):
    item = candidate()
    evaluate(item, mode="suppress",omit=np.zeros(shape,np.uint8),detector=detector)
    assert item["dust"]["status"] == "unavailable"
    assert item["remaining_area_px"] == 800


def test_zone_transform_uses_detection_image_coordinates():
    transform = np.array([[2,0,10],[0,2,10],[0,0,1]],np.float64)
    mask,zones = _zone_mask({"zones":[{"x":50,"y":50,"w":40,"h":20}]},(100,100),transform,normalize_side_white_params())
    assert mask[25,25] == 255 and mask[45,45] == 0
    assert np.allclose(zones[0]["side_polygon"],[[20,20],[40,20],[40,30],[20,30]])


def bomb_context(points, kind="point", prefix="W0F00000"):
    return {"product_resolution":[100,100],"bombs":[{"coordinates":points,"defect_type":kind,"image_prefix":prefix,"defect_code":"TEST"}]}


def test_bomb_requires_detected_evidence_and_front_source():
    quad = np.array([[0,0],[100,0],[100,100],[0,100]],np.float32)
    options = normalize_side_white_params({"bomb_tolerance_product_px":5,"bomb_force_detection_enabled":True})
    bombs, force = _bomb_geometry(bomb_context([[40,30],[80,80]]),quad,np.eye(3),(100,100),options)
    assert force[30,40] and all(b["status"] == "missed" for b in bombs)
    item = candidate()
    _match_bombs([item],bombs)
    assert bombs[0]["status"] == "matched" and bombs[1]["status"] == "missed"
    assert item["bomb_ids"] == [1]
    invalid,_ = _bomb_geometry(bomb_context([[20,20]],prefix="SW0F00000"),quad,np.eye(3),(100,100),options)
    assert invalid == []


@pytest.mark.parametrize('prefix', ['R0F00000', 'G0F00000', 'B0F00000', 'U0F00000',
                                    'WGF00000', 'WGF50500', 'STANDARD', 'SW0F00000', ''])
def test_other_screen_bombs_do_not_force_or_exempt_side_candidates(prefix):
    quad = np.array([[0,0],[100,0],[100,100],[0,100]], np.float32)
    options = normalize_side_white_params({'bomb_force_detection_enabled': True})
    bombs, force = _bomb_geometry(bomb_context([[40,30]], prefix=prefix), quad, np.eye(3), (100,100), options)
    item = candidate()
    _match_bombs([item], bombs)
    assert bombs == [] and not force.any() and item['bomb_ids'] == []


@pytest.mark.parametrize('prefix', ['W0F00000', 'w0f00000', 'W0F00000_155813.tif'])
def test_only_white_screen_bombs_survive_mixed_definitions(prefix):
    quad = np.array([[0,0],[100,0],[100,100],[0,100]], np.float32)
    ctx = bomb_context([[80,80]], prefix='R0F00000')
    ctx['bombs'] += bomb_context([[40,30]], prefix=prefix)['bombs']
    options = normalize_side_white_params({'bomb_tolerance_product_px': 5, 'bomb_force_detection_enabled': True})
    bombs, force = _bomb_geometry(ctx, quad, np.eye(3), (100,100), options)
    item = candidate()
    _match_bombs([item], bombs)
    assert len(bombs) == 1 and bombs[0]['status'] == 'matched'
    assert bombs[0]['image_prefix'] == prefix and item['bomb_ids'] == [1]
    assert force[30,40] and not force[80,80]


def test_line_bomb_rejects_dot_and_wrong_direction():
    quad=np.array([[0,0],[100,0],[100,100],[0,100]],np.float32)
    bombs,_=_bomb_geometry(bomb_context([[0,30],[90,30]],kind="line"),quad,np.eye(3),(100,100),normalize_side_white_params({"bomb_tolerance_product_px":15}))
    dot=candidate()
    dot["side_contour"]=[[20,20],[30,20],[30,30],[20,30]]
    vertical=candidate();vertical["id"]=2
    vertical["side_contour"]=[[25,10],[28,10],[28,60],[25,60]]
    line=candidate();line["id"]=3
    line["side_contour"]=[[20,29],[60,29],[60,32],[20,32]]
    _match_bombs([dot,vertical,line],bombs)
    assert bombs[0]["candidate_ids"] == [3]


def pair(tmp_path, defect=True):
    front=np.zeros((800,1200),np.uint8)
    cv2.rectangle(front,(80,70),(1120,730),90,-1)
    side=front.copy()
    if defect:
        cv2.ellipse(side,(570,400),(18,8),0,0,360,125,-1)
    sp,fp=tmp_path/'SW0F00000_153501.tif',tmp_path/'W0F00000_153501.tif'
    assert cv2.imwrite(str(sp),side) and cv2.imwrite(str(fp),front)
    return sp,fp


def test_side_only_defect_kept_with_normal_front_and_generates_composite(tmp_path):
    side,front=pair(tmp_path)
    result=inspect_side_white_image(side,front,tmp_path/'out',omit_image=np.zeros((800,1200),np.uint8),
                                    dust_detector=lambda crop:(False,np.zeros_like(crop),0,"clean"))
    assert result['status']=='CANDIDATES'
    assert result['summary']['retained'] > 0
    c=result['candidates'][0]
    assert Path(result['artifacts'][c['composite_key']]).is_file()
    assert c['dust']['overlap_ratio']==0
    json.dumps(result)


def test_zero_candidates_still_reports_missed_bomb_and_zone_conflict(tmp_path):
    side,front=pair(tmp_path,False)
    context=bomb_context([[50,50]])
    context['zones']=[{'x':0,'y':0,'w':1200,'h':800}]
    p=inspect_side_white_image(side,front,tmp_path/'out',context=context)
    assert p['status']=='NO_CANDIDATES'
    assert p['bombs'][0]['status']=='missed'
    assert p['bombs'][0]['overlaps_exclusion']


def test_all_excluded_has_distinct_status_and_composite(tmp_path):
    side,front=pair(tmp_path)
    p=inspect_side_white_image(side,front,tmp_path/'out',context={'zones':[{'x':0,'y':0,'w':1200,'h':800}]})
    assert p['status']=='FILTERED'
    assert p['summary']['raw'] > 0 and p['summary']['retained']==0
    assert p['candidates'][0]['remaining_area_px']==0
    assert p['candidates'][0]['composite_key']


def test_missing_mapping_keeps_candidate_and_bomb_unavailable(tmp_path):
    side,_=pair(tmp_path)
    p=inspect_side_white_image(side,None,tmp_path/'out',context={**bomb_context([[50,50]]),'zones':[{'x':0,'y':0,'w':1200,'h':800}]})
    assert p['status']=='CANDIDATES'
    assert p['exclusion_status']=='unavailable'
    assert p['bombs'][0]['status']=='unavailable'


@pytest.mark.parametrize('params',[{'dust_mode':'invalid'},{'apply_exclusions':1},{'dust_overlap_ratio':0},{'bomb_tolerance_product_px':-1},{'crop_padding_px':1.1}])
def test_evidence_parameters_validate(params):
    with pytest.raises(ValueError):
        normalize_side_white_params(params)
