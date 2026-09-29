"""Validate side evidence integration with existing settings, OMIT and web paths."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from capi_config import CAPIConfig, normalize_side_white_params
from capi_side_white import load_omit_evidence, snapshot_context, inspect_side_white_image
from capi_server import CAPIServer
from capi_database import CAPIDatabase
from capi_web import CAPIWebHandler


def test_snapshot_does_not_mutate_active_product_or_follow_live_settings():
    config=CAPIConfig()
    edge=SimpleNamespace(all_exclude_zones_by_product={'J':[{'enabled':True,'x':50,'y':50,'w':30,'h':20}]},exclude_zones=[])
    parsed={'model_id':'GN140JCAL010S','bomb_info':{'image_prefix':'W0F00000','defect_type':'point','coordinates':[[20,30]]}}
    ctx=snapshot_context(config,edge,parsed)
    config.dust_extension=99
    edge.all_exclude_zones_by_product['J'][0]['x']=999
    parsed['bomb_info']['coordinates'][0][0]=999
    assert ctx['zones'][0]['x']==50 and edge.exclude_zones==[]
    assert ctx['bombs'][0]['coordinates']==[[20,30]]
    assert ctx['config'].dust_extension!=99
    assert ctx['product_resolution']==[1920,1200]
    other=snapshot_context(config,edge,{'model_id':'OTHERX'})
    assert other['zones']==[] and other['zone_warning']


def test_queued_nested_context_is_snapshot():
    import threading
    server=CAPIServer.__new__(CAPIServer)
    server._async_executor_lock=threading.Lock()
    server._async_executor_shutdown=False
    server._async_executor=MagicMock()
    context={'zones':[{'x':10}], 'config':CAPIConfig()}
    server._queue_save_results_async(side_white_context=context)
    context['zones'][0]['x']=20
    context['config'].dust_extension=99
    queued=server._async_executor.submit.call_args.kwargs['side_white_context']
    assert queued['zones'][0]['x']==10 and queued['config'].dust_extension!=99


def write_pair(tmp_path):
    image=np.zeros((500,700),np.uint8)
    cv2.rectangle(image,(50,50),(650,450),90,-1)
    front=tmp_path/'W0F00000_153501.tif'
    cv2.imwrite(str(front),image)
    cv2.circle(image,(350,250),8,120,-1)
    side=tmp_path/'SW0F00000_153501.tif'
    cv2.imwrite(str(side),image)
    return side,front


def test_real_omit_detector_reused_without_model_load(tmp_path):
    side,front=write_pair(tmp_path)
    omit=np.zeros((500,700),np.uint8)
    cv2.circle(omit,(350,250),12,230,-1)
    cv2.imwrite(str(tmp_path/'PINIGBI _153501.tif'),omit)
    config=CAPIConfig()
    context=snapshot_context(config,None,{'model_id':'GN140JCAL010S'})
    raw,detector,info=load_omit_evidence(tmp_path,side,context,False)
    assert raw is not None and callable(detector),info
    assert info['image']=='PINIGBI _153501.tif'
    payload=inspect_side_white_image(side,front,tmp_path/'out',context=context,omit_image=raw,dust_detector=detector,omit_info=info)
    assert payload['status']=='CANDIDATES',payload.get('reason')
    assert all(c['dust']['status']!='unavailable' for c in payload['candidates'])
    json.dumps(payload)


def test_omit_overexposure_and_ambiguous_acquisition_disable_suppression(tmp_path):
    side,_=write_pair(tmp_path)
    cv2.imwrite(str(tmp_path/'PINIGBI _153501.tif'),np.full((500,700),255,np.uint8))
    context=snapshot_context(CAPIConfig(),None,{'model_id':'GN140JCAL010S'})
    raw,detector,info=load_omit_evidence(tmp_path,side,context,False)
    assert raw is not None and detector is None and info['status']=='unavailable'
    (tmp_path/'PINIGBI _153501.tif').rename(tmp_path/'PINIGBI _153500.tif')
    cv2.imwrite(str(tmp_path/'PINIGBI _153502.tif'),np.zeros((500,700),np.uint8))
    raw,detector,info=load_omit_evidence(tmp_path,side,context,False)
    assert raw is None and detector is None and info['status']=='unavailable'


@pytest.mark.parametrize('rotated',[False,True])
def test_runtime_save_keeps_formal_verdict_and_renders_readonly(tmp_path,rotated):
    side,front=write_pair(tmp_path)
    server=CAPIServer.__new__(CAPIServer)
    server.db=CAPIDatabase(str(tmp_path/'results.db'))
    server.path_mapping={}
    server.heatmap_manager=SimpleNamespace(base_dir=tmp_path/'heatmaps')
    record=server.db.save_inference_record(glass_id='TEST',model_id='MODEL',machine_no='CAPI1',resolution=(1920,1080),machine_judgment='OK',ai_judgment='OK',image_dir=str(tmp_path),total_images=1,ng_images=0,ng_details='[]',request_time='2026-09-29 12:00:00',response_time='2026-09-29 12:00:01',processing_seconds=.1,client_response_text='unchanged')
    context={'product_resolution':[600,400], 'bombs':[{'image_prefix':'W0F00000','defect_type':'point','coordinates':[[300,200]],'defect_code':'B'}], 'zones':[{'enabled':True,'x':0,'y':0,'w':700,'h':500}]}
    server._save_side_white_result(record,{'image_dir':str(tmp_path)},{},rotated,context=context)
    detail=server.db.get_record_detail(record)
    assert detail['ai_judgment']=='OK' and detail['client_response_text']=='unchanged'
    result=detail['side_white_result']; p=result['payload']
    assert p['status']=='FILTERED' and p['bombs'][0]['status']=='matched',p
    assert p['bombs'][0]['overlaps_exclusion']
    assert np.allclose(p['candidates'][0]['side_raw_xy'],[350,250],atol=2)
    CAPIWebHandler.init_jinja()
    html=CAPIWebHandler.jinja_env.get_template('_side_white_result.html').render(detail=detail)
    assert 'Review' not in html and '炸彈' in html and 'candidate_' in html
    handler=CAPIWebHandler.__new__(CAPIWebHandler)
    handler.db=server.db;handler.heatmap_base_dir=str(tmp_path/'heatmaps')
    handler.send_response=MagicMock();handler.send_header=MagicMock();handler.end_headers=MagicMock();handler._send_404=MagicMock();handler.wfile=io.BytesIO()
    handler._handle_api_side_white_image({'id':[str(result['id'])],'kind':[p['candidates'][0]['composite_key']]})
    assert handler.wfile.getvalue().startswith(b'\xff\xd8')
    handler._send_404.assert_not_called()


def test_all_candidate_processing_precedes_composite_limit(tmp_path,monkeypatch):
    import capi_side_white as sw
    side,front=write_pair(tmp_path)
    def many_candidates(gray,quad,params,**kwargs):
        result=[]
        for index in range(105):
            result.append({'id':index+1,'kind':'bright_spot','side_xy':[350.,250.], 'side_bbox':[345,245,10,10], 'side_contour':[[345,245],[354,245],[354,254],[345,254]],'area_px':100,'contrast_gray':10.,'_mask':np.ones((10,10),np.uint8)})
        return result,np.full(gray.shape,10,np.float32),[],2.2,False
    monkeypatch.setattr(sw,'_candidates',many_candidates)
    monkeypatch.setattr(sw,'_save_preview',lambda path,image: str(path.resolve()))
    p=sw.inspect_side_white_image(side,front,tmp_path/'out',context={'product_resolution':[600,400],'bombs':[{'image_prefix':'W0F00000','defect_type':'point','coordinates':[[300,200]]}]})
    assert p['composites_truncated'] and len(p['candidates'])==105
    assert 105 in p['bombs'][0]['candidate_ids']
    assert p['summary']['bomb_candidates']==105


def test_missing_side_keeps_bomb_as_unavailable(tmp_path):
    server=CAPIServer.__new__(CAPIServer)
    server.path_mapping={}
    server.db=MagicMock()
    context={'product_resolution':[600,400], 'bombs':[{'image_prefix':'W0F00000','defect_type':'point','coordinates':[[300,200]]}]}
    server._save_side_white_result(1,{'image_dir':str(tmp_path)},{},False,context=context)
    payload=server.db.save_side_white_result.call_args.args[1]
    assert payload['status']=='NO_IMAGE'
    assert payload['bombs'][0]['status']=='unavailable'


def test_preview_failure_keeps_serializable_candidate_coordinates(tmp_path,monkeypatch):
    import capi_side_white as sw
    side,front=write_pair(tmp_path)
    monkeypatch.setattr(sw,'_save_preview',MagicMock(side_effect=OSError('disk unavailable')))
    p=sw.inspect_side_white_image(side,front,tmp_path/'out')
    assert p['status']=='ERROR'
    assert p['candidates'] and all('side_raw_xy' in c for c in p['candidates'])
    json.dumps(p)
