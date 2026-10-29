"""Validate side evidence integration with existing settings, OMIT and web paths."""
import io
import json
import threading
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


@pytest.mark.parametrize('scenario', ['normal', 'rotated', 'disabled', 'missing', 'error',
                                     'localization_error', 'without_csjbl'])
def test_page_rerun_replaces_side_results_and_waits_for_evidence(tmp_path, monkeypatch, scenario):
    import capi_server
    import capi_side_white
    from capi_station_adapter import create_station_adapter

    side, front = write_pair(tmp_path)
    omit = np.zeros((500, 700), np.uint8)
    cv2.circle(omit, (350, 250), 12, 230, -1)
    cv2.imwrite(str(tmp_path / 'SPINIGBI _153501.tif'), omit)
    config = CAPIConfig()
    config.side_white_detection_enabled = scenario != 'disabled'
    config.inference_rotate_180_enabled = scenario == 'rotated'
    config.side_white_detection_params = normalize_side_white_params()
    config.within_spec_judgment_enabled = False
    formal_front = cv2.imread(str(front), cv2.IMREAD_GRAYSCALE)
    if config.inference_rotate_180_enabled:
        formal_front = cv2.rotate(formal_front, cv2.ROTATE_180)
    formal_quad = capi_side_white.detect_panel_quad(formal_front)
    inferencer = SimpleNamespace(
        config=config, station_adapter=create_station_adapter('capi'),
        edge_inspector=SimpleNamespace(config=SimpleNamespace(
            all_exclude_zones_by_product={'J': [{'enabled': True, 'x': 0, 'y': 0, 'w': 10, 'h': 10}]})),
        process_panel=MagicMock(return_value=([SimpleNamespace(
            image_path=front, image_size=(700, 500), panel_polygon=formal_quad,
            edge_defects=[], preprocess_steps=[])], None, False, '', False, None, {})),
    )
    db = CAPIDatabase(str(tmp_path / 'results.db'))
    record = db.save_inference_record(
        glass_id='TEST', model_id='GN140JCAL010S', machine_no='CAPI1', resolution=(1920, 1200),
        machine_judgment='OK', ai_judgment='OK', image_dir=str(tmp_path), total_images=1, ng_images=0,
        ng_details='[]', request_time='2026-09-29 12:00:00', response_time='2026-09-29 12:00:01',
        processing_seconds=.1, client_response_text='unchanged',
        aoi_machine_coords=json.dumps({'W0F00000': [{'defect_code': 'OTHER' if scenario == 'without_csjbl' else 'CSJBL',
                                                   'product_x': 960, 'product_y': 600}]}))
    old_id = db.save_side_white_result(record, {'status': 'CANDIDATES', 'candidates': [{'id': 99}], 'algorithm': 'old'})
    db.save_side_statistics_review(old_id, 'confirmed', 'previous observation', 'QA', '')
    cls = CAPIWebHandler
    for name, value in {'inferencer': inferencer, '_capi_server_instance': None, 'db': db,
                        '_gpu_lock': None, '_rerun_lock': threading.Lock(),
                        '_rerun_tasks': {record: {'status': 'running'}}, 'heatmap_manager': None,
                        'heatmap_base_dir': str(tmp_path / 'heatmaps')}.items():
        monkeypatch.setattr(cls, name, value, raising=False)
    monkeypatch.setattr(capi_server, 'aggregate_judgment', lambda _: ('OK', '[]'))
    monkeypatch.setattr(capi_server, 'results_to_db_data', lambda *_: [])
    monkeypatch.setattr(capi_server.InferenceLogCapture, 'start_capture', lambda: None)
    monkeypatch.setattr(capi_server.InferenceLogCapture, 'stop_capture', lambda: '')
    if scenario == 'missing':
        side.unlink()
    elif scenario == 'error':
        monkeypatch.setattr(capi_side_white, 'inspect_side_white_image', MagicMock(side_effect=ValueError('side failure')))
    elif scenario == 'localization_error':
        failure = capi_side_white.PanelLocalizationError('unstable edge', {
            'source_image': side.name, 'view': 'side', 'stage': 'edge_fit',
            'edge': 'bottom', 'residual_p95_px': 12.0, 'limit_px': 3.0})
        monkeypatch.setattr(capi_side_white, 'detect_panel_quad', MagicMock(side_effect=failure))
    original_save = db.save_side_white_result
    def save_when_running(record_id, payload, **kwargs):
        assert cls._rerun_tasks[record_id]['status'] == 'running'
        assert db.get_side_white_result(old_id) is None
        assert all(Path(path).is_file() for path in payload.get('artifacts', {}).values())
        return original_save(record_id, payload, **kwargs)
    monkeypatch.setattr(db, 'save_side_white_result', save_when_running)
    detail = db.get_record_detail(record)
    detail['client_bomb_info'] = json.dumps({'image_prefix': 'W0F00000', 'defect_type': 'point', 'coordinates': [[960, 600]]})
    cls._rerun_worker(record, detail)
    assert cls._rerun_tasks[record]['status'] == 'done', cls._rerun_tasks[record]
    updated = db.get_record_detail(record)
    assert updated['ai_judgment'] == 'OK' and updated['client_response_text'] == 'unchanged'
    assert db.get_side_white_result(old_id) is None
    if scenario in {'disabled', 'without_csjbl'}:
        assert updated['side_white_result'] is None
        assert db.list_side_white_results()['total'] == 0
        return
    row = updated['side_white_result']
    assert row['review_decision'] == 'unreviewed' and row['review_note'] == ''
    payload = row['payload']
    side_notes = [line for line in updated['inference_log'].splitlines()
                  if line.startswith('[SIDE_WHITE_RESULT] ')]
    assert len(side_notes) == 1
    saved_summary = json.loads(side_notes[0].split(' ', 1)[1])
    assert saved_summary['status'] == payload['status']
    assert row['id'] != old_id and db.list_side_white_results()['total'] == 1
    if scenario in {'missing', 'error', 'localization_error'}:
        assert payload['status'] == ('NO_IMAGE' if scenario == 'missing' else 'ERROR')
        assert payload['candidates'] == [] and payload['artifacts'] == {}
        assert '側拍' in cls._rerun_tasks[record]['message']
        if scenario == 'localization_error':
            assert saved_summary['diagnostics'] == payload['diagnostics']
            assert saved_summary['diagnostics']['edge'] == 'bottom'
            assert saved_summary['diagnostics']['residual_p95_px'] == 12.0
        return
    assert payload['rotation_applied'] == (scenario == 'rotated')
    assert payload['mapping']['front_geometry_source'] == 'formal_inference'
    assert saved_summary['mapping']['front_geometry_source'] == 'formal_inference'
    assert saved_summary['localization']['method'] == 'straight_edges'
    assert payload['omit']['image'] == 'SPINIGBI _153501.tif'
    assert payload['candidates'][0]['dust']['overlap_ratio'] > .8
    assert payload['bombs'][0]['status'] == 'matched'
    assert payload['evidence_context']['zones'][0]['w'] == 10
    # A second click must use new parameters and discard the first run's candidates.
    config.side_white_detection_params = normalize_side_white_params({'min_contrast_gray': 60})
    cls._rerun_tasks[record] = {'status': 'running'}
    cls._rerun_worker(record, db.get_record_detail(record))
    assert cls._rerun_tasks[record]['status'] == 'done'
    again = db.get_record_detail(record)['side_white_result']
    assert again['payload']['parameters']['min_contrast_gray'] == 60
    assert again['payload']['status'] == 'NO_CANDIDATES'
    assert again['payload']['candidates'] == []
    assert db.list_side_white_results()['total'] == 1 and db.get_side_white_result(row['id']) is None


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


@pytest.mark.parametrize('code', ['CSJBL', 'OTHER'])
def test_live_save_only_schedules_aoi_csjbl_and_copies_inference_marks(tmp_path, monkeypatch, code):
    import capi_server
    server = CAPIServer.__new__(CAPIServer)
    server.db = CAPIDatabase(str(tmp_path / 'results.db'))
    server.heatmap_manager = SimpleNamespace(save_panel_heatmaps=lambda **kw: {})
    server.save_overview = server.save_tile_detail = False
    server._get_or_create_inferencer = lambda _: SimpleNamespace(config=CAPIConfig())
    server._save_side_white_result = MagicMock()
    images = [{'mark_source_image': 'W0F00000.tif', 'mark_exclusion_regions': [
        {'name': 'mark_binary', 'x1': 20, 'y1': 30, 'x2': 50, 'y2': 60}]}]
    monkeypatch.setattr(capi_server, 'results_to_db_data', lambda *_: images)
    parsed = {'glass_id': 'LIVE', 'model_id': 'MODEL', 'machine_no': 'CAPI1',
              'resolution': (1920, 1080), 'machine_judgment': 'OK', 'image_dir': str(tmp_path)}
    report = {'W0F00000': [SimpleNamespace(defect_code=code, product_x=100, product_y=100)]}
    server._save_results_async(('test', 123), parsed, [SimpleNamespace()], 'OK', '[]',
                               '2026-10-28 12:00:00', '2026-10-28 12:00:01', .1,
                               aoi_report=report, side_white_enabled=True, side_white_rotate_180=True)
    assert server.db.get_record_detail(1)['ai_judgment'] == 'OK'
    if code == 'OTHER':
        server._save_side_white_result.assert_not_called()
    else:
        context = server._save_side_white_result.call_args.args[-1]
        assert context['mark_rotation_applied'] is True
        assert context['inference_marks'][0]['front_bbox'] == [20, 30, 30, 30]
        images[0]['mark_exclusion_regions'][0]['x1'] = 999
        assert context['inference_marks'][0]['front_bbox'][0] == 20


@pytest.mark.parametrize('rotated', [False, True])
def test_real_side_candidate_uses_inference_mark_and_keeps_unmarked_black_patch(tmp_path, rotated):
    side, front = write_pair(tmp_path)
    # Keep the black patch below the detector's existing relative area ceiling.
    raw = np.zeros((1000, 1400), np.uint8)
    cv2.rectangle(raw, (50, 50), (1350, 950), 90, -1)
    cv2.imwrite(str(front), raw)
    cv2.circle(raw, (350, 250), 8, 120, -1)
    cv2.rectangle(raw, (450, 260), (478, 275), 8, -1)
    cv2.imwrite(str(side), raw)
    # The centre candidate is inside a saved MARK; the dark patch is not.
    context = {'inference_marks': [{'source_image': front.name, 'name': 'mark_binary',
                                    'front_bbox': [310, 210, 80, 80]}],
               'mark_rotation_applied': rotated}
    if rotated:
        context['inference_marks'][0]['front_bbox'] = [1010, 710, 80, 80]
    payload = inspect_side_white_image(side, front, tmp_path / 'out', context=context, rotate_180=rotated)
    assert payload['status'] == 'CANDIDATES', payload.get('reason')
    marked = min(payload['candidates'], key=lambda c: np.linalg.norm(np.array(c['side_raw_xy']) - [350, 250]))
    assert marked['disposition'] == 'mark_suppressed'
    black = min(payload['candidates'], key=lambda c: np.linalg.norm(np.array(c['side_raw_xy']) - [464, 268]))
    assert black['disposition'] == 'retained'
    assert payload['summary']['mark_suppressed'] >= 1
    assert payload['inference_mark_regions'][0]['source_image'] == front.name
    json.dumps(payload)


def test_real_omit_detector_reused_without_model_load(tmp_path):
    side,front=write_pair(tmp_path)
    omit=np.zeros((500,700),np.uint8)
    cv2.circle(omit,(350,250),12,230,-1)
    cv2.imwrite(str(tmp_path/'SPINIGBI _153501.tif'),omit)
    config=CAPIConfig()
    context=snapshot_context(config,None,{'model_id':'GN140JCAL010S'})
    raw,detector,info=load_omit_evidence(tmp_path,side,context,False)
    assert raw is not None and callable(detector),info
    assert info['image']=='SPINIGBI _153501.tif'
    payload=inspect_side_white_image(side,front,tmp_path/'out',context=context,omit_image=raw,dust_detector=detector,omit_info=info)
    assert payload['status']=='FILTERED',payload.get('reason')
    assert payload['summary']['dust_suppressed'] > 0
    assert all(c['dust']['status']!='unavailable' for c in payload['candidates'])
    json.dumps(payload)


@pytest.mark.parametrize('filename', ['SPINIGBI _155809.tif', 'spinigbi_155809.TIFF', 'SPINIGBI155809.tif'])
def test_side_omit_filename_formats_and_front_omit_ignored(tmp_path, filename):
    side, _ = write_pair(tmp_path)
    side = side.rename(tmp_path / 'SW0F00000_155809.tif')
    omit = np.full((500, 700), 15, np.uint8)
    cv2.imwrite(str(tmp_path / filename), omit)
    cv2.imwrite(str(tmp_path / 'PINIGBI _155809.tif'), np.full_like(omit, 255))
    cv2.imwrite(str(tmp_path / 'SPINIGBI _155808.tif'), np.full_like(omit, 255))
    raw, detector, info = load_omit_evidence(tmp_path, side, {'config': CAPIConfig()}, False)
    assert info['image'] == filename and info['pairing'] == 'exact_acquisition'
    assert info['coordinate_space'] == 'side_detection_pixels'
    assert callable(detector) and np.array_equal(raw, omit)


def test_side_omit_unique_timestamp_and_glass_prefix(tmp_path):
    side, _ = write_pair(tmp_path)
    side = side.rename(tmp_path / 'GLASS_SW0F00000_153501.tif')
    omit = np.full((500, 700), 15, np.uint8)
    cv2.imwrite(str(tmp_path / 'GLASS_SPINIGBI _153502.tif'), omit)
    cv2.imwrite(str(tmp_path / 'OTHER_SPINIGBI _153501.tif'), omit)
    _, detector, info = load_omit_evidence(tmp_path, side, {'config': CAPIConfig()}, False)
    assert callable(detector) and info['image'] == 'GLASS_SPINIGBI _153502.tif'
    assert info['pairing'] == 'unique_side_omit'


def test_front_omit_alone_is_not_used_for_side(tmp_path):
    side, _ = write_pair(tmp_path)
    cv2.imwrite(str(tmp_path / 'PINIGBI _153501.tif'), np.zeros((500, 700), np.uint8))
    raw, detector, info = load_omit_evidence(tmp_path, side, {'config': CAPIConfig()}, False)
    assert raw is None and detector is None and 'SPINIGBI' in info['reason']


@pytest.mark.parametrize('rotated', [False, True])
def test_side_omit_without_front_uses_same_rotation_and_generates_evidence(tmp_path, rotated):
    side, front = write_pair(tmp_path)
    front.unlink()
    # Asymmetric position verifies that both images rotate together.
    source = cv2.imread(str(side), cv2.IMREAD_GRAYSCALE)
    source[230:271, 330:371] = 90
    cv2.circle(source, (240, 180), 8, 125, -1)
    cv2.imwrite(str(side), source)
    omit = np.zeros_like(source)
    cv2.circle(omit, (240, 180), 12, 230, -1)
    cv2.imwrite(str(tmp_path / 'SPINIGBI _153501.tif'), omit)
    context = {'config': CAPIConfig()}
    raw, detector, info = load_omit_evidence(tmp_path, side, context, rotated)
    expected = cv2.rotate(omit, cv2.ROTATE_180) if rotated else omit
    assert np.array_equal(raw, expected)
    payload = inspect_side_white_image(side, None, tmp_path / 'out', rotate_180=rotated,
                                      context=context, omit_image=raw, dust_detector=detector, omit_info=info)
    assert payload['mapping']['status'] == 'unavailable'
    assert payload['candidates']
    c = min(payload['candidates'], key=lambda c: np.linalg.norm(np.array(c['side_raw_xy']) - [240, 180]))
    assert c['dust']['overlap_ratio'] > .8, c
    assert c['crop_bounds']['omit'] == c['crop_bounds']['side']
    assert Path(payload['artifacts'][c['composite_key']]).is_file()
    # Report crops use an unannotated grayscale copy with the same bounds as the marked preview.
    clean = cv2.imread(payload['artifacts']['side_clean'], cv2.IMREAD_UNCHANGED)
    assert clean.ndim == 2 and clean.shape == cv2.imread(payload['artifacts']['side']).shape[:2]
    assert payload['preview_bounds']['side_clean'] == payload['preview_bounds']['side']
    assert 'front_clean' not in payload['artifacts']
    CAPIWebHandler.init_jinja()
    html = CAPIWebHandler.jinja_env.get_template('_side_white_result.html').render(
        detail={'side_white_result': {'id': 1, 'status': payload['status'], 'payload': payload}})
    assert '側拍 OMIT（SPINIGBI）' in html and '3 側拍 px' in html


def test_side_omit_overexposure_uses_side_panel_and_shape(tmp_path):
    side, _ = write_pair(tmp_path)
    # Bright external fixture must not invalidate a dark side-panel OMIT.
    omit = np.full((500, 700), 255, np.uint8)
    omit[45:456, 45:656] = 10
    path = tmp_path / 'SPINIGBI _153501.tif'
    cv2.imwrite(str(path), omit)
    _, detector, info = load_omit_evidence(tmp_path, side, {'config': CAPIConfig()}, False)
    assert callable(detector) and 'Scope:product_polygon' in info['exposure_detail']
    cv2.imwrite(str(path), np.zeros((250, 350), np.uint8))
    raw, detector, info = load_omit_evidence(tmp_path, side, {'config': CAPIConfig()}, False)
    assert raw is not None and detector is None and '尺寸不一致' in info['reason']


def test_omit_overexposure_and_ambiguous_acquisition_disable_suppression(tmp_path):
    side,_=write_pair(tmp_path)
    cv2.imwrite(str(tmp_path/'SPINIGBI _153501.tif'),np.full((500,700),255,np.uint8))
    context=snapshot_context(CAPIConfig(),None,{'model_id':'GN140JCAL010S'})
    raw,detector,info=load_omit_evidence(tmp_path,side,context,False)
    assert raw is not None and detector is None and info['status']=='unavailable'
    (tmp_path/'SPINIGBI _153501.tif').rename(tmp_path/'SPINIGBI _153500.tif')
    cv2.imwrite(str(tmp_path/'SPINIGBI _153502.tif'),np.zeros((500,700),np.uint8))
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
