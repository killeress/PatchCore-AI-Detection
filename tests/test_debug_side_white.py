import copy
import io
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from jinja2 import Environment, FileSystemLoader

from capi_config import CAPIConfig, normalize_side_white_params
from capi_database import CAPIDatabase
from capi_side_white import find_side_white_pair
from capi_web import CAPIWebHandler


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def debug_side(tmp_path, monkeypatch):
    folder = tmp_path / 'images'
    folder.mkdir()
    raw = np.zeros((500, 700), np.uint8)
    cv2.rectangle(raw, (50, 50), (650, 450), 90, -1)
    cv2.imwrite(str(folder / 'W0F00000_153501.tif'), raw)
    cv2.circle(raw, (350, 250), 8, 120, -1)
    side = folder / 'SW0F00000_153501.tif'
    cv2.imwrite(str(side), raw)
    omit = np.zeros_like(raw)
    cv2.circle(omit, (350, 250), 12, 230, -1)
    cv2.imwrite(str(folder / 'SPINIGBI _153501.tif'), omit)
    db = CAPIDatabase(str(tmp_path / 'results.db'))
    record_id = db.save_inference_record(
        glass_id='DEBUG-SIDE', model_id='GN140JCAL010S', machine_no='CAPI1',
        resolution=(1920, 1200), machine_judgment='OK', ai_judgment='OK',
        image_dir=str(folder), total_images=1, ng_images=0, ng_details='[]',
        request_time='2026-10-28 10:00:00', response_time='2026-10-28 10:00:01',
        processing_seconds=.1, client_response_text='original response',
        aoi_machine_coords=json.dumps({'W0F00000': [{'defect_code': 'CSJBL',
            'product_x': 960, 'product_y': 600, 'coordinate_space': 'product'}]}),
        client_bomb_info=json.dumps({'image_prefix': 'W0F00000', 'defect_type': 'point',
                                     'coordinates': [[960, 600]]}),
    )
    db.save_side_white_result(record_id, {'status': 'NO_CANDIDATES', 'candidates': [],
        'algorithm': 'old', 'side_image': side.name,
        'parameters': normalize_side_white_params({'min_contrast_gray': 70})})
    db.update_config_param('side_white_detection_params', normalize_side_white_params({'dust_overlap_ratio': .6}))
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    config = CAPIConfig(side_white_detection_enabled=False)
    handler.inferencer = SimpleNamespace(config=config, edge_inspector=SimpleNamespace(config=SimpleNamespace(
        all_exclude_zones_by_product={'J': [{'enabled': True, 'x': 0, 'y': 0, 'w': 10, 'h': 10}]},
    )))
    handler.db = db
    monkeypatch.setattr(CAPIWebHandler, '_debug_heatmap_dir', tmp_path / 'debug')
    handler.init_jinja()
    responses = []
    handler._send_json = lambda payload, status=200, **kw: responses.append((status, payload))
    return SimpleNamespace(handler=handler, db=db, record_id=record_id, side=side,
                           config=config, responses=responses)


def post(env, data):
    body = json.dumps(data).encode('utf-8')
    env.handler.headers = {'Content-Length': str(len(body))}
    env.handler.rfile = io.BytesIO(body)
    env.handler._handle_debug_side_white_run()
    return env.responses[-1]


def test_defaults_use_current_settings_and_record_source(debug_side):
    env = debug_side
    env.db.update_config_param('inference_rotate_180_enabled', True)
    env.handler._handle_debug_side_white_defaults({'record_id': [str(env.record_id)]})
    status, data = env.responses[-1]
    assert status == 200
    assert data['parameters']['min_contrast_gray'] == 2.2  # not historical 70
    assert data['parameters']['dust_overlap_ratio'] == .6  # not config default .8
    assert data['parameters']['csjbl_preview_radius_product_px'] == 50
    assert data['rotate_180'] is True
    assert env.config.inference_rotate_180_enabled is False
    assert data['source']['image_path'] == str(env.side)
    assert data['source']['model_id'] == 'GN140JCAL010S'
    assert data['source']['product_resolution'] == [1920, 1200]
    env.db.update_config_param('side_white_detection_params', {'min_contrast_gray': 9,
                                                             'csjbl_preview_radius_product_px': 75})
    env.handler._handle_debug_side_white_defaults({})
    assert env.responses[-1][1]['parameters']['min_contrast_gray'] == 9
    assert env.responses[-1][1]['parameters']['csjbl_preview_radius_product_px'] == 75


def test_debug_reuses_exact_saved_front_geometry_and_rejects_rotation_mismatch(debug_side, monkeypatch):
    import capi_side_white as sw
    env = debug_side
    front = env.side.with_name('W0F00000_153501.tif')
    quad = sw.detect_panel_quad(cv2.imread(str(front), cv2.IMREAD_GRAYSCALE))
    geometries = sw.snapshot_front_geometries([
        SimpleNamespace(image_path=front, image_size=(700, 500), panel_polygon=quad)])
    stored = {
        'status': 'NO_CANDIDATES', 'candidates': [], 'side_image': env.side.name,
        'rotation_applied': False, 'evidence_context': {'front_geometries': geometries},
    }
    conn = env.db._get_conn()
    try:
        conn.execute('UPDATE side_white_results SET payload=? WHERE record_id=?',
                     (json.dumps(stored), env.record_id))
        conn.commit()
    finally:
        conn.close()
    before = env.db.get_record_detail(env.record_id)
    original = sw.detect_panel_quad

    def localize(gray, **kwargs):
        assert kwargs.get('view') != 'front', 'saved front must not be localized again'
        return original(gray, **kwargs)

    monkeypatch.setattr(sw, 'detect_panel_quad', localize)
    status, result = post(env, {'record_id': env.record_id, 'rotate_180': False})
    assert status == 200
    assert result['payload']['mapping']['front_geometry_source'] == 'formal_inference'
    status, result = post(env, {'record_id': env.record_id, 'rotate_180': True})
    assert status == 200
    assert result['payload']['mapping']['status'] == 'unavailable'
    assert '旋轉方向不一致' in result['payload']['mapping']['reason']
    assert env.db.get_record_detail(env.record_id) == before


def test_debug_radius_reads_settings_and_override_only_changes_preview(debug_side):
    env = debug_side
    image = cv2.imread(str(env.side), cv2.IMREAD_UNCHANGED)
    cv2.circle(image, (400, 250), 8, 120, -1)  # roughly 160 product px from CSJBL
    cv2.imwrite(str(env.side), image)
    env.db.update_config_param('side_white_detection_params', {'csjbl_preview_radius_product_px': 20})
    before, settings = env.db.get_record_detail(env.record_id), env.db.get_all_config_params()
    status, default = post(env, {'record_id': env.record_id})
    assert status == 200
    small = default['payload']
    assert small['parameters']['csjbl_preview_radius_product_px'] == 20
    assert small['candidate_display']['tolerance_product_px'] == 20
    assert small['parameters']['dust_mode'] == 'suppress'
    assert small['candidate_display']['candidate_ids'] == []
    assert small['candidates'][0]['disposition'] == 'dust_suppressed'
    status, overridden = post(env, {'record_id': env.record_id,
        'parameters': {'csjbl_preview_radius_product_px': 200}})
    assert status == 200
    large = overridden['payload']
    assert large['parameters']['csjbl_preview_radius_product_px'] == 200
    assert large['candidate_display']['tolerance_product_px'] == 200
    assert large['candidate_display']['candidate_ids'] == []
    assert small['candidate_display']['reference_total'] == large['candidate_display']['reference_total'] == 1
    assert 'CSJBL 附近 200 產品 px' in overridden['result_html']
    assert small['candidates'] == large['candidates']
    assert small['summary'] == large['summary']
    assert small['status'] == large['status']
    assert env.db.get_all_config_params() == settings
    assert env.db.get_record_detail(env.record_id) == before


@pytest.mark.parametrize('rotated', [False, True])
def test_run_uses_real_pipeline_isolated_settings_record_and_artifacts(debug_side, rotated):
    env = debug_side
    before = env.db.get_record_detail(env.record_id)
    settings = env.db.get_all_config_params()
    config = copy.deepcopy(env.config)
    status, result = post(env, {'record_id': env.record_id, 'rotate_180': rotated,
                               'parameters': {'min_contrast_gray': 3}})
    assert status == 200, result
    payload = result['payload']
    assert payload['status'] != 'ERROR', payload
    assert payload['parameters']['min_contrast_gray'] == 3
    assert payload['parameters']['dust_overlap_ratio'] == .6
    assert payload['parameters']['dust_mode'] == 'suppress'
    assert payload['rotation_applied'] is rotated
    assert payload['candidates'] and payload['bombs'][0]['status'] == 'matched'
    assert payload['omit']['image'] == 'SPINIGBI _153501.tif'
    assert payload['candidates'][0]['dust']['overlap_ratio'] > .8
    assert payload['candidates'][0]['dust']['status'] == 'suppressed'
    assert payload['candidates'][0]['disposition'] == 'dust_suppressed'
    assert payload['evidence_context']['zones'][0]['w'] == 10
    assert payload['candidate_display']['candidate_ids'] == []
    assert payload['evidence_context']['aoi_csjbl_points'][0]['defect_code'] == 'CSJBL'
    assert 'side_all' in payload['artifacts'] and 'residual_all' in payload['artifacts']
    assert all(Path(p).is_file() for p in payload['artifacts'].values())
    assert '/debug/heatmaps/side_white_' in result['result_html']
    assert '/api/side-white/image' not in result['result_html']
    assert 'DEBUG 結果' in result['result_html'] and 'DEBUG 側拍推論 →' not in result['result_html']
    assert 'data-sw-all-url="/debug/heatmaps/' in result['result_html']
    assert 'data-sw-nearby-note=' in result['result_html']
    status, second = post(env, {'record_id': env.record_id, 'parameters': {'min_contrast_gray': 60}})
    assert status == 200
    assert second['payload']['status'] == 'NO_CANDIDATES'
    assert second['payload']['candidates'] == []
    assert set(payload['artifacts'].values()).isdisjoint(second['payload']['artifacts'].values())
    assert env.db.get_record_detail(env.record_id) == before
    assert env.db.get_all_config_params() == settings
    assert env.config == config


def test_record_debug_keeps_original_acquisition_and_rejects_missing_image(debug_side):
    env = debug_side
    newer = env.side.with_name('SW0F00000_153502.tif')
    shutil.copyfile(env.side, newer)
    os.utime(env.side, (1, 1))
    assert find_side_white_pair(env.side.parent)[0] == newer
    status, data = post(env, {'record_id': env.record_id, 'image_path': str(newer)})
    assert status == 200 and data['payload']['side_image'] == env.side.name
    assert data['payload']['front_image'] == 'W0F00000_153501.tif'
    env.side.unlink()
    status, data = post(env, {'record_id': env.record_id})
    assert status == 400 and '找不到側拍白畫面' in data['error']


@pytest.fixture
def mapped_debug_side(debug_side, monkeypatch):
    env = debug_side
    client_root = r'\\side-debug-client\camera'
    env.client_folder = client_root + r'\images'
    env.handler._capi_server_instance = SimpleNamespace(path_mapping={
        client_root.upper(): env.side.parent.parent.as_posix(),
    })
    with env.db._get_conn() as conn:
        conn.execute('UPDATE inference_records SET image_dir=? WHERE id=?',
                     (env.client_folder, env.record_id))
    # The client share is unavailable directly; avoid network access on Windows.
    local_is_dir = Path.is_dir
    monkeypatch.setattr(Path, 'is_dir', lambda path: False
        if str(path).replace('\\', '/').lower().startswith('//side-debug-client/')
        else local_is_dir(path))
    return env


def test_record_debug_maps_client_directory_and_keeps_original_acquisition(mapped_debug_side):
    env = mapped_debug_side
    newer = env.side.with_name('SW0F00000_153502.tif')
    shutil.copyfile(env.side, newer)
    os.utime(env.side, (1, 1))
    assert find_side_white_pair(env.side.parent)[0] == newer
    before = env.db.get_record_detail(env.record_id)

    env.handler._handle_debug_side_white_defaults({'record_id': [str(env.record_id)]})
    status, data = env.responses[-1]
    assert status == 200, data
    assert data['source']['image_path'] == str(env.side)
    status, data = post(env, {'record_id': env.record_id, 'image_path': str(newer)})
    assert status == 200, data
    assert data['payload']['status'] not in {'ERROR', 'NO_IMAGE'}
    assert data['payload']['side_image'] == env.side.name
    assert data['payload']['front_image'] == 'W0F00000_153501.tif'
    assert env.db.get_record_detail(env.record_id) == before

    env.side.unlink()
    status, data = post(env, {'record_id': env.record_id})
    assert status == 400 and '找不到側拍白畫面' in data['error']


@pytest.mark.parametrize('folder', [False, True])
def test_standalone_debug_maps_client_file_or_directory(mapped_debug_side, folder):
    env = mapped_debug_side
    path = env.client_folder if folder else env.client_folder + '\\' + env.side.name
    status, data = post(env, {'image_path': path, 'model_id': 'GN140JCAL010S'})
    assert status == 200, data
    assert data['payload']['status'] not in {'ERROR', 'NO_IMAGE'}
    assert data['payload']['side_image'] == env.side.name


@pytest.mark.parametrize('folder', [False, True])
def test_standalone_source_without_front_or_omit(debug_side, folder):
    env = debug_side
    for filename in ['W0F00000_153501.tif', 'SPINIGBI _153501.tif']:
        (env.side.parent / filename).unlink()
    status, data = post(env, {'image_path': str(env.side.parent if folder else env.side),
                            'model_id': 'GN140JCAL010S'})
    assert status == 200 and data['payload']['candidates']
    assert data['payload']['mapping']['status'] == 'unavailable'
    assert data['payload']['omit']['status'] == 'unavailable'
    assert data['payload']['candidate_display']['candidate_ids'] == []
    assert '沒有 AOI CSJBL' in data['payload']['candidate_display']['reason']
    assert 'side_all' in data['payload']['artifacts']


@pytest.mark.parametrize('data', [None, [], {'record_id': True}, {'record_id': 'bad'},
    {'record_id': 999999}, {'parameters': []}, {'parameters': {'min_area_px': 1.5}},
    {'parameters': {'min_contrast_gray': 0}}, {'parameters': {'dust_mode': 'invalid'}},
    {'parameters': {'csjbl_preview_radius_product_px': -1}},
    {'parameters': {'csjbl_preview_radius_product_px': 501}},
    {'parameters': {'apply_exclusions': 'false'}}, {'rotate_180': 'false'},
    {'image_path': ''}, {'image_path': 'relative/SW0F00000.tif'}])
def test_invalid_debug_requests_return_error(debug_side, data):
    status, result = post(debug_side, data)
    assert status == 400 and result['error']


@pytest.mark.parametrize('embedded', [False, True])
def test_record_and_summary_links_open_side_debug(debug_side, embedded):
    detail = debug_side.db.get_record_detail(debug_side.record_id)
    template = debug_side.handler.jinja_env.get_template('_side_white_result.html')
    rendered = template.render(detail=detail, embedded=embedded)
    assert f'/debug?tab=side-white&amp;record_id={debug_side.record_id}' in rendered


def test_debug_routes_dispatch(debug_side):
    handler = debug_side.handler
    calls = []
    handler._handle_debug_side_white_defaults = lambda query: calls.append(query)
    handler._handle_debug_side_white_run = lambda: calls.append('run')
    handler.path = '/api/debug/side-white?record_id=1'
    handler.do_GET()
    handler.do_POST()
    assert calls == [{'record_id': ['1']}, 'run']


def test_debug_ui_load_edit_run_reset_and_error_states(tmp_path):
    if not shutil.which('node'):
        pytest.skip('Node required for debug interaction check')
    script = Environment(loader=FileSystemLoader(ROOT / 'templates')).get_template('_debug_side_white.js').render()
    harness = r'''
const assert = require('node:assert/strict');
const elements = {};
const el = id => elements[id] ||= {value:'', checked:false, innerHTML:'', textContent:'',querySelectorAll:()=>[]};
let ready, calls = [], fail = false, failDefaults = false;
const window = {location:{search:'?tab=side-white&record_id=42'}};
const document = {getElementById:el, querySelector:()=>({}), addEventListener:(_,fn)=>{ready=fn;}};
let tab;
function switchTab(value) {tab=value;}
const defaults = {min_contrast_gray:2.2, noise_sigma_factor:4.5, min_area_px:20, edge_margin_px:16,
 dust_overlap_ratio:.6,mapping_margin_px:3,crop_padding_px:32,bomb_tolerance_product_px:50,csjbl_preview_radius_product_px:70,
 apply_exclusions:true,bomb_check_enabled:true,bomb_force_detection_enabled:false,dust_mode:'observe'};
async function fetch(url, options) {
 calls.push([url,options]);
 if (options) {
  if (fail) throw new Error('測試連線失敗');
  return {ok:true,json:async()=>({result_html:'<section>測試結果</section>',payload:{status:'CANDIDATES'}})};
 }
 if (failDefaults) return {ok:false,json:async()=>({error:'影像不存在'})};
 return {ok:true,json:async()=>({parameters:defaults,rotate_180:true,
  source:{record_id:42,glass_id:'GLASS',model_id:'GN140JCAL010S',image_path:'D:/SW0F00000.tif',product_resolution:[1920,1200]}})};
}
'''
    checks = r'''
(async()=>{
 ready();
 await new Promise(resolve=>setImmediate(resolve));
 assert.equal(tab,'tab-side-white');
 assert.equal(el('sw-debug-path').readOnly,true);
 assert.equal(el('sw-debug-rotate').checked,true);
 assert.match(el('sw-debug-params').innerHTML,/value="0.6"/);
 assert.match(el('sw-debug-params').innerHTML,/id="sw-debug-csjbl_preview_radius_product_px"[^>]*value="70"/);
 assert.equal(el('sw-debug-params').innerHTML.includes('sw-debug-dust_mode'),false);
 assert.equal(el('sw-debug-run').disabled,false);
 for (const [key] of sideWhiteParamFields) el('sw-debug-'+key).valueAsNumber=defaults[key];
 for (const [key] of sideWhiteSwitches) el('sw-debug-'+key).checked=defaults[key];
 el('sw-debug-min_contrast_gray').valueAsNumber=8;
 el('sw-debug-csjbl_preview_radius_product_px').valueAsNumber=10;
 el('sw-debug-apply_exclusions').checked=false;
 el('sw-debug-bomb_force_detection_enabled').checked=true;
 const event={preventDefault(){},target:{reportValidity:()=>true}};
 const pending=runSideWhiteDebug(event);
 assert.equal(el('sw-debug-run').disabled,true);
 await runSideWhiteDebug(event); // double click must not start another request
 await pending;
 const sent=JSON.parse(calls[1][1].body);
 assert.equal(calls.length,2);
 assert.equal(sent.record_id,'42');
 assert.equal(sent.parameters.min_contrast_gray,8);
 assert.equal(sent.parameters.csjbl_preview_radius_product_px,10);
 assert.equal(sent.parameters.apply_exclusions,false);
 assert.equal(sent.parameters.bomb_force_detection_enabled,true);
 assert.equal(sent.parameters.dust_mode,'suppress');
 assert.equal(sent.rotate_180,true);
 assert.match(el('sw-debug-result').innerHTML,/測試結果/);
 assert.equal(el('sw-debug-fields').disabled,false);
 // The switch changes both displayed images and full-size/residual links without a request.
 const previews = ['IMG','A','A'].map(tagName => ({tagName,dataset:{swNearbyUrl:'/near.jpg',swAllUrl:'/all.jpg'},
  setAttribute(key,value){this[key]=value;}}));
 const note = {dataset:{swNearbyNote:'附近候選',swAllNote:'全圖候選'}};
 el('sw-debug-result').querySelectorAll=selector=>selector==='[data-sw-preview-kind]'?previews:[note];
 assert.equal(el('sw-debug-show-all').checked,false);
 el('sw-debug-show-all').checked=true;
 updateSideWhiteDebugCandidateDisplay();
 assert.equal(previews[0].src,'/all.jpg');
 assert.equal(previews[1].href,'/all.jpg');
 assert.equal(previews[2].href,'/all.jpg');
 assert.equal(note.textContent,'全圖候選');
 assert.equal(calls.length,2);
 el('sw-debug-show-all').checked=false;
 updateSideWhiteDebugCandidateDisplay();
 assert.equal(previews[0].src,'/near.jpg');
 assert.equal(previews[1].href,'/near.jpg');
 assert.equal(note.textContent,'附近候選');
 el('sw-debug-show-all').checked=true;
 defaults.min_contrast_gray=9;
 await loadSideWhiteDebugDefaults();
 assert.equal(el('sw-debug-show-all').checked,false);
 assert.equal(previews[0].src,'/near.jpg');
 assert.match(el('sw-debug-params').innerHTML,/value="9"/);
 assert.match(el('sw-debug-params').innerHTML,/id="sw-debug-csjbl_preview_radius_product_px"[^>]*value="70"/);
 fail=true;
 await runSideWhiteDebug(event);
 assert.equal(el('sw-debug-result').innerHTML,'');
 assert.match(el('sw-debug-message').textContent,/測試連線失敗/);
 assert.equal(el('sw-debug-run').disabled,false);
 swDebugReady=false; failDefaults=true;
 await loadSideWhiteDebugDefaults();
 assert.match(el('sw-debug-message').textContent,/影像不存在/);
 assert.equal(el('sw-debug-run').disabled,true);
 assert.equal(el('sw-debug-reset').disabled,false);
 assert.ok(calls.every(([url])=>url.startsWith('/api/debug/side-white')));
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    file = tmp_path / 'debug-side.cjs'
    file.write_text(harness + script + checks, encoding='utf-8')
    result = subprocess.run(['node', str(file)], capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
