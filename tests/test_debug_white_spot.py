import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from capi_white_spot_debug import DEFAULT_PARAMS, normalize_params, prepare_white_spot, summarize_prediction


def image_with_mura():
    y,x = np.mgrid[:800,:1000]
    white = 3*np.exp(-.5*(((x-500)/37)**2 + ((y-400)/15)**2))
    stripes = 4*np.sin(y*2.1) + .3*np.sin(x*.13)
    return np.rint(90 + .004*x + white + stripes).astype(np.uint8)


def test_context_filter_preserves_faint_mura_and_reduces_background_texture():
    image = image_with_mura()
    before = image.copy()
    result = prepare_white_spot(image, (500,400))
    np.testing.assert_array_equal(image,before)
    assert result['original'].shape == result['enhanced'].shape == (512,512)
    assert result['residual'].dtype == np.float32
    assert result['residual'].min() < 0 < result['residual'].max()
    y,x = np.mgrid[:512,:512]
    r = ((x-256)/35)**2 + ((y-256)/14)**2
    core,background = r<=1, (r>=2.5**2)&(r<=4.5**2)
    def cnr(a):
        return (a[core].mean()-a[background].mean())/a[background].std()
    assert cnr(result['enhanced']) > 3*cnr(result['original'])
    assert result['enhanced'][core].mean() > result['enhanced'][background].mean()+20


@pytest.mark.parametrize('center',[(500,400),(0,0),(999,799)])
def test_padded_crop_matches_full_image_filter_even_near_image_edges(center):
    image = image_with_mura()
    result = prepare_white_spot(image,center,tile_size=128)
    f = image.astype(np.float32)
    whole = cv2.GaussianBlur(f,(0,0),20,sigmaY=6)-cv2.GaussianBlur(f,(0,0),96,sigmaY=32)
    x1,y1,x2,y2 = result['crop_bounds']
    np.testing.assert_allclose(result['residual'],whole[y1:y2,x1:x2],atol=1e-4)
    assert result['center_in_tile'] == [center[0]-x1,center[1]-y1]
    assert result['enhanced'].shape == (128,128)


def test_flat_field_has_no_crop_border_rings():
    result = prepare_white_spot(np.full((700,900),91,np.uint8),(450,350))
    assert np.max(np.abs(result['residual'])) < 1e-4
    assert np.all(result['enhanced'] == 128)


@pytest.mark.parametrize('params',[
    {'signal_sigma_x':float('nan')},{'gain':float('inf')},{'offset':-1},
    {'signal_sigma_x':True},{'background_sigma_y':6},{'signal_sigma_y':0},[],
])
def test_invalid_filter_parameters_rejected(params):
    with pytest.raises(ValueError): normalize_params(params)


def test_center_score_ignores_stronger_outside_peak_on_small_model_map():
    amap = np.zeros((16,16),np.float32)
    amap[8,8] = .25
    amap[1,1] = 1
    result = summarize_prediction(.8,amap,(512,512),(256,256),64,.35)
    assert result['tile_score'] == .8
    assert result['roi_score'] == pytest.approx(.2)
    assert result['roi_passes_threshold'] is False
    amap[8,8] = .75
    assert summarize_prediction(.8,amap,(512,512),(256,256),64,.35)['roi_passes_threshold'] is True


def test_no_heatmap_never_reports_center_pass_or_fail():
    result = summarize_prediction(.8,None,(512,512),(256,256),64,.35)
    assert result['status'] == 'no_heatmap'
    assert result['roi_score'] is None and result['roi_passes_threshold'] is None


class FakePC:
    def __init__(self):
        self.config = SimpleNamespace(tile_size=128,machine_id='TEST',is_new_architecture=True,
            grid_canonicalization_enabled=False,inference_rotate_180_enabled=False,
            threshold_mapping={'WGF50500':{'inner':.35,'edge':.6}},
            image_preprocess_pipeline=[{'method':'laplace_sharpen'}])
        self.model = object()
        self.calls = []
        self.model_requests = []
        self.locked = False
        self.fail_enhanced = False

    def _get_image_prefix(self,name): return 'WGF50500'
    def _get_model_prefix(self,prefix): return prefix
    def _get_model_for(self,machine,prefix,zone):
        assert self.locked
        self.model_requests.append((machine,prefix,zone))
        return self.model

    def predict_tile(self,tile,**kwargs):
        assert self.locked and kwargs['inferencer'] is self.model
        self.calls.append((tile.image.copy(),kwargs))
        if self.fail_enhanced and len(self.calls)%2 == 0: raise RuntimeError('test model failure')
        amap = np.zeros((16,16),np.float32)
        amap[8,8] = .25 if len(self.calls)%2 else .75
        amap[0,0] = 1
        return .8,amap


@pytest.fixture
def env(tmp_path,monkeypatch):
    from capi_web import CAPIWebHandler
    fake = FakePC()
    class Lock:
        def __enter__(self): fake.locked=True
        def __exit__(self,*args): fake.locked=False
    path=tmp_path/'WGF50500_sample.tif'
    cv2.imencode('.tif',image_with_mura())[1].tofile(str(path))
    handler=CAPIWebHandler.__new__(CAPIWebHandler)
    handler.inferencer=fake
    handler._gpu_lock=Lock()
    responses=[]
    handler._send_json=lambda data,status=200:responses.append((status,data))
    monkeypatch.setattr(CAPIWebHandler,'_debug_heatmap_dir',tmp_path/'artifacts')
    def post(**options):
        payload={'image_path':str(path),'center_x':500,'center_y':400,'radius_px':24,**options}
        body=json.dumps(payload).encode()
        handler.headers={'Content-Length':str(len(body))}
        handler.rfile=io.BytesIO(body)
        handler._handle_debug_white_spot()
        return responses[-1]
    return SimpleNamespace(handler=handler,pc=fake,path=path,post=post,folder=tmp_path/'artifacts')


def test_endpoint_passes_displayed_enhancement_to_pc_without_config_mutation(env):
    old_config=copy.deepcopy(vars(env.pc.config))
    status,data=env.post(run_patchcore=True)
    assert status==200 and data['success']
    assert data['params']==DEFAULT_PARAMS
    assert len(env.pc.calls)==2
    for key,(pixels,kwargs) in zip(['original','enhanced'],env.pc.calls):
        saved=cv2.imdecode(np.fromfile(env.folder/Path(data['urls'][key]).name,np.uint8),0)
        np.testing.assert_array_equal(saved,pixels)
        assert kwargs['threshold']==.35 and kwargs['edge_margin_override']==0
        assert kwargs['patchcore_overrides']=={'patchcore_filter_enabled':False}
    assert vars(env.pc.config)==old_config
    assert data['pc']['variants']['original']['roi_score']==pytest.approx(.2)
    assert data['pc']['variants']['enhanced']['roi_score']==pytest.approx(.6)
    assert data['pc']['heatmap_display_range']==[0,1]
    assert env.pc.model_requests==[('TEST','WGF50500','inner')]
    assert all((env.folder/Path(v).name).is_file() for v in data['urls'].values())


def test_preview_and_missing_model_still_return_real_enhanced_images(env):
    status,preview=env.post(run_patchcore=False)
    assert status==200 and preview['pc']['status']=='not_requested' and not env.pc.calls
    env.handler.inferencer=None
    status,unavailable=env.post(run_patchcore=True)
    assert status==200 and unavailable['pc']['status']=='unavailable'
    assert (env.folder/Path(unavailable['urls']['enhanced']).name).is_file()
    assert preview['urls']['enhanced']!=unavailable['urls']['enhanced']


def test_edge_model_override_and_partial_failure_do_not_discard_preview(env):
    env.pc.fail_enhanced=True
    status,data=env.post(run_patchcore=True,zone='edge',threshold=.45)
    assert status==200 and data['pc']['status']=='partial'
    assert data['pc']['model']['zone']=='edge'
    assert data['pc']['threshold']==.45
    assert data['pc']['variants']['enhanced']['status']=='error'
    assert data['pc']['variants']['original']['heatmap_url']
    assert data['urls']['enhanced']


def test_image_rotation_matches_reported_coordinates(env):
    env.pc.config.inference_rotate_180_enabled=True
    status,data=env.post(run_patchcore=False,center_x=0,center_y=0)
    assert status==200 and data['rotation_applied'] is True
    original=cv2.imread(str(env.path),0)
    saved=cv2.imread(str(env.folder/Path(data['urls']['original']).name),0)
    np.testing.assert_array_equal(saved,cv2.rotate(original,cv2.ROTATE_180)[:128,:128])
    assert data['center']==[0,0]


def test_legacy_model_and_alias_mapping_use_the_same_selected_model(env):
    env.pc.config.is_new_architecture=False
    env.pc.threshold=.42
    env.pc._get_inferencer_for_prefix=lambda prefix:env.pc.model
    status,data=env.post(run_patchcore=True)
    assert status==200 and data['pc']['status']=='ok'
    assert data['pc']['threshold']==.42
    env.pc.config.is_new_architecture=True
    env.pc._get_model_prefix=lambda prefix:'SHARED_LIGHT'
    env.pc.config.threshold_mapping={'SHARED_LIGHT':{'inner':.31}}
    status,data=env.post(run_patchcore=True)
    assert data['pc']['threshold']==.31
    assert env.pc.model_requests[-1]==('TEST','SHARED_LIGHT','inner')


def test_incompatible_pixel_grid_model_keeps_preview_without_wrong_coordinate_inference(env):
    env.pc.config.grid_canonicalization_enabled=True
    status,data=env.post(run_patchcore=True)
    assert status==200 and data['pc']['status']=='unavailable'
    assert 'Pixel Grid' in data['pc']['message']
    assert data['urls']['enhanced'] and not env.pc.calls


@pytest.mark.parametrize('options',[
    {'center_x':10000},{'center_x':1.5},{'center_y':None},{'zone':'wrong'},
    {'run_patchcore':'true'},{'threshold':float('nan')},{'radius_px':0},
    {'params':{'signal_sigma_x':96}},{'image_path':'not-present.tif'},
])
def test_bad_requests_return_validation_error_without_pc(env,options):
    status,data=env.post(**options)
    assert status==400 and data['error'] and not env.pc.calls


def test_runtime_manifest_includes_new_debug_dependencies():
    from scripts.build_deploy_zip import CODE_FILES
    assert {'capi_white_spot_debug.py','templates/_debug_white_spot.html','templates/_debug_white_spot.js'} <= set(CODE_FILES)
