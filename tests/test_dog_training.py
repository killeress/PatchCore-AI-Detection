"""DoG contract across saved recipes, actual tiles and human review assets."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from capi_dog import DEFAULT_PARAMS, normalize_params
from capi_image_preprocess_lab import apply_preprocess_pipeline, normalize_preprocess_pipeline


def dog_pipeline():
    # Non-default anisotropic values catch swapped axes and lost settings.
    return normalize_preprocess_pipeline([{'method':'dog','params':{
        'signal_sigma_x':3.5,'signal_sigma_y':1.5,
        'background_sigma_x':18,'background_sigma_y':9,'gain':27,'offset':117,
    }}])


def sample_image(side=384):
    y,x = np.mgrid[:side,:side]
    white = 3*np.exp(-.5*(((x-side/2)/19)**2+((y-side/2)/9)**2))
    return np.rint(88 + x*.01 + white + 3*np.sin(y*2.1)).astype(np.uint8)


@pytest.mark.parametrize('params',[
    {'gain':float('nan')}, {'offset':float('inf')}, {'signal_sigma_x':True},
    {'gain':101}, {'offset':-1}, {'background_sigma_y':6}, {'unknown':1},
])
def test_dog_pipeline_rejects_invalid_recipe(params):
    with pytest.raises(ValueError):
        normalize_preprocess_pipeline([{'method':'dog','params':params}])


def test_dog_math_preserves_channels_and_uses_float_subtraction():
    image = sample_image(96)
    color = np.dstack([image, np.flipud(image), np.full_like(image,90)])
    alpha = np.arange(96,dtype=np.uint8)[None,:].repeat(96,0)
    rgba = np.dstack([color,alpha])
    params = dog_pipeline()[0]['params']
    original = rgba.copy()
    result = apply_preprocess_pipeline(rgba,dog_pipeline())['image']
    f = color.astype(np.float32)
    residual = (cv2.GaussianBlur(f,(0,0),3.5,sigmaY=1.5)
                - cv2.GaussianBlur(f,(0,0),18,sigmaY=9))
    expected = np.rint(np.clip(117+27*residual,0,255)).astype(np.uint8)
    np.testing.assert_array_equal(result[:,:,:3],expected)
    np.testing.assert_array_equal(result[:,:,3],alpha)
    np.testing.assert_array_equal(rgba,original)
    assert np.all(result[:,:,2] == params['offset'])


def test_training_inference_and_debug_share_dog_for_identical_extent():
    from capi_white_spot_debug import prepare_white_spot
    image = sample_image(128)
    debug = prepare_white_spot(image,(64,64),128,dog_pipeline()[0]['params'])
    expected = apply_preprocess_pipeline(image,dog_pipeline())['image']
    np.testing.assert_array_equal(debug['enhanced'],expected)
    assert normalize_preprocess_pipeline([{'method':'dog'}])[0]['params'] == DEFAULT_PARAMS


def test_dog_cannot_change_raw_boundary_detection_order():
    from capi_preprocess import PreprocessConfig
    from capi_config import CAPIConfig
    with pytest.raises(ValueError, match='先切分後處理'):
        PreprocessConfig(image_preprocess_pipeline=dog_pipeline())
    with pytest.raises(ValueError, match='先切分後處理'):
        CAPIConfig.from_dict({'image_preprocess_pipeline':dog_pipeline()})


@pytest.mark.parametrize('separate',[False,True])
def test_training_bundle_freezes_dog_and_inference_replays_identical_pixels(tmp_path,monkeypatch,separate):
    from capi_train_new import TrainingConfig, run_training_pipeline
    from capi_config import CAPIConfig
    from capi_preprocess import PreprocessConfig, _generate_tiles

    class DB:
        def list_tile_pool(self,*args,**kwargs):
            return [{'id':i} for i in range(30)]

    def fake_train(*,output_pt_path,**kwargs):
        output_pt_path.write_bytes(b'mock model')
        return {'threshold':.5,'tile_count':30,'ng_count':0,'size_bytes':10,
                'metrics':{'auroc':None,'auroc_grade':'n/a'},'used_tile_ids':[], 'elapsed_seconds':1}

    monkeypatch.setattr('capi_train_new._setup_offline_env',lambda *a,**k:None)
    monkeypatch.setattr('capi_train_new.train_single_submodel',fake_train)
    shared = [] if separate else dog_pipeline()
    zones = {'inner':dog_pipeline(),'edge':[]} if separate else {}
    cfg = TrainingConfig(machine_id='DOG-TEST',panel_paths=[],over_review_root=tmp_path/'unused',
        output_root=tmp_path/'models',training_units=[('WGF50500','inner'),('WGF50500','edge')],
        image_preprocess_pipeline=shared,image_preprocess_pipelines=zones,preprocess_after_tiling=True)
    bundle = run_training_pipeline(job_id='dog_recipe',cfg=cfg,db=DB(),log=lambda msg:None)
    manifest = json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    loaded = CAPIConfig.from_yaml(str(bundle/'machine_config.yaml'))
    assert manifest['preprocess_after_tiling'] is loaded.preprocess_after_tiling is True
    assert manifest['image_preprocess_pipeline'] == loaded.image_preprocess_pipeline == shared
    assert manifest['image_preprocess_pipelines'] == loaded.image_preprocess_pipelines == zones

    image = sample_image()
    def tiles(recipe):
        pc = PreprocessConfig(tile_size=128,tile_stride=128,outer_edge_extend=0,
            image_preprocess_pipeline=recipe.image_preprocess_pipeline,
            image_preprocess_pipelines=recipe.image_preprocess_pipelines,preprocess_after_tiling=True)
        return _generate_tiles(image,(0,0,384,384),None,pc,original_img=image)
    train, infer = tiles(cfg), tiles(loaded)
    assert len(train) == len(infer) > 0
    for left,right in zip(train,infer):
        assert (left.x1,left.y1,left.zone) == (right.x1,right.y1,right.zone)
        np.testing.assert_array_equal(left.image,right.image)
        np.testing.assert_array_equal(left.original_image,image[left.y1:left.y2,left.x1:left.x2])


def test_review_assets_are_model_input_and_aligned_before_image(tmp_path,monkeypatch):
    from capi_preprocess import PreprocessConfig, PanelPreprocessResult, _generate_tiles
    from capi_train_new import TrainingConfig, preprocess_panels_to_pool, review_original_path
    image = sample_image()
    panel = tmp_path/'panel'
    panel.mkdir()
    source = panel/'WGF50500_x.png'
    assert cv2.imwrite(str(source),image)
    pre = PreprocessConfig(tile_size=128,tile_stride=128,outer_edge_extend=0,
        image_preprocess_pipeline=dog_pipeline(),preprocess_after_tiling=True)
    tiles = _generate_tiles(image,(0,0,384,384),None,pre,original_img=image)
    result = PanelPreprocessResult(source,'WGF50500',(0,0,384,384),None,tiles)
    monkeypatch.setattr('capi_train_new.preprocess_panel_folder',lambda *a,**k:{'WGF50500':result})
    records = []
    db = SimpleNamespace(insert_tile_pool=lambda job,rows:records.extend(rows))
    cfg = TrainingConfig(machine_id='DOG',panel_paths=[panel],over_review_root=tmp_path/'unused')
    preprocess_panels_to_pool(job_id='dog',cfg=cfg,preprocess_cfg=pre,db=db,
                             thumb_dir=tmp_path/'review',log=lambda msg:None)
    assert len(records) == len(tiles)
    for row,tile in zip(records,tiles):
        np.testing.assert_array_equal(cv2.imread(row['source_path'],0),tile.image)
        np.testing.assert_array_equal(cv2.imread(row['thumb_path'],0),cv2.resize(tile.image,(96,96)))
        np.testing.assert_array_equal(cv2.imread(str(review_original_path(row['source_path'])),0),tile.original_image)


def test_after_tiling_original_is_aligned_in_canonical_coordinates():
    from capi_preprocess import PreprocessConfig, _generate_tiles
    # The raw source has different coordinates/size after a grid transform.
    canonical = sample_image()
    raw = np.zeros((768,768),np.uint8)
    cfg = PreprocessConfig(tile_size=128,tile_stride=128,outer_edge_extend=0,
        image_preprocess_pipeline=dog_pipeline(),preprocess_after_tiling=True,grid_canonicalization_enabled=True)
    tiles = _generate_tiles(canonical,(0,0,384,384),None,cfg,original_img=raw)
    for tile in tiles:
        np.testing.assert_array_equal(tile.original_image,canonical[tile.y1:tile.y2,tile.x1:tile.x2])


def test_cached_ng_uses_same_dog_and_keeps_unmodified_review_image(tmp_path):
    from capi_preprocess import PreprocessConfig
    from capi_train_new import sample_ng_tiles, review_original_path
    raw = sample_image(128)
    cached = tmp_path/'cache'/'sample.png'
    cached.parent.mkdir()
    assert cv2.imwrite(str(cached),raw)
    records = []
    class DB:
        def list_training_bomb_validation_samples(self,**kwargs):
            return [{'id':3,'lighting':'WGF50500','zone':'inner','crop_path':str(cached)}]
        def list_training_bomb_candidates(self,**kwargs):
            return []
        def insert_tile_pool(self,job_id,rows):
            records.extend(rows)
    result = sample_ng_tiles(job_id='dog_ng',machine_id='DOG',over_review_root=tmp_path/'unused',
        db=DB(),thumb_dir=tmp_path/'review',per_lighting=1,log=lambda msg:None,
        lightings=('WGF50500',),ng_validation_base_dir=cached.parent,
        preprocess_cfg=PreprocessConfig(tile_size=128,image_preprocess_pipeline=dog_pipeline(),preprocess_after_tiling=True))
    assert result['cache_reused'] == 1
    np.testing.assert_array_equal(cv2.imread(records[0]['source_path'],0),apply_preprocess_pipeline(raw,dog_pipeline())['image'])
    np.testing.assert_array_equal(cv2.imread(str(review_original_path(records[0]['source_path'])),0),raw)
    np.testing.assert_array_equal(cv2.imread(str(cached),0),raw)


def test_training_config_materializes_every_dog_parameter(tmp_path):
    from capi_train_new import TrainingConfig
    cfg = TrainingConfig(machine_id='DOG',panel_paths=[],over_review_root=tmp_path,
        image_preprocess_pipeline=[{'method':'dog','params':{'gain':30}}],preprocess_after_tiling=True)
    assert cfg.image_preprocess_pipeline == [{'method':'dog','params':{**DEFAULT_PARAMS,'gain':30.0}}]
