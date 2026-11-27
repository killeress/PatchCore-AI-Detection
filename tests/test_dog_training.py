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


def test_masked_dog_eliminates_boundary_halo_and_preserves_faint_edge_dark_mura():
    tile_h, tile_w = 256, 256
    # 模擬傾斜面板右邊緣切塊：面板內正常灰階 120，面板外黑背景 20（外露約 22~28 px 黑邊）
    yy, xx = np.mgrid[:tile_h, :tile_w]
    right_edge_x = 234.0 - 0.03 * yy
    mask = np.where(xx <= right_edge_x, 255, 0).astype(np.uint8)

    clean_edge = np.where(mask > 0, 120, 20).astype(np.uint8)
    recipe = [{'method': 'dog'}]  # 預設 sigma=(20,6)/(96,32), gain=40, offset=128

    unmasked_clean = apply_preprocess_pipeline(clean_edge, recipe)['image']
    masked_clean = apply_preprocess_pipeline(clean_edge, recipe, mask=mask)['image']
    # 未遮罩時黑邊會拉低背景平均，在面板內側爆出 255 假白光；遮罩正規化後整張穩定為基準 128
    assert int(unmasked_clean[mask > 0].max()) == 255
    np.testing.assert_array_equal(masked_clean, np.full_like(clean_edge, 128))

    # 在緊貼右邊界處放入僅暗 3 灰階的「極淡黑霧」
    faint_dark_mura = clean_edge.copy()
    mura_patch = np.exp(-0.5 * (((xx - 208.0) / 14.0) ** 2 + ((yy - 128.0) / 10.0) ** 2))
    faint_dark_mura = np.where(
        mask > 0,
        np.rint(120.0 - 3.0 * mura_patch),
        20,
    ).astype(np.uint8)

    unmasked_mura = apply_preprocess_pipeline(faint_dark_mura, recipe)['image']
    masked_mura = apply_preprocess_pipeline(faint_dark_mura, recipe, mask=mask)['image']
    # 未遮罩時黑霧中心被邊緣假白光推頂到 255 吃掉；智慧 DoG 則清楚強化出暗霧（128 -> ~71）
    assert int(unmasked_mura[128, 208]) == 255
    assert int(masked_mura[128, 208]) < 80
    assert int(masked_mura[20, 40]) == 128
    assert np.all(masked_mura[mask == 0] == 128)

    # 全 255 遮罩與無遮罩等價；全 0 遮罩輸出基準 offset；尺寸不符須報錯
    sample = sample_image(128)
    full_mask = np.full(sample.shape, 255, dtype=np.uint8)
    np.testing.assert_array_equal(
        apply_preprocess_pipeline(sample, dog_pipeline(), mask=full_mask)['image'],
        apply_preprocess_pipeline(sample, dog_pipeline())['image'],
    )
    zero_mask = np.zeros(sample.shape, dtype=np.uint8)
    np.testing.assert_array_equal(
        apply_preprocess_pipeline(sample, dog_pipeline(), mask=zero_mask)['image'],
        np.full_like(sample, 117),
    )
    with pytest.raises(ValueError, match='mask shape'):
        apply_preprocess_pipeline(sample, dog_pipeline(), mask=np.ones((64, 64), np.uint8))


def test_slanted_edge_training_and_aoi_inference_tiles_use_panel_masked_dog(tmp_path):
    from capi_config import CAPIConfig
    from capi_inference import AOIReportDefect, CAPIInferencer, ImageResult
    from capi_preprocess import PreprocessConfig, _generate_tiles

    # 窄面板（高度 ~112 < tile_size 128，coverage ~0.87 >= 0.5）使訓練 Grid Tile 必然跨出邊界
    narrow_raw = np.full((180, 420), 18, dtype=np.uint8)
    narrow_poly = np.array([[36, 32], [384, 38], [380, 140], [32, 134]], dtype=np.float32)
    cv2.fillPoly(narrow_raw, [narrow_poly.astype(np.int32)], 125)

    pre_cfg = PreprocessConfig(
        tile_size=128,
        tile_stride=128,
        outer_edge_extend=16,
        enable_panel_polygon=True,
        image_preprocess_pipeline=[{'method': 'dog'}],
        preprocess_after_tiling=True,
    )
    grid_tiles = _generate_tiles(
        narrow_raw, (32, 32, 384, 140), narrow_poly, pre_cfg, original_img=narrow_raw,
    )
    masked_grid_tiles = [t for t in grid_tiles if t.mask is not None]
    assert masked_grid_tiles
    for tile in masked_grid_tiles:
        unmasked = apply_preprocess_pipeline(tile.original_image, [{'method': 'dog'}])['image']
        assert int(unmasked[tile.mask > 0].max()) == 255
        np.testing.assert_array_equal(tile.image, np.full_like(tile.image, 128))

    # 正式推論 AOI 座標極靠近右邊緣時（保留 32px anchor margin 導致切進黑背景），同樣不爆假亮帶且保留黑霧
    raw = np.full((420, 520), 18, dtype=np.uint8)
    polygon = np.array([[40, 36], [476, 48], [468, 382], [32, 370]], dtype=np.float32)
    yy, xx = np.mgrid[:raw.shape[0], :raw.shape[1]]
    mura = np.exp(-0.5 * (((xx - 452.0) / 14.0) ** 2 + ((yy - 210.0) / 10.0) ** 2))
    panel_mask = np.zeros_like(raw)
    cv2.fillPoly(panel_mask, [polygon.astype(np.int32)], 255)
    raw_with_mura = np.where(panel_mask > 0, np.rint(125.0 - 3.0 * mura), 18).astype(np.uint8)

    cfg = CAPIConfig(
        tile_size=128,
        enable_panel_polygon=True,
        image_preprocess_pipeline=[{'method': 'dog'}],
        preprocess_after_tiling=True,
    )
    inferencer = CAPIInferencer(cfg)
    img_res = ImageResult(
        image_path=tmp_path / 'WGF50500_edge.tif',
        image_size=(raw.shape[1], raw.shape[0]),
        otsu_bounds=(32, 36, 476, 382),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        raw_bounds=(32, 36, 476, 382),
        panel_polygon=polygon,
    )
    defect = AOIReportDefect('PCDB6', 466, 210, 'WGF50500', coordinate_space='image')
    created = inferencer._create_aoi_centered_tiles_v2(
        raw_with_mura, img_res, [defect], (520, 420), pre_cfg,
    )
    assert created == 1
    aoi_tile = img_res.tiles[0]
    assert aoi_tile.mask is not None
    assert int(aoi_tile.image.max()) <= 145  # 不會因黑邊斷崖衝上 255 飽和亮帶
    mx, my = 452 - aoi_tile.x, 210 - aoi_tile.y
    assert int(aoi_tile.image[my, mx]) < 85  # 極淡黑霧清楚強化


def test_ng_sampling_saves_and_replays_crop_mask_sidecar_for_edge_dog(tmp_path):
    from capi_dog import crop_mask_sidecar_path
    from capi_inference import SubmodelScorer
    from capi_preprocess import PreprocessConfig
    from capi_train_new import sample_ng_tiles
    import threading
    from unittest.mock import MagicMock, patch

    raw = np.full((320, 360), 16, dtype=np.uint8)
    polygon = np.array([[28, 24], [330, 34], [322, 292], [22, 282]], dtype=np.float32)
    cv2.fillPoly(raw, [polygon.astype(np.int32)], 122)
    source_img = tmp_path / 'WGF50500_ng.tif'
    assert cv2.imwrite(str(source_img), raw)

    saved_validation = []
    first_pool = []

    class FirstDB:
        def list_training_bomb_validation_samples(self, **kwargs):
            return []

        def list_training_bomb_candidates(self, **kwargs):
            return [{
                'inference_record_id': 7,
                'source_result_id': 11,
                'glass_id': 'TL5B31BAY101',
                'machine_no': 'CAPI07HM-P4',
                'request_time': '2026-11-23 13:22:05',
                'image_name': source_img.name,
                'image_path': str(source_img),
                'image_dir': str(tmp_path),
                'otsu_bounds': '22,24,330,292',
                'resolution_x': 360,
                'resolution_y': 320,
                'client_bomb_info': json.dumps({
                    'image_prefix': 'WGF50500',
                    'defect_type': 'point',
                    'coordinates': [[356, 160]],
                }),
            }]

        def insert_tile_pool(self, job_id, rows):
            first_pool.extend(rows)

        def save_training_bomb_validation_samples(self, rows):
            saved_validation.extend(rows)
            return len(rows)

    pre_cfg = PreprocessConfig(
        tile_size=128,
        enable_panel_polygon=True,
        product_resolution=(360, 320),
        image_preprocess_pipeline=[{'method': 'dog'}],
        preprocess_after_tiling=True,
    )
    val_dir = tmp_path / 'ng_val'
    res1 = sample_ng_tiles(
        job_id='job1', machine_id='GZ0790KA0017S', over_review_root=tmp_path / 'unused',
        db=FirstDB(), thumb_dir=tmp_path / 'thumb1', per_lighting=1, log=lambda m: None,
        lightings=('WGF50500',), ng_validation_base_dir=val_dir, preprocess_cfg=pre_cfg,
    )
    assert res1['sampled'] == 1 and len(saved_validation) == 1
    cached_crop_path = Path(saved_validation[0]['crop_path'])
    sidecar = crop_mask_sidecar_path(cached_crop_path)
    assert sidecar.is_file()
    first_dog_img = cv2.imread(first_pool[0]['source_path'], cv2.IMREAD_GRAYSCALE)
    assert np.all(first_dog_img == 128)

    # 移除原圖，確認第二次重訓只靠快取 crop + .mask.png sidecar 即可重現完全相同的智慧 DoG
    source_img.unlink()
    reused_pool = []

    class ReuseDB:
        def list_training_bomb_validation_samples(self, **kwargs):
            return [dict(saved_validation[0], id=99)]

        def list_training_bomb_candidates(self, **kwargs):
            return []

        def insert_tile_pool(self, job_id, rows):
            reused_pool.extend(rows)

    res2 = sample_ng_tiles(
        job_id='job2', machine_id='GZ0790KA0017S', over_review_root=tmp_path / 'unused',
        db=ReuseDB(), thumb_dir=tmp_path / 'thumb2', per_lighting=1, log=lambda m: None,
        lightings=('WGF50500',), ng_validation_base_dir=val_dir, preprocess_cfg=pre_cfg,
    )
    assert res2['cache_reused'] == 1
    reused_dog_img = cv2.imread(reused_pool[0]['source_path'], cv2.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(reused_dog_img, first_dog_img)

    # 模型驗證器 SubmodelScorer.score_image_path 亦自動套用同一份 sidecar 遮罩
    bundle_dir = tmp_path / 'bundle'
    bundle_dir.mkdir()
    scorer = SubmodelScorer(gpu_lock=threading.Lock(), db=MagicMock(), log_fn=lambda m: None)
    scored_inputs = []
    with patch.object(scorer, '_load_inferencer_for_pt', return_value=MagicMock()), \
         patch.object(scorer, '_score_one_tile', side_effect=lambda img, inf: (scored_inputs.append(img.copy()), 0.4)[1]):
        scorer.score_image_path(
            bundle_dir=bundle_dir, lighting='WGF50500', zone='edge',
            image_path=cached_crop_path, preprocess_pipeline=[{'method': 'dog'}],
        )
    assert len(scored_inputs) == 1
    np.testing.assert_array_equal(scored_inputs[0], first_dog_img)
