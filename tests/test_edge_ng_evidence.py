import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from capi_database import CAPIDatabase
from capi_tile_diagnostics import decorate_edge_ng_evidence, edge_ng_evidence
from capi_tile_diagnostics import decision_evidence, tile_decision_context


LOG = (
    '🔴 W0F00000_082951.tif Tile@(5141,3287) [edge] '
    'Score:0.742 -> PER_REGION: 0real+1dust -> TWO_STAGE: 0real+0dust -> DUST '
    'EDGE_LIGHT_LEAK_RESCUE: DARK_DROP RIGHT delta=7.84 length=101 dust=0.000 -> REAL_NG'
)
OUTCOME = dict(detected=True, rescue_applied=True, side='right', anomaly_type='BRIGHT_LEAK',
               max_delta=9.6, continuous_length=48, dust_overlap=0.02,
               threshold=4.0, min_length=30, max_dust_overlap=0.2)


def tile(**overrides):
    data = dict(id=101, tile_id=1, x=5141, y=3287, width=512, height=512,
                is_anomaly=1, is_aoi_coord=1, is_dust=0, is_bomb=0,
                score=0.742, aoi_defect_code='C1111', aoi_product_x=1846,
                aoi_product_y=1092, aoi_image_x=5461, aoi_image_y=3543,
                heatmap_path='/heatmaps/tile1.png',
                dust_detail_text='PER_REGION: 0real+1dust -> TWO_STAGE: 0real+0dust -> DUST',
                edge_light_leak_result=json.dumps(OUTCOME))
    return dict(data, **overrides)


def record(tiles=None, **overrides):
    data = dict(id=1, glass_id='TEST', ai_judgment='NG', processing_seconds=1.0,
                image_prefix_labels={}, inference_log=LOG,
                images=[dict(id=1, image_name='W0F00000_082951.tif',
                             image_path='W0F00000_082951.tif', tiles=tiles or [tile()],
                             is_ng=True, max_score=.742, inference_time_ms=40.0)])
    return dict(data, **overrides)


def save(db, tiles, log=LOG):
    return db.save_inference_record(
        glass_id='TEST', model_id='MODEL', machine_no='CAPI03', resolution=(1920,1200),
        machine_judgment='NG', ai_judgment='NG', image_dir='/images', total_images=1,
        ng_images=1, ng_details='', request_time='2026-09-18 08:29:00',
        response_time='2026-09-18 08:29:01', processing_seconds=1.0,
        image_results_data=record(tiles)['images'], inference_log=log,
    )


def test_structured_evidence_wins_over_historical_log():
    detail = record()
    decorate_edge_ng_evidence(detail)
    evidence = detail['images'][0]['tiles'][0]['edge_ng_evidence']
    assert evidence['title'] == '右側泛白'
    assert evidence['threshold'] == 4.0
    assert evidence['source'] == 'stored'
    assert not evidence['historical']
    assert detail['images'][0]['has_edge_ng']


@pytest.mark.parametrize('flag', ['is_dust','is_bomb','scratch_filtered','is_exclude_zone'])
def test_later_filters_do_not_show_final_edge_ng(flag):
    assert edge_ng_evidence(tile(**{flag:1}), LOG) is None


def test_debug_hit_or_below_threshold_is_not_a_rescued_ng():
    assert edge_ng_evidence(tile(is_anomaly=0), LOG) is None
    assert edge_ng_evidence(tile(edge_light_leak_result=json.dumps(dict(OUTCOME, rescue_applied=False))), LOG) is None
    assert edge_ng_evidence(tile(edge_light_leak_result=json.dumps(dict(OUTCOME, detected=False))), LOG) is None


def test_old_log_is_matched_by_image_and_origin_without_inventing_thresholds():
    detail = record([tile(edge_light_leak_result='', dust_detail_text='')])
    decorate_edge_ng_evidence(detail)
    evidence = detail['images'][0]['tiles'][0]['edge_ng_evidence']
    assert evidence['title'] == '右側暗段'
    assert evidence['source'] == 'log'
    assert evidence['max_delta'] == 7.84
    assert evidence['continuous_length'] == 101
    assert evidence['threshold'] is None
    assert evidence['historical'] and evidence['two_stage']


@pytest.mark.parametrize('log', ['', LOG.replace('W0F00000', 'B0F00000'),
                                  LOG.replace('5141,3287','5142,3287'), LOG+'\n'+LOG])
def test_missing_mismatched_or_ambiguous_history_does_not_infer(log):
    detail = record([tile(edge_light_leak_result='', dust_detail_text='')], inference_log=log)
    decorate_edge_ng_evidence(detail)
    assert detail['images'][0]['tiles'][0]['edge_ng_evidence'] is None


def test_duplicate_tile_origins_do_not_share_historical_evidence():
    detail = record([tile(edge_light_leak_result='', dust_detail_text=''),
                     tile(id=102, tile_id=2, edge_light_leak_result='', dust_detail_text='')])
    decorate_edge_ng_evidence(detail)
    assert not any(t['edge_ng_evidence'] for t in detail['images'][0]['tiles'])


def test_migration_and_save_rerun_round_trip(tmp_path):
    path = tmp_path / 'evidence.db'
    db = CAPIDatabase(str(path))
    old_id = save(db, [tile(edge_light_leak_result='', dust_detail_text='')])
    with sqlite3.connect(path) as connection:
        connection.execute('ALTER TABLE tile_results DROP COLUMN dust_detail_text')
        connection.execute('ALTER TABLE tile_results DROP COLUMN edge_light_leak_result')
    db = CAPIDatabase(str(path))
    assert db.get_record_detail(old_id)['images'][0]['tiles'][0]['edge_ng_evidence']['historical']
    record_id = save(db, [tile()])
    stored = db.get_record_detail(record_id)['images'][0]['tiles'][0]
    assert json.loads(stored['edge_light_leak_result']) == OUTCOME
    assert stored['edge_ng_evidence']['threshold'] == 4.0
    db.update_record_for_rerun(record_id, ai_judgment='NG', total_images=1, ng_images=1,
                              ng_details='', processing_seconds=1,
                              image_results_data=record([tile(edge_light_leak_result=json.dumps(dict(OUTCOME, threshold=6)))])['images'])
    assert db.get_record_detail(record_id)['images'][0]['tiles'][0]['edge_ng_evidence']['threshold'] == 6
    db.update_record_for_rerun(record_id, ai_judgment='OK', total_images=1, ng_images=0,
                              ng_details='', processing_seconds=1,
                              image_results_data=record([tile(is_dust=1, edge_light_leak_result='')])['images'])
    assert db.get_record_detail(record_id)['images'][0]['tiles'][0]['edge_ng_evidence'] is None


@pytest.mark.parametrize('template', ['record_detail.html','record_detail_v3.html'])
@pytest.mark.parametrize('heatmap', ['/heatmaps/tile1.png', ''])
def test_record_templates_have_one_collapsed_evidence_panel_even_without_image(template, heatmap):
    from capi_web import CAPIWebHandler
    detail = record([tile(heatmap_path=heatmap)])
    decorate_edge_ng_evidence(detail)
    CAPIWebHandler.init_jinja()
    html = CAPIWebHandler.jinja_env.get_template(template).render(detail=detail, heatmap_base_dir='/heatmaps')
    assert html.count('id="edge-evidence-101"') == 1
    assert 'class="edge-ng-evidence" hidden' in html
    assert html.count('data-edge-evidence="edge-evidence-101"') == 2
    assert '右側泛白' in html and '≥ 4.00' in html
    assert '/static/js/edge-ng-evidence.js' in html
    assert '黑邊規則現已移除' not in html


def test_serializer_preserves_rule_result_and_detail_text():
    from capi_edge_cv import EdgeDefect
    from capi_inference import ImageResult, TileInfo
    from capi_server import results_to_db_data
    t = TileInfo(tile_id=1, x=0, y=0, width=16, height=16,
                 image=np.zeros((16,16), dtype=np.uint8), is_aoi_coord_tile=True)
    t.edge_light_leak_result = OUTCOME.copy()
    t.score_threshold = .35
    t.dust_detail_text = 'PER_REGION: 0real+1dust -> DUST EDGE_LIGHT_LEAK_RESCUE: BRIGHT_LEAK RIGHT delta=9.60 length=48 dust=0.020 -> REAL_NG'
    result = ImageResult(image_path=Path('W0F00000.tif'), image_size=(16,16),
                         otsu_bounds=(0,0,16,16), exclusion_regions=[], tiles=[t],
                         excluded_tile_count=0, processed_tile_count=1, processing_time=0,
                         anomaly_tiles=[(t,.8,np.ones((16,16), dtype=np.float32))])
    result.edge_defects = [EdgeDefect(side='right', area=20, bbox=(0,0,4,5), center=(2,2),
                          inspector_mode='patchcore', patchcore_score=.8, patchcore_threshold=.35,
                          dust_detail_text='OMIT: REAL')]
    image = results_to_db_data([result], {})[0]
    stored = image['tiles'][0]
    assert json.loads(stored['edge_light_leak_result']) == OUTCOME
    assert stored['dust_detail_text'] == t.dust_detail_text
    assert json.loads(stored['decision_context'])['score_threshold'] == .35
    assert image['edge_defects'][0]['dust_detail_text'] == 'OMIT: REAL'
    assert json.loads(image['edge_defects'][0]['decision_context'])['version'] == 1


def test_heatmap_puts_final_edge_reason_before_long_dust_detail(tmp_path, monkeypatch):
    import cv2
    from capi_heatmap import HeatmapManager
    captured = []
    original = cv2.putText
    def capture(img, text, *args, **kwargs):
        captured.append(text)
        return original(img, text, *args, **kwargs)
    monkeypatch.setattr(cv2, 'putText', capture)
    t = SimpleNamespace(is_bright_spot_detection=False, omit_crop_image=None,
                        is_suspected_dust_or_scratch=False, is_bomb=False,
                        dust_detail_text='OMIT ' * 50 + 'PER_REGION: 0real+1dust -> DUST',
                        edge_light_leak_result=OUTCOME.copy())
    HeatmapManager(str(tmp_path)).save_tile_heatmap(
        tmp_path,'test',1,np.full((512,512,3),128,dtype=np.uint8),
        np.ones((512,512),dtype=np.float32),.8,tile_info=t,score_threshold=.35)
    assert any('NG (Edge: BRIGHT_LEAK RIGHT)' in line for line in captured)
    assert any(line.startswith('DUST -> EDGE_LIGHT_LEAK_RESCUE') for line in captured)


def cv_result(**overrides):
    return dict(dict(id=101, side='aoi_edge', area=80, max_diff=20,
                     bbox_x=0, bbox_y=0, bbox_w=512, bbox_h=512,
                     is_cv_ok=0, is_dust=0, is_bomb=0, inspector_mode='cv',
                     heatmap_path='/heatmaps/tile1.png'), **overrides)


def test_actual_tile_threshold_is_saved_and_old_threshold_is_unknown():
    context = tile_decision_context(SimpleNamespace(score_threshold=.35))
    evidence = decision_evidence(tile(edge_light_leak_result='', decision_context=json.dumps(context)))
    assert evidence['status'] == 'NG'
    assert evidence['metrics'][0]['threshold'] == '≥ 0.3500'
    assert decision_evidence(tile())['metrics'][0]['threshold'] == '未記錄'


@pytest.mark.parametrize('flag', ['is_dust', 'is_bomb', 'scratch_filtered', 'is_exclude_zone'])
def test_filtered_items_have_ok_evidence_and_do_not_add_ng_badges(flag):
    detail = record([tile(**{flag: 1})])
    decorate_edge_ng_evidence(detail)
    evidence = detail['images'][0]['tiles'][0]['decision_evidence']
    assert evidence['status'] == 'OK' and evidence['filtered']
    assert '最終 OK' in evidence['flow']
    assert detail['images'][0]['ng_reason_labels'] == []


@pytest.mark.parametrize('source, expected', [('cv', 'CV 邊緣'), ('patchcore', '邊緣 PatchCore'), ('', '混合邊緣檢查')])
def test_fusion_uses_actual_source_and_pc_does_not_use_area_threshold(source, expected):
    evidence = decision_evidence(cv_result(inspector_mode='fusion', source_inspector=source,
                                patchcore_score=.8, patchcore_threshold=.3, min_area_used=999), is_cv=True)
    assert expected in evidence['rule_label']
    if source == 'patchcore':
        assert len(evidence['metrics']) == 1
        assert evidence['metrics'][0]['threshold'] == '≥ 0.3000'


def test_cv_thin_line_evidence_does_not_apply_component_area_threshold():
    evidence = decision_evidence(cv_result(decision_context={'rules': [dict(kind='thin_line',
                                length=90, width=2, min_length=80, max_width=3)]},
                                area=180, min_area_used=999), is_cv=True)
    assert [m['threshold'] for m in evidence['metrics']] == ['≥ 80 px', '≤ 3 px']
    assert not any('面積' in m['label'] for m in evidence['metrics'])


@pytest.mark.parametrize('orientation', ['horizontal', 'vertical'])
def test_cv_detector_records_real_line_branch(orientation):
    from capi_edge_cv import EdgeInspectionConfig, CVEdgeInspector
    config = EdgeInspectionConfig()
    config.aoi_line_min_length = 30
    config.aoi_line_max_width = 3
    inspector = CVEdgeInspector(config)
    mask = np.zeros((80, 80), np.uint8)
    if orientation == 'horizontal':
        mask[40, 10:70] = 255
    else:
        mask[10:70, 40] = 255
    defects, _ = inspector._detect_thin_lines(mask, mask, 0, 0)
    assert len(defects) == 1
    rule = defects[0].decision_context['rules'][0]
    assert rule == dict(kind='thin_line', length=60, width=1, min_length=30, max_width=3)


def test_cv_detector_records_actual_component_thresholds():
    from capi_edge_cv import EdgeInspectionConfig, EdgeSideConfig, CVEdgeInspector
    config = EdgeInspectionConfig(blur_kernel=1, median_kernel=15)
    inspector = CVEdgeInspector(config)
    roi = np.full((80, 80), 100, np.uint8)
    roi[35:40, 35:40] = 200
    defects, _ = inspector._inspect_side(roi, 'right', EdgeSideConfig(threshold=10, min_area=20), 0, 0)
    assert len(defects) == 1
    rule = defects[0].decision_context['rules'][0]
    assert rule['kind'] == 'component' and rule['area'] == 25
    assert rule['min_area'] == 20 and rule['threshold'] == 10
    evidence = decision_evidence(cv_result(decision_context=defects[0].decision_context), is_cv=True)
    assert evidence['metrics'][0]['threshold'] == '≥ 20 px'


def test_bright_spot_captures_absolute_path_when_diff_does_not_hit():
    from capi_config import CAPIConfig
    from capi_inference import CAPIInferencer, TileInfo
    worker = object.__new__(CAPIInferencer)
    worker.config = CAPIConfig()
    worker.config.bright_spot_diff_threshold = 255
    worker.config.bright_spot_threshold = 200
    worker.config.bright_spot_min_area = 4
    worker.config.bright_spot_median_kernel = 3
    worker.config.tile_corner_exclusion_enabled = False
    t = TileInfo(tile_id=1, x=0, y=0, width=32, height=32,
                 image=np.full((32, 32), 220, np.uint8))
    score, _ = worker._detect_bright_spots(t)
    assert score == 1
    context = tile_decision_context(t)
    assert context['max_diff'] == 0 and context['abs_threshold'] == 200
    assert context['max_component_area'] == 1024
    evidence = decision_evidence(tile(decision_context=context, edge_light_leak_result=''))
    assert evidence['rule_label'] == '亮點偵測'
    assert '或絕對亮度' in evidence['notes'][0]


def test_new_context_columns_migrate_save_and_rerun(tmp_path):
    path = tmp_path / 'general.db'
    db = CAPIDatabase(str(path))
    old_id = save(db, [tile(edge_light_leak_result='', dust_detail_text='')], log='')
    with sqlite3.connect(path) as connection:
        connection.execute('ALTER TABLE tile_results DROP COLUMN decision_context')
        connection.execute('ALTER TABLE edge_defect_results DROP COLUMN decision_context')
        connection.execute('ALTER TABLE edge_defect_results DROP COLUMN dust_detail_text')
    db = CAPIDatabase(str(path))
    assert db.get_record_detail(old_id)['images'][0]['tiles'][0]['decision_evidence']['metrics'][0]['threshold'] == '未記錄'
    for threshold in (.35, .5):
        detail = record([tile(edge_light_leak_result='', decision_context=json.dumps({'score_threshold': threshold}))])
        detail['images'][0]['edge_defects'] = [cv_result(
            inspector_mode='patchcore', patchcore_score=.7, patchcore_threshold=threshold,
            decision_context=json.dumps({'test': threshold}), dust_detail_text='OMIT: REAL')]
        if threshold == .35:
            new_id = db.save_inference_record(
                glass_id='TEST', model_id='MODEL', machine_no='CAPI03', resolution=(1920,1200),
                machine_judgment='NG', ai_judgment='NG', image_dir='/images', total_images=1,
                ng_images=1, ng_details='', request_time='', response_time='', processing_seconds=1,
                image_results_data=detail['images'])
        else:
            db.update_record_for_rerun(new_id, ai_judgment='NG', total_images=1, ng_images=1,
                                      ng_details='', processing_seconds=1, image_results_data=detail['images'])
        stored = db.get_record_detail(new_id)['images'][0]
        assert json.loads(stored['tiles'][0]['decision_context'])['score_threshold'] == threshold
        edge = stored['edge_defects'][0]
        assert json.loads(edge['decision_context']) == {'test': threshold}
        assert edge['decision_evidence']['detail'] == 'OMIT: REAL'
        assert edge['decision_evidence']['metrics'][0]['threshold'] == f'≥ {threshold:.4f}'


@pytest.mark.parametrize('template', ['record_detail.html', 'record_detail_v3.html'])
@pytest.mark.parametrize('heatmap', ['/heatmaps/tile1.png', ''])
@pytest.mark.parametrize('aoi', [0, 1])
def test_general_tile_and_cv_panels_are_unique_and_collapsed(template, heatmap, aoi):
    from capi_web import CAPIWebHandler
    detail = record([tile(is_aoi_coord=aoi, heatmap_path=heatmap, edge_light_leak_result='',
                          decision_context={'score_threshold': .35}, dust_detail_text='')], inference_log='')
    detail['images'][0]['edge_defects'] = [cv_result(heatmap_path=heatmap)]
    decorate_edge_ng_evidence(detail)
    CAPIWebHandler.init_jinja()
    html = CAPIWebHandler.jinja_env.get_template(template).render(detail=detail, heatmap_base_dir='/heatmaps')
    for prefix in ['edge-evidence', 'cv-evidence']:
        assert html.count(f'id="{prefix}-101"') == 1
        assert html.count(f'data-edge-evidence="{prefix}-101"') == 2
    assert html.count('class="edge-ng-evidence" hidden') == 2
    assert '≥ 0.3500' in html
