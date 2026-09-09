import io
import json
import re
import shutil
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from capi_database import CAPIDatabase
from capi_review_rules import apply_rules, exclusion_reason, normalize_rules


@pytest.fixture
def db(tmp_path):
    db = CAPIDatabase(str(tmp_path / "review.db"))
    db.create_training_job("job", "PRODUCT", [], training_scope={
        "mode": "partial", "selected_units": ["W0F00000-inner", "W0F00000-edge"]})
    db.update_training_job_state("job", "review")
    return db


def tile_image(tmp_path, image, **overrides):
    path = tmp_path / "切片.png"
    cv2.imencode('.png', image)[1].tofile(path)
    return {"source_path": str(path), "lighting": "W0F00000", "zone": "inner", "source": "ok",
            "tile_x": 100, "tile_y": 100, "tile_width": 256, "tile_height": 256,
            "review_geometry": {"panel_bbox": [0, 0, 1000, 1000]}, **overrides}


@pytest.mark.parametrize("zone", ["inner", "edge"])
@pytest.mark.parametrize("background", [50, 140, 230])
def test_detects_local_speck_across_brightness(tmp_path, zone, background):
    image = np.full((256, 256), background, np.uint8)
    cv2.circle(image, (128, 128), 4, background - 40, -1)
    tile = tile_image(tmp_path, image, zone=zone)
    before = image.copy()
    evidence = exclusion_reason(tile, normalize_rules({"dark_spots": True}))
    assert evidence["kind"] == "dark_spot" and evidence["area"] >= 8
    assert np.array_equal(cv2.imdecode(np.fromfile(tile['source_path'], np.uint8), 0), before)


@pytest.mark.parametrize("pattern", ["flat", "gradient", "shadow", "edge", "boundary", "weak"])
def test_retains_normal_backgrounds_and_uncertain_shapes(tmp_path, pattern):
    image = np.full((256, 256), 130, np.uint8)
    if pattern == "gradient":
        image[:] = np.linspace(30, 220, 256).astype(np.uint8)
    elif pattern == "shadow":
        image[:, :128] = 70
        image = cv2.GaussianBlur(image, (71, 71), 20)
    elif pattern == "edge":
        image[:, 80:85] = 20
    elif pattern == "boundary":
        cv2.circle(image, (0, 120), 5, 0, -1)
    elif pattern == "weak":
        cv2.circle(image, (120, 120), 4, 120, -1)
    assert exclusion_reason(tile_image(tmp_path, image, zone="edge"), normalize_rules({"dark_spots": True})) is None


def test_roi_translation_scale_overlap_and_no_contact(tmp_path):
    rules = normalize_rules({"regions": [[.1, .1, .1, .1]]})
    tile = tile_image(tmp_path, np.full((256, 256), 130, np.uint8))
    assert exclusion_reason(tile, rules)["kind"] == "region"
    tile.update(review_geometry={"panel_bbox": [500, 300, 2500, 2300]}, tile_x=700, tile_y=500)
    assert exclusion_reason(tile, rules)["kind"] == "region"
    tile["tile_x"] = 900  # touches right boundary, zero overlap
    assert exclusion_reason(tile, rules) is None


@pytest.mark.parametrize("rules", [{"regions": [[0, 0, 2, 1]]}, {"regions": [[float('nan'), 0, .1, .1]]},
    {"contrast": 0}, {"min_area": 100, "max_area": 10}, {"dark_spots": "false"}])
def test_invalid_rules(rules):
    with pytest.raises(ValueError):
        normalize_rules(rules)


def test_apply_persistence_scope_roles_and_manual_restore(db, tmp_path):
    image = np.full((256, 256), 130, np.uint8)
    cv2.circle(image, (100, 100), 4, 10, -1)
    tiles = [tile_image(tmp_path, image, dataset_role=role) for role in ('train', 'calibration', 'acceptance')]
    tiles += [tile_image(tmp_path, image, zone='edge')]
    ids = db.insert_tile_pool('job', tiles)
    rules = normalize_rules({'dark_spots': True})
    job = db.get_training_job('job')
    result = apply_rules(db, job, 'W0F00000', 'inner', rules, save=True)
    assert result == {'excluded': 3, 'read_errors': 0, 'missing_geometry': 0}
    assert db.get_review_rules('PRODUCT', 'W0F00000', 'inner') == rules
    assert db.get_review_rules('PRODUCT', 'W0F00000', 'edge') is None
    assert db.get_review_rules('OTHER', 'W0F00000', 'inner') is None
    assert db.list_tile_pool('job', zone='edge')[0]['decision'] == 'accept'
    db.update_tile_decisions('job', [ids[0]], 'accept')
    assert apply_rules(db, job, 'W0F00000', 'inner', rules)['excluded'] == 0
    assert db.list_tile_pool('job')[0]['decision'] == 'accept'
    with pytest.raises(ValueError, match='範圍'):
        apply_rules(db, job, 'R0F00000', 'inner', rules)


def test_state_change_prevents_commit_and_save(db, tmp_path, monkeypatch):
    tile = tile_image(tmp_path, np.full((256, 256), 130, np.uint8))
    db.insert_tile_pool('job', [tile])
    job = db.get_training_job('job')
    def detect(*args):
        db.update_training_job_state('job', 'training')
        return {'kind': 'region'}
    monkeypatch.setattr('capi_review_rules.exclusion_reason', detect)
    with pytest.raises(ValueError, match='凍結'):
        apply_rules(db, job, 'W0F00000', 'inner', {}, save=True)
    assert db.list_tile_pool('job')[0]['decision'] == 'accept'
    assert db.get_review_rules('PRODUCT', 'W0F00000', 'inner') is None


def test_legacy_geometry_and_unreadable_tiles_reported(db, tmp_path):
    tile = tile_image(tmp_path, np.zeros((256, 256), np.uint8), review_geometry=None)
    tile['source_path'] = str(tmp_path / 'missing.png')
    db.insert_tile_pool('job', [tile])
    result = apply_rules(db, db.get_training_job('job'), 'W0F00000', 'inner',
                         {'regions': [[.1,.1,.1,.1]], 'dark_spots': True})
    assert result == {'excluded': 0, 'read_errors': 1, 'missing_geometry': 1}


def test_api_load_apply_and_frozen(db, tmp_path):
    from capi_web import CAPIWebHandler
    h = CAPIWebHandler.__new__(CAPIWebHandler)
    h._capi_server_instance = SimpleNamespace(database=db)
    responses = []
    h._send_json = lambda data, status=200: responses.append((status, data))
    for action in ['load', 'apply']:
        body = json.dumps({'job_id': 'job', 'lighting': 'W0F00000', 'zone': 'inner', 'action': action, 'rules': {}}).encode()
        h.headers = {'Content-Length': str(len(body))}
        h.rfile = io.BytesIO(body)
        h._handle_train_new_auto_exclude()
        assert responses[-1][0] == 200
    db.update_training_job_state('job', 'training')
    h.rfile = io.BytesIO(body)
    h._handle_train_new_auto_exclude()
    assert responses[-1][0] == 409


def test_next_partial_job_automatically_uses_saved_rules(db, tmp_path):
    from capi_review_rules import apply_saved_rules
    rules = normalize_rules({'regions': [[.1, .1, .1, .1]]})
    for zone in ('inner', 'edge'):
        apply_rules(db, db.get_training_job('job'), 'W0F00000', zone, rules, save=True)
    db.create_training_job('next', 'PRODUCT', [], training_scope={
        'mode': 'partial', 'selected_units': ['W0F00000-edge']})
    tiles = [tile_image(tmp_path, np.zeros((256,256), np.uint8), zone=z) for z in ('inner', 'edge')]
    db.insert_tile_pool('next', tiles)
    logs = []
    apply_saved_rules(db, 'next', logs.append)
    rows = db.list_tile_pool('next')
    assert [t['decision'] for t in rows] == ['accept', 'reject']
    assert len(logs) == 1 and 'W0F00000-edge' in logs[0]


def test_report_distinguishes_rule_ng():
    from capi_training_validation import build_report
    samples = [dict(role=role, label=label, group=role, score=.8 if label == 'ng' else .1,
                    auto_exclusion={'kind': 'region'} if label == 'ng' else None)
               for role in ('calibration', 'acceptance') for label in ('ok', 'ng')]
    report = build_report(samples, {'split_mode': 'auto_panel', 'panels': {'p': {'role': 'acceptance'}}})
    assert report['automatic_ng_count'] == 2


def test_review_template_scripts_parse():
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader('templates'), autoescape=True)
    env.globals['app_version'] = {'version': 'test'}
    html = env.get_template('train_new/step3_review.html').render(job_id='job',
        selected_lightings=['W0F00000'], lighting_labels={}, training_scope={'selected_units': ['W0F00000-inner']})
    assert '儲存並套用到此 PT' in html
    node = shutil.which('node')
    assert node, 'Node required for UI syntax verification'
    for attributes, script in re.findall(r'<script([^>]*)>(.*?)</script>', html, re.S):
        cmd = [node, '--check'] + (['--input-type=module'] if 'type="module"' in attributes else [])
        result = subprocess.run(cmd, input=script, text=True, capture_output=True, encoding='utf-8')
        assert result.returncode == 0, result.stderr


def test_roi_drag_and_apply_ui_flow():
    from pathlib import Path
    script = re.search(r'<script>(.*?)</script>', Path('templates/train_new/_auto_exclude.html').read_text(encoding='utf-8'), re.S)[1]
    harness = r'''
const assert = require('node:assert/strict');
const elements = new Map();
let loaded;
global.document = {
  getElementById(id) { if (!elements.has(id)) elements.set(id, {value:'',disabled:false,textContent:''}); return elements.get(id); },
  addEventListener(event, fn) { loaded = fn; }
};
const canvas = document.getElementById('exclude-canvas');
Object.assign(canvas, {width:1000,height:500,setPointerCapture(){},
  getBoundingClientRect:()=>({left:0,top:0,width:500,height:250}),
  getContext:()=>({clearRect(){},drawImage(){},fillRect(){},strokeRect(){}})});
const jobId = 'job';
let captured, shouldFail = false, rendered = false;
global.fetch = async (url, options) => {
  captured = JSON.parse(options.body);
  if (shouldFail) throw new Error('offline');
  return {ok:true,json:async()=>({excluded:4,read_errors:0,missing_geometry:0})};
};
async function loadLighting() {}
function render() { rendered = true; }
'''
    checks = r'''
(async()=>{
 loaded(); excludeImage = {};
 canvas.onpointerdown({clientX:200,clientY:100,pointerId:1});
 canvas.onpointerup({clientX:100,clientY:50});
 assert.deepEqual(excludeRules.regions, [[.2,.2,.2,.2]]);
 document.getElementById('exclude-unit').value='W0F00000-edge';
 document.getElementById('exclude-spots').checked=true;
 document.getElementById('exclude-contrast').value='30';
 document.getElementById('exclude-min-area').value='8';
 document.getElementById('exclude-max-area').value='800';
 await applyAutoExclude();
 assert.equal(captured.zone,'edge'); assert.equal(captured.rules.contrast,30);
 assert.deepEqual(captured.rules.regions,[[.2,.2,.2,.2]]);
 assert.equal(rendered,true); assert.match(document.getElementById('exclude-status').textContent,/4/);
 shouldFail=true; await applyAutoExclude();
 assert.match(document.getElementById('exclude-status').textContent,/offline/);
 assert.equal(document.getElementById('start-training-btn').disabled,false);
 assert.equal(document.getElementById('auto-exclude-controls').disabled,false);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    result = subprocess.run([shutil.which('node')], input=harness + script + checks,
                            text=True, capture_output=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
