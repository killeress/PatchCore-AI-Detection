"""Execute the actual wizard JS for DoG settings and before/after review."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from jinja2 import Environment
from capi_image_preprocess_lab import get_method_specs


ROOT = Path(__file__).resolve().parents[1]


def run_node(script):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js needed for wizard behavior checks')
    result = subprocess.run([node,'-'],input=script,text=True,encoding='utf-8',capture_output=True)
    assert result.returncode == 0, result.stderr


def test_preset_and_manual_dog_require_tile_mode_and_validate_axes():
    source = (ROOT/'templates/train_new/step1_select.html').read_text(encoding='utf-8')
    body = source[source.index('function clonePipeline('):source.index('function showDiffMapHelp(')]
    setup = r"""
const assert = require('node:assert/strict');
const elements = new Map();
const element = id => {if (!elements.has(id)) elements.set(id,{value:'',hidden:true,checked:false,disabled:false});return elements.get(id);};
global.document = {getElementById: id => id === 'pipeline-builder' ? null : element(id),querySelectorAll:()=>[]};
let partial = false;
function isPartialTraining(){return partial;}
let _pipelineSeq=0, _activePipelineKey='shared', _zonePipelinesInitialized=false;
const _pipelineStepsByKey={shared:[],inner:[],edge:[]};
const DEFAULT_PREPROCESS_PIPELINE=[];
"""
    setup += 'const PREPROCESS_METHOD_SPECS='+json.dumps(get_method_specs(),ensure_ascii=False)+';\n'
    setup += 'const PREPROCESS_METHOD_MAP=Object.fromEntries(PREPROCESS_METHOD_SPECS.map(s=>[s.id,s]));\n'
    checks = r"""
element('pp-config-mode').value='shared';
useWhiteSpotDogPreset();
assert.equal(pipelineSteps().length,1);
assert.equal(pipelineSteps()[0].method,'dog');
assert.equal(element('pp-after-tiling').checked,true);
assert.equal(element('pp-after-tiling').disabled,true);
assert.equal(element('pp-dog-note').hidden,false);
assert.equal(collectImagePreprocessPipelines().pipeline[0].params.gain,40);
pipelineSteps()[0].params.background_sigma_y=6;
assert.match(collectImagePreprocessPipelines().error,/上下/);
pipelineSteps()[0].params.background_sigma_y=32;
const snapshot=JSON.stringify(_pipelineStepsByKey);
partial=true;
useWhiteSpotDogPreset();
assert.equal(JSON.stringify(_pipelineStepsByKey),snapshot);
partial=false;
element('pp-config-mode').value='separate';
_activePipelineKey='inner';
useWhiteSpotDogPreset();
assert.equal(_pipelineStepsByKey.inner[0].method,'dog');
assert.deepEqual(_pipelineStepsByKey.edge,[]);
const zones=collectImagePreprocessPipelines();
assert.equal(zones.pipelines.inner[0].params.signal_sigma_x,20);
assert.deepEqual(zones.pipelines.edge,[]);
assert.equal(zones.error,null);
"""
    run_node(setup+body+checks)


def test_review_switch_only_changes_display_and_handles_older_tiles():
    source = (ROOT/'templates/train_new/step3_review.html').read_text(encoding='utf-8')
    body = source[source.index('function renderTileImage('):source.index('function modalShowTile(')]
    setup = r"""
const assert=require('node:assert/strict');
const nodes=new Map();
global.document={getElementById(id){if(!nodes.has(id))nodes.set(id,{setAttribute(k,v){this[k]=v;}});return nodes.get(id);}};
let _tileImageMode='processed', _currentLighting='WGF50500', _modalTileId=1;
const tile={id:1,image_url:'/processed.png',original_image_url:'/original.png',source_path:'/model-input.png',decision:'accept'};
const _tilesByLighting={WGF50500:[tile]};
const before=JSON.stringify(tile);
"""
    checks = r"""
renderTileImage(tile);
assert.equal(document.getElementById('modal-img').src,'/processed.png');
setTileImageMode('original');
assert.equal(document.getElementById('modal-img').src,'/original.png');
assert.equal(document.getElementById('modal-original-btn')['aria-pressed'],'true');
assert.equal(JSON.stringify(tile),before);
setTileImageMode('processed');
assert.equal(document.getElementById('modal-img').src,'/processed.png');
tile.original_image_url='';
setTileImageMode('original');
renderTileImage(tile);
assert.equal(document.getElementById('modal-img').src,'/processed.png');
assert.equal(document.getElementById('modal-original-btn').disabled,true);
assert.match(document.getElementById('modal-input-label').textContent,/未保存/);
"""
    run_node(setup+body+checks)


@pytest.mark.parametrize('filename',['step1_select.html','step3_review.html'])
def test_full_rendered_wizard_script_parses(filename):
    source=(ROOT/'templates/train_new'/filename).read_text(encoding='utf-8')
    scripts=re.findall(r'<script>(.*?)</script>',source,re.S)
    rendered=Environment().from_string('\n'.join(scripts)).render(
        training_scope={'mode':'full','selected_units':[]},target_bundle=None,
        selected_lightings=['WGF50500'],lighting_labels={},job_id='test',machine_id='DOG',
        preprocess_methods=get_method_specs(),default_preprocess_pipeline=[],target_patchcore_params={},
    )
    run_node('new Function('+json.dumps(rendered,ensure_ascii=False)+');')
