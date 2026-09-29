import json
import shutil
import subprocess
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader


ROOT = Path(__file__).resolve().parents[1]


def test_settings_submit_preserves_numeric_boolean_and_mode_types(tmp_path):
    if not shutil.which('node'):
        pytest.skip('Node is required for the settings JavaScript check')
    js = Environment(loader=FileSystemLoader(ROOT/'templates')).get_template('_side_white_settings.js').render()
    harness = r'''
const assert = require('node:assert/strict');
let allParams = [], SETTINGS_USER = {can_manage_accounts:true}, currentTab='side-white';
const location={search:''};
const escapeHtml=v=>String(v ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const escapeAttr=escapeHtml;
const parseJsonLoose=(v,d)=>v?JSON.parse(v):d;
const getPlainParamValue=()=> 'true';
let sent;
const fetch=async(url,options)=>{sent=JSON.parse(options.body);return {ok:true,json:async()=>({success:true})}};
const loadHistory=()=>{};
const elements={};
const document={getElementById:id=>elements[id]};
'''
    check = r'''
(async()=>{
 const defaults=savedSideWhiteParams();
 for(const [key] of sideWhiteParamFields) elements['sw-param-'+key]={valueAsNumber:Number(defaults[key])};
 for(const [key] of sideWhiteSwitches) elements['sw-param-'+key]={checked:defaults[key]};
 elements['sw-param-apply_exclusions'].checked=false;
 elements['sw-param-bomb_force_detection_enabled'].checked=true;
 elements['sw-param-dust_mode']={value:'suppress'};
 elements['sw-params-form']={outerHTML:''};
 await saveSideWhiteParams({preventDefault(){},target:{reportValidity:()=>true}});
 assert.equal(sent.new_value.apply_exclusions,false);
 assert.equal(sent.new_value.bomb_force_detection_enabled,true);
 assert.equal(sent.new_value.dust_mode,'suppress');
 assert.equal(typeof sent.new_value.dust_overlap_ratio,'number');
 assert.equal(sideWhiteParamMessage.includes('已儲存'),true);
 const html=renderSideWhitePane();
 assert.equal(html.includes('儲存 Review'),false);
 assert.equal(html.includes('sw-filter-review'),false);
 console.log(JSON.stringify(sent.new_value));
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    file=tmp_path/'settings.cjs'
    file.write_text(harness+js+check,encoding='utf-8')
    result=subprocess.run(['node',str(file)],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['dust_mode']=='suppress'


def test_readonly_result_escapes_evidence_and_hides_legacy_review():
    payload={'algorithm':'test','candidates':[],'reason':'<img src=x onerror=alert(1)>',
             'omit':{'image':'<script>bad</script>','reason':'unavailable'}}
    row={'id':1,'status':'ERROR','candidate_count':0,'payload':payload,'review_note':'old review','review_decision':'unreviewed'}
    html=Environment(loader=FileSystemLoader(ROOT/'templates')).get_template('_side_white_result.html').render(detail={'glass_id':'test','side_white_result':row})
    assert '<script>bad</script>' not in html and '&lt;script&gt;bad&lt;/script&gt;' in html
    assert '<img src=x onerror=alert(1)>' not in html
    assert 'old review' not in html and 'Review' not in html
