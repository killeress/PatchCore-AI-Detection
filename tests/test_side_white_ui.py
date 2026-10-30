import json
import shutil
import subprocess
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader
from capi_side_report import side_white_view


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
 allParams=[{param_name:'side_white_detection_params',param_value:JSON.stringify({dust_mode:'observe'})}];
 const defaults=savedSideWhiteParams();
 assert.equal(defaults.csjbl_preview_radius_product_px,50);
 assert.equal(defaults.csjbl_match_tolerance_product_px,5);
 assert.equal(defaults.dust_mode,'suppress');
 for(const [key] of sideWhiteParamFields) elements['sw-param-'+key]={valueAsNumber:Number(defaults[key])};
 for(const [key] of sideWhiteSwitches) elements['sw-param-'+key]={checked:defaults[key]};
 elements['sw-param-apply_exclusions'].checked=false;
 elements['sw-param-csjbl_preview_radius_product_px'].valueAsNumber=75;
 elements['sw-param-csjbl_match_tolerance_product_px'].valueAsNumber=8;
 elements['sw-param-bomb_force_detection_enabled'].checked=true;
 elements['sw-params-form']={outerHTML:''};
 await saveSideWhiteParams({preventDefault(){},target:{reportValidity:()=>true}});
 assert.equal(sent.new_value.apply_exclusions,false);
 assert.equal(sent.new_value.csjbl_preview_radius_product_px,75);
 assert.equal(sent.new_value.csjbl_match_tolerance_product_px,8);
 assert.equal(savedSideWhiteParams().csjbl_preview_radius_product_px,75);
 assert.equal(sent.new_value.bomb_force_detection_enabled,true);
 assert.equal(sent.new_value.dust_mode,'suppress');
 assert.equal(typeof sent.new_value.dust_overlap_ratio,'number');
 assert.equal(sideWhiteParamMessage.includes('已儲存'),true);
 const html=renderSideWhitePane();
 assert.equal(html.includes('儲存 Review'),false);
 assert.equal(html.includes('sw-filter-review'),false);
 assert.equal(html.includes('sw-param-dust_mode'),false);
 assert.equal(html.includes('灰塵處理模式'),false);
 assert.equal(html.includes('貼紙／灰塵過濾固定啟用'),true);
 assert.ok(html.includes('href="/ric?tab=side-statistics"'));
 assert.ok(html.includes('href="/search"'));
 assert.ok(html.includes('沒有 CSJBL 的已保存結果'));
 assert.equal(html.includes('sw-filter-'),false);
 assert.equal(html.includes('pageSideWhite('),false);
 location.search='?' + new URLSearchParams({tab:'side-white',glass_id:'G & "<1>',machine_no:'CAPI 1'});
 const linkedHtml=renderSideWhitePane();
 assert.ok(linkedHtml.includes('href="/search?glass_id=G+%26+%22%3C1%3E&amp;machine_no=CAPI+1"'));
 SETTINGS_USER={can_manage_accounts:false};
 assert.equal(renderSideWhitePane(),'');
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
    env = Environment(loader=FileSystemLoader(ROOT/'templates'))
    env.globals['side_white_view'] = side_white_view
    html=env.get_template('_side_white_result.html').render(detail={'glass_id':'test','side_white_result':row})
    assert '<script>bad</script>' not in html and '&lt;script&gt;bad&lt;/script&gt;' in html
    assert '<img src=x onerror=alert(1)>' not in html
    assert 'old review' not in html and 'Review' not in html


def test_record_search_preserves_access_to_saved_results_without_csjbl(tmp_path):
    from html import unescape
    import re
    from urllib.parse import parse_qs, urlsplit

    from capi_database import CAPIDatabase
    from capi_web import CAPIWebHandler

    db = CAPIDatabase(str(tmp_path / 'records.db'))
    glass_id = 'NO-CSJBL & archived'
    record_id = db.save_inference_record(
        glass_id=glass_id, model_id='MODEL', machine_no='CAPI1',
        resolution=(1000, 500), machine_judgment='OK', ai_judgment='OK',
        image_dir='samples', total_images=1, ng_images=0, ng_details='[]',
        aoi_machine_coords='{}', request_time='2020-01-01 12:00:00',
        response_time='2020-01-01 12:00:01', processing_seconds=.1,
    )
    reason = '無 CSJBL 的已保存側拍明細'
    db.save_side_white_result(record_id, {
        'status': 'NO_CANDIDATES', 'algorithm': 'test',
        'candidates': [], 'reason': reason,
    })
    CAPIWebHandler.init_jinja()
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.db = db
    handler.inferencer = None
    handler._capi_server_instance = None
    handler.heatmap_base_dir = str(tmp_path)
    responses = []
    handler._send_response = lambda status, body: responses.append((status, body))

    handler._handle_search({'glass_id': [glass_id]}, '/search')
    assert responses[-1][0] == 200
    assert f'href="/record/{record_id}"' in responses[-1][1]

    handler._handle_record_detail(str(record_id), f'/record/{record_id}')
    status, html = responses[-1]
    assert status == 200
    assert 'id="side-white-result"' in html and reason in html
    assert '側拍結果總表' not in html
    link = re.search(r'href="(/search\?glass_id=[^"]*)"', html).group(1)
    query = parse_qs(urlsplit(unescape(link)).query)
    assert query == {'glass_id': [glass_id]}
    handler._handle_search(query, '/search')
    assert f'href="/record/{record_id}"' in responses[-1][1]
