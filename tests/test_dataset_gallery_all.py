import json
import re
import shutil
import subprocess
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from capi_dataset_export import write_manifest
from capi_web import CAPIWebHandler


def gallery(root, query=None):
    handler = object.__new__(CAPIWebHandler)
    handler._dataset_export_base_dir = lambda: root
    result = {}
    def render(**kwargs):
        result.update(kwargs)
        return "page"
    handler.jinja_env = SimpleNamespace(get_template=lambda name: SimpleNamespace(render=render))
    handler._send_response = lambda *args: None
    handler._handle_dataset_gallery_page(query or {})
    return result


def seed(root):
    for job, ip, machine, label in [
        ("manual_a", "10.174.1.1", "LINE1", "over_surface_scratch"),
        ("manual_b", "10.174.1.2", "LINE2", "true_ng"),
        ("legacy_root", "", "", "true_ng"),
    ]:
        path = root if job == "legacy_root" else root / job
        write_manifest(path / "manifest.csv", {"same_id": {
            "sample_id": "same_id", "status": "ok", "source_ip": ip, "machine_no": machine,
            "label": label, "prefix": "G0F", "crop_path": "crop.png", "collected_at": "2026-09-09",
        }})


def test_all_batches_default_keep_distinct_sample_identity(tmp_path):
    seed(tmp_path)
    result = gallery(tmp_path)
    assert result["current_job"] == ""
    assert result["total_count"] == 3
    assert len(result["sample_refs"]) == 3
    assert set(result["source_options"]) == {"10.174.1.1", "10.174.1.2", "__legacy__"}
    for item in result["items"]:
        assert parse_qs(urlsplit(item["crop_url"]).query)["job"] == [item["job_id"]]
        assert result["sample_refs"][item["ui_id"]] == {"job": item["job_id"], "sample_id": "same_id"}


def test_source_category_prefix_and_batch_filters(tmp_path):
    seed(tmp_path)
    result = gallery(tmp_path, {"source": ["10.174.1.1"], "label": ["over_surface_scratch"], "prefix": ["G0F"]})
    assert [item["job_id"] for item in result["items"]] == ["manual_a"]
    assert "LINE1" in result["source_options"]["10.174.1.1"]
    assert gallery(tmp_path, {"source": "__legacy__"})["items"][0]["job_id"] == "legacy_root"
    assert gallery(tmp_path, {"job": "manual_b"})["items"][0]["label"] == "true_ng"
    assert gallery(tmp_path, {"prefix": "R0F"})["filtered_count"] == 0


def test_pagination_keeps_source_and_category(tmp_path):
    seed(tmp_path)
    result = gallery(tmp_path, {"limit": "1", "label": "true_ng"})
    assert result["filtered_count"] == 2
    params = parse_qs(urlsplit(result["next_url"]).query)
    assert params["page"] == ["2"] and params["label"] == ["true_ng"]
    second = gallery(tmp_path, params)
    assert result["items"][0]["ui_id"] != second["items"][0]["ui_id"]


def test_browser_actions_use_each_samples_batch(tmp_path):
    seed(tmp_path)
    context = gallery(tmp_path)
    CAPIWebHandler.init_jinja()
    html = CAPIWebHandler.jinja_env.get_template("dataset_gallery.html").render(**context)
    assert '<option value="">全部批次</option>' in html
    assert '<details ><summary' in html
    script = next(text for text in re.findall(r"<script>(.*?)</script>", html, re.S)
                  if "const SAMPLE_REFS" in text)
    refs = context["sample_refs"]
    ids = list(refs)[:2]
    fixture = r'''
const assert = require('node:assert/strict');
const requests = [], removed = [];
global.document = {
  getElementById: () => ({value:'', addEventListener(){}, style:{}, classList:{add(){},remove(){}}}),
  addEventListener(){}, querySelector:()=>null, querySelectorAll:()=>[]
};
global.window = {location:{replace(){}}};
global.CSS = {escape: value=>value};
global.confirm = ()=>true;
global.alert = message=>{throw new Error(message)};
global.fetch = async (url, options)=>{
  const body = JSON.parse(options.body); requests.push({url,body});
  return {ok:true,json:async()=>({ok:true,deleted:body.sample_ids || [],not_found:[],batch_empty:false})};
};
'''
    fixture += script
    fixture += "\nconst ids = " + json.dumps(ids) + ";\n"
    fixture += r'''
removeCardFromDom = id=>removed.push(id);
(async()=>{
  await deleteSample(ids[0]);
  assert.deepEqual(requests[0].body, SAMPLE_REFS[ids[0]]);
  await moveSample(ids[1], 'true_ng');
  assert.deepEqual(requests[1].body, {...SAMPLE_REFS[ids[1]],new_label:'true_ng'});
  requests.length=0;
  ids.forEach(id=>selectedIds.add(id));
  await batchDelete();
  assert.equal(requests.length, 2);
  assert.deepEqual(new Set(requests.map(r=>r.body.job)), new Set(ids.map(id=>SAMPLE_REFS[id].job)));
  assert.ok(requests.every(r=>r.body.sample_ids[0] === 'same_id'));
  assert.equal(selectedIds.size,0);
})().catch(error=>{console.error(error);process.exitCode=1});
'''
    file = tmp_path / "actions.cjs"
    file.write_text(fixture, encoding="utf-8")
    result = subprocess.run([shutil.which("node") or "node", str(file)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
