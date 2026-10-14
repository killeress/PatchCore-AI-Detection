"""Execute the actual form serializer and validate its API payload."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("separate,edge_mode", [(False, "off"), (True, "off"), (True, "softpatch_plus_v1")])
def test_softpatch_form_payload_passes_api_validation(separate, edge_mode):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to execute the browser form serializer")
    template = (Path(__file__).parents[1] / "templates/train_new/step1_select.html").read_text(encoding="utf-8")
    source = template[template.index("const TRAIN_PARAM_KEYS ="):template.index("function collectTileStride()")]
    values = {
        "tp-batch_size": "", "tp-coreset_ratio": "", "tp-precision": "float16",
        "tp-feature_layers": "layer2_layer3", "tp-feature_pool_kernel_size": "3",
        "tp-feature_cleaning_mode": "softpatch_plus_v1", "tp-feature_cleaning_k": "6",
        "tp-feature_cleaning_scope": "inner_only", "tp-feature_cleaning_keep_ratio": "85",
        "tp-feature_cleaning_center_size": "384",
        "tp-feature_cleaning_inner_mode": "softpatch_plus_v1",
        "tp-feature_cleaning_inner_k": "6", "tp-feature_cleaning_inner_keep_ratio": "85",
        "tp-feature_cleaning_edge_mode": edge_mode,
        "tp-feature_cleaning_edge_k": "30", "tp-feature_cleaning_edge_keep_ratio": "99",
        "sp-discriminator": "lof_gaussian", "sp-soft_weight": "false",
        "sp-context_overlap": "true", "sp-weight_strength": "1.5",
        "sp-projection_dim": "32", "sp-reference_size": "2048",
    }
    for zone in ("inner", "edge"):
        for key in ("discriminator", "soft_weight", "context_overlap", "weight_strength", "projection_dim", "reference_size"):
            values[f"sp-{zone}-{key}"] = values[f"sp-{key}"]
    values["sp-inner-context_overlap"] = "false"
    values["sp-edge-weight_strength"] = "2.0"
    values["sp-edge-projection_dim"] = "64"
    values["sp-edge-reference_size"] = "4096"
    script = r"""
const fs = require('fs'), vm = require('vm');
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
const fields = Object.fromEntries(Object.entries(data.values).map(([id, value]) => [id, {
  value, disabled: id === 'tp-feature_cleaning_center_size' || (data.separate &&
    ['mode','k','scope','keep_ratio'].some(key => id === 'tp-feature_cleaning_' + key)),
  min: id === 'tp-feature_cleaning_keep_ratio' ? '50' : '',
  max: '', classList: {remove() {}, add() {}},
}]));
for (const prefix of ['sp-', 'sp-inner-', 'sp-edge-']) {
  for (const key of ['discriminator','soft_weight','context_overlap','weight_strength','projection_dim','reference_size']) {
    fields[prefix + key].disabled = prefix === 'sp-' ? data.separate : (!data.separate || fields['tp-feature_cleaning_' + (prefix === 'sp-inner-' ? 'inner' : 'edge') + '_mode'].value !== 'softpatch_plus_v1');
  }
for (const [key, min, max] of [['weight_strength',0,4],['projection_dim',8,128],['reference_size',256,8192]]) {
  fields[prefix + key].min = String(min); fields[prefix + key].max = String(max);
}
}
const context = {document: {getElementById: id => fields[id]},
  isPartialTraining: () => false, isSeparateFeatureCleaning: () => data.separate};
process.stdout.write(vm.runInNewContext(data.source + '\nJSON.stringify(collectTrainingParams())', context));
"""
    completed = subprocess.run([node, "-e", script], input=json.dumps({"values": values, "source": source,
                               "separate": separate}), text=True, capture_output=True, check=True)
    payload = json.loads(completed.stdout)
    assert payload["error"] is None
    from capi_web import CAPIWebHandler
    parsed, error = CAPIWebHandler._validate_training_params(payload["params"])
    assert error is None
    if separate:
        assert "feature_cleaning_mode" not in parsed
        assert parsed["feature_cleaning_by_zone"]["inner"]["keep_ratio"] == .85
        assert "softpatch_plus_config" not in parsed
        inner = parsed["feature_cleaning_by_zone"]["inner"]["softpatch_plus_config"]
        assert inner["soft_weight"] is False and inner["weight_strength"] == 1.5
        assert inner["context_overlap"] is False
        edge = parsed["feature_cleaning_by_zone"]["edge"]
        assert edge["mode"] == edge_mode
        if edge_mode == "off":
            assert "softpatch_plus_config" not in edge
        else:
            assert edge["softpatch_plus_config"] == {
                "discriminator": "lof_gaussian", "soft_weight": False, "context_overlap": True,
                "weight_strength": 2.0, "projection_dim": 64, "reference_size": 4096,
            }
    else:
        assert parsed["softpatch_plus_config"]["soft_weight"] is False
        assert parsed["softpatch_plus_config"]["weight_strength"] == 1.5
        assert parsed["feature_cleaning_k"] == 6
        assert parsed["feature_cleaning_keep_ratio"] == .85


@pytest.mark.parametrize("options", [{"context_overlap": "false"}, {"weight_strength": 5},
                                     {"projection_dim": 7}, {"reference_size": 12}, {"unknown": True}])
def test_api_rejects_invalid_zone_softpatch_options(options):
    from capi_web import CAPIWebHandler
    params = {"feature_cleaning_by_zone": {
        "inner": {"mode": "off"},
        "edge": {"mode": "softpatch_plus_v1", "softpatch_plus_config": options},
    }}
    _, error = CAPIWebHandler._validate_training_params(params)
    assert "feature_cleaning_by_zone.edge.softpatch_plus_config" in error
