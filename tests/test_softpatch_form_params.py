"""Execute the actual form serializer and validate its API payload."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("separate", [False, True])
def test_softpatch_form_payload_passes_api_validation(separate):
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
        "tp-feature_cleaning_edge_mode": "off",
        "tp-feature_cleaning_edge_k": "30", "tp-feature_cleaning_edge_keep_ratio": "99",
        "sp-discriminator": "lof_gaussian", "sp-soft_weight": "false",
        "sp-context_overlap": "true", "sp-weight_strength": "1.5",
        "sp-projection_dim": "32", "sp-reference_size": "2048",
    }
    script = r"""
const fs = require('fs'), vm = require('vm');
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
const fields = Object.fromEntries(Object.entries(data.values).map(([id, value]) => [id, {
  value, disabled: id === 'tp-feature_cleaning_center_size' || (data.separate &&
    ['mode','k','scope','keep_ratio'].some(key => id === 'tp-feature_cleaning_' + key)),
  min: id === 'tp-feature_cleaning_keep_ratio' ? '50' : '',
  max: '', classList: {remove() {}, add() {}},
}]));
for (const [key, min, max] of [['weight_strength',0,4],['projection_dim',8,128],['reference_size',256,8192]]) {
  fields['sp-' + key].min = String(min); fields['sp-' + key].max = String(max);
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
    assert parsed["softpatch_plus_config"]["soft_weight"] is False
    assert parsed["softpatch_plus_config"]["weight_strength"] == 1.5
    if separate:
        assert "feature_cleaning_mode" not in parsed
        assert parsed["feature_cleaning_by_zone"]["inner"]["keep_ratio"] == .85
        assert parsed["feature_cleaning_by_zone"]["edge"]["mode"] == "off"
    else:
        assert parsed["feature_cleaning_k"] == 6
        assert parsed["feature_cleaning_keep_ratio"] == .85
