from pathlib import Path

import pytest


def _register_bundle(db, tmp_path: Path, machine_id: str, name: str) -> int:
    bundle_dir = tmp_path / "model" / name
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "machine_config.yaml").write_text(f"machine_id: {machine_id}\n", encoding="utf-8")
    return db.register_model_bundle({
        "machine_id": machine_id,
        "bundle_path": str(bundle_dir),
        "trained_at": "2026-07-01T10:00:00",
        "panel_count": 1,
        "inner_tile_count": 0,
        "edge_tile_count": 0,
        "ng_tile_count": 0,
        "bundle_size_bytes": 0,
        "job_id": f"job_{machine_id}",
    })


def test_auto_model_switch_rule_crud_and_history(tmp_path):
    from capi_auto_model_switch import DEFAULT_SERIES_PREFIX
    from capi_database import CAPIDatabase

    db = CAPIDatabase(tmp_path / "test.db")
    bundle_id = _register_bundle(db, tmp_path, "GN156HRAAPF0S", "GN156HRAAPF0S-20260701_100000")

    rule = db.upsert_auto_model_switch_rule("GN156HRA", bundle_id, notes="line A")
    assert rule["series_prefix"] == "GN156HRA"
    assert rule["bundle_id"] == bundle_id
    assert rule["notes"] == "line A"

    default_rule = db.upsert_auto_model_switch_rule(DEFAULT_SERIES_PREFIX, bundle_id)
    rules = db.list_auto_model_switch_rules()
    assert [r["series_prefix"] for r in rules[:2]] == [DEFAULT_SERIES_PREFIX, "GN156HRA"]
    assert db.get_default_auto_model_switch_rule()["id"] == default_rule["id"]

    history_id = db.add_auto_model_switch_history({
        "requested_model_id": "GN156HRA9999",
        "series_prefix": "GN156HRA",
        "previous_bundle_id": None,
        "previous_bundle_label": "",
        "target_bundle_id": bundle_id,
        "target_bundle_label": "GN156HRAAPF0S-20260701_100000",
        "action": "switched",
        "status": "success",
        "message": "ok",
    })
    history = db.list_auto_model_switch_history(limit=10)
    assert history[0]["id"] == history_id
    assert history[0]["status"] == "success"


def test_select_target_bundle_match_and_default(tmp_path):
    from capi_auto_model_switch import DEFAULT_SERIES_PREFIX, select_target_bundle
    from capi_database import CAPIDatabase

    db = CAPIDatabase(tmp_path / "test.db")
    mapped_id = _register_bundle(db, tmp_path, "GN156HRAAPF0S", "GN156HRAAPF0S-20260701_100000")
    default_id = _register_bundle(db, tmp_path, "GN140HGAA390S", "GN140HGAA390S-20260701_110000")
    db.upsert_auto_model_switch_rule("GN156HRA", mapped_id)
    db.upsert_auto_model_switch_rule(DEFAULT_SERIES_PREFIX, default_id)

    matched = select_target_bundle(db, "GN156HRA9999")
    assert matched["reason"] == "matched"
    assert matched["series_prefix"] == "GN156HRA"
    assert matched["bundle"]["id"] == mapped_id

    fallback = select_target_bundle(db, "GN999ZZZ0000")
    assert fallback["reason"] == "fallback_default"
    assert fallback["used_default"] is True
    assert fallback["bundle"]["id"] == default_id


def test_series_prefix_validation():
    from capi_auto_model_switch import DEFAULT_SERIES_PREFIX, normalize_series_prefix

    assert normalize_series_prefix("gn156hra") == "GN156HRA"
    assert normalize_series_prefix(DEFAULT_SERIES_PREFIX) == DEFAULT_SERIES_PREFIX
    with pytest.raises(ValueError):
        normalize_series_prefix("GN156")


@pytest.mark.parametrize("requested, expected", [
    (" gn156hraapf0s ", "exact"),
    ("GN156HRAAPF0S-X", "prefix"),
    ("GN156HRA9999", "prefix"),
    ("GN999ZZZ0000", "default"),
    ("   ", None),
])
def test_exact_match_precedes_prefix_and_default(tmp_path, requested, expected):
    from capi_auto_model_switch import DEFAULT_SERIES_PREFIX, select_target_bundle
    from capi_database import CAPIDatabase

    db = CAPIDatabase(tmp_path / "test.db")
    bundles = {name: _register_bundle(db, tmp_path, name, name)
               for name in ("prefix", "exact", "default")}
    db.upsert_auto_model_switch_rule("GN156HRA", bundles["prefix"])
    db.upsert_auto_model_switch_rule("GN156HRAAPF0S", bundles["exact"], match_mode="exact")
    db.upsert_auto_model_switch_rule(DEFAULT_SERIES_PREFIX, bundles["default"])

    result = select_target_bundle(db, requested)
    if expected is None:
        assert result["bundle"] is None
        assert result["reason"] == "not_configured"
    else:
        assert result["bundle"]["id"] == bundles[expected]
        assert result["used_default"] is (expected == "default")
        assert result["series_prefix"] == requested.strip().upper()[:8]
        if expected == "exact":
            assert "完整機種" in result["message"]


def test_match_modes_can_share_value_and_be_edited_independently(tmp_path):
    from capi_auto_model_switch import select_target_bundle
    from capi_database import CAPIDatabase

    db = CAPIDatabase(tmp_path / "test.db")
    prefix_id = _register_bundle(db, tmp_path, "prefix", "prefix")
    exact_id = _register_bundle(db, tmp_path, "exact", "exact")
    prefix = db.upsert_auto_model_switch_rule("GN156HRA", prefix_id)
    exact = db.upsert_auto_model_switch_rule("gn156hra", exact_id, match_mode="exact")
    assert exact["id"] != prefix["id"]
    assert select_target_bundle(db, "GN156HRA")["bundle"]["id"] == exact_id
    assert select_target_bundle(db, "GN156HRA9999")["bundle"]["id"] == prefix_id

    updated = db.upsert_auto_model_switch_rule(
        "GN156HRA", exact_id, notes="updated", match_mode="exact",
    )
    assert updated["id"] == exact["id"]
    assert updated["notes"] == "updated"
    with pytest.raises(ValueError, match="不可重複"):
        db.upsert_auto_model_switch_rule("GN156HRA", exact_id, rule_id=exact["id"])
    assert db.get_auto_model_switch_rule_by_series("GN156HRA", "exact")["id"] == exact["id"]

    changed = db.upsert_auto_model_switch_rule("GN140HGA", exact_id, rule_id=exact["id"])
    assert changed["match_mode"] == "prefix"
    assert select_target_bundle(db, "GN140HGA9999")["bundle"]["id"] == exact_id
    assert db.delete_auto_model_switch_rule(changed["id"])
    assert select_target_bundle(db, "GN140HGA9999")["reason"] == "not_configured"


@pytest.mark.parametrize("value, mode", [
    ("", "exact"), ("   ", "exact"), ("GN156", "prefix"),
    ("GN156HRAAPF0S", "prefix"), ("GN156HRA", "contains"),
])
def test_match_validation_rejects_invalid_rules(value, mode):
    from capi_auto_model_switch import normalize_series_prefix

    with pytest.raises(ValueError):
        normalize_series_prefix(value, mode)


def test_exact_match_validation_and_no_partial_match(tmp_path):
    from capi_auto_model_switch import normalize_series_prefix, select_target_bundle
    from capi_database import CAPIDatabase

    assert normalize_series_prefix(" gn156hraapf0s ", "exact") == "GN156HRAAPF0S"
    db = CAPIDatabase(tmp_path / "test.db")
    bundle_id = _register_bundle(db, tmp_path, "exact", "exact")
    db.upsert_auto_model_switch_rule("GN156HRAAPF0S", bundle_id, match_mode="exact")
    for requested in ("GN156HRA", "GN156HRAAPF0S-X", "GN156HRA9999"):
        assert select_target_bundle(db, requested)["reason"] == "not_configured"


def test_legacy_rules_migrate_without_data_loss(tmp_path):
    import sqlite3

    from capi_auto_model_switch import DEFAULT_SERIES_PREFIX, select_target_bundle
    from capi_database import CAPIDatabase

    db_path = tmp_path / "test.db"
    db = CAPIDatabase(db_path)
    bundle_id = _register_bundle(db, tmp_path, "legacy", "legacy")
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE auto_model_switch_rules")
        conn.execute("""CREATE TABLE auto_model_switch_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            series_prefix TEXT NOT NULL UNIQUE,
            bundle_id INTEGER NOT NULL,
            notes TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now', 'localtime')),
            updated_at TEXT DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (bundle_id) REFERENCES model_registry(id) ON DELETE CASCADE
        )""")
        conn.executemany(
            "INSERT INTO auto_model_switch_rules VALUES (?, ?, ?, ?, ?, ?)",
            [(7, "GN156HRA", bundle_id, "line A", "2026-07-01", "2026-07-02"),
             (8, DEFAULT_SERIES_PREFIX, bundle_id, "fallback", "2026-07-03", "2026-07-04")],
        )

    for _ in range(2):
        db = CAPIDatabase(db_path)
        rule = db.get_auto_model_switch_rule_by_series("GN156HRA")
        assert rule == {"id": 7, "series_prefix": "GN156HRA", "match_mode": "prefix",
                        "bundle_id": bundle_id, "notes": "line A",
                        "created_at": "2026-07-01", "updated_at": "2026-07-02"}
        assert db.get_default_auto_model_switch_rule()["id"] == 8
        assert select_target_bundle(db, "GN156HRA9999")["bundle"]["id"] == bundle_id
        assert select_target_bundle(db, "GN999ZZZ0000")["used_default"] is True
    exact = db.upsert_auto_model_switch_rule("GN156HRA", bundle_id, match_mode="exact")
    assert exact["id"] > 8
    assert len(db.list_auto_model_switch_rules()) == 3
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("payload, expected_status, expected_mode", [
    ({"series_prefix": "gn156hra"}, 200, "prefix"),
    ({"series_prefix": " gn156hraapf0s ", "match_mode": "exact"}, 200, "exact"),
    ({"series_prefix": "GN156HRAAPF0S", "match_mode": "prefix"}, 400, None),
    ({"series_prefix": "GN156HRA", "match_mode": "contains"}, 400, None),
    ({"series_prefix": " ", "match_mode": "exact"}, 400, None),
    ({"is_default": True}, 200, "prefix"),
])
def test_rule_api_validates_and_persists_match_mode(tmp_path, payload, expected_status, expected_mode):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from capi_database import CAPIDatabase
    from capi_web import CAPIWebHandler

    db = CAPIDatabase(tmp_path / "test.db")
    bundle_id = _register_bundle(db, tmp_path, "api", "api")
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler._capi_server_instance = SimpleNamespace(database=db)
    handler._read_json_body = lambda: {**payload, "bundle_id": bundle_id}
    handler._send_json = Mock()
    handler._handle_auto_model_switch_rule_upsert()
    response = handler._send_json.call_args.args[0]
    assert handler._send_json.call_args.kwargs.get("status", 200) == expected_status
    if expected_mode:
        assert response["success"] is True
        assert response["rule"]["match_mode"] == expected_mode
        handler._handle_auto_model_switch_api()
        assert handler._send_json.call_args.args[0]["rules"][0]["match_mode"] == expected_mode
    else:
        assert response["error"]
        assert db.list_auto_model_switch_rules() == []


def test_server_switches_to_exact_bundle_and_records_match(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    from unittest.mock import Mock

    from capi_database import CAPIDatabase
    from capi_server import CAPIServer

    db = CAPIDatabase(tmp_path / "test.db")
    prefix_id = _register_bundle(db, tmp_path, "prefix", "prefix")
    exact_id = _register_bundle(db, tmp_path, "exact", "exact")
    db.upsert_auto_model_switch_rule("GN156HRA", prefix_id)
    db.upsert_auto_model_switch_rule("GN156HRAAPF0S", exact_id, match_mode="exact")
    server = CAPIServer.__new__(CAPIServer)
    server.db = db
    server._model_switch_lock = threading.Lock()
    server._gpu_lock = threading.Lock()
    server.server_config_path = str(tmp_path / "server.yaml")
    server.fallback_config = SimpleNamespace()
    server._is_bundle_loaded = Mock(return_value=False)
    server._build_inferencer_for_bundle = Mock(return_value=(server.fallback_config, object()))
    server._adopt_active_bundle_runtime = Mock()
    activate = Mock()
    monkeypatch.setattr("capi_model_registry.activate_bundle", activate)

    assert server._ensure_auto_model_switch_for_request({"model_id": "GN156HRAAPF0S"}) is server.fallback_config
    assert server._build_inferencer_for_bundle.call_args.args[0]["id"] == exact_id
    assert activate.call_args.args == (db, exact_id)
    history = db.list_auto_model_switch_history()[0]
    assert history["target_bundle_id"] == exact_id
    assert history["status"] == "success"
    assert "GN156HRAAPF0S 命中完整機種模型" in history["message"]


def test_settings_match_controls_render_and_submit_full_names():
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for settings UI behavior test")
    template = (Path(__file__).resolve().parents[1] / "templates/settings.html").read_text(encoding="utf-8")
    functions = "function renderAutoSwitchBundleOptions(" + template.split(
        "    function renderAutoSwitchBundleOptions(", 1,
    )[1].split("    function centralAccountLocationValue(", 1)[0]
    script = """
const assert = require('node:assert/strict');
const autoSwitchRules = [
    {id: 1, series_prefix: 'GN156HRA', bundle_id: 7},
    {id: 2, series_prefix: 'GN156HRAAPF0S', match_mode: 'exact', bundle_id: 8},
];
const autoSwitchBundles = [{id: 7, label: 'prefix'}, {id: 8, label: 'exact'}];
const autoSwitchHistory = [];
const autoSwitchActiveBundleId = 7;
const currentTab = 'auto-model-switch';
const escapeHtml = value => String(value);
const escapeAttr = escapeHtml;
const elements = {};
const document = {getElementById: id => elements[id]};
const payloads = [];
async function fetch(url, options) {
    assert.equal(url, '/api/auto-model-switch/rules/upsert');
    payloads.push(JSON.parse(options.body));
    return {json: async () => ({success: true})};
}
async function loadAutoModelSwitch() {}
function showToast(message, kind) { if (kind === 'error') throw Error(message); }
""" + functions + r"""
const html = renderAutoModelSwitchPane();
assert.match(html, /完整機種名稱 → 前綴（前 8 碼）→ 預設模型/);
assert.match(html.match(/<select id="auto-match-1"[\s\S]*?<\/select>/)[0], /value="prefix" selected/);
assert.match(html.match(/<select id="auto-match-2"[\s\S]*?<\/select>/)[0], /value="exact" selected/);
assert.match(html.match(/<input id="auto-series-1"[^>]*>/)[0], /maxlength="8"/);
const exactInput = html.match(/<input id="auto-series-2"[^>]*>/)[0];
assert.doesNotMatch(exactInput, /maxlength/);
assert.match(exactInput, /value="GN156HRAAPF0S"/);
elements['auto-new-series'] = {
    value: 'GN156HRAAPF0S',
    maxlength: '8',
    setAttribute(name, value) { this[name] = value; },
    removeAttribute(name) { delete this[name]; },
};
updateAutoSwitchMatchInput('auto-new-series', 'exact');
assert.equal(elements['auto-new-series'].maxlength, undefined);
updateAutoSwitchMatchInput('auto-new-series', 'prefix');
assert.equal(elements['auto-new-series'].maxlength, '8');
assert.equal(elements['auto-new-series'].value, 'GN156HRAAPF0S');
updateAutoSwitchMatchInput('auto-new-series', 'exact');
for (const [id, value] of Object.entries({
    'auto-new-match': 'exact', 'auto-new-bundle': '8', 'auto-new-notes': 'new',
    'auto-match-2': 'exact', 'auto-series-2': 'GN156HRAAPF0S', 'auto-bundle-2': '8', 'auto-notes-2': 'edit',
})) elements[id] = {value};
(async () => {
    await addAutoSwitchRule();
    await saveAutoSwitchRule(2);
    assert.equal(payloads.length, 2);
    assert.equal(payloads[1].id, 2);
    for (const payload of payloads) {
        assert.equal(payload.match_mode, 'exact');
        assert.equal(payload.series_prefix, 'GN156HRAAPF0S');
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run([node], input=script, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
