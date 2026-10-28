import io
import json
import shutil
import sqlite3
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from capi_database import CAPIDatabase
from capi_retention import (
    RETENTION_PARAM, initial_retention_policy, normalize_retention_policy,
    next_cleanup_time,
)
from capi_server import CAPIServer
from capi_web import CAPIWebHandler


LEGACY_CONFIG = {"cleanup": {
    "enabled": True, "schedule_time": "02:00", "ok_retain_days": 30,
    "ng_retain_days": 90, "tile_retain_days": 15,
    "heatmap_retain_days": 90, "vacuum_after_cleanup": False,
}}


def make_db(tmp_path, config=LEGACY_CONFIG):
    db = CAPIDatabase(tmp_path / "retention.db")
    db.initialize_retention_policy(config)
    return db


def saved_policy(db):
    return db.get_config_param(RETENTION_PARAM)["decoded_value"]


def insert_record(db, judgment, age, heatmap_dir="", with_review=False):
    created = (datetime.now() - timedelta(days=age)).strftime("%Y-%m-%d 00:00:00")
    with sqlite3.connect(str(db.db_path)) as conn:
        record_id = conn.execute(
            "INSERT INTO inference_records "
            "(glass_id, model_id, machine_no, machine_judgment, ai_judgment, request_time, created_at, heatmap_dir) "
            "VALUES (?, 'MODEL', 'CAPI07', 'NG', ?, ?, ?, ?)",
            (f"{judgment}-{age}", judgment, created, created, str(heatmap_dir)),
        ).lastrowid
        image_id = conn.execute(
            "INSERT INTO image_results (record_id, image_path, image_name) VALUES (?, 'source.png', 'source.png')",
            (record_id,),
        ).lastrowid
        tile_id = conn.execute("INSERT INTO tile_results (image_result_id) VALUES (?)", (image_id,)).lastrowid
        if with_review:
            conn.execute("INSERT INTO scratch_rescue_review (tile_result_id, is_misrescue) VALUES (?, 1)", (tile_id,))
    return record_id


def remaining_ids(db):
    with sqlite3.connect(str(db.db_path)) as conn:
        return {row[0] for row in conn.execute("SELECT id FROM inference_records")}


def test_migration_extends_records_only_and_preserves_later_user_settings(tmp_path):
    config = {"cleanup": {**LEGACY_CONFIG["cleanup"], "enabled": False,
                          "tile_retain_days": 11, "heatmap_retain_days": 76, "schedule_time": "04:25"}}
    db = make_db(tmp_path, config)
    assert saved_policy(db) == {**config["cleanup"], "ok_retain_days": 180, "ng_retain_days": 180}
    edited = {**saved_policy(db), "ok_retain_days": 240, "ng_retain_days": 270, "heatmap_retain_days": 0}
    db.update_config_param(RETENTION_PARAM, edited, "保留較久", changed_by="operator")
    reopened = CAPIDatabase(db.db_path)
    reopened.initialize_retention_policy(LEGACY_CONFIG)
    assert saved_policy(reopened) == edited
    history = reopened.get_config_change_history(param_name=RETENTION_PARAM)
    assert len(history) == 2
    assert {entry["changed_by"] for entry in history} == {"operator", "system"}
    assert config["cleanup"]["ok_retain_days"] == 30


def test_omitted_legacy_tile_and_image_policy_keeps_previous_defaults():
    policy = initial_retention_policy({})
    assert policy["tile_retain_days"] == 7
    assert policy["heatmap_retain_days"] == 0
    assert policy["enabled"] is False
    assert policy["ok_retain_days"] == policy["ng_retain_days"] == 180


def test_plan_a_keeps_searchable_records_but_expires_tiles_and_images(tmp_path):
    db = make_db(tmp_path)
    heatmap_dir = tmp_path / "heatmaps" / "old-panel"
    heatmap_dir.mkdir(parents=True)
    (heatmap_dir / "overview.png").write_bytes(b"old image")
    retained = {insert_record(db, judgment, age)
                for judgment in ("OK", "OK-i", "NG", "ERR:HY") for age in (179, 180)}
    for judgment in ("OK", "OK-i", "NG", "ERR:HY"):
        insert_record(db, judgment, 181)
    image_record = insert_record(db, "OK", 91, heatmap_dir=heatmap_dir)
    retained.add(image_record)
    young_record = insert_record(db, "OK-i", 14)
    retained.add(young_record)
    policy = saved_policy(db)
    stats = db.cleanup_old_records(**{k: v for k, v in policy.items()
                                      if k not in ("enabled", "schedule_time", "vacuum_after_cleanup")}, vacuum=False)
    assert stats["inference_records_deleted"] == 4
    assert remaining_ids(db) == retained
    assert not heatmap_dir.exists()
    rows, count = db.search_records(glass_id="OK-91", cross_filter="ng_ok")
    assert count == 1 and rows[0]["id"] == image_record
    with sqlite3.connect(str(db.db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tile_results").fetchone()[0] == 1
        assert conn.execute("SELECT heatmap_dir FROM inference_records WHERE id = ?", (image_record,)).fetchone()[0] == ""
        assert conn.execute("SELECT COUNT(*) FROM inference_statistics_history").fetchone()[0] == len(retained) + 4


@pytest.mark.parametrize("ok_days,ng_days,age,ok_survives", [(180, 90, 120, True), (30, 180, 31, False)])
def test_ok_i_uses_only_ok_retention(tmp_path, ok_days, ng_days, age, ok_survives):
    db = make_db(tmp_path)
    ok_ids = {insert_record(db, "OK", age), insert_record(db, "OK-i", age)}
    ng_id = insert_record(db, "NG", age)
    db.cleanup_old_records(ok_days, ng_days, vacuum=False)
    assert remaining_ids(db) == (ok_ids if ok_survives else {ng_id})


def test_shorter_record_retention_removes_reviews_before_parent_cascade(tmp_path):
    db = make_db(tmp_path)
    insert_record(db, "OK-i", 40, with_review=True)
    stats = db.cleanup_old_records(30, 180, 180, vacuum=False)
    assert stats["scratch_rescue_review_deleted"] == 1
    assert stats["inference_records_deleted"] == 1
    assert remaining_ids(db) == set()


def test_failed_image_delete_keeps_review_with_its_retry_parent(tmp_path, monkeypatch):
    db = make_db(tmp_path)
    folder = tmp_path / "heatmaps" / "old"
    folder.mkdir(parents=True)
    record_id = insert_record(db, "OK-i", 100, heatmap_dir=folder, with_review=True)
    monkeypatch.setattr(shutil, "rmtree", lambda path: (_ for _ in ()).throw(PermissionError("locked")))
    stats = db.cleanup_old_records(30, 180, 180, vacuum=False, heatmap_retain_days=90)
    assert stats["heatmap_dirs_failed"] == 1
    assert stats["scratch_rescue_review_deleted"] == 0
    assert remaining_ids(db) == {record_id}


def settings_handler(db):
    handler = object.__new__(CAPIWebHandler)
    handler.db = db
    handler.inferencer = None
    handler._capi_server_instance = SimpleNamespace(server_config=LEGACY_CONFIG)
    handler._current_settings_user = lambda: {"username": "operator", "can_manage_accounts": False}
    responses = []
    handler._send_json = lambda payload, status=200: responses.append((status, payload))
    return handler, responses


def post_policy(handler, policy, reason="依用戶需求調整"):
    body = json.dumps({"param_name": RETENTION_PARAM, "new_value": policy, "reason": reason}).encode()
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler._handle_api_settings_update()


def test_settings_get_save_and_restart_use_same_policy_and_audit_actor(tmp_path):
    db = make_db(tmp_path)
    handler, responses = settings_handler(db)
    handler._handle_api_settings()
    exposed = next(p for p in responses[-1][1]["params"] if p["param_name"] == RETENTION_PARAM)
    assert exposed["decoded_value"] == saved_policy(db)
    policy = {**saved_policy(db), "schedule_time": "03:15", "enabled": False, "ok_retain_days": 240}
    post_policy(handler, policy)
    assert responses[-1][0] == 200 and responses[-1][1]["policy"] == policy
    reopened = CAPIDatabase(db.db_path)
    reopened.initialize_retention_policy(LEGACY_CONFIG)
    handler.db = reopened
    handler._handle_api_settings()
    exposed = next(p for p in responses[-1][1]["params"] if p["param_name"] == RETENTION_PARAM)
    assert exposed["decoded_value"] == policy
    history = db.get_config_change_history(param_name=RETENTION_PARAM)
    edited = next(entry for entry in history if entry["changed_by"] == "operator")
    assert edited["change_reason"] == "依用戶需求調整"


@pytest.mark.parametrize("changes", [
    {"enabled": "false"}, {"vacuum_after_cleanup": 1}, {"schedule_time": "24:00"},
    {"schedule_time": "2:00"}, {"schedule_time": "02:60"}, {"schedule_time": 120},
    {"ok_retain_days": 0}, {"ok_retain_days": True}, {"ok_retain_days": "180"},
    {"ng_retain_days": -1}, {"tile_retain_days": 1.5}, {"tile_retain_days": 3651},
    {"heatmap_retain_days": -1}, {"unexpected": 1},
])
def test_invalid_policy_is_rejected_without_changing_any_settings(tmp_path, changes):
    db = make_db(tmp_path)
    before = saved_policy(db)
    handler, responses = settings_handler(db)
    post_policy(handler, {**before, **changes})
    assert responses[-1][0] == 400
    assert saved_policy(db) == before
    assert len(db.get_config_change_history(param_name=RETENTION_PARAM)) == 1


def test_incomplete_policy_and_missing_reason_do_not_save(tmp_path):
    db = make_db(tmp_path)
    handler, responses = settings_handler(db)
    post_policy(handler, {"ok_retain_days": 200})
    assert responses[-1][0] == 400
    post_policy(handler, saved_policy(db), reason=" ")
    assert "error" in responses[-1][1]
    assert len(db.get_config_change_history(param_name=RETENTION_PARAM)) == 1


def test_zero_heatmap_days_keeps_legacy_no_image_cleanup_semantics():
    policy = initial_retention_policy(LEGACY_CONFIG)
    policy["heatmap_retain_days"] = 0
    assert normalize_retention_policy(policy)["heatmap_retain_days"] == 0


def run_scheduler(monkeypatch, db, on_wait, *, max_waits=8):
    clock = SimpleNamespace(now=datetime(2026, 10, 26, 1, 59))
    calls = []
    waits = []
    server = SimpleNamespace(
        server_config=LEGACY_CONFIG, db=db, _running=True, _cleanup_thread=None,
        heatmap_manager=SimpleNamespace(base_dir="unused"),
    )

    class Event:
        def clear(self):
            pass

        def is_set(self):
            return not server._running

        def wait(self, seconds):
            assert 0 < seconds <= 60
            waits.append(seconds)
            clock.now += timedelta(seconds=seconds)
            on_wait(len(waits), clock.now)
            if len(waits) >= max_waits:
                server._running = False
            return not server._running

    class Thread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    def cleanup(*args, **kwargs):
        calls.append((clock.now, args))
        return defaultdict(int)

    server._cleanup_stop_event = Event()
    monkeypatch.setattr(db, "cleanup_old_records", cleanup)
    monkeypatch.setattr("capi_server.datetime", SimpleNamespace(now=lambda: clock.now))
    monkeypatch.setattr("capi_server.threading.Thread", Thread)
    CAPIServer._start_cleanup_scheduler(server)
    return calls


def test_scheduler_uses_latest_retention_at_execution_and_runs_once_per_day(tmp_path, monkeypatch):
    db = make_db(tmp_path)

    def change(wait, now):
        if wait == 1:
            db.update_config_param(RETENTION_PARAM, {**saved_policy(db), "ok_retain_days": 240}, "update")
        if wait == 2:
            db.update_config_param(RETENTION_PARAM, {**saved_policy(db), "schedule_time": "02:03"}, "update time")

    calls = run_scheduler(monkeypatch, db, change)
    assert calls == [(datetime(2026, 10, 26, 2, 0), (240, 180, 15, False, 90))]


def test_scheduler_can_enable_then_reschedule_without_restart(tmp_path, monkeypatch):
    db = make_db(tmp_path)
    db.update_config_param(RETENTION_PARAM, {**saved_policy(db), "enabled": False}, "off")

    def change(wait, now):
        if wait == 1:
            db.update_config_param(RETENTION_PARAM, {**saved_policy(db), "enabled": True, "schedule_time": "02:02"}, "on")
        if wait == 2:
            db.update_config_param(RETENTION_PARAM, {**saved_policy(db), "schedule_time": "02:03"}, "later")

    calls = run_scheduler(monkeypatch, db, change)
    assert [time for time, _ in calls] == [datetime(2026, 10, 26, 2, 3)]


def test_invalid_stored_policy_never_executes_cleanup(tmp_path, monkeypatch):
    db = make_db(tmp_path)
    db.update_config_param(RETENTION_PARAM, {**saved_policy(db), "ok_retain_days": -1}, "corrupt")
    assert run_scheduler(monkeypatch, db, lambda count, now: None) == []


def test_scheduler_start_after_daily_time_waits_until_tomorrow():
    assert next_cleanup_time(datetime(2026, 10, 26, 3), "02:00") == datetime(2026, 10, 27, 2)


def test_retention_ui_preserves_draft_validates_and_handles_save_failure():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for settings UI behavior test")
    source = (Path(__file__).resolve().parents[1] / "templates/_retention_settings.js").read_text(encoding="utf-8")
    script = """
const assert = require('node:assert/strict');
let currentTab = 'data-retention';
let allParams = [{param_name: 'data_retention_policy', decoded_value: {
  enabled:true, schedule_time:'02:00', ok_retain_days:180, ng_retain_days:180,
  tile_retain_days:11, heatmap_retain_days:76, vacuum_after_cleanup:true
}}];
const escapeHtml = value => String(value).replaceAll('<', '&lt;').replaceAll('"', '&quot;');
let rendered = '', historyLoads = 0, requests = [], fail = false;
const document = {getElementById: id => ({reportValidity: () => true})};
function renderSettings() { rendered = renderRetentionPane(); }
function showToast() {}
async function loadHistory() { historyLoads++; }
async function fetch(url, options) {
  const body = JSON.parse(options.body); requests.push(body);
  return {ok: !fail, json: async () => fail ? {error:'登入已過期'} : {success:true, policy:body.new_value}};
}
""" + source + """
(async () => {
  renderSettings();
  assert.match(rendered, /pane-data-retention[^>]*active/);
  assert.match(rendered, /value="11"/);
  updateRetentionDraft('ok_retain_days', '240'); retentionReason = '半年以上追溯';
  renderSettings();
  assert.equal(retentionDraft.ok_retain_days, '240');
  updateRetentionDraft('tile_retain_days', ''); await saveRetentionSettings();
  assert.equal(requests.length, 0);
  updateRetentionDraft('tile_retain_days', '11');
  fail = true; await saveRetentionSettings();
  assert.equal(retentionDraft.ok_retain_days, '240');
  assert.equal(retentionDirty, true);
  assert.equal(retentionSaving, false);
  assert.match(rendered, /登入已過期/);
  fail = false; await saveRetentionSettings();
  assert.equal(requests.at(-1).new_value.ok_retain_days, 240);
  assert.equal(requests.at(-1).new_value.heatmap_retain_days, 76);
  assert.equal(historyLoads, 1);
  assert.equal(retentionDirty, false);
  assert.equal(allParams[0].decoded_value.ok_retain_days, 240);
  updateRetentionDraft('ok_retain_days', '300'); resetRetentionDraft();
  assert.equal(retentionDraft.ok_retain_days, 240);
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    completed = subprocess.run([node, "-e", script], capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 0, completed.stderr
