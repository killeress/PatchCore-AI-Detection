import re
import sqlite3

import pytest

import capi_database
from capi_database import CAPIDatabase


def _save_record(db, judgment="OK"):
    return db.save_inference_record(
        glass_id="panel-1", model_id="model-1", machine_no="CAPI01",
        resolution=(100, 100), machine_judgment="OK", ai_judgment=judgment,
        image_dir="/images", total_images=0, ng_images=0, ng_details="[]",
        request_time="2026-10-14 12:00:00", response_time="2026-10-14 12:00:01",
        processing_seconds=1.0,
    )


@pytest.mark.parametrize("judgment", ["OK", "NG", "ERR"])
def test_records_keep_running_version_after_release_files_change(tmp_path, monkeypatch, judgment):
    monkeypatch.setattr(capi_database, "_APP_VERSION", "2026.10.14.2")
    db = CAPIDatabase(tmp_path / "records.db")
    record_id = _save_record(db, judgment)
    # Deployment files can change before the old service finishes its queued saves.
    monkeypatch.setattr(capi_database, "get_version_info", lambda: {"version": "2099.01.01.1"})
    second_id = _save_record(db, judgment)
    assert db.get_record_detail(record_id)["app_version"] == "2026.10.14.2"
    assert db.get_record_detail(second_id)["app_version"] == "2026.10.14.2"


def test_migration_leaves_history_unknown_and_rerun_records_new_version(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    db = CAPIDatabase(path)
    record_id = _save_record(db)
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE inference_records DROP COLUMN app_version")

    monkeypatch.setattr(capi_database, "_APP_VERSION", "2026.10.15.1")
    db = CAPIDatabase(path)
    assert db.get_record_detail(record_id)["app_version"] == ""
    new_id = _save_record(db)
    assert db.get_record_detail(new_id)["app_version"] == "2026.10.15.1"

    db.update_record_for_rerun(
        record_id, ai_judgment="NG", total_images=0, ng_images=0,
        ng_details="[]", processing_seconds=2.0,
    )
    detail = CAPIDatabase(path).get_record_detail(record_id)
    assert detail["app_version"] == "2026.10.15.1"
    assert detail["ai_judgment"] == "NG"


@pytest.mark.parametrize("template", ["record_detail.html", "record_detail_v3.html"])
@pytest.mark.parametrize("version", [None, "", "2026.10.14.2"])
def test_detail_displays_record_version_or_unknown(template, version):
    from capi_web import CAPIWebHandler

    detail = dict(id=1, ai_judgment="OK", processing_seconds=1.0, images=[])
    if version is not None:
        detail["app_version"] = version
    CAPIWebHandler.init_jinja()
    html = CAPIWebHandler.jinja_env.get_template(template).render(
        detail=detail, heatmap_base_dir="/heatmaps",
        app_version={"version": "2099.01.01.1"},
    )
    row = re.search(r"推論程式版本</(?:th|td)>\s*<td[^>]*>(.*?)</td>", html)
    assert row is not None
    assert row.group(1) == (version or "未記錄")
