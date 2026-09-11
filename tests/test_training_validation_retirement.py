from pathlib import Path

import pytest

from capi_database import CAPIDatabase
from capi_train_new import TrainingConfig, apply_user_training_params
from capi_web import CAPIWebHandler


@pytest.fixture
def old_review(tmp_path):
    db = CAPIDatabase(str(tmp_path / "old.db"))
    db.create_training_job("old", "M", [], training_params={
        "batch_size": 8, "validation_config": {"split_mode": "auto_panel", "panels": {}}})
    ids = db.insert_tile_pool("old", [dict(lighting="W0F00000", zone="inner", source="ok",
        source_path=str(tmp_path / f"{i}.png"), dataset_role=role,
        validation_group=str(i), validation_label="ok")
        for i, role in enumerate(("train", "calibration", "acceptance"))])
    db.update_tile_decisions("old", [ids[-1]], "reject")
    db.update_training_job_state("old", "review")
    return db


def test_old_review_returns_all_kept_tiles_to_training_without_changing_decisions(old_review):
    before = old_review.list_tile_pool("old")
    job = CAPIWebHandler._prepare_panel_validation_review(old_review, old_review.get_training_job("old"))
    assert job["training_params"] == {"batch_size": 8}
    after = old_review.list_tile_pool("old")
    assert [t["decision"] for t in after] == [t["decision"] for t in before]
    assert all(t["dataset_role"] == "train" and not t["validation_label"] and not t["validation_group"] for t in after)
    assert CAPIWebHandler._prepare_panel_validation_review(old_review, job) == job


@pytest.mark.parametrize("state", ["train", "preprocess", "completed", "failed"])
def test_retirement_does_not_mutate_non_review_jobs(old_review, state):
    old_review.update_training_job_state("old", state)
    before = old_review.list_tile_pool("old")
    job = old_review.get_training_job("old")
    assert not old_review.retire_training_validation("old")
    assert old_review.list_tile_pool("old") == before
    assert old_review.get_training_job("old") == job


def test_failed_job_is_retired_after_retry_returns_to_review(old_review):
    old_review.update_training_job_state("old", "failed")
    assert old_review.reset_failed_training_job_for_review("old")
    job = CAPIWebHandler._prepare_panel_validation_review(old_review, old_review.get_training_job("old"))
    assert "validation_config" not in job["training_params"]


@pytest.mark.parametrize("legacy", [{"split_mode": "auto_panel"}, "stale payload"])
def test_stale_browser_configuration_cannot_enable_calibration(legacy):
    cleaned, error = CAPIWebHandler._validate_training_params({"batch_size": 8, "validation_config": legacy})
    assert error is None and cleaned == {"batch_size": 8}
    cfg = TrainingConfig(machine_id="M", panel_paths=[], over_review_root=Path("unused"))
    apply_user_training_params(cfg, {"batch_size": 8, "validation_config": legacy})
    assert cfg.validation_config == {} and cfg.batch_size == 8


def test_training_page_keeps_exclusion_and_memory_controls():
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader("templates"))
    env.globals["app_version"] = {"version": "test"}
    html = env.get_template("train_new/step3_review.html").render(job_id="old",
        selected_lightings=["W0F00000"], lighting_labels={}, training_scope={"selected_units": ["W0F00000-inner"]})
    assert '影像規則自動排除' in html and 'oom-warning' in html
    assert '自動排除沿用 NG 用途' not in html
    for retired in ('modal-validation', 'labelValidation', 'auto-split-summary', '校正 ·', '驗收', '待標記'):
        assert retired not in html


def test_old_label_operation_is_disabled():
    import io
    import json
    from unittest.mock import MagicMock
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    raw = json.dumps({'job_id': 'old', 'tile_ids': [1], 'validation_label': 'ng'}).encode()
    handler.headers = {'Content-Length': str(len(raw))}
    handler.rfile = io.BytesIO(raw)
    handler._send_json = MagicMock()
    handler._handle_train_new_tiles_decision()
    assert handler._send_json.call_args.kwargs['status'] == 410
