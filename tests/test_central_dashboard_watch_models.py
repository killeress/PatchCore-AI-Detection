import io
import json
import sqlite3
from pathlib import Path

import pytest

from capi_database import CAPIDatabase
from capi_web import CAPIWebHandler


def test_watch_models_default_empty(tmp_path):
    db = CAPIDatabase(tmp_path / "dashboard.db")

    assert db.get_central_dashboard_watch_models() == []
    assert db.get_central_dashboard_config()["watchModels"] == []


def test_watch_models_are_normalized_on_save(tmp_path):
    db = CAPIDatabase(tmp_path / "dashboard.db")

    saved = db.save_central_dashboard_watch_models(
        ["  gn140jpaa020s ", "ＧＮ１４０ＨＧＡＡ４００Ｓ"],
        changed_by="tester",
    )

    assert saved == ["GN140JPAA020S", "GN140HGAA400S"]
    assert db.get_central_dashboard_watch_models() == saved
    with sqlite3.connect(db.db_path) as connection:
        assert connection.execute(
            "SELECT updated_by FROM central_dashboard_watch_models WHERE model = ?",
            ("GN140JPAA020S",),
        ).fetchone()[0] == "tester"


def test_watch_models_persist_across_reopen(tmp_path):
    db = CAPIDatabase(tmp_path / "dashboard.db")
    db.save_central_dashboard_watch_models(["GN116BCAAK50S"])

    reopened = CAPIDatabase(tmp_path / "dashboard.db")

    assert reopened.get_central_dashboard_watch_models() == ["GN116BCAAK50S"]


def test_watch_models_save_replaces_previous_list(tmp_path):
    db = CAPIDatabase(tmp_path / "dashboard.db")
    db.save_central_dashboard_watch_models(["MODEL_A", "MODEL_B"])

    assert db.save_central_dashboard_watch_models(["MODEL_C"]) == ["MODEL_C"]
    assert db.get_central_dashboard_watch_models() == ["MODEL_C"]


@pytest.mark.parametrize(
    ("models", "message"),
    [
        ([""], "不可空白"),
        (["   "], "不可空白"),
        (["MODEL_A", "model_a"], "重複"),
        (["MODEL A"], "不可包含空白"),
        (["BAD\tCODE"], "不可包含空白"),
        (["X" * 51], "不可超過 50 字"),
        ([f"M{i:03d}" for i in range(101)], "不可超過 100 筆"),
    ],
)
def test_watch_models_reject_invalid_values(tmp_path, models, message):
    db = CAPIDatabase(tmp_path / "dashboard.db")

    with pytest.raises(ValueError, match=message):
        db.save_central_dashboard_watch_models(models)


def test_watch_models_not_a_list_rejected(tmp_path):
    db = CAPIDatabase(tmp_path / "dashboard.db")

    with pytest.raises(ValueError, match="必須是陣列"):
        db.save_central_dashboard_watch_models("MODEL_A")


def test_save_dashboard_config_does_not_touch_watch_models(tmp_path):
    db = CAPIDatabase(tmp_path / "dashboard.db")
    db.save_central_dashboard_watch_models(["MODEL_A"])
    config = db.get_central_dashboard_config()
    config["title"] = "改名"
    config["watchModels"] = ["SHOULD_BE_IGNORED"]

    saved = db.save_central_dashboard_config(config, changed_by="tester")

    assert db.get_central_dashboard_watch_models() == ["MODEL_A"]
    assert saved["watchModels"] == ["MODEL_A"]
    assert db.get_central_dashboard_config()["watchModels"] == ["MODEL_A"]
