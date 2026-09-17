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
    ],
)
def test_watch_models_reject_invalid_values(tmp_path, models, message):
    db = CAPIDatabase(tmp_path / "dashboard.db")

    with pytest.raises(ValueError, match=message):
        db.save_central_dashboard_watch_models(models)


def test_watch_models_accepts_unbounded_list(tmp_path):
    db = CAPIDatabase(tmp_path / "dashboard.db")
    models = [f"M{i:05d}" for i in range(500)]

    assert db.save_central_dashboard_watch_models(models) == models
    assert db.get_central_dashboard_watch_models() == models


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


def _make_json_handler(db, payload):
    handler = object.__new__(CAPIWebHandler)
    handler.db = db
    body = json.dumps(payload).encode("utf-8")
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler._current_settings_user = lambda: {"username": "operator"}
    responses = []
    handler._send_json = lambda data, status=200, headers=None: responses.append(
        (status, data)
    )
    return handler, responses


def test_watch_models_api_saves_and_returns_list(tmp_path):
    db = CAPIDatabase(tmp_path / "dashboard.db")
    handler, responses = _make_json_handler(
        db, {"watchModels": ["model_a", " MODEL_B "]}
    )

    handler._handle_api_central_dashboard_watch_models_update()

    assert responses[-1][0] == 200
    assert responses[-1][1] == {
        "success": True,
        "watchModels": ["MODEL_A", "MODEL_B"],
    }


def test_watch_models_api_rejects_invalid_payload(tmp_path):
    db = CAPIDatabase(tmp_path / "dashboard.db")
    handler, responses = _make_json_handler(
        db, {"watchModels": ["MODEL_A", "MODEL_A"]}
    )

    handler._handle_api_central_dashboard_watch_models_update()

    assert responses[-1][0] == 400
    assert "重複" in responses[-1][1]["error"]


def test_watch_models_route_requires_settings_login():
    handler = CAPIWebHandler.__new__(CAPIWebHandler)
    handler.path = "/api/central-dashboard/watch-models"
    captured = {}

    def require_user(**kwargs):
        captured.update(kwargs)
        return None

    handler._require_settings_user = require_user
    handler._handle_api_central_dashboard_watch_models_update = (
        lambda: (_ for _ in ()).throw(
            AssertionError("unauthorized request must not update watch models")
        )
    )

    handler.do_POST()

    assert captured == {"api": True}


def test_config_all_api_includes_watch_models(tmp_path):
    db = CAPIDatabase(tmp_path / "dashboard.db")
    db.save_central_dashboard_watch_models(["MODEL_A"])
    handler = object.__new__(CAPIWebHandler)
    handler.db = db
    responses = []
    handler._send_json = lambda data, status=200, headers=None: responses.append(
        (status, data)
    )

    handler._handle_api_central_dashboard_config_all()

    assert responses[-1][0] == 200
    assert responses[-1][1]["watchModels"] == ["MODEL_A"]


def test_filtered_config_api_keeps_watch_models(tmp_path):
    db = CAPIDatabase(tmp_path / "dashboard.db")
    db.save_central_dashboard_watch_models(["MODEL_A"])
    handler = object.__new__(CAPIWebHandler)
    handler.db = db
    handler.connection = type(
        "_Connection",
        (),
        {"getsockname": lambda self: ("10.174.99.10", 80)},
    )()
    handler.headers = {}
    responses = []
    handler._send_json = lambda data, status=200, headers=None: responses.append(
        (status, data)
    )

    handler._handle_api_central_dashboard_config()

    assert responses[-1][0] == 200
    assert responses[-1][1]["watchModels"] == ["MODEL_A"]


def test_server_status_latest_event_carries_model_id():
    from capi_server import ServerStatusTracker

    tracker = ServerStatusTracker()
    tracker.last_judgment_result = {
        "glass_id": "G001",
        "model_id": "GN140JPAA020S",
        "machine_no": "AOI01",
        "judgment": "OK",
        "detail": "OK",
        "time": "12:00:00",
        "duration": "1.00s",
    }

    status = tracker.get_status()

    assert status["latest_event"]["model_id"] == "GN140JPAA020S"


def test_server_sets_model_id_on_both_judgment_paths():
    root = Path(__file__).resolve().parent.parent
    source = (root / "capi_server.py").read_text(encoding="utf-8")

    assert source.count('"model_id": parsed["model_id"],') == 2


def test_frontend_watch_badge_wiring():
    root = Path(__file__).resolve().parent.parent
    app_js = (root / "central_dashboard" / "app.js").read_text(encoding="utf-8")
    index_html = (root / "central_dashboard" / "index.html").read_text(
        encoding="utf-8"
    )
    styles_css = (root / "central_dashboard" / "styles.css").read_text(
        encoding="utf-8"
    )

    assert "watchModels: normalizeWatchModels(raw.watchModels)" in app_js
    assert "function normalizeWatchModels(value)" in app_js
    assert "modelId: textValue(latestEvent.model_id)," in app_js
    assert "function matchedWatchModel(state)" in app_js
    assert 'state.status !== "online"' in app_js
    assert "function updateWatchBadge(badge, state)" in app_js
    assert 'watchBadge.textContent = "★";' in app_js
    assert '[data-field="overview-watch"]' in app_js
    assert '[data-field="watch-badge"]' in app_js
    assert "正在生產關注機種：" in app_js

    assert 'data-field="watch-badge" hidden>★</span>' in index_html
    assert 'class="line-title-row"' in index_html

    assert ".overview-watch-badge," in styles_css
    badge_css = styles_css.split(".overview-watch-badge,", 1)[1][:400]
    badge_rule = badge_css.split("}")[0]
    assert "color: var(--red);" in badge_css
    assert "background:" not in badge_rule  # 純紅色星號，無膠囊底色與邊框
    assert "border:" not in badge_rule
    assert "#f6c945" not in badge_css
    assert "margin-top" not in badge_css
    assert ".overview-watch-badge[hidden] {" in styles_css
    assert "visibility: hidden;" in styles_css  # 總覽表星號隱藏時佔位，線體名對齊
    assert ".line-watch-badge[hidden]" in styles_css  # 卡片星號隱藏即移除
    assert ".line-title-row {" in styles_css


def test_settings_page_has_watch_models_section():
    root = Path(__file__).resolve().parent.parent
    settings_html = (root / "central_dashboard" / "settings.html").read_text(
        encoding="utf-8"
    )

    assert 'id="watch-title"' in settings_html
    assert "重點機種關注" in settings_html
    assert 'id="watch-input"' in settings_html
    assert 'id="watch-add"' in settings_html
    assert 'id="watch-chips"' in settings_html
    assert 'id="watch-message"' in settings_html
    assert 'id="watch-save"' in settings_html
    assert 'fetch("/api/central-dashboard/watch-models"' in settings_html
    assert "function normalizeWatchModelCode(" in settings_html
    assert "function addWatchModelsFromInput(" in settings_html
    assert "function removeWatchModel(" in settings_html
    assert "function renderWatchChips(" in settings_html
    assert "function saveWatchModels(" in settings_html
    assert "watchInput.value.split(/[\\s,;，、]+/)" in settings_html
    assert "state.watchModels" in settings_html
    assert settings_html.index('id="watch-title"') > settings_html.index(
        'id="save"'
    )
