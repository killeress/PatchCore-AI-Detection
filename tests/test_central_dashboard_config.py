import io
import json
import sqlite3
from pathlib import Path

import pytest

from capi_database import CAPIDatabase
from capi_web import CAPIWebHandler


def test_central_dashboard_defaults_are_imported_to_sqlite(tmp_path):
    db = CAPIDatabase(tmp_path / "dashboard.db")

    config = db.get_central_dashboard_config()

    assert config["title"] == "寧波廠區 CAPI AI 中控看板"
    assert config["refreshIntervalSeconds"] == 30
    assert config["requestTimeoutSeconds"] == 8
    assert [line["id"] for line in config["lines"]] == [
        "mod2-capi03",
        "mod2-capi13",
        "mod2-hm-83",
        "mod2-hm-103",
        "mod2-capi08",
        "mod2-capi01",
        "mod2-capi14",
        "mod2-capi02",
        "mod1-capi35",
        "mod1-capi34",
    ]
    assert all(
        set(line) == {
            "id",
            "factory",
            "line",
            "pcName",
            "apiUrl",
            "dashboardUrl",
            "overexposedUrl",
            "enabled",
            "isProduction",
        }
        for line in config["lines"]
    )
    assert all(line["isProduction"] is False for line in config["lines"])
    with sqlite3.connect(db.db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM central_dashboard_settings"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM central_dashboard_lines"
        ).fetchone()[0] == 10


def test_central_dashboard_first_read_can_import_local_file_config(tmp_path):
    db = CAPIDatabase(tmp_path / "dashboard.db")
    initial = {
        "title": "現場已修改看板",
        "refreshIntervalSeconds": 90,
        "requestTimeoutSeconds": 12,
        "lines": [
            {
                "id": "field-line",
                "factory": "MOD9",
                "line": "現場線",
                "pcName": "FIELD-PC",
                "apiUrl": "http://10.9.0.1/api/status",
                "dashboardUrl": "http://10.9.0.1/",
                "overexposedUrl": "",
                "enabled": True,
                "isProduction": False,
            }
        ],
    }

    assert db.get_central_dashboard_config(initial) == initial


def test_central_dashboard_config_js_can_be_parsed_for_first_import():
    config = CAPIWebHandler._load_central_dashboard_file_config()

    assert config["title"] == "寧波廠區 CAPI AI 中控看板"
    assert config["lines"][0]["id"] == "mod2-capi03"
    assert len(config["lines"]) == 10


def test_central_dashboard_config_can_be_replaced_and_keep_order(tmp_path):
    db = CAPIDatabase(tmp_path / "dashboard.db")
    db.get_central_dashboard_config()

    saved = db.save_central_dashboard_config(
        {
            "title": "全廠戰情室",
            "refreshIntervalSeconds": 60,
            "requestTimeoutSeconds": 10,
            "lines": [
                {
                    "id": "mod2-capi02",
                    "factory": "MOD2",
                    "line": "CAPI02",
                    "pcName": "PC-02",
                    "apiUrl": "http://10.0.0.2/api/status",
                    "dashboardUrl": "http://10.0.0.2/",
                    "overexposedUrl": "",
                    "enabled": False,
                    "isProduction": True,
                },
                {
                    "id": "mod1-capi01",
                    "factory": "MOD1",
                    "line": "CAPI01",
                    "pcName": "PC-01",
                    "apiUrl": "https://capi01.example/api/status",
                    "dashboardUrl": "https://capi01.example/",
                    "overexposedUrl": "https://capi01.example/overexposed",
                    "enabled": True,
                },
            ],
        },
        changed_by="tester",
    )

    assert saved["title"] == "全廠戰情室"
    assert [line["id"] for line in saved["lines"]] == [
        "mod2-capi02",
        "mod1-capi01",
    ]
    assert saved["lines"][0]["enabled"] is False
    assert saved["lines"][0]["isProduction"] is True
    assert saved["lines"][1]["isProduction"] is False
    assert db.get_central_dashboard_config() == saved
    with sqlite3.connect(db.db_path) as connection:
        assert connection.execute(
            "SELECT updated_by FROM central_dashboard_settings WHERE id = 1"
        ).fetchone()[0] == "tester"


def test_central_dashboard_existing_database_adds_is_production_column(tmp_path):
    db_path = tmp_path / "legacy-dashboard.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """CREATE TABLE central_dashboard_lines (
                   id TEXT PRIMARY KEY,
                   factory TEXT NOT NULL,
                   line_name TEXT NOT NULL,
                   pc_name TEXT NOT NULL,
                   api_url TEXT NOT NULL,
                   dashboard_url TEXT DEFAULT '',
                   overexposed_url TEXT DEFAULT '',
                   enabled INTEGER NOT NULL DEFAULT 1,
                   sort_order INTEGER NOT NULL DEFAULT 0,
                   updated_by TEXT DEFAULT '',
                   updated_at TEXT DEFAULT (datetime('now', 'localtime'))
               )"""
        )
        connection.execute(
            """INSERT INTO central_dashboard_lines
               (id, factory, line_name, pc_name, api_url)
               VALUES ('legacy-line', 'MOD2', 'CAPI01', 'CAPI01',
                       'http://10.174.37.137/api/status')"""
        )

    CAPIDatabase(db_path)

    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1] for row in connection.execute(
                "PRAGMA table_info(central_dashboard_lines)"
            ).fetchall()
        }
        is_production = connection.execute(
            "SELECT is_production FROM central_dashboard_lines WHERE id = ?",
            ("legacy-line",),
        ).fetchone()[0]

    assert "is_production" in columns
    assert is_production == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"refreshIntervalSeconds": 10},
            "更新週期必須介於 30 到 3600 秒",
        ),
        (
            {"requestTimeoutSeconds": 30},
            "API 逾時必須至少 3 秒，且小於更新週期",
        ),
        (
            {
                "lines": [
                    {
                        "id": "line-1",
                        "factory": "MOD1",
                        "line": "CAPI01",
                        "pcName": "PC-01",
                        "apiUrl": "javascript:alert(1)",
                    }
                ]
            },
            "URL 必須使用 http:// 或 https://",
        ),
    ],
)
def test_central_dashboard_config_rejects_invalid_values(tmp_path, change, message):
    db = CAPIDatabase(tmp_path / "dashboard.db")
    config = db.get_central_dashboard_config()
    config.update(change)

    with pytest.raises(ValueError, match=message):
        db.save_central_dashboard_config(config)


def test_central_dashboard_api_updates_sqlite(tmp_path):
    db = CAPIDatabase(tmp_path / "dashboard.db")
    config = db.get_central_dashboard_config()
    config["title"] = "中控測試"
    payload = json.dumps(config).encode("utf-8")

    handler = object.__new__(CAPIWebHandler)
    handler.db = db
    handler.headers = {"Content-Length": str(len(payload))}
    handler.rfile = io.BytesIO(payload)
    handler._current_settings_user = lambda: {"username": "operator"}
    responses = []
    handler._send_json = lambda data, status=200, headers=None: responses.append(
        (status, data)
    )

    handler._handle_api_central_dashboard_config_update()

    assert responses[-1][0] == 200
    assert responses[-1][1]["success"] is True
    assert db.get_central_dashboard_config()["title"] == "中控測試"


@pytest.mark.parametrize(
    ("webserver_ip", "expected_prefix", "expected_ids"),
    [
        (
            "10.172.99.10",
            "10.172",
            ["mod1-capi35", "mod1-capi34"],
        ),
        (
            "10.174.99.10",
            "10.174",
            [
                "mod2-capi03",
                "mod2-capi13",
                "mod2-hm-83",
                "mod2-hm-103",
                "mod2-capi08",
                "mod2-capi01",
                "mod2-capi14",
                "mod2-capi02",
            ],
        ),
    ],
)
def test_central_dashboard_api_filters_to_webserver_network(
    tmp_path,
    webserver_ip,
    expected_prefix,
    expected_ids,
):
    db = CAPIDatabase(tmp_path / "dashboard.db")
    handler = object.__new__(CAPIWebHandler)
    handler.db = db
    handler.connection = type(
        "_Connection",
        (),
        {"getsockname": lambda self: (webserver_ip, 80)},
    )()
    handler.headers = {}
    responses = []
    handler._send_json = lambda data, status=200, headers=None: responses.append(
        (status, data)
    )

    handler._handle_api_central_dashboard_config()

    payload = responses[-1][1]
    assert responses[-1][0] == 200
    assert payload["webServerIp"] == webserver_ip
    assert payload["networkPrefix"] == expected_prefix
    assert payload["networkFilterApplied"] is True
    assert payload["configuredLineCount"] == 10
    assert [line["id"] for line in payload["lines"]] == expected_ids


def test_central_dashboard_all_api_keeps_both_factories(tmp_path):
    db = CAPIDatabase(tmp_path / "dashboard.db")
    handler = object.__new__(CAPIWebHandler)
    handler.db = db
    responses = []
    handler._send_json = lambda data, status=200, headers=None: responses.append(
        (status, data)
    )

    handler._handle_api_central_dashboard_config_all()

    payload = responses[-1][1]
    assert responses[-1][0] == 200
    assert len(payload["lines"]) == 10
    assert {line["factory"] for line in payload["lines"]} == {"MOD1", "MOD2"}
    assert "networkPrefix" not in payload


def test_central_dashboard_pages_use_sqlite_config_and_settings_route():
    root = Path(__file__).resolve().parent.parent
    index_html = (root / "central_dashboard" / "index.html").read_text(
        encoding="utf-8"
    )
    app_js = (root / "central_dashboard" / "app.js").read_text(encoding="utf-8")
    styles_css = (root / "central_dashboard" / "styles.css").read_text(
        encoding="utf-8"
    )
    settings_html = (root / "central_dashboard" / "settings.html").read_text(
        encoding="utf-8"
    )

    assert 'href="/central_dashboard/settings"' in index_html
    assert 'fetch("/api/central-dashboard/config"' in app_js
    assert "中控看板設備設定" in settings_html
    assert 'method: "POST"' in settings_html
    assert 'fetch("/api/central-dashboard/config/all"' in settings_html
    assert 'data-link="overexposed"' not in index_html
    assert "Omit 過曝明細" not in index_html
    assert 'data-field="shift-name"' not in index_html
    assert 'data-field="shift-range"' not in index_html
    assert 'id="data-note"' not in index_html
    assert '設備清單儲存於中央 SQLite；即時狀態僅讀取各 PC API。' not in index_html
    assert 'id="footer-refresh-note"' not in index_html
    assert 'configureLink(card, "overexposed"' not in app_js
    assert '每 ${config.refreshIntervalSeconds} 秒由各 PC 的 API 更新一次' not in app_js
    assert 'refreshTimer = window.setTimeout' in app_js
    assert 'id="line-overview"' in index_html
    assert 'id="theme-toggle"' in index_html
    assert 'class="topbar-action-group"' in index_html
    assert 'class="toolbar-icon"' in index_html
    assert 'id="refresh-button"' not in index_html
    assert 'document.getElementById("refresh-button")' not in app_js
    assert 'id="refresh-status" data-state="ready"' in index_html
    assert '#refresh-status::before' in styles_css
    assert 'element.dataset.state = "refreshing"' in app_js
    assert "createOverviewRow(line)" in app_js
    assert "renderOverviewRow(state)" in app_js
    assert "<th scope=\"col\">AOI 連線</th>" in index_html
    assert "<th scope=\"col\">AOI 排片率</th>" in index_html
    assert "<th scope=\"col\">AI 排片率</th>" in index_html
    assert "<th scope=\"col\">最近生產活動</th>" in index_html
    assert "<th scope=\"col\">異常摘要</th>" in index_html
    assert "<th scope=\"col\">程式版本</th>" in index_html
    assert 'aoi.dataset.state = "connected";' in app_js
    assert 'setText(aoi, "AOI 未連線");' in app_js
    assert "aoiNg: optionalNumber(stats.aoi_ng_count)" in app_js
    assert "aiNg: optionalNumber(stats.ai_ng_count ?? stats.total_ng ?? stats.ng_count)" in app_js
    assert 'renderOverviewRejectRate(row, "aoi-rate", "AOI", data.aoiNg, data.total);' in app_js
    assert 'renderOverviewRejectRate(row, "ai-rate", "AI", data.aiNg, data.total);' in app_js
    assert "function formatRelativeTime(value, now = new Date())" in app_js
    assert "`最近 ${judgment} · ${relativeTime}`" in app_js
    assert "badge.textContent = `⚠ ${alert.summary}`;" in app_js
    assert 'link.textContent = "開啟";' in app_js
    assert 'link.textContent = "開啟設備";' not in app_js
    assert 'updateButton.textContent = "更新程式";' in app_js
    assert "function renderOverviewUpdate(state)" in app_js
    assert "function applyCentralUpdate(lineId)" in app_js
    assert 'fetch("/api/central-dashboard/update/apply"' in app_js
    assert 'body: JSON.stringify({' in app_js
    assert "window.confirm(" in app_js
    assert "訓練工作進行中" in app_js
    assert "overview-update-action" in styles_css
    assert ".overview-update-action[hidden]" in styles_css
    assert "focus_update" not in app_js
    assert 'localStorage.setItem(THEME_STORAGE_KEY, nextTheme)' in app_js
    assert ':root[data-theme="dark"]' in styles_css
    assert 'createInput(index, "ip", "設備 IP", 15, true)' in settings_html
    assert 'createInput(index, "pcName"' not in settings_html
    assert 'createInput(index, "apiUrl"' not in settings_html
    assert 'apiUrl: `${baseUrl}/api/status`' in settings_html
    assert 'dashboardUrl: `${baseUrl}/`' in settings_html
    assert 'overexposedUrl: `${baseUrl}/overexposed`' in settings_html
    assert 'productionLabel.append(production, "正式上線")' in settings_html
    assert "isProduction: line.isProduction === true" in settings_html
    assert "createMoveButton(index, -1" in settings_html
    assert "createMoveButton(index, 1" in settings_html
    assert 'row.dataset.production = "true"' in app_js
    assert 'badge.textContent = "上線"' in app_js
    assert "overview-production-badge" in styles_css
    assert 'tr[data-production="true"]' in styles_css
    assert "card.dataset.production" not in app_js
    assert "hostname: textValue(server.hostname)" in app_js
    assert 'data.hostname ||' in app_js
