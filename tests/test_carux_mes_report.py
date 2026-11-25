from datetime import datetime
import sqlite3
import sys
from types import SimpleNamespace

import pytest

import capi_mes_report
from capi_mes_report import OracleMESRepository, build_mes_comparison


CUTOFF = datetime(2026, 11, 23, 8, 0, 0)


@pytest.fixture
def carux_oracle(monkeypatch):
    """Use a real SQLite table for the SQL; replace only the external Oracle connection."""
    connection = sqlite3.connect(":memory:")
    connection.execute("""CREATE TABLE MAA02SHIS_DEFECTCODE (
        FAC_ID TEXT, ENTITY_NO TEXT, WORK_CTR TEXT, MAIN_WC TEXT, DFCT_CODE TEXT,
        TRANS_DATE TEXT, X_AXIS NUMERIC, Y_AXIS NUMERIC, ERROR_FLAG TEXT,
        PROD_NBR TEXT, TRANS_NBR TEXT, ITEM_NBR INTEGER
    )""")
    queries = []

    class Cursor:
        def __init__(self):
            self.cursor = connection.cursor()

        def execute(self, sql, binds):
            queries.append((sql, binds))
            self.cursor.execute(sql, binds)

        @property
        def description(self):
            return self.cursor.description

        def __iter__(self):
            return iter(self.cursor)

        def close(self):
            self.cursor.close()

    monkeypatch.setattr(capi_mes_report, "ORACLE_MES_PASSWORD", "test-password")
    monkeypatch.setitem(sys.modules, "oracledb", SimpleNamespace(
        makedsn=lambda *args, **kwargs: "test-dsn",
        connect=lambda **kwargs: SimpleNamespace(cursor=Cursor, close=lambda: None),
    ))
    config = {
        "facility": "CARUX",
        "oracle": {"user": "MDRDA1", "tns": {"CARUX": {
            "host": "example.invalid", "service_name": "tnbfmr",
        }}},
    }
    yield OracleMESRepository(config), connection, queries
    connection.close()


def insert_row(connection, *, panel="PANEL-1", fac="AA02", station="1600",
               code="PCU12", time="2026-11-23 08:00:00", x=10, y=20,
               error_flag=None):
    connection.execute(
        """INSERT INTO MAA02SHIS_DEFECTCODE
           (FAC_ID, ENTITY_NO, WORK_CTR, MAIN_WC, DFCT_CODE, TRANS_DATE, X_AXIS, Y_AXIS, ERROR_FLAG)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (fac, panel, station, "OTHER", code, time, x, y, error_flag),
    )


def test_carux_query_and_detail_apply_time_per_inference(carux_oracle):
    repository, connection, _ = carux_oracle
    insert_row(connection, time="2026-11-23 07:59:59")
    insert_row(connection, time="2026-11-23 08:00:00", error_flag="Y")
    insert_row(connection, station="1605", time="2026-11-24 00:01:00")
    insert_row(connection, station="1700")
    insert_row(connection, fac="OTHER")
    insert_row(connection, panel="OTHER")

    rows = repository.fetch_defects(["panel-1", "PANEL-1"], CUTOFF)
    assert {row["trans_date"] for row in rows["PANEL-1"]} == {
        "2026-11-23 07:59:59", "2026-11-23 08:00:00", "2026-11-24 00:01:00",
    }
    details = repository.fetch_report_details("panel-1", CUTOFF)
    assert len(details) == 2
    assert {row["TRANS_DATE"] for row in details} == {
        "2026-11-23 08:00:00", "2026-11-24 00:01:00",
    }
    assert {row["WORK_CTR"] for row in details} == {"1600", "1605"}
    assert {row["MAIN_WC"] for row in details} == {"OTHER"}
    assert all(row["ENTITY_NO"] == "PANEL-1" for row in details)
    assert any(row["ERROR_FLAG"] == "Y" for row in details)
    assert "MAA02SHIS_DEFECTCODE" in repository.source_label
    assert repository.rule_label == "FAC_ID=AA02、WORK_CTR=1600/1605、TRANS_DATE>=推論時間，不要求 X/Y，查無資料或未命中判 OK"


def test_carux_counts_pck21_and_missing_coordinates_after_each_inference(carux_oracle):
    repository, connection, _ = carux_oracle
    insert_row(connection, code="PCK21", time="2026-11-23 08:01:00", x=None, y=None)
    insert_row(connection, code="PCU12", time="2026-11-23 08:02:00", x=None)
    insert_row(connection, code="PCU12", time="2026-11-23 08:03:00", x=0, y=0)
    rows = repository.fetch_defects(["PANEL-1"], CUTOFF)
    records = [
        {"id": 1, "glass_id": "PANEL-1", "request_time": CUTOFF, "ai_judgment": "OK"},
        {"id": 2, "glass_id": "PANEL-1", "request_time": "2026-11-23 08:04:00", "ai_judgment": "NG"},
    ]
    report = build_mes_comparison(records, rows, station_profile="capi", facility="CARUX")
    assert [row["mes_judgment"] for row in report["records"]] == ["NG", "OK"]
    assert [row["qualifying_defect_count"] for row in report["records"]] == [3, 0]
    assert any(item["dfct_code"] == "PCK21" for item in report["records"][0]["qualifying_defects"])


@pytest.mark.parametrize("mes_time", [
    None, "not-a-date", "2026-11-23", "2026-11-23 08:00.00",
    "2026-11-23 08:00:00.1234567", "2026-11-23T08:00:00+08:00",
])
def test_carux_uncomparable_mes_time_reports_error(carux_oracle, mes_time):
    repository, connection, _ = carux_oracle
    insert_row(connection, time=mes_time)
    rows = repository.fetch_defects(["PANEL-1"], CUTOFF)
    with pytest.raises(ValueError, match="TRANS_DATE"):
        build_mes_comparison(
            [{"id": 1, "glass_id": "PANEL-1", "request_time": CUTOFF, "ai_judgment": "OK"}],
            rows, facility="CARUX",
        )
    with pytest.raises(ValueError, match="TRANS_DATE"):
        repository.fetch_report_details("PANEL-1", CUTOFF)


@pytest.mark.parametrize("mes_time", [
    "2026-11-23 08:00:00", "2026-11-23T08:00:00",
    "2026-11-23 08.00.00", "2026-11-23 08.00.00.123456",
])
def test_carux_supported_complete_time_formats_match(carux_oracle, mes_time):
    repository, connection, _ = carux_oracle
    insert_row(connection, time=mes_time)
    rows = repository.fetch_defects(["PANEL-1"], CUTOFF)
    report = build_mes_comparison(
        [{"id": 1, "glass_id": "PANEL-1", "request_time": CUTOFF, "ai_judgment": "OK"}],
        rows, facility="CARUX",
    )
    assert report["records"][0]["mes_judgment"] == "NG"
    assert len(repository.fetch_report_details("PANEL-1", CUTOFF)) == 1


@pytest.mark.parametrize("cutoff,expected", [
    (datetime(2026, 11, 23, 14, 48, 56, 659748), "NG"),
    (datetime(2026, 11, 23, 14, 48, 56, 659749), "NG"),
    (datetime(2026, 11, 23, 14, 48, 56, 659750), "OK"),
])
def test_carux_site_timestamp_example_keeps_microsecond_boundary(carux_oracle, cutoff, expected):
    repository, connection, _ = carux_oracle
    insert_row(connection, time="2026-11-23 14.48.56.659749")
    rows = repository.fetch_defects(["PANEL-1"], cutoff)
    report = build_mes_comparison(
        [{"id": 1, "glass_id": "PANEL-1", "request_time": cutoff, "ai_judgment": "OK"}],
        rows, facility="CARUX",
    )
    details = repository.fetch_report_details("PANEL-1", cutoff)
    assert report["records"][0]["mes_judgment"] == expected
    assert len(details) == (1 if expected == "NG" else 0)


@pytest.mark.parametrize("code,x,y,mes_time", [
    ("PCK21", 10, 20, "2026-11-23 09:00:00"),
    ("PCU12", None, 20, "2026-11-23 09:00:00"),
    ("PCU12", 10, None, "2026-11-23 09:00:00"),
])
def test_carux_each_condition_does_not_exclude_a_record(carux_oracle, code, x, y, mes_time):
    repository, connection, _ = carux_oracle
    insert_row(connection, code=code, x=x, y=y, time=mes_time)
    rows = repository.fetch_defects(["PANEL-1"], CUTOFF)
    report = build_mes_comparison(
        [{"id": 1, "glass_id": "PANEL-1", "request_time": CUTOFF, "ai_judgment": "OK"}],
        rows, facility="CARUX",
    )
    assert report["records"][0]["mes_judgment"] == "NG"
    assert report["records"][0]["qualifying_defect_count"] == 1


def test_carux_empty_code_still_counts_when_the_record_matches(carux_oracle):
    repository, connection, _ = carux_oracle
    insert_row(connection, code=None, time="2026-11-23 08:00:00", x=None, y=None)
    rows = repository.fetch_defects(["PANEL-1"], CUTOFF)
    report = build_mes_comparison(
        [{"id": 1, "glass_id": "PANEL-1", "request_time": CUTOFF, "ai_judgment": "OK"}],
        rows, facility="CARUX",
    )
    assert report["records"][0]["mes_judgment"] == "NG"


def test_carux_time_boundary_and_details_use_same_cutoff(carux_oracle):
    repository, connection, _ = carux_oracle
    insert_row(connection, time="2026-11-23 07:59:59.999999")
    insert_row(connection, time="2026-11-23 08:00:00")
    insert_row(connection, station="1605", time="2026-11-23 08:00:00.000001")
    rows = repository.fetch_defects(["PANEL-1"], CUTOFF)
    report = build_mes_comparison(
        [{"id": 1, "glass_id": "PANEL-1", "request_time": CUTOFF, "ai_judgment": "OK"}],
        rows, facility="CARUX",
    )
    details = repository.fetch_report_details("PANEL-1", CUTOFF)
    assert report["records"][0]["qualifying_defect_count"] == 2
    assert {row["TRANS_DATE"] for row in details} == {
        "2026-11-23 08:00:00", "2026-11-23 08:00:00.000001",
    }


def test_carux_only_older_mes_record_is_ok(carux_oracle):
    repository, connection, _ = carux_oracle
    insert_row(connection, time="2026-11-23 07:59:59")
    rows = repository.fetch_defects(["PANEL-1"], CUTOFF)
    report = build_mes_comparison(
        [{"id": 1, "glass_id": "PANEL-1", "request_time": CUTOFF, "ai_judgment": "NG"}],
        rows, facility="CARUX",
    )
    assert report["records"][0]["mes_judgment"] == "OK"
    assert report["records"][0]["comparison"] == "over_detection"
    assert repository.fetch_report_details("PANEL-1", CUTOFF) == []


def test_carux_no_matching_record_is_ok(carux_oracle):
    repository, connection, _ = carux_oracle
    insert_row(connection, station="1700")
    insert_row(connection, fac="OTHER")
    rows = repository.fetch_defects(["PANEL-1"], CUTOFF)
    report = build_mes_comparison(
        [{"id": 1, "glass_id": "PANEL-1", "request_time": CUTOFF, "ai_judgment": "NG"}],
        rows, facility="CARUX",
    )
    assert rows == {}
    assert report["records"][0]["mes_judgment"] == "OK"
    assert report["records"][0]["comparison"] == "over_detection"


def test_carux_uses_batches_without_duplicating_panels(carux_oracle):
    repository, connection, queries = carux_oracle
    insert_row(connection)
    rows = repository.fetch_defects(["PANEL-1", "panel-1"] + [f"P-{i:04d}" for i in range(900)], CUTOFF)
    assert len(rows["PANEL-1"]) == 1
    assert [sum(name.startswith("panel_") for name in binds) for _, binds in queries] == [900, 1]


def test_carux_summary_api_uses_real_query_and_displays_configured_rule(carux_oracle, monkeypatch):
    import capi_web

    repository, connection, _ = carux_oracle
    insert_row(connection, code="PCK21", time="2026-11-23 08:00:00", x=None, y=None)
    handler = object.__new__(capi_web.CAPIWebHandler)
    handler.db = SimpleNamespace(
        get_mes_comparison_records=lambda *args, **kwargs: [{
            "id": 1, "glass_id": "PANEL-1", "request_time": CUTOFF, "ai_judgment": "OK",
        }],
        get_mes_comparison_reviews=lambda ids: [],
        get_ng_validation_summary=lambda: {},
    )
    handler._capi_server_instance = SimpleNamespace(station_profile="capi", server_config={"mes_report": {}})
    responses = []

    def send_json(data, **kwargs):
        responses.append((data, kwargs))
        return {"serialize_seconds": 0, "compression_seconds": 0, "write_seconds": 0,
                "response_bytes": 0, "uncompressed_bytes": 0, "compressed": False}

    handler._send_json = send_json
    monkeypatch.setattr(capi_mes_report, "OracleMESRepository", lambda *args, **kwargs: repository)
    handler._handle_mes_comparison_api({})
    response = responses[-1][0]
    assert response["success"] is True
    assert response["records"][0]["mes_judgment"] == "NG"
    assert response["summary"]["miss_detection"] == 1
    assert response["rule"] == "FAC_ID=AA02、WORK_CTR=1600/1605、TRANS_DATE>=推論時間，不要求 X/Y，查無資料或未命中判 OK"


def test_carux_uncomparable_mes_time_is_reported_by_summary_and_detail_api(carux_oracle, monkeypatch):
    import capi_web

    repository, connection, _ = carux_oracle
    insert_row(connection, time="not-a-date")
    record = {"id": 1, "glass_id": "PANEL-1", "request_time": CUTOFF, "ai_judgment": "OK"}
    handler = object.__new__(capi_web.CAPIWebHandler)
    handler.db = SimpleNamespace(
        get_mes_comparison_records=lambda *args, **kwargs: [record],
        get_mes_comparison_record=lambda record_id: record,
    )
    handler._capi_server_instance = SimpleNamespace(station_profile="capi", server_config={"mes_report": {}})
    responses = []
    handler._send_json = lambda data, **kwargs: responses.append((data, kwargs))
    monkeypatch.setattr(capi_mes_report, "OracleMESRepository", lambda *args, **kwargs: repository)

    handler._handle_mes_comparison_api({})
    assert responses[-1][0]["success"] is False
    assert "TRANS_DATE" in responses[-1][0]["error"]
    handler._handle_mes_report_detail_api({"record_id": ["1"]})
    assert responses[-1][0]["success"] is False
    assert "TRANS_DATE" in responses[-1][0]["error"]


def test_carux_aapi_profile_is_rejected(monkeypatch):
    monkeypatch.setattr(capi_mes_report, "ORACLE_MES_PASSWORD", "test-password")
    config = {"facility": "CARUX", "oracle": {"user": "test", "tns": {
        "CARUX": {"host": "example.invalid", "service_name": "tnbfmr"},
    }}}
    with pytest.raises(capi_mes_report.MESReportConfigurationError, match="AAPI"):
        OracleMESRepository(config, station_profile="aapi")


def test_carux_empty_detail_does_not_claim_legacy_wp_defthis_columns(carux_oracle, monkeypatch):
    import capi_web

    repository, _, _ = carux_oracle
    handler = object.__new__(capi_web.CAPIWebHandler)
    handler.db = SimpleNamespace(get_mes_comparison_record=lambda record_id: {
        "id": record_id, "glass_id": "PANEL-1", "request_time": CUTOFF,
    })
    handler._capi_server_instance = SimpleNamespace(station_profile="capi", server_config={"mes_report": {}})
    responses = []
    handler._send_json = lambda data, **kwargs: responses.append((data, kwargs))
    monkeypatch.setattr(capi_mes_report, "OracleMESRepository", lambda *args, **kwargs: repository)
    handler._handle_mes_report_detail_api({"record_id": ["1"]})
    assert responses[-1][0]["rows"] == []
    assert responses[-1][0]["columns"] == []


def test_carux_detail_api_displays_carux_rule(carux_oracle, monkeypatch):
    import capi_web

    repository, connection, _ = carux_oracle
    insert_row(connection)
    handler = object.__new__(capi_web.CAPIWebHandler)
    handler.db = SimpleNamespace(get_mes_comparison_record=lambda record_id: {
        "id": record_id, "glass_id": "PANEL-1", "request_time": CUTOFF,
    })
    handler._capi_server_instance = SimpleNamespace(station_profile="capi", server_config={"mes_report": {}})
    responses = []
    handler._send_json = lambda data, **kwargs: responses.append((data, kwargs))
    monkeypatch.setattr(capi_mes_report, "OracleMESRepository", lambda *args, **kwargs: repository)
    handler._handle_mes_report_detail_api({"record_id": ["1"]})
    response = responses[-1][0]
    assert response["success"] is True
    assert response["rows"][0]["ENTITY_NO"] == "PANEL-1"
    assert response["rule"] == "FAC_ID=AA02、WORK_CTR=1600/1605、TRANS_DATE>=推論時間，不要求 X/Y，查無資料或未命中判 OK"
