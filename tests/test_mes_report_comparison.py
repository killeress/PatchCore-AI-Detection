from datetime import datetime
import sqlite3
from types import SimpleNamespace

import capi_mes_report
from capi_database import CAPIDatabase
from capi_mes_report import OracleMESRepository, build_mes_comparison, classify_mes_judgment


def _defect(
    trans_date: str,
    *,
    code: str = "PCM01",
    x=100,
    y=200,
):
    return {
        "pnl_id": "PANEL-1",
        "dfct_code": code,
        "trans_date": trans_date,
        "x_axis": x,
        "y_axis": y,
    }


def test_mes_judgment_follows_logic_workbook_filters():
    cutoff = datetime.fromisoformat("2026-07-19 10:00:00")

    rows = [
        _defect("2026-07-19 09:59:59"),
        _defect("2026-07-19 10:01:00", code="PCK21"),
        _defect("2026-07-19 10:02:00", x=None),
        _defect("2026-07-19 10:03:00", y=" "),
    ]
    result = classify_mes_judgment(rows, cutoff)

    assert result["judgment"] == "OK"
    assert result["qualifying_defects"] == []

    rows.append(_defect("2026-07-19 10:04:00", code="PCU12", x=0, y=0))
    result = classify_mes_judgment(rows, cutoff)

    assert result["judgment"] == "NG"
    assert result["qualifying_defects"][0]["dfct_code"] == "PCU12"


def test_build_mes_comparison_calculates_over_and_miss_rates():
    records = [
        {"id": 1, "glass_id": "OVER", "ai_judgment": "NG", "request_time": "2026-07-19 08:00:00"},
        {"id": 2, "glass_id": "MISS", "ai_judgment": "OK", "request_time": "2026-07-19 08:00:00"},
        {"id": 3, "glass_id": "MISS-I", "ai_judgment": "OK-i", "request_time": "2026-07-19 08:00:00"},
        {"id": 4, "glass_id": "MATCH", "ai_judgment": "NG:W0F00000", "request_time": "2026-07-19 08:00:00"},
        {"id": 5, "glass_id": "ERROR", "ai_judgment": "ERR:HY", "request_time": "2026-07-19 08:00:00"},
    ]
    valid = _defect("2026-07-19 09:00:00")
    defects = {
        "MISS": [{**valid, "pnl_id": "MISS"}],
        "MISS-I": [{**valid, "pnl_id": "MISS-I"}],
        "MATCH": [{**valid, "pnl_id": "MATCH"}],
    }

    report = build_mes_comparison(records, defects)

    assert report["summary"] == {
        "total": 4,
        "correct": 1,
        "over_detection": 1,
        "miss_detection": 2,
        "accuracy_rate": 25.0,
        "over_detection_rate": 25.0,
        "miss_detection_rate": 50.0,
        "uncomparable": 1,
    }
    assert [row["comparison"] for row in report["records"]] == [
        "over_detection",
        "miss_detection",
        "miss_detection",
        "correct",
        "uncomparable",
    ]


def test_oracle_repository_selects_tns_by_equipment_facility(monkeypatch):
    monkeypatch.setattr(capi_mes_report, "ORACLE_MES_PASSWORD", "secret")
    executed = []
    made_dsns = []

    class FakeCursor:
        def execute(self, sql, binds):
            executed.append((sql, binds))

        def __iter__(self):
            return iter([("PANEL-1", "PCM01", datetime(2026, 7, 19, 9), 10, 20)])

        def close(self):
            pass

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def close(self):
            pass

    fake_oracledb = SimpleNamespace(
        makedsn=lambda host, port, service_name: made_dsns.append(f"{host}:{port}/{service_name}") or made_dsns[-1],
        connect=lambda **kwargs: FakeConnection(),
    )
    monkeypatch.setitem(__import__("sys").modules, "oracledb", fake_oracledb)
    base_config = {
        "oracle": {
            "user": "MISSELECT",
            "tns": {
                "MOD1": {"host": "10.172.3.55", "port": 1521, "service_name": "pncmr"},
                "MOD2": {"host": "10.174.1.79", "port": 1521, "service_name": "pnemr"},
            },
        },
    }

    for facility, expected_dsn, expected_source in (
        ("MOD1", "10.172.3.55:1521/pncmr", "MOD1 / PNCMR / MERDA1.WP_DEFTHIS"),
        ("MOD2", "10.174.1.79:1521/pnemr", "MOD2 / PNEMR / MERDA1.WP_DEFTHIS"),
    ):
        repository = OracleMESRepository({**base_config, "facility": facility})
        rows = repository.fetch_defects(["PANEL-1"], datetime(2026, 7, 19, 8, 0, 0, 123000))
        assert rows["PANEL-1"][0]["dfct_code"] == "PCM01"
        assert made_dsns[-1] == expected_dsn
        assert repository.source_label == expected_source
        assert repository.password == "secret"

    sql, binds = executed[-1]
    assert "FROM MERDA1.WP_DEFTHIS" in sql
    assert "DEFT_OPER = :deft_oper" in sql
    assert "IF_NEWER = 'Y'" in sql
    assert "TRANS_DATE >= :min_trans_date" in sql
    assert binds["min_trans_date"] == "2026-07-19 08.00.00.123000"
    assert binds["panel_0"] == "PANEL-1"
    assert binds["deft_oper"] == "1600"


def test_mes_comparison_records_use_factory_day_window():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE inference_records (
            id INTEGER PRIMARY KEY,
            glass_id TEXT,
            model_id TEXT,
            machine_no TEXT,
            ai_judgment TEXT,
            image_dir TEXT,
            request_time TEXT
        )
    """)
    conn.executemany(
        """INSERT INTO inference_records
           (glass_id, model_id, machine_no, ai_judgment, image_dir, request_time)
           VALUES (?, 'MODEL', 'M1', 'OK', '/images', ?)""",
        [
            ("BEFORE", "2026-07-19 07:29:59"),
            ("START", "2026-07-19 07:30:00"),
            ("END", "2026-07-20 07:29:59"),
            ("AFTER", "2026-07-20 07:30:00"),
        ],
    )
    db = CAPIDatabase.__new__(CAPIDatabase)
    db._get_conn = lambda: conn

    rows = db.get_mes_comparison_records("2026-07-19", "2026-07-19")

    assert [row["glass_id"] for row in rows] == ["END", "START"]
