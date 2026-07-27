from datetime import datetime
import json
import sqlite3
from types import SimpleNamespace

import capi_mes_report
from capi_database import CAPIDatabase
from capi_mes_report import (
    ORACLE_PANEL_BATCH_SIZE,
    WP_DEFTHIS_COLUMNS,
    WP_DEFTHIS_FAC_ID_BY_FACILITY,
    OracleMESRepository,
    build_mes_comparison,
    classify_mes_judgment,
    load_defect_code_catalog,
)


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
    assert result["qualifying_defects"][0]["severity"] == "輕缺"
    assert result["qualifying_defects"][0]["description"] == "BL異物"


def test_defect_code_catalog_contains_excel_reference_values():
    catalog = load_defect_code_catalog()

    assert len(catalog) == 622
    assert catalog["PCDK2"] == {"severity": "僅判等級", "description": "暗點"}
    assert catalog["PCDD4"] == {"severity": "僅判等級", "description": "暗點2連結"}
    assert catalog["PCMD1"] == {"severity": "僅判等級", "description": "全黑畫素漏光"}


def test_unknown_defect_code_remains_reportable_as_unmapped():
    result = classify_mes_judgment(
        [_defect("2026-07-19 10:01:00", code="UNKNOWN-CODE")],
        datetime.fromisoformat("2026-07-19 10:00:00"),
    )

    assert result["judgment"] == "NG"
    assert result["qualifying_defects"][0]["severity"] == ""
    assert result["qualifying_defects"][0]["description"] == ""


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
        "MATCH": [
            {**valid, "pnl_id": "MATCH"},
            {**_defect("2026-07-19 09:05:00", code="PCM02"), "pnl_id": "MATCH"},
        ],
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
    assert [defect["dfct_code"] for defect in report["records"][3]["qualifying_defects"]] == [
        "PCM01",
        "PCM02",
    ]
    assert report["records"][3]["first_defect"]["dfct_code"] == "PCM01"


def test_build_mes_comparison_matches_coordinates_with_inclusive_20_tolerance():
    records = [
        {
            "id": 1,
            "glass_id": "MATCH",
            "ai_judgment": "OK",
            "request_time": "2026-07-19 08:00:00",
            "aoi_machine_coords": json.dumps({
                "G0F00000": [
                    {"defect_code": "PCDK2", "product_x": 720, "product_y": 300},
                ],
                "W0F00000": [
                    {"defect_code": "PCDK2", "product_x": 720, "product_y": 300},
                ],
            }),
        },
        {
            "id": 2,
            "glass_id": "MISS",
            "ai_judgment": "OK",
            "request_time": "2026-07-19 08:00:00",
            "aoi_machine_coords": json.dumps({
                "STANDARD": [
                    {"defect_code": "PCDK2", "product_x": 721, "product_y": 300},
                ],
            }),
        },
        {
            "id": 3,
            "glass_id": "NO-AOI",
            "ai_judgment": "OK",
            "request_time": "2026-07-19 08:00:00",
            "aoi_machine_coords": "",
        },
    ]
    defects = {
        panel_id: [{
            **_defect("2026-07-19 09:00:00", x=700, y=320),
            "pnl_id": panel_id,
        }]
        for panel_id in ("MATCH", "MISS", "NO-AOI")
    }

    report = build_mes_comparison(records, defects)
    matches = {row["id"]: row["coordinate_match"] for row in report["records"]}

    assert matches[1]["status"] == "matched"
    assert matches[1]["matched_count"] == 1
    assert matches[2]["status"] == "unmatched"
    assert matches[3]["status"] == "no_aoi_coordinates"


def test_build_mes_comparison_rotates_capihm_portrait_coordinates():
    record = {
        "id": 1,
        "glass_id": "CAPIHM-PANEL",
        "ai_judgment": "OK",
        "request_time": "2026-07-19 08:00:00",
        "aoi_machine_coords": json.dumps({
            "STANDARD": [
                {"defect_code": "PCDK2", "product_x": 338, "product_y": 360},
            ],
        }),
    }
    defects = {
        "CAPIHM-PANEL": [{
            **_defect("2026-07-19 09:00:00", x=700, y=320),
            "pnl_id": "CAPIHM-PANEL",
        }],
    }

    regular = build_mes_comparison([record], defects)
    capihm = build_mes_comparison([record], defects, host_name="CAPIHM")

    assert regular["records"][0]["coordinate_match"]["status"] == "unmatched"
    match = capihm["records"][0]["coordinate_match"]
    assert match["status"] == "matched"
    assert match["transform"] == "capihm_portrait_clockwise"


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

    for facility, expected_fac_id, expected_dsn, expected_source in (
        ("MOD1", "C", "10.172.3.55:1521/pncmr", "MOD1 / PNCMR / MCRDA1.WP_DEFTHIS"),
        ("MOD2", "E", "10.174.1.79:1521/pnemr", "MOD2 / PNEMR / MERDA1.WP_DEFTHIS"),
    ):
        repository = OracleMESRepository({**base_config, "facility": facility})
        rows = repository.fetch_defects(["PANEL-1"], datetime(2026, 7, 19, 8, 0, 0, 123000))
        assert rows["PANEL-1"][0]["dfct_code"] == "PCM01"
        assert made_dsns[-1] == expected_dsn
        assert repository.source_label == expected_source
        assert repository.wp_defthis_fac_id == expected_fac_id
        assert repository.password == "secret"

    sql, binds = executed[-1]
    assert WP_DEFTHIS_FAC_ID_BY_FACILITY == {"MOD1": "C", "MOD2": "E"}
    assert "FROM MERDA1.WP_DEFTHIS" in sql
    assert "FROM MCRDA1.WP_DEFTHIS" in executed[0][0]
    assert "INDEX(w WP_DEFTHIS_PK)" in sql
    assert "INDEX(w WP_DEFTHIS_PK)" not in executed[0][0]
    assert "FAC_ID = :fac_id" in sql
    assert executed[0][1]["fac_id"] == "C"
    assert binds["fac_id"] == "E"
    assert "DEFT_OPER = :deft_oper" in sql
    assert "IF_NEWER = 'Y'" in sql
    assert "TRANS_DATE >= :min_trans_date" in sql
    assert binds["min_trans_date"] == "2026-07-19 08.00.00.123000"
    assert binds["panel_0"] == "PANEL-1"
    assert binds["deft_oper"] == "1600"


def test_oracle_repository_splits_panels_into_900_item_batches(monkeypatch):
    monkeypatch.setattr(capi_mes_report, "ORACLE_MES_PASSWORD", "secret")
    executed = []

    class FakeCursor:
        def execute(self, sql, binds):
            executed.append((sql, binds))

        def __iter__(self):
            return iter([])

        def close(self):
            pass

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def close(self):
            pass

    fake_oracledb = SimpleNamespace(
        makedsn=lambda host, port, service_name: f"{host}:{port}/{service_name}",
        connect=lambda **kwargs: FakeConnection(),
    )
    monkeypatch.setitem(__import__("sys").modules, "oracledb", fake_oracledb)
    repository = OracleMESRepository({
        "facility": "MOD2",
        "oracle": {
            "user": "MISSELECT",
            "tns": {
                "MOD2": {
                    "host": "10.174.1.79",
                    "port": 1521,
                    "service_name": "pnemr",
                },
            },
        },
    })

    repository.fetch_defects(
        [f"PANEL-{index:04d}" for index in range(901)],
        datetime(2026, 7, 19, 8, 0, 0),
    )

    assert ORACLE_PANEL_BATCH_SIZE == 900
    assert [
        sum(name.startswith("panel_") for name in binds)
        for _, binds in executed
    ] == [900, 1]
    assert all(binds["fac_id"] == "E" for _, binds in executed)


def test_oracle_repository_fetches_all_wp_defthis_columns_on_demand(monkeypatch):
    monkeypatch.setattr(capi_mes_report, "ORACLE_MES_PASSWORD", "secret")
    executed = []
    values = [None] * len(WP_DEFTHIS_COLUMNS)
    values[WP_DEFTHIS_COLUMNS.index("PNL_ID")] = "PANEL-1"
    values[WP_DEFTHIS_COLUMNS.index("TRANS_NBR")] = 14
    values[WP_DEFTHIS_COLUMNS.index("COMMENTS")] = "完整資料"
    values[WP_DEFTHIS_COLUMNS.index("TRANS_DATE")] = "2026-07-19 09.00.00.123000"

    class FakeCursor:
        def execute(self, sql, binds):
            executed.append((sql, binds))

        def __iter__(self):
            return iter([tuple(values)])

        def close(self):
            pass

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def close(self):
            pass

    fake_oracledb = SimpleNamespace(
        makedsn=lambda host, port, service_name: f"{host}:{port}/{service_name}",
        connect=lambda **kwargs: FakeConnection(),
    )
    monkeypatch.setitem(__import__("sys").modules, "oracledb", fake_oracledb)
    repository = OracleMESRepository({
        "facility": "MOD2",
        "oracle": {
            "user": "MISSELECT",
            "tns": {
                "MOD2": {"host": "10.174.1.79", "port": 1521, "service_name": "pnemr"},
            },
        },
    })

    rows = repository.fetch_report_details("PANEL-1", datetime(2026, 7, 19, 8, 0, 0))

    assert len(WP_DEFTHIS_COLUMNS) == 37
    assert list(rows[0]) == list(WP_DEFTHIS_COLUMNS)
    assert rows[0]["PNL_ID"] == "PANEL-1"
    assert rows[0]["TRANS_NBR"] == 14
    assert rows[0]["COMMENTS"] == "完整資料"
    sql, binds = executed[0]
    assert "SELECT FAC_ID, PNL_ID, TRANS_NBR" in " ".join(sql.split())
    assert "RELAX_FLAG, RELAX_DESCRIPTION" in " ".join(sql.split())
    assert "FAC_ID = :fac_id" in sql
    assert "PNL_ID = :panel_id" in sql
    assert "DEFT_OPER = :deft_oper" in sql
    assert "IF_NEWER = 'Y'" in sql
    assert "TRANS_DATE >= :min_trans_date" in sql
    assert binds == {
        "fac_id": "E",
        "panel_id": "PANEL-1",
        "deft_oper": "1600",
        "min_trans_date": "2026-07-19 08.00.00.000000",
    }


def test_oracle_repository_uses_actual_mod1_wp_defthis_columns(monkeypatch):
    monkeypatch.setattr(capi_mes_report, "ORACLE_MES_PASSWORD", "secret")
    executed = []
    columns = ("PNL_ID", "TRANS_NBR", "COMMENTS", "TRANS_DATE")
    values = ("PANEL-1", 14, "MOD1資料", "2026-07-19 09.00.00.123000")

    class FakeCursor:
        description = [(column,) for column in columns]

        def execute(self, sql, binds):
            executed.append((sql, binds))

        def __iter__(self):
            return iter([values])

        def close(self):
            pass

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def close(self):
            pass

    fake_oracledb = SimpleNamespace(
        makedsn=lambda host, port, service_name: f"{host}:{port}/{service_name}",
        connect=lambda **kwargs: FakeConnection(),
    )
    monkeypatch.setitem(__import__("sys").modules, "oracledb", fake_oracledb)
    repository = OracleMESRepository({
        "facility": "MOD1",
        "oracle": {
            "user": "MISSELECT",
            "tns": {
                "MOD1": {"host": "10.172.3.55", "port": 1521, "service_name": "pncmr"},
            },
        },
    })

    rows = repository.fetch_report_details("PANEL-1", datetime(2026, 7, 19, 8, 0, 0))

    assert list(rows[0]) == list(columns)
    assert rows[0]["COMMENTS"] == "MOD1資料"
    sql, binds = executed[0]
    assert "SELECT *" in " ".join(sql.split())
    assert "FROM MCRDA1.WP_DEFTHIS" in sql
    assert "SYS_TRANS_FLAG" not in sql
    assert "FAC_ID = :fac_id" in sql
    assert binds["fac_id"] == "C"
    assert binds["panel_id"] == "PANEL-1"


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
            request_time TEXT,
            aoi_machine_coords TEXT
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
    statements = []
    conn.set_trace_callback(statements.append)
    db = CAPIDatabase.__new__(CAPIDatabase)
    db._get_conn = lambda: conn

    rows = db.get_mes_comparison_records("2026-07-19", "2026-07-19")

    assert [row["glass_id"] for row in rows] == ["END", "START"]
    assert not any("NOT INDEXED" in statement for statement in statements)


def test_mes_comparison_records_use_full_scan_for_30_day_range():
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
            request_time TEXT,
            aoi_machine_coords TEXT
        )
    """)
    statements = []
    conn.set_trace_callback(statements.append)
    db = CAPIDatabase.__new__(CAPIDatabase)
    db._get_conn = lambda: conn

    rows = db.get_mes_comparison_records("2026-06-25", "2026-07-24")

    assert rows == []
    assert any(
        "FROM inference_records NOT INDEXED" in statement
        for statement in statements
    )


def test_mes_comparison_datetime_index_supports_range_and_order():
    uri = "file:mes_index_test?mode=memory&cache=shared"
    keeper = sqlite3.connect(uri, uri=True)
    keeper.row_factory = sqlite3.Row
    db = CAPIDatabase.__new__(CAPIDatabase)

    def connect():
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    db._get_conn = connect
    db._init_db()
    try:
        indexes = {
            row["name"]
            for row in keeper.execute("PRAGMA index_list('inference_records')")
        }
        plan = keeper.execute(
            """EXPLAIN QUERY PLAN
               SELECT id
                 FROM inference_records
                WHERE datetime(request_time) >= datetime(?)
                  AND datetime(request_time) < datetime(?)
                ORDER BY datetime(request_time) DESC, id DESC""",
            ("2026-06-25 07:30:00", "2026-07-25 07:30:00"),
        ).fetchall()
    finally:
        keeper.close()

    assert "idx_records_request_time_dt" in indexes
    assert any(
        "SEARCH inference_records USING INDEX idx_records_request_time_dt" in row["detail"]
        for row in plan
    )


def test_mes_comparison_records_can_ignore_aoi_ok():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE inference_records (
            id INTEGER PRIMARY KEY,
            glass_id TEXT,
            model_id TEXT,
            machine_no TEXT,
            machine_judgment TEXT,
            ai_judgment TEXT,
            image_dir TEXT,
            request_time TEXT,
            aoi_machine_coords TEXT
        )
    """)
    conn.executemany(
        """INSERT INTO inference_records
           (glass_id, model_id, machine_no, machine_judgment, ai_judgment, image_dir, request_time)
           VALUES (?, 'MODEL', 'M1', ?, 'OK', '/images', '2026-07-19 08:00:00')""",
        [
            ("AOI-OK", "OK"),
            ("AOI-OK-SPACED", " ok "),
            ("AOI-NG", "NG"),
            ("AOI-UNKNOWN", ""),
        ],
    )
    db = CAPIDatabase.__new__(CAPIDatabase)
    db._get_conn = lambda: conn

    rows = db.get_mes_comparison_records(
        "2026-07-19",
        "2026-07-19",
        ignore_aoi_ok=True,
    )

    assert {row["glass_id"] for row in rows} == {"AOI-NG", "AOI-UNKNOWN"}


def test_mes_comparison_records_can_filter_by_panel_id_case_insensitive_partial_match():
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
            request_time TEXT,
            aoi_machine_coords TEXT
        )
    """)
    conn.executemany(
        """INSERT INTO inference_records
           (glass_id, model_id, machine_no, ai_judgment, image_dir, request_time)
           VALUES (?, 'MODEL', 'M1', 'OK', '/images', '2026-07-19 08:00:00')""",
        [("PANEL-ABC-001",), ("PANEL-XYZ-002",)],
    )
    db = CAPIDatabase.__new__(CAPIDatabase)
    db._get_conn = lambda: conn

    rows = db.get_mes_comparison_records("2026-07-19", "2026-07-19", panel_id="abc")

    assert [row["glass_id"] for row in rows] == ["PANEL-ABC-001"]


def test_get_mes_comparison_record_uses_server_side_inference_id():
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
    conn.execute(
        """INSERT INTO inference_records
           (id, glass_id, model_id, machine_no, ai_judgment, image_dir, request_time)
           VALUES (7, 'PANEL-7', 'MODEL', 'M1', 'OK', '/images', '2026-07-19 08:01:02')"""
    )
    db = CAPIDatabase.__new__(CAPIDatabase)
    db._get_conn = lambda: conn

    row = db.get_mes_comparison_record(7)

    assert row == {
        "id": 7,
        "glass_id": "PANEL-7",
        "model_id": "MODEL",
        "machine_no": "M1",
        "ai_judgment": "OK",
        "image_dir": "/images",
        "request_time": "2026-07-19 08:01:02",
    }
