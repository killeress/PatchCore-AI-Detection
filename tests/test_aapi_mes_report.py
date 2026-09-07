from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

import capi_mes_report
from capi_mes_report import build_mes_comparison


CUTOFF = datetime(2026, 9, 5, 7, 50)


def defect(code="PCU10", *, x=753, y=938, trans_date=None, rework_date=None):
    return {
        "pnl_id": "PANEL-1",
        "dfct_code": code,
        "trans_date": trans_date or CUTOFF + timedelta(minutes=2),
        "rework_trans_date": rework_date,
        "x_axis": x,
        "y_axis": y,
    }


def compare(rows, *, profile="aapi", cutoff=CUTOFF, ai="OK"):
    return build_mes_comparison(
        [{"id": 1, "glass_id": "PANEL-1", "request_time": cutoff, "ai_judgment": ai}],
        {"PANEL-1": rows},
        station_profile=profile,
    )


@pytest.mark.parametrize("code,x,y,expected", [
    ("PCU10", 753, 938, "NG"),
    ("PCU10", None, " ", "NG"),
    ("PCP02", None, None, "NG"),
    ("PCZCC", None, None, "NG"),
    ("PCK21", 753, 938, "OK"),
    ("UNKNOWN-CODE", 753, 938, "OK"),
    ("PCM01", 753, 938, "OK"),
])
def test_aapi_uses_screen_codes_without_requiring_coordinates(code, x, y, expected):
    report = compare([defect(code, x=x, y=y, rework_date=CUTOFF + timedelta(minutes=3))])
    assert report["records"][0]["mes_judgment"] == expected


@pytest.mark.parametrize("rework_date", [None, CUTOFF - timedelta(seconds=1), CUTOFF])
def test_aapi_requires_rework_strictly_after_each_request(rework_date):
    assert compare([defect(rework_date=rework_date)])["records"][0]["mes_judgment"] == "OK"


@pytest.mark.parametrize("trans_date", [CUTOFF - timedelta(seconds=1), CUTOFF])
def test_aapi_excludes_defects_at_or_before_request(trans_date):
    row = defect(trans_date=trans_date, rework_date=CUTOFF + timedelta(minutes=3))
    assert compare([row])["records"][0]["mes_judgment"] == "OK"


def test_aapi_any_matching_code_counts_and_empty_results_are_ok():
    rows = [defect("UNKNOWN-CODE"), defect("PCP02", rework_date=CUTOFF + timedelta(minutes=3))]
    report = compare(rows)
    assert report["summary"]["miss_detection"] == 1
    assert report["records"][0]["qualifying_defect_count"] == 1
    assert report["records"][0]["first_defect"]["description"] == "下偏光板來料不良"
    assert compare([], ai="NG")["summary"]["over_detection"] == 1


def test_aapi_repeated_panel_checks_rework_against_each_request():
    rows = [defect(trans_date=CUTOFF + timedelta(minutes=5), rework_date=CUTOFF + timedelta(minutes=1))]
    assert compare(rows)["records"][0]["mes_judgment"] == "NG"
    assert compare(rows, cutoff=CUTOFF + timedelta(minutes=2))["records"][0]["mes_judgment"] == "OK"


def test_capi_keeps_coordinate_filter_unknown_codes_and_inclusive_time():
    assert compare([defect("UNKNOWN-CODE", trans_date=CUTOFF)], profile="capi")["records"][0]["mes_judgment"] == "NG"
    assert compare([defect(x=None)], profile="capi")["records"][0]["mes_judgment"] == "OK"
    assert compare([defect("PCK21")], profile="capi")["records"][0]["mes_judgment"] == "OK"


def test_aapi_catalog_matches_customer_count_and_missing_descriptions():
    path = Path(__file__).resolve().parents[1] / "configs" / "aapi_mes_defect_codes.json"
    codes = json.loads(path.read_text(encoding="utf-8"))
    assert len(codes) == 85
    assert codes["PCP02"] == "下偏光板來料不良"
    assert codes["PCZCC"] == "無down grade機種panel報廢"
    assert "PCK21" not in codes


@pytest.fixture
def oracle(monkeypatch):
    """Execute the repository SQL against small relational fixtures without MES access."""
    connection = sqlite3.connect(":memory:")
    connection.execute("ATTACH DATABASE ':memory:' AS MERDA1")
    connection.execute("CREATE TABLE MERDA1.WP_DEFTHIS (" + ", ".join(
        f"{column} TEXT" for column in capi_mes_report.WP_DEFTHIS_COLUMNS
    ) + ")")
    connection.execute("CREATE TABLE MERDA1.WP_PNLHIST (FAC_ID TEXT, PNL_ID TEXT, OPER TEXT, TO_OPER TEXT, TRANS_DATE TEXT)")
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

    monkeypatch.setattr(capi_mes_report, "ORACLE_MES_PASSWORD", "test")
    monkeypatch.setitem(sys.modules, "oracledb", SimpleNamespace(
        makedsn=lambda *args, **kwargs: "test",
        connect=lambda **kwargs: SimpleNamespace(cursor=Cursor, close=lambda: None),
    ))
    repository = capi_mes_report.OracleMESRepository({
        "facility": "MOD2",
        "oracle": {"user": "test", "tns": {"MOD2": {"host": "test", "service_name": "pnemr"}}},
    }, station_profile="aapi")
    yield repository, connection, queries
    connection.close()


def add_defect(connection, panel="PANEL-1", *, code="PCU10", oper="1400", newer="Y", minute=2, deft_minute=None, fac="E"):
    trans_date = (CUTOFF + timedelta(minutes=minute)).strftime("%Y-%m-%d %H.%M.%S.%f")
    deft_date = (CUTOFF + timedelta(minutes=minute if deft_minute is None else deft_minute)).strftime("%Y-%m-%d %H.%M.%S.%f")
    connection.execute(
        "INSERT INTO MERDA1.WP_DEFTHIS (FAC_ID, PNL_ID, DFCT_CODE, DEFT_OPER, IF_NEWER, TRANS_DATE, DEFT_DATE) VALUES (?,?,?,?,?,?,?)",
        (fac, panel, code, oper, newer, trans_date, deft_date),
    )


def add_rework(connection, panel="PANEL-1", *, oper="1400", to_oper="2100", minute=3, fac="E"):
    connection.execute("INSERT INTO MERDA1.WP_PNLHIST VALUES (?,?,?,?,?)", (
        fac, panel, oper, to_oper, (CUTOFF + timedelta(minutes=minute)).strftime("%Y-%m-%d %H.%M.%S.%f"),
    ))


def test_aapi_query_and_details_use_rework_gate_and_defect_date_order(oracle):
    repository, connection, _ = oracle
    add_rework(connection, minute=3)
    add_rework(connection, minute=4)
    add_defect(connection, code="PCU10", minute=2, deft_minute=2)
    add_defect(connection, code="PCP02", minute=3, deft_minute=1)
    add_defect(connection, code="UNKNOWN-CODE", minute=4)
    rows = repository.fetch_defects(["PANEL-1"], CUTOFF)["PANEL-1"]
    assert [row["dfct_code"] for row in rows] == ["PCP02", "PCU10", "UNKNOWN-CODE"]
    assert all(row["rework_trans_date"] == "2026-09-05 07.54.00.000000" for row in rows)
    report = compare(rows)
    assert report["records"][0]["mes_row_count"] == 3
    assert report["records"][0]["qualifying_defect_count"] == 2
    details = repository.fetch_report_details("PANEL-1", CUTOFF)
    assert [row["DFCT_CODE"] for row in details] == ["PCP02", "PCU10", "UNKNOWN-CODE"]
    assert "WP_PNLHIST" in repository.source_label
    assert "1400→2100" in repository.rule_label


@pytest.mark.parametrize("rework_kwargs,defect_kwargs", [
    (None, {}),
    ({"oper": "1600"}, {}),
    ({"to_oper": "1500"}, {}),
    ({"minute": 0}, {}),
    ({"fac": "C"}, {}),
    ({"panel": "OTHER"}, {}),
    ({}, {"oper": "1600"}),
    ({}, {"newer": "N"}),
    ({}, {"minute": 0}),
    ({}, {"minute": -1}),
    ({}, {"fac": "C"}),
])
def test_aapi_query_and_details_reject_unrelated_or_old_mes_data(oracle, rework_kwargs, defect_kwargs):
    repository, connection, _ = oracle
    add_defect(connection, **defect_kwargs)
    if rework_kwargs is not None:
        add_rework(connection, **rework_kwargs)
    assert repository.fetch_defects(["PANEL-1"], CUTOFF) == {}
    assert repository.fetch_report_details("PANEL-1", CUTOFF) == []


def test_aapi_batch_and_single_record_details_agree_for_repeated_panel(oracle):
    repository, connection, _ = oracle
    add_rework(connection, minute=1)
    add_defect(connection, minute=5)
    rows = repository.fetch_defects(["PANEL-1"], CUTOFF)["PANEL-1"]
    later = CUTOFF + timedelta(minutes=2)
    assert compare(rows, cutoff=later)["records"][0]["mes_row_count"] == 0
    assert repository.fetch_report_details("PANEL-1", later) == []


def test_aapi_query_batches_and_does_not_duplicate_panels(oracle):
    repository, connection, queries = oracle
    add_rework(connection)
    add_defect(connection)
    rows = repository.fetch_defects(["PANEL-1", "PANEL-1"] + [f"P-{i}" for i in range(900)], CUTOFF)
    assert len(rows["PANEL-1"]) == 1
    assert [sum(key.startswith("panel_") for key in binds) for _, binds in queries] == [900, 1]


def test_aapi_codes_are_in_full_and_patch_deployment():
    from scripts.build_deploy_zip import CODE_FILES, _is_patch_deploy_file

    path = "configs/aapi_mes_defect_codes.json"
    assert path in CODE_FILES
    assert _is_patch_deploy_file(path)


@pytest.mark.parametrize("server,hostname,expected", [
    (SimpleNamespace(station_profile="aapi"), "CAPI01", "aapi"),
    (SimpleNamespace(station_profile="capi"), "AAPI01", "capi"),
    (SimpleNamespace(station_adapter=SimpleNamespace(profile="aapi")), "CAPI01", "aapi"),
    (None, "AAPI09", "aapi"),
    (None, "CAPIHM", "capi"),
    (None, "DEV-PC", "capi"),
])
def test_mes_api_uses_server_station_with_hostname_fallback(monkeypatch, server, hostname, expected):
    import capi_web

    monkeypatch.setattr(capi_web, "_get_host_identity", lambda: hostname)
    assert capi_web.CAPIWebHandler._mes_report_station_profile(server) == expected


@pytest.mark.parametrize("profile,oper,expected", [("aapi", "1400", "NG"), ("capi", "1600", "OK")])
def test_mes_summary_and_detail_api_use_the_same_station_rules(oracle, monkeypatch, profile, oper, expected):
    import capi_web

    repository, connection, queries = oracle
    add_rework(connection)
    add_defect(connection, oper=oper)
    record = {"id": 1, "glass_id": "PANEL-1", "request_time": CUTOFF, "ai_judgment": "OK"}
    config = {
        "facility": "MOD2",
        "oracle": {"user": "test", "tns": {"MOD2": {"host": "test", "service_name": "pnemr"}}},
    }
    handler = object.__new__(capi_web.CAPIWebHandler)
    handler._capi_server_instance = SimpleNamespace(station_profile=profile, server_config={"mes_report": config})
    handler.db = SimpleNamespace(
        get_mes_comparison_records=lambda *args, **kwargs: [record],
        get_mes_comparison_record=lambda record_id: record,
        get_mes_comparison_reviews=lambda ids: [],
        get_ng_validation_summary=lambda: {},
    )
    responses = []

    def send_json(data, **kwargs):
        responses.append(data)
        return {"serialize_seconds": 0, "compression_seconds": 0, "write_seconds": 0,
                "response_bytes": 0, "uncompressed_bytes": 0, "compressed": False}

    handler._send_json = send_json
    monkeypatch.setattr(capi_web, "_get_host_identity", lambda: profile.upper() + "01")
    handler._handle_mes_comparison_api({})
    summary = responses[-1]
    assert summary["success"] is True
    assert summary["station_profile"] == profile
    assert summary["records"][0]["mes_judgment"] == expected
    assert f"DEFT_OPER={oper}" in summary["rule"]
    handler._handle_mes_report_detail_api({"record_id": ["1"]})
    details = responses[-1]
    assert details["success"] is True
    assert details["station_profile"] == profile
    assert details["source"] == summary["source"]
    assert details["rows"][0]["DEFT_OPER"] == oper
    assert f"DEFT_OPER={oper}" in details["rule"]
    assert all(binds["deft_oper"] == oper for _, binds in queries)


def test_aapi_missing_code_file_fails_instead_of_classifying_everything_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(capi_mes_report, "AAPI_DEFECT_CODE_CATALOG_PATH", tmp_path / "missing.json")
    capi_mes_report.load_aapi_defect_codes.cache_clear()
    try:
        with pytest.raises(FileNotFoundError):
            compare([defect(rework_date=CUTOFF + timedelta(minutes=3))])
    finally:
        capi_mes_report.load_aapi_defect_codes.cache_clear()
