#!/usr/bin/env python3
"""唯讀診斷 MES Oracle 欄位型態與 ORA-01722 bind 問題。"""

from __future__ import annotations

import argparse
import logging
import math
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import oracledb
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capi_mes_credentials import ORACLE_MES_PASSWORD  # noqa: E402
from capi_database import _factory_day_end_ts, _factory_day_start_ts  # noqa: E402
from capi_mes_report import (  # noqa: E402
    ORACLE_PANEL_BATCH_SIZE,
    OracleMESRepository,
    WP_DEFTHIS_FAC_ID_BY_FACILITY,
    WP_DEFTHIS_INDEX_HINT_BY_FACILITY,
    WP_DEFTHIS_SCHEMA_BY_FACILITY,
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def run_test(cursor, name: str, sql: str, binds: dict) -> None:
    print(f"\n=== Test: {name} ===")
    try:
        cursor.execute(sql, binds)
        print("PASS:", cursor.fetchone())
    except Exception as exc:
        print("FAIL:", repr(exc))


def latest_panel_id(config: dict) -> str:
    db_path = str(config.get("database", {}).get("path") or "").strip()
    if not db_path:
        return ""

    try:
        with sqlite3.connect(db_path) as connection:
            row = connection.execute(
                """
                SELECT glass_id, request_time
                FROM inference_records
                WHERE TRIM(COALESCE(glass_id, '')) <> ''
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
    except Exception as exc:
        print("\nLocal SQLite lookup failed:", repr(exc))
        return ""

    if not row:
        return ""

    print("\nLatest local panel:", row[0])
    print("Request time:", row[1])
    return str(row[0]).strip()


def print_index_metadata(cursor, schema: str) -> None:
    print(f"\n=== Visible indexes: {schema}.WP_DEFTHIS ===")
    try:
        cursor.execute(
            """
            SELECT i.index_name, i.status, i.uniqueness,
                   c.column_position, c.column_name
            FROM ALL_INDEXES i
            JOIN ALL_IND_COLUMNS c
              ON c.index_owner = i.owner
             AND c.index_name = i.index_name
            WHERE i.table_owner = :owner
              AND i.table_name = 'WP_DEFTHIS'
            ORDER BY i.index_name, c.column_position
            """,
            {"owner": schema},
        )
        rows = cursor.fetchall()
    except Exception as exc:
        print("Index metadata unavailable:", repr(exc))
        return

    if not rows:
        print("No visible index metadata. The SELECT account may not have catalog visibility.")
        return

    indexes = {}
    for index_name, status, uniqueness, _, column_name in rows:
        info = indexes.setdefault(
            str(index_name),
            {"status": status, "uniqueness": uniqueness, "columns": []},
        )
        info["columns"].append(str(column_name))

    for index_name, info in indexes.items():
        print(
            f"{index_name}: status={info['status']}, uniqueness={info['uniqueness']}, "
            f"columns={', '.join(info['columns'])}"
        )

    matching = [
        name
        for name, info in indexes.items()
        if info["columns"]
        and info["columns"][0] == "FAC_ID"
        and any(column in info["columns"][1:] for column in ("PNL_ID", "TRANS_DATE"))
    ]
    if matching:
        print("Report-oriented index candidate visible:", ", ".join(matching))
    else:
        print(
            "WARNING: no visible index starts with FAC_ID and also contains "
            "PNL_ID or TRANS_DATE."
        )

    try:
        cursor.execute(
            """
            SELECT num_rows, blocks, last_analyzed
            FROM ALL_TABLES
            WHERE owner = :owner
              AND table_name = 'WP_DEFTHIS'
            """,
            {"owner": schema},
        )
        row = cursor.fetchone()
        print("Table statistics:", row if row else "not visible")
    except Exception as exc:
        print("Table statistics unavailable:", repr(exc))


def load_benchmark_panels(
    config: dict,
    start_date: str,
    end_date: str,
) -> tuple[list[str], datetime]:
    db_path = str(config.get("database", {}).get("path") or "").strip()
    if not db_path:
        raise RuntimeError("server_config.yaml 缺少 database.path")

    start_ts = _factory_day_start_ts(start_date)
    end_ts = _factory_day_end_ts(end_date)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT UPPER(TRIM(glass_id)) AS panel_id,
                   MIN(datetime(request_time)) AS first_request_time
            FROM inference_records
            WHERE datetime(request_time) >= datetime(?)
              AND datetime(request_time) < datetime(?)
              AND TRIM(COALESCE(glass_id, '')) <> ''
            GROUP BY UPPER(TRIM(glass_id))
            ORDER BY panel_id
            """,
            (start_ts, end_ts),
        ).fetchall()

    if not rows:
        raise RuntimeError(f"本機 SQLite 在 {start_date}..{end_date} 沒有 PANEL")

    panel_ids = [str(row[0]) for row in rows]
    valid_times = [
        datetime.fromisoformat(str(row[1]))
        for row in rows
        if row[1]
    ]
    if not valid_times:
        raise RuntimeError("本機 SQLite PANEL 都缺少有效 request_time")
    return panel_ids, min(valid_times)


def _sampled_batch_indexes(
    total_batches: int,
    max_batches: int,
) -> list[int]:
    if max_batches <= 0 or max_batches >= total_batches:
        return list(range(total_batches))
    if max_batches == 1:
        return [total_batches // 2]
    return sorted({
        round(index * (total_batches - 1) / (max_batches - 1))
        for index in range(max_batches)
    })


def benchmark_report_query(
    connection,
    schema: str,
    panel_ids: list[str],
    min_trans_date: datetime,
    *,
    batch_size: int,
    max_batches: int,
    fac_id: str,
    index_hint: str,
) -> None:
    total_batches = math.ceil(len(panel_ids) / batch_size)
    batch_indexes = _sampled_batch_indexes(total_batches, max_batches)

    print("\n=== Read-only Report query benchmark ===")
    print("Unique panels:", len(panel_ids))
    print("Application batch size:", batch_size)
    print("Total application batches:", total_batches)
    print("Sampled batch indexes:", ", ".join(str(index + 1) for index in batch_indexes))
    print("Global min TRANS_DATE:", min_trans_date)
    print("FAC_ID filter:", repr(fac_id))

    timings = []
    for batch_index in batch_indexes:
        offset = batch_index * batch_size
        chunk = panel_ids[offset:offset + batch_size]
        panel_binds = {f"panel_{index}": value for index, value in enumerate(chunk)}
        placeholders = ", ".join(f":panel_{index}" for index in range(len(chunk)))
        select_hint = index_hint or "/* MES_REPORT_DIAG */"
        sql = f"""
            SELECT {select_hint}
                   w.PNL_ID, w.DFCT_CODE, w.TRANS_DATE, w.X_AXIS, w.Y_AXIS
            FROM {schema}.WP_DEFTHIS w
            WHERE w.FAC_ID = :fac_id
              AND w.DEFT_OPER = :deft_oper
              AND w.IF_NEWER = 'Y'
              AND w.TRANS_DATE >= :min_trans_date
              AND w.PNL_ID IN ({placeholders})
            ORDER BY w.PNL_ID, w.TRANS_DATE
        """
        cursor = connection.cursor()
        cursor.arraysize = 1000
        cursor.prefetchrows = 1000
        started = time.monotonic()
        try:
            cursor.execute(
                sql,
                {
                    "fac_id": fac_id,
                    "deft_oper": "1600",
                    "min_trans_date": min_trans_date.strftime("%Y-%m-%d %H.%M.%S.%f"),
                    **panel_binds,
                },
            )
            row_count = sum(1 for _ in cursor)
        finally:
            cursor.close()
        elapsed = time.monotonic() - started
        timings.append(elapsed)
        print(
            f"Batch {batch_index + 1}/{total_batches}: "
            f"panels={len(chunk)}, rows={row_count}, elapsed={elapsed:.2f}s"
        )

    average = sum(timings) / len(timings)
    print(
        "Sample timing seconds: "
        f"min={min(timings):.2f}, avg={average:.2f}, max={max(timings):.2f}"
    )
    print(f"Estimated sequential full range: {average * total_batches:.2f}s")

    print("\n=== Last cursor execution plan (best effort) ===")
    plan_cursor = connection.cursor()
    try:
        plan_cursor.execute(
            """
            SELECT plan_table_output
            FROM TABLE(DBMS_XPLAN.DISPLAY_CURSOR(NULL, NULL, 'BASIC +PREDICATE'))
            """
        )
        plan_rows = plan_cursor.fetchall()
        if plan_rows:
            for row in plan_rows:
                print(row[0])
        else:
            print("No execution plan returned")
    except Exception as exc:
        print("Execution plan unavailable with this SELECT account:", repr(exc))
    finally:
        plan_cursor.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="server_config.yaml",
        help="Server YAML path (default: server_config.yaml)",
    )
    parser.add_argument(
        "--benchmark-days",
        type=int,
        default=0,
        help="Benchmark latest N production days; 0 disables benchmark (default: 0)",
    )
    parser.add_argument(
        "--benchmark-start",
        help="Benchmark start date YYYY-MM-DD; use with --benchmark-end",
    )
    parser.add_argument(
        "--benchmark-end",
        help="Benchmark end date YYYY-MM-DD; use with --benchmark-start",
    )
    parser.add_argument(
        "--benchmark-batches",
        type=int,
        default=3,
        help="Sample evenly distributed application batches; 0 runs all",
    )
    parser.add_argument(
        "--benchmark-app-query",
        action="store_true",
        help="Run the exact OracleMESRepository query path used by the Web API",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Oracle round-trip timeout per benchmark batch (default: 60)",
    )
    parser.add_argument(
        "--fac-id",
        help="Override FAC_ID; default: MOD1=C, MOD2=E",
    )
    parser.add_argument(
        "--auto-fac-id",
        action="store_true",
        help="Compatibility option; uses the fixed facility FAC_ID without querying Oracle",
    )
    args = parser.parse_args()
    if bool(args.benchmark_start) != bool(args.benchmark_end):
        parser.error("--benchmark-start 與 --benchmark-end 必須一起提供")
    if args.benchmark_days < 0 or args.benchmark_batches < 0:
        parser.error("benchmark days/batches 不可為負數")

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    mes_config = config["mes_report"]
    facility = str(mes_config["facility"]).strip().upper()
    schema = WP_DEFTHIS_SCHEMA_BY_FACILITY[facility]
    fac_id = str(args.fac_id or WP_DEFTHIS_FAC_ID_BY_FACILITY[facility]).strip().upper()
    index_hint = WP_DEFTHIS_INDEX_HINT_BY_FACILITY[facility]
    if not fac_id:
        parser.error("FAC_ID 不可為空白")
    oracle_config = mes_config["oracle"]
    tns_config = oracle_config["tns"][facility]
    dsn = oracledb.makedsn(
        str(tns_config["host"]),
        int(tns_config.get("port", 1521)),
        service_name=str(tns_config["service_name"]),
    )

    print("=== Connection ===")
    print("Config:", config_path)
    print("Facility:", facility)
    print("FAC_ID:", fac_id)
    print("DSN:", dsn)
    if args.auto_fac_id:
        print("--auto-fac-id: using fixed facility mapping; no discovery query is executed")

    connection = oracledb.connect(
        user=str(oracle_config["user"]),
        password=ORACLE_MES_PASSWORD,
        dsn=dsn,
    )
    if args.timeout_seconds > 0:
        connection.call_timeout = args.timeout_seconds * 1000
    print("Oracle version:", connection.version)

    cursor = connection.cursor()
    try:
        try:
            cursor.execute("ALTER SESSION SET ERROR_MESSAGE_DETAILS = ON")
            print("ERROR_MESSAGE_DETAILS: ON")
        except Exception as exc:
            print("ERROR_MESSAGE_DETAILS: unavailable:", repr(exc))

        print_index_metadata(cursor, schema)

        print(f"\n=== {schema}.WP_DEFTHIS column types ===")
        cursor.execute(
            f"""
            SELECT column_id, column_name, data_type,
                   data_length, data_precision, data_scale
            FROM ALL_TAB_COLUMNS
            WHERE owner = '{schema}'
              AND table_name = 'WP_DEFTHIS'
              AND column_name IN (
                  'FAC_ID', 'PNL_ID', 'DFCT_CODE', 'TRANS_DATE',
                  'X_AXIS', 'Y_AXIS', 'DEFT_OPER', 'IF_NEWER'
              )
            ORDER BY column_id
            """
        )
        rows = cursor.fetchall()
        if rows:
            for row in rows:
                print(row)
        else:
            print("No column metadata returned")

        print("\n=== Raw values, without comparison ===")
        try:
            cursor.execute(
                f"""
                SELECT DEFT_OPER, DUMP(DEFT_OPER),
                       FAC_ID, PNL_ID, DUMP(PNL_ID),
                       IF_NEWER, TRANS_DATE, X_AXIS, Y_AXIS
                FROM {schema}.WP_DEFTHIS
                WHERE ROWNUM <= 10
                """
            )
            for row in cursor.fetchall():
                print(row)
        except Exception as exc:
            print("RAW SELECT ERROR:", repr(exc))

        deft_oper_sql = f"""
            SELECT DEFT_OPER
            FROM {schema}.WP_DEFTHIS
            WHERE DEFT_OPER = :value
              AND ROWNUM = 1
        """
        run_test(cursor, "DEFT_OPER using string bind", deft_oper_sql, {"value": "1600"})
        run_test(cursor, "DEFT_OPER using number bind", deft_oper_sql, {"value": 1600})

        panel_id = latest_panel_id(config)
        if panel_id:
            run_test(
                cursor,
                "PNL_ID bind only",
                f"""
                SELECT PNL_ID
                FROM {schema}.WP_DEFTHIS
                WHERE FAC_ID = :fac_id
                  AND PNL_ID = :panel_id
                  AND ROWNUM = 1
                """,
                {"fac_id": fac_id, "panel_id": panel_id},
            )
            run_test(
                cursor,
                "Current filter using string 1600",
                f"""
                SELECT PNL_ID, DFCT_CODE, TRANS_DATE, X_AXIS, Y_AXIS
                FROM {schema}.WP_DEFTHIS
                WHERE FAC_ID = :fac_id
                  AND DEFT_OPER = :deft_oper
                  AND IF_NEWER = 'Y'
                  AND PNL_ID = :panel_id
                  AND ROWNUM <= 5
                """,
                {"fac_id": fac_id, "deft_oper": "1600", "panel_id": panel_id},
            )
            run_test(
                cursor,
                "Current filter using number 1600",
                f"""
                SELECT PNL_ID, DFCT_CODE, TRANS_DATE, X_AXIS, Y_AXIS
                FROM {schema}.WP_DEFTHIS
                WHERE FAC_ID = :fac_id
                  AND DEFT_OPER = :deft_oper
                  AND IF_NEWER = 'Y'
                  AND PNL_ID = :panel_id
                  AND ROWNUM <= 5
                """,
                {"fac_id": fac_id, "deft_oper": 1600, "panel_id": panel_id},
            )

        benchmark_start = args.benchmark_start
        benchmark_end = args.benchmark_end
        if args.benchmark_days:
            production_today = datetime.now()
            if production_today.hour * 60 + production_today.minute < 450:
                production_today -= timedelta(days=1)
            benchmark_end = production_today.strftime("%Y-%m-%d")
            benchmark_start = (
                production_today - timedelta(days=args.benchmark_days - 1)
            ).strftime("%Y-%m-%d")
        if benchmark_start and benchmark_end:
            panels, min_trans_date = load_benchmark_panels(
                config,
                benchmark_start,
                benchmark_end,
            )
            print(
                f"\nBenchmark local range: {benchmark_start}..{benchmark_end}, "
                f"panels={len(panels)}"
            )
            if args.benchmark_app_query:
                logging.basicConfig(
                    level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
                    stream=sys.stdout,
                )
                print("\n=== Exact Web application query benchmark ===")
                repository = OracleMESRepository(mes_config)
                started = time.monotonic()
                defects_by_panel = repository.fetch_defects(panels, min_trans_date)
                print("Matched panels:", len(defects_by_panel))
                print(f"Exact application query total: {time.monotonic() - started:.2f}s")
            else:
                benchmark_report_query(
                    connection,
                    schema,
                    panels,
                    min_trans_date,
                    batch_size=ORACLE_PANEL_BATCH_SIZE,
                    max_batches=args.benchmark_batches,
                    fac_id=fac_id,
                    index_hint=index_hint,
                )
    finally:
        cursor.close()
        connection.close()

    print("\n=== Diagnostic completed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
