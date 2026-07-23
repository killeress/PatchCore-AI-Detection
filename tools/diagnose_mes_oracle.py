#!/usr/bin/env python3
"""唯讀診斷 MES Oracle 欄位型態與 ORA-01722 bind 問題。"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import oracledb
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capi_mes_credentials import ORACLE_MES_PASSWORD  # noqa: E402
from capi_mes_report import WP_DEFTHIS_SCHEMA_BY_FACILITY  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="server_config.yaml",
        help="Server YAML path (default: server_config.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    mes_config = config["mes_report"]
    facility = str(mes_config["facility"]).strip().upper()
    schema = WP_DEFTHIS_SCHEMA_BY_FACILITY[facility]
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
    print("DSN:", dsn)

    connection = oracledb.connect(
        user=str(oracle_config["user"]),
        password=ORACLE_MES_PASSWORD,
        dsn=dsn,
    )
    print("Oracle version:", connection.version)

    cursor = connection.cursor()
    try:
        try:
            cursor.execute("ALTER SESSION SET ERROR_MESSAGE_DETAILS = ON")
            print("ERROR_MESSAGE_DETAILS: ON")
        except Exception as exc:
            print("ERROR_MESSAGE_DETAILS: unavailable:", repr(exc))

        print(f"\n=== {schema}.WP_DEFTHIS column types ===")
        cursor.execute(
            f"""
            SELECT column_id, column_name, data_type,
                   data_length, data_precision, data_scale
            FROM ALL_TAB_COLUMNS
            WHERE owner = '{schema}'
              AND table_name = 'WP_DEFTHIS'
              AND column_name IN (
                  'PNL_ID', 'DFCT_CODE', 'TRANS_DATE',
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
                       PNL_ID, DUMP(PNL_ID),
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
                WHERE PNL_ID = :panel_id
                  AND ROWNUM = 1
                """,
                {"panel_id": panel_id},
            )
            run_test(
                cursor,
                "Current filter using string 1600",
                f"""
                SELECT PNL_ID, DFCT_CODE, TRANS_DATE, X_AXIS, Y_AXIS
                FROM {schema}.WP_DEFTHIS
                WHERE DEFT_OPER = :deft_oper
                  AND IF_NEWER = 'Y'
                  AND PNL_ID = :panel_id
                  AND ROWNUM <= 5
                """,
                {"deft_oper": "1600", "panel_id": panel_id},
            )
            run_test(
                cursor,
                "Current filter using number 1600",
                f"""
                SELECT PNL_ID, DFCT_CODE, TRANS_DATE, X_AXIS, Y_AXIS
                FROM {schema}.WP_DEFTHIS
                WHERE DEFT_OPER = :deft_oper
                  AND IF_NEWER = 'Y'
                  AND PNL_ID = :panel_id
                  AND ROWNUM <= 5
                """,
                {"deft_oper": 1600, "panel_id": panel_id},
            )
    finally:
        cursor.close()
        connection.close()

    print("\n=== Diagnostic completed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
