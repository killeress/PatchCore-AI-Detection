#!/usr/bin/env python3
"""列出 MES Oracle WP_DEFTHIS 欄位定義與少量範例資料。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import oracledb
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capi_mes_credentials import ORACLE_MES_PASSWORD  # noqa: E402


OWNER = "MERDA1"
TABLE_NAME = "WP_DEFTHIS"


def display_value(value: object) -> str:
    if value is None:
        return "NULL"

    text = repr(value)
    if len(text) > 300:
        return f"{text[:300]}...(truncated)"
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="server_config.yaml",
        help="Server YAML path (default: server_config.yaml)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Sample row count (default: 5)",
    )
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be at least 1")

    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    mes_config = config["mes_report"]
    facility = str(mes_config["facility"]).strip().upper()
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
    cursor = connection.cursor()
    try:
        print("Oracle version:", connection.version)
        print(f"\n=== {OWNER}.{TABLE_NAME} columns ===")
        cursor.execute(
            """
            SELECT column_id, column_name, data_type, data_length,
                   data_precision, data_scale, nullable
            FROM ALL_TAB_COLUMNS
            WHERE owner = :owner
              AND table_name = :table_name
            ORDER BY column_id
            """,
            {"owner": OWNER, "table_name": TABLE_NAME},
        )
        columns = cursor.fetchall()
        for column in columns:
            print(column)
        print(f"Column count: {len(columns)}")

        print(f"\n=== First {args.rows} sample rows ===")
        cursor.execute(
            f"SELECT * FROM {OWNER}.{TABLE_NAME} WHERE ROWNUM <= :row_count",
            {"row_count": args.rows},
        )
        column_names = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        for row_number, row in enumerate(rows, start=1):
            print(f"\n--- Row {row_number} ---")
            for column_name, value in zip(column_names, row):
                print(f"{column_name:<30} = {display_value(value)}")
        print(f"\nFetched rows: {len(rows)}")
    finally:
        cursor.close()
        connection.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
