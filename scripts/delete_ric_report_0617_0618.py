#!/usr/bin/env python3
"""
Delete RIC Report data for 2026-06-17 through 2026-06-18.

Only these RIC Report tables are modified:
- client_accuracy_records
- miss_review
- over_review

Inference records are not touched.
"""

import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = Path("/aidata/capi_ai/capi_results.db")
START_DATE = "2026-06-17"
END_EXCLUSIVE = "2026-06-19"  # Includes all records on 2026-06-18.


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    backup_path = DB_PATH.with_suffix(DB_PATH.suffix + f".bak.{datetime.now():%Y%m%d_%H%M%S}")

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    try:
        with sqlite3.connect(backup_path) as backup_conn:
            conn.backup(backup_conn)
        print(f"Backup created: {backup_path}")

        target_ids = [
            row["id"]
            for row in conn.execute(
                """
                SELECT id
                FROM client_accuracy_records
                WHERE time_stamp >= ?
                  AND time_stamp < ?
                """,
                (START_DATE, END_EXCLUSIVE),
            )
        ]

        print(f"Target client_accuracy_records: {len(target_ids)}")

        conn.execute("BEGIN IMMEDIATE")
        try:
            deleted_miss = 0
            deleted_over = 0
            deleted_client = 0

            if target_ids:
                placeholders = ",".join("?" for _ in target_ids)

                cur = conn.execute(
                    f"DELETE FROM miss_review WHERE client_record_id IN ({placeholders})",
                    target_ids,
                )
                deleted_miss = cur.rowcount

                cur = conn.execute(
                    f"DELETE FROM over_review WHERE client_record_id IN ({placeholders})",
                    target_ids,
                )
                deleted_over = cur.rowcount

                cur = conn.execute(
                    f"DELETE FROM client_accuracy_records WHERE id IN ({placeholders})",
                    target_ids,
                )
                deleted_client = cur.rowcount

            conn.commit()
        except Exception:
            conn.rollback()
            raise

        remaining = conn.execute(
            """
            SELECT COUNT(*)
            FROM client_accuracy_records
            WHERE time_stamp >= ?
              AND time_stamp < ?
            """,
            (START_DATE, END_EXCLUSIVE),
        ).fetchone()[0]

        print(f"Deleted miss_review: {deleted_miss}")
        print(f"Deleted over_review: {deleted_over}")
        print(f"Deleted client_accuracy_records: {deleted_client}")
        print(f"Remaining RIC Report rows in range: {remaining}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
