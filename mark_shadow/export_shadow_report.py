#!/usr/bin/env python3
"""Export MARK shadow records and print comparison summary."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        default="/aidata/capi_ai/mark_shadow/mark_shadow.db",
    )
    parser.add_argument("--output", default="mark_shadow_results.csv")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.is_file():
        raise SystemExit(f"Database not found: {db_path}")
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT * FROM mark_shadow_results ORDER BY id"
    ).fetchall()
    fields = list(rows[0].keys()) if rows else []
    with Path(args.output).open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(dict(row) for row in rows)

    total = len(rows)
    valid = sum(int(row["valid_two_chars"] or 0) for row in rows)
    agreed = sum(int(row["agreed"] or 0) for row in rows)
    labeled = [row for row in rows if str(row["expected_text"] or "")]
    current_correct = sum(
        str(row["current_text"] or "") == str(row["expected_text"] or "")
        for row in labeled
    )
    paddle_correct = sum(
        str(row["paddle_text"] or "") == str(row["expected_text"] or "")
        for row in labeled
    )
    print(
        json.dumps(
            {
                "total": total,
                "valid_two_chars": valid,
                "agreed": agreed,
                "disagreed": total - agreed,
                "agreement_rate": agreed / total if total else 0.0,
                "labeled": len(labeled),
                "current_accuracy": (
                    current_correct / len(labeled) if labeled else None
                ),
                "paddle_accuracy": (
                    paddle_correct / len(labeled) if labeled else None
                ),
                "output": str(Path(args.output).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
