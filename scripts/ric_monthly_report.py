#!/usr/bin/env python3

import csv
import sqlite3
import sys
from pathlib import Path


DB_PATH = Path("/aidata/capi_ai/capi_results.db")

# Repo 中 RIC Review 於 2026-03-26 加入。
# 若正式機實際部署日期不同，請修改此日期。
RIC_REVIEW_AVAILABLE_FROM = "2026-03-26"


SQL = r"""
WITH
params(ric_review_available_from) AS (
    VALUES (date(?))
),

months(month_start) AS (
    VALUES
        ('2026-03-01'),
        ('2026-04-01'),
        ('2026-05-01'),
        ('2026-06-01'),
        ('2026-07-01')
),

source AS (
    SELECT
        strftime(
            '%Y-%m-01',
            datetime(c.time_stamp, '-7 hours', '-30 minutes')
        ) AS month_start,

        COALESCE(NULLIF(c.result_eqp, ''), 'OK') AS eqp,

        CASE
            WHEN COALESCE(NULLIF(c.result_ai, ''), 'OK') = 'OK-i'
                THEN 'OK'
            ELSE COALESCE(NULLIF(c.result_ai, ''), 'OK')
        END AS ai,

        CASE
            WHEN instr(COALESCE(c.datastr, ''), 'NG') > 0
                THEN 'NG'
            ELSE 'OK'
        END AS raw_ric,

        mr.id AS review_id,
        mr.category AS review_category

    FROM client_accuracy_records AS c
    LEFT JOIN miss_review AS mr
        ON mr.client_record_id = c.id

    WHERE datetime(c.time_stamp) >= datetime('2026-03-01 07:30:00')
      AND datetime(c.time_stamp) <  datetime('2026-08-01 07:30:00')

      /* 符合目前 RIC Report 預設：排除 RESULT_EQP=OK */
      AND COALESCE(NULLIF(c.result_eqp, ''), 'OK') <> 'OK'
),

classified AS (
    SELECT
        *,

        CASE
            WHEN ai = 'OK'
             AND raw_ric = 'NG'
             AND review_category IN (
                    'ric_misjudge',
                    'within_spec_misjudge',
                    'data_error_actually_ok'
                 )
                THEN 1
            ELSE 0
        END AS ric_misjudge,

        CASE
            WHEN ai = 'OK'
             AND raw_ric = 'NG'
             AND review_category IN (
                    'threshold_high',
                    'dust_misfilter'
                 )
                THEN 1
            ELSE 0
        END AS ai_miss,

        CASE
            WHEN ai = 'OK' AND raw_ric = 'NG'
                THEN 1
            ELSE 0
        END AS miss_candidate,

        CASE
            WHEN ai = 'OK'
             AND raw_ric = 'NG'
             AND review_id IS NOT NULL
                THEN 1
            ELSE 0
        END AS reviewed_miss

    FROM source
),

effective AS (
    SELECT
        *,
        CASE
            WHEN ric_misjudge = 1 THEN 'OK'
            ELSE raw_ric
        END AS effective_ric
    FROM classified
),

monthly AS (
    SELECT
        month_start,
        COUNT(*) AS total_count,

        SUM(ai_miss) AS ai_miss_count,

        SUM(
            CASE
                WHEN ai = 'NG' AND effective_ric = 'OK'
                    THEN 1
                ELSE 0
            END
        ) AS ai_over_count,

        SUM(
            CASE
                WHEN eqp = 'NG' AND effective_ric = 'OK'
                    THEN 1
                ELSE 0
            END
        ) AS aoi_over_count,

        SUM(ric_misjudge) AS ric_misjudge_count,
        SUM(miss_candidate) AS miss_candidate_count,
        SUM(reviewed_miss) AS reviewed_miss_count

    FROM effective
    GROUP BY month_start
),

inference_monthly AS (
    SELECT
        strftime(
            '%Y-%m-01',
            datetime(request_time, '-7 hours', '-30 minutes')
        ) AS month_start,

        SUM(
            CASE
                WHEN machine_judgment != '' AND machine_judgment != 'OK'
                    THEN 1
                ELSE 0
            END
        ) AS aoi_ng_count,

        SUM(
            CASE
                WHEN machine_judgment != ''
                 AND machine_judgment != 'OK'
                 AND ai_judgment IN ('OK', 'OK-i')
                    THEN 1
                ELSE 0
            END
        ) AS ai_revival_count

    FROM inference_records
    WHERE datetime(request_time) >= datetime('2026-03-01 07:30:00')
      AND datetime(request_time) <  datetime('2026-08-01 07:30:00')
    GROUP BY month_start
)

SELECT
    substr(m.month_start, 1, 7) AS month,
    COALESCE(x.total_count, 0) AS total_count,

    COALESCE(x.ai_miss_count, 0) AS ai_miss_count,
    ROUND(
        100.0 * x.ai_miss_count / NULLIF(x.total_count, 0),
        2
    ) AS ai_miss_rate_pct,

    COALESCE(x.ai_over_count, 0) AS ai_over_count,
    ROUND(
        100.0 * x.ai_over_count / NULLIF(x.total_count, 0),
        2
    ) AS ai_over_rate_pct,

    COALESCE(x.aoi_over_count, 0) AS aoi_over_count,
    ROUND(
        100.0 * x.aoi_over_count / NULLIF(x.total_count, 0),
        2
    ) AS aoi_over_rate_pct,

    COALESCE(i.aoi_ng_count, 0) AS inference_aoi_ng_count,
    COALESCE(i.ai_revival_count, 0) AS ai_revival_count,
    ROUND(
        100.0 * i.ai_revival_count / NULLIF(i.aoi_ng_count, 0),
        2
    ) AS ai_revival_rate_pct,

    COALESCE(x.ric_misjudge_count, 0) AS ric_misjudge_count,

    ROUND(
        100.0 * x.ric_misjudge_count / NULLIF(x.total_count, 0),
        2
    ) AS ric_misjudge_rate_observed_pct,

    COALESCE(x.miss_candidate_count, 0) AS miss_candidate_count,
    COALESCE(x.reviewed_miss_count, 0) AS reviewed_miss_count,

    ROUND(
        100.0 * x.reviewed_miss_count
        / NULLIF(x.miss_candidate_count, 0),
        2
    ) AS ric_review_coverage_pct,

    CASE
        WHEN x.total_count IS NULL
            THEN NULL
        WHEN p.ric_review_available_from > m.month_start
            THEN NULL
        WHEN COALESCE(x.reviewed_miss_count, 0)
           < COALESCE(x.miss_candidate_count, 0)
            THEN NULL
        ELSE ROUND(
            100.0 * x.ric_misjudge_count / x.total_count,
            2
        )
    END AS ric_misjudge_rate_strict_pct,

    CASE
        WHEN x.total_count IS NULL
            THEN 'NO_REPORT_DATA'
        WHEN p.ric_review_available_from > m.month_start
            THEN 'NOT_AVAILABLE_FULL_MONTH'
        WHEN COALESCE(x.miss_candidate_count, 0) = 0
            THEN 'NO_MISS_CANDIDATES'
        WHEN COALESCE(x.reviewed_miss_count, 0) = 0
            THEN 'NO_REVIEW_DATA'
        WHEN x.reviewed_miss_count < x.miss_candidate_count
            THEN 'PARTIAL_REVIEW'
        ELSE 'COMPLETE_REVIEW'
    END AS ric_data_status

FROM months AS m
CROSS JOIN params AS p
LEFT JOIN monthly AS x
    ON x.month_start = m.month_start
LEFT JOIN inference_monthly AS i
    ON i.month_start = m.month_start
ORDER BY m.month_start
"""


def main() -> int:
    connection = None

    try:
        db_uri = DB_PATH.as_uri() + "?mode=ro"
        connection = sqlite3.connect(
            db_uri,
            uri=True,
            timeout=30,
        )
        connection.execute("PRAGMA query_only = ON")

        cursor = connection.execute(
            SQL,
            (RIC_REVIEW_AVAILABLE_FROM,),
        )

        header = [column[0] for column in cursor.description]
        rows = cursor.fetchall()

    except sqlite3.Error as exc:
        print(
            f"SQLite query failed: {exc}",
            file=sys.stderr,
        )
        return 1

    finally:
        if connection is not None:
            connection.close()

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
