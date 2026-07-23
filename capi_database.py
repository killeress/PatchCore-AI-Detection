"""
CAPI AI 推論結果資料庫模組

使用 SQLite 持久化推論結果，支援追溯查詢。
三層資料結構: inference_records → image_results → tile_results

使用方式:
    from capi_database import CAPIDatabase
    db = CAPIDatabase("/data/capi_ai/capi_results.db")
    record_id = db.save_inference_record(...)
"""

import sqlite3
import threading
import json
import re
import os
import logging
import hashlib
import hmac
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple

_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_FACTORY_DAY_START_TIME = "07:30:00"
logger = logging.getLogger(__name__)


def _next_date_str(date_str: str) -> str:
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")


def _factory_day_start_ts(date_str: str) -> str:
    return f"{date_str} {_FACTORY_DAY_START_TIME}"


def _factory_day_end_ts(date_str: str) -> str:
    return f"{_next_date_str(date_str)} {_FACTORY_DAY_START_TIME}"


def _normalize_inference_error_type(ai_judgment: str, error_message: str) -> str:
    judgment = str(ai_judgment or "").strip()
    error_type = judgment[4:].strip() if judgment.startswith("ERR:") else ""
    if not error_type:
        error_type = str(error_message or "").strip()
    if not error_type:
        return "Unknown"

    match = re.match(r"([A-Za-z][A-Za-z0-9_-]*)", error_type)
    return match.group(1) if match else error_type


class CAPIDatabase:
    """CAPI AI 推論結果 SQLite 資料庫"""

    # 共用 SQL 條件片段（search_records 與 get_inference_stats 共用）
    _AOI_NG_COND = "machine_judgment != '' AND machine_judgment != 'OK'"
    _AI_OK_COND = "ai_judgment = 'OK' OR ai_judgment = 'OK-i'"
    _AI_NG_COND = "ai_judgment LIKE 'NG%'"
    _ERR_COND = "ai_judgment LIKE 'ERR%'"

    def __init__(self, db_path: str):
        """
        初始化資料庫連線

        Args:
            db_path: SQLite 資料庫檔案路徑
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """取得資料庫連線 (每個執行緒需要獨立連線)"""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        """建立資料表"""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS inference_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    glass_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    machine_no TEXT NOT NULL,
                    resolution_x INTEGER DEFAULT 0,
                    resolution_y INTEGER DEFAULT 0,
                    machine_judgment TEXT DEFAULT '',
                    ai_judgment TEXT DEFAULT '',
                    image_dir TEXT DEFAULT '',
                    total_images INTEGER DEFAULT 0,
                    ng_images INTEGER DEFAULT 0,
                    ng_details TEXT DEFAULT '',
                    request_time TEXT NOT NULL,
                    response_time TEXT DEFAULT '',
                    client_request_text TEXT DEFAULT '',
                    client_response_text TEXT DEFAULT '',
                    processing_seconds REAL DEFAULT 0.0,
                    heatmap_dir TEXT DEFAULT '',
                    error_message TEXT DEFAULT '',
                    client_bomb_info TEXT DEFAULT '',
                    aoi_machine_coords TEXT DEFAULT '',
                    image_preprocess_pipeline TEXT DEFAULT '',
                    image_preprocess_pipelines TEXT DEFAULT '',
                    image_preprocess_timing TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                CREATE TABLE IF NOT EXISTS image_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    image_width INTEGER DEFAULT 0,
                    image_height INTEGER DEFAULT 0,
                    otsu_bounds TEXT DEFAULT '',
                    tile_count INTEGER DEFAULT 0,
                    excluded_tiles INTEGER DEFAULT 0,
                    anomaly_count INTEGER DEFAULT 0,
                    max_score REAL DEFAULT 0.0,
                    is_ng INTEGER DEFAULT 0,
                    is_dust_only INTEGER DEFAULT 0,
                    is_bomb INTEGER DEFAULT 0,
                    inference_time_ms REAL DEFAULT 0.0,
                    heatmap_path TEXT DEFAULT '',
                    mark_text TEXT DEFAULT '',
                    mark_confidence REAL DEFAULT 0.0,
                    mark_bbox TEXT DEFAULT '',
                    mark_roi TEXT DEFAULT '',
                    mark_orientation TEXT DEFAULT '',
                    mark_source_image TEXT DEFAULT '',
                    FOREIGN KEY (record_id) REFERENCES inference_records(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS tile_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_result_id INTEGER NOT NULL,
                    tile_id INTEGER DEFAULT 0,
                    x INTEGER DEFAULT 0,
                    y INTEGER DEFAULT 0,
                    width INTEGER DEFAULT 0,
                    height INTEGER DEFAULT 0,
                    score REAL DEFAULT 0.0,
                    is_anomaly INTEGER DEFAULT 0,
                    is_dust INTEGER DEFAULT 0,
                    dust_iou REAL DEFAULT 0.0,
                    is_bomb INTEGER DEFAULT 0,
                    bomb_code TEXT DEFAULT '',
                    peak_x INTEGER DEFAULT -1,
                    peak_y INTEGER DEFAULT -1,
                    heatmap_path TEXT DEFAULT '',
                    is_exclude_zone INTEGER DEFAULT 0,
                    is_aoi_coord INTEGER DEFAULT 0,
                    aoi_defect_code TEXT DEFAULT '',
                    aoi_product_x INTEGER DEFAULT -1,
                    aoi_product_y INTEGER DEFAULT -1,
                    aoi_image_x INTEGER DEFAULT -1,
                    aoi_image_y INTEGER DEFAULT -1,
                    aoi_tile_shift_dx INTEGER DEFAULT 0,
                    aoi_tile_shift_dy INTEGER DEFAULT 0,
                    scratch_score REAL DEFAULT 0.0,
                    scratch_filtered INTEGER DEFAULT 0,
                    zone TEXT DEFAULT '',
                    FOREIGN KEY (image_result_id) REFERENCES image_results(id) ON DELETE CASCADE
                );

                -- CV 邊緣缺陷結果 (獨立於 tile_results)
                CREATE TABLE IF NOT EXISTS edge_defect_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_result_id INTEGER NOT NULL,
                    side TEXT NOT NULL DEFAULT '',
                    area INTEGER DEFAULT 0,
                    bbox_x INTEGER DEFAULT 0,
                    bbox_y INTEGER DEFAULT 0,
                    bbox_w INTEGER DEFAULT 0,
                    bbox_h INTEGER DEFAULT 0,
                    max_diff REAL DEFAULT 0.0,
                    center_x INTEGER DEFAULT 0,
                    center_y INTEGER DEFAULT 0,
                    heatmap_path TEXT DEFAULT '',
                    is_dust INTEGER DEFAULT 0,
                    FOREIGN KEY (image_result_id) REFERENCES image_results(id) ON DELETE CASCADE
                );

                -- RIC 匯入批次追蹤
                CREATE TABLE IF NOT EXISTS ric_import_batches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    total_records INTEGER DEFAULT 0,
                    import_time TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                -- RIC 人工檢驗原始資料
                CREATE TABLE IF NOT EXISTS ric_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id INTEGER NOT NULL,
                    timestamp TEXT,
                    ric_id TEXT,
                    pnl_id TEXT NOT NULL,
                    prod_id TEXT,
                    mach_id TEXT,
                    station TEXT,
                    ipaddress TEXT,
                    keytime TEXT,
                    datastr TEXT,
                    ric_judgment TEXT,
                    FOREIGN KEY (batch_id) REFERENCES ric_import_batches(id)
                );

                -- 索引
                CREATE INDEX IF NOT EXISTS idx_records_glass_id ON inference_records(glass_id);
                CREATE INDEX IF NOT EXISTS idx_records_created_at ON inference_records(created_at);
                CREATE INDEX IF NOT EXISTS idx_records_request_time ON inference_records(request_time);
                CREATE INDEX IF NOT EXISTS idx_records_machine_no ON inference_records(machine_no);
                CREATE INDEX IF NOT EXISTS idx_records_ai_judgment ON inference_records(ai_judgment);
                CREATE INDEX IF NOT EXISTS idx_records_glass_request_time ON inference_records(glass_id, request_time DESC);
                CREATE INDEX IF NOT EXISTS idx_image_results_record_id ON image_results(record_id);
                CREATE INDEX IF NOT EXISTS idx_tile_results_image_id ON tile_results(image_result_id);
                CREATE INDEX IF NOT EXISTS idx_edge_defects_image_id ON edge_defect_results(image_result_id);
                CREATE INDEX IF NOT EXISTS idx_ric_pnl_id ON ric_records(pnl_id);
                CREATE INDEX IF NOT EXISTS idx_ric_mach_id ON ric_records(mach_id);
                CREATE INDEX IF NOT EXISTS idx_ric_batch ON ric_records(batch_id);

                -- 設定參數表 (存放可調整的推論參數)
                CREATE TABLE IF NOT EXISTS config_params (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    param_name TEXT NOT NULL UNIQUE,
                    param_value TEXT NOT NULL DEFAULT '',
                    param_type TEXT NOT NULL DEFAULT 'str',
                    description TEXT DEFAULT '',
                    updated_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                -- 設定修改歷史紀錄
                CREATE TABLE IF NOT EXISTS config_change_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    param_name TEXT NOT NULL,
                    old_value TEXT DEFAULT '',
                    new_value TEXT DEFAULT '',
                    change_reason TEXT DEFAULT '',
                    changed_by TEXT DEFAULT '',
                    changed_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                -- 參數設定登入帳號
                CREATE TABLE IF NOT EXISTS settings_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    updated_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                CREATE INDEX IF NOT EXISTS idx_config_param_name ON config_params(param_name);
                CREATE INDEX IF NOT EXISTS idx_config_history_param ON config_change_history(param_name);
                CREATE INDEX IF NOT EXISTS idx_config_history_time ON config_change_history(changed_at);
                CREATE INDEX IF NOT EXISTS idx_settings_users_username ON settings_users(username);

                -- Client accuracy records (TIME_STAMP + PNL_ID 為複合唯一鍵)
                CREATE TABLE IF NOT EXISTS client_accuracy_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time_stamp TEXT NOT NULL,
                    pnl_id TEXT NOT NULL,
                    mach_id TEXT,
                    result_eqp TEXT,
                    result_ai TEXT,
                    result_ric TEXT,
                    datastr TEXT,
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    UNIQUE(time_stamp, pnl_id)
                );
                CREATE INDEX IF NOT EXISTS idx_client_acc_pnl ON client_accuracy_records(pnl_id);
                CREATE INDEX IF NOT EXISTS idx_client_acc_time ON client_accuracy_records(time_stamp);

                -- Miss review records (漏檢原因回填)
                CREATE TABLE IF NOT EXISTS miss_review (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_record_id INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    note TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (client_record_id) REFERENCES client_accuracy_records(id),
                    UNIQUE(client_record_id)
                );
                CREATE INDEX IF NOT EXISTS idx_miss_review_client ON miss_review(client_record_id);

                -- Over review records (過檢原因回填)
                CREATE TABLE IF NOT EXISTS over_review (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_record_id INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    note TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (client_record_id) REFERENCES client_accuracy_records(id),
                    UNIQUE(client_record_id)
                );
                CREATE INDEX IF NOT EXISTS idx_over_review_client ON over_review(client_record_id);

                -- MES Report comparison manual review.
                -- Deliberately no FK to inference_records: reviewed evidence must survive
                -- inference/tile retention cleanup.
                CREATE TABLE IF NOT EXISTS mes_comparison_review (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inference_record_id INTEGER NOT NULL UNIQUE,
                    glass_id TEXT NOT NULL,
                    model_id TEXT DEFAULT '',
                    machine_no TEXT DEFAULT '',
                    request_time TEXT DEFAULT '',
                    ai_judgment TEXT DEFAULT '',
                    mes_judgment TEXT DEFAULT '',
                    review_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    note TEXT DEFAULT '',
                    reviewer TEXT DEFAULT '',
                    confirmed_ng INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    updated_at TEXT DEFAULT (datetime('now', 'localtime'))
                );
                CREATE INDEX IF NOT EXISTS idx_mes_review_type
                    ON mes_comparison_review(review_type, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_mes_review_machine
                    ON mes_comparison_review(machine_no, model_id, updated_at DESC);

                -- Human-confirmed NG samples selected from AOI-coordinate tiles.
                -- Source IDs are snapshots only (no FK) so the validation DB remains
                -- usable after tile_results/image_results are cleaned.
                CREATE TABLE IF NOT EXISTS ng_validation_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id INTEGER NOT NULL,
                    inference_record_id INTEGER NOT NULL,
                    tile_result_id INTEGER NOT NULL,
                    image_result_id INTEGER NOT NULL,
                    glass_id TEXT NOT NULL,
                    model_id TEXT DEFAULT '',
                    machine_no TEXT DEFAULT '',
                    request_time TEXT DEFAULT '',
                    image_name TEXT NOT NULL,
                    source_image_path TEXT DEFAULT '',
                    lighting TEXT NOT NULL,
                    zone TEXT DEFAULT '',
                    aoi_defect_code TEXT DEFAULT '',
                    aoi_product_x INTEGER DEFAULT -1,
                    aoi_product_y INTEGER DEFAULT -1,
                    aoi_image_x INTEGER DEFAULT -1,
                    aoi_image_y INTEGER DEFAULT -1,
                    tile_x INTEGER DEFAULT 0,
                    tile_y INTEGER DEFAULT 0,
                    tile_w INTEGER DEFAULT 0,
                    tile_h INTEGER DEFAULT 0,
                    ai_score REAL DEFAULT 0.0,
                    crop_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'confirmed',
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                    UNIQUE(review_id, tile_result_id)
                );
                CREATE INDEX IF NOT EXISTS idx_ng_validation_status
                    ON ng_validation_samples(status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_ng_validation_unit
                    ON ng_validation_samples(model_id, lighting, zone, status);
                CREATE INDEX IF NOT EXISTS idx_ng_validation_review
                    ON ng_validation_samples(review_id, status);

                -- Over-retrain pool (過檢 tile 重訓候選池)
                CREATE TABLE IF NOT EXISTS over_retrain_pool (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_record_id INTEGER NOT NULL,
                    inference_record_id INTEGER NOT NULL,
                    tile_result_id INTEGER NOT NULL UNIQUE,
                    image_result_id INTEGER NOT NULL,
                    machine_id TEXT NOT NULL,
                    machine_no TEXT DEFAULT '',
                    pnl_id TEXT NOT NULL,
                    client_time_stamp TEXT DEFAULT '',
                    datastr TEXT DEFAULT '',
                    screen_prefix TEXT NOT NULL,
                    lighting TEXT NOT NULL,
                    zone TEXT DEFAULT '',
                    source_path TEXT NOT NULL,
                    thumb_path TEXT DEFAULT '',
                    tile_x INTEGER DEFAULT 0,
                    tile_y INTEGER DEFAULT 0,
                    tile_w INTEGER DEFAULT 0,
                    tile_h INTEGER DEFAULT 0,
                    score REAL DEFAULT 0.0,
                    added_to_job_id TEXT DEFAULT '',
                    added_to_bundle_id INTEGER,
                    added_to_unit TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (client_record_id) REFERENCES client_accuracy_records(id),
                    FOREIGN KEY (inference_record_id) REFERENCES inference_records(id),
                    FOREIGN KEY (tile_result_id) REFERENCES tile_results(id)
                );
                CREATE INDEX IF NOT EXISTS idx_over_retrain_pool_created
                    ON over_retrain_pool(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_over_retrain_pool_machine
                    ON over_retrain_pool(machine_id, lighting, zone);
                CREATE INDEX IF NOT EXISTS idx_over_retrain_pool_client
                    ON over_retrain_pool(client_record_id);

                -- Within-spec suggestion calculation logs (過檢 Review 規格內建議)
                CREATE TABLE IF NOT EXISTS within_spec_review_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_record_id INTEGER,
                    inference_record_id INTEGER NOT NULL,
                    suggested INTEGER NOT NULL DEFAULT 0,
                    category TEXT DEFAULT '',
                    reason TEXT DEFAULT '',
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    error_message TEXT DEFAULT '',
                    source TEXT DEFAULT 'review',
                    processing_seconds REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (client_record_id) REFERENCES client_accuracy_records(id) ON DELETE CASCADE,
                    FOREIGN KEY (inference_record_id) REFERENCES inference_records(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_within_spec_log_client
                    ON within_spec_review_log(client_record_id, id DESC);
                CREATE INDEX IF NOT EXISTS idx_within_spec_log_inference
                    ON within_spec_review_log(inference_record_id);

                -- Scratch 誤救標記 (以 tile 為單位，供 DINOv2 再訓練負樣本收集)
                CREATE TABLE IF NOT EXISTS scratch_rescue_review (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tile_result_id INTEGER NOT NULL,
                    is_misrescue INTEGER NOT NULL DEFAULT 1,
                    note TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (tile_result_id) REFERENCES tile_results(id),
                    UNIQUE(tile_result_id)
                );
                CREATE INDEX IF NOT EXISTS idx_scratch_rescue_review_tile ON scratch_rescue_review(tile_result_id);

                -- 訓練 Job 狀態追蹤
                -- panel_modes: JSON array，元素 full / inner_only / edge_only / corners_only，與 panel_paths 同長度。
                --   NULL 視同 ["full"] * len(panel_paths)（向下相容舊 job）。
                CREATE TABLE IF NOT EXISTS training_jobs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id          TEXT UNIQUE,
                    machine_id      TEXT NOT NULL,
                    state           TEXT NOT NULL,
                    started_at      TEXT,
                    completed_at    TEXT,
                    panel_paths     TEXT,
                    panel_modes     TEXT,
                    output_bundle   TEXT,
                    error_message   TEXT,
                    training_params TEXT,
                    training_scope  TEXT,
                    training_data_source TEXT,
                    image_preprocess_pipeline TEXT,
                    image_preprocess_pipelines TEXT,
                    preprocess_after_tiling   INTEGER DEFAULT 0,
                    tile_stride     INTEGER
                );

                -- 已訓練模型 bundle 元資料
                CREATE TABLE IF NOT EXISTS model_registry (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    machine_id        TEXT NOT NULL,
                    bundle_path       TEXT UNIQUE NOT NULL,
                    trained_at        TEXT NOT NULL,
                    panel_count       INTEGER,
                    inner_tile_count  INTEGER,
                    edge_tile_count   INTEGER,
                    ng_tile_count     INTEGER,
                    bundle_size_bytes INTEGER,
                    is_active         INTEGER DEFAULT 0,
                    job_id            TEXT,
                    notes             TEXT
                );

                -- Client 機種前 8 碼 → 模型 bundle 自動切換規則
                CREATE TABLE IF NOT EXISTS auto_model_switch_rules (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    series_prefix TEXT NOT NULL UNIQUE,
                    bundle_id     INTEGER NOT NULL,
                    notes         TEXT DEFAULT '',
                    created_at    TEXT DEFAULT (datetime('now', 'localtime')),
                    updated_at    TEXT DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (bundle_id) REFERENCES model_registry(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_auto_model_switch_rules_series
                    ON auto_model_switch_rules(series_prefix);

                CREATE TABLE IF NOT EXISTS auto_model_switch_history (
                    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                    checked_at            TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                    requested_model_id    TEXT NOT NULL DEFAULT '',
                    series_prefix         TEXT NOT NULL DEFAULT '',
                    previous_bundle_id    INTEGER,
                    previous_bundle_label TEXT DEFAULT '',
                    target_bundle_id      INTEGER,
                    target_bundle_label   TEXT DEFAULT '',
                    action                TEXT NOT NULL,
                    status                TEXT NOT NULL,
                    message               TEXT DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_auto_model_switch_history_checked
                    ON auto_model_switch_history(checked_at DESC);
                CREATE INDEX IF NOT EXISTS idx_auto_model_switch_history_series
                    ON auto_model_switch_history(series_prefix, checked_at DESC);

                -- Wizard step 3 review 用暫存 tile pool (zone 允許 NULL 以支援 NG tiles)
                CREATE TABLE IF NOT EXISTS training_tile_pool (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id      TEXT NOT NULL,
                    lighting    TEXT NOT NULL,
                    zone        TEXT,
                    source      TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    thumb_path  TEXT,
                    decision    TEXT DEFAULT 'accept'
                );
                CREATE INDEX IF NOT EXISTS idx_tile_pool_job ON training_tile_pool(job_id, lighting, zone, source);
                CREATE TABLE IF NOT EXISTS tile_score_cache (
                    tile_id           INTEGER NOT NULL,
                    scoring_bundle_id INTEGER NOT NULL,
                    score             REAL    NOT NULL,
                    computed_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (tile_id, scoring_bundle_id)
                );
                CREATE INDEX IF NOT EXISTS idx_score_cache_bundle
                    ON tile_score_cache(scoring_bundle_id);
            """)
            
            # Migration for adding missing columns to existing database
            def add_column_if_not_exists(table, column, def_type):
                cursor = conn.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in cursor.fetchall()]
                if column not in columns:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {def_type}")

            def ensure_within_spec_log_schema():
                cursor = conn.execute("PRAGMA table_info(within_spec_review_log)")
                columns = {row[1]: row for row in cursor.fetchall()}
                client_col = columns.get("client_record_id")
                needs_rebuild = bool(client_col and client_col[3])
                if needs_rebuild:
                    conn.execute("ALTER TABLE within_spec_review_log RENAME TO within_spec_review_log_old")
                    conn.executescript("""
                        CREATE TABLE within_spec_review_log (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            client_record_id INTEGER,
                            inference_record_id INTEGER NOT NULL,
                            suggested INTEGER NOT NULL DEFAULT 0,
                            category TEXT DEFAULT '',
                            reason TEXT DEFAULT '',
                            detail_json TEXT NOT NULL DEFAULT '{}',
                            error_message TEXT DEFAULT '',
                            source TEXT DEFAULT 'review',
                            processing_seconds REAL DEFAULT 0.0,
                            created_at TEXT DEFAULT (datetime('now', 'localtime')),
                            FOREIGN KEY (client_record_id) REFERENCES client_accuracy_records(id) ON DELETE CASCADE,
                            FOREIGN KEY (inference_record_id) REFERENCES inference_records(id) ON DELETE CASCADE
                        );
                    """)
                    conn.execute(
                        """INSERT INTO within_spec_review_log
                           (id, client_record_id, inference_record_id, suggested, category, reason,
                            detail_json, error_message, source, processing_seconds, created_at)
                           SELECT id, client_record_id, inference_record_id, suggested, category, reason,
                                  detail_json, error_message, 'review', processing_seconds, created_at
                             FROM within_spec_review_log_old"""
                    )
                    conn.execute("DROP TABLE within_spec_review_log_old")

                add_column_if_not_exists("within_spec_review_log", "source", "TEXT DEFAULT 'review'")
                conn.executescript("""
                    CREATE INDEX IF NOT EXISTS idx_within_spec_log_client
                        ON within_spec_review_log(client_record_id, id DESC);
                    CREATE INDEX IF NOT EXISTS idx_within_spec_log_inference
                        ON within_spec_review_log(inference_record_id);
                    CREATE INDEX IF NOT EXISTS idx_within_spec_log_source
                        ON within_spec_review_log(source, id DESC);
                """)

            ensure_within_spec_log_schema()

            add_column_if_not_exists("config_change_history", "changed_by", "TEXT DEFAULT ''")

            if not conn.execute(
                "SELECT id FROM settings_users WHERE username = ?",
                ("admin",),
            ).fetchone():
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    """INSERT INTO settings_users
                       (username, password_hash, is_admin, created_at, updated_at)
                       VALUES (?, ?, 1, ?, ?)""",
                    ("admin", self._hash_settings_password("INXCAPI"), now, now),
                )

            add_column_if_not_exists("inference_records", "error_message", "TEXT DEFAULT ''")
            add_column_if_not_exists("inference_records", "client_bomb_info", "TEXT DEFAULT ''")
            add_column_if_not_exists("inference_records", "client_request_text", "TEXT DEFAULT ''")
            add_column_if_not_exists("inference_records", "client_response_text", "TEXT DEFAULT ''")
            add_column_if_not_exists("inference_records", "aoi_machine_coords", "TEXT DEFAULT ''")
            add_column_if_not_exists("inference_records", "image_preprocess_pipeline", "TEXT DEFAULT ''")
            add_column_if_not_exists("inference_records", "image_preprocess_pipelines", "TEXT DEFAULT ''")
            add_column_if_not_exists("inference_records", "image_preprocess_timing", "TEXT DEFAULT ''")
            add_column_if_not_exists("image_results", "is_bomb", "INTEGER DEFAULT 0")
            add_column_if_not_exists("image_results", "mark_text", "TEXT DEFAULT ''")
            add_column_if_not_exists("image_results", "mark_confidence", "REAL DEFAULT 0.0")
            add_column_if_not_exists("image_results", "mark_bbox", "TEXT DEFAULT ''")
            add_column_if_not_exists("image_results", "mark_roi", "TEXT DEFAULT ''")
            add_column_if_not_exists("image_results", "mark_orientation", "TEXT DEFAULT ''")
            add_column_if_not_exists("image_results", "mark_source_image", "TEXT DEFAULT ''")
            add_column_if_not_exists("tile_results", "is_bomb", "INTEGER DEFAULT 0")
            add_column_if_not_exists("tile_results", "bomb_code", "TEXT DEFAULT ''")
            add_column_if_not_exists("tile_results", "peak_x", "INTEGER DEFAULT -1")
            add_column_if_not_exists("tile_results", "peak_y", "INTEGER DEFAULT -1")
            add_column_if_not_exists("edge_defect_results", "is_dust", "INTEGER DEFAULT 0")
            add_column_if_not_exists("edge_defect_results", "is_bomb", "INTEGER DEFAULT 0")
            add_column_if_not_exists("edge_defect_results", "bomb_code", "TEXT DEFAULT ''")
            add_column_if_not_exists("edge_defect_results", "is_cv_ok", "INTEGER DEFAULT 0")
            add_column_if_not_exists("edge_defect_results", "threshold_used", "INTEGER DEFAULT 0")
            add_column_if_not_exists("edge_defect_results", "min_area_used", "INTEGER DEFAULT 0")
            add_column_if_not_exists("edge_defect_results", "min_max_diff_used", "INTEGER DEFAULT 0")
            # PatchCore inspector 路徑 (aoi_edge 可切換)
            add_column_if_not_exists("edge_defect_results", "inspector_mode", "TEXT DEFAULT 'cv'")
            add_column_if_not_exists("edge_defect_results", "patchcore_score", "REAL DEFAULT 0.0")
            add_column_if_not_exists("edge_defect_results", "patchcore_threshold", "REAL DEFAULT 0.0")
            add_column_if_not_exists("edge_defect_results", "patchcore_ok_reason", "TEXT DEFAULT ''")
            # Phase 6 fusion 欄位
            add_column_if_not_exists("edge_defect_results", "source_inspector", "TEXT DEFAULT ''")
            add_column_if_not_exists("edge_defect_results", "d_edge_px", "REAL DEFAULT 0.0")
            add_column_if_not_exists("edge_defect_results", "fusion_fallback_reason", "TEXT DEFAULT ''")
            # Phase 7 PC ROI 內移欄位
            add_column_if_not_exists("edge_defect_results", "pc_roi_origin_x", "INTEGER DEFAULT 0")
            add_column_if_not_exists("edge_defect_results", "pc_roi_origin_y", "INTEGER DEFAULT 0")
            add_column_if_not_exists("edge_defect_results", "pc_roi_shift_dx", "INTEGER DEFAULT 0")
            add_column_if_not_exists("edge_defect_results", "pc_roi_shift_dy", "INTEGER DEFAULT 0")
            add_column_if_not_exists("edge_defect_results", "pc_roi_fallback_reason", "TEXT DEFAULT ''")
            add_column_if_not_exists("tile_results", "is_exclude_zone", "INTEGER DEFAULT 0")
            add_column_if_not_exists("tile_results", "is_aoi_coord", "INTEGER DEFAULT 0")
            add_column_if_not_exists("tile_results", "aoi_defect_code", "TEXT DEFAULT ''")
            add_column_if_not_exists("tile_results", "aoi_product_x", "INTEGER DEFAULT -1")
            add_column_if_not_exists("tile_results", "aoi_product_y", "INTEGER DEFAULT -1")
            add_column_if_not_exists("tile_results", "aoi_image_x", "INTEGER DEFAULT -1")
            add_column_if_not_exists("tile_results", "aoi_image_y", "INTEGER DEFAULT -1")
            add_column_if_not_exists("tile_results", "aoi_tile_shift_dx", "INTEGER DEFAULT 0")
            add_column_if_not_exists("tile_results", "aoi_tile_shift_dy", "INTEGER DEFAULT 0")
            add_column_if_not_exists("inference_records", "inference_log", "TEXT DEFAULT ''")
            add_column_if_not_exists("inference_records", "omit_overexposed", "INTEGER DEFAULT 0")
            add_column_if_not_exists("inference_records", "omit_overexposure_info", "TEXT DEFAULT ''")
            # Scratch classifier post-filter (over-review reduction)
            add_column_if_not_exists("tile_results", "scratch_score", "REAL DEFAULT 0.0")
            add_column_if_not_exists("tile_results", "scratch_filtered", "INTEGER DEFAULT 0")
            add_column_if_not_exists("image_results", "scratch_filter_count", "INTEGER DEFAULT 0")
            add_column_if_not_exists("training_jobs", "training_params", "TEXT")
            # 8 panel wizard：前 3 = full（收 inner+edge），後 5 = corners_only（只收 4 角給 edge 模型補強）
            add_column_if_not_exists("training_jobs", "panel_modes", "TEXT")
            # 6-step wizard：完整訓練或局部重訓 scope（mode / selected_units / target_bundle_id）
            add_column_if_not_exists("training_jobs", "training_scope", "TEXT")
            add_column_if_not_exists("training_jobs", "training_data_source", "TEXT")
            add_column_if_not_exists("training_jobs", "image_preprocess_pipeline", "TEXT")
            add_column_if_not_exists("training_jobs", "image_preprocess_pipelines", "TEXT")
            add_column_if_not_exists("training_jobs", "preprocess_after_tiling", "INTEGER DEFAULT 0")
            add_column_if_not_exists("training_jobs", "tile_stride", "INTEGER")
            add_column_if_not_exists("model_registry", "notes", "TEXT")
            # 新架構 (C-10) per-tile model routing 紀錄："inner" / "edge" / "bright_spot"；v1 為 ""
            add_column_if_not_exists("tile_results", "zone", "TEXT DEFAULT ''")

            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_image_results_dust_record
                    ON image_results(record_id) WHERE is_dust_only = 1;
                CREATE INDEX IF NOT EXISTS idx_tile_results_dust_image
                    ON tile_results(image_result_id) WHERE is_dust = 1;
                CREATE INDEX IF NOT EXISTS idx_tile_results_scratch_image
                    ON tile_results(image_result_id) WHERE scratch_filtered = 1;
                CREATE INDEX IF NOT EXISTS idx_edge_defects_dust_image
                    ON edge_defect_results(image_result_id) WHERE is_dust = 1;
            """)

            conn.commit()
        finally:
            conn.close()

    def save_inference_record(
        self,
        glass_id: str,
        model_id: str,
        machine_no: str,
        resolution: Tuple[int, int],
        machine_judgment: str,
        ai_judgment: str,
        image_dir: str,
        total_images: int,
        ng_images: int,
        ng_details: str,
        request_time: str,
        response_time: str,
        processing_seconds: float,
        heatmap_dir: str = "",
        error_message: str = "",
        client_bomb_info: str = "",
        client_request_text: str = "",
        client_response_text: str = "",
        aoi_machine_coords: str = "",
        image_results_data: Optional[List[Dict]] = None,
        inference_log: str = "",
        omit_overexposed: int = 0,
        omit_overexposure_info: str = "",
        image_preprocess_pipeline: Optional[list] = None,
        image_preprocess_timing: Optional[dict] = None,
        image_preprocess_pipelines: Optional[Dict[str, list]] = None,
    ) -> int:
        """
        儲存一筆完整推論記錄

        Args:
            glass_id: 玻璃 ID
            model_id: 機種 ID
            machine_no: 機台編號
            resolution: (寬, 高) 解析度
            machine_judgment: 機檢判定
            ai_judgment: AI 判定
            image_dir: 圖片目錄路徑
            total_images: 總圖片數
            ng_images: NG 圖片數
            ng_details: NG 詳細描述 (JSON string)
            request_time: 接收請求時間
            response_time: 回覆時間
            processing_seconds: 處理耗時 (秒)
            heatmap_dir: 熱力圖儲存目錄
            error_message: 錯誤訊息
            client_bomb_info: 客戶端傳來的炸彈座標資訊 (JSON 字串)
            client_request_text: 收到 Client 的原始請求字串
            client_response_text: 回覆 Client 的原始結果字串
            aoi_machine_coords: AOI 機台檢測座標 (TXT 報告解析, JSON 字串)
            image_results_data: 圖片級結果列表

        Returns:
            record_id
        """
        with self._lock:
            conn = self._get_conn()
            try:
                preprocess_json = (
                    json.dumps(image_preprocess_pipeline, ensure_ascii=False)
                    if image_preprocess_pipeline is not None else ""
                )
                preprocess_timing_json = (
                    json.dumps(image_preprocess_timing, ensure_ascii=False)
                    if image_preprocess_timing is not None else ""
                )
                preprocess_zones_json = (
                    json.dumps(image_preprocess_pipelines, ensure_ascii=False)
                    if image_preprocess_pipelines else ""
                )
                cursor = conn.execute(
                    """INSERT INTO inference_records
                       (glass_id, model_id, machine_no, resolution_x, resolution_y,
                        machine_judgment, ai_judgment, image_dir, total_images, ng_images,
                        ng_details, request_time, response_time, processing_seconds,
                        heatmap_dir, error_message, client_bomb_info,
                        client_request_text, client_response_text, aoi_machine_coords,
                        inference_log, omit_overexposed, omit_overexposure_info,
                        image_preprocess_pipeline, image_preprocess_pipelines,
                        image_preprocess_timing)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (glass_id, model_id, machine_no, resolution[0], resolution[1],
                     machine_judgment, ai_judgment, image_dir, total_images, ng_images,
                     ng_details, request_time, response_time, processing_seconds,
                     heatmap_dir, error_message, client_bomb_info,
                     client_request_text, client_response_text, aoi_machine_coords,
                     inference_log, omit_overexposed, omit_overexposure_info,
                     preprocess_json, preprocess_zones_json, preprocess_timing_json)
                )
                record_id = cursor.lastrowid

                # 儲存圖片級結果
                if image_results_data:
                    for img_data in image_results_data:
                        img_cursor = conn.execute(
                            """INSERT INTO image_results
                               (record_id, image_path, image_name, image_width, image_height,
                                otsu_bounds, tile_count, excluded_tiles, anomaly_count,
                                max_score, is_ng, is_dust_only, is_bomb, inference_time_ms,
                                heatmap_path, scratch_filter_count,
                                mark_text, mark_confidence, mark_bbox, mark_roi,
                                mark_orientation, mark_source_image)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (record_id,
                             img_data.get("image_path", ""),
                             img_data.get("image_name", ""),
                             img_data.get("image_width", 0),
                             img_data.get("image_height", 0),
                             img_data.get("otsu_bounds", ""),
                             img_data.get("tile_count", 0),
                             img_data.get("excluded_tiles", 0),
                             img_data.get("anomaly_count", 0),
                             img_data.get("max_score", 0.0),
                             img_data.get("is_ng", 0),
                             img_data.get("is_dust_only", 0),
                             img_data.get("is_bomb", 0),
                             img_data.get("inference_time_ms", 0.0),
                             img_data.get("heatmap_path", ""),
                             img_data.get("scratch_filter_count", 0),
                             img_data.get("mark_text", ""),
                             img_data.get("mark_confidence", 0.0),
                             img_data.get("mark_bbox", ""),
                             img_data.get("mark_roi", ""),
                             img_data.get("mark_orientation", ""),
                             img_data.get("mark_source_image", ""))
                        )
                        image_result_id = img_cursor.lastrowid

                        # 儲存 tile 級結果
                        for tile_data in img_data.get("tiles", []):
                            conn.execute(
                                """INSERT INTO tile_results
                                   (image_result_id, tile_id, x, y, width, height,
                                    score, is_anomaly, is_dust, dust_iou, is_bomb,
                                    bomb_code, peak_x, peak_y, heatmap_path,
                                    is_exclude_zone, is_aoi_coord, aoi_defect_code,
                                    aoi_product_x, aoi_product_y,
                                    aoi_image_x, aoi_image_y,
                                    aoi_tile_shift_dx, aoi_tile_shift_dy,
                                    scratch_score, scratch_filtered, zone)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (image_result_id,
                                 tile_data.get("tile_id", 0),
                                 tile_data.get("x", 0),
                                 tile_data.get("y", 0),
                                 tile_data.get("width", 0),
                                 tile_data.get("height", 0),
                                 tile_data.get("score", 0.0),
                                 tile_data.get("is_anomaly", 0),
                                 tile_data.get("is_dust", 0),
                                 tile_data.get("dust_iou", 0.0),
                                 tile_data.get("is_bomb", 0),
                                 tile_data.get("bomb_code", ""),
                                 tile_data.get("peak_x", -1),
                                 tile_data.get("peak_y", -1),
                                 tile_data.get("heatmap_path", ""),
                                 tile_data.get("is_exclude_zone", 0),
                                 tile_data.get("is_aoi_coord", 0),
                                 tile_data.get("aoi_defect_code", ""),
                                 tile_data.get("aoi_product_x", -1),
                                 tile_data.get("aoi_product_y", -1),
                                 tile_data.get("aoi_image_x", -1),
                                 tile_data.get("aoi_image_y", -1),
                                 tile_data.get("aoi_tile_shift_dx", 0),
                                 tile_data.get("aoi_tile_shift_dy", 0),
                                 tile_data.get("scratch_score", 0.0),
                                 int(tile_data.get("scratch_filtered", 0)),
                                 tile_data.get("zone", ""))
                            )

                        # 儲存 CV 邊緣缺陷結果
                        for edge_data in img_data.get("edge_defects", []):
                            conn.execute(
                                """INSERT INTO edge_defect_results
                                   (image_result_id, side, area,
                                    bbox_x, bbox_y, bbox_w, bbox_h,
                                    max_diff, center_x, center_y, heatmap_path,
                                    is_dust, is_bomb, bomb_code, is_cv_ok,
                                    threshold_used, min_area_used, min_max_diff_used,
                                    inspector_mode, patchcore_score,
                                    patchcore_threshold, patchcore_ok_reason,
                                    source_inspector, d_edge_px, fusion_fallback_reason,
                                    pc_roi_origin_x, pc_roi_origin_y,
                                    pc_roi_shift_dx, pc_roi_shift_dy, pc_roi_fallback_reason)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (image_result_id,
                                 edge_data.get("side", ""),
                                 edge_data.get("area", 0),
                                 edge_data.get("bbox_x", 0),
                                 edge_data.get("bbox_y", 0),
                                 edge_data.get("bbox_w", 0),
                                 edge_data.get("bbox_h", 0),
                                 edge_data.get("max_diff", 0.0),
                                 edge_data.get("center_x", 0),
                                 edge_data.get("center_y", 0),
                                 edge_data.get("heatmap_path", ""),
                                 edge_data.get("is_dust", 0),
                                 edge_data.get("is_bomb", 0),
                                 edge_data.get("bomb_code", ""),
                                 edge_data.get("is_cv_ok", 0),
                                 edge_data.get("threshold_used", 0),
                                 edge_data.get("min_area_used", 0),
                                 edge_data.get("min_max_diff_used", 0),
                                 edge_data.get("inspector_mode", "cv"),
                                 edge_data.get("patchcore_score", 0.0),
                                 edge_data.get("patchcore_threshold", 0.0),
                                 edge_data.get("patchcore_ok_reason", ""),
                                 edge_data.get("source_inspector", ""),
                                 edge_data.get("d_edge_px", 0.0),
                                 edge_data.get("fusion_fallback_reason", ""),
                                 edge_data.get("pc_roi_origin_x", 0),
                                 edge_data.get("pc_roi_origin_y", 0),
                                 edge_data.get("pc_roi_shift_dx", 0),
                                 edge_data.get("pc_roi_shift_dy", 0),
                                 edge_data.get("pc_roi_fallback_reason", ""))
                            )

                conn.commit()
                return record_id
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def update_record_for_rerun(
        self,
        record_id: int,
        ai_judgment: str,
        total_images: int,
        ng_images: int,
        ng_details: str,
        processing_seconds: float,
        heatmap_dir: str = "",
        error_message: str = "",
        machine_judgment: Optional[str] = None,
        aoi_machine_coords: Optional[str] = None,
        image_results_data: Optional[List[Dict]] = None,
        inference_log: str = "",
        omit_overexposed: int = 0,
        omit_overexposure_info: str = "",
        image_preprocess_pipeline: Optional[list] = None,
        image_preprocess_timing: Optional[dict] = None,
        image_preprocess_pipelines: Optional[Dict[str, list]] = None,
    ) -> None:
        """
        重新推論後覆蓋更新紀錄 (同一 record_id)

        1. 刪除舊的 tile_results, edge_defect_results, image_results
        2. 更新 inference_records 欄位
        3. 插入新的 image_results, tile_results, edge_defect_results
        """
        with self._lock:
            conn = self._get_conn()
            try:
                # --- 刪除舊的子紀錄 (CASCADE 會自動刪除 tile_results, edge_defect_results) ---
                conn.execute(
                    "DELETE FROM image_results WHERE record_id = ?",
                    (record_id,),
                )

                # --- 更新主紀錄 ---
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                preprocess_json = (
                    json.dumps(image_preprocess_pipeline, ensure_ascii=False)
                    if image_preprocess_pipeline is not None else ""
                )
                preprocess_timing_json = (
                    json.dumps(image_preprocess_timing, ensure_ascii=False)
                    if image_preprocess_timing is not None else ""
                )
                preprocess_zones_json = (
                    json.dumps(image_preprocess_pipelines, ensure_ascii=False)
                    if image_preprocess_pipelines else ""
                )
                cursor = conn.execute(
                    """UPDATE inference_records SET
                           ai_judgment = ?,
                           machine_judgment = COALESCE(?, machine_judgment),
                           total_images = ?,
                           ng_images = ?,
                           ng_details = ?,
                           response_time = ?,
                           processing_seconds = ?,
                           heatmap_dir = ?,
                           error_message = ?,
                           aoi_machine_coords = COALESCE(?, aoi_machine_coords),
                           inference_log = ?,
                           omit_overexposed = ?,
                           omit_overexposure_info = ?,
                           image_preprocess_pipeline = ?,
                           image_preprocess_pipelines = ?,
                           image_preprocess_timing = ?
                       WHERE id = ?""",
                    (ai_judgment, machine_judgment, total_images, ng_images, ng_details,
                     now_str, processing_seconds, heatmap_dir, error_message,
                     aoi_machine_coords, inference_log, omit_overexposed, omit_overexposure_info,
                     preprocess_json, preprocess_zones_json, preprocess_timing_json,
                     record_id),
                )
                if cursor.rowcount == 0:
                    raise ValueError(f"update_record_for_rerun: record_id {record_id} not found")

                # --- 插入新的子紀錄 ---
                if image_results_data:
                    for img_data in image_results_data:
                        img_cursor = conn.execute(
                            """INSERT INTO image_results
                               (record_id, image_path, image_name, image_width, image_height,
                                otsu_bounds, tile_count, excluded_tiles, anomaly_count,
                                max_score, is_ng, is_dust_only, is_bomb, inference_time_ms,
                                heatmap_path, scratch_filter_count,
                                mark_text, mark_confidence, mark_bbox, mark_roi,
                                mark_orientation, mark_source_image)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (record_id,
                             img_data.get("image_path", ""),
                             img_data.get("image_name", ""),
                             img_data.get("image_width", 0),
                             img_data.get("image_height", 0),
                             img_data.get("otsu_bounds", ""),
                             img_data.get("tile_count", 0),
                             img_data.get("excluded_tiles", 0),
                             img_data.get("anomaly_count", 0),
                             img_data.get("max_score", 0.0),
                             img_data.get("is_ng", 0),
                             img_data.get("is_dust_only", 0),
                             img_data.get("is_bomb", 0),
                             img_data.get("inference_time_ms", 0.0),
                             img_data.get("heatmap_path", ""),
                             img_data.get("scratch_filter_count", 0),
                             img_data.get("mark_text", ""),
                             img_data.get("mark_confidence", 0.0),
                             img_data.get("mark_bbox", ""),
                             img_data.get("mark_roi", ""),
                             img_data.get("mark_orientation", ""),
                             img_data.get("mark_source_image", ""))
                        )
                        image_result_id = img_cursor.lastrowid

                        for tile_data in img_data.get("tiles", []):
                            conn.execute(
                                """INSERT INTO tile_results
                                   (image_result_id, tile_id, x, y, width, height,
                                    score, is_anomaly, is_dust, dust_iou, is_bomb,
                                    bomb_code, peak_x, peak_y, heatmap_path,
                                    is_exclude_zone, is_aoi_coord, aoi_defect_code,
                                    aoi_product_x, aoi_product_y,
                                    aoi_image_x, aoi_image_y,
                                    aoi_tile_shift_dx, aoi_tile_shift_dy,
                                    scratch_score, scratch_filtered, zone)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (image_result_id,
                                 tile_data.get("tile_id", 0),
                                 tile_data.get("x", 0),
                                 tile_data.get("y", 0),
                                 tile_data.get("width", 0),
                                 tile_data.get("height", 0),
                                 tile_data.get("score", 0.0),
                                 tile_data.get("is_anomaly", 0),
                                 tile_data.get("is_dust", 0),
                                 tile_data.get("dust_iou", 0.0),
                                 tile_data.get("is_bomb", 0),
                                 tile_data.get("bomb_code", ""),
                                 tile_data.get("peak_x", -1),
                                 tile_data.get("peak_y", -1),
                                 tile_data.get("heatmap_path", ""),
                                 tile_data.get("is_exclude_zone", 0),
                                 tile_data.get("is_aoi_coord", 0),
                                 tile_data.get("aoi_defect_code", ""),
                                 tile_data.get("aoi_product_x", -1),
                                 tile_data.get("aoi_product_y", -1),
                                 tile_data.get("aoi_image_x", -1),
                                 tile_data.get("aoi_image_y", -1),
                                 tile_data.get("aoi_tile_shift_dx", 0),
                                 tile_data.get("aoi_tile_shift_dy", 0),
                                 tile_data.get("scratch_score", 0.0),
                                 int(tile_data.get("scratch_filtered", 0)),
                                 tile_data.get("zone", ""))
                            )

                        for edge_data in img_data.get("edge_defects", []):
                            conn.execute(
                                """INSERT INTO edge_defect_results
                                   (image_result_id, side, area,
                                    bbox_x, bbox_y, bbox_w, bbox_h,
                                    max_diff, center_x, center_y, heatmap_path,
                                    is_dust, is_bomb, bomb_code, is_cv_ok,
                                    threshold_used, min_area_used, min_max_diff_used,
                                    inspector_mode, patchcore_score,
                                    patchcore_threshold, patchcore_ok_reason,
                                    source_inspector, d_edge_px, fusion_fallback_reason,
                                    pc_roi_origin_x, pc_roi_origin_y,
                                    pc_roi_shift_dx, pc_roi_shift_dy, pc_roi_fallback_reason)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (image_result_id,
                                 edge_data.get("side", ""),
                                 edge_data.get("area", 0),
                                 edge_data.get("bbox_x", 0),
                                 edge_data.get("bbox_y", 0),
                                 edge_data.get("bbox_w", 0),
                                 edge_data.get("bbox_h", 0),
                                 edge_data.get("max_diff", 0.0),
                                 edge_data.get("center_x", 0),
                                 edge_data.get("center_y", 0),
                                 edge_data.get("heatmap_path", ""),
                                 edge_data.get("is_dust", 0),
                                 edge_data.get("is_bomb", 0),
                                 edge_data.get("bomb_code", ""),
                                 edge_data.get("is_cv_ok", 0),
                                 edge_data.get("threshold_used", 0),
                                 edge_data.get("min_area_used", 0),
                                 edge_data.get("min_max_diff_used", 0),
                                 edge_data.get("inspector_mode", "cv"),
                                 edge_data.get("patchcore_score", 0.0),
                                 edge_data.get("patchcore_threshold", 0.0),
                                 edge_data.get("patchcore_ok_reason", ""),
                                 edge_data.get("source_inspector", ""),
                                 edge_data.get("d_edge_px", 0.0),
                                 edge_data.get("fusion_fallback_reason", ""),
                                 edge_data.get("pc_roi_origin_x", 0),
                                 edge_data.get("pc_roi_origin_y", 0),
                                 edge_data.get("pc_roi_shift_dx", 0),
                                 edge_data.get("pc_roi_shift_dy", 0),
                                 edge_data.get("pc_roi_fallback_reason", ""))
                            )

                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def query_by_glass_id(self, glass_id: str) -> List[Dict]:
        """依玻璃 ID 查詢推論記錄"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM inference_records
                   WHERE glass_id = ?
                   ORDER BY created_at DESC""",
                (glass_id,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def query_by_date_range(
        self, start_date: str, end_date: str, limit: int = 1000
    ) -> List[Dict]:
        """依日期範圍查詢"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM inference_records
                   WHERE created_at >= ? AND created_at <= ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (start_date, end_date, limit)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def query_recent(self, limit: int = 50) -> List[Dict]:
        """查詢最近 N 筆推論記錄"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM inference_records
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def query_overexposed(self, limit: int = 50, offset: int = 0) -> tuple:
        """查詢過曝記錄，回傳 (records, total_count)"""
        conn = self._get_conn()
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM inference_records WHERE omit_overexposed = 1"
            ).fetchone()[0]
            rows = conn.execute(
                """SELECT * FROM inference_records
                   WHERE omit_overexposed = 1
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset)
            ).fetchall()
            return [dict(r) for r in rows], total
        finally:
            conn.close()

    def query_paged(self, limit: int = 50, offset: int = 0) -> tuple:
        """分頁查詢推論記錄，回傳 (records, total_count)"""
        conn = self._get_conn()
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM inference_records"
            ).fetchone()[0]
            rows = conn.execute(
                """SELECT * FROM inference_records
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset)
            ).fetchall()
            return [dict(r) for r in rows], total
        finally:
            conn.close()

    def get_record_detail(self, record_id: int) -> Optional[Dict]:
        """取得完整推論記錄 (含圖片和 tile 結果)"""
        conn = self._get_conn()
        try:
            record = conn.execute(
                "SELECT * FROM inference_records WHERE id = ?",
                (record_id,)
            ).fetchone()
            if not record:
                return None

            result = dict(record)
            latest_within_spec_log = conn.execute(
                """SELECT id FROM within_spec_review_log
                   WHERE inference_record_id = ?
                   ORDER BY id DESC LIMIT 1""",
                (record_id,)
            ).fetchone()
            result["within_spec_log_id"] = latest_within_spec_log["id"] if latest_within_spec_log else None

            # 取得圖片結果
            images = conn.execute(
                """SELECT * FROM image_results
                   WHERE record_id = ?
                   ORDER BY id""",
                (record_id,)
            ).fetchall()
            result["images"] = []
            for img in images:
                img_dict = dict(img)
                # 取得 tile 結果 (NG優先、然後依分數降冪)
                tiles = conn.execute(
                    """SELECT * FROM tile_results
                       WHERE image_result_id = ?
                       ORDER BY 
                           CASE 
                               WHEN is_dust = 0 AND is_bomb = 0 THEN 1
                               WHEN is_bomb = 1 THEN 2
                               WHEN is_dust = 1 THEN 3
                               ELSE 4
                           END ASC,
                           score DESC,
                           tile_id ASC""",
                    (img_dict["id"],)
                ).fetchall()
                img_dict["tiles"] = [dict(t) for t in tiles]

                # 取得 CV 邊緣缺陷結果
                edge_defects = conn.execute(
                    """SELECT * FROM edge_defect_results
                       WHERE image_result_id = ?
                       ORDER BY is_dust ASC, max_diff DESC, patchcore_score DESC""",
                    (img_dict["id"],)
                ).fetchall()
                img_dict["edge_defects"] = [dict(e) for e in edge_defects]

                result["images"].append(img_dict)

            return result
        finally:
            conn.close()

    def get_statistics(self, days: int = 7) -> Dict:
        """取得統計摘要"""
        conn = self._get_conn()
        try:
            stats = {}

            # 總記錄數
            row = conn.execute("SELECT COUNT(*) as cnt FROM inference_records").fetchone()
            stats["total_records"] = row["cnt"]

            # 最近 N 天統計
            row = conn.execute(
                """SELECT
                     COUNT(*) as total,
                     SUM(CASE WHEN ai_judgment = 'OK' OR ai_judgment = 'OK-i' THEN 1 ELSE 0 END) as ok_count,
                     SUM(CASE WHEN ai_judgment = 'NG' OR ai_judgment LIKE 'NG%' THEN 1 ELSE 0 END) as ng_count,
                     SUM(CASE WHEN ai_judgment LIKE 'ERR%' THEN 1 ELSE 0 END) as err_count,
                     AVG(processing_seconds) as avg_time,
                     MAX(processing_seconds) as max_time,
                     MIN(processing_seconds) as min_time
                   FROM inference_records
                   WHERE created_at >= datetime('now', 'localtime', ?)""",
                (f"-{days} days",)
            ).fetchone()
            stats["recent"] = dict(row) if row else {}

            # 按機台統計
            rows = conn.execute(
                """SELECT machine_no,
                     COUNT(*) as total,
                     SUM(CASE WHEN ai_judgment = 'OK' OR ai_judgment = 'OK-i' THEN 1 ELSE 0 END) as ok_count,
                     SUM(CASE WHEN ai_judgment = 'NG' OR ai_judgment LIKE 'NG%' THEN 1 ELSE 0 END) as ng_count
                   FROM inference_records
                   WHERE created_at >= datetime('now', 'localtime', ?)
                   GROUP BY machine_no
                   ORDER BY total DESC""",
                (f"-{days} days",)
            ).fetchall()
            stats["by_machine"] = [dict(r) for r in rows]

            return stats
        finally:
            conn.close()

    @staticmethod
    def _get_shift_window(now: datetime) -> Tuple[str, datetime, datetime]:
        minutes = now.hour * 60 + now.minute
        day_start = 7 * 60 + 30
        night_start = 19 * 60 + 30
        if day_start <= minutes < night_start:
            # 白班：當日 07:30 ~ 19:30
            shift_name = "白班"
            shift_start = now.replace(hour=7, minute=30, second=0, microsecond=0)
            shift_end = now.replace(hour=19, minute=30, second=0, microsecond=0)
        elif minutes >= night_start:
            # 夜班：當日 19:30 ~ 隔日 07:30
            shift_name = "夜班"
            shift_start = now.replace(hour=19, minute=30, second=0, microsecond=0)
            shift_end = (now + timedelta(days=1)).replace(hour=7, minute=30, second=0, microsecond=0)
        else:
            # 夜班：前日 19:30 ~ 當日 07:30
            shift_name = "夜班"
            shift_start = (now - timedelta(days=1)).replace(hour=19, minute=30, second=0, microsecond=0)
            shift_end = now.replace(hour=7, minute=30, second=0, microsecond=0)
        return shift_name, shift_start, shift_end

    def get_shift_statistics(self, now: Optional[datetime] = None) -> Dict:
        """取得當班統計（白班 07:30~19:30 / 夜班 19:30~07:30）"""
        shift_name, shift_start, shift_end = self._get_shift_window(now or datetime.now())

        start_str = shift_start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = shift_end.strftime("%Y-%m-%d %H:%M:%S")
        time_range_label = f"{shift_start.strftime('%m/%d %H:%M')} ~ {shift_end.strftime('%m/%d %H:%M')}"

        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT
                     COUNT(*) as total,
                     SUM(CASE WHEN ai_judgment = 'OK' OR ai_judgment = 'OK-i' THEN 1 ELSE 0 END) as ok_count,
                     SUM(CASE WHEN ai_judgment = 'NG' OR ai_judgment LIKE 'NG%' THEN 1 ELSE 0 END) as ng_count,
                     SUM(CASE WHEN ai_judgment LIKE 'ERR%' THEN 1 ELSE 0 END) as err_count,
                     AVG(processing_seconds) as avg_time,
                     SUM(omit_overexposed) as overexposed_count
                   FROM inference_records
                   WHERE datetime(created_at) >= datetime(?) AND datetime(created_at) < datetime(?)""",
                (start_str, end_str)
            ).fetchone()

            result = dict(row) if row else {}
            result["shift_name"] = shift_name
            result["time_range"] = time_range_label
            return result
        finally:
            conn.close()

    def search_records(
        self,
        glass_id: str = "",
        machine_no: str = "",
        ai_judgment: str = "",
        start_date: str = "",
        end_date: str = "",
        cross_filter: str = "",
        record_id: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict], int]:
        """多條件搜尋，回傳 (records, total_count)"""
        conditions = []
        params = []

        if record_id:
            conditions.append("CAST(id AS TEXT) LIKE ?")
            params.append(f"%{record_id}%")
        if glass_id:
            conditions.append("glass_id LIKE ?")
            params.append(f"%{glass_id}%")
        if machine_no:
            conditions.append("machine_no LIKE ?")
            params.append(f"%{machine_no}%")

        if cross_filter == "ng_ok":
            conditions.append(
                f"({self._AOI_NG_COND}) AND NOT ({self._AI_NG_COND}) AND NOT ({self._ERR_COND})"
            )
        elif cross_filter == "ok_ng":
            conditions.append(
                f"NOT ({self._AOI_NG_COND}) AND ({self._AI_NG_COND}) AND NOT ({self._ERR_COND})"
            )
        elif ai_judgment:
            conditions.append("ai_judgment LIKE ?")
            params.append(f"%{ai_judgment}%")

        # 使用 request_time 與 RIC Report 一致，日期代表 07:30 起算的工廠生產日
        if start_date:
            conditions.append("datetime(request_time) >= datetime(?)")
            params.append(_factory_day_start_ts(start_date))
        if end_date:
            conditions.append("datetime(request_time) < datetime(?)")
            params.append(_factory_day_end_ts(end_date))

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        conn = self._get_conn()
        try:
            total_count = conn.execute(
                f"SELECT COUNT(*) FROM inference_records WHERE {where_clause}",
                params
            ).fetchone()[0]

            rows = conn.execute(
                f"""SELECT * FROM inference_records
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?""",
                params + [limit, offset]
            ).fetchall()
            return [dict(r) for r in rows], total_count
        finally:
            conn.close()

    # ── RIC 人工檢驗相關方法 ─────────────────────────────────

    @staticmethod
    def parse_ric_judgment(datastr: str) -> str:
        """
        解析 DATASTR 欄位判定 RIC 結果
        例: "WGF50500,OK;STANDARD,NG;R0F00000,NG;W0F00000,OK;4;"
        任一項含 NG → 回傳 "NG"，否則 "OK"
        """
        if not datastr:
            return "OK"
        parts = datastr.strip().rstrip(";").split(";")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # 最後一項可能純數字 (計數)，跳過
            if part.isdigit():
                continue
            if "," in part:
                _, result = part.rsplit(",", 1)
                if result.strip().upper() == "NG":
                    return "NG"
        return "OK"

    def save_ric_batch(self, filename: str, records_data: List[Dict]) -> int:
        """
        儲存一批 RIC 匯入資料

        Args:
            filename: 匯入的檔案名稱
            records_data: RIC 記錄列表，每筆含 TIMESTAMP, ID, PNL_ID, ... 欄位

        Returns:
            batch_id
        """
        import_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    """INSERT INTO ric_import_batches (filename, total_records, import_time)
                       VALUES (?, ?, ?)""",
                    (filename, len(records_data), import_time)
                )
                batch_id = cursor.lastrowid

                for rec in records_data:
                    datastr = rec.get("DATASTR", "")
                    ric_judgment = self.parse_ric_judgment(datastr)
                    conn.execute(
                        """INSERT INTO ric_records
                           (batch_id, timestamp, ric_id, pnl_id, prod_id,
                            mach_id, station, ipaddress, keytime, datastr, ric_judgment)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (batch_id,
                         rec.get("TIMESTAMP", ""),
                         rec.get("ID", ""),
                         rec.get("PNL_ID", ""),
                         rec.get("PROD_ID", ""),
                         rec.get("MACH_ID", ""),
                         rec.get("STATION", ""),
                         rec.get("IPADDRESS", ""),
                         rec.get("KEYTIME", ""),
                         datastr,
                         ric_judgment)
                    )

                conn.commit()
                return batch_id
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def save_client_accuracy_records(self, records: list) -> dict:
        """
        儲存 client accuracy records (TIME_STAMP + PNL_ID 為唯一鍵)
        重複資料自動跳過 (INSERT OR IGNORE)

        Args:
            records: list of dict with keys: time_stamp, pnl_id, mach_id, result_eqp, result_ai, result_ric, datastr

        Returns:
            dict with inserted, skipped counts
        """
        with self._lock:
            conn = self._get_conn()
            try:
                count_before = conn.execute(
                    "SELECT COUNT(*) FROM client_accuracy_records"
                ).fetchone()[0]

                params = [
                    (rec.get("time_stamp", ""),
                     rec.get("pnl_id", ""),
                     rec.get("mach_id", ""),
                     rec.get("result_eqp", ""),
                     rec.get("result_ai", ""),
                     rec.get("result_ric", ""),
                     rec.get("datastr", ""))
                    for rec in records
                ]
                conn.executemany(
                    """INSERT OR IGNORE INTO client_accuracy_records
                       (time_stamp, pnl_id, mach_id, result_eqp, result_ai, result_ric, datastr)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    params
                )
                conn.commit()

                count_after = conn.execute(
                    "SELECT COUNT(*) FROM client_accuracy_records"
                ).fetchone()[0]

                inserted = count_after - count_before
                return {"inserted": inserted, "skipped": len(records) - inserted}
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def get_client_accuracy_records(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> list:
        """取得 client accuracy records，支援日期篩選，並 LEFT JOIN miss_review"""
        if start_date and not _DATE_RE.match(start_date):
            raise ValueError(f"Invalid start_date format: {start_date}")
        if end_date and not _DATE_RE.match(end_date):
            raise ValueError(f"Invalid end_date format: {end_date}")

        conn = self._get_conn()
        try:
            where_clauses = []
            params = []
            if start_date:
                where_clauses.append("datetime(c.time_stamp) >= datetime(?)")
                params.append(_factory_day_start_ts(start_date))
            if end_date:
                where_clauses.append("datetime(c.time_stamp) < datetime(?)")
                params.append(_factory_day_end_ts(end_date))
            where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

            rows = conn.execute(
                f"""SELECT c.id, c.time_stamp, c.pnl_id, c.mach_id,
                           c.result_eqp, c.result_ai, c.result_ric, c.datastr,
                           mr.id as review_id, mr.category as review_category,
                           mr.note as review_note, mr.updated_at as review_updated_at,
                           ovr.id as over_review_id, ovr.category as over_review_category,
                           ovr.note as over_review_note, ovr.updated_at as over_review_updated_at,
                           wsl.id as within_spec_log_id,
                           wsl.suggested as within_spec_suggested,
                           wsl.category as within_spec_category,
                           wsl.reason as within_spec_reason,
                           wsl.error_message as within_spec_error,
                           wsl.processing_seconds as within_spec_processing_seconds,
                           wsl.created_at as within_spec_created_at,
                           (SELECT COUNT(*) FROM over_retrain_pool orp
                            WHERE orp.client_record_id = c.id
                           ) as over_retrain_pool_count,
                           (SELECT MAX(created_at) FROM over_retrain_pool orp
                            WHERE orp.client_record_id = c.id
                           ) as over_retrain_pool_latest_at,
                           (SELECT ir.id FROM inference_records ir
                            WHERE ir.glass_id = c.pnl_id
                              AND datetime(ir.request_time) >= datetime(date(datetime(c.time_stamp), '-7 hours', '-30 minutes') || ' 07:30:00')
                              AND datetime(ir.request_time) < datetime(date(datetime(c.time_stamp), '-7 hours', '-30 minutes') || ' 07:30:00', '+1 day')
                            ORDER BY ir.request_time DESC LIMIT 1
                           ) as inference_record_id,
                           (SELECT ir.ai_judgment FROM inference_records ir
                            WHERE ir.glass_id = c.pnl_id
                              AND datetime(ir.request_time) >= datetime(date(datetime(c.time_stamp), '-7 hours', '-30 minutes') || ' 07:30:00')
                              AND datetime(ir.request_time) < datetime(date(datetime(c.time_stamp), '-7 hours', '-30 minutes') || ' 07:30:00', '+1 day')
                            ORDER BY ir.request_time DESC LIMIT 1
                           ) as inference_ai_judgment
                    FROM client_accuracy_records c
                    LEFT JOIN miss_review mr ON mr.client_record_id = c.id
                    LEFT JOIN over_review ovr ON ovr.client_record_id = c.id
                    LEFT JOIN within_spec_review_log wsl
                      ON wsl.id = (
                          SELECT id FROM within_spec_review_log
                          WHERE client_record_id = c.id
                          ORDER BY id DESC LIMIT 1
                      )
                    {where_sql}
                    ORDER BY c.time_stamp DESC""",
                params
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_client_accuracy_record(self, client_record_id: int) -> Optional[Dict]:
        """取得單筆 client accuracy record，附最近 inference 與 retrain pool 摘要。"""
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT c.id, c.time_stamp, c.pnl_id, c.mach_id,
                          c.result_eqp, c.result_ai, c.result_ric, c.datastr,
                          (SELECT COUNT(*) FROM over_retrain_pool orp
                           WHERE orp.client_record_id = c.id
                          ) as over_retrain_pool_count,
                          (SELECT MAX(created_at) FROM over_retrain_pool orp
                           WHERE orp.client_record_id = c.id
                          ) as over_retrain_pool_latest_at,
                          (SELECT ir.id FROM inference_records ir
                           WHERE ir.glass_id = c.pnl_id
                             AND datetime(ir.request_time) >= datetime(date(datetime(c.time_stamp), '-7 hours', '-30 minutes') || ' 07:30:00')
                             AND datetime(ir.request_time) < datetime(date(datetime(c.time_stamp), '-7 hours', '-30 minutes') || ' 07:30:00', '+1 day')
                           ORDER BY ir.request_time DESC LIMIT 1
                          ) as inference_record_id,
                          (SELECT ir.ai_judgment FROM inference_records ir
                           WHERE ir.glass_id = c.pnl_id
                             AND datetime(ir.request_time) >= datetime(date(datetime(c.time_stamp), '-7 hours', '-30 minutes') || ' 07:30:00')
                             AND datetime(ir.request_time) < datetime(date(datetime(c.time_stamp), '-7 hours', '-30 minutes') || ' 07:30:00', '+1 day')
                           ORDER BY ir.request_time DESC LIMIT 1
                          ) as inference_ai_judgment
                   FROM client_accuracy_records c
                   WHERE c.id = ?""",
                (int(client_record_id),),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # over_retrain_pool CRUD
    # ------------------------------------------------------------------

    def insert_over_retrain_pool_rows(self, rows: list) -> dict:
        """批次加入過檢重訓 pool；tile_result_id 已存在時視為 existing。"""
        if not rows:
            return {"inserted_ids": [], "existing_ids": []}
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            inserted_ids = []
            existing_ids = []
            for row in rows:
                cur.execute(
                    """INSERT OR IGNORE INTO over_retrain_pool
                       (client_record_id, inference_record_id, tile_result_id, image_result_id,
                        machine_id, machine_no, pnl_id, client_time_stamp, datastr,
                        screen_prefix, lighting, zone, source_path, thumb_path,
                        tile_x, tile_y, tile_w, tile_h, score)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        int(row["client_record_id"]),
                        int(row["inference_record_id"]),
                        int(row["tile_result_id"]),
                        int(row["image_result_id"]),
                        row["machine_id"],
                        row.get("machine_no", ""),
                        row["pnl_id"],
                        row.get("client_time_stamp", ""),
                        row.get("datastr", ""),
                        row["screen_prefix"],
                        row["lighting"],
                        row.get("zone", ""),
                        row["source_path"],
                        row.get("thumb_path", ""),
                        int(row.get("tile_x", 0) or 0),
                        int(row.get("tile_y", 0) or 0),
                        int(row.get("tile_w", 0) or 0),
                        int(row.get("tile_h", 0) or 0),
                        float(row.get("score", 0.0) or 0.0),
                    ),
                )
                if cur.rowcount:
                    inserted_ids.append(cur.lastrowid)
                    continue
                existing = cur.execute(
                    "SELECT id FROM over_retrain_pool WHERE tile_result_id = ?",
                    (int(row["tile_result_id"]),),
                ).fetchone()
                if existing:
                    existing_ids.append(existing["id"])
            conn.commit()
            return {"inserted_ids": inserted_ids, "existing_ids": existing_ids}
        finally:
            conn.close()

    def list_over_retrain_pool(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        machine_id: Optional[str] = None,
        lighting: Optional[str] = None,
        zone: Optional[str] = None,
        client_record_id: Optional[int] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> Tuple[list, int]:
        """查詢過檢重訓 pool，支援日期/機種/lighting/zone/來源 record 過濾。"""
        if start_date and not _DATE_RE.match(start_date):
            raise ValueError(f"Invalid start_date format: {start_date}")
        if end_date and not _DATE_RE.match(end_date):
            raise ValueError(f"Invalid end_date format: {end_date}")

        def _next_date_str(date_str: str) -> str:
            from datetime import timedelta
            return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

        where = []
        params: list = []
        if start_date:
            where.append("created_at >= ?")
            params.append(start_date)
        if end_date:
            where.append("created_at < ?")
            params.append(_next_date_str(end_date))
        if machine_id:
            where.append("machine_id = ?")
            params.append(machine_id)
        if lighting:
            where.append("lighting = ?")
            params.append(lighting)
        if zone is not None and zone != "":
            where.append("zone = ?")
            params.append(zone)
        if client_record_id:
            where.append("client_record_id = ?")
            params.append(int(client_record_id))

        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        limit = max(1, min(int(limit), 1000))
        offset = max(0, int(offset))
        conn = self._get_conn()
        try:
            total = conn.execute(
                f"SELECT COUNT(*) as cnt FROM over_retrain_pool{where_sql}",
                params,
            ).fetchone()["cnt"]
            rows = conn.execute(
                f"""SELECT * FROM over_retrain_pool
                    {where_sql}
                    ORDER BY created_at DESC, id DESC
                    LIMIT ? OFFSET ?""",
                params + [limit, offset],
            ).fetchall()
            return [dict(r) for r in rows], int(total or 0)
        finally:
            conn.close()

    def get_over_retrain_pool_items(self, pool_ids: list) -> list:
        """依 id 清單查詢 pool item。"""
        ids = [int(x) for x in pool_ids if str(x).strip()]
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        conn = self._get_conn()
        try:
            rows = conn.execute(
                f"SELECT * FROM over_retrain_pool WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
            by_id = {int(r["id"]): dict(r) for r in rows}
            return [by_id[i] for i in ids if i in by_id]
        finally:
            conn.close()

    def delete_over_retrain_pool_item(self, pool_id: int) -> Optional[Dict]:
        """刪除單筆 pool item，回傳刪除前資料；找不到回 None。"""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM over_retrain_pool WHERE id = ?",
                (int(pool_id),),
            ).fetchone()
            if not row:
                return None
            data = dict(row)
            conn.execute("DELETE FROM over_retrain_pool WHERE id = ?", (int(pool_id),))
            conn.commit()
            return data
        finally:
            conn.close()

    def mark_over_retrain_pool_added(
        self,
        pool_ids: list,
        bundle_id: int,
        job_id: str,
        unit_label: str,
    ) -> None:
        """標記 pool item 已匯入特定 bundle/job/unit。"""
        ids = [int(x) for x in pool_ids if str(x).strip()]
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        conn = self._get_conn()
        try:
            conn.execute(
                f"""UPDATE over_retrain_pool
                    SET added_to_bundle_id = ?,
                        added_to_job_id = ?,
                        added_to_unit = ?,
                        updated_at = datetime('now', 'localtime')
                    WHERE id IN ({placeholders})""",
                (int(bundle_id), job_id, unit_label, *ids),
            )
            conn.commit()
        finally:
            conn.close()

    def clear_over_retrain_pool_added(self, pool_ids: list) -> int:
        """清除 pool item 的已匯入訓練清單標記，回傳更新筆數。"""
        ids = [int(x) for x in pool_ids if str(x).strip()]
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        conn = self._get_conn()
        try:
            cur = conn.execute(
                f"""UPDATE over_retrain_pool
                    SET added_to_bundle_id = NULL,
                        added_to_job_id = '',
                        added_to_unit = '',
                        updated_at = datetime('now', 'localtime')
                    WHERE id IN ({placeholders})""",
                ids,
            )
            conn.commit()
            return int(cur.rowcount or 0)
        finally:
            conn.close()

    def get_dust_affected_record_ids(self, record_ids: list) -> set:
        """返回有灰塵過濾影響的 inference record IDs (image/tile/edge dust flags)."""
        if not record_ids:
            return set()
        conn = self._get_conn()
        try:
            result_ids = set()
            chunk_size = 450  # Each ID appears twice in UNION, stay under SQLite 999 limit
            for i in range(0, len(record_ids), chunk_size):
                chunk = record_ids[i:i + chunk_size]
                ph = ','.join('?' * len(chunk))
                rows = conn.execute(
                    f"SELECT DISTINCT record_id FROM image_results "
                    f"WHERE record_id IN ({ph}) AND is_dust_only = 1 "
                    f"UNION "
                    f"SELECT DISTINCT img.record_id "
                    f"FROM tile_results t "
                    f"JOIN image_results img ON img.id = t.image_result_id "
                    f"WHERE img.record_id IN ({ph}) AND t.is_dust = 1 "
                    f"UNION "
                    f"SELECT DISTINCT img.record_id "
                    f"FROM edge_defect_results edr "
                    f"JOIN image_results img ON img.id = edr.image_result_id "
                    f"WHERE img.record_id IN ({ph}) AND img.is_ng = 0 AND edr.is_dust = 1",
                    chunk + chunk + chunk
                ).fetchall()
                result_ids.update(r["record_id"] for r in rows)
            return result_ids
        finally:
            conn.close()

    def get_scratch_rescue_stats(self, record_ids: list) -> dict:
        """回傳 {record_id: {"tiles": N, "images": N}} — DINOv2 scratch filter 救回統計。

        tiles: 該 record 下 scratch_filtered=1 的 tile 總數
        images: 該 record 下至少有 1 個 tile 被救回的 image 數
        （未被列入結果的 record_id = 無救回）
        """
        if not record_ids:
            return {}
        conn = self._get_conn()
        try:
            stats: dict = {}
            chunk_size = 900
            for i in range(0, len(record_ids), chunk_size):
                chunk = record_ids[i:i + chunk_size]
                ph = ','.join('?' * len(chunk))
                rows = conn.execute(
                    f"SELECT img.record_id AS rid, "
                    f"       COUNT(DISTINCT img.id) AS img_cnt, "
                    f"       SUM(CASE WHEN t.scratch_filtered=1 THEN 1 ELSE 0 END) AS tile_cnt "
                    f"FROM tile_results t "
                    f"JOIN image_results img ON img.id = t.image_result_id "
                    f"WHERE img.record_id IN ({ph}) AND t.scratch_filtered = 1 "
                    f"GROUP BY img.record_id",
                    chunk
                ).fetchall()
                for r in rows:
                    stats[r["rid"]] = {
                        "tiles": int(r["tile_cnt"] or 0),
                        "images": int(r["img_cnt"] or 0),
                    }
            return stats
        finally:
            conn.close()

    _SCRATCH_REVIEW_ORDER = {
        "latest": "ir.created_at DESC, t.id DESC",
        "score_asc": "t.scratch_score ASC, ir.created_at DESC",
    }

    _SCRATCH_REVIEW_FILTER = {
        "pending": "srr.id IS NULL",
        "marked": "srr.is_misrescue = 1",
        "all": None,
    }

    def list_scratch_rescued_tiles(
        self,
        start_date: str = None,
        end_date: str = None,
        order_by: str = "latest",
        limit: int = 24,
        offset: int = 0,
        filter_state: str = "pending",
    ) -> list:
        """列出被 scratch filter 救回的 tile（scratch_filtered=1），含誤救標記狀態。

        filter_state: pending（未審查）/ marked（已標記誤救）/ all。

        回傳 list of dict：
          tile_id, record_id, glass_id, machine_no, created_at, ai_judgment,
          image_name, heatmap_path (tile 層，fallback 至 image 層), tile x/y,
          scratch_score, score, is_misrescue (0/1), review_note
        """
        order_clause = self._SCRATCH_REVIEW_ORDER.get(order_by, self._SCRATCH_REVIEW_ORDER["latest"])
        limit = max(1, min(int(limit or 24), 100))
        offset = max(0, int(offset or 0))

        where = ["t.scratch_filtered = 1"]
        params = []
        if start_date:
            where.append("DATE(ir.created_at) >= DATE(?)")
            params.append(start_date)
        if end_date:
            where.append("DATE(ir.created_at) <= DATE(?)")
            params.append(end_date)
        filter_clause = self._SCRATCH_REVIEW_FILTER.get(filter_state, self._SCRATCH_REVIEW_FILTER["pending"])
        if filter_clause:
            where.append(filter_clause)
        where_sql = " AND ".join(where)

        conn = self._get_conn()
        try:
            rows = conn.execute(
                f"""
                SELECT
                    t.id                 AS tile_id,
                    t.tile_id            AS tile_seq,
                    t.x                  AS x,
                    t.y                  AS y,
                    t.score              AS score,
                    t.scratch_score      AS scratch_score,
                    t.heatmap_path       AS tile_heatmap,
                    img.id               AS image_id,
                    img.image_name       AS image_name,
                    img.heatmap_path     AS image_heatmap,
                    ir.id                AS record_id,
                    ir.glass_id          AS glass_id,
                    ir.machine_no        AS machine_no,
                    ir.created_at        AS created_at,
                    ir.ai_judgment       AS ai_judgment,
                    srr.id               AS review_id,
                    srr.is_misrescue     AS is_misrescue,
                    srr.note             AS review_note
                FROM tile_results t
                JOIN image_results img ON img.id = t.image_result_id
                JOIN inference_records ir ON ir.id = img.record_id
                LEFT JOIN scratch_rescue_review srr ON srr.tile_result_id = t.id
                WHERE {where_sql}
                ORDER BY {order_clause}
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset)
            ).fetchall()
            out = []
            for r in rows:
                heatmap = r["tile_heatmap"] or r["image_heatmap"] or ""
                out.append({
                    "tile_id": r["tile_id"],
                    "tile_seq": r["tile_seq"],
                    "x": r["x"],
                    "y": r["y"],
                    "score": float(r["score"] or 0.0),
                    "scratch_score": float(r["scratch_score"] or 0.0),
                    "heatmap_path": heatmap,
                    "image_id": r["image_id"],
                    "image_name": r["image_name"],
                    "record_id": r["record_id"],
                    "glass_id": r["glass_id"],
                    "machine_no": r["machine_no"],
                    "created_at": r["created_at"],
                    "ai_judgment": r["ai_judgment"],
                    "is_misrescue": int(r["is_misrescue"] or 0) if r["review_id"] else 0,
                    "review_note": r["review_note"] or "",
                    "reviewed": r["review_id"] is not None,
                })
            return out
        finally:
            conn.close()

    def list_scratch_misrescue_for_export(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> list:
        """列出所有已標記誤救的 tile，附帶原圖路徑與 tile 幾何資訊供匯出流程 re-crop。

        回傳欄位：tile_result_id, record_id, image_result_id, tile_seq,
        glass_id, image_name, image_path, x, y, width, height,
        scratch_score, score, created_at (inference), reviewed_at,
        review_note, ai_judgment
        """
        where = ["t.scratch_filtered = 1", "srr.is_misrescue = 1"]
        params = []
        if start_date:
            where.append("DATE(ir.created_at) >= DATE(?)")
            params.append(start_date)
        if end_date:
            where.append("DATE(ir.created_at) <= DATE(?)")
            params.append(end_date)
        where_sql = " AND ".join(where)
        conn = self._get_conn()
        try:
            rows = conn.execute(
                f"""
                SELECT
                    t.id             AS tile_result_id,
                    t.tile_id        AS tile_seq,
                    t.x              AS x,
                    t.y              AS y,
                    t.width          AS width,
                    t.height         AS height,
                    t.score          AS score,
                    t.scratch_score  AS scratch_score,
                    img.id           AS image_result_id,
                    img.image_path   AS image_path,
                    img.image_name   AS image_name,
                    ir.id            AS record_id,
                    ir.glass_id      AS glass_id,
                    ir.created_at    AS created_at,
                    ir.ai_judgment   AS ai_judgment,
                    srr.updated_at   AS reviewed_at,
                    srr.note         AS review_note
                FROM tile_results t
                JOIN image_results img ON img.id = t.image_result_id
                JOIN inference_records ir ON ir.id = img.record_id
                JOIN scratch_rescue_review srr ON srr.tile_result_id = t.id
                WHERE {where_sql}
                ORDER BY ir.created_at DESC, t.id DESC
                """,
                params
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def count_scratch_rescued_tiles(self, start_date: str = None, end_date: str = None) -> dict:
        """回傳 {total, marked}：被救回 tile 總數、已標記誤救 tile 數。"""
        where = ["t.scratch_filtered = 1"]
        params = []
        if start_date:
            where.append("DATE(ir.created_at) >= DATE(?)")
            params.append(start_date)
        if end_date:
            where.append("DATE(ir.created_at) <= DATE(?)")
            params.append(end_date)
        where_sql = " AND ".join(where)
        conn = self._get_conn()
        try:
            total_row = conn.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM tile_results t
                JOIN image_results img ON img.id = t.image_result_id
                JOIN inference_records ir ON ir.id = img.record_id
                WHERE {where_sql}
                """,
                params
            ).fetchone()
            marked_row = conn.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM tile_results t
                JOIN image_results img ON img.id = t.image_result_id
                JOIN inference_records ir ON ir.id = img.record_id
                JOIN scratch_rescue_review srr ON srr.tile_result_id = t.id
                WHERE {where_sql} AND srr.is_misrescue = 1
                """,
                params
            ).fetchone()
            return {
                "total": int(total_row["cnt"] or 0),
                "marked": int(marked_row["cnt"] or 0),
            }
        finally:
            conn.close()

    def mark_scratch_misrescue(self, tile_result_id: int, note: str = '') -> int:
        """標記一個 tile 為誤救。UPSERT，回傳 review id。"""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT id FROM tile_results WHERE id = ?",
                    (tile_result_id,)
                ).fetchone()
                if not row:
                    raise ValueError(f"Tile not found: {tile_result_id}")

                conn.execute(
                    """INSERT INTO scratch_rescue_review (tile_result_id, is_misrescue, note)
                       VALUES (?, 1, ?)
                       ON CONFLICT(tile_result_id)
                       DO UPDATE SET is_misrescue = 1,
                                     note = excluded.note,
                                     updated_at = datetime('now', 'localtime')""",
                    (tile_result_id, note)
                )
                conn.commit()
                review_id = conn.execute(
                    "SELECT id FROM scratch_rescue_review WHERE tile_result_id = ?",
                    (tile_result_id,)
                ).fetchone()["id"]
                return review_id
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def unmark_scratch_misrescue(self, tile_result_id: int) -> bool:
        """取消誤救標記。"""
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    "DELETE FROM scratch_rescue_review WHERE tile_result_id = ?",
                    (tile_result_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    VALID_MISS_CATEGORIES = {'dust_misfilter', 'threshold_high', 'ai_miss_within_spec', 'within_spec_misjudge', 'ric_misjudge', 'outside_aoi_area', 'data_error_actually_ok', 'other'}
    VALID_OVER_CATEGORIES = {'edge_false_positive', 'within_spec', 'overexposure', 'surface_scratch', 'surface_dirt', 'bubble', 'aoi_ai_false_positive', 'dust_mask_incomplete', 'other'}
    VALID_MES_REVIEW_CATEGORIES = {
        "over_detection": {
            "edge_false_positive", "within_spec", "overexposure",
            "surface_scratch", "surface_dirt", "bubble",
            "dust_mask_incomplete", "mes_not_registered", "actual_ng", "other",
        },
        "miss_detection": {
            "score_below_threshold", "low_contrast", "dust_misfilter",
            "not_visible_in_image", "outside_aoi_area",
            "image_issue", "mes_misjudge", "other",
        },
        "true_ng": {
            "confirmed_ng", "aoi_point_mismatch", "image_issue", "uncertain",
        },
    }

    def _save_review(self, table: str, valid_categories: set, client_record_id: int, category: str, note: str = '') -> int:
        """儲存或更新 Review (UPSERT by client_record_id)"""
        if category not in valid_categories:
            raise ValueError(f"Invalid category: {category}")

        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT id FROM client_accuracy_records WHERE id = ?",
                    (client_record_id,)
                ).fetchone()
                if not row:
                    raise ValueError(f"Record not found: {client_record_id}")

                conn.execute(
                    f"""INSERT INTO {table} (client_record_id, category, note)
                       VALUES (?, ?, ?)
                       ON CONFLICT(client_record_id)
                       DO UPDATE SET category = excluded.category,
                                     note = excluded.note,
                                     updated_at = datetime('now', 'localtime')""",
                    (client_record_id, category, note)
                )
                conn.commit()
                review_id = conn.execute(
                    f"SELECT id FROM {table} WHERE client_record_id = ?",
                    (client_record_id,)
                ).fetchone()["id"]
                return review_id
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def _delete_review(self, table: str, client_record_id: int) -> bool:
        """刪除 Review"""
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    f"DELETE FROM {table} WHERE client_record_id = ?",
                    (client_record_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def save_miss_review(self, client_record_id: int, category: str, note: str = '') -> int:
        return self._save_review('miss_review', self.VALID_MISS_CATEGORIES, client_record_id, category, note)

    def delete_miss_review(self, client_record_id: int) -> bool:
        return self._delete_review('miss_review', client_record_id)

    def save_over_review(self, client_record_id: int, category: str, note: str = '') -> int:
        return self._save_review('over_review', self.VALID_OVER_CATEGORIES, client_record_id, category, note)

    def delete_over_review(self, client_record_id: int) -> bool:
        return self._delete_review('over_review', client_record_id)

    def save_within_spec_review_log(
        self,
        client_record_id: Optional[int],
        inference_record_id: int,
        suggestion: Optional[Dict[str, Any]],
        detail: Dict[str, Any],
        processing_seconds: float,
        error_message: str = "",
        source: str = "review",
    ) -> Dict[str, Any]:
        """保存一筆規格內建議計算紀錄，保留歷史供反查。"""
        suggested = bool(suggestion and suggestion.get("suggested"))
        category = str((suggestion or {}).get("category") or "")
        reason = str((suggestion or {}).get("reason") or "")
        detail_json = json.dumps(detail or {}, ensure_ascii=False)
        log_source = "inference" if source == "inference" else "review"

        with self._lock:
            conn = self._get_conn()
            try:
                if client_record_id is not None:
                    row = conn.execute(
                        "SELECT id FROM client_accuracy_records WHERE id = ?",
                        (client_record_id,)
                    ).fetchone()
                    if not row:
                        raise ValueError(f"Record not found: {client_record_id}")

                row = conn.execute(
                    "SELECT id FROM inference_records WHERE id = ?",
                    (inference_record_id,)
                ).fetchone()
                if not row:
                    raise ValueError(f"Inference record not found: {inference_record_id}")

                cursor = conn.execute(
                    """INSERT INTO within_spec_review_log
                       (client_record_id, inference_record_id, suggested, category, reason,
                        detail_json, error_message, source, processing_seconds)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        client_record_id,
                        inference_record_id,
                        1 if suggested else 0,
                        category,
                        reason,
                        detail_json,
                        error_message or "",
                        log_source,
                        float(processing_seconds or 0.0),
                    )
                )
                conn.commit()
                return self.get_within_spec_review_log(int(cursor.lastrowid)) or {}
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def _format_within_spec_review_log(self, row) -> Dict[str, Any]:
        data = dict(row)
        try:
            detail = json.loads(data.get("detail_json") or "{}")
        except Exception:
            detail = {}
        suggestion = None
        if data.get("suggested"):
            suggestion = {
                "suggested": True,
                "category": data.get("category") or "within_spec",
                "reason": data.get("reason") or "",
                "matches": detail.get("matches") or [],
            }
        return {
            "id": data["id"],
            "client_record_id": data["client_record_id"],
            "inference_record_id": data["inference_record_id"],
            "suggested": bool(data.get("suggested")),
            "category": data.get("category") or "",
            "reason": data.get("reason") or "",
            "suggestion": suggestion,
            "detail": detail,
            "error_message": data.get("error_message") or "",
            "source": data.get("source") or "review",
            "processing_seconds": data.get("processing_seconds") or 0.0,
            "created_at": data.get("created_at"),
        }

    def get_within_spec_review_log(self, log_id: int) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM within_spec_review_log WHERE id = ?",
                (log_id,)
            ).fetchone()
            return self._format_within_spec_review_log(row) if row else None
        finally:
            conn.close()

    def get_latest_within_spec_review_log(self, client_record_id: int) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT * FROM within_spec_review_log
                   WHERE client_record_id = ?
                   ORDER BY id DESC LIMIT 1""",
                (client_record_id,)
            ).fetchone()
            return self._format_within_spec_review_log(row) if row else None
        finally:
            conn.close()

    def get_within_spec_review_logs(self, client_record_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit or 10), 50))
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM within_spec_review_log
                   WHERE client_record_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (client_record_id, limit)
            ).fetchall()
            return [self._format_within_spec_review_log(r) for r in rows]
        finally:
            conn.close()

    def list_within_spec_review_log_report(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        keyword: str = "",
        suggested: Optional[bool] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """列出規格內建議計算紀錄，供報表頁快速查閱。"""
        if start_date and not _DATE_RE.match(start_date):
            raise ValueError(f"Invalid start_date format: {start_date}")
        if end_date and not _DATE_RE.match(end_date):
            raise ValueError(f"Invalid end_date format: {end_date}")

        def _next_date_str(date_str: str) -> str:
            from datetime import timedelta
            return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

        limit = max(1, min(int(limit or 200), 500))
        where = []
        params: List[Any] = []
        if start_date:
            where.append("wsl.created_at >= ?")
            params.append(start_date)
        if end_date:
            where.append("wsl.created_at < ?")
            params.append(_next_date_str(end_date))
        if suggested is not None:
            where.append("wsl.suggested = ?")
            params.append(1 if suggested else 0)

        keyword = str(keyword or "").strip()
        if keyword:
            like = f"%{keyword}%"
            where.append(
                "(COALESCE(c.pnl_id, ir.glass_id, '') LIKE ? OR c.mach_id LIKE ? OR ir.model_id LIKE ? "
                "OR ir.machine_no LIKE ? OR wsl.reason LIKE ?)"
            )
            params.extend([like, like, like, like, like])

        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                f"""SELECT wsl.*,
                           c.time_stamp AS client_time_stamp,
                           COALESCE(c.pnl_id, ir.glass_id, '') AS pnl_id,
                           c.mach_id AS mach_id,
                           c.result_eqp AS result_eqp,
                           c.result_ai AS result_ai,
                           c.result_ric AS result_ric,
                           ir.model_id AS model_id,
                           ir.machine_no AS machine_no,
                           ir.ai_judgment AS inference_ai_judgment,
                           ir.machine_judgment AS inference_machine_judgment
                    FROM within_spec_review_log wsl
                    LEFT JOIN client_accuracy_records c ON c.id = wsl.client_record_id
                    LEFT JOIN inference_records ir ON ir.id = wsl.inference_record_id
                    {where_sql}
                    ORDER BY wsl.id DESC
                    LIMIT ?""",
                (*params, limit),
            ).fetchall()

            report_rows = []
            for row in rows:
                data = dict(row)
                try:
                    detail = json.loads(data.get("detail_json") or "{}")
                except Exception:
                    detail = {}
                summary = detail.get("panel_summary") or {}
                rule_selection = detail.get("rule_selection") or {}
                first_match = (detail.get("matches") or [{}])[0] or {}
                report_rows.append({
                    "id": data["id"],
                    "client_record_id": data["client_record_id"],
                    "inference_record_id": data["inference_record_id"],
                    "created_at": data.get("created_at"),
                    "client_time_stamp": data.get("client_time_stamp") or "",
                    "pnl_id": data.get("pnl_id") or "",
                    "mach_id": data.get("mach_id") or "",
                    "model_id": data.get("model_id") or "",
                    "machine_no": data.get("machine_no") or "",
                    "result_eqp": data.get("result_eqp") or "",
                    "result_ai": data.get("result_ai") or "",
                    "result_ric": data.get("result_ric") or "",
                    "suggested": bool(data.get("suggested")),
                    "category": data.get("category") or "",
                    "reason": data.get("reason") or "",
                    "error_message": data.get("error_message") or "",
                    "processing_seconds": data.get("processing_seconds") or 0.0,
                    "matched_machine_key": rule_selection.get("matched_machine_key") or "",
                    "fallback_used": bool(rule_selection.get("fallback_used")),
                    "total_dot_count": int(summary.get("total_dot_count") or 0),
                    "target_tile_count": int(summary.get("target_tile_count") or 0),
                    "evaluated_tile_count": int(summary.get("evaluated_tile_count") or 0),
                    "screen": first_match.get("screen") or "",
                    "dot_label": first_match.get("dot_label") or "",
                })
            return report_rows
        finally:
            conn.close()

    def get_client_accuracy_count(self) -> int:
        """取得 client accuracy records 總數"""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM client_accuracy_records").fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    def clear_client_accuracy_records(self) -> int:
        """清除所有 client accuracy records"""
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute("DELETE FROM client_accuracy_records")
                conn.commit()
                return cursor.rowcount
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def get_ric_batches(self) -> List[Dict]:
        """列出所有 RIC 匯入批次"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM ric_import_batches
                   ORDER BY created_at DESC"""
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def delete_ric_batch(self, batch_id: int) -> int:
        """
        刪除指定的 RIC 匯入批次及其所有記錄

        Args:
            batch_id: 要刪除的批次 ID

        Returns:
            被刪除的 ric_records 筆數
        """
        with self._lock:
            conn = self._get_conn()
            try:
                # 先計算要刪除的記錄數
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM ric_records WHERE batch_id = ?",
                    (batch_id,)
                ).fetchone()
                deleted_count = row["cnt"] if row else 0

                # 刪除該批次的所有記錄
                conn.execute(
                    "DELETE FROM ric_records WHERE batch_id = ?",
                    (batch_id,)
                )
                # 刪除批次本身
                conn.execute(
                    "DELETE FROM ric_import_batches WHERE id = ?",
                    (batch_id,)
                )
                conn.commit()
                return deleted_count
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def cleanup_old_records(
        self,
        ok_retain_days: int = 14,
        ng_retain_days: int = 90,
        tile_retain_days: int = 7,
        vacuum: bool = True,
        heatmap_retain_days: int = 0,
        heatmap_base_dir: Optional[str] = None,
    ) -> dict:
        from datetime import datetime, timedelta
        import shutil
        now = datetime.now()
        ok_cutoff   = (now - timedelta(days=ok_retain_days)).strftime('%Y-%m-%d')
        ng_cutoff   = (now - timedelta(days=ng_retain_days)).strftime('%Y-%m-%d')
        tile_cutoff = (now - timedelta(days=tile_retain_days)).strftime('%Y-%m-%d')

        stats = {
            "tile_results_deleted": 0,
            "inference_records_deleted": 0,
            "heatmap_dirs_deleted": 0,
            "heatmap_dirs_failed": 0,
            "heatmap_date_dirs_deleted": 0,
            "heatmap_date_dirs_failed": 0,
            "scratch_rescue_review_deleted": 0,
            "within_spec_dirs_deleted": 0,
            "within_spec_dirs_failed": 0,
        }

        # Step 0: 先找出並刪除過期 heatmap；只有成功後才清 DB 路徑。
        heatmap_records_by_dir = {}
        heatmap_date_dirs_to_delete = []
        within_spec_dirs_to_delete = []
        if heatmap_retain_days > 0:
            hm_cutoff = (now - timedelta(days=heatmap_retain_days)).strftime('%Y-%m-%d')
            with self._lock:
                conn = self._get_conn()
                try:
                    rows = conn.execute("""
                        SELECT id, heatmap_dir FROM inference_records
                        WHERE heatmap_dir != '' AND created_at < ?
                    """, (hm_cutoff,)).fetchall()
                    for row in rows:
                        if row[1]:
                            heatmap_records_by_dir.setdefault(row[1], []).append(row[0])
                finally:
                    conn.close()

        if heatmap_retain_days > 0 and heatmap_base_dir:
            heatmap_root = Path(heatmap_base_dir)
            heatmap_cutoff_date = (now - timedelta(days=heatmap_retain_days)).date()
            try:
                if heatmap_root.is_dir():
                    for child in heatmap_root.iterdir():
                        if child.is_symlink() or not child.is_dir():
                            continue
                        if not re.fullmatch(r"\d{8}", child.name):
                            continue
                        try:
                            child_date = datetime.strptime(child.name, "%Y%m%d").date()
                        except ValueError:
                            continue
                        if child_date < heatmap_cutoff_date:
                            heatmap_date_dirs_to_delete.append(child)
            except OSError as exc:
                stats["heatmap_date_dirs_failed"] += 1
                logger.warning(
                    "[Cleanup] Failed to scan heatmap date directories under %s: %s",
                    heatmap_root,
                    exc,
                )

            within_spec_root = heatmap_root / "within_spec_inference"
            within_spec_cutoff_ts = (
                now - timedelta(days=heatmap_retain_days)
            ).timestamp()
            try:
                if within_spec_root.is_dir():
                    for child in within_spec_root.iterdir():
                        if child.is_symlink() or not child.is_dir():
                            continue
                        try:
                            child_mtime = child.stat().st_mtime
                            if child_mtime < within_spec_cutoff_ts:
                                within_spec_dirs_to_delete.append((child, child_mtime))
                        except OSError as exc:
                            stats["within_spec_dirs_failed"] += 1
                            logger.warning(
                                "[Cleanup] Failed to inspect within-spec directory %s: %s",
                                child,
                                exc,
                            )
            except OSError as exc:
                stats["within_spec_dirs_failed"] += 1
                logger.warning(
                    "[Cleanup] Failed to scan within-spec directories under %s: %s",
                    within_spec_root,
                    exc,
                )

        heatmap_records_to_clear = []
        heatmap_retry_record_ids = set()
        for directory, record_ids in heatmap_records_by_dir.items():
            path = Path(directory)
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    stats["heatmap_dirs_deleted"] += 1
                elif path.exists():
                    raise OSError("heatmap path is not a directory")
                heatmap_records_to_clear.extend(
                    (record_id, directory) for record_id in record_ids
                )
            except Exception as exc:
                heatmap_retry_record_ids.update(record_ids)
                stats["heatmap_dirs_failed"] += 1
                logger.warning(
                    "[Cleanup] Failed to delete heatmap directory %s: %s",
                    path,
                    exc,
                )

        # 舊版可能已刪除 DB record，因此直接掃描嚴格 YYYYMMDD 日期根目錄。
        for path in heatmap_date_dirs_to_delete:
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                    stats["heatmap_date_dirs_deleted"] += 1
            except Exception as exc:
                stats["heatmap_date_dirs_failed"] += 1
                logger.warning(
                    "[Cleanup] Failed to delete heatmap date directory %s: %s",
                    path,
                    exc,
                )

        for path, original_mtime in within_spec_dirs_to_delete:
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                    stats["within_spec_dirs_deleted"] += 1
            except Exception as exc:
                stats["within_spec_dirs_failed"] += 1
                logger.warning(
                    "[Cleanup] Failed to delete within-spec directory %s: %s",
                    path,
                    exc,
                )
                if path.exists() and not path.is_symlink():
                    try:
                        os.utime(path, (original_mtime, original_mtime))
                    except OSError as restore_exc:
                        logger.warning(
                            "[Cleanup] Failed to restore within-spec directory mtime "
                            "for %s: %s",
                            path,
                            restore_exc,
                        )

        with self._lock:
            conn = self._get_conn()
            try:
                if heatmap_records_to_clear:
                    conn.executemany(
                        """
                        UPDATE inference_records
                        SET heatmap_dir = ''
                        WHERE id = ? AND heatmap_dir = ?
                        """,
                        heatmap_records_to_clear,
                    )

                if heatmap_retry_record_ids:
                    conn.execute(
                        "CREATE TEMP TABLE cleanup_heatmap_retry_ids "
                        "(id INTEGER PRIMARY KEY)"
                    )
                    conn.executemany(
                        "INSERT INTO cleanup_heatmap_retry_ids (id) VALUES (?)",
                        ((record_id,) for record_id in heatmap_retry_record_ids),
                    )

                # scratch_rescue_review belongs to tile_results and must be
                # removed first because its foreign key is not cascading.
                cur = conn.execute("""
                    DELETE FROM scratch_rescue_review
                    WHERE tile_result_id IN (
                        SELECT t.id FROM tile_results t
                        JOIN image_results im ON t.image_result_id = im.id
                        JOIN inference_records ir ON im.record_id = ir.id
                        WHERE ir.created_at < ?
                    )
                """, (tile_cutoff,))
                stats["scratch_rescue_review_deleted"] = cur.rowcount

                # Step 1: 清除超過 tile_retain_days 的 tile_results
                cur = conn.execute("""
                    DELETE FROM tile_results
                    WHERE image_result_id IN (
                        SELECT im.id FROM image_results im
                        JOIN inference_records ir ON im.record_id = ir.id
                        WHERE ir.created_at < ?
                    )
                """, (tile_cutoff,))
                stats["tile_results_deleted"] = cur.rowcount

                # Step 2: 清除過期 inference_records (cascade 自動刪子表)
                cur = conn.execute("""
                    DELETE FROM inference_records
                    WHERE (
                        ((ai_judgment = 'OK' OR ai_judgment = 'OK-i') AND created_at < ?)
                        OR (ai_judgment != 'OK' AND created_at < ?)
                    )
                """ + (
                    " AND id NOT IN (SELECT id FROM cleanup_heatmap_retry_ids)"
                    if heatmap_retry_record_ids else ""
                ), (ok_cutoff, ng_cutoff))
                stats["inference_records_deleted"] = cur.rowcount

                conn.commit()
            except Exception as e:
                conn.rollback()
                raise
            finally:
                conn.close()

        # Step 3: VACUUM (在鎖外，不阻塞其他操作)
        if vacuum and (stats["tile_results_deleted"] > 0 or stats["inference_records_deleted"] > 0):
            conn = self._get_conn()
            try:
                conn.execute("VACUUM")
            finally:
                conn.close()

        return stats

    def get_ric_comparison(self, batch_id: int = None) -> List[Dict]:
        """
        取得 RIC 比對結果

        邏輯:
        1. 取所有 inference_records
        2. 用 glass_id = pnl_id JOIN ric_records
        3. 找不到 RIC 對應 → RIC 當 OK
        4. MACH_ID 前6碼比對 machine_no
        """
        conn = self._get_conn()
        try:
            # 取所有的記錄，不再限制 machine_judgment != 'OK'
            all_rows = conn.execute(
                """SELECT id, glass_id, machine_no, machine_judgment, ai_judgment, created_at
                   FROM inference_records
                   ORDER BY created_at DESC"""
            ).fetchall()

            # 取 RIC 資料 (指定批次或全部)
            if batch_id:
                ric_rows = conn.execute(
                    "SELECT pnl_id, mach_id, ric_judgment, datastr, timestamp FROM ric_records WHERE batch_id = ?",
                    (batch_id,)
                ).fetchall()
            else:
                ric_rows = conn.execute(
                    "SELECT pnl_id, mach_id, ric_judgment, datastr, timestamp FROM ric_records"
                ).fetchall()

            # 建立 RIC lookup: pnl_id → ric record
            ric_lookup = {}
            for r in ric_rows:
                ric_lookup[r["pnl_id"]] = dict(r)

            results = []
            for row in all_rows:
                rec = dict(row)
                pnl_id = rec["glass_id"]
                machine_no = rec["machine_no"]

                # 查找 RIC 對應
                ric_rec = ric_lookup.get(pnl_id)

                if ric_rec:
                    # MACH_ID 前6碼比對
                    ric_mach = ric_rec.get("mach_id", "")[:6]
                    sys_mach = machine_no[:6] if machine_no else ""
                    if ric_mach and sys_mach and ric_mach != sys_mach:
                        # 機台不匹配，跳過此筆 (或可依需求保留)
                        pass
                    rec["ric_judgment"] = ric_rec["ric_judgment"]
                    rec["ric_datastr"] = ric_rec.get("datastr", "")
                    rec["ric_timestamp"] = ric_rec.get("timestamp", "")
                    rec["ric_found"] = True
                else:
                    # 找不到 RIC → 當 OK
                    rec["ric_judgment"] = "OK"
                    rec["ric_datastr"] = ""
                    rec["ric_timestamp"] = ""
                    rec["ric_found"] = False

                results.append(rec)

            return results
        finally:
            conn.close()

    def get_ric_accuracy_stats(self, batch_id: int = None) -> Dict:
        """
        計算 AOI 及 AI 的準確率、過檢率、漏檢率統計

        Returns:
            {
                "total": 總比對數,
                "aoi_accuracy": AOI 準確率 (AOI 與 RIC 一致比率),
                "ai_accuracy": AI 準確率 (AI 與 RIC 一致比率),
                "aoi_over_rate": AOI 過檢率 (AOI NG, RIC OK),
                "aoi_miss_rate": AOI 漏檢率 (AOI OK, RIC NG),
                "ai_over_rate": AI 過檢率 (AI NG, RIC OK),
                "ai_miss_rate": AI 漏檢率 (AI OK, RIC NG),
                "ric_ng_total": RIC NG 總數,
                ...
            }
        """
        comparisons = self.get_ric_comparison(batch_id)

        empty_result = {
            "total": 0,
            "aoi_accuracy": 0, "ai_accuracy": 0,
            "aoi_ng_correct": 0, "ai_correct": 0,
            "aoi_over": 0, "aoi_over_rate": 0,
            "aoi_miss": 0, "aoi_miss_rate": 0,
            "ai_over": 0, "ai_over_rate": 0, "ai_ng_count": 0,
            "ai_miss": 0, "ai_miss_rate": 0,
            "by_day": [], "by_machine": [], "details": [],
        }

        if not comparisons:
            return empty_result

        total = len(comparisons)
        aoi_correct_count = 0
        ai_correct_count = 0
        ai_ng_count = 0     
        ai_over = 0         
        ai_miss = 0         

        day_stats = {}   
        mach_stats = {}  

        for rec in comparisons:
            ric_j = rec["ric_judgment"]
            ai_j = "OK" if rec["ai_judgment"] in ("OK", "OK-i") else "NG"
            aoi_j = "OK" if rec["machine_judgment"] == "OK" else "NG"

            # AOI 準確率: AOI 判定與 RIC 一致
            if aoi_j == ric_j:
                aoi_correct_count += 1
            
            # AI 準確率: AI 判定與 RIC 一致
            if ai_j == ric_j:
                ai_correct_count += 1

            # AI 過檢/漏檢
            if ai_j == "NG":
                ai_ng_count += 1
                if ric_j == "OK":
                    ai_over += 1
            else:  # AI OK
                if ric_j == "NG":
                    ai_miss += 1

            # 按日統計
            date_str = rec["created_at"][:10] if rec.get("created_at") else "unknown"
            if date_str not in day_stats:
                day_stats[date_str] = {"total": 0, "aoi_correct": 0, "ai_correct": 0}
            day_stats[date_str]["total"] += 1
            if aoi_j == ric_j:
                day_stats[date_str]["aoi_correct"] += 1
            if ai_j == ric_j:
                day_stats[date_str]["ai_correct"] += 1

            # 按機台統計
            machine = rec.get("machine_no", "unknown")
            if machine not in mach_stats:
                mach_stats[machine] = {"total": 0, "aoi_correct": 0, "ai_correct": 0}
            mach_stats[machine]["total"] += 1
            if aoi_j == ric_j:
                mach_stats[machine]["aoi_correct"] += 1
            if ai_j == ric_j:
                mach_stats[machine]["ai_correct"] += 1

        aoi_accuracy = (aoi_correct_count / total * 100) if total > 0 else 0
        ai_accuracy = (ai_correct_count / total * 100) if total > 0 else 0

        # AOI 過檢與漏檢
        aoi_ng_count = 0
        aoi_over = 0
        aoi_miss = 0
        ric_ng_total = 0

        for rec in comparisons:
            ric_j = rec["ric_judgment"]
            aoi_j = "OK" if rec["machine_judgment"] == "OK" else "NG"
            
            if aoi_j == "NG":
                aoi_ng_count += 1
                if ric_j == "OK":
                    aoi_over += 1
            else:
                if ric_j == "NG":
                    aoi_miss += 1
                    
            if ric_j == "NG":
                ric_ng_total += 1
                
        aoi_over_rate = round((aoi_over / total * 100), 2) if total > 0 else 0
        aoi_miss_rate = round((aoi_miss / total * 100), 2) if total > 0 else 0

        # AI 過檢率 / 漏檢率
        ai_over_rate = round(ai_over / total * 100, 2) if total > 0 else 0
        ai_miss_rate = round(ai_miss / total * 100, 2) if total > 0 else 0

        by_day = []
        for date_str in sorted(day_stats.keys()):
            s = day_stats[date_str]
            by_day.append({
                "date": date_str,
                "total": s["total"],
                "aoi_acc": round(s["aoi_correct"] / s["total"] * 100, 2) if s["total"] > 0 else 0,
                "ai_acc": round(s["ai_correct"] / s["total"] * 100, 2) if s["total"] > 0 else 0,
            })

        by_machine = []
        for machine in sorted(mach_stats.keys()):
            s = mach_stats[machine]
            by_machine.append({
                "machine": machine,
                "total": s["total"],
                "aoi_acc": round(s["aoi_correct"] / s["total"] * 100, 2) if s["total"] > 0 else 0,
                "ai_acc": round(s["ai_correct"] / s["total"] * 100, 2) if s["total"] > 0 else 0,
            })

        return {
            "total": total,
            "aoi_accuracy": round(aoi_accuracy, 2),
            "ai_accuracy": round(ai_accuracy, 2),
            "aoi_ng_correct": aoi_correct_count,
            "ai_correct": ai_correct_count,
            # 過檢/漏檢
            "aoi_over": aoi_over,
            "aoi_over_rate": aoi_over_rate,
            "aoi_miss": aoi_miss,
            "aoi_miss_rate": aoi_miss_rate,
            "aoi_ng_count": aoi_ng_count,
            "ai_ng_count": ai_ng_count,
            "ai_over": ai_over,
            "ai_over_rate": ai_over_rate,
            "ai_miss": ai_miss,
            "ai_miss_rate": ai_miss_rate,
            "ric_ng_total": ric_ng_total,
            "by_day": by_day,
            "by_machine": by_machine,
            "details": comparisons,
        }

    def get_inference_stats(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        """
        取得推論紀錄統計資料（供 AI 推論紀錄 Tab 使用）

        Args:
            start_date: 起始日期 YYYY-MM-DD（含）
            end_date: 結束日期 YYYY-MM-DD（含）
        """
        if start_date and not _DATE_RE.match(start_date):
            return {"success": False, "error": f"Invalid start_date format: {start_date}"}
        if end_date and not _DATE_RE.match(end_date):
            return {"success": False, "error": f"Invalid end_date format: {end_date}"}

        conn = self._get_conn()
        try:
            # 建立日期篩選條件
            where_clauses = []
            params = []
            if start_date:
                where_clauses.append("datetime(request_time) >= datetime(?)")
                params.append(_factory_day_start_ts(start_date))
            if end_date:
                where_clauses.append("datetime(request_time) < datetime(?)")
                params.append(_factory_day_end_ts(end_date))
            where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

            _aoi_ng = self._AOI_NG_COND
            _ai_ng = self._AI_NG_COND
            _err_all = self._ERR_COND
            _hy = "ai_judgment LIKE 'ERR:HY%'"
            _err = f"({_err_all}) AND NOT ({_hy})"

            # ── 1. Summary + Cross Matrix (single SQL aggregate) ──
            summary_row = conn.execute(
                f"""SELECT COUNT(*) as total,
                           SUM(CASE WHEN {_aoi_ng} THEN 1 ELSE 0 END) as aoi_ng,
                           SUM(CASE WHEN {_ai_ng} THEN 1 ELSE 0 END) as ai_ng,
                           SUM(CASE WHEN ({_aoi_ng}) AND (ai_judgment = 'OK' OR ai_judgment = 'OK-i') THEN 1 ELSE 0 END) as ai_revival,
                           SUM(CASE WHEN {_err} THEN 1 ELSE 0 END) as err_count,
                           SUM(CASE WHEN {_hy} THEN 1 ELSE 0 END) as hy_count,
                           SUM(CASE WHEN NOT ({_aoi_ng}) AND NOT ({_ai_ng}) AND NOT ({_err_all}) THEN 1 ELSE 0 END) as ok_ok,
                           SUM(CASE WHEN ({_aoi_ng}) AND NOT ({_ai_ng}) AND NOT ({_err_all}) THEN 1 ELSE 0 END) as ng_ok,
                           SUM(CASE WHEN NOT ({_aoi_ng}) AND ({_ai_ng}) AND NOT ({_err_all}) THEN 1 ELSE 0 END) as ok_ng,
                           SUM(CASE WHEN ({_aoi_ng}) AND ({_ai_ng}) AND NOT ({_err_all}) THEN 1 ELSE 0 END) as ng_ng
                    FROM inference_records{where_sql}""",
                params
            ).fetchone()
            s = dict(summary_row)
            total = s["total"] or 0

            # ── 2. 每日趨勢 ──
            daily_rows = conn.execute(
                f"""SELECT DATE(datetime(request_time), '-7 hours', '-30 minutes') as date,
                           COUNT(*) as total,
                           SUM(CASE WHEN {_aoi_ng} THEN 1 ELSE 0 END) as aoi_ng,
                           SUM(CASE WHEN {_ai_ng} THEN 1 ELSE 0 END) as ai_ng,
                           SUM(CASE WHEN {_err} THEN 1 ELSE 0 END) as err,
                           SUM(CASE WHEN {_hy} THEN 1 ELSE 0 END) as hy
                    FROM inference_records{where_sql}
                    GROUP BY DATE(datetime(request_time), '-7 hours', '-30 minutes')
                    ORDER BY date""",
                params
            ).fetchall()
            daily_trend = [dict(r) for r in daily_rows]

            client_where_clauses = ["COALESCE(NULLIF(TRIM(c.result_eqp), ''), 'OK') != 'OK'"]
            client_params = []
            if start_date:
                client_where_clauses.append("datetime(c.time_stamp) >= datetime(?)")
                client_params.append(_factory_day_start_ts(start_date))
            if end_date:
                client_where_clauses.append("datetime(c.time_stamp) < datetime(?)")
                client_params.append(_factory_day_end_ts(end_date))
            client_where_sql = " WHERE " + " AND ".join(client_where_clauses)
            counted_ai_miss_categories = {"threshold_high", "dust_misfilter"}
            client_rows = conn.execute(
                f"""SELECT DATE(datetime(c.time_stamp), '-7 hours', '-30 minutes') as date,
                           c.result_ai,
                           c.datastr,
                           mr.category as review_category
                    FROM client_accuracy_records c
                    LEFT JOIN miss_review mr ON mr.client_record_id = c.id
                    {client_where_sql}""",
                client_params
            ).fetchall()
            ai_miss_by_day = {}
            for row in client_rows:
                day = row["date"]
                if not day:
                    continue
                stats = ai_miss_by_day.setdefault(day, {"total": 0, "ai_miss": 0})
                stats["total"] += 1
                ai = (row["result_ai"] or "OK").strip()
                if ai == "OK-i":
                    ai = "OK"
                if (
                    ai == "OK"
                    and self.parse_ric_judgment(row["datastr"] or "") == "NG"
                    and row["review_category"] in counted_ai_miss_categories
                ):
                    stats["ai_miss"] += 1
            for row in daily_trend:
                miss_stats = ai_miss_by_day.get(row["date"], {"total": 0, "ai_miss": 0})
                row["ai_miss"] = miss_stats["ai_miss"]
                row["ai_miss_total"] = miss_stats["total"]
                row["ai_miss_rate"] = (
                    round(miss_stats["ai_miss"] * 100.0 / miss_stats["total"], 2)
                    if miss_stats["total"] > 0 else None
                )

            # ── 3. 機台統計 ──
            machine_rows = conn.execute(
                f"""SELECT machine_no as machine,
                           COUNT(*) as total,
                           ROUND(SUM(CASE WHEN {_aoi_ng} THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as aoi_ng_rate,
                           ROUND(SUM(CASE WHEN {_ai_ng} THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as ai_ng_rate
                    FROM inference_records{where_sql}
                    GROUP BY machine_no
                    ORDER BY total DESC""",
                params
            ).fetchall()
            by_machine = [dict(r) for r in machine_rows]

            # ── 4. 產品型號統計 ──
            model_rows = conn.execute(
                f"""SELECT model_id as model, COUNT(*) as total
                    FROM inference_records{where_sql}
                    GROUP BY model_id
                    ORDER BY total DESC""",
                params
            ).fetchall()
            by_model = [dict(r) for r in model_rows]

            # ── 5. ERR 類型 (SQL GROUP BY) ──
            err_where = f" WHERE {_err}" if not where_clauses else where_sql + f" AND {_err}"
            err_rows = conn.execute(
                f"""SELECT ai_judgment, error_message
                    FROM inference_records{err_where}""",
                params
            ).fetchall()
            err_type_counts = {}
            for row in err_rows:
                error_type = _normalize_inference_error_type(row["ai_judgment"], row["error_message"])
                err_type_counts[error_type] = err_type_counts.get(error_type, 0) + 1
            err_types = [
                {"type": error_type, "count": count}
                for error_type, count in sorted(
                    err_type_counts.items(), key=lambda item: (-item[1], item[0])
                )
            ]

            return {
                "success": True,
                "summary": {
                    "total": total,
                    "aoi_ng": s["aoi_ng"] or 0,
                    "ai_ng": s["ai_ng"] or 0,
                    "ai_revival": s["ai_revival"] or 0,
                    "err_count": s["err_count"] or 0,
                    "hy_count": s["hy_count"] or 0,
                },
                "daily_trend": daily_trend,
                "by_machine": by_machine,
                "by_model": by_model,
                "err_types": err_types,
                "cross_matrix": {
                    "ok_ok": s["ok_ok"] or 0,
                    "ng_ok": s["ng_ok"] or 0,
                    "ok_ng": s["ok_ng"] or 0,
                    "ng_ng": s["ng_ng"] or 0,
                },
            }
        finally:
            conn.close()

    def get_mes_comparison_records(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        ignore_aoi_ok: bool = False,
        panel_id: Optional[str] = None,
    ) -> list:
        """取得 Report 數據比對使用的 AI 推論紀錄。"""
        if start_date and not _DATE_RE.match(start_date):
            raise ValueError(f"Invalid start_date format: {start_date}")
        if end_date and not _DATE_RE.match(end_date):
            raise ValueError(f"Invalid end_date format: {end_date}")

        where_clauses = []
        params = []
        if start_date:
            where_clauses.append("datetime(request_time) >= datetime(?)")
            params.append(_factory_day_start_ts(start_date))
        if end_date:
            where_clauses.append("datetime(request_time) < datetime(?)")
            params.append(_factory_day_end_ts(end_date))
        if ignore_aoi_ok:
            where_clauses.append("UPPER(TRIM(COALESCE(machine_judgment, ''))) != 'OK'")
        if panel_id and panel_id.strip():
            where_clauses.append("UPPER(TRIM(COALESCE(glass_id, ''))) LIKE UPPER(?)")
            params.append(f"%{panel_id.strip()}%")
        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        conn = self._get_conn()
        try:
            rows = conn.execute(
                f"""SELECT id, glass_id, model_id, machine_no, ai_judgment,
                           image_dir, request_time
                    FROM inference_records{where_sql}
                    ORDER BY datetime(request_time) DESC, id DESC""",
                params,
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_mes_comparison_record(self, record_id: int) -> Optional[dict]:
        """依本地推論紀錄 ID 取得 MES 完整資料查詢條件。"""
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT id, glass_id, model_id, machine_no, ai_judgment,
                          image_dir, request_time
                   FROM inference_records
                   WHERE id = ?""",
                (record_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_mes_review_aoi_candidates(
        self,
        inference_record_id: int,
        tile_result_id: Optional[int] = None,
    ) -> List[Dict]:
        """取得一筆 Report 比對紀錄下，由 AOI 座標產生的候選 tile。"""
        conditions = ["ir.id = ?", "t.is_aoi_coord = 1"]
        params: List[Any] = [int(inference_record_id)]
        if tile_result_id is not None:
            conditions.append("t.id = ?")
            params.append(int(tile_result_id))

        conn = self._get_conn()
        try:
            rows = conn.execute(
                f"""SELECT
                        ir.id AS inference_record_id,
                        ir.glass_id, ir.model_id, ir.machine_no,
                        ir.machine_judgment, ir.ai_judgment,
                        ir.request_time, ir.aoi_machine_coords,
                        im.id AS image_result_id,
                        im.image_path, im.image_name,
                        im.image_width, im.image_height,
                        im.is_ng AS image_is_ng,
                        im.is_bomb AS image_is_bomb,
                        t.id AS tile_result_id, t.tile_id,
                        t.x AS tile_x, t.y AS tile_y,
                        t.width AS tile_w, t.height AS tile_h,
                        t.score AS ai_score, t.is_anomaly,
                        t.is_dust, t.is_bomb, t.is_exclude_zone,
                        t.peak_x, t.peak_y, t.zone,
                        t.aoi_defect_code,
                        t.aoi_product_x, t.aoi_product_y,
                        t.aoi_image_x, t.aoi_image_y,
                        EXISTS (
                            SELECT 1
                              FROM ng_validation_samples s
                              JOIN mes_comparison_review mr ON mr.id = s.review_id
                             WHERE mr.inference_record_id = ir.id
                               AND s.tile_result_id = t.id
                               AND s.status = 'confirmed'
                        ) AS is_selected
                    FROM tile_results t
                    JOIN image_results im ON im.id = t.image_result_id
                    JOIN inference_records ir ON ir.id = im.record_id
                    WHERE {" AND ".join(conditions)}
                    ORDER BY
                        t.aoi_product_y, t.aoi_product_x,
                        im.id, t.score DESC, t.id""",
                params,
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_mes_review_aoi_candidate(self, tile_result_id: int) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            owner = conn.execute(
                """SELECT im.record_id
                     FROM tile_results t
                     JOIN image_results im ON im.id = t.image_result_id
                    WHERE t.id = ? AND t.is_aoi_coord = 1""",
                (int(tile_result_id),),
            ).fetchone()
        finally:
            conn.close()
        if not owner:
            return None
        rows = self.get_mes_review_aoi_candidates(
            int(owner["record_id"]),
            tile_result_id=int(tile_result_id),
        )
        return rows[0] if rows else None

    def get_mes_comparison_reviews(
        self,
        inference_record_ids: Optional[List[int]] = None,
    ) -> List[Dict]:
        """取得 Report 比對人工 Review，附目前有效 NG 樣本數。"""
        normalized_ids = sorted({
            int(value) for value in (inference_record_ids or [])
            if value is not None
        })
        conn = self._get_conn()
        try:
            def _fetch(where_sql: str = "", params: Optional[List[Any]] = None):
                return conn.execute(
                    f"""SELECT r.*,
                               (SELECT COUNT(*)
                                  FROM ng_validation_samples s
                                 WHERE s.review_id = r.id
                                   AND s.status = 'confirmed') AS ng_sample_count
                          FROM mes_comparison_review r
                          {where_sql}
                          ORDER BY r.updated_at DESC, r.id DESC""",
                    params or [],
                ).fetchall()

            if not normalized_ids:
                rows = _fetch()
            else:
                rows = []
                for offset in range(0, len(normalized_ids), 800):
                    chunk = normalized_ids[offset:offset + 800]
                    placeholders = ",".join("?" for _ in chunk)
                    rows.extend(_fetch(
                        f"WHERE r.inference_record_id IN ({placeholders})",
                        chunk,
                    ))
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_mes_comparison_review(self, inference_record_id: int) -> Optional[Dict]:
        rows = self.get_mes_comparison_reviews([int(inference_record_id)])
        return rows[0] if rows else None

    def save_mes_comparison_review(
        self,
        *,
        inference_record_id: int,
        glass_id: str,
        model_id: str,
        machine_no: str,
        request_time: str,
        ai_judgment: str,
        mes_judgment: str,
        review_type: str,
        category: str,
        note: str = "",
        reviewer: str = "",
        confirmed_ng: bool = False,
        samples: Optional[List[Dict]] = None,
    ) -> Dict:
        """UPSERT 人工 Review，並同步該 Review 選取的 NG 驗證樣本。"""
        review_type = str(review_type or "").strip()
        category = str(category or "").strip()
        valid_categories = self.VALID_MES_REVIEW_CATEGORIES.get(review_type)
        if valid_categories is None:
            raise ValueError(f"Invalid review type: {review_type}")
        if category not in valid_categories:
            raise ValueError(f"Invalid category for {review_type}: {category}")

        sample_rows = list(samples or [])
        if confirmed_ng and not sample_rows:
            raise ValueError("confirmed_ng requires at least one AOI sample")
        if not confirmed_ng:
            sample_rows = []

        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT INTO mes_comparison_review
                       (inference_record_id, glass_id, model_id, machine_no,
                        request_time, ai_judgment, mes_judgment,
                        review_type, category, note, reviewer, confirmed_ng)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(inference_record_id)
                       DO UPDATE SET
                           glass_id = excluded.glass_id,
                           model_id = excluded.model_id,
                           machine_no = excluded.machine_no,
                           request_time = excluded.request_time,
                           ai_judgment = excluded.ai_judgment,
                           mes_judgment = excluded.mes_judgment,
                           review_type = excluded.review_type,
                           category = excluded.category,
                           note = excluded.note,
                           reviewer = excluded.reviewer,
                           confirmed_ng = excluded.confirmed_ng,
                           updated_at = datetime('now', 'localtime')""",
                    (
                        int(inference_record_id),
                        str(glass_id or ""),
                        str(model_id or ""),
                        str(machine_no or ""),
                        str(request_time or ""),
                        str(ai_judgment or ""),
                        str(mes_judgment or ""),
                        review_type,
                        category,
                        str(note or ""),
                        str(reviewer or ""),
                        int(bool(confirmed_ng)),
                    ),
                )
                review_id = conn.execute(
                    """SELECT id FROM mes_comparison_review
                       WHERE inference_record_id = ?""",
                    (int(inference_record_id),),
                ).fetchone()["id"]

                conn.execute(
                    """UPDATE ng_validation_samples
                       SET status = 'removed',
                           updated_at = datetime('now', 'localtime')
                       WHERE review_id = ?""",
                    (review_id,),
                )

                for sample in sample_rows:
                    conn.execute(
                        """INSERT INTO ng_validation_samples
                           (review_id, inference_record_id, tile_result_id,
                            image_result_id, glass_id, model_id, machine_no,
                            request_time, image_name, source_image_path,
                            lighting, zone, aoi_defect_code,
                            aoi_product_x, aoi_product_y,
                            aoi_image_x, aoi_image_y,
                            tile_x, tile_y, tile_w, tile_h,
                            ai_score, crop_path, status)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                   ?, ?, ?, ?, ?, ?, ?, ?, ?, 'confirmed')
                           ON CONFLICT(review_id, tile_result_id)
                           DO UPDATE SET
                               image_result_id = excluded.image_result_id,
                               glass_id = excluded.glass_id,
                               model_id = excluded.model_id,
                               machine_no = excluded.machine_no,
                               request_time = excluded.request_time,
                               image_name = excluded.image_name,
                               source_image_path = excluded.source_image_path,
                               lighting = excluded.lighting,
                               zone = excluded.zone,
                               aoi_defect_code = excluded.aoi_defect_code,
                               aoi_product_x = excluded.aoi_product_x,
                               aoi_product_y = excluded.aoi_product_y,
                               aoi_image_x = excluded.aoi_image_x,
                               aoi_image_y = excluded.aoi_image_y,
                               tile_x = excluded.tile_x,
                               tile_y = excluded.tile_y,
                               tile_w = excluded.tile_w,
                               tile_h = excluded.tile_h,
                               ai_score = excluded.ai_score,
                               crop_path = excluded.crop_path,
                               status = 'confirmed',
                               updated_at = datetime('now', 'localtime')""",
                        (
                            review_id,
                            int(inference_record_id),
                            int(sample["tile_result_id"]),
                            int(sample["image_result_id"]),
                            str(glass_id or ""),
                            str(model_id or ""),
                            str(machine_no or ""),
                            str(request_time or ""),
                            str(sample.get("image_name") or ""),
                            str(sample.get("source_image_path") or ""),
                            str(sample.get("lighting") or ""),
                            str(sample.get("zone") or ""),
                            str(sample.get("aoi_defect_code") or ""),
                            int(sample.get("aoi_product_x", -1)),
                            int(sample.get("aoi_product_y", -1)),
                            int(sample.get("aoi_image_x", -1)),
                            int(sample.get("aoi_image_y", -1)),
                            int(sample.get("tile_x", 0)),
                            int(sample.get("tile_y", 0)),
                            int(sample.get("tile_w", 0)),
                            int(sample.get("tile_h", 0)),
                            float(sample.get("ai_score", 0.0)),
                            str(sample.get("crop_path") or ""),
                        ),
                    )

                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        return self.get_mes_comparison_review(int(inference_record_id))

    def delete_mes_comparison_review(self, inference_record_id: int) -> bool:
        """移除 Review 並停用其 NG 樣本；crop 保留供稽核與復原。"""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    """SELECT id FROM mes_comparison_review
                       WHERE inference_record_id = ?""",
                    (int(inference_record_id),),
                ).fetchone()
                if not row:
                    return False
                conn.execute(
                    """UPDATE ng_validation_samples
                       SET status = 'removed',
                           updated_at = datetime('now', 'localtime')
                       WHERE review_id = ?""",
                    (row["id"],),
                )
                conn.execute(
                    """DELETE FROM mes_comparison_review
                       WHERE inference_record_id = ?""",
                    (int(inference_record_id),),
                )
                conn.commit()
                return True
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def list_ng_validation_samples(
        self,
        *,
        machine_no: str = "",
        model_id: str = "",
        lighting: str = "",
        zone: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict], int]:
        where = ["s.status = 'confirmed'"]
        params: List[Any] = []
        for column, value in (
            ("s.machine_no", machine_no),
            ("s.model_id", model_id),
            ("s.lighting", lighting),
            ("s.zone", zone),
        ):
            if str(value or "").strip():
                normalized = str(value).strip()
                if column == "s.model_id" and normalized == "__unassigned__":
                    where.append("TRIM(COALESCE(s.model_id, '')) = ''")
                else:
                    where.append(f"{column} = ?")
                    params.append(normalized)
        where_sql = " AND ".join(where)
        limit = max(1, min(int(limit), 1000))
        offset = max(0, int(offset))

        conn = self._get_conn()
        try:
            total = conn.execute(
                f"""SELECT COUNT(*) FROM ng_validation_samples s
                    WHERE {where_sql}""",
                params,
            ).fetchone()[0]
            rows = conn.execute(
                f"""SELECT s.*, r.review_type, r.category,
                           r.note, r.reviewer, r.updated_at AS reviewed_at
                      FROM ng_validation_samples s
                      LEFT JOIN mes_comparison_review r ON r.id = s.review_id
                     WHERE {where_sql}
                     ORDER BY s.created_at DESC, s.id DESC
                     LIMIT ? OFFSET ?""",
                params + [limit, offset],
            ).fetchall()
            return [dict(row) for row in rows], int(total)
        finally:
            conn.close()

    def get_ng_validation_sample(self, sample_id: int) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT * FROM ng_validation_samples
                   WHERE id = ? AND status = 'confirmed'""",
                (int(sample_id),),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def remove_ng_validation_sample(self, sample_id: int) -> bool:
        """停用單筆 NG 驗證樣本；實體 crop 由呼叫端在路徑驗證後刪除。"""
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    """UPDATE ng_validation_samples
                       SET status = 'removed',
                           updated_at = datetime('now', 'localtime')
                       WHERE id = ? AND status = 'confirmed'""",
                    (int(sample_id),),
                )
                conn.commit()
                return cursor.rowcount > 0
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def get_ng_validation_summary(self) -> Dict:
        conn = self._get_conn()
        try:
            total_row = conn.execute(
                """SELECT COUNT(*) AS samples,
                          COUNT(DISTINCT review_id) AS reviews
                     FROM ng_validation_samples
                    WHERE status = 'confirmed'"""
            ).fetchone()
            lighting_rows = conn.execute(
                """SELECT lighting, COUNT(*) AS count
                     FROM ng_validation_samples
                    WHERE status = 'confirmed'
                    GROUP BY lighting
                    ORDER BY lighting"""
            ).fetchall()
            zone_rows = conn.execute(
                """SELECT zone, COUNT(*) AS count
                     FROM ng_validation_samples
                    WHERE status = 'confirmed'
                    GROUP BY zone
                    ORDER BY zone"""
            ).fetchall()
            model_rows = conn.execute(
                """SELECT model_id, COUNT(*) AS count
                     FROM ng_validation_samples
                    WHERE status = 'confirmed'
                    GROUP BY model_id
                    ORDER BY model_id"""
            ).fetchall()
            return {
                "samples": int(total_row["samples"] or 0),
                "reviews": int(total_row["reviews"] or 0),
                "by_lighting": {
                    str(row["lighting"] or "unknown"): int(row["count"])
                    for row in lighting_rows
                },
                "by_zone": {
                    str(row["zone"] or "unknown"): int(row["count"])
                    for row in zone_rows
                },
                "by_model": {
                    str(row["model_id"] or ""): int(row["count"])
                    for row in model_rows
                },
            }
        finally:
            conn.close()

    # ── 設定參數管理方法 ─────────────────────────────────

    @staticmethod
    def _hash_settings_password(password: str, salt: Optional[bytes] = None) -> str:
        salt = salt or os.urandom(16)
        iterations = 120000
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            str(password).encode("utf-8"),
            salt,
            iterations,
        )
        return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"

    @staticmethod
    def _verify_settings_password(password: str, password_hash: str) -> bool:
        try:
            algo, iter_text, salt_hex, digest_hex = str(password_hash).split("$", 3)
            if algo != "pbkdf2_sha256":
                return False
            expected = hashlib.pbkdf2_hmac(
                "sha256",
                str(password).encode("utf-8"),
                bytes.fromhex(salt_hex),
                int(iter_text),
            ).hex()
            return hmac.compare_digest(expected, digest_hex)
        except Exception:
            return False

    @staticmethod
    def _validate_settings_username(username: str) -> str:
        username = str(username or "").strip()
        if not username:
            raise ValueError("帳號不可空白")
        if len(username) > 32:
            raise ValueError("帳號長度不可超過 32 字")
        if any(ch.isspace() for ch in username):
            raise ValueError("帳號不可包含空白")
        return username

    @staticmethod
    def _validate_settings_password(password: str) -> str:
        password = str(password or "")
        if not password:
            raise ValueError("密碼不可空白")
        if len(password) > 128:
            raise ValueError("密碼長度不可超過 128 字")
        return password

    @staticmethod
    def _format_settings_user(row) -> Dict[str, Any]:
        return {
            "id": int(row["id"]),
            "username": row["username"],
            "is_admin": bool(row["is_admin"]),
            "can_manage_accounts": bool(row["is_admin"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_settings_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        username = str(username or "").strip()
        if not username:
            return None
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT id, username, is_admin, created_at, updated_at
                   FROM settings_users
                   WHERE username = ?""",
                (username,),
            ).fetchone()
            return self._format_settings_user(row) if row else None
        finally:
            conn.close()

    def verify_settings_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        username = str(username or "").strip()
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM settings_users WHERE username = ?",
                (username,),
            ).fetchone()
            if not row or not self._verify_settings_password(password, row["password_hash"]):
                return None
            return self._format_settings_user(row)
        finally:
            conn.close()

    def list_settings_users(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT id, username, is_admin, created_at, updated_at
                   FROM settings_users
                   ORDER BY is_admin DESC, username"""
            ).fetchall()
            return [self._format_settings_user(row) for row in rows]
        finally:
            conn.close()

    def create_settings_user(
        self, username: str, password: str, is_admin: bool = False
    ) -> Dict[str, Any]:
        username = self._validate_settings_username(username)
        password = self._validate_settings_password(password)
        with self._lock:
            conn = self._get_conn()
            try:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cur = conn.execute(
                    """INSERT INTO settings_users
                       (username, password_hash, is_admin, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        username,
                        self._hash_settings_password(password),
                        1 if is_admin else 0,
                        now,
                        now,
                    ),
                )
                conn.commit()
                row = conn.execute(
                    """SELECT id, username, is_admin, created_at, updated_at
                       FROM settings_users WHERE id = ?""",
                    (cur.lastrowid,),
                ).fetchone()
                return self._format_settings_user(row)
            except sqlite3.IntegrityError:
                conn.rollback()
                raise ValueError("帳號已存在")
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def update_settings_user(
        self,
        user_id: int,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM settings_users WHERE id = ?",
                    (int(user_id),),
                ).fetchone()
                if not row:
                    return None

                updates = []
                params: List[Any] = []
                if username is not None:
                    new_username = self._validate_settings_username(username)
                    if row["username"] == "admin" and new_username != "admin":
                        raise ValueError("admin 帳號名稱不可修改")
                    updates.append("username = ?")
                    params.append(new_username)
                if password is not None and password != "":
                    updates.append("password_hash = ?")
                    params.append(self._hash_settings_password(self._validate_settings_password(password)))
                if updates:
                    updates.append("updated_at = ?")
                    params.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    params.append(int(user_id))
                    conn.execute(
                        f"UPDATE settings_users SET {', '.join(updates)} WHERE id = ?",
                        params,
                    )
                    conn.commit()

                new_row = conn.execute(
                    """SELECT id, username, is_admin, created_at, updated_at
                       FROM settings_users WHERE id = ?""",
                    (int(user_id),),
                ).fetchone()
                return self._format_settings_user(new_row)
            except sqlite3.IntegrityError:
                conn.rollback()
                raise ValueError("帳號已存在")
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def delete_settings_user(self, user_id: int) -> bool:
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT username, is_admin FROM settings_users WHERE id = ?",
                    (int(user_id),),
                ).fetchone()
                if not row:
                    return False
                if row["username"] == "admin" or int(row["is_admin"]):
                    raise ValueError("admin 帳號不可刪除")
                conn.execute("DELETE FROM settings_users WHERE id = ?", (int(user_id),))
                conn.commit()
                return True
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def get_config_param(self, param_name: str) -> Optional[Dict]:
        """取得單一設定參數"""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM config_params WHERE param_name = ?",
                (param_name,)
            ).fetchone()
            if row:
                result = dict(row)
                result["decoded_value"] = self._decode_config_value(
                    result["param_value"], result["param_type"]
                )
                return result
            return None
        finally:
            conn.close()

    def get_all_config_params(self) -> List[Dict]:
        """取得所有設定參數"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM config_params ORDER BY id"
            ).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                r["decoded_value"] = self._decode_config_value(
                    r["param_value"], r["param_type"]
                )
                results.append(r)
            return results
        finally:
            conn.close()

    def update_config_param(
        self, param_name: str, new_value: Any, reason: str = "", changed_by: str = ""
    ) -> bool:
        """
        更新設定參數並記錄修改歷史

        Args:
            param_name: 參數名稱
            new_value: 新值 (Python 原生型別)
            reason: 修改原因
            changed_by: 修改帳號

        Returns:
            是否更新成功
        """
        changed_by = str(changed_by or "").strip()[:64]
        with self._lock:
            conn = self._get_conn()
            try:
                # 取得舊值
                old_row = conn.execute(
                    "SELECT param_value, param_type FROM config_params WHERE param_name = ?",
                    (param_name,)
                ).fetchone()

                new_value_json = json.dumps(new_value, ensure_ascii=False)
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(new_value, bool):
                    new_param_type = "bool"
                elif isinstance(new_value, int):
                    new_param_type = "int"
                elif isinstance(new_value, float):
                    new_param_type = "float"
                elif isinstance(new_value, dict):
                    new_param_type = "dict"
                elif isinstance(new_value, list):
                    new_param_type = "list"
                else:
                    new_param_type = "str"

                if not old_row:
                    # 參數不存在於 DB → 自動新增 (從 config dataclass 補上的參數)
                    param_type = new_param_type
                    old_value = ""
                    conn.execute(
                        """INSERT INTO config_params
                           (param_name, param_value, param_type, updated_at)
                           VALUES (?, ?, ?, ?)""",
                        (param_name, new_value_json, param_type, now)
                    )
                else:
                    old_value = old_row["param_value"]
                    param_type = old_row["param_type"]
                    updated_param_type = param_type
                    if param_type == "str" and isinstance(new_value, bool):
                        updated_param_type = "bool"
                    elif (
                        param_type in ("int", "float")
                        and isinstance(new_value, (int, float))
                        and not isinstance(new_value, bool)
                    ):
                        updated_param_type = (
                            "float" if (param_type == "float" or isinstance(new_value, float))
                            else "int"
                        )

                    # 更新設定值
                    conn.execute(
                        "UPDATE config_params SET param_value = ?, param_type = ?, updated_at = ? WHERE param_name = ?",
                        (new_value_json, updated_param_type, now, param_name)
                    )

                # 記錄修改歷史
                conn.execute(
                    """INSERT INTO config_change_history
                       (param_name, old_value, new_value, change_reason, changed_by, changed_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (param_name, old_value, new_value_json, reason, changed_by, now)
                )

                conn.commit()
                return True
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def get_config_change_history(
        self, param_name: str = "", limit: int = 50
    ) -> List[Dict]:
        """查詢設定修改歷史紀錄"""
        conn = self._get_conn()
        try:
            if param_name:
                rows = conn.execute(
                    """SELECT * FROM config_change_history
                       WHERE param_name = ?
                       ORDER BY changed_at DESC
                       LIMIT ?""",
                    (param_name, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM config_change_history
                       ORDER BY changed_at DESC
                       LIMIT ?""",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def init_config_from_yaml(self, config) -> int:
        """
        從 CAPIConfig 物件初始化 DB 設定 (僅在 DB 無設定時執行)

        Args:
            config: CAPIConfig 物件

        Returns:
            新增的參數數量
        """
        # 定義要遷移的參數
        params_def = [
            ("anomaly_threshold", config.anomaly_threshold, "float", "異常分數閾值 (fallback)"),
            ("inference_rotate_180_enabled", config.inference_rotate_180_enabled, "bool", "推論來源影像統一旋轉 180°（正式推論、規格內判定與 Debug 共用；不修改原始檔）"),
            ("model_mapping", config.model_mapping, "dict", "前綴 → 模型路徑映射"),
            ("threshold_mapping", config.threshold_mapping, "dict", "前綴 → 獨立閾值映射"),
            ("patchcore_filter_enabled", config.patchcore_filter_enabled, "bool", "啟用 PatchCore 後處理進階過濾"),
            ("patchcore_blur_sigma", config.patchcore_blur_sigma, "float", "異常圖高斯平滑強度 (抑制噪點)"),
            ("patchcore_min_area", config.patchcore_min_area, "int", "異常判定最小連通面積(px)"),
            ("patchcore_score_metric", config.patchcore_score_metric, "string", "計分方式 (max, top_k_avg, percentile_99)"),
            ("mark_exclusion_padding_px", config.mark_exclusion_padding_px, "int", "MARK 不檢測區硬遮罩外擴 (px)"),
            ("mark_exclusion_soft_decay_px", config.mark_exclusion_soft_decay_px, "int", "MARK 硬遮罩外圈 heatmap 降權寬度 (px)"),
            ("cv_edge_exclude_soft_decay_px", config.cv_edge_exclude_soft_decay_px, "int", "手動不檢測區外圈 heatmap 降權寬度 (px)"),
            ("no_detect_soft_decay_min_weight", config.no_detect_soft_decay_min_weight, "float", "不檢測區 soft decay 邊界最低權重 (0~1)"),
            ("dust_brightness_threshold", config.dust_brightness_threshold, "int", "灰塵亮度閾值 (備用)"),
            ("dust_area_min", config.dust_area_min, "int", "灰塵顆粒最小面積 (px)"),
            ("dust_area_max", config.dust_area_max, "int", "灰塵顆粒最大面積 (px)"),
            ("dust_extension", config.dust_extension, "int", "灰塵區域膨脹像素"),
            ("dust_pixel_grid_filter_enabled", config.dust_pixel_grid_filter_enabled, "bool", "啟用 OMIT 像素紋理平滑（所有產品解析度）"),
            ("dust_pixel_grid_blur_kernel", config.dust_pixel_grid_blur_kernel, "int", "OMIT Gaussian 核大小（奇數，建議 7）"),
            ("dust_heatmap_iou_threshold", config.dust_heatmap_iou_threshold, "float", "Heatmap-Dust IOU/Coverage 閾值"),
            ("dust_heatmap_top_percent", config.dust_heatmap_top_percent, "float", "Heatmap 熱區取前 X%；two-stage REAL 特徵須落在此核心附近"),
            ("dust_heatmap_metric", config.dust_heatmap_metric, "string", 'Heatmap 判定指標: "coverage" (覆蓋率) 或是 "iou"'),
            ("dust_detect_dark_particles", config.dust_detect_dark_particles, "bool", "偵測暗色顆粒/圖案 (如偏黑 MARK) 並過濾"),
            # Otsu 邊緣裁切
            ("otsu_offset", config.otsu_offset, "int", "Otsu 產品邊緣裁切內縮 (px)"),
            # CV 邊緣檢測
            ("cv_edge_enabled", False, "bool", "是否啟用傳統 CV 邊緣檢測"),
            ("cv_edge_dust_filter_enabled", False, "bool", "是否啟用 CV 邊緣檢測的灰塵過濾"),
            ("cv_edge_left_width", 450, "int", "左邊界檢測寬度 (px)"),
            ("cv_edge_left_threshold", 5, "int", "左邊界明暗差閾值"),
            ("cv_edge_left_min_area", 70, "int", "左邊界最小缺陷面積 (px)"),
            ("cv_edge_left_exclude_top", 80, "int", "左邊界避開上 (px)"),
            ("cv_edge_left_exclude_bottom", 80, "int", "左邊界避開下 (px)"),
            ("cv_edge_left_exclude_left", 10, "int", "左邊界避開左 (px)"),
            ("cv_edge_left_exclude_right", 10, "int", "左邊界避開右 (px)"),
            ("cv_edge_right_width", 650, "int", "右邊界檢測寬度 (px)"),
            ("cv_edge_right_threshold", 5, "int", "右邊界明暗差閾值"),
            ("cv_edge_right_min_area", 60, "int", "右邊界最小缺陷面積 (px)"),
            ("cv_edge_right_exclude_top", 110, "int", "右邊界避開上 (px)"),
            ("cv_edge_right_exclude_bottom", 110, "int", "右邊界避開下 (px)"),
            ("cv_edge_right_exclude_left", 100, "int", "右邊界避開左 (px)"),
            ("cv_edge_right_exclude_right", 10, "int", "右邊界避開右 (px)"),
            ("cv_edge_top_width", 550, "int", "上邊界檢測寬度 (px)"),
            ("cv_edge_top_threshold", 5, "int", "上邊界明暗差閾值"),
            ("cv_edge_top_min_area", 60, "int", "上邊界最小缺陷面積 (px)"),
            ("cv_edge_top_exclude_top", 10, "int", "上邊界避開上 (px)"),
            ("cv_edge_top_exclude_bottom", 10, "int", "上邊界避開下 (px)"),
            ("cv_edge_top_exclude_left", 80, "int", "上邊界避開左 (px)"),
            ("cv_edge_top_exclude_right", 80, "int", "上邊界避開右 (px)"),
            ("cv_edge_bottom_width", 360, "int", "下邊界檢測寬度 (px)"),
            ("cv_edge_bottom_threshold", 4, "int", "下邊界明暗差閾值"),
            ("cv_edge_bottom_min_area", 65, "int", "下邊界最小缺陷面積 (px)"),
            ("cv_edge_bottom_exclude_top", 10, "int", "下邊界避開上 (px)"),
            ("cv_edge_bottom_exclude_bottom", 10, "int", "下邊界避開下 (px)"),
            ("cv_edge_bottom_exclude_left", 80, "int", "下邊界避開左 (px)"),
            ("cv_edge_bottom_exclude_right", 80, "int", "下邊界避開右 (px)"),
            # 邊緣檢測排除區域
            ("cv_edge_exclude_enabled", False, "bool", "是否啟用邊緣檢測排除區域"),
            ("cv_edge_exclude_x", 0, "int", "排除區域起始 X (px)"),
            ("cv_edge_exclude_y", 0, "int", "排除區域起始 Y (px)"),
            ("cv_edge_exclude_w", 100, "int", "排除區域寬度 (px)"),
            ("cv_edge_exclude_h", 100, "int", "排除區域高度 (px)"),
            ("cv_edge_exclude_zones", [], "dict", "不檢測排除區域列表 (適用於 PatchCore 推論及邊緣檢測)"),
            ("cv_edge_aoi_threshold", 4, "int", "AOI 座標邊緣明暗差閾值 (獨立於四邊)"),
            ("cv_edge_aoi_min_area", 40, "int", "AOI 座標邊緣最小缺陷面積 (px, 獨立於四邊)"),
            ("cv_edge_aoi_solidity_min", 0.2, "float", "AOI 邊緣 Solidity 下限 (低於此值視為 L 形偽影排除, 0=停用)"),
            ("cv_edge_aoi_polygon_erode_px", 3, "int", "AOI 邊緣 polygon fg_mask 內縮 px 數 (避開面板邊緣亮帶轉換區, 0=停用; 僅 polygon 模式有效)"),
            ("cv_edge_aoi_morph_open_kernel", 3, "int", "AOI 邊緣二值化後 morphological opening kernel 大小 (去除 1-px 條紋與細雜訊橋, 0=停用)"),
            ("cv_edge_aoi_min_max_diff", 20, "int", "AOI 邊緣 component 最大 diff 下限 (低於此值視為低對比紋理雜訊, 建議 threshold×5~7, 0=停用)"),
            ("cv_edge_aoi_line_min_length", 30, "int", "AOI 邊緣薄線偵測最小長度 px (投影法, 旁路 min_max_diff/solidity 過濾以抓faint 線狀缺陷; 0=停用)"),
            ("cv_edge_aoi_line_max_width", 3, "int", "AOI 邊緣薄線最大寬度 px (超過視為一般 component, 由 CC path 處理)"),
            ("aoi_edge_inspector", "cv", "string", "AOI 座標邊緣 inspector: 'cv' (傳統 CV) | 'patchcore' (PatchCore 模型) | 'fusion' (Phase 6 空間分權，CV 管 band+PC 管 interior)"),
            ("aoi_edge_boundary_band_px", 40, "int", "AOI 邊緣 fusion 模式 CV 管轄帶寬度 (polygon 邊往 panel 內延伸 px), 僅 inspector='fusion' 時生效, 0=等同 patchcore"),
            ("aoi_edge_pc_roi_inward_shift_enabled", True, "bool", "Phase 7: fusion 模式下 PC ROI 自動內移到距 polygon ≥ band_px 處，讓 PC feature map 完全脫離 panel 邊 discontinuity，進一步抑制近邊過檢；凹角 polygon 會 fallback"),
            # B0F 亮點偵測設定
            ("bright_spot_threshold", 200, "int", "絕對亮度上限 (超過直接判定亮點)"),
            ("bright_spot_min_area", 5, "int", "亮點最小連通面積 (px)"),
            ("bright_spot_median_kernel", 21, "int", "背景估計 median filter 核大小"),
            ("bright_spot_diff_threshold", 10, "int", "局部對比差異閾值"),
            ("within_spec_judgment_rules", config.within_spec_judgment_rules, "dict", "規格內點狀不良判定條件（依機種/畫面/黑白點分開設定；dot_detection.segmentation_method 可選 background_diff、hysteresis(V2 雙參數組)、morph_hat、adaptive_mean、halo、auto 或 off 關閉）"),
            # 畫異設定
            ("image_abnormal_detection_enabled", config.image_abnormal_detection_enabled, "bool", "啟用推論前畫異預檢（只檢查 AOI Report 涉及畫面的產品 polygon 內平均亮度，低於下限或高於上限時回報 PCO05）"),
            ("image_abnormal_standard_mean_lower", config.image_abnormal_standard_mean_lower, "int", "STANDARD 產品區平均亮度下限"),
            ("image_abnormal_standard_mean_upper", config.image_abnormal_standard_mean_upper, "int", "STANDARD 產品區平均亮度上限"),
            ("image_abnormal_wgf50500_mean_lower", config.image_abnormal_wgf50500_mean_lower, "int", "WGF50500 產品區平均亮度下限"),
            ("image_abnormal_wgf50500_mean_upper", config.image_abnormal_wgf50500_mean_upper, "int", "WGF50500 產品區平均亮度上限"),
            ("image_abnormal_g0f00000_mean_lower", config.image_abnormal_g0f00000_mean_lower, "int", "G0F00000 產品區平均亮度下限"),
            ("image_abnormal_g0f00000_mean_upper", config.image_abnormal_g0f00000_mean_upper, "int", "G0F00000 產品區平均亮度上限"),
            ("image_abnormal_r0f00000_mean_lower", config.image_abnormal_r0f00000_mean_lower, "int", "R0F00000 產品區平均亮度下限"),
            ("image_abnormal_r0f00000_mean_upper", config.image_abnormal_r0f00000_mean_upper, "int", "R0F00000 產品區平均亮度上限"),
            ("image_abnormal_w0f00000_mean_lower", config.image_abnormal_w0f00000_mean_lower, "int", "W0F00000 產品區平均亮度下限"),
            ("image_abnormal_w0f00000_mean_upper", config.image_abnormal_w0f00000_mean_upper, "int", "W0F00000 產品區平均亮度上限"),
            ("image_abnormal_b0f00000_mean_lower", config.image_abnormal_b0f00000_mean_lower, "int", "B0F00000 產品區平均亮度下限"),
            ("image_abnormal_b0f00000_mean_upper", config.image_abnormal_b0f00000_mean_upper, "int", "B0F00000 產品區平均亮度上限"),
            # 回報結果設定
            ("report_black_dot_defect_code", config.report_black_dot_defect_code, "string", "QJPG 回報格式：黑點 defect code"),
            ("report_white_dot_defect_code", config.report_white_dot_defect_code, "string", "QJPG 回報格式：白點 defect code"),
            ("report_unknown_dot_defect_code", config.report_unknown_dot_defect_code, "string", "QJPG 回報格式：無法判斷黑/白點時使用的 defect code"),
            ("report_bomb_defect_code", config.report_bomb_defect_code, "string", "QJPG 回報格式：炸彈 defect code"),
            ("report_image_abnormal_defect_code", config.report_image_abnormal_defect_code, "string", "QJPG 回報格式：畫異 defect code"),
            # AOI 機檢座標設定
            ("grid_tiling_enabled", True, "bool", "啟用全面板 Grid Tiling 推論"),
            # 新架構 attribution 模式（找包含 AOI 座標的既存 grid tile 標屬性）成本近零，
            # 預設開啟，否則記錄頁的 🎯 AOI 機檢座標推論 區塊永遠不會出現。
            ("aoi_coord_inspection_enabled", True, "bool", "啟用 AOI 機檢座標推論"),
            ("aoi_heatmap_center_seed_enabled", config.aoi_heatmap_center_seed_enabled, "bool", "啟用 AOI 中心 seed 保護，Top% heatmap 額外保留 AOI 座標附近弱熱區"),
            ("bomb_area_force_detection_enabled", config.bomb_area_force_detection_enabled, "bool", "炸彈區域強制偵測：AOI Report 未給 Client 炸彈座標時補切 tile 偵測"),
            ("aoi_report_path_replace_from", "yuantu", "string", "報告路徑替換來源字串"),
            ("aoi_report_path_replace_to", "Report", "string", "報告路徑替換目標字串"),
        ]

        # 新架構：threshold_mapping / model_mapping 不灌進 DB，避免「首次啟動灌
        # 舊值 → 後續 yaml 改動被 DB 蓋掉」這條漏水路徑。這兩個 key 對 v2 而言
        # 屬於 bundle 內 yaml 自包含的設定，唯一來源就是 machine_config.yaml。
        if getattr(config, "is_new_architecture", False):
            params_def = [p for p in params_def if p[0] not in ("threshold_mapping", "model_mapping")]

        image_abnormal_default_migrations = {
            "image_abnormal_standard_mean_lower": (47, config.image_abnormal_standard_mean_lower),
            "image_abnormal_standard_mean_upper": (67, config.image_abnormal_standard_mean_upper),
            "image_abnormal_wgf50500_mean_lower": (50, config.image_abnormal_wgf50500_mean_lower),
            "image_abnormal_wgf50500_mean_upper": (70, config.image_abnormal_wgf50500_mean_upper),
            "image_abnormal_g0f00000_mean_lower": (46, config.image_abnormal_g0f00000_mean_lower),
            "image_abnormal_g0f00000_mean_upper": (66, config.image_abnormal_g0f00000_mean_upper),
            "image_abnormal_r0f00000_mean_lower": (50, config.image_abnormal_r0f00000_mean_lower),
            "image_abnormal_r0f00000_mean_upper": (70, config.image_abnormal_r0f00000_mean_upper),
            "image_abnormal_w0f00000_mean_lower": (49, config.image_abnormal_w0f00000_mean_lower),
            "image_abnormal_w0f00000_mean_upper": (69, config.image_abnormal_w0f00000_mean_upper),
            "image_abnormal_b0f00000_mean_lower": (0, config.image_abnormal_b0f00000_mean_lower),
            "image_abnormal_b0f00000_mean_upper": (12, config.image_abnormal_b0f00000_mean_upper),
        }
        image_abnormal_param_names = {
            name for name, _value, _ptype, _desc in params_def
            if name.startswith("image_abnormal_")
        }
        description_refresh_param_names = image_abnormal_param_names | {
            "dust_pixel_grid_filter_enabled",
            "dust_pixel_grid_blur_kernel",
        }

        count = 0
        with self._lock:
            conn = self._get_conn()
            try:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for name, value, ptype, desc in params_def:
                    # 只在 DB 中尚無此參數時才新增
                    existing = conn.execute(
                        "SELECT id, param_value, param_type FROM config_params WHERE param_name = ?",
                        (name,)
                    ).fetchone()
                    if not existing:
                        value_json = json.dumps(value, ensure_ascii=False)
                        conn.execute(
                            """INSERT INTO config_params
                               (param_name, param_value, param_type, description, updated_at)
                               VALUES (?, ?, ?, ?, ?)""",
                            (name, value_json, ptype, desc, now)
                        )
                        count += 1
                    else:
                        if name in description_refresh_param_names:
                            conn.execute(
                                "UPDATE config_params SET description = ? WHERE param_name = ?",
                                (desc, name)
                            )
                        if name in image_abnormal_default_migrations:
                            old_default, new_default = image_abnormal_default_migrations[name]
                            current_value = self._decode_config_value(
                                existing["param_value"],
                                existing["param_type"],
                            )
                            if current_value == old_default:
                                new_value_json = json.dumps(new_default, ensure_ascii=False)
                                conn.execute(
                                    """UPDATE config_params
                                       SET param_value = ?, description = ?, updated_at = ?
                                       WHERE param_name = ?""",
                                    (new_value_json, desc, now, name)
                                )
                                conn.execute(
                                    """INSERT INTO config_change_history
                                       (param_name, old_value, new_value, change_reason, changed_by, changed_at)
                                       VALUES (?, ?, ?, ?, ?, ?)""",
                                    (
                                        name,
                                        existing["param_value"],
                                        new_value_json,
                                        "自動更新畫異 polygon mean 預設門檻",
                                        "system",
                                        now,
                                    )
                                )
                conn.commit()
                return count
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    @staticmethod
    def _decode_config_value(value_json: str, param_type: str) -> Any:
        """將 JSON 字串解碼為 Python 原生型別"""
        try:
            value = json.loads(value_json)
            if param_type == "float":
                return float(value)
            elif param_type == "int":
                return int(value)
            elif param_type == "dict":
                return dict(value) if value else {}
            return value
        except (json.JSONDecodeError, TypeError, ValueError):
            return value_json

    # ------------------------------------------------------------------
    # Training Job CRUD
    # ------------------------------------------------------------------

    def create_training_job(
        self,
        job_id: str,
        machine_id: str,
        panel_paths: list,
        training_params: Optional[Dict[str, Any]] = None,
        panel_modes: Optional[list] = None,
        training_scope: Optional[Dict[str, Any]] = None,
        training_data_source: Optional[Dict[str, Any]] = None,
        image_preprocess_pipeline: Optional[list] = None,
        preprocess_after_tiling: bool = False,
        tile_stride: Optional[int] = 256,
        image_preprocess_pipelines: Optional[Dict[str, list]] = None,
    ) -> int:
        """建立一筆新的訓練 job，初始 state 為 'preprocess'。回傳 rowid。

        training_params 為 step1 使用者覆寫的 PatchCore 超參數（JSON 序列化後寫入），
        None 表示完全使用 TrainingConfig 的 dataclass 預設值。

        panel_modes 為與 panel_paths 等長的 list，元素 full / inner_only / edge_only / corners_only。
        None 寫入 NULL，由 caller / get_*_training_job 視同全 full 處理。
        """
        # 用 `is not None` 而非 falsy；空 dict 與 None 語意不同（前者代表
        # 來源已驗證但無覆寫項，留給呼叫端決定是否要寫入）。
        params_json = json.dumps(training_params) if training_params is not None else None
        modes_json = json.dumps(panel_modes) if panel_modes is not None else None
        scope_json = json.dumps(training_scope) if training_scope is not None else None
        source_json = (
            json.dumps(training_data_source, ensure_ascii=False)
            if training_data_source is not None else None
        )
        preprocess_json = (
            json.dumps(image_preprocess_pipeline, ensure_ascii=False)
            if image_preprocess_pipeline is not None else None
        )
        preprocess_zones_json = (
            json.dumps(image_preprocess_pipelines, ensure_ascii=False)
            if image_preprocess_pipelines is not None else None
        )
        tile_stride_value = int(tile_stride) if tile_stride is not None else None
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO training_jobs
                   (job_id, machine_id, state, started_at, panel_paths, panel_modes,
                    training_params, training_scope, training_data_source,
                    image_preprocess_pipeline, image_preprocess_pipelines,
                    preprocess_after_tiling, tile_stride)
                   VALUES (?, ?, 'preprocess', datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id, machine_id, json.dumps(panel_paths), modes_json,
                    params_json, scope_json, source_json, preprocess_json, preprocess_zones_json,
                    1 if preprocess_after_tiling else 0, tile_stride_value,
                ),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    @staticmethod
    def _decode_training_job_row(cols: list, row: tuple) -> Dict:
        """共用解碼：JSON 欄位反序列化、舊 job 的 panel_modes 補成全 full。"""
        job = dict(zip(cols, row))
        if job.get("panel_paths"):
            job["panel_paths"] = json.loads(job["panel_paths"])
        else:
            job["panel_paths"] = []
        raw_modes = job.get("panel_modes")
        if raw_modes:
            job["panel_modes"] = json.loads(raw_modes)
        else:
            # 舊 job 沒有 panel_modes 欄位 → 視同全 full（與 8-panel wizard 前的行為一致）
            job["panel_modes"] = ["full"] * len(job["panel_paths"])
        raw_params = job.get("training_params")
        job["training_params"] = json.loads(raw_params) if raw_params else None
        raw_scope = job.get("training_scope")
        job["training_scope"] = json.loads(raw_scope) if raw_scope else None
        raw_source = job.get("training_data_source")
        job["training_data_source"] = (
            json.loads(raw_source) if raw_source else {"type": "inference_records"}
        )
        raw_preprocess = job.get("image_preprocess_pipeline")
        job["image_preprocess_pipeline"] = json.loads(raw_preprocess) if raw_preprocess else []
        raw_zone_preprocess = job.get("image_preprocess_pipelines")
        job["image_preprocess_pipelines"] = (
            json.loads(raw_zone_preprocess) if raw_zone_preprocess else {}
        )
        job["preprocess_after_tiling"] = bool(job.get("preprocess_after_tiling", 0))
        raw_tile_stride = job.get("tile_stride")
        job["tile_stride"] = int(raw_tile_stride) if raw_tile_stride else 512
        return job

    def get_training_job(self, job_id: str) -> Optional[Dict]:
        """依 job_id 查詢訓練 job，panel_paths / panel_modes / training_params 自動 JSON 反序列化。找不到回傳 None。"""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM training_jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return self._decode_training_job_row(cols, row)
        finally:
            conn.close()

    def update_training_job_state(
        self,
        job_id: str,
        state: str,
        error_message: Optional[str] = None,
        output_bundle: Optional[str] = None,
    ) -> None:
        """更新訓練 job 的 state，並選擇性設定 error_message / output_bundle。
        state 為 'completed' 或 'failed' 時自動填入 completed_at。
        """
        fields = ["state = ?"]
        args: list = [state]
        if state in ("completed", "failed"):
            fields.append("completed_at = datetime('now')")
        if error_message is not None:
            fields.append("error_message = ?")
            args.append(error_message)
        if output_bundle is not None:
            fields.append("output_bundle = ?")
            args.append(output_bundle)
        args.append(job_id)
        conn = self._get_conn()
        try:
            conn.execute(
                f"UPDATE training_jobs SET {', '.join(fields)} WHERE job_id = ?",
                tuple(args),
            )
            conn.commit()
        finally:
            conn.close()

    def get_active_training_job(self) -> Optional[Dict]:
        """回傳目前進行中的 job（preprocess / review / train），依 started_at DESC 取最新一筆。"""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT * FROM training_jobs
                   WHERE state IN ('preprocess', 'review', 'train')
                   ORDER BY started_at DESC LIMIT 1"""
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return self._decode_training_job_row(cols, row)
        finally:
            conn.close()

    def list_active_training_jobs(self) -> List[Dict]:
        """回傳所有 state 在 (preprocess, review, train) 的 job，依 started_at DESC 排序。"""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """SELECT * FROM training_jobs
                   WHERE state IN ('preprocess', 'review', 'train')
                   ORDER BY started_at DESC"""
            )
            rows = cur.fetchall()
            if not rows:
                return []
            cols = [d[0] for d in cur.description]
            return [self._decode_training_job_row(cols, row) for row in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # training_tile_pool CRUD
    # ------------------------------------------------------------------

    def insert_tile_pool(self, job_id: str, tiles: list) -> list:
        """批次插入 tile pool 紀錄，回傳各列的 lastrowid 清單。"""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            ids = []
            for t in tiles:
                cur.execute(
                    """INSERT INTO training_tile_pool
                       (job_id, lighting, zone, source, source_path, thumb_path)
                       VALUES (?,?,?,?,?,?)""",
                    (job_id, t["lighting"], t.get("zone"), t["source"],
                     t["source_path"], t.get("thumb_path")),
                )
                ids.append(cur.lastrowid)
            conn.commit()
            return ids
        finally:
            conn.close()

    def list_tile_pool(self, job_id: str, lighting: str = None, zone: str = None,
                       source: str = None, decision: str = None) -> list:
        """查詢 tile pool，支援 lighting / zone / source / decision 任意組合過濾。"""
        sql = "SELECT * FROM training_tile_pool WHERE job_id = ?"
        args = [job_id]
        for fld, val in [("lighting", lighting), ("zone", zone),
                         ("source", source), ("decision", decision)]:
            if val is not None:
                sql += f" AND {fld} = ?"
                args.append(val)
        sql += " ORDER BY id"
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(args))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            conn.close()

    def update_tile_decisions(self, job_id: str, tile_ids: list, decision: str) -> None:
        """批次更新指定 tile id 清單的 decision 欄位（空清單時為 no-op）。"""
        if not tile_ids:
            return
        placeholders = ",".join("?" * len(tile_ids))
        conn = self._get_conn()
        try:
            conn.execute(
                f"UPDATE training_tile_pool SET decision = ? WHERE job_id = ? AND id IN ({placeholders})",
                (decision, job_id, *tile_ids),
            )
            conn.commit()
        finally:
            conn.close()

    def cleanup_tile_pool(self, job_id: str) -> None:
        """刪除 job 的所有 tile pool 紀錄（thumb 檔不刪，由 caller 處理）。"""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM training_tile_pool WHERE job_id = ?", (job_id,))
            conn.commit()
        finally:
            conn.close()

    def delete_tile_pool_by_source_paths(self, job_id: str, source_paths: list) -> int:
        """依 source_path 刪除指定 job 的 tile pool 紀錄，回傳刪除筆數。"""
        paths = [str(p) for p in source_paths if str(p).strip()]
        if not job_id or not paths:
            return 0
        placeholders = ",".join("?" * len(paths))
        conn = self._get_conn()
        try:
            cur = conn.execute(
                f"DELETE FROM training_tile_pool WHERE job_id = ? AND source_path IN ({placeholders})",
                (job_id, *paths),
            )
            conn.commit()
            return int(cur.rowcount or 0)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # tile_score_cache CRUD
    # ------------------------------------------------------------------

    def insert_score_cache(self, rows: list) -> None:
        """批次 UPSERT (tile_id, scoring_bundle_id) → score。空清單 no-op。"""
        if not rows:
            return
        conn = self._get_conn()
        try:
            conn.executemany(
                """INSERT INTO tile_score_cache
                       (tile_id, scoring_bundle_id, score, computed_at)
                   VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(tile_id, scoring_bundle_id)
                   DO UPDATE SET score = excluded.score,
                                 computed_at = CURRENT_TIMESTAMP""",
                [(r["tile_id"], r["scoring_bundle_id"], r["score"]) for r in rows],
            )
            conn.commit()
        finally:
            conn.close()

    def get_score_cache(self, scoring_bundle_id: int, tile_ids: list) -> dict:
        """回傳 {tile_id: score}，只包含 cache 中存在的 row。空 tile_ids → 空 dict。"""
        if not tile_ids:
            return {}
        placeholders = ",".join("?" * len(tile_ids))
        conn = self._get_conn()
        try:
            cur = conn.execute(
                f"""SELECT tile_id, score FROM tile_score_cache
                    WHERE scoring_bundle_id = ? AND tile_id IN ({placeholders})""",
                (scoring_bundle_id, *tile_ids),
            )
            return {row[0]: row[1] for row in cur.fetchall()}
        finally:
            conn.close()

    def delete_score_cache(self, scoring_bundle_id: int = None,
                           tile_ids: list = None,
                           lighting: str = None, zone: str = None) -> int:
        """彈性刪除 cache。回傳刪除筆數。

        - scoring_bundle_id only: 清該 bundle 全部
        - tile_ids only: 清這些 tile 在所有 bundle 的分
        - scoring_bundle_id + lighting + zone: 清該 bundle 對該 lighting+zone tile 的分
          （join training_tile_pool 過濾）
        """
        conn = self._get_conn()
        try:
            if scoring_bundle_id is not None and lighting is not None and zone is not None:
                cur = conn.execute(
                    """DELETE FROM tile_score_cache
                       WHERE scoring_bundle_id = ?
                         AND tile_id IN (
                           SELECT id FROM training_tile_pool
                           WHERE lighting = ? AND zone = ?
                         )""",
                    (scoring_bundle_id, lighting, zone),
                )
            elif tile_ids:
                placeholders = ",".join("?" * len(tile_ids))
                cur = conn.execute(
                    f"DELETE FROM tile_score_cache WHERE tile_id IN ({placeholders})",
                    tuple(tile_ids),
                )
            elif scoring_bundle_id is not None:
                cur = conn.execute(
                    "DELETE FROM tile_score_cache WHERE scoring_bundle_id = ?",
                    (scoring_bundle_id,),
                )
            else:
                return 0
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # model_registry CRUD
    # ------------------------------------------------------------------

    def register_model_bundle(self, info: dict) -> int:
        """新增一筆 model_registry 紀錄，is_active 預設為 0，回傳 rowid。"""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO model_registry
                   (machine_id, bundle_path, trained_at, panel_count, inner_tile_count,
                    edge_tile_count, ng_tile_count, bundle_size_bytes, is_active, job_id, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (info["machine_id"], info["bundle_path"], info["trained_at"],
                 info.get("panel_count"), info.get("inner_tile_count"),
                 info.get("edge_tile_count"), info.get("ng_tile_count"),
                 info.get("bundle_size_bytes"), 0, info.get("job_id"), info.get("notes")),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def list_model_bundles(self, machine_id: str = None) -> list:
        """列出 model_registry 紀錄，可選擇依 machine_id 過濾，依 trained_at DESC 排序。"""
        sql = "SELECT * FROM model_registry"
        args: tuple = ()
        if machine_id:
            sql += " WHERE machine_id = ?"
            args = (machine_id,)
        sql += " ORDER BY trained_at DESC"
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(sql, args)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            conn.close()

    def get_active_model_bundle(self) -> Optional[Dict]:
        """取得目前 active bundle；若資料異常有多筆，取最後啟用的一筆。"""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM model_registry WHERE is_active = 1 ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
        finally:
            conn.close()

    def get_model_bundle(self, bundle_id: int) -> Optional[Dict]:
        """依 id 查詢單筆 model_registry，找不到回傳 None。"""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM model_registry WHERE id = ?", (bundle_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
        finally:
            conn.close()

    def set_bundle_active(self, bundle_id: int, active: bool) -> None:
        """設定指定 bundle 的 is_active 狀態。"""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE model_registry SET is_active = ? WHERE id = ?",
                (1 if active else 0, bundle_id),
            )
            conn.commit()
        finally:
            conn.close()

    def update_model_bundle_notes(self, bundle_id: int, notes: str) -> bool:
        """更新指定 bundle 的使用者備註，回傳是否有更新到資料列。"""
        conn = self._get_conn()
        try:
            cur = conn.execute(
                "UPDATE model_registry SET notes = ? WHERE id = ?",
                (notes, bundle_id),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def list_auto_model_switch_rules(self) -> List[Dict]:
        """列出自動切換規則，附帶目前 bundle 顯示欄位。"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT r.*, b.machine_id, b.bundle_path, b.trained_at, b.is_active
                     FROM auto_model_switch_rules r
                     LEFT JOIN model_registry b ON b.id = r.bundle_id
                    ORDER BY CASE WHEN r.series_prefix = '__DEFAULT__' THEN 0 ELSE 1 END,
                             r.series_prefix"""
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_auto_model_switch_rule_by_series(self, series_prefix: str) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM auto_model_switch_rules WHERE series_prefix = ?",
                (series_prefix,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_default_auto_model_switch_rule(self) -> Optional[Dict]:
        return self.get_auto_model_switch_rule_by_series("__DEFAULT__")

    def upsert_auto_model_switch_rule(
        self,
        series_prefix: str,
        bundle_id: int,
        notes: str = "",
        rule_id: int = None,
    ) -> Dict:
        """新增或更新一筆自動切換規則。"""
        with self._lock:
            conn = self._get_conn()
            try:
                bundle = conn.execute(
                    "SELECT id FROM model_registry WHERE id = ?",
                    (bundle_id,),
                ).fetchone()
                if not bundle:
                    raise ValueError(f"bundle_id={bundle_id} 不存在")

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if rule_id:
                    cur = conn.execute(
                        """UPDATE auto_model_switch_rules
                              SET series_prefix = ?, bundle_id = ?, notes = ?, updated_at = ?
                            WHERE id = ?""",
                        (series_prefix, bundle_id, notes or "", now, rule_id),
                    )
                    if cur.rowcount == 0:
                        raise ValueError(f"auto_model_switch_rules id={rule_id} 不存在")
                    saved_id = rule_id
                else:
                    existing = conn.execute(
                        "SELECT id FROM auto_model_switch_rules WHERE series_prefix = ?",
                        (series_prefix,),
                    ).fetchone()
                    if existing:
                        saved_id = int(existing["id"])
                        conn.execute(
                            """UPDATE auto_model_switch_rules
                                  SET bundle_id = ?, notes = ?, updated_at = ?
                                WHERE id = ?""",
                            (bundle_id, notes or "", now, saved_id),
                        )
                    else:
                        cur = conn.execute(
                            """INSERT INTO auto_model_switch_rules
                               (series_prefix, bundle_id, notes, created_at, updated_at)
                               VALUES (?, ?, ?, ?, ?)""",
                            (series_prefix, bundle_id, notes or "", now, now),
                        )
                        saved_id = cur.lastrowid

                conn.commit()
                row = conn.execute(
                    """SELECT r.*, b.machine_id, b.bundle_path, b.trained_at, b.is_active
                         FROM auto_model_switch_rules r
                         LEFT JOIN model_registry b ON b.id = r.bundle_id
                        WHERE r.id = ?""",
                    (saved_id,),
                ).fetchone()
                return dict(row)
            except sqlite3.IntegrityError as e:
                conn.rollback()
                raise ValueError(f"自動切換規則不可重複: {series_prefix}") from e
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def delete_auto_model_switch_rule(self, rule_id: int) -> bool:
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    "DELETE FROM auto_model_switch_rules WHERE id = ?",
                    (rule_id,),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def add_auto_model_switch_history(self, entry: Dict) -> int:
        conn = self._get_conn()
        try:
            cur = conn.execute(
                """INSERT INTO auto_model_switch_history
                   (requested_model_id, series_prefix, previous_bundle_id,
                    previous_bundle_label, target_bundle_id, target_bundle_label,
                    action, status, message)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.get("requested_model_id", ""),
                    entry.get("series_prefix", ""),
                    entry.get("previous_bundle_id"),
                    entry.get("previous_bundle_label", ""),
                    entry.get("target_bundle_id"),
                    entry.get("target_bundle_label", ""),
                    entry.get("action", ""),
                    entry.get("status", ""),
                    entry.get("message", ""),
                ),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def list_auto_model_switch_history(
        self,
        limit: int = 100,
        series_prefix: str = "",
        status: str = "",
    ) -> List[Dict]:
        sql = "SELECT * FROM auto_model_switch_history"
        where = []
        args: List[Any] = []
        if series_prefix:
            where.append("series_prefix = ?")
            args.append(series_prefix)
        if status:
            where.append("status = ?")
            args.append(status)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY checked_at DESC, id DESC LIMIT ?"
        args.append(limit)

        conn = self._get_conn()
        try:
            rows = conn.execute(sql, tuple(args)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def deactivate_other_bundles_for_machine(self, machine_id: str, except_id: int) -> None:
        """將指定機種下除 except_id 外的所有 bundle 設為 is_active = 0。"""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE model_registry SET is_active = 0 WHERE machine_id = ? AND id != ?",
                (machine_id, except_id),
            )
            conn.commit()
        finally:
            conn.close()

    def deactivate_all_bundles(self, except_id: int) -> None:
        """將除 except_id 外的所有 bundle 設為 is_active = 0（跨 machine_id）。"""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE model_registry SET is_active = 0 WHERE id != ?",
                (except_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def list_ok_panels_for_machine(
        self,
        machine_id: str = "",
        days: int = 3,
        limit: int = 100,
        machine_id_prefix: str = "",
    ) -> list:
        """回傳近 N 天 machine_judgment='OK' 的 inference_records。

        供訓練 wizard 第一步選擇訓練樣本使用。
        machine_id 為空時回傳所有機種，供 UI 從最近推論紀錄直接挑選。
        machine_id_prefix 用於局部重訓：依料號前綴找同 family panel。
        """
        days = max(1, min(int(days or 3), 3))
        machine_id = str(machine_id or "").strip()
        machine_id_prefix = str(machine_id_prefix or "").strip()
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            params = []
            where = []
            if machine_id_prefix:
                where.append("substr(model_id, 1, ?) = ?")
                params.extend([len(machine_id_prefix), machine_id_prefix])
            elif machine_id:
                where.append("model_id = ?")
                params.append(machine_id)
            where.extend(["machine_judgment = 'OK'", "created_at >= datetime('now', ? || ' days')"])
            params.append(f"-{days}")
            params.append(limit)
            cur.execute(
                f"""SELECT id, glass_id, model_id, machine_no,
                           machine_judgment, ai_judgment, image_dir,
                           request_time, created_at
                    FROM inference_records
                    WHERE {' AND '.join(where)}
                    ORDER BY created_at DESC LIMIT ?""",
                params,
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            conn.close()

    def delete_model_bundle(self, bundle_id: int) -> None:
        """刪除指定 id 的 model_registry 紀錄。"""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM model_registry WHERE id = ?", (bundle_id,))
            conn.commit()
        finally:
            conn.close()


if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 60)
    print("CAPI Database Module Test")
    print("=" * 60)

    # 使用暫存檔測試
    test_db_path = os.path.join(tempfile.gettempdir(), "capi_test.db")
    db = CAPIDatabase(test_db_path)
    print(f"✅ Database created: {test_db_path}")

    # 測試寫入
    record_id = db.save_inference_record(
        glass_id="YPB21Y015A13",
        model_id="GN156HCAB6G0S",
        machine_no="CAPI1403",
        resolution=(1920, 1080),
        machine_judgment="OK",
        ai_judgment="NG@G0F00000(1024,512)",
        image_dir="/capi01/TIANMU/yuantu/GN116BCAA240S/20260225/T55BR592AE22",
        total_images=5,
        ng_images=1,
        ng_details='[{"image": "G0F00000.png", "tiles": [{"x": 1024, "y": 512, "score": 0.85}]}]',
        request_time="2026-02-25 17:00:00",
        response_time="2026-02-25 17:00:05",
        processing_seconds=5.23,
        heatmap_dir="/data/capi_ai/heatmaps/20260225/YPB21Y015A13",
        image_results_data=[
            {
                "image_path": "/capi01/test/G0F00000.png",
                "image_name": "G0F00000.png",
                "image_width": 6576,
                "image_height": 4384,
                "otsu_bounds": "20,20,6556,3384",
                "tile_count": 78,
                "excluded_tiles": 2,
                "anomaly_count": 1,
                "max_score": 0.85,
                "is_ng": 1,
                "is_dust_only": 0,
                "is_bomb": 0,
                "inference_time_ms": 4200.0,
                "heatmap_path": "/data/capi_ai/heatmaps/20260225/YPB21Y015A13/overview_G0F00000.png",
                "tiles": [
                    {
                        "tile_id": 15,
                        "x": 1024, "y": 512, "width": 512, "height": 512,
                        "score": 0.85, "is_anomaly": 1,
                        "is_dust": 0, "dust_iou": 0.0,
                        "is_bomb": 0, "bomb_code": "",
                        "peak_x": 1280, "peak_y": 768,
                        "heatmap_path": "/data/capi_ai/heatmaps/20260225/YPB21Y015A13/heatmap_G0F00000_tile15.png"
                    }
                ]
            }
        ]
    )
    print(f"✅ Record saved, ID: {record_id}")

    # 測試查詢
    records = db.query_by_glass_id("YPB21Y015A13")
    print(f"✅ Query by glass_id: {len(records)} records found")

    # 測試詳細查詢
    detail = db.get_record_detail(record_id)
    print(f"✅ Record detail: {detail['glass_id']} / AI={detail['ai_judgment']}")
    print(f"   Images: {len(detail['images'])}")
    if detail['images']:
        img = detail['images'][0]
        print(f"   - {img['image_name']}: tiles={img['tile_count']}, NG={img['is_ng']}")
        print(f"     Anomaly tiles: {len(img['tiles'])}")

    # 測試統計
    stats = db.get_statistics(days=30)
    print(f"✅ Statistics: total={stats['total_records']}")

    # 測試最近記錄
    recent = db.query_recent(10)
    print(f"✅ Recent records: {len(recent)}")

    # 測試搜尋
    results, total = db.search_records(machine_no="CAPI1403")
    print(f"✅ Search by machine: {len(results)} results (total: {total})")

    # 清理
    os.remove(test_db_path)
    print(f"\n✅ All tests passed! Test DB removed.")
