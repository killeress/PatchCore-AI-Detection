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
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


class CAPIDatabase:
    """CAPI AI 推論結果 SQLite 資料庫"""

    # 共用 SQL 條件片段（search_records 與 get_inference_stats 共用）
    _AOI_NG_COND = "machine_judgment != '' AND machine_judgment != 'OK'"
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
                    processing_seconds REAL DEFAULT 0.0,
                    heatmap_dir TEXT DEFAULT '',
                    error_message TEXT DEFAULT '',
                    client_bomb_info TEXT DEFAULT '',
                    aoi_machine_coords TEXT DEFAULT '',
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
                CREATE INDEX IF NOT EXISTS idx_records_machine_no ON inference_records(machine_no);
                CREATE INDEX IF NOT EXISTS idx_records_ai_judgment ON inference_records(ai_judgment);
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
                    changed_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                CREATE INDEX IF NOT EXISTS idx_config_param_name ON config_params(param_name);
                CREATE INDEX IF NOT EXISTS idx_config_history_param ON config_change_history(param_name);
                CREATE INDEX IF NOT EXISTS idx_config_history_time ON config_change_history(changed_at);

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
                CREATE TABLE IF NOT EXISTS training_jobs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id          TEXT UNIQUE,
                    machine_id      TEXT NOT NULL,
                    state           TEXT NOT NULL,
                    started_at      TEXT,
                    completed_at    TEXT,
                    panel_paths     TEXT,
                    output_bundle   TEXT,
                    error_message   TEXT,
                    training_params TEXT
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
            """)
            
            # Migration for adding missing columns to existing database
            def add_column_if_not_exists(table, column, def_type):
                cursor = conn.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in cursor.fetchall()]
                if column not in columns:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {def_type}")

            add_column_if_not_exists("inference_records", "error_message", "TEXT DEFAULT ''")
            add_column_if_not_exists("inference_records", "client_bomb_info", "TEXT DEFAULT ''")
            add_column_if_not_exists("inference_records", "aoi_machine_coords", "TEXT DEFAULT ''")
            add_column_if_not_exists("image_results", "is_bomb", "INTEGER DEFAULT 0")
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
            add_column_if_not_exists("inference_records", "inference_log", "TEXT DEFAULT ''")
            add_column_if_not_exists("inference_records", "omit_overexposed", "INTEGER DEFAULT 0")
            add_column_if_not_exists("inference_records", "omit_overexposure_info", "TEXT DEFAULT ''")
            # Scratch classifier post-filter (over-review reduction)
            add_column_if_not_exists("tile_results", "scratch_score", "REAL DEFAULT 0.0")
            add_column_if_not_exists("tile_results", "scratch_filtered", "INTEGER DEFAULT 0")
            add_column_if_not_exists("image_results", "scratch_filter_count", "INTEGER DEFAULT 0")
            add_column_if_not_exists("training_jobs", "training_params", "TEXT")

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
        aoi_machine_coords: str = "",
        image_results_data: Optional[List[Dict]] = None,
        inference_log: str = "",
        omit_overexposed: int = 0,
        omit_overexposure_info: str = "",
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
            aoi_machine_coords: AOI 機台檢測座標 (TXT 報告解析, JSON 字串)
            image_results_data: 圖片級結果列表

        Returns:
            record_id
        """
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    """INSERT INTO inference_records
                       (glass_id, model_id, machine_no, resolution_x, resolution_y,
                        machine_judgment, ai_judgment, image_dir, total_images, ng_images,
                        ng_details, request_time, response_time, processing_seconds,
                        heatmap_dir, error_message, client_bomb_info, aoi_machine_coords,
                        inference_log, omit_overexposed, omit_overexposure_info)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (glass_id, model_id, machine_no, resolution[0], resolution[1],
                     machine_judgment, ai_judgment, image_dir, total_images, ng_images,
                     ng_details, request_time, response_time, processing_seconds,
                     heatmap_dir, error_message, client_bomb_info, aoi_machine_coords,
                     inference_log, omit_overexposed, omit_overexposure_info)
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
                                heatmap_path, scratch_filter_count)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                             img_data.get("scratch_filter_count", 0))
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
                                    scratch_score, scratch_filtered)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                                 tile_data.get("scratch_score", 0.0),
                                 int(tile_data.get("scratch_filtered", 0)))
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
        image_results_data: Optional[List[Dict]] = None,
        inference_log: str = "",
        omit_overexposed: int = 0,
        omit_overexposure_info: str = "",
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
                cursor = conn.execute(
                    """UPDATE inference_records SET
                           ai_judgment = ?,
                           total_images = ?,
                           ng_images = ?,
                           ng_details = ?,
                           response_time = ?,
                           processing_seconds = ?,
                           heatmap_dir = ?,
                           error_message = ?,
                           inference_log = ?,
                           omit_overexposed = ?,
                           omit_overexposure_info = ?
                       WHERE id = ?""",
                    (ai_judgment, total_images, ng_images, ng_details,
                     now_str, processing_seconds, heatmap_dir, error_message,
                     inference_log, omit_overexposed, omit_overexposure_info,
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
                                heatmap_path, scratch_filter_count)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                             img_data.get("scratch_filter_count", 0))
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
                                    scratch_score, scratch_filtered)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                                 tile_data.get("scratch_score", 0.0),
                                 int(tile_data.get("scratch_filtered", 0)))
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
                     SUM(CASE WHEN ai_judgment = 'OK' THEN 1 ELSE 0 END) as ok_count,
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
                     SUM(CASE WHEN ai_judgment = 'OK' THEN 1 ELSE 0 END) as ok_count,
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

    def get_shift_statistics(self) -> Dict:
        """取得當班統計（白班 08:00~19:59 / 夜班 20:00~07:59）"""
        from datetime import timedelta
        now = datetime.now()
        hour = now.hour

        if 8 <= hour < 20:
            # 白班：當日 08:00 ~ 19:59
            shift_name = "白班"
            shift_start = now.replace(hour=8, minute=0, second=0, microsecond=0)
            shift_end = now.replace(hour=19, minute=59, second=59, microsecond=0)
        else:
            # 夜班：20:00 ~ 隔日 07:59
            shift_name = "夜班"
            if hour >= 20:
                shift_start = now.replace(hour=20, minute=0, second=0, microsecond=0)
                shift_end = (now + timedelta(days=1)).replace(hour=7, minute=59, second=59, microsecond=0)
            else:
                # 凌晨 0~7 點，班次起點是前一天 20:00
                shift_start = (now - timedelta(days=1)).replace(hour=20, minute=0, second=0, microsecond=0)
                shift_end = now.replace(hour=7, minute=59, second=59, microsecond=0)

        start_str = shift_start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = shift_end.strftime("%Y-%m-%d %H:%M:%S")
        time_range_label = f"{shift_start.strftime('%m/%d %H:%M')} ~ {shift_end.strftime('%m/%d %H:%M')}"

        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT
                     COUNT(*) as total,
                     SUM(CASE WHEN ai_judgment = 'OK' THEN 1 ELSE 0 END) as ok_count,
                     SUM(CASE WHEN ai_judgment = 'NG' OR ai_judgment LIKE 'NG%' THEN 1 ELSE 0 END) as ng_count,
                     SUM(CASE WHEN ai_judgment LIKE 'ERR%' THEN 1 ELSE 0 END) as err_count,
                     AVG(processing_seconds) as avg_time,
                     SUM(omit_overexposed) as overexposed_count
                   FROM inference_records
                   WHERE created_at >= ? AND created_at <= ?""",
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

        if cross_filter:
            # 使用 DATE(request_time) 與 get_inference_stats 一致，確保數字對得上
            if start_date:
                conditions.append("DATE(request_time) >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("DATE(request_time) <= ?")
                params.append(end_date)
        else:
            if start_date:
                conditions.append("created_at >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("created_at <= ?")
                params.append(end_date)

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
                where_clauses.append("DATE(c.time_stamp) >= ?")
                params.append(start_date)
            if end_date:
                where_clauses.append("DATE(c.time_stamp) <= ?")
                params.append(end_date)
            where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

            rows = conn.execute(
                f"""SELECT c.id, c.time_stamp, c.pnl_id, c.mach_id,
                           c.result_eqp, c.result_ai, c.result_ric, c.datastr,
                           mr.id as review_id, mr.category as review_category,
                           mr.note as review_note, mr.updated_at as review_updated_at,
                           ovr.id as over_review_id, ovr.category as over_review_category,
                           ovr.note as over_review_note, ovr.updated_at as over_review_updated_at,
                           (SELECT ir.id FROM inference_records ir
                            WHERE ir.glass_id = c.pnl_id
                              AND DATE(ir.request_time) = DATE(c.time_stamp)
                            ORDER BY ir.request_time DESC LIMIT 1
                           ) as inference_record_id
                    FROM client_accuracy_records c
                    LEFT JOIN miss_review mr ON mr.client_record_id = c.id
                    LEFT JOIN over_review ovr ON ovr.client_record_id = c.id
                    {where_sql}
                    ORDER BY c.time_stamp DESC""",
                params
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_dust_affected_record_ids(self, record_ids: list) -> set:
        """返回有灰塵過濾影響的 inference record IDs (is_dust_only 或 edge is_dust)"""
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
                    f"FROM edge_defect_results edr "
                    f"JOIN image_results img ON img.id = edr.image_result_id "
                    f"WHERE img.record_id IN ({ph}) AND img.is_ng = 0 AND edr.is_dust = 1",
                    chunk + chunk
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

    VALID_MISS_CATEGORIES = {'dust_misfilter', 'threshold_high', 'ric_misjudge', 'outside_aoi_area', 'data_error_actually_ok', 'other'}
    VALID_OVER_CATEGORIES = {'edge_false_positive', 'within_spec', 'overexposure', 'surface_scratch', 'surface_dirt', 'bubble', 'aoi_ai_false_positive', 'dust_mask_incomplete', 'other'}

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
        }

        # Step 0: 清除過期 heatmap 目錄 (獨立天數，在 DB 記錄刪除前先查詢)
        heatmap_dirs_to_delete = []
        if heatmap_retain_days > 0:
            hm_cutoff = (now - timedelta(days=heatmap_retain_days)).strftime('%Y-%m-%d')
            with self._lock:
                conn = self._get_conn()
                try:
                    rows = conn.execute("""
                        SELECT heatmap_dir FROM inference_records
                        WHERE heatmap_dir != '' AND created_at < ?
                    """, (hm_cutoff,)).fetchall()
                    heatmap_dirs_to_delete = [r[0] for r in rows if r[0]]
                finally:
                    conn.close()

        with self._lock:
            conn = self._get_conn()
            try:
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
                    WHERE (ai_judgment = 'OK' AND created_at < ?)
                       OR (ai_judgment != 'OK'  AND created_at < ?)
                """, (ok_cutoff, ng_cutoff))
                stats["inference_records_deleted"] = cur.rowcount

                # Step 2.5: 清除已刪記錄的 heatmap_dir 欄位 (圖檔已刪但記錄還在的情況)
                if heatmap_retain_days > 0:
                    hm_cutoff = (now - timedelta(days=heatmap_retain_days)).strftime('%Y-%m-%d')
                    conn.execute("""
                        UPDATE inference_records
                        SET heatmap_dir = ''
                        WHERE heatmap_dir != '' AND created_at < ?
                    """, (hm_cutoff,))

                conn.commit()
            except Exception as e:
                conn.rollback()
                raise
            finally:
                conn.close()

        # Step 3: 刪除實體 heatmap 目錄 (在鎖外執行，避免 IO 阻塞)
        for d in heatmap_dirs_to_delete:
            try:
                p = Path(d)
                if p.is_dir():
                    shutil.rmtree(p)
                    stats["heatmap_dirs_deleted"] += 1
            except Exception:
                pass  # 目錄不存在或權限問題，跳過

        # Step 4: VACUUM (在鎖外，不阻塞其他操作)
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
            ai_j = "OK" if rec["ai_judgment"] == "OK" else "NG"
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
                
        aoi_over_rate = round((aoi_over / total * 100), 1) if total > 0 else 0
        aoi_miss_rate = round((aoi_miss / total * 100), 1) if total > 0 else 0

        # AI 過檢率 / 漏檢率
        ai_over_rate = round(ai_over / total * 100, 1) if total > 0 else 0
        ai_miss_rate = round(ai_miss / total * 100, 1) if total > 0 else 0

        by_day = []
        for date_str in sorted(day_stats.keys()):
            s = day_stats[date_str]
            by_day.append({
                "date": date_str,
                "total": s["total"],
                "aoi_acc": round(s["aoi_correct"] / s["total"] * 100, 1) if s["total"] > 0 else 0,
                "ai_acc": round(s["ai_correct"] / s["total"] * 100, 1) if s["total"] > 0 else 0,
            })

        by_machine = []
        for machine in sorted(mach_stats.keys()):
            s = mach_stats[machine]
            by_machine.append({
                "machine": machine,
                "total": s["total"],
                "aoi_acc": round(s["aoi_correct"] / s["total"] * 100, 1) if s["total"] > 0 else 0,
                "ai_acc": round(s["ai_correct"] / s["total"] * 100, 1) if s["total"] > 0 else 0,
            })

        return {
            "total": total,
            "aoi_accuracy": round(aoi_accuracy, 1),
            "ai_accuracy": round(ai_accuracy, 1),
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
                where_clauses.append("DATE(request_time) >= ?")
                params.append(start_date)
            if end_date:
                where_clauses.append("DATE(request_time) <= ?")
                params.append(end_date)
            where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

            _aoi_ng = self._AOI_NG_COND
            _ai_ng = self._AI_NG_COND
            _err = self._ERR_COND

            # ── 1. Summary + Cross Matrix (single SQL aggregate) ──
            summary_row = conn.execute(
                f"""SELECT COUNT(*) as total,
                           SUM(CASE WHEN {_aoi_ng} THEN 1 ELSE 0 END) as aoi_ng,
                           SUM(CASE WHEN {_ai_ng} THEN 1 ELSE 0 END) as ai_ng,
                           SUM(CASE WHEN ({_aoi_ng}) AND ai_judgment = 'OK' THEN 1 ELSE 0 END) as ai_revival,
                           SUM(CASE WHEN {_err} THEN 1 ELSE 0 END) as err_count,
                           SUM(CASE WHEN NOT ({_aoi_ng}) AND NOT ({_ai_ng}) AND NOT ({_err}) THEN 1 ELSE 0 END) as ok_ok,
                           SUM(CASE WHEN ({_aoi_ng}) AND NOT ({_ai_ng}) AND NOT ({_err}) THEN 1 ELSE 0 END) as ng_ok,
                           SUM(CASE WHEN NOT ({_aoi_ng}) AND ({_ai_ng}) AND NOT ({_err}) THEN 1 ELSE 0 END) as ok_ng,
                           SUM(CASE WHEN ({_aoi_ng}) AND ({_ai_ng}) AND NOT ({_err}) THEN 1 ELSE 0 END) as ng_ng
                    FROM inference_records{where_sql}""",
                params
            ).fetchone()
            s = dict(summary_row)
            total = s["total"] or 0

            # ── 2. 每日趨勢 ──
            daily_rows = conn.execute(
                f"""SELECT DATE(request_time) as date,
                           COUNT(*) as total,
                           SUM(CASE WHEN {_aoi_ng} THEN 1 ELSE 0 END) as aoi_ng,
                           SUM(CASE WHEN {_ai_ng} THEN 1 ELSE 0 END) as ai_ng,
                           SUM(CASE WHEN {_err} THEN 1 ELSE 0 END) as err
                    FROM inference_records{where_sql}
                    GROUP BY DATE(request_time)
                    ORDER BY date""",
                params
            ).fetchall()
            daily_trend = [dict(r) for r in daily_rows]

            # ── 3. 機台統計 ──
            machine_rows = conn.execute(
                f"""SELECT machine_no as machine,
                           COUNT(*) as total,
                           ROUND(SUM(CASE WHEN {_aoi_ng} THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as aoi_ng_rate,
                           ROUND(SUM(CASE WHEN {_ai_ng} THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as ai_ng_rate
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
            err_where = f" WHERE ai_judgment LIKE 'ERR%'" if not where_clauses else where_sql + f" AND {_err}"
            err_rows = conn.execute(
                f"""SELECT CASE WHEN LENGTH(TRIM(SUBSTR(ai_judgment, 5))) > 0
                                THEN TRIM(SUBSTR(ai_judgment, 5))
                                WHEN LENGTH(TRIM(error_message)) > 0
                                THEN TRIM(error_message)
                                ELSE 'Unknown'
                           END as type,
                           COUNT(*) as count
                    FROM inference_records{err_where}
                    GROUP BY type
                    ORDER BY count DESC""",
                params
            ).fetchall()
            err_types = [dict(r) for r in err_rows]

            return {
                "success": True,
                "summary": {
                    "total": total,
                    "aoi_ng": s["aoi_ng"] or 0,
                    "ai_ng": s["ai_ng"] or 0,
                    "ai_revival": s["ai_revival"] or 0,
                    "err_count": s["err_count"] or 0,
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

    # ── 設定參數管理方法 ─────────────────────────────────

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
        self, param_name: str, new_value: Any, reason: str = ""
    ) -> bool:
        """
        更新設定參數並記錄修改歷史

        Args:
            param_name: 參數名稱
            new_value: 新值 (Python 原生型別)
            reason: 修改原因

        Returns:
            是否更新成功
        """
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

                if not old_row:
                    # 參數不存在於 DB → 自動新增 (從 config dataclass 補上的參數)
                    if isinstance(new_value, bool):
                        param_type = "bool"
                    elif isinstance(new_value, int):
                        param_type = "int"
                    elif isinstance(new_value, float):
                        param_type = "float"
                    else:
                        param_type = "str"
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

                    # 更新設定值
                    conn.execute(
                        "UPDATE config_params SET param_value = ?, updated_at = ? WHERE param_name = ?",
                        (new_value_json, now, param_name)
                    )

                # 記錄修改歷史
                conn.execute(
                    """INSERT INTO config_change_history
                       (param_name, old_value, new_value, change_reason, changed_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (param_name, old_value, new_value_json, reason, now)
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
            ("model_mapping", config.model_mapping, "dict", "前綴 → 模型路徑映射"),
            ("threshold_mapping", config.threshold_mapping, "dict", "前綴 → 獨立閾值映射"),
            ("patchcore_filter_enabled", config.patchcore_filter_enabled, "bool", "啟用 PatchCore 後處理進階過濾"),
            ("patchcore_blur_sigma", config.patchcore_blur_sigma, "float", "異常圖高斯平滑強度 (抑制噪點)"),
            ("patchcore_min_area", config.patchcore_min_area, "int", "異常判定最小連通面積(px)"),
            ("patchcore_score_metric", config.patchcore_score_metric, "string", "計分方式 (max, top_k_avg, percentile_99)"),
            ("dust_brightness_threshold", config.dust_brightness_threshold, "int", "灰塵亮度閾值 (備用)"),
            ("dust_area_min", config.dust_area_min, "int", "灰塵顆粒最小面積 (px)"),
            ("dust_area_max", config.dust_area_max, "int", "灰塵顆粒最大面積 (px)"),
            ("dust_extension", config.dust_extension, "int", "灰塵區域膨脹像素"),
            ("dust_heatmap_iou_threshold", config.dust_heatmap_iou_threshold, "float", "Heatmap-Dust IOU/Coverage 閾值"),
            ("dust_heatmap_top_percent", config.dust_heatmap_top_percent, "float", "Heatmap 熱區取前 X%"),
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
            # AOI 機檢座標設定
            ("grid_tiling_enabled", True, "bool", "啟用全面板 Grid Tiling 推論"),
            ("aoi_coord_inspection_enabled", False, "bool", "啟用 AOI 機檢座標推論"),
            ("aoi_report_path_replace_from", "yuantu", "string", "報告路徑替換來源字串"),
            ("aoi_report_path_replace_to", "Report", "string", "報告路徑替換目標字串"),
        ]

        count = 0
        with self._lock:
            conn = self._get_conn()
            try:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for name, value, ptype, desc in params_def:
                    # 只在 DB 中尚無此參數時才新增
                    existing = conn.execute(
                        "SELECT id FROM config_params WHERE param_name = ?",
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
    ) -> int:
        """建立一筆新的訓練 job，初始 state 為 'preprocess'。回傳 rowid。

        training_params 為 step1 使用者覆寫的 PatchCore 超參數（JSON 序列化後寫入），
        None 表示完全使用 TrainingConfig 的 dataclass 預設值。
        """
        # 用 `is not None` 而非 falsy；空 dict 與 None 語意不同（前者代表
        # 來源已驗證但無覆寫項，留給呼叫端決定是否要寫入）。
        params_json = json.dumps(training_params) if training_params is not None else None
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO training_jobs
                   (job_id, machine_id, state, started_at, panel_paths, training_params)
                   VALUES (?, ?, 'preprocess', datetime('now'), ?, ?)""",
                (job_id, machine_id, json.dumps(panel_paths), params_json),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_training_job(self, job_id: str) -> Optional[Dict]:
        """依 job_id 查詢訓練 job，panel_paths / training_params 自動 JSON 反序列化。找不到回傳 None。"""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM training_jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            job = dict(zip(cols, row))
            if job.get("panel_paths"):
                job["panel_paths"] = json.loads(job["panel_paths"])
            else:
                job["panel_paths"] = []
            raw_params = job.get("training_params")
            job["training_params"] = json.loads(raw_params) if raw_params else None
            return job
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
            job = dict(zip(cols, row))
            if job.get("panel_paths"):
                job["panel_paths"] = json.loads(job["panel_paths"])
            else:
                job["panel_paths"] = []
            raw_params = job.get("training_params")
            job["training_params"] = json.loads(raw_params) if raw_params else None
            return job
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

    def list_ok_panels_for_machine(self, machine_id: str = "", days: int = 3, limit: int = 100) -> list:
        """回傳近 N 天 machine_judgment='OK' 的 inference_records。

        供訓練 wizard 第一步選擇訓練樣本使用。
        machine_id 為空時回傳所有機種，供 UI 從最近推論紀錄直接挑選。
        """
        days = max(1, min(int(days or 3), 3))
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            params = []
            where = ["machine_judgment = 'OK'", "created_at >= datetime('now', ? || ' days')"]
            params.append(f"-{days}")
            if machine_id:
                where.insert(0, "model_id = ?")
                params.insert(0, machine_id)
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
