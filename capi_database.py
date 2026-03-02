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
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple


class CAPIDatabase:
    """CAPI AI 推論結果 SQLite 資料庫"""

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

                -- 索引
                CREATE INDEX IF NOT EXISTS idx_records_glass_id ON inference_records(glass_id);
                CREATE INDEX IF NOT EXISTS idx_records_created_at ON inference_records(created_at);
                CREATE INDEX IF NOT EXISTS idx_records_machine_no ON inference_records(machine_no);
                CREATE INDEX IF NOT EXISTS idx_records_ai_judgment ON inference_records(ai_judgment);
                CREATE INDEX IF NOT EXISTS idx_image_results_record_id ON image_results(record_id);
                CREATE INDEX IF NOT EXISTS idx_tile_results_image_id ON tile_results(image_result_id);
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
        image_results_data: Optional[List[Dict]] = None,
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
                        heatmap_dir, error_message)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (glass_id, model_id, machine_no, resolution[0], resolution[1],
                     machine_judgment, ai_judgment, image_dir, total_images, ng_images,
                     ng_details, request_time, response_time, processing_seconds,
                     heatmap_dir, error_message)
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
                                heatmap_path)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                             img_data.get("heatmap_path", ""))
                        )
                        image_result_id = img_cursor.lastrowid

                        # 儲存 tile 級結果
                        for tile_data in img_data.get("tiles", []):
                            conn.execute(
                                """INSERT INTO tile_results
                                   (image_result_id, tile_id, x, y, width, height,
                                    score, is_anomaly, is_dust, dust_iou, is_bomb,
                                    bomb_code, peak_x, peak_y, heatmap_path)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                                 tile_data.get("heatmap_path", ""))
                            )

                conn.commit()
                return record_id
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
                # 取得 tile 結果
                tiles = conn.execute(
                    """SELECT * FROM tile_results
                       WHERE image_result_id = ?
                       ORDER BY tile_id""",
                    (img_dict["id"],)
                ).fetchall()
                img_dict["tiles"] = [dict(t) for t in tiles]
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
                     AVG(processing_seconds) as avg_time
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
        limit: int = 100,
    ) -> List[Dict]:
        """多條件搜尋"""
        conditions = []
        params = []

        if glass_id:
            conditions.append("glass_id LIKE ?")
            params.append(f"%{glass_id}%")
        if machine_no:
            conditions.append("machine_no LIKE ?")
            params.append(f"%{machine_no}%")
        if ai_judgment:
            conditions.append("ai_judgment LIKE ?")
            params.append(f"%{ai_judgment}%")
        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        conn = self._get_conn()
        try:
            rows = conn.execute(
                f"""SELECT * FROM inference_records
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ?""",
                params
            ).fetchall()
            return [dict(r) for r in rows]
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
    results = db.search_records(machine_no="CAPI1403")
    print(f"✅ Search by machine: {len(results)} results")

    # 清理
    os.remove(test_db_path)
    print(f"\n✅ All tests passed! Test DB removed.")
