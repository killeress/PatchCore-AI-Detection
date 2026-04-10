"""
CAPI AI Web 查閱介面

提供推論結果的 Web 查閱功能，包含：
- 推論記錄列表 (首頁)
- 記錄詳情 (含熱力圖)
- 搜尋功能
- 統計 API
- 熱力圖靜態檔案服務

使用 Python 內建 http.server + 簡單路由，無需額外依賴。
"""

import os
import tempfile
import json
import urllib.parse
import mimetypes
from datetime import datetime, timedelta
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, Tuple
import threading
import logging
from jinja2 import Environment, FileSystemLoader
import shutil
from capi_dataset_export import (
    DatasetExporter, JobSummary,
    JOB_STATE_IDLE, JOB_STATE_RUNNING, JOB_STATE_COMPLETED,
    JOB_STATE_FAILED, JOB_STATE_CANCELLED,
)

logger = logging.getLogger("capi.web")

# 幫 Jinja2 準備一些好用的過濾器
def ai_simple(ai_judgment):
    if not ai_judgment: return ""
    return "OK" if ai_judgment == "OK" else ("NG" if ai_judgment.startswith("NG") else ("ERR" if ai_judgment.startswith("ERR") else ai_judgment))

def ai_badge(ai_judgment):
    simple = ai_simple(ai_judgment)
    return "badge-ok" if simple == "OK" else ("badge-ng" if simple == "NG" else "badge-err")

def mj_badge(machine_judgment):
    if machine_judgment == "OK":
        return "badge-ok"
    elif machine_judgment == "HY":
        return "badge-err"
    else:
        return "badge-ng"

def img_status_info(img):
    if img.get("is_dust_only"): return "灰塵 (DUST)", "badge-err"
    if img.get("is_bomb"): return "炸彈 (BOMB)", "badge-err"
    if img["is_ng"]: return "NG", "badge-ng"
    return "OK", "badge-ok"

def tile_info(t):
    badge = "badge-ng"
    info = f"Score: {t['score']:.3f}"
    if t.get("is_aoi_coord"):
        code = t.get('aoi_defect_code', '')
        if t.get("is_anomaly"):
            info += f" | 🎯 AOI座標 ({code}) AI也判NG"
        else:
            badge = "badge-ok"
            info += f" | 🎯 AOI座標 ({code}) AI判OK"
    if t.get("is_exclude_zone"):
        badge = "badge-ok"
        info += " | 不檢測排除區域"
    elif t.get("is_dust"):
        badge = "badge-err"
        info += f" | 灰塵 Region COV: {t.get('dust_iou',0):.3f}"
    if t.get("is_bomb"):
        badge = "badge-err"
        info += f" | 炸彈代碼: {t.get('bomb_code','')}"
    return badge, info

def get_img_stem(img):
    img_path_str = img.get("image_path", "")
    img_stem = Path(img_path_str).stem if img_path_str else ""
    raw_name = img.get("image_name", "")
    if raw_name.startswith("overview_"):
        img_stem = raw_name.replace("overview_", "").replace(".png", "").replace(".jpg", "")
    return img_stem or raw_name

def hm_relative(path_str, base_dir):
    if not path_str or not base_dir: return ""
    try:
        rel = Path(path_str).relative_to(base_dir)
        return rel.as_posix()
    except (ValueError, TypeError):
        return ""


class CAPIWebHandler(BaseHTTPRequestHandler):
    """CAPI Web 請求處理器"""

    # 類別變數，由 create_web_server 設定
    db = None
    # Jinja2 環境
    jinja_env = None
    # Debug 推論用 (可選)
    inferencer = None
    heatmap_manager = None
    _debug_heatmap_dir = None  # Debug 推論暫存目錄
    _capi_server_instance = None  # CAPIServer 實例 (用於 hot-reload)
    _log_file = None  # 日誌檔案路徑 (用於 Log Viewer)

    @classmethod
    def init_jinja(cls):
        if cls.jinja_env is None:
            templates_dir = Path(__file__).parent / "templates"
            cls.jinja_env = Environment(loader=FileSystemLoader(templates_dir))
            cls.jinja_env.filters['ai_simple'] = ai_simple
            cls.jinja_env.filters['ai_badge'] = ai_badge
            cls.jinja_env.filters['mj_badge'] = mj_badge
            cls.jinja_env.filters['img_status_info'] = img_status_info
            cls.jinja_env.filters['tile_info'] = tile_info
            cls.jinja_env.filters['get_img_stem'] = get_img_stem
            cls.jinja_env.filters['fromjson'] = lambda s: json.loads(s) if s else {}
            cls.jinja_env.globals['hm_relative'] = hm_relative
            
    def log_message(self, format, *args):
        """靜默 Web HTTP 存取日誌，避免污染 server.log 與 CMD"""
        pass

    def do_GET(self):
        """處理 GET 請求"""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if path == "/" or path == "":
                self._handle_index(query, path)
            elif path == "/dashboard":
                self._handle_dashboard(query, path)
            elif path == "/dashboard_v2":
                self._handle_dashboard_v2(query, path)
            elif path == "/v3/dashboard":
                self._handle_dashboard_v3(query, path)
            elif path.startswith("/api/record/") and path.endswith("/rerun/status"):
                record_id_str = path.split("/api/record/")[1].split("/rerun/status")[0]
                self._handle_rerun_status_sse(record_id_str)
            elif path.startswith("/v3/record/"):
                record_id = path.split("/v3/record/")[1].rstrip("/")
                self._handle_record_detail_v3(record_id, path)
            elif path.startswith("/record/"):
                record_id = path.split("/record/")[1].rstrip("/")
                self._handle_record_detail(record_id, path)
            elif path == "/overexposed":
                self._handle_overexposed(query, path)
            elif path == "/search":
                self._handle_search(query, path)
            elif path == "/search/export":
                self._handle_search_export(query)
            elif path == "/logs":
                self._handle_logs_page(query, path)
            elif path == "/api/logs":
                self._handle_api_logs(query)
            elif path == "/debug":
                self._handle_debug_page(path)
            elif path == "/ric":
                self._handle_ric_page(query, path)
            elif path == "/api/ric/report":
                self._handle_ric_report_api(query)
            elif path == "/api/ric/client-data":
                self._handle_client_data_api(query)
            elif path == "/api/ric/inference-stats":
                self._handle_inference_stats_api(query)
            elif path == "/api/stats":
                self._handle_api_stats(query)
            elif path == "/api/status":
                self._handle_api_status()
            elif path == "/settings":
                self._handle_settings_page(path)
            elif path == "/settings_v2":
                self._handle_settings_v2_page(path)
            elif path == "/api/settings":
                self._handle_api_settings()
            elif path == "/api/settings/history":
                self._handle_api_settings_history(query)
            elif path.startswith("/heatmaps/"):
                self._handle_static_file(path)
            elif path.startswith("/debug/heatmaps/"):
                self._handle_debug_heatmap_file(path)
            elif path == "/api/debug/serve-image":
                self._handle_debug_serve_image(query)
            elif path.startswith("/images/"):
                self._handle_source_image(path)
            elif path.startswith("/imgs/"):
                self._handle_imgs_file(path)
            elif path.startswith("/static/"):
                self._handle_static_assets(path)
            elif path == "/api/dataset_export/status":
                self._handle_dataset_export_status()
                return
            elif path.startswith("/api/dataset_export/summary/"):
                job_id = path.split("/api/dataset_export/summary/", 1)[1]
                self._handle_dataset_export_summary(job_id)
                return
            elif path == "/favicon.ico":
                self._send_response(204, "")
            else:
                self._send_404(path)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # 客戶端已斷線，靜默忽略即可
            pass
        except Exception as e:
            logger.error(f"Error handling {path}: {e}", exc_info=True)
            try:
                self._send_error(500, str(e))
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    def do_POST(self):
        """處理 POST 請求"""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        try:
            if path == "/api/debug/inference":
                self._handle_debug_inference_run()
            elif path == "/api/debug/coord-inference":
                self._handle_debug_coord_inference()
            elif path == "/api/debug/edge-inspect":
                self._handle_api_debug_edge_inspect()
            elif path == "/api/debug/bright-spot-inference":
                self._handle_debug_bright_spot_inference()
            elif path == "/api/ric/upload":
                self._handle_ric_upload()
            elif path == "/api/ric/delete":
                self._handle_ric_delete()
            elif path == "/api/ric/import-client":
                self._handle_client_import()
            elif path == "/api/ric/clear-client":
                self._handle_client_clear()
            elif path == "/api/ric/miss-review":
                self._handle_miss_review_save()
            elif path == "/api/ric/miss-review/delete":
                self._handle_miss_review_delete()
            elif path == "/api/ric/over-review":
                self._handle_over_review_save()
            elif path == "/api/ric/over-review/delete":
                self._handle_over_review_delete()
            elif path == "/api/settings/update":
                self._handle_api_settings_update()
            elif path == "/api/settings/reload":
                self._handle_api_settings_reload()
            elif path.startswith("/api/record/") and path.endswith("/rerun"):
                record_id_str = path.split("/api/record/")[1].split("/rerun")[0]
                self._handle_rerun_trigger(record_id_str)
            elif path == "/api/dataset_export/start":
                self._handle_dataset_export_start()
                return
            elif path == "/api/dataset_export/cancel":
                self._handle_dataset_export_cancel()
                return
            else:
                self._send_404(path)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            logger.error(f"Error handling POST {path}: {e}", exc_info=True)
            try:
                self._send_json({"error": str(e)})
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    def _send_response(self, code: int, content: str, content_type: str = "text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def _send_json(self, data, status=200):
        content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        content_bytes = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content_bytes)))
        self.end_headers()
        self.wfile.write(content_bytes)

    def _send_404(self, path=""):
        self._send_error(404, "Page Not Found", path)

    def _send_error(self, code: int, message: str, path=""):
        html = """
        {% extends "base.html" %}
        {% block content %}
        <div class="card" style="border-color:var(--err)">
            <h2 style="color:var(--err)">Error """ + str(code) + """</h2>
            <p>""" + message + """</p>
        </div>
        {% endblock %}
        """
        try:
            template = self.jinja_env.from_string(html)
            rendered = template.render(request_path=path)
        except Exception:
            rendered = f"<h2>Error {code}</h2><p>{message}</p>"
        self._send_response(code, rendered)

    def _send_binary(self, filepath: str):
        """發送二進位檔案 (圖片等)"""
        path = Path(filepath)
        if not path.exists():
            self._send_404()
            return
        mime_type, _ = mimetypes.guess_type(str(path))
        
        # Fallback for .mjs if not in standard mimetypes
        if mime_type is None and str(path).endswith(".mjs"):
            mime_type = "application/javascript"
            
        if mime_type is None:
            mime_type = "application/octet-stream"
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    # ── Route Handlers ─────────────────────────────────

    def _handle_index(self, query: dict, path: str):
        """首頁 — 最近推論記錄（支援分頁）"""
        try:
            limit = int(query.get("limit", [50])[0])
            limit = max(1, min(limit, 500))  # 限制 1~500
        except (ValueError, TypeError):
            limit = 50
        try:
            page = int(query.get("page", [1])[0])
            page = max(1, page)
        except (ValueError, TypeError):
            page = 1

        offset = (page - 1) * limit
        records, total_count = self.db.query_paged(limit, offset) if self.db else ([], 0)
        shift_stats = self.db.get_shift_statistics() if self.db else {}

        # 計算 OK/NG 比率
        s_total = shift_stats.get('total', 0) or 0
        s_ok = shift_stats.get('ok_count', 0) or 0
        s_ng = shift_stats.get('ng_count', 0) or 0
        ok_rate = (s_ok / s_total * 100) if s_total > 0 else 0
        ng_rate = (s_ng / s_total * 100) if s_total > 0 else 0

        import math
        total_pages = max(1, math.ceil(total_count / limit))

        template = self.jinja_env.get_template("index.html")
        html = template.render(
            records=records,
            stats=shift_stats,
            ok_rate=ok_rate,
            ng_rate=ng_rate,
            request_path=path,
            page=page,
            limit=limit,
            total_count=total_count,
            total_pages=total_pages,
        )
        self._send_response(200, html)

    def _handle_overexposed(self, query: dict, path: str):
        """過曝記錄列表"""
        try:
            limit = int(query.get("limit", [50])[0])
            limit = max(1, min(limit, 500))
        except (ValueError, TypeError):
            limit = 50
        try:
            page = int(query.get("page", [1])[0])
            page = max(1, page)
        except (ValueError, TypeError):
            page = 1

        offset = (page - 1) * limit
        records, total_count = self.db.query_overexposed(limit, offset) if self.db else ([], 0)

        import math
        total_pages = max(1, math.ceil(total_count / limit))

        template = self.jinja_env.get_template("overexposed.html")
        html = template.render(
            records=records,
            total_count=total_count,
            request_path=path,
            page=page,
            limit=limit,
            total_pages=total_pages,
        )
        self._send_response(200, html)

    def _handle_record_detail(self, record_id_str: str, path: str):
        """記錄詳情頁"""
        try:
            record_id = int(record_id_str)
        except ValueError:
            self._send_404(path)
            return

        detail = self.db.get_record_detail(record_id) if self.db else None
        if not detail:
            self._send_404(path)
            return

        template = self.jinja_env.get_template("record_detail.html")
        html = template.render(
            detail=detail,
            heatmap_base_dir=self.heatmap_base_dir,
            request_path=path
        )
        self._send_response(200, html)

    def _handle_search(self, query: dict, path: str):
        """搜尋頁面（含日期篩選、分頁）"""
        record_id = query.get("record_id", [""])[0]
        glass_id = query.get("glass_id", [""])[0]
        machine_no = query.get("machine_no", [""])[0]
        ai_judgment = query.get("ai_judgment", [""])[0]
        start_date = query.get("start_date", [""])[0]
        end_date = query.get("end_date", [""])[0]
        cross_filter = query.get("cross_filter", [""])[0]

        # 預設顯示近 7 天（首次進入頁面時）
        if not any([record_id, glass_id, machine_no, ai_judgment, start_date, end_date, cross_filter]):
            today = datetime.now()
            end_date = today.strftime("%Y-%m-%d")
            start_date = (today - timedelta(days=6)).strftime("%Y-%m-%d")

        per_page = 50
        try:
            page = max(1, int(query.get("page", ["1"])[0]))
        except (ValueError, IndexError):
            page = 1

        end_date_full = f"{end_date} 23:59:59" if end_date else ""

        # 先查總數以便在查詢前校正頁碼
        records = []
        total_count = 0
        if self.db:
            records, total_count = self.db.search_records(
                glass_id=glass_id,
                machine_no=machine_no,
                ai_judgment=ai_judgment,
                start_date=start_date,
                end_date=end_date_full,
                cross_filter=cross_filter,
                record_id=record_id,
                limit=per_page,
                offset=(page - 1) * per_page,
            )

        total_pages = max(1, (total_count + per_page - 1) // per_page)
        if page > total_pages:
            page = total_pages

        template = self.jinja_env.get_template("search.html")
        html = template.render(
            record_id=record_id,
            glass_id=glass_id,
            machine_no=machine_no,
            ai_judgment=ai_judgment,
            start_date=start_date,
            end_date=end_date,
            cross_filter=cross_filter,
            records=records,
            total_count=total_count,
            page=page,
            total_pages=total_pages,
            request_path=path
        )
        self._send_response(200, html)

    def _handle_search_export(self, query: dict):
        """匯出搜尋結果為 CSV"""
        import csv
        import io
        from datetime import datetime as _dt

        record_id  = query.get("record_id",   [""])[0]
        glass_id   = query.get("glass_id",    [""])[0]
        machine_no = query.get("machine_no",  [""])[0]
        ai_judgment = query.get("ai_judgment", [""])[0]
        start_date = query.get("start_date",  [""])[0]
        end_date   = query.get("end_date",    [""])[0]
        cross_filter = query.get("cross_filter", [""])[0]
        end_date_full = f"{end_date} 23:59:59" if end_date else ""

        records, _ = self.db.search_records(
            glass_id=glass_id,
            machine_no=machine_no,
            ai_judgment=ai_judgment,
            start_date=start_date,
            end_date=end_date_full,
            cross_filter=cross_filter,
            record_id=record_id,
            limit=10000,
        ) if self.db else ([], 0)

        # 建立 CSV 內容（UTF-8 BOM，讓 Excel 正常顯示中文）
        buf = io.StringIO()
        buf.write("\ufeff")  # BOM
        writer = csv.writer(buf)
        writer.writerow(["ID", "玻璃編號", "機種", "機台", "機檢判定", "AI判定", "耗時(s)", "建立時間", "圖片數", "NG圖片數"])
        for r in records:
            writer.writerow([
                r.get("id", ""),
                r.get("glass_id", ""),
                r.get("model_id", ""),
                r.get("machine_no", ""),
                r.get("machine_judgment", ""),
                r.get("ai_judgment", ""),
                r.get("processing_seconds", ""),
                r.get("created_at", ""),
                r.get("total_images", ""),
                r.get("ng_images", ""),
            ])

        csv_bytes = buf.getvalue().encode("utf-8-sig")

        # 組出有意義的檔名
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        date_part = ""
        if start_date or end_date:
            date_part = f"_{start_date or ''}~{end_date or ''}"
        filename = f"capi_records{date_part}_{ts}.csv"

        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8-sig")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(csv_bytes)))
        self.end_headers()
        self.wfile.write(csv_bytes)

    def _handle_dashboard(self, query: dict, path: str):
        """即時儀表板頁面"""
        template = self.jinja_env.get_template("dashboard.html")
        html = template.render(request_path=path)
        self._send_response(200, html)

    def _handle_dashboard_v2(self, query: dict, path: str):
        """AI-Native 即時儀表板頁面"""
        template = self.jinja_env.get_template("dashboard_v2.html")
        html = template.render(request_path=path)
        self._send_response(200, html)
        
    def _handle_dashboard_v3(self, query: dict, path: str):
        """V3 高階 UI 即時儀表板頁面"""
        template = self.jinja_env.get_template("dashboard_v3.html")
        html = template.render(request_path=path)
        self._send_response(200, html)

    def _handle_record_detail_v3(self, record_id_str: str, path: str):
        """V3 高階 UI 記錄詳情頁"""
        try:
            record_id = int(record_id_str)
        except ValueError:
            self._send_404(path)
            return

        detail = self.db.get_record_detail(record_id) if self.db else None
        if not detail:
            self._send_404(path)
            return

        template = self.jinja_env.get_template("record_detail_v3.html")
        html = template.render(
            detail=detail,
            heatmap_base_dir=self.heatmap_base_dir,
            request_path=path
        )
        self._send_response(200, html)
        
    def _handle_logs_page(self, query: dict, path: str):
        """Log Viewer 頁面"""
        template = self.jinja_env.get_template("logs.html")
        # 列出可用的 log 檔案 (current + rotated)
        log_files = []
        if self._log_file:
            log_path = Path(self._log_file)
            if log_path.exists():
                log_files.append({"name": log_path.name, "path": str(log_path), "size": log_path.stat().st_size})
            # rotated files: server.log.1, server.log.2, ...
            for i in range(1, 10):
                rotated = log_path.parent / f"{log_path.name}.{i}"
                if rotated.exists():
                    log_files.append({"name": rotated.name, "path": str(rotated), "size": rotated.stat().st_size})
        html = template.render(request_path=path, log_files=log_files, log_configured=bool(self._log_file))
        self._send_response(200, html)

    def _handle_api_logs(self, query: dict):
        """API: 讀取 log 檔案內容"""
        if not self._log_file:
            self._send_json({"error": "未設定日誌檔案路徑", "lines": []})
            return

        # 選擇要讀取的 log 檔案 (支援 rotated files)
        file_index = int(query.get("file", [0])[0])  # 0=current, 1=.1, 2=.2, ...
        tail_lines = int(query.get("lines", [500])[0])
        tail_lines = min(tail_lines, 5000)  # 上限 5000 行
        search = query.get("search", [""])[0]
        level_filter = query.get("level", [""])[0].upper()

        log_path = Path(self._log_file)
        if file_index > 0:
            log_path = log_path.parent / f"{log_path.name}.{file_index}"

        if not log_path.exists():
            self._send_json({"error": f"日誌檔案不存在: {log_path.name}", "lines": []})
            return

        try:
            # 讀取最後 N 行 (高效能 tail)
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                # 快速 tail: 讀取末尾 chunk
                f.seek(0, 2)
                file_size = f.tell()
                # 估算每行約 200 bytes, 多讀一些
                chunk_size = min(file_size, tail_lines * 300)
                f.seek(max(0, file_size - chunk_size))
                if f.tell() > 0:
                    f.readline()  # 跳過不完整的第一行
                all_lines = f.readlines()
                lines = all_lines[-tail_lines:]

            # 過濾
            if level_filter:
                lines = [l for l in lines if level_filter in l]
            if search:
                search_lower = search.lower()
                lines = [l for l in lines if search_lower in l.lower()]

            self._send_json({
                "file": log_path.name,
                "total_lines": len(lines),
                "lines": [l.rstrip("\n\r") for l in lines],
            })
        except Exception as e:
            self._send_json({"error": str(e), "lines": []})

    def _handle_api_status(self):
        """API: 即時伺服器狀態"""
        try:
            if hasattr(self, 'status_tracker') and self.status_tracker:
                status = self.status_tracker.get_status()
            else:
                from capi_server import server_status
                status = server_status.get_status()
                
            # 將當班統計數據替換為 DB (支援重啟後恢復)
            if self.db:
                shift_stats = self.db.get_shift_statistics()
                status.setdefault("stats", {})
                status["stats"]["total_requests"] = shift_stats.get("total", 0) or 0
                status["stats"]["total_ok"] = shift_stats.get("ok_count", 0) or 0
                status["stats"]["total_ng"] = shift_stats.get("ng_count", 0) or 0
                status["stats"]["total_err"] = shift_stats.get("err_count", 0) or 0
                status["stats"]["shift_name"] = shift_stats.get("shift_name", "當班")
                status["stats"]["time_range"] = shift_stats.get("time_range", "")

                # 取最近 1 筆 image_results (有熱力圖)
                try:
                    import sqlite3 as _sqlite3
                    from pathlib import Path as _Path

                    conn = _sqlite3.connect(str(self.db.db_path), timeout=5)
                    conn.row_factory = _sqlite3.Row

                    # 按時間最新，取最新 1 筆 (無論 OK/NG)
                    row = conn.execute(
                        """SELECT
                               ir.id             AS img_id,
                               ir.record_id,
                               ir.heatmap_path,
                               ir.is_ng,
                               ir.image_name,
                               rec.glass_id,
                               rec.ai_judgment,
                               rec.created_at
                           FROM image_results ir
                           JOIN inference_records rec ON rec.id = ir.record_id
                           WHERE ir.heatmap_path != ''
                           ORDER BY rec.created_at DESC, ir.id DESC
                           LIMIT 1"""
                    ).fetchone()

                    recent_heatmaps = []
                    base = self.heatmap_base_dir
                    
                    if row:
                        # 1. 放入 Overview
                        hm_abs = row["heatmap_path"]
                        url = None
                        if hm_abs and base:
                            try:
                                rel = str(_Path(hm_abs).relative_to(base)).replace("\\", "/")
                                url = f"/heatmaps/{rel}"
                            except ValueError:
                                url = None
                        if url:
                            recent_heatmaps.append({
                                "url":        url,
                                "glass_id":   row["glass_id"] or "",
                                "image_name": row["image_name"] or "",
                                "is_ng":      bool(row["is_ng"]),
                                "status":     "NG" if row["is_ng"] else "OK",
                                "label":      "Overview",
                                "judgment":   row["ai_judgment"] or "",
                                "created_at": row["created_at"] or "",
                                "record_id":  row["record_id"],
                            })
                            
                        # 2. 抓取區域熱力圖 (Tiles)，最多取 8 張以填好 9 宮格
                        img_id = row["img_id"]
                        tiles = conn.execute(
                            """SELECT tile_id, heatmap_path, is_anomaly, is_dust, is_bomb, is_aoi_coord
                               FROM tile_results
                               WHERE image_result_id = ? AND heatmap_path != ''
                               ORDER BY is_anomaly DESC, score DESC
                               LIMIT 8""",
                            (img_id,)
                        ).fetchall()
                        
                        for t in tiles:
                            thm_abs = t["heatmap_path"]
                            t_url = None
                            if thm_abs and base:
                                try:
                                    t_rel = str(_Path(thm_abs).relative_to(base)).replace("\\", "/")
                                    t_url = f"/heatmaps/{t_rel}"
                                except ValueError:
                                    t_url = None
                            if t_url:
                                is_bomb = bool(t["is_bomb"])
                                is_dust = bool(t["is_dust"])
                                is_ng = bool(t["is_anomaly"])
                                
                                is_exclude_zone = bool(t.get("is_exclude_zone", 0))
                                is_aoi_coord = bool(t.get("is_aoi_coord", 0))

                                tile_status = "OK"
                                if is_exclude_zone:
                                    tile_status = "EXCLUDED"
                                elif is_bomb:
                                    tile_status = "BOMB"
                                elif is_dust:
                                    tile_status = "DUST"
                                elif is_ng:
                                    tile_status = "NG"

                                tile_label = f"Tile #{t['tile_id']}"
                                if is_aoi_coord:
                                    tile_label = f"🎯 AOI #{t['tile_id']}"

                                recent_heatmaps.append({
                                    "url":        t_url,
                                    "glass_id":   row["glass_id"] or "",
                                    "image_name": row["image_name"] or "",
                                    "is_ng":      is_ng or is_bomb or is_dust,
                                    "status":     tile_status,
                                    "label":      tile_label,
                                    "judgment":   row["ai_judgment"] or "",
                                    "created_at": row["created_at"] or "",
                                    "record_id":  row["record_id"],
                                })

                    conn.close()
                    status["recent_heatmaps"] = recent_heatmaps
                except Exception:
                    status["recent_heatmaps"] = []

            self._send_json(status)
        except Exception as e:
            self._send_error(500, f"Cannot get server status: {e}")


    def _handle_api_stats(self, query: dict):
        """API: 統計資料"""
        try:
            days = int(query.get("days", [7])[0])
        except (ValueError, TypeError):
            days = 7
        try:
            limit = int(query.get("limit", [15])[0])
        except (ValueError, TypeError):
            limit = 15
        
        stats = self.db.get_statistics(days) if self.db else {}
        
        # 附加最近的一批記錄給 dashboard_v2
        recent_records = []
        if self.db:
            recent_list = self.db.query_recent(limit)
            for r in recent_list:
                rec_dict = dict(r)
                hm_path = rec_dict.get("first_heatmap_path")
                if hm_path and self.heatmap_base_dir:
                    rec_dict["hm_url"] = hm_relative(hm_path, self.heatmap_base_dir)
                recent_records.append(rec_dict)
                
        stats["recent_records"] = recent_records
        
        self._send_json(stats)

    def _handle_static_file(self, path: str):
        """靜態檔案服務 (熱力圖)"""
        # /heatmaps/20260225/GLASS001/overview_G0F00000.png
        rel_path = path[len("/heatmaps/"):]
        # 安全檢查：防止路徑穿越
        rel_path = rel_path.replace("..", "").lstrip("/")
        full_path = Path(self.heatmap_base_dir) / rel_path
        if full_path.exists() and full_path.is_file():
            self._send_binary(str(full_path))
        else:
            self._send_404()

    def _handle_source_image(self, path: str):
        """靜態檔案服務 (原始圖片)"""
        # /images/{record_id}/{image_name}
        try:
            parts = path.strip("/").split("/")
            if len(parts) != 3:
                self._send_404()
                return
            record_id = int(parts[1])
            image_name = parts[2]
            
            # 安全檢查：防止路徑穿越
            image_name = image_name.replace("..", "").replace("/", "").replace("\\", "")
            
            detail = self.db.get_record_detail(record_id) if self.db else None
            if not detail or not detail.get("image_dir"):
                self._send_404()
                return
                
            full_path = Path(detail["image_dir"]) / image_name
            if full_path.exists() and full_path.is_file():
                self._send_binary(str(full_path))
            else:
                self._send_404()
        except Exception as e:
            logger.error(f"Error serving source image {path}: {e}")
            self._send_404()

    def _handle_imgs_file(self, path: str):
        """靜態檔案服務 (UI 圖片/影片)"""
        rel_path = path[len("/imgs/"):]
        rel_path = rel_path.replace("..", "").lstrip("/")
        full_path = Path(__file__).parent / "templates" / "imgs" / rel_path
        if full_path.exists() and full_path.is_file():
            self._send_binary(str(full_path))
        else:
            self._send_404()

    def _handle_static_assets(self, path: str):
        """靜態檔案服務 (CSS/JS)"""
        rel_path = path[len("/static/"):]
        rel_path = rel_path.replace("..", "").lstrip("/")
        full_path = Path(__file__).parent / "static" / rel_path
        if full_path.exists() and full_path.is_file():
            self._send_binary(str(full_path))
        else:
            self._send_404()

    # ── Debug 推論功能 ─────────────────────────────────

    def _handle_debug_serve_image(self, query):
        """API: 以絕對路徑提供原始圖片 (僅 Debug 用)
        瀏覽器不支援 TIF/TIFF/BMP，自動轉為 PNG 回傳。
        """
        try:
            # query 已由 do_GET 透過 parse_qs 解析為 dict
            params = query if isinstance(query, dict) else urllib.parse.parse_qs(query)
            img_path = params.get("path", [None])[0]
            if not img_path:
                self._send_error(400, "missing path parameter")
                return
            p = Path(img_path)
            if not p.exists() or not p.is_file():
                self._send_error(404, f"file not found: {img_path}")
                return

            suffix = p.suffix.lower()
            # 瀏覽器原生可顯示的格式直接回傳
            if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"):
                self._send_binary(str(p))
                return

            # 其餘格式 (tif, tiff, bmp …) 用 cv2 轉 PNG
            import cv2
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                self._send_error(400, f"cv2 cannot read: {img_path}")
                return
            # 16-bit → 8-bit
            if img.dtype != "uint8":
                import numpy as np
                img = (img.astype(np.float32) / img.max() * 255).astype(np.uint8)
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                self._send_error(500, "PNG encode failed")
                return
            data = buf.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            logger.error(f"Error serving debug image: {e}")
            self._send_error(500, str(e))

    def _handle_debug_page(self, path: str):
        """Debug 推論頁面"""
        # 從 DB 讀取最新設定，若無則 fallback 到推論器的 config
        db_params = {}
        if self.db:
            try:
                for p in self.db.get_all_config_params():
                    db_params[p["param_name"]] = p["decoded_value"]
            except Exception as e:
                logger.error(f"無法讀取 DB 設定: {e}")

        def get_val(name, default_val):
            if name in db_params:
                return db_params[name]
            if self.inferencer and hasattr(self.inferencer.config, name):
                return getattr(self.inferencer.config, name)
            return default_val
        
        template = self.jinja_env.get_template("debug_inference.html")
        html = template.render(
            request_path=path,
            default_threshold=get_val('anomaly_threshold', 0.5),
            default_edge_margin=get_val('edge_margin_px', 0),
            default_dust_extension=get_val('dust_extension', 0),
            default_dust_metric=get_val('dust_heatmap_metric', 'iou'),
            default_dust_iou_thr=get_val('dust_heatmap_iou_threshold', 0.01),
            default_dust_top_pct=get_val('dust_heatmap_top_percent', 5.0),
            model_resolution_map=get_val('model_resolution_map', {}),
            default_patchcore_filter_enabled=get_val('patchcore_filter_enabled', False),
            default_patchcore_blur_sigma=get_val('patchcore_blur_sigma', 1.5),
            default_patchcore_min_area=get_val('patchcore_min_area', 10),
            default_patchcore_score_metric=get_val('patchcore_score_metric', 'max'),
            default_otsu_offset=get_val('otsu_offset', 5),
            default_bs_diff_threshold=get_val('bright_spot_diff_threshold', 10),
            default_bs_median_kernel=get_val('bright_spot_median_kernel', 21),
            default_bs_min_area=get_val('bright_spot_min_area', 5),
            default_bs_threshold=get_val('bright_spot_threshold', 200),
        )
        self._send_response(200, html)

    # ── RIC 人工檢驗報表功能 ─────────────────────────

    def _handle_ric_page(self, query: dict, path: str):
        """人工檢驗 (RIC) 比對報表頁面"""
        batches = self.db.get_ric_batches() if self.db else []
        template = self.jinja_env.get_template("ric_report.html")
        html = template.render(request_path=path, batches=batches)
        self._send_response(200, html)

    def _handle_ric_upload(self):
        """上傳 XLS 檔案並匯入 RIC 資料"""
        import cgi
        import io

        content_type = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in content_type:
            self._send_json({"error": "請使用 multipart/form-data 上傳檔案"})
            return

        # 解析 multipart form data
        try:
            boundary = content_type.split('boundary=')[1]
        except IndexError:
            self._send_json({"error": "無法解析 boundary"})
            return

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        # 用 cgi 解析 multipart
        environ = {
            'REQUEST_METHOD': 'POST',
            'CONTENT_TYPE': content_type,
            'CONTENT_LENGTH': str(content_length),
        }
        fs = cgi.FieldStorage(
            fp=io.BytesIO(body),
            environ=environ,
            keep_blank_values=True
        )

        if 'file' not in fs:
            self._send_json({"error": "請提供檔案 (field name: file)"})
            return

        file_item = fs['file']
        filename = file_item.filename or 'unknown.xlsx'
        file_data = file_item.file.read()

        if not file_data:
            self._send_json({"error": "檔案為空"})
            return

        try:
            # 先偵測是否為 HTML 格式 (很多舊系統匯出的 .xls 其實是 HTML)
            is_html = file_data[:100].strip().startswith(b'<')

            if is_html:
                # HTML Table 格式 → 用 html.parser
                from html.parser import HTMLParser
                import html as html_module

                class TableParser(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.in_th = False
                        self.in_td = False
                        self.headers = []
                        self.rows = []
                        self.current_row = []
                        self.current_data = ''

                    def handle_starttag(self, tag, attrs):
                        if tag == 'th':
                            self.in_th = True
                            self.current_data = ''
                        elif tag == 'td':
                            self.in_td = True
                            self.current_data = ''
                        elif tag == 'tr':
                            self.current_row = []

                    def handle_endtag(self, tag):
                        if tag == 'th':
                            self.in_th = False
                            self.headers.append(self.current_data.strip())
                        elif tag == 'td':
                            self.in_td = False
                            val = self.current_data.strip()
                            if val == '\xa0' or val == '&nbsp;':
                                val = ''
                            self.current_row.append(val)
                        elif tag == 'tr':
                            if self.current_row:
                                self.rows.append(self.current_row)

                    def handle_data(self, data):
                        if self.in_th or self.in_td:
                            self.current_data += data

                    def handle_entityref(self, name):
                        if self.in_th or self.in_td:
                            ch = html_module.unescape(f'&{name};')
                            self.current_data += ch

                # 嘗試多種編碼
                text = ''
                for enc in ['utf-8', 'gb2312', 'gbk', 'big5', 'latin-1']:
                    try:
                        text = file_data.decode(enc)
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue

                parser = TableParser()
                parser.feed(text)

                headers = parser.headers
                records_data = []
                for row in parser.rows:
                    if not any(row):
                        continue
                    rec = {}
                    for i, header in enumerate(headers):
                        if i < len(row):
                            rec[header] = row[i]
                        else:
                            rec[header] = ''
                    if rec.get('PNL_ID'):
                        records_data.append(rec)

            elif filename.lower().endswith('.xls') and not filename.lower().endswith('.xlsx'):
                # .xls 格式 (Excel 97-2003) → 用 xlrd
                import xlrd
                wb = xlrd.open_workbook(file_contents=file_data)
                ws = wb.sheet_by_index(0)

                if ws.nrows == 0:
                    self._send_json({"error": "檔案無資料"})
                    return

                headers = [str(ws.cell_value(0, c)).strip() for c in range(ws.ncols)]

                records_data = []
                for r in range(1, ws.nrows):
                    row_vals = [ws.cell_value(r, c) for c in range(ws.ncols)]
                    if not any(row_vals):
                        continue
                    rec = {}
                    for i, header in enumerate(headers):
                        if i < len(row_vals):
                            val = row_vals[i]
                            rec[header] = str(val) if val is not None else ''
                        else:
                            rec[header] = ''
                    if rec.get('PNL_ID'):
                        records_data.append(rec)

            else:
                self._send_json({"error": "不支援的檔案格式，請上傳 .xls 檔案"})
                return

            if not records_data:
                self._send_json({"error": "檔案中無有效 RIC 記錄 (缺少 PNL_ID)"})
                return

            batch_id = self.db.save_ric_batch(filename, records_data)

            self._send_json({
                "success": True,
                "batch_id": batch_id,
                "filename": filename,
                "total_records": len(records_data),
                "message": f"成功匯入 {len(records_data)} 筆 RIC 記錄"
            })

        except ImportError as ie:
            self._send_json({"error": f"缺少套件，請執行: pip install xlrd ({ie})"})
        except Exception as e:
            logger.error(f"RIC upload error: {e}", exc_info=True)
            self._send_json({"error": f"解析檔案失敗: {str(e)}"})

    def _handle_ric_delete(self):
        """API: 刪除 RIC 匯入批次"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            batch_id = data.get('batch_id')

            if not batch_id:
                self._send_json({"error": "缺少 batch_id 參數"})
                return

            batch_id = int(batch_id)
            deleted_count = self.db.delete_ric_batch(batch_id) if self.db else 0

            self._send_json({
                "success": True,
                "batch_id": batch_id,
                "deleted_records": deleted_count,
                "message": f"已刪除批次 #{batch_id}，共 {deleted_count} 筆記錄"
            })
        except (ValueError, TypeError) as e:
            self._send_json({"error": f"參數錯誤: {e}"})
        except Exception as e:
            logger.error(f"RIC delete error: {e}", exc_info=True)
            self._send_json({"error": f"刪除失敗: {str(e)}"})

    def _handle_client_import(self):
        """API: 匯入 client accuracy records 至資料庫"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            records = data.get("records", [])
            if not records:
                self._send_json({"error": "無資料可匯入"})
                return

            result = self.db.save_client_accuracy_records(records)
            self._send_json({
                "success": True,
                "inserted": result["inserted"],
                "skipped": result["skipped"],
                "message": f"新增 {result['inserted']} 筆，略過 {result['skipped']} 筆重複資料"
            })
        except Exception as e:
            logger.error(f"Client import error: {e}", exc_info=True)
            self._send_json({"error": f"匯入失敗: {str(e)}"})

    def _handle_client_data_api(self, query: dict):
        """API: 取得已儲存的 client accuracy records（支援日期篩選 + summary 統計）"""
        try:
            start_date = query.get('start_date', [''])[0] or None
            end_date = query.get('end_date', [''])[0] or None
            records = self.db.get_client_accuracy_records(start_date, end_date)
            inf_ids = list({r["inference_record_id"] for r in records if r.get("inference_record_id")})
            dust_ids = self.db.get_dust_affected_record_ids(inf_ids) if inf_ids else set()
            summary, out_records = self._compute_client_summary(records, dust_ids)

            self._send_json({
                "success": True,
                "total": summary["total"],
                "summary": summary,
                "records": out_records,
            })
        except ValueError as ve:
            self._send_json({"success": False, "error": str(ve)})
        except Exception as e:
            logger.error(f"Client data API error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _compute_client_summary(self, records: list, dust_affected_ids: set = None):
        """從 client accuracy records 計算統計摘要並格式化 records，單次遍歷。
        Returns: (summary_dict, out_records_list)
        """
        from capi_database import CAPIDatabase

        _empty_miss_cats = lambda: {c: 0 for c in CAPIDatabase.VALID_MISS_CATEGORIES}
        _empty_over_cats = lambda: {c: 0 for c in CAPIDatabase.VALID_OVER_CATEGORIES}
        total = len(records)
        if total == 0:
            return {
                "total": 0, "aoiNG": 0, "aiNG": 0, "ricNG": 0,
                "aoiCorrect": 0, "aiCorrect": 0,
                "aoiOver": 0, "aoiOverRate": 0,
                "aiOver": 0, "aiOverRate": 0,
                "aiMiss": 0, "aiMissRate": 0,
                "revival": 0, "revivalRate": 0,
                "combos": {}, "daily": {},
                "missReviewStats": {
                    "total": 0, "reviewed": 0, "unreviewed": 0,
                    "byCategory": _empty_miss_cats(),
                },
                "overReviewStats": {
                    "total": 0, "reviewed": 0, "unreviewed": 0,
                    "byCategory": _empty_over_cats(),
                },
            }, []

        aoiNG = aiNG = ricNG = 0
        aoiCorrect = aiCorrect = 0
        aoiOver = aiOver = aiMiss = 0
        revival = 0
        combos = {}
        daily = {}
        miss_reviewed = 0
        miss_by_cat = _empty_miss_cats()
        over_reviewed = 0
        over_by_cat = _empty_over_cats()
        out_records = []

        for rec in records:
            eqp = rec["result_eqp"] or "OK"
            ai = rec["result_ai"] or "OK"
            ric = CAPIDatabase.parse_ric_judgment(rec.get("datastr", ""))

            # Build formatted output record in the same pass
            out_rec = {
                "id": rec["id"],
                "time_stamp": rec["time_stamp"],
                "pnl_id": rec["pnl_id"],
                "mach_id": rec["mach_id"],
                "result_eqp": eqp,
                "result_ai": ai,
                "result_ric": rec["result_ric"],
                "datastr": rec["datastr"] or "",
                "inference_record_id": rec.get("inference_record_id"),
                "has_dust_filtering": bool(
                    dust_affected_ids
                    and rec.get("inference_record_id")
                    and rec["inference_record_id"] in dust_affected_ids
                ),
                "miss_review": None,
                "over_review": None,
            }
            if rec.get("review_id"):
                out_rec["miss_review"] = {
                    "id": rec["review_id"],
                    "category": rec["review_category"],
                    "note": rec["review_note"] or "",
                    "updated_at": rec["review_updated_at"],
                }
            if rec.get("over_review_id"):
                out_rec["over_review"] = {
                    "id": rec["over_review_id"],
                    "category": rec["over_review_category"],
                    "note": rec["over_review_note"] or "",
                    "updated_at": rec["over_review_updated_at"],
                }
            out_records.append(out_rec)

            if eqp == "NG":
                aoiNG += 1
            if ai == "NG":
                aiNG += 1
            if ric == "NG":
                ricNG += 1

            if eqp == ric:
                aoiCorrect += 1
            if ai == ric:
                aiCorrect += 1

            if eqp == "NG" and ric == "OK":
                aoiOver += 1
            if ai == "NG" and ric == "OK":
                aiOver += 1
                if rec.get("over_review_category"):
                    over_reviewed += 1
                    cat = rec["over_review_category"]
                    if cat in over_by_cat:
                        over_by_cat[cat] += 1
            if ai == "OK" and ric == "NG":
                aiMiss += 1
                if rec.get("review_category"):
                    miss_reviewed += 1
                    cat = rec["review_category"]
                    if cat in miss_by_cat:
                        miss_by_cat[cat] += 1
            if eqp == "NG" and ai == "OK" and ric == "OK":
                revival += 1

            combo = f"{eqp}/{ai}/{ric}"
            combos[combo] = combos.get(combo, 0) + 1

            day = (rec.get("time_stamp") or "unknown")[:10]
            if day not in daily:
                daily[day] = {"total": 0, "aoiCorrect": 0, "aiCorrect": 0, "aiMiss": 0, "aiOver": 0, "aoiOver": 0}
            daily[day]["total"] += 1
            if eqp == ric:
                daily[day]["aoiCorrect"] += 1
            if ai == ric:
                daily[day]["aiCorrect"] += 1
            if ai == "OK" and ric == "NG":
                daily[day]["aiMiss"] += 1
            if ai == "NG" and ric == "OK":
                daily[day]["aiOver"] += 1
            if eqp == "NG" and ric == "OK":
                daily[day]["aoiOver"] += 1

        aoiOverRate = round(aoiOver / total * 100, 1) if total > 0 else 0
        aiOverRate = round(aiOver / total * 100, 1) if total > 0 else 0
        aiMissRate = round(aiMiss / total * 100, 1) if total > 0 else 0
        revivalRate = round(revival / total * 100, 1) if total > 0 else 0

        return {
            "total": total,
            "aoiNG": aoiNG, "aiNG": aiNG, "ricNG": ricNG,
            "aoiCorrect": aoiCorrect, "aiCorrect": aiCorrect,
            "aoiOver": aoiOver, "aoiOverRate": aoiOverRate,
            "aiOver": aiOver, "aiOverRate": aiOverRate,
            "aiMiss": aiMiss, "aiMissRate": aiMissRate,
            "revival": revival, "revivalRate": revivalRate,
            "combos": combos, "daily": daily,
            "missReviewStats": {
                "total": aiMiss,
                "reviewed": miss_reviewed,
                "unreviewed": aiMiss - miss_reviewed,
                "byCategory": miss_by_cat,
            },
            "overReviewStats": {
                "total": aiOver,
                "reviewed": over_reviewed,
                "unreviewed": aiOver - over_reviewed,
                "byCategory": over_by_cat,
            },
        }, out_records

    def _handle_client_clear(self):
        """API: 清除所有 client accuracy records"""
        try:
            count = self.db.clear_client_accuracy_records()
            self._send_json({
                "success": True,
                "deleted": count,
                "message": f"已清除 {count} 筆資料"
            })
        except Exception as e:
            logger.error(f"Client clear error: {e}", exc_info=True)
            self._send_json({"error": str(e)})

    def _handle_miss_review_save(self):
        """API: 儲存/更新漏檢 Review"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            client_record_id = data.get("client_record_id")
            category = data.get("category", "")
            note = data.get("note", "")

            if not client_record_id:
                self._send_json({"success": False, "error": "缺少 client_record_id"})
                return

            review_id = self.db.save_miss_review(int(client_record_id), category, note)
            self._send_json({"success": True, "id": review_id, "message": "Review 已儲存"})
        except ValueError as ve:
            self._send_json({"success": False, "error": str(ve)})
        except Exception as e:
            logger.error(f"Miss review save error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_miss_review_delete(self):
        """API: 刪除漏檢 Review"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            client_record_id = data.get("client_record_id")
            if not client_record_id:
                self._send_json({"success": False, "error": "缺少 client_record_id"})
                return

            deleted = self.db.delete_miss_review(int(client_record_id))
            if deleted:
                self._send_json({"success": True, "message": "Review 已刪除"})
            else:
                self._send_json({"success": False, "error": "Review 不存在"})
        except Exception as e:
            logger.error(f"Miss review delete error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_over_review_save(self):
        """API: 儲存/更新過檢 Review"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            client_record_id = data.get("client_record_id")
            category = data.get("category", "")
            note = data.get("note", "")

            if not client_record_id:
                self._send_json({"success": False, "error": "缺少 client_record_id"})
                return

            review_id = self.db.save_over_review(int(client_record_id), category, note)
            self._send_json({"success": True, "id": review_id, "message": "Review 已儲存"})
        except ValueError as ve:
            self._send_json({"success": False, "error": str(ve)})
        except Exception as e:
            logger.error(f"Over review save error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_over_review_delete(self):
        """API: 刪除過檢 Review"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            client_record_id = data.get("client_record_id")
            if not client_record_id:
                self._send_json({"success": False, "error": "缺少 client_record_id"})
                return

            deleted = self.db.delete_over_review(int(client_record_id))
            if deleted:
                self._send_json({"success": True, "message": "Review 已刪除"})
            else:
                self._send_json({"success": False, "error": "Review 不存在"})
        except Exception as e:
            logger.error(f"Over review delete error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_ric_report_api(self, query: dict):
        """API: 取得 RIC 比對報表資料"""
        try:
            batch_id_str = query.get('batch_id', [''])[0]
            batch_id = int(batch_id_str) if batch_id_str else None
        except (ValueError, TypeError):
            batch_id = None

        stats = self.db.get_ric_accuracy_stats(batch_id) if self.db else {}
        self._send_json(stats)

    def _handle_inference_stats_api(self, query: dict):
        """API: 取得 AI 推論紀錄統計資料"""
        try:
            start_date = query.get('start_date', [''])[0] or None
            end_date = query.get('end_date', [''])[0] or None
            stats = self.db.get_inference_stats(start_date, end_date) if self.db else {"success": False, "error": "DB not available"}
            self._send_json(stats)
        except Exception as e:
            logger.error(f"Inference stats API error: {e}", exc_info=True)
            self._send_json({"success": False, "error": str(e)})

    def _handle_debug_inference_run(self):
        """API: 執行 Debug 單圖推論"""
        import time as _time
        import cv2
        import numpy as np

        # 讀取 POST body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode('utf-8'))
        except Exception:
            self._send_json({"error": "Invalid JSON body"})
            return

        image_path_str = data.get("image_path", "").strip()
        if not image_path_str:
            self._send_json({"error": "請提供圖片路徑 (image_path)"})
            return

        image_path = Path(image_path_str)
        if not image_path.exists():
            self._send_json({"error": f"檔案不存在: {image_path}"})
            return
        if not image_path.is_file():
            self._send_json({"error": f"不是檔案: {image_path}"})
            return

        if self.inferencer is None:
            self._send_json({"error": "推論器尚未載入 (inferencer is None)"})
            return

        # Debug 門檻：預設 0.5（比正式環境低，用於漏檢排查）
        debug_threshold = float(data.get("threshold", 0.5))
        # 邊緣衰減覆寫：None = 使用 config 預設值，0 = 停用，>0 = 自訂寬度
        edge_margin_raw = data.get("edge_margin_px")
        edge_margin_override = int(edge_margin_raw) if edge_margin_raw is not None and str(edge_margin_raw).strip() != "" else None
        
        # 灰塵檢測參數覆寫
        dust_ext_raw = data.get("dust_extension")
        dust_ext_override = int(dust_ext_raw) if dust_ext_raw is not None and str(dust_ext_raw).strip() != "" else None
        
        dust_iou_thr_raw = data.get("dust_heatmap_iou_threshold")
        dust_iou_thr_override = float(dust_iou_thr_raw) if dust_iou_thr_raw is not None and str(dust_iou_thr_raw).strip() != "" else None
        
        dust_top_pct_raw = data.get("dust_heatmap_top_percent")
        dust_top_pct_override = float(dust_top_pct_raw) if dust_top_pct_raw is not None and str(dust_top_pct_raw).strip() != "" else None
        
        dust_metric_override = data.get("dust_heatmap_metric")
        
        otsu_offset_raw = data.get("otsu_offset")
        otsu_offset_override = int(otsu_offset_raw) if otsu_offset_raw is not None and str(otsu_offset_raw).strip() != "" else None

        patchcore_overrides = {}
        if "patchcore_filter_enabled" in data:
            patchcore_overrides["patchcore_filter_enabled"] = bool(data["patchcore_filter_enabled"])
        if "patchcore_blur_sigma" in data and str(data["patchcore_blur_sigma"]).strip() != "":
            patchcore_overrides["patchcore_blur_sigma"] = float(data["patchcore_blur_sigma"])
        if "patchcore_min_area" in data and str(data["patchcore_min_area"]).strip() != "":
            patchcore_overrides["patchcore_min_area"] = int(data["patchcore_min_area"])
        if "patchcore_score_metric" in data and str(data["patchcore_score_metric"]).strip() != "":
            patchcore_overrides["patchcore_score_metric"] = str(data["patchcore_score_metric"])

        try:
            total_start = _time.time()

            # 1. 預處理（不需要 GPU，也不用 threshold）
            result = self.inferencer.preprocess_image(image_path, otsu_offset_override=otsu_offset_override)
            if result is None:
                self._send_json({"error": f"無法載入或預處理圖片: {image_path}"})
                return

            # 2. 多模型路由：依圖片前綴選擇對應的 inferencer
            img_prefix = self.inferencer._get_image_prefix(image_path.name)
            target_inferencer = self.inferencer._get_inferencer_for_prefix(img_prefix)
            
            model_name = "預設模型"
            if img_prefix in self.inferencer._model_mapping:
                model_name = self.inferencer._model_mapping[img_prefix].name

            if target_inferencer is None:
                self._send_json({"error": f"找不到 {image_path.name} 對應的模型"})
                return

            # 3. 推論 — 若有 GPU lock 則排隊
            # model_id: 從 POST 資料取得（可選），用於推導產品解析度
            debug_model_id = data.get("model_id")

            if hasattr(self, '_gpu_lock') and self._gpu_lock:
                with self._gpu_lock:
                    result = self.inferencer.run_inference(
                        result,
                        inferencer=target_inferencer,
                        threshold=debug_threshold,
                        edge_margin_override=edge_margin_override,
                        patchcore_overrides=patchcore_overrides if patchcore_overrides else None,
                        model_id=debug_model_id,
                    )
            else:
                result = self.inferencer.run_inference(
                    result,
                    inferencer=target_inferencer,
                    threshold=debug_threshold,
                    edge_margin_override=edge_margin_override,
                    patchcore_overrides=patchcore_overrides if patchcore_overrides else None,
                    model_id=debug_model_id,
                )

            total_time = _time.time() - total_start

            # 3. 建立 Debug heatmap 暫存目錄
            if CAPIWebHandler._debug_heatmap_dir is None:
                CAPIWebHandler._debug_heatmap_dir = Path(tempfile.mkdtemp(prefix="capi_debug_hm_"))
            debug_dir = CAPIWebHandler._debug_heatmap_dir
            debug_dir.mkdir(parents=True, exist_ok=True)

            image_name = image_path.stem

            # 4. 產生 Overview 圖
            overview_img = self.inferencer.visualize_inference_result(image_path, result)
            overview_filename = f"debug_overview_{image_name}.png"
            overview_path = debug_dir / overview_filename
            # 縮小存檔
            max_dim = 2000
            h, w = overview_img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                overview_img = cv2.resize(overview_img, (int(w * scale), int(h * scale)))
            cv2.imwrite(str(overview_path), overview_img)
            overview_url = f"/debug/heatmaps/{overview_filename}"

            # 5. 產生各 Tile 組合圖 (與推論記錄格式一致)
            tiles_data = []
            image_dir = image_path.parent
            
            # 先找是否有 OMIT 圖片 (Panel 級別共用)
            omit_candidates = []
            for pattern in ["PINIGBI*.*", "OMIT0000*.*"]:
                omit_candidates.extend(list(image_dir.glob(pattern)))
            
            omit_full = None
            if omit_candidates:
                omit_full = cv2.imread(str(omit_candidates[0]), cv2.IMREAD_UNCHANGED)
                if omit_full is not None:
                    logger.info(f"[DEBUG] Found OMIT image for dust check: {omit_candidates[0].name}")

            for tile, score, anomaly_map in result.anomaly_tiles:
                # 準備 TileInfo 擴充資訊 (灰塵檢查)
                if omit_full is not None:
                    try:
                        tx, ty, tw, th = tile.x, tile.y, tile.width, tile.height
                        oh, ow = omit_full.shape[:2]
                        if tx < ow and ty < oh:
                            x2_o = min(tx + tw, ow)
                            y2_o = min(ty + th, oh)
                            omit_crop = omit_full[ty:y2_o, tx:x2_o].copy()
                            tile.omit_crop_image = omit_crop
                            
                            # A. 灰塵偵測
                            is_dust, dust_mask, bright_ratio, detail_text = self.inferencer.check_dust_or_scratch_feature(omit_crop, extension_override=dust_ext_override)
                            tile.dust_mask = dust_mask
                            tile.dust_bright_ratio = bright_ratio
                            
                            # B. IOU 計算
                            top_pct = dust_top_pct_override if dust_top_pct_override is not None else self.inferencer.config.dust_heatmap_top_percent
                            metric_mode = dust_metric_override if dust_metric_override else self.inferencer.config.dust_heatmap_metric
                            dust_iou_thr = dust_iou_thr_override if dust_iou_thr_override is not None else self.inferencer.config.dust_heatmap_iou_threshold
                            if is_dust and anomaly_map is not None:
                                iou, heatmap_binary = self.inferencer.compute_dust_heatmap_iou(
                                    dust_mask, anomaly_map, top_percent=top_pct, metric=metric_mode
                                )
                                tile.dust_heatmap_iou = iou
                                # 判定灰塵 (與正式路徑 _dust_check_one 一致)
                                metric_name = "COV" if metric_mode == "coverage" else "IOU"
                                if iou >= dust_iou_thr:
                                    tile.is_suspected_dust_or_scratch = True
                                    detail_text += f" {metric_name}:{iou:.3f}>={metric_name}_THR -> DUST"
                                else:
                                    detail_text += f" {metric_name}:{iou:.3f}<{metric_name}_THR -> REAL_NG"
                                # 產生 Debug 圖
                                tile.dust_iou_debug_image = self.inferencer.generate_dust_iou_debug_image(
                                    tile.image, anomaly_map, dust_mask, heatmap_binary, iou, top_pct, tile.is_suspected_dust_or_scratch
                                )
                            elif is_dust:
                                tile.is_suspected_dust_or_scratch = True
                                detail_text += " (no heatmap, marked as dust)"
                            else:
                                detail_text += " NO_DUST -> REAL_NG"
                            tile.dust_detail_text = detail_text
                    except Exception as e:
                        logger.warning(f"[DEBUG] Dust check processing failed for tile {tile.tile_id}: {e}")

                # 產生組合圖
                if self.heatmap_manager:
                    try:
                        composite_path = self.heatmap_manager.save_tile_heatmap(
                            save_dir=debug_dir,
                            image_name=f"debug_{image_name}",
                            tile_id=tile.tile_id,
                            tile_image=tile.image,
                            anomaly_map=anomaly_map,
                            score=score,
                            tile_info=tile,
                            score_threshold=debug_threshold,
                            iou_threshold=dust_iou_thr_override if dust_iou_thr_override is not None else getattr(self.inferencer.config, 'dust_heatmap_iou_threshold', 0.01),
                        )
                        tile_url = f"/debug/heatmaps/{Path(composite_path).name}"
                    except Exception as e:
                        logger.error(f"[DEBUG] Composite image generation failed for tile {tile.tile_id}: {e}")
                        # Fallback to simple overlay if composite fails
                        overlay = self.heatmap_manager.generate_heatmap_overlay(tile.image, anomaly_map, alpha=0.5)
                        tile_filename = f"debug_tile_{image_name}_t{tile.tile_id}_fallback.png"
                        cv2.imwrite(str(debug_dir / tile_filename), overlay)
                        tile_url = f"/debug/heatmaps/{tile_filename}"
                else:
                    # Fallback (無 HeatmapManager)
                    tile_img = tile.image.copy()
                    if len(tile_img.shape) == 2:
                        tile_img = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2BGR)
                    norm_map = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
                    if heatmap_color.shape[:2] != tile_img.shape[:2]:
                        heatmap_color = cv2.resize(heatmap_color, (tile_img.shape[1], tile_img.shape[0]))
                    overlay = cv2.addWeighted(tile_img, 0.5, heatmap_color, 0.5, 0)
                    tile_filename = f"debug_tile_{image_name}_t{tile.tile_id}_simple.png"
                    cv2.imwrite(str(debug_dir / tile_filename), overlay)
                    tile_url = f"/debug/heatmaps/{tile_filename}"

                tile_status = "NG"
                if tile.is_in_exclude_zone:
                    tile_status = "EXCLUDED"
                elif tile.is_bomb:
                    tile_status = "BOMB"
                elif tile.is_suspected_dust_or_scratch:
                    tile_status = "DUST"

                tiles_data.append({
                    "tile_id": tile.tile_id,
                    "x": tile.x,
                    "y": tile.y,
                    "width": tile.width,
                    "height": tile.height,
                    "score": round(score, 4),
                    "status": tile_status,
                    "is_dust": tile.is_suspected_dust_or_scratch,
                    "dust_iou": round(getattr(tile, 'dust_region_max_cov', tile.dust_heatmap_iou), 4),
                    "is_bomb": tile.is_bomb,
                    "bomb_code": tile.bomb_defect_code,
                    "is_exclude_zone": tile.is_in_exclude_zone,
                    "heatmap_url": tile_url,
                })

            # 6. 判定結果
            has_real_ng = any(
                not t.is_suspected_dust_or_scratch and not t.is_bomb and not t.is_in_exclude_zone
                for t, s, m in result.anomaly_tiles
            )
            all_dust = (
                len(result.anomaly_tiles) > 0 and
                all(t.is_suspected_dust_or_scratch for t, s, m in result.anomaly_tiles)
            )

            if has_real_ng:
                judgment = "NG"
            elif all_dust:
                judgment = "OK (DUST Filtered)"
            elif len(result.anomaly_tiles) > 0:
                judgment = "NG"
            else:
                judgment = "OK"

            response_data = {
                "success": True,
                "image_path": str(image_path),
                "image_name": image_path.name,
                "image_size": list(result.image_size),
                "judgment": judgment,
                "total_tiles": result.processed_tile_count,
                "excluded_tiles": result.excluded_tile_count,
                "anomaly_count": len(result.anomaly_tiles),
                "processing_time": round(total_time, 3),
                "threshold": debug_threshold,
                "edge_margin_px": edge_margin_override if edge_margin_override is not None else self.inferencer.config.edge_margin_px,
                "overview_url": overview_url,
                "tiles": tiles_data,
                "image_prefix": img_prefix,
                "model_name": model_name,
            }

            self._send_json(response_data)
            logger.info(f"[DEBUG] Inference {image_path.name}: {judgment} ({total_time:.2f}s, {len(result.anomaly_tiles)} anomalies)")

        except Exception as e:
            logger.error(f"[DEBUG] Inference error: {e}", exc_info=True)
            self._send_json({"error": f"推論失敗: {str(e)}"})

    def _handle_api_debug_edge_inspect(self):
        """API: 測試單邊 CV 邊緣檢測"""
        import cv2
        import numpy as np
        import base64
        from capi_edge_cv import CVEdgeInspector, EdgeSideConfig
        
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode('utf-8'))
        except Exception:
            self._send_json({"error": "Invalid JSON body"})
            return

        image_path_str = data.get("image_path", "").strip()
        side = data.get("side", "left")
        
        if not image_path_str:
            self._send_json({"error": "請提供圖片路徑 (image_path)"})
            return

        image_path = Path(image_path_str)
        if not image_path.exists():
            self._send_json({"error": f"檔案不存在: {image_path}"})
            return

        try:
            # 讀取參數
            cfg = EdgeSideConfig(
                width=int(data.get("width", 450)),
                threshold=int(data.get("threshold", 5)),
                min_area=int(data.get("min_area", 70)),
                exclude_top=int(data.get("exclude_top", 80)),
                exclude_bottom=int(data.get("exclude_bottom", 80)),
                exclude_left=int(data.get("exclude_left", 10)),
                exclude_right=int(data.get("exclude_right", 10)),
            )

            # 準備推論器 (拿掉這段檢查，因為我們自己算邊界)
            # if self.inferencer is None:
            #     self._send_json({"error": "AI 推論器尚未載入，無法取得邊界參數"})
            #     return
                
            # 讀取圖片並自行找範圍，不依賴 inferencer
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                self._send_json({"error": "無法讀取圖片"})
                return
                
            image_size = (image.shape[1], image.shape[0])
            
            def _fast_otsu_bounds(img: np.ndarray) -> Tuple[int, int, int, int]:
                """輕量版 Otsu 邊界尋找，不載入模型"""
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = np.ones((15, 15), np.uint8)
                closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                img_h, img_w = img.shape[:2]
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = -float('inf'), -float('inf')
                
                for contour in contours:
                    if cv2.contourArea(contour) > 1000:
                        x, y, w, h = cv2.boundingRect(contour)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x + w, x_max)
                        y_max = max(y + h, y_max)
                
                if x_min == float('inf'):
                    return 0, 0, img_w, img_h
                return int(x_min), int(y_min), int(x_max), int(y_max)

            # 放棄使用 self.inferencer 的 calculate_otsu_bounds，因為這會觸發卡死
            # 直接強制使用 _fast_otsu_bounds 算出來的邊界
            otsu_bounds = _fast_otsu_bounds(image)

            # 準備 EdgeInspector (從自己 init)
            inspector = CVEdgeInspector()
            defects, debug_imgs = inspector.inspect_single_side(
                image, otsu_bounds, side, config_override=cfg
            )

            # 將 debug 圖片轉為 base64
            encoded_imgs = {}
            for k, img in debug_imgs.items():
                if img is not None:
                    _, buffer = cv2.imencode('.png', img)
                    b64 = base64.b64encode(buffer).decode('utf-8')
                    encoded_imgs[k] = f"data:image/png;base64,{b64}"

            self._send_json({
                "success": True,
                "defects": [
                    {
                        "area": d.area,
                        "bbox": d.bbox,
                        "center": d.center,
                        "max_diff": d.max_diff
                    } for d in defects
                ],
                "images": encoded_imgs,
                "otsu_bounds": otsu_bounds,
                "image_size": image_size,
            })

        except Exception as e:
            logger.error(f"[DEBUG] Edge Inspect error: {e}", exc_info=True)
            self._send_json({"error": f"邊緣檢測失敗: {str(e)}"})

    def _handle_debug_coord_inference(self):
        """API: 人工座標推論 — 以指定產品座標為中心裁切 512x512 做推論"""
        import time as _time
        import cv2
        import numpy as np
        from capi_inference import TileInfo

        # 讀取 POST body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode('utf-8'))
        except Exception:
            self._send_json({"error": "Invalid JSON body"})
            return

        image_path_str = data.get("image_path", "").strip()
        if not image_path_str:
            self._send_json({"error": "請提供圖片路徑 (image_path)"})
            return

        image_path = Path(image_path_str)
        if not image_path.exists():
            self._send_json({"error": f"檔案不存在: {image_path}"})
            return

        if self.inferencer is None:
            self._send_json({"error": "推論器尚未載入 (inferencer is None)"})
            return

        # 解析參數
        try:
            product_x = int(data.get("product_x", 0))
            product_y = int(data.get("product_y", 0))
            product_w = int(data.get("product_w", 1920))
            product_h = int(data.get("product_h", 1080))
        except (ValueError, TypeError) as e:
            self._send_json({"error": f"座標或解析度參數無效: {e}"})
            return

        debug_threshold = float(data.get("threshold", 0.5))
        edge_margin_raw = data.get("edge_margin_px")
        edge_margin_override = int(edge_margin_raw) if edge_margin_raw is not None else None

        try:
            total_start = _time.time()

            # 1. 載入圖片
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                self._send_json({"error": f"無法載入圖片: {image_path}"})
                return

            img_h, img_w = image.shape[:2]

            # 2. 計算 raw_bounds (面板在圖片中的實際邊界) 與 otsu_bounds
            raw_bounds = self.inferencer._find_raw_object_bounds(image)
            if raw_bounds is None:
                raw_bounds = (0, 0, img_w, img_h)
            
            otsu_bounds, _ = self.inferencer.calculate_otsu_bounds(image)
            if otsu_bounds is None:
                otsu_bounds = raw_bounds

            x_start, y_start, x_end, y_end = raw_bounds

            # 3. 產品座標 → 圖片座標
            scale_x = (x_end - x_start) / product_w if product_w > 0 else 1.0
            scale_y = (y_end - y_start) / product_h if product_h > 0 else 1.0
            img_cx = int(product_x * scale_x + x_start)
            img_cy = int(product_y * scale_y + y_start)

            # 4. 以 (img_cx, img_cy) 為中心裁切 512x512，邊界 clamp
            tile_size = 512
            half = tile_size // 2
            crop_x1 = max(0, img_cx - half)
            crop_y1 = max(0, img_cy - half)
            crop_x2 = crop_x1 + tile_size
            crop_y2 = crop_y1 + tile_size

            # 如果超出右/下邊界，向前推
            if crop_x2 > img_w:
                crop_x2 = img_w
                crop_x1 = max(0, crop_x2 - tile_size)
            if crop_y2 > img_h:
                crop_y2 = img_h
                crop_y1 = max(0, crop_y2 - tile_size)

            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            tile_image = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            # 5. 建立 TileInfo + 推論
            tile_info = TileInfo(
                tile_id=0,
                x=crop_x1, y=crop_y1,
                width=crop_w, height=crop_h,
                image=tile_image,
            )

            # 多模型路由
            img_prefix = self.inferencer._get_image_prefix(image_path.name)
            target_inferencer = self.inferencer._get_inferencer_for_prefix(img_prefix)
            model_name = "預設模型"
            if img_prefix in self.inferencer._model_mapping:
                model_name = self.inferencer._model_mapping[img_prefix].name

            if target_inferencer is None:
                self._send_json({"error": f"找不到 {image_path.name} 對應的模型"})
                return

            # 推論 (含 GPU lock)
            if hasattr(self, '_gpu_lock') and self._gpu_lock:
                with self._gpu_lock:
                    score, anomaly_map = self.inferencer.predict_tile(
                        tile_info, inferencer=target_inferencer,
                        edge_margin_override=edge_margin_override,
                    )
            else:
                score, anomaly_map = self.inferencer.predict_tile(
                    tile_info, inferencer=target_inferencer,
                    edge_margin_override=edge_margin_override,
                )

            total_time = _time.time() - total_start

            # 6. 建立 Debug heatmap 暫存目錄
            if CAPIWebHandler._debug_heatmap_dir is None:
                CAPIWebHandler._debug_heatmap_dir = Path(tempfile.mkdtemp(prefix="capi_debug_hm_"))
            debug_dir = CAPIWebHandler._debug_heatmap_dir
            debug_dir.mkdir(parents=True, exist_ok=True)

            image_name = image_path.stem
            ts = int(_time.time() * 1000) % 100000  # 避免快取

            # 7. 儲存原始裁切圖
            crop_bgr = tile_image.copy()
            if len(crop_bgr.shape) == 2:
                crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
            elif len(crop_bgr.shape) == 3 and crop_bgr.shape[2] == 1:
                crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
            crop_filename = f"debug_coord_crop_{image_name}_{ts}.png"
            cv2.imwrite(str(debug_dir / crop_filename), crop_bgr)
            crop_url = f"/debug/heatmaps/{crop_filename}"

            # 8. 儲存熱力圖
            heatmap_url = ""
            if anomaly_map is not None:
                if self.heatmap_manager:
                    overlay = self.heatmap_manager.generate_heatmap_overlay(
                        tile_image, anomaly_map, alpha=0.5
                    )
                else:
                    norm_map = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
                    if heatmap_color.shape[:2] != crop_bgr.shape[:2]:
                        heatmap_color = cv2.resize(heatmap_color, (crop_bgr.shape[1], crop_bgr.shape[0]))
                    overlay = cv2.addWeighted(crop_bgr, 0.5, heatmap_color, 0.5, 0)
                hm_filename = f"debug_coord_hm_{image_name}_{ts}.png"
                cv2.imwrite(str(debug_dir / hm_filename), overlay)
                heatmap_url = f"/debug/heatmaps/{hm_filename}"

            # 8. 產生 Overview 圖 (加上裁切框)
            overview_img = image.copy()
            if len(overview_img.shape) == 2:
                overview_img = cv2.cvtColor(overview_img, cv2.COLOR_GRAY2BGR)
            elif len(overview_img.shape) == 3 and overview_img.shape[2] == 1:
                overview_img = cv2.cvtColor(overview_img, cv2.COLOR_GRAY2BGR)
            
            # 使用半透明遮罩凸顯區域
            overlay_bg = overview_img.copy()
            cv2.rectangle(overlay_bg, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay_bg, 0.3, overview_img, 0.7, 0, overview_img)
            # 畫 Otsu 範圍 (黃色框)
            ox1, oy1, ox2, oy2 = otsu_bounds
            cv2.rectangle(overview_img, (ox1, oy1), (ox2, oy2), (0, 255, 255), 4)
            # 畫紅框 + 中心點
            cv2.rectangle(overview_img, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 0, 255), 6)
            cv2.circle(overview_img, (img_cx, img_cy), 10, (0, 255, 0), -1)

            # 在前景有效區域 (Otsu bounds 黃框) 左上與右下標上圖片座標
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            font_thickness = 3
            # 左上座標 (置於黃框上方或內部)
            tl_text = f"({ox1}, {oy1})"
            cv2.putText(overview_img, tl_text, (ox1, max(30, oy1 - 10)), font, font_scale, (0, 255, 255), font_thickness)
            # 右下座標 (置於黃框下方或內部)
            br_text = f"({ox2}, {oy2})"
            cv2.putText(overview_img, br_text, (max(0, ox2 - 300), min(img_h - 10, oy2 + 40)), font, font_scale, (0, 255, 255), font_thickness)

            # 縮小 Overview 存檔 (最多 2000px)
            max_dim = 2000
            oh, ow = overview_img.shape[:2]
            if max(oh, ow) > max_dim:
                scale_o = max_dim / max(oh, ow)
                overview_img = cv2.resize(overview_img, (int(ow * scale_o), int(oh * scale_o)))
            ov_filename = f"debug_coord_overview_{image_name}_{ts}.png"
            cv2.imwrite(str(debug_dir / ov_filename), overview_img)
            overview_url = f"/debug/heatmaps/{ov_filename}"

            # 9. 嘗試尋找並裁切 OMIT 圖片
            omit_url = ""
            image_dir = image_path.parent
            omit_candidates = []
            for pattern in ["PINIGBI*.*", "OMIT0000*.*"]:
                omit_candidates.extend(list(image_dir.glob(pattern)))
            if omit_candidates:
                omit_path = omit_candidates[0]
                omit_full = cv2.imread(str(omit_path), cv2.IMREAD_UNCHANGED)
                if omit_full is not None:
                    try:
                        omit_crop = omit_full[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                        if len(omit_crop.shape) == 2:
                            omit_crop = cv2.cvtColor(omit_crop, cv2.COLOR_GRAY2BGR)
                        elif len(omit_crop.shape) == 3 and omit_crop.shape[2] == 1:
                            omit_crop = cv2.cvtColor(omit_crop, cv2.COLOR_GRAY2BGR)
                        omit_filename = f"debug_coord_omit_{image_name}_{ts}.png"
                        cv2.imwrite(str(debug_dir / omit_filename), omit_crop)
                        omit_url = f"/debug/heatmaps/{omit_filename}"
                    except Exception as e:
                        logger.warning(f"OMIT crop failed: {e}")

            # 9.5 產生組合圖 (Coordinate 推論專屬)
            composite_url = ""
            if self.heatmap_manager:
                try:
                    # 如果有 omit_crop 則放進 tile_info
                    if 'omit_crop' in locals() and omit_crop is not None:
                        tile_info.omit_crop_image = omit_crop
                        
                        # Step A: 進行灰塵偵測
                        is_dust, dust_mask, bright_ratio, detail_text = self.inferencer.check_dust_or_scratch_feature(omit_crop)
                        tile_info.dust_mask = dust_mask
                        tile_info.dust_bright_ratio = bright_ratio
                        
                        # Step B: 計算重疊指標 (Coverage 或 IOU)
                        iou = 0.0
                        heatmap_binary = None
                        top_pct = self.inferencer.config.dust_heatmap_top_percent
                        metric_mode = self.inferencer.config.dust_heatmap_metric
                        metric_name = "COV" if metric_mode == "coverage" else "IOU"
                        
                        if is_dust and anomaly_map is not None:
                            iou, heatmap_binary = self.inferencer.compute_dust_heatmap_iou(
                                dust_mask, anomaly_map, top_percent=top_pct, metric=metric_mode
                            )
                            tile_info.dust_heatmap_iou = iou
                            if iou >= self.inferencer.config.dust_heatmap_iou_threshold:
                                tile_info.is_suspected_dust_or_scratch = True
                                detail_text += f" {metric_name}:{iou:.3f}>={metric_name}_THR -> DUST"
                            else:
                                tile_info.is_suspected_dust_or_scratch = False
                                detail_text += f" {metric_name}:{iou:.3f}<{metric_name}_THR -> REAL_NG"
                            
                            # 產生 Debug 可視化圖
                            try:
                                tile_info.dust_iou_debug_image = self.inferencer.generate_dust_iou_debug_image(
                                    tile_image, anomaly_map, dust_mask,
                                    heatmap_binary, iou, top_pct,
                                    tile_info.is_suspected_dust_or_scratch,
                                )
                            except Exception as dbg_err:
                                logger.warning(f"Debug image generation failed: {dbg_err}")
                        elif is_dust:
                            tile_info.is_suspected_dust_or_scratch = True
                            detail_text += " (no heatmap, marked as dust)"
                        else:
                            tile_info.is_suspected_dust_or_scratch = False
                            detail_text += " NO_DUST"

                        tile_info.dust_detail_text = detail_text
                    
                    composite_path = self.heatmap_manager.save_tile_heatmap(
                        save_dir=debug_dir,
                        image_name=f"coord_{image_name}_{ts}",
                        tile_id=0,
                        tile_image=tile_image,
                        anomaly_map=anomaly_map,
                        score=score,
                        tile_info=tile_info,
                        score_threshold=debug_threshold,
                        iou_threshold=getattr(self.inferencer.config, 'dust_heatmap_iou_threshold', 0.01),
                    )
                    composite_filename = Path(composite_path).name
                    composite_url = f"/debug/heatmaps/{composite_filename}"
                except Exception as comp_err:
                    logger.warning(f"[DEBUG-COORD] 組合圖產生失敗: {comp_err}")

            # 10. 判定結果
            actual_threshold = debug_threshold
            judgment = "NG" if score >= actual_threshold else "OK"

            response_data = {
                "success": True,
                "product_coord": [product_x, product_y],
                "product_resolution": [product_w, product_h],
                "image_coord": [img_cx, img_cy],
                "crop_region": [crop_x1, crop_y1, crop_x2, crop_y2],
                "raw_bounds": list(raw_bounds),
                "scale": [round(scale_x, 4), round(scale_y, 4)],
                "image_size": [img_w, img_h],
                "score": round(score, 4),
                "threshold": actual_threshold,
                "judgment": judgment,
                "processing_time": round(total_time, 3),
                "crop_url": crop_url,
                "heatmap_url": heatmap_url,
                "overview_url": overview_url,
                "omit_url": omit_url,
                "composite_url": composite_url,
                "image_prefix": img_prefix,
                "model_name": model_name,
                "edge_margin_px": edge_margin_override if edge_margin_override is not None else self.inferencer.config.edge_margin_px,
            }

            self._send_json(response_data)
            logger.info(f"[DEBUG-COORD] ({product_x},{product_y})→({img_cx},{img_cy}) "
                        f"Score={score:.4f} {judgment} ({total_time:.2f}s)")

        except Exception as e:
            logger.error(f"[DEBUG-COORD] Error: {e}", exc_info=True)
            self._send_json({"error": f"座標推論失敗: {str(e)}"})

    def _handle_debug_bright_spot_inference(self):
        """API: 黑畫面亮點偵測 — 以指定產品座標為中心裁切 512x512 做 B0F 偵測"""
        import time as _time
        import cv2
        import numpy as np
        from capi_inference import TileInfo

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode('utf-8'))
        except Exception:
            self._send_json({"error": "Invalid JSON body"})
            return

        image_path_str = data.get("image_path", "").strip()
        if not image_path_str:
            self._send_json({"error": "請提供圖片路徑 (image_path)"})
            return

        image_path = Path(image_path_str)
        if not image_path.exists():
            self._send_json({"error": f"檔案不存在: {image_path}"})
            return

        if self.inferencer is None:
            self._send_json({"error": "推論器尚未載入 (inferencer is None)"})
            return

        # 解析座標參數
        try:
            product_x = int(data.get("product_x", 0))
            product_y = int(data.get("product_y", 0))
            product_w = int(data.get("product_w", 1920))
            product_h = int(data.get("product_h", 1080))
        except (ValueError, TypeError) as e:
            self._send_json({"error": f"座標或解析度參數無效: {e}"})
            return

        # 解析 bright_spot 參數覆蓋
        bs_diff_threshold = int(data.get("bs_diff_threshold", self.inferencer.config.bright_spot_diff_threshold))
        bs_median_kernel = int(data.get("bs_median_kernel", self.inferencer.config.bright_spot_median_kernel))
        bs_min_area = int(data.get("bs_min_area", self.inferencer.config.bright_spot_min_area))
        bs_threshold = int(data.get("bs_threshold", self.inferencer.config.bright_spot_threshold))

        try:
            total_start = _time.time()

            # 1. 載入圖片
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                self._send_json({"error": f"無法載入圖片: {image_path}"})
                return

            img_h, img_w = image.shape[:2]

            # 2. 計算 raw_bounds / otsu_bounds
            #    黑圖 Otsu 無法正確偵測邊界，從同資料夾找白圖計算參考邊界
            #    (與 process_panel 中 reference_raw_bounds_for_dark 邏輯一致)
            reference_bounds = None
            ref_image_name = None
            image_dir = image_path.parent
            _DARK_PREFIXES = ("B0F",)
            is_dark = image_path.name.upper().startswith(_DARK_PREFIXES)
            if is_dark:
                # 找同資料夾的白圖 (W0F00000_ 開頭優先，其次任何非 B0F/OMIT/PINIGBI 圖)
                _IMG_EXTS = ('.bmp', '.tif', '.tiff', '.png', '.jpg', '.jpeg')
                all_files = sorted(image_dir.iterdir())
                # 第一輪：優先找 W0F00000_ 開頭
                for candidate in all_files:
                    if not candidate.is_file() or candidate.suffix.lower() not in _IMG_EXTS:
                        continue
                    if candidate.name.upper().startswith("W0F00000"):
                        try:
                            ref_img = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
                            if ref_img is not None:
                                reference_bounds = self.inferencer._find_raw_object_bounds(ref_img)
                                ref_image_name = candidate.name
                                break
                        except Exception:
                            continue
                # 第二輪 fallback：任何非暗色、非 OMIT 圖片
                if reference_bounds is None:
                    for candidate in all_files:
                        if not candidate.is_file() or candidate.suffix.lower() not in _IMG_EXTS:
                            continue
                        cname = candidate.name.upper()
                        if cname.startswith(_DARK_PREFIXES) or cname.startswith("OMIT0000") or cname.startswith("PINIGBI"):
                            continue
                        try:
                            ref_img = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
                            if ref_img is not None:
                                reference_bounds = self.inferencer._find_raw_object_bounds(ref_img)
                                ref_image_name = candidate.name
                                break
                        except Exception:
                            continue
                if reference_bounds is not None:
                    logger.info(f"[DEBUG-BS] 黑圖參考邊界已從 {ref_image_name} 計算 → {reference_bounds}")

            if reference_bounds is not None:
                raw_bounds = reference_bounds
                otsu_bounds, _ = self.inferencer.calculate_otsu_bounds(image, reference_raw_bounds=reference_bounds)
                if otsu_bounds is None:
                    otsu_bounds = raw_bounds
            else:
                raw_bounds = self.inferencer._find_raw_object_bounds(image)
                if raw_bounds is None:
                    raw_bounds = (0, 0, img_w, img_h)
                otsu_bounds, _ = self.inferencer.calculate_otsu_bounds(image)
                if otsu_bounds is None:
                    otsu_bounds = raw_bounds
                if is_dark:
                    logger.warning(f"[DEBUG-BS] 無法找到白圖計算參考邊界，使用自身 Otsu 邊界 (可能不準確)")

            x_start, y_start, x_end, y_end = raw_bounds

            # 3. 產品座標 → 圖片座標
            scale_x = (x_end - x_start) / product_w if product_w > 0 else 1.0
            scale_y = (y_end - y_start) / product_h if product_h > 0 else 1.0
            img_cx = int(product_x * scale_x + x_start)
            img_cy = int(product_y * scale_y + y_start)

            # 4. 以 (img_cx, img_cy) 為中心裁切 512x512
            tile_size = 512
            half = tile_size // 2
            crop_x1 = max(0, img_cx - half)
            crop_y1 = max(0, img_cy - half)
            crop_x2 = crop_x1 + tile_size
            crop_y2 = crop_y1 + tile_size

            if crop_x2 > img_w:
                crop_x2 = img_w
                crop_x1 = max(0, crop_x2 - tile_size)
            if crop_y2 > img_h:
                crop_y2 = img_h
                crop_y1 = max(0, crop_y2 - tile_size)

            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            tile_image = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            # 5. 建立 TileInfo
            tile_info = TileInfo(
                tile_id=0,
                x=crop_x1, y=crop_y1,
                width=crop_w, height=crop_h,
                image=tile_image,
            )

            # 6. 暫時覆蓋 config 值，呼叫 _detect_bright_spots，再還原
            cfg = self.inferencer.config
            orig_diff = cfg.bright_spot_diff_threshold
            orig_kernel = cfg.bright_spot_median_kernel
            orig_area = cfg.bright_spot_min_area
            orig_thr = cfg.bright_spot_threshold
            try:
                cfg.bright_spot_diff_threshold = bs_diff_threshold
                cfg.bright_spot_median_kernel = bs_median_kernel
                cfg.bright_spot_min_area = bs_min_area
                cfg.bright_spot_threshold = bs_threshold
                score, anomaly_map = self.inferencer._detect_bright_spots(tile_info)
            finally:
                cfg.bright_spot_diff_threshold = orig_diff
                cfg.bright_spot_median_kernel = orig_kernel
                cfg.bright_spot_min_area = orig_area
                cfg.bright_spot_threshold = orig_thr

            total_time = _time.time() - total_start

            # 7. 準備暫存目錄
            if CAPIWebHandler._debug_heatmap_dir is None:
                CAPIWebHandler._debug_heatmap_dir = Path(tempfile.mkdtemp(prefix="capi_debug_hm_"))
            debug_dir = CAPIWebHandler._debug_heatmap_dir
            debug_dir.mkdir(parents=True, exist_ok=True)

            image_name = image_path.stem
            ts = int(_time.time() * 1000) % 100000

            # 8. 儲存原始裁切圖
            crop_bgr = tile_image.copy()
            if len(crop_bgr.shape) == 2:
                crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
            elif len(crop_bgr.shape) == 3 and crop_bgr.shape[2] == 1:
                crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
            crop_filename = f"debug_bs_crop_{image_name}_{ts}.png"
            cv2.imwrite(str(debug_dir / crop_filename), crop_bgr)
            crop_url = f"/debug/heatmaps/{crop_filename}"

            # 9. 產生亮點偵測結果圖 (binary overlay)
            detect_url = ""
            if anomaly_map is not None:
                binary_mask = (anomaly_map * 255).astype(np.uint8)
                # 紅色 overlay 標記亮點
                overlay = crop_bgr.copy()
                red_mask = np.zeros_like(overlay)
                red_mask[:, :, 2] = binary_mask  # Red channel
                overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.8, 0)
                # 畫亮點輪廓
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
                detect_filename = f"debug_bs_detect_{image_name}_{ts}.png"
                cv2.imwrite(str(debug_dir / detect_filename), overlay)
                detect_url = f"/debug/heatmaps/{detect_filename}"

            # 10. 產生差異圖 (diff visualization)
            diff_url = ""
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(tile_image.shape) == 3 else tile_image.copy()
            from capi_edge_cv import clamp_median_kernel
            mk = clamp_median_kernel(bs_median_kernel, min(gray.shape[:2]) - 1)
            bg = cv2.medianBlur(gray, mk)
            diff = cv2.subtract(gray, bg)
            diff_color = cv2.applyColorMap(
                cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            diff_filename = f"debug_bs_diff_{image_name}_{ts}.png"
            cv2.imwrite(str(debug_dir / diff_filename), diff_color)
            diff_url = f"/debug/heatmaps/{diff_filename}"

            # 11. 產生 Overview 圖 (加上裁切框)
            overview_img = image.copy()
            if len(overview_img.shape) == 2:
                overview_img = cv2.cvtColor(overview_img, cv2.COLOR_GRAY2BGR)
            elif len(overview_img.shape) == 3 and overview_img.shape[2] == 1:
                overview_img = cv2.cvtColor(overview_img, cv2.COLOR_GRAY2BGR)

            overlay_bg = overview_img.copy()
            cv2.rectangle(overlay_bg, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay_bg, 0.3, overview_img, 0.7, 0, overview_img)
            ox1, oy1, ox2, oy2 = otsu_bounds
            cv2.rectangle(overview_img, (ox1, oy1), (ox2, oy2), (0, 255, 255), 4)
            cv2.rectangle(overview_img, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 0, 255), 6)
            cv2.circle(overview_img, (img_cx, img_cy), 10, (0, 255, 0), -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(overview_img, f"({ox1}, {oy1})", (ox1, max(30, oy1 - 10)), font, 1.5, (0, 255, 255), 3)
            cv2.putText(overview_img, f"({ox2}, {oy2})", (max(0, ox2 - 300), min(img_h - 10, oy2 + 40)), font, 1.5, (0, 255, 255), 3)

            max_dim = 2000
            oh, ow = overview_img.shape[:2]
            if max(oh, ow) > max_dim:
                scale_o = max_dim / max(oh, ow)
                overview_img = cv2.resize(overview_img, (int(ow * scale_o), int(oh * scale_o)))
            ov_filename = f"debug_bs_overview_{image_name}_{ts}.png"
            cv2.imwrite(str(debug_dir / ov_filename), overview_img)
            overview_url = f"/debug/heatmaps/{ov_filename}"

            # 12. 判定結果
            judgment = "NG" if score >= 1.0 else "OK"

            # 從 tile_info 取出偵測統計
            bright_spot_area = getattr(tile_info, 'bright_spot_area', 0)
            bright_spot_max_diff = getattr(tile_info, 'bright_spot_max_diff', 0)

            response_data = {
                "success": True,
                "product_coord": [product_x, product_y],
                "product_resolution": [product_w, product_h],
                "image_coord": [img_cx, img_cy],
                "crop_region": [crop_x1, crop_y1, crop_x2, crop_y2],
                "raw_bounds": list(raw_bounds),
                "otsu_bounds": [ox1, oy1, ox2, oy2],
                "scale": [round(scale_x, 4), round(scale_y, 4)],
                "image_size": [img_w, img_h],
                "score": round(score, 4),
                "judgment": judgment,
                "processing_time": round(total_time, 3),
                "crop_url": crop_url,
                "detect_url": detect_url,
                "diff_url": diff_url,
                "overview_url": overview_url,
                "bright_spot_area": bright_spot_area,
                "bright_spot_max_diff": bright_spot_max_diff,
                "ref_image": ref_image_name,
                "params_used": {
                    "diff_threshold": bs_diff_threshold,
                    "median_kernel": bs_median_kernel,
                    "min_area": bs_min_area,
                    "threshold": bs_threshold,
                },
            }

            self._send_json(response_data)
            logger.info(f"[DEBUG-BS] ({product_x},{product_y})→({img_cx},{img_cy}) "
                        f"Score={score:.1f} {judgment} area={bright_spot_area} ({total_time:.2f}s)")

        except Exception as e:
            logger.error(f"[DEBUG-BS] Error: {e}", exc_info=True)
            self._send_json({"error": f"黑畫面推論失敗: {str(e)}"})

    def _handle_debug_heatmap_file(self, path: str):
        """靜態檔案服務 (Debug 推論熱力圖)"""
        if self._debug_heatmap_dir is None:
            self._send_404()
            return
        rel_path = path[len("/debug/heatmaps/"):]
        rel_path = rel_path.replace("..", "").lstrip("/")
        full_path = self._debug_heatmap_dir / rel_path
        if full_path.exists() and full_path.is_file():
            self._send_binary(str(full_path))
        else:
            self._send_404()

    # ── 設定管理功能 ─────────────────────────────────────

    def _handle_settings_page(self, path: str):
        """設定管理頁面 (舊版)"""
        template = self.jinja_env.get_template("settings.html")
        html = template.render(request_path=path)
        self._send_response(200, html)

    def _handle_settings_v2_page(self, path: str):
        """設定管理頁面 (新版 V2)"""
        template = self.jinja_env.get_template("settings_v2.html")
        html = template.render(request_path=path)
        self._send_response(200, html)

    def _handle_api_settings(self):
        """API: 取得所有設定參數"""
        try:
            params = self.db.get_all_config_params() if self.db else []
            # 補上 config 中有但 DB 沒有的參數（用目前執行值作為預設）
            if self.inferencer and hasattr(self.inferencer, 'config') and self.inferencer.config:
                existing_names = {p["param_name"] for p in params}
                import dataclasses
                for f in dataclasses.fields(self.inferencer.config):
                    if f.name in existing_names:
                        continue
                    val = getattr(self.inferencer.config, f.name)
                    # 只處理 JSON 可序列化的基本型別
                    if isinstance(val, (str, int, float, bool)):
                        val_str = str(val)
                    elif isinstance(val, (dict, list)):
                        try:
                            val_str = json.dumps(val)
                        except (TypeError, ValueError):
                            continue
                    else:
                        continue
                    params.append({
                        "param_name": f.name,
                        "param_value": val_str,
                        "updated_at": None,
                    })
            # 附帶 model_resolution_map 給前端產品選擇器使用
            resolution_map = {}
            if self.inferencer and hasattr(self.inferencer, 'config') and self.inferencer.config:
                resolution_map = getattr(self.inferencer.config, 'model_resolution_map', {})
            self._send_json({"params": params, "model_resolution_map": resolution_map})
        except Exception as e:
            self._send_json({"error": str(e)})

    def _handle_api_settings_update(self):
        """API: 更新設定參數"""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)

            param_name = data.get("param_name", "")
            new_value = data.get("new_value")
            reason = data.get("reason", "")

            if not param_name:
                self._send_json({"error": "缺少 param_name"})
                return
            if new_value is None:
                self._send_json({"error": "缺少 new_value"})
                return
            if not reason.strip():
                self._send_json({"error": "請填寫修改原因"})
                return

            success = self.db.update_config_param(param_name, new_value, reason)
            if success:
                # Hot-reload edge config if a cv_edge_* parameter was updated
                if param_name.startswith("cv_edge") and hasattr(self, 'inferencer') and self.inferencer:
                    try:
                        from capi_edge_cv import EdgeInspectionConfig
                        db_params = {r["param_name"]: r for r in self.db.get_all_config_params()}
                        edge_cfg = EdgeInspectionConfig.from_db_params(db_params)
                        self.inferencer.update_edge_config(edge_cfg)
                        logger.info(f"[Edge Hot-Reload] CV Edge config synced after updating '{param_name}'")
                    except Exception as e:
                        logger.warning(f"[Edge Hot-Reload] Failed to sync edge config: {e}")
                self._send_json({"success": True, "message": f"已更新 {param_name}"})
            else:
                self._send_json({"error": f"找不到參數: {param_name}"})
        except json.JSONDecodeError:
            self._send_json({"error": "無效的 JSON 格式"})
        except Exception as e:
            self._send_json({"error": str(e)})

    def _handle_api_settings_history(self, query: dict):
        """API: 取得設定修改紀錄"""
        try:
            param_name = query.get("param_name", [""])[0]
            limit = int(query.get("limit", [50])[0])
            history = self.db.get_config_change_history(param_name, limit) if self.db else []
            self._send_json({"history": history})
        except Exception as e:
            self._send_json({"error": str(e)})

    def _handle_api_settings_reload(self):
        """API: 重新載入設定 (Hot-reload inferencer)"""
        try:
            if not self._capi_server_instance:
                self._send_json({"error": "Server 實例未設定，無法重載"})
                return

            server_inst = self._capi_server_instance
            gpu_lock = self._gpu_lock

            logger.info("Settings reload: re-initializing inferencer from DB config...")

            # 使用 GPU lock 阻止推論期間重建
            if gpu_lock:
                gpu_lock.acquire()

            try:
                server_inst._load_inferencer()
                # 更新 Web handler 的 inferencer 參照
                CAPIWebHandler.inferencer = server_inst.inferencer
                logger.info("Settings reload: inferencer re-initialized successfully")
                self._send_json({"success": True, "message": "設定已重新載入，推論器已重建"})
            finally:
                if gpu_lock:
                    gpu_lock.release()

        except Exception as e:
            logger.error(f"Settings reload failed: {e}", exc_info=True)
            self._send_json({"error": f"重載失敗: {str(e)}"})

    # ── Rerun inference endpoints ──────────────────────────────────────

    def _handle_rerun_trigger(self, record_id_str: str):
        """API: 觸發重新推論"""
        try:
            record_id = int(record_id_str)
        except ValueError:
            self._send_json({"status": "error", "message": "invalid record_id"})
            return

        if not self.inferencer:
            self._send_json({"status": "error", "message": "推論器未載入"})
            return

        with self._rerun_lock:
            task = self._rerun_tasks.get(record_id)
            if task and task["status"] == "running":
                self._send_json({"status": "already_running"})
                return

        detail = self.db.get_record_detail(record_id) if self.db else None
        if not detail:
            self._send_json({"status": "error", "message": f"找不到紀錄 #{record_id}"})
            return

        image_dir = detail.get("image_dir", "")
        if not image_dir or not Path(image_dir).is_dir():
            self._send_json({"status": "error", "message": f"圖片目錄不存在: {image_dir}"})
            return

        with self._rerun_lock:
            # 再次檢查防止 TOCTOU race
            task = self._rerun_tasks.get(record_id)
            if task and task["status"] == "running":
                self._send_json({"status": "already_running"})
                return
            self._rerun_tasks[record_id] = {"status": "running", "message": "正在準備推論..."}

        thread = threading.Thread(
            target=CAPIWebHandler._rerun_worker,
            args=(record_id, detail),
            daemon=True,
        )
        thread.start()

        self._send_json({"status": "started", "record_id": record_id})

    @classmethod
    def _rerun_worker(cls, record_id: int, detail: dict):
        """背景執行緒：重新推論並覆蓋紀錄"""
        import time as _time
        from capi_server import results_to_db_data, aggregate_judgment, append_cv_edge_to_judgment

        def _update_status(msg, *_):
            with cls._rerun_lock:
                task = cls._rerun_tasks.get(record_id)
                if task:
                    if isinstance(msg, int) and _:
                        task["message"] = f"推論中 {msg}/{_[0]}..."
                    else:
                        task["message"] = str(msg)

        try:
            panel_dir = Path(detail["image_dir"])
            model_id = detail.get("model_id", "")
            resolution = None
            if detail.get("resolution_x") and detail.get("resolution_y"):
                resolution = (detail["resolution_x"], detail["resolution_y"])

            bomb_info = None
            if detail.get("client_bomb_info"):
                try:
                    bomb_info = json.loads(detail["client_bomb_info"])
                except (json.JSONDecodeError, TypeError):
                    pass

            _update_status("正在等待 GPU...")
            start_time = _time.time()

            if cls._gpu_lock:
                with cls._gpu_lock:
                    _update_status("正在推論中...")
                    panel_result = cls.inferencer.process_panel(
                        panel_dir,
                        progress_callback=_update_status,
                        product_resolution=resolution,
                        bomb_info=bomb_info,
                        model_id=model_id,
                    )
            else:
                _update_status("正在推論中...")
                panel_result = cls.inferencer.process_panel(
                    panel_dir,
                    progress_callback=_update_status,
                    product_resolution=resolution,
                    bomb_info=bomb_info,
                    model_id=model_id,
                )

            processing_seconds = _time.time() - start_time

            results = panel_result[0]
            omit_overexposed = panel_result[2] if len(panel_result) > 2 else False
            omit_overexposure_info = panel_result[3] if len(panel_result) > 3 else ""
            omit_image_raw = panel_result[5] if len(panel_result) > 5 else None

            if not results:
                with cls._rerun_lock:
                    cls._rerun_tasks[record_id] = {"status": "error", "message": "推論完成但無圖片結果"}
                return

            ai_judgment, ng_details = aggregate_judgment(results)
            for result in results:
                if hasattr(result, 'edge_defects') and result.edge_defects:
                    ai_judgment, ng_details = append_cv_edge_to_judgment(
                        ai_judgment, ng_details, result.edge_defects, result.image_path.stem
                    )

            _update_status("正在儲存 heatmap...")
            heatmap_info = {}
            if cls.heatmap_manager:
                old_heatmap_dir = detail.get("heatmap_dir", "")
                if old_heatmap_dir and Path(old_heatmap_dir).is_dir():
                    import shutil
                    try:
                        shutil.rmtree(old_heatmap_dir)
                    except Exception:
                        pass

                heatmap_info = cls.heatmap_manager.save_panel_heatmaps(
                    glass_id=detail["glass_id"],
                    results=results,
                    inferencer=cls.inferencer,
                    save_overview=True,
                    save_tile_detail=True,
                    omit_image=omit_image_raw,
                )

            _update_status("正在更新資料庫...")
            image_results_data = results_to_db_data(results, heatmap_info) if results else []
            total_images = len(image_results_data)
            ng_images = sum(1 for d in image_results_data if d.get("is_ng"))
            error_message = ai_judgment if ai_judgment.startswith("ERR") else ""

            inference_log = ""
            if hasattr(cls.inferencer, '_log_capture') and cls.inferencer._log_capture:
                try:
                    inference_log = cls.inferencer._log_capture.get_log()
                except Exception:
                    pass

            cls.db.update_record_for_rerun(
                record_id=record_id,
                ai_judgment=ai_judgment,
                total_images=total_images,
                ng_images=ng_images,
                ng_details=ng_details,
                processing_seconds=processing_seconds,
                heatmap_dir=heatmap_info.get("dir", ""),
                error_message=error_message,
                image_results_data=image_results_data,
                inference_log=inference_log,
                omit_overexposed=int(omit_overexposed),
                omit_overexposure_info=omit_overexposure_info if omit_overexposure_info else "",
            )

            with cls._rerun_lock:
                cls._rerun_tasks[record_id] = {"status": "done", "message": "完成"}

        except Exception as e:
            import traceback
            traceback.print_exc()
            with cls._rerun_lock:
                cls._rerun_tasks[record_id] = {"status": "error", "message": f"推論失敗: {e}"}

    # ==== Dataset Export Endpoints ====

    @classmethod
    def _dataset_export_worker(cls, days: int, include_true_ng: bool,
                               skip_existing: bool, output_dir: str):
        """背景執行 DatasetExporter.run()"""
        state = cls._dataset_export_state
        try:
            server_inst = cls._capi_server_instance
            path_mapping = getattr(server_inst, "path_mapping", {}) if server_inst else {}

            def status_callback(current, total, last_glass_id):
                with state["lock"]:
                    if state["current_job"]:
                        state["current_job"]["current"] = current
                        state["current_job"]["total"] = total
                        state["current_job"]["last_glass_id"] = last_glass_id

            exporter = DatasetExporter(
                db=cls.db, base_dir=output_dir, path_mapping=path_mapping,
            )
            summary = exporter.run(
                days=days, include_true_ng=include_true_ng,
                skip_existing=skip_existing,
                status_callback=status_callback,
                cancel_event=state["cancel_event"],
            )
            with state["lock"]:
                if state["current_job"]:
                    state["current_job"]["state"] = (
                        JOB_STATE_CANCELLED if state["cancel_event"].is_set() else JOB_STATE_COMPLETED
                    )
                state["last_summary"] = summary
        except Exception as e:
            logger.exception("dataset_export worker failed")
            with state["lock"]:
                if state["current_job"]:
                    state["current_job"]["state"] = JOB_STATE_FAILED
                    state["current_job"]["error"] = str(e)
        finally:
            with state["lock"]:
                state["cancel_event"].clear()

    def _handle_dataset_export_start(self):
        """POST /api/dataset_export/start"""
        import json as _json
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            data = _json.loads(body) if body else {}
        except Exception:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return

        days = int(data.get("days", 3))
        include_true_ng = bool(data.get("include_true_ng", True))
        skip_existing = bool(data.get("skip_existing", True))

        # 決定 output_dir
        server_inst = self._capi_server_instance
        default_cfg = {}
        if server_inst:
            default_cfg = server_inst.server_config.get("dataset_export", {})
        output_dir = data.get("output_dir") or default_cfg.get("base_dir") or "./datasets/over_review"
        min_free_gb = float(default_cfg.get("min_free_space_gb", 1))

        state = self._dataset_export_state
        with state["lock"]:
            if state["current_job"] and state["current_job"].get("state") == JOB_STATE_RUNNING:
                self._send_json({
                    "error": "job_already_running",
                    "current_job_id": state["current_job"].get("job_id"),
                }, status=409)
                return

            # 磁碟空間檢查
            try:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                free_bytes = shutil.disk_usage(str(output_path)).free
                if free_bytes < min_free_gb * (1024 ** 3):
                    self._send_json({
                        "error": "insufficient_disk_space",
                        "free_gb": round(free_bytes / (1024 ** 3), 2),
                        "required_gb": min_free_gb,
                    }, status=409)
                    return
            except Exception as e:
                self._send_json({"error": f"cannot access output_dir: {e}"}, status=400)
                return

            job_id = datetime.now().strftime("job_%Y%m%d_%H%M%S")
            state["cancel_event"].clear()
            state["current_job"] = {
                "job_id": job_id,
                "state": JOB_STATE_RUNNING,
                "current": 0,
                "total": 0,
                "last_glass_id": "",
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }

        thread = threading.Thread(
            target=CAPIWebHandler._dataset_export_worker,
            args=(days, include_true_ng, skip_existing, output_dir),
            daemon=True,
            name=f"dataset-export-{job_id}",
        )
        thread.start()
        self._send_json({"job_id": job_id, "started_at": state["current_job"]["started_at"]})

    def _handle_dataset_export_status(self):
        """GET /api/dataset_export/status"""
        state = self._dataset_export_state
        with state["lock"]:
            job = state["current_job"]
            if not job:
                self._send_json({"state": JOB_STATE_IDLE})
                return
            resp = dict(job)
            if resp.get("started_at"):
                try:
                    started = datetime.fromisoformat(resp["started_at"])
                    resp["elapsed_sec"] = round((datetime.now() - started).total_seconds(), 1)
                except Exception:
                    resp["elapsed_sec"] = 0
            self._send_json(resp)

    def _handle_dataset_export_summary(self, job_id: str):
        """GET /api/dataset_export/summary/<job_id>"""
        from dataclasses import asdict as _asdict
        state = self._dataset_export_state
        with state["lock"]:
            summary = state["last_summary"]
            if not summary or summary.job_id != job_id:
                self._send_json({"error": "not_found"}, status=404)
                return
            self._send_json(_asdict(summary))

    def _handle_dataset_export_cancel(self):
        """POST /api/dataset_export/cancel"""
        state = self._dataset_export_state
        with state["lock"]:
            if not state["current_job"] or state["current_job"].get("state") != JOB_STATE_RUNNING:
                self._send_json({"error": "no_running_job"}, status=404)
                return
            state["cancel_event"].set()
        self._send_json({"ok": True})

    def _handle_rerun_status_sse(self, record_id_str: str):
        """SSE: 串流重跑進度"""
        import time as _time

        try:
            record_id = int(record_id_str)
        except ValueError:
            self._send_json({"status": "error", "message": "invalid record_id"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def sse_send(event_type, data):
            msg = f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            self.wfile.write(msg.encode("utf-8"))
            self.wfile.flush()

        last_msg = ""
        try:
            while True:
                with self._rerun_lock:
                    task = self._rerun_tasks.get(record_id)

                if not task:
                    sse_send("status", {"message": "idle"})
                    break

                if task["message"] != last_msg:
                    sse_send("status", {"message": task["message"]})
                    last_msg = task["message"]

                if task["status"] == "done":
                    sse_send("done", {"message": "完成", "record_id": record_id})
                    with self._rerun_lock:
                        self._rerun_tasks.pop(record_id, None)
                    break
                elif task["status"] == "error":
                    sse_send("error", {"message": task["message"]})
                    with self._rerun_lock:
                        self._rerun_tasks.pop(record_id, None)
                    break

                _time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def create_web_server(
    host: str,
    port: int,
    db,
    heatmap_base_dir: str,
    status_tracker=None,
    inferencer=None,
    heatmap_manager=None,
    gpu_lock=None,
    capi_server_instance=None,
    log_file=None,
) -> ThreadingHTTPServer:
    """
    建立 Web 伺服器

    Args:
        host: 綁定地址
        port: 綁定端口
        db: CAPIDatabase 實例
        heatmap_base_dir: 熱力圖儲存根目錄
        status_tracker: 伺服器狀態追蹤物件
        inferencer: CAPIInferencer 實例 (Optional, for debug inference)
        heatmap_manager: HeatmapManager 實例 (Optional, for debug inference)
        gpu_lock: GPU 排隊鎖 (Optional, for debug inference)
        capi_server_instance: CAPIServer 實例 (Optional, for config hot-reload)
        log_file: 日誌檔案路徑 (Optional, for log viewer)
    """
    CAPIWebHandler.db = db
    CAPIWebHandler.heatmap_base_dir = heatmap_base_dir
    CAPIWebHandler.status_tracker = status_tracker
    CAPIWebHandler.inferencer = inferencer
    CAPIWebHandler.heatmap_manager = heatmap_manager
    CAPIWebHandler._gpu_lock = gpu_lock
    CAPIWebHandler._capi_server_instance = capi_server_instance
    CAPIWebHandler._log_file = log_file
    CAPIWebHandler._rerun_tasks = {}
    CAPIWebHandler._rerun_lock = threading.Lock()
    CAPIWebHandler._dataset_export_state = {
        "lock": threading.Lock(),
        "current_job": None,         # dict: job_id, state, current, total, last_glass_id, started_at
        "cancel_event": threading.Event(),
        "last_summary": None,        # JobSummary instance
    }
    CAPIWebHandler.init_jinja()

    server = ThreadingHTTPServer((host, port), CAPIWebHandler)
    return server


def start_web_server_thread(
    host: str,
    port: int,
    db,
    heatmap_base_dir: str,
    status_tracker=None,
    inferencer=None,
    heatmap_manager=None,
    gpu_lock=None,
    capi_server_instance=None,
    log_file=None,
) -> threading.Thread:
    """
    在背景執行緒啟動 Web 伺服器

    Returns:
        Web 伺服器執行緒
    """
    server = create_web_server(
        host, port, db, heatmap_base_dir, status_tracker,
        inferencer=inferencer,
        heatmap_manager=heatmap_manager,
        gpu_lock=gpu_lock,
        capi_server_instance=capi_server_instance,
        log_file=log_file,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="web-server")
    thread.start()
    logger.info(f"Web server started on http://{host}:{port}")
    return thread


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from capi_database import CAPIDatabase

    logging.basicConfig(level=logging.INFO)

    # 使用 test_results.db 測試 (含真實推論資料)
    db_path = os.path.join(str(Path(__file__).parent), "test_results.db")
    db = CAPIDatabase(db_path)

    # 熱力圖目錄
    test_heatmap_dir = os.path.join(str(Path(__file__).parent), "test_heatmaps")

    print(f"Test DB: {db_path}")
    print(f"Starting web server on http://localhost:8080")
    print("Press Ctrl+C to stop")

    server = create_web_server("0.0.0.0", 8080, db, test_heatmap_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        print("\nStopped.")


