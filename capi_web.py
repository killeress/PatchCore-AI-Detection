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
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional
import threading
import logging
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger("capi.web")

# 幫 Jinja2 準備一些好用的過濾器
def ai_simple(ai_judgment):
    if not ai_judgment: return ""
    return "OK" if ai_judgment == "OK" else ("NG" if ai_judgment.startswith("NG") else ("ERR" if ai_judgment.startswith("ERR") else ai_judgment))

def ai_badge(ai_judgment):
    simple = ai_simple(ai_judgment)
    return "badge-ok" if simple == "OK" else ("badge-ng" if simple == "NG" else "badge-err")

def mj_badge(machine_judgment):
    return "badge-ok" if machine_judgment == "OK" else "badge-ng"

def img_status_info(img):
    if img.get("is_dust_only"): return "灰塵 (DUST)", "badge-err"
    if img.get("is_bomb"): return "炸彈 (BOMB)", "badge-err"
    if img["is_ng"]: return "NG", "badge-ng"
    return "OK", "badge-ok"

def tile_info(t):
    badge = "badge-ng"
    info = f"Score: {t['score']:.3f}"
    if t.get("is_dust"):
        badge = "badge-err"
        info += f" | 灰塵 IOU: {t.get('dust_iou',0):.3f}"
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
            elif path.startswith("/v3/record/"):
                record_id = path.split("/v3/record/")[1].rstrip("/")
                self._handle_record_detail_v3(record_id, path)
            elif path.startswith("/record/"):
                record_id = path.split("/record/")[1].rstrip("/")
                self._handle_record_detail(record_id, path)
            elif path == "/search":
                self._handle_search(query, path)
            elif path == "/search/export":
                self._handle_search_export(query)
            elif path == "/debug":
                self._handle_debug_page(path)
            elif path == "/ric":
                self._handle_ric_page(query, path)
            elif path == "/api/ric/report":
                self._handle_ric_report_api(query)
            elif path == "/api/stats":
                self._handle_api_stats(query)
            elif path == "/api/status":
                self._handle_api_status()
            elif path.startswith("/heatmaps/"):
                self._handle_static_file(path)
            elif path.startswith("/debug/heatmaps/"):
                self._handle_debug_heatmap_file(path)
            elif path.startswith("/images/"):
                self._handle_source_image(path)
            elif path.startswith("/imgs/"):
                self._handle_imgs_file(path)
            elif path.startswith("/static/"):
                self._handle_static_assets(path)
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
            elif path == "/api/ric/upload":
                self._handle_ric_upload()
            elif path == "/api/ric/delete":
                self._handle_ric_delete()
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

    def _send_json(self, data):
        content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        self._send_response(200, content, "application/json; charset=utf-8")

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

    def _handle_record_detail_v3(self, record_id_str: str, path: str):
        """記錄詳情頁 (v3 UI)"""
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

    def _handle_search(self, query: dict, path: str):
        """搜尋頁面（含日期篩選）"""
        glass_id = query.get("glass_id", [""])[0]
        machine_no = query.get("machine_no", [""])[0]
        ai_judgment = query.get("ai_judgment", [""])[0]
        start_date = query.get("start_date", [""])[0]  # 格式 YYYY-MM-DD
        end_date = query.get("end_date", [""])[0]      # 格式 YYYY-MM-DD

        # end_date 加上時間尾端，確保涵蓋當天全天
        end_date_full = f"{end_date} 23:59:59" if end_date else ""

        records = []
        has_search = False
        if any([glass_id, machine_no, ai_judgment, start_date, end_date]):
            has_search = True
            records = self.db.search_records(
                glass_id=glass_id,
                machine_no=machine_no,
                ai_judgment=ai_judgment,
                start_date=start_date,
                end_date=end_date_full,
                limit=500,
            ) if self.db else []

        template = self.jinja_env.get_template("search.html")
        html = template.render(
            glass_id=glass_id,
            machine_no=machine_no,
            ai_judgment=ai_judgment,
            start_date=start_date,
            end_date=end_date,
            records=records,
            has_search=has_search,
            request_path=path
        )
        self._send_response(200, html)

    def _handle_search_export(self, query: dict):
        """匯出搜尋結果為 CSV"""
        import csv
        import io
        from datetime import datetime as _dt

        glass_id   = query.get("glass_id",    [""])[0]
        machine_no = query.get("machine_no",  [""])[0]
        ai_judgment = query.get("ai_judgment", [""])[0]
        start_date = query.get("start_date",  [""])[0]
        end_date   = query.get("end_date",    [""])[0]
        end_date_full = f"{end_date} 23:59:59" if end_date else ""

        records = self.db.search_records(
            glass_id=glass_id,
            machine_no=machine_no,
            ai_judgment=ai_judgment,
            start_date=start_date,
            end_date=end_date_full,
            limit=10000,
        ) if self.db else []

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
                            """SELECT tile_id, heatmap_path, is_anomaly, is_dust, is_bomb 
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
                                
                                tile_status = "OK"
                                if is_bomb:
                                    tile_status = "BOMB"
                                elif is_dust:
                                    tile_status = "DUST"
                                elif is_ng:
                                    tile_status = "NG"

                                recent_heatmaps.append({
                                    "url":        t_url,
                                    "glass_id":   row["glass_id"] or "",
                                    "image_name": row["image_name"] or "",
                                    "is_ng":      is_ng or is_bomb or is_dust,
                                    "status":     tile_status,
                                    "label":      f"Tile #{t['tile_id']}",
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
        days = int(query.get("days", [7])[0])
        limit = int(query.get("limit", [15])[0])
        
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

    def _handle_debug_page(self, path: str):
        """Debug 推論頁面"""
        # 從正在運行的推論器讀取 config 預設值
        default_threshold = 0.5
        default_edge_margin = 256
        if self.inferencer and hasattr(self.inferencer, 'config'):
            default_threshold = self.inferencer.config.anomaly_threshold
            default_edge_margin = self.inferencer.config.edge_margin_px
        
        template = self.jinja_env.get_template("debug_inference.html")
        html = template.render(
            request_path=path,
            default_threshold=default_threshold,
            default_edge_margin=default_edge_margin,
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

    def _handle_ric_report_api(self, query: dict):
        """API: 取得 RIC 比對報表資料"""
        try:
            batch_id_str = query.get('batch_id', [''])[0]
            batch_id = int(batch_id_str) if batch_id_str else None
        except (ValueError, TypeError):
            batch_id = None

        stats = self.db.get_ric_accuracy_stats(batch_id) if self.db else {}
        self._send_json(stats)

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
        edge_margin_override = int(edge_margin_raw) if edge_margin_raw is not None else None

        try:
            total_start = _time.time()

            # 1. 預處理（不需要 GPU，也不用 threshold）
            result = self.inferencer.preprocess_image(image_path)
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
            if hasattr(self, '_gpu_lock') and self._gpu_lock:
                with self._gpu_lock:
                    result = self.inferencer.run_inference(
                        result,
                        inferencer=target_inferencer,
                        threshold=debug_threshold,
                        edge_margin_override=edge_margin_override,
                    )
            else:
                result = self.inferencer.run_inference(
                    result,
                    inferencer=target_inferencer,
                    threshold=debug_threshold,
                    edge_margin_override=edge_margin_override,
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

            # 5. 產生各 Tile 熱力圖
            tiles_data = []
            for tile, score, anomaly_map in result.anomaly_tiles:
                # 產生 heatmap overlay
                tile_filename = f"debug_tile_{image_name}_t{tile.tile_id}.png"
                tile_hm_path = debug_dir / tile_filename

                # 使用 HeatmapManager 的 overlay 方法
                if self.heatmap_manager:
                    overlay = self.heatmap_manager.generate_heatmap_overlay(
                        tile.image, anomaly_map, alpha=0.5
                    )
                else:
                    # Fallback: 簡易 heatmap
                    tile_img = tile.image.copy()
                    if len(tile_img.shape) == 2:
                        tile_img = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2BGR)
                    norm_map = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
                    if heatmap_color.shape[:2] != tile_img.shape[:2]:
                        heatmap_color = cv2.resize(heatmap_color, (tile_img.shape[1], tile_img.shape[0]))
                    overlay = cv2.addWeighted(tile_img, 0.5, heatmap_color, 0.5, 0)

                cv2.imwrite(str(tile_hm_path), overlay)
                tile_url = f"/debug/heatmaps/{tile_filename}"

                tile_status = "NG"
                if tile.is_bomb:
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
                    "dust_iou": round(tile.dust_heatmap_iou, 4),
                    "is_bomb": tile.is_bomb,
                    "bomb_code": tile.bomb_defect_code,
                    "heatmap_url": tile_url,
                })

            # 6. 判定結果
            has_real_ng = any(
                not t.is_suspected_dust_or_scratch and not t.is_bomb
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


def create_web_server(
    host: str,
    port: int,
    db,
    heatmap_base_dir: str,
    status_tracker=None,
    inferencer=None,
    heatmap_manager=None,
    gpu_lock=None,
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
    """
    CAPIWebHandler.db = db
    CAPIWebHandler.heatmap_base_dir = heatmap_base_dir
    CAPIWebHandler.status_tracker = status_tracker
    CAPIWebHandler.inferencer = inferencer
    CAPIWebHandler.heatmap_manager = heatmap_manager
    CAPIWebHandler._gpu_lock = gpu_lock
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

