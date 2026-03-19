"""
CAPI AI 推論伺服器

TCP/IP Socket Server，接收 Testing 客戶端的推論請求，
呼叫 CAPIInferencer 進行 AI 推論，並回覆判定結果。

通訊協議:
    [Request]  無炸彈: AOI@玻璃ID;機種ID;機台編號;解析度X,解析度Y;機檢判定;圖片目錄路徑
               有炸彈: AOI@玻璃ID;機種ID;機台編號;解析度X,解析度Y;機檢判定;圖片前綴;(座標);圖片目錄路徑
               機檢判定: OK / NG / HY (畫異，HY 時跳過推論直接回傳 ERR:HY)
    [Response] AOI@玻璃ID;機種ID;機台編號;機檢判定;AI判定

AI 判定:
    OK                           — 正常
    NG@圖片名(X,Y)|圖片名(X,Y)  — 異常，附帶座標
    ERR:錯誤描述                 — 錯誤

啟動方式:
    python3 capi_server.py --config server_config.yaml
"""

import os
# 設置環境變數 (必須在 import anomalib 之前)
os.environ["TRUST_REMOTE_CODE"] = "1"

import sys
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import time
import logging
import logging.handlers
import argparse
import signal
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Any

# 確保模組路徑
sys.path.insert(0, str(Path(__file__).parent))

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer, ImageResult
from capi_database import CAPIDatabase
from capi_heatmap import HeatmapManager
from capi_web import start_web_server_thread


# ── 伺服器狀態追蹤 ──────────────────────────────────────────

class ServerStatusTracker:
    """追蹤伺服器即時運作狀態，供 Web 即時儀表板使用"""
    def __init__(self):
        self.lock = threading.Lock()
        
        # 基本狀態
        self.is_running = False
        self.start_time = None
        self.model_version = "Unknown"
        self.threshold = 0.6
        self.threshold_mapping = {}  # {prefix: threshold}
        self.device = "CPU"
        
        # 連線與推論狀態
        self.active_connections = 0
        self.connected_machines = set()  # "client_ip:port (machine_no)"
        self.active_inferences = 0
        
        # 統計數據 (當前 Session)
        self.total_requests = 0
        self.total_ng = 0
        self.total_ok = 0
        self.total_err = 0
        self.last_inference_time = None
        self.last_judgment_result = None  # 最近一筆判定結果 {glass_id, ai_judgment, time, duration}
        
    def get_status(self):
        """取得即時狀態 JSON Object"""
        with self.lock:
            # 計算 Uptime
            uptime_str = "0s"
            if self.is_running and self.start_time:
                diff = int(time.time() - self.start_time)
                h, r = divmod(diff, 3600)
                m, s = divmod(r, 60)
                uptime_str = f"{h}h {m}m {s}s"
                
            return {
                "server": {
                    "running": self.is_running,
                    "uptime": uptime_str,
                    "model_version": self.model_version,
                    "device": self.device,
                    "threshold": self.threshold,
                    "threshold_mapping": dict(self.threshold_mapping),
                },
                "traffic": {
                    "active_connections": self.active_connections,
                    "connected_machines": list(self.connected_machines),
                    "active_inferences": self.active_inferences,
                },
                "stats": {
                    "total_requests": self.total_requests,
                    "total_ok": self.total_ok,
                    "total_ng": self.total_ng,
                    "total_err": self.total_err,
                },
                "latest_event": self.last_judgment_result,
            }

# 全域狀態單例
server_status = ServerStatusTracker()


# ── 日誌設定 ──────────────────────────────────────────

logger = logging.getLogger("capi.server")


def setup_logging(config: dict):
    """設定日誌系統"""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("file", "")
    max_bytes = log_cfg.get("max_bytes", 10 * 1024 * 1024)
    backup_count = log_cfg.get("backup_count", 5)

    # 根 logger
    root = logging.getLogger("capi")
    root.setLevel(level)

    # 格式
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)


# ── 協議解析 ──────────────────────────────────────────

class ProtocolError(Exception):
    """協議解析錯誤"""
    pass


def _parse_bomb_coordinates(image_prefix: str, coords_raw: str) -> Optional[Dict[str, Any]]:
    """
    解析炸彈座標字串

    格式:
        點型: (x1/y1;x2/y2;x3/y3;...)   — 多組 x/y 用分號分隔
        線型: (x1/y1/x2/y2)             — 一組含 4 個值

    區分邏輯:
        括號內只有一組且含 4 個值 → 線型 (line)
        其他 → 點型 (point)

    Returns:
        {
            "image_prefix": str,
            "defect_type": "point" | "line",
            "coordinates": [(x, y), ...]
        }
        若解析失敗返回 None
    """
    # 去除首尾括號和空白
    coords_str = coords_raw.strip()
    if coords_str.startswith("(") and coords_str.endswith(")"):
        coords_str = coords_str[1:-1]

    if not coords_str:
        return None

    # 用分號分隔各組座標
    groups = [g.strip() for g in coords_str.split(";") if g.strip()]

    if len(groups) == 1:
        # 只有一組 — 檢查是否為線型 (4 個值: x1/y1/x2/y2)
        values = groups[0].split("/")
        if len(values) == 4:
            try:
                x1, y1, x2, y2 = int(values[0]), int(values[1]), int(values[2]), int(values[3])
                return {
                    "image_prefix": image_prefix,
                    "defect_type": "line",
                    "coordinates": [(x1, y1), (x2, y2)],
                }
            except ValueError:
                logger.warning(f"炸彈座標解析失敗 (line): {coords_raw}")
                return None
        elif len(values) == 2:
            # 單一點
            try:
                x, y = int(values[0]), int(values[1])
                return {
                    "image_prefix": image_prefix,
                    "defect_type": "point",
                    "coordinates": [(x, y)],
                }
            except ValueError:
                logger.warning(f"炸彈座標解析失敗 (single point): {coords_raw}")
                return None
        else:
            logger.warning(f"炸彈座標格式不明: {coords_raw}")
            return None
    else:
        # 多組 — 點型 (每組 x/y)
        coordinates = []
        for g in groups:
            values = g.split("/")
            if len(values) == 2:
                try:
                    coordinates.append((int(values[0]), int(values[1])))
                except ValueError:
                    logger.warning(f"炸彈座標解析失敗 (point group): {g}")
                    continue
            else:
                logger.warning(f"炸彈座標格式不明 (group): {g}")
                continue

        if not coordinates:
            return None

        return {
            "image_prefix": image_prefix,
            "defect_type": "point",
            "coordinates": coordinates,
        }


def parse_request(data: str) -> Dict[str, Any]:
    """
    解析客戶端請求

    格式:
        無炸彈 (6 欄位): AOI@玻璃ID;機種ID;機台編號;解析度X,解析度Y;機檢判定;圖片目錄路徑
        有炸彈 (8 欄位): AOI@玻璃ID;機種ID;機台編號;解析度X,解析度Y;機檢判定;圖片前綴;(座標);圖片目錄路徑
        機檢判定: OK / NG / HY (畫異)

    Returns:
        {
            "glass_id": str,
            "model_id": str,
            "machine_no": str,
            "resolution": (int, int),
            "machine_judgment": str,
            "image_dir": str,
            "bomb_info": Optional[Dict],  # None 或 {image_prefix, defect_type, coordinates}
        }
    """
    data = data.strip()
    if not data.startswith("AOI@"):
        raise ProtocolError(f"Invalid prefix, expected 'AOI@', got: {data[:20]}")

    body = data[4:]  # 去掉 "AOI@"
    parts = body.split(";")

    if len(parts) < 6:
        raise ProtocolError(
            f"Invalid format, expected >=6 fields separated by ';', got {len(parts)}: {data[:100]}"
        )

    # 解析解析度
    try:
        res_parts = parts[3].split(",")
        resolution = (int(res_parts[0]), int(res_parts[1]))
    except (ValueError, IndexError):
        raise ProtocolError(f"Invalid resolution format: {parts[3]}")

    bomb_info = None
    image_dir = ""

    # 判斷是否包含炸彈資訊 (8 欄位)
    # 由於炸彈座標內部也包含 ';' (例如 `(350/174;1465/363)`), split(";") 會把座標切碎。
    # 特徵：parts[6] 開頭是 "(", 且倒數第二個 parts[-2] 結尾是 ")"
    if len(parts) >= 8 and parts[6].strip().startswith("(") and parts[-2].strip().endswith(")"):
        bomb_image_prefix = parts[5].strip()
        bomb_coords_raw = ";".join(parts[6:-1]).strip()
        image_dir = parts[-1].strip()

        bomb_info = _parse_bomb_coordinates(bomb_image_prefix, bomb_coords_raw)
        if bomb_info:
            logger.info(
                f"💣 協議炸彈資料: prefix={bomb_info['image_prefix']} "
                f"type={bomb_info['defect_type']} "
                f"coords={bomb_info['coordinates']}"
            )
    else:
        # 6 欄位模式，或單純的 fallback (路徑可能包含 ';' 所以重新 join)
        image_dir = ";".join(parts[5:]).strip()

    return {
        "glass_id": parts[0],
        "model_id": parts[1],
        "machine_no": parts[2],
        "resolution": resolution,
        "machine_judgment": parts[4],
        "image_dir": image_dir,
        "bomb_info": bomb_info,
    }


def build_response(
    glass_id: str,
    model_id: str,
    machine_no: str,
    machine_judgment: str,
    ai_judgment: str,
) -> str:
    """
    組裝回覆字串

    格式: AOI@玻璃ID;機種ID;機台編號;機檢判定;AI判定
    """
    return f"AOI@{glass_id};{model_id};{machine_no};{machine_judgment};{ai_judgment}"


def resolve_unc_path(unc_path: str, path_mapping: Dict[str, str]) -> str:
    """
    將 Windows UNC 路徑轉換為 Linux 路徑

    Args:
        unc_path: Windows UNC 路徑 (e.g. \\\\192.168.2.101\\d\\folder\\...)
        path_mapping: UNC 前綴 → Linux 路徑的映射表

    Returns:
        Linux 路徑
    """
    # 正規化: 統一使用 \\ 作為分隔符
    normalized = unc_path.replace("/", "\\")

    for unc_prefix, linux_prefix in path_mapping.items():
        # 正規化 prefix
        norm_prefix = unc_prefix.replace("/", "\\")
        if normalized.lower().startswith(norm_prefix.lower()):
            remainder = normalized[len(norm_prefix):]
            # 轉換分隔符
            remainder = remainder.replace("\\", "/")
            result = linux_prefix.rstrip("/") + remainder
            return result

    # 如果沒有匹配的映射，嘗試直接使用 (可能已經是 Linux 路徑)
    return unc_path.replace("\\", "/")


# ── AI 判定彙總 ──────────────────────────────────────

def aggregate_judgment(results: List[ImageResult]) -> Tuple[str, str]:
    """
    彙總推論結果為最終 AI 判定

    Returns:
        (ai_judgment, ng_details_json)
        - ai_judgment: "OK" / "NG" / "ERR:描述"
        - ng_details_json: JSON 格式的 NG 詳情
    """
    ng_details = []

    for result in results:
        if not result.anomaly_tiles:
            continue

        image_name = result.image_path.stem

        for tile, score, anomaly_map in result.anomaly_tiles:
            # 跳過炸彈系統模擬缺陷
            if tile.is_bomb:
                continue
            # 跳過疑似灰塵
            if tile.is_suspected_dust_or_scratch:
                continue
            # 跳過不檢測排除區域
            if tile.is_in_exclude_zone:
                continue
            # 跳過 AOI 座標但分數未達閾值的 tile (僅記錄用，不影響判定)
            if getattr(tile, 'is_aoi_coord_below_threshold', False):
                continue

            # 使用熱力圖峰值座標 (更精確)
            if tile.anomaly_peak_x >= 0 and tile.anomaly_peak_y >= 0:
                px, py = tile.anomaly_peak_x, tile.anomaly_peak_y
            else:
                px, py = tile.x + tile.width // 2, tile.y + tile.height // 2

            ng_details.append({
                "image": image_name,
                "tile_id": tile.tile_id,
                "x": tile.x, "y": tile.y,
                "peak_x": px, "peak_y": py,
                "score": round(score, 4),
                "is_dust": tile.is_suspected_dust_or_scratch,
                "dust_iou": round(tile.dust_heatmap_iou, 4),
            })

    if not ng_details:
        return "OK", "[]"

    ai_judgment = "NG"
    ng_details_json = json.dumps(ng_details, ensure_ascii=False)
    return ai_judgment, ng_details_json

def append_cv_edge_to_judgment(
    ai_judgment: str, ng_details_json: str, edge_defects: List['EdgeDefect'], image_name: str
) -> Tuple[str, str]:
    """將 CV 邊緣瑕疵加入到判定中"""
    if not edge_defects:
        return ai_judgment, ng_details_json
        
    real_edge_defects = [d for d in edge_defects if not getattr(d, 'is_suspected_dust_or_scratch', False) and not getattr(d, 'is_bomb', False)]
        
    if real_edge_defects:
        # 如果原本是 OK，現在因為真實邊緣瑕疵變成 NG
        if ai_judgment == "OK":
            ai_judgment = "NG"
        elif ai_judgment.startswith("NG"):
            # 保留原本的 NG，也可以標記為 NG_Mixed
            pass
        
    try:
        details = json.loads(ng_details_json) if ng_details_json != "[]" else []
    except Exception:
        details = []
        
    for d in real_edge_defects:
        details.append({
            "image": image_name,
            "type": f"cv_edge_{d.side}",
            "x": int(d.bbox[0]),
            "y": int(d.bbox[1]),
            "peak_x": int(d.center[0]),
            "peak_y": int(d.center[1]),
            "score": float(d.max_diff), # 用 max_diff 暫代 score
            "area": int(d.area),
            "is_cv_edge": True,
            "is_dust": False,
        })
        
    return ai_judgment, json.dumps(details, ensure_ascii=False)

# ── 推論結果 → DB 資料轉換 ──────────────────────────

def results_to_db_data(
    results: List[ImageResult],
    heatmap_info: Dict,
) -> List[Dict]:
    """將推論結果轉換為資料庫儲存格式"""
    db_images = []

    for result in results:
        image_name = result.image_path.stem

        # 判斷圖片級別的狀態
        is_ng = 0
        is_dust_only = 0
        is_bomb = 0
        anomaly_count = len(result.anomaly_tiles)
        
        # 加上 CV Edge 的 NG 數量 (排除灰塵和炸彈)
        cv_edge_count = len([d for d in result.edge_defects if not getattr(d, 'is_suspected_dust_or_scratch', False) and not getattr(d, 'is_bomb', False)])

        if anomaly_count > 0 or cv_edge_count > 0:
            real_ng = [t for t, s, m in result.anomaly_tiles
                       if not t.is_suspected_dust_or_scratch and not t.is_bomb and not t.is_in_exclude_zone
                       and not getattr(t, 'is_aoi_coord_below_threshold', False)]
            
            # 如果有真實 NG 或是 CV 邊緣 NG，就判定為 NG
            if real_ng or cv_edge_count > 0:
                is_ng = 1
                
            all_dust = (anomaly_count > 0 and 
                       all(t.is_suspected_dust_or_scratch for t, s, m in result.anomaly_tiles) and 
                       cv_edge_count == 0)
            
            has_bomb = any(t.is_bomb for t, s, m in result.anomaly_tiles)

            if all_dust:
                is_dust_only = 1
            if has_bomb and not real_ng and cv_edge_count == 0:
                is_bomb = 1

        max_score = max((s for _, s, _ in result.anomaly_tiles), default=0.0)

        # 找到對應的 overview heatmap 路徑
        heatmap_dir = heatmap_info.get("dir", "")
        overview_path = ""
        if heatmap_dir:
            expected = Path(heatmap_dir) / f"overview_{image_name}.png"
            if expected.exists():
                overview_path = str(expected)

        img_data = {
            "image_path": str(result.image_path),
            "image_name": result.image_path.name,
            "image_width": result.image_size[0],
            "image_height": result.image_size[1],
            "otsu_bounds": f"{result.otsu_bounds[0]},{result.otsu_bounds[1]},{result.otsu_bounds[2]},{result.otsu_bounds[3]}",
            "tile_count": result.processed_tile_count,
            "excluded_tiles": result.excluded_tile_count,
            "anomaly_count": anomaly_count,
            "edge_defect_count": cv_edge_count,
            "max_score": max_score,
            "is_ng": is_ng,
            "is_dust_only": is_dust_only,
            "is_bomb": is_bomb,
            "inference_time_ms": result.inference_time * 1000,
            "heatmap_path": overview_path,
            "tiles": [],
        }

        # Tile 資料
        for tile, score, anomaly_map in result.anomaly_tiles:
            tile_hp = ""
            if heatmap_dir:
                expected_t = Path(heatmap_dir) / f"heatmap_{image_name}_tile{tile.tile_id}.png"
                if expected_t.exists():
                    tile_hp = str(expected_t)

            is_below_thr = getattr(tile, 'is_aoi_coord_below_threshold', False)
            img_data["tiles"].append({
                "tile_id": tile.tile_id,
                "x": tile.x, "y": tile.y,
                "width": tile.width, "height": tile.height,
                "score": score,
                "is_anomaly": 0 if is_below_thr else 1,
                "is_dust": 1 if tile.is_suspected_dust_or_scratch else 0,
                "dust_iou": tile.dust_heatmap_iou,
                "is_bomb": 1 if tile.is_bomb else 0,
                "bomb_code": tile.bomb_defect_code,
                "peak_x": tile.anomaly_peak_x,
                "peak_y": tile.anomaly_peak_y,
                "heatmap_path": tile_hp,
                "is_exclude_zone": 1 if tile.is_in_exclude_zone else 0,
                "is_aoi_coord": 1 if tile.is_aoi_coord_tile else 0,
                "aoi_defect_code": tile.aoi_defect_code,
            })

        # CV 邊緣缺陷 — 獨立儲存 (不放入 tiles)
        img_data["edge_defects"] = []
        if hasattr(result, 'edge_defects') and result.edge_defects:
            for i, edge in enumerate(result.edge_defects):
                img_data["edge_defects"].append({
                    "side": edge.side,
                    "area": int(edge.area),
                    "bbox_x": int(edge.bbox[0]),
                    "bbox_y": int(edge.bbox[1]),
                    "bbox_w": int(edge.bbox[2]),
                    "bbox_h": int(edge.bbox[3]),
                    "max_diff": float(edge.max_diff),
                    "center_x": int(edge.center[0]),
                    "center_y": int(edge.center[1]),
                    "heatmap_path": getattr(edge, '_heatmap_path', ''),
                    "is_dust": 1 if getattr(edge, 'is_suspected_dust_or_scratch', False) else 0,
                    "is_bomb": 1 if getattr(edge, 'is_bomb', False) else 0,
                    "bomb_code": getattr(edge, 'bomb_defect_code', ''),
                })

        db_images.append(img_data)

    return db_images


# ── 主伺服器 ──────────────────────────────────────────

class CAPIServer:
    """CAPI AI TCP Socket 推論伺服器"""

    def __init__(self, config_path: str):
        """
        Args:
            config_path: server_config.yaml 路徑
        """
        # 載入伺服器設定
        with open(config_path, "r", encoding="utf-8") as f:
            self.server_config = yaml.safe_load(f)

        # 設定日誌
        setup_logging(self.server_config)

        server_cfg = self.server_config.get("server", {})
        self.host = server_cfg.get("host", "0.0.0.0")
        self.port = server_cfg.get("port", 7890)
        self.max_connections = server_cfg.get("max_connections", 10)
        self.recv_timeout = server_cfg.get("recv_timeout", 60)
        self.recv_buffer_size = server_cfg.get("recv_buffer_size", 4096)

        # 路徑映射
        self.path_mapping = self.server_config.get("path_mapping", {})

        # GPU 推論鎖 (確保 GPU 不被同時存取)
        self._gpu_lock = threading.Lock()

        # 停止旗標
        self._running = False
        self._server_socket = None

        # 推論器 (延遲載入)
        self.inferencer: Optional[CAPIInferencer] = None

        # 資料庫
        db_cfg = self.server_config.get("database", {})
        db_path = db_cfg.get("path", "/data/capi_ai/capi_results.db")
        self.db = CAPIDatabase(db_path)
        logger.info(f"Database: {db_path}")

        # 熱力圖管理
        heatmap_cfg = self.server_config.get("heatmap", {})
        self.heatmap_manager = HeatmapManager(
            base_dir=heatmap_cfg.get("base_dir", "/data/capi_ai/heatmaps"),
            save_format=heatmap_cfg.get("save_format", "png"),
        )
        self.save_overview = heatmap_cfg.get("save_overview", True)
        self.save_tile_detail = heatmap_cfg.get("save_tile_detail", True)

        # 寫入初始狀態
        server_status.start_time = time.time()
        server_status.is_running = False

        # 推論設定
        self.inference_config = self.server_config.get("inference", {})
        self.cpu_workers = self.inference_config.get("cpu_workers", 4)

        # 非同步儲存執行緒池 (Heatmap + DB 在背景完成，不阻塞回覆)
        self._async_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="async-save")

    def _load_inferencer(self):
        """載入 AI 推論模型"""
        inf_cfg = self.inference_config
        config_path = inf_cfg.get("config_path", "configs/capi_3f.yaml")
        device = inf_cfg.get("device", "auto")

        logger.info(f"Loading inference config: {config_path}")
        capi_config = CAPIConfig.from_yaml(config_path)

        # DB 設定覆蓋: 首次啟動自動從 YAML 匯入, 後續以 DB 為主
        try:
            # 優先從 YAML 同步新增的參數至 DB (會自動跳過已存在的)
            count = self.db.init_config_from_yaml(capi_config)
            if count > 0:
                logger.info(f"Seeded {count} new config params from YAML to DB")
            
            # 從 DB 讀取最終設定實體
            db_params = self.db.get_all_config_params()
            if db_params:
                # 建立 dict for quick lookup
                db_dict = {p["param_name"]: p for p in db_params}
                capi_config.apply_db_overrides(db_params)
                logger.info(f"Applied {len(db_params)} config overrides from DB")
        except Exception as e:
            logger.warning(f"Failed to load DB config, using YAML values: {e}")
            db_dict = {}

        # model_path 和 threshold 統一由 config 管理 (DB 優先, YAML fallback)
        model_path = capi_config.model_path or "./model.pt"
        threshold = capi_config.anomaly_threshold

        # 多模型模式: 當 capi_config 有 model_mapping 時，model_path 僅作為 fallback
        if capi_config.model_mapping:
            logger.info(f"Multi-model mode: {len(capi_config.model_mapping)} prefix mappings detected")
            for prefix, mpath in capi_config.model_mapping.items():
                logger.info(f"  {prefix} → {mpath}")
        else:
            logger.info(f"Single-model mode: {model_path}")

        logger.info(f"Loading model(s) (device={device}, default_threshold={threshold})")
        self.inferencer = CAPIInferencer(
            config=capi_config,
            model_path=model_path,
            device=device,
            threshold=threshold,
        )
        
        # 同步 CV Edge 設定到 Inferencer
        try:
            from capi_edge_cv import EdgeInspectionConfig
            edge_cfg = EdgeInspectionConfig.from_db_params(db_dict)
            self.inferencer.update_edge_config(edge_cfg)
        except Exception as e:
            logger.warning(f"Failed to load CV Edge config from DB: {e}")
        
        # 更新全域狀態的模型資訊
        with server_status.lock:
            if capi_config.model_mapping:
                # 多模型模式：顯示已載入數量
                loaded_count = len(self.inferencer._inferencers)
                total_count = len(capi_config.model_mapping)
                server_status.model_version = f"Multi-Model ({loaded_count}/{total_count} loaded)"
            else:
                # 單一模型模式：顯示檔名和修改時間
                try:
                    mtime = os.path.getmtime(model_path)
                    mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                    server_status.model_version = f"{Path(model_path).name} ({mtime_str})"
                except Exception:
                    server_status.model_version = Path(model_path).name
                
            # 判斷是否為 GPU 並取得型號
            display_device = device.upper()
            try:
                import torch
                if torch.cuda.is_available() and device != "cpu":
                    gpu_name = torch.cuda.get_device_name(0)
                    display_device = f"GPU ({gpu_name})"
            except Exception:
                pass
                
            server_status.device = display_device
            server_status.threshold = threshold
            server_status.threshold_mapping = dict(capi_config.threshold_mapping)
            
        logger.info("Model loaded successfully")

    def start(self):
        """啟動伺服器"""
        print("[SERVER] Starting...", flush=True)
        logger.info("=" * 60)
        logger.info("CAPI AI Inference Server Starting")
        logger.info("=" * 60)

        # 載入模型
        print("[SERVER] Loading model...", flush=True)
        try:
            self._load_inferencer()
            print("[SERVER] Model loaded OK", flush=True)
        except Exception as e:
            print(f"[SERVER] Model load FAILED: {e}", flush=True)
            logger.error(f"Failed to load model: {e}")
            logger.warning("Server will start without model (ERR for all requests)")

        # 啟動 Web 伺服器
        web_cfg = self.server_config.get("web", {})
        if web_cfg.get("enabled", True):
            web_host = web_cfg.get("host", "0.0.0.0")
            web_port = web_cfg.get("port", 8080)
            try:
                start_web_server_thread(
                    web_host, web_port, self.db,
                    str(self.heatmap_manager.base_dir),
                    server_status,
                    inferencer=self.inferencer,
                    heatmap_manager=self.heatmap_manager,
                    gpu_lock=self._gpu_lock,
                    capi_server_instance=self,
                )
                print(f"[SERVER] Web server: http://{web_host}:{web_port}", flush=True)
                logger.info(f"Web server: http://{web_host}:{web_port}")
            except Exception as e:
                logger.error(f"Failed to start web server: {e}")

        # 啟動清理排程
        self._start_cleanup_scheduler()

        # 建立 TCP Socket
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(self.max_connections)
        self._server_socket.settimeout(1.0)  # 讓主迴圈可以定期檢查 _running

        self._running = True
        
        with server_status.lock:
            server_status.is_running = True
            server_status.start_time = time.time()

        print(f"[SERVER] TCP listening on {self.host}:{self.port}", flush=True)
        logger.info(f"TCP server listening on {self.host}:{self.port}")
        logger.info(f"Max connections: {self.max_connections}")
        logger.info(f"Path mapping: {self.path_mapping}")
        logger.info("Waiting for connections...")
        logger.info("=" * 60)

        # 主迴圈
        while self._running:
            try:
                client_socket, client_addr = self._server_socket.accept()
                print(f"[SERVER] ACCEPT from {client_addr}", flush=True)
                logger.info(f"New connection from {client_addr}")
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_addr),
                    daemon=True,
                    name=f"client-{client_addr[0]}:{client_addr[1]}"
                )
                thread.start()
            except socket.timeout:
                continue
            except OSError as e:
                if self._running:
                    print(f"[SERVER] Socket OSError: {e}", flush=True)
                    logger.error(f"Socket error: {e}, server stopping")
                break

        logger.info("Server stopped")

    def _start_cleanup_scheduler(self):
        cleanup_cfg = self.server_config.get("cleanup", {})
        if not cleanup_cfg.get("enabled", False):
            return

        schedule_time = cleanup_cfg.get("schedule_time", "02:00")
        ok_retain     = cleanup_cfg.get("ok_retain_days", 14)
        ng_retain     = cleanup_cfg.get("ng_retain_days", 90)
        tile_retain   = cleanup_cfg.get("tile_retain_days", 7)
        vacuum        = cleanup_cfg.get("vacuum_after_cleanup", True)
        hour, minute  = map(int, schedule_time.split(":"))

        def _scheduler_loop():
            logger.info(
                f"[Cleanup] Scheduler started, daily at {schedule_time} "
                f"(OK={ok_retain}d, NG={ng_retain}d, tiles={tile_retain}d)"
            )
            while self._running:
                now    = datetime.now()
                target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if target <= now:
                    target += timedelta(days=1)
                sleep_secs = int((target - now).total_seconds())
                # 每 60 秒醒來檢查 _running
                for _ in range(sleep_secs // 60 + 1):
                    if not self._running:
                        return
                    time.sleep(min(60, sleep_secs))
                if not self._running:
                    return
                try:
                    logger.info("[Cleanup] Starting scheduled database cleanup...")
                    stats = self.db.cleanup_old_records(ok_retain, ng_retain, tile_retain, vacuum)
                    logger.info(
                        f"[Cleanup] Done — "
                        f"inference_records={stats['inference_records_deleted']}, "
                        f"tile_results={stats['tile_results_deleted']}"
                    )
                except Exception as e:
                    logger.error(f"[Cleanup] Failed: {e}")

        t = threading.Thread(target=_scheduler_loop, name="cleanup-scheduler", daemon=True)
        t.start()

    def stop(self):
        """停止伺服器"""
        logger.info("Stopping server...")
        self._running = False
        
        with server_status.lock:
            server_status.is_running = False
            
        # 等待背景儲存任務完成
        logger.info("Waiting for background save tasks to complete...")
        self._async_executor.shutdown(wait=True, cancel_futures=False)
        logger.info("Background save tasks completed")
            
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass

    def _handle_client(self, client_socket: socket.socket, client_addr: tuple):
        """處理客戶端連線（長連線模式）
        
        連線建立後持續等待資料，處理完一筆請求後不斷線，
        繼續等待下一筆請求，直到客戶端主動斷開或 idle timeout。
        """
        print(f"[HANDLER] Start for {client_addr}", flush=True)
        
        # 標記連線
        client_key = f"{client_addr[0]}:{client_addr[1]}"
        with server_status.lock:
            server_status.active_connections += 1
            server_status.connected_machines.add(client_key)

        # idle timeout: 0 表示不超時，否則使用設定值 (建議 600 秒)
        idle_timeout = self.recv_timeout if self.recv_timeout > 0 else None
        client_socket.settimeout(idle_timeout)

        machine_label = None  # 在解析到機台資訊後更新
        request_count = 0     # 此連線處理的請求數
        pending_buffer = []   # 暫存尚未處理的請求佇列 (多筆 AOI@ 黏在一起時使用)

        try:
            # ── 長連線主迴圈 ──
            while True:
                request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                request_data = None
                parsed = None
                raw_data = b""

                # ── 優先從 pending_buffer 取出下一筆請求 ──
                if pending_buffer:
                    request_data = pending_buffer.pop(0)
                    logger.debug(f"[{client_addr}] Processing buffered request ({len(pending_buffer)} remaining)")
                else:
                    try:
                        # 接收資料
                        while True:
                            chunk = client_socket.recv(self.recv_buffer_size)
                            if not chunk:
                                # 客戶端主動斷開連線
                                print(f"[HANDLER] Client disconnected: {client_addr}", flush=True)
                                if request_count > 0:
                                    logger.info(f"[{client_addr}] Client disconnected after {request_count} request(s)")
                                else:
                                    logger.info(f"[{client_addr}] Client disconnected (no requests)")
                                return  # 跳到 finally 清理
                            raw_data += chunk
                            print(f"[HANDLER] Received {len(raw_data)} bytes", flush=True)
                            # 檢查是否收到完整訊息 (以換行或 null 結尾)
                            if b"\n" in raw_data or b"\r" in raw_data or b"\x00" in raw_data:
                                break
                            # 如果資料包含完整的 AOI@ 格式 (不一定有結尾符)
                            try:
                                test_str = raw_data.decode("utf-8", errors="ignore").strip()
                                if test_str.startswith("AOI@") and test_str.count(";") >= 5:
                                    break
                            except Exception:
                                pass

                    except (TimeoutError, socket.timeout):
                        # idle timeout — 長時間未收到任何資料，清理死連線
                        if request_count > 0:
                            logger.info(f"[{client_addr}] Idle timeout after {request_count} request(s), closing")
                        else:
                            logger.debug(f"[{client_addr}] Idle timeout (no requests), closing")
                        return  # 跳到 finally 清理

                    if not raw_data:
                        continue

                    request_data = raw_data.decode("utf-8", errors="ignore").strip()
                    # 去除 null 字元和控制字元
                    request_data = request_data.replace("\x00", "").replace("\r", "").replace("\n", "")

                # ── 拆分多筆 AOI@ 請求 ──
                # 客戶端有時會一次送出多筆 AOI@ 請求黏在一起 (無分隔符)
                # 例如: AOI@...第一筆...AOI@...第二筆...
                # 需要按 "AOI@" 拆分，只處理第一筆，剩餘放入 pending_buffer
                if request_data.count("AOI@") > 1:
                    aoi_parts = request_data.split("AOI@")
                    # aoi_parts[0] 應該是空字串 (因為字串以 AOI@ 開頭)
                    # aoi_parts[1] 是第一筆請求內容
                    # aoi_parts[2:] 是剩餘請求
                    first_request = "AOI@" + aoi_parts[1]
                    remaining = ["AOI@" + p for p in aoi_parts[2:] if p.strip()]
                    if remaining:
                        pending_buffer.extend(remaining)
                    logger.info(f"[{client_addr}] Split {len(aoi_parts)-1} concatenated requests: "
                               f"processing first, {len(remaining)} buffered")
                    request_data = first_request

                logger.info(f"[{client_addr}] << {request_data}")

                inference_started = False  # 追蹤是否已遞增 active_inferences
                try:
                    # 解析請求
                    parsed = parse_request(request_data)
                    machine_label = f"{client_addr[0]} ({parsed['machine_no']})"

                    with server_status.lock:
                        server_status.total_requests += 1
                        server_status.active_inferences += 1
                        inference_started = True
                        # 將包含機台名稱的連線標籤存入
                        if client_key in server_status.connected_machines:
                            server_status.connected_machines.remove(client_key)
                        server_status.connected_machines.add(machine_label)

                    logger.info(
                        f"[{client_addr}] Glass={parsed['glass_id']} "
                        f"Model={parsed['model_id']} Machine={parsed['machine_no']} "
                        f"MJ={parsed['machine_judgment']} Dir={parsed['image_dir']}"
                    )

                    # 畫異 (HY) — 跳過推論，直接回傳 ERR:HY
                    if parsed["machine_judgment"] == "HY":
                        logger.info(f"[{client_addr}] 機檢判定=HY (畫異)，跳過推論")
                        ai_judgment = "ERR:HY"
                        response = build_response(
                            parsed["glass_id"], parsed["model_id"],
                            parsed["machine_no"], parsed["machine_judgment"],
                            ai_judgment,
                        )
                        response_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        logger.info(f"[{client_addr}] >> {response} (skipped)")
                        with server_status.lock:
                            server_status.active_inferences = max(0, server_status.active_inferences - 1)
                            server_status.total_err += 1
                            server_status.last_inference_time = response_time
                            server_status.last_judgment_result = {
                                "glass_id": parsed["glass_id"],
                                "machine_no": parsed["machine_no"],
                                "judgment": "ERR",
                                "detail": ai_judgment,
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "duration": "0.00s"
                            }
                        client_socket.sendall((response + "\r\n").encode("utf-8"))
                        request_count += 1
                        self._async_executor.submit(
                            self._save_results_async,
                            client_addr, parsed, [],
                            ai_judgment, "[]",
                            request_time, response_time, 0.0,
                            False, None,
                        )
                        continue

                    # 執行推論（不含 Heatmap 儲存，快速回覆）
                    start_time = time.time()
                    ai_judgment, ng_details, inference_results, is_duplicate, omit_image_raw = self._process_request(parsed)
                    processing_seconds = time.time() - start_time

                    # 重複投片時在 LOG 標記
                    if is_duplicate:
                        logger.warning(f"[{client_addr}] [DUPLICATE_PANEL] 重複投片，已選取最新圖片推論，判定={ai_judgment}")

                    # 組裝回覆
                    response = build_response(
                        parsed["glass_id"],
                        parsed["model_id"],
                        parsed["machine_no"],
                        parsed["machine_judgment"],
                        ai_judgment,
                    )

                    response_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    dup_tag = " [DUPLICATE]" if is_duplicate else ""
                    logger.info(f"[{client_addr}] >> {response} ({processing_seconds:.2f}s){dup_tag}")
                    
                    # 更新全域狀態 - 判定結果與推論結束
                    j_simple = "OK" if ai_judgment == "OK" else ("NG" if ai_judgment.startswith("NG") else "ERR")
                    with server_status.lock:
                        server_status.active_inferences = max(0, server_status.active_inferences - 1)
                        server_status.last_inference_time = response_time
                        if j_simple == "OK": server_status.total_ok += 1
                        elif j_simple == "NG": server_status.total_ng += 1
                        else: server_status.total_err += 1
                        
                        server_status.last_judgment_result = {
                            "glass_id": parsed["glass_id"],
                            "machine_no": parsed["machine_no"],
                            "judgment": j_simple,
                            "detail": ai_judgment if j_simple != "OK" else "OK",
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "duration": f"{processing_seconds:.2f}s"
                        }

                    # 🚀 先發送回覆（不等 Heatmap 儲存）
                    client_socket.sendall((response + "\r\n").encode("utf-8"))
                    request_count += 1

                    # 🔄 非同步儲存 Heatmap + DB（背景執行，不阻塞下一筆請求）
                    self._async_executor.submit(
                        self._save_results_async,
                        client_addr, parsed, inference_results,
                        ai_judgment, ng_details,
                        request_time, response_time, processing_seconds,
                        is_duplicate, omit_image_raw,
                    )

                except ProtocolError as e:
                    if inference_started:
                        with server_status.lock:
                            server_status.active_inferences = max(0, server_status.active_inferences - 1)
                    logger.error(f"[{client_addr}] Protocol error: {e}")
                    error_msg = f"ERR:PROTOCOL_ERROR ({str(e)[:80]})"
                    if parsed:
                        response = build_response(
                            parsed["glass_id"], parsed["model_id"],
                            parsed["machine_no"], parsed["machine_judgment"],
                            error_msg,
                        )
                    else:
                        response = f"AOI@;;;;{error_msg}"
                    try:
                        client_socket.sendall((response + "\r\n").encode("utf-8"))
                    except Exception:
                        pass
                    self._save_error_record(request_time, parsed, request_data, str(e))
                    # 協議錯誤不斷線，繼續等待下一筆

                except Exception as e:
                    if inference_started:
                        with server_status.lock:
                            server_status.active_inferences = max(0, server_status.active_inferences - 1)
                    logger.error(f"[{client_addr}] Unexpected error: {e}", exc_info=True)
                    error_msg = f"ERR:INTERNAL_ERROR ({type(e).__name__})"
                    if parsed:
                        response = build_response(
                            parsed["glass_id"], parsed["model_id"],
                            parsed["machine_no"], parsed["machine_judgment"],
                            error_msg,
                        )
                    else:
                        response = f"AOI@;;;;{error_msg}"
                    try:
                        client_socket.sendall((response + "\r\n").encode("utf-8"))
                    except Exception:
                        return  # 發送失敗，連線已斷
                    self._save_error_record(request_time, parsed, request_data, str(e))
                    # 內部錯誤不斷線，繼續等待下一筆

        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            # 連線層級錯誤 — 客戶端強制斷開
            logger.info(f"[{client_addr}] Connection lost: {e}")

        finally:
            try:
                client_socket.close()
            except Exception:
                pass
                
            # 清理連線狀態
            with server_status.lock:
                server_status.active_connections = max(0, server_status.active_connections - 1)
                
                if client_key in server_status.connected_machines:
                    server_status.connected_machines.remove(client_key)
                if machine_label and machine_label in server_status.connected_machines:
                    server_status.connected_machines.remove(machine_label)
                    
            logger.info(f"[{client_addr}] Connection closed (handled {request_count} request(s))")

    def _process_request(self, parsed: Dict) -> Tuple[str, str, List]:
        """
        處理推論請求（僅推論，不含 Heatmap 儲存）

        Returns:
            (ai_judgment, ng_details_json, inference_results)
            - inference_results: List[ImageResult] 供背景儲存使用
        """
        # 檢查推論器
        if self.inferencer is None:
            return "ERR:MODEL_NOT_LOADED", "[]", [], False, None

        # 轉換路徑
        image_dir = resolve_unc_path(parsed["image_dir"], self.path_mapping)
        panel_dir = Path(image_dir)

        if not panel_dir.exists():
            return f"ERR:DIR_NOT_FOUND ({image_dir})", "[]", [], False, None

        if not panel_dir.is_dir():
            return f"ERR:NOT_A_DIR ({image_dir})", "[]", [], False, None

        logger.info(f"Inference directory: {panel_dir}")

        # GPU 排隊 — 確保同一時刻只有一個推論任務使用 GPU
        # 注意：Heatmap 儲存已移至背景執行，GPU lock 只保護推論
        with self._gpu_lock:
            try:
                # 呼叫 process_panel 進行推論
                panel_result = self.inferencer.process_panel(
                    panel_dir,
                    cpu_workers=self.cpu_workers,
                    product_resolution=parsed["resolution"],
                    bomb_info=parsed.get("bomb_info"),
                    model_id=parsed.get("model_id"),
                )

                # process_panel 回傳: (results, omit_vis, omit_overexposed, omit_info, is_duplicate)
                results = panel_result[0]
                is_duplicate = panel_result[4]
                omit_image_raw = panel_result[5] if len(panel_result) > 5 else None

                if is_duplicate:
                    logger.warning(
                        f"重複投片 (DUPLICATE_PANEL): {panel_dir} — "
                        f"已依建立時間選出最新圖片繼續推論，結果將標記 [DUPLICATE]"
                    )

                if not results:
                    return "ERR:NO_IMAGES_FOUND", "[]", [], False, None

                # 彙總判定
                ai_judgment, ng_details = aggregate_judgment(results)

                # 加上 CV Edge 判定
                for result in results:
                    if hasattr(result, 'edge_defects') and result.edge_defects:
                        ai_judgment, ng_details = append_cv_edge_to_judgment(
                            ai_judgment, ng_details, result.edge_defects, result.image_path.stem
                        )

                return ai_judgment, ng_details, results, is_duplicate, omit_image_raw

            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)
                return f"ERR:INFERENCE_FAILED ({type(e).__name__}: {str(e)[:100]})", "[]", [], False, None


    def _save_results_async(
        self,
        client_addr: tuple,
        parsed: Dict,
        results: List,
        ai_judgment: str,
        ng_details: str,
        request_time: str,
        response_time: str,
        processing_seconds: float,
        is_duplicate: bool = False,
        omit_image_raw: Any = None,
    ):
        """
        非同步儲存 Heatmap 和 DB 記錄（在背景執行緒中執行）

        此方法在回覆客戶端後才執行，不影響回應速度。
        """
        save_start = time.time()
        try:
            # 儲存熱力圖
            heatmap_info = {}
            if results:
                try:
                    heatmap_info = self.heatmap_manager.save_panel_heatmaps(
                        glass_id=parsed["glass_id"],
                        results=results,
                        inferencer=self.inferencer,
                        save_overview=self.save_overview,
                        save_tile_detail=self.save_tile_detail,
                        omit_image=omit_image_raw,
                    )
                except Exception as e:
                    logger.error(f"[{client_addr}] Async heatmap save error: {e}")

            # 轉換為 DB 格式
            image_results_data = results_to_db_data(results, heatmap_info) if results else []

            # 儲存到資料庫
            total_images = len(image_results_data)
            ng_images = sum(1 for d in image_results_data if d.get("is_ng"))

            # 組合 error_message：重複投片加上標記
            if is_duplicate:
                dup_note = "[DUPLICATE_PANEL] 重複投片，已依建立時間選取最新圖片推論"
                err_suffix = ai_judgment if ai_judgment.startswith("ERR") else ""
                error_message = f"{dup_note}\n{err_suffix}".strip()
            else:
                error_message = ai_judgment if ai_judgment.startswith("ERR") else ""

            client_bomb_info_str = json.dumps(parsed["bomb_info"], ensure_ascii=False) if parsed.get("bomb_info") else ""

            self.db.save_inference_record(
                glass_id=parsed["glass_id"],
                model_id=parsed["model_id"],
                machine_no=parsed["machine_no"],
                resolution=parsed["resolution"],
                machine_judgment=parsed["machine_judgment"],
                ai_judgment=ai_judgment,
                image_dir=parsed["image_dir"],
                total_images=total_images,
                ng_images=ng_images,
                ng_details=ng_details,
                request_time=request_time,
                response_time=response_time,
                processing_seconds=processing_seconds,
                heatmap_dir=heatmap_info.get("dir", ""),
                error_message=error_message,
                client_bomb_info=client_bomb_info_str,
                image_results_data=image_results_data,
            )

            dup_tag = " [DUPLICATE]" if is_duplicate else ""
            save_time = time.time() - save_start
            logger.info(f"[{client_addr}] Async save completed: heatmap+DB in {save_time:.2f}s{dup_tag}")

        except Exception as e:
            logger.error(f"[{client_addr}] Async save failed: {e}", exc_info=True)


    def _save_error_record(self, request_time: str, parsed: Optional[Dict], raw_data: Optional[str], error: str):
        """儲存錯誤記錄到資料庫"""
        try:
            self.db.save_inference_record(
                glass_id=parsed["glass_id"] if parsed else "",
                model_id=parsed["model_id"] if parsed else "",
                machine_no=parsed["machine_no"] if parsed else "",
                resolution=parsed["resolution"] if parsed else (0, 0),
                machine_judgment=parsed["machine_judgment"] if parsed else "",
                ai_judgment="ERR",
                image_dir=parsed["image_dir"] if parsed else "",
                total_images=0, ng_images=0, ng_details="[]",
                request_time=request_time,
                response_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                processing_seconds=0.0,
                error_message=f"{error}\nRaw: {raw_data[:200] if raw_data else 'N/A'}",
            )
        except Exception as db_err:
            logger.error(f"Failed to save error record: {db_err}")


# ── CLI ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CAPI AI Inference Server")
    parser.add_argument(
        "--config", "-c",
        default="server_config.yaml",
        help="Server configuration YAML file (default: server_config.yaml)"
    )
    parser.add_argument(
        "--test-protocol",
        action="store_true",
        help="Run protocol parsing tests and exit"
    )
    args = parser.parse_args()

    if args.test_protocol:
        _run_protocol_tests()
        return

    # 檢查設定檔
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # 建立伺服器
    server = CAPIServer(args.config)

    # Signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        server.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 啟動伺服器
    server.start()


def _run_protocol_tests():
    """協議解析測試"""
    print("=" * 60)
    print("Protocol Parsing Tests")
    print("=" * 60)

    # Test 1: 正常解析
    test_data = r"AOI@YPB21Y015A13;GN156HCAB6G0S;CAPI1403;1920,1080;OK;\\192.168.2.101\d\TIANMU\yuantu\GN116BCAA240S\20260225\T55BR592AE22"
    parsed = parse_request(test_data)
    assert parsed["glass_id"] == "YPB21Y015A13", f"glass_id mismatch: {parsed['glass_id']}"
    assert parsed["model_id"] == "GN156HCAB6G0S"
    assert parsed["machine_no"] == "CAPI1403"
    assert parsed["resolution"] == (1920, 1080)
    assert parsed["machine_judgment"] == "OK"
    print(f"✅ Test 1 PASSED: {parsed}")

    # Test 2: 路徑轉換
    path_mapping = {"\\\\192.168.2.101\\d": "/capi01"}
    linux_path = resolve_unc_path(r"\\192.168.2.101\d\TIANMU\yuantu\test", path_mapping)
    assert linux_path == "/capi01/TIANMU/yuantu/test", f"Path mismatch: {linux_path}"
    print(f"✅ Test 2 PASSED: UNC → {linux_path}")

    # Test 3: 回覆組裝
    response = build_response("GLASS001", "MODEL01", "CAPI1403", "OK", "NG@G0F00000(1024,512)")
    expected = "AOI@GLASS001;MODEL01;CAPI1403;OK;NG@G0F00000(1024,512)"
    assert response == expected, f"Response mismatch: {response}"
    print(f"✅ Test 3 PASSED: {response}")

    # Test 4: NG 判定回覆
    response = build_response("GLASS002", "MODEL01", "CAPI1403", "NG", "OK")
    print(f"✅ Test 4 PASSED: {response}")

    # Test 5: ERR 回覆
    response = build_response("GLASS003", "MODEL01", "CAPI1403", "OK", "ERR:MODEL_NOT_LOADED")
    print(f"✅ Test 5 PASSED: {response}")

    # Test 6: 無效格式
    try:
        parse_request("INVALID DATA")
        print("❌ Test 6 FAILED: should have raised")
    except ProtocolError:
        print("✅ Test 6 PASSED: invalid prefix detected")

    # Test 7: 欄位不足
    try:
        parse_request("AOI@a;b;c")
        print("❌ Test 7 FAILED: should have raised")
    except ProtocolError:
        print("✅ Test 7 PASSED: insufficient fields detected")

    # Test 8: 路徑映射 — 大小寫不敏感
    linux_path = resolve_unc_path(r"\\192.168.2.101\D\data\test", path_mapping)
    assert "/capi01" in linux_path, f"Case insensitive path mismatch: {linux_path}"
    print(f"✅ Test 8 PASSED: case insensitive → {linux_path}")

    print("\n" + "=" * 60)
    print("All protocol tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
