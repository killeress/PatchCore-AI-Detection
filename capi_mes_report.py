"""MES 人工判定與 AI 推論結果比對。"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


logger = logging.getLogger("capi.mes_report")


DEFECT_CODE_CATALOG_PATH = Path(__file__).resolve().parent / "configs" / "mes_defect_codes.json"


try:
    from capi_mes_credentials import ORACLE_MES_PASSWORD
except ImportError:
    ORACLE_MES_PASSWORD = ""


WP_DEFTHIS_COLUMNS = (
    "FAC_ID", "PNL_ID", "TRANS_NBR", "DFCT_CODE", "DFCT_REASON",
    "DFCT_REASON2", "ITEM_NBR", "DFCT_CODE2", "ERROR_FLAG", "COMMENTS",
    "DFCT_DISP", "TRANS_DATE", "X_AXIS", "Y_AXIS", "PATTERN_CODE",
    "RGB_FLAG", "IC_ADDRESS", "TRANS_OPERATION", "UPDATE_STAMP",
    "SYS_TRANS_FLAG", "RW_ROUTE", "DEFT_DATE", "DEFT_OPER", "IF_NEWER",
    "IF_MENDED", "IF_OK", "IC_COUNT", "IC_COF", "SUB_CODE", "T_STAMP",
    "DFCT_GRADE", "MAT_ID", "LOT_QTY", "DEFECT_AREA", "PROJECT_COMMENTS",
    "RELAX_FLAG", "RELAX_DESCRIPTION",
)

WP_DEFTHIS_SCHEMA_BY_FACILITY = {
    "MOD1": "MCRDA1",
    "MOD2": "MERDA1",
}

WP_DEFTHIS_FAC_ID_BY_FACILITY = {
    "MOD1": "C",
    "MOD2": "E",
}

WP_DEFTHIS_INDEX_HINT_BY_FACILITY = {
    "MOD1": "",
    "MOD2": "/*+ INDEX(w WP_DEFTHIS_PK) */",
}

ORACLE_PANEL_BATCH_SIZE = 900

COORDINATE_MATCH_TOLERANCE = 20
MES_COUNTED_MISS_REVIEW_CATEGORIES = frozenset({
    "score_below_threshold",
    "low_contrast",
    "dust_misfilter",
})
CAPIHM_HOSTNAME = "capihm"
CAPIHM_AOI_HEIGHT = 1080


class MESReportConfigurationError(RuntimeError):
    """MES Oracle 連線設定不完整。"""


@lru_cache(maxsize=1)
def load_defect_code_catalog() -> Dict[str, Dict[str, str]]:
    """載入 MES defect code 的嚴重程度與不良描述。"""
    try:
        with DEFECT_CODE_CATALOG_PATH.open("r", encoding="utf-8") as stream:
            raw_catalog = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("MES defect code 字典載入失敗：%s", exc)
        return {}

    if not isinstance(raw_catalog, dict):
        logger.warning("MES defect code 字典格式錯誤：根節點必須是物件")
        return {}

    catalog = {}
    for raw_code, raw_info in raw_catalog.items():
        code = str(raw_code or "").strip().upper()
        if not code or not isinstance(raw_info, dict):
            continue
        catalog[code] = {
            "severity": str(raw_info.get("severity") or "").strip(),
            "description": str(raw_info.get("description") or "").strip(),
        }
    return catalog


def _parse_datetime(value) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H.%M.%S.%f", "%Y-%m-%d %H.%M.%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _display_datetime(value) -> str:
    parsed = _parse_datetime(value)
    return parsed.isoformat(sep=" ", timespec="seconds") if parsed is not None else str(value or "")


def _has_coordinate(value) -> bool:
    return value is not None and str(value).strip() != ""


def _coordinate_value(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _aoi_coordinates(raw_value) -> List[tuple]:
    try:
        payload = json.loads(raw_value or "{}")
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, Mapping):
        return []

    coordinates = []
    seen = set()
    for entries in payload.values():
        if not isinstance(entries, (list, tuple)):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            x = _coordinate_value(entry.get("product_x"))
            y = _coordinate_value(entry.get("product_y"))
            if x is None or y is None or (x, y) in seen:
                continue
            seen.add((x, y))
            coordinates.append((x, y))
    return coordinates


def _mes_coordinate_in_aoi_space(x: float, y: float, host_name: str) -> tuple:
    if str(host_name or "").strip().lower() == CAPIHM_HOSTNAME:
        # CAPIHM 的 MES 人工座標是將 AOI 橫式畫面順時針轉成直式後輸入。
        return y, CAPIHM_AOI_HEIGHT - x
    return x, y


def _build_coordinate_match(
    defects: Sequence[Mapping],
    raw_aoi_coordinates,
    host_name: str,
) -> Dict:
    mes_coordinates = []
    for defect in defects:
        x = _coordinate_value(defect.get("x_axis"))
        y = _coordinate_value(defect.get("y_axis"))
        if x is not None and y is not None:
            mes_coordinates.append((x, y))

    transform = (
        "capihm_portrait_clockwise"
        if str(host_name or "").strip().lower() == CAPIHM_HOSTNAME
        else "none"
    )
    result = {
        "status": "not_applicable",
        "matched_count": 0,
        "mes_coordinate_count": len(mes_coordinates),
        "tolerance": COORDINATE_MATCH_TOLERANCE,
        "method": "coordinate_only",
        "transform": transform,
    }
    if not defects:
        return result
    if not mes_coordinates:
        result["status"] = "invalid_mes_coordinates"
        return result

    aoi_coordinates = _aoi_coordinates(raw_aoi_coordinates)
    if not aoi_coordinates:
        result["status"] = "no_aoi_coordinates"
        return result

    for mes_x, mes_y in mes_coordinates:
        compared_x, compared_y = _mes_coordinate_in_aoi_space(
            mes_x,
            mes_y,
            host_name,
        )
        if any(
            abs(aoi_x - compared_x) <= COORDINATE_MATCH_TOLERANCE
            and abs(aoi_y - compared_y) <= COORDINATE_MATCH_TOLERANCE
            for aoi_x, aoi_y in aoi_coordinates
        ):
            result["matched_count"] += 1

    if result["matched_count"] == len(mes_coordinates):
        result["status"] = "matched"
    elif result["matched_count"]:
        result["status"] = "partial"
    else:
        result["status"] = "unmatched"
    return result


def classify_mes_judgment(rows: Iterable[Mapping], cutoff: datetime) -> Dict:
    """依 LOGIC.xlsx 判斷一筆 AI 推論之後，MES 是否有人員有效不良。"""
    defect_code_catalog = load_defect_code_catalog()
    qualifying = []
    row_count = 0
    for row in rows:
        row_count += 1
        trans_date = _parse_datetime(row.get("trans_date"))
        defect_code = str(row.get("dfct_code") or "").strip().upper()
        if trans_date is None or trans_date < cutoff:
            continue
        if defect_code == "PCK21":
            continue
        if not _has_coordinate(row.get("x_axis")) or not _has_coordinate(row.get("y_axis")):
            continue
        defect_info = defect_code_catalog.get(defect_code, {})
        qualifying.append({
            "dfct_code": defect_code,
            "severity": defect_info.get("severity", ""),
            "description": defect_info.get("description", ""),
            "trans_date": _display_datetime(trans_date),
            "x_axis": row.get("x_axis"),
            "y_axis": row.get("y_axis"),
        })

    return {
        "judgment": "NG" if qualifying else "OK",
        "mes_row_count": row_count,
        "qualifying_defects": qualifying,
    }


def _normalize_ai_judgment(value: str) -> Optional[str]:
    text = str(value or "").strip().upper()
    if text in {"OK", "OK-I"}:
        return "OK"
    if text.startswith("NG"):
        return "NG"
    return None


def _rate(count: int, total: int) -> float:
    return round(count * 100.0 / total, 2) if total else 0.0


def is_mes_miss_counted(row: Mapping) -> bool:
    """未 Review 的漏檢全算；已 Review 只計入確認屬於 AI 漏檢的原因。"""
    if row.get("comparison") != "miss_detection":
        return False
    review = row.get("review") if isinstance(row.get("review"), Mapping) else None
    category = str((review or {}).get("category") or "").strip()
    return not category or category in MES_COUNTED_MISS_REVIEW_CATEGORIES


def apply_mes_review_miss_policy(report: Dict) -> Dict:
    """套用人工 Review 後的漏檢分子，分母仍為全部可比對數。"""
    counted_misses = 0
    for row in report.get("records") or []:
        counted = is_mes_miss_counted(row)
        row["counts_as_miss_detection"] = counted
        if counted:
            counted_misses += 1

    summary = report["summary"]
    summary["miss_detection"] = counted_misses
    summary["miss_detection_rate"] = _rate(counted_misses, int(summary.get("total") or 0))
    return report


def build_mes_comparison(
    records: Sequence[Mapping],
    defects_by_panel: Mapping[str, Sequence[Mapping]],
    *,
    host_name: str = "",
) -> Dict:
    """組合逐筆結果與以全部可比對推論為分母的過／漏檢統計。"""
    normalized_defects = {
        str(panel_id or "").strip().upper(): rows
        for panel_id, rows in defects_by_panel.items()
    }
    output = []
    correct = over_detection = miss_detection = uncomparable = 0

    for record in records:
        glass_id = str(record.get("glass_id") or "").strip()
        ai_judgment = _normalize_ai_judgment(record.get("ai_judgment"))
        cutoff = _parse_datetime(record.get("request_time"))
        mes_result = (
            classify_mes_judgment(normalized_defects.get(glass_id.upper(), []), cutoff)
            if cutoff is not None else
            {"judgment": None, "mes_row_count": 0, "qualifying_defects": []}
        )

        if ai_judgment is None or cutoff is None or not glass_id:
            comparison = "uncomparable"
            uncomparable += 1
        elif ai_judgment == "NG" and mes_result["judgment"] == "OK":
            comparison = "over_detection"
            over_detection += 1
        elif ai_judgment == "OK" and mes_result["judgment"] == "NG":
            comparison = "miss_detection"
            miss_detection += 1
        else:
            comparison = "correct"
            correct += 1
        if comparison == "over_detection":
            review_type = "over_detection"
        elif comparison == "miss_detection":
            review_type = "miss_detection"
        elif ai_judgment == "NG" and mes_result["judgment"] == "NG":
            review_type = "true_ng"
        else:
            review_type = ""

        first_defect = mes_result["qualifying_defects"][0] if mes_result["qualifying_defects"] else None
        coordinate_match = _build_coordinate_match(
            mes_result["qualifying_defects"],
            record.get("aoi_machine_coords"),
            host_name,
        )
        output.append({
            "id": record.get("id"),
            "glass_id": glass_id,
            "model_id": record.get("model_id", ""),
            "machine_no": record.get("machine_no", ""),
            "request_time": _display_datetime(record.get("request_time")),
            "image_dir": record.get("image_dir", ""),
            "ai_raw_judgment": record.get("ai_judgment", ""),
            "ai_judgment": ai_judgment,
            "mes_judgment": mes_result["judgment"],
            "comparison": comparison,
            "review_type": review_type,
            "mes_row_count": mes_result["mes_row_count"],
            "qualifying_defect_count": len(mes_result["qualifying_defects"]),
            "qualifying_defects": mes_result["qualifying_defects"],
            "first_defect": first_defect,
            "coordinate_match": coordinate_match,
        })

    total = correct + over_detection + miss_detection
    return {
        "summary": {
            "total": total,
            "correct": correct,
            "over_detection": over_detection,
            "miss_detection": miss_detection,
            "accuracy_rate": _rate(correct, total),
            "over_detection_rate": _rate(over_detection, total),
            "miss_detection_rate": _rate(miss_detection, total),
            "uncomparable": uncomparable,
        },
        "records": output,
    }


def build_mes_review_summary(records: Sequence[Mapping]) -> Dict:
    """統計目前 Report 範圍內的人工 Review 完成率、原因及 NG 樣本數。"""
    review_types = ("over_detection", "miss_detection", "true_ng")
    by_type = {
        key: {"total": 0, "reviewed": 0, "pending": 0, "by_category": {}}
        for key in review_types
    }
    reviewed = pending = confirmed_ng_reviews = ng_samples = 0

    for row in records:
        review_type = str(row.get("review_type") or "")
        if review_type not in by_type:
            continue
        type_stats = by_type[review_type]
        type_stats["total"] += 1
        review = row.get("review") if isinstance(row.get("review"), Mapping) else None
        category = str((review or {}).get("category") or "")
        if not category:
            pending += 1
            type_stats["pending"] += 1
            continue

        reviewed += 1
        type_stats["reviewed"] += 1
        categories = type_stats["by_category"]
        categories[category] = categories.get(category, 0) + 1
        if bool(review.get("confirmed_ng")):
            confirmed_ng_reviews += 1
        ng_samples += int(review.get("ng_sample_count") or 0)

    return {
        "total": reviewed + pending,
        "reviewed": reviewed,
        "pending": pending,
        "confirmed_ng_reviews": confirmed_ng_reviews,
        "ng_samples": ng_samples,
        "by_type": by_type,
    }


class OracleMESRepository:
    """依設備廠別，從對應 Oracle TNS 讀取 MES 人員不良判定。"""

    def __init__(self, config: Mapping):
        self.facility = str(config.get("facility") or "").strip().upper()
        oracle_config = config.get("oracle") if isinstance(config.get("oracle"), Mapping) else {}
        tns_configs = oracle_config.get("tns") if isinstance(oracle_config.get("tns"), Mapping) else {}
        normalized_tns = {str(name).upper(): value for name, value in tns_configs.items()}
        if not self.facility:
            raise MESReportConfigurationError("MES Oracle 設定缺少：facility")
        self.wp_defthis_schema = WP_DEFTHIS_SCHEMA_BY_FACILITY.get(self.facility)
        if not self.wp_defthis_schema:
            raise MESReportConfigurationError(f"MES Oracle 找不到廠別 WP_DEFTHIS schema：{self.facility}")
        self.wp_defthis_fac_id = WP_DEFTHIS_FAC_ID_BY_FACILITY[self.facility]
        self.wp_defthis_index_hint = WP_DEFTHIS_INDEX_HINT_BY_FACILITY[self.facility]
        if self.facility not in normalized_tns:
            raise MESReportConfigurationError(f"MES Oracle 找不到廠別 TNS：{self.facility}")

        tns_config = normalized_tns[self.facility]
        self.user = str(oracle_config.get("user") or "").strip()
        self.password = ORACLE_MES_PASSWORD
        self.host = str(tns_config.get("host") or "").strip()
        self.port = int(tns_config.get("port") or 1521)
        self.service_name = str(tns_config.get("service_name") or "").strip()
        missing = [
            name for name, value in (
                ("user", self.user),
                ("password", self.password),
                ("host", self.host),
                ("service_name", self.service_name),
            ) if not value
        ]
        if missing:
            raise MESReportConfigurationError("MES Oracle 設定缺少：" + ", ".join(missing))

    @property
    def source_label(self) -> str:
        return f"{self.facility} / {self.service_name.upper()} / {self.wp_defthis_schema}.WP_DEFTHIS"

    def fetch_defects(self, panel_ids: Sequence[str], min_trans_date: datetime) -> Dict[str, List[Dict]]:
        panel_ids = sorted({str(value or "").strip().upper() for value in panel_ids if str(value or "").strip()})
        if not panel_ids:
            return {}
        try:
            import oracledb
        except ImportError as exc:
            raise MESReportConfigurationError("Server 尚未安裝 python-oracledb，請執行 pip install -r requirements.txt") from exc

        dsn = oracledb.makedsn(self.host, self.port, service_name=self.service_name)
        started_at = time.monotonic()
        logger.info("[MES Report] Oracle connect start: facility=%s, panels=%d", self.facility, len(panel_ids))
        connection = oracledb.connect(user=self.user, password=self.password, dsn=dsn)
        logger.info("[MES Report] Oracle connected in %.2fs", time.monotonic() - started_at)
        result: Dict[str, List[Dict]] = {}
        min_trans_date_text = min_trans_date.strftime("%Y-%m-%d %H.%M.%S.%f")
        batch_size = ORACLE_PANEL_BATCH_SIZE
        batch_count = (len(panel_ids) + batch_size - 1) // batch_size
        try:
            cursor = connection.cursor()
            cursor.arraysize = 1000
            cursor.prefetchrows = 1000
            try:
                for offset in range(0, len(panel_ids), batch_size):
                    batch_number = offset // batch_size + 1
                    batch_started_at = time.monotonic()
                    chunk = panel_ids[offset:offset + batch_size]
                    panel_binds = {f"panel_{idx}": value for idx, value in enumerate(chunk)}
                    placeholders = ", ".join(f":panel_{idx}" for idx in range(len(chunk)))
                    sql = f"""
                        SELECT {self.wp_defthis_index_hint}
                               w.PNL_ID, w.DFCT_CODE, w.TRANS_DATE, w.X_AXIS, w.Y_AXIS
                        FROM {self.wp_defthis_schema}.WP_DEFTHIS w
                        WHERE w.FAC_ID = :fac_id
                          AND w.DEFT_OPER = :deft_oper
                          AND w.IF_NEWER = 'Y'
                          AND w.TRANS_DATE >= :min_trans_date
                          AND w.PNL_ID IN ({placeholders})
                        ORDER BY w.PNL_ID, w.TRANS_DATE
                    """
                    cursor.execute(sql, {
                        "fac_id": self.wp_defthis_fac_id,
                        "deft_oper": "1600",
                        "min_trans_date": min_trans_date_text,
                        **panel_binds,
                    })
                    row_count = 0
                    for pnl_id, code, trans_date, x_axis, y_axis in cursor:
                        row_count += 1
                        key = str(pnl_id or "").strip().upper()
                        result.setdefault(key, []).append({
                            "pnl_id": key,
                            "dfct_code": code,
                            "trans_date": trans_date,
                            "x_axis": x_axis,
                            "y_axis": y_axis,
                        })
                    logger.info(
                        "[MES Report] Oracle batch %d/%d: panels=%d, rows=%d, elapsed=%.2fs",
                        batch_number, batch_count, len(chunk), row_count,
                        time.monotonic() - batch_started_at,
                    )
            finally:
                cursor.close()
        finally:
            connection.close()
        logger.info(
            "[MES Report] Oracle query complete: panels=%d, matched_panels=%d, elapsed=%.2fs",
            len(panel_ids), len(result), time.monotonic() - started_at,
        )
        return result

    def fetch_report_details(self, panel_id: str, min_trans_date: datetime) -> List[Dict]:
        """按需取得單筆 AI 推論所使用 MES 記錄的全部 WP_DEFTHIS 欄位。"""
        normalized_panel_id = str(panel_id or "").strip().upper()
        if not normalized_panel_id:
            return []
        try:
            import oracledb
        except ImportError as exc:
            raise MESReportConfigurationError("Server 尚未安裝 python-oracledb，請執行 pip install -r requirements.txt") from exc

        dsn = oracledb.makedsn(self.host, self.port, service_name=self.service_name)
        started_at = time.monotonic()
        logger.info(
            "[MES Report] Detail query start: facility=%s, panel=%s",
            self.facility, normalized_panel_id,
        )
        connection = oracledb.connect(user=self.user, password=self.password, dsn=dsn)
        rows = []
        try:
            cursor = connection.cursor()
            cursor.arraysize = 1000
            cursor.prefetchrows = 1000
            try:
                # MOD1/MCRDA1 與 MOD2/MERDA1 的 WP_DEFTHIS 欄位並非完全相同。
                # MOD1 使用實際欄位清單，避免查詢不存在的欄位（例如 SYS_TRANS_FLAG）。
                columns_sql = "*" if self.facility == "MOD1" else ", ".join(WP_DEFTHIS_COLUMNS)
                cursor.execute(
                    f"""
                        SELECT {columns_sql}
                        FROM {self.wp_defthis_schema}.WP_DEFTHIS
                        WHERE FAC_ID = :fac_id
                          AND PNL_ID = :panel_id
                          AND DEFT_OPER = :deft_oper
                          AND IF_NEWER = 'Y'
                          AND TRANS_DATE >= :min_trans_date
                        ORDER BY TRANS_DATE, TRANS_NBR
                    """,
                    {
                        "fac_id": self.wp_defthis_fac_id,
                        "panel_id": normalized_panel_id,
                        "deft_oper": "1600",
                        "min_trans_date": min_trans_date.strftime("%Y-%m-%d %H.%M.%S.%f"),
                    },
                )
                description = getattr(cursor, "description", None) or []
                columns = [str(item[0]).upper() for item in description] or list(WP_DEFTHIS_COLUMNS)
                rows = [dict(zip(columns, values)) for values in cursor]
            finally:
                cursor.close()
        finally:
            connection.close()
        logger.info(
            "[MES Report] Detail query complete: panel=%s, rows=%d, elapsed=%.2fs",
            normalized_panel_id, len(rows), time.monotonic() - started_at,
        )
        return rows
