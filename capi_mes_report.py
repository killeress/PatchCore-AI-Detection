"""MES 人工判定與 AI 推論結果比對。"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


logger = logging.getLogger(__name__)


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


class MESReportConfigurationError(RuntimeError):
    """MES Oracle 連線設定不完整。"""


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


def classify_mes_judgment(rows: Iterable[Mapping], cutoff: datetime) -> Dict:
    """依 LOGIC.xlsx 判斷一筆 AI 推論之後，MES 是否有人員有效不良。"""
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
        qualifying.append({
            "dfct_code": defect_code,
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


def build_mes_comparison(records: Sequence[Mapping], defects_by_panel: Mapping[str, Sequence[Mapping]]) -> Dict:
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

        first_defect = mes_result["qualifying_defects"][0] if mes_result["qualifying_defects"] else None
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
            "mes_row_count": mes_result["mes_row_count"],
            "qualifying_defect_count": len(mes_result["qualifying_defects"]),
            "qualifying_defects": mes_result["qualifying_defects"],
            "first_defect": first_defect,
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


class OracleMESRepository:
    """依設備廠別，從對應 Oracle TNS 讀取 MES 人員不良判定。"""

    def __init__(self, config: Mapping):
        self.facility = str(config.get("facility") or "").strip().upper()
        oracle_config = config.get("oracle") if isinstance(config.get("oracle"), Mapping) else {}
        tns_configs = oracle_config.get("tns") if isinstance(oracle_config.get("tns"), Mapping) else {}
        normalized_tns = {str(name).upper(): value for name, value in tns_configs.items()}
        if not self.facility:
            raise MESReportConfigurationError("MES Oracle 設定缺少：facility")
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
        return f"{self.facility} / {self.service_name.upper()} / MERDA1.WP_DEFTHIS"

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
        batch_count = (len(panel_ids) + 899) // 900
        try:
            cursor = connection.cursor()
            cursor.arraysize = 1000
            cursor.prefetchrows = 1000
            try:
                for offset in range(0, len(panel_ids), 900):
                    batch_number = offset // 900 + 1
                    batch_started_at = time.monotonic()
                    chunk = panel_ids[offset:offset + 900]
                    panel_binds = {f"panel_{idx}": value for idx, value in enumerate(chunk)}
                    placeholders = ", ".join(f":panel_{idx}" for idx in range(len(chunk)))
                    sql = f"""
                        SELECT PNL_ID, DFCT_CODE, TRANS_DATE, X_AXIS, Y_AXIS
                        FROM MERDA1.WP_DEFTHIS
                        WHERE DEFT_OPER = :deft_oper
                          AND IF_NEWER = 'Y'
                          AND TRANS_DATE >= :min_trans_date
                          AND PNL_ID IN ({placeholders})
                        ORDER BY PNL_ID, TRANS_DATE
                    """
                    cursor.execute(sql, {
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
                columns_sql = ", ".join(WP_DEFTHIS_COLUMNS)
                cursor.execute(
                    f"""
                        SELECT {columns_sql}
                        FROM MERDA1.WP_DEFTHIS
                        WHERE PNL_ID = :panel_id
                          AND DEFT_OPER = :deft_oper
                          AND IF_NEWER = 'Y'
                          AND TRANS_DATE >= :min_trans_date
                        ORDER BY TRANS_DATE, TRANS_NBR
                    """,
                    {
                        "panel_id": normalized_panel_id,
                        "deft_oper": "1600",
                        "min_trans_date": min_trans_date.strftime("%Y-%m-%d %H.%M.%S.%f"),
                    },
                )
                rows = [dict(zip(WP_DEFTHIS_COLUMNS, values)) for values in cursor]
            finally:
                cursor.close()
        finally:
            connection.close()
        logger.info(
            "[MES Report] Detail query complete: panel=%s, rows=%d, elapsed=%.2fs",
            normalized_panel_id, len(rows), time.monotonic() - started_at,
        )
        return rows
