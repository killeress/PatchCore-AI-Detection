"""MES 人工判定與 AI 推論結果比對。"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


try:
    from capi_mes_credentials import ORACLE_MES_PASSWORD
except ImportError:
    ORACLE_MES_PASSWORD = ""


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

    def fetch_defects(self, panel_ids: Sequence[str]) -> Dict[str, List[Dict]]:
        panel_ids = sorted({str(value or "").strip().upper() for value in panel_ids if str(value or "").strip()})
        if not panel_ids:
            return {}
        try:
            import oracledb
        except ImportError as exc:
            raise MESReportConfigurationError("Server 尚未安裝 python-oracledb，請執行 pip install -r requirements.txt") from exc

        dsn = oracledb.makedsn(self.host, self.port, service_name=self.service_name)
        connection = oracledb.connect(user=self.user, password=self.password, dsn=dsn)
        result: Dict[str, List[Dict]] = {}
        try:
            cursor = connection.cursor()
            try:
                for offset in range(0, len(panel_ids), 900):
                    chunk = panel_ids[offset:offset + 900]
                    panel_binds = {f"panel_{idx}": value for idx, value in enumerate(chunk)}
                    placeholders = ", ".join(f":panel_{idx}" for idx in range(len(chunk)))
                    sql = f"""
                        SELECT PNL_ID, DFCT_CODE, TRANS_DATE, X_AXIS, Y_AXIS
                        FROM MERDA1.WP_DEFTHIS
                        WHERE DEFT_OPER = :deft_oper
                          AND IF_NEWER = 'Y'
                          AND PNL_ID IN ({placeholders})
                        ORDER BY PNL_ID, TRANS_DATE
                    """
                    cursor.execute(sql, {
                        "deft_oper": "1600",
                        **panel_binds,
                    })
                    for pnl_id, code, trans_date, x_axis, y_axis in cursor:
                        key = str(pnl_id or "").strip().upper()
                        result.setdefault(key, []).append({
                            "pnl_id": key,
                            "dfct_code": code,
                            "trans_date": trans_date,
                            "x_axis": x_axis,
                            "y_axis": y_axis,
                        })
            finally:
                cursor.close()
        finally:
            connection.close()
        return result
