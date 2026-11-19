"""CarUX CAPI 站點公共網盤 AOI Report 掛載路徑與唯讀查找。"""

from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import re
from typing import Optional


logger = logging.getLogger("capi.inference")
DEFAULT_CARUX_AOI_REPORT_ROOT = "/10.170.5.102"
CARUX_REPORT_NOT_FOUND = object()


def uses_carux_report(server_config, machine_no) -> bool:
    """僅 CARUX 的 CAPI 加四位數機台使用公共網盤報告。"""
    config = server_config or {}
    facility = str((config.get("mes_report") or {}).get("facility") or "").strip().upper()
    machine = str(machine_no or "").strip()
    return facility == "CARUX" and re.fullmatch(r"CAPI[0-9]{4}", machine, re.IGNORECASE | re.ASCII) is not None


def resolve_carux_report_dir(server_config, machine_no, panel_dir) -> Optional[Path]:
    """為符合規則的 CARUX CAPI 機台建立日期/CAHM編號/OK 路徑。"""
    config = server_config or {}
    if not uses_carux_report(config, machine_no):
        return None

    date = Path(panel_dir).parent.name
    try:
        if not re.fullmatch(r"\d{8}", date):
            raise ValueError("invalid date format")
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        logger.error("AOI Report[CARUX]: 圖片目錄日期無效，無法查詢公共網盤報告: %s", panel_dir)
        return None

    root = config.get("carux_aoi_report_root") or DEFAULT_CARUX_AOI_REPORT_ROOT
    machine_folder = "CAHM" + str(machine_no).strip()[-4:]
    return Path(root) / date / machine_folder / "OK"


def read_carux_report_lines(report_dir: Path, glass_id: str, model_id: str, date: str):
    """回傳報告行、未找到標記；其他讀取錯誤回傳 None。"""
    if not glass_id or not model_id or not re.fullmatch(r"\d{8}", str(date)):
        logger.warning("AOI Report[CARUX]: 缺少 PID、機種或日期，無法查詢公共網盤報告")
        return None

    try:
        requested_date = datetime.strptime(date, "%Y%m%d")
    except ValueError:
        return None

    report_dir = Path(report_dir)
    for day_offset in (0, 1, -1):
        try:
            candidate_date = (requested_date + timedelta(days=day_offset)).strftime("%Y%m%d")
        except OverflowError:
            continue
        candidate_dir = (
            report_dir if day_offset == 0 else
            report_dir.parent.parent.parent / candidate_date / report_dir.parent.name / report_dir.name
        )
        pattern = re.compile(
            rf"^{re.escape(glass_id)}_{re.escape(candidate_date)}\d{{6}}_{re.escape(model_id)}_\d{{6}}\.txt$",
            re.IGNORECASE,
        )
        try:
            matches = []
            with os.scandir(candidate_dir) as entries:
                for entry in entries:
                    if pattern.fullmatch(entry.name) and entry.is_file():
                        matches.append((entry.stat().st_mtime, entry.name))
            if not matches:
                continue

            _, filename = max(matches)
            report_file = candidate_dir / filename
            with open(report_file, "r", encoding="utf-8") as file:
                lines = file.readlines()
            if day_offset:
                logger.warning(
                    "[REPORT_DATE_FALLBACK] Requested=%s Resolved=%s (date %+d day)",
                    report_dir, candidate_dir, day_offset,
                )
            logger.info("AOI Report[CARUX]: 讀取公共網盤報告 %s", report_file)
            return lines
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError) as exc:
            logger.warning("AOI Report[CARUX]: 公共網盤報告無法讀取: glass=%s dir=%s error=%s", glass_id, candidate_dir, exc)
            return None
    return CARUX_REPORT_NOT_FOUND
