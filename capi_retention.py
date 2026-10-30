"""Shared retention policy for the settings UI and scheduled cleanup."""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional


RETENTION_PARAM = "data_retention_policy"
RETENTION_DESCRIPTION = "資料保留與每日清理設定（全機台共用）"
RETENTION_DAY_LABELS = {
    "ok_retain_days": "OK／OK-i／OK-s 判定紀錄",
    "ng_retain_days": "NG／錯誤等其他判定紀錄",
    "tile_retain_days": "Tile 切塊明細",
    "heatmap_retain_days": "熱圖與相關檢查圖片",
}
RETENTION_KEYS = set(RETENTION_DAY_LABELS) | {
    "enabled", "schedule_time", "vacuum_after_cleanup",
}


def normalize_retention_policy(value: Any) -> Dict[str, Any]:
    """Validate a complete policy before saving or running destructive cleanup."""
    if not isinstance(value, dict) or set(value) != RETENTION_KEYS:
        raise ValueError("資料保留設定欄位不完整或包含未知欄位")
    result = dict(value)
    for key, label in (
        ("enabled", "每日自動清理"),
        ("vacuum_after_cleanup", "清理後回收資料庫空間"),
    ):
        if type(value[key]) is not bool:
            raise ValueError(f"{label}必須為開啟或關閉")
    schedule = value["schedule_time"]
    if not isinstance(schedule, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", schedule):
        raise ValueError("每日清理時間須為 00:00～23:59")
    for key, label in RETENTION_DAY_LABELS.items():
        minimum = 0 if key == "heatmap_retain_days" else 1
        if type(value[key]) is not int or not minimum <= value[key] <= 3650:
            raise ValueError(f"{label}保留天數須為 {minimum}～3650 的整數")
    return result


def initial_retention_policy(server_config: Optional[Dict] = None) -> Dict[str, Any]:
    """Adopt 180-day records once; preserve all other legacy cleanup settings.

    The persisted policy then takes precedence over YAML, including on restart
    and future upgrades. Legacy omitted tile/heatmap settings keep their old
    defaults (7 days / no image cleanup), not the production example's values.
    """
    cleanup = (server_config or {}).get("cleanup", {}) or {}
    return normalize_retention_policy({
        "enabled": cleanup.get("enabled", False),
        "schedule_time": cleanup.get("schedule_time", "02:00"),
        "ok_retain_days": 180,
        "ng_retain_days": 180,
        "tile_retain_days": cleanup.get("tile_retain_days", 7),
        "heatmap_retain_days": cleanup.get("heatmap_retain_days", 0),
        "vacuum_after_cleanup": cleanup.get("vacuum_after_cleanup", True),
    })


def resolve_retention_policy(server_config: Optional[Dict], saved_value: Any = None) -> Dict[str, Any]:
    if saved_value is not None:
        return normalize_retention_policy(saved_value)
    return initial_retention_policy(server_config)


def next_cleanup_time(now: datetime, schedule_time: str, last_run_date=None) -> datetime:
    hour, minute = map(int, schedule_time.split(":"))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now or last_run_date == now.date():
        target += timedelta(days=1)
    return target
