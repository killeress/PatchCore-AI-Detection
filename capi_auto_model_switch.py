"""自動切換模型規則解析。

此模組只處理低成本的資料判斷；真正載入模型與切換 runtime 由 capi_server 負責。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


DEFAULT_SERIES_PREFIX = "__DEFAULT__"
SERIES_PREFIX_LENGTH = 8


def requested_series_prefix(model_id: str) -> str:
    """Client 機種欄位取前 8 碼作為系列名。"""
    return str(model_id or "").strip().upper()[:SERIES_PREFIX_LENGTH]


def normalize_series_prefix(series_prefix: str) -> str:
    prefix = str(series_prefix or "").strip().upper()
    if prefix == DEFAULT_SERIES_PREFIX:
        return DEFAULT_SERIES_PREFIX
    if len(prefix) != SERIES_PREFIX_LENGTH:
        raise ValueError(f"系列名必須是機種前 {SERIES_PREFIX_LENGTH} 碼")
    return prefix


def bundle_label(bundle: Optional[Dict]) -> str:
    if not bundle:
        return ""
    return Path(str(bundle.get("bundle_path", "") or "")).name or str(bundle.get("machine_id", "") or "")


def select_target_bundle(db, requested_model_id: str) -> Dict:
    """依 Client 機種選出目標 bundle。

    回傳格式：
        {
            "requested_model_id": str,
            "series_prefix": str,
            "rule": dict | None,
            "bundle": dict | None,
            "used_default": bool,
            "reason": "matched" | "fallback_default" | "not_configured" | "bundle_missing",
            "message": str,
        }
    """
    raw_model_id = str(requested_model_id or "").strip()
    series = requested_series_prefix(raw_model_id)
    if not series:
        return {
            "requested_model_id": raw_model_id,
            "series_prefix": "",
            "rule": None,
            "bundle": None,
            "used_default": False,
            "reason": "not_configured",
            "message": "Client 未提供機種 ID，略過自動切換",
        }

    rule = db.get_auto_model_switch_rule_by_series(series)
    used_default = False
    if rule is None:
        rule = db.get_default_auto_model_switch_rule()
        used_default = rule is not None

    if rule is None:
        return {
            "requested_model_id": raw_model_id,
            "series_prefix": series,
            "rule": None,
            "bundle": None,
            "used_default": False,
            "reason": "not_configured",
            "message": "尚未設定此系列與預設模型，略過自動切換",
        }

    bundle = db.get_model_bundle(int(rule["bundle_id"]))
    if not bundle:
        return {
            "requested_model_id": raw_model_id,
            "series_prefix": series,
            "rule": rule,
            "bundle": None,
            "used_default": used_default,
            "reason": "bundle_missing",
            "message": f"自動切換規則指向不存在的 bundle_id={rule['bundle_id']}",
        }

    return {
        "requested_model_id": raw_model_id,
        "series_prefix": series,
        "rule": rule,
        "bundle": bundle,
        "used_default": used_default,
        "reason": "fallback_default" if used_default else "matched",
        "message": "使用預設模型" if used_default else "命中系列模型",
    }
