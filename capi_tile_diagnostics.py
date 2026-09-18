"""Persisted tile decision evidence and conservative historical-log fallback."""
import json
import math
import re


_NUMBER = r"\d+(?:\.\d+)?"
_RESCUE = re.compile(
    rf"EDGE_LIGHT_LEAK_RESCUE:\s*(BRIGHT_LEAK|DARK_DROP)\s+"
    rf"(TOP|BOTTOM|LEFT|RIGHT)\s+delta=({_NUMBER})\s+"
    rf"length=(\d+)\s+dust=({_NUMBER})\s*(?:->|→)\s*REAL_NG"
)
_TILE_LINE = re.compile(r"(?P<image>[^\s]+\.(?:tif|tiff|png|jpg|bmp))\s+Tile@\((?P<x>\d+),(?P<y>\d+)\)", re.I)
_SIDES = {"top": "上側", "bottom": "下側", "left": "左側", "right": "右側"}


def historical_tile_details(log):
    """Index full filename + tile origin; ambiguous repeated runs are not inferred."""
    lines = {}
    for line in str(log or "").splitlines():
        match = _TILE_LINE.search(line)
        if match:
            key = (match["image"], int(match["x"]), int(match["y"]))
            # Even identical coordinates can refer to different AOI points/runs.
            lines[key] = None if key in lines else line
    return lines


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError):
        return None


def edge_ng_evidence(tile, historical_detail=None):
    """Only explain a final NG actually rescued by the edge rule, not a debug hit."""
    if not tile.get("is_anomaly") or any(tile.get(key) for key in (
        "is_dust", "is_bomb", "scratch_filtered", "is_exclude_zone",
    )):
        return None
    raw = tile.get("edge_light_leak_result")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw else None
        except (ValueError, TypeError):
            raw = None
    detail = str(tile.get("dust_detail_text") or "")
    source = "stored"
    if isinstance(raw, dict):
        # Structured results are authoritative, even when the old log disagrees.
        if not raw.get("detected") or not raw.get("rescue_applied"):
            return None
        data = raw
    else:
        if not detail:
            detail = historical_detail or ""
            source = "log"
        match = _RESCUE.search(detail)
        if not match:
            return None
        data = dict(anomaly_type=match[1], side=match[2].lower(),
                    max_delta=match[3], continuous_length=match[4], dust_overlap=match[5])
    kind, side = data.get("anomaly_type"), data.get("side")
    if kind not in ("BRIGHT_LEAK", "DARK_DROP") or side not in _SIDES:
        return None
    historical = kind == "DARK_DROP"
    return {
        "side": side,
        "anomaly_type": kind,
        "title": _SIDES[side] + ("暗段" if historical else "泛白"),
        "rule_label": "邊緣檢查／舊規則" if historical else "邊緣漏光",
        "historical": historical,
        "source": source,
        "two_stage": "TWO_STAGE:" in detail,
        **{key: _number(data.get(key)) for key in (
            "max_delta", "continuous_length", "dust_overlap", "threshold",
            "min_length", "max_dust_overlap",
        )},
    }


def decorate_edge_ng_evidence(record):
    history = historical_tile_details(record.get("inference_log"))
    for image in record.get("images", []):
        counts = {}
        for tile in image.get("tiles", []):
            key = (tile.get("x"), tile.get("y"))
            counts[key] = counts.get(key, 0) + 1
        for tile in image.get("tiles", []):
            origin = (tile.get("x"), tile.get("y"))
            old_detail = history.get((image.get("image_name"), *origin)) if counts[origin] == 1 else None
            tile["edge_ng_evidence"] = edge_ng_evidence(tile, old_detail)
            tile["decision_evidence"] = decision_evidence(tile)
        image["has_edge_ng"] = any(t["edge_ng_evidence"] for t in image.get("tiles", []))
        for edge in image.get("edge_defects", []):
            edge["decision_evidence"] = decision_evidence(edge, is_cv=True)
        image["ng_reason_labels"] = list(dict.fromkeys(
            item["decision_evidence"]["rule_label"]
            for item in image.get("tiles", []) + image.get("edge_defects", [])
            if item.get("decision_evidence") and item["decision_evidence"]["status"] == "NG"
        ))


def _context(raw):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return {}
    return raw if isinstance(raw, dict) else {}


def tile_decision_context(tile):
    """Snapshot the actual detector settings, never the current global settings."""
    context = dict(_context(getattr(tile, "decision_context", None)))
    context["detector"] = "bright_spot" if getattr(tile, "is_bright_spot_detection", False) else "patchcore"
    context["score_threshold"] = _number(getattr(tile, "score_threshold", None))
    if context["detector"] == "bright_spot":
        for name in ("max_diff", "diff_threshold", "area", "min_area"):
            context[name] = _number(getattr(tile, "bright_spot_" + name, None))
    return context


def _metric(label, value, threshold=None, *, op="≥", unit="", digits=2):
    def fmt(raw):
        n = _number(raw)
        return "未記錄" if n is None else f"{n:.{digits}f}{unit}"
    return dict(label=label, value=fmt(value), threshold=(op + " " + fmt(threshold))
                if _number(threshold) is not None else "未記錄")


def decision_evidence(item, is_cv=False):
    """Explain final status separately from intermediate detector/filter results."""
    ctx = _context(item.get("decision_context"))
    filters = [label for flag, label in (
        ("is_exclude_zone", "排除區"), ("is_bomb", "BOMB 比對"),
        ("scratch_filtered", "刮痕過濾"), ("is_dust", "灰塵過濾"),
    ) if item.get(flag)]
    ng = not filters and (not item.get("is_cv_ok") if is_cv else bool(item.get("is_anomaly")))
    status = "NG" if ng else "OK"
    edge = item.get("edge_ng_evidence")
    if edge:
        return dict(edge, status="NG", kind="edge", metrics=[], notes=[])
    metrics, notes = [], []
    if is_cv:
        mode = item.get("inspector_mode") or "cv"
        source = item.get("source_inspector") if mode == "fusion" else mode
        if source == "patchcore":
            label, title = "邊緣 PatchCore", "模型分數達門檻"
            threshold = item.get("patchcore_threshold")
            if not ctx and not threshold:
                threshold = None
            metrics.append(_metric("模型分數", item.get("patchcore_score"), threshold, digits=4))
            notes.append("此路徑依模型分數判定；缺陷面積僅供參考。")
        elif source == "cv":
            label, title = "CV 邊緣", "邊緣缺陷檢查"
            rules = ctx.get("rules", [])
            if not isinstance(rules, list):
                rules = []
            for index, rule in enumerate(rules, 1):
                if not isinstance(rule, dict):
                    continue
                prefix = f"候選 {index} · " if len(rules) > 1 else ""
                if rule.get("kind") == "thin_line":
                    metrics.extend([
                        _metric(prefix + "線段活化長度", rule.get("length"), rule.get("min_length"), unit=" px", digits=0),
                        _metric(prefix + "線寬", rule.get("width"), rule.get("max_width"), op="≤", unit=" px", digits=0),
                    ])
                    notes.append("細線分支依活化長度與線寬判定，不套用連通區面積門檻。")
                elif rule.get("kind") == "component":
                    metrics.extend([
                        _metric(prefix + "連通區面積", rule.get("area"), rule.get("min_area"), unit=" px", digits=0),
                        _metric(prefix + "最大灰階差", rule.get("max_diff"), rule.get("threshold"), op=">", digits=0),
                    ])
                    if rule.get("min_max_diff", 0) > 0:
                        metrics.append(_metric(prefix + "對比複核", rule.get("max_diff"), rule["min_max_diff"], digits=0))
                    if rule.get("min_solidity", 0) > 0:
                        metrics.append(_metric(prefix + "實心度", rule.get("solidity"), rule["min_solidity"], digits=3))
            if not metrics:
                metrics = [_metric("記錄面積（參考）", item.get("area"), unit=" px", digits=0),
                           _metric("最大灰階差", item.get("max_diff"), item.get("threshold_used") or None, op=">", digits=0)]
                notes.append("此筆未保存連通區／細線分支依據，無法還原完整判定門檻。")
        else:
            label, title = "混合邊緣檢查", "缺陷來源未記錄"
            notes.append("此筆混合模式未記錄缺陷來源，無法區分 CV 或 PatchCore。")
        if mode == "fusion":
            label = "混合模式 · " + label
        if item.get("fusion_fallback_reason"):
            notes.append("模式回退：" + str(item["fusion_fallback_reason"]))
    elif ctx.get("detector") == "bright_spot" or item.get("zone") == "bright_spot":
        label, title = "亮點偵測", "亮點條件命中" if ng else "亮點檢查"
        metrics = [
            _metric("全 tile 最大灰階差", ctx.get("max_diff"), ctx.get("diff_threshold"), op=">", digits=0),
            _metric("全 tile 最大亮度", ctx.get("max_pixel"), ctx.get("abs_threshold"), op=">", digits=0),
            _metric("最大連通區面積", ctx.get("max_component_area"), ctx.get("min_area"), unit=" px", digits=0),
        ]
        notes.append("灰階差或絕對亮度任一條件命中後，有效區域內仍須有連通區面積達標；全 tile 最大值僅供參考。")
    else:
        label, title = "PatchCore", "模型分數達門檻"
        metrics = [_metric("異常分數", item.get("score"), ctx.get("score_threshold"), digits=4)]
        if not ctx:
            notes.append("舊紀錄未保存當時的模型門檻，不套用目前設定。")
    if filters:
        title = "、".join(filters) + " → OK"
        notes.append("此項目已過濾，不計入最終 NG。")
        if item.get("is_bomb") and item.get("bomb_code"):
            notes.append("BOMB 缺陷碼：" + str(item["bomb_code"]))
    if item.get("scratch_filtered") or (_number(item.get("scratch_score")) or 0) > 0:
        metrics.append(_metric("刮痕複核分數", item.get("scratch_score"), digits=4))
    if not filters and not ng:
        title = "檢查通過"
        if is_cv and item.get("patchcore_ok_reason"):
            notes.append(str(item["patchcore_ok_reason"]))
    detail = str(item.get("dust_detail_text") or "")
    flow = [label] + filters + ["最終 " + status]
    return dict(kind="general", status=status, title=title, rule_label=label,
                metrics=metrics, notes=list(dict.fromkeys(notes)), flow=" → ".join(flow),
                detail=detail, filtered=bool(filters))
