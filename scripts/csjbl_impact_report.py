#!/usr/bin/env python3
"""Read-only CSJBL impact estimate; same created_at/count basis as /api/status."""
import argparse
import json
import socket
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


def parse_json(value):
    try:
        return json.loads(value or "")
    except (TypeError, ValueError):
        return None


def classify(details_text, aoi_text):
    details, aoi = parse_json(details_text), parse_json(aoi_text)
    items = details if isinstance(details, list) else []
    complete = bool(items) and all(isinstance(item, dict) for item in items)
    passthrough = [item for item in items if isinstance(item, dict)
                   and item.get("type") == "aoi_passthrough"
                   and item.get("defect_code") == "CSJBL"]
    if passthrough and complete:
        return "only" if len(passthrough) == len(items) else "mixed"
    # Reruns can omit passthrough from ng_details; consult the saved AOI report,
    # never the original client response, which can describe an older judgment.
    aoi_items = [item for group in (aoi.values() if isinstance(aoi, dict) else [])
                 if isinstance(group, list) for item in group]
    if any(isinstance(item, dict) and item.get("defect_code") == "CSJBL"
           for item in items + aoi_items):
        return "unclassified"
    return "other" if complete else "missing"


def collect(db_path, start, end):
    conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(inference_records)")}
        required = {"created_at", "ai_judgment", "ng_details"}
        if not required <= columns:
            raise ValueError("缺少 inference_records 欄位：" + ", ".join(sorted(required - columns)))
        where = "datetime(created_at) >= datetime(?) AND datetime(created_at) < datetime(?)"
        summary = dict(conn.execute("""
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(ai_judgment IN ('OK', 'OK-i')), 0) AS ok,
                   COALESCE(SUM(ai_judgment LIKE 'NG%'), 0) AS ng,
                   COALESCE(SUM(ai_judgment LIKE 'ERR%'), 0) AS err
            FROM inference_records WHERE """ + where, (start, end)).fetchone())
        counts = dict.fromkeys(("only", "mixed", "unclassified", "other", "missing"), 0)
        aoi_column = "aoi_machine_coords" if "aoi_machine_coords" in columns else "''"
        for row in conn.execute(
            "SELECT ng_details, " + aoi_column + " FROM inference_records WHERE "
            + where + " AND ai_judgment LIKE 'NG%'", (start, end)
        ):
            counts[classify(row[0], row[1])] += 1
        return dict(summary, **counts)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="唯讀統計 CSJBL 對排片率的影響")
    parser.add_argument("--db", default="/aidata/capi_ai/capi_results.db")
    parser.add_argument("--start", help="開始時間（含），格式 YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--end", help="結束時間（不含），格式 YYYY-MM-DD HH:MM:SS")
    args = parser.parse_args()
    if bool(args.start) != bool(args.end):
        parser.error("--start 與 --end 請一起指定；皆省略時查當班")
    now = datetime.now(timezone(timedelta(hours=8))).replace(tzinfo=None)
    if args.start:
        try:
            start = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            parser.error("時間格式應為 YYYY-MM-DD HH:MM:SS")
    else:
        start = now.replace(hour=7, minute=30, second=0, microsecond=0)
        if now < start:
            start -= timedelta(hours=12)
        elif now >= start + timedelta(hours=12):
            start += timedelta(hours=12)
        end = start + timedelta(hours=12)
    if end <= start:
        parser.error("結束時間必須晚於開始時間")
    db_path = Path(args.db).expanduser()
    if not db_path.is_file():
        parser.error("找不到資料庫：" + str(db_path))
    start_text, end_text = (value.strftime("%Y-%m-%d %H:%M:%S") for value in (start, end))
    try:
        result = collect(db_path, start_text, end_text)
    except (sqlite3.Error, ValueError) as exc:
        parser.exit(1, "無法讀取資料庫：" + str(exc) + "\n")
    total, ng = result["total"], result["ng"]
    csjbl = result["only"] + result["mixed"] + result["unclassified"]

    def rate(count, denominator=total):
        return "—" if not denominator else "{:.2f}%".format(100 * count / denominator)

    print("主機：{}  資料庫：{}".format(socket.gethostname(), db_path))
    print("區間：[{}, {})  UTC+8；created_at 與中控當班統計一致".format(start_text, end_text))
    print("總投入：{total}  OK/OK-i：{ok}  NG：{ng}  ERR：{err}".format(**result))
    print("目前 AI 排片率：{}（分母含 ERR）".format(rate(ng)))
    print("含 CSJBL 的 NG：{}；占 NG {}，占投入 {}".format(csjbl, rate(csjbl, ng), rate(csjbl)))
    print("  A. 明細只有 CSJBL：{}；占 NG {}".format(result["only"], rate(result["only"], ng)))
    print("  B. CSJBL + 其他 NG：{}".format(result["mixed"]))
    print("  C. 有 CSJBL，但明細不足以區分 A/B：{}".format(result["unclassified"]))
    print("  其他 NG 未記錄 CSJBL：{}；NG 明細缺漏/無法解析：{}".format(result["other"], result["missing"]))
    print("情境 1｜只將 A 轉 OK：{}；下降 {} 個百分點".format(rate(ng - result["only"]), rate(result["only"]).rstrip("%")))
    print("情境 2｜所有已辨識含 CSJBL 的 NG 都轉 OK：{}；下降 {} 個百分點".format(rate(ng - csjbl), rate(csjbl).rstrip("%")))
    print("情境 2 是樂觀假設；B 的其他缺陷仍須檢查，C 不能確認可放行。")
    print("每筆推論計一次，同筆多個 CSJBL 不重複；重送另計，重推採目前判定。")
    print("僅讀取本台現存紀錄；不修改資料、不重新推論；跨線 19.7% 須各台同區間後取平均。")
    if result["missing"] or result["unclassified"]:
        print("注意：有明細不足資料，CSJBL 分類/數量可能不完整，不能當成精確放行率。")
    if not total:
        print("此區間無紀錄；請確認資料庫路徑、時間及紀錄是否已清理。")
    result.update(host=socket.gethostname(), db=str(db_path.resolve()), start=start_text,
                  end=end_text, queried_at=now.isoformat(timespec="seconds"), csjbl_ng=csjbl)
    print("CSJBL_RESULT=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
