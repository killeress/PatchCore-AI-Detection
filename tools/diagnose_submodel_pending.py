"""Read-only report for a missing OK-tiles retrain button; standard library only.

Run from the server project directory:
    python tools/diagnose_submodel_pending.py --db PATH --bundle BUNDLE_DIRECTORY
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3


def diagnose(conn, bundle_dir, lighting, tile_id=None):
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    bundles = [dict(r) for r in conn.execute(
        "SELECT id, bundle_path, job_id FROM model_registry"
    ) if Path(r["bundle_path"]).resolve() == bundle_dir]
    if len(bundles) != 1:
        raise ValueError(f"Expected one registered bundle at {bundle_dir}, found {len(bundles)}")
    bundle = bundles[0]
    report = {"bundle": bundle, "units": {}}
    for zone in ("inner", "edge"):
        label = f"{lighting}-{zone}"
        history = (manifest.get("submodel_history") or {}).get(label) or []
        latest = history[-1] if history else {}
        metrics = (manifest.get("unit_metrics") or {}).get(label) or {}
        ids = latest.get("used_tile_ids")
        baseline_source = "submodel_history"
        if ids is None:
            ids = metrics.get("used_tile_ids")
            baseline_source = "unit_metrics" if ids is not None else "legacy_reject_count"
        last_used = set(map(int, ids)) if ids is not None else None
        pending_job = latest.get("job_id") or latest.get("trained_with_job_id") or bundle["job_id"]
        groups = {}
        for name, job in (("displayed", bundle["job_id"]), ("pending", pending_job)):
            rows = [dict(r) for r in conn.execute(
                "SELECT * FROM training_tile_pool WHERE job_id = ? "
                "AND lighting = ? AND zone = ? AND source = 'ok' ORDER BY id",
                (job, lighting, zone),
            )]
            accepted = {r["id"] for r in rows if r["decision"] == "accept"
                        and r.get("dataset_role", "train") == "train"}
            count = (len(accepted ^ last_used) if last_used is not None
                     else sum(r["decision"] == "reject" for r in rows)) if job else 0
            groups[name] = {
                "job_id": job,
                "tile_count": len(rows),
                "role_decision_counts": dict(Counter(
                    f"{r.get('dataset_role', 'train')}/{r['decision']}" for r in rows
                )),
                "difference_count": count,
                "added_ids_first_20": sorted(accepted - last_used)[:20] if last_used is not None else [],
                "removed_ids_first_20": sorted(last_used - accepted)[:20] if last_used is not None else [],
            }
        report["units"][label] = {
            "latest_training": {k: latest.get(k) for k in (
                "kind", "trained_at", "job_id", "trained_with_job_id", "tile_count_used"
            )},
            "baseline_source": baseline_source,
            "baseline_tile_count": len(last_used) if last_used is not None else None,
            "source_job_mismatch": bundle["job_id"] != pending_job,
            **groups,
        }
    if tile_id is not None:
        row = conn.execute("SELECT * FROM training_tile_pool WHERE id = ?", (tile_id,)).fetchone()
        report["selected_tile"] = ({k: dict(row).get(k) for k in (
            "id", "job_id", "lighting", "zone", "source", "decision", "dataset_role"
        )} if row else None)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--lighting", default="WGF50500")
    parser.add_argument("--tile-id", type=int, help="ID of the tile just changed (from image tooltip)")
    args = parser.parse_args()
    with sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        report = diagnose(conn, args.bundle.resolve(), args.lighting, args.tile_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
