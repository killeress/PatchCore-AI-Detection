"""產線 scratch classifier 診斷工具.

執行方式 (在產線 /root/Code/CAPI_AD 目錄底下):
    cd /root/Code/CAPI_AD && python diag_scratch.py

會檢查：
  1. 正在跑的 server PID + 它開著哪些 yaml/db/pkl/pth 檔
  2. 從 capi_3f.yaml 讀進來的 scratch 設定值
  3. 套用 DB config_params override 後的最終值
  4. 直接嘗試載入 ScratchClassifier (驗證整條 path)
  5. 模擬 CAPIInferencer._get_scratch_filter() 的判斷邏輯
"""
import glob
import os
import sys
import traceback
from pathlib import Path

ROOT = Path("/root/Code/CAPI_AD")
PID_FILE = Path("/tmp/capi_server.pid")
DB_PATH = "/data/capi_ai/capi_results.db"
CONFIG_PATH = ROOT / "configs" / "capi_3f.yaml"


def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# --- 1. server PID + open files ---
banner("1. Running server process")
pid = None
try:
    pid = int(PID_FILE.read_text().strip())
    print(f"PID file: {PID_FILE} -> {pid}")
except Exception as e:
    print(f"Cannot read {PID_FILE}: {e}")

if pid:
    if not Path(f"/proc/{pid}").exists():
        print(f"  WARN: PID {pid} not alive (stale pidfile)")
        pid = None

if pid:
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_text().replace("\x00", " ").strip()
        print(f"cmdline: {cmdline}")
    except Exception as e:
        print(f"  cannot read cmdline: {e}")

    print("Open files (only yaml/db/pkl/pth):")
    found_any = False
    for fd in glob.glob(f"/proc/{pid}/fd/*"):
        try:
            target = os.readlink(fd)
            if any(x in target for x in (".yaml", ".db", ".pkl", ".pth")):
                print(f"  {target}")
                found_any = True
        except OSError:
            pass
    if not found_any:
        print("  (none currently open — that's normal for SQLite outside transactions)")


# --- 2. Load CAPIConfig from YAML ---
banner("2. CAPIConfig loaded from YAML")
sys.path.insert(0, str(ROOT))
try:
    from capi_config import CAPIConfig
    cfg = CAPIConfig.from_yaml(str(CONFIG_PATH))
    print(f"YAML path: {CONFIG_PATH}")
    print(f"  scratch_classifier_enabled    = {cfg.scratch_classifier_enabled}")
    print(f"  scratch_safety_multiplier     = {cfg.scratch_safety_multiplier}")
    print(f"  scratch_bundle_path           = {cfg.scratch_bundle_path}")
    print(f"  scratch_dinov2_weights_path   = {cfg.scratch_dinov2_weights_path}")
    print(f"  scratch_dinov2_repo_path      = {cfg.scratch_dinov2_repo_path}")
except Exception as e:
    print(f"FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)


# --- 3. Apply DB overrides (same as server boot does) ---
banner("3. After DB config_params override")
try:
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    rows = list(conn.execute(
        "SELECT param_name, param_value FROM config_params"
    ))
    print(f"DB: {DB_PATH}  ({len(rows)} total config_params rows)")
    scratch_rows = [r for r in rows if "scratch" in r[0]]
    if not scratch_rows:
        print("  (no scratch_* rows in DB — YAML values stay as-is)")
    else:
        for r in scratch_rows:
            print(f"  DB row: {r[0]} = {r[1]!r}")

    overrides = [
        {"param_name": r[0], "decoded_value": r[1]}
        for r in rows
    ]
    cfg.apply_db_overrides(overrides)
    print()
    print("After apply_db_overrides:")
    print(f"  scratch_classifier_enabled    = {cfg.scratch_classifier_enabled}")
    print(f"  scratch_safety_multiplier     = {cfg.scratch_safety_multiplier}")
    print(f"  scratch_bundle_path           = {cfg.scratch_bundle_path}")
    print(f"  scratch_dinov2_weights_path   = {cfg.scratch_dinov2_weights_path}")
    print(f"  scratch_dinov2_repo_path      = {cfg.scratch_dinov2_repo_path}")
except Exception as e:
    print(f"FAIL: {e}")
    traceback.print_exc()


# --- 4. Try loading ScratchClassifier directly ---
banner("4. ScratchClassifier load test (offline)")
try:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from scratch_classifier import ScratchClassifier
    clf = ScratchClassifier(
        bundle_path=str(ROOT / cfg.scratch_bundle_path),
        dinov2_weights_path=str(ROOT / cfg.scratch_dinov2_weights_path)
            if cfg.scratch_dinov2_weights_path else None,
        dinov2_repo_path=str(ROOT / cfg.scratch_dinov2_repo_path)
            if cfg.scratch_dinov2_repo_path else None,
        device="cuda",
    )
    print(f"OK — ScratchClassifier loaded.")
    print(f"   conformal_threshold = {clf.conformal_threshold:.6f}")
    print(f"   effective @ safety {cfg.scratch_safety_multiplier} = "
          f"{min(clf.conformal_threshold * cfg.scratch_safety_multiplier, 0.9999):.6f}")
except Exception as e:
    print(f"FAIL: {e}")
    traceback.print_exc()


# --- 5. Mimic CAPIInferencer._get_scratch_filter logic ---
banner("5. _get_scratch_filter() decision tree")
print(f"check 1: scratch_classifier_enabled = {cfg.scratch_classifier_enabled}")
if not cfg.scratch_classifier_enabled:
    print("  -> RETURN None (filter disabled)")
else:
    print("  -> pass")

print(f"\ncheck 2: _scratch_load_failed sentinel — fresh inferencer = False")
print("  -> pass (server's actual sentinel state may differ if a prior load failed)")

print(f"\nWould attempt ScratchClassifier(...) with:")
print(f"  bundle = {cfg.scratch_bundle_path}")
print(f"  weights= {cfg.scratch_dinov2_weights_path}")
print(f"  repo   = {cfg.scratch_dinov2_repo_path}")

print()
print("=" * 60)
print("DONE.  If section 4 succeeded but server log has no 'ScratchClassifier")
print("loaded' line after restart, the running server is most likely loading")
print("an OLD in-memory module. Verify by killing PID and restarting cleanly.")
print("=" * 60)
