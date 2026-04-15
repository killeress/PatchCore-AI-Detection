"""CAPI AI 推論伺服器啟動腳本 (Python 版).

為什麼用 Python：bash script 在 Windows -> Linux 透過 FTP 傳輸時容易帶
CRLF 行尾，導致 `/bin/bash^M: bad interpreter`。Python 對 CRLF 完全免疫。

使用方式 (在產線執行):
    python start_server.py                 # 重啟 (預設, stop 舊的 + start 新的)
    python start_server.py start           # 啟動 (若已在跑會拒絕)
    python start_server.py stop            # 只停止
    python start_server.py status          # 看狀態
    python start_server.py -c xxx.yaml     # 配合任一動作指定 config
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PID_FILE = Path("/tmp/capi_server.pid")
LOG_DIR = Path("/data/capi_ai/logs")
HEATMAP_DIR = Path("/data/capi_ai/heatmaps")
DEFAULT_CONFIG = "server_config.yaml"
SERVER_SCRIPT = "capi_server.py"


def _is_capi_server(pid: int) -> bool:
    """Verify the given pid is actually a capi_server.py process."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_text(errors="ignore")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    return SERVER_SCRIPT in cmdline


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _find_running_pid():
    """Find the current capi_server pid via PID_FILE first, pgrep fallback."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            if _alive(pid) and _is_capi_server(pid):
                return pid
        except (ValueError, OSError):
            pass
    # Fallback: pgrep
    try:
        out = subprocess.run(
            ["pgrep", "-f", SERVER_SCRIPT],
            capture_output=True, text=True, check=False,
        )
        for line in out.stdout.splitlines():
            pid = int(line.strip())
            if _is_capi_server(pid):
                return pid
    except (FileNotFoundError, ValueError):
        pass
    return None


def cmd_status() -> int:
    pid = _find_running_pid()
    if pid is None:
        print("  Status: NOT RUNNING")
        return 1
    print(f"  Status: RUNNING (pid={pid})")
    return 0


def cmd_stop() -> int:
    print("Stopping server...")
    pid = _find_running_pid()
    if pid is None:
        print("  No running server found.")
        if PID_FILE.exists():
            PID_FILE.unlink()
        return 0

    print(f"  Sending SIGTERM to pid {pid}...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    # Wait up to 10 seconds
    for _ in range(10):
        if not _alive(pid):
            break
        time.sleep(1)

    if _alive(pid):
        print("  Still alive after 10s -- sending SIGKILL")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(1)

    if _alive(pid):
        print(f"  ERROR: cannot stop pid {pid}")
        return 1
    print(f"  Stopped (pid {pid}).")
    if PID_FILE.exists():
        PID_FILE.unlink()
    return 0


def _check_dependencies(python_exe: str) -> None:
    for mod in ("yaml", "cv2", "numpy"):
        rc = subprocess.call(
            [python_exe, "-c", f"import {mod}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if rc != 0:
            print(f"ERROR: Python module not installed: {mod}")
            sys.exit(1)
    print("Dependencies OK")


def _clear_pyc(root: Path) -> int:
    """Remove __pycache__ dirs (defensive: avoid stale .pyc loading old modules)."""
    removed = 0
    for cache in root.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)
            removed += 1
    return removed


def cmd_start(config_file: str) -> int:
    print("=" * 60)
    print("  CAPI AI Inference Server")
    print("=" * 60)
    print(f"  Working dir : {SCRIPT_DIR}")
    print(f"  Config      : {config_file}")
    print(f"  PID file    : {PID_FILE}")
    print("=" * 60)

    config_path = SCRIPT_DIR / config_file
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable
    print(f"  Python      : {python_exe}")

    print()
    print("Checking dependencies...")
    _check_dependencies(python_exe)

    n = _clear_pyc(SCRIPT_DIR)
    if n:
        print(f"Cleared {n} __pycache__ dirs (avoid stale .pyc)")

    print()
    print("Starting server...")
    # Use os.execv to replace this Python process with capi_server.py.
    # That way, the running capi_server.py inherits our PID -> matches PID_FILE.
    PID_FILE.write_text(str(os.getpid()))
    server_path = SCRIPT_DIR / SERVER_SCRIPT
    os.chdir(SCRIPT_DIR)
    os.execv(python_exe, [python_exe, str(server_path), "--config", config_file])
    # not reached
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CAPI AI server launcher (Python, CRLF-immune)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "action", nargs="?", default="restart",
        choices=["start", "stop", "restart", "status"],
        help="What to do (default: restart)",
    )
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG,
        help=f"Config file path (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    if args.action == "status":
        return cmd_status()
    if args.action == "stop":
        return cmd_stop()
    if args.action == "start":
        if _find_running_pid() is not None:
            print("Server already running. Use 'restart' or 'stop' first.")
            cmd_status()
            return 1
        return cmd_start(args.config)
    if args.action == "restart":
        print("Stopping existing server (if any)...")
        cmd_stop()
        time.sleep(1)
        return cmd_start(args.config)
    return 1


if __name__ == "__main__":
    sys.exit(main())
