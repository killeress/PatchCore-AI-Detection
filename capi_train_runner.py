"""新機種 PatchCore 訓練 subprocess CLI。

由 capi_web._train_new_training_worker 用 subprocess.Popen 啟動，獨立 Python
process 跑訓練，讓推論 server 可以同時繼續服務 AOI 請求。GPU VRAM 透過
torch.cuda.set_per_process_memory_fraction 在兩邊各自切上限避免互搶。

Usage:
    python -m capi_train_runner \
        --job-id <id> \
        --server-config server_config.yaml \
        --log-file <path> \
        --cancel-flag <path>
"""
from __future__ import annotations

import os
# 必須在 import torch / anomalib 之前設定
os.environ.setdefault("TRUST_REMOTE_CODE", "1")
# 把 tqdm 進度刷新間隔拉到 5 秒：搭配 stderr→logging 接管後，每 5 秒一行進度
# 避免 17 分鐘 coreset 產生上萬筆 log
os.environ.setdefault("TQDM_MININTERVAL", "5")
import warnings as _warnings
_warnings.filterwarnings("ignore", message=r".*xFormers is not available.*")

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

# 確保可以 import 同目錄模組
sys.path.insert(0, str(Path(__file__).resolve().parent))


class _FileCancelEvent:
    """提供 .is_set() 介面，檢查 cancel flag 檔是否存在。

    web 端寫此檔即等同於對 subprocess 發送取消訊號；run_training_pipeline 在
    每個 unit 開始與訓練後都會檢查。

    第一次偵測到 flag 存在時會 log 一次取證資訊（檔 mtime / runner 啟動時間），
    幫助事後追查「為什麼訓練被取消」。
    """

    def __init__(self, flag_path: Path, started_at: datetime, logger: logging.Logger):
        self._path = flag_path
        self._started_at = started_at
        self._logger = logger
        self._logged = False

    def is_set(self) -> bool:
        try:
            st = self._path.stat()
        except FileNotFoundError:
            return False
        if not self._logged:
            self._logged = True
            mtime = datetime.fromtimestamp(st.st_mtime)
            self._logger.info(
                f"偵測到 cancel flag: path={self._path}, "
                f"flag_mtime={mtime.isoformat(timespec='seconds')}, "
                f"runner_started_at={self._started_at.isoformat(timespec='seconds')}"
            )
        return True


def _parse_args():
    p = argparse.ArgumentParser(description="CAPI 新機種 PatchCore 訓練 runner")
    p.add_argument("--job-id", required=True)
    p.add_argument("--server-config", required=True, help="server_config.yaml 路徑")
    p.add_argument("--log-file", required=True, help="輸出 log 檔（append 模式）")
    p.add_argument("--cancel-flag", required=True, help="cancel flag 檔路徑（存在即取消）")
    return p.parse_args()


class _StderrTee:
    """攔 sys.stderr 寫入轉成 logging。

    主要為了接住 anomalib coreset 的 tqdm 進度條（tqdm 寫到 stderr 用 \\r 覆寫
    同一行）。把 \\r 視為 line break，讓每一筆進度更新成為一行 log，supervisor
    端的 tail thread 就能同步顯示到前端 log_lines。
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._buf = ""

    def write(self, s):
        if not isinstance(s, str):
            try:
                s = s.decode("utf-8", errors="replace")
            except Exception:
                return
        s = s.replace("\r", "\n")
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                self._logger.info(line)

    def flush(self):
        if self._buf.strip():
            self._logger.info(self._buf.rstrip())
            self._buf = ""

    def isatty(self):
        return False


def _setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)
    # lightning / anomalib 的訓練進度也轉進來
    for name in ("lightning", "lightning_fabric", "pytorch_lightning", "anomalib"):
        lg = logging.getLogger(name)
        if lg.level == logging.NOTSET or lg.level > logging.INFO:
            lg.setLevel(logging.INFO)
    # tqdm 寫到 stderr，接住轉 logging（coreset 進度條才能進 log file）
    sys.stderr = _StderrTee(logging.getLogger("capi.stderr"))


def _apply_vram_limit(fraction: float, log: logging.Logger) -> None:
    if not fraction or fraction <= 0:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(fraction), 0)
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info(
                f"GPU memory fraction = {fraction} "
                f"(~{total_gb * fraction:.1f} / {total_gb:.1f} GB)"
            )
    except Exception as e:
        log.warning(f"無法套用 gpu_memory_fraction: {e}")


def _load_server_config(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_training_paths(server_cfg: dict, base_dir: Path) -> dict:
    training_cfg = server_cfg.get("training", {}) or {}

    def resolve(key: str, default: str) -> Path:
        val = training_cfg.get(key, default)
        p = Path(val)
        return p if p.is_absolute() else base_dir / p

    required = training_cfg.get("required_backbones") or [
        "wide_resnet50_2-32ee1156.pth"
    ]
    return {
        "over_review_root": resolve(
            "over_review_root", "/aidata/capi_ai/datasets/over_review"
        ),
        "backbone_cache_dir": resolve(
            "backbone_cache_dir", "deployment/torch_hub_cache"
        ),
        "output_root": resolve("output_root", "model"),
        "required_backbones": list(required),
        "gpu_memory_fraction": float(
            training_cfg.get("gpu_memory_fraction", 0) or 0
        ),
    }


def main() -> int:
    args = _parse_args()

    server_cfg_path = Path(args.server_config).resolve()
    base_dir = server_cfg_path.parent
    server_cfg = _load_server_config(server_cfg_path)
    train_paths = _resolve_training_paths(server_cfg, base_dir)

    _setup_logging(Path(args.log_file))
    log = logging.getLogger("capi.train_runner")
    runner_started_at = datetime.now()
    log.info(f"啟動訓練 runner: job_id={args.job_id}, pid={os.getpid()}")

    # HF cache 在 import torch 前設好（_setup_offline_env 之後也會設，這裡提前更穩）
    os.environ.setdefault("HF_HOME", str(train_paths["backbone_cache_dir"]))
    _apply_vram_limit(train_paths["gpu_memory_fraction"], log)

    db_cfg = server_cfg.get("database", {}) or {}
    db_path = db_cfg.get("path", "/data/capi_ai/capi_results.db")

    from capi_database import CAPIDatabase
    db = CAPIDatabase(db_path)

    job = db.get_training_job(args.job_id)
    if not job:
        log.error(f"找不到 job_id={args.job_id}")
        return 2

    cancel_event = _FileCancelEvent(
        Path(args.cancel_flag),
        started_at=runner_started_at,
        logger=log,
    )

    from capi_train_new import (
        TrainingConfig, apply_user_training_params, run_training_pipeline,
    )

    cfg = TrainingConfig(
        machine_id=job["machine_id"],
        panel_paths=[Path(p) for p in job["panel_paths"]],
        over_review_root=train_paths["over_review_root"],
        backbone_cache_dir=train_paths["backbone_cache_dir"],
        output_root=train_paths["output_root"],
        required_backbones=train_paths["required_backbones"],
    )
    apply_user_training_params(cfg, job.get("training_params"), log_fn=log.info)

    try:
        bundle_dir = run_training_pipeline(
            job_id=args.job_id,
            cfg=cfg,
            db=db,
            gpu_lock=None,
            log=log.info,
            cancel_event=cancel_event,
        )

        sizes = sum(p.stat().st_size for p in bundle_dir.glob("*.pt"))
        manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        inner_count = sum(
            v["train"]
            for k, v in manifest["tiles_per_unit"].items()
            if k.endswith("-inner")
        )
        edge_count = sum(
            v["train"]
            for k, v in manifest["tiles_per_unit"].items()
            if k.endswith("-edge")
        )
        ng_count = sum(v["ng"] for v in manifest["tiles_per_unit"].values())

        db.register_model_bundle({
            "machine_id": job["machine_id"],
            "bundle_path": str(bundle_dir),
            "trained_at": manifest["trained_at"],
            "panel_count": manifest["panel_count"],
            "inner_tile_count": inner_count,
            "edge_tile_count": edge_count,
            "ng_tile_count": ng_count,
            "bundle_size_bytes": sizes,
            "job_id": args.job_id,
        })
        db.update_training_job_state(
            args.job_id, "completed", output_bundle=str(bundle_dir)
        )
        log.info(f"✓ 訓練完成，bundle={bundle_dir}")
        return 0

    except Exception as e:
        traceback.print_exc()
        msg = "取消" if cancel_event.is_set() else str(e)
        db.update_training_job_state(args.job_id, "failed", error_message=msg)
        log.error(f"✗ 訓練失敗: {msg}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
