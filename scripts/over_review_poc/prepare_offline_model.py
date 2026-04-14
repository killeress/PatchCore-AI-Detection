"""Helper：在有外網的機器執行，預先下載 DINOv2 並印出 cache 路徑 / state_dict 檔。

用途：產線 Linux 無外網 → 在此機器執行 → 把印出的檔案搬到產線相同路徑
(或指定 --export-state-dict <path>，產出可搬運的 .pth 檔)。

Usage:
    python -m scripts.over_review_poc.prepare_offline_model
    python -m scripts.over_review_poc.prepare_offline_model --export-state-dict /tmp/dinov2_vitb14.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.over_review_poc.features import DINOV2_MODEL, DINOV2_REPO


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare DINOv2 for offline deployment")
    parser.add_argument("--export-state-dict", type=Path, default=None,
                        help="選填：把 state_dict 另存為 .pth（方便搬運）")
    args = parser.parse_args(argv)

    print(f"Loading {DINOV2_MODEL} via torch.hub (this will download on first run)...")
    model = torch.hub.load(DINOV2_REPO, DINOV2_MODEL, source="github")
    model.eval()

    hub_dir = torch.hub.get_dir()
    print("=" * 70)
    print(f"DINOv2 cache root: {hub_dir}")
    print("Contents to copy to offline machine (same path):")
    for entry in sorted(Path(hub_dir).rglob("*")):
        if entry.is_file():
            print(f"  {entry}")
    print("=" * 70)

    if args.export_state_dict is not None:
        args.export_state_dict.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.export_state_dict)
        print(f"State dict exported to: {args.export_state_dict}")
        print(f"On offline machine: pass --checkpoint {args.export_state_dict} to run_poc.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
