"""Build production deployment ZIP for new model training wizard feature.

Output: deployment/capi_train_wizard_deploy_<date>.zip

ZIP layout preserves project-root relative paths so operator just unzips
on top of production install. Includes:
  - 8 modified/new .py files
  - 6 new templates
  - 2 deprecated tools (with deprecation header)
  - deployment/torch_hub_cache/ (offline backbone cache, ~264 MB)
  - server_config_patch.yaml.example (showing fields to merge)
  - README.txt with deployment steps
"""
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CODE_FILES = [
    "capi_config.py",
    "capi_database.py",
    "capi_inference.py",
    "capi_model_registry.py",
    "capi_preprocess.py",
    "capi_server.py",
    "capi_train_new.py",
    "capi_train_runner.py",
    "capi_web.py",
    "templates/base.html",
    "templates/dashboard.html",
    "templates/training.html",
    "templates/models.html",
    "templates/settings.html",
    "templates/train_new/_modal.html",
    "templates/train_new/step1_select.html",
    "templates/train_new/step2_progress.html",
    "templates/train_new/step3_review.html",
    "templates/train_new/step4_progress.html",
    "templates/train_new/step5_done.html",
    "static/favicon.svg",
    "tools/build_bga_tiles.py",
    "tools/train_bga_all.py",
]

BACKBONE_CACHE_DIR = "deployment/torch_hub_cache"

SERVER_CONFIG_PATCH = """# === 新機種 PatchCore 訓練 wizard 需要在 server_config.yaml 加入以下欄位 ===
# 將此檔的內容合併進 production 既有的 server_config.yaml（不要整個覆蓋）

# 推論端 GPU VRAM 上限（讓訓練 subprocess 可同時跑而不互搶）
# 16GB GPU 實測：5 個 model load 完即 ~4.2GB；推論 working set 再 ~1-2GB
# 0 = 不限制（舊行為）
inference:
  gpu_memory_fraction: 0.40

# 多機種 model 配置列表（之後啟用新 bundle 時，從模型庫頁面自動新增）
model_configs:
  - configs/capi_3f.yaml
fallback_model_config: configs/capi_3f.yaml

# 訓練 wizard 設定
training:
  backbone_cache_dir: deployment/torch_hub_cache
  # Backbone 由 timm/HuggingFace 下載
  # Pre-stage 在開發機:
  #   HF_HOME=deployment/torch_hub_cache python -c "import timm; timm.create_model('wide_resnet50_2', pretrained=True)"
  over_review_root: /aidata/capi_ai/datasets/over_review
  output_root: model
  # 訓練 subprocess GPU VRAM 上限（與 inference.gpu_memory_fraction 配對）
  # 0.40 + 0.50 = 0.90，剩 ~10% 給桌面/buffer
  gpu_memory_fraction: 0.50
"""

README_TEXT = """新機種 PatchCore 訓練 Wizard — Production 部署說明
================================================================

部署步驟
----------------------------------------------------------------

1. 先備份 production 整個 capi_ai/ 目錄（防止部署失敗回滾用）：
     tar -czf capi_ai_backup_$(date +%Y%m%d).tar.gz /capi_ai/

2. 解壓本 ZIP 到 production /capi_ai/，保留路徑結構覆蓋既有檔：
     cd /capi_ai
     unzip /path/to/capi_train_wizard_deploy_<date>.zip

3. 把 server_config_patch.yaml.example 內容**合併**進既有 server_config.yaml：
   （不要整個覆蓋既有檔，只加缺少的 keys）
   - 加 model_configs 列表
   - 加 fallback_model_config
   - 加 training 區段

4. 確認 deployment/torch_hub_cache/ 目錄完整（應 ~264 MB）：
     du -sh /capi_ai/deployment/torch_hub_cache/
     # 期望: 約 264 MB

5. 重啟服務：
     systemctl restart capi_server
     # 或舊式: ./start_server.sh restart

6. 驗證啟動 log（grep "[SERVER] Loaded" "[MultiConfig]"）：
     journalctl -u capi_server | tail -20

7. 開瀏覽器確認 wizard 入口：
     http://<production-ip>:8080/training
     # 應看到 2 張卡：刮痕分類器 + 新機種 PatchCore


新功能使用說明
----------------------------------------------------------------

訓練新機種 PatchCore 模型：
  1. http://<server>:8080/training → 點「新機種 PatchCore」卡的「開始訓練」
  2. Step 1: 輸入機種 ID，從 DB 列出 AOI 判 OK panel，勾選 5 片
  3. Step 2: 系統自動前處理 + 切 tile（5 panel × 5 lighting × ~150 tile）
  4. Step 3: 審核 tile pool（5 個 lighting tab × inner/edge × OK/NG 4 group）
  5. Step 4: 開始訓練 10 個 PatchCore 模型（GPU lock 序列跑，~80 分鐘）
  6. Step 5: 完成頁顯示子模型摘要

部署訓練好的 bundle：
  1. http://<server>:8080/models → 找到剛訓練的 bundle
  2. 點「啟用」→ 提示重啟 server 才會生效
  3. 點「匯出 ZIP」→ 下載 ZIP 給其他 production 機部署
  4. 重啟 server: systemctl restart capi_server


回滾
----------------------------------------------------------------

如部署後 inference 行為異常，回滾步驟：
  1. systemctl stop capi_server
  2. tar -xzf capi_ai_backup_<date>.tar.gz -C /
  3. systemctl start capi_server


注意事項
----------------------------------------------------------------

- 既有 CAPI 3F（5-model）機種**完全不受影響**，仍走 legacy 路徑（capi_3f.yaml）
- 新機種訓練資料來源：DB inference_records WHERE machine_judgment = 'OK'
- NG 樣本來源：/aidata/capi_ai/datasets/over_review/{*}/true_ng/
- backbone 完全離線，不會嘗試從外網下載

支援檔案
----------------------------------------------------------------

- README.txt（本檔）
- server_config_patch.yaml.example
- 8 個 capi_*.py 模組（含修改與新增）
- 6 個 templates/train_new/*.html + templates/models.html
- deployment/torch_hub_cache/（HuggingFace timm wide_resnet50_2 cache）
"""


CODEONLY_README_NOTE = """\

【本 ZIP 為 code-only 增量包】
- 不含 deployment/torch_hub_cache/（之前的部署包已含，production 機應已落地）
- 解壓覆蓋既有檔即可，不會動到 backbone cache 目錄
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build CAPI training wizard deploy ZIP")
    parser.add_argument(
        "--no-backbone", action="store_true",
        help="Skip backbone cache (use when production already has it from previous deploy)",
    )
    args = parser.parse_args(argv)

    output_dir = PROJECT_ROOT / "deployment"
    output_dir.mkdir(exist_ok=True)

    today = date.today().isoformat()
    suffix = "_codeonly" if args.no_backbone else ""
    zip_path = output_dir / f"capi_train_wizard_deploy_{today}{suffix}.zip"

    print(f"Building deploy ZIP: {zip_path}")
    print(f"Project root: {PROJECT_ROOT}")
    if args.no_backbone:
        print("Mode: code-only (--no-backbone)")

    if zip_path.exists():
        zip_path.unlink()

    code_size = 0
    backbone_size = 0
    file_count = 0
    backbone_files = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. Application code
        print(f"\n[1/4] Adding {len(CODE_FILES)} code files...")
        for rel in CODE_FILES:
            src = PROJECT_ROOT / rel
            if not src.exists():
                print(f"  ⚠ MISSING: {rel}")
                continue
            zf.write(src, arcname=rel)
            code_size += src.stat().st_size
            file_count += 1
            print(f"  + {rel}")

        # 2. Backbone cache (skip xet logs — useless on production)
        if args.no_backbone:
            print(f"\n[2/4] Skipping backbone cache (--no-backbone)")
        else:
            backbone_dir = PROJECT_ROOT / BACKBONE_CACHE_DIR
            if backbone_dir.exists():
                print(f"\n[2/4] Adding backbone cache ({BACKBONE_CACHE_DIR})...")
                for src in backbone_dir.rglob("*"):
                    if not src.is_file():
                        continue
                    rel = src.relative_to(PROJECT_ROOT)
                    rel_str = str(rel).replace("\\", "/")
                    if "xet/logs" in rel_str or rel_str.endswith(".log"):
                        continue
                    zf.write(src, arcname=str(rel))
                    backbone_size += src.stat().st_size
                    file_count += 1
                    backbone_files += 1
                print(f"  + {backbone_files} files in {BACKBONE_CACHE_DIR}")
            else:
                print(f"\n⚠ [2/4] Backbone cache missing at {backbone_dir}")

        # 3. server_config patch example
        print(f"\n[3/4] Adding server_config_patch.yaml.example...")
        zf.writestr("server_config_patch.yaml.example", SERVER_CONFIG_PATCH)
        file_count += 1

        # 4. README
        print(f"\n[4/4] Adding README.txt...")
        readme = README_TEXT
        if args.no_backbone:
            readme = readme + CODEONLY_README_NOTE
        zf.writestr("README.txt", readme)
        file_count += 1

    final_size = zip_path.stat().st_size
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Output:        {zip_path}")
    print(f"  ZIP size:      {final_size / 1e6:.1f} MB")
    print(f"  Code size:     {code_size / 1e6:.2f} MB ({len(CODE_FILES)} files)")
    if not args.no_backbone:
        print(f"  Backbone size: {backbone_size / 1e6:.1f} MB ({backbone_files} files)")
    print(f"  Total files:   {file_count}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
