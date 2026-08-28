"""Build production deployment ZIP for CAPI AI release updates.

Output:
  deployment/patchcore_ai_release_<version>.zip
  deployment/patchcore_ai_patch_<version>.zip when --patch-only is used

ZIP layout preserves project-root relative paths so operator just unzips
on top of production install. Includes:
  - application files listed in CODE_FILES
  - VERSION / CHANGELOG.md / release_manifest.json / checksums.txt
  - deployment/torch_hub_cache/ (offline backbone cache, ~264 MB)
  - server_config_patch.yaml.example (showing fields to merge)
  - README.txt with deployment steps
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = PROJECT_ROOT / "VERSION"
CHANGELOG_FILE = PROJECT_ROOT / "CHANGELOG.md"
GIT_SAFE_DIR = PROJECT_ROOT.as_posix()

CODE_FILES = [
    "capi_version.py",
    "capi_auto_model_switch.py",
    "capi_config.py",
    "capi_database.py",
    "capi_dataset_export.py",
    "capi_edge_cv.py",
    "capi_heatmap.py",
    "capi_heatmap_diagnostics.py",
    "capi_image_orientation.py",
    "capi_image_naming.py",
    "capi_image_preprocess_lab.py",
    "capi_grid_canonicalization.py",
    "capi_inference.py",
    "capi_mark_calibration.py",
    "capi_mark_detector.py",
    "capi_mark_shadow.py",
    "mark_shadow/paddle_shadow_worker.py",
    "mark_shadow/install_worker_hotfix.sh",
    "mark_shadow/README_WORKER_HOTFIX.txt",
    "capi_mes_report.py",
    "capi_model_registry.py",
    "capi_model_validation.py",
    "capi_preprocess.py",
    "capi_patchcore_feature_cleaning.py",
    "capi_scratch_batch.py",
    "capi_scratch_export.py",
    "capi_server.py",
    "capi_station_adapter.py",
    "capi_train_new.py",
    "capi_train_runner.py",
    "capi_update_agent.py",
    "capi_web.py",
    "capi_white_frame.py",
    "configs/mes_defect_codes.json",
    "requirements.txt",
    "server_config_mes_report.yaml.example",
    "scratch_classifier.py",
    "scratch_filter.py",
    "start_server.py",
    "templates/base.html",
    "templates/dashboard.html",
    "templates/debug_inference.html",
    "templates/_white_frame_result.html",
    "templates/white_frame.html",
    "templates/record_detail.html",
    "templates/record_detail_v3.html",
    "templates/retrain_pool.html",
    "templates/ric_report.html",
    "templates/training.html",
    "templates/release_notes.html",
    "templates/models.html",
    "templates/settings.html",
    "templates/within_spec_detail.html",
    "templates/train_new/_modal.html",
    "templates/train_new/step1_scope.html",
    "templates/train_new/step1_select.html",
    "templates/train_new/step2_progress.html",
    "templates/train_new/step3_review.html",
    "templates/train_new/step4_progress.html",
    "templates/train_new/step5_done.html",
    "central_dashboard/README.md",
    "central_dashboard/app.js",
    "central_dashboard/banner.png",
    "central_dashboard/config.js",
    "central_dashboard/index.html",
    "central_dashboard/settings.html",
    "central_dashboard/styles.css",
    "static/favicon.svg",
    "scripts/over_review_poc/train_final_model.py",
    "tools/build_bga_tiles.py",
    "tools/diagnose_mes_oracle.py",
    "tools/train_bga_all.py",
    "start_server.sh",
    "install_patch.sh",
    "rollback_patch.sh",
    "promote_update.sh",
    "setup_auto_update_client.sh",
]

# MES Oracle 密碼是設備本機的 ignored secret。預設不打包；只有呼叫端明確
# 指定 --include-local-credentials 時才納入，並要求使用 --allow-dirty。
CODEONLY_LOCAL_FILES = [
    "capi_mes_credentials.py",
]

PATCH_UTILITY_FILES = [
    "start_server.sh",
    "install_patch.sh",
    "rollback_patch.sh",
    "promote_update.sh",
    "setup_auto_update_client.sh",
]

PATCH_DEPLOY_ROOT_FILES = {
    "VERSION",
    "CHANGELOG.md",
    "start_server.sh",
    "install_patch.sh",
    "rollback_patch.sh",
    "promote_update.sh",
    "setup_auto_update_client.sh",
    "requirements.txt",
    "server_config_mes_report.yaml.example",
    "mark_shadow/paddle_shadow_worker.py",
    "mark_shadow/install_worker_hotfix.sh",
    "mark_shadow/README_WORKER_HOTFIX.txt",
}

PATCH_DEPLOY_PREFIXES = (
    "scripts/over_review_poc/",
    "templates/",
    "static/",
    "tools/",
)

GENERATED_METADATA_FILES = {
    "VERSION",
    "CHANGELOG.md",
}

BACKBONE_CACHE_DIR = "deployment/torch_hub_cache"
CODEONLY_EXCLUDED_PREFIXES = (
    "templates/imgs/",
    "static/",
)

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

# Report 數據比對：每台設備只選擇一個廠別 Oracle TNS
# 更新包預設不含 capi_mes_credentials.py；除非建包時明確要求，否則請保留設備既有檔。
mes_report:
  facility: MOD2
  oracle:
    user: MISSELECT
    tns:
      MOD1:
        host: 10.172.3.55
        port: 1521
        service_name: pncmr
      MOD2:
        host: 10.174.1.79
        port: 1521
        service_name: pnemr
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
   - 加 mes_report.oracle 區段
   - 若啟用 MES Report，請保留設備既有的 capi_mes_credentials.py；除非建包時明確要求，更新包不含明文密碼
   - 安裝 Oracle thin driver：python3 -m pip install "oracledb>=2.0.0"

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
  2. Step 1: 輸入機種 ID，從 DB 列出 AOI 判 OK panel，勾選 3 片
  3. Step 2: 系統自動前處理 + 切 tile（3 panel × 5 lighting × ~150 tile，含 edge 外推取樣）
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
- central_dashboard/（CAPI AI 中控看板）
- 8 個 capi_*.py 模組（含修改與新增）
- 7 個 templates/train_new/*.html + templates/models.html
- deployment/torch_hub_cache/（HuggingFace timm wide_resnet50_2 cache）
"""


CODEONLY_README_NOTE = """\

【本 ZIP 為 code-only 增量包】
- 此段安裝方式取代上方的一般解壓步驟；請先從 ZIP 換入新版 installer，再執行完整安裝：
    cd /root/Code/CAPI_AD
    unzip -o /path/to/patchcore_ai_release_<version>_codeonly.zip install_patch.sh
    chmod +x install_patch.sh
    sudo ./install_patch.sh /path/to/patchcore_ai_release_<version>_codeonly.zip
- 只手動解壓並重啟主程式，不會更新 /aidata/capi_ai/mark_shadow/current 內的正式 worker
- 不含 deployment/torch_hub_cache/（之前的部署包已含，production 機應已落地）
- 不含 templates/imgs/ 與 static/（沿用 production 機已有的靜態資源）
- 解壓覆蓋既有檔即可，不會動到 backbone cache 目錄
- 內含 MARK PaddleOCR worker 更新；使用 install_patch.sh 時，若現場已有 MARK worker，會自動備份、套用、重啟並檢查健康狀態
- 不含 PaddleOCR runtime 與模型，沿用 /aidata/capi_ai/mark_shadow/current 既有安裝
"""

CODEONLY_CREDENTIALS_EXCLUDED_README_NOTE = """\

【本機 credentials】
- 本 ZIP 不含 capi_mes_credentials.py，請保留 production 機既有檔案
"""

CODEONLY_CREDENTIALS_README_NOTE = """\

【敏感檔案警告】
- 本 ZIP 依明確要求納入 capi_mes_credentials.py；此檔含有 MES 密碼
- 請限制 ZIP 與解壓後檔案的存取權限及傳輸範圍
"""


PATCH_README_TEXT = """CAPI AI Patch 更新包
================================================================

用途
----------------------------------------------------------------

本 ZIP 是 patch-only 更新包，只包含本次 Git 變更中可部署到現場的檔案，
以及 VERSION、CHANGELOG.md、release_manifest.json、checksums.txt。

第一次使用 install_patch.sh 的設備
----------------------------------------------------------------

若設備上尚未有 install_patch.sh，先解出更新腳本：

  cd /root/Code/CAPI_AD
  unzip -o /path/to/patchcore_ai_patch_<version>.zip start_server.sh install_patch.sh rollback_patch.sh
  chmod +x start_server.sh install_patch.sh rollback_patch.sh
  ./install_patch.sh /path/to/patchcore_ai_patch_<version>.zip

之後再次更新
----------------------------------------------------------------

  cd /root/Code/CAPI_AD
  ./install_patch.sh /path/to/patchcore_ai_patch_<version>.zip

更新腳本會執行：
  1. 檢查 checksums.txt
  2. 備份即將被覆蓋的檔案到 .patch_backups/
  3. 解壓 patch ZIP
  4. 執行 ./start_server.sh restart --no-tail
  5. 使用 /api/version 做健康檢查

回滾
----------------------------------------------------------------

install_patch.sh 完成後會顯示 rollback 指令，例如：

  ./rollback_patch.sh ".patch_backups/2026.06.29.1_20260629_150000"

注意事項
----------------------------------------------------------------

- 此包只更新程式檔與版本資訊，不應包含 DB、模型權重、heatmap、現場設定檔。
- 若更新包內包含 start_server.sh，會一併更新現場啟動腳本。
- Report 數據比對需安裝 `oracledb`，並合併 server_config_mes_report.yaml.example。
"""


def _default_version() -> str:
    if VERSION_FILE.exists():
        version = VERSION_FILE.read_text(encoding="utf-8").strip()
        if version:
            return version
    return date.today().strftime("%Y.%m.%d.1")


def _bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_commit() -> str:
    return _git_revision("--short")


def _git_commit_full() -> str:
    return _git_revision()


def _git_revision(*args: str) -> str:
    proc = _run_git(["rev-parse", *args, "HEAD"])
    revision = proc.stdout.strip()
    if not revision:
        raise RuntimeError("git rev-parse HEAD returned an empty revision")
    return revision


def _run_git(args: list[str]) -> subprocess.CompletedProcess:
    command = ["git", "-c", f"safe.directory={GIT_SAFE_DIR}", *args]
    try:
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or getattr(exc, "stdout", "") or str(exc)
        raise RuntimeError(f"git {' '.join(args)} failed: {detail.strip()}") from exc


def _git_file_list(args: list[str]) -> list[str]:
    proc = _run_git(args)
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def _git_changed_files() -> list[str]:
    changed = set(_git_file_list(["diff", "--name-only", "HEAD"]))
    changed.update(_git_file_list(["ls-files", "--others", "--exclude-standard"]))
    return sorted(changed)


def _git_managed_asset_files() -> list[str]:
    files = _git_file_list([
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "templates",
        "static",
    ])
    return sorted(rel for rel in files if (PROJECT_ROOT / rel).is_file())


def _is_patch_deploy_file(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    if rel in PATCH_DEPLOY_ROOT_FILES:
        return True
    if rel.endswith(".py") and "/" not in rel:
        return True
    return rel.startswith(PATCH_DEPLOY_PREFIXES)


def _is_codeonly_excluded_file(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    return rel.startswith(CODEONLY_EXCLUDED_PREFIXES)


def _release_files(
    *,
    codeonly: bool = False,
    include_local_credentials: bool = False,
) -> list[str]:
    source_files = [*CODE_FILES]
    if codeonly and include_local_credentials:
        source_files.extend(CODEONLY_LOCAL_FILES)
    files = list(dict.fromkeys([*source_files, *_git_managed_asset_files()]))
    if codeonly:
        files = [rel for rel in files if not _is_codeonly_excluded_file(rel)]
    return files


def _codeonly_excluded_changes(changed_files: list[str]) -> list[str]:
    return sorted({
        rel.replace("\\", "/")
        for rel in changed_files
        if _is_codeonly_excluded_file(rel)
    })


def _validate_required_code_files(
    *,
    codeonly: bool = False,
    include_local_credentials: bool = False,
) -> None:
    required_files = CODE_FILES
    if codeonly:
        required_files = [
            *[rel for rel in required_files if not _is_codeonly_excluded_file(rel)],
        ]
        if include_local_credentials:
            required_files.extend(CODEONLY_LOCAL_FILES)
    missing = [rel for rel in required_files if not (PROJECT_ROOT / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"required CODE_FILES missing: {', '.join(missing)}")


def _release_dirty_files(
    changed_files: list[str],
    package_files: list[str],
    *,
    patch_only: bool,
    include_backbone: bool,
    codeonly: bool,
) -> list[str]:
    package_sources = {rel.replace("\\", "/") for rel in package_files}
    package_sources.update(GENERATED_METADATA_FILES)
    package_sources.add("scripts/build_deploy_zip.py")
    dirty_files = []

    for rel in changed_files:
        rel = rel.replace("\\", "/")
        relevant = rel in package_sources
        if patch_only:
            relevant = relevant or _is_patch_deploy_file(rel)
        else:
            if _is_codeonly_excluded_file(rel):
                relevant = relevant or not codeonly
            else:
                relevant = relevant or rel.startswith("templates/")
            if include_backbone:
                relevant = relevant or rel.startswith(f"{BACKBONE_CACHE_DIR}/")
        if relevant:
            dirty_files.append(rel)

    return sorted(set(dirty_files))


def _patch_files(changed_files: list[str] | None = None) -> tuple[list[str], list[str]]:
    selected = []
    skipped = []
    seen = set()
    if changed_files is None:
        changed_files = _git_changed_files()

    for rel in [*changed_files, *PATCH_UTILITY_FILES]:
        rel = rel.replace("\\", "/")
        if rel in seen:
            continue
        seen.add(rel)
        if rel in GENERATED_METADATA_FILES:
            continue
        src = PROJECT_ROOT / rel
        if not src.is_file():
            skipped.append(rel)
            continue
        if _is_patch_deploy_file(rel):
            selected.append(rel)
        else:
            skipped.append(rel)

    return sorted(selected), sorted(skipped)


def _add_file(zf: zipfile.ZipFile, src: Path, arcname: str, entries: list[dict]) -> None:
    zf.write(src, arcname=arcname)
    h = hashlib.sha256()
    size_bytes = 0
    info = zf.getinfo(arcname)
    original_name = info.orig_filename
    info.orig_filename = info.filename  # Windows ZipInfo keeps backslashes here until archive close.
    try:
        with zf.open(info) as packed:
            for chunk in iter(lambda: packed.read(1024 * 1024), b""):
                h.update(chunk)
                size_bytes += len(chunk)
    finally:
        info.orig_filename = original_name
    entries.append({
        "path": arcname,
        "size_bytes": size_bytes,
        "sha256": h.hexdigest(),
    })


def _add_text(zf: zipfile.ZipFile, arcname: str, text: str, entries=None) -> None:
    data = text.encode("utf-8")
    zf.writestr(arcname, data)
    if entries is not None:
        entries.append({
            "path": arcname,
            "size_bytes": len(data),
            "sha256": _bytes_sha256(data),
        })


def _content_tree_sha256(entries: list[dict]) -> str:
    canonical = json.dumps(
        sorted(entries, key=lambda item: item["path"]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _bytes_sha256(canonical)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build CAPI AI release deploy ZIP")
    parser.add_argument(
        "--version",
        default=None,
        help="Release version. Defaults to VERSION file or YYYY.MM.DD.1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "deployment",
        help="Directory for the generated ZIP.",
    )
    parser.add_argument(
        "--no-backbone", action="store_true",
        help="Skip backbone cache (use when production already has it from previous deploy)",
    )
    parser.add_argument(
        "--patch-only", action="store_true",
        help="Build a small patch ZIP from deployable Git changes only",
    )
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="Allow full/code-only build from deploy-relevant uncommitted files",
    )
    parser.add_argument(
        "--include-local-credentials", action="store_true",
        help=(
            "Include the ignored local capi_mes_credentials.py in a code-only ZIP. "
            "This embeds plaintext secrets and requires --allow-dirty."
        ),
    )
    args = parser.parse_args(argv)

    version = (args.version or _default_version()).strip()
    if not version:
        raise ValueError("release version cannot be empty")

    if args.patch_only:
        args.no_backbone = True
    if args.include_local_credentials and (args.patch_only or not args.no_backbone):
        parser.error("--include-local-credentials is only valid with --no-backbone")
    if args.include_local_credentials and not args.allow_dirty:
        parser.error("--include-local-credentials requires --allow-dirty")

    if not args.patch_only:
        _validate_required_code_files(
            codeonly=args.no_backbone,
            include_local_credentials=args.include_local_credentials,
        )

    changed_files = _git_changed_files()
    if args.include_local_credentials:
        changed_files = sorted(set(changed_files) | {
            rel for rel in CODEONLY_LOCAL_FILES
            if (PROJECT_ROOT / rel).is_file()
        })
    if args.patch_only:
        package_files, skipped_files = _patch_files(changed_files)
        if not package_files:
            raise RuntimeError("no deployable changed files found for --patch-only")
    else:
        package_files, skipped_files = _release_files(
            codeonly=args.no_backbone,
            include_local_credentials=args.include_local_credentials,
        ), []

    excluded_asset_changes = (
        _codeonly_excluded_changes(changed_files)
        if args.no_backbone and not args.patch_only
        else []
    )

    git_dirty_files = _release_dirty_files(
        changed_files,
        package_files,
        patch_only=args.patch_only,
        include_backbone=not args.no_backbone,
        codeonly=args.no_backbone and not args.patch_only,
    )
    git_worktree_dirty = bool(changed_files)
    git_dirty = bool(git_dirty_files)
    if git_dirty and not args.patch_only and not args.allow_dirty:
        raise RuntimeError(
            "deploy-relevant dirty files: "
            f"{', '.join(git_dirty_files)}; commit/stash them or rerun with --allow-dirty"
        )

    built_at = datetime.now().astimezone().isoformat(timespec="seconds")
    git_commit = _git_commit()
    git_commit_full = _git_commit_full()
    source_mode = "working_tree" if args.patch_only or git_dirty else "git_commit"

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(exist_ok=True)

    suffix = "_codeonly" if args.no_backbone and not args.patch_only else ""
    package_name = "patchcore_ai_patch" if args.patch_only else "patchcore_ai_release"
    zip_path = output_dir / f"{package_name}_{version}{suffix}.zip"

    print(f"Building deploy ZIP: {zip_path}")
    print(f"Project root: {PROJECT_ROOT}")
    if args.patch_only:
        print("Mode: patch-only (--patch-only)")
    if args.no_backbone and not args.patch_only:
        print("Mode: code-only (--no-backbone)")
        print("Excluded static directories: templates/imgs/, static/")
        if args.include_local_credentials:
            print("WARNING: including plaintext local MES credentials by explicit request")
        if excluded_asset_changes:
            print("WARNING: excluded static assets changed and will not be packaged:")
            for rel in excluded_asset_changes:
                print(f"  ! {rel}")
            print("Deploy these assets separately if the code update needs them.")

    if zip_path.exists():
        zip_path.unlink()

    code_size = 0
    backbone_size = 0
    backbone_files = 0
    entries = []

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. Application code
        print(f"\n[1/5] Adding {len(package_files)} code files...")
        for rel in package_files:
            src = PROJECT_ROOT / rel
            if not src.is_file():
                raise FileNotFoundError(f"package file disappeared during build: {rel}")
            _add_file(zf, src, rel.replace("\\", "/"), entries)
            code_size += entries[-1]["size_bytes"]
            print(f"  + {rel}")
        if args.patch_only and skipped_files:
            print("\nSkipped non-deployable changed files:")
            for rel in skipped_files:
                print(f"  - {rel}")

        # 2. Backbone cache (skip xet logs — useless on production)
        if args.no_backbone:
            print(f"\n[2/5] Skipping backbone cache (--no-backbone)")
        else:
            backbone_dir = PROJECT_ROOT / BACKBONE_CACHE_DIR
            if backbone_dir.exists():
                print(f"\n[2/5] Adding backbone cache ({BACKBONE_CACHE_DIR})...")
                for src in backbone_dir.rglob("*"):
                    if not src.is_file():
                        continue
                    rel = src.relative_to(PROJECT_ROOT)
                    rel_str = str(rel).replace("\\", "/")
                    if "xet/logs" in rel_str or rel_str.endswith(".log"):
                        continue
                    _add_file(zf, src, rel.as_posix(), entries)
                    backbone_size += entries[-1]["size_bytes"]
                    backbone_files += 1
                print(f"  + {backbone_files} files in {BACKBONE_CACHE_DIR}")
            else:
                print(f"\n⚠ [2/5] Backbone cache missing at {backbone_dir}")

        # 3. server_config patch example
        if args.patch_only:
            print(f"\n[3/5] Skipping server_config_patch.yaml.example (--patch-only)")
        else:
            print(f"\n[3/5] Adding server_config_patch.yaml.example...")
            _add_text(zf, "server_config_patch.yaml.example", SERVER_CONFIG_PATCH, entries)

        # 4. Release metadata
        print(f"\n[4/5] Adding release metadata...")
        changelog = (
            CHANGELOG_FILE.read_text(encoding="utf-8")
            if CHANGELOG_FILE.exists()
            else f"# Changelog\n\n## {version}\n\n- Release notes not provided.\n"
        )
        _add_text(zf, "VERSION", f"{version}\n", entries)
        _add_text(zf, "CHANGELOG.md", changelog, entries)

        # 5. README + generated manifest/checksums
        print(f"\n[5/5] Adding README.txt, release_manifest.json, checksums.txt...")
        base_readme = PATCH_README_TEXT if args.patch_only else README_TEXT
        readme = (
            f"Release version: {version}\n"
            f"Git commit: {git_commit or 'unknown'}\n"
            f"Built at: {built_at}\n\n"
            + base_readme
        )
        if args.no_backbone and not args.patch_only:
            readme = readme + CODEONLY_README_NOTE
            if args.include_local_credentials:
                readme = readme + CODEONLY_CREDENTIALS_README_NOTE
            else:
                readme = readme + CODEONLY_CREDENTIALS_EXCLUDED_README_NOTE
        _add_text(zf, "README.txt", readme, entries)

        payload_entries = sorted(entries, key=lambda item: item["path"])
        manifest = {
            "version": version,
            "git_commit": git_commit,
            "git_commit_full": git_commit_full,
            "git_worktree_dirty": git_worktree_dirty,
            "git_dirty": git_dirty,
            "git_dirty_files": git_dirty_files,
            "source_mode": source_mode,
            "content_tree_sha256": _content_tree_sha256(payload_entries),
            "built_at": built_at,
            "artifact": zip_path.name,
            "requires_restart": True,
            "package_type": "patch" if args.patch_only else ("codeonly" if args.no_backbone else "full"),
            "contains_local_credentials": bool(args.include_local_credentials),
            "files": payload_entries,
        }
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        _add_text(zf, "release_manifest.json", manifest_text, entries)

        checksums = "".join(
            f"{entry['sha256']}  {entry['path']}\n"
            for entry in sorted(entries, key=lambda item: item["path"])
        )
        _add_text(zf, "checksums.txt", checksums)

    final_size = zip_path.stat().st_size
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Output:        {zip_path}")
    print(f"  ZIP size:      {final_size / 1e6:.1f} MB")
    print(f"  Code size:     {code_size / 1e6:.2f} MB ({len(package_files)} files)")
    if not args.no_backbone:
        print(f"  Backbone size: {backbone_size / 1e6:.1f} MB ({backbone_files} files)")
    print(f"  Total files:   {len(entries) + 1}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
