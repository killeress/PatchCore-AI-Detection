# Code-only 發行包打包 SOP

本文件供維運人員從內部 GitLab 的預設分支產生 code-only 更新 ZIP。整個流程只需要 Git、Python 與 PowerShell，不需要 Codex 或其他 AI 工具。

## 適用範圍

Code-only ZIP 包含後端 Python、HTML template、啟動／更新腳本、版本資訊、manifest 與 checksum。它不包含：

- `deployment/torch_hub_cache/` backbone cache
- `static/` 與 `templates/imgs/` 靜態資源
- 模型權重、資料庫、heatmap、正式機設定與現場資料
- `capi_mes_credentials.py` 等本機密碼檔

若本次版本修改了上述排除內容，不可只交付 code-only ZIP，請交由開發負責人確認完整部署方式。

## 環境需求

- 可連線內部 GitLab
- Git
- Python 3.10 以上
- Windows PowerShell 5.1 或 PowerShell 7

打包器使用 Python 標準函式庫，不需要先執行 `pip install -r requirements.txt`，也不需要 Codex。

## 第一次準備

```powershell
git clone http://fsrserver.cminl.oa/ai_team/ngb_capi_ad.git
cd ngb_capi_ad
git checkout master
```

## 每次打包

版本格式固定為 `YYYY.MM.DD.N`；同一天第幾版由最後一碼依序增加，例如 `2026.08.31.1`、`2026.08.31.2`。

先同步內部 GitLab 的預設分支，並確認 working tree 沒有任何修改：

```powershell
git checkout master
git pull --ff-only origin master
git status --short
```

`git status --short` 應完全沒有輸出。接著執行一鍵打包：

```powershell
.\build_codeonly.ps1 -Version 2026.08.31.1
```

若 Windows 執行原則阻擋 `.ps1`，只針對本次命令使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_codeonly.ps1 -Version 2026.08.31.1
```

成功後會產生：

```text
deployment/patchcore_ai_release_2026.08.31.1_codeonly.zip
```

腳本會拒絕：

- 不符合 `YYYY.MM.DD.N` 的版本號
- 不乾淨的 Git working tree
- Python 版本低於 3.10
- 覆寫已存在的同版本 ZIP

## 驗收 ZIP

以下命令會顯示 ZIP 內記錄的版本、包型態、Git commit 與 dirty 狀態：

```powershell
$zip = "deployment/patchcore_ai_release_2026.08.31.1_codeonly.zip"
python -c "import json,sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); m=json.loads(z.read('release_manifest.json')); print(m['version'], m['package_type'], m['git_commit'], m['git_dirty'])" $zip
```

預期結果中的包型態為 `codeonly`，最後一欄為 `False`。ZIP 內的 `checksums.txt` 會由 production 的 `install_patch.sh` 在安裝前逐檔驗證。

## 交付與安裝

不要解壓後再重新壓縮，也不要手動修改 ZIP 內容。將原始 ZIP 放到更新主機的 staging 目錄後，在第一台設備執行：

```bash
cd /root/Code/CAPI_AD
./promote_update.sh /aidata/capi_ai/update_repo/staging/patchcore_ai_release_<version>_codeonly.zip
```

`promote_update.sh` 會安裝、健康檢查、發布 `latest.json`，再供其他設備拉取。完整更新流程請參考 [實驗版 pull 模式自動更新流程](./experimental_auto_update.zh-TW.md)。

## 常見失敗

- **Git working tree 不乾淨**：先確認修改內容並完成 commit，或依團隊流程 stash；不要用 `--allow-dirty` 製作正式包。
- **required CODE_FILES missing**：通常是 clone 不完整、分支錯誤或打包清單未同步；先重新 pull，仍失敗就交由開發負責人處理。
- **同版本 ZIP 已存在**：使用下一個版本號，或先將舊 ZIP 移到備份位置；不要直接覆寫已發布版本。
- **本次有 static／圖片／backbone 變更**：停止 code-only 流程，改由開發負責人準備完整部署包。
