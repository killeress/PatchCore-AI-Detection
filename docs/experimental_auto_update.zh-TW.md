# CAPI AI 實驗版自動更新同步

目的：先在未正式 RUN 的兩台設備上驗證「只更新一台，其餘設備自動拉取更新包、套用、重啟」的流程。

## 設計邊界

- 採用 pull 模式：實驗機定期讀取更新主機上的 `latest.json`，不是由更新主機主動連進每台設備。
- 同步的是 patch ZIP，不是散檔。
- 不同步 `server_config.yaml`、SQLite DB、heatmap、log、現場資料、AOI 圖片路徑。
- 安裝仍走既有 `install_patch.sh`，會先驗 checksum、備份被覆蓋檔案、解壓、重啟、檢查 `/api/version`。
- updater 預設會設定 `CAPI_PATCH_AUTO_ROLLBACK=1`；若健康檢查失敗且 `rollback_patch.sh` 可執行，會自動回滾。

## 更新主機

在你 FTP 更新的那台主機上，準備一個可被其他設備讀取的目錄，例如：

```bash
mkdir -p /aidata/capi_ai/update_repo
```

## 簡化操作流程

第一台設備每次新版只需要跑：

```bash
cd /root/Code/CAPI_AD
./promote_update.sh
```

腳本會從 `/aidata/capi_ai/update_repo/staging` 自動選擇版本號最大的
`patchcore_ai_release_*_codeonly.zip`。若要指定特定版本，仍可直接傳入完整 ZIP 路徑：

```bash
./promote_update.sh /aidata/capi_ai/update_repo/staging/patchcore_ai_release_<version>_codeonly.zip
```

這支腳本會依序完成：

1. 安裝 ZIP 到第一台。
2. 用 `/api/version` 做健康檢查。
3. 發布 ZIP 與 `latest.json` 到 `/aidata/capi_ai/update_repo`。
4. 確認 `8088` 更新檔案服務可讀到 `latest.json`。

第二、三台只需各跑一次：

```bash
cd /root/Code/CAPI_AD
./setup_auto_update_client.sh http://<第一台IP>:8088/latest.json --run-now
```

這支腳本會建立 cron，每 5 分鐘檢查一次 `latest.json`。`--run-now` 會在設定完成後立刻執行一次更新。

之後每次新版只要重複第一台的 `promote_update.sh`，第二、三台會由 cron 自動拉取。

產生 patch ZIP：

```bash
cd /root/Code/CAPI_AD
python3 scripts/build_deploy_zip.py --patch-only --version 2026.07.06.2 --output-dir /aidata/capi_ai/update_repo
```

產生 `latest.json`：

```bash
python3 capi_update_agent.py publish \
  --package /aidata/capi_ai/update_repo/patchcore_ai_patch_2026.07.06.2.zip \
  --output-dir /aidata/capi_ai/update_repo
```

`latest.json` 會包含：

```json
{
  "version": "2026.07.06.2",
  "package": "patchcore_ai_patch_2026.07.06.2.zip",
  "sha256": "...",
  "size_bytes": 12345,
  "published_at": "...",
  "requires_restart": true
}
```

實驗階段可以先用 Python HTTP server 讓其他設備讀取：

```bash
cd /aidata/capi_ai/update_repo
python3 -m http.server 8088
```

正式化時建議改成固定的內網 HTTP/Nginx/Apache/檔案服務，不要依賴臨時 shell。

## 實驗機 Dry Run

第一次實驗機若還沒有 `capi_update_agent.py`，仍需先手動安裝一次包含 updater 的 patch ZIP，或先用 FTP 放入 `capi_update_agent.py`、`install_patch.sh`、`rollback_patch.sh`、`start_server.sh`。這是 bootstrap，之後才交給自動更新流程。

先不要真的安裝，只確認抓得到 manifest：

```bash
cd /root/Code/CAPI_AD
python3 capi_update_agent.py check \
  --manifest-url http://<更新主機IP>:8088/latest.json \
  --health-url http://127.0.0.1/api/version \
  --dry-run
```

預期會看到目前版本與遠端版本差異，且不會解壓或重啟。

上面的健康檢查 URL 依目前 production `server_config.yaml` 的 web port 80 撰寫；若設備實際使用 8080，改成 `http://127.0.0.1:8080/api/version`。

## 實際套用一次

```bash
cd /root/Code/CAPI_AD
python3 capi_update_agent.py check \
  --manifest-url http://<更新主機IP>:8088/latest.json \
  --health-url http://127.0.0.1/api/version
```

成功後檢查：

```bash
cat VERSION
curl -fsS http://127.0.0.1/api/version
tail -80 update/auto_update.log
cat update/auto_update_state.json
```

## 週期性檢查

cron 範例，每 5 分鐘檢查一次：

```cron
*/5 * * * * cd /root/Code/CAPI_AD && /usr/bin/python3 capi_update_agent.py check --manifest-url http://<更新主機IP>:8088/latest.json --health-url http://127.0.0.1/api/version >> /aidata/capi_ai/logs/auto_update_cron.log 2>&1
```

也可以先用 loop 模式觀察：

```bash
cd /root/Code/CAPI_AD
nohup python3 capi_update_agent.py check \
  --manifest-url http://<更新主機IP>:8088/latest.json \
  --health-url http://127.0.0.1/api/version \
  --loop --interval 300 \
  >/aidata/capi_ai/logs/auto_update_agent.out 2>&1 &
```

## 失敗處理

- checksum 不符：不會安裝，會記錄到 `update/auto_update_state.json`。
- installer 失敗：不會標記成功，同一版本預設不重試；修好原因後加 `--retry-failed` 再跑。
- 健康檢查失敗：`install_patch.sh` 在 updater 呼叫下會自動執行 rollback。
- 若設備沒有 `curl`，`install_patch.sh` 會略過健康檢查；實驗機請先安裝或確認有可用 `curl`。

手動回滾仍可用：

```bash
cd /root/Code/CAPI_AD
./rollback_patch.sh .patch_backups/<版本_時間戳>
```

## 停用實驗

如果用 cron，移除對應 cron entry。

如果用 `--loop`，找出程序並停止：

```bash
pgrep -af capi_update_agent.py
kill <pid>
```
