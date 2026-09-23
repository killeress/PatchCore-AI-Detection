# GPU 顯存診斷

## 正式投片後的條件式快取回收

正式 TCP 投片的分數與判定完成後、釋放共用 GPU 鎖之前，服務會檢查
是否需要回收未使用的 PyTorch CUDA 快取。預設啟用，舊設定檔無須新增欄位；
更新程式並重新啟動服務後生效。

預設必須同時符合以下條件才呼叫既有快取回收函式：

- `reserved - allocated >= 2048 MiB`。
- 整張 GPU 的 `device_free <= 2048 MiB`。
- 距離上次回收嘗試至少 60 秒；首次符合條件可立即執行。

冷卻時間由伺服器共用，不因切換機種／模型而重新計算。回收失敗或未釋放
任何空間也適用冷卻時間，避免每片反覆嘗試。未達門檻不強制同步；CPU／
尚未初始化 CUDA 時直接略過。推論失敗、沒有結果或前置檢查提早返回時
不執行此維護。此機制只接在正式 TCP 投片流程，獨立 CLI、Web 重跑與
背景診斷不新增自動回收。

可在 `server_config.yaml` 設定：

```yaml
inference:
  cuda_cache_cleanup:
    enabled: true
    min_unused_mib: 2048
    max_device_free_mib: 2048
    cooldown_seconds: 60
```

`enabled: false` 可停用；`cooldown_seconds: 0` 表示取消冷卻時間。
`min_unused_mib` 須大於零，其餘數值須為有限非負數。格式錯誤時略過回收、
記錄一次設定警告，保留正常推論流程。以上門檻是保守起始值，需依現場
顯存餘裕及延遲調整，不是顯存占用上限。

觸發時記錄 `[CUDA-MEM] cache-clear post-panel glass=...`，包含回收前後的
allocated、reserved、device free、實際 reserved 下降量、耗時與 PID。
請比較同一 PID，不能將重啟後下降當成原程序成功回收。既有每片
`[CUDA-SUMMARY]` 在 `process_panel` 結束時產生，早於此回收動作；回收結果
以 `cache-clear post-panel` 紀錄為準，不重設 peak 統計。

這項變更只回收未使用快取，不卸載模型、不修改 memory bank、KNN、精度、
分數、anomaly map 或 threshold，預期不影響模型分數及準確度。回收失敗
只記錄警告，不將已完成的判定改成錯誤。測試以模擬 CUDA 驗證門檻、鎖、
冷卻與回傳結果保持不變；實際釋放量、GPU 分數重播與耗時仍需現場驗證。

此機制不能釋放仍被活躍 Tensor 使用的記憶體，也不會消除下一次 KNN
計算的暫存高峰；同步與後續重新配置可能增加延遲。因此不逐 tile 清快取，
也不要求將模型常駐顯存降到零。

## 逐片與運算階段追蹤

顯存異常追蹤 **預設開啟**，部署新版並重新啟動 AI 服務即可取樣，
不需要修改 `server_config.yaml` 或另設環境變數。原有 `[CUDA-MEM]` 保持不變。

需要關閉時才設定 AI 服務程序的環境變數 `CAPI_CUDA_MEMORY_TRACE=0` 並重啟。
若使用 systemd，將 `Environment="CAPI_CUDA_MEMORY_TRACE=0"` 加入實際 AI
服務 unit 的 `[Service]` override，再 reload/restart 該服務。僅在登入 shell
執行 export 不會改變已運行的 systemd 服務環境。手動啟動則在同一個 shell
export 後使用原本啟動命令。若已有明確設定為 `0`，該設定仍優先於預設值。

正常情況僅在每片 process_panel 結束時寫一筆 `[CUDA-SUMMARY]`。
詳細階段取樣一直進行，先放入共用、執行緒安全的 RAM 環形緩衝區，最多
256 筆且序列化 UTF-8 內容不超過 1 MiB（Python 容器另有少量開銷）。
僅保存 JSON 數字／文字，不保留圖片、模型或 tensor。較舊紀錄會被淘汰，
程序被強制終止時尚未輸出的 RAM 紀錄會消失。

符合任一條件即寫 `[CUDA-TRIGGER]`，立即輸出已保留的 `[CUDA-TRACE]`：

- reserved 相較最近低點／上次增長觸發基準，累積增加至少 512 MiB；
  多次小幅增長也能觸發。
- 整張 GPU 顯存占用首次達到 85%；降到 80% 以下才重新啟用此門檻，
  避免持續高位或臨界波動反覆輸出。
- allocator 配置重試或 OOM 累計次數增加。
- 階段拋出例外，或顯存取樣失敗。

觸發後繼續輸出最多 64 筆、30 秒內的後續取樣，任一限制先到即結束。
這段期間新的異常數值仍在 TRACE 中，但不延長視窗、不重複輸出歷史。
此視窗以取樣驅動，不另開背景等待執行緒；後續取樣少時可能少於 64 筆。
高位穩定後恢復緩衝，仍持續寫每片摘要。異常頻繁時日誌仍會增加。

```bash
# 摘要與觸發事件
grep -E '\[CUDA-(SUMMARY|TRIGGER)\]' /aidata/capi_ai/logs/server.log | tail -n 100
# 已輸出的某片詳細階段（將玻璃編號換成實際值）
grep '\[CUDA-TRACE\]' /aidata/capi_ai/logs/server.log | grep 'YQ41TX206B15'
```

每筆為 JSON，`request` 區分重複投片，`span` 配對同階段 before/after，
`pid` 區分服務重啟。`sampled_at` 是原始 UTC 取樣時間，
LOG 行首時間是實際輸出時間；回溯分析請使用 sampled_at 與 sequence 排序。panel 包住 process_panel（包含 early return 與失敗），
不包含伺服器後續規格內判定、背景存檔或在呼叫前已跳過的請求。
PatchCore 逐 tile 與 batch、刮痕 image/tile/batch/forward 分層記錄。
scratch-image 可能只是跳過檢查；實際分類運算以 scratch-forward 為準。

- `allocated_mib`／`reserved_mib`：目前 tensor 用量／allocator 總保留量。
- `inactive_split_mib`：`inactive_split_bytes.all.current`，協助查碎片化；
  非 native allocator 可能不提供有效數據，零不代表沒有碎片化。
- `allocation_retries_total`／`oom_total`：程序 allocator 累計計數。
- `delta_*`：同一 span 的 after 減 before；巢狀階段不可相加。
- `input_shape`：進入推論器的影像尺寸，尚未經模型內部轉換；
  `tensor_shape`：明確建構 batch 的實際 NCHW 尺寸，`batch` 是實際批次量。
- 圖片、畫面、zone、模型路徑與 tile 編號依入口可取得的資訊記錄；
  單獨執行分類器或除錯入口可能沒有玻璃編號。

先找 SUMMARY 的 `delta_reserved_mib` 或 TRIGGER，再以同一 request 找已輸出的階段。
正常返回後的 tile/batch 呼叫摘要能觀察暫存釋放後用量；forward 摘要仍可能
包含輸入／輸出 tensor。after 的 `outcome=error` 表示該呼叫拋出例外。
CUDA 非同步執行且同程序可能有其他 GPU 工作，差值是相關線索，不是獨占
歸因或階段峰值；此功能不增加同步、不清快取，也不重設峰值。

詳細追蹤會增加日誌與查詢開銷，抓到重現案例後關閉，並及時保存現有輪替
日誌。正式 GPU 上仍需確認額外耗時；CPU／未初始化 CUDA 不輸出顯存快照。

## 原有五分鐘摘要

服務預設每 300 秒輸出一行 `[CUDA-MEM] periodic`，即使沒有投片也會記錄。
自動切換模型、刮痕分類器實際載入前後另有事件記錄。所有記錄沿用既有
`logging.file` 與檔案輪替，不建立額外報告或保存 tensor、圖片內容。

更新服務程式並重啟後生效；舊設定檔無須補欄位。可在 `server_config.yaml`
調整間隔，`0` 表示關閉週期記錄（模型載入等事件記錄仍保留）：

```yaml
inference:
  cuda_memory_log_interval_seconds: 300
```

在預設日誌位置查看：

```bash
grep '\[CUDA-MEM\] periodic' /aidata/capi_ai/logs/server.log | tail -n 20
```

- `pid`：服務程序編號，可對照 `nvidia-smi`，避免混用重啟前後資料。
- `machine`：取樣時的 fallback/active 機種。
- `inferencers`：服務管理的推論器數量，別名引用不重複計算。
- `patchcore_models`：推論器快取與 fallback 內已載入的模型物件數，重複引用只計一次。
- `scratch_models`：服務推論器中已載入的刮痕分類器數量。
- `allocated`：本程序 PyTorch tensor 目前佔用的顯存。
- `reserved`：本程序 PyTorch allocator 管理的顯存，包含 allocated，兩者不能相加。
- `peak_allocated` / `peak_reserved`：自上次重設峰值統計後的最高值；模型預熱會重設，並非固定五分鐘視窗。
- `device_used` / `device_free`：整張 GPU 的使用量／可用量，可能包含其他程序及非 PyTorch 配置。

模型數量僅統計服務管理的推論器，不是所有存活 tensor 或背景掃描模型的完整盤點。
週期摘要不鎖住推論流程，各欄位是近似時間的快照，切換模型期間可能看到過渡狀態。

週期與新增事件記錄不呼叫 `torch.cuda.synchronize()`、不清理快取，也不重設峰值。
尚未初始化 CUDA 的程序不取樣，避免 CPU／training-only 服務因診斷建立 CUDA context。
原有模型預熱的同步與快取清理行為維持不變。

若空閒時 `allocated` 穩定、`reserved` 偏高，優先調查 allocator 保留空間；若
`allocated` 隨工作持續上升，再追查模型或 tensor 引用。需要時再於服務程序內安排
清快取前後比較；此變更不新增清快取 API，也不自動清快取。

正常摘要每行約數百 bytes，每五分鐘一行，每日約 0.1 MB，另加少量事件記錄。
總日誌保留容量仍由既有 `logging.max_bytes` 與 `logging.backup_count` 決定。
