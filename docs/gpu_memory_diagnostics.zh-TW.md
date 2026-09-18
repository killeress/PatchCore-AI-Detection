# GPU 顯存診斷

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
