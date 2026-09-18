# Dashboard GPU 異常提醒

機台 `/dashboard`、`/v3/dashboard` 與中控總覽共用 `/api/status` 的 `gpu_health`。機台畫面頂端持續顯示提醒，狀態指示同步標示異常；中控將該線體計入「服務異常」，並在目前告警、總覽異常欄與卡片 GPU 欄顯示。診斷資訊包含錯誤內容及檢查時間。

| 情境 | 提醒 | 恢復條件 |
|---|---|---|
| NVIDIA 驅動查詢失敗或回報沒有 GPU | 紅色「GPU 無法存取」 | 後續硬體查詢恢復正常，且沒有本次程序的致命 CUDA 錯誤 |
| 查詢超過 3 秒、找不到 nvidia-smi 或回應無法解析 | 黃色「GPU 狀態無法確認」 | 後續查詢成功；這類提醒不代表已證實硬體損壞 |
| 要求 auto/cuda，但實際載入的模型使用 CPU | 黃色「推論已降級 CPU」；若同時已確認驅動不可用則維持紅色 | 模型實際重新載入 GPU；只有 nvidia-smi 恢復仍不會解除 |
| 推論回報 launch failure、illegal memory access、device-side assert、CUDA unknown error 等致命錯誤 | 紅色「GPU 推論異常」 | 保留首個錯誤，確認 GPU 恢復後重新啟動 AI 服務 |
| 明確指定 CPU，模型亦使用 CPU | 不因 GPU 查詢失敗而告警 | 預期行為 |

「運算裝置」讀取執行中推論器的實際 device；API 另回傳設定值 `server.requested_device` 與實際清單 `server.inference_devices`，可辨識 AUTO 設定下的 CPU 降級。多個可服務模型同時使用不同裝置時，仍提示其中的 CPU 降級。

硬體查詢沿用 30 秒快取，nvidia-smi 單次逾時為 3 秒；畫面於下一次既有輪詢更新。本次程序的致命 CUDA 錯誤直接由狀態追蹤器回報，不必等硬體快取到期。中控沿用原設定的輪詢週期，因此顯示時間可能比機台畫面晚。

致命 CUDA 錯誤不會被一次成功的 nvidia-smi 查詢清除：驅動可回應時，既有 CUDA context 仍可能已失效。一般圖片／模型錯誤或單純 CUDA out-of-memory 不會被此功能鎖定為致命 GPU 故障。正常硬體查詢也只代表監測介面可回應，並非額外執行 CUDA 推論自測。

本功能不改變判定結果、TCP 回覆、CPU fallback 策略或服務生命週期，不會自動 reboot、重載驅動或重啟 AI。GPU 異常提醒是維護線索，不能單靠它判定哪個零件損壞。
