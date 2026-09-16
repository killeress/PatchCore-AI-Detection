# SoftPatch+ 實驗模式

## 操作位置

新機種訓練 → Step 2「選擇訓練資料」→ 展開「訓練設定（進階）」→ 特徵域清洗選 **SoftPatch+（實驗：Tile 適配）**。

可選共用設定，或 INNER／EDGE 分別設定模式、K 與保留比例。下方 SoftPatch+ 進階參數由所有採用此模式的區域共用。局部／單子模型重訓沿用原 bundle 配方。

## 參數

| 設定 | 預設 | 可選範圍與作用 |
|---|---|---|
| 清洗模式 | 關閉 | 手動選 SoftPatch+ 才啟用 |
| 清洗 K | 30 | 1～200；建議固定其他條件後比較 6／15／30 |
| 保留比例 | 99% | SoftPatch+ 為 50%～100%；其他清洗模式維持 90%～100% |
| 離群評分 | LOF＋Gaussian | 可改成僅 LOF 做對照 |
| 推論權重 | 開 | 關閉時只清洗，所有 memory-bank 權重為 1 |
| 權重強度 | 1 | 0～4；倍率 `1 + 強度 × 離群排名`，預設介於 1～2；0 等同無權重 |
| 重疊／邊界保護 | 開 | 使用 Panel 座標與重疊視角的一致性；關閉後所有特徵可清洗 |
| 評分特徵維度 | 32 | 8～128，原維度較小時不升維；只影響清洗評分 |
| 參考特徵上限 | 2048 | 256～8192，實際不超過可用特徵數 |

保留比例作用於可清洗候選特徵。相同排名或重疊保護會使實際移除量減少，並非保證刪除固定數量。設定保留 100% 時仍可計算與使用推論權重，可用來單獨觀察加權效果。

重疊保護開啟時，缺少座標、受邊界保護及重疊投票不一致的位置，權重維持 1。沒有 overlap 可估算保護距離時，單視角位置可能全部受保護；請檢查報告的 candidate 數量與 reason，而非只看移除率。

## API 範例

以下為 `training_params` 內容，先從 INNER 開始，EDGE 保持關閉：

```json
{
  "feature_cleaning_by_zone": {
    "inner": {"mode": "softpatch_plus_v1", "k": 6, "keep_ratio": 0.99},
    "edge": {"mode": "off", "k": 30, "keep_ratio": 0.99}
  },
  "softpatch_plus_config": {
    "discriminator": "lof_gaussian",
    "soft_weight": true,
    "weight_strength": 1.0,
    "context_overlap": true,
    "projection_dim": 32,
    "reference_size": 2048
  }
}
```

共用模式可使用 `feature_cleaning_mode`、`feature_cleaning_k`、`feature_cleaning_keep_ratio`、`feature_cleaning_scope`。區域設定存在時優先使用區域設定。

## 此版本的實作範圍

- 名稱 `softpatch_plus_v1`，屬於現有 Tile 流程的改編版本。
- 各訓練單元內混合位置建模，使用固定 seed 的隨機投影與參考抽樣。
- 以參考鄰域計算 LOF；Gaussian 採 10% 對角 shrinkage 和數值正則化。兩者分數各轉成含 ties 平均排名的百分位，再平均融合。
- 使用排名分位數截尾，可加重疊一致性保護；不對排名套用現有 MAD 門檻。
- coreset 保存原特徵，另保存與其行順序一致的 `softpatch_weights` buffer；Torch 匯出與重新載入保留權重。
- 推論先找最近鄰，再加權 query patch 分數；熱圖、原有 PatchCore 圖像聚合及 MARK 排除均使用該 patch 分數。內部支援鄰居搜尋仍使用原距離。
- 單一 memory bank；沒有實作論文逐位置跨影像建模、分類／分割雙 coreset，因此不能直接聲稱重現論文結果。
- 實驗模型沿用既有 experimental_training 標記與人工啟用流程。部署程式需包含新增的 `capi_patchcore_softpatch.py`、`capi_softpatch_config.py`；部署打包清單已納入。

模型庫與訓練完成頁會顯示配方。清洗報告的 `score_metric=outlier_rank`，沿用 `distances` 陣列欄位保存排名分數，以相容既有可視化；數值不再代表 cosine 距離。

## 推論紀錄辨識

推論 Log → 中文摘要 →「本次使用模型」會依光源與 INNER／EDGE 列出清洗方式、LOF／LOF＋Gaussian、推論權重開關、強度、實際倍率範圍與模型路徑。原始 Log 使用 `[MODEL_TRAINING]` 保存當次資訊，快取模型亦會記錄。

新訓練模型將配方嵌入模型；先前匯出的模型在載入時由同目錄 manifest 補讀配方，並以實際載入的模型類別辨識 SoftPatch+。更換檔案後仍使用舊快取時，紀錄保留舊載入資訊；重新載入後才更新。沒有配方的欄位顯示未記錄，不推定為清洗關閉。此功能僅適用更新後產生的推論紀錄，既有歷史 Log 無法回補。

## 對照實驗

固定相同 Panel／前處理／coreset ratio／precision，比較：清洗關閉、現有重疊清洗、僅 LOF 無權重、LOF＋Gaussian 無權重、LOF＋Gaussian 有權重。

以相同 NG 召回率下的 OK 誤報率，以及相同人工複判量下的 NG 漏檢率共同評估；INNER／EDGE 分開看。訓練、驗證、測試按實體 Panel 分組，每個模型重新校準 threshold。

本次程式測試使用小型合成特徵與輕量模型，不代表已驗證現場缺陷效果。未執行正式 GPU 訓練、未更換已部署模型。

## 同次修正

舊版 cosine 清洗改為非原地正規化 query，避免 float32 特徵被連帶改變尺度。原特徵保存測試已覆蓋 float16、float32、float64。
