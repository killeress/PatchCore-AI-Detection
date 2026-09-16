# SoftPatch+ 與現有特徵域清洗：差異與實驗方案

研究日期：2026-09-15。比對基準：本機專案 HEAD `8e78eb9` 的訓練、清洗與推論程式，以及本機載入的 anomalib 原始碼。

後續實作：已新增 `softpatch_plus_v1` Tile 適配模式，設定與限制見 [SoftPatch+ 模式說明](softpatch_plus_mode.zh-TW.md)。以下保留研究當時的基準描述；float32 原地正規化問題已在本次功能實作修正。

本次完成原始碼研究與小型數值檢查，未訓練新模型、未更換部署模型、未修改產品程式。尚無本機資料上的效果結論。

使用者已確認雙目標：改善訓練混入 NG 後的漏檢，同時減少正常邊緣／稀有紋理的誤報。

## 結論

值得實驗，優先保留目前處理重疊 Tile 的機制，再測試 LOF 評分與推論時的 memory-bank 權重。兩項改動應拆開比較，才能分辨效果來源。

現有系統已具備「抽特徵 → 清洗 patch → coreset → memory bank」的接入點。主要欠缺的是相對密度評分、多評分器融合，以及清洗分數隨 memory-bank 特徵進入推論。

新模型訓練頁預設仍是清洗關閉；以下比較的是功能開啟時的行為，不能據此認定某個已部署模型開啟了清洗。實際需查看該 bundle 的 `patchcore_params`、`feature_cleaning_by_zone` 和清洗報告。

## 1. 論文與官方程式的要點

論文 [SoftPatch+: Fully Unsupervised Anomaly Classification and Segmentation](https://arxiv.org/html/2412.20870v2) 的問題是正常訓練資料混入缺陷。它在 coreset 前進行 patch 級清洗，保留污染影像中的正常區域；SoftPatch+ 預設融合 LOF 與多變量 Gaussian 的排名分數。分類與分割分別建立 coreset，移除比例為 15% 與 50%，LOF 的 k 為 6。這些是論文設定，不是本系統的建議預設值。

官方 [softpatch.py](https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch/blob/main/src/softpatch.py) 將特徵按 feature-map 位置分組，跨影像計算 LOF 或 Gaussian 分數；選取 coreset 後，依相同索引保存權重。推論先以原距離找最近的 memory-bank 特徵，再把距離乘上該特徵的權重。

```text
m* = 距離 query patch 最近的 memory-bank 特徵
patch_score = distance(query, m*) × weight(m*)
```

這裡是降低可疑記憶點作為「正常證據」的可信度。以權重 2 為例，距離 0.2 會變成分數 0.4；這只是方向示例，不是實測結果。它不是先乘權重再找最近鄰，也不是把 query 的异常分數一律壓低。距離恰為零時，有限乘法權重仍無法把零變成正值。

版本注意：目前讀到的官方 main 分支 `_compute_patch_weight` 只有 lof、lof_gpu、nearest、gaussian 分支。README 雖提及 SoftPatch+，不能把這個函式直接視為已實作論文的完整排名融合與雙 coreset；重現前須確認版本或自行補足。

論文亦非所有設定都勝出：Table 1 的 MVTec、10% noise、No Overlap 分類 AUROC，PatchCore / SoftPatch-LOF / SoftPatch+ 分別為 0.984 / 0.986 / 0.982。其高污染 Overlap 實驗含訓練與測試影像來源重疊；本案應另保留獨立 Panel 測試集。[論文 Table 1、§4](https://arxiv.org/html/2412.20870v2#S4)

## 2. 本機系統實際怎麼做

### 2.1 共通流程與預設值

- `capi_train_new.py:102`：提供 off、`knn_cosine_q99_v1`、`context_overlap_adaptive_v1`。
- `capi_train_new.py:111`：預設 k=30、keep_ratio=0.99、reference_size=20,000、query_chunk=1,024、seed=42、MAD z=6。
- `capi_train_new.py:224`：預設輸入 512×512、coreset ratio=0.1、precision=float16。
- `capi_train_new.py:366`：支援 INNER／EDGE 各自指定 mode、k、keep_ratio；未分區時預設範圍為 INNER。
- `capi_train_new.py:1447`：將清洗 callback 加入訓練。
- `capi_patchcore_feature_cleaning.py:236`：在 validation 開始前清洗，epoch end 為備援，避免重複執行。

清洗的基本分數是第 k 個最近參考特徵的 cosine distance。參考集合最多隨機取 20,000 個；會排除 query 本身的同一筆索引。它不是 LOF：沒有計算「鄰居的局部密度相對於自己的密度」。

參考集合跨目前訓練單元內的位置混合；不要求鄰居來自另一片 Panel，也不限制在相同 feature-map 座標。INNER／EDGE 與 lighting 的模型分組仍由原訓練流程維持。

### 2.2 舊版 KNN 百分位模式

候選區域預設為輸入中央 384×384，按 feature-cell 中心投影選取。候選分數超過 keep_ratio 分位數時移除，其餘位置受保護。

keep_ratio=0.99 表示候選區域約移除最外圍的 1%；相同分數可能使實際比例更少。此比例不是整張影像或整個訓練集的 NG 比例。

證據：`capi_patchcore_feature_cleaning.py:360`、`:716`。

### 2.3 重疊上下文自適應模式

1. 把 feature cell 映射到 Panel 實體座標；分組 key 包含 `panel_path`，所以是在同一片 Panel 內配對。
2. 同一實體位置的多個 Tile 視角，只選離 Tile 邊界最遠的一個加入候選參考集合，再從參考集合抽樣。
3. 使用同一個全域參考集合計算 kNN cosine 分數。
4. 門檻為 `max(median + 6 × 1.4826 × MAD, Q_keep_ratio)`。
5. 重疊組所有視角都超標才移除；有一個視角不同意，整組保留。
6. 沒有座標資料的位置受保護；單視角位置依自動邊界保護距離決定是否可清洗。

因此 keep_ratio=0.99 是候選特徵最大約 1% 的移除上限，可能實際移除 0 個。沒有可估算的 overlap guard 時，單視角位置不會成為候選，應先看 candidate 數量，不要把「移除 0」直接解讀為資料很乾淨。

證據：`capi_patchcore_feature_cleaning.py:488`、`:592`、`:646`、`:686`、`:704`。

### 2.4 目前分數只用於清洗與追溯

系統會保存 distances、reason_codes、removed_indices、overlap votes 與 coreset_indices，已能定位某個特徵來自哪張 Tile。但目前建構 memory bank 時只存入特徵張量，沒有以同順序保存清洗權重供推論使用。

不要把「未使用清洗權重」理解成「整個系統沒有權重」。本機 anomalib 本身有 PatchCore 的鄰域 image-score reweighting；系統也有 MARK 區域處理。這兩者都不等同訓練污染可信度。

證據：`capi_patchcore_feature_cleaning.py:98`、`:837`；`capi_inference.py:2200`、`:2370`；本機 `D:/SourceCode/anomalib/src/anomalib/models/image/patchcore/torch_model.py:385`。

## 3. 對本案的推論

以下是依本機架構提出的假設，須用資料驗證。

### 稀有正常紋理與邊角

現有全域 kNN 距離可能把少見但正常的紋理視為低密度區。LOF 改為比較局部相對密度，值得測試能否減少這類誤刪；但如果正常小群的樣本比 k 還少，鄰居仍會跨群，不能保證保護稀有正常。

### 重複缺陷與密集污染

同類缺陷重複出現時，會互相成為近鄰。現有方法與 LOF 都可能把它看成正常群。可以在後續比較 Gaussian 排名融合，並另測排除同一 Panel 來源的參考鄰居，確認是否受重複樣本影響。

### 現有重疊保護值得保留

它針對切圖邊界與上下文不足的特徵不穩定，已有實體座標與可追溯的保留原因。直接換成整批特徵截尾會失去這些保護。

### 同位置統計不能直接套在任意 Tile

不同 Tile 的左上角可能是完全不同結構。若要做跨影像同位置統計，應先建立相同產品、光源、區域及對齊後結構座標的對應。Panel 樣本很少時，逐位置 LOF 與 Gaussian 估計也會不穩定；完整 covariance 還需要降維與正則化。先做分區的 LOF 混合方案比較實際，名稱應標明是本系統改編，不能稱為論文完整重現。

## 4. 建議實驗順序

### 第一階段：現有 UI 即可比較

固定訓練 Panel、Tile 清單、前處理、feature layers、pool kernel、precision、coreset ratio 和驗證資料，建立三個新訓練 job：

| 組別 | INNER | EDGE | 用途 |
|---|---|---|---|
| A0 | off | off | 無清洗基準 |
| A1 | 舊版 KNN，k=30，keep=99% | off | 檢查固定截尾的效果 |
| A2 | 重疊上下文，k=30，keep=99% | off | 檢查目前自適應與重疊保護的效果 |

A1 與 A2 的候選區域和規則不同，這是完整配方比較，不能將差異只歸因於 MAD 門檻。

若要測 EDGE，另建 A3，只把 A2 的 EDGE 改成重疊上下文、k=30、keep=99.9%；再視結果比較 99.5%。這是探索起點，並非已驗證最佳值。

操作位置：新模型訓練 Step 1 → INNER／EDGE 分開設定。局部重訓會沿用原 bundle 的設定，不適合切換清洗方法的對照實驗。

### 第二階段：新增可選實驗模式

| 組別 | 清洗評分 | 推論清洗權重 | 要回答的問題 |
|---|---|---|---|
| B0 | 現有 kNN + 重疊保護 | 關 | 重用 A2 基準 |
| B1 | 現有 kNN + 重疊保護 | 開 | 單獨測試保留可疑特徵的權重 |
| B2 | LOF + 同樣重疊保護 | 關 | 單獨測試相對密度評分 |
| B3 | LOF + 同樣重疊保護 | 開 | 是否有疊加效益 |
| B4 | LOF / Gaussian 排名融合 + 重疊保護 | 開 | 第二輪才測高污染與成群缺陷 |

B0～B3 先固定候選集合、參考抽樣、k、移除上限；再把 k=6 / 15 / 30 當作獨立參數探索。比較時需記錄實際移除數，必要時另做相同移除數的對照，避免只測到清洗強度不同。

B1 的 kNN soft weight 需要新定義，不能直接把 cosine distance 當成可信度倍率。可試 `1 + alpha × percentile(score)`，令 alpha=0 能精確回到無權重；這是本案可調的設計，不是論文公式重現。B3 若試原始 LOF 權重，需另外處理極端值並記錄是否做過裁切。

第一輪保留本機原有 image-score 聚合，將實驗明確標成混合方案。若要精確重現論文，需另控制聚合規則、位置分組、距離定義與雙 coreset，不能把多項差異混在同一個效果數字內。

## 5. 資料與判定標準

1. 依實體 Panel／批次分成訓練、驗證、最終測試集；同一 Panel 的 Tile、重疊切片、同源不同光影像與增強版本放在同一分組。
2. 準備可信正常資料，以及獨立來源、已確認缺陷的污染池；先測 0%、5%、10% 的訓練污染情境。明確記錄比例分母，例 `NG Panel / 全部訓練 Panel`，另記 NG Tile 與缺陷 patch 的占比。
3. 如模擬現場未確認的資料，可另外測自然污染組，但不要把未知污染率標為已知比例。
4. 驗證集選 threshold，測試集只用來評估。每個模型重新校準 threshold，不能沿用同一個 raw threshold 比較有權重與無權重模型。
5. 主要指標：相同 NG 召回率下的 OK 誤報率；以及相同複判工作量下的 NG 漏檢率。分別報 Tile 與 Panel 結果，INNER／EDGE、微小缺陷、邊緣缺陷分開觀察。
6. 有缺陷 mask 才報 pixel AUROC／AUPRO；否則不要只憑熱圖外觀宣稱分割改善。報原始混淆矩陣、NG 數量；若測到零漏檢，也不代表未知資料零風險。
7. 清洗本身也要評估：被移除的正常區域、污染區域移除率、進入 coreset 的污染率、重疊不一致保留率、候選／受保護特徵數。可先抽樣人工核對刪除最多的 Tile。
8. 第一輪只篩方向，對候選勝出方法再做至少三個抽樣／coreset seeds 的確認，並記錄訓練耗時、峰值 RAM/VRAM、推論延遲、memory-bank 大小。

coreset ratio 相同不代表 memory-bank 數量相同，因為清洗後特徵數不同。主實驗可維持現行 ratio 以比較實際配方，另補相同 bank 大小的對照，確認差異不是容量造成。

## 6. 實驗前要排除的 float32 副作用

本次直接呼叫現有 `_kth_cosine_distances` 的小型 CPU 檢查已重現：

```text
float32 input: [[3,4], [4,3], [0,2]]
after call:    [[0.6,0.8], [0.8,0.6], [0,1]]
fp32_changed = True
fp16_changed = False
```

原因在 `capi_patchcore_feature_cleaning.py:955`：`detach().to(device, dtype=float32)` 在原張量已是同裝置 float32 時可能共享儲存空間；下一行 `div_` 會改動原 embedding。這會讓開啟清洗的 float32 組同時改變 coreset 的特徵尺度，干擾比較，也可能造成訓練與推論特徵尺度不一致。

目前預設 float16；本機 anomalib 對應路徑會將模型轉 half，不能因此宣稱所有既有模型都受影響。應在真正訓練時記錄 embedding dtype，再確認是否落在此條件。

既有 `test_removes_isolated_feature_and_preserves_raw_embeddings` 使用 float64，轉成 float32 時產生副本，因此沒覆蓋此情況。後續實作實驗模式時，先改成非原地 normalize 或明確複製，並讓原特徵不變的測試覆蓋 float16、float32、float64。本次僅記錄發現，尚未修改此函式。

## 7. 接入工作清單

- 評分：在 cleaning callback 分離「候選／reference 規劃」、「評分」、「門檻」、「重疊決策」，重用現有追溯資料。
- 權重：保存與 memory-bank 行數、順序完全一致的 tensor；coreset 選取、儲存、重新載入與裝置轉換都要維持對齊。
- 推論：在 query 的最近鄰查詢後套權重，使 image score 與 anomaly map 都收到一致的 patch score。不要對 PatchCore 內部支援鄰居查詢一概套相同權重。
- 相容性：處理 `capi_inference.py` 的 fp16 nearest-neighbor 優化與 MARK scoring；舊模型缺少權重時使用全 1。
- 格式：新增明確策略版本與權重參數到 bundle manifest，維持 experimental_training 標記。
- 測試：聚焦 LOF／排名計算、原特徵不變、coreset 權重索引、save/load、全 1 權重等價、MARK 與熱圖一致性。依專案偏好，不需為研究或這些局部改動執行完整回歸。

目前 API 的 keep_ratio 下限為 0.90，不能直接設定論文的 0.85 或 0.50。若將來需要精確重現，應另設實驗入口，而不是直接改既有訓練預設。

## 建議決策

先跑 A0／A1／A2，建立本機現有配方的基準；新功能優先做 B1～B3。使用者同時重視漏檢與誤報，因此 B2 與 B3 必須共同比較，B1 用來辨認權重本身的影響。

以相同 NG 召回率下的誤報、相同複判量下的漏檢率共同選型，並檢查各缺陷類別與 EDGE 是否退步。權重可能改善漏檢卻提高誤報，不應僅因總體 AUROC 上升就採用；若沒有同時改善的方案，明確呈現取捨與操作門檻。

逐位置 Gaussian 與雙 coreset 放到後續，等確認對齊、樣本量與前一輪效果後再投入。
