# Over-Review Scratch — Deployment Handoff

**Date:** 2026-04-14
**Status:** POC 實驗階段結束，進入部署規劃
**Previous context:** `docs/over_review_scratch_poc_result.md`

---

## 1. 新對話一開始讀這份

請先讀：
1. **這份**（整體部署計畫）
2. `docs/over_review_scratch_poc_result.md`（完整 ablation 歷史 + 當前最強組合）
3. `CLAUDE.md` §"Inference Pipeline" 與 §"Threading & Concurrency Model"（知道要整合到哪）

---

## 2. 確定採用的方案（POC 結論）

### 2.1 當前最強配置

**LoRA r=16 / 2 blocks / 15 epochs + CLAHE cl=4**

| 指標 | 數值 |
|---|---|
| Realistic scratch_recall | 90.4% ± 5.8% |
| Realistic true_ng_recall | 98.2% ± 1.7% |
| **Oracle scratch_recall**（零漏檢上限） | **86.0% ± 9.2%** |
| PR-AUC | 0.931 |
| 推論 VRAM | ~1 GB |
| 推論時間 | ~50ms / crop（2080 Ti）/ ~25ms（5060 Ti 生產機）|
| 訓練時間 | ~25 分鐘（2080 Ti, 5-fold）/ ~10 分鐘（5060 Ti）|

### 2.2 部署用 preprocessing pipeline（必須原封不動）

```python
# 順序不能改！
1. PIL.Image.open(path).convert("RGB")
2. CLAHE (clipLimit=4.0, tileGridSize=(8,8)) on grayscale → stack ×3 channel
3. Resize((224, 224))  # torchvision default bilinear
4. ToTensor()
5. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
6. model(x) → CLS token (768-d)
7. LogReg.predict_proba(x)[:, 1] → scratch score
```

參考實作：`scripts/over_review_poc/features.py::preprocess_clahe` + `build_transform_clahe`。

### 2.3 Conformal 部署 threshold

`scripts/over_review_poc/zero_leak_analysis.py` 的 conformal 機制：
- 訓練集中拿 20% 當 calibration set（group-aware，同 glass_id 不跨）
- threshold = max(calibration true_ng 分數) × safety_multiplier
- safety=1.0 時 leak ~1.6%；safety=1.05 時 scratch catch 稍降、leak 更小

**Conformal 實測結果**（2026-04-14 跑完 r=16 + conformal 版本）：
```
Conformal scratch_recall = 90.4% ± 5.8%   （跟 Realistic 幾乎一樣）
Conformal leak           = 2.6% ± 2.8%    （fold 5 拖累平均到 8.1%）
```

**結論**：LoRA 後的 LogReg 分數分離度夠好，conformal calibration 在 r=16 這組 embedding **沒帶來額外效益**。純 threshold 方案的 leak 下限約 1.7-2.6%，要達真 0 漏檢必須走**兩階段人工覆判**（見 §3.1-E）。

---

## 3. 部署規劃（這就是 Route A 要做的）

### 3.1 開一份 deployment spec（新對話第一件事）

位置：`docs/superpowers/specs/YYYY-MM-DD-over-review-scratch-deployment.md`

要涵蓋：

#### A. ScratchClassifier pipeline 類別設計
- 輸入：`PIL.Image` 或 `np.ndarray`
- 輸出：`(scratch_score: float, is_scratch: bool)`
- 內部封裝 CLAHE + LoRA DINOv2 + LogReg + threshold
- Metadata（threshold、clahe params、preprocessing_id）綁 pickle

#### B. 整合點選擇
CAPI inference 流程（見 `capi_inference.py`）決定 scratch classifier 插在哪：
- 選項 1：PatchCore 判 NG 之後，逐 tile 餵給 scratch classifier → 分數高才保留 NG
- 選項 2：只對「過檢疑似 scratch 類」送覆判（但目前沒有 scratch 前置判斷）
- 推薦：**選項 1**，所有 NG tile 都過 scratch classifier，classifier 分數低（not scratch）才翻 OK

#### C. Threading / concurrency
`capi_server.py` 已有 `_gpu_lock` 序列化 GPU 推論。ScratchClassifier 要不要佔用同一把鎖？
- 建議：**共用 `_gpu_lock`**，順序變成 PatchCore → Scratch classifier 串行
- 否則 GPU OOM 風險

#### D. Latency budget
- 單張 panel 典型 ~500 tiles
- 現行 PatchCore ~X ms/tile（請量測）
- 新增 LoRA DINOv2 ~25-50ms/tile
- 總增加：預估 **+15-25 秒/panel**
- 需跟 AOI 產線確認可接受

#### E. 兩階段系統設計（達真 0% 漏檢）
```
crop → LoRA DINOv2 → LogReg score
                     │
            score > HIGH_THR  ───→ 高信心 scratch，自動擋下
            LOW < score < HIGH ──→ 送人工覆判介面
            score ≤ LOW_THR ────→ 放過
```
- HIGH_THR 設為 conformal threshold（確保 0-leak）
- LOW_THR 可設為 HIGH_THR × 0.5 或根據人工覆判人力決定
- 需開一個人工覆判介面 / dashboard 頁面

#### F. Runtime config 開關
仿造現行 `config_params` 機制，加：
- `scratch_classifier_enabled`: bool
- `scratch_threshold_high`: float
- `scratch_threshold_low`: float
- `scratch_model_path`: str

讓現場可以動態關掉 / 調整 threshold 而不重啟。

#### G. 離線部署
- DINOv2 base 權重 (330MB) copy 到產線
- LoRA 權重 (~1MB) + LogReg (~10KB) 打包
- 參考 `scripts/over_review_poc/prepare_offline_model.py` 作為基礎擴充

### 3.2 產出部署用模型檔（Todo C1）

新 script `scripts/over_review_poc/train_final_model.py`：
```bash
python -m scripts.over_review_poc.train_final_model \
    --manifest datasets/over_review/manifest.csv \
    --transform clahe --clahe-clip 4.0 \
    --rank 16 --n-lora-blocks 2 --epochs 15 \
    --calib-frac 0.2 \
    --output deployment/scratch_classifier_v1.pkl
```

內容：
- 用全部 1748 筆訓練單一 LoRA（不再 k-fold）
- 保留 20% group-aware 當 conformal calibration
- Pickle：
  - LoRA weights (state_dict)
  - LogReg (sklearn model)
  - conformal threshold
  - preprocessing metadata
  - git_commit / run_at / dataset_sha256

### 3.3 整合到 CAPI pipeline

具體程式碼變更點（參考 `capi_inference.py`）：
- 在 `CAPIInferencer.__init__` 載入 `ScratchClassifier`（如果 config enabled）
- 在 `process_panel` tile 判 NG 後，若 config enabled → 過 classifier
- score 高 → 保持 NG；score 低 → 翻 OK
- DB schema 可能要加 `scratch_score`、`scratch_filter_flag` 欄位

### 3.4 測試計畫

- `tests/test_scratch_classifier.py`：單元測試 preprocessing 一致性、推論數值正確
- `tests/test_integration_scratch.py`：送一張已知 scratch 的 panel + 一張真 NG，驗證判斷正確
- Staging 環境實跑一天流量，驗證 latency、recall、false rate

---

## 4. 相關參考文件

- **資料維護策略** — `docs/over_review_scratch_data_maintenance.md`
  涵蓋：重訓觸發條件、三道驗證 gate、hold-out test set、drift monitoring、
  active learning、標註 QC、模型版本管理、自動化 pipeline。**上線前建置
  hold-out test 必讀**。
- **POC 完整 ablation 歷史** — `docs/over_review_scratch_poc_result.md`

---

## 5. Recent commits (context)

```
844a3e5 feat: finetune_lora.py 加入 conformal threshold 計算
6d8d090 docs: LoRA r=16 結果 — Oracle 86.0%
8fa81c1 docs: 加入 patch-pool + LoRA fine-tune 結果
a3f2d16 feat: patch-level pool + conformal threshold + LoRA fine-tune
54cff69 docs: 加入 A-D 實驗結果 + clip sweep / classifier 表
8bc081d feat: CLAHE + clip sweep + tile-only + classifier compare
56742cf docs: 加入 ablation log + A1/A2 結論
4895fa4 feat: Otsu panel-mask preprocessing ablation
45e4432 docs: 結果摘要 + 待辦清單 + 新對話交接
```

## 6. 新對話開場建議

告訴下一個對話：
> 「接續 over-review scratch POC，POC 已完成，進入部署規劃。先依序讀：
> 1. `docs/over_review_scratch_deployment_handoff.md`（這份，部署規劃主文件）
> 2. `docs/over_review_scratch_data_maintenance.md`（資料維護與持續學習策略，上線前的 hold-out 必讀）
> 3. `docs/over_review_scratch_poc_result.md`（完整 ablation 歷史，可選）
>
> 然後開一份 deployment spec：`docs/superpowers/specs/YYYY-MM-DD-over-review-scratch-deployment.md`，涵蓋上面 §3.1 A-G 全部項目 + §4 引用的資料維護策略。spec 完成後再進 plan → implementation。」
