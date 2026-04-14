# Over-Review Scratch POC — 結果與交接

**Date:** 2026-04-14
**Status:** POC 完成，verdict = GO（但有 3.3% 漏檢風險需注意）
**Branch:** `main`

---

## 1. 專案脈絡（新對話讀這裡）

目標是降低 CAPI AI 的過檢率。先拿 `datasets/over_review/` 的 `over_surface_scratch`（114 筆）做 POC，用 DINOv2 ViT-B/14 凍結特徵 + LogReg/k-NN 二元分類器（`scratch` vs `not_scratch`），驗證能否在不漏真 NG 的前提下擋下 scratch 過檢。其他過檢類別（overexposure、within_spec…）之後逐一擴充。

**參考文件：**
- **Spec（設計）：** `docs/superpowers/specs/2026-04-14-over-review-scratch-poc-design.md`
- **Plan（實作計畫）：** `docs/superpowers/plans/2026-04-14-over-review-scratch-poc.md`
- **本檔（結果交接）：** `docs/over_review_scratch_poc_result.md`

---

## 2. 結果摘要

### Main Metrics（5-fold CV mean ± std）

| Metric | LogReg | k-NN |
|---|---|---|
| **Realistic scratch_recall** | **87.7% ± 4.2%** | 24.5% ± 11.1% |
| **Realistic true_ng_recall** | 96.7% ± 3.8% | 99.8% ± 0.4% |
| Oracle scratch_recall | 66.7% ± 19.1% | 45.8% ± 23.3% |
| PR-AUC | 0.881 ± 0.078 | 0.721 ± 0.082 |

### Per-Prefix Breakdown（LogReg realistic threshold）

| 光源 prefix | scratch_recall | n_scratch |
|---|---|---|
| G0F00000 | 93.0% | 57 |
| R0F00000 | 93.0% | 43 |
| STANDARD | 66.7% | 3 |
| W0F00000 | 33.3% | 6 |
| WGF50500 | 60.0% | 5 |

**主要結論：** G0F + R0F 包 88% scratch 樣本、recall 各 93%。其他光源樣本太少（共 14 筆），數字不可靠。

### Verdict 解讀

| 門檻策略 | scratch 擋下 | 真 NG 漏檢 | 適用場景 |
|---|---|---|---|
| **Realistic**（部署實況：threshold = train max true_ng score） | **87.7%** | **3.3%** 會被誤翻 OK | 有人工覆判 / 可接受少量漏檢 |
| **Oracle**（理想 threshold：保證 100% true_ng） | 66.7% | 0% | AOI 零容忍漏檢 |

原 spec 設「100% true_ng recall」為硬門檻 → **嚴格講是 Oracle 的 66.7% 才算通過**。Realistic 的 87.7% 雖高但伴隨 3.3% 漏檢，需保守評估。

差距主因：scratch 樣本少（114 筆），5-fold 切下去每 fold 只有 ~92 筆 train scratch，train 看到的 "true_ng max score" 不夠有代表性 → test 裡常出現更高分的 true_ng 被誤翻。

---

## 3. 產出的檔案

### Git 已提交（commits on `main`）

| SHA | 內容 |
|---|---|
| `e9a9109` | Task 1: skeleton + gitignore |
| `f69e7ac` | Task 2: `dataset.py` + 4 tests |
| `1fa8688` | Task 3: `splits.py` + 2 tests |
| `5f42a7a` | Task 4: `features.py`（DINOv2 + cache） |
| `3e1e20e` | Task 5: `evaluate.py` + 3 tests |
| `3f2b635` | Task 6: `report.py` |
| `851b001` | Task 7: `run_poc.py` CLI |
| `ad71686` | Task 8: `prepare_offline_model.py` |

### 本地檔案（gitignored，會在後續實驗中被覆寫）

```
reports/over_review_scratch_poc/
  embeddings_cache.npz         1748 crops 的 DINOv2 768-d 特徵（重跑秒級）
  report.md                    人看的摘要
  report.json                  機器可讀 metrics
  missed_scratch.csv           14 筆被漏掉的 scratch 樣本
  fold_{1..5}_umap.png         UMAP 2D 視覺化
  fold_{1..5}_pr.png           PR curve

~/.cache/torch/hub/checkpoints/
  dinov2_vitb14_pretrain.pth   330 MB DINOv2 主幹
```

### ⚠️ 目前沒有訓練完的「可部署」模型檔

POC 只做可行性驗證，**沒有存 LogReg .pkl / k-NN .pkl**。5-fold CV 每次跑都重新訓練，跑完就丟。要部署得另外寫「用全 1748 筆訓練單一 LogReg 存 .pkl」的 step（見 Todo C）。

---

## 4. 重跑 POC 的指令

```bash
# 完整 POC（cache 已建 → 第二次約 10 秒完成）
python -m scripts.over_review_poc.run_poc \
    --manifest datasets/over_review/manifest.csv \
    --output reports/over_review_scratch_poc

# 指定 batch size（小 GPU 可降）
python -m scripts.over_review_poc.run_poc --manifest ... --output ... --batch-size 16

# 指定不同 seed / k
python -m scripts.over_review_poc.run_poc --manifest ... --output ... --seed 123 --k 10

# 離線部署前置（在有外網機器執行，印 cache 路徑）
python -m scripts.over_review_poc.prepare_offline_model
python -m scripts.over_review_poc.prepare_offline_model --export-state-dict /tmp/dinov2_vitb14.pth

# 跑單元測試（9 個 test）
python -m pytest tests/test_over_review_poc.py -v
```

---

## 5. 待辦清單（依建議順序）

### A. 結果分析 — 快，30 分鐘內回答
- [ ] A1. 打開 `reports/over_review_scratch_poc/missed_scratch.csv` 的 14 筆（`crop_path` 欄），肉眼看 scratch 被漏掉的原因（光源？cut-off 位置？形狀？）
- [ ] A2. 看 `fold_{1..5}_umap.png`：scratch（紅）是否跟 true_ng（藍）在 embedding 空間明顯分群？若完全混在一起 = 警訊，這 87.7% 可能是 overfit；若清楚分離 = 可信
- [ ] A3. 看 `fold_{1..5}_pr.png`：LogReg 的 PR curve 是否平滑/高位，還是陡降？
- [ ] A4. 檢查 `report.json` 的 `per_fold` 陣列，看哪些 fold 表現最差

### B. 降低漏檢風險（決定能否上線）
- [ ] B1. **Threshold margin 掃描**：寫個 helper script，threshold = `train_max_true_ng × margin`，margin 從 1.00、1.05、1.10、1.15、1.20 各跑一次，看 scratch_recall 和 true_ng_recall 的 trade-off 曲線。找到「true_ng_recall ≥ 99.5%」對應的 margin
- [ ] B2. 分析 3.3% 被誤翻的 true_ng：這些分數為什麼這麼高？是不是同時也長得像 scratch？（human ambiguous label）
- [ ] B3. 考慮加入 Platt scaling / isotonic calibration 讓 LogReg 輸出機率更校準
- [ ] B4. 考慮兩段式 threshold：高 confidence 直接翻 OK、中 confidence 標「待人工覆判」、低 confidence 保留 NG

### C. 產出部署用模型
- [ ] C1. 新增 `scripts/over_review_poc/train_final_model.py`：用全 1748 筆資料訓練單一 LogReg → pickle 成 `deployment/scratch_classifier.pkl`（含 threshold、metadata）
- [ ] C2. 決定最終 threshold 策略（B 的結論直接套用）
- [ ] C3. 寫 inference helper：`predict_scratch(crop_image) -> (score, is_scratch)`

### D. 整合到 CAPI pipeline
- [ ] D1. 另開 deployment spec 討論：插入點、GPU 共享、inference latency budget、是否要離線 model
- [ ] D2. 參考 `capi_inference.py`，在 PatchCore 判 NG 之後接 scratch 分類器
- [ ] D3. 加 runtime config 讓這個 post-filter 可以開關（`config_params` 機制）
- [ ] D4. 寫整合測試：一張真實 NG + 一張 scratch 都送進來，驗證兩者的最終 judgment

### E. 擴充其他過檢類別（重用同個 pipeline）
- [ ] E1. `over_overexposure`（306 筆，最大宗）— 改 `dataset.py` 的 label mapping 即可重跑
- [ ] E2. `over_within_spec`（208 筆）
- [ ] E3. `over_edge_false_positive`（132 筆）
- [ ] E4. 討論最終是 N 個 one-vs-all 分類器 vs 多標籤模型

---

## 6. 開放問題

- Q1. 目前 POC 的 Realistic vs Oracle 落差是因為 scratch 樣本數（114）不足 → 要不要蒐集更多 scratch 樣本讓門檻穩定？還是直接接受 margin 策略？
- Q2. W0F / WGF / STANDARD 光源 recall 偏低但樣本少，是雜訊還是結構性問題？（建議等資料更多再討論）
- Q3. 部署時每張 crop 都要過 DINOv2（ViT-B/14 推論 ~50ms GPU、~500ms CPU），對 AOI pipeline 的 latency 可接受嗎？
- Q4. 離線部署產線 Linux 無外網，是要 copy `torch.hub` cache 過去，還是用 `prepare_offline_model.py` 產出 .pth 檔 + 改用 `--checkpoint`？

---

## 7. 開新對話接續的方式

告訴下一個對話：
> 「接續 over-review scratch POC，先讀 `docs/over_review_scratch_poc_result.md` 這份交接文件，然後做 Todo [A1 / B1 / C1 / …]」

或直接指定一個 Todo 項目請它執行。所有 spec/plan/commit hash/檔案路徑都記錄在上面。

---

## 8. 快速 glossary（複習用）

| 詞 | 意思 |
|---|---|
| **scratch** | `over_surface_scratch`，使用者標注的「表面刮痕」類過檢 |
| **true_ng** | RIC 覆判確認的真正 NG（不可翻 OK） |
| **other_over** | scratch 以外的 8 類過檢，POC 中併入 `not_scratch` 負類 |
| **Realistic threshold** | 只用 train 資料決定的門檻，部署時能做到的真實表現 |
| **Oracle threshold** | 偷看 test 答案決定的門檻，理想狀況的能力上限 |
| **Group k-fold** | 按 `glass_id` 分群的交叉驗證，同玻璃的 tiles 不會同時出現在 train/test |
| **DINOv2 CLS token** | 768 維向量，代表整張 crop 的語意特徵 |
| **scratch_recall** | 被正確識別為 scratch 的比例（擋下過檢的比例） |
| **true_ng_recall** | 真 NG 沒被誤翻為 OK 的比例（1 - 漏檢率） |
