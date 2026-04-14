# Over-Review Scratch — 資料維護與模型持續更新策略

**Date:** 2026-04-14
**Status:** 部署規劃階段產出，上線 day 1 就要啟用
**Scope:** 過檢樣本累積後，如何維護資料、觸發重訓、確保模型不退化

---

## 核心原則

1. **觸發式重訓**，不是一有資料就動模型
2. **Hold-out test set 上線前就凍結**，沒這個永遠不知道新 model 有沒有退化
3. **版本管理與 metadata 從 day 1 就要有**，上線後再補會是災難
4. **自動化不等於全自動部署**，最後上線一步永遠需要人工按鈕

---

## 1. 重訓觸發機制

### 1.1 資料生命週期

```
AOI 判 NG → CAPI model 推論 → 保留 → 人工覆判 → RIC 進 DB
                                          │
                                          ▼
                  標 over_surface_scratch / over_overexposure / true_ng / ...
                                          │
                                          ▼
        capi_dataset_export.py 定期匯出 → datasets/over_review/manifest.csv
                                          │
                                          ▼
                               觸發決策：要不要重訓？
```

### 1.2 重訓觸發條件（任一滿足即可）

| 觸發 | 閾值 | 理由 |
|---|---|---|
| 新 scratch 樣本數 | **+30 筆以上** | 目前 114 筆，+30 ≈ +26% 資料量，值得重訓 |
| 距上次訓練時間 | **滿 30 天** | 即使樣本少也要定期確認模型沒退化 |
| 稀缺光源 scratch 新增 | W0F/WGF/STANDARD 任一 **+10 筆** | 這三類目前各 < 10 筆，每一筆都是結構性補強 |
| 人工覆判發現 model 誤判 | **週誤判 > 5 筆** | 代表生產資料 drift 或 label 混淆 |
| 新增重大類別 | 新 prefix 光源上線 / 新產品 | 需立即重訓不等週期 |

### 1.3 每次重訓必做的三道驗證 gate

**新模型絕對不能直接上線**，必須依序過三個 gate，任一失敗就回滾：

#### Gate 1 — 5-fold CV Oracle 不退化
```bash
python -m scripts.over_review_poc.finetune_lora \
    --manifest datasets/over_review/manifest.csv \
    --transform clahe --clahe-clip 4.0 --rank 16 --n-lora-blocks 2 --epochs 15 \
    --output reports/candidate_model_YYYYMMDD
```
**要求**：`Oracle scratch_recall ≥ 當前生產模型 − 2%`

#### Gate 2 — Hold-out test set 通過
```bash
python -m scripts.over_review_poc.eval_on_holdout \
    --model deployment/candidate.pkl \
    --holdout datasets/over_review/holdout_frozen_2026-04-14.csv
```
**要求**：`scratch_recall` 不退 **且** `true_ng_recall` 不退

#### Gate 3 — Shadow mode 在 staging 跑 3-7 天
新舊 model 並排推論產線流量，比對差異。
**要求**：
- 新 model scratch 判定量 vs 舊 model 差異 < 20%
- 無 true_ng 被誤翻 OK（人工抽查確認）
- P95 latency 增加 < 10%

---

## 2. 資料品質與 Drift 偵測

### 2.1 Hold-out test set（永久凍結）— **上線前就要建**

從現有 1748 筆隨機抽 ~300 筆（按 glass 分群保留比例），**永不用於訓練**，只做驗證。

```
datasets/over_review/
  manifest.csv                          # 訓練用，持續增長
  holdout_manifest.csv                  # 凍結 300 筆，永不改
  holdout_frozen_2026-04-14.csv         # 日期 snapshot，永不覆蓋
```

**凍結原則**：
- 按 `glass_id` 分群取樣（跟 group k-fold 一致）
- 保留各 prefix 光源比例
- 保留 scratch / true_ng / other_over 比例
- 寫入 SHA256 避免被改動

### 2.2 Drift monitoring

`capi_inference.py` 推論時，把 LoRA scratch classifier 分數**記進 DB**（擴充 `tile_results` 加 `scratch_score` 欄位）。每週跑腳本：

```python
# scripts/over_review_poc/drift_monitor.py
# 比對本週 vs 歷史 scratch 分數分布的 KL divergence
# KL > threshold → 產線資料 drift → 觸發重訓 + 人工檢查
```

**要監控的 drift 指標**：
- 每 prefix（G0F / R0F / W0F / WGF / STANDARD）的**平均 scratch 分數**
- 判為 scratch 的**比例**（若忽然從 5% 跳到 15% → 異常）
- 高信心 / 中信心 / 低信心**三檔比例**（兩階段系統的人工覆判量）
- Scratch / true_ng / other_over 比例變化

**警報條件**：
- 任一 prefix 平均分數週變化 > 15%
- 人工覆判量週變化 > 50%
- 連續 3 週呈單向 drift

### 2.3 Active learning — 不是所有樣本一樣重要

蒐集 114 筆有 114 筆成本，但貢獻度不同。**優先蒐集三類**：

| 樣本類型 | 為什麼重要 | 效率 |
|---|---|---|
| Model 信心中間地帶（score 0.4-0.7） | 直接教 model 邊界在哪 | 比隨機 3-5 倍 |
| W0F / WGF / STANDARD 光源任何 scratch | 結構性稀缺，每筆都珍貴 | 高 |
| Edge_defect 的 scratch（5/114） | 特殊形態（貼 panel 邊緣），可能需單獨子模型 | 中 |

**實作方向**：
- 人工覆判介面加「貢獻度排序」
- 優先顯示中信心樣本，讓人工先看這些
- 顯示各 prefix 現有樣本數，讓覆判員知道哪類優先標

### 2.4 標註品質控制（多覆判員）

**問題**：不同覆判員對「輕微刮痕」容忍度不一 → 標註不一致

**解法**：
- **雙重覆判**：每筆 borderline scratch 至少 2 人看，不一致時送第三人仲裁
- **定期黃金標準校驗**：每月抽 50 筆全員盲測，計算 inter-rater agreement（Cohen's kappa）
  - kappa ≥ 0.7 → 標註規則一致，OK
  - kappa < 0.7 → **停止接受新標註**，開會對齊規則再重開
- **標註規則文件化**：寫 `docs/scratch_labeling_guide.md` 含明確 edge case 決策樹

---

## 3. 長期持續學習基礎設施

### 3.1 模型版本管理

```
deployment/
  scratch_classifier_v1.0_20260514.pkl   ← 初版
  scratch_classifier_v1.1_20260614.pkl   ← +20 筆 W0F
  scratch_classifier_v1.2_20260714.pkl   ← 定期更新
  scratch_classifier_v1.3_20260814.pkl   ← ...
  active.pkl -> scratch_classifier_v1.3_...  ← symlink，當前上線版本
  rollback.pkl -> scratch_classifier_v1.2_...  ← 上一版，隨時可回
```

每個 pickle metadata 必含：
- **訓練資料 SHA256**（資料不同 → 不同 model，完全可重現）
- **訓練 commit hash**（對應的 code 版本）
- **Hold-out 驗證分數**
- **Shadow mode 測試報告**
- **訓練參數**（rank、epochs、clahe_clip 等）
- **Conformal threshold**
- **訓練日期 + 訓練人員**

### 3.2 全量 vs 增量訓練

| 方式 | 優點 | 缺點 | 用不用 |
|---|---|---|---|
| **全量重訓** | 乾淨可重現、無 catastrophic forgetting | 每次 10-25 分鐘 | **每次都用** |
| 增量 fine-tune | 快 | 舊樣本會被忘、版本追蹤困難 | 不用 |

**理由**：114 → 1748 總樣本完全沒到需要增量訓練的等級。訓練 25 分鐘是小錢，**無腦全量重訓**換可重現性是划算的。

### 3.3 自動化 pipeline（長期目標）

**最終目標**：每週一凌晨自動跑
```
1. capi_dataset_export.py     # 從 DB 匯出最新 manifest
2. data_quality_check.py      # 驗證標註一致性、抓異常標
3. 若新樣本數達觸發門檻：
   4. finetune_lora.py         # 重訓候選模型
   5. eval_on_holdout.py       # Gate 1 + 2
   6. shadow_mode.py           # Gate 3 啟動，跑 3-7 天
7. 通過 → 產生 deploy 候選 → Slack 通知 → 等人工按鈕確認上線
8. 不通過 → Slack alert 給 AI team 排查
```

**關鍵原則**：
- ✅ 自動化資料準備、自動化訓練、自動化驗證
- ❌ **不要**自動上線 — 最後一步永遠人工 confirm，避免靜默退化

---

## 4. 時間軸建議

### 🟢 現階段（POC 剛結束，準備部署）

**必做**：
1. 把 Route A 部署做起來（參考 `docs/over_review_scratch_deployment_handoff.md`）
2. **上線前**建立：
   - Hold-out test set（凍結 300 筆，SHA256 寫入檔名）
   - `scratch_classifier_vX.Y.pkl` 命名規範 + metadata 格式
   - DB schema 加 `scratch_score` 欄位（為 drift monitoring 準備）
3. 寫 `scripts/over_review_poc/eval_on_holdout.py`

### 🟡 上線 1-3 個月內

**每月做**：
- 檢視新累積樣本數 + 生產 drift metrics
- 達觸發門檻重訓一次，**走完整三個 gate**，驗證流程是否跑得動
- 關注 W0F / WGF / STANDARD recall 改善情形

**建置**：
- 人工覆判介面加 active learning 優先排序
- `drift_monitor.py` 週報腳本

### 🔴 上線 6 個月後

- 評估自動化 pipeline 投資（人工每週重訓夠不夠）
- 考慮擴充到 over_overexposure（306 筆）/ over_within_spec（208 筆）等其他類
- 評估是否需要 LoRA rank 再往上（r=32）或更多 training blocks

---

## 5. 常見陷阱（避雷）

| 陷阱 | 後果 | 怎麼避 |
|---|---|---|
| 沒有 hold-out test set | 無法知道新 model 有沒有退 | **上線前就凍結** |
| 覆判員標註漂移 | 訓練資料品質劣化 | 每月 kappa 校驗 |
| Preprocessing 不一致 | LoRA 特徵失效，效果歸零 | 整合 test 檢查 feature 數值 |
| 全自動上線 | 模型靜默退化無人發現 | 最後一步人工 confirm |
| 每次有新資料就重訓 | 無謂消耗、版本混亂 | 觸發式 + schedule |
| 忘記把 scratch_score 記 DB | 無法做 drift monitoring | schema day 1 就要加 |
| 訓練 / 推論 CLAHE 參數漂移 | 特徵偏移 model 劣化 | metadata 綁定 clahe_clip |

---

## 6. 相關文件

- **部署規劃**：`docs/over_review_scratch_deployment_handoff.md`
- **POC 實驗歷史**：`docs/over_review_scratch_poc_result.md`
- **訓練腳本**：`scripts/over_review_poc/finetune_lora.py`
- **Conformal threshold**：`scripts/over_review_poc/zero_leak_analysis.py`
