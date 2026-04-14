# Over-Review Reduction POC: Surface Scratch Classifier

**Date:** 2026-04-14
**Status:** Design
**Related:** [2026-04-10 Over-Review Dataset Collector](2026-04-10-over-review-dataset-collector-design.md)

## 1. 目的與成功定義

### 1.1 Context

`capi_dataset_export.py` 已從 RIC 回填結果蒐集出 `datasets/over_review/` 資料集（manifest.csv + 分類 crop），目前樣本分布：

| Label | Crops | 中文 |
|---|---:|---|
| `true_ng` | 966 | 真實 NG |
| `over_overexposure` | 306 | 曝光過度 |
| `over_within_spec` | 208 | 規格內 |
| `over_edge_false_positive` | 132 | 邊緣誤判 |
| `over_surface_scratch` | 114 | 表面刮痕 |
| `over_bubble` | 13 | 氣泡 |
| `over_dust_mask_incomplete` | 8 | 灰塵屏蔽需完善 |
| `over_other` | 1 | 其他 |

### 1.2 POC 目標

驗證：以 **DINOv2 ViT-B/14 凍結特徵 + 線性分類器**，能否把 `over_surface_scratch` (114) 從 `true_ng + 其他 over` (1634) 中分開，且在「不漏任何 `true_ng`」的 threshold 下擋下有意義比例的 scratch 過檢。

### 1.3 Task Formulation

**二元分類（one-vs-all）**：
- 正類（positive）：`over_surface_scratch` (114)
- 負類（negative）：`true_ng` (966) + 其他所有 over 類別 (668) = 1634

其他 over 類別之所以併入負類而非獨立類別，理由有二：(a) 樣本量過小（`other`=1, `dust_mask_incomplete`=8, `bubble`=13），無法當獨立類別訓練；(b) POC 目標是 scratch 單類的分離度驗證，其他類別留待未來逐步擴充。

### 1.4 成功判準（硬門檻）

在 5-fold CV 的 held-out test set 上，以 **Realistic threshold** 評估：

| 結果區間 | 判定 | 下一步 |
|---|---|---|
| scratch_recall ≥ 50% @ true_ng_recall ~100% | **GO** | 另開 spec：部署整合 + 啟動 `over_overexposure` (306) POC |
| 30% ≤ scratch_recall < 50% | **部分可行** | 試 ViT-L/14 / TTA / patch mean pool；仍 < 50% → 討論是否接受低降檢率上線 |
| scratch_recall < 30% | **NO-GO** | 回頭討論：純 `true_ng` vs `scratch` 不混其他 over、CV 規則路線、多類 formulation |

輔助觀察（不當判準）：UMAP 可視化分離度、k-NN baseline、per-prefix / per-source_type breakdown。

## 2. 架構與元件

### 2.1 檔案配置

```
scripts/over_review_poc/
  __init__.py
  dataset.py                     # 讀 manifest、過濾、產 Sample records
  features.py                    # DINOv2 載入 + batch 抽特徵 + 快取
  splits.py                      # Group k-fold by glass_id + stratify
  evaluate.py                    # LogReg + k-NN 訓練、threshold 搜尋、metric 計算
  report.py                      # UMAP 可視化 + markdown/JSON 報告
  run_poc.py                     # CLI entry：一條指令跑完整個 POC
  prepare_offline_model.py       # 離線部署用：預先下載 DINOv2 checkpoint

reports/over_review_scratch_poc/ # 輸出（gitignore）
  embeddings_cache.npz
  fold_{1..5}_umap.png
  fold_{1..5}_pr.png
  report.md
  report.json
  missed_scratch.csv

tests/
  test_over_review_poc.py
```

### 2.2 資料流

```
manifest.csv
  ↓ dataset.load_samples()          → List[Sample] (status=ok 且檔案存在)
  ↓ binary label mapping            → scratch(114) vs not_scratch(1634)
features.get_or_extract(samples)
  ↓ DINOv2 CLS token                → embeddings (N × 768)  [cache: .npz]
splits.group_kfold_stratified(samples, k=5)
  ↓                                 → 5 個 (train_idx, test_idx)
for each fold:
  evaluate.run_fold(emb, samples, train, test)
    ↓ train LogReg + k-NN
    ↓ predict test set
    ↓ compute realistic + oracle thresholds
    ↓ per-prefix / per-source_type breakdown
  → FoldResult
report.aggregate(folds, out_dir)
  ↓ UMAP per fold
  ↓ PR curves per fold
  ↓ markdown + JSON
```

### 2.3 元件介面

| Module | 對外 API | 依賴 |
|---|---|---|
| `dataset.py` | `load_samples(manifest_path: Path) -> list[Sample]` | csv, pathlib |
| `features.py` | `load_dinov2(checkpoint_path=None) -> model`、`get_or_extract(samples, cache_path, checkpoint_path=None) -> np.ndarray` | torch, torchvision, PIL |
| `splits.py` | `group_kfold_stratified(samples, k=5, seed=42) -> list[(train_idx, test_idx)]` | sklearn |
| `evaluate.py` | `run_fold(emb, samples, train_idx, test_idx) -> FoldResult` | sklearn, numpy |
| `report.py` | `aggregate(folds, out_dir)`、`plot_umap(...)`、`plot_pr(...)` | umap-learn, matplotlib |
| `run_poc.py` | CLI orchestration | argparse |
| `prepare_offline_model.py` | CLI：下載 DINOv2 並印出 cache 路徑 | torch.hub |

### 2.4 Sample dataclass

```python
@dataclass(frozen=True)
class Sample:
    sample_id: str
    crop_path: Path        # 絕對路徑
    label: str             # "scratch" | "not_scratch"
    original_label: str    # "over_surface_scratch" / "true_ng" / "over_*"
    glass_id: str
    prefix: str            # 光源 prefix (G0F00000 等)
    source_type: str       # "patchcore_tile" | "edge_defect"
    ai_score: float
    defect_x: int
    defect_y: int
```

### 2.5 關鍵 invariant

`evaluate.find_threshold_at_full_recall(scores, samples)`：
- 正類 = scratch，預測為正 → 部署時翻 NG→OK
- **Realistic threshold**：`max(scores over train_true_ng)`
- **Oracle threshold**：`max(scores over test_true_ng)`
- 任一 threshold 下的 scratch_recall = `#(scratch test samples with score > threshold) / #scratch_in_test`

## 3. 評估協議

### 3.1 兩種 threshold

**Realistic（主指標）**
```
threshold_real = max(P_scratch over TRAIN true_ng samples)
  → 套到 test set
  → 報 test_true_ng_recall（通常接近但不保證 100%）
  → 報 test_scratch_recall
```

**Oracle（能力上限）**
```
threshold_oracle = max(P_scratch over TEST true_ng samples)
  → test_true_ng_recall = 100%（定義下強制）
  → 報 test_scratch_recall
```

兩者差距反映「DINOv2 特徵對 true_ng 分布的泛化程度」。

### 3.2 Aggregation（5-fold）

| 指標 | LogReg | k-NN | 備註 |
|---|---|---|---|
| Realistic scratch_recall | mean ± std | mean ± std | **Go / No-Go 依據** |
| Realistic true_ng_recall | mean ± std | mean ± std | 應 ≥ 99%，跌破要警示 |
| Oracle scratch_recall | mean ± std | mean ± std | 能力上限 |
| PR-AUC | scalar | scalar | sanity check |

### 3.3 Breakdown

- Per-prefix（G0F / R0F / W0F / WGF / STANDARD 等）：每 prefix 的 scratch_recall + 樣本數
- Per-source_type（`patchcore_tile` vs `edge_defect`）：兩來源 recall 差
- 被 miss 的 scratch 樣本列表（sample_id, score, crop_path）→ 寫 `missed_scratch.csv`

### 3.4 輸出產物

```
reports/over_review_scratch_poc/
  report.md              ← 人看的摘要 + 結論
  report.json            ← 機器可讀：aggregate + per_fold + per_breakdown
  fold_{1..5}_umap.png   ← DINOv2 embedding UMAP 2D（scratch 紅、true_ng 藍、other over 灰）
  fold_{1..5}_pr.png     ← PR curve（LogReg + k-NN 疊圖）
  missed_scratch.csv     ← Realistic threshold 下沒擋下的 scratch
  embeddings_cache.npz   ← 特徵快取（加速 rerun）
```

### 3.5 `report.md` 骨架

```markdown
# Over-Review POC: Surface Scratch Classifier
Date: 2026-04-14 | Model: DINOv2 ViT-B/14 | Folds: 5

## Verdict
- Realistic scratch_recall: XX.X% ± Y.Y%   →  【GO / 部分可行 / NO-GO】
- Realistic true_ng_recall: XX.X% ± Y.Y%

## Main Metrics
## Per-Prefix Breakdown
## Per-Source-Type Breakdown
## Missed Scratch Samples
## Appendix: UMAP & PR curves
```

## 4. Split 策略

### 4.1 演算法

`StratifiedGroupKFold` (sklearn)：
- **Group**：`glass_id`（同玻璃的 tiles 必在同 fold）
- **Stratify**：因 sklearn `StratifiedGroupKFold` 只吃單一 `y` 陣列，將複合 key `(label, prefix, source_type)` 編碼成單一字串（如 `"scratch|G0F00000|patchcore_tile"`），再傳入。實務上 bin 數可能過細，若某 bin 樣本數 < k，退到 `(label, source_type)` 兩欄做主 stratify，prefix 改做 fold 後驗證（log fold-prefix 分布）。
- **k = 5**
- **seed = 42**

### 4.2 理由

AOI 常見陷阱：同玻璃有多個 tile 被標 → random split 讓 train/test 共享 glass → 分數虛高。Group k-fold 避免此洩漏，部署時見到的都是新玻璃。

### 4.3 Fallback

若 stratify key 過細導致某 fold test set 缺 scratch → fail fast，訊息建議降 k 或放寬 stratify。

## 5. DINOv2 特徵配置

| 項目 | 選擇 | 備註 |
|---|---|---|
| 變體 | `dinov2_vitb14` (86M params, 768-dim CLS) | 透過 `torch.hub.load('facebookresearch/dinov2', ...)` |
| Feature | CLS token (1×768) | Linear probe 標準做法 |
| Input size | 224×224 | 從 512×512 crop resize |
| Preprocessing | ImageNet mean/std | DINOv2 default |
| Inference mode | `eval() + no_grad()`、batch=32 | OOM 自動降階到 16 / 8 / 4 / CPU |
| 本地 checkpoint 支援 | `features.load_dinov2(checkpoint_path=None)` 可指向本地 `.pth` | 離線部署用 |

## 6. Classifier & Imbalance 處理

| Classifier | 設定 |
|---|---|
| LogReg（主） | `class_weight="balanced"`, `max_iter=2000`, `C=1.0`, `solver="lbfgs"` |
| k-NN（sanity check） | `n_neighbors=7`, `metric="cosine"`（直接使用 sklearn 的 cosine metric） |

LogReg 的 positive-class probability 即為 P(scratch)。k-NN 的 positive-neighbor ratio 作為 score。

## 7. 錯誤處理

| 情境 | 處理 |
|---|---|
| manifest 中 crop 不存在 | `dataset.load_samples` skip + warn，結尾 summary |
| crop 無法解碼 | skip + warn |
| DINOv2 下載失敗（無網路） | fail fast，訊息指向 `~/.cache/torch/hub/` 手動路徑 |
| GPU OOM | batch 32 → 16 → 8 → 4 → CPU，log 每層 fallback |
| Embedding cache 不一致 | cache 內含 manifest SHA-256 + sample_id 列表；不一致自動重算 |
| Stratified group k-fold 某 fold test 缺 scratch | fail fast，建議降 k |
| Realistic threshold 下 test_true_ng_recall < 95% | report.md 頂端紅色警示（非失敗，但要關注） |

## 8. 離線部署考量

### 8.1 POC 階段（本 spec）

開發機有外網 → `torch.hub.load` 自動下載並 cache 到 `~/.cache/torch/hub/`，無痛。

### 8.2 產線 Linux 無外網

部署整合另開 spec，本 spec 只留 code 面 hook：

- `features.load_dinov2(checkpoint_path=None)` 支援本地 `.pth` fallback
- `scripts/over_review_poc/prepare_offline_model.py`：在有外網機器執行，下載 DINOv2、印出 cache 路徑與檔名，供產線 copy

實際部署流程（打包、scp、驗證 checksum）在後續 deployment spec 處理。

## 9. 可重現性

- 全域 seed = 42（numpy / torch / random / sklearn `random_state`）
- `torch.backends.cudnn.deterministic = True`、`benchmark = False`
- DINOv2 `eval() + no_grad()`，無 dropout
- `report.json` `metadata` 區塊：
  - manifest 路徑 + SHA-256
  - git commit hash
  - Python / torch / sklearn / umap 版本
  - DINOv2 model 名稱 + `torch.hub` commit ref
  - 執行時間 + 主機名

同 manifest + 同 commit → bit-exact 可比。

## 10. 測試範圍

`tests/test_over_review_poc.py`：

1. `test_group_kfold_no_leakage` — 每 fold train/test glass_id 無交集
2. `test_group_kfold_stratify_balance` — 每 fold test 包含 ≥1 scratch、≥1 true_ng、兩種 source_type
3. `test_find_threshold_full_recall` — fabricated scores 的邊界情境（全分離 / 完全重疊 / 部分重疊）
4. `test_label_binary_mapping` — `over_surface_scratch` → `scratch`，其他 8 類 → `not_scratch`

不測 DINOv2 inference（慢、靠 pretrained）、不測 UMAP（純視覺化）、不測 report 排版。

## 11. 相依套件

| 套件 | 版本 | 用途 | 狀態 |
|---|---|---|---|
| `torch` | ≥ 2.0 | DINOv2 載入與 inference | 既有 |
| `torchvision` | matches torch | 預處理 transform | 既有 |
| `scikit-learn` | ≥ 1.3 | LogReg, k-NN, StratifiedGroupKFold | 既有 |
| `umap-learn` | ≥ 0.5 | UMAP 2D 視覺化 | **新增** |
| `matplotlib` | ≥ 3.7 | 畫圖 | 既有 |

`run_poc.py` 啟動時檢查 `umap-learn` 缺失 → 報錯提示 `pip install umap-learn`。

## 12. 範圍邊界

### IN

- `scripts/over_review_poc/` 下 7 個模組（6 個核心 + 1 個 offline prep）
- Embedding 快取
- 5-fold StratifiedGroupKFold
- LogReg + k-NN 雙 classifier
- Realistic + Oracle 雙 threshold 評估
- markdown + JSON 報告、UMAP、PR curves、missed_scratch CSV
- 4 個單元測試
- CLI entry

### OUT

- 部署整合進 `capi_inference.py` / `capi_server.py`（POC 通過後另開 spec）
- 其他 over 類別（overexposure / within_spec / edge_false_positive...）
- DINOv2 fine-tune
- Data augmentation / TTA
- 主動學習、標註 UI
- 多類別 / 階層分類器
- 離線部署的完整 packaging / 驗證流程

## 13. 時程估算

| 階段 | 估時 |
|---|---:|
| dataset / splits / run_poc | 2 hr |
| features（DINOv2 整合 + 快取） | 2 hr |
| evaluate（threshold + metrics） | 1.5 hr |
| report（md + UMAP + PR） | 2 hr |
| tests | 1 hr |
| end-to-end debug | 1.5 hr |
| **POC 首輪交付** | **~1 工作日（10 hr）** |

下一個類別擴充（例如 overexposure）：~3 hr（改 label mapping + rerun）。

## 14. 開放問題

- POC code 是否生產化：先留 `scripts/` 當一次性 POC，GO 之後再決定是否抽為 `capi_over_review.py`。
- DINOv2 model 檔案的產線 packaging 流程（checksum、部署驗證）→ deployment spec 處理。
