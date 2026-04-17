---
title: Scratch 分類器批次驗證頁 UI 優化
date: 2026-04-17
status: draft
---

# Scratch 分類器批次驗證頁 UI 優化

## 問題

`/debug/scratch-batch` 頁面（`templates/debug_scratch_batch.html`）是 Scratch 分類器批次驗證的主力畫面。目前是為工程師設計，實務上工程師會 demo 給主管或非工程師看模型表現，但頁面對非工程師有三個痛點：

1. **術語太多看不懂**：Recall、Leak Rate、TP/FN/FP/TN、threshold、conformal、陽性/陰性 等專有名詞完全沒有說明
2. **缺少一句話總結**：6 張統計卡排在一起同樣大小，沒有明確的「重點數字」引導視線；demo 時要一個個解釋
3. **數字格式不統一**：`conformal = 0.943066`（6 位小數）太雜、計數 `1740` 沒千分位、`Leak = 0.00%` 沒對齊

目標：在**不動任何後端邏輯**的前提下，把 template 改成「工程師主要看、非工程師 demo 時也能秒懂」。

## 非目標

- 不改 `capi_scratch_batch.py` 背景任務邏輯、不改 `capi_web.py` API routes、不改 SQLite schema
- 不做「通過／不通過」自動判定（避免誤訂門檻，使用者明確要求不做）
- 不改 Recall / Leak Rate 計算公式
- 不引入新前端框架（保持純 template + vanilla JS）

## 讀者定位

主要讀者：**工程師**（保留所有技術術語）
次要讀者：**demo 時的非工程師主管**（加上可選的 ⓘ tooltip 白話說明）

所有技術術語（Recall、Leak Rate、TP/FN/FP/TN、conformal、safety）**保留原文**，不替換為中文直譯；只在旁邊加 `ⓘ` 圖示，hover 顯示白話氣泡。

## 設計

### 1. 版面結構

```
┌─────────────────────────────────────────────────────┐
│ 批次 Scratch 分類器驗證 (DINOv2 over-review)        │
├─────────────────────────────────────────────────────┤
│ ▶ 關於這個分類器  （<details> 摺疊，預設收起）     │ ← 新增
├─────────────────────────────────────────────────────┤
│ 開場段（改寫）：陽性/陰性加 ⓘ                     │
├─────────────────────────────────────────────────────┤
│ 工具列：Job 下拉、開始按鈕、最近任務              │
├─────────────────────────────────────────────────────┤
│ Threshold 面板（排版重構）                          │
│  [slider ────────────●──]     threshold = 0.9431    │
│  生效門檻 0.9431 ＝ 校準值 0.9431 ⓘ × 倍率 1.00 ⓘ │
├─────────────────────────────────────────────────────┤
│ HERO 列（2 張大卡）                                  │ ← 新
│ ┌──────────────────┐ ┌──────────────────┐          │
│ │ RECALL ⓘ          │ │ LEAK RATE ⓘ      │          │
│ │ 99.2%            │ │ 0.00%            │          │
│ │ 125 張中 124 張  │ │ 1,615 張中 0 張  │          │
│ │ 成功翻回 OK      │ │ 被誤翻為 OK      │          │
│ └──────────────────┘ └──────────────────┘          │
├─────────────────────────────────────────────────────┤
│ 次要 4 張小卡（縮小）                                 │
│  [樣本 1,740] [跳過 0] [陽性 125 ⓘ] [陰性 1,615 ⓘ]│
├─────────────────────────────────────────────────────┤
│ 混淆矩陣（主數字放大，TP/FN/FP/TN 變灰字 + ⓘ）     │
├─────────────────────────────────────────────────────┤
│ 分數分佈 ⓘ        │  Leak 清單 ⓘ / 漏翻清單 ⓘ   │
└─────────────────────────────────────────────────────┘
```

### 2. 「關於這個分類器」摺疊區塊

放在頁首標題下方，`<details>` 元素，預設收起。展開內容三段：

**DINOv2 — 通用「看圖腦袋」**
DINOv2 是 Meta 開源、預先用上億張網路圖片訓練好的視覺模型。就像一個「已經會看圖的腦袋」，內建對紋理、形狀、表面特徵的理解，不限制看過哪種產品。

**LoRA — 輕量微調補丁**
LoRA（Low-Rank Adaptation）是一種只訓練「少量補丁」的微調技術。不動 DINOv2 主體，只在旁邊接上幾層小網路，用我們手上的 scratch 樣本學「什麼才是真 scratch」。訓練快、資料需求少、不會破壞 DINOv2 原本的視覺能力。

**為什麼這個組合？**
AOI 過檢的 scratch 和其他 NG 長得很像（都是黑點或短線）。DINOv2 給通用視覺能力，LoRA 負責把它特化成「只分辨 scratch vs 其他」。這比從零訓練小模型更準、也比整個重新訓練 DINOv2 更省資源。

**樣式**：淡藍色左邊框（`#60a5fa`）、正常 sans-serif 字體（非 monospace）、`max-width: 720px` 限制避免過寬。

### 3. Tooltip 白話對照

每個術語旁加 `ⓘ`（淡藍色 `#60a5fa`），CSS 純 hover 氣泡（absolute 定位，寬 260px）。

| 術語 | Tooltip 文字 |
|---|---|
| 陽性樣本 | 標註為 over_surface_scratch 的過檢類型。這些原本是 AOI 誤判成 NG 的，希望分類器能把它們翻回 OK。 |
| 陰性樣本 | 真 NG + 其他過檢類型。這些不該被翻回 OK，任何被翻走的都計入 leak。 |
| 匯出批次 (Job) | 從 Over-Review 標註頁匯出的一組樣本。每個 job 是一次獨立的驗證資料集。 |
| Threshold 試算 | score 高於門檻 → 分類器判斷「應該翻回 OK」。拖動可即時重算下方所有指標。 |
| 模型校準值 (conformal) | 訓練時以校準集算出的保守門檻，用來控制 leak 率。 |
| 安全倍率 (safety ×) | 在校準值上再乘的倍率。> 1 更嚴格、< 1 更寬鬆。 |
| Recall | 分類器成功抓出的陽性比例 = TP / (TP + FN)。越高越好，代表該翻的都翻到了。 |
| Leak Rate | 陰性被誤翻的比例 = FP / (FP + TN)。越低越好。0% 代表沒有真 NG 被放行。 |
| TP | True Positive：陽性且分類器判翻 OK。 |
| FN | False Negative：陽性但分類器沒翻，應翻而漏翻。 |
| FP | False Positive：陰性被翻 OK，就是 leak。這是最嚴重的錯誤。 |
| TN | True Negative：陰性且分類器保持 NG。 |
| 分數分佈 | 每張樣本的分類器輸出分數分佈。綠=陽性、紅=陰性。越好的模型，兩色重疊越少。 |
| Leak 清單 | 應保持 NG 但被翻回 OK 的樣本，屬誤放，需檢查並修正。 |
| 漏翻清單 | 應翻回 OK 但保持 NG 的樣本，屬漏抓；對出廠安全無害但會增加人工複檢成本。 |

### 4. Hero 雙主卡

2 張大卡，CSS Grid 兩欄，gap 14px：

**左 · Recall（綠）**
- 淡綠漸層背景 `linear-gradient(135deg, rgba(34,197,94,0.12), rgba(10,15,26,0.4))`
- 綠色邊框 `1px solid rgba(34,197,94,0.35)`
- 標籤 `RECALL ⓘ`（大寫、綠字 `#22c55e`、letter-spacing 1px）
- 主數字 `99.2%`（font-size 2.4rem、Fira Code、綠色粗體）
- 副標白話：`在 N 張 scratch 樣本中，M 張成功翻回 OK`（灰字、0.82rem、line-height 1.5）

**右 · Leak Rate（紅）**
- 紅色漸層 + 紅邊框（同上換色）
- 標籤 `LEAK RATE ⓘ`、主數字 `0.00%`（紅字）
- 副標：`在 N 張真 NG 或其他類型中，M 張被誤翻為 OK`

副標的 `N`、`M` 由 JS 動態帶入（見「實作範圍」中的 `recomputeAll` 片段）。

### 5. 次要統計列

4 張小卡，grid 四欄：`樣本總數 / 跳過 / 陽性 / 陰性`。
樣式沿用現有 `.sb-card` 但縮減 padding、字級（value 從 1.4rem → 1.1rem）。

### 6. 數字格式規範

| 欄位 | 格式 |
|---|---|
| 門檻值（conformal / safety 乘完後的 effective / slider 當前值） | `.toFixed(4)` 例 `0.9431` |
| 安全倍率 | `.toFixed(2)` 例 `1.00` |
| Recall | 1 位小數百分比 `99.2%` |
| Leak Rate | 2 位小數百分比 `0.00%` |
| 計數（樣本/跳過/陽性/陰性/TP/FN/FP/TN） | `.toLocaleString()` 千分位 `1,740`、`1,615` |
| 樣本分數（grid 卡內） | `.toFixed(4)` 不變 |
| 進度剩餘時間 | `< 60s` 顯示 `Xs`；`≥ 60s` 顯示 `Xm Ys` |
| 除以零 | 顯示 `—`（em dash U+2014） |

### 7. 混淆矩陣排版

- 保留 2x2 表格結構
- 主數字放大（現 `.cm-table td.val` 1.2rem → 1.4rem、粗體）
- `TP / FN / FP / TN` 縮為 0.7rem 灰色小字放在數字右側作為「代號」
- 4 個代號各自加 ⓘ tooltip（見 §3 對照表）
- 表格下的一行 legend（`TP = 正確翻 OK | FN = ...`）保留但把代號改灰色，與矩陣內標籤呼應

### 8. 其他區塊 ⓘ

以下標題／圖例旁加 ⓘ：
- `Threshold 試算`
- `匯出批次 (Job)` 下拉 label
- 「分數分佈」panel title
- 「Leak 清單」panel title
- 「漏翻清單」panel title
- histogram 圖例三個色塊（陽性／陰性／threshold）

## 實作範圍

**唯一異動檔案**：`templates/debug_scratch_batch.html`

**HTML**：新增 `<details>` 區塊、重構 hero row 與 secondary row、所有 ⓘ 標記與 tooltip data 屬性

**CSS**（`{% block extra_css %}` 內）：
- `.sb-hero-row` / `.sb-hero-card` / `.sb-hero-label` / `.sb-hero-value` / `.sb-hero-sub`（新）
- `.sb-stat-compact`（次要縮小版）
- `.sb-tip`（ⓘ 圖示）與 `.sb-tip:hover::after`（氣泡）
- `.sb-about-classifier`（摺疊區塊樣式）

**JS**：
- `recomputeAll(thr)`：新增 hero 副標字串產生器
  ```js
  document.getElementById('s-recall-sub').textContent =
      `在 ${pos.toLocaleString()} 張 scratch 樣本中，${tp.toLocaleString()} 張成功翻回 OK`;
  document.getElementById('s-leak-sub').textContent =
      `在 ${neg.toLocaleString()} 張真 NG 或其他類型中，${fp.toLocaleString()} 張被誤翻為 OK`;
  ```
- `formatSeconds(s)` 新工具函式
- 既有 `.toFixed(6)` → `.toFixed(4)`（`thrConformal`、`thrEffective`）
- 既有整數顯示改 `.toLocaleString()`
- 除以零判斷：`pos === 0 ? '—' : (recall*100).toFixed(1) + '%'`

## 驗收

手動驗收：啟動 web，`/debug/scratch-batch`：

1. ✅ 載入 `recent_tasks` 中的既有結果（例 `batch_20260417_082431`），所有數字正確渲染
2. ✅ 拖動 threshold slider → hero 副標（`N 張中 M 張…`）即時更新，TP/FN/FP/TN 同步
3. ✅ 滑過所有 ⓘ → tooltip 氣泡正確顯示、不被上層 `overflow:hidden` 裁掉
4. ✅ 展開「關於這個分類器」→ 文字排版正確、不破壞下方佈局
5. ✅ 跑一個全新 job → 完成後 UI 畫面與載入既有結果一致
6. ✅ 找一個 pos=0 或 neg=0 的場景（手動編輯 cachedResults 或選特殊 job）確認 `—` 顯示
7. ✅ 進度時間 >60s 時顯示 `Xm Ys`

## 回退

單檔異動（`templates/debug_scratch_batch.html`），`git revert` 即可完全復原。無 schema / API / 後端改動。
