# CAPI 面板檢測邏輯流程圖

此流程圖展示了 CAPI 系統從讀取圖片到判定 OK/NG 的完整檢測邏輯。包含最新的「表面膜灰塵/刮痕過濾」機制。

```mermaid
graph TD
    %% 樣式設定
    classDef process fill:#eef2ff,stroke:#6366f1,stroke-width:2px;
    classDef decision fill:#fefce8,stroke:#f59e0b,stroke-width:2px;
    classDef startend fill:#ecfccb,stroke:#84cc16,stroke-width:2px;
    classDef data fill:#f0fdf4,stroke:#10b981,stroke-width:2px;
    classDef fail fill:#fee2e2,stroke:#ef4444,stroke-width:2px;

    Start([開始推論]):::startend --> ConfigLoad["載入配置 & 模型"]:::process
    ConfigLoad --> Iterate["遍歷資料夾圖片"]:::process
    Iterate --> CheckSkip{"跳過檔案?"}:::decision
    CheckSkip -- Yes --> LogSkip[/"記錄跳過"/]:::data --> Iterate
    CheckSkip -- No --> Preprocess["預處理"]:::process
    
    subgraph Preprocessing ["預處理階段"]
        Preprocess --> Otsu["Otsu 計算前景邊界"]:::process
        Otsu --> BottomCrop{"底部裁切?"}:::decision
        BottomCrop -- Yes --> DoCrop["執行底部裁切"]:::process
        BottomCrop -- No --> Exclusion["計算排除區域"]:::process
        DoCrop --> Exclusion
        Exclusion --> FindMark["Template Match: MARK"]:::process
        Exclusion --> FindMech["Relative: 機構位置"]:::process
        FindMark --> Tile["影像切塊 (512x512)"]:::process
        FindMech --> Tile
        Tile --> ApplyMask["套用排除區域遮罩"]:::process
    end

    Preprocessing --> Inference["模型推論 (PatchCore)"]:::process
    
    subgraph InferenceLogic ["推論與後處理"]
        Inference --> TileScore{"分數 > 閾值?"}:::decision
        TileScore -- No --> TileOK["Tile OK"]:::data
        TileScore -- Yes --> AnomalyCheck{"檢查異常特徵"}:::decision
        
        AnomalyCheck --> DustCheck{"疑似灰塵/刮痕?"}:::decision
        DustCheck -- "Yes (亮點/Feature Check)" --> MarkDust["標記: Suspected Dust"]:::data
        DustCheck -- No --> MarkReal["標記: Real Defect"]:::fail
        
        MarkDust --> TileResult
        MarkReal --> TileResult
        TileOK --> TileResult
    end

    TileResult --> PanelDecision{"面板判定"}:::decision
    
    subgraph FinalDecision ["最終判定"]
        PanelDecision --> CheckReal{"有真實缺陷?"}:::decision
        CheckReal -- Yes --> FinalNG(["🔴 NG (真實異常)"]):::fail
        CheckReal -- No --> CheckDust{"有灰塵?"}:::decision
        CheckDust -- Yes --> FinalOKDust(["🟢 OK (表面膜灰塵)"]):::startend
        CheckDust -- No --> FinalOK(["🟢 OK (Pass)"]):::startend
    end

    FinalNG --> Visualization["生成視覺化結果"]:::process
    FinalOKDust --> Visualization
    FinalOK --> Visualization
    
    Visualization --> Report["更新 HTML/CSV/TXT 報告"]:::process
    Report --> Iterate
    Iterate -- 結束 --> End([流程結束]):::startend
```

## 邏輯說明

1. **檔案過濾**：檢查檔名是否在配置的 `skip_files` 清單中。
2. **預處理**：
   - 使用 Otsu 二值化自動裁切前景。
   - 根據設定執行底部裁切 (Bottom Crop)。
   - 識別排除區域 (MARK 二維碼, 機構位置)。
   - 將影像切分為 512x512 的 tiles (stride 512, 非重疊)。
   - 對重疊排除區域的 tiles 建立遮罩 (Mask)。
3. **推論**：
   - PatchCore 模型計算每個 tile 的異常分數和熱圖。
   - 若有遮罩，排除區域內的異常分數會被忽略。
4. **異常特徵檢查**：
   - 若分數超過閾值，檢查是否為「疑似灰塵/刮痕」(高亮度點特徵)。
   - 若檢查通過，標記為 `Suspected Dust`。
5. **面板最終判定**：
   - **🔴 NG**：只要有任何一個 tile 是真實缺陷 (Real Defect)。
   - **🟢 OK (表面膜)**：所有異常 tiles 都是 `Suspected Dust` (視為表面膜問題，忽略)。
   - **🟢 OK**：沒有任何異常 tiles。
