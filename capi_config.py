"""
CAPI 配置管理模組

提供 CAPI 推論系統的配置載入、驗證和管理功能。
支援 YAML 配置檔格式，可針對不同機種/產品設定排除區域等參數。

使用方式:
    from capi_config import CAPIConfig
    config = CAPIConfig.from_yaml("configs/capi_3f.yaml")
"""

import yaml
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple


def _default_within_spec_judgment_rules() -> Dict[str, Any]:
    """Default within-spec dot judgment settings, keyed by machine id."""
    dot_detection = {
        "preprocess_method": "gaussian",
        "preprocess_params": {"kernel_size": 7, "sigma": 1.0},
        "segmentation_method": "background_diff",
        "diff_threshold": 4,
        "hysteresis_low_threshold": 2,
        "hysteresis_high_threshold": 4,
        "hysteresis_edge_width_percent": 3.0,
        "hysteresis_edge_extra_threshold": 2,
        "hysteresis_second_low_threshold": 3,
        "hysteresis_second_high_threshold": 4,
        "hysteresis_second_edge_width_percent": 9.5,
        "hysteresis_second_edge_extra_threshold": 2,
        "hysteresis_switch_count_threshold": 5,
        "hysteresis_second_max_count": 5,
        "hysteresis_edge_suppress_percent": 0.0,
        "background_kernel": 33,
        "min_area_px": 2,
        "max_area_px": 50000,
        "morph_open": 0,
        "min_aspect_ratio": 0.45,
        "edge_margin_px": 4,
        "non_dot_residue_enabled": True,
        "non_dot_residue_min_area_px": 500,
        "non_dot_residue_min_long_side_px": 80,
        "non_dot_residue_min_long_side_ratio": 0.15,
        "non_dot_residue_min_max_diff": 12,
        "non_dot_residue_reasons": ["aspect_ratio_below_min", "edge_margin", "area_too_large"],
        "size_metric": "bbox_diagonal",
        "unit_per_px": 0.0245,
    }

    def standard_screen(label: str) -> Dict[str, Any]:
        return {
            "label": label,
            "black_dot": {
                "enabled": True,
                "defect_code": "C1111",
                "area_threshold_mm": 0.3,
                "screen_count_limit": 3,
                "tile_count_threshold": 2,
            },
            "white_dot": {
                "enabled": True,
                "defect_code": "C1111",
                "area_threshold_mm": 0.3,
                "screen_count_limit": 1,
                "tile_count_threshold": 2,
            },
        }

    return {
        "default": {
            "display_name": "預設機種",
            "dot_detection": dot_detection,
            "screens": {
                "STANDARD": standard_screen("STANDARD(標準)"),
                "WGF50500": standard_screen("WGF50500(50灰)"),
                "G0F00000": standard_screen("G0F00000(綠畫面)"),
                "R0F00000": standard_screen("R0F00000(紅畫面)"),
                "W0F00000": standard_screen("W0F00000(白畫面)"),
                "B0F00000": {
                    "label": "B0F00000(黑畫面)",
                    "black_dot": {
                        "enabled": False,
                        "defect_code": "C1111",
                        "area_threshold_mm": 0.2,
                        "screen_count_limit": 1,
                        "tile_count_threshold": 1,
                    },
                    "white_dot": {
                        "enabled": True,
                        "defect_code": "C1111",
                        "area_threshold_mm": 0.2,
                        "screen_count_limit": 1,
                        "tile_count_threshold": 1,
                    },
                },
            },
        }
    }


@dataclass
class ExclusionZone:
    """排除區域定義"""
    name: str
    type: str  # "template_match" 或 "relative_bottom_right"
    description: str = ""
    enabled: bool = True
    width: int = 0
    height: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExclusionZone":
        return cls(
            name=data.get("name", "unknown"),
            type=data.get("type", "unknown"),
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            width=data.get("width", 0),
            height=data.get("height", 0),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "enabled": self.enabled,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class BombDefect:
    """炸彈模擬缺陷定義 (機台模擬假 Defect 用於檢驗檢出能力)"""
    image_prefix: str       # 照片名前綴, e.g. "G0F00000"
    defect_code: str        # Defect Code, e.g. "PCLV6GA0"
    defect_type: str        # "line" (豎線) 或 "point" (點)
    coordinates: List[Tuple[int, int]] = field(default_factory=list)
    # 豎線: 2 點定義起終點 [(x1,y1),(x2,y2)]
    # 點: 多個點座標 [(x1,y1),(x2,y2),...]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BombDefect":
        coords = data.get("coordinates", [])
        # 支援 [[x,y],[x,y]] 格式
        coord_tuples = [tuple(c) for c in coords]
        return cls(
            image_prefix=data.get("image_prefix", ""),
            defect_code=data.get("defect_code", ""),
            defect_type=data.get("defect_type", "point"),
            coordinates=coord_tuples,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_prefix": self.image_prefix,
            "defect_code": self.defect_code,
            "defect_type": self.defect_type,
            "coordinates": [list(c) for c in self.coordinates],
        }


@dataclass
class CAPIConfig:
    """CAPI 推論配置"""
    
    # 機種識別
    machine_id: str = "CAPI_3F"
    product_name: str = "CAPI 面板"
    is_new_architecture: bool = False  # 新架構：model_mapping value 為 nested dict（inner/edge）
    edge_threshold_px: int = 768       # 新架構邊緣區域判定門檻（px）
    
    # MARK 模板設定
    mark_template_path: str = "capi_mark.png"
    mark_fallback_position: Optional[Dict[str, int]] = None  # Fallback MARK 位置 {x, y, width, height}
    mark_match_threshold: float = 0.45
    mark_min_y_ratio: float = 0.6
    mark_exclusion_padding_px: int = 32  # MARK 不檢測區外擴，吸收 PatchCore heatmap 擴散
    mark_exclusion_soft_decay_px: int = 48  # MARK 硬遮罩外圈 heatmap 線性降權寬度
    cv_edge_exclude_soft_decay_px: int = 64  # 手動不檢測區外圈 heatmap 線性降權寬度
    no_detect_soft_decay_min_weight: float = 0.10  # soft decay 邊界最低權重 (0~1)
    
    # Otsu 裁剪設定
    otsu_offset: int = 5
    otsu_bottom_crop: int = 1000  # Otsu 後裁切底部像素

    # 面板 4 角 polygon 偵測（新增）
    enable_panel_polygon: bool = True  # 啟用後在 tile.mask 上套用 polygon 做精準裁切
    image_preprocess_pipeline: List[Dict[str, Any]] = field(default_factory=list)
    preprocess_after_tiling: bool = False  # 先切塊後前處理選項

    # 排除區域
    exclusion_zones: List[ExclusionZone] = field(default_factory=list)
    
    # 切塊設定
    tile_size: int = 512
    tile_stride: int = 512
    
    # 推論設定
    anomaly_threshold: float = 0.5
    model_path: str = ""  # 預設模型路徑 (fallback，當 model_mapping 無對應時使用)
    
    # 多模型映射 {image_prefix: model_path} — 依圖片前綴自動選用對應模型
    # 新架構格式: {image_prefix: {"inner": path, "edge": path}}
    model_mapping: Dict[str, Any] = field(default_factory=dict)

    # 每個前綴的獨立閾值 {image_prefix: threshold}，未指定則使用 anomaly_threshold
    # 新架構格式: {image_prefix: {"inner": float, "edge": float}}
    threshold_mapping: Dict[str, Any] = field(default_factory=dict)
    
    # PatchCore 後處理進階過濾
    patchcore_filter_enabled: bool = False             # 是否啟用後處理過濾
    patchcore_blur_sigma: float = 1.5                # 異常圖高斯平滑強度
    patchcore_min_area: int = 10                     # 判定為真異常的最小連通面積(像素)
    patchcore_score_metric: str = "max"              # 計分方式: "max", "top_k_avg", "percentile_99"

    # 集中度檢查 (Concentration Check) — 過濾 heatmap 均勻偏暖的瀰漫性假陽性
    patchcore_concentration_enabled: bool = True       # 是否啟用集中度檢查
    patchcore_concentration_min_ratio: float = 2.0     # Peak/Mean 最小比值 (低於此值觸發降權)
    patchcore_concentration_penalty: float = 0.5       # 降權因子 (比值=1.0 時的最大懲罰乘數)

    # 擴散面積檢查 (Diffuse Area Check) — 過濾 heatmap 大面積偏暖的梯度/擴散型假陽性
    patchcore_diffuse_area_enabled: bool = True        # 是否啟用擴散面積檢查
    patchcore_diffuse_area_threshold: float = 0.3      # 超過 peak 50% 的像素佔比閾值 (大於此值觸發降權)
    patchcore_diffuse_area_penalty: float = 0.5        # 降權因子 (佔比=100% 時的最大懲罰乘數)

    # 灰塵偵測設定 (OMIT 圖片分析)
    dust_brightness_threshold: int = 80       # OMIT 亮度閾值 (自適應 Otsu 時此為備用上限)
    dust_threshold_floor: int = 25            # Otsu 自適應閾值下限 (防止低噪 OMIT 抓出過多背景噪點)
    dust_bright_rescue_threshold: int = 180   # CLAHE 增強後的高亮直接檢測閾值 (救回被 Top-Hat 吃掉的大面積污染，0=停用)
    dust_area_min: int = 15                   # 灰塵顆粒最小面積 (px)
    dust_area_max: int = 100000               # 灰塵顆粒最大面積 (px)
    dust_extension: int = 5                   # 灰塵區域膨脹像素
    dust_heatmap_iou_threshold: float = 0.02  # Heatmap-Dust IOU/Coverage 閾值
    dust_heatmap_top_percent: float = 5.0     # Heatmap 熱區取前 X% (Percentile 二值化)
    dust_heatmap_metric: str = "coverage"     # Heatmap 判定指標: "coverage" (灰塵覆蓋率) 或是 "iou" (交集/聯集)
    dust_mask_before_binarize: bool = False   # 先遮罩灰塵區域再二值化 (解決灰塵強訊號淹沒弱缺陷的問題)
    dust_two_stage_enabled: bool = False      # 兩階段灰塵判定：heatmap定位→原圖找特徵點→精準比對dust_mask
    dust_two_stage_dust_ratio: float = 0.3    # 特徵點與灰塵重疊比例 >= 此值判為灰塵
    dust_two_stage_bg_blur: int = 31          # 局部背景估計的 Gaussian blur kernel size
    dust_two_stage_diff_percentile: float = 90.0  # 取 diff 分布的此百分位作為特徵閾值
    dust_two_stage_min_area: int = 3          # 特徵最小面積 (px)
    dust_detect_dark_particles: bool = True   # 偵測暗色顆粒/圖案 (如偏黑 MARK)，當作表面灰塵過濾
    dust_detect_bubbles_enabled: bool = False  # 偵測低對比氣泡/大片暗斑，併入 OMIT dust mask
    dust_high_cov_threshold: float = 0.5       # 高覆蓋率門檻：region COV >= 此值時直接判 dust，不要求 peak_in_dust（因 heatmap peak 有膨脹偏移）
    dust_peak_fraction_threshold: float = 0.80 # 次峰救援門檻：peak 不在灰塵上時，若灰塵區內最強分數 >= 區域 peak 的此比例，視為 heatmap 偏移，仍判 dust
    
    # OMIT 過曝偵測設定 (曝光過高的 OMIT 圖無法檢測灰塵，需記錄供工程追蹤)
    omit_overexposure_mean_threshold: int = 200    # 平均亮度超過此值視為過曝
    omit_overexposure_ratio_threshold: float = 0.5 # 高亮像素(>230)佔比超過此值視為過曝

    # 畫異預檢設定 (推論前依 AOI Report 涉及畫面檢查平均亮度，異常時回報 PCO05)
    image_abnormal_detection_enabled: bool = False
    image_abnormal_standard_mean_lower: int = 68
    image_abnormal_standard_mean_upper: int = 88
    image_abnormal_wgf50500_mean_lower: int = 72
    image_abnormal_wgf50500_mean_upper: int = 92
    image_abnormal_g0f00000_mean_lower: int = 67
    image_abnormal_g0f00000_mean_upper: int = 87
    image_abnormal_r0f00000_mean_lower: int = 71
    image_abnormal_r0f00000_mean_upper: int = 91
    image_abnormal_w0f00000_mean_lower: int = 70
    image_abnormal_w0f00000_mean_upper: int = 90
    image_abnormal_b0f00000_mean_lower: int = 0
    image_abnormal_b0f00000_mean_upper: int = 13
    
    # 邊緣衰減設定 (過濾光影假陽性)
    edge_margin_px: int = 0                   # 邊緣衰減寬度 (px)，0=停用
    edge_margin_sides: Dict[str, bool] = field(default_factory=lambda: {
        'top': False, 'bottom': False, 'left': False, 'right': False
    })  # 各邊是否啟用衰減
    
    # 跳過檔案二值化偵測設定 (B0F00000 等無模型圖片)
    bright_spot_threshold: int = 200          # 絕對亮度上限 (超過此值直接判定為亮點)
    bright_spot_min_area: int = 5             # 亮點最小連通面積 (px, 小於此視為雜訊)
    bright_spot_median_kernel: int = 21       # 背景估計 median filter 核大小
    bright_spot_diff_threshold: int = 10      # 局部對比差異閾值 (與背景差值超過此值為異常)
    within_spec_judgment_rules: Dict[str, Any] = field(default_factory=_default_within_spec_judgment_rules)

    # 回報結果設定
    report_black_dot_defect_code: str = "PCDK2"
    report_white_dot_defect_code: str = "PTMD6"
    report_unknown_dot_defect_code: str = "PCDK2"
    report_bomb_defect_code: str = "PCDK3"
    report_image_abnormal_defect_code: str = "PCO05"

    # 跳過檔案設定 (不進行推論的檔案名稱)
    skip_files: List[str] = field(default_factory=list)
    
    # 側拍圖前綴列表 (以這些前綴開頭的檔案自動跳過，不在模型訓練資料中)
    side_shot_prefixes: List[str] = field(default_factory=list)
    
    # 重複投片檢查 (Panel 資料夾內圖片數超過此值則跳過)
    max_images_per_panel: int = 20
    
    # 炸彈系統設定 (機台模擬缺陷)
    bomb_defects: List[BombDefect] = field(default_factory=list)
    bomb_match_tolerance: int = 50  # 座標匹配容忍度 (產品座標系像素)
    bomb_line_min_aspect_ratio: float = 3.0  # Line 型炸彈 heatmap 最小長寬比
    bomb_area_force_detection_enabled: bool = False  # AOI 漏給炸彈座標時，使用 Client 炸彈座標補切 tile 偵測
    
    # 機種第六碼 → 產品解析度映射表 (寬, 高)
    # 例: {'B': [1366, 768], 'H': [1920, 1080], 'J': [1920, 1200], 'K': [2560, 1440], 'G': [2560, 1600]}
    model_resolution_map: Dict[str, List[int]] = field(default_factory=lambda: {
        'B': [1366, 768],
        'H': [1920, 1080],
        'J': [1920, 1200],
        'K': [2560, 1440],
        'G': [2560, 1600],
    })
    
    # Grid Tiling 推論開關
    grid_tiling_enabled: bool = True

    # AOI 機檢座標推論設定
    aoi_coord_inspection_enabled: bool = False
    aoi_heatmap_center_seed_enabled: bool = True  # AOI 座標 tile 在 Top% heatmap 中額外保留 AOI 中心弱熱區
    aoi_report_path_replace_from: str = "yuantu"    # 報告路徑替換來源
    aoi_report_path_replace_to: str = "Report"      # 報告路徑替換目標

    # Scratch classifier post-filter (over-review reduction)
    scratch_classifier_enabled: bool = True
    scratch_safety_multiplier: float = 1.1
    scratch_bundle_path: str = "deployment/scratch_classifier_v1.pkl"
    scratch_dinov2_weights_path: str = "deployment/dinov2_vitb14.pth"
    # Local path to DINOv2 repo code for torch.hub source="local" (offline deploy).
    # When empty (""), falls back to torch.hub GitHub/cache — needs network or
    # pre-warmed ~/.cache/torch/hub.
    scratch_dinov2_repo_path: str = ""

    # 配置檔路徑（載入後記錄）
    config_path: Optional[Path] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "CAPIConfig":
        """從 YAML 檔案載入配置"""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"配置檔不存在: {yaml_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        config = cls.from_dict(data)
        config.config_path = path
        return config

    @staticmethod
    def _normalize_threshold_mapping(raw_threshold_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize legacy flat and v2 nested threshold mappings to numeric values."""
        normalized: Dict[str, Any] = {}
        for prefix, value in (raw_threshold_mapping or {}).items():
            if isinstance(value, dict):
                normalized[prefix] = {
                    zone: float(zone_value)
                    for zone, zone_value in value.items()
                }
            else:
                normalized[prefix] = float(value)
        return normalized

    @staticmethod
    def _normalize_within_spec_judgment_rules(raw_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Merge saved within-spec rules with the current default schema."""
        defaults = _default_within_spec_judgment_rules()
        if not isinstance(raw_rules, dict) or not raw_rules:
            return copy.deepcopy(defaults)

        rules = copy.deepcopy(raw_rules)
        rules.pop("CAPI_3F", None)
        if not rules:
            rules = {"default": copy.deepcopy(defaults["default"])}
        if "default" not in rules:
            rules["default"] = copy.deepcopy(defaults["default"])

        normalized = {}
        for machine_id, machine_raw in rules.items():
            if not isinstance(machine_raw, dict):
                machine_raw = {}

            machine_defaults = copy.deepcopy(defaults["default"])
            if machine_id != "default":
                machine_defaults["display_name"] = machine_id

            machine_cfg = copy.deepcopy(machine_defaults)
            for key, value in machine_raw.items():
                if key not in ("dot_detection", "screens"):
                    machine_cfg[key] = value

            raw_dot_detection = machine_raw.get("dot_detection", {})
            if isinstance(raw_dot_detection, dict):
                for key, value in raw_dot_detection.items():
                    if key != "preprocess_params":
                        machine_cfg["dot_detection"][key] = value
                raw_preprocess = raw_dot_detection.get("preprocess_params", {})
                if isinstance(raw_preprocess, dict):
                    machine_cfg["dot_detection"]["preprocess_params"].update(raw_preprocess)

            raw_screens = machine_raw.get("screens", {})
            if isinstance(raw_screens, dict):
                for screen_code, screen_raw in raw_screens.items():
                    if not isinstance(screen_raw, dict):
                        continue
                    if screen_code not in machine_cfg["screens"]:
                        machine_cfg["screens"][screen_code] = copy.deepcopy(screen_raw)
                    else:
                        if "label" in screen_raw:
                            machine_cfg["screens"][screen_code]["label"] = screen_raw["label"]
                        for dot_type in ("black_dot", "white_dot"):
                            raw_rule = screen_raw.get(dot_type)
                            if not isinstance(raw_rule, dict):
                                continue
                            machine_cfg["screens"][screen_code].setdefault(dot_type, {})
                            machine_cfg["screens"][screen_code][dot_type].update(raw_rule)

                    for dot_type in ("black_dot", "white_dot"):
                        if isinstance(machine_cfg["screens"][screen_code].get(dot_type), dict):
                            machine_cfg["screens"][screen_code][dot_type].setdefault("defect_code", "C1111")

            normalized[machine_id] = machine_cfg

        return normalized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CAPIConfig":
        """從 dict 建立 CAPIConfig（供 round-trip / 測試使用）"""
        if data is None:
            data = {}
        from capi_image_preprocess_lab import normalize_preprocess_pipeline

        # 解析排除區域
        exclusion_zones = []
        for zone_data in data.get("exclusion_zones", []):
            exclusion_zones.append(ExclusionZone.from_dict(zone_data))

        # 判斷新架構：model_mapping 任一 value 為含 inner/edge key 的 nested dict
        # （不依賴 machine_id，因為舊版 yaml 也有 machine_id="CAPI_3F"，不能作為判別依據）
        raw_model_mapping = data.get("model_mapping", {})
        is_new = any(
            isinstance(v, dict) and {"inner", "edge"}.issubset(v.keys())
            for v in raw_model_mapping.values()
        )

        # threshold_mapping：舊架構為 {prefix: float}，新架構為 {prefix: {inner: float, edge: float}}
        raw_threshold_mapping = data.get("threshold_mapping", {})
        threshold_mapping = cls._normalize_threshold_mapping(raw_threshold_mapping)

        config = cls(
            machine_id=data.get("machine_id", "CAPI_3F"),
            product_name=data.get("product_name", "CAPI 面板"),
            mark_template_path=data.get("mark_template_path", "capi_mark.png"),
            mark_match_threshold=data.get("mark_match_threshold", 0.45),
            mark_fallback_position=data.get("mark_fallback_position", None),
            mark_min_y_ratio=data.get("mark_min_y_ratio", 0.6),
            mark_exclusion_padding_px=int(data.get("mark_exclusion_padding_px", 32)),
            mark_exclusion_soft_decay_px=int(data.get("mark_exclusion_soft_decay_px", 48)),
            cv_edge_exclude_soft_decay_px=int(data.get("cv_edge_exclude_soft_decay_px", 64)),
            no_detect_soft_decay_min_weight=float(data.get("no_detect_soft_decay_min_weight", 0.10)),
            otsu_offset=data.get("otsu_offset", 5),
            otsu_bottom_crop=data.get("otsu_bottom_crop", 1000),
            enable_panel_polygon=data.get("enable_panel_polygon", True),
            image_preprocess_pipeline=normalize_preprocess_pipeline(
                data.get("image_preprocess_pipeline", [])
            ),
            preprocess_after_tiling=data.get("preprocess_after_tiling", False),
            exclusion_zones=exclusion_zones,
            tile_size=data.get("tile_size", 512),
            tile_stride=data.get("tile_stride", 512),
            anomaly_threshold=data.get("anomaly_threshold", 0.5),
            model_path=data.get("model_path", ""),
            model_mapping=raw_model_mapping,
            threshold_mapping=threshold_mapping,
            is_new_architecture=is_new,
            edge_threshold_px=int(data.get("edge_threshold_px", 768)),
            patchcore_filter_enabled=data.get("patchcore_filter_enabled", False),
            patchcore_blur_sigma=data.get("patchcore_blur_sigma", 1.5),
            patchcore_min_area=data.get("patchcore_min_area", 10),
            patchcore_score_metric=data.get("patchcore_score_metric", "max"),
            patchcore_concentration_enabled=data.get("patchcore_concentration_enabled", True),
            patchcore_concentration_min_ratio=data.get("patchcore_concentration_min_ratio", 2.0),
            patchcore_concentration_penalty=data.get("patchcore_concentration_penalty", 0.5),
            patchcore_diffuse_area_enabled=data.get("patchcore_diffuse_area_enabled", True),
            patchcore_diffuse_area_threshold=data.get("patchcore_diffuse_area_threshold", 0.3),
            patchcore_diffuse_area_penalty=data.get("patchcore_diffuse_area_penalty", 0.5),
            dust_brightness_threshold=data.get("dust_brightness_threshold", 80),
            dust_threshold_floor=data.get("dust_threshold_floor", 25),
            dust_bright_rescue_threshold=data.get("dust_bright_rescue_threshold", 180),
            dust_area_min=data.get("dust_area_min", 15),
            dust_area_max=data.get("dust_area_max", 100000),
            dust_extension=data.get("dust_extension", 5),
            dust_heatmap_iou_threshold=data.get("dust_heatmap_iou_threshold", 0.02),
            dust_heatmap_top_percent=data.get("dust_heatmap_top_percent", 5.0),
            dust_heatmap_metric=data.get("dust_heatmap_metric", "coverage"),
            dust_mask_before_binarize=data.get("dust_mask_before_binarize", False),
            dust_two_stage_enabled=data.get("dust_two_stage_enabled", False),
            dust_two_stage_dust_ratio=data.get("dust_two_stage_dust_ratio", 0.3),
            dust_two_stage_bg_blur=data.get("dust_two_stage_bg_blur", 31),
            dust_two_stage_diff_percentile=data.get("dust_two_stage_diff_percentile", 90.0),
            dust_two_stage_min_area=data.get("dust_two_stage_min_area", 3),
            dust_detect_dark_particles=data.get("dust_detect_dark_particles", True),
            dust_detect_bubbles_enabled=data.get("dust_detect_bubbles_enabled", False),
            dust_high_cov_threshold=data.get("dust_high_cov_threshold", 0.5),
            dust_peak_fraction_threshold=data.get("dust_peak_fraction_threshold", 0.80),
            omit_overexposure_mean_threshold=data.get("omit_overexposure_mean_threshold", 200),
            omit_overexposure_ratio_threshold=data.get("omit_overexposure_ratio_threshold", 0.5),
            image_abnormal_detection_enabled=data.get("image_abnormal_detection_enabled", False),
            image_abnormal_standard_mean_lower=data.get("image_abnormal_standard_mean_lower", 68),
            image_abnormal_standard_mean_upper=data.get(
                "image_abnormal_standard_mean_upper",
                data.get("image_abnormal_standard_mean_threshold", 88),
            ),
            image_abnormal_wgf50500_mean_lower=data.get("image_abnormal_wgf50500_mean_lower", 72),
            image_abnormal_wgf50500_mean_upper=data.get(
                "image_abnormal_wgf50500_mean_upper",
                data.get("image_abnormal_wgf50500_mean_threshold", 92),
            ),
            image_abnormal_g0f00000_mean_lower=data.get("image_abnormal_g0f00000_mean_lower", 67),
            image_abnormal_g0f00000_mean_upper=data.get(
                "image_abnormal_g0f00000_mean_upper",
                data.get("image_abnormal_g0f00000_mean_threshold", 87),
            ),
            image_abnormal_r0f00000_mean_lower=data.get("image_abnormal_r0f00000_mean_lower", 71),
            image_abnormal_r0f00000_mean_upper=data.get(
                "image_abnormal_r0f00000_mean_upper",
                data.get("image_abnormal_r0f00000_mean_threshold", 91),
            ),
            image_abnormal_w0f00000_mean_lower=data.get("image_abnormal_w0f00000_mean_lower", 70),
            image_abnormal_w0f00000_mean_upper=data.get(
                "image_abnormal_w0f00000_mean_upper",
                data.get("image_abnormal_w0f00000_mean_threshold", 90),
            ),
            image_abnormal_b0f00000_mean_lower=data.get("image_abnormal_b0f00000_mean_lower", 0),
            image_abnormal_b0f00000_mean_upper=data.get(
                "image_abnormal_b0f00000_mean_upper",
                data.get("image_abnormal_b0f00000_mean_threshold", 13),
            ),
            edge_margin_px=data.get("edge_margin_px", 0),
            edge_margin_sides=data.get("edge_margin_sides", cls._migrate_edge_margin(data)),
            bright_spot_threshold=data.get("bright_spot_threshold", 200),
            bright_spot_min_area=data.get("bright_spot_min_area", 5),
            bright_spot_median_kernel=data.get("bright_spot_median_kernel", 21),
            bright_spot_diff_threshold=data.get("bright_spot_diff_threshold", 10),
            within_spec_judgment_rules=cls._normalize_within_spec_judgment_rules(
                data.get("within_spec_judgment_rules", {})
            ),
            report_black_dot_defect_code=data.get("report_black_dot_defect_code", "PCDK2"),
            report_white_dot_defect_code=data.get("report_white_dot_defect_code", "PTMD6"),
            report_unknown_dot_defect_code=data.get("report_unknown_dot_defect_code", "PCDK2"),
            report_bomb_defect_code=data.get("report_bomb_defect_code", "PCDK3"),
            report_image_abnormal_defect_code=data.get("report_image_abnormal_defect_code", "PCO05"),
            skip_files=data.get("skip_files", []),
            side_shot_prefixes=data.get("side_shot_prefixes", []),
            max_images_per_panel=data.get("max_images_per_panel", 7),
            bomb_defects=[BombDefect.from_dict(b) for b in data.get("bomb_defects", [])],
            bomb_match_tolerance=data.get("bomb_match_tolerance", 50),
            bomb_line_min_aspect_ratio=data.get("bomb_line_min_aspect_ratio", 3.0),
            bomb_area_force_detection_enabled=data.get("bomb_area_force_detection_enabled", False),
            model_resolution_map=data.get("model_resolution_map", {
                'B': [1366, 768], 'H': [1920, 1080], 'J': [1920, 1200],
                'K': [2560, 1440], 'G': [2560, 1600],
            }),
            grid_tiling_enabled=data.get("grid_tiling_enabled", True),
            aoi_coord_inspection_enabled=data.get("aoi_coord_inspection_enabled", False),
            aoi_heatmap_center_seed_enabled=data.get("aoi_heatmap_center_seed_enabled", True),
            aoi_report_path_replace_from=data.get("aoi_report_path_replace_from", "yuantu"),
            aoi_report_path_replace_to=data.get("aoi_report_path_replace_to", "Report"),
            scratch_classifier_enabled=data.get("scratch_classifier_enabled", True),
            scratch_safety_multiplier=float(data.get("scratch_safety_multiplier", 1.1)),
            scratch_bundle_path=data.get("scratch_bundle_path", "deployment/scratch_classifier_v1.pkl"),
            scratch_dinov2_weights_path=data.get("scratch_dinov2_weights_path", "deployment/dinov2_vitb14.pth"),
            scratch_dinov2_repo_path=data.get("scratch_dinov2_repo_path", ""),
        )

        return config

    def to_dict(self) -> Dict[str, Any]:
        """序列化為 dict（供 round-trip / 測試使用）"""
        return {
            "machine_id": self.machine_id,
            "is_new_architecture": self.is_new_architecture,
            "edge_threshold_px": self.edge_threshold_px,
            "product_name": self.product_name,
            "mark_template_path": self.mark_template_path,
            "mark_match_threshold": self.mark_match_threshold,
            "mark_min_y_ratio": self.mark_min_y_ratio,
            "mark_exclusion_padding_px": self.mark_exclusion_padding_px,
            "mark_exclusion_soft_decay_px": self.mark_exclusion_soft_decay_px,
            "cv_edge_exclude_soft_decay_px": self.cv_edge_exclude_soft_decay_px,
            "no_detect_soft_decay_min_weight": self.no_detect_soft_decay_min_weight,
            "mark_fallback_position": self.mark_fallback_position,
            "otsu_offset": self.otsu_offset,
            "otsu_bottom_crop": self.otsu_bottom_crop,
            "enable_panel_polygon": self.enable_panel_polygon,
            "image_preprocess_pipeline": self.image_preprocess_pipeline,
            "preprocess_after_tiling": self.preprocess_after_tiling,
            "exclusion_zones": [zone.to_dict() for zone in self.exclusion_zones],
            "tile_size": self.tile_size,
            "tile_stride": self.tile_stride,
            "anomaly_threshold": self.anomaly_threshold,
            "model_path": self.model_path,
            "model_mapping": self.model_mapping,
            "threshold_mapping": self.threshold_mapping,
            "patchcore_filter_enabled": self.patchcore_filter_enabled,
            "patchcore_blur_sigma": self.patchcore_blur_sigma,
            "patchcore_min_area": self.patchcore_min_area,
            "patchcore_score_metric": self.patchcore_score_metric,
            "patchcore_concentration_enabled": self.patchcore_concentration_enabled,
            "patchcore_concentration_min_ratio": self.patchcore_concentration_min_ratio,
            "patchcore_concentration_penalty": self.patchcore_concentration_penalty,
            "patchcore_diffuse_area_enabled": self.patchcore_diffuse_area_enabled,
            "patchcore_diffuse_area_threshold": self.patchcore_diffuse_area_threshold,
            "patchcore_diffuse_area_penalty": self.patchcore_diffuse_area_penalty,
            "dust_brightness_threshold": self.dust_brightness_threshold,
            "dust_threshold_floor": self.dust_threshold_floor,
            "dust_bright_rescue_threshold": self.dust_bright_rescue_threshold,
            "dust_area_min": self.dust_area_min,
            "dust_area_max": self.dust_area_max,
            "dust_extension": self.dust_extension,
            "dust_heatmap_iou_threshold": self.dust_heatmap_iou_threshold,
            "dust_heatmap_top_percent": self.dust_heatmap_top_percent,
            "dust_heatmap_metric": self.dust_heatmap_metric,
            "dust_mask_before_binarize": self.dust_mask_before_binarize,
            "dust_two_stage_enabled": self.dust_two_stage_enabled,
            "dust_two_stage_dust_ratio": self.dust_two_stage_dust_ratio,
            "dust_two_stage_bg_blur": self.dust_two_stage_bg_blur,
            "dust_two_stage_diff_percentile": self.dust_two_stage_diff_percentile,
            "dust_two_stage_min_area": self.dust_two_stage_min_area,
            "dust_detect_dark_particles": self.dust_detect_dark_particles,
            "dust_detect_bubbles_enabled": self.dust_detect_bubbles_enabled,
            "dust_high_cov_threshold": self.dust_high_cov_threshold,
            "dust_peak_fraction_threshold": self.dust_peak_fraction_threshold,
            "omit_overexposure_mean_threshold": self.omit_overexposure_mean_threshold,
            "omit_overexposure_ratio_threshold": self.omit_overexposure_ratio_threshold,
            "image_abnormal_detection_enabled": self.image_abnormal_detection_enabled,
            "image_abnormal_standard_mean_lower": self.image_abnormal_standard_mean_lower,
            "image_abnormal_standard_mean_upper": self.image_abnormal_standard_mean_upper,
            "image_abnormal_wgf50500_mean_lower": self.image_abnormal_wgf50500_mean_lower,
            "image_abnormal_wgf50500_mean_upper": self.image_abnormal_wgf50500_mean_upper,
            "image_abnormal_g0f00000_mean_lower": self.image_abnormal_g0f00000_mean_lower,
            "image_abnormal_g0f00000_mean_upper": self.image_abnormal_g0f00000_mean_upper,
            "image_abnormal_r0f00000_mean_lower": self.image_abnormal_r0f00000_mean_lower,
            "image_abnormal_r0f00000_mean_upper": self.image_abnormal_r0f00000_mean_upper,
            "image_abnormal_w0f00000_mean_lower": self.image_abnormal_w0f00000_mean_lower,
            "image_abnormal_w0f00000_mean_upper": self.image_abnormal_w0f00000_mean_upper,
            "image_abnormal_b0f00000_mean_lower": self.image_abnormal_b0f00000_mean_lower,
            "image_abnormal_b0f00000_mean_upper": self.image_abnormal_b0f00000_mean_upper,
            "edge_margin_px": self.edge_margin_px,
            "edge_margin_sides": self.edge_margin_sides,
            "bright_spot_threshold": self.bright_spot_threshold,
            "bright_spot_min_area": self.bright_spot_min_area,
            "bright_spot_median_kernel": self.bright_spot_median_kernel,
            "bright_spot_diff_threshold": self.bright_spot_diff_threshold,
            "within_spec_judgment_rules": self.within_spec_judgment_rules,
            "report_black_dot_defect_code": self.report_black_dot_defect_code,
            "report_white_dot_defect_code": self.report_white_dot_defect_code,
            "report_unknown_dot_defect_code": self.report_unknown_dot_defect_code,
            "report_bomb_defect_code": self.report_bomb_defect_code,
            "report_image_abnormal_defect_code": self.report_image_abnormal_defect_code,
            "skip_files": self.skip_files,
            "side_shot_prefixes": self.side_shot_prefixes,
            "max_images_per_panel": self.max_images_per_panel,
            "bomb_defects": [b.to_dict() for b in self.bomb_defects],
            "bomb_match_tolerance": self.bomb_match_tolerance,
            "bomb_line_min_aspect_ratio": self.bomb_line_min_aspect_ratio,
            "bomb_area_force_detection_enabled": self.bomb_area_force_detection_enabled,
            "model_resolution_map": self.model_resolution_map,
            "grid_tiling_enabled": self.grid_tiling_enabled,
            "aoi_coord_inspection_enabled": self.aoi_coord_inspection_enabled,
            "aoi_heatmap_center_seed_enabled": self.aoi_heatmap_center_seed_enabled,
            "aoi_report_path_replace_from": self.aoi_report_path_replace_from,
            "aoi_report_path_replace_to": self.aoi_report_path_replace_to,
            "scratch_classifier_enabled": self.scratch_classifier_enabled,
            "scratch_safety_multiplier": self.scratch_safety_multiplier,
            "scratch_bundle_path": self.scratch_bundle_path,
            "scratch_dinov2_weights_path": self.scratch_dinov2_weights_path,
            "scratch_dinov2_repo_path": self.scratch_dinov2_repo_path,
        }

    def to_yaml(self, yaml_path: str) -> None:
        """儲存配置到 YAML 檔案"""
        data = {
            "machine_id": self.machine_id,
            "product_name": self.product_name,
            "mark_template_path": self.mark_template_path,
            "mark_match_threshold": self.mark_match_threshold,
            "mark_min_y_ratio": self.mark_min_y_ratio,
            "mark_exclusion_padding_px": self.mark_exclusion_padding_px,
            "mark_exclusion_soft_decay_px": self.mark_exclusion_soft_decay_px,
            "cv_edge_exclude_soft_decay_px": self.cv_edge_exclude_soft_decay_px,
            "no_detect_soft_decay_min_weight": self.no_detect_soft_decay_min_weight,
            "mark_fallback_position": self.mark_fallback_position,
            "otsu_offset": self.otsu_offset,
            "otsu_bottom_crop": self.otsu_bottom_crop,
            "enable_panel_polygon": self.enable_panel_polygon,
            "image_preprocess_pipeline": self.image_preprocess_pipeline,
            "preprocess_after_tiling": self.preprocess_after_tiling,
            "exclusion_zones": [zone.to_dict() for zone in self.exclusion_zones],
            "tile_size": self.tile_size,
            "tile_stride": self.tile_stride,
            "anomaly_threshold": self.anomaly_threshold,
            "model_mapping": self.model_mapping,
            "threshold_mapping": self.threshold_mapping,
            "patchcore_filter_enabled": self.patchcore_filter_enabled,
            "patchcore_blur_sigma": self.patchcore_blur_sigma,
            "patchcore_min_area": self.patchcore_min_area,
            "patchcore_score_metric": self.patchcore_score_metric,
            "patchcore_concentration_enabled": self.patchcore_concentration_enabled,
            "patchcore_concentration_min_ratio": self.patchcore_concentration_min_ratio,
            "patchcore_concentration_penalty": self.patchcore_concentration_penalty,
            "patchcore_diffuse_area_enabled": self.patchcore_diffuse_area_enabled,
            "patchcore_diffuse_area_threshold": self.patchcore_diffuse_area_threshold,
            "patchcore_diffuse_area_penalty": self.patchcore_diffuse_area_penalty,
            "dust_brightness_threshold": self.dust_brightness_threshold,
            "dust_area_min": self.dust_area_min,
            "dust_area_max": self.dust_area_max,
            "dust_extension": self.dust_extension,
            "dust_heatmap_iou_threshold": self.dust_heatmap_iou_threshold,
            "dust_heatmap_top_percent": self.dust_heatmap_top_percent,
            "dust_heatmap_metric": self.dust_heatmap_metric,
            "dust_mask_before_binarize": self.dust_mask_before_binarize,
            "dust_two_stage_enabled": self.dust_two_stage_enabled,
            "dust_two_stage_dust_ratio": self.dust_two_stage_dust_ratio,
            "dust_two_stage_bg_blur": self.dust_two_stage_bg_blur,
            "dust_two_stage_diff_percentile": self.dust_two_stage_diff_percentile,
            "dust_two_stage_min_area": self.dust_two_stage_min_area,
            "dust_detect_dark_particles": self.dust_detect_dark_particles,
            "dust_detect_bubbles_enabled": self.dust_detect_bubbles_enabled,
            "dust_high_cov_threshold": self.dust_high_cov_threshold,
            "dust_peak_fraction_threshold": self.dust_peak_fraction_threshold,
            "omit_overexposure_mean_threshold": self.omit_overexposure_mean_threshold,
            "omit_overexposure_ratio_threshold": self.omit_overexposure_ratio_threshold,
            "image_abnormal_detection_enabled": self.image_abnormal_detection_enabled,
            "image_abnormal_standard_mean_lower": self.image_abnormal_standard_mean_lower,
            "image_abnormal_standard_mean_upper": self.image_abnormal_standard_mean_upper,
            "image_abnormal_wgf50500_mean_lower": self.image_abnormal_wgf50500_mean_lower,
            "image_abnormal_wgf50500_mean_upper": self.image_abnormal_wgf50500_mean_upper,
            "image_abnormal_g0f00000_mean_lower": self.image_abnormal_g0f00000_mean_lower,
            "image_abnormal_g0f00000_mean_upper": self.image_abnormal_g0f00000_mean_upper,
            "image_abnormal_r0f00000_mean_lower": self.image_abnormal_r0f00000_mean_lower,
            "image_abnormal_r0f00000_mean_upper": self.image_abnormal_r0f00000_mean_upper,
            "image_abnormal_w0f00000_mean_lower": self.image_abnormal_w0f00000_mean_lower,
            "image_abnormal_w0f00000_mean_upper": self.image_abnormal_w0f00000_mean_upper,
            "image_abnormal_b0f00000_mean_lower": self.image_abnormal_b0f00000_mean_lower,
            "image_abnormal_b0f00000_mean_upper": self.image_abnormal_b0f00000_mean_upper,
            "edge_margin_px": self.edge_margin_px,
            "edge_margin_sides": self.edge_margin_sides,
            "bright_spot_threshold": self.bright_spot_threshold,
            "bright_spot_min_area": self.bright_spot_min_area,
            "bright_spot_median_kernel": self.bright_spot_median_kernel,
            "bright_spot_diff_threshold": self.bright_spot_diff_threshold,
            "within_spec_judgment_rules": self.within_spec_judgment_rules,
            "report_black_dot_defect_code": self.report_black_dot_defect_code,
            "report_white_dot_defect_code": self.report_white_dot_defect_code,
            "report_unknown_dot_defect_code": self.report_unknown_dot_defect_code,
            "report_bomb_defect_code": self.report_bomb_defect_code,
            "report_image_abnormal_defect_code": self.report_image_abnormal_defect_code,
            "skip_files": self.skip_files,
            "max_images_per_panel": self.max_images_per_panel,
            "bomb_defects": [b.to_dict() for b in self.bomb_defects],
            "bomb_match_tolerance": self.bomb_match_tolerance,
            "bomb_line_min_aspect_ratio": self.bomb_line_min_aspect_ratio,
            "bomb_area_force_detection_enabled": self.bomb_area_force_detection_enabled,
            "model_resolution_map": self.model_resolution_map,
            "grid_tiling_enabled": self.grid_tiling_enabled,
            "aoi_coord_inspection_enabled": self.aoi_coord_inspection_enabled,
            "aoi_heatmap_center_seed_enabled": self.aoi_heatmap_center_seed_enabled,
            "aoi_report_path_replace_from": self.aoi_report_path_replace_from,
            "aoi_report_path_replace_to": self.aoi_report_path_replace_to,
            "scratch_classifier_enabled": self.scratch_classifier_enabled,
            "scratch_safety_multiplier": self.scratch_safety_multiplier,
            "scratch_bundle_path": self.scratch_bundle_path,
            "scratch_dinov2_weights_path": self.scratch_dinov2_weights_path,
            "scratch_dinov2_repo_path": self.scratch_dinov2_repo_path,
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def get_default(cls) -> "CAPIConfig":
        """取得預設配置（CAPI_3F）"""
        return cls(
            exclusion_zones=[
                ExclusionZone(
                    name="mark_area",
                    type="template_match",
                    description="MARK 二維碼區域",
                    enabled=True,
                ),
                ExclusionZone(
                    name="bottom_right_mechanism",
                    type="relative_bottom_right",
                    description="右下角機構區域",
                    width=460,
                    height=135,
                    enabled=True,
                ),
            ]
        )
    
    @classmethod
    def _migrate_edge_margin(cls, data: Dict[str, Any]) -> Dict[str, bool]:
        """向後相容：將舊版 edge_margin_bottom_only 轉換為新版 edge_margin_sides

        注意：edge_margin_px 預設 0 = 衰減停用，此函式回傳的 sides 沒實際作用；
        留作 yaml 內顯式設 px>0 時的 sides fallback 行為。
        """
        bottom_only = data.get("edge_margin_bottom_only", True)
        if bottom_only:
            return {'top': False, 'bottom': True, 'left': False, 'right': False}
        else:
            return {'top': True, 'bottom': True, 'left': True, 'right': True}
    
    def apply_db_overrides(self, db_params: list) -> None:
        """
        將 DB 設定覆蓋至 config 屬性 (DB 優先原則)

        Args:
            db_params: get_all_config_params() 回傳的列表
        """
        param_map = {p["param_name"]: p["decoded_value"] for p in db_params}

        if "anomaly_threshold" in param_map:
            self.anomaly_threshold = float(param_map["anomaly_threshold"])
        # 新架構：threshold_mapping / model_mapping 永遠以 yaml 為唯一來源，
        # 不接受 DB override。否則 yaml 改完重啟會被首次 init 灌進 DB 的舊值蓋掉
        # （bug 案例：machine_config.yaml 改 0.5→0.4，重啟後 DB 殘留 0.5 把 yaml 蓋回去）
        if not self.is_new_architecture:
            if "model_mapping" in param_map and isinstance(param_map["model_mapping"], dict):
                self.model_mapping = param_map["model_mapping"]
            if "threshold_mapping" in param_map and isinstance(param_map["threshold_mapping"], dict):
                self.threshold_mapping = self._normalize_threshold_mapping(param_map["threshold_mapping"])
        if "patchcore_filter_enabled" in param_map:
            val = param_map["patchcore_filter_enabled"]
            self.patchcore_filter_enabled = str(val).lower() == "true" if isinstance(val, str) else bool(val)
        if "patchcore_blur_sigma" in param_map:
            self.patchcore_blur_sigma = float(param_map["patchcore_blur_sigma"])
        if "patchcore_min_area" in param_map:
            self.patchcore_min_area = int(param_map["patchcore_min_area"])
        if "patchcore_score_metric" in param_map:
            self.patchcore_score_metric = str(param_map["patchcore_score_metric"])
        if "patchcore_concentration_enabled" in param_map:
            val = param_map["patchcore_concentration_enabled"]
            self.patchcore_concentration_enabled = str(val).lower() == "true" if isinstance(val, str) else bool(val)
        if "patchcore_concentration_min_ratio" in param_map:
            self.patchcore_concentration_min_ratio = float(param_map["patchcore_concentration_min_ratio"])
        if "patchcore_concentration_penalty" in param_map:
            self.patchcore_concentration_penalty = float(param_map["patchcore_concentration_penalty"])
        if "patchcore_diffuse_area_enabled" in param_map:
            val = param_map["patchcore_diffuse_area_enabled"]
            self.patchcore_diffuse_area_enabled = str(val).lower() == "true" if isinstance(val, str) else bool(val)
        if "patchcore_diffuse_area_threshold" in param_map:
            self.patchcore_diffuse_area_threshold = float(param_map["patchcore_diffuse_area_threshold"])
        if "patchcore_diffuse_area_penalty" in param_map:
            self.patchcore_diffuse_area_penalty = float(param_map["patchcore_diffuse_area_penalty"])
        if "mark_exclusion_padding_px" in param_map:
            self.mark_exclusion_padding_px = int(param_map["mark_exclusion_padding_px"])
        if "mark_exclusion_soft_decay_px" in param_map:
            self.mark_exclusion_soft_decay_px = int(param_map["mark_exclusion_soft_decay_px"])
        if "cv_edge_exclude_soft_decay_px" in param_map:
            self.cv_edge_exclude_soft_decay_px = int(param_map["cv_edge_exclude_soft_decay_px"])
        if "no_detect_soft_decay_min_weight" in param_map:
            self.no_detect_soft_decay_min_weight = float(param_map["no_detect_soft_decay_min_weight"])
        if "dust_brightness_threshold" in param_map:
            self.dust_brightness_threshold = int(param_map["dust_brightness_threshold"])
        if "dust_area_min" in param_map:
            self.dust_area_min = int(param_map["dust_area_min"])
        if "dust_area_max" in param_map:
            self.dust_area_max = int(param_map["dust_area_max"])
        if "dust_extension" in param_map:
            self.dust_extension = int(param_map["dust_extension"])
        if "dust_heatmap_iou_threshold" in param_map:
            self.dust_heatmap_iou_threshold = float(param_map["dust_heatmap_iou_threshold"])
        if "dust_heatmap_top_percent" in param_map:
            self.dust_heatmap_top_percent = float(param_map["dust_heatmap_top_percent"])
        if "dust_heatmap_metric" in param_map:
            self.dust_heatmap_metric = str(param_map["dust_heatmap_metric"]).lower()
        if "dust_mask_before_binarize" in param_map:
            val = param_map["dust_mask_before_binarize"]
            self.dust_mask_before_binarize = str(val).lower() == "true" if isinstance(val, str) else bool(val)
        if "dust_two_stage_enabled" in param_map:
            val = param_map["dust_two_stage_enabled"]
            self.dust_two_stage_enabled = str(val).lower() == "true" if isinstance(val, str) else bool(val)
        if "dust_two_stage_dust_ratio" in param_map:
            self.dust_two_stage_dust_ratio = float(param_map["dust_two_stage_dust_ratio"])
        if "dust_two_stage_bg_blur" in param_map:
            self.dust_two_stage_bg_blur = int(param_map["dust_two_stage_bg_blur"])
        if "dust_two_stage_diff_percentile" in param_map:
            self.dust_two_stage_diff_percentile = float(param_map["dust_two_stage_diff_percentile"])
        if "dust_two_stage_min_area" in param_map:
            self.dust_two_stage_min_area = int(param_map["dust_two_stage_min_area"])
        if "dust_detect_dark_particles" in param_map:
            val = param_map["dust_detect_dark_particles"]
            self.dust_detect_dark_particles = str(val).lower() == "true" if isinstance(val, str) else bool(val)
        if "dust_detect_bubbles_enabled" in param_map:
            val = param_map["dust_detect_bubbles_enabled"]
            self.dust_detect_bubbles_enabled = str(val).lower() == "true" if isinstance(val, str) else bool(val)
        if "dust_high_cov_threshold" in param_map:
            self.dust_high_cov_threshold = float(param_map["dust_high_cov_threshold"])
        if "dust_peak_fraction_threshold" in param_map:
            self.dust_peak_fraction_threshold = float(param_map["dust_peak_fraction_threshold"])
        if "image_abnormal_detection_enabled" in param_map:
            val = param_map["image_abnormal_detection_enabled"]
            self.image_abnormal_detection_enabled = str(val).lower() == "true" if isinstance(val, str) else bool(val)
        for name in (
            "image_abnormal_standard_mean_lower",
            "image_abnormal_standard_mean_upper",
            "image_abnormal_wgf50500_mean_lower",
            "image_abnormal_wgf50500_mean_upper",
            "image_abnormal_g0f00000_mean_lower",
            "image_abnormal_g0f00000_mean_upper",
            "image_abnormal_r0f00000_mean_lower",
            "image_abnormal_r0f00000_mean_upper",
            "image_abnormal_w0f00000_mean_lower",
            "image_abnormal_w0f00000_mean_upper",
            "image_abnormal_b0f00000_mean_lower",
            "image_abnormal_b0f00000_mean_upper",
        ):
            if name in param_map:
                setattr(self, name, int(param_map[name]))
        legacy_image_abnormal_upper_fields = {
            "image_abnormal_standard_mean_threshold": "image_abnormal_standard_mean_upper",
            "image_abnormal_wgf50500_mean_threshold": "image_abnormal_wgf50500_mean_upper",
            "image_abnormal_g0f00000_mean_threshold": "image_abnormal_g0f00000_mean_upper",
            "image_abnormal_r0f00000_mean_threshold": "image_abnormal_r0f00000_mean_upper",
            "image_abnormal_w0f00000_mean_threshold": "image_abnormal_w0f00000_mean_upper",
            "image_abnormal_b0f00000_mean_threshold": "image_abnormal_b0f00000_mean_upper",
        }
        for legacy_name, upper_name in legacy_image_abnormal_upper_fields.items():
            if upper_name not in param_map and legacy_name in param_map:
                setattr(self, upper_name, int(param_map[legacy_name]))
        if "grid_tiling_enabled" in param_map:
            val = param_map["grid_tiling_enabled"]
            self.grid_tiling_enabled = str(val).lower() == "true" if isinstance(val, str) else bool(val)
        if "aoi_coord_inspection_enabled" in param_map:
            val = param_map["aoi_coord_inspection_enabled"]
            self.aoi_coord_inspection_enabled = str(val).lower() == "true" if isinstance(val, str) else bool(val)
        if "aoi_heatmap_center_seed_enabled" in param_map:
            val = param_map["aoi_heatmap_center_seed_enabled"]
            self.aoi_heatmap_center_seed_enabled = str(val).lower() == "true" if isinstance(val, str) else bool(val)
        if "bomb_area_force_detection_enabled" in param_map:
            val = param_map["bomb_area_force_detection_enabled"]
            self.bomb_area_force_detection_enabled = str(val).lower() == "true" if isinstance(val, str) else bool(val)
        if "bright_spot_threshold" in param_map:
            self.bright_spot_threshold = int(param_map["bright_spot_threshold"])
        if "bright_spot_min_area" in param_map:
            self.bright_spot_min_area = int(param_map["bright_spot_min_area"])
        if "bright_spot_median_kernel" in param_map:
            self.bright_spot_median_kernel = int(param_map["bright_spot_median_kernel"])
        if "bright_spot_diff_threshold" in param_map:
            self.bright_spot_diff_threshold = int(param_map["bright_spot_diff_threshold"])
        if "within_spec_judgment_rules" in param_map and isinstance(param_map["within_spec_judgment_rules"], dict):
            self.within_spec_judgment_rules = self._normalize_within_spec_judgment_rules(
                param_map["within_spec_judgment_rules"]
            )
        if "report_black_dot_defect_code" in param_map:
            self.report_black_dot_defect_code = str(param_map["report_black_dot_defect_code"]).strip() or "PCDK2"
        if "report_white_dot_defect_code" in param_map:
            self.report_white_dot_defect_code = str(param_map["report_white_dot_defect_code"]).strip() or "PTMD6"
        if "report_unknown_dot_defect_code" in param_map:
            self.report_unknown_dot_defect_code = str(param_map["report_unknown_dot_defect_code"]).strip() or self.report_black_dot_defect_code
        if "report_bomb_defect_code" in param_map:
            self.report_bomb_defect_code = str(param_map["report_bomb_defect_code"]).strip() or "PCDK3"
        if "report_image_abnormal_defect_code" in param_map:
            self.report_image_abnormal_defect_code = str(param_map["report_image_abnormal_defect_code"]).strip() or "PCO05"
        if "aoi_report_path_replace_from" in param_map:
            self.aoi_report_path_replace_from = str(param_map["aoi_report_path_replace_from"])
        if "aoi_report_path_replace_to" in param_map:
            self.aoi_report_path_replace_to = str(param_map["aoi_report_path_replace_to"])
        if "scratch_classifier_enabled" in param_map:
            val = param_map["scratch_classifier_enabled"]
            self.scratch_classifier_enabled = str(val).lower() == "true" if isinstance(val, str) else bool(val)
        if "scratch_safety_multiplier" in param_map:
            self.scratch_safety_multiplier = float(param_map["scratch_safety_multiplier"])
        if "scratch_bundle_path" in param_map:
            self.scratch_bundle_path = str(param_map["scratch_bundle_path"])
        if "scratch_dinov2_weights_path" in param_map:
            self.scratch_dinov2_weights_path = str(param_map["scratch_dinov2_weights_path"])
        if "scratch_dinov2_repo_path" in param_map:
            self.scratch_dinov2_repo_path = str(param_map["scratch_dinov2_repo_path"])

    def get_enabled_exclusion_zones(self) -> List[ExclusionZone]:
        """取得已啟用的排除區域"""
        return [zone for zone in self.exclusion_zones if zone.enabled]
    
    def should_skip_file(self, filename: str) -> bool:
        """檢查是否應該跳過此檔案（使用前綴比對，支援帶時間戳的檔名）"""
        stem = Path(filename).stem  # 取得不含副檔名的名稱
        
        # 側拍圖自動跳過 (e.g. ["SG0F00000", "SSTANDARD"] → SG0F00000_114438.tif 等)
        for prefix in self.side_shot_prefixes:
            if stem == prefix or stem.startswith(prefix + "_"):
                return True
        
        for skip_pattern in self.skip_files:
            # 支援前綴比對：skip_pattern "B0F00000" 可匹配 "B0F00000_031447.tif"
            if stem == skip_pattern or stem.startswith(skip_pattern + "_") or filename == skip_pattern:
                return True
        return False
    
    def get_mark_template_full_path(self, base_dir: Path) -> Path:
        """取得 MARK 模板的完整路徑"""
        template_path = Path(self.mark_template_path)
        if template_path.is_absolute():
            return template_path
        return base_dir / template_path
    
    def __str__(self) -> str:
        zones_str = ", ".join([z.name for z in self.get_enabled_exclusion_zones()])
        return (
            f"CAPIConfig(machine={self.machine_id}, "
            f"tile={self.tile_size}x{self.tile_size}, "
            f"exclusions=[{zones_str}])"
        )


def list_available_configs(configs_dir: str = "configs") -> List[Path]:
    """列出可用的配置檔"""
    config_path = Path(configs_dir)
    if not config_path.exists():
        return []
    return list(config_path.glob("*.yaml")) + list(config_path.glob("*.yml"))


if __name__ == "__main__":
    # 測試配置載入
    print("=" * 50)
    print("CAPI 配置管理模組測試")
    print("=" * 50)
    
    # 測試預設配置
    default_config = CAPIConfig.get_default()
    print(f"\n預設配置: {default_config}")
    
    # 列出可用配置
    configs = list_available_configs()
    print(f"\n可用配置檔: {[str(c) for c in configs]}")
    
    # 載入 YAML 配置
    if configs:
        config = CAPIConfig.from_yaml(str(configs[0]))
        print(f"\n載入配置: {config}")
        print(f"  - 機種: {config.machine_id}")
        print(f"  - 產品: {config.product_name}")
        print(f"  - MARK 模板: {config.mark_template_path}")
        print(f"  - 切塊尺寸: {config.tile_size}")
        print(f"  - 排除區域:")
        for zone in config.get_enabled_exclusion_zones():
            print(f"    - {zone.name}: {zone.type}")
