"""
CAPI 配置管理模組

提供 CAPI 推論系統的配置載入、驗證和管理功能。
支援 YAML 配置檔格式，可針對不同機種/產品設定排除區域等參數。

使用方式:
    from capi_config import CAPIConfig
    config = CAPIConfig.from_yaml("configs/capi_3f.yaml")
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple


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
    
    # MARK 模板設定
    mark_template_path: str = "capi_mark.png"
    mark_fallback_position: Optional[Dict[str, int]] = None  # Fallback MARK 位置 {x, y, width, height}
    mark_match_threshold: float = 0.45
    mark_min_y_ratio: float = 0.6
    
    # Otsu 裁剪設定
    otsu_offset: int = 5
    otsu_bottom_crop: int = 1000  # Otsu 後裁切底部像素
    
    # 排除區域
    exclusion_zones: List[ExclusionZone] = field(default_factory=list)
    
    # 切塊設定
    tile_size: int = 512
    tile_stride: int = 512
    
    # 推論設定
    anomaly_threshold: float = 0.5
    model_path: str = ""  # 預設模型路徑
    
    # 灰塵偵測設定 (OMIT 圖片分析)
    dust_brightness_threshold: int = 80       # OMIT 亮度閾值 (自適應 Otsu 時此為備用)
    dust_area_min: int = 10                   # 灰塵顆粒最小面積 (px)
    dust_area_max: int = 50000                # 灰塵顆粒最大面積 (px)
    dust_extension: int = 5                   # 灰塵區域膨脹像素
    dust_heatmap_iou_threshold: float = 0.02  # Heatmap-Dust IOU 閾值
    dust_heatmap_top_percent: float = 5.0     # Heatmap 熱區取前 X% (Percentile 二值化)
    
    # OMIT 過曝偵測設定 (曝光過高的 OMIT 圖無法檢測灰塵，需記錄供工程追蹤)
    omit_overexposure_mean_threshold: int = 200    # 平均亮度超過此值視為過曝
    omit_overexposure_ratio_threshold: float = 0.5 # 高亮像素(>230)佔比超過此值視為過曝
    
    # 底部邊緣衰減設定 (過濾光影假陽性)
    edge_margin_px: int = 80                  # 底部邊緣衰減高度 (px)，0=停用
    edge_margin_bottom_only: bool = True       # 是否只對底排 tile 生效
    
    # 跳過檔案設定 (不進行推論的檔案名稱)
    skip_files: List[str] = field(default_factory=list)
    
    # 側拍圖前綴列表 (以這些前綴開頭的檔案自動跳過，不在模型訓練資料中)
    side_shot_prefixes: List[str] = field(default_factory=list)
    
    # 重複投片檢查 (Panel 資料夾內圖片數超過此值則跳過)
    max_images_per_panel: int = 20
    
    # 炸彈系統設定 (機台模擬缺陷)
    bomb_defects: List[BombDefect] = field(default_factory=list)
    bomb_match_tolerance: int = 100  # 座標匹配容忍度 (產品座標系像素)
    
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
        
        # 解析排除區域
        exclusion_zones = []
        for zone_data in data.get("exclusion_zones", []):
            exclusion_zones.append(ExclusionZone.from_dict(zone_data))
        
        config = cls(
            machine_id=data.get("machine_id", "CAPI_3F"),
            product_name=data.get("product_name", "CAPI 面板"),
            mark_template_path=data.get("mark_template_path", "capi_mark.png"),
            mark_match_threshold=data.get("mark_match_threshold", 0.45),
            mark_fallback_position=data.get("mark_fallback_position", None),
            mark_min_y_ratio=data.get("mark_min_y_ratio", 0.6),
            otsu_offset=data.get("otsu_offset", 5),
            otsu_bottom_crop=data.get("otsu_bottom_crop", 1000),
            exclusion_zones=exclusion_zones,
            tile_size=data.get("tile_size", 512),
            tile_stride=data.get("tile_stride", 512),
            anomaly_threshold=data.get("anomaly_threshold", 0.5),
            model_path=data.get("model_path", ""),
            dust_brightness_threshold=data.get("dust_brightness_threshold", 80),
            dust_area_min=data.get("dust_area_min", 10),
            dust_area_max=data.get("dust_area_max", 50000),
            dust_extension=data.get("dust_extension", 5),
            dust_heatmap_iou_threshold=data.get("dust_heatmap_iou_threshold", 0.02),
            dust_heatmap_top_percent=data.get("dust_heatmap_top_percent", 5.0),
            omit_overexposure_mean_threshold=data.get("omit_overexposure_mean_threshold", 200),
            omit_overexposure_ratio_threshold=data.get("omit_overexposure_ratio_threshold", 0.5),
            edge_margin_px=data.get("edge_margin_px", 80),
            edge_margin_bottom_only=data.get("edge_margin_bottom_only", True),
            skip_files=data.get("skip_files", []),
            side_shot_prefixes=data.get("side_shot_prefixes", []),
            max_images_per_panel=data.get("max_images_per_panel", 7),
            bomb_defects=[BombDefect.from_dict(b) for b in data.get("bomb_defects", [])],
            bomb_match_tolerance=data.get("bomb_match_tolerance", 100),
            config_path=path,
        )
        
        return config
    
    def to_yaml(self, yaml_path: str) -> None:
        """儲存配置到 YAML 檔案"""
        data = {
            "machine_id": self.machine_id,
            "product_name": self.product_name,
            "mark_template_path": self.mark_template_path,
            "mark_match_threshold": self.mark_match_threshold,
            "mark_min_y_ratio": self.mark_min_y_ratio,
            "mark_fallback_position": self.mark_fallback_position,
            "otsu_offset": self.otsu_offset,
            "otsu_bottom_crop": self.otsu_bottom_crop,
            "exclusion_zones": [zone.to_dict() for zone in self.exclusion_zones],
            "tile_size": self.tile_size,
            "tile_stride": self.tile_stride,
            "anomaly_threshold": self.anomaly_threshold,
            "dust_brightness_threshold": self.dust_brightness_threshold,
            "dust_area_min": self.dust_area_min,
            "dust_area_max": self.dust_area_max,
            "dust_extension": self.dust_extension,
            "dust_heatmap_iou_threshold": self.dust_heatmap_iou_threshold,
            "dust_heatmap_top_percent": self.dust_heatmap_top_percent,
            "omit_overexposure_mean_threshold": self.omit_overexposure_mean_threshold,
            "omit_overexposure_ratio_threshold": self.omit_overexposure_ratio_threshold,
            "edge_margin_px": self.edge_margin_px,
            "edge_margin_bottom_only": self.edge_margin_bottom_only,
            "skip_files": self.skip_files,
            "max_images_per_panel": self.max_images_per_panel,
            "bomb_defects": [b.to_dict() for b in self.bomb_defects],
            "bomb_match_tolerance": self.bomb_match_tolerance,
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
