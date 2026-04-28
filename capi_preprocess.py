"""共用前處理：訓練 / 推論皆使用此模組。

從 capi_inference.py 抽出 Otsu / panel polygon / tile 切分 / zone 分類，
讓訓練端與推論端走同一套邏輯。
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np


@dataclass
class PreprocessConfig:
    tile_size: int = 512
    tile_stride: int = 512
    otsu_offset: int = 5
    enable_panel_polygon: bool = True
    edge_threshold_px: int = 768
    coverage_min: float = 0.3


@dataclass
class TileResult:
    tile_id: int
    x1: int; y1: int; x2: int; y2: int
    image: np.ndarray
    mask: Optional[np.ndarray]
    coverage: float
    zone: str  # "inner" | "edge" | "outside"
    center_dist_to_edge: float


@dataclass
class PanelPreprocessResult:
    image_path: Path
    lighting: str
    foreground_bbox: Tuple[int, int, int, int]
    panel_polygon: Optional[np.ndarray]
    tiles: List[TileResult] = field(default_factory=list)
    polygon_detection_failed: bool = False


LIGHTING_PREFIXES = ("G0F00000", "R0F00000", "W0F00000", "WGF50500", "STANDARD")
SKIP_PREFIXES = ("S", "B0F", "PINIGBI", "OMIT")
SKIP_EXACT = ("Optics.log",)


def filter_panel_lighting_files(folder: Path) -> Dict[str, Path]:
    """從 panel folder 過濾出 5 個有效 lighting 圖。

    跳過：S* (側拍) / B0F (黑屏) / PINIGBI (點燈狀態檔) / OMIT (光源圖) /
          Optics.log。

    Returns: {"G0F00000": Path, ...}，缺哪個 lighting 就少哪個 key。
    """
    result: Dict[str, Path] = {}
    for entry in folder.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name in SKIP_EXACT:
            continue
        # 優先比對 lighting prefix（STANDARD 開頭含 S，須先於 skip 判斷）
        matched = False
        for lighting in LIGHTING_PREFIXES:
            if name.startswith(lighting):
                if lighting not in result:
                    result[lighting] = entry
                matched = True
                break
        if matched:
            continue
        # 其餘非 lighting 的檔案再套用 skip 規則（可省略，但保留語意清晰）
    return result
