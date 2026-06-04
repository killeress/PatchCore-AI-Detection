import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_web import tile_info


def test_tile_info_does_not_show_bomb_for_ai_ok_tile():
    _badge, info = tile_info({
        "score": 0.0,
        "is_aoi_coord": True,
        "aoi_defect_code": "C1111",
        "is_anomaly": False,
        "is_bomb": True,
        "bomb_code": "B01",
    })

    assert "AI判OK" in info
    assert "炸彈" not in info
    assert "B01" not in info


def test_tile_info_still_shows_bomb_for_anomaly_tile():
    _badge, info = tile_info({
        "score": 1.0,
        "is_aoi_coord": True,
        "aoi_defect_code": "C1111",
        "is_anomaly": True,
        "is_bomb": True,
        "bomb_code": "B01",
    })

    assert "AI也判NG" in info
    assert "炸彈代碼: B01" in info
