import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_database import CAPIDatabase


def _save_record(db: CAPIDatabase, glass_id: str, *, tile_is_dust: int) -> int:
    image_results_data = [{
        "image_path": f"/fake/{glass_id}.jpg",
        "image_name": f"{glass_id}.jpg",
        "image_width": 1024,
        "image_height": 768,
        "otsu_bounds": "",
        "tile_count": 1,
        "excluded_tiles": 0,
        "anomaly_count": 1,
        "max_score": 0.95,
        "is_ng": 0,
        "is_dust_only": 0,
        "is_bomb": 0,
        "inference_time_ms": 100.0,
        "heatmap_path": "",
        "tiles": [{
            "tile_id": 1,
            "x": 10,
            "y": 20,
            "width": 512,
            "height": 512,
            "score": 0.95,
            "is_anomaly": 1,
            "is_dust": tile_is_dust,
            "dust_iou": 0.8 if tile_is_dust else 0.0,
            "is_bomb": 0,
            "bomb_code": "",
            "peak_x": 100,
            "peak_y": 120,
            "heatmap_path": "",
            "is_exclude_zone": 0,
            "is_aoi_coord": 0,
            "aoi_defect_code": "",
            "aoi_product_x": -1,
            "aoi_product_y": -1,
        }],
    }]
    return db.save_inference_record(
        glass_id=glass_id,
        model_id="M1",
        machine_no="1",
        resolution=(1920, 1080),
        machine_judgment="NG",
        ai_judgment="OK",
        image_dir="/fake",
        total_images=1,
        ng_images=0,
        ng_details="[]",
        request_time="2026-05-26T10:00:00",
        response_time="2026-05-26T10:00:01",
        processing_seconds=1.0,
        image_results_data=image_results_data,
    )


def test_dust_affected_records_include_tile_dust_when_image_is_not_dust_only(tmp_path):
    db = CAPIDatabase(str(tmp_path / "ric_omit.db"))
    dust_record_id = _save_record(db, "DUST_TILE", tile_is_dust=1)
    clean_record_id = _save_record(db, "CLEAN_TILE", tile_is_dust=0)

    assert db.get_dust_affected_record_ids([dust_record_id, clean_record_id]) == {dust_record_id}
