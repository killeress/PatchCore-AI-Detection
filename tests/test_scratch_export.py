import cv2
import numpy as np

from capi_scratch_export import export_misrescue_samples


class _FakeDB:
    def __init__(self, candidate):
        self.candidate = candidate

    def list_scratch_misrescue_for_export(self, start_date=None, end_date=None):
        return [self.candidate]


def test_scratch_export_crops_from_rotated_inference_orientation(tmp_path):
    """Scratch 誤救重裁切應沿用推論方向與座標。"""
    source_path = tmp_path / "G0F00000.tif"
    image = np.full((512, 1024, 3), 10, dtype=np.uint8)
    image[:, 512:] = 240
    assert cv2.imwrite(str(source_path), image)

    db = _FakeDB({
        "glass_id": "GLS123",
        "image_name": source_path.name,
        "image_path": str(source_path),
        "record_id": 1,
        "image_result_id": 2,
        "tile_result_id": 3,
        "tile_seq": 0,
        "x": 0,
        "y": 0,
        "width": 512,
        "height": 512,
        "scratch_score": 0.9,
        "score": 0.8,
        "created_at": "2026-07-22T00:00:00",
    })

    summary = export_misrescue_samples(
        db,
        tmp_path / "out",
        rotate_180=True,
    )
    crops = list((tmp_path / "out").glob("*/misrescue_negative/*/crop/*.png"))

    assert summary["exported"] == 1
    assert len(crops) == 1
    crop = cv2.imread(str(crops[0]), cv2.IMREAD_UNCHANGED)
    assert crop is not None
    assert np.all(crop == 240)
