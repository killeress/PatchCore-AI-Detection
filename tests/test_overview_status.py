import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from capi_edge_cv import EdgeDefect
from capi_inference import CAPIInferencer, ImageResult


def _make_inferencer_for_overview():
    inferencer = object.__new__(CAPIInferencer)
    inferencer.config = SimpleNamespace(dust_heatmap_metric="coverage")
    inferencer.edge_inspector = None
    return inferencer


def _make_result(image_path, edge_defects):
    return ImageResult(
        image_path=Path(image_path),
        image_size=(300, 300),
        otsu_bounds=(10, 10, 290, 290),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        anomaly_tiles=[],
        edge_defects=edge_defects,
    )


def _capture_put_text(monkeypatch):
    captured = []
    real_put_text = cv2.putText

    def capture(img, text, org, *args, **kwargs):
        captured.append((str(text), org))
        return real_put_text(img, text, org, *args, **kwargs)

    monkeypatch.setattr("capi_inference.cv2.putText", capture)
    return captured


def test_overview_status_is_ng_when_real_edge_defect_exists(tmp_path, monkeypatch):
    captured = _capture_put_text(monkeypatch)
    image_path = tmp_path / "edge_ng.png"
    cv2.imwrite(str(image_path), np.full((300, 300, 3), 128, dtype=np.uint8))

    edge = EdgeDefect(
        side="aoi_edge",
        area=212,
        bbox=(220, 220, 40, 20),
        center=(240, 230),
        max_diff=8,
    )
    result = _make_result(image_path, [edge])

    inferencer = _make_inferencer_for_overview()
    inferencer.visualize_inference_result(image_path, result)

    status_text = [text for text, org in captured if org == (30, 110)]
    assert status_text == ["NG"]


def test_overview_status_stays_ok_for_cv_ok_edge_record(tmp_path, monkeypatch):
    captured = _capture_put_text(monkeypatch)
    image_path = tmp_path / "edge_ok.png"
    cv2.imwrite(str(image_path), np.full((300, 300, 3), 128, dtype=np.uint8))

    edge = EdgeDefect(
        side="aoi_coord_ok",
        area=0,
        bbox=(100, 100, 50, 50),
        center=(125, 125),
        max_diff=0,
        is_cv_ok=True,
    )
    result = _make_result(image_path, [edge])

    inferencer = _make_inferencer_for_overview()
    inferencer.visualize_inference_result(image_path, result)

    status_text = [text for text, org in captured if org == (30, 110)]
    assert status_text == ["OK"]
