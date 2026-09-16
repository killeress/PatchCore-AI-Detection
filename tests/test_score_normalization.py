import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("anomalib.post_processing")
from anomalib.data import InferenceBatch
from anomalib.post_processing import PostProcessor

from capi_normalization_config import OK_MAX_NORMALIZATION
from capi_patchcore_post_processor import OKMaxPostProcessor


def calibrated_processor():
    processor = OKMaxPostProcessor()
    processor.image_min.fill_(10)
    processor.image_max.fill_(20)
    processor.pixel_min.fill_(2)
    processor.pixel_max.fill_(10)
    processor._image_threshold.fill_(20)
    processor._pixel_threshold.fill_(10)
    return processor


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_positive_distances_keep_order_and_values_above_one(dtype):
    processor = calibrated_processor()
    raw = torch.tensor([0., .001, 5., 10., 15., 20., 30., 60.], dtype=dtype)
    maps = raw.reshape(-1, 1, 1)
    result = processor(InferenceBatch(pred_score=raw, anomaly_map=maps))
    torch.testing.assert_close(result.pred_score, raw.float() / 40)
    torch.testing.assert_close(result.anomaly_map, maps.float() / 20)
    assert result.pred_score.dtype == torch.float32
    assert (torch.diff(result.pred_score) > 0).all()
    assert result.pred_score[0] == 0
    assert result.pred_score[-1] == 1.5
    assert result.anomaly_map[-1].item() == 3


@pytest.mark.parametrize("invalid", [0., -1., float("nan"), float("inf")])
def test_invalid_calibration_is_an_error_instead_of_zero_scores(invalid):
    processor = calibrated_processor()
    processor.image_max.fill_(invalid)
    with pytest.raises(ValueError, match="OK"):
        processor.validate_calibration()
    with pytest.raises(ValueError, match="OK"):
        processor(InferenceBatch(pred_score=torch.tensor([5.])))


def test_constant_nonzero_ok_scores_are_valid_without_min_max_division():
    processor = calibrated_processor()
    processor.image_min.fill_(20)
    processor.validate_calibration()
    result = processor(InferenceBatch(pred_score=torch.tensor([5., 20.])))
    torch.testing.assert_close(result.pred_score, torch.tensor([.125, .5]))


def test_torch_save_reload_and_mark_scoring_use_the_same_scale(tmp_path):
    from types import SimpleNamespace
    from capi_inference import CAPIInferencer

    path = tmp_path / "model.pt"
    torch.save({"post_processor": calibrated_processor()}, path)
    loaded = torch.load(path, weights_only=False)["post_processor"]
    raw = torch.tensor([10.])
    result = loaded(InferenceBatch(pred_score=raw, anomaly_map=raw.reshape(1, 1, 1)))
    assert loaded.normalization_mode == OK_MAX_NORMALIZATION
    assert result.pred_score.item() == .25
    assert result.anomaly_map.item() == .5
    assert CAPIInferencer._normalize_patchcore_image_score(SimpleNamespace(post_processor=loaded), raw) == .25
    inferencer = SimpleNamespace(
        model=SimpleNamespace(post_processor=loaded),
        predict=lambda _: loaded(InferenceBatch(pred_score=raw, anomaly_map=raw.reshape(1, 1, 1))),
    )
    worker = CAPIInferencer.__new__(CAPIInferencer)
    diagnostic = worker._capture_model_score_diagnostics(inferencer, None, result)
    assert diagnostic["model_normalization_mode"] == OK_MAX_NORMALIZATION
    assert diagnostic["model_pixel_max"] == 10
    assert diagnostic["raw_model_score"] == 10
    assert diagnostic["normalized_anomaly_map_max"] == .5
    assert loaded.enable_normalization is True
    loaded.enable_normalization = False
    assert loaded(InferenceBatch(pred_score=raw)).pred_score.item() == 10


def test_old_post_processor_still_uses_legacy_clipping():
    processor = PostProcessor()
    processor.image_min.fill_(10)
    processor.image_max.fill_(20)
    processor._image_threshold.fill_(20)
    assert processor(InferenceBatch(pred_score=torch.tensor([10.]))).pred_score.item() == 0


def test_linear_debug_does_not_report_low_scores_as_clamped():
    from capi_inference import score_normalization_diagnostic
    diagnostic = score_normalization_diagnostic(5., 10., 20., 20., True, OK_MAX_NORMALIZATION)
    assert diagnostic["normalization_available"]
    assert diagnostic["normalization_zero_boundary"] == 0
    assert diagnostic["normalization_zero_clamped"] is False


def test_heatmap_display_preserves_constant_positive_maps_and_does_not_wrap():
    from capi_heatmap import heatmap_to_uint8
    values = np.array([[0., .25, .5, 1., 1.5, 3.]], dtype=np.float32)
    original = values.copy()
    assert heatmap_to_uint8(values).tolist() == [[0, 63, 127, 255, 255, 255]]
    assert (heatmap_to_uint8(np.full((4, 4), .5)) == 127).all()
    np.testing.assert_array_equal(values, original)
