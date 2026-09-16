"""OK-calibrated linear scaling, persisted inside exported Torch models."""

import torch
from anomalib.post_processing import PostProcessor

from capi_normalization_config import OK_MAX_NORMALIZATION


class OKMaxPostProcessor(PostProcessor):
    """Map each OK maximum to 0.5 without subtracting or clipping scores.

    Image scores use image_max; anomaly-map pixels use pixel_max. Calibration
    data is supplied by the OK-only Folder in capi_train_new.
    """

    normalization_mode = OK_MAX_NORMALIZATION

    @staticmethod
    def _validate_maximum(maximum):
        if maximum.numel() != 1 or not bool(torch.isfinite(maximum).all()) or float(maximum) <= 0:
            raise ValueError("OK 正規化校準最高值必須為有限正數，請檢查 OK 校準圖並重新訓練")

    def validate_calibration(self):
        self._validate_maximum(self.image_max)
        self._validate_maximum(self.pixel_max)

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        self.validate_calibration()
        # The internal decision center has the same meaning for images/pixels,
        # including datasets without pixel-level ground truth masks.
        self._image_threshold.copy_(self.image_max)
        self._pixel_threshold.copy_(self.pixel_max)

    @staticmethod
    def _normalize(preds, norm_min, norm_max, threshold):
        del norm_min, threshold
        if preds is None:
            return None
        OKMaxPostProcessor._validate_maximum(norm_max)
        # Compute in float32 even for fp16 backbones, retaining small distances
        # and values above one for downstream scoring/masks.
        return (preds.float() / norm_max.float()) * 0.5
