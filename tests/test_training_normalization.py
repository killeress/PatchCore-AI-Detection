"""Exercise OK-only calibration with real Anomalib data and post-processing.

Use a deterministic, inexpensive score function instead of downloading/training
a backbone. Bomb changes must leave image AND pixel calibration unchanged.
"""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


@pytest.mark.parametrize("separate_ok", [False, True])
def test_bomb_changes_do_not_change_normalization(tmp_path, monkeypatch, separate_ok):
    data = pytest.importorskip("anomalib.data")
    pytest.importorskip("anomalib.post_processing")
    from capi_patchcore_post_processor import OKMaxPostProcessor
    import torch

    from capi_train_new import train_one_patchcore

    staging = tmp_path / "staging"
    train = staging / "train"
    bombs = staging / "test" / "anormal"
    train.mkdir(parents=True)
    bombs.mkdir(parents=True)

    def write_image(path, brightness):
        assert cv2.imwrite(str(path), np.full((16, 16, 3), brightness, np.uint8))

    for index in range(30):
        write_image(train / f"ok_{index:02d}.png", 10 + index * 2)
    if separate_ok:
        calibration = staging / "test" / "normal"
        calibration.mkdir(parents=True)
        for index in range(3):
            write_image(calibration / f"held_out_{index}.png", 40 + index * 10)

    snapshots = []

    def real_folder(**kwargs):
        # Stable split and no worker processes keep this integration test small.
        return data.Folder(**{**kwargs, "num_workers": 0, "seed": 42})

    class ScoreModel:
        def __init__(self, **_kwargs):
            self.model = SimpleNamespace()
            self.post_processor = _kwargs["post_processor"]

        @staticmethod
        def configure_pre_processor(image_size):
            return None

    class CalibrationEngine:
        def __init__(self, *, default_root_dir, **_kwargs):
            self.root = Path(default_root_dir)

        def fit(self, *, datamodule, model):
            datamodule.setup()
            train_paths = set(datamodule.train_data.samples.image_path)
            val_paths = set(datamodule.val_data.samples.image_path)
            assert train_paths.isdisjoint(val_paths)
            assert len(train_paths) == (30 if separate_ok else 24)
            assert len(val_paths) == (3 if separate_ok else 6)
            assert set(datamodule.val_data.samples.label_index) == {0}
            assert all(Path(p).parent == (calibration if separate_ok else train) for p in val_paths)

            processor = model.post_processor
            for batch in datamodule.val_dataloader():
                # Stand in for raw PatchCore outputs; all calibration labels
                # and actual image inputs come from the real Folder loader.
                batch.pred_score = batch.image.flatten(1).mean(1) * 100 + 1
                batch.anomaly_map = batch.image[:, 0] * 80 + 1
                processor.on_validation_batch_end(None, None, batch)
            processor.on_validation_epoch_end(None, None)

        def export(self, *, model, **_kwargs):
            processor = model.post_processor
            fields = ("image_min", "image_max", "image_threshold",
                      "pixel_min", "pixel_max", "pixel_threshold")
            stats = torch.stack([getattr(processor, name).detach().clone() for name in fields])
            assert torch.isfinite(stats).all()
            assert processor.image_max > processor.image_min
            assert processor.pixel_max > processor.pixel_min
            assert processor.image_threshold == processor.image_max
            snapshots.append(stats)
            assert model.model.training_provenance["normalization_source"] == "ok_only"
            out = self.root / "weights" / "torch" / "model.pt"
            out.parent.mkdir(parents=True)
            out.write_bytes(b"calibration-tested")

    monkeypatch.setattr("capi_train_new._import_anomalib", lambda: (
        real_folder, ScoreModel, CalibrationEngine,
        SimpleNamespace(TORCH="torch"), "same_as_test", OKMaxPostProcessor,
    ))

    for run in range(3):
        if run == 1:
            write_image(bombs / "bomb.png", 255)
        elif run == 2:
            write_image(bombs / "bomb.png", 0)
            for index in range(5):
                write_image(bombs / f"extra_{index}.png", 200 + index)
        stats = {}
        train_one_patchcore(staging, tmp_path / f"run_{run}", "G0F00000-inner",
                            experiment_stats_out=stats)
        assert stats["normalization_source"] == "ok_only"

    for current in snapshots[1:]:
        torch.testing.assert_close(current, snapshots[0], rtol=0, atol=0)
