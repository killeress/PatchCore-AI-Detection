import os
import platform
import tempfile
from pathlib import Path


def test_stage_dataset_creates_links_or_copies(tmp_path):
    from capi_train_new import stage_dataset

    # 假設 train tile 路徑列表
    train_dir = tmp_path / "src_train"
    train_dir.mkdir()
    for i in range(5):
        (train_dir / f"t{i}.png").write_bytes(b"x")
    train_paths = list(train_dir.glob("*.png"))

    ng_dir = tmp_path / "src_ng"
    ng_dir.mkdir()
    for i in range(3):
        (ng_dir / f"n{i}.png").write_bytes(b"y")
    ng_paths = list(ng_dir.glob("*.png"))

    staging = tmp_path / "staging" / "G0F-inner"
    stage_dataset(staging, train_paths, ng_paths)

    assert (staging / "train").exists()
    assert (staging / "test" / "anormal").exists()
    train_files = list((staging / "train").glob("*"))
    ng_files = list((staging / "test" / "anormal").glob("*"))
    assert len(train_files) == 5
    assert len(ng_files) == 3


def test_train_one_patchcore_smoke(tmp_path, monkeypatch):
    """smoke test：mock anomalib，確認 orchestration 順序正確。"""
    from capi_train_new import train_one_patchcore

    staging = tmp_path / "staging"
    (staging / "train").mkdir(parents=True)
    (staging / "test" / "anormal").mkdir(parents=True)

    calls = []

    class FakeEngine:
        def __init__(self, *a, **kw):
            calls.append(("Engine.__init__", kw))
            self._default_root_dir = kw.get("default_root_dir", "")

        def fit(self, **kw):
            calls.append(("fit", kw))

        def export(self, **kw):
            calls.append(("export", kw))
            # 從 Engine.__init__ 取得 default_root_dir 建立 fake model.pt
            run_root = Path(self._default_root_dir)
            (run_root / "weights" / "torch").mkdir(parents=True, exist_ok=True)
            (run_root / "weights" / "torch" / "model.pt").write_bytes(b"fake")

    class FakePatchcore:
        def __init__(self, *a, **kw):
            calls.append(("Patchcore", kw))

        @staticmethod
        def configure_pre_processor(image_size):
            return None

        pre_processor = None

    class FakeFolder:
        def __init__(self, *a, **kw):
            calls.append(("Folder", kw))

    monkeypatch.setattr("capi_train_new._import_anomalib", lambda: (
        FakeFolder,
        FakePatchcore,
        FakeEngine,
        type("ExportType", (), {"TORCH": "torch"}),
        "same_as_test",
    ))

    out = train_one_patchcore(
        staging_dir=staging,
        run_root=tmp_path / "run",
        unit_label="G0F-inner",
    )
    call_names = [c[0] for c in calls]
    assert "fit" in call_names
    assert "export" in call_names
    # fit 必須在 export 之前
    assert call_names.index("fit") < call_names.index("export")
    # 回傳路徑要存在
    assert out.exists()
    assert out.name == "model.pt"


def test_calibrate_threshold_uses_p10_and_train_max():
    from capi_train_new import calibrate_threshold
    # 假設 NG scores 與 train_max
    ng_scores = sorted([0.5, 0.55, 0.6, 0.7, 0.8, 0.9])  # P10 ≈ 0.5
    threshold = calibrate_threshold(ng_scores=ng_scores, train_max_score=0.4)
    # max(P10=0.5, 0.4*1.05=0.42) → 0.5
    assert threshold == 0.5
    # train_max_score 較高的情況
    threshold = calibrate_threshold(ng_scores=ng_scores, train_max_score=0.6)
    # max(0.5, 0.63) → 0.63
    assert abs(threshold - 0.63) < 1e-6


def test_write_manifest_yaml(tmp_path):
    from capi_train_new import write_manifest, write_machine_config_yaml
    bundle = tmp_path / "GN160-20260428"
    bundle.mkdir()

    write_manifest(bundle, {
        "machine_id": "GN160", "trained_at": "2026-04-28T15:30:45",
        "trained_with_job_id": "j1", "panel_count": 5,
        "panel_glass_ids": ["YQ21KU218E45"],
        "edge_threshold_px": 768,
        "patchcore_params": {"batch_size": 8, "image_size": [512, 512],
                             "coreset_ratio": 0.1, "max_epochs": 1},
        "tiles_per_unit": {"G0F00000-inner": {"train": 480, "ng": 30}},
        "model_files": {"G0F00000-inner": {"path": "G0F00000-inner.pt", "size_bytes": 47340000}},
    })

    import json as _json
    m = _json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert m["machine_id"] == "GN160"
    assert m["version_schema"] == 1

    write_machine_config_yaml(bundle, "GN160", {
        "G0F00000": {"inner": 0.62, "edge": 0.71},
    })
    import yaml
    y = yaml.safe_load((bundle / "machine_config.yaml").read_text(encoding="utf-8"))
    assert y["machine_id"] == "GN160"
    assert y["model_mapping"]["G0F00000"]["inner"].endswith("G0F00000-inner.pt")
    assert y["threshold_mapping"]["G0F00000"]["inner"] == 0.62
