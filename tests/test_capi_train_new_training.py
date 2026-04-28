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
