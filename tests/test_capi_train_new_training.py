import os
import platform
import tempfile
from pathlib import Path

import pytest


def test_apply_user_training_params_none_is_noop():
    from capi_train_new import TrainingConfig, apply_user_training_params
    cfg = TrainingConfig(
        machine_id="M", panel_paths=[], over_review_root=Path("/r"),
    )
    snapshot = (cfg.batch_size, cfg.coreset_ratio, cfg.max_epochs, cfg.inner_panels)
    apply_user_training_params(cfg, None)
    apply_user_training_params(cfg, {})
    assert (cfg.batch_size, cfg.coreset_ratio, cfg.max_epochs, cfg.inner_panels) == snapshot


def test_apply_user_training_params_overrides_match_keys():
    from capi_train_new import TrainingConfig, apply_user_training_params
    cfg = TrainingConfig(
        machine_id="M", panel_paths=[], over_review_root=Path("/r"),
    )
    apply_user_training_params(cfg, {
        "batch_size": 16, "coreset_ratio": 0.05,
        "max_epochs": 2, "inner_panels": 4,
    })
    assert cfg.batch_size == 16
    assert cfg.coreset_ratio == 0.05
    assert cfg.max_epochs == 2
    assert cfg.inner_panels == 4


def test_apply_user_training_params_unknown_key_raises():
    """DB 內的髒資料不能 silent fall-through 到訓練。"""
    from capi_train_new import TrainingConfig, apply_user_training_params
    cfg = TrainingConfig(
        machine_id="M", panel_paths=[], over_review_root=Path("/r"),
    )
    with pytest.raises(ValueError, match="unknown user training params"):
        apply_user_training_params(cfg, {"learning_rate": 0.01})


def test_apply_user_training_params_logs_only_when_applied():
    from capi_train_new import TrainingConfig, apply_user_training_params
    cfg = TrainingConfig(
        machine_id="M", panel_paths=[], over_review_root=Path("/r"),
    )
    msgs = []
    apply_user_training_params(cfg, None, log_fn=msgs.append)
    assert msgs == []
    apply_user_training_params(cfg, {"batch_size": 4}, log_fn=msgs.append)
    assert len(msgs) == 1 and "batch_size" in msgs[0]


def test_compute_unit_metrics_normal_case():
    from capi_train_new import compute_unit_metrics
    ng_scores = [0.4 + i * 0.02 for i in range(30)]
    train_scores = [0.1 + i * 0.004 for i in range(100)]  # 0.1 ~ 0.496
    m = compute_unit_metrics(0.5, ng_scores, threshold=0.6, train_scores=train_scores)
    assert m["train_max"] == 0.5
    assert m["ng_count"] == 30
    assert m["threshold"] == 0.6
    assert m["ng_min"] == 0.4
    assert m["ng_max"] == round(0.4 + 29 * 0.02, 4)
    assert m["ng_median"] == 0.7
    assert m["separation"] == 0.2
    assert m["ng_caught_count"] == 20
    assert abs(m["ng_caught_rate"] - 20/30) < 1e-3
    # AUROC：所有 ng (>=0.4) >= 所有 train (<=0.496)，分得很開
    assert m["auroc"] >= 0.95
    assert m["auroc_grade"] == "excellent"


def test_compute_unit_metrics_no_ng():
    from capi_train_new import compute_unit_metrics
    m = compute_unit_metrics(0.5, [], threshold=0.525, train_scores=[0.1, 0.2])
    assert m["ng_count"] == 0
    assert m["ng_caught_count"] == 0
    assert m["ng_min"] is None
    assert m["separation"] is None
    assert m["auroc"] is None
    assert m["auroc_grade"] == "n/a"


def test_compute_unit_metrics_negative_separation():
    from capi_train_new import compute_unit_metrics
    ng_scores = [0.2, 0.25, 0.3, 0.35]
    train_scores = [0.4, 0.45, 0.5]
    m = compute_unit_metrics(0.5, ng_scores, threshold=0.525, train_scores=train_scores)
    assert m["separation"] < 0
    assert m["ng_caught_count"] == 0
    assert m["auroc"] == 0.0  # ng 全部低於 train，AUROC 為 0


def test_compute_auroc_perfect_separation():
    from capi_train_new import _compute_auroc
    # NG 分數全部 > train 分數 → AUROC = 1
    auroc = _compute_auroc(train_scores=[0.1, 0.2, 0.3], ng_scores=[0.7, 0.8, 0.9])
    assert auroc == 1.0


def test_compute_auroc_random():
    from capi_train_new import _compute_auroc
    # 完全交錯（均勻打散）→ AUROC ≈ 0.5
    auroc = _compute_auroc(
        train_scores=[0.1, 0.3, 0.5, 0.7, 0.9],
        ng_scores=[0.2, 0.4, 0.6, 0.8, 1.0],
    )
    # 每個 ng 都剛好「贏」一半略多 train，AUROC 高於 0.5 但接近
    assert 0.5 < auroc < 0.7


def test_compute_auroc_inverted():
    from capi_train_new import _compute_auroc
    # NG 比 train 還低 → AUROC = 0（最差）
    auroc = _compute_auroc(train_scores=[0.7, 0.8, 0.9], ng_scores=[0.1, 0.2, 0.3])
    assert auroc == 0.0


def test_compute_auroc_ties():
    from capi_train_new import _compute_auroc
    # 完全打平 → AUROC = 0.5
    auroc = _compute_auroc(train_scores=[0.5, 0.5], ng_scores=[0.5, 0.5])
    assert auroc == 0.5


def test_compute_auroc_no_data():
    from capi_train_new import _compute_auroc
    assert _compute_auroc([], [0.5]) is None
    assert _compute_auroc([0.5], []) is None


def test_auroc_grade_ranges():
    from capi_train_new import _auroc_grade
    assert _auroc_grade(0.99) == "excellent"
    assert _auroc_grade(0.95) == "excellent"
    assert _auroc_grade(0.90) == "good"
    assert _auroc_grade(0.85) == "good"
    assert _auroc_grade(0.75) == "fair"
    assert _auroc_grade(0.60) == "poor"
    assert _auroc_grade(0.50) == "fail"
    assert _auroc_grade(None) == "n/a"


def test_compute_unit_metrics_with_auroc():
    from capi_train_new import compute_unit_metrics
    m = compute_unit_metrics(
        train_max=0.5,
        ng_scores=[0.7, 0.8, 0.9],
        threshold=0.6,
        train_scores=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    assert m["auroc"] == 1.0
    assert m["auroc_grade"] == "excellent"
    assert m["train_count_eval"] == 5


def test_compute_unit_metrics_empty_train_scores():
    from capi_train_new import compute_unit_metrics
    # 空 train_scores → AUROC = None / grade = n/a，其他仍正常
    m = compute_unit_metrics(0.5, [0.6, 0.7, 0.8], 0.55, train_scores=[])
    assert m["auroc"] is None
    assert m["auroc_grade"] == "n/a"
    assert m["ng_caught_count"] == 3


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


def test_repair_hf_snapshot_symlinks_restores_zero_byte_weight(tmp_path):
    from capi_train_new import _has_valid_hf_snapshot_weights, _repair_hf_snapshot_symlinks

    model_dir = tmp_path / "huggingface" / "models--timm--wide_resnet50_2.racm_in1k"
    blob = model_dir / "blobs" / "03b71d65"
    snapshot = model_dir / "snapshots" / "30f73ace" / "model.safetensors"
    blob.parent.mkdir(parents=True)
    snapshot.parent.mkdir(parents=True)
    blob.write_bytes(b"x" * (1024 * 1024 + 1))
    snapshot.write_bytes(b"")

    logs = []
    _repair_hf_snapshot_symlinks(tmp_path / "huggingface", logs.append)

    assert snapshot.read_bytes() == blob.read_bytes()
    assert _has_valid_hf_snapshot_weights(model_dir)
    assert any("修復 HF cache 權重檔" in line for line in logs)


def test_make_local_only_hf_download_forces_local_files_only():
    from capi_train_new import _make_local_only_hf_download

    calls = []

    def fake_download(*args, **kwargs):
        calls.append(kwargs.copy())
        return "cached"

    wrapped = _make_local_only_hf_download(fake_download)

    assert wrapped("repo", "model.safetensors", local_files_only=False) == "cached"
    assert calls == [{"local_files_only": True}]
    assert _make_local_only_hf_download(wrapped) is wrapped


def test_setup_offline_env_patches_imported_hf_constants(tmp_path, monkeypatch):
    from capi_train_new import _setup_offline_env
    import huggingface_hub.constants as hf_constants

    cache_dir = tmp_path / "torch_hub_cache"
    model_dir = cache_dir / "huggingface" / "models--timm--wide_resnet50_2.racm_in1k"
    snapshot = model_dir / "snapshots" / "30f73ace" / "model.safetensors"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"x" * (1024 * 1024 + 1))

    preflight_calls = []
    monkeypatch.setattr(
        "capi_train_new._preflight_timm_backbone",
        lambda log, cache_dir=None: preflight_calls.append(cache_dir),
    )

    _setup_offline_env(cache_dir, lambda msg: None)

    hf_cache = cache_dir.resolve() / "huggingface"
    assert os.environ["HF_HUB_CACHE"] == str(hf_cache)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert hf_constants.HF_HUB_CACHE == str(hf_cache)
    assert hf_constants.HF_HUB_OFFLINE is True
    assert preflight_calls == [hf_cache]


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


def test_run_training_pipeline_orchestrates_10_units(tmp_path, monkeypatch):
    from capi_train_new import run_training_pipeline, TrainingConfig, LIGHTINGS, ZONES

    # 準備假 tile pool
    pool = []
    for lighting in LIGHTINGS:
        for zone in ZONES:
            for i in range(50):
                p = tmp_path / "tiles" / f"{lighting}_{zone}_{i}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")
                pool.append({
                    "id": len(pool), "lighting": lighting, "zone": zone,
                    "source": "ok", "source_path": str(p), "decision": "accept",
                })
        for i in range(30):
            p = tmp_path / "tiles_ng" / f"{lighting}_ng_{i}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
            pool.append({
                "id": len(pool), "lighting": lighting, "zone": None,
                "source": "ng", "source_path": str(p), "decision": "accept",
            })

    class MockDB:
        def list_tile_pool(self, job_id, **filt):
            res = list(pool)
            for k, v in filt.items():
                res = [r for r in res if r.get(k) == v]
            return res

    db = MockDB()

    trained_units = []
    def fake_train(staging_dir, run_root, unit_label, cfg=None, log=None):
        trained_units.append(unit_label)
        out = run_root / "weights" / "torch" / "model.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fakemodel")
        return out
    monkeypatch.setattr("capi_train_new.train_one_patchcore", fake_train)
    monkeypatch.setattr("capi_train_new._calibrate_from_model", lambda *a, **kw: (0.5, [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]))

    # bypass backbone check for test
    monkeypatch.setattr("capi_train_new._setup_offline_env", lambda *a, **kw: None)

    cfg = TrainingConfig(
        machine_id="GN160TEST", panel_paths=[Path("p1")],
        over_review_root=tmp_path / "or",
        output_root=tmp_path / "model",
    )
    bundle_dir = run_training_pipeline(
        job_id="j1", cfg=cfg, db=db,
        gpu_lock=__import__("threading").Lock(),
        log=lambda m: None,
    )

    assert bundle_dir.exists()
    # 路徑格式：<machine_id>-<YYYYMMDD_HHMMSS>，job_id 在 manifest 內
    assert bundle_dir.name.startswith("GN160TEST-")
    # job_id 寫入 manifest 而非路徑
    import json as _json
    manifest = _json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["trained_with_job_id"] == "j1"
    assert len(trained_units) == 10
    # 應有 10 個 .pt
    pts = list(bundle_dir.glob("*.pt"))
    assert len(pts) == 10
    # 應有 manifest + thresholds + yaml
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "thresholds.json").exists()
    assert (bundle_dir / "machine_config.yaml").exists()


def test_run_training_pipeline_requires_all_units(tmp_path, monkeypatch):
    from capi_train_new import run_training_pipeline, TrainingConfig, LIGHTINGS, ZONES

    pool = []
    for lighting in LIGHTINGS:
        for zone in ZONES:
            for i in range(50):
                p = tmp_path / "tiles" / f"{lighting}_{zone}_{i}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")
                pool.append({
                    "id": len(pool), "lighting": lighting, "zone": zone,
                    "source": "ok", "source_path": str(p), "decision": "accept",
                })
        for i in range(30):
            p = tmp_path / "tiles_ng" / f"{lighting}_ng_{i}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
            pool.append({
                "id": len(pool), "lighting": lighting, "zone": None,
                "source": "ng", "source_path": str(p), "decision": "accept",
            })

    class MockDB:
        def list_tile_pool(self, job_id, **filt):
            res = list(pool)
            for k, v in filt.items():
                res = [r for r in res if r.get(k) == v]
            return res

    def fake_train(staging_dir, run_root, unit_label, cfg=None, log=None):
        if unit_label == "G0F00000-edge":
            raise RuntimeError("edge model failed")
        out = run_root / "weights" / "torch" / "model.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fakemodel")
        return out

    monkeypatch.setattr("capi_train_new.train_one_patchcore", fake_train)
    monkeypatch.setattr("capi_train_new._calibrate_from_model", lambda *a, **kw: (0.5, [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]))
    monkeypatch.setattr("capi_train_new._setup_offline_env", lambda *a, **kw: None)

    output_root = tmp_path / "model"
    cfg = TrainingConfig(
        machine_id="GN160TEST", panel_paths=[Path("p1")],
        over_review_root=tmp_path / "or",
        output_root=output_root,
    )

    with pytest.raises(RuntimeError, match="G0F00000-edge"):
        run_training_pipeline(
            job_id="j_partial", cfg=cfg, db=MockDB(),
            gpu_lock=__import__("threading").Lock(),
            log=lambda m: None,
        )

    assert not list(output_root.iterdir())
