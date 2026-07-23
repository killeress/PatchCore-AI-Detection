"""測試 capi_model_registry 模型庫 CRUD 功能。"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock


def test_list_bundles_grouped(tmp_path):
    from capi_database import CAPIDatabase
    from capi_model_registry import list_bundles_grouped

    db = CAPIDatabase(tmp_path / "test.db")
    db.register_model_bundle({
        "machine_id": "GN160", "bundle_path": "model/GN160-20260428",
        "trained_at": "2026-04-28T15:30:45", "panel_count": 5,
        "inner_tile_count": 2400, "edge_tile_count": 900,
        "ng_tile_count": 150, "bundle_size_bytes": 478_000_000, "job_id": "j1",
    })
    db.register_model_bundle({
        "machine_id": "OTHER", "bundle_path": "model/OTHER-20260420",
        "trained_at": "2026-04-20T10:00:00", "panel_count": 3,
        "inner_tile_count": 1200, "edge_tile_count": 400,
        "ng_tile_count": 100, "bundle_size_bytes": 300_000_000, "job_id": "j2",
    })

    grouped = list_bundles_grouped(db)
    assert "GN160" in grouped
    assert "OTHER" in grouped
    assert len(grouped["GN160"]) == 1


def test_discover_and_sync_bundle_from_model_root(tmp_path):
    import json
    import yaml as yaml_mod
    from capi_database import CAPIDatabase
    from capi_model_registry import discover_model_bundles, sync_discovered_bundles

    server_config = tmp_path / "server_config.yaml"
    server_config.write_text(yaml_mod.dump({"training": {"output_root": "model"}}))
    bundle = tmp_path / "model" / "GN156HRAA2L0S-20260529_155800"
    bundle.mkdir(parents=True)
    (bundle / "G0F00000-inner.pt").write_bytes(b"inner")
    (bundle / "G0F00000-edge.pt").write_bytes(b"edge")
    (bundle / "manifest.json").write_text(json.dumps({
        "machine_id": "GN156HRAA2L0S",
        "trained_at": "2026-05-29T08:14:32",
        "trained_with_job_id": "train_GN156HRAA2L0S_1",
        "panel_count": 8,
        "tiles_per_unit": {
            "G0F00000-inner": {"train": 10, "ng": 2},
            "G0F00000-edge": {"train": 5, "ng": 3},
        },
    }), encoding="utf-8")
    (bundle / "thresholds.json").write_text("{}", encoding="utf-8")
    old_root = "/old/capi_ai/model/GN156HRAA2L0S-20260529_155800"
    (bundle / "machine_config.yaml").write_text(yaml_mod.dump({
        "machine_id": "GN156HRAA2L0S",
        "model_mapping": {
            "G0F00000": {
                "inner": f"{old_root}/G0F00000-inner.pt",
                "edge": f"{old_root}/G0F00000-edge.pt",
            },
        },
    }), encoding="utf-8")

    db = CAPIDatabase(tmp_path / "test.db")
    discovery = discover_model_bundles(db, server_config)

    assert discovery["model_root"] == str((tmp_path / "model").resolve())
    assert len(discovery["bundles"]) == 1
    assert len(discovery["bundles"][0]["path_mismatches"]) == 2

    result = sync_discovered_bundles(db, server_config)
    assert len(result["imported"]) == 1
    assert result["imported"][0]["path_rewritten"] is True

    rows = db.list_model_bundles()
    assert len(rows) == 1
    assert rows[0]["machine_id"] == "GN156HRAA2L0S"
    assert rows[0]["panel_count"] == 8
    assert rows[0]["inner_tile_count"] == 10
    assert rows[0]["edge_tile_count"] == 5
    assert rows[0]["ng_tile_count"] == 5
    assert rows[0]["is_active"] == 0
    assert "訓練資料未同步" in rows[0]["notes"]

    repaired = yaml_mod.safe_load((bundle / "machine_config.yaml").read_text(encoding="utf-8"))
    assert repaired["model_mapping"]["G0F00000"]["inner"] == str(
        (bundle / "G0F00000-inner.pt").resolve()
    )
    assert discover_model_bundles(db, server_config)["bundles"] == []


def test_get_bundle_detail_exposes_per_panel_training_modes(tmp_path, monkeypatch):
    import capi_model_registry as registry

    bundle_dir = tmp_path / "model" / "M-20260714"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "manifest.json").write_text(
        '{"machine_id":"M","panel_glass_ids":["PANEL_A","PANEL_B"]}',
        encoding="utf-8",
    )

    db = MagicMock()
    db.get_model_bundle.return_value = {
        "id": 1,
        "machine_id": "M",
        "bundle_path": str(bundle_dir),
        "job_id": "train_M_1",
    }
    db.get_training_job.return_value = {
        "panel_paths": ["D:/train/PANEL_A", "D:/train/PANEL_B"],
        "panel_modes": ["inner_only", "edge_only"],
    }
    monkeypatch.setattr(registry, "get_training_data_summary", lambda *_: {})
    monkeypatch.setattr(registry, "get_pending_change_summary_for_bundle", lambda *_: {})

    detail = registry.get_bundle_detail(db, 1)

    assert detail["training_panel_modes"] == [
        {"panel_name": "PANEL_A", "mode": "inner_only"},
        {"panel_name": "PANEL_B", "mode": "edge_only"},
    ]


def test_activate_bundle_writes_server_config(tmp_path):
    import yaml as yaml_mod
    from capi_database import CAPIDatabase
    from capi_model_registry import activate_bundle, deactivate_bundle

    sc_path = tmp_path / "server_config.yaml"
    sc_path.write_text(yaml_mod.dump({"model_configs": ["configs/capi_3f.yaml"]}))

    bundle_dir = tmp_path / "model" / "GN160-20260428"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "machine_config.yaml").write_text(yaml_mod.dump({"machine_id": "GN160"}))

    db = CAPIDatabase(tmp_path / "test.db")
    bid = db.register_model_bundle({
        "machine_id": "GN160", "bundle_path": str(bundle_dir),
        "trained_at": "2026-04-28T15:30:45", "panel_count": 5,
        "inner_tile_count": 2400, "edge_tile_count": 900,
        "ng_tile_count": 150, "bundle_size_bytes": 478_000_000, "job_id": "j1",
    })

    activate_bundle(db, bid, server_config_path=sc_path)
    sc = yaml_mod.safe_load(sc_path.read_text())
    assert any("machine_config.yaml" in p for p in sc["model_configs"])
    assert db.get_model_bundle(bid)["is_active"] == 1

    deactivate_bundle(db, bid, server_config_path=sc_path)
    sc = yaml_mod.safe_load(sc_path.read_text())
    assert not any("GN160-20260428" in p for p in sc["model_configs"])
    assert db.get_model_bundle(bid)["is_active"] == 0


def test_activate_bundle_deactivates_across_machine_ids(tmp_path):
    """全域單一 active：啟用 B 機種 bundle 時，A 機種 active bundle 也要被下架。"""
    import yaml as yaml_mod
    from capi_database import CAPIDatabase
    from capi_model_registry import activate_bundle

    sc_path = tmp_path / "server_config.yaml"
    # 預設只有 legacy；待會兒兩個 bundle 都 activate 後 legacy 應該還在
    sc_path.write_text(yaml_mod.dump({"model_configs": ["configs/capi_3f.yaml"]}))

    a_dir = tmp_path / "model" / "MACHINE_A-20260428"
    a_dir.mkdir(parents=True)
    (a_dir / "machine_config.yaml").write_text(yaml_mod.dump({"machine_id": "MACHINE_A"}))
    b_dir = tmp_path / "model" / "MACHINE_B-20260429"
    b_dir.mkdir(parents=True)
    (b_dir / "machine_config.yaml").write_text(yaml_mod.dump({"machine_id": "MACHINE_B"}))

    db = CAPIDatabase(tmp_path / "test.db")
    a_id = db.register_model_bundle({
        "machine_id": "MACHINE_A", "bundle_path": str(a_dir),
        "trained_at": "2026-04-28T15:30:45", "panel_count": 5,
        "inner_tile_count": 0, "edge_tile_count": 0, "ng_tile_count": 0,
        "bundle_size_bytes": 0, "job_id": "ja",
    })
    b_id = db.register_model_bundle({
        "machine_id": "MACHINE_B", "bundle_path": str(b_dir),
        "trained_at": "2026-04-29T15:30:45", "panel_count": 5,
        "inner_tile_count": 0, "edge_tile_count": 0, "ng_tile_count": 0,
        "bundle_size_bytes": 0, "job_id": "jb",
    })

    # 先啟用 A
    activate_bundle(db, a_id, server_config_path=sc_path)
    assert db.get_model_bundle(a_id)["is_active"] == 1
    assert db.get_model_bundle(b_id)["is_active"] == 0
    sc = yaml_mod.safe_load(sc_path.read_text())
    assert any("MACHINE_A-20260428" in p for p in sc["model_configs"])

    # 再啟用 B → A 應該被自動下架（跨 machine_id），legacy capi_3f.yaml 仍保留
    activate_bundle(db, b_id, server_config_path=sc_path)
    assert db.get_model_bundle(a_id)["is_active"] == 0
    assert db.get_model_bundle(b_id)["is_active"] == 1
    sc = yaml_mod.safe_load(sc_path.read_text())
    assert not any("MACHINE_A-20260428" in p for p in sc["model_configs"]), \
        "MACHINE_A 的 yaml 應該被自動移出 model_configs"
    assert any("MACHINE_B-20260429" in p for p in sc["model_configs"])
    assert "configs/capi_3f.yaml" in sc["model_configs"], \
        "legacy capi_3f.yaml 不在 DB，不應被踢掉，留作最後 fallback"


def test_activate_bundle_removes_absolute_old_path_for_relative_db_bundle(tmp_path):
    import yaml as yaml_mod
    from capi_database import CAPIDatabase
    from capi_model_registry import activate_bundle

    old_dir = tmp_path / "model" / "OLD"
    new_dir = tmp_path / "model" / "NEW"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old_dir / "machine_config.yaml").write_text("machine_id: OLD\n")
    (new_dir / "machine_config.yaml").write_text("machine_id: NEW\n")

    sc_path = tmp_path / "server_config.yaml"
    sc_path.write_text(yaml_mod.dump({
        "model_configs": [
            "configs/capi_3f.yaml",
            str((old_dir / "machine_config.yaml").resolve()),
        ],
    }))

    db = CAPIDatabase(tmp_path / "test.db")
    old_id = db.register_model_bundle({
        "machine_id": "OLD", "bundle_path": "model/OLD",
        "trained_at": "2026-07-01T00:00:00", "job_id": "old",
    })
    new_id = db.register_model_bundle({
        "machine_id": "NEW", "bundle_path": "model/NEW",
        "trained_at": "2026-07-10T00:00:00", "job_id": "new",
    })
    db.set_bundle_active(old_id, True)

    activate_bundle(db, new_id, server_config_path=sc_path)

    configs = yaml_mod.safe_load(sc_path.read_text())["model_configs"]
    assert configs == [
        "configs/capi_3f.yaml",
        str((new_dir / "machine_config.yaml").resolve()),
    ]
    assert db.get_model_bundle(old_id)["is_active"] == 0
    assert db.get_model_bundle(new_id)["is_active"] == 1


def test_delete_active_bundle_rejected(tmp_path):
    import pytest
    from capi_database import CAPIDatabase
    from capi_model_registry import delete_bundle
    db = CAPIDatabase(tmp_path / "test.db")
    bid = db.register_model_bundle({
        "machine_id": "GN160", "bundle_path": str(tmp_path / "model/x"),
        "trained_at": "2026-04-28T15:30:45", "panel_count": 5,
        "inner_tile_count": 0, "edge_tile_count": 0, "ng_tile_count": 0,
        "bundle_size_bytes": 0, "job_id": "j1",
    })
    db.set_bundle_active(bid, True)
    sc = tmp_path / "server_config.yaml"
    sc.write_text("model_configs: []")

    with pytest.raises(ValueError, match="active"):
        delete_bundle(db, bid, server_config_path=sc)


def test_delete_inactive_bundle_removes_dir(tmp_path):
    from capi_database import CAPIDatabase
    from capi_model_registry import delete_bundle
    db = CAPIDatabase(tmp_path / "test.db")
    bdir = tmp_path / "model" / "y"
    bdir.mkdir(parents=True)
    (bdir / "x.pt").write_bytes(b"x")
    (bdir / "machine_config.yaml").write_text("machine_id: G")
    bid = db.register_model_bundle({
        "machine_id": "G", "bundle_path": str(bdir),
        "trained_at": "2026-04-28T15:30:45", "panel_count": 5,
        "inner_tile_count": 0, "edge_tile_count": 0, "ng_tile_count": 0,
        "bundle_size_bytes": 0, "job_id": "j1",
    })
    sc = tmp_path / "server_config.yaml"
    sc.write_text("model_configs:\n  - configs/capi_3f.yaml\n")
    delete_bundle(db, bid, server_config_path=sc)
    assert not bdir.exists()
    assert db.get_model_bundle(bid) is None


def test_delete_bundle_rejects_path_outside_model_root(tmp_path):
    import pytest
    from capi_database import CAPIDatabase
    from capi_model_registry import delete_bundle

    db = CAPIDatabase(tmp_path / "test.db")
    outside = tmp_path / "outside_bundle"
    outside.mkdir()
    (outside / "machine_config.yaml").write_text("machine_id: G")
    bid = db.register_model_bundle({
        "machine_id": "G", "bundle_path": str(outside),
        "trained_at": "2026-04-28T15:30:45", "panel_count": 5,
        "inner_tile_count": 0, "edge_tile_count": 0, "ng_tile_count": 0,
        "bundle_size_bytes": 0, "job_id": "j1",
    })
    sc = tmp_path / "server_config.yaml"
    sc.write_text("model_configs: []")

    with pytest.raises(ValueError, match="outside model root"):
        delete_bundle(db, bid, server_config_path=sc)
    assert outside.exists()


def test_get_training_data_summary_counts_db_and_disk(tmp_path, monkeypatch):
    from capi_database import CAPIDatabase
    from capi_model_registry import get_training_data_summary
    import capi_model_registry as reg

    db = CAPIDatabase(tmp_path / "test.db")
    job_id = "job_test_summary"
    db.insert_tile_pool(job_id, [
        {"lighting": "G0F00000", "zone": "inner", "source": "ok",
         "source_path": "/x/a.png", "thumb_path": "/x/a.png"},
        {"lighting": "G0F00000", "zone": "edge", "source": "ok",
         "source_path": "/x/b.png", "thumb_path": "/x/b.png"},
        {"lighting": "G0F00000", "zone": None, "source": "ng",
         "source_path": "/x/c.png", "thumb_path": "/x/c.png"},
    ])

    fake_dir = tmp_path / "train_new_thumbs" / job_id
    fake_dir.mkdir(parents=True)
    (fake_dir / "x.png").write_bytes(b"X" * 100)
    (fake_dir / "y.png").write_bytes(b"Y" * 200)
    monkeypatch.setattr(reg, "_training_data_dir", lambda jid: fake_dir if jid == job_id else tmp_path / "missing")

    bundle = {"job_id": job_id}
    s = get_training_data_summary(db, bundle)
    assert s["ok_count"] == 2
    assert s["ng_count"] == 1
    assert s["size_bytes"] == 300


def test_get_training_data_summary_missing(tmp_path, monkeypatch):
    from capi_database import CAPIDatabase
    from capi_model_registry import get_training_data_summary
    import capi_model_registry as reg

    db = CAPIDatabase(tmp_path / "test.db")
    monkeypatch.setattr(reg, "_training_data_dir", lambda jid: tmp_path / "nonexistent" / jid)
    s = get_training_data_summary(db, {"job_id": "nope"})
    assert s["ok_count"] == 0
    assert s["ng_count"] == 0
    assert s["size_bytes"] == 0


def test_get_training_data_summary_no_job_id():
    from capi_model_registry import get_training_data_summary
    s = get_training_data_summary(db=None, bundle={"job_id": ""})
    assert s["ok_count"] == 0
    assert s["ng_count"] == 0
    assert s["size_bytes"] == 0
    assert s["thumb_dir"] == ""


def test_delete_training_data_clears_db_and_dir(tmp_path, monkeypatch):
    from capi_database import CAPIDatabase
    from capi_model_registry import delete_training_data
    import capi_model_registry as reg

    db = CAPIDatabase(tmp_path / "test.db")
    job_id = "job_delete_test"
    bid = db.register_model_bundle({
        "machine_id": "GN160", "bundle_path": str(tmp_path / "model/x"),
        "trained_at": "2026-04-29T15:30:45", "panel_count": 5,
        "inner_tile_count": 0, "edge_tile_count": 0, "ng_tile_count": 0,
        "bundle_size_bytes": 0, "job_id": job_id,
    })
    db.insert_tile_pool(job_id, [
        {"lighting": "G0F00000", "zone": "inner", "source": "ok",
         "source_path": "/x/a.png", "thumb_path": "/x/a.png"},
        {"lighting": "G0F00000", "zone": None, "source": "ng",
         "source_path": "/x/c.png", "thumb_path": "/x/c.png"},
    ])
    fake_dir = tmp_path / "thumbs" / job_id
    fake_dir.mkdir(parents=True)
    (fake_dir / "tile1.png").write_bytes(b"X" * 50)
    (fake_dir / "sub").mkdir()
    (fake_dir / "sub" / "tile2.png").write_bytes(b"Y" * 80)
    monkeypatch.setattr(reg, "_training_data_dir", lambda jid: fake_dir if jid == job_id else tmp_path / "missing")

    result = delete_training_data(db, bid)
    assert result["ok"] is True
    assert result["deleted_tile_rows"] == 2
    assert result["deleted_files"] == 2
    assert result["freed_bytes"] == 130
    # DB 已清空
    assert db.list_tile_pool(job_id) == []
    # 目錄已不存在
    assert not fake_dir.exists()
    # bundle 本身仍存在
    assert db.get_model_bundle(bid) is not None


def test_cleanup_training_job_artifacts_removes_stale_training_data(tmp_path, monkeypatch):
    from capi_database import CAPIDatabase
    from capi_model_registry import cleanup_training_job_artifacts

    monkeypatch.chdir(tmp_path)
    db = CAPIDatabase(tmp_path / "test.db")
    job_id = "stale_cleanup"
    db.create_training_job(job_id, "M", [])
    db.insert_tile_pool(job_id, [
        {
            "lighting": "G0F00000",
            "zone": "inner",
            "source": "ok",
            "source_path": str(tmp_path / "tile.png"),
            "thumb_path": str(tmp_path / "thumb.png"),
        },
    ])

    for root in ("training_staging", "training_runs", "train_new_thumbs"):
        target = tmp_path / ".tmp" / root / job_id
        target.mkdir(parents=True)
        (target / "tile.png").write_bytes(b"tile")

    result = cleanup_training_job_artifacts(db, job_id)

    assert result["ok"] is True
    assert result["deleted_tile_rows"] == 1
    assert result["deleted_files"] == 3
    assert db.list_tile_pool(job_id) == []
    assert not (tmp_path / ".tmp" / "training_staging" / job_id).exists()
    assert not (tmp_path / ".tmp" / "training_runs" / job_id).exists()
    assert not (tmp_path / ".tmp" / "train_new_thumbs" / job_id).exists()


def test_cleanup_training_job_artifacts_preserves_review_thumbnails(tmp_path, monkeypatch):
    from capi_database import CAPIDatabase
    from capi_model_registry import cleanup_training_job_artifacts

    monkeypatch.chdir(tmp_path)
    db = CAPIDatabase(tmp_path / "test.db")
    job_id = "review_cleanup"
    db.create_training_job(job_id, "M", [])
    db.update_training_job_state(job_id, "review")

    for root in ("training_staging", "training_runs", "train_new_thumbs"):
        target = tmp_path / ".tmp" / root / job_id
        target.mkdir(parents=True)
        (target / "tile.png").write_bytes(b"tile")

    result = cleanup_training_job_artifacts(
        db,
        job_id,
        remove_training_data=False,
    )

    assert result["ok"] is True
    assert not (tmp_path / ".tmp" / "training_staging" / job_id).exists()
    assert not (tmp_path / ".tmp" / "training_runs" / job_id).exists()
    assert (tmp_path / ".tmp" / "train_new_thumbs" / job_id / "tile.png").exists()


def test_delete_training_data_no_job_id(tmp_path):
    from capi_database import CAPIDatabase
    from capi_model_registry import delete_training_data
    db = CAPIDatabase(tmp_path / "test.db")
    bid = db.register_model_bundle({
        "machine_id": "G", "bundle_path": str(tmp_path / "model/y"),
        "trained_at": "2026-04-29T15:30:45", "panel_count": 1,
        "inner_tile_count": 0, "edge_tile_count": 0, "ng_tile_count": 0,
        "bundle_size_bytes": 0, "job_id": "",
    })
    result = delete_training_data(db, bid)
    assert result["ok"] is True
    assert result["deleted_tile_rows"] == 0


def test_delete_training_data_unknown_bundle(tmp_path):
    import pytest
    from capi_database import CAPIDatabase
    from capi_model_registry import delete_training_data
    db = CAPIDatabase(tmp_path / "test.db")
    with pytest.raises(ValueError, match="not found"):
        delete_training_data(db, 9999)


def test_update_bundle_notes_persists_free_text(tmp_path):
    import pytest
    from capi_database import CAPIDatabase
    from capi_model_registry import update_bundle_notes

    db = CAPIDatabase(tmp_path / "test.db")
    bid = db.register_model_bundle({
        "machine_id": "GN160", "bundle_path": "model/GN160-20260428",
        "trained_at": "2026-04-28T15:30:45", "panel_count": 5,
        "inner_tile_count": 0, "edge_tile_count": 0, "ng_tile_count": 0,
        "bundle_size_bytes": 0, "job_id": "j1",
    })
    notes = "用途：量產驗證\n<script>text only</script>"

    result = update_bundle_notes(db, bid, notes)

    assert result["ok"] is True
    assert result["notes"] == notes
    assert db.get_model_bundle(bid)["notes"] == notes
    with pytest.raises(ValueError, match="not found"):
        update_bundle_notes(db, 9999, "missing")


def test_update_threshold_writes_yaml_and_json(tmp_path):
    import json as _json
    import yaml as yaml_mod
    from capi_database import CAPIDatabase
    from capi_model_registry import update_threshold

    bundle_dir = tmp_path / "model" / "MX-20260430"
    bundle_dir.mkdir(parents=True)
    yaml_mod_text = yaml_mod.dump({
        "machine_id": "MX",
        "model_mapping": {"G0F00000": {"inner": "x.pt", "edge": "y.pt"}},
        "threshold_mapping": {"G0F00000": {"inner": 0.5, "edge": 0.5}},
    })
    (bundle_dir / "machine_config.yaml").write_text(yaml_mod_text)
    (bundle_dir / "thresholds.json").write_text(
        _json.dumps({"G0F00000": {"inner": 0.5, "edge": 0.5}})
    )

    db = CAPIDatabase(tmp_path / "test.db")
    bid = db.register_model_bundle({
        "machine_id": "MX", "bundle_path": str(bundle_dir),
        "trained_at": "2026-04-30T10:00:00", "panel_count": 1,
        "inner_tile_count": 0, "edge_tile_count": 0, "ng_tile_count": 0,
        "bundle_size_bytes": 0, "job_id": "j1",
    })

    res = update_threshold(db, bid, lighting="G0F00000", zone="inner", value=0.78)
    assert res["ok"] is True
    cfg = yaml_mod.safe_load((bundle_dir / "machine_config.yaml").read_text())
    assert cfg["threshold_mapping"]["G0F00000"]["inner"] == 0.78
    assert cfg["threshold_mapping"]["G0F00000"]["edge"] == 0.5    # 不動其他
    thr_json = _json.loads((bundle_dir / "thresholds.json").read_text())
    assert thr_json["G0F00000"]["inner"] == 0.78


def test_update_threshold_preserves_machine_config_comments(tmp_path):
    import json as _json
    import yaml as yaml_mod
    from capi_database import CAPIDatabase
    from capi_model_registry import update_threshold

    bundle_dir = tmp_path / "model" / "MX-20260430"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "machine_config.yaml").write_text("""\
# ============================================================================
# CAPI 新架構 bundle 推論設定（machine_config.yaml）
# 由 capi_train_new.write_machine_config_yaml 自動產生
# ============================================================================

machine_id: MX

# === 模型映射（lighting → inner/edge 模型路徑 + threshold）===
model_mapping:
  G0F00000:
    inner: x.pt
    edge: y.pt

threshold_mapping:
  G0F00000:
    inner: 0.5  # 保留 inner 註解
    edge: 0.5

# === OMIT 過曝偵測 ===
omit_overexposure_mean_threshold: 82
""", encoding="utf-8")
    (bundle_dir / "thresholds.json").write_text(
        _json.dumps({"G0F00000": {"inner": 0.5, "edge": 0.5}})
    )

    db = CAPIDatabase(tmp_path / "test.db")
    bid = db.register_model_bundle({
        "machine_id": "MX", "bundle_path": str(bundle_dir),
        "trained_at": "2026-04-30T10:00:00", "panel_count": 1,
        "inner_tile_count": 0, "edge_tile_count": 0, "ng_tile_count": 0,
        "bundle_size_bytes": 0, "job_id": "j1",
    })

    update_threshold(db, bid, lighting="G0F00000", zone="edge", value=0.66)

    yaml_text = (bundle_dir / "machine_config.yaml").read_text(encoding="utf-8")
    assert "# CAPI 新架構 bundle 推論設定" in yaml_text
    assert "# === OMIT 過曝偵測 ===" in yaml_text
    assert "inner: 0.5  # 保留 inner 註解" in yaml_text
    cfg = yaml_mod.safe_load(yaml_text)
    assert cfg["threshold_mapping"]["G0F00000"]["inner"] == 0.5
    assert cfg["threshold_mapping"]["G0F00000"]["edge"] == 0.66


def test_update_threshold_rejects_bad_zone(tmp_path):
    import pytest, yaml as yaml_mod
    from capi_database import CAPIDatabase
    from capi_model_registry import update_threshold

    bundle_dir = tmp_path / "model" / "MX-20260430"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "machine_config.yaml").write_text(yaml_mod.dump({"threshold_mapping": {}}))
    db = CAPIDatabase(tmp_path / "test.db")
    bid = db.register_model_bundle({
        "machine_id": "MX", "bundle_path": str(bundle_dir),
        "trained_at": "2026-04-30T10:00:00", "panel_count": 1,
        "inner_tile_count": 0, "edge_tile_count": 0, "ng_tile_count": 0,
        "bundle_size_bytes": 0, "job_id": "j1",
    })
    with pytest.raises(ValueError, match="zone"):
        update_threshold(db, bid, lighting="G0F00000", zone="middle", value=0.5)


def test_update_threshold_rejects_unknown_unit(tmp_path):
    import pytest, yaml as yaml_mod
    from capi_database import CAPIDatabase
    from capi_model_registry import update_threshold

    bundle_dir = tmp_path / "model" / "MX-20260430"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "machine_config.yaml").write_text(yaml_mod.dump({
        "threshold_mapping": {"G0F00000": {"inner": 0.5, "edge": 0.5}}
    }))
    db = CAPIDatabase(tmp_path / "test.db")
    bid = db.register_model_bundle({
        "machine_id": "MX", "bundle_path": str(bundle_dir),
        "trained_at": "2026-04-30T10:00:00", "panel_count": 1,
        "inner_tile_count": 0, "edge_tile_count": 0, "ng_tile_count": 0,
        "bundle_size_bytes": 0, "job_id": "j1",
    })
    with pytest.raises(ValueError, match="找不到"):
        update_threshold(db, bid, lighting="UNKNOWN", zone="inner", value=0.5)


def test_export_zip_streams(tmp_path):
    import zipfile, io
    from capi_model_registry import export_bundle_zip
    bundle = tmp_path / "model" / "GN160-20260428"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text('{"machine_id":"GN160"}')
    (bundle / "machine_config.yaml").write_text("machine_id: GN160")
    (bundle / "G0F00000-inner.pt").write_bytes(b"\x00" * 1024)

    zip_bytes = export_bundle_zip(bundle, machine_id="GN160")
    assert isinstance(zip_bytes, bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = z.namelist()
        assert any("manifest.json" in n for n in names)
        assert any("machine_config.yaml" in n for n in names)
        assert any(n.endswith(".pt") for n in names)
        assert any("README.txt" in n for n in names)
