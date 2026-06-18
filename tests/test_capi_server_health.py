import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_health_check_new_arch_resolves_relative_model_paths(tmp_path):
    from capi_config import CAPIConfig
    from capi_server import CAPIServer

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "inner.pt").write_bytes(b"x")
    (model_dir / "edge.pt").write_bytes(b"x")

    cfg = CAPIConfig(
        machine_id="M",
        is_new_architecture=True,
        model_mapping={"G0F00000": {"inner": "model/inner.pt", "edge": "model/edge.pt"}},
    )
    server = CAPIServer.__new__(CAPIServer)
    server.base_dir = tmp_path
    server.configs_by_machine = {"M": cfg}

    server._health_check_models()


def test_health_check_new_arch_fails_fast_on_missing_model(tmp_path):
    from capi_config import CAPIConfig
    from capi_server import CAPIServer

    cfg = CAPIConfig(
        machine_id="M",
        is_new_architecture=True,
        model_mapping={"G0F00000": {"inner": "model/missing_inner.pt", "edge": "model/missing_edge.pt"}},
    )
    server = CAPIServer.__new__(CAPIServer)
    server.base_dir = tmp_path
    server.configs_by_machine = {"M": cfg}

    with pytest.raises(RuntimeError, match="Server 停止啟動"):
        server._health_check_models()


def test_get_or_create_new_arch_prewarms_and_skips_legacy_model_path():
    from capi_config import CAPIConfig
    from capi_server import CAPIServer

    cfg = CAPIConfig(
        machine_id="M",
        is_new_architecture=True,
        model_mapping={"G0F00000": {"inner": "inner.pt", "edge": "edge.pt"}},
    )
    server = CAPIServer.__new__(CAPIServer)
    server.configs_by_machine = {"M": cfg}
    server.inferencers = {}
    server.inference_config = {"device": "cpu"}
    server.inferencer = "LEGACY"

    with patch("capi_server.CAPIInferencer") as mock_cls:
        mock_inf = mock_cls.return_value
        mock_inf.preload_v2_models.return_value = (2, 2)

        result = server._get_or_create_inferencer("M")

    assert result is mock_inf
    assert server.inferencers["M"] is mock_inf
    assert mock_cls.call_args.kwargs["model_path"] is None
    mock_inf.preload_v2_models.assert_called_once_with()


def test_adopt_loaded_config_replaces_fallback_object():
    from capi_config import CAPIConfig
    from capi_server import CAPIServer

    old_cfg = CAPIConfig(
        machine_id="MX",
        is_new_architecture=True,
        threshold_mapping={"G0F00000": {"inner": 0.5, "edge": 0.5}},
    )
    loaded_cfg = CAPIConfig(
        machine_id="MX",
        is_new_architecture=True,
        threshold_mapping={"G0F00000": {"inner": 0.7, "edge": 0.8}},
    )
    server = CAPIServer.__new__(CAPIServer)
    server.configs_by_machine = {"MX": old_cfg}
    server.fallback_config = old_cfg
    server.config = old_cfg

    server._adopt_loaded_config(loaded_cfg)

    assert server.configs_by_machine["MX"] is loaded_cfg
    assert server.fallback_config is loaded_cfg
    assert server.config is loaded_cfg


def test_apply_threshold_inplace_updates_config_and_status():
    from capi_config import CAPIConfig
    from capi_server import CAPIServer, server_status

    cfg = CAPIConfig(
        machine_id="MX",
        is_new_architecture=True,
        model_mapping={"G0F00000": {"inner": "x.pt", "edge": "y.pt"}},
        threshold_mapping={"G0F00000": {"inner": 0.5, "edge": 0.5}},
    )
    server = CAPIServer.__new__(CAPIServer)
    server.configs_by_machine = {"MX": cfg}
    server.fallback_config = cfg

    ok = server.apply_threshold_inplace("MX", "G0F00000", "inner", 0.83)
    assert ok is True
    # in-memory cfg 立即更新（process_panel_v2 下一次推論即生效）
    assert cfg.threshold_mapping["G0F00000"]["inner"] == 0.83
    assert cfg.threshold_mapping["G0F00000"]["edge"] == 0.5
    # server_status 同步（dashboard 立刻反映）
    assert server_status.threshold_mapping["G0F00000"]["inner"] == 0.83


def test_apply_threshold_inplace_returns_false_when_no_active_bundle():
    from capi_server import CAPIServer

    server = CAPIServer.__new__(CAPIServer)
    server.configs_by_machine = {}
    server.fallback_config = None

    assert server.apply_threshold_inplace("UNKNOWN", "G0F00000", "inner", 0.7) is False


def test_reload_runtime_config_from_db_keeps_new_arch_model_cache():
    from capi_config import CAPIConfig
    from capi_server import CAPIServer

    cfg = CAPIConfig(
        machine_id="MX",
        is_new_architecture=True,
        model_mapping={"G0F00000": {"inner": "x.pt", "edge": "y.pt"}},
        threshold_mapping={"G0F00000": {"inner": 0.5, "edge": 0.6}},
        dust_area_min=15,
    )
    model_cache = {("MX", "G0F00000", "inner"): object()}
    inferencer = SimpleNamespace(
        config=cfg,
        _model_cache_v2=model_cache,
        update_edge_config=MagicMock(),
    )

    server = CAPIServer.__new__(CAPIServer)
    server.configs_by_machine = {"MX": cfg}
    server.fallback_config = cfg
    server.config = cfg
    server.inferencer = inferencer
    server.inferencers = {"MX": inferencer}
    server.db = MagicMock()
    server.db.get_all_config_params.return_value = [
        {"param_name": "dust_area_min", "decoded_value": 123},
        {
            "param_name": "threshold_mapping",
            "decoded_value": {"G0F00000": {"inner": 0.1, "edge": 0.2}},
        },
    ]

    edge_cfg = object()
    with patch("capi_edge_cv.EdgeInspectionConfig.from_db_params", return_value=edge_cfg):
        synced = server.reload_runtime_config_from_db()

    assert synced == 1
    assert inferencer._model_cache_v2 is model_cache
    assert server.inferencer is inferencer
    assert server.inferencers["MX"] is inferencer
    assert cfg.dust_area_min == 123
    assert cfg.threshold_mapping == {"G0F00000": {"inner": 0.5, "edge": 0.6}}
    inferencer.update_edge_config.assert_called_once_with(edge_cfg)


def test_apply_threshold_inplace_rejects_bad_unit():
    from capi_config import CAPIConfig
    from capi_server import CAPIServer

    cfg = CAPIConfig(
        machine_id="MX",
        is_new_architecture=True,
        model_mapping={"G0F00000": {"inner": "x.pt", "edge": "y.pt"}},
        threshold_mapping={"G0F00000": {"inner": 0.5, "edge": 0.5}},
    )
    server = CAPIServer.__new__(CAPIServer)
    server.configs_by_machine = {"MX": cfg}
    server.fallback_config = cfg

    # lighting 不存在
    assert server.apply_threshold_inplace("MX", "UNKNOWN", "inner", 0.7) is False
    # zone 不存在
    assert server.apply_threshold_inplace("MX", "G0F00000", "middle", 0.7) is False
