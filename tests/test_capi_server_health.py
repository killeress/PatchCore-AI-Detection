import pytest


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
