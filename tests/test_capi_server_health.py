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
