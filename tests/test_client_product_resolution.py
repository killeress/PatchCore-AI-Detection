"""
tests/test_client_product_resolution.py

CarUX 廠別：產品解析度採用 Client 傳入值，不再依機種第六碼查表。
其他廠別行為不變。
"""
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from capi_config import CAPIConfig


def _make_grid_config(tmp_dir: Path) -> CAPIConfig:
    """Pixel Grid 模型：訓練記錄 1920x720，機種第六碼 '0' 不在映射表內。"""
    inner_pt = tmp_dir / "g_inner.pt"
    edge_pt = tmp_dir / "g_edge.pt"
    cfg = CAPIConfig(
        machine_id="GD1030BC0160M",
        is_new_architecture=True,
        tile_size=512,
        model_mapping={"G0F00000": {"inner": str(inner_pt), "edge": str(edge_pt)}},
        threshold_mapping={"G0F00000": {"inner": 0.5, "edge": 0.5}},
        scratch_classifier_enabled=False,
    )
    cfg.grid_canonicalization_enabled = True
    cfg.grid_samples_per_cell = 3
    cfg.grid_product_resolution = (1920, 720)
    return cfg


def _write_grey_panel_image(folder: Path) -> None:
    import cv2

    img = np.zeros((1024, 1024), dtype=np.uint8)
    img[100:900, 100:900] = 200
    cv2.imwrite(str(folder / "G0F00000_test.png"), img)


# ---------------------------------------------------------------------------
# resolve_request_product_resolution helper
# ---------------------------------------------------------------------------

def test_carux_uses_client_resolution():
    from capi_inference import resolve_request_product_resolution

    assert resolve_request_product_resolution(
        "GD1030BC0160M", None,
        client_resolution=(1920, 720), use_client_resolution=True,
    ) == (1920, 720)


def test_carux_falls_back_to_table_when_client_resolution_missing():
    from capi_inference import resolve_request_product_resolution

    assert resolve_request_product_resolution(
        "GN140JCAL010S", None,
        client_resolution=None, use_client_resolution=True,
    ) == (1920, 1200)


def test_other_facilities_ignore_client_resolution():
    from capi_inference import resolve_request_product_resolution

    assert resolve_request_product_resolution(
        "GN140JCAL010S", None,
        client_resolution=(1920, 720), use_client_resolution=False,
    ) == (1920, 1200)


@pytest.mark.parametrize("bad", [(0, 720), (1920, -1), None])
def test_carux_rejects_invalid_client_resolution(bad):
    from capi_inference import resolve_request_product_resolution

    # '0' 不在預設映射表 → 回退預設值 (1920, 1080)
    assert resolve_request_product_resolution(
        "GD1030BC0160M", None,
        client_resolution=bad, use_client_resolution=True,
    ) == (1920, 1080)


# ---------------------------------------------------------------------------
# process_panel：Pixel Grid 模型在 CarUX 下接受 Client 解析度
# ---------------------------------------------------------------------------

def test_grid_model_rejects_table_fallback_reproducing_production_error(tmp_path):
    """第六碼 '0' 查不到 → 預設 1920x1080 ≠ 訓練 1920x720 → 報錯（重現現場）。"""
    _write_grey_panel_image(tmp_path)
    cfg = _make_grid_config(tmp_path)

    from capi_inference import CAPIInferencer

    inferencer = CAPIInferencer(cfg)
    with pytest.raises(ValueError, match="不一致"):
        inferencer.process_panel(tmp_path)


def test_grid_model_accepts_client_product_resolution(tmp_path, monkeypatch):
    _write_grey_panel_image(tmp_path)
    cfg = _make_grid_config(tmp_path)
    captured = {}

    def fake_preprocess_panel_folder(
        _panel_path, pre_cfg, image_files=None, boundary_reference_files=None, **kwargs
    ):
        captured["product_resolution"] = pre_cfg.product_resolution
        return {}

    monkeypatch.setattr(
        "capi_preprocess.preprocess_panel_folder", fake_preprocess_panel_folder
    )

    from capi_inference import CAPIInferencer

    inferencer = CAPIInferencer(cfg)
    inferencer.process_panel(tmp_path, client_product_resolution=(1920, 720))

    assert captured["product_resolution"] == (1920, 720)


def test_legacy_carux_passes_client_resolution_to_boundary_processing():
    from capi_inference import CAPIInferencer

    inferencer = CAPIInferencer(CAPIConfig(machine_id="GD1030BC0160M"))
    inferencer._dispatch_process_panel = MagicMock(return_value=([],))
    inferencer.process_panel(Path("unused"), client_product_resolution=(1366, 768))
    assert inferencer._dispatch_process_panel.call_args.kwargs["boundary_product_resolution"] == (1366, 768)


def test_legacy_boundary_selection_uses_request_resolution():
    from capi_inference import CAPIInferencer

    inferencer = CAPIInferencer(CAPIConfig(machine_id="GD1030BC0160M"))
    assert inferencer._use_robust_panel_boundary() is False
    assert inferencer._use_robust_panel_boundary((1366, 768)) is True


def test_grid_model_rejects_mismatched_client_resolution():
    from capi_inference import CAPIInferencer

    cfg = CAPIConfig(machine_id="GD1030BC0160M", is_new_architecture=True)
    cfg.grid_canonicalization_enabled = True
    cfg.grid_product_resolution = (1920, 720)
    inferencer = CAPIInferencer(cfg)
    with pytest.raises(ValueError, match="不一致"):
        inferencer.process_panel(Path("unused"), client_product_resolution=(1920, 1080))


def test_grid_model_ignores_client_resolution_without_flag(tmp_path):
    """未走 CarUX 路徑時，原有查表行為不變。"""
    _write_grey_panel_image(tmp_path)
    cfg = _make_grid_config(tmp_path)

    from capi_inference import CAPIInferencer

    inferencer = CAPIInferencer(cfg)
    with pytest.raises(ValueError, match="不一致"):
        # 既有 product_resolution 參數維持「不採用」的相容行為
        inferencer.process_panel(tmp_path, product_resolution=(1920, 720))


# ---------------------------------------------------------------------------
# server 接線：依 mes_report.facility 決定是否採用 Client 解析度
# ---------------------------------------------------------------------------

def _make_server(facility: str):
    from capi_server import CAPIServer

    server = CAPIServer.__new__(CAPIServer)
    server.path_mapping = {}
    server.cpu_workers = 1
    server._gpu_lock = threading.Lock()
    server.server_config = {"mes_report": {"facility": facility}}
    return server


def _make_inferencer_spy():
    config = CAPIConfig()
    inferencer = SimpleNamespace(
        config=config,
        process_panel=MagicMock(side_effect=RuntimeError("stop after dispatch")),
    )
    return inferencer


def test_carux_server_threads_client_resolution(tmp_path):
    server = _make_server("CARUX")
    inferencer = _make_inferencer_spy()
    server._get_or_create_inferencer = lambda _model_id: inferencer
    parsed = {
        "model_id": "GD1030BC0160M", "glass_id": "G1",
        "image_dir": str(tmp_path), "resolution": (1920, 720),
    }

    server._process_request(parsed)

    assert inferencer.process_panel.call_args.kwargs["client_product_resolution"] == (1920, 720)
    assert parsed["_resolved_product_resolution"] == (1920, 720)


def test_mod2_server_keeps_table_resolution(tmp_path):
    server = _make_server("MOD2")
    inferencer = _make_inferencer_spy()
    server._get_or_create_inferencer = lambda _model_id: inferencer
    parsed = {
        "model_id": "GD1030BC0160M", "glass_id": "G1",
        "image_dir": str(tmp_path), "resolution": (1920, 720),
    }

    server._process_request(parsed)

    # 第六碼 '0' 查不到 → 預設 (1920, 1080)；不走 Client 值
    assert inferencer.process_panel.call_args.kwargs["product_resolution"] == (1920, 1080)
    assert inferencer.process_panel.call_args.kwargs.get("client_product_resolution") is None
    assert "_resolved_product_resolution" not in parsed


# ---------------------------------------------------------------------------
# QJPG 回覆：使用推理時採用的解析度換算座標
# ---------------------------------------------------------------------------

def _image_result(image_name: str):
    from capi_inference import ImageResult

    return ImageResult(
        image_path=Path(image_name),
        image_size=(1200, 800),
        otsu_bounds=(100, 200, 1100, 700),
        exclusion_regions=[],
        tiles=[],
        excluded_tile_count=0,
        processed_tile_count=0,
        processing_time=0.0,
        raw_bounds=(100, 200, 1100, 700),
        mark_text="EJ",
    )


def _ng_result():
    from capi_inference import TileInfo

    result = _image_result("W0F00000_114438.tif")
    tile = TileInfo(
        tile_id=1, x=0, y=0, width=512, height=512,
        image=np.zeros((512, 512, 3), dtype=np.uint8),
    )
    tile.anomaly_peak_x = 600
    tile.anomaly_peak_y = 450
    result.tiles = [tile]
    result.anomaly_tiles = [(tile, 0.91, None)]
    return result


def test_qjpg_response_uses_resolved_product_resolution():
    from capi_server import build_qjpg_response

    parsed = {
        "glass_id": "G1", "model_id": "GD1030BC0160M",
        "_resolved_product_resolution": (1920, 720),
    }
    response = build_qjpg_response(parsed, "NG", [_ng_result()], CAPIConfig())

    assert response.endswith("NGPCDK20096000360W0F00000,")


def test_side_white_context_uses_request_resolution_for_carux():
    from capi_side_white import snapshot_context

    context = snapshot_context(CAPIConfig(), None, {
        "model_id": "GD1030BC0160M",
        "_resolved_product_resolution": (1920, 720),
    })
    assert context["product_resolution"] == [1920, 720]


def test_side_white_context_keeps_model_mapping_without_request_resolution():
    from capi_side_white import snapshot_context

    context = snapshot_context(CAPIConfig(), None, {"model_id": "GD1030BC0160M"})
    assert context["product_resolution"] == [1920, 1080]


def test_qjpg_response_without_resolved_resolution_keeps_table_behavior():
    from capi_server import build_qjpg_response

    parsed = {"glass_id": "G1", "model_id": "GD1030BC0160M"}
    response = build_qjpg_response(parsed, "NG", [_ng_result()], CAPIConfig())

    # 第六碼 '0' 查不到 → 預設 (1920, 1080)
    assert response.endswith("NGPCDK20096000540W0F00000,")
