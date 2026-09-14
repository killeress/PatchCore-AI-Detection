import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


_AAPI_PAYLOAD = "W0F00000,CDK2(03078,00497)"
_AAPI_REQUEST = (
    "AOI@PANEL001;MODEL-A;AAPI07;1920,1080;NG;"
    f"/image/PANEL001;{_AAPI_PAYLOAD}"
)


@pytest.mark.parametrize("use_gpu_lock", [False, True])
@pytest.mark.parametrize(
    ("machine_judgment", "client_request_text", "expected_report_payload"),
    [
        ("OK", "", None),
        ("NG", _AAPI_REQUEST, _AAPI_PAYLOAD),
    ],
)
def test_rerun_passes_stored_request_context_to_inference(
    tmp_path,
    monkeypatch,
    use_gpu_lock,
    machine_judgment,
    client_request_text,
    expected_report_payload,
):
    import capi_server
    from capi_web import CAPIWebHandler

    result = SimpleNamespace(edge_defects=[], preprocess_steps=[])
    inferencer = MagicMock()
    inferencer.config.image_preprocess_pipeline = []
    inferencer.process_panel.return_value = (
        [result],
        None,
        False,
        "",
        False,
        None,
        {},
    )
    parsed_aoi_report = {"W0F00000": [object()]}
    inferencer._parse_aoi_report_txt.return_value = parsed_aoi_report
    db = MagicMock()

    monkeypatch.setattr(CAPIWebHandler, "inferencer", inferencer)
    monkeypatch.setattr(CAPIWebHandler, "_capi_server_instance", None)
    monkeypatch.setattr(
        CAPIWebHandler,
        "_gpu_lock",
        threading.Lock() if use_gpu_lock else None,
        raising=False,
    )
    monkeypatch.setattr(CAPIWebHandler, "heatmap_manager", None)
    monkeypatch.setattr(CAPIWebHandler, "db", db)
    monkeypatch.setattr(
        CAPIWebHandler,
        "_rerun_lock",
        threading.Lock(),
        raising=False,
    )
    monkeypatch.setattr(
        CAPIWebHandler,
        "_rerun_tasks",
        {7: {"status": "running", "message": ""}},
        raising=False,
    )
    monkeypatch.setattr(capi_server, "aggregate_judgment", lambda _results: ("OK", "[]"))
    monkeypatch.setattr(
        capi_server,
        "results_to_db_data",
        lambda _results, _heatmap_info: [{"is_ng": False}],
    )
    monkeypatch.setattr(
        capi_server,
        "_stored_machine_judgment_for_record",
        lambda judgment, _results, _report: judgment,
    )
    monkeypatch.setattr(capi_server.InferenceLogCapture, "start_capture", lambda: None)
    monkeypatch.setattr(capi_server.InferenceLogCapture, "stop_capture", lambda: "")

    CAPIWebHandler._rerun_worker(
        7,
        {
            "image_dir": str(tmp_path),
            "model_id": "MODEL-A",
            "machine_judgment": machine_judgment,
            "machine_no": "CAPI07",
            "glass_id": "PANEL001",
            "resolution_x": 1920,
            "resolution_y": 1080,
            "client_bomb_info": "",
            "client_request_text": client_request_text,
            "heatmap_dir": "",
        },
    )

    assert inferencer.process_panel.call_args.kwargs["machine_judgment"] == machine_judgment
    if expected_report_payload is not None:
        inferencer._parse_aoi_report_txt.assert_called_once_with(
            tmp_path,
            glass_id="PANEL001",
            machine_judgment="NG",
            report_payload=expected_report_payload,
        )
        assert (
            inferencer.process_panel.call_args.kwargs["aoi_report_override"]
            is parsed_aoi_report
        )
    else:
        inferencer._parse_aoi_report_txt.assert_not_called()
    assert CAPIWebHandler._rerun_tasks[7]["status"] == "done"
    db.update_record_for_rerun.assert_called_once()


@pytest.mark.parametrize("profile,new_arch,grid,use_lock,fast", [
    ("capi", True, False, True, True),
    ("capi", True, False, False, True),
    ("aapi", True, False, True, False),
    ("capi", True, True, True, False),
    ("capi", False, False, True, False),
])
@pytest.mark.parametrize("converted", [False, True])
def test_rerun_keeps_cache_through_decision_and_saves_deferred_visuals(
    tmp_path, monkeypatch, profile, new_arch, grid, use_lock, fast, converted,
):
    import cv2
    import numpy as np
    import capi_image_orientation as image_io
    import capi_server
    import capi_web
    from capi_station_adapter import create_station_adapter

    cls = capi_web.CAPIWebHandler
    path = tmp_path / "W0F00000_001.tif"
    pixels = np.arange(32 * 48, dtype=np.uint8).reshape(32, 48)
    assert cv2.imwrite(str(path), pixels, [259, 1, 278, 1])
    config = SimpleNamespace(is_new_architecture=new_arch, grid_tiling_enabled=grid,
                             aoi_coord_inspection_enabled=True, image_preprocess_pipeline=[])
    inferencer = SimpleNamespace(config=config, station_adapter=create_station_adapter(profile))
    lock = threading.Lock() if use_lock else None
    result = SimpleNamespace(edge_defects=[], preprocess_steps=[])
    pending = {"pending": True}
    jobs = [{"kwargs": {}, "result": pending}]
    info = {"status": "within_spec" if converted else "not_within_spec", "converted": converted,
            "detail": {"visuals": [pending]}, "_visual_jobs": jobs, "reason": "test"}
    cache_at_inference = []
    events = []

    def process(*args, **kwargs):
        assert lock is None or lock.locked()
        cache_at_inference.append(image_io._panel_image_cache.get())
        np.testing.assert_array_equal(image_io.read_detection_image(path, -1, False), pixels)
        return [result], None, False, "", False, None, {}

    def evaluate(*args):
        cache = image_io._panel_image_cache.get()
        assert (cache is not None) is fast
        assert cache is cache_at_inference[0]
        if lock is not None:
            assert lock.locked() is fast
        color = image_io.read_detection_image(path, cv2.IMREAD_COLOR, False)
        np.testing.assert_array_equal(color, cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR))
        if fast:
            assert cache["misses"] == cache["hits"] == 1
        events.append("decision")
        return info

    def save_visual(**kwargs):
        assert image_io._panel_image_cache.get() is None
        assert lock is None or not lock.locked()
        events.append("visual")
        output = tmp_path / "within-spec.png"
        cv2.imwrite(str(output), pixels)
        return {"urls": {"crop_url": "/within-spec.png"}}

    def persist(**kwargs):
        assert events == ["decision", "visual"]
        assert (tmp_path / "within-spec.png").is_file()

    inferencer.process_panel = process
    server = SimpleNamespace(station_adapter=inferencer.station_adapter,
                             _evaluate_within_spec_for_inference=evaluate)
    db = MagicMock()
    db.update_record_for_rerun.side_effect = persist
    for name, value in {"inferencer": inferencer, "_capi_server_instance": server,
                        "_gpu_lock": lock, "_rerun_lock": threading.Lock(),
                        "_rerun_tasks": {7: {"status": "running"}},
                        "heatmap_manager": None, "db": db}.items():
        monkeypatch.setattr(cls, name, value, raising=False)
    monkeypatch.setattr(capi_server, "aggregate_judgment", lambda *a: ("NG", "[]"))
    monkeypatch.setattr(capi_server, "results_to_db_data", lambda *a: [{"is_ng": True}])
    monkeypatch.setattr(capi_server, "_stored_machine_judgment_for_record", lambda *a: "NG")
    monkeypatch.setattr(capi_server.InferenceLogCapture, "start_capture", lambda: None)
    monkeypatch.setattr(capi_server.InferenceLogCapture, "stop_capture", lambda: "")
    monkeypatch.setattr(capi_web, "_save_within_spec_dot_visuals", save_visual)
    cls._rerun_worker(7, {"image_dir": str(tmp_path), "glass_id": "G1", "model_id": "M1"})
    assert cls._rerun_tasks[7]["status"] == "done"
    assert jobs == []
    assert "_visual_jobs" not in info
    assert db.update_record_for_rerun.call_args.kwargs["ai_judgment"] == ("OK-i" if converted else "NG")
    assert db.save_within_spec_review_log.call_args.kwargs["detail"]["visuals"] == [
        {"urls": {"crop_url": "/within-spec.png"}},
    ]
    assert image_io._panel_image_cache.get() is None
    assert lock is None or not lock.locked()


@pytest.mark.parametrize("failure", ["inference", "no_results", "within_spec"])
def test_rerun_releases_cache_and_lock_on_failure(tmp_path, monkeypatch, failure):
    import capi_image_orientation as image_io
    import capi_server
    from capi_web import CAPIWebHandler as cls
    from capi_station_adapter import create_station_adapter

    lock = threading.Lock()
    config = SimpleNamespace(is_new_architecture=True, grid_tiling_enabled=False,
                             aoi_coord_inspection_enabled=True)
    inferencer = SimpleNamespace(config=config, station_adapter=create_station_adapter("capi"))

    def process(*args, **kwargs):
        assert lock.locked()
        assert image_io._panel_image_cache.get() is not None
        if failure == "inference":
            raise RuntimeError("inference failed")
        results = [] if failure == "no_results" else [SimpleNamespace(edge_defects=[])]
        return results, None, False, "", False, None, {}

    def evaluate(*args):
        assert lock.locked()
        assert image_io._panel_image_cache.get() is not None
        raise RuntimeError("within-spec failed")

    inferencer.process_panel = process
    server = SimpleNamespace(station_adapter=inferencer.station_adapter,
                             _evaluate_within_spec_for_inference=evaluate)
    for name, value in {"inferencer": inferencer, "_capi_server_instance": server,
                        "_gpu_lock": lock, "_rerun_lock": threading.Lock(),
                        "_rerun_tasks": {7: {"status": "running"}}, "db": MagicMock()}.items():
        monkeypatch.setattr(cls, name, value, raising=False)
    monkeypatch.setattr(capi_server, "aggregate_judgment", lambda *a: ("NG", "[]"))
    monkeypatch.setattr(capi_server.InferenceLogCapture, "start_capture", lambda: None)
    monkeypatch.setattr(capi_server.InferenceLogCapture, "stop_capture", lambda: "")
    cls._rerun_worker(7, {"image_dir": str(tmp_path), "glass_id": "G1", "model_id": "M1"})
    assert cls._rerun_tasks[7]["status"] == "error"
    assert not lock.locked()
    assert image_io._panel_image_cache.get() is None
    cls.db.update_record_for_rerun.assert_not_called()
