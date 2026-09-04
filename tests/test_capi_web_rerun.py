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
