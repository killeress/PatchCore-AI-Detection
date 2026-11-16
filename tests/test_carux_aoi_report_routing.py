"""CarUX CAPI 報告路徑從 TCP 與重跑入口傳到推論器。"""
from contextlib import nullcontext
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from capi_config import CAPIConfig
from capi_inference import CAPIInferencer
from capi_server import CAPIServer
from capi_station_adapter import create_station_adapter


PANEL = Path('/158/TIANMU/yuantu/MODEL/20261112/PID')
PRIMARY = Path('/102/20261112/CAHM0701/OK')


def _server(facility, machine, precheck=False):
    server = CAPIServer.__new__(CAPIServer)
    server.path_mapping = {}
    server.cpu_workers = 1
    server._gpu_lock = threading.Lock()
    server.server_config = {'mes_report': {'facility': facility}, 'carux_aoi_report_root': '/102'}
    server.station_adapter = create_station_adapter('capi')
    cfg = CAPIConfig()
    cfg.image_abnormal_detection_enabled = precheck
    inf = SimpleNamespace(config=cfg, process_panel=MagicMock(return_value=([], None, False, '', False, None, {})),
                          _parse_aoi_report_txt=MagicMock(return_value={}))
    server._get_or_create_inferencer = lambda _: inf
    request = {'model_id': 'MODEL', 'glass_id': 'PID', 'machine_no': machine,
               'machine_judgment': 'NG', 'image_dir': str(PANEL), 'resolution': (1920, 720)}
    return server, inf, request


def _run(server, request, panel=PANEL):
    with patch('capi_server.resolve_panel_image_directory', return_value=(str(panel), panel)), \
         patch.object(Path, 'exists', return_value=True), \
         patch.object(Path, 'is_dir', return_value=True), \
         patch.object(server.station_adapter, 'find_white_frame_image', return_value=None), \
         patch('capi_server.panel_image_cache', return_value=nullcontext()), \
         patch('capi_server.use_capi_aoi_fast_path', return_value=False), \
         patch('capi_server.check_image_abnormal_precheck', return_value=None):
        return server._process_request(request)


@pytest.mark.parametrize('facility,machine,expected', [
    ('CARUX', 'CAPI0701', PRIMARY),
    ('CARUX', 'CAPI0702', Path('/102/20261112/CAHM0702/OK')),
    ('CARUX', 'CAPI0601', Path('/102/20261112/CAHM0601/OK')),
    ('CARUX', 'CAPI07', None), ('MOD2', 'CAPI0701', None),
])
def test_tcp_passes_primary_source_only_for_matching_carux_machine(facility, machine, expected):
    server, inf, request = _server(facility, machine)
    _run(server, request)
    assert inf.process_panel.call_args.kwargs.get('carux_report_dir') == expected


@pytest.mark.parametrize('machine', ['CAPI0701', 'CAPI0601'])
def test_tcp_invalid_date_still_marks_carux_102_only(machine):
    server, inf, request = _server('CARUX', machine)
    bad_panel = Path('/158/TIANMU/yuantu/MODEL/invalid/PID')
    _run(server, request, bad_panel)
    assert inf.process_panel.call_args.kwargs.get('carux_report_only') is True
    assert inf.process_panel.call_args.kwargs.get('carux_report_dir') is None


def test_precheck_reads_same_primary_source_as_inference():
    server, inf, request = _server('CARUX', 'CAPI0701', precheck=True)
    _run(server, request)
    assert inf._parse_aoi_report_txt.call_args.kwargs.get('carux_report_dir') == PRIMARY
    assert inf._parse_aoi_report_txt.call_args.kwargs.get('model_id') == 'MODEL'
    assert inf.process_panel.call_args.kwargs.get('aoi_report_override') == {}


def test_process_panel_threads_primary_to_architecture_dispatch():
    cfg = CAPIConfig()
    cfg.is_new_architecture = True
    inf = CAPIInferencer.__new__(CAPIInferencer)
    inf.config = cfg
    inf._dispatch_process_panel = MagicMock(return_value=([],))
    with patch('capi_inference.panel_image_cache', return_value=nullcontext()), \
         patch('capi_inference.use_capi_aoi_fast_path', return_value=False):
        inf.process_panel(PANEL, glass_id='PID', model_id='MODEL',
                          carux_report_dir=PRIMARY, client_product_resolution=(1920, 720))
    assert inf._dispatch_process_panel.call_args.kwargs['carux_report_dir'] == PRIMARY


def test_v2_passes_actual_report_source_and_model_to_parser():
    cfg = CAPIConfig()
    cfg.aoi_coord_inspection_enabled = True
    inf = CAPIInferencer.__new__(CAPIInferencer)
    inf.config = cfg
    inf._prepare_panel_image_files = MagicMock(return_value=([], False))
    inf._detect_panel_mark_binary_region = MagicMock(return_value=(None, []))
    inf._parse_aoi_report_txt = MagicMock(side_effect=RuntimeError('stop after report dispatch'))
    with pytest.raises(RuntimeError, match='stop after report dispatch'):
        inf._process_panel_v2(PANEL, glass_id='PID', model_id='MODEL', carux_report_dir=PRIMARY)
    assert inf._parse_aoi_report_txt.call_args.kwargs['carux_report_dir'] == PRIMARY
    assert inf._parse_aoi_report_txt.call_args.kwargs['model_id'] == 'MODEL'


def test_v2_invalid_date_keeps_carux_102_only():
    cfg = CAPIConfig()
    cfg.aoi_coord_inspection_enabled = True
    inf = CAPIInferencer.__new__(CAPIInferencer)
    inf.config = cfg
    inf._prepare_panel_image_files = MagicMock(return_value=([], False))
    inf._detect_panel_mark_binary_region = MagicMock(return_value=(None, []))
    inf._parse_aoi_report_txt = MagicMock(side_effect=RuntimeError('stop after report dispatch'))
    with pytest.raises(RuntimeError, match='stop after report dispatch'):
        inf._process_panel_v2(PANEL, glass_id='PID', model_id='MODEL', carux_report_only=True)
    assert inf._parse_aoi_report_txt.call_args.kwargs['carux_report_only'] is True


@pytest.mark.parametrize('facility,machine,expected,panel,report_only', [
    ('CARUX', 'CAPI0701', PRIMARY, PANEL, True),
    ('CARUX', 'CAPI0702', Path('/102/20261112/CAHM0702/OK'), PANEL, True),
    ('CARUX', 'CAPI0601', Path('/102/20261112/CAHM0601/OK'), PANEL, True),
    ('CARUX', 'CAPI07', None, PANEL, False),
    ('MOD2', 'CAPI0701', None, PANEL, False),
    ('CARUX', 'CAPI0701', None, Path('/158/TIANMU/yuantu/MODEL/invalid/PID'), True),
    ('CARUX', 'CAPI0601', None, Path('/158/TIANMU/yuantu/MODEL/invalid/PID'), True),
])
def test_web_rerun_uses_same_primary_report_source(monkeypatch, facility, machine, expected, panel, report_only):
    from capi_web import CAPIWebHandler

    cfg = CAPIConfig()
    inf = MagicMock()
    inf.config = cfg
    inf.station_adapter = create_station_adapter('capi')
    inf.process_panel.return_value = ([], None, False, '', False, None, {})
    server = SimpleNamespace(server_config={'mes_report': {'facility': facility}, 'carux_aoi_report_root': '/102'},
                             station_adapter=inf.station_adapter, _get_or_create_inferencer=lambda _: inf)
    monkeypatch.setattr(CAPIWebHandler, 'inferencer', inf)
    monkeypatch.setattr(CAPIWebHandler, '_capi_server_instance', server)
    monkeypatch.setattr(CAPIWebHandler, '_gpu_lock', None, raising=False)
    monkeypatch.setattr(CAPIWebHandler, '_rerun_lock', threading.Lock(), raising=False)
    monkeypatch.setattr(CAPIWebHandler, '_rerun_tasks', {7: {'status': 'running'}}, raising=False)
    monkeypatch.setattr(CAPIWebHandler, 'heatmap_manager', None)
    with patch.object(inf.station_adapter, 'find_white_frame_image', return_value=None):
        CAPIWebHandler._rerun_worker(7, {
            'image_dir': str(panel), 'model_id': 'MODEL', 'machine_no': machine,
            'glass_id': 'PID', 'machine_judgment': 'NG', 'resolution_x': 1920,
            'resolution_y': 720, 'client_bomb_info': '', 'client_request_text': '',
        })
    assert inf.process_panel.call_args is not None, CAPIWebHandler._rerun_tasks[7]
    kwargs = inf.process_panel.call_args.kwargs
    assert kwargs.get('carux_report_dir') == expected
    assert bool(kwargs.get('carux_report_only')) == report_only
