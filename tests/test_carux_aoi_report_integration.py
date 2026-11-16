"""CarUX 報告來源與現有 AOI 解析的整合測試。"""
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

from capi_config import CAPIConfig
from capi_inference import CAPIInferencer
from capi_station_adapter import create_station_adapter


PANEL = Path('/158/TIANMU/yuantu/MODEL/20261113/PID')
PRIMARY = Path('/102/20261113/CAHM0701/OK')
LEGACY = Path('/158/TIANMU/Report/MODEL/20261113/PID/report.TXT')


def _worker():
    cfg = CAPIConfig()
    cfg.aoi_report_path_replace_from = 'yuantu'
    cfg.aoi_report_path_replace_to = 'Report'
    cfg.model_mapping = {}
    inf = CAPIInferencer.__new__(CAPIInferencer)
    inf.config = cfg
    inf._model_mapping = {}
    return inf


def _report(coord):
    return ['PID\n', f'@;NG;;NGPCDK2{coord}W0F00000,\n']


def _parse(worker):
    return worker._parse_aoi_report_txt(PANEL, glass_id='PID', model_id='MODEL', carux_report_dir=PRIMARY)


def test_102_readable_report_has_priority_and_is_parsed():
    worker = _worker()
    with patch('capi_inference.read_carux_report_lines', return_value=_report('0000100002')), \
         patch.object(Path, 'exists', side_effect=AssertionError('legacy should not be read')):
        parsed = _parse(worker)
    defect = parsed['W0F00000'][0]
    assert (defect.product_x, defect.product_y) == (1, 2)


def test_102_no_matching_txt_for_ng_does_not_read_158(caplog):
    worker = _worker()
    scan = MagicMock()
    scan.__enter__.return_value = iter([])
    with caplog.at_level(logging.INFO, logger='capi.inference'), \
         patch('capi_carux_report.os.scandir', return_value=scan), \
         patch.object(Path, 'exists', side_effect=AssertionError('158 should not be read')):
        parsed = worker._parse_aoi_report_txt(
            PANEL, glass_id='PID', model_id='MODEL', machine_judgment='NG',
            carux_report_dir=PRIMARY,
        )
    assert parsed == {}
    assert any(r.levelno == logging.INFO and '報告目錄中無 TXT 檔案' in r.message
               for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_102_no_matching_txt_for_ok_is_not_warning_or_error(caplog):
    worker = _worker()
    scan = MagicMock()
    scan.__enter__.return_value = iter([])
    with caplog.at_level(logging.INFO, logger='capi.inference'), \
         patch('capi_carux_report.os.scandir', return_value=scan), \
         patch.object(Path, 'exists', side_effect=AssertionError('158 should not be read')):
        parsed = worker._parse_aoi_report_txt(
            PANEL, glass_id='PID', model_id='MODEL', machine_judgment='OK',
            carux_report_dir=PRIMARY,
        )
    assert parsed == {}
    assert any(r.levelno == logging.INFO and '報告目錄中無 TXT 檔案' in r.message
               for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_102_report_directory_missing_does_not_read_158():
    worker = _worker()
    with patch('capi_carux_report.os.scandir', side_effect=FileNotFoundError('missing')), \
         patch.object(Path, 'exists', side_effect=AssertionError('158 should not be read')):
        assert worker._parse_aoi_report_txt(
            PANEL, glass_id='PID', model_id='MODEL', machine_judgment='OK',
            carux_report_dir=PRIMARY,
        ) == {}


def test_102_unavailable_does_not_read_original_report():
    worker = _worker()
    with patch('capi_inference.read_carux_report_lines', return_value=None), \
         patch.object(Path, 'exists', side_effect=AssertionError('158 should not be read')):
        assert _parse(worker) == {}


def test_network_access_error_does_not_read_158(caplog):
    worker = _worker()
    with patch('capi_carux_report.os.scandir', side_effect=PermissionError('denied')), \
         patch.object(Path, 'exists', side_effect=AssertionError('158 should not be read')):
        assert _parse(worker) == {}
    assert '102 報告無法讀取' in caplog.text
    assert '回退' not in caplog.text


def test_102_selected_file_unreadable_does_not_read_158():
    worker = _worker()
    entry = SimpleNamespace(name='PID_20261113130000_MODEL_130000.TXT',
                            is_file=lambda: True, stat=lambda: SimpleNamespace(st_mtime=1))
    scan = MagicMock()
    scan.__enter__.return_value = iter([entry])
    with patch('capi_carux_report.os.scandir', return_value=scan), \
         patch.object(Path, 'exists', side_effect=AssertionError('158 should not be read')), \
         patch('builtins.open', side_effect=PermissionError('denied')):
        assert _parse(worker) == {}


def test_readable_102_ok_report_does_not_fall_back():
    worker = _worker()
    with patch('capi_inference.read_carux_report_lines', return_value=['PID\n', '@;OK;;OK,\n']), \
         patch.object(Path, 'exists', side_effect=AssertionError('legacy should not be read')):
        assert _parse(worker) == {}


def test_invalid_date_primary_does_not_read_158():
    worker = _worker()
    bad_panel = Path('/158/TIANMU/yuantu/MODEL/invalid/PID')
    with patch.object(Path, 'exists', side_effect=AssertionError('158 should not be read')):
        assert worker._parse_aoi_report_txt(
            bad_panel, glass_id='PID', model_id='MODEL',
            carux_report_only=True, carux_report_dir=None,
        ) == {}


def test_image_pid_mismatch_does_not_read_158():
    worker = _worker()
    with patch.object(Path, 'exists', side_effect=AssertionError('158 should not be read')):
        assert worker._parse_aoi_report_txt(
            PANEL, glass_id='OTHER', model_id='MODEL', carux_report_dir=PRIMARY,
        ) == {}


def test_original_parser_without_carux_source_keeps_existing_behavior():
    worker = _worker()
    file_open = mock_open(read_data='PID\n@;NG;;NGPCDK20000300004W0F00000,\n')
    with patch('capi_inference.read_carux_report_lines', side_effect=AssertionError('unexpected 102 read')), \
         patch.object(Path, 'exists', return_value=True), \
         patch.object(Path, 'glob', return_value=[LEGACY]), \
         patch.object(Path, 'resolve', lambda self: self), \
         patch.object(Path, 'stat', return_value=SimpleNamespace(st_mtime=1)), \
         patch('builtins.open', file_open):
        parsed = worker._parse_aoi_report_txt(PANEL, glass_id='PID')
    assert (parsed['W0F00000'][0].product_x, parsed['W0F00000'][0].product_y) == (3, 4)


def test_observed_filename_and_report_are_matched_and_parsed():
    worker = _worker()
    panel = Path('/158/TIANMU/yuantu/GD1030BC0160M/20261113/T365C8X6NN27')
    entry = SimpleNamespace(
        name='T365C8X6NN27_20261113135712_GD1030BC0160M_135712.TXT',
        is_file=lambda: True, stat=lambda: SimpleNamespace(st_mtime=1),
    )
    scan = MagicMock()
    scan.__enter__.return_value = iter([entry])
    text = ('T365C8X6NN27\n@QJPG-T365C8X6NN27;NG;;NG'
            'PCDB50088000513B0F00000PCDB50066200629WGF50500,\n')
    with patch('capi_carux_report.os.scandir', return_value=scan), \
         patch('builtins.open', mock_open(read_data=text)), \
         patch.object(Path, 'exists', side_effect=AssertionError('legacy should not be read')):
        parsed = worker._parse_aoi_report_txt(
            panel, glass_id='T365C8X6NN27', model_id='GD1030BC0160M',
            carux_report_dir=PRIMARY,
        )
    assert (parsed['B0F00000'][0].product_x, parsed['B0F00000'][0].product_y) == (880, 513)
    assert (parsed['WGF50500'][0].product_x, parsed['WGF50500'][0].product_y) == (662, 629)


def test_aapi_report_payload_does_not_read_carux_share():
    worker = _worker()
    worker.station_adapter = create_station_adapter('aapi')
    with patch('capi_inference.read_carux_report_lines', side_effect=AssertionError('not a CAPI report')):
        parsed = worker._parse_aoi_report_txt(
            PANEL, glass_id='PID', machine_judgment='NG',
            report_payload='W0F00000,CDK2(03078,00497)',
            carux_report_dir=PRIMARY, model_id='MODEL',
        )
    assert parsed['W0F00000'][0].product_x == 1026
