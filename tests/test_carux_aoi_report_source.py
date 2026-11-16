"""CarUX CAPI 報告來源與錯誤處理。"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import pytest

from capi_carux_report import CARUX_REPORT_NOT_FOUND, resolve_carux_report_dir, read_carux_report_lines


def _entry(name, mtime):
    return SimpleNamespace(name=name, is_file=lambda: True,
                           stat=lambda: SimpleNamespace(st_mtime=mtime))


def _scan(entries):
    context = MagicMock()
    context.__enter__.return_value = iter(entries)
    return context


def test_carux_seven_line_uses_image_path_date_and_configured_root():
    config = {"mes_report": {"facility": "CARUX"},
              "carux_aoi_report_root": "/mock102"}
    image_dir = Path('/158/TIANMU/yuantu/GD1030BC0160M/20261112/T365C8V1NK35')
    assert resolve_carux_report_dir(config, 'CAPI0701', image_dir) == (
        Path('/mock102/20261112/CAHM0701/OK')
    )


@pytest.mark.parametrize('machine,folder', [
    ('CAPI0701', 'CAHM0701'), ('CAPI0601', 'CAHM0601'), ('CAPI0702', 'CAHM0702'),
    ('capi0701', 'CAHM0701'),
])
def test_carux_machine_numbers_map_to_corresponding_folder(machine, folder):
    config = {'mes_report': {'facility': 'CARUX'}, 'carux_aoi_report_root': '/mock102'}
    assert resolve_carux_report_dir(config, machine, Path('/x/M/20261113/PID')) == (
        Path('/mock102/20261113') / folder / 'OK'
    )


@pytest.mark.parametrize('facility,machine', [
    ('MOD2', 'CAPI0701'), ('CARUX', 'CAPI07'), ('CARUX', 'CAPI07011'),
    ('CARUX', 'XCAPI0701'), ('CARUX', 'CAPI0701_extra'),
    ('CARUX', 'CAPI０７０１'), ('CARUX', 'CApı0701'),
])
def test_other_factories_and_invalid_machine_numbers_keep_original_source(facility, machine):
    config = {'mes_report': {'facility': facility}}
    assert resolve_carux_report_dir(config, machine, Path('/x/M/20261113/PID')) is None


def test_report_matching_uses_exact_pid_model_date_and_latest_mtime():
    entries = [
        _entry('PID_20261113130000_MODEL_130000.TXT', 10),
        _entry('PID_20261113120000_MODEL_120000.txt', 20),
        _entry('PID2_20261113140000_MODEL_140000.TXT', 50),
        _entry('PID_20261113150000_OTHER_150000.TXT', 60),
        _entry('PID_20261112160000_MODEL_160000.TXT', 70),
        _entry('PID_20261113160000_MODEL_W0F00000_160000.jpg', 80),
    ]
    file_open = mock_open(read_data='PID\n@;NG;;NGPCDK20000100002W0F00000,\n')
    with patch('capi_carux_report.os.scandir', return_value=_scan(entries)), \
         patch('builtins.open', file_open):
        result = read_carux_report_lines(Path('/102/20261113/CAHM0701/OK'), 'PID', 'MODEL', '20261113')
    assert result == ['PID\n', '@;NG;;NGPCDK20000100002W0F00000,\n']
    assert file_open.call_args.args[0].name == 'PID_20261113120000_MODEL_120000.txt'


def test_no_matching_report_returns_not_found():
    with patch('capi_carux_report.os.scandir', return_value=_scan([])):
        assert read_carux_report_lines(Path('/102/20261113/CAHM0701/OK'), 'PID', 'MODEL', '20261113') is CARUX_REPORT_NOT_FOUND


def test_missing_report_directory_returns_not_found():
    with patch('capi_carux_report.os.scandir', side_effect=FileNotFoundError('missing')):
        assert read_carux_report_lines(Path('/102/20261113/CAHM0701/OK'), 'PID', 'MODEL', '20261113') is CARUX_REPORT_NOT_FOUND


@pytest.mark.parametrize('error', [PermissionError('denied'), OSError('disconnected')])
def test_unreadable_directory_returns_none(error):
    with patch('capi_carux_report.os.scandir', side_effect=error):
        assert read_carux_report_lines(Path('/102/20261113/CAHM0701/OK'), 'PID', 'MODEL', '20261113') is None


def test_unreadable_selected_report_returns_none():
    entries = [_entry('PID_20261113130000_MODEL_130000.TXT', 10)]
    with patch('capi_carux_report.os.scandir', return_value=_scan(entries)), \
         patch('builtins.open', side_effect=PermissionError('denied')):
        assert read_carux_report_lines(Path('/102/20261113/CAHM0701/OK'), 'PID', 'MODEL', '20261113') is None


def test_empty_readable_report_is_not_mistaken_for_missing():
    entries = [_entry('PID_20261113130000_MODEL_130000.TXT', 10)]
    with patch('capi_carux_report.os.scandir', return_value=_scan(entries)), \
         patch('builtins.open', mock_open(read_data='')):
        assert read_carux_report_lines(Path('/102/20261113/CAHM0701/OK'), 'PID', 'MODEL', '20261113') == []
