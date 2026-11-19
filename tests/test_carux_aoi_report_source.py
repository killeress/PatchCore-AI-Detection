"""CarUX CAPI 報告來源與錯誤處理。"""
import logging
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


def test_report_matching_uses_exact_pid_model_date_and_latest_mtime(caplog):
    entries = [
        _entry('PID_20261113130000_MODEL_130000.TXT', 10),
        _entry('PID_20261113120000_MODEL_120000.txt', 20),
        _entry('PID2_20261113140000_MODEL_140000.TXT', 50),
        _entry('PID_20261113150000_OTHER_150000.TXT', 60),
        _entry('PID_20261112160000_MODEL_160000.TXT', 70),
        _entry('PID_20261113160000_MODEL_W0F00000_160000.jpg', 80),
    ]
    file_open = mock_open(read_data='PID\n@;NG;;NGPCDK20000100002W0F00000,\n')
    with caplog.at_level(logging.INFO, logger='capi.inference'), \
         patch('capi_carux_report.os.scandir', return_value=_scan(entries)), \
         patch('builtins.open', file_open):
        result = read_carux_report_lines(Path('/102/20261113/CAHM0701/OK'), 'PID', 'MODEL', '20261113')
    assert result == ['PID\n', '@;NG;;NGPCDK20000100002W0F00000,\n']
    assert file_open.call_args.args[0].name == 'PID_20261113120000_MODEL_120000.txt'
    assert '讀取公共網盤報告' in caplog.text
    assert '[REPORT_DATE_FALLBACK]' not in caplog.text


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


def _read_from_public_fixtures(date, reports):
    """Simulate the external network share; exercise real date and filename matching."""
    def scan(directory):
        entries = [_entry(name, 1) for folder, name in reports if folder == Path(directory)]
        if not entries:
            raise FileNotFoundError(str(directory))
        return _scan(entries)

    def open_report(path, *args, **kwargs):
        import io
        path = Path(path)
        return io.StringIO(reports[(path.parent, path.name)])

    with patch('capi_carux_report.os.scandir', side_effect=scan), \
         patch('builtins.open', side_effect=open_report):
        return read_carux_report_lines(Path('/102') / date / 'CAHM0701' / 'OK', 'PID', 'MODEL', date)


def _report_entry(date, *, pid='PID', model='MODEL', filename_date=None):
    return (Path('/102') / date / 'CAHM0701' / 'OK',
            f'{pid}_{filename_date or date}120000_{model}_120000.TXT')


@pytest.mark.parametrize('dates,expected', [
    (('20261117', '20261118', '20261116'), '20261117'),
    (('20261118', '20261116'), '20261118'),
    (('20261116',), '20261116'),
])
def test_public_report_search_prioritizes_original_then_next_then_previous(dates, expected):
    reports = {_report_entry(date): f'{date}\n' for date in dates}
    assert _read_from_public_fixtures('20261117', reports) == [f'{expected}\n']


@pytest.mark.parametrize('date,candidate', [
    ('20261130', '20261201'),
    ('20270101', '20261231'),
    ('20280228', '20280229'),
])
def test_public_report_search_uses_calendar_days_across_boundaries(date, candidate):
    assert _read_from_public_fixtures(date, {_report_entry(candidate): 'matched\n'}) == ['matched\n']


def test_adjacent_report_must_match_pid_model_and_candidate_date():
    reports = {
        _report_entry('20261118', pid='OTHER'): 'wrong pid\n',
        _report_entry('20261118', model='OTHER'): 'wrong model\n',
        _report_entry('20261118', filename_date='20261117'): 'wrong date\n',
        _report_entry('20261116'): 'correct\n',
    }
    assert _read_from_public_fixtures('20261117', reports) == ['correct\n']


def test_no_matching_report_in_existing_original_directory_checks_next_day():
    reports = {
        _report_entry('20261117', pid='OTHER'): 'wrong pid\n',
        _report_entry('20261118'): 'correct\n',
    }
    assert _read_from_public_fixtures('20261117', reports) == ['correct\n']


def test_error_on_next_day_preserves_unreadable_result():
    with patch('capi_carux_report.os.scandir', side_effect=[
        FileNotFoundError('missing'), PermissionError('denied'),
    ]):
        result = read_carux_report_lines(Path('/102/20261117/CAHM0701/OK'), 'PID', 'MODEL', '20261117')
    assert result is None


def test_missing_reports_on_all_three_dates_returns_not_found():
    assert _read_from_public_fixtures('20261117', {}) is CARUX_REPORT_NOT_FOUND


def test_invalid_image_date_preserves_original_error_log(caplog):
    config = {'mes_report': {'facility': 'CARUX'}}
    panel = Path('/x/M/invalid/PID')
    with caplog.at_level(logging.INFO, logger='capi.inference'):
        assert resolve_carux_report_dir(config, 'CAPI0701', panel) is None
    assert '圖片目錄日期無效' in caplog.text
    assert str(panel) in caplog.text
    assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_invalid_report_date_has_no_duplicate_date_warning(caplog):
    with caplog.at_level(logging.INFO, logger='capi.inference'):
        assert read_carux_report_lines(Path('/102/20260230/CAHM0701/OK'), 'PID', 'MODEL', '20260230') is None
    assert '圖片目錄日期無效' not in caplog.text


@pytest.mark.parametrize('candidate,offset', [('20261118', '+1'), ('20261116', '-1')])
def test_adjacent_report_logs_requested_and_actual_directory(caplog, candidate, offset):
    with caplog.at_level(logging.INFO, logger='capi.inference'):
        assert _read_from_public_fixtures('20261117', {_report_entry(candidate): 'matched\n'}) == ['matched\n']
    fallback = [record for record in caplog.records if '[REPORT_DATE_FALLBACK]' in record.message]
    assert len(fallback) == 1
    assert fallback[0].levelno == logging.WARNING
    assert str(Path('/102/20261117/CAHM0701/OK')) in fallback[0].message
    assert str(Path('/102') / candidate / 'CAHM0701' / 'OK') in fallback[0].message
    assert f'(date {offset} day)' in fallback[0].message
    assert str(Path('/102') / candidate / 'CAHM0701' / 'OK' / f'PID_{candidate}120000_MODEL_120000.TXT') in caplog.text
