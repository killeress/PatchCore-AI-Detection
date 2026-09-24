import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import capi_server
from capi_config import CAPIConfig


@pytest.mark.parametrize("original,next_day", [
    ("20260923", "20260924"),
    ("20260930", "20261001"),
    ("20261231", "20270101"),
    ("20280228", "20280229"),
    ("20280229", "20280301"),
])
def test_missing_panel_uses_next_calendar_day(tmp_path, original, next_day):
    model = tmp_path / "CAPI03" / "yuantu" / "GN160JCA5020S"
    requested = model / original / "YQ702J206D14"
    actual = model / next_day / requested.name
    actual.mkdir(parents=True)

    selected, resolved = capi_server.resolve_panel_image_directory(str(requested), {})

    assert selected == str(actual)
    assert resolved == actual


@pytest.mark.parametrize("separator", ["/", "\\"])
def test_unc_mapping_preserves_client_path_and_trailing_separator(tmp_path, separator):
    actual = tmp_path / "TIANMU" / "yuantu" / "MODEL" / "20260924" / "GLASS"
    actual.mkdir(parents=True)
    requested = "//aoi/d/TIANMU/yuantu/MODEL/20260923/GLASS/".replace("/", separator)

    selected, resolved = capi_server.resolve_panel_image_directory(
        requested, {"//aoi/d": str(tmp_path)},
    )

    assert selected == requested.replace("20260923", "20260924")
    assert resolved == actual


def test_existing_original_directory_always_wins_even_if_empty(tmp_path):
    original = tmp_path / "MODEL" / "20260923" / "GLASS"
    next_day = tmp_path / "MODEL" / "20260924" / "GLASS"
    original.mkdir(parents=True)
    next_day.mkdir(parents=True)
    (next_day / "W0F00000.tif").write_bytes(b"next day")

    assert capi_server.resolve_panel_image_directory(str(original), {}) == (
        str(original), original,
    )


@pytest.mark.parametrize("date", ["20260230", "20261301", "99991231", "260923", "not-a-date"])
def test_invalid_or_unsupported_date_is_unchanged(tmp_path, date):
    original = tmp_path / "MODEL" / date / "GLASS"
    assert capi_server.resolve_panel_image_directory(str(original), {}) == (
        str(original), original,
    )


def test_only_immediate_parent_date_is_changed(tmp_path):
    original = tmp_path / "20260923" / "MODEL" / "GLASS"
    (tmp_path / "20260924" / "MODEL" / "GLASS").mkdir(parents=True)
    assert capi_server.resolve_panel_image_directory(str(original), {}) == (
        str(original), original,
    )


def test_fallback_cannot_change_mapping_to_another_machine(tmp_path):
    original = "//aoi/d/MODEL/20260923/GLASS"
    other = tmp_path / "CAPI04" / "MODEL" / "20260924" / "GLASS"
    other.mkdir(parents=True)
    mapping = {
        "//aoi/d/MODEL/20260924": str(other.parent),
        "//aoi/d": str(tmp_path / "CAPI03"),
    }
    assert capi_server.resolve_panel_image_directory(original, mapping) == (
        original, tmp_path / "CAPI03" / "MODEL" / "20260923" / "GLASS",
    )


def _server(mapping):
    server = capi_server.CAPIServer.__new__(capi_server.CAPIServer)
    server.path_mapping = mapping
    server.cpu_workers = 1
    server._gpu_lock = threading.Lock()
    inferencer = SimpleNamespace(
        config=CAPIConfig(image_abnormal_detection_enabled=True),
        _parse_aoi_report_txt=MagicMock(return_value={}),
        process_panel=MagicMock(return_value=([], None, False, "", False, None, {})),
    )
    server._get_or_create_inferencer = lambda _model: inferencer
    server.db = MagicMock()
    server.db.save_inference_record.return_value = 123
    return server, inferencer


def _request(path):
    return capi_server.parse_request(
        f"AOI@GLASS;GN160JCA5020S;CAPI03;1920,1200;NG;{path}"
    )


@pytest.mark.parametrize("obstruction", ["missing", "original_file", "next_day_file"])
def test_unusable_paths_keep_existing_error_behavior(tmp_path, obstruction):
    original = tmp_path / "CAPI03" / "MODEL" / "20260923" / "GLASS"
    next_day = original.parent.parent / "20260924" / original.name
    if obstruction == "original_file":
        original.parent.mkdir(parents=True)
        original.write_text("not a directory")
        next_day.mkdir(parents=True)
    elif obstruction == "next_day_file":
        next_day.parent.mkdir(parents=True)
        next_day.write_text("not a directory")
    # Nearby dates, machines, models and panels must never be selected.
    for candidate in [
        original.parent.parent / "20260925" / "GLASS",
        original.parent.parent / "20260922" / "GLASS",
        original.parent.parent / "20260924" / "OTHER_GLASS",
        tmp_path / "CAPI03" / "OTHER_MODEL" / "20260924" / "GLASS",
        tmp_path / "CAPI04" / "MODEL" / "20260924" / "GLASS",
    ]:
        candidate.mkdir(parents=True)
    server, inferencer = _server({})
    parsed = _request(str(original))

    result = server._process_request(parsed)

    expected = "NOT_A_DIR" if obstruction == "original_file" else "DIR_NOT_FOUND"
    assert result[0] == f"ERR:{expected} ({original})"
    assert parsed["image_dir"] == str(original)
    inferencer.process_panel.assert_not_called()


def test_fallback_reaches_inference_database_and_background_reads(tmp_path, monkeypatch):
    requested = r"\\aoi\d\TIANMU\yuantu\GN160JCA5020S\20260923\GLASS"
    selected = requested.replace("20260923", "20260924")
    actual = tmp_path / "TIANMU" / "yuantu" / "GN160JCA5020S" / "20260924" / "GLASS"
    actual.mkdir(parents=True)
    server, inferencer = _server({r"\\aoi\d": str(tmp_path)})
    raw_request = f"AOI@GLASS;GN160JCA5020S;CAPI03;1920,1200;NG;{requested}"
    parsed = capi_server.parse_request(raw_request)
    precheck = MagicMock(return_value=None)
    monkeypatch.setattr(capi_server, "check_image_abnormal_precheck", precheck)
    warning = MagicMock()
    monkeypatch.setattr(capi_server.logger, "warning", warning)

    result = server._process_request(parsed)

    # The mocked inference returns no images; reaching it proves fallback dispatched.
    assert result[0] == "ERR:NO_IMAGES_FOUND"
    assert parsed["image_dir"] == selected
    assert inferencer._parse_aoi_report_txt.call_args.args[0] == actual
    assert precheck.call_args.args[0] == actual
    assert inferencer.process_panel.call_args.args[0] == actual
    assert any("IMAGE_DATE_FALLBACK" in call.args[0] for call in warning.call_args_list)

    find_pair = MagicMock(return_value=(None, None))
    monkeypatch.setattr("capi_side_white.find_side_white_pair", find_pair)
    server._save_results_async(
        ("test", 1), parsed, [], result[0], "[]",
        "2026-09-24 01:10:16", "2026-09-24 01:10:17", 1.0,
        client_request_text=raw_request, side_white_enabled=True,
    )

    saved = server.db.save_inference_record.call_args.kwargs
    assert saved["image_dir"] == selected
    assert saved["client_request_text"] == raw_request
    find_pair.assert_called_once_with(actual)
    server.db.save_side_white_result.assert_called_once()
