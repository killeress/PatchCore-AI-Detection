from types import SimpleNamespace

import pytest

from capi_server import (
    _normalize_machine_judgment_for_bomb_only_panel,
    _stored_machine_judgment_for_record,
)


def _tile(
    *,
    is_bomb=False,
    is_dust=False,
    is_excluded=False,
    is_scratch_filtered=False,
    is_below_threshold=False,
):
    return SimpleNamespace(
        is_bomb=is_bomb,
        is_suspected_dust_or_scratch=is_dust,
        is_in_exclude_zone=is_excluded,
        scratch_filtered=is_scratch_filtered,
        is_aoi_coord_below_threshold=is_below_threshold,
    )


def _edge(*, is_bomb=False, is_dust=False, is_cv_ok=False):
    return SimpleNamespace(
        is_bomb=is_bomb,
        is_suspected_dust_or_scratch=is_dust,
        is_cv_ok=is_cv_ok,
    )


def _result(*, tiles=(), edges=()):
    return SimpleNamespace(
        anomaly_tiles=[(tile, 0.9, None) for tile in tiles],
        edge_defects=list(edges),
    )


def test_bomb_only_panel_stores_machine_judgment_as_ok():
    result = _result(tiles=[_tile(is_bomb=True)])

    assert _normalize_machine_judgment_for_bomb_only_panel("NG", [result]) == "OK"


def test_edge_bomb_only_panel_stores_machine_judgment_as_ok():
    result = _result(edges=[_edge(is_bomb=True)])

    assert _normalize_machine_judgment_for_bomb_only_panel("NG", [result]) == "OK"


def test_bomb_panel_keeps_machine_ng_when_real_tile_exists():
    result = _result(tiles=[_tile(is_bomb=True), _tile()])

    assert _normalize_machine_judgment_for_bomb_only_panel("NG", [result]) == "NG"


def test_bomb_panel_keeps_machine_ng_when_excluded_zone_tile_exists():
    result = _result(tiles=[_tile(is_bomb=True), _tile(is_excluded=True)])

    assert _normalize_machine_judgment_for_bomb_only_panel("NG", [result]) == "NG"


def test_panel_without_bomb_keeps_machine_ng():
    result = _result(tiles=[_tile(is_dust=True)])

    assert _normalize_machine_judgment_for_bomb_only_panel("NG", [result]) == "NG"


def test_machine_ok_is_not_changed():
    result = _result(tiles=[_tile(is_bomb=True)])

    assert _normalize_machine_judgment_for_bomb_only_panel("OK", [result]) == "OK"


def test_record_machine_judgment_preserves_client_ok_with_aoi_report():
    result = _result(tiles=[_tile(is_bomb=True)])
    aoi_report = {"WGF50500": [SimpleNamespace(defect_code="C1111")]}

    assert _stored_machine_judgment_for_record("OK", [result], aoi_report) == "OK"


def test_record_machine_judgment_normalizes_all_five_reported_bombs_to_ok():
    result = _result(tiles=[_tile(is_bomb=True) for _ in range(5)])
    aoi_report = {"WINDOWS_BG": [SimpleNamespace(defect_code="CDK2") for _ in range(5)]}

    assert _stored_machine_judgment_for_record("NG", [result], aoi_report) == "OK"


def test_record_machine_judgment_keeps_ng_for_bomb_and_real_tile():
    result = _result(tiles=[_tile(is_bomb=True), _tile()])
    aoi_report = {"WINDOWS_BG": [SimpleNamespace(defect_code="CDK2") for _ in range(2)]}

    assert _stored_machine_judgment_for_record("NG", [result], aoi_report) == "NG"


@pytest.mark.parametrize("other_result", [
    pytest.param(_result(tiles=[_tile()]), id="other-lighting-real-tile"),
    pytest.param(_result(edges=[_edge()]), id="real-edge"),
    pytest.param(SimpleNamespace(white_frame_result={"status": "NG"}), id="white-frame-ng"),
])
def test_record_machine_judgment_keeps_ng_for_real_defect_elsewhere(other_result):
    bomb_result = _result(tiles=[_tile(is_bomb=True)])
    aoi_report = {"WINDOWS_BG": [SimpleNamespace(defect_code="CDK2")]}

    assert _stored_machine_judgment_for_record("NG", [bomb_result, other_result], aoi_report) == "NG"


def test_record_machine_judgment_keeps_ng_for_passthrough_defect_with_bombs():
    result = _result(tiles=[_tile(is_bomb=True)])
    aoi_report = {"WINDOWS_BG": [
        SimpleNamespace(defect_code="CDK2"), SimpleNamespace(defect_code="CSJBL"),
    ]}

    assert _stored_machine_judgment_for_record("NG", [result], aoi_report) == "NG"


@pytest.mark.parametrize("results", [[], [_result()], [_result(tiles=[_tile(is_dust=True)])]])
def test_record_machine_judgment_does_not_convert_without_bomb_evidence(results):
    aoi_report = {"WINDOWS_BG": [SimpleNamespace(defect_code="CDK2")]}

    assert _stored_machine_judgment_for_record("NG", results, aoi_report) == "NG"


def test_record_machine_judgment_preserves_hy_with_bombs_and_report():
    result = _result(tiles=[_tile(is_bomb=True)])
    aoi_report = {"WINDOWS_BG": [SimpleNamespace(defect_code="CDK2")]}

    assert _stored_machine_judgment_for_record("HY", [result], aoi_report) == "HY"
