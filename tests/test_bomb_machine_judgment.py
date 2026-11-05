from types import SimpleNamespace

import pytest

from capi_server import (
    _normalize_machine_judgment_for_bomb_only_panel,
    _original_machine_judgment_for_record,
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


@pytest.mark.parametrize("non_bomb", [
    pytest.param(_tile(is_dust=True), id="dust"),
    pytest.param(_tile(is_scratch_filtered=True), id="scratch"),
    pytest.param(_tile(is_below_threshold=True), id="below-threshold"),
    pytest.param(_tile(is_excluded=True), id="excluded"),
])
@pytest.mark.parametrize("other_lighting", [False, True])
def test_non_bomb_point_keeps_machine_ng_even_when_ai_filters_it(non_bomb, other_lighting):
    bomb = _tile(is_bomb=True)
    results = ([_result(tiles=[bomb]), _result(tiles=[non_bomb])] if other_lighting
               else [_result(tiles=[bomb, non_bomb])])
    report = {"W0F00000": [SimpleNamespace(defect_code="CM00")],
              "B0F00000": [SimpleNamespace(defect_code="CDB6")]}

    assert _stored_machine_judgment_for_record("NG", results, report) == "NG"


@pytest.mark.parametrize("edge", [_edge(is_dust=True), _edge(is_cv_ok=True)])
def test_non_bomb_edge_keeps_machine_ng_even_when_ai_filters_it(edge):
    result = _result(tiles=[_tile(is_bomb=True)], edges=[edge])

    assert _normalize_machine_judgment_for_bomb_only_panel("NG", [result]) == "NG"


def test_non_bomb_aoi_tracking_tile_cannot_disappear_from_machine_judgment():
    result = _result(tiles=[_tile(is_bomb=True)])
    tracking = _tile(is_below_threshold=True)
    tracking.is_aoi_coord_tile = True
    result.tiles = [tracking]

    assert _normalize_machine_judgment_for_bomb_only_panel("NG", [result]) == "NG"


@pytest.mark.parametrize("is_bomb,expected", [(False, "NG"), (True, "OK")])
def test_white_frame_aoi_ok_does_not_replace_confirmation_of_bomb(is_bomb, expected):
    result = SimpleNamespace(white_frame_result={
        "status": "OK", "bomb_excluded_count": 1,
        "aoi_tiles": [{"status": "OK", "is_bomb": is_bomb}],
    })

    assert _normalize_machine_judgment_for_bomb_only_panel("NG", [result]) == expected


def test_logged_six_w0f_bombs_and_b8f_dust_keep_machine_ng_but_ai_ok():
    from capi_config import CAPIConfig
    from capi_inference import AOIReportDefect
    from capi_server import aggregate_judgment, build_qjpg_response
    from test_qjpg_report_response import _image_result, _tile as real_tile

    white = _image_result("YQ71CU218C23W0F00000081649.tif", mark_text="KA")
    black = _image_result("YQ71CU218C23B8F00000081654.tif", mark_text="")
    points = [(9, 210), (1904, 311), (10, 606), (1904, 607), (597, 1184), (602, 1184)]
    for index, (x, y) in enumerate(points):
        tile = real_tile(index, x, y)
        tile.is_bomb = tile.is_aoi_coord_tile = True
        tile.aoi_product_x, tile.aoi_product_y = x, y
        white.anomaly_tiles.append((tile, 0.9, None))
    dust = real_tile(0, 1789, 619)
    dust.is_aoi_coord_tile = dust.is_suspected_dust_or_scratch = True
    black.anomaly_tiles = [(dust, 1.0, None)]
    report = {
        "W0F00000": [AOIReportDefect("CM00", x, y, "W0F00000") for x, y in points],
        "B0F00000": [AOIReportDefect("CDB6", 1789, 619, "B0F00000")],
    }

    results = [white, black]
    ai_judgment, _ = aggregate_judgment(results)
    assert ai_judgment == "OK"
    assert _stored_machine_judgment_for_record("NG", results, report) == "NG"
    response = build_qjpg_response({"glass_id": "YQ71CU218C23"}, ai_judgment, results, CAPIConfig())
    assert response.startswith("@QJPG-YQ71CU218C23;OK;KA;NG")
    assert response.count("PCDK3") == 6


@pytest.mark.parametrize("record,expected", [
    ({"machine_judgment": "OK", "client_request_text":
      "AOI@G1;M1;AAPI2612;1920,1200;NG;/images"}, "NG"),
    ({"machine_judgment": "OK", "client_response_text":
      "AOI@G1;M1;AAPI2612;NG;OK\r\n@QJPG-G1;OK;KA;NG,"}, "NG"),
    ({"machine_judgment": "OK", "client_request_text":
      "AOI@G1;M1;AAPI2612;1920,1200;OK;/images",
      "client_response_text": "AOI@G1;M1;AAPI2612;NG;OK"}, "OK"),
    ({"machine_judgment": "NG", "client_request_text": "malformed"}, "NG"),
    ({"machine_judgment": "OK", "client_response_text": "@QJPG-G1;NG;00;OK,"}, "OK"),
])
def test_original_machine_judgment_uses_preserved_client_protocol(record, expected):
    assert _original_machine_judgment_for_record(record) == expected
