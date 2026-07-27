import capi_mes_report
from capi_mes_report import WP_DEFTHIS_FAC_ID_BY_FACILITY
from tools.diagnose_mes_oracle import _sampled_batch_indexes


def test_sampled_batch_indexes_cover_first_middle_and_last():
    assert _sampled_batch_indexes(77, 3) == [0, 38, 76]


def test_sampled_batch_indexes_can_run_one_or_all_batches():
    assert _sampled_batch_indexes(77, 1) == [38]
    assert _sampled_batch_indexes(3, 0) == [0, 1, 2]


def test_fac_id_mapping_matches_mes_tables():
    assert WP_DEFTHIS_FAC_ID_BY_FACILITY == {
        "MOD1": "C",
        "MOD2": "E",
    }


def test_mes_report_logger_uses_production_capi_namespace():
    assert capi_mes_report.logger.name == "capi.mes_report"
