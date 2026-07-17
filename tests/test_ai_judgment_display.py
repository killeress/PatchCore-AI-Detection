from capi_web import ai_simple


def test_client_hy_error_is_displayed_as_hy_without_changing_other_errors():
    assert ai_simple("ERR:HY") == "HY"
    assert ai_simple("ERR:HY:W0F00000") == "HY"
    assert ai_simple("ERR:CONNECTION_TIMEOUT") == "ERR"
