from capi_web import CAPIWebHandler


def test_mes_comparison_rejects_duplicate_request():
    handler = object.__new__(CAPIWebHandler)
    response = {}
    handler._send_json = lambda data, status=200, **kwargs: response.update(
        data=data,
        status=status,
    )

    CAPIWebHandler._mes_comparison_lock.acquire()
    try:
        handler._handle_mes_comparison_api({})
    finally:
        CAPIWebHandler._mes_comparison_lock.release()

    assert response["status"] == 409
    assert response["data"] == {
        "success": False,
        "error": "MES Report 查詢進行中，請等待目前查詢完成。",
    }


def test_mes_comparison_releases_lock_after_early_response():
    handler = object.__new__(CAPIWebHandler)
    handler.db = None
    handler._send_json = lambda *args, **kwargs: None

    handler._handle_mes_comparison_api({})

    assert CAPIWebHandler._mes_comparison_lock.acquire(blocking=False)
    CAPIWebHandler._mes_comparison_lock.release()


def test_mes_report_template_disables_query_controls_while_loading():
    html = open("templates/ric_report.html", encoding="utf-8").read()

    assert "let _loading = false;" in html
    assert "if (_loading) return;" in html
    assert "_setQueryControlsDisabled(true);" in html
    assert "_setQueryControlsDisabled(false);" in html
