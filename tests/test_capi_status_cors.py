from capi_web import CAPIWebHandler


class _StatusTracker:
    def __init__(self, status=None, error=None):
        self.status = status
        self.error = error

    def get_status(self):
        if self.error:
            raise self.error
        return self.status


def _handler_with_capture(tracker):
    handler = object.__new__(CAPIWebHandler)
    handler.status_tracker = tracker
    handler.db = None
    captured = {}

    def capture(payload, status=200, headers=None):
        captured.update({
            "payload": payload,
            "status": status,
            "headers": dict(headers or {}),
        })

    handler._send_json = capture
    return handler, captured


def test_api_status_allows_static_dashboard_cross_origin_read():
    handler, captured = _handler_with_capture(
        _StatusTracker({
            "server": {"running": True},
            "traffic": {},
            "stats": {},
        })
    )

    handler._handle_api_status()

    assert captured["status"] == 200
    assert captured["headers"] == {"Access-Control-Allow-Origin": "*"}
    assert captured["payload"]["server"]["running"] is True


def test_api_status_error_keeps_cors_header():
    handler, captured = _handler_with_capture(
        _StatusTracker(error=RuntimeError("status unavailable"))
    )

    handler._handle_api_status()

    assert captured["status"] == 500
    assert captured["headers"] == {"Access-Control-Allow-Origin": "*"}
    assert captured["payload"] == {
        "error": "Cannot get server status: status unavailable"
    }
